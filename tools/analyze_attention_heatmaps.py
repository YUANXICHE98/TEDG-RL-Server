#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


ACTION_LABELS_23 = [
    "MORE",  # 0
    "N", "E", "S", "W", "NE", "SE", "SW", "NW",  # 1-8
    "RUN_N", "RUN_E", "RUN_S", "RUN_W", "RUN_NE", "RUN_SE", "RUN_SW", "RUN_NW",  # 9-16
    "UP", "DOWN",  # 17-18
    "WAIT", "KICK", "EAT", "SEARCH",  # 19-22
]


def _nice_low_priority() -> None:
    try:
        os.nice(10)
    except Exception:
        pass
    # Avoid network access during offline analysis runs by default.
    # Set to "0" if you explicitly want to use the embedding API.
    os.environ.setdefault("TEDG_OFFLINE_EMBEDDINGS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _ensure_repo_root_on_path() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _get_device(device: str) -> torch.device:
    d = device.strip().lower()
    if d in {"cpu", "musa", "cuda"}:
        if d == "cpu":
            return torch.device("cpu")
        if d == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        if d == "musa":
            try:
                import torch_musa  # noqa: F401

                if torch.musa.is_available():
                    return torch.device("musa:0")
            except Exception:
                pass
    return torch.device("cpu")


def _depth_bucket(depth: int) -> str:
    if depth <= 0:
        return "dlvl_?"
    if depth <= 3:
        return "dlvl_1-3"
    if depth <= 6:
        return "dlvl_4-6"
    if depth <= 10:
        return "dlvl_7-10"
    return "dlvl_11+"


def _scenario_key_simple(pre_nodes: List[str], scene_atoms: List[str], depth: int) -> str:
    low_hp = "low_hp" in pre_nodes
    has_gold = "has_gold" in pre_nodes
    return f"{_depth_bucket(depth)}|low_hp={int(low_hp)}|gold={int(has_gold)}"


def _scenario_key_atoms(pre_nodes: List[str], scene_atoms: List[str], depth: int, cap: int = 3) -> str:
    pre = ",".join(sorted(pre_nodes)[:cap]) if pre_nodes else "-"
    scn = ",".join(sorted(scene_atoms)[:cap]) if scene_atoms else "-"
    return f"{_depth_bucket(depth)}|pre:{pre}|scene:{scn}"


@dataclass
class ScenarioAgg:
    steps: int = 0
    action_counts: np.ndarray | None = None
    action_prob_sums: np.ndarray | None = None
    attn_sums: np.ndarray | None = None  # (A,) where A is attention dims

    def __post_init__(self) -> None:
        if self.action_counts is None:
            self.action_counts = np.zeros(23, dtype=np.int64)
        if self.action_prob_sums is None:
            self.action_prob_sums = np.zeros(23, dtype=np.float64)
        if self.attn_sums is None:
            self.attn_sums = np.zeros(4, dtype=np.float64)


def _masked_probs(logits: torch.Tensor, mask: np.ndarray | None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (logits_for_dist, probs) for a single state (action_dim,).
    """
    if mask is None:
        logits_for_dist = logits
    else:
        m = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
        logits_for_dist = logits.masked_fill(~m, float("-inf"))
        if not torch.isfinite(logits_for_dist).any():
            logits_for_dist = logits

    probs = F.softmax(logits_for_dist, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    s = probs.sum()
    if float(s) > 0:
        probs = probs / s
    return logits_for_dist, probs


def _infer_net_cfg(exp_name: str) -> Dict[str, Any]:
    n = exp_name.lower()
    use_gumbel = "gumbel" in n
    sparse_topk = None
    if "sparse" in n or "moe" in n:
        # repo's default ablation uses top-2 when sparse is enabled
        sparse_topk = 2
    return {"use_gumbel": use_gumbel, "sparse_topk": sparse_topk}

def _infer_model_mode(exp: str, training_log: str | None) -> str:
    """
    Returns: "multichannel" | "hram_doc" | "hram_e2e"
    """
    e = exp.lower()
    if "hram_doc" in e or "doc" in e and "hram" in e:
        return "hram_doc"
    if "hram" in e:
        return "hram_e2e"
    # also allow training.log hints
    if training_log:
        if "H-RAM 文档方案" in training_log or "HRAMPolicyNetDoc" in training_log:
            return "hram_doc"
        if "V3 H-RAM" in training_log or "HRAMPolicyNet" in training_log:
            return "hram_e2e"
    return "multichannel"


def main() -> None:
    _nice_low_priority()
    _ensure_repo_root_on_path()

    parser = argparse.ArgumentParser(description="Analyze policy attention + action selection by scenario (heatmaps).")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name under ablation_v2/results/")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (defaults to best_model.pth)")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|musa (default cpu)")
    parser.add_argument(
        "--model",
        choices=["auto", "multichannel", "hram_doc", "hram_e2e"],
        default="auto",
        help="Which policy/attention to analyze: auto|multichannel|hram_doc|hram_e2e",
    )
    parser.add_argument(
        "--mask",
        choices=["auto", "on", "off"],
        default="auto",
        help="Action masking: auto|on|off (auto infers from exp name)",
    )
    parser.add_argument(
        "--matcher",
        choices=["auto", "embedding", "coverage"],
        default="auto",
        help="Hypergraph matcher: auto|embedding|coverage (auto tries to infer from training.log)",
    )
    parser.add_argument(
        "--hard-routing",
        action="store_true",
        help="If supported (gumbel router), use hard routing during eval (else soft).",
    )
    parser.add_argument("--scenario-mode", choices=["simple", "atoms"], default="simple")
    parser.add_argument("--top-scenarios", type=int, default=20, help="Max scenarios to include in plots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=None, help="Output dir (default docs/visualizations/attention/<exp>/)")
    args = parser.parse_args()

    if len(ACTION_LABELS_23) != 23:
        raise SystemExit("ACTION_LABELS_23 must have length 23")

    exp_dir = ROOT / "ablation_v2" / "results" / args.exp
    if not exp_dir.exists():
        raise SystemExit(f"Missing exp dir: {exp_dir}")

    ckpt = Path(args.checkpoint) if args.checkpoint else (exp_dir / "checkpoints" / "best_model.pth")
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    outdir = Path(args.outdir) if args.outdir else (ROOT / "docs" / "visualizations" / "attention" / args.exp)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        ckpt_disp = str(ckpt.relative_to(ROOT))
    except Exception:
        ckpt_disp = str(ckpt)

    # Import heavy deps only after arg parsing so `--help` stays fast.
    import gymnasium as gym
    import nle.env  # noqa: F401
    import nle.nethack as nh

    from src.core.action_masking import ActionMasker
    from src.core.hypergraph_loader import EmbeddingMatcher
    from src.core.hypergraph_matcher import HypergraphMatcher
    from src.core.networks_correct import MultiChannelPolicyNet
    from src.core.networks_hram import HRAMPolicyNet, HRAMPolicyNetDoc
    from src.core.state_constructor import StateConstructor

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _get_device(args.device)

    # Infer model/matcher/mask configs
    exp_lower = args.exp.lower()
    log_path = exp_dir / "training.log"
    training_log = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else None

    model_mode = args.model
    if model_mode == "auto":
        model_mode = _infer_model_mode(args.exp, training_log)

    mask_on: bool
    if args.mask == "on":
        mask_on = True
    elif args.mask == "off":
        mask_on = False
    else:
        mask_on = "no_mask" not in exp_lower

    matcher_mode = args.matcher
    if matcher_mode == "auto":
        if training_log:
            # example: "网络配置: Gumbel=False, Sparse Top-K=None, Embedding=True"
            if "Embedding=True" in training_log or "Embedding: True" in training_log:
                matcher_mode = "embedding"
            elif "Embedding=False" in training_log or "Embedding: False" in training_log:
                matcher_mode = "coverage"
        if matcher_mode == "auto":
            matcher_mode = "embedding" if "embedding" in exp_lower else "coverage"

    # HRAM* training scripts don't use the ActionMasker in practice; disable by default to avoid mismatch.
    if model_mode in {"hram_doc", "hram_e2e"} and args.mask == "auto":
        mask_on = False

    # Helpers for multichannel (V1/V2) only
    state_constructor = None
    action_masker = None
    embedding_matcher = None
    matcher = None
    if model_mode == "multichannel":
        state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
        action_masker = ActionMasker(state_constructor.hypergraph, num_actions=23) if mask_on else None
        if matcher_mode == "embedding":
            embedding_matcher = EmbeddingMatcher(min_support=5)
            matcher = None
        else:
            embedding_matcher = None
            matcher = HypergraphMatcher(state_constructor.hypergraph, weights=(0.35, 0.35, 0.2, 0.1), tau=200.0)

    # Build policy net
    inferred = _infer_net_cfg(args.exp)
    if model_mode == "multichannel":
        policy_net = MultiChannelPolicyNet(
            state_dim=115,
            action_dim=23,
            actor_hidden_dim=128,
            attention_hidden_dim=64,
            use_gumbel=inferred["use_gumbel"],
            gumbel_tau=1.0,
            sparse_topk=inferred["sparse_topk"],
        ).to(device)
        attn_labels = ["α_pre", "α_scene", "α_effect", "α_rule"]
        attn_kind = "channel_alpha"
    elif model_mode == "hram_doc":
        policy_net = HRAMPolicyNetDoc(state_dim=115, action_dim=23, gumbel_tau=1.0).to(device)
        attn_labels = ["α_pre", "α_scene", "α_effect", "α_rule"]
        attn_kind = "channel_alpha"
    else:
        policy_net = HRAMPolicyNet(state_dim=115, action_dim=23).to(device)
        # top_k is a field on HRAMPolicyNet
        k = int(getattr(policy_net, "top_k", 10))
        attn_labels = [f"k{i}" for i in range(k)]
        attn_kind = "knowledge_attn"

    payload = torch.load(str(ckpt), map_location=device)
    state_dict = payload.get("policy_net", payload)
    # Backward-compat: keys_norm is a derived buffer (may exist in some checkpoints)
    if isinstance(state_dict, dict):
        state_dict = {k: v for (k, v) in state_dict.items() if not str(k).endswith("keys_norm")}
    policy_net.load_state_dict(state_dict, strict=True)
    policy_net.eval()

    # Env
    try:
        env = gym.make("NetHackScore-v0")
    except Exception:
        env = gym.make("NetHack-v0")

    def extract_state_from_obs_multichannel(obs: Dict[str, Any], t_now: int) -> Tuple[np.ndarray, List[str], List[str], float, int, int]:
        """
        Minimal extractor consistent with ablation_v2/train/train_v2.py, but supports both
        matcher APIs (embedding / coverage).
        Returns: (state115, pre_nodes, scene_atoms, confidence, depth, gold)
        """
        assert state_constructor is not None
        assert (matcher is not None) or (embedding_matcher is not None)

        blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
        hp = int(blstats[nh.NLE_BL_HP])
        hp_max = int(blstats[nh.NLE_BL_HPMAX])
        depth = int(blstats[nh.NLE_BL_DEPTH])
        gold = int(blstats[nh.NLE_BL_GOLD])
        ac = int(blstats[nh.NLE_BL_AC])
        hunger = int(blstats[nh.NLE_BL_HUNGER])

        belief = np.zeros(50, dtype=np.float32)
        belief[0] = 1.0
        belief[1] = 1.0 if hp > 0 else 0.0
        belief[2] = 1.0
        belief[3] = float(hp) / float(max(hp_max, 1))
        belief[4] = 1.0 if hunger == 0 else 0.0
        belief[5] = 1.0 if gold == 0 else 0.0
        belief[6] = 1.0
        belief[7] = 1.0
        belief[8] = 1.0
        belief[9] = 1.0 if ac > 0 else 0.0

        pre_nodes: List[str] = []
        scene_atoms: List[str] = []
        effect_atoms: List[str] = []
        rule_atoms: List[str] = []

        if hp > 0:
            pre_nodes.append("player_alive")
        if hp_max > 0 and float(hp) / float(hp_max) < 0.5:
            pre_nodes.append("low_hp")
        if depth > 0:
            scene_atoms.append(f"dlvl_{depth}")
        if gold > 0:
            pre_nodes.append("has_gold")

        if embedding_matcher is not None:
            confidence, _matched_edges = embedding_matcher.match(pre_nodes, scene_atoms, effect_atoms, rule_atoms, top_k=8)
        else:
            assert matcher is not None
            topk = matcher.match({"pre": pre_nodes, "scene": scene_atoms, "effect": effect_atoms, "rule": rule_atoms}, t_now=float(t_now), t_i=None, top_k=8)
            confidence = float(topk[0].score) if topk else 0.0

        goal = np.zeros(16, dtype=np.float32)
        goal[0] = 1.0

        state = state_constructor.construct_state(
            belief_vector=belief,
            pre_nodes=pre_nodes,
            scene_atoms=scene_atoms,
            eff_metadata={},
            conditional_effects=[],
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            goal_embedding=goal,
        )
        return state, pre_nodes, scene_atoms, float(np.clip(confidence, 0.0, 1.0)), depth, gold

    def extract_state_from_obs_hram(obs: Dict[str, Any]) -> Tuple[np.ndarray, int, bool, bool]:
        """
        HRAM training uses a simpler raw 115-dim state based on blstats.
        Returns: (state115, depth, low_hp, has_gold)
        """
        blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
        hp = int(blstats[nh.NLE_BL_HP])
        hp_max = int(blstats[nh.NLE_BL_HPMAX])
        depth = int(blstats[nh.NLE_BL_DEPTH])
        gold = int(blstats[nh.NLE_BL_GOLD])
        low_hp = (hp_max > 0) and (float(hp) / float(hp_max) < 0.5)
        has_gold = gold > 0

        if model_mode == "hram_doc":
            from ablation_v2.train.train_hram_doc import extract_state_from_nethack_obs as _extract  # type: ignore

            state = _extract(obs)
        else:
            from ablation_v2.train.train_hram import extract_state_from_nethack_obs as _extract  # type: ignore

            state = _extract(obs, None, verbose=False)
        return state, depth, low_hp, has_gold

    def _new_bucket() -> ScenarioAgg:
        return ScenarioAgg(attn_sums=np.zeros(len(attn_labels), dtype=np.float64))

    agg: Dict[str, ScenarioAgg] = defaultdict(_new_bucket)

    def scenario_key(pre_nodes: List[str], scene_atoms: List[str], depth: int) -> str:
        if args.scenario_mode == "atoms":
            return _scenario_key_atoms(pre_nodes, scene_atoms, depth)
        return _scenario_key_simple(pre_nodes, scene_atoms, depth)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            steps += 1
            if model_mode == "multichannel":
                state, pre_nodes, scene_atoms, confidence, depth, gold = extract_state_from_obs_multichannel(obs, steps)
                skey = scenario_key(pre_nodes, scene_atoms, depth)
            else:
                state, depth, low_hp, has_gold = extract_state_from_obs_hram(obs)
                skey = f"{_depth_bucket(int(depth))}|low_hp={int(bool(low_hp))}|gold={int(bool(has_gold))}"

            st = torch.as_tensor(state, device=device, dtype=torch.float32)
            with torch.no_grad():
                if model_mode == "multichannel":
                    logits, attn, _value = policy_net(st)  # attn = alpha (4,)
                elif model_mode == "hram_doc":
                    logits, attn, _value = policy_net(st, use_gumbel=bool(args.hard_routing))  # attn = alpha (4,)
                else:
                    # HRAMPolicyNet expects batch input
                    logits, attn_w, _value = policy_net(st.unsqueeze(0))  # attn_w: (batch,1,K)
                    if attn_w.dim() == 3:
                        attn = attn_w.squeeze(0).squeeze(0)
                    elif attn_w.dim() == 2:
                        attn = attn_w.squeeze(0)
                    else:
                        attn = attn_w

            # Normalize shapes to 1D for downstream aggregation
            if logits.dim() == 2 and logits.size(0) == 1:
                logits = logits.squeeze(0)
            if attn.dim() == 2 and attn.size(0) == 1:
                attn = attn.squeeze(0)

            mask = None
            if action_masker is not None:
                # masking only supported in multichannel mode (needs atoms)
                mask = action_masker.get_action_mask(pre_nodes, scene_atoms, confidence)

            logits_for_dist, probs = _masked_probs(logits, mask)
            dist = torch.distributions.Categorical(logits=logits_for_dist)
            action = int(dist.sample().item())

            a_np = attn.detach().float().cpu().numpy()
            p_np = probs.detach().float().cpu().numpy()

            bucket = agg[skey]
            bucket.steps += 1
            bucket.action_counts[action] += 1
            bucket.action_prob_sums += p_np
            bucket.attn_sums += a_np

            obs, _rew, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

    env.close()

    # Select top scenarios by step count
    items = sorted(agg.items(), key=lambda kv: kv[1].steps, reverse=True)
    items = items[: max(1, int(args.top_scenarios))]
    scenarios = [k for (k, _v) in items]
    n = len(scenarios)

    attn_mat = np.zeros((n, len(attn_labels)), dtype=np.float64)
    act_freq = np.zeros((n, 23), dtype=np.float64)
    act_prob = np.zeros((n, 23), dtype=np.float64)
    counts = np.zeros((n,), dtype=np.int64)

    for i, k in enumerate(scenarios):
        b = agg[k]
        counts[i] = b.steps
        if b.steps <= 0:
            continue
        attn_mat[i] = b.attn_sums / float(b.steps)
        act_freq[i] = b.action_counts.astype(np.float64) / float(b.steps)
        act_prob[i] = b.action_prob_sums / float(b.steps)

    # Write machine-readable summary
    summary = {
        "exp": args.exp,
        "model": model_mode,
        "attn_kind": attn_kind,
        "checkpoint": ckpt_disp,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "scenario_mode": args.scenario_mode,
        "masking": "on" if action_masker is not None else "off",
        "matcher": matcher_mode,
        "scenarios": [
            {
                "scenario": scenarios[i],
                "steps": int(counts[i]),
                "attn_mean": attn_mat[i].tolist(),
                "top_actions_by_freq": sorted(
                    [(ACTION_LABELS_23[j], float(act_freq[i, j])) for j in range(23)],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
            }
            for i in range(n)
        ],
        "attn_labels": attn_labels,
        "action_labels": ACTION_LABELS_23,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 220, "font.size": 10})

    def heatmap(mat: np.ndarray, ylabels: List[str], xlabels: List[str], title: str, out: Path, vmin=None, vmax=None):
        fig_h = max(4.5, 0.35 * len(ylabels) + 1.0)
        fig_w = max(8.0, 0.28 * len(xlabels) + 6.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.set_ylabel("value", rotation=270, labelpad=12)
        plt.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    heatmap(
        attn_mat,
        scenarios,
        attn_labels,
        f"{args.exp} - {attn_kind} (mean) by scenario",
        outdir / "alpha_heatmap.png",
        vmin=0.0,
        vmax=1.0,
    )
    heatmap(
        act_freq,
        scenarios,
        ACTION_LABELS_23,
        f"{args.exp} - Action frequency (sampled) by scenario",
        outdir / "action_freq_heatmap.png",
        vmin=0.0,
        vmax=float(np.max(act_freq)) if float(np.max(act_freq)) > 0 else 1.0,
    )
    heatmap(
        act_prob,
        scenarios,
        ACTION_LABELS_23,
        f"{args.exp} - Mean action probability by scenario",
        outdir / "action_prob_heatmap.png",
        vmin=0.0,
        vmax=float(np.max(act_prob)) if float(np.max(act_prob)) > 0 else 1.0,
    )

    # Transposed (action -> scenario) for quick axis-switch
    heatmap(
        act_freq.T,
        ACTION_LABELS_23,
        scenarios,
        f"{args.exp} - Action frequency (sampled), transposed",
        outdir / "action_freq_heatmap_T.png",
        vmin=0.0,
        vmax=float(np.max(act_freq)) if float(np.max(act_freq)) > 0 else 1.0,
    )
    heatmap(
        act_prob.T,
        ACTION_LABELS_23,
        scenarios,
        f"{args.exp} - Mean action probability, transposed",
        outdir / "action_prob_heatmap_T.png",
        vmin=0.0,
        vmax=float(np.max(act_prob)) if float(np.max(act_prob)) > 0 else 1.0,
    )

    # Short markdown
    lines = [
        f"# Attention/Action heatmaps: `{args.exp}`",
        "",
        f"- checkpoint: `{ckpt_disp}`",
        f"- model: `{model_mode}`",
        f"- attn_kind: `{attn_kind}`",
        f"- episodes: `{args.episodes}`, max_steps: `{args.max_steps}`",
        f"- scenario_mode: `{args.scenario_mode}`",
        f"- masking: `{'on' if action_masker is not None else 'off'}`",
        f"- matcher: `{matcher_mode}`",
        "",
        "## Outputs",
        f"- `{(outdir / 'alpha_heatmap.png').relative_to(ROOT)}`",
        f"- `{(outdir / 'action_freq_heatmap.png').relative_to(ROOT)}`",
        f"- `{(outdir / 'action_prob_heatmap.png').relative_to(ROOT)}`",
        f"- `{(outdir / 'action_freq_heatmap_T.png').relative_to(ROOT)}` (axis switched)",
        f"- `{(outdir / 'action_prob_heatmap_T.png').relative_to(ROOT)}` (axis switched)",
        f"- `{(outdir / 'summary.json').relative_to(ROOT)}`",
        "",
        "## Notes",
        "- `alpha_heatmap` shows which channel the policy trusts more in each scenario.",
        "- `action_freq_heatmap` uses sampled actions; use `action_prob_heatmap` for the underlying distribution.",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✓ Wrote to: {outdir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
