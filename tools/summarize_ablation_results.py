#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V1_RESULTS_DIR = ROOT / "ablation_v1" / "results"
V2_RESULTS_DIR = ROOT / "ablation_v2" / "results"
OUT_MD = ROOT / "docs" / "ablation_results_comparison.md"
OUT_CSV = ROOT / "docs" / "ablation_results_comparison.csv"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:  # NaN
            return "NaN"
        return f"{x:.2f}"
    return str(x)


def _find_latest_checkpoint(exp_dir: Path) -> str | None:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    model_ckpts = sorted(ckpt_dir.glob("model_*.pth"))
    if model_ckpts:
        # pick largest suffix
        def _key(p: Path) -> int:
            m = re.search(r"model_(\d+)\.pth$", p.name)
            return int(m.group(1)) if m else -1

        best = max(model_ckpts, key=_key)
        return str(best.relative_to(ROOT))
    best = ckpt_dir / "best_model.pth"
    if best.exists():
        return str(best.relative_to(ROOT))
    final = ckpt_dir / "model_final.pth"
    if final.exists():
        return str(final.relative_to(ROOT))
    return None


def _parse_steps_per_episode(text: str) -> int | None:
    m = re.search(r"每Episode最大步数[:：]\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"训练配置:\s*\d+\s*episodes,\s*(\d+)\s*steps/episode", text)
    if m:
        return int(m.group(1))
    return None


def _ps_for_pids(pids: list[int]) -> str:
    if not pids:
        return ""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "pid,etime,cmd", "-p", *[str(p) for p in pids]],
            text=True,
        )
        return out.strip()
    except Exception:
        return ""

def _ps_for_ablation_v2_train() -> str:
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return ""
    lines = []
    for line in out.splitlines():
        if "ablation_v2/train/" in line and "python" in line:
            lines.append(line)
    return "\n".join(lines).strip()


@dataclass
class V1Row:
    exp: str
    matcher_mode: str | None
    episodes: int
    steps_per_ep: int | None
    best_reward: float | None
    best_score: float | None
    avg_reward: float | None
    avg_score: float | None
    last500_reward: float | None
    last500_score: float | None
    last500_len: float | None
    alpha_last500_mean: tuple[float, float, float, float] | None
    alpha_last500_std: tuple[float, float, float, float] | None
    log_json: str
    log_txt: str | None


def _mean_last(arr: list[float], n: int) -> float | None:
    if not arr:
        return None
    tail = arr[-min(n, len(arr)) :]
    return float(sum(tail) / len(tail))


def parse_v1() -> list[V1Row]:
    rows: list[V1Row] = []
    if not V1_RESULTS_DIR.exists():
        return rows

    for exp_dir in sorted(V1_RESULTS_DIR.iterdir()):
        log_json = exp_dir / "logs" / "training_log.json"
        if not log_json.exists():
            continue

        d = json.loads(_read_text(log_json) or "{}")
        rewards = [float(x) for x in d.get("episode_rewards", [])]
        scores = [float(x) for x in d.get("episode_scores", [])]
        lengths = [float(x) for x in d.get("episode_lengths", [])]
        alpha = d.get("alpha_history", [])

        training_log_txt = exp_dir / "training.log"
        txt = _read_text(training_log_txt) if training_log_txt.exists() else ""
        steps_per_ep = _parse_steps_per_episode(txt)

        matcher_mode = None
        m = re.search(r"初始化匹配器\s*\(mode=([a-zA-Z_]+)\)", txt)
        if m:
            matcher_mode = m.group(1)

        alpha_mean = alpha_std = None
        if isinstance(alpha, list) and alpha and isinstance(alpha[0], list) and len(alpha[0]) >= 4:
            tail = alpha[-min(500, len(alpha)) :]
            cols = list(zip(*tail))
            alpha_mean = tuple(float(sum(c) / len(c)) for c in cols[:4])  # type: ignore[arg-type]
            alpha_std = tuple(
                float((sum((x - mu) ** 2 for x in c) / max(len(c), 1)) ** 0.5)
                for c, mu in zip(cols[:4], alpha_mean)
            )

        rows.append(
            V1Row(
                exp=exp_dir.name,
                matcher_mode=matcher_mode,
                episodes=len(rewards),
                steps_per_ep=steps_per_ep,
                best_reward=float(max(rewards)) if rewards else None,
                best_score=float(max(scores)) if scores else None,
                avg_reward=float(sum(rewards) / len(rewards)) if rewards else None,
                avg_score=float(sum(scores) / len(scores)) if scores else None,
                last500_reward=_mean_last(rewards, 500),
                last500_score=_mean_last(scores, 500),
                last500_len=_mean_last(lengths, 500),
                alpha_last500_mean=alpha_mean,
                alpha_last500_std=alpha_std,
                log_json=str(log_json.relative_to(ROOT)),
                log_txt=str(training_log_txt.relative_to(ROOT)) if training_log_txt.exists() else None,
            )
        )

    return rows


@dataclass
class V2Row:
    exp: str
    steps_per_ep: int | None
    episodes_target: int | None
    episodes_seen: int | None
    last_eval_ep: int | None
    last_avg_reward: float | None
    best_reward: float | None
    last_avg_score: float | None
    best_score: float | None
    last_avg_alpha: tuple[float, float, float, float] | None
    last_route_pct: tuple[float, float, float, float] | None
    uses_gumbel: bool | None
    sparse_topk: int | None
    uses_embedding: bool | None
    traceback: bool
    nan_warn: bool
    latest_ckpt: str | None
    log_txt: str


def parse_v2() -> list[V2Row]:
    rows: list[V2Row] = []
    if not V2_RESULTS_DIR.exists():
        return rows

    re_eval_a = re.compile(r"^Episode\s+(\d+)/(\d+)\s+\(patience:")
    re_eval_b = re.compile(r"^Episode\s+(\d+)/(\d+)\s*$")
    re_avg_reward = re.compile(r"^\s*平均奖励:\s*([\d\.-]+),\s*最佳奖励:\s*([\d\.-]+)")
    re_avg_score = re.compile(r"^\s*平均分数:\s*([\d\.-]+),\s*最佳分数:\s*([\d\.-]+)")
    re_alpha = re.compile(
        r"^\s*平均α权重:\s*pre=([\d\.-]+),\s*scene=([\d\.-]+),\s*effect=([\d\.-]+),\s*rule=([\d\.-]+)"
    )
    re_route = re.compile(
        r"^\s*路由分布:\s*Pre=([\d\.-]+)%,\s*Scene=([\d\.-]+)%,\s*Effect=([\d\.-]+)%,\s*Rule=([\d\.-]+)%"
    )
    re_any_ep = re.compile(r"Episode\s+(\d+)/(\d+)")

    for exp_dir in sorted([p for p in V2_RESULTS_DIR.iterdir() if p.is_dir()]):
        training_log = exp_dir / "training.log"
        if not training_log.exists():
            continue

        txt = _read_text(training_log)
        steps_per_ep = _parse_steps_per_episode(txt)
        m = re.search(r"训练配置:\s*(\d+)\s*episodes,\s*(\d+)\s*steps/episode", txt)
        episodes_target = int(m.group(1)) if m else None

        uses_gumbel = None
        sparse_topk = None
        uses_embedding = None
        m = re.search(r"网络配置:\s*Gumbel=(True|False),\s*Sparse Top-K=([A-Za-z0-9_-]+),\s*Embedding=(True|False)", txt)
        if m:
            uses_gumbel = m.group(1) == "True"
            raw_topk = m.group(2)
            sparse_topk = int(raw_topk) if raw_topk.isdigit() else None
            uses_embedding = m.group(3) == "True"

        traceback = "Traceback" in txt
        nan_warn = ("NaN/Inf" in txt) or ("nan/inf" in txt.lower())

        episodes_seen = None
        last_eval_ep = None
        last_avg_reward = None
        best_reward = None
        last_avg_score = None
        best_score = None
        last_avg_alpha = None
        last_route_pct = None

        lines = txt.splitlines()
        for i, line in enumerate(lines):
            ep_match = re_any_ep.search(line)
            if ep_match:
                ep = int(ep_match.group(1))
                episodes_seen = max(episodes_seen or 0, ep)

            s = line.rstrip("\n")
            m = re_eval_a.match(s) or re_eval_b.match(s)
            if not m:
                continue

            ep = int(m.group(1))
            last_eval_ep = ep
            # lookahead
            for j in range(i + 1, min(i + 12, len(lines))):
                l = lines[j]
                mr = re_avg_reward.match(l)
                if mr:
                    last_avg_reward = float(mr.group(1))
                    cand_best_r = float(mr.group(2))
                    best_reward = max(best_reward or float("-inf"), cand_best_r)
                ms = re_avg_score.match(l)
                if ms:
                    last_avg_score = float(ms.group(1))
                    cand_best_s = float(ms.group(2))
                    best_score = max(best_score or float("-inf"), cand_best_s)
                ma = re_alpha.match(l)
                if ma:
                    last_avg_alpha = tuple(map(float, ma.groups()))  # type: ignore[assignment]
                mt = re_route.match(l)
                if mt:
                    last_route_pct = tuple(map(float, mt.groups()))  # type: ignore[assignment]

        latest_ckpt = _find_latest_checkpoint(exp_dir)

        rows.append(
            V2Row(
                exp=exp_dir.name,
                steps_per_ep=steps_per_ep,
                episodes_target=episodes_target,
                episodes_seen=episodes_seen,
                last_eval_ep=last_eval_ep,
                last_avg_reward=last_avg_reward,
                best_reward=best_reward,
                last_avg_score=last_avg_score,
                best_score=best_score if best_score != float("-inf") else None,
                last_avg_alpha=last_avg_alpha,
                last_route_pct=last_route_pct,
                uses_gumbel=uses_gumbel,
                sparse_topk=sparse_topk,
                uses_embedding=uses_embedding,
                traceback=traceback,
                nan_warn=nan_warn,
                latest_ckpt=latest_ckpt,
                log_txt=str(training_log.relative_to(ROOT)),
            )
        )

    return rows


def write_csv(v1: list[V1Row], v2: list[V2Row]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "version",
                "exp",
                "matcher_or_arch",
                "steps_per_ep",
                "episodes",
                "best_reward",
                "best_score",
                "avg_reward_or_last_avg_reward",
                "avg_score_or_last_avg_score",
                "stability",
                "latest_ckpt",
                "log_path",
            ]
        )
        for r in v1:
            w.writerow(
                [
                    "v1",
                    r.exp,
                    r.matcher_mode or "-",
                    r.steps_per_ep or "-",
                    r.episodes,
                    _fmt(r.best_reward),
                    _fmt(r.best_score),
                    _fmt(r.avg_reward),
                    _fmt(r.avg_score),
                    "ok",
                    "-",
                    r.log_json,
                ]
            )
        for r in v2:
            stability = []
            if r.traceback:
                stability.append("traceback")
            if r.nan_warn:
                stability.append("nan_warn")
            w.writerow(
                [
                    "v2",
                    r.exp,
                    f"gumbel={r.uses_gumbel},topk={r.sparse_topk},emb={r.uses_embedding}",
                    r.steps_per_ep or "-",
                    r.episodes_seen or "-",
                    _fmt(r.best_reward),
                    _fmt(r.best_score),
                    _fmt(r.last_avg_reward),
                    _fmt(r.last_avg_score),
                    "|".join(stability) if stability else "ok",
                    r.latest_ckpt or "-",
                    r.log_txt,
                ]
            )


def write_md(v1: list[V1Row], v2: list[V2Row]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    pids_path = V2_RESULTS_DIR / "experiment_pids.txt"
    pids: list[int] = []
    if pids_path.exists():
        try:
            raw = pids_path.read_text(encoding="utf-8", errors="ignore").strip()
            pids = [int(x) for x in raw.split() if x.isdigit()]
        except Exception:
            pids = []
    ps_out = _ps_for_pids(pids)
    ps_train_out = _ps_for_ablation_v2_train()

    v2_main_order = ["baseline", "no_mask", "gumbel", "sparse_moe", "gumbel_sparse", "hram_doc", "hram_e2e"]
    v2_by_name = {r.exp: r for r in v2}
    v2_main = [v2_by_name[n] for n in v2_main_order if n in v2_by_name]
    v2_others = [r for r in v2 if r.exp not in set(v2_main_order)]

    def _alpha_str(a: tuple[float, float, float, float] | None) -> str:
        if not a:
            return "-"
        return f"pre={a[0]:.3f}, scene={a[1]:.3f}, eff={a[2]:.3f}, rule={a[3]:.3f}"

    def _route_str(a: tuple[float, float, float, float] | None) -> str:
        if not a:
            return "-"
        return f"Pre={a[0]:.1f}%, Scene={a[1]:.1f}%, Eff={a[2]:.1f}%, Rule={a[3]:.1f}%"

    # stop/continue heuristics
    running = {int(line.split()[0]) for line in ps_out.splitlines()[1:]} if ps_out else set()
    no_mask_running = False
    if running and "no_mask" in v2_by_name and "no_mask" in (ps_out or ""):
        no_mask_running = True

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"# TEDG-RL V1 vs V2 实验对比（自动汇总）")
    lines.append("")
    lines.append(f"- 生成时间: `{now}`")
    lines.append(f"- 汇总脚本: `tools/summarize_ablation_results.py`")
    lines.append("")
    if ps_out:
        lines.append("## 当前仍在跑的实验（来自 `experiment_pids.txt`）")
        lines.append("")
        lines.append("```")
        lines.append(ps_out)
        lines.append("```")
        lines.append("")
    else:
        lines.append("## 当前仍在跑的实验（来自 `experiment_pids.txt`）")
        lines.append("")
        lines.append("- 未检测到 `experiment_pids.txt` 对应的存活进程（或 `ps` 不可用）。")
        lines.append("")
    
    lines.append("## 当前仍在跑的训练进程（实时 `ps` 扫描）")
    lines.append("")
    if ps_train_out:
        lines.append("```")
        lines.append(ps_train_out)
        lines.append("```")
    else:
        lines.append("- 未检测到运行中的 `ablation_v2/train/*.py` 进程。")
    lines.append("")

    lines.append("## 可比性说明（非常重要）")
    lines.append("")
    lines.append("- V1 默认 `500 steps/episode`（另有 `results_extended_steps=2000 steps/episode`）。")
    lines.append("- V2 默认 `2000 steps/episode`，回报/分数分布与 V1 不在同一量纲；建议只比较：稳定性、是否学到“持续得分/生存”、以及同设置下的对照组。")
    lines.append("")

    lines.append("## V1 结果总表（来自 `training_log.json`）")
    lines.append("")
    lines.append("| Exp | Matcher | Steps/ep | Episodes | BestR | BestS | AvgR | AvgS | Last500R | Last500S | α(last500 mean) | 日志 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in sorted(v1, key=lambda x: x.exp):
        lines.append(
            "| "
            + " | ".join(
                [
                    r.exp,
                    (r.matcher_mode or "-"),
                    str(r.steps_per_ep or "-"),
                    str(r.episodes),
                    _fmt(r.best_reward),
                    _fmt(r.best_score),
                    _fmt(r.avg_reward),
                    _fmt(r.avg_score),
                    _fmt(r.last500_reward),
                    _fmt(r.last500_score),
                    _alpha_str(r.alpha_last500_mean),
                    f"`{r.log_json}`",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## V2 结果总表（来自 `training.log` 的周期性评估块）")
    lines.append("")
    lines.append("| Exp | Steps/ep | SeenEp | LastEvalEp | LastAvgR | BestR | LastAvgS | BestS | α(last) | Route(last) | 稳定性 | 最新ckpt | 日志 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|")
    for r in v2_main:
        stability = []
        if r.traceback:
            stability.append("traceback")
        if r.nan_warn:
            stability.append("nan_warn")
        if not stability:
            stability.append("ok")
        lines.append(
            "| "
            + " | ".join(
                [
                    r.exp,
                    str(r.steps_per_ep or "-"),
                    str(r.episodes_seen or "-"),
                    str(r.last_eval_ep or "-"),
                    _fmt(r.last_avg_reward),
                    _fmt(r.best_reward),
                    _fmt(r.last_avg_score),
                    _fmt(r.best_score),
                    _alpha_str(r.last_avg_alpha),
                    _route_str(r.last_route_pct),
                    ",".join(stability),
                    f"`{r.latest_ckpt}`" if r.latest_ckpt else "-",
                    f"`{r.log_txt}`",
                ]
            )
            + " |"
        )
    lines.append("")
    if v2_others:
        lines.append("### V2 其他目录（debug/smoke 等）")
        lines.append("")
        lines.append("- 这些不纳入主对比，但保留在CSV里。")
        lines.append("")

    # rankings (V2 only)
    v2_rankable = [r for r in v2_main if r.best_reward is not None or r.best_score is not None]
    lines.append("## V2 组内排名（截至当前日志）")
    lines.append("")
    by_best_r = sorted(v2_rankable, key=lambda r: (r.best_reward or float("-inf")), reverse=True)
    by_best_s = sorted(v2_rankable, key=lambda r: (r.best_score or float("-inf")), reverse=True)
    lines.append("### 按 `BestR` 排名")
    lines.append("")
    lines.append("| Rank | Exp | BestR | BestS | 稳定性 |")
    lines.append("|---:|---|---:|---:|---|")
    for i, r in enumerate(by_best_r, 1):
        stability = "traceback" if r.traceback else ("nan_warn" if r.nan_warn else "ok")
        lines.append(f"| {i} | {r.exp} | {_fmt(r.best_reward)} | {_fmt(r.best_score)} | {stability} |")
    lines.append("")
    lines.append("### 按 `BestS` 排名")
    lines.append("")
    lines.append("| Rank | Exp | BestS | BestR | 稳定性 |")
    lines.append("|---:|---|---:|---:|---|")
    for i, r in enumerate(by_best_s, 1):
        stability = "traceback" if r.traceback else ("nan_warn" if r.nan_warn else "ok")
        lines.append(f"| {i} | {r.exp} | {_fmt(r.best_score)} | {_fmt(r.best_reward)} | {stability} |")
    lines.append("")

    lines.append("## 现在“停谁/让谁继续跑”的建议（基于当前日志+稳定性）")
    lines.append("")
    lines.append("- **立刻暂停/不要继续**: `gumbel`（已发生 `logits=NaN` 导致 Traceback），`hram_doc`（出现 NaN/Inf 警告，虽然未必立刻崩，但训练质量不可信）。")
    lines.append("- **可以继续跑到 `min_episodes=10000` 再下结论**: `baseline`、`sparse_moe`、`hram_e2e`（目前无崩溃迹象，best_score 也不差）。")
    lines.append("- **作为对照组，建议“跑够就停”**: `no_mask`（用于验证mask必要性；如果跑到 10k 仍然 `LastAvgS≈0`，就可以停掉，把算力让给mask版本/HRAM）。")
    if no_mask_running:
        lines.append(f"- **当前正在跑**: `no_mask`。建议让它至少跑到下一次 `model_05000.pth`（或直接跑到 10k），然后停止并启动 `baseline`/`hram_e2e` 的续跑。")
    lines.append("")

    lines.append("## 推荐的续跑命令（尽量复用已有checkpoint）")
    lines.append("")
    lines.append("- `baseline`（稳定基线，建议先跑到 10k）:")
    if "baseline" in v2_by_name and v2_by_name["baseline"].latest_ckpt:
        lines.append(
            f"  - `python -u ablation_v2/train/train_v2.py --exp-name baseline --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --resume {v2_by_name['baseline'].latest_ckpt}`"
        )
    else:
        lines.append("  - `python -u ablation_v2/train/train_v2.py --exp-name baseline --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding`")

    lines.append("- `sparse_moe`（如果要做稀疏专家对比，建议续跑到 10k）:")
    if "sparse_moe" in v2_by_name and v2_by_name["sparse_moe"].latest_ckpt:
        lines.append(
            f"  - `python -u ablation_v2/train/train_v2.py --exp-name sparse_moe --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --use-gumbel --sparse-topk 2 --resume {v2_by_name['sparse_moe'].latest_ckpt}`"
        )
    else:
        lines.append("  - `python -u ablation_v2/train/train_v2.py --exp-name sparse_moe --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --use-gumbel --sparse-topk 2`")

    lines.append("- `hram_e2e`（端到端检索，建议单独跑、学习率更稳）:")
    if "hram_e2e" in v2_by_name and v2_by_name["hram_e2e"].latest_ckpt:
        lines.append(
            f"  - `python -u ablation_v2/train/train_hram.py --exp-name hram_e2e --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --resume {v2_by_name['hram_e2e'].latest_ckpt}`"
        )
    else:
        lines.append("  - `python -u ablation_v2/train/train_hram.py --exp-name hram_e2e --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000`")

    lines.append("")
    lines.append("## 我建议你下一步先做的 2 件事")
    lines.append("")
    lines.append("- 把 **对照组**跑全：让 `baseline`（mask+soft融合）和 `no_mask`（无mask）都跑到 10k，然后用 `best_score` + `last_avg_score` 判断 mask 的必要性。")
    lines.append("- 再做 **创新组稳定化**：修好 `gumbel/hram_doc` 的 NaN 后再继续跑，不然 best_reward 峰值没有论文价值（不可复现/不稳定）。")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    v1 = parse_v1()
    v2 = parse_v2()
    write_csv(v1, v2)
    write_md(v1, v2)
    print(f"✓ Wrote: {OUT_MD.relative_to(ROOT)}")
    print(f"✓ Wrote: {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
