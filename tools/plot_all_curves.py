#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "visualizations" / "current" / "all_curves.png"
LEGEND_CSV = ROOT / "docs" / "visualizations" / "current" / "all_curves_legend.csv"
OUT_V1 = ROOT / "docs" / "visualizations" / "current" / "v1_curves.png"
OUT_V2 = ROOT / "docs" / "visualizations" / "current" / "v2_curves.png"
LEGEND_V1 = ROOT / "docs" / "visualizations" / "current" / "v1_curves_legend.csv"
LEGEND_V2 = ROOT / "docs" / "visualizations" / "current" / "v2_curves_legend.csv"


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _moving_average(y: list[float], w: int) -> list[float]:
    if w <= 1 or len(y) < w:
        return y
    s = sum(y[:w])
    out = [s / w]
    for i in range(w, len(y)):
        s += y[i] - y[i - w]
        out.append(s / w)
    return out


@dataclass
class Curve:
    name: str
    source: str
    x: list[int]
    score: list[float]
    reward: list[float]
    kind: str  # "per-episode" | "eval-block"


def _pretty(name: str) -> str:
    name = name.strip()
    for prefix in ("results_", "exp_", "experiment_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("__", "_")


def _load_json_curve(name: str, log_json: Path) -> Curve | None:
    try:
        d = json.loads(_read_text(log_json) or "{}")
        scores = d.get("episode_scores") or []
        rewards = d.get("episode_rewards") or []
        if not scores and not rewards:
            return None
        n = max(len(scores), len(rewards))
        scores = [float(x) for x in scores] + [0.0] * (n - len(scores))
        rewards = [float(x) for x in rewards] + [0.0] * (n - len(rewards))
        x = list(range(1, n + 1))
        return Curve(name=name, source=str(log_json.relative_to(ROOT)), x=x, score=scores, reward=rewards, kind="per-episode")
    except Exception:
        return None


def _parse_train_v2_per_episode(name: str, log_txt: Path) -> Curve | None:
    """
    Parse per-episode metrics from train_v2.py logs.
    Uses:
      [DEBUG] Episode N/...
      [DEBUG] 奖励: R, 步数: ..., 分数: S
    """
    txt = _read_text(log_txt)
    if not txt:
        return None

    re_ep = re.compile(r"^\[DEBUG\]\s+Episode\s+(\d+)/")
    re_metrics = re.compile(r"^\[DEBUG\]\s+奖励:\s*([\d\.-]+),\s*步数:\s*(\d+),\s*分数:\s*([\d\.-]+)")

    x: list[int] = []
    rewards: list[float] = []
    scores: list[float] = []

    lines = txt.splitlines()
    i = 0
    while i < len(lines):
        m = re_ep.match(lines[i].strip())
        if not m:
            i += 1
            continue
        ep = int(m.group(1))
        # lookahead for metrics
        r = s = None
        for j in range(i + 1, min(i + 6, len(lines))):
            mm = re_metrics.match(lines[j].strip())
            if mm:
                r = float(mm.group(1))
                s = float(mm.group(3))
                break
        if r is not None and s is not None:
            x.append(ep)
            rewards.append(r)
            scores.append(s)
        i += 1

    if len(x) >= 10:
        return Curve(
            name=name,
            source=str(log_txt.relative_to(ROOT)),
            x=x,
            score=scores,
            reward=rewards,
            kind="per-episode",
        )
    return None


def _parse_eval_blocks(name: str, log_txt: Path) -> Curve | None:
    """
    Parse evaluation blocks (usually every 50 episodes).
    Supports both:
      Episode N/50000 (patience: ...)
        平均奖励: ...
        平均分数: ...
    and:
      Episode N/50000
        平均奖励: ...
        平均分数: ...
    """
    txt = _read_text(log_txt)
    if not txt:
        return None
    re_hdr_a = re.compile(r"^Episode\s+(\d+)/(\d+)\s+\(patience:")
    re_hdr_b = re.compile(r"^Episode\s+(\d+)/(\d+)\s*$")
    re_avg_r = re.compile(r"^\s*平均奖励:\s*([\d\.-]+),")
    re_avg_s = re.compile(r"^\s*平均分数:\s*([\d\.-]+),")

    x: list[int] = []
    rewards: list[float] = []
    scores: list[float] = []

    lines = txt.splitlines()
    for i, line in enumerate(lines):
        m = re_hdr_a.match(line) or re_hdr_b.match(line)
        if not m:
            continue
        ep = int(m.group(1))
        avg_r = avg_s = None
        for j in range(i + 1, min(i + 10, len(lines))):
            mr = re_avg_r.match(lines[j])
            if mr:
                avg_r = float(mr.group(1))
            ms = re_avg_s.match(lines[j])
            if ms:
                avg_s = float(ms.group(1))
        if avg_r is not None or avg_s is not None:
            x.append(ep)
            rewards.append(avg_r if avg_r is not None else 0.0)
            scores.append(avg_s if avg_s is not None else 0.0)

    if len(x) >= 2:
        return Curve(
            name=name,
            source=str(log_txt.relative_to(ROOT)),
            x=x,
            score=scores,
            reward=rewards,
            kind="eval-block",
        )
    return None


def _iter_v1_curves() -> Iterable[Curve]:
    v1 = ROOT / "ablation_v1" / "results"
    if not v1.exists():
        return []
    curves: list[Curve] = []
    for exp_dir in sorted(v1.iterdir()):
        log_json = exp_dir / "logs" / "training_log.json"
        if not log_json.exists():
            continue
        c = _load_json_curve(f"V1/{_pretty(exp_dir.name)}", log_json)
        if c:
            curves.append(c)
    return curves


def _iter_v2_curves() -> list[Curve]:
    v2 = ROOT / "ablation_v2" / "results"
    wanted = [
        "baseline",
        "no_mask",
        "sparse_moe",
        "gumbel_sparse",
        "gumbel",
        "gumbel_fixed",
        "hram_e2e",
        "hram_doc",
        "hram_doc_fixed",
    ]
    curves: list[Curve] = []
    for name in wanted:
        log = v2 / name / "training.log"
        if not log.exists():
            continue
        c = _parse_train_v2_per_episode(f"V2/{_pretty(name)}", log) or _parse_eval_blocks(f"V2/{_pretty(name)}", log)
        if c:
            curves.append(c)
    return curves


def _write_legend_csv(curves: list[Curve], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "kind", "points", "source"])
        for c in sorted(curves, key=lambda x: x.name):
            w.writerow([c.name, c.kind, len(c.x), c.source])


def _plot(curves: list[Curve], out: Path, title_suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)

    if not curves:
        raise SystemExit("No curves found")

    # Sort for stable legend
    curves = sorted(curves, key=lambda c: c.name)

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
        }
    )

    fig, (axS, axR) = plt.subplots(1, 2, figsize=(16, 5))

    for c in curves:
        if c.kind == "per-episode":
            # smooth more aggressively for crowded plot
            w = 200 if c.name.startswith("V2/") else 100
            s_sm = _moving_average(c.score, w)
            r_sm = _moving_average(c.reward, w)
            x_sm = c.x[len(c.x) - len(s_sm) :]
            axS.plot(x_sm, s_sm, linewidth=1.2, label=f"{c.name} MA{w}")
            axR.plot(x_sm, r_sm, linewidth=1.2, label=f"{c.name} MA{w}")
        else:
            axS.plot(c.x, c.score, linewidth=1.6, linestyle="--", label=f"{c.name} eval")
            axR.plot(c.x, c.reward, linewidth=1.6, linestyle="--", label=f"{c.name} eval")

    axS.set_title(f"Score Curves ({title_suffix})")
    axS.set_xlabel("Episode")
    axS.set_ylabel("Score")

    axR.set_title(f"Reward Curves ({title_suffix})")
    axR.set_xlabel("Episode")
    axR.set_ylabel("Reward")

    # One combined legend (right side). Use bbox_inches="tight" on save so the
    # outside legend isn't cropped away.
    handles, labels = axS.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    curves = list(_iter_v1_curves()) + list(_iter_v2_curves())
    if not curves:
        raise SystemExit("No curves found")

    _write_legend_csv(curves, LEGEND_CSV)

    v1_curves = [c for c in curves if c.name.startswith("V1/")]
    v2_curves = [c for c in curves if c.name.startswith("V2/")]
    _write_legend_csv(v1_curves, LEGEND_V1)
    _write_legend_csv(v2_curves, LEGEND_V2)

    _plot(curves, OUT, "all experiments")
    _plot(v1_curves, OUT_V1, "V1")
    _plot(v2_curves, OUT_V2, "V2")

    print(f"✓ Wrote: {OUT.relative_to(ROOT)}")
    print(f"✓ Wrote: {OUT_V1.relative_to(ROOT)}")
    print(f"✓ Wrote: {OUT_V2.relative_to(ROOT)}")
    print(f"✓ Wrote: {LEGEND_CSV.relative_to(ROOT)}")
    print(f"✓ Wrote: {LEGEND_V1.relative_to(ROOT)}")
    print(f"✓ Wrote: {LEGEND_V2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
