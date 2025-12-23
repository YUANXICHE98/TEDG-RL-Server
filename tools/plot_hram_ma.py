#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _moving_average(vals: list[float], w: int) -> list[float]:
    if w <= 1 or len(vals) < w:
        return vals[:]
    s = sum(vals[:w])
    out = [s / w]
    for i in range(w, len(vals)):
        s += vals[i] - vals[i - w]
        out.append(s / w)
    return out


@dataclass
class EvalPoint:
    ep: int
    avg_reward: float | None
    best_reward: float | None
    avg_score: float | None
    best_score: float | None
    route_pre: float | None
    route_scene: float | None
    route_effect: float | None
    route_rule: float | None


_RE_HDR = re.compile(r"^Episode\s+(\d+)/(\d+)\s*(?:\(patience:.*\))?\s*$")
_RE_AVG_R = re.compile(r"^\s*平均奖励:\s*([-\d.]+)\s*,\s*最佳奖励:\s*([-\d.]+)\s*$")
_RE_AVG_S = re.compile(r"^\s*平均分数:\s*([-\d.]+)\s*,\s*最佳分数:\s*([-\d.]+)\s*$")
_RE_ROUTE = re.compile(
    r"^\s*路由分布:\s*Pre=([-\d.]+)%\s*,\s*Scene=([-\d.]+)%\s*,\s*Effect=([-\d.]+)%\s*,\s*Rule=([-\d.]+)%\s*$"
)


def parse_eval_blocks(log_path: Path) -> list[EvalPoint]:
    txt = _read_text(log_path)
    if not txt:
        return []

    lines = txt.splitlines()
    points: list[EvalPoint] = []
    for i, line in enumerate(lines):
        m = _RE_HDR.match(line.strip())
        if not m:
            continue
        ep = int(m.group(1))
        avg_r = best_r = avg_s = best_s = None
        rp = rs = reff = rr = None
        for j in range(i + 1, min(i + 12, len(lines))):
            mr = _RE_AVG_R.match(lines[j].strip())
            if mr:
                avg_r = float(mr.group(1))
                best_r = float(mr.group(2))
                continue
            ms = _RE_AVG_S.match(lines[j].strip())
            if ms:
                avg_s = float(ms.group(1))
                best_s = float(ms.group(2))
                continue
            mt = _RE_ROUTE.match(lines[j].strip())
            if mt:
                rp = float(mt.group(1))
                rs = float(mt.group(2))
                reff = float(mt.group(3))
                rr = float(mt.group(4))
                continue

        if avg_r is None and avg_s is None and best_r is None and best_s is None:
            continue
        points.append(
            EvalPoint(
                ep=ep,
                avg_reward=avg_r,
                best_reward=best_r,
                avg_score=avg_s,
                best_score=best_s,
                route_pre=rp,
                route_scene=rs,
                route_effect=reff,
                route_rule=rr,
            )
        )
    return points


def _maybe_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    return plt


def _last_improvement_episode(x: list[int], best: list[float]) -> int | None:
    if not x or not best:
        return None
    last = best[0]
    last_ep = x[0]
    for ep, v in zip(x, best):
        if v > last:
            last = v
            last_ep = ep
    return last_ep


def plot_experiment(
    title: str,
    series: list[tuple[str, list[EvalPoint]]],
    out_path: Path,
    ma_blocks: int = 5,
    nz_blocks: int = 10,
) -> None:
    plt = _maybe_import_matplotlib()

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 2)

    ax_score = fig.add_subplot(gs[0, 0])
    ax_reward = fig.add_subplot(gs[0, 1])
    ax_nz = fig.add_subplot(gs[1, 0])
    ax_extra = fig.add_subplot(gs[1, 1])

    ax_score.set_title("MA Score (from log blocks)")
    ax_reward.set_title("MA Reward (from log blocks)")
    ax_nz.set_title(f"Non-zero MA Score Rate (rolling {nz_blocks} blocks)")
    ax_extra.set_title("Best (left) + Route% (right, if available)")

    colors = ["#2980b9", "#e67e22", "#27ae60", "#8e44ad", "#2c3e50"]
    has_route_any = False
    ax_route = ax_extra.twinx()
    ax_route.set_ylabel("route %")
    ax_route.set_ylim(-1.0, 101.0)

    for idx, (label, pts) in enumerate(series):
        if not pts:
            continue
        c = colors[idx % len(colors)]

        x = [p.ep for p in pts]
        avg_s = [float(p.avg_score or 0.0) for p in pts]
        avg_r = [float(p.avg_reward or 0.0) for p in pts]
        best_s = [float(p.best_score or 0.0) for p in pts]
        best_r = [float(p.best_reward or 0.0) for p in pts]

        ax_score.plot(x, avg_s, color=c, alpha=0.25, linewidth=1.0, label=f"{label} MA(block)")
        ax_reward.plot(x, avg_r, color=c, alpha=0.25, linewidth=1.0, label=f"{label} MA(block)")

        if len(avg_s) >= ma_blocks:
            ax_score.plot(x[ma_blocks - 1 :], _moving_average(avg_s, ma_blocks), color=c, linewidth=2.0, label=f"{label} MA({ma_blocks} blocks)")
        if len(avg_r) >= ma_blocks:
            ax_reward.plot(x[ma_blocks - 1 :], _moving_average(avg_r, ma_blocks), color=c, linewidth=2.0, label=f"{label} MA({ma_blocks} blocks)")

        # non-zero MA score rate (per block)
        nz = [1.0 if v > 0 else 0.0 for v in avg_s]
        if len(nz) >= nz_blocks:
            ax_nz.plot(
                x[nz_blocks - 1 :],
                _moving_average(nz, nz_blocks),
                color=c,
                linewidth=2.0,
                label=f"{label}",
            )
        else:
            ax_nz.plot(x, nz, color=c, linewidth=1.0, alpha=0.5, label=f"{label}")

        # best curves (left axis)
        ax_extra.step(x, best_s, where="post", color=c, linewidth=2.0, label=f"{label} bestS")
        ax_extra.step(x, best_r, where="post", color=c, linewidth=1.5, linestyle="--", label=f"{label} bestR")

        # route curves if available
        if any(p.route_pre is not None for p in pts):
            has_route_any = True
            rp = [p.route_pre if p.route_pre is not None else float("nan") for p in pts]
            rs = [p.route_scene if p.route_scene is not None else float("nan") for p in pts]
            reff = [p.route_effect if p.route_effect is not None else float("nan") for p in pts]
            rr = [p.route_rule if p.route_rule is not None else float("nan") for p in pts]
            ax_route.plot(x, rp, color="#c0392b", linewidth=1.4, alpha=0.85, label=f"{label} Pre%")
            ax_route.plot(x, rs, color="#16a085", linewidth=1.4, alpha=0.85, label=f"{label} Scene%")
            ax_route.plot(x, reff, color="#f39c12", linewidth=1.4, alpha=0.85, label=f"{label} Effect%")
            ax_route.plot(x, rr, color="#7f8c8d", linewidth=1.4, alpha=0.85, label=f"{label} Rule%")

    ax_score.set_xlabel("episode")
    ax_score.set_ylabel("avg score")
    ax_reward.set_xlabel("episode")
    ax_reward.set_ylabel("avg reward")
    ax_nz.set_xlabel("episode")
    ax_nz.set_ylabel("rate")
    ax_nz.set_ylim(-0.05, 1.05)
    ax_extra.set_xlabel("episode")
    ax_extra.set_ylabel("best value")

    ax_score.legend(loc="upper left", fontsize=9)
    ax_reward.legend(loc="upper left", fontsize=9)
    ax_nz.legend(loc="upper left", fontsize=9)
    ax_extra.legend(loc="upper left", fontsize=8, ncol=1)
    if has_route_any:
        ax_route.legend(loc="upper right", fontsize=8, ncol=1)

    if not has_route_any:
        ax_extra.text(
            0.02,
            0.05,
            "No route distribution blocks found in these logs.\n(For HRAMDoc you should see a 'route distribution: Pre=.. Scene=.. Effect=.. Rule=..' line.)",
            transform=ax_extra.transAxes,
            fontsize=10,
            color="#2c3e50",
            alpha=0.9,
        )

    fig.suptitle(title, fontsize=14, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def _pick_existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot MA/diagnostics for HRAM runs from training.log blocks.")
    ap.add_argument("--exp", required=True, help="experiment name (for title only)")
    ap.add_argument("--out", required=True, help="output png path (relative to repo root ok)")
    ap.add_argument("--log", action="append", default=[], help="training log path(s) (can pass multiple)")
    ap.add_argument("--label", action="append", default=[], help="label(s) for each --log")
    ap.add_argument("--ma-blocks", type=int, default=5, help="rolling window over blocks (each block ~50 eps)")
    ap.add_argument("--nz-blocks", type=int, default=10, help="rolling window for non-zero rate over blocks")
    args = ap.parse_args()

    logs = [Path(p) if Path(p).is_absolute() else (ROOT / p) for p in args.log]
    logs = _pick_existing(logs)
    if not logs:
        raise SystemExit("No log files found. Pass --log ablation_v2/results/<exp>/training.log")

    labels = args.label[:]
    if len(labels) < len(logs):
        for i in range(len(labels), len(logs)):
            labels.append(logs[i].name)

    series: list[tuple[str, list[EvalPoint]]] = []
    for lab, lp in zip(labels, logs):
        pts = parse_eval_blocks(lp)
        series.append((lab, pts))

    out_path = Path(args.out) if Path(args.out).is_absolute() else (ROOT / args.out)
    plot_experiment(
        title=f"{args.exp} — MA/Diagnostics",
        series=series,
        out_path=out_path,
        ma_blocks=args.ma_blocks,
        nz_blocks=args.nz_blocks,
    )

    # Print a tiny summary to stdout for quick decision making.
    for lab, pts in series:
        if not pts:
            print(f"[{args.exp}] {lab}: no eval blocks parsed yet")
            continue
        x = [p.ep for p in pts]
        avg_s = [float(p.avg_score or 0.0) for p in pts]
        best_s = [float(p.best_score or 0.0) for p in pts]
        last_imp = _last_improvement_episode(x, best_s)
        nz = sum(1 for v in avg_s[-10:] if v > 0) / max(min(len(avg_s), 10), 1)
        print(f"[{args.exp}] {lab}: points={len(pts)} last_ep={x[-1]} last_avgS={avg_s[-1]:.2f} bestS={best_s[-1]:.0f} nz_rate(last10blocks)={nz:.2f} last_bestS_improve_ep={last_imp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
