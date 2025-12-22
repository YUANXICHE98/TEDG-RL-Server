#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import statistics


ROOT = Path(__file__).resolve().parents[1]
STATUS_MD = ROOT / "docs" / "training_status.md"
STATUS_SCRIPT = ROOT / "tools" / "training_status.py"
ABLATION_CSV = ROOT / "docs" / "ablation_results_comparison.csv"
OUT_DIR = ROOT / "docs" / "visualizations" / "current"


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _ensure_status_md() -> None:
    if STATUS_SCRIPT.exists():
        _run(["python", str(STATUS_SCRIPT)])
    if not STATUS_MD.exists():
        raise FileNotFoundError(f"Missing {STATUS_MD}")


def _safe_float(x: str) -> float | None:
    x = (x or "").strip()
    if x in {"-", ""}:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: str) -> int | None:
    x = (x or "").strip()
    if x in {"-", ""}:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


@dataclass
class StatusRow:
    exp: str
    running: str
    pid: int | None
    ni: int | None
    cpu_pct: float | None
    flags: str
    cfg: str
    device: str
    last_eval_ep: int | None
    last_avgS: float | None
    bestS: float | None
    last_avgR: float | None
    bestR: float | None
    epCnt: int | None
    epAvgS200: float | None
    epNZ200: float | None
    epBestS200: float | None
    epAvgR200: float | None
    tb: str
    nan: str


def _parse_md_table(md: str) -> list[StatusRow]:
    # Find first markdown table with header containing "epAvgS(200)"
    lines = md.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("| exp ") and "epAvgS(200)" in line:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find status table in training_status.md")

    header = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
    # Skip separator line
    rows: list[StatusRow] = []
    for line in lines[header_idx + 2 :]:
        if not line.startswith("|"):
            break
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) != len(header):
            continue
        d = dict(zip(header, cols))
        rows.append(
            StatusRow(
                exp=d["exp"],
                running=d["running"],
                pid=_safe_int(d["pid"]),
                ni=_safe_int(d["ni"]),
                cpu_pct=_safe_float(d["cpu%"]),
                flags=d.get("flags", "-"),
                cfg=d.get("cfg", "-"),
                device=d.get("device", "-"),
                last_eval_ep=_safe_int(d.get("lastEvalEp", "-")),
                last_avgS=_safe_float(d.get("lastAvgS", "-")),
                bestS=_safe_float(d.get("bestS", "-")),
                last_avgR=_safe_float(d.get("lastAvgR", "-")),
                bestR=_safe_float(d.get("bestR", "-")),
                epCnt=_safe_int(d.get("epCnt", "-")),
                epAvgS200=_safe_float(d.get("epAvgS(200)", "-")),
                epNZ200=_safe_float(d.get("epNZ%(200)", "-")),
                epBestS200=_safe_float(d.get("epBestS(200)", "-")),
                epAvgR200=_safe_float(d.get("epAvgR(200)", "-")),
                tb=d.get("tb", "-"),
                nan=d.get("nan", "-"),
            )
        )
    return rows


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


def _barh(
    title: str,
    rows: list[Any],
    value_fn,
    out: Path,
    sort_desc: bool = True,
    x_label: str | None = None,
):
    plt = _maybe_import_matplotlib()
    items = [(r.exp, value_fn(r), r) for r in rows]
    items = [(n, v, r) for (n, v, r) in items if v is not None]
    items.sort(key=lambda x: x[1], reverse=sort_desc)

    if not items:
        return

    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    meta = [x[2] for x in items]

    colors = []
    for r in meta:
        device = (getattr(r, "device", "") or "")
        if "MUSA" in device:
            colors.append("#2ecc71")  # green
        elif "CPU" in device:
            colors.append("#95a5a6")  # gray
        else:
            colors.append("#3498db")  # blue

    fig_h = max(3.0, 0.35 * len(names) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.barh(names, vals, color=colors, alpha=0.85)
    ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)

    # annotate flags for quick scan
    for i, (v, r) in enumerate(zip(vals, meta)):
        tag = []
        if getattr(r, "running", "no") == "yes":
            tag.append("RUN")
        if getattr(r, "tb", "no") == "yes":
            tag.append("TB")
        if getattr(r, "nan", "no") == "yes":
            tag.append("NaN")
        cfg = getattr(r, "cfg", "ok")
        if cfg != "ok" and cfg != "-":
            tag.append(str(cfg))
        s = ",".join(tag)
        if s:
            ax.text(v, i, f"  {s}", va="center", ha="left", fontsize=9, color="#2c3e50")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _dashboard(rows: list[StatusRow], out: Path) -> None:
    plt = _maybe_import_matplotlib()
    # Keep only main experiments for compact dashboard
    order = ["baseline", "no_mask", "sparse_moe", "gumbel_sparse", "gumbel_fixed", "hram_e2e", "hram_doc_fixed"]
    by_name = {r.exp: r for r in rows}
    rows2 = [by_name[n] for n in order if n in by_name]
    if not rows2:
        return

    names = [r.exp for r in rows2]
    bestS = [r.bestS or 0.0 for r in rows2]
    avgS200 = [r.epAvgS200 if r.epAvgS200 is not None else (r.last_avgS or 0.0) for r in rows2]
    nz200 = [r.epNZ200 or 0.0 for r in rows2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].bar(names, bestS, color="#8e44ad", alpha=0.85)
    axes[0].set_title("Best Score (BestS)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, avgS200, color="#2980b9", alpha=0.85)
    axes[1].set_title("Avg Score (epAvgS(200) or lastAvgS)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(names, nz200, color="#16a085", alpha=0.85)
    axes[2].set_title("Non-zero Score Rate (epNZ%(200))")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _parse_ablation_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)

@dataclass
class V1AggRow:
    exp: str
    bestS: float | None
    avgS200: float | None
    nz200: float | None
    bestR: float | None
    avgR200: float | None


def _pretty_v1_name(d: str) -> str:
    name = d.strip()
    for prefix in ("results_", "exp_", "experiment_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def _load_v1_agg_rows() -> list[V1AggRow]:
    base = ROOT / "ablation_v1" / "results"
    if not base.exists():
        return []

    out: list[V1AggRow] = []
    for exp_dir in sorted(base.iterdir()):
        log_json = exp_dir / "logs" / "training_log.json"
        if not log_json.exists():
            continue
        try:
            d = json.loads(_read_text(log_json) or "{}")
        except Exception:
            continue

        scores = [float(x) for x in (d.get("episode_scores") or []) if x is not None]
        rewards = [float(x) for x in (d.get("episode_rewards") or []) if x is not None]
        if not scores and not rewards:
            continue

        w = 200
        s_tail = scores[-w:] if len(scores) >= w else scores[:]
        r_tail = rewards[-w:] if len(rewards) >= w else rewards[:]

        def _mean(xs: list[float]) -> float | None:
            return (sum(xs) / len(xs)) if xs else None

        def _best(xs: list[float]) -> float | None:
            return max(xs) if xs else None

        def _nz_pct(xs: list[float]) -> float | None:
            if not xs:
                return None
            return 100.0 * sum(1 for x in xs if x > 0.0) / float(len(xs))

        out.append(
            V1AggRow(
                exp=_pretty_v1_name(exp_dir.name),
                bestS=_best(scores),
                avgS200=_mean(s_tail),
                nz200=_nz_pct(s_tail),
                bestR=_best(rewards),
                avgR200=_mean(r_tail),
            )
        )
    return out


def main() -> None:
    _ensure_status_md()
    rows = _parse_md_table(_read_text(STATUS_MD))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # V2 live metrics (from training_status)
    _barh(
        "V2 - Best Score (BestS)",
        rows,
        lambda r: r.bestS,
        OUT_DIR / "v2_best_score.png",
        x_label="Score",
    )
    _barh(
        "V2 - Avg Score (epAvgS(200) or lastAvgS)",
        rows,
        lambda r: r.epAvgS200 if r.epAvgS200 is not None else r.last_avgS,
        OUT_DIR / "v2_avg_score.png",
        x_label="Score",
    )
    _barh(
        "V2 - Non-zero Score Rate (epNZ%(200))",
        rows,
        lambda r: r.epNZ200,
        OUT_DIR / "v2_nonzero_score_rate.png",
        x_label="%",
    )
    _barh(
        "V2 - Best Reward (BestR)",
        rows,
        lambda r: r.bestR,
        OUT_DIR / "v2_best_reward.png",
        x_label="Reward",
    )
    _barh(
        "V2 - Avg Reward (epAvgR(200) or lastAvgR)",
        rows,
        lambda r: r.epAvgR200 if r.epAvgR200 is not None else r.last_avgR,
        OUT_DIR / "v2_avg_reward.png",
        x_label="Reward",
    )
    _dashboard(rows, OUT_DIR / "v2_dashboard.png")

    # V1 aggregated metrics from training_log.json
    v1_rows = _load_v1_agg_rows()
    if v1_rows:
        _barh(
            "V1 - Best Score (BestS)",
            v1_rows,
            lambda r: r.bestS,
            OUT_DIR / "v1_best_score.png",
            x_label="Score",
        )
        _barh(
            "V1 - Avg Score (last 200 episodes)",
            v1_rows,
            lambda r: r.avgS200,
            OUT_DIR / "v1_avg_score.png",
            x_label="Score",
        )
        _barh(
            "V1 - Non-zero Score Rate (last 200 episodes)",
            v1_rows,
            lambda r: r.nz200,
            OUT_DIR / "v1_nonzero_score_rate.png",
            x_label="%",
        )
        _barh(
            "V1 - Best Reward (BestR)",
            v1_rows,
            lambda r: r.bestR,
            OUT_DIR / "v1_best_reward.png",
            x_label="Reward",
        )
        _barh(
            "V1 - Avg Reward (last 200 episodes)",
            v1_rows,
            lambda r: r.avgR200,
            OUT_DIR / "v1_avg_reward.png",
            x_label="Reward",
        )

        # Compact dashboard (score metrics only, like V2)
        plt = _maybe_import_matplotlib()
        names = [r.exp for r in v1_rows]
        bestS = [r.bestS or 0.0 for r in v1_rows]
        avgS200 = [r.avgS200 or 0.0 for r in v1_rows]
        nz200 = [r.nz200 or 0.0 for r in v1_rows]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        axes[0].bar(names, bestS, color="#8e44ad", alpha=0.85)
        axes[0].set_title("Best Score (BestS)")
        axes[0].tick_params(axis="x", rotation=20)

        axes[1].bar(names, avgS200, color="#2980b9", alpha=0.85)
        axes[1].set_title("Avg Score (last 200 episodes)")
        axes[1].tick_params(axis="x", rotation=20)

        axes[2].bar(names, nz200, color="#16a085", alpha=0.85)
        axes[2].set_title("Non-zero Score Rate (last 200)")
        axes[2].set_ylabel("%")
        axes[2].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        fig.savefig(OUT_DIR / "v1_dashboard.png")
        plt.close(fig)

    # Cross-version quick plot (best_score / best_reward from comparison CSV)
    records = _parse_ablation_csv(ABLATION_CSV)
    if records:
        # Map exp->best score by version
        v1 = [r for r in records if r.get("version") == "v1"]
        v2 = [r for r in records if r.get("version") == "v2"]

        def _bar(title: str, pairs: Iterable[tuple[str, float]], out: Path, x_label: str):
            plt = _maybe_import_matplotlib()
            pairs = [(n, v) for (n, v) in pairs if v is not None]
            pairs.sort(key=lambda x: x[1], reverse=True)
            if not pairs:
                return
            names = [p[0] for p in pairs]
            vals = [p[1] for p in pairs]
            fig_h = max(3.0, 0.35 * len(names) + 1.2)
            fig, ax = plt.subplots(figsize=(10.5, fig_h))
            ax.barh(names, vals, color="#34495e", alpha=0.85)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            plt.tight_layout()
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out)
            plt.close(fig)

        def _get_float(d: dict[str, str], k: str) -> float | None:
            try:
                return float(d.get(k, "nan"))
            except Exception:
                return None

        # Only produce V1-from-CSV plots if we didn't already write V1 plots from raw logs.
        if not (OUT_DIR / "v1_best_score.png").exists():
            _bar(
                "V1 - Best Score",
                [(r.get("exp", "?"), _get_float(r, "best_score")) for r in v1],
                OUT_DIR / "v1_best_score.png",
                "Score",
            )
        if not (OUT_DIR / "v1_best_reward.png").exists():
            _bar(
                "V1 - Best Reward",
                [(r.get("exp", "?"), _get_float(r, "best_reward")) for r in v1],
                OUT_DIR / "v1_best_reward.png",
                "Reward",
            )
        # v2 from CSV is from eval blocks; still useful
        _bar(
            "V2 (from logs) - Best Score",
            [(r.get("exp", "?"), _get_float(r, "best_score")) for r in v2],
            OUT_DIR / "v2_best_score_from_csv.png",
            "Score",
        )
        _bar(
            "V2 (from logs) - Best Reward",
            [(r.get("exp", "?"), _get_float(r, "best_reward")) for r in v2],
            OUT_DIR / "v2_best_reward_from_csv.png",
            "Reward",
        )

    print(f"✓ Wrote plots to: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
