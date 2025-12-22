#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import deque


ROOT = Path(__file__).resolve().parents[1]
V2_RESULTS = ROOT / "ablation_v2" / "results"
OUT_MD = ROOT / "docs" / "training_status.md"


@dataclass
class ProcInfo:
    pid: int
    ni: int | None
    cpu: float | None
    etime: str | None
    cmd: str
    script: str | None
    flags: str


@dataclass
class EvalBlock:
    ep: int
    avg_reward: float | None
    best_reward: float | None
    avg_score: float | None
    best_score: float | None


@dataclass
class EpisodeStats:
    count: int
    last_n: int
    last_avg_score: float | None
    last_nonzero_score_pct: float | None
    last_best_score: float | None
    last_avg_reward: float | None
    last_best_reward: float | None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def get_running_procs() -> dict[str, ProcInfo]:
    """
    Return mapping exp_name -> ProcInfo for running processes that include:
    - ablation_v2/train/train_v2.py --exp-name X
    - ablation_v2/train/train_hram.py --exp-name X
    - ablation_v2/train/train_hram_doc.py --exp-name X
    """
    out = _run(["ps", "-eo", "pid,ni,etime,%cpu,cmd"])
    procs: dict[str, ProcInfo] = {}
    for line in out.splitlines()[1:]:
        line = line.strip()
        if "ablation_v2/train/" not in line:
            continue
        if "python" not in line:
            continue
        # Split into 5 fields: pid, ni, etime, %cpu, cmd (rest)
        parts = line.split(maxsplit=4)
        if len(parts) < 5:
            continue
        pid_s, ni_s, etime, cpu_s, cmd = parts
        m = re.search(r"--exp-name\s+(\S+)", cmd)
        if not m:
            continue
        exp = m.group(1)
        script_m = re.search(r"(ablation_v2/train/\S+?\.py)", cmd)
        script = script_m.group(1) if script_m else None
        flags = []
        if "--no-mask" in cmd:
            flags.append("no_mask")
        if "--use-embedding" in cmd:
            flags.append("emb")
        if "--use-gumbel" in cmd:
            flags.append("gumbel")
        st = re.search(r"--sparse-topk\s+(\d+)", cmd)
        if st:
            flags.append(f"topk={st.group(1)}")
        tau = re.search(r"--gumbel-tau\s+([\d\.]+)", cmd)
        if tau:
            flags.append(f"tau={tau.group(1)}")
        if "--resume" in cmd:
            flags.append("resume")
        try:
            pid = int(pid_s)
        except Exception:
            continue
        try:
            ni = int(ni_s)
        except Exception:
            ni = None
        try:
            cpu = float(cpu_s)
        except Exception:
            cpu = None
        procs[exp] = ProcInfo(
            pid=pid,
            ni=ni,
            cpu=cpu,
            etime=etime,
            cmd=cmd,
            script=script,
            flags=",".join(flags) if flags else "-",
        )
    return procs


def config_health(exp: str, proc: ProcInfo | None) -> str:
    """
    Light sanity checks between exp name and flags.
    Only uses process args (if running).
    """
    if not proc:
        return "-"
    issues: list[str] = []
    if "gumbel" in exp and "gumbel" not in proc.flags and (proc.script or "").endswith("train_v2.py"):
        issues.append("missing_gumbel_flag")
    if exp == "gumbel_sparse" and "topk=1" not in proc.flags:
        issues.append("missing_topk1")
    if exp == "sparse_moe" and "topk=2" not in proc.flags:
        issues.append("missing_topk2")
    if exp in {"baseline", "gumbel", "sparse_moe", "gumbel_sparse"} and "no_mask" in proc.flags:
        issues.append("unexpected_no_mask")
    if exp == "no_mask" and "no_mask" not in proc.flags:
        issues.append("missing_no_mask_flag")
    return ",".join(issues) if issues else "ok"


def parse_device(log_text: str) -> str | None:
    for pat in [r"✓ 使用MUSA设备:.*", r"✓ 使用CUDA设备:.*", r"✓ 使用CPU设备"]:
        m = re.search(pat, log_text)
        if m:
            return m.group(0).strip()
    return None


def parse_eval_blocks(log_text: str) -> list[EvalBlock]:
    re_eval_a = re.compile(r"^Episode\s+(\d+)/(\d+)\s+\(patience:")
    re_eval_b = re.compile(r"^Episode\s+(\d+)/(\d+)\s*$")
    re_avg_reward = re.compile(r"^\s*平均奖励:\s*([\d\.-]+),\s*最佳奖励:\s*([\d\.-]+)")
    re_avg_score = re.compile(r"^\s*平均分数:\s*([\d\.-]+),\s*最佳分数:\s*([\d\.-]+)")

    lines = log_text.splitlines()
    blocks: list[EvalBlock] = []
    for i, s in enumerate(lines):
        m = re_eval_a.match(s) or re_eval_b.match(s)
        if not m:
            continue
        ep = int(m.group(1))
        avg_r = best_r = avg_s = best_s = None
        for j in range(i + 1, min(i + 12, len(lines))):
            l = lines[j]
            mr = re_avg_reward.match(l)
            if mr:
                avg_r = float(mr.group(1))
                best_r = float(mr.group(2))
            ms = re_avg_score.match(l)
            if ms:
                avg_s = float(ms.group(1))
                best_s = float(ms.group(2))
        blocks.append(EvalBlock(ep=ep, avg_reward=avg_r, best_reward=best_r, avg_score=avg_s, best_score=best_s))
    return blocks


def parse_episode_stats(log_path: Path, last_n: int = 200) -> EpisodeStats:
    """
    Parse per-episode summary lines to reduce noise vs 50-ep eval blocks.
    Looks for:
      - "[DEBUG] 统计完成: reward=..., score=..., steps=..."
    """
    re_done = re.compile(r"统计完成:\s*reward=([\d\.-]+),\s*score=([\d\.-]+),\s*steps=(\d+)")
    scores: deque[float] = deque(maxlen=last_n)
    rewards: deque[float] = deque(maxlen=last_n)
    best_s: float | None = None
    best_r: float | None = None
    count = 0
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re_done.search(line)
                if not m:
                    continue
                count += 1
                r = float(m.group(1))
                s = float(m.group(2))
                rewards.append(r)
                scores.append(s)
                best_r = r if best_r is None else max(best_r, r)
                best_s = s if best_s is None else max(best_s, s)
    except Exception:
        return EpisodeStats(count=0, last_n=last_n, last_avg_score=None, last_nonzero_score_pct=None, last_best_score=None, last_avg_reward=None, last_best_reward=None)

    last_avg_s = (sum(scores) / len(scores)) if scores else None
    last_avg_r = (sum(rewards) / len(rewards)) if rewards else None
    nonzero_pct = None
    if scores:
        nonzero_pct = 100.0 * sum(1 for x in scores if x > 0) / len(scores)
    return EpisodeStats(
        count=count,
        last_n=last_n,
        last_avg_score=last_avg_s,
        last_nonzero_score_pct=nonzero_pct,
        last_best_score=max(scores) if scores else None,
        last_avg_reward=last_avg_r,
        last_best_reward=max(rewards) if rewards else None,
    )


def trend(values: list[float]) -> float | None:
    """
    Simple slope estimate over equally spaced points.
    Returns slope per eval-block step.
    """
    if len(values) < 2:
        return None
    x0 = 0
    xn = len(values) - 1
    y0 = values[0]
    yn = values[-1]
    return (yn - y0) / (xn - x0) if xn != x0 else None


def recommend(exp: str, blocks: list[EvalBlock], device_line: str | None) -> str:
    """
    Heuristic recommendation:
    - If no eval blocks yet: "warming up"
    - If avg_score very low and best_score low: likely stuck -> stop/repurpose
    - If best_score high but avg_score low: rare spikes -> continue but tune
    - If avg_score improving: continue
    """
    if not blocks:
        return "warming up (no eval blocks yet)"

    last = blocks[-1]
    last5 = blocks[-5:] if len(blocks) >= 5 else blocks
    avg_scores = [b.avg_score for b in last5 if b.avg_score is not None]
    avg_rewards = [b.avg_reward for b in last5 if b.avg_reward is not None]
    mean_last5_s = sum(avg_scores) / len(avg_scores) if avg_scores else None
    mean_last5_r = sum(avg_rewards) / len(avg_rewards) if avg_rewards else None

    all_best_s = [b.best_score for b in blocks if b.best_score is not None]
    best_s = max(all_best_s) if all_best_s else None
    all_best_r = [b.best_reward for b in blocks if b.best_reward is not None]
    best_r = max(all_best_r) if all_best_r else None

    slope_s = trend([b.avg_score for b in blocks[-10:] if b.avg_score is not None])
    slope_r = trend([b.avg_reward for b in blocks[-10:] if b.avg_reward is not None])

    # Flag if running on CPU but we expect MUSA
    cpu_flag = ""
    if device_line and "CPU" in device_line:
        cpu_flag = " (CPU)"

    if mean_last5_s is not None and best_s is not None:
        if mean_last5_s < 1.0 and best_s < 80:
            return f"stop/repurpose{cpu_flag}: avgS stuck low, bestS low"
        if mean_last5_s < 1.0 and best_s >= 120:
            return f"continue+tune{cpu_flag}: rare high score, avgS low (needs stability)"
        if mean_last5_s >= 2.0 and (slope_s is None or slope_s >= -0.05):
            return f"continue{cpu_flag}: avgS ok"

    # fallback by reward if score missing
    if mean_last5_r is not None and best_r is not None:
        if mean_last5_r < 5.0 and best_r < 100:
            return f"stop/repurpose{cpu_flag}: avgR stuck low"
        if slope_r is not None and slope_r > 0.2:
            return f"continue{cpu_flag}: avgR improving"

    return f"continue (insufficient evidence){cpu_flag}"


def main() -> None:
    if not V2_RESULTS.exists():
        raise SystemExit("No ablation_v2/results directory found")

    running = get_running_procs()

    rows = []
    for exp_dir in sorted([p for p in V2_RESULTS.iterdir() if p.is_dir()]):
        log = exp_dir / "training.log"
        if not log.exists():
            continue
        txt = _read_text(log)
        blocks = parse_eval_blocks(txt)
        dev = parse_device(txt)
        ep_stats = parse_episode_stats(log, last_n=200)
        has_tb = "Traceback" in txt
        has_nan = ("NaN/Inf" in txt) or ("nan, nan" in txt.lower()) or ("logits contains NaN/Inf" in txt)

        last_ep = blocks[-1].ep if blocks else None
        last_avg_s = blocks[-1].avg_score if blocks else None
        last_best_s = max([b.best_score for b in blocks if b.best_score is not None], default=None)
        last_avg_r = blocks[-1].avg_reward if blocks else None
        last_best_r = max([b.best_reward for b in blocks if b.best_reward is not None], default=None)

        proc = running.get(exp_dir.name)
        rows.append(
            {
                "exp": exp_dir.name,
                "running": "yes" if proc else "no",
                "pid": proc.pid if proc else None,
                "ni": proc.ni if proc else None,
                "cpu%": proc.cpu if proc else None,
                "flags": proc.flags if proc else "-",
                "cfg": config_health(exp_dir.name, proc),
                "device": dev,
                "last_eval_ep": last_ep,
                "last_avgS": last_avg_s,
                "bestS": last_best_s,
                "last_avgR": last_avg_r,
                "bestR": last_best_r,
                "ep_count": ep_stats.count,
                "ep_last200_avgS": ep_stats.last_avg_score,
                "ep_last200_nz%": ep_stats.last_nonzero_score_pct,
                "ep_last200_bestS": ep_stats.last_best_score,
                "ep_last200_avgR": ep_stats.last_avg_reward,
                "traceback": "yes" if has_tb else "no",
                "nan_warn": "yes" if has_nan else "no",
                "recommendation": recommend(exp_dir.name, blocks, dev),
                "log": str(log.relative_to(ROOT)),
            }
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Training Status (Auto)")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Scope: `{V2_RESULTS.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| exp | running | pid | ni | cpu% | flags | cfg | device | lastEvalEp | lastAvgS | bestS | lastAvgR | bestR | epCnt | epAvgS(200) | epNZ%(200) | epBestS(200) | epAvgR(200) | tb | nan | recommendation | log |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["exp"],
                    r["running"],
                    str(r["pid"] or "-"),
                    str(r["ni"] or "-"),
                    _fmt(r["cpu%"]),
                    r["flags"],
                    r["cfg"],
                    (r["device"] or "-"),
                    str(r["last_eval_ep"] or "-"),
                    _fmt(r["last_avgS"]),
                    _fmt(r["bestS"]),
                    _fmt(r["last_avgR"]),
                    _fmt(r["bestR"]),
                    str(r["ep_count"]),
                    _fmt(r["ep_last200_avgS"]),
                    _fmt(r["ep_last200_nz%"]),
                    _fmt(r["ep_last200_bestS"]),
                    _fmt(r["ep_last200_avgR"]),
                    r["traceback"],
                    r["nan_warn"],
                    r["recommendation"],
                    f"`{r['log']}`",
                ]
            )
            + " |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
