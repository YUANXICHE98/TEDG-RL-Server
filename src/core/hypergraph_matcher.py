"""Hypergraph matching utilities.

Implements a multi-evidence-channel matcher that returns a Top-K set of
explanations (hyperedges) with per-channel coverage + time decay scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    edge: Dict[str, Any]
    match_vec: Tuple[float, float, float, float]  # (cov_pre, cov_scene, cov_effect, cov_rule)
    global_score: float
    decay: float
    score: float
    channel_argmax: int


def _to_str_set(items: Optional[Iterable[Any]]) -> set[str]:
    if not items:
        return set()
    out: set[str] = set()
    for it in items:
        if it is None:
            continue
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.add(s)
        else:
            s = str(it).strip()
            if s:
                out.add(s)
    return out


def _coverage(query: Sequence[str], target: Sequence[str]) -> float:
    """Coverage of query by target: |query ∩ target| / max(|query|, 1)."""
    q = _to_str_set(query)
    if not q:
        return 0.0
    t = _to_str_set(target)
    return float(len(q & t)) / float(len(q))


def _edge_rule_atoms(edge: Dict[str, Any]) -> set[str]:
    """Extract rule-like atoms from an edge (conditional_effects + side_effects)."""
    meta = edge.get("eff_metadata", {}) or {}
    atoms: set[str] = set()
    ces = meta.get("conditional_effects") or []
    for ce in ces:
        if not isinstance(ce, dict):
            continue
        eff = ce.get("effect")
        cond = ce.get("condition")
        if isinstance(eff, str) and eff.strip():
            atoms.add(eff.strip())
        if isinstance(cond, str) and cond.strip():
            # Keep the raw condition string; caller can provide exact tokens if desired.
            atoms.add(cond.strip())
    for se in meta.get("side_effects") or []:
        if isinstance(se, str) and se.strip():
            atoms.add(se.strip())
    return atoms


class HypergraphMatcher:
    """Match plot atoms to hyperedges using 4-channel coverage + exponential time decay."""

    def __init__(
        self,
        hypergraph: Dict[str, Any],
        weights: Optional[Sequence[float]] = None,
        tau: float = 200.0,
    ):
        self.hypergraph = hypergraph
        self.edges: List[Dict[str, Any]] = list(hypergraph.get("hyperedges", []) or [])
        if weights is None:
            weights = (0.35, 0.35, 0.2, 0.1)
        if len(weights) != 4:
            raise ValueError("weights must have length 4")
        w = np.asarray(weights, dtype=np.float32)
        if float(w.sum()) <= 0:
            raise ValueError("weights sum must be > 0")
        self.w = w / float(w.sum())
        self.tau = float(tau)

    def match(
        self,
        plot_atoms: Dict[str, Sequence[str]],
        *,
        t_now: float,
        t_i: Optional[float] = None,
        top_k: int = 8,
    ) -> List[MatchResult]:
        """
        Args:
            plot_atoms: {"pre": [...], "scene": [...], "effect": [...], "rule": [...]}
            t_now: current time index (step)
            t_i: time index of the plot/explanation; if None, decay=1
            top_k: return top-k matches by score
        """
        pre_q = plot_atoms.get("pre") or []
        scene_q = plot_atoms.get("scene") or []
        eff_q = plot_atoms.get("effect") or []
        rule_q = plot_atoms.get("rule") or []

        if t_i is None or self.tau <= 0:
            decay = 1.0
        else:
            decay = math.exp(-max(0.0, float(t_now) - float(t_i)) / self.tau)

        results: List[MatchResult] = []
        for edge in self.edges:
            cov_pre = _coverage(pre_q, edge.get("pre_nodes") or [])
            cov_scene = _coverage(scene_q, edge.get("scene_atoms") or [])
            cov_eff = _coverage(eff_q, edge.get("eff_nodes") or [])
            cov_rule = _coverage(rule_q, list(_edge_rule_atoms(edge)))

            match_vec = np.asarray([cov_pre, cov_scene, cov_eff, cov_rule], dtype=np.float32)
            global_score = float(np.dot(self.w, match_vec))
            score = global_score * float(decay)
            channel_argmax = int(np.argmax(match_vec)) if match_vec.size else 0

            results.append(
                MatchResult(
                    edge=edge,
                    match_vec=(float(cov_pre), float(cov_scene), float(cov_eff), float(cov_rule)),
                    global_score=float(global_score),
                    decay=float(decay),
                    score=float(score),
                    channel_argmax=channel_argmax,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: max(1, int(top_k))]

    @staticmethod
    def select_channel_edges(topk: Sequence[MatchResult]) -> Dict[str, MatchResult]:
        """Pick per-channel best edge from Top-K (ties broken by overall score)."""
        def best_by(idx: int) -> MatchResult:
            return max(topk, key=lambda r: (r.match_vec[idx], r.score))

        if not topk:
            raise ValueError("topk is empty")

        return {
            "pre": best_by(0),
            "scene": best_by(1),
            "effect": best_by(2),
            "rule": best_by(3),
        }

