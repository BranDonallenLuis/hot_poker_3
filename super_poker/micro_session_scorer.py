"""Poker44 v3.0 micro-session scorer — schema v4.1 (MicroSessionDetectionSynapse).

v4.1 items carry NO telemetry (no timing, no mouse) — only 4 strategic
`decisions`, each purely categorical: phase, position_group, pressure,
action_type, size_bucket, is_all_in. So the telemetry approach in
`session_features_v3.py` does NOT apply here.

`micro_features(item)`  -> dict[str, float] over the 4-decision structure.
`heuristic_micro_score(item)` -> float in [0,1]. PROVISIONAL — a placeholder
    that returns valid, varied scores so the miner responds (instead of a hard
    reject / zero) until labeled tournament data lets us train a real model.
Set `POKER44_V4_MODEL_PATH` to a trained artifact to replace the heuristic.
"""
from __future__ import annotations

import os
from typing import Any

_AGGR = {"bet", "raise", "all_in"}
_PASSIVE = {"check", "call"}
_BIG_SIZE = {"pot", "overbet", "all_in"}
_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")


def micro_features(item: dict[str, Any]) -> dict[str, float]:
    decisions = item.get("decisions") or []
    n = float(len(decisions)) or 1.0
    acts = [str(d.get("action_type", "")) for d in decisions]
    sizes = [str(d.get("size_bucket", "")) for d in decisions]
    positions = [str(d.get("position_group", "")) for d in decisions]

    out: dict[str, float] = {"n_decisions": float(len(decisions))}
    for a in _ACTIONS:
        out[f"share_{a}"] = acts.count(a) / n
    out["aggr_rate"] = sum(1 for a in acts if a in _AGGR) / n
    out["passive_rate"] = sum(1 for a in acts if a in _PASSIVE) / n
    out["allin_rate"] = sum(
        1 for d in decisions if d.get("is_all_in") or d.get("action_type") == "all_in"
    ) / n
    out["bigsize_rate"] = sum(1 for s in sizes if s in _BIG_SIZE) / n
    out["action_diversity"] = len(set(acts)) / n
    out["size_diversity"] = len(
        {s for s in sizes if s not in ("not_applicable", "unknown")}
    ) / n
    out["position_diversity"] = len(set(positions)) / n

    faced = [d for d in decisions if d.get("pressure") == "facing_bet"]
    out["frac_faced"] = len(faced) / n
    out["fold_vs_pressure"] = (
        sum(1 for d in faced if d.get("action_type") == "fold") / len(faced)
    ) if faced else 0.0
    out["aggr_vs_pressure"] = (
        sum(1 for d in faced if d.get("action_type") in _AGGR) / len(faced)
    ) if faced else 0.0
    nocall = [d for d in decisions if d.get("pressure") == "no_call"]
    out["aggr_vs_nocall"] = (
        sum(1 for d in nocall if d.get("action_type") in _AGGR) / len(nocall)
    ) if nocall else 0.0
    out["n_postflop"] = float(
        sum(1 for d in decisions if d.get("phase") in ("flop", "turn", "river"))
    )
    return out


def heuristic_micro_score(item: dict[str, Any]) -> float:
    """PROVISIONAL strategic heuristic (no labels yet). Moderate weights keep
    scores near 0.5 so a wrong-direction guess does limited damage, while still
    producing a usable spread with some items above 0.5."""
    f = micro_features(item)
    score = 0.5
    score += 0.12 * (f["aggr_rate"] - 0.5) * 2.0        # more aggressive -> higher
    score += 0.10 * f["allin_rate"]                     # frequent all-ins
    score += 0.08 * f["bigsize_rate"]                   # pot/overbet/all-in sizing
    score += 0.10 * (0.5 - f["action_diversity"])       # mechanical / low diversity
    score += 0.06 * f["aggr_vs_nocall"]                 # betting into no pressure
    return max(0.0, min(1.0, score))


class MicroSessionScorer:
    """Scores v4.1 micro-session items; trained model via POKER44_V4_MODEL_PATH,
    else the provisional heuristic."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.feature_names = None
        self.model_version = "v4-micro-heuristic"
        path = model_path or os.getenv("POKER44_V4_MODEL_PATH", "")
        if path and os.path.exists(path):
            try:
                import joblib

                art = joblib.load(path)
                self.model = art["model"]
                self.feature_names = list(art["feature_names"])
                self.model_version = str(
                    art.get("metadata", {}).get("model_version", "v4-micro")
                )
            except Exception:
                self.model = None

        # Optional empirical-CDF calibration for the heuristic (used ONLY when no
        # trained model). Recenters the heuristic's output distribution using
        # captured live items so ~50% of items score >=0.5 instead of clustering
        # below it. Monotone -> ranking is unchanged. JSON produced by
        # scripts/calibrate_micro_heuristic.py; path via POKER44_V4_CALIBRATION_PATH.
        self._cal = None
        if self.model is None:
            cal_path = os.getenv("POKER44_V4_CALIBRATION_PATH", "")
            if cal_path and os.path.exists(cal_path):
                try:
                    import json as _json

                    with open(cal_path) as _fh:
                        cal = _json.load(_fh)
                    ref = sorted(float(x) for x in cal.get("ref_scores", []))
                    if ref:
                        self._cal = ref
                        self.model_version = str(
                            cal.get("model_version", "v4-micro-cal")
                        )
                except Exception:
                    self._cal = None

    def score_items(self, items: list[dict]) -> list[float]:
        if not items:
            return []
        if self.model is not None and self.feature_names is not None:
            import pandas as pd

            X = pd.DataFrame([micro_features(it) for it in items])
            X = X.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
            raw = self.model.predict_proba(X.astype(float))[:, 1]
            return [float(min(1.0, max(0.0, v))) for v in raw]
        scores = [heuristic_micro_score(it) for it in items]
        if self._cal is not None:
            import bisect

            n = len(self._cal)
            # midpoint percentile: ties map to the MIDDLE of their rank band so
            # the distribution centers at 0.5 (bisect_right alone overshoots).
            scores = [
                max(0.0, min(1.0,
                    (bisect.bisect_left(self._cal, s)
                     + bisect.bisect_right(self._cal, s)) / (2.0 * n)))
                for s in scores
            ]
        return scores
