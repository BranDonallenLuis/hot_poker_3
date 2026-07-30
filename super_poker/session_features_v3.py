"""Poker44 v3.0 session feature extraction + launch heuristic.

Extracts bot-detection features from a v2 subject-session, with emphasis on the
NEW telemetry signals the v2.0 redaction had stripped:
  - decision timing (decision_time_ms, decision_std_ms) — bots are fast + low-variance
  - mouse/pointer dynamics (pointer_move events, x/y buckets) — bots move little
Plus richer behavioral fields (real amounts/pot/stack/position).

`session_features(session)` -> dict of float features.
`heuristic_bot_score(session)` -> float in [0,1], deployable at launch with NO
training data (pure timing+mouse priors). Replace with a trained model once the
v2 benchmark exists.
"""
from __future__ import annotations
import math
import statistics as st
from typing import Any


def _num(v, default=0.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _stats(xs, prefix, out):
    if not xs:
        for s in ("mean", "std", "min", "max", "cv", "q10", "q90"):
            out[f"{prefix}_{s}"] = 0.0
        return
    m = st.mean(xs)
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    out[f"{prefix}_mean"] = m
    out[f"{prefix}_std"] = sd
    out[f"{prefix}_min"] = min(xs)
    out[f"{prefix}_max"] = max(xs)
    out[f"{prefix}_cv"] = sd / m if m > 1e-9 else 0.0          # coefficient of variation
    xs_sorted = sorted(xs)
    out[f"{prefix}_q10"] = xs_sorted[max(0, int(0.1 * (len(xs) - 1)))]
    out[f"{prefix}_q90"] = xs_sorted[int(0.9 * (len(xs) - 1))]


def session_features(session: dict[str, Any]) -> dict[str, float]:
    hands = session.get("hands") or []
    telem = session.get("telemetry") or {}
    summary = telem.get("summary") or {}
    events = telem.get("events") or []

    actions = []
    for h in hands:
        for a in (h.get("actions") or []):
            if isinstance(a, dict):
                actions.append(a)

    out: dict[str, float] = {}
    out["n_hands"] = float(len(hands))
    out["n_actions"] = float(len(actions))

    # --- TIMING (the strongest bot signal) ---
    dts = [_num(a.get("decision_time_ms")) for a in actions if a.get("decision_time_ms") is not None]
    gaps = [_num(a.get("time_since_last_action_ms")) for a in actions if a.get("time_since_last_action_ms") is not None]
    _stats(dts, "decision_ms", out)
    _stats(gaps, "gap_ms", out)
    out["frac_fast_lt500"] = (sum(1 for d in dts if d < 500) / len(dts)) if dts else 0.0
    out["frac_fast_lt300"] = (sum(1 for d in dts if d < 300) / len(dts)) if dts else 0.0
    out["frac_slow_gt8000"] = (sum(1 for d in dts if d > 8000) / len(dts)) if dts else 0.0
    # summary timing (authoritative from the platform)
    out["sum_decision_mean_ms"] = _num(summary.get("decision_mean_ms"))
    out["sum_decision_std_ms"] = _num(summary.get("decision_std_ms"))
    out["sum_duration_ms"] = _num(summary.get("duration_ms"))
    out["sum_decision_count"] = _num(summary.get("decision_count"))
    out["sum_event_count"] = _num(summary.get("event_count"))
    out["sum_action_count"] = _num(summary.get("action_count"))

    # --- MOUSE / TELEMETRY (second strongest) ---
    moves = [e for e in events if e.get("event_type") == "pointer_move"]
    clicks = [e for e in events if e.get("event_type") == "click"]
    out["n_events"] = float(len(events))
    out["n_pointer_moves"] = float(len(moves))
    out["n_clicks"] = float(len(clicks))
    out["events_per_action"] = (len(events) / len(actions)) if actions else 0.0
    out["moves_per_action"] = (len(moves) / len(actions)) if actions else 0.0
    out["move_to_click_ratio"] = (len(moves) / len(clicks)) if clicks else 0.0
    # mouse travel: distinct buckets visited + path length over x/y buckets
    xy = [(e.get("value", {}).get("x_bucket"), e.get("value", {}).get("y_bucket")) for e in moves
          if isinstance(e.get("value"), dict)]
    xy = [(x, y) for x, y in xy if x is not None and y is not None]
    out["distinct_mouse_cells"] = float(len({(x, y) for x, y in xy}))
    path = sum(abs(_num(xy[i][0]) - _num(xy[i-1][0])) + abs(_num(xy[i][1]) - _num(xy[i-1][1]))
               for i in range(1, len(xy)))
    out["mouse_path_len"] = float(path)
    out["frac_actions_no_mouse"] = 1.0 - min(1.0, out["moves_per_action"])  # low mouse => bot-like

    # --- BEHAVIORAL (now with real values) ---
    at = [str(a.get("action_type", "")).lower() for a in actions]
    for k in ("fold", "call", "check", "bet", "raise"):
        out[f"share_{k}"] = (at.count(k) / len(at)) if at else 0.0
    out["allin_share"] = (sum(1 for a in actions if a.get("is_all_in")) / len(actions)) if actions else 0.0
    _stats([_num(a.get("pot_size")) for a in actions], "pot", out)
    _stats([_num(a.get("player_stack")) for a in actions], "stack", out)
    return out


def heuristic_bot_score(session: dict[str, Any]) -> float:
    """Launch heuristic — NO training data needed. Bot priors:
    fast decisions + low timing variance + little mouse movement => high risk.
    Calibrate/replace once the v2 benchmark is available."""
    f = session_features(session)
    score = 0.0
    # timing consistency: very low decision variance is mechanical (strongest prior)
    std = f["sum_decision_std_ms"] or f["decision_ms_std"]
    mean = f["sum_decision_mean_ms"] or f["decision_ms_mean"]
    cv = std / mean if mean > 1e-9 else 0.0
    if f["n_actions"] >= 3:                       # need enough actions to judge variance
        score += 0.35 * (1.0 - min(1.0, cv / 0.5))          # cv<0.5 => suspicious
    # fast decisions
    score += 0.25 * f["frac_fast_lt500"]
    # little/no mouse movement relative to actions
    score += 0.25 * f["frac_actions_no_mouse"]
    # very sparse telemetry events (bots interact minimally)
    score += 0.15 * (1.0 - min(1.0, f["events_per_action"] / 3.0))
    return max(0.0, min(1.0, score))
