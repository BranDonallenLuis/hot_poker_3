#!/usr/bin/env python3
"""Build a LABELED v4.1 micro-session corpus (build-your-own training data).

We have no official labels, so we synthesize a labeled corpus from a human
policy + 6 bot archetypes (matching Poker44's "6 bot families" structure) and
train against it. The bot signal is grounded in poker domain knowledge:

  * characteristic action frequencies per archetype (nit folds, station calls,
    maniac shoves, TAG/LAG/GTO in between),
  * CONCENTRATED bet-sizing (bots reuse a small set of sizes; humans spread),
  * lower within-item variability (a bot applies one consistent style across
    the 4 sampled decisions; a human is more variable).

Contexts (phase / position / pressure frequencies) are calibrated to your
CAPTURED real items via --capture so generated items look realistic, while the
labels come from the policy that produced each item.

This is a HYPOTHESIS about how the real bots differ. Deploy, read the live
score, and iterate the policies. Output is a labeled JSON that
train_micro_model.py consumes directly (is_bot field).

Usage:
    python scripts/build_micro_corpus.py \
        --capture ~/super_poker_3/live_capture/micro_*.jsonl \
        --out artifacts/micro_corpus.json \
        --n 12000 --seed 7
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

PHASES = ["preflop", "flop", "turn", "river"]
POSITIONS = ["early", "late", "blinds"]
PRESSURES = ["facing_bet", "no_call"]
FACING_ACTIONS = ["fold", "call", "raise", "all_in"]
NOCALL_ACTIONS = ["check", "fold", "bet", "raise"]
AGGR_SIZES = ["third_pot_or_less", "half_pot", "three_quarter_pot", "pot", "overbet"]

# --- policies: action prob given pressure, and size prob for bet/raise -------
# Human = variable baseline, spread sizes. Bots = characteristic + concentrated.
HUMAN = {
    "facing_bet": {"fold": 0.30, "call": 0.55, "raise": 0.12, "all_in": 0.03},
    "no_call": {"check": 0.45, "fold": 0.20, "bet": 0.25, "raise": 0.10},
    "size": {"third_pot_or_less": 0.22, "half_pot": 0.28, "three_quarter_pot": 0.22,
             "pot": 0.17, "overbet": 0.11},
}
BOTS = {
    "nit": {
        "facing_bet": {"fold": 0.62, "call": 0.30, "raise": 0.07, "all_in": 0.01},
        "no_call": {"check": 0.60, "fold": 0.25, "bet": 0.12, "raise": 0.03},
        "size": {"third_pot_or_less": 0.20, "half_pot": 0.55, "three_quarter_pot": 0.20,
                 "pot": 0.05, "overbet": 0.00},
    },
    "station": {
        "facing_bet": {"fold": 0.10, "call": 0.82, "raise": 0.07, "all_in": 0.01},
        "no_call": {"check": 0.60, "fold": 0.08, "bet": 0.22, "raise": 0.10},
        "size": {"third_pot_or_less": 0.45, "half_pot": 0.40, "three_quarter_pot": 0.12,
                 "pot": 0.03, "overbet": 0.00},
    },
    "tag": {
        "facing_bet": {"fold": 0.45, "call": 0.35, "raise": 0.18, "all_in": 0.02},
        "no_call": {"check": 0.38, "fold": 0.12, "bet": 0.38, "raise": 0.12},
        "size": {"third_pot_or_less": 0.20, "half_pot": 0.55, "three_quarter_pot": 0.20,
                 "pot": 0.05, "overbet": 0.00},
    },
    "lag": {
        "facing_bet": {"fold": 0.15, "call": 0.42, "raise": 0.36, "all_in": 0.07},
        "no_call": {"check": 0.22, "fold": 0.05, "bet": 0.46, "raise": 0.27},
        "size": {"third_pot_or_less": 0.15, "half_pot": 0.35, "three_quarter_pot": 0.35,
                 "pot": 0.15, "overbet": 0.00},
    },
    "maniac": {
        "facing_bet": {"fold": 0.08, "call": 0.22, "raise": 0.45, "all_in": 0.25},
        "no_call": {"check": 0.12, "fold": 0.03, "bet": 0.40, "raise": 0.45},
        "size": {"third_pot_or_less": 0.05, "half_pot": 0.15, "three_quarter_pot": 0.25,
                 "pot": 0.30, "overbet": 0.25},
    },
    "gto": {  # human-like frequencies but CLEAN solver sizes (1/3, 1/2 only)
        "facing_bet": {"fold": 0.33, "call": 0.50, "raise": 0.15, "all_in": 0.02},
        "no_call": {"check": 0.42, "fold": 0.10, "bet": 0.34, "raise": 0.14},
        "size": {"third_pot_or_less": 0.50, "half_pot": 0.45, "three_quarter_pot": 0.05,
                 "pot": 0.00, "overbet": 0.00},
    },
}

# Default context frequencies (overridden by --capture calibration).
CTX = {
    "phase": {"preflop": 0.50, "flop": 0.35, "turn": 0.10, "river": 0.05},
    "position": {"early": 0.30, "late": 0.40, "blinds": 0.30},
    "pressure": {"facing_bet": 0.60, "no_call": 0.40},
}


def _pick(rng, dist):
    r = rng.random()
    c = 0.0
    for k, p in dist.items():
        c += p
        if r <= c:
            return k
    return next(iter(dist))


def _sharpen(dist, t):
    """Temperature<1 sharpens (more consistent/bot-like); >1 flattens (human)."""
    powered = {k: (p ** (1.0 / t) if p > 0 else 0.0) for k, p in dist.items()}
    s = sum(powered.values()) or 1.0
    return {k: v / s for k, v in powered.items()}


def calibrate_ctx(capture_globs):
    from collections import Counter
    phase, pos, pres = Counter(), Counter(), Counter()
    n = 0
    files = []
    for g in capture_globs:
        files.extend(glob.glob(os.path.expanduser(g)))
    for fp in files:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for it in rec.get("items", []):
                for d in it.get("decisions", []):
                    phase[d.get("phase")] += 1
                    pos[d.get("position_group")] += 1
                    pres[d.get("pressure")] += 1
                    n += 1
    if n == 0:
        return None
    norm = lambda c, keys: {k: c.get(k, 0) / sum(c.values()) for k in keys}
    return {
        "phase": norm(phase, PHASES),
        "position": norm(pos, POSITIONS),
        "pressure": norm(pres, PRESSURES),
    }


def gen_decision(rng, policy, ctx, temp):
    phase = _pick(rng, ctx["phase"])
    pos = _pick(rng, ctx["position"])
    pres = _pick(rng, ctx["pressure"])
    act = _pick(rng, _sharpen(policy[pres], temp))
    if act in ("bet", "raise"):
        size = _pick(rng, _sharpen(policy["size"], temp))
    elif act == "all_in":
        size = "all_in"
    else:
        size = "not_applicable"
    return {"phase": phase, "position_group": pos, "pressure": pres,
            "action_type": act, "size_bucket": size, "is_all_in": act == "all_in"}


def gen_item(rng, policy, ctx, temp, item_id):
    # 4 decisions, at least one postflop (matches the v4.1 contract).
    for _ in range(20):
        decs = [gen_decision(rng, policy, ctx, temp) for _ in range(4)]
        if any(d["phase"] in ("flop", "turn", "river") for d in decs):
            break
    for i, d in enumerate(decs):
        d["decision_number"] = i + 1
    return {"schema_version": "4.1", "item_id": item_id, "window_id": "synthetic",
            "decisions": decs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", nargs="*", default=[])
    ap.add_argument("--out", default="artifacts/micro_corpus.json")
    ap.add_argument("--n", type=int, default=12000, help="total items (~50/50 bot/human)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--human-temp", type=float, default=1.6, help=">1 = more variable")
    ap.add_argument("--bot-temp", type=float, default=0.7, help="<1 = more consistent")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ctx = CTX
    if args.capture:
        cal = calibrate_ctx(args.capture)
        if cal:
            ctx = cal
            print(f"[ctx] calibrated context freqs from capture: "
                  f"phase={ {k: round(v,2) for k,v in ctx['phase'].items()} }")

    records = []
    n_bot = args.n // 2
    n_human = args.n - n_bot
    fams = list(BOTS.keys())
    for k in range(n_human):
        it = gen_item(rng, HUMAN, ctx, args.human_temp, f"h{k}")
        records.append({**it, "is_bot": 0, "bot_family": "human"})
    for k in range(n_bot):
        fam = fams[k % len(fams)]
        it = gen_item(rng, BOTS[fam], ctx, args.bot_temp, f"b{k}")
        records.append({**it, "is_bot": 1, "bot_family": fam})
    rng.shuffle(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f)
    print(f"[corpus] wrote {len(records)} items "
          f"({n_bot} bots across {len(fams)} families / {n_human} humans) -> {out}")
    print(f"[next] python scripts/train_micro_model.py --data {out} "
          f"--out artifacts/micro_model.joblib --label-key is_bot "
          f"--live <capture.jsonl> --model-version v4-micro-byo-$(date +%Y%m%d)")


if __name__ == "__main__":
    main()
