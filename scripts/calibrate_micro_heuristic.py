#!/usr/bin/env python3
"""Fit an empirical-CDF calibration for the v4.1 micro heuristic from captured
live items.

The raw heuristic clusters most scores below 0.5 (only ~28% >=0.5), which makes
accuracy sub-random. This reads the captured micro-session items, recomputes the
raw heuristic score for each, and saves the sorted reference distribution. At
inference the scorer maps each raw score to its percentile in that reference, so
~50% of items land >=0.5. It is a MONOTONE transform -> ranking (recall@FPR.05,
AP) is unchanged; only threshold metrics (accuracy/precision) shift.

NOT a trained model — a distribution recentering derived from unlabeled data.

Usage:
    python scripts/calibrate_micro_heuristic.py \
        --capture ~/super_poker_3/live_capture/micro_*.jsonl \
        --out artifacts/micro_calibration.json \
        --model-version v4-micro-cal-YYYYMMDD

Then stage:
    export POKER44_V4_CALIBRATION_PATH=$(pwd)/artifacts/micro_calibration.json
and restart the miner (between tournament windows).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from super_poker.micro_session_scorer import _select_heuristic  # noqa: E402

_hfn = _select_heuristic()  # respects POKER44_V4_HEURISTIC (default | atypicality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", nargs="+", required=True,
                    help="micro_*.jsonl file(s) or globs")
    ap.add_argument("--out", default="artifacts/micro_calibration.json")
    ap.add_argument("--model-version", default="v4-micro-cal")
    ap.add_argument("--max-ref", type=int, default=20000)
    ap.add_argument("--target-rate", type=float, default=0.5,
                    help="fraction of items to score >=0.5 (set to the inferred "
                         "true bot rate to match the eval base rate; 0.5=center)")
    args = ap.parse_args()

    files = []
    for pat in args.capture:
        files.extend(glob.glob(os.path.expanduser(pat)))
    if not files:
        raise SystemExit(f"no capture files matched: {args.capture}")

    raw, n_items = [], 0
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
                try:
                    raw.append(_hfn(it))
                    n_items += 1
                except Exception:
                    continue
    if not raw:
        raise SystemExit("no items found in the capture files")

    raw.sort()
    if len(raw) > args.max_ref:  # subsample evenly, preserve distribution shape
        step = len(raw) / args.max_ref
        raw = [raw[int(i * step)] for i in range(args.max_ref)]

    med = st.median(raw)
    frac_hi = sum(1 for s in raw if s >= 0.5) / len(raw)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"ref_scores": raw, "model_version": args.model_version,
                   "type": "empirical-cdf", "n_items": n_items,
                   "target_rate": args.target_rate}, fh)

    print(f"[cal] {n_items} items | ref size {len(raw)} | "
          f"raw median={med:.3f} | raw frac>=0.5={frac_hi:.3f}")
    print(f"[cal] after calibration: ~{args.target_rate*100:.0f}% of items will "
          f"score >=0.5 (target_rate={args.target_rate})")
    print(f"[cal] wrote {out}  version={args.model_version}")
    print(f"[next] export POKER44_V4_CALIBRATION_PATH={out.resolve()}  "
          "then restart between rounds")


if __name__ == "__main__":
    main()
