#!/usr/bin/env python3
"""Re-key a trained super_poker model artifact to a new identity.

Deploys super's proven 194-feature model onto hot WITHOUT reporting super's
model_name/version in hot's manifest (which would be a consistency mismatch
and an integrity risk). ONLY the metadata labels change — the booster,
feature_names, and threshold are untouched, so scores are byte-for-byte
identical to the source model.

Usage:
    python scripts/rekey_artifact.py \
        --src   artifacts/super_poker_194.joblib \
        --out   artifacts/hot_poker_3_194.joblib \
        --model-name hot-poker-3 \
        --model-version 20260731-hot194swap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-version", required=True)
    args = ap.parse_args()

    art = joblib.load(args.src)
    if not isinstance(art, dict):
        raise SystemExit(f"unexpected artifact type: {type(art).__name__}")

    meta = dict(art.get("metadata") or {})
    before = (meta.get("model_name"), meta.get("model_version"))
    meta["model_name"] = args.model_name
    meta["model_version"] = args.model_version
    # record provenance so we never lose track of what this really is
    meta["rekeyed_from"] = {"model_name": before[0], "model_version": before[1]}
    art["metadata"] = meta

    fn = art.get("feature_names")
    thr = art.get("threshold")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, args.out)

    print(f"[rekey] model_name/version {before} -> "
          f"({args.model_name}, {args.model_version})")
    print(f"[rekey] n_features={len(fn) if fn else 'None(sequence!)'}  "
          f"threshold={thr}  type={meta.get('type', 'xgboost')}")
    print(f"[rekey] wrote {args.out}")
    if not fn:
        print("[rekey] ⚠️  feature_names is None — this is a SEQUENCE artifact, "
              "not the 194. Check you passed super's 194 model.")


if __name__ == "__main__":
    main()
