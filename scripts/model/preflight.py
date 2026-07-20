#!/usr/bin/env python3
"""Refuse a Hot Poker 3 deployment when source and artifact are incompatible."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from super_poker.features import chunk_features
from super_poker.inference import SuperPokerModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/hot_poker_3.joblib"))
    parser.add_argument("--expected-model", default="hot-poker-3")
    args = parser.parse_args()

    model = SuperPokerModel(args.artifact)
    runtime = set(chunk_features([]))
    required = set(model.feature_names)
    missing = sorted(required - runtime)
    extra = sorted(runtime - required)
    scores = model.predict_chunk_scores([[], [{}]])
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    sha256 = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    report = {
        "ok": not missing and not extra
        and model.metadata.get("model_name") == args.expected_model
        and len(scores) == 2
        and all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores),
        "model_name": model.metadata.get("model_name"),
        "model_version": model.metadata.get("model_version"),
        "git_commit": commit,
        "artifact_sha256": sha256,
        "artifact_features": len(required),
        "runtime_features": len(runtime),
        "missing_features": missing,
        "extra_features": extra,
        "smoke_scores": scores,
    }
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit("Hot Poker 3 preflight failed; refusing deployment")


if __name__ == "__main__":
    main()
