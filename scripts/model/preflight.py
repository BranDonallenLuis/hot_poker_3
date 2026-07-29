#!/usr/bin/env python3
"""Refuse a Hot Poker 3 deployment when source and artifact are incompatible."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_poker.features import chunk_features
from super_poker.inference import SuperPokerModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/hot_poker_3.joblib"))
    parser.add_argument("--expected-model")
    args = parser.parse_args()

    model = SuperPokerModel(args.artifact)
    # Sequence/Set-Transformer artifacts consume raw chunks and have no feature
    # frame, so the tabular feature-compatibility check does not apply to them.
    is_sequence = getattr(model, "is_sequence", False) or model.feature_names is None
    runtime = set(chunk_features([]))
    required = set(model.feature_names) if model.feature_names else set()
    missing = [] if is_sequence else sorted(required - runtime)
    extra = [] if is_sequence else sorted(runtime - required)
    scores = model.predict_chunk_scores([[], [{}]])
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    sha256 = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    report = {
        "ok": not missing
        and (
            not args.expected_model
            or model.metadata.get("model_name") == args.expected_model
        )
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
        "selected_feature_subset": bool(extra),
        "smoke_scores": scores,
    }
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit("Hot Poker 3 preflight failed; refusing deployment")


if __name__ == "__main__":
    main()
