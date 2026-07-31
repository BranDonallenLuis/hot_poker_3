#!/usr/bin/env python3
"""Train the Poker44 v3.0 telemetry session model from the v2 benchmark.

Run this the moment the v2 benchmark drops (schemaVersion changes from
shadow-training-v1). It:
  1. Loads the labeled benchmark (robust to the exact wrapper format).
  2. Builds telemetry features via super_poker.session_features_v3.
  3. Reports honest holdout metrics (AP, recall@FPR<=5%, AUC).
  4. Fits a calibrated deployment model and saves an artifact that
     super_poker.session_scorer.SessionScorer can load directly.
  5. Optionally KS-checks benchmark features vs captured live sessions to
     catch the benchmark->live domain shift BEFORE trusting the model.

Usage:
    python scripts/train_session_model.py \
        --benchmark path/to/v2_benchmark.json \
        --out artifacts/session_model.joblib \
        [--live path/to/live_sessions.jsonl] \
        [--label-key is_bot] \
        [--model-version v3-telemetry-YYYYMMDD]

Then stage it:
    export POKER44_V3_MODEL_PATH=$(pwd)/artifacts/session_model.joblib
and restart the miner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `super_poker` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from super_poker.session_features_v3 import session_features  # noqa: E402

# Candidate keys we look for a boolean/int bot label under, in priority order.
_LABEL_KEYS = (
    "is_bot", "label", "bot", "is_bot_label", "subject_is_bot",
    "botLabel", "is_bot_flag", "y", "target", "class",
)
# Candidate keys the actual session dict may be nested under.
_SESSION_KEYS = ("session", "subject_session", "data", "payload")
# Candidate keys a top-level container may hold the record list under.
_LIST_KEYS = ("sessions", "records", "examples", "data", "items", "samples")


def _to_binary(value) -> int | None:
    """Coerce a label value to 0/1, or None if it isn't a usable label."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "bot", "yes", "y", "positive"}:
            return 1
        if v in {"0", "false", "human", "no", "n", "negative"}:
            return 0
    return None


def _find_label(record: dict, label_key: str | None) -> int | None:
    keys = (label_key,) if label_key else _LABEL_KEYS
    # direct on the record
    for k in keys:
        if k and k in record:
            b = _to_binary(record[k])
            if b is not None:
                return b
    # nested under a metadata/label container
    for container in ("metadata", "labels", "label", "meta", "ground_truth"):
        sub = record.get(container)
        if isinstance(sub, dict):
            for k in keys:
                if k and k in sub:
                    b = _to_binary(sub[k])
                    if b is not None:
                        return b
    return None


def _find_session(record: dict) -> dict | None:
    # record is itself a session
    if "hands" in record or "telemetry" in record or "session_id" in record:
        return record
    for k in _SESSION_KEYS:
        sub = record.get(k)
        if isinstance(sub, dict):
            return sub
    return None


def _iter_records(raw):
    """Yield dict records from a benchmark that may be a list or a container."""
    if isinstance(raw, list):
        yield from (r for r in raw if isinstance(r, dict))
        return
    if isinstance(raw, dict):
        for k in _LIST_KEYS:
            if isinstance(raw.get(k), list):
                yield from (r for r in raw[k] if isinstance(r, dict))
                return
        # a dict keyed by session_id -> record
        vals = [v for v in raw.values() if isinstance(v, dict)]
        if vals:
            yield from vals
            return
    raise SystemExit(
        "Could not locate a list of records in the benchmark. Top-level type="
        f"{type(raw).__name__}. Inspect the file and pass the right structure."
    )


def load_benchmark(path: str, label_key: str | None):
    raw = json.loads(Path(path).read_text())
    sessions, labels, skipped = [], [], 0
    seen_keys: set = set()
    for rec in _iter_records(raw):
        seen_keys.update(rec.keys())
        sess = _find_session(rec)
        lab = _find_label(rec, label_key)
        if sess is None or lab is None:
            skipped += 1
            continue
        sessions.append(sess)
        labels.append(lab)
    if not sessions:
        raise SystemExit(
            "Parsed the benchmark but found 0 labeled sessions.\n"
            f"Keys seen on records: {sorted(seen_keys)}\n"
            "Re-run with --label-key <the bot-label field> (and check the "
            "session lives at the record root or under a 'session' key)."
        )
    print(f"[load] {len(sessions)} labeled sessions "
          f"({sum(labels)} bots / {len(labels) - sum(labels)} humans), "
          f"{skipped} records skipped (no session/label).")
    return sessions, labels


def build_matrix(sessions):
    import pandas as pd
    X = pd.DataFrame([session_features(s) for s in sessions]).fillna(0.0)
    X = X.reindex(sorted(X.columns), axis=1)  # deterministic column order
    return X


def evaluate(y_true, p):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    ap = average_precision_score(y_true, p)
    auc = roc_auc_score(y_true, p) if len(set(y_true)) > 1 else float("nan")
    fpr, tpr, thr = roc_curve(y_true, p)
    mask = fpr <= 0.05
    recall_at_5 = float(tpr[mask].max()) if mask.any() else 0.0
    # threshold that first satisfies FPR<=5% (useful operating point)
    op_thr = float(thr[mask][tpr[mask].argmax()]) if mask.any() else 0.5
    return {"ap": float(ap), "auc": float(auc),
            "recall_at_fpr5": recall_at_5, "op_threshold_fpr5": op_thr}


def ks_vs_live(X_bench, live_path):
    """Warn about benchmark->live drift per feature (the v2.0 killer)."""
    from scipy.stats import ks_2samp
    import pandas as pd
    rows = []
    for line in Path(live_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        sess = _find_session(obj) if isinstance(obj, dict) else None
        if sess is not None:
            rows.append(session_features(sess))
    if not rows:
        print("[ks] no usable live sessions found; skipping drift check.")
        return
    X_live = pd.DataFrame(rows).reindex(columns=X_bench.columns, fill_value=0.0)
    print(f"[ks] comparing {len(X_bench)} benchmark vs {len(X_live)} live sessions")
    flagged = []
    for c in X_bench.columns:
        b, l = X_bench[c].values, X_live[c].values
        if len(set(b)) <= 1 and len(set(l)) <= 1:
            continue
        stat, _ = ks_2samp(b, l)
        if stat > 0.2:
            flagged.append((c, stat))
    flagged.sort(key=lambda t: -t[1])
    if flagged:
        print(f"[ks] ⚠️  {len(flagged)} features shifted (KS>0.2) benchmark vs live:")
        for c, s in flagged[:15]:
            print(f"       {c:32s} KS={s:.3f}")
        print("[ks] These features are unreliable live. Consider dropping the "
              "worst offenders and retraining before trusting the model.")
    else:
        print("[ks] ✅ no severe feature drift (all KS<=0.2). Benchmark aligns with live.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", default="artifacts/session_model.joblib")
    ap.add_argument("--live", default="", help="jsonl of captured live v3 sessions for KS check")
    ap.add_argument("--label-key", default="", help="explicit bot-label field name")
    ap.add_argument("--model-version", default="v3-telemetry")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import joblib
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    sessions, labels = load_benchmark(args.benchmark, args.label_key or None)
    X = build_matrix(sessions)
    y = np.array(labels)
    feature_names = list(X.columns)
    print(f"[feat] {len(feature_names)} features")

    # --- honest holdout metrics ---
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    base = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        n_jobs=1, random_state=args.seed,
    )
    base.fit(Xtr, ytr)
    metrics = evaluate(yte, base.predict_proba(Xte)[:, 1])
    print(f"[eval] holdout AP={metrics['ap']:.4f}  "
          f"recall@FPR5%={metrics['recall_at_fpr5']:.4f}  "
          f"AUC={metrics['auc']:.4f}  op_thr={metrics['op_threshold_fpr5']:.4f}")
    if metrics["ap"] > 0.98:
        print("[eval] ⚠️  AP>0.98 — suspiciously high. Check the benchmark bots "
              "aren't trivially separable (synthetic-overfit risk). Verify live.")

    # --- calibrated deployment model on ALL data ---
    n_pos = int(y.sum())
    method = "isotonic" if n_pos >= 200 else "sigmoid"
    cv = min(5, n_pos) if n_pos >= 2 else 2
    print(f"[fit] calibrating final model (method={method}, cv={cv}) on all data")
    final = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=1, random_state=args.seed,
        ),
        method=method, cv=cv,
    )
    final.fit(X, y)

    # sanity: does the calibrated model put bots above 0.5? (zero-gate safety)
    p_all = final.predict_proba(X)[:, 1]
    frac_bots_above_05 = float((p_all[y == 1] >= 0.5).mean()) if n_pos else 0.0
    print(f"[fit] calibrated: {frac_bots_above_05:.3f} of true bots score >=0.5 "
          f"(zero-gate needs >=1 true bot >=0.5)")

    # --- optional drift check ---
    if args.live:
        ks_vs_live(X, args.live)

    # --- save artifact in the exact shape SessionScorer expects ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final,
        "feature_names": feature_names,
        "metadata": {
            "model_version": args.model_version,
            "n_train": int(len(y)),
            "n_bots": n_pos,
            "holdout_metrics": metrics,
            "calibration": method,
            "frac_bots_above_0.5": frac_bots_above_05,
        },
    }
    joblib.dump(artifact, out)
    print(f"[save] wrote {out}  version={args.model_version}")
    print(f"[next] export POKER44_V3_MODEL_PATH={out.resolve()}  then restart the miner")


if __name__ == "__main__":
    main()
