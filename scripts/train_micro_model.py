#!/usr/bin/env python3
"""Train the Poker44 v3.0 micro-session (schema-v4.1) model from labeled data.

Run this the moment the labeled "actionable feedback" dataset drops. It:
  1. Loads the labeled micro-sessions (robust to the exact wrapper/label format).
  2. Builds features via super_poker.micro_session_scorer.micro_features.
  3. Reports the SAME metrics the tournament dashboard shows (accuracy,
     precision, recall@FPR<=5%) plus AP/AUC, on a holdout.
  4. Fits a calibrated deployment model and saves an artifact that
     MicroSessionScorer loads directly via POKER44_V4_MODEL_PATH.
  5. Optionally scores your CAPTURED live items to sanity-check the output
     distribution against real tournament data.

Usage:
    python scripts/train_micro_model.py \
        --data path/to/labeled_microsessions.json \
        --out artifacts/micro_model.joblib \
        [--label-key is_bot] \
        [--live ~/super_poker_3/live_capture/micro_poker44_miner.jsonl] \
        [--model-version v4-micro-YYYYMMDD]

Then stage it:
    export POKER44_V4_MODEL_PATH=$(pwd)/artifacts/micro_model.joblib
and restart the miner (between tournament windows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from super_poker.micro_session_scorer import micro_features  # noqa: E402

# label may be under any of these; is_human is INVERTED.
_LABEL_KEYS = ("is_bot", "label", "bot", "is_bot_label", "target", "y", "class")
_INVERSE_KEYS = ("is_human", "human")
_PRESENCE_BOT_KEYS = ("bot_family",)  # non-empty => bot
_ITEM_KEYS = ("item", "micro_session", "session", "data", "payload")
_LIST_KEYS = ("items", "sessions", "records", "examples", "data", "samples")


def _to_bin(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if v >= 0.5 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "bot", "yes", "y", "positive"}:
            return 1
        if s in {"0", "false", "human", "no", "n", "negative"}:
            return 0
    return None


def _find_label(rec, label_key):
    keys = (label_key,) if label_key else _LABEL_KEYS
    for container in (rec, rec.get("metadata"), rec.get("labels"), rec.get("label")):
        if not isinstance(container, dict):
            continue
        for k in keys:
            if k and k in container:
                b = _to_bin(container[k])
                if b is not None:
                    return b
        if not label_key:
            for k in _INVERSE_KEYS:
                if k in container:
                    b = _to_bin(container[k])
                    if b is not None:
                        return 1 - b
            for k in _PRESENCE_BOT_KEYS:
                if k in container and container[k]:
                    return 1
    return None


def _find_item(rec):
    if "decisions" in rec or "item_id" in rec:
        return rec
    for k in _ITEM_KEYS:
        if isinstance(rec.get(k), dict):
            return rec[k]
    return None


def _iter_records(raw):
    if isinstance(raw, list):
        yield from (r for r in raw if isinstance(r, dict))
        return
    if isinstance(raw, dict):
        for k in _LIST_KEYS:
            if isinstance(raw.get(k), list):
                yield from (r for r in raw[k] if isinstance(r, dict))
                return
        vals = [v for v in raw.values() if isinstance(v, dict)]
        if vals:
            yield from vals
            return
    raise SystemExit(f"Could not find a record list; top-level={type(raw).__name__}")


def load_labeled(path, label_key):
    raw = json.loads(Path(path).read_text())
    items, labels, skipped, keys_seen = [], [], 0, set()
    for rec in _iter_records(raw):
        keys_seen.update(rec.keys())
        it = _find_item(rec)
        lab = _find_label(rec, label_key)
        if it is None or lab is None:
            skipped += 1
            continue
        items.append(it)
        labels.append(lab)
    if not items:
        raise SystemExit(
            "Parsed the data but found 0 labeled items.\n"
            f"Keys seen: {sorted(keys_seen)}\n"
            "Re-run with --label-key <the bot-label field>."
        )
    print(f"[load] {len(items)} labeled items "
          f"({sum(labels)} bots / {len(labels) - sum(labels)} humans), "
          f"{skipped} skipped")
    return items, labels


def evaluate(y_true, p):
    import numpy as np
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 roc_curve, accuracy_score, precision_score)
    ap = float(average_precision_score(y_true, p))
    auc = float(roc_auc_score(y_true, p)) if len(set(y_true)) > 1 else float("nan")
    pred = (np.asarray(p) >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, pred))
    prec = float(precision_score(y_true, pred, zero_division=0))
    fpr, tpr, _ = roc_curve(y_true, p)
    mask = fpr <= 0.05
    recall_at5 = float(tpr[mask].max()) if mask.any() else 0.0
    return {"ap": ap, "auc": auc, "accuracy": acc, "precision": prec,
            "recall_at_fpr5": recall_at5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts/micro_model.joblib")
    ap.add_argument("--label-key", default="")
    ap.add_argument("--live", default="", help="captured micro_*.jsonl for a distribution sanity-check")
    ap.add_argument("--model-version", default="v4-micro")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    items, labels = load_labeled(args.data, args.label_key or None)
    X = pd.DataFrame([micro_features(it) for it in items]).fillna(0.0)
    X = X.reindex(sorted(X.columns), axis=1)
    y = np.array(labels)
    feats = list(X.columns)
    print(f"[feat] {len(feats)} features over {len(items)} items")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    base = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         n_jobs=1, random_state=args.seed)
    base.fit(Xtr, ytr)
    m = evaluate(yte, base.predict_proba(Xte)[:, 1])
    print(f"[eval] holdout  AP={m['ap']:.4f}  recall@FPR5%={m['recall_at_fpr5']:.4f}  "
          f"accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  AUC={m['auc']:.4f}")
    print("[eval] (compare accuracy/precision/recall@FPR5 to the dashboard's ~0.440/0.526/0.092)")

    # calibrated deployment model on ALL data
    n_pos = int(y.sum())
    method = "isotonic" if n_pos >= 200 else "sigmoid"
    cv = min(5, max(2, n_pos))
    print(f"[fit] calibrating final model (method={method}, cv={cv})")
    final = CalibratedClassifierCV(
        XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      n_jobs=1, random_state=args.seed),
        method=method, cv=cv)
    final.fit(X, y)

    if args.live:
        rows = []
        for line in Path(args.live).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                for it in json.loads(line).get("items", []):
                    rows.append(micro_features(it))
            except Exception:
                continue
        if rows:
            XL = pd.DataFrame(rows).reindex(columns=feats, fill_value=0.0).fillna(0.0)
            pl = final.predict_proba(XL.astype(float))[:, 1]
            print(f"[live] scored {len(pl)} captured items: "
                  f"mean={pl.mean():.3f} frac>=0.5={float((pl >= 0.5).mean()):.3f} "
                  f"(dashboard heuristic put ~0.28 above 0.5)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_names": feats,
                 "metadata": {"model_version": args.model_version,
                              "n_train": int(len(y)), "n_bots": n_pos,
                              "holdout_metrics": m, "calibration": method}}, out)
    print(f"[save] wrote {out}  version={args.model_version}")
    print(f"[next] export POKER44_V4_MODEL_PATH={out.resolve()}  then restart between rounds")


if __name__ == "__main__":
    main()
