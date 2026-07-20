"""Train Hot Poker 3 with chronological walk-forward evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from poker44.validator.payload_view import prepare_hand_for_miner
from super_poker.dataset import Example, load_examples
from super_poker.features import chunk_features
from super_poker.scoring import metrics

DEFAULT_DATA = Path("../Poker44-subnet/data/raw")
DEFAULT_ARTIFACT = Path("artifacts/hot_poker_3.joblib")
LIVE_SCORE_HISTORY = Path("config/live_scores.json")
LIVE_CHUNK_RANGE = (90, 105)
LIVE_CHUNKS_PER_DATE_LABEL = 3


def make_model(seed: int = 44) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
    )


def matrix(examples: list[Example], columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    # Train on exactly what validators expose to miners. Raw benchmark hands
    # contain outcomes, cards, original seats, and amounts that are removed or
    # canonicalized before a DetectionSynapse is sent.
    visible_chunks = [
        [prepare_hand_for_miner(hand) for hand in example.hands]
        for example in examples
    ]
    frame = pd.DataFrame([chunk_features(chunk) for chunk in visible_chunks]).fillna(0.0)
    if columns is None:
        columns = sorted(frame.columns)
    return frame.reindex(columns=columns, fill_value=0.0).astype(float), columns


def augment_live_size_chunks(examples: list[Example], seed: int = 481) -> list[Example]:
    """Pool same-date, same-label examples into validator-sized training groups."""
    rng = random.Random(seed)
    grouped: dict[tuple[str, int], list[Example]] = {}
    for example in examples:
        grouped.setdefault((example.source_date, example.label), []).append(example)

    augmented: list[Example] = []
    for (source_date, label), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        for index in range(LIVE_CHUNKS_PER_DATE_LABEL):
            target = rng.randint(*LIVE_CHUNK_RANGE)
            order = rng.sample(group, len(group))
            hands: list[dict] = []
            used = 0
            for example in order:
                hands.extend(example.hands)
                used += 1
                if len(hands) >= target:
                    break
            if used >= 2 and len(hands) >= LIVE_CHUNK_RANGE[0]:
                augmented.append(Example(
                    hands=hands[:target],
                    label=label,
                    source_date=source_date,
                    split="augmented-live-size",
                    chunk_hash=f"aug:{source_date}:{label}:{index}",
                ))
    return augmented


def live_score_context(path: Path = LIVE_SCORE_HISTORY) -> dict | None:
    if not path.is_file():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    return records[-1] if records else None


def threshold_for_fpr(human_scores: np.ndarray, target_fpr: float) -> float:
    if not len(human_scores):
        return 0.5
    return float(np.quantile(human_scores, 1.0 - target_fpr))


def remap_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    threshold = min(max(float(threshold), 1e-6), 1 - 1e-6)
    return np.clip(
        np.where(
            scores >= threshold,
            0.5 + 0.5 * (scores - threshold) / (1.0 - threshold),
            0.5 * scores / threshold,
        ),
        0.0,
        1.0,
    )


def train(data_dir: Path, artifact_path: Path, *, folds: int = 5, target_fpr: float = 0.035) -> dict:
    examples = load_examples(data_dir)
    dates = sorted({example.source_date for example in examples})
    if len(dates) < folds + 2:
        required = folds + 2
        raise ValueError(
            f"Training requires at least {required} distinct release dates, but "
            f"found {len(dates)} under {data_dir.resolve()}. Download the full "
            "history with automation --backfill, or select the existing "
            "historical data directory."
        )
    augmented = augment_live_size_chunks(examples)
    training_examples = examples + augmented
    all_frame, columns = matrix(training_examples)
    labels = np.asarray([example.label for example in training_examples], dtype=int)
    date_array = np.asarray([example.source_date for example in training_examples])
    real_count = len(examples)
    real_labels = labels[:real_count]
    real_dates = date_array[:real_count]
    oof = np.full(len(examples), np.nan)
    fold_results = []

    for fold_index, test_date in enumerate(dates[-folds:]):
        train_mask = date_array < test_date
        test_mask = real_dates == test_date
        if train_mask.sum() < 60 or len(set(labels[train_mask])) < 2:
            continue
        model = make_model(44 + fold_index)
        model.fit(all_frame.loc[train_mask], labels[train_mask])
        raw_test = model.predict_proba(all_frame.iloc[:real_count].loc[test_mask])[:, 1]

        earlier_dates = sorted(set(date_array[train_mask]))
        calibration_date = earlier_dates[-1]
        inner_fit = date_array < calibration_date
        inner_cal = real_dates == calibration_date
        inner_model = make_model(144 + fold_index)
        inner_model.fit(all_frame.loc[inner_fit], labels[inner_fit])
        calibration_scores = inner_model.predict_proba(all_frame.iloc[:real_count].loc[inner_cal])[:, 1]
        threshold = threshold_for_fpr(calibration_scores[real_labels[inner_cal] == 0], target_fpr)
        mapped = remap_threshold(raw_test, threshold)
        oof[test_mask] = mapped
        fold_results.append({"date": test_date, "threshold": threshold, **metrics(real_labels[test_mask], mapped)})

    valid = np.isfinite(oof)
    if not valid.any():
        raise RuntimeError("Walk-forward evaluation produced no predictions")
    overall = metrics(real_labels[valid], oof[valid])

    deployment_calibration_date = dates[-1]
    deployment_fit = date_array < deployment_calibration_date
    calibration_mask = real_dates == deployment_calibration_date
    calibration_model = make_model(244)
    calibration_model.fit(all_frame.loc[deployment_fit], labels[deployment_fit])
    calibration_scores = calibration_model.predict_proba(all_frame.iloc[:real_count].loc[calibration_mask])[:, 1]
    deployment_threshold = threshold_for_fpr(
        calibration_scores[real_labels[calibration_mask] == 0], target_fpr
    )

    final_model = make_model(344)
    final_model.fit(all_frame, labels)
    metadata = {
        "model_name": "hot-poker-3",
        "model_version": time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
        "framework": "xgboost+validator-stable-signatures+live-size-augmentation",
        "feature_version": "super-poker-3.v4-r3-live-gap",
        "example_count": len(examples),
        "augmented_example_count": len(augmented),
        "augmentation": {
            "kind": "same-date-same-label-live-size-pooling",
            "hand_count_range": list(LIVE_CHUNK_RANGE),
            "examples_per_date_label": LIVE_CHUNKS_PER_DATE_LABEL,
        },
        "release_dates": dates,
        "walk_forward_dates": [result["date"] for result in fold_results],
        "walk_forward": fold_results,
        "walk_forward_overall": overall,
        "target_fpr": target_fpr,
        "deployment_threshold": deployment_threshold,
        "calibration_release": deployment_calibration_date,
        "feature_count": len(columns),
        "feature_schema_sha256": hashlib.sha256("\n".join(columns).encode()).hexdigest(),
        "live_score_context": live_score_context(),
        "change_reason": (
            "R3 live reward 0.481 exposed an offline/live transfer gap; add exact validator "
            "amount buckets, hero-independent signatures, and live-size training groups."
        ),
        "training_data": (
            "Poker44 public benchmark only, projected through the validator-visible "
            "payload sanitizer; no validator-private labels"
        ),
    }
    artifact = {"model": final_model, "feature_names": columns, "threshold": deployment_threshold, "metadata": metadata}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(artifact_path)
    artifact_path.with_suffix(".metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.035)
    args = parser.parse_args()
    metadata = train(args.data_dir, args.artifact, folds=args.folds, target_fpr=args.target_fpr)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
