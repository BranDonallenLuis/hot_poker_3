"""Artifact loading and independent chunk inference."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from super_poker.features import chunk_features


class SuperPokerModel:
    def __init__(self, artifact_path: str | Path):
        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        feature_names = artifact.get("feature_names")
        self.feature_names = list(feature_names) if feature_names else None
        self.threshold = float(artifact["threshold"])
        self.metadata = dict(artifact.get("metadata") or {})
        # Sequence/Set-Transformer artifacts consume RAW chunk payloads (no feature
        # frame). Detect them and route to the wrapper's chunk-level predictor.
        self.is_sequence = self.feature_names is None or str(self.metadata.get("type")) == "sequence"

    def _raw_scores(self, chunks: list[list[dict]]) -> np.ndarray:
        if self.is_sequence:
            # The transformer can't process a chunk with no real actions (all-padding
            # input errors). Score only chunks that have >=1 hand with actions; default
            # the rest to a low raw score. Real live chunks always qualify.
            raw = np.zeros(len(chunks), dtype=float)
            idx = [
                i for i, c in enumerate(chunks)
                if c and any(isinstance(h, dict) and h.get("actions") for h in c)
            ]
            if idx:
                vals = np.asarray(
                    self.model.predict_chunk_scores([chunks[i] for i in idx]), dtype=float
                )
                for j, i in enumerate(idx):
                    raw[i] = float(vals[j])
            return raw
        frame = pd.DataFrame([chunk_features(chunk) for chunk in chunks])
        frame = frame.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
        return self.model.predict_proba(frame.astype(float))[:, 1]

    @staticmethod
    def _remap(score: float, threshold: float) -> float:
        threshold = min(max(threshold, 1e-6), 1 - 1e-6)
        if score >= threshold:
            return 0.5 + 0.5 * (score - threshold) / (1 - threshold)
        return 0.5 * score / threshold

    def predict_chunk_scores(self, chunks: list[list[dict]]) -> list[float]:
        if not chunks:
            return []
        raw = self._raw_scores(chunks)
        scores = []
        for chunk, value in zip(chunks, raw):
            if not chunk:
                scores.append(0.1)
                continue
            score = self._remap(float(value), self.threshold)
            scores.append(round(min(1.0, max(0.0, score)) if math.isfinite(score) else 0.5, 6))
        return scores
