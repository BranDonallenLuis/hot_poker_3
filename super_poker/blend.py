"""Blend model: average of the XGBoost (feature) model and the sequence
(Set-Transformer) model. Combines two uncorrelated detectors (tree over
aggregate features + transformer over action order) into one raw score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from super_poker.features import chunk_features


class BlendModel:
    """Weighted average of an XGBoost feature model and a sequence model.

    Exposes ``predict_chunk_scores(chunks) -> list[float]`` returning RAW blended
    probabilities (the serving layer applies the live-anchored remap on top).
    Empty / action-less chunks are defaulted to 0.0 on the sequence side, which
    the transformer cannot process.
    """

    def __init__(self, xgb_model, xgb_feature_names, seq_wrapper, w_xgb: float = 0.5, w_seq: float = 0.5):
        self.xgb_model = xgb_model
        self.xgb_feature_names = list(xgb_feature_names)
        self.seq_wrapper = seq_wrapper
        self.w_xgb = float(w_xgb)
        self.w_seq = float(w_seq)

    def _xgb_raw(self, chunks):
        frame = pd.DataFrame([chunk_features(c) for c in chunks])
        frame = frame.reindex(columns=self.xgb_feature_names, fill_value=0.0).fillna(0.0)
        return self.xgb_model.predict_proba(frame.astype(float))[:, 1]

    def _seq_raw(self, chunks):
        raw = np.zeros(len(chunks), dtype=float)
        idx = [
            i for i, c in enumerate(chunks)
            if c and any(isinstance(h, dict) and h.get("actions") for h in c)
        ]
        if idx:
            vals = np.asarray(self.seq_wrapper.predict_chunk_scores([chunks[i] for i in idx]), dtype=float)
            for j, i in enumerate(idx):
                raw[i] = float(vals[j])
        return raw

    def predict_chunk_scores(self, chunks):
        if not chunks:
            return []
        blended = self.w_xgb * self._xgb_raw(chunks) + self.w_seq * self._seq_raw(chunks)
        return [float(x) for x in blended]
