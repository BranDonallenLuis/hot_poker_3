"""Poker44 v3.0 session scorer. Uses a trained telemetry model if present,
else the launch heuristic (timing+mouse priors, no training data needed)."""
from __future__ import annotations
import os
from super_poker.session_features_v3 import session_features, heuristic_bot_score


class SessionScorer:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.feature_names = None
        self.model_version = "v3-launch-heuristic"
        path = model_path or os.getenv("POKER44_V3_MODEL_PATH", "")
        if path and os.path.exists(path):
            try:
                import joblib
                art = joblib.load(path)
                self.model = art["model"]
                self.feature_names = list(art["feature_names"])
                self.model_version = str(art.get("metadata", {}).get("model_version", "v3-model"))
            except Exception:
                self.model = None

    def score_sessions(self, sessions: list[dict]) -> list[float]:
        if not sessions:
            return []
        if self.model is not None and self.feature_names is not None:
            import numpy as np, pandas as pd
            X = pd.DataFrame([session_features(s) for s in sessions])
            X = X.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
            raw = self.model.predict_proba(X.astype(float))[:, 1]
            return [float(min(1.0, max(0.0, v))) for v in raw]
        return [heuristic_bot_score(s) for s in sessions]
