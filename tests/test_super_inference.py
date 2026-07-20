import joblib
import numpy as np

from super_poker.inference import SuperPokerModel


class FixedModel:
    def predict_proba(self, frame):
        return np.asarray([[0.2, 0.8] for _ in range(len(frame))])


def test_inference_returns_one_bounded_score_per_chunk(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"model": FixedModel(), "feature_names": ["hand_count"], "threshold": 0.7}, path)
    scores = SuperPokerModel(path).predict_chunk_scores([[], [{}]])
    assert len(scores) == 2
    assert all(0.5 < score <= 1.0 for score in scores)
