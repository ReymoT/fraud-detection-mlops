import numpy as np
import pandas as pd
from src.evaluate import evaluate_model


class DummyModel:
    def predict_proba(self, X):
        return np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ])


def test_evaluate_model_returns_metrics():
    X_test = pd.DataFrame({"x": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 0, 1])

    metrics, y_scores, top_k = evaluate_model(
        DummyModel(),
        X_test,
        y_test,
        top_percentile=50
    )

    assert "pr_auc" in metrics
    assert "threshold" in metrics
    assert "precision_at_k" in metrics
    assert len(y_scores) == 4
    assert len(top_k) == 4