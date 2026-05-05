import numpy as np
from sklearn.metrics import average_precision_score, classification_report


def evaluate_model(model, X_test, y_test, top_percentile=99.5):
    """
    Evaluate fraud model using PR-AUC and top-k threshold metrics.
    """

    y_scores = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_scores)

    threshold = np.percentile(y_scores, top_percentile)
    top_k = y_scores >= threshold

    precision_at_k = y_test[top_k].mean()

    recall_at_k = (y_test[top_k] == 1).sum() / (y_test == 1).sum()
    baseline_fraud_rate = y_test.mean()

    report = classification_report(
        y_test,
        top_k,
        digits=4,
        output_dict=False
    )

    metrics = {
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "precision_at_k": precision_at_k,
        "classification_report": report,
        "recall_at_k": recall_at_k,
        "baseline_fraud_rate": baseline_fraud_rate
    }

    return metrics, y_scores, top_k