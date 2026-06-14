import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def probability_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    scores = estimator.decision_function(X)
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn) if mx > mn else np.full_like(scores, 0.5, dtype=float)


def metrics_at_threshold(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Macro_Precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, pred, average="macro"),
        "Macro_F1": f1_score(y_true, pred, average="macro"),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
    }, pred
