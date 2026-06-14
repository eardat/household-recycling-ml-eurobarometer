import numpy as np
from sklearn.metrics import f1_score


def choose_threshold(y_true, y_prob, low=0.30, high=0.70, step=0.01):
    thresholds = np.round(np.arange(low, high + step / 2, step), 2)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), average="macro") for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])
