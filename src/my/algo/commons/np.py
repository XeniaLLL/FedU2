import numpy as np
import scipy.optimize
from sklearn.metrics import accuracy_score


def best_mapping(labels_true: np.ndarray, labels_pred: np.ndarray):
    """Find best mapping between ground truth labels and cluster labels."""
    D = max(max(labels_true), max(labels_pred)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(labels_pred)):
        w[labels_pred[i], labels_true[i]] += 1
    old_pred, new_pred = scipy.optimize.linear_sum_assignment(w.max() - w)
    label_map = dict(zip(old_pred, new_pred))
    labels_pred = np.array([label_map[x] for x in labels_pred])
    return labels_true, labels_pred


def cluster_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = best_mapping(y_true, y_pred)
    return float(accuracy_score(y_true, y_pred))
