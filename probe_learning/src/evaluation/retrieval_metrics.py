"""Stable-rank AP and frozen-threshold F1 used by paper evaluation.

fit_training_threshold receives the caller-selected fitting partition;
the Main17 paper protocol supplies original VQA Validation labels.
"""
import numpy as np


def aligned_inputs(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("Labels/scores must be aligned one-dimensional binary/score vectors")
    if not np.isfinite(s).all():
        raise ValueError("Scores must be finite")
    return y.astype(np.uint8, copy=False), s


def fit_training_threshold(labels, scores):
    """Maximize F1 over complete training-score groups, ties: highest threshold.

    Uses only its training inputs. The empty-prediction boundary is included;
    a one-class all-negative fixture therefore predicts no positives. With any
    positive training rows an attained positive F1 always beats that boundary.
    """
    y, s = aligned_inputs(labels, scores)
    if not len(s):
        raise ValueError("Threshold fitting requires nonempty original training rows")
    order = np.argsort(-s, kind="stable")
    sorted_s, sorted_y = s[order], y[order]
    ends = np.flatnonzero(np.r_[sorted_s[1:] != sorted_s[:-1], True])
    cumulative = np.cumsum(sorted_y, dtype=np.int64)
    positives = int(y.sum())
    best_tp, best_count = 0, 0
    threshold = float(np.nextafter(sorted_s[0], np.inf))
    if not np.isfinite(threshold):
        raise ValueError("No finite empty-prediction threshold can represent these scores")
    # Integer cross-products resolve exact F1 ties without floating ambiguity.
    for end in ends:
        tp, predicted = int(cumulative[end]), int(end + 1)
        if tp * (positives + best_count) > best_tp * (positives + predicted):
            best_tp, best_count, threshold = tp, predicted, float(sorted_s[end])
    denominator = positives + best_count
    return {"threshold": threshold, "trainF1": 2 * best_tp / denominator if denominator else 0.0,
            "trainCount": len(y), "trainPositiveCount": positives,
            "trainNegativeCount": len(y) - positives, "trainTP": best_tp,
            "trainFP": best_count - best_tp, "trainFN": positives - best_tp,
            "trainThresholdGroups": len(ends)}


def evaluation_metrics(labels, scores, threshold):
    """Stable-row AP; F1 uses the supplied frozen training threshold only."""
    y, s = aligned_inputs(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    order = np.argsort(-s, kind="stable")
    ordered = y[order].astype(np.float64)
    positives = int(y.sum())
    precision = np.cumsum(ordered) / np.arange(1, len(y) + 1)
    ap = float(np.sum(precision * ordered) / positives) if positives else None
    predicted = s >= threshold
    tp = int(np.count_nonzero(predicted & (y == 1)))
    fp = int(np.count_nonzero(predicted & (y == 0)))
    fn = positives - tp
    denominator = 2 * tp + fp + fn
    return {"Ap": ap, "F1": 2 * tp / denominator if denominator else 0.0,
            "Count": len(y), "PositiveCount": positives, "NegativeCount": len(y) - positives,
            "TP": tp, "FP": fp, "FN": fn, "TN": len(y) - positives - fp,
            "PredictedPositiveCount": int(predicted.sum()),
            "TiedRows": len(y) - int(np.unique(s).size)}
