"""Post-probe score and rank fusion utilities.

The formal recipes combine complete method score sets after attribute probes
have been trained.  RRF averages ``1/(k+rank)`` with k=60; z-score fusion
averages standardized member scores; and soft-gate fusion min-max normalizes
the averaged attribute scores before a soft conjunction.  These operations do
not train probes or alter their cached scores.  The module also contains
validation-driven fusion helpers used by older analyses, separate from the
fixed recipes in ``../probe_suite.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class FusionResult:
    name: str
    scores: dict[str, np.ndarray]
    weights: dict[str, dict[str, float]]
    selected: dict[str, str]


def average_precision(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or s.size == 0 or int(y.sum()) <= 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    pos = int(y_sorted.sum())
    if pos <= 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, y_sorted.size + 1, dtype=np.float64)
    precision = tp / ranks
    return float((precision * y_sorted).sum() / pos)


def rank_metrics(y_true, scores, topks=(50, 100)) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    out = {"ap": average_precision(y, s)}
    if y.size == 0:
        for k in topks:
            out[f"p@{k}"] = float("nan")
            out[f"r@{k}"] = float("nan")
        return out
    order = np.argsort(-s, kind="mergesort")
    pos = max(int(y.sum()), 1)
    for k in topks:
        kk = min(int(k), y.size)
        hit = int(y[order[:kk]].sum()) if kk > 0 else 0
        out[f"p@{k}"] = float(hit / kk) if kk > 0 else float("nan")
        out[f"r@{k}"] = float(hit / pos)
    return out


def precision_at_k(y_true, scores, k: int = 50) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or s.size == 0:
        return float("nan")
    kk = min(int(k), y.size)
    if kk <= 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    return float(y[order[:kk]].sum() / kk)


def separation_score(y_true, scores) -> float:
    """Bounded positive-vs-negative separation from z-scored validation scores."""
    y = np.asarray(y_true, dtype=np.int32)
    s = zscore(scores)
    if y.size == 0 or int(y.sum()) == 0 or int((1 - y).sum()) == 0:
        return float("nan")
    diff = float(np.mean(s[y == 1]) - np.mean(s[y == 0]))
    return float(1.0 / (1.0 + np.exp(-diff)))


def reliability_score(y_true, scores, topk: int = 50) -> float:
    """Task-local probe reliability used for fusion weights.

    AP captures full-ranking quality, P@K captures retrieval-head precision,
    and separation captures whether the score scale cleanly separates positives
    from negatives on the current task's validation split.
    """
    ap = average_precision(y_true, scores)
    p_at_k = precision_at_k(y_true, scores, topk)
    sep = separation_score(y_true, scores)
    parts = []
    weights = []
    if np.isfinite(ap):
        parts.append(ap)
        weights.append(0.55)
    if np.isfinite(p_at_k):
        parts.append(p_at_k)
        weights.append(0.25)
    if np.isfinite(sep):
        parts.append(sep)
        weights.append(0.20)
    if not parts:
        return float("nan")
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), EPS)
    return float(np.dot(w, np.asarray(parts, dtype=np.float64)))


def zscore(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    mu = float(np.mean(s))
    sd = float(np.std(s))
    if not np.isfinite(sd) or sd < EPS:
        return np.zeros_like(s, dtype=np.float64)
    return (s - mu) / sd


def minmax(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    lo = float(np.min(s))
    hi = float(np.max(s))
    if not np.isfinite(hi - lo) or (hi - lo) < EPS:
        return np.zeros_like(s, dtype=np.float64)
    return (s - lo) / (hi - lo)


def ranks_from_scores(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return ranks


def softmax_metric_weights(metrics: dict[str, float], tau: float = 0.05,
                           floor: float = 0.02) -> dict[str, float]:
    valid = {m: float(v) for m, v in metrics.items() if np.isfinite(v)}
    if not valid:
        return {}
    names = list(valid)
    vals = np.asarray([valid[n] for n in names], dtype=np.float64)
    vals = vals - np.max(vals)
    denom = max(float(tau), EPS)
    raw = np.exp(vals / denom)
    raw = raw / max(float(raw.sum()), EPS)
    if floor > 0 and len(raw) > 1:
        raw = (1.0 - floor * len(raw)) * raw + floor
        raw = raw / max(float(raw.sum()), EPS)
    return {n: float(w) for n, w in zip(names, raw)}


def _common_methods(method_scores_by_key: dict[str, dict[str, np.ndarray]],
                    key: str,
                    allow_methods: set[str] | None = None) -> list[str]:
    names = []
    for method, scores_by_key in method_scores_by_key.items():
        if allow_methods is not None and method not in allow_methods:
            continue
        if key in scores_by_key:
            names.append(method)
    return sorted(names)


def best_val_fusion(val_scores_by_method, val_gt_by_key, gallery_scores_by_method,
                    keys: list[str], allow_methods: set[str] | None = None) -> FusionResult:
    scores = {}
    selected = {}
    weights = {}
    for key in keys:
        methods = _common_methods(gallery_scores_by_method, key, allow_methods)
        metrics = {
            m: average_precision(val_gt_by_key[key], val_scores_by_method[m][key])
            for m in methods
            if m in val_scores_by_method and key in val_scores_by_method[m] and key in val_gt_by_key
        }
        valid = {m: v for m, v in metrics.items() if np.isfinite(v)}
        if not valid:
            continue
        best = max(valid, key=valid.get)
        scores[key] = np.asarray(gallery_scores_by_method[best][key], dtype=np.float64)
        selected[key] = best
        weights[key] = {best: 1.0}
    return FusionResult("best_val", scores, weights, selected)


def zscore_average_fusion(gallery_scores_by_method, keys: list[str],
                          allow_methods: set[str] | None = None,
                          name: str = "zscore_avg") -> FusionResult:
    scores = {}
    weights = {}
    for key in keys:
        methods = _common_methods(gallery_scores_by_method, key, allow_methods)
        if not methods:
            continue
        arr = np.vstack([zscore(gallery_scores_by_method[m][key]) for m in methods])
        scores[key] = np.mean(arr, axis=0)
        weights[key] = {m: 1.0 / len(methods) for m in methods}
    return FusionResult(name, scores, weights, {})


def metric_weighted_zscore_fusion(val_scores_by_method, val_gt_by_key,
                                  gallery_scores_by_method, keys: list[str],
                                  allow_methods: set[str] | None = None,
                                  tau: float = 0.05,
                                  floor: float = 0.02,
                                  name: str | None = None) -> FusionResult:
    scores = {}
    weights = {}
    for key in keys:
        methods = _common_methods(gallery_scores_by_method, key, allow_methods)
        metrics = {
            m: average_precision(val_gt_by_key[key], val_scores_by_method[m][key])
            for m in methods
            if m in val_scores_by_method and key in val_scores_by_method[m] and key in val_gt_by_key
        }
        w = softmax_metric_weights(metrics, tau=tau, floor=floor)
        if not w:
            continue
        arr = np.vstack([zscore(gallery_scores_by_method[m][key]) * w[m] for m in w])
        scores[key] = np.sum(arr, axis=0)
        weights[key] = w
    return FusionResult(name or f"metric_weighted_zscore_tau{tau:g}", scores, weights, {})


def reliability_weighted_zscore_fusion(val_scores_by_method, val_gt_by_key,
                                       gallery_scores_by_method, keys: list[str],
                                       allow_methods: set[str] | None = None,
                                       tau: float = 0.05,
                                       floor: float = 0.02,
                                       topk: int = 50,
                                       name: str | None = None) -> FusionResult:
    scores = {}
    weights = {}
    for key in keys:
        methods = _common_methods(gallery_scores_by_method, key, allow_methods)
        metrics = {
            m: reliability_score(val_gt_by_key[key], val_scores_by_method[m][key], topk=topk)
            for m in methods
            if m in val_scores_by_method and key in val_scores_by_method[m] and key in val_gt_by_key
        }
        w = softmax_metric_weights(metrics, tau=tau, floor=floor)
        if not w:
            continue
        arr = np.vstack([zscore(gallery_scores_by_method[m][key]) * w[m] for m in w])
        scores[key] = np.sum(arr, axis=0)
        weights[key] = w
    return FusionResult(name or f"reliability_weighted_zscore_tau{tau:g}", scores, weights, {})


def rrf_fusion(gallery_scores_by_method, keys: list[str],
               allow_methods: set[str] | None = None,
               k: float = 60.0,
               metric_weights: dict[str, dict[str, float]] | None = None,
               name: str = "rrf") -> FusionResult:
    scores = {}
    weights = {}
    for key in keys:
        methods = _common_methods(gallery_scores_by_method, key, allow_methods)
        if not methods:
            continue
        total = None
        local_weights = metric_weights.get(key, {}) if metric_weights else {}
        if local_weights:
            methods = [m for m in methods if m in local_weights]
        if not methods:
            continue
        if not local_weights:
            local_weights = {m: 1.0 / len(methods) for m in methods}
        for method in methods:
            ranks = ranks_from_scores(gallery_scores_by_method[method][key])
            part = float(local_weights[method]) / (float(k) + ranks)
            total = part if total is None else total + part
        scores[key] = total
        weights[key] = {m: float(local_weights[m]) for m in methods}
    return FusionResult(name, scores, weights, {})


def weighted_rrf_fusion(val_scores_by_method, val_gt_by_key, gallery_scores_by_method,
                        keys: list[str], allow_methods: set[str] | None = None,
                        tau: float = 0.05, floor: float = 0.02,
                        k: float = 60.0) -> FusionResult:
    metric_weights = {}
    for key in keys:
        methods = _common_methods(gallery_scores_by_method, key, allow_methods)
        metrics = {
            m: average_precision(val_gt_by_key[key], val_scores_by_method[m][key])
            for m in methods
            if m in val_scores_by_method and key in val_scores_by_method[m] and key in val_gt_by_key
        }
        w = softmax_metric_weights(metrics, tau=tau, floor=floor)
        if w:
            metric_weights[key] = w
    return rrf_fusion(gallery_scores_by_method, keys, allow_methods, k,
                      metric_weights=metric_weights,
                      name=f"weighted_rrf_tau{tau:g}")


def soft_gate_mean(attr_scores: list[np.ndarray], threshold: float = 0.5,
                   temp: float = 0.1) -> np.ndarray:
    vals = [minmax(v) for v in attr_scores]
    mat = np.vstack(vals)
    gate = 1.0 / (1.0 + np.exp(-(mat - threshold) / max(temp, EPS)))
    return np.mean(mat, axis=0) * np.prod(gate, axis=0)


def per_attr_softgate_fusion(val_scores_by_method, val_gt_by_key,
                             gallery_scores_by_method, single_keys: list[str],
                             joint_key: str = "joint",
                             allow_methods: set[str] | None = None,
                             tau: float = 0.05,
                             floor: float = 0.02) -> FusionResult:
    scores = {}
    weights = {}
    for key in single_keys:
        sub = metric_weighted_zscore_fusion(
            val_scores_by_method, val_gt_by_key, gallery_scores_by_method,
            [key], allow_methods=allow_methods, tau=tau, floor=floor,
            name="tmp",
        )
        if key in sub.scores:
            scores[key] = minmax(sub.scores[key])
            weights[key] = sub.weights.get(key, {})
    if len(scores) == len(single_keys) and single_keys:
        scores[joint_key] = soft_gate_mean([scores[k] for k in single_keys])
        # Joint weights are summarized from attribute weights for diagnostics.
        merged = {}
        for key in single_keys:
            for method, w in weights.get(key, {}).items():
                merged[method] = merged.get(method, 0.0) + float(w) / len(single_keys)
        weights[joint_key] = merged
    return FusionResult(f"per_attr_softgate_tau{tau:g}", scores, weights, {})


def reliability_per_attr_softgate_fusion(val_scores_by_method, val_gt_by_key,
                                         gallery_scores_by_method, single_keys: list[str],
                                         joint_key: str = "joint",
                                         allow_methods: set[str] | None = None,
                                         tau: float = 0.05,
                                         floor: float = 0.02,
                                         topk: int = 50) -> FusionResult:
    scores = {}
    weights = {}
    for key in single_keys:
        sub = reliability_weighted_zscore_fusion(
            val_scores_by_method, val_gt_by_key, gallery_scores_by_method,
            [key], allow_methods=allow_methods, tau=tau, floor=floor,
            topk=topk, name="tmp",
        )
        if key in sub.scores:
            scores[key] = minmax(sub.scores[key])
            weights[key] = sub.weights.get(key, {})
    if len(scores) == len(single_keys) and single_keys:
        scores[joint_key] = soft_gate_mean([scores[k] for k in single_keys])
        merged = {}
        for key in single_keys:
            for method, w in weights.get(key, {}).items():
                merged[method] = merged.get(method, 0.0) + float(w) / len(single_keys)
        weights[joint_key] = merged
    return FusionResult(f"reliability_per_attr_softgate_tau{tau:g}", scores, weights, {})
