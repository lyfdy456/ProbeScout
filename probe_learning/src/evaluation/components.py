"""The four paper component variants and their Validation-only selection."""
from __future__ import annotations

import math
import numpy as np

from src.evaluation.retrieval_metrics import evaluation_metrics, fit_training_threshold


def ordered_gate_cases(config):
    """Prefer the least changed gate in AP ties, matching Appendix C.4."""
    return sorted([(float(dt), float(k)) for dt in config['theta_shifts']
                   for k in config['temperature_scales']],
                  key=lambda v: (abs(v[0]) / .05 + abs(np.log(v[1])), abs(v[0]), v[1], v[0]))


def component_scores(attribute_scores, holistic_scores, theta, temperature,
                     variant, config, shift=0., scale=1.):
    """Frozen eight-probe means; ablations do not refit probes or fusion weights.

    Keep the reference evaluator's float64 gate/product operations and float32
    output so ties and the saved score ordering remain reproducible.
    """
    q = np.asarray(attribute_scores, dtype=np.float64)
    h = np.asarray(holistic_scores, dtype=np.float64)
    if variant['softgate']:
        new_theta = np.clip(np.asarray(theta, dtype=np.float64) + shift, 0., 1.)
        new_temperature = np.minimum(np.asarray(temperature, dtype=np.float64) * scale,
                                     config['temperature_cap'])
        gates = np.exp(-np.logaddexp(0., -(q - new_theta) / new_temperature))
        conjunction = gates.prod(axis=1)
    else:
        conjunction = q.prod(axis=1)
    strength = config['embedding_strength']
    factor = (1. - strength) + strength * h if variant['embedding'] else 1.
    return np.clip(conjunction * factor, 0., 1.).astype(np.float32)


def select_on_validation(q, h, theta, temperature, labels, variant, config):
    """Accept only Val-sliced arrays; this function cannot inspect Test labels."""
    if set(np.asarray(labels).tolist()) != {0, 1}:
        raise ValueError('Paper gate selection requires positive and negative VQA Val labels')
    candidates = []
    cases = ordered_gate_cases(config) if variant['softgate'] else [(0., 1.)]
    for shift, scale in cases:
        scores = component_scores(q, h, theta, temperature, variant, config, shift, scale)
        fitted = fit_training_threshold(labels, scores)
        metrics = evaluation_metrics(labels, scores, fitted['threshold'])
        candidates.append({'theta_shift': shift, 'temperature_scale': scale,
                           'threshold': fitted['threshold'], 'val_ap': metrics['Ap'],
                           'val_f1': metrics['F1']})
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate['val_ap'] > best['val_ap'] + 1e-12:
            best = candidate
    return dict(best), candidates


def macro_metrics(records, variants):
    summaries = []
    task_ids = {r['task_id'] for r in records}
    for variant in variants:
        rows = [r for r in records if r['variant'] == variant['id']]
        if len(rows) != len(task_ids) or {r['task_id'] for r in rows} != task_ids:
            raise ValueError('Every variant must cover every selected task exactly once')
        summaries.append({'variant': variant['id'], 'task_count': len(rows), **{
            metric: math.fsum(r[metric] for r in rows) / len(rows)
            for metric in ('test_ap', 'test_f1', 'gallery_ap', 'gallery_f1')}})
    return summaries
