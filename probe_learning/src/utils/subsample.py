"""Stratified subsampling of VQA labels for ablation experiments."""

import numpy as np


def stratified_subsample(
    labeled: list[dict],
    attributes: list[str],
    n_samples: int,
    seed: int,
    min_pos_per_attr: int = 3,
) -> list[dict]:
    """Subsample VQA labels while ensuring minimum positives per attribute.

    1. For each attribute, guarantee at least ``min_pos_per_attr`` positive
       entries are included (capped by actual available positives).
    2. Fill remaining slots with random entries from the rest.
    """
    rng = np.random.RandomState(seed)
    n_samples = min(n_samples, len(labeled))

    selected: set[int] = set()
    for attr in attributes:
        pos_indices = [i for i, e in enumerate(labeled) if e.get(attr) == 1]
        need = min(min_pos_per_attr, len(pos_indices))
        if need > 0:
            chosen = rng.choice(pos_indices, size=need, replace=False)
            selected.update(chosen.tolist())

    remaining_target = n_samples - len(selected)
    if remaining_target > 0:
        candidates = [i for i in range(len(labeled)) if i not in selected]
        fill = rng.choice(candidates, size=min(remaining_target, len(candidates)), replace=False)
        selected.update(fill.tolist())

    return [labeled[i] for i in sorted(selected)]
