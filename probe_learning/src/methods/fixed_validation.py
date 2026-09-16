"""Shared explicit holdout support without changing legacy random splits.

Callers must establish image-ID isolation before creating these arrays. When
``validation_data`` is supplied, every input training row remains a fit row;
the external holdout is never concatenated into training or split again.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_labeled_validation(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if validation_data is None:
        return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
        raise ValueError("validation_data must contain exactly (X_val, y_val)")
    X_fit, y_fit = np.asarray(X), np.asarray(y)
    X_val, y_val = (np.asarray(value) for value in validation_data)
    for name, features, labels in (("fit", X_fit, y_fit), ("validation", X_val, y_val)):
        if features.ndim < 2 or labels.ndim != 1 or len(labels) == 0 or len(features) != len(labels):
            raise ValueError(f"explicit {name} features and labels must be non-empty and row-aligned")
        if not np.all(np.isfinite(features)) or not np.all(np.isin(labels, (0, 1))):
            raise ValueError(f"explicit {name} data must have finite features and binary labels")
    if X_fit.shape[1:] != X_val.shape[1:]:
        raise ValueError("explicit validation feature shape must match the fit feature shape")
    return X_fit, X_val, y_fit, y_val
