"""Evaluate predicted labels against SUN ground-truth attributes."""

import json
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.paths import SUN_ROOT, SUN_TASK_REGISTRY, SNOW_ROAD_TASK
from src.data.sun import image_field_to_relative_path

SUN_MAPPING: dict[str, dict] = {
    "Dark asphalt road": {"sun_idx": 47, "inverse": False},
    "Ground covered in white snow": {"sun_idx": 70, "inverse": False},
    "Visible tire tracks or wet slush on road": {"sun_idx": 81, "inverse": False},
    "Overcast gray lighting": {"sun_idx": 75, "inverse": True},
}
GT_THRESHOLD = 0.5


def get_sun_mapping(task: str | None = None) -> dict[str, dict]:
    """Return SUN_MAPPING for *task*; falls back to the module-level default."""
    if task is None:
        return SUN_MAPPING
    cfg = SUN_TASK_REGISTRY.get(task)
    if cfg is None:
        return SUN_MAPPING
    return cfg["sun_mapping"]


def load_sun_ground_truth() -> tuple[dict[str, int], np.ndarray]:
    attr_db = SUN_ROOT / "SUNAttributeDB"
    images_mat = scipy.io.loadmat(str(attr_db / "images.mat"))
    img_list = [e[0] for e in images_mat["images"].ravel()]
    img_to_idx = {name: i for i, name in enumerate(img_list)}
    labels_mat = scipy.io.loadmat(str(attr_db / "attributeLabels_continuous.mat"))
    labels_cv = labels_mat["labels_cv"]
    return img_to_idx, labels_cv


def sun_gt_binary(
    img_to_idx: dict[str, int],
    labels_cv: np.ndarray,
    rel_path: str,
    sun_idx: int,
    inverse: bool,
) -> int | None:
    mat_idx = img_to_idx.get(rel_path)
    if mat_idx is None:
        return None
    val = float(labels_cv[mat_idx, sun_idx])
    if inverse:
        return 1 if val < (1 - GT_THRESHOLD) else 0
    return 1 if val >= GT_THRESHOLD else 0


def evaluate_inmemory(
    predictions: list[dict],
    img_to_idx: dict[str, int],
    labels_cv: np.ndarray,
    task: str | None = None,
) -> dict[str, dict]:
    """Evaluate a list of prediction dicts against SUN GT (in-memory, no file IO)."""
    mapping_table = get_sun_mapping(task)
    results = {}
    for attr, mapping in mapping_table.items():
        y_true, y_pred = [], []
        for entry in predictions:
            rel_path = image_field_to_relative_path(entry["image"])
            gt = sun_gt_binary(img_to_idx, labels_cv, rel_path,
                               mapping["sun_idx"], mapping["inverse"])
            if gt is None:
                continue
            y_true.append(gt)
            y_pred.append(int(entry.get(attr, 0)))

        if not y_true:
            continue
        yt, yp = np.array(y_true), np.array(y_pred)
        results[attr] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "pred_pos": int(yp.sum()),
            "gt_pos": int(yt.sum()),
            "evaluated": len(yt),
        }
    return results
