"""Evaluate predicted labels against Stanford Cars ground-truth attributes.

Ground truth is derived from cars_annos.mat class names:
  - is_bmw         : class name starts with "BMW"
  - is_ford        : class name starts with "Ford"
  - is_mustang     : class name contains "Mustang"
  - is_sedan       : class name contains "Sedan"
  - is_convertible : class name contains "Convertible"
Keyed by bare image filename (e.g. "000123.jpg"), matching records.csv relative_path
and the VQA jsonl `image` field.
"""

import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.paths import CARS_ANNOS_MAT, CARS_TASK_REGISTRY, get_cars_task


def get_cars_mapping(task: str | None = None) -> dict[str, dict]:
    if task is None:
        return CARS_TASK_REGISTRY["task_bmw_sedan"]["cars_mapping"]
    try:
        return get_cars_task(task)["cars_mapping"]
    except Exception:
        return CARS_TASK_REGISTRY["task_bmw_sedan"]["cars_mapping"]


def load_cars_ground_truth() -> dict[str, dict[str, int]]:
    """Return {basename: GT fields} for all annotated images.

    Built-in fields cover the historical tasks. ``class_name`` is kept so
    sidecar-defined Cars tasks can evaluate class-name predicates without
    hardcoding a new GT column for every brand/model/body variant.
    """
    mat = scipy.io.loadmat(str(CARS_ANNOS_MAT))
    class_names = [str(c[0]) for c in mat["class_names"].ravel()]
    annotations = mat["annotations"].ravel()

    gt: dict[str, dict[str, int]] = {}
    for ann in annotations:
        rel_im_path = str(ann["relative_im_path"][0])
        basename = rel_im_path.split("/")[-1]
        class_id = int(ann["class"][0][0])  # 1-indexed
        name = class_names[class_id - 1]
        gt[basename] = {
            "class_name": name,
            "is_bmw": 1 if name.startswith("BMW") else 0,
            "is_ford": 1 if name.startswith("Ford") else 0,
            "is_mustang": 1 if "Mustang" in name else 0,
            "is_sedan": 1 if "Sedan" in name else 0,
            "is_convertible": 1 if "Convertible" in name else 0,
        }
    return gt


def evaluate_cars_predictions(pred_file, gt_lookup, cars_mapping, tag=""):
    """Load a prediction jsonl and print per-attribute metrics vs Cars GT."""
    import json

    print(f"\n{'='*60}")
    print(f"  Cars GT Evaluation: {tag}")
    print(f"{'='*60}")

    preds_by_attr = {a: [] for a in cars_mapping}
    gt_by_attr = {a: [] for a in cars_mapping}

    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            gt_entry = gt_lookup.get(entry["image"])
            if gt_entry is None:
                continue
            for attr, mapping in cars_mapping.items():
                if mapping.get("gt_type") == "class_name_contains":
                    needle = str(mapping["contains"])
                    gt_val = 1 if needle.lower() in str(gt_entry["class_name"]).lower() else 0
                elif mapping.get("gt_type") == "class_name_startswith":
                    needle = str(mapping["prefix"])
                    gt_val = 1 if str(gt_entry["class_name"]).lower().startswith(needle.lower()) else 0
                else:
                    gt_val = gt_entry[mapping["gt_type"]]
                if mapping.get("inverse", False):
                    gt_val = 1 - gt_val
                preds_by_attr[attr].append(int(entry.get(attr, 0)))
                gt_by_attr[attr].append(gt_val)

    results = {}
    for attr in cars_mapping:
        yt = np.array(gt_by_attr[attr])
        yp = np.array(preds_by_attr[attr])
        metrics = {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "eval": len(yt),
            "pred_pos": int(yp.sum()),
            "gt_pos": int(yt.sum()),
        }
        results[attr] = metrics
        print(f"  {attr:50s}  acc={metrics['accuracy']:.4f}  prec={metrics['precision']:.4f}  "
              f"rec={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  "
              f"(eval={metrics['eval']}, pred+={metrics['pred_pos']}, gt+={metrics['gt_pos']})")
    return results


def evaluate_inmemory(
    predictions: list[dict],
    gt_lookup: dict[str, dict[str, int]],
    task: str | None = None,
) -> dict[str, dict]:
    """Evaluate prediction dicts against Cars GT (in-memory, no file IO)."""
    mapping_table = get_cars_mapping(task)
    results = {}
    for attr, mapping in mapping_table.items():
        gt_type = mapping["gt_type"]
        inverse = mapping.get("inverse", False)
        y_true, y_pred = [], []
        for entry in predictions:
            basename = entry["image"]
            gt_entry = gt_lookup.get(basename)
            if gt_entry is None:
                continue
            if gt_type == "class_name_contains":
                needle = str(mapping["contains"])
                gt_val = 1 if needle.lower() in str(gt_entry["class_name"]).lower() else 0
            elif gt_type == "class_name_startswith":
                needle = str(mapping["prefix"])
                gt_val = 1 if str(gt_entry["class_name"]).lower().startswith(needle.lower()) else 0
            else:
                gt_val = gt_entry[gt_type]
            if inverse:
                gt_val = 1 - gt_val
            y_true.append(gt_val)
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
