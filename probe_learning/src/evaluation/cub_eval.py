"""Evaluate predicted labels against CUB ground-truth attributes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.paths import PROCESSED_DIR
from src.data.cub import image_field_to_relative_path

CUB_ATTR_MAPPING: dict[str, list[int]] = {
    "dominant_vivid_blue_plumage": [248],
    "short_conical_beak": [7, 151],
    "perched_bird_pose": [235],
}


def _load_cub_ground_truth() -> tuple[np.ndarray, pd.DataFrame]:
    attrs = np.load(PROCESSED_DIR / "attrs.npy")
    records = pd.read_csv(PROCESSED_DIR / "records.csv")
    return attrs, records


def _cub_label_for_attr(attrs: np.ndarray, idx: int, attr: str) -> int:
    """Return CUB ground-truth binary label for one image (by embedding_index)."""
    indices = CUB_ATTR_MAPPING[attr]
    return int(all(attrs[idx, i] == 1 for i in indices))


def evaluate_predictions(
    pred_file: Path,
    attributes: list[str],
    tag: str = "",
) -> dict[str, dict]:
    """Compare a JSONL prediction file against CUB and return per-attribute metrics.

    Returns ``{attr: {accuracy, precision, recall, f1, evaluated, pred_pos, cub_pos, ...}}``.
    """
    attrs, records = _load_cub_ground_truth()
    path_to_idx = dict(zip(records["relative_path"], records["embedding_index"]))

    preds_by_image: dict[str, dict] = {}
    with open(pred_file) as f:
        for line in f:
            entry = json.loads(line)
            preds_by_image[entry["image"]] = entry

    results = {}
    for attr in attributes:
        if attr not in CUB_ATTR_MAPPING:
            continue
        y_true, y_pred = [], []
        for image_name, entry in preds_by_image.items():
            if attr not in entry or entry[attr] is None:
                continue
            rel_path = image_field_to_relative_path(image_name)
            idx = path_to_idx.get(rel_path)
            if idx is None:
                continue
            y_true.append(_cub_label_for_attr(attrs, idx, attr))
            y_pred.append(int(entry[attr]))

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        results[attr] = {
            "evaluated": len(y_true),
            "pred_pos": int(y_pred_arr.sum()),
            "cub_pos": int(y_true_arr.sum()),
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        }

    if tag:
        print(f"\n{'='*60}")
        print(f"  CUB Evaluation: {tag}")
        print(f"{'='*60}")
    for attr, m in results.items():
        print(f"  {attr:35s}  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
              f"rec={m['recall']:.4f}  f1={m['f1']:.4f}  "
              f"(eval={m['evaluated']}, pred+={m['pred_pos']}, cub+={m['cub_pos']})")

    return results


def evaluate_inmemory(
    predictions: list[dict],
    attributes: list[str],
) -> dict[str, dict]:
    """Evaluate a list of prediction dicts against CUB GT (in-memory, no file IO)."""
    attrs, records = _load_cub_ground_truth()
    path_to_idx = dict(zip(records["relative_path"], records["embedding_index"]))

    results = {}
    for attr in attributes:
        if attr not in CUB_ATTR_MAPPING:
            continue
        y_true, y_pred = [], []
        for entry in predictions:
            if attr not in entry or entry[attr] is None:
                continue
            rel_path = image_field_to_relative_path(entry["image"])
            idx = path_to_idx.get(rel_path)
            if idx is None:
                continue
            y_true.append(_cub_label_for_attr(attrs, idx, attr))
            y_pred.append(int(entry[attr]))

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


def print_comparison_table(
    baseline_metrics: dict[str, dict],
    expanded_metrics: dict[str, dict],
    attributes: list[str],
) -> None:
    """Print a side-by-side comparison of baseline vs expanded metrics."""
    print(f"\n{'='*80}")
    print("  CUB Evaluation Comparison: Baseline (Round 1) vs Expanded (Round 2)")
    print(f"{'='*80}")
    header = (f"  {'Attribute':35s}  {'':5s}  {'Acc':>7s}  {'Prec':>7s}  "
              f"{'Rec':>7s}  {'F1':>7s}  {'Eval':>5s}  {'P+':>5s}  {'C+':>5s}")
    print(header)
    print("  " + "-" * 78)

    for attr in attributes:
        if attr not in baseline_metrics or attr not in expanded_metrics:
            continue
        b, e = baseline_metrics[attr], expanded_metrics[attr]
        print(f"  {attr:35s}  {'R1':5s}  {b['accuracy']:7.4f}  {b['precision']:7.4f}  "
              f"{b['recall']:7.4f}  {b['f1']:7.4f}  {b['evaluated']:5d}  "
              f"{b['pred_pos']:5d}  {b['cub_pos']:5d}")
        diff_acc = e["accuracy"] - b["accuracy"]
        diff_f1 = e["f1"] - b["f1"]
        print(f"  {'':35s}  {'R2':5s}  {e['accuracy']:7.4f}  {e['precision']:7.4f}  "
              f"{e['recall']:7.4f}  {e['f1']:7.4f}  {e['evaluated']:5d}  "
              f"{e['pred_pos']:5d}  {e['cub_pos']:5d}")
        print(f"  {'':35s}  {'diff':5s}  {diff_acc:+7.4f}  "
              f"{e['precision']-b['precision']:+7.4f}  "
              f"{e['recall']-b['recall']:+7.4f}  {diff_f1:+7.4f}")
        print("  " + "-" * 78)
