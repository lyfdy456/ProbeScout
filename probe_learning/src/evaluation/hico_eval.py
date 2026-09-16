"""Ground-truth attributes for HICO tasks, derived from anno.mat image-level
HOI labels (600 HOI classes = noun+verb; values {1,-1,0,nan}).

We expose binary per-image attributes keyed by records.csv `relative_path`
(e.g. "train2015/HICO_train2015_00000001.jpg"). The GT columns are precomputed
into records.csv by scripts/precompute_backbone_embeddings.py --dataset hico --backbone siglip, so loading is a cheap
CSV read here.

For task_bike_jumping the two probing attributes are:
  - "bicycle" -> is_bicycle  (any bicycle HOI present)
  - "jumping" -> is_jumping  (any jump HOI present)
The joint concept (bicycle & jumping) corresponds to HOI 'bicycle jump'.
"""

import numpy as np
import pandas as pd
import scipy.io

from src.utils.paths import HICO_ANNO_MAT, HICO_PROCESSED_DIR

# attr name (as used in tasks / VQA answer keys) -> records.csv column + inverse flag
HICO_ATTR_MAPPING: dict[str, dict] = {
    "bicycle": {"gt_col": "is_bicycle", "inverse": False},
    "jumping": {"gt_col": "is_jumping", "inverse": False},
    "not jumping": {"gt_col": "is_jumping", "inverse": True},
    "bike jump": {"gt_col": "is_bike_jump", "inverse": False},
    "not bike jump": {"gt_col": "is_bike_jump", "inverse": True},
}


def load_hico_ground_truth() -> dict[str, dict[str, int]]:
    """Return {relative_path: {is_bicycle, is_jumping, is_bike_jump}} for all images."""
    records = pd.read_csv(HICO_PROCESSED_DIR / "records.csv")
    gt: dict[str, dict[str, int]] = {}
    for _, row in records.iterrows():
        gt[row["relative_path"]] = {
            "is_bicycle": int(row["is_bicycle"]),
            "is_jumping": int(row["is_jumping"]),
            "is_bike_jump": int(row["is_bike_jump"]),
        }
    return gt


def load_hico_task_ground_truth(
    mapping: dict[str, dict],
) -> dict[str, dict[str, int]]:
    """Build per-image GT for object, verb, and exact-HOI task attributes."""
    mat = scipy.io.loadmat(str(HICO_ANNO_MAT))
    actions = mat["list_action"]
    train_names = [mat["list_train"][i, 0].item() for i in range(mat["list_train"].shape[0])]
    test_names = [mat["list_test"][i, 0].item() for i in range(mat["list_test"].shape[0])]

    action_pairs = []
    for i in range(actions.shape[0]):
        entry = actions[i, 0]
        action_pairs.append((entry["vname"].item(), entry["nname"].item()))

    attr_indices: dict[str, list[int]] = {}
    for attr, spec in mapping.items():
        gt_type = spec.get("gt_type")
        if gt_type == "hico_object":
            obj = spec["object"]
            indices = [i for i, (_, noun) in enumerate(action_pairs) if noun == obj]
            if not indices:
                raise ValueError(f"HICO object {obj!r} for {attr!r} has no official HOI rows")
        elif gt_type == "hico_verb":
            verb = spec["verb"]
            indices = [i for i, (row_verb, _) in enumerate(action_pairs) if row_verb == verb]
            if not indices:
                raise ValueError(f"HICO verb {verb!r} for {attr!r} has no official HOI rows")
        elif gt_type == "hico_exact_hoi":
            idx = int(spec["hoi_idx"])
            if not 0 <= idx < len(action_pairs):
                raise ValueError(f"HICO hoi_idx {idx} for {attr!r} is out of range")
            verb, noun = action_pairs[idx]
            expected = (spec.get("verb", verb), spec.get("object", noun))
            if (verb, noun) != expected:
                raise ValueError(
                    f"HICO hoi_idx {idx} for {attr!r} is {(verb, noun)}, expected {expected}"
                )
            indices = [idx]
        else:
            raise ValueError(f"unsupported HICO gt_type {gt_type!r} for {attr!r}")
        attr_indices[attr] = indices

    gt: dict[str, dict[str, int]] = {}
    for split, names, anno in (
        ("train2015", train_names, mat["anno_train"]),
        ("test2015", test_names, mat["anno_test"]),
    ):
        labels = {
            attr: (np.nan_to_num(anno[indices, :], nan=0.0) == 1.0).any(axis=0).astype(np.int8)
            for attr, indices in attr_indices.items()
        }
        for col, name in enumerate(names):
            gt[f"{split}/{name}"] = {attr: int(values[col]) for attr, values in labels.items()}
    return gt
