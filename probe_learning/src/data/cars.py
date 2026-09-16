"""Stanford Cars dataset helpers: load embeddings, VQA labels, and build training arrays.

The Cars pipeline uses bare image filenames (e.g. "000123.jpg") consistently across the
VQA jsonl `image` field, records.csv `relative_path`, and GT lookup, so the
image-field <-> relative-path conversion is the identity.
"""

import json

import numpy as np
import pandas as pd

from src.utils.paths import CARS_PROCESSED_DIR, CARS_EMBEDDING_FILES


def image_field_to_relative_path(image_name: str) -> str:
    """Cars jsonl `image` already equals records.csv `relative_path` (bare filename)."""
    return image_name


def load_data(embedding_name: str, label_file):
    """Return embeddings array, records DataFrame, and labeled entries."""
    emb_file, _ = CARS_EMBEDDING_FILES[embedding_name]
    embeddings = np.load(CARS_PROCESSED_DIR / emb_file)
    records = pd.read_csv(CARS_PROCESSED_DIR / "records.csv")

    labeled = []
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labeled.append(json.loads(line))

    return embeddings, records, labeled


def build_path_to_index(records: pd.DataFrame) -> dict[str, int]:
    return dict(zip(records["relative_path"], records["embedding_index"]))


def prepare_attribute_data(
    attr: str,
    labeled: list[dict],
    embeddings: np.ndarray,
    path_to_idx: dict[str, int],
):
    """Build X, y arrays for one attribute, skipping nulls and unmatched."""
    X_list, y_list, matched_images = [], [], []
    for entry in labeled:
        label = entry.get(attr)
        if label is None:
            continue
        rel_path = image_field_to_relative_path(entry["image"])
        idx = path_to_idx.get(rel_path)
        if idx is None:
            continue
        X_list.append(embeddings[idx])
        y_list.append(int(label))
        matched_images.append(entry["image"])

    return np.array(X_list), np.array(y_list), matched_images
