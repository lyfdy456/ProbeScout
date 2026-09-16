#!/usr/bin/env python3
"""Add deterministic fine-grained K-means schemes to exported web bundles.

The clustering input is the same Joint method-rank profile used by the existing
absolute clusters.  Columns are z-standardized, GT is excluded from fitting,
and cluster IDs are reordered from low to high mean rank for stable reading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WEB_ROOT = SCRIPT_DIR.parent
VENDOR_DIR = SCRIPT_DIR / ".vendor"
FINE_CLUSTER_COUNTS = (30, 50, 100)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
if VENDOR_DIR.is_dir() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        action="append",
        type=Path,
        default=[],
        help="Exported bundle directory. Repeat for multiple bundles; defaults to catalog tasks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clusters",
        default=",".join(str(value) for value in FINE_CLUSTER_COUNTS),
        help="Comma-separated fine-grained cluster counts (default: 30,50,100).",
    )
    return parser.parse_args()


def parse_cluster_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--clusters must be a comma-separated list of integers") from error
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("--clusters must contain unique cluster counts")
    if set(counts) != set(FINE_CLUSTER_COUNTS):
        raise ValueError("--clusters must contain the complete supported set: 30,50,100")
    return FINE_CLUSTER_COUNTS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def stage_bytes(path: Path, payload: bytes) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    return temporary


def file_spec(path: Path, rows: int) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "dtype": "uint8",
        "byteOrder": "not-applicable",
        "layout": "row-major",
        "shape": [rows],
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def catalog_bundle_roots() -> list[Path]:
    public_root = WEB_ROOT / "public"
    catalog_path = public_root / "data" / "catalog.json"
    catalog = read_json(catalog_path)
    roots: list[Path] = []
    seen: set[Path] = set()
    for dataset in catalog.get("datasets", []):
        for task in dataset.get("tasks", []):
            data_root = str(task.get("dataRoot", "")).lstrip("/")
            root = (public_root / data_root).resolve()
            if root not in seen:
                roots.append(root)
                seen.add(root)
    return roots


def standardized_joint_profiles(
    bundle: Path,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    rows = int(manifest["rowCount"])
    method_count = int(manifest["methodCount"])
    target_count = int(manifest["targetCount"])
    cluster_method_count = int(manifest["clusterMethodCount"])
    targets = [str(target["id"]) for target in manifest["retrievalTargets"]]
    if "joint" not in targets:
        raise ValueError(f"{bundle} has no Joint retrieval target")
    joint_index = targets.index("joint")
    ranks_path = bundle / manifest["files"]["ranks"]["path"]
    ranks = np.memmap(
        ranks_path,
        mode="r",
        dtype="<f4",
        shape=(rows, method_count, target_count),
    )
    raw = np.asarray(ranks[:, :cluster_method_count, joint_index], dtype=np.float32)
    development_spec = manifest.get("files", {}).get("developmentMask")
    if not isinstance(development_spec, dict):
        raise ValueError(
            f"{bundle} has no developmentMask; refresh evaluation partitions before clustering"
        )
    development_mask = np.fromfile(
        bundle / str(development_spec["path"]), dtype=np.uint8
    )
    if development_mask.shape != (rows,) or not np.isin(development_mask, (0, 1)).all():
        raise ValueError(f"Invalid developmentMask in {bundle}")
    fit_indices = np.flatnonzero(development_mask)
    if len(fit_indices) < max(FINE_CLUSTER_COUNTS):
        raise ValueError(f"Development split is too small for fine clustering: {len(fit_indices)}")
    fit_raw = raw[fit_indices]
    mean = fit_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = fit_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    features = np.ascontiguousarray((raw - mean) / scale, dtype=np.float32)
    return raw, features, joint_index, fit_indices


def fit_labels(
    raw: np.ndarray,
    features: np.ndarray,
    clusters: int,
    seed: int,
    fit_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fit_rows = np.arange(len(features), dtype=np.int64) if fit_indices is None else np.asarray(fit_indices)
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=4096,
        n_init=10,
        max_iter=300,
        reassignment_ratio=0.01,
    ).fit(features[fit_rows])
    raw_labels = np.asarray(model.predict(features), dtype=np.int32)
    counts = np.bincount(raw_labels, minlength=clusters)
    if np.any(counts == 0):
        empty = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Fine clustering produced empty clusters: {empty}")

    fit_labels = raw_labels[fit_rows]
    fit_counts = np.bincount(fit_labels, minlength=clusters)
    if np.any(fit_counts == 0):
        empty = np.flatnonzero(fit_counts == 0).tolist()
        raise ValueError(f"Fine clustering produced empty Development clusters: {empty}")
    fit_raw_centers = np.vstack(
        [raw[fit_rows][fit_labels == cluster].mean(axis=0) for cluster in range(clusters)]
    )
    order = np.argsort(fit_raw_centers.mean(axis=1), kind="stable")
    raw_to_canonical = np.empty(clusters, dtype=np.uint8)
    raw_to_canonical[order] = np.arange(clusters, dtype=np.uint8)
    labels = raw_to_canonical[raw_labels]
    canonical_centers = np.vstack(
        [raw[labels == cluster].mean(axis=0) for cluster in range(clusters)]
    )
    canonical_counts = counts[order]
    return labels, canonical_centers, canonical_counts, float(model.inertia_)


def build_summaries(
    manifest: dict[str, Any],
    raw: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    counts: np.ndarray,
    ground_truth: np.ndarray,
    joint_index: int,
    scheme: str,
    clusters: int,
) -> list[dict[str, Any]]:
    rows = len(raw)
    methods = [str(value) for value in manifest["clusterMethods"]]
    targets = [target for target in manifest["retrievalTargets"]]
    fixed_count = methods.index("MLP") if "MLP" in methods else max(1, min(5, len(methods)))
    overall_positive_rate = float(np.mean(ground_truth[:, joint_index]))
    digits = max(2, len(str(clusters)))
    summaries: list[dict[str, Any]] = []
    for cluster_id, (center, size) in enumerate(zip(centers, counts, strict=True)):
        mask = labels == cluster_id
        joint_count = int(np.sum(ground_truth[mask, joint_index]))
        joint_rate = float(np.mean(ground_truth[mask, joint_index]))
        fixed_mean = float(np.mean(center[:fixed_count]))
        learned_mean = float(np.mean(center[fixed_count:]))
        top = np.argsort(-center)[:4]
        bottom = np.argsort(center)[:4]
        summaries.append(
            {
                "scheme": scheme,
                "cluster_id": cluster_id,
                "label": f"K{clusters} cluster {cluster_id + 1:0{digits}d}",
                "label_zh": f"K{clusters} 簇 {cluster_id + 1:0{digits}d}",
                "size": int(size),
                "fraction": float(size / rows),
                "mean_rank": float(np.mean(center)),
                "fixed_mean": fixed_mean,
                "learned_mean": learned_mean,
                "learned_minus_fixed": learned_mean - fixed_mean,
                "mean_within_image_rank_std": float(np.mean(np.std(raw[mask], axis=1))),
                "joint_positive_count": joint_count,
                "joint_positive_rate": joint_rate,
                "joint_positive_enrichment": (
                    joint_rate / overall_positive_rate if overall_positive_rate > 0 else None
                ),
                "attribute_positive_count": {
                    str(target["id"]): int(np.sum(ground_truth[mask, target_index]))
                    for target_index, target in enumerate(targets)
                    if str(target["id"]) != "joint"
                },
                "top_methods": [methods[int(index)] for index in top],
                "bottom_methods": [methods[int(index)] for index in bottom],
            }
        )
    return summaries


def fine_scheme(clusters: int) -> str:
    return f"fine{clusters}"


def fine_file_key(clusters: int) -> str:
    return f"fine{clusters}Labels"


def fine_labels_name(clusters: int) -> str:
    return f"fine{clusters}-labels.u8"


def process_bundle(bundle: Path, cluster_counts: tuple[int, ...], seed: int) -> None:
    bundle = bundle.resolve()
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing manifest/metadata in {bundle}")
    manifest = read_json(manifest_path)
    metadata = read_json(metadata_path)
    rows = int(manifest["rowCount"])
    for clusters in cluster_counts:
        if clusters < 2 or clusters > 255 or clusters >= rows:
            raise ValueError("--clusters must be between 2 and min(255, rowCount - 1)")

    raw, features, joint_index, fit_indices = standardized_joint_profiles(bundle, manifest)
    sample_rng = np.random.default_rng(20260807)
    sample_indices = np.sort(
        sample_rng.choice(rows, min(6000, rows), replace=False)
    )
    gt_spec = manifest["files"]["groundTruth"]
    ground_truth = np.fromfile(bundle / gt_spec["path"], dtype=np.uint8).reshape(
        rows, int(manifest["targetCount"])
    )

    requested_schemes = {fine_scheme(clusters) for clusters in cluster_counts}
    cluster_summaries = [
        item
        for item in metadata.get("clusterSummary", [])
        if item.get("scheme") not in requested_schemes
    ]
    fine_clusterings: dict[str, dict[str, Any]] = {}
    staged_labels: list[tuple[Path, Path, int]] = []

    for clusters in cluster_counts:
        scheme = fine_scheme(clusters)
        labels, centers, counts, inertia = fit_labels(
            raw, features, clusters, seed, fit_indices=fit_indices
        )
        silhouette_sample = float(
            silhouette_score(features[sample_indices], labels[sample_indices])
        )
        cluster_summaries.extend(
            build_summaries(
                manifest,
                raw,
                labels,
                centers,
                counts,
                ground_truth,
                joint_index,
                scheme,
                clusters,
            )
        )
        labels_path = bundle / fine_labels_name(clusters)
        staged_labels.append(
            (stage_bytes(labels_path, labels.tobytes(order="C")), labels_path, clusters)
        )
        fine_clusterings[scheme] = {
            "scheme": scheme,
            "algorithm": "MiniBatchKMeans",
            "implementation": "scikit-learn",
            "implementationVersion": sklearn.__version__,
            "clusters": clusters,
            "randomState": seed,
            "nInit": 10,
            "batchSize": 4096,
            "maxIter": 300,
            "reassignmentRatio": 0.01,
            "basisTarget": "joint",
            "featureMethods": manifest["clusterMethods"],
            "featureTransform": "per-column z-score",
            "fitScope": "development",
            "fitMaskFileKey": "developmentMask",
            "fitRowCount": int(len(fit_indices)),
            "assignmentScope": "all rows via MiniBatchKMeans.predict",
            "groundTruthUsedForClustering": False,
            "canonicalOrdering": "ascending cluster mean Joint rank",
            "inertia": inertia,
            "silhouetteSample": silhouette_sample,
            "silhouetteSampleSize": int(len(sample_indices)),
            "smallestCluster": int(np.min(counts)),
            "largestCluster": int(np.max(counts)),
        }
        print(
            f"{bundle}: K={clusters}, smallest={int(np.min(counts))}, "
            f"largest={int(np.max(counts))}, silhouette={silhouette_sample:.4f}"
        )

    for temporary, labels_path, clusters in staged_labels:
        temporary.replace(labels_path)
        manifest["files"][fine_file_key(clusters)] = file_spec(labels_path, rows)

    metadata["clusterSummary"] = cluster_summaries
    metadata["fineClusterings"] = {
        fine_scheme(clusters): fine_clusterings[fine_scheme(clusters)]
        for clusters in FINE_CLUSTER_COUNTS
        if fine_scheme(clusters) in fine_clusterings
    }
    if "fine50" in metadata["fineClusterings"]:
        # Keep the original single-K field as a compatibility alias.
        metadata["fineClustering"] = metadata["fineClusterings"]["fine50"]
    write_json(metadata_path, metadata)
    manifest["files"]["metadata"] = {
        "path": "metadata.json",
        "encoding": "utf-8-json",
        "shape": [1],
        "bytes": metadata_path.stat().st_size,
        "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    manifest["clusters"]["schemes"] = [
        {
            "id": fine_scheme(clusters),
            "label": "Fine-grained",
            "family": "fine",
            "clusters": clusters,
            "labelsFileKey": fine_file_key(clusters),
        }
        for clusters in FINE_CLUSTER_COUNTS
    ] + [
        {
            "id": "absolute",
            "label": "Absolute",
            "family": "structural",
            "clusters": 4,
            "labelsFileKey": "absoluteLabels",
        },
        {
            "id": "shape",
            "label": "Shape",
            "family": "structural",
            "clusters": 3,
            "labelsFileKey": "shapeLabels",
        },
    ]
    manifest["clusters"]["defaultScheme"] = "fine50"
    manifest["clusters"]["note"] = (
        "Fine-grained, absolute, and shape clusters use Joint method-rank profiles; "
        "ground truth is only used for post-hoc summaries."
    )
    manifest["generatedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    cluster_counts = parse_cluster_counts(args.clusters)
    roots = [path.resolve() for path in args.data_root] or catalog_bundle_roots()
    for root in roots:
        process_bundle(root, cluster_counts, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
