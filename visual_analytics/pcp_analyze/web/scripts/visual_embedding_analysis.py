"""Visual-embedding projections and clustering for the workbench.

The canonical browser path consumes precomputed static PCA, UMAP and cluster
files.  This module owns the offline computation and a private disk cache used
by the exporter.  The source SigLIP embedding remains on the host.  Every
fitted object sees Development rows only; Validation and Frozen Test rows are
assigned with ``transform``/``predict``.  MiniBatchKMeans always uses the
L2-normalized source embedding rather than either 2-D projection.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ALGORITHM_VERSION = "siglip-visual-l2-pca50-umap2-minibatch-kmeans-v2"
SUPPORTED_CLUSTER_COUNTS = (30, 50, 100)
RANDOM_STATE = 42
PCA_REDUCED_DIMENSION = 50
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_TRANSFORM_BATCH_SIZE = 4096
KMEANS_BATCH_SIZE = 4096
KMEANS_N_INIT = 3
KMEANS_MAX_ITER = 150
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: dict[str, threading.Lock] = {}
# CelebA's normalized Development matrix is several hundred MiB.  Keep cache
# misses globally serial so two remote users cannot multiply that peak memory.
_COMPUTE_GUARD = threading.Lock()


@dataclass(frozen=True)
class VisualEmbeddingAnalysis:
    algorithm_version: str
    payload: bytes
    row_count: int
    cluster_count: int
    fit_row_count: int
    embedding_dimension: int
    pca_byte_length: int
    umap_byte_length: int
    explained_variance_ratio: tuple[float, float]
    umap_n_neighbors: int
    umap_min_dist: float
    umap_metric: str
    umap_random_state: int
    umap_input_dimension: int
    umap_transform_batch_size: int
    cache_hit: bool


def _cache_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _normalized_rows(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= np.float32(1e-12)):
        raise ValueError("SigLIP embedding contains a non-finite or zero-norm row")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _validated_analysis_inputs(
    embeddings: Any, development_mask: Any
) -> tuple[Any, Any, Any]:
    import numpy as np

    values = np.asarray(embeddings)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"embedding matrix must be [row, dimension], got {values.shape}")
    if values.dtype.kind != "f":
        raise ValueError(f"embedding matrix must be floating point, got {values.dtype}")
    mask = np.asarray(development_mask, dtype=np.uint8).reshape(-1)
    if mask.shape != (values.shape[0],) or not np.isin(mask, (0, 1)).all():
        raise ValueError("Development mask must be a row-aligned binary vector")
    fit_indices = np.flatnonzero(mask)
    if len(fit_indices) < 3:
        raise ValueError("Development needs at least three rows for 2-D projections")
    return values, mask, fit_indices


def compute_visual_embedding_projection(
    embeddings: Any,
    development_mask: Any,
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[Any, Any, tuple[float, float], int, int, int]:
    """Fit PCA and UMAP on Development, then transform held-out rows.

    The returned projections are visualization-only.  UMAP uses a deterministic
    seed and cosine distance over a Development-fitted PCA reduction (at most
    50 dimensions) of the L2-normalized SigLIP rows.  This bounds UMAP memory;
    clustering remains in the unreduced normalized embedding space.
    """

    import numpy as np
    import umap
    from sklearn.decomposition import PCA

    values, mask, fit_indices = _validated_analysis_inputs(
        embeddings, development_mask
    )
    fit_values = _normalized_rows(values[fit_indices])
    reduced_dimension = min(
        PCA_REDUCED_DIMENSION,
        int(values.shape[1]),
        int(len(fit_indices) - 1),
    )
    pca_model = PCA(
        n_components=reduced_dimension,
        svd_solver="randomized",
        random_state=int(random_state),
    )
    fit_reduced = np.asarray(pca_model.fit_transform(fit_values), dtype=np.float32)
    fit_pca = fit_reduced[:, :2]

    effective_neighbors = min(UMAP_N_NEIGHBORS, len(fit_indices) - 1)
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=int(random_state),
        transform_seed=int(random_state),
        n_jobs=1,
        low_memory=True,
    )
    fit_umap = np.asarray(umap_model.fit_transform(fit_reduced), dtype=np.float32)

    row_count = int(values.shape[0])
    pca_coordinates = np.empty((row_count, 2), dtype=np.float32)
    umap_coordinates = np.empty((row_count, 2), dtype=np.float32)
    pca_coordinates[fit_indices] = fit_pca
    umap_coordinates[fit_indices] = fit_umap

    held_out_indices = np.flatnonzero(mask == 0)
    for start in range(0, len(held_out_indices), UMAP_TRANSFORM_BATCH_SIZE):
        batch_indices = held_out_indices[start : start + UMAP_TRANSFORM_BATCH_SIZE]
        batch = _normalized_rows(values[batch_indices])
        reduced = np.asarray(pca_model.transform(batch), dtype=np.float32)
        pca_coordinates[batch_indices] = reduced[:, :2]
        umap_coordinates[batch_indices] = np.asarray(
            umap_model.transform(reduced), dtype=np.float32
        )

    if not np.isfinite(pca_coordinates).all():
        raise RuntimeError("Visual PCA coordinates contain non-finite values")
    if not np.isfinite(umap_coordinates).all():
        raise RuntimeError("Visual UMAP coordinates contain non-finite values")
    explained = tuple(float(value) for value in pca_model.explained_variance_ratio_)
    return (
        np.ascontiguousarray(pca_coordinates, dtype="<f4"),
        np.ascontiguousarray(umap_coordinates, dtype="<f4"),
        (explained[0], explained[1]),
        int(len(fit_indices)),
        int(effective_neighbors),
        int(reduced_dimension),
    )


def compute_visual_embedding_clusters(
    embeddings: Any,
    development_mask: Any,
    cluster_count: int,
    pca_coordinates: Any,
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[Any, int]:
    """Fit K-means in normalized embedding space and predict held-out rows."""

    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    if cluster_count not in SUPPORTED_CLUSTER_COUNTS:
        raise ValueError(
            f"clusters must be one of {SUPPORTED_CLUSTER_COUNTS}, got {cluster_count}"
        )
    values, mask, fit_indices = _validated_analysis_inputs(
        embeddings, development_mask
    )
    if len(fit_indices) <= cluster_count:
        raise ValueError(
            f"Development needs more than K={cluster_count} rows, found {len(fit_indices)}"
        )
    pca = np.asarray(pca_coordinates, dtype=np.float32)
    if pca.shape != (values.shape[0], 2) or not np.isfinite(pca).all():
        raise ValueError("PCA coordinates must be a finite row-aligned [row, 2] matrix")

    fit_values = _normalized_rows(values[fit_indices])
    model = MiniBatchKMeans(
        n_clusters=int(cluster_count),
        random_state=int(random_state),
        batch_size=KMEANS_BATCH_SIZE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        reassignment_ratio=0.01,
    ).fit(fit_values)

    raw_labels = np.empty(values.shape[0], dtype=np.int32)
    raw_labels[fit_indices] = np.asarray(model.labels_, dtype=np.int32)
    held_out_indices = np.flatnonzero(mask == 0)
    for start in range(0, len(held_out_indices), KMEANS_BATCH_SIZE):
        batch_indices = held_out_indices[start : start + KMEANS_BATCH_SIZE]
        batch = _normalized_rows(values[batch_indices])
        raw_labels[batch_indices] = np.asarray(model.predict(batch), dtype=np.int32)

    fit_raw_labels = raw_labels[fit_indices]
    fit_counts = np.bincount(fit_raw_labels, minlength=cluster_count)
    if np.any(fit_counts == 0):
        missing = np.flatnonzero(fit_counts == 0).tolist()
        raise RuntimeError(f"Visual K-means produced empty Development clusters: {missing}")

    # K-means IDs are arbitrary.  Canonicalize them by Development PCA centroid
    # so cache rebuilds and cluster legends keep a deterministic order.  PCA is
    # used only for naming the clusters, never for assigning membership.
    fit_pca = pca[fit_indices]
    centroids = np.vstack(
        [
            fit_pca[fit_raw_labels == cluster_id].mean(axis=0)
            for cluster_id in range(cluster_count)
        ]
    )
    order = np.lexsort((centroids[:, 1], centroids[:, 0]))
    raw_to_canonical = np.empty(cluster_count, dtype=np.uint8)
    raw_to_canonical[order] = np.arange(cluster_count, dtype=np.uint8)
    labels = raw_to_canonical[raw_labels]
    if not np.array_equal(
        np.unique(labels[fit_indices]), np.arange(cluster_count, dtype=np.uint8)
    ):
        raise RuntimeError("Canonical visual cluster labels are incomplete")
    return np.ascontiguousarray(labels, dtype=np.uint8), int(len(fit_indices))


def compute_visual_embedding_analysis(
    embeddings: Any,
    development_mask: Any,
    cluster_count: int,
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[Any, Any, Any, tuple[float, float], int, int, int]:
    """Convenience wrapper used by focused tests and one-shot callers."""

    (
        pca,
        umap_coordinates,
        explained,
        fit_rows,
        effective_neighbors,
        reduced_dimension,
    ) = (
        compute_visual_embedding_projection(
            embeddings, development_mask, random_state=random_state
        )
    )
    labels, cluster_fit_rows = compute_visual_embedding_clusters(
        embeddings,
        development_mask,
        cluster_count,
        pca,
        random_state=random_state,
    )
    if cluster_fit_rows != fit_rows:
        raise RuntimeError("Projection and clustering Development scopes disagree")
    return (
        pca,
        umap_coordinates,
        labels,
        explained,
        fit_rows,
        effective_neighbors,
        reduced_dimension,
    )


def _validate_cached_file(path: Path, spec: Any, expected_bytes: int) -> bool:
    return bool(
        isinstance(spec, dict)
        and path.is_file()
        and path.stat().st_size == expected_bytes
        and int(spec.get("bytes", -1)) == expected_bytes
        and str(spec.get("sha256", "")) == _sha256_file(path)
    )


def load_or_build_visual_embedding_analysis(
    *,
    cache_root: Path,
    task_id: str,
    embedding_path: Path,
    development_mask: bytes,
    expected_image_ids: Sequence[str],
    source_image_ids: Sequence[str],
    cluster_count: int,
    embedding_sha256: str | None = None,
) -> VisualEmbeddingAnalysis:
    """Return compact PCA2+UMAP2+label data backed by an atomic disk cache.

    PCA and UMAP are task-level artifacts shared by every supported K.  A cache
    miss for a new K therefore fits only MiniBatchKMeans and appends its labels.
    """

    import numpy as np

    if cluster_count not in SUPPORTED_CLUSTER_COUNTS:
        raise ValueError(
            f"clusters must be one of {SUPPORTED_CLUSTER_COUNTS}, got {cluster_count}"
        )
    source = embedding_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SigLIP embedding is unavailable: {source}")
    expected_ids = tuple(str(value) for value in expected_image_ids)
    observed_ids = tuple(str(value) for value in source_image_ids)
    if observed_ids != expected_ids:
        raise RuntimeError("SigLIP embedding row order does not match the browser bundle")
    row_count = len(expected_ids)
    if len(development_mask) != row_count:
        raise RuntimeError("Development mask length does not match the browser bundle")
    mask_vector = np.frombuffer(development_mask, dtype=np.uint8)
    if not np.isin(mask_vector, (0, 1)).all():
        raise RuntimeError("Development mask must be binary")
    expected_fit_row_count = int(mask_vector.sum())
    if expected_fit_row_count <= cluster_count:
        raise RuntimeError(
            f"Development needs more than K={cluster_count} rows, "
            f"found {expected_fit_row_count}"
        )
    effective_neighbors = min(UMAP_N_NEIGHBORS, expected_fit_row_count - 1)

    embedding = np.load(source, mmap_mode="r", allow_pickle=False)
    if embedding.shape[0] != row_count or embedding.ndim != 2:
        raise RuntimeError(
            f"SigLIP embedding shape {embedding.shape} does not match {row_count} rows"
        )
    if embedding.dtype != np.dtype("float32"):
        raise RuntimeError(f"SigLIP embedding must be float32, got {embedding.dtype}")
    expected_reduced_dimension = min(
        PCA_REDUCED_DIMENSION,
        int(embedding.shape[1]),
        expected_fit_row_count - 1,
    )

    source_stat = source.stat()
    source_sha256 = embedding_sha256 or _sha256_file(source)
    if (
        len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("SigLIP embedding SHA-256 must be lowercase hexadecimal")
    identity = {
        "algorithmVersion": ALGORITHM_VERSION,
        "taskId": task_id,
        "rowCount": row_count,
        "embeddingDimension": int(embedding.shape[1]),
        "embeddingBytes": int(source_stat.st_size),
        "embeddingMtimeNs": int(source_stat.st_mtime_ns),
        "embeddingSha256": source_sha256,
        "developmentMaskSha256": _sha256_bytes(development_mask),
        "imageIdsSha256": _sha256_bytes("\n".join(expected_ids).encode("utf-8")),
        "umapTransformBatchSize": UMAP_TRANSFORM_BATCH_SIZE,
    }
    task_cache = cache_root.resolve() / ALGORITHM_VERSION / task_id
    metadata_path = task_cache / "metadata.json"
    pca_path = task_cache / "pca-2d.f32"
    umap_path = task_cache / "umap-2d.f32"
    labels_path = task_cache / f"labels-k{cluster_count}.u8"
    expected_pca_bytes = row_count * 2 * np.dtype("<f4").itemsize
    expected_umap_bytes = row_count * 2 * np.dtype("<f4").itemsize
    expected_label_bytes = row_count

    with _cache_lock(task_cache):
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                metadata = {}
        identity_matches = all(metadata.get(key) == value for key, value in identity.items())
        pca_spec = metadata.get("pca") if identity_matches else None
        umap_spec = metadata.get("umap") if identity_matches else None
        cluster_specs = metadata.get("clusters") if identity_matches else None
        cluster_spec = (
            cluster_specs.get(str(cluster_count))
            if isinstance(cluster_specs, dict)
            else None
        )
        pca_valid = _validate_cached_file(pca_path, pca_spec, expected_pca_bytes)
        umap_valid = _validate_cached_file(
            umap_path, umap_spec, expected_umap_bytes
        )
        labels_valid = _validate_cached_file(
            labels_path, cluster_spec, expected_label_bytes
        )
        explained = pca_spec.get("explainedVarianceRatio") if isinstance(pca_spec, dict) else None
        explained_valid = bool(
            isinstance(explained, list)
            and len(explained) == 2
            and all(isinstance(value, (int, float)) for value in explained)
            and isinstance(pca_spec, dict)
            and int(pca_spec.get("reducedDimension", -1))
            == expected_reduced_dimension
        )
        fit_row_count = int(metadata.get("fitRowCount", -1)) if identity_matches else -1
        umap_metadata_valid = bool(
            isinstance(umap_spec, dict)
            and int(umap_spec.get("nNeighbors", -1)) == effective_neighbors
            and float(umap_spec.get("minDist", -1.0)) == UMAP_MIN_DIST
            and str(umap_spec.get("metric", "")) == UMAP_METRIC
            and int(umap_spec.get("randomState", -1)) == RANDOM_STATE
            and int(umap_spec.get("transformSeed", -1)) == RANDOM_STATE
            and int(umap_spec.get("inputDimension", -1))
            == expected_reduced_dimension
            and str(umap_spec.get("input", "")) == "development-fitted-pca"
            and int(umap_spec.get("transformBatchSize", -1))
            == UMAP_TRANSFORM_BATCH_SIZE
        )
        projection_valid = bool(
            pca_valid
            and umap_valid
            and explained_valid
            and umap_metadata_valid
            and fit_row_count == expected_fit_row_count
        )
        cache_hit = projection_valid and labels_valid

        if not cache_hit:
            with _COMPUTE_GUARD:
                retained_clusters = (
                    dict(cluster_specs)
                    if identity_matches and isinstance(cluster_specs, dict)
                    else {}
                )
                if not projection_valid:
                    (
                        pca_coordinates,
                        umap_coordinates,
                        explained_tuple,
                        fit_row_count,
                        fitted_neighbors,
                        fitted_reduced_dimension,
                    ) = compute_visual_embedding_projection(
                        embedding, mask_vector
                    )
                    if fitted_neighbors != effective_neighbors:
                        raise RuntimeError("UMAP effective neighbor count changed unexpectedly")
                    if fitted_reduced_dimension != expected_reduced_dimension:
                        raise RuntimeError("UMAP input dimension changed unexpectedly")
                    pca_payload = pca_coordinates.tobytes(order="C")
                    umap_payload = umap_coordinates.tobytes(order="C")
                    _atomic_write_bytes(pca_path, pca_payload)
                    _atomic_write_bytes(umap_path, umap_payload)
                    pca_spec = {
                        "path": pca_path.name,
                        "bytes": len(pca_payload),
                        "sha256": _sha256_bytes(pca_payload),
                        "explainedVarianceRatio": list(explained_tuple),
                        "reducedDimension": fitted_reduced_dimension,
                    }
                    umap_spec = {
                        "path": umap_path.name,
                        "bytes": len(umap_payload),
                        "sha256": _sha256_bytes(umap_payload),
                        "nNeighbors": fitted_neighbors,
                        "minDist": UMAP_MIN_DIST,
                        "metric": UMAP_METRIC,
                        "randomState": RANDOM_STATE,
                        "transformSeed": RANDOM_STATE,
                        "input": "development-fitted-pca",
                        "inputDimension": fitted_reduced_dimension,
                        "transformBatchSize": UMAP_TRANSFORM_BATCH_SIZE,
                    }
                    explained = list(explained_tuple)
                else:
                    pca_payload = pca_path.read_bytes()
                    pca_coordinates = np.frombuffer(
                        pca_payload, dtype="<f4"
                    ).reshape(row_count, 2)

                if not labels_valid:
                    labels, cluster_fit_row_count = compute_visual_embedding_clusters(
                        embedding,
                        mask_vector,
                        cluster_count,
                        pca_coordinates,
                    )
                    if cluster_fit_row_count != fit_row_count:
                        raise RuntimeError(
                            "Projection and clustering Development scopes disagree"
                        )
                    label_payload = labels.tobytes(order="C")
                    _atomic_write_bytes(labels_path, label_payload)
                    retained_clusters[str(cluster_count)] = {
                        "path": labels_path.name,
                        "bytes": len(label_payload),
                        "sha256": _sha256_bytes(label_payload),
                    }
                metadata = {
                    **identity,
                    "backbone": "siglip",
                    "fitScope": "development",
                    "fitRowCount": fit_row_count,
                    "assignmentScope": (
                        "Development fit; held-out rows via PCA.transform, "
                        "UMAP.transform, and MiniBatchKMeans.predict"
                    ),
                    "groundTruthUsed": False,
                    "pca": pca_spec,
                    "umap": umap_spec,
                    "clusters": retained_clusters,
                }
                _atomic_write_json(metadata_path, metadata)

        pca_payload = pca_path.read_bytes()
        umap_payload = umap_path.read_bytes()
        label_payload = labels_path.read_bytes()
        if (
            len(pca_payload) != expected_pca_bytes
            or len(umap_payload) != expected_umap_bytes
            or len(label_payload) != expected_label_bytes
        ):
            raise RuntimeError("Visual analysis cache failed its byte-length contract")
        if not isinstance(explained, list) or len(explained) != 2:
            raise RuntimeError("Visual analysis cache has no PCA variance metadata")
        result = VisualEmbeddingAnalysis(
            algorithm_version=ALGORITHM_VERSION,
            payload=pca_payload + umap_payload + label_payload,
            row_count=row_count,
            cluster_count=cluster_count,
            fit_row_count=fit_row_count,
            embedding_dimension=int(embedding.shape[1]),
            pca_byte_length=expected_pca_bytes,
            umap_byte_length=expected_umap_bytes,
            explained_variance_ratio=(float(explained[0]), float(explained[1])),
            umap_n_neighbors=effective_neighbors,
            umap_min_dist=UMAP_MIN_DIST,
            umap_metric=UMAP_METRIC,
            umap_random_state=RANDOM_STATE,
            umap_input_dimension=expected_reduced_dimension,
            umap_transform_batch_size=UMAP_TRANSFORM_BATCH_SIZE,
            cache_hit=cache_hit,
        )
        memory_map = getattr(embedding, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
        return result
