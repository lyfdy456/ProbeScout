#!/usr/bin/env python3
"""Build and validate all real Main17 browser bundles.

This is an orchestration layer around the audited PCP, multi-target score,
clustering, web-data, and fine-cluster exporters.  It never synthesizes a task
or substitutes scores: every task identity comes from the Main17
manifest and every generated bundle must pass the browser contract audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation_scope import (
    CLEAN_TEST_POLICY,
    build_evaluation_masks,
    evaluation_contract,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WEB_ROOT = SCRIPT_DIR.parent
ANALYSIS_ROOT = WEB_ROOT.parent
REPOSITORY_ROOT = ANALYSIS_ROOT.parents[1]
PUBLIC_DATA_ROOT = WEB_ROOT / "public" / "data"
TASK_DATA_ROOT = PUBLIC_DATA_ROOT / "tasks"
PCP_ROOT = ANALYSIS_ROOT / "pcp_result"
CLUSTER_ROOT = ANALYSIS_ROOT / "cluster_results"
MANIFEST_PATH = REPOSITORY_ROOT / "configs/main17_tasks.json"

# Use the paper manifest for both export scope and modeled target identities.
# Immutable asset names are preserved even when their task order is non-contiguous.
PAPER_SCOPE = json.loads(
    (REPOSITORY_ROOT / "manifests/paper_main17.json").read_text(encoding="utf-8")
)
MAIN17_ORDERS = tuple(int(row["task_id"].split("_", 1)[0]) for row in PAPER_SCOPE["tasks"])
EXPECTED_TASKS = {
    int(row["task_id"].split("_", 1)[0]): (
        row["dataset"], row["task_id"].split("_", 2)[2], tuple(row["attribute_ids"])
    )
    for row in PAPER_SCOPE["tasks"]
}
TASK_LABELS = {
    int(row["task_id"].split("_", 1)[0]): row["label"] for row in PAPER_SCOPE["tasks"]
}

DATASET_LABELS = {
    "cars": "Stanford Cars",
    "celeba": "CelebA",
    "hico": "HICO-DET",
}

IMAGE_ROOTS = {
    "cars": REPOSITORY_ROOT / "dataset" / "raw" / "stanford_cars" / "images",
    "celeba": REPOSITORY_ROOT / "dataset" / "raw" / "CelebA" / "img_celeba",
    "hico": REPOSITORY_ROOT / "dataset" / "raw" / "HICO" / "images",
}

INITIAL_FILE_KEYS = (
    "metadata",
    "imageIds",
    "rawScores",
    "calibratedScores",
    "ranks",
    "fine30Labels",
    "fine50Labels",
    "fine100Labels",
    "absoluteLabels",
    "shapeLabels",
    "pca2d",
    "metrics",
    "groundTruth",
    "developmentMask",
    "validationMask",
    "testMask",
)

VISUAL_CLUSTER_COUNTS = (30, 50, 100)
VISUAL_FILE_KEYS = {
    "pca": "visualPca2d",
    "umap": "visualUmap2d",
    30: "visualFine30Labels",
    50: "visualFine50Labels",
    100: "visualFine100Labels",
}
VISUAL_CACHE_ROOT = ANALYSIS_ROOT / "runtime" / "tuning" / "visual-analysis"
_EMBEDDING_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class TaskSpec:
    order: int
    dataset: str
    task_name: str
    attributes: tuple[str, ...]
    supervision_stage: str
    probe_stage: str = "iterative"
    explicit_data_root: Path | None = None
    explicit_score_bundle: Path | None = None
    explicit_task_root: Path | None = None
    explicit_supervision_ids_path: Path | None = None

    @property
    def task_id(self) -> str:
        return f"{self.order:03d}_{self.dataset}_{self.task_name}"

    @property
    def artifact_stem(self) -> str:
        return f"{self.order:03d}_{self.dataset}_{self.task_name.removeprefix('task_')}"

    @property
    def pcp_html(self) -> Path:
        return PCP_ROOT / f"{self.artifact_stem}_joint_mean_rank.html"

    @property
    def score_bundle(self) -> Path:
        if self.explicit_score_bundle is not None:
            return self.explicit_score_bundle.resolve()
        return PCP_ROOT / f"{self.artifact_stem}_multitarget_scores.npz"

    @property
    def cluster_dir(self) -> Path:
        if self.order == 32:
            return ANALYSIS_ROOT / "cluster_result"
        return CLUSTER_ROOT / self.task_id

    @property
    def data_root(self) -> Path:
        if self.explicit_data_root is not None:
            return self.explicit_data_root.resolve()
        return PUBLIC_DATA_ROOT if self.order == 32 else TASK_DATA_ROOT / self.task_id

    @property
    def public_data_root(self) -> str:
        return "/data" if self.order == 32 else f"/data/tasks/{self.task_id}"

    @property
    def task_root(self) -> Path:
        if self.explicit_task_root is not None:
            return self.explicit_task_root.resolve()
        return REPOSITORY_ROOT / "dataset" / "tasks" / self.dataset / self.task_name

    @property
    def supervision_ids_path(self) -> Path:
        return (
            self.task_root
            / "supervision"
            / self.supervision_stage
            / "train_labeled_indices.json"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders",
        default=",".join(str(value) for value in MAIN17_ORDERS),
        help="Comma-separated Main17 order subset.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild valid existing intermediate and browser artifacts.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing bundles and compare the catalog without modifying files.",
    )
    parser.add_argument(
        "--refresh-evaluation-only",
        action="store_true",
        help=(
            "Rebuild Development/Validation/Frozen Test using the audited "
            "candidate-Test-minus-active-supervision protocol."
        ),
    )
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help=(
            "Precompute and statically export visual PCA2, UMAP2, and "
            "K30/K50/K100 labels without rebuilding scores or atlases."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    replace_with_retry(temporary, path)


def replace_with_retry(source: Path, target: Path) -> None:
    """Commit a generated file despite transient Windows no-delete readers.

    Rename remains the normal atomic path.  Some long-lived local preview
    readers open large JSON files without delete sharing; after bounded
    retries, use a flushed in-place copy and verify its complete SHA before
    removing the temporary source.
    """

    for attempt in range(40):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 39:
                expected_sha256 = file_sha256(source)
                with source.open("rb") as input_handle, target.open("wb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                if file_sha256(target) != expected_sha256:
                    raise RuntimeError(
                        f"Verified in-place fallback failed for {target}"
                    )
                source.unlink()
                return
            time.sleep(0.125)


def select_tasks(orders: Iterable[int]) -> list[TaskSpec]:
    requested = tuple(int(value) for value in orders)
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate task orders: {requested}")
    invalid = sorted(set(requested).difference(MAIN17_ORDERS))
    if invalid:
        raise ValueError(f"Orders are not in the active Main17 set: {invalid}")

    manifest = read_json(MANIFEST_PATH)
    rows_by_order = {int(row["order"]): row for row in manifest["tasks"]}
    tasks: list[TaskSpec] = []
    for order in requested:
        row = rows_by_order.get(order)
        if row is None:
            raise ValueError(f"Formal task order {order} is missing from {MANIFEST_PATH}")
        task = TaskSpec(
            order=order,
            dataset=str(row["dataset"]),
            task_name=str(row["task"]),
            attributes=tuple(str(value) for value in row["attributes"]),
            supervision_stage=str(
                row.get("supervision_stages", {}).get("iterative", "iterative")
            ),
        )
        expected_dataset, expected_task_name, _ = EXPECTED_TASKS[order]
        if (task.dataset, task.task_name) != (expected_dataset, expected_task_name):
            raise ValueError(
                f"Frozen Main17 identity drift at order {order}: "
                f"{(task.dataset, task.task_name)} != "
                f"{(expected_dataset, expected_task_name)}"
            )
        expected_id = f"{order:03d}_{task.dataset}_{task.task_name}"
        if task.task_id != expected_id or order not in TASK_LABELS:
            raise ValueError(f"Unregistered Main17 identity: {task}")
        tasks.append(task)
    return tasks


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def has_files(root: Path, names: Iterable[str]) -> bool:
    return all((root / name).is_file() for name in names)


def ensure_intermediates(task: TaskSpec, force: bool) -> None:
    python = sys.executable
    if force or not has_files(
        task.pcp_html.parent,
        (task.pcp_html.name, task.pcp_html.with_suffix(".audit.json").name),
    ):
        run(
            [
                python,
                str(ANALYSIS_ROOT / "build_score_rank_pcp.py"),
                "--order",
                str(task.order),
                "--split",
                "all",
                "--score-key",
                "joint",
                "--seed-mode",
                "mean-rank",
                "--output",
                str(task.pcp_html),
            ]
        )
    if force or not task.score_bundle.is_file():
        run(
            [
                python,
                str(ANALYSIS_ROOT / "export_multitarget_scores.py"),
                "--order",
                str(task.order),
                "--output",
                str(task.score_bundle),
            ]
        )
    cluster_files = (
        "cluster_data.npz",
        "image_assignments.csv",
        "cluster_summary.csv",
        "representative_images.csv",
        "cluster_audit.json",
    )
    if force or not has_files(task.cluster_dir, cluster_files):
        run(
            [
                python,
                str(ANALYSIS_ROOT / "cluster_score_rank_profiles.py"),
                "--pcp-html",
                str(task.pcp_html),
                "--output",
                str(task.cluster_dir),
                "--absolute-k",
                "4",
                "--shape-k",
                "3",
            ]
        )


def atlas_source(task: TaskSpec, tasks_by_order: dict[int, TaskSpec]) -> Path | None:
    canonical_order = {"cars": 1, "celeba": 6, "hico": 8}[task.dataset]
    if task.order == canonical_order:
        return None
    return tasks_by_order[canonical_order].data_root


def ensure_browser_bundle(
    task: TaskSpec,
    tasks_by_order: dict[int, TaskSpec],
    force: bool,
) -> None:
    required = (
        "manifest.json",
        "metadata.json",
        "image-ids.json",
        "raw-scores.f32",
        "calibrated-scores.f32",
        "ranks.f32",
        "absolute-labels.u8",
        "shape-labels.u8",
        "pca-2d.f32",
        "umap-2d.f32",
        "metrics.f32",
        "ground-truth.u8",
        "development-mask.u8",
        "validation-mask.u8",
        "test-mask.u8",
    )
    if force or not has_files(task.data_root, required):
        command = [
            sys.executable,
            str(SCRIPT_DIR / "export_web_data.py"),
            "--pcp-html",
            str(task.pcp_html),
            "--score-bundle",
            str(task.score_bundle),
            "--cluster-dir",
            str(task.cluster_dir),
            "--image-root",
            str(IMAGE_ROOTS[task.dataset]),
            "--output-data-dir",
            str(task.data_root),
            "--require-umap",
        ]
        command.extend(
            (
                "--active-training-supervision-file",
                str(resolve_active_supervision_ids_path(task)),
                "--active-training-supervision-stage",
                task.supervision_stage,
            )
        )
        source = atlas_source(task, tasks_by_order)
        if source is not None:
            command.extend(("--reuse-atlases-from", str(source)))
        elif force:
            command.extend(("--force-atlases", "--force-umap"))
        elif (task.data_root / "atlases").is_dir() and not (
            task.data_root / "manifest.json"
        ).is_file():
            # A prior export may have failed after atlas creation but before
            # the query/manifest contract was committed. Rebuild rather than
            # trusting an uncommitted cache.
            command.append("--force-atlases")
        run(command)

    fine_cluster_files = tuple(
        task.data_root / f"fine{clusters}-labels.u8"
        for clusters in (30, 50, 100)
    )
    if force or not all(path.is_file() for path in fine_cluster_files):
        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "add_fine_clusters.py"),
                "--data-root",
                str(task.data_root),
            ]
        )


def process_task(
    task: TaskSpec,
    tasks_by_order: dict[int, TaskSpec],
    force: bool,
) -> str:
    print(f"[{task.task_id}] exporting", flush=True)
    ensure_intermediates(task, force)
    ensure_browser_bundle(task, tasks_by_order, force)
    refresh_evaluation_contract(task)
    return task.task_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_file_sha256(path: Path) -> str:
    """Hash a shared source embedding once per exporter process."""

    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _EMBEDDING_SHA256_CACHE.get(key)
    if cached is None:
        cached = file_sha256(resolved)
        _EMBEDDING_SHA256_CACHE[key] = cached
    return cached


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def binary_file_spec(
    path: Path,
    *,
    dtype: str,
    shape: list[int],
) -> dict[str, Any]:
    return {
        "path": path.name,
        "dtype": dtype,
        "byteOrder": "little" if dtype == "float32" else "not-applicable",
        "layout": "row-major",
        "shape": shape,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def resolve_visual_embedding_source(
    task: TaskSpec,
) -> tuple[Path, tuple[str, ...]]:
    """Resolve the same SigLIP rows used by the private tuning service."""

    linear_root = REPOSITORY_ROOT / "probe_learning"
    for path in (linear_root, linear_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from src.data.dataset_adapter import build_adapter

    adapter = build_adapter(task.dataset, task.task_name)
    records = adapter.records
    try:
        embedding_indices = np.asarray(records["embedding_index"], dtype=np.int64)
        source_ids = np.asarray(records["relative_path"], dtype=str)
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Source records cannot align visual embeddings for {task.task_id}"
        ) from error
    order = np.argsort(embedding_indices, kind="stable")
    if not np.array_equal(embedding_indices[order], np.arange(len(order))):
        raise RuntimeError(
            f"Source embedding indices are not contiguous for {task.task_id}"
        )
    if getattr(adapter, "query_embedding_paths", {}).get("siglip") is not None:
        raise RuntimeError(
            "Visual analysis requires Query rows to share the gallery embedding file"
        )
    embedding_path = Path(adapter.emb_path).resolve().parent / "siglip_embedding.npy"
    return embedding_path, tuple(str(value) for value in source_ids[order])


def export_visual_embedding_bundle(task: TaskSpec) -> dict[str, Any]:
    """Publish one complete, immutable visual-analysis bundle atomically.

    Projection/KMeans computation reuses the private cache, but the browser
    contract consists only of five static files referenced by the task
    manifest.  The manifest is committed last, so an interrupted run cannot
    expose a partial visualEmbedding contract.
    """

    from visual_embedding_analysis import (
        ALGORITHM_VERSION,
        KMEANS_BATCH_SIZE,
        KMEANS_MAX_ITER,
        KMEANS_N_INIT,
        RANDOM_STATE,
        UMAP_TRANSFORM_BATCH_SIZE,
        load_or_build_visual_embedding_analysis,
    )

    bundle = task.data_root.resolve()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Browser bundle is missing for visual export: {task.task_id}"
        )
    manifest = read_json(manifest_path)
    files = manifest["files"]
    rows = int(manifest["rowCount"])
    image_ids = tuple(
        str(value)
        for value in read_json(bundle / str(files["imageIds"]["path"]))
    )
    development_mask = (
        bundle / str(files["developmentMask"]["path"])
    ).read_bytes()
    if len(image_ids) != rows or len(development_mask) != rows:
        raise RuntimeError(
            f"Visual export row contract failed for {task.task_id}"
        )
    embedding_path, source_image_ids = resolve_visual_embedding_source(task)
    if source_image_ids != image_ids:
        raise RuntimeError(
            f"SigLIP embedding row order does not match {task.task_id}"
        )
    embedding_sha256 = cached_file_sha256(embedding_path)

    pca_payload: bytes | None = None
    umap_payload: bytes | None = None
    label_payloads: dict[int, bytes] = {}
    common_result: Any | None = None
    cache_hits = 0
    for cluster_count in VISUAL_CLUSTER_COUNTS:
        result = load_or_build_visual_embedding_analysis(
            cache_root=VISUAL_CACHE_ROOT,
            task_id=task.task_id,
            embedding_path=embedding_path,
            development_mask=development_mask,
            expected_image_ids=image_ids,
            source_image_ids=source_image_ids,
            cluster_count=cluster_count,
            embedding_sha256=embedding_sha256,
        )
        pca_end = result.pca_byte_length
        umap_end = pca_end + result.umap_byte_length
        current_pca = result.payload[:pca_end]
        current_umap = result.payload[pca_end:umap_end]
        current_labels = result.payload[umap_end:]
        if pca_payload is None:
            pca_payload = current_pca
            umap_payload = current_umap
            common_result = result
        elif pca_payload != current_pca or umap_payload != current_umap:
            raise RuntimeError(
                f"Visual projections changed between K values for {task.task_id}"
            )
        if len(current_labels) != rows:
            raise RuntimeError(
                f"Visual K{cluster_count} labels are not row aligned for {task.task_id}"
            )
        label_payloads[cluster_count] = current_labels
        cache_hits += int(result.cache_hit)

    if pca_payload is None or umap_payload is None or common_result is None:
        raise RuntimeError(f"Visual export produced no projections for {task.task_id}")

    # Content-addressed names keep manifest-last publishing atomic even when a
    # previous static generation is already live: new bytes never replace a
    # path referenced by the old manifest.
    output_paths = {
        "pca": bundle
        / f"visual-pca-2d.{hashlib.sha256(pca_payload).hexdigest()[:16]}.f32",
        "umap": bundle
        / f"visual-umap-2d.{hashlib.sha256(umap_payload).hexdigest()[:16]}.f32",
        **{
            cluster_count: bundle
            / (
                f"visual-fine{cluster_count}-labels."
                f"{hashlib.sha256(label_payloads[cluster_count]).hexdigest()[:16]}.u8"
            )
            for cluster_count in VISUAL_CLUSTER_COUNTS
        },
    }
    atomic_write_bytes(output_paths["pca"], pca_payload)
    atomic_write_bytes(output_paths["umap"], umap_payload)
    for cluster_count in VISUAL_CLUSTER_COUNTS:
        atomic_write_bytes(output_paths[cluster_count], label_payloads[cluster_count])

    files[VISUAL_FILE_KEYS["pca"]] = binary_file_spec(
        output_paths["pca"], dtype="float32", shape=[rows, 2]
    )
    files[VISUAL_FILE_KEYS["umap"]] = binary_file_spec(
        output_paths["umap"], dtype="float32", shape=[rows, 2]
    )
    for cluster_count in VISUAL_CLUSTER_COUNTS:
        files[VISUAL_FILE_KEYS[cluster_count]] = binary_file_spec(
            output_paths[cluster_count], dtype="uint8", shape=[rows]
        )

    source_stat = embedding_path.stat()
    canonical_ids_sha256 = hashlib.sha256(
        "\n".join(image_ids).encode("utf-8")
    ).hexdigest()
    manifest["visualEmbedding"] = {
        "contractVersion": 1,
        "available": True,
        "loading": "lazy-static",
        "algorithmVersion": ALGORITHM_VERSION,
        "backbone": "siglip",
        "embeddingDimension": common_result.embedding_dimension,
        "fitScope": "development",
        "fitMaskFileKey": "developmentMask",
        "fitRowCount": common_result.fit_row_count,
        "assignmentScope": (
            "Development fit; held-out rows via PCA.transform, UMAP.transform, "
            "and MiniBatchKMeans.predict"
        ),
        "groundTruthUsed": False,
        "inputIdentity": {
            "embeddingFile": embedding_path.name,
            "embeddingDtype": "float32",
            "embeddingShape": [rows, common_result.embedding_dimension],
            "embeddingBytes": int(source_stat.st_size),
            "embeddingMtimeNs": str(source_stat.st_mtime_ns),
            "embeddingSha256": embedding_sha256,
            "developmentMaskSha256": str(files["developmentMask"]["sha256"]),
            "canonicalImageIdsSha256": canonical_ids_sha256,
        },
        "pca": {
            "fileKey": VISUAL_FILE_KEYS["pca"],
            "algorithm": "sklearn.decomposition.PCA",
            "solver": "randomized",
            "randomState": RANDOM_STATE,
            "reducedDimension": common_result.umap_input_dimension,
            "explainedVarianceRatio": list(common_result.explained_variance_ratio),
        },
        "umap": {
            "fileKey": VISUAL_FILE_KEYS["umap"],
            "algorithm": "umap-learn",
            "input": "development-fitted-pca",
            "inputDimension": common_result.umap_input_dimension,
            "nNeighbors": common_result.umap_n_neighbors,
            "minDist": common_result.umap_min_dist,
            "metric": common_result.umap_metric,
            "randomState": common_result.umap_random_state,
            "transformSeed": common_result.umap_random_state,
            "transformBatchSize": UMAP_TRANSFORM_BATCH_SIZE,
        },
        "clusterings": [
            {
                "id": f"fine{cluster_count}",
                "label": "Fine-grained",
                "clusters": cluster_count,
                "labelsFileKey": VISUAL_FILE_KEYS[cluster_count],
                "algorithm": "sklearn.cluster.MiniBatchKMeans",
                "featureSpace": "L2-normalized source SigLIP embedding",
                "randomState": RANDOM_STATE,
                "batchSize": KMEANS_BATCH_SIZE,
                "nInit": KMEANS_N_INIT,
                "maxIter": KMEANS_MAX_ITER,
            }
            for cluster_count in VISUAL_CLUSTER_COUNTS
        ],
    }
    manifest["generatedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    static_bytes = sum(path.stat().st_size for path in output_paths.values())
    print(
        f"[{task.task_id}] visual static export: "
        f"{static_bytes / (1024 * 1024):.2f} MiB, cache hits {cache_hits}/3",
        flush=True,
    )
    return {
        "bytes": static_bytes,
        "cacheHits": cache_hits,
        "embeddingSha256": embedding_sha256,
    }


def visual_export_order(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """Run small Cars, then HICO, and reserve the two large CelebA fits for last."""

    priority = {"cars": 0, "hico": 1, "celeba": 2}
    return sorted(tasks, key=lambda task: (priority[task.dataset], task.order))


def validate_file_spec(bundle: Path, name: str, spec: dict[str, Any]) -> int:
    path = bundle / str(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing for {bundle}: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != int(spec["bytes"]):
        raise ValueError(
            f"{name} byte count mismatch for {bundle}: {observed_bytes} != {spec['bytes']}"
        )
    if file_sha256(path) != str(spec["sha256"]):
        raise ValueError(f"{name} SHA-256 mismatch for {bundle}")
    return observed_bytes


def validate_visual_embedding_contract(
    *,
    task: TaskSpec,
    bundle: Path,
    manifest: dict[str, Any],
    image_ids: list[Any],
    development_mask: np.ndarray[Any, Any],
) -> int:
    from visual_embedding_analysis import (
        ALGORITHM_VERSION,
        KMEANS_BATCH_SIZE,
        KMEANS_MAX_ITER,
        KMEANS_N_INIT,
        RANDOM_STATE,
        UMAP_METRIC,
        UMAP_MIN_DIST,
        UMAP_TRANSFORM_BATCH_SIZE,
    )

    visual = manifest.get("visualEmbedding")
    files = manifest["files"]
    rows = int(manifest["rowCount"])
    if not isinstance(visual, dict):
        raise ValueError(f"Visual embedding contract is missing for {task.task_id}")
    if (
        visual.get("contractVersion") != 1
        or visual.get("available") is not True
        or visual.get("loading") != "lazy-static"
        or visual.get("algorithmVersion") != ALGORITHM_VERSION
        or visual.get("backbone") != "siglip"
        or int(visual.get("embeddingDimension", -1)) <= 1
        or visual.get("fitScope") != "development"
        or visual.get("fitMaskFileKey") != "developmentMask"
        or int(visual.get("fitRowCount", -1)) != int(development_mask.sum())
        or visual.get("groundTruthUsed") is not False
    ):
        raise ValueError(f"Visual embedding scope contract failed for {task.task_id}")

    expected_specs = {
        VISUAL_FILE_KEYS["pca"]: ("float32", [rows, 2], rows * 2 * 4),
        VISUAL_FILE_KEYS["umap"]: ("float32", [rows, 2], rows * 2 * 4),
        **{
            VISUAL_FILE_KEYS[cluster_count]: ("uint8", [rows], rows)
            for cluster_count in VISUAL_CLUSTER_COUNTS
        },
    }
    for file_key, (dtype, shape, byte_count) in expected_specs.items():
        spec = files.get(file_key)
        if (
            not isinstance(spec, dict)
            or spec.get("dtype") != dtype
            or spec.get("layout") != "row-major"
            or spec.get("shape") != shape
            or int(spec.get("bytes", -1)) != byte_count
            or (
                dtype == "float32"
                and spec.get("byteOrder") != "little"
            )
            or (
                dtype == "uint8"
                and spec.get("byteOrder") != "not-applicable"
            )
        ):
            raise ValueError(
                f"Visual file contract failed for {task.task_id}: {file_key}"
            )

    pca_contract = visual.get("pca")
    umap_contract = visual.get("umap")
    explained = pca_contract.get("explainedVarianceRatio") if isinstance(pca_contract, dict) else None
    if (
        not isinstance(pca_contract, dict)
        or pca_contract.get("fileKey") != VISUAL_FILE_KEYS["pca"]
        or pca_contract.get("algorithm") != "sklearn.decomposition.PCA"
        or pca_contract.get("solver") != "randomized"
        or int(pca_contract.get("randomState", -1)) != RANDOM_STATE
        or int(pca_contract.get("reducedDimension", -1)) <= 1
        or not isinstance(explained, list)
        or len(explained) != 2
        or not all(np.isfinite(float(value)) and float(value) >= 0 for value in explained)
    ):
        raise ValueError(f"Visual PCA metadata failed for {task.task_id}")
    if (
        not isinstance(umap_contract, dict)
        or umap_contract.get("fileKey") != VISUAL_FILE_KEYS["umap"]
        or umap_contract.get("algorithm") != "umap-learn"
        or umap_contract.get("input") != "development-fitted-pca"
        or int(umap_contract.get("inputDimension", -1))
        != int(pca_contract["reducedDimension"])
        or int(umap_contract.get("nNeighbors", -1))
        != min(30, int(development_mask.sum()) - 1)
        or float(umap_contract.get("minDist", -1)) != UMAP_MIN_DIST
        or umap_contract.get("metric") != UMAP_METRIC
        or int(umap_contract.get("randomState", -1)) != RANDOM_STATE
        or int(umap_contract.get("transformSeed", -1)) != RANDOM_STATE
        or int(umap_contract.get("transformBatchSize", -1))
        != UMAP_TRANSFORM_BATCH_SIZE
    ):
        raise ValueError(f"Visual UMAP metadata failed for {task.task_id}")

    pca = np.fromfile(
        bundle / str(files[VISUAL_FILE_KEYS["pca"]]["path"]), dtype="<f4"
    )
    umap = np.fromfile(
        bundle / str(files[VISUAL_FILE_KEYS["umap"]]["path"]), dtype="<f4"
    )
    if pca.shape != (rows * 2,) or not np.isfinite(pca).all():
        raise ValueError(f"Visual PCA values failed for {task.task_id}")
    if umap.shape != (rows * 2,) or not np.isfinite(umap).all():
        raise ValueError(f"Visual UMAP values failed for {task.task_id}")

    clusterings = visual.get("clusterings")
    if not isinstance(clusterings, list) or len(clusterings) != len(VISUAL_CLUSTER_COUNTS):
        raise ValueError(f"Visual clustering list failed for {task.task_id}")
    by_k = {
        int(row.get("clusters", -1)): row
        for row in clusterings
        if isinstance(row, dict)
    }
    for cluster_count in VISUAL_CLUSTER_COUNTS:
        row = by_k.get(cluster_count)
        file_key = VISUAL_FILE_KEYS[cluster_count]
        if (
            not isinstance(row, dict)
            or row.get("id") != f"fine{cluster_count}"
            or row.get("label") != "Fine-grained"
            or row.get("labelsFileKey") != file_key
            or row.get("algorithm") != "sklearn.cluster.MiniBatchKMeans"
            or row.get("featureSpace") != "L2-normalized source SigLIP embedding"
            or int(row.get("randomState", -1)) != RANDOM_STATE
            or int(row.get("batchSize", -1)) != KMEANS_BATCH_SIZE
            or int(row.get("nInit", -1)) != KMEANS_N_INIT
            or int(row.get("maxIter", -1)) != KMEANS_MAX_ITER
        ):
            raise ValueError(
                f"Visual K{cluster_count} metadata failed for {task.task_id}"
            )
        labels = np.fromfile(bundle / str(files[file_key]["path"]), dtype=np.uint8)
        if labels.shape != (rows,):
            raise ValueError(
                f"Visual K{cluster_count} row alignment failed for {task.task_id}"
            )
        observed_fit = np.unique(labels[development_mask == 1])
        if not np.array_equal(
            observed_fit, np.arange(cluster_count, dtype=np.uint8)
        ):
            raise ValueError(
                f"Visual K{cluster_count} Development clusters are incomplete for "
                f"{task.task_id}"
            )

    identity = visual.get("inputIdentity")
    canonical_ids_sha256 = hashlib.sha256(
        "\n".join(str(value) for value in image_ids).encode("utf-8")
    ).hexdigest()
    expected_embedding_shape = [rows, int(visual["embeddingDimension"])]
    if (
        not isinstance(identity, dict)
        or identity.get("embeddingFile") != "siglip_embedding.npy"
        or identity.get("embeddingDtype") != "float32"
        or identity.get("embeddingShape") != expected_embedding_shape
        or int(identity.get("embeddingBytes", -1)) <= 0
        or not str(identity.get("embeddingMtimeNs", "")).isdigit()
        or len(str(identity.get("embeddingSha256", ""))) != 64
        or identity.get("developmentMaskSha256")
        != files["developmentMask"]["sha256"]
        or identity.get("canonicalImageIdsSha256") != canonical_ids_sha256
    ):
        raise ValueError(f"Visual input identity failed for {task.task_id}")

    return sum(int(files[file_key]["bytes"]) for file_key in expected_specs)


def resolve_atlas_directory(bundle: Path, manifest: dict[str, Any]) -> Path:
    thumbnails = manifest["thumbnails"]
    return (bundle / str(thumbnails["directory"])).resolve()


def write_metadata_and_manifest(
    *,
    metadata_path: Path,
    manifest_path: Path,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Commit metadata first and refresh its integrity spec in the manifest."""

    write_json(metadata_path, metadata)
    metadata_spec = manifest["files"]["metadata"]
    metadata_spec["bytes"] = metadata_path.stat().st_size
    metadata_spec["sha256"] = file_sha256(metadata_path)
    write_json(manifest_path, manifest)


def _stable_string_list_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_active_supervision_ids_path(task: TaskSpec) -> Path:
    """Resolve the exact saved selection used by the published ProbeBank."""

    if task.explicit_supervision_ids_path is not None:
        explicit = task.explicit_supervision_ids_path.resolve()
        try:
            explicit.relative_to(REPOSITORY_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"Explicit active supervision escapes the repository: {explicit}"
            ) from error
        if not explicit.is_file():
            raise FileNotFoundError(
                f"Explicit active training supervision is missing: {explicit}"
            )
        return explicit

    remote_task_root = (
        REPOSITORY_ROOT
        / "outputs"
        / "formal23_allv2_remote_assets_v1"
        / "tasks"
        / task.task_id
    ).resolve()
    remote_audit_path = remote_task_root / "supervision_audit.json"
    if remote_audit_path.is_file():
        audit = read_json(remote_audit_path)
        selected_value = audit.get("selected_manifest")
        if not isinstance(selected_value, str) or not selected_value:
            raise ValueError(
                f"Remote supervision audit has no selected manifest: {remote_audit_path}"
            )
        selected_path = Path(selected_value).resolve()
        allowed_root = (remote_task_root / "vqa" / "iterative").resolve()
        try:
            selected_path.relative_to(allowed_root)
        except ValueError as error:
            raise ValueError(
                f"Remote selected manifest escapes its training snapshot: {selected_path}"
            ) from error
        if not selected_path.is_file():
            raise FileNotFoundError(
                f"Remote selected manifest is missing: {selected_path}"
            )
        return selected_path

    canonical_root = (
        task.task_root / "supervision" / task.supervision_stage
    ).resolve()
    exact = canonical_root / "split" / "train_labeled_indices.json"
    if exact.is_file():
        return exact
    candidates = (
        sorted(canonical_root.rglob("train_labeled_indices.json"))
        if canonical_root.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        raise ValueError(
            f"Active supervision selection is ambiguous for {task.task_id}: {candidates}"
        )
    fallback = (
        sorted(
            (task.task_root / "supervision").glob(
                "iterative*/split/train_labeled_indices.json"
            )
        )
        if (task.task_root / "supervision").is_dir()
        else []
    )
    if len(fallback) == 1:
        return fallback[0].resolve()
    if len(fallback) > 1:
        raise ValueError(
            f"Fallback active supervision is ambiguous for {task.task_id}: {fallback}"
        )
    raise FileNotFoundError(
        f"Active training supervision is missing for {task.task_id}: {canonical_root}"
    )


def load_active_training_supervision(
    task: TaskSpec,
    *,
    image_ids: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Load the exact image list used by the active learned ProbeBank models."""

    path = resolve_active_supervision_ids_path(task)
    payload = read_json(path)
    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise ValueError(
            f"Active training supervision must be a JSON string list: {path}"
        )
    training_ids = [str(value) for value in payload]
    if len(set(training_ids)) != len(training_ids):
        raise ValueError(
            f"Active training supervision contains duplicate IDs: {path}"
        )
    unknown = sorted(set(training_ids).difference(image_ids))
    if unknown:
        raise ValueError(
            f"Active training supervision is outside {task.task_id}: {unknown[:3]}"
        )

    with np.load(task.score_bundle, allow_pickle=False) as source_scores:
        source_metadata = json.loads(str(source_scores["metadata_json"]))
    score_task = source_metadata.get("task")
    expected_score_task = {
        "key": task.task_id,
        "dataset": task.dataset,
        "taskName": task.task_name,
        "stage": task.probe_stage,
    }
    if not isinstance(score_task, dict) or any(
        score_task.get(key) != value for key, value in expected_score_task.items()
    ):
        raise ValueError(
            f"Audited score-bundle task/stage identity drifted for {task.task_id}: "
            f"{score_task}"
        )
    cache_rows = source_metadata.get("sources", {}).get("learnedScoreCaches", [])
    if not isinstance(cache_rows, list) or not cache_rows:
        raise ValueError(f"ProbeBank cache provenance is missing for {task.task_id}")
    training_records_hash = _stable_string_list_hash(sorted(training_ids))
    seen_cache_metadata: set[Path] = set()
    for row_index, cache_row in enumerate(cache_rows):
        if not isinstance(cache_row, dict):
            raise ValueError(
                f"ProbeBank cache provenance row {row_index} is invalid for {task.task_id}"
            )
        raw_metadata_path = cache_row.get("metadata")
        if not isinstance(raw_metadata_path, str) or not raw_metadata_path:
            raise ValueError(
                f"ProbeBank cache metadata path is missing for {task.task_id} row {row_index}"
            )
        cache_metadata_path = (REPOSITORY_ROOT / raw_metadata_path).resolve()
        try:
            cache_metadata_path.relative_to(REPOSITORY_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"ProbeBank cache metadata escapes the repository: {cache_metadata_path}"
            ) from error
        if cache_metadata_path in seen_cache_metadata:
            raise ValueError(
                f"ProbeBank cache metadata is duplicated for {task.task_id}: "
                f"{cache_metadata_path}"
            )
        seen_cache_metadata.add(cache_metadata_path)
        if not cache_metadata_path.is_file():
            raise FileNotFoundError(
                f"ProbeBank cache metadata is missing for {task.task_id}: "
                f"{cache_metadata_path}"
            )
        cache_metadata = read_json(cache_metadata_path)
        if (
            int(cache_metadata.get("training_count", -1)) != len(training_ids)
            or cache_metadata.get("training_records_hash") != training_records_hash
            or cache_metadata.get("supervision_stage") != task.probe_stage
        ):
            raise ValueError(
                "Active supervision does not match a published ProbeBank cache for "
                f"{task.task_id}: {cache_metadata_path}"
            )
    relative_path = Path(os.path.relpath(path, REPOSITORY_ROOT)).as_posix()
    return training_ids, {
        "logicalStage": task.probe_stage,
        "sourceStage": task.supervision_stage,
        "idsFile": relative_path,
        "rowCount": len(training_ids),
        "sha256": file_sha256(path),
        "trainingRecordsSha256": training_records_hash,
    }


def _resolve_task_contract_file(
    task: TaskSpec,
    relative_path: object,
    *,
    label: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label} must be a non-empty task-relative path")
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError(f"{label} must not be absolute: {requested}")
    task_root = task.task_root.resolve()
    resolved = (task_root / requested).resolve()
    try:
        resolved.relative_to(task_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the task root: {requested}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def validate_source_evaluation_contract(
    task: TaskSpec,
    *,
    image_ids: list[str],
    masks: Any,
    supervision: dict[str, Any],
) -> None:
    """Bind any persisted task Test IDs to the same clean-Test derivation."""

    evaluation_manifest_path = task.task_root / "evaluation" / "manifest.json"
    manifest = read_json(evaluation_manifest_path)
    excluded_count = int(np.count_nonzero(masks.excluded_training))
    if manifest.get("status") != "frozen":
        if excluded_count:
            raise ValueError(
                f"A contaminated candidate Test is not frozen for {task.task_id}"
            )
        return
    if manifest.get("gallery") != "full_database":
        raise ValueError(f"Source Frozen Test is not full-gallery: {task.task_id}")
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("id") != CLEAN_TEST_POLICY:
        raise ValueError(f"Source Frozen Test policy drifted for {task.task_id}")

    clean_ids = sorted(
        image_id
        for image_id, selected in zip(image_ids, masks.frozen_test, strict=True)
        if int(selected)
    )
    candidate_ids = sorted(
        image_id
        for image_id, selected in zip(
            image_ids, masks.candidate_frozen_test, strict=True
        )
        if int(selected)
    )
    excluded_ids = sorted(
        image_id
        for image_id, selected in zip(
            image_ids, masks.excluded_training, strict=True
        )
        if int(selected)
    )
    test_path = _resolve_task_contract_file(
        task,
        manifest.get("test_ids_file"),
        label="Source Frozen Test IDs file",
    )
    excluded_path = _resolve_task_contract_file(
        task,
        manifest.get("excluded_test_ids_file"),
        label="Source excluded Test IDs file",
    )
    if read_json(test_path) != clean_ids:
        raise ValueError(f"Source Frozen Test IDs drifted for {task.task_id}")
    if read_json(excluded_path) != excluded_ids:
        raise ValueError(f"Source excluded Test IDs drifted for {task.task_id}")
    if manifest.get("test_ids_sha256") != file_sha256(test_path):
        raise ValueError(f"Source Frozen Test checksum drifted for {task.task_id}")
    if manifest.get("excluded_test_ids_sha256") != file_sha256(excluded_path):
        raise ValueError(f"Source excluded Test checksum drifted for {task.task_id}")
    if manifest.get("candidate_test_ids_sha256") != hashlib.sha256(
        "\n".join(candidate_ids).encode("utf-8")
    ).hexdigest():
        raise ValueError(f"Source candidate Test checksum drifted for {task.task_id}")
    if int(manifest.get("candidate_test_row_count", -1)) != len(candidate_ids):
        raise ValueError(f"Source candidate Test count drifted for {task.task_id}")
    if int(manifest.get("test_row_count", -1)) != len(clean_ids):
        raise ValueError(f"Source clean Test count drifted for {task.task_id}")
    if int(manifest.get("excluded_training_row_count", -1)) != len(excluded_ids):
        raise ValueError(f"Source excluded Test count drifted for {task.task_id}")
    source_supervision = manifest.get("active_training_supervision")
    if not isinstance(source_supervision, dict):
        raise ValueError(f"Source training provenance is missing for {task.task_id}")
    active_ids_path = (
        REPOSITORY_ROOT / str(supervision["idsFile"])
    ).resolve()
    expected_ids_file = Path(
        os.path.relpath(active_ids_path, task.task_root)
    ).as_posix()
    if (
        source_supervision.get("stage") != supervision["sourceStage"]
        or source_supervision.get("ids_file") != expected_ids_file
        or int(source_supervision.get("row_count", -1)) != supervision["rowCount"]
        or source_supervision.get("sha256") != supervision["sha256"]
    ):
        raise ValueError(f"Source training provenance drifted for {task.task_id}")


def refresh_evaluation_contract(task: TaskSpec) -> None:
    """Refresh disjoint evaluation partitions without rebuilding dense scores."""

    bundle = task.data_root.resolve()
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Browser bundle is missing for {task.task_id}: {bundle}")

    manifest = read_json(manifest_path)
    metadata = read_json(metadata_path)
    files = manifest["files"]
    rows = int(manifest["rowCount"])
    target_count = int(manifest["targetCount"])
    image_ids = [
        str(value)
        for value in read_json(bundle / str(files["imageIds"]["path"]))
    ]
    if len(image_ids) != rows:
        raise ValueError(f"Image ID count mismatch for {task.task_id}")

    query = manifest.get("query")
    if not isinstance(query, dict) or not query.get("images"):
        raise ValueError(f"Fixed Query is required for Frozen Test: {task.task_id}")
    retrieval_targets = list(manifest["retrievalTargets"])
    attribute_indices = [
        index
        for index, target in enumerate(retrieval_targets)
        if str(target.get("kind")) == "attribute"
    ]
    if not attribute_indices:
        raise ValueError(f"No attribute targets are available for {task.task_id}")

    ground_truth = np.fromfile(
        bundle / str(files["groundTruth"]["path"]), dtype=np.uint8
    )
    if ground_truth.size != rows * target_count:
        raise ValueError(f"Ground-truth byte count mismatch for {task.task_id}")
    ground_truth = ground_truth.reshape(rows, target_count)
    query_image_ids = [str(item["imageId"]) for item in query["images"]]
    training_ids, training_supervision = load_active_training_supervision(
        task,
        image_ids=image_ids,
    )
    masks = build_evaluation_masks(
        image_ids=image_ids,
        ground_truth=ground_truth,
        attribute_indices=attribute_indices,
        query_image_ids=query_image_ids,
        training_image_ids=training_ids,
    )
    validate_source_evaluation_contract(
        task,
        image_ids=image_ids,
        masks=masks,
        supervision=training_supervision,
    )
    evaluation = evaluation_contract(
        development_mask=masks.development,
        validation_mask=masks.validation,
        test_mask=masks.frozen_test,
        ground_truth=ground_truth,
        target_ids=[str(target["id"]) for target in retrieval_targets],
        attribute_ids=[
            str(retrieval_targets[index]["id"])
            for index in attribute_indices
        ],
        candidate_test_mask=masks.candidate_frozen_test,
        excluded_training_mask=masks.excluded_training,
        training_supervision=training_supervision,
    )

    for key, filename, mask in (
        ("developmentMask", "development-mask.u8", masks.development),
        ("validationMask", "validation-mask.u8", masks.validation),
        ("testMask", "test-mask.u8", masks.frozen_test),
    ):
        mask_path = bundle / filename
        temporary = mask_path.with_suffix(mask_path.suffix + ".tmp")
        np.ascontiguousarray(mask, dtype=np.uint8).tofile(temporary)
        replace_with_retry(temporary, mask_path)
        files[key] = {
            "path": filename,
            "dtype": "uint8",
            "byteOrder": "not-applicable",
            "layout": "row-major",
            "shape": [rows],
            "bytes": mask_path.stat().st_size,
            "sha256": file_sha256(mask_path),
        }
    metadata["evaluation"] = evaluation
    manifest["evaluation"] = evaluation
    manifest["generatedAt"] = datetime.now(timezone.utc).isoformat()
    write_metadata_and_manifest(
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        metadata=metadata,
        manifest=manifest,
    )
    print(
        f"[{task.task_id}] Development/Validation/Frozen Test: "
        f"{evaluation['development']['rowCount']:,}/"
        f"{evaluation['validation']['rowCount']:,}/"
        f"{evaluation['frozenTest']['rowCount']:,}",
        flush=True,
    )


def validate_bundle(
    task: TaskSpec,
    *,
    expected_target_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    bundle = task.data_root.resolve()
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Browser bundle is missing for {task.task_id}: {bundle}")
    manifest = read_json(manifest_path)
    metadata = read_json(metadata_path)
    if metadata.get("task", {}).get("dataset") != task.dataset:
        raise ValueError(f"Dataset identity mismatch for {task.task_id}")
    if metadata.get("task", {}).get("taskName") != task.task_name:
        raise ValueError(f"Task-name identity mismatch for {task.task_id}")
    if metadata.get("task", {}).get("task") != task.task_id:
        raise ValueError(f"Task-key identity mismatch for {task.task_id}")
    if int(manifest.get("methodCount", -1)) != 14:
        raise ValueError(f"Expected 14 methods for {task.task_id}")
    target_ids = [str(row["id"]) for row in manifest["retrievalTargets"]]
    if "joint" not in target_ids:
        raise ValueError(f"Joint retrieval target is missing for {task.task_id}")
    with np.load(task.score_bundle, allow_pickle=False) as source_scores:
        source_target_ids = [str(value) for value in source_scores["target_ids"].tolist()]
    frozen_target_ids = (
        [str(value) for value in expected_target_ids]
        if expected_target_ids is not None
        else [*EXPECTED_TASKS[task.order][2], "joint"]
    )
    if target_ids != frozen_target_ids:
        raise ValueError(
            f"Frozen modeled retrieval targets drifted for {task.task_id}: "
            f"{target_ids} != {frozen_target_ids}"
        )
    if target_ids != source_target_ids:
        raise ValueError(
            f"Modeled retrieval targets diverge from the audited score bundle for "
            f"{task.task_id}: {target_ids} != {source_target_ids}"
        )
    if metadata.get("task", {}).get("baseAttributes") != [
        target for target in target_ids if target != "joint"
    ]:
        raise ValueError(f"Modeled base-attribute contract mismatch for {task.task_id}")

    files = manifest["files"]
    validated_bytes = {
        key: validate_file_spec(bundle, key, spec)
        for key, spec in files.items()
    }
    fine_cluster_contract = (
        ("fine30", "fine30Labels", 30),
        ("fine50", "fine50Labels", 50),
        ("fine100", "fine100Labels", 100),
    )
    cluster_schemes = {
        str(scheme.get("id")): scheme
        for scheme in manifest.get("clusters", {}).get("schemes", [])
    }
    if manifest.get("clusters", {}).get("defaultScheme") != "fine50":
        raise ValueError(f"Fine K50 must remain the default for {task.task_id}")
    fine_clusterings = metadata.get("fineClusterings")
    if not isinstance(fine_clusterings, dict):
        raise ValueError(f"Fine clustering metadata is missing for {task.task_id}")
    for scheme_id, file_key, cluster_count in fine_cluster_contract:
        if file_key not in files:
            raise ValueError(f"Fine K{cluster_count} labels are missing for {task.task_id}")
        scheme = cluster_schemes.get(scheme_id)
        if (
            not isinstance(scheme, dict)
            or int(scheme.get("clusters", -1)) != cluster_count
            or str(scheme.get("labelsFileKey")) != file_key
            or str(scheme.get("family")) != "fine"
        ):
            raise ValueError(
                f"Fine K{cluster_count} manifest contract failed for {task.task_id}"
            )
        spec = files[file_key]
        labels = np.fromfile(bundle / str(spec["path"]), dtype=np.uint8)
        if labels.shape != (int(manifest["rowCount"]),):
            raise ValueError(f"Fine K{cluster_count} row alignment failed for {task.task_id}")
        observed = np.unique(labels)
        if not np.array_equal(observed, np.arange(cluster_count, dtype=np.uint8)):
            raise ValueError(f"Fine K{cluster_count} labels are not contiguous for {task.task_id}")
        summaries = [
            row
            for row in metadata.get("clusterSummary", [])
            if str(row.get("scheme")) == scheme_id
        ]
        summary_ids = sorted(int(row.get("cluster_id", -1)) for row in summaries)
        if (
            len(summaries) != cluster_count
            or summary_ids != list(range(cluster_count))
            or sum(int(row.get("size", 0)) for row in summaries) != int(manifest["rowCount"])
        ):
            raise ValueError(f"Fine K{cluster_count} summaries failed for {task.task_id}")
        summary_counts = np.asarray(
            [int(row.get("size", 0)) for row in sorted(summaries, key=lambda row: int(row["cluster_id"]))]
        )
        if not np.array_equal(summary_counts, np.bincount(labels, minlength=cluster_count)):
            raise ValueError(f"Fine K{cluster_count} summary sizes drifted for {task.task_id}")
        clustering = fine_clusterings.get(scheme_id)
        if (
            not isinstance(clustering, dict)
            or int(clustering.get("clusters", -1)) != cluster_count
            or clustering.get("basisTarget") != "joint"
            or clustering.get("groundTruthUsedForClustering") is not False
            or (
                "fitScope" in clustering
                and (
                    clustering.get("fitScope") != "development"
                    or clustering.get("fitMaskFileKey") != "developmentMask"
                    or int(clustering.get("fitRowCount", -1)) != int(
                        manifest.get("evaluation", {}).get("development", {}).get("rowCount", -2)
                    )
                )
            )
        ):
            raise ValueError(f"Fine K{cluster_count} metadata contract failed for {task.task_id}")

    image_ids = read_json(bundle / str(files["imageIds"]["path"]))
    if len(image_ids) != int(manifest["rowCount"]):
        raise ValueError(f"Image ID count mismatch for {task.task_id}")

    query = manifest.get("query")
    if not isinstance(query, dict) or query.get("mode") != "fixed":
        raise ValueError(f"Fixed query contract is missing for {task.task_id}")
    if not query.get("images"):
        raise ValueError(f"Fixed query contains no images for {task.task_id}")
    query_bytes = 0
    for item in query.get("images", []):
        path = bundle / str(item["path"])
        index = int(item["imageIndex"])
        if not path.is_file() or image_ids[index] != item["imageId"]:
            raise ValueError(f"Query alignment failed for {task.task_id}: {item}")
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Query byte count failed for {task.task_id}: {path}")
        if file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"Query SHA-256 failed for {task.task_id}: {path}")
        query_bytes += path.stat().st_size

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict) or metadata.get("evaluation") != evaluation:
        raise ValueError(f"Evaluation contract mismatch for {task.task_id}")
    for key in ("developmentMask", "validationMask", "testMask"):
        if key not in files:
            raise ValueError(f"{key} is missing for {task.task_id}")
    development_mask = np.fromfile(
        bundle / str(files["developmentMask"]["path"]), dtype=np.uint8
    )
    validation_mask = np.fromfile(
        bundle / str(files["validationMask"]["path"]), dtype=np.uint8
    )
    test_mask = np.fromfile(bundle / str(files["testMask"]["path"]), dtype=np.uint8)
    for name, mask in (
        ("Development", development_mask),
        ("Validation", validation_mask),
        ("Frozen Test", test_mask),
    ):
        if mask.shape != (int(manifest["rowCount"]),) or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{name} mask contract failed for {task.task_id}")
    ground_truth = np.fromfile(
        bundle / str(files["groundTruth"]["path"]), dtype=np.uint8
    ).reshape(int(manifest["rowCount"]), int(manifest["targetCount"]))
    attribute_indices = [
        index
        for index, target in enumerate(manifest["retrievalTargets"])
        if str(target.get("kind")) == "attribute"
    ]
    ordered_image_ids = [str(value) for value in image_ids]
    training_ids, training_supervision = load_active_training_supervision(
        task,
        image_ids=ordered_image_ids,
    )
    expected_masks = build_evaluation_masks(
        image_ids=ordered_image_ids,
        ground_truth=ground_truth,
        attribute_indices=attribute_indices,
        query_image_ids=[str(item["imageId"]) for item in query["images"]],
        training_image_ids=training_ids,
    )
    validate_source_evaluation_contract(
        task,
        image_ids=ordered_image_ids,
        masks=expected_masks,
        supervision=training_supervision,
    )
    if not np.array_equal(test_mask, expected_masks.frozen_test):
        raise ValueError(f"Frozen Test membership drifted for {task.task_id}")
    if not np.array_equal(development_mask, expected_masks.development):
        raise ValueError(f"Development membership drifted for {task.task_id}")
    if not np.array_equal(validation_mask, expected_masks.validation):
        raise ValueError(f"Validation membership drifted for {task.task_id}")
    expected_evaluation = evaluation_contract(
        development_mask=development_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        ground_truth=ground_truth,
        target_ids=target_ids,
        attribute_ids=[target_ids[index] for index in attribute_indices],
        candidate_test_mask=expected_masks.candidate_frozen_test,
        excluded_training_mask=expected_masks.excluded_training,
        training_supervision=training_supervision,
    )
    if evaluation != expected_evaluation:
        raise ValueError(f"Evaluation audit counts drifted for {task.task_id}")

    visual_embedding_bytes = validate_visual_embedding_contract(
        task=task,
        bundle=bundle,
        manifest=manifest,
        image_ids=image_ids,
        development_mask=development_mask,
    )

    thumbnails = manifest.get("thumbnails", {})
    if not thumbnails.get("available"):
        raise ValueError(f"Thumbnail atlas is unavailable for {task.task_id}")
    if int(thumbnails.get("missingSourceCount", -1)) != 0:
        raise ValueError(f"Thumbnail atlas contains placeholders for {task.task_id}")
    shared_ids_hash = thumbnails.get("sharedImageIdsSha256")
    if shared_ids_hash is not None and shared_ids_hash != files["imageIds"]["sha256"]:
        raise ValueError(f"Shared atlas image-ID hash mismatch for {task.task_id}")
    atlas_dir = resolve_atlas_directory(bundle, manifest)
    atlas_count = int(thumbnails["atlasCount"])
    atlas_paths = [atlas_dir / f"atlas-{index:04d}.webp" for index in range(atlas_count)]
    missing_atlases = [str(path) for path in atlas_paths if not path.is_file()]
    if missing_atlases:
        raise FileNotFoundError(
            f"{len(missing_atlases)} thumbnail atlases are missing for {task.task_id}; "
            f"first={missing_atlases[0]}"
        )
    thumbnail_bytes = sum(path.stat().st_size for path in atlas_paths)
    if thumbnail_bytes != int(thumbnails["bytes"]):
        raise ValueError(f"Thumbnail byte count mismatch for {task.task_id}")

    initial_keys = [*INITIAL_FILE_KEYS]
    if files.get("umap2d") and manifest.get("projections", {}).get("umap", {}).get("available"):
        initial_keys.append("umap2d")
    initial_download_bytes = manifest_path.stat().st_size + sum(
        validated_bytes[key] for key in initial_keys
    )
    owned_paths = {manifest_path.resolve(), metadata_path.resolve()}
    owned_paths.update(
        (bundle / str(spec["path"])).resolve() for spec in files.values()
    )
    owned_paths.update((bundle / str(item["path"])).resolve() for item in query.get("images", []))
    try:
        atlas_dir.relative_to(bundle)
    except ValueError:
        pass
    else:
        owned_paths.update(path.resolve() for path in atlas_paths)
    bundle_bytes = sum(path.stat().st_size for path in owned_paths)
    return {
        "rowCount": int(manifest["rowCount"]),
        "targetCount": int(manifest["targetCount"]),
        "queryBytes": query_bytes,
        "initialDownloadBytes": initial_download_bytes,
        "bundleBytes": bundle_bytes,
        "thumbnailBytes": thumbnail_bytes,
        "visualEmbeddingBytes": visual_embedding_bytes,
        "retrievalTargets": manifest["retrievalTargets"],
        "developmentRowCount": int(evaluation["development"]["rowCount"]),
        "developmentPositiveCounts": evaluation["development"]["positiveCounts"],
        "validationRowCount": int(evaluation["validation"]["rowCount"]),
        "validationPositiveCounts": evaluation["validation"]["positiveCounts"],
        "testRowCount": int(evaluation["frozenTest"]["rowCount"]),
        "testPositiveCounts": evaluation["frozenTest"]["positiveCounts"],
    }


def migrate_legacy_owned_atlas(task: TaskSpec, source_task: TaskSpec) -> bool:
    """Point a legacy duplicate atlas at the dataset owner, then remove the copy.

    Older Cars exports predate cross-task atlas reuse, so order 002 can contain
    a byte-for-byte copy of order 001's atlas.  Migration is deliberately
    conservative: ordered image IDs, atlas geometry, byte totals, individual
    atlas hashes, and both browser JSON documents must agree before the local
    generated directory is removed.
    """

    bundle = task.data_root.resolve()
    source_bundle = source_task.data_root.resolve()
    manifest_path = bundle / "manifest.json"
    metadata_path = bundle / "metadata.json"
    source_manifest_path = source_bundle / "manifest.json"
    if not (manifest_path.is_file() and metadata_path.is_file() and source_manifest_path.is_file()):
        return False

    manifest = read_json(manifest_path)
    metadata = read_json(metadata_path)
    source_manifest = read_json(source_manifest_path)
    thumbnails = manifest.get("thumbnails", {})
    source_thumbnails = source_manifest.get("thumbnails", {})
    image_spec = manifest["files"]["imageIds"]
    source_image_spec = source_manifest["files"]["imageIds"]
    current_atlas_dir = resolve_atlas_directory(bundle, manifest)
    source_atlas_dir = resolve_atlas_directory(source_bundle, source_manifest)
    if current_atlas_dir == source_atlas_dir:
        if image_spec["sha256"] != source_image_spec["sha256"]:
            raise ValueError(f"Ordered image IDs differ for atlas-sharing task {task.task_id}")
        # The low-level exporter records the source directory basename.  Use
        # the stable task identity in browser contracts, including HICO whose
        # canonical bundle lives directly at public/data (basename `data`).
        shared_fields = {
            "sharedFrom": source_task.task_id,
            "sharedImageIdsSha256": str(image_spec["sha256"]),
        }
        changed = any(thumbnails.get(key) != value for key, value in shared_fields.items())
        if changed:
            manifest["thumbnails"].update(shared_fields)
            metadata.setdefault("thumbnails", {}).update(shared_fields)
        # Rewriting is intentional even when only a prior interrupted
        # migration left the metadata integrity spec stale.
        write_metadata_and_manifest(
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            metadata=metadata,
            manifest=manifest,
        )
        # Recover safely if an earlier migration committed the shared JSON
        # contract but stopped before deleting the now-unreferenced copy.
        redundant_dir = (bundle / "atlases").resolve()
        if redundant_dir != source_atlas_dir and redundant_dir.is_dir():
            if redundant_dir.parent != bundle or redundant_dir.name != "atlases":
                raise ValueError(f"Unsafe redundant atlas cleanup target: {redundant_dir}")
            atlas_count = int(source_thumbnails["atlasCount"])
            expected_names = {f"atlas-{index:04d}.webp" for index in range(atlas_count)}
            observed_paths = list(redundant_dir.iterdir())
            if any(not path.is_file() for path in observed_paths):
                raise ValueError(f"Redundant atlas directory contains a subdirectory: {redundant_dir}")
            if {path.name for path in observed_paths} != expected_names:
                raise ValueError(f"Redundant atlas directory contains unexpected files: {redundant_dir}")
            for name in sorted(expected_names):
                redundant_path = redundant_dir / name
                source_path = source_atlas_dir / name
                if not source_path.is_file():
                    raise FileNotFoundError(f"Shared atlas owner is missing {source_path}")
                if redundant_path.stat().st_size != source_path.stat().st_size:
                    raise ValueError(f"Redundant atlas byte count differs: {redundant_path}")
                if file_sha256(redundant_path) != file_sha256(source_path):
                    raise ValueError(f"Redundant atlas contents differ: {redundant_path}")
            validate_bundle(task)
            shutil.rmtree(redundant_dir)
            print(
                f"[{task.task_id}] removed {atlas_count} validated redundant atlas files",
                flush=True,
            )
            changed = True
        return changed
    if str(thumbnails.get("directory")) != "atlases":
        raise ValueError(
            f"Refusing to migrate unexpected atlas directory for {task.task_id}: "
            f"{thumbnails.get('directory')!r}"
        )

    if image_spec["sha256"] != source_image_spec["sha256"]:
        raise ValueError(f"Ordered image IDs differ for atlas-sharing task {task.task_id}")
    if int(manifest["rowCount"]) != int(source_manifest["rowCount"]):
        raise ValueError(f"Row counts differ for atlas-sharing task {task.task_id}")

    geometry_keys = (
        "kind",
        "format",
        "tileWidth",
        "tileHeight",
        "columns",
        "rows",
        "itemsPerAtlas",
        "atlasWidth",
        "atlasHeight",
        "atlasCount",
        "imageCount",
        "bytes",
    )
    for key in geometry_keys:
        if thumbnails.get(key) != source_thumbnails.get(key):
            raise ValueError(
                f"Atlas geometry differs for {task.task_id} at {key}: "
                f"{thumbnails.get(key)!r} != {source_thumbnails.get(key)!r}"
            )
    if int(thumbnails.get("missingSourceCount", -1)) != 0:
        raise ValueError(f"Legacy atlas has missing source images for {task.task_id}")
    if int(source_thumbnails.get("missingSourceCount", -1)) != 0:
        raise ValueError(f"Shared atlas owner has missing source images for {source_task.task_id}")

    local_atlas_dir = (bundle / "atlases").resolve()
    if local_atlas_dir.parent != bundle or local_atlas_dir.name != "atlases":
        raise ValueError(f"Unsafe legacy atlas cleanup target: {local_atlas_dir}")
    atlas_count = int(thumbnails["atlasCount"])
    local_atlases = [local_atlas_dir / f"atlas-{index:04d}.webp" for index in range(atlas_count)]
    source_atlases = [source_atlas_dir / f"atlas-{index:04d}.webp" for index in range(atlas_count)]
    for local_path, source_path in zip(local_atlases, source_atlases, strict=True):
        if not local_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(
                f"Atlas migration source/copy is missing: {local_path}, {source_path}"
            )
        if local_path.stat().st_size != source_path.stat().st_size:
            raise ValueError(f"Atlas byte count differs during migration: {local_path}")
        if file_sha256(local_path) != file_sha256(source_path):
            raise ValueError(f"Atlas contents differ during migration: {local_path}")

    relative_directory = Path(os.path.relpath(source_atlas_dir, bundle)).as_posix()
    shared_fields = {
        "directory": relative_directory,
        "pathPattern": f"{relative_directory}/atlas-{{atlas:04d}}.webp",
        "generatedAtlases": 0,
        "reusedAtlases": atlas_count,
        "sharedFrom": source_task.task_id,
        "sharedImageIdsSha256": str(image_spec["sha256"]),
    }
    manifest["thumbnails"].update(shared_fields)
    metadata.setdefault("thumbnails", {}).update(shared_fields)
    write_metadata_and_manifest(
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        metadata=metadata,
        manifest=manifest,
    )

    # Validate the now-shared browser contract before deleting the redundant
    # generated copy.  The exact target was resolved and constrained above.
    validate_bundle(task)
    shutil.rmtree(local_atlas_dir)
    print(
        f"[{task.task_id}] migrated duplicate atlases to {source_task.task_id} "
        f"and removed {atlas_count} redundant files",
        flush=True,
    )
    return True


def migrate_legacy_shared_atlases(tasks: list[TaskSpec], tasks_by_order: dict[int, TaskSpec]) -> None:
    for task in tasks:
        canonical_order = {"cars": 1, "celeba": 6, "hico": 8}[task.dataset]
        if task.order != canonical_order:
            migrate_legacy_owned_atlas(task, tasks_by_order[canonical_order])


def build_catalog(
    tasks: list[TaskSpec],
    *,
    write: bool = True,
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for dataset_id in ("cars", "celeba", "hico"):
        entries: list[dict[str, Any]] = []
        for task in (item for item in tasks if item.dataset == dataset_id):
            audit = validate_bundle(task)
            retrieval_targets = audit.pop("retrievalTargets")
            target_labels = [
                str(target["label"])
                for target in retrieval_targets
                if str(target["id"]) != "joint"
            ]
            entries.append(
                {
                    "id": task.task_id,
                    "label": TASK_LABELS[task.order],
                    "description": ", ".join(target_labels) + ", and Joint retrieval targets",
                    "dataRoot": task.public_data_root,
                    "defaultRetrievalTarget": "joint",
                    "declaredAttributes": list(task.attributes),
                    "modeledRetrievalTargets": [
                        {
                            "id": str(target["id"]),
                            "label": str(target["label"]),
                        }
                        for target in retrieval_targets
                        if str(target["id"]) != "joint"
                    ],
                    "modeledTargetIds": [
                        str(target["id"])
                        for target in retrieval_targets
                        if str(target["id"]) != "joint"
                    ],
                    **audit,
                }
            )
        datasets.append(
            {
                "id": dataset_id,
                "label": DATASET_LABELS[dataset_id],
                "tasks": entries,
            }
        )
    formal_catalog = {
        "schemaVersion": 2,
        "defaultDataset": "hico",
        "defaultTask": "032_hico_task_hico_hugging_cat",
        "taskCount": len(tasks),
        "datasets": datasets,
    }
    if write:
        from addon_catalog import (
            atomic_write_catalog,
            catalog_lock,
            merge_published_addons,
        )

        catalog_path = PUBLIC_DATA_ROOT / "catalog.json"
        with catalog_lock(catalog_path):
            published_catalog = (
                read_json(catalog_path) if catalog_path.is_file() else None
            )
            catalog = merge_published_addons(formal_catalog, published_catalog)
            atomic_write_catalog(catalog_path, catalog)
        return catalog
    return formal_catalog


def validate_catalog(tasks: list[TaskSpec]) -> dict[str, Any]:
    """Validate every Formal bundle and its exact published catalog entries."""

    from addon_catalog import validate_formal_catalog_extension

    catalog_path = PUBLIC_DATA_ROOT / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Main17 catalog is missing: {catalog_path}")
    expected = build_catalog(tasks, write=False)
    observed = read_json(catalog_path)
    validate_formal_catalog_extension(expected, observed)
    return observed


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.visual_only and (args.validate_only or args.refresh_evaluation_only):
        raise ValueError(
            "--visual-only cannot be combined with --validate-only or "
            "--refresh-evaluation-only"
        )
    if args.refresh_evaluation_only and (args.validate_only or args.force):
        raise ValueError(
            "--refresh-evaluation-only cannot be combined with --validate-only or --force"
        )
    raw_orders = [value.strip() for value in str(args.orders).split(",") if value.strip()]
    tasks = select_tasks(int(value) for value in raw_orders)
    tasks_by_order = {task.order: task for task in select_tasks(MAIN17_ORDERS)}

    if args.visual_only:
        for task in visual_export_order(tasks):
            export_visual_embedding_bundle(task)
        if tuple(task.order for task in tasks) != MAIN17_ORDERS:
            for task in tasks:
                validate_bundle(task)
            print("Visual subset complete; catalog is unchanged.")
            return 0
        catalog = build_catalog(tasks)
        total_visual = sum(
            int(task["visualEmbeddingBytes"])
            for dataset in catalog["datasets"]
            for task in dataset["tasks"]
        )
        print(
            json.dumps(
                {
                    "catalog": str(PUBLIC_DATA_ROOT / "catalog.json"),
                    "tasks": catalog["taskCount"],
                    "visualEmbeddingBytes": total_visual,
                    "mode": "visual-only-static-export",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.validate_only:
        if tuple(task.order for task in tasks) != MAIN17_ORDERS:
            for task in tasks:
                validate_bundle(task)
            print("Subset validated; catalog is unchanged because it must remain all-or-nothing.")
            return 0
        catalog = validate_catalog(tasks)
        total_initial = sum(
            int(task["initialDownloadBytes"])
            for dataset in catalog["datasets"]
            for task in dataset["tasks"]
        )
        print(
            json.dumps(
                {
                    "catalog": str(PUBLIC_DATA_ROOT / "catalog.json"),
                    "tasks": catalog["taskCount"],
                    "formalTasks": len(tasks),
                    "catalogTasks": catalog["taskCount"],
                    "totalInitialDownloadBytes": total_initial,
                    "readOnly": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.refresh_evaluation_only:
        for task in tasks:
            refresh_evaluation_contract(task)
        for task in visual_export_order(tasks):
            export_visual_embedding_bundle(task)
        if tuple(task.order for task in tasks) != MAIN17_ORDERS:
            print("Subset complete; catalog is unchanged because it must remain all-or-nothing.")
            return 0
        migrate_legacy_shared_atlases(tasks, tasks_by_order)
        catalog = build_catalog(tasks)
        total_initial = sum(
            int(task["initialDownloadBytes"])
            for dataset in catalog["datasets"]
            for task in dataset["tasks"]
        )
        print(
            json.dumps(
                {
                    "catalog": str(PUBLIC_DATA_ROOT / "catalog.json"),
                    "tasks": catalog["taskCount"],
                    "totalInitialDownloadBytes": total_initial,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    # Canonical atlas owners must exist before aligned sibling tasks can reuse them.
    requested_by_order = {task.order: task for task in tasks}
    for order in (1, 32, 5):
        task = requested_by_order.get(order)
        if task is not None:
            process_task(task, tasks_by_order, args.force)
    remaining = [task for task in tasks if task.order not in {1, 5, 32}]
    with ThreadPoolExecutor(max_workers=min(args.workers, len(remaining) or 1)) as pool:
        futures = {
            pool.submit(process_task, task, tasks_by_order, args.force): task
            for task in remaining
        }
        for future in as_completed(futures):
            print(f"[{future.result()}] complete", flush=True)

    # Visual embeddings are intentionally processed sequentially.  CelebA's
    # normalized Development matrix is large, and parallel UMAP fits would
    # multiply memory without improving the static browser contract.
    for task in visual_export_order(tasks):
        export_visual_embedding_bundle(task)

    if tuple(task.order for task in tasks) != MAIN17_ORDERS:
        print("Subset complete; catalog is unchanged because it must remain all-or-nothing.")
        return 0
    migrate_legacy_shared_atlases(tasks, tasks_by_order)
    catalog = build_catalog(tasks)
    total_initial = sum(
        int(task["initialDownloadBytes"])
        for dataset in catalog["datasets"]
        for task in dataset["tasks"]
    )
    print(
        json.dumps(
            {
                "catalog": str(PUBLIC_DATA_ROOT / "catalog.json"),
                "tasks": catalog["taskCount"],
                "totalInitialDownloadBytes": total_initial,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
