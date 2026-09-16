#!/usr/bin/env python3
"""Export the PCP/cluster analysis into browser-friendly static assets.

The exporter deliberately keeps the dense arrays out of JSON.  Every binary
array is little-endian, C-contiguous (row-major), and shares the same first
dimension / image index.  ``manifest.json`` is the contract consumed by the
web application.

The only required third-party dependency is NumPy.  Pillow is required when
thumbnail atlases are enabled.  A genuine UMAP projection is emitted only
when ``umap-learn`` is importable; otherwise the manifest records why it is
unavailable.  No substitute projection is ever labelled as UMAP.

Typical usage, from the ``web`` directory::

    uv run --project ../../../probe_learning python ../export_multitarget_scores.py
    python scripts/export_web_data.py

Install ``umap-learn`` into the script-local vendor directory and add only
``umap-2d.f32`` without touching the existing atlases::

    python -m pip install --target scripts/.vendor umap-learn
    python scripts/export_web_data.py --umap-only --require-umap --force-umap
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / ".vendor"
# A local target install keeps the analysis exporter reproducible without
# mutating the user's global Python environment.  Add it before importing
# NumPy as umap-learn may have placed its compiled dependencies there too.
if VENDOR_DIR.is_dir():
    vendor_path = str(VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np

from evaluation_scope import (
    CLEAN_TEST_POLICY,
    FROZEN_TEST_FRACTION,
    FROZEN_TEST_SEED,
    build_evaluation_masks,
    evaluation_contract,
)


WEB_ROOT = SCRIPT_DIR.parent
ANALYSIS_ROOT = WEB_ROOT.parent
REPOSITORY_ROOT = ANALYSIS_ROOT.parents[1]

DEFAULT_HTML = (
    ANALYSIS_ROOT
    / "pcp_result"
    / "032_hico_hugging_cat_joint_mean_rank.html"
)
DEFAULT_SCORE_BUNDLE = (
    ANALYSIS_ROOT
    / "pcp_result"
    / "032_hico_hugging_cat_multitarget_scores.npz"
)
DEFAULT_CLUSTER_DIR = ANALYSIS_ROOT / "cluster_result"
DEFAULT_IMAGE_ROOT = REPOSITORY_ROOT / "dataset" / "raw" / "HICO" / "images"
DEFAULT_OUTPUT = WEB_ROOT / "public" / "data"

METRIC_COLUMNS = [
    "absolute_distance",
    "absolute_margin",
    "shape_distance",
    "shape_margin",
    "mean_rank",
    "rank_std",
    "fixed_mean",
    "learned_mean",
    "learned_minus_fixed",
]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcp-html", type=Path, default=DEFAULT_HTML)
    parser.add_argument(
        "--score-bundle",
        type=Path,
        default=DEFAULT_SCORE_BUNDLE,
        help=(
            "Aligned multi-target NPZ produced by export_multitarget_scores.py. "
            "It must contain raw/calibrated scores and ranks in "
            "[image, method, target] order."
        ),
    )
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=None,
        help="Task folder containing task.json/query_ids.json; inferred by default.",
    )
    parser.add_argument(
        "--active-training-supervision-file",
        type=Path,
        default=None,
        help=(
            "Explicit active learner training-ID JSON. Required for a full export "
            "when the task has no frozen clean-Test contract."
        ),
    )
    parser.add_argument(
        "--active-training-supervision-stage",
        type=str,
        default=None,
        help="Audit label for --active-training-supervision-file.",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        default=None,
        help="Optional fixed-query label override.",
    )
    parser.add_argument(
        "--output",
        "--output-data-dir",
        dest="output",
        type=Path,
        default=None,
        help=(
            "Browser data directory. Non-default tasks must set this explicitly "
            "so the current HICO bundle is never overwritten accidentally."
        ),
    )
    parser.add_argument(
        "--umap-only",
        action="store_true",
        help=(
            "Read the existing manifest/ranks and only generate UMAP plus an "
            "updated manifest; source files and thumbnail atlases are untouched."
        ),
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Only refresh fixed query pictures and the manifest query contract.",
    )
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument(
        "--require-umap",
        action="store_true",
        help="Fail instead of recording UMAP as unavailable.",
    )
    parser.add_argument(
        "--force-umap",
        action="store_true",
        help="Recompute UMAP even when a correctly sized output already exists.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-atlases", action="store_true")
    parser.add_argument(
        "--reuse-atlases-from",
        type=Path,
        default=None,
        help=(
            "Reuse a previously exported bundle's atlas set. The source bundle "
            "must have exactly the same ordered image IDs and atlas geometry."
        ),
    )
    parser.add_argument("--force-atlases", action="store_true")
    parser.add_argument("--atlas-workers", type=int, default=4)
    parser.add_argument("--atlas-columns", type=int, default=10)
    parser.add_argument("--atlas-rows", type=int, default=10)
    parser.add_argument("--tile-width", type=int, default=96)
    parser.add_argument("--tile-height", type=int, default=72)
    parser.add_argument("--webp-quality", type=int, default=62)
    return parser.parse_args()


def extract_pcp_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script\s+id=["\']pcp-data["\']\s+type=["\']application/json["\']>(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Could not find #pcp-data JSON in {path}")
    payload = json.loads(match.group(1))
    required = {"rows", "columns", "methods", "images", "rankFloat32"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"PCP payload is missing fields: {sorted(missing)}")
    return payload


def decode_pcp_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    rows = int(payload["rows"])
    columns = int(payload["columns"])
    raw = base64.b64decode(payload["rankFloat32"], validate=True)
    ranks = np.frombuffer(raw, dtype="<f4")
    expected = rows * columns
    if ranks.size != expected:
        raise ValueError(f"Rank data has {ranks.size} values; expected {expected}")
    ranks = np.ascontiguousarray(ranks.reshape(rows, columns), dtype=np.float32)
    if not np.isfinite(ranks).all():
        raise ValueError("Rank matrix contains NaN or infinity")

    priority = None
    encoded_priority = payload.get("priorityUint32")
    if encoded_priority:
        priority_raw = base64.b64decode(encoded_priority, validate=True)
        priority = np.frombuffer(priority_raw, dtype="<u4").copy()
        if priority.shape != (rows,):
            raise ValueError(
                f"Priority order has shape {priority.shape}; expected {(rows,)}"
            )
        if np.unique(priority).size != rows or int(priority.max()) >= rows:
            raise ValueError("Priority order is not a permutation of image indices")
    return ranks, priority


def load_multitarget_scores(
    path: Path,
    image_ids: Sequence[str],
    methods: Sequence[str],
    joint_ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Load and strictly align the multi-target score cube.

    The legacy PCP payload remains the independent audit anchor for the Joint
    slice.  This prevents a stale or differently ordered score bundle from
    silently changing the existing analysis.
    """

    with np.load(path, allow_pickle=False) as data:
        required = {
            "image_ids",
            "methods",
            "target_ids",
            "raw_scores",
            "calibrated_scores",
            "ranks",
            "metadata_json",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Multi-target score bundle is missing fields: {sorted(missing)}"
            )
        bundle_image_ids = [str(value) for value in data["image_ids"].tolist()]
        bundle_methods = [str(value) for value in data["methods"].tolist()]
        target_ids = [str(value) for value in data["target_ids"].tolist()]
        raw_scores = np.asarray(data["raw_scores"], dtype=np.float32)
        calibrated_scores = np.asarray(data["calibrated_scores"], dtype=np.float32)
        ranks = np.asarray(data["ranks"], dtype=np.float32)
        metadata_json = str(np.asarray(data["metadata_json"]).item())

    if bundle_image_ids != list(image_ids):
        raise ValueError("Multi-target image IDs do not align with the PCP payload")
    if bundle_methods != list(methods):
        raise ValueError("Multi-target method order does not align with the PCP payload")
    if len(set(target_ids)) != len(target_ids) or "joint" not in target_ids:
        raise ValueError(f"Invalid retrieval targets in score bundle: {target_ids}")

    expected_shape = (len(image_ids), len(methods), len(target_ids))
    for name, values in (
        ("raw_scores", raw_scores),
        ("calibrated_scores", calibrated_scores),
        ("ranks", ranks),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinity")

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json in score bundle is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json in score bundle must contain an object")

    joint_index = target_ids.index("joint")
    bundle_joint = ranks[:, :, joint_index]
    if not np.allclose(bundle_joint, joint_ranks, rtol=0.0, atol=1e-5):
        max_delta = float(np.max(np.abs(bundle_joint - joint_ranks)))
        raise ValueError(
            "Multi-target Joint ranks differ from the audited PCP payload "
            f"(maximum delta {max_delta})"
        )
    # scipy/numpy version changes can alter mean-rank float rounding by a few
    # ULPs.  Keep the previously audited Joint bytes as the canonical slice so
    # cluster labels, projections, and the browser rank cube remain identical.
    ranks = np.array(ranks, dtype=np.float32, order="C", copy=True)
    ranks[:, :, joint_index] = joint_ranks
    return raw_scores, calibrated_scores, ranks, target_ids, metadata


def retrieval_target_definitions(
    score_metadata: dict[str, Any],
    target_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Build manifest retrieval targets while preserving canonical GT labels."""

    configured = score_metadata.get("retrievalTargets")
    if isinstance(configured, list) and [
        str(item.get("id")) for item in configured if isinstance(item, dict)
    ] == list(target_ids):
        return [dict(item) for item in configured]

    audit_targets = score_metadata.get("targets")
    if isinstance(audit_targets, list):
        by_id = {
            str(item.get("id")): item
            for item in audit_targets
            if isinstance(item, dict) and item.get("id") is not None
        }
        if all(target_id in by_id for target_id in target_ids):
            base_targets = [target_id for target_id in target_ids if target_id != "joint"]
            definitions: list[dict[str, Any]] = []
            for target_id in target_ids:
                source = by_id[target_id]
                if target_id == "joint":
                    members = source.get("attributes", base_targets)
                    definitions.append(
                        {
                            "id": "joint",
                            "label": str(source.get("label", "Joint")),
                            "kind": "derived",
                            "members": [str(value) for value in members],
                            "rule": str(source.get("rule", "product")),
                        }
                    )
                else:
                    canonical = str(source.get("canonicalAttribute", target_id))
                    default_label = canonical.replace("_", " ").title()
                    definitions.append(
                        {
                            "id": target_id,
                            "label": str(source.get("label", default_label)),
                            "kind": "attribute",
                            "canonicalAttribute": canonical,
                        }
                    )
            return definitions

    base_targets = [target_id for target_id in target_ids if target_id != "joint"]
    return [
        (
            {
                "id": "joint",
                "label": "Joint",
                "kind": "derived",
                "members": base_targets,
                "rule": "product",
            }
            if target_id == "joint"
            else {
                "id": target_id,
                "label": target_id.replace("_", " ").title(),
                "kind": "attribute",
                "canonicalAttribute": target_id,
            }
        )
        for target_id in target_ids
    ]


def _humanize_task_name(task_name: str) -> str:
    value = task_name.removeprefix("task_").replace("_", " ").strip()
    return value[:1].upper() + value[1:] if value else "Fixed query"


class ImageSourceResolver:
    """Resolve gallery IDs beneath one image root without guessing ambiguities.

    Most datasets store the gallery ID as a path relative to ``image_root``.
    AwA2 instead keeps bare basenames in its records while the source files live
    below class directories.  Preserve the direct-path fast path and build one
    recursive basename index only when that fallback is actually needed.
    """

    def __init__(self, image_root: Path):
        self.image_root = image_root.resolve()
        self._basename_index: dict[str, Path] | None = None
        self._ambiguous_basenames: set[str] = set()

    def _build_basename_index(self) -> None:
        unique: dict[str, Path] = {}
        ambiguous: set[str] = set()
        for path in self.image_root.rglob("*"):
            if not path.is_file():
                continue
            basename = path.name
            if basename in ambiguous:
                continue
            if basename in unique:
                del unique[basename]
                ambiguous.add(basename)
                continue
            unique[basename] = path.resolve()
        self._basename_index = unique
        self._ambiguous_basenames = ambiguous

    def resolve(self, image_id: str) -> Path | None:
        relative = Path(str(image_id).replace("\\", "/"))
        direct = (self.image_root / relative).resolve()
        try:
            direct.relative_to(self.image_root)
        except ValueError:
            direct = self.image_root / relative.name
        if direct.is_file():
            return direct

        if self._basename_index is None:
            self._build_basename_index()
        basename = relative.name
        if basename in self._ambiguous_basenames:
            raise ValueError(
                f"Image source basename {basename!r} is ambiguous under "
                f"{self.image_root}"
            )
        return self._basename_index.get(basename)


def export_fixed_query(
    *,
    output_dir: Path,
    image_root: Path,
    image_ids: Sequence[str],
    dataset: str,
    task_name: str,
    task_dir_override: Path | None,
    query_text_override: str | None,
    image_resolver: ImageSourceResolver | None = None,
) -> dict[str, Any] | None:
    """Copy the task's ordered, immutable query pictures into its web bundle."""

    task_dir = resolve_task_directory(dataset, task_name, task_dir_override)
    task_config: dict[str, Any] = {}
    task_json_path = task_dir / "task.json"
    if task_json_path.is_file():
        loaded = json.loads(task_json_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            task_config = loaded

    # The immutable gallery-aligned IDs are authoritative. Some historical
    # task.json files also carry temporary VQA upload filenames in
    # ``query_files``; those names are not necessarily gallery records.
    query_ids_name = str(task_config.get("query_ids_file", "query_ids.json"))
    query_ids_path = task_dir / query_ids_name
    configured = None
    if query_ids_path.is_file():
        query_payload = json.loads(query_ids_path.read_text(encoding="utf-8"))
        configured = query_payload.get("image_ids") if isinstance(query_payload, dict) else None
    if not isinstance(configured, list) or not configured:
        configured = task_config.get("query_files")
    if not isinstance(configured, list) or not configured:
        print(
            f"Warning: no fixed query configuration found in {task_dir}",
            file=sys.stderr,
        )
        return None
    if not isinstance(configured, list) or not configured:
        print(f"Warning: fixed query is empty in {task_dir}", file=sys.stderr)
        return None

    exact_index = {image_id: index for index, image_id in enumerate(image_ids)}
    basename_index: dict[str, tuple[str, int]] = {}
    duplicate_basenames: set[str] = set()
    for index, image_id in enumerate(image_ids):
        basename = Path(image_id).name
        if basename in basename_index:
            duplicate_basenames.add(basename)
        else:
            basename_index[basename] = (image_id, index)

    resolver = image_resolver or ImageSourceResolver(image_root)
    query_dir = output_dir / "query-pics"
    query_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for configured_id in (str(value) for value in configured):
        configured_basename = Path(configured_id).name
        normalized_basename = re.sub(
            r"^query_\d+_", "", configured_basename, flags=re.IGNORECASE
        )
        identity_candidates = list(
            dict.fromkeys((configured_id, configured_basename, normalized_basename))
        )
        exact_match = next(
            (candidate for candidate in identity_candidates if candidate in exact_index),
            None,
        )
        if exact_match is not None:
            image_id = exact_match
            image_index = exact_index[exact_match]
        else:
            basename_match = next(
                (
                    candidate
                    for candidate in identity_candidates
                    if candidate not in duplicate_basenames and candidate in basename_index
                ),
                None,
            )
            if basename_match is None:
                raise ValueError(
                    f"Query image {configured_id!r} cannot be uniquely aligned with gallery IDs"
                )
            image_id, image_index = basename_index[basename_match]
        if image_id in seen_ids:
            raise ValueError(f"Duplicate fixed query image: {image_id}")
        seen_ids.add(image_id)

        source = resolver.resolve(image_id)
        if source is None:
            raise FileNotFoundError(
                f"Fixed query source does not exist for {configured_id!r} "
                f"(gallery ID {image_id!r}) under {resolver.image_root}"
            )
        destination_name = Path(image_id).name
        destination = query_dir / destination_name
        shutil.copy2(source, destination)
        relative_path = destination.relative_to(output_dir).as_posix()
        records.append(
            {
                "path": relative_path,
                "imageId": image_id,
                "imageIndex": image_index,
                "mimeType": mimetypes.guess_type(destination.name)[0] or "application/octet-stream",
                "bytes": destination.stat().st_size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    query_text = (
        query_text_override
        or task_config.get("query_text")
        or task_config.get("joint_label")
        or _humanize_task_name(task_name)
    )
    return {
        "mode": "fixed",
        "text": str(query_text),
        "images": records,
    }


def resolve_task_directory(
    dataset: str,
    task_name: str,
    task_dir_override: Path | None,
) -> Path:
    return (
        task_dir_override.resolve()
        if task_dir_override is not None
        else (REPOSITORY_ROOT / "dataset" / "tasks" / dataset / task_name).resolve()
    )


def _strict_supervision_list(
    path: Path,
    *,
    image_ids: Sequence[str],
    expected_sha256: object,
    expected_count: object,
) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Active training supervision is missing: {path}")
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 != observed_sha256:
        raise ValueError(f"Active training supervision checksum drifted: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
        raise ValueError(f"Active training supervision must be a JSON string list: {path}")
    training_ids = [str(value) for value in payload]
    if len(set(training_ids)) != len(training_ids):
        raise ValueError(f"Active training supervision contains duplicate IDs: {path}")
    if int(expected_count) != len(training_ids):
        raise ValueError(f"Active training supervision count drifted: {path}")
    unknown = sorted(set(training_ids).difference(str(value) for value in image_ids))
    if unknown:
        raise ValueError(f"Active training supervision is outside the gallery: {unknown[:3]}")
    return training_ids


def _sorted_image_id_hash(values: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    ).hexdigest()


def _stable_training_id_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def training_supervision_from_existing_evaluation(
    evaluation: Any,
    *,
    image_ids: Sequence[str],
) -> tuple[list[str], dict[str, Any] | None]:
    isolation = evaluation.get("testIsolation") if isinstance(evaluation, dict) else None
    if not isinstance(isolation, dict):
        raise ValueError(
            "Query-only refresh requires an existing audited Test-isolation contract; "
            "run a full export first"
        )
    provenance = (
        isolation.get("activeTrainingSupervision")
        if isinstance(isolation, dict)
        else None
    )
    if provenance is None:
        raise ValueError(
            "Query-only refresh requires active-training provenance; run a full export first"
        )
    if not isinstance(provenance, dict):
        raise ValueError("Existing active-training provenance is invalid")
    raw_path = provenance.get("idsFile")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Existing active-training provenance has no IDs file")
    path = (REPOSITORY_ROOT / raw_path).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Active-training IDs escape the repository: {path}") from error
    return (
        _strict_supervision_list(
            path,
            image_ids=image_ids,
            expected_sha256=provenance.get("sha256"),
            expected_count=provenance.get("rowCount"),
        ),
        dict(provenance),
    )


def training_supervision_from_task_contract(
    task_dir: Path,
    *,
    image_ids: Sequence[str],
) -> tuple[list[str], dict[str, Any] | None, dict[str, Any] | None]:
    manifest_path = task_dir / "evaluation" / "manifest.json"
    if not manifest_path.is_file():
        return [], None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        return [], None, None
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("id") != CLEAN_TEST_POLICY:
        raise ValueError(f"Unsupported source Frozen Test policy: {manifest_path}")
    source = manifest.get("active_training_supervision")
    if not isinstance(source, dict):
        raise ValueError(f"Source active-training provenance is missing: {manifest_path}")
    raw_path = source.get("ids_file")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Source active-training IDs file is missing: {manifest_path}")
    path = (task_dir / raw_path).resolve()
    try:
        path.relative_to(task_dir.resolve())
    except ValueError as error:
        raise ValueError(f"Source active-training IDs escape the task: {path}") from error
    training_ids = _strict_supervision_list(
        path,
        image_ids=image_ids,
        expected_sha256=source.get("sha256"),
        expected_count=source.get("row_count"),
    )
    source_stage = str(source.get("stage", "")).strip()
    logical_stage = (
        "iterative"
        if source_stage == "iterative" or source_stage.startswith("iterative_")
        else "two_stage"
        if source_stage == "two_stage"
        else None
    )
    if logical_stage is None:
        raise ValueError(
            f"Unsupported source active-training stage {source_stage!r}: {manifest_path}"
        )
    return training_ids, {
        "logicalStage": logical_stage,
        "sourceStage": source_stage,
        "idsFile": Path(os.path.relpath(path, REPOSITORY_ROOT)).as_posix(),
        "rowCount": len(training_ids),
        "sha256": str(source["sha256"]),
        "trainingRecordsSha256": _stable_training_id_hash(training_ids),
    }, manifest


def training_supervision_from_explicit_file(
    path: Path,
    *,
    stage: str | None,
    image_ids: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Explicit active-training IDs escape the repository: {resolved}"
        ) from error
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError(
            "--active-training-supervision-stage is required with the explicit IDs file"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Active training supervision is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(
            f"Active training supervision must be a JSON string list: {resolved}"
        )
    observed_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    training_ids = _strict_supervision_list(
        resolved,
        image_ids=image_ids,
        expected_sha256=observed_sha256,
        expected_count=len(payload),
    )
    source_stage = stage.strip()
    logical_stage = (
        "iterative"
        if source_stage == "iterative" or source_stage.startswith("iterative_")
        else "two_stage"
        if source_stage == "two_stage"
        else None
    )
    if logical_stage is None:
        raise ValueError(
            f"Unsupported explicit active-training stage: {source_stage!r}"
        )
    return training_ids, {
        "logicalStage": logical_stage,
        "sourceStage": source_stage,
        "idsFile": Path(os.path.relpath(resolved, REPOSITORY_ROOT)).as_posix(),
        "rowCount": len(training_ids),
        "sha256": observed_sha256,
        "trainingRecordsSha256": _stable_training_id_hash(training_ids),
    }


def validate_source_clean_test(
    task_dir: Path,
    source_manifest: dict[str, Any] | None,
    *,
    image_ids: Sequence[str],
    masks: Any,
) -> None:
    if source_manifest is None:
        return
    policy = source_manifest.get("policy")
    expected_policy = {
        "id": CLEAN_TEST_POLICY,
        "candidate_split_seed": FROZEN_TEST_SEED,
        "candidate_test_fraction": FROZEN_TEST_FRACTION,
        "stratified_by": "task_attribute_gt_tuple",
        "query_excluded": True,
        "no_replacement": True,
        "excluded_rows_destination": "development",
    }
    if not isinstance(policy, dict):
        raise ValueError("Source Frozen Test policy is missing")
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(
                f"Source Frozen Test policy field {key!r} drifted: "
                f"{policy.get(key)!r} != {expected!r}"
            )
    candidate_ids = sorted(
        str(image_id)
        for image_id, selected in zip(
            image_ids, masks.candidate_frozen_test, strict=True
        )
        if int(selected)
    )
    excluded_ids = sorted(
        str(image_id)
        for image_id, selected in zip(image_ids, masks.excluded_training, strict=True)
        if int(selected)
    )
    raw_path = source_manifest.get("test_ids_file")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Source Frozen Test IDs file is missing")
    path = (task_dir / raw_path).resolve()
    try:
        path.relative_to(task_dir.resolve())
    except ValueError as error:
        raise ValueError(f"Source Frozen Test IDs escape the task: {path}") from error
    clean_ids = sorted(
        str(image_id)
        for image_id, selected in zip(image_ids, masks.frozen_test, strict=True)
        if int(selected)
    )
    if json.loads(path.read_text(encoding="utf-8")) != clean_ids:
        raise ValueError(f"Source Frozen Test IDs do not match the clean split: {path}")
    if source_manifest.get("test_ids_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"Source Frozen Test checksum drifted: {path}")
    if int(source_manifest.get("candidate_test_row_count", -1)) != len(candidate_ids):
        raise ValueError("Source candidate Frozen Test count drifted")
    if source_manifest.get("candidate_test_ids_sha256") != _sorted_image_id_hash(
        candidate_ids
    ):
        raise ValueError("Source candidate Frozen Test checksum drifted")
    if int(source_manifest.get("test_row_count", -1)) != len(clean_ids):
        raise ValueError("Source clean Frozen Test count drifted")
    if int(source_manifest.get("excluded_training_row_count", -1)) != len(
        excluded_ids
    ):
        raise ValueError("Source excluded Frozen Test count drifted")
    raw_excluded_path = source_manifest.get("excluded_test_ids_file")
    if not isinstance(raw_excluded_path, str) or not raw_excluded_path:
        raise ValueError("Source excluded Frozen Test IDs file is missing")
    excluded_path = (task_dir / raw_excluded_path).resolve()
    try:
        excluded_path.relative_to(task_dir.resolve())
    except ValueError as error:
        raise ValueError(
            f"Source excluded Frozen Test IDs escape the task: {excluded_path}"
        ) from error
    if json.loads(excluded_path.read_text(encoding="utf-8")) != excluded_ids:
        raise ValueError(
            f"Source excluded Frozen Test IDs do not match the split: {excluded_path}"
        )
    if source_manifest.get("excluded_test_ids_sha256") != hashlib.sha256(
        excluded_path.read_bytes()
    ).hexdigest():
        raise ValueError(
            f"Source excluded Frozen Test checksum drifted: {excluded_path}"
        )


def refresh_query_only(args: argparse.Namespace, output_dir: Path) -> int:
    manifest_path = output_dir / "manifest.json"
    metadata_path = output_dir / "metadata.json"
    image_ids_path = output_dir / "image-ids.json"
    if not manifest_path.is_file() or not metadata_path.is_file() or not image_ids_path.is_file():
        raise FileNotFoundError(
            f"Query-only refresh requires manifest.json, metadata.json, and image-ids.json in {output_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_ids = json.loads(image_ids_path.read_text(encoding="utf-8"))
    task = metadata.get("task", {})
    query = export_fixed_query(
        output_dir=output_dir,
        image_root=args.image_root,
        image_ids=[str(value) for value in image_ids],
        dataset=str(task.get("dataset", "")),
        task_name=str(task.get("taskName", "")),
        task_dir_override=args.task_dir,
        query_text_override=args.query_text,
        image_resolver=ImageSourceResolver(args.image_root),
    )
    if query is None:
        manifest.pop("query", None)
        manifest.pop("evaluation", None)
        metadata.pop("evaluation", None)
        for key in ("developmentMask", "validationMask", "testMask"):
            manifest.get("files", {}).pop(key, None)
    else:
        manifest["query"] = query
        rows = int(manifest["rowCount"])
        target_count = int(manifest["targetCount"])
        files = manifest["files"]
        ground_truth = np.fromfile(
            output_dir / str(files["groundTruth"]["path"]), dtype=np.uint8
        ).reshape(rows, target_count)
        retrieval_targets = list(manifest["retrievalTargets"])
        attribute_indices = [
            index
            for index, target in enumerate(retrieval_targets)
            if str(target.get("kind")) == "attribute"
        ]
        training_ids, training_supervision = (
            training_supervision_from_existing_evaluation(
                manifest.get("evaluation"),
                image_ids=[str(value) for value in image_ids],
            )
        )
        masks = build_evaluation_masks(
            image_ids=[str(value) for value in image_ids],
            ground_truth=ground_truth,
            attribute_indices=attribute_indices,
            query_image_ids=[str(item["imageId"]) for item in query["images"]],
            training_image_ids=training_ids,
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
        for key, mask in (
            ("developmentMask", masks.development),
            ("validationMask", masks.validation),
            ("testMask", masks.frozen_test),
        ):
            existing_spec = files.get(key)
            if not isinstance(existing_spec, dict) or not isinstance(
                existing_spec.get("path"), str
            ):
                raise ValueError(
                    f"Query-only refresh requires an existing {key} file contract"
                )
            existing_path = output_dir / str(existing_spec["path"])
            existing_mask = np.fromfile(existing_path, dtype=np.uint8)
            if existing_mask.shape != (rows,) or not np.array_equal(
                existing_mask, mask
            ):
                raise ValueError(
                    "Query-only refresh would change the audited evaluation masks; "
                    "run a full export instead"
                )
        for key, filename, mask in (
            ("developmentMask", "development-mask.u8", masks.development),
            ("validationMask", "validation-mask.u8", masks.validation),
            ("testMask", "test-mask.u8", masks.frozen_test),
        ):
            write_array(output_dir / filename, mask, "u1")
            files[key] = file_spec(output_dir, filename, "uint8", [rows])
        manifest["evaluation"] = evaluation
        metadata["evaluation"] = evaluation

    write_json(metadata_path, metadata)
    manifest["files"]["metadata"] = json_file_spec(output_dir, "metadata.json", [1])
    manifest["generatedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(f"Updated fixed query contract: {manifest_path}")
    return 0


def load_cluster_data(
    cluster_dir: Path,
    image_ids: Sequence[str],
    ranks: np.ndarray,
    methods: Sequence[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    path = cluster_dir / "cluster_data.npz"
    with np.load(path, allow_pickle=False) as data:
        cluster_ranks = np.asarray(data["ranks"], dtype=np.float32)
        cluster_ids = [str(value) for value in data["image_ids"].tolist()]
        cluster_methods = [str(value) for value in data["methods"].tolist()]
        absolute_labels = np.asarray(data["absolute_labels"], dtype=np.uint8)
        shape_labels = np.asarray(data["shape_labels"], dtype=np.uint8)

    if cluster_ids != list(image_ids):
        raise ValueError("cluster_data.npz image_ids do not align with PCP image IDs")
    if cluster_methods != list(methods[: len(cluster_methods)]):
        raise ValueError("cluster_data.npz method order does not match the PCP data")
    if cluster_ranks.shape != (len(image_ids), len(cluster_methods)):
        raise ValueError(f"Unexpected cluster rank shape: {cluster_ranks.shape}")
    if not np.array_equal(cluster_ranks, ranks[:, : len(cluster_methods)]):
        max_delta = float(np.max(np.abs(cluster_ranks - ranks[:, : len(cluster_methods)])))
        raise ValueError(f"Cluster and PCP ranks differ (maximum delta {max_delta})")
    if absolute_labels.shape != (len(image_ids),):
        raise ValueError(f"Unexpected absolute label shape: {absolute_labels.shape}")
    if shape_labels.shape != (len(image_ids),):
        raise ValueError(f"Unexpected shape label shape: {shape_labels.shape}")
    return cluster_methods, cluster_ranks, absolute_labels, shape_labels


def load_assignments(
    path: Path,
    image_ids: Sequence[str],
    absolute_labels: np.ndarray,
    shape_labels: np.ndarray,
    retrieval_targets: Sequence[dict[str, Any]],
    target_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(image_ids)
    metrics = np.empty((count, len(METRIC_COLUMNS)), dtype=np.float32)
    ground_truth = np.empty((count, len(target_ids)), dtype=np.uint8)
    target_by_id = {str(item.get("id")): item for item in retrieval_targets}
    if list(target_by_id) != list(target_ids):
        raise ValueError(
            "Retrieval target definitions do not align with score target IDs: "
            f"{list(target_by_id)} versus {list(target_ids)}"
        )
    gt_fields = []
    for target_id in target_ids:
        definition = target_by_id[target_id]
        canonical = (
            "joint"
            if target_id == "joint"
            else str(definition.get("canonicalAttribute", target_id))
        )
        gt_fields.append(f"gt_{canonical}")
    seen = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_gt = [field for field in gt_fields if field not in (reader.fieldnames or [])]
        if missing_gt:
            raise ValueError(
                f"image_assignments.csv is missing target GT columns: {missing_gt}"
            )
        for expected_index, row in enumerate(reader):
            if expected_index >= count:
                raise ValueError("image_assignments.csv has more rows than the PCP data")
            index = int(row["image_index"])
            if index != expected_index:
                raise ValueError(
                    f"image_assignments.csv row {expected_index} has image_index {index}"
                )
            if row["image_id"] != image_ids[index]:
                raise ValueError(f"Image ID mismatch at row {index}")
            if int(row["absolute_cluster"]) != int(absolute_labels[index]):
                raise ValueError(f"Absolute cluster mismatch at row {index}")
            if int(row["shape_cluster"]) != int(shape_labels[index]):
                raise ValueError(f"Shape cluster mismatch at row {index}")
            metrics[index] = [float(row[column]) for column in METRIC_COLUMNS]
            ground_truth[index] = [int(row[field]) for field in gt_fields]
            seen += 1
    if seen != count:
        raise ValueError(f"image_assignments.csv has {seen} rows; expected {count}")
    if not np.isfinite(metrics).all():
        raise ValueError("Metrics contain NaN or infinity")
    return metrics, ground_truth


def parse_summary(path: Path) -> list[dict[str, Any]]:
    integer_fields = {
        "cluster_id",
        "size",
        "joint_positive_count",
    }
    float_fields = {
        "fraction",
        "mean_rank",
        "fixed_mean",
        "learned_mean",
        "learned_minus_fixed",
        "mean_within_image_rank_std",
        "joint_positive_rate",
        "joint_positive_enrichment",
    }
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item: dict[str, Any] = {}
            for key, value in row.items():
                if key in integer_fields:
                    item[key] = int(value)
                elif key in float_fields:
                    item[key] = float(value)
                elif key == "attribute_positive_count_json":
                    item["attribute_positive_count"] = json.loads(value)
                elif key in {"top_methods", "bottom_methods"}:
                    item[key] = [part.strip() for part in value.split("|") if part.strip()]
                else:
                    item[key] = value
            records.append(item)
    return records


def parse_representatives(path: Path, atlas_config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    per_atlas = int(atlas_config["itemsPerAtlas"])
    columns = int(atlas_config["columns"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["image_index"])
            cell = index % per_atlas
            records.append(
                {
                    "scheme": row["scheme"],
                    "clusterId": int(row["cluster_id"]),
                    "clusterLabel": row["cluster_label"],
                    "clusterLabelZh": row["cluster_label_zh"],
                    "sampleGroup": row["sample_group"],
                    "sampleRank": int(row["sample_rank"]),
                    "imageIndex": index,
                    "imageId": row["image_id"],
                    "atlas": index // per_atlas,
                    "cell": cell,
                    "column": cell % columns,
                    "row": cell // columns,
                    "distance": float(row["distance"]),
                    "margin": float(row["margin"]),
                    "meanRank": float(row["mean_rank"]),
                    "rankStd": float(row["rank_std"]),
                    "fixedMean": float(row["fixed_mean"]),
                    "learnedMean": float(row["learned_mean"]),
                    "jointGt": int(row["joint_gt"]),
                    "attributeGt": json.loads(row["attribute_gt_json"]),
                }
            )
    return records


def standardized_features(
    ranks: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(ranks, dtype=np.float64)
    fit_values = values
    if fit_mask is not None:
        normalized_mask = np.asarray(fit_mask, dtype=np.uint8).reshape(-1)
        if normalized_mask.shape != (len(values),) or not np.isin(normalized_mask, (0, 1)).all():
            raise ValueError("Projection fit mask must be a row-aligned binary vector")
        fit_values = values[normalized_mask.astype(bool)]
        if len(fit_values) < 2:
            raise ValueError("Projection fit scope must contain at least two rows")
    mean = fit_values.mean(axis=0)
    scale = fit_values.std(axis=0)
    if np.any(scale <= np.finfo(np.float64).eps):
        bad = np.flatnonzero(scale <= np.finfo(np.float64).eps).tolist()
        raise ValueError(f"Cannot standardize constant rank columns: {bad}")
    standardized = ((values - mean) / scale).astype(np.float32)
    return standardized, mean.astype(np.float32), scale.astype(np.float32)


def compute_pca(
    features: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    # With 13 columns, the covariance eigendecomposition is much faster and
    # more memory efficient than a full SVD of the 47k x 13 matrix.
    values = np.asarray(features, dtype=np.float64)
    fit_values = values
    if fit_mask is not None:
        normalized_mask = np.asarray(fit_mask, dtype=np.uint8).reshape(-1)
        if normalized_mask.shape != (len(values),) or not np.isin(normalized_mask, (0, 1)).all():
            raise ValueError("PCA fit mask must be a row-aligned binary vector")
        fit_values = values[normalized_mask.astype(bool)]
    covariance = fit_values.T @ fit_values / max(1, fit_values.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:2]].copy()
    # Resolve the arbitrary eigenvector sign for byte-stable reruns.
    for component_index in range(components.shape[1]):
        vector = components[:, component_index]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            components[:, component_index] *= -1
    coordinates = np.asarray(values @ components, dtype=np.float32)
    total = float(eigenvalues.sum())
    explained = (
        [float(value / total) for value in eigenvalues[:2]]
        if total > 0
        else [0.0, 0.0]
    )
    return coordinates, explained


def compute_or_load_umap(
    features: np.ndarray,
    output_path: Path,
    args: argparse.Namespace,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    fit_scope = "development" if fit_mask is not None else "all rows"
    fit_mask_file_key = "developmentMask" if fit_mask is not None else None
    fit_row_count = (
        int(np.count_nonzero(np.asarray(fit_mask, dtype=np.uint8)))
        if fit_mask is not None
        else int(len(features))
    )
    assignment_scope = (
        "Development fit_transform; Validation/Frozen Test/Query transform"
        if fit_mask is not None
        else "all rows via fit_transform"
    )
    parameters = {
        "nComponents": 2,
        "nNeighbors": int(args.umap_neighbors),
        "minDist": float(args.umap_min_dist),
        "metric": "euclidean",
        "randomState": int(args.seed),
    }
    if args.skip_umap:
        return None, {
            "available": False,
            "algorithm": "UMAP",
            "reason": "Skipped by --skip-umap.",
            "parameters": parameters,
            "fitScope": fit_scope,
            "fitMaskFileKey": fit_mask_file_key,
            "fitRowCount": fit_row_count,
            "assignmentScope": assignment_scope,
        }

    try:
        from umap import UMAP  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        reason = (
            "umap-learn is not installed in this Python environment; "
            "from the web directory install it with "
            "`python -m pip install --target scripts/.vendor umap-learn` "
            "and rerun. "
            f"Import error: {exc}"
        )
        if args.require_umap:
            raise RuntimeError(reason) from exc
        return None, {
            "available": False,
            "algorithm": "UMAP",
            "reason": reason,
            "parameters": parameters,
            "fitScope": fit_scope,
            "fitMaskFileKey": fit_mask_file_key,
            "fitRowCount": fit_row_count,
            "assignmentScope": assignment_scope,
        }

    expected_bytes = features.shape[0] * 2 * np.dtype("<f4").itemsize
    if (
        fit_mask is None
        and output_path.is_file()
        and output_path.stat().st_size == expected_bytes
        and not args.force_umap
    ):
        cached = np.fromfile(output_path, dtype="<f4").reshape(features.shape[0], 2)
        if np.isfinite(cached).all():
            print(f"Reusing existing UMAP coordinates: {output_path}")
            return cached, {
                "available": True,
                "algorithm": "UMAP",
                "implementation": "umap-learn",
                "parameters": parameters,
                "reused": True,
            }

    print("Computing genuine UMAP coordinates; this may take several minutes ...")
    model = UMAP(
        n_components=2,
        n_neighbors=int(args.umap_neighbors),
        min_dist=float(args.umap_min_dist),
        metric="euclidean",
        random_state=int(args.seed),
        transform_seed=int(args.seed),
        n_jobs=1,
        low_memory=True,
    )
    if fit_mask is None:
        coordinates = np.asarray(model.fit_transform(features), dtype=np.float32)
    else:
        normalized_mask = np.asarray(fit_mask, dtype=np.uint8).reshape(-1)
        fit_indices = np.flatnonzero(normalized_mask)
        held_out_indices = np.flatnonzero(1 - normalized_mask)
        coordinates = np.empty((len(features), 2), dtype=np.float32)
        coordinates[fit_indices] = np.asarray(
            model.fit_transform(features[fit_indices]), dtype=np.float32
        )
        if len(held_out_indices):
            coordinates[held_out_indices] = np.asarray(
                model.transform(features[held_out_indices]), dtype=np.float32
            )
    if coordinates.shape != (features.shape[0], 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"UMAP returned an invalid array with shape {coordinates.shape}")
    return coordinates, {
        "available": True,
        "algorithm": "UMAP",
        "implementation": "umap-learn",
        "implementationVersion": _package_version("umap-learn"),
        "parameters": parameters,
        "fitScope": fit_scope,
        "fitMaskFileKey": fit_mask_file_key,
        "fitRowCount": fit_row_count,
        "assignmentScope": assignment_scope,
        "reused": False,
    }


def _package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_array(path: Path, array: np.ndarray, dtype: str) -> None:
    normalized = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    atomic_write_bytes(path, normalized.tobytes(order="C"))


def write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    atomic_write_bytes(path, encoded)


def file_spec(
    output_dir: Path,
    filename: str,
    dtype: str,
    shape: Sequence[int],
    *,
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    path = output_dir / filename
    spec: dict[str, Any] = {
        "path": filename,
        "dtype": dtype,
        "byteOrder": "little" if dtype != "uint8" else "not-applicable",
        "layout": "row-major",
        "shape": [int(value) for value in shape],
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if columns is not None:
        spec["columns"] = list(columns)
    return spec


def json_file_spec(output_dir: Path, filename: str, shape: Sequence[int]) -> dict[str, Any]:
    path = output_dir / filename
    return {
        "path": filename,
        "encoding": "utf-8-json",
        "shape": [int(value) for value in shape],
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def atlas_config(args: argparse.Namespace, image_count: int) -> dict[str, Any]:
    per_atlas = int(args.atlas_columns * args.atlas_rows)
    return {
        "kind": "fixed-grid-atlas",
        "format": "webp",
        "directory": "atlases",
        "pathPattern": "atlases/atlas-{atlas:04d}.webp",
        "tileWidth": int(args.tile_width),
        "tileHeight": int(args.tile_height),
        "columns": int(args.atlas_columns),
        "rows": int(args.atlas_rows),
        "itemsPerAtlas": per_atlas,
        "atlasWidth": int(args.tile_width * args.atlas_columns),
        "atlasHeight": int(args.tile_height * args.atlas_rows),
        "atlasCount": int(math.ceil(image_count / per_atlas)),
        "imageCount": int(image_count),
        "mapping": {
            "atlas": "Math.floor(imageIndex / itemsPerAtlas)",
            "cell": "imageIndex % itemsPerAtlas",
            "column": "cell % columns",
            "row": "Math.floor(cell / columns)",
            "sourceX": "column * tileWidth",
            "sourceY": "row * tileHeight",
        },
    }


def _valid_cached_atlas(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.format == "WEBP" and image.size == expected_size
    except Exception:
        return False


def build_atlases(
    image_ids: Sequence[str],
    image_root: Path,
    output_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    image_resolver: ImageSourceResolver | None = None,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageOps, features
    except ImportError as exc:
        raise RuntimeError("Pillow is required to generate thumbnail atlases") from exc
    if not features.check("webp"):
        raise RuntimeError("This Pillow build does not support WebP")

    atlas_dir = output_dir / str(config["directory"])
    atlas_dir.mkdir(parents=True, exist_ok=True)
    per_atlas = int(config["itemsPerAtlas"])
    atlas_count = int(config["atlasCount"])
    columns = int(config["columns"])
    tile_size = (int(config["tileWidth"]), int(config["tileHeight"]))
    atlas_size = (int(config["atlasWidth"]), int(config["atlasHeight"]))
    resolver = image_resolver or ImageSourceResolver(image_root)
    # Resolve before starting worker threads so the recursive fallback index is
    # constructed at most once and basename ambiguity fails deterministically.
    source_paths = [resolver.resolve(image_id) for image_id in image_ids]

    def create_one(atlas_index: int) -> tuple[int, bool, list[int], int]:
        target = atlas_dir / f"atlas-{atlas_index:04d}.webp"
        if not args.force_atlases and _valid_cached_atlas(target, atlas_size):
            return atlas_index, True, [], target.stat().st_size

        canvas = Image.new("RGB", atlas_size, (236, 235, 232))
        missing: list[int] = []
        start = atlas_index * per_atlas
        stop = min(len(image_ids), start + per_atlas)
        for image_index in range(start, stop):
            cell = image_index - start
            x = (cell % columns) * tile_size[0]
            y = (cell // columns) * tile_size[1]
            source = source_paths[image_index]
            try:
                if source is None:
                    raise FileNotFoundError(image_ids[image_index])
                with Image.open(source) as raw:
                    corrected = ImageOps.exif_transpose(raw)
                    thumbnail = ImageOps.fit(
                        corrected.convert("RGB"),
                        tile_size,
                        method=Image.Resampling.LANCZOS,
                        centering=(0.5, 0.5),
                    )
                canvas.paste(thumbnail, (x, y))
            except (FileNotFoundError, OSError, ValueError):
                missing.append(image_index)
                draw = ImageDraw.Draw(canvas)
                draw.rectangle(
                    (x, y, x + tile_size[0] - 1, y + tile_size[1] - 1),
                    fill=(226, 223, 217),
                    outline=(191, 188, 181),
                )
                draw.line(
                    (x + 8, y + 8, x + tile_size[0] - 9, y + tile_size[1] - 9),
                    fill=(190, 79, 72),
                    width=2,
                )
                draw.line(
                    (x + tile_size[0] - 9, y + 8, x + 8, y + tile_size[1] - 9),
                    fill=(190, 79, 72),
                    width=2,
                )

        temporary = target.with_name(target.name + ".tmp")
        canvas.save(
            temporary,
            format="WEBP",
            quality=int(args.webp_quality),
            method=4,
            exact=False,
        )
        os.replace(temporary, target)
        return atlas_index, False, missing, target.stat().st_size

    workers = max(1, min(int(args.atlas_workers), atlas_count))
    generated = 0
    reused = 0
    missing_indices: list[int] = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(create_one, index) for index in range(atlas_count)]
        completed = 0
        for future in as_completed(futures):
            _, was_reused, missing, byte_count = future.result()
            completed += 1
            reused += int(was_reused)
            generated += int(not was_reused)
            missing_indices.extend(missing)
            total_bytes += byte_count
            if completed == atlas_count or completed % 25 == 0:
                print(
                    f"Thumbnail atlases: {completed}/{atlas_count} "
                    f"({total_bytes / (1024 * 1024):.1f} MiB so far)"
                )

    return {
        "available": True,
        "generatedAtlases": generated,
        "reusedAtlases": reused,
        "missingSourceCount": len(missing_indices),
        "missingSourceIndices": sorted(missing_indices),
        "bytes": total_bytes,
        "webpQuality": int(args.webp_quality),
    }


def reuse_atlases(
    source_bundle: Path,
    image_ids: Sequence[str],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate and reference an atlas set shared by an aligned task bundle."""

    source_bundle = source_bundle.resolve()
    public_data_root = DEFAULT_OUTPUT.resolve()
    try:
        source_bundle.relative_to(public_data_root)
    except ValueError as exc:
        raise ValueError(
            f"Reusable atlas bundle must stay inside {public_data_root}: {source_bundle}"
        ) from exc
    source_manifest_path = source_bundle / "manifest.json"
    source_ids_path = source_bundle / "image-ids.json"
    if not source_manifest_path.is_file() or not source_ids_path.is_file():
        raise FileNotFoundError(
            "Atlas reuse requires manifest.json and image-ids.json in "
            f"{source_bundle}"
        )

    source_ids = json.loads(source_ids_path.read_text(encoding="utf-8"))
    if source_ids != list(image_ids):
        raise ValueError(
            f"Atlas source image IDs do not align with {output_dir}: {source_bundle}"
        )

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_config = source_manifest.get("thumbnails", {})
    if not source_config.get("available"):
        raise ValueError(f"Atlas source is not available: {source_bundle}")
    if int(source_config.get("missingSourceCount", -1)) != 0:
        raise ValueError(
            f"Atlas source contains missing/placeholder images: {source_bundle}"
        )

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
        "mapping",
    )
    mismatches = {
        key: {"source": source_config.get(key), "expected": config.get(key)}
        for key in geometry_keys
        if source_config.get(key) != config.get(key)
    }
    if mismatches:
        raise ValueError(
            "Atlas source geometry does not match the requested bundle: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    source_directory = str(source_config.get("directory", "atlases"))
    source_atlas_dir = (source_bundle / source_directory).resolve()
    try:
        source_atlas_dir.relative_to(public_data_root)
    except ValueError as exc:
        raise ValueError(
            f"Reusable atlas directory escapes {public_data_root}: {source_atlas_dir}"
        ) from exc
    expected_size = (int(config["atlasWidth"]), int(config["atlasHeight"]))
    total_bytes = 0
    for atlas_index in range(int(config["atlasCount"])):
        atlas_path = source_atlas_dir / f"atlas-{atlas_index:04d}.webp"
        if not _valid_cached_atlas(atlas_path, expected_size):
            raise ValueError(f"Missing or invalid reusable atlas: {atlas_path}")
        total_bytes += atlas_path.stat().st_size

    relative_directory = Path(
        os.path.relpath(source_atlas_dir, start=output_dir.resolve())
    ).as_posix()
    config["directory"] = relative_directory
    config["pathPattern"] = f"{relative_directory}/atlas-{{atlas:04d}}.webp"
    return {
        "available": True,
        "generatedAtlases": 0,
        "reusedAtlases": int(config["atlasCount"]),
        "missingSourceCount": 0,
        "missingSourceIndices": [],
        "bytes": total_bytes,
        "webpQuality": source_config.get("webpQuality"),
        "sharedFrom": source_bundle.name,
        "sharedImageIdsSha256": source_manifest["files"]["imageIds"]["sha256"],
    }


def validate_positive_args(args: argparse.Namespace) -> None:
    positive = {
        "atlas-workers": args.atlas_workers,
        "atlas-columns": args.atlas_columns,
        "atlas-rows": args.atlas_rows,
        "tile-width": args.tile_width,
        "tile-height": args.tile_height,
        "umap-neighbors": args.umap_neighbors,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not 0 <= args.webp_quality <= 100:
        raise ValueError("--webp-quality must be between 0 and 100")
    if args.umap_min_dist < 0:
        raise ValueError("--umap-min-dist must be non-negative")
    if args.umap_only and args.skip_umap:
        raise ValueError("--umap-only cannot be combined with --skip-umap")


def generate_umap_only(args: argparse.Namespace, output_dir: Path) -> int:
    """Generate UMAP from already exported ranks and update only manifest.

    This fast path intentionally does not read any source HTML/CSV/NPZ files,
    import Pillow, enumerate atlases, or rewrite other exported arrays.
    """

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"--umap-only requires an existing manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = int(manifest.get("schemaVersion", -1))
    if schema_version not in {1, 2}:
        raise ValueError(
            f"Unsupported existing manifest schema: {manifest.get('schemaVersion')!r}"
        )

    rows = int(manifest["rowCount"])
    columns = int(manifest["methodCount"])
    cluster_columns = int(manifest["clusterMethodCount"])
    cluster_methods = [str(value) for value in manifest["clusterMethods"]]
    if len(cluster_methods) != cluster_columns:
        raise ValueError(
            "Existing manifest clusterMethods length does not match clusterMethodCount"
        )
    if not 1 <= cluster_columns <= columns:
        raise ValueError(
            f"Invalid cluster method count {cluster_columns} for {columns} rank columns"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("ranks"), dict):
        raise ValueError("Existing manifest does not describe files.ranks")
    rank_spec = files["ranks"]
    if rank_spec.get("dtype") != "float32":
        raise ValueError(
            f"--umap-only expects float32 ranks, found {rank_spec.get('dtype')!r}"
        )
    target_count = int(manifest.get("targetCount", 1))
    expected_rank_shape = (
        [rows, columns]
        if schema_version == 1
        else [rows, columns, target_count]
    )
    if [int(value) for value in rank_spec.get("shape", [])] != expected_rank_shape:
        raise ValueError(
            f"Rank descriptor shape {rank_spec.get('shape')!r} does not match "
            f"{expected_rank_shape}"
        )
    rank_path = (output_dir / str(rank_spec["path"])).resolve()
    try:
        rank_path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(f"Rank path escapes the output directory: {rank_path}") from exc
    expected_rank_bytes = math.prod(expected_rank_shape) * np.dtype("<f4").itemsize
    if not rank_path.is_file() or rank_path.stat().st_size != expected_rank_bytes:
        actual = rank_path.stat().st_size if rank_path.is_file() else None
        raise ValueError(
            f"Rank file has {actual} bytes; expected {expected_rank_bytes}: {rank_path}"
        )
    ranks = np.fromfile(rank_path, dtype="<f4").reshape(expected_rank_shape)
    if not np.isfinite(ranks).all():
        raise ValueError("Existing rank data contains NaN or infinity")

    if schema_version == 2:
        retrieval_targets = manifest.get("retrievalTargets", [])
        target_ids = [
            str(item.get("id"))
            for item in retrieval_targets
            if isinstance(item, dict)
        ]
        if len(target_ids) != target_count or "joint" not in target_ids:
            raise ValueError(
                f"Invalid retrievalTargets for targetCount={target_count}: {target_ids}"
            )
        projection_ranks = ranks[:, :, target_ids.index("joint")]
    else:
        projection_ranks = ranks

    print(
        f"UMAP-only mode: validated existing rank shape {expected_rank_shape}; "
        "source files and thumbnail atlases will not be touched."
    )
    development_mask = None
    development_spec = files.get("developmentMask")
    if isinstance(development_spec, dict):
        development_mask = np.fromfile(
            output_dir / str(development_spec["path"]), dtype=np.uint8
        )
    features, _, _ = standardized_features(
        projection_ranks[:, :cluster_columns], development_mask
    )
    existing_umap_spec = files.get("umap2d")
    umap_filename = (
        str(existing_umap_spec["path"])
        if isinstance(existing_umap_spec, dict) and existing_umap_spec.get("path")
        else "umap-2d.f32"
    )
    umap_path = (output_dir / umap_filename).resolve()
    try:
        relative_umap_path = umap_path.relative_to(output_dir).as_posix()
    except ValueError as exc:
        raise ValueError(f"UMAP path escapes the output directory: {umap_path}") from exc

    coordinates, umap_status = compute_or_load_umap(
        features, umap_path, args, fit_mask=development_mask
    )
    projections = manifest.setdefault("projections", {})
    projections["basisTarget"] = "joint"
    projections["featureMethods"] = cluster_methods
    projections["featureTransform"] = "per-column z-score"
    projections["umap"] = umap_status

    if coordinates is None:
        # The manifest must never advertise an absent or unverified UMAP.  A
        # stale file, if any, is left untouched but made unreachable here.
        files.pop("umap2d", None)
        manifest["generatedAt"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        print(f"UMAP unavailable: {umap_status['reason']}", file=sys.stderr)
        print("Updated manifest only; thumbnail atlases were untouched.")
        return 0

    write_array(umap_path, coordinates, "<f4")
    files["umap2d"] = file_spec(
        output_dir, relative_umap_path, "float32", [rows, 2]
    )
    manifest["generatedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(f"Wrote genuine UMAP coordinates: {umap_path}")
    print(f"Updated manifest: {manifest_path}")
    print("Thumbnail atlases and all other exported arrays were untouched.")
    return 0


def main() -> int:
    args = parse_args()
    validate_positive_args(args)
    output_dir = (args.output or DEFAULT_OUTPUT).resolve()

    if args.umap_only and args.query_only:
        raise ValueError("--umap-only and --query-only cannot be used together")
    if args.skip_atlases and args.reuse_atlases_from is not None:
        raise ValueError("--skip-atlases and --reuse-atlases-from are mutually exclusive")

    if args.query_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        return refresh_query_only(args, output_dir)

    if args.umap_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        return generate_umap_only(args, output_dir)

    print(f"Reading PCP payload: {args.pcp_html}")
    payload = extract_pcp_payload(args.pcp_html.resolve())
    payload_metadata = payload.get("metadata", {})
    is_legacy_default_task = (
        payload_metadata.get("dataset") == "hico"
        and payload_metadata.get("task_name") == "task_hico_hugging_cat"
    )
    if not is_legacy_default_task and output_dir == DEFAULT_OUTPUT.resolve():
        raise ValueError(
            "Refusing to export a non-default task over the current HICO data. "
            "Set --output-data-dir to a distinct task-bundle directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = [str(value) for value in payload["images"]]
    image_resolver = ImageSourceResolver(args.image_root)
    methods = [str(value) for value in payload["methods"]]
    legacy_joint_ranks, draw_order = decode_pcp_arrays(payload)
    rows, columns = legacy_joint_ranks.shape
    if rows != len(image_ids) or columns != len(methods):
        raise ValueError("PCP rows/columns do not match image/method metadata")

    print(f"Reading multi-target score bundle: {args.score_bundle}")
    raw_scores, calibrated_scores, ranks, target_ids, score_metadata = (
        load_multitarget_scores(
            args.score_bundle.resolve(),
            image_ids,
            methods,
            legacy_joint_ranks,
        )
    )
    target_count = len(target_ids)
    joint_target_index = target_ids.index("joint")
    joint_ranks = ranks[:, :, joint_target_index]
    retrieval_targets = retrieval_target_definitions(score_metadata, target_ids)

    cluster_methods, cluster_ranks, absolute_labels, shape_labels = load_cluster_data(
        args.cluster_dir.resolve(), image_ids, joint_ranks, methods
    )
    metrics, ground_truth = load_assignments(
        args.cluster_dir.resolve() / "image_assignments.csv",
        image_ids,
        absolute_labels,
        shape_labels,
        retrieval_targets,
        target_ids,
    )
    cluster_summary = parse_summary(args.cluster_dir.resolve() / "cluster_summary.csv")

    fixed_query = export_fixed_query(
        output_dir=output_dir,
        image_root=args.image_root,
        image_ids=image_ids,
        dataset=str(payload_metadata.get("dataset", "hico")),
        task_name=str(payload_metadata.get("task_name", "")),
        task_dir_override=args.task_dir,
        query_text_override=args.query_text,
        image_resolver=image_resolver,
    )
    evaluation_masks = None
    evaluation: dict[str, Any] | None = None
    if fixed_query is not None:
        attribute_indices = [
            index
            for index, target in enumerate(retrieval_targets)
            if str(target.get("kind")) == "attribute"
        ]
        task_dir = resolve_task_directory(
            str(payload_metadata.get("dataset", "hico")),
            str(payload_metadata.get("task_name", "")),
            args.task_dir,
        )
        training_ids, training_supervision, source_evaluation = (
            training_supervision_from_task_contract(
                task_dir,
                image_ids=image_ids,
            )
        )
        if args.active_training_supervision_file is not None:
            explicit_ids, explicit_supervision = training_supervision_from_explicit_file(
                args.active_training_supervision_file,
                stage=args.active_training_supervision_stage,
                image_ids=image_ids,
            )
            if source_evaluation is not None and explicit_ids != training_ids:
                raise ValueError(
                    "Explicit active-training supervision disagrees with the frozen "
                    f"task contract: {task_dir}"
                )
            training_ids = explicit_ids
            training_supervision = explicit_supervision
        elif source_evaluation is None or training_supervision is None:
            raise ValueError(
                "A full export requires a frozen clean-Test task contract or "
                "--active-training-supervision-file with its stage"
            )
        evaluation_masks = build_evaluation_masks(
            image_ids=image_ids,
            ground_truth=ground_truth,
            attribute_indices=attribute_indices,
            query_image_ids=[str(item["imageId"]) for item in fixed_query["images"]],
            training_image_ids=training_ids,
        )
        validate_source_clean_test(
            task_dir,
            source_evaluation,
            image_ids=image_ids,
            masks=evaluation_masks,
        )
        evaluation = evaluation_contract(
            development_mask=evaluation_masks.development,
            validation_mask=evaluation_masks.validation,
            test_mask=evaluation_masks.frozen_test,
            ground_truth=ground_truth,
            target_ids=target_ids,
            attribute_ids=[target_ids[index] for index in attribute_indices],
            candidate_test_mask=evaluation_masks.candidate_frozen_test,
            excluded_training_mask=evaluation_masks.excluded_training,
            training_supervision=training_supervision,
        )

    print(
        f"Validated {rows:,} images x {columns} methods x "
        f"{target_count} retrieval targets"
    )
    development_mask = (
        evaluation_masks.development if evaluation_masks is not None else None
    )
    features, feature_mean, feature_scale = standardized_features(
        cluster_ranks, development_mask
    )
    pca_coordinates, explained_variance = compute_pca(features, development_mask)
    print(
        "PCA explained variance: "
        + ", ".join(f"{value:.3%}" for value in explained_variance)
    )

    # Write the UMAP output after all other validation.  An existing valid file
    # can be reused, making atlas-only reruns inexpensive.
    umap_path = output_dir / "umap-2d.f32"
    umap_coordinates, umap_status = compute_or_load_umap(
        features, umap_path, args, fit_mask=development_mask
    )
    if not umap_status["available"]:
        print(f"UMAP unavailable: {umap_status['reason']}", file=sys.stderr)

    print("Writing compact data arrays ...")
    write_json(output_dir / "image-ids.json", image_ids)
    write_array(output_dir / "raw-scores.f32", raw_scores, "<f4")
    write_array(output_dir / "calibrated-scores.f32", calibrated_scores, "<f4")
    write_array(output_dir / "ranks.f32", ranks, "<f4")
    write_array(output_dir / "absolute-labels.u8", absolute_labels, "u1")
    write_array(output_dir / "shape-labels.u8", shape_labels, "u1")
    write_array(output_dir / "pca-2d.f32", pca_coordinates, "<f4")
    write_array(output_dir / "metrics.f32", metrics, "<f4")
    write_array(output_dir / "ground-truth.u8", ground_truth, "u1")
    if draw_order is not None:
        write_array(output_dir / "draw-order.u32", draw_order, "<u4")
    if umap_coordinates is not None:
        write_array(umap_path, umap_coordinates, "<f4")
    elif umap_path.exists():
        # Do not leave an old file that a deployment could mistake for a
        # projection described by the current manifest.
        umap_path.unlink()

    thumbnail_config = atlas_config(args, rows)
    representatives = parse_representatives(
        args.cluster_dir.resolve() / "representative_images.csv", thumbnail_config
    )
    if args.reuse_atlases_from is not None:
        print(f"Reusing validated thumbnail atlases from {args.reuse_atlases_from} ...")
        atlas_status = reuse_atlases(
            args.reuse_atlases_from,
            image_ids,
            output_dir,
            thumbnail_config,
        )
    elif args.skip_atlases:
        atlas_status: dict[str, Any] = {
            "available": False,
            "reason": "Skipped by --skip-atlases.",
            "generatedAtlases": 0,
            "reusedAtlases": 0,
            "missingSourceCount": 0,
            "missingSourceIndices": [],
            "bytes": 0,
            "webpQuality": int(args.webp_quality),
        }
    else:
        print(
            f"Building {thumbnail_config['atlasCount']} WebP atlases from "
            f"{args.image_root.resolve()} ..."
        )
        atlas_status = build_atlases(
            image_ids,
            args.image_root.resolve(),
            output_dir,
            thumbnail_config,
            args,
            image_resolver=image_resolver,
        )
    thumbnail_config.update(atlas_status)

    if evaluation_masks is not None:
        write_array(output_dir / "development-mask.u8", evaluation_masks.development, "u1")
        write_array(output_dir / "validation-mask.u8", evaluation_masks.validation, "u1")
        write_array(output_dir / "test-mask.u8", evaluation_masks.frozen_test, "u1")

    base_attributes = [target for target in target_ids if target != "joint"]
    raw_score_metadata = score_metadata.get("rawScore", {})
    rank_metadata = score_metadata.get("rank", {})
    calibration_metadata = score_metadata.get("calibration", {})

    metadata = {
        "schemaVersion": 2,
        "task": {
            "dataset": payload.get("metadata", {}).get("dataset", "hico"),
            "task": payload.get("metadata", {}).get("task"),
            "taskName": payload.get("metadata", {}).get("task_name"),
            "split": payload.get("metadata", {}).get("split"),
            "defaultRetrievalTarget": "joint",
            "jointRule": payload.get("metadata", {}).get("joint_rule"),
            "rawScoreDefinition": raw_score_metadata.get("definition"),
            "calibratedScoreDefinition": calibration_metadata.get("definition"),
            "rankDefinition": rank_metadata.get(
                "definition", payload.get("metadata", {}).get("rank_definition")
            ),
            "tiePolicy": payload.get("metadata", {}).get("tie_policy"),
            "baseAttributes": base_attributes,
            "retrievalTargets": retrieval_targets,
        },
        "clusterSummary": cluster_summary,
        "representatives": representatives,
        "metricColumns": METRIC_COLUMNS,
        "groundTruthColumns": target_ids,
        "featureStandardization": {
            "definition": "per-method z-standardization of the 13 cluster rank columns",
            "basisTarget": "joint",
            "fitScope": "development" if development_mask is not None else "all rows",
            "fitMaskFileKey": "developmentMask" if development_mask is not None else None,
            "fitRowCount": int(np.count_nonzero(development_mask)) if development_mask is not None else rows,
            "mean": [float(value) for value in feature_mean],
            "scale": [float(value) for value in feature_scale],
        },
        "scoreCalibration": calibration_metadata,
        "thumbnails": thumbnail_config,
    }
    if evaluation is not None:
        metadata["evaluation"] = evaluation
    write_json(output_dir / "metadata.json", metadata)

    files: dict[str, Any] = {
        "imageIds": json_file_spec(output_dir, "image-ids.json", [rows]),
        "metadata": json_file_spec(output_dir, "metadata.json", [1]),
        "rawScores": file_spec(
            output_dir,
            "raw-scores.f32",
            "float32",
            [rows, columns, target_count],
        ),
        "calibratedScores": file_spec(
            output_dir,
            "calibrated-scores.f32",
            "float32",
            [rows, columns, target_count],
        ),
        "ranks": file_spec(
            output_dir,
            "ranks.f32",
            "float32",
            [rows, columns, target_count],
        ),
        "absoluteLabels": file_spec(
            output_dir, "absolute-labels.u8", "uint8", [rows]
        ),
        "shapeLabels": file_spec(output_dir, "shape-labels.u8", "uint8", [rows]),
        "pca2d": file_spec(output_dir, "pca-2d.f32", "float32", [rows, 2]),
        "metrics": file_spec(
            output_dir,
            "metrics.f32",
            "float32",
            [rows, len(METRIC_COLUMNS)],
            columns=METRIC_COLUMNS,
        ),
        "groundTruth": file_spec(
            output_dir,
            "ground-truth.u8",
            "uint8",
            [rows, target_count],
            columns=target_ids,
        ),
    }
    if evaluation_masks is not None:
        files["developmentMask"] = file_spec(
            output_dir, "development-mask.u8", "uint8", [rows]
        )
        files["validationMask"] = file_spec(
            output_dir, "validation-mask.u8", "uint8", [rows]
        )
        files["testMask"] = file_spec(output_dir, "test-mask.u8", "uint8", [rows])
    if draw_order is not None:
        files["drawOrder"] = file_spec(
            output_dir, "draw-order.u32", "uint32", [rows]
        )
    if umap_coordinates is not None:
        files["umap2d"] = file_spec(
            output_dir, "umap-2d.f32", "float32", [rows, 2]
        )

    manifest = {
        "schemaVersion": 2,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "rowCount": rows,
        "methodCount": columns,
        "targetCount": target_count,
        "clusterMethodCount": len(cluster_methods),
        "methods": methods,
        "retrievalTargets": retrieval_targets,
        "defaultRetrievalTarget": "joint",
        "clusterMethods": cluster_methods,
        "defaultRankMethod": "Ours-Full" if "Ours-Full" in methods else methods[-1],
        "binaryContract": {
            "indexing": "Every first dimension is the image index in image-ids.json.",
            "scoreDimensions": ["image", "method", "target"],
            "scoreOffset": "((image * methodCount) + method) * targetCount + target",
            "layout": "row-major",
            "byteOrder": "little-endian",
        },
        "files": files,
        "clusters": {
            "basisTarget": "joint",
            "note": "Absolute and shape clusters are computed from Joint method-rank profiles.",
        },
        "projections": {
            "basisTarget": "joint",
            "featureMethods": cluster_methods,
            "featureTransform": "per-column z-score",
            "pca": {
                "available": True,
                "algorithm": "PCA",
                "implementation": "NumPy covariance eigendecomposition",
                "fitScope": "development" if development_mask is not None else "all rows",
                "fitMaskFileKey": "developmentMask" if development_mask is not None else None,
                "fitRowCount": int(np.count_nonzero(development_mask)) if development_mask is not None else rows,
                "assignmentScope": "all rows projected by fitted components",
                "explainedVarianceRatio": explained_variance,
            },
            "umap": umap_status,
        },
        "thumbnails": thumbnail_config,
    }
    if evaluation is not None:
        manifest["evaluation"] = evaluation
    if fixed_query is not None:
        manifest["query"] = fixed_query
    write_json(output_dir / "manifest.json", manifest)

    total_static_bytes = sum(
        path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
    )
    print(f"Wrote {output_dir / 'manifest.json'}")
    print(f"Total public/data size: {total_static_bytes / (1024 * 1024):.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
