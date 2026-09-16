"""Catalog helpers for publishing audited add-on task bundles.

The Formal task entries are deliberately unmarked and therefore immutable to
this module.  Only entries carrying ``catalogRole == "addon"`` may be inserted
or replaced.  This lets the Formal catalog validator keep its exact per-task
checks while allowing independently audited bundles to coexist in the runtime
catalog.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ADDON_CATALOG_ROLE = "addon"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def iter_catalog_tasks(catalog: dict[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for dataset in catalog.get("datasets", []):
        for task in dataset.get("tasks", []):
            yield dataset, task


def validate_addon_entry(entry: dict[str, Any]) -> None:
    if entry.get("catalogRole") != ADDON_CATALOG_ROLE:
        raise ValueError("Add-on task entry must declare catalogRole='addon'")
    task_id = _nonempty_string(entry.get("id"), label="Add-on task id")
    _nonempty_string(entry.get("label"), label="Add-on task label")
    _nonempty_string(entry.get("description"), label="Add-on task description")
    data_root = _nonempty_string(entry.get("dataRoot"), label="Add-on dataRoot")
    expected_data_root = f"/data/tasks/{task_id}"
    if data_root != expected_data_root:
        raise ValueError(
            f"Add-on dataRoot must match its canonical task path: "
            f"{data_root} != {expected_data_root}"
        )
    default_target = _nonempty_string(
        entry.get("defaultRetrievalTarget"),
        label="Add-on default retrieval target",
    )

    declared = entry.get("declaredAttributes")
    if not isinstance(declared, list) or not declared or not all(
        isinstance(value, str) and value.strip() for value in declared
    ):
        raise ValueError("Add-on declaredAttributes must be a non-empty string list")
    modeled = entry.get("modeledRetrievalTargets")
    if not isinstance(modeled, list) or not modeled:
        raise ValueError("Add-on modeledRetrievalTargets must be non-empty")
    modeled_ids: list[str] = []
    for index, target in enumerate(modeled):
        if not isinstance(target, dict):
            raise ValueError(f"Add-on modeled target {index} must be an object")
        modeled_ids.append(
            _nonempty_string(target.get("id"), label=f"Add-on modeled target {index} id")
        )
        _nonempty_string(
            target.get("label"), label=f"Add-on modeled target {index} label"
        )
    if len(set(modeled_ids)) != len(modeled_ids):
        raise ValueError("Add-on modeled target ids must be unique")
    if entry.get("modeledTargetIds") != modeled_ids:
        raise ValueError("Add-on modeledTargetIds must match modeledRetrievalTargets")
    if default_target not in {*modeled_ids, "joint"}:
        raise ValueError("Add-on defaultRetrievalTarget is not modeled")

    tuning_source = entry.get("tuningSource")
    if not isinstance(tuning_source, dict) or tuning_source.get("schemaVersion") != 1:
        raise ValueError("Add-on tuningSource schemaVersion must be 1")
    expected_source_keys = {
        "schemaVersion",
        "manifest",
        "suite",
        "oursFullRoot",
        "supervisionAudit",
        "probeStage",
    }
    if set(tuning_source) != expected_source_keys:
        raise ValueError("Add-on tuningSource fields are incomplete or unsupported")

    def normalized_relative_source(key: str, prefix: str) -> str:
        value = _nonempty_string(
            tuning_source.get(key), label=f"Add-on tuningSource.{key}"
        )
        path = Path(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or path.as_posix() != value
            or not value.startswith(prefix)
        ):
            raise ValueError(
                f"Add-on tuningSource.{key} must be a normalized repository-relative "
                f"path under {prefix}"
            )
        return value

    normalized_relative_source("manifest", "configs/")
    normalized_relative_source("suite", "configs/")
    ours_full_root = normalized_relative_source("oursFullRoot", "outputs/").rstrip("/")
    supervision_audit = normalized_relative_source("supervisionAudit", "outputs/")
    expected_audit = f"{ours_full_root}/tasks/{task_id}/supervision_audit.json"
    if supervision_audit != expected_audit:
        raise ValueError(
            "Add-on tuningSource.supervisionAudit must belong to the task's Ours-Full run"
        )
    if tuning_source.get("probeStage") not in {"iterative", "two_stage"}:
        raise ValueError("Add-on tuningSource.probeStage is unsupported")

    row_count = int(entry.get("rowCount", -1))
    target_count = int(entry.get("targetCount", -1))
    if row_count <= 0 or target_count != len(modeled_ids) + 1:
        raise ValueError("Add-on rowCount/targetCount contract is invalid")
    for key in (
        "queryBytes",
        "initialDownloadBytes",
        "bundleBytes",
        "thumbnailBytes",
        "visualEmbeddingBytes",
    ):
        if int(entry.get(key, -1)) <= 0:
            raise ValueError(f"Add-on {key} must be positive")

    split_rows = 0
    expected_positive_keys = [*modeled_ids, "joint"]
    for prefix in ("development", "validation", "test"):
        split_count = int(entry.get(f"{prefix}RowCount", -1))
        if split_count < 0:
            raise ValueError(f"Add-on {prefix}RowCount must be non-negative")
        split_rows += split_count
        counts = entry.get(f"{prefix}PositiveCounts")
        if not isinstance(counts, dict) or list(counts) != expected_positive_keys:
            raise ValueError(
                f"Add-on {prefix}PositiveCounts must follow target order"
            )
        if any(int(value) < 0 or int(value) > split_count for value in counts.values()):
            raise ValueError(f"Add-on {prefix}PositiveCounts are out of range")
    if split_rows > row_count:
        raise ValueError("Add-on split row counts exceed rowCount")


def validate_catalog_shape(catalog: dict[str, Any]) -> None:
    if not isinstance(catalog, dict) or int(catalog.get("schemaVersion", -1)) != 2:
        raise ValueError("Catalog schemaVersion must be 2")
    datasets = catalog.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Catalog datasets must be a non-empty list")

    dataset_ids: set[str] = set()
    task_ids: set[str] = set()
    data_roots: set[str] = set()
    task_locations: dict[str, str] = {}
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Catalog dataset entries must be objects")
        dataset_id = _nonempty_string(dataset.get("id"), label="Dataset id")
        _nonempty_string(dataset.get("label"), label=f"Dataset {dataset_id} label")
        if dataset_id in dataset_ids:
            raise ValueError(f"Duplicate catalog dataset id: {dataset_id}")
        dataset_ids.add(dataset_id)
        tasks = dataset.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError(f"Dataset {dataset_id} tasks must be a list")
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError(f"Dataset {dataset_id} task entries must be objects")
            task_id = _nonempty_string(task.get("id"), label="Task id")
            data_root = _nonempty_string(
                task.get("dataRoot"), label=f"Task {task_id} dataRoot"
            )
            if task_id in task_ids:
                raise ValueError(f"Duplicate catalog task id: {task_id}")
            if data_root in data_roots:
                raise ValueError(f"Duplicate catalog dataRoot: {data_root}")
            task_ids.add(task_id)
            data_roots.add(data_root)
            task_locations[task_id] = dataset_id
            if task.get("catalogRole") == "experimental":
                from experimental_parent_overlay import TASK_ID, PROTOCOL
                overlay = task.get("experimentalOverlay", {})
                if (task_id != TASK_ID or dataset_id != "hico"
                        or data_root != f"/data/tasks/{TASK_ID}"
                        or overlay.get("protocol") != PROTOCOL
                        or not re.fullmatch(r"[0-9a-f]{64}", str(overlay.get("manifestSha256", "")))):
                    raise ValueError("Unknown or invalid experimental task registration")
            elif "catalogRole" in task:
                validate_addon_entry(task)

    if int(catalog.get("taskCount", -1)) != len(task_ids):
        raise ValueError("Catalog taskCount does not match datasets.tasks")
    default_dataset = _nonempty_string(
        catalog.get("defaultDataset"), label="Catalog defaultDataset"
    )
    default_task = _nonempty_string(
        catalog.get("defaultTask"), label="Catalog defaultTask"
    )
    if default_dataset not in dataset_ids:
        raise ValueError("Catalog defaultDataset is not present")
    if task_locations.get(default_task) != default_dataset:
        raise ValueError("Catalog defaultTask is not in defaultDataset")


def upsert_addon_entry(
    catalog: dict[str, Any],
    *,
    dataset_id: str,
    dataset_label: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Return a catalog with one add-on inserted or replaced.

    Existing unmarked task entries are protected.  An existing add-on keeps
    its list position, while a new add-on is appended to its dataset.
    """

    validate_catalog_shape(catalog)
    validate_addon_entry(entry)
    dataset_id = _nonempty_string(dataset_id, label="Dataset id")
    dataset_label = _nonempty_string(dataset_label, label="Dataset label")
    updated = copy.deepcopy(catalog)
    task_id = str(entry["id"])

    existing_dataset: dict[str, Any] | None = None
    existing_task: dict[str, Any] | None = None
    existing_index = -1
    for dataset in updated["datasets"]:
        for index, task in enumerate(dataset["tasks"]):
            if task["id"] == task_id:
                existing_dataset = dataset
                existing_task = task
                existing_index = index
                break
        if existing_task is not None:
            break
    if existing_task is not None:
        if existing_task.get("catalogRole") != ADDON_CATALOG_ROLE:
            raise ValueError(f"Refusing to replace protected catalog task: {task_id}")
        if existing_dataset is None or existing_dataset["id"] != dataset_id:
            raise ValueError(f"Refusing to move add-on task between datasets: {task_id}")

    target_dataset = next(
        (dataset for dataset in updated["datasets"] if dataset["id"] == dataset_id),
        None,
    )
    if target_dataset is None:
        target_dataset = {"id": dataset_id, "label": dataset_label, "tasks": []}
        updated["datasets"].append(target_dataset)
    elif target_dataset["label"] != dataset_label:
        raise ValueError(
            f"Dataset label drift for {dataset_id}: "
            f"{target_dataset['label']!r} != {dataset_label!r}"
        )

    if existing_task is None:
        target_dataset["tasks"].append(copy.deepcopy(entry))
    else:
        existing_dataset["tasks"][existing_index] = copy.deepcopy(entry)
    updated["taskCount"] = sum(len(dataset["tasks"]) for dataset in updated["datasets"])
    validate_catalog_shape(updated)
    return updated


def merge_published_addons(
    formal_catalog: dict[str, Any],
    published_catalog: dict[str, Any] | None,
) -> dict[str, Any]:
    """Preserve validated add-on entries while rebuilding Formal entries."""

    validate_catalog_shape(formal_catalog)
    if published_catalog is None:
        return copy.deepcopy(formal_catalog)
    validate_catalog_shape(published_catalog)
    formal_ids = {task["id"] for _, task in iter_catalog_tasks(formal_catalog)}
    for dataset, task in iter_catalog_tasks(published_catalog):
        if task["id"] in formal_ids:
            continue
        if task.get("catalogRole") not in (ADDON_CATALOG_ROLE, "experimental"):
            raise ValueError(
                f"Published non-Formal task is not a marked add-on: {task['id']}"
            )

    merged = copy.deepcopy(formal_catalog)
    for dataset, task in iter_catalog_tasks(published_catalog):
        if task.get("catalogRole") == "experimental":
            target = next((d for d in merged["datasets"] if d["id"] == dataset["id"]), None)
            if target is None:
                target = {"id": dataset["id"], "label": dataset["label"], "tasks": []}
                merged["datasets"].append(target)
            target["tasks"].append(copy.deepcopy(task))
            merged["taskCount"] += 1
            validate_catalog_shape(merged)
            continue
        if task.get("catalogRole") != ADDON_CATALOG_ROLE:
            continue
        merged = upsert_addon_entry(
            merged,
            dataset_id=str(dataset["id"]),
            dataset_label=str(dataset["label"]),
            entry=task,
        )
    return merged


def validate_formal_catalog_extension(
    formal_catalog: dict[str, Any],
    published_catalog: dict[str, Any],
) -> None:
    """Require every Formal entry exactly, permitting only marked add-ons."""

    validate_catalog_shape(formal_catalog)
    validate_catalog_shape(published_catalog)
    for key in ("schemaVersion", "defaultDataset", "defaultTask"):
        if published_catalog.get(key) != formal_catalog.get(key):
            raise ValueError(f"Published catalog changed Formal top-level field: {key}")

    formal_datasets = {
        str(dataset["id"]): dataset for dataset in formal_catalog["datasets"]
    }
    published_datasets = {
        str(dataset["id"]): dataset for dataset in published_catalog["datasets"]
    }
    formal_ids: set[str] = set()
    for dataset_id, formal_dataset in formal_datasets.items():
        observed_dataset = published_datasets.get(dataset_id)
        if observed_dataset is None:
            raise ValueError(f"Published catalog is missing Formal dataset: {dataset_id}")
        if observed_dataset.get("label") != formal_dataset.get("label"):
            raise ValueError(f"Published catalog changed Formal dataset label: {dataset_id}")
        observed_by_id = {
            str(task["id"]): task for task in observed_dataset["tasks"]
        }
        observed_formal_order: list[str] = []
        for task in formal_dataset["tasks"]:
            task_id = str(task["id"])
            formal_ids.add(task_id)
            if observed_by_id.get(task_id) != task:
                raise ValueError(f"Published Formal task entry drifted: {task_id}")
            observed_formal_order.append(task_id)
        actual_order = [
            str(task["id"])
            for task in observed_dataset["tasks"]
            if str(task["id"]) in formal_ids
        ]
        if actual_order != observed_formal_order:
            raise ValueError(f"Published Formal task order drifted: {dataset_id}")

    for _, task in iter_catalog_tasks(published_catalog):
        if str(task["id"]) in formal_ids:
            continue
        if task.get("catalogRole") == "experimental":
            continue  # Exact experimental identity already checked by shape validation.
        validate_addon_entry(task)


@contextmanager
def catalog_lock(catalog_path: Path, *, timeout_seconds: float = 10.0) -> Iterator[None]:
    lock_path = catalog_path.with_suffix(catalog_path.suffix + ".lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for catalog lock: {lock_path}")
            time.sleep(0.05)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def atomic_write_catalog(path: Path, payload: dict[str, Any]) -> None:
    """Commit a complete catalog with one same-directory atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
