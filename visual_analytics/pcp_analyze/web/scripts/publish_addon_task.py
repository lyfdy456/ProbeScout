#!/usr/bin/env python3
"""Validate one complete browser task bundle and atomically publish it.

The bundle must already live under ``public/data/tasks/<task-id>`` and retain
the source score, active-supervision, task evaluation, embedding, Query and
atlas provenance required by the existing strict bundle validator.  This
command never generates scores or data and never replaces an unmarked
(Formal/core) catalog task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WEB_ROOT = SCRIPT_DIR.parent
ANALYSIS_ROOT = WEB_ROOT.parent
REPOSITORY_ROOT = ANALYSIS_ROOT.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_task_bundles as FORMAL  # noqa: E402
from addon_catalog import (  # noqa: E402
    ADDON_CATALOG_ROLE,
    atomic_write_catalog,
    catalog_lock,
    read_json,
    upsert_addon_entry,
    validate_catalog_shape,
)


ValidationResult = tuple[FORMAL.TaskSpec, dict[str, Any], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Human-readable task label.")
    parser.add_argument(
        "--description",
        help="Optional catalog description; derived from target labels by default.",
    )
    parser.add_argument(
        "--declared-attribute",
        action="append",
        default=[],
        help="Optional declared attribute override; repeat in task order.",
    )
    parser.add_argument(
        "--dataset-label",
        help="Required only when the bundle introduces a new dataset id.",
    )
    parser.add_argument("--score-bundle", type=Path)
    parser.add_argument("--task-root", type=Path)
    parser.add_argument("--active-training-supervision-file", type=Path)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=WEB_ROOT / "public" / "data" / "catalog.json",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=WEB_ROOT / "public",
        help="Public root used to derive the task dataRoot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show the proposed entry without changing catalog.json.",
    )
    return parser.parse_args()


def _resolved_repository_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository: {resolved}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _resolved_tuning_source(
    raw_value: Any,
    *,
    allowed_root: Path,
    label: str,
    directory: bool = False,
) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{label} must be a repository-relative path")
    requested = Path(raw_value)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError(f"{label} must be a normalized repository-relative path")
    resolved = (REPOSITORY_ROOT / requested).resolve()
    if not resolved.is_relative_to(allowed_root.resolve()):
        raise ValueError(f"{label} escapes its allowed repository root: {resolved}")
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} is missing: {resolved}")
    return resolved


def _portable_repository_path(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _build_tuning_source(
    *,
    score_metadata: dict[str, Any],
    task_id: str,
    order: int,
    dataset_id: str,
    task_name: str,
    declared_attributes: tuple[str, ...],
    probe_stage: str,
    active_supervision_file: Path,
) -> dict[str, Any]:
    """Bind an add-on to the exact audited training sources used by Tune."""

    sources = score_metadata.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("Audited score metadata has no tuning source provenance")
    config_root = REPOSITORY_ROOT / "configs" / "experiments"
    outputs_root = REPOSITORY_ROOT / "outputs"
    manifest_path = _resolved_tuning_source(
        sources.get("manifest"),
        allowed_root=config_root,
        label="Tune task manifest",
    )
    suite_path = _resolved_tuning_source(
        sources.get("suite"),
        allowed_root=config_root,
        label="Tune ProbeBank suite",
    )
    ours_full_root = _resolved_tuning_source(
        sources.get("oursFullRoot"),
        allowed_root=outputs_root,
        label="Tune Ours-Full root",
        directory=True,
    )
    supervision_audit_path = ours_full_root / "tasks" / task_id / "supervision_audit.json"
    if not supervision_audit_path.is_file():
        raise FileNotFoundError(
            f"Tune supervision audit is missing: {supervision_audit_path}"
        )
    ours_full_result = ours_full_root / "tasks" / task_id / "front_minmax_result.json"
    if not ours_full_result.is_file():
        raise FileNotFoundError(f"Tune Ours-Full fit is missing: {ours_full_result}")

    task_manifest = read_json(manifest_path)
    matches = [
        row
        for row in task_manifest.get("tasks", [])
        if int(row.get("order", -1)) == order
    ]
    if len(matches) != 1:
        raise ValueError(f"Tune task manifest must contain exactly one order {order}")
    source_task = matches[0]
    expected_identity = {
        "dataset": dataset_id,
        "task": task_name,
        "attributes": list(declared_attributes),
    }
    mismatches = {
        key: {"observed": source_task.get(key), "expected": value}
        for key, value in expected_identity.items()
        if source_task.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Tune task manifest identity mismatch: {mismatches}")

    suite = read_json(suite_path)
    learned_methods = [str(value) for value in suite.get("learned_methods", [])]
    roots = suite.get("probe_source_roots")
    if (
        len(learned_methods) != 8
        or len(set(learned_methods)) != 8
        or not isinstance(roots, dict)
        or "__default__" not in roots
    ):
        raise ValueError("Tune ProbeBank suite must define eight unique learners and a default root")
    cache_rows = sources.get("learnedScoreCaches")
    expected_cache_pairs = {
        (method, attribute)
        for method in learned_methods
        for attribute in declared_attributes
    }
    observed_cache_pairs = {
        (str(row.get("method")), str(row.get("attribute")))
        for row in cache_rows or []
        if isinstance(row, dict)
    }
    if observed_cache_pairs != expected_cache_pairs or len(cache_rows or []) != len(
        expected_cache_pairs
    ):
        raise ValueError("Tune score-cache provenance does not cover the suite and attributes")

    supervision_audit = read_json(supervision_audit_path)
    if supervision_audit.get("status") != "valid":
        raise ValueError("Tune supervision audit is not valid")
    selected_manifest = Path(str(supervision_audit.get("selected_manifest", ""))).resolve()
    _validate_equivalent_supervision_manifests(
        selected_manifest,
        active_supervision_file.resolve(),
        matched_count=int(supervision_audit.get("matched_count", -1)),
    )

    return {
        "schemaVersion": 1,
        "manifest": _portable_repository_path(manifest_path),
        "suite": _portable_repository_path(suite_path),
        "oursFullRoot": _portable_repository_path(ours_full_root),
        "supervisionAudit": _portable_repository_path(supervision_audit_path),
        "probeStage": probe_stage,
    }


def _validate_equivalent_supervision_manifests(
    audited_path: Path,
    published_path: Path,
    *,
    matched_count: int,
) -> None:
    """Allow a canonical mirror only when its ordered IDs exactly match Tune.

    Run-local supervision is the immutable ProbeBank audit source, while Web
    publication uses a task-local canonical copy so the Frozen-Test contract is
    self-contained.  Path or JSON-formatting equality is not meaningful here;
    the ordered, duplicate-free image-ID list and audited count are.
    """

    manifests: list[list[str]] = []
    for label, path in (
        ("Tune supervision audit selected manifest", audited_path),
        ("Published active supervision", published_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
        payload = read_json(path)
        if not isinstance(payload, list) or not all(
            isinstance(value, str) for value in payload
        ):
            raise ValueError(f"{label} must be a JSON string list: {path}")
        if len(set(payload)) != len(payload):
            raise ValueError(f"{label} contains duplicate image IDs: {path}")
        manifests.append([str(value) for value in payload])
    audited, published = manifests
    if audited != published:
        raise ValueError(
            "Tune supervision audit selected manifest does not match the published bundle"
        )
    if matched_count != len(audited):
        raise ValueError("Tune supervision audit count does not match the selected manifest")


def _infer_supervision_path(
    manifest: dict[str, Any],
    override: Path | None,
) -> tuple[Path, str]:
    isolation = manifest.get("evaluation", {}).get("testIsolation", {})
    supervision = isolation.get("activeTrainingSupervision", {})
    source_stage = str(supervision.get("sourceStage", "")).strip()
    if not source_stage:
        raise ValueError("Bundle evaluation has no active-supervision sourceStage")
    if override is not None:
        return _resolved_repository_file(
            override, label="Active training supervision"
        ), source_stage
    relative = supervision.get("idsFile")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("Bundle evaluation has no active-supervision idsFile")
    requested = Path(relative)
    if requested.is_absolute():
        raise ValueError("Bundle active-supervision idsFile must be repository-relative")
    return _resolved_repository_file(
        REPOSITORY_ROOT / requested,
        label="Active training supervision",
    ), source_stage


def _canonical_bundle_root(bundle: Path, public_root: Path, task_id: str) -> Path:
    resolved_bundle = bundle.resolve()
    resolved_public = public_root.resolve()
    expected = (resolved_public / "data" / "tasks" / task_id).resolve()
    if resolved_bundle != expected:
        raise ValueError(
            "Add-on bundle must already use its canonical public path: "
            f"{resolved_bundle} != {expected}"
        )
    return resolved_bundle


def validate_addon_bundle(
    *,
    bundle: Path,
    public_root: Path,
    score_bundle: Path | None = None,
    task_root: Path | None = None,
    active_supervision_file: Path | None = None,
) -> ValidationResult:
    """Run the existing strict bundle audit with explicit add-on identities."""

    manifest_path = bundle.resolve() / "manifest.json"
    metadata_path = bundle.resolve() / "metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Complete task bundle is missing: {bundle.resolve()}")
    manifest = read_json(manifest_path)
    metadata = read_json(metadata_path)
    task_metadata = metadata.get("task")
    if not isinstance(task_metadata, dict):
        raise ValueError("Bundle metadata.task is missing")
    task_id = str(task_metadata.get("task", "")).strip()
    dataset_id = str(task_metadata.get("dataset", "")).strip()
    task_name = str(task_metadata.get("taskName", "")).strip()
    components = task_id.split("_", 2)
    if (
        len(components) != 3
        or len(components[0]) != 3
        or not components[0].isdigit()
        or components[1] != dataset_id
        or components[2] != task_name
    ):
        raise ValueError(f"Bundle task identity is not canonical: {task_id!r}")
    order = int(components[0])
    canonical_bundle = _canonical_bundle_root(bundle, public_root, task_id)

    resolved_task_root = (
        task_root.resolve()
        if task_root is not None
        else (
            REPOSITORY_ROOT / "dataset" / "tasks" / dataset_id / task_name
        ).resolve()
    )
    try:
        resolved_task_root.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Task root escapes the repository: {resolved_task_root}") from error
    if not resolved_task_root.is_dir():
        raise FileNotFoundError(f"Task root is missing: {resolved_task_root}")

    artifact_stem = f"{order:03d}_{dataset_id}_{task_name.removeprefix('task_')}"
    resolved_scores = _resolved_repository_file(
        score_bundle
        if score_bundle is not None
        else ANALYSIS_ROOT / "pcp_result" / f"{artifact_stem}_multitarget_scores.npz",
        label="Audited multi-target score bundle",
    )
    supervision_path, supervision_stage = _infer_supervision_path(
        manifest, active_supervision_file
    )
    targets = manifest.get("retrievalTargets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Bundle retrievalTargets are missing")
    declared_attributes = tuple(
        str(target.get("canonicalAttribute") or target.get("label") or "").strip()
        for target in targets
        if target.get("kind") == "attribute"
    )
    if not declared_attributes or any(not value for value in declared_attributes):
        raise ValueError("Bundle has no complete declared attribute identities")

    with np.load(resolved_scores, allow_pickle=False) as source:
        expected_target_ids = [str(value) for value in source["target_ids"].tolist()]
        score_metadata = json.loads(str(source["metadata_json"]))
    probe_stage = str(score_metadata.get("task", {}).get("stage", "")).strip()
    if probe_stage not in {"iterative", "two_stage"}:
        raise ValueError(
            f"Audited score bundle has an unsupported ProbeBank stage: {probe_stage!r}"
        )
    task = FORMAL.TaskSpec(
        order=order,
        dataset=dataset_id,
        task_name=task_name,
        attributes=declared_attributes,
        supervision_stage=supervision_stage,
        probe_stage=probe_stage,
        explicit_data_root=canonical_bundle,
        explicit_score_bundle=resolved_scores,
        explicit_task_root=resolved_task_root,
        explicit_supervision_ids_path=supervision_path,
    )
    audit = FORMAL.validate_bundle(task, expected_target_ids=expected_target_ids)
    audit["tuningSource"] = _build_tuning_source(
        score_metadata=score_metadata,
        task_id=task_id,
        order=order,
        dataset_id=dataset_id,
        task_name=task_name,
        declared_attributes=declared_attributes,
        probe_stage=probe_stage,
        active_supervision_file=supervision_path,
    )
    return task, audit, manifest


def build_addon_entry(
    *,
    task: FORMAL.TaskSpec,
    audit: dict[str, Any],
    manifest: dict[str, Any],
    label: str,
    description: str | None,
    declared_attributes: Iterable[str],
    public_root: Path,
) -> dict[str, Any]:
    retrieval_targets = audit.get("retrievalTargets")
    if not isinstance(retrieval_targets, list):
        raise ValueError("Strict bundle audit returned no retrieval targets")
    modeled = [
        {"id": str(target["id"]), "label": str(target["label"])}
        for target in retrieval_targets
        if str(target.get("id")) != "joint"
    ]
    modeled_ids = [str(target["id"]) for target in modeled]
    declared = [str(value).strip() for value in declared_attributes if str(value).strip()]
    if not declared:
        declared = [
            str(target.get("canonicalAttribute") or target.get("label") or "").strip()
            for target in manifest["retrievalTargets"]
            if target.get("kind") == "attribute"
        ]
    if not declared or any(not value for value in declared):
        raise ValueError("Declared attributes cannot be derived from the bundle")
    default_target = str(manifest.get("defaultRetrievalTarget", "joint"))
    all_target_ids = [str(target["id"]) for target in retrieval_targets]
    if default_target not in all_target_ids:
        raise ValueError("Bundle defaultRetrievalTarget is not modeled")
    bundle_relative = task.data_root.resolve().relative_to(public_root.resolve())
    data_root = "/" + bundle_relative.as_posix()
    visible_label = str(label).strip()
    if not visible_label:
        raise ValueError("Task label cannot be empty")
    visible_description = (
        str(description).strip()
        if description is not None and str(description).strip()
        else ", ".join(target["label"] for target in modeled)
        + ", and Joint retrieval targets"
    )
    entry = {
        "id": task.task_id,
        "label": visible_label,
        "description": visible_description,
        "dataRoot": data_root,
        "defaultRetrievalTarget": default_target,
        "declaredAttributes": declared,
        "modeledRetrievalTargets": modeled,
        "modeledTargetIds": modeled_ids,
        "catalogRole": ADDON_CATALOG_ROLE,
        **{key: value for key, value in audit.items() if key != "retrievalTargets"},
    }
    return entry


def publish_addon_task(
    *,
    bundle: Path,
    label: str,
    description: str | None,
    declared_attributes: Iterable[str],
    dataset_label: str | None,
    catalog_path: Path,
    public_root: Path,
    score_bundle: Path | None = None,
    task_root: Path | None = None,
    active_supervision_file: Path | None = None,
    dry_run: bool = False,
    validator: Callable[..., ValidationResult] = validate_addon_bundle,
) -> dict[str, Any]:
    task, audit, manifest = validator(
        bundle=bundle,
        public_root=public_root,
        score_bundle=score_bundle,
        task_root=task_root,
        active_supervision_file=active_supervision_file,
    )
    entry = build_addon_entry(
        task=task,
        audit=audit,
        manifest=manifest,
        label=label,
        description=description,
        declared_attributes=declared_attributes,
        public_root=public_root,
    )
    resolved_catalog = catalog_path.resolve()
    if not resolved_catalog.is_file():
        raise FileNotFoundError(f"Catalog is missing: {resolved_catalog}")

    def updated_catalog() -> dict[str, Any]:
        current = read_json(resolved_catalog)
        validate_catalog_shape(current)
        existing_dataset = next(
            (
                dataset
                for dataset in current["datasets"]
                if str(dataset["id"]) == task.dataset
            ),
            None,
        )
        visible_dataset_label = (
            str(existing_dataset["label"])
            if existing_dataset is not None
            else str(dataset_label or "").strip()
        )
        if not visible_dataset_label:
            raise ValueError(
                f"--dataset-label is required for new dataset: {task.dataset}"
            )
        if dataset_label is not None and str(dataset_label).strip() != visible_dataset_label:
            raise ValueError(
                f"Dataset label override conflicts with catalog: {task.dataset}"
            )
        return upsert_addon_entry(
            current,
            dataset_id=task.dataset,
            dataset_label=visible_dataset_label,
            entry=entry,
        )

    if dry_run:
        proposed = updated_catalog()
    else:
        with catalog_lock(resolved_catalog):
            proposed = updated_catalog()
            atomic_write_catalog(resolved_catalog, proposed)
            committed = read_json(resolved_catalog)
            if committed != proposed:
                raise RuntimeError("Catalog atomic commit verification failed")
    return {
        "catalog": str(resolved_catalog),
        "catalogUpdated": not dry_run,
        "taskCount": int(proposed["taskCount"]),
        "dataset": task.dataset,
        "task": task.task_id,
        "dataRoot": entry["dataRoot"],
        "rowCount": int(entry["rowCount"]),
        "targetCount": int(entry["targetCount"]),
        "dryRun": bool(dry_run),
    }


def main() -> int:
    args = parse_args()
    result = publish_addon_task(
        bundle=args.bundle,
        label=args.label,
        description=args.description,
        declared_attributes=args.declared_attribute,
        dataset_label=args.dataset_label,
        catalog_path=args.catalog,
        public_root=args.public_root,
        score_bundle=args.score_bundle,
        task_root=args.task_root,
        active_supervision_file=args.active_training_supervision_file,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
