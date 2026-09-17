"""Freeze one task-common VQA holdout; never retrain models or resample on load.

The members are the existing seed-0 derived-Joint 20% split.  Every target
keeps its own recovered VQA labels and fits on its full source minus those
members.  Existing five-seed predictors are reference-only on this holdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PROTOCOL = "fixed-vqa-joint-seed0-holdout-v1"
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: Any) -> str:
    payload = _json(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha(payload)


def _read(path: Path, expected: str | None = None) -> tuple[dict, str]:
    try:
        payload = path.read_bytes()
        digest = _sha(payload)
        if expected is not None and digest != expected:
            raise ValueError("checksum mismatch")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value, digest
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Fixed VQA Validation unavailable or corrupt: {path.name}; "
            "restore the frozen files (no automatic resampling)"
        ) from error


def _version(value: str) -> str:
    if not isinstance(value, str) or not VERSION_PATTERN.fullmatch(value):
        raise RuntimeError("Invalid fixed VQA Validation version")
    return value


def _baseline_source(service: Any, task_id: str) -> dict:
    """Bind row identities and exclusion boundaries without reading public GT."""
    task = service.task(task_id)
    import local_tasks
    bundle = service.bundle(task_id)
    if len(bundle.image_ids) != task.row_count or len(set(bundle.image_ids)) != task.row_count:
        raise RuntimeError("Fixed VQA Validation requires unique aligned image IDs")
    for mask in (bundle.development_mask, bundle.validation_mask, bundle.test_mask):
        if len(mask) != task.row_count or any(value not in (0, 1) for value in mask):
            raise RuntimeError("Fixed VQA Validation scope mask is invalid")
    return {
        "rowCount": task.row_count,
        "targetIds": list(task.target_ids),
        "targetSchemaSha256": _sha(_json(task.manifest.get("retrievalTargets", []))),
        "imageIdsSha256": _sha(_json(list(bundle.image_ids))),
        "developmentMaskSha256": _sha(bytes(bundle.development_mask)),
        "validationMaskSha256": _sha(bytes(bundle.validation_mask)),
        "testMaskSha256": _sha(bytes(bundle.test_mask)),
        "querySha256": _sha(_json(sorted(int(row) for row in bundle.query_indices))),
        "evaluationLabelSource": "user-provided-supervision" if local_tasks.available(service, task_id) else "original-vqa-supervision",
        "usesPublicGroundTruth": False,
    }


def _rows(values: Any, row_count: int, name: str, *, sorted_rows: bool = False) -> list[int]:
    import numpy as np

    array = np.asarray(values)
    if (array.ndim != 1 or not array.size or array.dtype.kind not in "iu"
            or int(array.min()) < 0 or int(array.max()) >= row_count):
        raise RuntimeError(f"Invalid fixed VQA Validation {name} rows")
    rows = [int(row) for row in array]
    if len(set(rows)) != len(rows) or (sorted_rows and rows != sorted(rows)):
        raise RuntimeError(f"Invalid fixed VQA Validation {name} row order")
    return rows


def _labels(values: Any, count: int, name: str) -> list[int]:
    import numpy as np

    array = np.asarray(values)
    if array.shape != (count,) or array.dtype.kind not in "iub" or not np.isin(array, (0, 1)).all():
        raise RuntimeError(f"Invalid fixed VQA Validation {name} labels")
    return [int(label) for label in array]


def _original(service: Any, task_id: str, target_id: str) -> dict:
    task = service.task(task_id)
    bundle = service.bundle(task_id)
    original = service._original_development_supervision(task_id, target_id)
    rows = _rows(original.selected_indices, task.row_count, "source")
    labels = _labels(original.selected_labels, len(rows), "source")
    if any(bundle.test_mask[row] or row in bundle.query_indices for row in rows):
        raise RuntimeError("Original VQA supervision overlaps Frozen Test or Query")
    recovered_hash = original.audit.get("recoveredSupervisionHash")
    if not isinstance(recovered_hash, str) or not recovered_hash:
        raise RuntimeError("Original VQA supervision is missing its recovered hash")
    return {"rows": rows, "labels": labels, "audit": dict(original.audit)}


def _fingerprint(partition: str, task_id: str, target_id: str,
                 rows: list[int], labels: list[int], image_ids: Any) -> str:
    return _sha(_json({
        "partition": partition, "protocol": PROTOCOL,
        "taskId": task_id, "targetId": target_id,
        "records": [[row, image_ids[row], label]
                    for row, label in zip(rows, labels, strict=True)],
    }))


def _target_document(task_id: str, target_id: str, target_kind: str,
                     original: dict, rows: list[int], bundle: Any,
                     history: set[int], selection: dict) -> dict:
    source_rows, source_labels = original["rows"], original["labels"]
    lookup = dict(zip(source_rows, source_labels, strict=True))
    if not set(rows).issubset(lookup):
        raise RuntimeError(f"Common VQA holdout has missing target labels: {task_id}/{target_id}")
    holdout = set(rows)
    fit_rows = [row for row in source_rows if row not in holdout]
    fit_labels = [lookup[row] for row in fit_rows]
    labels = [lookup[row] for row in rows]
    if not fit_rows:
        raise RuntimeError(f"Fixed VQA fit partition is empty: {task_id}/{target_id}")
    # Do not resample a common holdout to improve a target's class balance.
    # Joint is stratified; some attribute populations may be single-class.
    history_overlap = sorted(holdout.intersection(history))
    audit = {
        "schemaVersion": SCHEMA_VERSION, "protocol": PROTOCOL,
        "replayKind": "task-common-derived-joint-seed0",
        "targetId": target_id, "targetKind": target_kind,
        "seed": 0, "validationFraction": 0.2,
        "selectionTargetId": "joint", "selection": selection,
        "evaluationLabelSource": original["audit"].get("labelSource", "original-vqa-supervision"),
        "usesPublicGroundTruth": False,
        "fitProtocol": "original-vqa-minus-shared-validation-v1",
        "sourceSupervisionHash": original["audit"]["recoveredSupervisionHash"],
        "originalSupervisionAuditSha256": _sha(_json(original["audit"])),
        "initialModelHoldoutIndependent": False,
        "referenceOnly": True, "baseModelsRetrained": False,
        "initialModelLimitation": (
            "Existing five-seed Probe/SoftGate predictors were not retrained; "
            "their historical supervision can include these task-common holdout rows. "
            "Val is a fixed system-internal reference, not a five-seed OOF estimate."
        ),
        "historicalFeedbackOverlapCount": len(history_overlap),
        "historicalFeedbackOverlapImageIdsSha256": _sha(_json(
            [bundle.image_ids[row] for row in history_overlap]
        )),
        "historicalExposurePolicy": "record-only-never-remove-or-resample-holdout",
        "historicalExposureLimitation": (
            "Prior annotations, exploration and historical model runs are not erased. "
            "Only retained database feedback exposure is counted; unrecorded exposure is unknown."
        ),
        "validationImageIdsSha256": _sha(_json([bundle.image_ids[row] for row in rows])),
    }
    for name, partition_rows, partition_labels in (
        ("source", source_rows, source_labels),
        ("fit", fit_rows, fit_labels), ("validation", rows, labels),
    ):
        audit.update({
            f"{name}Count": len(partition_rows),
            f"{name}PositiveCount": sum(partition_labels),
            f"{name}NegativeCount": len(partition_labels) - sum(partition_labels),
            f"{name}Fingerprint": _fingerprint(
                name, task_id, target_id, partition_rows, partition_labels, bundle.image_ids
            ),
            f"{name}RowsSha256": _sha(_json(partition_rows)),
            f"{name}ImageIdsSha256": _sha(_json([bundle.image_ids[row] for row in partition_rows])),
            f"{name}LabelsSha256": _sha(_json(partition_labels)),
        })
    audit["sourceSupervisionFingerprint"] = audit["sourceFingerprint"]
    for name, partition_rows in (("fit", fit_rows), ("validation", rows)):
        audit[f"{name}WebDevelopmentCount"] = sum(bundle.development_mask[row] for row in partition_rows)
        audit[f"{name}WebValidationCount"] = sum(bundle.validation_mask[row] for row in partition_rows)
    return {"sourceRows": source_rows, "sourceLabels": source_labels,
            "fitRows": fit_rows, "fitLabels": fit_labels,
            "labels": labels, "originalSupervisionAudit": original["audit"], "audit": audit}


def _manifest(service: Any, version: str | None) -> tuple[dict, str, str]:
    root = service.vqa_validation_root
    expected_hash = None
    if version is None:
        active, _ = _read(root / "active.json")
        if active.get("schemaVersion") != SCHEMA_VERSION or not active.get("manifestSha256"):
            raise RuntimeError("Invalid active fixed VQA Validation manifest")
        version = active.get("version")
        expected_hash = active["manifestSha256"]
    version = _version(version)
    manifest, digest = _read(root / version / "manifest.json", expected_hash)
    if (manifest.get("schemaVersion") != SCHEMA_VERSION or manifest.get("version") != version
            or manifest.get("protocol") != PROTOCOL or not isinstance(manifest.get("tasks"), dict)
            or manifest.get("taskCount") != len(manifest["tasks"])
            or manifest.get("targetCount") != sum(len(task["targets"]) for task in manifest["tasks"].values())):
        raise RuntimeError("Fixed VQA Validation manifest contract mismatch")
    return manifest, digest, version


def active_vqa_validation_info(service: Any) -> dict | None:
    import portable_tasks
    portable = portable_tasks.active_info(service)
    if portable is not None:
        return portable
    if not (service.vqa_validation_root / "active.json").exists():
        return None
    manifest, digest, version = _manifest(service, None)
    return {"version": version, "manifestSha256": digest,
            "taskCount": manifest["taskCount"], "targetCount": manifest["targetCount"],
            "protocol": PROTOCOL, "initialModelHoldoutIndependent": False,
            "referenceOnly": True}


def freeze_vqa_validation(service: Any, version: str, *, activate: bool = True) -> dict:
    """Explicitly create a new immutable version, then atomically activate it."""
    version = _version(version)
    root = service.vqa_validation_root
    root.mkdir(parents=True, exist_ok=True)
    directory = root / version
    directory.mkdir(exist_ok=False)
    created_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    with service.connect() as connection:
        records = connection.execute(
            "SELECT s.task_id, e.row_index FROM annotation_events e "
            "JOIN sessions s ON s.id=e.session_id "
            "UNION SELECT s.task_id, a.row_index FROM annotations a "
            "JOIN sessions s ON s.id=a.session_id"
        ).fetchall()
    history: dict[str, set[int]] = {}
    for task_id, row in records:
        history.setdefault(str(task_id), set()).add(int(row))
    manifest = {
        "schemaVersion": SCHEMA_VERSION, "version": version, "protocol": PROTOCOL,
        "createdAt": created_at, "tasks": {},
        "historySnapshotSha256": _sha(_json(sorted([str(task_id), int(row)] for task_id, row in records))),
        "description": "Fixed original-VQA 20% task-common Val; existing models are reference-only.",
        "initialModelHoldoutIndependent": False, "baseModelsRetrained": False,
        "evaluationLabelSource": "original-vqa-supervision", "usesPublicGroundTruth": False,
    }
    for task_id in sorted(service.tasks):
        task, bundle = service.task(task_id), service.bundle(task_id)
        source = _baseline_source(service, task_id)
        target_specs = {target["id"]: target for target in task.manifest["retrievalTargets"]}
        if "joint" not in task.target_ids or target_specs["joint"].get("kind") != "derived":
            raise RuntimeError(f"Task has no derived Joint VQA target: {task_id}")
        selected = service._probe_validation_split(task_id, "joint")
        if selected.audit.get("seed") != 0 or selected.audit.get("validationFraction") != 0.2:
            raise RuntimeError("Fixed VQA Validation requires the original seed-0 20% split")
        rows = sorted(_rows(selected.validation_indices, task.row_count, "validation"))
        selection = {key: selected.audit[key] for key in (
            "protocol", "replayKind", "seed", "validationFraction", "sourceFingerprint",
            "fitFingerprint", "validationFingerprint",
        )}
        targets = {}
        for target_id in task.target_ids:
            original = _original(service, task_id, target_id)
            targets[target_id] = _target_document(
                task_id, target_id, target_specs[target_id]["kind"], original, rows,
                bundle, history.get(task_id, set()), selection,
            )
        joint_lookup = dict(zip(targets["joint"]["sourceRows"], targets["joint"]["sourceLabels"], strict=True))
        selected_labels = _labels(selected.validation_labels, len(rows), "Joint replay")
        if selected_labels != [joint_lookup[int(row)] for row in selected.validation_indices]:
            raise RuntimeError("Joint replay labels differ from recovered VQA supervision")
        if len(rows) != (len(joint_lookup) + 4) // 5:
            raise RuntimeError("Joint VQA holdout is not the original 20% selection")
        if _baseline_source(service, task_id) != source:
            raise RuntimeError(f"VQA Validation source changed during freeze: {task_id}")
        document = {
            "schemaVersion": SCHEMA_VERSION, "version": version, "taskId": task_id,
            "source": source, "rows": rows,
            "imageIds": [bundle.image_ids[row] for row in rows],
            "historicalFeedbackRows": sorted(history.get(task_id, set())),
            "selection": selection, "targets": targets,
        }
        filename = f"{len(manifest['tasks']):03d}.json"
        digest = _write(directory / filename, document)
        manifest["tasks"][task_id] = {
            "file": filename, "sha256": digest, "count": len(rows),
            "targets": {key: {"positiveCount": sum(value["labels"]),
                              "fingerprint": value["audit"]["validationFingerprint"]}
                        for key, value in targets.items()},
        }
        print(f"Frozen VQA {task_id}: {len(rows)} rows / {len(targets)} targets", flush=True)
    if not manifest["tasks"]:
        raise RuntimeError("Cannot freeze an empty VQA Validation catalog")
    manifest["taskCount"] = len(manifest["tasks"])
    manifest["targetCount"] = sum(len(task["targets"]) for task in manifest["tasks"].values())
    manifest_hash = _write(directory / "manifest.json", manifest)
    for task_id, task in service.tasks.items():
        for target_id in task.target_ids:
            load_vqa_validation_split(service, task_id, target_id, version)
    if activate:
        _write(root / "active.json", {
            "schemaVersion": SCHEMA_VERSION, "version": version,
            "manifestSha256": manifest_hash, "activatedAt": created_at,
        })
    return manifest


def load_vqa_validation_split(service: Any, task_id: str, target_id: str,
                              version: str | None = None) -> Any:
    """Read fixed members and revalidate source/fit/label identities, never split."""
    import local_tasks
    if local_tasks.available(service, task_id):
        expected = local_tasks.settings(service, task_id)["version"]
        if version is not None and version != expected:
            raise RuntimeError("Local task Val version differs from its immutable input")
        version = expected
    import portable_tasks
    if portable_tasks.available(service, task_id):
        return portable_tasks.validation_split(service, task_id, target_id, version)
    import numpy as np
    from tuning_supervision import ProbeValidationSplit

    manifest, manifest_hash, version = _manifest(service, version)
    entry = manifest["tasks"].get(task_id)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Task is not in fixed VQA Validation {version}: {task_id}")
    filename = entry.get("file", "")
    if not isinstance(filename, str) or not re.fullmatch(r"[0-9]+\.json", filename) or not entry.get("sha256"):
        raise RuntimeError("Invalid fixed VQA Validation task file")
    document, task_hash = _read(service.vqa_validation_root / version / filename, entry["sha256"])
    task, bundle = service.task(task_id), service.bundle(task_id)
    if (document.get("schemaVersion") != SCHEMA_VERSION or document.get("version") != version
            or document.get("taskId") != task_id or document.get("source") != _baseline_source(service, task_id)):
        raise RuntimeError(f"Fixed VQA Validation source changed: {task_id}; restore or explicitly version a new split")
    if (target_id not in task.target_ids or set(document.get("targets", {})) != set(task.target_ids)
            or set(entry.get("targets", {})) != set(task.target_ids)):
        raise RuntimeError("Fixed VQA Validation target contract mismatch")
    rows = _rows(document.get("rows"), task.row_count, "validation", sorted_rows=True)
    if len(rows) != entry["count"] or document.get("imageIds") != [bundle.image_ids[row] for row in rows]:
        raise RuntimeError("Fixed VQA Validation image order or count mismatch")
    if any(bundle.test_mask[row] or row in bundle.query_indices for row in rows):
        raise RuntimeError("Fixed VQA Validation overlaps Frozen Test or Query")
    selection = document.get("selection", {})
    if selection.get("seed") != 0 or selection.get("validationFraction") != 0.2:
        raise RuntimeError("Invalid fixed VQA Validation selection contract")
    history = set(document.get("historicalFeedbackRows", []))
    target_specs = {target["id"]: target for target in task.manifest["retrievalTargets"]}
    # Validate every target so a shared row can never silently lose an attribute label.
    for candidate in task.target_ids:
        original = _original(service, task_id, candidate)
        expected = _target_document(task_id, candidate, target_specs[candidate]["kind"],
                                    original, rows, bundle, history, selection)
        actual = document["targets"][candidate]
        if actual != expected:
            raise RuntimeError(f"Fixed VQA Validation supervision, fit or labels changed: {task_id}/{candidate}")
        summary = {"positiveCount": sum(expected["labels"]),
                   "fingerprint": expected["audit"]["validationFingerprint"]}
        if entry["targets"][candidate] != summary:
            raise RuntimeError("Fixed VQA Validation target summary mismatch")
    target = document["targets"][target_id]
    audit = dict(target["audit"])
    audit.update({"frozenVersion": version, "frozenAt": manifest["createdAt"],
                  "frozenManifestSha256": manifest_hash, "frozenTaskSha256": task_hash})
    arrays = (np.asarray(target["fitRows"], dtype=np.int64),
              np.asarray(target["fitLabels"], dtype=np.uint8),
              np.asarray(rows, dtype=np.int64), np.asarray(target["labels"], dtype=np.uint8))
    for array in arrays:
        array.setflags(write=False)
    return ProbeValidationSplit(fit_indices=arrays[0], fit_labels=arrays[1],
                                validation_indices=arrays[2], validation_labels=arrays[3], audit=audit)


if __name__ == "__main__":
    import argparse
    from tuning_server import TuningService

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    service = TuningService(args.web_root, args.runtime_root, start_worker=False)
    result = freeze_vqa_validation(service, args.version)
    print(json.dumps({"version": result["version"], "tasks": result["taskCount"],
                      "targets": result["targetCount"], "path": str(service.vqa_validation_root)},
                     ensure_ascii=False), flush=True)
