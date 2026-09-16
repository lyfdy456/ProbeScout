"""Explicitly freeze Clean Validation; serving never rebuilds its membership."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
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
            f"Frozen Clean Validation unavailable or corrupt: {path.name}; "
            "restore the frozen files (no automatic resampling)"
        ) from error


def _version(value: str) -> str:
    if not isinstance(value, str) or not VERSION_PATTERN.fullmatch(value):
        raise RuntimeError("Invalid frozen Clean Validation version")
    return value


def baseline_source(service: Any, task_id: str) -> dict:
    """Bind indices to the complete ordered gallery, labels and scope masks."""
    task = service.task(task_id)
    bundle = service.bundle(task_id)
    truth = service._task_ground_truth(task_id)
    return {
        "rowCount": task.row_count,
        "targetIds": list(task.target_ids),
        "imageIdsSha256": _sha(_json(list(bundle.image_ids))),
        "developmentMaskSha256": _sha(bytes(bundle.development_mask)),
        "validationMaskSha256": _sha(bytes(bundle.validation_mask)),
        "testMaskSha256": _sha(bytes(bundle.test_mask)),
        "querySha256": _sha(_json(sorted(int(row) for row in bundle.query_indices))),
        "groundTruthSha256": _sha(truth.tobytes(order="C")),
    }


def active_validation_info(service: Any) -> dict | None:
    """Small health/provenance record; split validation still happens per run."""
    root = service.clean_validation_root
    if not (root / "active.json").exists():
        return None
    active, _ = _read(root / "active.json")
    version = _version(active.get("version"))
    if active.get("schemaVersion") != SCHEMA_VERSION or not active.get("manifestSha256"):
        raise RuntimeError("Invalid active Clean Validation manifest")
    manifest, digest = _read(root / version / "manifest.json", active["manifestSha256"])
    if manifest.get("version") != version or manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise RuntimeError("Frozen Clean Validation manifest version mismatch")
    return {"version": version, "manifestSha256": digest,
            "taskCount": manifest["taskCount"], "targetCount": manifest["targetCount"]}


def freeze_clean_validation(service: Any, version: str) -> dict:
    """Freeze the current catalog against one consistent feedback-DB snapshot.

    Existing versions are never overwritten. Only a complete, verified export
    can replace active.json; an interrupted export cannot become active.
    """
    import numpy as np

    version = _version(version)
    root = service.clean_validation_root
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
        "schemaVersion": SCHEMA_VERSION, "version": version,
        "createdAt": created_at, "tasks": {},
        "historySnapshotSha256": _sha(_json(sorted(
            [str(task_id), int(row)] for task_id, row in records
        ))),
        "description": "Fixed label-clean transductive Web Validation; not an inductive holdout.",
    }
    for task_id in sorted(service.tasks):
        task = service.task(task_id)
        bundle = service.bundle(task_id)
        source = baseline_source(service, task_id)
        targets = {}
        shared_rows = None
        for target_id in task.target_ids:
            split = service._build_clean_validation_split(
                task_id, target_id, reviewed_rows=history.get(task_id, set())
            )
            rows = split.validation_indices.tolist()
            if shared_rows is not None and rows != shared_rows:
                raise RuntimeError(f"Clean Validation membership differs by target: {task_id}")
            shared_rows = rows
            targets[target_id] = {
                "labels": split.validation_labels.tolist(), "audit": dict(split.audit),
            }
        if not shared_rows or baseline_source(service, task_id) != source:
            raise RuntimeError(f"Clean Validation source changed during freeze: {task_id}")
        # Current training scope must already exclude this fixed evaluation set.
        if np.any(np.frombuffer(bundle.development_mask, dtype=np.uint8)[shared_rows]):
            raise RuntimeError(f"Clean Validation overlaps Development: {task_id}")
        task_document = {
            "schemaVersion": SCHEMA_VERSION, "version": version,
            "taskId": task_id, "source": source, "rows": shared_rows,
            "imageIds": [bundle.image_ids[row] for row in shared_rows],
            "targets": targets,
        }
        filename = f"{len(manifest['tasks']):03d}.json"
        digest = _write(directory / filename, task_document)
        manifest["tasks"][task_id] = {
            "file": filename, "sha256": digest, "count": len(shared_rows),
            "targets": {key: {"positiveCount": sum(value["labels"]),
                              "fingerprint": value["audit"]["validationFingerprint"]}
                        for key, value in targets.items()},
        }
        print(f"Frozen {task_id}: {len(shared_rows)} rows / {len(targets)} targets", flush=True)

    manifest["taskCount"] = len(manifest["tasks"])
    manifest["targetCount"] = sum(len(task["targets"]) for task in manifest["tasks"].values())
    manifest_hash = _write(directory / "manifest.json", manifest)
    # Re-read through the serving path before activation, including any new
    # historical contamination that arrived after the initial DB snapshot.
    for task_id, task in service.tasks.items():
        for target_id in task.target_ids:
            load_frozen_validation_split(service, task_id, target_id, version=version)
    _write(root / "active.json", {
        "schemaVersion": SCHEMA_VERSION, "version": version,
        "manifestSha256": manifest_hash, "activatedAt": created_at,
    })
    return manifest


def load_frozen_validation_split(
    service: Any, task_id: str, target_id: str, *, version: str | None = None,
) -> Any:
    import numpy as np
    from tuning_supervision import ProbeValidationSplit

    root = service.clean_validation_root
    expected_hash = None
    if version is None:
        active, _ = _read(root / "active.json")
        version = active.get("version")
        expected_hash = active.get("manifestSha256")
        if not expected_hash or active.get("schemaVersion") != SCHEMA_VERSION:
            raise RuntimeError("Invalid active Clean Validation manifest")
    version = _version(version)
    directory = root / version
    manifest, manifest_hash = _read(directory / "manifest.json", expected_hash)
    if manifest.get("version") != version or manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise RuntimeError("Frozen Clean Validation manifest version mismatch")
    entry = manifest.get("tasks", {}).get(task_id)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Task is not in frozen Clean Validation {version}: {task_id}")
    filename = entry.get("file", "")
    if not re.fullmatch(r"[0-9]+\.json", filename) or not entry.get("sha256"):
        raise RuntimeError("Invalid frozen Clean Validation task file")
    document, digest = _read(directory / filename, entry["sha256"])
    task = service.task(task_id)
    bundle = service.bundle(task_id)
    if (document.get("schemaVersion") != SCHEMA_VERSION
            or document.get("version") != version or document.get("taskId") != task_id
            or document.get("source") != baseline_source(service, task_id)):
        raise RuntimeError(f"Frozen Clean Validation source changed: {task_id}; restore or explicitly version a new split")
    if target_id not in task.target_ids or target_id not in document.get("targets", {}):
        raise RuntimeError(f"Target is not in frozen Clean Validation: {task_id}/{target_id}")
    rows = np.asarray(document["rows"], dtype=np.int64)
    if (rows.ndim != 1 or rows.size == 0 or rows.min() < 0 or rows.max() >= task.row_count
            or not np.all(rows[1:] > rows[:-1]) or rows.size != entry["count"]):
        raise RuntimeError("Invalid frozen Clean Validation row indices")
    if document["imageIds"] != [bundle.image_ids[int(row)] for row in rows]:
        raise RuntimeError("Frozen Clean Validation image order mismatch")
    selected = set(int(row) for row in rows)
    if (not all(bundle.validation_mask[row] for row in selected)
            or any(bundle.development_mask[row] or bundle.test_mask[row] for row in selected)
            or selected.intersection(bundle.query_indices)):
        raise RuntimeError("Frozen Clean Validation overlaps Development, Test or Query")
    # History is a safety assertion only: removing/adding unrelated feedback
    # cannot change membership or the frozen exclusion audit.
    if selected.intersection(service._historically_reviewed_rows(task_id)):
        raise RuntimeError("Frozen Clean Validation overlaps historical feedback; membership was not changed")

    target = document["targets"][target_id]
    audit = dict(target["audit"])
    original_fingerprints = {}
    for candidate in task.target_ids:
        original = service._original_development_supervision(task_id, candidate)
        if selected.intersection(int(row) for row in original.selected_indices):
            raise RuntimeError("Frozen Clean Validation overlaps original VQA supervision")
        original_fingerprints[candidate] = str(original.audit.get("recoveredSupervisionHash") or "")
    if original_fingerprints != audit.get("originalSupervisionFingerprints"):
        raise RuntimeError("Original supervision changed since Clean Validation was frozen")
    historical_fit = service._probe_validation_split(task_id, target_id)
    for key in ("fitFingerprint", "sourceFingerprint"):
        if historical_fit.audit[key] != audit[key]:
            raise RuntimeError(f"Frozen Clean Validation {key} changed")
    labels = np.asarray(target["labels"], dtype=np.uint8)
    truth = service._task_ground_truth(task_id)[rows, task.target_ids.index(target_id)]
    if labels.shape != rows.shape or not np.array_equal(labels, truth):
        raise RuntimeError("Frozen Clean Validation labels changed")
    fingerprint = _sha(_json({
        "partition": "clean-validation", "protocol": audit["protocol"],
        "taskId": task_id, "targetId": target_id,
        "records": [[bundle.image_ids[int(row)], int(label)]
                    for row, label in zip(rows, labels, strict=True)],
    }))
    if (fingerprint != audit["validationFingerprint"]
            or rows.size != audit["validationCount"]
            or int(labels.sum()) != audit["validationPositiveCount"]
            or int(rows.size - labels.sum()) != audit["validationNegativeCount"]
            or not 0 < labels.sum() < rows.size):
        raise RuntimeError("Frozen Clean Validation target audit mismatch")
    audit.update({"frozenVersion": version, "frozenAt": manifest["createdAt"],
                  "frozenManifestSha256": manifest_hash, "frozenTaskSha256": digest})
    rows.setflags(write=False)
    labels.setflags(write=False)
    return ProbeValidationSplit(
        fit_indices=historical_fit.fit_indices, fit_labels=historical_fit.fit_labels,
        validation_indices=rows, validation_labels=labels, audit=audit,
    )


if __name__ == "__main__":
    import argparse
    from tuning_server import TuningService

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    service = TuningService(args.web_root, args.runtime_root, start_worker=False)
    result = freeze_clean_validation(service, args.version)
    print(json.dumps({"version": result["version"], "tasks": result["taskCount"],
                      "targets": result["targetCount"], "path": str(service.clean_validation_root)},
                     ensure_ascii=False), flush=True)
