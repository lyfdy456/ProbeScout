"""Strict task-local frozen-parent augmentation; never rewrites a formal task.

The extra static Val images are a separately reserved robustness audit, not
fabricated VQA validation labels. Weight Tune retains the parent's frozen VQA
holdout; new DG images only acquire training labels through human feedback.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

TASK_ID = "059_hico_task_hico_hugging_cat_robust_test"
PARENT_ID = "032_hico_task_hico_hugging_cat"
PROTOCOL = "frozen-parent-gallery-augmentation-v1"


def applies(task_id):
    return task_id == TASK_ID


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def directory(service):
    return service.web_root.parent / "runtime/experimental-tasks" / TASK_ID


def contract(service):
    folder = directory(service)
    value = _read(folder / "overlay.json")
    if (value.get("protocol") != PROTOCOL or value.get("taskId") != TASK_ID
            or value.get("parentTaskId") != PARENT_ID or value.get("complete") is not True
            or value.get("modelsRetrained") is not False or value.get("vqaRun") is not False):
        raise RuntimeError("Experimental task contract is invalid or incomplete")
    parent = service.task(PARENT_ID)
    if value["parentRowCount"] != parent.row_count:
        raise RuntimeError("Experimental parent Gallery has changed")
    from unified_initial_baseline import load_task_baseline
    parent_base, parent_metadata, _ = load_task_baseline(service, PARENT_ID, value["parentPublicationVersion"],
                                                       fingerprint=value["parentBaseStateFingerprint"])
    if parent_base.base_state_fingerprint != value["parentBaseStateFingerprint"]:
        raise RuntimeError("Experimental parent baseline no longer matches")
    # The immutable input identities, not the mutable active publication, bind this task.
    for name in ("validation.json", "inference.json", "base/manifest.json"):
        if _sha(folder / name) != value["files"][name]:
            raise RuntimeError(f"Experimental task artifact checksum mismatch: {name}")
    if value["parentBankFingerprint"] != parent_metadata["bankFingerprint"]:
        raise RuntimeError("Experimental task belongs to another Probe bank")
    if applies(TASK_ID) and TASK_ID in service.tasks:
        if service.bundle(TASK_ID).image_ids[:parent.row_count] != service.bundle(PARENT_ID).image_ids:
            raise RuntimeError("Experimental parent image-ID prefix has changed")
    return value


def validate_registration(service, entry, manifest):
    if entry.get("id") != TASK_ID or entry.get("catalogRole") != "experimental":
        raise RuntimeError("Unknown experimental task; explicit registration required")
    value = _read(directory(service) / "overlay.json")
    if value.get("protocol") != PROTOCOL or value.get("parentTaskId") != PARENT_ID or value.get("complete") is not True:
        raise RuntimeError("Experimental registration is incomplete")
    registered = entry.get("experimentalOverlay", {})
    if registered.get("protocol") != PROTOCOL or registered.get("manifestSha256") != _sha(directory(service) / "overlay.json"):
        raise RuntimeError("Experimental catalog source identity mismatch")
    if (manifest.get("experimentalOverlay") != registered
            or manifest.get("rowCount") != value["parentRowCount"] + len(value["rows"])):
        raise RuntimeError("Experimental browser bundle is not aligned")


def metadata(service, task_id, version=None, *, fingerprint=None):
    if not applies(task_id):
        return None
    value = contract(service)
    if version is not None and version != value["version"]:
        raise RuntimeError("Unknown experimental publication; cannot fall back to parent")
    folder = directory(service) / "base"
    meta = _read(folder / "manifest.json")
    if (meta.get("protocol") != "isolated-initial-baseline-v1" or meta.get("verified") is not True
            or meta["taskId"] != TASK_ID or meta["version"] != value["version"]
            or meta["baseStateFingerprint"] != value["baseStateFingerprint"]
            or meta.get("frozenParent", {}).get("baseStateFingerprint") != value["parentBaseStateFingerprint"]
            or fingerprint is not None and fingerprint != meta["baseStateFingerprint"]):
        raise RuntimeError("Experimental F0 identity mismatch")
    return meta, folder


def original_supervision(service, task_id, target_id):
    value = contract(service)
    source = service._original_development_supervision(PARENT_ID, target_id)
    if np.any(source.selected_indices >= value["parentRowCount"]):
        raise RuntimeError("Parent VQA rows extend beyond the original Gallery")
    frozen = _read(directory(service) / "validation.json")["targets"][target_id]
    expected = dict(zip(frozen["fitRows"] + frozen["validationRows"], frozen["fitLabels"] + frozen["validationLabels"]))
    current = dict(zip(map(int, source.selected_indices), map(int, source.selected_labels)))
    if current != expected:
        raise RuntimeError("Parent VQA labels no longer match this experiment's frozen supervision")
    return dataclasses.replace(source, audit={**source.audit, "taskId": TASK_ID,
        "parentTaskId": PARENT_ID, "augmentationPolicy": "no-vqa-labels-for-generated-images"})


def validation_split(service, task_id, target_id, version=None):
    from tuning_supervision import ProbeValidationSplit
    value = contract(service)
    data = _read(directory(service) / "validation.json")
    if version is not None and version != data["version"]:
        raise RuntimeError("Experimental fixed Val version mismatch")
    row = data["targets"][target_id]
    arrays = [np.asarray(row[name], dtype=dtype) for name, dtype in (
        ("fitRows", np.int64), ("fitLabels", np.uint8),
        ("validationRows", np.int64), ("validationLabels", np.uint8))]
    fit, labels, val, truth = arrays
    if (len(fit) != len(labels) or len(val) != len(truth) or not len(fit) or not len(val)
            or np.intersect1d(fit, val).size or np.any(fit < 0) or np.any(val < 0)
            or len(np.unique(fit)) != len(fit) or len(np.unique(val)) != len(val)
            or np.any(fit >= value["parentRowCount"])
            or np.any(val >= value["parentRowCount"]) or not np.isin(labels, [0, 1]).all()
            or not np.isin(truth, [0, 1]).all()):
        raise RuntimeError("Experimental inherited VQA/Val membership is invalid")
    for array in arrays:
        array.setflags(write=False)
    audit = {**row["audit"], "taskId": TASK_ID, "parentTaskId": PARENT_ID,
             "frozenVersion": data["version"], "frozenManifestSha256": value["files"]["validation.json"],
             "extraSyntheticValUsedForSelection": False}
    return ProbeValidationSplit(fit, labels, val, truth, audit)


def gallery_image(service, row_index):
    value = contract(service)
    if not isinstance(row_index, int) or not 0 <= row_index < value["parentRowCount"] + len(value["rows"]):
        raise ValueError("Experimental image row is out of range")
    if row_index < value["parentRowCount"]:
        return service.gallery_image(PARENT_ID, row_index)
    record = value["rows"][row_index - value["parentRowCount"]]
    root = service.web_root.parents[2] / "dataset/tasks/hico/task_hico_hugging_cat_robust_test"
    path = (root / record["path"]).resolve()
    if not path.is_relative_to((root / "generated").resolve()) or _sha(path) != record["sha256"]:
        raise RuntimeError("Experimental image source identity mismatch")
    return path
