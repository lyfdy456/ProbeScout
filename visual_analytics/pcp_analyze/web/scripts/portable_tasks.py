"""Read published task inputs without the author's acquisition or session folders.

The release pins the clean package manifest. Original Val hashes are retained as
provenance so existing model-bank identities remain valid; they are NOT claimed
to be hashes of the clean files, which have their own verified package hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = "probescout-portable-tasks-v1"
DIRECTORIES = {"cars": "stanford_cars", "hico": "HICO", "celeba": "CelebA"}


def root(service):
    return service.web_root.parents[2]


def available(service, task_id):
    dataset = getattr(service.task(task_id), "dataset_id", None)
    if dataset not in DIRECTORIES:
        return False
    return (root(service) / "dataset/tasks" / dataset / "manifest.json").is_file()


def _read(service, path, expected):
    path = Path(path).resolve()
    if not path.is_relative_to(root(service).resolve()):
        raise RuntimeError("Portable task path escapes the repository")
    stamp = (path.stat().st_size, path.stat().st_mtime_ns, expected)
    cache = getattr(service, "_portable_file_cache", {})
    cached = cache.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"Portable task checksum mismatch: {path.name}")
    value = json.loads(data)
    cache[str(path)] = stamp, value
    service._portable_file_cache = cache
    return value


def manifest(service, dataset):
    index = json.loads((root(service) / "manifests/task_packages.json").read_text(encoding="utf-8"))
    pin = index["datasets"][dataset]
    value = _read(service, root(service) / "dataset/tasks" / dataset / "manifest.json", pin["manifestSha256"])
    if value.get("protocol") != PROTOCOL or value.get("dataset") != dataset:
        raise RuntimeError("Portable task manifest identity mismatch")
    return value, pin["manifestSha256"]


def document(service, task_id):
    task = service.task(task_id)
    publication, digest = manifest(service, task.dataset_id)
    entry = publication["tasks"][task_id]
    value = _read(service, root(service) / entry["path"], entry["sha256"])
    if value.get("taskId") != task_id or value.get("protocol") != PROTOCOL:
        raise RuntimeError("Portable task identity mismatch")
    contract_spec = value["isolationContract"]
    contract = _read(service, root(service) / contract_spec["path"], contract_spec["sha256"])
    return value, contract, digest


def validation_split(service, task_id, target_id, version=None):
    import fixed_vqa_validation as fixed
    from tuning_supervision import ProbeValidationSplit
    from val_isolation import digest

    value, contract, package_hash = document(service, task_id)
    bundle = service.bundle(task_id)
    task = service.task(task_id)
    provenance = value["validationProvenance"]
    if version is not None and version != provenance["version"]:
        raise RuntimeError("Portable task does not contain the requested Val version")
    if value["source"] != fixed._baseline_source(service, task_id):
        raise RuntimeError("Portable task image/partition identity mismatch")
    body = dict(contract)
    fingerprint = body.pop("fingerprint", None)
    if fingerprint != digest(body) or contract["taskId"] != task_id:
        raise RuntimeError("Portable isolation contract fingerprint mismatch")
    if (contract["valVersion"] != provenance["version"]
            or contract["valManifestSha256"] != provenance["manifestSha256"]
            or contract["valTaskSha256"] != provenance["taskSha256"]):
        raise RuntimeError("Portable Val provenance differs from the frozen bank")
    rows = fixed._rows(value["validationRows"], task.row_count, "Val", sorted_rows=True)
    if rows != contract["valRows"] or any(bundle.test_mask[r] or r in bundle.query_indices for r in rows):
        raise RuntimeError("Portable Val rows overlap a protected partition or differ from the bank")
    if value["validationImageIds"] != [bundle.image_ids[r] for r in rows]:
        raise RuntimeError("Portable Val image order differs")
    if set(value["targets"]) != set(task.target_ids) or target_id not in task.target_ids:
        raise RuntimeError("Portable target set differs from the Web task")
    if value["selection"].get("seed") != 0 or value["selection"].get("validationFraction") != .2:
        raise RuntimeError("Portable Val selection contract differs")
    audits = {}
    for name, target in value["targets"].items():
        source = fixed._rows(target["sourceRows"], task.row_count, "source")
        labels = fixed._labels(target["sourceLabels"], len(source), "source")
        if any(bundle.test_mask[r] or r in bundle.query_indices for r in source):
            raise RuntimeError("Portable original supervision overlaps Test or Query")
        lookup = dict(zip(source, labels, strict=True))
        fit = [r for r in source if r not in set(rows)]
        if (not set(rows) <= set(source) or target["fitRows"] != fit
                or target["fitLabels"] != [lookup[r] for r in fit]
                or target["labels"] != [lookup[r] for r in rows]):
            raise RuntimeError("Portable original/fit/Val labels are inconsistent")
        frozen = contract["targets"][name]
        if frozen["fitRows"] != fit or frozen["fitLabels"] != target["fitLabels"] or frozen["valLabels"] != target["labels"]:
            raise RuntimeError("Portable labels differ from the frozen training contract")
        audit = {"schemaVersion": 1, "protocol": fixed.PROTOCOL,
                 "targetId": name, "targetKind": "derived" if name == "joint" else "attribute",
                 "replayKind": "task-common-derived-joint-seed0", "seed": 0,
                 "validationFraction": .2, "selection": value["selection"],
                 "selectionTargetId": "joint", "evaluationLabelSource": "original-vqa-supervision",
                 "usesPublicGroundTruth": False, "initialModelHoldoutIndependent": False,
                 "referenceOnly": True, "baseModelsRetrained": False,
                 "fitProtocol": "original-vqa-minus-shared-validation-v1",
                 "sourceSupervisionHash": target["supervisionHash"],
                 "frozenVersion": provenance["version"], "frozenAt": provenance["createdAt"],
                 "frozenManifestSha256": provenance["manifestSha256"],
                 "frozenTaskSha256": provenance["taskSha256"],
                 "portableManifestSha256": package_hash}
        for part, indices, truth, contract_key in [
            ("source", source, labels, "sourceFingerprint"),
            ("fit", fit, target["fitLabels"], "fitFingerprint"),
            ("validation", rows, target["labels"], "valFingerprint"),
        ]:
            actual = fixed._fingerprint(part, task_id, name, indices, truth, bundle.image_ids)
            if actual != frozen[contract_key]:
                raise RuntimeError("Portable label fingerprint mismatch")
            audit.update({f"{part}Fingerprint": actual, f"{part}Count": len(indices),
                          f"{part}PositiveCount": sum(truth), f"{part}NegativeCount": len(truth)-sum(truth),
                          f"{part}RowsSha256": digest(indices),
                          f"{part}ImageIdsSha256": digest([bundle.image_ids[r] for r in indices]),
                          f"{part}LabelsSha256": digest(truth)})
        audit["sourceSupervisionFingerprint"] = audit["sourceFingerprint"]
        audits[name] = audit
    target = value["targets"][target_id]
    arrays = [np.asarray(v, dtype=np.int64 if i % 2 == 0 else np.uint8) for i, v in enumerate(
        [target["fitRows"], target["fitLabels"], rows, target["labels"]])]
    for array in arrays:
        array.setflags(write=False)
    return ProbeValidationSplit(*arrays, audit=audits[target_id])


def original_supervision(service, task_id, target_id):
    from tuning_supervision import OriginalSupervision
    validation_split(service, task_id, target_id)
    value, _, _ = document(service, task_id)
    selected = value["targets"][target_id]
    rows = np.asarray(selected["sourceRows"], dtype=np.int64)
    labels = np.asarray(selected["sourceLabels"], dtype=np.uint8)
    mask = np.frombuffer(service.bundle(task_id).development_mask, dtype=np.uint8)[rows].astype(bool)
    audit = {"recoveredSupervisionHash": selected["supervisionHash"],
             "originalSupervisionCount": len(rows), "developmentCount": int(mask.sum()),
             "positiveCount": int(labels[mask].sum()), "negativeCount": int(mask.sum()-labels[mask].sum()),
             "excludedNonDevelopmentCount": int((~mask).sum()),
             "labelSource": "original-vqa-supervision"}
    return OriginalSupervision(rows[mask], labels[mask], rows, labels, audit)


def active_info(service):
    tasks = [tid for tid in service.tasks if available(service, tid)]
    if not tasks:
        return None
    docs = [document(service, tid)[0] for tid in tasks]
    identities = {(d["validationProvenance"]["version"], d["validationProvenance"]["manifestSha256"]) for d in docs}
    if len(identities) != 1:
        raise RuntimeError("Installed task packages use different Val versions")
    version, checksum = identities.pop()
    return {"version": version, "manifestSha256": checksum, "taskCount": len(tasks),
            "targetCount": sum(len(d["targets"]) for d in docs),
            "protocol": "fixed-vqa-joint-seed0-holdout-v1",
            "initialModelHoldoutIndependent": False, "referenceOnly": True}


def adapter(service, task_id):
    """Construct the existing adapter interface from released task inputs."""
    import sys
    import pandas as pd
    learning = root(service) / "probe_learning"
    if str(learning) not in sys.path:
        sys.path.insert(0, str(learning))
    from src.data.dataset_adapter import DatasetAdapter
    task = service.task(task_id)
    value, _, _ = document(service, task_id)
    processed = root(service) / "dataset/raw" / DIRECTORIES[task.dataset_id] / "processed"
    publication, _ = manifest(service, task.dataset_id)
    records_path = processed / "records.csv"
    expected = publication["files"][records_path.relative_to(root(service)).as_posix()]
    if hashlib.sha256(records_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Dataset record order differs from the released task package")
    records = pd.read_csv(records_path)
    images = list(service.bundle(task_id).image_ids)
    if records["relative_path"].tolist() != images or records["embedding_index"].tolist() != list(range(len(images))):
        raise RuntimeError("Portable records and Web image order differ")
    attributes = value["attributes"]
    raw = processed.parent / ("img_celeba" if task.dataset_id == "celeba" else "images")
    return DatasetAdapter(dataset=task.dataset_id, task=task.task_name,
        attrs=[a["name"] for a in attributes], key_slugs=[a["id"] for a in attributes],
        joint_label=value["jointLabel"], emb_path=processed/"siglip_embedding.npy",
        patch_path=processed/"siglip_patch_tokens.npy", records=records,
        raw_jsonl_path=root(service)/"dataset/tasks"/task.dataset_id/task.task_name/"original_vqa.json",
        vqa_to_key=lambda text: str(text).replace("\\", "/"),
        query_idx=sorted(service.bundle(task_id).query_indices), raw_images_dir=raw,
        gt_by_attr={a["name"]: {} for a in attributes},
        task_root=root(service)/"dataset/tasks"/task.dataset_id/task.task_name)
