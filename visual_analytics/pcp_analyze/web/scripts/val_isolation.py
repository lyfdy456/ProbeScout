"""Immutable, task-common Val isolation for the next Probe training run.

This prepares data contracts, not models. Existing Val members, source labels,
published scores and historical runs are never rewritten. Full-gallery inference
is allowed; fitting pools must exclude Val, Frozen Test and Query.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fixed_vqa_validation as fixed

PROTOCOL = "task-common-val-isolation-v1"
SEEDS = (0, 1, 2, 3, 4)
METHODS = (
    "mlp_baseline", "kfold_pu", "triplet_loss", "attention_pooling",
    "attribute_conditioned_attention", "nnpu", "dcpu", "pu_ranking",
)


def digest(value: Any) -> str:
    return hashlib.sha256(fixed._json(value)).hexdigest()


def partition_rows(row_count: int, development: bytes, test: bytes,
                   query: set[int], val: list[int], source: dict) -> dict:
    """Pure partition construction. Does not inspect public GT."""
    if len(development) != row_count or len(test) != row_count:
        raise RuntimeError("Isolation mask row count mismatch")
    if any(value not in (0, 1) for value in development + test):
        raise RuntimeError("Isolation masks must be binary")
    val_set = set(fixed._rows(val, row_count, "Val", sorted_rows=True))
    if any(type(row) is not int or not 0 <= row < row_count for row in query):
        raise RuntimeError("Invalid Query row")
    test_set = {row for row, flag in enumerate(test) if flag}
    if val_set & (test_set | query) or test_set & query:
        raise RuntimeError("Val, Test and Query must be disjoint")
    blocked = val_set | test_set | query
    train_pool = [row for row in range(row_count) if row not in blocked]
    normalization = [row for row in train_pool if development[row]]
    if not normalization:
        raise RuntimeError("Empty Val-isolated normalization pool")
    targets = {}
    for target, split in source.items():
        fit = [int(row) for row in split.fit_indices]
        labels = [int(label) for label in split.fit_labels]
        if not fit or len(set(fit)) != len(fit) or any(row in blocked for row in fit):
            raise RuntimeError(f"Invalid Val-isolated fit rows: {target}")
        if any(not 0 <= row < row_count for row in fit):
            raise RuntimeError("Fit row outside gallery")
        if [int(row) for row in split.validation_indices] != val:
            raise RuntimeError("All attributes and seeds must share the same Val")
        fixed._labels(labels, len(fit), "fit")
        val_labels = fixed._labels(split.validation_labels, len(val), "Val")
        targets[target] = {
            "fitRows": fit, "fitLabels": labels, "valLabels": val_labels,
            "sourceFingerprint": split.audit["sourceFingerprint"],
            "fitFingerprint": split.audit["fitFingerprint"],
            "valFingerprint": split.audit["validationFingerprint"],
        }
    return {
        "valRows": val, "testRows": sorted(test_set), "queryRows": sorted(query),
        "trainPoolRows": train_pool, "normalizationRows": normalization,
        "targets": targets,
    }


def build_task_contract(service: Any, task_id: str, val_version: str) -> dict:
    task, bundle = service.task(task_id), service.bundle(task_id)
    context = service._tuning_source_context(task_id)
    splits = {target: service._vqa_validation_split(task_id, target, val_version)
              for target in task.target_ids}
    joint = splits["joint"]
    rows = [int(row) for row in joint.validation_indices]
    pools = partition_rows(task.row_count, bundle.development_mask,
                           bundle.test_mask, set(bundle.query_indices), rows, splits)
    attributes = [target for target in task.manifest["retrievalTargets"]
                  if target["kind"] == "attribute"]
    if len(attributes) != len(context.attributes):
        raise RuntimeError("Isolation attribute identity mismatch")
    result = {
        "schemaVersion": 1, "protocol": PROTOCOL, "taskId": task_id,
        "dataset": task.dataset_id, "taskName": task.task_name,
        "rowCount": task.row_count, "imageIdsSha256": digest(list(bundle.image_ids)),
        "valVersion": val_version,
        "valManifestSha256": joint.audit["frozenManifestSha256"],
        "valTaskSha256": joint.audit["frozenTaskSha256"],
        "attributes": [{"id": target["id"], "name": name}
                       for target, name in zip(attributes, context.attributes, strict=True)],
        "methods": list(METHODS), "seeds": list(SEEDS),
        "labelSource": joint.audit.get("evaluationLabelSource", "original-vqa-supervision"), "usesPublicGroundTruth": False,
        "trainingPolicy": "fit-only-minibatches; fixed-Val-checkpoint-selection",
        "unlabeledPolicy": "trainPool-minus-all-original-supervised-fit-images",
        "normalizationPolicy": "static-Development-minus-Val-Test-Query",
        "calibrationPolicy": "original-VQA-fit-only; no-Val-labels",
        "crossTaskReuse": False, "legacyCacheReuse": False,
        "modelsRetrained": False,
        **pools,
    }
    result["poolFingerprints"] = {
        key: digest([bundle.image_ids[row] for row in result[key]])
        for key in ("valRows", "testRows", "queryRows", "trainPoolRows", "normalizationRows")
    }
    result["fingerprint"] = digest(result)
    return result


def materialize_training_inputs(contract: dict, image_ids: list[str],
                                offline_paths: list[str]) -> dict:
    """Map by image ID, never assume Web rows equal offline embedding rows."""
    check = dict(contract)
    fingerprint = check.pop("fingerprint", None)
    if fingerprint != digest(check) or contract.get("protocol") != PROTOCOL:
        raise RuntimeError("Val isolation contract fingerprint mismatch")
    if (len(image_ids) != contract["rowCount"] or len(set(image_ids)) != len(image_ids)
            or digest(image_ids) != contract["imageIdsSha256"]):
        raise RuntimeError("Val isolation gallery identity mismatch")
    if len(set(offline_paths)) != len(offline_paths) or set(offline_paths) != set(image_ids):
        raise RuntimeError("Offline/gallery image identities differ")
    p2i = {path: index for index, path in enumerate(offline_paths)}
    val_rows = contract["valRows"]
    targets = contract["targets"]
    fit_rows = sorted({row for attr in contract["attributes"]
                       for row in targets[attr["id"]]["fitRows"]})
    lookups = {attr["id"]: dict(zip(targets[attr["id"]]["fitRows"],
                                    targets[attr["id"]]["fitLabels"], strict=True))
               for attr in contract["attributes"]}
    fit_labels = [{"image": image_ids[row], **{
        attr["name"]: lookups[attr["id"]].get(row) for attr in contract["attributes"]
    }} for row in fit_rows]
    val_labels = [{"image": image_ids[row], **{
        attr["name"]: targets[attr["id"]]["valLabels"][index]
        for attr in contract["attributes"]
    }} for index, row in enumerate(val_rows)]
    for attr in contract["attributes"]:
        name = attr["name"]
        if any(record[name] not in (0, 1) for record in fit_labels + val_labels):
            raise RuntimeError(f"Incomplete fixed training/Val attribute labels: {name}")
        if min(sum(record[name] == label for record in fit_labels) for label in (0, 1)) < 2:
            raise RuntimeError(f"Insufficient fixed training class counts: {name}")
        if {record[name] for record in val_labels} != {0, 1}:
            raise RuntimeError(f"Val cannot select checkpoints with one class: {name}")
    fit_paths = [image_ids[row] for row in fit_rows]
    train_pool = [image_ids[row] for row in contract["trainPoolRows"]]
    unlabeled = set(train_pool) - set(fit_paths)
    blocked = {image_ids[row] for key in ("valRows", "testRows", "queryRows")
               for row in contract[key]}
    if (set(fit_paths) | unlabeled) & blocked:
        raise RuntimeError("Val/Test/Query leaked into fitting inputs")
    normalization = [image_ids[row] for row in contract["normalizationRows"]]
    if set(normalization) & blocked:
        raise RuntimeError("Val/Test/Query leaked into normalization")
    return {
        "fit_paths": fit_paths, "fit_labels": fit_labels, "val_labels": val_labels,
        "train_pool": train_pool,
        "normalization_indices": [p2i[path] for path in normalization],
        "val_indices": [p2i[image_ids[row]] for row in val_rows],
        "cache_identity": {
            "val_isolation_protocol": PROTOCOL, "val_isolation_fingerprint": fingerprint,
            "val_version": contract["valVersion"],
            "val_manifest_sha256": contract["valManifestSha256"],
            "namespace_policy": "task-and-val-contract-isolated-no-legacy-reuse",
        },
    }


def freeze_contracts(service: Any, destination: Path) -> dict:
    """Only create a new private directory; never overwrite a prior contract."""
    info = fixed.active_vqa_validation_info(service)
    if info is None:
        raise RuntimeError("Val must already be frozen")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schemaVersion": 1, "protocol": PROTOCOL,
                "valVersion": info["version"], "valManifestSha256": info["manifestSha256"],
                "createdAt": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "tasks": {}, "modelsRetrained": False}
    for task_id in sorted(service.tasks):
        document = build_task_contract(service, task_id, info["version"])
        filename = f"{len(manifest['tasks']):03d}.json"
        sha = fixed._write(destination / filename, document)
        manifest["tasks"][task_id] = {"file": filename, "sha256": sha,
            "fingerprint": document["fingerprint"], "valCount": len(document["valRows"]),
            "fitCount": len(document["targets"]["joint"]["fitRows"]),
            "normalizationCount": len(document["normalizationRows"])}
        print(f"Isolated {task_id}: fit={manifest['tasks'][task_id]['fitCount']} Val={len(document['valRows'])}", flush=True)
    if fixed.active_vqa_validation_info(service) != info:
        raise RuntimeError("Val changed while preparing isolation")
    fixed._write(destination / "manifest.json", manifest)
    return manifest


def load_task_contract(service: Any, directory: Path, task_id: str) -> dict:
    manifest, _ = fixed._read(directory / "manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("schemaVersion") != 1:
        raise RuntimeError("Unsupported Val isolation manifest")
    info = fixed.active_vqa_validation_info(service)
    if info is None or any(manifest[key] != info[other] for key, other in (
        ("valVersion", "version"), ("valManifestSha256", "manifestSha256"))):
        raise RuntimeError("Isolation contract does not match active Val")
    if task_id not in manifest["tasks"]:
        raise RuntimeError(f"Task has no frozen Val isolation contract: {task_id}")
    entry = manifest["tasks"][task_id]
    if not re.fullmatch(r"[0-9]+\.json", entry["file"]):
        raise RuntimeError("Invalid isolation task path")
    document, _ = fixed._read(directory / entry["file"], entry["sha256"])
    expected = build_task_contract(service, task_id, manifest["valVersion"])
    if document != expected or entry["fingerprint"] != expected["fingerprint"]:
        raise RuntimeError("Val isolation source or partition changed")
    return document


def main() -> None:
    from tuning_server import TuningService
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    web = Path(__file__).resolve().parents[1]
    allowed = (web.parent / "runtime" / "evaluation" / "val-isolation").resolve()
    directory = args.directory.resolve()
    if directory == allowed or not directory.is_relative_to(allowed):
        raise RuntimeError("Use a versioned directory under runtime/evaluation/val-isolation")
    service = TuningService(web, start_worker=False)
    if args.validate_only:
        for task_id in sorted(service.tasks):
            contract = load_task_contract(service, directory, task_id)
            ids = list(service.bundle(task_id).image_ids)
            records = service._tuning_source_context(task_id).adapter.records.sort_values("embedding_index")
            if records["embedding_index"].astype(int).tolist() != list(range(len(records))):
                raise RuntimeError("Offline records are not in contiguous embedding order")
            inputs = materialize_training_inputs(contract, ids, records["relative_path"].astype(str).tolist())
            print(f"Verified {task_id}: {len(inputs['fit_paths'])} fit, {len(inputs['val_labels'])} Val", flush=True)
    else:
        freeze_contracts(service, directory)


if __name__ == "__main__":
    main()
