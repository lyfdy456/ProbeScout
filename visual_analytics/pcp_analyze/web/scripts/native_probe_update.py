"""Private, schedule-independent native Probe snapshots for system two.

No bank is synthesized here. A complete Val-isolated bank must be explicitly
linked to the current fusion base before native continuation can be available.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from val_isolation import METHODS, SEEDS, build_task_contract, digest

PROTOCOL = "native-probe-update-then-freeze-v1"
DEFAULT_CONFIG = {"epochs": 20, "lr": 0.0001, "batch_size": 64,
                  "batch_size_u": 256, "anchor_strength": 0.001,
                  "kfold_epochs": 20, "kfold_folds": 5}
_LOCK = threading.RLock()
_BUILD_LOCK = threading.RLock()
_BANK_CACHE: dict[str, tuple[Any, dict]] = {}


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Native Probe manifest must be an object")
    return value


def write_json(path: Path, value: Any) -> None:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(data + b"\n")


def config_contract(raw: Any = None) -> dict:
    if raw is None:
        return dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown native Probe update configuration")
    result = {**DEFAULT_CONFIG, **raw}
    for key in ("epochs", "batch_size", "batch_size_u", "kfold_epochs", "kfold_folds"):
        if type(result[key]) is not int or not 1 <= result[key] <= 10000:
            raise ValueError(f"Invalid native update {key}")
    for key in ("lr", "anchor_strength"):
        if not isinstance(result[key], (int, float)) or not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"Invalid native update {key}")
    if not 0 < result["lr"] <= 1 or result["epochs"] > 500 or not 2 <= result["kfold_folds"] <= 20:
        raise ValueError("Native update configuration is outside the allowed range")
    return result


def _bank_directory(service: Any, task_id: str, base: Any = None) -> Path:
    service.task(task_id)  # Only server-catalogue task IDs may select a bank.
    root = service.web_root.parent / "runtime" / "isolated-probes" / task_id
    initial_identity = getattr(base, "initial_baseline_identity", None)
    if initial_identity is None and base is None:
        from unified_initial_baseline import _task_metadata

        published = _task_metadata(service, task_id)
        if published is not None:
            initial_identity = {"bankDirectory": published[0]["bankDirectory"]}
    if initial_identity is not None:
        name = initial_identity.get("bankDirectory")
        if not isinstance(name, str) or not re.fullmatch(r"[a-f0-9]{64}", name):
            raise RuntimeError("Invalid published isolated bank identity")
        return root / name
    pointer = root / "active.json"
    if not pointer.is_file():
        raise RuntimeError("Base probe checkpoints unavailable: waiting for a shared isolated baseline and complete checkpoints")
    name = read_json(pointer).get("directory")
    if not isinstance(name, str) or not re.fullmatch(r"[a-f0-9]{64}", name):
        raise RuntimeError("Invalid active isolated Probe bank identity")
    directory = (root / name).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise RuntimeError("Native Probe bank escapes its private root")
    return directory


def coordinate_contract(service: Any, task_id: str, base: Any) -> tuple[list[str], tuple[str, ...], np.ndarray]:
    source = service._tuning_source_context(task_id)
    methods = getattr(base, "probe_method_ids", None)
    fusion = None if methods else service._weighted_fusion_context(task_id)
    records = source.adapter.records.sort_values("embedding_index")
    if records["embedding_index"].astype(int).tolist() != list(range(len(records))):
        raise RuntimeError("Native Probe embedding index is not contiguous")
    offline = records["relative_path"].astype(str).tolist()
    images = list(service.bundle(task_id).image_ids)
    methods = tuple(methods or fusion.learned_method_ids)
    if (len(set(offline)) != len(offline) or len(set(images)) != len(images)
            or set(offline) != set(images) or set(methods) != set(METHODS)
            or len(methods) != len(METHODS) or (fusion is not None and tuple(fusion.learned_method_labels) != base.learner_names)):
        raise RuntimeError("Native Probe image/method identity mismatch")
    lookup = {image: index for index, image in enumerate(offline)}
    return offline, methods, np.asarray([lookup[image] for image in images], dtype=np.int64)


def resolve_bank(service: Any, task_id: str, base: Any) -> dict:
    """Validate same-task checkpoints and raw probabilities against shared phi0."""
    directory = _bank_directory(service, task_id, base).resolve()
    from unified_initial_baseline import published_bank_attestation
    publication_link = published_bank_attestation(service, task_id, base)
    required = [directory / name for name in ("complete.json", "run_identity.json", "isolation_contract.json")]
    required.append(publication_link or directory / "refinement_base.json")
    metadata_paths = sorted(directory.rglob("metadata.json"))
    checkpoints = sorted(directory.rglob("model_seed_*.pt"))
    score_paths = sorted(directory.rglob("scores.npz"))
    files = [*required, *metadata_paths, *checkpoints, *score_paths]
    if any(not path.is_file() or (path != publication_link and not path.resolve().is_relative_to(directory)) for path in files):
        raise RuntimeError("Incomplete isolated Probe bank or shared-baseline link")
    stamps = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files)
    cache_key = str(directory) + ":" + base.base_state_fingerprint
    with _LOCK:
        cached = _BANK_CACHE.get(cache_key)
        if cached is not None and cached[0] == stamps:
            return cached[1]
    complete, identity, contract, link = [read_json(path) for path in required]
    if (complete != {**identity, "modelsRetrained": True, "published": False}
            or identity.get("taskId") != task_id or identity.get("methods") != list(METHODS)
            or identity.get("seeds") != list(SEEDS) or identity.get("protocol") != "val-isolated-probebank-v1"):
        raise RuntimeError("Isolated Probe bank completion contract mismatch")
    expected = build_task_contract(service, task_id, base.normalization_audit["vqaValidationVersion"])
    if contract != expected or identity.get("isolationFingerprint") != contract["fingerprint"]:
        raise RuntimeError("Isolated Probe bank Val/source contract mismatch")
    if (link.get("schemaVersion") != 1 or link.get("baseStateFingerprint") != base.base_state_fingerprint
            or link.get("normalizationContract") != base.normalization_audit):
        raise RuntimeError("Base probe checkpoints unavailable: isolated bank has not been linked to the shared current baseline")
    offline_ids, methods, to_offline = coordinate_contract(service, task_id, base)
    if (link.get("scoreImageIds") != offline_ids or link.get("probeMethods") != list(methods)
            or link.get("forwardVerified") is not True):
        raise RuntimeError("Native Probe published score/checkpoint identity is unverified")
    # This immutable publication attestation binds phi0 to its probabilities;
    # a changed checkpoint may not be accepted merely by recomputing its hash.
    pinned_files = [*metadata_paths, *checkpoints, *score_paths]
    pinned_hashes = {str(path.relative_to(directory)): file_sha(path) for path in pinned_files}
    if link.get("files") != pinned_hashes:
        raise RuntimeError("Native Probe published checkpoint/score checksum mismatch")
    attr_names = tuple(attr["name"] for attr in contract["attributes"])
    if attr_names != base.attribute_names or tuple(attr["id"] for attr in contract["attributes"]) != base.attribute_ids:
        raise RuntimeError("Native Probe attribute order mismatch")
    entries = {}
    for path in metadata_paths:
        meta = read_json(path)
        key = (meta.get("canonical_attribute"), meta.get("method"))
        if key in entries or key[0] not in attr_names or key[1] not in METHODS:
            raise RuntimeError("Unexpected or duplicate native Probe entry")
        if (meta.get("seeds") != list(SEEDS) or meta.get("fallback_seeds")
                or meta.get("val_isolation_fingerprint") != contract["fingerprint"]
                or meta.get("score_length") != contract["rowCount"]):
            raise RuntimeError("Native Probe entry is not a complete isolated five-seed bank")
        entries[key] = path.parent
    if len(entries) != len(attr_names) * len(METHODS):
        raise RuntimeError("Base probe checkpoints unavailable: not all attributes and eight learners are present")
    raw = np.empty((len(SEEDS), contract["rowCount"], len(attr_names), len(METHODS)), dtype=np.float32)
    for a, attr in enumerate(attr_names):
        for m, method in enumerate(methods):
            entry = entries[attr, method]
            if any(not (entry / f"model_seed_{seed}.pt").is_file() for seed in SEEDS):
                raise RuntimeError("Base probe checkpoints unavailable: incomplete seed checkpoints")
            with np.load(entry / "scores.npz", allow_pickle=False) as archive:
                values = np.asarray(archive["scores"], dtype=np.float32)
            if values.shape != (len(SEEDS), contract["rowCount"]) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise RuntimeError("Invalid native Probe probabilities")
            raw[:, :, a, m] = values[:, to_offline]
    original = getattr(base, "raw_probe_probabilities", None)
    mean = raw.astype(np.float64).mean(axis=0).astype(np.float32)
    if original is None or not np.array_equal(mean, np.asarray(original, dtype=np.float32)):
        raise RuntimeError("Base probe checkpoints unavailable: isolated bank probabilities differ from the shared baseline")
    hashes = {(str(path.relative_to(directory)) if path != publication_link else "publication-bank-attestation.json"):
              file_sha(path) for path in files}
    result = {"directory": str(directory), "fingerprint": digest(hashes), "files": hashes,
              "contract": contract, "entries": {f"{a}:{m}": str(entries[attr, method])
                  for a, attr in enumerate(attr_names) for m, method in enumerate(methods)},
              "rawProbabilities": raw, "methods": methods, "toOffline": to_offline,
              "offlineImageIds": offline_ids}
    with _LOCK:
        _BANK_CACHE.clear()
        _BANK_CACHE[cache_key] = (stamps, result)
    return result


def capability(service: Any, task_id: str) -> dict:
    try:
        _bank_directory(service, task_id)  # Missing prerequisites stay cheap.
        base = service._refinement_base_context(task_id)
        bank = resolve_bank(service, task_id, base)
        return {"available": True, "reason": None, "baseBankFingerprint": bank["fingerprint"]}
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return {"available": False, "reason": str(error)}


def current_code_fingerprint(service: Any) -> str:
    """Pin training code, but do not require it to load an already frozen snapshot."""
    source_root = service.web_root.parents[2] / "probe_learning"
    source_files = sorted((source_root / "src" / "methods").glob("*.py"))
    source_files += [Path(__file__).resolve(), source_root / "scripts" / "run_probebank_batch.py",
                     source_root / "scripts" / "run_retrieval_harness.py"]
    return digest([[str(path), file_sha(path)] for path in source_files])


def prepare_request(service: Any, user_id: str, session_id: str, task_id: str,
                    base: Any, annotations: list[dict], params: dict) -> dict:
    bank = resolve_bank(service, task_id, base)
    prepared = service._prepare_weight_refinement_supervision(
        task_id, base, annotations, feedback_weight=float(params["feedbackWeight"]),
        vqa_validation=True, vqa_validation_version=params["vqaValidationVersion"],
    )
    blocked = set(bank["contract"]["valRows"] + bank["contract"]["testRows"] + bank["contract"]["queryRows"])
    examples = []
    for a, attr in enumerate(base.attribute_ids):
        example = prepared["attributeExamples"][a]
        rows = [int(row) for row in example["rows"]]
        if set(rows) & blocked:
            raise RuntimeError("Val/Test/Query entered native Probe supervision")
        has_feedback = any(int(row["rowIndex"]) not in blocked and (
            int(row["label"]) > 0 or (int(row["label"]) < 0 and row.get("failureAttributionConfirmed")
                                      and attr in row.get("failedAttributeIds", []))) for row in annotations)
        examples.append({"attributeId": attr, "update": has_feedback, "rows": rows,
                         "labels": [int(value) for value in example["labels"]],
                         "weights": [float(value) for value in example["weights"]]})
    request = {"protocol": PROTOCOL, "userId": user_id, "sessionId": session_id, "taskId": task_id,
               "baseBankFingerprint": bank["fingerprint"], "baseStateFingerprint": base.base_state_fingerprint,
               "normalizationContract": base.normalization_audit, "isolationFingerprint": bank["contract"]["fingerprint"],
               "config": config_contract(params.get("nativeUpdateConfig")), "examples": examples,
               "codeFingerprint": current_code_fingerprint(service),
               "seeds": list(SEEDS), "methods": list(bank["methods"]),
               "attributeIds": list(base.attribute_ids),
               "sourceImageIdsSha256": digest(list(service.bundle(task_id).image_ids))}
    request["snapshotId"] = digest(request)
    return request


def updated_base(base: Any, raw: np.ndarray, snapshot_fingerprint: str) -> Any:
    mean = np.asarray(raw, dtype=np.float64).mean(axis=0).astype(np.float32)
    span = np.asarray(base.probe_max - base.probe_min, dtype=np.float64)
    z = np.divide(np.asarray(mean, dtype=np.float64) - base.probe_min, span, out=np.zeros_like(mean, dtype=np.float64), where=span > 0)
    z = np.clip(z, 0, 1).astype(np.float32)
    # New probabilities reuse the original fitting statistics; phi is detached.
    return dataclasses.replace(base, probe_features=z, raw_probe_probabilities=mean,
                               base_state_fingerprint=digest([base.base_state_fingerprint, snapshot_fingerprint]))


def load_snapshot(service: Any, request: dict, base: Any) -> tuple[Any, dict]:
    identity = dict(request)
    key = identity.pop("snapshotId", None)
    if key != digest(identity) or request.get("protocol") != PROTOCOL:
        raise RuntimeError("Native Probe snapshot identity mismatch")
    if (request.get("baseStateFingerprint") != base.base_state_fingerprint
            or request.get("normalizationContract") != base.normalization_audit
            or request.get("attributeIds") != list(base.attribute_ids)):
        raise RuntimeError("Native Probe snapshot base contract mismatch")
    directory = service.runtime_root / "native-probe-updates" / key
    manifest = read_json(directory / "complete.json")
    if manifest.get("request") != request or manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("Native Probe snapshot request changed")
    if not isinstance(manifest.get("files"), dict) or not {"probabilities.npy", "training_audit.json"} <= set(manifest["files"]):
        raise RuntimeError("Native Probe snapshot files are incomplete")
    for filename, expected in manifest["files"].items():
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha(path) != expected:
            raise RuntimeError("Native Probe snapshot checksum mismatch")
    raw = np.load(directory / "probabilities.npy", allow_pickle=False)
    expected_shape = (len(SEEDS), base.probe_features.shape[0], len(base.attribute_ids), len(METHODS))
    if raw.shape != expected_shape or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise RuntimeError("Native Probe snapshot probabilities are invalid")
    fingerprint = digest(manifest)
    return updated_base(base, raw, fingerprint), {
        "snapshotId": key, "snapshotFingerprint": fingerprint,
        "baseBankFingerprint": request["baseBankFingerprint"],
        "sharedAcrossFusionSchedules": True, "frozenDuringFusion": True,
        "updatedAttributes": [entry["attributeId"] for entry in request["examples"] if entry["update"]],
    }


def ensure_snapshot(service: Any, request: dict, base: Any) -> tuple[Any, dict]:
    """Publish only a fully verified native update; fusion schedule is absent."""
    with _BUILD_LOCK:
        directory = service.runtime_root / "native-probe-updates" / request["snapshotId"]
        if (directory / "complete.json").is_file():
            return load_snapshot(service, request, base)
        if directory.exists():
            raise RuntimeError("Incomplete native Probe snapshot; inspect it before retrying")
        bank = resolve_bank(service, request["taskId"], base)
        if bank["fingerprint"] != request["baseBankFingerprint"]:
            raise RuntimeError("Native Probe bank changed after queuing")
        temporary = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.pending")
        temporary.mkdir(parents=True, exist_ok=False)
        raw, audit = service._run_native_probe_update(request, bank, temporary)
        expected_shape = (len(SEEDS), base.probe_features.shape[0], len(base.attribute_ids), len(METHODS))
        if np.shape(raw) != expected_shape or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
            raise RuntimeError("Native Probe returned invalid probabilities; snapshot was not published")
        np.save(temporary / "probabilities.npy", np.asarray(raw, dtype=np.float32), allow_pickle=False)
        write_json(temporary / "training_audit.json", audit)
        manifest = {"protocol": PROTOCOL, "request": request,
                    "files": {str(path.relative_to(temporary)): file_sha(path)
                              for path in sorted(temporary.rglob("*")) if path.is_file()}}
        write_json(temporary / "complete.json", manifest)
        os.replace(temporary, directory)
        return load_snapshot(service, request, base)


def run_native_updates(service: Any, request: dict, bank: dict, output: Path) -> tuple[np.ndarray, dict]:
    """Continue each native learner, score phi', and never expose its graph."""
    root = service.web_root.parents[2] / "probe_learning"
    for path in (root, root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    import run_probebank_batch as bank_api
    import run_retrieval_harness as harness
    from src.methods.native_probe_update import NativeUpdateConfig, update_native_probe, score_native_probe

    task = service.task(request["taskId"])
    context = service._tuning_source_context(task.task_id)
    ids = list(service.bundle(task.task_id).image_ids)
    contract = bank["contract"]
    # These helpers mutate harness globals; source loading uses the same lock.
    with service._weighted_fusion_context_guard:
        harness.configure(task.dataset_id, task.task_name)
        adapter = context.adapter
        if bank_api.database_paths(adapter) != bank["offlineImageIds"]:
            raise RuntimeError("Native Probe feature/gallery image order mismatch")
        emb, _ = harness.load_backbone_embeddings(adapter, "siglip")
        patches, patch_meta = harness.load_backbone_patches(adapter, "siglip")
        text_queries = bank_api.cached_attribute_text_queries(
            [attr["name"] for attr in contract["attributes"]], "siglip", harness.BACKBONE_TEXT_MODEL["siglip"])
    raw = np.asarray(bank["rawProbabilities"], dtype=np.float32).copy()
    val_rows = contract["valRows"]
    all_labeled = {row for example in request["examples"] for row in example["rows"]}
    blocked = set(contract["valRows"] + contract["testRows"] + contract["queryRows"])
    u_rows = sorted(set(contract["trainPoolRows"]) - all_labeled - blocked)
    audits = []
    for a, example in enumerate(request["examples"]):
        if not example["update"]:
            audits.append({"attributeId": example["attributeId"], "updated": False, "reason": "no-confirmed-feedback"})
            continue
        attr = contract["attributes"][a]["name"]
        fit_rows = example["rows"]
        for m, method in enumerate(bank["methods"]):
            text = bank_api.text_query_for_method(method, attr, text_queries)
            entry = Path(bank["entries"][f"{a}:{m}"])
            metadata = read_json(entry / "metadata.json")
            if metadata.get("embedding_dim") != emb.shape[1] or metadata.get("feature_context") != bank_api.attention_feature_context(method, patch_meta, text):
                raise RuntimeError("Native Probe feature contract differs from its initial bank")
            features = patches if method in {"attention_pooling", "attribute_conditioned_attention"} else emb
            if features is None:
                raise RuntimeError("Native attention probes require original patch features")
            for s, seed in enumerate(SEEDS):
                checkpoint = entry / f"model_seed_{seed}.pt"
                relative = str(checkpoint.relative_to(Path(bank["directory"])))
                fit_offline = bank["toOffline"][fit_rows]
                val_offline = bank["toOffline"][val_rows]
                u_offline = bank["toOffline"][u_rows]
                model, audit = update_native_probe(
                    method, checkpoint, np.asarray(features[fit_offline]), np.asarray(example["labels"]),
                    sample_weights=np.asarray(example["weights"]),
                    validation_data=(np.asarray(features[val_offline]), np.asarray(contract["targets"][example["attributeId"]]["valLabels"])),
                    fit_ids=[ids[row] for row in fit_rows], validation_ids=[ids[row] for row in val_rows],
                    expected_checkpoint_sha256=bank["files"][relative],
                    X_unlabeled=np.asarray(features[u_offline]) if method in {"nnpu", "dcpu", "pu_ranking"} else None,
                    unlabeled_ids=[ids[row] for row in u_rows] if method in {"nnpu", "dcpu", "pu_ranking"} else (),
                    blocked_ids=[ids[row] for row in blocked], text_query=text,
                    config=NativeUpdateConfig(**request["config"]), seed=seed,
                )
                values = np.asarray(score_native_probe(method, model, features, text_query=text), dtype=np.float32)
                if values.shape != (len(ids),):
                    raise RuntimeError("Native Probe scoring returned the wrong Gallery shape")
                raw[s, :, a, m] = values[bank["toOffline"]]
                state_path = output / f"attribute-{a}-method-{m}-seed-{seed}.pt"
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "nativeAudit": audit}, state_path)
                audits.append({"attributeId": example["attributeId"], "method": method, "seed": seed, **audit})
                del model
    return raw, {"protocol": PROTOCOL, "frozenDuringFusion": True, "entries": audits}
