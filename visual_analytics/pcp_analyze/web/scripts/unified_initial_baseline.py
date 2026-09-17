"""Build and atomically publish immutable, Val-isolated shared F0 snapshots.

Verify completed checkpoints, construct reference gates from fit labels, and
optionally select the paper's final gates and F1 cutoff on fixed Val.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import threading
import uuid

import numpy as np

from val_isolation import METHODS, SEEDS, build_task_contract, digest

PROTOCOL = "isolated-initial-baseline-v1"
CALIBRATION = "fit-only-mean-prob-grid-init-fixed-T-joint-BCE-v1"
VAL_CALIBRATION = "full-only-per-task-vqa-val-ap-gates-val-f1-cutoff-v1"
# Checkpoints are verified on CPU against float32 caches produced using the
# training device and original gallery batches. Attention reductions can differ
# by a few float32 ulps across these paths. Keep a fixed absolute error ceiling:
# relative tolerance must not silently permit a larger discrepancy near one.
FORWARD_ABSOLUTE_TOLERANCE = 1e-5
FORWARD_RELATIVE_TOLERANCE = 0.0
METHOD_LABELS = {"mlp_baseline": "MLP", "kfold_pu": "K-Fold", "triplet_loss": "Triplet Loss",
    "attention_pooling": "Attention Pooling", "attribute_conditioned_attention": "Attribute-conditioned Attention",
    "nnpu": "nnPU", "dcpu": "DC-PU", "pu_ranking": "Ours-PURA"}
_CACHE: dict[str, tuple[tuple, tuple]] = {}
_GUARD = threading.RLock()


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Initial baseline manifest is malformed")
    return value


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def root_path(service) -> Path:
    return service.web_root.parent / "runtime" / "unified-initial"


def version_directory(service, version: str) -> Path:
    if not isinstance(version, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}", version):
        raise RuntimeError("Invalid initial baseline publication version")
    return root_path(service) / version


def active_info(service) -> dict | None:
    pointer = root_path(service) / "active.json"
    if not pointer.is_file():
        return None
    active = read(pointer)
    directory = version_directory(service, active.get("version"))
    manifest = directory / "manifest.json"
    if sha(manifest) != active.get("manifestSha256"):
        raise RuntimeError("Initial baseline publication checksum mismatch")
    result = read(manifest)
    if result.get("protocol") != PROTOCOL or result.get("version") != active["version"] or result.get("complete") is not True:
        raise RuntimeError("Initial baseline publication is incomplete")
    return result


def _task_metadata(service, task_id: str, version: str | None = None, *, fingerprint: str | None = None):
    service.task(task_id)
    import local_tasks
    if local_tasks.available(service, task_id):
        expected = local_tasks.settings(service, task_id)["version"]
        if version is not None and version != expected:
            raise RuntimeError("Local task baseline version differs from its immutable input")
        version = expected
    from experimental_parent_overlay import applies, metadata as overlay_metadata
    if applies(task_id):
        return overlay_metadata(service, task_id, version, fingerprint=fingerprint)
    if version is None:
        active = active_info(service)
        if active is None:
            return None
        version = active["version"]
        if fingerprint is not None and active.get("tasks", {}).get(task_id, {}).get("baseStateFingerprint") != fingerprint:
            # Pinned requests keep reading their original publication after a
            # later global switch. Never fall back to another score snapshot.
            candidates = []
            for path in root_path(service).glob("*/manifest.json"):
                publication = read(path)
                if publication.get("complete") is True and publication.get("tasks", {}).get(task_id, {}).get("baseStateFingerprint") == fingerprint:
                    candidates.append(publication["version"])
            if not candidates:
                raise RuntimeError("Pinned initial baseline fingerprint is unavailable")
            version = sorted(candidates)[0]
    directory = version_directory(service, version)
    publication_path = directory / "manifest.json"
    if not publication_path.is_file():
        # Builders/reports may inspect a verified task before all 36 complete.
        path = directory / task_id / "manifest.json"
        if not path.is_file():
            return None
    else:
        publication = read(publication_path)
        if publication.get("protocol") != PROTOCOL or publication.get("complete") is not True:
            raise RuntimeError("Initial baseline publication is incomplete")
        item = publication.get("tasks", {}).get(task_id)
        if not isinstance(item, dict):
            raise RuntimeError("Task is absent from this baseline publication")
        path = directory / task_id / "manifest.json"
        if sha(path) != item.get("manifestSha256"):
            raise RuntimeError("Initial baseline task checksum mismatch")
    metadata = read(path)
    if (metadata.get("protocol") != PROTOCOL or metadata.get("version") != version
            or metadata.get("taskId") != task_id or metadata.get("verified") is not True):
        raise RuntimeError("Initial baseline task is not verified")
    if fingerprint is not None and metadata.get("baseStateFingerprint") != fingerprint:
        raise RuntimeError("Initial baseline fingerprint mismatch")
    return metadata, path.parent


def load_task_baseline(service, task_id: str, version: str | None = None, *, fingerprint: str | None = None):
    """Return (RefinementBaseContext, metadata, directory), or None if pending."""
    info = _task_metadata(service, task_id, version, fingerprint=fingerprint)
    if info is None:
        return None
    metadata, directory = info
    files = metadata.get("files")
    required = {"base.npz", "scores.f32", "ranks.f32", "visualization.f32", "calibration.json"}
    if not isinstance(files, dict) or not required <= set(files):
        raise RuntimeError("Initial baseline export files are incomplete")
    paths = [(directory / name).resolve() for name in files]
    if any(not path.is_relative_to(directory.resolve()) for path in paths):
        raise RuntimeError("Initial baseline file escapes publication directory")
    stamps = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
    key = str(directory) + sha(directory / "manifest.json")
    with _GUARD:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamps:
            return cached[1]
    for name, expected in files.items():
        if sha(directory / name) != expected:
            raise RuntimeError("Initial baseline artifact checksum mismatch")
    bundle = service.bundle(task_id)
    if metadata["rowCount"] != len(bundle.image_ids) or metadata["imageIdsSha256"] != digest(list(bundle.image_ids)):
        raise RuntimeError("Initial baseline Gallery image order mismatch")
    from tuning_server import RefinementBaseContext
    with np.load(directory / "base.npz", allow_pickle=False) as values:
        data = {name: values[name].copy() for name in values.files}
    identity = {"protocol": PROTOCOL, "publicationVersion": metadata["version"],
                "bankDirectory": metadata["bankDirectory"], "bankFingerprint": metadata["bankFingerprint"],
                "calibration": metadata.get("calibration", CALIBRATION),
                "initialModelHoldoutIndependent": metadata.get("initialModelHoldoutIndependent", True)}
    if metadata.get("calibration") == VAL_CALIBRATION:
        if metadata.get("gateArrayEncoding") != "<f8" or metadata.get("calibrationUsedValidation") is not True:
            raise RuntimeError("Val-calibrated baseline precision/provenance is invalid")
        identity.update(calibrationUsedValidation=True, gateArrayEncoding="<f8",
                        classificationThreshold=metadata["classificationThreshold"])
    if metadata.get("frozenParent"):
        identity["frozenParent"] = metadata["frozenParent"]
    base = RefinementBaseContext(
        attribute_ids=tuple(metadata["attributeIds"]), attribute_names=tuple(metadata["attributeNames"]),
        learner_names=tuple(metadata["learnerMethods"]), embedding_names=tuple(metadata["embeddingMethods"]),
        probe_features=data["probeFeatures"], raw_probe_probabilities=data["rawProbeProbabilities"],
        probe_min=data["probeMin"], probe_max=data["probeMax"], initial_theta=data["theta"], temperature=data["temperature"],
        embedding_features=data["embeddingFeatures"], embedding_min=data["embeddingMin"], embedding_max=data["embeddingMax"],
        development_indices=data["developmentIndices"], base_state_fingerprint=metadata["baseStateFingerprint"],
        normalization_audit=metadata["normalizationContract"], initial_baseline_identity=identity,
        probe_method_ids=tuple(metadata["probeMethodIds"]),
    )
    n, a = metadata["rowCount"], len(base.attribute_ids)
    if (base.probe_features.shape != (n, a, 8) or base.raw_probe_probabilities.shape != (n, a, 8)
            or base.embedding_features.shape != (n, 2) or base.initial_theta.shape != (a,)
            or np.any(base.temperature <= 0) or not all(np.isfinite(array).all() for array in data.values())):
        raise RuntimeError("Initial baseline array contract is invalid")
    result = base, metadata, directory
    with _GUARD:
        _CACHE.clear()
        _CACHE[key] = (stamps, result)
    return result


def gate_array_encoding(base) -> str:
    """Legacy requests retain f32 hashes; Val-selected gates retain f64."""
    identity = getattr(base, "initial_baseline_identity", None) or {}
    encoding = identity.get("gateArrayEncoding", "<f4")
    if encoding not in ("<f4", "<f8"):
        raise RuntimeError("Unknown frozen gate array encoding")
    return encoding


def published_bank_attestation(service, task_id: str, base) -> Path | None:
    """A derived F0 links the same immutable checkpoints without editing them."""
    identity = getattr(base, "initial_baseline_identity", None)
    if not identity or identity.get("calibration") != VAL_CALIBRATION:
        return None
    info = _task_metadata(service, task_id, identity["publicationVersion"],
                          fingerprint=base.base_state_fingerprint)
    if info is None:
        raise RuntimeError("Derived baseline publication is unavailable")
    metadata, directory = info
    proof = metadata.get("bankAttestation", {})
    if proof.get("path") != "bank_attestation.json":
        raise RuntimeError("Derived baseline checkpoint attestation is missing")
    path = directory / proof["path"]
    if not path.is_file() or sha(path) != proof.get("sha256"):
        raise RuntimeError("Derived baseline checkpoint attestation checksum mismatch")
    return path


def minmax(values, rows):
    raw = np.asarray(values, dtype=np.float32)
    low, high = raw[rows].min(0), raw[rows].max(0)
    span = (high - low).astype(np.float64)
    z = np.divide(raw.astype(np.float64) - low, span, out=np.zeros_like(raw, dtype=np.float64), where=span > 0)
    return np.clip(z, 0, 1).astype(np.float32), low, high


def calibrate(service, z, contract):
    """Reuse the original grid + constrained Joint-BCE code, on fit rows only."""
    source = service.web_root.parents[2] / "probe_learning"
    for path in (source, source / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from src.methods import gate_calibration as grid
    from src.methods import gate_calibration as joint
    q = np.asarray(z, dtype=np.float64).mean(axis=2)
    attrs = [attr["id"] for attr in contract["attributes"]]
    fitted = {}
    for a, attr in enumerate(attrs):
        target = contract["targets"][attr]
        fitted[attr] = grid.search_attribute_gate(q[target["fitRows"], a], np.asarray(target["fitLabels"]),
            coarse_step=.025, fine_radius=.05, fine_step=.005, temperatures=grid.TEMPERATURES)
    target = contract["targets"]["joint"]
    result = joint.fit_joint_bce_from_grid(q[target["fitRows"]], np.asarray(target["fitLabels"]), attrs,
        {attr: fitted[attr]["theta"] for attr in attrs}, {attr: fitted[attr]["temperature"] for attr in attrs},
        train_temperature=False, theta_radius=.05, temperature_bounds=(.03, .30))
    return (np.asarray([result["theta_by_attr"][attr] for attr in attrs], dtype=np.float32),
            np.asarray([result["temperature_by_attr"][attr] for attr in attrs], dtype=np.float32),
            {"protocol": CALIBRATION, "attributeGrid": fitted, "jointOptimization": result,
              "labelSource": "original-vqa-fit-only", "usesValLabels": False, "usesTestLabels": False})


def select_paper_gates(service, z, embeddings, theta, temperature, contract, reference):
    """Appendix C.4: select one shared gate shift/scale using fixed-Val AP."""
    source = service.web_root.parents[2] / "probe_learning"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from src.evaluation.components import select_on_validation
    config = read(service.web_root.parents[2] / "configs/component_ablation.json")
    rows = np.asarray(contract["valRows"], dtype=np.int64)
    if not len(rows) or any(set(rows) & set(contract[key])
                           for key in ("normalizationRows", "testRows", "queryRows")):
        raise RuntimeError("Gate selection requires an isolated, nonempty Val")
    if any(set(rows) & set(target["fitRows"]) for target in contract["targets"].values()):
        raise RuntimeError("Gate selection Val overlaps probe fitting")
    labels = np.asarray(contract["targets"]["joint"]["valLabels"], dtype=np.uint8)
    q = np.asarray(z[rows], dtype=np.float64).mean(axis=2)
    h = np.asarray(embeddings[rows], dtype=np.float64).mean(axis=1)
    best, candidates = select_on_validation(q, h, theta, temperature, labels,
        {"softgate": True, "embedding": True}, config)
    selected_theta = np.clip(np.asarray(theta, dtype=np.float64) + best["theta_shift"], 0., 1.)
    selected_temperature = np.minimum(np.asarray(temperature, dtype=np.float64)
                                     * best["temperature_scale"], config["temperature_cap"])
    return selected_theta, selected_temperature, {
        "protocol": VAL_CALIBRATION, "usesValLabels": True, "usesTestLabels": False,
        "labelSource": contract["labelSource"], "isolationFingerprint": contract["fingerprint"],
        "referenceCalibration": reference, "referenceTheta": theta.tolist(),
        "referenceTemperature": temperature.tolist(), "selectionConfig": config,
        "gateSelection": "fixed-Val Joint AP; beta=1/8,gamma=1,eta=1/2,lambda=1/4",
        "cutoffSelection": "fixed-Val Joint maximum F1; complete tied-score groups",
        "thetaShift": best["theta_shift"], "TScale": best["temperature_scale"],
        "theta": selected_theta.tolist(), "temperature": selected_temperature.tolist(),
        "classificationThreshold": best["threshold"], "valAp": best["val_ap"],
        "valF1": best["val_f1"], "valN": len(rows), "valP": int(labels.sum()),
        "candidates": candidates, "gateArrayEncoding": "<f8"}


def _forward_probability_error(actual, expected, *, context: str) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if (actual.shape != expected.shape or actual.ndim != 1 or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or np.any((actual < 0) | (actual > 1)) or np.any((expected < 0) | (expected > 1))):
        raise RuntimeError(f"Checkpoint forward/cache probabilities are invalid: {context}")
    return float(np.max(np.abs(actual - expected)))


def _verify_forward_probabilities(actual, expected, *, context: str) -> float:
    delta = _forward_probability_error(actual, expected, context=context)
    if not np.allclose(actual, expected, rtol=FORWARD_RELATIVE_TOLERANCE,
                       atol=FORWARD_ABSOLUTE_TOLERANCE):
        raise RuntimeError(f"Checkpoint forward/cache mismatch: {context} ({delta})")
    return delta


def _verify_checkpoint_forward(actual, cached, sample_offline, *, context: str, matched_batch_score=None) -> dict:
    """Keep the CPU fast path; recheck only discrepant original attention blocks.

    The optional scorer must use the original CUDA attention scoring path and
    1024-row gallery batch boundaries. Entire rechecked blocks must pass the same
    absolute ceiling; a CPU discrepancy never raises the allowed tolerance.
    """
    actual = np.asarray(actual, dtype=np.float64)
    cached = np.asarray(cached, dtype=np.float64)
    sample_offline = np.asarray(sample_offline)
    if (cached.ndim != 1 or sample_offline.ndim != 1
            or sample_offline.dtype.kind not in "iu" or not sample_offline.size
            or np.any((sample_offline < 0) | (sample_offline >= len(cached)))
            or len(np.unique(sample_offline)) != len(sample_offline)):
        raise RuntimeError(f"Checkpoint forward/cache row alignment is invalid: {context}")
    expected = cached[sample_offline]
    cpu_delta = _forward_probability_error(actual, expected, context=context)
    errors = np.abs(actual - expected)
    failed = errors > FORWARD_ABSOLUTE_TOLERANCE
    audit = {"cpuMaxAbsoluteError": cpu_delta, "maxAbsoluteError": cpu_delta,
             "matchedBackendFallbackUsed": False, "matchedBackendBlocks": []}
    if not failed.any():
        return audit
    if matched_batch_score is None:
        _verify_forward_probabilities(actual, expected, context=context)
    audit.update(matchedBackendFallbackUsed=True, matchedBackendBatchSize=1024,
                 cpuExceededSampleOfflineRows=sample_offline[failed].astype(int).tolist())
    for start in sorted(set((sample_offline[failed] // 1024) * 1024)):
        start = int(start)
        stop = min(start + 1024, len(cached))
        block_actual = np.asarray(matched_batch_score(start, stop), dtype=np.float64)
        delta = _verify_forward_probabilities(block_actual, cached[start:stop],
            context=f"{context}/matched-CUDA-block-{start}-{stop}")
        included = (sample_offline >= start) & (sample_offline < stop)
        errors[included] = np.abs(block_actual[sample_offline[included] - start] - expected[included])
        audit["matchedBackendBlocks"].append({"startRow": start, "endRowExclusive": stop,
            "sampleCount": int(included.sum()), "maxAbsoluteError": delta})
    audit["matchedBackendMaxAbsoluteError"] = max(block["maxAbsoluteError"] for block in audit["matchedBackendBlocks"])
    audit["maxAbsoluteError"] = max(float(errors.max()), audit["matchedBackendMaxAbsoluteError"])
    return audit


def verify_bank(service, task_id: str, directory: Path):
    """Check all score bytes and representative rows of every real checkpoint."""
    from native_probe_update import file_sha
    directory = directory.resolve()
    allowed = (service.web_root.parent / "runtime" / "isolated-probes" / task_id).resolve()
    if directory.parent != allowed or not re.fullmatch(r"[a-f0-9]{64}", directory.name):
        raise RuntimeError("Initial baseline requires a server-scoped isolated bank")
    identity, complete, contract = [read(directory / name) for name in ("run_identity.json", "complete.json", "isolation_contract.json")]
    if (directory.name != digest(identity) or complete != {**identity, "modelsRetrained": True, "published": False}
            or identity.get("taskId") != task_id or identity.get("methods") != list(METHODS)
            or identity.get("seeds") != list(SEEDS) or identity.get("isolationFingerprint") != contract.get("fingerprint")):
        raise RuntimeError("Isolated Probe bank is incomplete")
    if contract != build_task_contract(service, task_id, contract["valVersion"]):
        raise RuntimeError("Isolated bank Val/source identity changed")
    source_root = service.web_root.parents[2] / "probe_learning"
    for path in (source_root, source_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    import run_probebank_batch as bank
    import run_retrieval_harness as harness
    from src.methods.native_probe_update import _load_base, score_native_probe
    adapter = service._tuning_source_context(task_id).adapter
    with service._weighted_fusion_context_guard:
        harness.configure(contract["dataset"], contract["taskName"], adapter=adapter)
        paths = bank.database_paths(adapter)
        emb, _ = harness.load_backbone_embeddings(adapter, "siglip")
        patches, patch_meta = harness.load_backbone_patches(adapter, "siglip")
        text_queries = bank.cached_attribute_text_queries([attr["name"] for attr in contract["attributes"]], "siglip", harness.BACKBONE_TEXT_MODEL["siglip"])
    web_ids = list(service.bundle(task_id).image_ids)
    if len(set(paths)) != len(paths) or set(paths) != set(web_ids):
        raise RuntimeError("Initial baseline image identities differ")
    lookup = {image: row for row, image in enumerate(paths)}
    to_offline = np.asarray([lookup[image] for image in web_ids], dtype=np.int64)
    generator = np.random.default_rng(0)
    sampled_web = set()
    # Sampling is provenance QA only; it never reads image labels or updates a
    # model. Every held-out/training role is represented where it is nonempty.
    for pool in (contract["targets"]["joint"]["fitRows"], contract["valRows"], contract["testRows"], contract["queryRows"]):
        sampled_web.update(map(int, generator.choice(pool, min(8, len(pool)), replace=False)))
    remainder = sorted(set(range(len(web_ids))) - sampled_web)
    sampled_web.update(map(int, generator.choice(remainder, min(max(0, 128-len(sampled_web)), len(remainder)), replace=False)))
    sample_rows = np.asarray(sorted(sampled_web), dtype=np.int64)
    sample_offline = to_offline[sample_rows]
    entries = {}
    for path in sorted(directory.rglob("metadata.json")):
        metadata = read(path)
        key = metadata.get("canonical_attribute"), metadata.get("method")
        if key in entries:
            raise RuntimeError("Duplicate isolated probe entry")
        entries[key] = path.parent, metadata
    raw = np.empty((5, len(web_ids), len(contract["attributes"]), 8), dtype=np.float32)
    errors = []
    hashes = {}
    for a, attribute in enumerate(contract["attributes"]):
        attr = attribute["name"]
        for m, method in enumerate(METHODS):
            if (attr, method) not in entries:
                raise RuntimeError("Missing isolated learner checkpoint entry")
            entry, metadata = entries[attr, method]
            text = bank.text_query_for_method(method, attr, text_queries)
            if (metadata.get("seeds") != list(SEEDS) or metadata.get("fallback_seeds")
                    or metadata.get("val_isolation_fingerprint") != contract["fingerprint"]
                    or metadata.get("score_length") != len(paths) or metadata.get("embedding_dim") != emb.shape[1]
                    or metadata.get("feature_context") != bank.attention_feature_context(method, patch_meta, text)):
                raise RuntimeError("Isolated checkpoint feature/provenance mismatch")
            features = patches if method in {"attention_pooling", "attribute_conditioned_attention"} else emb
            if features is None:
                raise RuntimeError("Original patch features are missing")
            with np.load(entry / "scores.npz", allow_pickle=False) as scores:
                cached = np.asarray(scores["scores"], dtype=np.float32)
            if cached.shape != (5, len(paths)) or not np.isfinite(cached).all() or np.any((cached < 0) | (cached > 1)):
                raise RuntimeError("Isolated raw probability cache is invalid")
            for path in (entry / "metadata.json", entry / "scores.npz"):
                hashes[str(path.relative_to(directory))] = file_sha(path)
            for s, seed in enumerate(SEEDS):
                path = entry / f"model_seed_{seed}.pt"
                checksum = file_sha(path)
                model, _ = _load_base(method, path, checksum, int(features.shape[-1]), text)
                # Use CPU first; only an attention discrepancy requires bounded
                # original-batch CUDA inference (never training or cache repair).
                model = model.to("cpu")
                actual = score_native_probe(method, model, np.asarray(features[sample_offline]), text_query=text)

                def matched_attention_batch(start, stop):
                    if not torch.cuda.is_available():
                        raise RuntimeError("Checkpoint CPU discrepancy requires CUDA original-batch verification")
                    model.to("cuda")
                    model.eval()
                    return harness.score_attn_many([model], np.asarray(features[start:stop]), text, chunk=1024)[0]

                audit = _verify_checkpoint_forward(actual, cached[s], sample_offline,
                    context=f"{attr}/{method}/{seed}", matched_batch_score=matched_attention_batch
                    if method in {"attention_pooling", "attribute_conditioned_attention"} else None)
                if audit["matchedBackendFallbackUsed"]:
                    audit.update(matchedBackendDevice=str(next(model.parameters()).device),
                        matchedBackendDeviceName=torch.cuda.get_device_name(next(model.parameters()).device),
                        matchedBackendScorer="run_retrieval_harness.score_attn_many")
                if file_sha(path) != checksum:
                    raise RuntimeError("Checkpoint changed during forward verification")
                hashes[str(path.relative_to(directory))] = checksum
                errors.append({"attributeId": attribute["id"], "method": method, "seed": seed, **audit})
                del matched_attention_batch
                del model
            raw[:, :, a, m] = cached[:, to_offline]
    if len(entries) != len(contract["attributes"]) * 8:
        raise RuntimeError("Unexpected isolated bank entries")
    return raw, contract, hashes, {"forwardVerified": True, "entries": errors, "scoreImageIds": paths,
        "forwardScope": "full-gallery" if len(sample_rows) == len(web_ids) else "representative-rows-all-checkpoints",
        "samplingProtocol": "role-stratified-eight-plus-random-seed0-up-to128-v1",
        "sampleImageIds": [web_ids[row] for row in sample_rows], "sampleCount": len(sample_rows),
        "absoluteTolerance": FORWARD_ABSOLUTE_TOLERANCE, "relativeTolerance": FORWARD_RELATIVE_TOLERANCE,
        "forwardDevice": "cpu-with-matched-CUDA-fallback" if any(entry["matchedBackendFallbackUsed"] for entry in errors) else "cpu",
        "forwardDtype": "float32", "torchVersion": str(torch.__version__),
        "comparisonProtocol": "cpu-then-original-attention-CUDA-batch-float32-absolute-v2",
        "matchedBackendFallbackCount": sum(entry["matchedBackendFallbackUsed"] for entry in errors),
        "cpuMaxAbsoluteError": max(entry["cpuMaxAbsoluteError"] for entry in errors),
        "maxAbsoluteError": max(entry["maxAbsoluteError"] for entry in errors), "bankTrainingIdentity": identity,
        "fullScoreVerification": "sha256-shape-finite-unit-range-image-method-identity"}


def build_task_baseline(service, task_id: str, version: str, bank_directory: Path, *, select_on_val=False):
    """Build one verified task without making it active in the website."""
    from tuning_server import WEIGHTED_FUSION_LEARNERS, REFINEMENT_EMBEDDING_METHODS, normalized_ranks
    from tuning_models import unified_weight_scores
    if tuple(METHOD_LABELS[method] for method in METHODS) != tuple(WEIGHTED_FUSION_LEARNERS):
        raise RuntimeError("Initial baseline learner ID/display-label order changed")
    task = service.task(task_id)
    parent = version_directory(service, version)
    destination = parent / task_id
    if destination.exists():
        existing = load_task_baseline(service, task_id, version)
        if (existing is None or existing[1]["bankDirectory"] != bank_directory.name
                or existing[1].get("calibration", CALIBRATION) != (VAL_CALIBRATION if select_on_val else CALIBRATION)):
            raise RuntimeError("Refusing to overwrite an existing baseline task")
        return existing
    raw, contract, bank_hashes, verification = verify_bank(service, task_id, bank_directory)
    rows, normalization = service._refinement_normalization_scope(task_id, contract["valVersion"])
    if list(map(int, rows)) != contract["normalizationRows"]:
        raise RuntimeError("Baseline normalization rows differ from isolated training contract")
    mean = raw.astype(np.float64).mean(0).astype(np.float32)
    z, low, high = minmax(mean, rows)
    theta, temperature, calibration = calibrate(service, z, contract)
    embedding_raw = np.column_stack([service._exported_method_column(task, "rawScores", method, task.target_ids.index("joint"))
                                     for method in REFINEMENT_EMBEDDING_METHODS]).astype(np.float32)
    embeddings, embedding_min, embedding_max = minmax(embedding_raw, rows)
    if select_on_val:
        theta, temperature, calibration = select_paper_gates(
            service, z, embeddings, theta, temperature, contract, calibration)
    n, a, _ = z.shape
    output = unified_weight_scores(z, embeddings, np.full((a, 8), 1/8), np.ones(a), np.full(2, .5), .25, theta, temperature)
    columns = [output.final_scores, output.conjunction_scores]
    for index in range(a):
        columns += [output.gates[:, index], *[z[:, index, m] for m in range(8)]]
    columns += [output.holistic_scores, embeddings[:, 0], embeddings[:, 1]]
    scores = np.asarray(np.column_stack(columns), dtype=np.float32)
    ranks = np.column_stack([normalized_ranks(scores[:, col]) for col in range(scores.shape[1])]).astype(np.float32)
    temporary = destination.with_name(f".{task_id}.{uuid.uuid4().hex}.pending")
    temporary.mkdir(parents=True, exist_ok=False)
    np.savez(temporary / "base.npz", rawProbeProbabilities=mean, probeFeatures=z, probeMin=low, probeMax=high,
             theta=theta, temperature=temperature, embeddingFeatures=embeddings, embeddingMin=embedding_min,
             embeddingMax=embedding_max, developmentIndices=np.asarray(rows, dtype=np.int64))
    (temporary / "scores.f32").write_bytes(scores[:, 0].astype("<f4").tobytes())
    (temporary / "ranks.f32").write_bytes(ranks[:, 0].astype("<f4").tobytes())
    (temporary / "visualization.f32").write_bytes(np.concatenate([scores.reshape(-1), ranks.reshape(-1)]).astype("<f4").tobytes())
    write(temporary / "calibration.json", calibration)
    write(temporary / "verification.json", verification)
    files = {path.name: sha(path) for path in temporary.iterdir() if path.is_file()}
    base_fingerprint = digest({"protocol": PROTOCOL, "normalization": normalization, "files": files,
                               "bankHashes": bank_hashes, "calibration": calibration["protocol"]})
    metadata = {"schemaVersion": 1, "protocol": PROTOCOL, "version": version, "taskId": task_id,
                "verified": True, "baseStateFingerprint": base_fingerprint, "rowCount": n,
                "componentCount": scores.shape[1], "imageIdsSha256": digest(list(service.bundle(task_id).image_ids)),
                "attributeIds": [attr["id"] for attr in contract["attributes"]],
                "attributeNames": [attr["name"] for attr in contract["attributes"]],
                "learnerMethods": list(WEIGHTED_FUSION_LEARNERS), "probeMethodIds": list(METHODS),
                "embeddingMethods": list(REFINEMENT_EMBEDDING_METHODS), "normalizationContract": normalization,
                "bankDirectory": bank_directory.name, "bankFingerprint": digest(bank_hashes), "files": files,
                "bankTrainingIdentity": verification["bankTrainingIdentity"],
                "initialModelHoldoutIndependent": not select_on_val, "calibration": calibration["protocol"],
                "formula": "F=C*((1-lambda)+lambda*H); beta=1/8,gamma=1,eta=1/2,lambda=1/4"}
    # Prepared links do not enable native tuning until the publication pointer
    # switches. Existing bank assets and scores themselves remain unchanged.
    link = {"schemaVersion": 1, "baseStateFingerprint": base_fingerprint, "normalizationContract": normalization,
            "scoreImageIds": verification["scoreImageIds"], "probeMethods": list(METHODS),
            "forwardVerified": True, "files": bank_hashes,
            "forwardScope": verification["forwardScope"], "samplingProtocol": verification["samplingProtocol"],
            "sampleImageIds": verification["sampleImageIds"]}
    link_path = bank_directory / "refinement_base.json"
    if link_path.exists() and read(link_path) != link:
        raise RuntimeError("Refusing to overwrite another bank/baseline link")
    if not link_path.exists():
        write(link_path, link)
    if select_on_val:
        write(temporary / "bank_attestation.json", link)
        metadata.update(calibrationUsedValidation=True, probeFitExcludesValidation=True,
                        gateArrayEncoding="<f8", classificationThreshold=calibration["classificationThreshold"],
                        bankAttestation={"path": "bank_attestation.json", "sha256": sha(temporary / "bank_attestation.json")})
    write(temporary / "manifest.json", metadata)
    os.replace(temporary, destination)
    return load_task_baseline(service, task_id, version)


def publish(service, version: str, *, activate: bool = True):
    """One pointer update after every task passes; no partial-task publication."""
    directory = version_directory(service, version)
    tasks = {}
    val_versions = set()
    training_contracts = set()
    for task_id in sorted(service.tasks):
        loaded = load_task_baseline(service, task_id, version)
        if loaded is None:
            raise RuntimeError(f"Initial baseline task is still pending: {task_id}")
        _, metadata, path = loaded
        tasks[task_id] = {"manifestSha256": sha(path / "manifest.json"), "baseStateFingerprint": metadata["baseStateFingerprint"]}
        val_versions.add(metadata["normalizationContract"]["vqaValidationVersion"])
        training_contracts.add(digest({key: metadata["bankTrainingIdentity"][key]
            for key in ("trainerSha256", "epochs", "methods", "seeds")}))
    if len(val_versions) != 1:
        raise RuntimeError("Cannot publish mixed Val versions")
    if len(training_contracts) != 1:
        raise RuntimeError("Cannot publish mixed trainer/epoch/seed/method contracts")
    manifest = {"schemaVersion": 1, "protocol": PROTOCOL, "version": version,
                "complete": True, "tasks": tasks, "vqaValidationVersion": next(iter(val_versions)),
                "trainingContractFingerprint": next(iter(training_contracts))}
    path = directory / "manifest.json"
    if path.exists() and read(path) != manifest:
        raise RuntimeError("Published baseline manifest cannot be overwritten")
    if not path.exists():
        write(path, manifest)
    if activate:
        pointer = root_path(service) / "active.json"
        temporary = pointer.with_name(f".active.{uuid.uuid4().hex}.pending")
        write(temporary, {"version": version, "manifestSha256": sha(path)})
        os.replace(temporary, pointer)
    return manifest


def descriptor(service, task_id: str):
    from tuning_server import REFINEMENT_VISUALIZATION_ALGORITHM, REFINEMENT_CLUSTER_ALGORITHM
    loaded = load_task_baseline(service, task_id)
    if loaded is None:
        return {"available": False, "taskId": task_id, "reason": "The isolated initial baseline is pending publication"}
    base, metadata, _ = loaded
    fingerprint = base.base_state_fingerprint
    source = digest([PROTOCOL, fingerprint, metadata["files"]["visualization.f32"]])
    return {"available": True, "version": PROTOCOL, "publicationVersion": metadata["version"],
        "taskId": task_id, "fingerprint": fingerprint, "baseStateFingerprint": fingerprint,
        "sourceFingerprint": source, "visualizationFingerprint": source,
        "normalizationFingerprint": digest(base.normalization_audit),
        "visualizationVersion": REFINEMENT_VISUALIZATION_ALGORITHM, "clusterAlgorithm": REFINEMENT_CLUSTER_ALGORITHM,
        "rowCount": metadata["rowCount"], "attributeIds": list(base.attribute_ids),
        "learnerMethods": list(base.learner_names), "embeddingMethods": list(base.embedding_names),
        "initialModelHoldoutIndependent": metadata.get("initialModelHoldoutIndependent", True), "referenceOnly": False,
        "calibration": metadata.get("calibration", CALIBRATION),
        "calibrationUsedValidation": metadata.get("calibrationUsedValidation", False),
        "classificationThreshold": metadata.get("classificationThreshold"),
        "modelSummary": {"beta": {attr: {method: 1/8 for method in base.learner_names} for attr in base.attribute_ids},
            "gamma": {attr: 1.0 for attr in base.attribute_ids},
            "embeddingWeights": {method: .5 for method in base.embedding_names}, "embeddingFusionStrength": .25,
            "theta": {attr: float(base.initial_theta[a]) for a, attr in enumerate(base.attribute_ids)},
            "temperature": {attr: float(base.temperature[a]) for a, attr in enumerate(base.attribute_ids)}}}


def visualization(service, task_id: str, fingerprint: str):
    from tuning_server import RefinementVisualizationResult
    loaded = load_task_baseline(service, task_id, fingerprint=fingerprint)
    if loaded is None:
        raise RuntimeError("Initial baseline is not published")
    base, metadata, directory = loaded
    n, c = metadata["rowCount"], metadata["componentCount"]
    payload = (directory / "visualization.f32").read_bytes()
    if len(payload) != 2 * n * c * 4:
        raise RuntimeError("Initial baseline visualization dimensions are invalid")
    ranks = np.frombuffer(payload, dtype="<f4", offset=n*c*4).reshape(n, c)
    mask = np.zeros(n, dtype=np.uint8)
    mask[base.development_indices] = 1
    return RefinementVisualizationResult(payload=payload, row_count=n, component_count=c,
        component_ranks=ranks, cluster_fit_mask=mask.tobytes(), base_state_fingerprint=fingerprint,
        source_fingerprint=digest([PROTOCOL, fingerprint, metadata["files"]["visualization.f32"]]))


def clusters(service, task_id: str, fingerprint: str, scheme: str):
    from tuning_server import DYNAMIC_PCP_CLUSTER_SCHEMES, DynamicPcpClusterResult, fit_dynamic_pcp_cluster_labels
    if scheme not in DYNAMIC_PCP_CLUSTER_SCHEMES:
        raise ValueError("Unknown PCP clustering scheme")
    view = visualization(service, task_id, fingerprint)
    key = ("initial:" + fingerprint, view.source_fingerprint, scheme)
    with service._refinement_cluster_guard:
        cached = service._refinement_clusters.get(key)
        if cached is not None:
            return view, cached
    labels, count, fitted = fit_dynamic_pcp_cluster_labels(view.component_ranks, view.cluster_fit_mask, scheme)
    result = DynamicPcpClusterResult(payload=labels.tobytes(), row_count=view.row_count, cluster_count=count,
                                     scheme=scheme, feature_count=view.component_count, fit_row_count=fitted)
    with service._refinement_cluster_guard:
        if len(service._refinement_clusters) >= 8:
            service._refinement_clusters.pop(next(iter(service._refinement_clusters)))
        service._refinement_clusters[key] = result
    return view, result


def main():
    from tuning_server import TuningService
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--bank-dir", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--status-json", type=Path, help="Stage completed training tasks in isolated subprocesses")
    args = parser.parse_args()
    service = TuningService(Path(__file__).resolve().parents[1], start_worker=False)
    if args.status_json:
        status = read(args.status_json.resolve())
        expected = {task["taskId"] for task in status["identity"]["tasks"]}
        if expected != set(service.tasks):
            raise RuntimeError("Training status does not cover the current complete catalogue")
        completed = {task["taskId"]: task for task in status["tasks"] if task.get("status") == "complete"}
        for task_id in sorted(completed):
            if args.task_id and task_id != args.task_id:
                continue
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--version", args.version,
                "--task-id", task_id, "--bank-dir", completed[task_id]["bankDirectory"]], check=False)
            if process.returncode:
                raise RuntimeError(f"Initial baseline staging failed: {task_id}")
        if args.publish:
            if set(completed) != expected:
                raise RuntimeError("Training is incomplete; no publication pointer was changed")
            print(json.dumps(publish(service, args.version), ensure_ascii=False))
        else:
            print(json.dumps({"stagedCompletedTasks": len(completed), "published": False}))
    elif args.publish:
        print(json.dumps(publish(service, args.version), ensure_ascii=False))
    elif args.task_id and args.bank_dir:
        print(json.dumps(build_task_baseline(service, args.task_id, args.version, args.bank_dir.resolve())[1], ensure_ascii=False))
    else:
        parser.error("Build requires --task-id and --bank-dir; publish requires --publish")


if __name__ == "__main__":
    main()
