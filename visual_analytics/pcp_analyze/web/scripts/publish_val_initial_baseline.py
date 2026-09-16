"""Derive an immutable F0 from frozen, Val-selected gates; never retrain probes.

Stage first (default), inspect/verify, then pass --publish for one atomic switch.
Existing publications, checkpoint banks and personal tuning records are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import uuid

import numpy as np

import unified_initial_baseline as initial
from tuning_models import unified_weight_scores
from tuning_server import TuningService, normalized_ranks


def stage_task(service, version, inputs, selection, record, source_audit):
    task_id = record["taskId"]
    parent = initial.load_task_baseline(service, task_id, selection["sourceVersion"])
    if parent is None:
        raise RuntimeError(f"Source baseline missing: {task_id}")
    base, old, source = parent
    if (initial.sha(source / "manifest.json") != source_audit["manifestSha256"]
            or old["baseStateFingerprint"] != source_audit["oldBaseStateFingerprint"]
            or old["attributeIds"] != record["attributeIds"]
            or old["rowCount"] != record["rowCount"]):
        raise RuntimeError("Offline selection and published source identities differ")
    contract_path = Path(source_audit["isolationContract"])
    contract = initial.read(contract_path)
    if (initial.sha(contract_path) != source_audit["isolationContractSha256"]
            or contract["fingerprint"] != old["bankTrainingIdentity"]["isolationFingerprint"]):
        raise RuntimeError("Val selection isolation contract changed")
    val_rows = set(contract["valRows"])
    if any(val_rows & set(contract[key]) for key in ("normalizationRows", "testRows")):
        raise RuntimeError("Val selection overlaps normalization/Test")
    if val_rows & set(contract["targets"]["joint"]["fitRows"]):
        raise RuntimeError("Val selection overlaps supervised fit")
    source_link_path = service.web_root.parent / "runtime/isolated-probes" / task_id / old["bankDirectory"] / "refinement_base.json"
    source_link = initial.read(source_link_path)
    if (source_link["baseStateFingerprint"] != base.base_state_fingerprint
            or source_link["normalizationContract"] != base.normalization_audit
            or source_link.get("forwardVerified") is not True
            or initial.digest(source_link["files"]) != old["bankFingerprint"]):
        raise RuntimeError("Parent checkpoint attestation does not match the verified source")
    expected_path = inputs / task_id / "scores.f32"
    if initial.sha(expected_path) != record["scoreSha256"]:
        raise RuntimeError("Offline Val-selected score checksum mismatch")
    destination = initial.version_directory(service, version) / task_id
    if destination.exists():
        loaded = initial.load_task_baseline(service, task_id, version)
        if (loaded is None or loaded[1].get("selectionSha256") != initial.sha(inputs / "val_selection.json")
                or initial.sha(destination / "scores.f32") != record["scoreSha256"]):
            raise RuntimeError("Refusing to overwrite another staged baseline")
        initial.published_bank_attestation(service, task_id, loaded[0])
        return loaded[1]
    theta = np.asarray(record["actualTheta"], dtype="<f8")
    temperature = np.asarray(record["actualT"], dtype="<f8")
    n, a, _ = base.probe_features.shape
    output = unified_weight_scores(base.probe_features, base.embedding_features,
        np.full((a, 8), 1 / 8), np.ones(a), np.full(2, .5), .25, theta, temperature)
    if output.final_scores.astype("<f4").tobytes() != expected_path.read_bytes():
        raise RuntimeError(f"Website scoring does not exactly reproduce offline F0: {task_id}")
    columns = [output.final_scores, output.conjunction_scores]
    for index in range(a):
        columns.extend([output.gates[:, index], *[base.probe_features[:, index, m] for m in range(8)]])
    columns.extend([output.holistic_scores, base.embedding_features[:, 0], base.embedding_features[:, 1]])
    matrix = np.column_stack(columns).astype("<f4")
    ranks = np.column_stack([normalized_ranks(matrix[:, c]) for c in range(matrix.shape[1])]).astype("<f4")
    temporary = destination.with_name(f".{task_id}.{uuid.uuid4().hex}.pending")
    temporary.mkdir(parents=True, exist_ok=False)
    with np.load(source / "base.npz", allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    arrays.update(theta=theta, temperature=temperature)
    np.savez(temporary / "base.npz", **arrays)
    (temporary / "scores.f32").write_bytes(output.final_scores.astype("<f4").tobytes())
    (temporary / "ranks.f32").write_bytes(ranks[:, 0].tobytes())
    (temporary / "visualization.f32").write_bytes(np.concatenate([matrix.ravel(), ranks.ravel()]).astype("<f4").tobytes())
    calibration = {"protocol": initial.VAL_CALIBRATION, "usesValLabels": True, "usesTestLabels": False,
        "selectedAt": selection["selectedAt"], "selectionSha256": initial.sha(inputs / "val_selection.json"),
        "gateSelection": selection["gateSelection"], "cutoffSelection": selection["cutoffSelection"],
        "case": record["case"], "thetaShift": record["thetaShift"], "TScale": record["TScale"],
        "theta": theta.tolist(), "temperature": temperature.tolist(), "classificationThreshold": record["tau"],
        "valN": record["valN"], "valP": record["valP"], "valAp": record["valAp"], "valF1": record["valF1"],
        "isolationFingerprint": contract["fingerprint"], "gateArrayEncoding": "<f8"}
    initial.write(temporary / "calibration.json", calibration)
    initial.write(temporary / "verification.json", {"parentPublicationVersion": selection["sourceVersion"],
        "parentManifestSha256": initial.sha(source / "manifest.json"), "parentBaseStateFingerprint": base.base_state_fingerprint,
        "parentBankAttestationSha256": initial.sha(source_link_path), "sourceAudit": source_audit,
        "scoreSha256": record["scoreSha256"], "websiteScoresExactlyMatchOffline": True,
        "probeFeaturesUnchanged": True, "normalizationUnchanged": True, "modelsRetrained": False,
        "createdAt": datetime.now().astimezone().isoformat(timespec="seconds")})
    files = {path.name: initial.sha(path) for path in temporary.iterdir() if path.is_file()}
    fingerprint = initial.digest({"protocol": initial.PROTOCOL, "parent": base.base_state_fingerprint,
        "files": files, "calibration": initial.VAL_CALIBRATION, "bankFingerprint": old["bankFingerprint"]})
    # The attestation is separately hashed by the manifest to avoid a circular
    # fingerprint and, crucially, never replaces the old bank's own link.
    initial.write(temporary / "bank_attestation.json", {**source_link, "baseStateFingerprint": fingerprint})
    metadata = {**old, "version": version, "baseStateFingerprint": fingerprint, "files": files,
        "calibration": initial.VAL_CALIBRATION, "calibrationUsedValidation": True,
        "initialModelHoldoutIndependent": False, "probeFitExcludesValidation": True,
        "gateArrayEncoding": "<f8", "classificationThreshold": record["tau"],
        "selectionSha256": initial.sha(inputs / "val_selection.json"),
        "parentPublicationVersion": selection["sourceVersion"], "parentBaseStateFingerprint": base.base_state_fingerprint,
        "bankAttestation": {"path": "bank_attestation.json", "sha256": initial.sha(temporary / "bank_attestation.json")}}
    initial.write(temporary / "manifest.json", metadata)
    os.replace(temporary, destination)
    loaded = initial.load_task_baseline(service, task_id, version)
    initial.published_bank_attestation(service, task_id, loaded[0])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    inputs = args.inputs.resolve()
    selection = initial.read(inputs / "val_selection.json")
    results = initial.read(inputs / "results.json")
    service = TuningService(Path(__file__).resolve().parents[1], start_worker=False)
    if (results.get("complete") is not True or selection["protocol"] != initial.VAL_CALIBRATION
            or results["selectionSha256"] != initial.sha(inputs / "val_selection.json")):
        raise RuntimeError("Incomplete/unverified offline selection inputs")
    records = {row["taskId"]: row for row in results["records"]}
    choices = {row["taskId"]: row for row in selection["choices"]}
    audit = {row["taskId"]: row for row in selection["sourceAudit"]}
    if not set(records) == set(choices) == set(audit) == set(service.tasks):
        raise RuntimeError("Selection inputs do not cover exactly the current full catalogue")
    parent = initial.version_directory(service, selection["sourceVersion"])
    if initial.sha(parent / "manifest.json") != selection["sourcePublicationSha256"]:
        raise RuntimeError("Source publication identity changed")
    for index, task_id in enumerate(sorted(records), 1):
        for key in ("actualTheta", "actualT", "tau", "case"):
            if records[task_id][key] != choices[task_id][key]:
                raise RuntimeError("Results changed the frozen Val choices")
        stage_task(service, args.version, inputs, selection, records[task_id], audit[task_id])
        print(f"Verified {index}/{len(records)} {task_id}", flush=True)
    if args.publish:
        initial.publish(service, args.version)
        print(f"Published {args.version}: {len(records)} tasks", flush=True)
    else:
        print("All tasks staged; active pointer unchanged", flush=True)


if __name__ == "__main__":
    main()
