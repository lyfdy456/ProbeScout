"""Preflight or explicitly train the eight probes under one frozen Val contract.

Default is preflight only. This does not publish scores, run VQA, tune fusion or
enable Probe-adaptive. The separate bank is input to a future baseline export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from val_isolation import METHODS, SEEDS, digest, load_task_contract, materialize_training_inputs


def main() -> None:
    from tuning_server import TuningService
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, help="Optional legacy Val contract directory")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise RuntimeError("epochs must be positive")
    web = Path(__file__).resolve().parents[1]
    service = TuningService(web, start_worker=False)
    if args.directory is not None:
        contract = load_task_contract(service, args.directory.resolve(), args.task_id)
    else:
        import portable_tasks
        from val_isolation import build_task_contract
        portable_tasks.validation_split(service, args.task_id, "joint")
        value, contract, _ = portable_tasks.document(service, args.task_id)
        if build_task_contract(service, args.task_id, value["validationProvenance"]["version"]) != contract:
            raise RuntimeError("Portable training contract differs from the installed task")
    context = service._tuning_source_context(args.task_id)
    adapter = context.adapter
    records = adapter.records.sort_values("embedding_index")
    if records["embedding_index"].astype(int).tolist() != list(range(len(records))):
        raise RuntimeError("Offline records are not in contiguous embedding order")
    paths = records["relative_path"].astype(str).tolist()
    inputs = materialize_training_inputs(contract, list(service.bundle(args.task_id).image_ids), paths)
    source_root = web.parents[2] / "probe_learning"
    # Bind trainer code as well as data; no resume across changed implementations.
    source_files = sorted((source_root / "src" / "methods").glob("*.py")) + [
        source_root / "scripts" / "run_retrieval_harness.py",
        source_root / "scripts" / "run_probebank_batch.py", Path(__file__).resolve(),
        Path(__file__).with_name("val_isolation.py"),
    ]
    code_sha = digest([[str(path.relative_to(web.parents[2])), hashlib.sha256(path.read_bytes()).hexdigest()]
                       for path in source_files])
    identity = {"protocol": "val-isolated-probebank-v1", "taskId": args.task_id,
                "isolationFingerprint": contract["fingerprint"], "epochs": args.epochs,
                "seeds": list(SEEDS), "methods": list(METHODS), "trainerSha256": code_sha}
    output = web.parent / "runtime" / "isolated-probes" / args.task_id / digest(identity)
    print(json.dumps({"taskId": args.task_id, "fitCount": len(inputs["fit_paths"]),
                      "valCount": len(inputs["val_labels"]), "seeds": list(SEEDS),
                      "normalizationCount": len(inputs["normalization_indices"]),
                      "output": str(output), "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    # Imports above use metadata only. Heavy features/training begin solely here.
    for path in (source_root, source_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_probebank_batch as bank
    import run_retrieval_harness as harness
    import fixed_vqa_validation as fixed

    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        previous, _ = fixed._read(identity_path)
        if previous != identity:
            raise RuntimeError("Isolated bank identity changed; use a new output namespace")
    else:
        if any(output.iterdir()):
            raise RuntimeError("Refusing an unverified existing output directory")
        fixed._write(identity_path, identity)
    harness.configure(contract["dataset"], contract["taskName"], adapter=adapter)
    if list(harness.ATTRS) != [attr["name"] for attr in contract["attributes"]]:
        raise RuntimeError("Adapter attribute identity changed")
    harness.EPOCHS = args.epochs
    bank.BACKBONE = "siglip"
    emb, _ = harness.load_backbone_embeddings(adapter, "siglip")
    patches, patch_meta = harness.load_backbone_patches(adapter, "siglip")
    text_queries = bank.cached_attribute_text_queries(
        list(harness.ATTRS), "siglip", harness.BACKBONE_TEXT_MODEL["siglip"])
    cache_identity = {**inputs["cache_identity"], "isolated_trainer_sha256": code_sha}
    supervision_digest = digest(inputs["fit_labels"])
    for method in METHODS:
        for attr in harness.ATTRS:
            text_query = bank.text_query_for_method(method, attr, text_queries)
            feature_context = bank.attention_feature_context(method, patch_meta, text_query)
            cached = bank.read_cache(
                output, contract["dataset"], contract["taskName"], "val_isolated",
                method, attr, emb, paths, list(SEEDS), supervision_digest,
                args.epochs, feature_context=feature_context, cache_identity=cache_identity,
            )
            if cached is None:
                cached = bank.train_cache_entry(
                    output, contract["dataset"], contract["taskName"], "val_isolated",
                    method, attr, emb, paths, adapter.p2i, inputs["train_pool"],
                    inputs["fit_paths"], inputs["fit_labels"], list(SEEDS),
                    supervision_digest, args.epochs, patches=patches, text_query=text_query,
                    feature_context=feature_context, cache_identity=cache_identity,
                    validation_labels=inputs["val_labels"],
                )
            metadata, _ = cached
            entry = bank.cache_dir(output, contract["dataset"], contract["taskName"],
                                   "val_isolated", method, attr)
            if metadata.get("fallback_seeds") or any(
                not (entry / f"model_seed_{seed}.pt").is_file() for seed in SEEDS
            ):
                raise RuntimeError("Isolated bank requires every trained seed checkpoint")
    fixed._write(output / "isolation_contract.json", contract)
    fixed._write(output / "complete.json", {**identity, "modelsRetrained": True,
                                            "published": False})


if __name__ == "__main__":
    main()
