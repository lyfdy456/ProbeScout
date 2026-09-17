"""Validate, train and open a new attribute task on Cars, HICO or CelebA.

Run --check first. --execute trains all eight probes with five seeds, exports
the Web bundle, and registers it locally. No VQA requests or uploads are made.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "visual_analytics/pcp_analyze/web"
sys.path.insert(0, str(WEB / "scripts"))


def write(path, value):
    from export_web_data import write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def put_array(bundle, manifest, key, filename, values, dtype="float32", **kwargs):
    import numpy as np
    from export_web_data import file_spec, write_array
    values = np.asarray(values)
    write_array(bundle / filename, values, dtype)
    manifest["files"][key] = file_spec(bundle, filename, dtype, values.shape, **kwargs)


def put_json(bundle, manifest, key, filename, value, count):
    from export_web_data import json_file_spec
    write(bundle / filename, value)
    manifest["files"][key] = json_file_spec(bundle, filename, [count])


def create_inputs(value, ids, epochs):
    """Create a private input snapshot and an unregistered, metadata-only bundle."""
    import numpy as np
    from local_tasks import PROTOCOL, read, sha
    from val_isolation import digest
    config = value["config"]
    identity = digest({**value, "epochs": epochs})[:16]
    task_id = f"local_{config['dataset']}_{config['name']}_{identity}"
    version = f"local-{identity}"
    private = WEB.parent / "runtime/local-tasks" / task_id
    bundle = WEB / "public/data/local" / task_id
    snapshot = {**value, "taskId": task_id, "version": version, "epochs": epochs}
    path = private / "input.json"
    if path.exists() and read(path) != snapshot:
        raise RuntimeError("Existing local task input differs; use a new task name")
    write(path, snapshot)
    entry = {"id": task_id, "label": config["query_text"] + " (local)",
             "dataRoot": f"/data/local/{task_id}", "defaultRetrievalTarget": "joint"}
    catalog = {"schemaVersion": 1, "defaultDataset": config["dataset"], "defaultTask": task_id,
               "taskCount": 1, "datasets": [{"id": config["dataset"], "label": config["dataset"].upper(), "tasks": [entry]}]}
    write(private / "catalog.json", catalog)
    if not (bundle / "manifest.json").exists():
        bundle.mkdir(parents=True, exist_ok=True)
        attrs = config["attributes"]
        targets = [{"id": a["id"], "label": a["name"], "canonicalAttribute": a["name"], "kind": "attribute"} for a in attrs]
        targets.append({"id": "joint", "label": "Joint", "kind": "derived", "members": [a["id"] for a in attrs], "rule": "product"})
        from tuning_server import EMBEDDING_BASELINE_METHODS, WEIGHTED_FUSION_LEARNERS
        methods = [*EMBEDDING_BASELINE_METHODS, *WEIGHTED_FUSION_LEARNERS, "Ours-Full"]
        manifest = {"schemaVersion": 2, "rowCount": len(ids), "methodCount": len(methods),
            "targetCount": len(targets), "methods": methods, "clusterMethods": methods[:-1],
            "clusterMethodCount": len(methods)-1, "retrievalTargets": targets,
            "defaultRetrievalTarget": "joint", "defaultRankMethod": "Ours-Full", "files": {},
            "localTask": {"protocol": PROTOCOL, "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)},
            "query": {"mode": "fixed", "text": config["query_text"], "images": [
                {"imageId": ids[row], "imageIndex": row} for row in value["queryRows"]]}}
        put_json(bundle, manifest, "imageIds", "image-ids.json", ids, len(ids))
        val = np.zeros(len(ids), dtype=np.uint8)
        val[value["validationRows"]] = 1
        dev = 1 - val
        dev[value["queryRows"]] = 0
        for key, filename, array in [("developmentMask", "development-mask.u8", dev),
                ("validationMask", "validation-mask.u8", val), ("testMask", "test-mask.u8", np.zeros_like(dev))]:
            put_array(bundle, manifest, key, filename, array, "uint8")
        write(bundle / "manifest.json", manifest)
    return task_id, version, private, bundle, catalog


def embedding_scores(adapter):
    """Existing five training-free baselines; no ground-truth labels are read."""
    import numpy as np
    import run_retrieval_harness as harness
    harness.configure(adapter.dataset, adapter.task, adapter=adapter)
    emb, _ = harness.load_backbone_embeddings(adapter, "siglip")
    x = harness.l2norm(emb)
    prototype = harness.l2norm(x[adapter.query_idx].mean(axis=0, keepdims=True))[0]
    image = x @ prototype
    maxsim = (x @ x[adapter.query_idx].T).max(axis=1)
    text = harness.text_feats_for_backbone("siglip")
    prompt = harness.prompt_ensemble_feats_for_backbone("siglip")
    combined = x @ harness.l2norm(.5 * prototype[None, :] + .5 * text).T
    keys = [harness.ranking_key_for_combo((a,)) for a in adapter.attrs] + [harness.JOINT_KEY]
    result = np.empty((len(x), 5, len(keys)), dtype=np.float32)
    for a, key in enumerate(keys):
        score = x @ prompt[key]
        result[:, :, a] = np.column_stack([image, maxsim,
            combined[:, a] if a < len(adapter.attrs) else combined.mean(axis=1),
            score, .5*harness._zscore(image) + .5*harness._zscore(score)])
    return result


def export_scores(service, task_id, bundle, baseline):
    import numpy as np
    from local_tasks import read
    from tuning_server import normalized_ranks
    from tuning_models import unified_weight_scores
    from unified_initial_baseline import minmax
    base = baseline[0]
    manifest = read(bundle / "manifest.json")
    shape = (manifest["rowCount"], manifest["methodCount"], manifest["targetCount"])
    raw = np.fromfile(bundle / "raw-scores.f32", dtype="<f4").reshape(shape)
    a = len(base.attribute_ids)
    raw[:, 5:13, :a] = base.raw_probe_probabilities.transpose(0, 2, 1)
    raw[:, 5:13, a] = base.raw_probe_probabilities.prod(axis=1)
    output = unified_weight_scores(base.probe_features, base.embedding_features,
        np.full((a, 8), 1/8), np.ones(a), np.full(2, .5), .25, base.initial_theta, base.temperature)
    raw[:, 13, :a] = output.gates
    raw[:, 13, a] = output.final_scores
    calibrated, _, _ = minmax(raw, base.development_indices)
    ranks = np.stack([normalized_ranks(raw[:, m, t]) for m in range(shape[1])
                      for t in range(shape[2])], axis=1).reshape(shape)
    for key, filename, array in [("rawScores", "raw-scores.f32", raw),
            ("calibratedScores", "calibrated-scores.f32", calibrated), ("ranks", "ranks.f32", ranks)]:
        put_array(bundle, manifest, key, filename, array)
    write(bundle / "manifest.json", manifest)


def export_views(service, task_id, bundle):
    import numpy as np
    from local_tasks import read, sha
    from export_web_data import atlas_config, build_atlases, compute_pca
    from add_fine_clusters import fit_labels
    from export_task_bundles import export_visual_embedding_bundle
    manifest = read(bundle / "manifest.json")
    adapter = service._source_adapter(task_id)
    ids = list(service.bundle(task_id).image_ids)
    n = len(ids)
    ranks = np.fromfile(bundle / "ranks.f32", dtype="<f4").reshape(n, manifest["methodCount"], manifest["targetCount"])
    profiles = ranks[:, :-1, -1]
    mask = np.frombuffer(service.bundle(task_id).development_mask, dtype=np.uint8)
    fit = np.flatnonzero(mask)
    standardized = (profiles-profiles[fit].mean(axis=0)) / np.maximum(profiles[fit].std(axis=0), 1e-8)
    pca, explained = compute_pca(standardized, mask)
    put_array(bundle, manifest, "pca2d", "pca-2d.f32", pca)
    summaries, schemes = [], []
    for k in (30, 50, 100):
        labels, centers, counts, _ = fit_labels(profiles, standardized, k, 42, fit)
        key = f"fine{k}Labels"
        put_array(bundle, manifest, key, f"fine{k}-labels.u8", labels, "uint8")
        schemes.append({"id": f"fine{k}", "label": "Fine-grained", "family": "fine", "clusters": k, "labelsFileKey": key})
        for cluster, (center, size) in enumerate(zip(centers, counts)):
            summaries.append({"scheme": f"fine{k}", "cluster_id": cluster, "label": f"K{k} cluster {cluster+1}",
                "label_zh": f"K{k} cluster {cluster+1}", "size": int(size), "fraction": float(size/n),
                "mean_rank": float(center.mean()), "learned_minus_fixed": float(center[5:].mean()-center[:5].mean())})
    # Only rank statistics are emitted; unavailable ground truth has no placeholder labels.
    metrics = np.column_stack([profiles.mean(axis=1), profiles.std(axis=1), profiles[:, :5].mean(axis=1), profiles[:, 5:].mean(axis=1)])
    put_array(bundle, manifest, "metrics", "metrics.f32", metrics,
              columns=["mean_rank", "rank_std", "fixed_mean", "learned_mean"])
    manifest["clusters"] = {"basisTarget": "joint", "defaultScheme": "fine50", "schemes": schemes}
    manifest["projections"] = {"basisTarget": "joint", "pca": {"available": True, "explainedVarianceRatio": explained,
        "fitScope": "development", "fitMaskFileKey": "developmentMask"},
        "umap": {"available": False, "reason": "Use the SigLIP visual UMAP projection"}}
    args = SimpleNamespace(atlas_columns=10, atlas_rows=10, tile_width=96, tile_height=72,
                           atlas_workers=4, force_atlases=False, webp_quality=62)
    thumbs = atlas_config(args, n)
    thumbs.update(build_atlases(ids, adapter.raw_images_dir, bundle, thumbs, args))
    if thumbs["missingSourceCount"]:
        raise RuntimeError("Original images could not be decoded; fix images before rerunning")
    manifest["thumbnails"] = thumbs
    for query in manifest["query"]["images"]:
        source = adapter.raw_images_dir / query["imageId"]
        filename = f"query-pics/{query['imageIndex']}{source.suffix}"
        (bundle / "query-pics").mkdir(exist_ok=True)
        shutil.copyfile(source, bundle / filename)
        query.update(path=filename, mimeType=mimetypes.guess_type(filename)[0] or "image/jpeg", bytes=source.stat().st_size, sha256=sha(source))
    empty_counts = {t["id"]: 0 for t in manifest["retrievalTargets"]}
    # No Frozen Test exists for user supervision alone. Val is a held-out subset of that supervision.
    evaluation = {"defaultResultScope": "development", "scopes": [{"id": "development", "label": "Development Gallery", "rowCount": int(mask.sum())}],
        "development": {"rowCount": int(mask.sum()), "maskFileKey": "developmentMask", "queryExcluded": True,
                        "positiveCounts": {}, "groundTruthUsage": "unavailable; feedback and fitting only"},
        "validation": {"rowCount": len(read(service.web_root.parents[2] / manifest["localTask"]["path"])["validationRows"]),
                       "maskFileKey": "validationMask", "positiveCounts": {}, "groundTruthUsage": "private user-supervision holdout"},
        "frozenTest": {"rowCount": 0, "maskFileKey": "testMask", "positiveCounts": empty_counts,
                       "groundTruthUsage": "unavailable; no independent Test labels supplied"}}
    manifest["evaluation"] = evaluation
    metadata = {"schemaVersion": 2, "task": {"dataset": adapter.dataset, "task": task_id, "taskName": adapter.task,
        "split": "gallery", "defaultRetrievalTarget": "joint", "baseAttributes": adapter.key_slugs,
        "retrievalTargets": manifest["retrievalTargets"]}, "clusterSummary": summaries, "thumbnails": thumbs, "evaluation": evaluation}
    put_json(bundle, manifest, "metadata", "metadata.json", metadata, 1)
    write(bundle / "manifest.json", manifest)
    export_visual_embedding_bundle(SimpleNamespace(task_id=task_id, data_root=bundle), adapter=adapter)


def register(catalog):
    from local_tasks import read
    target = WEB / "public/data/catalog.json"
    current = read(target) if target.exists() else {"schemaVersion": 1, "datasets": []}
    incoming = catalog["datasets"][0]
    dataset = next((d for d in current["datasets"] if d["id"] == incoming["id"]), None)
    if dataset is None:
        dataset = {"id": incoming["id"], "label": incoming["label"], "tasks": []}
        current["datasets"].append(dataset)
    entry = incoming["tasks"][0]
    dataset["tasks"] = [t for t in dataset["tasks"] if t["id"] != entry["id"]] + [entry]
    current.update(defaultDataset=incoming["id"], defaultTask=entry["id"],
                   taskCount=sum(len(d["tasks"]) for d in current["datasets"]))
    write(target, current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Path to task.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate inputs without generating assets or training")
    mode.add_argument("--execute", action="store_true", help="Train, export and register in the local Web catalog")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    import numpy as np
    from local_tasks import validate_input, read, sha
    value, ids, processed = validate_input(args.task, ROOT)
    raw = processed.parent / ("img_celeba" if value["config"]["dataset"] == "celeba" else "images")
    missing = [name for name in ids if not (raw/name).is_file()]
    if missing:
        raise RuntimeError(f"Original images are required for visualization: {len(missing)} missing; first {missing[0]}")
    value["queryImageHashes"] = {name: sha(raw/name) for name in value["config"]["query_images"]}
    value["featureFiles"] = {}
    for name, dims in [("siglip_embedding.npy", 2), ("siglip_patch_tokens.npy", 3)]:
        path = processed/name
        if not path.is_file():
            raise RuntimeError(f"Missing {path.name}; download HF features or run the feature extraction commands")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != dims or len(array) != len(ids) or array.dtype.kind != "f":
            raise RuntimeError(f"{name} must have {dims} dimensions and canonical records.csv row order")
        value["featureFiles"][name] = {"bytes": path.stat().st_size, "mtimeNs": path.stat().st_mtime_ns}
    print(f"Inputs checked: {len(ids)} images, {len(value['labels'])} labeled, {len(value['validationRows'])} fixed Val rows.", flush=True)
    if args.check:
        print("Ready. Add --execute instead of --check to train and export; no API key is needed.")
        return
    task_id, version, private, bundle, catalog = create_inputs(value, ids, args.epochs)
    from tuning_server import TuningService
    service = TuningService(WEB, start_worker=False, catalog_path=private/"catalog.json")
    if (private / "complete.json").exists():
        if read(private / "complete.json").get("manifestSha256") != sha(bundle / "manifest.json"):
            raise RuntimeError("Completed local Web manifest changed; restore it before enabling the task")
        from unified_initial_baseline import load_task_baseline
        if load_task_baseline(service, task_id) is None:
            raise RuntimeError("Completed task is missing its initial baseline")
        register(catalog)
        print(f"Enabled existing task {task_id}; restart the Web app.")
        return
    from fixed_vqa_validation import freeze_vqa_validation
    from val_isolation import build_task_contract
    if not (service.vqa_validation_root / version / "manifest.json").exists():
        freeze_vqa_validation(service, version, activate=False)
    contract = build_task_contract(service, task_id, version)
    from train_val_isolated_probes import train_task
    bank = train_task(service, task_id, contract, epochs=args.epochs, execute=True)
    print("Training complete; exporting query baselines and verified initial fusion.", flush=True)
    manifest = read(bundle / "manifest.json")
    if "rawScores" not in manifest["files"]:
        raw_scores = np.zeros((len(ids), manifest["methodCount"], manifest["targetCount"]), dtype=np.float32)
        raw_scores[:, :5] = embedding_scores(service._source_adapter(task_id))
        put_array(bundle, manifest, "rawScores", "raw-scores.f32", raw_scores)
        write(bundle / "manifest.json", manifest)
    service = TuningService(WEB, start_worker=False, catalog_path=private/"catalog.json")
    from unified_initial_baseline import build_task_baseline, publish
    baseline = build_task_baseline(service, task_id, version, bank)
    export_scores(service, task_id, bundle, baseline)
    export_views(service, task_id, bundle)
    publish(service, version, activate=False)
    write(private / "complete.json", {"taskId": task_id, "version": version, "bank": bank.name,
                                      "manifestSha256": sha(bundle / "manifest.json")})
    register(catalog)
    print(f"Registered {task_id}. Run npm run dev in visual_analytics/pcp_analyze/web, then open http://localhost:3000")


if __name__ == "__main__":
    main()
