"""Persist the existing local CLAY reproduction on the current 36-task Gallery.

Offline inference only: no TuningService, database, split creation, metrics,
training, downloads, publication, or writes to previous experiment artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROTOCOL = "clay-full-gallery-current-f0-v1"
METHOD = "sota_clay_siglip_b16_224"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    # Only new files inside this run's exclusive output directory.
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def checked_file(root, spec):
    path = (root / spec["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Bundle path escapes its task directory")
    payload = path.read_bytes()
    if len(payload) != spec["bytes"] or hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise ValueError(f"Bundle checksum/size mismatch: {path}")
    return payload


def prompt_bank(attrs, config):
    # Same ordering/deduplication as the original clay_prompt_bank helper.
    values = []
    for attr in attrs:
        attr = str(attr).strip().replace("_", " ")
        for templates in (config["positive_templates"], config["negative_templates"]):
            values.extend(template.format(attribute=attr) for template in templates)
    return list(dict.fromkeys(values))


def row_mapping(offline_ids, web_ids):
    if len(set(offline_ids)) != len(offline_ids) or len(set(web_ids)) != len(web_ids):
        raise ValueError("Duplicate image IDs")
    if len(web_ids) != len(offline_ids) or set(web_ids) != set(offline_ids):
        raise ValueError("Embedding and Web image IDs differ")
    lookup = {value: index for index, value in enumerate(offline_ids)}
    return np.asarray([lookup[value] for value in web_ids], dtype=np.int64)


def validate_scores(scores, rows, columns):
    values = np.asarray(scores)
    if values.shape != (rows, columns) or values.dtype != np.float32:
        raise ValueError("CLAY score dimensions/dtype differ from task")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.00001):
        raise ValueError("CLAY cosine scores are non-finite or outside [-1,1]")


def load_inputs(web, task_ids=None):
    catalog_path = web / "public/data/catalog.json"
    catalog = read_json(catalog_path)
    root = web.parent / "runtime/unified-initial"
    active_bytes = (root / "active.json").read_bytes()
    active = json.loads(active_bytes)
    publication = root / active["version"]
    if sha(publication / "manifest.json") != active["manifestSha256"]:
        raise ValueError("Initial F0 publication checksum differs")
    pub = read_json(publication / "manifest.json")
    entries = [(d["id"], t) for d in catalog["datasets"] for t in d["tasks"]]
    if task_ids is not None:
        requested = set(task_ids)
        if len(requested) != len(task_ids) or not requested <= {t["id"] for _, t in entries}:
            raise ValueError("Requested cohort is missing from the catalog")
        entries = [(d, t) for d, t in entries if t["id"] in requested]
    if not entries or not pub["complete"] or not {t["id"] for _, t in entries} <= set(pub["tasks"]):
        raise ValueError("Expected a complete F0 publication covering the selected catalog")
    contract_root = web.parent / "runtime/evaluation/val-isolation"
    # Match the immutable contract to the bank identity, not the latest name.
    contracts = {}
    for path in contract_root.glob("*/*.json"):
        value = read_json(path)
        if "taskId" in value and "fingerprint" in value:
            contracts[(value["taskId"], value["fingerprint"])] = (path, value)
    result = []
    for dataset, entry in entries:
        tid = entry["id"]
        directory = (web / "public" / entry["dataRoot"].lstrip("/")).resolve()
        if not directory.is_relative_to((web / "public/data").resolve()):
            raise ValueError("Catalog directory escapes public data")
        m = read_json(directory / "manifest.json")
        f0 = read_json(publication / tid / "manifest.json")
        if (sha(publication / tid / "manifest.json") != pub["tasks"][tid]["manifestSha256"]
                or f0["baseStateFingerprint"] != pub["tasks"][tid]["baseStateFingerprint"]):
            raise ValueError(f"Task F0 publication checksum differs: {tid}")
        ids = json.loads(checked_file(directory, m["files"]["imageIds"]))
        n = m["rowCount"]
        if len(ids) != n or f0["rowCount"] != n or f0["imageIdsSha256"] != digest(ids):
            raise ValueError(f"F0 image identity mismatch: {tid}")
        targets = [x["id"] for x in m["retrievalTargets"]]
        attrs = [x for x in m["retrievalTargets"] if x["kind"] == "attribute"]
        if (targets != f0["attributeIds"] + ["joint"] or len(attrs) != len(f0["attributeNames"])
                or any(x.get("canonicalAttribute", name) != name
                       for x, name in zip(attrs, f0["attributeNames"]))):
            raise ValueError(f"F0 attributes differ from Web targets: {tid}")
        query = m["query"]["images"]
        qi = [int(x["imageIndex"]) for x in query]
        if not qi or len(set(qi)) != len(qi) or any(i < 0 or i >= n for i in qi):
            raise ValueError(f"Invalid task query: {tid}")
        if any(ids[i] != x["imageId"] for i, x in zip(qi, query)):
            raise ValueError(f"Query ID/index mismatch: {tid}")
        truth = np.frombuffer(checked_file(directory, m["files"]["groundTruth"]), dtype=np.uint8).reshape(n, len(targets))
        test = np.frombuffer(checked_file(directory, m["files"]["testMask"]), dtype=np.uint8)
        if list(m["files"]["groundTruth"]["columns"]) != targets or not np.isin(truth, [0, 1]).all() or test.shape != (n,) or not np.isin(test, [0, 1]).all():
            raise ValueError(f"Invalid aligned evaluation labels/mask: {tid}")
        cp, contract = contracts[(tid, f0["bankTrainingIdentity"]["isolationFingerprint"])]
        train = contract["targets"]["joint"]
        if (not train["fitRows"] or len(train["fitRows"]) != len(train["fitLabels"])
                or len(set(train["fitRows"])) != len(train["fitRows"])
                or any(type(i) is not int or not 0 <= i < n for i in train["fitRows"])
                or any(y not in (0, 1) for y in train["fitLabels"])):
            raise ValueError(f"Invalid original training rows/labels: {tid}")
        if set(train["fitRows"]) & (set(contract["valRows"]) | set(np.flatnonzero(test)) | set(qi)):
            raise ValueError(f"Original training rows overlap protected scope: {tid}")
        result.append(dict(task_id=tid, dataset=dataset, label=entry["label"], directory=directory,
                           image_ids=ids, targets=targets, attrs=f0["attributeNames"], query=qi,
                           truth=truth, test=test, train=train, val=contract["valRows"],
                           val_labels=train["valLabels"],
                           source=dict(webManifestSha256=sha(directory / "manifest.json"),
                                       imageIdsSha256=m["files"]["imageIds"]["sha256"],
                                       groundTruthSha256=m["files"]["groundTruth"]["sha256"],
                                       testMaskSha256=m["files"]["testMask"]["sha256"],
                                       f0ManifestSha256=sha(publication / tid / "manifest.json"),
                                       isolationContractSha256=sha(cp))))
    return result, active, active_bytes, catalog_path


def encode_banks(banks, model_path, device):
    import torch
    from transformers import AutoTokenizer, SiglipModel
    model = SiglipModel.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    prompts = list(dict.fromkeys(p for bank in banks.values() for p in bank))
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 64):
            tokens = tokenizer(prompts[start:start+64], padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt")
            values = model.get_text_features(**{k: v.to(device) for k, v in tokens.items()})
            if not isinstance(values, torch.Tensor):
                values = getattr(values, "text_embeds", None) if getattr(values, "text_embeds", None) is not None else getattr(values, "pooler_output", None)
            if values is None:
                raise TypeError("SigLIP text encoder returned no embedding tensor")
            encoded.append(torch.nn.functional.normalize(values.float(), dim=-1).cpu().numpy())
    matrix = np.concatenate(encoded)
    lookup = {p: i for i, p in enumerate(prompts)}
    result = {k: matrix[[lookup[p] for p in bank]] for k, bank in banks.items()}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from threadpoolctl import threadpool_limits
    web = args.web.resolve()
    experiment = web.parents[2]
    linear = experiment / "probe_learning"
    sys.path.insert(0, str(linear))
    from src.data.dataset_adapter import build_adapter
    from src.methods.conditional_similarity_baselines import clay_scores, l2_normalize_np

    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config_path = experiment / "configs/clay.json"
    config = read_json(config_path)["clay"]
    model_path = args.model_path.resolve()
    model_config = read_json(model_path / "config.json")
    if model_config.get("model_type") != "siglip":
        raise ValueError("Expected the local SigLIP model")
    tasks, active, active_bytes, catalog_path = load_inputs(web)
    banks = {}
    for task in tasks:
        for target, attr in zip(task["targets"][:-1], task["attrs"]):
            banks[task["task_id"] + "/" + target] = prompt_bank([attr], config)
        banks[task["task_id"] + "/joint"] = prompt_bank(task["attrs"], config)
    provenance = dict(protocol=PROTOCOL, methodId=METHOD, taskCount=len(tasks),
        createdAt=datetime.now(timezone.utc).isoformat(), publication=active,
        galleryScope="all-Web-image-IDs", testScoring="slice-single-full-Gallery-scores",
        fidelity="official_geometry_local_binary_condition_banks", config=config,
        device=str(device), torchVersion=torch.__version__, numpyVersion=np.__version__,
        modelFiles={p.name: sha(p) for p in model_path.iterdir() if p.is_file()},
        codeSha256={"runner": sha(Path(__file__)), "clay": sha(linear / "src/methods/conditional_similarity_baselines.py")},
        catalogSha256=sha(catalog_path), externalConfigSha256=sha(config_path),
        usesLabelsForScores=False, fitsModels=False, computesMetrics=False,
        promptBanks=banks)
    # Refuse to overwrite or merge any old/partial run.
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "run-inputs.json", provenance)
    print(f"Loaded {len(tasks)} task inputs; encoding {len(banks)} text banks on {device}", flush=True)
    text = encode_banks(banks, model_path, device)
    with (args.output / "text-features.npz").open("xb") as handle:
        np.savez_compressed(handle, **text)
    print(f"Text encoding complete at {time.monotonic()-start:.1f}s; starting full-Gallery scoring", flush=True)
    source_hashes = {}
    cached_key, cached_images = None, None
    completed = []
    with threadpool_limits(limits=4):
        for number, task in enumerate(tasks, 1):
            tick = time.monotonic()
            tid = task["task_id"]
            adapter = build_adapter(task["dataset"], tid.split("_", 2)[2])
            records = adapter.records.sort_values("embedding_index")
            if list(map(int, records["embedding_index"])) != list(range(len(records))):
                raise ValueError(f"Non-contiguous embedding rows: {tid}")
            offline_ids = records["relative_path"].astype(str).tolist()
            mapping = row_mapping(offline_ids, task["image_ids"])
            if list(adapter.attrs) != task["attrs"] or list(adapter.key_slugs) + ["joint"] != task["targets"]:
                raise ValueError(f"Adapter differs from current modeled attributes: {tid}")
            lookup = {x: i for i, x in enumerate(task["image_ids"])}
            queries = [lookup[offline_ids[int(i)]] for i in adapter.query_idx]
            if (len(queries) != len(set(queries)) or len(queries) != len(task["query"])
                    or set(queries) != set(task["query"])):
                raise ValueError(f"Adapter query images differ: {tid}")
            path = Path(adapter.emb_path).with_name("siglip_embedding.npy")
            extra = getattr(adapter, "query_embedding_paths", {}).get("siglip")
            key = (str(path), str(extra))
            if key != cached_key:
                cached_images = None
                images = np.load(path, mmap_mode="r", allow_pickle=False)
                if extra is not None:
                    images = np.vstack([images, np.load(extra, allow_pickle=False)])
                # Same first normalization as the old load_backbone_embeddings.
                cached_images = l2_normalize_np(images)
                cached_key = key
            if (cached_images.shape != (len(offline_ids), 768) or not np.isfinite(cached_images).all()
                    or np.any(np.linalg.norm(cached_images, axis=1) < 1e-8)):
                raise ValueError(f"Invalid SigLIP image feature matrix: {tid}")
            images = cached_images if np.array_equal(mapping, np.arange(len(mapping))) else cached_images[mapping]
            sources = [path] + ([] if extra is None else [Path(extra)])
            for source in sources:
                if str(source) not in source_hashes:
                    source_hashes[str(source)] = sha(source)
            columns, effective = [], {}
            for target in task["targets"]:
                scores, rank = clay_scores(images, queries, text[tid + "/" + target],
                                          max_rank=int(config["max_rank"]), chunk_size=8192, device=device)
                columns.append(scores)
                effective[target] = rank
            scores = np.column_stack(columns).astype(np.float32, copy=False)
            validate_scores(scores, len(task["image_ids"]), len(task["targets"]))
            folder = args.output / tid
            folder.mkdir()
            with (folder / "scores.npz").open("xb") as handle:
                np.savez_compressed(handle, image_ids=np.asarray(task["image_ids"]),
                    target_ids=np.asarray(task["targets"]), scores=scores,
                    ground_truth=task["truth"], test_mask=task["test"],
                    query_indices=np.asarray(queries, dtype=np.int64),
                    original_train_indices=np.asarray(task["train"]["fitRows"], dtype=np.int64),
                    original_train_joint_labels=np.asarray(task["train"]["fitLabels"], dtype=np.uint8),
                    val_indices=np.asarray(task["val"], dtype=np.int64))
            with np.load(folder / "scores.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["scores"], scores) or saved["image_ids"].tolist() != task["image_ids"]:
                    raise ValueError(f"Saved scores/IDs differ: {tid}")
            meta = dict(taskId=tid, dataset=task["dataset"], label=task["label"],
                rowCount=len(scores), targetIds=task["targets"], attributeNames=task["attrs"],
                effectiveRanks=effective, queryCount=len(queries), source=task["source"],
                imageFeatureSha256={str(p): source_hashes[str(p)] for p in sources},
                scoreFileSha256=sha(folder / "scores.npz"), dtype="float32",
                scoreKind="CLAY projected cosine similarity", fullGalleryMeanForAlignment=True,
                seconds=round(time.monotonic()-tick, 3))
            write_json(folder / "manifest.json", meta)
            completed.append(meta)
            print(f"[{number:02}/36] {task['label']}: {scores.shape}, {meta['seconds']:.1f}s", flush=True)
            del adapter, records, images, scores
    if (web.parent / "runtime/unified-initial/active.json").read_bytes() != active_bytes or sha(catalog_path) != provenance["catalogSha256"]:
        raise ValueError("Active publication/catalog changed during this run")
    write_json(args.output / "manifest.json", dict(protocol=PROTOCOL, complete=True,
        methodId=METHOD, publication=active, taskCount=len(completed),
        totalImageRows=sum(x["rowCount"] for x in completed), tasks=completed,
        inputsSha256=sha(args.output / "run-inputs.json"),
        textFeaturesSha256=sha(args.output / "text-features.npz"),
        seconds=round(time.monotonic()-start, 3)))
    print(f"COMPLETE: {len(completed)} tasks in {time.monotonic()-start:.1f}s; {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    run(parser.parse_args())
