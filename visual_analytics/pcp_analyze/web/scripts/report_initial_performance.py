"""Current 36-task, 12-method AP and original-fit-selected-threshold F1.

Only saved scores are evaluated. No model is fitted, no score is changed, and
Test/Gallery ground truth never selects thresholds. New JSON/CSV reports are
the only writes; TuningService and the production database are not accessed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "probe_learning"))
from src.evaluation.retrieval_metrics import aligned_inputs, fit_training_threshold, evaluation_metrics

from run_clay_full_gallery import checked_file, load_inputs, read_json, sha, write_json

PROTOCOL = "current-f0-36-methods-train-threshold-ap-f1-v1"
PROBE_IDS = ("mlp_baseline", "kfold_pu", "triplet_loss", "attention_pooling",
             "attribute_conditioned_attention", "nnpu", "dcpu", "pu_ranking")
PROBE_LABELS = ("MLP", "K-Fold", "Triplet Loss", "Attention Pooling",
                "Attribute-conditioned Attention", "nnPU", "DC-PU", "Ours-PURA")
METHODS = [
    {"methodId": "query_maxsim", "methodLabel": "Query MaxSim", "methodOrder": 1},
    {"methodId": "text_prompt_ensemble", "methodLabel": "Text Prompt Ensemble", "methodOrder": 2},
    {"methodId": "sota_clay_siglip_b16_224", "methodLabel": "CLAY", "methodOrder": 3},
    *[{"methodId": mid, "methodLabel": label, "methodOrder": i + 4}
      for i, (mid, label) in enumerate(zip(PROBE_IDS, PROBE_LABELS))],
    {"methodId": "initial_f0", "methodLabel": "F0 (Overall)", "methodOrder": 12},
]








def evaluate_task_method(task, method, scores):
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(task["truth"][:, task["targets"].index("joint")], dtype=np.uint8)
    aligned_inputs(truth, scores)
    train_rows = np.asarray(task["train"]["fitRows"], dtype=np.int64)
    fitted = fit_training_threshold(task["train"]["fitLabels"], scores[train_rows])
    test = np.asarray(task["test"], dtype=bool)
    result = {"taskOrder": int(task["task_id"].split("_", 1)[0]), "taskId": task["task_id"],
              "taskLabel": task["label"], "dataset": task["dataset"], **method, **fitted}
    result.update({"test" + key: value for key, value in evaluation_metrics(truth[test], scores[test], fitted["threshold"]).items()})
    result.update({"gallery" + key: value for key, value in evaluation_metrics(truth, scores, fitted["threshold"]).items()})
    return result


def checked_current_scores(web, task, publication, clay_root, clay_entry):
    tid = task["task_id"]
    directory = web.parent / "runtime/unified-initial" / publication["version"] / tid
    f0 = read_json(directory / "manifest.json")
    n, a = len(task["image_ids"]), len(task["attrs"])
    if (f0["verified"] is not True or f0["version"] != publication["version"]
            or tuple(f0["probeMethodIds"]) != PROBE_IDS or tuple(f0["learnerMethods"]) != PROBE_LABELS
            or f0["embeddingMethods"] != ["Query MaxSim", "Text Prompt Ensemble"]
            or f0["bankTrainingIdentity"]["seeds"] != [0, 1, 2, 3, 4]):
        raise ValueError(f"Initial F0 method/seed identity differs: {tid}")
    for name in ("base.npz", "scores.f32", "ranks.f32"):
        if sha(directory / name) != f0["files"][name]:
            raise ValueError(f"Initial F0 source checksum differs: {tid}/{name}")
    with np.load(directory / "base.npz", allow_pickle=False) as data:
        probabilities = data["rawProbeProbabilities"]
    if (probabilities.shape != (n, a, 8) or probabilities.dtype != np.float32
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError(f"Invalid current mean-probability cube: {tid}")
    # Every method uses its own native Joint product after the agreed seed mean.
    probe_joint = np.prod(probabilities.astype(np.float64), axis=1, dtype=np.float64)
    f0_scores = np.fromfile(directory / "scores.f32", dtype="<f4")
    f0_ranks = np.fromfile(directory / "ranks.f32", dtype="<f4")
    if f0_scores.shape != (n,) or f0_ranks.shape != (n,) or not np.isfinite(f0_scores).all() or not np.isfinite(f0_ranks).all():
        raise ValueError(f"Invalid published F0 vectors: {tid}")
    if not np.array_equal(np.argsort(-f0_scores, kind="stable"), np.argsort(-f0_ranks, kind="stable")):
        raise ValueError(f"F0 published score/rank ordering differs: {tid}")

    wm = read_json(task["directory"] / "manifest.json")
    spec = wm["files"]["rawScores"]
    if spec["dtype"] != "float32" or spec["shape"] != [n, len(wm["methods"]), len(task["targets"])]:
        raise ValueError(f"Web raw-score layout differs: {tid}")
    raw_payload = checked_file(task["directory"], spec)
    raw = np.frombuffer(raw_payload, dtype="<f4").reshape(spec["shape"])
    target = task["targets"].index("joint")
    embeddings = [raw[:, wm["methods"].index(label), target].copy()
                  for label in ("Query MaxSim", "Text Prompt Ensemble")]
    if any(not np.isfinite(x).all() or np.any(np.abs(x) > 1.00001) for x in embeddings):
        raise ValueError(f"Invalid embedding raw cosine scores: {tid}")

    clay_dir = clay_root / tid
    # CLAY is independent of F0 gate calibration. A Val-selected child may use
    # the same CLAY cache only when every image/label/split hash is unchanged
    # and its recorded source is exactly the attested parent F0 manifest.
    clay_source = task["source"]
    if clay_entry["source"] != clay_source and f0.get("parentPublicationVersion"):
        parent = web.parent / "runtime/unified-initial" / f0["parentPublicationVersion"] / tid
        parent_meta = read_json(parent / "manifest.json")
        if parent_meta["baseStateFingerprint"] != f0["parentBaseStateFingerprint"]:
            raise ValueError(f"F0 parent identity differs: {tid}")
        clay_source = {**clay_source, "f0ManifestSha256": sha(parent / "manifest.json")}
    if (read_json(clay_dir / "manifest.json") != clay_entry or clay_entry["source"] != clay_source
            or clay_entry["rowCount"] != n or clay_entry["targetIds"] != task["targets"]
            or sha(clay_dir / "scores.npz") != clay_entry["scoreFileSha256"]):
        raise ValueError(f"CLAY source identity/checksum differs: {tid}")
    with np.load(clay_dir / "scores.npz", allow_pickle=False) as data:
        if (data["image_ids"].tolist() != task["image_ids"] or data["target_ids"].tolist() != task["targets"]
                or not np.array_equal(data["ground_truth"], task["truth"])
                or not np.array_equal(data["test_mask"], task["test"])
                or set(data["query_indices"].tolist()) != set(task["query"])
                or data["original_train_indices"].tolist() != task["train"]["fitRows"]
                or data["original_train_joint_labels"].tolist() != task["train"]["fitLabels"]
                or data["val_indices"].tolist() != task["val"]):
            raise ValueError(f"CLAY rows/labels/partition alignment differs: {tid}")
        clay = data["scores"]
        if clay.shape != (n, len(task["targets"])) or clay.dtype != np.float32 or not np.isfinite(clay).all() or np.any(np.abs(clay) > 1.00001):
            raise ValueError(f"Invalid CLAY full scores: {tid}")
        clay_joint = clay[:, target].copy()
    audit = {"taskId": tid, **task["source"], "f0BaseSha256": f0["files"]["base.npz"],
             "f0ScoresSha256": f0["files"]["scores.f32"], "f0RanksSha256": f0["files"]["ranks.f32"],
             "webRawScoresSha256": spec["sha256"], "webRawScoresShape": spec["shape"],
             "clayScoresSha256": clay_entry["scoreFileSha256"], "probeProbabilityShape": list(probabilities.shape),
             "imageCount": n, "queryCount": len(task["query"]), "valCount": len(task["val"]),
             "originalTrainCount": len(task["train"]["fitRows"]), "testCount": int(np.count_nonzero(task["test"]))}
    values = [*embeddings, clay_joint, *[probe_joint[:, i] for i in range(8)], f0_scores]
    return values, audit


def macro_summary(records):
    summaries = []
    for method in METHODS:
        selected = [row for row in records if row["methodId"] == method["methodId"]]
        if len(selected) != 36 or len({row["taskId"] for row in selected}) != 36:
            raise ValueError("Every method must cover the same 36 tasks exactly once")
        summary = {**method, "taskCount": len(selected)}
        for key in ("testAp", "testF1", "galleryAp", "galleryF1"):
            values = [row[key] for row in selected if row[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
            if key.endswith("Ap"):
                summary[key + "DefinedTasks"] = len(values)
        summaries.append(summary)
    return summaries


def build_report(web, clay_root):
    start = time.monotonic()
    tasks, active, active_bytes, catalog_path = load_inputs(web)
    tasks.sort(key=lambda item: (int(item["task_id"].split("_", 1)[0]), item["task_id"]))
    clay_manifest = read_json(clay_root / "manifest.json")
    if (clay_manifest.get("protocol") != "clay-full-gallery-current-f0-v1"
            or clay_manifest.get("complete") is not True or clay_manifest["publication"] != active
            or clay_manifest["taskCount"] != 36 or sha(clay_root / "run-inputs.json") != clay_manifest["inputsSha256"]):
        raise ValueError("CLAY publication is incomplete or belongs to another initial F0")
    clay_entries = {item["taskId"]: item for item in clay_manifest["tasks"]}
    if len(clay_entries) != 36 or set(clay_entries) != {task["task_id"] for task in tasks}:
        raise ValueError("CLAY task set differs from the current 36 tasks")
    catalog_sha = sha(catalog_path)
    records, audits = [], []
    for index, task in enumerate(tasks, 1):
        values, audit = checked_current_scores(web, task, active, clay_root, clay_entries[task["task_id"]])
        for method, scores in zip(METHODS, values):
            records.append(evaluate_task_method(task, method, scores))
        audits.append(audit)
        print(f"[{index:02d}/36] {task['task_id']}: 12 methods complete", flush=True)
    if ((web.parent / "runtime/unified-initial/active.json").read_bytes() != active_bytes
            or sha(catalog_path) != catalog_sha):
        raise ValueError("Active publication/catalog changed during evaluation")
    macros = macro_summary(records)
    fusion = [row for row in records if row["methodId"] == "initial_f0"]
    fusion_summary = {**macros[-1], "taskCount": len(fusion)}
    for scope in ("test", "gallery"):
        for field in ("Count", "PositiveCount", "TP", "FP", "FN"):
            fusion_summary[scope + field + "Sum"] = sum(row[scope + field] for row in fusion)
    return {"schemaVersion": 1, "protocol": PROTOCOL, "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "publication": active, "taskCount": 36, "methodCount": 12, "recordCount": len(records),
            "metricPolicy": {"aggregation": "equal-weight macro average across the same 36 tasks",
                "gallery": "all current Web image IDs, including Query, training, Val and Test",
                "test": "current frozen Test mask; slice the same full-Gallery scores",
                "ap": "sum precision at each positive position / all positives; stable descending score, equal score Gallery row ascending",
                "zeroPositiveAp": None, "f1": "fixed threshold selected on original VQA Joint fit rows only; score >= threshold",
                "thresholdSelection": "maximum training F1 over complete equal-score groups; tied F1 chooses highest threshold",
                "thresholdLabels": "original VQA joint fitLabels, not dataset GT; never use Val/Test/Gallery labels for threshold selection",
                "evaluationLabels": "dataset Joint GT for Test and Gallery",
                "seedAggregation": "Probe: mean five seed attribute probabilities, then product across attributes; not mean of seed metrics",
                "embeddingScores": "raw native Joint cosine from Web rawScores, without MinMax clipping",
                "clayScores": "saved native CLAY full-Gallery projected cosine; no artificial seeds",
                "fusionScores": "published initial F0 scores.f32, unchanged",
                "bestTestF1": False, "bestGalleryF1": False},
            "methods": METHODS, "records": records, "macro": macros, "fusionSummary": fusion_summary,
            "sourceAudit": {"catalogSha256": catalog_sha, "clayManifestSha256": sha(clay_root / "manifest.json"),
                "clayDirectory": str(clay_root), "scriptSha256": sha(Path(__file__)), "tasks": audits,
                "modelsTrained": False, "thresholdSelectionUsesEvaluationLabels": False,
                "databaseAccessed": False}, "seconds": round(time.monotonic() - start, 3)}


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; old reports are immutable")
    report = build_report(args.web.resolve(), args.clay_dir.resolve())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", report)
    write_csv(output / "records.csv", report["records"])
    write_csv(output / "macro.csv", report["macro"])
    print(json.dumps({"output": str(output), "records": len(report["records"]), "seconds": report["seconds"],
                      "macro": report["macro"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
