#!/usr/bin/env python3
"""Export audited per-attribute and Joint method scores into one compact NPZ.

The existing PCP builder intentionally emits one retrieval target at a time.
This companion exporter reuses its audited task/cache loading and score
reconstruction functions, but materializes every task attribute plus the full
Joint target together.  The array layout is always ``[image, method, target]``.

Three score views are stored:

``raw_scores``
    Each method's deployed score.  Learned ProbeBank entries are cached
    post-sigmoid probabilities (not logits), deterministic embedding methods
    retain their native cosine/z-score definitions, and Ours-Full retains its
    deployed SoftGate output.  Multi-seed deployed scores are averaged.

``calibrated_scores``
    A visualization/comparison view obtained by fitting min/max on the frozen
    train pool for each method-target pair, then clipping the exported split to
    ``[0, 1]``.  This external calibration does not redefine retrieval ranks.

``ranks``
    The existing PCP ranking policy: deterministic methods use their score
    percentile and multi-seed methods use the mean of seed-wise percentiles.
    One is highest ranked and zero is lowest ranked.

Typical usage from the repository root::

    python visual_analytics/pcp_analyze/export_multitarget_scores.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = ROOT / "configs/main17_tasks.json"
DEFAULT_SUITE = ROOT / "configs/probe_suite.json"
DEFAULT_OURS_FULL_ROOT = (
    REPOSITORY_ROOT
    / "outputs"
    / "formal23_iterative_selected8_r_softgate_front_minmax_allv2_v1"
)
LEGACY_DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "pcp_result"
    / "032_hico_hugging_cat_multitarget_scores.npz"
)

# Loading the audited PCP module also imports the training stack.  Keep that
# import behind ``main`` so ``--help`` remains usable in a lightweight Python
# environment; every actual score operation below still calls that module.
P: Any = None


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique CSV list")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--order",
        type=int,
        default=32,
        help="Task order in the experiment manifest (default: 32).",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--stage",
        choices=("iterative", "two_stage"),
        default="iterative",
    )
    parser.add_argument(
        "--split",
        choices=("all", "gallery", "test", "train_pool"),
        default="all",
        help="Exported image split; calibration always uses frozen train_pool.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("0,1,2,3,4"),
    )
    parser.add_argument(
        "--tie-policy",
        choices=("average", "stable"),
        default="average",
    )
    parser.add_argument("--backbone", choices=("siglip",), default="siglip")
    parser.add_argument(
        "--ours-full-root",
        type=Path,
        default=DEFAULT_OURS_FULL_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output NPZ. By default order 32 keeps its historical filename; "
            "other tasks use <task-key>_multitarget_scores.npz in pcp_result/."
        ),
    )
    return parser.parse_args()


def configured_learned_methods(suite: dict[str, Any]) -> list[str]:
    configured = list(suite["learned_methods"])
    return [
        method for method in P.LEARNED_METHOD_ORDER if method in configured
    ] + [
        method for method in configured if method not in P.LEARNED_METHOD_ORDER
    ]


def mean_deployed_score(score_rows: np.ndarray) -> np.ndarray:
    values = np.asarray(score_rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected non-empty [seed, image] scores, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Deployed score rows contain NaN or infinity")
    return np.asarray(values.mean(axis=0), dtype=np.float32)


def train_pool_minmax(
    full_scores: np.ndarray,
    train_indices: np.ndarray,
    export_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(full_scores, dtype=np.float64)
    train = values[np.asarray(train_indices, dtype=np.int64)]
    if train.size == 0:
        raise ValueError("Frozen train_pool is empty; min-max calibration is undefined")
    if not np.isfinite(values).all() or not np.isfinite(train).all():
        raise ValueError("Cannot calibrate scores containing NaN or infinity")

    low = float(np.min(train))
    high = float(np.max(train))
    span = high - low
    if span <= np.finfo(np.float64).eps:
        calibrated_full = np.zeros_like(values, dtype=np.float64)
        status = "constant_train_pool_zero_fallback"
    else:
        calibrated_full = np.clip((values - low) / span, 0.0, 1.0)
        status = "fitted"

    below = int(np.count_nonzero(values < low))
    above = int(np.count_nonzero(values > high))
    calibration = {
        "status": status,
        "trainMin": low,
        "trainMax": high,
        "trainSpan": span,
        "trainCount": int(train.size),
        "fullCount": int(values.size),
        "belowTrainMinCount": below,
        "aboveTrainMaxCount": above,
        "clippedCount": below + above,
        "clippedRate": float((below + above) / max(1, values.size)),
    }
    selected = calibrated_full[np.asarray(export_indices, dtype=np.int64)]
    return np.asarray(selected, dtype=np.float32), calibration


def target_specifications(
    adapter: Any,
    attributes: Sequence[str],
) -> list[tuple[str, str | None]]:
    """Return stable browser target IDs paired with canonical attributes.

    ``DatasetAdapter.key_slugs`` is the canonical filesystem-safe identifier
    already used by the training/cache pipeline.  The final ``joint`` target
    represents the conjunction of every task attribute.
    """

    target_ids = [str(value).strip() for value in adapter.key_slugs]
    if len(target_ids) != len(attributes):
        raise ValueError(
            "Adapter key_slugs do not align with task attributes: "
            f"{target_ids} versus {list(attributes)}"
        )
    if any(not value for value in target_ids):
        raise ValueError(f"Adapter contains an empty target slug: {target_ids}")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(f"Adapter target slugs are not unique: {target_ids}")
    if "joint" in target_ids:
        raise ValueError("Adapter target slug 'joint' is reserved for the conjunction")
    return [*zip(target_ids, attributes, strict=True), ("joint", None)]


def joint_rule_by_method(
    learned_methods: Sequence[str],
    attributes: Sequence[str],
) -> dict[str, str]:
    attribute_count = len(attributes)
    rules = {
        "image_prototype": (
            "task-level query-image prototype similarity; target-invariant"
        ),
        "query_maxsim": (
            "maximum similarity to a task query image; target-invariant"
        ),
        "img_text_fusion": (
            f"mean of all {attribute_count} attribute-specific image-text fusion scores"
        ),
        "text_prompt_ensemble": "dedicated Joint prompt-ensemble score",
        "zscore_img_text_fusion": (
            "split-local z-score fusion using the dedicated Joint prompt score"
        ),
        P.OURS_FULL_ID: (
            f"within each seed, product of all {attribute_count} deployed "
            "Ours-Full attribute SoftGates"
        ),
    }
    for method in learned_methods:
        rules[method] = (
            f"within each seed, product of all {attribute_count} cached "
            "post-sigmoid attribute scores"
        )
    return rules


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    global P
    import build_score_rank_pcp as audited_pcp

    P = audited_pcp
    manifest_path = args.manifest.resolve()
    suite_path = args.suite.resolve()
    ours_full_root = args.ours_full_root.resolve()
    manifest = P.load_json(manifest_path)
    suite = P.load_json(suite_path)
    task = P.find_task(manifest, args.order)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            LEGACY_DEFAULT_OUTPUT.resolve()
            if int(task["order"]) == 32
            and task["dataset"] == "hico"
            and task["task"] == "task_hico_hugging_cat"
            else (SCRIPT_DIR / "pcp_result" / f"{P.task_key(task)}_multitarget_scores.npz").resolve()
        )
    )
    P.H.configure(task["dataset"], task["task"])
    adapter = P.H.ADAPTER
    attributes = list(P.H.ATTRS)
    target_specs = target_specifications(adapter, attributes)
    target_ids = [target_id for target_id, _ in target_specs]

    database_paths = P.B.database_paths(adapter)
    export_paths, export_indices = P.split_indices(adapter, args.split)
    _, train_indices = P.split_indices(adapter, "train_pool")
    export_indices = np.asarray(export_indices, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    full_indices = np.arange(len(database_paths), dtype=np.int64)

    learned_methods = configured_learned_methods(suite)
    tensors, seeds_by_method, score_audit = P.load_score_tensor(
        suite,
        task,
        learned_methods,
        attributes,
        database_paths,
        args.stage,
    )
    selected_seeds = list(args.seeds)

    method_ids = [
        *P.EMBEDDING_BASELINES,
        *learned_methods,
        P.OURS_FULL_ID,
    ]
    method_names = [P.DISPLAY_NAMES.get(method, method) for method in method_ids]
    if len(method_ids) != 14 or len(set(method_ids)) != len(method_ids):
        raise ValueError(
            "Expected the audited 5 embedding + 8 learned + Ours-Full methods, "
            f"found {len(method_ids)}: {method_ids}"
        )

    image_count = len(export_paths)
    method_count = len(method_ids)
    target_count = len(target_specs)
    output_shape = (image_count, method_count, target_count)
    raw_scores = np.empty(output_shape, dtype=np.float32)
    calibrated_scores = np.empty(output_shape, dtype=np.float32)
    ranks = np.empty(output_shape, dtype=np.float32)
    calibration_records: list[dict[str, Any]] = []
    embedding_sources: dict[str, Any] = {}

    method_index = {method: index for index, method in enumerate(method_ids)}
    for target_position, (target_id, target_attribute) in enumerate(target_specs):
        fixed_scores, embedding_audit = P.embedding_baseline_scores(
            adapter,
            attributes,
            target_attribute,
            full_indices,
            args.backbone,
        )
        embedding_sources[target_id] = embedding_audit

        score_rows_by_method: dict[str, np.ndarray] = {
            method: np.asarray(fixed_scores[method], dtype=np.float32)[None, :]
            for method in P.EMBEDDING_BASELINES
        }
        for method in learned_methods:
            score_rows_by_method[method] = P.method_scores_by_seed(
                tensors,
                seeds_by_method,
                method,
                selected_seeds,
                attributes,
                target_attribute,
                full_indices,
            )
        score_rows_by_method[P.OURS_FULL_ID] = P.reconstruct_ours_full_scores(
            tensors,
            seeds_by_method,
            learned_methods,
            attributes,
            target_attribute,
            selected_seeds,
            full_indices,
            task,
            ours_full_root,
            expected_stage=(args.stage if args.stage != "iterative" else None),
        )

        for method in method_ids:
            position = method_index[method]
            # Preserve the audited score precision until percentile ranking is
            # complete.  Casting learned scores to float32 here can collapse
            # near-equal values into ties; the effect is tiny but large enough
            # to make a regenerated Joint slice differ from the PCP audit
            # anchor on smaller datasets such as Stanford Cars.
            score_rows = np.asarray(score_rows_by_method[method])
            if score_rows.shape[1:] != (len(database_paths),):
                raise ValueError(
                    f"Unexpected {method}/{target_id} score shape {score_rows.shape}"
                )
            if not np.isfinite(score_rows).all():
                raise ValueError(f"{method}/{target_id} scores contain NaN or infinity")
            full_deployed = mean_deployed_score(score_rows)
            raw_scores[:, position, target_position] = full_deployed[export_indices]
            calibrated, calibration = train_pool_minmax(
                full_deployed,
                train_indices,
                export_indices,
            )
            calibrated_scores[:, position, target_position] = calibrated

            selected_rows = score_rows[:, export_indices]
            if method in P.EMBEDDING_BASELINES:
                rank = P.normalized_rank(selected_rows[0], args.tie_policy)
                rank_seed_policy = "deterministic_single_score_percentile"
            else:
                rank = P.aggregate_seed_ranks(
                    selected_rows,
                    "mean-rank",
                    args.tie_policy,
                )
                rank_seed_policy = "mean_of_seedwise_score_percentiles"
            ranks[:, position, target_position] = rank

            calibration_records.append(
                {
                    "methodId": method,
                    "method": method_names[position],
                    "targetId": target_id,
                    "fitSplit": "frozen_train_pool",
                    "application": "clip((mean_deployed_score - min) / (max - min), 0, 1)",
                    "rawSeedAggregation": (
                        "seed-independent"
                        if method in P.EMBEDDING_BASELINES
                        else f"arithmetic mean over seeds {selected_seeds}"
                    ),
                    "rankSeedPolicy": rank_seed_policy,
                    **calibration,
                }
            )

    for name, values in {
        "raw_scores": raw_scores,
        "calibrated_scores": calibrated_scores,
        "ranks": ranks,
    }.items():
        if values.shape != output_shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} array with shape {values.shape}")

    base_target_ids = target_ids[:-1]
    target_metadata = [
        {
            "id": target_id,
            "label": attribute,
            "kind": "single_attribute",
            "canonicalAttribute": attribute,
        }
        for target_id, attribute in target_specs
        if attribute is not None
    ] + [
        {
            "id": "joint",
            "label": "Joint",
            "kind": "multi_attribute_joint",
            "attributes": base_target_ids,
            "canonicalAttributes": attributes,
            "rule": "product",
        }
    ]
    retrieval_targets = [
        {
            "id": target_id,
            "label": attribute,
            "kind": "attribute",
            "canonicalAttribute": attribute,
        }
        for target_id, attribute in target_specs
        if attribute is not None
    ] + [
        {
            "id": "joint",
            "label": "Joint",
            "kind": "derived",
            "members": base_target_ids,
            "rule": "product",
        }
    ]
    method_metadata = []
    for method, display_name in zip(method_ids, method_names):
        if method in P.EMBEDDING_BASELINES:
            family = "deterministic_embedding_baseline"
            raw_definition = "method-native deployed similarity/fusion score"
        elif method == P.OURS_FULL_ID:
            family = "structured_fusion"
            raw_definition = "deployed Ours-Full attribute gate or Joint gate product"
        else:
            family = "learned_attribute_probe"
            raw_definition = (
                "cached post-sigmoid probability for a single attribute; "
                "same-seed probability product for Joint"
            )
        method_metadata.append(
            {
                "id": method,
                "name": display_name,
                "family": family,
                "rawDefinition": raw_definition,
                "jointRule": joint_rule_by_method(learned_methods, attributes)[method],
            }
        )

    metadata = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "task": {
            "order": int(args.order),
            "key": P.task_key(task),
            "dataset": task["dataset"],
            "taskName": task["task"],
            "stage": args.stage,
            "exportSplit": "gallery" if args.split == "all" else args.split,
            "calibrationFitSplit": "frozen_train_pool",
            "recordsHash": P.B.stable_hash(database_paths),
        },
        "arrayContract": {
            "layout": ["image", "method", "target"],
            "shape": list(output_shape),
            "dtype": "float32",
            "rankDirection": "1=highest, 0=lowest",
        },
        "targets": target_metadata,
        "retrievalTargets": retrieval_targets,
        "methods": method_metadata,
        "rawScore": {
            "definition": "method-native deployed score, not a common logit scale",
            "learnedCacheDefinition": (
                "ProbeBank scores.npz stores post-sigmoid probabilities, not logits"
            ),
            "seedAggregation": (
                f"arithmetic mean of deployed score over seeds {selected_seeds}; "
                "embedding baselines are seed-independent"
            ),
            "crossMethodComparability": False,
        },
        "calibration": {
            "definition": (
                "per-method, per-target min-max fitted on frozen train_pool "
                "after raw deployed-score seed averaging, clipped to [0,1]"
            ),
            "usesGroundTruth": False,
            "changesRanks": False,
            "records": calibration_records,
        },
        "rank": {
            "seedPolicy": (
                "mean of seed-wise score percentiles for learned/Ours-Full; "
                "one score percentile for deterministic embedding methods"
            ),
            "tiePolicy": args.tie_policy,
            "definition": "1=highest, 0=lowest within the exported split",
        },
        "joint": {
            "definition": "method-native deployed Joint score",
            "rulesByMethodId": joint_rule_by_method(learned_methods, attributes),
        },
        "sources": {
            "manifest": P.portable_path(manifest_path),
            "suite": P.portable_path(suite_path),
            "oursFullRoot": P.portable_path(ours_full_root),
            "learnedScoreCaches": score_audit,
            "embeddingReconstructionByTarget": embedding_sources,
        },
    }
    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    arrays = {
        "image_ids": np.asarray(export_paths, dtype=np.str_),
        "methods": np.asarray(method_names, dtype=np.str_),
        "target_ids": np.asarray(target_ids, dtype=np.str_),
        "raw_scores": raw_scores,
        "calibrated_scores": calibrated_scores,
        "ranks": ranks,
        "metadata_json": np.asarray(metadata_json, dtype=np.str_),
    }
    write_npz_atomic(output_path, arrays)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "shape": list(output_shape),
                "layout": ["image", "method", "target"],
                "methods": method_names,
                "targets": target_ids,
                "bytes": output_path.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
