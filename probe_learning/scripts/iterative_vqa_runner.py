"""Isolated, resumable iterative VQA supervision runner.

This runner deliberately does not modify the legacy ``onestage``, ``twostage``
or ``exploit300_coverage200`` layouts.  Ground truth is used only to construct
the pre-frozen benchmark split and for post-selection evaluation; candidate
scoring and acquisition never read gallery/test ground-truth labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The harness is imported rather than duplicated so its adapters, VQA parser,
# model definitions, and metrics stay identical to the one-shot experiments.
import run_retrieval_harness as H


ROUND0_SIZE = 100
REFINE_SIZE = 50
AUDIT_PER_REFINE = 10
TRAIN_PER_REFINE = 40
DEFAULT_STAGE = "iterative_vqa_100_50_v1"
MAX_ITERATIVE_LABELS = 500
BOUNDARY_MIN_PER_CLASS = 5
BOUNDARY_POS_QUANTILE = 0.10
BOUNDARY_NEG_QUANTILE = 0.90
BOUNDARY_TEMPERATURE = 0.13
FROZEN_TEST_POLICY = "stratified-seed42-minus-active-training-supervision-v1"
ISOLATED_SOURCE_POLICY = "run-local-iterative-results-only-v1"
ISOLATED_EVALUATION_POLICY = "seed42-candidate-test-ignore-task-active-manifest"
ISOLATED_IDENTITY_SCHEMA = "isolated-iterative-run-identity-v1"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _uniq(values):
    out, seen = [], set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _rank_desc(values: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(values), kind="stable")


def _take_ranked(paths, score, count, blocked):
    if count <= 0:
        return []
    selected = []
    for i in _rank_desc(score):
        rp = paths[int(i)]
        if rp not in blocked:
            selected.append(rp)
            blocked.add(rp)
            if len(selected) == count:
                break
    return selected


def _score_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "min": round(float(values.min()), 6), "mean": round(float(values.mean()), 6),
        "max": round(float(values.max()), 6),
    }


def _supervised_quantile_boundary(scores, labels, fallback_scores) -> tuple[float, dict]:
    """Estimate a robust acquisition boundary from existing VQA supervision."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    pos = scores[valid & (labels == 1)]
    neg = scores[valid & (labels == 0)]
    fallback = np.asarray(fallback_scores, dtype=float)
    fallback = fallback[np.isfinite(fallback)]

    if len(pos) >= BOUNDARY_MIN_PER_CLASS and len(neg) >= BOUNDARY_MIN_PER_CLASS:
        pos_edge = float(np.quantile(pos, BOUNDARY_POS_QUANTILE))
        neg_edge = float(np.quantile(neg, BOUNDARY_NEG_QUANTILE))
        boundary = (pos_edge + neg_edge) / 2.0
        source = "labeled_q10_positive_q90_negative_midpoint"
    else:
        pos_edge = None
        neg_edge = None
        boundary = float(np.median(fallback)) if len(fallback) else 0.5
        source = "candidate_median_fallback" if len(fallback) else "fixed_0.5_fallback"

    return boundary, {
        "value": round(boundary, 6),
        "source": source,
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "positive_q10": None if pos_edge is None else round(pos_edge, 6),
        "negative_q90": None if neg_edge is None else round(neg_edge, 6),
    }


@dataclass
class Context:
    args: object
    stage: str
    root: Path
    qa_round_root: Path
    split_root: Path
    embedding_root: Path
    report_root: Path
    adapter: object
    emb: np.ndarray
    emb_norm: np.ndarray
    p2i: dict
    train_set: list[str]
    test_set: list[str]
    gallery_paths: list[str]
    query_idx: list[int]
    input_dim: int
    gt_by_attr: dict
    cached_map: dict
    cached_prob: dict
    cache_audit: dict
    isolated: bool = False


def _all_cached_sources(adapter, stage: str) -> list[Path]:
    """Reuse historical VQA answers, but never write into their folders."""
    sources = H.discover_task_vqa_sources(adapter.task_root)
    own_split = adapter.task_root / "split" / stage
    sources += sorted(own_split.glob("*_vqa.json"))
    return _uniq([p for p in sources if p.is_file()])


def _context_cached_sources(ctx: Context) -> list[Path]:
    if not ctx.isolated:
        return _all_cached_sources(ctx.adapter, ctx.stage)
    return _isolated_iterative_sources(ctx.qa_round_root)


def _iterative_cache_source_priority(
    sources: list[Path], qa_round_root: Path,
) -> dict[str, dict]:
    """Document the deterministic source precedence used by acquisition.

    Current-stage live/manual round results outrank explicitly materialized
    ``imported_cache`` sources, which outrank task-history/legacy sources.  A
    stable path order breaks ties within a category.  This preserves historical
    cache reuse without allowing a newly discovered lower-priority archive to
    replace a committed same-stage answer.
    """

    root = qa_round_root.resolve()
    groups = {"task_history_fixed": [], "allowlisted_imported_cache": [],
              "same_stage_live_round": []}
    for source in sources:
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(root)
            parts = relative.parts
        except ValueError:
            parts = ()
        if parts and parts[0].startswith("round_"):
            category = "same_stage_live_round"
        elif "imported_cache" in parts:
            category = "allowlisted_imported_cache"
        else:
            category = "task_history_fixed"
        groups[category].append(resolved)

    base = {
        "task_history_fixed": 100000,
        "allowlisted_imported_cache": 200000,
        "same_stage_live_round": 300000,
    }
    policy = {}
    for category, paths in groups.items():
        ordered = sorted(paths, key=lambda value: str(value))
        for index, path in enumerate(ordered):
            # Historical archives are immutable fallbacks: the earliest stable
            # path wins, so appending a later-named archive cannot take over.
            # Same-stage/imported materializations use later stable paths as
            # intentional corrections and remain protected by committed hashes.
            tie_rank = (len(ordered) - index
                        if category == "task_history_fixed" else index + 1)
            policy[str(path)] = {
                "priority": base[category] + tie_rank,
                "category": category,
            }
    return policy


def _load_iterative_cache(
    sources: list[Path], adapter, qa_round_root: Path,
) -> tuple[dict, dict, dict]:
    priority = _iterative_cache_source_priority(sources, qa_round_root)
    return H.load_vqa_many_strict(
        sources, adapter.vqa_to_key, H.ATTRS, source_priority=priority,
    )


def _isolated_iterative_sources(qa_round_root: Path) -> list[Path]:
    """Return only VQA results produced below one isolated iterative run root.

    In particular, this path must never call ``discover_task_vqa_sources`` or
    inspect task-local random/two-stage folders.  Same-stage resume is supported
    by replaying result files already present below this exact ``qa`` root.
    """

    root = qa_round_root.resolve()
    sources = sorted(qa_round_root.glob("round_*/**/*_results.jsonl"))
    sources += sorted(qa_round_root.glob("round_*/**/*_vqa.json"))
    unique = _uniq([path for path in sources if path.is_file()])
    for path in unique:
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"isolated iterative VQA source escapes run root: {path}") from error
    return unique


def _stable_sequence_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_json_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _isolated_run_identity(args, adapter, stage: str, work_root: Path,
                           gallery: list[str], test: list[str], train: list[str]) -> dict:
    """Bind an isolated run to all inputs that can change acquisition/resume."""

    rows = adapter.records.sort_values("embedding_index")
    record_paths = rows["relative_path"].astype(str).tolist()
    record_indices = rows["embedding_index"].astype(int).to_numpy()
    if not np.array_equal(record_indices, np.arange(len(record_indices))):
        raise ValueError("records are not in a contiguous embedding order")
    query_indices = [int(index) for index in adapter.query_idx]
    if any(index < 0 or index >= len(record_paths) for index in query_indices):
        raise ValueError("query indices are outside the ordered database records")
    query_paths = [record_paths[index] for index in query_indices]
    return {
        "schema": ISOLATED_IDENTITY_SCHEMA,
        "source_policy": ISOLATED_SOURCE_POLICY,
        "dataset": str(H.DATASET),
        "task": str(H.TASK),
        "stage": stage,
        "backbone": str(args.backbone),
        "attributes": list(H.ATTRS),
        "work_root": str(work_root.resolve()),
        "evaluation_split_policy": ISOLATED_EVALUATION_POLICY,
        "split": {
            "seed": int(H.SPLIT_SEED),
            "test_fraction": float(H.TEST_FRAC),
            "gallery_hash": _stable_sequence_hash(gallery),
            "candidate_test_hash": _stable_sequence_hash(test),
            "train_pool_hash": _stable_sequence_hash(train),
        },
        "budget": {
            "initial": int(args.iterative_initial_labels),
            "round": int(args.iterative_round_labels),
            "train": int(args.iterative_train_per_round),
            "audit": int(args.iterative_audit_per_round),
            "max": int(args.iterative_max_labels),
            "default_stop": int(args.iterative_stop_at_labels),
            "max_refine_rounds": int(args.iterative_max_rounds),
            "adaptive": bool(args.iterative_adaptive_budget),
            "audit_min_gain": float(args.iterative_audit_min_gain),
            "exclude_provider_content_rejections": bool(getattr(
                args, "iterative_exclude_provider_content_rejections", False,
            )),
        },
        "records_hash": _stable_sequence_hash(record_paths),
        "record_count": len(record_paths),
        "query_indices": query_indices,
        "query_paths_hash": _stable_sequence_hash(query_paths),
        "vqa_config": str(args.vqa_config),
        "joint_label": args.joint_label,
    }


def _validate_isolated_resume_manifest(path: Path, *, expected_identity: dict,
                                       qa_root: Path) -> dict:
    """Fail closed unless a resume is the exact same isolated acquisition run."""

    payload = _read_json(path, {})
    stage = expected_identity.get("stage")
    if payload.get("version") != stage:
        raise RuntimeError(
            f"isolated iterative resume stage mismatch: {payload.get('version')!r} != {stage!r}"
        )
    if payload.get("source_policy") != ISOLATED_SOURCE_POLICY:
        raise RuntimeError(
            "isolated iterative resume lacks the run-local-only VQA source contract; "
            "use a new --iterative-work-root"
        )
    observed_identity = payload.get("identity")
    if not isinstance(observed_identity, dict):
        raise RuntimeError(
            "isolated iterative resume lacks a bound run identity; "
            "use a new --iterative-work-root"
        )
    if observed_identity != expected_identity:
        changed = sorted(
            key for key in set(observed_identity) | set(expected_identity)
            if observed_identity.get(key) != expected_identity.get(key)
        )
        raise RuntimeError(
            "isolated iterative resume identity mismatch for "
            f"{', '.join(changed) or 'unknown fields'}; use a new --iterative-work-root"
        )
    expected_hash = _stable_json_hash(expected_identity)
    if payload.get("run_identity_sha256") != expected_hash:
        raise RuntimeError(
            "isolated iterative resume manifest has an invalid run identity hash; "
            "use a new --iterative-work-root"
        )
    root = qa_root.resolve()
    for raw_source in payload.get("vqa_sources", []):
        source = Path(raw_source).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"isolated iterative manifest references an external VQA source: {source}"
            ) from error
    return payload


def _validate_or_initialize_isolated_work_root(
    work_root: Path, expected_identity: dict,
) -> dict | None:
    """Validate an exact resume or reject an orphaned/pre-contract work root.

    Empty pre-created directories are harmless.  Any file without the root
    manifest is ambiguous state and must never be adopted by a new run.
    """

    work_root = work_root.resolve()
    qa_root = work_root / "qa"
    manifest_path = qa_root / "manifest.json"
    if manifest_path.is_file():
        return _validate_isolated_resume_manifest(
            manifest_path, expected_identity=expected_identity, qa_root=qa_root,
        )
    if work_root.exists():
        orphaned = sorted(
            path for path in work_root.rglob("*")
            if path.is_file() or path.is_symlink()
        )
        if orphaned:
            preview = ", ".join(str(path) for path in orphaned[:3])
            raise RuntimeError(
                "isolated iterative work root has files but no qa/manifest.json; "
                f"refusing to adopt orphaned state ({preview}). Use a new work root."
            )
    return None


def _candidate_frozen_split(adapter, p2i, gt_by_attr):
    """Return the deterministic pre-exclusion seed-42 split."""

    # ``run_retrieval_harness.configure`` stores the most recently configured
    # task in module globals.  The Web tuning server is multi-user, so another
    # task can replace ``H.ATTRS`` while this split is being resolved.  The
    # labels passed to this function are the authoritative task-local schema.
    attributes = tuple(gt_by_attr)
    if not attributes:
        raise ValueError("Frozen Test split requires at least one modeled attribute")
    i2p = {int(i): rp for rp, i in p2i.items()}
    query_rp = {i2p[int(i)] for i in adapter.query_idx if int(i) in i2p}
    # Gallery is the full searchable database. Query and acquisition-train
    # images remain visible there; only the frozen Test split is held out from
    # the supervision pool.
    gallery = adapter.records["relative_path"].astype(str).tolist()
    evaluation_pool = [rp for rp in gallery if rp not in query_rp]
    rng = np.random.default_rng(H.SPLIT_SEED)
    buckets = {}
    for rp in evaluation_pool:
        buckets.setdefault(
            tuple(int(gt_by_attr[attribute].get(rp, 0)) for attribute in attributes),
            [],
        ).append(rp)
    test = []
    for items in buckets.values():
        items = sorted(items)
        rng.shuffle(items)
        test.extend(items[:int(round(H.TEST_FRAC * len(items)))])
    test = sorted(test)
    test_lookup = set(test)
    return gallery, test, sorted(rp for rp in evaluation_pool if rp not in test_lookup)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_task_file(task_root: Path, relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label} must be a non-empty task-relative path")
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError(f"{label} must not be absolute: {requested}")
    root = task_root.resolve()
    resolved = (root / requested).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the task root: {requested}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _load_frozen_test_ids(
    adapter,
    *,
    gallery: list[str],
    candidate_test: list[str],
    query_rp: set[str],
) -> list[str] | None:
    """Load and audit a persisted clean Test split, if this task freezes one."""

    task_root = Path(adapter.task_root).resolve()
    manifest_path = task_root / "evaluation" / "manifest.json"
    manifest = _read_json(manifest_path, {})
    if manifest.get("status") != "frozen":
        return None
    if manifest.get("gallery") != "full_database":
        raise ValueError(f"Frozen Test must use the full database: {manifest_path}")
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("id") != FROZEN_TEST_POLICY:
        raise ValueError(f"Unsupported Frozen Test policy: {manifest_path}")
    expected_policy = {
        "candidate_split_seed": int(H.SPLIT_SEED),
        "candidate_test_fraction": float(H.TEST_FRAC),
        "stratified_by": "task_attribute_gt_tuple",
        "query_excluded": True,
        "no_replacement": True,
        "excluded_rows_destination": "development",
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(
                f"Frozen Test policy field {key!r} drifted in {manifest_path}: "
                f"{policy.get(key)!r} != {expected!r}"
            )

    test_path = _resolved_task_file(
        task_root,
        manifest.get("test_ids_file"),
        label="Frozen Test IDs file",
    )
    if manifest.get("test_ids_sha256") != _sha256_file(test_path):
        raise ValueError(f"Frozen Test ID checksum drifted: {test_path}")
    test_ids = _read_json(test_path, None)
    if not isinstance(test_ids, list) or not all(isinstance(value, str) for value in test_ids):
        raise ValueError(f"Frozen Test IDs must be a JSON string list: {test_path}")
    if test_ids != sorted(test_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError(f"Frozen Test IDs must be sorted and unique: {test_path}")

    gallery_ids = set(gallery)
    candidate_ids = set(candidate_test)
    frozen_ids = set(test_ids)
    unknown = sorted(frozen_ids.difference(gallery_ids))
    if unknown:
        raise ValueError(f"Frozen Test contains unknown gallery IDs: {unknown[:3]}")
    query_overlap = sorted(frozen_ids.intersection(query_rp))
    if query_overlap:
        raise ValueError(f"Frozen Test contains Query IDs: {query_overlap[:3]}")
    additions = sorted(frozen_ids.difference(candidate_ids))
    if additions:
        raise ValueError(f"Frozen Test adds rows outside the seed-42 candidate: {additions[:3]}")

    supervision = manifest.get("active_training_supervision")
    if not isinstance(supervision, dict):
        raise ValueError(f"Frozen Test training provenance is missing: {manifest_path}")
    supervision_path = _resolved_task_file(
        task_root,
        supervision.get("ids_file"),
        label="Active training supervision IDs file",
    )
    if supervision.get("sha256") != _sha256_file(supervision_path):
        raise ValueError(f"Active training supervision checksum drifted: {supervision_path}")
    training_ids = _read_json(supervision_path, None)
    if not isinstance(training_ids, list) or not all(
        isinstance(value, str) for value in training_ids
    ):
        raise ValueError(
            f"Active training supervision IDs must be a JSON string list: {supervision_path}"
        )
    if len(set(training_ids)) != len(training_ids):
        raise ValueError(f"Active training supervision contains duplicate IDs: {supervision_path}")
    if int(supervision.get("row_count", -1)) != len(training_ids):
        raise ValueError(f"Active training supervision count drifted: {manifest_path}")
    if not isinstance(supervision.get("stage"), str) or not supervision.get("stage"):
        raise ValueError(f"Active training supervision stage is missing: {manifest_path}")
    unknown_training = sorted(set(training_ids).difference(gallery_ids))
    if unknown_training:
        raise ValueError(
            "Active training supervision contains unknown gallery IDs: "
            f"{unknown_training[:3]}"
        )

    expected_frozen = candidate_ids.difference(training_ids)
    if frozen_ids != expected_frozen:
        missing = sorted(expected_frozen.difference(frozen_ids))
        unexpected = sorted(frozen_ids.difference(expected_frozen))
        raise ValueError(
            "Frozen Test is not the candidate Test minus active training supervision: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    removed_ids = sorted(candidate_ids.difference(frozen_ids))
    excluded_path = _resolved_task_file(
        task_root,
        manifest.get("excluded_test_ids_file"),
        label="Excluded Test IDs audit file",
    )
    if manifest.get("excluded_test_ids_sha256") != _sha256_file(excluded_path):
        raise ValueError(f"Excluded Test ID checksum drifted: {excluded_path}")
    if _read_json(excluded_path, None) != removed_ids:
        raise ValueError(f"Excluded Test ID audit drifted: {excluded_path}")
    if int(manifest.get("candidate_test_row_count", -1)) != len(candidate_ids):
        raise ValueError(f"Candidate Test count drifted: {manifest_path}")
    candidate_hash = hashlib.sha256(
        "\n".join(sorted(candidate_ids)).encode("utf-8")
    ).hexdigest()
    if manifest.get("candidate_test_ids_sha256") != candidate_hash:
        raise ValueError(f"Candidate Test checksum drifted: {manifest_path}")
    if int(manifest.get("test_row_count", -1)) != len(frozen_ids):
        raise ValueError(f"Frozen Test count drifted: {manifest_path}")
    if int(manifest.get("excluded_training_row_count", -1)) != len(removed_ids):
        raise ValueError(f"Excluded Test count drifted: {manifest_path}")
    if int(supervision.get("query_overlap_count", -1)) != len(
        set(training_ids).intersection(query_rp)
    ):
        raise ValueError(f"Active training Query-overlap audit drifted: {manifest_path}")
    return test_ids


def _frozen_split(
    adapter,
    emb_norm,
    p2i,
    gt_by_attr,
    *,
    use_persisted_evaluation: bool = True,
):
    gallery, candidate_test, _ = _candidate_frozen_split(adapter, p2i, gt_by_attr)
    i2p = {int(i): rp for rp, i in p2i.items()}
    query_rp = {i2p[int(i)] for i in adapter.query_idx if int(i) in i2p}
    persisted_test = None
    if use_persisted_evaluation:
        persisted_test = _load_frozen_test_ids(
            adapter,
            gallery=gallery,
            candidate_test=candidate_test,
            query_rp=query_rp,
        )
    test = candidate_test if persisted_test is None else persisted_test
    test_lookup = set(test)
    train = sorted(
        relative_path
        for relative_path in gallery
        if relative_path not in query_rp and relative_path not in test_lookup
    )
    return gallery, test, train


def _label_stats(vqa_map: dict, paths: list[str]) -> dict:
    rows = [
        vqa_map[p] for p in paths
        if H.is_complete_binary_labels(vqa_map.get(p), H.ATTRS)
    ]
    attr_pos = {a: int(sum(int(r[a]) for r in rows)) for a in H.ATTRS}
    joint_pos = int(sum(all(int(r[a]) == 1 for a in H.ATTRS) for r in rows))
    return {"n": len(rows), "attribute_positive": attr_pos, "joint_positive": joint_pos}


def _strict_cached_paths(vqa_map: dict, paths) -> list[str]:
    return [
        path for path in paths
        if H.is_complete_binary_labels(vqa_map.get(path), H.ATTRS)
    ]


def _selected_cache_snapshot(paths, vqa_map: dict, cache_audit: dict) -> dict:
    """Bind selected labels to the exact complete sources that supplied them."""

    selected = sorted(_uniq(list(paths)))
    missing = [
        path for path in selected
        if not H.is_complete_binary_labels(vqa_map.get(path), H.ATTRS)
    ]
    conflicts = sorted(set(selected).intersection(cache_audit.get("conflicts", {})))
    if missing or conflicts:
        raise ValueError(
            "cannot snapshot incomplete/conflicted VQA selection: "
            f"missing={missing[:3]}, conflicts={conflicts[:3]}"
        )

    labels = [
        {
            "image": path,
            "labels": {attribute: int(vqa_map[path][attribute]) for attribute in H.ATTRS},
        }
        for path in selected
    ]
    provenance = []
    contributing_sources = {}
    for path in selected:
        records = list(cache_audit.get("provenance_by_image", {}).get(path, []))
        if not records:
            raise ValueError(f"strict VQA provenance is missing for selected image: {path}")
        normalized_records = []
        for record in records:
            normalized = {
                key: record.get(key)
                for key in (
                    "source_path", "source_sha256", "row", "raw_image", "label_sha256",
                )
            }
            normalized_records.append(normalized)
            source_key = (normalized["source_path"], normalized["source_sha256"])
            contributing_sources[source_key] = {
                "path": normalized["source_path"],
                "sha256": normalized["source_sha256"],
            }
        provenance.append({
            "image": path,
            "records": sorted(
                normalized_records,
                key=lambda row: (
                    str(row.get("source_path")), int(row.get("row") or -1),
                    str(row.get("raw_image")),
                ),
            ),
        })
    source_set = sorted(
        contributing_sources.values(), key=lambda row: (row["path"], row["sha256"]),
    )
    return {
        "schema": "iterative-selected-vqa-cache-snapshot-v1",
        "attributes": list(H.ATTRS),
        "selected_count": len(selected),
        "selected_labels_sha256": _stable_json_hash({
            "attributes": list(H.ATTRS), "labels": labels,
        }),
        "source_set_sha256": _stable_json_hash({"sources": source_set}),
        "selected_label_provenance_sha256": _stable_json_hash({
            "attributes": list(H.ATTRS), "provenance": provenance,
        }),
        "cache_catalog_sha256": cache_audit.get("source_set_sha256"),
        "sources": source_set,
    }


def _committed_cache_audit(state: dict, vqa_map: dict, cache_audit: dict) -> dict:
    """Audit every committed round and stage deterministic legacy migrations.

    The returned report is read-only with respect to ``state``.  Callers may
    apply ``migrations`` only when ``ok`` is true, so an invalid later round can
    never partially rewrite an older trajectory.
    """

    invalid_rounds = []
    migrations = []
    committed = set()
    for round_index, record in enumerate(state.get("rounds", [])):
        round_no = int(record.get("round", round_index))
        selected = _uniq(list(record.get("train", [])) + list(record.get("audit", [])))
        committed.update(selected)
        incomplete = [
            path for path in selected
            if not H.is_complete_binary_labels(vqa_map.get(path), H.ATTRS)
        ]
        conflicts = sorted(set(selected).intersection(cache_audit.get("conflicts", {})))
        if incomplete or conflicts:
            invalid_rounds.append({
                "round": round_no,
                "incomplete_or_missing": sorted(incomplete),
                "conflicts": conflicts,
            })
            continue
        try:
            observed = _selected_cache_snapshot(selected, vqa_map, cache_audit)
        except ValueError as error:
            invalid_rounds.append({
                "round": round_no,
                "snapshot_error": str(error),
            })
            continue
        expected = record.get("cache_snapshot")
        if expected is None:
            migrations.append({
                "round": round_no,
                "round_index": round_index,
                "cache_snapshot": observed,
                "migration": {
                    "schema": "legacy-complete-conflict-free-cache-migration-v1",
                    "reason": "legacy_round_lacked_strict_cache_hashes",
                },
            })
            continue
        drift = {
            field: {"expected": expected.get(field), "observed": observed.get(field)}
            for field in (
                "selected_labels_sha256", "source_set_sha256",
                "selected_label_provenance_sha256",
            )
            if expected.get(field) != observed.get(field)
        }
        if drift:
            invalid_rounds.append({
                "round": round_no,
                "cache_snapshot_drift": drift,
            })

    uncommitted_conflicts = sorted(
        set(cache_audit.get("conflicts", {})).difference(committed)
    )
    earliest = min(
        (int(item["round"]) for item in invalid_rounds), default=None,
    )
    ok = not invalid_rounds and not uncommitted_conflicts
    return {
        "schema": "iterative-committed-cache-audit-v1",
        "ok": ok,
        "earliest_invalid_committed_round": earliest,
        "invalid_committed_rounds": invalid_rounds,
        "uncommitted_conflicts": uncommitted_conflicts,
        "global_conflict_count": len(cache_audit.get("conflicts", {})),
        "shadowed_conflict_count": len(cache_audit.get("shadowed_conflicts", {})),
        "strict_complete_image_count": int(cache_audit.get("complete_image_count", 0)),
        "cache_source_set_sha256": cache_audit.get("source_set_sha256"),
        "migrations": migrations if ok else [],
    }


def _apply_cache_audit_migrations(state: dict, report: dict) -> None:
    if not report.get("ok"):
        raise ValueError("refusing to migrate an invalid iterative cache state")
    for migration in report.get("migrations", []):
        record = state["rounds"][int(migration["round_index"])]
        record["cache_snapshot"] = migration["cache_snapshot"]
        record["cache_snapshot_migration"] = migration["migration"]
    state["cache_audit"] = {
        key: report.get(key)
        for key in (
            "schema", "cache_source_set_sha256", "strict_complete_image_count",
            "global_conflict_count", "shadowed_conflict_count",
        )
    }


def _round_paths(ctx: Context, round_no: int):
    name = f"round_{round_no:02d}"
    return {
        "qa": ctx.qa_round_root / name,
        "run": ctx.embedding_root / name,
        "manifest": ctx.qa_round_root / name / "manifest.json",
        "state": ctx.split_root / "round_state.json",
    }


_DATA_INSPECTION_FAILED_FIELD = re.compile(
    r"[\"'](?:code|type)[\"']\s*:\s*[\"']data_inspection_failed[\"']"
)


def _is_exact_data_inspection_failure(row: dict) -> bool:
    """Recognize the provider's terminal inspection code, not generic failures."""

    if row.get("error_code") == "data_inspection_failed":
        return True
    error = row.get("error")
    if isinstance(error, dict):
        if error.get("code") == "data_inspection_failed" or error.get("type") == "data_inspection_failed":
            return True
        nested = error.get("error")
        if isinstance(nested, dict) and (
            nested.get("code") == "data_inspection_failed"
            or nested.get("type") == "data_inspection_failed"
        ):
            return True
        return False
    text = str(error or "").strip()
    return text == "data_inspection_failed" or bool(
        _DATA_INSPECTION_FAILED_FIELD.search(text)
    )


def _provider_content_rejections(ctx: Context) -> dict[str, list[dict]]:
    """Return unresolved terminal provider rejections from this stage only.

    The result is keyed by the adapter's canonical logical image ID.  Evidence
    deliberately records hashes and source rows rather than inventing a label;
    complete strict cache entries are resolved and therefore omitted.
    """

    if not bool(getattr(
        getattr(ctx, "args", None),
        "iterative_exclude_provider_content_rejections",
        False,
    )):
        return {}

    root = ctx.qa_round_root.resolve()
    sources = sorted(ctx.qa_round_root.glob("round_*/**/*_results.jsonl"))
    rejected: dict[str, list[dict]] = {}
    for source in sources:
        if not source.is_file():
            continue
        resolved = source.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"provider rejection source escapes iterative stage root: {source}"
            ) from error
        source_sha256 = _sha256_file(resolved)
        with resolved.open("r", encoding="utf-8", errors="replace") as handle:
            for row_no, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("answer") != "[FAILED]":
                    continue
                if not _is_exact_data_inspection_failure(row):
                    continue
                try:
                    image = ctx.adapter.vqa_to_key(str(row["image"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if H.is_complete_binary_labels(ctx.cached_map.get(image), H.ATTRS):
                    continue
                error_text = str(row.get("error") or "")
                evidence = {
                    "image": image,
                    "error_code": "data_inspection_failed",
                    "source_path": str(resolved),
                    "source_sha256": source_sha256,
                    "source_row": int(row_no),
                    "error_sha256": hashlib.sha256(
                        error_text.encode("utf-8")
                    ).hexdigest(),
                    "model": row.get("model"),
                    "timestamp": row.get("timestamp"),
                }
                rejected.setdefault(image, []).append(evidence)
    for image, evidence_rows in rejected.items():
        unique = {
            _stable_json_hash(row): row for row in evidence_rows
        }
        rejected[image] = [unique[key] for key in sorted(unique)]
    return dict(sorted(rejected.items()))


def _pending_selection_contract_sha256(pick: dict, diagnostics: dict) -> str:
    roles = {
        str(role): list(map(str, paths or []))
        for role, paths in sorted((pick.get("roles") or {}).items())
    }
    role_by_image = {
        image: role for role, paths in roles.items() for image in paths
    }
    backfill_reasons = diagnostics.get("backfill_reasons") or {}
    shared_reason = diagnostics.get("backfill_reason")
    metadata = {}
    train = list(map(str, pick.get("train") or []))
    audit = list(map(str, pick.get("audit") or []))
    for image in train + audit:
        role = role_by_image.get(image)
        reason = None
        if role is None:
            if isinstance(backfill_reasons, dict):
                reason = backfill_reasons.get(image)
            if reason is None and isinstance(shared_reason, str):
                reason = shared_reason
        metadata[image] = {
            "partition": "train" if image in set(train) else "audit",
            "acquisition_role": role or "backfill",
            "backfill_reason": reason,
        }
    return _stable_json_hash({
        "train": train,
        "audit": audit,
        "roles": roles,
        "selection_metadata": {
            image: metadata[image] for image in sorted(metadata)
        },
    })


def _validate_frozen_selection(
    ctx: Context,
    round_no: int,
    pick: dict,
    available: list[str],
    *,
    allowed_terminal_rejections=(),
) -> None:
    """Fail closed on quota, overlap, role, or candidate-pool drift."""

    train = list(map(str, pick.get("train") or []))
    audit = list(map(str, pick.get("audit") or []))
    expected = (ROUND0_SIZE, 0) if int(round_no) == 0 else (
        TRAIN_PER_REFINE, AUDIT_PER_REFINE,
    )
    if (len(train), len(audit)) != expected:
        raise RuntimeError(
            f"round {round_no} selection quota drifted: "
            f"observed={(len(train), len(audit))}, expected={expected}"
        )
    selected = train + audit
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"round {round_no} selection has duplicates or train/audit overlap")
    train_pool = set(map(str, getattr(ctx, "train_set", available)))
    outside_train_pool = sorted(set(selected).difference(train_pool))
    if outside_train_pool:
        raise RuntimeError(
            f"round {round_no} selection escapes the frozen train pool: "
            f"{outside_train_pool[:3]}"
        )
    contract_pool = set(map(str, available)) | set(map(str, allowed_terminal_rejections))
    outside_contract = sorted(set(selected).difference(contract_pool))
    if outside_contract:
        raise RuntimeError(
            f"round {round_no} selection escapes the available candidate contract: "
            f"{outside_contract[:3]}"
        )
    role_images = [
        str(image)
        for images in (pick.get("roles") or {}).values()
        for image in images or []
    ]
    if len(role_images) != len(set(role_images)):
        raise RuntimeError(f"round {round_no} selection assigns multiple roles to one image")
    unknown_role_images = sorted(set(role_images).difference(selected))
    if unknown_role_images:
        raise RuntimeError(
            f"round {round_no} roles contain unselected images: {unknown_role_images[:3]}"
        )


def _refresh_rejected_frozen_selection(
    ctx: Context,
    state: dict,
    round_no: int,
    pick: dict,
    diagnostics: dict,
    available: list[str],
    rejected: dict[str, list[dict]],
) -> tuple[dict, dict, dict]:
    """Replace rejected rows in one uncommitted frozen round, deterministically.

    Existing successful candidates retain their partition, role, and position.
    A replacement comes from the same policy/role whenever possible; only a
    deterministic partition backfill is allowed as a last resort.  No label is
    fabricated and an underfilled round is never returned.
    """

    original_pick = {
        "train": list(map(str, pick.get("train") or [])),
        "audit": list(map(str, pick.get("audit") or [])),
        "roles": {
            str(role): list(map(str, paths or []))
            for role, paths in (pick.get("roles") or {}).items()
        },
    }
    original_diag = json.loads(json.dumps(diagnostics, ensure_ascii=False))
    before_hash = _pending_selection_contract_sha256(original_pick, original_diag)
    selected = _uniq(original_pick["train"] + original_pick["audit"])
    selected_rejections = [image for image in selected if image in rejected]
    _validate_frozen_selection(
        ctx, round_no, original_pick, available,
        allowed_terminal_rejections=selected_rejections,
    )
    if not selected_rejections:
        return original_pick, original_diag, {
            "changed": False,
            "selection_contract_before_sha256": before_hash,
            "selection_contract_after_sha256": before_hash,
            "rejected_not_budgeted": [],
            "replacement_map": {},
            "selection_before": original_pick,
            "selection_after": original_pick,
        }

    pending_round = state.get("pending_round")
    if pending_round is not None and int(pending_round) != int(round_no):
        raise RuntimeError(
            "content-filter backfill is allowed only for state.pending_round"
        )
    if pending_round is None and int(round_no) != len(state.get("rounds") or []):
        raise RuntimeError(
            "content-filter backfill cannot identify a new uncommitted pending round"
        )
    committed = {
        str(image)
        for record in state.get("rounds") or []
        for partition in ("train", "audit")
        for image in record.get(partition, [])
    }
    committed_rejections = committed.intersection(rejected)
    if committed_rejections:
        raise RuntimeError(
            "content-filter rejection intersects committed acquisition state: "
            f"{sorted(committed_rejections)[:3]}"
        )

    role_by_image = {}
    for role, images in original_pick["roles"].items():
        for image in images:
            if image in role_by_image:
                raise RuntimeError(f"selected image has multiple acquisition roles: {image}")
            role_by_image[image] = role

    excluded = set(rejected) | committed
    eligible = [
        image for image in _uniq(map(str, available))
        if image not in excluded
    ]
    committed_train = _uniq([
        str(image)
        for record in state.get("rounds") or []
        for image in record.get("train", [])
    ])
    if round_no == 0:
        proposal, proposal_diag = _round0_selection(ctx, eligible)
    else:
        proposal, proposal_diag = _role_selection(
            ctx, eligible, committed_train, ctx.cached_map,
        )

    survivors = set(selected).difference(selected_rejections)
    blocked = set(survivors) | excluded
    new_pick = {
        "train": list(original_pick["train"]),
        "audit": list(original_pick["audit"]),
        "roles": {role: list(images) for role, images in original_pick["roles"].items()},
    }
    replacement_map: dict[str, str] = {}
    replacement_details = {}
    backfill_reasons = dict(original_diag.get("backfill_reasons") or {})

    for old_image in selected_rejections:
        partition = "train" if old_image in original_pick["train"] else "audit"
        old_role = role_by_image.get(old_image)
        same_role = list((proposal.get("roles") or {}).get(old_role, [])) if old_role else []
        partition_proposal = list(proposal.get(partition) or [])
        choices = _uniq(same_role + partition_proposal + eligible)
        replacement = next((image for image in choices if image not in blocked), None)
        if replacement is None:
            raise RuntimeError(
                "candidate pool cannot backfill terminal provider rejection without "
                f"underfilling {partition}: {old_image}"
            )
        role_preserved = bool(old_role and replacement in same_role)
        slot = new_pick[partition].index(old_image)
        new_pick[partition][slot] = replacement
        if old_role:
            role_slot = new_pick["roles"][old_role].index(old_image)
            if role_preserved:
                new_pick["roles"][old_role][role_slot] = replacement
            else:
                new_pick["roles"][old_role].pop(role_slot)
        reason = f"provider_data_inspection_failed_replacement_for:{old_image}"
        if not role_preserved:
            backfill_reasons[replacement] = reason
        replacement_map[old_image] = replacement
        replacement_details[old_image] = {
            "replacement": replacement,
            "partition": partition,
            "original_role": old_role,
            "role_preserved": role_preserved,
            "reason": reason,
        }
        blocked.add(replacement)

    if len(new_pick["train"]) != len(original_pick["train"]):
        raise RuntimeError("terminal rejection backfill changed the train quota")
    if len(new_pick["audit"]) != len(original_pick["audit"]):
        raise RuntimeError("terminal rejection backfill changed the audit quota")
    refreshed = new_pick["train"] + new_pick["audit"]
    if len(refreshed) != len(set(refreshed)):
        raise RuntimeError("terminal rejection backfill produced duplicate selections")
    if survivors.difference(refreshed):
        raise RuntimeError("terminal rejection backfill changed successful frozen candidates")
    _validate_frozen_selection(ctx, round_no, new_pick, available)

    new_diag = original_diag
    new_diag["backfill_reasons"] = backfill_reasons
    new_diag["selection_revision"] = int(new_diag.get("selection_revision", 0)) + 1
    new_diag["selection_revision_reason"] = "provider_data_inspection_failed"
    proposal_scores = proposal_diag.get("selected_scores") or {}
    selected_scores = dict(new_diag.get("selected_scores") or {})
    for old_image, replacement in replacement_map.items():
        selected_scores.pop(old_image, None)
        if replacement in proposal_scores:
            selected_scores[replacement] = proposal_scores[replacement]
    if selected_scores:
        new_diag["selected_scores"] = selected_scores
    after_hash = _pending_selection_contract_sha256(new_pick, new_diag)
    audit = {
        "schema": "iterative-provider-content-rejection-backfill-v1",
        "changed": True,
        "round": int(round_no),
        "selection_contract_before_sha256": before_hash,
        "selection_contract_after_sha256": after_hash,
        "rejected_not_budgeted": sorted(selected_rejections),
        "replacement_map": replacement_map,
        "selection_before": original_pick,
        "selection_after": new_pick,
        "replacement_details": replacement_details,
        "rejection_evidence": {
            image: rejected[image] for image in sorted(selected_rejections)
        },
        "quota_before": {
            "train": len(original_pick["train"]), "audit": len(original_pick["audit"]),
        },
        "quota_after": {
            "train": len(new_pick["train"]), "audit": len(new_pick["audit"]),
        },
        "successful_candidates_preserved": True,
        "requires_new_preflight_before_network": True,
    }
    return new_pick, new_diag, audit


def _persist_rejected_pending_selection(
    ctx: Context,
    state: dict,
    rpaths: dict,
    round_no: int,
    previous_manifest: dict,
    pick: dict,
    diagnostics: dict,
    audit: dict,
    selected_train: list[str],
    selected_audit: list[str],
) -> bool:
    """Atomically refreeze a rejected pending selection and stop before VQA."""

    candidates, missing, complete = _round_label_status(pick, ctx.cached_map)
    reused = _strict_cached_paths(ctx.cached_map, candidates)
    prequential = (
        _prequential_predictions(ctx, pick["audit"], selected_train, ctx.cached_map)
        if round_no else {"n": 0}
    )
    audit = dict(audit)
    audit["previous_manifest_sha256"] = (
        _sha256_file(rpaths["manifest"]) if rpaths["manifest"].is_file() else None
    )
    candidate_pool_path = rpaths["qa"] / "candidate_pool.json"
    if not candidate_pool_path.is_file():
        raise RuntimeError("pending rejection backfill lacks a frozen candidate_pool.json")
    audit["candidate_pool_sha256"] = _sha256_file(candidate_pool_path)
    superseded = list(map(
        str, previous_manifest.get("requested_to_label") or [],
    ))
    audit["superseded_requested_to_label"] = superseded
    provider_request_history = list(
        previous_manifest.get("provider_request_history") or []
    )
    if superseded:
        validation = _read_json(
            rpaths["qa"] / "outbound_preflight_validation.json", {},
        )
        legacy_event = {
            "round": int(round_no),
            "requested": superseded,
            "request_count": len(superseded),
            "selection_contract_sha256": audit[
                "selection_contract_before_sha256"
            ],
            "preflight_manifest_sha256": validation.get("manifest_sha256"),
            "status": "superseded_by_provider_content_rejection",
        }
        event_hash = _stable_json_hash(legacy_event)
        if not any(
            int(row.get("round", -1)) == int(round_no)
            and row.get("selection_contract_sha256")
            == audit["selection_contract_before_sha256"]
            and set(map(str, row.get("requested") or [])) == set(superseded)
            for row in provider_request_history
        ):
            legacy_event["event_sha256"] = event_hash
            provider_request_history.append(legacy_event)
    audit["provider_request_history"] = provider_request_history
    history = list(previous_manifest.get("content_rejection_audits") or [])
    if not any(
        row.get("selection_contract_after_sha256")
        == audit["selection_contract_after_sha256"]
        for row in history
    ):
        history.append(audit)
    rejected_not_budgeted = sorted(set(
        map(str, previous_manifest.get("rejected_not_budgeted") or [])
    ) | set(audit["rejected_not_budgeted"]))
    replacement_map = dict(previous_manifest.get("replacement_map") or {})
    replacement_map.update(audit["replacement_map"])
    manifest = {
        **previous_manifest,
        "round": int(round_no),
        "stage": ctx.stage,
        "train": pick["train"],
        "audit": pick["audit"],
        "roles": pick["roles"],
        "reused": reused,
        "to_label": missing,
        # Replacements have not been sent.  The old approval cannot authorize
        # them, even when the provider invocation that exposed the rejection
        # happened earlier in this process.
        "requested_to_label": [],
        "prequential_audit": prequential,
        "diagnostics": diagnostics,
        "complete": complete,
        "cache_snapshot": None,
        "selection_revision": int(previous_manifest.get("selection_revision", 0)) + 1,
        "selection_contract_before_sha256": audit[
            "selection_contract_before_sha256"
        ],
        "selection_contract_after_sha256": audit[
            "selection_contract_after_sha256"
        ],
        "rejected_not_budgeted": rejected_not_budgeted,
        "replacement_map": replacement_map,
        "content_rejection_audits": history,
        "provider_request_history": provider_request_history,
        "provider_requested_unique": sorted({
            image
            for event in provider_request_history
            for image in event.get("requested", [])
        }),
        "provider_request_count": sum(
            int(event.get("request_count", len(event.get("requested", []))))
            for event in provider_request_history
        ),
        "requires_new_preflight_before_network": bool(missing),
    }
    _write_json(rpaths["manifest"], manifest)
    _write_json(rpaths["qa"] / "selected_train.json", pick["train"])
    _write_json(rpaths["qa"] / "selected_audit.json", pick["audit"])
    _write_json(rpaths["qa"] / "to_label.json", sorted(missing))
    _write_json(rpaths["qa"] / "reused_labels.json", sorted(reused))
    _write_json(rpaths["qa"] / "provider_content_rejection_audit.json", {
        "schema": "iterative-provider-content-rejection-ledger-v1",
        "round": int(round_no),
        "audits": history,
    })

    existing_rows = _read_json(rpaths["qa"] / "selection_diagnostics.json", [])
    existing_by_image = {
        str(row.get("image")): row for row in existing_rows
        if isinstance(row, dict) and row.get("image")
    }
    role_lookup = {
        image: role for role, paths in pick["roles"].items() for image in paths
    }
    selected_scores = diagnostics.get("selected_scores") or {}
    rows = []
    for image in candidates:
        row = dict(existing_by_image.get(image) or {})
        score = selected_scores.get(image) or {}
        row.update({
            "image": image,
            "candidate_type": role_lookup.get(image, "backfill"),
            "reused_vqa": image in reused,
            "partition": "train" if image in pick["train"] else "audit",
            "query_similarity": score.get("query_similarity"),
            "attribute_scores": score.get("attribute_scores"),
            "joint_score": score.get("joint_score"),
            "joint_boundary_distance": score.get("joint_boundary_distance"),
            "joint_boundary_score": score.get("joint_boundary_score"),
            "committee_disagreement": score.get("committee_disagreement"),
            "mmr_score": None,
            "labels": ctx.cached_map.get(image),
            "seed": H.SPLIT_SEED,
        })
        rows.append(row)
    _write_json(rpaths["qa"] / "selection_diagnostics.json", rows)

    state["status"] = "ready_to_commit_refrozen" if complete else "awaiting_vqa"
    state["pending_round"] = int(round_no)
    if missing:
        state["requires_repreflight"] = True
        state["pending_selection_contract_sha256"] = audit[
            "selection_contract_after_sha256"
        ]
    else:
        state.pop("requires_repreflight", None)
        state.pop("pending_selection_contract_sha256", None)
    state["provider_rejected_not_budgeted"] = sorted(set(
        map(str, state.get("provider_rejected_not_budgeted") or [])
    ) | set(audit["rejected_not_budgeted"]))
    _write_json(rpaths["state"], state)
    _write_json(ctx.split_root / "train_labeled_indices.json", selected_train)
    _write_json(ctx.split_root / "audit_labeled_indices.json", selected_audit)
    _write_report(ctx, state)
    return complete


def _append_provider_request_event(
    ctx: Context,
    history: list[dict],
    round_no: int,
    requested_paths: list[str],
    selection_contract_sha256: str,
    out_dir: Path,
) -> list[dict]:
    """Append one auditable provider invocation without losing earlier retries."""

    requested = _uniq(map(str, requested_paths))
    if not requested:
        return list(history)
    result_sources = [
        {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for path in sorted(out_dir.glob("*_results.jsonl"))
        if path.is_file()
    ]
    event = {
        "round": int(round_no),
        "requested": requested,
        "request_count": len(requested),
        "selection_contract_sha256": selection_contract_sha256,
        "preflight_manifest_sha256": getattr(
            ctx.args, "vqa_preflight_sha256", None,
        ),
        "provider_result_sources": result_sources,
        "status": "provider_invoked",
    }
    event["event_sha256"] = _stable_json_hash(event)
    updated = list(history)
    if not any(row.get("event_sha256") == event["event_sha256"] for row in updated):
        updated.append(event)
    return updated


def _provider_request_totals(history: list[dict]) -> tuple[list[str], int]:
    unique = sorted({
        str(image)
        for event in history
        for image in event.get("requested", [])
    })
    count = sum(
        int(event.get("request_count", len(event.get("requested", []))))
        for event in history
    )
    return unique, count


def _seed_legacy_manifest_request(
    history: list[dict], manifest: dict, round_no: int, contract_sha256: str,
) -> list[dict]:
    """Preserve pre-ledger requested_to_label metadata during a safe resume."""

    requested = _uniq(map(str, manifest.get("requested_to_label") or []))
    migrated_from_rejection = False
    if not requested and not history:
        requested = _uniq(map(str, manifest.get("rejected_not_budgeted") or []))
        migrated_from_rejection = bool(requested)
    if not requested:
        return list(history)
    if any(
        int(event.get("round", -1)) == int(round_no)
        and set(map(str, event.get("requested") or [])) == set(requested)
        for event in history
    ):
        return list(history)
    event = {
        "round": int(round_no),
        "requested": requested,
        "request_count": len(requested),
        "selection_contract_sha256": (
            manifest.get("selection_contract_before_sha256")
            if migrated_from_rejection else contract_sha256
        ),
        "preflight_manifest_sha256": None,
        "status": (
            "migrated_from_rejected_not_budgeted"
            if migrated_from_rejection else "migrated_from_requested_to_label"
        ),
    }
    event["event_sha256"] = _stable_json_hash(event)
    return [*history, event]


def _matching_repreflight_ready(
    ctx: Context,
    state: dict,
    round_no: int,
    pick: dict,
    diagnostics: dict,
    missing: list[str],
    out_dir: Path,
) -> tuple[bool, str | None]:
    """Validate a replacement selection's fresh approval before any networking."""

    if not missing or not state.get("requires_repreflight"):
        return True, None
    expected = state.get("pending_selection_contract_sha256")
    observed = _pending_selection_contract_sha256(pick, diagnostics)
    if not expected or expected != observed:
        raise RuntimeError(
            "pending replacement selection drifted from its frozen contract"
        )
    if not getattr(ctx.args, "auto_vqa", False):
        return False, "auto_vqa_disabled"
    required = (
        getattr(ctx.args, "vqa_preflight_manifest", None),
        getattr(ctx.args, "vqa_preflight_sha256", None),
        getattr(ctx.args, "vqa_preflight_task_order", None),
    )
    if not all(value is not None for value in required):
        return False, "replacement_selection_requires_new_preflight"

    from vqa_budget_preflight import (
        PreflightError,
        validate_external_credential_source,
        validate_runtime_batch,
    )

    config_path = Path(ctx.args.vqa_config)
    if not config_path.is_absolute():
        config_path = H.VQA_DIR / config_path
    try:
        validate_external_credential_source(config_path)
        validation = validate_runtime_batch(
            Path(ctx.args.vqa_preflight_manifest),
            str(ctx.args.vqa_preflight_sha256),
            order=int(ctx.args.vqa_preflight_task_order),
            dataset=H.DATASET,
            task=H.TASK,
            stage=ctx.stage,
            round_no=round_no,
            candidate_rels=missing,
            adapter=ctx.adapter,
            modeled_attributes=list(H.ATTRS),
            attributes_file=ctx.adapter.task_root / "qa" / "attributes.txt",
            vqa_config=config_path,
        )
    except PreflightError as error:
        return False, str(error)
    _write_json(out_dir / "outbound_preflight_validation.json", validation)
    return True, None


def _round0_selection(ctx: Context, available: list[str]) -> tuple[dict, dict]:
    """70 query-near + 15 text/proxy coverage + 15 MMR diversity.

    The proxy is explicitly recorded when text-space scoring is unavailable for a
    backbone. It is not an inferred ground-truth attribute label.
    """
    idx = np.asarray([ctx.p2i[p] for p in available], dtype=np.int64)
    proto = H.l2norm(ctx.emb_norm[ctx.query_idx].mean(axis=0, keepdims=True)).squeeze()
    full_qsim = ctx.emb_norm @ proto
    qsim = full_qsim[idx]
    blocked = set()
    exploit = _take_ranked(available, qsim, 70, blocked)

    # Text prompts are only comparable in CLIP/SigLIP spaces. Fall back to a
    # query-near MMR proxy on other backbones and make that visible in manifest.
    coverage_mode = "attribute_text"
    try:
        text = H.text_feats_for_backbone(ctx.args.backbone)
        attr_text = np.mean([ctx.emb_norm[idx] @ text[j] for j in range(len(H.ATTRS))], axis=0)
        coverage = _take_ranked(available, attr_text, 15, blocked)
    except Exception as exc:
        coverage_mode = f"query_near_proxy:{type(exc).__name__}"
        coverage = _take_ranked(available, qsim, 15, blocked)
    mmr = H.mmr_coverage_paths(available, ctx.emb_norm, ctx.p2i, full_qsim, 15, already=blocked, lam=0.55)
    blocked.update(mmr)
    roles = {"query_near": exploit, "attribute_text_coverage": coverage, "mmr_diversity": mmr}
    selected = _uniq(exploit + coverage + mmr)
    # Deterministic backfill protects exact batch size on tiny pools.
    selected += _take_ranked(available, qsim, ROUND0_SIZE - len(selected), set(selected))
    diagnostics = {"selection": "round0", "coverage_mode": coverage_mode,
                   "pool": len(available), "qsim": _score_summary(qsim)}
    return {"train": selected[:ROUND0_SIZE], "audit": [], "roles": roles}, diagnostics


def _fit_committee(ctx: Context, train_paths: list[str], vqa_map: dict):
    """Fit a light acquisition committee, never a ReCAP hard-negative model alone."""
    labeled_paths = _strict_cached_paths(vqa_map, train_paths)
    labeled = [{"image": rp, **vqa_map[rp]} for rp in labeled_paths]
    if len(labeled) < 12:
        return {}, {"reason": "fewer_than_12_labeled_train_examples"}
    train_idx = np.asarray([ctx.p2i[p] for p in train_paths], dtype=np.int64)
    all_idx = np.asarray([ctx.p2i[p] for p in ctx.train_set if p not in set(train_paths)], dtype=np.int64)
    if not len(all_idx):
        return {}, {"reason": "empty_unlabeled_pool"}
    cap_rng = np.random.default_rng(1107)
    u_sample = all_idx if len(all_idx) <= H.U_CAP else all_idx[cap_rng.choice(len(all_idx), H.U_CAP, replace=False)]
    run_ctx = {"emb": ctx.emb, "patches": None, "p2i": ctx.p2i,
               "text_q": {a: None for a in H.ATTRS}, "input_dim": ctx.input_dim,
               "text_dim": 1, "patch_dim": 0, "U_sample": u_sample}
    committee = {}
    failures = {}
    for method in ("mlp_baseline", "pu_ranking"):
        try:
            models = H.train_method(method, "mlp", labeled, train_paths, all_idx, run_ctx, H.SPLIT_SEED)
            if models:
                committee[method] = models
        except Exception as exc:  # Selection must remain resumable on sparse labels.
            failures[method] = f"{type(exc).__name__}: {exc}"
    return committee, {"n_train": len(labeled), "models": list(committee), "failures": failures}


def _committee_scores(ctx: Context, candidates: list[str], committee: dict):
    idx = np.asarray([ctx.p2i[p] for p in candidates], dtype=np.int64)
    qproto = H.l2norm(ctx.emb_norm[ctx.query_idx].mean(axis=0, keepdims=True)).squeeze()
    full_qsim = ctx.emb_norm @ qproto
    qsim = full_qsim[idx]
    out = {"query_prototype": qsim, "query_prototype_full": full_qsim}
    for name, models in committee.items():
        attrs = H.score_attr_models(models, ctx.emb[idx], None, {a: None for a in H.ATTRS})
        out[name] = attrs
        out[name + "_joint"] = H.joint_score_from(attrs)
    return out


def _role_selection(ctx: Context, available: list[str], train_paths: list[str], vqa_map: dict):
    committee, fit_diag = _fit_committee(ctx, train_paths, vqa_map)
    scores = _committee_scores(ctx, available, committee)
    qsim = scores["query_prototype"]
    full_qsim = scores["query_prototype_full"]
    if committee:
        attrs_by_method = [scores[m] for m in committee]
        mean_attr = {a: np.mean([s[a] for s in attrs_by_method], axis=0) for a in H.ATTRS}
        joint_vals = np.vstack([scores[m + "_joint"] for m in committee])
        disagreement = np.std(joint_vals, axis=0)
    else:
        # With insufficient class balance we can still collect new evidence; the
        # explicit zero disagreement avoids pretending a model was available.
        mean_attr = {a: qsim for a in H.ATTRS}
        disagreement = np.zeros_like(qsim)
    joint = H.joint_score_from(mean_attr)

    # Estimate joint and attribute boundaries from scores on the existing VQA
    # supervision. These are acquisition heuristics, not evaluation thresholds.
    labeled_train = _strict_cached_paths(vqa_map, train_paths)
    labeled_scores = _committee_scores(ctx, labeled_train, committee)
    if committee:
        labeled_attrs_by_method = [labeled_scores[m] for m in committee]
        labeled_mean_attr = {
            a: np.mean([s[a] for s in labeled_attrs_by_method], axis=0)
            for a in H.ATTRS
        }
    else:
        labeled_mean_attr = {
            a: labeled_scores["query_prototype"] for a in H.ATTRS
        }
    labeled_joint = H.joint_score_from(labeled_mean_attr)
    joint_labels = np.asarray([
        int(all(int(vqa_map[p][a]) == 1 for a in H.ATTRS))
        for p in labeled_train
    ], dtype=int)
    joint_boundary_value, joint_boundary_diag = _supervised_quantile_boundary(
        labeled_joint, joint_labels, joint
    )
    attr_boundary_values = {}
    attr_boundary_diag = {}
    for a in H.ATTRS:
        attr_labels = np.asarray([int(vqa_map[p][a]) for p in labeled_train], dtype=int)
        attr_boundary_values[a], attr_boundary_diag[a] = _supervised_quantile_boundary(
            labeled_mean_attr[a], attr_labels, mean_attr[a]
        )

    prior = _label_stats(vqa_map, train_paths)
    sparse = (prior["joint_positive"] < 5 or any(v < 5 for v in prior["attribute_positive"].values()))
    n_prior = max(1, prior["n"])
    extreme_attrs = [a for a, npos in prior["attribute_positive"].items()
                     if (npos / n_prior) > 0.95 or (npos / n_prior) < 0.05]
    blocked = set()
    roles = {}

    if extreme_attrs and any(prior["attribute_positive"][a] / n_prior > 0.95 for a in extreme_attrs):
        # High-positive saturation is the mirror image of rare-positive data:
        # without lower-score evidence a binary probe has no negative class at
        # all. Select a broad mid/low similarity band, rather than asserting
        # that such samples are negatives before VQA checks them.
        lo, hi = np.quantile(qsim, [0.15, 0.65])
        balance = -np.abs(qsim - (lo + hi) / 2.0)
        roles["class_balance_evidence"] = _take_ranked(available, balance, 20, blocked)
        boundary = -np.abs(joint - joint_boundary_value) + 0.05 * qsim
        roles["targeted_boundary"] = _take_ranked(available, boundary, 8, blocked)
        roles["committee_disagreement"] = _take_ranked(available, disagreement, 6, blocked)
        roles["mmr_diversity"] = H.mmr_coverage_paths(available, ctx.emb_norm, ctx.p2i, full_qsim, 6, already=blocked, lam=0.45)
        blocked.update(roles["mmr_diversity"])
        role_order = ["class_balance_evidence", "targeted_boundary", "committee_disagreement", "mmr_diversity"]
    elif sparse:
        # Rare-positive mode is intentionally harvest-first and keeps boundaries
        # modest, so unlabeled samples are not converted into hard negatives.
        roles["positive_harvest"] = _take_ranked(available, joint + 0.15 * qsim, 24, blocked)
        near = np.exp(-np.abs(joint - joint_boundary_value) / BOUNDARY_TEMPERATURE) + 0.1 * qsim
        roles["targeted_boundary"] = _take_ranked(available, near, 8, blocked)
        roles["committee_disagreement"] = _take_ranked(available, disagreement, 4, blocked)
        roles["mmr_diversity"] = H.mmr_coverage_paths(available, ctx.emb_norm, ctx.p2i, full_qsim, 4, already=blocked, lam=0.55)
        blocked.update(roles["mmr_diversity"])
        role_order = ["positive_harvest", "targeted_boundary", "committee_disagreement", "mmr_diversity"]
    else:
        # A joint boundary means: high query relevance, all-but-one attribute
        # evidence high, and one attribute uncertain. This is a hypothesis only;
        # VQA supplies the label after selection.
        boundary = np.zeros(len(available), dtype=float)
        for a in H.ATTRS:
            others = [mean_attr[b] for b in H.ATTRS if b != a]
            other_high = np.min(np.stack(others), axis=0) if others else np.ones(len(available))
            uncertain = np.exp(
                -np.abs(mean_attr[a] - attr_boundary_values[a]) / BOUNDARY_TEMPERATURE
            )
            boundary = np.maximum(boundary, other_high * uncertain)
        boundary *= np.clip((qsim - np.quantile(qsim, 0.35)) / (np.ptp(qsim) + 1e-6), 0, 1)
        roles["joint_boundary"] = _take_ranked(available, boundary, 14, blocked)
        deficit = np.zeros(len(available), dtype=float)
        for a, npos in prior["attribute_positive"].items():
            if npos < 12:
                deficit = np.maximum(deficit, mean_attr[a] * (12 - npos) / 12)
        roles["attribute_deficit"] = _take_ranked(available, deficit, 10, blocked)
        roles["committee_disagreement"] = _take_ranked(available, disagreement, 10, blocked)
        roles["mmr_diversity"] = H.mmr_coverage_paths(available, ctx.emb_norm, ctx.p2i, full_qsim, 6, already=blocked, lam=0.55)
        blocked.update(roles["mmr_diversity"])
        role_order = ["joint_boundary", "attribute_deficit", "committee_disagreement", "mmr_diversity"]

    train = _uniq([p for role in role_order for p in roles[role]])
    train += _take_ranked(available, joint + 0.05 * qsim, TRAIN_PER_REFINE - len(train), set(train))
    # The permanent audit is sampled from model-visible cohorts but never enters
    # the next training set. It is predicted before its VQA labels are read.
    audit_blocked = set(train)
    audit_query = _take_ranked(available, qsim, 4, audit_blocked)
    audit_boundary = _take_ranked(
        available, np.abs(joint - joint_boundary_value) * -1, 3, audit_blocked
    )
    audit_mmr = H.mmr_coverage_paths(available, ctx.emb_norm, ctx.p2i, full_qsim, 3, already=audit_blocked, lam=0.55)
    audit_blocked.update(audit_mmr)
    audit = _uniq(audit_query + audit_boundary + audit_mmr)
    audit += _take_ranked(available, qsim, AUDIT_PER_REFINE - len(audit), set(train) | set(audit))
    diagnostics = {
        "selection": ("class_balance_guard" if extreme_attrs and any(prior["attribute_positive"][a] / n_prior > 0.95 for a in extreme_attrs)
                      else "rare_positive_harvest" if sparse else "committee_acquisition"),
        "prior": prior, "committee": fit_diag, "qsim": _score_summary(qsim),
        "joint": _score_summary(joint), "disagreement": _score_summary(disagreement),
        "extreme_attributes": extreme_attrs,
        "boundary_estimation": {
            "joint": joint_boundary_diag,
            "attributes": attr_boundary_diag,
        },
    }
    selected_for_diag = _uniq(train[:TRAIN_PER_REFINE] + audit[:AUDIT_PER_REFINE])
    pos = {p: i for i, p in enumerate(available)}
    diagnostics["selected_scores"] = {
        p: {"query_similarity": round(float(qsim[pos[p]]), 6),
            "attribute_scores": {a: round(float(mean_attr[a][pos[p]]), 6) for a in H.ATTRS},
            "joint_score": round(float(joint[pos[p]]), 6),
            "joint_boundary_distance": round(
                float(abs(joint[pos[p]] - joint_boundary_value)), 6
            ),
            "joint_boundary_score": round(float(np.exp(
                -abs(joint[pos[p]] - joint_boundary_value) / BOUNDARY_TEMPERATURE
            )), 6),
            "committee_disagreement": round(float(disagreement[pos[p]]), 6)}
        for p in selected_for_diag
    }
    return {"train": train[:TRAIN_PER_REFINE], "audit": audit[:AUDIT_PER_REFINE], "roles": roles}, diagnostics


def _prequential_predictions(ctx: Context, audit: list[str], train: list[str], vqa_map: dict):
    committee, diag = _fit_committee(ctx, train, vqa_map)
    if not audit or not committee:
        return {"n": len(audit), "committee": diag, "joint_scores": {}}
    scores = _committee_scores(ctx, audit, committee)
    return {
        "n": len(audit), "committee": diag,
        "joint_scores": {m: [round(float(x), 6) for x in scores[m + "_joint"]] for m in committee},
    }


def _audit_prequential_metrics(vqa_map: dict, audit: list[str], prequential: dict):
    """Evaluate predictions made before the audit labels were consumed."""
    invalid = [
        path for path in audit
        if not H.is_complete_binary_labels(vqa_map.get(path), H.ATTRS)
    ]
    if invalid:
        raise ValueError(f"prequential audit has incomplete VQA labels: {invalid[:3]}")
    y = np.asarray([
        int(all(int(vqa_map[p][a]) == 1 for a in H.ATTRS)) for p in audit
    ])
    result = {"n": int(len(y)), "joint_positive": int(y.sum()), "methods": {}}
    for method, values in (prequential.get("joint_scores") or {}).items():
        score = np.asarray(values, dtype=float)
        if len(score) != len(y) or not len(score):
            continue
        order = np.argsort(-score)
        result["methods"][method] = {
            "ap": float(H.average_precision_score(y, score)) if y.sum() else None,
            "p@10": float(y[order[:min(10, len(y))]].mean()),
        }
    return result


def _audit_budget_decision(rounds: list[dict], stats: dict, total_labeled: int,
                           max_labels: int, default_stop: int,
                           min_gain: float = 0.05) -> dict:
    """Decide the next budget from VQA-only prequential audits.

    The decision deliberately never reads gallery/test ground truth.  It is
    conservative because each audit batch has only ten images: a task can stop
    early only after high, stable audit precision *and* adequate attribute
    coverage.  At the nominal 250-label point, one extra round is allowed only
    when the audit signal is still improving.
    """
    signal = []
    for rec in rounds:
        methods = rec.get("audit_prequential_metrics", {}).get("methods", {})
        value = methods.get("pu_ranking", {}).get("p@10")
        if value is None:
            value = methods.get("mlp_baseline", {}).get("p@10")
        if value is not None:
            signal.append(float(value))
    min_attr = min(stats.get("attribute_positive", {}).values(), default=0)
    decision = {
        "total_labeled": int(total_labeled), "audit_p10_history": signal,
        "minimum_attribute_positives": int(min_attr), "action": "continue",
        "reason": "below_adaptive_decision_budget",
    }
    if total_labeled >= max_labels:
        decision.update(action="stop", reason="hard_budget_ceiling")
        return decision
    if total_labeled < 200 or len(signal) < 2:
        return decision
    recent_gain = signal[-1] - signal[-2]
    decision["recent_audit_p10_gain"] = round(float(recent_gain), 6)
    # Early stopping is intentionally strict: do not end a sparse-positive task
    # merely because one tiny audit batch happens to be flat.
    if (total_labeled == 200 and min_attr >= 8 and min(signal[-2:]) >= 0.8
            and abs(recent_gain) < min_gain):
        decision.update(action="stop", reason="stable_high_audit_precision_at_200")
        return decision
    if total_labeled >= default_stop:
        if recent_gain >= min_gain:
            decision.update(action="extend", reason="audit_precision_still_improving")
        else:
            decision.update(action="stop", reason="no_reliable_audit_improvement_at_nominal_budget")
    return decision


def _new_labels(
    ctx: Context,
    paths: list[str],
    vqa_map: dict,
    out_dir: Path,
    round_no: int,
):
    candidates = list(paths)
    reused = _strict_cached_paths(vqa_map, candidates)
    missing = [p for p in candidates if p not in set(reused)]
    requested_paths = list(missing)
    _write_json(out_dir / "to_label.json", sorted(missing))
    _write_json(out_dir / "reused_labels.json", sorted(reused))
    if missing and not ctx.args.iterative_dry_run:
        if not ctx.args.auto_vqa:
            return reused, missing, False, requested_paths
        if getattr(ctx.args, "vqa_preflight_manifest", None):
            from vqa_budget_preflight import (
                PreflightError,
                validate_external_credential_source,
                validate_runtime_batch,
            )

            config_path = Path(ctx.args.vqa_config)
            if not config_path.is_absolute():
                config_path = H.VQA_DIR / config_path
            try:
                validate_external_credential_source(config_path)
                validation = validate_runtime_batch(
                    Path(ctx.args.vqa_preflight_manifest),
                    str(ctx.args.vqa_preflight_sha256),
                    order=int(ctx.args.vqa_preflight_task_order),
                    dataset=H.DATASET,
                    task=H.TASK,
                    stage=ctx.stage,
                    round_no=round_no,
                    candidate_rels=missing,
                    adapter=ctx.adapter,
                    modeled_attributes=list(H.ATTRS),
                    attributes_file=ctx.adapter.task_root / "qa" / "attributes.txt",
                    vqa_config=config_path,
                )
            except PreflightError as error:
                raise SystemExit(f"[auto-vqa] outbound preflight rejected: {error}") from error
            _write_json(out_dir / "outbound_preflight_validation.json", validation)
        H.run_vqa_pipeline(
            ctx.adapter, missing, ctx.args.vqa_config, ctx.args.vqa_python,
            ctx.args.vqa_workers, output_dir=out_dir,
            attr_config=ctx.args.attr_config, require_min_success=False,
            apply_manual_content_filter_labels=False,
        )
        source_paths = _context_cached_sources(ctx)
        ctx.cached_map, ctx.cached_prob, ctx.cache_audit = _load_iterative_cache(
            source_paths, ctx.adapter, ctx.qa_round_root,
        )
        missing = [
            p for p in candidates
            if not H.is_complete_binary_labels(ctx.cached_map.get(p), H.ATTRS)
        ]
        return reused, missing, not missing, requested_paths
    return reused, missing, not missing, requested_paths


def _round_label_status(
    pick: dict,
    cached_map: dict,
    attrs=None,
) -> tuple[list[str], list[str], bool]:
    """Audit a frozen selection without shrinking it after provider failures.

    A VQA budget counts selected images, not only the images answered in one
    provider attempt.  Keeping the original selection intact makes partial
    batches resumable and prevents underfilled trajectories from being marked
    complete.
    """
    attrs = tuple(H.ATTRS if attrs is None else attrs)
    candidates = _uniq(list(pick["train"]) + list(pick["audit"]))
    missing = [
        path for path in candidates
        if not H.is_complete_binary_labels(cached_map.get(path), attrs)
    ]
    return candidates, missing, not missing


def _eval_fixed(ctx: Context, train_paths: list[str], method: str, score_path: Path | None = None):
    """Post-selection evaluation only; never used by acquisition."""
    vqa_map = ctx.cached_map
    labeled_paths = _strict_cached_paths(vqa_map, train_paths)
    labeled = [{"image": p, **vqa_map[p]} for p in labeled_paths]
    if len(labeled) < 12:
        return {"status": "skipped_insufficient_labels"}
    train_idx = np.asarray([ctx.p2i[p] for p in train_paths], dtype=np.int64)
    u_paths = [p for p in ctx.train_set if p not in set(train_paths)]
    u_idx = np.asarray([ctx.p2i[p] for p in u_paths], dtype=np.int64)
    if not len(u_idx):
        return {"status": "skipped_empty_unlabeled"}
    sample = u_idx[:min(len(u_idx), H.U_CAP)]
    run_ctx = {"emb": ctx.emb, "patches": None, "p2i": ctx.p2i,
               "text_q": {a: None for a in H.ATTRS}, "input_dim": ctx.input_dim,
               "text_dim": 1, "patch_dim": 0, "U_sample": sample}
    try:
        models = H.train_method(method, "mlp", labeled, train_paths, u_idx, run_ctx, H.SPLIT_SEED)
        if not models:
            return {"status": "skipped_no_models"}
        def score(paths):
            idx = np.asarray([ctx.p2i[p] for p in paths], dtype=np.int64)
            attrs = H.score_attr_models(models, ctx.emb[idx], None, {a: None for a in H.ATTRS})
            return attrs, H.joint_score_from(attrs)
        test_attr, test_joint = score(ctx.test_set)
        gal_attr, gal_joint = score(ctx.gallery_paths)
        gt_test = {a: np.asarray([int(ctx.gt_by_attr[a].get(p, 0)) for p in ctx.test_set]) for a in H.ATTRS}
        gt_gal = {a: np.asarray([int(ctx.gt_by_attr[a].get(p, 0)) for p in ctx.gallery_paths]) for a in H.ATTRS}
        if score_path is not None:
            score_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"test_joint": test_joint, "gallery_joint": gal_joint,
                       "test_joint_gt": H.joint_gt_from(gt_test), "gallery_joint_gt": H.joint_gt_from(gt_gal)}
            for a in H.ATTRS:
                payload[f"test_{a}"] = test_attr[a]
                payload[f"gallery_{a}"] = gal_attr[a]
            np.savez_compressed(score_path, **payload)
        return {"status": "ok",
                "test": H.eval_method_on_test(test_attr, test_joint, gt_test, H.joint_gt_from(gt_test), True),
                "gallery": H.eval_method_on_test(gal_attr, gal_joint, gt_gal, H.joint_gt_from(gt_gal), True)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _static_control(ctx: Context, total_train: int, score_root: Path | None = None):
    """Same-budget static two-stage control, using cached labels only."""
    if ctx.isolated:
        return {
            "status": "disabled_iterative_only",
            "source_policy": ISOLATED_SOURCE_POLICY,
        }
    _, sources, _ = H.resolve_vqa_sources(ctx.adapter.task_root / "qa", ["two_stage"], ctx.adapter.task_root / "split")
    control_map, _, control_audit = H.load_vqa_many_strict(
        sources.get("two_stage", []), ctx.adapter.vqa_to_key, H.ATTRS,
    )
    if control_audit.get("conflicts"):
        return {
            "status": "unavailable_conflicting_two_stage_cache",
            "conflicts": sorted(control_audit["conflicts"])[:10],
        }
    available = _strict_cached_paths(control_map, ctx.train_set)
    if not available:
        return {"status": "unavailable_no_two_stage_cache"}
    ranked, _ = H.query_ranked_paths(available, ctx.emb_norm, ctx.p2i, ctx.query_idx)
    chosen = ranked[:min(total_train, len(ranked))]
    _write_json(ctx.split_root / f"static_two_stage_train_indices_{len(chosen)}.json", chosen)
    old_map = ctx.cached_map
    ctx.cached_map = control_map
    result = {"n_train": len(chosen), "indices_file": str(ctx.split_root / f"static_two_stage_train_indices_{len(chosen)}.json"),
              "mlp_baseline": _eval_fixed(ctx, chosen, "mlp_baseline", None if score_root is None else score_root / "mlp_baseline.npz"),
              "pu_ranking": _eval_fixed(ctx, chosen, "pu_ranking", None if score_root is None else score_root / "pu_ranking.npz")}
    ctx.cached_map = old_map
    return result


def _ap_p50(result: dict):
    if result.get("status") != "ok":
        return None
    joint = result["test"].get(H.JOINT_KEY, {})
    return {"ap": joint.get("ap"), "p@50": joint.get("p@50")}


def _write_report(ctx: Context, state: dict):
    rows = ["# Iterative VQA 100+50 Report", "",
            f"- Dataset/task: `{H.DATASET}/{H.TASK}`", f"- Stage: `{ctx.stage}`",
            "- Ground truth is used only for the pre-frozen benchmark split and post-selection evaluation; candidate scoring/acquisition never reads gallery/test ground-truth labels.", "",
            "| Round | Total VQA | Train | Audit | New calls | Joint positives | Policy | Budget decision |", "|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in state.get("rounds", []):
        stats = r.get("label_stats", {})
        rows.append("| {round} | {total} | {train} | {audit} | {new} | {joint} | {policy} | {decision} |".format(
            round=r["round"], total=r.get("total_labeled", 0), train=r.get("n_train", 0),
            audit=r.get("n_audit", 0), new=r.get("n_new", 0), joint=stats.get("joint_positive", 0),
            policy=r.get("diagnostics", {}).get("selection", "round0"),
            decision=r.get("budget_decision", {}).get("reason", "pending")))
    control_note = (
        "This isolated run intentionally disables the historical static two-stage "
        "control and reads only VQA results produced below its own work root."
        if ctx.isolated else
        "`controls.json` stores the same-budget static two-stage comparison, "
        "generated solely from cached labels."
    )
    rows += ["", "## Evidence", "",
             "Each refinement batch reserves ten permanent prequential-audit images. Their VQA labels are not included in subsequent training. " + control_note,
             "", "## Sparse-Positive Rule", "",
             "Rare-positive mode activates when the accumulated train set has fewer than five joint positives or any required attribute has fewer than five positives. It shifts the 40 train slots to 24 positive-harvesting, 8 targeted-boundary, 4 disagreement, and 4 MMR-diversity samples. A later stop decision requires two low-yield harvest rounds and no audit/P@50 movement; this runner records the required evidence but does not silently declare a task solved."]
    _write_json(ctx.report_root / "summary.json", state)
    (ctx.report_root / "iterative_vqa_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with open(ctx.report_root / "round_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "total_vqa", "train", "audit", "new_calls", "joint_positive", "policy", "budget_decision"])
        writer.writeheader()
        for r in state.get("rounds", []):
            writer.writerow({"round": r["round"], "total_vqa": r.get("total_labeled", 0),
                             "train": r.get("n_train", 0), "audit": r.get("n_audit", 0),
                             "new_calls": r.get("n_new", 0),
                             "joint_positive": r.get("label_stats", {}).get("joint_positive", 0),
                             "policy": r.get("diagnostics", {}).get("selection", "round0"),
                             "budget_decision": r.get("budget_decision", {}).get("reason", "pending")})


def _validate_iterative_protocol(args) -> None:
    protocol = (args.iterative_initial_labels, args.iterative_round_labels,
                args.iterative_train_per_round, args.iterative_audit_per_round)
    if protocol != (ROUND0_SIZE, REFINE_SIZE, TRAIN_PER_REFINE, AUDIT_PER_REFINE):
        raise SystemExit("iterative_vqa_100_50_v1 requires 100/50/40/10; create a new version for another protocol")
    if (
        args.iterative_max_labels > MAX_ITERATIVE_LABELS
        or args.iterative_stop_at_labels > args.iterative_max_labels
    ):
        raise SystemExit(
            "iterative label budgets must satisfy "
            f"stop <= max <= {MAX_ITERATIVE_LABELS}"
        )
    reachable_budget = ROUND0_SIZE + REFINE_SIZE * args.iterative_max_rounds
    if reachable_budget > args.iterative_max_labels:
        raise SystemExit(
            "iterative rounds exceed the hard label ceiling: "
            f"{reachable_budget} > {args.iterative_max_labels}"
        )
    preflight_values = (
        getattr(args, "vqa_preflight_manifest", None),
        getattr(args, "vqa_preflight_sha256", None),
        getattr(args, "vqa_preflight_task_order", None),
    )
    if any(value is not None for value in preflight_values) and not all(
        value is not None for value in preflight_values
    ):
        raise SystemExit(
            "VQA preflight requires manifest, sha256, and task order together"
        )
    if getattr(args, "vqa_preflight_manifest", None) and not getattr(
        args, "auto_vqa", False
    ):
        raise SystemExit("VQA preflight arguments require --auto-vqa")
    if (
        getattr(args, "auto_vqa", False)
        and int(args.iterative_max_labels) == MAX_ITERATIVE_LABELS
        and not getattr(args, "vqa_preflight_manifest", None)
    ):
        raise SystemExit(
            "500-label iterative auto-VQA requires an approved outbound preflight"
        )


def run_iterative_vqa(args) -> None:
    _validate_iterative_protocol(args)
    if args.iterative_stage == DEFAULT_STAGE:
        stage = DEFAULT_STAGE
    elif not args.iterative_stage.startswith("iterative_vqa_100_50_v"):
        raise SystemExit("--iterative-stage must start with iterative_vqa_100_50_v")
    else:
        stage = args.iterative_stage
    H.configure(args.dataset, args.task, args.joint_label)
    adapter = H.ADAPTER
    emb, _ = H.load_backbone_embeddings(adapter, args.backbone)
    emb_norm = H.l2norm(emb)
    work_root = Path(args.iterative_work_root).resolve() if args.iterative_work_root else None
    gallery, test, train = _frozen_split(
        adapter,
        emb_norm,
        adapter.p2i,
        adapter.gt_by_attr,
        use_persisted_evaluation=work_root is None,
    )
    isolated_identity = None
    isolated_identity_hash = None
    if work_root is None:
        sources = _all_cached_sources(adapter, stage)
        qa_round_root = adapter.task_root / "qa" / stage
        split_root = adapter.task_root / "split" / stage
        embedding_root = adapter.task_root / "embedding_runs" / H.backbone_slug(args.backbone) / stage
        repo_root = Path(__file__).resolve().parents[2]
        report_root = repo_root / "dataset" / "tasks" / "iterative_vqa_100_50" / args.dataset / args.task
    else:
        qa_round_root = work_root / "qa"
        split_root = work_root / "split"
        embedding_root = work_root / "embedding"
        report_root = work_root / "report"
        isolated_identity = _isolated_run_identity(
            args, adapter, stage, work_root, gallery, test, train,
        )
        isolated_identity_hash = _stable_json_hash(isolated_identity)
        _validate_or_initialize_isolated_work_root(work_root, isolated_identity)
        sources = _isolated_iterative_sources(qa_round_root)
    cached_map, cached_prob, cache_audit = _load_iterative_cache(
        sources, adapter, qa_round_root,
    )
    repo_root = Path(__file__).resolve().parents[2]
    ctx = Context(args, stage, adapter.task_root, qa_round_root, split_root,
                  embedding_root, report_root,
                  adapter, emb, emb_norm, adapter.p2i, train, test, gallery, list(adapter.query_idx),
                  int(emb.shape[1]), adapter.gt_by_attr, cached_map, cached_prob, cache_audit,
                  isolated=work_root is not None)
    for p in (ctx.qa_round_root, ctx.split_root, ctx.embedding_root, ctx.report_root):
        p.mkdir(parents=True, exist_ok=True)
    root_manifest = ctx.qa_round_root / "manifest.json"
    if not root_manifest.is_file():
        manifest_payload = {
            "version": stage, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "budget": {"initial": ROUND0_SIZE, "round": REFINE_SIZE, "train": TRAIN_PER_REFINE,
                       "audit": AUDIT_PER_REFINE, "max": args.iterative_max_labels, "default_stop": args.iterative_stop_at_labels},
            "backbone": args.backbone, "seeds": [H.SPLIT_SEED], "attributes": list(H.ATTRS),
            "vqa_sources": [str(p) for p in sources], "vqa_config": str(args.vqa_config),
            "exclude_provider_content_rejections": bool(getattr(
                args, "iterative_exclude_provider_content_rejections", False,
            )),
            "isolated_work_root": str(work_root) if work_root is not None else None,
            "source_policy": ISOLATED_SOURCE_POLICY if work_root is not None else "task-history-compatible",
            "evaluation_split_policy": (
                ISOLATED_EVALUATION_POLICY
                if work_root is not None else "task-frozen-manifest-compatible"
            ),
        }
        if isolated_identity is not None:
            manifest_payload.update({
                "identity": isolated_identity,
                "run_identity_sha256": isolated_identity_hash,
                "budget": isolated_identity["budget"],
            })
        _write_json(root_manifest, manifest_payload)
    state_path = ctx.split_root / "round_state.json"
    if state_path.is_file() and not args.iterative_resume:
        raise SystemExit(f"iterative stage already exists: {state_path}; pass --iterative-resume or choose a v2 stage")
    state = _read_json(state_path, {"stage": stage, "dataset": H.DATASET, "task": H.TASK,
                                     "config": {"round0": 100, "refine": 50, "train_refine": 40,
                                                "audit_refine": 10, "max_refine_rounds": args.iterative_max_rounds,
                                                "exclude_provider_content_rejections": bool(getattr(
                                                    args, "iterative_exclude_provider_content_rejections", False,
                                                ))},
                                    "sources": [str(p) for p in sources], "rounds": [], "status": "running",
                                    "source_policy": (
                                        ISOLATED_SOURCE_POLICY if work_root is not None
                                        else "task-history-compatible"
                                    ),
                                    "run_identity_sha256": isolated_identity_hash})
    if ctx.isolated and state.get("source_policy") != ISOLATED_SOURCE_POLICY:
        raise RuntimeError(
            "isolated iterative state lacks the run-local-only VQA source contract; "
            "use a new --iterative-work-root"
        )
    if ctx.isolated and state.get("run_identity_sha256") != isolated_identity_hash:
        raise RuntimeError(
            "isolated iterative state does not belong to this exact run identity; "
            "use a new --iterative-work-root"
        )
    cache_resume_audit = _committed_cache_audit(
        state, ctx.cached_map, ctx.cache_audit,
    )
    _write_json(ctx.split_root / "cache_resume_audit.json", cache_resume_audit)
    if not cache_resume_audit["ok"]:
        earliest = cache_resume_audit.get("earliest_invalid_committed_round")
        invalid = cache_resume_audit.get("invalid_committed_rounds", [])
        uncommitted = cache_resume_audit.get("uncommitted_conflicts", [])
        raise RuntimeError(
            "iterative VQA cache audit failed; no state was repaired. "
            f"earliest invalid committed round={earliest!r}; "
            f"invalid committed rounds={invalid[:3]}; "
            f"unresolved uncommitted conflicts={uncommitted[:3]}. "
            f"See {ctx.split_root / 'cache_resume_audit.json'} and archive/reset "
            "the affected trajectory or correct its source cache explicitly."
        )
    migrated_cache_rounds = bool(cache_resume_audit.get("migrations"))
    _apply_cache_audit_migrations(state, cache_resume_audit)
    if migrated_cache_rounds or state_path.is_file():
        _write_json(state_path, state)
    if (args.iterative_resume and not args.iterative_export_final_scores
            and state.get("status") in ("completed", "stopped_audit_budget_policy", "stopped_empty_pool")):
        print(f"Iterative stage already finalized: {state_path} ({state['status']})")
        return
    # A short-lived pre-release version reloaded its cache before recording
    # ``n_new``. Repair only that metadata from the immutable manifest; labels
    # and selection are never changed.
    for old_round in state.get("rounds", []):
        manifest_path = _round_paths(ctx, int(old_round["round"]))["manifest"]
        manifest = _read_json(manifest_path, {})
        if old_round.get("n_new", 0) == 0:
            requested = manifest.get("requested_to_label") or manifest.get("to_label")
            if requested:
                old_round["n_new"] = len(requested)
            else:
                # Migration for the first pilot: its manifest was refreshed after
                # cache reload, so the round-local JSONL is the immutable record.
                result_rows = H.all_jsonl_results(_round_paths(ctx, int(old_round["round"]))["qa"])
                if result_rows:
                    old_round["n_new"] = sum(1 for _ in H._load_direct_label_rows(result_rows[-1]))
        old_control = old_round.get("static_two_stage_control", {})
        if (not ctx.isolated
                and old_control.get("status") == "unavailable_no_two_stage_cache"):
            old_round["static_two_stage_control"] = _static_control(ctx, int(old_round.get("n_train", 0)))
        if "audit_label_stats" not in old_round:
            old_round["audit_label_stats"] = _label_stats(ctx.cached_map, list(old_round.get("audit", [])))
        if "audit_prequential_metrics" not in old_round:
            old_round["audit_prequential_metrics"] = _audit_prequential_metrics(
                ctx.cached_map, list(old_round.get("audit", [])), old_round.get("prequential_audit", {}))
    selected_train = _uniq([p for r in state["rounds"] for p in r.get("train", [])])
    selected_audit = _uniq([p for r in state["rounds"] for p in r.get("audit", [])])
    if args.iterative_export_final_scores:
        if not state.get("rounds"):
            raise SystemExit("cannot export iterative scores before a completed round")
        export_root = ctx.embedding_root / "final" / "fixed_scores"
        final = state["rounds"][-1]
        final["score_exports"] = {
            "iterative_mlp": str(export_root / "iterative" / "mlp_baseline.npz"),
            "iterative_pu": str(export_root / "iterative" / "pu_ranking.npz"),
        }
        if not ctx.isolated:
            final["score_exports"].update({
                "static_mlp": str(export_root / "static_two_stage" / "mlp_baseline.npz"),
                "static_pu": str(export_root / "static_two_stage" / "pu_ranking.npz"),
            })
        final["fixed_eval"] = {
            "mlp_baseline": _eval_fixed(ctx, selected_train, "mlp_baseline", export_root / "iterative" / "mlp_baseline.npz"),
            "pu_ranking": _eval_fixed(ctx, selected_train, "pu_ranking", export_root / "iterative" / "pu_ranking.npz"),
        }
        final["static_two_stage_control"] = _static_control(
            ctx,
            len(selected_train),
            None if ctx.isolated else export_root / "static_two_stage",
        )
        _write_json(state_path, state)
        _write_json(ctx.embedding_root / "final" / "final_state.json", state)
        _write_report(ctx, state)
        print(f"Fixed score exports written to: {export_root}")
        return
    start = len(state["rounds"])
    target_rounds = 1 + args.iterative_max_rounds
    print(f"=== iterative VQA === dataset={H.DATASET} task={H.TASK} stage={stage} rounds={start}/{target_rounds}")
    print(f"    cache rows={len(ctx.cached_map)}; train_pool={len(train)}; dry_run={args.iterative_dry_run}; auto_vqa={args.auto_vqa}")
    for round_no in range(start, target_rounds):
        exclude_provider_rejections = bool(getattr(
            args, "iterative_exclude_provider_content_rejections", False,
        ))
        provider_rejected = (
            _provider_content_rejections(ctx) if exclude_provider_rejections else {}
        )
        unavailable = set(selected_train) | set(selected_audit)
        if exclude_provider_rejections:
            unavailable.update(provider_rejected)
            unavailable.update(map(
                str, state.get("provider_rejected_not_budgeted") or [],
            ))
        available = [p for p in train if p not in unavailable]
        if not available:
            state["status"] = "stopped_empty_pool"
            break
        rpaths = _round_paths(ctx, round_no)
        frozen_manifest = _read_json(rpaths["manifest"], {}) if args.iterative_resume else {}
        if frozen_manifest.get("stage") == stage and int(frozen_manifest.get("round", -1)) == round_no:
            # A dry-run manifest is a contract: resume must label exactly those
            # candidates, even if stochastic probe training would otherwise rank
            # an equivalent-looking batch differently on a later process.
            pick = {"train": list(frozen_manifest.get("train", [])),
                    "audit": list(frozen_manifest.get("audit", [])),
                    "roles": dict(frozen_manifest.get("roles", {}))}
            diagnostics = dict(frozen_manifest.get("diagnostics", {}))
            diagnostics["selection_source"] = "frozen_manifest"
        elif round_no == 0:
            pick, diagnostics = _round0_selection(ctx, available)
        else:
            pick, diagnostics = _role_selection(ctx, available, selected_train, ctx.cached_map)
        rpaths["qa"].mkdir(parents=True, exist_ok=True)
        candidate_pool_path = rpaths["qa"] / "candidate_pool.json"
        if not candidate_pool_path.is_file():
            _write_json(candidate_pool_path, available)
        if exclude_provider_rejections and provider_rejected:
            pick, diagnostics, rejection_audit = _refresh_rejected_frozen_selection(
                ctx, state, round_no, pick, diagnostics, available, provider_rejected,
            )
            if rejection_audit["changed"]:
                refrozen_complete = _persist_rejected_pending_selection(
                    ctx, state, rpaths, round_no, frozen_manifest, pick,
                    diagnostics, rejection_audit, selected_train, selected_audit,
                )
                print(
                    f"  round {round_no}: excluded "
                    f"{len(rejection_audit['rejected_not_budgeted'])} terminal "
                    "provider rejection(s); selection re-frozen."
                )
                frozen_manifest = _read_json(rpaths["manifest"], {})
                if not refrozen_complete:
                    print("  replacement rows require a newly frozen outbound preflight.")
                    return
        _validate_frozen_selection(ctx, round_no, pick, available)
        candidates = _uniq(pick["train"] + pick["audit"])
        preq = _prequential_predictions(ctx, pick["audit"], selected_train, ctx.cached_map) if round_no else {"n": 0}
        pre_call_missing = [
            image for image in candidates
            if not H.is_complete_binary_labels(ctx.cached_map.get(image), H.ATTRS)
        ]
        repreflight_ready, repreflight_reason = _matching_repreflight_ready(
            ctx, state, round_no, pick, diagnostics, pre_call_missing, rpaths["qa"],
        )
        if not repreflight_ready:
            state["status"] = "awaiting_vqa"
            state["pending_round"] = int(round_no)
            state["repreflight_block_reason"] = repreflight_reason
            _write_json(state_path, state)
            _write_report(ctx, state)
            print(
                f"  round {round_no}: replacement selection is waiting for a "
                f"matching preflight ({repreflight_reason})."
            )
            return
        state.pop("repreflight_block_reason", None)
        provider_request_history = list(
            frozen_manifest.get("provider_request_history") or []
        )
        call_contract_sha256 = _pending_selection_contract_sha256(pick, diagnostics)
        provider_request_history = _seed_legacy_manifest_request(
            provider_request_history, frozen_manifest, round_no,
            call_contract_sha256,
        )
        reused, _, _, requested_paths = _new_labels(
            ctx, candidates, ctx.cached_map, rpaths["qa"], round_no
        )
        if (
            requested_paths and getattr(args, "auto_vqa", False)
            and not getattr(args, "iterative_dry_run", False)
        ):
            provider_request_history = _append_provider_request_event(
                ctx, provider_request_history, round_no, requested_paths,
                call_contract_sha256, rpaths["qa"],
            )
        working_manifest = dict(frozen_manifest)
        working_manifest["requested_to_label"] = list(requested_paths)
        working_manifest["provider_request_history"] = provider_request_history
        provider_rejected = (
            _provider_content_rejections(ctx) if exclude_provider_rejections else {}
        )
        if exclude_provider_rejections and provider_rejected:
            pick, diagnostics, rejection_audit = _refresh_rejected_frozen_selection(
                ctx, state, round_no, pick, diagnostics, available, provider_rejected,
            )
            if rejection_audit["changed"]:
                refrozen_complete = _persist_rejected_pending_selection(
                    ctx, state, rpaths, round_no, working_manifest, pick,
                    diagnostics, rejection_audit, selected_train, selected_audit,
                )
                print(
                    f"  round {round_no}: excluded "
                    f"{len(rejection_audit['rejected_not_budgeted'])} terminal "
                    "provider rejection(s); selection re-frozen."
                )
                frozen_manifest = _read_json(rpaths["manifest"], {})
                provider_request_history = list(
                    frozen_manifest.get("provider_request_history") or []
                )
                if not refrozen_complete:
                    print("  replacement rows require a newly frozen outbound preflight.")
                    return
        candidates, missing, complete = _round_label_status(pick, ctx.cached_map)
        # A provider/cache reload may add sources.  Re-audit every already
        # committed label before allowing the new source set to influence this
        # round, and record the resulting catalog change explicitly.
        current_cache_audit = _committed_cache_audit(
            state, ctx.cached_map, ctx.cache_audit,
        )
        _write_json(ctx.split_root / "cache_resume_audit.json", current_cache_audit)
        if not current_cache_audit["ok"]:
            raise RuntimeError(
                "iterative VQA cache changed a committed label or introduced an "
                "unresolved conflict after provider reload; no round was committed. "
                f"earliest invalid committed round="
                f"{current_cache_audit.get('earliest_invalid_committed_round')!r}; "
                f"see {ctx.split_root / 'cache_resume_audit.json'}"
            )
        _apply_cache_audit_migrations(state, current_cache_audit)
        cache_snapshot = (
            _selected_cache_snapshot(candidates, ctx.cached_map, ctx.cache_audit)
            if complete else None
        )
        provider_requested_unique, provider_request_count = _provider_request_totals(
            provider_request_history,
        )
        manifest = {"round": round_no, "stage": stage,
                    "run_identity_sha256": isolated_identity_hash if ctx.isolated else None,
                    "candidate_pool_size": len(available),
                    "train": pick["train"], "audit": pick["audit"], "roles": pick["roles"],
                    "reused": reused, "to_label": missing, "requested_to_label": requested_paths, "prequential_audit": preq,
                    "diagnostics": diagnostics, "complete": complete, "skipped_missing": [],
                    "cache_snapshot": cache_snapshot,
                    "cache_priority_policy": ctx.cache_audit.get("priority_policy"),
                    "provider_request_history": provider_request_history,
                    "provider_requested_unique": provider_requested_unique,
                    "provider_request_count": provider_request_count}
        for audit_field in (
            "selection_revision",
            "selection_contract_before_sha256",
            "selection_contract_after_sha256",
            "rejected_not_budgeted",
            "replacement_map",
            "content_rejection_audits",
        ):
            if audit_field in frozen_manifest:
                manifest[audit_field] = frozen_manifest[audit_field]
        manifest["requires_new_preflight_before_network"] = bool(
            missing and state.get("requires_repreflight")
        )
        _write_json(rpaths["manifest"], manifest)
        _write_json(rpaths["qa"] / "selected_train.json", pick["train"])
        _write_json(rpaths["qa"] / "selected_audit.json", pick["audit"])
        diag_rows = []
        role_lookup = {p: role for role, ps in pick["roles"].items() for p in ps}
        selected_scores = diagnostics.get("selected_scores", {})
        for p in candidates:
            score = selected_scores.get(p, {})
            diag_rows.append({"image": p, "candidate_type": role_lookup.get(p, "backfill"),
                              "reused_vqa": p in reused, "partition": "train" if p in pick["train"] else "audit",
                              "query_similarity": score.get("query_similarity"),
                              "attribute_scores": score.get("attribute_scores"),
                              "joint_score": score.get("joint_score"),
                              "joint_boundary_distance": score.get("joint_boundary_distance"),
                              "joint_boundary_score": score.get("joint_boundary_score"),
                              "committee_disagreement": score.get("committee_disagreement"),
                              "mmr_score": None, "labels": ctx.cached_map.get(p), "seed": H.SPLIT_SEED})
        _write_json(rpaths["qa"] / "selection_diagnostics.json", diag_rows)
        if args.iterative_dry_run:
            state["status"] = "dry_run_complete" if complete else "dry_run_pending_labels"
            state["pending_round"] = round_no
            _write_json(state_path, state)
            _write_json(ctx.split_root / "train_labeled_indices.json", selected_train)
            _write_json(ctx.split_root / "audit_labeled_indices.json", selected_audit)
            _write_report(ctx, state)
            print(f"  round {round_no}: cached={len(reused)} new={len(requested_paths)}. Dry-run manifest written; training skipped.")
            return
        if not complete:
            state["status"] = "awaiting_vqa"
            state["pending_round"] = round_no
            _write_json(state_path, state)
            _write_json(ctx.split_root / "train_labeled_indices.json", selected_train)
            _write_json(ctx.split_root / "audit_labeled_indices.json", selected_audit)
            _write_report(ctx, state)
            print(f"  round {round_no}: cached={len(reused)} new={len(missing)}. Manifest written; no training until labels are available.")
            return
        selected_train = _uniq(selected_train + pick["train"])
        selected_audit = _uniq(selected_audit + pick["audit"])
        stats = _label_stats(ctx.cached_map, selected_train)
        defer_round_eval = ctx.isolated or args.iterative_defer_round_evaluation
        fixed_eval = ({"status": "deferred_to_batch_final_evaluation"} if defer_round_eval else {
            "mlp_baseline": _eval_fixed(ctx, selected_train, "mlp_baseline"),
            "pu_ranking": _eval_fixed(ctx, selected_train, "pu_ranking"),
        })
        static_control = (
            {"status": "deferred_to_batch_final_evaluation"}
            if args.iterative_defer_round_evaluation
            else _static_control(ctx, len(selected_train))
        )
        rrecord = {**manifest, "n_train": len(selected_train), "n_audit": len(selected_audit),
                   "total_labeled": len(selected_train) + len(selected_audit),
                   "n_new": provider_request_count,
                   "provider_requested_unique": provider_requested_unique,
                   "provider_request_count": provider_request_count,
                   "label_stats": stats,
                   "audit_label_stats": _label_stats(ctx.cached_map, pick["audit"]),
                   "audit_prequential_metrics": _audit_prequential_metrics(ctx.cached_map, pick["audit"], preq),
                   "fixed_eval": fixed_eval,
                   "static_two_stage_control": static_control}
        rrecord["budget_decision"] = _audit_budget_decision(
            state["rounds"] + [rrecord], stats, rrecord["total_labeled"],
            args.iterative_max_labels, args.iterative_stop_at_labels,
            args.iterative_audit_min_gain,
        )
        state["rounds"].append(rrecord)
        state.pop("pending_round", None)
        state.pop("requires_repreflight", None)
        state.pop("pending_selection_contract_sha256", None)
        _write_json(state_path, state)
        _write_json(ctx.split_root / "train_labeled_indices.json", selected_train)
        _write_json(ctx.split_root / "audit_labeled_indices.json", selected_audit)
        if not ctx.isolated:
            control_file = rrecord["static_two_stage_control"].get("indices_file")
            _write_json(ctx.split_root / "static_control_indices.json",
                        _read_json(Path(control_file), []) if control_file else [])
        _write_json(ctx.embedding_root / f"round_{round_no:02d}" / "round_metrics.json", rrecord)
        _write_report(ctx, state)
        print(f"  round {round_no}: complete total={len(selected_train) + len(selected_audit)} train={len(selected_train)} audit={len(selected_audit)} joint_pos={stats['joint_positive']}")
        if getattr(args, "vqa_preflight_manifest", None) and requested_paths:
            # The approval describes only the currently frozen cache-missing
            # batch.  Stop before selection can expose a later round that was
            # not present in that approval.
            state["status"] = "preflight_batch_completed"
            state["approved_preflight_sha256"] = str(args.vqa_preflight_sha256)
            state["preflight_batch_round"] = int(round_no)
            _write_json(state_path, state)
            _write_report(ctx, state)
            print("  approved preflight batch complete; stopped before selecting another VQA batch")
            return
        if args.iterative_adaptive_budget and rrecord["budget_decision"]["action"] == "stop":
            state["status"] = "stopped_audit_budget_policy"
            break
    state["status"] = "completed" if len(state["rounds"]) >= target_rounds else state.get("status", "stopped")
    _write_json(state_path, state)
    _write_json(ctx.embedding_root / "final" / "final_state.json", state)
    _write_report(ctx, state)
    print(f"Iterative report: {ctx.report_root / 'iterative_vqa_report.md'}")


if __name__ == "__main__":
    raise SystemExit("Run through run_retrieval_harness.py --iterative-vqa")
