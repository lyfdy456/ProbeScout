"""Eight-probe bank caching, supervision identities and batch execution.

Fixed-Validation paper training uses scripts/train_probes.py. This module
also supplies the cache and acquisition-budget helpers used by the system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_retrieval_harness as H
from iterative_vqa_runner import (
    ISOLATED_EVALUATION_POLICY,
    ISOLATED_IDENTITY_SCHEMA,
    ISOLATED_SOURCE_POLICY,
    _frozen_split,
    _stable_json_hash,
)
from src.methods.probe_fusion import (
    average_precision,
    minmax,
    ranks_from_scores,
    rrf_fusion,
    soft_gate_mean,
    zscore,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs" / "main17_tasks.json"
DEFAULT_SUITE = ROOT / "configs" / "probe_suite.json"
DEFAULT_RUN_ROOT = ROOT / "outputs" / "runs" / "main17"
DEFAULT_PROBEBANK_ROOT = ROOT / "outputs" / "probebank"
PROTOCOL_VERSION = "probebank_paper_full_v2"
BACKBONE = "siglip"
ITERATIVE_STAGE = "iterative"
STAGES = ("random", "two_stage", ITERATIVE_STAGE)
ITERATIVE_ONLY_SOURCE_POLICY = "run-local-iterative-supervision-only-v1"
EXACT_STATIC_SOURCE_POLICY = "run-local-exact-static-prefix-only-v1"
STATIC_PREFIX_PROTOCOL = "static-ordered-prefix-permanent-audit-100-50-40-10-v1"
EMBEDDING_METHODS: tuple[str, ...] = ()
BASE_METHODS: tuple[str, ...] = ()
REPORTED_METHODS: tuple[str, ...] = ()
JOINT_VARIANTS: tuple[str, ...] = ()
FUSION_RECIPES: tuple[dict[str, Any], ...] = ()
POSTHOC_JOINT_RULES: tuple[str, ...] = ()
POSTHOC_BASE_METHODS: tuple[str, ...] = ()
POSTHOC_STAGE = ITERATIVE_STAGE
TEXT_FREE_ATTENTION_METHODS = {"attention_pooling"}
TEXT_QUERY_CACHE = ROOT / "outputs" / "backbone_text_query_cache"


def text_query_for_method(
    method: str,
    attr: str,
    text_queries: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if method in TEXT_FREE_ATTENTION_METHODS:
        return None
    if H.TRAINED_METHODS.get(method) in ("attn", "attn_ens"):
        if attr not in text_queries:
            raise ValueError(f"attention method {method} is missing the text query for {attr!r}")
        return text_queries[attr]
    return None


def cached_attribute_text_queries(
    attrs: list[str], backbone: str, model_name: str,
) -> dict[str, torch.Tensor]:
    """Encode an exact attribute list once and safely share it across jobs."""
    identity = json.dumps(
        {"attrs": attrs, "backbone": backbone, "model_name": model_name},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    TEXT_QUERY_CACHE.mkdir(parents=True, exist_ok=True)
    cache = TEXT_QUERY_CACHE / f"{key}.pt"
    lock = TEXT_QUERY_CACHE / f"{key}.lock"

    def load() -> dict[str, torch.Tensor]:
        try:
            payload = torch.load(cache, map_location="cpu", weights_only=True)
        except TypeError:  # compatibility with older PyTorch releases
            payload = torch.load(cache, map_location="cpu")
        if payload.get("identity") != identity:
            raise ValueError(f"text-query cache identity mismatch: {cache}")
        values = payload.get("queries", {})
        if set(values) != set(attrs):
            raise ValueError(f"text-query cache attribute mismatch: {cache}")
        return {attr: values[attr].detach().cpu() for attr in attrs}

    if cache.is_file():
        return load()

    owns_lock = False
    try:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            owns_lock = True
        except FileExistsError:
            deadline = time.time() + 900
            while time.time() < deadline:
                if cache.is_file():
                    return load()
                if lock.is_file() and time.time() - lock.stat().st_mtime > 900:
                    lock.unlink(missing_ok=True)
                    return cached_attribute_text_queries(attrs, backbone, model_name)
                time.sleep(2)
            raise TimeoutError(f"timed out waiting for text-query cache: {cache}")

        queries = H.encode_attribute_texts_for_backbone(attrs, backbone, model_name)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        torch.save({
            "identity": identity,
            "queries": {key: value.detach().cpu() for key, value in queries.items()},
        }, tmp)
        os.replace(tmp, cache)
        return load()
    finally:
        if owns_lock:
            lock.unlink(missing_ok=True)


def configure_suite(path: Path) -> dict[str, Any]:
    global EMBEDDING_METHODS, BASE_METHODS, REPORTED_METHODS, JOINT_VARIANTS
    global FUSION_RECIPES, POSTHOC_JOINT_RULES, POSTHOC_BASE_METHODS, POSTHOC_STAGE
    suite = json.loads(path.read_text(encoding="utf-8"))
    EMBEDDING_METHODS = tuple(suite["embedding_methods"])
    BASE_METHODS = tuple(suite["learned_methods"])
    unknown = set(BASE_METHODS) - set(H.TRAINED_METHODS)
    if unknown:
        raise ValueError(f"Methods outside the paper release: {sorted(unknown)}")
    if suite.get("recap_methods"):
        raise ValueError("This release supports only the eight paper probes")
    REPORTED_METHODS = BASE_METHODS
    JOINT_VARIANTS = tuple(suite["joint_rules"])
    expanded_fusion_recipes = []
    for recipe in suite["fusion_recipes"]:
        if recipe["aggregation"] == "pool_baseline_matrix":
            pool_joint_rules = list(
                recipe.get(
                    "member_joint_rules",
                    [recipe.get("member_joint_rule", "product")],
                )
            )
            for pool in recipe["pools"]:
                for pool_aggregation in recipe["aggregations"]:
                    for pool_joint_rule in pool_joint_rules:
                        expanded = {
                            key: value for key, value in recipe.items()
                            if key not in {"pools", "aggregations", "member_joint_rules"}
                        }
                        suffix = (
                            f"_{pool_joint_rule}"
                            if "member_joint_rules" in recipe else ""
                        )
                        expanded.update({
                            "id": (
                                f"{recipe['id_prefix']}_{pool['id']}_"
                                f"{pool_aggregation}{suffix}"
                            ),
                            "aggregation": "pool_baseline",
                            "pool_id": pool["id"],
                            "pool_strategy": pool["strategy"],
                            "pool_size": int(pool["size"]),
                            "pool_aggregation": pool_aggregation,
                            "pool_joint_rule": pool_joint_rule,
                            "joint_rule": pool_joint_rule,
                            "members": list(pool.get("members", recipe["members"])),
                        })
                        expanded_fusion_recipes.append(expanded)
            continue
        if recipe["aggregation"] == "cross_method_softgate_matrix":
            for gate_strategy in recipe["gate_strategies"]:
                for joint_rule in recipe["joint_rules"]:
                    expanded = {
                        key: value for key, value in recipe.items()
                        if key not in {"gate_strategies", "joint_rules"}
                    }
                    expanded.update({
                        "id": f"{recipe['id_prefix']}_{gate_strategy}_{joint_rule}",
                        "aggregation": "cross_method_softgate",
                        "gate_strategy": gate_strategy,
                        "joint_rule": joint_rule,
                    })
                    expanded_fusion_recipes.append(expanded)
            continue
        if recipe["aggregation"] != "bounded_linear_matrix":
            expanded_fusion_recipes.append(recipe)
            continue
        for training_strategy in recipe["training_strategies"]:
            for joint_rule in recipe["joint_rules"]:
                expanded = {
                    key: value for key, value in recipe.items()
                    if key not in {"training_strategies", "joint_rules"}
                }
                expanded.update({
                    "id": f"{recipe['id_prefix']}_{training_strategy}_{joint_rule}",
                    "aggregation": "bounded_linear",
                    "training_strategy": training_strategy,
                    "joint_rule": joint_rule,
                })
                expanded_fusion_recipes.append(expanded)
    FUSION_RECIPES = tuple(expanded_fusion_recipes)
    POSTHOC_JOINT_RULES = tuple(suite.get("posthoc_joint_rules", ()))
    POSTHOC_BASE_METHODS = tuple(suite.get("posthoc_base_methods", ()))
    POSTHOC_STAGE = str(suite.get("posthoc_stage", ITERATIVE_STAGE))
    missing = [method for method in BASE_METHODS if method not in H.TRAINED_METHODS]
    if missing:
        raise ValueError(f"method suite contains unavailable trained methods: {missing}")
    missing_embedding = [method for method in EMBEDDING_METHODS if method not in H.EMB_METHODS]
    if missing_embedding:
        raise ValueError(f"method suite contains unavailable embedding methods: {missing_embedding}")
    if set(POSTHOC_JOINT_RULES) - {"bounded_residual_h16"}:
        raise ValueError(f"unknown posthoc joint rules: {POSTHOC_JOINT_RULES}")
    if set(POSTHOC_BASE_METHODS) - set(BASE_METHODS):
        raise ValueError("posthoc_base_methods must be trained methods in this suite")
    if POSTHOC_JOINT_RULES and not POSTHOC_BASE_METHODS:
        raise ValueError("posthoc_joint_rules requires posthoc_base_methods")
    if POSTHOC_JOINT_RULES or POSTHOC_BASE_METHODS:
        raise ValueError("ReCAP-derived post-hoc learned Joint is disabled for new runs")
    allowed_joint = {
        "product", "zmean", "zmean_minmix_b025", "zmean_minmix", "lower_semideviation",
        "bounded_rms_deficit", "adaptive_mean_softmin", "attr_rrf",
        "train_calibrator", "joint_logreg",
    }
    if not set(JOINT_VARIANTS) <= allowed_joint:
        raise ValueError(f"unknown joint rules: {sorted(set(JOINT_VARIANTS) - allowed_joint)}")
    recipe_ids = [recipe["id"] for recipe in FUSION_RECIPES]
    if len(recipe_ids) != len(set(recipe_ids)):
        raise ValueError("fusion recipe IDs must be unique")
    generated_ids = set(EMBEDDING_METHODS) | set(BASE_METHODS)
    generated_ids.update(
        f"{method}__{rule}"
        for method in BASE_METHODS
        for rule in JOINT_VARIANTS
        if rule != "product"
    )
    generated_ids.add("pu_ranking_calib")
    for recipe in FUSION_RECIPES:
        if recipe["aggregation"] not in {
            "zmean", "rrf", "bounded_linear", "cross_method_softgate",
            "pool_baseline", "r_softgate",
        }:
            raise ValueError(f"unknown fusion aggregation in {recipe['id']}")
        member_joint_rule = recipe.get("member_joint_rule", "product")
        if member_joint_rule not in JOINT_VARIANTS:
            raise ValueError(
                f"fusion {recipe['id']} uses unavailable member joint rule: "
                f"{member_joint_rule}"
            )
        unavailable = set(resolve_fusion_members(recipe)) - generated_ids
        if unavailable:
            raise ValueError(f"fusion {recipe['id']} has unavailable members: {sorted(unavailable)}")
        if recipe["aggregation"] == "bounded_linear":
            non_probe_members = set(recipe["members"]) - set(BASE_METHODS)
            if non_probe_members:
                raise ValueError(
                    f"bounded-linear fusion {recipe['id']} accepts trained probes only: "
                    f"{sorted(non_probe_members)}"
                )
            if recipe.get("training_strategy") not in {
                "method_joint", "shared_attr_joint_loss", "per_attr",
            }:
                raise ValueError(f"unknown bounded-linear training strategy in {recipe['id']}")
            if recipe.get("joint_rule") not in {"product", "zmean", "elu_mean"}:
                raise ValueError(f"unknown bounded-linear joint rule in {recipe['id']}")
        if recipe["aggregation"] == "cross_method_softgate":
            if recipe.get("gate_strategy") not in {"joint_score", "probe_level"}:
                raise ValueError(f"unknown cross-method SoftGate strategy in {recipe['id']}")
            if recipe.get("joint_rule") not in {"product", "zmean", "elu_mean"}:
                raise ValueError(f"unknown cross-method SoftGate joint rule in {recipe['id']}")
        if recipe["aggregation"] == "r_softgate":
            non_probe_members = set(recipe["members"]) - set(BASE_METHODS)
            if non_probe_members:
                raise ValueError(
                    f"R-SoftGate {recipe['id']} accepts trained probes only: "
                    f"{sorted(non_probe_members)}"
                )
            if recipe.get("joint_rule", "product") != "product":
                raise ValueError(f"R-SoftGate {recipe['id']} must multiply attribute gates")
            if recipe.get("loss", "binary_cross_entropy") not in {
                "binary_cross_entropy", "joint_pairwise_ranking",
            }:
                raise ValueError(f"unknown R-SoftGate loss in {recipe['id']}")
            if recipe.get("probe_aggregation", "mean") not in {"mean", "min", "max"}:
                raise ValueError(f"unknown R-SoftGate probe aggregation in {recipe['id']}")
            if float(recipe.get("ranking_temperature", 0.1)) <= 0.0:
                raise ValueError(f"invalid R-SoftGate ranking temperature in {recipe['id']}")
            top_k = recipe.get("top_k", "all")
            if top_k != "all":
                try:
                    top_k = int(top_k)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid R-SoftGate top_k in {recipe['id']}") from error
                if top_k < 1 or top_k > len(recipe["members"]):
                    raise ValueError(f"invalid R-SoftGate top_k in {recipe['id']}: {top_k}")
            expected_size = len(recipe["members"]) if top_k == "all" else top_k
            if int(recipe.get("pool_size", expected_size)) != expected_size:
                raise ValueError(f"R-SoftGate pool_size/top_k mismatch in {recipe['id']}")
            theta_min = float(recipe.get("theta_min", 0.0))
            theta_max = float(recipe.get("theta_max", 1.0))
            if not 0.0 <= theta_min <= theta_max <= 1.0:
                raise ValueError(f"invalid R-SoftGate theta bounds in {recipe['id']}")
            t_min = float(recipe.get("temperature_min", 0.01))
            t_max = float(recipe.get("temperature_max", 1.0))
            if not 0.0 < t_min <= t_max:
                raise ValueError(f"invalid R-SoftGate temperature bounds in {recipe['id']}")
        if recipe["aggregation"] == "pool_baseline":
            if recipe.get("pool_strategy") not in {
                "fixed_all", "fixed_curated", "elite_train_ap", "coverage_train_topk",
            }:
                raise ValueError(f"unknown pool strategy in {recipe['id']}")
            if recipe.get("pool_aggregation") not in {"rrf", "zavg"}:
                raise ValueError(f"unknown pool aggregation in {recipe['id']}")
            if recipe.get("pool_joint_rule", "product") not in {
                "product", "zmean", "elu_mean",
            }:
                raise ValueError(f"unknown pool joint rule in {recipe['id']}")
            pool_size = int(recipe.get("pool_size", 0))
            if pool_size < 1 or pool_size > len(recipe["members"]):
                raise ValueError(f"invalid pool size in {recipe['id']}: {pool_size}")
            if recipe["pool_strategy"] in {"fixed_all", "fixed_curated"}:
                if pool_size != len(recipe["members"]):
                    raise ValueError(
                        f"fixed pool {recipe['id']} must use every declared member"
                    )
    return suite


def resolve_fusion_members(recipe: dict[str, Any]) -> list[str]:
    """Resolve one fixed Joint rule for every trained member in a recipe."""
    rule = recipe.get("member_joint_rule", "product")
    return [
        member if member not in BASE_METHODS or rule == "product"
        else f"{member}__{rule}"
        for member in recipe["members"]
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def stable_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iterative_only_probebank_root(
    bank_root: Path, run_identity_sha256: str, order: int, dataset: str, task: str,
) -> Path:
    """Use a run-identity namespace that no legacy logical-stage alias can hit."""

    if (len(run_identity_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in run_identity_sha256)):
        raise ValueError("invalid iterative run identity hash for ProbeBank namespace")
    return (
        bank_root / "iterative_only" / run_identity_sha256 / "task_isolated"
        / f"{order:03d}_{dataset}_{task}"
    )


def _tensor_hash(value: torch.Tensor | None) -> str | None:
    """Stable hash for the exact attribute text query used by an attention probe."""
    if value is None:
        return None
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _portable_source_signature(value: Any) -> Any:
    """Describe feature files without binding caches to one machine's root path."""
    if isinstance(value, dict):
        return {key: _portable_source_signature(item) for key, item in sorted(value.items())}
    path = Path(str(value))
    signature: dict[str, Any] = {"name": path.name}
    if path.is_file():
        signature["size"] = int(path.stat().st_size)
    return signature


def attention_feature_context(
    method: str,
    patch_meta: dict[str, Any] | None,
    text_query: torch.Tensor | None,
) -> dict[str, Any] | None:
    """Cache identity for patch-token methods; pooled probes stay v2-compatible."""
    kind = H.TRAINED_METHODS.get(method)
    if kind not in ("attn", "attn_ens"):
        return None
    if patch_meta is None:
        raise ValueError(f"attention method {method} is missing patch metadata")
    source_signature = _portable_source_signature(patch_meta["patch_sources"])
    payload = json.dumps(
        source_signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return {
        "model_kind": kind,
        "patch_shape": list(patch_meta["patch_shape"]),
        "patch_dtype": str(patch_meta["patch_dtype"]),
        "patch_source_hash": stable_hash([payload]),
        "text_query_hash": _tensor_hash(text_query),
    }


def supervision_hash(selected: list[str], labels_map: dict[str, dict],
                     attrs: list[str]) -> str:
    """Hash the complete, validated supervision set used by every consumer.

    The hash deliberately includes the ordered attribute schema and every
    selected image/label tuple.  Missing or non-binary labels are errors; they
    must never be converted into a smaller, apparently valid supervision set.
    """
    rows = [json.dumps(
        ["supervision_schema", "complete_selected_labels_v1", attrs],
        ensure_ascii=False, separators=(",", ":"),
    )]
    for path in sorted(selected):
        if path not in labels_map:
            raise ValueError(f"cannot hash missing supervision label: {path}")
        values = [labels_map[path].get(attr) for attr in attrs]
        if any(value not in (0, 1) for value in values):
            raise ValueError(
                f"cannot hash incomplete/non-binary supervision label: {path} -> {values}"
            )
        rows.append(json.dumps(
            [path, values], ensure_ascii=False, separators=(",", ":"),
        ))
    return stable_hash(rows)


def legacy_supervision_hash(selected: list[str], labels_map: dict[str, dict],
                            attrs: list[str]) -> str:
    """Reproduce the pre-schema digest for exact legacy-cache validation.

    This is intentionally not used as the identity of any new run.  It only
    proves that an older ProbeBank entry saw the same ordered per-image binary
    labels before ``complete_selected_labels_v1`` was added to the digest.
    """
    rows = []
    for path in sorted(set(selected)):
        if path not in labels_map:
            raise ValueError(f"cannot hash missing legacy supervision label: {path}")
        values = [labels_map[path].get(attr) for attr in attrs]
        if any(value not in (0, 1) for value in values):
            raise ValueError(
                f"cannot hash incomplete/non-binary legacy label: {path} -> {values}"
            )
        rows.append(json.dumps(
            [path, values], ensure_ascii=False, separators=(",", ":"),
        ))
    return stable_hash(rows)


def _unique_existing_sources(paths: list[Path]) -> list[Path]:
    """Return existing label sources once, preserving precedence order."""
    unique: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        key = os.path.normcase(str(path.resolve()))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_label_sources(
    sources: list[Path], vqa_to_key, attrs: list[str],
) -> tuple[dict[str, dict[str, int]], set[str]]:
    """Merge source rows at attribute granularity; later valid values win."""
    merged: dict[str, dict[str, int]] = {}
    present: set[str] = set()
    for source in _unique_existing_sources(sources):
        source_map, _ = H.load_vqa(source, vqa_to_key, attrs)
        for image, row in source_map.items():
            present.add(image)
            target = merged.setdefault(image, {})
            for attr in attrs:
                value = row.get(attr)
                if value is None:
                    continue
                value = int(value)
                if value not in (0, 1):
                    raise ValueError(
                        f"non-binary VQA label in {source}: image={image}, "
                        f"attribute={attr!r}, value={value!r}"
                    )
                target[attr] = value
    return merged, present


def resolve_training_supervision(
    *,
    selected: list[str],
    reused_sources: list[Path],
    new_sources: list[Path],
    vqa_to_key,
    attrs: list[str],
    audit_path: Path,
    selected_manifest: Path | None,
) -> tuple[list[str], dict[str, dict[str, int]], list[dict[str, Any]], str, dict[str, Any]]:
    """Resolve the one strict supervision set used by train, resume, and fusion.

    Historical/cache labels are loaded first.  Labels produced inside the
    current run are then overlaid per attribute.  A selected image may never be
    silently discarded: a missing image or unresolved attribute writes a
    failed audit and raises before any probe is read, trained, or fused.
    """
    if not isinstance(selected, list) or not all(isinstance(path, str) for path in selected):
        raise TypeError("selected supervision must be a JSON list of image paths")
    duplicate_paths = [path for path, count in Counter(selected).items() if count > 1]
    selected_unique = list(dict.fromkeys(selected))

    reused_sources = _unique_existing_sources(reused_sources)
    new_sources = _unique_existing_sources(new_sources)
    new_source_keys = {os.path.normcase(str(path.resolve())) for path in new_sources}
    reused_sources = [
        path for path in reused_sources
        if os.path.normcase(str(path.resolve())) not in new_source_keys
    ]
    reused_map, reused_present = _load_label_sources(
        reused_sources, vqa_to_key, attrs,
    )
    new_map, new_present = _load_label_sources(new_sources, vqa_to_key, attrs)

    labels_map = {path: dict(row) for path, row in reused_map.items()}
    for path, row in new_map.items():
        labels_map.setdefault(path, {}).update(row)

    missing_paths: list[str] = []
    incomplete_attributes: dict[str, list[str]] = {}
    matched_paths: list[str] = []
    for path in selected_unique:
        if path not in reused_present and path not in new_present:
            missing_paths.append(path)
            continue
        unresolved = [attr for attr in attrs if labels_map.get(path, {}).get(attr) not in (0, 1)]
        if unresolved:
            missing_paths.append(path)
            incomplete_attributes[path] = unresolved
            continue
        matched_paths.append(path)

    matched_set = set(matched_paths)
    new_count = sum(path in new_present for path in matched_paths)
    reused_count = sum(path not in new_present and path in reused_present for path in matched_paths)
    overlap_count = sum(
        path in new_present and path in reused_present for path in matched_paths
    )
    attribute_counts = {}
    for attr in attrs:
        positive = sum(labels_map[path][attr] == 1 for path in matched_paths)
        attribute_counts[attr] = {
            "positive": positive,
            "negative": len(matched_paths) - positive,
        }
    joint_positive = sum(
        all(labels_map[path][attr] == 1 for attr in attrs)
        for path in matched_paths
    )
    digest = (
        supervision_hash(matched_paths, labels_map, attrs)
        if not missing_paths and not duplicate_paths and matched_paths else None
    )
    audit: dict[str, Any] = {
        "status": (
            "valid" if digest is not None else "invalid"
        ),
        "selected_manifest": str(selected_manifest.resolve()) if selected_manifest else None,
        "selected_count_target": len(selected_unique),
        "selected_count_raw": len(selected),
        "matched_count": len(matched_paths),
        "reused_label_count": reused_count,
        "new_label_count": new_count,
        "overlap_label_count": overlap_count,
        "missing_count": len(missing_paths),
        "duplicate_selected_count": len(duplicate_paths),
        "missing_paths": missing_paths,
        "incomplete_attributes": incomplete_attributes,
        "duplicate_selected_paths": duplicate_paths,
        "attribute_counts": attribute_counts,
        "joint_counts": {
            "positive": joint_positive,
            "negative": len(matched_paths) - joint_positive,
        },
        "supervision_hash": digest,
        "reused_sources": [str(path.resolve()) for path in reused_sources],
        "new_sources": [str(path.resolve()) for path in new_sources],
    }
    write_json(audit_path, audit)

    if duplicate_paths:
        raise RuntimeError(
            "selected supervision contains duplicate paths; see audit "
            f"{audit_path}: {duplicate_paths}"
        )
    if missing_paths:
        detail = "\n".join(
            f"  - {path}"
            + (f" (missing attributes: {incomplete_attributes[path]})"
               if path in incomplete_attributes else "")
            for path in missing_paths
        )
        raise RuntimeError(
            f"strict supervision validation failed: {len(missing_paths)}/"
            f"{len(selected_unique)} selected images lack complete labels. "
            f"Audit: {audit_path}\n{detail}"
        )
    if not matched_set:
        raise RuntimeError(f"supervision has no selected records; audit: {audit_path}")

    labels = [{"image": path, **labels_map[path]} for path in selected_unique]
    print(
        "  [supervision] "
        f"target={len(selected_unique)} matched={len(matched_paths)} "
        f"reused={reused_count} new={new_count} missing=0 "
        f"hash={digest[:12]}"
    )
    return selected_unique, labels_map, labels, digest, audit


def validate_frozen_test_isolation(
    *,
    selected: list[str],
    gallery: list[str],
    frozen_test: list[str],
) -> dict[str, Any]:
    """Fail before training if active supervision crosses the Test boundary."""

    gallery_ids = set(gallery)
    unknown = sorted(set(selected).difference(gallery_ids))
    if unknown:
        raise RuntimeError(
            "active training supervision contains rows outside the gallery: "
            f"{unknown[:3]}"
        )
    overlap = sorted(set(selected).intersection(frozen_test))
    if overlap:
        raise RuntimeError(
            "active training supervision overlaps Frozen Test; repair/freeze the "
            f"evaluation split before fitting: count={len(overlap)}, first={overlap[:3]}"
        )
    return {
        "gallery_membership_valid": True,
        "frozen_test_overlap_count": 0,
        "frozen_test_row_count": len(frozen_test),
        "frozen_test_ids_sha256": stable_hash(list(frozen_test)),
        "frozen_test_sequence_sha256_nul": stable_hash(list(frozen_test)),
        "test_selection_sha256": stable_hash(list(frozen_test)),
    }


def iterative_selected_manifest(
    task_out: Path, canonical_vqa_dir: Path,
) -> Path | None:
    """Prefer the current run's complete cumulative selection, then canonical."""
    roots = [task_out / "vqa" / ITERATIVE_STAGE, canonical_vqa_dir]
    for root in roots:
        exact = root / "split" / "train_labeled_indices.json"
        if exact.is_file():
            return exact
        candidates = sorted(root.rglob("train_labeled_indices.json")) if root.is_dir() else []
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(
                f"ambiguous iterative train_labeled_indices.json under {root}: {candidates}"
            )
    fallback = sorted(
        canonical_vqa_dir.parent.glob("iterative*/split/train_labeled_indices.json")
    ) if canonical_vqa_dir.parent.is_dir() else []
    if len(fallback) == 1:
        return fallback[0]
    if len(fallback) > 1:
        raise RuntimeError(
            "ambiguous canonical iterative supervision stages; configure "
            f"supervision_stages explicitly: {fallback}"
        )
    return None


def static_budget_counts(logical_budget: int) -> tuple[int, int]:
    """Return the train/permanent-audit counts used by the fair static curve."""

    if logical_budget == 50:
        return 50, 0
    if logical_budget < 100 or (logical_budget - 100) % 50:
        raise RuntimeError(
            f"exact static logical budget must be 50 or 100+n*50, got {logical_budget}"
        )
    rounds = (logical_budget - 100) // 50
    return 100 + 40 * rounds, 10 * rounds


def static_budget_partition(
    ordered: list[str], logical_budget: int,
) -> tuple[list[str], list[str]]:
    train_count, audit_count = static_budget_counts(logical_budget)
    if len(ordered) < logical_budget:
        raise RuntimeError(
            f"exact static source has {len(ordered)} rows, below budget {logical_budget}"
        )
    if logical_budget == 50:
        return list(ordered[:50]), []
    train = list(ordered[:100])
    audit: list[str] = []
    for start in range(100, logical_budget, 50):
        block = ordered[start : start + 50]
        if len(block) != 50:
            raise RuntimeError(f"incomplete exact static block at offset {start}")
        train.extend(block[:40])
        audit.extend(block[40:])
    if len(train) != train_count or len(audit) != audit_count:
        raise AssertionError("exact static partition count drift")
    return train, audit


def _json_string_list(path: Path, label: str) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if not isinstance(payload, list) or not all(
        isinstance(value, str) and value for value in payload
    ):
        raise TypeError(f"{label} must be a JSON list of non-empty strings: {path}")
    if len(payload) != len(set(payload)):
        raise RuntimeError(f"{label} contains duplicate paths: {path}")
    return list(payload)


def exact_static_source_directory(task, manifest_path: Path, default_task_root: Path,
                                  source_stage: str) -> Path:
    """Honor the explicit materializer source root without relocating image data."""
    if not source_stage or Path(source_stage).is_absolute() or any(
        separator in source_stage for separator in ("/", "\\")
    ) or source_stage in (".", ".."):
        raise RuntimeError("exact static source stage must be a directory name")
    override = task.get("task_root")
    if isinstance(override, str) and override:
        root = Path(override)
        if not root.is_absolute():
            root = manifest_path.resolve().parent / root
    else:
        root = default_task_root
    return (root / "supervision" / source_stage).resolve()


def exact_static_supervision_inputs(
    task_out: Path,
    canonical_vqa_dir: Path,
    *,
    expected_order: int,
    expected_dataset: str,
    expected_task: str,
    expected_stage: str,
    expected_source_stage: str,
    expected_attributes: list[str],
    logical_budget: int,
    expected_train_count: int,
) -> tuple[Path, list[str], list[str], list[Path], dict[str, Any]]:
    """Resolve a hash-bound run-local static prefix and one canonical label source.

    Unlike the legacy static path, this mode never scans task VQA history.  The
    selected fitting IDs live below the current run and the only accepted label
    file is ``canonical_results`` declared by the configured strict source
    stage.  Both are bound by the materializer provenance.
    """

    if expected_stage not in ("random", "two_stage"):
        raise RuntimeError(f"exact static mode does not support stage {expected_stage!r}")
    split_root = task_out / "vqa" / expected_stage / "split"
    selected_path = split_root / "train_labeled_indices.json"
    audit_path = split_root / "permanent_audit_indices.json"
    provenance_path = split_root / "static_budget_prefix_provenance.json"
    selected = _json_string_list(selected_path, "exact static train manifest")
    permanent_audit = _json_string_list(audit_path, "exact static permanent-audit manifest")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"exact static prefix provenance is missing: {provenance_path}"
        ) from error
    if not isinstance(provenance, dict):
        raise TypeError(f"exact static provenance must be an object: {provenance_path}")
    identity = provenance.get("task")
    expected_identity = {
        "order": int(expected_order),
        "dataset": expected_dataset,
        "task": expected_task,
        "output_id": f"{int(expected_order):03d}_{expected_dataset}_{expected_task}",
    }
    if identity != expected_identity:
        raise RuntimeError(
            f"exact static task identity mismatch: {identity!r} != {expected_identity!r}"
        )
    if provenance.get("protocol_id") != STATIC_PREFIX_PROTOCOL:
        raise RuntimeError(f"exact static protocol mismatch: {provenance_path}")
    if provenance.get("stage") != expected_stage:
        raise RuntimeError(f"exact static logical stage mismatch: {provenance_path}")
    if provenance.get("source_stage") != expected_source_stage:
        raise RuntimeError(f"exact static source stage mismatch: {provenance_path}")
    if int(provenance.get("logical_budget", -1)) != logical_budget:
        raise RuntimeError(f"exact static logical budget mismatch: {provenance_path}")

    train_count, audit_count = static_budget_counts(logical_budget)
    if expected_train_count != train_count:
        raise RuntimeError(
            f"--sup-size {expected_train_count} does not equal exact static train count "
            f"{train_count} for logical budget {logical_budget}"
        )
    materialized = provenance.get("materialized")
    if not isinstance(materialized, dict):
        raise RuntimeError(f"exact static materialized contract is missing: {provenance_path}")
    expected_values = {
        "train_count": train_count,
        "audit_count": audit_count,
        "observed_total_labeled_count": logical_budget,
        "logical_budget_shortfall": 0,
    }
    for key, expected in expected_values.items():
        if int(materialized.get(key, -1)) != expected:
            raise RuntimeError(
                f"exact static {key} mismatch: {materialized.get(key)!r} != {expected}"
            )
    if len(selected) != train_count or len(permanent_audit) != audit_count:
        raise RuntimeError("exact static manifest lengths do not match the budget contract")
    if set(selected).intersection(permanent_audit):
        raise RuntimeError("exact static train and permanent-audit manifests overlap")
    if materialized.get("no_duplicate_train_ids") is not True:
        raise RuntimeError("exact static provenance lacks the duplicate-free proof")
    if materialized.get("train_audit_disjoint") is not True:
        raise RuntimeError("exact static provenance lacks the train/audit-disjoint proof")
    if materialized.get("train_selection_sha256") != stable_hash(selected):
        raise RuntimeError(f"exact static train hash mismatch: {selected_path}")
    if materialized.get("audit_selection_sha256") != stable_hash(permanent_audit):
        raise RuntimeError(f"exact static audit hash mismatch: {audit_path}")

    source = provenance.get("source")
    if not isinstance(source, dict):
        raise RuntimeError(f"exact static source contract is missing: {provenance_path}")
    source_dir = canonical_vqa_dir.resolve()

    def required_source_path(key: str, expected: Path) -> Path:
        value = source.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"exact static source field {key!r} is missing")
        observed = Path(value).resolve()
        if observed != expected.resolve():
            raise RuntimeError(
                f"exact static source path mismatch for {key}: {observed} != {expected.resolve()}"
            )
        try:
            observed.relative_to(source_dir)
        except ValueError as error:
            raise RuntimeError(f"exact static source escapes configured stage: {observed}") from error
        if not observed.is_file():
            raise FileNotFoundError(f"exact static source file is missing: {observed}")
        return observed

    stage_manifest_path = required_source_path(
        "stage_manifest", source_dir / "_stage_manifest.json"
    )
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(stage_manifest, dict)
        or stage_manifest.get("stage") != expected_stage
        or stage_manifest.get("stage_id") != expected_source_stage
        or stage_manifest.get("attributes") != expected_attributes
    ):
        raise RuntimeError(f"exact static stage manifest identity drifted: {stage_manifest_path}")
    canonical_name = stage_manifest.get("canonical_results", "canonical_complete_results.jsonl")
    if not isinstance(canonical_name, str) or not canonical_name:
        raise RuntimeError(f"exact static canonical_results is invalid: {stage_manifest_path}")
    canonical_relative = Path(canonical_name)
    if canonical_relative.is_absolute():
        raise RuntimeError("exact static canonical_results must be stage-relative")
    canonical_path = (source_dir / canonical_relative).resolve()
    try:
        canonical_path.relative_to(source_dir)
    except ValueError as error:
        raise RuntimeError(f"exact static canonical source escapes stage: {canonical_path}") from error
    canonical_path = required_source_path("canonical_results", canonical_path)
    source_selected = required_source_path(
        "selected_manifest", source_dir / "selected_candidates_500.json"
    )
    hash_bindings = {
        "stage_manifest_sha256": stage_manifest_path,
        "canonical_results_sha256": canonical_path,
        "selected_manifest_sha256": source_selected,
    }
    for key, path in hash_bindings.items():
        if source.get(key) != file_sha256(path):
            raise RuntimeError(f"exact static source hash mismatch for {path}")
    ordered = _json_string_list(source_selected, "exact static ordered source")
    if len(ordered) != int(source.get("source_prefix_count", -1)) or len(ordered) != 500:
        raise RuntimeError("exact static ordered source must contain exactly 500 rows")
    if source.get("selected_sequence_sha256") != stable_hash(ordered):
        raise RuntimeError(f"exact static source sequence hash mismatch: {source_selected}")
    expected_train, expected_audit = static_budget_partition(ordered, logical_budget)
    if selected != expected_train or permanent_audit != expected_audit:
        raise RuntimeError(
            "run-local exact static manifests do not match the source budget partition"
        )

    contract = {
        "source_policy": EXACT_STATIC_SOURCE_POLICY,
        "protocol_id": STATIC_PREFIX_PROTOCOL,
        "logical_stage": expected_stage,
        "source_stage": expected_source_stage,
        "logical_budget": logical_budget,
        "train_count": train_count,
        "audit_count": audit_count,
        "selected_manifest": str(selected_path.resolve()),
        "permanent_audit_manifest": str(audit_path.resolve()),
        "provenance": str(provenance_path.resolve()),
        "provenance_sha256": file_sha256(provenance_path),
        "canonical_results": str(canonical_path),
        "canonical_results_sha256": file_sha256(canonical_path),
        "ignored_task_vqa_history": True,
    }
    return selected_path, selected, permanent_audit, [canonical_path], contract


def iterative_run_supervision_inputs(
    task_out: Path,
    *,
    expected_dataset: str | None = None,
    expected_task: str | None = None,
    expected_backbone: str | None = None,
    expected_attributes: list[str] | None = None,
    expected_paths: list[str] | None = None,
    expected_query_indices: list[int] | None = None,
    expected_train_pool: list[str] | None = None,
    expected_candidate_test: list[str] | None = None,
) -> tuple[Path, list[str], list[Path], dict[str, Any]]:
    """Resolve one isolated iterative run without scanning task history.

    The exact cumulative training list, round manifests, round state, and VQA
    result JSONLs must all live below ``task_out/vqa/iterative`` and agree.
    Random, two-stage, canonical task supervision, and sibling iterative stages
    are deliberately outside this contract and are never discovered.
    """

    run_root = (task_out / "vqa" / ITERATIVE_STAGE).resolve()
    qa_root = run_root / "qa"
    split_root = run_root / "split"
    root_manifest_path = qa_root / "manifest.json"
    state_path = split_root / "round_state.json"
    selected_path = split_root / "train_labeled_indices.json"
    for label, path in (
        ("isolated iterative root manifest", root_manifest_path),
        ("isolated iterative round state", state_path),
        ("isolated iterative selected manifest", selected_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if root_manifest.get("source_policy") != ISOLATED_SOURCE_POLICY:
        raise RuntimeError(
            "iterative run root does not declare the run-local-only VQA source policy; "
            f"use a new isolated work root: {run_root}"
        )
    identity = root_manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("schema") != ISOLATED_IDENTITY_SCHEMA:
        raise RuntimeError(
            f"isolated iterative root lacks a bound v1 run identity: {root_manifest_path}"
        )
    run_identity_hash = _stable_json_hash(identity)
    if root_manifest.get("run_identity_sha256") != run_identity_hash:
        raise RuntimeError(f"isolated iterative run identity hash mismatch: {root_manifest_path}")
    identity_expectations = {
        "source_policy": ISOLATED_SOURCE_POLICY,
        "dataset": expected_dataset,
        "task": expected_task,
        "backbone": expected_backbone,
        "attributes": expected_attributes,
        "work_root": str(run_root),
        "evaluation_split_policy": ISOLATED_EVALUATION_POLICY,
    }
    for key, expected in identity_expectations.items():
        if expected is not None and identity.get(key) != expected:
            raise RuntimeError(
                f"isolated iterative identity mismatch: {key}={identity.get(key)!r}, "
                f"expected {expected!r}"
            )
    split_identity = identity.get("split")
    budget_identity = identity.get("budget")
    if not isinstance(split_identity, dict) or not isinstance(budget_identity, dict):
        raise RuntimeError("isolated iterative identity lacks split/budget bindings")
    required_budget = {"initial", "round", "train", "audit", "max", "default_stop",
                       "max_refine_rounds", "adaptive", "audit_min_gain"}
    if not required_budget.issubset(budget_identity):
        raise RuntimeError("isolated iterative identity has an incomplete budget binding")
    if root_manifest.get("budget") != budget_identity:
        raise RuntimeError("isolated iterative root budget drifted from its run identity")
    if root_manifest.get("evaluation_split_policy") != identity.get("evaluation_split_policy"):
        raise RuntimeError("isolated iterative root split policy drifted from its run identity")
    if root_manifest.get("isolated_work_root") != identity.get("work_root"):
        raise RuntimeError("isolated iterative root path drifted from its run identity")
    if expected_paths is not None:
        if identity.get("records_hash") != stable_hash(expected_paths):
            raise RuntimeError("isolated iterative database records changed")
        if identity.get("record_count") != len(expected_paths):
            raise RuntimeError("isolated iterative database record count changed")
    if expected_query_indices is not None and expected_paths is not None:
        expected_query_paths = [expected_paths[int(index)] for index in expected_query_indices]
        if identity.get("query_indices") != [int(index) for index in expected_query_indices]:
            raise RuntimeError("isolated iterative query indices changed")
        if identity.get("query_paths_hash") != stable_hash(expected_query_paths):
            raise RuntimeError("isolated iterative query records changed")
    if expected_train_pool is not None and split_identity.get("train_pool_hash") != stable_hash(expected_train_pool):
        raise RuntimeError("isolated iterative train pool changed")
    if (expected_candidate_test is not None
            and split_identity.get("candidate_test_hash") != stable_hash(expected_candidate_test)):
        raise RuntimeError("isolated iterative candidate Test split changed")
    if state.get("source_policy") != ISOLATED_SOURCE_POLICY:
        raise RuntimeError(
            "iterative round state does not declare the run-local-only VQA source policy; "
            f"use a new isolated work root: {run_root}"
        )
    if state.get("run_identity_sha256") != run_identity_hash:
        raise RuntimeError("iterative round state belongs to a different run identity")
    stage_name = state.get("stage")
    if not isinstance(stage_name, str) or not stage_name.startswith("iterative_vqa_100_50_v"):
        raise RuntimeError(f"invalid isolated iterative stage in {state_path}: {stage_name!r}")
    if root_manifest.get("version") != stage_name:
        raise RuntimeError(
            f"isolated iterative manifest/state stage mismatch: "
            f"{root_manifest.get('version')!r} != {stage_name!r}"
        )
    if root_manifest.get("evaluation_split_policy") != ISOLATED_EVALUATION_POLICY:
        raise RuntimeError(
            "isolated iterative run did not explicitly ignore the task's active "
            f"evaluation manifest: {root_manifest_path}"
        )
    declared_work_root = root_manifest.get("isolated_work_root")
    if not isinstance(declared_work_root, str) or Path(declared_work_root).resolve() != run_root:
        raise RuntimeError(
            f"isolated iterative work-root identity mismatch in {root_manifest_path}"
        )
    allowed_statuses = {"completed", "stopped_audit_budget_policy", "stopped_empty_pool"}
    if state.get("status") not in allowed_statuses:
        raise RuntimeError(
            f"iterative acquisition is not finalized: {state.get('status')!r}"
        )

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not isinstance(selected, list) or not all(isinstance(path, str) for path in selected):
        raise TypeError(f"iterative selected manifest must be a JSON string list: {selected_path}")
    if len(set(selected)) != len(selected):
        raise RuntimeError(f"iterative selected manifest contains duplicate paths: {selected_path}")

    rounds = state.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise RuntimeError(f"iterative round state contains no completed rounds: {state_path}")
    manifest_train: list[str] = []
    result_sources: list[Path] = []
    seen_rounds: set[int] = set()
    round_manifests: list[str] = []
    for record in rounds:
        round_no = int(record.get("round", -1))
        if round_no < 0 or round_no in seen_rounds:
            raise RuntimeError(f"invalid or duplicate iterative round id: {round_no}")
        seen_rounds.add(round_no)
        round_root = qa_root / f"round_{round_no:02d}"
        manifest_path = round_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"iterative round manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("stage") != stage_name or int(manifest.get("round", -1)) != round_no:
            raise RuntimeError(f"iterative round manifest identity mismatch: {manifest_path}")
        if manifest.get("run_identity_sha256") != run_identity_hash:
            raise RuntimeError(f"iterative round belongs to another run identity: {manifest_path}")
        if manifest.get("complete") is not True:
            raise RuntimeError(f"iterative round manifest is incomplete: {manifest_path}")
        manifest_rows = manifest.get("train")
        state_rows = record.get("train")
        if not isinstance(manifest_rows, list) or manifest_rows != state_rows:
            raise RuntimeError(f"iterative round train rows drifted: {manifest_path}")
        manifest_train.extend(manifest_rows)
        round_manifests.append(str(manifest_path.resolve()))
        result_sources.extend(sorted(round_root.glob("**/*_results.jsonl")))
        result_sources.extend(sorted(round_root.glob("**/*_vqa.json")))

    expected_selected = list(dict.fromkeys(manifest_train))
    if selected != expected_selected:
        raise RuntimeError(
            "iterative cumulative selected manifest does not equal the ordered union "
            f"of completed round manifests: {selected_path}"
        )
    result_sources = _unique_existing_sources(result_sources)
    if selected and not result_sources:
        raise RuntimeError(f"isolated iterative run has no run-local VQA result files: {run_root}")
    for path in result_sources:
        try:
            path.resolve().relative_to(run_root)
        except ValueError as error:
            raise RuntimeError(f"iterative VQA result escapes its run root: {path}") from error

    provenance = {
        "source_policy": ITERATIVE_ONLY_SOURCE_POLICY,
        "logical_stage": ITERATIVE_STAGE,
        "source_stage": stage_name,
        "acquisition_source_policy": ISOLATED_SOURCE_POLICY,
        "evaluation_split_policy": ISOLATED_EVALUATION_POLICY,
        "run_identity_sha256": run_identity_hash,
        "train_pool_hash": split_identity["train_pool_hash"],
        "run_root": str(run_root),
        "selected_manifest": str(selected_path.resolve()),
        "round_state": str(state_path.resolve()),
        "round_manifests": round_manifests,
        "result_sources": [str(path.resolve()) for path in result_sources],
        "ignored_task_supervision_stages": True,
        "ignored_task_evaluation_manifest": True,
    }
    return selected_path, selected, result_sources, provenance


def slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in clean.split("_") if part)


def database_paths(adapter) -> list[str]:
    rows = adapter.records.sort_values("embedding_index")
    paths = rows["relative_path"].astype(str).tolist()
    indices = rows["embedding_index"].astype(int).to_numpy()
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("records are not in a contiguous embedding order")
    return paths


def cache_dir(bank_root: Path, dataset: str, task: str, stage: str,
              method: str, attr: str) -> Path:
    return (
        bank_root / dataset / task / BACKBONE / stage / method
        / slug(attr) / PROTOCOL_VERSION
    )


def cache_meta_path(run_root: Path, dataset: str, task: str, stage: str,
                    method: str, attr: str) -> Path:
    return cache_dir(run_root, dataset, task, stage, method, attr) / "metadata.json"


def expected_cache_meta(dataset: str, task: str, stage: str, method: str, attr: str,
                        emb: np.ndarray, paths: list[str], seeds: list[int],
                        supervision_digest: str, epochs: int,
                        feature_context: dict[str, Any] | None = None,
                        cache_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "dataset": dataset,
        "task": task,
        "backbone": BACKBONE,
        "embedding_dim": int(emb.shape[1]),
        "supervision_stage": stage,
        "method": method,
        "canonical_attribute": attr,
        "attribute_gt_definition": f"{dataset}:{attr}",
        "protocol_version": PROTOCOL_VERSION,
        "records_hash": stable_hash(paths),
        "supervision_hash": supervision_digest,
        "score_length": len(paths),
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
    }
    if feature_context is not None:
        meta["feature_context"] = feature_context
    if cache_identity is not None:
        overlap = set(meta).intersection(cache_identity)
        if overlap:
            raise ValueError(f"cache identity shadows standard metadata: {sorted(overlap)}")
        meta.update(cache_identity)
    return meta


def read_cache(run_root: Path, dataset: str, task: str, stage: str, method: str, attr: str,
               emb: np.ndarray, paths: list[str], seeds: list[int],
               supervision_digest: str, epochs: int,
               legacy_supervision_digest: str | None = None,
               feature_context: dict[str, Any] | None = None,
               cache_identity: dict[str, Any] | None = None) -> tuple[dict, np.ndarray] | None:
    entry = cache_dir(run_root, dataset, task, stage, method, attr)
    meta_path = entry / "metadata.json"
    score_path = entry / "scores.npz"
    if not meta_path.is_file() or not score_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = expected_cache_meta(
        dataset, task, stage, method, attr, emb, paths, seeds, supervision_digest, epochs,
        feature_context, cache_identity,
    )
    legacy_digest_match = False
    cached_seed_indices: list[int] | None = None
    observed_seed_count = len(seeds)
    for key, value in expected.items():
        observed = meta.get(key)
        if observed == value:
            continue
        if key == "seeds" and isinstance(observed, list):
            observed_int = [int(seed) for seed in observed]
            requested_int = [int(seed) for seed in value]
            if all(seed in observed_int for seed in requested_int):
                cached_seed_indices = [observed_int.index(seed) for seed in requested_int]
                observed_seed_count = len(observed_int)
                continue
        if (key == "supervision_hash" and legacy_supervision_digest is not None
                and observed == legacy_supervision_digest):
            legacy_digest_match = True
            continue
        if observed != value:
            raise ValueError(
                f"incompatible ProbeBank entry {entry}: {key}={observed!r}, expected {value!r}"
            )
    with np.load(score_path, allow_pickle=False) as data:
        scores = np.asarray(data["scores"], dtype=np.float32)
    if scores.shape != (observed_seed_count, len(paths)):
        raise ValueError(f"invalid score shape in {score_path}: {scores.shape}")
    if cached_seed_indices is not None:
        scores = scores[np.asarray(cached_seed_indices, dtype=np.int64)]
        meta["_cache_seed_mode"] = "requested_subset_of_cached_superset"
        meta["_cache_seed_indices"] = cached_seed_indices
    else:
        meta["_cache_seed_mode"] = "exact"
    meta["_cache_supervision_hash_mode"] = (
        "legacy_v0_exact_label_digest" if legacy_digest_match else "current_v1"
    )
    return meta, scores


def chunked_score(
    model,
    kind: str,
    emb: np.ndarray,
    patches=None,
    text_query: torch.Tensor | None = None,
) -> np.ndarray:
    """Score one cached probe using its real feature kind."""
    if kind == "mlp":
        chunk_size = 8192
        chunks = [H.score_mlp(model, emb[start:start + chunk_size])
                  for start in range(0, len(emb), chunk_size)]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, dtype=np.float32)
    if kind == "attn":
        if patches is None:
            raise ValueError("attention scoring requires patch tokens")
        return H.score_attn(model, patches, text_query, chunk=1024).astype(np.float32)
    raise RuntimeError(f"unsupported single-probe ProbeBank model kind: {kind}")


def train_cache_entry(run_root: Path, dataset: str, task: str, stage: str, method: str, attr: str,
                      emb: np.ndarray, paths: list[str], p2i: dict[str, int],
                      train_pool: list[str], selected: list[str], labels: list[dict],
                      seeds: list[int], supervision_digest: str,
                      epochs: int, *, patches=None,
                      text_query: torch.Tensor | None = None,
                      feature_context: dict[str, Any] | None = None,
                      cache_identity: dict[str, Any] | None = None,
                      validation_labels: list[dict] | None = None) -> tuple[dict, np.ndarray]:
    if validation_labels is not None:
        val_paths = {row["image"] for row in validation_labels}
        if (not val_paths or len(val_paths) != len(validation_labels)
                or val_paths.intersection(selected) or val_paths.intersection(train_pool)):
            raise ValueError("Fixed Val must be disjoint from supervised and unlabeled pools")
        if not cache_identity or not cache_identity.get("val_isolation_fingerprint"):
            raise ValueError("Fixed Val training requires a versioned isolation cache identity")
    entry = cache_dir(run_root, dataset, task, stage, method, attr)
    entry.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected)
    unlabeled = [p2i[path] for path in train_pool if path not in selected_set]
    u_idx = np.asarray(unlabeled, dtype=np.int64)
    scores_by_seed: list[np.ndarray | None] = [None] * len(seeds)
    attention_models: list[torch.nn.Module] = []
    attention_positions: list[int] = []
    state_files = []
    fallback_seeds = []
    training_metrics_by_seed: dict[str, dict[str, Any]] = {}
    original_attrs = list(H.ATTRS)
    try:
        H.ATTRS = [attr]
        for seed_position, seed in enumerate(seeds):
            rng = np.random.default_rng(1000 + seed)
            u_sample = u_idx if len(u_idx) <= H.U_CAP else u_idx[
                rng.choice(len(u_idx), H.U_CAP, replace=False)
            ]
            ctx = {
                "emb": emb,
                "patches": patches,
                "p2i": p2i,
                "text_q": {attr: text_query},
                # The formal ProbeBank trains one cache entry at a time and
                # temporarily narrows H.ATTRS to the current attribute.  Joint
                # auxiliary methods still need the audited full task schema.
                "task_attrs": tuple(original_attrs),
                "input_dim": int(emb.shape[1]),
                "text_dim": 1 if text_query is None else int(text_query.shape[-1]),
                "patch_dim": 0 if patches is None else int(patches.shape[-1]),
                "U_sample": u_sample,
            }
            if validation_labels is not None:
                ctx["val_labeled"] = validation_labels
            models = H.train_method(
                method, H.TRAINED_METHODS[method], labels, selected, u_idx, ctx, seed,
            )
            if attr not in models:
                # Some task/stage samples contain both classes but too few positives or
                # negatives for a learned probe (for example, a single positive).  That
                # is an expected supervision outcome, not a reason to abort the task.
                # Use the empirical prior as a neutral constant scorer and record the
                # fallback in metadata so downstream analysis can identify it.
                attr_values = [
                    int(record[attr]) for record in labels
                    if record.get(attr) is not None
                ]
                prior = float(np.mean(attr_values)) if attr_values else 0.5
                scores_by_seed[seed_position] = np.full(len(emb), prior, dtype=np.float32)
                fallback_seeds.append(int(seed))
                continue
            model, kind = models[attr]
            training_metrics = getattr(model, "_probe_training_metrics", None)
            if isinstance(training_metrics, dict):
                training_metrics_by_seed[str(seed)] = training_metrics
            if kind not in ("mlp", "attn"):
                raise RuntimeError(f"unsupported ProbeBank model kind: {kind}")
            if kind == "attn":
                attention_models.append(model)
                attention_positions.append(seed_position)
            else:
                scores_by_seed[seed_position] = chunked_score(
                    model, kind, emb, patches, text_query,
                )
            state_path = entry / f"model_seed_{seed}.pt"
            tmp_state = entry / f"model_seed_{seed}.pt.tmp"
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({
                "state_dict": cpu_state,
                "input_dim": int(emb.shape[1]),
                "patch_dim": None if patches is None else int(patches.shape[-1]),
                "text_dim": None if text_query is None else int(text_query.shape[-1]),
                "kind": kind,
                "model_class": type(model).__name__,
            }, tmp_state)
            tmp_state.replace(state_path)
            state_files.append(state_path.name)
            del models
            if kind != "attn":
                del model
            if torch.cuda.is_available() and kind != "attn":
                torch.cuda.empty_cache()

        if attention_models:
            batched_scores = H.score_attn_many(
                attention_models, patches, text_query, chunk=1024,
            )
            for row, position in enumerate(attention_positions):
                scores_by_seed[position] = batched_scores[row]
            attention_models.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        H.ATTRS = original_attrs

    if any(score is None for score in scores_by_seed):
        raise RuntimeError("one or more probe seeds were not scored")
    scores = np.vstack(scores_by_seed).astype(np.float32)
    score_tmp = entry / "scores.npz.tmp"
    with score_tmp.open("wb") as handle:
        np.savez_compressed(handle, scores=scores)
    score_tmp.replace(entry / "scores.npz")
    attr_values = [record.get(attr) for record in labels if record.get(attr) is not None]
    meta = expected_cache_meta(
        dataset, task, stage, method, attr, emb, paths, seeds, supervision_digest, epochs,
        feature_context, cache_identity,
    )
    meta.update({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_files": state_files,
        "training_records_hash": stable_hash(sorted(record["image"] for record in labels)),
        "training_count": len(attr_values),
        "positive_count": int(sum(int(value) for value in attr_values)),
        "negative_count": int(sum(int(value) == 0 for value in attr_values)),
        "source_task": H.TASK,
        "fallback": (
            "constant_empirical_prior_insufficient_class_balance"
            if fallback_seeds else None
        ),
        "fallback_seeds": fallback_seeds,
        "training_metrics_by_seed": training_metrics_by_seed or None,
    })
    write_json(entry / "metadata.json", meta)
    return meta, scores


def candidate_paths(stage: str, train_pool: list[str], emb_norm: np.ndarray,
                    p2i: dict[str, int], query_idx: list[int], label_size: int) -> list[str]:
    if stage == "two_stage":
        ranked, _ = H.query_ranked_paths(train_pool, emb_norm, p2i, query_idx)
    elif stage == "random":
        rng = np.random.default_rng(H.SPLIT_SEED)
        ranked = [train_pool[int(i)] for i in rng.permutation(len(train_pool))]
    else:
        raise ValueError(stage)
    return ranked[:min(label_size, len(ranked))]


def run_vqa(adapter, candidates: list[str], attrs: list[str], output_dir: Path,
            config: str, python: str | None, workers: int) -> tuple[dict, dict, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attr_file = output_dir / "attributes.txt"
    attr_file.write_text("{" + ", ".join(attrs) + "}", encoding="utf-8")
    basenames = [Path(path).name for path in candidates]
    if len(set(basenames)) != len(basenames):
        raise ValueError("VQA candidate basenames are not unique")
    list_file = output_dir / "to_label.json"
    list_file.write_text(json.dumps(basenames, ensure_ascii=False), encoding="utf-8")
    manifest_file = output_dir / "vqa_image_manifest.json"
    manifest_file.write_text(
        json.dumps(adapter.vqa_image_manifest(candidates), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    vqa_python = H.find_vqa_python(python)
    cfg = config if Path(config).is_absolute() else str(H.VQA_DIR / config)
    cmd = vqa_python + [
        "vqa_label.py", "--config", cfg,
        "--images-dir", str(adapter.database_dir),
        "--output-dir", str(output_dir),
        "--attributes-file", str(attr_file),
        "--images-manifest", str(manifest_file),
        "--workers", str(workers),
    ]
    old_result = H.latest_jsonl(output_dir)
    old_labels, _ = H.load_vqa(old_result, adapter.vqa_to_key, attrs)
    old_successful = sum(path in old_labels for path in candidates)
    # A second invocation resumes only failed/missing rows from the same JSONL.
    for attempt in range(2):
        subprocess.run(cmd, cwd=str(H.VQA_DIR), check=True)
        result = H.latest_jsonl(output_dir)
        labels, probs = H.load_vqa(result, adapter.vqa_to_key, attrs)
        successful = [path for path in candidates if path in labels]
        failed = len(candidates) - len(successful)
        print(f"  [vqa] pass {attempt + 1}: successful={len(successful)}/{len(candidates)}, "
              f"api_failed={failed}")
        if failed == 0 or attempt == 1:
            if len(successful) < max(1, len(candidates) // 2):
                raise RuntimeError(
                    f"VQA bulk failure after retry: {len(successful)}/{len(candidates)} successful"
                )
            return labels, probs, max(0, len(successful) - old_successful), failed
    raise AssertionError("unreachable VQA retry state")


def run_iterative_supervision(task: dict, task_out: Path, adapter, args) -> tuple[int, int]:
    """Run isolated 100+50 iterative acquisition without resolving training labels.

    Label resolution intentionally happens only in
    :func:`resolve_training_supervision`, after acquisition has written both
    reused-label snapshots and newly generated result JSONL files.
    """
    work_root = task_out / "vqa" / ITERATIVE_STAGE
    cmd = [
        sys.executable, str(Path(__file__).with_name("run_retrieval_harness.py")),
        "--dataset", task["dataset"], "--task", task["task"],
        "--backbone", BACKBONE, "--iterative-vqa",
        "--iterative-stage", "iterative_vqa_100_50_v2",
        "--iterative-work-root", str(work_root),
        "--iterative-initial-labels", "100",
        "--iterative-round-labels", "50",
        # Round 0 labels 100 rows. Four possible 50-row refinements let the
        # adaptive policy stop normally at 250, or use one final round only
        # when the prequential audit justifies extending to the 300 hard cap.
        "--iterative-max-rounds", "4",
        "--iterative-stop-at-labels", "250",
        "--iterative-max-labels", "300",
        "--iterative-adaptive-budget", "--iterative-resume", "--auto-vqa",
        "--vqa-config", args.vqa_config,
        "--vqa-workers", str(args.vqa_workers),
    ]
    if task.get("joint_label"):
        cmd += ["--joint-label", task["joint_label"]]
    if args.vqa_python:
        cmd += ["--vqa-python", args.vqa_python]
    subprocess.run(cmd, cwd=str(Path(__file__).parent), check=True)

    state_file = work_root / "split" / "round_state.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("status") not in ("completed", "stopped_audit_budget_policy", "stopped_empty_pool"):
        raise RuntimeError(f"iterative acquisition did not finalize: {state.get('status')}")
    requested = []
    for manifest_file in sorted((work_root / "qa").glob("round_*/manifest.json")):
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        requested.extend(manifest.get("requested_to_label", []))
    requested = list(dict.fromkeys(requested))
    new_sources = sorted((work_root / "qa").glob("**/*_results.jsonl"))
    new_labels, _ = H.load_vqa_many(new_sources, adapter.vqa_to_key, list(H.ATTRS))
    failed = sum(path not in new_labels for path in requested)
    return len(requested), failed


def refresh_vqa_counts(summary: dict, task_out: Path, adapter, attrs: list[str],
                       task: dict[str, Any], *, iterative_only: bool = False) -> None:
    """Rebuild logical VQA counts from disk so resumed runs keep earlier stages."""
    attempted = 0
    successful = 0
    manual_content_filter = 0
    if iterative_only:
        stage_roots = [(ITERATIVE_STAGE, task_out / "vqa" / ITERATIVE_STAGE)]
    else:
        stage_roots = []
        for stage in task.get("stages", STAGES):
            source_stage = task.get("supervision_stages", {}).get(stage, stage)
            canonical = adapter.task_root / "supervision" / source_stage
            vqa_dir = canonical if canonical.is_dir() else task_out / "vqa" / stage
            stage_roots.append((stage, vqa_dir))
    for _, vqa_dir in stage_roots:
        for list_file in vqa_dir.rglob("to_label.json"):
            requested = json.loads(list_file.read_text(encoding="utf-8"))
            attempted += len(requested)
            result = H.latest_jsonl(list_file.parent)
            labels, _ = H.load_vqa(result, adapter.vqa_to_key, attrs)
            successful_names = {Path(path).name for path in labels}
            successful += sum(Path(path).name in successful_names for path in requested)
            rows = H._load_direct_label_rows(result) if result else []
            manual_names = {
                Path(adapter.vqa_to_key(row["image"])).name
                for row in rows
                if row.get("label_source") == "manual_content_filter"
            }
            manual_content_filter += sum(
                Path(path).name in manual_names for path in requested
            )
    summary["vqa_attempted"] = attempted
    summary["vqa_successful"] = successful
    summary["vqa_failed"] = attempted - successful
    summary["vqa_manual_content_filter"] = manual_content_filter
    summary["vqa_manual_content_filter_fraction"] = (
        manual_content_filter / successful if successful else 0.0
    )


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sd = float(values.std())
    return np.zeros_like(values) if sd < 1e-12 else (values - float(values.mean())) / sd


def ranks_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=np.float64))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    return ranks


def softmin(values: np.ndarray, tau: float = 1.0) -> np.ndarray:
    scaled = -np.asarray(values, dtype=np.float64) / max(float(tau), 1e-12)
    offset = np.max(scaled, axis=0, keepdims=True)
    return -float(tau) * (
        np.log(np.mean(np.exp(scaled - offset), axis=0)) + offset.reshape(-1)
    )


def joint_scores(attr_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    probs = np.vstack([np.asarray(attr_scores[attr], dtype=np.float64) for attr in H.ATTRS])
    zs = np.vstack([zscore(attr_scores[attr]) for attr in H.ATTRS])
    ranks = np.vstack([1.0 / (60.0 + ranks_desc(attr_scores[attr])) for attr in H.ATTRS])
    zmean = zs.mean(axis=0)
    downside = np.maximum(zmean.reshape(1, -1) - zs, 0.0)
    bounded_deficit = 1.0 - np.exp(-np.maximum(-zs, 0.0))
    soft_min = softmin(zs)
    disagreement = 1.0 - np.exp(-np.std(zs, axis=0))
    return {
        "product": probs.prod(axis=0),
        "zmean": zmean,
        "zmean_minmix_b025": 0.75 * zmean + 0.25 * zs.min(axis=0),
        "zmean_minmix": 0.70 * zmean + 0.30 * zs.min(axis=0),
        "lower_semideviation": zmean - np.sqrt(np.mean(downside * downside, axis=0)),
        "bounded_rms_deficit": zmean - np.sqrt(np.mean(bounded_deficit ** 2, axis=0)),
        "adaptive_mean_softmin": (1.0 - disagreement) * zmean + disagreement * soft_min,
        "attr_rrf": ranks.mean(axis=0),
    }


def gt_for(paths: list[str], gt_by_attr: dict[str, dict[str, int]]) -> tuple[dict, np.ndarray]:
    by_attr = {
        attr: np.asarray([int(gt_by_attr[attr].get(path, 0)) for path in paths], dtype=np.int32)
        for attr in H.ATTRS
    }
    return by_attr, H.joint_gt_from(by_attr)


def metrics(scores: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    ranked = H.rank_metrics(np.asarray(scores), np.asarray(gt))
    ranked.update(H.best_clf_metrics(np.asarray(scores), np.asarray(gt)))
    return {
        "ap": float(ranked["ap"]),
        "f1_star": float(ranked["f1_best"]),
        "p_at_50": float(ranked["p@50"]),
        "r_at_50": float(ranked["r@50"]),
        "p_at_100": float(ranked["p@100"]),
        "r_at_100": float(ranked["r@100"]),
    }


def fit_calibrator(attr_scores_full: dict[str, np.ndarray], selected: list[str],
                   labels: list[dict], p2i: dict[str, int], seed: int):
    selected_scores = {
        attr: np.asarray(values)[np.asarray([p2i[path] for path in selected], dtype=np.int64)]
        for attr, values in attr_scores_full.items()
    }
    return H.fit_joint_calibrator(selected_scores, labels, seed)


def ranking_score_set(attr_scores: dict[str, np.ndarray], joint: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(values)
        for key, values in H.scores_by_ranking_key(attr_scores, joint).items()
    }


def zmean_fusion(score_sets: dict[str, dict[str, np.ndarray]], members: list[str]) -> dict[str, np.ndarray]:
    return {
        key: np.mean(np.vstack([zscore(score_sets[member][key]) for member in members]), axis=0)
        for key in H.RANKING_SPEC
    }


def learned_fusion_joint(attr_scores: dict[str, np.ndarray], attrs: tuple[str, ...],
                         joint_rule: str) -> np.ndarray:
    values = [np.asarray(attr_scores[attr], dtype=np.float64) for attr in attrs]
    if len(values) == 1:
        return values[0]
    if joint_rule == "product":
        return np.prod(np.vstack(values), axis=0)
    zs = np.vstack([zscore(value) for value in values])
    if joint_rule == "zmean":
        return zs.mean(axis=0)
    if joint_rule == "elu_mean":
        return np.where(zs >= 0.0, zs, np.exp(zs) - 1.0).mean(axis=0)
    raise ValueError(f"unknown learned-fusion joint rule: {joint_rule}")


def cross_method_softgate_fusion(
    score_sets: dict[str, dict[str, np.ndarray]],
    members: list[str],
    gate_strategy: str,
    joint_rule: str,
) -> dict[str, np.ndarray]:
    """Apply SoftGate across methods, never inside a single method.

    ``joint_score`` first computes one Joint score per method and gates those
    method-level scores. ``probe_level`` gates all method probes separately for
    each attribute, then applies the requested Joint rule to the fused attrs.
    """
    if gate_strategy == "joint_score":
        output = {}
        for key, info in H.RANKING_SPEC.items():
            attrs = tuple(info.get("attrs") or info["gt"][1:])
            member_scores = [
                learned_fusion_joint(
                    {
                        attr: np.asarray(score_sets[member][_single_attr_key(attr)])
                        for attr in attrs
                    },
                    attrs,
                    joint_rule,
                )
                for member in members
            ]
            output[key] = soft_gate_mean(member_scores)
        return output
    if gate_strategy == "probe_level":
        fused_attrs = {
            attr: soft_gate_mean([
                np.asarray(score_sets[member][_single_attr_key(attr)])
                for member in members
            ])
            for attr in H.ATTRS
        }
        return {
            key: learned_fusion_joint(
                fused_attrs,
                tuple(info.get("attrs") or info["gt"][1:]),
                joint_rule,
            )
            for key, info in H.RANKING_SPEC.items()
        }
    raise ValueError(f"unknown cross-method SoftGate strategy: {gate_strategy}")


def _training_rows(selected: list[str], labels: list[dict], p2i: dict[str, int],
                   attrs: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if len(selected) != len(labels):
        raise ValueError("selected paths and training labels must have the same length")
    indices = []
    targets = []
    for path, record in zip(selected, labels):
        values = [record.get(attr) for attr in attrs]
        if path not in p2i or any(value is None for value in values):
            continue
        indices.append(p2i[path])
        targets.append(int(all(int(value) == 1 for value in values)))
    return np.asarray(indices, dtype=np.int64), np.asarray(targets, dtype=np.float64)


def _fit_metadata(recipe: dict[str, Any], members: list[str], y: np.ndarray) -> dict[str, Any]:
    return {
        "training_strategy": recipe["training_strategy"],
        "joint_rule": recipe["joint_rule"],
        "members": members,
        "n_train": int(len(y)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "n_negative": int(len(y) - y.sum()) if len(y) else 0,
        "ridge": float(recipe.get("ridge", 1e-3)),
        "min_class_samples": int(recipe.get("min_class_samples", 2)),
        "loss": "class_balanced_mse",
        "training_source": "selected_training_supervision_only",
        "status": "fallback_equal_weights",
    }


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float64)
    return np.asarray([len(y) / (2.0 * counts[int(target)]) for target in y])


def _can_fit(y: np.ndarray, n_members: int, model: dict[str, Any],
             recipe: dict[str, Any]) -> bool:
    minimum = int(recipe.get("min_train_samples", max(8, n_members + 1)))
    counts = np.bincount(y.astype(np.int64), minlength=2) if len(y) else np.zeros(2, dtype=int)
    return bool(
        n_members > 0 and len(y) >= minimum and len(np.unique(y)) == 2
        and int(counts.min()) >= model["min_class_samples"]
    )


def _bounded_simplex_least_squares(X: np.ndarray, y: np.ndarray, ridge: float,
                                   default_weights: np.ndarray,
                                   max_iter: int = 250) -> tuple[np.ndarray, dict[str, Any]]:
    """Class-balanced bounded regression with non-negative weights summing to one."""
    if not np.isfinite(X).all():
        return default_weights, {"status": "fallback_equal_weights", "fallback_reason": "non_finite_training_scores"}
    sample_weight = _balanced_sample_weight(y)
    def objective(weights):
        prediction = X @ weights
        return float(
            np.average((prediction - y) ** 2, weights=sample_weight)
            + ridge * np.dot(weights, weights)
        )
    one_hot_losses = [objective(np.eye(X.shape[1])[i]) for i in range(X.shape[1])]
    starts = [default_weights, np.eye(X.shape[1])[int(np.argmin(one_hot_losses))]]
    solutions = [
        minimize(
            objective, start, method="SLSQP", bounds=[(0.0, 1.0)] * X.shape[1],
            constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
            options={"maxiter": int(max_iter), "ftol": 1e-9},
        )
        for start in starts
    ]
    valid = [solution for solution in solutions if solution.success and np.isfinite(solution.fun)]
    if not valid:
        return default_weights, {
            "status": "fallback_equal_weights", "fallback_reason": "optimizer_failed_simplex_regression",
        }
    solution = min(valid, key=lambda item: float(item.fun))
    weights = np.clip(solution.x, 0.0, 1.0)
    weights /= max(float(weights.sum()), 1e-12)
    prediction = X @ weights
    return weights, {
        "status": "fitted",
        "train_balanced_mse": float(np.average((prediction - y) ** 2, weights=sample_weight)),
        "active_members": int(np.count_nonzero(weights > 1e-6)),
        "weight_sum_constraint": 1.0,
    }


def _single_attr_key(attr: str) -> str:
    for key, info in H.RANKING_SPEC.items():
        attrs = tuple(info.get("attrs") or info["gt"][1:])
        if attrs == (attr,):
            return key
    raise KeyError(f"ranking specification has no single-attribute key for {attr}")


def _member_attr_scores(score_sets: dict[str, dict[str, np.ndarray]],
                        members: list[str], attr: str) -> np.ndarray:
    key = _single_attr_key(attr)
    return np.column_stack([
        np.asarray(score_sets[member][key], dtype=np.float64) for member in members
    ])


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _probability_matrix(values: np.ndarray, context: str) -> np.ndarray:
    """Validate the ProbeBank probability contract, allowing only round-off."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains non-finite scores")
    if values.size and (float(values.min()) < -1e-6 or float(values.max()) > 1.0 + 1e-6):
        raise ValueError(f"{context} scores must stay in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def _aggregate_probe_scores(values: np.ndarray, aggregation: str) -> np.ndarray:
    """Reduce selected probe scores to one input score per image."""
    if aggregation == "mean":
        return values.mean(axis=1)
    if aggregation == "min":
        return values.min(axis=1)
    if aggregation == "max":
        return values.max(axis=1)
    raise ValueError(f"unknown R-SoftGate probe aggregation: {aggregation}")


def fit_r_softgate(recipe: dict[str, Any],
                   full_score_sets: dict[str, dict[str, np.ndarray]],
                   selected: list[str], labels: list[dict],
                   p2i: dict[str, int]) -> dict[str, Any]:
    """Fit per-attribute gates with a train-only Joint objective.

    Top-1/3 probes are selected by each attribute's training AP.  Their raw
    probabilities are aggregated before gating; Gallery/Test never enter either
    selection or optimization.
    """
    members = resolve_fusion_members(recipe)
    missing = [member for member in members if member not in full_score_sets]
    if missing:
        raise RuntimeError(f"fusion {recipe['id']} is missing full scores: {missing}")
    requested_top_k = recipe.get("top_k", "all")
    top_k = len(members) if requested_top_k == "all" else int(requested_top_k)
    probe_aggregation = recipe.get("probe_aggregation", "mean")
    selected_by_attr: dict[str, list[str]] = {}
    selection_diagnostics: dict[str, dict[str, Any]] = {}

    for attr in H.ATTRS:
        idx, y_attr = _training_rows(selected, labels, p2i, (attr,))
        matrix = _probability_matrix(
            _member_attr_scores(full_score_sets, members, attr), f"R-SoftGate {attr}",
        )
        train_ap = {
            member: average_precision(y_attr, matrix[idx, member_idx])
            for member_idx, member in enumerate(members)
        }
        if requested_top_k == "all":
            chosen = list(members)
        else:
            chosen = sorted(
                members,
                key=lambda member: (
                    -(train_ap[member] if np.isfinite(train_ap[member]) else -np.inf),
                    members.index(member),
                ),
            )[:top_k]
        selected_by_attr[attr] = chosen
        selection_diagnostics[attr] = {
            "n_train": int(len(y_attr)),
            "n_positive": int(y_attr.sum()) if len(y_attr) else 0,
            "selected_members": chosen,
            "selection_uses_labels": requested_top_k != "all",
            "train_ap_by_member": {
                member: (float(value) if np.isfinite(value) else None)
                for member, value in train_ap.items()
            },
        }

    idx, y = _training_rows(selected, labels, p2i, tuple(H.ATTRS))
    attrs = tuple(H.ATTRS)
    attribute_inputs = np.column_stack([
        _aggregate_probe_scores(
            _probability_matrix(
                _member_attr_scores(full_score_sets, selected_by_attr[attr], attr),
                f"R-SoftGate {attr}",
            ),
            probe_aggregation,
        )[idx]
        for attr in attrs
    ])
    theta_min = float(recipe.get("theta_min", 0.0))
    theta_max = float(recipe.get("theta_max", 1.0))
    temp_min = float(recipe.get("temperature_min", 0.01))
    temp_max = float(recipe.get("temperature_max", 1.0))
    loss_name = recipe.get("loss", "binary_cross_entropy")
    ranking_temperature = float(recipe.get("ranking_temperature", 0.1))
    default = np.r_[
        np.full(len(attrs), np.clip(0.5, theta_min, theta_max)),
        np.full(len(attrs), np.clip(0.1, temp_min, temp_max)),
    ]

    def objective(parameters: np.ndarray) -> float:
        theta, temperature = np.split(np.asarray(parameters), 2)
        gates = _stable_sigmoid((attribute_inputs - theta) / temperature)
        prediction = np.clip(np.prod(gates, axis=1), 1e-8, 1.0 - 1e-8)
        if loss_name == "binary_cross_entropy":
            return float(-np.mean(
                y * np.log(prediction) + (1.0 - y) * np.log1p(-prediction)
            ))
        if loss_name == "joint_pairwise_ranking":
            positive = prediction[y == 1]
            negative = prediction[y == 0]
            # Soft pairwise ranking: every positive Joint should outrank every
            # negative Joint. logaddexp is the stable softplus implementation.
            violations = (negative[None, :] - positive[:, None]) / ranking_temperature
            return float(np.logaddexp(0.0, violations).mean())
        raise ValueError(f"unknown R-SoftGate loss: {loss_name}")

    model: dict[str, Any] = {
        "aggregation": "r_softgate",
        "members": members,
        "candidate_members": members,
        "top_k": requested_top_k,
        "pool_id": recipe.get("pool_id", f"top{top_k}"),
        "pool_size": top_k,
        "probe_aggregation": probe_aggregation,
        "selected_members_by_attr": selected_by_attr,
        "selection_diagnostics": selection_diagnostics,
        "n_train": int(len(y)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "n_negative": int(len(y) - y.sum()) if len(y) else 0,
        "loss": loss_name,
        "ranking_temperature": (
            ranking_temperature if loss_name == "joint_pairwise_ranking" else None
        ),
        "training_source": "selected_training_supervision_only",
        "score_definition": "product_of_attribute_gates_only",
        "attribute_input_definition": f"{probe_aggregation}_of_selected_probe_scores",
    }
    minimum = int(recipe.get("min_train_samples", max(8, 2 * len(attrs) + 1)))
    class_counts = np.bincount(y.astype(np.int64), minlength=2) if len(y) else np.zeros(2)
    can_fit = (
        len(y) >= minimum and len(np.unique(y)) == 2
        and int(class_counts.min()) >= int(recipe.get("min_class_samples", 2))
    )
    best = None
    if can_fit:
        median_theta = np.median(attribute_inputs, axis=0)
        starts = [
            default,
            np.r_[median_theta, np.full(len(attrs), 0.1)],
            np.r_[np.full(len(attrs), 0.25), np.full(len(attrs), 0.25)],
        ]
        bounds = (
            [(theta_min, theta_max)] * len(attrs)
            + [(temp_min, temp_max)] * len(attrs)
        )
        solutions = [
            minimize(
                objective, np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds]),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": int(recipe.get("max_iter", 500)), "ftol": 1e-10},
            )
            for start in starts
        ]
        finite = [solution for solution in solutions if np.isfinite(solution.fun)]
        best = min(finite, key=lambda solution: float(solution.fun)) if finite else None

    parameters = np.asarray(best.x if best is not None else default, dtype=np.float64)
    theta, temperature = np.split(parameters, 2)
    model.update({
        "theta_by_attr": {attr: float(theta[i]) for i, attr in enumerate(attrs)},
        "temperature_by_attr": {
            attr: float(temperature[i]) for i, attr in enumerate(attrs)
        },
        "status": (
            "fitted" if best is not None and best.success
            else "fitted_optimizer_warning" if best is not None
            else "fallback_default_gates"
        ),
        "train_loss": float(objective(parameters)) if len(y) else None,
        "train_bce": (
            float(objective(parameters))
            if len(y) and loss_name == "binary_cross_entropy" else None
        ),
        "train_pairwise_ranking_loss": (
            float(objective(parameters))
            if len(y) and loss_name == "joint_pairwise_ranking" else None
        ),
    })
    if best is None:
        model["fallback_reason"] = (
            "insufficient_complete_or_two_class_training_labels"
            if not can_fit else "optimizer_failed"
        )
    elif not best.success:
        model["optimizer_message"] = str(best.message)
    return model


def r_softgate_fusion(score_sets: dict[str, dict[str, np.ndarray]],
                      fitted: dict[str, Any]) -> dict[str, np.ndarray]:
    """Aggregate probes, apply learned gates, then multiply only those gates."""
    gates = {}
    probe_aggregation = fitted.get("probe_aggregation", "mean")
    for attr in H.ATTRS:
        members = list(fitted["selected_members_by_attr"][attr])
        attribute_input = _aggregate_probe_scores(
            _probability_matrix(
                _member_attr_scores(score_sets, members, attr), f"R-SoftGate {attr}",
            ),
            probe_aggregation,
        )
        theta = float(fitted["theta_by_attr"][attr])
        temperature = float(fitted["temperature_by_attr"][attr])
        if temperature <= 0.0:
            raise ValueError(f"R-SoftGate temperature must be positive for {attr}")
        gates[attr] = _stable_sigmoid((attribute_input - theta) / temperature)
    return {
        key: np.prod(
            np.vstack([gates[attr] for attr in tuple(info.get("attrs") or info["gt"][1:])]),
            axis=0,
        )
        for key, info in H.RANKING_SPEC.items()
    }


def fit_bounded_linear_fusion(recipe: dict[str, Any],
                              full_score_sets: dict[str, dict[str, np.ndarray]],
                              selected: list[str], labels: list[dict],
                              p2i: dict[str, int]) -> dict[str, Any]:
    """Fit one of the nine train-only bounded linear fusion variants."""
    members = resolve_fusion_members(recipe)
    missing = [member for member in members if member not in full_score_sets]
    if missing:
        raise RuntimeError(f"fusion {recipe['id']} is missing full scores: {missing}")
    n_members = len(members)
    equal = np.full(n_members, 1.0 / max(n_members, 1), dtype=np.float64)
    strategy = recipe["training_strategy"]
    joint_rule = recipe["joint_rule"]
    ridge = float(recipe.get("ridge", 1e-3))

    if strategy == "per_attr":
        weights_by_attr = {}
        attr_fit = {}
        for attr in H.ATTRS:
            idx, y = _training_rows(selected, labels, p2i, (attr,))
            meta = _fit_metadata(recipe, members, y)
            if _can_fit(y, n_members, meta, recipe):
                X = _member_attr_scores(full_score_sets, members, attr)[idx]
                weights, update = _bounded_simplex_least_squares(
                    X, y, ridge, equal, max_iter=int(recipe.get("max_iter", 250)),
                )
                meta.update(update)
            else:
                weights = equal.copy()
                meta["fallback_reason"] = "insufficient_complete_or_two_class_training_labels"
            weights_by_attr[attr] = weights
            attr_fit[attr] = {key: value for key, value in meta.items() if key != "members"}
        return {
            "training_strategy": strategy, "joint_rule": joint_rule, "members": members,
            "weights_by_attr": weights_by_attr, "attr_fit": attr_fit,
            "training_source": "selected_training_supervision_only",
            "status": ("fitted" if all(v["status"] == "fitted" for v in attr_fit.values())
                       else "fitted_with_attr_fallbacks"),
        }

    idx, y = _training_rows(selected, labels, p2i, tuple(H.ATTRS))
    model = _fit_metadata(recipe, members, y)
    if strategy == "method_joint":
        model["weights"] = equal.copy()
    elif strategy == "shared_attr_joint_loss":
        # The historical strategy ID is retained for result compatibility, but
        # the parameters are not shared across attributes: one jointly trained
        # model owns a complete w[attr, method] matrix.
        model["weights_by_attr"] = {attr: equal.copy() for attr in H.ATTRS}
    else:
        raise ValueError(f"unknown bounded-linear training strategy: {strategy}")
    if not _can_fit(y, n_members, model, recipe):
        model["fallback_reason"] = "insufficient_complete_or_two_class_training_labels"
        return model

    if strategy == "method_joint":
        X = np.column_stack([
            learned_fusion_joint({
                attr: np.asarray(full_score_sets[member][_single_attr_key(attr)])[idx]
                for attr in H.ATTRS
            }, tuple(H.ATTRS), joint_rule)
            for member in members
        ])
        weights, update = _bounded_simplex_least_squares(
            X, y, ridge, equal, max_iter=int(recipe.get("max_iter", 250)),
        )
        model.update(update)
        model["weights"] = weights
        return model

    if strategy == "shared_attr_joint_loss":
        X_by_attr = {
            attr: _member_attr_scores(full_score_sets, members, attr)[idx]
            for attr in H.ATTRS
        }
        if not all(np.isfinite(values).all() for values in X_by_attr.values()):
            model["fallback_reason"] = "non_finite_training_scores"
            return model
        sample_weight = _balanced_sample_weight(y)

        attrs_order = tuple(H.ATTRS)
        n_attrs = len(attrs_order)
        equal_matrix = np.tile(equal, (n_attrs, 1))

        def as_matrix(flat_weights):
            return np.asarray(flat_weights, dtype=np.float64).reshape(n_attrs, n_members)

        def objective(flat_weights):
            weight_matrix = as_matrix(flat_weights)
            attrs = {
                attr: X_by_attr[attr] @ weight_matrix[attr_idx]
                for attr_idx, attr in enumerate(attrs_order)
            }
            prediction = learned_fusion_joint(attrs, tuple(H.ATTRS), joint_rule)
            flat_weights = weight_matrix.ravel()
            return float(
                np.average((prediction - y) ** 2, weights=sample_weight)
                + ridge * np.dot(flat_weights, flat_weights)
            )

        shared_one_hot_starts = [
            np.tile(np.eye(n_members)[member_idx], (n_attrs, 1)).ravel()
            for member_idx in range(n_members)
        ]
        best_one_hot = min(shared_one_hot_starts, key=objective)
        starts = [equal_matrix.ravel(), best_one_hot]
        constraints = [
            {
                "type": "eq",
                "fun": lambda flat_weights, attr_idx=attr_idx: float(
                    as_matrix(flat_weights)[attr_idx].sum() - 1.0
                ),
            }
            for attr_idx in range(n_attrs)
        ]
        solutions = [
            minimize(
                objective, start, method="SLSQP",
                bounds=[(0.0, 1.0)] * (n_attrs * n_members),
                constraints=constraints,
                options={"maxiter": int(recipe.get("max_iter", 250)), "ftol": 1e-9},
            )
            for start in starts
        ]
        valid = [solution for solution in solutions if solution.success and np.isfinite(solution.fun)]
        if not valid:
            model["fallback_reason"] = "optimizer_failed_shared_attr_joint_loss"
            return model
        best = min(valid, key=lambda solution: float(solution.fun))
        weight_matrix = np.clip(as_matrix(best.x), 0.0, 1.0)
        weight_matrix /= np.maximum(weight_matrix.sum(axis=1, keepdims=True), 1e-12)
        flat_weights = weight_matrix.ravel()
        model.update({
            "weights_by_attr": {
                attr: weight_matrix[attr_idx]
                for attr_idx, attr in enumerate(attrs_order)
            },
            "status": "fitted", "train_objective": float(best.fun),
            "train_balanced_mse": float(
                objective(flat_weights) - ridge * np.dot(flat_weights, flat_weights)
            ),
            "active_weights": int(np.count_nonzero(weight_matrix > 1e-6)),
            "weight_sum_constraint": "one_per_attribute",
            "weight_sums_by_attr": {
                attr: float(weight_matrix[attr_idx].sum())
                for attr_idx, attr in enumerate(attrs_order)
            },
        })
        return model

    raise ValueError(f"unknown bounded-linear training strategy: {strategy}")


def bounded_linear_fusion(score_sets: dict[str, dict[str, np.ndarray]],
                          fitted: dict[str, Any]) -> dict[str, np.ndarray]:
    members = list(fitted["members"])
    strategy = fitted["training_strategy"]
    joint_rule = fitted["joint_rule"]

    if strategy == "method_joint":
        weights = np.asarray(fitted["weights"], dtype=np.float64)
        output = {}
        for key, info in H.RANKING_SPEC.items():
            attrs = tuple(info.get("attrs") or info["gt"][1:])
            member_scores = []
            for member in members:
                by_attr = {
                    attr: score_sets[member][_single_attr_key(attr)]
                    for attr in attrs
                }
                member_scores.append(learned_fusion_joint(by_attr, attrs, joint_rule))
            output[key] = weights @ np.vstack(member_scores)
        return output

    if strategy in {"shared_attr_joint_loss", "per_attr"}:
        weights_by_attr = {
            attr: np.asarray(fitted["weights_by_attr"][attr], dtype=np.float64)
            for attr in H.ATTRS
        }
    else:
        raise ValueError(f"unknown bounded-linear training strategy: {strategy}")
    if any(np.any(weights < 0.0) or np.any(weights > 1.0) for weights in weights_by_attr.values()):
        raise ValueError("bounded-linear fusion weights must stay in [0, 1]")
    fused_attrs = {
        attr: _member_attr_scores(score_sets, members, attr) @ weights_by_attr[attr]
        for attr in H.ATTRS
    }
    return {
        key: learned_fusion_joint(
            fused_attrs, tuple(info.get("attrs") or info["gt"][1:]), joint_rule,
        )
        for key, info in H.RANKING_SPEC.items()
    }


def fusion_fit_output(fitted: dict[str, Any]) -> dict[str, Any]:
    """Convert fitted arrays to explicit method-keyed JSON metadata."""
    members = list(fitted.get("members", []))
    output = {
        key: value for key, value in fitted.items()
        if key not in {"members", "weights", "weights_by_attr"}
    }
    if "weights" in fitted:
        if not members:
            raise ValueError("fitted fusion weights require an explicit members list")
        output["weights"] = {
            member: float(weight)
            for member, weight in zip(members, fitted["weights"])
        }
    if "weights_by_attr" in fitted:
        if not members:
            raise ValueError("fitted per-attribute weights require an explicit members list")
        output["weights_by_attr"] = {
            attr: {
                member: float(weight)
                for member, weight in zip(members, weights)
            }
            for attr, weights in fitted["weights_by_attr"].items()
        }
    return output


def _pool_member_score(
    score_sets: dict[str, dict[str, np.ndarray]],
    member: str,
    key: str,
    joint_rule: str,
) -> np.ndarray:
    """Compose one pool member's requested ranking from its attribute probes."""
    info = H.RANKING_SPEC[key]
    attrs = tuple(info.get("attrs") or info["gt"][1:])
    if len(attrs) == 1:
        return np.asarray(score_sets[member][_single_attr_key(attrs[0])])
    cache_key = f"__pool_joint__{joint_rule}__{key}"
    if cache_key in score_sets[member]:
        return np.asarray(score_sets[member][cache_key])
    by_attr = {
        attr: np.asarray(score_sets[member][_single_attr_key(attr)])
        for attr in attrs
    }
    composed = learned_fusion_joint(by_attr, attrs, joint_rule)
    score_sets[member][cache_key] = composed
    return composed


def fit_pool_baseline_fusion(recipe: dict[str, Any],
                             full_score_sets: dict[str, dict[str, np.ndarray]],
                             selected: list[str], labels: list[dict],
                             p2i: dict[str, int]) -> dict[str, Any]:
    """Select a candidate pool using only the complete training supervision.

    Fixed All/Curated pools never inspect labels. Elite ranks candidates by
    training AP. Coverage greedily maximizes newly retrieved training-positive
    images inside each candidate's training Top-K, with training AP as the
    deterministic tie-breaker. Gallery and Test labels are never inputs.
    """
    members = resolve_fusion_members(recipe)
    pool_joint_rule = recipe.get("pool_joint_rule", "product")
    missing = [member for member in members if member not in full_score_sets]
    if missing:
        raise RuntimeError(f"fusion {recipe['id']} is missing full scores: {missing}")
    strategy = recipe["pool_strategy"]
    pool_size = int(recipe["pool_size"])
    selected_by_key: dict[str, list[str]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    for key, info in H.RANKING_SPEC.items():
        attrs = tuple(info.get("attrs") or info["gt"][1:])
        idx, y = _training_rows(selected, labels, p2i, attrs)
        available = [member for member in members if key in full_score_sets[member]]
        if len(available) < pool_size:
            raise RuntimeError(
                f"fusion {recipe['id']} key={key} has {len(available)} candidates; "
                f"requires {pool_size}"
            )
        record: dict[str, Any] = {
            "n_train": int(len(y)),
            "n_positive": int(y.sum()) if len(y) else 0,
            "n_negative": int(len(y) - y.sum()) if len(y) else 0,
        }
        if strategy in {"fixed_all", "fixed_curated"}:
            chosen = available[:pool_size]
            record["selection_uses_labels"] = False
        else:
            train_ap = {
                member: average_precision(
                    y,
                    _pool_member_score(full_score_sets, member, key, pool_joint_rule)[idx],
                )
                for member in available
            }
            ap_order = sorted(
                available,
                key=lambda member: (
                    -(train_ap[member] if np.isfinite(train_ap[member]) else -np.inf),
                    member,
                ),
            )
            if strategy == "elite_train_ap":
                chosen = ap_order[:pool_size]
            elif strategy == "coverage_train_topk":
                topk = min(int(recipe.get("coverage_topk", 50)), len(idx))
                positives = set(np.flatnonzero(y == 1).tolist())
                retrieved_positive: dict[str, set[int]] = {}
                for member in available:
                    scores = _pool_member_score(
                        full_score_sets, member, key, pool_joint_rule,
                    )[idx]
                    order = np.argsort(-scores, kind="mergesort")[:topk]
                    retrieved_positive[member] = set(map(int, order)).intersection(positives)
                chosen = []
                uncovered = set(positives)
                while len(chosen) < pool_size:
                    candidates = [member for member in available if member not in chosen]
                    best = max(
                        candidates,
                        key=lambda member: (
                            len(retrieved_positive[member].intersection(uncovered)),
                            train_ap[member] if np.isfinite(train_ap[member]) else -np.inf,
                            -available.index(member),
                        ),
                    )
                    chosen.append(best)
                    uncovered.difference_update(retrieved_positive[best])
                record.update({
                    "coverage_topk": topk,
                    "training_positive_covered": int(len(positives - uncovered)),
                    "training_positive_total": int(len(positives)),
                })
            else:
                raise ValueError(f"unknown pool strategy: {strategy}")
            record["selection_uses_labels"] = True
            record["train_ap_by_member"] = {
                member: (float(value) if np.isfinite(value) else None)
                for member, value in train_ap.items()
            }
        selected_by_key[key] = chosen
        record["selected_members"] = chosen
        diagnostics[key] = record
    return {
        "pool_id": recipe["pool_id"],
        "pool_strategy": strategy,
        "pool_aggregation": recipe["pool_aggregation"],
        "pool_joint_rule": pool_joint_rule,
        "pool_size": pool_size,
        "candidate_members": members,
        "selected_members_by_key": selected_by_key,
        "selection_diagnostics": diagnostics,
        "training_source": (
            "fixed_predeclared_members_no_label_selection"
            if strategy in {"fixed_all", "fixed_curated"}
            else "selected_training_supervision_only"
        ),
        "status": "selected",
    }


def pool_baseline_fusion(score_sets: dict[str, dict[str, np.ndarray]],
                         fitted: dict[str, Any], k: float = 60.0) -> dict[str, np.ndarray]:
    """Apply equal-weight RRF or z-score averaging to the fitted candidate pool."""
    output = {}
    aggregation = fitted["pool_aggregation"]
    pool_joint_rule = fitted.get("pool_joint_rule", "product")
    for key in H.RANKING_SPEC:
        members = list(fitted["selected_members_by_key"][key])
        if aggregation == "rrf":
            first = _pool_member_score(score_sets, members[0], key, pool_joint_rule)
            total = np.zeros_like(first, dtype=np.float64)
            for member in members:
                score = _pool_member_score(score_sets, member, key, pool_joint_rule)
                total += 1.0 / (float(k) + ranks_from_scores(score))
            output[key] = total / len(members)
        elif aggregation == "zavg":
            output[key] = np.mean(
                np.vstack([
                    zscore(_pool_member_score(score_sets, member, key, pool_joint_rule))
                    for member in members
                ]),
                axis=0,
            )
        else:
            raise ValueError(f"unknown pool aggregation: {aggregation}")
    return output


def apply_fusion_recipe(recipe: dict[str, Any],
                        score_sets: dict[str, dict[str, np.ndarray]],
                        fitted: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    members = resolve_fusion_members(recipe)
    missing = [member for member in members if member not in score_sets]
    if missing:
        raise RuntimeError(f"fusion {recipe['id']} is missing members: {missing}")
    aggregation = recipe["aggregation"]
    if aggregation == "zmean":
        return zmean_fusion(score_sets, members)
    if aggregation == "cross_method_softgate":
        return cross_method_softgate_fusion(
            score_sets,
            members,
            recipe["gate_strategy"],
            recipe["joint_rule"],
        )
    if aggregation == "rrf":
        result = rrf_fusion(
            score_sets, list(H.RANKING_SPEC), allow_methods=set(members),
            k=float(recipe.get("k", 60.0)), name=recipe["id"],
        )
        return result.scores
    if aggregation == "bounded_linear":
        if fitted is None:
            raise ValueError(f"fusion {recipe['id']} requires train-fitted bounded weights")
        return bounded_linear_fusion(score_sets, fitted)
    if aggregation == "r_softgate":
        if fitted is None:
            raise ValueError(f"fusion {recipe['id']} requires train-fitted gates")
        return r_softgate_fusion(score_sets, fitted)
    if aggregation == "pool_baseline":
        if fitted is None:
            raise ValueError(f"fusion {recipe['id']} requires a train-fitted pool")
        return pool_baseline_fusion(
            score_sets, fitted, k=float(recipe.get("rrf_k", 60.0)),
        )
    raise ValueError(f"unknown fusion aggregation: {aggregation}")


def evaluate_task(adapter, emb_norm: np.ndarray, paths: list[str], gallery: list[str], test: list[str],
                  selected: list[str], labels: list[dict], stage_scores: dict[str, dict[str, np.ndarray]],
                  seeds: list[int], stage: str, task_out: Path,
                  supervision_digest: str) -> dict[str, Any]:
    split_paths = {"gallery": gallery, "test": test}
    split_idx = {name: np.asarray([adapter.p2i[path] for path in values], dtype=np.int64)
                 for name, values in split_paths.items()}
    split_gt = {name: gt_for(values, adapter.gt_by_attr) for name, values in split_paths.items()}
    rows = []
    top50: dict[str, Any] = defaultdict(dict)

    embedding_score_sets: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split, indices in split_idx.items():
        gt_by_attr, joint_gt = split_gt[split]
        if EMBEDDING_METHODS:
            _, raw = H.compute_embedding_baselines(
                emb_norm, indices, list(adapter.query_idx), None, gt_by_attr, joint_gt,
                emb_methods=list(EMBEDDING_METHODS), backbone=BACKBONE,
            )
        else:
            raw = {}
        embedding_score_sets[split] = {}
        for method in EMBEDDING_METHODS:
            scores = {key: np.asarray(raw[method][key]) for key in H.RANKING_SPEC}
            embedding_score_sets[split][method] = scores
            ranking_metrics = {
                attr: metrics(scores[H.ranking_key_for_combo((attr,))], gt_by_attr[attr])
                for attr in H.ATTRS
            }
            joint = scores[H.JOINT_KEY]
            ranking_metrics["joint"] = metrics(joint, joint_gt)
            rows.append({"stage": stage, "split": split, "seed": None, "method": method,
                         "joint_rule": "native", "metrics": ranking_metrics})
            if split == "gallery":
                order = np.argsort(-joint)[:50]
                top50[method]["native"] = [
                    {"rank": rank + 1, "image": gallery[int(i)], "score": float(joint[int(i)]),
                     "gt": int(joint_gt[int(i)])}
                    for rank, i in enumerate(order)
                ]

    for seed_pos, seed in enumerate(seeds):
        full_attrs_by_method = {
            method: {attr: stage_scores[method][attr][seed_pos] for attr in H.ATTRS}
            for method in BASE_METHODS
        }
        calibrators = ({
            method: fit_calibrator(scores, selected, labels, adapter.p2i, seed)
            for method, scores in full_attrs_by_method.items()
        } if "joint_logreg" in JOINT_VARIANTS or "train_calibrator" in JOINT_VARIANTS else {})

        # Build full-database score vectors only to index the selected training
        # rows below.  No gallery/test labels are available to the fitter.
        full_ranking_scores_by_method: dict[str, dict[str, np.ndarray]] = {}
        for method in REPORTED_METHODS:
            attrs_full = full_attrs_by_method[method]
            variants_full = joint_scores(attrs_full)
            for rule in JOINT_VARIANTS:
                joint_full = (
                    H.joint_score(attrs_full, calibrator=calibrators[method])
                    if rule in {"train_calibrator", "joint_logreg"}
                    else variants_full[rule]
                )
                variant_id = method if rule == "product" else f"{method}__{rule}"
                full_ranking_scores_by_method[variant_id] = ranking_score_set(attrs_full, joint_full)
        train_fitted_fusions = {}
        for recipe in FUSION_RECIPES:
            if recipe["aggregation"] == "bounded_linear":
                train_fitted_fusions[recipe["id"]] = fit_bounded_linear_fusion(
                    recipe, full_ranking_scores_by_method, selected, labels, adapter.p2i,
                )
            elif recipe["aggregation"] == "r_softgate":
                train_fitted_fusions[recipe["id"]] = fit_r_softgate(
                    recipe, full_ranking_scores_by_method, selected, labels, adapter.p2i,
                )
            elif recipe["aggregation"] == "pool_baseline":
                train_fitted_fusions[recipe["id"]] = fit_pool_baseline_fusion(
                    recipe, full_ranking_scores_by_method, selected, labels, adapter.p2i,
                )

        for split, indices in split_idx.items():
            gt_by_attr, joint_gt = split_gt[split]
            ranking_scores_by_method = dict(embedding_score_sets[split])
            for method in REPORTED_METHODS:
                attrs = {attr: full_attrs_by_method[method][attr][indices] for attr in H.ATTRS}
                variants = joint_scores(attrs)
                ranking_metrics = {attr: metrics(attrs[attr], gt_by_attr[attr]) for attr in H.ATTRS}
                for rule in JOINT_VARIANTS:
                    joint = (H.joint_score(attrs, calibrator=calibrators[method])
                             if rule in {"train_calibrator", "joint_logreg"} else variants[rule])
                    variant_id = method if rule == "product" else f"{method}__{rule}"
                    ranking_scores_by_method[variant_id] = ranking_score_set(attrs, joint)
                    variant_metrics = ({**ranking_metrics, "joint": metrics(joint, joint_gt)}
                                       if rule == "product"
                                       else {"joint": metrics(joint, joint_gt)})
                    rows.append({
                        "stage": stage, "split": split, "seed": seed,
                        "method": variant_id, "base_method": method,
                        "joint_rule": rule, "metrics": variant_metrics,
                    })
                if method == "pu_ranking":
                    calibrated = next((
                        f"pu_ranking__{rule}" for rule in ("joint_logreg", "train_calibrator")
                        if f"pu_ranking__{rule}" in ranking_scores_by_method
                    ), None)
                    if calibrated:
                        ranking_scores_by_method["pu_ranking_calib"] = (
                            ranking_scores_by_method[calibrated]
                        )
                if split == "gallery":
                    joint_default = ranking_scores_by_method[method][H.JOINT_KEY]
                    order = np.argsort(-joint_default)[:50]
                    top50[method][f"seed_{seed}"] = [
                        {"rank": rank + 1, "image": gallery[int(i)],
                         "score": float(joint_default[int(i)]), "gt": int(joint_gt[int(i)])}
                        for rank, i in enumerate(order)
                    ]

            for recipe in FUSION_RECIPES:
                fitted = train_fitted_fusions.get(recipe["id"])
                fused = apply_fusion_recipe(recipe, ranking_scores_by_method, fitted=fitted)
                fusion_metrics = {
                    attr: metrics(fused[H.ranking_key_for_combo((attr,))], gt_by_attr[attr])
                    for attr in H.ATTRS
                }
                fusion_metrics["joint"] = metrics(fused[H.JOINT_KEY], joint_gt)
                rows.append({
                    "stage": stage, "split": split, "seed": seed,
                    "method": recipe["id"], "family": "fusion",
                    "aggregation": recipe["aggregation"],
                    "joint_rule": recipe.get("joint_rule", recipe["aggregation"]),
                    "training_strategy": recipe.get("training_strategy"),
                    "pool_id": recipe.get("pool_id"),
                    "pool_strategy": recipe.get("pool_strategy"),
                    "pool_aggregation": recipe.get("pool_aggregation"),
                    "metrics": fusion_metrics,
                    "members": list(recipe["members"]),
                    "resolved_members": resolve_fusion_members(recipe),
                    "member_joint_rule": recipe.get("member_joint_rule", "product"),
                })
                if fitted is not None:
                    rows[-1]["fusion_fit"] = fusion_fit_output(fitted)
                if split == "gallery":
                    joint = fused[H.JOINT_KEY]
                    order = np.argsort(-joint)[:50]
                    top50[recipe["id"]][f"seed_{seed}"] = [
                        {"rank": rank + 1, "image": gallery[int(i)],
                         "score": float(joint[int(i)]), "gt": int(joint_gt[int(i)])}
                        for rank, i in enumerate(order)
                    ]

    # Learned probes and every train-fitted fusion above consume the exact same
    # validated ``selected``/``labels`` objects.  Persist their shared identity
    # on every output row so initial runs and resumes can be audited directly.
    for row in rows:
        row["supervision_hash"] = supervision_digest
    write_json(task_out / f"{stage}_metrics.json", rows)
    write_json(task_out / f"{stage}_top50.json", top50)
    return {"rows": len(rows), "top50_methods": sorted(top50)}


def export_posthoc_scores(
    score_root: Path,
    task: dict[str, Any],
    adapter: Any,
    gallery: list[str],
    test: list[str],
    selected: list[str],
    labels: list[dict[str, Any]],
    stage_scores: dict[str, dict[str, np.ndarray]],
    seeds: list[int],
) -> None:
    """Export leakage-safe inputs consumed by cross-task post-hoc joint rules."""
    if not POSTHOC_JOINT_RULES:
        return
    label_by_path = {str(row["image"]): row for row in labels}
    missing_paths = [path for path in selected if path not in label_by_path]
    if missing_paths:
        raise RuntimeError(f"posthoc export received missing supervision paths: {missing_paths}")
    train_paths = list(selected)
    train_labels = np.asarray([
        [int(label_by_path[path].get(attr, -1)) for attr in H.ATTRS]
        for path in train_paths
    ], dtype=np.int8)
    complete = np.all((train_labels == 0) | (train_labels == 1), axis=1)
    if not bool(np.all(complete)):
        bad_paths = [path for path, keep in zip(train_paths, complete) if not keep]
        raise RuntimeError(f"posthoc export received incomplete supervision: {bad_paths}")
    if not train_paths:
        raise RuntimeError("posthoc export has no completely labeled training rows")
    gallery_idx = np.asarray([adapter.p2i[path] for path in gallery], dtype=np.int64)
    test_idx = np.asarray([adapter.p2i[path] for path in test], dtype=np.int64)
    train_idx = np.asarray([adapter.p2i[path] for path in train_paths], dtype=np.int64)
    _, gallery_joint = gt_for(gallery, adapter.gt_by_attr)
    _, test_joint = gt_for(test, adapter.gt_by_attr)
    for method in POSTHOC_BASE_METHODS:
        for seed_pos, seed in enumerate(seeds):
            payload: dict[str, np.ndarray] = {
                "train_paths": np.asarray(train_paths),
                "train_labels": train_labels,
                "train_attributes": np.asarray(H.ATTRS),
                "gallery_paths": np.asarray(gallery),
                "test_paths": np.asarray(test),
                "gallery_joint_gt": gallery_joint.astype(np.int8),
                "test_joint_gt": test_joint.astype(np.int8),
            }
            for attr in H.ATTRS:
                slug = H.ATTR_KEY[attr]
                scores = np.asarray(stage_scores[method][attr][seed_pos], dtype=np.float32)
                payload[f"train_attr_{slug}"] = scores[train_idx]
                payload[f"gallery_attr_{slug}"] = scores[gallery_idx]
                payload[f"test_attr_{slug}"] = scores[test_idx]
            target = (
                score_root / "scores" / "per_task" / task["dataset"] / task["task"]
                / method / f"seed_{seed}.npz"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            temporary.replace(target)


def run_posthoc(run_root, tasks, seeds, epochs):
    """The paper release has no cross-task fitted posthoc rules."""
    if POSTHOC_JOINT_RULES:
        raise ValueError("Posthoc exploration is outside the paper release")
    result = {"status": "disabled", "rules": []}
    write_json(run_root / "posthoc/status.json", result)
    return result


def aggregate_report(
    run_root: Path,
    manifest_rows: list[dict],
    stop_order: int,
    *,
    stages: tuple[str, ...] = STAGES,
) -> None:
    events = []
    task_summaries = []
    all_metric_rows = []
    for row in manifest_rows:
        if int(row["order"]) > stop_order:
            continue
        task_dir = run_root / "tasks" / f"{int(row['order']):03d}_{row['dataset']}_{row['task']}"
        event_file = task_dir / "cache_events.json"
        summary_file = task_dir / "summary.json"
        if event_file.is_file():
            events.extend(json.loads(event_file.read_text(encoding="utf-8")))
        if summary_file.is_file():
            task_summaries.append(json.loads(summary_file.read_text(encoding="utf-8")))
        for stage in stages:
            metric_file = task_dir / f"{stage}_metrics.json"
            if metric_file.is_file():
                for metric_row in json.loads(metric_file.read_text(encoding="utf-8")):
                    metric_row.update(
                        order=row["order"], dataset=row["dataset"], task=row["task"],
                        scope=row.get("scope", "unspecified"),
                        difficulty=row.get("difficulty", row.get("scope", "unspecified")),
                    )
                    all_metric_rows.append(metric_row)

    cache_counts = Counter((e["stage"], e["method"], e["status"]) for e in events)
    hard_tasks = []
    for summary in task_summaries:
        best = summary.get("best_test_ap")
        if best is not None and best < 0.6:
            hard_tasks.append({"order": summary["order"], "task": summary["task"], "best_test_ap": best})

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "stopped_after_order": stop_order,
        "completed_tasks": sum(s.get("status") == "completed" for s in task_summaries),
        "failed_tasks": [s for s in task_summaries if s.get("status") != "completed"],
        "cache_counts": [
            {"stage": stage, "method": method, "status": status, "count": count}
            for (stage, method, status), count in sorted(cache_counts.items())
        ],
        "all_methods_below_0_6_tasks": hard_tasks,
        "vqa_calls": sum(int(s.get("vqa_attempted", 0)) for s in task_summaries),
        "vqa_successful": sum(int(s.get("vqa_successful", 0)) for s in task_summaries),
        "vqa_failures": sum(int(s.get("vqa_failed", 0)) for s in task_summaries),
        "vqa_manual_content_filter": sum(
            int(s.get("vqa_manual_content_filter", 0)) for s in task_summaries
        ),
        "vqa_manual_content_filter_fraction": (
            sum(int(s.get("vqa_manual_content_filter", 0)) for s in task_summaries)
            / max(1, sum(int(s.get("vqa_successful", 0)) for s in task_summaries))
        ),
    }
    write_json(run_root / "batch_summary.json", report)
    write_json(run_root / "all_metrics.json", all_metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite-config", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--probebank-root", type=Path, default=DEFAULT_PROBEBANK_ROOT)
    parser.add_argument(
        "--disable-cross-task-reuse",
        action="store_true",
        help=(
            "isolate every task in its own ProbeBank namespace; same-task resume is "
            "allowed, but no probe produced by another task can be read"
        ),
    )
    parser.add_argument(
        "--iterative-only",
        action="store_true",
        help=(
            "run only isolated iterative acquisition/training; ignore all task-local "
            "random/two-stage supervision, VQA caches, stage aliases, and active "
            "evaluation manifests"
        ),
    )
    parser.add_argument(
        "--exact-static-prefix-root",
        type=Path,
        help=(
            "for Random/Two-stage budget curves, consume only hash-bound run-local "
            "train/permanent-audit manifests beneath ROOT/tasks and the configured "
            "strict canonical label source"
        ),
    )
    parser.add_argument("--start-order", type=int, default=1)
    parser.add_argument("--stop-order", type=int, default=None)
    parser.add_argument("--orders", type=int, nargs="+", default=None,
                        help="explicit non-contiguous task orders; overrides start/stop range")
    parser.add_argument("--scope", choices=("all", "primary16", "auxiliary23"), default="all")
    parser.add_argument("--label-size", type=int, default=500)
    parser.add_argument("--sup-size", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=H.EPOCHS,
                        help="training epochs for every learned/ReCAP probe (use 1 for smoke tests)")
    parser.add_argument("--vqa-config", default="config.yaml")
    parser.add_argument("--vqa-python", default=None)
    parser.add_argument("--vqa-workers", type=int, default=64)
    parser.add_argument("--iterative-rounds", type=int, default=None,
                        help=(
                            "deprecated acquisition bookkeeping only; training always uses the "
                            "complete train_labeled_indices.json"
                        ))
    parser.add_argument("--allow-vqa", action="store_true",
                        help="allow new API labeling; by default supervision must already exist")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=None,
                        help="stages to run; by default use every stage available for each task")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-posthoc", action="store_true",
                        help="export posthoc inputs but do not fit cross-task joint rules")
    args = parser.parse_args()
    if args.sup_size > args.label_size:
        raise SystemExit("--sup-size must be <= --label-size")
    if args.iterative_rounds is not None and args.iterative_rounds < 1:
        raise SystemExit("--iterative-rounds must be >= 1")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.iterative_only:
        if args.stages is not None and args.stages != [ITERATIVE_STAGE]:
            raise SystemExit("--iterative-only only accepts --stages iterative")
        if not args.disable_cross_task_reuse:
            raise SystemExit("--iterative-only requires --disable-cross-task-reuse")
    if args.exact_static_prefix_root is not None:
        if args.iterative_only:
            raise SystemExit("--exact-static-prefix-root cannot be combined with --iterative-only")
        if args.allow_vqa:
            raise SystemExit("--exact-static-prefix-root forbids --allow-vqa")
        if not args.disable_cross_task_reuse:
            raise SystemExit("--exact-static-prefix-root requires --disable-cross-task-reuse")
        if not args.stages or any(stage not in ("random", "two_stage") for stage in args.stages):
            raise SystemExit(
                "--exact-static-prefix-root requires explicit --stages random/two_stage"
            )

    H.EPOCHS = args.epochs

    suite = configure_suite(args.suite_config)
    H.validate_methods_for_backbone(BACKBONE, list(BASE_METHODS))
    attention_methods = [
        method for method in BASE_METHODS
        if H.TRAINED_METHODS[method] in ("attn", "attn_ens")
    ]
    conditioned_attention_methods = [
        method for method in attention_methods
        if method not in TEXT_FREE_ATTENTION_METHODS
    ]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_orders = set(args.orders or [])
    stop_order = (max(selected_orders) if selected_orders else args.stop_order) or max(
        int(row["order"]) for row in manifest["tasks"]
    )
    tasks = [row for row in manifest["tasks"]
             if ((int(row["order"]) in selected_orders) if selected_orders
                 else args.start_order <= int(row["order"]) <= stop_order)
             and (args.scope == "all" or row.get("scope") == args.scope)]
    if selected_orders - {int(row["order"]) for row in tasks}:
        raise SystemExit(f"requested task orders are missing from the selected scope: "
                         f"{sorted(selected_orders - {int(row['order']) for row in tasks})}")
    run_root = args.run_root.resolve()
    bank = args.probebank_root.resolve()
    state_file = run_root / "run_state.json"
    if (run_root / "run_state.json").exists() and not args.resume:
        raise SystemExit(f"run already exists: {run_root}; use a new --run-root or --resume")
    bank.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(state_file, {
        "batch_id": manifest["batch_id"], "protocol_version": PROTOCOL_VERSION,
        "suite_id": suite["suite_id"], "suite_config": str(args.suite_config.resolve()),
        "start_order": args.start_order, "stop_order": stop_order, "scope": args.scope,
        "orders": sorted(selected_orders) if selected_orders else None,
        "seeds": args.seeds,
        # These two knobs belong to the static random/two-stage selector.  Do
        # not serialize their 500-row defaults for an iterative-only run: the
        # values are ignored there and make the audit record look as if 500
        # VQA labels were requested.
        "label_size": None if args.iterative_only else args.label_size,
        "sup_size": None if args.iterative_only else args.sup_size,
        "iterative_budget": ({
            "initial_vqa": 100,
            "vqa_per_refine_round": 50,
            "train_per_refine_round": 40,
            "permanent_audit_per_refine_round": 10,
            "nominal_stop_vqa": 250,
            "hard_ceiling_vqa": 300,
        } if args.iterative_only else None),
        "iterative_rounds": args.iterative_rounds, "epochs": args.epochs,
        "cross_task_probe_reuse": not args.disable_cross_task_reuse,
        "supervision_source_policy": (
            ITERATIVE_ONLY_SOURCE_POLICY
            if args.iterative_only
            else EXACT_STATIC_SOURCE_POLICY
            if args.exact_static_prefix_root is not None
            else "configured-multistage"
        ),
        "exact_static_prefix_root": (
            str(args.exact_static_prefix_root.resolve())
            if args.exact_static_prefix_root is not None else None
        ),
        "method_counts": {
            "embedding": len(EMBEDDING_METHODS), "learned": len(BASE_METHODS),
            "joint_rules": len(JOINT_VARIANTS), "fusion_recipes": len(FUSION_RECIPES),
            "posthoc_joint_rules": len(suite.get("posthoc_joint_rules", [])),
        },
        "status": "dry_run" if args.dry_run else "running",
    })

    failed_orders: list[int] = []
    for task in tasks:
        order = int(task["order"])
        task_out = run_root / "tasks" / f"{order:03d}_{task['dataset']}_{task['task']}"
        task_bank = (
            bank / "task_isolated" / f"{order:03d}_{task['dataset']}_{task['task']}"
            if args.disable_cross_task_reuse else bank
        )
        available_stages = list(task.get("stages", STAGES))
        if args.iterative_only:
            if ITERATIVE_STAGE not in available_stages:
                raise SystemExit(
                    f"task {task['task']} does not declare the iterative logical stage"
                )
            requested_stages = [ITERATIVE_STAGE]
        else:
            requested_stages = [stage for stage in (args.stages or available_stages)
                                if stage in available_stages]
        summary_file = task_out / "summary.json"
        old = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else None
        if args.resume and summary_file.is_file():
            complete_stages = [stage for stage in requested_stages
                               if (task_out / f"{stage}_metrics.json").is_file()]
            if (old or {}).get("suite_id") == suite["suite_id"] and len(complete_stages) == len(requested_stages):
                print(
                    f"=== order {order}: resume will revalidate completed stages="
                    f"{complete_stages} and supervision hash for {task['task']}"
                )
        print(f"\n=== order {order}: {task['dataset']}/{task['task']} ({task.get('scope', '')})")
        summary = {**task, **(old or {}), "status": "running",
                   "requested_stages": requested_stages, "suite_id": suite["suite_id"],
                   "supervision_source_policy": (
                       ITERATIVE_ONLY_SOURCE_POLICY
                       if args.iterative_only
                       else EXACT_STATIC_SOURCE_POLICY
                       if args.exact_static_prefix_root is not None
                       else "configured-multistage"
                   )}
        summary.setdefault("stage_status", {})
        if args.iterative_only:
            summary["stage_status"] = {
                key: value for key, value in summary["stage_status"].items()
                if key == ITERATIVE_STAGE
            }
            for audit_key in ("supervision_audits", "supervision_hashes"):
                existing = summary.get(audit_key, {})
                summary[audit_key] = {
                    key: value for key, value in existing.items()
                    if key == ITERATIVE_STAGE
                }
        if args.iterative_only:
            summary.update(
                vqa_new_successful=0,
                vqa_successful=0,
                vqa_failed=0,
                vqa_attempted=0,
            )
        else:
            summary.setdefault("vqa_new_successful", 0)
            summary.setdefault("vqa_successful", 0)
            summary.setdefault("vqa_failed", 0)
            summary.setdefault("vqa_attempted", 0)
        event_file = task_out / "cache_events.json"
        cache_events = json.loads(event_file.read_text(encoding="utf-8")) if event_file.is_file() else []
        cache_events = (
            [] if args.iterative_only else
            [event for event in cache_events
             if event.get("stage") not in set(requested_stages)]
        )
        adapter = None
        stage = "setup"
        try:
            H.configure(task["dataset"], task["task"])
            adapter = H.ADAPTER
            emb, emb_meta = H.load_backbone_embeddings(adapter, BACKBONE)
            patches = None
            patch_meta = None
            text_queries: dict[str, torch.Tensor] = {}
            if attention_methods:
                patches, patch_meta = H.load_backbone_patches(adapter, BACKBONE)
                if conditioned_attention_methods:
                    text_queries = cached_attribute_text_queries(
                        list(H.ATTRS), BACKBONE, H.BACKBONE_TEXT_MODEL[BACKBONE],
                    )
            if not adapter.query_idx:
                raise ValueError(
                    "task has no query images mapped into the processed database; "
                    "restore/precompute its query images before training"
                )
            query_idx = np.asarray(adapter.query_idx, dtype=np.int64)
            if query_idx.min() < 0 or query_idx.max() >= len(emb):
                raise ValueError(f"query indices are outside the embedding table: {adapter.query_idx}")
            if not np.isfinite(emb).all():
                bad_rows = int((~np.isfinite(emb)).any(axis=1).sum())
                raise ValueError(f"processed embedding table contains {bad_rows} non-finite rows")
            if not np.isfinite(emb[query_idx]).all():
                raise ValueError("query embeddings contain non-finite values")
            emb_norm = H.l2norm(emb)
            paths = database_paths(adapter)
            gallery, test, train_pool = _frozen_split(
                adapter,
                emb_norm,
                adapter.p2i,
                adapter.gt_by_attr,
                use_persisted_evaluation=not args.iterative_only,
            )
            if args.dry_run:
                summary.update(status="dry_run", gallery=len(gallery), test=len(test),
                               train_pool=len(train_pool), embedding=emb_meta,
                               patch_tokens=patch_meta)
                write_json(summary_file, summary)
                continue

            for stage in requested_stages:
                staging_root = task_out / "cache_staging" / stage
                new_vqa_calls = 0
                selected_manifest_path: Path | None = None
                current_vqa_dir = task_out / "vqa" / stage
                source_contract: dict[str, Any] | None = None
                static_source_contract: dict[str, Any] | None = None
                static_permanent_audit: list[str] | None = None
                if args.iterative_only:
                    # Deliberately ignore task.supervision_stages, including a
                    # logical iterative -> static/two-stage alias.
                    source_stage = ITERATIVE_STAGE
                    canonical_vqa_dir = current_vqa_dir
                    vqa_dir = current_vqa_dir
                    reused_sources: list[Path] = []
                    new_sources: list[Path] = []
                else:
                    source_stage = task.get("supervision_stages", {}).get(stage, stage)
                    if not isinstance(source_stage, str) or not source_stage:
                        raise RuntimeError(
                            f"task {task['task']} has no configured supervision source for {stage}"
                        )
                    canonical_vqa_dir = adapter.task_root / "supervision" / source_stage
                    if args.exact_static_prefix_root is not None:
                        canonical_vqa_dir = exact_static_source_directory(
                            task, args.manifest, adapter.task_root, source_stage,
                        )
                    vqa_dir = (canonical_vqa_dir if canonical_vqa_dir.is_dir()
                               else current_vqa_dir)
                    if args.exact_static_prefix_root is not None:
                        # Fail closed: exact-static must never merge labels from
                        # another acquisition stage or legacy task history.
                        reused_sources = []
                        new_sources = []
                    else:
                        label_stages = task.get("supervision_label_stages", {}).get(
                            stage, [source_stage],
                        )
                        reused_sources = list(H.discover_task_vqa_sources(adapter.task_root))
                        for label_stage in label_stages:
                            label_dir = adapter.task_root / "supervision" / label_stage
                            reused_sources.extend(sorted(label_dir.rglob("reused_labels.json")))
                            reused_sources.extend(sorted(label_dir.rglob("*_results.jsonl")))
                        new_sources = sorted(current_vqa_dir.rglob("*_results.jsonl"))
                if stage == ITERATIVE_STAGE:
                    if args.iterative_only:
                        exact_manifest = current_vqa_dir / "split" / "train_labeled_indices.json"
                        if not exact_manifest.is_file() and args.allow_vqa:
                            attempted_vqa, failed_vqa = run_iterative_supervision(
                                task, task_out, adapter, args,
                            )
                            new_vqa_calls = attempted_vqa - failed_vqa
                        if not exact_manifest.is_file():
                            raise RuntimeError(
                                "iterative-only supervision is missing its run-local exact "
                                f"train manifest: {exact_manifest}"
                            )
                        (
                            selected_manifest_path,
                            selected,
                            new_sources,
                            source_contract,
                        ) = iterative_run_supervision_inputs(
                            task_out,
                            expected_dataset=task["dataset"],
                            expected_task=task["task"],
                            expected_backbone=BACKBONE,
                            expected_attributes=list(H.ATTRS),
                            expected_paths=paths,
                            expected_query_indices=list(adapter.query_idx),
                            expected_train_pool=train_pool,
                            expected_candidate_test=test,
                        )
                        task_bank = iterative_only_probebank_root(
                            bank, source_contract["run_identity_sha256"], order,
                            task["dataset"], task["task"],
                        )
                        staging_root = (
                            task_out / "cache_staging" / "iterative_only"
                            / source_contract["run_identity_sha256"] / stage
                        )
                    else:
                        selected_manifest_path = iterative_selected_manifest(
                            task_out, canonical_vqa_dir,
                        )
                        if args.iterative_rounds is not None:
                            print(
                                "  [supervision] --iterative-rounds no longer truncates training; "
                                "using the complete train_labeled_indices.json"
                            )
                        if selected_manifest_path is None and args.allow_vqa:
                            attempted_vqa, failed_vqa = run_iterative_supervision(
                                task, task_out, adapter, args,
                            )
                            new_vqa_calls = attempted_vqa - failed_vqa
                            selected_manifest_path = iterative_selected_manifest(
                                task_out, canonical_vqa_dir,
                            )
                            new_sources = sorted(current_vqa_dir.rglob("*_results.jsonl"))
                        if selected_manifest_path is None:
                            raise RuntimeError(
                                "iterative supervision is missing the complete "
                                f"train_labeled_indices.json: {current_vqa_dir} or {canonical_vqa_dir}"
                            )
                        selected = json.loads(
                            selected_manifest_path.read_text(encoding="utf-8")
                        )
                else:
                    if args.exact_static_prefix_root is not None:
                        prefix_task_out = (
                            args.exact_static_prefix_root.resolve() / "tasks"
                            / f"{order:03d}_{task['dataset']}_{task['task']}"
                        )
                        (
                            selected_manifest_path,
                            selected,
                            static_permanent_audit,
                            reused_sources,
                            static_source_contract,
                        ) = exact_static_supervision_inputs(
                            prefix_task_out,
                            canonical_vqa_dir,
                            expected_order=order,
                            expected_dataset=task["dataset"],
                            expected_task=task["task"],
                            expected_stage=stage,
                            expected_source_stage=str(source_stage),
                            expected_attributes=list(H.ATTRS),
                            logical_budget=args.label_size,
                            expected_train_count=args.sup_size,
                        )
                    else:
                        # Strict supervision may replace content-filtered rows
                        # with the next item in the frozen rank. Reuse that
                        # audited selection instead of reconstructing it.
                        selected_manifest_path = vqa_dir / f"selected_candidates_{args.label_size}.json"
                        if selected_manifest_path.is_file():
                            selected = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
                            selected = selected[:args.sup_size]
                        else:
                            selected_manifest_path = None
                            selected = candidate_paths(
                                stage, train_pool, emb_norm, adapter.p2i,
                                list(adapter.query_idx), args.label_size,
                            )[:args.sup_size]
                        if not _unique_existing_sources(reused_sources + new_sources) and args.allow_vqa:
                            _, _, new_vqa_calls, _ = run_vqa(
                                adapter, selected, list(H.ATTRS), task_out / "vqa" / stage,
                                args.vqa_config, args.vqa_python, args.vqa_workers,
                            )
                            new_sources = sorted(current_vqa_dir.rglob("*_results.jsonl"))
                        elif not _unique_existing_sources(reused_sources + new_sources):
                            raise RuntimeError(f"static supervision is missing: {vqa_dir}")

                selected, labels_map, labels, supervision_digest, supervision_audit = (
                    resolve_training_supervision(
                        selected=selected,
                        reused_sources=reused_sources,
                        new_sources=new_sources,
                        vqa_to_key=adapter.vqa_to_key,
                        attrs=list(H.ATTRS),
                        audit_path=task_out / "supervision_audit" / f"{stage}.json",
                        selected_manifest=selected_manifest_path,
                    )
                )
                supervision_audit["evaluation_boundary"] = validate_frozen_test_isolation(
                    selected=selected,
                    gallery=gallery,
                    frozen_test=test,
                )
                if source_contract is not None:
                    supervision_audit["source_contract"] = source_contract
                    supervision_audit["evaluation_boundary"].update({
                        "policy": "seed42-candidate-test-minus-current-iterative-supervision-v1",
                        "task_active_manifest_consumed": False,
                        "candidate_test_row_count": len(test),
                        "excluded_current_iterative_rows": 0,
                    })
                if static_source_contract is not None:
                    if static_permanent_audit is None:
                        raise AssertionError("exact static permanent-audit rows were not resolved")
                    permanent_boundary = validate_frozen_test_isolation(
                        selected=static_permanent_audit,
                        gallery=gallery,
                        frozen_test=test,
                    )
                    all_labeled_boundary = validate_frozen_test_isolation(
                        selected=[*selected, *static_permanent_audit],
                        gallery=gallery,
                        frozen_test=test,
                    )
                    supervision_audit["source_contract"] = static_source_contract
                    supervision_audit["evaluation_boundary"].update({
                        "policy": "current-persisted-clean-test-v1",
                        "task_active_manifest_consumed": True,
                        "logical_label_budget": args.label_size,
                        "permanent_audit_row_count": static_source_contract["audit_count"],
                        "permanent_audit_gallery_membership_valid": permanent_boundary[
                            "gallery_membership_valid"
                        ],
                        "permanent_audit_test_overlap_count": permanent_boundary[
                            "frozen_test_overlap_count"
                        ],
                        "all_labeled_gallery_membership_valid": all_labeled_boundary[
                            "gallery_membership_valid"
                        ],
                        "all_labeled_test_overlap_count": all_labeled_boundary[
                            "frozen_test_overlap_count"
                        ],
                    })
                write_json(
                    task_out / "supervision_audit" / f"{stage}.json",
                    supervision_audit,
                )
                summary.setdefault("supervision_audits", {})[stage] = supervision_audit
                summary.setdefault("supervision_hashes", {})[stage] = supervision_digest
                summary["vqa_successful"] += len(selected)
                summary["vqa_new_successful"] += new_vqa_calls
                legacy_supervision_digest = legacy_supervision_hash(
                    selected, labels_map, list(H.ATTRS),
                )
                cache_identity = None
                if source_contract is not None:
                    cache_identity = {
                        "source_policy": ITERATIVE_ONLY_SOURCE_POLICY,
                        "acquisition_source_policy": ISOLATED_SOURCE_POLICY,
                        "train_pool_hash": stable_hash(train_pool),
                        "evaluation_split_policy": source_contract["evaluation_split_policy"],
                        "iterative_run_identity_sha256": source_contract["run_identity_sha256"],
                        "probebank_namespace_policy": "iterative-only-run-identity-v1",
                    }
                    legacy_supervision_digest = None
                elif static_source_contract is not None:
                    cache_identity = {
                        "source_policy": EXACT_STATIC_SOURCE_POLICY,
                        "static_prefix_protocol": static_source_contract["protocol_id"],
                        "static_logical_budget": static_source_contract["logical_budget"],
                        "static_train_count": static_source_contract["train_count"],
                        "static_audit_count": static_source_contract["audit_count"],
                        "static_prefix_provenance_sha256": static_source_contract[
                            "provenance_sha256"
                        ],
                        "probebank_namespace_policy": "exact-static-task-job-v1",
                    }
                    # Never accept a legacy cache that lacks this exact-static
                    # source identity, even when the selected label digest is
                    # coincidentally equal.
                    legacy_supervision_digest = None

                stage_scores: dict[str, dict[str, np.ndarray]] = {m: {} for m in BASE_METHODS}
                cached_entries: dict[tuple[str, str], tuple[dict, np.ndarray]] = {}
                staged_existing: set[tuple[str, str]] = set()
                retrain_incompatible: dict[tuple[str, str], str] = {}
                for method in BASE_METHODS:
                    for attr in H.ATTRS:
                        method_text_query = text_query_for_method(method, attr, text_queries)
                        feature_context = attention_feature_context(
                            method, patch_meta, method_text_query,
                        )
                        try:
                            cached = read_cache(
                                task_bank, task["dataset"], task["task"], stage, method, attr,
                                emb, paths, args.seeds, supervision_digest, args.epochs,
                                legacy_supervision_digest,
                                feature_context,
                                cache_identity,
                            )
                        except ValueError as exc:
                            # A corrected supervision set legitimately invalidates an
                            # older same-task probe.  Retrain it transactionally and
                            # archive the old entry; all other incompatibilities remain
                            # hard failures because they may indicate feature/data drift.
                            permitted_identity_mismatch = args.iterative_only and any(
                                marker in str(exc) for marker in (
                                    "source_policy=", "acquisition_source_policy=",
                                    "train_pool_hash=", "evaluation_split_policy=",
                                    "iterative_run_identity_sha256=",
                                    "probebank_namespace_policy=",
                                )
                            )
                            if "supervision_hash=" not in str(exc) and not permitted_identity_mismatch:
                                raise
                            cached = None
                            retrain_incompatible[(method, attr)] = str(exc)
                        status = (
                            "hit_legacy_label_digest"
                            if cached is not None and cached[0].get(
                                "_cache_supervision_hash_mode"
                            ) == "legacy_v0_exact_label_digest"
                            else "hit_seed_subset"
                            if cached is not None and cached[0].get("_cache_seed_mode") == "requested_subset_of_cached_superset"
                            else "retrain_incompatible_supervision"
                            if (method, attr) in retrain_incompatible
                            else "hit" if cached is not None else "miss"
                        )
                        event = {
                            "order": order, "stage": stage, "method": method,
                            "attribute": attr, "status": status,
                            "reuse_scope": (
                                "same_task_only" if args.disable_cross_task_reuse
                                else "configured_probebank"
                            ),
                        }
                        if (method, attr) in retrain_incompatible:
                            event["incompatibility"] = retrain_incompatible[(method, attr)]
                        cache_events.append(event)
                        if cached is None:
                            staged = read_cache(
                                staging_root, task["dataset"], task["task"], stage, method, attr,
                                emb, paths, args.seeds, supervision_digest, args.epochs,
                                legacy_supervision_digest,
                                feature_context,
                                cache_identity,
                            )
                            if staged is not None:
                                cached_entries[(method, attr)] = staged
                                staged_existing.add((method, attr))
                        else:
                            cached_entries[(method, attr)] = cached
                staged_entries = list(staged_existing)
                for method in BASE_METHODS:
                    for attr in H.ATTRS:
                        cached = cached_entries.get((method, attr))
                        if cached is None:
                            method_text_query = text_query_for_method(method, attr, text_queries)
                            feature_context = attention_feature_context(
                                method, patch_meta, method_text_query,
                            )
                            cached = train_cache_entry(
                                staging_root, task["dataset"], task["task"], stage, method,
                                attr, emb, paths,
                                adapter.p2i, train_pool, selected, labels, args.seeds,
                                supervision_digest, args.epochs,
                                patches=patches,
                                text_query=method_text_query,
                                feature_context=feature_context,
                                cache_identity=cache_identity,
                            )
                            staged_entries.append((method, attr))
                        stage_scores[method][attr] = cached[1]
                # Promote only complete entries. metadata.json was written last inside staging.
                for method, attr in staged_entries:
                    source = cache_dir(
                        staging_root, task["dataset"], task["task"], stage, method, attr,
                    )
                    target = cache_dir(
                        task_bank, task["dataset"], task["task"], stage, method, attr,
                    )
                    if not (source / "metadata.json").is_file():
                        raise RuntimeError(f"staged cache entry is incomplete: {source}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if (method, attr) not in retrain_incompatible:
                            raise RuntimeError(f"refusing to replace existing cache path: {target}")
                        quarantine = cache_dir(
                            task_out / "cache_quarantine", task["dataset"], task["task"],
                            stage, method, attr,
                        )
                        quarantine.parent.mkdir(parents=True, exist_ok=True)
                        if quarantine.exists():
                            raise RuntimeError(
                                f"cache quarantine already exists; refusing overwrite: {quarantine}"
                            )
                        target.replace(quarantine)
                        cache_events.append({
                            "order": order, "stage": stage, "method": method,
                            "attribute": attr, "status": "archived_incompatible_cache",
                            "source": str(target), "quarantine": str(quarantine),
                        })
                    source.replace(target)
                if stage == POSTHOC_STAGE:
                    export_posthoc_scores(
                        run_root / "posthoc_inputs", task, adapter, gallery, test,
                        selected, labels, stage_scores, args.seeds,
                    )
                evaluate_task(
                    adapter, emb_norm, paths, gallery, test, selected, labels,
                    stage_scores, args.seeds, stage, task_out, supervision_digest,
                )
                summary["stage_status"][stage] = "completed"

            metric_rows = []
            metric_stages = (ITERATIVE_STAGE,) if args.iterative_only else STAGES
            for stage in metric_stages:
                metric_file = task_out / f"{stage}_metrics.json"
                if metric_file.is_file():
                    metric_rows.extend(json.loads(metric_file.read_text(encoding="utf-8")))
            test_aps = [float(row["metrics"]["joint"]["ap"]) for row in metric_rows
                        if row["split"] == "test" and "joint" in row["metrics"]]
            refresh_vqa_counts(
                summary,
                task_out,
                adapter,
                list(H.ATTRS),
                task,
                iterative_only=args.iterative_only,
            )
            if args.exact_static_prefix_root is not None:
                # The task-local strict sources contain the complete historical
                # acquisition (normally 1,000 rows), whereas this run consumes
                # only the materialized logical prefix.  Keep the legacy summary
                # counters honest for this exact-static run and expose the
                # per-stage train/audit split explicitly.
                completed_static_stages = sum(
                    summary.get("stage_status", {}).get(name) == "completed"
                    for name in requested_stages
                )
                summary.update(
                    vqa_attempted=completed_static_stages * args.label_size,
                    vqa_successful=completed_static_stages * args.label_size,
                    vqa_failed=0,
                    vqa_manual_content_filter=0,
                    vqa_manual_content_filter_fraction=0.0,
                    exact_static_logical_budget_per_stage=args.label_size,
                    exact_static_train_count_per_stage=args.sup_size,
                    exact_static_permanent_audit_count_per_stage=(
                        args.label_size - args.sup_size
                    ),
                )
            all_complete = all((task_out / f"{stage}_metrics.json").is_file()
                               for stage in requested_stages)
            # A resumed task may inherit failure diagnostics from an earlier
            # attempt through ``old``.  Once the requested stages complete,
            # those stale fields no longer describe the current run state.
            summary.pop("error", None)
            summary.pop("traceback", None)
            summary.update(status="completed" if all_complete else "partial", gallery=len(gallery), test=len(test),
                           train_pool=len(train_pool), embedding=emb_meta,
                           patch_tokens=patch_meta,
                           best_test_ap=max(test_aps) if test_aps else None)
        except Exception as exc:
            failed_orders.append(order)
            if adapter is not None:
                refresh_vqa_counts(
                    summary,
                    task_out,
                    adapter,
                    list(H.ATTRS),
                    task,
                    iterative_only=args.iterative_only,
                )
            summary["stage_status"][stage] = "failed"
            summary.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc())
            print(f"  [failed] {summary['error']}")
        write_json(task_out / "cache_events.json", cache_events)
        write_json(summary_file, summary)
        aggregate_report(
            run_root,
            manifest["tasks"],
            stop_order,
            stages=((ITERATIVE_STAGE,) if args.iterative_only else STAGES),
        )

    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["status"] = "failed" if failed_orders else "completed_requested_range"
    state["failed_orders"] = failed_orders
    if not failed_orders and not args.dry_run and not args.skip_posthoc:
        try:
            state["posthoc"] = run_posthoc(run_root, tasks, args.seeds, args.epochs)
        except Exception as exc:
            state["status"] = "posthoc_failed"
            state["posthoc"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            write_json(state_file, state)
            raise
    elif args.skip_posthoc:
        state["posthoc"] = {"status": "skipped", "reason": "--skip-posthoc"}
    write_json(state_file, state)
    aggregate_report(
        run_root,
        manifest["tasks"],
        stop_order,
        stages=((ITERATIVE_STAGE,) if args.iterative_only else STAGES),
    )
    if failed_orders:
        raise SystemExit(f"failed task orders: {failed_orders}")
    print(f"\nCompleted requested range through order {stop_order}.")


if __name__ == "__main__":
    main()
