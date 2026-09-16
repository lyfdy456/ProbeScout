"""Strictly recover the original ProbeBank supervision retained in Development.

This module is intentionally independent of ``tuning_server.py``.  It reuses
the ProbeBank supervision resolver that produced the immutable score caches,
checks the recovered supervision identity against the cache metadata, aligns
the selected image paths to the Web gallery, and only then intersects them with
the Development mask.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_SUITE = Path(
    "configs/probe_suite.json"
)
DEFAULT_STAGE = "iterative"
REMOTE_ASSETS_ROOT = Path("outputs/formal23_allv2_remote_assets_v1/tasks")
PROBE_VALIDATION_FRACTION = 0.2
PROBE_VALIDATION_SEED = 0
PROBE_VALIDATION_PROTOCOL = "sklearn-stratified-train-test-split-v1"


class _UnmatchedCacheSupervisionHash(ValueError):
    """The cache is structurally valid but belongs to another label snapshot."""


@dataclass(frozen=True)
class OriginalSupervision:
    """Original supervision aligned to Web rows, with a DG-only training view."""

    indices: np.ndarray
    labels: np.ndarray
    # Complete original VQA-selected rows before the Web Development mask is
    # applied.  They support provenance display and the separate, reproducible
    # Probe 80:20 replay; legacy DG-only fitting still uses ``indices``/labels.
    selected_indices: np.ndarray
    selected_labels: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class ProbeValidationSplit:
    """One reproducible target-level replay of the Probe 80:20 protocol."""

    fit_indices: np.ndarray
    fit_labels: np.ndarray
    validation_indices: np.ndarray
    validation_labels: np.ndarray
    audit: dict[str, Any]


def build_probe_validation_split(
    original: OriginalSupervision,
    *,
    target_id: str,
    target_kind: str,
    seed: int = PROBE_VALIDATION_SEED,
    validation_fraction: float = PROBE_VALIDATION_FRACTION,
) -> ProbeValidationSplit:
    """Replay the original Probe split rule on the complete VQA supervision.

    Attribute targets exactly reproduce the seed-specific split membership used
    by all eight Probe learners.  There was no historical Joint Probe head, so a
    derived target uses the same algorithm over its already recovered AND label
    and is explicitly marked as a Probe-style rather than historical replay.
    """

    from sklearn.model_selection import train_test_split

    source_indices = np.asarray(original.selected_indices, dtype=np.int64).reshape(-1)
    source_labels = np.asarray(original.selected_labels, dtype=np.uint8).reshape(-1)
    if source_indices.shape != source_labels.shape or source_indices.size == 0:
        raise ValueError("complete original supervision rows and labels must align")
    if len(set(int(value) for value in source_indices)) != int(source_indices.size):
        raise ValueError("complete original supervision rows must be unique")
    if not np.isin(source_labels, (0, 1)).all():
        raise ValueError("complete original supervision labels must be binary")
    positive_count = int(np.count_nonzero(source_labels))
    negative_count = int(source_labels.size - positive_count)
    if positive_count < 2 or negative_count < 2:
        raise ValueError(
            "Probe Validation requires at least two positive and two negative "
            "original supervision rows"
        )
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")

    source_positions = np.arange(source_indices.size, dtype=np.int64)
    fit_positions, validation_positions = train_test_split(
        source_positions,
        test_size=float(validation_fraction),
        random_state=int(seed),
        stratify=source_labels,
    )
    # Keep the recovered supervision order inside each partition.  Membership
    # is identical to sklearn's output; ordering no longer depends on its
    # internal shuffle implementation and is easier to audit downstream.
    fit_positions = np.sort(np.asarray(fit_positions, dtype=np.int64))
    validation_positions = np.sort(np.asarray(validation_positions, dtype=np.int64))
    fit_indices = np.asarray(source_indices[fit_positions], dtype=np.int64)
    fit_labels = np.asarray(source_labels[fit_positions], dtype=np.uint8)
    validation_indices = np.asarray(source_indices[validation_positions], dtype=np.int64)
    validation_labels = np.asarray(source_labels[validation_positions], dtype=np.uint8)
    if set(fit_indices.tolist()).intersection(validation_indices.tolist()):
        raise RuntimeError("Probe fit and validation rows overlap")
    if set(fit_indices.tolist()).union(validation_indices.tolist()) != set(
        source_indices.tolist()
    ):
        raise RuntimeError("Probe fit and validation rows do not cover supervision")

    replay_kind = (
        "exact-attribute-seed0"
        if str(target_kind) == "attribute" and int(seed) == 0
        else (
            "exact-attribute-seed"
            if str(target_kind) == "attribute"
            else "derived-joint-probe-style"
        )
    )
    def partition_fingerprint(name: str, rows: np.ndarray, labels: np.ndarray) -> str:
        payload = {
            "partition": name,
            "protocol": PROBE_VALIDATION_PROTOCOL,
            "seed": int(seed),
            "validationFraction": float(validation_fraction),
            "targetId": str(target_id),
            "targetKind": str(target_kind),
            "rows": [
                [int(row), int(label)]
                for row, label in zip(rows, labels, strict=True)
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    source_fingerprint = partition_fingerprint(
        "source", source_indices, source_labels
    )
    fit_fingerprint = partition_fingerprint("fit", fit_indices, fit_labels)
    validation_fingerprint = partition_fingerprint(
        "validation", validation_indices, validation_labels
    )
    audit = {
        "schemaVersion": 1,
        "protocol": PROBE_VALIDATION_PROTOCOL,
        "replayKind": replay_kind,
        "seed": int(seed),
        "validationFraction": float(validation_fraction),
        "targetId": str(target_id),
        "targetKind": str(target_kind),
        "sourceCount": int(source_indices.size),
        "sourcePositiveCount": positive_count,
        "sourceNegativeCount": negative_count,
        "fitCount": int(fit_indices.size),
        "fitPositiveCount": int(np.count_nonzero(fit_labels)),
        "fitNegativeCount": int(fit_labels.size - np.count_nonzero(fit_labels)),
        "validationCount": int(validation_indices.size),
        "validationPositiveCount": int(np.count_nonzero(validation_labels)),
        "validationNegativeCount": int(
            validation_labels.size - np.count_nonzero(validation_labels)
        ),
        "sourceFingerprint": source_fingerprint,
        "fitFingerprint": fit_fingerprint,
        "validationFingerprint": validation_fingerprint,
        "sourceSupervisionHash": original.audit.get("recoveredSupervisionHash"),
    }
    for values in (
        fit_indices,
        fit_labels,
        validation_indices,
        validation_labels,
    ):
        values.setflags(write=False)
    return ProbeValidationSplit(
        fit_indices=fit_indices,
        fit_labels=fit_labels,
        validation_indices=validation_indices,
        validation_labels=validation_labels,
        audit=audit,
    )


def _load_probebank_modules(experiment_root: Path) -> tuple[Any, Any]:
    linear_root = experiment_root / "probe_learning"
    scripts_root = linear_root / "scripts"
    for path in (linear_root, scripts_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    import run_probebank_batch as bank  # type: ignore
    import run_retrieval_harness as harness  # type: ignore

    return bank, harness


def _task_identity(task_config: Mapping[str, Any], adapter: Any) -> tuple[int, str, str]:
    try:
        order = int(task_config["order"])
        dataset = str(task_config["dataset"])
        task = str(task_config["task"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("task_config must contain integer order, dataset and task") from error
    if getattr(adapter, "dataset", dataset) != dataset or getattr(adapter, "task", task) != task:
        raise ValueError("task_config identity does not match the configured adapter")
    return order, dataset, task


def _modeled_attributes(adapter: Any) -> tuple[str, ...]:
    gt_by_attr = getattr(adapter, "gt_by_attr", None)
    if not isinstance(gt_by_attr, Mapping) or not gt_by_attr:
        raise ValueError("adapter must expose an ordered, non-empty gt_by_attr mapping")
    attributes = tuple(str(value) for value in gt_by_attr.keys())
    if len(set(attributes)) != len(attributes):
        raise ValueError("adapter attribute names must be unique")
    return attributes


def _resolve_target_attribute(
    adapter: Any,
    attributes: tuple[str, ...],
    target_attribute: str | None,
) -> str | None:
    if target_attribute is None or str(target_attribute).lower() == "joint":
        return None
    requested = str(target_attribute)
    exact = [attribute for attribute in attributes if attribute == requested]
    if len(exact) == 1:
        return exact[0]
    insensitive = [
        attribute for attribute in attributes if attribute.lower() == requested.lower()
    ]
    if len(insensitive) == 1:
        return insensitive[0]
    slugs = tuple(str(value) for value in getattr(adapter, "key_slugs", ()))
    if len(slugs) == len(attributes):
        matches = [
            attribute
            for attribute, slug in zip(attributes, slugs)
            if slug.lower() == requested.lower()
        ]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"unknown supervision target {requested!r}; expected Joint or {attributes}")


def _source_supervision(
    bank: Any,
    harness: Any,
    task_config: Mapping[str, Any],
    adapter: Any,
    attributes: tuple[str, ...],
    stage: str,
) -> tuple[
    list[str],
    dict[str, dict[str, int]],
    str,
    str,
    dict[str, Any],
    Path,
    Path | None,
]:
    source_stage = str(
        (task_config.get("supervision_stages") or {}).get(stage, stage)
    )
    canonical_dir = Path(adapter.task_root) / "supervision" / source_stage
    selected_manifest = bank.iterative_selected_manifest(
        Path(adapter.task_root) / ".tuning_supervision_no_run",
        canonical_dir,
    )
    if selected_manifest is None:
        raise FileNotFoundError(
            f"no unambiguous original {stage} train_labeled_indices.json for {adapter.task}"
        )
    selected = json.loads(selected_manifest.read_text(encoding="utf-8"))

    label_stages = (task_config.get("supervision_label_stages") or {}).get(
        stage, [source_stage]
    )
    reused_sources = list(harness.discover_task_vqa_sources(adapter.task_root))
    for label_stage in label_stages:
        label_dir = Path(adapter.task_root) / "supervision" / str(label_stage)
        reused_sources.extend(sorted(label_dir.rglob("reused_labels.json")))
        reused_sources.extend(sorted(label_dir.rglob("*_results.jsonl")))

    with tempfile.TemporaryDirectory(prefix="pcp-original-supervision-") as temporary:
        selected_paths, labels_map, _, supervision_hash, strict_audit = (
            bank.resolve_training_supervision(
                selected=selected,
                reused_sources=reused_sources,
                new_sources=[],
                vqa_to_key=adapter.vqa_to_key,
                attrs=list(attributes),
                audit_path=Path(temporary) / "strict-audit.json",
                selected_manifest=selected_manifest,
            )
        )
    if strict_audit.get("status") != "valid":
        raise RuntimeError("original ProbeBank supervision did not pass strict validation")
    legacy_supervision_hash = bank.legacy_supervision_hash(
        selected_paths, labels_map, list(attributes)
    )
    return (
        selected_paths,
        labels_map,
        supervision_hash,
        legacy_supervision_hash,
        strict_audit,
        selected_manifest,
        None,
    )


def _trusted_audit_path(
    experiment_root: Path,
    order: int,
    dataset: str,
    task: str,
) -> Path:
    task_key = f"{order:03d}_{dataset}_{task}"
    return experiment_root / REMOTE_ASSETS_ROOT / task_key / "supervision_audit.json"


def _require_source_path(
    raw_value: Any,
    *,
    allowed_root: Path,
    description: str,
) -> Path:
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"invalid {description} in saved supervision audit")
    path = Path(raw_value).resolve()
    root = allowed_root.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(
            f"saved {description} is unavailable or escapes its training-only root: {path}"
        )
    return path


def _audited_source_supervision(
    experiment_root: Path,
    bank: Any,
    task_config: Mapping[str, Any],
    adapter: Any,
    attributes: tuple[str, ...],
    stage: str,
    source_audit_path: Path | None = None,
) -> tuple[
    list[str],
    dict[str, dict[str, int]],
    str,
    str,
    dict[str, Any],
    Path,
    Path,
] | None:
    """Replay a saved run-local training snapshot from its explicit audit.

    A few Formal-23 caches were trained with labels created inside the original
    run rather than copied back into the dataset's canonical supervision tree.
    The remote-assets snapshot retains those inputs and its strict audit.  It
    is accepted only by replaying the resolver from the audit's exact files,
    recomputing every label/hash/count, and later matching all cache metadata.
    """
    order, dataset, task = _task_identity(task_config, adapter)
    if source_audit_path is None:
        audit_path = _trusted_audit_path(experiment_root, order, dataset, task)
    else:
        audit_path = Path(source_audit_path).resolve()
        task_key = f"{order:03d}_{dataset}_{task}"
        experiment_root = experiment_root.resolve()
        if (
            not audit_path.is_relative_to(experiment_root)
            or audit_path.name != "supervision_audit.json"
            or audit_path.parent.name != task_key
        ):
            raise ValueError(
                "explicit supervision audit does not match the repository task identity: "
                f"{audit_path}"
            )
    if not audit_path.is_file():
        return None
    saved = json.loads(audit_path.read_text(encoding="utf-8"))
    if saved.get("status") != "valid":
        raise ValueError(f"saved supervision audit is not valid: {audit_path}")

    task_assets_root = audit_path.parent.resolve()
    training_assets_root = (task_assets_root / "vqa" / stage).resolve()
    selected_manifest = _require_source_path(
        saved.get("selected_manifest"),
        allowed_root=training_assets_root,
        description="selected manifest",
    )
    selected = json.loads(selected_manifest.read_text(encoding="utf-8"))
    if not isinstance(selected, list) or not all(isinstance(value, str) for value in selected):
        raise ValueError(f"invalid selected manifest: {selected_manifest}")

    canonical_supervision_root = (Path(adapter.task_root) / "supervision").resolve()
    reused_sources = [
        _require_source_path(
            value,
            allowed_root=canonical_supervision_root,
            description="reused label source",
        )
        for value in saved.get("reused_sources", [])
    ]
    new_sources = [
        _require_source_path(
            value,
            allowed_root=training_assets_root,
            description="run-local label source",
        )
        for value in saved.get("new_sources", [])
    ]
    if not reused_sources and not new_sources:
        raise ValueError(f"saved supervision audit has no label sources: {audit_path}")

    with tempfile.TemporaryDirectory(prefix="pcp-audited-supervision-") as temporary:
        selected_paths, labels_map, _, supervision_hash, strict_audit = (
            bank.resolve_training_supervision(
                selected=selected,
                reused_sources=reused_sources,
                new_sources=new_sources,
                vqa_to_key=adapter.vqa_to_key,
                attrs=list(attributes),
                audit_path=Path(temporary) / "strict-audit.json",
                selected_manifest=selected_manifest,
            )
        )
    saved_contract = {
        "status": "valid",
        "selected_count_target": len(selected_paths),
        "selected_count_raw": len(selected_paths),
        "matched_count": len(selected_paths),
        "missing_count": 0,
        "duplicate_selected_count": 0,
        "attribute_counts": strict_audit.get("attribute_counts"),
        "joint_counts": strict_audit.get("joint_counts"),
        "supervision_hash": supervision_hash,
    }
    mismatches = {
        key: {"observed": saved.get(key), "recomputed": value}
        for key, value in saved_contract.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"saved supervision audit does not reproduce exactly: {audit_path}: {mismatches}"
        )
    legacy_supervision_hash = bank.legacy_supervision_hash(
        selected_paths, labels_map, list(attributes)
    )
    return (
        selected_paths,
        labels_map,
        supervision_hash,
        legacy_supervision_hash,
        strict_audit,
        selected_manifest,
        audit_path,
    )


def _cache_supervision_identity(
    experiment_root: Path,
    bank: Any,
    task_config: Mapping[str, Any],
    attributes: tuple[str, ...],
    stage: str,
    gallery_image_ids: tuple[str, ...],
    selected: Sequence[str],
    labels_map: Mapping[str, Mapping[str, int]],
    strict_supervision_hash: str,
    legacy_supervision_hash: str,
    suite_path: Path | None = None,
) -> dict[str, Any]:
    """Audit every learner cache against the same recovered label records.

    ProbeBank caches straddle a historical hash-schema migration.  Older
    entries store ``legacy_supervision_hash`` while newer entries store the
    schema-tagged strict hash.  Both are acceptable only when recomputed from
    the exact same strictly resolved paths and binary labels.  This function
    additionally checks the training path hash and per-attribute label counts,
    so accepting the legacy digest cannot hide a different supervision set.
    """
    suite_path = (
        experiment_root / DEFAULT_SUITE
        if suite_path is None
        else Path(suite_path).resolve()
    )
    if (
        not suite_path.is_relative_to(experiment_root.resolve())
        or not suite_path.is_file()
    ):
        raise ValueError(f"ProbeBank suite escapes the repository or is missing: {suite_path}")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    methods = tuple(str(value) for value in suite.get("learned_methods", ()))
    roots = suite.get("probe_source_roots") or {}
    if not methods or "__default__" not in roots:
        raise ValueError(f"invalid ProbeBank suite contract: {suite_path}")

    order = int(task_config["order"])
    dataset = str(task_config["dataset"])
    task = str(task_config["task"])
    task_key = f"{order:03d}_{dataset}_{task}"
    expected_records_hash = bank.stable_hash(list(gallery_image_ids))
    accepted_hashes = {
        strict_supervision_hash: "complete_selected_labels_v1",
        legacy_supervision_hash: "legacy_pre_schema",
    }
    expected_training_records_hash = bank.stable_hash(sorted(str(value) for value in selected))
    variants: dict[str, dict[str, Any]] = {}
    metadata_count = 0
    for method in methods:
        relative_root = roots.get(method, roots["__default__"])
        bank_root = (
            experiment_root / str(relative_root) / "task_isolated" / task_key
        ).resolve()
        for attribute in attributes:
            metadata_path = bank.cache_dir(
                bank_root, dataset, task, stage, method, attribute
            ) / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"missing ProbeBank metadata for {method}/{attribute}: {metadata_path}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            required = {
                "dataset": dataset,
                "task": task,
                "supervision_stage": stage,
                "method": method,
                "canonical_attribute": attribute,
                "records_hash": expected_records_hash,
                "score_length": len(gallery_image_ids),
            }
            mismatches = {
                key: {"observed": metadata.get(key), "expected": value}
                for key, value in required.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"ProbeBank supervision metadata mismatch at {metadata_path}: {mismatches}"
                )
            supervision_hash = metadata.get("supervision_hash")
            if not isinstance(supervision_hash, str) or not supervision_hash:
                raise ValueError(f"missing supervision_hash in {metadata_path}")
            if supervision_hash not in accepted_hashes:
                raise _UnmatchedCacheSupervisionHash(
                    "ProbeBank cache supervision_hash is neither the strict nor "
                    f"legacy digest of the recovered labels: {metadata_path}"
                )
            training_records_hash = metadata.get("training_records_hash")
            if training_records_hash != expected_training_records_hash:
                raise ValueError(
                    f"ProbeBank training record identity mismatch at {metadata_path}"
                )
            expected_positive = sum(
                int(labels_map[image_id][attribute]) for image_id in selected
            )
            expected_counts = {
                "training_count": len(selected),
                "positive_count": expected_positive,
                "negative_count": len(selected) - expected_positive,
            }
            count_mismatches = {
                key: {"observed": metadata.get(key), "expected": value}
                for key, value in expected_counts.items()
                if metadata.get(key) != value
            }
            if count_mismatches:
                raise ValueError(
                    f"ProbeBank recovered-label count mismatch at {metadata_path}: "
                    f"{count_mismatches}"
                )
            variant = variants.setdefault(
                supervision_hash,
                {
                    "schema": accepted_hashes[supervision_hash],
                    "count": 0,
                    "methods": set(),
                },
            )
            variant["count"] += 1
            variant["methods"].add(method)
            metadata_count += 1
    return {
        "metadataCount": metadata_count,
        "trainingRecordsHash": expected_training_records_hash,
        "variants": {
            digest: {
                "schema": value["schema"],
                "count": int(value["count"]),
                "methods": sorted(value["methods"]),
            }
            for digest, value in sorted(variants.items())
        },
    }


def load_original_development_supervision(
    experiment_root: Path,
    task_config: Mapping[str, Any],
    adapter: Any,
    gallery_image_ids: Sequence[str],
    development_mask: bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    target_attribute: str | None,
    *,
    stage: str = DEFAULT_STAGE,
    expected_supervision_hash: str | None = None,
    source_audit_path: Path | None = None,
    suite_path: Path | None = None,
) -> OriginalSupervision:
    """Recover original binary labels and retain only Development rows.

    ``target_attribute=None`` (or ``"joint"``) produces the logical AND of all
    modeled attribute labels.  The returned rows preserve the original audited
    supervision order; Validation, Test and Query rows are excluded solely by
    the supplied Development mask and are reported in the audit.
    """

    experiment_root = Path(experiment_root).resolve()
    bank, harness = _load_probebank_modules(experiment_root)
    order, dataset, task = _task_identity(task_config, adapter)
    attributes = _modeled_attributes(adapter)
    canonical_target = _resolve_target_attribute(
        adapter, attributes, target_attribute
    )

    image_ids = tuple(str(value) for value in gallery_image_ids)
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("gallery_image_ids must be non-empty and unique")
    source_paths = tuple(str(value) for value in bank.database_paths(adapter))
    if source_paths != image_ids:
        raise ValueError("Web gallery IDs do not exactly match ProbeBank embedding order")

    if isinstance(development_mask, (bytes, bytearray, memoryview)):
        mask = np.frombuffer(development_mask, dtype=np.uint8)
    else:
        mask = np.asarray(development_mask, dtype=np.uint8).reshape(-1)
    if mask.shape != (len(image_ids),) or not np.isin(mask, (0, 1)).all():
        raise ValueError("development_mask must be a row-aligned binary vector")

    explicit_source_audit = (
        Path(source_audit_path).resolve() if source_audit_path is not None else None
    )
    audited = _audited_source_supervision(
        experiment_root,
        bank,
        task_config,
        adapter,
        attributes,
        str(stage),
        source_audit_path,
    )
    if explicit_source_audit is not None and audited is None:
        raise FileNotFoundError(
            f"explicit supervision audit is unavailable: {explicit_source_audit}"
        )
    using_audited_source = audited is not None
    (
        selected,
        labels_map,
        recovered_hash,
        legacy_recovered_hash,
        strict_audit,
        selected_manifest,
        source_audit_path,
    ) = (
        audited
        if audited is not None
        else _source_supervision(
            bank, harness, task_config, adapter, attributes, str(stage)
        )
    )
    if expected_supervision_hash is None:
        try:
            cache_identity = _cache_supervision_identity(
                experiment_root,
                bank,
                task_config,
                attributes,
                str(stage),
                image_ids,
                selected,
                labels_map,
                recovered_hash,
                legacy_recovered_hash,
                suite_path,
            )
        except _UnmatchedCacheSupervisionHash as primary_error:
            if explicit_source_audit is not None:
                raise primary_error
            fallback = (
                _source_supervision(
                    bank, harness, task_config, adapter, attributes, str(stage)
                )
                if using_audited_source
                else _audited_source_supervision(
                    experiment_root,
                    bank,
                    task_config,
                    adapter,
                    attributes,
                    str(stage),
                    None,
                )
            )
            if fallback is None:
                raise primary_error
            (
                selected,
                labels_map,
                recovered_hash,
                legacy_recovered_hash,
                strict_audit,
                selected_manifest,
                source_audit_path,
            ) = fallback
            cache_identity = _cache_supervision_identity(
                experiment_root,
                bank,
                task_config,
                attributes,
                str(stage),
                image_ids,
                selected,
                labels_map,
                recovered_hash,
                legacy_recovered_hash,
                suite_path,
            )
        expected_hash = recovered_hash
        cache_metadata_count = int(cache_identity["metadataCount"])
        hash_source = "ProbeBank cache metadata"
    else:
        expected_hash = str(expected_supervision_hash)
        cache_metadata_count = 0
        cache_identity = None
        hash_source = "caller"
    if not expected_hash or recovered_hash != expected_hash:
        raise ValueError(
            "recovered original supervision_hash does not match the expected ProbeBank hash"
        )

    row_by_id = {image_id: index for index, image_id in enumerate(image_ids)}
    missing = [image_id for image_id in selected if image_id not in row_by_id]
    if missing:
        raise ValueError(
            f"original supervision contains {len(missing)} rows outside the Web gallery"
        )

    selected_indices: list[int] = []
    selected_labels: list[int] = []
    retained_indices: list[int] = []
    retained_labels: list[int] = []
    for image_id in selected:
        row_index = row_by_id[image_id]
        row = labels_map[image_id]
        if canonical_target is None:
            label = int(all(int(row[attribute]) == 1 for attribute in attributes))
        else:
            label = int(row[canonical_target])
        if label not in (0, 1):
            raise ValueError(f"non-binary recovered label for {image_id}")
        selected_indices.append(row_index)
        selected_labels.append(label)
        if mask[row_index] != 1:
            continue
        retained_indices.append(row_index)
        retained_labels.append(label)

    all_indices = np.asarray(selected_indices, dtype=np.int64)
    all_labels = np.asarray(selected_labels, dtype=np.uint8)
    indices = np.asarray(retained_indices, dtype=np.int64)
    labels = np.asarray(retained_labels, dtype=np.uint8)
    all_indices.setflags(write=False)
    all_labels.setflags(write=False)
    indices.setflags(write=False)
    labels.setflags(write=False)
    audit = {
        "schemaVersion": 1,
        "taskOrder": order,
        "taskId": f"{order:03d}_{dataset}_{task}",
        "dataset": dataset,
        "task": task,
        "stage": str(stage),
        "target": "joint" if canonical_target is None else canonical_target,
        "targetRule": "all modeled attributes equal 1" if canonical_target is None else "binary attribute label",
        "attributes": list(attributes),
        "selectedManifest": str(selected_manifest.resolve()),
        "sourceSupervisionAudit": (
            str(source_audit_path.resolve()) if source_audit_path is not None else None
        ),
        "expectedSupervisionHash": expected_hash,
        "recoveredSupervisionHash": recovered_hash,
        "legacyRecoveredSupervisionHash": legacy_recovered_hash,
        "hashSource": hash_source,
        "cacheMetadataCount": cache_metadata_count,
        "cacheSupervisionIdentity": cache_identity,
        "originalSupervisionCount": len(selected),
        "originalPositiveCount": int(all_labels.sum()),
        "originalNegativeCount": int(len(all_labels) - all_labels.sum()),
        "developmentCount": int(len(indices)),
        "excludedNonDevelopmentCount": int(len(selected) - len(indices)),
        "positiveCount": int(labels.sum()),
        "negativeCount": int(len(labels) - labels.sum()),
        "strictValidation": {
            "status": strict_audit.get("status"),
            "matchedCount": strict_audit.get("matched_count"),
            "missingCount": strict_audit.get("missing_count"),
            "duplicateSelectedCount": strict_audit.get("duplicate_selected_count"),
            "attributeCounts": strict_audit.get("attribute_counts"),
            "jointCounts": strict_audit.get("joint_counts"),
        },
    }
    return OriginalSupervision(
        indices=indices,
        labels=labels,
        selected_indices=all_indices,
        selected_labels=all_labels,
        audit=audit,
    )


__all__ = [
    "OriginalSupervision",
    "PROBE_VALIDATION_FRACTION",
    "PROBE_VALIDATION_PROTOCOL",
    "PROBE_VALIDATION_SEED",
    "ProbeValidationSplit",
    "build_probe_validation_split",
    "load_original_development_supervision",
]
