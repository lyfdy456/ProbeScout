"""Local multi-user tuning API for the PCP workbench.

The service deliberately binds to loopback and is exposed to browsers through
Vite's same-origin ``/api/tuning`` proxy.  It keeps all mutable state in the
analysis-level ``runtime/tuning`` directory, outside Vite's web root; files
below ``public`` are opened read-only.

The HTTP layer and persistence layer use only Python's standard library.  Model
jobs run serially and use the already provisioned ``probe_learning`` environment
for NumPy and scikit-learn.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import json
import math
import mimetypes
import os
import queue
import re
import secrets
import sqlite3
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse


# The sidecar is normally launched as a script, while backend tests import it
# through an explicit module spec.  Keep sibling helper imports deterministic
# in both entry modes.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))


API_PREFIX = "/api/tuning"
COOKIE_NAME = "pcp_tuning_identity"
MAX_JSON_BYTES = 1_048_576
LABELS = {-2, -1, 0, 1, 2}
MAX_BULK_ANNOTATIONS = 1000
LEGACY_RUN_MODES = frozenset({"prototype", "label"})
HISTORICAL_LOW_DIM_RUN_MODES = frozenset({"fusion-weight", "residual"})
REFINEMENT_RUN_MODES = frozenset(
    {"weight_staged", "weight_joint", "probe_staged", "probe_joint"}
)
WEIGHT_REFINEMENT_RUN_MODES = frozenset({"weight_staged", "weight_joint"})
PROBE_REFINEMENT_RUN_MODES = frozenset({"probe_staged", "probe_joint"})
# Historical modes remain readable. New UI actions separate native Probe
# snapshots from the two weight schedules; combined Probe modes cannot launch.
# Keep this historical name stable for regression helpers and old clients.
CREATABLE_RUN_MODES = HISTORICAL_LOW_DIM_RUN_MODES
REQUESTABLE_RUN_MODES = CREATABLE_RUN_MODES | REFINEMENT_RUN_MODES
RUN_MODES = LEGACY_RUN_MODES | REQUESTABLE_RUN_MODES
FUSION_WEIGHT_ALGORITHM_V1 = "fusion-weight-tune-v1"
FUSION_WEIGHT_ALGORITHM_V2 = "fusion-weight-per-attribute-tune-v2"
FUSION_WEIGHT_ALGORITHM_V3 = "fusion-weight-per-attribute-joint-product-tune-v3"
FUSION_WEIGHT_ALGORITHM_V4 = "fusion-weight-hierarchical-rank-13-v4"
SUPPORTED_FUSION_WEIGHT_ALGORITHMS = frozenset(
    {
        FUSION_WEIGHT_ALGORITHM_V1,
        FUSION_WEIGHT_ALGORITHM_V2,
        FUSION_WEIGHT_ALGORITHM_V3,
        FUSION_WEIGHT_ALGORITHM_V4,
    }
)
RUN_ALGORITHM_VERSIONS = {
    "prototype": "prototype-tune-v1",
    "label": "label-linear-head-v1",
    "fusion-weight": FUSION_WEIGHT_ALGORITHM_V4,
    "residual": "residual-tune-v1",
    "weight_staged": "conjunction-holistic-weight-staged-v5",
    "weight_joint": "conjunction-holistic-weight-joint-v5",
    "probe_staged": "conjunction-holistic-probe-staged-v3",
    "probe_joint": "conjunction-holistic-probe-joint-v3",
}
PRE_FIXED_GATE_REFINEMENT_ALGORITHMS = {
    "weight_staged": "conjunction-holistic-weight-staged-v4",
    "weight_joint": "conjunction-holistic-weight-joint-v4",
    "probe_staged": "conjunction-holistic-probe-staged-v2",
    "probe_joint": "conjunction-holistic-probe-joint-v2",
}
FIXED_GATE_POLICY = "fixed-base-theta-temperature-v1"
LEGACY_WEIGHT_REFINEMENT_ALGORITHMS = {
    "weight_staged": "conjunction-holistic-weight-staged-v2",
    "weight_joint": "conjunction-holistic-weight-joint-v2",
}
PRE_ISOLATION_WEIGHT_REFINEMENT_ALGORITHMS = {
    "weight_staged": "conjunction-holistic-weight-staged-v3",
    "weight_joint": "conjunction-holistic-weight-joint-v3",
}
REFINEMENT_NORMALIZATION_POLICY = "development-minus-fixed-vqa-val-minmax-v1"
LEGACY_SUPERVISION_POLICY = "original-dg-plus-feedback-v1"
PROBE_VALIDATION_SUPERVISION_POLICY = "probe-train80-plus-feedback-val20-v2"
SUPERVISION_POLICY = "probe-train80-plus-feedback-clean-web-validation-v3"
VQA_VALIDATION_SUPERVISION_POLICY = "original-vqa-minus-shared-val-plus-feedback-v4"
VQA_VALIDATION_PROTOCOL = "fixed-vqa-joint-seed0-holdout-v1"
SUPPORTED_SUPERVISION_POLICIES = frozenset(
    {
        LEGACY_SUPERVISION_POLICY,
        PROBE_VALIDATION_SUPERVISION_POLICY,
        SUPERVISION_POLICY,
        VQA_VALIDATION_SUPERVISION_POLICY,
    }
)
PROBE_VALIDATION_PROTOCOL = "sklearn-stratified-train-test-split-v1"
PROBE_VALIDATION_SEED = 0
PROBE_VALIDATION_FRACTION = 0.2
CLEAN_VALIDATION_PROTOCOL = (
    "web-validation-minus-all-original-supervision-and-feedback-v1"
)
RUN_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}
SOURCE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,48}$")
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
WEIGHTED_FUSION_LABEL = "Weighted Fusion"
HIERARCHICAL_FUSION_LABEL = "Hierarchical Fusion"
WEIGHTED_FUSION_LEARNERS = (
    "MLP",
    "K-Fold",
    "Triplet Loss",
    "Attention Pooling",
    "Attribute-conditioned Attention",
    "nnPU",
    "DC-PU",
    "Ours-PURA",
)
EMBEDDING_BASELINE_METHODS = (
    "Image Prototype",
    "Query MaxSim",
    "Image--Text Fusion",
    "Text Prompt Ensemble",
    "Z-score Image--Text Fusion",
)
# Only the current score-level refinement uses this smaller holistic branch.
# The five static baselines and historical 13-method rank fusion stay intact.
REFINEMENT_EMBEDDING_METHODS = ("Query MaxSim", "Text Prompt Ensemble")
RANK_FUSION_METHODS = (*EMBEDDING_BASELINE_METHODS, *WEIGHTED_FUSION_LEARNERS)
GLOBAL_FUSION_METHODS = ("Ours-Full", *EMBEDDING_BASELINE_METHODS)
REFINEMENT_LAMBDA_0 = 0.25
REFINEMENT_GAMMA_MIN = 0.05
REFINEMENT_GAMMA_MAX = 3.0
OURS_ONLY_GLOBAL_WEIGHTS = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
MAX_FUSION_WEIGHT = 100.0
HIERARCHICAL_FUSION_ALGORITHM = "hierarchical-ours-full-global-rank-v2"
DYNAMIC_PCP_CLUSTER_ALGORITHM = "dynamic-hierarchical-pcp-clusters-v2"
DYNAMIC_PCP_CLUSTER_FEATURE_BASIS = "full-hierarchy-global-rank-v2"
RANK_FUSION_ALGORITHM = "hierarchical-rank-fusion-13-v1"
RANK_FUSION_CLUSTER_ALGORITHM = "dynamic-attribute-rank-fusion-clusters-v1"
RANK_FUSION_CLUSTER_FEATURE_BASIS = "attribute-hierarchy-13-rank-v1"
REFINEMENT_VISUALIZATION_ALGORITHM = "conjunction-holistic-visualization-v2"
REFINEMENT_VISUALIZATION_LAYOUT = "score,rank"
REFINEMENT_CLUSTER_ALGORITHM = "dynamic-refinement-pcp-clusters-v2"
REFINEMENT_CLUSTER_FEATURE_BASIS = "exact-refinement-visible-rank-profile-v2"
DYNAMIC_PCP_CLUSTER_SCHEMES = {
    "fine30": ("fine", 30),
    "fine50": ("fine", 50),
    "fine100": ("fine", 100),
    "absolute": ("absolute", 4),
    "shape": ("shape", 3),
}
DYNAMIC_PCP_CLUSTER_CACHE_SIZE = 40


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.message = message


class ClosingConnection(sqlite3.Connection):
    """Make ``with service.connect()`` close handles as well as commit them."""

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_value: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback_value))
        finally:
            self.close()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def resolve_runtime_root(web_root: Path, explicit_root: Path | None = None) -> Path:
    """Resolve mutable state outside Vite's web root and migrate the legacy path.

    An explicit CLI path takes precedence over ``PCP_TUNING_ROOT``.  Neither
    override is migrated automatically because callers may intentionally own
    that location.  The one-time default migration stays on the same analysis
    volume, preserves the database/artifacts, and rotates the browser-cookie
    signing key that may previously have been reachable through the dev server.
    """

    resolved_web_root = web_root.resolve()
    configured_root = explicit_root
    if configured_root is None:
        environment_root = os.environ.get("PCP_TUNING_ROOT", "").strip()
        configured_root = Path(environment_root) if environment_root else None
    if configured_root is not None:
        return configured_root.resolve()

    legacy_root = (resolved_web_root / "runtime" / "tuning").resolve()
    runtime_root = (resolved_web_root.parent / "runtime" / "tuning").resolve()
    if legacy_root.exists():
        if runtime_root.exists():
            raise RuntimeError(
                "Both legacy and protected tuning runtime directories exist; "
                "refusing to merge mutable user data automatically"
            )
        runtime_root.parent.mkdir(parents=True, exist_ok=True)
        legacy_root.replace(runtime_root)
        # The legacy key lived below the Vite project root.  Preserve all user
        # data, but retire the key so prior browser identities are invalidated.
        # Keep one offline copy outside the web root for deliberate local
        # recovery; the service never reads keys from this directory.
        legacy_secret = runtime_root / ".cookie-secret"
        if legacy_secret.is_file():
            retired_root = runtime_root / "retired"
            retired_root.mkdir(parents=True, exist_ok=True)
            legacy_secret.replace(retired_root / "legacy-cookie-secret.bin")
    return runtime_root


def new_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json_dumps(value) + "\n").encode("utf-8"))


def atomic_savez(path: Path, **arrays: Any) -> None:
    """Publish a compressed NumPy model artifact only after a complete write."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json_dumps(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def normalize_rows(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, np.float32(1e-12))


def supports_relation_mismatch(task_id: str, target_id: str) -> bool:
    """Only the explicit robustness pilot supports confirmed query-only negatives."""
    from experimental_parent_overlay import applies

    return applies(task_id) and target_id == "joint"


def normalized_ranks(scores: Any) -> Any:
    """Return average-tie ascending percentiles (1 means highest score)."""
    import numpy as np

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contain non-finite values")
    count = int(values.size)
    if count == 0:
        return np.empty(0, dtype=np.float32)
    if count == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    result = np.empty(count, dtype=np.float32)
    start = 0
    denominator = float(count - 1)
    while start < count:
        stop = start + 1
        while stop < count and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_position = (start + stop - 1) / 2.0
        result[order[start:stop]] = np.float32(average_position / denominator)
        start = stop
    return result


def normalize_fusion_weights(values: dict[str, Any]) -> tuple[float, ...]:
    """Validate the eight learner weights and return a scale-invariant tuple."""
    if set(values) != set(WEIGHTED_FUSION_LEARNERS):
        raise ValueError("weights must contain exactly the eight audited learners")
    weights: list[float] = []
    for learner in WEIGHTED_FUSION_LEARNERS:
        value = values[learner]
        if isinstance(value, bool):
            raise ValueError(f"weight for {learner} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"weight for {learner} must be numeric") from error
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > MAX_FUSION_WEIGHT:
            raise ValueError(
                f"weight for {learner} must be finite and between 0 and {MAX_FUSION_WEIGHT:g}"
            )
        weights.append(numeric)
    total = float(math.fsum(weights))
    if total <= 0.0:
        raise ValueError("at least one learner weight must be greater than zero")
    return tuple(value / total for value in weights)


def normalize_global_fusion_weights(
    values: dict[str, Any] | None,
) -> tuple[float, ...]:
    """Validate and normalize the six fixed global rank-fusion sources."""

    if values is None:
        return OURS_ONLY_GLOBAL_WEIGHTS
    if set(values) != set(GLOBAL_FUSION_METHODS):
        raise ValueError(
            "globalWeights must contain exactly Ours-Full and the five embedding baselines"
        )
    weights: list[float] = []
    for method in GLOBAL_FUSION_METHODS:
        value = values[method]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"global weight for {method} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > MAX_FUSION_WEIGHT:
            raise ValueError(
                f"global weight for {method} must be finite and between "
                f"0 and {MAX_FUSION_WEIGHT:g}"
            )
        weights.append(numeric)
    total = float(math.fsum(weights))
    if total <= 0.0:
        raise ValueError("at least one global weight must be greater than zero")
    return tuple(value / total for value in weights)


def _normalize_exact_weight_object(
    values: dict[str, Any],
    ordered_ids: tuple[str, ...],
    label: str,
) -> tuple[float, ...]:
    if set(values) != set(ordered_ids):
        raise ValueError(f"{label} must contain exactly the required entries")
    weights: list[float] = []
    for item_id in ordered_ids:
        value = values[item_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"weight for {item_id} in {label} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > MAX_FUSION_WEIGHT:
            raise ValueError(
                f"weight for {item_id} in {label} must be finite and between "
                f"0 and {MAX_FUSION_WEIGHT:g}"
            )
        weights.append(numeric)
    total = float(math.fsum(weights))
    if total <= 0.0:
        raise ValueError(f"at least one weight in {label} must be greater than zero")
    return tuple(value / total for value in weights)


def normalize_rank_fusion_weights(
    attribute_ids: Iterable[str],
    attribute_weights: dict[str, Any],
    method_weights_by_attribute: dict[str, Any],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Normalize the outer attribute simplex and every fixed 13-method row."""

    attributes = tuple(str(value) for value in attribute_ids)
    if not attributes or len(set(attributes)) != len(attributes):
        raise ValueError("rank fusion requires unique task attributes")
    normalized_attributes = _normalize_exact_weight_object(
        attribute_weights,
        attributes,
        "attributeWeights",
    )
    if set(method_weights_by_attribute) != set(attributes):
        raise ValueError(
            "methodWeightsByAttribute must contain exactly the task attributes"
        )
    normalized_methods: list[tuple[float, ...]] = []
    for attribute_id in attributes:
        values = method_weights_by_attribute[attribute_id]
        if not isinstance(values, dict):
            raise ValueError(f"method weights for {attribute_id} must be an object")
        normalized_methods.append(
            _normalize_exact_weight_object(
                values,
                RANK_FUSION_METHODS,
                f"methodWeightsByAttribute.{attribute_id}",
            )
        )
    return normalized_attributes, tuple(normalized_methods)


def normalize_hierarchical_fusion_weights(
    attribute_ids: Iterable[str],
    attribute_weights: dict[str, Any],
    learner_weights_by_attribute: dict[str, Any],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Validate per-attribute and per-learner weights in manifest order.

    Attribute weights are normalized to mean one so a common scale factor does
    not change the Joint score. Learner weights remain independently normalized
    within each attribute through ``normalize_fusion_weights``.
    """

    ordered_attributes = tuple(str(value) for value in attribute_ids)
    expected = set(ordered_attributes)
    if not ordered_attributes or set(attribute_weights) != expected:
        raise ValueError("attributeWeights must contain exactly the task attributes")
    if set(learner_weights_by_attribute) != expected:
        raise ValueError(
            "learnerWeightsByAttribute must contain exactly the task attributes"
        )

    raw_attribute_weights: list[float] = []
    normalized_learner_weights: list[tuple[float, ...]] = []
    for attribute_id in ordered_attributes:
        value = attribute_weights[attribute_id]
        if isinstance(value, bool):
            raise ValueError(f"attribute weight for {attribute_id} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"attribute weight for {attribute_id} must be numeric"
            ) from error
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > MAX_FUSION_WEIGHT:
            raise ValueError(
                f"attribute weight for {attribute_id} must be finite and between "
                f"0 and {MAX_FUSION_WEIGHT:g}"
            )
        learner_values = learner_weights_by_attribute[attribute_id]
        if not isinstance(learner_values, dict):
            raise ValueError(f"learner weights for {attribute_id} must be an object")
        raw_attribute_weights.append(numeric)
        normalized_learner_weights.append(normalize_fusion_weights(learner_values))

    total = float(math.fsum(raw_attribute_weights))
    if total <= 0.0:
        raise ValueError("at least one attribute weight must be greater than zero")
    mean_scale = len(raw_attribute_weights) / total
    normalized_attributes = tuple(value * mean_scale for value in raw_attribute_weights)
    return normalized_attributes, tuple(normalized_learner_weights)


def build_full_hierarchical_rank_features(
    final_ranks: Any,
    hierarchical_ranks: Any,
    exported_ranks: Any,
    *,
    target_ids: Iterable[str],
    method_labels: Iterable[str],
    attribute_ids: Iterable[str],
    overall_target_id: str,
) -> Any:
    """Build the expansion-independent global -> hierarchy PCP feature cube.

    Coordinates are the final global Joint, the inner Ours Joint, the five
    embedding-baseline Joint ranks, then each manifest-ordered attribute's
    inner hierarchy aggregate followed by its eight frozen learner ranks.
    UI disclosure state is intentionally absent from this contract.
    """

    import numpy as np

    targets = tuple(str(value) for value in target_ids)
    methods = tuple(str(value) for value in method_labels)
    attributes = tuple(str(value) for value in attribute_ids)
    if not attributes or len(set(attributes)) != len(attributes):
        raise ValueError("PCP clustering requires unique modeled attributes")
    if set(attributes) - set(targets) or overall_target_id not in targets:
        raise ValueError("PCP clustering targets do not match the task")
    if any(method not in methods for method in EMBEDDING_BASELINE_METHODS):
        raise ValueError("PCP clustering requires all five embedding baseline columns")
    if any(method not in methods for method in WEIGHTED_FUSION_LEARNERS):
        raise ValueError("PCP clustering requires all eight learner rank columns")

    final = np.asarray(final_ranks, dtype=np.float32)
    hierarchy = np.asarray(hierarchical_ranks, dtype=np.float32)
    exported = np.asarray(exported_ranks, dtype=np.float32)
    if final.ndim != 2 or hierarchy.ndim != 2 or exported.ndim != 3:
        raise ValueError("PCP clustering rank tensors have invalid dimensions")
    rows = hierarchy.shape[0]
    if final.shape != (rows, len(targets)) or hierarchy.shape != (
        rows,
        len(targets),
    ) or exported.shape != (
        rows,
        len(methods),
        len(targets),
    ):
        raise ValueError("PCP clustering rank tensors are not row aligned")

    overall_target_index = targets.index(overall_target_id)
    columns: list[Any] = [
        final[:, overall_target_index],
        hierarchy[:, overall_target_index],
    ]
    columns.extend(
        exported[:, methods.index(method), overall_target_index]
        for method in EMBEDDING_BASELINE_METHODS
    )
    for attribute_id in attributes:
        target_index = targets.index(attribute_id)
        columns.append(hierarchy[:, target_index])
        columns.extend(
            exported[:, methods.index(method), target_index]
            for method in WEIGHTED_FUSION_LEARNERS
        )
    features = np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("PCP clustering rank features contain non-finite values")
    if float(features.min()) < -1e-6 or float(features.max()) > 1.0 + 1e-6:
        raise ValueError("PCP clustering rank features must stay in [0,1]")
    return features


def build_attribute_rank_fusion_cluster_features(
    fused_ranks: Any,
    exported_ranks: Any,
    *,
    target_ids: Iterable[str],
    method_labels: Iterable[str],
    attribute_ids: Iterable[str],
    joint_target_id: str,
) -> Any:
    """Build the complete visible attribute -> 13-method rank hierarchy."""

    import numpy as np

    targets = tuple(str(value) for value in target_ids)
    methods = tuple(str(value) for value in method_labels)
    attributes = tuple(str(value) for value in attribute_ids)
    if not attributes or len(set(attributes)) != len(attributes):
        raise ValueError("Rank-fusion clustering requires unique task attributes")
    if set(attributes) - set(targets) or joint_target_id not in targets:
        raise ValueError("Rank-fusion clustering targets do not match the task")
    if any(method not in methods for method in RANK_FUSION_METHODS):
        raise ValueError("Rank-fusion clustering requires all thirteen method columns")

    fused = np.asarray(fused_ranks, dtype=np.float32)
    exported = np.asarray(exported_ranks, dtype=np.float32)
    rows = fused.shape[0] if fused.ndim == 2 else -1
    if fused.shape != (rows, len(targets)) or exported.shape != (
        rows,
        len(methods),
        len(targets),
    ):
        raise ValueError("Rank-fusion clustering tensors are not row aligned")

    joint_index = targets.index(joint_target_id)
    columns: list[Any] = [fused[:, joint_index]]
    for attribute_id in attributes:
        target_index = targets.index(attribute_id)
        columns.append(fused[:, target_index])
        columns.extend(
            exported[:, methods.index(method), joint_index]
            for method in EMBEDDING_BASELINE_METHODS
        )
        columns.extend(
            exported[:, methods.index(method), target_index]
            for method in WEIGHTED_FUSION_LEARNERS
        )
    features = np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Rank-fusion clustering features contain non-finite values")
    if float(features.min()) < -1e-6 or float(features.max()) > 1.0 + 1e-6:
        raise ValueError("Rank-fusion clustering features must stay in [0,1]")
    return features


def fit_dynamic_pcp_cluster_labels(
    rank_features: Any,
    development_mask: Any,
    scheme: str,
) -> tuple[Any, int, int]:
    """Fit one rank-profile clustering on Development and predict every row."""

    import numpy as np
    from sklearn.cluster import KMeans, MiniBatchKMeans

    spec = DYNAMIC_PCP_CLUSTER_SCHEMES.get(str(scheme))
    if spec is None:
        raise ValueError(
            "scheme must be one of " + ", ".join(DYNAMIC_PCP_CLUSTER_SCHEMES)
        )
    family, cluster_count = spec
    ranks = np.asarray(rank_features, dtype=np.float32)
    if ranks.ndim != 2 or ranks.shape[0] == 0 or ranks.shape[1] == 0:
        raise ValueError("PCP clustering requires a non-empty rank matrix")
    if not np.isfinite(ranks).all():
        raise ValueError("PCP clustering rank matrix contains non-finite values")
    if float(ranks.min()) < -1e-6 or float(ranks.max()) > 1.0 + 1e-6:
        raise ValueError("PCP clustering rank matrix must stay in [0,1]")

    if isinstance(development_mask, (bytes, bytearray, memoryview)):
        mask = np.frombuffer(development_mask, dtype=np.uint8)
    else:
        mask = np.asarray(development_mask, dtype=np.uint8).reshape(-1)
    if mask.shape != (len(ranks),) or not np.isin(mask, (0, 1)).all():
        raise ValueError("PCP clustering Development mask must be row-aligned and binary")
    fit_indices = np.flatnonzero(mask)
    if len(fit_indices) <= cluster_count:
        raise ValueError(
            f"Development needs more than K={cluster_count} rows, found {len(fit_indices)}"
        )

    if family in {"fine", "absolute"}:
        fit_ranks = ranks[fit_indices]
        mean = fit_ranks.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = fit_ranks.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        features = np.ascontiguousarray((ranks - mean) / scale, dtype=np.float32)
    else:
        features = np.ascontiguousarray(
            ranks - ranks.mean(axis=1, keepdims=True, dtype=np.float64),
            dtype=np.float32,
        )

    if family == "fine":
        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=42,
            batch_size=4096,
            n_init=10,
            max_iter=300,
            reassignment_ratio=0.01,
        )
    else:
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=30,
            max_iter=500,
            algorithm="lloyd",
        )
    model.fit(features[fit_indices])
    raw_labels = np.asarray(model.predict(features), dtype=np.int32)
    fit_raw_labels = raw_labels[fit_indices]
    fit_counts = np.bincount(fit_raw_labels, minlength=cluster_count)
    if np.any(fit_counts == 0):
        missing = np.flatnonzero(fit_counts == 0).tolist()
        raise RuntimeError(f"PCP clustering produced empty Development clusters: {missing}")

    # Cluster IDs are arbitrary.  Use Development raw-rank centroids only so
    # Validation/Test values cannot influence stable legend ordering.
    fit_ranks = ranks[fit_indices]
    centroids = np.vstack(
        [
            fit_ranks[fit_raw_labels == cluster_id].mean(axis=0)
            for cluster_id in range(cluster_count)
        ]
    )
    centroid_means = centroids.mean(axis=1)
    lexicographic_keys = (
        np.arange(cluster_count),
        *(centroids[:, index] for index in range(centroids.shape[1] - 1, -1, -1)),
        centroid_means,
    )
    order = np.lexsort(lexicographic_keys)
    raw_to_canonical = np.empty(cluster_count, dtype=np.uint8)
    raw_to_canonical[order] = np.arange(cluster_count, dtype=np.uint8)
    labels = np.ascontiguousarray(raw_to_canonical[raw_labels], dtype=np.uint8)
    if not np.array_equal(
        np.unique(labels[fit_indices]), np.arange(cluster_count, dtype=np.uint8)
    ):
        raise RuntimeError("Canonical PCP cluster labels are incomplete in Development")
    return labels, int(cluster_count), int(len(fit_indices))


def require_selected_seeds(
    seeds_by_method: dict[str, list[int]],
    methods: Iterable[str],
    selected_seeds: Iterable[int],
) -> None:
    """Require the deployed seeds while allowing audited caches to be supersets."""
    required = tuple(int(seed) for seed in selected_seeds)
    if not required or len(set(required)) != len(required):
        raise RuntimeError(f"Invalid selected seed contract: {list(required)}")
    for method in methods:
        available = tuple(int(seed) for seed in seeds_by_method.get(method, []))
        if len(set(available)) != len(available):
            raise RuntimeError(f"Duplicate cached seed for {method}: {list(available)}")
        missing = [seed for seed in required if seed not in available]
        if missing:
            raise RuntimeError(
                f"Score cache for {method} lacks selected seeds {missing}; "
                f"available={list(available)}"
            )


def average_precision(labels: Any, scores: Any) -> float:
    import numpy as np

    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != values.shape:
        raise ValueError("labels and scores must have the same shape")
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        raise ValueError("Evaluation split has no positive examples for this target")
    order = np.argsort(-values, kind="stable")
    ordered = truth[order].astype(np.float64, copy=False)
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(ordered) + 1, dtype=np.float64)
    return float(np.sum(precision * ordered) / positives)


def retrieval_metrics(labels: Any, scores: Any) -> dict[str, Any]:
    import numpy as np

    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-values, kind="stable")
    ordered = truth[order]
    ap = average_precision(truth, values)
    positives = int(np.count_nonzero(truth))
    cumulative = np.cumsum(ordered, dtype=np.float64)
    cutoffs = np.arange(1, truth.size + 1, dtype=np.float64)
    precision = cumulative / cutoffs
    recall = cumulative / positives
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_position = int(np.argmax(f1))
    metrics: dict[str, Any] = {
        "ap": ap,
        "evaluationCount": int(truth.size),
        "positiveCount": positives,
        "bestF1": float(f1[best_position]),
        "bestF1Cutoff": best_position + 1,
    }
    for cutoff in (30, 50, 100, 200):
        metrics[f"tpAt{cutoff}"] = int(np.count_nonzero(ordered[:cutoff]))
    return metrics


def prototype_tuned_scores(
    gallery_embeddings: Any,
    base_prototype: Any,
    labeled_indices: Any,
    labels: Any,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> tuple[Any, Any]:
    import numpy as np

    embeddings = normalize_rows(gallery_embeddings)
    prototype = normalize_rows(np.asarray(base_prototype, dtype=np.float32).reshape(1, -1))[0]
    indices = np.asarray(labeled_indices, dtype=np.int64).reshape(-1)
    annotations = np.asarray(labels, dtype=np.int8).reshape(-1)
    if indices.shape != annotations.shape:
        raise ValueError("labeled indices and labels must have the same shape")
    positive = annotations > 0
    negative = annotations < 0
    if not np.any(positive):
        raise ValueError("prototype tuning requires at least one positive annotation")

    positive_vectors = embeddings[indices[positive]]
    positive_weights = np.abs(annotations[positive]).astype(np.float64)
    positive_mean = np.average(positive_vectors, axis=0, weights=positive_weights)
    update = prototype.astype(np.float64) + float(alpha) * positive_mean
    if np.any(negative):
        negative_vectors = embeddings[indices[negative]]
        negative_weights = np.abs(annotations[negative]).astype(np.float64)
        negative_mean = np.average(negative_vectors, axis=0, weights=negative_weights)
        update -= float(beta) * negative_mean
    tuned = normalize_rows(update.reshape(1, -1))[0]
    scores = np.asarray(embeddings @ tuned, dtype=np.float32)
    return scores, tuned.astype(np.float32, copy=False)


def label_tuned_scores(
    gallery_embeddings: Any,
    labeled_indices: Any,
    labels: Any,
    *,
    regularization_c: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from scipy.optimize import minimize

    embeddings = normalize_rows(gallery_embeddings)
    indices = np.asarray(labeled_indices, dtype=np.int64).reshape(-1)
    annotations = np.asarray(labels, dtype=np.int8).reshape(-1)
    if indices.shape != annotations.shape:
        raise ValueError("labeled indices and labels must have the same shape")
    usable = annotations != 0
    binary = (annotations[usable] > 0).astype(np.uint8)
    if int(np.count_nonzero(binary)) < 2 or int(np.count_nonzero(binary == 0)) < 2:
        raise ValueError("label tuning requires at least two positive and two negative annotations")
    features = np.asarray(embeddings[indices[usable]], dtype=np.float64)
    targets = binary.astype(np.float64)
    weights = np.abs(annotations[usable]).astype(np.float64)
    positive_count = float(np.count_nonzero(targets))
    negative_count = float(targets.size - positive_count)
    weights *= np.where(
        targets > 0.5,
        float(targets.size) / (2.0 * positive_count),
        float(targets.size) / (2.0 * negative_count),
    )
    weight_sum = max(float(np.sum(weights)), 1e-12)
    regularization = 1.0 / max(float(regularization_c), 1e-8)

    def objective(parameters: Any) -> tuple[float, Any]:
        coefficient = parameters[:-1]
        intercept = float(parameters[-1])
        logits = features @ coefficient + intercept
        losses = np.logaddexp(0.0, logits) - targets * logits
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        weighted_error = weights * (probability - targets)
        value = float(np.sum(weights * losses) / weight_sum)
        value += 0.5 * regularization * float(np.dot(coefficient, coefficient))
        gradient = np.empty_like(parameters)
        gradient[:-1] = features.T @ weighted_error / weight_sum
        gradient[:-1] += regularization * coefficient
        gradient[-1] = float(np.sum(weighted_error) / weight_sum)
        return value, gradient

    result = minimize(
        objective,
        np.zeros(features.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"legacy label tuning failed: {result.message}")
    coefficient = np.asarray(result.x[:-1], dtype=np.float64)
    intercept = float(result.x[-1])
    logits = np.asarray(embeddings, dtype=np.float64) @ coefficient + intercept
    scores = (1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))).astype(np.float32)
    state = {
        "coef": coefficient.reshape(1, -1).astype(np.float32),
        "intercept": np.asarray([intercept], dtype=np.float32),
        "classes": np.asarray([0, 1], dtype=np.int8),
        "iterations": np.asarray([int(result.nit)], dtype=np.int32),
        "regularizationC": float(regularization_c),
    }
    return scores, state


@dataclass(frozen=True)
class TuningSourceSpec:
    manifest_path: Path
    suite_path: Path
    ours_full_root: Path
    supervision_audit_path: Path
    probe_stage: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset_id: str
    task_name: str
    data_root: Path
    row_count: int
    methods: tuple[str, ...]
    target_ids: tuple[str, ...]
    manifest: dict[str, Any]
    tuning_source: TuningSourceSpec | None = None


@dataclass(frozen=True)
class BundleData:
    spec: TaskSpec
    image_ids: tuple[str, ...]
    development_mask: bytes
    validation_mask: bytes
    test_mask: bytes
    query_indices: frozenset[int]


@dataclass(frozen=True)
class TuningSourceContext:
    adapter: Any
    task_config: dict[str, Any]
    attributes: tuple[str, ...]
    target_attributes: tuple[str | None, ...]
    probe_stage: str
    suite_path: Path | None
    supervision_audit_path: Path | None


@dataclass(frozen=True)
class WeightedFusionContext:
    pcp: Any
    adapter: Any
    task_config: dict[str, Any]
    tensors: dict[str, dict[str, Any]]
    seeds_by_method: dict[str, list[int]]
    learned_method_ids: tuple[str, ...]
    learned_method_labels: tuple[str, ...]
    attributes: tuple[str, ...]
    target_attributes: tuple[str | None, ...]
    selected_seeds: tuple[int, ...]
    gallery_indices: Any
    train_indices: Any
    ours_full_root: Path
    probe_stage: str = "iterative"


@dataclass(frozen=True)
class WeightedFusionResult:
    payload: bytes
    row_count: int
    target_count: int
    equal_weights: bool
    baseline_max_abs_error: float | None
    normalized_weights: tuple[float, ...]


@dataclass(frozen=True)
class RefinementBaseContext:
    """Immutable, task-level inputs shared by every refinement comparison.

    ``probe_features`` are the five-seed mean outputs of the eight frozen
    learner/probe families after a MinMax fit on Development minus fixed Val
    (the static Development mask is retained only for historical runs).
    The run-pinned holistic methods (two current, five legacy) occur only in
    ``embedding_features``; they are never copied into attribute rows.
    """

    attribute_ids: tuple[str, ...]
    attribute_names: tuple[str, ...]
    learner_names: tuple[str, ...]
    embedding_names: tuple[str, ...]
    probe_features: Any
    probe_min: Any
    probe_max: Any
    temperature: Any
    initial_theta: Any
    embedding_features: Any
    embedding_min: Any
    embedding_max: Any
    development_indices: Any
    base_state_fingerprint: str
    normalization_audit: dict[str, Any] | None = None
    raw_probe_probabilities: Any | None = None
    initial_baseline_identity: dict[str, Any] | None = None
    probe_method_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HierarchicalFusionResult:
    payload: bytes
    row_count: int
    target_count: int
    baseline: bool
    baseline_max_abs_error: float | None
    normalized_attribute_weights: tuple[float, ...]
    normalized_learner_weights: tuple[tuple[float, ...], ...]
    normalized_global_weights: tuple[float, ...]


@dataclass(frozen=True)
class RankFusionResult:
    payload: bytes
    row_count: int
    target_count: int
    normalized_attribute_weights: tuple[float, ...]
    normalized_method_weights_by_attribute: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class DynamicPcpClusterResult:
    payload: bytes
    row_count: int
    cluster_count: int
    scheme: str
    feature_count: int
    fit_row_count: int


@dataclass(frozen=True)
class RefinementVisualizationResult:
    """Exact, expansion-independent PCP snapshot for one immutable Tune run."""

    payload: bytes
    row_count: int
    component_count: int
    component_ranks: Any
    source_fingerprint: str
    base_state_fingerprint: str
    cluster_fit_mask: bytes | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    identity_key TEXT NOT NULL UNIQUE,
    identity_source TEXT NOT NULL CHECK (identity_source IN ('cookie', 'oai')),
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    base_method TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    row_index INTEGER NOT NULL CHECK (row_index >= 0),
    image_id TEXT NOT NULL,
    label INTEGER NOT NULL CHECK (label IN (-2, -1, 0, 1, 2)),
    failed_attributes_json TEXT NOT NULL DEFAULT '[]',
    suggested_failure_attribute_id TEXT,
    failure_attribution_confirmed INTEGER NOT NULL DEFAULT 0
        CHECK (failure_attribution_confirmed IN (0, 1)),
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, row_index)
);

CREATE TABLE IF NOT EXISTS annotation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    row_index INTEGER NOT NULL,
    image_id TEXT NOT NULL,
    previous_label INTEGER,
    label INTEGER,
    event TEXT NOT NULL CHECK (event IN ('upsert', 'delete')),
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mode TEXT NOT NULL CHECK (mode IN (
        'prototype', 'label', 'fusion-weight', 'residual',
        'weight_staged', 'weight_joint', 'probe_staged', 'probe_joint'
    )),
    base_method TEXT NOT NULL,
    evaluation_scope TEXT NOT NULL DEFAULT 'clean-validation'
        CHECK (evaluation_scope IN (
            'vqa-validation', 'clean-validation', 'probe-validation', 'validation', 'test'
        )),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    params_json TEXT NOT NULL,
    annotation_sha256 TEXT NOT NULL,
    snapshot_sha256 TEXT,
    annotation_count INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    artifact_relpath TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS retired_cookie_migrations (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    migrated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS probe_updates (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
    request_json TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    annotation_sha256 TEXT NOT NULL,
    annotation_count INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    audit_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE(session_id, request_sha256)
);
CREATE INDEX IF NOT EXISTS idx_probe_updates_session_created
ON probe_updates(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
ON sessions(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_annotations_session_updated
ON annotations(session_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_annotation_events_session_created
ON annotation_events(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_runs_session_created
ON model_runs(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_runs_queued
ON model_runs(created_at)
WHERE status = 'queued';
"""


class TuningService:
    def __init__(
        self,
        web_root: Path,
        runtime_root: Path | None = None,
        *,
        trust_oai_headers: bool | None = None,
        start_worker: bool = True,
        clean_validation_root: Path | None = None,
        vqa_validation_root: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        self.web_root = web_root.resolve()
        self.public_root = (self.web_root / "public").resolve()
        self.catalog_path = catalog_path or self.public_root / "data" / "catalog.json"
        self.runtime_root = resolve_runtime_root(self.web_root, runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.runtime_root / "workbench.sqlite3"
        # Evaluation identity is independent of browser/user tuning archives.
        self.clean_validation_root = (
            clean_validation_root
            or self.web_root.parent / "runtime" / "evaluation" / "clean-validation"
        ).resolve()
        self.vqa_validation_root = (
            vqa_validation_root
            or self.web_root.parent / "runtime" / "evaluation" / "vqa-validation"
        ).resolve()
        self.trust_oai_headers = (
            os.environ.get("PCP_TRUST_OAI_HEADERS") == "1"
            if trust_oai_headers is None
            else bool(trust_oai_headers)
        )
        self.tasks = self._load_catalog()
        self.cookie_secret = self._load_cookie_secret()
        self.retired_cookie_secret = self._load_retired_cookie_secret()
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()
        self._source_adapter_guard = threading.Lock()
        self._original_supervision_guard = threading.Lock()
        self._original_supervision_cache: dict[tuple[str, str], Any] = {}
        self._weighted_fusion_context_guard = threading.Lock()
        self._weighted_fusion_contexts: dict[str, WeightedFusionContext] = {}
        self._refinement_context_guard = threading.Lock()
        self._refinement_contexts: dict[tuple[str, tuple[str, ...], str], RefinementBaseContext] = {}
        self._weighted_fusion_result_guard = threading.Lock()
        self._weighted_fusion_results: dict[
            tuple[str, tuple[float, ...]], WeightedFusionResult
        ] = {}
        self._hierarchical_fusion_result_guard = threading.Lock()
        self._hierarchical_fusion_results: dict[
            tuple[
                str,
                tuple[float, ...],
                tuple[tuple[float, ...], ...],
                tuple[float, ...],
            ],
            HierarchicalFusionResult,
        ] = {}
        self._pcp_cluster_result_guard = threading.Lock()
        self._pcp_cluster_results: dict[
            tuple[
                str,
                tuple[float, ...],
                tuple[tuple[float, ...], ...],
                tuple[float, ...],
                str,
            ],
            DynamicPcpClusterResult,
        ] = {}
        self._pcp_cluster_inflight: dict[
            tuple[
                str,
                tuple[float, ...],
                tuple[tuple[float, ...], ...],
                tuple[float, ...],
                str,
            ],
            threading.Event,
        ] = {}
        self._rank_fusion_result_guard = threading.Lock()
        self._rank_fusion_results: dict[
            tuple[
                str,
                tuple[float, ...],
                tuple[tuple[float, ...], ...],
            ],
            RankFusionResult,
        ] = {}
        self._rank_fusion_cluster_result_guard = threading.Lock()
        self._rank_fusion_cluster_results: dict[
            tuple[
                str,
                tuple[float, ...],
                tuple[tuple[float, ...], ...],
                str,
            ],
            DynamicPcpClusterResult,
        ] = {}
        self._rank_fusion_cluster_inflight: dict[
            tuple[
                str,
                tuple[float, ...],
                tuple[tuple[float, ...], ...],
                str,
            ],
            threading.Event,
        ] = {}
        self._refinement_visualization_guard = threading.Lock()
        self._refinement_visualizations: dict[
            str, RefinementVisualizationResult
        ] = {}
        self._refinement_cluster_guard = threading.Lock()
        self._refinement_clusters: dict[
            tuple[str, str, str], DynamicPcpClusterResult
        ] = {}
        self._refinement_cluster_inflight: dict[
            tuple[str, str, str], threading.Event
        ] = {}
        self.run_queue: queue.Queue[str | None] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self._initialize_database()
        if start_worker:
            self.start_worker()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=5.0,
            isolation_level=None,
            factory=ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize_database(self) -> None:
        with self.connect() as connection:
            mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            if mode != "wal":
                raise RuntimeError(f"SQLite WAL mode is unavailable: {mode}")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(SCHEMA)
            annotation_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(annotations)")
            }
            if "failed_attributes_json" not in annotation_columns:
                connection.execute(
                    "ALTER TABLE annotations ADD COLUMN failed_attributes_json "
                    "TEXT NOT NULL DEFAULT '[]'"
                )
            if "suggested_failure_attribute_id" not in annotation_columns:
                connection.execute(
                    "ALTER TABLE annotations ADD COLUMN "
                    "suggested_failure_attribute_id TEXT"
                )
            if "failure_attribution_confirmed" not in annotation_columns:
                connection.execute(
                    "ALTER TABLE annotations ADD COLUMN "
                    "failure_attribution_confirmed INTEGER NOT NULL DEFAULT 0 "
                    "CHECK (failure_attribution_confirmed IN (0,1))"
                )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(model_runs)")
            }
            if "evaluation_scope" not in columns:
                # Existing runs were evaluated on Frozen Test.  Preserve that
                # provenance while giving every new run an explicit Validation
                # scope below.
                connection.execute(
                    "ALTER TABLE model_runs ADD COLUMN evaluation_scope TEXT NOT NULL "
                    "DEFAULT 'test' CHECK (evaluation_scope IN ('validation','test'))"
                )
            if "snapshot_sha256" not in columns:
                # Historical rows overloaded annotation_sha256 with either a
                # label-state hash or a whole-file hash.  Keep this nullable so
                # they remain readable; every newly created run stores both
                # meanings independently.
                connection.execute(
                    "ALTER TABLE model_runs ADD COLUMN snapshot_sha256 TEXT"
                )
            self._migrate_model_run_modes(connection)
            # Historical runs retain their recorded scope.  In particular, an
            # immutable queued snapshot must not be silently relabelled from
            # Test to Web Validation during a schema migration.
            connection.execute("PRAGMA optimize")
            connection.execute(
                "UPDATE model_runs SET status='queued', started_at=NULL "
                "WHERE status='running'"
            )
            connection.execute(
                "UPDATE probe_updates SET status='queued', started_at=NULL "
                "WHERE status='running'"
            )

    @staticmethod
    def _migrate_model_run_modes(connection: sqlite3.Connection) -> None:
        """Expand legacy mode/scope CHECKs without changing run provenance.

        SQLite cannot alter a CHECK constraint in place.  Rebuild the table only
        when its stored SQL predates the low-dimensional tuning modes or the
        Clean Validation scope; the guard makes startup migration idempotent.
        """
        schema_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_runs'"
        ).fetchone()
        schema_sql = str(schema_row[0] or "") if schema_row is not None else ""
        if all(mode in schema_sql for mode in REQUESTABLE_RUN_MODES) and (
            "vqa-validation" in schema_sql
        ):
            return

        columns = [
            str(row[1]) for row in connection.execute("PRAGMA table_info(model_runs)")
        ]
        expected = [
            "id",
            "session_id",
            "user_id",
            "mode",
            "base_method",
            "evaluation_scope",
            "status",
            "params_json",
            "annotation_sha256",
            "snapshot_sha256",
            "annotation_count",
            "positive_count",
            "negative_count",
            "artifact_relpath",
            "before_json",
            "after_json",
            "error",
            "created_at",
            "started_at",
            "finished_at",
        ]
        if set(columns) != set(expected) or len(columns) != len(expected):
            raise RuntimeError(
                "Cannot migrate model_runs mode contract with unexpected columns: "
                f"{columns}"
            )
        supported_mode_sql = ",".join(f"'{mode}'" for mode in sorted(RUN_MODES))
        invalid_modes = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT mode FROM model_runs "
                f"WHERE mode NOT IN ({supported_mode_sql})"
            )
        ]
        if invalid_modes:
            raise RuntimeError(f"Cannot migrate unknown model run modes: {invalid_modes}")
        invalid_scopes = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT evaluation_scope FROM model_runs "
                "WHERE evaluation_scope NOT IN ("
                "'vqa-validation','clean-validation','probe-validation','validation','test')"
            )
        ]
        if invalid_scopes:
            raise RuntimeError(
                f"Cannot migrate unknown model run evaluation scopes: {invalid_scopes}"
            )

        savepoint = "migrate_model_run_contract_v4"
        connection.execute(f"SAVEPOINT {savepoint}")
        try:
            connection.execute("DROP INDEX IF EXISTS idx_model_runs_session_created")
            connection.execute("DROP INDEX IF EXISTS idx_model_runs_queued")
            connection.execute(
                """
                CREATE TABLE model_runs_new (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    mode TEXT NOT NULL CHECK (
                        mode IN (
                            'prototype','label','fusion-weight','residual',
                            'weight_staged','weight_joint',
                            'probe_staged','probe_joint'
                        )
                    ),
                    base_method TEXT NOT NULL,
                    evaluation_scope TEXT NOT NULL DEFAULT 'clean-validation'
                        CHECK (
                            evaluation_scope IN (
                                'vqa-validation','clean-validation','probe-validation','validation','test'
                            )
                        ),
                    status TEXT NOT NULL CHECK (
                        status IN ('queued','running','succeeded','failed','cancelled')
                    ),
                    params_json TEXT NOT NULL,
                    annotation_sha256 TEXT NOT NULL,
                    snapshot_sha256 TEXT,
                    annotation_count INTEGER NOT NULL,
                    positive_count INTEGER NOT NULL,
                    negative_count INTEGER NOT NULL,
                    artifact_relpath TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            column_list = ",".join(expected)
            connection.execute(
                f"INSERT INTO model_runs_new({column_list}) "
                f"SELECT {column_list} FROM model_runs"
            )
            connection.execute("DROP TABLE model_runs")
            connection.execute("ALTER TABLE model_runs_new RENAME TO model_runs")
            connection.execute(
                "CREATE INDEX idx_model_runs_session_created "
                "ON model_runs(session_id, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX idx_model_runs_queued ON model_runs(created_at) "
                "WHERE status = 'queued'"
            )
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise

    def _load_cookie_secret(self) -> bytes:
        path = self.runtime_root / ".cookie-secret"
        if path.is_file():
            value = path.read_bytes()
            if len(value) >= 32:
                return value
        value = secrets.token_bytes(32)
        atomic_write_bytes(path, value)
        return value

    def _load_retired_cookie_secret(self) -> bytes | None:
        path = self.runtime_root / "retired" / "legacy-cookie-secret.bin"
        if not path.is_file():
            return None
        value = path.read_bytes()
        return value if len(value) >= 32 else None

    def _catalog_tuning_source(
        self,
        entry: Mapping[str, Any],
        task_id: str,
    ) -> TuningSourceSpec | None:
        raw = entry.get("tuningSource")
        if raw is None:
            if entry.get("catalogRole") == "addon":
                raise ValueError(f"Add-on task has no audited tuningSource: {task_id}")
            return None
        if not isinstance(raw, Mapping) or raw.get("schemaVersion") != 1:
            raise ValueError(f"Invalid tuningSource schema for {task_id}")
        expected_keys = {
            "schemaVersion",
            "manifest",
            "suite",
            "oursFullRoot",
            "supervisionAudit",
            "probeStage",
        }
        if set(raw) != expected_keys:
            raise ValueError(f"Incomplete tuningSource contract for {task_id}")

        experiment_root = self.web_root.parents[2].resolve()

        def source_path(
            key: str,
            allowed_root: Path,
            *,
            directory: bool = False,
        ) -> Path:
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid tuningSource.{key} for {task_id}")
            requested = Path(value)
            if (
                requested.is_absolute()
                or ".." in requested.parts
                or "\\" in value
                or requested.as_posix() != value
            ):
                raise ValueError(
                    f"tuningSource.{key} must be repository-relative for {task_id}"
                )
            resolved = (experiment_root / requested).resolve()
            if not resolved.is_relative_to(allowed_root.resolve()):
                raise ValueError(f"tuningSource.{key} escapes its allowed root for {task_id}")
            available = resolved.is_dir() if directory else resolved.is_file()
            if not available:
                kind = "directory" if directory else "file"
                raise FileNotFoundError(
                    f"tuningSource.{key} {kind} is unavailable for {task_id}: {resolved}"
                )
            return resolved

        config_root = experiment_root / "configs" / "experiments"
        outputs_root = experiment_root / "outputs"
        manifest_path = source_path("manifest", config_root)
        suite_path = source_path("suite", config_root)
        ours_full_root = source_path("oursFullRoot", outputs_root, directory=True)
        supervision_audit_path = source_path("supervisionAudit", outputs_root)
        expected_audit = (
            ours_full_root / "tasks" / task_id / "supervision_audit.json"
        ).resolve()
        if supervision_audit_path != expected_audit:
            raise ValueError(
                f"tuningSource supervision audit is not owned by {task_id}"
            )
        probe_stage = str(raw.get("probeStage", ""))
        if probe_stage not in {"iterative", "two_stage"}:
            raise ValueError(f"Unsupported tuningSource probe stage for {task_id}")
        return TuningSourceSpec(
            manifest_path=manifest_path,
            suite_path=suite_path,
            ours_full_root=ours_full_root,
            supervision_audit_path=supervision_audit_path,
            probe_stage=probe_stage,
        )

    def _load_catalog(self) -> dict[str, TaskSpec]:
        catalog_path = self.catalog_path
        if not catalog_path.is_file():
            raise FileNotFoundError(f"Task catalog is unavailable: {catalog_path}")
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        tasks: dict[str, TaskSpec] = {}
        for dataset in catalog.get("datasets", []):
            dataset_id = str(dataset.get("id", "")).strip()
            for entry in dataset.get("tasks", []):
                task_id = str(entry.get("id", "")).strip()
                if not SAFE_ID_PATTERN.fullmatch(task_id):
                    raise ValueError(f"Unsafe task id in catalog: {task_id!r}")
                relative_root = str(entry.get("dataRoot", "")).lstrip("/").replace("/", os.sep)
                data_root = (self.public_root / relative_root).resolve()
                if not data_root.is_relative_to(self.public_root):
                    raise ValueError(f"Task data root escapes public: {task_id}")
                manifest_path = data_root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if entry.get("catalogRole") == "experimental":
                    from experimental_parent_overlay import validate_registration
                    validate_registration(self, entry, manifest)
                methods = tuple(str(value) for value in manifest.get("methods", []))
                targets = tuple(
                    str(value.get("id")) for value in manifest.get("retrievalTargets", [])
                )
                if not methods or not targets:
                    raise ValueError(f"Incomplete task manifest: {task_id}")
                components = task_id.split("_", 2)
                if len(components) != 3 or components[1] != dataset_id:
                    raise ValueError(f"Task id does not encode dataset/task: {task_id}")
                tasks[task_id] = TaskSpec(
                    task_id=task_id,
                    dataset_id=dataset_id,
                    task_name=components[2],
                    data_root=data_root,
                    row_count=int(manifest["rowCount"]),
                    methods=methods,
                    target_ids=targets,
                    manifest=manifest,
                    tuning_source=self._catalog_tuning_source(entry, task_id),
                )
        if not tasks:
            raise ValueError("Task catalog contains no tasks")
        return tasks

    def task(self, task_id: str) -> TaskSpec:
        task = self.tasks.get(task_id)
        if task is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown task: {task_id}")
        return task

    @lru_cache(maxsize=4)
    def bundle(self, task_id: str) -> BundleData:
        spec = self.task(task_id)
        files = spec.manifest["files"]
        image_path = self._bundle_file(spec, files["imageIds"]["path"])
        development_path = self._bundle_file(spec, files["developmentMask"]["path"])
        validation_path = self._bundle_file(spec, files["validationMask"]["path"])
        test_path = self._bundle_file(spec, files["testMask"]["path"])
        image_ids = tuple(json.loads(image_path.read_text(encoding="utf-8")))
        development_mask = development_path.read_bytes()
        validation_mask = validation_path.read_bytes()
        test_mask = test_path.read_bytes()
        if (
            len(image_ids) != spec.row_count
            or len(development_mask) != spec.row_count
            or len(validation_mask) != spec.row_count
            or len(test_mask) != spec.row_count
        ):
            raise RuntimeError(f"Bundle row contract failed for {task_id}")
        query_indices: set[int] = set()
        query = spec.manifest.get("query")
        if isinstance(query, dict):
            for item in query.get("images", []):
                index = int(item["imageIndex"])
                if index < 0 or index >= spec.row_count:
                    raise RuntimeError(f"Fixed Query index is invalid for {task_id}: {index}")
                if str(item.get("imageId")) != image_ids[index]:
                    raise RuntimeError(
                        f"Fixed Query image does not match row {index} for {task_id}"
                    )
                query_indices.add(index)
        split_masks = {
            "Development": development_mask,
            "Validation": validation_mask,
            "Frozen Test": test_mask,
        }
        for name, mask in split_masks.items():
            if any(value not in (0, 1) for value in mask):
                raise RuntimeError(f"{name} mask is not binary for {task_id}")
        for index in range(spec.row_count):
            membership = sum(mask[index] for mask in split_masks.values())
            expected = 0 if index in query_indices else 1
            if membership != expected:
                requirement = "no split" if expected == 0 else "exactly one split"
                raise RuntimeError(
                    f"Row {index} must belong to {requirement} for {task_id}; "
                    f"found {membership}"
                )
        return BundleData(
            spec=spec,
            image_ids=image_ids,
            development_mask=development_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            query_indices=frozenset(query_indices),
        )

    def _bundle_file(self, task: TaskSpec, relative: str) -> Path:
        path = (task.data_root / str(relative)).resolve()
        if not path.is_relative_to(task.data_root) or not path.is_file():
            raise RuntimeError(f"Bundle file is unavailable for {task.task_id}: {relative}")
        return path

    @lru_cache(maxsize=3)
    def _source_adapter(self, task_id: str) -> Any:
        """Load only the dataset metadata needed to resolve an original image."""
        import local_tasks
        if local_tasks.available(self, task_id):
            return local_tasks.adapter(self, task_id)
        import portable_tasks
        if portable_tasks.available(self, task_id):
            return portable_tasks.adapter(self, task_id)
        with self._source_adapter_guard:
            task = self.task(task_id)
            experiment_root = self.web_root.parents[2]
            linear_root = experiment_root / "probe_learning"
            for path in (linear_root, linear_root / "scripts"):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            from src.data.dataset_adapter import build_adapter

            return build_adapter(task.dataset_id, task.task_name)

    @lru_cache(maxsize=3)
    def _tuning_source_context(self, task_id: str) -> TuningSourceContext:
        """Resolve task/target metadata without loading the learner score tensors."""
        task = self.task(task_id)
        adapter = self._source_adapter(task_id)
        import local_tasks
        if local_tasks.available(self, task_id):
            attributes = tuple(adapter.attrs)
            return TuningSourceContext(adapter=adapter,
                task_config={"dataset": task.dataset_id, "task": task.task_name},
                attributes=attributes, target_attributes=attributes + (None,),
                probe_stage="local", suite_path=None, supervision_audit_path=None)
        experiment_root = self.web_root.parents[2]
        linear_root = experiment_root / "probe_learning"
        pcp_root = self.web_root.parent
        for path in (linear_root, linear_root / "scripts", pcp_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import build_score_rank_pcp as pcp

        try:
            order = int(task_id.split("_", 1)[0])
        except ValueError as error:
            raise RuntimeError(f"Task id has no numeric order: {task_id}") from error
        source_spec = task.tuning_source
        manifest_path = (
            source_spec.manifest_path if source_spec is not None else pcp.DEFAULT_MANIFEST
        )
        task_config = dict(pcp.find_task(pcp.load_json(manifest_path), order))
        if pcp.task_key(task_config) != task_id:
            raise RuntimeError(
                f"Source task identity does not match bundle: {pcp.task_key(task_config)}"
            )
        gt_by_attr = getattr(adapter, "gt_by_attr", None)
        if not isinstance(gt_by_attr, Mapping) or not gt_by_attr:
            raise RuntimeError(f"Source adapter has no modeled attributes for {task_id}")
        attributes = tuple(str(value) for value in gt_by_attr.keys())
        target_ids = tuple(str(value) for value in getattr(adapter, "key_slugs", ())) + (
            "joint",
        )
        if len(attributes) + 1 != len(target_ids) or target_ids != task.target_ids:
            raise RuntimeError(
                f"Source target order does not match bundle for {task_id}: {target_ids}"
            )
        return TuningSourceContext(
            adapter=adapter,
            task_config=task_config,
            attributes=attributes,
            target_attributes=attributes + (None,),
            probe_stage=(
                source_spec.probe_stage if source_spec is not None else "iterative"
            ),
            suite_path=(source_spec.suite_path if source_spec is not None else None),
            supervision_audit_path=(
                source_spec.supervision_audit_path if source_spec is not None else None
            ),
        )

    @lru_cache(maxsize=3)
    def _frozen_train_indices(self, task_id: str) -> Any:
        """Resolve the frozen train pool without loading learner score tensors."""

        import numpy as np
        import local_tasks
        if local_tasks.available(self, task_id):
            bundle = self.bundle(task_id)
            val = set(self._vqa_validation_split(task_id, "joint").validation_indices)
            return np.asarray([r for r in range(len(bundle.image_ids))
                               if r not in val and r not in bundle.query_indices
                               and not bundle.test_mask[r]], dtype=np.int64)
        import portable_tasks
        if portable_tasks.available(self, task_id):
            portable_tasks.validation_split(self, task_id, "joint")
            return np.asarray(portable_tasks.document(self, task_id)[1]["trainPoolRows"], dtype=np.int64)

        context = self._tuning_source_context(task_id)
        pcp_root = self.web_root.parent
        if str(pcp_root) not in sys.path:
            sys.path.insert(0, str(pcp_root))
        import build_score_rank_pcp as pcp

        _, train_indices = pcp.split_indices(context.adapter, "train_pool")
        indices = np.asarray(train_indices, dtype=np.int64)
        task = self.task(task_id)
        if (
            indices.ndim != 1
            or indices.size == 0
            or int(indices.min()) < 0
            or int(indices.max()) >= task.row_count
        ):
            raise RuntimeError(f"Frozen train pool is invalid for {task_id}")
        return indices

    def _original_development_supervision(self, task_id: str, target_id: str) -> Any:
        """Load the audited original labels plus their legacy DG-only view."""
        import local_tasks
        if local_tasks.available(self, task_id):
            return local_tasks.original_supervision(self, task_id, target_id)
        import portable_tasks
        if portable_tasks.available(self, task_id):
            return portable_tasks.original_supervision(self, task_id, target_id)
        from experimental_parent_overlay import applies, original_supervision
        if applies(task_id):
            return original_supervision(self, task_id, target_id)
        key = (task_id, target_id)
        with self._original_supervision_guard:
            cached = self._original_supervision_cache.get(key)
            if cached is not None:
                return cached
            task = self.task(task_id)
            if target_id not in task.target_ids:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"Unknown target for {task_id}: {target_id}",
                )
            bundle = self.bundle(task_id)
            source = self._tuning_source_context(task_id)
            target_attribute = source.target_attributes[task.target_ids.index(target_id)]
            from tuning_supervision import load_original_development_supervision

            original = load_original_development_supervision(
                self.web_root.parents[2],
                source.task_config,
                source.adapter,
                bundle.image_ids,
                bundle.development_mask,
                target_attribute,
                stage=source.probe_stage,
                source_audit_path=source.supervision_audit_path,
                suite_path=source.suite_path,
            )
            if len(self._original_supervision_cache) >= 12:
                oldest = next(iter(self._original_supervision_cache))
                del self._original_supervision_cache[oldest]
            self._original_supervision_cache[key] = original
            return original

    @lru_cache(maxsize=12)
    def _probe_validation_split(self, task_id: str, target_id: str) -> Any:
        """Return the frozen target-level seed-0 Probe 80:20 replay."""

        task = self.task(task_id)
        target = next(
            (
                value
                for value in task.manifest.get("retrievalTargets", [])
                if str(value.get("id")) == target_id
            ),
            None,
        )
        if not isinstance(target, Mapping):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"Unknown target for {task_id}: {target_id}",
            )
        original = self._original_development_supervision(task_id, target_id)
        from tuning_supervision import build_probe_validation_split

        split = build_probe_validation_split(
            original,
            target_id=target_id,
            target_kind=str(target.get("kind") or "derived"),
            seed=PROBE_VALIDATION_SEED,
            validation_fraction=PROBE_VALIDATION_FRACTION,
        )
        if split.audit.get("protocol") != PROBE_VALIDATION_PROTOCOL:
            raise RuntimeError("Probe Validation protocol version mismatch")
        bundle = self.bundle(task_id)
        forbidden = [
            int(row)
            for row in original.selected_indices
            if bundle.test_mask[int(row)] or int(row) in bundle.query_indices
        ]
        if forbidden:
            raise RuntimeError(
                "Original Probe supervision overlaps Frozen Test or Query rows"
            )
        split.audit["fitWebDevelopmentCount"] = int(
            sum(bundle.development_mask[int(row)] for row in split.fit_indices)
        )
        split.audit["fitWebValidationCount"] = int(
            sum(bundle.validation_mask[int(row)] for row in split.fit_indices)
        )
        split.audit["validationWebDevelopmentCount"] = int(
            sum(
                bundle.development_mask[int(row)]
                for row in split.validation_indices
            )
        )
        split.audit["validationWebValidationCount"] = int(
            sum(
                bundle.validation_mask[int(row)]
                for row in split.validation_indices
            )
        )
        return split

    def _task_ground_truth(self, task_id: str) -> Any:
        """Read the immutable row-aligned target labels for evaluation only."""

        import numpy as np

        task = self.task(task_id)
        specification = task.manifest["files"].get("groundTruth")
        if specification is None:
            raise ValueError("This local task has no independent ground truth or Frozen Test evaluation")
        path = self._bundle_file(task, specification["path"])
        truth = np.fromfile(path, dtype=np.uint8)
        expected = task.row_count * len(task.target_ids)
        if truth.size != expected:
            raise RuntimeError(
                f"Ground-truth shape is invalid for {task_id}: "
                f"{truth.size} != {expected}"
            )
        truth = truth.reshape(task.row_count, len(task.target_ids))
        if not np.isin(truth, (0, 1)).all():
            raise RuntimeError(f"Ground truth is not binary for {task_id}")
        return truth

    def _historically_reviewed_rows(self, task_id: str) -> set[int]:
        """Return every row ever annotated for a task, including deleted labels."""

        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT e.row_index FROM annotation_events e "
                "JOIN sessions s ON s.id=e.session_id WHERE s.task_id=? "
                "UNION SELECT DISTINCT a.row_index FROM annotations a "
                "JOIN sessions s ON s.id=a.session_id WHERE s.task_id=?",
                (task_id, task_id),
            ).fetchall()
        return {int(row[0]) for row in rows}

    def _clean_validation_split(self, task_id: str, target_id: str) -> Any:
        """Load the active fixed set; never rebuild it from live annotations."""
        from frozen_validation import load_frozen_validation_split

        return load_frozen_validation_split(self, task_id, target_id)

    def _vqa_validation_split(
        self, task_id: str, target_id: str, version: str | None = None,
    ) -> Any:
        from experimental_parent_overlay import applies, validation_split
        if applies(task_id):
            return validation_split(self, task_id, target_id, version)
        from fixed_vqa_validation import load_vqa_validation_split

        return load_vqa_validation_split(self, task_id, target_id, version=version)

    def task_validation(self, task_id: str) -> dict[str, Any]:
        """Expose membership, never evaluation labels, for the DG safety gate."""
        split = self._vqa_validation_split(task_id, "joint")
        return {
            "taskId": task_id,
            "version": split.audit["frozenVersion"],
            "manifestSha256": split.audit["frozenManifestSha256"],
            "rowIndices": [int(row) for row in split.validation_indices],
            "count": int(split.audit["validationCount"]),
            "protocol": split.audit["protocol"],
            "labelSource": split.audit.get("evaluationLabelSource", "original-vqa-supervision"),
            "initialModelHoldoutIndependent": False,
            "referenceOnly": True,
        }

    @staticmethod
    def _vqa_validation_contract(split: Any) -> dict[str, Any]:
        """Pin the entire verified frozen audit, including all source hashes."""
        return dict(split.audit)

    def _validation_split_for_run(
        self, task_id: str, target_id: str, version: str | None = None,
    ) -> Any:
        if version is None:
            return self._clean_validation_split(task_id, target_id)
        from frozen_validation import load_frozen_validation_split

        return load_frozen_validation_split(self, task_id, target_id, version=version)

    def _build_clean_validation_split(
        self, task_id: str, target_id: str, *, reviewed_rows: set[int] | None = None,
    ) -> Any:
        """Build a label-isolated Validation split for a five-seed ensemble.

        The training side deliberately preserves the historical seed-0 Probe
        fit partition.  Evaluation instead uses the public Web Validation
        population after excluding every row in the complete original VQA
        supervision snapshot and every historically reviewed row.  Query and
        Frozen Test are already disjoint from Web Validation and are asserted
        again here.  Consequently, no label used by any deployed Probe seed or
        by a user can appear in the before/after metric population.

        Existing Probes may still have observed these images as *unlabelled*
        gallery data, so this is a label-clean transductive Validation rather
        than an inductive image holdout.
        """

        import numpy as np
        from tuning_supervision import ProbeValidationSplit

        task = self.task(task_id)
        if target_id not in task.target_ids:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"Unknown target for {task_id}: {target_id}",
            )
        target_index = task.target_ids.index(target_id)
        historical_fit = self._probe_validation_split(task_id, target_id)
        bundle = self.bundle(task_id)

        original_rows: set[int] = set()
        original_fingerprints: dict[str, str] = {}
        for candidate_target in task.target_ids:
            original = self._original_development_supervision(
                task_id, candidate_target
            )
            original_rows.update(
                int(value) for value in original.selected_indices
            )
            original_fingerprints[candidate_target] = str(
                original.audit.get("recoveredSupervisionHash") or ""
            )

        reviewed_rows = (
            self._historically_reviewed_rows(task_id)
            if reviewed_rows is None else reviewed_rows
        )
        validation_mask = np.frombuffer(
            bundle.validation_mask, dtype=np.uint8
        ).astype(bool)
        test_mask = np.frombuffer(bundle.test_mask, dtype=np.uint8).astype(bool)
        excluded_original = sorted(
            row for row in original_rows if validation_mask[row]
        )
        excluded_reviewed = sorted(
            row
            for row in reviewed_rows
            if validation_mask[row] and row not in original_rows
        )
        validation_mask[np.asarray(excluded_original, dtype=np.int64)] = False
        validation_mask[np.asarray(excluded_reviewed, dtype=np.int64)] = False
        validation_indices = np.flatnonzero(validation_mask).astype(np.int64)
        if validation_indices.size == 0:
            raise RuntimeError(f"Clean Validation is empty for {task_id}")
        if any(int(row) in bundle.query_indices for row in validation_indices):
            raise RuntimeError("Clean Validation overlaps the fixed Query")
        if np.any(test_mask[validation_indices]):
            raise RuntimeError("Clean Validation overlaps Frozen Test")
        if original_rows.intersection(int(row) for row in validation_indices):
            raise RuntimeError("Clean Validation overlaps original VQA supervision")
        if reviewed_rows.intersection(int(row) for row in validation_indices):
            raise RuntimeError("Clean Validation overlaps historical user feedback")

        ground_truth = self._task_ground_truth(task_id)
        validation_labels = np.asarray(
            ground_truth[validation_indices, target_index], dtype=np.uint8
        )
        positive_count = int(np.count_nonzero(validation_labels))
        negative_count = int(validation_labels.size - positive_count)
        if positive_count == 0 or negative_count == 0:
            raise RuntimeError(
                f"Clean Validation requires both classes for {task_id}/{target_id}"
            )

        def fingerprint(partition: str, rows: Any, labels: Any) -> str:
            payload = {
                "partition": partition,
                "protocol": CLEAN_VALIDATION_PROTOCOL,
                "taskId": task_id,
                "targetId": target_id,
                "records": [
                    [bundle.image_ids[int(row)], int(label)]
                    for row, label in zip(rows, labels, strict=True)
                ],
            }
            return hashlib.sha256(
                json_dumps(payload).encode("utf-8")
            ).hexdigest()

        excluded_original_fingerprint = hashlib.sha256(
            json_dumps(
                [bundle.image_ids[row] for row in excluded_original]
            ).encode("utf-8")
        ).hexdigest()
        excluded_reviewed_fingerprint = hashlib.sha256(
            json_dumps(
                [bundle.image_ids[row] for row in excluded_reviewed]
            ).encode("utf-8")
        ).hexdigest()
        validation_fingerprint = fingerprint(
            "clean-validation", validation_indices, validation_labels
        )
        audit = {
            "schemaVersion": 2,
            "protocol": CLEAN_VALIDATION_PROTOCOL,
            "replayKind": "clean-web-validation-ground-truth",
            "targetId": target_id,
            "targetKind": "derived" if target_id == "joint" else "attribute",
            "labelSource": "dataset-ground-truth",
            "sourceScope": "web-validation",
            "transductive": True,
            "queryExcluded": True,
            "testExcluded": True,
            "allOriginalSupervisionExcluded": True,
            "allHistoricalFeedbackExcluded": True,
            "fitProtocol": PROBE_VALIDATION_PROTOCOL,
            "fitSeed": PROBE_VALIDATION_SEED,
            "fitFraction": 1.0 - PROBE_VALIDATION_FRACTION,
            "sourceCount": int(historical_fit.audit["sourceCount"]),
            "fitCount": int(historical_fit.fit_indices.size),
            "fitPositiveCount": int(np.count_nonzero(historical_fit.fit_labels)),
            "fitNegativeCount": int(
                historical_fit.fit_labels.size
                - np.count_nonzero(historical_fit.fit_labels)
            ),
            "validationCount": int(validation_indices.size),
            "validationPositiveCount": positive_count,
            "validationNegativeCount": negative_count,
            "sourceFingerprint": str(
                historical_fit.audit["sourceFingerprint"]
            ),
            "fitFingerprint": str(historical_fit.audit["fitFingerprint"]),
            "validationFingerprint": validation_fingerprint,
            "originalSupervisionFingerprints": original_fingerprints,
            "excludedOriginalSupervisionCount": len(excluded_original),
            "excludedOriginalSupervisionFingerprint": (
                excluded_original_fingerprint
            ),
            "excludedHistoricalFeedbackCount": len(excluded_reviewed),
            "excludedHistoricalFeedbackFingerprint": (
                excluded_reviewed_fingerprint
            ),
        }
        for values in (
            validation_indices,
            validation_labels,
        ):
            values.setflags(write=False)
        return ProbeValidationSplit(
            fit_indices=historical_fit.fit_indices,
            fit_labels=historical_fit.fit_labels,
            validation_indices=validation_indices,
            validation_labels=validation_labels,
            audit=audit,
        )

    def _prepare_probe_tuning_supervision(
        self,
        task_id: str,
        target_id: str,
        feedback_indices: Any,
        feedback_labels: Any,
        *,
        feedback_weight: float,
        clean_validation: bool = True,
        clean_validation_version: str | None = None,
        vqa_validation: bool = False,
        vqa_validation_version: str | None = None,
    ) -> tuple[Any, Any, Any, dict[str, Any], Any]:
        """Build fit supervision while keeping the chosen holdout immutable."""

        import numpy as np

        original = self._original_development_supervision(task_id, target_id)
        split = self._vqa_validation_split(task_id, target_id, vqa_validation_version) if vqa_validation else (
            self._validation_split_for_run(task_id, target_id, clean_validation_version)
            if clean_validation
            else self._probe_validation_split(task_id, target_id)
        )
        feedback_rows = np.asarray(feedback_indices, dtype=np.int64).reshape(-1)
        signed_feedback = np.asarray(feedback_labels, dtype=np.int8).reshape(-1)
        if feedback_rows.shape != signed_feedback.shape:
            raise RuntimeError("Feedback rows and labels are not aligned")
        validation_rows = set(int(value) for value in split.validation_indices)
        keep = np.asarray(
            [int(row) not in validation_rows for row in feedback_rows],
            dtype=bool,
        )
        excluded_count = int(np.count_nonzero(~keep))
        if feedback_rows.size and not np.any(keep):
            raise RuntimeError(
                "All feedback labels fall in the fixed Validation holdout; "
                "label at least one other Development image"
            )
        rows, truth, weights, merge_audit = self._merge_tuning_supervision(
            split.fit_indices,
            split.fit_labels,
            feedback_rows[keep],
            signed_feedback[keep],
            feedback_weight=feedback_weight,
        )
        audit = {
            **original.audit,
            **merge_audit,
            "supervisionPolicy": (
                VQA_VALIDATION_SUPERVISION_POLICY if vqa_validation else
                SUPERVISION_POLICY
                if clean_validation or vqa_validation
                else PROBE_VALIDATION_SUPERVISION_POLICY
            ),
            "originalProbeSupervisionCount": int(split.audit["sourceCount"]),
            "originalProbeFitCount": int(split.audit["fitCount"]),
            "probeValidationCount": (
                0
                if clean_validation or vqa_validation
                else int(split.audit["validationCount"])
            ),
            "cleanValidationCount": int(split.audit["validationCount"])
            if clean_validation and not vqa_validation
            else 0,
            "vqaValidationCount": int(split.audit["validationCount"])
            if vqa_validation else 0,
            "feedbackSnapshotCount": int(feedback_rows.size),
            "feedbackHoldoutExcludedCount": excluded_count,
            "probeValidation": split.audit,
        }
        return rows, truth, weights, audit, split

    def _prepare_weight_refinement_supervision(
        self,
        task_id: str,
        base: RefinementBaseContext,
        annotations: list[dict[str, Any]],
        *,
        feedback_weight: float,
        clean_validation: bool = True,
        clean_validation_version: str | None = None,
        vqa_validation: bool = False,
        vqa_validation_version: str | None = None,
    ) -> dict[str, Any]:
        """Create audited attribute and Joint supervision for refinement.

        A positive Joint correction supplies a positive label for every query
        attribute.  A negative correction supplies an attribute label only
        when ``failedAttributeIds`` was explicitly confirmed; absent IDs remain
        unknown and therefore cannot update an attribute branch.  The merge
        path stays conservative for historical/replayed snapshots, although
        creation of a new Weight-only run requires confirmed failed attributes,
        except explicit relation-only confirmations in the robustness pilot.
        Those pilot rows keep all attributes unknown and route Joint gradients
        only to the holistic eta/lambda branch.
        """

        import numpy as np

        feedback_by_row = {
            int(row["rowIndex"]): row
            for row in annotations
            if int(row.get("label", 0)) != 0
        }
        attribute_examples: list[dict[str, Any]] = []
        calibration_examples: list[dict[str, Any]] = []
        original_attribute_fit_labels: list[dict[int, int]] = []
        attribute_audits: dict[str, Any] = {}

        for attribute_index, attribute_id in enumerate(base.attribute_ids):
            split = self._vqa_validation_split(task_id, attribute_id, vqa_validation_version) if vqa_validation else (
                self._validation_split_for_run(task_id, attribute_id, clean_validation_version)
                if clean_validation
                else self._probe_validation_split(task_id, attribute_id)
            )
            fit_rows = np.asarray(split.fit_indices, dtype=np.int64)
            fit_labels = np.asarray(split.fit_labels, dtype=np.uint8)
            original_attribute_fit_labels.append(
                {
                    int(row): int(label)
                    for row, label in zip(fit_rows, fit_labels, strict=True)
                }
            )
            holdout = {int(row) for row in split.validation_indices}
            feedback_rows: list[int] = []
            feedback_labels: list[int] = []
            for row_index, annotation in feedback_by_row.items():
                if row_index in holdout:
                    continue
                signed_label = int(annotation["label"])
                if signed_label > 0:
                    feedback_rows.append(row_index)
                    feedback_labels.append(signed_label)
                    continue
                failed_ids = (
                    annotation.get("failedAttributeIds") or ()
                    if annotation.get("failureAttributionConfirmed") is True
                    else ()
                )
                if attribute_id in {str(value) for value in failed_ids}:
                    feedback_rows.append(row_index)
                    feedback_labels.append(signed_label)

            merged_rows, merged_labels, merged_weights, merge_audit = (
                self._merge_tuning_supervision(
                    fit_rows,
                    fit_labels,
                    np.asarray(feedback_rows, dtype=np.int64),
                    np.asarray(feedback_labels, dtype=np.int8),
                    feedback_weight=feedback_weight,
                )
            )
            attribute_examples.append(
                {
                    "attributeIndex": attribute_index,
                    "rows": merged_rows,
                    "labels": merged_labels,
                    "weights": merged_weights,
                }
            )
            calibration_examples.append(
                {
                    "attributeIndex": attribute_index,
                    "rows": fit_rows,
                    "labels": fit_labels,
                    "weights": np.ones(fit_rows.size, dtype=np.float64),
                }
            )
            attribute_audits[attribute_id] = {
                **dict(split.audit),
                **merge_audit,
            }

        feedback_rows = np.asarray(
            [int(row["rowIndex"]) for row in annotations], dtype=np.int64
        )
        feedback_labels = np.asarray(
            [int(row["label"]) for row in annotations], dtype=np.int8
        )
        (
            joint_rows,
            joint_labels,
            joint_weights,
            joint_audit,
            joint_split,
        ) = self._prepare_probe_tuning_supervision(
            task_id,
            "joint",
            feedback_rows,
            feedback_labels,
            feedback_weight=feedback_weight,
            clean_validation=clean_validation,
            clean_validation_version=clean_validation_version,
            vqa_validation=vqa_validation,
            vqa_validation_version=vqa_validation_version,
        )
        joint_failure_mask = np.zeros(
            (joint_rows.size, len(base.attribute_ids)), dtype=bool
        )
        explicit_failure_rows = 0
        for position, (row_index, label) in enumerate(
            zip(joint_rows, joint_labels, strict=True)
        ):
            row_value = int(row_index)
            annotation = feedback_by_row.get(row_value)
            if annotation is not None:
                if int(annotation["label"]) > 0:
                    joint_failure_mask[position, :] = True
                else:
                    failed_ids = (
                        {
                            str(value)
                            for value in (
                                annotation.get("failedAttributeIds") or ()
                            )
                        }
                        if annotation.get("failureAttributionConfirmed") is True
                        else set()
                    )
                    for attribute_index, attribute_id in enumerate(base.attribute_ids):
                        joint_failure_mask[position, attribute_index] = (
                            attribute_id in failed_ids
                        )
                    if failed_ids:
                        explicit_failure_rows += 1
                continue
            if int(label) > 0:
                joint_failure_mask[position, :] = True
            else:
                for attribute_index, labels_by_row in enumerate(
                    original_attribute_fit_labels
                ):
                    joint_failure_mask[position, attribute_index] = (
                        labels_by_row.get(row_value) == 0
                    )

        return {
            "attributeExamples": attribute_examples,
            "calibrationExamples": calibration_examples,
            "jointRows": joint_rows,
            "jointLabels": joint_labels,
            "jointWeights": joint_weights,
            "jointFailureMask": joint_failure_mask,
            "jointSplit": joint_split,
            "audit": {
                **joint_audit,
                "attributeTargets": attribute_audits,
                "explicitFailureAttributeCount": explicit_failure_rows,
                **({"relationOnlyFeedbackCount": sum(
                    int(row["label"]) < 0 and row.get("failureAttributionConfirmed") is True
                    and row.get("failedAttributeIds") == []
                    and int(row["rowIndex"]) in joint_rows
                    for row in annotations
                )} if supports_relation_mismatch(task_id, "joint") else {}),
                "negativeFeedbackWithoutAttributionCount": sum(
                    int(row["label"]) < 0
                    and not (
                        row.get("failureAttributionConfirmed") is True
                        and (row.get("failedAttributeIds") or ())
                    )
                    for row in annotations
                ),
            },
        }

    @staticmethod
    def _materialize_refinement_training_arrays(
        base: RefinementBaseContext,
        prepared: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Align sparse per-target supervision on compact union row sets."""

        import numpy as np

        attribute_examples = list(prepared["attributeExamples"])
        joint_rows = np.asarray(prepared["jointRows"], dtype=np.int64)
        union_rows = np.asarray(
            sorted(
                {int(value) for value in joint_rows}
                | {
                    int(value)
                    for example in attribute_examples
                    for value in np.asarray(example["rows"], dtype=np.int64)
                }
            ),
            dtype=np.int64,
        )
        if union_rows.size == 0:
            raise RuntimeError("Refinement supervision is empty")
        position_by_row = {
            int(row): position for position, row in enumerate(union_rows)
        }
        attribute_count = len(base.attribute_ids)
        attribute_labels = np.full(
            (union_rows.size, attribute_count), np.nan, dtype=np.float64
        )
        attribute_weights = np.ones_like(attribute_labels)
        for example in attribute_examples:
            attribute_index = int(example["attributeIndex"])
            for row, label, weight in zip(
                np.asarray(example["rows"], dtype=np.int64),
                np.asarray(example["labels"], dtype=np.uint8),
                np.asarray(example["weights"], dtype=np.float64),
                strict=True,
            ):
                position = position_by_row[int(row)]
                attribute_labels[position, attribute_index] = int(label)
                attribute_weights[position, attribute_index] = float(weight)

        joint_labels = np.full(union_rows.size, np.nan, dtype=np.float64)
        joint_weights = np.ones(union_rows.size, dtype=np.float64)
        joint_failure_mask = np.zeros(
            (union_rows.size, attribute_count), dtype=bool
        )
        for source_position, (row, label, weight) in enumerate(
            zip(
                joint_rows,
                np.asarray(prepared["jointLabels"], dtype=np.uint8),
                np.asarray(prepared["jointWeights"], dtype=np.float64),
                strict=True,
            )
        ):
            position = position_by_row[int(row)]
            joint_labels[position] = int(label)
            joint_weights[position] = float(weight)
            joint_failure_mask[position] = np.asarray(
                prepared["jointFailureMask"][source_position], dtype=bool
            )

        calibration_examples = list(prepared["calibrationExamples"])
        calibration_rows = np.asarray(
            sorted(
                {
                    int(value)
                    for example in calibration_examples
                    for value in np.asarray(example["rows"], dtype=np.int64)
                }
            ),
            dtype=np.int64,
        )
        calibration_position = {
            int(row): position for position, row in enumerate(calibration_rows)
        }
        calibration_labels = np.full(
            (calibration_rows.size, attribute_count), np.nan, dtype=np.float64
        )
        calibration_weights = np.ones_like(calibration_labels)
        for example in calibration_examples:
            attribute_index = int(example["attributeIndex"])
            for row, label, weight in zip(
                np.asarray(example["rows"], dtype=np.int64),
                np.asarray(example["labels"], dtype=np.uint8),
                np.asarray(example["weights"], dtype=np.float64),
                strict=True,
            ):
                position = calibration_position[int(row)]
                calibration_labels[position, attribute_index] = int(label)
                calibration_weights[position, attribute_index] = float(weight)

        return {
            "rows": union_rows,
            "probeScores": np.asarray(base.probe_features[union_rows], dtype=np.float32),
            "embeddingScores": np.asarray(
                base.embedding_features[union_rows], dtype=np.float32
            ),
            "jointLabels": joint_labels,
            "jointWeights": joint_weights,
            "attributeLabels": attribute_labels,
            "attributeWeights": attribute_weights,
            "jointFailureMask": joint_failure_mask,
            "calibrationRows": calibration_rows,
            "calibrationProbeScores": np.asarray(
                base.probe_features[calibration_rows], dtype=np.float32
            ),
            "calibrationLabels": calibration_labels,
            "calibrationWeights": calibration_weights,
        }

    def _refinement_calibration_arrays(
        self,
        task_id: str,
        base: RefinementBaseContext,
        *,
        vqa_validation_version: str | None = None,
    ) -> tuple[Any, Any, Any]:
        """Return the fixed per-attribute Probe-fit calibration matrix.

        This set is independent of the current user's feedback.  It is used
        both when freezing a run and when verifying that snapshot in
        the worker, so the formal initial threshold cannot drift between the
        two phases.
        """

        import numpy as np

        examples: list[tuple[int, Any, Any]] = []
        calibration_rows: set[int] = set()
        for attribute_index, attribute_id in enumerate(base.attribute_ids):
            split = (
                self._vqa_validation_split(task_id, attribute_id, vqa_validation_version)
                if vqa_validation_version is not None
                else self._probe_validation_split(task_id, attribute_id)
            )
            rows = np.asarray(split.fit_indices, dtype=np.int64)
            labels = np.asarray(split.fit_labels, dtype=np.uint8)
            if rows.shape != labels.shape:
                raise RuntimeError(
                    "Refinement calibration rows and labels are not aligned"
                )
            examples.append((attribute_index, rows, labels))
            calibration_rows.update(int(row) for row in rows)
        ordered_rows = np.asarray(sorted(calibration_rows), dtype=np.int64)
        if ordered_rows.size == 0:
            raise RuntimeError("Refinement calibration set is empty")
        position_by_row = {
            int(row): position for position, row in enumerate(ordered_rows)
        }
        labels = np.full(
            (ordered_rows.size, len(base.attribute_ids)),
            np.nan,
            dtype=np.float64,
        )
        weights = np.ones_like(labels)
        for attribute_index, rows, truth in examples:
            for row, label in zip(rows, truth, strict=True):
                labels[position_by_row[int(row)], attribute_index] = int(label)
        return (
            np.asarray(base.probe_features[ordered_rows], dtype=np.float32),
            labels,
            weights,
        )

    def original_vqa_supervision(self, task_id: str, target_id: str) -> dict[str, Any]:
        """Return immutable original VQA labels for provenance display only.

        The response intentionally uses the strictly recovered selected VQA
        snapshot rather than public ground truth.  ``trainableInDevelopment``
        makes the Web-purpose boundary explicit; new Personal Tune runs replay
        their separate seed-0 Probe fit/validation split from the complete
        selected supervision.
        """

        task = self.task(task_id)
        if target_id not in task.target_ids:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"Unknown target for {task_id}: {target_id}",
            )
        bundle = self.bundle(task_id)
        original = self._original_development_supervision(task_id, target_id)
        selected_indices = tuple(int(value) for value in original.selected_indices)
        selected_labels = tuple(int(value) for value in original.selected_labels)
        validation_rows = {
            int(row) for row in self._vqa_validation_split(task_id, "joint").validation_indices
        }
        if len(selected_indices) != len(selected_labels):
            raise RuntimeError("Original VQA supervision rows and labels are not aligned")

        seen: set[int] = set()
        items: list[dict[str, Any]] = []
        for row_index, label in zip(selected_indices, selected_labels, strict=True):
            if row_index in seen:
                raise RuntimeError("Original VQA supervision contains duplicate rows")
            if row_index < 0 or row_index >= task.row_count:
                raise RuntimeError("Original VQA supervision row is outside the task bundle")
            if label not in (0, 1):
                raise RuntimeError("Original VQA supervision must be binary")
            seen.add(row_index)
            items.append(
                {
                    "rowIndex": row_index,
                    "imageId": bundle.image_ids[row_index],
                    "label": label,
                    "trainableInDevelopment": bool(bundle.development_mask[row_index])
                    and row_index not in validation_rows,
                }
            )
        return {
            "taskId": task_id,
            "targetId": target_id,
            "selectedCount": len(items),
            "developmentCount": sum(
                int(item["trainableInDevelopment"]) for item in items
            ),
            "items": items,
        }

    def gallery_image(self, task_id: str, row_index: int) -> Path:
        """Resolve a catalog row to a source image without accepting filesystem input."""
        from experimental_parent_overlay import applies, gallery_image
        if applies(task_id):
            return gallery_image(self, row_index)
        bundle = self.bundle(task_id)
        if row_index < 0 or row_index >= bundle.spec.row_count:
            raise ApiError(HTTPStatus.BAD_REQUEST, "rowIndex is outside the task bundle")
        image_id = bundle.image_ids[row_index]
        adapter = self._source_adapter(task_id)
        source_root = Path(adapter.raw_images_dir).resolve()
        try:
            image_path = Path(adapter.resolve_image_path(image_id)).resolve()
        except (KeyError, FileNotFoundError) as error:
            raise ApiError(HTTPStatus.NOT_FOUND, "Original image is unavailable") from error
        if not image_path.is_relative_to(source_root) or not image_path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "Original image is unavailable")
        return image_path

    def _weighted_fusion_context(self, task_id: str) -> WeightedFusionContext:
        """Load the audited per-seed inputs for one task on first non-equal use."""
        with self._weighted_fusion_context_guard:
            cached = self._weighted_fusion_contexts.get(task_id)
            if cached is not None:
                return cached
            import numpy as np

            task = self.task(task_id)
            bundle = self.bundle(task_id)
            experiment_root = self.web_root.parents[2]
            linear_root = experiment_root / "probe_learning"
            pcp_root = self.web_root.parent
            for path in (linear_root, linear_root / "scripts", pcp_root):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            import build_score_rank_pcp as pcp

            source_context = self._tuning_source_context(task_id)
            source_spec = task.tuning_source
            suite = pcp.load_json(
                source_spec.suite_path if source_spec is not None else pcp.DEFAULT_SUITE
            )
            task_config = source_context.task_config

            pcp.H.configure(task.dataset_id, task.task_name)
            adapter = pcp.H.ADAPTER
            attributes = tuple(pcp.H.ATTRS)
            target_ids = tuple(str(value) for value in adapter.key_slugs) + ("joint",)
            target_attributes = attributes + (None,)
            if target_ids != task.target_ids:
                raise RuntimeError(
                    f"Source target order does not match bundle for {task_id}: {target_ids}"
                )
            database_paths = pcp.B.database_paths(adapter)
            gallery_paths, gallery_indices = pcp.split_indices(adapter, "gallery")
            _, train_indices = pcp.split_indices(adapter, "train_pool")
            if tuple(str(value) for value in gallery_paths) != bundle.image_ids:
                raise RuntimeError(
                    f"Source score order does not match browser bundle for {task_id}"
                )
            identity = np.arange(len(database_paths), dtype=np.int64)
            gallery_indices = np.asarray(gallery_indices, dtype=np.int64)
            if not np.array_equal(gallery_indices, identity):
                raise RuntimeError(f"Gallery is not the full source order for {task_id}")

            configured = list(suite["learned_methods"])
            learned_ids = tuple(
                [method for method in pcp.LEARNED_METHOD_ORDER if method in configured]
                + [method for method in configured if method not in pcp.LEARNED_METHOD_ORDER]
            )
            learned_labels = tuple(pcp.DISPLAY_NAMES.get(method, method) for method in learned_ids)
            if learned_labels != WEIGHTED_FUSION_LEARNERS:
                raise RuntimeError(
                    "Audited learner order does not match the weighted-fusion contract: "
                    f"{learned_labels}"
                )
            tensors, seeds_by_method, _ = pcp.load_score_tensor(
                suite,
                task_config,
                list(learned_ids),
                list(attributes),
                database_paths,
                source_context.probe_stage,
            )
            selected_seeds = (0, 1, 2, 3, 4)
            require_selected_seeds(seeds_by_method, learned_ids, selected_seeds)
            context = WeightedFusionContext(
                pcp=pcp,
                adapter=adapter,
                task_config=task_config,
                tensors=tensors,
                seeds_by_method=seeds_by_method,
                learned_method_ids=learned_ids,
                learned_method_labels=learned_labels,
                attributes=attributes,
                target_attributes=target_attributes,
                selected_seeds=selected_seeds,
                gallery_indices=gallery_indices,
                train_indices=np.asarray(train_indices, dtype=np.int64),
                ours_full_root=(
                    source_spec.ours_full_root
                    if source_spec is not None
                    else pcp.DEFAULT_OURS_FULL_ROOT
                ),
                probe_stage=source_context.probe_stage,
            )
            if len(self._weighted_fusion_contexts) >= 2:
                oldest = next(iter(self._weighted_fusion_contexts))
                del self._weighted_fusion_contexts[oldest]
            self._weighted_fusion_contexts[task_id] = context
            return context

    @staticmethod
    def _refinement_embedding_methods(mode: str, params: Mapping[str, Any]) -> tuple[str, ...]:
        """Resolve a run-pinned branch; never reinterpret five-method artifacts."""
        algorithm = params.get("algorithmVersion")
        if ((mode in REFINEMENT_RUN_MODES and algorithm in {
                RUN_ALGORITHM_VERSIONS[mode], PRE_FIXED_GATE_REFINEMENT_ALGORITHMS[mode]})
                or (mode in WEIGHT_REFINEMENT_RUN_MODES and algorithm == PRE_ISOLATION_WEIGHT_REFINEMENT_ALGORITHMS[mode])):
            expected = REFINEMENT_EMBEDDING_METHODS
        elif mode in WEIGHT_REFINEMENT_RUN_MODES and algorithm == LEGACY_WEIGHT_REFINEMENT_ALGORITHMS[mode]:
            expected = EMBEDDING_BASELINE_METHODS
        else:
            raise ApiError(HTTPStatus.CONFLICT, "Legacy refinement runs do not provide the current PCP snapshot")
        if tuple(params.get("embeddingMethods") or ()) != expected:
            raise RuntimeError("Refinement embedding order does not match its algorithm version")
        return expected

    def _refinement_normalization_scope(
        self, task_id: str, version: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Derive immutable fitting membership without rewriting frozen masks."""
        import numpy as np

        bundle = self.bundle(task_id)
        split = self._vqa_validation_split(task_id, "joint", version)
        if split.audit.get("protocol") != VQA_VALIDATION_PROTOCOL or (
            version is not None and split.audit.get("frozenVersion") != version
        ):
            raise RuntimeError("Refinement normalization Val contract mismatch")
        mask = np.frombuffer(bundle.development_mask, dtype=np.uint8).astype(bool)
        val_rows = np.asarray(split.validation_indices, dtype=np.int64)
        if (val_rows.ndim != 1 or not val_rows.size or np.any(val_rows < 0)
                or np.any(val_rows >= mask.size) or np.unique(val_rows).size != val_rows.size):
            raise RuntimeError("Refinement normalization Val rows are invalid")
        mask[val_rows] = False
        mask[np.frombuffer(bundle.test_mask, dtype=np.uint8).astype(bool)] = False
        if bundle.query_indices:
            mask[list(bundle.query_indices)] = False
        fit_rows = np.flatnonzero(mask)
        if not fit_rows.size:
            raise RuntimeError("No Development images remain outside fixed Val for normalization")

        def digest(values: Any) -> str:
            return hashlib.sha256(json_dumps(values).encode("utf-8")).hexdigest()

        audit = {
            "protocol": REFINEMENT_NORMALIZATION_POLICY,
            "taskId": task_id,
            "vqaValidationVersion": str(split.audit["frozenVersion"]),
            "vqaValidationManifestSha256": str(split.audit["frozenManifestSha256"]),
            "validationImageIdsSha256": digest([bundle.image_ids[int(row)] for row in val_rows]),
            "sourceImageIdsSha256": digest(list(bundle.image_ids)),
            "fitRowsSha256": digest([int(row) for row in fit_rows]),
            "fitImageIdsSha256": digest([bundle.image_ids[int(row)] for row in fit_rows]),
            "fitCount": int(fit_rows.size),
            "validationCount": int(val_rows.size),
        }
        return fit_rows, audit

    def _refinement_base_for_run(
        self, task_id: str, mode: str, params: Mapping[str, Any],
    ) -> RefinementBaseContext:
        methods = self._refinement_embedding_methods(mode, params)
        isolated = params.get("algorithmVersion") in {
            RUN_ALGORITHM_VERSIONS[mode], PRE_FIXED_GATE_REFINEMENT_ALGORITHMS[mode],
        }
        version = params.get("vqaValidationVersion") if isolated else None
        if isolated and (not isinstance(version, str) or not version):
            raise RuntimeError("Refinement run is missing its pinned normalization Val")
        initial_identity = params.get("initialBaselineIdentity")
        if initial_identity is not None:
            from unified_initial_baseline import load_task_baseline

            loaded = load_task_baseline(self, task_id, initial_identity["publicationVersion"],
                                        fingerprint=params.get("baseStateFingerprint"))
            if loaded is None or loaded[0].initial_baseline_identity != initial_identity:
                raise RuntimeError("Pinned isolated initial baseline is unavailable or changed")
            base = loaded[0]
        else:
            # A historical run must never resolve through a newer active phi0.
            base = self._refinement_base_context(
                task_id, methods, isolate_validation=isolated, vqa_validation_version=version,
                prefer_published=False,
            )
        if isolated and (
            not isinstance(base.normalization_audit, Mapping)
            or params.get("normalizationPolicy") != REFINEMENT_NORMALIZATION_POLICY
            or params.get("normalizationContract") != base.normalization_audit
            or params.get("vqaValidationManifestSha256") != base.normalization_audit["vqaValidationManifestSha256"]
        ):
            raise RuntimeError("Refinement normalization snapshot changed after queuing")
        return base

    def _refinement_base_context(
        self, task_id: str, embedding_methods: tuple[str, ...] = REFINEMENT_EMBEDDING_METHODS,
        *, isolate_validation: bool = True, vqa_validation_version: str | None = None,
        prefer_published: bool = True,
    ) -> RefinementBaseContext:
        """Build the immutable score-level base shared by all four modes.

        The expensive source tensors are opened only once per task.  All
        new normalization ranges use Development minus the fixed Val. Historical
        runs explicitly retain their static Development normalization. Hashing
        the membership contract and resulting
        arrays gives every run a directly comparable base-state fingerprint.
        """

        embedding_methods = tuple(embedding_methods)
        if embedding_methods not in (REFINEMENT_EMBEDDING_METHODS, EMBEDDING_BASELINE_METHODS):
            raise RuntimeError("Unsupported refinement embedding methods or order")
        if prefer_published and isolate_validation and embedding_methods == REFINEMENT_EMBEDDING_METHODS:
            from unified_initial_baseline import load_task_baseline

            loaded = load_task_baseline(self, task_id)
            if loaded is not None:
                published = loaded[0]
                if vqa_validation_version is not None and published.normalization_audit["vqaValidationVersion"] != vqa_validation_version:
                    raise RuntimeError("Published initial baseline does not match the requested Val")
                return published
        normalization_audit = None
        fitting_indices = None
        if isolate_validation:
            fitting_indices, normalization_audit = self._refinement_normalization_scope(
                task_id, vqa_validation_version,
            )
        normalization_key = json_dumps(normalization_audit) if normalization_audit else "legacy-static-development"
        cache_key = (task_id, embedding_methods, normalization_key)
        with self._refinement_context_guard:
            cached = self._refinement_contexts.get(cache_key)
            if cached is not None:
                return cached

            import numpy as np

            task = self.task(task_id)
            bundle = self.bundle(task_id)
            source = self._weighted_fusion_context(task_id)
            attribute_ids = tuple(
                str(target.get("id"))
                for target in task.manifest.get("retrievalTargets", [])
                if target.get("kind") == "attribute"
            )
            source_attribute_ids = tuple(
                target_id
                for target_id, attribute in zip(
                    task.target_ids,
                    source.target_attributes,
                    strict=True,
                )
                if attribute is not None
            )
            source_attribute_names = tuple(
                attribute
                for attribute in source.target_attributes
                if attribute is not None
            )
            if (
                not attribute_ids
                or attribute_ids != source_attribute_ids
                or source_attribute_names != source.attributes
            ):
                raise RuntimeError(
                    "Refinement attribute ID/name order does not match the "
                    "audited probe source"
                )
            development_mask = np.frombuffer(
                bundle.development_mask, dtype=np.uint8
            ).astype(bool)
            development_indices = np.flatnonzero(development_mask) if fitting_indices is None else fitting_indices
            if development_indices.size == 0:
                raise RuntimeError("Refinement requires a non-empty Development split")

            row_count = task.row_count
            attribute_count = len(source.attributes)
            learner_count = len(source.learned_method_ids)
            raw_probe = np.empty(
                (row_count, attribute_count, learner_count), dtype=np.float32
            )
            for attribute_index, attribute in enumerate(source.attributes):
                for learner_index, method in enumerate(source.learned_method_ids):
                    seed_positions = [
                        source.seeds_by_method[method].index(seed)
                        for seed in source.selected_seeds
                    ]
                    seed_scores = np.asarray(
                        source.tensors[method][attribute][seed_positions, :],
                        dtype=np.float64,
                    )
                    raw_probe[:, attribute_index, learner_index] = np.asarray(
                        seed_scores.mean(axis=0), dtype=np.float32
                    )

            probe_min = np.min(raw_probe[development_indices], axis=0).astype(
                np.float32
            )
            probe_max = np.max(raw_probe[development_indices], axis=0).astype(
                np.float32
            )
            probe_span = np.asarray(probe_max - probe_min, dtype=np.float64)
            probe_features = np.divide(
                np.asarray(raw_probe, dtype=np.float64) - probe_min[None, :, :],
                probe_span[None, :, :],
                out=np.zeros_like(raw_probe, dtype=np.float64),
                where=probe_span[None, :, :] > 0.0,
            )
            probe_features = np.clip(probe_features, 0.0, 1.0).astype(np.float32)

            result_path = (
                source.ours_full_root
                / "tasks"
                / source.pcp.task_key(source.task_config)
                / "front_minmax_result.json"
            )
            fit_payload = source.pcp.load_json(result_path)
            fits = fit_payload.get("audit", {}).get("fits", {})
            saved_theta = np.empty(
                (len(source.selected_seeds), attribute_count), dtype=np.float64
            )
            saved_temperature = np.empty_like(saved_theta)
            for seed_position, seed in enumerate(source.selected_seeds):
                model = fits.get(str(seed), {}).get(source.pcp.OURS_FULL_ID)
                if not isinstance(model, Mapping):
                    raise RuntimeError(
                        f"Ours-Full calibration is missing for seed {seed}"
                    )
                for attribute_index, attribute in enumerate(source.attributes):
                    saved_theta[seed_position, attribute_index] = float(
                        model["theta_by_attr"][attribute]
                    )
                    saved_temperature[seed_position, attribute_index] = float(
                        model["temperature_by_attr"][attribute]
                    )
            temperature = np.asarray(
                saved_temperature.mean(axis=0), dtype=np.float32
            )
            initial_theta = np.asarray(saved_theta.mean(axis=0), dtype=np.float32)
            if (
                not np.all(np.isfinite(temperature))
                or np.any(temperature <= 0.0)
                or not np.all(np.isfinite(initial_theta))
            ):
                raise RuntimeError("Refinement SoftGate calibration is invalid")

            joint_index = task.target_ids.index("joint")
            embedding_raw = np.column_stack(
                [
                    self._exported_method_column(
                        task, "rawScores", method, joint_index
                    )
                    for method in embedding_methods
                ]
            ).astype(np.float32, copy=False)
            embedding_min = np.min(
                embedding_raw[development_indices], axis=0
            ).astype(np.float32)
            embedding_max = np.max(
                embedding_raw[development_indices], axis=0
            ).astype(np.float32)
            embedding_span = np.asarray(
                embedding_max - embedding_min, dtype=np.float64
            )
            embedding_features = np.divide(
                np.asarray(embedding_raw, dtype=np.float64)
                - embedding_min[None, :],
                embedding_span[None, :],
                out=np.zeros_like(embedding_raw, dtype=np.float64),
                where=embedding_span[None, :] > 0.0,
            )
            embedding_features = np.clip(
                embedding_features, 0.0, 1.0
            ).astype(np.float32)

            hasher = hashlib.sha256()
            hasher.update(b"pcp-conjunction-holistic-base-v2\0" if isolate_validation else b"pcp-conjunction-holistic-base-v1\0")
            if normalization_audit is not None:
                hasher.update(normalization_key.encode("utf-8"))
            hasher.update(task_id.encode("utf-8"))
            for values in (
                attribute_ids,
                source.attributes,
                source.learned_method_labels,
                embedding_methods,
            ):
                hasher.update(json_dumps(list(values)).encode("utf-8"))
            for array in (
                probe_features,
                probe_min,
                probe_max,
                temperature,
                initial_theta,
                embedding_features,
                embedding_min,
                embedding_max,
            ):
                contiguous = np.ascontiguousarray(array, dtype="<f4")
                hasher.update(str(contiguous.shape).encode("ascii"))
                hasher.update(contiguous.tobytes(order="C"))
            base = RefinementBaseContext(
                attribute_ids=attribute_ids,
                attribute_names=tuple(source.attributes),
                learner_names=tuple(source.learned_method_labels),
                embedding_names=embedding_methods,
                probe_features=probe_features,
                probe_min=probe_min,
                probe_max=probe_max,
                temperature=temperature,
                initial_theta=initial_theta,
                embedding_features=embedding_features,
                embedding_min=embedding_min,
                embedding_max=embedding_max,
                development_indices=development_indices,
                base_state_fingerprint=hasher.hexdigest(),
                normalization_audit=normalization_audit,
                raw_probe_probabilities=raw_probe,
            )
            # One task is already tens of MB.  Retain only the most recently
            # used base state while the lower-level score loader keeps its own
            # small source cache.
            self._refinement_contexts.clear()
            self._refinement_contexts[cache_key] = base
            return base

    def _fusion_context_for_algorithm(
        self,
        task_id: str,
        algorithm_version: str,
    ) -> WeightedFusionContext | None:
        """Keep rank-only and unified jobs off the legacy dispatch path."""

        if algorithm_version == FUSION_WEIGHT_ALGORITHM_V4 or algorithm_version in {
            *(RUN_ALGORITHM_VERSIONS[mode] for mode in REFINEMENT_RUN_MODES),
            *LEGACY_WEIGHT_REFINEMENT_ALGORITHMS.values(),
            *PRE_ISOLATION_WEIGHT_REFINEMENT_ALGORITHMS.values(),
            *PRE_FIXED_GATE_REFINEMENT_ALGORITHMS.values(),
        }:
            return None
        return self._weighted_fusion_context(task_id)

    def _exported_ours_full_arrays(self, task: TaskSpec) -> tuple[Any, Any, Any]:
        """Read the immutable Ours-Full browser columns for an exact equal-weight anchor."""
        import numpy as np

        try:
            method_index = task.methods.index("Ours-Full")
        except ValueError as error:
            raise RuntimeError(f"Ours-Full is unavailable for {task.task_id}") from error
        shape = (task.row_count, len(task.methods), len(task.target_ids))

        def column(file_key: str) -> Any:
            file_spec = task.manifest["files"][file_key]
            values = np.memmap(
                self._bundle_file(task, file_spec["path"]),
                dtype="<f4",
                mode="r",
                shape=shape,
            )
            return np.asarray(values[:, method_index, :], dtype=np.float32)

        return column("rawScores"), column("calibratedScores"), column("ranks")

    def _exported_method_column(
        self,
        task: TaskSpec,
        file_key: str,
        method: str,
        target_index: int,
    ) -> Any:
        """Read one immutable browser tensor column without copying other methods."""
        import numpy as np

        method_index = task.methods.index(method)
        file_spec = task.manifest["files"][file_key]
        values = np.memmap(
            self._bundle_file(task, file_spec["path"]),
            dtype="<f4",
            mode="r",
            shape=(task.row_count, len(task.methods), len(task.target_ids)),
        )
        return np.asarray(values[:, method_index, target_index], dtype=np.float32)

    def _learner_calibrated_features(self, task: TaskSpec, target_index: int) -> Any:
        """Return the eight frozen learner calibrated columns in audited order."""
        import numpy as np

        missing = [method for method in WEIGHTED_FUSION_LEARNERS if method not in task.methods]
        if missing:
            raise RuntimeError(f"Tuning learner columns are unavailable: {missing}")
        return np.column_stack(
            [
                self._exported_method_column(task, "calibratedScores", method, target_index)
                for method in WEIGHTED_FUSION_LEARNERS
            ]
        ).astype(np.float32, copy=False)

    @staticmethod
    def _merge_tuning_supervision(
        original_indices: Any,
        original_labels: Any,
        feedback_indices: Any,
        feedback_labels: Any,
        *,
        feedback_weight: float,
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Merge an audited original fit partition with user overrides."""
        import numpy as np

        original_rows = np.asarray(original_indices, dtype=np.int64).reshape(-1)
        original_truth = np.asarray(original_labels, dtype=np.uint8).reshape(-1)
        feedback_rows = np.asarray(feedback_indices, dtype=np.int64).reshape(-1)
        signed_feedback = np.asarray(feedback_labels, dtype=np.int8).reshape(-1)
        if original_rows.shape != original_truth.shape:
            raise RuntimeError("Original supervision rows and labels are not aligned")
        if feedback_rows.shape != signed_feedback.shape:
            raise RuntimeError("Feedback rows and labels are not aligned")
        original_by_row = {
            int(row): int(label)
            for row, label in zip(original_rows, original_truth, strict=True)
        }
        feedback_by_row = {
            int(row): int(label)
            for row, label in zip(feedback_rows, signed_feedback, strict=True)
            if int(label) != 0
        }
        merged: dict[int, tuple[int, float, str]] = {
            int(row): (int(label), 1.0, "original-fit")
            for row, label in zip(original_rows, original_truth, strict=True)
        }
        for row, label in feedback_by_row.items():
            merged[row] = (
                int(label > 0),
                float(feedback_weight) * abs(label),
                "user-feedback",
            )
        new_feedback_count = sum(row not in original_by_row for row in feedback_by_row)
        override_count = sum(
            row in original_by_row and int(label > 0) != original_by_row[row]
            for row, label in feedback_by_row.items()
        )
        reinforce_count = sum(
            row in original_by_row and int(label > 0) == original_by_row[row]
            for row, label in feedback_by_row.items()
        )
        ordered = sorted(merged)
        rows = np.asarray(ordered, dtype=np.int64)
        truth = np.asarray([merged[row][0] for row in ordered], dtype=np.uint8)
        weights = np.asarray([merged[row][1] for row in ordered], dtype=np.float64)
        positive = int(np.count_nonzero(truth))
        negative = int(truth.size - positive)
        if positive < 2 or negative < 2:
            raise RuntimeError(
                "Combined original fit supervision and feedback requires "
                "at least two positive and two negative rows"
            )
        return rows, truth, weights, {
            "originalDevelopmentCount": int(original_rows.size),
            "feedbackCount": int(len(feedback_by_row)),
            "newFeedbackCount": int(new_feedback_count),
            "feedbackExistingCount": int(override_count + reinforce_count),
            "feedbackOverrideCount": int(override_count),
            "feedbackReinforceCount": int(reinforce_count),
            "feedbackReviewOnlyCount": int(np.count_nonzero(signed_feedback == 0)),
            "combinedCount": int(rows.size),
            "combinedPositiveCount": positive,
            "combinedNegativeCount": negative,
            "feedbackWeight": float(feedback_weight),
        }

    def _compute_weighted_fusion(
        self,
        task_id: str,
        normalized_weights: tuple[float, ...],
    ) -> WeightedFusionResult:
        """Compute a complete virtual method cube for one normalized weight vector."""
        import numpy as np

        task = self.task(task_id)
        if len(normalized_weights) != len(WEIGHTED_FUSION_LEARNERS):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Weighted fusion requires eight weights")
        equal_weights = max(normalized_weights) - min(normalized_weights) <= 1e-15
        if equal_weights:
            raw, calibrated, ranks = self._exported_ours_full_arrays(task)
            baseline_error: float | None = 0.0
        else:
            context = self._weighted_fusion_context(task_id)
            member_weights = {
                method: normalized_weights[index]
                for index, method in enumerate(context.learned_method_ids)
            }
            row_count = task.row_count
            target_count = len(task.target_ids)
            raw = np.empty((row_count, target_count), dtype=np.float32)
            calibrated = np.empty_like(raw)
            ranks = np.empty_like(raw)
            for target_index, target_id in enumerate(task.target_ids):
                target_attribute = context.target_attributes[target_index]
                seed_scores = context.pcp.reconstruct_ours_full_scores(
                    context.tensors,
                    context.seeds_by_method,
                    list(context.learned_method_ids),
                    list(context.attributes),
                    target_attribute,
                    list(context.selected_seeds),
                    context.gallery_indices,
                    context.task_config,
                    context.ours_full_root,
                    expected_stage=context.probe_stage,
                    member_weights=member_weights,
                )
                deployed = np.asarray(seed_scores.mean(axis=0), dtype=np.float64)
                raw[:, target_index] = deployed
                train_values = deployed[context.train_indices]
                low = float(np.min(train_values))
                high = float(np.max(train_values))
                if high <= low:
                    calibrated[:, target_index] = 0.0
                else:
                    calibrated[:, target_index] = np.clip(
                        (deployed - low) / (high - low),
                        0.0,
                        1.0,
                    )
                ranks[:, target_index] = context.pcp.aggregate_seed_ranks(
                    seed_scores,
                    "mean-rank",
                    "average",
                )
            baseline_error = None

        packed = np.concatenate(
            [raw.reshape(-1), calibrated.reshape(-1), ranks.reshape(-1)]
        ).astype("<f4", copy=False)
        return WeightedFusionResult(
            payload=packed.tobytes(order="C"),
            row_count=task.row_count,
            target_count=len(task.target_ids),
            equal_weights=equal_weights,
            baseline_max_abs_error=baseline_error,
            normalized_weights=normalized_weights,
        )

    def weighted_fusion(
        self,
        task_id: str,
        normalized_weights: tuple[float, ...],
    ) -> WeightedFusionResult:
        """Serialize cache misses so concurrent users never duplicate heavy work."""
        key = (task_id, normalized_weights)
        with self._weighted_fusion_result_guard:
            cached = self._weighted_fusion_results.get(key)
            if cached is not None:
                return cached
            result = self._compute_weighted_fusion(task_id, normalized_weights)
            if len(self._weighted_fusion_results) >= 24:
                oldest = next(iter(self._weighted_fusion_results))
                del self._weighted_fusion_results[oldest]
            self._weighted_fusion_results[key] = result
            return result

    def _compute_hierarchical_fusion(
        self,
        task_id: str,
        normalized_attribute_weights: tuple[float, ...],
        normalized_learner_weights: tuple[tuple[float, ...], ...],
        normalized_global_weights: tuple[float, ...],
    ) -> HierarchicalFusionResult:
        """Compute the inner Ours hierarchy and global Joint rank fusion."""
        import numpy as np

        task = self.task(task_id)
        attribute_count = sum(
            1
            for target in task.manifest.get("retrievalTargets", [])
            if target.get("kind") == "attribute"
        )
        if (
            attribute_count <= 0
            or len(normalized_attribute_weights) != attribute_count
            or len(normalized_learner_weights) != attribute_count
            or any(
                len(weights) != len(WEIGHTED_FUSION_LEARNERS)
                for weights in normalized_learner_weights
            )
            or len(normalized_global_weights) != len(GLOBAL_FUSION_METHODS)
            or any(
                not math.isfinite(weight) or weight < 0.0
                for weight in normalized_global_weights
            )
            or abs(math.fsum(normalized_global_weights) - 1.0) > 1e-12
        ):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "Hierarchical fusion weights do not match the task dimensions",
            )

        inner_baseline = (
            max(normalized_attribute_weights) - min(normalized_attribute_weights) <= 1e-15
            and all(max(weights) - min(weights) <= 1e-15 for weights in normalized_learner_weights)
        )
        if inner_baseline:
            hierarchy_raw, hierarchy_calibrated, hierarchy_ranks = (
                self._exported_ours_full_arrays(task)
            )
        else:
            context = self._weighted_fusion_context(task_id)
            if len(context.attributes) != attribute_count:
                raise RuntimeError(
                    f"Hierarchical source attributes do not match {task_id}"
                )
            member_weights_by_attribute = {
                attribute: {
                    method: normalized_learner_weights[attribute_index][method_index]
                    for method_index, method in enumerate(context.learned_method_ids)
                }
                for attribute_index, attribute in enumerate(context.attributes)
            }
            attribute_weights = {
                attribute: normalized_attribute_weights[attribute_index]
                for attribute_index, attribute in enumerate(context.attributes)
            }
            row_count = task.row_count
            target_count = len(task.target_ids)
            hierarchy_raw = np.empty((row_count, target_count), dtype=np.float32)
            hierarchy_calibrated = np.empty_like(hierarchy_raw)
            hierarchy_ranks = np.empty_like(hierarchy_raw)
            for target_index, target_attribute in enumerate(context.target_attributes):
                seed_scores = context.pcp.reconstruct_ours_full_scores(
                    context.tensors,
                    context.seeds_by_method,
                    list(context.learned_method_ids),
                    list(context.attributes),
                    target_attribute,
                    list(context.selected_seeds),
                    context.gallery_indices,
                    context.task_config,
                    context.ours_full_root,
                    expected_stage=context.probe_stage,
                    member_weights_by_attribute=member_weights_by_attribute,
                    attribute_weights=attribute_weights,
                )
                deployed = np.asarray(seed_scores.mean(axis=0), dtype=np.float64)
                hierarchy_raw[:, target_index] = deployed
                train_values = deployed[context.train_indices]
                low = float(np.min(train_values))
                high = float(np.max(train_values))
                if high <= low:
                    hierarchy_calibrated[:, target_index] = 0.0
                else:
                    hierarchy_calibrated[:, target_index] = np.clip(
                        (deployed - low) / (high - low),
                        0.0,
                        1.0,
                    )
                hierarchy_ranks[:, target_index] = context.pcp.aggregate_seed_ranks(
                    seed_scores,
                    "mean-rank",
                    "average",
                )

        global_ours_only = (
            abs(normalized_global_weights[0] - 1.0) <= 1e-15
            and all(abs(weight) <= 1e-15 for weight in normalized_global_weights[1:])
        )
        if global_ours_only:
            raw = hierarchy_raw
            calibrated = hierarchy_calibrated
            ranks = hierarchy_ranks
        else:
            if "joint" not in task.target_ids:
                raise RuntimeError(f"Canonical Joint target is unavailable for {task_id}")
            joint_index = task.target_ids.index("joint")
            global_sources = [
                np.asarray(hierarchy_ranks[:, joint_index], dtype=np.float64)
            ]
            global_sources.extend(
                np.asarray(
                    self._exported_method_column(
                        task,
                        "ranks",
                        method,
                        joint_index,
                    ),
                    dtype=np.float64,
                )
                for method in EMBEDDING_BASELINE_METHODS
            )
            borda = np.zeros(task.row_count, dtype=np.float64)
            for weight, source in zip(
                normalized_global_weights,
                global_sources,
                strict=True,
            ):
                borda += weight * source

            raw = np.array(hierarchy_raw, dtype=np.float32, copy=True)
            calibrated = np.array(hierarchy_calibrated, dtype=np.float32, copy=True)
            ranks = np.array(hierarchy_ranks, dtype=np.float32, copy=True)
            raw[:, joint_index] = borda
            train_values = borda[self._frozen_train_indices(task_id)]
            low = float(np.min(train_values))
            high = float(np.max(train_values))
            if high <= low:
                calibrated[:, joint_index] = 0.0
            else:
                calibrated[:, joint_index] = np.clip(
                    (borda - low) / (high - low),
                    0.0,
                    1.0,
                )
            ranks[:, joint_index] = normalized_ranks(borda)

        baseline = inner_baseline and global_ours_only
        baseline_error: float | None = 0.0 if baseline else None

        packed = np.concatenate(
            [
                raw.reshape(-1),
                calibrated.reshape(-1),
                ranks.reshape(-1),
                hierarchy_calibrated.reshape(-1),
                hierarchy_ranks.reshape(-1),
            ]
        ).astype("<f4", copy=False)
        return HierarchicalFusionResult(
            payload=packed.tobytes(order="C"),
            row_count=task.row_count,
            target_count=len(task.target_ids),
            baseline=baseline,
            baseline_max_abs_error=baseline_error,
            normalized_attribute_weights=normalized_attribute_weights,
            normalized_learner_weights=normalized_learner_weights,
            normalized_global_weights=normalized_global_weights,
        )

    def hierarchical_fusion(
        self,
        task_id: str,
        normalized_attribute_weights: tuple[float, ...],
        normalized_learner_weights: tuple[tuple[float, ...], ...],
        normalized_global_weights: tuple[float, ...],
    ) -> HierarchicalFusionResult:
        """Serialize heavy hierarchy recomputations and cache recent configurations."""
        key = (
            task_id,
            normalized_attribute_weights,
            normalized_learner_weights,
            normalized_global_weights,
        )
        with self._hierarchical_fusion_result_guard:
            cached = self._hierarchical_fusion_results.get(key)
            if cached is not None:
                return cached
            result = self._compute_hierarchical_fusion(
                task_id,
                normalized_attribute_weights,
                normalized_learner_weights,
                normalized_global_weights,
            )
            if len(self._hierarchical_fusion_results) >= 24:
                oldest = next(iter(self._hierarchical_fusion_results))
                del self._hierarchical_fusion_results[oldest]
            self._hierarchical_fusion_results[key] = result
            return result

    def _compute_dynamic_pcp_clusters(
        self,
        task_id: str,
        normalized_attribute_weights: tuple[float, ...],
        normalized_learner_weights: tuple[tuple[float, ...], ...],
        normalized_global_weights: tuple[float, ...],
        scheme: str,
    ) -> DynamicPcpClusterResult:
        """Recluster the complete hierarchy after one weight configuration."""
        import numpy as np

        if scheme not in DYNAMIC_PCP_CLUSTER_SCHEMES:
            raise ValueError(
                "scheme must be one of " + ", ".join(DYNAMIC_PCP_CLUSTER_SCHEMES)
            )
        task = self.task(task_id)
        targets = tuple(task.manifest.get("retrievalTargets", []))
        attribute_ids = tuple(
            str(target.get("id"))
            for target in targets
            if target.get("kind") == "attribute"
        )
        if "joint" not in task.target_ids:
            raise RuntimeError(f"PCP clustering has no canonical Joint target for {task_id}")
        overall_target_id = "joint"

        hierarchy = self.hierarchical_fusion(
            task_id,
            normalized_attribute_weights,
            normalized_learner_weights,
            normalized_global_weights,
        )
        cube_size = hierarchy.row_count * hierarchy.target_count
        packed = np.frombuffer(hierarchy.payload, dtype="<f4")
        if packed.size != 5 * cube_size:
            raise RuntimeError(f"Hierarchical fusion payload is malformed for {task_id}")
        final_ranks = packed[2 * cube_size : 3 * cube_size].reshape(
            hierarchy.row_count,
            hierarchy.target_count,
        )
        hierarchical_ranks = packed[4 * cube_size : 5 * cube_size].reshape(
            hierarchy.row_count,
            hierarchy.target_count,
        )

        ranks_spec = task.manifest["files"]["ranks"]
        learner_ranks = np.memmap(
            self._bundle_file(task, ranks_spec["path"]),
            dtype="<f4",
            mode="r",
            shape=(task.row_count, len(task.methods), len(task.target_ids)),
        )
        features = build_full_hierarchical_rank_features(
            final_ranks,
            hierarchical_ranks,
            learner_ranks,
            target_ids=task.target_ids,
            method_labels=task.methods,
            attribute_ids=attribute_ids,
            overall_target_id=overall_target_id,
        )
        expected_feature_count = (
            2
            + len(EMBEDDING_BASELINE_METHODS)
            + len(attribute_ids) * (1 + len(WEIGHTED_FUSION_LEARNERS))
        )
        if features.shape != (task.row_count, expected_feature_count):
            raise RuntimeError(f"Dynamic PCP feature contract failed for {task_id}")
        labels, cluster_count, fit_row_count = fit_dynamic_pcp_cluster_labels(
            features,
            self.bundle(task_id).development_mask,
            scheme,
        )
        payload = labels.tobytes(order="C")
        if len(payload) != task.row_count:
            raise RuntimeError(f"Dynamic PCP label payload is malformed for {task_id}")
        return DynamicPcpClusterResult(
            payload=payload,
            row_count=task.row_count,
            cluster_count=cluster_count,
            scheme=scheme,
            feature_count=expected_feature_count,
            fit_row_count=fit_row_count,
        )

    def pcp_clusters(
        self,
        task_id: str,
        normalized_attribute_weights: tuple[float, ...],
        normalized_learner_weights: tuple[tuple[float, ...], ...],
        normalized_global_weights: tuple[float, ...],
        scheme: str,
    ) -> DynamicPcpClusterResult:
        """Cache by normalized hierarchy and single-flight identical requests."""
        if scheme not in DYNAMIC_PCP_CLUSTER_SCHEMES:
            raise ValueError(
                "scheme must be one of " + ", ".join(DYNAMIC_PCP_CLUSTER_SCHEMES)
            )
        key = (
            task_id,
            normalized_attribute_weights,
            normalized_learner_weights,
            normalized_global_weights,
            scheme,
        )
        while True:
            with self._pcp_cluster_result_guard:
                cached = self._pcp_cluster_results.pop(key, None)
                if cached is not None:
                    self._pcp_cluster_results[key] = cached
                    return cached
                pending = self._pcp_cluster_inflight.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._pcp_cluster_inflight[key] = pending
                    owner = True
                else:
                    owner = False
            if owner:
                break
            pending.wait()

        try:
            result = self._compute_dynamic_pcp_clusters(
                task_id,
                normalized_attribute_weights,
                normalized_learner_weights,
                normalized_global_weights,
                scheme,
            )
        except BaseException:
            with self._pcp_cluster_result_guard:
                self._pcp_cluster_inflight.pop(key, None)
                pending.set()
            raise

        with self._pcp_cluster_result_guard:
            if len(self._pcp_cluster_results) >= DYNAMIC_PCP_CLUSTER_CACHE_SIZE:
                oldest = next(iter(self._pcp_cluster_results))
                del self._pcp_cluster_results[oldest]
            self._pcp_cluster_results[key] = result
            self._pcp_cluster_inflight.pop(key, None)
            pending.set()
        return result

    @staticmethod
    def _rank_fusion_attribute_ids(task: TaskSpec) -> tuple[str, ...]:
        attributes = tuple(
            str(target.get("id"))
            for target in task.manifest.get("retrievalTargets", [])
            if target.get("kind") == "attribute"
        )
        if not attributes or len(set(attributes)) != len(attributes):
            raise RuntimeError(f"Rank fusion has invalid attributes for {task.task_id}")
        if "joint" not in task.target_ids:
            raise RuntimeError(
                f"Canonical Joint target is unavailable for {task.task_id}"
            )
        if set(task.target_ids) != set(attributes) | {"joint"}:
            raise RuntimeError(
                f"Rank fusion only supports attribute and canonical Joint targets for {task.task_id}"
            )
        missing = [method for method in RANK_FUSION_METHODS if method not in task.methods]
        if missing:
            raise RuntimeError(f"Rank-fusion method columns are unavailable: {missing}")
        return attributes

    def _rank_fusion_source_features(
        self,
        task: TaskSpec,
        row_indices: Any | None = None,
    ) -> tuple[tuple[str, ...], Any]:
        """Return [row, attribute, 13 method] frozen rank features."""

        import numpy as np

        attributes = self._rank_fusion_attribute_ids(task)
        rank_spec = task.manifest["files"]["ranks"]
        exported = np.memmap(
            self._bundle_file(task, rank_spec["path"]),
            dtype="<f4",
            mode="r",
            shape=(task.row_count, len(task.methods), len(task.target_ids)),
        )
        selected = exported if row_indices is None else exported[np.asarray(row_indices)]
        row_count = int(selected.shape[0])
        features = np.empty(
            (row_count, len(attributes), len(RANK_FUSION_METHODS)),
            dtype=np.float32,
        )
        joint_index = task.target_ids.index("joint")
        for method_index, method in enumerate(EMBEDDING_BASELINE_METHODS):
            values = selected[:, task.methods.index(method), joint_index]
            features[:, :, method_index] = values[:, None]
        learner_offset = len(EMBEDDING_BASELINE_METHODS)
        for attribute_index, attribute_id in enumerate(attributes):
            target_index = task.target_ids.index(attribute_id)
            for learner_index, method in enumerate(WEIGHTED_FUSION_LEARNERS):
                features[:, attribute_index, learner_offset + learner_index] = selected[
                    :, task.methods.index(method), target_index
                ]
        if not np.isfinite(features).all():
            raise RuntimeError(f"Rank-fusion sources contain non-finite values for {task.task_id}")
        return attributes, features

    def _compute_rank_fusion(
        self,
        task_id: str,
        normalized_attribute_weights: tuple[float, ...],
        normalized_method_weights_by_attribute: tuple[tuple[float, ...], ...],
    ) -> RankFusionResult:
        """Compute the two-level attribute/13-method Borda hierarchy."""

        import numpy as np

        task = self.task(task_id)
        attributes, features = self._rank_fusion_source_features(task)
        if (
            len(normalized_attribute_weights) != len(attributes)
            or len(normalized_method_weights_by_attribute) != len(attributes)
            or any(
                len(row) != len(RANK_FUSION_METHODS)
                for row in normalized_method_weights_by_attribute
            )
            or any(
                not math.isfinite(value) or value < 0.0
                for value in normalized_attribute_weights
            )
            or any(
                any(not math.isfinite(value) or value < 0.0 for value in row)
                for row in normalized_method_weights_by_attribute
            )
        ):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "Rank-fusion weights do not match the task dimensions",
            )

        # Optimizer artifacts intentionally store compact float32 arrays.  A
        # mathematically normalized 13-member simplex therefore commonly
        # round-trips with a sum such as 1.000000037.  Normalize again at this
        # computation boundary instead of treating harmless representation
        # error as a task-dimension mismatch.
        attribute_total = math.fsum(normalized_attribute_weights)
        method_totals = tuple(
            math.fsum(row) for row in normalized_method_weights_by_attribute
        )
        if attribute_total <= 0.0 or any(total <= 0.0 for total in method_totals):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "Rank-fusion weights must contain a positive value in every level",
            )
        normalized_attribute_weights = tuple(
            value / attribute_total for value in normalized_attribute_weights
        )
        normalized_method_weights_by_attribute = tuple(
            tuple(value / method_totals[index] for value in row)
            for index, row in enumerate(normalized_method_weights_by_attribute)
        )

        method_weights = np.asarray(
            normalized_method_weights_by_attribute,
            dtype=np.float64,
        )
        inner_borda = np.einsum(
            "ram,am->ra",
            np.asarray(features, dtype=np.float64),
            method_weights,
            optimize=True,
        )
        raw_values = np.empty(
            (task.row_count, len(task.target_ids)),
            dtype=np.float64,
        )
        for attribute_index, attribute_id in enumerate(attributes):
            raw_values[:, task.target_ids.index(attribute_id)] = inner_borda[
                :, attribute_index
            ]
        raw_values[:, task.target_ids.index("joint")] = inner_borda @ np.asarray(
            normalized_attribute_weights,
            dtype=np.float64,
        )

        raw = np.asarray(raw_values, dtype=np.float32)
        calibrated = np.empty_like(raw)
        ranks = np.empty_like(raw)
        train_indices = self._frozen_train_indices(task_id)
        for target_index in range(len(task.target_ids)):
            values = raw_values[:, target_index]
            train_values = values[train_indices]
            low = float(np.min(train_values))
            high = float(np.max(train_values))
            if high <= low:
                calibrated[:, target_index] = 0.0
            else:
                calibrated[:, target_index] = np.clip(
                    (values - low) / (high - low),
                    0.0,
                    1.0,
                )
            ranks[:, target_index] = normalized_ranks(values)

        packed = np.concatenate(
            [raw.reshape(-1), calibrated.reshape(-1), ranks.reshape(-1)]
        ).astype("<f4", copy=False)
        return RankFusionResult(
            payload=packed.tobytes(order="C"),
            row_count=task.row_count,
            target_count=len(task.target_ids),
            normalized_attribute_weights=normalized_attribute_weights,
            normalized_method_weights_by_attribute=(
                normalized_method_weights_by_attribute
            ),
        )

    def rank_fusion(
        self,
        task_id: str,
        normalized_attribute_weights: tuple[float, ...],
        normalized_method_weights_by_attribute: tuple[tuple[float, ...], ...],
    ) -> RankFusionResult:
        key = (
            task_id,
            normalized_attribute_weights,
            normalized_method_weights_by_attribute,
        )
        with self._rank_fusion_result_guard:
            cached = self._rank_fusion_results.get(key)
            if cached is not None:
                return cached
            result = self._compute_rank_fusion(
                task_id,
                normalized_attribute_weights,
                normalized_method_weights_by_attribute,
            )
            if len(self._rank_fusion_results) >= 24:
                oldest = next(iter(self._rank_fusion_results))
                del self._rank_fusion_results[oldest]
            self._rank_fusion_results[key] = result
            return result

    def _compute_rank_fusion_clusters(
        self,
        task_id: str,
        normalized_attribute_weights: tuple[float, ...],
        normalized_method_weights_by_attribute: tuple[tuple[float, ...], ...],
        scheme: str,
    ) -> DynamicPcpClusterResult:
        import numpy as np

        if scheme not in DYNAMIC_PCP_CLUSTER_SCHEMES:
            raise ValueError(
                "scheme must be one of " + ", ".join(DYNAMIC_PCP_CLUSTER_SCHEMES)
            )
        task = self.task(task_id)
        attributes = self._rank_fusion_attribute_ids(task)
        fusion = self.rank_fusion(
            task_id,
            normalized_attribute_weights,
            normalized_method_weights_by_attribute,
        )
        cube_size = fusion.row_count * fusion.target_count
        packed = np.frombuffer(fusion.payload, dtype="<f4")
        if packed.size != 3 * cube_size:
            raise RuntimeError(f"Rank-fusion payload is malformed for {task_id}")
        fused_ranks = packed[2 * cube_size : 3 * cube_size].reshape(
            fusion.row_count,
            fusion.target_count,
        )
        rank_spec = task.manifest["files"]["ranks"]
        exported = np.memmap(
            self._bundle_file(task, rank_spec["path"]),
            dtype="<f4",
            mode="r",
            shape=(task.row_count, len(task.methods), len(task.target_ids)),
        )
        features = build_attribute_rank_fusion_cluster_features(
            fused_ranks,
            exported,
            target_ids=task.target_ids,
            method_labels=task.methods,
            attribute_ids=attributes,
            joint_target_id="joint",
        )
        expected_feature_count = 1 + 14 * len(attributes)
        if features.shape != (task.row_count, expected_feature_count):
            raise RuntimeError(f"Rank-fusion cluster feature contract failed for {task_id}")
        labels, cluster_count, fit_row_count = fit_dynamic_pcp_cluster_labels(
            features,
            self.bundle(task_id).development_mask,
            scheme,
        )
        return DynamicPcpClusterResult(
            payload=labels.tobytes(order="C"),
            row_count=task.row_count,
            cluster_count=cluster_count,
            scheme=scheme,
            feature_count=expected_feature_count,
            fit_row_count=fit_row_count,
        )

    def rank_fusion_clusters(
        self,
        task_id: str,
        normalized_attribute_weights: tuple[float, ...],
        normalized_method_weights_by_attribute: tuple[tuple[float, ...], ...],
        scheme: str,
    ) -> DynamicPcpClusterResult:
        if scheme not in DYNAMIC_PCP_CLUSTER_SCHEMES:
            raise ValueError(
                "scheme must be one of " + ", ".join(DYNAMIC_PCP_CLUSTER_SCHEMES)
            )
        key = (
            task_id,
            normalized_attribute_weights,
            normalized_method_weights_by_attribute,
            scheme,
        )
        while True:
            with self._rank_fusion_cluster_result_guard:
                cached = self._rank_fusion_cluster_results.pop(key, None)
                if cached is not None:
                    self._rank_fusion_cluster_results[key] = cached
                    return cached
                pending = self._rank_fusion_cluster_inflight.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._rank_fusion_cluster_inflight[key] = pending
                    owner = True
                else:
                    owner = False
            if owner:
                break
            pending.wait()

        try:
            result = self._compute_rank_fusion_clusters(
                task_id,
                normalized_attribute_weights,
                normalized_method_weights_by_attribute,
                scheme,
            )
        except BaseException:
            with self._rank_fusion_cluster_result_guard:
                self._rank_fusion_cluster_inflight.pop(key, None)
                pending.set()
            raise
        with self._rank_fusion_cluster_result_guard:
            if len(self._rank_fusion_cluster_results) >= DYNAMIC_PCP_CLUSTER_CACHE_SIZE:
                oldest = next(iter(self._rank_fusion_cluster_results))
                del self._rank_fusion_cluster_results[oldest]
            self._rank_fusion_cluster_results[key] = result
            self._rank_fusion_cluster_inflight.pop(key, None)
            pending.set()
        return result

    def _compute_refinement_visualization(
        self, run: sqlite3.Row
    ) -> RefinementVisualizationResult:
        """Rebuild the exact score-level hierarchy represented by a Tune run.

        The browser intentionally does not approximate this from exported
        method columns: the immutable base uses five-seed probe means and
        Development-fitted normalization.  The component order is
        ``F, C, [g_a, z_a1..z_a8]*A, H, e_1..e_R`` and is independent of which
        disclosure rows happen to be open in the PCP.
        """

        import numpy as np

        mode = str(run["mode"])
        if mode not in REFINEMENT_RUN_MODES or str(run["status"]) != "succeeded":
            raise ApiError(HTTPStatus.CONFLICT, "Refinement visualization is unavailable")
        with self.connect() as connection:
            session = connection.execute(
                "SELECT task_id,target_id FROM sessions WHERE id=?",
                (str(run["session_id"]),),
            ).fetchone()
        if session is None:
            raise RuntimeError("Refinement run session is missing")
        task_id = str(session["task_id"])
        # A refinement request deliberately freezes its own base method to
        # Ours-Full.  The owning session may have been bootstrapped while a
        # different browser rank method was selected, so it is not part of
        # the immutable model contract.
        if str(session["target_id"]) != "joint" or str(run["base_method"]) != "Ours-Full":
            raise RuntimeError("Refinement visualization requires the Joint Ours-Full session")
        task = self.task(task_id)
        run_directory = (self.runtime_root / str(run["artifact_relpath"])).resolve()
        if not run_directory.is_relative_to(self.runtime_root):
            raise RuntimeError("Refinement artifact path escapes the runtime root")
        model_path = run_directory / "model.npz"
        if not model_path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "Refinement model artifact is missing")

        params = json.loads(str(run["params_json"] or "{}"))
        if not isinstance(params, dict):
            raise RuntimeError("Refinement parameters are malformed")
        if "gammaMax" not in params:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Legacy refinement runs do not provide the current PCP snapshot",
            )
        base = self._refinement_base_for_run(task_id, mode, params)
        if params.get("probeSource") == "updated":
            base, audit, _ = self._load_selected_probe_update(
                str(run["user_id"]), str(run["session_id"]), task_id,
                str(params.get("probeUpdateId") or ""), base, params.get("nativeProbeRequest"),
            )
            if audit["snapshotFingerprint"] != params.get("probeSnapshotFingerprint"):
                raise RuntimeError("Selected Probe snapshot differs from the refinement artifact")
        elif mode in PROBE_REFINEMENT_RUN_MODES:
            from native_probe_update import load_snapshot

            base, _ = load_snapshot(self, params["nativeProbeRequest"], base)
        with np.load(model_path, allow_pickle=False) as model:
            required = {
                "beta",
                "gamma",
                "embeddingWeights",
                "embeddingFusionStrength",
                "theta",
                "temperature",
                "attributeIds",
                "learnerMethods",
                "embeddingMethods",
                "baseStateFingerprint",
            }
            if required - set(model.files):
                raise RuntimeError("Refinement model artifact is incomplete")
            artifact_attribute_ids = tuple(str(value) for value in model["attributeIds"].tolist())
            artifact_learners = tuple(str(value) for value in model["learnerMethods"].tolist())
            artifact_embeddings = tuple(str(value) for value in model["embeddingMethods"].tolist())
            artifact_base_fingerprint = str(model["baseStateFingerprint"].reshape(-1)[0])
            if (
                artifact_attribute_ids != base.attribute_ids
                or artifact_learners != base.learner_names
                or artifact_embeddings != base.embedding_names
                or artifact_base_fingerprint != base.base_state_fingerprint
            ):
                raise RuntimeError("Refinement visualization base snapshot does not match the run")
            beta = np.asarray(model["beta"], dtype=np.float64)
            gamma = np.asarray(model["gamma"], dtype=np.float64)
            embedding_weights = np.asarray(model["embeddingWeights"], dtype=np.float64)
            stored_embedding_strength = np.asarray(model["embeddingFusionStrength"])
            embedding_strength = float(stored_embedding_strength.reshape(-1)[0])
            theta = np.asarray(model["theta"], dtype=np.float64)
            temperature = np.asarray(model["temperature"], dtype=np.float64)

        if stored_embedding_strength.dtype.kind == "f" and stored_embedding_strength.dtype.itemsize == 4:
            # Older artifacts rounded lambda to float32 *after* scoring with
            # its full Python-float precision. Recover only the same run's
            # persisted value, and only when it round-trips to that artifact.
            # Near ties can change ranks even when score differences are tiny;
            # all base/score/rank consistency checks below still apply.
            try:
                saved_after = json.loads(str(run["after_json"] or "{}"))
            except (ValueError, TypeError) as error:
                raise RuntimeError("Refinement model summary is malformed") from error
            if not isinstance(saved_after, dict):
                raise RuntimeError("Refinement model summary is malformed")
            saved_summary = saved_after.get("modelSummary", {})
            if not isinstance(saved_summary, dict):
                raise RuntimeError("Refinement model summary is malformed")
            if "embeddingFusionStrength" in saved_summary:
                full_strength = saved_summary["embeddingFusionStrength"]
                if (
                    isinstance(full_strength, bool)
                    or not isinstance(full_strength, (int, float))
                    or not 0.0 <= full_strength <= 1.0
                    or not np.isfinite(full_strength)
                    or float(np.float32(full_strength)) != embedding_strength
                ):
                    raise RuntimeError("Refinement model summary lambda does not match the saved artifact")
                embedding_strength = float(full_strength)

        from tuning_models import unified_weight_scores

        refined = unified_weight_scores(
            base.probe_features,
            base.embedding_features,
            beta,
            gamma,
            embedding_weights,
            embedding_strength,
            theta,
            temperature,
            gamma_min=float(params.get("gammaMin", REFINEMENT_GAMMA_MIN)),
            gamma_max=float(params.get("gammaMax", REFINEMENT_GAMMA_MAX)),
        )
        # C is the exact log-domain conjunction output, not F divided by the
        # embedding factor (which can be zero). The two branches are siblings.
        component_columns: list[Any] = [
            refined.final_scores, refined.conjunction_scores,
        ]
        for attribute_index in range(len(base.attribute_ids)):
            component_columns.append(refined.gates[:, attribute_index])
            component_columns.extend(
                base.probe_features[:, attribute_index, learner_index]
                for learner_index in range(len(base.learner_names))
            )
        component_columns.append(refined.holistic_scores)
        component_columns.extend(
            base.embedding_features[:, method_index]
            for method_index in range(len(base.embedding_names))
        )
        component_scores = np.ascontiguousarray(
            np.column_stack(component_columns), dtype=np.float32
        )
        expected_component_count = (
            2
            + len(base.attribute_ids) * (1 + len(base.learner_names))
            + 1
            + len(base.embedding_names)
        )
        if component_scores.shape != (task.row_count, expected_component_count):
            raise RuntimeError("Refinement visualization component shape is invalid")
        if not np.isfinite(component_scores).all():
            raise RuntimeError("Refinement visualization contains non-finite scores")
        component_ranks = np.ascontiguousarray(
            np.column_stack(
                [
                    normalized_ranks(component_scores[:, index])
                    for index in range(expected_component_count)
                ]
            ),
            dtype=np.float32,
        )

        tuned_scores_path = run_directory / "tuned-scores.f32"
        tuned_ranks_path = run_directory / "tuned-ranks.f32"
        tuned_scores = np.fromfile(tuned_scores_path, dtype="<f4")
        tuned_ranks = np.fromfile(tuned_ranks_path, dtype="<f4")
        if (
            tuned_scores.shape != (task.row_count,)
            or tuned_ranks.shape != (task.row_count,)
            or not np.allclose(component_scores[:, 0], tuned_scores, rtol=0.0, atol=1e-7)
            or not np.allclose(component_ranks[:, 0], tuned_ranks, rtol=0.0, atol=1e-7)
        ):
            raise RuntimeError("Refinement PCP Joint column does not match the applied ranking")

        source_hasher = hashlib.sha256()
        source_hasher.update(REFINEMENT_VISUALIZATION_ALGORITHM.encode("ascii"))
        source_hasher.update(str(run["id"]).encode("utf-8"))
        source_hasher.update(base.base_state_fingerprint.encode("ascii"))
        source_hasher.update(hashlib.sha256(model_path.read_bytes()).digest())
        # A legacy lambda may come from after_json rather than model.npz.
        source_hasher.update(np.asarray([embedding_strength], dtype="<f8").tobytes())
        source_fingerprint = source_hasher.hexdigest()
        payload = np.concatenate(
            [component_scores.reshape(-1), component_ranks.reshape(-1)]
        ).astype("<f4", copy=False).tobytes(order="C")
        cluster_fit_mask = np.zeros(task.row_count, dtype=np.uint8)
        cluster_fit_mask[base.development_indices] = 1
        return RefinementVisualizationResult(
            payload=payload,
            row_count=task.row_count,
            component_count=expected_component_count,
            component_ranks=component_ranks,
            source_fingerprint=source_fingerprint,
            base_state_fingerprint=base.base_state_fingerprint,
            cluster_fit_mask=cluster_fit_mask.tobytes(),
        )

    def refinement_visualization(
        self, run: sqlite3.Row
    ) -> RefinementVisualizationResult:
        run_id = str(run["id"])
        with self._refinement_visualization_guard:
            cached = self._refinement_visualizations.pop(run_id, None)
            if cached is not None:
                self._refinement_visualizations[run_id] = cached
                return cached
            result = self._compute_refinement_visualization(run)
            if len(self._refinement_visualizations) >= 4:
                oldest = next(iter(self._refinement_visualizations))
                del self._refinement_visualizations[oldest]
            self._refinement_visualizations[run_id] = result
            return result

    def refinement_clusters(
        self, run: sqlite3.Row, scheme: str
    ) -> DynamicPcpClusterResult:
        if scheme not in DYNAMIC_PCP_CLUSTER_SCHEMES:
            raise ValueError(
                "scheme must be one of " + ", ".join(DYNAMIC_PCP_CLUSTER_SCHEMES)
            )
        visualization = self.refinement_visualization(run)
        key = (str(run["id"]), visualization.source_fingerprint, scheme)
        while True:
            with self._refinement_cluster_guard:
                cached = self._refinement_clusters.pop(key, None)
                if cached is not None:
                    self._refinement_clusters[key] = cached
                    return cached
                pending = self._refinement_cluster_inflight.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._refinement_cluster_inflight[key] = pending
                    owner = True
                else:
                    owner = False
            if owner:
                break
            pending.wait()

        try:
            with self.connect() as connection:
                session = connection.execute(
                    "SELECT task_id FROM sessions WHERE id=?", (str(run["session_id"]),)
                ).fetchone()
            if session is None:
                raise RuntimeError("Refinement run session is missing")
            task_id = str(session["task_id"])
            labels, cluster_count, fit_row_count = fit_dynamic_pcp_cluster_labels(
                visualization.component_ranks,
                visualization.cluster_fit_mask or self.bundle(task_id).development_mask,
                scheme,
            )
            result = DynamicPcpClusterResult(
                payload=labels.tobytes(order="C"),
                row_count=visualization.row_count,
                cluster_count=cluster_count,
                scheme=scheme,
                feature_count=visualization.component_count,
                fit_row_count=fit_row_count,
            )
        except BaseException:
            with self._refinement_cluster_guard:
                self._refinement_cluster_inflight.pop(key, None)
                pending.set()
            raise
        with self._refinement_cluster_guard:
            if len(self._refinement_clusters) >= DYNAMIC_PCP_CLUSTER_CACHE_SIZE:
                oldest = next(iter(self._refinement_clusters))
                del self._refinement_clusters[oldest]
            self._refinement_clusters[key] = result
            self._refinement_cluster_inflight.pop(key, None)
            pending.set()
        return result

    def validate_target_method(
        self, task_id: str, target_id: str, base_method: str | None
    ) -> tuple[TaskSpec, str]:
        task = self.task(task_id)
        if target_id not in task.target_ids:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown target for {task_id}: {target_id}")
        method = base_method or str(task.manifest.get("defaultRankMethod", ""))
        if method not in task.methods:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown method for {task_id}: {method}")
        return task, method

    def validate_annotation(
        self, session: sqlite3.Row, row_index: int, image_id: str,
        *, enforce_vqa_holdout: bool = True,
        vqa_validation_rows: set[int] | None = None,
    ) -> None:
        bundle = self.bundle(str(session["task_id"]))
        if row_index < 0 or row_index >= bundle.spec.row_count:
            raise ApiError(HTTPStatus.BAD_REQUEST, "rowIndex is outside the task bundle")
        expected = bundle.image_ids[row_index]
        if image_id != expected:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"imageId does not match row {row_index}",
            )
        if row_index in bundle.query_indices:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Fixed Query images are read-only and cannot be used for tuning",
            )
        if bundle.test_mask[row_index] == 1:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Frozen Test images are read-only and cannot be used for tuning",
            )
        if enforce_vqa_holdout and vqa_validation_rows is None:
            vqa_validation_rows = {
                int(row) for row in self._vqa_validation_split(
                    str(session["task_id"]), "joint"
                ).validation_indices
            }
        if enforce_vqa_holdout and row_index in (vqa_validation_rows or set()):
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Fixed VQA Validation images are read-only and cannot be used for tuning",
            )
        if bundle.development_mask[row_index] == 1:
            return
        if bundle.validation_mask[row_index] == 1:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Validation images are read-only and cannot be used for tuning",
            )
        if bundle.test_mask[row_index] == 1:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Frozen Test images are read-only and cannot be used for tuning",
            )
        raise ApiError(
            HTTPStatus.CONFLICT,
            "Only Development images can be used for tuning",
        )

    def _suggest_failed_attributes(
        self, task_id: str, row_indices: Iterable[int]
    ) -> dict[int, str]:
        """Recommend the lowest current Ours-Full attribute gate per row.

        This is a model-only hint.  It is stored separately from confirmed
        failed attributes and never becomes attribute supervision until the
        user confirms or changes it.
        """

        import numpy as np

        task = self.task(task_id)
        attributes = [
            str(target.get("id"))
            for target in task.manifest.get("retrievalTargets", [])
            if target.get("kind") == "attribute"
        ]
        rows = np.asarray([int(value) for value in row_indices], dtype=np.int64)
        if rows.size == 0 or not attributes:
            return {}
        required_methods = {
            *WEIGHTED_FUSION_LEARNERS,
            *REFINEMENT_EMBEDDING_METHODS,
            "Ours-Full",
        }
        if not required_methods.issubset(task.methods):
            # Minimal test manifests cannot reconstruct the formal gate. Keep
            # this a recommendation only and require explicit confirmation.
            return {int(row): attributes[0] for row in rows}
        try:
            from tuning_models import unified_weight_scores

            base = self._refinement_base_context(task_id)
            uniform_beta = np.full(
                (len(base.attribute_ids), len(base.learner_names)),
                1.0 / len(base.learner_names),
                dtype=np.float64,
            )
            # Suggestions must describe the same fixed gate as the baseline.
            theta = base.initial_theta
            initial = unified_weight_scores(
                base.probe_features[rows],
                base.embedding_features[rows],
                uniform_beta,
                np.ones(len(base.attribute_ids), dtype=np.float64),
                np.full(
                    len(base.embedding_names),
                    1.0 / len(base.embedding_names),
                    dtype=np.float64,
                ),
                REFINEMENT_LAMBDA_0,
                theta,
                base.temperature,
                gamma_min=REFINEMENT_GAMMA_MIN,
                gamma_max=REFINEMENT_GAMMA_MAX,
            )
            choices = np.argmin(initial.gates, axis=1)
        except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
            # Annotation must remain available even when a task's optional
            # refinement assets are incomplete. Without an exact formal gate,
            # return no model suggestion rather than silently using another
            # score scale.
            print(
                f"warning: failed-attribute suggestion unavailable for {task_id}: {error}",
                file=sys.stderr,
            )
            return {}
        return {
            int(row): attributes[int(choices[position])]
            for position, row in enumerate(rows)
        }

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._session_locks_guard:
            return self._session_locks.setdefault(session_id, threading.Lock())

    def _sign_cookie(self, user_id: str) -> str:
        signature = hmac.new(
            self.cookie_secret, user_id.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{user_id}.{signature}"

    @staticmethod
    def _cookie_user_id(value: str | None, secret: bytes) -> str | None:
        if not value or "." not in value:
            return None
        user_id, signature = value.rsplit(".", 1)
        if not SAFE_ID_PATTERN.fullmatch(user_id):
            return None
        expected = hmac.new(
            secret, user_id.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return user_id if hmac.compare_digest(signature, expected) else None

    def _verify_cookie(self, value: str | None) -> tuple[str, bool] | None:
        current_user_id = self._cookie_user_id(value, self.cookie_secret)
        if current_user_id is not None:
            return current_user_id, False
        if self.retired_cookie_secret is None:
            return None
        retired_user_id = self._cookie_user_id(value, self.retired_cookie_secret)
        return (retired_user_id, True) if retired_user_id is not None else None

    def cookie_header(self, user_id: str, secure: bool) -> str:
        parts = [
            f"{COOKIE_NAME}={self._sign_cookie(user_id)}",
            "Path=/",
            "HttpOnly",
            "SameSite=Lax",
            "Max-Age=31536000",
        ]
        if secure:
            parts.append("Secure")
        return "; ".join(parts)

    def identify(
        self,
        headers: Any,
        *,
        create: bool,
        display_name: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        now = utc_now()
        identity_key: str | None = None
        cookie_user_id: str | None = None
        retired_cookie = False
        identity_source = "cookie"
        proposed_name = (display_name or "").strip()
        if len(proposed_name) > 80:
            raise ApiError(HTTPStatus.BAD_REQUEST, "displayName must be at most 80 characters")

        oai_id = headers.get("oai-authenticated-user-id") if self.trust_oai_headers else None
        if oai_id:
            identity_source = "oai"
            identity_key = "oai:" + hashlib.sha256(oai_id.encode("utf-8")).hexdigest()
            if not proposed_name:
                proposed_name = self._oai_display_name(headers)
        else:
            cookies = SimpleCookie()
            try:
                cookies.load(headers.get("Cookie", ""))
            except Exception:
                cookies = SimpleCookie()
            morsel = cookies.get(COOKIE_NAME)
            verified_cookie = self._verify_cookie(morsel.value if morsel else None)
            if verified_cookie:
                cookie_user_id, retired_cookie = verified_cookie
                identity_key = f"cookie:{cookie_user_id}"

        with self.connect() as connection:
            row = None
            if cookie_user_id is not None:
                # A server-signed cookie is also a safe fallback for an OAI user
                # if an upstream request temporarily omits the identity header.
                row = connection.execute(
                    "SELECT * FROM users WHERE id=?", (cookie_user_id,)
                ).fetchone()
                if retired_cookie:
                    already_migrated = connection.execute(
                        "SELECT 1 FROM retired_cookie_migrations WHERE user_id=?",
                        (cookie_user_id,),
                    ).fetchone()
                    if not create or row is None or already_migrated is not None:
                        row = None
                        cookie_user_id = None
                        identity_key = None
                    else:
                        connection.execute(
                            "INSERT INTO retired_cookie_migrations(user_id,migrated_at) "
                            "VALUES(?,?)",
                            (cookie_user_id, now),
                        )
            elif identity_key is not None:
                row = connection.execute(
                    "SELECT * FROM users WHERE identity_key=?", (identity_key,)
                ).fetchone()
            created = False
            if row is None:
                if not create:
                    raise ApiError(HTTPStatus.UNAUTHORIZED, "Tuning identity is required")
                user_id = new_id("user")
                if identity_source == "cookie":
                    identity_key = f"cookie:{user_id}"
                assert identity_key is not None
                name = proposed_name or "Researcher"
                connection.execute(
                    "INSERT INTO users(id,identity_key,identity_source,display_name,created_at,last_seen_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (user_id, identity_key, identity_source, name, now, now),
                )
                created = True
            else:
                user_id = str(row["id"])
                name = proposed_name or str(row["display_name"])
                connection.execute(
                    "UPDATE users SET display_name=?,last_seen_at=? WHERE id=?",
                    (name, now, user_id),
                )
            result = connection.execute(
                "SELECT id,display_name,identity_source,created_at,last_seen_at "
                "FROM users WHERE id=?",
                (user_id,),
            ).fetchone()
        return (
            {
                "id": str(result["id"]),
                "displayName": str(result["display_name"]),
                "identitySource": (
                    "openai"
                    if str(result["identity_source"]) == "oai"
                    else "local-cookie"
                ),
                "createdAt": str(result["created_at"]),
                "lastSeenAt": str(result["last_seen_at"]),
            },
            created,
        )

    @staticmethod
    def _oai_display_name(headers: Any) -> str:
        encoded = headers.get("oai-authenticated-user-full-name")
        encoding = headers.get("oai-authenticated-user-full-name-encoding")
        if encoded and encoding == "percent-encoded-utf-8":
            try:
                return unquote(encoded)[:80]
            except Exception:
                pass
        return (headers.get("oai-authenticated-user-email") or "Researcher")[:80]

    def refinement_capabilities(self, task_id: str) -> dict[str, Any]:
        self.task(task_id)
        from experimental_parent_overlay import applies
        if applies(task_id):
            return {
                "update_probes": {"available": False, "reason": "Robust test inherits frozen parent Probes; use Weight Tune.",
                                  "algorithmVersion": "native-probe-update-then-freeze-v1"},
                **{mode: {"available": mode in WEIGHT_REFINEMENT_RUN_MODES,
                           "reason": "Frozen-parent robustness task",
                           "algorithmVersion": RUN_ALGORITHM_VERSIONS[mode]} for mode in sorted(REFINEMENT_RUN_MODES)},
            }
        from native_probe_update import capability

        native = capability(self, task_id)
        return {
            "update_probes": {**native, "algorithmVersion": "native-probe-update-then-freeze-v1"},
            **{mode: {**({"available": False, "reason": "Historical combined mode; use Update Probes then Weight Tune"}
                        if mode in PROBE_REFINEMENT_RUN_MODES else {"available": True}),
                   "algorithmVersion": RUN_ALGORITHM_VERSIONS[mode]}
            for mode in sorted(REFINEMENT_RUN_MODES)},
        }

    def _run_native_probe_update(self, request: dict, bank: dict, output: Path) -> tuple[Any, dict]:
        from native_probe_update import run_native_updates

        return run_native_updates(self, request, bank, output)

    @staticmethod
    def _assert_session_idle(connection: sqlite3.Connection, session_id: str) -> None:
        """Call under BEGIN IMMEDIATE so two browser tabs cannot both enqueue."""
        for table in ("model_runs", "probe_updates"):
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE session_id=? AND status IN ('queued','running') LIMIT 1",
                (session_id,),
            ).fetchone():
                raise ApiError(HTTPStatus.CONFLICT, "This session already has a queued or running job")

    @staticmethod
    def _verified_probe_request(row: sqlite3.Row) -> dict:
        raw = str(row["request_json"])
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != row["request_sha256"]:
            raise RuntimeError("Probe update request checksum mismatch")
        envelope = json.loads(raw)
        request = envelope["nativeRequest"]
        if (request.get("userId") != row["user_id"]
                or request.get("sessionId") != row["session_id"]
                or request.get("taskId") != row["task_id"]
                or row["target_id"] != "joint"):
            raise RuntimeError("Probe update owner/session/task mismatch")
        return request

    def _probe_update_compatible(self, row: sqlite3.Row) -> bool:
        try:
            request = self._verified_probe_request(row)
            base = self._refinement_base_context(str(row["task_id"]))
            return (request.get("baseStateFingerprint") == base.base_state_fingerprint
                    and request.get("normalizationContract") == base.normalization_audit
                    and request.get("attributeIds") == list(base.attribute_ids))
        except (OSError, ValueError, KeyError, RuntimeError):
            return False

    def probe_update_json(self, row: sqlite3.Row, *, annotation_sha256: str | None = None) -> dict:
        request = self._verified_probe_request(row)
        if annotation_sha256 is None:
            with self.connect() as connection:
                annotations = connection.execute(
                    "SELECT * FROM annotations WHERE session_id=? ORDER BY row_index", (row["session_id"],),
                ).fetchall()
            annotation_sha256 = self.annotation_state_sha256(annotations)
        return {
            "id": str(row["id"]), "sessionId": str(row["session_id"]), "taskId": str(row["task_id"]),
            "status": str(row["status"]), "createdAt": row["created_at"],
            "startedAt": row["started_at"], "finishedAt": row["finished_at"], "error": row["error"],
            "snapshotId": request["snapshotId"], "baseStateFingerprint": request["baseStateFingerprint"],
            "annotationCount": int(row["annotation_count"]), "positiveCount": int(row["positive_count"]),
            "negativeCount": int(row["negative_count"]), "annotationStateSha256": str(row["annotation_sha256"]),
            "updatedAttributes": [item["attributeId"] for item in request["examples"] if item["update"]],
            "compatible": self._probe_update_compatible(row),
            "stale": str(row["annotation_sha256"]) != annotation_sha256,
        }

    def owned_probe_update(self, update_id: str, user_id: str, *, session_id: str | None = None) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT p.*,s.task_id,s.target_id FROM probe_updates p JOIN sessions s ON s.id=p.session_id "
                "WHERE p.id=? AND p.user_id=?", (update_id, user_id),
            ).fetchone()
        if row is None or (session_id is not None and row["session_id"] != session_id):
            raise ApiError(HTTPStatus.NOT_FOUND, "Probe update not found in this session")
        return row

    def list_probe_updates(self, user_id: str, session_id: str) -> list[dict]:
        with self.connect() as connection:
            self.owned_session(connection, session_id, user_id)
            rows = connection.execute(
                "SELECT p.*,s.task_id,s.target_id FROM probe_updates p JOIN sessions s ON s.id=p.session_id "
                "WHERE p.session_id=? AND p.user_id=? ORDER BY p.created_at DESC LIMIT 100",
                (session_id, user_id),
            ).fetchall()
            annotations = connection.execute(
                "SELECT * FROM annotations WHERE session_id=? ORDER BY row_index", (session_id,),
            ).fetchall()
        annotation_sha256 = self.annotation_state_sha256(annotations)
        return [self.probe_update_json(row, annotation_sha256=annotation_sha256) for row in rows]

    def create_probe_update(self, user_id: str, session_id: str, payload: dict) -> dict:
        """Persist an independent native update; no fusion model/ranking is created."""
        from native_probe_update import config_contract, prepare_request
        from experimental_parent_overlay import applies

        with self.connect() as connection:
            initial_session = self.owned_session(connection, session_id, user_id)
            if applies(str(initial_session["task_id"])):
                raise ApiError(HTTPStatus.CONFLICT, "This experimental task supports frozen Probe Weight Tune only")

        feedback_weight = self.bounded_float(payload.get("feedbackWeight", 8.0), "feedbackWeight", 1, 100)
        try:
            native_config = config_contract(payload.get("nativeUpdateConfig"))
        except ValueError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self.owned_session(connection, session_id, user_id)
            if session["target_id"] != "joint":
                raise ApiError(HTTPStatus.CONFLICT, "Update Probes requires the Joint target")
            annotations = connection.execute(
                "SELECT * FROM annotations WHERE session_id=? AND label != 0 ORDER BY row_index", (session_id,),
            ).fetchall()
            split = self._vqa_validation_split(str(session["task_id"]), "joint")
            val_rows = {int(row) for row in split.validation_indices}
            for annotation in annotations:
                self.validate_annotation(session, int(annotation["row_index"]), str(annotation["image_id"]), enforce_vqa_holdout=False)
            trainable = [row for row in annotations if int(row["row_index"]) not in val_rows]
            if not trainable:
                raise ApiError(HTTPStatus.CONFLICT, "Update Probes requires Development feedback outside Val")
            if any(int(row["label"]) < 0 and (
                    not bool(row["failure_attribution_confirmed"])
                    or not json.loads(str(row["failed_attributes_json"] or "[]"))) for row in trainable):
                raise ApiError(HTTPStatus.CONFLICT, "Confirm at least one failed query attribute for each negative correction")
            base = self._refinement_base_context(str(session["task_id"]), vqa_validation_version=split.audit["frozenVersion"])
            try:
                request = prepare_request(self, user_id, session_id, str(session["task_id"]), base,
                    [self.annotation_json(row) for row in annotations], {
                        "feedbackWeight": feedback_weight, "nativeUpdateConfig": native_config,
                        "vqaValidationVersion": split.audit["frozenVersion"],
                    })
            except (OSError, ValueError, RuntimeError) as error:
                raise ApiError(HTTPStatus.CONFLICT, str(error)) from error
            if not any(item["update"] for item in request["examples"]):
                raise ApiError(HTTPStatus.CONFLICT, "No confirmed attribute feedback is available for Probe updating")
            request_text = json_dumps({"nativeRequest": request,
                                       "initialBaselineIdentity": getattr(base, "initial_baseline_identity", None)})
            checksum = hashlib.sha256(request_text.encode("utf-8")).hexdigest()
            existing = connection.execute(
                "SELECT id,status FROM probe_updates WHERE session_id=? AND request_sha256=?",
                (session_id, checksum),
            ).fetchone()
            if existing is not None and existing["status"] in {"queued", "running", "succeeded"}:
                update_id = str(existing["id"])
                enqueue = False
            else:
                self._assert_session_idle(connection, session_id)
                update_id = str(existing["id"]) if existing is not None else new_id("probe_update")
                if existing is not None:
                    connection.execute(
                        "UPDATE probe_updates SET status='queued',error=NULL,started_at=NULL,finished_at=NULL WHERE id=?", (update_id,),
                    )
                else:
                    connection.execute(
                        "INSERT INTO probe_updates(id,session_id,user_id,status,request_json,request_sha256,annotation_sha256,"
                        "annotation_count,positive_count,negative_count,created_at) VALUES(?,?,?,'queued',?,?,?,?,?,?,?)",
                        (update_id, session_id, user_id, request_text, checksum, self.annotation_state_sha256(annotations),
                         len(annotations), sum(int(row["label"]) > 0 for row in annotations),
                         sum(int(row["label"]) < 0 for row in annotations), utc_now()),
                    )
                enqueue = True
        if enqueue:
            self.run_queue.put("probe-update:" + update_id)
        return self.probe_update_json(self.owned_probe_update(update_id, user_id))

    def _load_selected_probe_update(self, user_id: str, session_id: str, task_id: str,
                                    update_id: str, base: Any, expected_request: dict | None = None) -> tuple[Any, dict, dict]:
        from native_probe_update import load_snapshot

        row = self.owned_probe_update(update_id, user_id, session_id=session_id)
        request = self._verified_probe_request(row)
        if row["status"] != "succeeded" or row["task_id"] != task_id:
            raise ApiError(HTTPStatus.CONFLICT, "Select a completed Probe update for this task")
        if expected_request is not None and request != expected_request:
            raise RuntimeError("Selected frozen Probe request changed after queuing")
        # No new native request is prepared: later feedback trains only weights.
        updated, audit = load_snapshot(self, request, base)
        stored_audit = json.loads(str(row["audit_json"] or "{}"))
        if stored_audit != audit:
            raise RuntimeError("Selected frozen Probe snapshot fingerprint changed")
        return updated, audit, request

    def _run_probe_update_job(self, update_id: str) -> None:
        with self.connect() as connection:
            changed = connection.execute(
                "UPDATE probe_updates SET status='running',started_at=?,error=NULL WHERE id=? AND status='queued'",
                (utc_now(), update_id),
            ).rowcount
            row = connection.execute(
                "SELECT p.*,s.task_id,s.target_id FROM probe_updates p JOIN sessions s ON s.id=p.session_id WHERE p.id=?", (update_id,),
            ).fetchone()
        if changed != 1:
            return
        try:
            from native_probe_update import current_code_fingerprint, ensure_snapshot

            request = self._verified_probe_request(row)
            identity = json.loads(str(row["request_json"])).get("initialBaselineIdentity")
            if identity is not None:
                from unified_initial_baseline import load_task_baseline

                loaded = load_task_baseline(self, str(row["task_id"]), identity["publicationVersion"],
                                            fingerprint=request["baseStateFingerprint"])
                if loaded is None or loaded[0].initial_baseline_identity != identity:
                    raise RuntimeError("Pinned initial Probe baseline is unavailable or changed")
                base = loaded[0]
            else:
                base = self._refinement_base_context(str(row["task_id"]),
                    vqa_validation_version=request["normalizationContract"]["vqaValidationVersion"], prefer_published=False)
            if (request.get("baseStateFingerprint") != base.base_state_fingerprint
                    or request.get("normalizationContract") != base.normalization_audit):
                raise RuntimeError("Probe update base/Val snapshot changed after queuing")
            completed = self.runtime_root / "native-probe-updates" / request["snapshotId"] / "complete.json"
            if not completed.is_file() and request.get("codeFingerprint") != current_code_fingerprint(self):
                raise RuntimeError("Native Probe training code changed after queuing; submit a new update")
            _, audit = ensure_snapshot(self, request, base)
            with self.connect() as connection:
                connection.execute(
                    "UPDATE probe_updates SET status='succeeded',audit_json=?,finished_at=? WHERE id=? AND status='running'",
                    (json_dumps(audit), utc_now(), update_id),
                )
        except Exception as error:
            traceback.print_exc()
            with self.connect() as connection:
                connection.execute(
                    "UPDATE probe_updates SET status='failed',error=?,finished_at=? WHERE id=? AND status='running'",
                    (f"{type(error).__name__}: {error}"[:2000], utc_now(), update_id),
                )

    def bootstrap(
        self, user: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        task_id = self.required_string(payload, "taskId")
        target_id = self.required_string(payload, "targetId")
        base_method_value = payload.get("baseMethod")
        base_method = str(base_method_value).strip() if base_method_value is not None else None
        _, method = self.validate_target_method(task_id, target_id, base_method)
        now = utc_now()
        with self.connect() as connection:
            session = None
            if payload.get("newSession") is not True:
                session = connection.execute(
                    "SELECT * FROM sessions WHERE user_id=? AND task_id=? AND target_id=? "
                    "AND base_method=? ORDER BY updated_at DESC LIMIT 1",
                    (user["id"], task_id, target_id, method),
                ).fetchone()
            if session is None:
                session_id = new_id("session")
                connection.execute(
                    "INSERT INTO sessions(id,user_id,task_id,target_id,base_method,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (session_id, user["id"], task_id, target_id, method, now, now),
                )
                session = connection.execute(
                    "SELECT * FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
            annotations = connection.execute(
                "SELECT row_index,image_id,label,failed_attributes_json,"
                "suggested_failure_attribute_id,failure_attribution_confirmed,"
                "source,created_at,updated_at "
                "FROM annotations WHERE session_id=? ORDER BY row_index",
                (session["id"],),
            ).fetchall()
            runs = connection.execute(
                "SELECT * FROM model_runs WHERE session_id=? ORDER BY created_at DESC LIMIT 20",
                (session["id"],),
            ).fetchall()
        current_annotation_state_sha256 = self.annotation_state_sha256(annotations)
        if annotations:
            validation_rows = {
                int(row) for row in self._vqa_validation_split(task_id, "joint").validation_indices
            }
            original = self._original_development_supervision(task_id, target_id)
            original_labels = {
                int(row_index): int(label)
                for row_index, label in zip(
                    original.indices,
                    original.labels,
                    strict=True,
                )
            }
            annotation_payloads = [
                self.annotation_with_supervision_json(row, original_labels, validation_rows)
                for row in annotations
            ]
        else:
            annotation_payloads = []
        return {
            "user": user,
            "session": self.session_json(session),
            "refinementCapabilities": self.refinement_capabilities(task_id),
            "probeUpdates": self.list_probe_updates(str(user["id"]), str(session["id"])),
            "annotations": annotation_payloads,
            "runs": [
                {
                    **self.run_json(row),
                    "stale": self._run_annotation_state_sha256(row)
                    != current_annotation_state_sha256,
                }
                for row in runs
            ],
        }

    def _run_annotation_state_sha256(self, run: sqlite3.Row) -> str:
        """Return the label-state hash, including for pre-separation legacy rows."""
        stored = str(run["annotation_sha256"])
        row_keys = set(run.keys())
        if (
            str(run["mode"]) not in LEGACY_RUN_MODES
            or ("snapshot_sha256" in row_keys and run["snapshot_sha256"])
        ):
            return stored
        run_directory = (self.runtime_root / str(run["artifact_relpath"])).resolve()
        if not run_directory.is_relative_to(self.runtime_root):
            return stored
        try:
            snapshot, snapshot_sha256 = self._read_verified_run_snapshot(
                run, run_directory
            )
        except (OSError, RuntimeError, KeyError, TypeError, ValueError):
            return stored
        if not hmac.compare_digest(snapshot_sha256, stored):
            return stored
        return self.annotation_state_sha256(snapshot["annotations"])

    @staticmethod
    def required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} is required")
        return value.strip()

    @staticmethod
    def session_json(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "taskId": str(row["task_id"]),
            "targetId": str(row["target_id"]),
            "baseMethod": str(row["base_method"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    @staticmethod
    def annotation_json(row: sqlite3.Row) -> dict[str, Any]:
        keys = set(row.keys())
        failed_attributes: list[str] = []
        if "failed_attributes_json" in keys and row["failed_attributes_json"]:
            try:
                raw_failed = json.loads(str(row["failed_attributes_json"]))
            except json.JSONDecodeError as error:
                raise RuntimeError("Stored failed-attribute list is invalid") from error
            if not isinstance(raw_failed, list) or not all(
                isinstance(value, str) for value in raw_failed
            ):
                raise RuntimeError("Stored failed-attribute list is invalid")
            failed_attributes = [str(value) for value in raw_failed]
        result = {
            "rowIndex": int(row["row_index"]),
            "imageId": str(row["image_id"]),
            "label": int(row["label"]),
            "source": str(row["source"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }
        if int(row["label"]) < 0:
            result["failedAttributeIds"] = failed_attributes
            result["suggestedFailedAttributeId"] = (
                str(row["suggested_failure_attribute_id"])
                if "suggested_failure_attribute_id" in keys
                and row["suggested_failure_attribute_id"]
                else None
            )
            result["failureAttributionConfirmed"] = bool(
                row["failure_attribution_confirmed"]
                if "failure_attribution_confirmed" in keys
                else False
            )
        return result

    @classmethod
    def annotation_with_supervision_json(
        cls,
        row: sqlite3.Row,
        original_labels: Mapping[int, int],
        validation_rows: set[int] | None = None,
    ) -> dict[str, Any]:
        """Add a response-only relation to the immutable original DG labels."""
        annotation = cls.annotation_json(row)
        row_index = int(annotation["rowIndex"])
        label = int(annotation["label"])
        original_label = original_labels.get(row_index)
        if original_label is not None and int(original_label) not in (0, 1):
            raise RuntimeError("Original Development supervision must be binary")
        effective_label = None if label == 0 else int(label > 0)
        if effective_label is None:
            relation = "uncertain"
        elif original_label is None:
            relation = "new"
        elif effective_label == int(original_label):
            relation = "reinforce"
        else:
            relation = "override"
        annotation["supervision"] = {
            "origin": "new" if original_label is None else "existing",
            "relation": relation,
            "originalLabel": (
                None if original_label is None else int(original_label)
            ),
            "effectiveLabel": effective_label,
            "includedInTune": effective_label is not None and row_index not in (validation_rows or set()),
        }
        if row_index in (validation_rows or set()):
            annotation["supervision"]["excludedReason"] = "fixed-vqa-validation"
        return annotation

    @classmethod
    def annotation_state_sha256(cls, rows: Iterable[Any]) -> str:
        """Hash the effective, trainable labels independently of run metadata."""
        state = []
        for row in rows:
            row_keys = set(row.keys())
            row_index_key = "row_index" if "row_index" in row_keys else "rowIndex"
            image_id_key = "image_id" if "image_id" in row_keys else "imageId"
            label = int(row["label"])
            if label == 0:
                continue
            state.append(
                {
                    "rowIndex": int(row[row_index_key]),
                    "imageId": str(row[image_id_key]),
                    "label": label,
                    **(
                        {
                            "failedAttributeIds": sorted(
                                str(value)
                                for value in (
                                    row["failed_attributes_json"]
                                    if "failed_attributes_json" in row_keys
                                    and isinstance(row["failed_attributes_json"], list)
                                    else (
                                        json.loads(str(row["failed_attributes_json"]))
                                        if "failed_attributes_json" in row_keys
                                        and row["failed_attributes_json"]
                                        else row.get("failedAttributeIds", [])
                                        if isinstance(row, dict)
                                        else []
                                    )
                                )
                            )
                        }
                        if label < 0
                        and (
                            (
                                "failure_attribution_confirmed" in row_keys
                                and bool(row["failure_attribution_confirmed"])
                            )
                            or (
                                "failureAttributionConfirmed" in row_keys
                                and bool(row["failureAttributionConfirmed"])
                            )
                        )
                        else {}
                    ),
                }
            )
        state.sort(key=lambda value: (value["rowIndex"], value["imageId"]))
        return hashlib.sha256((json_dumps(state) + "\n").encode("utf-8")).hexdigest()

    def owned_session(
        self, connection: sqlite3.Connection, session_id: str, user_id: str
    ) -> sqlite3.Row:
        if not SAFE_ID_PATTERN.fullmatch(session_id):
            raise ApiError(HTTPStatus.NOT_FOUND, "Session not found")
        row = connection.execute(
            "SELECT * FROM sessions WHERE id=? AND user_id=?", (session_id, user_id)
        ).fetchone()
        if row is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "Session not found")
        return row

    def annotation_counts(
        self, connection: sqlite3.Connection, session_id: str
    ) -> dict[str, int]:
        counts = {value: 0 for value in LABELS}
        for row in connection.execute(
            "SELECT label,COUNT(*) AS count FROM annotations WHERE session_id=? GROUP BY label",
            (session_id,),
        ):
            counts[int(row["label"])] = int(row["count"])
        return {
            "strongNegative": counts[-2],
            "negative": counts[-1],
            "uncertain": counts[0],
            "positive": counts[1],
            "strongPositive": counts[2],
            "usablePositive": counts[1] + counts[2],
            "usableNegative": counts[-1] + counts[-2],
        }

    def put_annotation(
        self,
        user_id: str,
        session_id: str,
        row_index: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        image_id = self.required_string(payload, "imageId")
        label = payload.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in LABELS:
            raise ApiError(HTTPStatus.BAD_REQUEST, "label must be one of -2,-1,0,1,2")
        source = str(payload.get("source") or "manual")
        if not SOURCE_PATTERN.fullmatch(source):
            raise ApiError(HTTPStatus.BAD_REQUEST, "source contains unsupported characters")
        now = utc_now()
        with self._session_lock(session_id):
            with self.connect() as connection:
                session = self.owned_session(connection, session_id, user_id)
                self.validate_annotation(session, row_index, image_id)
                attribute_ids = tuple(
                    str(target.get("id"))
                    for target in self.task(
                        str(session["task_id"])
                    ).manifest.get("retrievalTargets", [])
                    if target.get("kind") == "attribute"
                )
                failed_payload = payload.get("failedAttributeIds")
                if failed_payload is not None and (
                    not isinstance(failed_payload, list)
                    or not all(isinstance(value, str) for value in failed_payload)
                ):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "failedAttributeIds must be an array of attribute IDs",
                    )
                failed_attributes = tuple(
                    dict.fromkeys(str(value) for value in (failed_payload or ()))
                )
                if any(value not in attribute_ids for value in failed_attributes):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "failedAttributeIds contains an unknown query attribute",
                    )
                requested_confirmation = payload.get(
                    "failureAttributionConfirmed",
                    bool(failed_attributes),
                )
                if not isinstance(requested_confirmation, bool):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "failureAttributionConfirmed must be boolean",
                    )
                if label >= 0:
                    failed_attributes = ()
                    requested_confirmation = False
                elif requested_confirmation and not failed_attributes and not (
                    failed_payload == []
                    and supports_relation_mismatch(str(session["task_id"]), str(session["target_id"]))
                ):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "Confirm at least one failed attribute for a negative label",
                    )
                original = self._original_development_supervision(
                    str(session["task_id"]),
                    str(session["target_id"]),
                )
                original_labels = {
                    int(original_row): int(original_label)
                    for original_row, original_label in zip(
                        original.indices,
                        original.labels,
                        strict=True,
                    )
                }
                previous = connection.execute(
                    "SELECT label,failed_attributes_json,"
                    "suggested_failure_attribute_id,failure_attribution_confirmed,"
                    "created_at FROM annotations WHERE session_id=? AND row_index=?",
                    (session_id, row_index),
                ).fetchone()
                if label < 0 and failed_payload is None and previous is not None:
                    if int(previous["label"]) < 0:
                        try:
                            stored_failed = json.loads(
                                str(previous["failed_attributes_json"] or "[]")
                            )
                        except json.JSONDecodeError as error:
                            raise RuntimeError(
                                "Stored failed-attribute list is invalid"
                            ) from error
                        failed_attributes = tuple(str(value) for value in stored_failed)
                        requested_confirmation = bool(
                            previous["failure_attribution_confirmed"]
                        )
                suggested_failure = None
                if label < 0:
                    suggested_failure = self._suggest_failed_attributes(
                        str(session["task_id"]), [row_index]
                    ).get(row_index)
                    if previous is not None and previous[
                        "suggested_failure_attribute_id"
                    ]:
                        suggested_failure = str(
                            previous["suggested_failure_attribute_id"]
                        )
                connection.execute("BEGIN IMMEDIATE")
                created_at = str(previous["created_at"]) if previous else now
                connection.execute(
                    "INSERT INTO annotations(session_id,row_index,image_id,label,"
                    "failed_attributes_json,suggested_failure_attribute_id,"
                    "failure_attribution_confirmed,source,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id,row_index) "
                    "DO UPDATE SET image_id=excluded.image_id,label=excluded.label,"
                    "failed_attributes_json=excluded.failed_attributes_json,"
                    "suggested_failure_attribute_id="
                    "excluded.suggested_failure_attribute_id,"
                    "failure_attribution_confirmed="
                    "excluded.failure_attribution_confirmed,"
                    "source=excluded.source,updated_at=excluded.updated_at",
                    (
                        session_id,
                        row_index,
                        image_id,
                        label,
                        json_dumps(list(failed_attributes)),
                        suggested_failure,
                        int(requested_confirmation),
                        source,
                        created_at,
                        now,
                    ),
                )
                connection.execute(
                    "INSERT INTO annotation_events(session_id,user_id,row_index,image_id,previous_label,label,event,source,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        session_id,
                        user_id,
                        row_index,
                        image_id,
                        int(previous["label"]) if previous else None,
                        label,
                        "upsert",
                        source,
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
                )
                connection.execute("COMMIT")
                row = connection.execute(
                    "SELECT row_index,image_id,label,failed_attributes_json,"
                    "suggested_failure_attribute_id,failure_attribution_confirmed,"
                    "source,created_at,updated_at "
                    "FROM annotations WHERE session_id=? AND row_index=?",
                    (session_id, row_index),
                ).fetchone()
                counts = self.annotation_counts(connection, session_id)
                self._write_annotation_files(connection, session, user_id, {
                    "event": "upsert",
                    "rowIndex": row_index,
                    "imageId": image_id,
                    "previousLabel": int(previous["label"]) if previous else None,
                    "label": label,
                    "failedAttributeIds": list(failed_attributes),
                    "suggestedFailedAttributeId": suggested_failure,
                    "failureAttributionConfirmed": requested_confirmation,
                    "source": source,
                    "createdAt": now,
                })
        return {
            "annotation": self.annotation_with_supervision_json(row, original_labels),
            "counts": counts,
        }

    def put_annotations_bulk(
        self,
        user_id: str,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_annotations = payload.get("annotations")
        if not isinstance(raw_annotations, list) or not raw_annotations:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "annotations must be a non-empty array",
            )
        if len(raw_annotations) > MAX_BULK_ANNOTATIONS:
            raise ApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"A bulk annotation request may contain at most {MAX_BULK_ANNOTATIONS} rows",
            )
        source = str(payload.get("source") or "selection-bulk")
        if not SOURCE_PATTERN.fullmatch(source):
            raise ApiError(HTTPStatus.BAD_REQUEST, "source contains unsupported characters")

        annotations: list[tuple[int, str, int]] = []
        seen_rows: set[int] = set()
        for item in raw_annotations:
            if not isinstance(item, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "each annotation must be an object")
            row_index = item.get("rowIndex")
            image_id = item.get("imageId")
            label = item.get("label")
            if isinstance(row_index, bool) or not isinstance(row_index, int):
                raise ApiError(HTTPStatus.BAD_REQUEST, "rowIndex must be an integer")
            if row_index in seen_rows:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"duplicate rowIndex {row_index}")
            if not isinstance(image_id, str) or not image_id:
                raise ApiError(HTTPStatus.BAD_REQUEST, "imageId must be a non-empty string")
            if isinstance(label, bool) or not isinstance(label, int) or label not in LABELS:
                raise ApiError(HTTPStatus.BAD_REQUEST, "label must be one of -2,-1,0,1,2")
            seen_rows.add(row_index)
            annotations.append((row_index, image_id, label))

        now = utc_now()
        with self._session_lock(session_id):
            with self.connect() as connection:
                session = self.owned_session(connection, session_id, user_id)
                # Validate every row before BEGIN so a bad member cannot produce a
                # partially saved batch.
                validation_rows = {
                    int(row) for row in self._vqa_validation_split(
                        str(session["task_id"]), "joint"
                    ).validation_indices
                }
                for row_index, image_id, _ in annotations:
                    self.validate_annotation(
                        session, row_index, image_id, vqa_validation_rows=validation_rows
                    )
                negative_suggestions = self._suggest_failed_attributes(
                    str(session["task_id"]),
                    [row_index for row_index, _, label in annotations if label < 0],
                )
                original = self._original_development_supervision(
                    str(session["task_id"]),
                    str(session["target_id"]),
                )
                original_labels = {
                    int(original_row): int(original_label)
                    for original_row, original_label in zip(
                        original.indices,
                        original.labels,
                        strict=True,
                    )
                }
                connection.execute("BEGIN IMMEDIATE")
                event_items: list[dict[str, Any]] = []
                for row_index, image_id, label in annotations:
                    previous = connection.execute(
                        "SELECT label,created_at FROM annotations WHERE session_id=? AND row_index=?",
                        (session_id, row_index),
                    ).fetchone()
                    created_at = str(previous["created_at"]) if previous else now
                    previous_label = int(previous["label"]) if previous else None
                    connection.execute(
                        "INSERT INTO annotations(session_id,row_index,image_id,label,"
                        "failed_attributes_json,suggested_failure_attribute_id,"
                        "failure_attribution_confirmed,source,created_at,updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id,row_index) "
                        "DO UPDATE SET image_id=excluded.image_id,label=excluded.label,"
                        "failed_attributes_json='[]',suggested_failure_attribute_id="
                        "excluded.suggested_failure_attribute_id,"
                        "failure_attribution_confirmed=0,source=excluded.source,"
                        "updated_at=excluded.updated_at",
                        (
                            session_id,
                            row_index,
                            image_id,
                            label,
                            "[]",
                            negative_suggestions.get(row_index),
                            0,
                            source,
                            created_at,
                            now,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO annotation_events(session_id,user_id,row_index,image_id,previous_label,label,event,source,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            session_id,
                            user_id,
                            row_index,
                            image_id,
                            previous_label,
                            label,
                            "upsert",
                            source,
                            now,
                        ),
                    )
                    event_items.append({
                        "rowIndex": row_index,
                        "imageId": image_id,
                        "previousLabel": previous_label,
                        "label": label,
                        "failedAttributeIds": [],
                        "suggestedFailedAttributeId": negative_suggestions.get(
                            row_index
                        ),
                        "failureAttributionConfirmed": False,
                    })
                connection.execute(
                    "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
                )
                connection.execute("COMMIT")
                requested_rows = {row_index for row_index, _, _ in annotations}
                all_rows = connection.execute(
                    "SELECT row_index,image_id,label,failed_attributes_json,"
                    "suggested_failure_attribute_id,failure_attribution_confirmed,"
                    "source,created_at,updated_at "
                    "FROM annotations WHERE session_id=? ORDER BY row_index",
                    (session_id,),
                ).fetchall()
                rows = [
                    row for row in all_rows if int(row["row_index"]) in requested_rows
                ]
                counts = self.annotation_counts(connection, session_id)
                try:
                    self._write_annotation_files(connection, session, user_id, {
                        "event": "bulk-upsert",
                        "source": source,
                        "count": len(event_items),
                        "items": event_items,
                        "createdAt": now,
                    })
                except OSError as error:
                    # SQLite is authoritative. A failed human-readable mirror
                    # must not turn an already committed atomic batch into a
                    # misleading HTTP failure; a later write rebuilds it.
                    print(
                        f"warning: annotation mirror write failed for {session_id}: {error}",
                        file=sys.stderr,
                    )
        return {
            "annotations": [
                self.annotation_with_supervision_json(row, original_labels)
                for row in rows
            ],
            "counts": counts,
        }

    def delete_annotation(
        self, user_id: str, session_id: str, row_index: int
    ) -> dict[str, Any]:
        now = utc_now()
        with self._session_lock(session_id):
            with self.connect() as connection:
                session = self.owned_session(connection, session_id, user_id)
                connection.execute("BEGIN IMMEDIATE")
                previous = connection.execute(
                    "SELECT * FROM annotations WHERE session_id=? AND row_index=?",
                    (session_id, row_index),
                ).fetchone()
                if previous is not None:
                    connection.execute(
                        "DELETE FROM annotations WHERE session_id=? AND row_index=?",
                        (session_id, row_index),
                    )
                    connection.execute(
                        "INSERT INTO annotation_events(session_id,user_id,row_index,image_id,previous_label,label,event,source,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            session_id,
                            user_id,
                            row_index,
                            str(previous["image_id"]),
                            int(previous["label"]),
                            None,
                            "delete",
                            "manual",
                            now,
                        ),
                    )
                    connection.execute(
                        "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
                    )
                connection.execute("COMMIT")
                counts = self.annotation_counts(connection, session_id)
                if previous is not None:
                    self._write_annotation_files(connection, session, user_id, {
                        "event": "delete",
                        "rowIndex": row_index,
                        "imageId": str(previous["image_id"]),
                        "previousLabel": int(previous["label"]),
                        "label": None,
                        "source": "manual",
                        "createdAt": now,
                    })
        return {"deleted": previous is not None, "counts": counts}

    def _session_directory(self, user_id: str, session: sqlite3.Row) -> Path:
        return (
            self.runtime_root
            / "users"
            / user_id
            / str(session["task_id"])
            / str(session["id"])
        )

    def _write_annotation_files(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        user_id: str,
        event: dict[str, Any],
    ) -> None:
        directory = self._session_directory(user_id, session)
        append_json_line(directory / "annotations.jsonl", event)
        rows = connection.execute(
            "SELECT row_index,image_id,label,failed_attributes_json,"
            "suggested_failure_attribute_id,failure_attribution_confirmed,"
            "source,created_at,updated_at "
            "FROM annotations WHERE session_id=? ORDER BY row_index",
            (session["id"],),
        ).fetchall()
        atomic_write_json(
            directory / "effective_labels.json",
            {
                "schemaVersion": 1,
                "userId": user_id,
                "sessionId": str(session["id"]),
                "taskId": str(session["task_id"]),
                "targetId": str(session["target_id"]),
                "generatedAt": utc_now(),
                "annotations": [self.annotation_json(row) for row in rows],
            },
        )

    def create_run(
        self, user_id: str, session_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        mode = str(payload.get("mode") or "")
        if mode in PROBE_REFINEMENT_RUN_MODES:
            raise ApiError(HTTPStatus.CONFLICT, "Update Probes separately, then select its snapshot for Weight Tune")
        if mode not in REQUESTABLE_RUN_MODES:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "unsupported tuning mode; prototype and label are legacy-only",
            )
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self.owned_session(connection, session_id, user_id)
            self._assert_session_idle(connection, session_id)
            requested_method = payload.get("baseMethod")
            method_value = (
                str(requested_method).strip()
                if requested_method is not None
                else str(session["base_method"])
            )
            task_spec, base_method = self.validate_target_method(
                str(session["task_id"]), str(session["target_id"]), method_value
            )
            if mode == "fusion-weight" or mode in REFINEMENT_RUN_MODES:
                base_method = "Ours-Full"
                if base_method not in task_spec.methods:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Ours-Full is unavailable")
            if mode in REFINEMENT_RUN_MODES and str(session["target_id"]) != "joint":
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "The four-way refinement comparison is defined for the Joint target",
                )
            annotations = connection.execute(
                "SELECT row_index,image_id,label,failed_attributes_json,"
                "suggested_failure_attribute_id,failure_attribution_confirmed,"
                "source,created_at,updated_at "
                "FROM annotations WHERE session_id=? AND label != 0 ORDER BY row_index",
                (session_id,),
            ).fetchall()
            # Re-validate persisted rows so a record created under an older
            # split policy can never leak Validation/Test examples into a new
            # model run.
            for annotation in annotations:
                self.validate_annotation(
                    session,
                    int(annotation["row_index"]),
                    str(annotation["image_id"]),
                    enforce_vqa_holdout=False,
                )
            validation_split = self._vqa_validation_split(
                str(session["task_id"]), str(session["target_id"])
            )
            validation_rows = {int(row) for row in validation_split.validation_indices}
            positive_count = sum(int(row["label"]) > 0 for row in annotations)
            negative_count = sum(int(row["label"]) < 0 for row in annotations)
            usable_feedback_count = positive_count + negative_count
            # Weight-only refinement can fit the original VQA supervision
            # without new corrections. Legacy tuners and Update Probes retain
            # their feedback requirements; held-out feedback is not "empty".
            original_only_weight_run = (
                mode in WEIGHT_REFINEMENT_RUN_MODES and usable_feedback_count == 0
            )
            if usable_feedback_count < 1 and not original_only_weight_run:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"{mode} tuning requires at least one positive or negative annotation",
                )
            if mode in REFINEMENT_RUN_MODES:
                relation_only_allowed = mode in WEIGHT_REFINEMENT_RUN_MODES and supports_relation_mismatch(
                    str(session["task_id"]), str(session["target_id"])
                )
                unconfirmed_negative_count = sum(
                    int(row["label"]) < 0
                    and (
                        not bool(row["failure_attribution_confirmed"])
                        or (not relation_only_allowed and not json.loads(str(row["failed_attributes_json"] or "[]")))
                    )
                    for row in annotations
                    if int(row["row_index"]) not in validation_rows
                )
                if unconfirmed_negative_count:
                    confirmation = ("failed attributes or the relation mismatch" if relation_only_allowed
                                    else "at least one failed query attribute")
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        f"Confirm {confirmation} for each "
                        f"negative correction ({unconfirmed_negative_count} remaining)",
                    )

            # Historical holdout feedback stays in the immutable snapshot for
            # audit, but cannot enter training or require failure attribution.
            trainable_feedback_count = sum(
                int(row["label"]) != 0
                and int(row["row_index"]) not in validation_rows
                for row in annotations
            )
            if trainable_feedback_count < 1 and not original_only_weight_run:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "All feedback labels belong to fixed VQA Validation; "
                    "label at least one other Development image",
                )

            # The unified refinement has four separately anchored parameter
            # groups, so its default coefficient is intentionally smaller
            # than the historical single-vector tuners. Explicit client
            # values still override either default.
            default_regularization = (
                0.05 if mode in REFINEMENT_RUN_MODES else 0.5
            )
            default_bias_regularization = (
                0.05 if mode in REFINEMENT_RUN_MODES else 0.1
            )
            regularization_lambda = self.bounded_float(
                payload.get("regularizationLambda", default_regularization),
                "regularizationLambda",
                0.0,
                1e4,
            )
            bias_regularization_eta = self.bounded_float(
                payload.get(
                    "biasRegularizationEta", default_bias_regularization
                ),
                "biasRegularizationEta",
                0.0,
                1e4,
            )
            feedback_weight = self.bounded_float(
                payload.get("feedbackWeight", 8.0),
                "feedbackWeight",
                1.0,
                100.0,
            )
            max_iterations = self.bounded_int(
                payload.get("maxIterations", 500), "maxIterations", 10, 5000
            )
            params = {
                "algorithmVersion": RUN_ALGORITHM_VERSIONS[mode],
                "supervisionPolicy": VQA_VALIDATION_SUPERVISION_POLICY,
                "vqaValidationProtocol": VQA_VALIDATION_PROTOCOL,
                "vqaValidationVersion": validation_split.audit["frozenVersion"],
                "vqaValidationManifestSha256": validation_split.audit["frozenManifestSha256"],
                "vqaValidationContract": self._vqa_validation_contract(validation_split),
                "vqaValidationFingerprint": str(
                    validation_split.audit["validationFingerprint"]
                ),
                "vqaValidationCount": int(
                    validation_split.audit["validationCount"]
                ),
                "vqaValidationPositiveCount": int(
                    validation_split.audit["validationPositiveCount"]
                ),
                "vqaValidationNegativeCount": int(
                    validation_split.audit["validationNegativeCount"]
                ),
                "probeFitProtocol": "original-vqa-minus-shared-validation-v1",
                "probeFitSeed": PROBE_VALIDATION_SEED,
                "probeFitFraction": 1.0 - PROBE_VALIDATION_FRACTION,
                "probeFitFingerprint": str(
                    validation_split.audit["fitFingerprint"]
                ),
                "probeSourceFingerprint": str(
                    validation_split.audit["sourceFingerprint"]
                ),
                "initialModelHoldoutIndependent": False,
                "referenceOnly": True,
                "evaluationLabelSource": "original-vqa-supervision",
                "feedbackHoldoutExcludedCount": int(
                    usable_feedback_count - trainable_feedback_count
                ),
                "feedbackMode": "original-only" if original_only_weight_run else "original-plus-feedback",
                "regularizationLambda": regularization_lambda,
                "biasRegularizationEta": bias_regularization_eta,
                "feedbackWeight": feedback_weight,
                "maxIterations": max_iterations,
            }
            if mode in WEIGHT_REFINEMENT_RUN_MODES and supports_relation_mismatch(
                str(session["task_id"]), str(session["target_id"])
            ):
                params["relationMismatchPolicy"] = "explicit-joint-negative-unknown-attributes-v1"
            if mode in REFINEMENT_RUN_MODES:
                import numpy as np

                learning_rate = self.bounded_float(
                    payload.get("learningRate", 0.05),
                    "learningRate",
                    1e-5,
                    1.0,
                )
                outer_iterations = self.bounded_int(
                    payload.get("outerIterations", 3),
                    "outerIterations",
                    1,
                    100,
                )
                base_context = self._refinement_base_context(
                    str(session["task_id"]), vqa_validation_version=params["vqaValidationVersion"],
                )
                from unified_initial_baseline import gate_array_encoding
                gate_encoding = gate_array_encoding(base_context)
                initial_theta = np.asarray(base_context.initial_theta, dtype=gate_encoding).copy()
                initial_temperature = np.asarray(base_context.temperature, dtype=gate_encoding).copy()
                initial_identity = getattr(base_context, "initial_baseline_identity", None)
                if initial_identity is not None:
                    # Keep the exact calibration and precision of the pinned F0.
                    params["initialBaselineIdentity"] = dict(initial_identity)
                    params["initialModelHoldoutIndependent"] = initial_identity["initialModelHoldoutIndependent"]
                    params["calibrationUsedValidation"] = initial_identity.get("calibrationUsedValidation", False)
                    params["referenceOnly"] = False
                theta_fingerprint = hashlib.sha256(
                    np.ascontiguousarray(initial_theta, dtype=gate_encoding).tobytes(
                        order="C"
                    )
                ).hexdigest()
                params.update(
                    {
                        "baseStateContract": "pcp-conjunction-holistic-base-v2",
                        "baseStateFingerprint": (
                            base_context.base_state_fingerprint
                        ),
                        "attributeIds": list(base_context.attribute_ids),
                        "attributeNames": list(base_context.attribute_names),
                        "learnerMethods": list(base_context.learner_names),
                        "embeddingMethods": list(base_context.embedding_names),
                        "parameterization": (
                            "attribute-8-simplex-softgate-product-holistic-late-fusion"
                        ),
                        "initialBeta": {
                            attribute_id: {
                                learner: 1.0 / len(base_context.learner_names)
                                for learner in base_context.learner_names
                            }
                            for attribute_id in base_context.attribute_ids
                        },
                        "initialGamma": {
                            attribute_id: 1.0
                            for attribute_id in base_context.attribute_ids
                        },
                        "initialEmbeddingWeights": {
                            method: 1.0 / len(base_context.embedding_names)
                            for method in base_context.embedding_names
                        },
                        "lambda0": REFINEMENT_LAMBDA_0,
                        "gammaMin": REFINEMENT_GAMMA_MIN,
                        "gammaMax": REFINEMENT_GAMMA_MAX,
                        "rhoAttribute": 1.0,
                        "rhoJoint": 1.0,
                        "alphaBeta": regularization_lambda,
                        "alphaGamma": regularization_lambda,
                        "alphaEta": regularization_lambda,
                        "alphaLambda": bias_regularization_eta,
                        "gateCalibrationPolicy": FIXED_GATE_POLICY,
                        "thetaCalibration": FIXED_GATE_POLICY,
                        "temperaturePolicy": "fixed-initial-snapshot",
                        "normalizationPolicy": REFINEMENT_NORMALIZATION_POLICY,
                        "normalizationContract": dict(base_context.normalization_audit),
                        "initialTheta": [
                            float(value) for value in initial_theta
                        ],
                        "initialThetaFingerprint": theta_fingerprint,
                        "gateArrayEncoding": gate_encoding,
                        "initialTemperature": initial_temperature.tolist(),
                        "initialTemperatureFingerprint": hashlib.sha256(
                            np.ascontiguousarray(initial_temperature, dtype=gate_encoding).tobytes()
                        ).hexdigest(),
                        "trainingStrategy": (
                            "staged" if mode.endswith("_staged") else "joint"
                        ),
                        "maxIterationsMeaning": "total-optimizer-steps",
                        "learningRate": learning_rate,
                        "outerIterations": outer_iterations,
                        "randomSeed": 0,
                    }
                )
                self._refinement_embedding_methods(mode, params)
                if initial_identity is not None:
                    params["initialCalibrationSource"] = initial_identity["calibration"]
                params["vqaValidationByTarget"] = {
                    validation_target: self._vqa_validation_contract(split)
                    for validation_target in (
                        *base_context.attribute_ids,
                        "joint",
                    )
                    for split in (
                        self._vqa_validation_split(
                            str(session["task_id"]), validation_target,
                            params["vqaValidationVersion"],
                        ),
                    )
                }
                source = payload.get("probeSource", "original")
                if not isinstance(source, str) or source not in {"original", "updated"}:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "probeSource must be original or updated")
                params["probeSource"] = source
                params["probeSelectionProtocol"] = "explicit-frozen-probe-update-v1"
                if source == "updated":
                    update_id = self.required_string(payload, "probeUpdateId")
                    try:
                        _, audit, request = self._load_selected_probe_update(
                            user_id, session_id, str(session["task_id"]), update_id, base_context,
                        )
                    except (OSError, ValueError, RuntimeError) as error:
                        raise ApiError(HTTPStatus.CONFLICT, str(error)) from error
                    params["probeUpdateId"] = update_id
                    params["nativeProbeRequest"] = request
                    params["probeSnapshotFingerprint"] = audit["snapshotFingerprint"]
                elif payload.get("probeUpdateId") is not None:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Original probes cannot carry an updated snapshot ID")
            elif mode == "fusion-weight":
                shared_regularization_lambda = self.bounded_float(
                    payload.get("sharedRegularizationLambda", 0.25),
                    "sharedRegularizationLambda",
                    0.0,
                    1e4,
                )
                joint_regularization_lambda = self.bounded_float(
                    payload.get("jointRegularizationLambda", 0.5),
                    "jointRegularizationLambda",
                    0.0,
                    1e4,
                )
                attribute_ids = tuple(
                    str(target.get("id"))
                    for target in task_spec.manifest.get("retrievalTargets", [])
                    if target.get("kind") == "attribute"
                )
                target_id = str(session["target_id"])
                active_attribute_ids = (
                    (target_id,) if target_id in attribute_ids else attribute_ids
                )
                active_joint_attribute_ids = (
                    attribute_ids if target_id not in attribute_ids else ()
                )
                params["attributeIds"] = list(attribute_ids)
                params["activeAttributeIds"] = list(active_attribute_ids)
                params["activeJointAttributeIds"] = list(
                    active_joint_attribute_ids
                )
                params["sharedRegularizationLambda"] = shared_regularization_lambda
                params["jointRegularizationLambda"] = joint_regularization_lambda
                if params["algorithmVersion"] == FUSION_WEIGHT_ALGORITHM_V4:
                    params["parameterization"] = (
                        "per-attribute-13-simplex-plus-attribute-simplex"
                    )
                    params["methodIds"] = list(RANK_FUSION_METHODS)
                    params["initialMethodWeightsByAttribute"] = {
                        attribute_id: {
                            method: 1.0 / len(RANK_FUSION_METHODS)
                            for method in RANK_FUSION_METHODS
                        }
                        for attribute_id in attribute_ids
                    }
                    params["jointAggregation"] = "weighted-mean"
                    params["jointWeightConstraint"] = "nonnegative-sum-one"
                    params["initialAttributeWeights"] = {
                        attribute_id: 1.0 / len(attribute_ids)
                        for attribute_id in attribute_ids
                    }
                elif params["algorithmVersion"] == FUSION_WEIGHT_ALGORITHM_V3:
                    # This construction path is retained for regression tests and
                    # historical replay. New production runs always select v4.
                    params["parameterization"] = (
                        "per-attribute-simplex-plus-joint-product-exponents"
                    )
                    params["learnerMethods"] = list(WEIGHTED_FUSION_LEARNERS)
                    params["initialWeightsByAttribute"] = {
                        attribute_id: {
                            learner: 1.0 / len(WEIGHTED_FUSION_LEARNERS)
                            for learner in WEIGHTED_FUSION_LEARNERS
                        }
                        for attribute_id in attribute_ids
                    }
                    params["jointAggregation"] = "product"
                    params["jointWeightConstraint"] = "nonnegative-mean-one"
                    params["initialAttributeWeights"] = {
                        attribute_id: 1.0 for attribute_id in attribute_ids
                    }
                else:
                    raise RuntimeError(
                        "Unsupported creatable fusion-weight algorithm: "
                        f"{params['algorithmVersion']}"
                    )
            elif mode == "residual":
                params["featureMethods"] = list(WEIGHTED_FUSION_LEARNERS)
            else:
                raise RuntimeError(f"Unsupported creatable tuning mode: {mode}")
            snapshot = {
                "schemaVersion": 9 if mode in REFINEMENT_RUN_MODES else 3,
                "sessionId": session_id,
                "taskId": str(session["task_id"]),
                "targetId": str(session["target_id"]),
                "baseMethod": base_method,
                "mode": mode,
                "algorithmVersion": RUN_ALGORITHM_VERSIONS[mode],
                "supervisionPolicy": VQA_VALIDATION_SUPERVISION_POLICY,
                "params": params,
                "evaluationScope": "vqa-validation",
                "createdAt": now,
                "annotations": [self.annotation_json(row) for row in annotations],
            }
            snapshot_bytes = (json_dumps(snapshot) + "\n").encode("utf-8")
            snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
            annotation_sha256 = self.annotation_state_sha256(annotations)
            run_id = new_id("run")
            run_directory = self._session_directory(user_id, session) / "runs" / run_id
            artifact_relpath = run_directory.relative_to(self.runtime_root).as_posix()
            atomic_write_bytes(run_directory / "labels_snapshot.json", snapshot_bytes)
            connection.execute(
                "INSERT INTO model_runs(id,session_id,user_id,mode,base_method,evaluation_scope,status,params_json,"
                "annotation_sha256,snapshot_sha256,annotation_count,positive_count,negative_count,"
                "artifact_relpath,created_at) "
                "VALUES(?,?,?,?,?,'vqa-validation','queued',?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    session_id,
                    user_id,
                    mode,
                    base_method,
                    json_dumps(params),
                    annotation_sha256,
                    snapshot_sha256,
                    len(annotations),
                    positive_count,
                    negative_count,
                    artifact_relpath,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM model_runs WHERE id=?", (run_id,)
            ).fetchone()
        self.run_queue.put(run_id)
        return self.run_json(row)

    @staticmethod
    def bounded_float(value: Any, name: str, lower: float, upper: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be numeric") from exc
        if not math.isfinite(result) or not lower <= result <= upper:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{name} must be between {lower:g} and {upper:g}",
            )
        return result

    @staticmethod
    def bounded_int(value: Any, name: str, lower: int, upper: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be an integer")
        if not lower <= value <= upper:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{name} must be between {lower} and {upper}",
            )
        return value

    def owned_run(self, run_id: str, user_id: str) -> sqlite3.Row:
        if not SAFE_ID_PATTERN.fullmatch(run_id):
            raise ApiError(HTTPStatus.NOT_FOUND, "Run not found")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_runs WHERE id=? AND user_id=?", (run_id, user_id)
            ).fetchone()
        if row is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "Run not found")
        return row

    def run_json(self, row: sqlite3.Row) -> dict[str, Any]:
        row_keys = set(row.keys())
        evaluation_scope = (
            str(row["evaluation_scope"])
            if "evaluation_scope" in row_keys and row["evaluation_scope"]
            else "test"
        )
        result: dict[str, Any] = {
            "id": str(row["id"]),
            "sessionId": str(row["session_id"]),
            "mode": str(row["mode"]),
            "baseMethod": str(row["base_method"]),
            "beforeMethod": (
                "Unified base F⁽⁰⁾"
                if str(row["mode"]) in REFINEMENT_RUN_MODES
                else str(row["base_method"])
            ),
            "evaluationScope": evaluation_scope,
            "status": str(row["status"]),
            "annotationCount": int(row["annotation_count"]),
            "positiveCount": int(row["positive_count"]),
            "negativeCount": int(row["negative_count"]),
            "createdAt": str(row["created_at"]),
            "startedAt": row["started_at"],
            "finishedAt": row["finished_at"],
            "error": row["error"],
        }
        mode = str(row["mode"])
        params = json.loads(str(row["params_json"] or "{}"))
        if not isinstance(params, dict):
            params = {}
        result["algorithmVersion"] = str(
            params.get("algorithmVersion") or RUN_ALGORITHM_VERSIONS.get(mode, mode)
        )
        if params.get("supervisionPolicy"):
            result["supervisionPolicy"] = str(params["supervisionPolicy"])
        if params.get("cleanValidationVersion"):
            result["cleanValidationVersion"] = params["cleanValidationVersion"]
        if params.get("vqaValidationVersion"):
            result.update({
                "vqaValidationVersion": params["vqaValidationVersion"],
                "vqaValidationManifestSha256": params["vqaValidationManifestSha256"],
                "evaluationLabelSource": "original-vqa-supervision",
                "initialModelHoldoutIndependent": bool(params.get("initialModelHoldoutIndependent", False)),
                "referenceOnly": bool(params.get("referenceOnly", True)),
            })
        result["legacyMode"] = mode in LEGACY_RUN_MODES
        if mode in REFINEMENT_RUN_MODES:
            result["attributeIds"] = params.get("attributeIds", [])
            result["embeddingMethods"] = params.get("embeddingMethods", [])
            result["probeSource"] = params.get("probeSource", "updated" if mode in PROBE_REFINEMENT_RUN_MODES else "original")
            result["probeUpdateId"] = params.get("probeUpdateId")
            result["probeSnapshotId"] = (params.get("nativeProbeRequest") or {}).get("snapshotId")
        if row["before_json"]:
            result["before"] = json.loads(row["before_json"])
        if row["after_json"]:
            result["after"] = json.loads(row["after_json"])
        if "before" in result and "after" in result:
            result["deltaAp"] = float(result["after"]["ap"] - result["before"]["ap"])
            split_audit = result["after"].get("splitAudit")
            if isinstance(split_audit, dict):
                result["splitAudit"] = split_audit
        if str(row["status"]) == "succeeded":
            # Test is optional post-training diagnostics stored with this run,
            # never a replacement for its persisted Val before/after metrics.
            # In particular, reading an old run must not evaluate or backfill it.
            metrics_path = (
                self.runtime_root / str(row["artifact_relpath"]) / "metrics.json"
            ).resolve()
            if metrics_path.is_relative_to(self.runtime_root):
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    metrics = None
                if isinstance(metrics, dict) and metrics.get("runId") == str(row["id"]):
                    if isinstance(metrics.get("testEvaluation"), dict):
                        result["testEvaluation"] = metrics["testEvaluation"]
                    elif isinstance(metrics.get("testEvaluationError"), str):
                        result["testEvaluationError"] = metrics["testEvaluationError"]
            base = f"{API_PREFIX}/runs/{row['id']}"
            result["scoresUrl"] = f"{base}/tuned-scores.f32"
            result["ranksUrl"] = f"{base}/tuned-ranks.f32"
            # Schema-4/v1 refinement artifacts remain usable through their
            # original score/rank URLs, but their coupled-gamma contract must
            # not be reinterpreted as the schema-5 independent-gamma PCP.
            if (
                mode in REFINEMENT_RUN_MODES
                and params.get("algorithmVersion") in {
                    RUN_ALGORITHM_VERSIONS[mode],
                    LEGACY_WEIGHT_REFINEMENT_ALGORITHMS.get(mode),
                    PRE_ISOLATION_WEIGHT_REFINEMENT_ALGORITHMS.get(mode),
                    PRE_FIXED_GATE_REFINEMENT_ALGORITHMS.get(mode),
                }
                and "gammaMax" in params
            ):
                result["visualizationUrl"] = f"{base}/refinement-visualization"
                result["clusterUrl"] = f"{base}/refinement-clusters"
        return result

    def _post_training_test_evaluation(
        self,
        task: TaskSpec,
        bundle: BundleData,
        target_id: str,
        baseline_ranks: Any,
        tuned_ranks: Any,
        before_method: str,
    ) -> dict[str, Any]:
        """Evaluate frozen outputs only; never feed Test labels back to fitting."""

        import numpy as np

        test_mask = np.frombuffer(bundle.test_mask, dtype=np.uint8)
        if test_mask.size != task.row_count or not np.isin(test_mask, (0, 1)).all():
            raise ValueError("Frozen Test mask is invalid")
        test_rows = np.flatnonzero(test_mask)
        if not test_rows.size:
            raise ValueError("Frozen Test has no evaluation images")
        ground_truth = self._task_ground_truth(task.task_id)
        target_index = task.target_ids.index(target_id)
        test_truth = ground_truth[test_rows, target_index]
        if not np.count_nonzero(test_truth):
            raise ValueError("Frozen Test has no positive examples for this target")
        rank_columns = []
        for ranks in (baseline_ranks, tuned_ranks):
            values = np.asarray(ranks, dtype=np.float32).reshape(-1)
            if values.size != task.row_count or not np.isfinite(values[test_rows]).all():
                raise ValueError("Frozen Test ranking output is invalid")
            rank_columns.append(values[test_rows])
        return {
            "protocol": "post-training-frozen-test-v1",
            "evaluationScope": "test",
            "before": retrieval_metrics(test_truth, rank_columns[0]),
            "after": retrieval_metrics(test_truth, rank_columns[1]),
            "targetId": target_id,
            "beforeMethod": before_method,
            "testMaskSha256": hashlib.sha256(bundle.test_mask).hexdigest(),
            "groundTruthSha256": hashlib.sha256(
                np.asarray(ground_truth, dtype=np.uint8).tobytes(order="C")
            ).hexdigest(),
        }

    def artifact(self, run: sqlite3.Row, filename: str) -> Path:
        if filename not in {"tuned-ranks.f32", "tuned-scores.f32"}:
            raise ApiError(HTTPStatus.NOT_FOUND, "Artifact not found")
        path = (self.runtime_root / str(run["artifact_relpath"]) / filename).resolve()
        if not path.is_relative_to(self.runtime_root) or not path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "Artifact not found")
        return path

    def start_worker(self) -> None:
        if self.worker_thread is not None:
            return
        with self.connect() as connection:
            queued = [
                str(row["id"])
                for row in connection.execute(
                    "SELECT id,created_at FROM model_runs WHERE status='queued' UNION ALL "
                    "SELECT 'probe-update:' || id,created_at FROM probe_updates WHERE status='queued' ORDER BY created_at"
                )
            ]
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="pcp-tuning-worker",
            daemon=True,
        )
        self.worker_thread.start()
        for run_id in queued:
            self.run_queue.put(run_id)

    def stop_worker(self) -> None:
        if self.worker_thread is None:
            return
        self.run_queue.put(None)
        self.worker_thread.join(timeout=10)
        self.worker_thread = None

    def _worker_loop(self) -> None:
        while True:
            run_id = self.run_queue.get()
            try:
                if run_id is None:
                    return
                if run_id.startswith("probe-update:"):
                    self._run_probe_update_job(run_id.removeprefix("probe-update:"))
                else:
                    self._run_job(run_id)
            finally:
                self.run_queue.task_done()

    def _claim_run(self, run_id: str) -> sqlite3.Row | None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE model_runs SET status='running',started_at=?,error=NULL "
                "WHERE id=? AND status='queued'",
                (now, run_id),
            ).rowcount
            connection.execute("COMMIT")
            if changed != 1:
                return None
            return connection.execute(
                "SELECT r.*,s.task_id,s.target_id FROM model_runs r "
                "JOIN sessions s ON s.id=r.session_id WHERE r.id=?",
                (run_id,),
            ).fetchone()

    def _run_job(self, run_id: str) -> None:
        run = self._claim_run(run_id)
        if run is None:
            return
        try:
            before, after = self._execute_model_run(run)
            with self.connect() as connection:
                connection.execute(
                    "UPDATE model_runs SET status='succeeded',before_json=?,after_json=?,finished_at=? "
                    "WHERE id=? AND status='running'",
                    (json_dumps(before), json_dumps(after), utc_now(), run_id),
                )
        except Exception as error:  # worker errors must not stop later jobs
            traceback.print_exc()
            message = f"{type(error).__name__}: {error}"[:2000]
            with self.connect() as connection:
                connection.execute(
                    "UPDATE model_runs SET status='failed',error=?,finished_at=? "
                    "WHERE id=? AND status='running'",
                    (message, utc_now(), run_id),
                )

    def _read_verified_run_snapshot(
        self, run: sqlite3.Row, run_directory: Path
    ) -> tuple[dict[str, Any], str]:
        """Read one immutable label snapshot and verify both hash domains.

        ``annotation_sha256`` identifies the effective non-zero label state and
        is used to mark a run stale when live annotations change.
        ``snapshot_sha256`` identifies the exact frozen JSON artifact.  Rows
        created before the two fields were separated remain compatible: some
        legacy rows stored the whole-file hash in ``annotation_sha256``, while
        early schema-3 rows stored the label-state hash there.
        """
        snapshot_path = run_directory / "labels_snapshot.json"
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
        row_keys = set(run.keys())
        expected_snapshot_sha256 = (
            str(run["snapshot_sha256"])
            if "snapshot_sha256" in row_keys and run["snapshot_sha256"]
            else None
        )
        if expected_snapshot_sha256 is not None and not hmac.compare_digest(
            snapshot_sha256, expected_snapshot_sha256
        ):
            raise RuntimeError("Annotation snapshot file checksum mismatch")

        try:
            snapshot = json.loads(snapshot_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Annotation snapshot is not valid JSON") from error
        if not isinstance(snapshot, dict) or not isinstance(
            snapshot.get("annotations"), list
        ):
            raise RuntimeError("Annotation snapshot has an invalid annotation contract")

        expected_annotation_sha256 = str(run["annotation_sha256"])
        annotation_sha256 = self.annotation_state_sha256(snapshot["annotations"])
        legacy_whole_file_hash = (
            expected_snapshot_sha256 is None
            and hmac.compare_digest(snapshot_sha256, expected_annotation_sha256)
        )
        if not legacy_whole_file_hash and not hmac.compare_digest(
            annotation_sha256, expected_annotation_sha256
        ):
            raise RuntimeError("Annotation snapshot checksum mismatch")
        return snapshot, snapshot_sha256

    def _execute_model_run(self, run: sqlite3.Row) -> tuple[dict[str, Any], dict[str, Any]]:
        import numpy as np

        task = self.task(str(run["task_id"]))
        target_id = str(run["target_id"])
        target_index = task.target_ids.index(target_id)
        method_index = task.methods.index(str(run["base_method"]))
        run_directory = (self.runtime_root / str(run["artifact_relpath"])).resolve()
        snapshot, snapshot_sha256 = self._read_verified_run_snapshot(
            run, run_directory
        )

        mode = str(run["mode"])
        if str(snapshot.get("mode")) != mode:
            raise RuntimeError("Annotation snapshot mode does not match the queued run")
        if str(snapshot.get("taskId")) != task.task_id:
            raise RuntimeError("Annotation snapshot task does not match the queued run")
        if str(snapshot.get("targetId")) != target_id:
            raise RuntimeError("Annotation snapshot target does not match the queued run")
        if str(snapshot.get("baseMethod")) != str(run["base_method"]):
            raise RuntimeError("Annotation snapshot base method does not match the queued run")
        snapshot_scope = snapshot.get("evaluationScope")
        if snapshot_scope is not None and str(snapshot_scope) != str(
            run["evaluation_scope"]
        ):
            raise RuntimeError(
                "Annotation snapshot evaluation scope does not match the queued run"
            )
        snapshot_algorithm_version = str(snapshot.get("algorithmVersion") or "")
        if mode in REQUESTABLE_RUN_MODES:
            expected_schema = (
                5 if mode in WEIGHT_REFINEMENT_RUN_MODES and snapshot_algorithm_version
                == LEGACY_WEIGHT_REFINEMENT_ALGORITHMS[mode]
                else 6 if mode in WEIGHT_REFINEMENT_RUN_MODES and snapshot_algorithm_version
                == PRE_ISOLATION_WEIGHT_REFINEMENT_ALGORITHMS[mode]
                else 8 if mode in PROBE_REFINEMENT_RUN_MODES and snapshot_algorithm_version
                == PRE_FIXED_GATE_REFINEMENT_ALGORITHMS[mode]
                else 7 if mode in WEIGHT_REFINEMENT_RUN_MODES and snapshot_algorithm_version
                == PRE_FIXED_GATE_REFINEMENT_ALGORITHMS[mode]
                else 9 if mode in REFINEMENT_RUN_MODES else 3
            )
            if int(snapshot.get("schemaVersion", -1)) != expected_schema:
                raise RuntimeError(
                    f"Current tuning run requires a schema-{expected_schema} snapshot"
                )
            supported_versions = (
                SUPPORTED_FUSION_WEIGHT_ALGORITHMS
                if mode == "fusion-weight"
                else {RUN_ALGORITHM_VERSIONS[mode], LEGACY_WEIGHT_REFINEMENT_ALGORITHMS[mode], PRE_ISOLATION_WEIGHT_REFINEMENT_ALGORITHMS[mode], PRE_FIXED_GATE_REFINEMENT_ALGORITHMS[mode]}
                if mode in WEIGHT_REFINEMENT_RUN_MODES
                else {RUN_ALGORITHM_VERSIONS[mode], PRE_FIXED_GATE_REFINEMENT_ALGORITHMS[mode]}
                if mode in PROBE_REFINEMENT_RUN_MODES
                else {RUN_ALGORITHM_VERSIONS[mode]}
            )
            if snapshot_algorithm_version not in supported_versions:
                raise RuntimeError("Annotation snapshot algorithm version mismatch")
            if snapshot.get("supervisionPolicy") not in SUPPORTED_SUPERVISION_POLICIES:
                raise RuntimeError("Annotation snapshot supervision policy mismatch")

        annotations = snapshot["annotations"]
        indices = np.asarray([int(row["rowIndex"]) for row in annotations], dtype=np.int64)
        labels = np.asarray([int(row["label"]) for row in annotations], dtype=np.int8)
        bundle = self.bundle(task.task_id)
        for row in annotations:
            # Old workers retain their old DG contract. New VQA workers keep
            # historical holdout rows for audit, then explicitly exclude them.
            self.validate_annotation(
                run, int(row["rowIndex"]), str(row["imageId"]),
                enforce_vqa_holdout=False,
            )

        params = json.loads(str(run["params_json"]))
        if (
            mode in REQUESTABLE_RUN_MODES
            and str(params.get("algorithmVersion") or "") != snapshot_algorithm_version
        ):
            raise RuntimeError("Stored run parameters do not match the snapshot algorithm")
        if mode in REQUESTABLE_RUN_MODES and snapshot.get("params") != params:
            raise RuntimeError("Stored run parameters do not match the immutable snapshot")
        embedding_meta: dict[str, Any] | None = None
        model_summary: dict[str, Any] = {}
        supervision_audit: dict[str, Any] | None = None
        probe_validation_split: Any | None = None
        tuned_ranks: Any | None = None
        comparison_baseline_ranks: Any | None = None
        if mode in LEGACY_RUN_MODES:
            embeddings, base_prototype, embedding_meta = self._load_embeddings(
                task, bundle.image_ids
            )
        if mode == "prototype":
            tuned_scores, tuned_prototype = prototype_tuned_scores(
                embeddings,
                base_prototype,
                indices,
                labels,
                alpha=float(params["alpha"]),
                beta=float(params["beta"]),
            )
            atomic_savez(
                run_directory / "model.npz",
                prototype=tuned_prototype,
                alpha=np.float32(params["alpha"]),
                beta=np.float32(params["beta"]),
            )
        elif mode == "label":
            tuned_scores, model_state = label_tuned_scores(
                embeddings,
                indices,
                labels,
                regularization_c=float(params["regularizationC"]),
            )
            atomic_savez(run_directory / "model.npz", **model_state)
        elif mode in REQUESTABLE_RUN_MODES:
            from tuning_models import (
                fit_attribute_and_joint_fusion_weights,
                fit_attribute_fusion_weights,
                fit_fusion_weights,
                fit_residual,
                residual_probabilities,
                stable_logit,
                stable_sigmoid,
            )
            context = self._fusion_context_for_algorithm(
                task.task_id,
                snapshot_algorithm_version,
            )
            target_attribute = (
                None if context is None else context.target_attributes[target_index]
            )
            supervision_policy = str(snapshot.get("supervisionPolicy") or "")
            if supervision_policy in {
                SUPERVISION_POLICY,
                PROBE_VALIDATION_SUPERVISION_POLICY,
                VQA_VALIDATION_SUPERVISION_POLICY,
            }:
                use_clean_validation = supervision_policy == SUPERVISION_POLICY
                use_vqa_validation = supervision_policy == VQA_VALIDATION_SUPERVISION_POLICY
                if use_vqa_validation and (
                    str(run["evaluation_scope"]) != "vqa-validation"
                    or not params.get("vqaValidationVersion")
                    or not params.get("vqaValidationManifestSha256")
                ):
                    raise RuntimeError("Fixed VQA Validation identity is missing or scope mismatched")
                (
                    supervision_indices,
                    supervision_labels,
                    supervision_weights,
                    supervision_audit,
                    probe_validation_split,
                ) = self._prepare_probe_tuning_supervision(
                    task.task_id,
                    target_id,
                    indices,
                    labels,
                    feedback_weight=float(params["feedbackWeight"]),
                    clean_validation=use_clean_validation,
                    clean_validation_version=params.get("cleanValidationVersion"),
                    vqa_validation=use_vqa_validation,
                    vqa_validation_version=params.get("vqaValidationVersion"),
                )
                if use_vqa_validation:
                    if (
                        probe_validation_split.audit.get("frozenVersion") != params["vqaValidationVersion"]
                        or probe_validation_split.audit.get("frozenManifestSha256") != params["vqaValidationManifestSha256"]
                    ):
                        raise RuntimeError("Frozen VQA Validation version/checksum changed after queuing")
                    expected_validation_contract = params.get("vqaValidationContract")
                    observed_validation_contract = self._vqa_validation_contract(probe_validation_split)
                    mismatch_message = "Stored VQA Validation contract changed after queuing"
                elif use_clean_validation:
                    if params.get("cleanValidationVersion") and (
                        probe_validation_split.audit.get("frozenVersion")
                        != params["cleanValidationVersion"]
                        or probe_validation_split.audit.get("frozenManifestSha256")
                        != params.get("cleanValidationManifestSha256")
                    ):
                        raise RuntimeError("Frozen Clean Validation version/checksum changed after queuing")
                    expected_validation_contract = {
                        "protocol": str(
                            params.get("cleanValidationProtocol") or ""
                        ),
                        "validationFingerprint": str(
                            params.get("cleanValidationFingerprint") or ""
                        ),
                        "validationCount": int(
                            params.get("cleanValidationCount", -1)
                        ),
                        "positiveCount": int(
                            params.get("cleanValidationPositiveCount", -1)
                        ),
                        "negativeCount": int(
                            params.get("cleanValidationNegativeCount", -1)
                        ),
                        "fitProtocol": str(params.get("probeFitProtocol") or ""),
                        "fitSeed": int(params.get("probeFitSeed", -1)),
                        "fitFraction": float(params.get("probeFitFraction", -1.0)),
                        "fitFingerprint": str(
                            params.get("probeFitFingerprint") or ""
                        ),
                        "sourceFingerprint": str(
                            params.get("probeSourceFingerprint") or ""
                        ),
                        "excludedOriginalFingerprint": str(
                            params.get("excludedOriginalSupervisionFingerprint")
                            or ""
                        ),
                        "excludedFeedbackFingerprint": str(
                            params.get("excludedHistoricalFeedbackFingerprint")
                            or ""
                        ),
                    }
                    observed_validation_contract = {
                        "protocol": str(
                            probe_validation_split.audit["protocol"]
                        ),
                        "validationFingerprint": str(
                            probe_validation_split.audit["validationFingerprint"]
                        ),
                        "validationCount": int(
                            probe_validation_split.audit["validationCount"]
                        ),
                        "positiveCount": int(
                            probe_validation_split.audit[
                                "validationPositiveCount"
                            ]
                        ),
                        "negativeCount": int(
                            probe_validation_split.audit[
                                "validationNegativeCount"
                            ]
                        ),
                        "fitProtocol": str(
                            probe_validation_split.audit["fitProtocol"]
                        ),
                        "fitSeed": int(probe_validation_split.audit["fitSeed"]),
                        "fitFraction": float(
                            probe_validation_split.audit["fitFraction"]
                        ),
                        "fitFingerprint": str(
                            probe_validation_split.audit["fitFingerprint"]
                        ),
                        "sourceFingerprint": str(
                            probe_validation_split.audit["sourceFingerprint"]
                        ),
                        "excludedOriginalFingerprint": str(
                            probe_validation_split.audit[
                                "excludedOriginalSupervisionFingerprint"
                            ]
                        ),
                        "excludedFeedbackFingerprint": str(
                            probe_validation_split.audit[
                                "excludedHistoricalFeedbackFingerprint"
                            ]
                        ),
                    }
                    mismatch_message = (
                        "Stored Clean Validation contract does not match the "
                        "frozen label-isolated split"
                    )
                else:
                    expected_validation_contract = {
                        "protocol": str(
                            params.get("probeValidationProtocol") or ""
                        ),
                        "seed": int(params.get("probeValidationSeed", -1)),
                        "validationFraction": float(
                            params.get("probeValidationFraction", -1.0)
                        ),
                        "validationFingerprint": str(
                            params.get("probeValidationFingerprint") or ""
                        ),
                        "fitFingerprint": str(
                            params.get("probeFitFingerprint") or ""
                        ),
                        "sourceFingerprint": str(
                            params.get("probeSourceFingerprint") or ""
                        ),
                        "replayKind": str(
                            params.get("probeValidationReplayKind") or ""
                        ),
                        "validationCount": int(
                            params.get("probeValidationCount", -1)
                        ),
                    }
                    observed_validation_contract = {
                        "protocol": str(
                            probe_validation_split.audit["protocol"]
                        ),
                        "seed": int(probe_validation_split.audit["seed"]),
                        "validationFraction": float(
                            probe_validation_split.audit["validationFraction"]
                        ),
                        "validationFingerprint": str(
                            probe_validation_split.audit["validationFingerprint"]
                        ),
                        "fitFingerprint": str(
                            probe_validation_split.audit["fitFingerprint"]
                        ),
                        "sourceFingerprint": str(
                            probe_validation_split.audit["sourceFingerprint"]
                        ),
                        "replayKind": str(
                            probe_validation_split.audit["replayKind"]
                        ),
                        "validationCount": int(
                            probe_validation_split.audit["validationCount"]
                        ),
                    }
                    mismatch_message = (
                        "Stored Probe Validation contract does not match the "
                        "frozen split"
                    )
                if expected_validation_contract != observed_validation_contract:
                    raise RuntimeError(mismatch_message)
                atomic_write_json(
                    run_directory
                    / (
                        "vqa_validation.json" if use_vqa_validation else "clean_validation.json"
                        if use_clean_validation
                        else "probe_validation.json"
                    ),
                    {
                        "schemaVersion": 3 if use_vqa_validation else (2 if use_clean_validation else 1),
                        "audit": probe_validation_split.audit,
                        "fit": [
                            {"rowIndex": int(row), "label": int(label)}
                            for row, label in zip(
                                probe_validation_split.fit_indices,
                                probe_validation_split.fit_labels,
                                strict=True,
                            )
                        ],
                        "validation": [
                            {"rowIndex": int(row), "label": int(label)}
                            for row, label in zip(
                                probe_validation_split.validation_indices,
                                probe_validation_split.validation_labels,
                                strict=True,
                            )
                        ],
                    },
                )
            else:
                original = self._original_development_supervision(task.task_id, target_id)
                (
                    supervision_indices,
                    supervision_labels,
                    supervision_weights,
                    merge_audit,
                ) = self._merge_tuning_supervision(
                    original.indices,
                    original.labels,
                    indices,
                    labels,
                    feedback_weight=float(params["feedbackWeight"]),
                )
                supervision_audit = {**original.audit, **merge_audit}
            regularization_lambda = float(params["regularizationLambda"])
            bias_regularization_eta = float(params["biasRegularizationEta"])
            max_iterations = int(params["maxIterations"])
            if mode in REFINEMENT_RUN_MODES:
                from tuning_models import (
                    fit_unified_weight_refinement,
                    recalibrate_unified_theta,
                    unified_weight_scores,
                )

                base = self._refinement_base_for_run(task.task_id, mode, params)
                fixed_gates = snapshot_algorithm_version == RUN_ALGORITHM_VERSIONS[mode]
                if fixed_gates and (
                    params.get("gateCalibrationPolicy") != FIXED_GATE_POLICY
                    or params.get("thetaCalibration") != FIXED_GATE_POLICY
                    or params.get("temperaturePolicy") != "fixed-initial-snapshot"
                ):
                    raise RuntimeError("Fixed-gate refinement policy changed after queuing")
                if params.get("baseStateContract") != (
                    "pcp-conjunction-holistic-base-v2" if snapshot_algorithm_version in {
                        RUN_ALGORITHM_VERSIONS[mode], PRE_FIXED_GATE_REFINEMENT_ALGORITHMS[mode]}
                    else "pcp-conjunction-holistic-base-v1"
                ):
                    raise RuntimeError("Refinement base-state contract mismatch")
                if params.get("baseStateFingerprint") != (
                    base.base_state_fingerprint
                ):
                    raise RuntimeError(
                        "Refinement base-state fingerprint changed after queuing"
                    )
                if tuple(params.get("attributeIds") or ()) != base.attribute_ids:
                    raise RuntimeError("Refinement attribute order changed after queuing")
                if tuple(params.get("learnerMethods") or ()) != base.learner_names:
                    raise RuntimeError("Refinement learner order changed after queuing")
                if tuple(params.get("embeddingMethods") or ()) != base.embedding_names:
                    raise RuntimeError("Refinement embedding order changed after queuing")

                prepared = self._prepare_weight_refinement_supervision(
                    task.task_id,
                    base,
                    annotations,
                    feedback_weight=float(params["feedbackWeight"]),
                    clean_validation=(
                        supervision_policy == SUPERVISION_POLICY
                    ),
                    clean_validation_version=params.get("cleanValidationVersion"),
                    vqa_validation=(supervision_policy == VQA_VALIDATION_SUPERVISION_POLICY),
                    vqa_validation_version=params.get("vqaValidationVersion"),
                )
                if supervision_policy == VQA_VALIDATION_SUPERVISION_POLICY:
                    observed_vqa_contracts = {
                        validation_target: self._vqa_validation_contract(
                            self._vqa_validation_split(
                                task.task_id, validation_target, params["vqaValidationVersion"]
                            )
                        )
                        for validation_target in (*base.attribute_ids, "joint")
                    }
                    if params.get("vqaValidationByTarget") != observed_vqa_contracts:
                        raise RuntimeError("A per-target VQA Validation contract changed after queuing")
                if supervision_policy == SUPERVISION_POLICY:
                    split_audits = {
                        **{
                            attribute_id: prepared["audit"]["attributeTargets"][
                                attribute_id
                            ]
                            for attribute_id in base.attribute_ids
                        },
                        "joint": prepared["jointSplit"].audit,
                    }
                    observed_target_contracts = {
                        validation_target: {
                            "fingerprint": str(
                                audit["validationFingerprint"]
                            ),
                            "count": int(audit["validationCount"]),
                            "positiveCount": int(
                                audit["validationPositiveCount"]
                            ),
                            "negativeCount": int(
                                audit["validationNegativeCount"]
                            ),
                            "fitFingerprint": str(audit["fitFingerprint"]),
                            "sourceFingerprint": str(
                                audit["sourceFingerprint"]
                            ),
                        }
                        for validation_target, audit in split_audits.items()
                    }
                    if (
                        params.get("cleanValidationByTarget")
                        != observed_target_contracts
                    ):
                        raise RuntimeError(
                            "A per-target Clean Validation contract changed "
                            "after queuing"
                        )
                training = self._materialize_refinement_training_arrays(
                    base, prepared
                )
                probe_validation_split = prepared["jointSplit"]
                supervision_audit = dict(prepared["audit"])
                uniform_beta = np.full(
                    (len(base.attribute_ids), len(base.learner_names)),
                    1.0 / len(base.learner_names),
                    dtype=np.float64,
                )
                from unified_initial_baseline import gate_array_encoding
                gate_encoding = gate_array_encoding(base)
                if params.get("gateArrayEncoding", "<f4") != gate_encoding:
                    raise RuntimeError("Frozen gate array encoding changed after queuing")
                if fixed_gates or getattr(base, "initial_baseline_identity", None) is not None:
                    observed_theta = np.asarray(base.initial_theta, dtype=gate_encoding).copy()
                else:
                    observed_theta = recalibrate_unified_theta(
                        training["calibrationProbeScores"], training["calibrationLabels"],
                        uniform_beta, base.initial_theta, base.temperature,
                        calibration_sample_weights=training["calibrationWeights"],
                    ).astype(np.float32, copy=False)
                if fixed_gates:
                    frozen_temperature = np.asarray(params.get("initialTemperature"), dtype=gate_encoding)
                    if (frozen_temperature.shape != (len(base.attribute_ids),)
                            or not np.all(np.isfinite(frozen_temperature))
                            or np.any(frozen_temperature <= 0)
                            or not np.array_equal(frozen_temperature, np.asarray(base.temperature, dtype=gate_encoding))
                            or params.get("initialTemperatureFingerprint") != hashlib.sha256(
                                np.ascontiguousarray(frozen_temperature, dtype=gate_encoding).tobytes()).hexdigest()):
                        raise RuntimeError("Frozen refinement temperature changed after queuing")
                base_theta = np.asarray(
                    params.get("initialTheta"), dtype=gate_encoding
                ).reshape(-1)
                if base_theta.shape != (len(base.attribute_ids),):
                    raise RuntimeError(
                        "Frozen refinement threshold shape is invalid"
                    )
                observed_theta_fingerprint = hashlib.sha256(
                    np.ascontiguousarray(observed_theta, dtype=gate_encoding).tobytes(
                        order="C"
                    )
                ).hexdigest()
                frozen_theta_fingerprint = hashlib.sha256(
                    np.ascontiguousarray(base_theta, dtype=gate_encoding).tobytes(
                        order="C"
                    )
                ).hexdigest()
                if (
                    params.get("initialThetaFingerprint")
                    != frozen_theta_fingerprint
                    or observed_theta_fingerprint != frozen_theta_fingerprint
                    or not np.array_equal(observed_theta, base_theta)
                ):
                    raise RuntimeError(
                        "Frozen refinement threshold calibration changed after queuing"
                    )
                original_base = base
                fit_theta = base_theta
                probe_state = {"updated": False, "source": "frozen-five-seed-mean-score-cache"}
                if params.get("probeSource") == "updated":
                    if params.get("probeSelectionProtocol") != "explicit-frozen-probe-update-v1":
                        raise RuntimeError("Frozen Probe selection policy changed")
                    base, native_audit, _ = self._load_selected_probe_update(
                        str(run["user_id"]), str(run["session_id"]), task.task_id,
                        str(params.get("probeUpdateId") or ""), base, params.get("nativeProbeRequest"),
                    )
                    if native_audit["snapshotFingerprint"] != params.get("probeSnapshotFingerprint"):
                        raise RuntimeError("Selected Probe snapshot fingerprint changed after queuing")
                    probe_state = {"updated": True, "source": "native-updated-frozen-probe-snapshot",
                                   "probeUpdateId": params["probeUpdateId"], **native_audit}
                    training = self._materialize_refinement_training_arrays(base, prepared)
                elif mode in PROBE_REFINEMENT_RUN_MODES:
                    from native_probe_update import ensure_snapshot, prepare_request

                    observed_request = prepare_request(
                        self, str(run["user_id"]), str(run["session_id"]), task.task_id,
                        base, annotations, params,
                    )
                    if observed_request != params.get("nativeProbeRequest"):
                        raise RuntimeError("Native Probe update inputs changed after queuing")
                    base, native_audit = ensure_snapshot(self, observed_request, base)
                    probe_state = {"updated": True, "source": "native-updated-frozen-probe-snapshot", **native_audit}
                    training = self._materialize_refinement_training_arrays(base, prepared)
                    if not fixed_gates:
                        fit_theta = recalibrate_unified_theta(
                            training["calibrationProbeScores"], training["calibrationLabels"],
                            uniform_beta, base.initial_theta, base.temperature,
                            calibration_sample_weights=training["calibrationWeights"],
                        ).astype(np.float32, copy=False)
                model = fit_unified_weight_refinement(
                    mode,
                    training["probeScores"],
                    training["embeddingScores"],
                    training["jointLabels"],
                    theta=fit_theta,
                    temperature=base.temperature,
                    gate_calibration_policy="fixed-base" if fixed_gates else "legacy-recalibrate",
                    joint_sample_weights=training["jointWeights"],
                    attribute_labels=training["attributeLabels"],
                    attribute_sample_weights=training["attributeWeights"],
                    negative_failure_mask=training["jointFailureMask"],
                    calibration_probe_scores=training[
                        "calibrationProbeScores"
                    ],
                    calibration_attribute_labels=training[
                        "calibrationLabels"
                    ],
                    calibration_sample_weights=training[
                        "calibrationWeights"
                    ],
                    lambda0=float(params["lambda0"]),
                    gamma_min=float(params["gammaMin"]),
                    gamma_max=float(params["gammaMax"]),
                    rho_attribute=float(params["rhoAttribute"]),
                    rho_joint=float(params["rhoJoint"]),
                    beta_regularization=float(params["alphaBeta"]),
                    gamma_regularization=float(params["alphaGamma"]),
                    embedding_regularization=float(params["alphaEta"]),
                    lambda_regularization=float(params["alphaLambda"]),
                    learning_rate=float(params["learningRate"]),
                    max_iterations=max_iterations,
                    outer_iterations=int(params["outerIterations"]),
                    seed=int(params["randomSeed"]),
                )
                if fixed_gates and (
                    not np.array_equal(model.theta, base_theta)
                    or not np.array_equal(model.temperature, frozen_temperature)
                ):
                    raise RuntimeError("Weight optimization modified the frozen gate parameters")
                initial_all = unified_weight_scores(
                    original_base.probe_features,
                    original_base.embedding_features,
                    uniform_beta,
                    np.ones(len(base.attribute_ids), dtype=np.float64),
                    np.full(
                        len(base.embedding_names),
                        1.0 / len(base.embedding_names),
                        dtype=np.float64,
                    ),
                    float(params["lambda0"]),
                    base_theta,
                    base.temperature,
                    gamma_min=float(params["gammaMin"]),
                    gamma_max=float(params["gammaMax"]),
                )
                refined_all = unified_weight_scores(
                    base.probe_features,
                    base.embedding_features,
                    model.beta,
                    model.gamma,
                    model.embedding_weights,
                    model.embedding_fusion_strength,
                    model.theta,
                    model.temperature,
                    gamma_min=float(params["gammaMin"]),
                    gamma_max=float(params["gammaMax"]),
                )
                tuned_scores = np.asarray(
                    refined_all.final_scores, dtype=np.float32
                )
                tuned_ranks = normalized_ranks(tuned_scores)
                comparison_baseline_ranks = normalized_ranks(
                    initial_all.final_scores
                )
                atomic_write_bytes(
                    run_directory / "initial-scores.f32",
                    np.asarray(
                        initial_all.final_scores, dtype="<f4"
                    ).tobytes(order="C"),
                )
                atomic_write_bytes(
                    run_directory / "initial-ranks.f32",
                    np.asarray(
                        comparison_baseline_ranks, dtype="<f4"
                    ).tobytes(order="C"),
                )
                atomic_savez(
                    run_directory / "model.npz",
                    beta=np.asarray(model.beta, dtype=np.float32),
                    gamma=np.asarray(model.gamma, dtype=np.float32),
                    embeddingWeights=np.asarray(
                        model.embedding_weights, dtype=np.float32
                    ),
                    embeddingFusionStrength=np.float64(
                        model.embedding_fusion_strength
                    ),
                    theta=np.asarray(model.theta, dtype=np.float64),
                    temperature=np.asarray(model.temperature, dtype=np.float64),
                    probeMin=np.asarray(base.probe_min, dtype=np.float32),
                    probeMax=np.asarray(base.probe_max, dtype=np.float32),
                    embeddingMin=np.asarray(base.embedding_min, dtype=np.float32),
                    embeddingMax=np.asarray(base.embedding_max, dtype=np.float32),
                    attributeIds=np.asarray(base.attribute_ids),
                    learnerMethods=np.asarray(base.learner_names),
                    embeddingMethods=np.asarray(base.embedding_names),
                    baseStateFingerprint=np.asarray(
                        [base.base_state_fingerprint]
                    ),
                    originalBaseStateFingerprint=np.asarray([original_base.base_state_fingerprint]),
                    rankingIndices=np.asarray(
                        np.argsort(-tuned_scores, kind="stable"),
                        dtype=np.int64,
                    ),
                )
                history = [dict(item) for item in model.training_history]
                atomic_write_json(
                    run_directory / "training_history.json",
                    {
                        "schemaVersion": 1,
                        "mode": mode,
                        "baseStateFingerprint": base.base_state_fingerprint,
                        "history": history,
                        "gateCalibrationPolicy": params.get("gateCalibrationPolicy", "legacy-recalibrate"),
                    },
                )
                atomic_write_json(
                    run_directory / "base_state.json",
                    {
                        "schemaVersion": 1,
                        "contract": params["baseStateContract"],
                        "fingerprint": base.base_state_fingerprint,
                        "taskId": task.task_id,
                        "attributeIds": list(base.attribute_ids),
                        "learnerMethods": list(base.learner_names),
                        "embeddingMethods": list(base.embedding_names),
                        "normalizationPolicy": params["normalizationPolicy"],
                        "normalizationContract": params.get("normalizationContract"),
                        "thetaCalibration": params["thetaCalibration"],
                        "gateCalibrationPolicy": params.get("gateCalibrationPolicy", "legacy-recalibrate"),
                        "initialTemperature": [float(value) for value in original_base.temperature],
                        "lambda0": float(params["lambda0"]),
                        "gammaMin": float(params["gammaMin"]),
                        "gammaMax": float(params["gammaMax"]),
                        "initialTheta": [float(value) for value in base_theta],
                        "initialThetaFingerprint": params[
                            "initialThetaFingerprint"
                        ],
                        "originalBaseStateFingerprint": original_base.base_state_fingerprint,
                        "probeState": probe_state,
                    },
                )
                model_summary = {
                    "gateCalibrationPolicy": params.get("gateCalibrationPolicy", "legacy-recalibrate"),
                    "temperature": {
                        attribute_id: float(model.temperature[a])
                        for a, attribute_id in enumerate(base.attribute_ids)
                    },
                    "beta": {
                        attribute_id: {
                            learner: float(model.beta[attribute_index, learner_index])
                            for learner_index, learner in enumerate(base.learner_names)
                        }
                        for attribute_index, attribute_id in enumerate(
                            base.attribute_ids
                        )
                    },
                    "gamma": {
                        attribute_id: float(model.gamma[attribute_index])
                        for attribute_index, attribute_id in enumerate(
                            base.attribute_ids
                        )
                    },
                    "gammaConstraint": {
                        "kind": "independent-bounded-sigmoid",
                        "min": float(params["gammaMin"]),
                        "max": float(params["gammaMax"]),
                        "anchor": 1.0,
                    },
                    "embeddingWeights": {
                        method: float(model.embedding_weights[index])
                        for index, method in enumerate(base.embedding_names)
                    },
                    "embeddingFusionStrength": float(
                        model.embedding_fusion_strength
                    ),
                    "theta": {
                        attribute_id: float(model.theta[attribute_index])
                        for attribute_index, attribute_id in enumerate(
                            base.attribute_ids
                        )
                    },
                    "probeUpdated": bool(probe_state["updated"]),
                    "probeState": probe_state,
                    "minMaxStats": {
                        "probe": {
                            attribute_id: {
                                learner: {
                                    "min": float(
                                        base.probe_min[
                                            attribute_index, learner_index
                                        ]
                                    ),
                                    "max": float(
                                        base.probe_max[
                                            attribute_index, learner_index
                                        ]
                                    ),
                                }
                                for learner_index, learner in enumerate(
                                    base.learner_names
                                )
                            }
                            for attribute_index, attribute_id in enumerate(
                                base.attribute_ids
                            )
                        },
                        "embedding": {
                            method: {
                                "min": float(base.embedding_min[index]),
                                "max": float(base.embedding_max[index]),
                            }
                            for index, method in enumerate(base.embedding_names)
                        },
                    },
                    "trainingStrategy": params["trainingStrategy"],
                    "jointAggregation": "weighted-product-with-holistic-late-fusion",
                    "baseStateFingerprint": base.base_state_fingerprint,
                    "normalizationContract": params.get("normalizationContract"),
                    "initialObjective": float(model.initial_objective),
                    "finalObjective": float(model.final_objective),
                    "iterations": int(model.iterations),
                    "trainingHistory": history,
                }
            elif mode == "fusion-weight":
                if snapshot_algorithm_version == FUSION_WEIGHT_ALGORITHM_V1:
                    # Preserve queued v1 snapshots exactly.  They must never be
                    # reinterpreted as the new per-attribute parameterization.
                    def supervision_score_function(weights: Any) -> Any:
                        member_weights = {
                            method: float(weights[position])
                            for position, method in enumerate(context.learned_method_ids)
                        }
                        seed_scores = context.pcp.reconstruct_ours_full_scores(
                            context.tensors,
                            context.seeds_by_method,
                            list(context.learned_method_ids),
                            list(context.attributes),
                            target_attribute,
                            list(context.selected_seeds),
                            supervision_indices,
                            context.task_config,
                            context.ours_full_root,
                            expected_stage=context.probe_stage,
                            member_weights=member_weights,
                        )
                        return np.asarray(seed_scores.mean(axis=0), dtype=np.float64)

                    model = fit_fusion_weights(
                        supervision_score_function,
                        supervision_labels,
                        supervision_weights,
                        member_count=len(context.learned_method_ids),
                        regularization_lambda=regularization_lambda,
                        bias_regularization_eta=bias_regularization_eta,
                        max_iterations=max_iterations,
                    )
                    normalized_weights = tuple(float(value) for value in model.weights)
                    fusion = self._compute_weighted_fusion(
                        task.task_id, normalized_weights
                    )
                    atomic_savez(
                        run_directory / "model.npz",
                        weights=np.asarray(model.weights, dtype=np.float32),
                        bias=np.float32(model.bias),
                        initialWeights=np.full(
                            len(context.learned_method_ids),
                            1.0 / len(context.learned_method_ids),
                            dtype=np.float32,
                        ),
                        learnerMethods=np.asarray(context.learned_method_labels),
                    )
                    model_summary = {
                        "learnerWeights": {
                            label: float(model.weights[position])
                            for position, label in enumerate(
                                context.learned_method_labels
                            )
                        },
                        "bias": float(model.bias),
                        "biasAffectsRanking": False,
                        "initialObjective": model.initial_objective,
                        "finalObjective": model.final_objective,
                        "iterations": model.iterations,
                    }
                elif snapshot_algorithm_version == FUSION_WEIGHT_ALGORITHM_V2:
                    attribute_count = len(context.attributes)
                    member_count = len(context.learned_method_ids)
                    if attribute_count < 1:
                        raise RuntimeError(
                            "Per-attribute Fusion tuning requires modeled attributes"
                        )
                    attribute_ids = tuple(
                        str(target.get("id"))
                        for target in task.manifest.get("retrievalTargets", [])
                        if target.get("kind") == "attribute"
                    )
                    if len(attribute_ids) != attribute_count:
                        raise RuntimeError(
                            "Fusion snapshot attributes do not match the source model"
                        )
                    if tuple(params.get("attributeIds") or ()) != attribute_ids:
                        raise RuntimeError(
                            "Fusion snapshot attribute order does not match the task"
                        )
                    if tuple(params.get("learnerMethods") or ()) != tuple(
                        context.learned_method_labels
                    ):
                        raise RuntimeError(
                            "Fusion snapshot learner order does not match the task"
                        )
                    expected_initial_weights = {
                        attribute_id: {
                            learner: 1.0 / member_count
                            for learner in context.learned_method_labels
                        }
                        for attribute_id in attribute_ids
                    }
                    if params.get("parameterization") != "per-attribute-simplex":
                        raise RuntimeError(
                            "Fusion snapshot has an unsupported parameterization"
                        )
                    if params.get("initialWeightsByAttribute") != expected_initial_weights:
                        raise RuntimeError(
                            "Fusion snapshot initial weights do not match the task"
                        )
                    if target_attribute is None:
                        active_attribute_indices = tuple(range(attribute_count))
                    else:
                        active_attribute_indices = (
                            context.attributes.index(target_attribute),
                        )
                    expected_active_ids = tuple(
                        attribute_ids[index] for index in active_attribute_indices
                    )
                    if tuple(params.get("activeAttributeIds") or ()) != expected_active_ids:
                        raise RuntimeError(
                            "Fusion snapshot active attributes do not match its target"
                        )

                    def attribute_supervision_score_function(weights: Any) -> Any:
                        matrix = np.asarray(weights, dtype=np.float64)
                        if matrix.shape != (attribute_count, member_count):
                            raise RuntimeError(
                                "Per-attribute Fusion optimizer returned an invalid shape"
                            )
                        member_weights_by_attribute = {
                            attribute: {
                                method: float(matrix[attribute_index, method_index])
                                for method_index, method in enumerate(
                                    context.learned_method_ids
                                )
                            }
                            for attribute_index, attribute in enumerate(
                                context.attributes
                            )
                        }
                        seed_scores = context.pcp.reconstruct_ours_full_scores(
                            context.tensors,
                            context.seeds_by_method,
                            list(context.learned_method_ids),
                            list(context.attributes),
                            target_attribute,
                            list(context.selected_seeds),
                            supervision_indices,
                            context.task_config,
                            context.ours_full_root,
                            expected_stage=context.probe_stage,
                            member_weights_by_attribute=member_weights_by_attribute,
                        )
                        return np.asarray(seed_scores.mean(axis=0), dtype=np.float64)

                    model = fit_attribute_fusion_weights(
                        attribute_supervision_score_function,
                        supervision_labels,
                        supervision_weights,
                        attribute_count=attribute_count,
                        member_count=member_count,
                        active_attribute_indices=active_attribute_indices,
                        regularization_lambda=regularization_lambda,
                        shared_regularization_lambda=float(
                            params["sharedRegularizationLambda"]
                        ),
                        bias_regularization_eta=bias_regularization_eta,
                        max_iterations=max_iterations,
                    )
                    normalized_learner_weights = tuple(
                        tuple(float(value) for value in row)
                        for row in np.asarray(model.weights, dtype=np.float64)
                    )
                    fusion = self._compute_hierarchical_fusion(
                        task.task_id,
                        tuple(1.0 for _ in range(attribute_count)),
                        normalized_learner_weights,
                        OURS_ONLY_GLOBAL_WEIGHTS,
                    )
                    active_attribute_ids = tuple(
                        attribute_ids[index]
                        for index in model.active_attribute_indices
                    )
                    atomic_savez(
                        run_directory / "model.npz",
                        weights=np.asarray(model.weights, dtype=np.float32),
                        bias=np.float32(model.bias),
                        initialWeights=np.full(
                            (attribute_count, member_count),
                            1.0 / member_count,
                            dtype=np.float32,
                        ),
                        attributeIds=np.asarray(attribute_ids),
                        attributeNames=np.asarray(context.attributes),
                        activeAttributeIds=np.asarray(active_attribute_ids),
                        learnerMethods=np.asarray(context.learned_method_labels),
                    )
                    model_summary = {
                        "learnerWeightsByAttribute": {
                            attribute_id: {
                                label: float(model.weights[attribute_index, method_index])
                                for method_index, label in enumerate(
                                    context.learned_method_labels
                                )
                            }
                            for attribute_index, attribute_id in enumerate(attribute_ids)
                        },
                        "activeAttributeIds": list(active_attribute_ids),
                        "learnerWeightCount": attribute_count * member_count,
                        "optimizedLearnerWeightCount": (
                            len(model.active_attribute_indices) * member_count
                        ),
                        "bias": float(model.bias),
                        "biasAffectsRanking": False,
                        "initialObjective": model.initial_objective,
                        "finalObjective": model.final_objective,
                        "iterations": model.iterations,
                        "sharedRegularizationLambda": float(
                            params["sharedRegularizationLambda"]
                        ),
                    }
                elif snapshot_algorithm_version == FUSION_WEIGHT_ALGORITHM_V3:
                    # v3 extends the v2 learner simplexes with one
                    # non-negative, mean-one exponent per attribute in the
                    # Joint product.  The separate branch is intentional:
                    # immutable v1/v2 snapshots retain their original
                    # parameterization and artifact semantics.
                    attribute_count = len(context.attributes)
                    member_count = len(context.learned_method_ids)
                    if attribute_count < 1:
                        raise RuntimeError(
                            "Per-attribute Fusion tuning requires modeled attributes"
                        )
                    attribute_ids = tuple(
                        str(target.get("id"))
                        for target in task.manifest.get("retrievalTargets", [])
                        if target.get("kind") == "attribute"
                    )
                    if len(attribute_ids) != attribute_count:
                        raise RuntimeError(
                            "Fusion snapshot attributes do not match the source model"
                        )
                    if tuple(params.get("attributeIds") or ()) != attribute_ids:
                        raise RuntimeError(
                            "Fusion snapshot attribute order does not match the task"
                        )
                    if tuple(params.get("learnerMethods") or ()) != tuple(
                        context.learned_method_labels
                    ):
                        raise RuntimeError(
                            "Fusion snapshot learner order does not match the task"
                        )
                    expected_initial_weights = {
                        attribute_id: {
                            learner: 1.0 / member_count
                            for learner in context.learned_method_labels
                        }
                        for attribute_id in attribute_ids
                    }
                    if params.get("parameterization") != (
                        "per-attribute-simplex-plus-joint-product-exponents"
                    ):
                        raise RuntimeError(
                            "Fusion snapshot has an unsupported parameterization"
                        )
                    if params.get("initialWeightsByAttribute") != expected_initial_weights:
                        raise RuntimeError(
                            "Fusion snapshot initial learner weights do not match the task"
                        )
                    expected_initial_attribute_weights = {
                        attribute_id: 1.0 for attribute_id in attribute_ids
                    }
                    if (
                        params.get("initialAttributeWeights")
                        != expected_initial_attribute_weights
                    ):
                        raise RuntimeError(
                            "Fusion snapshot initial Joint weights do not match the task"
                        )
                    if params.get("jointAggregation") != "product":
                        raise RuntimeError(
                            "Fusion snapshot has an unsupported Joint aggregation"
                        )
                    if params.get("jointWeightConstraint") != "nonnegative-mean-one":
                        raise RuntimeError(
                            "Fusion snapshot has an unsupported Joint weight constraint"
                        )
                    if target_attribute is None:
                        active_attribute_indices = tuple(range(attribute_count))
                        active_joint_attribute_indices = tuple(range(attribute_count))
                    else:
                        active_attribute_indices = (
                            context.attributes.index(target_attribute),
                        )
                        # Joint exponents do not participate in a
                        # single-attribute target, so they stay exactly one and
                        # are not reported as learned.
                        active_joint_attribute_indices = ()
                    expected_active_ids = tuple(
                        attribute_ids[index] for index in active_attribute_indices
                    )
                    if tuple(params.get("activeAttributeIds") or ()) != expected_active_ids:
                        raise RuntimeError(
                            "Fusion snapshot active attributes do not match its target"
                        )
                    expected_active_joint_ids = tuple(
                        attribute_ids[index]
                        for index in active_joint_attribute_indices
                    )
                    if (
                        tuple(params.get("activeJointAttributeIds") or ())
                        != expected_active_joint_ids
                    ):
                        raise RuntimeError(
                            "Fusion snapshot active Joint attributes do not match its target"
                        )

                    def joint_supervision_score_function(
                        weights: Any, attribute_weights: Any
                    ) -> Any:
                        matrix = np.asarray(weights, dtype=np.float64)
                        exponents = np.asarray(attribute_weights, dtype=np.float64)
                        if matrix.shape != (attribute_count, member_count):
                            raise RuntimeError(
                                "Per-attribute Fusion optimizer returned an invalid shape"
                            )
                        if exponents.shape != (attribute_count,):
                            raise RuntimeError(
                                "Joint Fusion optimizer returned an invalid shape"
                            )
                        member_weights_by_attribute = {
                            attribute: {
                                method: float(matrix[attribute_index, method_index])
                                for method_index, method in enumerate(
                                    context.learned_method_ids
                                )
                            }
                            for attribute_index, attribute in enumerate(
                                context.attributes
                            )
                        }
                        joint_attribute_weights = {
                            attribute: float(exponents[attribute_index])
                            for attribute_index, attribute in enumerate(
                                context.attributes
                            )
                        }
                        seed_scores = context.pcp.reconstruct_ours_full_scores(
                            context.tensors,
                            context.seeds_by_method,
                            list(context.learned_method_ids),
                            list(context.attributes),
                            target_attribute,
                            list(context.selected_seeds),
                            supervision_indices,
                            context.task_config,
                            context.ours_full_root,
                            expected_stage=context.probe_stage,
                            member_weights_by_attribute=member_weights_by_attribute,
                            attribute_weights=joint_attribute_weights,
                        )
                        return np.asarray(seed_scores.mean(axis=0), dtype=np.float64)

                    model = fit_attribute_and_joint_fusion_weights(
                        joint_supervision_score_function,
                        supervision_labels,
                        supervision_weights,
                        attribute_count=attribute_count,
                        member_count=member_count,
                        active_attribute_indices=active_attribute_indices,
                        active_joint_attribute_indices=(
                            active_joint_attribute_indices
                        ),
                        regularization_lambda=regularization_lambda,
                        shared_regularization_lambda=float(
                            params["sharedRegularizationLambda"]
                        ),
                        joint_regularization_lambda=float(
                            params["jointRegularizationLambda"]
                        ),
                        bias_regularization_eta=bias_regularization_eta,
                        max_iterations=max_iterations,
                    )
                    normalized_learner_weights = tuple(
                        tuple(float(value) for value in row)
                        for row in np.asarray(model.weights, dtype=np.float64)
                    )
                    normalized_attribute_weights = tuple(
                        float(value)
                        for value in np.asarray(
                            model.attribute_weights, dtype=np.float64
                        )
                    )
                    fusion = self._compute_hierarchical_fusion(
                        task.task_id,
                        normalized_attribute_weights,
                        normalized_learner_weights,
                        OURS_ONLY_GLOBAL_WEIGHTS,
                    )
                    active_attribute_ids = tuple(
                        attribute_ids[index]
                        for index in model.active_attribute_indices
                    )
                    active_joint_attribute_ids = tuple(
                        attribute_ids[index]
                        for index in model.active_joint_attribute_indices
                    )
                    atomic_savez(
                        run_directory / "model.npz",
                        weights=np.asarray(model.weights, dtype=np.float32),
                        attributeWeights=np.asarray(
                            model.attribute_weights, dtype=np.float32
                        ),
                        bias=np.float32(model.bias),
                        initialWeights=np.full(
                            (attribute_count, member_count),
                            1.0 / member_count,
                            dtype=np.float32,
                        ),
                        initialAttributeWeights=np.ones(
                            attribute_count, dtype=np.float32
                        ),
                        attributeIds=np.asarray(attribute_ids),
                        attributeNames=np.asarray(context.attributes),
                        activeAttributeIds=np.asarray(active_attribute_ids),
                        activeJointAttributeIds=np.asarray(
                            active_joint_attribute_ids, dtype=np.str_
                        ),
                        learnerMethods=np.asarray(context.learned_method_labels),
                        jointAggregation=np.asarray(["product"]),
                        jointWeightConstraint=np.asarray(
                            ["nonnegative-mean-one"]
                        ),
                    )
                    model_summary = {
                        "learnerWeightsByAttribute": {
                            attribute_id: {
                                label: float(
                                    model.weights[attribute_index, method_index]
                                )
                                for method_index, label in enumerate(
                                    context.learned_method_labels
                                )
                            }
                            for attribute_index, attribute_id in enumerate(attribute_ids)
                        },
                        "attributeWeights": {
                            attribute_id: float(
                                model.attribute_weights[attribute_index]
                            )
                            for attribute_index, attribute_id in enumerate(attribute_ids)
                        },
                        "activeAttributeIds": list(active_attribute_ids),
                        "activeJointAttributeIds": list(
                            active_joint_attribute_ids
                        ),
                        "learnerWeightCount": attribute_count * member_count,
                        "optimizedLearnerWeightCount": (
                            len(model.active_attribute_indices) * member_count
                        ),
                        "attributeWeightCount": attribute_count,
                        "optimizedAttributeWeightCount": len(
                            model.active_joint_attribute_indices
                        ),
                        "jointAggregation": "product",
                        "jointWeightConstraint": "nonnegative-mean-one",
                        "bias": float(model.bias),
                        "biasAffectsRanking": False,
                        "initialObjective": model.initial_objective,
                        "finalObjective": model.final_objective,
                        "iterations": model.iterations,
                        "sharedRegularizationLambda": float(
                            params["sharedRegularizationLambda"]
                        ),
                        "jointRegularizationLambda": float(
                            params["jointRegularizationLambda"]
                        ),
                    }
                elif snapshot_algorithm_version == FUSION_WEIGHT_ALGORITHM_V4:
                    attribute_ids, supervision_rank_features = (
                        self._rank_fusion_source_features(
                            task,
                            supervision_indices,
                        )
                    )
                    attribute_count = len(attribute_ids)
                    member_count = len(RANK_FUSION_METHODS)
                    if tuple(params.get("attributeIds") or ()) != attribute_ids:
                        raise RuntimeError(
                            "Rank-fusion snapshot attribute order does not match the task"
                        )
                    if tuple(params.get("methodIds") or ()) != RANK_FUSION_METHODS:
                        raise RuntimeError(
                            "Rank-fusion snapshot method order does not match the contract"
                        )
                    if params.get("parameterization") != (
                        "per-attribute-13-simplex-plus-attribute-simplex"
                    ):
                        raise RuntimeError(
                            "Rank-fusion snapshot has an unsupported parameterization"
                        )
                    expected_initial_method_weights = {
                        attribute_id: {
                            method: 1.0 / member_count
                            for method in RANK_FUSION_METHODS
                        }
                        for attribute_id in attribute_ids
                    }
                    if (
                        params.get("initialMethodWeightsByAttribute")
                        != expected_initial_method_weights
                    ):
                        raise RuntimeError(
                            "Rank-fusion snapshot initial method weights do not match"
                        )
                    expected_initial_attribute_weights = {
                        attribute_id: 1.0 / attribute_count
                        for attribute_id in attribute_ids
                    }
                    if (
                        params.get("initialAttributeWeights")
                        != expected_initial_attribute_weights
                    ):
                        raise RuntimeError(
                            "Rank-fusion snapshot initial attribute weights do not match"
                        )
                    if params.get("jointAggregation") != "weighted-mean":
                        raise RuntimeError(
                            "Rank-fusion snapshot has an unsupported Joint aggregation"
                        )
                    if params.get("jointWeightConstraint") != "nonnegative-sum-one":
                        raise RuntimeError(
                            "Rank-fusion snapshot has an unsupported attribute constraint"
                        )
                    if target_id == "joint":
                        active_attribute_indices = tuple(range(attribute_count))
                        active_joint_attribute_indices = tuple(range(attribute_count))
                    elif target_id in attribute_ids:
                        active_attribute_indices = (attribute_ids.index(target_id),)
                        active_joint_attribute_indices = ()
                    else:
                        raise RuntimeError(
                            "Rank-fusion tuning target is not an attribute or Joint"
                        )
                    expected_active_ids = tuple(
                        attribute_ids[index] for index in active_attribute_indices
                    )
                    expected_active_joint_ids = tuple(
                        attribute_ids[index]
                        for index in active_joint_attribute_indices
                    )
                    if tuple(params.get("activeAttributeIds") or ()) != expected_active_ids:
                        raise RuntimeError(
                            "Rank-fusion snapshot active attributes do not match its target"
                        )
                    if (
                        tuple(params.get("activeJointAttributeIds") or ())
                        != expected_active_joint_ids
                    ):
                        raise RuntimeError(
                            "Rank-fusion snapshot active Joint attributes do not match"
                        )

                    def rank_supervision_score_function(
                        weights: Any,
                        attribute_weights: Any,
                    ) -> Any:
                        matrix = np.asarray(weights, dtype=np.float64)
                        outer = np.asarray(attribute_weights, dtype=np.float64)
                        if matrix.shape != (attribute_count, member_count):
                            raise RuntimeError(
                                "Rank-fusion optimizer returned an invalid method matrix"
                            )
                        if outer.shape != (attribute_count,):
                            raise RuntimeError(
                                "Rank-fusion optimizer returned invalid attribute weights"
                            )
                        inner = np.einsum(
                            "ram,am->ra",
                            np.asarray(supervision_rank_features, dtype=np.float64),
                            matrix,
                            optimize=True,
                        )
                        if target_id != "joint":
                            return inner[:, attribute_ids.index(target_id)]
                        outer_total = float(np.sum(outer))
                        if not math.isfinite(outer_total) or outer_total <= 0.0:
                            raise RuntimeError(
                                "Rank-fusion optimizer returned empty attribute weights"
                            )
                        return inner @ (outer / outer_total)

                    model = fit_attribute_and_joint_fusion_weights(
                        rank_supervision_score_function,
                        supervision_labels,
                        supervision_weights,
                        attribute_count=attribute_count,
                        member_count=member_count,
                        active_attribute_indices=active_attribute_indices,
                        active_joint_attribute_indices=(
                            active_joint_attribute_indices
                        ),
                        regularization_lambda=regularization_lambda,
                        shared_regularization_lambda=float(
                            params["sharedRegularizationLambda"]
                        ),
                        joint_regularization_lambda=float(
                            params["jointRegularizationLambda"]
                        ),
                        bias_regularization_eta=bias_regularization_eta,
                        max_iterations=max_iterations,
                    )
                    normalized_method_weights = tuple(
                        tuple(float(value) for value in row)
                        for row in np.asarray(model.weights, dtype=np.float64)
                    )
                    outer_values = np.asarray(
                        model.attribute_weights,
                        dtype=np.float64,
                    )
                    outer_total = float(np.sum(outer_values))
                    if outer_total <= 0.0:
                        raise RuntimeError(
                            "Rank-fusion optimizer returned empty attribute weights"
                        )
                    normalized_attribute_weights = tuple(
                        float(value / outer_total) for value in outer_values
                    )
                    fusion = self._compute_rank_fusion(
                        task.task_id,
                        normalized_attribute_weights,
                        normalized_method_weights,
                    )
                    normalized_attribute_weights = (
                        fusion.normalized_attribute_weights
                    )
                    normalized_method_weights = (
                        fusion.normalized_method_weights_by_attribute
                    )
                    active_attribute_ids = tuple(
                        attribute_ids[index]
                        for index in model.active_attribute_indices
                    )
                    active_joint_attribute_ids = tuple(
                        attribute_ids[index]
                        for index in model.active_joint_attribute_indices
                    )
                    atomic_savez(
                        run_directory / "model.npz",
                        weights=np.asarray(
                            normalized_method_weights,
                            dtype=np.float64,
                        ),
                        attributeWeights=np.asarray(
                            normalized_attribute_weights,
                            dtype=np.float32,
                        ),
                        bias=np.float32(model.bias),
                        initialWeights=np.full(
                            (attribute_count, member_count),
                            1.0 / member_count,
                            dtype=np.float32,
                        ),
                        initialAttributeWeights=np.full(
                            attribute_count,
                            1.0 / attribute_count,
                            dtype=np.float32,
                        ),
                        attributeIds=np.asarray(attribute_ids),
                        methodIds=np.asarray(RANK_FUSION_METHODS),
                        activeAttributeIds=np.asarray(active_attribute_ids),
                        activeJointAttributeIds=np.asarray(
                            active_joint_attribute_ids,
                            dtype=np.str_,
                        ),
                        jointAggregation=np.asarray(["weighted-mean"]),
                        jointWeightConstraint=np.asarray(["nonnegative-sum-one"]),
                    )
                    model_summary = {
                        "methodWeightsByAttribute": {
                            attribute_id: {
                                method: float(
                                    normalized_method_weights[
                                        attribute_index
                                    ][method_index]
                                )
                                for method_index, method in enumerate(
                                    RANK_FUSION_METHODS
                                )
                            }
                            for attribute_index, attribute_id in enumerate(
                                attribute_ids
                            )
                        },
                        "attributeWeights": {
                            attribute_id: normalized_attribute_weights[attribute_index]
                            for attribute_index, attribute_id in enumerate(
                                attribute_ids
                            )
                        },
                        "activeAttributeIds": list(active_attribute_ids),
                        "activeJointAttributeIds": list(
                            active_joint_attribute_ids
                        ),
                        "methodWeightCount": attribute_count * member_count,
                        "optimizedMethodWeightCount": (
                            len(model.active_attribute_indices) * member_count
                        ),
                        "attributeWeightCount": attribute_count,
                        "optimizedAttributeWeightCount": len(
                            model.active_joint_attribute_indices
                        ),
                        "jointAggregation": "weighted-mean",
                        "jointWeightConstraint": "nonnegative-sum-one",
                        "bias": float(model.bias),
                        "biasAffectsRanking": False,
                        "biasAppliedToArtifactScores": False,
                        "initialObjective": model.initial_objective,
                        "finalObjective": model.final_objective,
                        "iterations": model.iterations,
                        "sharedRegularizationLambda": float(
                            params["sharedRegularizationLambda"]
                        ),
                        "jointRegularizationLambda": float(
                            params["jointRegularizationLambda"]
                        ),
                    }
                else:
                    raise RuntimeError(
                        f"Unsupported Fusion Weight version: {snapshot_algorithm_version}"
                    )
                packed = np.frombuffer(fusion.payload, dtype="<f4")
                cube_size = task.row_count * len(task.target_ids)
                fusion_raw = packed[:cube_size].reshape(task.row_count, len(task.target_ids))
                fusion_calibrated = packed[cube_size : 2 * cube_size].reshape(
                    task.row_count, len(task.target_ids)
                )
                fusion_ranks = packed[2 * cube_size : 3 * cube_size].reshape(
                    task.row_count, len(task.target_ids)
                )
                if snapshot_algorithm_version == FUSION_WEIGHT_ALGORITHM_V4:
                    tuned_scores = np.asarray(
                        fusion_calibrated[:, target_index],
                        dtype=np.float32,
                    )
                else:
                    uncalibrated = np.asarray(
                        fusion_raw[:, target_index], dtype=np.float64
                    )
                    tuned_scores = stable_sigmoid(
                        stable_logit(uncalibrated) + model.bias
                    ).astype(np.float32)
                # The deployed Ours-Full contract is mean seed-wise rank.  A
                # global calibration bias is monotonic and therefore does not
                # change this exact ranking artifact.
                tuned_ranks = np.asarray(fusion_ranks[:, target_index], dtype=np.float32)
            elif mode == "residual":
                base_ranks_all = self._exported_method_column(
                    task, "ranks", str(run["base_method"]), target_index
                )
                base_calibrated_all = self._exported_method_column(
                    task, "calibratedScores", str(run["base_method"]), target_index
                )
                learner_features_all = self._learner_calibrated_features(task, target_index)
                model = fit_residual(
                    base_ranks_all[supervision_indices],
                    learner_features_all[supervision_indices],
                    base_calibrated_all[supervision_indices],
                    supervision_labels,
                    supervision_weights,
                    regularization_lambda=regularization_lambda,
                    bias_regularization_eta=bias_regularization_eta,
                    max_iterations=max_iterations,
                )
                tuned_scores = residual_probabilities(
                    base_ranks_all,
                    learner_features_all,
                    base_calibrated_all,
                    model.coefficients,
                    model.bias,
                ).astype(np.float32)
                tuned_ranks = normalized_ranks(tuned_scores)
                atomic_savez(
                    run_directory / "model.npz",
                    coefficients=np.asarray(model.coefficients, dtype=np.float32),
                    bias=np.float32(model.bias),
                    featureMethods=np.asarray(WEIGHTED_FUSION_LEARNERS),
                    baseMethod=np.asarray([str(run["base_method"])]),
                )
                model_summary = {
                    "residualCoefficients": {
                        label: float(model.coefficients[position])
                        for position, label in enumerate(WEIGHTED_FUSION_LEARNERS)
                    },
                    "bias": float(model.bias),
                    "initialObjective": model.initial_objective,
                    "finalObjective": model.final_objective,
                    "iterations": model.iterations,
                }
            else:
                raise RuntimeError(f"Unsupported low-dimensional run mode: {mode}")
        else:
            raise RuntimeError(f"Unsupported run mode: {mode}")

        rank_spec = task.manifest["files"]["ranks"]
        rank_path = self._bundle_file(task, rank_spec["path"])
        exported_ranks = np.memmap(
            rank_path,
            dtype="<f4",
            mode="r",
            shape=(task.row_count, len(task.methods), len(task.target_ids)),
        )
        # The browser's Top Gallery orders by the exported rank tensor.  In
        # particular, learned methods use mean seed-wise percentiles, which can
        # differ from sorting their arithmetic-mean raw score.
        baseline_ranks = np.asarray(
            exported_ranks[:, method_index, target_index], dtype=np.float32
        )
        if comparison_baseline_ranks is not None:
            # New refinement modes compare against their shared formal F^(0),
            # which includes the run-pinned holistic embedding branch.
            # Historical runs retain their recorded exported-method baseline.
            baseline_ranks = np.asarray(
                comparison_baseline_ranks, dtype=np.float32
            )

        evaluation_scope = (
            str(run["evaluation_scope"])
            if "evaluation_scope" in run.keys() and run["evaluation_scope"]
            else "test"
        )
        split_audit: dict[str, Any] | None = None
        if evaluation_scope in {"vqa-validation", "clean-validation", "probe-validation"}:
            if probe_validation_split is None:
                raise RuntimeError(
                    "Validation scope requires the frozen audited split"
                )
            evaluation_rows = np.asarray(
                probe_validation_split.validation_indices, dtype=np.int64
            )
            evaluation_truth = np.asarray(
                probe_validation_split.validation_labels, dtype=np.uint8
            )
            split_audit = dict(probe_validation_split.audit)
            if mode in REFINEMENT_RUN_MODES and params.get("initialBaselineIdentity"):
                # The pinned base loader above validated the isolated bank and
                # immutable Val contract, rather than trusting a UI flag.
                split_audit["initialModelHoldoutIndependent"] = params["initialBaselineIdentity"]["initialModelHoldoutIndependent"]
                split_audit["calibrationUsedValidation"] = params["initialBaselineIdentity"].get("calibrationUsedValidation", False)
                split_audit["referenceOnly"] = False
                split_audit["initialBaselineIdentity"] = params["initialBaselineIdentity"]
            split_audit["feedbackHoldoutExcludedCount"] = int(
                (supervision_audit or {}).get("feedbackHoldoutExcludedCount", 0)
            )
        elif evaluation_scope in {"validation", "test"}:
            ground_truth_spec = task.manifest["files"]["groundTruth"]
            ground_truth_path = self._bundle_file(task, ground_truth_spec["path"])
            ground_truth = np.fromfile(ground_truth_path, dtype=np.uint8).reshape(
                task.row_count, len(task.target_ids)
            )
            if evaluation_scope == "validation":
                evaluation_mask = np.frombuffer(
                    bundle.validation_mask, dtype=np.uint8
                ).astype(bool)
            else:
                # Compatibility for immutable historical records.  New jobs
                # use label-isolated Web Validation; only historical jobs can
                # still reach the Frozen Test evaluation branch.
                evaluation_mask = np.frombuffer(
                    bundle.test_mask, dtype=np.uint8
                ).astype(bool)
            evaluation_rows = np.flatnonzero(evaluation_mask)
            evaluation_truth = ground_truth[evaluation_rows, target_index]
        else:
            raise RuntimeError(f"Unsupported evaluation scope: {evaluation_scope}")
        before = retrieval_metrics(evaluation_truth, baseline_ranks[evaluation_rows])
        if tuned_ranks is None:
            tuned_ranks = normalized_ranks(tuned_scores)
        after = retrieval_metrics(evaluation_truth, tuned_ranks[evaluation_rows])
        test_diagnostics: dict[str, Any] = {}
        if (
            mode in WEIGHT_REFINEMENT_RUN_MODES
            and snapshot_algorithm_version == RUN_ALGORITHM_VERSIONS[mode]
            and evaluation_scope == "vqa-validation"
            and task.manifest["files"].get("groundTruth") is not None
        ):
            # All training, calibration, checkpoint selection and final scoring
            # above have finished. Both Probe sources compare the same original
            # run-pinned F0 with the exact ranks that will be deployed below.
            try:
                test_diagnostics["testEvaluation"] = self._post_training_test_evaluation(
                    task, bundle, target_id, baseline_ranks, tuned_ranks,
                    "Unified base F⁽⁰⁾",
                )
            except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
                # Optional diagnostics cannot turn a completed Tune into a
                # failed training job. Never substitute Val metrics or zeros.
                test_diagnostics["testEvaluationError"] = (
                    str(error) if isinstance(error, (ValueError, RuntimeError))
                    else "Frozen Test evaluation inputs could not be read"
                )
        if split_audit is not None:
            after["splitAudit"] = split_audit
        if mode in REQUESTABLE_RUN_MODES:
            # Persist a compact, user-visible audit alongside the scalar
            # validation metrics.  This keeps the learned low-dimensional
            # parameters inspectable without exposing source tensors.
            after["modelSummary"] = model_summary
            after["supervisionSummary"] = {
                "originalDevelopmentCount": int(
                    (supervision_audit or {}).get("originalDevelopmentCount", 0)
                ),
                "originalProbeSupervisionCount": int(
                    (supervision_audit or {}).get(
                        "originalProbeSupervisionCount", 0
                    )
                ),
                "originalProbeFitCount": int(
                    (supervision_audit or {}).get("originalProbeFitCount", 0)
                ),
                "probeValidationCount": int(
                    (supervision_audit or {}).get("probeValidationCount", 0)
                ),
                "cleanValidationCount": int(
                    (supervision_audit or {}).get("cleanValidationCount", 0)
                ),
                "vqaValidationCount": int(
                    (supervision_audit or {}).get("vqaValidationCount", 0)
                ),
                "feedbackSnapshotCount": int(
                    (supervision_audit or {}).get("feedbackSnapshotCount", 0)
                ),
                "feedbackHoldoutExcludedCount": int(
                    (supervision_audit or {}).get(
                        "feedbackHoldoutExcludedCount", 0
                    )
                ),
                "feedbackCount": int((supervision_audit or {}).get("feedbackCount", 0)),
                "newFeedbackCount": int(
                    (supervision_audit or {}).get("newFeedbackCount", 0)
                ),
                "feedbackExistingCount": int(
                    (supervision_audit or {}).get("feedbackExistingCount", 0)
                ),
                "feedbackOverrideCount": int(
                    (supervision_audit or {}).get("feedbackOverrideCount", 0)
                ),
                "feedbackReinforceCount": int(
                    (supervision_audit or {}).get("feedbackReinforceCount", 0)
                ),
                "feedbackReviewOnlyCount": int(
                    (supervision_audit or {}).get("feedbackReviewOnlyCount", 0)
                ),
                "feedbackWeight": float(params["feedbackWeight"]),
                "combinedCount": int((supervision_audit or {}).get("combinedCount", 0)),
            }
        atomic_write_bytes(
            run_directory / "tuned-scores.f32",
            np.asarray(tuned_scores, dtype="<f4").tobytes(order="C"),
        )
        atomic_write_bytes(
            run_directory / "tuned-ranks.f32",
            np.asarray(tuned_ranks, dtype="<f4").tobytes(order="C"),
        )
        atomic_write_json(
            run_directory / "metrics.json",
            {
                "schemaVersion": int(snapshot["schemaVersion"]) if mode in REFINEMENT_RUN_MODES else (
                    3 if mode in REQUESTABLE_RUN_MODES else 2
                ),
                "runId": str(run["id"]),
                "mode": mode,
                "taskId": task.task_id,
                "targetId": target_id,
                "baseMethod": str(run["base_method"]),
                "beforeMethod": (
                    "Unified base F⁽⁰⁾"
                    if mode in REFINEMENT_RUN_MODES
                    else str(run["base_method"])
                ),
                "evaluationScope": evaluation_scope,
                "annotationSha256": str(run["annotation_sha256"]),
                "snapshotSha256": snapshot_sha256,
                "embedding": embedding_meta,
                "algorithmVersion": params.get(
                    "algorithmVersion", RUN_ALGORITHM_VERSIONS.get(mode)
                ),
                "supervisionPolicy": params.get("supervisionPolicy"),
                "supervision": supervision_audit,
                "splitAudit": split_audit,
                "modelSummary": model_summary,
                "before": before,
                "after": after,
                **test_diagnostics,
                "deltaAp": float(after["ap"] - before["ap"]),
                "groundTruthUsage": (
                    "Val uses fixed original VQA labels; any dataset ground-truth read is only "
                    "for frozen Test post-training diagnostics, never fitting, calibration or checkpoint selection"
                    if test_diagnostics else
                    "None; fixed original VQA labels only (reference-only, not independent of initial models)"
                    if evaluation_scope == "vqa-validation" else
                    "Label-isolated Web Validation ground truth; excluded from fitting"
                    if evaluation_scope == "clean-validation"
                    else (
                        "Recovered VQA Probe Validation labels only"
                        if evaluation_scope == "probe-validation"
                        else (
                            "Validation model-selection evaluation only"
                            if evaluation_scope == "validation"
                            else "Frozen Test post-hoc evaluation only (legacy run)"
                        )
                    )
                ),
                "completedAt": utc_now(),
            },
        )
        return before, after

    def _load_embeddings(
        self, task: TaskSpec, expected_image_ids: Iterable[str]
    ) -> tuple[Any, Any, dict[str, Any]]:
        # run_retrieval_harness.configure mutates module globals.  Serialize it
        # with the Weighted Fusion source loader so concurrent users cannot
        # observe another task's adapter while either source is being opened.
        with self._weighted_fusion_context_guard:
            return self._load_embeddings_locked(task, expected_image_ids)

    def _load_embeddings_locked(
        self, task: TaskSpec, expected_image_ids: Iterable[str]
    ) -> tuple[Any, Any, dict[str, Any]]:
        import numpy as np

        experiment_root = self.web_root.parents[2]
        linear_root = experiment_root / "probe_learning"
        script_root = linear_root / "scripts"
        pcp_root = self.web_root.parent
        for path in (linear_root, script_root, pcp_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import build_score_rank_pcp as pcp
        import run_retrieval_harness as harness

        harness.configure(task.dataset_id, task.task_name)
        adapter = harness.ADAPTER
        paths, gallery_indices = pcp.split_indices(adapter, "gallery")
        path_ids = tuple(str(value) for value in paths)
        expected = tuple(expected_image_ids)
        if path_ids != expected:
            raise RuntimeError(f"Source embedding order does not match bundle for {task.task_id}")
        full_embeddings, meta = harness.load_backbone_embeddings(adapter, "siglip")
        full_embeddings = normalize_rows(full_embeddings)
        identity = np.arange(len(gallery_indices), dtype=np.int64)
        embeddings = (
            full_embeddings
            if np.array_equal(gallery_indices, identity)
            else full_embeddings[gallery_indices]
        )
        query_indices = np.asarray(adapter.query_idx, dtype=np.int64)
        if query_indices.size == 0:
            raise RuntimeError(f"Task has no fixed Query embeddings: {task.task_id}")
        base_prototype = normalize_rows(
            full_embeddings[query_indices].mean(axis=0, keepdims=True)
        )[0]
        return embeddings, base_prototype, {
            "backbone": "siglip",
            "dimension": int(embeddings.shape[1]),
            "queryCount": int(query_indices.size),
            "source": meta,
        }


class TuningHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: TuningService):
        self.service = service
        super().__init__(address, TuningRequestHandler)


class TuningRequestHandler(BaseHTTPRequestHandler):
    server: TuningHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "PCPTuning/1"

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _dispatch(self, method: str) -> None:
        try:
            request_url = urlparse(self.path)
            path = request_url.path.rstrip("/") or "/"
            if not path.startswith(API_PREFIX):
                raise ApiError(HTTPStatus.NOT_FOUND, "Endpoint not found")
            suffix = path[len(API_PREFIX) :]
            segments = [unquote(value) for value in suffix.split("/") if value]
            if method in {"POST", "PUT", "DELETE"}:
                self._validate_origin()

            if method == "GET" and segments == ["health"]:
                from frozen_validation import active_validation_info
                from fixed_vqa_validation import active_vqa_validation_info

                self._json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "database": "sqlite-wal",
                        "worker": "serial",
                        "taskCount": len(self.server.service.tasks),
                        "weightAlgorithms": {
                            mode: RUN_ALGORITHM_VERSIONS[mode]
                            for mode in sorted(WEIGHT_REFINEMENT_RUN_MODES)
                        },
                        "normalizationPolicy": REFINEMENT_NORMALIZATION_POLICY,
                        "cleanValidation": active_validation_info(self.server.service),
                        "vqaValidation": active_vqa_validation_info(self.server.service),
                    },
                )
                return

            if (method == "GET" and len(segments) == 3
                    and segments[0] == "tasks" and segments[2] == "refinement-capabilities"):
                self._json(HTTPStatus.OK, {
                    "taskId": segments[1],
                    "refinementCapabilities": self.server.service.refinement_capabilities(segments[1]),
                })
                return
            if (method == "GET" and len(segments) in (3, 4)
                    and segments[0] == "tasks" and segments[2] == "initial-baseline"):
                import unified_initial_baseline as initial

                if len(segments) == 3:
                    self._json(HTTPStatus.OK, initial.descriptor(self.server.service, segments[1]))
                    return
                query = parse_qs(request_url.query, keep_blank_values=False)
                fingerprints = query.get("fingerprint", [])
                if len(fingerprints) != 1 or not re.fullmatch(r"[a-f0-9]{64}", fingerprints[0]):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "A pinned baseline fingerprint is required")
                fingerprint = fingerprints[0]
                headers = {"X-PCP-Baseline-Fingerprint": fingerprint, "X-PCP-Base-Fingerprint": fingerprint}
                try:
                    if segments[3] == "visualization":
                        view = initial.visualization(self.server.service, segments[1], fingerprint)
                        self._binary_payload(HTTPStatus.OK, view.payload, headers={**headers,
                            "X-PCP-Algorithm": REFINEMENT_VISUALIZATION_ALGORITHM, "X-PCP-Layout": REFINEMENT_VISUALIZATION_LAYOUT,
                            "X-PCP-Row-Count": str(view.row_count), "X-PCP-Component-Count": str(view.component_count),
                            "X-PCP-Source-Fingerprint": view.source_fingerprint})
                        return
                    if segments[3] == "clusters":
                        schemes = query.get("scheme", [])
                        if len(schemes) != 1:
                            raise ValueError("A cluster scheme is required")
                        view, result = initial.clusters(self.server.service, segments[1], fingerprint, schemes[0])
                        self._binary_payload(HTTPStatus.OK, result.payload, allow_gzip=False, headers={**headers,
                            "X-PCP-Algorithm": REFINEMENT_CLUSTER_ALGORITHM, "X-PCP-Layout": "labels",
                            "X-PCP-Source-Fingerprint": view.source_fingerprint, "X-PCP-Cluster-Scheme": result.scheme,
                            "X-PCP-Row-Count": str(result.row_count), "X-PCP-Cluster-Count": str(result.cluster_count),
                            "X-PCP-Feature-Basis": REFINEMENT_CLUSTER_FEATURE_BASIS, "X-PCP-Feature-Count": str(result.feature_count),
                            "X-PCP-Fit-Scope": "development", "X-PCP-Fit-Row-Count": str(result.fit_row_count)})
                        return
                    if segments[3] in {"scores", "ranks"}:
                        loaded = initial.load_task_baseline(self.server.service, segments[1], fingerprint=fingerprint)
                        if loaded is None:
                            raise RuntimeError("Initial baseline is pending publication")
                        self._binary_payload(HTTPStatus.OK, (loaded[2] / f"{segments[3]}.f32").read_bytes(), headers=headers)
                        return
                except (RuntimeError, ValueError) as error:
                    raise ApiError(HTTPStatus.CONFLICT, str(error)) from error
            if (method == "GET" and len(segments) == 3
                    and segments[0] == "tasks" and segments[2] == "validation"):
                self._json(HTTPStatus.OK, self.server.service.task_validation(segments[1]))
                return

            if method == "GET" and len(segments) == 3 and segments[0] == "images":
                try:
                    row_index = int(segments[2])
                except ValueError as exc:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "rowIndex must be an integer") from exc
                image_path = self.server.service.gallery_image(segments[1], row_index)
                self._image(image_path)
                return

            if (
                method == "GET"
                and len(segments) == 5
                and segments[0] == "tasks"
                and segments[2] == "targets"
                and segments[4] == "original-supervision"
            ):
                result = self.server.service.original_vqa_supervision(
                    segments[1], segments[3]
                )
                self._json(HTTPStatus.OK, result)
                return

            if method == "POST" and segments == ["bootstrap"]:
                payload = self._read_json()
                display_name = payload.get("displayName")
                if display_name is not None and not isinstance(display_name, str):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "displayName must be a string")
                user, _ = self.server.service.identify(
                    self.headers, create=True, display_name=display_name
                )
                response = self.server.service.bootstrap(user, payload)
                secure = (self.headers.get("x-forwarded-proto") or "").lower() == "https"
                self._json(
                    HTTPStatus.OK,
                    response,
                    headers={
                        "Set-Cookie": self.server.service.cookie_header(user["id"], secure)
                    },
                )
                return

            if method == "POST" and segments == ["weighted-fusion"]:
                payload = self._read_json()
                task_id = payload.get("taskId")
                weights = payload.get("weights")
                if not isinstance(task_id, str) or not task_id:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "taskId must be a string")
                if not isinstance(weights, dict):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "weights must be an object")
                try:
                    normalized_weights = normalize_fusion_weights(weights)
                except ValueError as error:
                    raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
                result = self.server.service.weighted_fusion(
                    task_id,
                    normalized_weights,
                )
                headers = {
                    "X-PCP-Algorithm": "weighted-ours-full-v1",
                    "X-PCP-Layout": "raw,calibrated,rank",
                    "X-PCP-Row-Count": str(result.row_count),
                    "X-PCP-Target-Count": str(result.target_count),
                    "X-PCP-Equal-Weights": "1" if result.equal_weights else "0",
                }
                if result.baseline_max_abs_error is not None:
                    headers["X-PCP-Baseline-Max-Abs-Error"] = (
                        f"{result.baseline_max_abs_error:.9g}"
                    )
                self._binary_payload(HTTPStatus.OK, result.payload, headers=headers)
                return

            if method == "POST" and segments == ["hierarchical-fusion"]:
                payload = self._read_json()
                task_id = payload.get("taskId")
                attribute_weights = payload.get("attributeWeights")
                learner_weights_by_attribute = payload.get("learnerWeightsByAttribute")
                global_weights = payload.get("globalWeights")
                if not isinstance(task_id, str) or not task_id:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "taskId must be a string")
                if not isinstance(attribute_weights, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "attributeWeights must be an object",
                    )
                if not isinstance(learner_weights_by_attribute, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "learnerWeightsByAttribute must be an object",
                    )
                if global_weights is not None and not isinstance(global_weights, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "globalWeights must be an object",
                    )
                task = self.server.service.task(task_id)
                attribute_ids = tuple(
                    str(target.get("id"))
                    for target in task.manifest.get("retrievalTargets", [])
                    if target.get("kind") == "attribute"
                )
                try:
                    normalized_attributes, normalized_learners = (
                        normalize_hierarchical_fusion_weights(
                            attribute_ids,
                            attribute_weights,
                            learner_weights_by_attribute,
                        )
                    )
                    normalized_global = normalize_global_fusion_weights(global_weights)
                except ValueError as error:
                    raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
                result = self.server.service.hierarchical_fusion(
                    task_id,
                    normalized_attributes,
                    normalized_learners,
                    normalized_global,
                )
                headers = {
                    "X-PCP-Algorithm": HIERARCHICAL_FUSION_ALGORITHM,
                    "X-PCP-Layout": (
                        "raw,calibrated,rank,hierarchy-calibrated,hierarchy-rank"
                    ),
                    "X-PCP-Row-Count": str(result.row_count),
                    "X-PCP-Target-Count": str(result.target_count),
                    "X-PCP-Baseline": "1" if result.baseline else "0",
                }
                if result.baseline_max_abs_error is not None:
                    headers["X-PCP-Baseline-Max-Abs-Error"] = (
                        f"{result.baseline_max_abs_error:.9g}"
                    )
                self._binary_payload(HTTPStatus.OK, result.payload, headers=headers)
                return

            if method == "POST" and segments == ["pcp-clusters"]:
                payload = self._read_json()
                task_id = payload.get("taskId")
                attribute_weights = payload.get("attributeWeights")
                learner_weights_by_attribute = payload.get("learnerWeightsByAttribute")
                global_weights = payload.get("globalWeights")
                scheme = payload.get("scheme")
                if not isinstance(task_id, str) or not task_id:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "taskId must be a string")
                if not isinstance(attribute_weights, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "attributeWeights must be an object",
                    )
                if not isinstance(learner_weights_by_attribute, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "learnerWeightsByAttribute must be an object",
                    )
                if global_weights is not None and not isinstance(global_weights, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "globalWeights must be an object",
                    )
                if not isinstance(scheme, str):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "scheme must be a string")
                task = self.server.service.task(task_id)
                attribute_ids = tuple(
                    str(target.get("id"))
                    for target in task.manifest.get("retrievalTargets", [])
                    if target.get("kind") == "attribute"
                )
                try:
                    normalized_attributes, normalized_learners = (
                        normalize_hierarchical_fusion_weights(
                            attribute_ids,
                            attribute_weights,
                            learner_weights_by_attribute,
                        )
                    )
                    normalized_global = normalize_global_fusion_weights(global_weights)
                    result = self.server.service.pcp_clusters(
                        task_id,
                        normalized_attributes,
                        normalized_learners,
                        normalized_global,
                        scheme,
                    )
                except ValueError as error:
                    raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
                headers = {
                    "X-PCP-Algorithm": DYNAMIC_PCP_CLUSTER_ALGORITHM,
                    "X-PCP-Layout": "labels",
                    "X-PCP-Row-Count": str(result.row_count),
                    "X-PCP-Cluster-Scheme": result.scheme,
                    "X-PCP-Cluster-Count": str(result.cluster_count),
                    "X-PCP-Feature-Basis": DYNAMIC_PCP_CLUSTER_FEATURE_BASIS,
                    "X-PCP-Feature-Count": str(result.feature_count),
                    "X-PCP-Fit-Scope": "development",
                    "X-PCP-Fit-Row-Count": str(result.fit_row_count),
                }
                self._binary_payload(
                    HTTPStatus.OK,
                    result.payload,
                    headers=headers,
                    allow_gzip=False,
                )
                return

            if method == "POST" and segments == ["rank-fusion"]:
                payload = self._read_json()
                task_id = payload.get("taskId")
                attribute_weights = payload.get("attributeWeights")
                method_weights_by_attribute = payload.get(
                    "methodWeightsByAttribute"
                )
                if not isinstance(task_id, str) or not task_id:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "taskId must be a string")
                if not isinstance(attribute_weights, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "attributeWeights must be an object",
                    )
                if not isinstance(method_weights_by_attribute, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "methodWeightsByAttribute must be an object",
                    )
                task = self.server.service.task(task_id)
                attribute_ids = tuple(
                    str(target.get("id"))
                    for target in task.manifest.get("retrievalTargets", [])
                    if target.get("kind") == "attribute"
                )
                try:
                    normalized_attributes, normalized_methods = (
                        normalize_rank_fusion_weights(
                            attribute_ids,
                            attribute_weights,
                            method_weights_by_attribute,
                        )
                    )
                except ValueError as error:
                    raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
                result = self.server.service.rank_fusion(
                    task_id,
                    normalized_attributes,
                    normalized_methods,
                )
                self._binary_payload(
                    HTTPStatus.OK,
                    result.payload,
                    headers={
                        "X-PCP-Algorithm": RANK_FUSION_ALGORITHM,
                        "X-PCP-Layout": "raw,calibrated,rank",
                        "X-PCP-Row-Count": str(result.row_count),
                        "X-PCP-Target-Count": str(result.target_count),
                    },
                )
                return

            if method == "POST" and segments == ["rank-fusion-clusters"]:
                payload = self._read_json()
                task_id = payload.get("taskId")
                attribute_weights = payload.get("attributeWeights")
                method_weights_by_attribute = payload.get(
                    "methodWeightsByAttribute"
                )
                scheme = payload.get("scheme")
                if not isinstance(task_id, str) or not task_id:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "taskId must be a string")
                if not isinstance(attribute_weights, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "attributeWeights must be an object",
                    )
                if not isinstance(method_weights_by_attribute, dict):
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "methodWeightsByAttribute must be an object",
                    )
                if not isinstance(scheme, str):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "scheme must be a string")
                task = self.server.service.task(task_id)
                attribute_ids = tuple(
                    str(target.get("id"))
                    for target in task.manifest.get("retrievalTargets", [])
                    if target.get("kind") == "attribute"
                )
                try:
                    normalized_attributes, normalized_methods = (
                        normalize_rank_fusion_weights(
                            attribute_ids,
                            attribute_weights,
                            method_weights_by_attribute,
                        )
                    )
                    result = self.server.service.rank_fusion_clusters(
                        task_id,
                        normalized_attributes,
                        normalized_methods,
                        scheme,
                    )
                except ValueError as error:
                    raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
                self._binary_payload(
                    HTTPStatus.OK,
                    result.payload,
                    headers={
                        "X-PCP-Algorithm": RANK_FUSION_CLUSTER_ALGORITHM,
                        "X-PCP-Layout": "labels",
                        "X-PCP-Row-Count": str(result.row_count),
                        "X-PCP-Cluster-Scheme": result.scheme,
                        "X-PCP-Cluster-Count": str(result.cluster_count),
                        "X-PCP-Feature-Basis": RANK_FUSION_CLUSTER_FEATURE_BASIS,
                        "X-PCP-Feature-Count": str(result.feature_count),
                        "X-PCP-Fit-Scope": "development",
                        "X-PCP-Fit-Row-Count": str(result.fit_row_count),
                    },
                    allow_gzip=False,
                )
                return

            user, _ = self.server.service.identify(self.headers, create=False)
            if len(segments) == 3 and segments[0] == "sessions" and segments[2] == "probe-updates":
                if method == "POST":
                    result = self.server.service.create_probe_update(user["id"], segments[1], self._read_json())
                    self._json(HTTPStatus.ACCEPTED, {"probeUpdate": result})
                    return
                if method == "GET":
                    self._json(HTTPStatus.OK, {"probeUpdates": self.server.service.list_probe_updates(user["id"], segments[1])})
                    return
            if method == "GET" and len(segments) == 2 and segments[0] == "probe-updates":
                row = self.server.service.owned_probe_update(segments[1], user["id"])
                self._json(HTTPStatus.OK, {"probeUpdate": self.server.service.probe_update_json(row)})
                return
            if (
                method == "POST"
                and len(segments) == 4
                and segments[0] == "sessions"
                and segments[2] == "annotations"
                and segments[3] == "bulk"
            ):
                result = self.server.service.put_annotations_bulk(
                    user["id"], segments[1], self._read_json()
                )
                self._json(HTTPStatus.OK, result)
                return
            if (
                len(segments) == 4
                and segments[0] == "sessions"
                and segments[2] == "annotations"
            ):
                session_id = segments[1]
                try:
                    row_index = int(segments[3])
                except ValueError as exc:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "rowIndex must be an integer") from exc
                if method == "PUT":
                    result = self.server.service.put_annotation(
                        user["id"], session_id, row_index, self._read_json()
                    )
                    self._json(HTTPStatus.OK, result)
                    return
                if method == "DELETE":
                    result = self.server.service.delete_annotation(
                        user["id"], session_id, row_index
                    )
                    self._json(HTTPStatus.OK, result)
                    return

            if (
                method == "POST"
                and len(segments) == 3
                and segments[0] == "sessions"
                and segments[2] == "runs"
            ):
                payload = self._read_json()
                if str(payload.get("mode") or "") not in REFINEMENT_RUN_MODES:
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        "Historical tuning modes are read-only; create a Weight-only refinement instead",
                    )
                result = self.server.service.create_run(
                    user["id"], segments[1], payload
                )
                self._json(HTTPStatus.ACCEPTED, {"run": result})
                return

            if len(segments) >= 2 and segments[0] == "runs":
                run = self.server.service.owned_run(segments[1], user["id"])
                if method == "GET" and len(segments) == 2:
                    self._json(HTTPStatus.OK, {"run": self.server.service.run_json(run)})
                    return
                if (
                    method == "GET"
                    and len(segments) == 3
                    and segments[2] == "refinement-visualization"
                ):
                    result = self.server.service.refinement_visualization(run)
                    self._binary_payload(
                        HTTPStatus.OK,
                        result.payload,
                        headers={
                            "X-PCP-Algorithm": REFINEMENT_VISUALIZATION_ALGORITHM,
                            "X-PCP-Layout": REFINEMENT_VISUALIZATION_LAYOUT,
                            "X-PCP-Run-Id": str(run["id"]),
                            "X-PCP-Row-Count": str(result.row_count),
                            "X-PCP-Component-Count": str(result.component_count),
                            "X-PCP-Source-Fingerprint": result.source_fingerprint,
                            "X-PCP-Base-Fingerprint": result.base_state_fingerprint,
                        },
                    )
                    return
                if (
                    method == "GET"
                    and len(segments) == 3
                    and segments[2] == "refinement-clusters"
                ):
                    query = parse_qs(request_url.query, keep_blank_values=False)
                    scheme_values = query.get("scheme", [])
                    if len(scheme_values) != 1:
                        raise ApiError(HTTPStatus.BAD_REQUEST, "scheme is required")
                    scheme = scheme_values[0]
                    try:
                        visualization = self.server.service.refinement_visualization(run)
                        result = self.server.service.refinement_clusters(run, scheme)
                    except ValueError as error:
                        raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
                    self._binary_payload(
                        HTTPStatus.OK,
                        result.payload,
                        headers={
                            "X-PCP-Algorithm": REFINEMENT_CLUSTER_ALGORITHM,
                            "X-PCP-Layout": "labels",
                            "X-PCP-Run-Id": str(run["id"]),
                            "X-PCP-Source-Fingerprint": visualization.source_fingerprint,
                            "X-PCP-Cluster-Scheme": result.scheme,
                            "X-PCP-Row-Count": str(result.row_count),
                            "X-PCP-Cluster-Count": str(result.cluster_count),
                            "X-PCP-Feature-Basis": REFINEMENT_CLUSTER_FEATURE_BASIS,
                            "X-PCP-Feature-Count": str(result.feature_count),
                            "X-PCP-Fit-Scope": "development",
                            "X-PCP-Fit-Row-Count": str(result.fit_row_count),
                        },
                        allow_gzip=False,
                    )
                    return
                if method == "GET" and len(segments) == 3:
                    artifact = self.server.service.artifact(run, segments[2])
                    self._binary(artifact)
                    return

            raise ApiError(HTTPStatus.NOT_FOUND, "Endpoint not found")
        except ApiError as error:
            self._json(error.status, {"error": error.message})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            traceback.print_exc()
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Internal tuning service error: {type(error).__name__}"},
            )

    def _read_json(self) -> dict[str, Any]:
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from exc
        if length <= 0 or length > MAX_JSON_BYTES:
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON body size is invalid")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid JSON body") from exc
        if not isinstance(value, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
        return value

    def _validate_origin(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            return
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != self.headers.get("Host"):
            raise ApiError(HTTPStatus.FORBIDDEN, "Cross-origin state changes are not allowed")

    def _json(
        self,
        status: int,
        value: Any,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        payload = (json_dumps(value) + "\n").encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        for key, item in (headers or {}).items():
            self.send_header(key, item)
        self.end_headers()
        self.wfile.write(payload)

    def _binary(self, path: Path) -> None:
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    def _binary_payload(
        self,
        status: int,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
        allow_gzip: bool = True,
    ) -> None:
        use_gzip = allow_gzip and len(payload) >= 64 * 1024 and "gzip" in (
            self.headers.get("Accept-Encoding") or ""
        ).lower()
        body = gzip.compress(payload, compresslevel=1, mtime=0) if use_gzip else payload
        self.send_response(int(status))
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("Content-Length", str(len(body)))
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _image(self, path: Path) -> None:
        stat = path.stat()
        etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "private, max-age=86400")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "private, max-age=86400")
        self.send_header("ETag", etag)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.send_header("Content-Length", str(stat.st_size))
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            f"[{utc_now()}] {self.client_address[0]} {format % args}\n"
        )


def parse_args() -> argparse.Namespace:
    default_web_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--web-root", type=Path, default=default_web_root)
    parser.add_argument("--runtime-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("The tuning sidecar must bind to loopback; expose it through Vite")
    service = TuningService(args.web_root, args.runtime_root)
    server = TuningHTTPServer((args.host, args.port), service)
    print(
        f"PCP tuning service ready at http://{args.host}:{server.server_address[1]}{API_PREFIX}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        service.stop_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
