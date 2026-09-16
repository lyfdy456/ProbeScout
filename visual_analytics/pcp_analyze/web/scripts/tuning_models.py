"""Low-dimensional personal tuning models for the PCP workbench.

All fusion models operate on frozen learner outputs. System two may first
update the original probes in its separate native-training phase; this module
never backpropagates the fusion objective into those probe parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize


EPSILON = 1e-6


def stable_sigmoid(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    non_negative = array >= 0.0
    output[non_negative] = 1.0 / (1.0 + np.exp(-array[non_negative]))
    exponent = np.exp(array[~non_negative])
    output[~non_negative] = exponent / (1.0 + exponent)
    return output


def stable_logit(probabilities: Any) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(values) - np.log1p(-values)


def class_balanced_weights(labels: Any, sample_weights: Any) -> np.ndarray:
    """Balance positive/negative total mass while preserving per-row emphasis."""
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if truth.shape != weights.shape or truth.size == 0:
        raise ValueError("labels and sample weights must be non-empty and aligned")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("tuning labels must be binary")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sample weights must be finite and positive")
    positive = truth == 1
    negative = ~positive
    if not np.any(positive) or not np.any(negative):
        raise ValueError("tuning requires both positive and negative supervision")
    balanced = weights.copy()
    target_mass = float(weights.sum()) / 2.0
    balanced[positive] *= target_mass / float(weights[positive].sum())
    balanced[negative] *= target_mass / float(weights[negative].sum())
    return balanced


def weighted_binary_cross_entropy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    truth = np.asarray(labels, dtype=np.float64)
    losses = -(truth * np.log(values) + (1.0 - truth) * np.log1p(-values))
    return float(np.average(losses, weights=sample_weights))


def weighted_binary_cross_entropy_with_logits(
    labels: Any,
    logits: Any,
    sample_weights: Any,
) -> float:
    """Numerically stable weighted Bernoulli loss from unconstrained logits."""

    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if truth.shape != values.shape or truth.shape != weights.shape:
        raise ValueError("labels, logits and sample weights must have the same shape")
    if np.any(~np.isin(truth, (0.0, 1.0))):
        raise ValueError("labels must be binary")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample weights must be finite and positive")
    losses = np.logaddexp(0.0, values) - truth * values
    return float(np.average(losses, weights=weights))


@dataclass(frozen=True)
class FusionWeightModel:
    weights: np.ndarray
    bias: float
    initial_objective: float
    final_objective: float
    iterations: int


@dataclass(frozen=True)
class AttributeFusionWeightModel:
    """Per-attribute learner mixtures with inactive rows kept at equal weight."""

    weights: np.ndarray
    bias: float
    active_attribute_indices: tuple[int, ...]
    initial_objective: float
    final_objective: float
    iterations: int


@dataclass(frozen=True)
class AttributeJointFusionWeightModel:
    """Per-attribute learner mixtures plus mean-one Joint exponents.

    ``attribute_weights`` are the non-negative exponents used by the Joint
    product.  Inactive exponents remain exactly one, just as inactive learner
    rows remain exactly equal-weight.  Keeping those two active sets explicit
    prevents a single-attribute run from presenting unidentifiable Joint
    weights as learned parameters.
    """

    weights: np.ndarray
    attribute_weights: np.ndarray
    bias: float
    active_attribute_indices: tuple[int, ...]
    active_joint_attribute_indices: tuple[int, ...]
    initial_objective: float
    final_objective: float
    iterations: int


def fit_fusion_weights(
    score_function: Callable[[np.ndarray], Any],
    labels: Any,
    sample_weights: Any,
    *,
    member_count: int,
    regularization_lambda: float = 0.5,
    bias_regularization_eta: float = 0.1,
    max_iterations: int = 500,
) -> FusionWeightModel:
    """Fit non-negative simplex learner weights around the equal-weight fusion.

    ``score_function`` must return the frozen SoftGate probability for every
    supervision row.  Only the eight mixture weights and a calibration bias are
    optimized; the underlying learner scores, train MinMax values, theta and
    temperature remain frozen.
    """
    if member_count < 2:
        raise ValueError("fusion tuning requires at least two members")
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    weights = class_balanced_weights(truth, sample_weights)
    initial_weights = np.full(member_count, 1.0 / member_count, dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        mixture = np.asarray(parameters[:member_count], dtype=np.float64)
        bias = float(parameters[-1])
        probabilities = np.asarray(score_function(mixture), dtype=np.float64).reshape(-1)
        if probabilities.shape != truth.shape or not np.all(np.isfinite(probabilities)):
            return float("inf")
        calibrated = stable_sigmoid(stable_logit(probabilities) + bias)
        data_loss = weighted_binary_cross_entropy(truth, calibrated, weights)
        regularizer = (
            float(regularization_lambda) * float(np.sum((mixture - initial_weights) ** 2))
            + float(bias_regularization_eta) * bias * bias
        )
        return data_loss + regularizer

    initial = np.concatenate([initial_weights, np.zeros(1, dtype=np.float64)])
    initial_objective = objective(initial)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * member_count + [(-8.0, 8.0)],
        constraints=[
            {
                "type": "eq",
                "fun": lambda parameters: float(np.sum(parameters[:member_count]) - 1.0),
            }
        ],
        options={"maxiter": int(max_iterations), "ftol": 1e-9, "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Fusion weight optimizer failed: {result.message}")
    learned = np.clip(np.asarray(result.x[:member_count], dtype=np.float64), 0.0, 1.0)
    learned /= float(learned.sum())
    final_objective = objective(np.concatenate([learned, [float(result.x[-1])]]))
    if final_objective > initial_objective + 1e-8:
        raise RuntimeError("Fusion weight optimizer returned a worse objective")
    return FusionWeightModel(
        weights=learned.astype(np.float32),
        bias=float(result.x[-1]),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        iterations=int(result.nit),
    )


def fit_attribute_fusion_weights(
    score_function: Callable[[np.ndarray], Any],
    labels: Any,
    sample_weights: Any,
    *,
    attribute_count: int,
    member_count: int,
    active_attribute_indices: Any | None = None,
    regularization_lambda: float = 0.5,
    shared_regularization_lambda: float = 0.25,
    bias_regularization_eta: float = 0.1,
    max_iterations: int = 500,
) -> AttributeFusionWeightModel:
    """Fit one non-negative learner simplex per active attribute.

    ``score_function`` receives the complete ``attribute_count x member_count``
    matrix and must return the frozen SoftGate probability for each supervision
    row.  Joint tuning activates every attribute row.  A single-attribute
    target activates only its own row, leaving all other rows at the exact
    equal-weight anchor so unidentifiable parameters are never presented as
    learned.

    The shared-row penalty is useful when the supervision set is small: it
    allows attribute-specific mixtures while shrinking them toward their mean
    learner profile.  It vanishes when only one attribute is active.
    """
    if attribute_count < 1:
        raise ValueError("attribute fusion tuning requires at least one attribute")
    if member_count < 2:
        raise ValueError("attribute fusion tuning requires at least two members")
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    balanced_weights = class_balanced_weights(truth, sample_weights)
    if active_attribute_indices is None:
        active = tuple(range(attribute_count))
    else:
        active = tuple(int(value) for value in active_attribute_indices)
    if not active or len(set(active)) != len(active):
        raise ValueError("active attribute indices must be non-empty and unique")
    if any(index < 0 or index >= attribute_count for index in active):
        raise ValueError("active attribute index is outside the weight matrix")
    if not np.isfinite(regularization_lambda) or regularization_lambda < 0.0:
        raise ValueError("regularization_lambda must be finite and non-negative")
    if (
        not np.isfinite(shared_regularization_lambda)
        or shared_regularization_lambda < 0.0
    ):
        raise ValueError(
            "shared_regularization_lambda must be finite and non-negative"
        )
    if not np.isfinite(bias_regularization_eta) or bias_regularization_eta < 0.0:
        raise ValueError("bias_regularization_eta must be finite and non-negative")

    equal_row = np.full(member_count, 1.0 / member_count, dtype=np.float64)
    equal_matrix = np.tile(equal_row, (attribute_count, 1))
    active_count = len(active)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, float]:
        matrix = equal_matrix.copy()
        active_values = np.asarray(parameters[:-1], dtype=np.float64).reshape(
            active_count, member_count
        )
        for position, attribute_index in enumerate(active):
            matrix[attribute_index] = active_values[position]
        return matrix, float(parameters[-1])

    def objective(parameters: np.ndarray) -> float:
        matrix, bias = unpack(parameters)
        probabilities = np.asarray(score_function(matrix), dtype=np.float64).reshape(-1)
        if probabilities.shape != truth.shape or not np.all(np.isfinite(probabilities)):
            return float("inf")
        calibrated = stable_sigmoid(stable_logit(probabilities) + bias)
        data_loss = weighted_binary_cross_entropy(
            truth, calibrated, balanced_weights
        )
        active_matrix = matrix[np.asarray(active, dtype=np.int64)]
        anchor_penalty = float(
            np.mean(np.sum((active_matrix - equal_row[None, :]) ** 2, axis=1))
        )
        if active_count > 1:
            shared_profile = np.mean(active_matrix, axis=0, keepdims=True)
            shared_penalty = float(
                np.mean(np.sum((active_matrix - shared_profile) ** 2, axis=1))
            )
        else:
            shared_penalty = 0.0
        regularizer = (
            float(regularization_lambda) * anchor_penalty
            + float(shared_regularization_lambda) * shared_penalty
            + float(bias_regularization_eta) * bias * bias
        )
        return data_loss + regularizer

    initial = np.concatenate(
        [np.tile(equal_row, active_count), np.zeros(1, dtype=np.float64)]
    )
    initial_objective = objective(initial)
    constraints = []
    for position in range(active_count):
        start = position * member_count
        stop = start + member_count
        constraints.append(
            {
                "type": "eq",
                "fun": lambda parameters, start=start, stop=stop: float(
                    np.sum(parameters[start:stop]) - 1.0
                ),
            }
        )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (active_count * member_count) + [(-8.0, 8.0)],
        constraints=constraints,
        options={"maxiter": int(max_iterations), "ftol": 1e-9, "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Attribute fusion weight optimizer failed: {result.message}")
    learned, bias = unpack(np.asarray(result.x, dtype=np.float64))
    for attribute_index in active:
        row = np.clip(learned[attribute_index], 0.0, 1.0)
        total = float(row.sum())
        if total <= 0.0:
            raise RuntimeError("Attribute fusion optimizer returned an empty simplex")
        learned[attribute_index] = row / total
    final_parameters = np.concatenate(
        [learned[np.asarray(active, dtype=np.int64)].reshape(-1), [bias]]
    )
    final_objective = objective(final_parameters)
    if final_objective > initial_objective + 1e-8:
        raise RuntimeError("Attribute fusion optimizer returned a worse objective")
    return AttributeFusionWeightModel(
        weights=learned.astype(np.float32),
        bias=bias,
        active_attribute_indices=active,
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        iterations=int(result.nit),
    )


def fit_attribute_and_joint_fusion_weights(
    score_function: Callable[[np.ndarray, np.ndarray], Any],
    labels: Any,
    sample_weights: Any,
    *,
    attribute_count: int,
    member_count: int,
    active_attribute_indices: Any | None = None,
    active_joint_attribute_indices: Any | None = None,
    regularization_lambda: float = 0.5,
    shared_regularization_lambda: float = 0.25,
    joint_regularization_lambda: float = 0.5,
    bias_regularization_eta: float = 0.1,
    max_iterations: int = 500,
) -> AttributeJointFusionWeightModel:
    """Fit learner simplexes and non-negative mean-one Joint exponents.

    The score callback receives ``(learner_weights, attribute_weights)`` where
    the first value has shape ``attribute_count x member_count`` and the second
    has shape ``attribute_count``.  A Joint target should activate every Joint
    exponent; a single-attribute target should pass an empty active Joint set,
    leaving every exponent exactly one because those parameters do not affect
    a single-attribute score.

    Active Joint exponents are constrained to keep their mean at one.  This
    removes the otherwise arbitrary common scale of
    ``product(gate[a] ** attribute_weight[a])`` while still allowing the
    relative importance of each attribute to be learned.
    """
    if attribute_count < 1:
        raise ValueError("attribute fusion tuning requires at least one attribute")
    if member_count < 2:
        raise ValueError("attribute fusion tuning requires at least two members")
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    balanced_weights = class_balanced_weights(truth, sample_weights)

    if active_attribute_indices is None:
        active = tuple(range(attribute_count))
    else:
        active = tuple(int(value) for value in active_attribute_indices)
    if not active or len(set(active)) != len(active):
        raise ValueError("active attribute indices must be non-empty and unique")
    if any(index < 0 or index >= attribute_count for index in active):
        raise ValueError("active attribute index is outside the weight matrix")

    if active_joint_attribute_indices is None:
        active_joint = tuple(range(attribute_count))
    else:
        active_joint = tuple(int(value) for value in active_joint_attribute_indices)
    if len(set(active_joint)) != len(active_joint):
        raise ValueError("active Joint attribute indices must be unique")
    if any(index < 0 or index >= attribute_count for index in active_joint):
        raise ValueError("active Joint attribute index is outside the weight vector")

    regularizers = {
        "regularization_lambda": regularization_lambda,
        "shared_regularization_lambda": shared_regularization_lambda,
        "joint_regularization_lambda": joint_regularization_lambda,
        "bias_regularization_eta": bias_regularization_eta,
    }
    for name, value in regularizers.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    equal_row = np.full(member_count, 1.0 / member_count, dtype=np.float64)
    equal_matrix = np.tile(equal_row, (attribute_count, 1))
    initial_joint_weights = np.ones(attribute_count, dtype=np.float64)
    active_count = len(active)
    joint_active_count = len(active_joint)
    learner_parameter_count = active_count * member_count

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        matrix = equal_matrix.copy()
        active_values = np.asarray(
            parameters[:learner_parameter_count], dtype=np.float64
        ).reshape(active_count, member_count)
        for position, attribute_index in enumerate(active):
            matrix[attribute_index] = active_values[position]
        joint_weights = initial_joint_weights.copy()
        joint_stop = learner_parameter_count + joint_active_count
        for position, attribute_index in enumerate(active_joint):
            joint_weights[attribute_index] = float(
                parameters[learner_parameter_count + position]
            )
        return matrix, joint_weights, float(parameters[joint_stop])

    def objective(parameters: np.ndarray) -> float:
        matrix, joint_weights, bias = unpack(parameters)
        probabilities = np.asarray(
            score_function(matrix, joint_weights), dtype=np.float64
        ).reshape(-1)
        if probabilities.shape != truth.shape or not np.all(np.isfinite(probabilities)):
            return float("inf")
        calibrated = stable_sigmoid(stable_logit(probabilities) + bias)
        data_loss = weighted_binary_cross_entropy(
            truth, calibrated, balanced_weights
        )
        active_matrix = matrix[np.asarray(active, dtype=np.int64)]
        anchor_penalty = float(
            np.mean(np.sum((active_matrix - equal_row[None, :]) ** 2, axis=1))
        )
        if active_count > 1:
            shared_profile = np.mean(active_matrix, axis=0, keepdims=True)
            shared_penalty = float(
                np.mean(np.sum((active_matrix - shared_profile) ** 2, axis=1))
            )
        else:
            shared_penalty = 0.0
        if joint_active_count:
            joint_penalty = float(
                np.mean(
                    (joint_weights[np.asarray(active_joint, dtype=np.int64)] - 1.0)
                    ** 2
                )
            )
        else:
            joint_penalty = 0.0
        regularizer = (
            float(regularization_lambda) * anchor_penalty
            + float(shared_regularization_lambda) * shared_penalty
            + float(joint_regularization_lambda) * joint_penalty
            + float(bias_regularization_eta) * bias * bias
        )
        return data_loss + regularizer

    initial = np.concatenate(
        [
            np.tile(equal_row, active_count),
            np.ones(joint_active_count, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        ]
    )
    initial_objective = objective(initial)
    constraints = []
    for position in range(active_count):
        start = position * member_count
        stop = start + member_count
        constraints.append(
            {
                "type": "eq",
                "fun": lambda parameters, start=start, stop=stop: float(
                    np.sum(parameters[start:stop]) - 1.0
                ),
            }
        )
    if joint_active_count:
        joint_start = learner_parameter_count
        joint_stop = joint_start + joint_active_count
        constraints.append(
            {
                "type": "eq",
                "fun": lambda parameters: float(
                    np.sum(parameters[joint_start:joint_stop]) - joint_active_count
                ),
            }
        )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=(
            [(0.0, 1.0)] * learner_parameter_count
            + [(0.0, float(max(1, joint_active_count)))] * joint_active_count
            + [(-8.0, 8.0)]
        ),
        constraints=constraints,
        options={"maxiter": int(max_iterations), "ftol": 1e-9, "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Attribute and Joint fusion weight optimizer failed: {result.message}"
        )
    learned, learned_joint, bias = unpack(np.asarray(result.x, dtype=np.float64))
    for attribute_index in active:
        row = np.clip(learned[attribute_index], 0.0, 1.0)
        total = float(row.sum())
        if total <= 0.0:
            raise RuntimeError("Attribute fusion optimizer returned an empty simplex")
        learned[attribute_index] = row / total
    if joint_active_count:
        active_joint_array = np.asarray(active_joint, dtype=np.int64)
        joint_values = np.maximum(learned_joint[active_joint_array], 0.0)
        joint_total = float(joint_values.sum())
        if joint_total <= 0.0:
            raise RuntimeError("Joint fusion optimizer returned empty attribute weights")
        learned_joint[active_joint_array] = joint_values * (
            joint_active_count / joint_total
        )
    final_parameters = np.concatenate(
        [
            learned[np.asarray(active, dtype=np.int64)].reshape(-1),
            learned_joint[np.asarray(active_joint, dtype=np.int64)],
            [bias],
        ]
    )
    final_objective = objective(final_parameters)
    if final_objective > initial_objective + 1e-8:
        raise RuntimeError(
            "Attribute and Joint fusion optimizer returned a worse objective"
        )
    return AttributeJointFusionWeightModel(
        weights=learned.astype(np.float32),
        attribute_weights=learned_joint.astype(np.float32),
        bias=bias,
        active_attribute_indices=active,
        active_joint_attribute_indices=active_joint,
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        iterations=int(result.nit),
    )


@dataclass(frozen=True)
class ResidualModel:
    coefficients: np.ndarray
    bias: float
    initial_objective: float
    final_objective: float
    iterations: int


def residual_probabilities(
    base_ranks: Any,
    learner_calibrated_scores: Any,
    base_calibrated_scores: Any,
    coefficients: Any,
    bias: float,
) -> np.ndarray:
    """Apply a small cross-learner correction to the frozen base-rank logit."""
    base_rank = np.asarray(base_ranks, dtype=np.float64).reshape(-1)
    learner = np.asarray(learner_calibrated_scores, dtype=np.float64)
    base_calibrated = np.asarray(base_calibrated_scores, dtype=np.float64).reshape(-1)
    beta = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if learner.ndim != 2 or learner.shape != (base_rank.size, beta.size):
        raise ValueError("residual learner feature shape is invalid")
    if base_calibrated.shape != base_rank.shape:
        raise ValueError("base calibrated score shape is invalid")
    features = learner - base_calibrated[:, None]
    logits = stable_logit(base_rank) + features @ beta + float(bias)
    return stable_sigmoid(logits)


def fit_residual(
    base_ranks: Any,
    learner_calibrated_scores: Any,
    base_calibrated_scores: Any,
    labels: Any,
    sample_weights: Any,
    *,
    regularization_lambda: float = 0.5,
    bias_regularization_eta: float = 0.1,
    max_iterations: int = 500,
) -> ResidualModel:
    """Fit an L2-regularized low-dimensional correction around a frozen rank."""
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    base_rank = np.asarray(base_ranks, dtype=np.float64).reshape(-1)
    learner = np.asarray(learner_calibrated_scores, dtype=np.float64)
    base_calibrated = np.asarray(base_calibrated_scores, dtype=np.float64).reshape(-1)
    if learner.ndim != 2 or learner.shape[0] != truth.size:
        raise ValueError("residual learner rows must align with labels")
    if base_rank.shape != truth.shape or base_calibrated.shape != truth.shape:
        raise ValueError("residual base arrays must align with labels")
    weights = class_balanced_weights(truth, sample_weights)
    member_count = int(learner.shape[1])

    def objective(parameters: np.ndarray) -> float:
        coefficients = parameters[:member_count]
        bias = float(parameters[-1])
        probabilities = residual_probabilities(
            base_rank, learner, base_calibrated, coefficients, bias
        )
        data_loss = weighted_binary_cross_entropy(truth, probabilities, weights)
        regularizer = (
            float(regularization_lambda) * float(np.sum(coefficients**2))
            + float(bias_regularization_eta) * bias * bias
        )
        return data_loss + regularizer

    initial = np.zeros(member_count + 1, dtype=np.float64)
    initial_objective = objective(initial)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * member_count + [(-8.0, 8.0)],
        options={"maxiter": int(max_iterations), "ftol": 1e-12, "maxls": 50},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Residual optimizer failed: {result.message}")
    final_objective = objective(np.asarray(result.x, dtype=np.float64))
    if final_objective > initial_objective + 1e-8:
        raise RuntimeError("Residual optimizer returned a worse objective")
    return ResidualModel(
        coefficients=np.asarray(result.x[:member_count], dtype=np.float32),
        bias=float(result.x[-1]),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        iterations=int(result.nit),
    )


# ---------------------------------------------------------------------------
# Unified conjunction-preserving fusion refinement (all four modes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnifiedWeightScores:
    """Components of the shared four-mode ranking formula.

    The input ``z`` is already normalized with the *fixed Development*
    min/max statistics.  The holistic embedding scores are likewise
    expected to be normalized independently to ``[0, 1]`` before this layer.
    """

    attribute_scores: np.ndarray
    gates: np.ndarray
    conjunction_scores: np.ndarray
    holistic_scores: np.ndarray
    final_scores: np.ndarray


@dataclass(frozen=True)
class WeightRefinementResult:
    """Fusion result; probe-updated modes consume an upstream frozen snapshot."""

    mode: str
    initial_scores: np.ndarray
    refined_scores: np.ndarray
    ranking_indices: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    embedding_weights: np.ndarray
    embedding_fusion_strength: float
    theta: np.ndarray
    temperature: np.ndarray
    initial_objective: float
    final_objective: float
    iterations: int
    training_history: tuple[dict[str, Any], ...]


def _validate_simplex(values: np.ndarray, *, axis: int, name: str) -> None:
    if not np.all(np.isfinite(values)) or np.any(values < -1e-8):
        raise ValueError(f"{name} must be finite and non-negative")
    totals = np.sum(values, axis=axis)
    if not np.allclose(totals, 1.0, atol=1e-6):
        raise ValueError(f"{name} must sum to one")


def _unified_embedding_count(embeddings: np.ndarray, row_count: int) -> int:
    """Accept the active two-channel branch or an intact legacy five-channel run.

    Method identities and order are checked by the versioned service contract;
    this numerical layer must never silently truncate, pad or broadcast them.
    """

    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != row_count
        or embeddings.shape[1] not in (2, 5)
    ):
        raise ValueError("normalized embedding scores must have shape [N, 2] or [N, 5]")
    return int(embeddings.shape[1])


def unified_weight_scores(
    normalized_probe_scores: Any,
    normalized_embedding_scores: Any,
    beta: Any,
    gamma: Any,
    embedding_weights: Any,
    embedding_fusion_strength: float,
    theta: Any,
    temperature: Any,
    *,
    gamma_min: float = 0.05,
    gamma_max: float = 3.0,
) -> UnifiedWeightScores:
    """Evaluate the shared conjunction-preserving ranking formula.

    ``normalized_probe_scores`` has shape ``[N, A, 8]`` and the holistic
    branch has shape ``[N, R]``: two active methods or five legacy methods.
    Embedding methods participate only in ``H`` and can therefore never appear
    in an attribute mixture.
    """

    z = np.asarray(normalized_probe_scores, dtype=np.float64)
    embeddings = np.asarray(normalized_embedding_scores, dtype=np.float64)
    beta_values = np.asarray(beta, dtype=np.float64)
    gamma_values = np.asarray(gamma, dtype=np.float64).reshape(-1)
    eta_values = np.asarray(embedding_weights, dtype=np.float64).reshape(-1)
    theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
    temperature_values = np.asarray(temperature, dtype=np.float64).reshape(-1)
    if z.ndim != 3 or z.shape[2] != 8:
        raise ValueError("normalized probe scores must have shape [N, A, 8]")
    row_count, attribute_count, member_count = z.shape
    embedding_count = _unified_embedding_count(embeddings, row_count)
    if beta_values.shape != (attribute_count, member_count):
        raise ValueError("beta shape does not match the probe-score cube")
    if gamma_values.shape != (attribute_count,):
        raise ValueError("gamma shape does not match the attribute count")
    if theta_values.shape != (attribute_count,) or temperature_values.shape != (
        attribute_count,
    ):
        raise ValueError("theta and temperature must contain one value per attribute")
    if eta_values.shape != (embedding_count,):
        raise ValueError("embedding weights must match the embedding-score columns")
    if not np.all(np.isfinite(z)) or np.any(z < -1e-8) or np.any(z > 1.0 + 1e-8):
        raise ValueError("normalized probe scores must be finite values in [0, 1]")
    if (
        not np.all(np.isfinite(embeddings))
        or np.any(embeddings < -1e-8)
        or np.any(embeddings > 1.0 + 1e-8)
    ):
        raise ValueError("normalized embedding scores must be finite values in [0, 1]")
    _validate_simplex(beta_values, axis=1, name="beta")
    _validate_simplex(eta_values, axis=0, name="embedding weights")
    if (
        not np.isfinite(gamma_min)
        or not np.isfinite(gamma_max)
        or not 0.0 < gamma_min < gamma_max
    ):
        raise ValueError("gamma bounds must be finite and satisfy 0 < min < max")
    if (
        not np.all(np.isfinite(gamma_values))
        or np.any(gamma_values < gamma_min - 1e-7)
        or np.any(gamma_values > gamma_max + 1e-7)
    ):
        raise ValueError("each gamma must lie inside the configured bounds")
    if not np.all(np.isfinite(temperature_values)) or np.any(temperature_values <= 0.0):
        raise ValueError("temperature must be finite and positive")
    lambda_value = float(embedding_fusion_strength)
    if not np.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("embedding fusion strength must lie in [0, 1]")

    attribute_scores = np.einsum("nam,am->na", z, beta_values)
    attribute_logits = (
        attribute_scores - theta_values[None, :]
    ) / temperature_values[None, :]
    gates = stable_sigmoid(attribute_logits)
    log_gates = -np.logaddexp(0.0, -attribute_logits)
    log_conjunction = np.sum(
        gamma_values[None, :] * log_gates, axis=1
    )
    conjunction = np.exp(log_conjunction)
    holistic = embeddings @ eta_values
    final = conjunction * ((1.0 - lambda_value) + lambda_value * holistic)
    return UnifiedWeightScores(
        attribute_scores=attribute_scores.astype(np.float32),
        gates=gates.astype(np.float32),
        conjunction_scores=conjunction.astype(np.float32),
        holistic_scores=holistic.astype(np.float32),
        final_scores=np.clip(final, 0.0, 1.0).astype(np.float32),
    )


def initial_unified_weight_state(
    attribute_count: int,
    theta: Any,
    temperature: Any,
    *,
    lambda0: float = 0.25,
    embedding_count: int = 2,
) -> dict[str, Any]:
    """Return the identical starting point used by both modes.

    New runs use two holistic methods; legacy replay must explicitly request
    ``embedding_count=5`` to retain the original uniform one-fifth weights.
    """

    if attribute_count < 1:
        raise ValueError("attribute_count must be positive")
    if (
        isinstance(embedding_count, (bool, np.bool_))
        or not isinstance(embedding_count, (int, np.integer))
        or embedding_count not in (2, 5)
    ):
        raise ValueError("embedding_count must be 2 or 5")
    theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
    temperature_values = np.asarray(temperature, dtype=np.float64).reshape(-1)
    if theta_values.shape != (attribute_count,) or temperature_values.shape != (
        attribute_count,
    ):
        raise ValueError("theta and temperature must match attribute_count")
    if not np.isfinite(lambda0) or not 0.0 < float(lambda0) < 1.0:
        raise ValueError("lambda0 must lie strictly between zero and one")
    return {
        "beta": np.full((attribute_count, 8), 1.0 / 8.0, dtype=np.float64),
        "gamma": np.ones(attribute_count, dtype=np.float64),
        "embedding_weights": np.full(
            embedding_count, 1.0 / embedding_count, dtype=np.float64
        ),
        "embedding_fusion_strength": float(lambda0),
        "theta": theta_values.copy(),
        "temperature": temperature_values.copy(),
    }


def _known_balanced_weights(labels: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
    """Balance two-class labels, while allowing a one-class attribute slice."""

    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if truth.shape != weights.shape or truth.size == 0:
        raise ValueError("known labels and sample weights must be non-empty and aligned")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample weights must be finite and positive")
    if np.unique(truth).size == 2:
        return class_balanced_weights(truth, weights)
    return weights / float(np.mean(weights))


def recalibrate_unified_theta(
    calibration_probe_scores: Any,
    calibration_attribute_labels: Any,
    beta: Any,
    initial_theta: Any,
    temperature: Any,
    *,
    calibration_sample_weights: Any | None = None,
) -> np.ndarray:
    """Fit only theta on a fixed Development calibration set.

    Missing attribute labels are represented by ``NaN``.  An attribute with no
    calibration labels keeps its original threshold.  ``T`` is never changed.
    """

    z = np.asarray(calibration_probe_scores, dtype=np.float64)
    labels = np.asarray(calibration_attribute_labels, dtype=np.float64)
    beta_values = np.asarray(beta, dtype=np.float64)
    theta_values = np.asarray(initial_theta, dtype=np.float64).reshape(-1).copy()
    temperatures = np.asarray(temperature, dtype=np.float64).reshape(-1)
    if z.ndim != 3 or z.shape[2] != 8 or labels.shape != z.shape[:2]:
        raise ValueError("calibration arrays must have shapes [N,A,8] and [N,A]")
    if beta_values.shape != z.shape[1:] or theta_values.shape != (z.shape[1],):
        raise ValueError("calibration parameter shapes are inconsistent")
    if temperatures.shape != theta_values.shape or np.any(temperatures <= 0.0):
        raise ValueError("calibration temperatures must be positive and aligned")
    if calibration_sample_weights is None:
        weights = np.ones(labels.shape, dtype=np.float64)
    else:
        weights = np.asarray(calibration_sample_weights, dtype=np.float64)
        if weights.ndim == 1 and weights.shape == (z.shape[0],):
            weights = np.repeat(weights[:, None], z.shape[1], axis=1)
        if weights.shape != labels.shape:
            raise ValueError("calibration sample weights are not aligned")
    q = np.einsum("nam,am->na", z, beta_values)
    for attribute_index in range(z.shape[1]):
        known = np.isfinite(labels[:, attribute_index])
        if not np.any(known):
            continue
        truth = labels[known, attribute_index].astype(np.uint8)
        if not np.all(np.isin(truth, (0, 1))):
            raise ValueError("calibration labels must be binary or NaN")
        balanced = _known_balanced_weights(truth, weights[known, attribute_index])

        def threshold_objective(value: float) -> float:
            logits = (
                q[known, attribute_index] - float(value)
            ) / temperatures[attribute_index]
            return weighted_binary_cross_entropy_with_logits(
                truth, logits, balanced
            )

        lower = float(min(-0.5, np.min(q[known, attribute_index]) - 5.0 * temperatures[attribute_index]))
        upper = float(max(1.5, np.max(q[known, attribute_index]) + 5.0 * temperatures[attribute_index]))
        calibrated = minimize(
            lambda value: threshold_objective(float(value[0])),
            np.asarray([theta_values[attribute_index]], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(lower, upper)],
            options={"maxiter": 100, "ftol": 1e-12},
        )
        if calibrated.success and np.all(np.isfinite(calibrated.x)):
            candidate = float(calibrated.x[0])
            if threshold_objective(candidate) <= threshold_objective(
                theta_values[attribute_index]
            ) + 1e-10:
                theta_values[attribute_index] = candidate
    return theta_values.astype(np.float32)


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by server environments
        raise RuntimeError(
            "Weight-only v5 optimization requires PyTorch; scoring remains NumPy-only"
        ) from error
    return torch


def fusion_training_schedule(mode: str) -> str:
    """Only fusion weights distinguish Staged from Joint in the four-way study.

    Native probe updates happen once, upstream of this module. Both system-two
    modes consume the same verified frozen probability snapshot.
    """
    schedules = {
        "weight_staged": "staged", "weight_joint": "joint",
        "probe_staged": "staged", "probe_joint": "joint",
    }
    if mode not in schedules:
        raise ValueError("unsupported four-way refinement mode")
    return schedules[mode]


def _frozen_score_array(values: Any, name: str) -> np.ndarray:
    """Own an immutable-in-practice copy; never connect fusion to probe autograd."""
    if getattr(values, "requires_grad", False):
        raise ValueError(f"{name} must be a frozen score snapshot, not a probe autograd graph")
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()
    return np.array(values, dtype=np.float64, copy=True)


def fit_unified_weight_refinement(
    mode: str,
    normalized_probe_scores: Any,
    normalized_embedding_scores: Any,
    joint_labels: Any,
    *,
    theta: Any,
    temperature: Any,
    joint_sample_weights: Any | None = None,
    attribute_labels: Any | None = None,
    attribute_sample_weights: Any | None = None,
    negative_failure_mask: Any | None = None,
    calibration_probe_scores: Any | None = None,
    calibration_attribute_labels: Any | None = None,
    calibration_sample_weights: Any | None = None,
    lambda0: float = 0.25,
    gamma_min: float = 0.05,
    gamma_max: float = 3.0,
    rho_attribute: float = 1.0,
    rho_joint: float = 1.0,
    beta_regularization: float = 0.05,
    gamma_regularization: float = 0.05,
    embedding_regularization: float = 0.05,
    lambda_regularization: float = 0.05,
    learning_rate: float = 0.05,
    max_iterations: int = 300,
    outer_iterations: int = 3,
    seed: int = 0,
    gate_calibration_policy: str = "fixed-base",
) -> WeightRefinementResult:
    """Fit fusion weights from frozen original or natively updated probes.

    ``probe_staged``/``probe_joint`` require an upstream native update and a
    verified common snapshot; they select the same two fusion schedules as
    their weight-only counterparts. This function never creates, loads, or
    optimizes probe parameters, and never adds a probe loss to fusion losses.

    ``joint_labels`` and ``attribute_labels`` use ``NaN`` for unknown entries.
    For a negative joint row, only attributes marked in
    ``negative_failure_mask`` propagate joint-loss gradients to beta/gamma;
    the holistic eta/lambda branch always receives the complete joint loss.
    Positive rows route gradients through every query attribute.

    Both modes recreate uniform beta/eta, unit gamma and ``lambda0`` internally,
    so one tuning result can never become the starting point of another mode.
    Eta has one coordinate per input embedding column (two active or five
    legacy methods); its uniform anchor is always ``1 / embedding_count``.

    The default ``fixed-base`` policy keeps the supplied theta and temperature
    unchanged throughout optimization and checkpoint selection. Calibration
    inputs cannot change them. Only explicit ``legacy-recalibrate`` replay
    performs the historical between-stage/outer-loop theta recalibration.
    """

    if gate_calibration_policy not in {"fixed-base", "legacy-recalibrate"}:
        raise ValueError("gate_calibration_policy must be fixed-base or legacy-recalibrate")
    fixed_gate = gate_calibration_policy == "fixed-base"
    schedule = fusion_training_schedule(mode)
    z = _frozen_score_array(normalized_probe_scores, "probe scores")
    embeddings = _frozen_score_array(normalized_embedding_scores, "embedding scores")
    if z.ndim != 3 or z.shape[2] != 8:
        raise ValueError("normalized probe scores must have shape [N, A, 8]")
    row_count, attribute_count, _member_count = z.shape
    embedding_count = _unified_embedding_count(embeddings, row_count)
    if not np.all(np.isfinite(z)) or np.any(z < 0.0) or np.any(z > 1.0):
        raise ValueError("normalized probe scores must lie in [0, 1]")
    if (
        not np.all(np.isfinite(embeddings))
        or np.any(embeddings < 0.0)
        or np.any(embeddings > 1.0)
    ):
        raise ValueError("normalized embedding scores must lie in [0, 1]")
    initial = initial_unified_weight_state(
        attribute_count, theta, temperature, lambda0=lambda0,
        embedding_count=embedding_count,
    )
    if (
        not np.isfinite(gamma_min)
        or not np.isfinite(gamma_max)
        or not 0.0 < gamma_min < 1.0 < gamma_max
    ):
        raise ValueError("gamma bounds must satisfy 0 < gamma_min < 1 < gamma_max")
    joint_truth = np.asarray(joint_labels, dtype=np.float64).reshape(-1)
    if joint_truth.shape != (row_count,):
        raise ValueError("joint labels must align with score rows")
    joint_known = np.isfinite(joint_truth)
    if np.any(~np.isin(joint_truth[joint_known], (0.0, 1.0))):
        raise ValueError("joint labels must be binary or NaN")
    if joint_sample_weights is None:
        joint_weights = np.ones(row_count, dtype=np.float64)
    else:
        joint_weights = np.asarray(joint_sample_weights, dtype=np.float64).reshape(-1)
        if joint_weights.shape != (row_count,):
            raise ValueError("joint sample weights must align with score rows")
    if np.any(joint_weights <= 0.0) or not np.all(np.isfinite(joint_weights)):
        raise ValueError("joint sample weights must be finite and positive")

    if attribute_labels is None:
        attribute_truth = np.full((row_count, attribute_count), np.nan, dtype=np.float64)
    else:
        attribute_truth = np.asarray(attribute_labels, dtype=np.float64)
        if attribute_truth.shape != (row_count, attribute_count):
            raise ValueError("attribute labels must have shape [N, A]")
        known_values = attribute_truth[np.isfinite(attribute_truth)]
        if np.any(~np.isin(known_values, (0.0, 1.0))):
            raise ValueError("attribute labels must be binary or NaN")
    if attribute_sample_weights is None:
        attribute_weights = np.ones((row_count, attribute_count), dtype=np.float64)
    else:
        attribute_weights = np.asarray(attribute_sample_weights, dtype=np.float64)
        if attribute_weights.ndim == 1 and attribute_weights.shape == (row_count,):
            attribute_weights = np.repeat(attribute_weights[:, None], attribute_count, axis=1)
        if attribute_weights.shape != (row_count, attribute_count):
            raise ValueError("attribute sample weights must have shape [N] or [N,A]")
    if np.any(attribute_weights <= 0.0) or not np.all(np.isfinite(attribute_weights)):
        raise ValueError("attribute sample weights must be finite and positive")

    if negative_failure_mask is None:
        failure_mask = np.zeros((row_count, attribute_count), dtype=bool)
    else:
        failure_mask = np.asarray(negative_failure_mask, dtype=bool)
        if failure_mask.shape != (row_count, attribute_count):
            raise ValueError("negative failure mask must have shape [N, A]")
    failure_mask = failure_mask.copy()
    failure_mask[(~joint_known) | (joint_truth != 0.0)] = False

    if (calibration_probe_scores is None) != (calibration_attribute_labels is None):
        raise ValueError("calibration scores and labels must be supplied together")
    calibration_z = (
        None
        if calibration_probe_scores is None
        else _frozen_score_array(calibration_probe_scores, "calibration probe scores")
    )
    calibration_truth = (
        None
        if calibration_attribute_labels is None
        else np.asarray(calibration_attribute_labels, dtype=np.float64)
    )

    torch = _require_torch()
    torch.manual_seed(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass
    dtype = torch.float64
    z_tensor = torch.as_tensor(z, dtype=dtype)
    embedding_tensor = torch.as_tensor(embeddings, dtype=dtype)
    attr_truth_tensor = torch.as_tensor(np.nan_to_num(attribute_truth), dtype=dtype)
    attr_known_tensor = torch.as_tensor(np.isfinite(attribute_truth), dtype=torch.bool)
    joint_truth_tensor = torch.as_tensor(np.nan_to_num(joint_truth), dtype=dtype)
    joint_known_tensor = torch.as_tensor(joint_known, dtype=torch.bool)
    failure_tensor = torch.as_tensor(failure_mask, dtype=torch.bool)

    balanced_joint = np.ones(row_count, dtype=np.float64)
    if np.any(joint_known):
        balanced_joint[joint_known] = _known_balanced_weights(
            joint_truth[joint_known].astype(np.uint8), joint_weights[joint_known]
        )
    balanced_attribute = np.ones((row_count, attribute_count), dtype=np.float64)
    for attribute_index in range(attribute_count):
        known = np.isfinite(attribute_truth[:, attribute_index])
        if np.any(known):
            balanced_attribute[known, attribute_index] = _known_balanced_weights(
                attribute_truth[known, attribute_index].astype(np.uint8),
                attribute_weights[known, attribute_index],
            )
    joint_weight_tensor = torch.as_tensor(balanced_joint, dtype=dtype)
    attr_weight_tensor = torch.as_tensor(balanced_attribute, dtype=dtype)
    temperature_tensor = torch.as_tensor(initial["temperature"], dtype=dtype)

    beta_logits = torch.nn.Parameter(torch.zeros((attribute_count, 8), dtype=dtype))
    initial_gamma_fraction = (1.0 - gamma_min) / (gamma_max - gamma_min)
    gamma_logits = torch.nn.Parameter(
        torch.full(
            (attribute_count,),
            float(stable_logit(initial_gamma_fraction)),
            dtype=dtype,
        )
    )
    eta_logits = torch.nn.Parameter(torch.zeros(embedding_count, dtype=dtype))
    lambda_logit = torch.nn.Parameter(
        torch.as_tensor(float(stable_logit(lambda0)), dtype=dtype)
    )
    theta_values = np.asarray(initial["theta"], dtype=np.float64).copy()
    history: list[dict[str, Any]] = []
    completed_iterations = 0

    def parameter_values():
        beta_value = torch.softmax(beta_logits, dim=1)
        eta_value = torch.softmax(eta_logits, dim=0)
        gamma_value = gamma_min + (gamma_max - gamma_min) * torch.sigmoid(
            gamma_logits
        )
        lambda_value = torch.sigmoid(lambda_logit)
        return beta_value, gamma_value, eta_value, lambda_value

    def losses(*, route_negative_joint: bool):
        beta_value, gamma_value, eta_value, lambda_value = parameter_values()
        theta_tensor = torch.as_tensor(theta_values, dtype=dtype)
        q = torch.sum(z_tensor * beta_value.unsqueeze(0), dim=2)
        attribute_logits = (
            q - theta_tensor.unsqueeze(0)
        ) / temperature_tensor.unsqueeze(0)
        log_gates = torch.nn.functional.logsigmoid(attribute_logits)
        if route_negative_joint:
            positive_route = (
                joint_known_tensor & (joint_truth_tensor == 1.0)
            ).unsqueeze(1)
            routed_attributes = positive_route | failure_tensor
            routed_log_gates = log_gates.detach() + routed_attributes * (
                log_gates - log_gates.detach()
            )
            # Gamma coordinates are independent.  On a negative row, only the
            # explicitly confirmed failed attributes retain gradient to their
            # own unconstrained parameter v; there is no cross-coordinate
            # simplex denominator through which unconfirmed gammas can move.
            negative_gamma = gamma_value.detach().unsqueeze(0) + failure_tensor * (
                gamma_value.unsqueeze(0) - gamma_value.detach().unsqueeze(0)
            )
            routed_gamma = torch.where(
                positive_route, gamma_value.unsqueeze(0), negative_gamma
            )
            log_conjunction = torch.sum(
                routed_gamma * routed_log_gates, dim=1
            )
        else:
            log_conjunction = torch.sum(
                gamma_value.unsqueeze(0) * log_gates,
                dim=1,
            )
        holistic = embedding_tensor @ eta_value
        log_holistic_factor = torch.log1p(
            -lambda_value * (1.0 - holistic)
        )
        log_final = log_conjunction + log_holistic_factor
        if torch.any(attr_known_tensor):
            attr_targets = attr_truth_tensor[attr_known_tensor]
            attr_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                attribute_logits[attr_known_tensor],
                attr_targets,
                reduction="none",
            )
            attr_loss = torch.sum(
                attr_losses * attr_weight_tensor[attr_known_tensor]
            ) / torch.sum(attr_weight_tensor[attr_known_tensor])
        else:
            attr_loss = torch.zeros((), dtype=dtype)
        if torch.any(joint_known_tensor):
            known_log_final = log_final[joint_known_tensor]
            joint_targets = joint_truth_tensor[joint_known_tensor]
            known_weights = joint_weight_tensor[joint_known_tensor]
            positive = joint_targets == 1.0
            weighted_loss_sum = torch.zeros((), dtype=dtype)
            if torch.any(positive):
                weighted_loss_sum = weighted_loss_sum + torch.sum(
                    -known_log_final[positive] * known_weights[positive]
                )
            negative = ~positive
            if torch.any(negative):
                negative_log_final = known_log_final[negative]
                log_one_minus_final = torch.empty_like(negative_log_final)
                far_from_zero = negative_log_final < -np.log(2.0)
                log_one_minus_final[far_from_zero] = torch.log1p(
                    -torch.exp(negative_log_final[far_from_zero])
                )
                log_one_minus_final[~far_from_zero] = torch.log(
                    -torch.expm1(negative_log_final[~far_from_zero])
                )
                weighted_loss_sum = weighted_loss_sum + torch.sum(
                    -log_one_minus_final * known_weights[negative]
                )
            joint_loss = weighted_loss_sum / torch.sum(known_weights)
        else:
            joint_loss = torch.zeros((), dtype=dtype)
        beta_anchor = torch.mean(torch.sum((beta_value - 1.0 / 8.0) ** 2, dim=1))
        gamma_anchor = torch.sum((gamma_value - 1.0) ** 2)
        eta_anchor = torch.sum((eta_value - 1.0 / embedding_count) ** 2)
        lambda_anchor = (lambda_value - float(lambda0)) ** 2
        regularizer = (
            beta_regularization * beta_anchor
            + gamma_regularization * gamma_anchor
            + embedding_regularization * eta_anchor
            + lambda_regularization * lambda_anchor
        )
        total = rho_attribute * attr_loss + rho_joint * joint_loss + regularizer
        return total, attr_loss, joint_loss

    def numpy_state() -> dict[str, Any]:
        with torch.no_grad():
            beta_value, gamma_value, eta_value, lambda_value = parameter_values()
        return {
            "beta": beta_value.detach().cpu().numpy().copy(),
            "gamma": gamma_value.detach().cpu().numpy().copy(),
            "embedding_weights": eta_value.detach().cpu().numpy().copy(),
            "embedding_fusion_strength": float(lambda_value.detach().cpu()),
            "theta": theta_values.copy(),
        }

    def load_numpy_state(state: dict[str, Any]) -> None:
        beta_array = np.clip(np.asarray(state["beta"], dtype=np.float64), EPSILON, 1.0)
        gamma_array = np.asarray(state["gamma"], dtype=np.float64)
        eta_array = np.clip(
            np.asarray(state["embedding_weights"], dtype=np.float64), EPSILON, 1.0
        )
        gamma_fraction = np.clip(
            (gamma_array - gamma_min) / (gamma_max - gamma_min),
            EPSILON,
            1.0 - EPSILON,
        )
        with torch.no_grad():
            beta_logits.copy_(torch.as_tensor(np.log(beta_array), dtype=dtype))
            gamma_logits.copy_(
                torch.as_tensor(stable_logit(gamma_fraction), dtype=dtype)
            )
            eta_logits.copy_(torch.as_tensor(np.log(eta_array), dtype=dtype))
            lambda_logit.copy_(
                torch.as_tensor(
                    float(stable_logit(state["embedding_fusion_strength"])), dtype=dtype
                )
            )
        nonlocal theta_values
        theta_values = np.asarray(state["theta"], dtype=np.float64).copy()

    def evaluate_objective() -> float:
        with torch.no_grad():
            total, _attr_loss, _joint_loss = losses(route_negative_joint=False)
        return float(total.detach().cpu())

    initial_scores = unified_weight_scores(
        z,
        embeddings,
        initial["beta"],
        initial["gamma"],
        initial["embedding_weights"],
        initial["embedding_fusion_strength"],
        initial["theta"],
        initial["temperature"],
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    ).final_scores
    initial_objective = evaluate_objective()
    best_objective = initial_objective
    best_state = numpy_state()

    def record(stage: str, iteration: int) -> None:
        with torch.no_grad():
            total, attr_loss, joint_loss = losses(route_negative_joint=False)
        history.append(
            {
                "stage": stage,
                "iteration": int(iteration),
                "objective": float(total.detach().cpu()),
                "attribute_bce": float(attr_loss.detach().cpu()),
                "joint_bce": float(joint_loss.detach().cpu()),
            }
        )

    def maybe_keep_best() -> None:
        nonlocal best_objective, best_state
        objective_value = evaluate_objective()
        if objective_value < best_objective:
            best_objective = objective_value
            best_state = numpy_state()

    def optimize(parameters: list[Any], steps: int, stage: str, *, routed: bool) -> None:
        nonlocal completed_iterations
        if steps <= 0:
            return
        # Freeze the other fusion groups too. In particular, query-stage BCE
        # cannot update beta, and no optimizer here can ever own native phi.
        parameter_ids = {id(parameter) for parameter in parameters}
        for parameter in (beta_logits, gamma_logits, eta_logits, lambda_logit):
            parameter.requires_grad_(id(parameter) in parameter_ids)
            parameter.grad = None
        optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
        for _step in range(steps):
            optimizer.zero_grad()
            total, attr_loss, joint_loss = losses(route_negative_joint=routed)
            stage_loss = total
            if stage == "beta":
                beta_value, _gamma_value, _eta_value, _lambda_value = parameter_values()
                beta_anchor = torch.mean(
                    torch.sum((beta_value - 1.0 / 8.0) ** 2, dim=1)
                )
                stage_loss = rho_attribute * attr_loss + beta_regularization * beta_anchor
            elif stage == "query":
                _beta_value, gamma_value, eta_value, lambda_value = parameter_values()
                stage_loss = (
                    rho_joint * joint_loss
                    + gamma_regularization * torch.sum((gamma_value - 1.0) ** 2)
                    + embedding_regularization
                    * torch.sum((eta_value - 1.0 / embedding_count) ** 2)
                    + lambda_regularization * (lambda_value - float(lambda0)) ** 2
                )
            stage_loss.backward()
            optimizer.step()
            completed_iterations += 1
        record(stage, completed_iterations)

    def recalibrate(stage: str) -> None:
        nonlocal theta_values
        if fixed_gate or calibration_z is None or calibration_truth is None:
            return
        state = numpy_state()
        theta_values = recalibrate_unified_theta(
            calibration_z,
            calibration_truth,
            state["beta"],
            initial["theta"],
            initial["temperature"],
            calibration_sample_weights=calibration_sample_weights,
        ).astype(np.float64)
        record(stage, completed_iterations)

    record("initial", 0)
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations:
        raise ValueError("max_iterations must be an integer")
    total_steps = int(max_iterations)
    if total_steps < 2:
        raise ValueError("max_iterations must be at least two")
    if schedule == "staged":
        beta_steps = total_steps // 2
        query_steps = total_steps - beta_steps
        optimize([beta_logits], beta_steps, "beta", routed=False)
        if not fixed_gate:
            recalibrate("theta-calibration")
        # Fixed-gate stages compare the same objective under the same theta/T.
        # Legacy replay may only retain beta after its historical calibration.
        maybe_keep_best()
        optimize(
            [gamma_logits, eta_logits, lambda_logit],
            query_steps,
            "query",
            routed=True,
        )
        # The query stage never changes beta or the gate calibration, so its
        # checkpoint is comparable with the preceding candidate.
        maybe_keep_best()
    else:
        if isinstance(outer_iterations, bool) or int(outer_iterations) != outer_iterations:
            raise ValueError("outer_iterations must be an integer")
        outer_count = min(max(1, int(outer_iterations)), total_steps)
        base_steps, extra_steps = divmod(total_steps, outer_count)
        for outer_index in range(outer_count):
            steps_this_outer = base_steps + int(outer_index < extra_steps)
            optimize(
                [beta_logits, gamma_logits, eta_logits, lambda_logit],
                steps_this_outer,
                f"joint-{outer_index + 1}",
                routed=True,
            )
            if not fixed_gate:
                recalibrate(f"theta-calibration-{outer_index + 1}")
            # Fixed-gate outer segments retain the existing optimization-step
            # budget/schedule without introducing a separate calibration loss.
            # Legacy checkpoints still follow their historical recalibration.
            maybe_keep_best()

    load_numpy_state(best_state)
    final_state = numpy_state()
    final_scores = unified_weight_scores(
        z,
        embeddings,
        final_state["beta"],
        final_state["gamma"],
        final_state["embedding_weights"],
        final_state["embedding_fusion_strength"],
        final_state["theta"],
        initial["temperature"],
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    ).final_scores
    ranking_indices = np.argsort(-final_scores, kind="stable").astype(np.int64)
    final_objective = evaluate_objective()
    if final_objective > initial_objective + 1e-8:
        raise RuntimeError("Weight refinement returned a worse objective")
    return WeightRefinementResult(
        mode=mode,
        initial_scores=initial_scores,
        refined_scores=final_scores,
        ranking_indices=ranking_indices,
        beta=np.asarray(final_state["beta"], dtype=np.float32),
        gamma=np.asarray(final_state["gamma"], dtype=np.float32),
        embedding_weights=np.asarray(
            final_state["embedding_weights"], dtype=np.float32
        ),
        embedding_fusion_strength=float(
            final_state["embedding_fusion_strength"]
        ),
        theta=(
            np.asarray(theta).reshape(-1).copy()
            if fixed_gate
            else np.asarray(final_state["theta"], dtype=np.float32)
        ),
        temperature=(
            np.asarray(temperature).reshape(-1).copy()
            if fixed_gate
            else np.asarray(initial["temperature"], dtype=np.float32)
        ),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        iterations=int(completed_iterations),
        training_history=tuple(history),
    )


def fit_weight_staged(*args: Any, **kwargs: Any) -> WeightRefinementResult:
    """Convenience wrapper for ``Weight-only · Staged``."""

    return fit_unified_weight_refinement("weight_staged", *args, **kwargs)


def fit_weight_joint(*args: Any, **kwargs: Any) -> WeightRefinementResult:
    """Convenience wrapper for ``Weight-only · Joint``."""

    return fit_unified_weight_refinement("weight_joint", *args, **kwargs)
