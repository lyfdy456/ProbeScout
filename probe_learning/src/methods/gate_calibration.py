"""Fit-only attribute gates and constrained joint calibration.

Validation selection for the paper is a separate step.
"""

from __future__ import annotations

import math

from typing import Any, Iterable

import numpy as np

from scipy.optimize import minimize

TEMPERATURES = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)

def _inclusive_grid(lower: float, upper: float, step: float) -> np.ndarray:
    if step <= 0.0 or upper < lower:
        raise ValueError("invalid grid bounds or step")
    count = int(math.floor((upper - lower) / step + 1e-9))
    values = lower + step * np.arange(count + 1, dtype=np.float64)
    if values.size == 0 or values[-1] < upper - 1e-9:
        values = np.r_[values, upper]
    return np.unique(np.round(np.clip(values, lower, upper), 12))

def _attribute_bce(
    scores: np.ndarray,
    targets: np.ndarray,
    theta: float,
    temperature: float,
) -> float:
    """Numerically stable BCE-with-logits without probability epsilons."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = (np.asarray(scores, dtype=np.float64) - float(theta)) / float(temperature)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))

def _best_grid_point(
    scores: np.ndarray,
    targets: np.ndarray,
    theta_values: Iterable[float],
    temperatures: Iterable[float],
) -> dict[str, float]:
    candidates = []
    for temperature in temperatures:
        for theta in theta_values:
            candidates.append(
                {
                    "theta": float(theta),
                    "temperature": float(temperature),
                    "bce": _attribute_bce(scores, targets, theta, temperature),
                }
            )
    if not candidates:
        raise ValueError("grid search has no candidates")
    # Exact ties prefer the smoother gate, then the more central threshold.
    return min(
        candidates,
        key=lambda row: (
            row["bce"],
            -row["temperature"],
            abs(row["theta"] - 0.5),
            row["theta"],
        ),
    )

def search_attribute_gate(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    coarse_step: float = 0.025,
    fine_radius: float = 0.05,
    fine_step: float = 0.005,
    temperatures: Iterable[float] = TEMPERATURES,
    single_class_policy: str = "error",
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if scores.ndim != 1 or targets.ndim != 1 or len(scores) != len(targets):
        raise ValueError("scores and targets must be aligned one-dimensional arrays")
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("attribute grid search requires finite non-empty scores")
    if float(scores.min()) < -1e-6 or float(scores.max()) > 1.0 + 1e-6:
        raise ValueError("All-Mean input scores must stay in [0, 1]")
    if single_class_policy not in {"error", "default_gate"}:
        raise ValueError(
            f"unknown attribute single-class policy: {single_class_policy!r}"
        )
    single_class = set(np.unique(targets).tolist()) != {0.0, 1.0}
    if single_class and single_class_policy == "error":
        raise ValueError("attribute grid search requires both positive and negative labels")

    temperatures = tuple(float(value) for value in temperatures)
    if any(value <= 0.0 for value in temperatures):
        raise ValueError("all temperatures must be positive")
    if single_class:
        # The established R-SoftGate fallback is a neutral fixed gate.  A
        # one-class BCE grid would otherwise drive theta/temperature to a
        # saturated boundary that reflects acquisition bias, not separation.
        theta = 0.5
        temperature = 0.10
        bce = _attribute_bce(scores, targets, theta, temperature)
        return {
            "status": "single_class_default_gate",
            "single_class_fallback": True,
            "single_class_policy": single_class_policy,
            "parameter_source": "fixed_legacy_default_train_only",
            "theta": theta,
            "temperature": temperature,
            "bce": bce,
            "coarse_theta": theta,
            "coarse_temperature": temperature,
            "coarse_bce": bce,
            "fine_theta_min": theta,
            "fine_theta_max": theta,
            "fine_candidate_count": 0,
            "coarse_candidate_count": 0,
            "fine_improvement": 0.0,
            "theta_on_global_boundary": False,
            "theta_on_fine_boundary": False,
        }
    coarse_theta = _inclusive_grid(0.0, 1.0, coarse_step)
    coarse = _best_grid_point(scores, targets, coarse_theta, temperatures)
    fine_min = max(0.0, coarse["theta"] - fine_radius)
    fine_max = min(1.0, coarse["theta"] + fine_radius)
    fine_theta = _inclusive_grid(fine_min, fine_max, fine_step)
    if not np.any(np.isclose(fine_theta, coarse["theta"], atol=1e-12)):
        fine_theta = np.unique(np.r_[fine_theta, coarse["theta"]])
    fine = _best_grid_point(scores, targets, fine_theta, temperatures)
    return {
        "status": "fitted",
        "single_class_fallback": False,
        "single_class_policy": single_class_policy,
        "parameter_source": "train_attribute_bce_grid",
        "theta": fine["theta"],
        "temperature": fine["temperature"],
        "bce": fine["bce"],
        "coarse_theta": coarse["theta"],
        "coarse_temperature": coarse["temperature"],
        "coarse_bce": coarse["bce"],
        "fine_theta_min": float(fine_theta.min()),
        "fine_theta_max": float(fine_theta.max()),
        "fine_candidate_count": int(len(fine_theta) * len(temperatures)),
        "coarse_candidate_count": int(len(coarse_theta) * len(temperatures)),
        "fine_improvement": float(coarse["bce"] - fine["bce"]),
        "theta_on_global_boundary": bool(fine["theta"] in (0.0, 1.0)),
        "theta_on_fine_boundary": bool(
            np.isclose(fine["theta"], fine_theta.min())
            or np.isclose(fine["theta"], fine_theta.max())
        ),
    }

def _log1mexp(log_probability: np.ndarray) -> np.ndarray:
    """Stable log(1-exp(x)) for x < 0, without adding an epsilon."""
    values = np.asarray(log_probability, dtype=np.float64)
    if np.any(values >= 0.0):
        raise ValueError("log probabilities must be strictly negative")
    output = np.empty_like(values)
    cutoff = -math.log(2.0)
    mask = values < cutoff
    output[mask] = np.log1p(-np.exp(values[mask]))
    output[~mask] = np.log(-np.expm1(values[~mask]))
    return output

def joint_bce_from_gate_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """BCE of product(sigmoid(logits_a)), evaluated in log space."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("Joint BCE logits and targets are not aligned")
    if set(np.unique(targets).tolist()) - {0.0, 1.0}:
        raise ValueError("Joint BCE targets must be binary")
    log_gates = -np.logaddexp(0.0, -logits)
    log_joint = log_gates.sum(axis=1)
    log_not_joint = _log1mexp(log_joint)
    losses = np.where(targets == 1.0, -log_joint, -log_not_joint)
    return float(losses.mean())

def fit_joint_bce_from_grid(
    attribute_inputs: np.ndarray,
    targets: np.ndarray,
    attrs: list[str],
    theta_init: dict[str, float],
    temperature_init: dict[str, float],
    *,
    train_temperature: bool,
    theta_radius: float = 0.05,
    temperature_bounds: tuple[float, float] = (0.03, 0.10),
    max_iter: int = 500,
) -> dict[str, Any]:
    attribute_inputs = np.asarray(attribute_inputs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if attribute_inputs.shape != (len(targets), len(attrs)):
        raise ValueError("attribute-input shape does not match labels and attributes")
    if set(np.unique(targets).tolist()) != {0.0, 1.0}:
        raise ValueError("Joint-BCE fitting requires positive and negative Joint labels")
    initial_theta = np.asarray([theta_init[attr] for attr in attrs], dtype=np.float64)
    initial_temperature = np.asarray(
        [temperature_init[attr] for attr in attrs], dtype=np.float64
    )
    if np.any(initial_temperature <= 0.0):
        raise ValueError("grid temperatures must be positive")
    theta_bounds = [
        (max(0.0, value - theta_radius), min(1.0, value + theta_radius))
        for value in initial_theta
    ]
    temp_low, temp_high = map(float, temperature_bounds)
    if not (0.0 < temp_low <= temp_high):
        raise ValueError("invalid temperature bounds")
    if train_temperature and (
        np.any(initial_temperature < temp_low - 1e-12)
        or np.any(initial_temperature > temp_high + 1e-12)
    ):
        raise ValueError("grid temperature initialization lies outside trainable bounds")

    def loss(theta: np.ndarray, temperature: np.ndarray) -> float:
        logits = (attribute_inputs - theta) / temperature
        return joint_bce_from_gate_logits(logits, targets)

    initial_loss = loss(initial_theta, initial_temperature)
    if train_temperature:
        initial_parameters = np.r_[initial_theta, initial_temperature]
        bounds = theta_bounds + [(temp_low, temp_high)] * len(attrs)

        def objective(parameters: np.ndarray) -> float:
            theta, temperature = np.split(np.asarray(parameters, dtype=np.float64), 2)
            return loss(theta, temperature)

    else:
        initial_parameters = initial_theta
        bounds = theta_bounds

        def objective(parameters: np.ndarray) -> float:
            return loss(np.asarray(parameters, dtype=np.float64), initial_temperature)

    solution = minimize(
        objective,
        initial_parameters,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8},
    )
    if not np.isfinite(solution.fun):
        raise RuntimeError("GridInit Joint-BCE optimization returned a non-finite loss")
    if train_temperature:
        final_theta, final_temperature = np.split(
            np.asarray(solution.x, dtype=np.float64), 2
        )
    else:
        final_theta = np.asarray(solution.x, dtype=np.float64)
        final_temperature = initial_temperature.copy()
    final_loss = loss(final_theta, final_temperature)
    tolerance = 1e-7
    return {
        "theta_by_attr": {attr: float(final_theta[i]) for i, attr in enumerate(attrs)},
        "temperature_by_attr": {
            attr: float(final_temperature[i]) for i, attr in enumerate(attrs)
        },
        "theta_init_by_attr": {attr: float(initial_theta[i]) for i, attr in enumerate(attrs)},
        "temperature_init_by_attr": {
            attr: float(initial_temperature[i]) for i, attr in enumerate(attrs)
        },
        "theta_bounds_by_attr": {
            attr: [float(theta_bounds[i][0]), float(theta_bounds[i][1])]
            for i, attr in enumerate(attrs)
        },
        "temperature_bounds": (
            [temp_low, temp_high] if train_temperature else None
        ),
        "train_temperature": train_temperature,
        "initial_joint_bce": float(initial_loss),
        "final_joint_bce": float(final_loss),
        "joint_bce_improvement": float(initial_loss - final_loss),
        "optimizer_success": bool(solution.success),
        "optimizer_message": str(solution.message),
        "optimizer_iterations": int(solution.nit),
        "theta_boundary_hits": int(sum(
            abs(final_theta[i] - theta_bounds[i][0]) <= tolerance
            or abs(final_theta[i] - theta_bounds[i][1]) <= tolerance
            for i in range(len(attrs))
        )),
        "temperature_boundary_hits": int(sum(
            abs(value - temp_low) <= tolerance or abs(value - temp_high) <= tolerance
            for value in final_temperature
        )) if train_temperature else 0,
    }
