from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


WEB_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pcp_fixed_gate_models", WEB_ROOT / "scripts" / "tuning_models.py"
)
assert SPEC is not None and SPEC.loader is not None
MODELS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODELS
SPEC.loader.exec_module(MODELS)


class FixedGateRefinementTests(unittest.TestCase):
    @staticmethod
    def fixture(dtype=np.float64):
        attrs = np.asarray([[1, 1], [1, 0], [0, 1], [0, 0]] * 4, dtype=np.float64)
        joint = np.all(attrs == 1, axis=1).astype(np.float64)
        probes = np.full((len(joint), 2, 8), 0.5, dtype=np.float64)
        probes[:, 0, 0] = np.where(attrs[:, 0] == 1, 0.96, 0.04)
        probes[:, 0, 1] = np.where(attrs[:, 0] == 1, 0.10, 0.90)
        probes[:, 1, 1] = np.where(attrs[:, 1] == 1, 0.96, 0.04)
        probes[:, 1, 0] = np.where(attrs[:, 1] == 1, 0.10, 0.90)
        embeddings = np.column_stack((
            np.where(joint == 1, 0.95, 0.05),
            np.where(joint == 1, 0.15, 0.85),
        ))
        theta = np.asarray([0.4876543210123, 0.5123456789012], dtype=dtype)
        temperature = np.asarray([0.1432109876543, 0.1567890123456], dtype=dtype)
        return probes, embeddings, joint, dict(
            theta=theta,
            temperature=temperature,
            attribute_labels=attrs,
            negative_failure_mask=attrs == 0,
            calibration_probe_scores=probes,
            calibration_attribute_labels=attrs,
            beta_regularization=0.001,
            gamma_regularization=0.001,
            embedding_regularization=0.001,
            lambda_regularization=0.001,
            max_iterations=31,
            outer_iterations=3,
            seed=29,
        )

    def assert_same_bytes(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_all_four_modes_update_weights_with_bitwise_fixed_gates(self):
        for dtype in (np.float32, np.float64):
            for mode in ("weight_staged", "weight_joint", "probe_staged", "probe_joint"):
                with self.subTest(dtype=dtype, mode=mode):
                    probes, embeddings, joint, kwargs = self.fixture(dtype)
                    before = {
                        key: value.copy() for key, value in kwargs.items()
                        if isinstance(value, np.ndarray)
                    }
                    with mock.patch.object(
                        MODELS, "recalibrate_unified_theta",
                        side_effect=AssertionError("fixed gates must never be recalibrated"),
                    ) as calibrate:
                        result = MODELS.fit_unified_weight_refinement(
                            mode, probes, embeddings, joint, **kwargs
                        )
                    calibrate.assert_not_called()
                    for key in ("theta", "temperature"):
                        self.assert_same_bytes(getattr(result, key), before[key])
                        self.assertFalse(np.shares_memory(getattr(result, key), kwargs[key]))
                    for key, original in before.items():
                        self.assert_same_bytes(kwargs[key], original)
                    self.assertGreater(float(result.beta[0, 0]), 1.0 / 8.0)
                    self.assertGreater(float(result.beta[1, 1]), 1.0 / 8.0)
                    self.assertFalse(np.allclose(result.gamma, 1.0))
                    self.assertGreater(float(result.embedding_weights[0]), 0.5)
                    self.assertNotAlmostEqual(result.embedding_fusion_strength, 0.25)
                    self.assertEqual(result.iterations, 31)
                    self.assertLess(result.final_objective, result.initial_objective)
                    self.assertFalse(any(
                        "calibration" in item["stage"] for item in result.training_history
                    ))
                    self.assertAlmostEqual(
                        result.final_objective,
                        min(item["objective"] for item in result.training_history),
                        places=10,
                    )
                    deployed = MODELS.unified_weight_scores(
                        probes, embeddings, result.beta, result.gamma,
                        result.embedding_weights, result.embedding_fusion_strength,
                        before["theta"], before["temperature"],
                    )
                    np.testing.assert_allclose(
                        result.refined_scores, deployed.final_scores, rtol=1e-6, atol=1e-7
                    )

    def test_calibration_labels_cannot_influence_fixed_weight_training(self):
        for mode in ("weight_staged", "weight_joint"):
            with self.subTest(mode=mode):
                probes, embeddings, joint, kwargs = self.fixture()
                first = MODELS.fit_unified_weight_refinement(
                    mode, probes, embeddings, joint, **kwargs
                )
                second = MODELS.fit_unified_weight_refinement(
                    mode, probes, embeddings, joint,
                    **{**kwargs,
                       "calibration_attribute_labels": 1.0 - kwargs["attribute_labels"],
                       "gate_calibration_policy": "fixed-base"},
                )
                for key in ("beta", "gamma", "embedding_weights", "theta", "temperature",
                            "initial_scores", "refined_scores"):
                    self.assert_same_bytes(getattr(first, key), getattr(second, key))
                self.assertEqual(first.embedding_fusion_strength, second.embedding_fusion_strength)
                self.assertEqual(first.training_history, second.training_history)

    def test_explicit_legacy_policy_preserves_recalibration_stages(self):
        for mode, expected_calls in (("weight_staged", 1), ("weight_joint", 3)):
            with self.subTest(mode=mode):
                probes, embeddings, joint, kwargs = self.fixture()
                kwargs["theta"] = np.asarray([-0.25, 1.25], dtype=np.float64)
                with mock.patch.object(
                    MODELS, "recalibrate_unified_theta", wraps=MODELS.recalibrate_unified_theta,
                ) as calibrate:
                    result = MODELS.fit_unified_weight_refinement(
                        mode, probes, embeddings, joint,
                        gate_calibration_policy="legacy-recalibrate", **kwargs,
                    )
                self.assertEqual(calibrate.call_count, expected_calls)
                self.assertEqual(sum(
                    "calibration" in item["stage"] for item in result.training_history
                ), expected_calls)
                self.assertFalse(np.array_equal(result.theta, kwargs["theta"]))
                self.assertEqual(result.theta.dtype, np.dtype(np.float32))
                self.assertEqual(result.temperature.dtype, np.dtype(np.float32))

    def test_unknown_policy_fails_closed(self):
        probes, embeddings, joint, kwargs = self.fixture()
        with self.assertRaisesRegex(ValueError, "gate_calibration_policy"):
            MODELS.fit_unified_weight_refinement(
                "weight_joint", probes, embeddings, joint,
                gate_calibration_policy="auto", **kwargs,
            )


if __name__ == "__main__":
    unittest.main()
