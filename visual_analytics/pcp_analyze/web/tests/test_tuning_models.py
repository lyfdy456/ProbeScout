from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


WEB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WEB_ROOT / "scripts" / "tuning_models.py"
SPEC = importlib.util.spec_from_file_location("pcp_tuning_models", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODELS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODELS
SPEC.loader.exec_module(MODELS)


class TuningModelTests(unittest.TestCase):
    @staticmethod
    def _unified_fixture():
        attribute_truth = np.asarray(
            [[1, 1], [1, 0], [0, 1], [0, 0]] * 4, dtype=np.float64
        )
        joint_truth = np.all(attribute_truth == 1, axis=1).astype(np.float64)
        z = np.full((len(joint_truth), 2, 8), 0.5, dtype=np.float64)
        z[:, 0, 0] = np.where(attribute_truth[:, 0] == 1, 0.96, 0.04)
        z[:, 0, 1] = np.where(attribute_truth[:, 0] == 1, 0.10, 0.90)
        z[:, 1, 1] = np.where(attribute_truth[:, 1] == 1, 0.96, 0.04)
        z[:, 1, 0] = np.where(attribute_truth[:, 1] == 1, 0.10, 0.90)
        # Only the first holistic method is informative.  The others are
        # deterministic distractors, which makes eta identifiable.
        embeddings = np.full((len(joint_truth), 5), 0.5, dtype=np.float64)
        embeddings[:, 0] = np.where(joint_truth == 1, 0.95, 0.05)
        embeddings[:, 1] = np.where(joint_truth == 1, 0.15, 0.85)
        failure_mask = np.zeros((len(joint_truth), 2), dtype=bool)
        negative = joint_truth == 0
        failure_mask[negative] = attribute_truth[negative] == 0
        return z, embeddings, joint_truth, attribute_truth, failure_mask

    def test_class_balance_preserves_feedback_emphasis(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        weights = MODELS.class_balanced_weights(labels, [1.0, 8.0, 1.0, 16.0])
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2:].sum()))
        self.assertAlmostEqual(float(weights[1] / weights[0]), 8.0)
        self.assertAlmostEqual(float(weights[3] / weights[2]), 16.0)


    def test_unified_formula_constraints_and_score_range(self):
        z, embeddings, _joint, _attrs, _failures = self._unified_fixture()
        state = MODELS.initial_unified_weight_state(
            2, theta=[0.5, 0.5], temperature=[0.15, 0.15], lambda0=0.25,
            embedding_count=5,
        )
        scores = MODELS.unified_weight_scores(
            z,
            embeddings,
            state["beta"],
            state["gamma"],
            state["embedding_weights"],
            state["embedding_fusion_strength"],
            state["theta"],
            state["temperature"],
        )
        self.assertEqual(scores.attribute_scores.shape, (len(z), 2))
        self.assertEqual(scores.final_scores.shape, (len(z),))
        self.assertTrue(np.all(scores.final_scores >= 0.0))
        self.assertTrue(np.all(scores.final_scores <= 1.0))
        np.testing.assert_allclose(
            scores.conjunction_scores,
            np.prod(np.power(scores.gates, state["gamma"][None, :]), axis=1),
            atol=1e-6,
        )


    def test_unified_branch_rejects_unsupported_and_mismatched_dimensions(self):
        z, embeddings, joint, _attrs, _failures = self._unified_fixture()
        state = MODELS.initial_unified_weight_state(2, [0.5, 0.5], [0.15, 0.15])
        invalid_arrays = (
            embeddings[:, :1], embeddings[:, :3], embeddings[:, :4],
            np.zeros((len(z), 6)), embeddings[:, 0], embeddings[1:, :2],
            np.zeros((len(z), 1, 2)),
        )
        for invalid in invalid_arrays:
            with self.subTest(shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "embedding scores"):
                    MODELS.unified_weight_scores(z, invalid, **state)
                with self.assertRaisesRegex(ValueError, "embedding scores"):
                    MODELS.fit_weight_joint(
                        z, invalid, joint, theta=[0.5, 0.5], temperature=[0.15, 0.15]
                    )
        for count, wrong_count in ((2, 5), (5, 2)):
            with self.subTest(count=count, wrong_count=wrong_count):
                wrong_state = {
                    **state, "embedding_weights": np.full(wrong_count, 1.0 / wrong_count)
                }
                with self.assertRaisesRegex(ValueError, "embedding weights"):
                    MODELS.unified_weight_scores(z, embeddings[:, :count], **wrong_state)


    def test_weight_staged_respects_simplexes_and_does_not_worsen(self):
        z, embeddings, joint, attrs, failures = self._unified_fixture()
        result = MODELS.fit_weight_staged(
            z,
            embeddings,
            joint,
            theta=[0.5, 0.5],
            temperature=[0.15, 0.15],
            attribute_labels=attrs,
            negative_failure_mask=failures,
            calibration_probe_scores=z,
            calibration_attribute_labels=attrs,
            beta_regularization=0.001,
            gamma_regularization=0.001,
            embedding_regularization=0.001,
            lambda_regularization=0.001,
            max_iterations=80,
            seed=7,
        )
        self.assertEqual(result.mode, "weight_staged")
        np.testing.assert_allclose(result.beta.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(result.embedding_weights.sum(), 1.0, atol=1e-6)
        self.assertTrue(np.all(result.gamma >= 0.05 - 1e-6))
        self.assertTrue(np.all(result.gamma <= 3.0 + 1e-6))
        self.assertGreaterEqual(result.embedding_fusion_strength, 0.0)
        self.assertLessEqual(result.embedding_fusion_strength, 1.0)
        self.assertLessEqual(result.final_objective, result.initial_objective + 1e-8)
        self.assertGreater(float(result.beta[0, 0]), 1.0 / 8.0)
        self.assertGreater(float(result.beta[1, 1]), 1.0 / 8.0)

    def test_weight_joint_is_deterministic_and_uses_same_initial_snapshot(self):
        z, embeddings, joint, attrs, failures = self._unified_fixture()
        kwargs = dict(
            theta=[0.5, 0.5],
            temperature=[0.15, 0.15],
            attribute_labels=attrs,
            negative_failure_mask=failures,
            calibration_probe_scores=z,
            calibration_attribute_labels=attrs,
            beta_regularization=0.001,
            gamma_regularization=0.001,
            embedding_regularization=0.001,
            lambda_regularization=0.001,
            max_iterations=90,
            outer_iterations=3,
            seed=11,
        )
        first = MODELS.fit_weight_joint(z, embeddings, joint, **kwargs)
        second = MODELS.fit_weight_joint(z, embeddings, joint, **kwargs)
        staged = MODELS.fit_weight_staged(
            z,
            embeddings,
            joint,
            **{**kwargs, "max_iterations": 20},
        )
        np.testing.assert_allclose(first.initial_scores, staged.initial_scores, atol=0.0)
        np.testing.assert_allclose(first.beta, second.beta, atol=1e-8)
        np.testing.assert_allclose(first.gamma, second.gamma, atol=1e-8)
        np.testing.assert_allclose(
            first.embedding_weights, second.embedding_weights, atol=1e-8
        )
        self.assertAlmostEqual(
            first.embedding_fusion_strength,
            second.embedding_fusion_strength,
            places=10,
        )
        self.assertLessEqual(first.final_objective, first.initial_objective + 1e-8)
        self.assertTrue(np.all(first.refined_scores >= 0.0))
        self.assertTrue(np.all(first.refined_scores <= 1.0))

    def test_joint_negative_routes_beta_only_to_confirmed_failure_attribute(self):
        rng = np.random.default_rng(21)
        z = rng.uniform(0.05, 0.95, size=(12, 2, 8))
        embeddings = rng.uniform(0.05, 0.95, size=(12, 5))
        failure_mask = np.zeros((12, 2), dtype=bool)
        failure_mask[:, 0] = True
        result = MODELS.fit_weight_joint(
            z,
            embeddings,
            np.zeros(12, dtype=np.float64),
            theta=[0.5, 0.5],
            temperature=[0.15, 0.15],
            negative_failure_mask=failure_mask,
            max_iterations=25,
            outer_iterations=1,
            seed=4,
        )
        # There are no attribute labels or positive joint rows, so the second
        # beta row is completely outside the routed gradient graph.
        np.testing.assert_allclose(result.beta[1], np.full(8, 1.0 / 8.0), atol=1e-7)

    def test_independent_gamma_bounds_do_not_require_a_fixed_sum(self):
        z, embeddings, _joint, _attrs, _failures = self._unified_fixture()
        state = MODELS.initial_unified_weight_state(
            2, theta=[0.5, 0.5], temperature=[0.15, 0.15], embedding_count=5
        )
        gamma = np.asarray([0.2, 2.5], dtype=np.float64)
        scores = MODELS.unified_weight_scores(
            z,
            embeddings,
            state["beta"],
            gamma,
            state["embedding_weights"],
            state["embedding_fusion_strength"],
            state["theta"],
            state["temperature"],
        )
        expected = np.prod(np.power(scores.gates, gamma[None, :]), axis=1)
        np.testing.assert_allclose(scores.conjunction_scores, expected, atol=1e-6)
        with self.assertRaises(ValueError):
            MODELS.unified_weight_scores(
                z,
                embeddings,
                state["beta"],
                [0.2, 3.01],
                state["embedding_weights"],
                state["embedding_fusion_strength"],
                state["theta"],
                state["temperature"],
            )

    def test_negative_failure_mask_updates_only_confirmed_gamma_coordinate(self):
        z = np.full((12, 2, 8), 0.8, dtype=np.float64)
        embeddings = np.ones((12, 5), dtype=np.float64)
        failure_mask = np.zeros((12, 2), dtype=bool)
        failure_mask[:, 0] = True
        result = MODELS.fit_weight_joint(
            z,
            embeddings,
            np.zeros(12, dtype=np.float64),
            theta=[0.5, 0.5],
            temperature=[0.15, 0.15],
            negative_failure_mask=failure_mask,
            gamma_regularization=0.0,
            max_iterations=30,
            outer_iterations=1,
            seed=8,
        )
        self.assertGreater(float(result.gamma[0]), 1.0)
        self.assertAlmostEqual(float(result.gamma[1]), 1.0, places=7)

    def test_log_domain_joint_bce_keeps_gradient_below_probability_epsilon(self):
        # Each gate is approximately exp(-100), far below the old probability
        # clamp.  A positive Joint label must still pull both independent gamma
        # coordinates down instead of receiving a zero gradient.
        z = np.zeros((6, 2, 8), dtype=np.float64)
        embeddings = np.ones((6, 5), dtype=np.float64)
        result = MODELS.fit_weight_joint(
            z,
            embeddings,
            np.ones(6, dtype=np.float64),
            theta=[1.0, 1.0],
            temperature=[0.01, 0.01],
            gamma_regularization=0.0,
            max_iterations=10,
            outer_iterations=1,
            seed=9,
        )
        self.assertTrue(np.all(result.gamma < 1.0))
        self.assertTrue(np.isfinite(result.final_objective))

    def test_total_iteration_budget_is_exact_for_staged_and_joint(self):
        z, embeddings, joint, attrs, failures = self._unified_fixture()
        common = dict(
            theta=[0.5, 0.5],
            temperature=[0.15, 0.15],
            attribute_labels=attrs,
            negative_failure_mask=failures,
            calibration_probe_scores=z,
            calibration_attribute_labels=attrs,
            max_iterations=11,
            seed=3,
        )
        staged = MODELS.fit_weight_staged(z, embeddings, joint, **common)
        joint_result = MODELS.fit_weight_joint(
            z, embeddings, joint, outer_iterations=4, **common
        )
        self.assertEqual(staged.iterations, 11)
        self.assertEqual(joint_result.iterations, 11)
        self.assertEqual(staged.training_history[-1]["iteration"], 11)
        self.assertEqual(joint_result.training_history[-1]["iteration"], 11)
        staged_records = [
            item for item in staged.training_history if item["stage"] in {"beta", "query"}
        ]
        self.assertEqual([item["iteration"] for item in staged_records], [5, 11])
        joint_records = [
            item
            for item in joint_result.training_history
            if str(item["stage"]).startswith("joint-")
        ]
        self.assertEqual([item["iteration"] for item in joint_records], [3, 6, 9, 11])


    def test_iteration_contract_rejects_non_integral_or_too_small_budgets(self):
        z, embeddings, joint, attrs, failures = self._unified_fixture()
        common = dict(
            theta=[0.5, 0.5],
            temperature=[0.15, 0.15],
            attribute_labels=attrs,
            negative_failure_mask=failures,
        )
        for invalid in (True, 1, 2.5):
            with self.subTest(max_iterations=invalid), self.assertRaises(ValueError):
                MODELS.fit_weight_staged(
                    z,
                    embeddings,
                    joint,
                    max_iterations=invalid,
                    **common,
                )

    def test_four_modes_differ_only_in_upstream_snapshot_and_fusion_schedule(self):
        z, embeddings, joint, attrs, failures = self._unified_fixture()
        original = z.copy()
        common = dict(
            theta=[0.5, 0.5], temperature=[0.15, 0.15],
            attribute_labels=attrs, negative_failure_mask=failures,
            calibration_probe_scores=z, calibration_attribute_labels=attrs,
            max_iterations=12, outer_iterations=2, seed=17,
        )
        for schedule in ("staged", "joint"):
            with self.subTest(schedule=schedule):
                first = MODELS.fit_unified_weight_refinement(
                    f"weight_{schedule}", z, embeddings[:, :2], joint, **common,
                )
                second = MODELS.fit_unified_weight_refinement(
                    f"probe_{schedule}", z, embeddings[:, :2], joint, **common,
                )
                self.assertEqual(second.mode, f"probe_{schedule}")
                self.assertEqual(MODELS.fusion_training_schedule(second.mode), schedule)
                for field in ("initial_scores", "refined_scores", "beta", "gamma",
                              "theta", "embedding_weights"):
                    np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
                self.assertEqual(first.training_history, second.training_history)
        np.testing.assert_array_equal(z, original)

    def test_fusion_cannot_receive_live_probe_autograd_graph(self):
        torch = MODELS._require_torch()
        z, embeddings, joint, attrs, failures = self._unified_fixture()
        probe_output = torch.tensor(z, requires_grad=True)
        common = dict(
            theta=[0.5, 0.5], temperature=[0.15, 0.15],
            attribute_labels=attrs, negative_failure_mask=failures,
            max_iterations=4,
        )
        for mode in ("weight_staged", "weight_joint", "probe_staged", "probe_joint"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "frozen score snapshot"):
                MODELS.fit_unified_weight_refinement(
                    mode, probe_output, embeddings[:, :2], joint, **common,
                )
        self.assertIsNone(probe_output.grad)
        result = MODELS.fit_unified_weight_refinement(
            "probe_joint", probe_output.detach(), embeddings[:, :2], joint, **common,
        )
        self.assertEqual(result.mode, "probe_joint")
        self.assertIsNone(probe_output.grad)

    def test_frozen_score_copy_cannot_share_mutable_probe_buffer(self):
        source = np.zeros((2, 1, 8), dtype=np.float64)
        snapshot = MODELS._frozen_score_array(source, "probe scores")
        source[:] = 1.0
        np.testing.assert_array_equal(snapshot, 0.0)


if __name__ == "__main__":
    unittest.main()
