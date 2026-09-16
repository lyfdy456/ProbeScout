import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_probebank_batch as runner  # noqa: E402


class RSoftGateTest(unittest.TestCase):
    def setUp(self):
        self.old_attrs = list(runner.H.ATTRS)
        self.old_spec = copy.deepcopy(runner.H.RANKING_SPEC)
        runner.H.ATTRS = ["a", "b"]
        runner.H.RANKING_SPEC = {
            "a": {"attrs": ("a",)},
            "b": {"attrs": ("b",)},
            runner.H.JOINT_KEY: {"attrs": ("a", "b")},
        }
        self.paths = [f"image_{i}.jpg" for i in range(16)]
        self.p2i = {path: i for i, path in enumerate(self.paths)}
        train_truth = (
            [(1, 1)] * 4 + [(1, 0)] * 3 + [(0, 1)] * 3 + [(0, 0)] * 2
        )
        truth = train_truth + [(1, 1), (1, 0), (0, 1), (0, 0)]
        self.selected = self.paths[:12]
        self.labels = [
            {"image": path, "a": a, "b": b}
            for path, (a, b) in zip(self.selected, train_truth)
        ]
        a = np.asarray([row[0] for row in truth], dtype=np.float64)
        b = np.asarray([row[1] for row in truth], dtype=np.float64)
        self.scores = {
            "a_best": self.score_set(0.05 + 0.90 * a, 0.95 - 0.90 * b),
            "b_best": self.score_set(0.95 - 0.90 * a, 0.05 + 0.90 * b),
            "both_medium": self.score_set(0.20 + 0.60 * a, 0.20 + 0.60 * b),
            "bad": self.score_set(0.90 - 0.80 * a, 0.90 - 0.80 * b),
        }

    def tearDown(self):
        runner.H.ATTRS = self.old_attrs
        runner.H.RANKING_SPEC = self.old_spec

    @staticmethod
    def score_set(a, b):
        return {"a": np.asarray(a), "b": np.asarray(b), "joint": np.asarray(a) * b}

    @staticmethod
    def recipe(top_k):
        size = 4 if top_k == "all" else int(top_k)
        return {
            "id": f"r_softgate_{top_k}",
            "aggregation": "r_softgate",
            "joint_rule": "product",
            "member_joint_rule": "product",
            "pool_id": f"top{top_k}",
            "pool_size": size,
            "top_k": top_k,
            "members": ["a_best", "b_best", "both_medium", "bad"],
        }

    def fit(self, top_k, scores=None):
        return runner.fit_r_softgate(
            self.recipe(top_k), scores or self.scores,
            self.selected, self.labels, self.p2i,
        )

    def test_top1_is_attribute_specific_and_parameters_are_independent(self):
        fitted = self.fit(1)
        self.assertEqual(fitted["selected_members_by_attr"]["a"], ["a_best"])
        self.assertEqual(fitted["selected_members_by_attr"]["b"], ["b_best"])
        self.assertEqual(set(fitted["theta_by_attr"]), {"a", "b"})
        self.assertEqual(set(fitted["temperature_by_attr"]), {"a", "b"})
        self.assertTrue(all(0.0 <= value <= 1.0 for value in fitted["theta_by_attr"].values()))
        self.assertTrue(all(value > 0.0 for value in fitted["temperature_by_attr"].values()))
        json.dumps(runner.fusion_fit_output(fitted))

    def test_top1_top3_all_select_exact_requested_count(self):
        for top_k, expected in ((1, 1), (3, 3), ("all", 4)):
            fitted = self.fit(top_k)
            self.assertTrue(all(
                len(members) == expected
                for members in fitted["selected_members_by_attr"].values()
            ))

    def test_fit_never_reads_held_out_rows(self):
        changed = copy.deepcopy(self.scores)
        for method in changed.values():
            for key in method:
                method[key][12:] = np.asarray([1.0, 0.0, 1.0, 0.0])
        fitted = self.fit(3)
        refitted = self.fit(3, changed)
        self.assertEqual(fitted["selected_members_by_attr"], refitted["selected_members_by_attr"])
        np.testing.assert_allclose(
            list(fitted["theta_by_attr"].values()),
            list(refitted["theta_by_attr"].values()), atol=1e-9,
        )
        np.testing.assert_allclose(
            list(fitted["temperature_by_attr"].values()),
            list(refitted["temperature_by_attr"].values()), atol=1e-9,
        )

    def test_inference_multiplies_gates_not_scores_times_gates(self):
        fitted = {
            "selected_members_by_attr": {"a": ["a_best"], "b": ["b_best"]},
            "theta_by_attr": {"a": 0.50, "b": 0.50},
            "temperature_by_attr": {"a": 0.20, "b": 0.25},
        }
        fused = runner.r_softgate_fusion(self.scores, fitted)
        gate_a = runner._stable_sigmoid((self.scores["a_best"]["a"] - 0.50) / 0.20)
        gate_b = runner._stable_sigmoid((self.scores["b_best"]["b"] - 0.50) / 0.25)
        np.testing.assert_allclose(fused["a"], gate_a)
        np.testing.assert_allclose(fused["b"], gate_b)
        np.testing.assert_allclose(fused[runner.H.JOINT_KEY], gate_a * gate_b)
        self.assertFalse(np.allclose(fused["a"], self.scores["a_best"]["a"] * gate_a))

    def test_joint_pairwise_ranking_loss_uses_the_same_gate_parameters(self):
        recipe = self.recipe("all")
        recipe.update({
            "loss": "joint_pairwise_ranking",
            "ranking_temperature": 0.1,
        })
        fitted = runner.fit_r_softgate(
            recipe, self.scores, self.selected, self.labels, self.p2i,
        )
        self.assertEqual(fitted["loss"], "joint_pairwise_ranking")
        self.assertTrue(np.isfinite(fitted["train_pairwise_ranking_loss"]))
        self.assertIsNone(fitted["train_bce"])
        self.assertEqual(set(fitted["theta_by_attr"]), {"a", "b"})
        self.assertEqual(set(fitted["temperature_by_attr"]), {"a", "b"})
        json.dumps(runner.fusion_fit_output(fitted))

    def test_min_and_max_are_applied_before_the_attribute_gate(self):
        members = {
            "a": ["a_best", "bad"],
            "b": ["b_best", "bad"],
        }
        for aggregation, reducer in (("min", np.min), ("max", np.max)):
            fitted = {
                "probe_aggregation": aggregation,
                "selected_members_by_attr": members,
                "theta_by_attr": {"a": 0.50, "b": 0.50},
                "temperature_by_attr": {"a": 0.20, "b": 0.20},
            }
            fused = runner.r_softgate_fusion(self.scores, fitted)
            expected_gates = {}
            for attr in ("a", "b"):
                values = np.column_stack([
                    self.scores[member][attr] for member in members[attr]
                ])
                attribute_input = reducer(values, axis=1)
                expected_gates[attr] = runner._stable_sigmoid(
                    (attribute_input - 0.50) / 0.20
                )
                np.testing.assert_allclose(fused[attr], expected_gates[attr])
            np.testing.assert_allclose(
                fused[runner.H.JOINT_KEY], expected_gates["a"] * expected_gates["b"],
            )


if __name__ == "__main__":
    unittest.main()
