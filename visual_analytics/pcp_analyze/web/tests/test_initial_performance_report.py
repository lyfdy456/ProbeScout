from __future__ import annotations
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_initial_performance import evaluate_task_method, evaluation_metrics, fit_training_threshold


class InitialPerformanceTests(unittest.TestCase):
    def test_equal_score_group_cannot_be_split(self):
        result = fit_training_threshold([1, 0, 1, 0], [.9, .9, .8, .1])
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["trainF1"], .8)
        tied = fit_training_threshold([1, 0], [.5, .5])
        self.assertEqual(tied["trainF1"], 2 / 3)

    def test_f1_tie_selects_highest_threshold(self):
        # P=2: first cutoff 2/(2+1) = last cutoff 4/(2+4).
        result = fit_training_threshold([1, 0, 0, 1], [.9, .8, .7, .6])
        self.assertEqual(result["threshold"], .9)
        self.assertEqual(result["trainF1"], 2 / 3)

    def test_negative_clay_cosine_is_valid(self):
        selected = fit_training_threshold([1, 0, 1, 0], [-.1, -.7, -.2, -.8])
        self.assertEqual(selected["threshold"], -.2)
        result = evaluation_metrics([1, 0, 1, 0], [-.1, -.7, -.2, -.8], selected["threshold"])
        self.assertEqual(result["Ap"], 1.0)
        self.assertEqual(result["F1"], 1.0)

    def test_ap_ties_use_stable_input_gallery_rows(self):
        result = evaluation_metrics([0, 1, 1], [.5, .5, .4], .5)
        self.assertAlmostEqual(result["Ap"], (.5 + 2 / 3) / 2)
        self.assertEqual(result["TiedRows"], 1)
        self.assertEqual(result["F1"], .5)

    def test_no_positive_ap_is_undefined(self):
        result = evaluation_metrics([0, 0], [.8, .1], .5)
        self.assertIsNone(result["Ap"])
        self.assertEqual(result["F1"], 0.0)
        selected = fit_training_threshold([0, 0], [.8, .1])
        self.assertGreater(selected["threshold"], .8)
        self.assertEqual(selected["trainFP"], 0)

    def test_evaluation_labels_do_not_affect_training_threshold(self):
        task = {"task_id": "001_task", "targets": ["joint"], "truth": np.asarray([[0], [0], [1], [1]]),
                "train": {"fitRows": [0, 1], "fitLabels": [1, 0]},
                "test": np.asarray([0, 0, 1, 1]), "label": "fixture", "dataset": "toy"}
        method = {"methodId": "fixture", "methodLabel": "fixture", "methodOrder": 1}
        before = evaluate_task_method(task, method, [.8, .7, .95, .4])
        task["truth"] = 1 - task["truth"]
        after = evaluate_task_method(task, method, [.8, .7, .95, .4])
        self.assertEqual(before["threshold"], .8)
        self.assertEqual(after["threshold"], before["threshold"])
        self.assertEqual(after["trainF1"], before["trainF1"])

    def test_frozen_threshold_not_best_eval_f1(self):
        result = evaluation_metrics([1, 1, 0], [.7, .6, .2], .8)
        self.assertEqual(result["Ap"], 1.0)
        self.assertEqual(result["F1"], 0.0)
        self.assertEqual(result["FN"], 2)

    def test_invalid_inputs_rejected(self):
        for labels, scores in (([0], [float("nan")]), ([2], [.1]), ([1, 0], [.1])):
            with self.assertRaises(ValueError):
                fit_training_threshold(labels, scores)
        with self.assertRaises(ValueError):
            fit_training_threshold([], [])


if __name__ == "__main__":
    unittest.main()
