"""Frozen Test is diagnostic output, not Tune supervision or selection input."""
from __future__ import annotations

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests import test_tuning_backend as backend
import tuning_models


class TuningTestEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = backend.TuningApiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)
        self.service = self.fixture.service
        self.data_root = self.service.task(backend.TASK_ID).data_root

    def _install_test_split(self, *, positives=True):
        # Disjoint fit [0..3], Query [4], Val [5,6], Test [7,8].
        # Val VQA labels [1,0] intentionally differ from public GT [1,1].
        (self.data_root / "validation-mask.u8").write_bytes(bytes([0] * 5 + [1, 1, 0, 0]))
        (self.data_root / "test-mask.u8").write_bytes(bytes([0] * 7 + [1, 1]))
        (self.data_root / "ground-truth.u8").write_bytes(
            bytes([0, 0, 0, 0, 0, 1, 1, int(positives), 0])
        )

    def _runs(self):
        with self.service.connect() as connection:
            return connection.execute("SELECT * FROM model_runs ORDER BY created_at,id").fetchall()

    def _metrics(self, run):
        directory = self.fixture.runtime_root / run["artifact_relpath"]
        return directory, json.loads((directory / "metrics.json").read_text(encoding="utf-8"))

    def _check_current_source(self, *, native):
        self._install_test_split()
        events = []
        optimizer = tuning_models.fit_unified_weight_refinement
        read_truth = self.service._task_ground_truth

        def fit(*args, **kwargs):
            events.append("fit-start")
            result = optimizer(*args, **kwargs)
            events.append("fit-finished")
            return result

        def diagnostic_truth(task_id):
            self.assertEqual(events[-1], "fit-finished")
            events.append("test-read")
            return read_truth(task_id)

        self.service._task_ground_truth = diagnostic_truth
        input_hashes = {
            name: hashlib.sha256((self.data_root / name).read_bytes()).hexdigest()
            for name in ("test-mask.u8", "ground-truth.u8", "ranks.f32")
        }
        with mock.patch.object(tuning_models, "fit_unified_weight_refinement", side_effect=fit):
            self.fixture._check_weight_refinement_modes(
                native=native, allow_post_training_test=True,
            )
        self.assertEqual(events, ["fit-start", "fit-finished", "test-read"] * 2)
        self.assertEqual(len(self._runs()), 2)
        for run in self._runs():
            with self.subTest(source="updated" if native else "original", mode=run["mode"]):
                directory, metrics = self._metrics(run)
                public = self.service.run_json(run)
                diagnostic = public["testEvaluation"]
                self.assertEqual(run["status"], "succeeded")
                self.assertEqual(public["evaluationScope"], "vqa-validation")
                self.assertEqual(public["probeSource"], "updated" if native else "original")
                self.assertEqual(diagnostic, metrics["testEvaluation"])
                self.assertEqual(diagnostic["protocol"], "post-training-frozen-test-v1")
                self.assertEqual(diagnostic["evaluationScope"], "test")
                self.assertEqual(diagnostic["targetId"], "joint")
                self.assertEqual(diagnostic["beforeMethod"], "Unified base F⁽⁰⁾")
                self.assertEqual(diagnostic["testMaskSha256"], input_hashes["test-mask.u8"])
                self.assertEqual(diagnostic["groundTruthSha256"], input_hashes["ground-truth.u8"])
                before_ranks = np.fromfile(directory / "initial-ranks.f32", dtype="<f4")
                after_ranks = np.fromfile(directory / "tuned-ranks.f32", dtype="<f4")
                for key, ranks in (("before", before_ranks), ("after", after_ranks)):
                    self.assertEqual(diagnostic[key], backend.TUNING.retrieval_metrics([1, 0], ranks[[7, 8]]))
                    self.assertEqual(public[key]["ap"], backend.TUNING.retrieval_metrics([1, 0], ranks[[5, 6]])["ap"])
                    self.assertEqual(public[key]["positiveCount"], 1)
                    self.assertEqual(public[key]["evaluationCount"], 2)
                # Original F0 rises on [7,8], so the positive row is second;
                # exported legacy Ours-Full has the opposite Test ordering.
                self.assertEqual(diagnostic["before"]["ap"], 0.5)
                self.assertNotIn("testEvaluation", public["after"])
                self.assertEqual(public["deltaAp"], public["after"]["ap"] - public["before"]["ap"])
                self.assertIn("never fitting, calibration or checkpoint selection", metrics["groundTruthUsage"])
        for name, expected in input_hashes.items():
            self.assertEqual(hashlib.sha256((self.data_root / name).read_bytes()).hexdigest(), expected)

    def test_original_staged_and_joint_report_test_after_training(self):
        self._check_current_source(native=False)

    def test_updated_staged_and_joint_keep_original_f0_before(self):
        self._check_current_source(native=True)

    def test_empty_test_omits_diagnostic_without_reading_ground_truth(self):
        self.fixture._check_weight_refinement_modes()
        for run in self._runs():
            public = self.service.run_json(run)
            self.assertEqual(public["status"], "succeeded")
            self.assertNotIn("testEvaluation", public)
            self.assertIn("no evaluation images", public["testEvaluationError"])
            self.assertEqual(public["before"]["evaluationCount"], 2)

    def test_no_test_positives_keeps_successful_val_result(self):
        self._install_test_split(positives=False)
        self.fixture._check_weight_refinement_modes(allow_post_training_test=True)
        for run in self._runs():
            public = self.service.run_json(run)
            self.assertEqual(public["status"], "succeeded")
            self.assertNotIn("testEvaluation", public)
            self.assertIn("no positive examples", public["testEvaluationError"])
            self.assertEqual(public["after"]["positiveCount"], 1)

    def test_invalid_ground_truth_keeps_successful_val_result(self):
        self._install_test_split()
        (self.data_root / "ground-truth.u8").write_bytes(bytes([1]))
        self.fixture._check_weight_refinement_modes(allow_post_training_test=True)
        for run in self._runs():
            public = self.service.run_json(run)
            self.assertEqual(public["status"], "succeeded")
            self.assertNotIn("testEvaluation", public)
            self.assertIn("Ground-truth shape is invalid", public["testEvaluationError"])


    def test_changing_test_labels_does_not_change_trained_outputs(self):
        self._install_test_split()
        self.fixture._check_weight_refinement_modes(allow_post_training_test=True)
        original = {}
        for run in self._runs():
            directory, metrics = self._metrics(run)
            original[run["mode"]] = {
                "id": run["id"],
                "scores": (directory / "tuned-scores.f32").read_bytes(),
                "ranks": (directory / "tuned-ranks.f32").read_bytes(),
                "val": metrics["after"],
                "testAp": metrics["testEvaluation"]["after"]["ap"],
            }
        (self.data_root / "ground-truth.u8").write_bytes(bytes([0, 0, 0, 0, 0, 1, 1, 0, 1]))
        self.fixture._check_weight_refinement_modes(allow_post_training_test=True)
        compared = 0
        for run in self._runs():
            previous = original[run["mode"]]
            if run["id"] == previous["id"]:
                continue
            directory, metrics = self._metrics(run)
            self.assertEqual((directory / "tuned-scores.f32").read_bytes(), previous["scores"])
            self.assertEqual((directory / "tuned-ranks.f32").read_bytes(), previous["ranks"])
            self.assertEqual(metrics["after"], previous["val"])
            self.assertNotEqual(metrics["testEvaluation"]["after"]["ap"], previous["testAp"])
            compared += 1
        self.assertEqual(compared, 2)


    def test_diagnostic_selects_current_target_and_only_test_rows(self):
        task = SimpleNamespace(task_id=backend.TASK_ID, row_count=4, target_ids=("first", "joint"))
        bundle = SimpleNamespace(test_mask=bytes([1, 0, 1, 1]))
        truth = np.asarray([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.uint8)
        self.service._task_ground_truth = lambda task_id: truth
        ranks = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        result = self.service._post_training_test_evaluation(
            task, bundle, "joint", ranks, ranks[::-1], "Unified base F⁽⁰⁾",
        )
        self.assertEqual(result["before"], backend.TUNING.retrieval_metrics([0, 1, 0], ranks[[0, 2, 3]]))
        self.assertEqual(result["after"], backend.TUNING.retrieval_metrics([0, 1, 0], ranks[::-1][[0, 2, 3]]))
        self.assertEqual(result["before"]["evaluationCount"], 3)
        self.assertEqual(result["before"]["positiveCount"], 1)
        self.assertEqual(result["groundTruthSha256"], hashlib.sha256(truth.tobytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
