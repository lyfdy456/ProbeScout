"""Original-VQA-only refinement uses temporary data, never real user sessions."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_val_initial_publication as publication
from tests import test_tuning_backend as backend
import tuning_models


class OriginalOnlyWeightTests(unittest.TestCase):
    setUp = publication.ValInitialPublicationTests.setUp
    tearDown = publication.ValInitialPublicationTests.tearDown
    bootstrap = publication.ValInitialPublicationTests.bootstrap
    build = publication.ValInitialPublicationTests.build
    stage = publication.ValInitialPublicationTests.stage

    def _session(self):
        client = backend.ApiClient(self.base_url)
        with patch.object(self.service, "refinement_capabilities", return_value={}):
            bootstrap = self.bootstrap(client, "Original supervision only")
        return client, bootstrap["session"]["id"]

    def _original_only_run(self, uncertain):
        _, (base, _, directory) = self.stage()
        publication.INITIAL.publish(self.service, "val-new")
        for mode in ("weight_staged", "weight_joint"):
            client, session = self._session()
            if uncertain:
                status, response = client.request("PUT", f"/api/tuning/sessions/{session}/annotations/0",
                    {"imageId": self.image_ids[0], "label": 0, "source": "manual"})
                self.assertEqual(status, 200, response)
            status, response = client.request("POST", f"/api/tuning/sessions/{session}/runs", {"mode": mode, "maxIterations": 10})
            self.assertEqual(status, 202, response)
            self.assertEqual(response["run"]["annotationCount"], 0)
            with self.service.connect() as connection:
                run = connection.execute("SELECT r.*,s.task_id,s.target_id FROM model_runs r JOIN sessions s ON s.id=r.session_id WHERE r.id=?",
                    (response["run"]["id"],)).fetchone()
            params = json.loads(run["params_json"])
            self.assertEqual(params["feedbackMode"], "original-only")
            run_path = self.runtime_root / run["artifact_relpath"]
            snapshot = json.loads((run_path / "labels_snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["annotations"], [])
            prepared = self.service._prepare_weight_refinement_supervision(backend.TASK_ID, base, [], feedback_weight=8,
                vqa_validation=True, vqa_validation_version=params["vqaValidationVersion"])
            np.testing.assert_array_equal(prepared["jointRows"], [0, 1, 2, 3])
            np.testing.assert_array_equal(prepared["jointWeights"], [1, 1, 1, 1])
            self.assertEqual(prepared["audit"]["feedbackCount"], 0)
            self.assertEqual(prepared["audit"]["combinedCount"], 4)
            for example in prepared["attributeExamples"]:
                np.testing.assert_array_equal(example["rows"], [0, 1, 2, 3])
                np.testing.assert_array_equal(example["weights"], [1, 1, 1, 1])
            # Original confirmed attribute labels still route negative gradients.
            np.testing.assert_array_equal(prepared["jointFailureMask"], np.ones((4, 1), dtype=bool))
            with patch.object(tuning_models, "fit_unified_weight_refinement", wraps=tuning_models.fit_unified_weight_refinement) as fit:
                before, after = self.service._execute_model_run(run)
            self.assertTrue(fit.called)
            self.assertEqual(fit.call_args.args[1].shape[0], 4)
            self.assertEqual((run_path / "initial-scores.f32").read_bytes(), (directory / "scores.f32").read_bytes())
            self.assertTrue(np.isfinite(before["ap"]))
            self.assertTrue(np.isfinite(after["ap"]))
            with np.load(run_path / "model.npz", allow_pickle=False) as model:
                np.testing.assert_array_equal(model["theta"], base.initial_theta)
                np.testing.assert_array_equal(model["temperature"], base.temperature)

    def test_empty_feedback_trains_both_schedules_on_original_fit_only(self):
        self._original_only_run(uncertain=False)

    def test_uncertain_only_feedback_is_not_a_training_label(self):
        self._original_only_run(uncertain=True)

    def test_update_probes_still_requires_feedback(self):
        self.stage()
        publication.INITIAL.publish(self.service, "val-new")
        client, session = self._session()
        status, response = client.request("POST", f"/api/tuning/sessions/{session}/probe-updates", {})
        self.assertEqual(status, 409, response)
        self.assertIn("feedback", response["error"].lower())

    def test_existing_feedback_entirely_in_val_does_not_silently_become_original_only(self):
        self.stage()
        publication.INITIAL.publish(self.service, "val-new")
        client, session = self._session()
        status, response = client.request("PUT", f"/api/tuning/sessions/{session}/annotations/0",
            {"imageId": self.image_ids[0], "label": 1, "source": "manual"})
        self.assertEqual(status, 200, response)
        split = backend.vqa_contract_fixture(backend.clean_validation_split(
            fit_indices=[1, 2, 3, 4], fit_labels=[0, 0, 1, 1], validation_indices=[0, 6], validation_labels=[0, 1],
            fingerprint="all-feedback-in-val"))
        self.service._vqa_validation_split = lambda *args: split
        for mode in ("weight_staged", "weight_joint"):
            status, response = client.request("POST", f"/api/tuning/sessions/{session}/runs", {"mode": mode})
            self.assertEqual(status, 409, response)
            self.assertIn("fixed VQA Validation", response["error"])


if __name__ == "__main__":
    unittest.main()
