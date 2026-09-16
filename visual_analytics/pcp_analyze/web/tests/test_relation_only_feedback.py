"""Task-local relation feedback; all HTTP state and scores are temporary fixtures."""
import dataclasses
import json
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_tuning_backend as fixture
import tuning_models

TUNING = fixture.TUNING
ROBUST = "059_hico_task_hico_hugging_cat_robust_test"


class RelationOnlyFeedbackTests(unittest.TestCase):
    setUp = fixture.TuningApiTests.setUp
    tearDown = fixture.TuningApiTests.tearDown

    def setup_task(self, task_id=ROBUST, target="joint"):
        parent = self.service.tasks[fixture.TASK_ID]
        manifest = {**parent.manifest, "retrievalTargets": [
            {"id": "first", "kind": "attribute"}, {"id": "second", "kind": "attribute"},
            {"id": "joint", "kind": "derived"}]}
        self.service.tasks[task_id] = dataclasses.replace(parent, task_id=task_id,
            target_ids=("first", "second", "joint"), manifest=manifest)
        self.service._suggest_failed_attributes = lambda task, rows: {int(row): "first" for row in rows}
        split = fixture.vqa_contract_fixture(fixture.clean_validation_split(
            fit_indices=[0, 1, 3, 5], fit_labels=[0, 0, 1, 1],
            validation_indices=[6, 8], validation_labels=[1, 0], fingerprint="relation-fixture"))
        self.service._vqa_validation_split = lambda *args: split
        base = TUNING.RefinementBaseContext(
            attribute_ids=("first", "second"), attribute_names=("First", "Second"),
            learner_names=TUNING.WEIGHTED_FUSION_LEARNERS, embedding_names=TUNING.REFINEMENT_EMBEDDING_METHODS,
            probe_features=np.full((9, 2, 8), .8, dtype=np.float32),
            probe_min=np.zeros((2, 8)), probe_max=np.ones((2, 8)),
            initial_theta=np.full(2, .5), temperature=np.full(2, .15),
            embedding_features=np.tile([.1, .6], (9, 1)), embedding_min=np.zeros(2), embedding_max=np.ones(2),
            development_indices=np.arange(4), base_state_fingerprint="f" * 64,
            normalization_audit=self.service._refinement_normalization_scope(task_id)[1])
        self.service._refinement_base_context = lambda *args, **kwargs: base
        client = fixture.ApiClient(self.base_url)
        with patch.object(self.service, "refinement_capabilities", return_value={}):
            status, response = client.request("POST", "/api/tuning/bootstrap", {
                "taskId": task_id, "targetId": target, "baseMethod": "Ours-Full", "displayName": "Relation fixture"})
        self.assertEqual(status, 200, response)
        return client, response["session"]["id"], base

    def put(self, client, session, row=2, **values):
        return client.request("PUT", f"/api/tuning/sessions/{session}/annotations/{row}", {
            "imageId": self.image_ids[row], "label": -1, **values})

    def test_explicit_confirmation_required_and_only_experimental_joint_accepts_it(self):
        for task_id, target in [(fixture.TASK_ID, "joint"), (ROBUST, "first"), (ROBUST, "joint")]:
            with self.subTest(task=task_id, target=target):
                client, session, _ = self.setup_task(task_id, target)
                status, response = self.put(client, session, failureAttributionConfirmed=True)
                self.assertEqual(status, 400, f"Explicit [] is required, not just a confirmation flag: {response}")
                status, response = self.put(client, session, failedAttributeIds=[], failureAttributionConfirmed=True)
                expected = 200 if (task_id, target) == (ROBUST, "joint") else 400
                self.assertEqual(status, expected, response)

    def test_unconfirmed_negative_is_rejected_then_relation_can_be_queued_without_attribute_labels(self):
        client, session, base = self.setup_task()
        status, unconfirmed = self.put(client, session)
        self.assertEqual(status, 200, unconfirmed)
        self.assertFalse(unconfirmed["annotation"]["failureAttributionConfirmed"])
        status, rejected = client.request("POST", f"/api/tuning/sessions/{session}/runs", {"mode": "weight_staged"})
        self.assertEqual(status, 409, rejected)
        status, response = self.put(client, session, failedAttributeIds=[], failureAttributionConfirmed=True)
        self.assertEqual(status, 200, response)
        annotation = response["annotation"]
        self.assertEqual(annotation["failedAttributeIds"], [])
        self.assertTrue(annotation["failureAttributionConfirmed"])
        self.assertNotEqual(TUNING.TuningService.annotation_state_sha256([annotation]),
            TUNING.TuningService.annotation_state_sha256([unconfirmed["annotation"]]))
        prepared = self.service._prepare_weight_refinement_supervision(ROBUST, base, [annotation],
            feedback_weight=8, vqa_validation=True)
        training = self.service._materialize_refinement_training_arrays(base, prepared)
        for example in prepared["attributeExamples"]:
            self.assertNotIn(2, example["rows"])
        position = list(prepared["jointRows"]).index(2)
        self.assertEqual(prepared["jointLabels"][position], 0)
        self.assertEqual(prepared["jointWeights"][position], 8)
        self.assertFalse(prepared["jointFailureMask"][position].any())
        self.assertEqual(prepared["audit"]["relationOnlyFeedbackCount"], 1)
        self.assertTrue(np.isnan(training["attributeLabels"][list(training["rows"]).index(2)]).all())
        for mode in ("weight_staged", "weight_joint"):
            status, result = client.request("POST", f"/api/tuning/sessions/{session}/runs", {"mode": mode, "maxIterations": 10})
            self.assertEqual(status, 202, result)
            with self.service.connect() as connection:
                run = connection.execute("SELECT * FROM model_runs WHERE id=?", (result["run"]["id"],)).fetchone()
                params = json.loads(run["params_json"])
                self.assertEqual(params["relationMismatchPolicy"], "explicit-joint-negative-unknown-attributes-v1")
                saved = json.loads((self.runtime_root / run["artifact_relpath"] / "labels_snapshot.json").read_text())
                self.assertEqual(saved["annotations"][0]["failedAttributeIds"], [])
                self.assertTrue(saved["annotations"][0]["failureAttributionConfirmed"])
                connection.execute("UPDATE model_runs SET status='cancelled' WHERE id=?", (run["id"],))
        status, rejected = client.request("POST", f"/api/tuning/sessions/{session}/probe-updates", {})
        self.assertEqual(status, 409, rejected)

    def test_query_validation_and_test_safety_is_not_relaxed(self):
        client, session, _ = self.setup_task()
        for row in (4, 5, 6, 8):
            status, result = self.put(client, session, row=row, failedAttributeIds=[], failureAttributionConfirmed=True)
            self.assertEqual(status, 409, result)
        bundle = self.service.bundle(ROBUST)
        development = bytearray(bundle.development_mask)
        test = bytearray(bundle.test_mask)
        development[2], test[2] = 0, 1
        changed = dataclasses.replace(bundle, development_mask=bytes(development), test_mask=bytes(test))
        with patch.object(self.service, "bundle", return_value=changed):
            status, result = self.put(client, session, failedAttributeIds=[], failureAttributionConfirmed=True)
            self.assertEqual(status, 409, result)

    def test_attribute_confirmation_and_label_changes_preserve_existing_semantics(self):
        client, session, _ = self.setup_task()
        status, response = self.put(client, session, failedAttributeIds=["first"], failureAttributionConfirmed=True)
        self.assertEqual(status, 200, response)
        self.assertEqual(response["annotation"]["failedAttributeIds"], ["first"])
        status, response = self.put(client, session, failedAttributeIds=[], failureAttributionConfirmed=True)
        self.assertEqual(status, 200, response)
        status, response = self.put(client, session, label=-2)
        self.assertEqual(status, 200, response)
        self.assertEqual(response["annotation"]["failedAttributeIds"], [])
        self.assertTrue(response["annotation"]["failureAttributionConfirmed"])
        status, response = self.put(client, session, label=1)
        self.assertEqual(status, 200, response)
        self.assertNotIn("failureAttributionConfirmed", response["annotation"])
        status, response = self.put(client, session)
        self.assertEqual(status, 200, response)
        self.assertFalse(response["annotation"]["failureAttributionConfirmed"], "a newly negative label requires a fresh confirmation")

    def test_relation_only_loss_cannot_update_beta_or_gamma_but_updates_holistic_weights(self):
        for mode in ("weight_staged", "weight_joint"):
            result = tuning_models.fit_unified_weight_refinement(
                mode, np.full((6, 2, 8), .8), np.tile([.1, .6], (6, 1)), np.zeros(6),
                theta=[.5, .5], temperature=[.15, .15],
                attribute_labels=np.full((6, 2), np.nan), negative_failure_mask=np.zeros((6, 2), dtype=bool),
                beta_regularization=0, gamma_regularization=0, max_iterations=10, outer_iterations=1)
            np.testing.assert_allclose(result.beta, np.full((2, 8), 1/8), atol=1e-7)
            np.testing.assert_allclose(result.gamma, np.ones(2), atol=1e-7)
            self.assertGreater(result.embedding_fusion_strength, .25)
            self.assertGreater(result.embedding_weights[0], .5)


if __name__ == "__main__":
    unittest.main()
