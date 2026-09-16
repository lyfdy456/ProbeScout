"""Bounded synthetic Val publication + real Weight Tune request/worker checks."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_unified_initial_baseline import InitialBaselineTests, INITIAL, ApiClient, TASK_ID
from publish_val_initial_baseline import stage_task
from tuning_models import unified_weight_scores


class ValInitialPublicationTests(unittest.TestCase):
    setUp = InitialBaselineTests.setUp
    tearDown = InitialBaselineTests.tearDown
    bootstrap = InitialBaselineTests.bootstrap
    build = InitialBaselineTests.build

    def stage(self):
        old_base, old, path = self.build("old")
        old["bankTrainingIdentity"]["isolationFingerprint"] = "val-contract"
        INITIAL.write(path / "manifest.json", old)
        INITIAL.publish(self.service, "old")
        old_base = INITIAL.load_task_baseline(self.service, TASK_ID, "old")[0]
        inputs = self.web_root.parent / "inputs"
        (inputs / TASK_ID).mkdir(parents=True)
        theta = old_base.initial_theta.astype(np.float64) + .05
        temperature = old_base.temperature.astype(np.float64) * 1.5
        output = unified_weight_scores(old_base.probe_features, old_base.embedding_features,
            np.full((1, 8), 1 / 8), np.ones(1), [.5, .5], .25, theta, temperature)
        (inputs / TASK_ID / "scores.f32").write_bytes(output.final_scores.astype("<f4").tobytes())
        contract_path = inputs / "contract.json"
        INITIAL.write(contract_path, {"fingerprint": "val-contract", "valRows": [5, 6],
            "normalizationRows": list(map(int, old_base.development_indices)), "testRows": [7, 8],
            "targets": {"joint": {"fitRows": [0, 1, 2, 3]}}})
        selection = {"sourceVersion": "old", "selectedAt": "synthetic", "gateSelection": "Val AP", "cutoffSelection": "Val F1"}
        INITIAL.write(inputs / "val_selection.json", selection)
        record = {"taskId": TASK_ID, "attributeIds": list(old_base.attribute_ids), "rowCount": 9,
            "actualTheta": theta.tolist(), "actualT": temperature.tolist(), "tau": .3,
            "case": "theta+0.05_T1.5", "thetaShift": .05, "TScale": 1.5,
            "valN": 2, "valP": 1, "valAp": .5, "valF1": 2 / 3,
            "scoreSha256": INITIAL.sha(inputs / TASK_ID / "scores.f32")}
        audit = {"manifestSha256": INITIAL.sha(path / "manifest.json"), "oldBaseStateFingerprint": old_base.base_state_fingerprint,
            "isolationContract": str(contract_path), "isolationContractSha256": INITIAL.sha(contract_path)}
        stage_task(self.service, "val-new", inputs, selection, record, audit)
        return old_base, INITIAL.load_task_baseline(self.service, TASK_ID, "val-new")

    def test_derived_publication_keeps_bank_and_old_identity_and_f64(self):
        import native_probe_update as native
        old, (new, meta, directory) = self.stage()
        self.assertEqual(INITIAL.active_info(self.service)["version"], "old")
        bank = native._bank_directory(self.service, TASK_ID, old)
        self.assertEqual(native._bank_directory(self.service, TASK_ID, new), bank)
        self.assertEqual(INITIAL.read(bank / "refinement_base.json")["baseStateFingerprint"], old.base_state_fingerprint)
        self.assertIsNone(INITIAL.published_bank_attestation(self.service, TASK_ID, old))
        self.assertEqual(INITIAL.published_bank_attestation(self.service, TASK_ID, new), directory / "bank_attestation.json")
        self.assertEqual(INITIAL.gate_array_encoding(old), "<f4")
        self.assertEqual(INITIAL.gate_array_encoding(new), "<f8")
        self.assertEqual(new.initial_theta.dtype, np.float64)
        INITIAL.publish(self.service, "val-new")
        desc = INITIAL.descriptor(self.service, TASK_ID)
        self.assertFalse(desc["initialModelHoldoutIndependent"])
        self.assertTrue(desc["calibrationUsedValidation"])
        self.assertEqual(desc["classificationThreshold"], .3)
        pinned = INITIAL.load_task_baseline(self.service, TASK_ID, fingerprint=old.base_state_fingerprint)[0]
        self.assertEqual(pinned.initial_baseline_identity, old.initial_baseline_identity)
        (directory / "bank_attestation.json").write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            INITIAL.published_bank_attestation(self.service, TASK_ID, new)

    def test_staged_and_joint_before_exactly_match_val_f0_and_gates_stay_fixed(self):
        _, (base, _, directory) = self.stage()
        INITIAL.publish(self.service, "val-new")
        client = ApiClient(self.base_url)
        with patch.object(self.service, "refinement_capabilities", return_value={}):
            bootstrap = self.bootstrap(client, "Val baseline")
        session = bootstrap["session"]["id"]
        status, response = client.request("PUT", f"/api/tuning/sessions/{session}/annotations/2",
            {"imageId": self.image_ids[2], "label": 1, "source": "manual"})
        self.assertEqual(status, 200, response)
        for mode in ("weight_staged", "weight_joint"):
            status, response = client.request("POST", f"/api/tuning/sessions/{session}/runs", {"mode": mode, "maxIterations": 10})
            self.assertEqual(status, 202, response)
            with self.service.connect() as connection:
                run = connection.execute("SELECT r.*,s.task_id,s.target_id FROM model_runs r JOIN sessions s ON s.id=r.session_id WHERE r.id=?",
                    (response["run"]["id"],)).fetchone()
            params = json.loads(run["params_json"])
            self.assertEqual(params["gateArrayEncoding"], "<f8")
            self.assertEqual(params["initialTheta"], base.initial_theta.tolist())
            before, after = self.service._execute_model_run(run)
            run_path = self.runtime_root / run["artifact_relpath"]
            self.assertEqual((run_path / "initial-scores.f32").read_bytes(), (directory / "scores.f32").read_bytes())
            self.assertTrue(after["splitAudit"]["calibrationUsedValidation"])
            self.assertFalse(after["splitAudit"]["initialModelHoldoutIndependent"])
            with np.load(run_path / "model.npz", allow_pickle=False) as model:
                np.testing.assert_array_equal(model["theta"], base.initial_theta)
                np.testing.assert_array_equal(model["temperature"], base.temperature)
            # This fixture calls the numerical worker directly, without its
            # queue wrapper that normally marks the job complete.
            with self.service.connect() as connection:
                connection.execute("UPDATE model_runs SET status='succeeded' WHERE id=?", (run["id"],))


if __name__ == "__main__":
    unittest.main()
