"""Bounded synthetic publication tests; no training or real checkpoint inference."""
from __future__ import annotations

import dataclasses
import json
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from . import test_tuning_backend as fixtures
except ImportError:
    import test_tuning_backend as fixtures
TUNING, ApiClient, TASK_ID, clean_validation_split = fixtures.TUNING, fixtures.ApiClient, fixtures.TASK_ID, fixtures.clean_validation_split
import unified_initial_baseline as INITIAL
from tuning_models import unified_weight_scores


class InitialBaselineTests(unittest.TestCase):
    setUp = fixtures.TuningApiTests.setUp
    tearDown = fixtures.TuningApiTests.tearDown
    bootstrap = fixtures.TuningApiTests.bootstrap

    def build(self, version="v1", *, reverse=False):
        split = clean_validation_split(fit_indices=[0, 1, 2, 3], fit_labels=[0, 0, 1, 1],
            validation_indices=[5, 6], validation_labels=[1, 0], fingerprint="initial-val")
        self.service._clean_validation_split = lambda task, target: split
        rows, audit = self.service._refinement_normalization_scope(TASK_ID)
        signal = np.linspace(.05, .95, 9, dtype=np.float32)
        if reverse:
            signal = signal[::-1].copy()
        self.service._exported_method_column = lambda task, key, method, target: signal.copy()
        raw = np.repeat(np.repeat(signal[None, :, None, None], 5, 0), 8, 3)
        contract = {"attributes": [{"id": "first", "name": "first"}], "valVersion": audit["vqaValidationVersion"],
            "normalizationRows": list(map(int, rows)), "targets": {"first": {"fitRows": [0, 1, 2, 3], "fitLabels": [0, 0, 1, 1]},
                                                                   "joint": {"fitRows": [0, 1, 2, 3], "fitLabels": [0, 0, 1, 1]}}}
        training = {"trainerSha256": "same-code", "epochs": 100, "seeds": list(INITIAL.SEEDS), "methods": list(INITIAL.METHODS)}
        proof = {"forwardVerified": True, "scoreImageIds": self.image_ids, "bankTrainingIdentity": training,
                 "forwardScope": "full-gallery", "samplingProtocol": "synthetic", "sampleImageIds": self.image_ids}
        bank = self.web_root.parent / "runtime" / "isolated-probes" / TASK_ID / (("b" if reverse else "a") * 64)
        bank.mkdir(parents=True, exist_ok=True)
        with patch.object(INITIAL, "verify_bank", return_value=(raw, contract, {"model.pt": "model-sha"}, proof)), \
             patch.object(INITIAL, "calibrate", return_value=(np.asarray([.44], dtype=np.float32), np.asarray([.075], dtype=np.float32), {"protocol": INITIAL.CALIBRATION})):
            result = INITIAL.build_task_baseline(self.service, TASK_ID, version, bank)
        return result

    def test_staging_is_invisible_then_atomic_publication_exposes_exact_f0(self):
        self.assertFalse(INITIAL.descriptor(self.service, TASK_ID)["available"])
        base, metadata, directory = self.build()
        self.assertFalse(INITIAL.descriptor(self.service, TASK_ID)["available"])
        INITIAL.publish(self.service, "v1")
        descriptor = INITIAL.descriptor(self.service, TASK_ID)
        self.assertTrue(descriptor["available"])
        self.assertEqual(descriptor["version"], INITIAL.PROTOCOL)
        self.assertEqual(descriptor["learnerMethods"], list(TUNING.WEIGHTED_FUSION_LEARNERS))
        output = unified_weight_scores(base.probe_features, base.embedding_features, np.full((1, 8), 1/8),
            np.ones(1), np.full(2, .5), .25, base.initial_theta, base.temperature)
        np.testing.assert_array_equal(np.fromfile(directory / "scores.f32", dtype="<f4"), output.final_scores.astype(np.float32))
        view = INITIAL.visualization(self.service, TASK_ID, base.base_state_fingerprint)
        decoded = np.frombuffer(view.payload, dtype="<f4").reshape(2, 9, 14)
        np.testing.assert_array_equal(decoded[0, :, 2], output.gates[:, 0].astype(np.float32))
        np.testing.assert_array_equal(decoded[0, :, 3:11], base.probe_features[:, 0])
        self.assertEqual(view.source_fingerprint, descriptor["sourceFingerprint"])
        self.assertEqual(self.service._refinement_base_context(TASK_ID).base_state_fingerprint, base.base_state_fingerprint)
        client = ApiClient(self.base_url)
        status, http_manifest = client.request("GET", f"/api/tuning/tasks/{TASK_ID}/initial-baseline")
        self.assertEqual(status, 200, http_manifest)
        self.assertEqual(http_manifest["fingerprint"], base.base_state_fingerprint)
        with urllib.request.urlopen(f"{self.base_url}/api/tuning/tasks/{TASK_ID}/initial-baseline/visualization?fingerprint={base.base_state_fingerprint}") as response:
            self.assertEqual(response.headers["X-PCP-Baseline-Fingerprint"], base.base_state_fingerprint)
            self.assertEqual(response.headers["X-PCP-Source-Fingerprint"], descriptor["sourceFingerprint"])
            self.assertEqual(response.read(), view.payload)

    def test_publication_refuses_missing_task_without_changing_active(self):
        self.build()
        self.service.tasks["missing-task"] = self.service.tasks[TASK_ID]
        with self.assertRaisesRegex(RuntimeError, "pending"):
            INITIAL.publish(self.service, "v1")
        self.assertIsNone(INITIAL.active_info(self.service))

    def test_artifact_corruption_is_not_treated_as_pending(self):
        _, _, directory = self.build()
        (directory / "scores.f32").write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            INITIAL.load_task_baseline(self.service, TASK_ID, "v1")

    def test_same_raw_values_use_development_only_minmax(self):
        raw = np.asarray([[0.1], [0.5], [0.9], [5000.0]], dtype=np.float32)
        z, minimum, maximum = INITIAL.minmax(raw, np.asarray([0, 1]))
        self.assertAlmostEqual(float(minimum[0]), .1)
        self.assertEqual(float(maximum[0]), .5)
        np.testing.assert_array_equal(z, [[0], [1], [1], [1]])

    def test_pinned_old_publication_survives_new_active_and_native_bank_resolution(self):
        import native_probe_update as native
        old = self.build("v1")[0]
        INITIAL.publish(self.service, "v1")
        new = self.build("v2", reverse=True)[0]
        INITIAL.publish(self.service, "v2")
        self.assertNotEqual(old.base_state_fingerprint, new.base_state_fingerprint)
        loaded = INITIAL.load_task_baseline(self.service, TASK_ID, fingerprint=old.base_state_fingerprint)[0]
        self.assertEqual(loaded.initial_baseline_identity["publicationVersion"], "v1")
        self.assertEqual(native._bank_directory(self.service, TASK_ID, old).name, "a" * 64)
        self.assertEqual(native._bank_directory(self.service, TASK_ID, new).name, "b" * 64)

    def test_old_run_does_not_resolve_through_published_phi0(self):
        self.build()
        INITIAL.publish(self.service, "v1")
        base = self.service._refinement_base_context(TASK_ID)
        legacy = dataclasses.replace(base, base_state_fingerprint="old-score-cache", initial_baseline_identity=None)
        observed = []
        self.service._refinement_base_context = lambda *args, **kwargs: (observed.append(kwargs) or legacy)
        params = {"algorithmVersion": TUNING.RUN_ALGORITHM_VERSIONS["weight_joint"],
            "embeddingMethods": list(TUNING.REFINEMENT_EMBEDDING_METHODS),
            "vqaValidationVersion": base.normalization_audit["vqaValidationVersion"],
            "vqaValidationManifestSha256": base.normalization_audit["vqaValidationManifestSha256"],
            "normalizationPolicy": TUNING.REFINEMENT_NORMALIZATION_POLICY, "normalizationContract": base.normalization_audit}
        result = self.service._refinement_base_for_run(TASK_ID, "weight_joint", params)
        self.assertEqual(result.base_state_fingerprint, "old-score-cache")
        self.assertFalse(observed[-1]["prefer_published"])

    def test_new_weight_run_before_is_published_f0_without_recalibration(self):
        base, _, directory = self.build()
        INITIAL.publish(self.service, "v1")
        client = ApiClient(self.base_url)
        # Fake native readiness only: this test runs frozen-probe fusion.
        with patch.object(self.service, "refinement_capabilities", return_value={}):
            bootstrap = self.bootstrap(client, "new baseline")
        session, user = bootstrap["session"]["id"], bootstrap["user"]["id"]
        status, response = client.request("PUT", f"/api/tuning/sessions/{session}/annotations/2",
            {"imageId": self.image_ids[2], "label": 1, "source": "manual"})
        self.assertEqual(status, 200, response)
        for mode in ("weight_staged", "weight_joint"):
            status, response = client.request("POST", f"/api/tuning/sessions/{session}/runs", {"mode": mode, "maxIterations": 10})
            self.assertEqual(status, 202, response)
            run_id = response["run"]["id"]
            with self.service.connect() as connection:
                run = connection.execute("SELECT r.*,s.task_id,s.target_id FROM model_runs r JOIN sessions s ON s.id=r.session_id WHERE r.id=?", (run_id,)).fetchone()
            params = json.loads(run["params_json"])
            self.assertEqual(params["initialTheta"], base.initial_theta.astype(float).tolist())
            self.assertTrue(params["initialModelHoldoutIndependent"])
            # Complete the worker lifecycle: a raw scorer call leaves the first
            # request queued and makes the next mode correctly receive HTTP 409.
            self.service._run_job(run_id)
            with self.service.connect() as connection:
                completed = connection.execute(
                    "SELECT status,error,after_json FROM model_runs WHERE id=?", (run_id,)
                ).fetchone()
            self.assertEqual(completed["status"], "succeeded", completed["error"])
            after = json.loads(completed["after_json"])
            run_path = self.runtime_root / run["artifact_relpath"]
            self.assertEqual((run_path / "initial-scores.f32").read_bytes(), (directory / "scores.f32").read_bytes())
            self.assertTrue(after["splitAudit"]["initialModelHoldoutIndependent"])
            self.assertFalse(after["splitAudit"]["referenceOnly"])


if __name__ == "__main__":
    unittest.main()
