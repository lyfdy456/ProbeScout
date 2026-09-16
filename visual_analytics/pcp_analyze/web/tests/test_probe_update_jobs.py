"""Independent Probe jobs use only synthetic probabilities and a temporary DB."""
from __future__ import annotations

import dataclasses
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests import test_tuning_backend as backend
import native_probe_update as native

TUNING = backend.TUNING


@contextmanager
def probe_job_fixture():
    fixture = backend.TuningApiTests()
    fixture.setUp()
    patches = []
    try:
        client = backend.ApiClient(fixture.base_url)
        boot = fixture.bootstrap(client, "Probe job test")
        session_id, user_id = boot["session"]["id"], boot["user"]["id"]
        status, payload = client.request("PUT", f"/api/tuning/sessions/{session_id}/annotations/2",
                                        {"imageId": fixture.image_ids[2], "label": 1, "source": "manual"})
        fixture.assertEqual(status, 200, payload)
        z = np.repeat(np.linspace(.05, .95, 9, dtype=np.float32)[:, None, None], 8, axis=2)
        base = TUNING.RefinementBaseContext(
            attribute_ids=("first",), attribute_names=("First",), learner_names=TUNING.WEIGHTED_FUSION_LEARNERS,
            embedding_names=TUNING.REFINEMENT_EMBEDDING_METHODS, probe_features=z,
            probe_min=np.zeros((1, 8), dtype=np.float32), probe_max=np.ones((1, 8), dtype=np.float32),
            temperature=np.asarray([.2], dtype=np.float32), initial_theta=np.asarray([.5], dtype=np.float32),
            embedding_features=np.repeat(z[:, 0, :1], 2, axis=1), embedding_min=np.zeros(2), embedding_max=np.ones(2),
            development_indices=np.asarray([0, 1, 2, 3]), base_state_fingerprint="synthetic-base",
            normalization_audit=fixture.service._refinement_normalization_scope(backend.TASK_ID)[1],
            raw_probe_probabilities=z.copy(),
        )
        fixture.service._refinement_base_context = mock.Mock(return_value=base)
        calls = []
        def prepare(service, user, session, task, selected_base, annotations, params):
            request = {
                "protocol": native.PROTOCOL, "userId": user, "sessionId": session, "taskId": task,
                "baseBankFingerprint": "synthetic-bank", "baseStateFingerprint": selected_base.base_state_fingerprint,
                "normalizationContract": selected_base.normalization_audit, "attributeIds": ["first"],
                "codeFingerprint": "synthetic-code", "config": params["nativeUpdateConfig"],
                "feedbackWeight": params["feedbackWeight"], "annotations": annotations,
                "examples": [{"attributeId": "first", "update": True}],
            }
            request["snapshotId"] = native.digest(request)
            return request
        def train(request, bank, output):
            calls.append(request["snapshotId"])
            return np.repeat((1 - z)[None], 5, axis=0), {"synthetic": True}
        fixture.service._run_native_probe_update = mock.Mock(side_effect=train)
        for patch in (
            mock.patch.object(native, "prepare_request", side_effect=prepare),
            mock.patch.object(native, "current_code_fingerprint", return_value="synthetic-code"),
            mock.patch.object(native, "resolve_bank", return_value={"fingerprint": "synthetic-bank"}),
            mock.patch.object(native, "capability", return_value={"available": True}),
        ):
            patch.start()
            patches.append(patch)
        yield SimpleNamespace(fixture=fixture, service=fixture.service, client=client, session_id=session_id,
                              user_id=user_id, base=base, calls=calls)
    finally:
        for patch in reversed(patches):
            patch.stop()
        fixture.tearDown()
        fixture.doCleanups()


class ProbeUpdateJobTests(unittest.TestCase):
    def test_standalone_update_deduplicates_and_never_trains_weights(self):
        with probe_job_fixture() as env, mock.patch("tuning_models.fit_unified_weight_refinement", side_effect=AssertionError("weight fit forbidden")):
            status, body = env.client.request("POST", f"/api/tuning/sessions/{env.session_id}/probe-updates", {})
            self.assertEqual(status, 202, body)
            update = body["probeUpdate"]
            self.assertEqual(update["status"], "queued")
            again = env.service.create_probe_update(env.user_id, env.session_id, {})
            self.assertEqual(again["id"], update["id"])
            env.service._run_probe_update_job(update["id"])
            status, body = env.client.request("GET", f"/api/tuning/probe-updates/{update['id']}")
            self.assertEqual(status, 200)
            self.assertEqual(body["probeUpdate"]["status"], "succeeded")
            self.assertTrue(body["probeUpdate"]["compatible"])
            self.assertFalse(body["probeUpdate"]["stale"])
            self.assertEqual(env.service.create_probe_update(env.user_id, env.session_id, {})["id"], update["id"])
            self.assertEqual(len(env.calls), 1)
            with env.service.connect() as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM model_runs").fetchone()[0], 0)
            refreshed = env.fixture.bootstrap(env.client, "Probe job test")
            self.assertEqual(refreshed["probeUpdates"][0]["id"], update["id"])
            self.assertFalse(any(env.fixture.runtime_root.rglob("tuned-scores.f32")))

    def test_atomic_duplicate_submission_and_cross_kind_guard(self):
        with probe_job_fixture() as env:
            results, errors = [], []
            def submit():
                try:
                    results.append(env.service.create_probe_update(env.user_id, env.session_id, {}))
                except Exception as error:
                    errors.append(error)
            threads = [threading.Thread(target=submit) for _ in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertFalse(errors)
            self.assertEqual(results[0]["id"], results[1]["id"])
            with self.assertRaisesRegex(TUNING.ApiError, "queued or running"):
                env.service.create_run(env.user_id, env.session_id, {"mode": "weight_joint"})
            env.service._run_probe_update_job(results[0]["id"])
            env.service.create_run(env.user_id, env.session_id, {"mode": "weight_joint"})
            with self.assertRaisesRegex(TUNING.ApiError, "queued or running"):
                env.service.create_probe_update(env.user_id, env.session_id, {"feedbackWeight": 9})

    def test_later_feedback_marks_stale_but_snapshot_is_load_only(self):
        with probe_job_fixture() as env:
            update = env.service.create_probe_update(env.user_id, env.session_id, {})
            env.service._run_probe_update_job(update["id"])
            status, body = env.client.request("PUT", f"/api/tuning/sessions/{env.session_id}/annotations/1",
                {"imageId": env.fixture.image_ids[1], "label": 1, "source": "manual"})
            self.assertEqual(status, 200, body)
            self.assertTrue(env.service.list_probe_updates(env.user_id, env.session_id)[0]["stale"])
            with mock.patch.object(native, "prepare_request", side_effect=AssertionError("must not re-prepare")), mock.patch.object(native, "ensure_snapshot", side_effect=AssertionError("must not train")):
                run = env.service.create_run(env.user_id, env.session_id,
                    {"mode": "weight_staged", "probeSource": "updated", "probeUpdateId": update["id"], "maxIterations": 10})
                self.assertEqual(run["probeUpdateId"], update["id"])
                self.assertEqual(run["probeSource"], "updated")
                self.assertEqual(run["annotationCount"], 2)
            self.assertEqual(len(env.calls), 1)

    def test_fail_closed_missing_foreign_incomplete_or_changed_snapshot(self):
        with probe_job_fixture() as env:
            for payload in ({"mode": "probe_joint"}, {"mode": "weight_joint", "probeSource": "updated"},
                            {"mode": "weight_joint", "probeSource": "invalid"},
                            {"mode": "weight_joint", "probeSource": {}},
                            {"mode": "weight_joint", "probeSource": "original", "probeUpdateId": "fake"}):
                with self.assertRaises(TUNING.ApiError):
                    env.service.create_run(env.user_id, env.session_id, payload)
            update = env.service.create_probe_update(env.user_id, env.session_id, {})
            with self.assertRaises(TUNING.ApiError):
                env.service._load_selected_probe_update(env.user_id, env.session_id, backend.TASK_ID, update["id"], env.base)
            env.service._run_probe_update_job(update["id"])
            other = backend.ApiClient(env.fixture.base_url)
            other_boot = env.fixture.bootstrap(other, "Other user")
            status, _ = other.request("GET", f"/api/tuning/probe-updates/{update['id']}")
            self.assertEqual(status, 404)
            with self.assertRaises(TUNING.ApiError):
                env.service._load_selected_probe_update(other_boot["user"]["id"], other_boot["session"]["id"], backend.TASK_ID, update["id"], env.base)
            changed = dataclasses.replace(env.base, base_state_fingerprint="changed-base")
            env.service._refinement_base_context.return_value = changed
            self.assertFalse(env.service.list_probe_updates(env.user_id, env.session_id)[0]["compatible"])
            with self.assertRaisesRegex(RuntimeError, "base contract"):
                env.service._load_selected_probe_update(env.user_id, env.session_id, backend.TASK_ID, update["id"], changed)

    def test_failure_retry_and_recovery_keep_frozen_inputs(self):
        with probe_job_fixture() as env:
            update = env.service.create_probe_update(env.user_id, env.session_id, {})
            original = env.service._run_native_probe_update
            with mock.patch.object(env.service, "_run_native_probe_update", side_effect=RuntimeError("synthetic failure")):
                env.service._run_probe_update_job(update["id"])
            self.assertEqual(env.service.owned_probe_update(update["id"], env.user_id)["status"], "failed")
            retry = env.service.create_probe_update(env.user_id, env.session_id, {})
            self.assertEqual(retry["id"], update["id"])
            with env.service.connect() as connection:
                connection.execute("UPDATE probe_updates SET status='running' WHERE id=?", (update["id"],))
            env.service._initialize_database()
            self.assertEqual(env.service.owned_probe_update(update["id"], env.user_id)["status"], "queued")
            env.service._run_probe_update_job(update["id"])
            self.assertEqual(env.service.owned_probe_update(update["id"], env.user_id)["status"], "succeeded")
            self.assertEqual(original.call_count, 1)

    def test_fresh_training_rejects_changed_code_but_completed_snapshot_remains_readable(self):
        with probe_job_fixture() as env:
            update = env.service.create_probe_update(env.user_id, env.session_id, {})
            with mock.patch.object(native, "current_code_fingerprint", return_value="changed-code"):
                env.service._run_probe_update_job(update["id"])
            row = env.service.owned_probe_update(update["id"], env.user_id)
            self.assertEqual(row["status"], "failed")
            self.assertIn("code changed", row["error"])
            self.assertEqual(env.calls, [])
            env.service.create_probe_update(env.user_id, env.session_id, {})
            env.service._run_probe_update_job(update["id"])
            with mock.patch.object(native, "current_code_fingerprint", side_effect=AssertionError("no code check for frozen loading")):
                env.service._load_selected_probe_update(env.user_id, env.session_id, backend.TASK_ID, update["id"], env.base)

    def test_request_and_published_probability_tampering_are_rejected(self):
        with probe_job_fixture() as env:
            update = env.service.create_probe_update(env.user_id, env.session_id, {})
            env.service._run_probe_update_job(update["id"])
            path = env.fixture.runtime_root / "native-probe-updates" / update["snapshotId"] / "probabilities.npy"
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                env.service._load_selected_probe_update(env.user_id, env.session_id, backend.TASK_ID, update["id"], env.base)
            with env.service.connect() as connection:
                connection.execute("UPDATE probe_updates SET request_json='{}' WHERE id=?", (update["id"],))
            with self.assertRaisesRegex(RuntimeError, "request checksum"):
                env.service.probe_update_json(env.service.owned_probe_update(update["id"], env.user_id))


if __name__ == "__main__":
    unittest.main()
