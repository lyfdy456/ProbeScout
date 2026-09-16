"""Exact deployment/replay precision, using temporary synthetic artifacts only."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tests import test_tuning_backend as backend
import tuning_models


FULL_LAMBDA = 0.9490104663989628
MISSING = object()


class RefinementPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = backend.TuningApiTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)
        self.service = self.fixture.service

    def _synthetic_run(self, *, stored_dtype=np.float32, full_lambda=FULL_LAMBDA, summary=MISSING):
        client = backend.ApiClient(self.fixture.base_url)
        bootstrap = self.fixture.bootstrap(client, "Precision fixture")
        probe_signal = np.asarray([
            0.28785455226898193, 0.7235607504844666,
            0.02, 0.10, 0.20, 0.40, 0.60, 0.80, 0.99,
        ], dtype=np.float32)
        embedding_signal = np.asarray([
            0.12566953897476196, 0.007493264973163605,
            0.01, 0.10, 0.20, 0.40, 0.60, 0.80, 0.99,
        ], dtype=np.float32)
        base = SimpleNamespace(
            probe_features=np.repeat(probe_signal[:, None, None], 8, axis=2),
            embedding_features=np.repeat(embedding_signal[:, None], 2, axis=1),
            attribute_ids=("first",),
            learner_names=backend.TUNING.WEIGHTED_FUSION_LEARNERS,
            embedding_names=backend.TUNING.REFINEMENT_EMBEDDING_METHODS,
            base_state_fingerprint="precision-fixture-base",
            development_indices=np.arange(4),
        )
        self.service._refinement_base_for_run = lambda *args, **kwargs: base
        beta = np.full((1, 8), 1 / 8, dtype=np.float32)
        gamma = np.ones(1, dtype=np.float32)
        eta = np.full(2, 0.5, dtype=np.float32)
        theta = np.asarray([0.5], dtype=np.float32)
        temperature = np.asarray([0.2], dtype=np.float32)

        def score(strength):
            return tuning_models.unified_weight_scores(
                base.probe_features, base.embedding_features,
                beta, gamma, eta, strength, theta, temperature,
            ).final_scores

        scores = score(full_lambda)
        ranks = backend.TUNING.normalized_ranks(scores)
        rounded_ranks = backend.TUNING.normalized_ranks(score(float(np.float32(full_lambda))))
        directory = self.fixture.runtime_root / "precision-artifact"
        directory.mkdir()
        np.savez(
            directory / "model.npz", beta=beta, gamma=gamma,
            embeddingWeights=eta, embeddingFusionStrength=stored_dtype(full_lambda),
            theta=theta, temperature=temperature,
            attributeIds=np.asarray(base.attribute_ids),
            learnerMethods=np.asarray(base.learner_names),
            embeddingMethods=np.asarray(base.embedding_names),
            baseStateFingerprint=np.asarray([base.base_state_fingerprint]),
        )
        (directory / "tuned-scores.f32").write_bytes(scores.astype("<f4").tobytes())
        (directory / "tuned-ranks.f32").write_bytes(ranks.astype("<f4").tobytes())
        after = {} if summary is MISSING else {"modelSummary": {"embeddingFusionStrength": summary}}
        run = {
            "id": "run_precision_fixture", "session_id": bootstrap["session"]["id"],
            "user_id": bootstrap["user"]["id"], "mode": "weight_staged",
            "status": "succeeded", "base_method": "Ours-Full",
            "artifact_relpath": directory.relative_to(self.fixture.runtime_root).as_posix(),
            "after_json": json.dumps(after),
            "params_json": json.dumps({"gammaMax": 3.0, "gammaMin": 0.05}),
        }
        return run, directory, scores, ranks, rounded_ranks

    def _assert_exact(self, run, scores, ranks):
        result = self.service._compute_refinement_visualization(run)
        decoded = np.frombuffer(result.payload, dtype="<f4").reshape(2, 9, result.component_count)
        np.testing.assert_array_equal(decoded[0, :, 0], scores)
        np.testing.assert_array_equal(decoded[1, :, 0], ranks)
        return result

    def test_worker_preserves_precision_for_original_and_updated_both_schedules(self):
        fit = tuning_models.fit_unified_weight_refinement

        def force_non_float32_lambda(*args, **kwargs):
            return replace(fit(*args, **kwargs), embedding_fusion_strength=FULL_LAMBDA)

        with mock.patch.object(tuning_models, "fit_unified_weight_refinement", side_effect=force_non_float32_lambda):
            for native in (False, True):
                self.fixture._check_weight_refinement_modes(native=native)
        with self.service.connect() as connection:
            runs = connection.execute("SELECT * FROM model_runs").fetchall()
        self.assertEqual(len(runs), 4)
        for run in runs:
            with self.subTest(mode=run["mode"], source=json.loads(run["params_json"])["probeSource"]):
                path = self.fixture.runtime_root / run["artifact_relpath"] / "model.npz"
                with np.load(path, allow_pickle=False) as artifact:
                    self.assertEqual(artifact["embeddingFusionStrength"].dtype, np.dtype("float64"))
                    self.assertEqual(float(artifact["embeddingFusionStrength"]), FULL_LAMBDA)
                    self.assertEqual(artifact["theta"].dtype, np.dtype("float64"))
                    self.assertEqual(artifact["temperature"].dtype, np.dtype("float64"))

    def test_new_float64_artifact_round_trips_near_neighbor_ranks(self):
        run, _, scores, ranks, rounded_ranks = self._synthetic_run(stored_dtype=np.float64)
        self.assertFalse(np.array_equal(ranks, rounded_ranks))
        self._assert_exact(run, scores, ranks)
        # New full-precision artifacts never need to substitute the summary.
        run["after_json"] = json.dumps({"modelSummary": {"embeddingFusionStrength": "bad"}})
        self._assert_exact(run, scores, ranks)

    def test_old_float32_artifact_recovers_same_run_summary_without_writes(self):
        run, directory, scores, ranks, rounded_ranks = self._synthetic_run(summary=FULL_LAMBDA)
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}
        self.assertFalse(np.array_equal(ranks, rounded_ranks))
        self._assert_exact(run, scores, ranks)
        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}
        self.assertEqual(after, before)

    def test_missing_summary_cannot_hide_a_real_rank_mismatch(self):
        run, _, _, _, _ = self._synthetic_run()
        with self.assertRaisesRegex(RuntimeError, "Joint column does not match"):
            self.service._compute_refinement_visualization(run)

    def test_missing_summary_preserves_compatible_old_replay(self):
        run, _, scores, ranks, _ = self._synthetic_run(full_lambda=0.25)
        self._assert_exact(run, scores, ranks)

    def test_invalid_or_nonmatching_summary_is_rejected(self):
        run, _, _, _, _ = self._synthetic_run(summary=FULL_LAMBDA)
        for value in (None, True, "0.9490104663989628", float("nan"), float("inf"), -0.1, 1.1, 10 ** 100, 0.25):
            with self.subTest(value=value):
                run["after_json"] = json.dumps({"modelSummary": {"embeddingFusionStrength": value}})
                with self.assertRaisesRegex(RuntimeError, "summary lambda does not match"):
                    self.service._compute_refinement_visualization(run)
        for value in ("{", "null", '{"modelSummary":[]}'):
            with self.subTest(value=value):
                run["after_json"] = value
                with self.assertRaisesRegex(RuntimeError, "summary is malformed"):
                    self.service._compute_refinement_visualization(run)

    def test_recovered_summary_does_not_override_corrupted_saved_ranks(self):
        run, directory, _, ranks, _ = self._synthetic_run(summary=FULL_LAMBDA)
        corrupted = ranks.copy()
        corrupted[0] = 0.0
        (directory / "tuned-ranks.f32").write_bytes(corrupted.astype("<f4").tobytes())
        with self.assertRaisesRegex(RuntimeError, "Joint column does not match"):
            self.service._compute_refinement_visualization(run)

    def test_recovered_lambda_participates_in_source_fingerprint(self):
        run, _, scores, ranks, _ = self._synthetic_run(summary=FULL_LAMBDA)
        original = self._assert_exact(run, scores, ranks)
        adjacent = float(np.nextafter(FULL_LAMBDA, 1.0))
        self.assertEqual(np.float32(adjacent), np.float32(FULL_LAMBDA))
        run["after_json"] = json.dumps({"modelSummary": {"embeddingFusionStrength": adjacent}})
        changed = self._assert_exact(run, scores, ranks)
        self.assertNotEqual(original.source_fingerprint, changed.source_fingerprint)


if __name__ == "__main__":
    unittest.main()
