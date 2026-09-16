"""Independent synthetic F0 publication/pinned-run checks; no model training."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

try:
    from tests.test_tuning_backend import TUNING
except ModuleNotFoundError:
    from test_tuning_backend import TUNING
import unified_initial_baseline as INITIAL
from tuning_models import unified_weight_scores


class InitialBaselineContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.task_id = "001_synthetic_joint"
        self.web = Path(self.temp.name) / "web"
        self.images = ("image-z", "image-a", "image-c", "image-b")
        self.bundle = SimpleNamespace(image_ids=self.images)
        self.task = SimpleNamespace(task_id=self.task_id, row_count=4, target_ids=("a", "b", "joint"))
        self.service = SimpleNamespace(web_root=self.web, tasks={self.task_id: self.task},
                                       task=lambda _: self.task, bundle=lambda _: self.bundle)
        INITIAL._CACHE.clear()

    def publication(self, version, shift=0.):
        directory = INITIAL.root_path(self.service) / version / self.task_id
        directory.mkdir(parents=True)
        z = (np.linspace(.01, .85, 4 * 2 * 8).reshape(4, 2, 8) + shift).astype(np.float32)
        embeddings = np.asarray([[.2, .7], [.9, .1], [.3, .6], [.8, .4]], dtype=np.float32)
        theta = np.asarray([.3, .4], dtype=np.float32)
        temperature = np.asarray([.1, .2], dtype=np.float32)
        scores = unified_weight_scores(z, embeddings, np.full((2, 8), 1 / 8),
                                      np.ones(2), [.5, .5], .25, theta, temperature)
        columns = [scores.final_scores, scores.conjunction_scores]
        for a in range(2):
            columns.extend([scores.gates[:, a], *[z[:, a, m] for m in range(8)]])
        columns.extend([scores.holistic_scores, embeddings[:, 0], embeddings[:, 1]])
        matrix = np.asarray(np.column_stack(columns), dtype="<f4")
        ranks = np.column_stack([TUNING.normalized_ranks(matrix[:, col]) for col in range(matrix.shape[1])])
        np.savez(directory / "base.npz", probeFeatures=z, rawProbeProbabilities=z,
                 probeMin=np.zeros((2, 8)), probeMax=np.ones((2, 8)), theta=theta, temperature=temperature,
                 embeddingFeatures=embeddings, embeddingMin=np.zeros(2), embeddingMax=np.ones(2),
                 developmentIndices=np.asarray([0, 2], dtype=np.int64))
        (directory / "scores.f32").write_bytes(matrix[:, 0].astype("<f4").tobytes())
        (directory / "ranks.f32").write_bytes(ranks[:, 0].astype("<f4").tobytes())
        (directory / "visualization.f32").write_bytes(np.concatenate([matrix.reshape(-1), ranks.reshape(-1)]).astype("<f4").tobytes())
        INITIAL.write(directory / "calibration.json", {"usesValLabels": False, "usesTestLabels": False})
        normalization = {"protocol": TUNING.REFINEMENT_NORMALIZATION_POLICY,
                         "vqaValidationVersion": "val-original", "vqaValidationManifestSha256": "fixed-val-hash"}
        metadata = {"protocol": INITIAL.PROTOCOL, "version": version, "verified": True,
                    "taskId": self.task_id, "rowCount": 4, "componentCount": matrix.shape[1],
                    "imageIdsSha256": INITIAL.digest(list(self.images)),
                    "baseStateFingerprint": hashlib.sha256(version.encode()).hexdigest(),
                    "attributeIds": ["a", "b"], "attributeNames": ["A", "B"],
                    "learnerMethods": list(TUNING.WEIGHTED_FUSION_LEARNERS),
                    "probeMethodIds": list(INITIAL.METHODS),
                    "embeddingMethods": list(TUNING.REFINEMENT_EMBEDDING_METHODS),
                    "normalizationContract": normalization, "bankDirectory": "a" * 64, "bankFingerprint": "b" * 64,
                    "files": {path.name: INITIAL.sha(path) for path in directory.iterdir()}}
        INITIAL.write(directory / "manifest.json", metadata)
        return INITIAL.load_task_baseline(self.service, self.task_id, version)

    def activate(self, version):
        root = INITIAL.root_path(self.service)
        manifest = root / version / "manifest.json"
        metadata = INITIAL.read(root / version / self.task_id / "manifest.json")
        INITIAL.write(manifest, {"protocol": INITIAL.PROTOCOL, "version": version, "complete": True,
                      "tasks": {self.task_id: {"manifestSha256": INITIAL.sha(root / version / self.task_id / "manifest.json"),
                                               "baseStateFingerprint": metadata["baseStateFingerprint"]}}})
        INITIAL.write(root / "active.json", {"version": version, "manifestSha256": INITIAL.sha(manifest)})

    def test_all_four_mode_before_states_reproduce_published_f0_exactly(self):
        base, metadata, directory = self.publication("first")
        expected = (directory / "scores.f32").read_bytes()
        self.service._refinement_embedding_methods = lambda *_: TUNING.REFINEMENT_EMBEDDING_METHODS
        self.service._refinement_base_context = Mock(side_effect=AssertionError("Must use pinned publication"))
        for mode in TUNING.REFINEMENT_RUN_MODES:
            with self.subTest(mode=mode):
                params = {"algorithmVersion": TUNING.RUN_ALGORITHM_VERSIONS[mode],
                          "vqaValidationVersion": "val-original", "vqaValidationManifestSha256": "fixed-val-hash",
                          "normalizationPolicy": TUNING.REFINEMENT_NORMALIZATION_POLICY,
                          "normalizationContract": base.normalization_audit,
                          "initialBaselineIdentity": base.initial_baseline_identity,
                          "baseStateFingerprint": base.base_state_fingerprint}
                actual_base = TUNING.TuningService._refinement_base_for_run(self.service, self.task_id, mode, params)
                result = unified_weight_scores(actual_base.probe_features, actual_base.embedding_features,
                                               np.full((2, 8), 1/8), np.ones(2), [.5, .5], .25,
                                               actual_base.initial_theta, actual_base.temperature)
                self.assertEqual(result.final_scores.astype("<f4").tobytes(), expected)
        self.service._refinement_base_context.assert_not_called()

    def test_switching_active_publication_does_not_change_pinned_run(self):
        first, metadata, _ = self.publication("first")
        self.activate("first")
        second, _, _ = self.publication("second", .05)
        self.activate("second")
        self.assertEqual(INITIAL.load_task_baseline(self.service, self.task_id)[0].base_state_fingerprint,
                         second.base_state_fingerprint)
        loaded = INITIAL.load_task_baseline(self.service, self.task_id, "first", fingerprint=first.base_state_fingerprint)
        np.testing.assert_array_equal(loaded[0].probe_features, first.probe_features)
        pinned_browser = INITIAL.load_task_baseline(self.service, self.task_id, fingerprint=first.base_state_fingerprint)
        np.testing.assert_array_equal(pinned_browser[0].probe_features, first.probe_features)
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            INITIAL.load_task_baseline(self.service, self.task_id, fingerprint="f" * 64)

    def test_legacy_run_explicitly_bypasses_active_publication(self):
        historical = object()
        self.service._refinement_embedding_methods = lambda *_: TUNING.EMBEDDING_BASELINE_METHODS
        self.service._refinement_base_context = Mock(return_value=historical)
        params = {"algorithmVersion": TUNING.LEGACY_WEIGHT_REFINEMENT_ALGORITHMS["weight_joint"]}
        result = TUNING.TuningService._refinement_base_for_run(self.service, self.task_id, "weight_joint", params)
        self.assertIs(result, historical)
        self.assertFalse(self.service._refinement_base_context.call_args.kwargs["prefer_published"])

    def test_gallery_identity_permutation_is_not_silently_accepted(self):
        self.publication("first")
        INITIAL._CACHE.clear()
        self.bundle.image_ids = tuple(reversed(self.images))
        with self.assertRaisesRegex(RuntimeError, "image order mismatch"):
            INITIAL.load_task_baseline(self.service, self.task_id, "first")

    def test_changed_score_bytes_fail_before_becoming_current_results(self):
        _, _, directory = self.publication("first")
        (directory / "scores.f32").write_bytes(np.zeros(4, dtype="<f4").tobytes())
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            INITIAL.load_task_baseline(self.service, self.task_id, "first")


if __name__ == "__main__":
    unittest.main()
