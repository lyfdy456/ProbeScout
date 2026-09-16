from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


WEB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WEB_ROOT / "scripts" / "add_fine_clusters.py"
SPEC = importlib.util.spec_from_file_location("pcp_fine_clusters", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
FINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FINE
SPEC.loader.exec_module(FINE)


class FineClusterTests(unittest.TestCase):
    def test_supported_cluster_counts_require_the_complete_set(self):
        self.assertEqual(FINE.parse_cluster_counts("100,30,50"), (30, 50, 100))
        for invalid in ("", "30", "30,50", "30,50,100,100", "bad"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FINE.parse_cluster_counts(invalid)

    def test_scheme_and_file_names_are_stable(self):
        self.assertEqual(
            [FINE.fine_scheme(value) for value in FINE.FINE_CLUSTER_COUNTS],
            ["fine30", "fine50", "fine100"],
        )
        self.assertEqual(FINE.fine_file_key(100), "fine100Labels")
        self.assertEqual(FINE.fine_labels_name(100), "fine100-labels.u8")

    def test_minibatch_fit_is_deterministic_for_a_fixed_seed(self):
        rng = np.random.default_rng(7)
        raw = rng.uniform(0.0, 1.0, size=(240, 5)).astype(np.float32)
        features = (raw - raw.mean(axis=0)) / raw.std(axis=0)
        first = FINE.fit_labels(raw, features, clusters=6, seed=42)
        second = FINE.fit_labels(raw, features, clusters=6, seed=42)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(first[2], second[2])
        self.assertEqual(first[3], second[3])

    def test_minibatch_fit_uses_only_selected_fit_rows_and_predicts_all_rows(self):
        rng = np.random.default_rng(11)
        raw = rng.uniform(0.0, 1.0, size=(240, 5)).astype(np.float32)
        features = (raw - raw.mean(axis=0)) / raw.std(axis=0)
        fit_indices = np.arange(0, 180, dtype=np.int64)
        first = FINE.fit_labels(
            raw, features, clusters=6, seed=42, fit_indices=fit_indices
        )
        perturbed = features.copy()
        perturbed[180:] += 20
        second = FINE.fit_labels(
            raw, perturbed, clusters=6, seed=42, fit_indices=fit_indices
        )

        self.assertEqual(first[0].shape, (240,))
        self.assertEqual(second[0].shape, (240,))
        # Held-out perturbations may change predictions, but cannot affect the
        # fitted MiniBatchKMeans inertia or fitted-row assignments.
        self.assertEqual(first[3], second[3])
        np.testing.assert_array_equal(first[0][:180], second[0][:180])

    def test_staged_label_write_does_not_replace_the_final_file_early(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "fine30-labels.u8"
            final.write_bytes(b"old")
            staged = FINE.stage_bytes(final, np.asarray([0, 1, 2], dtype=np.uint8).tobytes())
            self.assertEqual(final.read_bytes(), b"old")
            self.assertEqual(staged.read_bytes(), b"\x00\x01\x02")


if __name__ == "__main__":
    unittest.main()
