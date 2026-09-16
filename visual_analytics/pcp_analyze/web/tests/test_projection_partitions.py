from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


WEB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WEB_ROOT / "scripts" / "export_web_data.py"
SCRIPT_DIR = str(MODULE_PATH.parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location("pcp_export_web_data", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


class ProjectionPartitionTests(unittest.TestCase):
    def test_standardization_and_pca_fit_ignore_held_out_rows(self):
        rng = np.random.default_rng(17)
        ranks = rng.normal(size=(120, 6)).astype(np.float32)
        development = np.zeros(120, dtype=np.uint8)
        development[:80] = 1

        features, mean, scale = EXPORT.standardized_features(ranks, development)
        coordinates, explained = EXPORT.compute_pca(features, development)

        perturbed = ranks.copy()
        perturbed[80:] += rng.normal(500.0, 50.0, size=(40, 6)).astype(np.float32)
        changed_features, changed_mean, changed_scale = EXPORT.standardized_features(
            perturbed, development
        )
        changed_coordinates, changed_explained = EXPORT.compute_pca(
            changed_features, development
        )

        np.testing.assert_allclose(changed_mean, mean, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(changed_scale, scale, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(changed_features[:80], features[:80], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            changed_coordinates[:80], coordinates[:80], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(changed_explained, explained, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
