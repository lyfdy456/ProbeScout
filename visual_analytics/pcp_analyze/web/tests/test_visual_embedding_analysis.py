from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


WEB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WEB_ROOT / "scripts" / "visual_embedding_analysis.py"
SPEC = importlib.util.spec_from_file_location("pcp_visual_embedding_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VISUAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VISUAL
SPEC.loader.exec_module(VISUAL)

class VisualEmbeddingAnalysisTests(unittest.TestCase):
    def synthetic(self, rows: int = 240, dimensions: int = 12):
        rng = np.random.default_rng(20260816)
        values = rng.normal(size=(rows, dimensions)).astype(np.float32)
        mask = np.zeros(rows, dtype=np.uint8)
        mask[:160] = 1
        return values, mask

    def test_held_out_rows_never_change_the_development_fit(self):
        values, mask = self.synthetic()
        (
            first_pca,
            first_umap,
            first_labels,
            first_variance,
            first_fit_rows,
            first_neighbors,
            first_reduced_dimension,
        ) = VISUAL.compute_visual_embedding_analysis(values, mask, 30)
        changed = values.copy()
        changed[mask == 0] = np.random.default_rng(7).normal(
            loc=50.0,
            scale=10.0,
            size=changed[mask == 0].shape,
        )
        (
            second_pca,
            second_umap,
            second_labels,
            second_variance,
            second_fit_rows,
            second_neighbors,
            second_reduced_dimension,
        ) = VISUAL.compute_visual_embedding_analysis(changed, mask, 30)

        np.testing.assert_allclose(first_pca[mask == 1], second_pca[mask == 1])
        np.testing.assert_allclose(first_umap[mask == 1], second_umap[mask == 1])
        np.testing.assert_array_equal(first_labels[mask == 1], second_labels[mask == 1])
        np.testing.assert_allclose(first_variance, second_variance)
        self.assertEqual(first_fit_rows, 160)
        self.assertEqual(second_fit_rows, 160)
        self.assertEqual(first_pca.shape, (240, 2))
        self.assertEqual(first_umap.shape, (240, 2))
        self.assertEqual(first_labels.dtype, np.uint8)
        self.assertEqual(first_neighbors, VISUAL.UMAP_N_NEIGHBORS)
        self.assertEqual(second_neighbors, VISUAL.UMAP_N_NEIGHBORS)
        self.assertEqual(first_reduced_dimension, values.shape[1])
        self.assertEqual(second_reduced_dimension, values.shape[1])
        np.testing.assert_array_equal(np.unique(first_labels[mask == 1]), np.arange(30))

    def test_row_scaling_is_removed_by_siglip_l2_normalization(self):
        values, mask = self.synthetic()
        scales = np.linspace(0.25, 4.0, len(values), dtype=np.float32)[:, None]
        first_pca, first_umap, first_labels, _, _, _, _ = (
            VISUAL.compute_visual_embedding_analysis(values, mask, 30)
        )
        second_pca, second_umap, second_labels, _, _, _, _ = (
            VISUAL.compute_visual_embedding_analysis(values * scales, mask, 30)
        )
        np.testing.assert_allclose(first_pca, second_pca, rtol=2e-5, atol=2e-5)
        self.assertTrue(np.isfinite(first_umap).all())
        self.assertTrue(np.isfinite(second_umap).all())
        np.testing.assert_array_equal(first_labels, second_labels)

    def test_cluster_membership_does_not_use_either_2d_projection(self):
        values, mask = self.synthetic()
        pca = np.random.default_rng(98).normal(size=(len(values), 2)).astype(np.float32)
        first_labels, _ = VISUAL.compute_visual_embedding_clusters(
            values, mask, 30, pca
        )
        unrelated_coordinates = np.random.default_rng(99).normal(size=pca.shape).astype(
            np.float32
        )
        second_labels, _ = VISUAL.compute_visual_embedding_clusters(
            values, mask, 30, unrelated_coordinates
        )

        # PCA can only rename cluster IDs for deterministic display ordering;
        # pairwise membership is fixed by the original normalized embedding.
        first_membership = first_labels[:, None] == first_labels[None, :]
        second_membership = second_labels[:, None] == second_labels[None, :]
        np.testing.assert_array_equal(first_membership, second_membership)

    def test_disk_cache_returns_only_compact_pca_and_requested_labels(self):
        values, mask = self.synthetic(rows=210)
        mask[:] = 0
        mask[:150] = 1
        image_ids = tuple(f"image-{index:04d}.jpg" for index in range(len(values)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding_path = root / "siglip_embedding.npy"
            np.save(embedding_path, values)
            arguments = {
                "cache_root": root / "cache",
                "task_id": "001_toy_task_visual",
                "embedding_path": embedding_path,
                "development_mask": mask.tobytes(),
                "expected_image_ids": image_ids,
                "source_image_ids": image_ids,
                "cluster_count": 30,
            }
            projection_calls = []
            original_projection = VISUAL.compute_visual_embedding_projection

            def counted_projection(*args, **kwargs):
                projection_calls.append(True)
                return original_projection(*args, **kwargs)

            with mock.patch.object(
                VISUAL,
                "compute_visual_embedding_projection",
                side_effect=counted_projection,
            ):
                first = VISUAL.load_or_build_visual_embedding_analysis(**arguments)
                second = VISUAL.load_or_build_visual_embedding_analysis(**arguments)
                k50_arguments = {**arguments, "cluster_count": 50}
                third = VISUAL.load_or_build_visual_embedding_analysis(**k50_arguments)
                fourth = VISUAL.load_or_build_visual_embedding_analysis(**k50_arguments)

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertFalse(third.cache_hit)
            self.assertTrue(fourth.cache_hit)
            self.assertEqual(len(projection_calls), 1)
            self.assertEqual(first.payload, second.payload)
            self.assertEqual(third.payload, fourth.payload)
            self.assertEqual(first.pca_byte_length, len(values) * 2 * 4)
            self.assertEqual(first.umap_byte_length, len(values) * 2 * 4)
            self.assertEqual(len(first.payload), len(values) * (4 * 4 + 1))
            self.assertEqual(first.embedding_dimension, values.shape[1])
            self.assertEqual(first.fit_row_count, 150)
            self.assertEqual(first.cluster_count, 30)
            self.assertEqual(first.umap_input_dimension, values.shape[1])
            self.assertEqual(third.cluster_count, 50)
            metadata = json.loads(
                next((root / "cache" / VISUAL.ALGORITHM_VERSION).glob("*/metadata.json"))
                .read_text(encoding="utf-8")
            )
            self.assertIn("umap", metadata)
            self.assertEqual(metadata["embeddingSha256"], VISUAL._sha256_file(embedding_path))
            self.assertEqual(metadata["umap"]["input"], "development-fitted-pca")
            self.assertEqual(metadata["umap"]["inputDimension"], values.shape[1])
            self.assertEqual(set(metadata["clusters"]), {"30", "50"})
            self.assertIn("UMAP.transform", metadata["assignmentScope"])

    def test_rejects_unsupported_k_and_row_order_drift(self):
        values, mask = self.synthetic()
        with self.assertRaisesRegex(ValueError, "clusters must be one of"):
            VISUAL.compute_visual_embedding_analysis(values, mask, 40)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding_path = root / "siglip_embedding.npy"
            np.save(embedding_path, values)
            image_ids = tuple(f"image-{index}.jpg" for index in range(len(values)))
            reversed_ids = tuple(reversed(image_ids))
            with self.assertRaisesRegex(RuntimeError, "row order"):
                VISUAL.load_or_build_visual_embedding_analysis(
                    cache_root=root / "cache",
                    task_id="001_toy_task_visual",
                    embedding_path=embedding_path,
                    development_mask=mask.tobytes(),
                    expected_image_ids=image_ids,
                    source_image_ids=reversed_ids,
                    cluster_count=30,
                )

if __name__ == "__main__":
    unittest.main()
