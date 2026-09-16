import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.methods.metric_attention_probes import (
    CosineMarginRanker,
    LearnedQueryAttentionProbe,
    TripletProjectionProbe,
    cosine_margin_loss,
    train_one_attribute_attention_pooling,
    train_one_attribute_cosine_margin_ranking,
    train_one_attribute_triplet_loss,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "probe_learning" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_probebank_batch as runner  # noqa: E402


class MetricAttentionProbeTest(unittest.TestCase):
    def setUp(self):
        self.runner_state = {
            name: copy.deepcopy(getattr(runner, name))
            for name in (
                "EMBEDDING_METHODS", "BASE_METHODS", "REPORTED_METHODS",
                "JOINT_VARIANTS", "FUSION_RECIPES", "POSTHOC_JOINT_RULES",
                "POSTHOC_BASE_METHODS", "POSTHOC_STAGE",
            )
        }
        self.old_epochs = runner.H.EPOCHS
        rng = np.random.default_rng(7)
        negative = rng.normal(-0.8, 0.35, size=(30, 8)).astype(np.float32)
        positive = rng.normal(0.8, 0.35, size=(30, 8)).astype(np.float32)
        self.X = np.vstack([negative, positive])
        self.y = np.r_[np.zeros(30, dtype=np.int64), np.ones(30, dtype=np.int64)]
        self.U = rng.normal(0.0, 0.8, size=(40, 8)).astype(np.float32)

    def tearDown(self):
        for name, value in self.runner_state.items():
            setattr(runner, name, value)
        runner.H.EPOCHS = self.old_epochs

    def test_cosine_margin_loss_obeys_explicit_margin(self):
        easy = cosine_margin_loss(
            torch.tensor([0.8, 0.9]), torch.tensor([0.1, 0.2]), margin=0.2,
        )
        violating = cosine_margin_loss(
            torch.tensor([0.2]), torch.tensor([0.3]), margin=0.2,
        )
        self.assertEqual(float(easy), 0.0)
        self.assertGreater(float(violating), 0.0)

    def test_metric_probes_train_and_return_finite_logits(self):
        with patch(
            "src.methods.metric_attention_probes._get_device",
            return_value=torch.device("cpu"),
        ):
            cosine, cosine_metrics = train_one_attribute_cosine_margin_ranking(
                "attr", self.X, self.y, self.U, input_dim=8,
                epochs=2, batch_size=16, batch_size_u=12, seed=3,
            )
            triplet, triplet_metrics = train_one_attribute_triplet_loss(
                "attr", self.X, self.y, input_dim=8,
                projection_dim=6, epochs=2, batch_size=16, seed=3,
            )
        self.assertIsInstance(cosine, CosineMarginRanker)
        self.assertIsInstance(triplet, TripletProjectionProbe)
        for model, metrics in ((cosine, cosine_metrics), (triplet, triplet_metrics)):
            logits = model(torch.as_tensor(self.X[:5])).detach().numpy()
            self.assertTrue(np.isfinite(logits).all())
            self.assertIn("val_auc", metrics)

    def test_attention_pooling_is_text_free_and_trainable(self):
        rng = np.random.default_rng(11)
        patches = rng.normal(size=(60, 5, 8)).astype(np.float32)
        patches[self.y == 1, :, 0] += 1.2
        with patch(
            "src.methods.metric_attention_probes._get_device",
            return_value=torch.device("cpu"),
        ):
            model, metrics = train_one_attribute_attention_pooling(
                "attr", patches, self.y, patch_dim=8, num_heads=2,
                epochs=2, batch_size=12, seed=5,
            )
        self.assertIsInstance(model, LearnedQueryAttentionProbe)
        sample = torch.as_tensor(patches[:4])
        with torch.no_grad():
            plain = model(sample, None)
            arbitrary_text = model(sample, torch.randn(1, 17))
        torch.testing.assert_close(plain, arbitrary_text)
        self.assertIn("val_auc", metrics)

    def test_suite_and_attention_cache_identity_are_explicit(self):
        suite_path = ROOT / "configs/probe_suite.json"
        suite = runner.configure_suite(suite_path)
        self.assertEqual(suite["learned_methods"], list(runner.H.TRAINED_METHODS))
        patch_meta = {
            "patch_shape": [12, 5, 8],
            "patch_dtype": "float32",
            "patch_sources": "toy.npy",
        }
        text = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        conditioned = runner.attention_feature_context(
            "attribute_conditioned_attention", patch_meta, text,
        )
        unconditioned = runner.attention_feature_context(
            "attention_pooling", patch_meta, None,
        )
        self.assertIsNotNone(conditioned["text_query_hash"])
        self.assertIsNone(unconditioned["text_query_hash"])
        self.assertEqual(conditioned["patch_source_hash"], unconditioned["patch_source_hash"])
        json.dumps(conditioned)

    def test_attention_feature_context_rejects_incompatible_cache(self):
        emb = np.zeros((4, 8), dtype=np.float32)
        paths = [f"image_{i}.jpg" for i in range(4)]
        seeds = [0]
        feature_a = {
            "model_kind": "attn",
            "patch_shape": [4, 5, 8],
            "patch_dtype": "float32",
            "patch_source_hash": "a",
            "text_query_hash": "text",
        }
        feature_b = {**feature_a, "patch_source_hash": "b"}
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            entry = runner.cache_dir(
                root, "toy", "task", "random",
                "attribute_conditioned_attention", "attr",
            )
            entry.mkdir(parents=True)
            meta = runner.expected_cache_meta(
                "toy", "task", "random", "attribute_conditioned_attention", "attr",
                emb, paths, seeds, "supervision", 2, feature_a,
            )
            (entry / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            np.savez_compressed(entry / "scores.npz", scores=np.zeros((1, 4)))
            self.assertIsNotNone(runner.read_cache(
                root, "toy", "task", "random",
                "attribute_conditioned_attention", "attr",
                emb, paths, seeds, "supervision", 2,
                feature_context=feature_a,
            ))
            with self.assertRaisesRegex(ValueError, "feature_context"):
                runner.read_cache(
                    root, "toy", "task", "random",
                    "attribute_conditioned_attention", "attr",
                    emb, paths, seeds, "supervision", 2,
                    feature_context=feature_b,
                )

    def test_probebank_trains_scores_and_serializes_attention_probe(self):
        rng = np.random.default_rng(23)
        paths = [f"image_{i}.jpg" for i in range(12)]
        p2i = {path: i for i, path in enumerate(paths)}
        selected = paths[:8]
        labels = [
            {"image": path, "attr": int(i >= 4)}
            for i, path in enumerate(selected)
        ]
        emb = rng.normal(size=(12, 8)).astype(np.float32)
        patches = rng.normal(size=(12, 5, 8)).astype(np.float32)
        patches[4:8, :, 0] += 1.0
        patch_meta = {
            "patch_shape": [12, 5, 8],
            "patch_dtype": "float32",
            "patch_sources": "toy.npy",
        }
        runner.H.EPOCHS = 1
        with tempfile.TemporaryDirectory() as raw_root, patch(
            "src.methods.metric_attention_probes._get_device",
            return_value=torch.device("cpu"),
        ), patch(
            "src.methods.attn_probe._get_device",
            return_value=torch.device("cpu"),
        ):
            root = Path(raw_root)
            cases = [
                ("attention_pooling", None, "LearnedQueryAttentionProbe"),
                ("attribute_conditioned_attention", torch.randn(1, 6), "AttnPoolMLPWithProj"),
            ]
            for method, text_query, expected_class in cases:
                with self.subTest(method=method):
                    feature_context = runner.attention_feature_context(
                        method, patch_meta, text_query,
                    )
                    meta, scores = runner.train_cache_entry(
                        root, "toy", "task", "random", method, "attr",
                        emb, paths, p2i, paths, selected, labels,
                        [0], "supervision", 1,
                        patches=patches,
                        text_query=text_query,
                        feature_context=feature_context,
                    )
                    self.assertEqual(scores.shape, (1, 12))
                    self.assertTrue(np.isfinite(scores).all())
                    self.assertEqual(meta["feature_context"], feature_context)
                    self.assertEqual(meta["model_files"], ["model_seed_0.pt"])
                    state_path = runner.cache_dir(
                        root, "toy", "task", "random", method, "attr",
                    ) / "model_seed_0.pt"
                    state = torch.load(state_path, map_location="cpu", weights_only=True)
                    self.assertEqual(state["kind"], "attn")
                    self.assertEqual(state["model_class"], expected_class)


if __name__ == "__main__":
    unittest.main()
