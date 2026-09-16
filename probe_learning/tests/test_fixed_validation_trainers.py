"""Tiny CPU tests for explicit holdout routing; no dataset or VQA access."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.methods.fixed_validation import split_labeled_validation
from src.methods import mlp_probe, pu_probe, nnpu_probe, dcpu_probe, pu_ranking_probe
from src.methods import attn_probe, metric_attention_probes
import run_retrieval_harness as H


class FixedValidationTrainerTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(12, 8)).astype(np.float32)
        self.y = np.asarray([0] * 4 + [1] * 8)
        self.Xv = rng.normal(size=(4, 8)).astype(np.float32) + 50
        self.yv = np.asarray([0, 1, 0, 1])
        self.U = rng.normal(size=(8, 8)).astype(np.float32)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_explicit_holdout_does_not_resplit_fit_and_legacy_is_exact(self):
        with patch("src.methods.fixed_validation.train_test_split", side_effect=AssertionError("must not resplit")):
            for seed in (0, 1, 4):
                actual = split_labeled_validation(self.X, self.y, seed, (self.Xv, self.yv))
                for observed, expected in zip(actual, (self.X, self.Xv, self.y, self.yv)):
                    np.testing.assert_array_equal(observed, expected)
        for observed, expected in zip(
            split_labeled_validation(self.X, self.y, 7),
            train_test_split(self.X, self.y, test_size=0.2, random_state=7, stratify=self.y),
        ):
            np.testing.assert_array_equal(observed, expected)

    def test_explicit_holdout_rejects_invalid_arrays(self):
        for data in (
            (), (self.Xv,), (self.Xv[:0], self.yv[:0]),
            (self.Xv, self.yv[:2]), (self.Xv[:, :3], self.yv),
            (self.Xv, np.asarray([0, 1, 2, 0])),
            (np.full_like(self.Xv, np.nan), self.yv),
        ):
            with self.subTest(data_shape=str([np.shape(value) for value in data])):
                with self.assertRaises(ValueError):
                    split_labeled_validation(self.X, self.y, 0, data)

    def test_all_eight_trainers_keep_all_fit_rows_and_validation_outside_updates(self):
        cases = [
            (mlp_probe, mlp_probe.train_one_attribute, (self.X, self.y, 8), {}),
            (pu_probe, pu_probe.train_one_attribute_weighted, (self.X, self.y, np.ones(12), 8), {}),
            (nnpu_probe, nnpu_probe.train_one_attribute_nnpu, (self.X, self.y, self.U, 8), {"batch_size_u": 4}),
            (dcpu_probe, dcpu_probe.train_one_attribute_dcpu, (self.X, self.y, self.U, 8), {"batch_size_u": 4}),
            (pu_ranking_probe, pu_ranking_probe.train_one_attribute_pu_ranking, (self.X, self.y, self.U, 8), {"batch_size_u": 4}),
            (metric_attention_probes, metric_attention_probes.train_one_attribute_triplet_loss, (self.X, self.y, 8), {"projection_dim": 4, "triplets_per_step": 4}),
            (metric_attention_probes, metric_attention_probes.train_one_attribute_attention_pooling,
             (np.repeat(self.X[:, None, :], 3, axis=1), self.y, 8), {"num_heads": 2}),
            (attn_probe, attn_probe.train_one_attribute_attn,
             (np.repeat(self.X[:, None, :], 3, axis=1), self.y, torch.ones(1, 4)),
             {"patch_dim": 8, "text_dim": 4, "num_heads": 2}),
        ]
        for module, trainer, args, extra in cases:
            with self.subTest(trainer=trainer.__name__), patch.object(module, "_get_device", return_value=torch.device("cpu")), contextlib.redirect_stdout(io.StringIO()):
                val_features = self.Xv if np.asarray(args[0]).ndim == 2 else np.repeat(self.Xv[:, None, :], 3, axis=1)
                first, metrics = trainer("attr", *args, epochs=1, batch_size=12, seed=2,
                                         validation_data=(val_features, self.yv), **extra)
                second, _ = trainer("attr", *args, epochs=1, batch_size=12, seed=2,
                                    validation_data=(-val_features, 1 - self.yv), **extra)
                self.assertEqual(metrics["train_samples"], 12)
                self.assertEqual(metrics["val_samples"], 4)
                self.assertAlmostEqual(metrics["train_pos_ratio"], 8 / 12)
                if "pi" in metrics:
                    self.assertAlmostEqual(metrics["pi"], 8 / 12)
                for key, value in first.state_dict().items():
                    self.assertTrue(torch.isfinite(value).all())
                    torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    def harness_context(self):
        emb = np.vstack([self.X, self.Xv, self.U])
        paths = [f"image-{index}" for index in range(len(emb))]
        fit = [{"image": path, "a": int(label), "b": 1 - int(label)} for path, label in zip(paths[:12], self.y)]
        val = [{"image": path, "a": int(label), "b": 1 - int(label)} for path, label in zip(paths[12:16], self.yv)]
        return fit, paths[:12], np.arange(16, 24), {
            "emb": emb, "patches": np.repeat(emb[:, None, :], 3, axis=1),
            "p2i": {path: index for index, path in enumerate(paths)},
            "text_q": {"a": torch.ones(1, 4), "b": torch.ones(1, 4)},
            "input_dim": 8, "patch_dim": 8, "text_dim": 4,
            "U_sample": np.arange(16, 20), "val_labeled": val,
        }

    def test_harness_routes_same_holdout_to_each_attribute_and_kfold_sees_fit_only(self):
        fit, selected, unlabeled, ctx = self.harness_context()
        with patch.object(H, "ATTRS", ["a", "b"]), patch.object(H, "EPOCHS", 1), \
             patch.object(H, "cross_validate_probs", return_value=np.full(12, 0.5)) as cv, \
             patch.object(H, "train_one_attribute_weighted", side_effect=lambda *args, **kwargs: (torch.nn.Linear(8, 1), {})) as train:
            models = H.train_method("kfold_pu", "mlp", fit, selected, unlabeled, ctx, 0)
        self.assertEqual(set(models), {"a", "b"})
        for model, _kind in models.values():
            self.assertEqual(model._probe_training_metrics["fit_count"], 12)
            self.assertEqual(model._probe_training_metrics["validation_count"], 4)
            self.assertEqual(model._probe_training_metrics["validation_policy"], "explicit-fixed-holdout-v1")
        self.assertEqual(cv.call_count, 2)
        for call in cv.call_args_list:
            np.testing.assert_array_equal(call.args[1], self.X)
            self.assertEqual(len(call.args[2]), 12)
        for index, call in enumerate(train.call_args_list):
            val_X, val_y = call.kwargs["validation_data"]
            np.testing.assert_array_equal(val_X, self.Xv)
            np.testing.assert_array_equal(val_y, self.yv if index == 0 else 1 - self.yv)

    def test_harness_rejects_unsupported_methods_id_overlap_unlabeled_leaks_and_missing_labels(self):
        for scenario in ("unsupported", "fit_overlap", "u_overlap", "sample_overlap", "missing_label", "empty_val"):
            fit, selected, unlabeled, ctx = self.harness_context()
            method = "mlp_baseline"
            if scenario == "unsupported":
                method = "high_conf"
            elif scenario == "fit_overlap":
                ctx["val_labeled"][0]["image"] = selected[0]
            elif scenario == "u_overlap":
                unlabeled = np.r_[unlabeled, 12]
            elif scenario == "sample_overlap":
                ctx["U_sample"] = np.asarray([12])
            elif scenario == "missing_label":
                del ctx["val_labeled"][0]["a"]
            elif scenario == "empty_val":
                ctx["val_labeled"] = []
            with self.subTest(scenario=scenario), patch.object(H, "ATTRS", ["a"]), \
                 patch.object(H, "train_one_attribute", side_effect=AssertionError("must reject before training")), \
                 self.assertRaises(ValueError):
                H.train_method(method, "mlp", fit, selected, unlabeled, ctx, 0)


if __name__ == "__main__":
    unittest.main()
