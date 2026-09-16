"""Synthetic contracts only: no production checkpoints or training jobs."""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from tests.test_tuning_backend import TUNING
except ModuleNotFoundError:
    from test_tuning_backend import TUNING
import native_probe_update as NATIVE


class NativeSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.web = root / "experiment" / "visual_analytics" / "pcp_analyze" / "web"
        self.web.mkdir(parents=True)
        self.images = ["img-c", "img-a", "img-b", "img-d", "img-e", "img-f"]
        self.offline = sorted(self.images)
        self.methods = tuple(reversed(NATIVE.METHODS))
        self.raw_offline = np.linspace(.07, .91, 5 * 6 * 2 * 8).reshape(5, 6, 2, 8).astype(np.float32)
        self.mapping = np.asarray([self.offline.index(image) for image in self.images])
        self.raw = self.raw_offline[:, self.mapping]
        mean = self.raw.astype(np.float64).mean(0).astype(np.float32)
        low, high = mean[:2].min(0), mean[:2].max(0)
        z = np.clip((mean.astype(np.float64) - low) / (high - low), 0, 1).astype(np.float32)
        self.base = TUNING.RefinementBaseContext(
            attribute_ids=("a", "b"), attribute_names=("Attr A", "Attr B"), learner_names=self.methods,
            embedding_names=TUNING.REFINEMENT_EMBEDDING_METHODS, probe_features=z,
            probe_min=low, probe_max=high, temperature=np.ones(2)*.2, initial_theta=np.ones(2)*.5,
            embedding_features=np.zeros((6, 2)), embedding_min=np.zeros(2), embedding_max=np.ones(2),
            development_indices=np.asarray([0, 1]), base_state_fingerprint="shared-base",
            normalization_audit={"vqaValidationVersion": "v1"}, raw_probe_probabilities=mean,
        )
        self.contract = {"fingerprint": "isolation", "rowCount": 6,
            "attributes": [{"id": "a", "name": "Attr A"}, {"id": "b", "name": "Attr B"}],
            "valRows": [3], "testRows": [4], "queryRows": [5]}
        records = pd.DataFrame({"embedding_index": range(6), "relative_path": self.offline})
        self.source = TUNING.TuningSourceContext(
            adapter=SimpleNamespace(records=records), task_config={}, attributes=self.base.attribute_names,
            target_attributes=("Attr A", "Attr B", None), probe_stage="isolated",
            suite_path=None, supervision_audit_path=None,
        )
        self.service = SimpleNamespace(web_root=self.web, runtime_root=root / "private",
            task=lambda task: SimpleNamespace(task_id=task),
            bundle=lambda task: SimpleNamespace(image_ids=self.images),
            _tuning_source_context=lambda task: self.source,
            _weighted_fusion_context=lambda task: SimpleNamespace(
                learned_method_ids=self.methods, learned_method_labels=self.methods),
            _refinement_base_context=lambda task: self.base)
        self.build_patch = patch.object(NATIVE, "build_task_contract", return_value=self.contract)
        self.build_patch.start()
        self.addCleanup(self.build_patch.stop)
        NATIVE._BANK_CACHE.clear()

    def write_bank(self):
        self.bank_root = self.web.parent / "runtime" / "isolated-probes" / "task" / ("a" * 64)
        self.bank_root.mkdir(parents=True)
        NATIVE.write_json(self.bank_root.parent / "active.json", {"directory": self.bank_root.name})
        identity = {"protocol": "val-isolated-probebank-v1", "taskId": "task",
                    "methods": list(NATIVE.METHODS), "seeds": list(NATIVE.SEEDS), "isolationFingerprint": "isolation"}
        NATIVE.write_json(self.bank_root / "run_identity.json", identity)
        NATIVE.write_json(self.bank_root / "complete.json", {**identity, "modelsRetrained": True, "published": False})
        NATIVE.write_json(self.bank_root / "isolation_contract.json", self.contract)
        for a, attr in enumerate(self.base.attribute_names):
            for m, method in enumerate(self.methods):
                directory = self.bank_root / f"{a}-{method}"
                directory.mkdir()
                NATIVE.write_json(directory / "metadata.json", {"canonical_attribute": attr, "method": method,
                    "seeds": list(NATIVE.SEEDS), "val_isolation_fingerprint": "isolation", "score_length": 6})
                np.savez(directory / "scores.npz", scores=self.raw_offline[:, :, a, m])
                for seed in NATIVE.SEEDS:
                    (directory / f"model_seed_{seed}.pt").write_bytes(f"synthetic-{a}-{m}-{seed}".encode())
        self.relink()

    def relink(self):
        names = ("metadata.json", "scores.npz")
        paths = [path for path in self.bank_root.rglob("*") if path.is_file() and (
            path.name in names or path.name.startswith("model_seed_"))]
        NATIVE.write_json(self.bank_root / "refinement_base.json", {
            "schemaVersion": 1, "baseStateFingerprint": self.base.base_state_fingerprint,
            "normalizationContract": self.base.normalization_audit, "scoreImageIds": self.offline,
            "probeMethods": list(self.methods), "forwardVerified": True,
            "files": {str(path.relative_to(self.bank_root)): NATIVE.file_sha(path) for path in paths},
        })

    def test_missing_bank_is_explicitly_unavailable(self):
        self.service._refinement_base_context = lambda task: self.fail("Missing bank should not load score matrices")
        result = NATIVE.capability(self.service, "task")
        self.assertFalse(result["available"])
        self.assertIn("Base probe checkpoints unavailable", result["reason"])

    def test_permuted_image_and_method_orders_are_mapped_by_identity(self):
        self.write_bank()
        bank = NATIVE.resolve_bank(self.service, "task", self.base)
        np.testing.assert_array_equal(bank["rawProbabilities"], self.raw)
        self.assertEqual(bank["methods"], self.methods)
        unchanged = NATIVE.updated_base(self.base, bank["rawProbabilities"], "snapshot")
        np.testing.assert_array_equal(unchanged.probe_features, self.base.probe_features)

    def test_published_checkpoint_replacement_is_not_silently_accepted(self):
        self.write_bank()
        NATIVE.resolve_bank(self.service, "task", self.base)
        checkpoint = next(self.bank_root.rglob("model_seed_0.pt"))
        checkpoint.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            NATIVE.resolve_bank(self.service, "task", self.base)

    def test_changed_same_shape_image_identity_is_rejected(self):
        self.write_bank()
        self.images[0] = "unrelated-image"
        with self.assertRaisesRegex(RuntimeError, "identity"):
            NATIVE.resolve_bank(self.service, "task", self.base)

    def test_probability_base_mismatch_is_rejected(self):
        self.write_bank()
        changed = dataclasses.replace(self.base, raw_probe_probabilities=self.base.raw_probe_probabilities + .01)
        with self.assertRaisesRegex(RuntimeError, "probabilities differ"):
            NATIVE.resolve_bank(self.service, "task", changed)

    def request(self, params=None, annotations=None, user="u", session="s"):
        params = {"feedbackWeight": 8, "vqaValidationVersion": "v1", **(params or {})}
        annotations = annotations or [{"rowIndex": 0, "label": -1,
                        "failureAttributionConfirmed": True, "failedAttributeIds": ["a"]}]
        def prepare(task, base, notes, **kwargs):
            return {"attributeExamples": [{"rows": [0, 1], "labels": [0, 1],
                    "weights": [kwargs["feedback_weight"] if attr in notes[0].get("failedAttributeIds", []) else 1, 1]}
                    for attr in base.attribute_ids]}
        self.service._prepare_weight_refinement_supervision = prepare
        bank = {"fingerprint": params.pop("bankFingerprint", "bank"), "contract": self.contract,
                "methods": self.methods}
        with patch.object(NATIVE, "resolve_bank", return_value=bank), patch.object(NATIVE, "file_sha", return_value="code-sha"):
            return NATIVE.prepare_request(self.service, user, session, "task", self.base, annotations, params)

    def test_snapshot_key_excludes_fusion_schedule_and_hyperparameters(self):
        staged = self.request({"mode": "probe_staged", "learningRate": .1, "maxIterations": 10, "alphaBeta": 1})
        joint = self.request({"mode": "probe_joint", "learningRate": .02, "maxIterations": 100, "alphaBeta": .2})
        self.assertEqual(staged, joint)
        self.assertTrue(staged["examples"][0]["update"])
        self.assertFalse(staged["examples"][1]["update"])

    def test_snapshot_key_binds_user_session_feedback_bank_val_and_native_config(self):
        initial = self.request()["snapshotId"]
        variants = [self.request(user="other"), self.request(session="other"),
                    self.request({"feedbackWeight": 16}), self.request({"bankFingerprint": "changed"}),
                    self.request({"nativeUpdateConfig": {"epochs": 21}}),
                    self.request(annotations=[{"rowIndex": 0, "label": -1,
                        "failureAttributionConfirmed": True, "failedAttributeIds": ["b"]}])]
        previous = self.base
        self.base = dataclasses.replace(self.base, normalization_audit={"vqaValidationVersion": "v2"})
        variants.append(self.request())
        self.base = dataclasses.replace(previous, base_state_fingerprint="changed-base")
        variants.append(self.request())
        self.assertTrue(all(request["snapshotId"] != initial for request in variants))

    def test_snapshot_reuse_and_checksum_validation(self):
        request = self.request()
        calls = []
        self.service._run_native_probe_update = lambda *args: (calls.append(1) or self.raw, {"native": True})
        with patch.object(NATIVE, "resolve_bank", return_value={"fingerprint": "bank"}):
            first, audit = NATIVE.ensure_snapshot(self.service, request, self.base)
            second, again = NATIVE.ensure_snapshot(self.service, request, self.base)
        self.assertEqual(calls, [1])
        self.assertEqual(audit, again)
        np.testing.assert_array_equal(first.probe_features, second.probe_features)
        path = self.service.runtime_root / "native-probe-updates" / request["snapshotId"] / "probabilities.npy"
        path.write_bytes(b"damaged")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            NATIVE.load_snapshot(self.service, request, self.base)

    def test_invalid_native_result_is_not_published(self):
        request = self.request()
        self.service._run_native_probe_update = lambda *args: (np.full_like(self.raw, np.nan), {})
        with patch.object(NATIVE, "resolve_bank", return_value={"fingerprint": "bank"}):
            with self.assertRaisesRegex(RuntimeError, "not published"):
                NATIVE.ensure_snapshot(self.service, request, self.base)
        self.assertFalse((self.service.runtime_root / "native-probe-updates" / request["snapshotId"]).exists())

    def test_build_lock_does_not_block_capability_lookup(self):
        self.write_bank()
        completed = threading.Event()
        with NATIVE._BUILD_LOCK:
            thread = threading.Thread(target=lambda: (NATIVE.capability(self.service, "task"), completed.set()))
            thread.start()
            self.assertTrue(completed.wait(5), "Read-only capabilities blocked on a native training lock")
        thread.join()

    def test_native_orchestration_calls_all_seeds_and_maps_permuted_coordinates(self):
        """Exercise real API signatures without features, training, or writes."""
        root = Path(__file__).resolve().parents[4]
        for path in (root / "probe_learning", root / "probe_learning" / "scripts"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import torch
        import run_probebank_batch as bank_api
        import run_retrieval_harness as harness
        from src.methods import native_probe_update as native

        count = 24
        images = [f"image-{row}" for row in range(count)]
        offline = list(reversed(images))
        to_offline = np.arange(count - 1, -1, -1)
        emb = np.repeat(np.arange(count, dtype=np.float32)[:, None], 8, axis=1)
        patches = np.repeat(emb[:, None, :], 3, axis=1)
        methods = tuple(reversed(native.NATIVE_METHODS))
        bank_root = Path(self.temporary.name) / "synthetic-bank"
        entries = {f"0:{m}": str(bank_root / f"method-{m}") for m in range(8)}
        files = {str((Path(entries[f"0:{m}"]) / f"model_seed_{seed}.pt").relative_to(bank_root)): "0" * 64
                 for m in range(8) for seed in range(5)}
        contract = {"attributes": [{"id": "a", "name": "a"}, {"id": "b", "name": "b"}],
            "valRows": [12, 13, 14, 15], "testRows": [22], "queryRows": [23],
            "trainPoolRows": list(range(12)) + list(range(16, 22)),
            "targets": {"a": {"valLabels": [0, 1, 0, 1]}, "b": {"valLabels": [1, 0, 1, 0]}}}
        request = {"taskId": "synthetic", "config": {}, "examples": [
            {"attributeId": "a", "update": True, "rows": list(range(12)), "labels": [0, 1] * 6, "weights": [1] * 12},
            {"attributeId": "b", "update": False, "rows": list(range(12)), "labels": [1, 0] * 6, "weights": [1] * 12}]}
        bank = {"contract": contract, "directory": str(bank_root), "entries": entries, "files": files,
            "rawProbabilities": np.full((5, count, 2, 8), .3, dtype=np.float32),
            "methods": methods, "toOffline": to_offline, "offlineImageIds": offline}
        service = SimpleNamespace(web_root=root / "visual_analytics" / "pcp_analyze" / "web",
            task=lambda task: SimpleNamespace(task_id=task, dataset_id="synthetic", task_name="synthetic"),
            _tuning_source_context=lambda task: SimpleNamespace(adapter=object()),
            bundle=lambda task: SimpleNamespace(image_ids=images),
            _weighted_fusion_context_guard=threading.RLock())
        with patch.object(harness, "configure"), \
             patch.object(bank_api, "database_paths", return_value=offline), \
             patch.object(harness, "load_backbone_embeddings", return_value=(emb, {})), \
             patch.object(harness, "load_backbone_patches", return_value=(patches, {})), \
             patch.object(bank_api, "cached_attribute_text_queries", return_value={"a": torch.ones(1, 4), "b": torch.ones(1, 4)}), \
             patch.object(bank_api, "attention_feature_context", return_value=None), \
             patch.object(NATIVE, "read_json", return_value={"embedding_dim": 8}), \
             patch.object(native, "update_native_probe", autospec=True, return_value=(torch.nn.Linear(8, 1), {})) as update, \
             patch.object(native, "score_native_probe", autospec=True, return_value=np.arange(count, dtype=np.float32) / count), \
             patch.object(torch, "save") as save:
            raw, audit = NATIVE.run_native_updates(service, request, bank, Path(self.temporary.name) / "output")
        self.assertEqual(update.call_count, 40)
        self.assertEqual(save.call_count, 40)
        self.assertEqual([call.args[0] for call in update.call_args_list], [method for method in methods for _ in range(5)])
        self.assertEqual([call.kwargs["seed"] for call in update.call_args_list], list(range(5)) * 8)
        for call in update.call_args_list:
            features = call.args[2]
            np.testing.assert_array_equal(features[:, 0] if features.ndim == 2 else features[:, 0, 0], to_offline[:12])
            self.assertEqual(call.kwargs["fit_ids"], images[:12])
            self.assertEqual(call.kwargs["validation_ids"], images[12:16])
            val_features = call.kwargs["validation_data"][0]
            np.testing.assert_array_equal(val_features[:, 0] if val_features.ndim == 2 else val_features[:, 0, 0], to_offline[12:16])
            if call.args[0] in native.PU_METHODS:
                np.testing.assert_array_equal(call.kwargs["X_unlabeled"][:, 0], to_offline[16:22])
                self.assertEqual(call.kwargs["unlabeled_ids"], images[16:22])
        np.testing.assert_array_equal(raw[:, :, 1, :], bank["rawProbabilities"][:, :, 1, :])
        np.testing.assert_array_equal(raw[0, :, 0, 0], np.arange(count, dtype=np.float32)[to_offline] / count)
        self.assertTrue(audit["frozenDuringFusion"])


if __name__ == "__main__":
    unittest.main()
