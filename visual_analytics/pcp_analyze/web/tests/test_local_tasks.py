"""New tasks must isolate user labels, Val and the published paper assets."""
import csv
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fixed_vqa_validation as fixed
import local_tasks
from tuning_server import TuningService, REFINEMENT_EMBEDDING_METHODS
from val_isolation import build_task_contract, materialize_training_inputs


class LocalTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.web = self.root / "visual_analytics/pcp_analyze/web"
        self.input = self.root / "artifacts/task"
        self.input.mkdir(parents=True)
        self.records = self.root / "dataset/raw/stanford_cars/processed/records.csv"
        self.records.parent.mkdir(parents=True)
        self.ids = [f"{r+1:06d}.jpg" for r in range(256)]
        with self.records.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["embedding_index", "relative_path"])
            writer.writerows(enumerate(self.ids))
        self.config = {"name": "new_task", "dataset": "cars",
                       "query_images": [self.ids[-1]], "attributes": [{"id": "first", "name": "First"},
                       {"id": "second", "name": "Second"}], "labels": "labels.csv"}
        (self.input / "task.json").write_text(json.dumps(self.config))
        self.rows = [[self.ids[r], r % 2, (r // 2) % 2] for r in range(100)]
        self.write_labels()
        source = Path(__file__).resolve().parents[4] / "scripts/custom_task.py"
        self.source_root = source.parent.parent
        sys.path.insert(0, str(self.source_root / "probe_learning"))
        (self.root / "configs").mkdir()
        shutil.copyfile(self.source_root / "configs/component_ablation.json", self.root / "configs/component_ablation.json")
        spec = importlib.util.spec_from_file_location("custom_task_test_cli", source)
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.cli.ROOT, self.cli.WEB = self.root, self.web

    def write_labels(self):
        with (self.input / "labels.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "first", "second"])
            writer.writerows(self.rows)

    def prepare(self):
        value, ids, _ = local_tasks.validate_input(self.input / "task.json", self.root)
        tid, version, private, bundle, _ = self.cli.create_inputs(value, ids, 1)
        service = TuningService(self.web, start_worker=False, catalog_path=private/"catalog.json")
        return service, tid, version, bundle

    def test_csv_order_does_not_change_holdout_and_unlisted_rows_stay_unlabeled(self):
        first, _, _ = local_tasks.validate_input(self.input / "task.json", self.root)
        self.rows.reverse()
        self.write_labels()
        second, _, _ = local_tasks.validate_input(self.input / "task.json", self.root)
        self.assertEqual(first, second)
        self.assertEqual(len(first["labels"]), 100)
        self.assertEqual(len(first["validationRows"]), 20)

    def test_rejects_duplicate_query_missing_and_nonbinary_supervision(self):
        original = list(self.rows)
        for bad in (self.rows[0], [self.ids[-1], 1, 1], ["unknown.jpg", 0, 1], [self.ids[100], "", 1], [self.ids[100], 2, 0]):
            with self.subTest(row=bad):
                self.rows = original + [bad]
                self.write_labels()
                with self.assertRaises(ValueError):
                    local_tasks.validate_input(self.input / "task.json", self.root)

    def test_rejects_label_path_escape(self):
        self.config["labels"] = "../labels.csv"
        (self.input / "task.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "inside"):
            local_tasks.validate_input(self.input / "task.json", self.root)

    def test_frozen_local_val_never_switches_paper_pointer_or_uses_public_gt(self):
        service, tid, version, bundle = self.prepare()
        pointer = service.vqa_validation_root / "active.json"
        pointer.parent.mkdir(parents=True)
        pointer.write_text('{"version":"paper-unchanged"}')
        before = pointer.read_bytes()
        fixed.freeze_vqa_validation(service, version, activate=False)
        self.assertEqual(pointer.read_bytes(), before)
        contract = build_task_contract(service, tid, version)
        self.assertEqual(contract["labelSource"], "user-provided-supervision")
        self.assertFalse(set(contract["valRows"]) & set(contract["trainPoolRows"]))
        self.assertFalse(set(contract["queryRows"]) & set(contract["trainPoolRows"]))
        self.assertEqual(contract["testRows"], [])
        self.assertNotIn("groundTruth", service.task(tid).manifest["files"])
        inputs = materialize_training_inputs(contract, self.ids, self.ids)
        self.assertEqual(len(inputs["fit_paths"]), 80)
        self.assertEqual(len(inputs["val_labels"]), 20)
        with self.assertRaisesRegex(ValueError, "no independent ground truth"):
            service._task_ground_truth(tid)

    def test_input_snapshot_tampering_is_rejected(self):
        service, tid, _, _ = self.prepare()
        spec = service.task(tid).manifest["localTask"]
        path = self.root / spec["path"]
        value = json.loads(path.read_text())
        value["labels"][0][1] = 1 - value["labels"][0][1]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "changed"):
            local_tasks.original_supervision(service, tid, "first")

    def test_legacy_query_text_has_no_effect_on_task_identity(self):
        first, _, _ = local_tasks.validate_input(self.input / "task.json", self.root)
        self.config["query_text"] = "An unused extra condition"
        (self.input / "task.json").write_text(json.dumps(self.config))
        second, _, _ = local_tasks.validate_input(self.input / "task.json", self.root)
        self.assertEqual(first, second)
        self.assertNotIn("query_text", second["config"])

    def test_paper_val_gates_export_reload_and_checkpoint_attestation(self):
        import unified_initial_baseline as initial
        from sklearn.metrics import average_precision_score
        from tuning_models import unified_weight_scores

        service, tid, version, _ = self.prepare()
        fixed.freeze_vqa_validation(service, version, activate=False)
        contract = build_task_contract(service, tid, version)
        rng = np.random.default_rng(42)
        raw = rng.uniform(.02, .98, (5, 256, 2, 8)).astype(np.float32)
        embedding = rng.uniform(.02, .98, (256, 2)).astype(np.float32)
        theta, temperature = np.array([.4, .55], dtype=np.float32), np.array([.11, .19], dtype=np.float32)
        reference = {"protocol": initial.CALIBRATION, "usesValLabels": False}
        bank = self.web.parent / "runtime/isolated-probes" / tid / "test-bank"
        bank.mkdir(parents=True)
        proof = {"bankTrainingIdentity": {"trainerSha256": "test", "epochs": 1,
                 "methods": list(initial.METHODS), "seeds": list(initial.SEEDS)},
                 "scoreImageIds": self.ids, "forwardScope": "fixture", "samplingProtocol": "fixture",
                 "sampleImageIds": self.ids[:2]}
        with patch.object(initial, "verify_bank", return_value=(raw, contract, {"model.pt": "fixture"}, proof)), \
             patch.object(initial, "calibrate", return_value=(theta, temperature, reference)), \
             patch.object(service, "_exported_method_column", side_effect=lambda task, field, method, target:
                          embedding[:, REFINEMENT_EMBEDDING_METHODS.index(method)]):
            base, metadata, directory = initial.build_task_baseline(service, tid, version, bank, select_on_val=True)

        self.assertEqual(metadata["calibration"], initial.VAL_CALIBRATION)
        self.assertTrue(metadata["calibrationUsedValidation"])
        self.assertFalse(metadata["initialModelHoldoutIndependent"])
        self.assertEqual(base.initial_theta.dtype, np.float64)
        self.assertEqual(base.temperature.dtype, np.float64)
        calibration = initial.read(directory / "calibration.json")
        self.assertEqual(len(calibration["candidates"]), 20)
        rows = np.asarray(contract["valRows"])
        labels = contract["targets"]["joint"]["valLabels"]
        def score(t, temp):
            return unified_weight_scores(base.probe_features, base.embedding_features,
                np.full((2, 8), 1/8), np.ones(2), np.full(2, .5), .25, t, temp).final_scores.astype("<f4")
        candidates = [score(np.clip(theta.astype(float) + shift, 0, 1),
                            np.minimum(temperature.astype(float) * scale, .30))
                      for shift in (0, -.05, .05, -.10, .10) for scale in (1, 1.5, 2, 3)]
        final = score(base.initial_theta, base.temperature)
        self.assertAlmostEqual(average_precision_score(labels, final[rows]),
                               max(average_precision_score(labels, s[rows]) for s in candidates))
        self.assertEqual((directory / "scores.f32").read_bytes(), final.tobytes())
        self.assertEqual(initial.read(initial.published_bank_attestation(service, tid, base)),
                         initial.read(bank / "refinement_base.json"))
        initial.publish(service, version, activate=False)
        self.assertFalse((initial.root_path(service) / "active.json").exists())
        self.assertEqual(initial.load_task_baseline(service, tid)[1]["classificationThreshold"],
                         calibration["classificationThreshold"])

        changed_z, changed_e = base.probe_features.copy(), base.embedding_features.copy()
        outside = np.setdiff1d(np.arange(256), rows)
        changed_z[outside], changed_e[outside] = np.nan, np.nan
        selected = initial.select_paper_gates(service, changed_z, changed_e, theta, temperature, contract, reference)
        np.testing.assert_array_equal(selected[0], base.initial_theta)
        np.testing.assert_array_equal(selected[1], base.temperature)
        self.assertEqual(selected[2]["candidates"], calibration["candidates"])
        contract["targets"]["joint"]["fitRows"].append(int(rows[0]))
        with self.assertRaisesRegex(RuntimeError, "overlaps probe fitting"):
            initial.select_paper_gates(service, changed_z, changed_e, theta, temperature, contract, reference)


if __name__ == "__main__":
    unittest.main()
