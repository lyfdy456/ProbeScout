"""New tasks must isolate user labels, Val and the published paper assets."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import fixed_vqa_validation as fixed
import local_tasks
from tuning_server import TuningService
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
        self.config = {"name": "new_task", "dataset": "cars", "query_text": "First and second",
                       "query_images": [self.ids[-1]], "attributes": [{"id": "first", "name": "First"},
                       {"id": "second", "name": "Second"}], "labels": "labels.csv"}
        (self.input / "task.json").write_text(json.dumps(self.config))
        self.rows = [[self.ids[r], r % 2, (r // 2) % 2] for r in range(100)]
        self.write_labels()
        source = Path(__file__).resolve().parents[4] / "scripts/custom_task.py"
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


if __name__ == "__main__":
    unittest.main()
