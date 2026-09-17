"""Portable labels must preserve the frozen bank without reading session history."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.test_fixed_vqa_validation import ToyService, TASK
import fixed_vqa_validation as fixed
import portable_tasks
from val_isolation import build_task_contract


class PortableTasksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ToyService(self.root)
        s = self.service
        s.web_root = self.root / "visual_analytics/pcp_analyze/web"
        s.spec.dataset_id, s.spec.task_id, s.spec.task_name = "cars", TASK, "toy"
        s._tuning_source_context = lambda tid: SimpleNamespace(attributes=("First", "Second"))
        s._vqa_validation_split = lambda tid, target, version=None: fixed.load_vqa_validation_split(s, tid, target, version)
        fixed.freeze_vqa_validation(s, "toy-v1")
        old_root = s.vqa_validation_root / "toy-v1"
        old_manifest, mh = fixed._read(old_root / "manifest.json")
        old, th = fixed._read(old_root / old_manifest["tasks"][TASK]["file"])
        self.contract = build_task_contract(s, TASK, "toy-v1")
        prefix = "dataset/tasks/cars/toy"
        (self.root / prefix).mkdir(parents=True)
        (self.root / "manifests").mkdir()
        ch = fixed._write(self.root / prefix / "isolation.json", self.contract)
        clean = {
            "protocol": portable_tasks.PROTOCOL, "taskId": TASK,
            "source": old["source"], "validationRows": old["rows"], "validationImageIds": old["imageIds"],
            "selection": old["selection"],
            "validationProvenance": {"version": "toy-v1", "manifestSha256": mh, "taskSha256": th, "createdAt": old_manifest["createdAt"]},
            "isolationContract": {"path": prefix + "/isolation.json", "sha256": ch},
            "targets": {name: {**{k: v[k] for k in ("sourceRows", "sourceLabels", "fitRows", "fitLabels", "labels")},
                                  "supervisionHash": v["audit"]["sourceSupervisionHash"]} for name, v in old["targets"].items()},
        }
        self.document_path = self.root / prefix / "original_vqa.json"
        dh = fixed._write(self.document_path, clean)
        ph = fixed._write(self.root / "dataset/tasks/cars/manifest.json", {
            "protocol": portable_tasks.PROTOCOL, "dataset": "cars",
            "tasks": {TASK: {"path": prefix + "/original_vqa.json", "sha256": dh}},
        })
        fixed._write(self.root / "manifests/task_packages.json", {"datasets": {"cars": {"manifestSha256": ph}}})
        s.vqa_validation_root = self.root / "not-installed"
        s.connect = lambda: self.fail("Portable loading must not inspect historical sessions")

    def test_portable_labels_reconstruct_identical_bank_contract_without_legacy_inputs(self):
        actual = build_task_contract(self.service, TASK, "toy-v1")
        self.assertEqual(actual, self.contract)
        split = fixed.load_vqa_validation_split(self.service, TASK, "joint")
        self.assertFalse(split.fit_indices.flags.writeable)
        self.assertFalse(set(split.fit_indices) & set(split.validation_indices))
        self.assertIn("portableManifestSha256", split.audit)
        original = portable_tasks.original_supervision(self.service, TASK, "joint")
        self.assertEqual(len(original.selected_indices), len(self.service.original_rows["joint"]))

    def test_changed_label_bytes_fail_instead_of_falling_back(self):
        value = json.loads(self.document_path.read_text())
        value["targets"]["joint"]["sourceLabels"][0] ^= 1
        fixed._write(self.document_path, value)
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            fixed.load_vqa_validation_split(self.service, TASK, "joint")


if __name__ == "__main__":
    unittest.main()
