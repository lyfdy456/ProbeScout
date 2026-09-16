"""Synthetic-only tests for task-common, original-VQA-label validation."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import fixed_vqa_validation as FIXED
from tuning_supervision import OriginalSupervision, build_probe_validation_split


TASK = "toy-task"


class ToyService:
    def __init__(self, root):
        self.vqa_validation_root = root / "vqa-validation"
        self.database_path = root / "history.sqlite3"
        self.spec = SimpleNamespace(
            row_count=34, target_ids=("first", "second", "joint"),
            manifest={"retrievalTargets": [
                {"id": "first", "kind": "attribute"},
                {"id": "second", "kind": "attribute"},
                {"id": "joint", "kind": "derived", "members": ["first", "second"]},
            ]},
        )
        self.tasks = {TASK: self.spec}
        self.data = SimpleNamespace(
            image_ids=[f"image-{row}.jpg" for row in range(34)],
            development_mask=bytes([1] * 26 + [0] * 8),
            validation_mask=bytes([0] * 26 + [1] * 6 + [0] * 2),
            test_mask=bytes([0] * 33 + [1]), query_indices={32},
        )
        # Deliberately non-row-sorted original ordering must survive in fit.
        source_rows = list(range(10, 20)) + list(range(10))
        self.original_rows = {target: list(source_rows) for target in self.spec.target_ids}
        self.original_rows["first"] += [25]
        self.original_rows["second"] += [26, 27]
        self.source_labels = {
            target: {row: int(row % 4 != (1 if target == "second" else 0)) for row in range(34)}
            for target in ("first", "second")
        }
        self.source_labels["joint"] = {
            row: self.source_labels["first"][row] & self.source_labels["second"][row]
            for row in range(34)
        }
        self.split_calls = []
        with self.connect() as connection:
            connection.executescript(
                "CREATE TABLE sessions(id TEXT PRIMARY KEY, task_id TEXT);"
                "CREATE TABLE annotation_events(session_id TEXT, row_index INTEGER);"
                "CREATE TABLE annotations(session_id TEXT, row_index INTEGER);"
            )
            connection.execute("INSERT INTO sessions VALUES ('session', ?)", (TASK,))

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database_path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def task(self, task_id):
        assert task_id == TASK
        return self.spec

    def bundle(self, task_id):
        assert task_id == TASK
        return self.data

    def _task_ground_truth(self, task_id):
        raise AssertionError("Public GT must never be read by the VQA artifact")

    def _original_development_supervision(self, task_id, target_id):
        assert task_id == TASK
        rows = np.asarray(self.original_rows[target_id], dtype=np.int64)
        labels = np.asarray([self.source_labels[target_id][int(row)] for row in rows], dtype=np.uint8)
        return OriginalSupervision(
            indices=rows, labels=labels, selected_indices=rows, selected_labels=labels,
            audit={"recoveredSupervisionHash": FIXED._sha(FIXED._json([rows.tolist(), labels.tolist()])),
                   "targetRule": "AND" if target_id == "joint" else "binary",
                   "strictValidation": {"status": "valid"}},
        )

    def _probe_validation_split(self, task_id, target_id):
        self.split_calls.append((task_id, target_id))
        return build_probe_validation_split(
            self._original_development_supervision(task_id, target_id), target_id=target_id,
            target_kind="derived" if target_id == "joint" else "attribute",
        )


class FixedVqaValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fixed-vqa-test-")
        self.addCleanup(self.temporary.cleanup)
        self.service = ToyService(Path(self.temporary.name))
        self.version = "vqa-toy-v1"

    def freeze(self):
        return FIXED.freeze_vqa_validation(self.service, self.version)

    def load(self, target="joint", version=None):
        return FIXED.load_vqa_validation_split(self.service, TASK, target, version)

    def rewrite_document(self, change):
        root = self.service.vqa_validation_root
        manifest_path = root / self.version / "manifest.json"
        manifest, _ = FIXED._read(manifest_path)
        document_path = root / self.version / manifest["tasks"][TASK]["file"]
        document, _ = FIXED._read(document_path)
        change(document)
        manifest["tasks"][TASK]["sha256"] = FIXED._write(document_path, document)
        manifest_hash = FIXED._write(manifest_path, manifest)
        FIXED._write(root / "active.json", {
            "schemaVersion": 1, "version": self.version, "manifestSha256": manifest_hash,
        })

    def test_common_joint_holdout_and_target_specific_source_minus_holdout(self):
        expected = self.service._probe_validation_split(TASK, "joint")
        self.service.split_calls.clear()
        self.freeze()
        self.assertEqual(self.service.split_calls, [(TASK, "joint")])
        for target in self.service.spec.target_ids:
            split = self.load(target)
            heldout = set(expected.validation_indices.tolist())
            self.assertEqual(split.validation_indices.tolist(), sorted(heldout))
            self.assertEqual(split.fit_indices.tolist(),
                             [row for row in self.service.original_rows[target] if row not in heldout])
            self.assertEqual(split.validation_labels.tolist(),
                             [self.service.source_labels[target][row] for row in sorted(heldout)])
            self.assertEqual(set(split.fit_indices) | heldout, set(self.service.original_rows[target]))
            self.assertFalse(set(split.fit_indices) & heldout)
            for array in (split.fit_indices, split.fit_labels, split.validation_indices, split.validation_labels):
                self.assertFalse(array.flags.writeable)

    def test_loader_never_replays_split_or_reads_public_truth(self):
        self.freeze()
        with mock.patch.object(self.service, "_probe_validation_split", side_effect=AssertionError("resampling")):
            first, second = self.load(), self.load()
        self.assertEqual(first.audit, second.audit)

    def test_existing_feedback_does_not_shrink_holdout_and_new_history_does_not_change_it(self):
        heldout = self.service._probe_validation_split(TASK, "joint").validation_indices.tolist()
        with self.service.connect() as connection:
            connection.execute("INSERT INTO annotation_events VALUES ('session', ?)", (heldout[0],))
        self.freeze()
        first = self.load()
        self.assertEqual(first.audit["historicalFeedbackOverlapCount"], 1)
        with self.service.connect() as connection:
            connection.execute("INSERT INTO annotations VALUES ('session', ?)", (heldout[1],))
            connection.execute("DELETE FROM annotation_events")
        second = self.load()
        self.assertEqual(first.audit, second.audit)
        self.assertEqual(second.validation_indices.tolist(), sorted(heldout))

    def test_reference_only_audit_and_hashes(self):
        self.freeze()
        audit = self.load().audit
        self.assertFalse(audit["initialModelHoldoutIndependent"])
        self.assertFalse(audit["baseModelsRetrained"])
        self.assertFalse(audit["usesPublicGroundTruth"])
        self.assertTrue(audit["referenceOnly"])
        self.assertEqual(audit["evaluationLabelSource"], "original-vqa-supervision")
        for partition in ("source", "fit", "validation"):
            for suffix in ("Fingerprint", "RowsSha256", "ImageIdsSha256", "LabelsSha256"):
                self.assertEqual(len(audit[partition + suffix]), 64)

    def test_existing_version_is_never_overwritten(self):
        self.freeze()
        root = self.service.vqa_validation_root
        before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*.json")}
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertEqual(before, {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*.json")})

    def test_pinned_version_survives_active_switch(self):
        self.freeze()
        original = self.load()
        FIXED.freeze_vqa_validation(self.service, "vqa-toy-v2")
        self.assertEqual(self.load(version=self.version).audit, original.audit)
        self.assertEqual(FIXED.active_vqa_validation_info(self.service)["version"], "vqa-toy-v2")

    def test_missing_artifact_fails_without_resampling(self):
        self.assertIsNone(FIXED.active_vqa_validation_info(self.service))
        with self.assertRaisesRegex(RuntimeError, "no automatic resampling"):
            self.load()
        self.assertEqual(self.service.split_calls, [])

    def test_corrupt_task_file_fails_checksum(self):
        self.freeze()
        path = self.service.vqa_validation_root / self.version / "000.json"
        FIXED._write(path, {"tampered": True})
        with self.assertRaisesRegex(RuntimeError, "corrupt"):
            self.load()

    def test_corrupt_active_manifest_fails_checksum(self):
        self.freeze()
        FIXED._write(self.service.vqa_validation_root / self.version / "manifest.json", {})
        with self.assertRaisesRegex(RuntimeError, "corrupt"):
            FIXED.active_vqa_validation_info(self.service)

    def test_changed_source_label_fails_for_even_unrequested_target(self):
        self.freeze()
        row = self.service.original_rows["second"][0]
        self.service.source_labels["second"][row] ^= 1
        with self.assertRaisesRegex(RuntimeError, "supervision, fit or labels changed"):
            self.load("joint")

    def test_changed_gallery_or_masks_fail(self):
        self.freeze()
        self.service.data.image_ids[0] = "other-image.jpg"
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            self.load()

    def test_tampered_fit_rows_fail_even_with_updated_file_checksums(self):
        self.freeze()
        self.rewrite_document(lambda document: document["targets"]["joint"]["fitRows"].append(document["rows"][0]))
        with self.assertRaisesRegex(RuntimeError, "supervision, fit or labels changed"):
            self.load()

    def test_tampered_labels_fail_even_with_updated_file_checksums(self):
        self.freeze()
        def change(document):
            document["targets"]["joint"]["labels"][0] ^= 1
        self.rewrite_document(change)
        with self.assertRaisesRegex(RuntimeError, "supervision, fit or labels changed"):
            self.load()

    def test_missing_attribute_holdout_labels_fail_without_activation(self):
        heldout = int(self.service._probe_validation_split(TASK, "joint").validation_indices[0])
        self.service.original_rows["first"].remove(heldout)
        with self.assertRaisesRegex(RuntimeError, "missing target labels"):
            self.freeze()
        self.assertIsNone(FIXED.active_vqa_validation_info(self.service))

    def test_query_and_test_source_rows_rejected(self):
        for row in (32, 33):
            with self.subTest(row=row):
                self.service.original_rows["first"].append(row)
                with self.assertRaisesRegex(RuntimeError, "Frozen Test or Query"):
                    FIXED.freeze_vqa_validation(self.service, f"invalid-{row}")
                self.service.original_rows["first"].remove(row)

    def test_failed_new_version_does_not_replace_active(self):
        self.freeze()
        before = (self.service.vqa_validation_root / "active.json").read_bytes()
        self.service.original_rows["first"].append(33)
        with self.assertRaises(RuntimeError):
            FIXED.freeze_vqa_validation(self.service, "vqa-broken-v2")
        self.assertEqual(before, (self.service.vqa_validation_root / "active.json").read_bytes())

    def test_invalid_versions_reject_path_escape(self):
        for version in ("../escape", "a/b", "a\\b", "", None):
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "Invalid"):
                FIXED.freeze_vqa_validation(self.service, version)

    def test_common_holdout_not_resampled_for_single_class_attribute(self):
        for row in self.service.original_rows["first"]:
            self.service.source_labels["first"][row] = 1
        self.freeze()
        split = self.load("first")
        self.assertTrue(np.all(split.validation_labels == 1))
        self.assertEqual(split.audit["validationNegativeCount"], 0)


if __name__ == "__main__":
    unittest.main()
