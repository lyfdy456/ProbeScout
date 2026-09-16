"""Isolated contract tests for the explicitly frozen Clean Validation set."""

from __future__ import annotations

import hashlib
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


WEB_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WEB_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import frozen_validation as FROZEN
from tuning_supervision import ProbeValidationSplit


TASK_ID = "toy-task"
PROTOCOL = "web-validation-minus-all-original-supervision-and-feedback-v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ToyService:
    """Only synthetic rows and a disposable SQLite fixture; no production data."""

    def __init__(self, root):
        self.clean_validation_root = root / "clean-validation"
        self.database_path = root / "toy.sqlite3"
        self.spec = SimpleNamespace(
            task_id=TASK_ID,
            target_ids=("attribute", "joint"),
            row_count=14,
            manifest={"retrievalTargets": [
                {"id": "attribute", "kind": "attribute"},
                {"id": "joint", "kind": "derived"},
            ]},
        )
        self.tasks = {TASK_ID: self.spec}
        self.data = SimpleNamespace(
            spec=self.spec,
            image_ids=[f"image-{index}.jpg" for index in range(14)],
            development_mask=bytes([1] * 4 + [0] * 10),
            validation_mask=bytes([0] * 4 + [1] * 8 + [0] * 2),
            test_mask=bytes([0] * 12 + [1] * 2),
            query_indices={0},
        )
        self.truth = np.asarray(
            [[row % 2, (row + 1) % 2] for row in range(14)], dtype=np.uint8
        )
        self.original_rows = {"attribute": [1, 2, 5], "joint": [1, 2, 5]}
        self.build_calls = []
        with self.connect() as connection:
            connection.executescript(
                "CREATE TABLE sessions(id TEXT PRIMARY KEY,task_id TEXT);"
                "CREATE TABLE annotation_events(session_id TEXT,row_index INTEGER);"
                "CREATE TABLE annotations(session_id TEXT,row_index INTEGER);"
            )
            connection.execute("INSERT INTO sessions VALUES (?,?)", ("session", TASK_ID))
            connection.execute("INSERT INTO annotation_events VALUES (?,?)", ("session", 6))

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def task(self, task_id):
        assert task_id == TASK_ID
        return self.spec

    def bundle(self, task_id):
        assert task_id == TASK_ID
        return self.data

    def _task_ground_truth(self, task_id):
        assert task_id == TASK_ID
        return self.truth.copy()

    def _historically_reviewed_rows(self, task_id):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT e.row_index FROM annotation_events e "
                "JOIN sessions s ON s.id=e.session_id WHERE s.task_id=? "
                "UNION SELECT a.row_index FROM annotations a "
                "JOIN sessions s ON s.id=a.session_id WHERE s.task_id=?",
                (task_id, task_id),
            ).fetchall()
        return {int(row[0]) for row in rows}

    def _original_development_supervision(self, task_id, target_id):
        assert task_id == TASK_ID
        rows = np.asarray(self.original_rows[target_id], dtype=np.int64)
        labels = self.truth[rows, self.spec.target_ids.index(target_id)]
        return SimpleNamespace(
            selected_indices=rows,
            selected_labels=labels,
            audit={"recoveredSupervisionHash": digest([rows.tolist(), labels.tolist()])},
        )

    def _probe_validation_split(self, task_id, target_id):
        source = self._original_development_supervision(task_id, target_id)
        return ProbeValidationSplit(
            fit_indices=source.selected_indices[:2],
            fit_labels=source.selected_labels[:2],
            validation_indices=source.selected_indices[2:],
            validation_labels=source.selected_labels[2:],
            audit={
                "sourceCount": len(source.selected_indices),
                "sourceFingerprint": source.audit["recoveredSupervisionHash"],
                "fitFingerprint": digest([
                    source.selected_indices[:2].tolist(),
                    source.selected_labels[:2].tolist(),
                ]),
            },
        )

    def _build_clean_validation_split(self, task_id, target_id, *, reviewed_rows=None):
        if reviewed_rows is None:
            reviewed_rows = self._historically_reviewed_rows(task_id)
        self.build_calls.append((target_id, frozenset(reviewed_rows)))
        original_rows = set().union(*self.original_rows.values())
        mask = np.frombuffer(self.data.validation_mask, dtype=np.uint8).astype(bool)
        excluded_original = sorted(row for row in original_rows if mask[row])
        excluded_reviewed = sorted(row for row in reviewed_rows if mask[row] and row not in original_rows)
        mask[excluded_original + excluded_reviewed] = False
        rows = np.flatnonzero(mask)
        labels = self.truth[rows, self.spec.target_ids.index(target_id)]
        fit = self._probe_validation_split(task_id, target_id)
        return ProbeValidationSplit(
            fit_indices=fit.fit_indices,
            fit_labels=fit.fit_labels,
            validation_indices=rows,
            validation_labels=labels,
            audit={
                **fit.audit,
                "schemaVersion": 2,
                "protocol": PROTOCOL,
                "replayKind": "clean-web-validation-ground-truth",
                "targetId": target_id,
                "targetKind": "derived" if target_id == "joint" else "attribute",
                "fitProtocol": "sklearn-stratified-train-test-split-v1",
                "fitSeed": 0,
                "fitFraction": 0.8,
                "fitCount": len(fit.fit_indices),
                "fitPositiveCount": int(np.count_nonzero(fit.fit_labels)),
                "fitNegativeCount": int(len(fit.fit_labels) - np.count_nonzero(fit.fit_labels)),
                "validationCount": len(rows),
                "validationPositiveCount": int(np.count_nonzero(labels)),
                "validationNegativeCount": int(len(labels) - np.count_nonzero(labels)),
                "validationFingerprint": digest({
                    "partition": "clean-validation",
                    "protocol": PROTOCOL,
                    "taskId": task_id,
                    "targetId": target_id,
                    "records": [[self.data.image_ids[int(row)], int(label)] for row, label in zip(rows, labels)],
                }),
                "originalSupervisionFingerprints": {
                    target: self._original_development_supervision(task_id, target).audit["recoveredSupervisionHash"]
                    for target in self.spec.target_ids
                },
                "excludedOriginalSupervisionCount": len(excluded_original),
                "excludedOriginalSupervisionFingerprint": digest([self.data.image_ids[row] for row in excluded_original]),
                "excludedHistoricalFeedbackCount": len(excluded_reviewed),
                "excludedHistoricalFeedbackFingerprint": digest([self.data.image_ids[row] for row in excluded_reviewed]),
                "allOriginalSupervisionExcluded": True,
                "allHistoricalFeedbackExcluded": True,
            },
        )


class FrozenValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pcp-frozen-validation-test-")
        self.addCleanup(temporary.cleanup)
        self.service = ToyService(Path(temporary.name))

    def freeze(self, version="v1"):
        return FROZEN.freeze_clean_validation(self.service, version)

    def load(self, target_id="joint", version=None):
        return FROZEN.load_frozen_validation_split(
            self.service, TASK_ID, target_id, version=version
        )

    def test_missing_freeze_fails_without_building(self):
        with self.assertRaises((RuntimeError, ValueError, FileNotFoundError)):
            self.load()
        self.assertEqual(self.service.build_calls, [])
        self.assertFalse(self.service.clean_validation_root.exists())

    def test_membership_does_not_follow_removed_history_or_new_development_feedback(self):
        self.freeze()
        before = self.load()
        self.assertEqual(before.validation_indices.tolist(), [4, 7, 8, 9, 10, 11])
        with self.service.connect() as connection:
            connection.execute("DELETE FROM annotation_events")
            connection.execute("INSERT INTO annotations VALUES (?,?)", ("session", 3))
        self.service.build_calls.clear()
        after = self.load()
        np.testing.assert_array_equal(before.validation_indices, after.validation_indices)
        np.testing.assert_array_equal(before.validation_labels, after.validation_labels)
        self.assertEqual(before.audit, after.audit)
        self.assertEqual(self.service.build_calls, [])

    def test_later_validation_feedback_fails_instead_of_shrinking(self):
        self.freeze()
        with self.service.connect() as connection:
            connection.execute("INSERT INTO annotation_events VALUES (?,?)", ("session", 7))
        self.service.build_calls.clear()
        with self.assertRaises((RuntimeError, ValueError)):
            self.load()
        self.assertEqual(self.service.build_calls, [])

    def test_all_targets_share_the_same_frozen_rows(self):
        self.freeze()
        np.testing.assert_array_equal(
            self.load("attribute").validation_indices,
            self.load("joint").validation_indices,
        )
        self.assertEqual(len(self.service.build_calls), 2)
        self.assertEqual({rows for _, rows in self.service.build_calls}, {frozenset({6})})

    def test_loaded_rows_and_labels_are_read_only(self):
        self.freeze()
        split = self.load()
        self.assertFalse(split.validation_indices.flags.writeable)
        self.assertFalse(split.validation_labels.flags.writeable)

    def test_existing_version_cannot_be_overwritten(self):
        self.freeze()
        files_before = {path: path.read_bytes() for path in self.service.clean_validation_root.rglob("*") if path.is_file()}
        with self.assertRaises((RuntimeError, ValueError, FileExistsError)):
            self.freeze()
        self.assertEqual(files_before, {path: path.read_bytes() for path in files_before})

    def test_explicit_older_version_remains_loadable_after_new_activation(self):
        self.freeze("v1")
        original = self.load(version="v1")
        with self.service.connect() as connection:
            connection.execute("DELETE FROM annotation_events")
        self.freeze("v2")
        latest = self.load()
        previous = self.load(version="v1")
        self.assertEqual(latest.audit["frozenVersion"], "v2")
        self.assertEqual(previous.audit["frozenVersion"], "v1")
        self.assertIn(6, latest.validation_indices.tolist())
        self.assertNotIn(6, previous.validation_indices.tolist())
        np.testing.assert_array_equal(original.validation_indices, previous.validation_indices)
        self.assertEqual(original.audit, previous.audit)

    def test_unknown_explicit_version_fails_without_building(self):
        self.freeze()
        self.service.build_calls.clear()
        with self.assertRaises(RuntimeError):
            self.load(version="missing-version")
        self.assertEqual(self.service.build_calls, [])

    def test_missing_task_file_is_not_rebuilt(self):
        manifest = self.freeze()
        path = self.service.clean_validation_root / "v1" / manifest["tasks"][TASK_ID]["file"]
        path.unlink()
        self.service.build_calls.clear()
        with self.assertRaises(RuntimeError):
            self.load()
        self.assertFalse(path.exists())
        self.assertEqual(self.service.build_calls, [])

    def test_corrupt_task_file_is_not_rebuilt(self):
        manifest = self.freeze()
        path = self.service.clean_validation_root / "v1" / manifest["tasks"][TASK_ID]["file"]
        path.write_bytes(b"corrupt task file")
        self.service.build_calls.clear()
        with self.assertRaises(RuntimeError):
            self.load()
        self.assertEqual(path.read_bytes(), b"corrupt task file")
        self.assertEqual(self.service.build_calls, [])

    def test_modified_manifest_is_rejected_by_active_checksum(self):
        self.freeze()
        path = self.service.clean_validation_root / "v1" / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["createdAt"] = "changed after freezing"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.load()

    def test_invalid_active_pointer_is_rejected(self):
        self.freeze()
        path = self.service.clean_validation_root / "active.json"
        path.write_bytes(b"not json")
        with self.assertRaises(RuntimeError):
            self.load()

    def test_invalid_version_cannot_escape_the_freeze_root(self):
        for version in ("../elsewhere", "v1/child", "", "..", "v1\\child"):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                self.freeze(version)
        self.assertFalse(self.service.clean_validation_root.exists())

    def test_feedback_arriving_during_freeze_prevents_activation(self):
        self.freeze("v1")
        active_path = self.service.clean_validation_root / "active.json"
        active_before = active_path.read_bytes()
        original_build = self.service._build_clean_validation_split
        calls = 0

        def build_with_concurrent_feedback(task_id, target_id, *, reviewed_rows):
            nonlocal calls
            split = original_build(task_id, target_id, reviewed_rows=reviewed_rows)
            calls += 1
            if calls == 1:
                with self.service.connect() as connection:
                    connection.execute("INSERT INTO annotation_events VALUES (?,?)", ("session", 7))
            return split

        with mock.patch.object(self.service, "_build_clean_validation_split", side_effect=build_with_concurrent_feedback):
            with self.assertRaisesRegex(RuntimeError, "historical feedback"):
                self.freeze("v2")
        self.assertEqual(active_path.read_bytes(), active_before)
        # The late row was never subtracted from a target-specific rebuild.
        self.assertEqual(self.service.build_calls[-2:], [
            ("attribute", frozenset({6})),
            ("joint", frozenset({6})),
        ])

    def test_image_order_drift_is_rejected(self):
        self.freeze()
        self.service.data.image_ids[4], self.service.data.image_ids[7] = self.service.data.image_ids[7], self.service.data.image_ids[4]
        with self.assertRaises((RuntimeError, ValueError)):
            self.load()

    def test_ground_truth_drift_is_rejected(self):
        self.freeze()
        self.service.truth[7, 1] ^= 1
        with self.assertRaises((RuntimeError, ValueError)):
            self.load()

    def test_partition_mask_drift_is_rejected(self):
        self.freeze()
        changed = bytearray(self.service.data.validation_mask)
        changed[7] = 0
        self.service.data.validation_mask = bytes(changed)
        with self.assertRaises((RuntimeError, ValueError)):
            self.load()

    def test_original_supervision_drift_is_rejected(self):
        self.freeze()
        self.service.original_rows["attribute"].append(7)
        with self.assertRaises((RuntimeError, ValueError)):
            self.load()

    def test_baseline_source_drift_is_rejected(self):
        self.freeze()
        with mock.patch.object(FROZEN, "baseline_source", return_value={"toyBaseline": "changed-v2"}):
            with self.assertRaises((RuntimeError, ValueError)):
                self.load()


if __name__ == "__main__":
    unittest.main()
