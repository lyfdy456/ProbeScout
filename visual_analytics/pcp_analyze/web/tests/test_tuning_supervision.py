from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


WEB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WEB_ROOT / "scripts" / "tuning_supervision.py"
SPEC = importlib.util.spec_from_file_location("pcp_tuning_supervision", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SUPERVISION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUPERVISION
SPEC.loader.exec_module(SUPERVISION)


class ToyAdapter:
    dataset = "toy"
    task = "task_example"
    key_slugs = ("attr_a", "attr_b")

    def __init__(self, task_root: Path, gallery: list[str]):
        self.task_root = task_root
        self.gallery = gallery
        self.gt_by_attr = {"Attribute A": {}, "Attribute B": {}}
        self.vqa_to_key = lambda value: value


def fake_modules(
    gallery: list[str],
    labels: dict[str, dict[str, int]],
    supervision_hash: str = "strict-hash",
):
    calls: list[dict] = []

    def iterative_selected_manifest(_task_out, canonical_dir):
        return canonical_dir / "train_labeled_indices.json"

    def resolve_training_supervision(**kwargs):
        calls.append(kwargs)
        selected = json.loads(Path(kwargs["selected_manifest"]).read_text(encoding="utf-8"))
        label_rows = [{"image": image_id, **labels[image_id]} for image_id in selected]
        audit = {
            "status": "valid",
            "matched_count": len(selected),
            "missing_count": 0,
            "duplicate_selected_count": 0,
            "attribute_counts": {},
            "joint_counts": {},
        }
        return selected, labels, label_rows, supervision_hash, audit

    bank = types.SimpleNamespace(
        iterative_selected_manifest=iterative_selected_manifest,
        resolve_training_supervision=resolve_training_supervision,
        legacy_supervision_hash=lambda _selected, _labels, _attrs: "legacy-hash",
        database_paths=lambda adapter: list(adapter.gallery),
    )
    harness = types.SimpleNamespace(
        discover_task_vqa_sources=lambda _task_root: [],
    )
    return bank, harness, calls


class OriginalDevelopmentSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gallery = [f"gallery/{index}.jpg" for index in range(6)]
        self.task_root = self.root / "dataset" / "tasks" / "toy" / "task_example"
        supervision_dir = self.task_root / "supervision" / "iterative"
        supervision_dir.mkdir(parents=True)
        self.selected = self.gallery[:5]
        (supervision_dir / "train_labeled_indices.json").write_text(
            json.dumps(self.selected), encoding="utf-8"
        )
        self.labels = {
            self.gallery[0]: {"Attribute A": 1, "Attribute B": 1},
            self.gallery[1]: {"Attribute A": 1, "Attribute B": 0},
            self.gallery[2]: {"Attribute A": 0, "Attribute B": 1},
            self.gallery[3]: {"Attribute A": 0, "Attribute B": 0},
            self.gallery[4]: {"Attribute A": 1, "Attribute B": 1},
        }
        self.adapter = ToyAdapter(self.task_root, self.gallery)
        self.task_config = {"order": 1, "dataset": "toy", "task": "task_example"}
        self.development = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8)

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, target, *, expected="strict-hash", recovered="strict-hash"):
        bank, harness, calls = fake_modules(self.gallery, self.labels, recovered)
        with mock.patch.object(
            SUPERVISION, "_load_probebank_modules", return_value=(bank, harness)
        ):
            result = SUPERVISION.load_original_development_supervision(
                self.root,
                self.task_config,
                self.adapter,
                self.gallery,
                self.development,
                target,
                expected_supervision_hash=expected,
            )
        return result, calls

    def test_attribute_target_uses_only_original_rows_in_development(self):
        result, calls = self.load("attr_a")
        np.testing.assert_array_equal(result.selected_indices, np.asarray([0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(result.selected_labels, np.asarray([1, 1, 0, 0, 1]))
        np.testing.assert_array_equal(result.indices, np.asarray([0, 2, 4]))
        np.testing.assert_array_equal(result.labels, np.asarray([1, 0, 1]))
        self.assertFalse(result.selected_indices.flags.writeable)
        self.assertFalse(result.selected_labels.flags.writeable)
        self.assertFalse(result.indices.flags.writeable)
        self.assertFalse(result.labels.flags.writeable)
        self.assertEqual(result.audit["originalSupervisionCount"], 5)
        self.assertEqual(result.audit["originalPositiveCount"], 3)
        self.assertEqual(result.audit["originalNegativeCount"], 2)
        self.assertEqual(result.audit["developmentCount"], 3)
        self.assertEqual(result.audit["excludedNonDevelopmentCount"], 2)
        self.assertEqual(result.audit["positiveCount"], 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["attrs"], ["Attribute A", "Attribute B"])

    def test_joint_target_is_logical_and_of_all_attributes(self):
        result, _ = self.load(None)
        np.testing.assert_array_equal(result.selected_indices, np.asarray([0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(result.selected_labels, np.asarray([1, 0, 0, 0, 1]))
        np.testing.assert_array_equal(result.indices, np.asarray([0, 2, 4]))
        np.testing.assert_array_equal(result.labels, np.asarray([1, 0, 1]))
        self.assertEqual(result.audit["target"], "joint")
        self.assertEqual(result.audit["targetRule"], "all modeled attributes equal 1")

    def test_accepts_raw_uint8_mask_bytes(self):
        original = self.development
        self.development = original.tobytes()
        try:
            result, _ = self.load("Attribute B")
        finally:
            self.development = original
        np.testing.assert_array_equal(result.indices, np.asarray([0, 2, 4]))
        np.testing.assert_array_equal(result.labels, np.asarray([1, 1, 1]))

    def test_probe_validation_split_exactly_replays_seed_zero_membership(self):
        from sklearn.model_selection import train_test_split

        rows = np.arange(100, 120, dtype=np.int64)
        labels = np.asarray([0, 1] * 10, dtype=np.uint8)
        original = SUPERVISION.OriginalSupervision(
            indices=rows,
            labels=labels,
            selected_indices=rows,
            selected_labels=labels,
            audit={"recoveredSupervisionHash": "strict-hash"},
        )
        split = SUPERVISION.build_probe_validation_split(
            original,
            target_id="attr_a",
            target_kind="attribute",
        )
        positions = np.arange(rows.size)
        expected_fit, expected_validation = train_test_split(
            positions,
            test_size=0.2,
            random_state=0,
            stratify=labels,
        )
        np.testing.assert_array_equal(split.fit_indices, rows[np.sort(expected_fit)])
        np.testing.assert_array_equal(
            split.validation_indices,
            rows[np.sort(expected_validation)],
        )
        self.assertEqual(split.audit["replayKind"], "exact-attribute-seed0")
        self.assertEqual(split.audit["fitCount"], 16)
        self.assertEqual(split.audit["validationCount"], 4)
        self.assertEqual(split.audit["validationPositiveCount"], 2)
        self.assertFalse(split.validation_indices.flags.writeable)
        repeated = SUPERVISION.build_probe_validation_split(
            original,
            target_id="attr_a",
            target_kind="attribute",
        )
        self.assertEqual(
            split.audit["validationFingerprint"],
            repeated.audit["validationFingerprint"],
        )

    def test_probe_validation_split_marks_joint_as_derived_and_is_target_specific(self):
        rows = np.arange(20, dtype=np.int64)
        first_labels = np.asarray([0, 1] * 10, dtype=np.uint8)
        second_labels = np.asarray(([0] * 10) + ([1] * 10), dtype=np.uint8)

        def original(labels):
            return SUPERVISION.OriginalSupervision(
                indices=rows,
                labels=labels,
                selected_indices=rows,
                selected_labels=labels,
                audit={},
            )

        first = SUPERVISION.build_probe_validation_split(
            original(first_labels), target_id="first", target_kind="attribute"
        )
        second = SUPERVISION.build_probe_validation_split(
            original(second_labels), target_id="second", target_kind="attribute"
        )
        joint = SUPERVISION.build_probe_validation_split(
            original(first_labels), target_id="joint", target_kind="derived"
        )
        self.assertNotEqual(
            set(first.validation_indices.tolist()),
            set(second.validation_indices.tolist()),
        )
        self.assertEqual(joint.audit["replayKind"], "derived-joint-probe-style")
        self.assertNotEqual(
            first.audit["validationFingerprint"],
            joint.audit["validationFingerprint"],
        )

    def test_rejects_recovered_hash_that_differs_from_probebank(self):
        with self.assertRaisesRegex(ValueError, "supervision_hash"):
            self.load("Attribute A", expected="cache-hash", recovered="other-hash")

    def test_accepts_mixed_strict_and_legacy_cache_hashes_only_after_full_audit(self):
        strict_hash = "strict-hash"
        legacy_hash = "legacy-hash"
        selected = list(self.selected)
        bank, _, _ = fake_modules(self.gallery, self.labels, strict_hash)
        bank.stable_hash = lambda values: "gallery-hash" if values == self.gallery else "records-hash"
        metadata_by_method = {
            "old": legacy_hash,
            "new": strict_hash,
        }

        def cache_dir(root, _dataset, _task, _stage, method, attribute):
            path = Path(root) / method / attribute
            path.mkdir(parents=True, exist_ok=True)
            values = [self.labels[image_id][attribute] for image_id in selected]
            (path / "metadata.json").write_text(json.dumps({
                "dataset": "toy",
                "task": "task_example",
                "supervision_stage": "iterative",
                "method": method,
                "canonical_attribute": attribute,
                "records_hash": "gallery-hash",
                "score_length": len(self.gallery),
                "supervision_hash": metadata_by_method[method],
                "training_records_hash": "records-hash",
                "training_count": len(selected),
                "positive_count": sum(values),
                "negative_count": len(values) - sum(values),
            }), encoding="utf-8")
            return path

        bank.cache_dir = cache_dir
        suite_path = self.root / SUPERVISION.DEFAULT_SUITE
        suite_path.parent.mkdir(parents=True)
        suite_path.write_text(json.dumps({
            "learned_methods": ["old", "new"],
            "probe_source_roots": {"__default__": "cache"},
        }), encoding="utf-8")
        audit = SUPERVISION._cache_supervision_identity(
            self.root,
            bank,
            self.task_config,
            ("Attribute A", "Attribute B"),
            "iterative",
            tuple(self.gallery),
            selected,
            self.labels,
            strict_hash,
            legacy_hash,
        )
        self.assertEqual(audit["metadataCount"], 4)
        self.assertEqual(audit["variants"][strict_hash]["schema"], "complete_selected_labels_v1")
        self.assertEqual(audit["variants"][legacy_hash]["schema"], "legacy_pre_schema")

    def test_rejects_unknown_cache_hash_even_when_counts_match(self):
        bank, _, _ = fake_modules(self.gallery, self.labels)
        bank.stable_hash = lambda values: "gallery-hash" if values == self.gallery else "records-hash"

        def cache_dir(root, _dataset, _task, _stage, method, attribute):
            path = Path(root) / method / attribute
            path.mkdir(parents=True, exist_ok=True)
            values = [self.labels[image_id][attribute] for image_id in self.selected]
            (path / "metadata.json").write_text(json.dumps({
                "dataset": "toy", "task": "task_example", "supervision_stage": "iterative",
                "method": method, "canonical_attribute": attribute,
                "records_hash": "gallery-hash", "score_length": len(self.gallery),
                "supervision_hash": "unrelated-hash", "training_records_hash": "records-hash",
                "training_count": len(self.selected), "positive_count": sum(values),
                "negative_count": len(values) - sum(values),
            }), encoding="utf-8")
            return path

        bank.cache_dir = cache_dir
        suite_path = self.root / SUPERVISION.DEFAULT_SUITE
        suite_path.parent.mkdir(parents=True)
        suite_path.write_text(json.dumps({
            "learned_methods": ["learner"],
            "probe_source_roots": {"__default__": "cache"},
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "neither the strict nor legacy"):
            SUPERVISION._cache_supervision_identity(
                self.root, bank, self.task_config,
                ("Attribute A", "Attribute B"), "iterative", tuple(self.gallery),
                self.selected, self.labels, "strict-hash", "legacy-hash",
            )

    def test_replays_run_local_supervision_only_from_matching_saved_audit(self):
        task_assets = (
            self.root
            / SUPERVISION.REMOTE_ASSETS_ROOT
            / "001_toy_task_example"
        )
        training_assets = task_assets / "vqa" / "iterative"
        selected_manifest = training_assets / "split" / "train_labeled_indices.json"
        selected_manifest.parent.mkdir(parents=True)
        selected_manifest.write_text(json.dumps(self.selected), encoding="utf-8")
        new_source = training_assets / "qa" / "round_01" / "results.jsonl"
        new_source.parent.mkdir(parents=True)
        new_source.write_text("{}\n", encoding="utf-8")
        reused_source = self.task_root / "supervision" / "iterative" / "labels.jsonl"
        reused_source.write_text("{}\n", encoding="utf-8")
        saved_audit = task_assets / "supervision_audit.json"
        saved_audit.write_text(json.dumps({
            "status": "valid",
            "selected_manifest": str(selected_manifest),
            "selected_count_target": len(self.selected),
            "selected_count_raw": len(self.selected),
            "matched_count": len(self.selected),
            "missing_count": 0,
            "duplicate_selected_count": 0,
            "attribute_counts": {},
            "joint_counts": {},
            "supervision_hash": "strict-hash",
            "reused_sources": [str(reused_source)],
            "new_sources": [str(new_source)],
        }), encoding="utf-8")

        bank, _harness, calls = fake_modules(self.gallery, self.labels)
        recovered = SUPERVISION._audited_source_supervision(
            self.root,
            bank,
            self.task_config,
            self.adapter,
            ("Attribute A", "Attribute B"),
            "iterative",
        )
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered[2], "strict-hash")
        self.assertEqual(recovered[6], saved_audit)
        self.assertEqual(calls[0]["reused_sources"], [reused_source])
        self.assertEqual(calls[0]["new_sources"], [new_source])

        payload = json.loads(saved_audit.read_text(encoding="utf-8"))
        payload["supervision_hash"] = "unrelated-hash"
        saved_audit.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not reproduce exactly"):
            SUPERVISION._audited_source_supervision(
                self.root,
                bank,
                self.task_config,
                self.adapter,
                ("Attribute A", "Attribute B"),
                "iterative",
            )

    def test_rejects_gallery_order_drift_before_loading_labels(self):
        bank, harness, calls = fake_modules(self.gallery, self.labels)
        drifted = list(self.gallery)
        drifted[0], drifted[1] = drifted[1], drifted[0]
        with mock.patch.object(
            SUPERVISION, "_load_probebank_modules", return_value=(bank, harness)
        ):
            with self.assertRaisesRegex(ValueError, "embedding order"):
                SUPERVISION.load_original_development_supervision(
                    self.root,
                    self.task_config,
                    self.adapter,
                    drifted,
                    self.development,
                    "Attribute A",
                    expected_supervision_hash="strict-hash",
                )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
