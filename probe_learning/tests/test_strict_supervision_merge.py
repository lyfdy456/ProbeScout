import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_probebank_batch as runner  # noqa: E402


class StrictSupervisionMergeTest(unittest.TestCase):
    attrs = ["animal", "being sheared", "outdoors"]

    @staticmethod
    def _write_rows(path: Path, rows: list[dict]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_reused_and_new_rows_form_one_complete_supervision_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reused = self._write_rows(root / "historical_labels.json", [
                {"image": "old.jpg", "animal": 1, "being sheared": 1, "outdoors": 0},
                {"image": "overlap.jpg", "animal": 0, "being sheared": 0, "outdoors": 1},
            ])
            new = self._write_rows(root / "round_results.jsonl", [
                {"image": "overlap.jpg", "animal": 1, "being sheared": 1, "outdoors": 1},
                {"image": "new.jpg", "animal": 1, "being sheared": 0, "outdoors": 1},
            ])
            audit_path = root / "audit.json"
            selected, labels_map, labels, digest, audit = runner.resolve_training_supervision(
                selected=["old.jpg", "overlap.jpg", "new.jpg"],
                reused_sources=[reused],
                new_sources=[new],
                vqa_to_key=lambda value: value,
                attrs=self.attrs,
                audit_path=audit_path,
                selected_manifest=None,
            )

            self.assertEqual(selected, ["old.jpg", "overlap.jpg", "new.jpg"])
            self.assertEqual(len(labels), 3)
            self.assertEqual(labels_map["overlap.jpg"]["animal"], 1)
            self.assertEqual(audit["matched_count"], 3)
            self.assertEqual(audit["reused_label_count"], 1)
            self.assertEqual(audit["new_label_count"], 2)
            self.assertEqual(audit["overlap_label_count"], 1)
            self.assertEqual(audit["missing_count"], 0)
            self.assertEqual(audit["joint_counts"], {"positive": 1, "negative": 2})
            self.assertEqual(audit["supervision_hash"], digest)
            self.assertTrue(audit_path.is_file())

            # Resume/fusion resolve the exact same full supervision identity.
            _, _, _, digest_again, _ = runner.resolve_training_supervision(
                selected=["old.jpg", "overlap.jpg", "new.jpg"],
                reused_sources=[reused],
                new_sources=[new],
                vqa_to_key=lambda value: value,
                attrs=self.attrs,
                audit_path=root / "audit_again.json",
                selected_manifest=None,
            )
            self.assertEqual(digest, digest_again)

    def test_partial_new_row_can_reuse_historical_attribute_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reused = self._write_rows(root / "historical_labels.json", [
                {"image": "mixed.jpg", "animal": 1, "being sheared": 0, "outdoors": 1},
            ])
            new = self._write_rows(root / "round_results.jsonl", [
                {"image": "mixed.jpg", "animal": None, "being sheared": 1, "outdoors": None},
            ])
            _, labels_map, _, _, audit = runner.resolve_training_supervision(
                selected=["mixed.jpg"],
                reused_sources=[reused],
                new_sources=[new],
                vqa_to_key=lambda value: value,
                attrs=self.attrs,
                audit_path=root / "audit.json",
                selected_manifest=None,
            )
            self.assertEqual(
                labels_map["mixed.jpg"],
                {"animal": 1, "being sheared": 1, "outdoors": 1},
            )
            self.assertEqual(audit["new_label_count"], 1)
            self.assertEqual(audit["overlap_label_count"], 1)

    def test_missing_selected_image_is_listed_and_never_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reused = self._write_rows(root / "historical_labels.json", [
                {"image": "present.jpg", "animal": 1, "being sheared": 0, "outdoors": 1},
            ])
            audit_path = root / "audit.json"
            with self.assertRaisesRegex(RuntimeError, "missing.jpg"):
                runner.resolve_training_supervision(
                    selected=["present.jpg", "missing.jpg"],
                    reused_sources=[reused],
                    new_sources=[],
                    vqa_to_key=lambda value: value,
                    attrs=self.attrs,
                    audit_path=audit_path,
                    selected_manifest=None,
                )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["selected_count_target"], 2)
            self.assertEqual(audit["matched_count"], 1)
            self.assertEqual(audit["missing_count"], 1)
            self.assertEqual(audit["missing_paths"], ["missing.jpg"])
            self.assertIsNone(audit["supervision_hash"])


if __name__ == "__main__":
    unittest.main()
