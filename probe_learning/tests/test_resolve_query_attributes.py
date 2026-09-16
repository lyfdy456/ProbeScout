import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import resolve_query_attributes as resolver  # noqa: E402


class ResolveQueryAttributesTest(unittest.TestCase):
    @staticmethod
    def _task(root: Path, *, aliases=None, canonical=None, gt=None) -> Path:
        task = root / "task_example"
        task.mkdir(parents=True)
        payload = {
            "dataset": "celeba",
            "task_id": "task_example",
            "attributes": ["Eyeglasses", "Bangs"],
        }
        if aliases is not None:
            payload["attribute_aliases"] = aliases
        (task / "task.json").write_text(json.dumps(payload), encoding="utf-8")
        if canonical is not None:
            (task / "attributes.txt").write_text(canonical, encoding="utf-8")
        if gt is not None:
            (task / "gt_attribute.txt").write_text(gt, encoding="utf-8")
        return task

    def test_canonical_attributes_win_and_gt_is_never_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(
                root,
                canonical="{eyeglasses, bangs}",
                gt="THIS MUST NEVER BE PARSED",
            )
            fake_vqa = root / "query_vqa.txt"
            fake_vqa.write_text("{wrong, values}", encoding="utf-8")

            audit, audit_path = resolver.resolve_task_attributes(
                task,
                root / "run",
                query_vqa_attributes=fake_vqa,
                allow_api=False,
            )

            self.assertEqual(audit["validation"]["status"], "passed")
            self.assertEqual(audit["source"]["kind"], "canonical_attributes_txt")
            self.assertEqual(audit["mapped_attributes"], ["Eyeglasses", "Bangs"])
            self.assertFalse(audit["api_calls"]["invoked_this_run"])
            self.assertFalse(audit["gt_attribute_reference"]["consumed"])
            self.assertTrue(audit_path.is_file())
            self.assertEqual(
                (root / "run" / "resolved_attributes.txt").read_text(encoding="utf-8"),
                "{Eyeglasses, Bangs}\n",
            )

    def test_explicit_query_vqa_output_uses_declared_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(
                root,
                aliases={"Eyeglasses": ["wearing eyeglasses"], "Bangs": ["fringe"]},
            )
            discovered = root / "query_vqa.txt"
            discovered.write_text("{Wearing eyeglasses, Fringe}", encoding="utf-8")

            audit, _ = resolver.resolve_task_attributes(
                task,
                root / "run",
                query_vqa_attributes=discovered,
                allow_api=False,
            )

            self.assertEqual(audit["validation"]["status"], "passed")
            self.assertEqual(audit["source"]["kind"], "query_vqa_explicit_output")
            self.assertEqual(
                [row["match_type"] for row in audit["mappings"]],
                ["declared_alias", "declared_alias"],
            )
            self.assertFalse(audit["api_calls"]["invoked_this_run"])

    def test_existing_query_vqa_output_is_reused_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(root)
            (task / "qa").mkdir()
            (task / "qa" / "attributes.txt").write_text(
                "{Eyeglasses, Bangs}", encoding="utf-8"
            )

            audit, _ = resolver.resolve_task_attributes(
                task,
                root / "run",
                allow_api=False,
            )

            self.assertEqual(audit["validation"]["status"], "passed")
            self.assertEqual(audit["source"]["kind"], "query_vqa_existing_output")
            self.assertFalse(audit["api_calls"]["invoked_this_run"])

    def test_missing_or_extra_discovered_attribute_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(root)
            discovered = root / "query_vqa.txt"
            discovered.write_text("{Eyeglasses, Smiling}", encoding="utf-8")

            audit, audit_path = resolver.resolve_task_attributes(
                task,
                root / "run",
                query_vqa_attributes=discovered,
                allow_api=False,
            )

            self.assertEqual(audit["validation"]["status"], "failed")
            self.assertEqual(
                set(audit["validation"]["reason_codes"]),
                {"unmapped_discovered_attributes", "missing_modeled_attributes"},
            )
            self.assertEqual(audit["unmapped_discovered_attributes"], ["Smiling"])
            self.assertEqual(audit["missing_modeled_attributes"], ["Bangs"])
            self.assertFalse((root / "run" / "resolved_attributes.txt").exists())
            self.assertTrue(audit_path.is_file())

    def test_gt_reference_is_not_fallback_when_api_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(root, gt="Eyeglasses\nBangs\n")

            audit, _ = resolver.resolve_task_attributes(
                task,
                root / "run",
                reuse_existing=False,
                allow_api=False,
            )

            self.assertEqual(audit["validation"]["status"], "failed")
            self.assertEqual(
                audit["validation"]["reason_codes"],
                ["query_vqa_required_but_api_disabled"],
            )
            self.assertEqual(audit["raw_discovered_attributes"], [])
            self.assertFalse(audit["gt_attribute_reference"]["consumed"])
            self.assertFalse((root / "run" / "resolved_attributes.txt").exists())

    def test_gt_reference_cannot_be_passed_as_explicit_query_vqa_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(root, gt="Eyeglasses\nBangs\n")

            audit, _ = resolver.resolve_task_attributes(
                task,
                root / "run",
                query_vqa_attributes=task / "gt_attribute.txt",
                allow_api=False,
            )

            self.assertEqual(audit["validation"]["status"], "failed")
            self.assertEqual(
                audit["validation"]["reason_codes"],
                ["gt_attribute_reference_forbidden"],
            )
            self.assertEqual(audit["raw_discovered_attributes"], [])
            self.assertFalse(audit["gt_attribute_reference"]["consumed"])
            self.assertFalse((root / "run" / "resolved_attributes.txt").exists())

    def test_alias_collision_is_ambiguous_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._task(
                root,
                aliases={"Eyeglasses": ["face feature"], "Bangs": ["face feature"]},
            )
            discovered = root / "query_vqa.txt"
            discovered.write_text("{face feature, Bangs}", encoding="utf-8")

            audit, _ = resolver.resolve_task_attributes(
                task,
                root / "run",
                query_vqa_attributes=discovered,
                allow_api=False,
            )

            self.assertEqual(audit["validation"]["status"], "failed")
            self.assertIn("ambiguous_discovered_attributes", audit["validation"]["reason_codes"])
            self.assertIn("missing_modeled_attributes", audit["validation"]["reason_codes"])
            self.assertFalse((root / "run" / "resolved_attributes.txt").exists())


if __name__ == "__main__":
    unittest.main()
