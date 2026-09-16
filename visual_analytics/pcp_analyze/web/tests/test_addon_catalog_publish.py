from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WEB_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

MODULE_PATH = SCRIPTS_ROOT / "publish_addon_task.py"
SPEC = importlib.util.spec_from_file_location("pcp_publish_addon_task", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PUBLISH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLISH
SPEC.loader.exec_module(PUBLISH)

CATALOG = sys.modules["addon_catalog"]


def core_catalog() -> dict:
    return {
        "schemaVersion": 2,
        "defaultDataset": "toy",
        "defaultTask": "001_toy_task_core",
        "taskCount": 1,
        "datasets": [
            {
                "id": "toy",
                "label": "Toy Dataset",
                "tasks": [
                    {
                        "id": "001_toy_task_core",
                        "label": "Protected core task",
                        "dataRoot": "/data/tasks/001_toy_task_core",
                    }
                ],
            }
        ],
    }


def addon_entry(*, label: str = "Addon task") -> dict:
    positives = {"attribute": 2, "joint": 1}
    return {
        "id": "090_toy_task_addon",
        "label": label,
        "description": "Attribute, and Joint retrieval targets",
        "dataRoot": "/data/tasks/090_toy_task_addon",
        "defaultRetrievalTarget": "joint",
        "declaredAttributes": ["Attribute"],
        "modeledRetrievalTargets": [{"id": "attribute", "label": "Attribute"}],
        "modeledTargetIds": ["attribute"],
        "catalogRole": "addon",
        "tuningSource": {
            "schemaVersion": 1,
            "manifest": "configs/experiments/addon.json",
            "suite": "configs/experiments/addon-suite.json",
            "oursFullRoot": "outputs/addon-run",
            "supervisionAudit": (
                "outputs/addon-run/tasks/090_toy_task_addon/supervision_audit.json"
            ),
            "probeStage": "iterative",
        },
        "rowCount": 10,
        "targetCount": 2,
        "queryBytes": 10,
        "initialDownloadBytes": 100,
        "bundleBytes": 200,
        "thumbnailBytes": 50,
        "visualEmbeddingBytes": 20,
        "developmentRowCount": 4,
        "developmentPositiveCounts": copy.deepcopy(positives),
        "validationRowCount": 2,
        "validationPositiveCounts": {"attribute": 1, "joint": 1},
        "testRowCount": 2,
        "testPositiveCounts": {"attribute": 1, "joint": 0},
    }


def fake_validation(public_root: Path):
    task_id = "090_toy_task_addon"
    bundle = public_root / "data" / "tasks" / task_id
    bundle.mkdir(parents=True, exist_ok=True)
    task = PUBLISH.FORMAL.TaskSpec(
        order=90,
        dataset="toy",
        task_name="task_addon",
        attributes=("Attribute",),
        supervision_stage="iterative",
        explicit_data_root=bundle,
    )
    audit = addon_entry()
    audit["retrievalTargets"] = [
        {"id": "attribute", "label": "Attribute", "kind": "attribute"},
        {"id": "joint", "label": "Joint", "kind": "derived"},
    ]
    audit.pop("id")
    audit.pop("label")
    audit.pop("description")
    audit.pop("dataRoot")
    audit.pop("defaultRetrievalTarget")
    audit.pop("declaredAttributes")
    audit.pop("modeledRetrievalTargets")
    audit.pop("modeledTargetIds")
    audit.pop("catalogRole")
    manifest = {
        "defaultRetrievalTarget": "joint",
        "retrievalTargets": [
            {
                "id": "attribute",
                "label": "Attribute",
                "kind": "attribute",
                "canonicalAttribute": "Attribute",
            },
            {"id": "joint", "label": "Joint", "kind": "derived"},
        ],
    }
    return task, audit, manifest


class AddonCatalogTests(unittest.TestCase):
    def test_canonical_supervision_mirror_is_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audited = root / "run-local.json"
            canonical = root / "canonical.json"
            audited.write_text('["b.jpg","a.jpg"]', encoding="utf-8")
            canonical.write_text(
                json.dumps(["b.jpg", "a.jpg"], indent=2) + "\n",
                encoding="utf-8",
            )
            PUBLISH._validate_equivalent_supervision_manifests(
                audited,
                canonical,
                matched_count=2,
            )
            canonical.write_text(
                json.dumps(["a.jpg", "b.jpg"]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                PUBLISH._validate_equivalent_supervision_manifests(
                    audited,
                    canonical,
                    matched_count=2,
                )

    def test_upsert_adds_and_replaces_only_marked_addons(self) -> None:
        original = core_catalog()
        protected = copy.deepcopy(original["datasets"][0]["tasks"][0])
        added = CATALOG.upsert_addon_entry(
            original,
            dataset_id="toy",
            dataset_label="Toy Dataset",
            entry=addon_entry(),
        )
        self.assertEqual(original["taskCount"], 1)
        self.assertEqual(added["taskCount"], 2)
        self.assertEqual(added["datasets"][0]["tasks"][0], protected)

        replaced = CATALOG.upsert_addon_entry(
            added,
            dataset_id="toy",
            dataset_label="Toy Dataset",
            entry=addon_entry(label="Updated addon"),
        )
        self.assertEqual(replaced["taskCount"], 2)
        self.assertEqual(replaced["datasets"][0]["tasks"][0], protected)
        self.assertEqual(replaced["datasets"][0]["tasks"][1]["label"], "Updated addon")

        collision = addon_entry()
        collision["id"] = "001_toy_task_core"
        collision["dataRoot"] = "/data/tasks/001_toy_task_core"
        collision["tuningSource"]["supervisionAudit"] = (
            "outputs/addon-run/tasks/001_toy_task_core/supervision_audit.json"
        )
        with self.assertRaisesRegex(ValueError, "protected catalog task"):
            CATALOG.upsert_addon_entry(
                original,
                dataset_id="toy",
                dataset_label="Toy Dataset",
                entry=collision,
            )

        invalid_default = addon_entry()
        invalid_default["defaultRetrievalTarget"] = "missing"
        with self.assertRaisesRegex(ValueError, "defaultRetrievalTarget is not modeled"):
            CATALOG.upsert_addon_entry(
                original,
                dataset_id="toy",
                dataset_label="Toy Dataset",
                entry=invalid_default,
            )

        invalid_root = addon_entry()
        invalid_root["dataRoot"] = "/data/tasks/different-task"
        with self.assertRaisesRegex(ValueError, "canonical task path"):
            CATALOG.upsert_addon_entry(
                original,
                dataset_id="toy",
                dataset_label="Toy Dataset",
                entry=invalid_root,
            )

        missing_tuning_source = addon_entry()
        missing_tuning_source.pop("tuningSource")
        with self.assertRaisesRegex(ValueError, "tuningSource schemaVersion"):
            CATALOG.upsert_addon_entry(
                original,
                dataset_id="toy",
                dataset_label="Toy Dataset",
                entry=missing_tuning_source,
            )

        unsafe_tuning_source = addon_entry()
        unsafe_tuning_source["tuningSource"]["suite"] = "../private-suite.json"
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            CATALOG.upsert_addon_entry(
                original,
                dataset_id="toy",
                dataset_label="Toy Dataset",
                entry=unsafe_tuning_source,
            )

    def test_formal_rebuild_preserves_only_marked_addons(self) -> None:
        formal = core_catalog()
        published = CATALOG.upsert_addon_entry(
            formal,
            dataset_id="toy",
            dataset_label="Toy Dataset",
            entry=addon_entry(),
        )
        rebuilt = CATALOG.merge_published_addons(formal, published)
        self.assertEqual(rebuilt, published)
        CATALOG.validate_formal_catalog_extension(formal, rebuilt)

        drifted = copy.deepcopy(rebuilt)
        drifted["datasets"][0]["tasks"][0]["label"] = "Changed core"
        with self.assertRaisesRegex(ValueError, "Formal task entry drifted"):
            CATALOG.validate_formal_catalog_extension(formal, drifted)

        unmarked = copy.deepcopy(formal)
        extra = copy.deepcopy(addon_entry())
        extra.pop("catalogRole")
        unmarked["datasets"][0]["tasks"].append(extra)
        unmarked["taskCount"] = 2
        with self.assertRaisesRegex(ValueError, "not a marked add-on"):
            CATALOG.merge_published_addons(formal, unmarked)

    def test_publish_is_atomic_and_validation_precedes_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_root = root / "public"
            catalog_path = public_root / "data" / "catalog.json"
            catalog_path.parent.mkdir(parents=True)
            catalog_path.write_text(json.dumps(core_catalog()), encoding="utf-8")
            task, audit, manifest = fake_validation(public_root)
            calls: list[str] = []

            def validator(**_kwargs):
                calls.append("validate")
                return task, copy.deepcopy(audit), copy.deepcopy(manifest)

            result = PUBLISH.publish_addon_task(
                bundle=task.data_root,
                label="Addon task",
                description=None,
                declared_attributes=[],
                dataset_label=None,
                catalog_path=catalog_path,
                public_root=public_root,
                dry_run=False,
                validator=validator,
            )
            self.assertEqual(calls, ["validate"])
            self.assertTrue(result["catalogUpdated"])
            committed = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(committed["taskCount"], 2)
            self.assertEqual(committed["datasets"][0]["tasks"][0], core_catalog()["datasets"][0]["tasks"][0])
            self.assertFalse(catalog_path.with_suffix(".json.lock").exists())
            self.assertEqual(list(catalog_path.parent.glob(".catalog.json.*.tmp")), [])

            before = catalog_path.read_bytes()
            dry_run = PUBLISH.publish_addon_task(
                bundle=task.data_root,
                label="Dry-run replacement",
                description=None,
                declared_attributes=[],
                dataset_label=None,
                catalog_path=catalog_path,
                public_root=public_root,
                dry_run=True,
                validator=validator,
            )
            self.assertFalse(dry_run["catalogUpdated"])
            self.assertEqual(catalog_path.read_bytes(), before)

            def failing_validator(**_kwargs):
                raise ValueError("strict bundle audit failed")

            with self.assertRaisesRegex(ValueError, "strict bundle audit failed"):
                PUBLISH.publish_addon_task(
                    bundle=task.data_root,
                    label="Must not publish",
                    description=None,
                    declared_attributes=[],
                    dataset_label=None,
                    catalog_path=catalog_path,
                    public_root=public_root,
                    dry_run=False,
                    validator=failing_validator,
                )
            self.assertEqual(catalog_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
