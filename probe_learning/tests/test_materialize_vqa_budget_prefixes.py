import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import materialize_vqa_budget_prefixes as budget_prefixes  # noqa: E402


STAGE = "iterative_vqa_100_50_budget500_fixture"


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BudgetPrefixFixture:
    def __init__(
        self,
        root: Path,
        *,
        role_shortage: bool = False,
        round_count: int = 9,
        snapshot_layout: bool = False,
        state_status: str = "completed",
    ):
        self.root = root
        self.task_root = root / "source_task"
        self.qa_root = (
            self.task_root / "supervision" / STAGE
            if snapshot_layout
            else self.task_root / "qa" / STAGE
        )
        self.state_path = (
            None
            if snapshot_layout
            else self.task_root / "split" / STAGE / "round_state.json"
        )
        self.output_root = root / "experiment"
        self.manifest_path = root / "tasks.json"

        query = [f"query_{index:02d}.jpg" for index in range(70)]
        coverage = [f"coverage_{index:02d}.jpg" for index in range(15)]
        diversity = [f"diversity_{index:02d}.jpg" for index in range(15)]
        round0_train = query + coverage + diversity
        roles = {
            "query_near": query[:30] if role_shortage else query,
            "attribute_text_coverage": coverage[:4] if role_shortage else coverage,
            "mmr_diversity": diversity[:3] if role_shortage else diversity,
        }

        write_json(self.qa_root / "manifest.json", {
            "version": STAGE,
            "budget": {
                "initial": 100,
                "round": 50,
                "train": 40,
                "audit": 10,
                "max": 500,
                "default_stop": 300,
            },
        })
        state_rounds = []
        for index in range(round_count):
            if index == 0:
                train = round0_train
                audit = []
                round_roles = roles
            else:
                train = [f"round_{index:02d}_train_{item:02d}.jpg" for item in range(40)]
                audit = [f"round_{index:02d}_audit_{item:02d}.jpg" for item in range(10)]
                round_roles = {"fixture_role": train}
            record = {
                "round": index,
                "stage": STAGE,
                "complete": True,
                "train": train,
                "audit": audit,
                "roles": round_roles,
            }
            write_json(self.qa_root / f"round_{index:02d}" / "manifest.json", record)
            state_rounds.append({"round": index, "train": train, "audit": audit})
        if self.state_path is not None:
            write_json(self.state_path, {
                "stage": STAGE,
                "status": state_status,
                "rounds": state_rounds,
            })
        write_json(self.manifest_path, {
            "task_count": 1,
            "tasks": [{
                "order": 7,
                "dataset": "fixture",
                "task": "task_budget_curve",
                "task_root": str(self.task_root),
            }],
        })

    @property
    def task_id(self) -> str:
        return "007_fixture_task_budget_curve"

    def selection_path(self, budget: int) -> Path:
        return (
            self.output_root
            / f"budget_{budget:03d}"
            / "tasks"
            / self.task_id
            / "vqa"
            / "iterative"
            / "split"
            / "train_labeled_indices.json"
        )

    def provenance_path(self, budget: int) -> Path:
        return self.selection_path(budget).with_name("budget_prefix_provenance.json")

    def run(self):
        return budget_prefixes.materialize_budget_prefixes(
            self.manifest_path,
            STAGE,
            self.output_root,
            repo_root=self.root,
        )


class MaterializeVqaBudgetPrefixesTest(unittest.TestCase):
    def test_optional_order_filter_materializes_only_requested_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp))
            summary = budget_prefixes.materialize_budget_prefixes(
                fixture.manifest_path,
                STAGE,
                fixture.output_root,
                repo_root=fixture.root,
                orders=[7],
            )
            self.assertEqual(summary["task_count"], 1)
            with self.assertRaisesRegex(
                budget_prefixes.MaterializationError,
                "absent from manifest",
            ):
                budget_prefixes.materialize_budget_prefixes(
                    fixture.manifest_path,
                    STAGE,
                    fixture.output_root / "missing",
                    repo_root=fixture.root,
                    orders=[999],
                )

    def test_materializes_exact_nested_budget_selections_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp))
            summary = fixture.run()
            self.assertEqual(summary["task_count"], 1)
            self.assertEqual(summary["budgets"], [50, 100, 150, 200, 300, 500])
            self.assertEqual(summary["file_count"], 12)

            expected_counts = {50: 50, 100: 100, 150: 140, 200: 180, 300: 260, 500: 420}
            previous = set()
            for budget, expected_count in expected_counts.items():
                selected = read_json(fixture.selection_path(budget))
                self.assertEqual(len(selected), expected_count)
                self.assertEqual(len(selected), len(set(selected)))
                self.assertTrue(previous.issubset(selected))
                previous = set(selected)

            fifty = read_json(fixture.selection_path(50))
            self.assertEqual(
                fifty,
                [f"query_{index:02d}.jpg" for index in range(35)]
                + [f"coverage_{index:02d}.jpg" for index in range(8)]
                + [f"diversity_{index:02d}.jpg" for index in range(7)],
            )

            provenance = read_json(fixture.provenance_path(500))
            self.assertEqual(provenance["logical_budget"], 500)
            self.assertEqual(provenance["materialized"]["train_count"], 420)
            self.assertEqual(provenance["materialized"]["audit_count"], 80)
            self.assertEqual(provenance["materialized"]["logical_budget_shortfall"], 0)
            self.assertEqual(provenance["materialized"]["nested_from_budget"], 300)
            self.assertEqual(provenance["source"]["source_round"], 8)
            self.assertEqual(len(provenance["source"]["cumulative_round_manifest_sha256"]), 9)
            self.assertFalse(provenance["ground_truth_read"])

    def test_round0_role_shortages_use_deterministic_round_order_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp), role_shortage=True)
            fixture.run()
            selected = read_json(fixture.selection_path(50))
            expected_roles = (
                [f"query_{index:02d}.jpg" for index in range(30)]
                + [f"coverage_{index:02d}.jpg" for index in range(4)]
                + [f"diversity_{index:02d}.jpg" for index in range(3)]
            )
            self.assertEqual(selected[:37], expected_roles)
            self.assertEqual(selected[37:], [f"query_{index:02d}.jpg" for index in range(30, 43)])
            provenance = read_json(fixture.provenance_path(50))
            self.assertEqual(provenance["materialized"]["selected_by_role"], {
                "query_near": 30,
                "attribute_text_coverage": 4,
                "mmr_diversity": 3,
            })
            self.assertEqual(provenance["materialized"]["round0_order_backfill_count"], 13)

    def test_supervision_snapshot_layout_without_round_state_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp), snapshot_layout=True)
            fixture.run()
            provenance = read_json(fixture.provenance_path(300))
            self.assertEqual(provenance["source"]["layout"], "supervision-snapshot")
            self.assertIsNone(provenance["source"]["round_state"])
            self.assertIsNone(provenance["source"]["round_state_sha256"])
            self.assertEqual(len(read_json(fixture.selection_path(300))), 260)

    def test_missing_required_source_round_fails_before_any_output_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp), round_count=8)
            with self.assertRaisesRegex(budget_prefixes.MaterializationError, "round_08"):
                fixture.run()
            self.assertFalse(fixture.output_root.exists())

    def test_active_acquisition_can_materialize_only_committed_budget_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(
                Path(tmp), round_count=3, state_status="awaiting_vqa"
            )
            summary = budget_prefixes.materialize_budget_prefixes(
                fixture.manifest_path,
                STAGE,
                fixture.output_root,
                repo_root=fixture.root,
                budgets=(50, 100, 150, 200),
            )
            self.assertEqual(summary["budgets"], [50, 100, 150, 200])
            self.assertEqual(summary["file_count"], 8)
            self.assertEqual(len(read_json(fixture.selection_path(200))), 180)
            self.assertFalse(fixture.selection_path(300).exists())

    def test_active_acquisition_cannot_materialize_full_declared_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp), state_status="awaiting_vqa")
            with self.assertRaisesRegex(
                budget_prefixes.MaterializationError,
                "not terminal for full-trajectory materialization",
            ):
                fixture.run()
            self.assertFalse(fixture.output_root.exists())

    def test_partial_budget_list_rejects_noncanonical_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp))
            with self.assertRaisesRegex(
                budget_prefixes.MaterializationError, "unsupported logical budgets"
            ):
                budget_prefixes.materialize_budget_prefixes(
                    fixture.manifest_path,
                    STAGE,
                    fixture.output_root,
                    repo_root=fixture.root,
                    budgets=(50, 125),
                )

    def test_duplicate_source_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp))
            path = fixture.qa_root / "round_01" / "manifest.json"
            manifest = read_json(path)
            manifest["train"].append(manifest["train"][0])
            write_json(path, manifest)
            with self.assertRaisesRegex(budget_prefixes.MaterializationError, "duplicate image IDs"):
                fixture.run()
            self.assertFalse(fixture.output_root.exists())

    def test_ground_truth_and_task_configuration_are_never_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp))
            # If the materializer accidentally discovers either file, parsing fails.
            (fixture.task_root / "ground_truth.json").write_text("not-json", encoding="utf-8")
            (fixture.task_root / "task.json").write_text("not-json", encoding="utf-8")
            opened = []
            original = budget_prefixes._read_json

            def guarded(path):
                opened.append(Path(path))
                if Path(path).name in {"ground_truth.json", "task.json"}:
                    raise AssertionError("ground truth/configuration must not be read")
                return original(Path(path))

            with mock.patch.object(budget_prefixes, "_read_json", side_effect=guarded):
                fixture.run()
            self.assertFalse(any(path.name == "ground_truth.json" for path in opened))
            self.assertFalse(any(path.name == "task.json" for path in opened))

    def test_idempotent_rerun_is_allowed_but_different_outputs_need_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = BudgetPrefixFixture(Path(tmp))
            fixture.run()
            fixture.run()
            write_json(fixture.selection_path(50), ["tampered.jpg"])
            with self.assertRaisesRegex(budget_prefixes.MaterializationError, "--overwrite"):
                fixture.run()


if __name__ == "__main__":
    unittest.main()
