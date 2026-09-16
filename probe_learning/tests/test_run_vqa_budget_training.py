import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_vqa_budget_training as training  # noqa: E402


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class TrainingFixture:
    def __init__(self, root: Path, budget: int = 150):
        self.root = root
        self.budget = budget
        self.task = {
            "order": 7,
            "dataset": "fixture",
            "task": "task_curve",
            "scope": "addon13",
            "stages": ["iterative"],
        }
        self.manifest = write_json(root / "manifest.json", {
            "batch_id": "fixture_all1",
            "task_count": 1,
            "tasks": [self.task],
        })
        self.suite = write_json(root / "suite.json", {
            "suite_id": "fixture_suite",
            "fusion_recipes": [{
                "id": "paper_ours_full_selected8",
                "id_prefix": "all_probe_softgate",
                "gate_strategies": ["probe_level"],
                "joint_rules": ["product"],
            }],
        })
        self.run_root = root / f"budget_{budget:03d}"
        source_round, train_count, audit_count = training.expected_prefix_counts(budget)
        self.selected = [f"image_{index:03d}.jpg" for index in range(train_count)]
        self.selection = (
            self.run_root / "tasks" / training.task_id(self.task)
            / "vqa" / "iterative" / "split" / "train_labeled_indices.json"
        )
        self.provenance = self.selection.with_name("budget_prefix_provenance.json")
        write_json(self.selection, self.selected)
        write_json(self.provenance, {
            "schema_version": 1,
            "task": {
                "order": 7,
                "dataset": "fixture",
                "task": "task_curve",
                "output_id": training.task_id(self.task),
            },
            "acquisition_stage": "iterative_vqa_fixture_budget500",
            "logical_budget": budget,
            "materialized": {
                "train_count": train_count,
                "audit_count": audit_count,
                "observed_total_labeled_count": budget,
                "logical_budget_shortfall": 0,
                "train_selection_sha256": training.sequence_sha256(self.selected),
                "no_duplicate_train_ids": True,
                "train_audit_disjoint": True,
            },
            "source": {
                "source_round": source_round,
                "cumulative_round_manifest_sha256": ["a" * 64] * (source_round + 1),
                "protocol": {
                    "initial": 100,
                    "round": 50,
                    "train": 40,
                    "audit": 10,
                    "max": 500,
                },
            },
        })
        manifest_payload, tasks = training.load_manifest(self.manifest)
        suite_payload = training.read_json(self.suite, label="suite")
        self.contract, _ = training.build_budget_contract(
            budget,
            tasks=tasks,
            manifest=self.manifest,
            manifest_payload=manifest_payload,
            suite=self.suite,
            suite_payload=suite_payload,
            run_root=self.run_root,
            seeds=training.DEFAULT_SEEDS,
            epochs=100,
        )

    def materialize_completed_output(self):
        write_json(self.run_root / training.CONTRACT_NAME, self.contract)
        write_json(self.run_root / "run_state.json", {
            "status": "completed_requested_range",
            "suite_id": "fixture_suite",
            "suite_config": str(self.suite.resolve()),
            "orders": [7],
            "seeds": list(training.DEFAULT_SEEDS),
            "epochs": 100,
            "cross_task_probe_reuse": False,
        })
        supervision_hash = "b" * 64
        task_root = self.run_root / "tasks" / training.task_id(self.task)
        write_json(task_root / "summary.json", {
            "order": 7,
            "dataset": "fixture",
            "task": "task_curve",
            "suite_id": "fixture_suite",
            "stage_status": {"iterative": "completed"},
            "supervision_hashes": {"iterative": supervision_hash},
            "supervision_audits": {"iterative": {
                "status": "valid",
                "selected_manifest": str(self.selection.resolve()),
                "selected_count_target": len(self.selected),
                "matched_count": len(self.selected),
                "missing_count": 0,
                "duplicate_selected_count": 0,
                "supervision_hash": supervision_hash,
                "evaluation_boundary": {
                    "gallery_membership_valid": True,
                    "frozen_test_overlap_count": 0,
                },
            }},
        })
        write_json(task_root / "iterative_metrics.json", [
            {
                "stage": "iterative",
                "split": "test",
                "seed": seed,
                "method": "all_probe_softgate_probe_level_product",
                "joint_rule": "product",
                "metrics": {"joint": {"ap": 0.5 + seed / 100}},
                "supervision_hash": supervision_hash,
            }
            for seed in training.DEFAULT_SEEDS
        ])


class VqaBudgetTrainingTest(unittest.TestCase):
    def test_valid_completed_budget_is_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = TrainingFixture(Path(tmp))
            fixture.materialize_completed_output()
            audit = training.audit_budget_completion(
                fixture.run_root, [fixture.task], fixture.contract,
            )
            self.assertTrue(audit["complete"], audit["reasons"])

    def test_stale_prefix_invalidates_completed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = TrainingFixture(Path(tmp))
            fixture.materialize_completed_output()
            changed = list(fixture.selected)
            changed[-1] = "replacement.jpg"
            write_json(fixture.selection, changed)
            with self.assertRaisesRegex(training.TrainingContractError, "hash mismatch"):
                training.build_budget_contract(
                    fixture.budget,
                    tasks=[fixture.task],
                    manifest=fixture.manifest,
                    manifest_payload=training.load_manifest(fixture.manifest)[0],
                    suite=fixture.suite,
                    suite_payload=training.read_json(fixture.suite, label="suite"),
                    run_root=fixture.run_root,
                    seeds=training.DEFAULT_SEEDS,
                    epochs=100,
                )

    def test_missing_seed_and_test_overlap_invalidate_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = TrainingFixture(Path(tmp))
            fixture.materialize_completed_output()
            task_root = fixture.run_root / "tasks" / training.task_id(fixture.task)
            metrics_path = task_root / "iterative_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            write_json(metrics_path, metrics[:-1])
            summary_path = task_root / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["supervision_audits"]["iterative"]["evaluation_boundary"][
                "frozen_test_overlap_count"
            ] = 1
            write_json(summary_path, summary)
            audit = training.audit_budget_completion(
                fixture.run_root, [fixture.task], fixture.contract,
            )
            self.assertFalse(audit["complete"])
            joined = "\n".join(audit["reasons"])
            self.assertIn("overlaps Frozen Test", joined)
            self.assertIn("exactly seeds", joined)

    def test_grid_preflight_rejects_non_nested_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = TrainingFixture(root, budget=50)
            second_root = root / "second"
            second = TrainingFixture(second_root, budget=100)
            # Build a shared experiment layout and shared manifest/suite.
            target = root / "experiment"
            for fixture in (first, second):
                target_selection = (
                    target / f"budget_{fixture.budget:03d}" / "tasks"
                    / training.task_id(fixture.task) / "vqa" / "iterative" / "split"
                    / "train_labeled_indices.json"
                )
                target_provenance = target_selection.with_name("budget_prefix_provenance.json")
                write_json(target_selection, fixture.selected)
                write_json(
                    target_provenance,
                    json.loads(fixture.provenance.read_text(encoding="utf-8")),
                )
            # Replace all budget-100 IDs so budget 50 is not a subset, updating
            # provenance so only the cross-budget invariant catches the issue.
            selection100 = (
                target / "budget_100" / "tasks" / training.task_id(first.task)
                / "vqa" / "iterative" / "split" / "train_labeled_indices.json"
            )
            changed = [f"other_{index:03d}.jpg" for index in range(100)]
            write_json(selection100, changed)
            provenance100 = selection100.with_name("budget_prefix_provenance.json")
            provenance = json.loads(provenance100.read_text(encoding="utf-8"))
            provenance["materialized"]["train_selection_sha256"] = training.sequence_sha256(changed)
            write_json(provenance100, provenance)

            manifest_payload, tasks = training.load_manifest(first.manifest)
            suite_payload = training.read_json(first.suite, label="suite")
            with self.assertRaisesRegex(training.TrainingContractError, "not nested"):
                training.build_grid_contracts(
                    (50, 100),
                    tasks=tasks,
                    manifest=first.manifest,
                    manifest_payload=manifest_payload,
                    suite=first.suite,
                    suite_payload=suite_payload,
                    experiment_root=target,
                    seeds=training.DEFAULT_SEEDS,
                    epochs=100,
                )


if __name__ == "__main__":
    unittest.main()
