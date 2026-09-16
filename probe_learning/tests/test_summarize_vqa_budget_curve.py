import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_vqa_budget_curve as budget_curve  # noqa: E402


class VqaBudgetCurveTest(unittest.TestCase):
    def test_supervision_class_diagnostics_supports_legacy_missing_block(self):
        self.assertEqual(
            budget_curve.supervision_class_diagnostics({}),
            {
                "min_attr_positive": None,
                "min_attr_negative": None,
                "single_class_attribute_count": None,
                "insufficient_balance_attribute_count": None,
                "joint_positive": None,
                "joint_negative": None,
            },
        )

    def test_supervision_class_diagnostics_rejects_inconsistent_totals(self):
        summary = {
            "supervision_audits": {
                "iterative": {
                    "attribute_counts": {
                        "a": {"positive": 0, "negative": 10},
                        "b": {"positive": 3, "negative": 8},
                    },
                    "joint_counts": {"positive": 2, "negative": 8},
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "totals disagree"):
            budget_curve.supervision_class_diagnostics(summary)

    def test_insufficient_balance_uses_probebank_threshold_two(self):
        summary = {
            "supervision_audits": {
                "iterative": {
                    "attribute_counts": {
                        "one_positive": {"positive": 1, "negative": 9},
                        "balanced": {"positive": 2, "negative": 8},
                    },
                    "joint_counts": {"positive": 2, "negative": 8},
                }
            }
        }
        diagnostics = budget_curve.supervision_class_diagnostics(summary)
        self.assertEqual(diagnostics["single_class_attribute_count"], 0)
        self.assertEqual(diagnostics["insufficient_balance_attribute_count"], 1)
        self.assertEqual(budget_curve.PROBEBANK_MIN_CLASS_SAMPLES, 2)

    def test_logical_budget_comes_from_prefix_provenance_not_api_calls(self):
        with tempfile.TemporaryDirectory() as raw_temp:
            task_dir = Path(raw_temp)
            provenance = (
                task_dir / "vqa" / "iterative" / "split" /
                "budget_prefix_provenance.json"
            )
            provenance.parent.mkdir(parents=True)
            provenance.write_text(
                json.dumps(
                    {
                        "logical_budget": 150,
                        "materialized": {
                            "train_count": 140,
                            "audit_count": 10,
                            "observed_total_labeled_count": 150,
                            "logical_budget_shortfall": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                budget_curve.budget_counts_from_task(
                    task_dir, {"vqa_successful": 0}
                ),
                (150, 140, 10),
            )

    def test_partial_legacy_round_is_not_added_to_curve(self):
        with tempfile.TemporaryDirectory() as raw_temp:
            path = Path(raw_temp) / "legacy.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("order", "round", "split", "mean_ap"),
                )
                writer.writeheader()
                for round_index in (0, 1):
                    for order in (1, 2):
                        writer.writerow(
                            {
                                "order": order,
                                "round": round_index,
                                "split": "test",
                                "mean_ap": 0.5 + 0.1 * round_index,
                            }
                        )
                writer.writerow(
                    {"order": 1, "round": 2, "split": "test", "mean_ap": 0.9}
                )
            rows = budget_curve.load_legacy_curve("legacy", path)
            self.assertEqual([row["budget"] for row in rows], [100, 150])
            self.assertTrue(all(row["task_count"] == 2 for row in rows))

    def test_resolves_materialized_selected8_recipe_id(self):
        suite = {
            "fusion_recipes": [
                {
                    "id": "paper_ours_full_selected8",
                    "id_prefix": "all_probe_softgate",
                    "aggregation": "cross_method_softgate_matrix",
                    "gate_strategies": ["probe_level"],
                    "joint_rules": ["product"],
                }
            ]
        }
        metrics = [{"method": "all_probe_softgate_probe_level_product"}]
        method, source = budget_curve.resolve_ours_full_method(suite, metrics)
        self.assertEqual(method, "all_probe_softgate_probe_level_product")
        self.assertIn("suite:fusion_recipes-expanded", source)

    def test_resolves_ours_full_from_suite_and_output(self):
        suite = {
            "fusion_recipes": [
                {
                    "id": "r_softgate_top1",
                    "aggregation": "r_softgate",
                    "top_k": 1,
                    "pool_id": "top1",
                },
                {
                    "id": "r_softgate_all",
                    "aggregation": "r_softgate",
                    "top_k": "all",
                    "pool_id": "all",
                },
            ]
        }
        metrics = [
            {"method": "r_softgate_top1"},
            {
                "method": "r_softgate_all",
                "aggregation": "r_softgate",
                "pool_id": "all",
                "fusion_fit": {"top_k": "all"},
            },
        ]
        method, source = budget_curve.resolve_ours_full_method(suite, metrics)
        self.assertEqual(method, "r_softgate_all")
        self.assertIn("suite:fusion_recipes", source)
        self.assertIn("output:r_softgate-all", source)

    def test_seed_then_task_macro_and_paired_bootstrap(self):
        seed_rows = []
        values = {
            50: {1: (0.1, 0.3), 2: (0.5, 0.7)},
            100: {1: (0.2, 0.4), 2: (0.7, 0.9)},
        }
        datasets = {1: "cars", 2: "hico"}
        for budget, by_task in values.items():
            for order, by_seed in by_task.items():
                for seed, value in enumerate(by_seed):
                    seed_rows.append(
                        {
                            "budget": budget,
                            "realized_budget": budget,
                            "order": order,
                            "dataset": datasets[order],
                            "task": f"task_{order}",
                            "task_key": f"{order:03d}_{datasets[order]}_task_{order}",
                            "seed": seed,
                            "method": "r_softgate_all",
                            "value": value,
                            "clean_test_verified": True,
                        }
                    )
        task_rows = budget_curve.mean_by_task(seed_rows)
        means = {
            (row["budget"], row["order"]): row["mean_ap"] for row in task_rows
        }
        self.assertAlmostEqual(means[(50, 1)], 0.2)
        self.assertAlmostEqual(means[(50, 2)], 0.6)
        self.assertAlmostEqual(means[(100, 1)], 0.3)
        self.assertAlmostEqual(means[(100, 2)], 0.8)

        by_budget = {
            budget: {
                row["order"]: row["mean_ap"]
                for row in task_rows if row["budget"] == budget
            }
            for budget in (50, 100)
        }
        paired, samples = budget_curve.paired_bootstrap_curve(
            by_budget, (50, 100), repetitions=2_000, seed=7
        )
        self.assertEqual(paired, [1, 2])
        self.assertAlmostEqual(float(np.mean([means[(50, 1)], means[(50, 2)]])), 0.4)
        delta = samples[100] - samples[50]
        low, high = budget_curve.quantile_interval(delta)
        self.assertGreaterEqual(low, 0.1 - 1e-12)
        self.assertLessEqual(high, 0.2 + 1e-12)

    def test_strict_panel_requires_all_six_budgets_and_exact_seeds(self):
        seed_rows = []
        task_rows = []
        for budget in budget_curve.EXPECTED_BUDGETS:
            train, audit = budget_curve.EXPECTED_LABEL_COUNTS[budget]
            for order in budget_curve.ALL36_ORDERS:
                task_key = f"{order:03d}_fixture_task_{order}"
                task_rows.append(
                    {
                        "budget": budget,
                        "realized_budget": budget,
                        "train_label_count": train,
                        "audit_label_count": audit,
                        "order": order,
                        "dataset": "fixture",
                        "task": f"task_{order}",
                        "task_key": task_key,
                        "supervision_hash": f"hash-{budget}-{order}",
                        "min_attr_positive": 1,
                        "min_attr_negative": train - 1,
                        "single_class_attribute_count": 0,
                        "insufficient_balance_attribute_count": 0,
                        "joint_positive": 1,
                        "joint_negative": train - 1,
                        "clean_test_verified": True,
                    }
                )
                for seed in budget_curve.EXPECTED_SEEDS:
                    seed_rows.append(
                        {"budget": budget, "order": order, "seed": seed}
                    )
        budget_curve.validate_panel_coverage(
            task_rows,
            seed_rows,
            budget_curve.EXPECTED_BUDGETS,
            allow_incomplete=False,
        )
        with self.assertRaisesRegex(ValueError, "seeds"):
            budget_curve.validate_panel_coverage(
                task_rows,
                seed_rows[:-1],
                budget_curve.EXPECTED_BUDGETS,
                allow_incomplete=False,
            )
        with self.assertRaisesRegex(ValueError, "budgets must be exactly"):
            budget_curve.validate_panel_coverage(
                [row for row in task_rows if row["budget"] != 500],
                [row for row in seed_rows if row["budget"] != 500],
                budget_curve.EXPECTED_BUDGETS[:-1],
                allow_incomplete=False,
            )
        missing_diagnostics = [dict(row) for row in task_rows]
        missing_diagnostics[0]["joint_positive"] = None
        with self.assertRaisesRegex(ValueError, "missing supervision diagnostics"):
            budget_curve.validate_panel_coverage(
                missing_diagnostics,
                seed_rows,
                budget_curve.EXPECTED_BUDGETS,
                allow_incomplete=False,
            )

    def test_task_identity_cannot_change_even_for_partial_pilot(self):
        rows = [
            {
                "budget": 50,
                "order": 1,
                "dataset": "cars",
                "task": "task_a",
                "task_key": "001_cars_task_a",
            },
            {
                "budget": 100,
                "order": 1,
                "dataset": "cars",
                "task": "task_b",
                "task_key": "001_cars_task_b",
            },
        ]
        with self.assertRaisesRegex(ValueError, "changes task identity"):
            budget_curve.validate_panel_coverage(
                rows, [], (50, 100), allow_incomplete=True
            )

    def test_partial_curve_skips_dataset_without_a_paired_task(self):
        rows = []
        for budget, order, dataset, value in (
            (50, 1, "cars", 0.2),
            (50, 8, "hico", 0.3),
            (100, 8, "hico", 0.5),
        ):
            rows.append(
                {
                    "budget": budget,
                    "order": order,
                    "dataset": dataset,
                    "mean_ap": value,
                    "seed_count": 5,
                    "realized_budget": budget,
                    "method": "r_softgate_all",
                }
            )
        macro = budget_curve.aggregate_macro_curves(
            rows,
            (50, 100),
            bootstrap_repetitions=100,
            bootstrap_seed=4,
            reference_budget=50,
        )
        self.assertFalse(
            any(row["group_type"] == "dataset" and row["group"] == "cars" for row in macro)
        )
        self.assertTrue(
            any(row["group_type"] == "dataset" and row["group"] == "hico" for row in macro)
        )

    def test_report_does_not_claim_plateau_without_post_reference_budget(self):
        rows = []
        values = {50: 0.30, 100: 0.49, 150: 0.68}
        budgets = (50, 100, 150)
        for index, budget in enumerate(budgets):
            previous_budget = None if index == 0 else budgets[index - 1]
            delta_previous = (
                None if previous_budget is None else values[budget] - values[previous_budget]
            )
            rows.append(
                {
                    "group_type": "cohort",
                    "group": "Formal-23",
                    "budget": budget,
                    "task_count": 23,
                    "declared_task_count": 23,
                    "macro_test_joint_ap": values[budget],
                    "ci95_low": values[budget] - 0.05,
                    "ci95_high": values[budget] + 0.05,
                    "delta_vs_previous": delta_previous,
                    "delta_vs_previous_ci95_low": (
                        None if delta_previous is None else delta_previous - 0.02
                    ),
                    "delta_vs_previous_ci95_high": (
                        None if delta_previous is None else delta_previous + 0.02
                    ),
                    "reference_budget": 150,
                    "delta_vs_reference": values[budget] - values[150],
                    "delta_vs_reference_ci95_low": values[budget] - values[150] - 0.02,
                    "delta_vs_reference_ci95_high": values[budget] - values[150] + 0.02,
                    "insufficient_balance_attribute_task_count": 0,
                    "single_class_attribute_task_count": 0,
                }
            )
        report = budget_curve.build_report(
            [], [], rows, [], bootstrap_repetitions=100, bootstrap_seed=7
        )
        self.assertIn("plateau claim must wait", report)
        self.assertNotIn("diminishing returns near 150", report)

    def test_end_to_end_outputs_keep_legacy_separate(self):
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            suite_path = temp / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "fixture_selected8",
                        "fusion_recipes": [
                            {
                                "id": "r_softgate_top1",
                                "aggregation": "r_softgate",
                                "top_k": 1,
                                "pool_id": "top1",
                            },
                            {
                                "id": "r_softgate_all",
                                "aggregation": "r_softgate",
                                "top_k": "all",
                                "pool_id": "all",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runs = []
            task_values = {
                50: {1: (0.20, 0.30), 46: (0.50, 0.70)},
                100: {1: (0.35, 0.45), 46: (0.70, 0.90)},
            }
            task_meta = {
                1: ("cars", "task_bmw_convertible"),
                46: ("cub", "task_cub_fixture"),
            }
            for budget, by_task in task_values.items():
                run_root = temp / f"budget_{budget}"
                run_root.mkdir()
                (run_root / "run_state.json").write_text(
                    json.dumps(
                        {
                            "suite_id": "fixture_selected8",
                            "suite_config": str(suite_path),
                        }
                    ),
                    encoding="utf-8",
                )
                for order, seed_values in by_task.items():
                    dataset, task = task_meta[order]
                    task_dir = run_root / "tasks" / f"{order:03d}_{dataset}_{task}"
                    task_dir.mkdir(parents=True)
                    summary = {
                        "order": order,
                        "dataset": dataset,
                        "task": task,
                        "status": "completed",
                        "stage_status": {"iterative": "completed"},
                        "vqa_successful": budget,
                        "supervision_audits": {
                            "iterative": {
                                "attribute_counts": (
                                    {
                                        "attribute_a": {"positive": 0, "negative": budget},
                                        "attribute_b": {
                                            "positive": budget // 2,
                                            "negative": budget - budget // 2,
                                        },
                                    }
                                    if order == 1
                                    else {
                                        "attribute_a": {"positive": 1, "negative": budget - 1},
                                    }
                                ),
                                "joint_counts": {"positive": 1, "negative": budget - 1},
                                "evaluation_boundary": {
                                    "frozen_test_overlap_count": 0,
                                    "gallery_membership_valid": True,
                                }
                            }
                        },
                    }
                    (task_dir / "summary.json").write_text(
                        json.dumps(summary), encoding="utf-8"
                    )
                    metric_rows = []
                    for seed, value in enumerate(seed_values):
                        metric_rows.extend(
                            [
                                {
                                    "stage": "iterative",
                                    "split": "test",
                                    "seed": seed,
                                    "method": "mlp_baseline",
                                    "metrics": {"joint": {"ap": value - 0.05}},
                                },
                                {
                                    "stage": "iterative",
                                    "split": "test",
                                    "seed": seed,
                                    "method": "r_softgate_all",
                                    "family": "fusion",
                                    "aggregation": "r_softgate",
                                    "pool_id": "all",
                                    "fusion_fit": {"top_k": "all"},
                                    "metrics": {
                                        "attribute_a": {"ap": min(value + 0.1, 1.0)},
                                        "joint": {"ap": value, "f1_star": value - 0.1},
                                    },
                                },
                            ]
                        )
                    (task_dir / "iterative_metrics.json").write_text(
                        json.dumps(metric_rows), encoding="utf-8"
                    )
                runs.append(budget_curve.BudgetRun(budget, run_root))

            legacy_path = temp / "old6.csv"
            with legacy_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("strategy", "total_vqa_labels", "macro_test_ap", "n_tasks"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "strategy": "Iterative",
                        "total_vqa_labels": 100,
                        "macro_test_ap": 0.745,
                        "n_tasks": 6,
                    }
                )

            output = temp / "summary"
            payload = budget_curve.summarize(
                runs,
                output,
                allow_incomplete_panels=True,
                bootstrap_repetitions=500,
                bootstrap_seed=11,
                legacy_specs=(("Old-6", legacy_path),),
            )
            self.assertEqual(payload["ours_full_method_ids"], ["r_softgate_all"])
            for name in (
                "metrics_long.csv",
                "per_task_seed.csv",
                "per_task_seed.json",
                "per_task_mean.csv",
                "per_task_mean.json",
                "macro_curve.csv",
                "macro_curve.json",
                "legacy_curves.csv",
                "macro_curve.png",
                "macro_curve.pdf",
                "dataset_curves.png",
                "dataset_curves.pdf",
                "REPORT.md",
            ):
                self.assertTrue((output / name).is_file(), name)

            with (output / "macro_curve.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                macro = list(csv.DictReader(handle))
            formal_50 = next(
                row for row in macro
                if row["group"] == "Formal-23" and int(row["budget"]) == 50
            )
            all36_100 = next(
                row for row in macro
                if row["group"] == "All-36" and int(row["budget"]) == 100
            )
            self.assertAlmostEqual(float(formal_50["macro_test_joint_ap"]), 0.25)
            self.assertAlmostEqual(float(all36_100["macro_test_joint_ap"]), 0.60)
            self.assertEqual(formal_50["single_class_attribute_task_count"], "1")
            self.assertEqual(all36_100["single_class_attribute_task_count"], "1")
            self.assertEqual(
                formal_50["insufficient_balance_attribute_task_count"], "1"
            )
            self.assertEqual(
                all36_100["insufficient_balance_attribute_task_count"], "2"
            )
            self.assertAlmostEqual(
                float(all36_100["insufficient_balance_attribute_task_fraction"]), 1.0
            )
            with (output / "per_task_mean.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                task_diagnostics = list(csv.DictReader(handle))
            order1_50 = next(
                row for row in task_diagnostics
                if int(row["order"]) == 1 and int(row["budget"]) == 50
            )
            self.assertEqual(order1_50["min_attr_positive"], "0")
            self.assertEqual(order1_50["min_attr_negative"], "25")
            self.assertEqual(order1_50["single_class_attribute_count"], "1")
            self.assertEqual(order1_50["insufficient_balance_attribute_count"], "1")
            self.assertEqual(order1_50["joint_positive"], "1")
            self.assertEqual(order1_50["joint_negative"], "49")
            with (output / "per_task_seed.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                seed_diagnostics = list(csv.DictReader(handle))
            order1_50_seed0 = next(
                row for row in seed_diagnostics
                if int(row["order"]) == 1
                and int(row["budget"]) == 50
                and int(row["seed"]) == 0
            )
            self.assertEqual(order1_50_seed0["single_class_attribute_count"], "1")
            self.assertEqual(
                order1_50_seed0["insufficient_balance_attribute_count"], "1"
            )
            self.assertEqual(order1_50_seed0["joint_positive"], "1")
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Historical context only", report)
            self.assertIn("must not be described as part", report)
            self.assertIn("prior ~150-label hypothesis", report)
            self.assertIn("successive increments", report)
            self.assertIn("single-class attr", report)
            self.assertIn("at least 2 positive and 2 negative", report)
            self.assertTrue(payload["legacy_context_is_excluded_from_current_estimates"])

    def test_clean_test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            suite = root / "suite.json"
            suite.write_text(
                json.dumps(
                    {
                        "fusion_recipes": [
                            {
                                "id": "r_softgate_all",
                                "aggregation": "r_softgate",
                                "top_k": "all",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "run_state.json").write_text(
                json.dumps({"suite_config": str(suite)}), encoding="utf-8"
            )
            task_dir = root / "tasks" / "001_cars_task_fixture"
            task_dir.mkdir(parents=True)
            (task_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "order": 1,
                        "dataset": "cars",
                        "task": "task_fixture",
                        "status": "completed",
                        "stage_status": {"iterative": "completed"},
                        "supervision_audits": {
                            "iterative": {
                                "evaluation_boundary": {
                                    "frozen_test_overlap_count": 1,
                                    "gallery_membership_valid": True,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "iterative_metrics.json").write_text(
                json.dumps(
                    [
                        {
                            "stage": "iterative",
                            "split": "test",
                            "seed": 0,
                            "method": "r_softgate_all",
                            "aggregation": "r_softgate",
                            "pool_id": "all",
                            "metrics": {"joint": {"ap": 0.5}},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "clean Test audit failed"):
                budget_curve.collect_task_records(
                    [budget_curve.BudgetRun(50, root)],
                    ours_full_override=None,
                    require_clean_test_audit=True,
                )


if __name__ == "__main__":
    unittest.main()
