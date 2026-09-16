import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import report_vqa_budget_acquisition as accounting  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class VqaBudgetAcquisitionAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stage = "iterative_vqa_100_50_v3_budget500"
        self.tasks_root = self.root / "tasks"
        self.acquisition = self.root / "acquisition"
        self.manifest = self.root / "manifest.json"
        self.task = {"order": 7, "dataset": "demo", "task": "task_demo"}
        write_json(self.manifest, {"tasks": [self.task]})

        task_root = self.tasks_root / "demo" / "task_demo"
        write_json(
            task_root / "split" / self.stage / "round_state.json",
            {
                "stage": self.stage,
                "dataset": "demo",
                "task": "task_demo",
                "status": "awaiting_vqa",
                "pending_round": 1,
                "provider_rejected_not_budgeted": ["blocked.jpg"],
                "rounds": [
                    {
                        "round": 0,
                        "complete": True,
                        "total_labeled": 2,
                        "train": ["cached.jpg", "api.jpg"],
                        "audit": [],
                        "reused": ["cached.jpg"],
                        "provider_request_history": [
                            {
                                "round": 0,
                                "requested": ["api.jpg"],
                                "request_count": 1,
                                "status": "provider_invoked",
                                "preflight_manifest_sha256": "a" * 64,
                            }
                        ],
                    }
                ],
            },
        )
        round0 = task_root / "qa" / self.stage / "round_00"
        round1 = task_root / "qa" / self.stage / "round_01"
        result = round0 / "20260902_demo_results.jsonl"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "image": "api.jpg",
                            "answer": "{\\\"ok\\\": 1}",
                            "timestamp": "2026-09-02T12:00:05",
                        }
                    ),
                    json.dumps(
                        {
                            "image": "blocked.jpg",
                            "answer": "[FAILED]",
                            "error": "Error code: 400 - {'error': {'code': 'data_inspection_failed'}}",
                            "timestamp": "2026-09-02T12:00:06",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        write_json(
            round0 / "manifest.json",
            {
                "round": 0,
                "train": ["cached.jpg", "api.jpg"],
                "audit": [],
                "complete": True,
                "reused": ["cached.jpg"],
                "rejected_not_budgeted": ["blocked.jpg"],
                "replacement_map": {"blocked.jpg": "replacement.jpg"},
            },
        )
        write_json(
            round1 / "manifest.json",
            {
                "round": 1,
                "train": ["cached2.jpg", "new.jpg"],
                "audit": [],
                "complete": False,
                "reused": ["cached2.jpg"],
                "to_label": ["new.jpg"],
            },
        )
        preflight = {
            "manifest_sha256": "a" * 64,
            "created_at_utc": "2026-09-02T04:00:00+00:00",
            "task_count": 1,
            "row_count": 2,
            "rows": [
                {"order": 7, "logical_image": "api.jpg"},
                {"order": 7, "logical_image": "blocked.jpg"},
            ],
        }
        write_json(self.acquisition / "outbound_preflight_wave01.json", preflight)
        command = [
            "python.exe",
            "run_retrieval_harness.py",
            "--auto-vqa",
            "--vqa-preflight-sha256",
            "a" * 64,
        ]
        log = self.acquisition / "logs" / "007_demo_task_demo.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "\n".join(
                [
                    f"[2026-09-02T12:00:00+0800] {command!r}",
                    f"[init] output file: {result.resolve()}",
                    "  [retry 1/3] temporary error",
                    "  [retry 2/3] temporary error",
                    "VQA labeling: 100%|########| 2/2 [00:06<00:00, 3.00s/img]",
                    "[done] success: 1, failed: 1",
                    f"[done] results saved to: {result.resolve()}",
                    # An auto-vqa command that fails before the labeler starts
                    # must not become a provider invocation.
                    f"[2026-09-02T12:10:00+0800] {command!r}",
                    "preflight validation failed closed",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_separates_logical_cache_provider_retry_and_rejection_counts(self):
        report = accounting.build_report(
            manifest_path=self.manifest,
            stage=self.stage,
            acquisition_root=self.acquisition,
            tasks_root=self.tasks_root,
        )
        summary = report["summary"]
        self.assertEqual(summary["logical_committed_task_images"], 2)
        self.assertEqual(summary["logical_pending_selected_task_images"], 2)
        self.assertEqual(summary["logical_selected_current_task_images"], 4)
        self.assertEqual(summary["logical_known_complete_current_task_images"], 3)
        self.assertEqual(
            summary["strict_cache_complete_selected_without_stage_provider_success"], 2
        )
        self.assertEqual(summary["provider_terminal_records"], 2)
        self.assertEqual(summary["provider_unique_successful_task_images"], 1)
        self.assertEqual(summary["provider_explicit_application_retries_logged"], 2)
        self.assertEqual(
            summary["provider_application_call_attempts_observed_lower_bound"], 4
        )
        self.assertIsNone(summary["provider_exact_http_request_count"])
        self.assertEqual(summary["provider_invocation_segments"], 1)
        self.assertEqual(summary["content_rejection_terminal_records"], 1)
        self.assertEqual(summary["rejected_not_budgeted_unique_task_images_ledger"], 1)
        self.assertEqual(summary["replacement_edges"], 1)
        self.assertEqual(summary["provider_labeler_elapsed_seconds_logged"], 6.0)
        self.assertEqual(
            summary["harness_to_last_provider_record_seconds_lower_bound"], 6.0
        )
        wave = next(row for row in report["waves"] if row["wave"] == "wave01")
        self.assertEqual(wave["approved_row_count"], 2)
        self.assertEqual(wave["provider_terminal_records"], 2)
        self.assertEqual(wave["provider_explicit_application_retries_logged"], 2)
        self.assertEqual(wave["incomplete_vqa_batches"], 0)
        self.assertTrue(report["evidence_coverage"]["stable_snapshot"])

    def test_supplemental_preflight_does_not_overwrite_base_wave(self):
        for wave, digest, row_count in (
            ("wave02", "d" * 64, 685),
            ("wave03", "e" * 64, 848),
        ):
            write_json(
                self.acquisition / f"outbound_preflight_{wave}.json",
                {
                    "manifest_sha256": digest,
                    "task_count": 1,
                    "row_count": row_count,
                    "rows": [{"order": 7, "logical_image": f"{wave}.jpg"}],
                },
            )
        write_json(
            self.acquisition / "outbound_preflight_wave04.json",
            {
                "manifest_sha256": "b" * 64,
                "task_count": 31,
                "row_count": 867,
                "rows": [{"order": 7, "logical_image": "base.jpg"}],
            },
        )
        write_json(
            self.acquisition / "outbound_preflight_wave04_replacement01.json",
            {
                "manifest_sha256": "c" * 64,
                "task_count": 2,
                "row_count": 2,
                "rows": [{"order": 7, "logical_image": "replacement.jpg"}],
            },
        )

        report = accounting.build_report(
            manifest_path=self.manifest,
            stage=self.stage,
            acquisition_root=self.acquisition,
            tasks_root=self.tasks_root,
        )
        waves = {row["wave"]: row for row in report["waves"]}

        self.assertEqual(waves["wave01"]["approved_row_count"], 2)
        self.assertEqual(waves["wave02"]["approved_row_count"], 685)
        self.assertEqual(waves["wave03"]["approved_row_count"], 848)
        self.assertEqual(waves["wave04"]["approved_row_count"], 867)
        self.assertEqual(
            waves["wave04_replacement01"]["approved_row_count"], 2
        )
        hash_to_wave, _ = accounting._preflight_catalog(self.acquisition)
        self.assertEqual(hash_to_wave["b" * 64], "wave04")
        self.assertEqual(hash_to_wave["c" * 64], "wave04_replacement01")


if __name__ == "__main__":
    unittest.main()
