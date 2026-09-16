import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_vqa_budget_acquisition as acquisition  # noqa: E402


class VqaBudgetAcquisitionControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state_path = self.root / "round_state.json"
        self.stage = "iterative_vqa_100_50_v3_budget500"
        self.task = {
            "order": 3,
            "dataset": "cars",
            "task": "task_demo",
            "joint_label": "demo",
        }

    def tearDown(self):
        self.temp.cleanup()

    def _round(self, round_no):
        total = 100 + 50 * round_no
        return {
            "round": round_no,
            "stage": self.stage,
            "complete": True,
            "total_labeled": total,
            "n_train": 80 + 40 * round_no,
            "n_audit": 20 + 10 * round_no,
            "n_new": 5,
        }

    def _write_state(self, rounds, *, status="preflight_batch_completed"):
        self.state_path.write_text(
            json.dumps(
                {
                    "stage": self.stage,
                    "dataset": self.task["dataset"],
                    "task": self.task["task"],
                    "status": status,
                    "rounds": rounds,
                }
            ),
            encoding="utf-8",
        )

    def _approved_manifest(self, *, round_no=1):
        row = {
            "order": 3,
            "dataset": "cars",
            "task": "task_demo",
            "round": round_no,
        }
        return {
            "stage": self.stage,
            "task_count": 1,
            "tasks": [{**row, "row_count": 1}],
            "rows": [{**row, "logical_image": "example.jpg"}],
        }

    def test_committed_preflight_round_is_reused_without_subprocess(self):
        self._write_state([self._round(0), self._round(1)], status="running")
        approved = acquisition.approved_preflight_rounds(
            self._approved_manifest(), [self.task], stage=self.stage
        )
        reused = acquisition.committed_preflight_result(
            self.task,
            state_path=self.state_path,
            stage=self.stage,
            approved_round=approved[3],
        )
        self.assertEqual(reused["status"], "reused_preflight_batch_completed")
        self.assertEqual(reused["approved_preflight_round"], 1)
        self.assertEqual(reused["logical_vqa"], 150)

        with patch.object(acquisition.subprocess, "run") as subprocess_run:
            result = acquisition.run_task(
                self.task,
                state_path=self.state_path,
                stage=self.stage,
                log_root=self.root / "logs",
                allow_vqa=True,
                vqa_config=str(self.root / "public.yaml"),
                vqa_workers=1,
                vqa_python=None,
                approved_preflight=self.root / "approved.json",
                approved_preflight_sha256="a" * 64,
                committed_preflight=reused,
            )
        subprocess_run.assert_not_called()
        self.assertEqual(result, reused)
        self.assertFalse((self.root / "logs").exists())

    def test_uncommitted_preflight_round_is_not_reused(self):
        self._write_state([self._round(0)], status="awaiting_vqa")
        result = acquisition.committed_preflight_result(
            self.task,
            state_path=self.state_path,
            stage=self.stage,
            approved_round=1,
        )
        self.assertIsNone(result)

    def test_later_progress_still_makes_stale_preflight_a_noop(self):
        self._write_state(
            [self._round(0), self._round(1), self._round(2)],
            status="awaiting_vqa",
        )
        result = acquisition.committed_preflight_result(
            self.task,
            state_path=self.state_path,
            stage=self.stage,
            approved_round=1,
        )
        self.assertEqual(result["status"], "reused_preflight_batch_completed")
        self.assertEqual(result["rounds"], 3)
        self.assertEqual(result["logical_vqa"], 200)

    def test_resume_checks_fail_closed_on_state_or_preflight_mismatch(self):
        self._write_state([self._round(0), {**self._round(1), "round": 2}])
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            acquisition.committed_preflight_result(
                self.task,
                state_path=self.state_path,
                stage=self.stage,
                approved_round=1,
            )

        wrong_stage = self._approved_manifest()
        wrong_stage["stage"] = "other-stage"
        with self.assertRaisesRegex(ValueError, "stage mismatch"):
            acquisition.approved_preflight_rounds(
                wrong_stage, [self.task], stage=self.stage
            )

        wrong_identity = self._approved_manifest()
        wrong_identity["tasks"][0]["task"] = "other-task"
        wrong_identity["rows"][0]["task"] = "other-task"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            acquisition.approved_preflight_rounds(
                wrong_identity, [self.task], stage=self.stage
            )


if __name__ == "__main__":
    unittest.main()
