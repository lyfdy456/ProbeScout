import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import iterative_vqa_runner as iterative  # noqa: E402


ATTRS = ["red object", "round shape"]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class FakeAdapter:
    def vqa_to_key(self, value: str) -> str:
        return str(value).replace("\\", "/")


def complete_labels(value: int = 1) -> dict[str, int]:
    return {attribute: value for attribute in ATTRS}


class ProviderContentRejectionScanTests(unittest.TestCase):
    def test_only_unresolved_data_inspection_failures_are_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa_root = root / "qa" / "iterative_vqa_fixture"
            write_jsonl(
                qa_root / "round_00" / "provider_results.jsonl",
                [
                    {
                        "image": "folder\\blocked.jpg",
                        "answer": "[FAILED]",
                        "error": {
                            "type": "data_inspection_failed",
                            "message": "provider content inspection rejected the image",
                        },
                    },
                    {
                        "image": "timeout.jpg",
                        "answer": "[FAILED]",
                        "error": "request timeout",
                    },
                    {
                        "image": "successful.jpg",
                        "answer": "valid answer",
                        "error": "old data_inspection_failed note",
                    },
                    {
                        "answer": "[FAILED]",
                        "error": "data_inspection_failed but image is absent",
                    },
                ],
            )
            # A matching row outside this exact iterative run is not evidence.
            write_jsonl(
                root / "other_stage" / "provider_results.jsonl",
                [{
                    "image": "outside.jpg",
                    "answer": "[FAILED]",
                    "error": "data_inspection_failed",
                }],
            )
            ctx = SimpleNamespace(
                qa_round_root=qa_root,
                adapter=FakeAdapter(),
                cached_map={},
                args=SimpleNamespace(
                    iterative_exclude_provider_content_rejections=True,
                ),
            )

            with mock.patch.object(iterative.H, "ATTRS", list(ATTRS)):
                rejected = iterative._provider_content_rejections(ctx)

            self.assertEqual(set(rejected), {"folder/blocked.jpg"})
            self.assertTrue(rejected["folder/blocked.jpg"])
            self.assertTrue(all(
                row.get("error_code") == "data_inspection_failed"
                for row in rejected["folder/blocked.jpg"]
            ))

            # A later strict, complete cache row resolves the provider failure.
            ctx.cached_map["folder/blocked.jpg"] = complete_labels()
            with mock.patch.object(iterative.H, "ATTRS", list(ATTRS)):
                self.assertEqual(iterative._provider_content_rejections(ctx), {})


class LegacyOptInIsolationTests(unittest.TestCase):
    def test_disabled_opt_in_never_scans_or_rewrites_provider_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / "task"
            train = [f"image_{index:03d}.jpg" for index in range(100)]
            adapter = SimpleNamespace(
                task_root=task_root,
                p2i={image: index for index, image in enumerate(train)},
                gt_by_attr={},
                query_idx=[0],
            )
            args = SimpleNamespace(
                dataset="fixture",
                task="legacy_iterative",
                joint_label=None,
                backbone="siglip",
                iterative_stage="iterative_vqa_100_50_v9_legacy_fixture",
                iterative_work_root=None,
                iterative_initial_labels=100,
                iterative_round_labels=50,
                iterative_train_per_round=40,
                iterative_audit_per_round=10,
                iterative_max_rounds=0,
                iterative_max_labels=100,
                iterative_stop_at_labels=100,
                iterative_adaptive_budget=False,
                iterative_audit_min_gain=0.05,
                iterative_resume=False,
                iterative_dry_run=True,
                iterative_export_final_scores=False,
                iterative_defer_round_evaluation=True,
                iterative_exclude_provider_content_rejections=False,
                auto_vqa=False,
                vqa_preflight_manifest=None,
                vqa_preflight_sha256=None,
                vqa_preflight_task_order=None,
                vqa_config="unused.yaml",
            )
            cache_audit = {
                "conflicts": {},
                "shadowed_conflicts": {},
                "priority_policy": {},
            }

            with (
                mock.patch.object(iterative.H, "configure"),
                mock.patch.object(iterative.H, "ADAPTER", adapter),
                mock.patch.object(iterative.H, "DATASET", "fixture"),
                mock.patch.object(iterative.H, "TASK", "legacy_iterative"),
                mock.patch.object(iterative.H, "ATTRS", list(ATTRS)),
                mock.patch.object(
                    iterative.H,
                    "load_backbone_embeddings",
                    return_value=(np.ones((100, 2), dtype=np.float32), {}),
                ),
                mock.patch.object(
                    iterative,
                    "_frozen_split",
                    return_value=(list(train), [], list(train)),
                ),
                mock.patch.object(iterative, "_all_cached_sources", return_value=[]),
                mock.patch.object(
                    iterative,
                    "_load_iterative_cache",
                    return_value=({}, {}, cache_audit),
                ),
                mock.patch.object(
                    iterative,
                    "_committed_cache_audit",
                    return_value={
                        "ok": True,
                        "migrations": [],
                        "invalid_committed_rounds": [],
                        "uncommitted_conflicts": [],
                    },
                ),
                mock.patch.object(
                    iterative,
                    "_round0_selection",
                    return_value=(
                        {"train": list(train), "audit": [], "roles": {}},
                        {"selection": "round0"},
                    ),
                ),
                mock.patch.object(
                    iterative,
                    "_new_labels",
                    side_effect=lambda _ctx, candidates, *_rest: (
                        [], list(candidates), False, list(candidates)
                    ),
                ),
                mock.patch.object(
                    iterative,
                    "_provider_content_rejections",
                    side_effect=AssertionError(
                        "disabled legacy run must not inspect provider rejections"
                    ),
                ),
                mock.patch.object(iterative, "_write_report"),
                mock.patch("builtins.print"),
            ):
                iterative.run_iterative_vqa(args)

            stage_root = task_root / "qa" / args.iterative_stage
            self.assertFalse(list(stage_root.glob("**/provider_content_rejection_audit.json")))


class FrozenSelectionBackfillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.qa_root = root / "qa" / "iterative_vqa_fixture"
        self.split_root = root / "split" / "iterative_vqa_fixture"
        self.qa_root.mkdir(parents=True)
        self.split_root.mkdir(parents=True)
        self.stage = "iterative_vqa_fixture"
        self.train = [f"train_{index:02d}.jpg" for index in range(40)]
        self.audit = [f"audit_{index:02d}.jpg" for index in range(10)]
        self.train_replacement = "replacement_train.jpg"
        self.audit_replacement = "replacement_audit.jpg"
        self.rejected_train = self.train[7]
        self.rejected_audit = self.audit[3]
        self.available = (
            self.train
            + self.audit
            + [self.train_replacement, self.audit_replacement]
        )
        self.pick = {
            "train": list(self.train),
            "audit": list(self.audit),
            "roles": {
                "joint_boundary": list(self.train[:14]),
                "audit_query": list(self.audit[:4]),
            },
        }
        self.diagnostics = {
            "selection": "committee_acquisition",
            "selected_scores": {
                image: {"query_similarity": index / 100.0}
                for index, image in enumerate(self.train + self.audit)
            },
        }
        self.cached_map = {
            image: complete_labels(index % 2)
            for index, image in enumerate(self.train + self.audit)
            if image not in {self.rejected_train, self.rejected_audit}
        }
        self.ctx = SimpleNamespace(
            qa_round_root=self.qa_root,
            split_root=self.split_root,
            report_root=root / "report",
            stage=self.stage,
            adapter=FakeAdapter(),
            cached_map=self.cached_map,
        )
        self.rejected = {
            self.rejected_train: [{
                "image": self.rejected_train,
                "answer": "[FAILED]",
                "error": "data_inspection_failed",
            }],
            self.rejected_audit: [{
                "image": self.rejected_audit,
                "answer": "[FAILED]",
                "error": "data_inspection_failed",
            }],
        }

    def tearDown(self):
        self.temp.cleanup()

    def _fresh_selection(self, *_args, **_kwargs):
        fresh_train = [self.train_replacement] + [
            image for image in self.train if image != self.rejected_train
        ]
        fresh_audit = [self.audit_replacement] + [
            image for image in self.audit if image != self.rejected_audit
        ]
        return (
            {
                "train": fresh_train[:40],
                "audit": fresh_audit[:10],
                "roles": {
                    "joint_boundary": fresh_train[:14],
                    "audit_query": fresh_audit[:4],
                },
            },
            {
                "selection": "committee_acquisition",
                "selected_scores": {
                    self.train_replacement: {"query_similarity": 0.91},
                    self.audit_replacement: {"query_similarity": 0.73},
                },
            },
        )

    def _refresh(self, pick=None, diagnostics=None, state=None):
        with mock.patch.object(
            iterative, "_role_selection", side_effect=self._fresh_selection
        ):
            return iterative._refresh_rejected_frozen_selection(
                self.ctx,
                ({
                    "rounds": [{
                        "round": 0,
                        "train": ["prior_committed.jpg"],
                        "audit": [],
                    }],
                    "pending_round": 1,
                }
                 if state is None else state),
                1,
                copy.deepcopy(self.pick if pick is None else pick),
                copy.deepcopy(
                    self.diagnostics if diagnostics is None else diagnostics
                ),
                list(self.available),
                copy.deepcopy(self.rejected),
            )

    def _persist(self, *, cache_replacements=False):
        if cache_replacements:
            self.ctx.cached_map[self.train_replacement] = complete_labels(1)
            self.ctx.cached_map[self.audit_replacement] = complete_labels(0)
        state = {
            "rounds": [{
                "round": 0,
                "train": ["prior_committed.jpg"],
                "audit": [],
            }],
            "pending_round": 1,
            "status": "awaiting_vqa",
        }
        refreshed, diagnostics, audit = self._refresh(state=state)
        qa_dir = self.qa_root / "round_01"
        rpaths = {
            "qa": qa_dir,
            "manifest": qa_dir / "manifest.json",
            "state": self.split_root / "round_state.json",
        }
        previous_manifest = {
            "round": 1,
            "stage": self.stage,
            "train": list(self.train),
            "audit": list(self.audit),
            "roles": copy.deepcopy(self.pick["roles"]),
            "to_label": [self.rejected_train, self.rejected_audit],
            "requested_to_label": [self.rejected_train, self.rejected_audit],
            "diagnostics": copy.deepcopy(self.diagnostics),
            "complete": False,
        }
        iterative._write_json(rpaths["manifest"], previous_manifest)
        iterative._write_json(qa_dir / "candidate_pool.json", self.available)
        previous_manifest_sha256 = iterative._sha256_file(rpaths["manifest"])
        candidate_pool_sha256 = iterative._sha256_file(
            qa_dir / "candidate_pool.json"
        )
        with (
            mock.patch.object(iterative.H, "ATTRS", list(ATTRS)),
            mock.patch.object(
                iterative,
                "_prequential_predictions",
                return_value={"n": 10, "committee": {"reason": "fixture"}},
            ),
            mock.patch.object(iterative, "_write_report"),
        ):
            result = iterative._persist_rejected_pending_selection(
                self.ctx,
                state,
                rpaths,
                1,
                previous_manifest,
                refreshed,
                diagnostics,
                audit,
                ["prior_committed.jpg"],
                [],
            )
        return {
            "result": result,
            "state": state,
            "pick": refreshed,
            "manifest": json.loads(rpaths["manifest"].read_text(encoding="utf-8")),
            "to_label": json.loads(
                (qa_dir / "to_label.json").read_text(encoding="utf-8")
            ),
            "reused": json.loads(
                (qa_dir / "reused_labels.json").read_text(encoding="utf-8")
            ),
            "ledger": json.loads(
                (qa_dir / "provider_content_rejection_audit.json").read_text(
                    encoding="utf-8"
                )
            ),
            "previous_manifest": previous_manifest,
            "previous_manifest_sha256": previous_manifest_sha256,
            "candidate_pool_sha256": candidate_pool_sha256,
        }

    def test_preserves_successes_partitions_and_exact_refinement_quotas(self):
        refreshed, diagnostics, audit = self._refresh()

        self.assertEqual(len(refreshed["train"]), 40)
        self.assertEqual(len(refreshed["audit"]), 10)
        self.assertFalse(set(refreshed["train"]) & set(refreshed["audit"]))
        self.assertNotIn(self.rejected_train, refreshed["train"])
        self.assertNotIn(self.rejected_audit, refreshed["audit"])
        self.assertEqual(
            set(self.train) - {self.rejected_train},
            set(refreshed["train"]) - {self.train_replacement},
        )
        self.assertEqual(
            set(self.audit) - {self.rejected_audit},
            set(refreshed["audit"]) - {self.audit_replacement},
        )
        self.assertIn(self.train_replacement, refreshed["train"])
        self.assertIn(self.audit_replacement, refreshed["audit"])
        self.assertEqual(
            audit["replacement_map"],
            {
                self.rejected_train: self.train_replacement,
                self.rejected_audit: self.audit_replacement,
            },
        )
        self.assertEqual(
            set(audit["rejected_not_budgeted"]),
            {self.rejected_train, self.rejected_audit},
        )
        self.assertEqual(len(audit["selection_contract_before_sha256"]), 64)
        self.assertEqual(len(audit["selection_contract_after_sha256"]), 64)
        self.assertNotEqual(
            audit["selection_contract_before_sha256"],
            audit["selection_contract_after_sha256"],
        )
        self.assertEqual(
            diagnostics["selected_scores"][self.train[0]],
            self.diagnostics["selected_scores"][self.train[0]],
        )
        self.assertEqual(
            diagnostics["selected_scores"][self.train_replacement],
            {"query_similarity": 0.91},
        )

    def test_refresh_is_deterministic_and_idempotent(self):
        first_pick, first_diagnostics, first_audit = self._refresh()
        repeated_pick, repeated_diagnostics, repeated_audit = self._refresh()
        self.assertEqual(repeated_pick, first_pick)
        self.assertEqual(repeated_diagnostics, first_diagnostics)
        self.assertEqual(repeated_audit, first_audit)

        # Replaying against the already-refreshed contract is a no-op.
        no_op_pick, no_op_diagnostics, no_op_audit = self._refresh(
            pick=first_pick,
            diagnostics=first_diagnostics,
        )
        self.assertEqual(no_op_pick, first_pick)
        self.assertEqual(no_op_diagnostics, first_diagnostics)
        self.assertEqual(
            no_op_audit["selection_contract_after_sha256"],
            first_audit["selection_contract_after_sha256"],
        )

    def test_missing_pending_marker_is_safe_for_current_uncommitted_round(self):
        state = {
            "rounds": [{
                "round": 0,
                "train": ["prior_committed.jpg"],
                "audit": [],
            }],
        }
        refreshed, _, audit = self._refresh(state=state)
        self.assertTrue(audit["changed"])
        self.assertIn(self.train_replacement, refreshed["train"])
        self.assertIn(self.audit_replacement, refreshed["audit"])

    def test_persist_synchronizes_pending_files_and_full_supersession_audit(self):
        persisted = self._persist()
        manifest = persisted["manifest"]
        self.assertEqual(set(persisted["to_label"]), set(manifest["to_label"]))
        self.assertEqual(set(persisted["reused"]), set(manifest["reused"]))
        self.assertEqual(set(manifest["to_label"]), {
            self.train_replacement, self.audit_replacement,
        })
        self.assertEqual(
            set(manifest["reused"]),
            set(persisted["pick"]["train"] + persisted["pick"]["audit"])
            - {self.train_replacement, self.audit_replacement},
        )
        self.assertEqual(manifest["requested_to_label"], [])
        audit = persisted["ledger"]["audits"][-1]
        self.assertEqual(
            audit["previous_manifest_sha256"],
            persisted["previous_manifest_sha256"],
        )
        self.assertEqual(
            audit["candidate_pool_sha256"],
            persisted["candidate_pool_sha256"],
        )
        self.assertEqual(
            audit["superseded_requested_to_label"],
            persisted["previous_manifest"]["requested_to_label"],
        )
        self.assertEqual(
            audit["selection_before"]["train"],
            persisted["previous_manifest"]["train"],
        )
        self.assertEqual(
            audit["selection_before"]["audit"],
            persisted["previous_manifest"]["audit"],
        )
        self.assertEqual(audit["selection_after"], persisted["pick"])
        self.assertEqual(len(audit["selection_contract_before_sha256"]), 64)
        self.assertEqual(len(audit["selection_contract_after_sha256"]), 64)

    def test_cached_replacements_never_create_zero_missing_await_state(self):
        persisted = self._persist(cache_replacements=True)
        manifest = persisted["manifest"]
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["to_label"], [])
        self.assertEqual(persisted["to_label"], [])
        self.assertNotEqual(persisted["state"].get("status"), "awaiting_vqa")
        self.assertFalse(manifest.get("requires_new_preflight_before_network", False))
        self.assertFalse(persisted["state"].get("requires_repreflight", False))

    def test_protocol_quota_and_unknown_survivor_fail_closed(self):
        underfilled = copy.deepcopy(self.pick)
        underfilled["train"].remove(self.train[-1])
        with self.assertRaisesRegex(RuntimeError, "quota|40"):
            self._refresh(pick=underfilled)

        unknown = copy.deepcopy(self.pick)
        unknown["train"][-1] = "not_in_candidate_pool.jpg"
        with self.assertRaisesRegex(
            RuntimeError, "candidate pool|train pool|unknown"
        ):
            self._refresh(pick=unknown)

    def test_rejection_in_a_committed_round_fails_closed(self):
        state = {
            "pending_round": 1,
            "rounds": [{
                "round": 0,
                "train": list(self.train),
                "audit": list(self.audit),
            }]
        }
        state_before = copy.deepcopy(state)
        with self.assertRaisesRegex(RuntimeError, "committed|Committed"):
            self._refresh(state=state)
        self.assertEqual(state, state_before)


if __name__ == "__main__":
    unittest.main()
