import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import iterative_vqa_runner as iterative  # noqa: E402
import run_probebank_batch as batch  # noqa: E402


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class IterativeOnlySourceContractTest(unittest.TestCase):
    def test_isolated_acquisition_never_discovers_task_history_or_static_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "task"
            qa_root = root / "run" / "qa"
            old_random = task_root / "qa" / "random" / "random_results.jsonl"
            old_two_stage = task_root / "qa" / "two_stage" / "two_stage_results.jsonl"
            current = qa_root / "round_00" / "current_results.jsonl"
            for path in (old_random, old_two_stage, current):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            ctx = SimpleNamespace(
                isolated=True,
                qa_round_root=qa_root,
                adapter=SimpleNamespace(task_root=task_root),
                stage="iterative_vqa_100_50_v2",
            )
            with mock.patch.object(
                iterative.H,
                "discover_task_vqa_sources",
                side_effect=AssertionError("task history must not be discovered"),
            ):
                self.assertEqual(iterative._context_cached_sources(ctx), [current])
            with mock.patch.object(
                iterative.H,
                "resolve_vqa_sources",
                side_effect=AssertionError("two-stage control must stay disabled"),
            ):
                self.assertEqual(
                    iterative._static_control(ctx, 100),
                    {
                        "status": "disabled_iterative_only",
                        "source_policy": iterative.ISOLATED_SOURCE_POLICY,
                    },
                )

    def test_legacy_isolated_manifest_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_json(root / "qa" / "manifest.json", {
                "version": "iterative_vqa_100_50_v2",
                "vqa_sources": [str(root / "outside" / "two_stage_results.jsonl")],
            })
            with self.assertRaisesRegex(RuntimeError, "run-local-only"):
                iterative._validate_isolated_resume_manifest(
                    manifest,
                    expected_identity={"stage": "iterative_vqa_100_50_v2"},
                    qa_root=root / "qa",
                )

    def test_manifestless_work_root_with_old_round_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_root = Path(tmp) / "iterative"
            write_json(work_root / "split" / "round_state.json", {"status": "awaiting_vqa"})
            (work_root / "qa" / "round_00").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "refusing to adopt orphaned state"):
                iterative._validate_or_initialize_isolated_work_root(
                    work_root, {"stage": "iterative_vqa_100_50_v2"},
                )

    def _make_completed_run(self, task_out: Path) -> tuple[Path, Path, Path]:
        run_root = task_out / "vqa" / "iterative"
        qa_root = run_root / "qa"
        split_root = run_root / "split"
        stage = "iterative_vqa_100_50_v2"
        budget = {
            "initial": 100, "round": 50, "train": 40, "audit": 10,
            "max": 300, "default_stop": 250, "max_refine_rounds": 3,
            "adaptive": True, "audit_min_gain": 0.05,
        }
        identity = {
            "schema": iterative.ISOLATED_IDENTITY_SCHEMA,
            "source_policy": iterative.ISOLATED_SOURCE_POLICY,
            "dataset": "fixture", "task": "fixture_task", "stage": stage,
            "backbone": batch.BACKBONE, "attributes": ["attr"],
            "work_root": str(run_root.resolve()),
            "evaluation_split_policy": iterative.ISOLATED_EVALUATION_POLICY,
            "split": {
                "seed": 42, "test_fraction": 0.2,
                "gallery_hash": batch.stable_hash(["fresh.jpg"]),
                "candidate_test_hash": batch.stable_hash([]),
                "train_pool_hash": batch.stable_hash(["fresh.jpg"]),
            },
            "budget": budget,
            "records_hash": batch.stable_hash(["fresh.jpg"]),
            "record_count": 1,
            "query_indices": [],
            "query_paths_hash": batch.stable_hash([]),
            "vqa_config": "fixture", "joint_label": None,
        }
        identity_hash = iterative._stable_json_hash(identity)
        write_json(qa_root / "manifest.json", {
            "version": stage,
            "source_policy": iterative.ISOLATED_SOURCE_POLICY,
            "evaluation_split_policy": iterative.ISOLATED_EVALUATION_POLICY,
            "isolated_work_root": str(run_root.resolve()),
            "budget": budget,
            "identity": identity,
            "run_identity_sha256": identity_hash,
            "vqa_sources": [],
        })
        write_json(split_root / "round_state.json", {
            "stage": stage,
            "status": "completed",
            "source_policy": iterative.ISOLATED_SOURCE_POLICY,
            "run_identity_sha256": identity_hash,
            "rounds": [{"round": 0, "train": ["fresh.jpg"]}],
        })
        selected = write_json(
            split_root / "train_labeled_indices.json", ["fresh.jpg"],
        )
        round_root = qa_root / "round_00"
        write_json(round_root / "manifest.json", {
            "stage": stage,
            "round": 0,
            "run_identity_sha256": identity_hash,
            "complete": True,
            "train": ["fresh.jpg"],
        })
        current = round_root / "current_results.jsonl"
        current.write_text(json.dumps({"image": "fresh.jpg", "attr": 1}) + "\n", encoding="utf-8")
        return run_root, selected, current

    def test_probebank_consumes_only_exact_run_manifest_and_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_out = root / "batch" / "task"
            _, selected_path, current = self._make_completed_run(task_out)
            # A conflicting historical label exists, but it is outside the
            # isolated iterative run and therefore cannot override or count.
            old_random = task_out / "vqa" / "random" / "random_results.jsonl"
            old_two_stage = task_out / "vqa" / "two_stage" / "two_stage_results.jsonl"
            for path in (old_random, old_two_stage):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"image": "fresh.jpg", "attr": 0}) + "\n",
                    encoding="utf-8",
                )

            manifest, selected, sources, provenance = batch.iterative_run_supervision_inputs(
                task_out,
            )
            self.assertEqual(manifest, selected_path)
            self.assertEqual(selected, ["fresh.jpg"])
            self.assertEqual(sources, [current])
            self.assertTrue(provenance["ignored_task_supervision_stages"])
            self.assertNotIn(old_random.resolve(), [path.resolve() for path in sources])
            self.assertNotIn(old_two_stage.resolve(), [path.resolve() for path in sources])

            _, labels_map, _, _, audit = batch.resolve_training_supervision(
                selected=selected,
                reused_sources=[],
                new_sources=sources,
                vqa_to_key=lambda value: value,
                attrs=["attr"],
                audit_path=root / "audit.json",
                selected_manifest=manifest,
            )
            self.assertEqual(labels_map["fresh.jpg"]["attr"], 1)
            self.assertEqual(audit["matched_count"], 1)
            self.assertEqual(audit["reused_label_count"], 0)
            self.assertEqual(audit["new_label_count"], 1)
            self.assertEqual(audit["reused_sources"], [])
            self.assertEqual(audit["new_sources"], [str(current.resolve())])

    def test_probebank_rejects_selected_rows_not_in_round_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_out = Path(tmp) / "batch" / "task"
            _, selected_path, _ = self._make_completed_run(task_out)
            write_json(selected_path, ["fresh.jpg", "poison-from-two-stage.jpg"])
            with self.assertRaisesRegex(RuntimeError, "ordered union"):
                batch.iterative_run_supervision_inputs(task_out)

    def test_iterative_only_summary_does_not_count_random_or_two_stage_vqa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_out = root / "batch" / "task"
            current_round = task_out / "vqa" / "iterative" / "qa" / "round_00"
            write_json(current_round / "to_label.json", ["fresh.jpg"])
            (current_round / "current_results.jsonl").write_text(
                json.dumps({"image": "fresh.jpg", "attr": 1}) + "\n",
                encoding="utf-8",
            )
            for stage in ("random", "two_stage"):
                old_root = task_out / "vqa" / stage
                write_json(old_root / "to_label.json", [f"{stage}.jpg"] * 50)
                (old_root / f"{stage}_results.jsonl").write_text(
                    json.dumps({"image": f"{stage}.jpg", "attr": 0}) + "\n",
                    encoding="utf-8",
                )
            summary = {}
            batch.refresh_vqa_counts(
                summary,
                task_out,
                SimpleNamespace(task_root=root / "source-task", vqa_to_key=lambda value: value),
                ["attr"],
                {"stages": ["random", "two_stage", "iterative"]},
                iterative_only=True,
            )
            self.assertEqual(summary["vqa_attempted"], 1)
            self.assertEqual(summary["vqa_successful"], 1)
            self.assertEqual(summary["vqa_failed"], 0)

    def test_candidate_split_can_explicitly_ignore_active_task_manifest(self):
        paths = ["query.jpg"] + [f"image_{idx:02d}.jpg" for idx in range(20)]
        records = pd.DataFrame({
            "relative_path": paths,
            "embedding_index": np.arange(len(paths)),
        })
        adapter = SimpleNamespace(
            records=records,
            query_idx=[0],
            p2i={path: idx for idx, path in enumerate(paths)},
            gt_by_attr={"attr": {path: idx % 2 for idx, path in enumerate(paths)}},
        )
        with mock.patch.object(
            iterative,
            "_load_frozen_test_ids",
            side_effect=AssertionError("active two-stage evaluation manifest was consumed"),
        ):
            gallery, candidate_test, train = iterative._frozen_split(
                adapter,
                np.zeros((len(paths), 2), dtype=np.float32),
                adapter.p2i,
                adapter.gt_by_attr,
                use_persisted_evaluation=False,
            )
        self.assertEqual(gallery, paths)
        self.assertNotIn("query.jpg", candidate_test)
        self.assertNotIn("query.jpg", train)
        self.assertEqual(set(candidate_test).intersection(train), set())
        self.assertEqual(set(candidate_test).union(train), set(paths[1:]))

    def test_legacy_alias_cache_cannot_hit_iterative_identity_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_root = root / "task_isolated" / "001_fixture_task"
            run_hash = "a" * 64
            isolated_root = batch.iterative_only_probebank_root(
                root, run_hash, 1, "fixture", "task",
            )
            self.assertNotEqual(legacy_root, isolated_root)
            emb = np.zeros((2, 3), dtype=np.float32)
            paths = ["a.jpg", "b.jpg"]
            entry = batch.cache_dir(
                isolated_root, "fixture", "task", "iterative", "probe", "attr",
            )
            entry.mkdir(parents=True)
            old_meta = batch.expected_cache_meta(
                "fixture", "task", "iterative", "probe", "attr",
                emb, paths, [42], "supervision", 1,
            )
            write_json(entry / "metadata.json", old_meta)
            np.savez_compressed(entry / "scores.npz", scores=np.zeros((1, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "source_policy"):
                batch.read_cache(
                    isolated_root, "fixture", "task", "iterative", "probe", "attr",
                    emb, paths, [42], "supervision", 1,
                    legacy_supervision_digest="supervision",
                    cache_identity={
                        "source_policy": batch.ITERATIVE_ONLY_SOURCE_POLICY,
                        "train_pool_hash": batch.stable_hash(paths),
                        "evaluation_split_policy": iterative.ISOLATED_EVALUATION_POLICY,
                        "iterative_run_identity_sha256": run_hash,
                    },
                )


if __name__ == "__main__":
    unittest.main()
