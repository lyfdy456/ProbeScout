import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_retrieval_harness as H  # noqa: E402
from iterative_vqa_runner import (  # noqa: E402
    _apply_cache_audit_migrations,
    _committed_cache_audit,
    _iterative_cache_source_priority,
    _new_labels,
    _round_label_status,
)


ATTRS = ["a", "b"]


def write_jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def direct(image, a, b):
    return {"image": image, "a": a, "b": b}


def raw(image, attributes):
    return {
        "image": image,
        "answer": json.dumps({"attributes": attributes}, ensure_ascii=False),
    }


def attr(present):
    return {"present": present, "confidence": 0.9}


class StrictBinaryPredicateTests(unittest.TestCase):
    def test_requires_exact_keys_and_integer_zero_or_one(self):
        self.assertTrue(H.is_complete_binary_labels({"a": 0, "b": 1}, ATTRS))
        rejected = [
            {"a": True, "b": 1},
            {"a": 0, "b": None},
            {"a": 0},
            {"a": 0, "b": 1, "extra": 0},
            {"a": 0.0, "b": 1},
            {"a": "0", "b": 1},
            {"a": 2, "b": 1},
        ]
        for labels in rejected:
            with self.subTest(labels=labels):
                self.assertFalse(H.is_complete_binary_labels(labels, ATTRS))

    def test_raw_answers_are_validated_before_legacy_integer_coercion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_jsonl(Path(directory) / "raw_results.jsonl", [
                raw("ok.jpg", {"a": attr(0), "b": attr(1)}),
                raw("bool.jpg", {"a": attr(True), "b": attr(1)}),
                raw("string.jpg", {"a": attr("1"), "b": attr(1)}),
                raw("float.jpg", {"a": attr(0.5), "b": attr(1)}),
                raw("missing.jpg", {"a": attr(0)}),
                raw("extra.jpg", {"a": attr(0), "b": attr(1), "c": attr(0)}),
            ])
            labels, _, audit = H.load_vqa_many_strict([path], lambda value: value, ATTRS)

        self.assertEqual(labels, {
            "ok.jpg": {"a": 0, "b": 1},
            "extra.jpg": {"a": 0, "b": 1},
        })
        self.assertEqual(
            set(audit["incomplete_by_image"]),
            {"bool.jpg", "string.jpg", "float.jpg", "missing.jpg"},
        )
        self.assertEqual(
            audit["provenance_by_image"]["extra.jpg"][0][
                "ignored_extra_raw_attributes"
            ],
            ["c"],
        )


class StrictCacheMergeTests(unittest.TestCase):
    def test_iterative_priority_is_live_then_imported_then_fixed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa_root = root / "qa" / "stage"
            history_a = root / "supervision" / "a_results.jsonl"
            history_z = root / "supervision" / "z_results.jsonl"
            imported = qa_root / "imported_cache" / "cache_results.jsonl"
            live = qa_root / "round_00" / "live_results.jsonl"
            policy = _iterative_cache_source_priority(
                [history_z, live, imported, history_a], qa_root,
            )

        value = lambda path: policy[str(path.resolve())]["priority"]
        self.assertGreater(value(live), value(imported))
        self.assertGreater(value(imported), value(history_a))
        self.assertGreater(value(history_a), value(history_z))

    def test_equal_priority_conflict_is_withheld(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_jsonl(root / "first.jsonl", [direct("x.jpg", 0, 1)])
            second = write_jsonl(root / "second.jsonl", [direct("x.jpg", 1, 1)])
            labels, _, audit = H.load_vqa_many_strict(
                [first, second], lambda value: value, ATTRS,
            )

        self.assertNotIn("x.jpg", labels)
        self.assertIn("x.jpg", audit["conflicts"])

    def test_explicit_priority_selects_and_audits_shadowed_alternative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = write_jsonl(root / "low.jsonl", [direct("x.jpg", 0, 1)])
            high = write_jsonl(root / "high.jsonl", [direct("x.jpg", 1, 1)])
            priority = {
                str(low.resolve()): {"priority": 10, "category": "history"},
                str(high.resolve()): {"priority": 20, "category": "live"},
            }
            labels, _, audit = H.load_vqa_many_strict(
                [low, high], lambda value: value, ATTRS, source_priority=priority,
            )

        self.assertEqual(labels["x.jpg"], {"a": 1, "b": 1})
        self.assertFalse(audit["conflicts"])
        self.assertIn("x.jpg", audit["shadowed_conflicts"])
        self.assertEqual(
            audit["provenance_by_image"]["x.jpg"][0]["source_path"],
            str(high.resolve()),
        )

    def test_last_complete_row_wins_but_incomplete_retry_does_not_erase_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_jsonl(Path(directory) / "retry.jsonl", [
                direct("x.jpg", 0, 1),
                direct("x.jpg", 1, 1),
                direct("x.jpg", None, 1),
            ])
            labels, _, audit = H.load_vqa_many_strict(
                [path], lambda value: value, ATTRS,
            )

        self.assertEqual(labels["x.jpg"], {"a": 1, "b": 1})
        self.assertEqual(len(audit["shadowed_within_source"]), 1)
        self.assertIn("x.jpg", audit["incomplete_by_image"])


class IterativeCacheResumeTests(unittest.TestCase):
    def setUp(self):
        self.old_attrs = list(H.ATTRS)
        H.ATTRS = list(ATTRS)

    def tearDown(self):
        H.ATTRS = self.old_attrs

    def _loaded(self, root: Path):
        source = write_jsonl(root / "labels.jsonl", [
            direct("x.jpg", 0, 1), direct("y.jpg", 1, 1),
        ])
        priority = {
            str(source.resolve()): {"priority": 1, "category": "test"},
        }
        return H.load_vqa_many_strict(
            [source], lambda value: value, ATTRS, source_priority=priority,
        )

    def test_valid_legacy_state_gets_deterministic_hash_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, _, audit = self._loaded(Path(directory))
            state = {"rounds": [
                {"round": 0, "train": ["x.jpg"], "audit": []},
                {"round": 1, "train": ["y.jpg"], "audit": []},
            ]}
            report = _committed_cache_audit(state, labels, audit)
            self.assertTrue(report["ok"])
            self.assertEqual(len(report["migrations"]), 2)
            _apply_cache_audit_migrations(state, report)
            self.assertIn("cache_snapshot", state["rounds"][0])
            self.assertEqual(
                state["rounds"][0]["cache_snapshot_migration"]["schema"],
                "legacy-complete-conflict-free-cache-migration-v1",
            )
            self.assertTrue(_committed_cache_audit(state, labels, audit)["ok"])

    def test_label_drift_after_migration_fails_at_earliest_round(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, _, audit = self._loaded(Path(directory))
            state = {"rounds": [
                {"round": 0, "train": ["x.jpg"], "audit": []},
                {"round": 1, "train": ["y.jpg"], "audit": []},
            ]}
            initial = _committed_cache_audit(state, labels, audit)
            _apply_cache_audit_migrations(state, initial)
            labels["x.jpg"] = {"a": 1, "b": 1}
            drift = _committed_cache_audit(state, labels, audit)

        self.assertFalse(drift["ok"])
        self.assertEqual(drift["earliest_invalid_committed_round"], 0)
        self.assertIn("cache_snapshot_drift", drift["invalid_committed_rounds"][0])

    def test_incomplete_committed_label_reports_earliest_round_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, _, audit = self._loaded(Path(directory))
            labels.pop("x.jpg")
            state = {"rounds": [
                {"round": 0, "train": ["x.jpg"], "audit": []},
                {"round": 1, "train": ["y.jpg"], "audit": []},
            ]}
            report = _committed_cache_audit(state, labels, audit)

        self.assertFalse(report["ok"])
        self.assertEqual(report["earliest_invalid_committed_round"], 0)
        self.assertFalse(report["migrations"])
        self.assertNotIn("cache_snapshot", state["rounds"][1])

    def test_new_labels_and_round_status_treat_malformed_cache_as_missing(self):
        pick = {"train": ["x.jpg", "y.jpg"], "audit": [], "roles": {}}
        cache = {"x.jpg": {"a": None, "b": 1}, "y.jpg": {"a": 1, "b": 0}}
        candidates, missing, complete = _round_label_status(pick, cache, attrs=ATTRS)
        self.assertEqual(candidates, ["x.jpg", "y.jpg"])
        self.assertEqual(missing, ["x.jpg"])
        self.assertFalse(complete)

        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(
                args=SimpleNamespace(iterative_dry_run=False, auto_vqa=False),
            )
            reused, missing, complete, requested = _new_labels(
                context, candidates, cache, Path(directory), 0,
            )
        self.assertEqual(reused, ["y.jpg"])
        self.assertEqual(missing, ["x.jpg"])
        self.assertFalse(complete)
        self.assertEqual(requested, ["x.jpg"])


if __name__ == "__main__":
    unittest.main()
