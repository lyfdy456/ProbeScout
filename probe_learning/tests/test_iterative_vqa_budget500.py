import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from iterative_vqa_runner import (  # noqa: E402
    _round_label_status,
    _validate_iterative_protocol,
)


def args(**overrides):
    values = {
        "iterative_initial_labels": 100,
        "iterative_round_labels": 50,
        "iterative_train_per_round": 40,
        "iterative_audit_per_round": 10,
        "iterative_max_rounds": 8,
        "iterative_max_labels": 500,
        "iterative_stop_at_labels": 500,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class IterativeVqaBudget500Tests(unittest.TestCase):
    def test_eight_refinement_rounds_reach_budget_500(self):
        _validate_iterative_protocol(args())

    def test_budget_above_500_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, r"stop <= max <= 500"):
            _validate_iterative_protocol(
                args(
                    iterative_max_rounds=9,
                    iterative_max_labels=550,
                    iterative_stop_at_labels=550,
                )
            )

    def test_rounds_must_fit_inside_hard_ceiling(self):
        with self.assertRaisesRegex(
            SystemExit, "rounds exceed the hard label ceiling"
        ):
            _validate_iterative_protocol(
                args(iterative_max_labels=300, iterative_stop_at_labels=300)
            )

    def test_partial_vqa_batch_keeps_the_frozen_selection_pending(self):
        pick = {
            "train": ["a.jpg", "b.jpg"],
            "audit": ["c.jpg"],
            "roles": {"query_near": ["a.jpg", "b.jpg"], "audit": ["c.jpg"]},
        }
        candidates, missing, complete = _round_label_status(
            pick,
            {"a.jpg": {"attr": 1}, "c.jpg": {"attr": 0}},
            attrs=["attr"],
        )
        self.assertEqual(candidates, ["a.jpg", "b.jpg", "c.jpg"])
        self.assertEqual(missing, ["b.jpg"])
        self.assertFalse(complete)
        self.assertEqual(pick["train"], ["a.jpg", "b.jpg"])
        self.assertEqual(pick["audit"], ["c.jpg"])

    def test_budget500_auto_vqa_requires_preflight(self):
        with self.assertRaisesRegex(SystemExit, "requires an approved outbound preflight"):
            _validate_iterative_protocol(args(auto_vqa=True))


if __name__ == "__main__":
    unittest.main()
