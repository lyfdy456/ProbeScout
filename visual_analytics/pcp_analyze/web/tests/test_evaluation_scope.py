from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


WEB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WEB_ROOT / "scripts" / "evaluation_scope.py"
SCRIPTS_DIR = MODULE_PATH.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("pcp_evaluation_scope", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SCOPE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCOPE
SPEC.loader.exec_module(SCOPE)

import export_web_data as WEB_EXPORT


class EvaluationScopeTests(unittest.TestCase):
    def setUp(self):
        self.image_ids = [f"image-{index:03d}" for index in range(103)]
        self.query_ids = self.image_ids[-3:]
        first = np.asarray([(index // 5) % 2 for index in range(103)], dtype=np.uint8)
        second = np.asarray([(index // 11) % 2 for index in range(103)], dtype=np.uint8)
        joint = first * second
        self.ground_truth = np.column_stack((first, second, joint))

    def build(self):
        return SCOPE.build_evaluation_masks(
            image_ids=self.image_ids,
            ground_truth=self.ground_truth,
            attribute_indices=[0, 1],
            query_image_ids=self.query_ids,
        )

    def test_partitions_are_deterministic_disjoint_and_exclude_query(self):
        first = self.build()
        second = self.build()
        for left, right in (
            (first.development, second.development),
            (first.validation, second.validation),
            (first.frozen_test, second.frozen_test),
        ):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(np.any(first.development & first.validation))
        self.assertFalse(np.any(first.development & first.frozen_test))
        self.assertFalse(np.any(first.validation & first.frozen_test))
        query_indices = [self.image_ids.index(value) for value in self.query_ids]
        for mask in (first.development, first.validation, first.frozen_test):
            self.assertFalse(np.any(mask[query_indices]))
        self.assertEqual(
            int(first.development.sum() + first.validation.sum() + first.frozen_test.sum()),
            len(self.image_ids) - len(self.query_ids),
        )

    def test_existing_frozen_test_membership_is_preserved(self):
        original = self.build()
        rebuilt = SCOPE.build_evaluation_masks(
            image_ids=self.image_ids,
            ground_truth=self.ground_truth,
            attribute_indices=[0, 1],
            query_image_ids=self.query_ids,
            test_mask=original.frozen_test,
        )
        np.testing.assert_array_equal(rebuilt.frozen_test, original.frozen_test)

    def test_training_overlap_is_removed_without_resampling_validation(self):
        original = self.build()
        candidate_indices = np.flatnonzero(original.frozen_test)
        removed_indices = candidate_indices[:3]
        training_ids = [self.image_ids[int(index)] for index in removed_indices]
        repaired = SCOPE.build_evaluation_masks(
            image_ids=self.image_ids,
            ground_truth=self.ground_truth,
            attribute_indices=[0, 1],
            query_image_ids=self.query_ids,
            training_image_ids=training_ids,
        )

        np.testing.assert_array_equal(repaired.validation, original.validation)
        self.assertEqual(int(repaired.frozen_test.sum()), int(original.frozen_test.sum()) - 3)
        self.assertTrue(np.all(repaired.frozen_test[removed_indices] == 0))
        self.assertTrue(np.all(repaired.development[removed_indices] == 1))
        self.assertTrue(np.all(repaired.excluded_training[removed_indices] == 1))
        np.testing.assert_array_equal(
            repaired.candidate_frozen_test,
            original.frozen_test,
        )

    def test_training_supervision_ids_are_strictly_validated(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SCOPE.build_evaluation_masks(
                image_ids=self.image_ids,
                ground_truth=self.ground_truth,
                attribute_indices=[0, 1],
                query_image_ids=self.query_ids,
                training_image_ids=[self.image_ids[0], self.image_ids[0]],
            )
        with self.assertRaisesRegex(ValueError, "outside the gallery"):
            SCOPE.build_evaluation_masks(
                image_ids=self.image_ids,
                ground_truth=self.ground_truth,
                attribute_indices=[0, 1],
                query_image_ids=self.query_ids,
                training_image_ids=["missing-image"],
            )

    def test_validation_is_stratified_fraction_of_eligible_remainder(self):
        masks = self.build()
        for key in ((0, 0), (0, 1), (1, 0), (1, 1)):
            bucket = np.all(self.ground_truth[:, :2] == key, axis=1)
            eligible = bucket & ~masks.frozen_test.astype(bool)
            eligible[[self.image_ids.index(value) for value in self.query_ids]] = False
            expected = round(0.25 * int(eligible.sum()))
            observed = int(np.count_nonzero(masks.validation.astype(bool) & bucket))
            self.assertEqual(observed, expected)

    def test_contract_declares_all_three_masks_and_counts(self):
        masks = self.build()
        contract = SCOPE.evaluation_contract(
            development_mask=masks.development,
            validation_mask=masks.validation,
            test_mask=masks.frozen_test,
            ground_truth=self.ground_truth,
            target_ids=["a", "b", "joint"],
            attribute_ids=["a", "b"],
        )
        self.assertEqual(contract["defaultResultScope"], "development")
        self.assertEqual([row["id"] for row in contract["scopes"]], ["development", "validation", "test"])
        self.assertEqual(contract["development"]["maskFileKey"], "developmentMask")
        self.assertEqual(contract["validation"]["maskFileKey"], "validationMask")
        self.assertEqual(contract["validation"]["splitSeed"], 43)
        self.assertEqual(contract["frozenTest"]["maskFileKey"], "testMask")
        self.assertEqual(contract["frozenTest"]["splitSeed"], 42)
        self.assertEqual(contract["testIsolation"]["excludedTrainingRowCount"], 0)
        self.assertTrue(contract["testIsolation"]["noReplacement"])

    def test_query_only_refresh_fails_closed_without_training_provenance(self):
        with self.assertRaisesRegex(ValueError, "Test-isolation contract"):
            WEB_EXPORT.training_supervision_from_existing_evaluation(
                {}, image_ids=self.image_ids
            )
        with self.assertRaisesRegex(ValueError, "active-training provenance"):
            WEB_EXPORT.training_supervision_from_existing_evaluation(
                {"testIsolation": {}}, image_ids=self.image_ids
            )


if __name__ == "__main__":
    unittest.main()
