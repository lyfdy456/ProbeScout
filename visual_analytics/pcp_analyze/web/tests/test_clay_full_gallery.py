"""Pure contract tests for the offline CLAY score export; no data/model writes."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_clay_full_gallery import prompt_bank, row_mapping, validate_scores


class ClayFullGalleryContractTests(unittest.TestCase):
    def test_row_mapping_reorders_scores_into_web_order(self):
        offline = ["query.jpg", "b.jpg", "a.jpg"]
        web = ["a.jpg", "query.jpg", "b.jpg"]
        mapping = row_mapping(offline, web)
        np.testing.assert_array_equal(mapping, [2, 0, 1])
        self.assertEqual(mapping.dtype, np.int64)
        values = np.asarray([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
        np.testing.assert_array_equal(values[mapping], values[[2, 0, 1]])
        # Query conversion must address the reordered feature matrix.
        query_web_row = {image: row for row, image in enumerate(web)}[offline[0]]
        np.testing.assert_array_equal(values[mapping][query_web_row], values[0])

    def test_row_mapping_rejects_duplicate_ids_on_either_side(self):
        for offline, web in ((["a", "a"], ["a", "b"]), (["a", "b"], ["a", "a"])):
            with self.subTest(offline=offline, web=web):
                with self.assertRaisesRegex(ValueError, "Duplicate image IDs"):
                    row_mapping(offline, web)

    def test_row_mapping_rejects_missing_or_substituted_images(self):
        for offline, web in ((["a", "b"], ["a"]), (["a", "b"], ["a", "c"])):
            with self.subTest(offline=offline, web=web):
                with self.assertRaisesRegex(ValueError, "image IDs differ"):
                    row_mapping(offline, web)

    def test_prompt_bank_keeps_positive_negative_then_attribute_order(self):
        config = {
            "positive_templates": ["photo with {attribute}", "shows {attribute}"],
            "negative_templates": ["without {attribute}", "not {attribute}"],
        }
        self.assertEqual(prompt_bank([" black_hair ", "bangs"], config), [
            "photo with black hair", "shows black hair", "without black hair", "not black hair",
            "photo with bangs", "shows bangs", "without bangs", "not bangs",
        ])

    def test_prompt_bank_deduplicates_after_normalizing_attribute_names(self):
        config = {"positive_templates": ["{attribute}", "{attribute}"],
                  "negative_templates": ["no {attribute}"]}
        self.assertEqual(prompt_bank(["black_hair", "black hair", "bangs"], config),
                         ["black hair", "no black hair", "bangs", "no bangs"])

    def test_score_validation_accepts_raw_cosine_without_mutation(self):
        scores = np.asarray([[-1.0, 0.0], [0.45, 1.0], [1.000005, -1.000005]], dtype=np.float32)
        before = scores.copy()
        self.assertIsNone(validate_scores(scores, 3, 2))
        np.testing.assert_array_equal(scores, before)

    def test_score_validation_rejects_shape_and_dtype_mismatch(self):
        for values in (np.zeros(6, dtype=np.float32), np.zeros((2, 3), dtype=np.float32),
                       np.zeros((3, 2), dtype=np.float64), np.zeros((3, 2), dtype=np.int32)):
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with self.assertRaisesRegex(ValueError, "dimensions/dtype"):
                    validate_scores(values, 3, 2)

    def test_score_validation_rejects_nonfinite_or_out_of_range_values(self):
        for invalid in (np.nan, np.inf, -np.inf, 1.00002, -1.00002):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "non-finite or outside"):
                    validate_scores(np.asarray([[invalid]], dtype=np.float32), 1, 1)


if __name__ == "__main__":
    unittest.main()
