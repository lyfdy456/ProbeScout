"""Numerical verification only: no training, data export, or publication."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unified_initial_baseline as INITIAL


class InitialForwardNumericsTests(unittest.TestCase):
    def verify(self, actual, expected):
        return INITIAL._verify_forward_probabilities(actual, expected, context="synthetic/attention_pooling/1")

    def test_observed_cpu_gpu_roundoff_passes_without_altering_scores(self):
        actual = np.asarray([0.40191924571990967], dtype=np.float32)
        expected = np.asarray([0.401917040348053], dtype=np.float32)
        original_actual, original_expected = actual.copy(), expected.copy()
        self.assertEqual(self.verify(actual, expected), 2.205371856689453e-6)
        np.testing.assert_array_equal(actual, original_actual)
        np.testing.assert_array_equal(expected, original_expected)

    def test_absolute_ceiling_does_not_grow_near_probability_one(self):
        self.assertEqual(INITIAL.FORWARD_RELATIVE_TOLERANCE, 0.0)
        self.assertLessEqual(INITIAL.FORWARD_ABSOLUTE_TOLERANCE, 1e-5)
        for expected in (0.0, 0.5, 0.99):
            with self.subTest(expected=expected):
                self.verify([expected + 0.999e-5], [expected])
                with self.assertRaisesRegex(RuntimeError, "forward/cache mismatch"):
                    self.verify([expected + 1.001e-5], [expected])

    def test_real_score_shift_or_row_permutation_is_rejected(self):
        for actual, expected in (([0.501], [0.5]), ([0.8, 0.2], [0.2, 0.8])):
            with self.subTest(actual=actual):
                with self.assertRaisesRegex(RuntimeError, "forward/cache mismatch"):
                    self.verify(actual, expected)

    def test_invalid_probabilities_cannot_pass_via_broadcasting_or_nan(self):
        for actual, expected in (([0.5], [0.5, 0.5]), ([], []),
                                 ([[0.5]], [[0.5]]), ([np.nan], [np.nan]),
                                 ([np.inf], [np.inf]), ([-1e-7], [0.0]),
                                 ([1.0], [1.0000001])):
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaisesRegex(RuntimeError, "probabilities are invalid"):
                    self.verify(actual, expected)

    def test_cpu_success_does_not_call_matched_backend(self):
        def unexpected_backend(start, stop):
            self.fail("CPU success must not invoke GPU inference")
        proof = INITIAL._verify_checkpoint_forward([0.4], [0.4], [0],
            context="synthetic", matched_batch_score=unexpected_backend)
        self.assertFalse(proof["matchedBackendFallbackUsed"])
        self.assertEqual(proof["matchedBackendBlocks"], [])

    def test_fallback_uses_only_failed_rows_original_blocks_and_preserves_alignment(self):
        cached = np.linspace(0.1, 0.9, 2300)
        # Deliberately use non-monotonic offline order, as web order can differ.
        rows = np.asarray([2299, 5, 1050, 1023, 1200])
        actual = cached[rows].copy()
        actual[[0, 2, 4]] += 3.6e-5
        original_cache = cached.copy()
        calls = []
        def matched_backend(start, stop):
            calls.append((start, stop))
            return cached[start:stop].copy()
        proof = INITIAL._verify_checkpoint_forward(actual, cached, rows,
            context="synthetic", matched_batch_score=matched_backend)
        self.assertEqual(calls, [(1024, 2048), (2048, 2300)])
        self.assertEqual(proof["cpuExceededSampleOfflineRows"], [2299, 1050, 1200])
        self.assertTrue(proof["matchedBackendFallbackUsed"])
        self.assertGreater(proof["cpuMaxAbsoluteError"], 1e-5)
        self.assertEqual(proof["matchedBackendMaxAbsoluteError"], 0)
        self.assertEqual(proof["maxAbsoluteError"], 0)
        np.testing.assert_array_equal(cached, original_cache)

    def test_matched_backend_failure_is_not_accepted_or_retried_with_larger_tolerance(self):
        cached = np.full(2048, 0.5)
        calls = []
        def shifted_backend(start, stop):
            calls.append((start, stop))
            values = cached[start:stop].copy()
            # Even an unsampled row of a rechecked block must match the cache.
            values[17] += 1.1e-5
            return values
        with self.assertRaisesRegex(RuntimeError, "matched-CUDA-block-1024-2048"):
            INITIAL._verify_checkpoint_forward([0.500036], cached, [1030],
                context="synthetic", matched_batch_score=shifted_backend)
        self.assertEqual(calls, [(1024, 2048)])

    def test_cpu_discrepancy_without_available_matching_backend_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "forward/cache mismatch"):
            INITIAL._verify_checkpoint_forward([0.500036], [0.5], [0], context="synthetic")
        def unavailable_backend(start, stop):
            raise RuntimeError("CUDA unavailable")
        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
            INITIAL._verify_checkpoint_forward([0.500036], [0.5], [0],
                context="synthetic", matched_batch_score=unavailable_backend)

    def test_invalid_sample_rows_are_rejected_before_backend(self):
        for rows in ([0, 0], [-1], [2], [0.5], [[0]], []):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(RuntimeError, "row alignment is invalid"):
                    INITIAL._verify_checkpoint_forward([0.5], [0.5, 0.5], rows, context="synthetic")

    def test_invalid_cpu_output_never_escapes_through_backend_fallback(self):
        def unexpected_backend(start, stop):
            self.fail("Malformed CPU output must fail before fallback")
        for actual in ([np.nan], [np.inf], [-1e-7], [1.0000001], [0.5, 0.5], [[0.5]]):
            with self.subTest(actual=actual):
                with self.assertRaisesRegex(RuntimeError, "probabilities are invalid"):
                    INITIAL._verify_checkpoint_forward(actual, [0.5], [0],
                        context="synthetic", matched_batch_score=unexpected_backend)


if __name__ == "__main__":
    unittest.main()
