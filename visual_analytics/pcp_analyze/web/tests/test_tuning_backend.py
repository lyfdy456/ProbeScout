from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np


WEB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WEB_ROOT / "scripts" / "tuning_server.py"
SPEC = importlib.util.spec_from_file_location("pcp_tuning_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TUNING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TUNING
SPEC.loader.exec_module(TUNING)


TASK_ID = "001_toy_task_example"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def clean_validation_split(
    *,
    fit_indices,
    fit_labels,
    validation_indices,
    validation_labels,
    fingerprint: str,
    target_id: str = "joint",
):
    """Build the complete Clean Validation contract expected by run snapshots."""

    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    fit_labels = np.asarray(fit_labels, dtype=np.uint8)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    validation_labels = np.asarray(validation_labels, dtype=np.uint8)
    return SimpleNamespace(
        fit_indices=fit_indices,
        fit_labels=fit_labels,
        validation_indices=validation_indices,
        validation_labels=validation_labels,
        audit={
            "schemaVersion": 2,
            "protocol": TUNING.CLEAN_VALIDATION_PROTOCOL,
            "replayKind": "clean-web-validation-ground-truth",
            "targetId": target_id,
            "targetKind": "derived" if target_id == "joint" else "attribute",
            "sourceCount": int(fit_indices.size + validation_indices.size),
            "fitCount": int(fit_indices.size),
            "fitPositiveCount": int(np.count_nonzero(fit_labels)),
            "fitNegativeCount": int(fit_labels.size - np.count_nonzero(fit_labels)),
            "validationCount": int(validation_indices.size),
            "validationPositiveCount": int(np.count_nonzero(validation_labels)),
            "validationNegativeCount": int(
                validation_labels.size - np.count_nonzero(validation_labels)
            ),
            "sourceFingerprint": f"{fingerprint}-source",
            "fitFingerprint": f"{fingerprint}-fit",
            "validationFingerprint": fingerprint,
            "fitProtocol": TUNING.PROBE_VALIDATION_PROTOCOL,
            "fitSeed": TUNING.PROBE_VALIDATION_SEED,
            "fitFraction": 1.0 - TUNING.PROBE_VALIDATION_FRACTION,
            "excludedOriginalSupervisionFingerprint": f"{fingerprint}-original",
            "excludedHistoricalFeedbackFingerprint": f"{fingerprint}-feedback",
        },
    )


def create_toy_web(root: Path) -> tuple[Path, list[str]]:
    web_root = root / "web"
    data_root = web_root / "public" / "data" / "tasks" / TASK_ID
    data_root.mkdir(parents=True)
    image_ids = [f"gallery/image-{index}.jpg" for index in range(9)]
    methods = ["Image Prototype", "Ours-Full"]
    targets = [{"id": "joint", "label": "Joint", "kind": "derived"}]
    raw_scores = np.asarray(
        [
            [0.10, 0.20],
            [0.20, 0.30],
            [0.30, 0.40],
            [0.40, 0.50],
            [0.50, 0.60],
            [0.90, 0.80],
            [0.80, 0.70],
            [0.70, 0.60],
            [0.60, 0.55],
        ],
        dtype="<f4",
    ).reshape(9, 2, 1)
    exported_ranks = np.asarray(
        [
            [0.10, 0.10],
            [0.20, 0.20],
            [0.30, 0.30],
            [0.40, 0.40],
            [0.50, 0.50],
            [0.90, 0.10],
            [0.80, 0.20],
            [0.70, 0.90],
            [0.60, 0.80],
        ],
        dtype="<f4",
    ).reshape(9, 2, 1)
    # Development: rows 0-3; fixed Query: row 4 and in no split;
    # Validation: rows 5-8; Frozen Test is held out from this toy tune.
    development_mask = bytes([1, 1, 1, 1, 0, 0, 0, 0, 0])
    validation_mask = bytes([0, 0, 0, 0, 0, 1, 1, 1, 1])
    test_mask = bytes([0, 0, 0, 0, 0, 0, 0, 0, 0])
    ground_truth = np.asarray([0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
    write_json(data_root / "image-ids.json", image_ids)
    (data_root / "development-mask.u8").write_bytes(development_mask)
    (data_root / "validation-mask.u8").write_bytes(validation_mask)
    (data_root / "test-mask.u8").write_bytes(test_mask)
    (data_root / "ground-truth.u8").write_bytes(ground_truth.tobytes())
    (data_root / "raw-scores.f32").write_bytes(raw_scores.tobytes())
    (data_root / "ranks.f32").write_bytes(exported_ranks.tobytes())
    write_json(
        data_root / "manifest.json",
        {
            "schemaVersion": 2,
            "rowCount": 9,
            "methodCount": 2,
            "targetCount": 1,
            "methods": methods,
            "retrievalTargets": targets,
            "defaultRankMethod": "Ours-Full",
            "query": {
                "mode": "fixed",
                "images": [{"imageIndex": 4, "imageId": image_ids[4]}],
            },
            "files": {
                "imageIds": {"path": "image-ids.json"},
                "developmentMask": {"path": "development-mask.u8"},
                "validationMask": {"path": "validation-mask.u8"},
                "testMask": {"path": "test-mask.u8"},
                "groundTruth": {"path": "ground-truth.u8"},
                "rawScores": {"path": "raw-scores.f32"},
                "ranks": {"path": "ranks.f32"},
            },
        },
    )
    write_json(
        web_root / "public" / "data" / "catalog.json",
        {
            "schemaVersion": 2,
            "defaultDataset": "toy",
            "defaultTask": TASK_ID,
            "taskCount": 1,
            "datasets": [
                {
                    "id": "toy",
                    "label": "Toy",
                    "tasks": [
                        {
                            "id": TASK_ID,
                            "dataRoot": f"/data/tasks/{TASK_ID}",
                        }
                    ],
                }
            ],
        },
    )
    return web_root, image_ids


def vqa_contract_fixture(split):
    """Adapt synthetic members to the new versioned contract, without GT."""
    return SimpleNamespace(
        fit_indices=split.fit_indices, fit_labels=split.fit_labels,
        validation_indices=split.validation_indices,
        validation_labels=split.validation_labels,
        audit={
            **split.audit,
            "protocol": TUNING.VQA_VALIDATION_PROTOCOL,
            "frozenVersion": "toy-vqa-v1",
            "frozenManifestSha256": "a" * 64,
            "frozenTaskSha256": "b" * 64,
            "evaluationLabelSource": "original-vqa-supervision",
            "initialModelHoldoutIndependent": False,
            "referenceOnly": True,
        },
    )


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.cookie: str | None = None

    def request(self, method: str, path: str, body=None, headers=None):
        payload = None
        request_headers = dict(headers or {})
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        if self.cookie:
            request_headers["Cookie"] = self.cookie
        request = urllib.request.Request(
            self.base_url + path,
            data=payload,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                set_cookie = response.headers.get("Set-Cookie")
                if set_cookie:
                    self.cookie = set_cookie.split(";", 1)[0]
                content = response.read()
                return response.status, json.loads(content) if content else None
        except urllib.error.HTTPError as error:
            content = error.read()
            return error.code, json.loads(content) if content else None

    def request_bytes(self, method: str, path: str):
        request_headers = {"Accept": "application/octet-stream"}
        if self.cookie:
            request_headers["Cookie"] = self.cookie
        request = urllib.request.Request(
            self.base_url + path,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()


class TuningMathTests(unittest.TestCase):
    def test_annotation_hash_tracks_only_confirmed_failure_attributes(self):
        unconfirmed = {
            "rowIndex": 7,
            "imageId": "gallery/example.jpg",
            "label": -1,
            "failedAttributeIds": [],
            "suggestedFailedAttributeId": "first",
            "failureAttributionConfirmed": False,
        }
        changed_suggestion = {
            **unconfirmed,
            "suggestedFailedAttributeId": "second",
        }
        confirmed_first = {
            **unconfirmed,
            "failedAttributeIds": ["first"],
            "failureAttributionConfirmed": True,
        }
        confirmed_second = {
            **unconfirmed,
            "failedAttributeIds": ["second"],
            "failureAttributionConfirmed": True,
        }
        self.assertEqual(
            TUNING.TuningService.annotation_state_sha256([unconfirmed]),
            TUNING.TuningService.annotation_state_sha256([changed_suggestion]),
        )
        self.assertNotEqual(
            TUNING.TuningService.annotation_state_sha256([unconfirmed]),
            TUNING.TuningService.annotation_state_sha256([confirmed_first]),
        )
        self.assertNotEqual(
            TUNING.TuningService.annotation_state_sha256([confirmed_first]),
            TUNING.TuningService.annotation_state_sha256([confirmed_second]),
        )

    def test_candidate_frozen_split_ignores_stale_harness_task_attributes(self):
        experiment_root = WEB_ROOT.parents[2]
        linear_root = experiment_root / "probe_learning"
        scripts_root = linear_root / "scripts"
        for path in (linear_root, scripts_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import iterative_vqa_runner as runner

        paths = [f"gallery/image-{index}.jpg" for index in range(10)]
        adapter = SimpleNamespace(
            records={"relative_path": np.asarray(paths)},
            query_idx=[],
        )
        p2i = {path: index for index, path in enumerate(paths)}
        gt_by_attr = {
            "airplane": {path: 1 for path in paths},
            "directing": {path: 1 for path in paths},
        }
        original_attributes = runner.H.ATTRS
        try:
            # Simulate another user's Cutting-cake request mutating the shared
            # harness immediately before Directing-airplane resolves its split.
            runner.H.ATTRS = ["cake", "cutting"]
            gallery, test, train = runner._candidate_frozen_split(
                adapter,
                p2i,
                gt_by_attr,
            )
        finally:
            runner.H.ATTRS = original_attributes

        self.assertEqual(gallery, paths)
        self.assertEqual(len(test), 2)
        self.assertEqual(len(train), 8)
        self.assertEqual(set(test).union(train), set(paths))

    def test_average_tie_ranks(self):
        ranks = TUNING.normalized_ranks(np.asarray([1.0, 2.0, 2.0, 0.0]))
        np.testing.assert_allclose(ranks, [1 / 3, 5 / 6, 5 / 6, 0.0])

    def test_retrieval_metrics_reports_best_f1(self):
        metrics = TUNING.retrieval_metrics(
            np.asarray([1, 0, 1, 0], dtype=np.uint8),
            np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float32),
        )
        self.assertAlmostEqual(metrics["bestF1"], 0.8)
        self.assertEqual(metrics["bestF1Cutoff"], 3)

    def test_selected_seed_contract_accepts_supersets_and_rejects_invalid_caches(self):
        methods = ["mlp_baseline", "ours_full"]
        selected = [0, 1, 2, 3, 4]
        TUNING.require_selected_seeds(
            {method: list(range(10)) for method in methods},
            methods,
            selected,
        )

        with self.assertRaisesRegex(RuntimeError, "lacks selected seeds"):
            TUNING.require_selected_seeds(
                {methods[0]: [0, 1, 2, 3], methods[1]: list(range(10))},
                methods,
                selected,
            )
        with self.assertRaisesRegex(RuntimeError, "Duplicate cached seed"):
            TUNING.require_selected_seeds(
                {methods[0]: [0, 1, 2, 3, 4, 4], methods[1]: list(range(10))},
                methods,
                selected,
            )

    def test_fusion_weight_normalization_is_scale_invariant_and_rejects_invalid_input(self):
        weights = {
            learner: float(index + 1)
            for index, learner in enumerate(TUNING.WEIGHTED_FUSION_LEARNERS)
        }
        scaled = {learner: value * 3.0 for learner, value in weights.items()}
        normalized = TUNING.normalize_fusion_weights(weights)
        scaled_normalized = TUNING.normalize_fusion_weights(scaled)
        np.testing.assert_allclose(normalized, scaled_normalized, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(sum(normalized), 1.0)

        invalid_cases = {
            "missing learner": {
                key: value
                for key, value in weights.items()
                if key != TUNING.WEIGHTED_FUSION_LEARNERS[-1]
            },
            "unknown learner": {**weights, "Unknown Learner": 1.0},
            "all zero": {learner: 0.0 for learner in weights},
            "boolean": {**weights, TUNING.WEIGHTED_FUSION_LEARNERS[0]: True},
            "not numeric": {**weights, TUNING.WEIGHTED_FUSION_LEARNERS[0]: "bad"},
            "nan": {**weights, TUNING.WEIGHTED_FUSION_LEARNERS[0]: float("nan")},
            "infinity": {
                **weights,
                TUNING.WEIGHTED_FUSION_LEARNERS[0]: float("inf"),
            },
            "negative": {**weights, TUNING.WEIGHTED_FUSION_LEARNERS[0]: -0.1},
            "over maximum": {
                **weights,
                TUNING.WEIGHTED_FUSION_LEARNERS[0]: TUNING.MAX_FUSION_WEIGHT + 0.1,
            },
        }
        for label, invalid in invalid_cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                TUNING.normalize_fusion_weights(invalid)

    def test_hierarchical_fusion_normalizes_each_attribute_independently(self):
        attributes = ("first", "second")
        first = {
            learner: float(index + 1)
            for index, learner in enumerate(TUNING.WEIGHTED_FUSION_LEARNERS)
        }
        second = {learner: 1.0 for learner in TUNING.WEIGHTED_FUSION_LEARNERS}
        attribute_weights, learner_weights = TUNING.normalize_hierarchical_fusion_weights(
            attributes,
            {"first": 1.0, "second": 3.0},
            {"first": first, "second": second},
        )
        np.testing.assert_allclose(attribute_weights, [0.5, 1.5])
        self.assertAlmostEqual(sum(learner_weights[0]), 1.0)
        np.testing.assert_allclose(
            learner_weights[1],
            np.full(len(TUNING.WEIGHTED_FUSION_LEARNERS), 1 / 8),
        )

        with self.assertRaisesRegex(ValueError, "exactly the task attributes"):
            TUNING.normalize_hierarchical_fusion_weights(
                attributes,
                {"first": 1.0},
                {"first": first, "second": second},
            )
        with self.assertRaisesRegex(ValueError, "at least one attribute"):
            TUNING.normalize_hierarchical_fusion_weights(
                attributes,
                {"first": 0.0, "second": 0.0},
                {"first": first, "second": second},
            )

    def test_global_fusion_weights_are_fixed_order_normalized_and_strict(self):
        self.assertEqual(
            TUNING.normalize_global_fusion_weights(None),
            TUNING.OURS_ONLY_GLOBAL_WEIGHTS,
        )
        weights = {
            method: float(index + 1)
            for index, method in enumerate(TUNING.GLOBAL_FUSION_METHODS)
        }
        observed = TUNING.normalize_global_fusion_weights(weights)
        expected = np.arange(1, 7, dtype=np.float64) / 21.0
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)

        invalid_values = (
            {key: value for key, value in weights.items() if key != "Ours-Full"},
            {method: 0.0 for method in TUNING.GLOBAL_FUSION_METHODS},
            {**weights, "Ours-Full": True},
            {**weights, "Ours-Full": "1"},
            {**weights, "Ours-Full": -1.0},
            {**weights, "Ours-Full": TUNING.MAX_FUSION_WEIGHT + 1.0},
        )
        for invalid in invalid_values:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TUNING.normalize_global_fusion_weights(invalid)

    def test_rank_fusion_normalizes_outer_and_all_thirteen_inner_weights(self):
        attributes = ("first", "second")
        first = {
            method: float(index + 1)
            for index, method in enumerate(TUNING.RANK_FUSION_METHODS)
        }
        second = {method: 1.0 for method in TUNING.RANK_FUSION_METHODS}
        outer, inner = TUNING.normalize_rank_fusion_weights(
            attributes,
            {"first": 1.0, "second": 3.0},
            {"first": first, "second": second},
        )
        np.testing.assert_allclose(outer, [0.25, 0.75], rtol=0.0, atol=1e-15)
        self.assertEqual(len(inner), 2)
        self.assertTrue(all(len(row) == 13 for row in inner))
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-15 for row in inner))
        np.testing.assert_allclose(inner[1], np.full(13, 1 / 13))

        invalid_values = (
            ({"first": 1.0}, {"first": first, "second": second}),
            (
                {"first": 1.0, "second": 1.0},
                {"first": first},
            ),
            (
                {"first": 1.0, "second": 1.0},
                {
                    "first": {
                        key: value
                        for key, value in first.items()
                        if key != TUNING.RANK_FUSION_METHODS[-1]
                    },
                    "second": second,
                },
            ),
            (
                {"first": 0.0, "second": 0.0},
                {"first": first, "second": second},
            ),
            (
                {"first": True, "second": 1.0},
                {"first": first, "second": second},
            ),
        )
        for outer_value, inner_value in invalid_values:
            with self.subTest(outer=outer_value), self.assertRaises(ValueError):
                TUNING.normalize_rank_fusion_weights(
                    attributes,
                    outer_value,
                    inner_value,
                )

    def test_rank_fusion_uses_baseline_joint_learner_attribute_and_raw_outer_borda(self):
        row_count = 4
        attributes = ("first", "second")
        targets = (*attributes, "joint")
        methods = TUNING.RANK_FUSION_METHODS
        exported = np.full((row_count, 13, 3), 0.5, dtype="<f4")
        baseline_joint = np.asarray([0.1, 0.2, 0.4, 0.8], dtype=np.float32)
        exported[:, 0, 2] = baseline_joint
        exported[:, 0, 0] = 0.99  # decoy: baselines must not use attribute ranks
        exported[:, 0, 1] = 0.98
        learner_first = np.asarray([0.8, 0.6, 0.3, 0.1], dtype=np.float32)
        learner_second = np.asarray([0.05, 0.25, 0.75, 0.95], dtype=np.float32)
        learner_index = len(TUNING.EMBEDDING_BASELINE_METHODS)
        exported[:, learner_index, 0] = learner_first
        exported[:, learner_index, 1] = learner_second
        exported[:, learner_index, 2] = 0.97  # decoy: learners use each attribute

        with tempfile.TemporaryDirectory() as directory:
            rank_path = Path(directory) / "ranks.f32"
            exported.tofile(rank_path)
            task = TUNING.TaskSpec(
                task_id="rank-toy",
                dataset_id="toy",
                task_name="Rank toy",
                data_root=Path(directory),
                row_count=row_count,
                methods=methods,
                target_ids=targets,
                manifest={
                    "retrievalTargets": [
                        {"id": "first", "kind": "attribute"},
                        {"id": "second", "kind": "attribute"},
                        {"id": "joint", "kind": "derived"},
                    ],
                    "files": {"ranks": {"path": "ranks.f32"}},
                },
            )
            source_service = SimpleNamespace(
                _bundle_file=lambda _task, _path: rank_path,
                _rank_fusion_attribute_ids=lambda _task: attributes,
                _weighted_fusion_context=lambda _task_id: (_ for _ in ()).throw(
                    AssertionError("rank fusion must not load source tensors")
                ),
            )
            observed_attributes, sources = (
                TUNING.TuningService._rank_fusion_source_features(
                    source_service,
                    task,
                )
            )
            self.assertEqual(observed_attributes, attributes)
            np.testing.assert_array_equal(sources[:, 0, 0], baseline_joint)
            np.testing.assert_array_equal(sources[:, 1, 0], baseline_joint)
            np.testing.assert_array_equal(sources[:, 0, learner_index], learner_first)
            np.testing.assert_array_equal(sources[:, 1, learner_index], learner_second)

            compute_service = SimpleNamespace(
                task=lambda _task_id: task,
                _rank_fusion_source_features=lambda _task: (
                    observed_attributes,
                    sources,
                ),
                _frozen_train_indices=lambda _task_id: np.asarray(
                    [0, 1], dtype=np.int64
                ),
            )
            first_weights = np.zeros(13, dtype=np.float64)
            first_weights[0] = 1.0
            second_weights = np.zeros(13, dtype=np.float64)
            second_weights[learner_index] = 1.0
            result = TUNING.TuningService._compute_rank_fusion(
                compute_service,
                "rank-toy",
                (0.25, 0.75),
                (tuple(first_weights), tuple(second_weights)),
            )
            rounded_uniform = tuple(
                float(value)
                for value in np.full(13, 1.0 / 13.0, dtype=np.float32)
            )
            self.assertGreater(
                abs(sum(rounded_uniform) - 1.0),
                1e-12,
            )
            rounded_result = TUNING.TuningService._compute_rank_fusion(
                compute_service,
                "rank-toy",
                (0.5, 0.5),
                (rounded_uniform, rounded_uniform),
            )

        cube = np.frombuffer(result.payload, dtype="<f4").reshape(3, row_count, 3)
        expected_joint = 0.25 * baseline_joint + 0.75 * learner_second
        expected_raw = np.column_stack(
            [baseline_joint, learner_second, expected_joint]
        )
        np.testing.assert_allclose(cube[0], expected_raw, atol=1e-7)
        for target_index in range(3):
            values = expected_raw[:, target_index]
            low, high = values[:2].min(), values[:2].max()
            expected_calibrated = np.clip((values - low) / (high - low), 0, 1)
            np.testing.assert_allclose(
                cube[1, :, target_index], expected_calibrated, atol=1e-7
            )
            np.testing.assert_allclose(
                cube[2, :, target_index], TUNING.normalized_ranks(values), atol=1e-7
            )
        for row in rounded_result.normalized_method_weights_by_attribute:
            self.assertAlmostEqual(math.fsum(row), 1.0, places=15)

    def test_v4_rank_fusion_never_loads_legacy_learner_tensors(self):
        sentinel = object()
        legacy_calls = []

        def legacy_context(task_id):
            legacy_calls.append(task_id)
            return sentinel

        service = SimpleNamespace(_weighted_fusion_context=legacy_context)
        observed = TUNING.TuningService._fusion_context_for_algorithm(
            service,
            "rank-toy",
            TUNING.FUSION_WEIGHT_ALGORITHM_V4,
        )
        self.assertIsNone(observed)
        self.assertEqual(legacy_calls, [])

        observed_legacy = TUNING.TuningService._fusion_context_for_algorithm(
            service,
            "rank-toy",
            TUNING.FUSION_WEIGHT_ALGORITHM_V3,
        )
        self.assertIs(observed_legacy, sentinel)
        self.assertEqual(legacy_calls, ["rank-toy"])

    def test_attribute_rank_fusion_cluster_features_have_exact_29_columns(self):
        targets = ("first", "second", "joint")
        methods = TUNING.RANK_FUSION_METHODS
        fused = np.arange(2 * 3, dtype=np.float32).reshape(2, 3) / 100
        exported = (
            np.arange(2 * 13 * 3, dtype=np.float32).reshape(2, 13, 3) / 100
        )
        observed = TUNING.build_attribute_rank_fusion_cluster_features(
            fused,
            exported,
            target_ids=targets,
            method_labels=methods,
            attribute_ids=("first", "second"),
            joint_target_id="joint",
        )
        expected_columns = [fused[:, 2]]
        for target_index in (0, 1):
            expected_columns.append(fused[:, target_index])
            expected_columns.extend(exported[:, method_index, 2] for method_index in range(5))
            expected_columns.extend(
                exported[:, 5 + method_index, target_index]
                for method_index in range(8)
            )
        self.assertEqual(observed.shape, (2, 29))
        np.testing.assert_array_equal(observed, np.column_stack(expected_columns))

    def test_dynamic_pcp_features_always_include_the_complete_hierarchy(self):
        targets = ("first", "second", "joint")
        methods = (
            *TUNING.EMBEDDING_BASELINE_METHODS,
            *TUNING.WEIGHTED_FUSION_LEARNERS,
        )
        final = (
            np.arange(2 * len(targets), dtype=np.float32).reshape(2, -1) / 100
            + 0.5
        )
        hierarchy = np.arange(2 * len(targets), dtype=np.float32).reshape(2, -1) / 100
        exported = np.arange(
            2 * len(methods) * len(targets), dtype=np.float32
        ).reshape(2, len(methods), len(targets)) / 100

        observed = TUNING.build_full_hierarchical_rank_features(
            final,
            hierarchy,
            exported,
            target_ids=targets,
            method_labels=methods,
            attribute_ids=("first", "second"),
            overall_target_id="joint",
        )

        expected_columns = [final[:, 2], hierarchy[:, 2]]
        expected_columns.extend(
            exported[:, method_index, 2]
            for method_index in range(len(TUNING.EMBEDDING_BASELINE_METHODS))
        )
        for target_index in (0, 1):
            expected_columns.append(hierarchy[:, target_index])
            expected_columns.extend(
                exported[
                    :,
                    len(TUNING.EMBEDDING_BASELINE_METHODS) + method_index,
                    target_index,
                ]
                for method_index in range(len(TUNING.WEIGHTED_FUSION_LEARNERS))
            )
        expected = np.column_stack(expected_columns)
        self.assertEqual(observed.shape, (2, 25))
        np.testing.assert_array_equal(observed, expected)

    def test_absolute_pcp_clustering_fits_only_development_rows(self):
        rng = np.random.default_rng(7)
        centers = np.asarray(
            [[0.15, 0.15], [0.15, 0.85], [0.85, 0.15], [0.85, 0.85]],
            dtype=np.float32,
        )
        development = np.vstack(
            [center + rng.normal(0, 0.01, size=(6, 2)) for center in centers]
        ).astype(np.float32)
        first = np.vstack([development, np.full((12, 2), 0.02, dtype=np.float32)])
        second = np.vstack([development, np.full((12, 2), 0.98, dtype=np.float32)])
        mask = np.concatenate(
            [np.ones(len(development), dtype=np.uint8), np.zeros(12, dtype=np.uint8)]
        )

        first_labels, first_k, first_fit = TUNING.fit_dynamic_pcp_cluster_labels(
            first, mask, "absolute"
        )
        second_labels, second_k, second_fit = TUNING.fit_dynamic_pcp_cluster_labels(
            second, mask, "absolute"
        )

        np.testing.assert_array_equal(
            first_labels[: len(development)], second_labels[: len(development)]
        )
        self.assertEqual((first_k, first_fit), (4, len(development)))
        self.assertEqual((second_k, second_fit), (4, len(development)))
        self.assertEqual(first_labels.dtype, np.uint8)

    def test_shape_pcp_clustering_is_invariant_to_per_row_offsets(self):
        patterns = np.asarray(
            [
                [0.25, 0.35, 0.45, 0.55],
                [0.55, 0.45, 0.35, 0.25],
                [0.25, 0.55, 0.25, 0.55],
            ],
            dtype=np.float32,
        )
        ranks = np.repeat(patterns, 5, axis=0)
        offsets = np.tile(
            np.asarray([-0.05, 0.05, -0.03, 0.03, 0.0], dtype=np.float32),
            3,
        )[:, None]
        mask = np.ones(len(ranks), dtype=np.uint8)

        baseline, baseline_k, _ = TUNING.fit_dynamic_pcp_cluster_labels(
            ranks, mask, "shape"
        )
        shifted, shifted_k, _ = TUNING.fit_dynamic_pcp_cluster_labels(
            ranks + offsets, mask, "shape"
        )

        self.assertEqual((baseline_k, shifted_k), (3, 3))
        np.testing.assert_array_equal(baseline, shifted)

    def test_fine_pcp_clustering_supports_thirty_canonical_clusters(self):
        cluster_count = 30
        centers = np.full((cluster_count, cluster_count), 0.1, dtype=np.float32)
        np.fill_diagonal(centers, 0.9)
        rng = np.random.default_rng(11)
        ranks = np.vstack(
            [
                center + rng.normal(0, 0.002, size=(6, cluster_count))
                for center in centers
            ]
        ).astype(np.float32)
        np.clip(ranks, 0.0, 1.0, out=ranks)
        mask = np.ones(len(ranks), dtype=np.uint8)

        labels, observed_k, fit_count = TUNING.fit_dynamic_pcp_cluster_labels(
            ranks, mask, "fine30"
        )

        self.assertEqual(observed_k, cluster_count)
        self.assertEqual(fit_count, len(ranks))
        self.assertEqual(labels.dtype, np.uint8)
        np.testing.assert_array_equal(
            np.unique(labels), np.arange(cluster_count, dtype=np.uint8)
        )

    def test_dynamic_pcp_cache_single_flights_identical_requests(self):
        fake_service = object.__new__(TUNING.TuningService)
        fake_service._pcp_cluster_result_guard = threading.Lock()
        fake_service._pcp_cluster_results = {}
        fake_service._pcp_cluster_inflight = {}
        started = threading.Event()
        release = threading.Event()
        compute_count = 0
        count_guard = threading.Lock()
        expected = TUNING.DynamicPcpClusterResult(
            payload=b"\x00\x01",
            row_count=2,
            cluster_count=4,
            scheme="absolute",
            feature_count=25,
            fit_row_count=1,
        )

        def compute(
            task_id,
            attribute_weights,
            learner_weights,
            global_weights,
            scheme,
        ):
            nonlocal compute_count
            with count_guard:
                compute_count += 1
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return expected

        fake_service._compute_dynamic_pcp_clusters = compute
        arguments = (
            "toy",
            (1.0, 1.0),
            tuple(tuple([1 / 8] * 8) for _ in range(2)),
            TUNING.OURS_ONLY_GLOBAL_WEIGHTS,
            "absolute",
        )
        results = []
        errors = []

        def invoke():
            try:
                results.append(fake_service.pcp_clusters(*arguments))
            except BaseException as error:  # pragma: no cover - assertion aid
                errors.append(error)

        first = threading.Thread(target=invoke)
        second = threading.Thread(target=invoke)
        first.start()
        self.assertTrue(started.wait(timeout=2))
        second.start()
        release.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertFalse(errors)
        self.assertEqual(compute_count, 1)
        self.assertEqual(results, [expected, expected])

    def test_equal_hierarchical_fusion_reuses_exports_without_loading_source_tensors(self):
        row_count = 3
        target_count = 3
        exported = tuple(
            np.full((row_count, target_count), value, dtype=np.float32)
            for value in (0.2, 0.4, 0.6)
        )
        task = SimpleNamespace(
            manifest={
                "retrievalTargets": [
                    {"id": "first", "kind": "attribute"},
                    {"id": "second", "kind": "attribute"},
                    {"id": "joint", "kind": "derived"},
                ]
            },
            row_count=row_count,
            target_ids=("first", "second", "joint"),
        )
        fake_service = SimpleNamespace(
            task=lambda task_id: task,
            _exported_ours_full_arrays=lambda observed: exported,
            _weighted_fusion_context=lambda task_id: (_ for _ in ()).throw(
                AssertionError("equal hierarchy must not load source tensors")
            ),
            _frozen_train_indices=lambda task_id: (_ for _ in ()).throw(
                AssertionError("ours-only hierarchy must not reload train metadata")
            ),
        )
        result = TUNING.TuningService._compute_hierarchical_fusion(
            fake_service,
            "toy",
            (1.0, 1.0),
            tuple(
                tuple([1 / 8] * len(TUNING.WEIGHTED_FUSION_LEARNERS))
                for _ in range(2)
            ),
            TUNING.OURS_ONLY_GLOBAL_WEIGHTS,
        )
        self.assertTrue(result.baseline)
        self.assertEqual(result.baseline_max_abs_error, 0.0)
        self.assertEqual(
            result.normalized_global_weights,
            TUNING.OURS_ONLY_GLOBAL_WEIGHTS,
        )
        observed = np.frombuffer(result.payload, dtype="<f4")
        expected = np.concatenate(
            [
                exported[0].reshape(-1),
                exported[1].reshape(-1),
                exported[2].reshape(-1),
                exported[1].reshape(-1),
                exported[2].reshape(-1),
            ]
        )
        np.testing.assert_array_equal(observed, expected)

    def test_global_borda_changes_only_canonical_joint_and_uses_lightweight_calibration(self):
        row_count = 4
        target_ids = ("first", "second", "joint")
        hierarchy_raw = np.asarray(
            [
                [0.11, 0.21, 0.31],
                [0.12, 0.22, 0.32],
                [0.13, 0.23, 0.33],
                [0.14, 0.24, 0.34],
            ],
            dtype=np.float32,
        )
        hierarchy_calibrated = hierarchy_raw + np.float32(0.1)
        hierarchy_ranks = np.asarray(
            [
                [0.0, 1.0, 0.25],
                [0.3, 0.7, 0.75],
                [0.6, 0.4, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        baseline_joint = {
            method: np.asarray(
                [
                    (method_index + 1) / 20,
                    (method_index + 3) / 20,
                    (method_index + 2) / 20,
                    (method_index + 4) / 20,
                ],
                dtype=np.float32,
            )
            for method_index, method in enumerate(TUNING.EMBEDDING_BASELINE_METHODS)
        }
        task = SimpleNamespace(
            task_id="toy",
            manifest={
                "retrievalTargets": [
                    {"id": "first", "kind": "attribute"},
                    {"id": "second", "kind": "attribute"},
                    {"id": "joint", "kind": "derived"},
                ]
            },
            row_count=row_count,
            target_ids=target_ids,
        )
        requested_target_indices = []

        def exported_method_column(_task, file_key, method, target_index):
            self.assertEqual(file_key, "ranks")
            requested_target_indices.append(target_index)
            return baseline_joint[method]

        fake_service = SimpleNamespace(
            task=lambda task_id: task,
            _exported_ours_full_arrays=lambda _task: (
                hierarchy_raw,
                hierarchy_calibrated,
                hierarchy_ranks,
            ),
            _exported_method_column=exported_method_column,
            _frozen_train_indices=lambda task_id: np.asarray([0, 2]),
            _weighted_fusion_context=lambda task_id: (_ for _ in ()).throw(
                AssertionError("baseline global fusion must not load learner tensors")
            ),
        )
        global_weights = TUNING.normalize_global_fusion_weights(
            dict(
                zip(
                    TUNING.GLOBAL_FUSION_METHODS,
                    (2.0, 1.0, 2.0, 0.0, 1.0, 2.0),
                    strict=True,
                )
            )
        )
        result = TUNING.TuningService._compute_hierarchical_fusion(
            fake_service,
            "toy",
            (1.0, 1.0),
            tuple(
                tuple([1 / 8] * len(TUNING.WEIGHTED_FUSION_LEARNERS))
                for _ in range(2)
            ),
            global_weights,
        )

        cube_size = row_count * len(target_ids)
        packed = np.frombuffer(result.payload, dtype="<f4")
        self.assertEqual(packed.size, 5 * cube_size)
        raw, calibrated, ranks, inner_calibrated, inner_ranks = (
            packed[index * cube_size : (index + 1) * cube_size].reshape(
                row_count, len(target_ids)
            )
            for index in range(5)
        )
        sources = [hierarchy_ranks[:, 2]] + [
            baseline_joint[method] for method in TUNING.EMBEDDING_BASELINE_METHODS
        ]
        borda = sum(
            weight * np.asarray(source, dtype=np.float64)
            for weight, source in zip(global_weights, sources, strict=True)
        )
        low, high = float(borda[[0, 2]].min()), float(borda[[0, 2]].max())
        expected_calibrated = np.clip((borda - low) / (high - low), 0.0, 1.0)

        np.testing.assert_allclose(raw[:, 2], borda, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            calibrated[:, 2], expected_calibrated, rtol=0.0, atol=1e-7
        )
        np.testing.assert_array_equal(ranks[:, 2], TUNING.normalized_ranks(borda))
        np.testing.assert_array_equal(raw[:, :2], hierarchy_raw[:, :2])
        np.testing.assert_array_equal(
            calibrated[:, :2], hierarchy_calibrated[:, :2]
        )
        np.testing.assert_array_equal(ranks[:, :2], hierarchy_ranks[:, :2])
        np.testing.assert_array_equal(inner_calibrated, hierarchy_calibrated)
        np.testing.assert_array_equal(inner_ranks, hierarchy_ranks)
        self.assertEqual(requested_target_indices, [2] * 5)
        self.assertFalse(result.baseline)
        self.assertIsNone(result.baseline_max_abs_error)

    def test_merge_supervision_feedback_overrides_label_and_uses_high_weight(self):
        original_rows = np.asarray([0, 1, 2])
        original_labels = np.asarray([0, 1, 0])
        feedback_rows = np.asarray([0, 1, 2, 3, 4, 5, 6])
        feedback_labels = np.asarray([2, 1, 0, -1, 1, -2, 0])
        original_rows_before = original_rows.copy()
        original_labels_before = original_labels.copy()
        rows, labels, weights, audit = TUNING.TuningService._merge_tuning_supervision(
            original_rows,
            original_labels,
            feedback_rows,
            feedback_labels,
            feedback_weight=8.0,
        )
        np.testing.assert_array_equal(rows, [0, 1, 2, 3, 4, 5])
        # Row zero is corrected from original negative to strong positive.
        np.testing.assert_array_equal(labels, [1, 1, 0, 0, 1, 0])
        # Row one reinforces its positive label; row two's uncertain feedback
        # leaves the original negative row at weight one.
        np.testing.assert_allclose(weights, [16.0, 8.0, 1.0, 8.0, 8.0, 16.0])
        self.assertEqual(audit["feedbackCount"], 5)
        self.assertEqual(audit["newFeedbackCount"], 3)
        self.assertEqual(audit["feedbackExistingCount"], 2)
        self.assertEqual(audit["feedbackOverrideCount"], 1)
        self.assertEqual(audit["feedbackReinforceCount"], 1)
        self.assertEqual(audit["feedbackReviewOnlyCount"], 2)
        self.assertEqual(audit["combinedCount"], 6)
        self.assertEqual(audit["combinedPositiveCount"], 3)
        np.testing.assert_array_equal(original_rows, original_rows_before)
        np.testing.assert_array_equal(original_labels, original_labels_before)

    def test_annotation_supervision_relations_are_response_only_and_exact(self):
        original = {0: 0, 1: 1, 2: 1}

        def annotation(row_index, label):
            return {
                "row_index": row_index,
                "image_id": f"image-{row_index}.jpg",
                "label": label,
                "source": "manual",
                "created_at": "2026-08-15T00:00:00Z",
                "updated_at": "2026-08-15T00:00:00Z",
            }

        cases = (
            (annotation(3, 2), "new", "new", None, 1, True),
            (annotation(0, -1), "existing", "reinforce", 0, 0, True),
            (annotation(1, -2), "existing", "override", 1, 0, True),
            (annotation(2, 0), "existing", "uncertain", 1, None, False),
            (annotation(4, 0), "new", "uncertain", None, None, False),
        )
        for row, origin, relation, original_label, effective_label, included in cases:
            with self.subTest(row=row["row_index"], relation=relation):
                payload = TUNING.TuningService.annotation_with_supervision_json(row, original)
                supervision = payload.pop("supervision")
                self.assertEqual(supervision["origin"], origin)
                self.assertEqual(supervision["relation"], relation)
                self.assertEqual(supervision["originalLabel"], original_label)
                self.assertEqual(supervision["effectiveLabel"], effective_label)
                self.assertEqual(supervision["includedInTune"], included)
                self.assertNotIn("supervision", row)

    def test_merge_supervision_accepts_one_sided_feedback_when_original_dg_has_both_classes(self):
        for feedback_label in (2, -2):
            with self.subTest(feedback_label=feedback_label):
                rows, labels, weights, audit = TUNING.TuningService._merge_tuning_supervision(
                    np.asarray([0, 1, 2, 3]),
                    np.asarray([1, 1, 0, 0]),
                    np.asarray([4]),
                    np.asarray([feedback_label]),
                    feedback_weight=8.0,
                )
                np.testing.assert_array_equal(rows, [0, 1, 2, 3, 4])
                self.assertEqual(audit["feedbackCount"], 1)
                self.assertGreaterEqual(audit["combinedPositiveCount"], 2)
                self.assertGreaterEqual(audit["combinedNegativeCount"], 2)
                self.assertEqual(weights[-1], 16.0)
                self.assertEqual(labels[-1], int(feedback_label > 0))

    def test_prototype_and_label_heads_improve_toy_validation_ap(self):
        embeddings = np.asarray(
            [
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.9, 0.1],
            ],
            dtype=np.float32,
        )
        indices = np.asarray([0, 1, 2, 3])
        labels = np.asarray([2, 1, -2, -1])
        test_truth = np.asarray([1, 1, 0, 0], dtype=np.uint8)
        baseline = embeddings[4:] @ np.asarray([1.0, 0.0], dtype=np.float32)

        prototype_scores, _ = TUNING.prototype_tuned_scores(
            embeddings,
            np.asarray([1.0, 0.0], dtype=np.float32),
            indices,
            labels,
            alpha=2.0,
            beta=1.0,
        )
        label_scores, state = TUNING.label_tuned_scores(
            embeddings, indices, labels
        )
        baseline_ap = TUNING.average_precision(test_truth, baseline)
        self.assertGreater(
            TUNING.average_precision(test_truth, prototype_scores[4:]), baseline_ap
        )
        self.assertGreater(
            TUNING.average_precision(test_truth, label_scores[4:]), baseline_ap
        )
        self.assertEqual(state["coef"].shape, (1, 2))


class RuntimeRootTests(unittest.TestCase):
    def test_default_runtime_is_moved_outside_web_root_and_rotates_cookie_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            web_root = root / "web"
            legacy_root = web_root / "runtime" / "tuning"
            legacy_root.mkdir(parents=True)
            (legacy_root / "workbench.sqlite3").write_bytes(b"preserved-database")
            (legacy_root / ".cookie-secret").write_bytes(b"x" * 32)
            artifact = legacy_root / "users" / "example" / "labels.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("preserved-labels", encoding="utf-8")

            resolved = TUNING.resolve_runtime_root(web_root)

            self.assertEqual(resolved, (root / "runtime" / "tuning").resolve())
            self.assertFalse(legacy_root.exists())
            self.assertEqual(
                (resolved / "workbench.sqlite3").read_bytes(), b"preserved-database"
            )
            self.assertEqual(
                (resolved / "users" / "example" / "labels.json").read_text(
                    encoding="utf-8"
                ),
                "preserved-labels",
            )
            self.assertFalse((resolved / ".cookie-secret").exists())
            self.assertEqual(
                (resolved / "retired" / "legacy-cookie-secret.bin").read_bytes(),
                b"x" * 32,
            )

    def test_default_runtime_refuses_to_merge_two_mutable_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            web_root = root / "web"
            (web_root / "runtime" / "tuning").mkdir(parents=True)
            (root / "runtime" / "tuning").mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "refusing to merge"):
                TUNING.resolve_runtime_root(web_root)

    def test_explicit_runtime_root_is_not_migrated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            web_root = root / "web"
            legacy_root = web_root / "runtime" / "tuning"
            explicit_root = root / "explicit"
            legacy_root.mkdir(parents=True)

            resolved = TUNING.resolve_runtime_root(web_root, explicit_root)

            self.assertEqual(resolved, explicit_root.resolve())
            self.assertTrue(legacy_root.exists())

    def test_retired_cookie_is_exchanged_once_without_changing_user(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            web_root, _ = create_toy_web(root)
            runtime_root = root / "runtime"
            retired_secret = b"r" * 32
            retired_path = runtime_root / "retired" / "legacy-cookie-secret.bin"
            retired_path.parent.mkdir(parents=True)
            retired_path.write_bytes(retired_secret)
            service = TUNING.TuningService(
                web_root, runtime_root, start_worker=False
            )
            legacy_user_id = "user_legacy_cookie"
            now = TUNING.utc_now()
            with service.connect() as connection:
                connection.execute(
                    "INSERT INTO users(id,identity_key,identity_source,display_name,"
                    "created_at,last_seen_at) VALUES(?,?,?,?,?,?)",
                    (
                        legacy_user_id,
                        f"cookie:{legacy_user_id}",
                        "cookie",
                        "Legacy",
                        now,
                        now,
                    ),
                )
            signature = TUNING.hmac.new(
                retired_secret,
                legacy_user_id.encode("utf-8"),
                TUNING.hashlib.sha256,
            ).hexdigest()
            headers = {
                "Cookie": f"{TUNING.COOKIE_NAME}={legacy_user_id}.{signature}"
            }

            migrated, created = service.identify(headers, create=True)

            self.assertFalse(created)
            self.assertEqual(migrated["id"], legacy_user_id)
            with service.connect() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT count(*) FROM retired_cookie_migrations WHERE user_id=?",
                        (legacy_user_id,),
                    ).fetchone()[0],
                    1,
                )
            current_cookie = service.cookie_header(legacy_user_id, False).split(";", 1)[0]
            verified = service._verify_cookie(current_cookie.split("=", 1)[1])
            self.assertEqual(verified, (legacy_user_id, False))

            replacement, replacement_created = service.identify(headers, create=True)
            self.assertTrue(replacement_created)
            self.assertNotEqual(replacement["id"], legacy_user_id)


class TuningApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.web_root, self.image_ids = create_toy_web(self.root)
        self.runtime_root = self.root / "runtime"
        self.service = TUNING.TuningService(
            self.web_root, self.runtime_root, start_worker=False
        )
        original = SimpleNamespace(
            indices=np.asarray([0, 1], dtype=np.int64),
            labels=np.asarray([0, 1], dtype=np.uint8),
            selected_indices=np.asarray([0, 1, 5], dtype=np.int64),
            # Row five deliberately disagrees with public GT.  The provenance
            # endpoint must return this recovered VQA value, never GT.
            selected_labels=np.asarray([0, 1, 0], dtype=np.uint8),
            audit={
                "developmentCount": 2,
                "recoveredSupervisionHash": "default-toy-supervision",
            },
        )
        self.service._original_development_supervision = lambda task_id, target_id: original
        default_probe_split = SimpleNamespace(
            fit_indices=np.asarray([0, 1], dtype=np.int64),
            fit_labels=np.asarray([0, 1], dtype=np.uint8),
            validation_indices=np.asarray([5], dtype=np.int64),
            validation_labels=np.asarray([0], dtype=np.uint8),
            audit={
                "schemaVersion": 1,
                "protocol": "sklearn-stratified-train-test-split-v1",
                "replayKind": "derived-joint-probe-style",
                "seed": 0,
                "validationFraction": 0.2,
                "targetId": "joint",
                "targetKind": "derived",
                "sourceCount": 3,
                "fitCount": 2,
                "validationCount": 1,
                "validationPositiveCount": 0,
                "validationNegativeCount": 1,
                "sourceFingerprint": "default-toy-source",
                "fitFingerprint": "default-toy-fit",
                "validationFingerprint": "default-toy-probe-split",
            },
        )
        self.service._probe_validation_split = (
            lambda task_id, target_id: default_probe_split
        )
        default_clean_split = clean_validation_split(
            fit_indices=default_probe_split.fit_indices,
            fit_labels=default_probe_split.fit_labels,
            validation_indices=[6, 7, 8],
            validation_labels=[1, 0, 0],
            fingerprint="default-toy-clean-validation",
        )
        self.service._clean_validation_split = (
            lambda task_id, target_id: default_clean_split
        )
        self.service._vqa_validation_split = lambda task_id, target_id, version=None: vqa_contract_fixture(
            self.service._clean_validation_split(task_id, target_id)
        )
        self.server = TUNING.TuningHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.service.stop_worker()
        self.temporary.cleanup()

    def bootstrap(self, client: ApiClient, name: str, new_session=False):
        status, value = client.request(
            "POST",
            "/api/tuning/bootstrap",
            {
                "taskId": TASK_ID,
                "targetId": "joint",
                "baseMethod": "Ours-Full",
                "displayName": name,
                "newSession": new_session,
            },
        )
        self.assertEqual(status, 200, value)
        return value

    def historical_run_fixture(self, user_id, session_id, payload):
        """Construct historical jobs internally; their public POST is read-only.

        Preserve snapshot/worker/migration coverage without bypassing the HTTP
        guard in the dedicated request-contract tests.
        """
        try:
            return 202, {"run": self.service.create_run(user_id, session_id, payload)}
        except TUNING.ApiError as error:
            return error.status, {"error": error.message}

    def test_refinement_requires_confirmed_negative_failure_and_freezes_v4_base(self):
        original_task = self.service.tasks[TASK_ID]
        retrieval_targets = [
            {"id": "first", "label": "First", "kind": "attribute"},
            {"id": "second", "label": "Second", "kind": "attribute"},
            {
                "id": "joint",
                "label": "Joint",
                "kind": "derived",
                "members": ["first", "second"],
            },
        ]
        self.service.tasks[TASK_ID] = TUNING.TaskSpec(
            task_id=original_task.task_id,
            dataset_id=original_task.dataset_id,
            task_name=original_task.task_name,
            data_root=original_task.data_root,
            row_count=original_task.row_count,
            methods=original_task.methods,
            target_ids=("first", "second", "joint"),
            manifest={
                **original_task.manifest,
                "targetCount": 3,
                "retrievalTargets": retrieval_targets,
                # This focused API fixture has no three-target score cube;
                # omission exercises the model-only fallback suggestion.
                "files": {
                    key: value
                    for key, value in original_task.manifest["files"].items()
                    if key != "rawScores"
                },
            },
        )
        self.service.bundle.cache_clear()
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Unified refinement")
        session_id = bootstrap["session"]["id"]

        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/2",
            {"imageId": self.image_ids[2], "label": -1, "source": "manual"},
        )
        self.assertEqual(status, 200, value)
        annotation = value["annotation"]
        self.assertEqual(annotation["suggestedFailedAttributeId"], "first")
        self.assertEqual(annotation["failedAttributeIds"], [])
        self.assertFalse(annotation["failureAttributionConfirmed"])

        status, value = client.request(
            "POST",
            f"/api/tuning/sessions/{session_id}/runs",
            {"mode": "weight_staged", "baseMethod": "Ours-Full"},
        )
        self.assertEqual(status, 409, value)
        self.assertIn("Confirm at least one failed query attribute", value["error"])

        status, value = client.request(
            "POST",
            f"/api/tuning/sessions/{session_id}/runs",
            {"mode": "probe_joint", "baseMethod": "Ours-Full"},
        )
        self.assertEqual(status, 409, value)
        self.assertIn("Update Probes separately", value["error"])

        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/2",
            {
                "imageId": self.image_ids[2],
                "label": -1,
                "source": "manual",
                "failedAttributeIds": ["second"],
                "failureAttributionConfirmed": True,
            },
        )
        self.assertEqual(status, 200, value)
        self.assertEqual(value["annotation"]["failedAttributeIds"], ["second"])
        self.assertTrue(value["annotation"]["failureAttributionConfirmed"])

        self.service._refinement_base_context = lambda task_id, *args, **kwargs: SimpleNamespace(
            attribute_ids=("first", "second"),
            attribute_names=("first", "second"),
            learner_names=TUNING.WEIGHTED_FUSION_LEARNERS,
            embedding_names=TUNING.REFINEMENT_EMBEDDING_METHODS,
            initial_theta=np.asarray([0.5, 0.5], dtype=np.float32),
            temperature=np.asarray([0.1, 0.1], dtype=np.float32),
            base_state_fingerprint="toy-unified-base",
            normalization_audit=self.service._refinement_normalization_scope(task_id)[1],
        )
        calibration_scores = np.stack(
            [
                np.full((2, 8), 0.1, dtype=np.float32),
                np.full((2, 8), 0.9, dtype=np.float32),
            ],
            axis=0,
        )
        self.service._refinement_calibration_arrays = lambda task_id, base, **kwargs: (
            calibration_scores,
            np.asarray([[0, 0], [1, 1]], dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
        )
        status, value = client.request(
            "POST",
            f"/api/tuning/sessions/{session_id}/runs",
            {"mode": "weight_staged", "baseMethod": "Ours-Full"},
        )
        self.assertEqual(status, 202, value)
        run = value["run"]
        self.assertEqual(run["mode"], "weight_staged")
        snapshot_path = next(
            path
            for path in self.runtime_root.rglob("labels_snapshot.json")
            if path.parent.name == run["id"]
        )
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["schemaVersion"], 9)
        self.assertEqual(snapshot["params"]["gateCalibrationPolicy"], TUNING.FIXED_GATE_POLICY)
        self.assertEqual(snapshot["params"]["initialTheta"], [0.5, 0.5])
        np.testing.assert_array_equal(snapshot["params"]["initialTemperature"], np.full(2, 0.1, dtype=np.float32))
        self.assertEqual(snapshot["params"]["normalizationPolicy"], TUNING.REFINEMENT_NORMALIZATION_POLICY)
        self.assertEqual(snapshot["params"]["embeddingMethods"], list(TUNING.REFINEMENT_EMBEDDING_METHODS))
        self.assertEqual(snapshot["params"]["initialEmbeddingWeights"], {name: 0.5 for name in TUNING.REFINEMENT_EMBEDDING_METHODS})
        self.assertEqual(run["embeddingMethods"], list(TUNING.REFINEMENT_EMBEDDING_METHODS))
        self.assertEqual(snapshot["params"]["baseStateFingerprint"], "toy-unified-base")
        self.assertEqual(snapshot["params"]["trainingStrategy"], "staged")
        self.assertEqual(snapshot["params"]["regularizationLambda"], 0.05)
        self.assertEqual(snapshot["params"]["biasRegularizationEta"], 0.05)
        self.assertEqual(snapshot["params"]["learningRate"], 0.05)
        self.assertEqual(snapshot["params"]["outerIterations"], 3)
        self.assertEqual(
            snapshot["params"]["maxIterationsMeaning"],
            "total-optimizer-steps",
        )
        self.assertEqual(len(snapshot["params"]["initialTheta"]), 2)
        self.assertEqual(len(snapshot["params"]["initialThetaFingerprint"]), 64)
        self.assertEqual(
            snapshot["annotations"][0]["failedAttributeIds"], ["second"]
        )

    def test_weight_refinement_modes_execute_from_the_same_synthetic_base(self):
        self._check_weight_refinement_modes()

    def test_http_legacy_tuning_creation_is_read_only_but_history_remains_readable(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Read-only historical modes")
        session_id = bootstrap["session"]["id"]
        for mode in ("fusion-weight", "residual", "prototype", "label"):
            status, value = client.request(
                "POST", f"/api/tuning/sessions/{session_id}/runs", {"mode": mode},
            )
            self.assertEqual(status, 409, value)
            self.assertIn("read-only", value["error"])
        with self.service.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM model_runs").fetchone()[0], 0)
        # Existing historical rows are untouched and retain their read URLs.
        now = TUNING.utc_now()
        with self.service.connect() as connection:
            connection.execute(
                "INSERT INTO model_runs(id,session_id,user_id,mode,base_method,evaluation_scope,status,"
                "params_json,annotation_sha256,annotation_count,positive_count,negative_count,"
                "artifact_relpath,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("run_old_read_only", session_id, bootstrap["user"]["id"], "residual", "Ours-Full",
                 "test", "succeeded", "{}", "historical", 1, 1, 0, "old-unused", now),
            )
        status, value = client.request("GET", "/api/tuning/runs/run_old_read_only")
        self.assertEqual(status, 200, value)
        self.assertIn("scoresUrl", value["run"])
        self.assertIn("ranksUrl", value["run"])
        status, value = client.request(
            "POST", f"/api/tuning/sessions/{session_id}/runs", {"mode": "probe_joint"},
        )
        self.assertEqual(status, 409, value)
        self.assertIn("Update Probes separately", value["error"])

    def test_legacy_five_embedding_refinement_worker_preserves_its_branch(self):
        self._check_weight_refinement_modes(legacy=True)

    def test_pre_isolation_two_embedding_worker_preserves_its_normalizer(self):
        self._check_weight_refinement_modes(legacy="v3")

    def test_pre_fixed_gate_weight_worker_preserves_recalibration(self):
        self._check_weight_refinement_modes(legacy="v4")

    def test_pre_fixed_gate_native_worker_preserves_recalibration(self):
        self._check_weight_refinement_modes(legacy="probe-v2", native=True)

    def test_native_probe_modes_share_snapshot_and_preserve_before(self):
        self._check_weight_refinement_modes(native=True)

    def _check_weight_refinement_modes(
        self, *, legacy=False, native=False, allow_post_training_test=False
    ):
        if legacy is True:
            legacy = "v2"
        if not legacy:
            from unittest.mock import patch
            calibration_guard = patch("tuning_models.recalibrate_unified_theta",
                side_effect=AssertionError("New refinement must never recalibrate theta"))
            calibration_guard.start()
            self.addCleanup(calibration_guard.stop)
        if not allow_post_training_test:
            self.service._task_ground_truth = lambda task_id: self.fail(
                "VQA-only runs without Test rows must never read public ground truth"
            )
        balanced_split = clean_validation_split(
            fit_indices=[0, 1, 2, 3],
            fit_labels=[0, 0, 1, 1],
            validation_indices=[5, 6],
            validation_labels=[1, 0],
            fingerprint="synthetic-clean-validation",
        )
        self.service._clean_validation_split = (
            lambda task_id, target_id: balanced_split
        )
        # Initial theta calibration intentionally reuses the historical Probe
        # fit membership; keep that source aligned with the fit side embedded
        # in the Clean Validation contract.
        self.service._probe_validation_split = (
            lambda task_id, target_id: balanced_split
        )
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Weight comparison")
        session_id = bootstrap["session"]["id"]
        user_id = bootstrap["user"]["id"]
        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/2",
            {"imageId": self.image_ids[2], "label": 1, "source": "manual"},
        )
        self.assertEqual(status, 200, value)

        row_signal = np.linspace(0.05, 0.95, 9, dtype=np.float32)
        probe_features = np.repeat(
            row_signal[:, None, None],
            len(TUNING.WEIGHTED_FUSION_LEARNERS),
            axis=2,
        )
        embedding_features = np.repeat(
            row_signal[:, None],
            len(TUNING.REFINEMENT_EMBEDDING_METHODS),
            axis=1,
        )
        base = SimpleNamespace(
            attribute_ids=("first",),
            attribute_names=("first",),
            learner_names=TUNING.WEIGHTED_FUSION_LEARNERS,
            embedding_names=TUNING.REFINEMENT_EMBEDDING_METHODS,
            probe_features=probe_features,
            probe_min=np.zeros((1, 8), dtype=np.float32),
            probe_max=np.ones((1, 8), dtype=np.float32),
            temperature=np.asarray([0.2], dtype=np.float32),
            initial_theta=np.asarray([0.5], dtype=np.float32),
            embedding_features=embedding_features,
            embedding_min=np.zeros(2, dtype=np.float32),
            embedding_max=np.ones(2, dtype=np.float32),
            development_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
            base_state_fingerprint="shared-synthetic-base",
            normalization_audit=self.service._refinement_normalization_scope(TASK_ID)[1],
        )
        self.service._refinement_base_context = lambda task_id, *args, **kwargs: base
        native_calls = []
        if native:
            import native_probe_update as native_update
            from unittest.mock import patch

            base.raw_probe_probabilities = probe_features.copy()
            base = TUNING.RefinementBaseContext(**vars(base))
            bank = {"fingerprint": "native-bank"}
            request_body = {"protocol": native_update.PROTOCOL, "userId": user_id,
                            "sessionId": session_id, "taskId": TASK_ID,
                            "baseBankFingerprint": "native-bank", "baseStateFingerprint": base.base_state_fingerprint,
                            "normalizationContract": base.normalization_audit,
                            "codeFingerprint": "synthetic-native-code",
                            "attributeIds": list(base.attribute_ids),
                            "examples": [{"attributeId": "first", "update": True}]}
            request_body["snapshotId"] = native_update.digest(request_body)
            updated_raw = np.repeat((1 - probe_features)[None], 5, axis=0)
            def native_runner(request, bank, output):
                native_calls.append(request["snapshotId"])
                return updated_raw, {"frozen": True}
            self.service._run_native_probe_update = native_runner
            for mocked in (
                patch.object(native_update, "capability", return_value={"available": True}),
                patch.object(native_update, "resolve_bank", return_value=bank),
                patch.object(native_update, "prepare_request", return_value=request_body),
                patch.object(native_update, "current_code_fingerprint", return_value="synthetic-native-code"),
            ):
                mocked.start()
                self.addCleanup(mocked.stop)
            update = self.service.create_probe_update(user_id, session_id, {})
            self.service._run_probe_update_job(update["id"])
            self.assertEqual(self.service.owned_probe_update(update["id"], user_id)["status"], "succeeded")

        observed_fingerprints = []
        initial_payloads = []
        for mode in (("weight_staged", "weight_joint") if native else (("weight_joint",) if legacy else ("weight_staged", "weight_joint"))):
            status, value = client.request(
                "POST",
                f"/api/tuning/sessions/{session_id}/runs",
                {
                    "mode": mode,
                    "baseMethod": "Ours-Full",
                    "maxIterations": 10,
                    "outerIterations": 3,
                    **({"probeSource": "updated", "probeUpdateId": update["id"]} if native and not legacy else {}),
                },
            )
            self.assertEqual(status, 202, value)
            run_id = value["run"]["id"]
            if legacy:
                # Emulate an already queued schema-5/v2 artifact. Its five
                # columns and immutable snapshot must survive this upgrade.
                with self.service.connect() as connection:
                    old = connection.execute("SELECT * FROM model_runs WHERE id=?", (run_id,)).fetchone()
                if not native:
                    base.embedding_names = TUNING.REFINEMENT_EMBEDDING_METHODS if legacy in {"v3", "v4", "probe-v2"} else TUNING.EMBEDDING_BASELINE_METHODS
                    base.embedding_features = np.repeat(row_signal[:, None], len(base.embedding_names), axis=1)
                    base.embedding_min = np.zeros(len(base.embedding_names), dtype=np.float32)
                    base.embedding_max = np.ones(len(base.embedding_names), dtype=np.float32)
                if legacy in {"v2", "v3"}:
                    base.normalization_audit = None
                path = self.runtime_root / str(old["artifact_relpath"]) / "labels_snapshot.json"
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                params = snapshot["params"]
                if native:
                    mode = mode.replace("weight_", "probe_")
                    snapshot["mode"] = mode
                    params["nativeProbeRequest"] = request_body
                    params.pop("probeSource", None)
                    params.pop("probeSelectionProtocol", None)
                params["algorithmVersion"] = (TUNING.PRE_FIXED_GATE_REFINEMENT_ALGORITHMS if legacy in {"v4", "probe-v2"} else TUNING.PRE_ISOLATION_WEIGHT_REFINEMENT_ALGORITHMS if legacy == "v3" else TUNING.LEGACY_WEIGHT_REFINEMENT_ALGORITHMS)[mode]
                params["embeddingMethods"] = list(base.embedding_names)
                params["initialEmbeddingWeights"] = {name: 1.0 / len(base.embedding_names) for name in base.embedding_names}
                if legacy in {"v2", "v3"}:
                    params["normalizationPolicy"] = "development-minmax-fixed"
                    params.pop("normalizationContract", None)
                    params["baseStateContract"] = "pcp-conjunction-holistic-base-v1"
                for key in ("gateCalibrationPolicy", "initialTemperature", "initialTemperatureFingerprint"):
                    params.pop(key, None)
                from tuning_models import recalibrate_unified_theta
                calibration_z, calibration_y, calibration_w = self.service._refinement_calibration_arrays(
                    TASK_ID, base, vqa_validation_version=params["vqaValidationVersion"])
                old_theta = recalibrate_unified_theta(calibration_z, calibration_y, np.full((1, 8), 1/8),
                    base.initial_theta, base.temperature, calibration_sample_weights=calibration_w)
                params["initialTheta"] = old_theta.tolist()
                params["initialThetaFingerprint"] = hashlib.sha256(old_theta.astype("<f4").tobytes()).hexdigest()
                params["thetaCalibration"] = "original-vqa-minus-shared-validation"
                params["temperaturePolicy"] = "fixed-base-mean"
                snapshot["algorithmVersion"] = params["algorithmVersion"]
                snapshot["schemaVersion"] = 8 if native else 7 if legacy == "v4" else 6 if legacy == "v3" else 5
                payload = (TUNING.json_dumps(snapshot) + "\n").encode("utf-8")
                TUNING.atomic_write_bytes(path, payload)
                with self.service.connect() as connection:
                    connection.execute(
                        "UPDATE model_runs SET mode=?,params_json=?,snapshot_sha256=? WHERE id=?",
                        (mode, TUNING.json_dumps(params), hashlib.sha256(payload).hexdigest(), run_id),
                    )
            with self.service.connect() as connection:
                run = connection.execute(
                    "SELECT r.*,s.task_id,s.target_id FROM model_runs r "
                    "JOIN sessions s ON s.id=r.session_id "
                    "WHERE r.id=? AND r.user_id=?",
                    (run_id, user_id),
                ).fetchone()
            self.assertIsNotNone(run)
            before, after = self.service._execute_model_run(run)
            self.assertEqual(before["evaluationCount"], 2)
            self.assertEqual(after["evaluationCount"], 2)
            self.assertEqual(after["modelSummary"]["iterations"], 10)
            self.assertEqual(
                after["modelSummary"]["trainingStrategy"],
                "staged" if mode.endswith("_staged") else "joint",
            )
            self.assertEqual(after["modelSummary"]["probeUpdated"], native)
            run_directory = self.runtime_root / str(run["artifact_relpath"])
            self.assertTrue((run_directory / "model.npz").is_file())
            self.assertTrue((run_directory / "training_history.json").is_file())
            self.assertTrue((run_directory / "vqa_validation.json").is_file())
            self.assertFalse(after["splitAudit"]["initialModelHoldoutIndependent"])
            self.assertTrue(after["splitAudit"]["referenceOnly"])
            self.assertTrue((run_directory / "initial-scores.f32").is_file())
            self.assertTrue((run_directory / "initial-ranks.f32").is_file())
            self.assertEqual(
                (run_directory / "tuned-scores.f32").stat().st_size,
                9 * np.dtype("<f4").itemsize,
            )
            with np.load(run_directory / "model.npz", allow_pickle=False) as artifact:
                observed_fingerprints.append(
                    str(artifact["baseStateFingerprint"][0])
                )
                self.assertEqual(artifact["beta"].shape, (1, 8))
                self.assertEqual(artifact["gamma"].shape, (1,))
                self.assertEqual(artifact["embeddingWeights"].shape, (len(base.embedding_names),))
                self.assertEqual(artifact["embeddingMethods"].tolist(), list(base.embedding_names))
                self.assertEqual(artifact["rankingIndices"].shape, (9,))
                if not legacy:
                    np.testing.assert_array_equal(artifact["theta"], base.initial_theta)
                    np.testing.assert_array_equal(artifact["temperature"], base.temperature)
                    self.assertFalse(any("calibration" in item["stage"] for item in after["modelSummary"]["trainingHistory"]))
            metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["schemaVersion"], (8 if native else 7 if legacy == "v4" else 6 if legacy == "v3" else 5) if legacy else 9)
            self.assertEqual(list(after["modelSummary"]["embeddingWeights"]), list(base.embedding_names))
            with self.service.connect() as connection:
                connection.execute(
                    "UPDATE model_runs SET status='succeeded',before_json=?,after_json=? WHERE id=?",
                    (TUNING.json_dumps(before), TUNING.json_dumps(after), run_id),
                )
            completed = self.service.owned_run(run_id, user_id)
            visualization = self.service.refinement_visualization(completed)
            self.assertEqual(visualization.component_count, 12 + len(base.embedding_names))
            self.assertEqual(self.service.run_json(completed)["embeddingMethods"], list(base.embedding_names))
            if native:
                payload = np.frombuffer(visualization.payload, dtype="<f4").reshape(2, 9, visualization.component_count)
                np.testing.assert_array_equal(payload[0, :, 3:11], (1 - probe_features)[:, 0, :])
                initial_payloads.append((run_directory / "initial-scores.f32").read_bytes())
                from tuning_models import unified_weight_scores
                params = json.loads(str(run["params_json"]))
                initial = unified_weight_scores(
                    probe_features, embedding_features, np.full((1, 8), 1/8), np.ones(1), np.full(2, .5),
                    params["lambda0"], np.asarray(params["initialTheta"]), base.temperature,
                )
                np.testing.assert_array_equal(np.frombuffer(initial_payloads[-1], dtype="<f4"), initial.final_scores.astype(np.float32))
                state = after["modelSummary"]["probeState"]
                self.assertTrue(state["frozenDuringFusion"])
                self.assertTrue(state["sharedAcrossFusionSchedules"])
        if native:
            status, capabilities = client.request("GET", f"/api/tuning/tasks/{TASK_ID}/refinement-capabilities")
            self.assertEqual(status, 200, capabilities)
            self.assertEqual(capabilities["taskId"], TASK_ID)
            for capability_mode in TUNING.REFINEMENT_RUN_MODES:
                self.assertEqual(capabilities["refinementCapabilities"][capability_mode]["available"],
                                 capability_mode in TUNING.WEIGHT_REFINEMENT_RUN_MODES)
                self.assertEqual(capabilities["refinementCapabilities"][capability_mode]["algorithmVersion"], TUNING.RUN_ALGORITHM_VERSIONS[capability_mode])
            self.assertTrue(capabilities["refinementCapabilities"]["update_probes"]["available"])
            self.assertEqual(native_calls, [request_body["snapshotId"]])
            self.assertEqual(initial_payloads[0], initial_payloads[1])
            self.assertEqual(observed_fingerprints[0], observed_fingerprints[1])
            self.assertNotEqual(observed_fingerprints[0], base.base_state_fingerprint)
            return
        self.assertEqual(
            observed_fingerprints,
            ["shared-synthetic-base"] * (1 if legacy else 2),
        )

    def test_refinement_base_selects_exact_columns_and_pins_order_in_cache(self):
        row_signal = np.linspace(0.05, 0.85, 9, dtype=np.float64)
        self.service.task = lambda task_id: SimpleNamespace(
            row_count=9, target_ids=("first", "joint"),
            manifest={"retrievalTargets": [
                {"id": "first", "kind": "attribute"},
                {"id": "joint", "kind": "derived"},
            ]},
        )
        self.service.bundle = lambda task_id: SimpleNamespace(
            development_mask=bytes([1, 1, 1, 1, 0, 0, 0, 0, 0]), test_mask=bytes(9),
            query_indices=frozenset(), image_ids=tuple(f"image-{row}" for row in range(9)),
        )
        split = vqa_contract_fixture(clean_validation_split(
            fit_indices=[0, 1, 2, 5], fit_labels=[0, 0, 1, 1],
            validation_indices=[3, 6], validation_labels=[1, 0], fingerprint="normalizer-val",
        ))
        self.service._vqa_validation_split = lambda task_id, target_id, version=None: split
        source = SimpleNamespace(
            attributes=("first",), target_attributes=("first", None),
            learned_method_ids=TUNING.WEIGHTED_FUSION_LEARNERS,
            learned_method_labels=TUNING.WEIGHTED_FUSION_LEARNERS,
            selected_seeds=(0, 1, 2, 3, 4),
            seeds_by_method={name: [0, 1, 2, 3, 4] for name in TUNING.WEIGHTED_FUSION_LEARNERS},
            tensors={name: {"first": np.stack([row_signal ** (1 + seed * 0.2) for seed in range(5)])}
                     for name in TUNING.WEIGHTED_FUSION_LEARNERS},
            ours_full_root=Path("unused"), task_config={},
            pcp=SimpleNamespace(
                OURS_FULL_ID="full", task_key=lambda config: "toy",
                load_json=lambda path: {"audit": {"fits": {
                    str(seed): {"full": {"theta_by_attr": {"first": 0.5},
                                           "temperature_by_attr": {"first": 0.2}}}
                    for seed in range(5)
                }}},
            ),
        )
        self.service._weighted_fusion_context = lambda task_id: source
        requested = []
        def read_column(task, key, method, target_index):
            requested.append(method)
            return row_signal ** (1 + TUNING.EMBEDDING_BASELINE_METHODS.index(method))
        self.service._exported_method_column = read_column
        current = self.service._refinement_base_context(TASK_ID)
        self.assertEqual(requested, list(TUNING.REFINEMENT_EMBEDDING_METHODS))
        self.assertEqual(current.embedding_names, TUNING.REFINEMENT_EMBEDDING_METHODS)
        self.assertEqual(current.embedding_features.shape, (9, 2))
        self.assertIs(current, self.service._refinement_base_context(TASK_ID))
        raw_mean = source.tensors[TUNING.WEIGHTED_FUSION_LEARNERS[0]]["first"].mean(axis=0).astype(np.float32)
        expected_probe = np.clip((raw_mean - raw_mean[:3].min()) / (raw_mean[:3].max() - raw_mean[:3].min()), 0, 1)
        np.testing.assert_allclose(current.probe_features[:, 0, 0], expected_probe, atol=1e-7)
        self.assertEqual(current.development_indices.tolist(), [0, 1, 2])
        self.assertEqual(current.normalization_audit["fitCount"], 3)
        legacy = self.service._refinement_base_context(TASK_ID, TUNING.EMBEDDING_BASELINE_METHODS, isolate_validation=False)
        self.assertEqual(legacy.embedding_features.shape, (9, 5))
        self.assertNotEqual(current.base_state_fingerprint, legacy.base_state_fingerprint)
        self.assertEqual(legacy.development_indices.tolist(), [0, 1, 2, 3])
        self.assertGreater(float(legacy.probe_max[0, 0]), float(current.probe_max[0, 0]))
        reloaded = self.service._refinement_base_context(TASK_ID)
        self.assertEqual(reloaded.base_state_fingerprint, current.base_state_fingerprint)
        # Arbitrary held-out scores must never move either branch's MinMax.
        row_signal[[3, 6]] = 50.0
        for method in source.tensors:
            source.tensors[method]["first"][:, [3, 6]] = 50.0
        self.service._refinement_contexts.clear()
        changed_holdout = self.service._refinement_base_context(TASK_ID)
        for name in ("probe_min", "probe_max", "embedding_min", "embedding_max"):
            np.testing.assert_array_equal(getattr(current, name), getattr(changed_holdout, name))
        pinned_params = {
            "algorithmVersion": TUNING.RUN_ALGORITHM_VERSIONS["weight_joint"],
            "embeddingMethods": list(current.embedding_names),
            "vqaValidationVersion": current.normalization_audit["vqaValidationVersion"],
            "vqaValidationManifestSha256": current.normalization_audit["vqaValidationManifestSha256"],
            "normalizationPolicy": TUNING.REFINEMENT_NORMALIZATION_POLICY,
            "normalizationContract": dict(current.normalization_audit),
        }
        self.service._refinement_base_for_run(TASK_ID, "weight_joint", pinned_params)
        # Even identical rows under a different manifest are a different base;
        # an already queued run must refuse silently switching namespaces.
        split.audit["frozenManifestSha256"] = "c" * 64
        changed_manifest = self.service._refinement_base_context(TASK_ID)
        self.assertNotEqual(changed_holdout.base_state_fingerprint, changed_manifest.base_state_fingerprint)
        with self.assertRaisesRegex(RuntimeError, "normalization snapshot changed"):
            self.service._refinement_base_for_run(TASK_ID, "weight_joint", pinned_params)
        with self.assertRaisesRegex(RuntimeError, "methods or order"):
            self.service._refinement_base_context(TASK_ID, tuple(reversed(TUNING.REFINEMENT_EMBEDDING_METHODS)))
        for mode in ("weight_staged", "weight_joint"):
            with self.assertRaisesRegex(RuntimeError, "embedding order"):
                self.service._refinement_embedding_methods(mode, {
                    "algorithmVersion": TUNING.RUN_ALGORITHM_VERSIONS[mode],
                    "embeddingMethods": list(TUNING.EMBEDDING_BASELINE_METHODS),
                })

    def test_refinement_visualization_is_exact_and_clusters_its_frozen_snapshot(self):
        self._check_refinement_visualization(legacy=True)

    def test_two_embedding_visualization_exposes_exact_conjunction_even_at_zero_final(self):
        self._check_refinement_visualization(legacy=False)

    def _check_refinement_visualization(self, *, legacy):
        from tuning_models import unified_weight_scores

        embedding_methods = TUNING.EMBEDDING_BASELINE_METHODS if legacy else TUNING.REFINEMENT_EMBEDDING_METHODS
        embedding_count = len(embedding_methods)
        component_count = 3 + 9 + embedding_count

        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Refinement visualization")
        session_id = bootstrap["session"]["id"]
        user_id = bootstrap["user"]["id"]
        if not legacy:
            split = vqa_contract_fixture(clean_validation_split(
                fit_indices=[0, 1, 2, 5], fit_labels=[0, 0, 1, 1],
                validation_indices=[3, 6], validation_labels=[1, 0], fingerprint="visualization-val",
            ))
            self.service._vqa_validation_split = lambda task_id, target_id, version=None: split
        # A run freezes Ours-Full independently of the browser method that was
        # active when its owning session was created.
        with self.service.connect() as connection:
            connection.execute(
                "UPDATE sessions SET base_method='Image Prototype' WHERE id=?",
                (session_id,),
            )

        row_count = 9
        probe_features = np.empty((row_count, 1, 8), dtype=np.float32)
        row_signal = np.linspace(0.05, 0.85, row_count, dtype=np.float32)
        for learner_index in range(8):
            probe_features[:, 0, learner_index] = np.clip(
                row_signal + learner_index * 0.01, 0.0, 1.0
            )
        embedding_features = np.column_stack(
            [
                np.linspace(0.05 + index * 0.02, 0.75 + index * 0.02, row_count)
                for index in range(embedding_count)
            ]
        ).astype(np.float32)
        beta = np.asarray([[0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04]])
        gamma = np.asarray([1.4], dtype=np.float64)
        eta = np.asarray([0.40, 0.25, 0.15, 0.10, 0.10] if legacy else [0.4, 0.6], dtype=np.float64)
        theta = np.asarray([0.43], dtype=np.float64)
        temperature = np.asarray([0.20], dtype=np.float64)
        embedding_strength = 0.35 if legacy else 1.0
        if not legacy:
            embedding_features[0] = 0.0
        base_fingerprint = "exact-refinement-base"
        base = SimpleNamespace(
            attribute_ids=("first",),
            attribute_names=("First",),
            learner_names=TUNING.WEIGHTED_FUSION_LEARNERS,
            embedding_names=embedding_methods,
            probe_features=probe_features,
            probe_min=np.zeros((1, 8), dtype=np.float32),
            probe_max=np.ones((1, 8), dtype=np.float32),
            temperature=temperature.astype(np.float32),
            initial_theta=np.asarray([0.5], dtype=np.float32),
            embedding_features=embedding_features,
            embedding_min=np.zeros(embedding_count, dtype=np.float32),
            embedding_max=np.ones(embedding_count, dtype=np.float32),
            development_indices=np.arange(4 if legacy else 3, dtype=np.int64),
            base_state_fingerprint=base_fingerprint,
            normalization_audit=None if legacy else self.service._refinement_normalization_scope(TASK_ID)[1],
        )
        self.service._refinement_base_context = lambda task_id, *args, **kwargs: base
        expected = unified_weight_scores(
            probe_features,
            embedding_features,
            beta,
            gamma,
            eta,
            embedding_strength,
            theta,
            temperature,
            gamma_min=TUNING.REFINEMENT_GAMMA_MIN,
            gamma_max=TUNING.REFINEMENT_GAMMA_MAX,
        )
        expected_scores = np.column_stack(
            [
                expected.final_scores,
                expected.conjunction_scores,
                expected.gates[:, 0],
                *(probe_features[:, 0, index] for index in range(8)),
                expected.holistic_scores,
                *(embedding_features[:, index] for index in range(embedding_count)),
            ]
        ).astype(np.float32)
        expected_ranks = np.column_stack(
            [TUNING.normalized_ranks(expected_scores[:, index]) for index in range(component_count)]
        ).astype(np.float32)

        run_id = "run_exact_refinement_visualization"
        run_directory = self.runtime_root / "exact-refinement-artifacts"
        run_directory.mkdir(parents=True)
        np.savez(
            run_directory / "model.npz",
            beta=beta.astype(np.float32),
            gamma=gamma.astype(np.float32),
            embeddingWeights=eta.astype(np.float32),
            embeddingFusionStrength=np.float32(embedding_strength),
            theta=theta.astype(np.float32),
            temperature=temperature.astype(np.float32),
            attributeIds=np.asarray(["first"]),
            learnerMethods=np.asarray(TUNING.WEIGHTED_FUSION_LEARNERS),
            embeddingMethods=np.asarray(embedding_methods),
            baseStateFingerprint=np.asarray([base_fingerprint]),
        )
        expected.final_scores.astype("<f4").tofile(run_directory / "tuned-scores.f32")
        TUNING.normalized_ranks(expected.final_scores).astype("<f4").tofile(
            run_directory / "tuned-ranks.f32"
        )
        params = {
            "algorithmVersion": (TUNING.LEGACY_WEIGHT_REFINEMENT_ALGORITHMS if legacy else TUNING.RUN_ALGORITHM_VERSIONS)["weight_joint"],
            "embeddingMethods": list(embedding_methods),
            "attributeIds": ["first"],
            "gammaMin": TUNING.REFINEMENT_GAMMA_MIN,
            "gammaMax": TUNING.REFINEMENT_GAMMA_MAX,
        }
        if not legacy:
            params.update({
                "vqaValidationVersion": base.normalization_audit["vqaValidationVersion"],
                "vqaValidationManifestSha256": base.normalization_audit["vqaValidationManifestSha256"],
                "normalizationPolicy": TUNING.REFINEMENT_NORMALIZATION_POLICY,
                "normalizationContract": dict(base.normalization_audit),
            })
        with self.service.connect() as connection:
            connection.execute(
                "INSERT INTO model_runs(id,session_id,user_id,mode,base_method,"
                "evaluation_scope,status,params_json,annotation_sha256,annotation_count,"
                "positive_count,negative_count,artifact_relpath,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    session_id,
                    user_id,
                    "weight_joint",
                    "Ours-Full",
                    "clean-validation",
                    "succeeded",
                    json.dumps(params),
                    "annotation-state",
                    1,
                    1,
                    0,
                    run_directory.relative_to(self.runtime_root).as_posix(),
                    TUNING.utc_now(),
                ),
            )
        run = self.service.owned_run(run_id, user_id)
        public_run = self.service.run_json(run)
        self.assertIn("visualizationUrl", public_run)
        self.assertIn("clusterUrl", public_run)

        snapshot = self.service.refinement_visualization(run)
        self.assertEqual(snapshot.row_count, row_count)
        self.assertEqual(snapshot.component_count, component_count)
        decoded = np.frombuffer(snapshot.payload, dtype="<f4").reshape(
            2, row_count, snapshot.component_count
        )
        np.testing.assert_allclose(decoded[0], expected_scores, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(decoded[1], expected_ranks, rtol=0.0, atol=1e-7)
        # Independent formula check for C, preserving its distinction from g,
        # H and F. This also rejects an F/embedding-factor reconstruction.
        q = np.einsum("nam,am->na", probe_features.astype(np.float64), beta)
        log_gates = -np.logaddexp(0.0, -(q - theta) / temperature)
        exact_conjunction = np.exp(np.sum(gamma * log_gates, axis=1))
        np.testing.assert_allclose(decoded[0, :, 1], exact_conjunction, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(decoded[1, :, 1], TUNING.normalized_ranks(decoded[0, :, 1]))
        np.testing.assert_allclose(decoded[0, :, 0], decoded[0, :, 1] * ((1 - embedding_strength) + embedding_strength * decoded[0, :, 11]), atol=1e-7)
        if not legacy:
            self.assertEqual(decoded[0, 0, 0], 0.0)
            self.assertGreater(decoded[0, 0, 1], 0.0)
        self.assertEqual(snapshot.base_state_fingerprint, base_fingerprint)
        self.assertEqual(len(snapshot.source_fingerprint), 64)
        self.assertIs(self.service.refinement_visualization(run), snapshot)

        observed_cluster_inputs = []
        original_cluster_fit = TUNING.fit_dynamic_pcp_cluster_labels

        def fake_cluster_fit(features, development_mask, scheme):
            observed_cluster_inputs.append(
                (
                    np.asarray(features).copy(),
                    bytes(development_mask),
                    scheme,
                )
            )
            return np.asarray([index % 3 for index in range(row_count)], dtype=np.uint8), 3, len(base.development_indices)

        TUNING.fit_dynamic_pcp_cluster_labels = fake_cluster_fit
        try:
            clustered = self.service.refinement_clusters(run, "shape")
            cached_clustered = self.service.refinement_clusters(run, "shape")
        finally:
            TUNING.fit_dynamic_pcp_cluster_labels = original_cluster_fit
        self.assertIs(cached_clustered, clustered)
        self.assertEqual(len(observed_cluster_inputs), 1)
        np.testing.assert_array_equal(observed_cluster_inputs[0][0], expected_ranks)
        self.assertEqual(observed_cluster_inputs[0][1], bytes([1, 1, 1, 1 if legacy else 0, 0, 0, 0, 0, 0]))
        self.assertEqual(observed_cluster_inputs[0][2], "shape")
        self.assertEqual(clustered.feature_count, component_count)
        self.assertEqual(clustered.fit_row_count, 4 if legacy else 3)
        self.assertEqual(clustered.payload, bytes([0, 1, 2, 0, 1, 2, 0, 1, 2]))

        status, headers, body = client.request_bytes(
            "GET", public_run["visualizationUrl"]
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            headers["X-PCP-Algorithm"], TUNING.REFINEMENT_VISUALIZATION_ALGORITHM
        )
        self.assertEqual(headers["X-PCP-Layout"], "score,rank")
        self.assertEqual(headers["X-PCP-Component-Count"], str(component_count))
        self.assertEqual(headers["X-PCP-Source-Fingerprint"], snapshot.source_fingerprint)
        self.assertEqual(body, snapshot.payload)

        status, headers, body = client.request_bytes(
            "GET", f'{public_run["clusterUrl"]}?scheme=shape'
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-PCP-Algorithm"], TUNING.REFINEMENT_CLUSTER_ALGORITHM)
        self.assertEqual(headers["X-PCP-Feature-Basis"], TUNING.REFINEMENT_CLUSTER_FEATURE_BASIS)
        self.assertEqual(headers["X-PCP-Feature-Count"], str(component_count))
        self.assertEqual(headers["X-PCP-Fit-Scope"], "development")
        self.assertEqual(body, clustered.payload)

        legacy_run_id = "run_legacy_refinement_visualization"
        legacy_params = {"algorithmVersion": "conjunction-holistic-weight-joint-v1"}
        with self.service.connect() as connection:
            connection.execute(
                "INSERT INTO model_runs(id,session_id,user_id,mode,base_method,"
                "evaluation_scope,status,params_json,annotation_sha256,annotation_count,"
                "positive_count,negative_count,artifact_relpath,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    legacy_run_id,
                    session_id,
                    user_id,
                    "weight_joint",
                    "Ours-Full",
                    "probe-validation",
                    "succeeded",
                    json.dumps(legacy_params),
                    "legacy-annotation-state",
                    1,
                    1,
                    0,
                    run_directory.relative_to(self.runtime_root).as_posix(),
                    TUNING.utc_now(),
                ),
            )
        legacy_run = self.service.owned_run(legacy_run_id, user_id)
        legacy_public = self.service.run_json(legacy_run)
        self.assertIn("scoresUrl", legacy_public)
        self.assertNotIn("visualizationUrl", legacy_public)
        self.assertNotIn("clusterUrl", legacy_public)
        with self.assertRaisesRegex(TUNING.ApiError, "Legacy refinement"):
            self.service.refinement_visualization(legacy_run)

    def test_original_supervision_endpoint_is_target_scoped_and_never_uses_gt(self):
        client = ApiClient(self.base_url)
        status, value = client.request(
            "GET",
            f"/api/tuning/tasks/{TASK_ID}/targets/joint/original-supervision",
        )
        self.assertEqual(status, 200, value)
        self.assertEqual(value["taskId"], TASK_ID)
        self.assertEqual(value["targetId"], "joint")
        self.assertEqual(value["selectedCount"], 3)
        self.assertEqual(value["developmentCount"], 2)
        self.assertEqual(
            value["items"],
            [
                {
                    "rowIndex": 0,
                    "imageId": self.image_ids[0],
                    "label": 0,
                    "trainableInDevelopment": True,
                },
                {
                    "rowIndex": 1,
                    "imageId": self.image_ids[1],
                    "label": 1,
                    "trainableInDevelopment": True,
                },
                {
                    "rowIndex": 5,
                    "imageId": self.image_ids[5],
                    "label": 0,
                    "trainableInDevelopment": False,
                },
            ],
        )
        # Toy GT for row five is positive, proving the returned negative came
        # from recovered VQA supervision rather than ground-truth.u8.
        self.assertEqual(
            (self.web_root / "public" / "data" / "tasks" / TASK_ID / "ground-truth.u8")
            .read_bytes()[5],
            1,
        )
        self.assertNotIn("audit", value)

        status, value = client.request(
            "GET",
            f"/api/tuning/tasks/{TASK_ID}/targets/missing/original-supervision",
        )
        self.assertEqual(status, 400, value)
        self.assertIn("Unknown target", value["error"])

    def test_real_clean_validation_split_excludes_all_original_and_reviewed_rows(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Clean Validation exclusions")
        session_id = bootstrap["session"]["id"]
        user_id = bootstrap["user"]["id"]
        now = TUNING.utc_now()
        # Insert a historical event directly so the test exercises the real
        # cross-session exclusion query even though Validation is API read-only.
        with self.service.connect() as connection:
            connection.execute(
                "INSERT INTO annotation_events(session_id,user_id,row_index,image_id,"
                "previous_label,label,event,source,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    user_id,
                    7,
                    self.image_ids[7],
                    None,
                    1,
                    "upsert",
                    "legacy-import",
                    now,
                ),
            )

        del self.service._clean_validation_split
        split = self.service._build_clean_validation_split(TASK_ID, "joint")

        self.assertEqual(split.validation_indices.tolist(), [6, 8])
        self.assertEqual(split.validation_labels.tolist(), [1, 0])
        self.assertNotIn(5, split.validation_indices.tolist())
        self.assertNotIn(7, split.validation_indices.tolist())
        self.assertTrue(split.audit["allOriginalSupervisionExcluded"])
        self.assertTrue(split.audit["allHistoricalFeedbackExcluded"])
        self.assertEqual(split.audit["excludedOriginalSupervisionCount"], 1)
        self.assertEqual(split.audit["excludedHistoricalFeedbackCount"], 1)
        self.assertEqual(split.audit["validationPositiveCount"], 1)
        self.assertEqual(split.audit["validationNegativeCount"], 1)

    def test_probe_tuning_supervision_excludes_clean_validation_feedback(self):
        split = clean_validation_split(
            fit_indices=[0, 1, 2, 3],
            fit_labels=[0, 1, 0, 1],
            validation_indices=[5, 6],
            validation_labels=[1, 0],
            fingerprint="clean-validation-holdout",
        )
        self.service._clean_validation_split = lambda task_id, target_id: split
        rows, labels, weights, audit, observed = (
            self.service._prepare_probe_tuning_supervision(
                TASK_ID,
                "joint",
                np.asarray([1, 5, 7], dtype=np.int64),
                np.asarray([-2, 2, 2], dtype=np.int8),
                feedback_weight=8.0,
            )
        )
        self.assertIs(observed, split)
        self.assertNotIn(5, rows.tolist())
        self.assertIn(7, rows.tolist())
        self.assertEqual(audit["feedbackSnapshotCount"], 3)
        self.assertEqual(audit["feedbackHoldoutExcludedCount"], 1)
        self.assertEqual(audit["originalProbeFitCount"], 4)
        self.assertEqual(audit["probeValidationCount"], 0)
        self.assertEqual(audit["cleanValidationCount"], 2)
        self.assertEqual(len(rows), len(labels))
        self.assertEqual(len(rows), len(weights))

        with self.assertRaisesRegex(RuntimeError, "All feedback labels fall"):
            self.service._prepare_probe_tuning_supervision(
                TASK_ID,
                "joint",
                np.asarray([5], dtype=np.int64),
                np.asarray([2], dtype=np.int8),
                feedback_weight=8.0,
            )

    def test_original_image_endpoint_is_row_aligned_cached_and_path_confined(self):
        source_root = self.root / "source-images"
        valid_path = source_root / self.image_ids[0]
        valid_payload = b"toy-jpeg-payload"
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        valid_path.write_bytes(valid_payload)
        outside_path = self.root / "outside.jpg"
        outside_path.write_bytes(b"must-not-be-served")

        class FakeAdapter:
            raw_images_dir = source_root

            def resolve_image_path(self, image_id):
                if image_id == self_image_ids[1]:
                    return outside_path
                return source_root / image_id

        self_image_ids = self.image_ids
        self.service._source_adapter = lambda task_id: FakeAdapter()

        with urllib.request.urlopen(
            f"{self.base_url}/api/tuning/images/{TASK_ID}/0", timeout=5
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "image/jpeg")
            self.assertEqual(response.headers.get("Cache-Control"), "private, max-age=86400")
            self.assertTrue(response.headers.get("ETag"))
            self.assertEqual(response.read(), valid_payload)

        for row_index, expected_status in ((1, 404), (99, 400)):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(
                    f"{self.base_url}/api/tuning/images/{TASK_ID}/{row_index}", timeout=5
                )
            self.assertEqual(raised.exception.code, expected_status)

    def test_weighted_fusion_endpoint_is_identity_free_binary_and_origin_protected(self):
        row_count = len(self.image_ids)
        target_count = 1
        expected_values = np.arange(
            row_count * target_count * 3,
            dtype="<f4",
        )
        expected_payload = expected_values.tobytes(order="C")
        calls = []

        def fake_weighted_fusion(task_id, normalized_weights):
            calls.append((task_id, normalized_weights))
            return TUNING.WeightedFusionResult(
                payload=expected_payload,
                row_count=row_count,
                target_count=target_count,
                equal_weights=True,
                baseline_max_abs_error=0.0,
                normalized_weights=normalized_weights,
            )

        self.service.weighted_fusion = fake_weighted_fusion
        self.service.identify = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("weighted fusion must not require an identity")
        )
        weights = {
            learner: 1.0 for learner in TUNING.WEIGHTED_FUSION_LEARNERS
        }
        body = json.dumps({"taskId": TASK_ID, "weights": weights}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/tuning/weighted-fusion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response_payload = response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "application/octet-stream")
            self.assertEqual(response.headers.get("Cache-Control"), "private, no-store")
            self.assertEqual(response.headers.get("Content-Length"), str(len(expected_payload)))
            self.assertEqual(response.headers.get("X-PCP-Algorithm"), "weighted-ours-full-v1")
            self.assertEqual(response.headers.get("X-PCP-Layout"), "raw,calibrated,rank")
            self.assertEqual(response.headers.get("X-PCP-Row-Count"), str(row_count))
            self.assertEqual(response.headers.get("X-PCP-Target-Count"), str(target_count))
            self.assertEqual(response.headers.get("X-PCP-Equal-Weights"), "1")
            self.assertEqual(response.headers.get("X-PCP-Baseline-Max-Abs-Error"), "0")
            self.assertIsNone(response.headers.get("Set-Cookie"))
        self.assertEqual(response_payload, expected_payload)
        self.assertEqual(len(response_payload), row_count * target_count * 3 * 4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], TASK_ID)
        np.testing.assert_allclose(
            calls[0][1],
            np.full(len(TUNING.WEIGHTED_FUSION_LEARNERS), 1 / 8),
            rtol=0.0,
            atol=0.0,
        )

        status, value = ApiClient(self.base_url).request(
            "POST",
            "/api/tuning/weighted-fusion",
            {"taskId": TASK_ID, "weights": weights},
            headers={"Origin": "https://attacker.invalid"},
        )
        self.assertEqual(status, 403, value)
        self.assertIn("Cross-origin", value["error"])
        self.assertEqual(len(calls), 1)

    def test_hierarchical_fusion_endpoint_defaults_global_weights_and_returns_five_segments(self):
        row_count = len(self.image_ids)
        target_count = 3
        expected_values = np.arange(
            row_count * target_count * 5,
            dtype="<f4",
        )
        expected_payload = expected_values.tobytes(order="C")
        calls = []
        original_task = self.service.task
        task = SimpleNamespace(
            manifest={
                "retrievalTargets": [
                    {"id": "first", "kind": "attribute"},
                    {"id": "second", "kind": "attribute"},
                    {"id": "joint", "kind": "derived"},
                ]
            }
        )
        self.service.task = lambda task_id: task

        def fake_hierarchical_fusion(
            task_id,
            normalized_attribute_weights,
            normalized_learner_weights,
            normalized_global_weights,
        ):
            calls.append(
                (
                    task_id,
                    normalized_attribute_weights,
                    normalized_learner_weights,
                    normalized_global_weights,
                )
            )
            return TUNING.HierarchicalFusionResult(
                payload=expected_payload,
                row_count=row_count,
                target_count=target_count,
                baseline=True,
                baseline_max_abs_error=0.0,
                normalized_attribute_weights=normalized_attribute_weights,
                normalized_learner_weights=normalized_learner_weights,
                normalized_global_weights=normalized_global_weights,
            )

        self.service.hierarchical_fusion = fake_hierarchical_fusion
        learner_weights = {
            attribute: {
                learner: 1.0 for learner in TUNING.WEIGHTED_FUSION_LEARNERS
            }
            for attribute in ("first", "second")
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/tuning/hierarchical-fusion",
            data=json.dumps(
                {
                    "taskId": TASK_ID,
                    "attributeWeights": {"first": 1.0, "second": 1.0},
                    "learnerWeightsByAttribute": learner_weights,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                response_payload = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers.get("X-PCP-Algorithm"),
                    "hierarchical-ours-full-global-rank-v2",
                )
                self.assertEqual(
                    response.headers.get("X-PCP-Layout"),
                    "raw,calibrated,rank,hierarchy-calibrated,hierarchy-rank",
                )
                self.assertEqual(response.headers.get("X-PCP-Baseline"), "1")
            self.assertEqual(response_payload, expected_payload)
            self.assertEqual(len(response_payload), row_count * target_count * 5 * 4)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][3], TUNING.OURS_ONLY_GLOBAL_WEIGHTS)
        finally:
            self.service.task = original_task

    def test_dynamic_pcp_cluster_endpoint_has_a_strict_uint8_binary_contract(self):
        row_count = len(self.image_ids)
        expected_payload = bytes(index % 4 for index in range(row_count))
        calls = []
        original_task = self.service.task
        task = SimpleNamespace(
            manifest={
                "retrievalTargets": [
                    {"id": "first", "kind": "attribute"},
                    {"id": "second", "kind": "attribute"},
                    {"id": "joint", "kind": "derived"},
                ]
            }
        )
        self.service.task = lambda task_id: task

        def fake_pcp_clusters(
            task_id,
            normalized_attribute_weights,
            normalized_learner_weights,
            normalized_global_weights,
            scheme,
        ):
            calls.append(
                (
                    task_id,
                    normalized_attribute_weights,
                    normalized_learner_weights,
                    normalized_global_weights,
                    scheme,
                )
            )
            return TUNING.DynamicPcpClusterResult(
                payload=expected_payload,
                row_count=row_count,
                cluster_count=4,
                scheme="absolute",
                feature_count=25,
                fit_row_count=4,
            )

        self.service.pcp_clusters = fake_pcp_clusters
        self.service.identify = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("PCP clustering must not require an identity")
        )
        learner_weights = {
            attribute: {
                learner: 1.0 for learner in TUNING.WEIGHTED_FUSION_LEARNERS
            }
            for attribute in ("first", "second")
        }
        global_weights = {
            method: float(index + 1)
            for index, method in enumerate(TUNING.GLOBAL_FUSION_METHODS)
        }
        body = json.dumps(
            {
                "taskId": TASK_ID,
                "attributeWeights": {"first": 1.0, "second": 1.0},
                "learnerWeightsByAttribute": learner_weights,
                "globalWeights": global_weights,
                "scheme": "absolute",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/tuning/pcp-clusters",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                response_payload = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers.get_content_type(), "application/octet-stream"
                )
                self.assertEqual(response.headers.get("Content-Encoding"), None)
                self.assertEqual(response.headers.get("Content-Length"), str(row_count))
                self.assertEqual(
                    response.headers.get("X-PCP-Algorithm"),
                    TUNING.DYNAMIC_PCP_CLUSTER_ALGORITHM,
                )
                self.assertEqual(response.headers.get("X-PCP-Layout"), "labels")
                self.assertEqual(response.headers.get("X-PCP-Row-Count"), str(row_count))
                self.assertEqual(response.headers.get("X-PCP-Cluster-Scheme"), "absolute")
                self.assertEqual(response.headers.get("X-PCP-Cluster-Count"), "4")
                self.assertEqual(
                    response.headers.get("X-PCP-Feature-Basis"),
                    TUNING.DYNAMIC_PCP_CLUSTER_FEATURE_BASIS,
                )
                self.assertEqual(response.headers.get("X-PCP-Feature-Count"), "25")
                self.assertEqual(response.headers.get("X-PCP-Fit-Scope"), "development")
                self.assertEqual(response.headers.get("X-PCP-Fit-Row-Count"), "4")
                self.assertIsNone(response.headers.get("Set-Cookie"))
            self.assertEqual(response_payload, expected_payload)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], TASK_ID)
            self.assertEqual(calls[0][4], "absolute")
            np.testing.assert_allclose(calls[0][1], [1.0, 1.0])
            for weights in calls[0][2]:
                np.testing.assert_allclose(weights, np.full(8, 1 / 8))
            np.testing.assert_allclose(
                calls[0][3],
                np.arange(1, 7, dtype=np.float64) / 21.0,
            )
        finally:
            self.service.task = original_task

    def test_wal_cookie_user_isolation_and_non_development_rejection(self):
        with self.service.connect() as connection:
            self.assertEqual(
                connection.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal"
            )

        first = ApiClient(self.base_url)
        second = ApiClient(self.base_url)
        first_bootstrap = self.bootstrap(first, "Alice")
        second_bootstrap = self.bootstrap(second, "Bob")
        self.assertNotEqual(first_bootstrap["user"]["id"], second_bootstrap["user"]["id"])
        self.assertNotEqual(
            first_bootstrap["session"]["id"], second_bootstrap["session"]["id"]
        )

        first_session = first_bootstrap["session"]["id"]
        second_session = second_bootstrap["session"]["id"]
        status, value = first.request(
            "PUT",
            f"/api/tuning/sessions/{first_session}/annotations/0",
            {"imageId": self.image_ids[0], "label": 2, "source": "top"},
        )
        self.assertEqual(status, 200, value)
        self.assertEqual(value["counts"]["usablePositive"], 1)

        status, value = second.request(
            "PUT",
            f"/api/tuning/sessions/{second_session}/annotations/0",
            {"imageId": self.image_ids[0], "label": -2, "source": "top"},
        )
        self.assertEqual(status, 200, value)
        self.assertEqual(value["counts"]["usableNegative"], 1)

        status, value = first.request(
            "PUT",
            f"/api/tuning/sessions/{first_session}/annotations/4",
            {"imageId": self.image_ids[4], "label": 1, "source": "top"},
        )
        self.assertEqual(status, 409)
        self.assertIn("Fixed Query", value["error"])

        status, value = first.request(
            "PUT",
            f"/api/tuning/sessions/{first_session}/annotations/5",
            {"imageId": self.image_ids[5], "label": 1, "source": "top"},
        )
        self.assertEqual(status, 409)
        self.assertIn("Validation", value["error"])

        status, value = first.request(
            "PUT",
            f"/api/tuning/sessions/{first_session}/annotations/1",
            {"imageId": "wrong-id.jpg", "label": 1, "source": "top"},
        )
        self.assertEqual(status, 400)
        self.assertIn("does not match", value["error"])

        with self.service.connect() as connection:
            rows = connection.execute(
                "SELECT session_id,label FROM annotations ORDER BY session_id"
            ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual({int(row["label"]) for row in rows}, {-2, 2})

        effective = list(self.runtime_root.rglob("effective_labels.json"))
        events = list(self.runtime_root.rglob("annotations.jsonl"))
        self.assertEqual(len(effective), 2)
        self.assertEqual(len(events), 2)

    def test_put_and_bootstrap_report_new_override_reinforce_and_review_only(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Relation audit")
        session_id = bootstrap["session"]["id"]
        expected = {
            0: ("existing", "reinforce"),
            1: ("existing", "override"),
            2: ("new", "new"),
            3: ("new", "uncertain"),
        }
        for row_index, label in ((0, -1), (1, -2), (2, 2), (3, 0)):
            status, value = client.request(
                "PUT",
                f"/api/tuning/sessions/{session_id}/annotations/{row_index}",
                {
                    "imageId": self.image_ids[row_index],
                    "label": label,
                    "source": "top",
                },
            )
            self.assertEqual(status, 200, value)
            supervision = value["annotation"]["supervision"]
            self.assertEqual(
                (supervision["origin"], supervision["relation"]),
                expected[row_index],
            )

        refreshed = self.bootstrap(client, "Relation audit")
        by_row = {row["rowIndex"]: row for row in refreshed["annotations"]}
        self.assertEqual(set(by_row), set(expected))
        for row_index, relation in expected.items():
            self.assertEqual(
                (
                    by_row[row_index]["supervision"]["origin"],
                    by_row[row_index]["supervision"]["relation"],
                ),
                relation,
            )
        effective_path = next(self.runtime_root.rglob("effective_labels.json"))
        self.assertNotIn("supervision", effective_path.read_text(encoding="utf-8"))

    def test_bulk_annotations_are_atomic_and_session_scoped(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Bulk feedback")
        session_id = bootstrap["session"]["id"]
        status, value = client.request(
            "POST",
            f"/api/tuning/sessions/{session_id}/annotations/bulk",
            {
                "source": "selection-bulk-positive",
                "annotations": [
                    {"rowIndex": 0, "imageId": self.image_ids[0], "label": 1},
                    {"rowIndex": 2, "imageId": self.image_ids[2], "label": 1},
                ],
            },
        )
        self.assertEqual(status, 200, value)
        self.assertEqual([row["rowIndex"] for row in value["annotations"]], [0, 2])
        self.assertEqual(value["counts"]["usablePositive"], 2)

        # A single invalid member rejects the entire next batch. Row three must
        # not be written when the same request also contains Validation row five.
        status, value = client.request(
            "POST",
            f"/api/tuning/sessions/{session_id}/annotations/bulk",
            {
                "source": "selection-bulk-negative",
                "annotations": [
                    {"rowIndex": 3, "imageId": self.image_ids[3], "label": -1},
                    {"rowIndex": 5, "imageId": self.image_ids[5], "label": -1},
                ],
            },
        )
        self.assertEqual(status, 409, value)
        self.assertIn("Validation", value["error"])
        refreshed = self.bootstrap(client, "Bulk feedback")
        self.assertEqual(
            {row["rowIndex"] for row in refreshed["annotations"]},
            {0, 2},
        )
        with self.service.connect() as connection:
            events = connection.execute(
                "SELECT event,source FROM annotation_events WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        self.assertEqual(len(events), 2)
        self.assertTrue(all(str(row["event"]) == "upsert" for row in events))
        self.assertTrue(
            all(str(row["source"]) == "selection-bulk-positive" for row in events)
        )

    def test_frozen_test_rows_are_rejected_even_outside_test_ui_scope(self):
        data_root = self.web_root / "public" / "data" / "tasks" / TASK_ID
        (data_root / "validation-mask.u8").write_bytes(
            bytes([0, 0, 0, 0, 0, 1, 1, 1, 0])
        )
        (data_root / "test-mask.u8").write_bytes(
            bytes([0, 0, 0, 0, 0, 0, 0, 0, 1])
        )
        self.service.bundle.cache_clear()
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Test guard")
        session_id = bootstrap["session"]["id"]
        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/8",
            {"imageId": self.image_ids[8], "label": 1, "source": "top"},
        )
        self.assertEqual(status, 409)
        self.assertIn("Frozen Test", value["error"])

    def test_bootstrap_reuses_session_and_run_snapshot_is_immutable(self):
        public_hash_before = hashlib.sha256(
            (self.web_root / "public" / "data" / "tasks" / TASK_ID / "ground-truth.u8").read_bytes()
        ).hexdigest()
        client = ApiClient(self.base_url)
        first = self.bootstrap(client, "Alice")
        repeated = self.bootstrap(client, "Alice")
        self.assertEqual(first["session"]["id"], repeated["session"]["id"])
        session_id = first["session"]["id"]

        for row_index, label in ((0, 2), (1, 1), (2, -2), (3, -1)):
            status, value = client.request(
                "PUT",
                f"/api/tuning/sessions/{session_id}/annotations/{row_index}",
                {
                    "imageId": self.image_ids[row_index],
                    "label": label,
                    "source": "manual",
                },
            )
            self.assertEqual(status, 200, value)

        # Historical label runs remain executable after the public creation
        # API moves to the two low-dimensional tuning modes.
        run_id = TUNING.new_id("run")
        run_directory = self.runtime_root / "legacy-label" / run_id
        snapshot = {
            "schemaVersion": 2,
            "sessionId": session_id,
            "taskId": TASK_ID,
            "targetId": "joint",
            "baseMethod": "Ours-Full",
            "mode": "label",
            "evaluationScope": "validation",
            "createdAt": TUNING.utc_now(),
            "annotations": [
                {
                    "rowIndex": row_index,
                    "imageId": self.image_ids[row_index],
                    "label": label,
                    "source": "manual",
                    "updatedAt": TUNING.utc_now(),
                }
                for row_index, label in ((0, 2), (1, 1), (2, -2), (3, -1))
            ],
        }
        snapshot_bytes = (TUNING.json_dumps(snapshot) + "\n").encode("utf-8")
        run_directory.mkdir(parents=True)
        (run_directory / "labels_snapshot.json").write_bytes(snapshot_bytes)
        with self.service.connect() as connection:
            connection.execute(
                "INSERT INTO model_runs(id,session_id,user_id,mode,base_method,evaluation_scope,"
                "status,params_json,annotation_sha256,annotation_count,positive_count,negative_count,"
                "artifact_relpath,created_at) VALUES(?,?,?,?,?,'validation','queued',?,?,?,?,?,?,?)",
                (
                    run_id,
                    session_id,
                    first["user"]["id"],
                    "label",
                    "Ours-Full",
                    json.dumps({"regularizationC": 1.0}),
                    hashlib.sha256(snapshot_bytes).hexdigest(),
                    4,
                    2,
                    2,
                    run_directory.relative_to(self.runtime_root).as_posix(),
                    TUNING.utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM model_runs WHERE id=?", (run_id,)
            ).fetchone()
        run = self.service.run_json(row)
        self.assertTrue(run["legacyMode"])
        snapshot_before = (run_directory / "labels_snapshot.json").read_bytes()

        status, fresh_bootstrap = client.request(
            "POST",
            "/api/tuning/bootstrap",
            {
                "taskId": TASK_ID,
                "targetId": "joint",
                "baseMethod": "Ours-Full",
            },
        )
        self.assertEqual(status, 200, fresh_bootstrap)
        fresh_run = next(
            value for value in fresh_bootstrap["runs"] if value["id"] == run_id
        )
        self.assertFalse(fresh_run["stale"])

        embeddings = np.asarray(
            [
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.9, 0.1],
                [0.5, 0.5],
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.9, 0.1],
            ],
            dtype=np.float32,
        )
        self.service._load_embeddings = lambda task, ids: (
            embeddings,
            np.asarray([1.0, 0.0], dtype=np.float32),
            {"backbone": "toy", "dimension": 2, "queryCount": 1},
        )
        self.service._run_job(run["id"])
        completed = self.service.owned_run(run["id"], first["user"]["id"])
        completed_json = self.service.run_json(completed)
        self.assertEqual(completed_json["status"], "succeeded")
        self.assertEqual(completed_json["beforeMethod"], "Ours-Full")
        self.assertEqual(completed_json["evaluationScope"], "validation")
        expected_before = TUNING.average_precision(
            np.asarray([1, 1, 0, 0], dtype=np.uint8),
            np.asarray([0.10, 0.20, 0.90, 0.80], dtype=np.float32),
        )
        self.assertAlmostEqual(completed_json["before"]["ap"], expected_before)
        self.assertGreater(completed_json["after"]["ap"], expected_before)
        self.assertEqual(completed_json["before"]["evaluationCount"], 4)
        metrics_path = next(self.runtime_root.rglob("metrics.json"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["evaluationScope"], "validation")
        self.assertEqual(metrics["groundTruthUsage"], "Validation model-selection evaluation only")

        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/0",
            {"imageId": self.image_ids[0], "label": -1, "source": "manual"},
        )
        self.assertEqual(status, 200, value)
        # A completed run remains stale across a full bootstrap after the
        # effective label state changes; this cannot rely on client memory.
        status, stale_bootstrap = client.request(
            "POST",
            "/api/tuning/bootstrap",
            {
                "taskId": TASK_ID,
                "targetId": "joint",
                "baseMethod": "Ours-Full",
            },
        )
        self.assertEqual(status, 200)
        stale_run = next(value for value in stale_bootstrap["runs"] if value["id"] == run_id)
        self.assertTrue(stale_run["stale"])
        self.assertEqual(
            (run_directory / "labels_snapshot.json").read_bytes(), snapshot_before
        )

        self.assertIsNone(self.service._claim_run(run["id"]))
        public_hash_after = hashlib.sha256(
            (self.web_root / "public" / "data" / "tasks" / TASK_ID / "ground-truth.u8").read_bytes()
        ).hexdigest()
        self.assertEqual(public_hash_before, public_hash_after)

    def test_new_run_modes_freeze_v4_contract_and_legacy_modes_cannot_be_created(self):
        original_task = self.service.tasks[TASK_ID]
        retrieval_targets = [
            {"id": "first", "label": "First", "kind": "attribute"},
            {"id": "second", "label": "Second", "kind": "attribute"},
            {
                "id": "joint",
                "label": "Joint",
                "kind": "derived",
                "members": ["first", "second"],
            },
        ]
        self.service.tasks[TASK_ID] = TUNING.TaskSpec(
            task_id=original_task.task_id,
            dataset_id=original_task.dataset_id,
            task_name=original_task.task_name,
            data_root=original_task.data_root,
            row_count=original_task.row_count,
            methods=original_task.methods,
            target_ids=("first", "second", "joint"),
            manifest={
                **original_task.manifest,
                "targetCount": 3,
                "retrievalTargets": retrieval_targets,
            },
        )
        self.service.bundle.cache_clear()
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Low-dimensional modes")
        session_id = bootstrap["session"]["id"]
        for row_index, label in ((0, 2), (1, 1), (2, -2), (3, -1)):
            status, value = client.request(
                "PUT",
                f"/api/tuning/sessions/{session_id}/annotations/{row_index}",
                {"imageId": self.image_ids[row_index], "label": label, "source": "manual"},
            )
            self.assertEqual(status, 200, value)

        for legacy_mode in ("prototype", "label"):
            status, value = client.request(
                "POST",
                f"/api/tuning/sessions/{session_id}/runs",
                {"mode": legacy_mode, "baseMethod": "Ours-Full"},
            )
            self.assertEqual(status, 409, value)
            self.assertIn("read-only", value["error"])

        verified_fusion = None
        for mode, expected_method in (
            ("fusion-weight", "Ours-Full"),
            ("residual", "Image Prototype"),
        ):
            status, value = self.historical_run_fixture(
                bootstrap["user"]["id"], session_id,
                {
                    "mode": mode,
                    "baseMethod": "Image Prototype",
                    "regularizationLambda": 0.75,
                    "biasRegularizationEta": 0.2,
                    "feedbackWeight": 12.0,
                    "maxIterations": 750,
                },
            )
            self.assertEqual(status, 202, value)
            run = value["run"]
            self.assertEqual(run["mode"], mode)
            self.assertEqual(run["baseMethod"], expected_method)
            self.assertFalse(run["legacyMode"])
            self.assertEqual(run["evaluationScope"], "vqa-validation")
            self.assertEqual(run["supervisionPolicy"], TUNING.VQA_VALIDATION_SUPERVISION_POLICY)
            snapshot_path = next(
                path
                for path in self.runtime_root.rglob("labels_snapshot.json")
                if path.parent.name == run["id"]
            )
            snapshot_bytes = snapshot_path.read_bytes()
            snapshot = json.loads(snapshot_bytes.decode("utf-8"))
            self.assertEqual(snapshot["schemaVersion"], 3)
            self.assertEqual(snapshot["mode"], mode)
            self.assertEqual(snapshot["baseMethod"], expected_method)
            self.assertEqual(snapshot["algorithmVersion"], run["algorithmVersion"])
            self.assertEqual(snapshot["supervisionPolicy"], TUNING.VQA_VALIDATION_SUPERVISION_POLICY)
            params = snapshot["params"]
            self.assertEqual(params["regularizationLambda"], 0.75)
            self.assertEqual(params["biasRegularizationEta"], 0.2)
            self.assertEqual(params["feedbackWeight"], 12.0)
            self.assertEqual(params["maxIterations"], 750)
            self.assertEqual(
                params["vqaValidationProtocol"],
                TUNING.VQA_VALIDATION_PROTOCOL,
            )
            self.assertEqual(
                params["vqaValidationFingerprint"],
                "default-toy-clean-validation",
            )
            self.assertEqual(params["vqaValidationCount"], 3)
            self.assertEqual(params["vqaValidationPositiveCount"], 1)
            self.assertEqual(params["vqaValidationNegativeCount"], 2)
            self.assertEqual(params["probeFitProtocol"], "original-vqa-minus-shared-validation-v1")
            self.assertEqual(params["probeFitSeed"], 0)
            self.assertEqual(params["probeFitFraction"], 0.8)
            self.assertEqual(
                params["probeFitFingerprint"],
                "default-toy-clean-validation-fit",
            )
            self.assertEqual(
                params["probeSourceFingerprint"],
                "default-toy-clean-validation-source",
            )
            self.assertEqual(
                params["vqaValidationManifestSha256"],
                "a" * 64,
            )
            self.assertEqual(
                params["vqaValidationVersion"],
                "toy-vqa-v1",
            )
            if mode == "fusion-weight":
                self.assertEqual(
                    params["algorithmVersion"],
                    TUNING.FUSION_WEIGHT_ALGORITHM_V4,
                )
                self.assertEqual(
                    params["parameterization"],
                    "per-attribute-13-simplex-plus-attribute-simplex",
                )
                self.assertEqual(params["attributeIds"], ["first", "second"])
                self.assertEqual(params["activeAttributeIds"], ["first", "second"])
                self.assertEqual(
                    params["activeJointAttributeIds"], ["first", "second"]
                )
                self.assertEqual(
                    params["methodIds"],
                    list(TUNING.RANK_FUSION_METHODS),
                )
                self.assertEqual(
                    set(params["initialMethodWeightsByAttribute"]),
                    {"first", "second"},
                )
                for weights in params["initialMethodWeightsByAttribute"].values():
                    self.assertEqual(
                        set(weights), set(TUNING.RANK_FUSION_METHODS)
                    )
                    self.assertAlmostEqual(sum(weights.values()), 1.0)
                self.assertEqual(params["sharedRegularizationLambda"], 0.25)
                self.assertEqual(
                    params["initialAttributeWeights"],
                    {"first": 0.5, "second": 0.5},
                )
                self.assertEqual(params["jointAggregation"], "weighted-mean")
                self.assertEqual(
                    params["jointWeightConstraint"], "nonnegative-sum-one"
                )
                self.assertEqual(params["jointRegularizationLambda"], 0.5)
            else:
                self.assertEqual(params["featureMethods"], list(TUNING.WEIGHTED_FUSION_LEARNERS))

            with self.service.connect() as connection:
                run_row = connection.execute(
                    "SELECT r.*,s.task_id,s.target_id FROM model_runs r "
                    "JOIN sessions s ON s.id=r.session_id WHERE r.id=?",
                    (run["id"],),
                ).fetchone()
            expected_snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
            expected_annotation_sha256 = self.service.annotation_state_sha256(
                snapshot["annotations"]
            )
            self.assertEqual(
                str(run_row["snapshot_sha256"]), expected_snapshot_sha256
            )
            self.assertEqual(
                str(run_row["annotation_sha256"]), expected_annotation_sha256
            )
            self.assertNotEqual(
                str(run_row["snapshot_sha256"]),
                str(run_row["annotation_sha256"]),
            )
            verified_snapshot, verified_sha256 = (
                self.service._read_verified_run_snapshot(
                    run_row, snapshot_path.parent
                )
            )
            self.assertEqual(verified_snapshot, snapshot)
            self.assertEqual(verified_sha256, expected_snapshot_sha256)
            if mode == "fusion-weight":
                verified_fusion = (run_row, snapshot_path, snapshot_bytes, snapshot)
            # These are historical snapshot fixtures, not concurrently queued jobs.
            with self.service.connect() as connection:
                connection.execute("UPDATE model_runs SET status='cancelled' WHERE id=?", (run["id"],))

        self.assertIsNotNone(verified_fusion)
        fusion_row, fusion_path, original_bytes, original_snapshot = verified_fusion
        tampered_snapshot = dict(original_snapshot)
        tampered_snapshot["params"] = {
            **original_snapshot["params"],
            "feedbackWeight": 99.0,
        }
        fusion_path.write_bytes(
            (TUNING.json_dumps(tampered_snapshot) + "\n").encode("utf-8")
        )
        with self.assertRaisesRegex(RuntimeError, "snapshot file checksum mismatch"):
            self.service._read_verified_run_snapshot(
                fusion_row, fusion_path.parent
            )
        fusion_path.write_bytes(original_bytes)

        status, refreshed = client.request(
            "POST",
            "/api/tuning/bootstrap",
            {
                "taskId": TASK_ID,
                "targetId": "joint",
                "baseMethod": "Ours-Full",
            },
        )
        self.assertEqual(status, 200, refreshed)
        self.assertTrue(refreshed["runs"])
        self.assertTrue(all(not run["stale"] for run in refreshed["runs"]))

    def test_new_run_modes_accept_one_sided_feedback_and_reject_no_usable_feedback(self):
        for mode, label in (("fusion-weight", 2), ("residual", -2)):
            client = ApiClient(self.base_url)
            bootstrap = self.bootstrap(client, f"One-sided {mode}")
            session_id = bootstrap["session"]["id"]
            status, value = client.request(
                "PUT",
                f"/api/tuning/sessions/{session_id}/annotations/0",
                {"imageId": self.image_ids[0], "label": label, "source": "manual"},
            )
            self.assertEqual(status, 200, value)
            status, value = self.historical_run_fixture(
                bootstrap["user"]["id"], session_id,
                {"mode": mode, "baseMethod": "Ours-Full"},
            )
            self.assertEqual(status, 202, value)
            self.assertEqual(value["run"]["annotationCount"], 1)
            self.assertEqual(value["run"]["positiveCount"], int(label > 0))
            self.assertEqual(value["run"]["negativeCount"], int(label < 0))

        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "No usable feedback")
        session_id = bootstrap["session"]["id"]
        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/0",
            {"imageId": self.image_ids[0], "label": 0, "source": "manual"},
        )
        self.assertEqual(status, 200, value)
        for mode in TUNING.CREATABLE_RUN_MODES:
            status, value = self.historical_run_fixture(
                bootstrap["user"]["id"], session_id,
                {"mode": mode, "baseMethod": "Ours-Full"},
            )
            self.assertEqual(status, 409, value)
            self.assertIn("at least one positive or negative", value["error"])

    def test_run_creation_rejects_feedback_entirely_inside_clean_validation(self):
        split = clean_validation_split(
            fit_indices=[1, 2],
            fit_labels=[1, 0],
            validation_indices=[0],
            validation_labels=[0],
            fingerprint="all-feedback-held-out",
        )
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Held-out feedback")
        session_id = bootstrap["session"]["id"]
        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/0",
            {"imageId": self.image_ids[0], "label": 2, "source": "manual"},
        )
        self.assertEqual(status, 200, value)
        # This is historical feedback that predates freezing the new holdout.
        self.service._clean_validation_split = lambda task_id, target_id: split
        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/1",
            {"imageId": self.image_ids[1], "label": 0, "source": "manual"},
        )
        self.assertEqual(status, 200, value)
        status, value = self.historical_run_fixture(
            bootstrap["user"]["id"], session_id,
            {"mode": "fusion-weight", "baseMethod": "Ours-Full"},
        )
        self.assertEqual(status, 409, value)
        self.assertIn("fixed VQA Validation", value["error"])

    def test_vqa_validation_endpoint_returns_membership_without_labels_or_gt(self):
        self.service._task_ground_truth = lambda task_id: self.fail("GT was read")
        client = ApiClient(self.base_url)
        status, value = client.request("GET", f"/api/tuning/tasks/{TASK_ID}/validation")
        self.assertEqual(status, 200, value)
        self.assertEqual(value["taskId"], TASK_ID)
        self.assertEqual(value["rowIndices"], [6, 7, 8])
        self.assertEqual(value["count"], 3)
        self.assertEqual(value["version"], "toy-vqa-v1")
        self.assertEqual(value["manifestSha256"], "a" * 64)
        self.assertEqual(value["labelSource"], "original-vqa-supervision")
        self.assertFalse(value["initialModelHoldoutIndependent"])
        self.assertTrue(value["referenceOnly"])
        self.assertNotIn("labels", value)

    def test_vqa_holdout_rejects_new_labels_and_bulk_atomically(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "VQA gate")
        session_id = bootstrap["session"]["id"]
        heldout_split = vqa_contract_fixture(clean_validation_split(
            fit_indices=[1, 2, 3, 5], fit_labels=[0, 0, 1, 1],
            validation_indices=[0, 6], validation_labels=[0, 1], fingerprint="small-val",
        ))
        calls = []
        def load_split(task_id, target_id, version=None):
            calls.append((task_id, target_id, version))
            return heldout_split
        self.service._vqa_validation_split = load_split
        status, value = client.request(
            "PUT", f"/api/tuning/sessions/{session_id}/annotations/0",
            {"imageId": self.image_ids[0], "label": 1, "source": "manual"},
        )
        self.assertEqual(status, 409, value)
        self.assertIn("Fixed VQA Validation", value["error"])
        calls.clear()
        status, value = client.request(
            "POST", f"/api/tuning/sessions/{session_id}/annotations/bulk",
            {"annotations": [
                {"rowIndex": row, "imageId": self.image_ids[row], "label": 1}
                for row in [1, 0]
            ]},
        )
        self.assertEqual(status, 409, value)
        self.assertEqual(len(calls), 1, "Bulk membership must load only once")
        with self.service.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM annotations").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM annotation_events").fetchone()[0], 0)

    def test_historical_vqa_holdout_stays_reviewable_but_never_trains_or_calibrates(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Historical VQA feedback")
        session_id = bootstrap["session"]["id"]
        for row, label in [(0, -1), (3, 1)]:
            status, value = client.request(
                "PUT", f"/api/tuning/sessions/{session_id}/annotations/{row}",
                {"imageId": self.image_ids[row], "label": label, "source": "manual"},
            )
            self.assertEqual(status, 200, value)
        split = vqa_contract_fixture(clean_validation_split(
            fit_indices=[1, 2, 3, 5], fit_labels=[0, 0, 1, 1],
            validation_indices=[0, 6], validation_labels=[0, 1], fingerprint="historical-val",
        ))
        self.service._vqa_validation_split = lambda task, target, version=None: split
        base = SimpleNamespace(
            attribute_ids=("first", "second"), attribute_names=("First", "Second"),
            learner_names=TUNING.WEIGHTED_FUSION_LEARNERS,
            embedding_names=TUNING.REFINEMENT_EMBEDDING_METHODS,
            initial_theta=np.asarray([0.5, 0.5], dtype=np.float32),
            temperature=np.asarray([0.2, 0.2], dtype=np.float32),
            probe_features=np.repeat(np.arange(9, dtype=np.float32)[:, None, None], 16, axis=2).reshape(9, 2, 8),
            base_state_fingerprint="historical-feedback-base",
            normalization_audit=self.service._refinement_normalization_scope(TASK_ID)[1],
        )
        self.service._refinement_base_context = lambda task_id, *args, **kwargs: base
        observed = self.bootstrap(client, "Historical VQA feedback")
        self.assertEqual(len(observed["annotations"]), 2)
        self.assertFalse(observed["annotations"][0]["supervision"]["includedInTune"])
        self.assertEqual(observed["annotations"][0]["supervision"]["excludedReason"], "fixed-vqa-validation")
        self.assertTrue(observed["annotations"][1]["supervision"]["includedInTune"])
        status, value = client.request(
            "POST", f"/api/tuning/sessions/{session_id}/runs",
            {"mode": "weight_staged", "maxIterations": 10},
        )
        self.assertEqual(status, 202, value)
        self.assertEqual(value["run"]["evaluationScope"], "vqa-validation")
        prepared = self.service._prepare_weight_refinement_supervision(
            TASK_ID, base, observed["annotations"], feedback_weight=8,
            vqa_validation=True, vqa_validation_version="toy-vqa-v1",
        )
        self.assertEqual(prepared["audit"]["feedbackHoldoutExcludedCount"], 1)
        for examples in [prepared["attributeExamples"], prepared["calibrationExamples"]]:
            for example in examples:
                self.assertFalse(set(example["rows"]) & {0, 6})
        self.assertFalse(set(prepared["jointRows"]) & {0, 6})
        calibration_scores, _, _ = self.service._refinement_calibration_arrays(
            TASK_ID, base, vqa_validation_version="toy-vqa-v1",
        )
        self.assertEqual(calibration_scores[:, 0, 0].tolist(), [1, 2, 3, 5])
        status, value = client.request(
            "PUT", f"/api/tuning/sessions/{session_id}/annotations/0",
            {"imageId": self.image_ids[0], "label": 2, "source": "annotation-review"},
        )
        self.assertEqual(status, 409, value)
        status, value = client.request("DELETE", f"/api/tuning/sessions/{session_id}/annotations/0")
        self.assertEqual(status, 200, value)
        self.assertEqual(split.validation_indices.tolist(), [0, 6])

    def test_queued_vqa_run_rejects_manifest_drift(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Pinned VQA version")
        session_id = bootstrap["session"]["id"]
        client.request("PUT", f"/api/tuning/sessions/{session_id}/annotations/3",
                       {"imageId": self.image_ids[3], "label": 1})
        split = vqa_contract_fixture(clean_validation_split(
            fit_indices=[0, 1, 2, 3], fit_labels=[0, 0, 1, 1],
            validation_indices=[5, 6], validation_labels=[1, 0], fingerprint="pinned-val",
        ))
        versions = []
        def load_split(task_id, target_id, version=None):
            versions.append(version)
            return split
        self.service._vqa_validation_split = load_split
        run = self.service.create_run(bootstrap["user"]["id"], session_id, {"mode": "residual"})
        with self.service.connect() as connection:
            stored = connection.execute(
                "SELECT r.*,s.task_id,s.target_id FROM model_runs r JOIN sessions s ON s.id=r.session_id WHERE r.id=?",
                (run["id"],),
            ).fetchone()
        split.audit["frozenManifestSha256"] = "c" * 64
        self.service._fusion_context_for_algorithm = lambda task, algorithm: None
        with self.assertRaisesRegex(RuntimeError, "version/checksum changed"):
            self.service._execute_model_run(stored)
        self.assertEqual(versions[-1], "toy-vqa-v1")

    def test_legacy_clean_and_probe_workers_do_not_use_new_vqa_membership(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Legacy worker provenance")
        session_id = bootstrap["session"]["id"]
        status, value = client.request(
            "PUT", f"/api/tuning/sessions/{session_id}/annotations/0",
            {"imageId": self.image_ids[0], "label": 1},
        )
        self.assertEqual(status, 200, value)
        self.service._fusion_context_for_algorithm = lambda task, algorithm: None
        for policy, scope, clean in [
            (TUNING.SUPERVISION_POLICY, "clean-validation", True),
            (TUNING.PROBE_VALIDATION_SUPERVISION_POLICY, "probe-validation", False),
        ]:
            created = self.service.create_run(bootstrap["user"]["id"], session_id, {"mode": "residual"})
            with self.service.connect() as connection:
                stored = dict(connection.execute(
                    "SELECT r.*,s.task_id,s.target_id FROM model_runs r JOIN sessions s ON s.id=r.session_id WHERE r.id=?",
                    (created["id"],),
                ).fetchone())
            directory = self.runtime_root / stored["artifact_relpath"]
            snapshot = json.loads((directory / "labels_snapshot.json").read_text(encoding="utf-8"))
            snapshot["supervisionPolicy"] = policy
            snapshot["evaluationScope"] = scope
            snapshot["params"]["supervisionPolicy"] = policy
            TUNING.atomic_write_json(directory / "labels_snapshot.json", snapshot)
            stored["params_json"] = TUNING.json_dumps(snapshot["params"])
            stored["snapshot_sha256"] = hashlib.sha256((directory / "labels_snapshot.json").read_bytes()).hexdigest()
            stored["evaluation_scope"] = scope
            current_loader = self.service._vqa_validation_split
            self.service._vqa_validation_split = lambda *args, **kwargs: self.fail("Legacy worker consulted active VQA holdout")
            old_prepare = self.service._prepare_probe_tuning_supervision
            def observe_prepare(*args, **kwargs):
                self.assertEqual(kwargs["clean_validation"], clean)
                self.assertFalse(kwargs["vqa_validation"])
                raise RuntimeError("legacy-route-verified")
            self.service._prepare_probe_tuning_supervision = observe_prepare
            try:
                with self.assertRaisesRegex(RuntimeError, "legacy-route-verified"):
                    self.service._execute_model_run(stored)
            finally:
                self.service._vqa_validation_split = current_loader
                self.service._prepare_probe_tuning_supervision = old_prepare
                with self.service.connect() as connection:
                    connection.execute("UPDATE model_runs SET status='cancelled' WHERE id=?", (created["id"],))

    def test_single_attribute_fusion_snapshot_keeps_joint_weights_inactive(self):
        original_task = self.service.tasks[TASK_ID]
        retrieval_targets = [
            {"id": "first", "label": "First", "kind": "attribute"},
            {"id": "second", "label": "Second", "kind": "attribute"},
            {
                "id": "joint",
                "label": "Joint",
                "kind": "derived",
                "members": ["first", "second"],
            },
        ]
        self.service.tasks[TASK_ID] = TUNING.TaskSpec(
            task_id=original_task.task_id,
            dataset_id=original_task.dataset_id,
            task_name=original_task.task_name,
            data_root=original_task.data_root,
            row_count=original_task.row_count,
            methods=original_task.methods,
            target_ids=("first", "second", "joint"),
            manifest={
                **original_task.manifest,
                "targetCount": 3,
                "retrievalTargets": retrieval_targets,
            },
        )
        self.service.bundle.cache_clear()
        client = ApiClient(self.base_url)
        status, bootstrap = client.request(
            "POST",
            "/api/tuning/bootstrap",
            {
                "taskId": TASK_ID,
                "targetId": "first",
                "baseMethod": "Ours-Full",
                "displayName": "Single attribute v4",
            },
        )
        self.assertEqual(status, 200, bootstrap)
        session_id = bootstrap["session"]["id"]
        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/0",
            {"imageId": self.image_ids[0], "label": 2, "source": "manual"},
        )
        self.assertEqual(status, 200, value)
        status, value = self.historical_run_fixture(
            bootstrap["user"]["id"], session_id,
            {"mode": "fusion-weight", "baseMethod": "Ours-Full"},
        )
        self.assertEqual(status, 202, value)
        run_id = value["run"]["id"]
        snapshot_path = next(
            path
            for path in self.runtime_root.rglob("labels_snapshot.json")
            if path.parent.name == run_id
        )
        params = json.loads(snapshot_path.read_text(encoding="utf-8"))["params"]
        self.assertEqual(
            params["algorithmVersion"], TUNING.FUSION_WEIGHT_ALGORITHM_V4
        )
        self.assertEqual(params["activeAttributeIds"], ["first"])
        self.assertEqual(params["activeJointAttributeIds"], [])
        self.assertEqual(
            params["initialAttributeWeights"], {"first": 0.5, "second": 0.5}
        )
        self.assertEqual(
            TUNING.SUPPORTED_FUSION_WEIGHT_ALGORITHMS,
            {
                TUNING.FUSION_WEIGHT_ALGORITHM_V1,
                TUNING.FUSION_WEIGHT_ALGORITHM_V2,
                TUNING.FUSION_WEIGHT_ALGORITHM_V3,
                TUNING.FUSION_WEIGHT_ALGORITHM_V4,
            },
        )

    def test_v3_execution_persists_and_deploys_joint_attribute_weights(self):
        original_task = self.service.tasks[TASK_ID]
        retrieval_targets = [
            {"id": "first", "label": "First", "kind": "attribute"},
            {"id": "second", "label": "Second", "kind": "attribute"},
            {
                "id": "joint",
                "label": "Joint",
                "kind": "derived",
                "members": ["first", "second"],
            },
        ]
        data_root = original_task.data_root
        for filename in ("raw-scores.f32", "ranks.f32"):
            source = np.fromfile(data_root / filename, dtype="<f4").reshape(9, 2, 1)
            np.repeat(source, 3, axis=2).astype("<f4").tofile(data_root / filename)
        truth = np.fromfile(data_root / "ground-truth.u8", dtype=np.uint8).reshape(9, 1)
        np.repeat(truth, 3, axis=1).astype(np.uint8).tofile(
            data_root / "ground-truth.u8"
        )
        self.service.tasks[TASK_ID] = TUNING.TaskSpec(
            task_id=original_task.task_id,
            dataset_id=original_task.dataset_id,
            task_name=original_task.task_name,
            data_root=original_task.data_root,
            row_count=original_task.row_count,
            methods=original_task.methods,
            target_ids=("first", "second", "joint"),
            manifest={
                **original_task.manifest,
                "targetCount": 3,
                "retrievalTargets": retrieval_targets,
            },
        )
        self.service.bundle.cache_clear()

        class FakePcp:
            @staticmethod
            def reconstruct_ours_full_scores(
                _tensors,
                _seeds_by_method,
                _methods,
                _attributes,
                target_attribute,
                _selected_seeds,
                indices,
                _task_config,
                _ours_full_root,
                **kwargs,
            ):
                rows = np.asarray(indices, dtype=np.int64)
                first_gate = np.where(np.isin(rows, (1, 3)), 0.95, 0.08)
                second_gate = np.full(rows.size, 0.5, dtype=np.float64)
                if target_attribute == "first":
                    scores = first_gate
                elif target_attribute == "second":
                    scores = second_gate
                else:
                    alpha = kwargs["attribute_weights"]
                    scores = np.power(first_gate, alpha["first"]) * np.power(
                        second_gate, alpha["second"]
                    )
                return scores.reshape(1, -1)

        context = TUNING.WeightedFusionContext(
            pcp=FakePcp(),
            adapter=None,
            task_config={},
            tensors={},
            seeds_by_method={},
            learned_method_ids=TUNING.WEIGHTED_FUSION_LEARNERS,
            learned_method_labels=TUNING.WEIGHTED_FUSION_LEARNERS,
            attributes=("first", "second"),
            target_attributes=("first", "second", None),
            selected_seeds=(0,),
            gallery_indices=np.arange(9),
            train_indices=np.arange(4),
            ours_full_root=Path("unused"),
        )
        self.service._weighted_fusion_context = lambda _task_id: context
        deployed: dict[str, tuple] = {}

        def fake_hierarchical_fusion(
            task_id,
            attribute_weights,
            learner_weights,
            global_weights,
        ):
            deployed["attributeWeights"] = tuple(attribute_weights)
            deployed["learnerWeights"] = tuple(learner_weights)
            deployed["globalWeights"] = tuple(global_weights)
            raw = np.tile(np.linspace(0.1, 0.9, 9)[:, None], (1, 3)).astype(
                np.float32
            )
            ranks = np.tile(TUNING.normalized_ranks(raw[:, 0])[:, None], (1, 3))
            packed = np.concatenate(
                [
                    raw.reshape(-1),
                    raw.reshape(-1),
                    ranks.reshape(-1),
                    raw.reshape(-1),
                    ranks.reshape(-1),
                ]
            ).astype("<f4")
            return TUNING.HierarchicalFusionResult(
                payload=packed.tobytes(),
                row_count=9,
                target_count=3,
                baseline=False,
                baseline_max_abs_error=None,
                normalized_attribute_weights=tuple(attribute_weights),
                normalized_learner_weights=tuple(learner_weights),
                normalized_global_weights=tuple(global_weights),
            )

        self.service._compute_hierarchical_fusion = fake_hierarchical_fusion
        split = clean_validation_split(
            fit_indices=[0, 1],
            fit_labels=[0, 1],
            validation_indices=[6, 7],
            validation_labels=[1, 0],
            fingerprint="toy-clean-validation",
        )
        self.service._clean_validation_split = lambda task_id, target_id: split
        client = ApiClient(self.base_url)
        status, bootstrap = client.request(
            "POST",
            "/api/tuning/bootstrap",
            {
                "taskId": TASK_ID,
                "targetId": "joint",
                "baseMethod": "Ours-Full",
                "displayName": "Execute v3",
            },
        )
        self.assertEqual(status, 200, bootstrap)
        session_id = bootstrap["session"]["id"]
        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/2",
            {"imageId": self.image_ids[2], "label": -2, "source": "manual"},
        )
        self.assertEqual(status, 200, value)
        status, value = client.request(
            "PUT",
            f"/api/tuning/sessions/{session_id}/annotations/3",
            {"imageId": self.image_ids[3], "label": 2, "source": "manual"},
        )
        self.assertEqual(status, 200, value)
        current_version = TUNING.RUN_ALGORITHM_VERSIONS["fusion-weight"]
        TUNING.RUN_ALGORITHM_VERSIONS["fusion-weight"] = (
            TUNING.FUSION_WEIGHT_ALGORITHM_V3
        )
        try:
            status, value = self.historical_run_fixture(
                bootstrap["user"]["id"], session_id,
                {
                    "mode": "fusion-weight",
                    "baseMethod": "Ours-Full",
                    "jointRegularizationLambda": 0.01,
                },
            )
        finally:
            TUNING.RUN_ALGORITHM_VERSIONS["fusion-weight"] = current_version
        self.assertEqual(status, 202, value)
        run_id = value["run"]["id"]
        # The frozen Clean Validation contract is reused by the worker and the
        # resulting two-row public-GT audit remains stable across execution.
        scripts_path = str(WEB_ROOT / "scripts")
        sys.path.insert(0, scripts_path)
        try:
            self.service._run_job(run_id)
        finally:
            sys.path.remove(scripts_path)
        completed = self.service.owned_run(run_id, bootstrap["user"]["id"])
        completed_json = self.service.run_json(completed)
        self.assertEqual(completed_json["status"], "succeeded")
        self.assertEqual(completed_json["evaluationScope"], "vqa-validation")
        self.assertEqual(completed_json["before"]["evaluationCount"], 2)
        self.assertEqual(completed_json["before"]["positiveCount"], 1)
        self.assertEqual(
            completed_json["splitAudit"]["validationFingerprint"],
            "toy-clean-validation",
        )
        summary = completed_json["after"]["modelSummary"]
        self.assertEqual(summary["jointAggregation"], "product")
        self.assertEqual(summary["activeJointAttributeIds"], ["first", "second"])
        self.assertEqual(summary["optimizedAttributeWeightCount"], 2)
        self.assertAlmostEqual(
            sum(summary["attributeWeights"].values()) / 2.0, 1.0, places=6
        )
        np.testing.assert_allclose(
            deployed["attributeWeights"],
            [summary["attributeWeights"]["first"], summary["attributeWeights"]["second"]],
            atol=1e-7,
        )
        run_directory = self.runtime_root / str(completed["artifact_relpath"])
        with np.load(run_directory / "model.npz", allow_pickle=False) as artifact:
            self.assertIn("attributeWeights", artifact.files)
            self.assertIn("initialAttributeWeights", artifact.files)
            self.assertIn("activeJointAttributeIds", artifact.files)
            self.assertEqual(str(artifact["jointAggregation"][0]), "product")

    def test_legacy_run_without_scope_defaults_to_frozen_test_provenance(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Legacy")
        session_id = bootstrap["session"]["id"]
        user_id = bootstrap["user"]["id"]
        now = TUNING.utc_now()
        with self.service.connect() as connection:
            connection.execute(
                "INSERT INTO model_runs(id,session_id,user_id,mode,base_method,"
                "evaluation_scope,status,params_json,annotation_sha256,annotation_count,"
                "positive_count,negative_count,artifact_relpath,before_json,after_json,created_at) "
                "VALUES(?,?,?,?,?,'test','succeeded','{}','legacy',0,0,0,?,?,?,?)",
                (
                    "run_legacy",
                    session_id,
                    user_id,
                    "prototype",
                    "Image Prototype",
                    "legacy/run",
                    json.dumps({"ap": 0.25, "testCount": 2}),
                    json.dumps({"ap": 0.5, "testCount": 2}),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM model_runs WHERE id='run_legacy'"
            ).fetchone()
        value = self.service.run_json(row)
        self.assertEqual(value["evaluationScope"], "test")
        self.assertEqual(value["before"]["testCount"], 2)
        self.assertAlmostEqual(value["deltaAp"], 0.25)

    def test_database_migration_preserves_legacy_scopes_and_adds_probe_validation(self):
        legacy_runtime = self.root / "legacy-runtime"
        legacy_runtime.mkdir(parents=True)
        database = legacy_runtime / "workbench.sqlite3"
        import sqlite3

        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE users (
              id TEXT PRIMARY KEY, identity_key TEXT, identity_source TEXT,
              display_name TEXT, created_at TEXT, last_seen_at TEXT
            );
            CREATE TABLE sessions (
              id TEXT PRIMARY KEY, user_id TEXT, task_id TEXT, target_id TEXT,
              base_method TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE model_runs (
              id TEXT PRIMARY KEY, session_id TEXT, user_id TEXT, mode TEXT,
              base_method TEXT, status TEXT, params_json TEXT,
              annotation_sha256 TEXT, annotation_count INTEGER,
              positive_count INTEGER, negative_count INTEGER,
              artifact_relpath TEXT, before_json TEXT, after_json TEXT,
              error TEXT, created_at TEXT, started_at TEXT, finished_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO users VALUES(?,?,?,?,?,?)",
            (
                "user_legacy",
                "identity_legacy",
                "cookie",
                "Legacy",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )
        values = (
            "session_legacy", "user_legacy", TASK_ID, "joint", "Ours-Full",
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
        )
        connection.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?)", values)
        run_prefix = (
            "session_legacy", "user_legacy", "prototype", "Image Prototype",
        )
        run_suffix = (
            "{}", "hash", 0, 0, 0, "legacy/run", None, None, None,
            "2026-01-01T00:00:00Z", None, None,
        )
        connection.execute(
            "INSERT INTO model_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("run_succeeded", *run_prefix, "succeeded", *run_suffix),
        )
        connection.execute(
            "INSERT INTO model_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("run_queued", *run_prefix, "queued", *run_suffix),
        )
        connection.commit()
        connection.close()

        migrated = TUNING.TuningService(
            self.web_root, legacy_runtime, start_worker=False
        )
        with migrated.connect() as current:
            scopes = dict(
                current.execute(
                    "SELECT id,evaluation_scope FROM model_runs ORDER BY id"
                ).fetchall()
            )
        self.assertEqual(scopes["run_succeeded"], "test")
        self.assertEqual(scopes["run_queued"], "test")

    def test_database_mode_check_migration_is_lossless_and_idempotent(self):
        legacy_runtime = self.root / "legacy-mode-runtime"
        legacy_runtime.mkdir(parents=True)
        database = legacy_runtime / "workbench.sqlite3"
        import sqlite3

        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE users (
              id TEXT PRIMARY KEY, identity_key TEXT UNIQUE, identity_source TEXT,
              display_name TEXT, created_at TEXT, last_seen_at TEXT
            );
            CREATE TABLE sessions (
              id TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
              task_id TEXT, target_id TEXT, base_method TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE model_runs (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
              user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              mode TEXT NOT NULL CHECK (mode IN ('prototype','label')),
              base_method TEXT NOT NULL,
              evaluation_scope TEXT NOT NULL DEFAULT 'validation'
                CHECK (evaluation_scope IN ('validation','test')),
              status TEXT NOT NULL CHECK (
                status IN ('queued','running','succeeded','failed','cancelled')
              ),
              params_json TEXT NOT NULL,
              annotation_sha256 TEXT NOT NULL,
              annotation_count INTEGER NOT NULL,
              positive_count INTEGER NOT NULL,
              negative_count INTEGER NOT NULL,
              artifact_relpath TEXT NOT NULL,
              before_json TEXT, after_json TEXT, error TEXT,
              created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
            );
            CREATE INDEX idx_model_runs_session_created
              ON model_runs(session_id, created_at DESC);
            CREATE INDEX idx_model_runs_queued ON model_runs(created_at)
              WHERE status='queued';
            """
        )
        now = "2026-01-01T00:00:00Z"
        connection.execute(
            "INSERT INTO users VALUES(?,?,?,?,?,?)",
            ("user", "identity", "cookie", "Legacy", now, now),
        )
        connection.execute(
            "INSERT INTO sessions VALUES(?,?,?,?,?,?,?)",
            ("session", "user", TASK_ID, "joint", "Ours-Full", now, now),
        )
        connection.execute(
            "INSERT INTO model_runs(id,session_id,user_id,mode,base_method,evaluation_scope,"
            "status,params_json,annotation_sha256,annotation_count,positive_count,negative_count,"
            "artifact_relpath,before_json,after_json,error,created_at,started_at,finished_at) "
            "VALUES(?,?,?,?,?,'test','succeeded','{}','hash',4,2,2,'legacy/run',?,?,NULL,?,?,?)",
            (
                "legacy-run",
                "session",
                "user",
                "prototype",
                "Image Prototype",
                json.dumps({"ap": 0.25}),
                json.dumps({"ap": 0.5}),
                now,
                now,
                now,
            ),
        )
        connection.commit()
        connection.close()

        first = TUNING.TuningService(self.web_root, legacy_runtime, start_worker=False)
        with first.connect() as current:
            legacy = current.execute(
                "SELECT mode,evaluation_scope,status,before_json,after_json FROM model_runs "
                "WHERE id='legacy-run'"
            ).fetchone()
            current.execute(
                "INSERT INTO model_runs(id,session_id,user_id,mode,base_method,evaluation_scope,"
                "status,params_json,annotation_sha256,annotation_count,positive_count,negative_count,"
                "artifact_relpath,created_at) VALUES(?,?,?,?,?,'validation','queued','{}','hash',"
                "4,2,2,'new/run',?)",
                ("new-run", "session", "user", "fusion-weight", "Ours-Full", now),
            )
            migrated_sql = str(current.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_runs'"
            ).fetchone()[0])
        self.assertEqual(tuple(legacy[:3]), ("prototype", "test", "succeeded"))
        self.assertAlmostEqual(json.loads(legacy[3])["ap"], 0.25)
        self.assertAlmostEqual(json.loads(legacy[4])["ap"], 0.5)
        self.assertIn("fusion-weight", migrated_sql)
        self.assertIn("residual", migrated_sql)
        self.assertIn("weight_staged", migrated_sql)
        self.assertIn("weight_joint", migrated_sql)
        self.assertIn("probe_staged", migrated_sql)
        self.assertIn("probe_joint", migrated_sql)
        self.assertIn("probe-validation", migrated_sql)
        self.assertIn("clean-validation", migrated_sql)

        second = TUNING.TuningService(self.web_root, legacy_runtime, start_worker=False)
        with second.connect() as current:
            rows = current.execute(
                "SELECT id,mode FROM model_runs ORDER BY id"
            ).fetchall()
            second_sql = str(current.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_runs'"
            ).fetchone()[0])
        self.assertEqual(
            [(str(row[0]), str(row[1])) for row in rows],
            [("legacy-run", "prototype"), ("new-run", "fusion-weight")],
        )
        self.assertEqual(second_sql, migrated_sql)

    def test_bundle_rejects_overlap_and_non_query_split_gaps(self):
        data_root = self.web_root / "public" / "data" / "tasks" / TASK_ID
        development_path = data_root / "development-mask.u8"
        original = development_path.read_bytes()
        development_path.write_bytes(bytes([1, 1, 1, 1, 0, 1, 0, 0, 0]))
        self.service.bundle.cache_clear()
        with self.assertRaisesRegex(RuntimeError, "exactly one split"):
            self.service.bundle(TASK_ID)

        development_path.write_bytes(bytes([1, 1, 1, 0, 0, 0, 0, 0, 0]))
        self.service.bundle.cache_clear()
        with self.assertRaisesRegex(RuntimeError, "exactly one split"):
            self.service.bundle(TASK_ID)

        development_path.write_bytes(original)
        self.service.bundle.cache_clear()
        self.service.bundle(TASK_ID)

    def test_create_run_revalidates_legacy_annotations_against_development(self):
        client = ApiClient(self.base_url)
        bootstrap = self.bootstrap(client, "Legacy annotation")
        session_id = bootstrap["session"]["id"]
        now = TUNING.utc_now()
        with self.service.connect() as connection:
            connection.execute(
                "INSERT INTO annotations(session_id,row_index,image_id,label,source,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (session_id, 5, self.image_ids[5], 1, "legacy", now, now),
            )
        status, value = self.historical_run_fixture(
            bootstrap["user"]["id"], session_id,
            {"mode": "residual", "baseMethod": "Image Prototype"},
        )
        self.assertEqual(status, 409)
        self.assertIn("Validation", value["error"])

    def test_trusted_oai_header_maps_to_stable_internal_user(self):
        trusted = TUNING.TuningService(
            self.web_root,
            self.root / "runtime-oai",
            trust_oai_headers=True,
            start_worker=False,
        )
        headers = {
            "oai-authenticated-user-id": "workspace-user-123",
            "oai-authenticated-user-email": "researcher@example.test",
        }
        first, created = trusted.identify(headers, create=True)
        second, created_again = trusted.identify(headers, create=True)
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["identitySource"], "openai")


class AddonTuningSourceTests(unittest.TestCase):
    def test_addon_tuning_source_resolves_only_audited_repository_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_root = Path(temporary)
            web_root = experiment_root / "visual_analytics" / "pcp_analyze" / "web"
            manifest = experiment_root / "configs" / "experiments" / "addon.json"
            suite = experiment_root / "configs" / "experiments" / "suite.json"
            ours_root = experiment_root / "outputs" / "addon-run"
            audit = ours_root / "tasks" / TASK_ID / "supervision_audit.json"
            write_json(manifest, {"tasks": []})
            write_json(suite, {"learned_methods": []})
            write_json(audit, {"status": "valid"})

            service = object.__new__(TUNING.TuningService)
            service.web_root = web_root.resolve()
            entry = {
                "catalogRole": "addon",
                "tuningSource": {
                    "schemaVersion": 1,
                    "manifest": "configs/experiments/addon.json",
                    "suite": "configs/experiments/suite.json",
                    "oursFullRoot": "outputs/addon-run",
                    "supervisionAudit": (
                        f"outputs/addon-run/tasks/{TASK_ID}/supervision_audit.json"
                    ),
                    "probeStage": "iterative",
                },
            }
            observed = service._catalog_tuning_source(entry, TASK_ID)
            self.assertEqual(observed.manifest_path, manifest.resolve())
            self.assertEqual(observed.suite_path, suite.resolve())
            self.assertEqual(observed.ours_full_root, ours_root.resolve())
            self.assertEqual(observed.supervision_audit_path, audit.resolve())
            self.assertEqual(observed.probe_stage, "iterative")

            with self.assertRaisesRegex(ValueError, "no audited tuningSource"):
                service._catalog_tuning_source({"catalogRole": "addon"}, TASK_ID)
            unsafe = json.loads(json.dumps(entry))
            unsafe["tuningSource"]["suite"] = "../private.json"
            with self.assertRaisesRegex(ValueError, "repository-relative"):
                service._catalog_tuning_source(unsafe, TASK_ID)


if __name__ == "__main__":
    unittest.main()
