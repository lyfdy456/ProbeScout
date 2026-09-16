from __future__ import annotations

import hashlib
import importlib.util
import json
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


    def test_weight_refinement_modes_execute_from_the_same_synthetic_base(self):
        self._check_weight_refinement_modes()


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


if __name__ == "__main__":
    unittest.main()
