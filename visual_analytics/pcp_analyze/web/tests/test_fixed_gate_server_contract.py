from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests import test_tuning_backend as backend


TUNING = backend.TUNING


@contextmanager
def toy_api_fixture():
    """Use only a temporary toy runtime and an ephemeral loopback test server."""
    fixture = backend.TuningApiTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.tearDown()
        fixture.doCleanups()


class FixedGateServerContractTests(unittest.TestCase):
    def test_current_and_legacy_normalization_dispatch_stays_version_pinned(self):
        for mode in TUNING.REFINEMENT_RUN_MODES:
            versions = [
                (TUNING.RUN_ALGORITHM_VERSIONS[mode], True),
                (TUNING.PRE_FIXED_GATE_REFINEMENT_ALGORITHMS[mode], True),
            ]
            if mode in TUNING.WEIGHT_REFINEMENT_RUN_MODES:
                versions.extend([
                    (TUNING.PRE_ISOLATION_WEIGHT_REFINEMENT_ALGORITHMS[mode], False),
                    (TUNING.LEGACY_WEIGHT_REFINEMENT_ALGORITHMS[mode], False),
                ])
            for version, isolated in versions:
                with self.subTest(mode=mode, version=version):
                    methods = (
                        TUNING.EMBEDDING_BASELINE_METHODS
                        if version == TUNING.LEGACY_WEIGHT_REFINEMENT_ALGORITHMS.get(mode)
                        else TUNING.REFINEMENT_EMBEDDING_METHODS
                    )
                    audit = {"vqaValidationManifestSha256": "val-manifest"} if isolated else None
                    base = SimpleNamespace(normalization_audit=audit)
                    service = object.__new__(TUNING.TuningService)
                    service._refinement_base_context = mock.Mock(return_value=base)
                    params = {
                        "algorithmVersion": version,
                        "embeddingMethods": list(methods),
                        "vqaValidationVersion": "frozen-val",
                        "vqaValidationManifestSha256": "val-manifest",
                        "normalizationPolicy": TUNING.REFINEMENT_NORMALIZATION_POLICY,
                        "normalizationContract": audit,
                    }
                    self.assertIs(service._refinement_base_for_run("toy", mode, params), base)
                    service._refinement_base_context.assert_called_once_with(
                        "toy", methods, isolate_validation=isolated,
                        vqa_validation_version="frozen-val" if isolated else None,
                        prefer_published=False,
                    )

    def test_worker_rejects_self_consistent_gate_snapshot_tampering_before_fit(self):
        # Recompute both the snapshot SHA and stored params so these checks
        # exercise gate/base identity guards, not only generic file integrity.
        changes = (
            ("gateCalibrationPolicy", "legacy-recalibrate", "policy changed"),
            ("thetaCalibration", "original-vqa-minus-shared-validation", "policy changed"),
            ("temperaturePolicy", "fixed-base-mean", "policy changed"),
            ("initialTheta", [0.6], "threshold calibration changed"),
            ("initialTemperature", [0.3], "temperature changed"),
            ("initialThetaFingerprint", "0" * 64, "threshold calibration changed"),
            ("initialTemperatureFingerprint", "0" * 64, "temperature changed"),
        )
        for native in (False, True):
            for key, value, error in changes:
                with self.subTest(native=native, field=key), toy_api_fixture() as fixture:
                    execute = fixture.service._execute_model_run

                    def tampered_execution(run):
                        run_copy = dict(run)
                        path = fixture.runtime_root / run["artifact_relpath"] / "labels_snapshot.json"
                        snapshot = json.loads(path.read_text(encoding="utf-8"))
                        params = snapshot["params"]
                        params[key] = value
                        if key in {"initialTheta", "initialTemperature"}:
                            params[f"{key}Fingerprint"] = hashlib.sha256(
                                np.asarray(value, dtype="<f4").tobytes()
                            ).hexdigest()
                        payload = (TUNING.json_dumps(snapshot) + "\n").encode("utf-8")
                        TUNING.atomic_write_bytes(path, payload)
                        run_copy["params_json"] = TUNING.json_dumps(params)
                        run_copy["snapshot_sha256"] = hashlib.sha256(payload).hexdigest()
                        return execute(run_copy)

                    fixture.service._execute_model_run = tampered_execution
                    with mock.patch(
                        "tuning_models.fit_unified_weight_refinement",
                        side_effect=AssertionError("tampered gates must be rejected before fitting"),
                    ) as fit, self.assertRaisesRegex(RuntimeError, error):
                        fixture._check_weight_refinement_modes(native=native)
                    fit.assert_not_called()
                    self.assertFalse(any(fixture.runtime_root.rglob("model.npz")))

    def test_worker_rejects_optimizer_gate_drift_before_model_publication(self):
        import tuning_models

        original_fit = tuning_models.fit_unified_weight_refinement
        for key in ("theta", "temperature"):
            with self.subTest(field=key), toy_api_fixture() as fixture:
                def drifting_fit(*args, **kwargs):
                    result = original_fit(*args, **kwargs)
                    return dataclasses.replace(result, **{key: getattr(result, key) + 0.01})

                with mock.patch.object(
                    tuning_models, "fit_unified_weight_refinement", side_effect=drifting_fit,
                ), self.assertRaisesRegex(RuntimeError, "modified the frozen gate parameters"):
                    fixture._check_weight_refinement_modes()
                self.assertFalse(any(fixture.runtime_root.rglob("model.npz")))

    def test_old_worker_versions_explicitly_select_legacy_recalibration(self):
        import tuning_models

        original_fit = tuning_models.fit_unified_weight_refinement
        for legacy, native in (("v2", False), ("v3", False), ("v4", False), ("probe-v2", True)):
            with self.subTest(version=legacy), toy_api_fixture() as fixture:
                with mock.patch.object(
                    tuning_models, "fit_unified_weight_refinement", wraps=original_fit,
                ) as fit:
                    fixture._check_weight_refinement_modes(legacy=legacy, native=native)
                self.assertTrue(fit.called)
                for call in fit.call_args_list:
                    self.assertEqual(call.kwargs["gate_calibration_policy"], "legacy-recalibrate")


if __name__ == "__main__":
    unittest.main()
