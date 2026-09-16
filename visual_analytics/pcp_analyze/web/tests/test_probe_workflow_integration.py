"""End-to-end toy workflow; never touches the production runtime or models."""
import unittest
from unittest import mock

from tests import test_tuning_backend as backend
import native_probe_update
import tuning_models


class ProbeWorkflowIntegrationTests(unittest.TestCase):
    def test_one_independent_update_then_two_frozen_weight_schedules(self):
        fixture = backend.TuningApiTests()
        fixture.setUp()
        events = []
        native = native_probe_update.ensure_snapshot
        weights = tuning_models.fit_unified_weight_refinement

        def update_only(*args, **kwargs):
            events.append("update-probes")
            return native(*args, **kwargs)

        def weights_only(mode, *args, **kwargs):
            events.append(mode)
            self.assertEqual(kwargs["gate_calibration_policy"], "fixed-base")
            return weights(mode, *args, **kwargs)

        try:
            with mock.patch.object(native_probe_update, "ensure_snapshot", side_effect=update_only), \
                    mock.patch.object(tuning_models, "fit_unified_weight_refinement", side_effect=weights_only):
                # Exercises real HTTP creation, owned persistence, snapshot I/O,
                # both optimizers, initial F0 comparison, and exact PCP rebuild.
                fixture._check_weight_refinement_modes(native=True)
            self.assertEqual(events, ["update-probes", "weight_staged", "weight_joint"])
            with fixture.service.connect() as connection:
                self.assertEqual(connection.execute(
                    "SELECT status,COUNT(*) FROM probe_updates GROUP BY status"
                ).fetchall()[0][0], "succeeded")
        finally:
            fixture.tearDown()
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
