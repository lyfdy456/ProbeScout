"""Val is one immutable population, never an unlabeled fallback or per-seed split."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from val_isolation import PROTOCOL, digest, materialize_training_inputs, partition_rows


def fixture():
    fit = list(range(12))
    val = [12, 13, 14, 15]
    splits = {name: SimpleNamespace(
        fit_indices=fit, fit_labels=[row % 2 for row in fit],
        validation_indices=val, validation_labels=[0, 1, 0, 1],
        audit={"sourceFingerprint": name, "fitFingerprint": name, "validationFingerprint": name},
    ) for name in ("first", "second", "joint")}
    pools = partition_rows(24, bytes([1] * 20 + [0] * 4),
                           bytes([0] * 22 + [1, 0]), {23}, val, splits)
    ids = [f"folder/{row}.jpg" for row in range(24)]
    contract = {"protocol": PROTOCOL, "rowCount": 24,
                "imageIdsSha256": digest(ids), "valVersion": "v1", "valManifestSha256": "val-hash",
                "attributes": [{"id": name, "name": name} for name in ("first", "second")], **pools}
    contract["fingerprint"] = digest(contract)
    return ids, splits, contract


class IsolationTests(unittest.TestCase):
    def test_fit_u_calibration_and_normalization_never_include_protected_rows(self):
        ids, _, contract = fixture()
        result = materialize_training_inputs(contract, ids, ids)
        blocked = {ids[row] for row in [12, 13, 14, 15, 22, 23]}
        self.assertFalse(set(result["fit_paths"]) & blocked)
        self.assertFalse(set(result["train_pool"]) & blocked)
        unlabeled = set(result["train_pool"]) - set(result["fit_paths"])
        self.assertEqual(unlabeled, {ids[row] for row in range(16, 22)})
        self.assertEqual(result["normalization_indices"], list(range(12)) + list(range(16, 20)))
        self.assertEqual(contract["targets"]["joint"]["fitRows"], list(range(12)))

    def test_offline_order_is_mapped_by_id(self):
        ids, _, contract = fixture()
        result = materialize_training_inputs(contract, ids, list(reversed(ids)))
        self.assertEqual(result["val_indices"], [11, 10, 9, 8])
        self.assertEqual(result["fit_labels"][0]["image"], ids[0])

    def test_all_seeds_receive_the_same_immutable_inputs(self):
        ids, _, contract = fixture()
        outputs = [materialize_training_inputs(contract, ids, ids) for _ in range(5)]
        self.assertTrue(all(output == outputs[0] for output in outputs))

    def test_stale_or_wrong_gallery_identity_fails(self):
        ids, _, contract = fixture()
        for invalid in [list(reversed(ids)), ids[:-1], ids[:-1] + [ids[0]]]:
            with self.assertRaisesRegex(RuntimeError, "identity"):
                materialize_training_inputs(contract, invalid, ids)

    def test_contract_mutation_fails(self):
        ids, _, contract = fixture()
        contract["normalizationRows"].append(12)
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            materialize_training_inputs(contract, ids, ids)

    def test_val_cannot_overlap_test_query_fit_or_differ_by_attribute(self):
        _, source, _ = fixture()
        for mode in ("fit", "test", "query", "attr"):
            splits = copy.deepcopy(source)
            test = bytes([0] * 24)
            query = set()
            if mode == "fit":
                splits["first"].fit_indices[-1] = 12
            elif mode == "test":
                test = bytes([0] * 12 + [1] + [0] * 11)
            elif mode == "query":
                query = {12}
            else:
                splits["second"].validation_indices = [12, 13, 14, 16]
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                partition_rows(24, bytes([1] * 24), test, query, [12, 13, 14, 15], splits)

    def test_incomplete_or_single_class_labels_do_not_silently_resample(self):
        ids, _, contract = fixture()
        for mode in ("missing", "single_class"):
            changed = copy.deepcopy(contract)
            if mode == "missing":
                changed["targets"]["first"]["fitRows"].pop()
                changed["targets"]["first"]["fitLabels"].pop()
            else:
                changed["targets"]["first"]["valLabels"] = [0] * 4
            changed.pop("fingerprint")
            changed["fingerprint"] = digest(changed)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                materialize_training_inputs(changed, ids, ids)


if __name__ == "__main__":
    unittest.main()
