"""Pure helpers for the browser's audited evaluation partitions.

The Frozen Test split intentionally mirrors
``iterative_vqa_runner._frozen_split`` without importing the training stack.
It only needs the already-audited ordered image IDs, per-attribute GT columns,
and fixed Query IDs that are present in every browser bundle.

The canonical seed-42 split is a *candidate* Frozen Test.  Any row present in
the declared active training supervision is removed from that candidate
without replacement and assigned to Development.  Validation is still formed
against the original candidate Test, so repairing Test contamination cannot
silently resample Validation.  The fixed Query remains outside every scope.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


FROZEN_TEST_FRACTION = 0.20
FROZEN_TEST_SEED = 42
VALIDATION_REMAINDER_FRACTION = 0.25
VALIDATION_SEED = 43
CLEAN_TEST_POLICY = "stratified-seed42-minus-active-training-supervision-v1"


@dataclass(frozen=True)
class EvaluationMasks:
    development: np.ndarray
    validation: np.ndarray
    frozen_test: np.ndarray
    candidate_frozen_test: np.ndarray
    excluded_training: np.ndarray


def build_frozen_test_mask(
    *,
    image_ids: Sequence[str],
    ground_truth: np.ndarray,
    attribute_indices: Sequence[int],
    query_image_ids: Iterable[str],
    fraction: float = FROZEN_TEST_FRACTION,
    seed: int = FROZEN_TEST_SEED,
) -> np.ndarray:
    """Return a row-aligned uint8 mask for the canonical frozen Test split."""

    ids = [str(value) for value in image_ids]
    labels = np.asarray(ground_truth)
    attributes = [int(value) for value in attribute_indices]
    query_ids = {str(value) for value in query_image_ids}

    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Frozen Test requires non-empty, unique image IDs")
    if labels.ndim != 2 or labels.shape[0] != len(ids):
        raise ValueError(
            "Ground truth must be [image,target] and align with image IDs: "
            f"{labels.shape} versus {len(ids)}"
        )
    if not attributes or len(set(attributes)) != len(attributes):
        raise ValueError("Frozen Test requires unique attribute target indices")
    if min(attributes) < 0 or max(attributes) >= labels.shape[1]:
        raise ValueError(f"Attribute target indices are out of range: {attributes}")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("Frozen Test fraction must be strictly between zero and one")
    if not query_ids or not query_ids.issubset(set(ids)):
        missing = sorted(query_ids.difference(ids))
        raise ValueError(f"Frozen Test Query IDs are missing from the gallery: {missing[:3]}")
    if not np.isin(labels[:, attributes], (0, 1)).all():
        raise ValueError("Frozen Test stratification labels must be binary")

    buckets: dict[tuple[int, ...], list[str]] = {}
    for row_index, image_id in enumerate(ids):
        if image_id in query_ids:
            continue
        key = tuple(int(labels[row_index, index]) for index in attributes)
        buckets.setdefault(key, []).append(image_id)

    rng = np.random.default_rng(int(seed))
    test_ids: set[str] = set()
    for bucket in buckets.values():
        ordered = sorted(bucket)
        rng.shuffle(ordered)
        selected_count = int(round(float(fraction) * len(ordered)))
        test_ids.update(ordered[:selected_count])

    mask = np.fromiter(
        (1 if image_id in test_ids else 0 for image_id in ids),
        dtype=np.uint8,
        count=len(ids),
    )
    if int(mask.sum()) != len(test_ids):
        raise RuntimeError("Frozen Test mask construction lost row alignment")
    return mask


def _validated_inputs(
    *,
    image_ids: Sequence[str],
    ground_truth: np.ndarray,
    attribute_indices: Sequence[int],
    query_image_ids: Iterable[str],
) -> tuple[list[str], np.ndarray, list[int], set[str]]:
    ids = [str(value) for value in image_ids]
    labels = np.asarray(ground_truth)
    attributes = [int(value) for value in attribute_indices]
    query_ids = {str(value) for value in query_image_ids}
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Evaluation partitions require non-empty, unique image IDs")
    if labels.ndim != 2 or labels.shape[0] != len(ids):
        raise ValueError(
            "Ground truth must be [image,target] and align with image IDs: "
            f"{labels.shape} versus {len(ids)}"
        )
    if not attributes or len(set(attributes)) != len(attributes):
        raise ValueError("Evaluation partitions require unique attribute target indices")
    if min(attributes) < 0 or max(attributes) >= labels.shape[1]:
        raise ValueError(f"Attribute target indices are out of range: {attributes}")
    if not query_ids or not query_ids.issubset(set(ids)):
        missing = sorted(query_ids.difference(ids))
        raise ValueError(f"Evaluation Query IDs are missing from the gallery: {missing[:3]}")
    if not np.isin(labels[:, attributes], (0, 1)).all():
        raise ValueError("Evaluation stratification labels must be binary")
    return ids, labels, attributes, query_ids


def build_evaluation_masks(
    *,
    image_ids: Sequence[str],
    ground_truth: np.ndarray,
    attribute_indices: Sequence[int],
    query_image_ids: Iterable[str],
    test_mask: np.ndarray | None = None,
    training_image_ids: Iterable[str] = (),
    validation_fraction: float = VALIDATION_REMAINDER_FRACTION,
    validation_seed: int = VALIDATION_SEED,
) -> EvaluationMasks:
    """Build mutually exclusive Development/Validation/Frozen Test masks.

    ``test_mask`` is the pre-exclusion candidate Test, retained for backward
    compatibility with callers that persist the canonical seed-42 mask.  When
    omitted, that candidate is regenerated.  Rows named by
    ``training_image_ids`` are removed from the candidate Test without
    replacement.  Validation samples 25% of each attribute-GT tuple bucket
    after excluding the *candidate* Test, which keeps Validation byte-stable
    when contaminated Test rows are moved to Development.  Query rows are zero
    in all three public masks.
    """

    ids, labels, attributes, query_ids = _validated_inputs(
        image_ids=image_ids,
        ground_truth=ground_truth,
        attribute_indices=attribute_indices,
        query_image_ids=query_image_ids,
    )
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("Validation remainder fraction must be strictly between zero and one")

    if test_mask is None:
        candidate_frozen_test = build_frozen_test_mask(
            image_ids=ids,
            ground_truth=labels,
            attribute_indices=attributes,
            query_image_ids=query_ids,
        )
    else:
        candidate_frozen_test = np.asarray(
            test_mask, dtype=np.uint8
        ).reshape(-1).copy()
        if (
            candidate_frozen_test.shape != (len(ids),)
            or not np.isin(candidate_frozen_test, (0, 1)).all()
        ):
            raise ValueError(
                "Candidate Frozen Test mask must be a row-aligned binary vector"
            )

    id_to_index = {image_id: index for index, image_id in enumerate(ids)}
    query_indices = np.asarray([id_to_index[value] for value in query_ids], dtype=np.int64)
    if np.any(candidate_frozen_test[query_indices]):
        raise ValueError("Candidate Frozen Test mask includes a fixed Query row")

    training_ids = [str(value) for value in training_image_ids]
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("Active training supervision contains duplicate image IDs")
    unknown_training = sorted(set(training_ids).difference(id_to_index))
    if unknown_training:
        raise ValueError(
            "Active training supervision contains image IDs outside the gallery: "
            f"{unknown_training[:3]}"
        )
    training_mask = np.zeros(len(ids), dtype=np.uint8)
    if training_ids:
        training_mask[
            np.asarray([id_to_index[value] for value in training_ids], dtype=np.int64)
        ] = 1
    excluded_training = np.asarray(
        candidate_frozen_test & training_mask, dtype=np.uint8
    )
    frozen_test = np.asarray(
        candidate_frozen_test & (1 - training_mask), dtype=np.uint8
    )

    buckets: dict[tuple[int, ...], list[str]] = {}
    for row_index, image_id in enumerate(ids):
        if image_id in query_ids or candidate_frozen_test[row_index]:
            continue
        key = tuple(int(labels[row_index, index]) for index in attributes)
        buckets.setdefault(key, []).append(image_id)

    rng = np.random.default_rng(int(validation_seed))
    validation_ids: set[str] = set()
    for key in sorted(buckets):
        ordered = sorted(buckets[key])
        rng.shuffle(ordered)
        selected_count = int(round(float(validation_fraction) * len(ordered)))
        validation_ids.update(ordered[:selected_count])

    validation = np.fromiter(
        (1 if image_id in validation_ids else 0 for image_id in ids),
        dtype=np.uint8,
        count=len(ids),
    )
    query_mask = np.fromiter(
        (1 if image_id in query_ids else 0 for image_id in ids),
        dtype=np.uint8,
        count=len(ids),
    )
    development = np.asarray(
        1 - np.maximum.reduce((frozen_test, validation, query_mask)),
        dtype=np.uint8,
    )

    if np.any(development & validation) or np.any(development & frozen_test) or np.any(validation & frozen_test):
        raise RuntimeError("Evaluation partitions are not mutually exclusive")
    if np.any((development | validation | frozen_test | query_mask) != 1):
        raise RuntimeError("Evaluation partitions do not cover every non-Query row exactly once")
    if np.any(validation & candidate_frozen_test):
        raise RuntimeError("Validation must remain outside the candidate Frozen Test")
    if not np.array_equal(
        candidate_frozen_test,
        np.asarray(frozen_test | excluded_training, dtype=np.uint8),
    ):
        raise RuntimeError("Frozen Test exclusion did not preserve the candidate membership")
    if np.any(excluded_training & (1 - development)):
        raise RuntimeError("Excluded training rows must be assigned to Development")
    return EvaluationMasks(
        development=development,
        validation=validation,
        frozen_test=frozen_test,
        candidate_frozen_test=candidate_frozen_test,
        excluded_training=excluded_training,
    )


def _positive_counts(
    mask: np.ndarray,
    ground_truth: np.ndarray,
    target_ids: Sequence[str],
) -> dict[str, int]:
    selected = np.asarray(mask, dtype=np.uint8).reshape(-1).astype(bool)
    return {
        str(target): int(np.count_nonzero(ground_truth[selected, index]))
        for index, target in enumerate(target_ids)
    }


def evaluation_contract(
    *,
    development_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    ground_truth: np.ndarray,
    target_ids: Sequence[str],
    attribute_ids: Sequence[str],
    candidate_test_mask: np.ndarray | None = None,
    excluded_training_mask: np.ndarray | None = None,
    training_supervision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the three disjoint model-development/evaluation scopes."""

    masks = {
        "development": np.asarray(development_mask, dtype=np.uint8).reshape(-1),
        "validation": np.asarray(validation_mask, dtype=np.uint8).reshape(-1),
        "frozenTest": np.asarray(test_mask, dtype=np.uint8).reshape(-1),
    }
    labels = np.asarray(ground_truth)
    targets = [str(value) for value in target_ids]
    attributes = [str(value) for value in attribute_ids]
    row_count = masks["frozenTest"].size
    if labels.shape != (row_count, len(targets)):
        raise ValueError(
            "Evaluation GT shape does not match masks/targets: "
            f"{labels.shape} versus {(row_count, len(targets))}"
        )
    for name, mask in masks.items():
        if mask.size != row_count or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{name} mask must be a row-aligned binary vector")
    if np.any(
        masks["development"] + masks["validation"] + masks["frozenTest"] > 1
    ):
        raise ValueError("Evaluation masks must be mutually exclusive")

    development_count = int(np.count_nonzero(masks["development"]))
    validation_count = int(np.count_nonzero(masks["validation"]))
    test_count = int(np.count_nonzero(masks["frozenTest"]))
    candidate = (
        masks["frozenTest"].copy()
        if candidate_test_mask is None
        else np.asarray(candidate_test_mask, dtype=np.uint8).reshape(-1)
    )
    excluded = (
        np.asarray(candidate & (1 - masks["frozenTest"]), dtype=np.uint8)
        if excluded_training_mask is None
        else np.asarray(excluded_training_mask, dtype=np.uint8).reshape(-1)
    )
    for name, mask in (("candidate Test", candidate), ("excluded training", excluded)):
        if mask.size != row_count or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{name} mask must be a row-aligned binary vector")
    if not np.array_equal(
        candidate,
        np.asarray(masks["frozenTest"] | excluded, dtype=np.uint8),
    ):
        raise ValueError(
            "Candidate Test must equal clean Frozen Test plus excluded training rows"
        )
    if np.any(excluded & (1 - masks["development"])):
        raise ValueError("Excluded training rows must belong to Development")
    supervision = None if training_supervision is None else dict(training_supervision)
    return {
        "defaultResultScope": "development",
        "testIsolation": {
            "policy": CLEAN_TEST_POLICY,
            "candidateRowCount": int(np.count_nonzero(candidate)),
            "excludedTrainingRowCount": int(np.count_nonzero(excluded)),
            "noReplacement": True,
            "excludedRowsDestination": "development",
            "activeTrainingSupervision": supervision,
        },
        "scopes": [
            {"id": "development", "label": "Development Gallery", "rowCount": development_count},
            {"id": "validation", "label": "Validation", "rowCount": validation_count},
            {"id": "test", "label": "Frozen Test", "rowCount": test_count},
        ],
        "development": {
            "maskFileKey": "developmentMask",
            "queryExcluded": True,
            "rowCount": development_count,
            "positiveCounts": _positive_counts(masks["development"], labels, targets),
            "groundTruthUsage": "human feedback and model fitting scope; never used for ranking or filtering",
        },
        "validation": {
            "maskFileKey": "validationMask",
            "splitSeed": VALIDATION_SEED,
            "fractionOfEligibleRemainder": VALIDATION_REMAINDER_FRACTION,
            "queryExcluded": True,
            "stratifiedBy": attributes,
            "rowCount": validation_count,
            "positiveCounts": _positive_counts(masks["validation"], labels, targets),
            "groundTruthUsage": "Fusion/SoftGate and model-selection evaluation; never used for ranking or filtering",
        },
        "frozenTest": {
            "maskFileKey": "testMask",
            "splitSeed": FROZEN_TEST_SEED,
            "fraction": FROZEN_TEST_FRACTION,
            "policy": CLEAN_TEST_POLICY,
            "queryExcluded": True,
            "stratifiedBy": attributes,
            "rowCount": test_count,
            "positiveCounts": _positive_counts(masks["frozenTest"], labels, targets),
            "groundTruthUsage": "final post-hoc evaluation only; excluded from active training supervision and never used for ranking or filtering",
        },
    }
