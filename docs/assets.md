# Prepare assets

Hosting repositories have not yet been published. These paths define a prepared
package; a fresh clone does not contain the assets. See
[asset_distribution.md](asset_distribution.md) for sizes and hosting plans.

## Features and task metadata

```text
<EXPERIMENT_ROOT>/
  dataset/raw/stanford_cars/processed/records.csv
  dataset/raw/stanford_cars/processed/siglip_embedding.npy
  dataset/raw/stanford_cars/processed/siglip_patch_tokens.npy
  dataset/raw/HICO/processed/...
  dataset/raw/CelebA/processed/...
  dataset/tasks/<dataset>/<task>/...
```

Tasks need attributes, query IDs, original VQA supervision, evaluation splits
and image identities. Raw images follow the dataset adapters' layout. Preserve
record order; arrays and labels are aligned by image ID.

## Pretrained inference and Web assets

The prepared package is rooted at `visual_analytics/pcp_analyze/`:

```text
web/public/data/catalog.json
web/public/data/tasks/<task-id>/...
runtime/unified-initial/active.json
runtime/unified-initial/<version>/manifest.json
runtime/unified-initial/<version>/<task-id>/...
runtime/isolated-probes/<task-id>/<bank-id>/...
runtime/evaluation/...             # immutable original-VQA Val contracts
```

Some pinned bundles use older relative data-root names recorded in the catalog.
Preserve those paths and checksums. A version can include `36` while evaluation
explicitly selects Main17. Include parent manifest/attestation metadata needed
to verify bank provenance. Human-feedback sessions are not package inputs.

Table 2 also requires the CLAY cache: `manifest.json`, `run-inputs.json`, and
per-task `manifest.json` / `scores.npz`. Source/image/label/split identities are
checked. A gate-only F0 change can reuse the attested parent's unchanged CLAY scores.

## Fixed-Validation training

`train_probes.py --directory ...` expects the original contract directory:
`manifest.json` plus the numeric task files named by it. These are checked against
active immutable VQA Val metadata and task sources. This differs from the
simplified Table 5 index below; features alone do not suffice for paper training.

## Table 5 inputs

[component_inputs.json](../manifests/component_inputs.json) pins 17 evidence
snapshots and 17 original-VQA contracts. Paths resolve against `--assets`:

```text
<evaluation-assets>/
  evaluation/component_ablation/<task-id>.npz
  evaluation/val-isolation/<task-id>.json
```

Snapshots hold normalized probe/holistic scores, reference gates, ordered IDs,
original fit labels, Val/normalization rows, Test mask and evaluation GT.
Contracts supply VQA Val labels and protected partitions. Checks cover SHA-256,
fingerprints, shapes, identities and partition isolation. These evaluation arrays
are distinct from large patch features; preserve their pinned bytes.

## Provenance and exclusions

Old checkpoints retain their trainer identities; refactored code produces new
training identities. Hosting manifests must pin immutable revisions and checksums.
Historical human annotations, sessions, snapshots, refined models/scores and case
replay are excluded from both GitHub and Hugging Face. Keep original VQA labels
separate from subsequent human edits.
