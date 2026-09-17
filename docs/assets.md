# Prepare assets

Public task, feature and probe packages are available on HF. A fresh clone has
source only. Follow [manual_setup.md](manual_setup.md) to install one dataset and
launch the Web interface; [asset_distribution.md](asset_distribution.md) lists
download links, pinned revisions and optional feature/model subsets.

## Features and task metadata

```text
ProbeScout/
  dataset/raw/stanford_cars/processed/records.csv
  dataset/raw/stanford_cars/processed/siglip_embedding.npy
  dataset/raw/stanford_cars/processed/siglip_patch_tokens.npy
  dataset/raw/HICO/processed/...
  dataset/raw/CelebA/processed/...
  dataset/tasks/<dataset>/<task>/...
```

Tasks need attributes, query IDs, original VQA supervision, evaluation splits
and image identities. Raw images follow the manual setup layout. Preserve
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
```

Portable label and partition contracts are rooted at
`dataset/tasks/<dataset>/<task>/original_vqa.json` and `isolation.json` in the
repository root, alongside `task.json`, `query_ids.json` and `attributes.txt`.

Table 2 also requires the CLAY cache: `manifest.json`, `run-inputs.json`, and
per-task `manifest.json` / `scores.npz`. Source/image/label/split identities are
checked. A gate-only F0 change can reuse the attested parent's unchanged CLAY scores.

## Fixed-Validation training

The default `train_probes.py` path uses the portable task ZIP's verified original
VQA labels and isolation contract. No additional Val download is required.

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

Package scope and exclusions are listed in [asset distribution](asset_distribution.md#scope).
