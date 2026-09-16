# Workflow

Commands run from the repository root unless stated otherwise. See
[assets.md](assets.md) for input layout.

## 1. Extract or load frozen features

Fix image order with the prepared `records.csv`. Skip extraction when using
downloaded SigLIP features.

```sh
uv run --project probe_learning python probe_learning/scripts/precompute_backbone_embeddings.py --dataset cars --backbone siglip
uv run --project probe_learning python probe_learning/scripts/precompute_backbone_patch_tokens.py --dataset cars --backbone siglip
```

Repeat with `--dataset hico` and `--dataset celeba`. Input: original images and
ordered records. Output: `siglip_embedding.npy` (float32) and
`siglip_patch_tokens.npy` (float16 by default) in each processed dataset directory.
Keep their row order aligned. `EXPERIMENT_ROOT` may point learning utilities to
existing external `dataset/raw` and `dataset/tasks` folders without copying arrays.
Web bundle/runtime paths are configured separately by asset manifests.

## 2. Train or load the eight probes

For pretrained inference, install the bank and initial evidence packages and
continue to step 3. Starting a Web session reads saved scores without retraining.

For training, prepare original VQA labels, fixed Validation contracts, task
metadata, features and matching Web task bundles:

```sh
python scripts/train_probes.py --list
uv run --project probe_learning python scripts/train_probes.py --directory /path/to/frozen-val-contracts --task-id 001_cars_task_bmw_convertible
uv run --project probe_learning python scripts/train_probes.py --directory /path/to/frozen-val-contracts --task-id 001_cars_task_bmw_convertible --execute
```

Omit `--task-id` for all Main17 tasks. Default: preflight. With `--execute`, use
[paper_training.json](../configs/paper_training.json): eight methods, seeds 0-4,
100 epochs and fixed-Validation checkpoint selection. Val, Test and query images
stay out of fitting and the unlabeled pool.

Output: a new bank in
`visual_analytics/pcp_analyze/runtime/isolated-probes/<task-id>/<identity>/`,
with checkpoints, caches and metadata. Code/path changes produce a new training
identity; old assets retain their original provenance.

The trainer uses `probe_learning/scripts/run_probebank_batch.py` for cache
operations. Its acquisition-budget CLI does not replace fixed-Val paper training.

## 3. Fuse evidence and open the interface

The application loads the pinned initial evidence package:

```text
u = weighted mixture of the eight attribute-probe scores
g = sigmoid((u - theta) / temperature)
C = product(g ** gamma)
H = weighted mixture of query-image and query-text scores
F = C * ((1 - lambda) + lambda * H)
```

Read `unified_weight_scores` in
`visual_analytics/pcp_analyze/web/scripts/tuning_models.py`. Initial weights are
uniform over probes and holistic methods, with `gamma = 1`, `lambda = 0.25`.
The final initial gates are selected using original VQA Validation.

Run from `visual_analytics/pcp_analyze/web`:

```sh
npm run dev
```

Input: catalog, task bundles, images/thumbnails, initial evidence and matching
immutable runtime contracts. Output: the local API and interactive views.

For a newly trained bank, the low-level
`scripts/unified_initial_baseline.py --task-id ... --bank-dir ... --version ...`
stages fit-only reference evidence. Final Validation calibration/publication is
a separate step. Do not call newly staged reference scores the pinned paper F0;
the downloadable F0 package is the default inference path. The final staging
utility is `scripts/publish_val_initial_baseline.py --inputs /path/to/frozen-f0-selection --version <new-version>`;
add `--publish` to activate a fully verified local version. Its inputs are the
dedicated F0 selection/evidence contract, not a Table 2 or Table 5 report. Every
task in the installed catalog must be covered before activation.

## 4. Inspect and refine with new feedback

Select images in linked views and provide attribute/query judgments. The paper
uses `weight_staged` with frozen probes and gates. Stage 1 updates attribute
mixtures; Stage 2 updates attribute exponents and the holistic branch.

Read `fit_unified_weight_refinement` in `tuning_models.py`, and supervision
construction in `tuning_supervision.py` / `tuning_server.py`. New local session
state stays in runtime and is excluded from source publication.

Development displays Validation metrics only, including expanded result details.
Test AP, F1 and TP@K are shown only in the read-only Frozen Test view. Historical
Test-only runs and runs without an explicit Validation scope do not display
metrics in Development; their saved rankings can still be applied.

## 5. Reproduce initial results

```sh
uv run --project probe_learning python scripts/evaluate_main17.py --clay /path/to/clay-cache --output outputs/table2
uv run --project probe_learning python scripts/evaluate_ablation.py --assets /path/to/evaluation-assets --output outputs/table5
```

Both commands write `val_selection.json` and `results.json` in a new directory.
Choices use original VQA Val and are frozen before Test/Gallery evaluation.
Historical human-feedback sessions are not read.

Table 2 accepts `--web /path/to/prepared/web` to read existing evidence directly.
Table 5 uses [component_inputs.json](../manifests/component_inputs.json), checking
SHA-256, image identity and partition isolation before selection.
