# Workflow

Run commands from the repository root unless a different directory is shown.
[Manual setup](manual_setup.md) covers images, task ZIPs and Web launch;
[new tasks](new_tasks.md) covers your own queries and supervision.

## Features

Download the selected dataset's [HF features](asset_distribution.md), or extract
them from local images in canonical `records.csv` order:

```sh
uv run --project probe_learning python probe_learning/scripts/precompute_backbone_embeddings.py --dataset cars --backbone siglip
uv run --project probe_learning python probe_learning/scripts/precompute_backbone_patch_tokens.py --dataset cars --backbone siglip
```

Replace `cars` with `hico` or `celeba`. Outputs are `siglip_embedding.npy` and
`siglip_patch_tokens.npy` in that dataset's `processed/` directory.

## Main17 training

The task ZIP provides original VQA labels and fixed Val inputs. Prepare the Web
catalog, install features, then run:

```sh
python scripts/train_probes.py --list
uv run --project probe_learning python scripts/train_probes.py --task-id 001_cars_task_bmw_convertible
uv run --project probe_learning python scripts/train_probes.py --task-id 001_cars_task_bmw_convertible --execute
```

The first training command checks inputs; `--execute` trains. Repeat `--task-id`
for a subset or omit it for all Main17 tasks. Defaults are eight methods, five
seeds and 100 epochs with fixed-Val checkpoint selection, as specified in
[paper_training.json](../configs/paper_training.json).

New banks are written to
`visual_analytics/pcp_analyze/runtime/isolated-probes/<task-id>/<identity>/`.
They can be selected/exported separately from the published initial ranking.

## Fusion

The shared implementation is `unified_weight_scores` in
`visual_analytics/pcp_analyze/web/scripts/tuning_models.py`:

```text
u = weighted mixture of eight attribute-probe scores
g = sigmoid((u - theta) / temperature)
C = product(g ** gamma)
H = weighted mixture of query-image and attribute-text prompt scores
F = C * ((1 - lambda) + lambda * H)
```

Initial weights are uniform, with `gamma=1` and `lambda=0.25`. Final gates are
selected on fixed Val. Published tasks load the F0 included in their ZIP;
`custom_task.py` performs the calibration and export for new tasks.

For low-level Main17 exports, the scripts are
`visual_analytics/pcp_analyze/web/scripts/unified_initial_baseline.py` (fit reference)
and `visual_analytics/pcp_analyze/web/scripts/publish_val_initial_baseline.py`
(Val-selected publication). The latter accepts `--inputs` pointing to a frozen F0
selection bundle and `--version`; `--publish` activates the complete catalog.
These inputs are distinct from Table 2/5 result reports.

## Feedback

**Weight Tune / Staged is the paper method.** It freezes probes, normalization
and gates. Stage 1 updates attribute-probe weights; Stage 2 updates attribute
exponents and the holistic branch. The implementation is
`fit_unified_weight_refinement` in `tuning_models.py`.

**Update Probes is an optional extension.** It updates probe parameters using
supervision and new feedback and requires features plus the matching checkpoint
bank. It is not the paper's frozen-probe feedback protocol.

Development shows Val metrics. Test evaluation is confined to Frozen Test.

## Paper evaluation

After preparing the separate [evaluation inputs](assets.md):

```sh
uv run --project probe_learning python scripts/evaluate_main17.py --clay artifacts/evaluation/clay --output outputs/table2
uv run --project probe_learning python scripts/evaluate_ablation.py --assets artifacts/evaluation --output outputs/table5
```

The paths above are destinations for the required inputs, not files included in
the source checkout. Both commands write `val_selection.json` and `results.json`.
Table 2 can use `--web visual_analytics/pcp_analyze/web` for prepared Web evidence;
Table 5 reads the snapshots pinned in [component_inputs.json](../manifests/component_inputs.json).
