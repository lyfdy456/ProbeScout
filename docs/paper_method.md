# Paper-to-code map

## Eight probes

Canonical order: [catalog.py](../probe_learning/src/methods/catalog.py), matching
[paper_main17.json](../manifests/paper_main17.json).

| Paper name | ID | File / function in `probe_learning/src/methods/` | Features |
|---|---|---|---|
| MLP | `mlp_baseline` | `mlp_probe.py`: `train_one_attribute` | Global |
| K-Fold | `kfold_pu` | `pu_probe.py`: `cross_validate_probs`, `train_one_attribute_weighted` | Global |
| Triplet Loss | `triplet_loss` | `metric_attention_probes.py`: `train_one_attribute_triplet_loss` | Global |
| Attention Pooling | `attention_pooling` | `metric_attention_probes.py`: `train_one_attribute_attention_pooling` | Patches |
| Attribute-conditioned Attention | `attribute_conditioned_attention` | `attn_probe.py`: `train_one_attribute_attn` | Patches + text |
| nnPU | `nnpu` | `nnpu_probe.py`: `train_one_attribute_nnpu` | Global |
| DC-PU | `dcpu` | `dcpu_probe.py`: `train_one_attribute_dcpu` | Global |
| PURA | `pu_ranking` | `pu_ranking_probe.py`: `train_one_attribute_pu_ranking` | Global |

Dispatch: `run_retrieval_harness.train_method`. Public entry:
`scripts/train_probes.py`. Names such as `Ours-PURA` in immutable assets are
retained identifiers, not additional probes.

## Main17 and algorithms

| Paper content | Entry / implementation | Configuration |
|---|---|---|
| Table 2 | `scripts/evaluate_main17.py` | `manifests/paper_main17.json` |
| CLAY | `web/scripts/run_clay_full_gallery.py` | `configs/clay.json` |
| Table 5 | `scripts/evaluate_ablation.py`; `src/evaluation/components.py` | `configs/component_ablation.json` |
| AP and frozen-cutoff F1 | `src/evaluation/retrieval_metrics.py` | Original VQA Val labels |
| Fit-only reference gates | `src/methods/gate_calibration.py` | Attribute grid, constrained joint BCE |
| Forward score | `web/scripts/tuning_models.py`: `unified_weight_scores` | Frozen evidence and parameters |
| Staged feedback | `web/scripts/tuning_models.py`: `fit_unified_weight_refinement` | `weight_staged` |

Here `src/` is under `probe_learning/`; `web/` is under
`visual_analytics/pcp_analyze/`.

## Component ablations

| ID | Attribute evidence | Holistic factor |
|---|---|---|
| `full` | Product of SoftGate outputs | `0.75 + 0.25 * H` |
| `no_softgate` | Product of ungated attribute means | `0.75 + 0.25 * H` |
| `no_embedding` | Product of SoftGate outputs | `1` |
| `neither` | Product of ungated attribute means | `1` |

Gated variants select one task-local shift from `{0, -0.05, 0.05, -0.10, 0.10}`
and temperature multiplier from `{1, 1.5, 2, 3}`. Clip theta to `[0,1]` and cap T
at `0.30`. Select by original VQA Val AP; ties within `1e-12` prefer the earlier,
least changed candidate. F1 cutoffs use complete tied-score groups on Val.
No probe retraining is needed for these ablations.

## Acquisition and other study scopes

`iterative_vqa_runner.py` implements initialization with 100 images and rounds
of 40 fitting plus 10 audit images. Acquisition, prefix and budget-training
helpers remain; trajectories and per-budget contracts are separate assets.
The paper's fixed budgets are 50, 100, 150, 200 and 300 images.

The 12-query reuse and 10-task exhaustive-VQA studies are distinct cohorts.
Their standalone study bundles are outside the Table 2/Table 5 entries above;
do not substitute Main17 or a historical 23/45-task list for them.

Historical human-operated feedback comparisons and case replay data are excluded.
The feedback algorithm/interface remain available for new input.
