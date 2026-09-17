# Release validation

## Current checks — 2026-09-17

| Check | Result |
|---|---|
| Probe learning | 38 tests passed |
| Web backend | 88 tests passed |
| Frontend | 37 tests passed |
| TypeScript | `tsc --noEmit` passed |
| Release boundary | Source/configuration, credential fields and exclusions passed |

The gate tests cover fixed-Val AP selection over all 20 paper candidates,
64-bit gate export/reload, checkpoint linkage, and exclusion of non-Val data.
Legacy `query_text` is ignored without changing a task's input identity.

A real Cars new task completed all eight methods and five seeds for two
attributes: 80 checkpoints, 16,185 gallery images, 176 fit rows and 44 Val rows.
The one-epoch run exported paper-calibrated fusion, 162 thumbnail atlases,
projections and clusters. Reloaded scores matched the recorded Val AP/F1, and
Weight Tune / Staged completed against the same 44 Val rows.

This is a pipeline check; the default scientific training run uses 100 epochs.
Validation used Windows with existing Python/Node dependencies. Fresh dependency
installation and full HICO/CelebA new-task training remain unverified.

## Earlier release checks

All 17 task ZIP entries loaded their original labels, fixed Val and F0 in an
isolated copy. Cars passed browser checks for query images, galleries, projection,
feedback and native probe updates. The production Web build passed on 2026-09-17.

The earlier paper-cleanup evaluation matched the frozen reference scores and
metrics for Table 2 (204 task-method rows) and Table 5 (68 task-variant rows).
Full-model macro Test AP was `0.8320685393249247` and Gallery AP
`0.8232782494346567`. These checks used local evaluation assets; see
[assets](assets.md) for the required inputs.

## Run checks

```sh
# From the repository root
python scripts/check_release.py
cd probe_learning
uv run python -B -m unittest discover -s tests
cd ../visual_analytics/pcp_analyze/web
npm run test:tuning
npm run test:unit
```

The release retains 27 core test files plus two support files. Tests cover probe
numerics, Val isolation, feedback and Development/Frozen Test metric separation.
`npm run test:assets` additionally checks prepared Main17 bundles; set
`PROBESCOUT_ASSET_WEB` to use bundles outside the source checkout.
