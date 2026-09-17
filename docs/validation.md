# Release validation

## Current checks — 2026-09-17

| Check | Result |
|---|---|
| Probe learning | 38 tests passed |
| Web backend | 88 tests passed |
| Frontend | 37 tests passed |
| Local launcher | 17 tests passed: asset selection, download/import recovery, archive layouts and boundaries, disk space, query identity, setup access and shard reconstruction |
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
Full HICO/CelebA new-task training remains unverified.

## One-click launcher

A separate Windows checkout installed uv, Python 3.12.12, Node 24.11.1, Web
dependencies and CPU PyTorch from their public download sources. Starter fetched
the pinned Cars task ZIP, prepared all 7 tasks and 162 thumbnail atlases, and
opened the Web/API pair on available ports. Browser checks passed for the setup
page, gallery thumbnails and enlarged original images. Weight Tune / Staged ran
10 iterations using the same 44 fixed Val examples without Test metrics.

Full installed its additional Python dependencies and verified all Cars feature
and probe assets. Existing large public payloads were reused through hard links
in the test checkout; manifests and a missing metadata file were downloaded.
All 7 Cars banks reported native-update capability. Basic-to-Full switching,
offline restart and duplicate-launch handling passed.

HICO/CelebA download selection uses the published dataset manifest entries;
CelebA reconstruction is covered with a small sharded fixture. Their complete
downloads and thumbnail preparation were not rerun for this launcher release.

## Automatic original-image extraction

The browser setup imported the original Cars TGZ (1.96 GB compressed), extracted
all 16,185 images directly to a selected destination (1.85 GiB of pictures),
checked the 7-task catalog and opened the gallery with working full-size images.
No intermediate TAR or second extracted image tree was created.

ZIP/TGZ wrapper removal, HICO train/test layout, interrupted-import reuse, query
image mismatches, low disk space and unsafe archive entries are covered by tests.
The split-7z test used the real pinned 7-Zip executable, including Unicode paths
and a missing-volume failure. Complete HICO/CelebA original archives were not
extracted in this validation.

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
python -m unittest discover -s scripts/tests
cd probe_learning
uv run python -B -m unittest discover -s tests
cd ../visual_analytics/pcp_analyze/web
npm run test:tuning
npm run test:unit
```

Core tests cover probe
numerics, Val isolation, feedback and Development/Frozen Test metric separation.
`npm run test:assets` additionally checks prepared Main17 bundles; set
`PROBESCOUT_ASSET_WEB` to use bundles outside the source checkout.
