# Visual analytics

The interface, fusion algorithms and API are in `web/`. See the
[workflow](../../docs/workflow.md) for public paper commands.

This directory holds the single canonical implementations of
`build_score_rank_pcp.py`, `cluster_score_rank_profiles.py` and
`export_multitarget_scores.py`. Training-side duplicate copies were removed.
Their input manifests must describe the matching source cache generation.

`web/scripts/export_task_bundles.py` orchestrates bundle export, defaulting to
Main17. Rebuilding needs task records, cache paths, images and calibration
metadata. For pinned paper results and inference, use frozen evidence packages
with the root evaluation commands.
