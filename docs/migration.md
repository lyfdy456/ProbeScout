# Source migration, 2026-09-15

The source copy uses these names:

| Original directory | Release directory |
|---|---|
| `linear_probing` | `probe_learning` |
| `pVIS` (code under `pcp_analyze`) | `visual_analytics` |
| `vlm auto attributes extraction` | `attribute_annotation` |

The training project is named `probescout-learning`; the Web package is named `probescout-visual-analytics`. Python imports, relative filesystem paths, scripts, dependency-lock project names, and documentation references were updated together. The internal `src` package and `pcp_analyze/web` nesting are preserved.

The old research directories remain intact. A ZIP of original authored code/configuration/documentation was created and verified before rewriting source files. Its file inventory, original SHA-256 values, and the source-to-release mapping are retained in the private workspace backup/report directories. Environments, model weights, large datasets, generated results, and mutable runtime data were not duplicated.

Old trained checkpoints keep their original trainer, supervision, split, and calibration identities. New training with the renamed code produces a new trainer identity. Do not rewrite archived hashes merely because directories were renamed. Public asset packaging must add the release-side location mapping while retaining source provenance.

This directory intentionally excludes historical human feedback, session databases, tuned user-model runs, and case replay records. Feedback algorithms and synthetic tests remain source code. The old `method_code_submission` duplicate is retained in the backup/original workspace; it is not the release implementation of the current full scoring method.

After the naming migration, the publication copy was separately backed up and curated around Main17. Historical experiment configurations and exploration drivers were removed. The eight probes share a method catalogue; gate calibration and retrieval metrics now have reusable modules; score/rank analysis has one canonical implementation. Root-level commands cover fixed-Validation training and Tables 2 and 5. Runtime compatibility needed to read frozen asset identities remains.

The Web configuration now runs locally without private hosting files. See [validation.md](validation.md) for the current test/build results and numerical reproduction, and [workflow.md](workflow.md) for the public commands. Full paper retraining was not part of this cleanup.
