# Asset distribution

GitHub is the source-code repository. Hugging Face is the intended destination for large immutable research assets. Its Hub repositories support ML model and dataset files and versioned revisions: https://huggingface.co/docs/hub/repositories . Storage depends on account policy: https://huggingface.co/docs/hub/storage-limits .

| Destination | Proposed content | Observed primary-file size |
|---|---|---:|
| GitHub | Source, locks, configuration, paper scope, metadata and download/checksum scripts | Small source files |
| Hugging Face model repository | Main17 pretrained probes; 1,520 seed checkpoints and associated metadata | 5.35 GB weights |
| Hugging Face dataset repository | SigLIP global features, records and image-ID alignment, grouped by dataset | 0.82 GB arrays |
| Hugging Face dataset repository | SigLIP patch features, grouped by dataset and sharded for selective download | 80.25 GB arrays |
| Hugging Face dataset repository | Frozen, pre-human-feedback probe scores and initial evidence | 0.314 GB probe caches plus other evidence |

These are decimal GB from the local inventory, not final compressed package sizes. Keep global and patch features separate. Preserve float32 global features, float16 patch tokens, image order, and the current float64 gate parameters. Feature shards should carry row ranges and SHA-256 values.

No Hugging Face repository has been created or uploaded by this source migration. Account namespace, repository IDs, revisions, exact file checksums, licenses, and completed package sizes remain unset. A manifest must pin an immutable revision instead of relying on a moving `main` or `latest` pointer.

## Publication selection

The main17 initial snapshot is `f0-val-36-20260908`. Models are selected by the `bankDirectory` in each frozen task manifest. Avoid directory-wide uploads of the research runtime: that tree also contains mutable user state. A future exporter should copy only selected frozen-bank, initial-evidence, feature and index files into a clean staging folder, then upload that staging folder.

Human feedback is excluded from GitHub **and** Hugging Face: no annotations, user/session databases, labels snapshots, human-refined weights/scores, or case replay bundles. Original VQA supervision for probe training is a separate artifact class; it must not be replaced with later human-edited feedback data.

Every downloadable file should identify its relative path, bytes, SHA-256, artifact type, dataset/gallery identity, feature identity, training/split/calibration provenance, dependency IDs, Hugging Face repository type/ID/revision, and license. Keep data paths separate from public source paths; do not change original artifact hashes to match renamed code.
