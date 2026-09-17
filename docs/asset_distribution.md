# Asset distribution

GitHub contains source, configuration, dependency locks, paper scope and core
tests. Large immutable features, checkpoints and prediction caches are hosted
separately on Hugging Face.

## Available packages

Both private packages are fully uploaded. The upload process verified all
packaged remote file hashes; the pinned revisions were checked again on
2026-09-17. Sizes below are decimal GB, without compression.

| Package | Contents | Size | Pinned revision |
|---|---|---:|---|
| [ProbeScout-probes](https://huggingface.co/Ian100/ProbeScout-probes) | 1,520 checkpoints, 304 frozen prediction caches, training and isolation metadata | 5.69 GB | `3d8fbdcc9921c0a3a20e2c8cbd0cd102b124d904` |
| [ProbeScout-features](https://huggingface.co/datasets/Ian100/ProbeScout-features) | Cars, HICO and CelebA SigLIP global features, patch tokens, records and image IDs | 81.10 GB | `3ef237a82d0c738a159aa49bb86764732d331265` |

These are private review repositories: an authorized HF account is required.
Each package includes `asset_manifest.json` with paths, sizes and SHA-256 values.
The model package has 2,199 packaged files (5,694,587,019 bytes); the feature
package has 44 (81,101,385,535 bytes). The Hub also adds a `.gitattributes` file
to each repository, outside those package counts.

## Download

Use an environment with `huggingface_hub` installed and authenticate locally
with `hf auth login`. Do not put an access token in source code. From the
ProbeScout repository root, run the following Python in that environment:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Ian100/ProbeScout-probes",
    repo_type="model",
    revision="3d8fbdcc9921c0a3a20e2c8cbd0cd102b124d904",
    local_dir="artifacts/downloads/probes",
)

# Optional: this downloads the full 81.10 GB feature package.
snapshot_download(
    repo_id="Ian100/ProbeScout-features",
    repo_type="dataset",
    revision="3ef237a82d0c738a159aa49bb86764732d331265",
    local_dir="artifacts/downloads/features",
)
```

Merge the downloaded probe package's `visual_analytics/` directory into the code
root, preserving its internal paths. The feature package has its own
`dataset/raw/` tree; use it within the prepared `EXPERIMENT_ROOT` described in
[assets.md](assets.md). Keep the package manifests for verification.

To fetch only one dataset, add an `allow_patterns` list to the feature download,
for example `['dataset/raw/stanford_cars/**', 'asset_manifest.json', 'README.md']`.
Cached ranking display does not require downloading all patch features; feature
extraction and probe training/update have their own input requirements.

The 60.99 GB CelebA patch NPY is stored as 29 byte-range shards. After downloading
all feature files, reconstruct it with the script included in that HF package:

```sh
python artifacts/downloads/features/restore_patch_features.py --root artifacts/downloads/features
```

The script verifies each shard and the reconstructed file. It preserves the
original NPY bytes, dtype, shape and row order, retains the shards, and refuses
to overwrite a differing file. Reconstruction needs another 60.99 GB of free
disk space. Cars and HICO arrays retain their original NPY files.

## Publication selection

The main17 initial snapshot is `f0-val-36-20260908`. Models are selected by the `bankDirectory` in each frozen task manifest. Avoid directory-wide uploads of the research runtime: that tree also contains mutable user state. The exporter uses an explicit file allowlist and uploads directly from the original files. Large features and checkpoints are not copied into the source repository or a duplicate staging directory.

Human feedback is excluded from GitHub **and** Hugging Face: no annotations, user/session databases, labels snapshots, human-refined weights/scores, or case replay bundles. Original VQA supervision for probe training is a separate artifact class; it must not be replaced with later human-edited feedback data.

Every downloadable file should identify its relative path, bytes, SHA-256, artifact type, dataset/gallery identity, feature identity, training/split/calibration provenance, dependency IDs, Hugging Face repository type/ID/revision, and license. Keep data paths separate from public source paths; do not change original artifact hashes to match renamed code.

## What remains for a portable Web system

The separate Main17 Web/evaluation package is prepared locally but has not
been uploaded; there is no published HF link for it yet. It contains the task
catalog and bundles, initial fusion evidence, evaluation inputs, original VQA
labels and query/thumbnail images. Its upload awaits authorization for those
additional assets. The two available packages above do not supply all of this.

The original fixed-VQA Validation documents contain historical feedback exposure rows and overlap statistics. Those legacy documents are excluded from the first evidence package. A separate portable label export retains original VQA source/fit/Validation labels and memberships without feedback history. It does not yet replace the legacy interactive Validation loader. The prepared package has been checked with the frozen offline evaluation commands: all 204 Table 2 rows and 68 Table 5 rows match the prior verified results.

Uploading the evidence alone would not complete the interactive loader
adaptation. The two available HF packages do not yet form an asset-complete
portable Web demo.
