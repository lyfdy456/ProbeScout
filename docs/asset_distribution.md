# Asset distribution

GitHub contains source, configuration, dependency locks, paper scope and core
tests. Hugging Face contains tasks, frozen Web evidence, features and models.
All three HF repositories are public; download authorization is not required.

| Package | Contents | Size | Immutable identity |
|---|---|---:|---|
| [ProbeScout-tasks](https://huggingface.co/datasets/Ian100/ProbeScout-tasks) | 3 dataset ZIPs: Main17 definitions, original VQA labels, records, splits, Web arrays and initial F0 | 0.85 GB compressed | [task_packages.json](../manifests/task_packages.json) |
| [ProbeScout-probes](https://huggingface.co/Ian100/ProbeScout-probes) | 1,520 checkpoints, 304 prediction caches, training/isolation metadata | 5.69 GB | `3d8fbdcc9921c0a3a20e2c8cbd0cd102b124d904` |
| [ProbeScout-features](https://huggingface.co/datasets/Ian100/ProbeScout-features) | Global SigLIP features, patch tokens, records and image IDs | 81.10 GB | `3ef237a82d0c738a159aa49bb86764732d331265` |

Sizes are decimal. Download only your selected datasets. See
[manual setup](manual_setup.md) for original image layout, ZIP extraction and
launch commands. Original photographs are supplied locally by the user.

## Download tasks

Use the HF Files tab, or run this Python from the repository root in an
environment containing `huggingface_hub`. Change `dataset` as desired:

```python
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

dataset = "cars"  # cars, hico, or celeba
index = json.loads(Path("manifests/task_packages.json").read_text())
hf_hub_download(
    repo_id=index["repoId"], repo_type="dataset", revision=index["revision"],
    filename=index["datasets"][dataset]["archive"], local_dir="artifacts/downloads/tasks",
)
```

```sh
python -m zipfile -e artifacts/downloads/tasks/cars.zip .
```

`prepare_web.py` verifies every extracted file against the source-pinned manifest
before enabling the catalog. The ZIP hash is also recorded in the index.

## Download features and probes

These are optional for cached Web browsing/Weight Tune. They are needed for
native probe updates, and features are needed for training from scratch.
For a **Cars-only** setup, run from the repository root:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download

dataset = "cars"
folder = {"cars": "stanford_cars", "hico": "HICO", "celeba": "CelebA"}[dataset]
snapshot_download(
    repo_id="Ian100/ProbeScout-features", repo_type="dataset",
    revision="3ef237a82d0c738a159aa49bb86764732d331265",
    allow_patterns=[f"dataset/raw/{folder}/**", "asset_manifest.json", "restore_patch_features.py"],
    local_dir="artifacts/downloads/features",
)
scope = json.loads(Path("manifests/paper_main17.json").read_text())
task_ids = [t["task_id"] for t in scope["tasks"] if t["dataset"] == dataset]
snapshot_download(
    repo_id="Ian100/ProbeScout-probes", repo_type="model",
    revision="3d8fbdcc9921c0a3a20e2c8cbd0cd102b124d904",
    allow_patterns=[f"visual_analytics/pcp_analyze/runtime/isolated-probes/{tid}/**" for tid in task_ids]
                   + ["asset_manifest.json"],
    local_dir="artifacts/downloads/probes",
)
```

Merge `artifacts/downloads/features/dataset/` into `<repo>/dataset/`, and
`artifacts/downloads/probes/visual_analytics/` into `<repo>/visual_analytics/`.
Preserve internal paths, ordered records and metadata. Keep the downloaded
`asset_manifest.json` files for checking hashes. To select another dataset,
change `dataset`; to fetch the full packages, omit `allow_patterns`.

The 60.99 GB CelebA patch NPY is stored as 29 byte-range shards. After downloading
the CelebA subset, reconstruct it **before merging the dataset tree**:

```sh
python artifacts/downloads/features/restore_patch_features.py --root artifacts/downloads/features
```

The script verifies all shards and the restored NPY, retains the shards and
refuses to overwrite a differing file. Reconstruction requires another 60.99 GB
of disk space. Cars and HICO already contain ordinary NPY files.

## Task provenance and exclusions

The task package loader checks the clean manifest SHA-256, task/record identities,
original VQA source/fit/Val labels, fingerprints and protected partitions.
Original Val hashes are explicitly retained as **provenance**, while the clean
files have their own verified hashes. This preserves compatibility with the
published probe banks without exporting historical feedback exposure records.

The initial F0 version is `f0-val-36-20260908`; Main17 still selects exactly 17
tasks. Checkpoints remain immutable. A new training run has a new identity and
does not replace the published initial ranking automatically.

Historical human annotations, sessions, snapshots, refined models/scores and case
replay are excluded from both GitHub and HF. Task ZIPs also exclude photographs
and thumbnails, which are generated locally. Original dataset terms apply.

The separate CLAY/Table 5 evaluation packages are not included in these Web task
ZIPs. They are needed only for the dedicated full-paper reproduction commands,
not for launching the interface or using new feedback.
