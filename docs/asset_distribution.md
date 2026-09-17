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

Move the contents of `artifacts/downloads/features/dataset/` into `dataset/`,
and `artifacts/downloads/probes/visual_analytics/` into `visual_analytics/`,
merging matching directories. Moving avoids retaining an extra copy of the large
arrays. Preserve paths and keep each `asset_manifest.json`. Change `dataset` to
select another dataset; omit `allow_patterns` to fetch all datasets.

The 60.99 GB CelebA patch NPY is stored as 29 byte-range shards. After downloading
the CelebA subset, reconstruct it **before merging the dataset tree**:

```sh
python artifacts/downloads/features/restore_patch_features.py --root artifacts/downloads/features
```

The script verifies all shards and the restored NPY, retains the shards and
refuses to overwrite a differing file. Reconstruction requires another 60.99 GB
of disk space. Cars and HICO already contain ordinary NPY files.

## Scope

Task packages contain Main17 definitions, original VQA supervision, splits and
initial evidence. The F0 identifier `f0-val-36-20260908` is historical; the public
catalog selects 17 tasks. Manifests retain training provenance and payload hashes.

Original photographs, historical human feedback, sessions and feedback-derived
models are excluded. Original dataset terms apply. Separate CLAY/Table 5 inputs
are described in [assets](assets.md); they are not part of the task ZIPs.
