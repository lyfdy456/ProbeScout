# Run one dataset locally

Windows users can use [one-click setup](launcher.md) to install dependencies and
download the selected assets automatically. The commands below are the manual
route, including the CUDA training environment.

Choose Cars, HICO-DET or CelebA. Download its original images and task ZIP,
then follow the paths and commands below. Saved rankings and Weight Tune / Staged
need no feature or checkpoint download. Training and Update Probes use the
optional assets in step 5.

## 1. Install the code dependencies

From the cloned repository root:

```sh
uv sync --project probe_learning --locked
cd visual_analytics/pcp_analyze/web
npm ci
cd ../../..
```

Python 3.11+ and Node.js 22.13+ are required. The Python lock uses CUDA 12.8
PyTorch wheels; `PROBESCOUT_PYTHON` can select another compatible environment.

## 2. Download the task package

Open [ProbeScout-tasks](https://huggingface.co/datasets/Ian100/ProbeScout-tasks/tree/main)
and download `cars.zip` (7 tasks, 94 MB), `hico.zip` (8 tasks, 320 MB), or
`celeba.zip` (2 tasks, 439 MB). Revisions and checksums are pinned in
[task_packages.json](../manifests/task_packages.json).

Extract into the **repository root**, preserving the paths inside the ZIP.
For example, put `cars.zip` in `artifacts/downloads/` and run:

```sh
python -m zipfile -e artifacts/downloads/cars.zip .
```

Each ZIP supplies `dataset/tasks/<dataset>/`, ordered `records.csv`, frozen Web
arrays and initial F0 evidence under `visual_analytics/`. Shared files are identical across packages, so the ZIPs can be merged into
a fresh release clone.

## 3. Place original images for image galleries

The [one-click launcher](launcher.md) can extract the original image archive and
recognize these paths automatically. The layout below is for manual preparation.

Download the matching original dataset and arrange its images as follows:

```text
ProbeScout/
  dataset/
    raw/
      stanford_cars/
        images/000001.jpg ... 016185.jpg
        processed/records.csv                 # supplied by cars.zip
      HICO/
        images/train2015/HICO_train2015_00000001.jpg
        images/test2015/HICO_test2015_00000001.jpg
        processed/records.csv                 # supplied by hico.zip
      CelebA/
        img_celeba/000001.jpg ... 202599.jpg
        processed/records.csv                 # supplied by celeba.zip
    tasks/
      cars/<task-name>/task.json
      cars/<task-name>/query_ids.json
      cars/<task-name>/attributes.txt
      cars/<task-name>/original_vqa.json
      cars/<task-name>/isolation.json
      cars/manifest.json
      cars/catalog.json
      hico/...                                # if installed
      celeba/...                              # if installed
```

- **Cars:** use the original combined `car_ims.tgz` image set, with six-digit
  filenames. Place the *contents* of `car_ims/` directly in `images/`. The
  separately numbered `cars_train/` and `cars_test/` layout is not interchangeable.
  The original [Stanford Cars page](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
  may be unavailable; any mirror must preserve the original names and image bytes.
- **HICO-DET:** use the detection image set from the
  [official HICO/HICO-DET site](https://umich-ywchao-hico.github.io/).
  Copy its `images/train2015/` and `images/test2015/` into `HICO/images/`.
- **CelebA:** use **In-The-Wild Images**, `img_celeba`, from the
  [official CelebA page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
  Aligned/cropped images are different inputs, even when filenames match.

Keep the supplied record order. The preparation command checks all expected
image paths and exact query-image checksums, then generates local thumbnails.
Follow the original datasets' terms.

## 4. Prepare and launch

From the repository root, for Cars:

```sh
uv run --project probe_learning python scripts/prepare_web.py --dataset cars
cd visual_analytics/pcp_analyze/web
npm run dev
```

Open **http://localhost:3000**. The launcher starts the Python API too.
Use `--dataset hico`, `--dataset celeba`, or `--dataset cars hico celeba` to enable
exactly those installed datasets. Rerun preparation and restart the app when
changing the selection. `--check-only` verifies inputs without generating files;
`--force-atlases` regenerates thumbnails after replacing images.

The interface provides rankings, scatterplots, parallel coordinates and feedback.
**Weight Tune / Staged** is the paper method: probes and gates remain frozen.
Development shows Val metrics; Frozen Test has its own view.

## 5. Use HF embeddings or extract your own

Training new probes and **Update Probes** require global SigLIP embeddings and
patch tokens. Update Probes is an optional extension that changes the probes. Download only your selected dataset from
[ProbeScout-features](https://huggingface.co/datasets/Ian100/ProbeScout-features),
then merge its `dataset/` tree into the repository root. See the exact
[subset download commands](asset_distribution.md#download-features-and-probes).
For CelebA, reconstruct the sharded patch file with the package's restore script.

Alternatively, after installing images and the task ZIP, extract features:

```sh
uv run --project probe_learning python probe_learning/scripts/precompute_backbone_embeddings.py --dataset cars --backbone siglip
uv run --project probe_learning python probe_learning/scripts/precompute_backbone_patch_tokens.py --dataset cars --backbone siglip
```

Replace `cars` as needed. Outputs are `siglip_embedding.npy` and
`siglip_patch_tokens.npy` in that dataset's `processed/` directory. The first run
downloads the SigLIP backbone if it is not cached. Retain extraction metadata.

For **Update Probes**, also install the selected tasks' pretrained banks from
[ProbeScout-probes](https://huggingface.co/Ian100/ProbeScout-probes), merging its
`visual_analytics/` tree into the repository root. Attribute text encoding may
download SigLIP once if no text-query/model cache exists locally.

For training from scratch, after preparing the selected catalog:

```sh
python scripts/train_probes.py --list
uv run --project probe_learning python scripts/train_probes.py --task-id 001_cars_task_bmw_convertible
uv run --project probe_learning python scripts/train_probes.py --task-id 001_cars_task_bmw_convertible --execute
```

Repeat `--task-id` to select multiple installed tasks. Without `--execute`, this
validates labels and partitions only. Training produces a new bank and does not
replace the published initial Web ranking automatically; see [workflow](workflow.md).

For numerical use without photograph previews, prepare with `--without-images`.

## 6. Try your own task

Follow [New tasks](new_tasks.md) to define query images and attributes, supply
labels, train and export a task to the Web interface.
