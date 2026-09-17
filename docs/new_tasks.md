# Run a new task

This entry point supports **new query images and 2–5 new binary attributes on
Cars, HICO-DET or CelebA**. It uses the same eight probes, five seeds, fusion and
interactive feedback implementation as the released system. A new dataset with
its own image ordering requires an additional dataset adapter; it is not covered
by these commands.

Original images are required for image visualization. Downloading
[HF features](https://huggingface.co/datasets/Ian100/ProbeScout-features) skips
feature extraction only. Follow [manual setup](manual_setup.md) to install the
selected dataset's images, canonical `records.csv`, global SigLIP embeddings and
patch tokens. You can instead extract both feature files locally. Do not reorder,
subset or rename the records/images when using the supplied HF features.

The published checkpoint banks are **not required** to train a new task from
scratch. The new run produces its own bank. SigLIP's text encoder may be downloaded
on first use; no VQA service or API key is required by this workflow.

## 1. Prepare the task folder

Run commands from the repository root. Start with the template:

```sh
python -c "import shutil; shutil.copytree('examples/custom_task', 'artifacts/my_task')"
```

```text
ProbeScout/
  artifacts/
    my_task/
      task.json
      labels.csv
  dataset/raw/
    stanford_cars/                          # or HICO/ or CelebA/
      images/                              # CelebA uses img_celeba/
      processed/
        records.csv
        siglip_embedding.npy
        siglip_patch_tokens.npy
```

`task.json` has exactly these fields:

```json
{
  "name": "red_convertible",
  "dataset": "cars",
  "query_text": "a red convertible car",
  "query_images": ["000001.jpg", "000002.jpg"],
  "attributes": [
    {"id": "red", "name": "Red body color"},
    {"id": "convertible", "name": "Convertible body type"}
  ],
  "labels": "labels.csv"
}
```

The image IDs above illustrate the format; **select actual matching query
images yourself**. The template leaves `query_images` empty deliberately.
Use exact `relative_path` values from `records.csv`, with forward slashes:

- Cars: `000001.jpg`
- HICO: `train2015/HICO_train2015_00000001.jpg`
- CelebA: `000001.jpg`

Query images must already belong to the selected dataset and its feature cache.
They are excluded from fitting, Val and feedback. This input does not accept an
external query photograph with no corresponding embedding row.

`name` and attribute IDs use lowercase letters, numbers and underscores, starting
with a letter; maximum 48 characters. Attribute IDs and names must be unique.
`joint` and `image_id` are reserved IDs. Attribute names should describe visible
properties clearly; they are also inputs to the text encoder.

## 2. Supply original task supervision

`labels.csv` is UTF-8. Its first column is `image_id`; the remaining columns are
the attribute IDs **in the same order as `task.json`**:

```csv
image_id,red,convertible
```

Add one row for each image you actually labeled. Every attribute cell must be
`0` (absent) or `1` (present). Do not include query images, duplicate IDs, blank
labels or unknown image IDs. The Joint label is computed as the AND of all
attribute labels, so do not add a `joint` column.

Labels can come from your own annotation or from a VQA result you have checked
and converted to this format. The command does not call an annotation API. The
template contains only a CSV header; it supplies no invented labels. Unlisted
images remain unlabeled and can enter the appropriate PU fitting pools; they
are not automatically converted to negative supervision.

Provide both positive and negative examples for every attribute and Joint.
The program sorts by canonical image order and freezes one seed-0, Joint-stratified
80:20 fit/Val split shared by all methods, attributes and seeds. Each attribute
needs at least two positive and two negative **fit** examples and both classes
in **Val**. Preflight stops if these conditions fail. Supply more labeled examples
instead of repeatedly resampling Val to improve results.

## 3. Check, train and export

```sh
uv run --project probe_learning python scripts/custom_task.py --task artifacts/my_task/task.json --check
uv run --project probe_learning python scripts/custom_task.py --task artifacts/my_task/task.json --execute
```

`--check` validates the format, labels, fixed split, original image paths and
feature dimensions. It performs no training and writes no task assets.

`--execute` freezes the input and Val identities, trains all eight probe methods
with seeds 0–4, verifies the checkpoint scores, constructs initial fusion, creates
thumbnails/PCA/UMAP/clusters, then registers the task in the local Web catalog.
The default is 100 training epochs. For a pipeline check, use `--epochs 1`; that
is a smoke run, **not a trained result for scientific comparison**.

For these new tasks, initial gate calibration uses fit labels only. Val selects
probe checkpoints and supports subsequent refinement evaluation. This is a new
local run, not a reproduction of the paper's pinned, Val-calibrated Main17 F0.
Query and Val rows are excluded from probe fitting and normalization.

Generated files stay in ignored local directories:

- `visual_analytics/pcp_analyze/runtime/local-tasks/<task-id>/`: input snapshot,
  local catalog and completion record.
- `.../runtime/evaluation/vqa-validation/<version>/`: frozen supervised holdout
  (the directory name is retained by the shared backend; your CSV is the label source).
- `.../runtime/isolated-probes/<task-id>/<bank-id>/`: checkpoints and score caches.
- `.../runtime/unified-initial/<version>/`: verified initial fusion.
- `visual_analytics/pcp_analyze/web/public/data/local/<task-id>/`: browser assets.

The task ID includes an input/epoch fingerprint. Changing query images,
attributes, labels or epochs creates a separate task. A completed identical run
can be enabled again with the same command without retraining. Interrupted
training can reuse verified completed cache entries. No upload is performed.

## 4. Open the task

```sh
cd visual_analytics/pcp_analyze/web
npm run dev
```

Open **http://localhost:3000** and choose the task marked `(local)`. Restart the
app after registering a task. It supports image galleries, visual projections,
rank comparison, new feedback, Weight Tune and Update Probes using its own bank.

Your CSV provides training supervision and its held-out Val; it does not provide
an independent full-gallery ground truth. Therefore this workflow exposes
**Development and Val diagnostics, with no Frozen Test view or Test metrics**.
No unknown labels are filled with zeros. Main17's separate Frozen Test view is
unchanged, and Development continues to exclude Test metrics.

`prepare_web.py --dataset ...` enables exactly the selected published task ZIPs.
To add a completed local task back afterward, rerun its identical `--execute`
command, then restart the app. Paper task definitions and global paper F0/Val
pointers are not changed by a local task run.
