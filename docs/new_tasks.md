# Run a new task

This entry supports Cars, HICO-DET and CelebA, with 2–5 binary attributes and
query images from the selected dataset. Follow [manual setup](manual_setup.md)
to install its images, ordered `records.csv`, global embeddings and patch tokens.
Use HF features or extract them locally. New tasks train their own probe bank.

## 1. Define the task

Run commands from the repository root:

```sh
python -c "import shutil; shutil.copytree('examples/custom_task', 'artifacts/my_task')"
```

Edit `artifacts/my_task/task.json`:

```json
{
  "name": "red_convertible",
  "dataset": "cars",
  "query_images": ["000001.jpg", "000002.jpg"],
  "attributes": [
    {"id": "red", "name": "Red body color"},
    {"id": "convertible", "name": "Convertible body type"}
  ],
  "labels": "labels.csv"
}
```

Choose actual matching query images; these IDs illustrate the format. Use exact
`relative_path` values from `records.csv`:

- Cars and CelebA: `000001.jpg`
- HICO: `train2015/HICO_train2015_00000001.jpg`

Query images are excluded from fitting, Val and feedback. This entry does not yet
support external query photographs or text-only queries.

Task and attribute IDs use lowercase letters, numbers and underscores, start with
a letter, and have at most 48 characters. Attribute names describe the visual
conditions and supply the text encoder's prompts. The interface builds its query
caption from these names.

## 2. Prepare image labels

The original annotation workflow uses **`attributes.txt` for attribute names** and
saves **VQA responses as JSONL**. The new-task command accepts **`labels.csv` for
per-image binary supervision**. These files serve different purposes; see
[annotation formats and commands](../attribute_annotation/README.md).

Keep `labels.csv` next to `task.json`, encoded as UTF-8. Its columns are `image_id`
followed by the attribute IDs in the same order as the JSON:

```csv
image_id,red,convertible
```

Add one row for each labeled image, with `0` or `1` for every attribute. Joint is
the AND of those labels and is computed automatically. Omit query images and
images you have not labeled. IDs must match `records.csv` and appear only once.

Supply positive and negative examples for each attribute and Joint. The command
freezes a seed-0, Joint-stratified 80:20 fit/Val split shared by all methods and
seeds. Each attribute needs at least two examples of each class in fit and both
classes in Val; preflight reports insufficient supervision.

## 3. Train and export

```sh
uv run --project probe_learning python scripts/custom_task.py --task artifacts/my_task/task.json --check
uv run --project probe_learning python scripts/custom_task.py --task artifacts/my_task/task.json --execute
```

`--check` validates inputs. `--execute` trains eight probe methods with seeds 0–4,
selects checkpoints on fixed Val, and exports scores, thumbnails, projections and
clusters. The default is 100 epochs; `--epochs 1` is useful for checking the pipeline.

Initial fusion follows the paper's Appendix C.4. Starting from the fitted reference
gates, select a task-wide threshold shift from `{0, ±0.05, ±0.10}` and temperature
multiplier from `{1, 1.5, 2, 3}` by **fixed-Val Joint AP**. Thresholds are clipped to
`[0,1]`, temperatures capped at `0.30`; ties favor the least changed candidate.
The F1 cutoff is also selected on Val. Probe weights are uniform, `gamma=1`,
`eta=(0.5,0.5)` and `lambda=0.25`. Val and query images stay out of fitting and
normalization.

Results are stored under `visual_analytics/pcp_analyze/runtime/`, with browser
assets in `web/public/data/local/<task-id>/`. The task identity includes its inputs,
protocol and epochs. An identical completed run can be enabled again without
retraining.

## 4. Open the task

```sh
cd visual_analytics/pcp_analyze/web
npm run dev
```

Open http://localhost:3000 and select the task marked `(local)`. Restart the app
after registration. **Weight Tune / Staged** is the paper feedback method: it
keeps probes and gates frozen. **Update Probes** is an optional extension.

The task provides Development browsing and held-out Val diagnostics. An independent
Test set is not supplied by this CSV. To re-enable a local task after running
`prepare_web.py` for published tasks, rerun its `--execute` command.
