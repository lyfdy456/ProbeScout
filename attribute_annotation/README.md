# Attribute extraction and VQA labeling

The original workflow uses three different formats:

| File | Role |
|---|---|
| `attributes.txt` | The attributes to ask about, for example `{Red body color, Convertible body type}` |
| `*_results.jsonl` | One VQA response per image, including its attribute judgments |
| `records.csv` | Image IDs and their embedding row order |

The new `custom_task.py` entry additionally uses `labels.csv`: one image ID and
one 0/1 column per attribute. It is a compact training input, not a replacement
for the attribute TXT file.

## Attribute names

Run from the repository root. Put selected reference images in
`artifacts/my_task/query_pics/`, then set `DASHSCOPE_API_KEY` in your environment
and run:

```sh
uv run --project probe_learning python attribute_annotation/extract_attributes.py
```

This writes descriptions, proposed attributes and `attributes.txt` to
`artifacts/my_task/qa/`. Review the TXT and use the same attribute names in the
new task JSON. The extraction settings are in `config_attributes.yaml`.

## Per-image judgments

Place the images you want to label in `artifacts/my_task/candidates/` and run:

```sh
uv run --project probe_learning python attribute_annotation/vqa_label.py --attributes-file artifacts/my_task/qa/attributes.txt --dry-run
uv run --project probe_learning python attribute_annotation/vqa_label.py --attributes-file artifacts/my_task/qa/attributes.txt
```

The first command lists the pending images; the second calls Qwen-VL-Max and
writes `*_results.jsonl` under `artifacts/my_task/qa/`. For an existing dataset
task, `--task cars/task_bmw_convertible` loads its `attributes.txt` and writes to
that task's `qa/` directory. Use `--images-manifest` to preserve full image IDs
such as HICO's `train2015/...`; its JSON rows contain `image_id` and `image_path`.

Configuration paths are relative to the repository root. CLI path overrides are
relative to your current working directory. `--config` selects another YAML.

## Use the labels in a new task

For each successful JSONL row, parse the JSON text in its `answer` field and read
`attributes[attribute_name].present`. Place that 0/1 value in the matching column
of `labels.csv`.
Use the response's `image` as `image_id`, aligned to `records.csv`. Include only
complete binary judgments; resolve failed or uncertain answers before training.
The [new-task guide](../docs/new_tasks.md#2-prepare-image-labels) defines the CSV.

The paper's iterative acquisition adds separate fit/audit batches; its entry is
`probe_learning/scripts/iterative_vqa_runner.py`. Audit responses are evaluation
records and must be excluded from training labels. The commands above label an
explicit image selection; they do not run that acquisition schedule.
