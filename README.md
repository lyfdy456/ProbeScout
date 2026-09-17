# ProbeScout

Visual analytics for attribute-guided image search. ProbeScout trains eight
probes per attribute on frozen SigLIP features, combines their evidence, and
supports interactive inspection and feedback.

## Start with one dataset

Choose Cars, HICO-DET or CelebA. For the Web interface, download the original
images and the matching [task ZIP](https://huggingface.co/datasets/Ian100/ProbeScout-tasks).
Follow [manual setup](docs/manual_setup.md) for the image layout and installation.

From the repository root, install dependencies (Python 3.11+, Node.js 22.13+):

```sh
uv sync --project probe_learning --locked
cd visual_analytics/pcp_analyze/web
npm ci
cd ../../..
```

After extracting the task ZIP into the repository root and placing the images:

```sh
uv run --project probe_learning python scripts/prepare_web.py --dataset cars
cd visual_analytics/pcp_analyze/web
npm run dev
```

Open http://localhost:3000. The launcher starts both the Web interface and Python
API. [Other launch options](visual_analytics/pcp_analyze/web/README.md#launch-options)
are available.

**Original images are required for photograph previews.** HF embeddings skip
feature extraction. Cached browsing and **Weight Tune / Staged** need only the
images and task ZIP; they do not require the large feature or checkpoint downloads.

## Choose a workflow

| Goal | Guide | Additional inputs |
|---|---|---|
| Browse rankings and use paper feedback | [Manual setup](docs/manual_setup.md) | None beyond images + task ZIP |
| Train a new task | [New tasks](docs/new_tasks.md) | Features, task JSON, per-image labels |
| Extract features or train Main17 probes | [Workflow](docs/workflow.md) | Features and task training inputs |
| Update probe weights with feedback | [Workflow](docs/workflow.md#feedback) | Features and matching checkpoint banks |
| Evaluate paper tables | [Paper-to-code map](docs/paper_method.md) | Separate evaluation inputs described in [assets](docs/assets.md) |

The paper's feedback method is **Weight Tune / Staged**: probes and gates stay
fixed while fusion weights are refined. **Update Probes** is an optional extension
that changes the probes.

[Main17](manifests/paper_main17.json) contains 7 Cars, 8 HICO-DET and 2 CelebA tasks.
The eight probes use five seeds (0–4) and 100 training epochs. See the
[paper-to-code map](docs/paper_method.md) for configurations and table commands.

## Downloads

- [Tasks and Web evidence](https://huggingface.co/datasets/Ian100/ProbeScout-tasks):
  dataset-specific ZIPs containing Main17 definitions, original VQA supervision,
  splits, saved scores and initial fusion.
- [Features](https://huggingface.co/datasets/Ian100/ProbeScout-features): frozen
  global embeddings and patch tokens; download only your selected dataset.
- [Probes](https://huggingface.co/Ian100/ProbeScout-probes): 1,520 checkpoints
  (38 task-attribute records × 8 methods × 5 seeds) and 304 prediction caches.

[Asset distribution](docs/asset_distribution.md) lists sizes, revisions and subset
download commands. Historical human feedback and sessions are excluded.

## Layout

| Directory | Purpose |
|---|---|
| `scripts/` | Training, new tasks, Web preparation and evaluation commands |
| `configs/` | Paper tasks, probe suite, training and ablation settings |
| `probe_learning/` | Feature extraction, acquisition, probes and evaluation |
| `attribute_annotation/` | Attribute TXT extraction and VQA JSONL labeling |
| `visual_analytics/pcp_analyze/web/` | Fusion, feedback, API and linked views |
| `manifests/` | Task scope and asset identities |

See [release validation](docs/validation.md) for checks and tested environments.
`python scripts/check_release.py` checks source files, configuration and publication
exclusions. Citation metadata is pending.
