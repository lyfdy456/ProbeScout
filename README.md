# ProbeScout

## One-click start (Windows x64)

1. Download the original image archive for **Cars, HICO-DET or CelebA** from the
   [dataset guide](docs/manual_setup.md#3-place-original-images-for-image-galleries).
   Leave the dataset archive compressed.
2. Download and extract [ProbeScout Starter](https://github.com/lyfdy456/ProbeScout/releases/latest/download/ProbeScout-Starter-Windows.zip),
   or clone this repository, then double-click **`start.bat`**.
3. Select your dataset and its image archive, then click **Prepare & open**.

ProbeScout automatically installs the environment, extracts the images, downloads
the task package and opens the visualization. First launch needs internet access;
later launches reuse the prepared files.

[Setup details and troubleshooting](docs/launcher.md)

## Starter or Full

| Version | Downloaded on first launch | From a Git clone |
|---|---|---|
| [Starter](https://github.com/lyfdy456/ProbeScout/releases/latest/download/ProbeScout-Starter-Windows.zip) | Task package for browsing and Weight Tune / Staged | `start.bat` |
| [Full](https://github.com/lyfdy456/ProbeScout/releases/latest/download/ProbeScout-Full-Windows.zip) | Task package + features + trained probes | `start-full.bat` |

Both ZIPs contain code and a launcher; assets are downloaded for the selected
dataset. You can also switch versions on the setup page. Original images provide
the gallery photographs; HF features let you skip feature extraction.

The default extraction folder is `dataset/imported/`. You can choose another
drive, or use an existing image folder. For other platforms or CUDA training,
see [manual setup](docs/manual_setup.md).

## Choose a workflow

ProbeScout combines eight attribute probes on frozen SigLIP features for image
search, visual inspection and interactive feedback.

| Goal | Guide | Additional inputs |
|---|---|---|
| Browse rankings and use paper feedback | [One-click setup](docs/launcher.md) | Original images |
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

[Asset distribution](docs/asset_distribution.md) lists sizes, revisions and manual
download commands.

## Layout

| Directory | Purpose |
|---|---|
| `scripts/` | Local launcher, Web preparation, training and evaluation commands |
| `configs/` | Paper tasks, probe suite, training and ablation settings |
| `probe_learning/` | Feature extraction, acquisition, probes and evaluation |
| `attribute_annotation/` | Attribute TXT extraction and VQA JSONL labeling |
| `visual_analytics/pcp_analyze/web/` | Fusion, feedback, API and linked views |
| `manifests/` | Task scope and asset identities |

See [release validation](docs/validation.md) for checks and tested environments.
`python scripts/check_release.py` checks source files, configuration and publication
exclusions.
