# ProbeScout

Visual analytics for attribute-guided image search. ProbeScout trains eight
probes per attribute on frozen SigLIP features, combines their evidence, and
supports interactive inspection and feedback.

## One-click local setup

On **Windows x64**, download either launcher package and extract it:

| Download | Installed for your selected dataset | From a Git clone |
|---|---|---|
| [Starter](https://github.com/lyfdy456/ProbeScout/releases/download/v0.3.0-archives/ProbeScout-Starter-Windows.zip) | Task package for browsing and Weight Tune / Staged | Double-click `start.bat` |
| [Full](https://github.com/lyfdy456/ProbeScout/releases/download/v0.3.0-archives/ProbeScout-Full-Windows.zip) | Task package + features + trained probes | Double-click `start-full.bat` |

Both ZIPs contain code and a launcher. The selected assets are downloaded from
HF on first launch; Full does not download all three datasets.

1. Download the original image archive for Cars, HICO-DET or CelebA using
   the [dataset guide](docs/manual_setup.md#3-place-original-images-for-image-galleries).
2. Double-click `start.bat` in the extracted package.
3. Choose your dataset, select its image archive and an extraction folder, then
   click **Prepare & open**. An existing image folder can also be used directly.

The launcher installs Python/Node and a CPU environment locally, downloads the
assets, extracts original images, builds thumbnails and opens the Web interface. Later launches reuse the
installation. [Setup details and troubleshooting](docs/launcher.md).

**Original images are required for photograph previews.** HF features let you
skip feature extraction. For other platforms or CUDA training, see
[manual setup](docs/manual_setup.md).

## Choose a workflow

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

[Asset distribution](docs/asset_distribution.md) lists sizes, revisions and subset
download commands. Historical human feedback and sessions are excluded.

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
exclusions. Citation metadata is pending.
