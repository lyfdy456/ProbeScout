# ProbeScout

Visual analytics for attribute-guided image search with reusable attribute probes.
ProbeScout learns eight probes per attribute from frozen features, combines their
evidence into a conjunction-aware ranking, and supports interactive refinement.

## Start here

- [Workflow](docs/workflow.md): features, training/loading, fusion and feedback.
- [Paper-to-code map](docs/paper_method.md): eight probes, Main17, Tables 2 and 5.
- [Assets](docs/assets.md): required inputs, locations and checksums.
- [Validation](docs/validation.md): checks performed on this release.

[Main17](manifests/paper_main17.json) contains 7 Cars, 8 HICO-DET and 2 CelebA
tasks. Initial evidence is pinned to `f0-val-36-20260908`; the historical version
name does not change the 17-task evaluation scope.

## Install

Python 3.11+ and Node.js 22.13+ are required by the project metadata.

```sh
uv sync --project probe_learning --locked
cd visual_analytics/pcp_analyze/web
npm ci
```

The Python lock uses CUDA 12.8 PyTorch wheels. The Web launcher finds
`probe_learning/.venv`; `PROBESCOUT_PYTHON` can select another compatible Python.

## Main commands

Run Python commands from the repository root. Training and evaluation require
the separate [asset packages](docs/assets.md).

```sh
# List tasks, probe IDs and seeds without loading assets.
python scripts/train_probes.py --list

# Check frozen training inputs; add --execute to train.
uv run --project probe_learning python scripts/train_probes.py --directory /path/to/frozen-val-contracts

# Table 2: two embedding baselines, CLAY, eight probes and initial F0.
uv run --project probe_learning python scripts/evaluate_main17.py --clay /path/to/clay-cache --output outputs/table2

# Table 5: the full model and three component-removal variants.
uv run --project probe_learning python scripts/evaluate_ablation.py --assets /path/to/evaluation-assets --output outputs/table5
```

After installing the Web evidence package, run `npm run dev` from
`visual_analytics/pcp_analyze/web`.

## Choose a workflow

| Goal | Entry point | Required assets |
|---|---|---|
| Explore saved rankings and give new feedback | Web interface: `npm run dev` | Web task bundles, initial fusion evidence and compatible Validation contracts; probes/features for probe updates |
| Train probes from existing features | `scripts/train_probes.py` | Downloaded features, task metadata, original VQA labels and frozen Validation contracts |
| Start from original images | Feature extraction, then probe training | Dataset images and ordered records, plus the training inputs above |
| Reproduce paper tables without the interface | `scripts/evaluate_main17.py` and `scripts/evaluate_ablation.py` | Frozen evaluation evidence; CLAY cache for Table 2 |

See the [workflow](docs/workflow.md) for the full sequence. The Web interface
supports a combined development launcher or separate frontend/API processes;
see [launch options](visual_analytics/pcp_analyze/web/README.md#launch-options).
These entry points require the assets listed above; the two available HF packages
do not yet form a complete portable Web demo.

## Layout

| Directory | Purpose |
|---|---|
| `scripts/` | Public paper training, evaluation and release checks |
| `configs/` | Main17, eight-probe suite, training, ablation and CLAY settings |
| `probe_learning/src/methods/` | Probes, checkpoint updates, gate calibration |
| `probe_learning/src/evaluation/` | Shared metrics and component ablations |
| `probe_learning/scripts/` | Features, acquisition and training/cache utilities |
| `attribute_annotation/` | Attribute extraction and VQA labeling |
| `visual_analytics/pcp_analyze/` | One canonical set of score/rank analysis tools |
| `visual_analytics/pcp_analyze/web/` | Fusion, feedback, API and linked views |
| `manifests/` | Paper scope and immutable asset identities |

Historical exploration scripts, unused independent methods and old experiment
configurations were removed. Numerical and provenance checks remain.

## Publication status

Features and trained probes are fully uploaded and verified in private Hugging
Face repositories:

- [Features](https://huggingface.co/datasets/Ian100/ProbeScout-features): 81.10 GB,
  including global features, patch tokens, records and image IDs.
- [Probes and prediction caches](https://huggingface.co/Ian100/ProbeScout-probes):
  5.69 GB, including 1,520 checkpoints and 304 frozen score caches.

See [asset distribution](docs/asset_distribution.md) for pinned revisions and
download commands. Access currently requires an authorized HF account.
The separate Web/evaluation evidence package has not been uploaded. Its portable
VQA labels still require adaptation for the interactive Validation loader;
legacy documents containing historical feedback exposure are excluded.

Historical human feedback, sessions, feedback-derived models and case replay are
excluded. The feedback algorithm/interface remain available for new input.
Original VQA supervision is a separate training asset.

Run `python scripts/check_release.py` to check the source boundary. Asset hosting,
licenses and citation metadata remain to be finalized before publication.
