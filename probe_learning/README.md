# Probe learning

Feature extraction, VQA acquisition, eight probes and retrieval evaluation.
See [workflow](../docs/workflow.md) for inputs and
[paper-to-code map](../docs/paper_method.md) for the method definitions.

After preparing a dataset's task ZIP and features, run from the repository root:

```sh
python scripts/train_probes.py --list
uv run --project probe_learning python scripts/train_probes.py --task-id 001_cars_task_bmw_convertible
```

Add `--execute` to train. `src/methods/catalog.py` defines the eight probe IDs;
this module's `scripts/` contains their shared utilities.

To run the core learning tests:

```sh
cd probe_learning
uv run python -B -m unittest discover -s tests
```
