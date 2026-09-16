# Probe learning

Frozen features, VQA acquisition, eight paper probes, cache validation and
evaluation. Start with the [workflow](../docs/workflow.md) and
[paper-to-code map](../docs/paper_method.md).

From the repository root:

```sh
python scripts/train_probes.py --list
uv run --project probe_learning python scripts/train_probes.py --directory /path/to/frozen-val-contracts
```

Add `--execute` to train with prepared assets. Public commands are in the root
`scripts/`; this module's `scripts/` contains shared utilities.
`src/methods/catalog.py` defines the eight probe IDs.

```sh
cd probe_learning
uv run python -B -m unittest discover -s tests
```

The seven core test files use synthetic inputs and temporary files. They cover
the eight probes, checkpoint loading, holdout isolation, gate calibration,
supervision merging, paper commands and publication exclusions. See
[validation](../docs/validation.md) for scope and results.
