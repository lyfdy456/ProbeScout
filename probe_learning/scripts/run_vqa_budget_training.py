"""Train/evaluate the fixed paper suite for materialized VQA-budget prefixes.

The controller is intentionally strict about resume.  A budget is reusable only
when its materialized prefix, suite file, manifest, seeds, epoch count, clean
Frozen-Test audit, and Ours-Full Test/Joint/AP rows all match the current
request.  A stale pilot ``summary.json`` is therefore never accepted merely
because it says ``completed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
BATCH_RUNNER = Path(__file__).with_name("run_probebank_batch.py")
DEFAULT_BUDGETS = (50, 100, 150, 200, 300, 500)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_INITIAL_BUDGET = 100
EXPECTED_ROUND_BUDGET = 50
EXPECTED_TRAIN_PER_ROUND = 40
EXPECTED_AUDIT_PER_ROUND = 10
CONTRACT_NAME = "budget_training_contract.json"


class TrainingContractError(RuntimeError):
    """Raised when an input cannot prove the fixed budget-training contract."""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TrainingContractError(f"missing {label}: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError(f"invalid {label}: {path}: {error}") from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TrainingContractError(f"cannot hash required file {path}: {error}") from error
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def task_id(task: Mapping[str, Any]) -> str:
    return f"{int(task['order']):03d}_{task['dataset']}_{task['task']}"


def expected_prefix_counts(budget: int) -> tuple[int, int, int]:
    """Return (source round, train labels, permanent-audit labels)."""

    if budget == 50:
        return 0, 50, 0
    if budget < EXPECTED_INITIAL_BUDGET:
        raise TrainingContractError(f"unsupported logical budget: {budget}")
    delta = budget - EXPECTED_INITIAL_BUDGET
    if delta % EXPECTED_ROUND_BUDGET:
        raise TrainingContractError(
            f"budget {budget} is not on the 100 + n*50 acquisition schedule"
        )
    source_round = delta // EXPECTED_ROUND_BUDGET
    return (
        source_round,
        EXPECTED_INITIAL_BUDGET + source_round * EXPECTED_TRAIN_PER_ROUND,
        source_round * EXPECTED_AUDIT_PER_ROUND,
    )


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(path, label="task manifest")
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise TrainingContractError("task manifest must contain a tasks list")
    tasks = list(payload["tasks"])
    if not tasks:
        raise TrainingContractError("task manifest contains no tasks")
    declared = payload.get("task_count")
    if declared is not None and int(declared) != len(tasks):
        raise TrainingContractError(
            f"task_count mismatch: declared {declared}, found {len(tasks)}"
        )
    orders: set[int] = set()
    identities: set[tuple[str, str]] = set()
    for position, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise TrainingContractError(f"tasks[{position}] must be an object")
        try:
            order = int(task["order"])
            dataset = str(task["dataset"])
            name = str(task["task"])
        except (KeyError, TypeError, ValueError) as error:
            raise TrainingContractError(
                f"tasks[{position}] requires order, dataset, and task"
            ) from error
        if order in orders or (dataset, name) in identities:
            raise TrainingContractError(
                f"duplicate manifest task identity: order={order}, {dataset}/{name}"
            )
        orders.add(order)
        identities.add((dataset, name))
    return payload, tasks


def resolve_ours_full_method(suite: Mapping[str, Any]) -> str:
    """Resolve the concrete emitted method ID for the suite's Ours-Full recipe."""

    for key in ("ours_full_method_id", "ours_full_method", "primary_fusion_method_id"):
        value = suite.get(key)
        if isinstance(value, str) and value:
            return value

    candidates: list[str] = []
    recipes = suite.get("fusion_recipes", [])
    if not isinstance(recipes, list):
        raise TrainingContractError("suite fusion_recipes must be a list")
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        recipe_id = str(recipe.get("id", ""))
        if "ours_full" not in recipe_id.lower() and "ours-full" not in recipe_id.lower():
            continue
        prefix = recipe.get("id_prefix")
        gates = recipe.get("gate_strategies", [])
        rules = recipe.get("joint_rules", [])
        if not isinstance(prefix, str) or not prefix:
            continue
        if not isinstance(gates, list) or not isinstance(rules, list):
            continue
        candidates.extend(
            f"{prefix}_{gate}_{rule}"
            for gate in gates for rule in rules
            if isinstance(gate, str) and gate and isinstance(rule, str) and rule
        )
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise TrainingContractError(
            "suite must resolve exactly one Ours-Full fusion method; "
            f"found {candidates or 'none'}"
        )
    return candidates[0]


def validate_prefix(
    run_root: Path,
    task: Mapping[str, Any],
    budget: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    split_root = run_root / "tasks" / task_id(task) / "vqa" / "iterative" / "split"
    selection_path = split_root / "train_labeled_indices.json"
    provenance_path = split_root / "budget_prefix_provenance.json"
    selected = read_json(selection_path, label="materialized train prefix")
    provenance = read_json(provenance_path, label="budget-prefix provenance")
    if (
        not isinstance(selected, list)
        or not all(isinstance(item, str) and item for item in selected)
    ):
        raise TrainingContractError(f"train prefix must be a list of image IDs: {selection_path}")
    if len(selected) != len(set(selected)):
        raise TrainingContractError(f"train prefix contains duplicates: {selection_path}")
    if not isinstance(provenance, dict):
        raise TrainingContractError(f"budget-prefix provenance must be an object: {provenance_path}")

    expected_round, expected_train, expected_audit = expected_prefix_counts(budget)
    materialized = provenance.get("materialized")
    source = provenance.get("source")
    identity = provenance.get("task")
    if not isinstance(materialized, dict) or not isinstance(source, dict):
        raise TrainingContractError(f"incomplete budget-prefix provenance: {provenance_path}")
    if not isinstance(identity, dict) or (
        int(identity.get("order", -1)) != int(task["order"])
        or identity.get("dataset") != task["dataset"]
        or identity.get("task") != task["task"]
        or identity.get("output_id") != task_id(task)
    ):
        raise TrainingContractError(f"task identity mismatch in {provenance_path}")
    if int(provenance.get("logical_budget", -1)) != budget:
        raise TrainingContractError(f"logical budget mismatch in {provenance_path}")
    actual_hash = sequence_sha256(selected)
    checks = {
        "train_count": (int(materialized.get("train_count", -1)), expected_train),
        "audit_count": (int(materialized.get("audit_count", -1)), expected_audit),
        "observed_total_labeled_count": (
            int(materialized.get("observed_total_labeled_count", -1)), budget,
        ),
        "logical_budget_shortfall": (
            int(materialized.get("logical_budget_shortfall", -1)), 0,
        ),
        "source_round": (int(source.get("source_round", -1)), expected_round),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise TrainingContractError(
                f"{name} mismatch for {task_id(task)}/budget {budget}: "
                f"{actual} != {expected}"
            )
    if len(selected) != expected_train:
        raise TrainingContractError(
            f"train prefix length mismatch for {task_id(task)}/budget {budget}: "
            f"{len(selected)} != {expected_train}"
        )
    if materialized.get("train_selection_sha256") != actual_hash:
        raise TrainingContractError(f"train prefix hash mismatch: {selection_path}")
    if materialized.get("no_duplicate_train_ids") is not True:
        raise TrainingContractError(f"duplicate-free proof is missing: {provenance_path}")
    if materialized.get("train_audit_disjoint") is not True:
        raise TrainingContractError(f"train/audit disjoint proof is missing: {provenance_path}")
    hashes = source.get("cumulative_round_manifest_sha256")
    if not isinstance(hashes, list) or len(hashes) != expected_round + 1:
        raise TrainingContractError(f"source-round hash chain is incomplete: {provenance_path}")
    protocol = source.get("protocol")
    if not isinstance(protocol, dict) or any(
        int(protocol.get(key, -1)) != value
        for key, value in (
            ("initial", EXPECTED_INITIAL_BUDGET),
            ("round", EXPECTED_ROUND_BUDGET),
            ("train", EXPECTED_TRAIN_PER_ROUND),
            ("audit", EXPECTED_AUDIT_PER_ROUND),
        )
    ):
        raise TrainingContractError(f"source protocol mismatch: {provenance_path}")

    record = {
        "order": int(task["order"]),
        "dataset": str(task["dataset"]),
        "task": str(task["task"]),
        "task_id": task_id(task),
        "selection_path": str(selection_path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": file_sha256(provenance_path),
        "acquisition_stage": str(provenance.get("acquisition_stage", "")),
        "logical_budget": budget,
        "train_count": expected_train,
        "audit_count": expected_audit,
        "train_selection_sha256": actual_hash,
    }
    if not record["acquisition_stage"]:
        raise TrainingContractError(f"acquisition stage is missing: {provenance_path}")
    return record, tuple(selected)


def build_budget_contract(
    budget: int,
    *,
    tasks: Sequence[Mapping[str, Any]],
    manifest: Path,
    manifest_payload: Mapping[str, Any],
    suite: Path,
    suite_payload: Mapping[str, Any],
    run_root: Path,
    seeds: Sequence[int],
    epochs: int,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    prefix_records: list[dict[str, Any]] = []
    selections: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        record, selected = validate_prefix(run_root, task, budget)
        prefix_records.append(record)
        selections[record["task_id"]] = selected
    stages = {record["acquisition_stage"] for record in prefix_records}
    if len(stages) != 1:
        raise TrainingContractError(
            f"budget {budget} mixes acquisition stages: {sorted(stages)}"
        )
    contract = {
        "schema_version": 1,
        "budget": budget,
        "manifest": {
            "path": str(manifest.resolve()),
            "sha256": file_sha256(manifest),
            "batch_id": manifest_payload.get("batch_id"),
            "declared_task_count": manifest_payload.get("task_count"),
        },
        "suite": {
            "path": str(suite.resolve()),
            "sha256": file_sha256(suite),
            "suite_id": suite_payload.get("suite_id"),
            "ours_full_method": resolve_ours_full_method(suite_payload),
        },
        "task_count": len(tasks),
        "tasks": prefix_records,
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "stages": ["iterative"],
        "cross_task_probe_reuse": False,
        "metric_contract": {
            "split": "test",
            "target": "joint",
            "metric": "average_precision",
        },
    }
    return contract, selections


def build_grid_contracts(
    budgets: Sequence[int],
    *,
    tasks: Sequence[Mapping[str, Any]],
    manifest: Path,
    manifest_payload: Mapping[str, Any],
    suite: Path,
    suite_payload: Mapping[str, Any],
    experiment_root: Path,
    seeds: Sequence[int],
    epochs: int,
) -> dict[int, dict[str, Any]]:
    contracts: dict[int, dict[str, Any]] = {}
    selections_by_budget: dict[int, dict[str, tuple[str, ...]]] = {}
    for budget in budgets:
        run_root = experiment_root / f"budget_{budget:03d}"
        contract, selections = build_budget_contract(
            budget,
            tasks=tasks,
            manifest=manifest,
            manifest_payload=manifest_payload,
            suite=suite,
            suite_payload=suite_payload,
            run_root=run_root,
            seeds=seeds,
            epochs=epochs,
        )
        contracts[budget] = contract
        selections_by_budget[budget] = selections

    previous_budget: int | None = None
    for budget in budgets:
        if previous_budget is not None:
            for identifier, selected in selections_by_budget[budget].items():
                previous = selections_by_budget[previous_budget][identifier]
                if not set(previous).issubset(selected):
                    raise TrainingContractError(
                        f"prefixes are not nested for {identifier}: "
                        f"budget {previous_budget} !subset budget {budget}"
                    )
        previous_budget = budget
    return contracts


def normalized_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return os.path.normcase(str(Path(value).resolve()))


def audit_budget_completion(
    run_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    contract_path = run_root / CONTRACT_NAME
    try:
        existing_contract = read_json(contract_path, label="budget training contract")
        if existing_contract != contract:
            reasons.append("budget training contract differs from the current request")
    except TrainingContractError as error:
        reasons.append(str(error))

    run_state_path = run_root / "run_state.json"
    try:
        run_state = read_json(run_state_path, label="run state")
        if not isinstance(run_state, dict):
            reasons.append("run state is not an object")
        else:
            if run_state.get("status") != "completed_requested_range":
                reasons.append(f"run state is not completed: {run_state.get('status')!r}")
            if run_state.get("suite_id") != contract["suite"]["suite_id"]:
                reasons.append("run-state suite ID mismatch")
            if normalized_path(run_state.get("suite_config")) != normalized_path(
                contract["suite"]["path"]
            ):
                reasons.append("run-state suite path mismatch")
            if list(run_state.get("seeds", [])) != list(contract["seeds"]):
                reasons.append("run-state seed list mismatch")
            if int(run_state.get("epochs", -1)) != int(contract["epochs"]):
                reasons.append("run-state epoch count mismatch")
            if run_state.get("cross_task_probe_reuse") is not False:
                reasons.append("run-state cross-task reuse was not disabled")
            expected_orders = [int(task["order"]) for task in tasks]
            if list(run_state.get("orders") or []) != expected_orders:
                reasons.append("run-state task-order list mismatch")
    except (TrainingContractError, TypeError, ValueError) as error:
        reasons.append(str(error))

    ours_full = str(contract["suite"]["ours_full_method"])
    expected_seeds = [int(seed) for seed in contract["seeds"]]
    prefix_by_id = {record["task_id"]: record for record in contract["tasks"]}
    task_results: list[dict[str, Any]] = []
    for task in tasks:
        identifier = task_id(task)
        task_reasons: list[str] = []
        task_root = run_root / "tasks" / identifier
        summary_path = task_root / "summary.json"
        metrics_path = task_root / "iterative_metrics.json"
        try:
            summary = read_json(summary_path, label=f"summary for {identifier}")
            metrics = read_json(metrics_path, label=f"metrics for {identifier}")
            if not isinstance(summary, dict) or not isinstance(metrics, list):
                raise TrainingContractError("summary/metrics payload type mismatch")
            if summary.get("stage_status", {}).get("iterative") != "completed":
                task_reasons.append("iterative stage is not completed")
            if summary.get("suite_id") != contract["suite"]["suite_id"]:
                task_reasons.append("summary suite ID mismatch")
            for key, expected in (
                ("order", int(task["order"])),
                ("dataset", task["dataset"]),
                ("task", task["task"]),
            ):
                if summary.get(key) != expected:
                    task_reasons.append(f"summary {key} mismatch")

            audit = summary.get("supervision_audits", {}).get("iterative")
            if not isinstance(audit, dict) or audit.get("status") != "valid":
                task_reasons.append("missing valid iterative supervision audit")
                audit = {}
            prefix = prefix_by_id[identifier]
            expected_count = int(prefix["train_count"])
            if int(audit.get("selected_count_target", -1)) != expected_count:
                task_reasons.append("audited selected-count mismatch")
            if int(audit.get("matched_count", -1)) != expected_count:
                task_reasons.append("audited matched-count mismatch")
            if int(audit.get("missing_count", -1)) != 0:
                task_reasons.append("supervision audit has missing labels")
            if int(audit.get("duplicate_selected_count", -1)) != 0:
                task_reasons.append("supervision audit has duplicate labels")
            if normalized_path(audit.get("selected_manifest")) != normalized_path(
                prefix["selection_path"]
            ):
                task_reasons.append("audited selected-manifest path mismatch")
            boundary = audit.get("evaluation_boundary")
            if not isinstance(boundary, dict):
                task_reasons.append("missing evaluation-boundary audit")
            else:
                if boundary.get("gallery_membership_valid") is not True:
                    task_reasons.append("training supervision is not proven gallery-only")
                if int(boundary.get("frozen_test_overlap_count", -1)) != 0:
                    task_reasons.append("training supervision overlaps Frozen Test")

            supervision_hash = summary.get("supervision_hashes", {}).get("iterative")
            if not isinstance(supervision_hash, str) or not supervision_hash:
                task_reasons.append("missing summary supervision hash")
            elif audit.get("supervision_hash") != supervision_hash:
                task_reasons.append("summary/audit supervision hash mismatch")
            metric_hashes = {
                row.get("supervision_hash") for row in metrics
                if isinstance(row, dict) and row.get("stage") == "iterative"
            }
            if supervision_hash and metric_hashes != {supervision_hash}:
                task_reasons.append("metric supervision hashes are incomplete or inconsistent")

            main_rows = [
                row for row in metrics
                if isinstance(row, dict)
                and row.get("stage") == "iterative"
                and row.get("split") == "test"
                and row.get("method") == ours_full
            ]
            seed_rows: dict[int, list[dict[str, Any]]] = {}
            for row in main_rows:
                try:
                    seed = int(row.get("seed"))
                except (TypeError, ValueError):
                    continue
                seed_rows.setdefault(seed, []).append(row)
            if sorted(seed_rows) != expected_seeds or any(
                len(seed_rows[seed]) != 1 for seed in seed_rows
            ):
                task_reasons.append(
                    f"Ours-Full Test rows do not contain exactly seeds {expected_seeds}"
                )
            else:
                for seed in expected_seeds:
                    joint = seed_rows[seed][0].get("metrics", {}).get("joint", {})
                    value = joint.get("ap") if isinstance(joint, dict) else None
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = math.nan
                    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                        task_reasons.append(
                            f"invalid Ours-Full Test Joint AP for seed {seed}: {value!r}"
                        )
        except (TrainingContractError, TypeError, ValueError) as error:
            task_reasons.append(str(error))
        reasons.extend(f"{identifier}: {reason}" for reason in task_reasons)
        task_results.append({
            "task_id": identifier,
            "complete": not task_reasons,
            "reasons": task_reasons,
        })
    return {
        "complete": not reasons,
        "reason_count": len(reasons),
        "reasons": reasons,
        "tasks": task_results,
    }


def budget_complete(
    run_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> bool:
    return bool(audit_budget_completion(run_root, tasks, contract)["complete"])


def run_budget(
    budget: int,
    *,
    tasks: list[dict[str, Any]],
    manifest: Path,
    suite: Path,
    experiment_root: Path,
    seeds: list[int],
    epochs: int,
    omp_threads: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    run_root = experiment_root / f"budget_{budget:03d}"
    bank_root = experiment_root / "probebank" / f"budget_{budget:03d}"
    before = audit_budget_completion(run_root, tasks, contract)
    if before["complete"]:
        return {
            "budget": budget,
            "status": "reused_completed",
            "run_root": str(run_root),
            "completion_audit": before,
        }

    # Persist the exact request before entering the resumable batch runner.  If
    # the process is interrupted, the next invocation can distinguish the same
    # request from a stale pilot or a newly materialized prefix.
    write_json(run_root / CONTRACT_NAME, contract)
    command = [
        str(Path(sys.executable).resolve()),
        "-u",
        str(BATCH_RUNNER),
        "--manifest",
        str(manifest.resolve()),
        "--suite-config",
        str(suite.resolve()),
        "--orders",
        *[str(int(task["order"])) for task in tasks],
        "--stages",
        "iterative",
        "--label-size",
        str(budget),
        "--sup-size",
        str(budget),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--epochs",
        str(epochs),
        "--disable-cross-task-reuse",
        "--run-root",
        str(run_root.resolve()),
        "--probebank-root",
        str(bank_root.resolve()),
        "--resume",
        "--skip-posthoc",
    ]
    log_root = experiment_root / "training_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"budget_{budget:03d}.log"
    environment = os.environ.copy()
    thread_count = str(omp_threads)
    environment.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
        }
    )
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {command!r}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    after = audit_budget_completion(run_root, tasks, contract)
    status = "completed" if result.returncode == 0 and after["complete"] else "failed"
    return {
        "budget": budget,
        "status": status,
        "returncode": int(result.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "run_root": str(run_root),
        "probebank_root": str(bank_root),
        "log": str(log_path),
        "preexisting_completion_audit": before,
        "completion_audit": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--orders", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument(
        "--allow-partial-grid",
        action="store_true",
        help="allow a non-paper subset of budgets or seeds for a smoke/pilot run",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all prefixes and report completion state without writing or training",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.epochs < 1 or args.omp_threads < 1:
        raise SystemExit("workers, epochs, and omp-threads must be positive")
    budgets = tuple(int(value) for value in args.budgets)
    seeds = tuple(int(value) for value in args.seeds)
    if tuple(sorted(set(budgets))) != budgets:
        raise SystemExit("budgets must be strictly increasing and unique")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise SystemExit("seeds must be non-empty and unique")
    if not args.allow_partial_grid:
        if budgets != DEFAULT_BUDGETS:
            raise SystemExit(
                f"paper grid requires budgets {DEFAULT_BUDGETS}; use --allow-partial-grid for a pilot"
            )
        if seeds != DEFAULT_SEEDS:
            raise SystemExit(
                f"paper grid requires seeds {DEFAULT_SEEDS}; use --allow-partial-grid for a pilot"
            )

    try:
        manifest_payload, tasks = load_manifest(args.manifest)
        suite_payload = read_json(args.suite_config, label="suite config")
        if not isinstance(suite_payload, dict) or not suite_payload.get("suite_id"):
            raise TrainingContractError("suite config must declare suite_id")
        if args.orders:
            requested = set(args.orders)
            tasks = [task for task in tasks if int(task["order"]) in requested]
            missing = requested.difference(int(task["order"]) for task in tasks)
            if missing:
                raise TrainingContractError(f"orders absent from manifest: {sorted(missing)}")
        tasks.sort(key=lambda row: int(row["order"]))
        if not tasks:
            raise TrainingContractError("no tasks selected")
        contracts = build_grid_contracts(
            budgets,
            tasks=tasks,
            manifest=args.manifest,
            manifest_payload=manifest_payload,
            suite=args.suite_config,
            suite_payload=suite_payload,
            experiment_root=args.experiment_root,
            seeds=seeds,
            epochs=args.epochs,
        )
    except TrainingContractError as error:
        raise SystemExit(f"training preflight failed: {error}") from error

    if args.validate_only:
        audits = [
            {
                "budget": budget,
                **audit_budget_completion(
                    args.experiment_root / f"budget_{budget:03d}", tasks, contracts[budget]
                ),
            }
            for budget in budgets
        ]
        print(json.dumps({
            "status": "inputs_valid",
            "task_count": len(tasks),
            "budgets": list(budgets),
            "seeds": list(seeds),
            "ours_full_method": contracts[budgets[0]]["suite"]["ours_full_method"],
            "completion_audits": audits,
        }, ensure_ascii=False, indent=2))
        return

    args.experiment_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_budget,
                budget,
                tasks=tasks,
                manifest=args.manifest,
                suite=args.suite_config,
                experiment_root=args.experiment_root,
                seeds=list(seeds),
                epochs=args.epochs,
                omp_threads=args.omp_threads,
                contract=contracts[budget],
            ): budget
            for budget in budgets
        }
        for future in as_completed(futures):
            budget = futures[future]
            try:
                result = future.result()
            except Exception as error:  # preserve other budget results/state on controller errors
                result = {
                    "budget": budget,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            results.append(result)
            results.sort(key=lambda row: int(row["budget"]))
            write_json(
                args.experiment_root / "training_state.json",
                {
                    "schema_version": 2,
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": file_sha256(args.manifest),
                    "suite_config": str(args.suite_config.resolve()),
                    "suite_config_sha256": file_sha256(args.suite_config),
                    "task_count": len(tasks),
                    "orders": [int(task["order"]) for task in tasks],
                    "budgets": list(budgets),
                    "seeds": list(seeds),
                    "epochs": args.epochs,
                    "ours_full_method": contracts[budgets[0]]["suite"]["ours_full_method"],
                    "results": results,
                },
            )
            print(
                f"[{len(results)}/{len(budgets)}] budget={result['budget']} "
                f"status={result['status']}",
                flush=True,
            )
    failed = [row for row in results if row["status"] == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} budget runs failed")


if __name__ == "__main__":
    main()
