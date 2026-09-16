#!/usr/bin/env python3
"""Materialize deterministic, nested training prefixes from iterative VQA rounds.

This is deliberately a selection-only utility.  It reads a task manifest plus
the acquisition stage's protocol, round state, and round manifests.  It never
loads task configuration, VQA answers, embeddings, or ground truth.

The historical 100/50 iterative protocol has no native 50-label checkpoint.
For that one point we take a deterministic stratified subset of round 0:
35 ``query_near``, 8 ``attribute_text_coverage``, and 7 ``mmr_diversity``.
Missing role entries are filled in round-0 training order.  Budgets 100 and
above use the cumulative training selection at the corresponding source round.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_BUDGETS = (50, 100, 150, 200, 300, 500)
EXPECTED_INITIAL_BUDGET = 100
EXPECTED_ROUND_BUDGET = 50
ROUND0_ROLE_QUOTAS = (
    ("query_near", 35),
    ("attribute_text_coverage", 8),
    ("mmr_diversity", 7),
)
TERMINAL_STATES = {"completed", "stopped_audit_budget_policy", "stopped_empty_pool"}


class MaterializationError(RuntimeError):
    """Raised when source selections cannot prove the requested budget contract."""


@dataclass(frozen=True)
class SourceLayout:
    kind: str
    source_root: Path
    round_root: Path
    root_manifest: Path
    round_state: Path | None


@dataclass(frozen=True)
class RoundSelection:
    index: int
    manifest_path: Path
    manifest_sha256: str
    train: tuple[str, ...]
    audit: tuple[str, ...]
    roles: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PendingWrite:
    path: Path
    payload: Any


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise MaterializationError(f"required JSON file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise MaterializationError(f"invalid JSON in {path}: {error}") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _require_string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise MaterializationError(f"{label} must be a JSON list of non-empty strings")
    if len(value) != len(set(value)):
        raise MaterializationError(f"{label} contains duplicate image IDs")
    return list(value)


def _append_disjoint(target: list[str], seen: set[str], values: Sequence[str], *, label: str) -> None:
    overlap = seen.intersection(values)
    if overlap:
        preview = sorted(overlap)[:3]
        raise MaterializationError(f"{label} reuses earlier train/audit IDs: {preview}")
    target.extend(values)
    seen.update(values)


def _resolve_relative(path_value: str, *, base: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _task_root(task: dict[str, Any], *, manifest_path: Path, repo_root: Path) -> Path:
    override = task.get("task_root")
    if override is not None:
        if not isinstance(override, str) or not override:
            raise MaterializationError("task_root override must be a non-empty string")
        root = _resolve_relative(override, base=manifest_path.parent)
    else:
        root = repo_root / "dataset" / "tasks" / str(task["dataset"]) / str(task["task"])
    if not root.is_dir():
        raise MaterializationError(
            f"task root is missing for {task['dataset']}/{task['task']}: {root}"
        )
    return root.resolve()


def _layout_from_override(root: Path, stage: str) -> SourceLayout:
    if (root / "qa" / "round_00" / "manifest.json").is_file():
        state = root / "split" / "round_state.json"
        return SourceLayout(
            "isolated-run",
            root,
            root / "qa",
            root / "qa" / "manifest.json",
            state if state.is_file() else None,
        )
    if (root / "round_00" / "manifest.json").is_file():
        state_candidates = (root / "split" / "round_state.json", root / "round_state.json")
        state = next((path for path in state_candidates if path.is_file()), None)
        return SourceLayout(
            "stage-snapshot", root, root, root / "manifest.json", state,
        )
    raise MaterializationError(
        f"acquisition_root does not contain round_00 for stage {stage!r}: {root}"
    )


def _resolve_source_layout(
    task: dict[str, Any], *, task_root: Path, manifest_path: Path, stage: str
) -> SourceLayout:
    override = task.get("acquisition_root") or task.get("iterative_run_root")
    if override is not None:
        if not isinstance(override, str) or not override:
            raise MaterializationError("acquisition_root override must be a non-empty string")
        return _layout_from_override(
            _resolve_relative(override, base=manifest_path.parent), stage,
        )

    native_round = task_root / "qa" / stage / "round_00" / "manifest.json"
    native_state = task_root / "split" / stage / "round_state.json"
    if native_round.is_file():
        return SourceLayout(
            "task-native",
            task_root,
            task_root / "qa" / stage,
            task_root / "qa" / stage / "manifest.json",
            native_state if native_state.is_file() else None,
        )

    snapshot = task_root / "supervision" / stage
    if (snapshot / "round_00" / "manifest.json").is_file():
        state_candidates = (
            snapshot / "split" / "round_state.json",
            snapshot / "round_state.json",
        )
        state = next((path for path in state_candidates if path.is_file()), None)
        return SourceLayout(
            "supervision-snapshot", snapshot, snapshot, snapshot / "manifest.json", state,
        )

    raise MaterializationError(
        f"cannot find acquisition stage {stage!r} under task root {task_root}"
    )


def _protocol_from_sources(
    layout: SourceLayout, state: dict[str, Any] | None, *, stage: str, max_budget: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not layout.root_manifest.is_file():
        raise MaterializationError(f"acquisition root manifest is missing: {layout.root_manifest}")
    root_manifest = _read_json(layout.root_manifest)
    if not isinstance(root_manifest, dict):
        raise MaterializationError(f"root manifest must be a JSON object: {layout.root_manifest}")
    manifest_stage = root_manifest.get("stage", root_manifest.get("version"))
    if manifest_stage is not None and manifest_stage != stage:
        raise MaterializationError(
            f"root manifest stage mismatch: {manifest_stage!r} != {stage!r}"
        )
    budget = root_manifest.get("budget")
    if not isinstance(budget, dict):
        raise MaterializationError(f"root manifest has no budget protocol: {layout.root_manifest}")
    initial = int(budget.get("initial", -1))
    increment = int(budget.get("round", -1))
    if (initial, increment) != (EXPECTED_INITIAL_BUDGET, EXPECTED_ROUND_BUDGET):
        raise MaterializationError(
            "budget-prefix materialization requires the historical 100/50 protocol; "
            f"found {initial}/{increment} in {layout.root_manifest}"
        )
    declared_max = budget.get("max")
    declared_max_int = int(declared_max) if declared_max is not None else None
    if declared_max_int is not None and declared_max_int < max_budget:
        raise MaterializationError(
            f"stage declares max budget {declared_max}, below requested {max_budget}"
        )

    if state is not None:
        state_stage = state.get("stage")
        if state_stage is not None and state_stage != stage:
            raise MaterializationError(
                f"round_state stage mismatch: {state_stage!r} != {stage!r}"
            )
        status = state.get("status")
        # A still-running acquisition may safely materialize an already
        # committed prefix.  _load_rounds below proves that every required
        # round has a complete manifest and matches round_state.  The full
        # declared trajectory remains terminal-only, so an in-progress state
        # can never masquerade as a completed budget-500 acquisition.
        requests_full_trajectory = (
            declared_max_int is None or max_budget >= declared_max_int
        )
        if (
            status is not None
            and status not in TERMINAL_STATES
            and requests_full_trajectory
        ):
            raise MaterializationError(
                "round_state is not terminal for full-trajectory materialization: "
                f"status={status!r}"
            )
    return root_manifest, state


def _round_for_budget(budget: int) -> int:
    if budget < EXPECTED_INITIAL_BUDGET:
        if budget == 50:
            return 0
        raise MaterializationError(f"unsupported sub-initial logical budget: {budget}")
    delta = budget - EXPECTED_INITIAL_BUDGET
    if delta % EXPECTED_ROUND_BUDGET:
        raise MaterializationError(
            f"logical budget {budget} is not on the 100 + n*50 acquisition schedule"
        )
    return delta // EXPECTED_ROUND_BUDGET


def _load_rounds(
    layout: SourceLayout,
    *,
    stage: str,
    max_round: int,
    state: dict[str, Any] | None,
) -> list[RoundSelection]:
    state_by_round: dict[int, dict[str, Any]] = {}
    if state is not None:
        state_rounds = state.get("rounds")
        if not isinstance(state_rounds, list):
            raise MaterializationError("round_state.rounds must be a JSON list")
        for position, record in enumerate(state_rounds):
            if not isinstance(record, dict):
                raise MaterializationError(f"round_state.rounds[{position}] must be an object")
            index = int(record.get("round", position))
            if index in state_by_round:
                raise MaterializationError(f"round_state contains duplicate round {index}")
            state_by_round[index] = record

    rounds: list[RoundSelection] = []
    cumulative_seen: set[str] = set()
    for index in range(max_round + 1):
        path = layout.round_root / f"round_{index:02d}" / "manifest.json"
        manifest = _read_json(path)
        if not isinstance(manifest, dict):
            raise MaterializationError(f"round manifest must be an object: {path}")
        if int(manifest.get("round", -1)) != index:
            raise MaterializationError(f"round number mismatch in {path}")
        if manifest.get("stage") not in (None, stage):
            raise MaterializationError(f"stage mismatch in {path}")
        if manifest.get("complete") is not True:
            raise MaterializationError(f"round is not complete: {path}")
        train = _require_string_list(manifest.get("train"), label=f"round {index} train")
        audit = _require_string_list(manifest.get("audit", []), label=f"round {index} audit")
        local_overlap = set(train).intersection(audit)
        if local_overlap:
            raise MaterializationError(
                f"round {index} train/audit overlap: {sorted(local_overlap)[:3]}"
            )
        _append_disjoint([], cumulative_seen, train, label=f"round {index} train")
        _append_disjoint([], cumulative_seen, audit, label=f"round {index} audit")

        roles_raw = manifest.get("roles", {})
        if not isinstance(roles_raw, dict):
            raise MaterializationError(f"round {index} roles must be an object")
        roles: dict[str, tuple[str, ...]] = {}
        for role, values in roles_raw.items():
            role_values = _require_string_list(values, label=f"round {index} role {role!r}")
            unknown = set(role_values).difference(train)
            if unknown:
                raise MaterializationError(
                    f"round {index} role {role!r} contains non-train IDs: {sorted(unknown)[:3]}"
                )
            roles[str(role)] = tuple(role_values)

        if state is not None:
            state_record = state_by_round.get(index)
            if state_record is None:
                raise MaterializationError(f"round_state is missing required round {index}")
            if "train" in state_record and list(state_record["train"]) != train:
                raise MaterializationError(f"round_state/manifest train mismatch at round {index}")
            if "audit" in state_record and list(state_record["audit"]) != audit:
                raise MaterializationError(f"round_state/manifest audit mismatch at round {index}")

        rounds.append(RoundSelection(
            index=index,
            manifest_path=path.resolve(),
            manifest_sha256=_file_sha256(path),
            train=tuple(train),
            audit=tuple(audit),
            roles=roles,
        ))
    return rounds


def _round0_fifty(round0: RoundSelection) -> tuple[list[str], dict[str, Any]]:
    selected: list[str] = []
    selected_set: set[str] = set()
    selected_by_role: dict[str, int] = {}
    for role, quota in ROUND0_ROLE_QUOTAS:
        added = 0
        for image in round0.roles.get(role, ()):
            if image in selected_set:
                continue
            selected.append(image)
            selected_set.add(image)
            added += 1
            if added >= quota:
                break
        selected_by_role[role] = added

    before_backfill = len(selected)
    for image in round0.train:
        if len(selected) >= 50:
            break
        if image not in selected_set:
            selected.append(image)
            selected_set.add(image)
    if len(selected) != 50:
        raise MaterializationError(
            f"round 0 cannot supply the required 50 unique training IDs; found {len(selected)}"
        )
    return selected, {
        "rule": "r0-stratified-35-query-8-attribute-7-mmr-v1",
        "requested_by_role": {role: quota for role, quota in ROUND0_ROLE_QUOTAS},
        "selected_by_role": selected_by_role,
        "round0_order_backfill_count": len(selected) - before_backfill,
    }


def _cumulative(rounds: Sequence[RoundSelection], target_round: int) -> tuple[list[str], list[str]]:
    train: list[str] = []
    audit: list[str] = []
    for record in rounds[: target_round + 1]:
        train.extend(record.train)
        audit.extend(record.audit)
    return train, audit


def _validate_manifest(payload: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise MaterializationError(f"task manifest must contain a tasks list: {path}")
    tasks = payload["tasks"]
    if not tasks:
        raise MaterializationError("task manifest contains no tasks")
    declared_count = payload.get("task_count")
    if declared_count is not None and int(declared_count) != len(tasks):
        raise MaterializationError(
            f"task_count mismatch: declared {declared_count}, found {len(tasks)}"
        )
    seen_orders: set[int] = set()
    seen_tasks: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for position, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise MaterializationError(f"tasks[{position}] must be an object")
        try:
            order = int(task["order"])
            dataset = str(task["dataset"])
            name = str(task["task"])
        except (KeyError, TypeError, ValueError) as error:
            raise MaterializationError(
                f"tasks[{position}] requires integer order plus dataset/task"
            ) from error
        if order < 0 or not dataset or not name:
            raise MaterializationError(f"invalid task identity at tasks[{position}]")
        if order in seen_orders:
            raise MaterializationError(f"duplicate task order in manifest: {order}")
        if (dataset, name) in seen_tasks:
            raise MaterializationError(f"duplicate dataset/task in manifest: {dataset}/{name}")
        seen_orders.add(order)
        seen_tasks.add((dataset, name))
        normalized.append({**task, "order": order, "dataset": dataset, "task": name})
    return normalized


def _validate_budgets(budgets: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in budgets)
    if not normalized or tuple(sorted(set(normalized))) != normalized:
        raise MaterializationError("budgets must be strictly increasing and unique")
    unsupported = [budget for budget in normalized if budget not in DEFAULT_BUDGETS]
    if unsupported:
        raise MaterializationError(
            f"unsupported logical budgets {unsupported}; allowed values are {DEFAULT_BUDGETS}"
        )
    for budget in normalized:
        _round_for_budget(budget)
    return normalized


def _task_pending_writes(
    task: dict[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
    output_root: Path,
    stage: str,
    budgets: Sequence[int],
) -> list[PendingWrite]:
    root = _task_root(task, manifest_path=manifest_path, repo_root=repo_root)
    layout = _resolve_source_layout(
        task, task_root=root, manifest_path=manifest_path, stage=stage,
    )
    state = _read_json(layout.round_state) if layout.round_state is not None else None
    if state is not None and not isinstance(state, dict):
        raise MaterializationError(f"round_state must be a JSON object: {layout.round_state}")
    root_manifest, state = _protocol_from_sources(
        layout, state, stage=stage, max_budget=max(budgets),
    )
    max_round = max(_round_for_budget(budget) for budget in budgets)
    rounds = _load_rounds(
        layout, stage=stage, max_round=max_round, state=state,
    )

    task_id = f"{task['order']:03d}_{task['dataset']}_{task['task']}"
    previous_budget: int | None = None
    previous_train: list[str] | None = None
    pending: list[PendingWrite] = []
    root_manifest_hash = _file_sha256(layout.root_manifest)
    state_hash = _file_sha256(layout.round_state) if layout.round_state is not None else None

    for budget in budgets:
        source_round = _round_for_budget(budget)
        cumulative_train, cumulative_audit = _cumulative(rounds, source_round)
        if budget == 50:
            materialized_train, selection_rule = _round0_fifty(rounds[0])
            materialized_audit: list[str] = []
        else:
            materialized_train = cumulative_train
            materialized_audit = cumulative_audit
            selection_rule = {
                "rule": "source-round-cumulative-train-v1",
                "requested_by_role": None,
                "selected_by_role": None,
                "round0_order_backfill_count": 0,
            }

        if len(materialized_train) != len(set(materialized_train)):
            raise MaterializationError(f"output selection has duplicates: {task_id}/budget {budget}")
        if set(materialized_train).intersection(materialized_audit):
            raise MaterializationError(f"output train/audit overlap: {task_id}/budget {budget}")
        if previous_train is not None and not set(previous_train).issubset(materialized_train):
            missing = sorted(set(previous_train).difference(materialized_train))[:3]
            raise MaterializationError(
                f"budget selections are not nested for {task_id}: "
                f"{previous_budget} !subset {budget}; missing={missing}"
            )
        observed_total = len(materialized_train) + len(materialized_audit)
        if observed_total > budget:
            raise MaterializationError(
                f"observed source labels exceed logical budget for {task_id}/budget {budget}: "
                f"{observed_total} > {budget}"
            )

        split_root = (
            output_root
            / f"budget_{budget:03d}"
            / "tasks"
            / task_id
            / "vqa"
            / "iterative"
            / "split"
        )
        selection_hash = _sequence_sha256(materialized_train)
        source_record = rounds[source_round]
        provenance = {
            "schema_version": 1,
            "materializer": Path(__file__).name,
            "task": {
                "order": task["order"],
                "dataset": task["dataset"],
                "task": task["task"],
                "output_id": task_id,
            },
            "acquisition_stage": stage,
            "logical_budget": budget,
            "materialized": {
                "train_count": len(materialized_train),
                "audit_count": len(materialized_audit),
                "observed_total_labeled_count": observed_total,
                "logical_budget_shortfall": budget - observed_total,
                "train_selection_sha256": selection_hash,
                "audit_selection_sha256": _sequence_sha256(materialized_audit),
                "no_duplicate_train_ids": True,
                "train_audit_disjoint": True,
                "nested_from_budget": previous_budget,
                "nested_from_train_selection_sha256": (
                    _sequence_sha256(previous_train) if previous_train is not None else None
                ),
                **selection_rule,
            },
            "source": {
                "layout": layout.kind,
                "source_root": str(layout.source_root),
                "root_manifest": str(layout.root_manifest.resolve()),
                "root_manifest_sha256": root_manifest_hash,
                "round_state": str(layout.round_state.resolve()) if layout.round_state else None,
                "round_state_sha256": state_hash,
                "source_round": source_round,
                "source_round_manifest": str(source_record.manifest_path),
                "source_round_manifest_sha256": source_record.manifest_sha256,
                "cumulative_round_manifest_sha256": [
                    record.manifest_sha256 for record in rounds[: source_round + 1]
                ],
                "source_cumulative_train_count": len(cumulative_train),
                "source_cumulative_audit_count": len(cumulative_audit),
                "protocol": {
                    key: root_manifest["budget"].get(key)
                    for key in ("initial", "round", "train", "audit", "max", "default_stop")
                    if key in root_manifest["budget"]
                },
            },
            "ground_truth_read": False,
        }
        pending.extend((
            PendingWrite(split_root / "train_labeled_indices.json", materialized_train),
            PendingWrite(split_root / "budget_prefix_provenance.json", provenance),
        ))
        previous_budget = budget
        previous_train = list(materialized_train)
    return pending


def _validate_destinations(pending: Sequence[PendingWrite], *, overwrite: bool) -> None:
    seen: set[Path] = set()
    for item in pending:
        resolved = item.path.resolve()
        if resolved in seen:
            raise MaterializationError(f"duplicate output target: {resolved}")
        seen.add(resolved)
        if not item.path.exists():
            continue
        current = _read_json(item.path)
        if current == item.payload:
            continue
        if not overwrite:
            raise MaterializationError(
                f"output already exists with different content: {item.path}; pass --overwrite"
            )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def materialize_budget_prefixes(
    manifest_path: Path,
    acquisition_stage: str,
    output_root: Path,
    *,
    repo_root: Path | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    orders: Sequence[int] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate every task/budget first, then atomically materialize selections."""

    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    if not acquisition_stage or any(separator in acquisition_stage for separator in ("/", "\\")):
        raise MaterializationError("acquisition_stage must be a stage name, not a path")
    normalized_budgets = _validate_budgets(budgets)
    manifest = _read_json(manifest_path)
    tasks = _validate_manifest(manifest, manifest_path)
    if orders is not None:
        requested_orders = {int(value) for value in orders}
        tasks = [task for task in tasks if task["order"] in requested_orders]
        missing_orders = requested_orders.difference(task["order"] for task in tasks)
        if missing_orders:
            raise MaterializationError(
                f"requested task orders are absent from manifest: {sorted(missing_orders)}"
            )
        if not tasks:
            raise MaterializationError("task-order filter selected no tasks")
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root.resolve()

    pending: list[PendingWrite] = []
    for task in sorted(tasks, key=lambda value: value["order"]):
        pending.extend(_task_pending_writes(
            task,
            manifest_path=manifest_path,
            repo_root=repo_root,
            output_root=output_root,
            stage=acquisition_stage,
            budgets=normalized_budgets,
        ))
    _validate_destinations(pending, overwrite=overwrite)
    for item in pending:
        _atomic_write_json(item.path, item.payload)
    return {
        "manifest": str(manifest_path),
        "acquisition_stage": acquisition_stage,
        "output_root": str(output_root),
        "budgets": list(normalized_budgets),
        "task_count": len(tasks),
        "file_count": len(pending),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest containing tasks")
    parser.add_argument("--acquisition-stage", required=True, help="iterative acquisition stage name")
    parser.add_argument("--output-root", type=Path, required=True, help="budget experiment root")
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="repository root; defaults to the root containing probe_learning",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace differing pre-existing materializations after full validation",
    )
    parser.add_argument(
        "--orders", type=int, nargs="+", default=None,
        help="optional subset of task orders (the default materializes every task)",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS),
        help=(
            "strictly increasing subset of 50 100 150 200 300 500; useful for "
            "materializing only committed prefixes while acquisition continues"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = materialize_budget_prefixes(
        args.manifest,
        args.acquisition_stage,
        args.output_root,
        repo_root=args.repo_root,
        budgets=args.budgets,
        orders=args.orders,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
