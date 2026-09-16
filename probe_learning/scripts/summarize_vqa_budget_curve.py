#!/usr/bin/env python3
"""Summarize clean-Test Joint AP across VQA-budget ProbeBank runs.

Each input is a completed ``run_probebank_batch.py`` output root for one
nominal VQA budget.  The script keeps a fully flattened metrics table, then
builds the paper curve from the suite-defined Ours-Full row only.  Seed scores
are averaged within task before any task-macro aggregation.  Bootstrap samples
reuse the same task indices at every budget, so both absolute intervals and
budget deltas are paired by task.

Historical curves may be supplied for visual context.  They are written to a
separate table and never enter current-task estimates or confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAL23_ORDERS = (
    1, 2, 3, 5, 6, 8, 9, 12, 17, 18, 19, 20, 31, 32, 33, 34, 35,
    37, 38, 40, 42, 43, 45,
)
ALL36_ORDERS = tuple(sorted((*FORMAL23_ORDERS, *range(46, 59))))
EXPECTED_BUDGETS = (50, 100, 150, 200, 300, 500)
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_LABEL_COUNTS = {
    50: (50, 0),
    100: (100, 0),
    150: (140, 10),
    200: (180, 20),
    300: (260, 40),
    500: (420, 80),
}
PRIOR_PLATEAU_BUDGET = 150
PROBEBANK_MIN_CLASS_SAMPLES = 2
CANONICAL_OURS_FULL_IDS = (
    "all_probe_softgate_probe_level_product",
    "r_softgate_minmax_all_gridinit_joint_bce_fixed_t",
    "paper_ours_full_selected8",
    "r_softgate_all",
)


@dataclass(frozen=True)
class BudgetRun:
    budget: int
    root: Path


@dataclass(frozen=True)
class TaskRecord:
    budget: int
    realized_budget: int | None
    train_label_count: int | None
    audit_label_count: int | None
    order: int
    dataset: str
    task: str
    task_key: str
    suite_id: str
    method_id: str
    method_resolution: str
    supervision_hash: str
    min_attr_positive: int | None
    min_attr_negative: int | None
    single_class_attribute_count: int | None
    insufficient_balance_attribute_count: int | None
    joint_positive: int | None
    joint_negative: int | None
    clean_test_verified: bool
    metrics_path: Path
    metrics: tuple[dict[str, Any], ...]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_assignment(value: str, kind: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{kind} must use NAME=VALUE syntax: {value!r}")
    left, right = value.split("=", 1)
    if not left.strip() or not right.strip():
        raise argparse.ArgumentTypeError(f"{kind} must have non-empty sides: {value!r}")
    return left.strip(), right.strip()


def load_budget_runs(assignments: Sequence[str], manifest_path: Path | None) -> list[BudgetRun]:
    records: list[BudgetRun] = []
    for assignment in assignments:
        raw_budget, raw_path = parse_assignment(assignment, "--budget-run")
        try:
            budget = int(raw_budget)
        except ValueError as exc:
            raise ValueError(f"invalid VQA budget {raw_budget!r}") from exc
        records.append(BudgetRun(budget=budget, root=Path(raw_path).resolve()))
    if manifest_path is not None:
        payload = read_json(manifest_path)
        items = payload.get("runs", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError("budget-runs manifest must be a list or an object with a runs list")
        for item in items:
            if not isinstance(item, dict) or "budget" not in item or "path" not in item:
                raise ValueError("every budget-runs manifest item needs budget and path")
            records.append(
                BudgetRun(int(item["budget"]), Path(str(item["path"])).resolve())
            )
    if not records:
        raise ValueError("at least one --budget-run or --budget-runs-json entry is required")
    seen: set[int] = set()
    for record in records:
        if record.budget <= 0:
            raise ValueError(f"VQA budget must be positive: {record.budget}")
        if record.budget in seen:
            raise ValueError(f"duplicate VQA budget: {record.budget}")
        if not record.root.is_dir():
            raise FileNotFoundError(f"budget output root does not exist: {record.root}")
        seen.add(record.budget)
    return sorted(records, key=lambda row: row.budget)


def find_metric_files(root: Path) -> list[Path]:
    direct = sorted((root / "tasks").glob("*/iterative_metrics.json"))
    if direct:
        return direct
    nested = sorted(
        path for path in root.rglob("iterative_metrics.json")
        if path.parent.parent.name == "tasks"
    )
    if not nested:
        raise FileNotFoundError(f"no per-task iterative_metrics.json under {root}")
    return nested


def nearest_run_root(metrics_path: Path, supplied_root: Path) -> Path:
    current = metrics_path.parent
    while True:
        if (current / "run_state.json").is_file():
            return current
        if current == supplied_root or supplied_root not in current.parents:
            return supplied_root
        current = current.parent


def load_suite(run_root: Path) -> tuple[dict[str, Any], str]:
    state_path = run_root / "run_state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    suite_id = str(state.get("suite_id") or "")
    raw_path = state.get("suite_config")
    if not raw_path:
        return {}, suite_id
    suite_path = Path(str(raw_path))
    if not suite_path.is_absolute():
        candidates = (run_root / suite_path, Path.cwd() / suite_path)
        suite_path = next((path for path in candidates if path.is_file()), candidates[0])
    if not suite_path.is_file():
        raise FileNotFoundError(f"suite_config recorded by {state_path} is missing: {suite_path}")
    suite = read_json(suite_path)
    return suite, str(suite.get("suite_id") or suite_id)


def normalized_display(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def suite_ours_full_candidates(suite: Mapping[str, Any]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for key in (
        "ours_full_method_id", "ours_full_method", "primary_fusion_method_id",
        "primary_fusion_method",
    ):
        value = suite.get(key)
        if value:
            candidates.append((str(value), f"suite:{key}"))

    display_names = suite.get("method_display_names", {})
    if isinstance(display_names, dict):
        for method_id, display in display_names.items():
            if "oursfull" in normalized_display(display):
                candidates.append((str(method_id), "suite:method_display_names"))

    recipes = suite.get("fusion_recipes", [])
    if isinstance(recipes, list):
        for recipe in recipes:
            if not isinstance(recipe, dict) or not recipe.get("id"):
                continue
            recipe_id = str(recipe["id"])
            display = recipe.get("display_name", recipe.get("name", ""))
            explicit_name = "oursfull" in normalized_display(display)
            explicit_id = "ours_full" in recipe_id.lower() or "ours-full" in recipe_id.lower()
            full_softgate = (
                str(recipe.get("aggregation", "")).lower() == "r_softgate"
                and (
                    str(recipe.get("top_k", "")).lower() == "all"
                    or str(recipe.get("pool_id", "")).lower() == "all"
                )
            )
            if explicit_name or explicit_id or full_softgate:
                candidates.append((recipe_id, "suite:fusion_recipes"))
                # Batch outputs materialize a fusion recipe into one method ID
                # per gate strategy and Joint rule.  The recipe ID itself is
                # therefore metadata (for example ``paper_ours_full_selected8``),
                # while the metric row uses an expanded ID such as
                # ``all_probe_softgate_probe_level_product``.
                prefix = recipe.get("id_prefix")
                gate_strategies = recipe.get("gate_strategies")
                joint_rules = recipe.get("joint_rules")
                if (
                    isinstance(prefix, str)
                    and prefix
                    and isinstance(gate_strategies, list)
                    and isinstance(joint_rules, list)
                ):
                    for gate_strategy in gate_strategies:
                        for joint_rule in joint_rules:
                            if gate_strategy and joint_rule:
                                candidates.append(
                                    (
                                        f"{prefix}_{gate_strategy}_{joint_rule}",
                                        "suite:fusion_recipes-expanded",
                                    )
                                )
    return candidates


def output_ours_full_candidates(metrics: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    available = {str(row.get("method")) for row in metrics if row.get("method")}
    candidates: list[tuple[str, str]] = []
    for method_id in CANONICAL_OURS_FULL_IDS:
        if method_id in available:
            candidates.append((method_id, "output:canonical-id"))
    for row in metrics:
        method_id = str(row.get("method") or "")
        if not method_id:
            continue
        display_values = (row.get("display_name"), row.get("method_display_name"), row.get("label"))
        if any("oursfull" in normalized_display(value) for value in display_values if value):
            candidates.append((method_id, "output:display-name"))
        fit = row.get("fusion_fit") if isinstance(row.get("fusion_fit"), dict) else {}
        aggregation = str(row.get("aggregation") or fit.get("aggregation") or "").lower()
        top_k = str(fit.get("top_k", "")).lower()
        pool_id = str(row.get("pool_id") or fit.get("pool_id") or "").lower()
        if aggregation == "r_softgate" and (top_k == "all" or pool_id == "all"):
            candidates.append((method_id, "output:r_softgate-all"))
    return candidates


def resolve_ours_full_method(
    suite: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    override: str | None = None,
) -> tuple[str, str]:
    available = {str(row.get("method")) for row in metrics if row.get("method")}
    if override:
        if override not in available:
            raise ValueError(
                f"requested Ours-Full method {override!r} is absent; available={sorted(available)}"
            )
        return override, "cli:override"

    candidates = [
        (method_id, source)
        for method_id, source in (*suite_ours_full_candidates(suite), *output_ours_full_candidates(metrics))
        if method_id in available
    ]
    by_id: dict[str, set[str]] = defaultdict(set)
    for method_id, source in candidates:
        by_id[method_id].add(source)
    if not by_id:
        raise ValueError(
            "could not resolve Ours-Full from suite/output; pass --ours-full-method. "
            f"Available methods: {sorted(available)}"
        )
    if len(by_id) > 1:
        for preferred in CANONICAL_OURS_FULL_IDS:
            if preferred in by_id:
                return preferred, "+".join(sorted(by_id[preferred]))
        raise ValueError(
            "ambiguous Ours-Full candidates from suite/output: "
            + ", ".join(f"{key} ({'+'.join(sorted(value))})" for key, value in sorted(by_id.items()))
        )
    method_id = next(iter(by_id))
    return method_id, "+".join(sorted(by_id[method_id]))


def parse_task_identity(task_dir: Path, summary: Mapping[str, Any]) -> tuple[int, str, str]:
    match = re.match(r"^(\d{3})_([^_]+)_(.+)$", task_dir.name)
    order = summary.get("order")
    dataset = summary.get("dataset")
    task = summary.get("task")
    if order is None and match:
        order = int(match.group(1))
    if not dataset and match:
        dataset = match.group(2)
    if not task and match:
        task = match.group(3)
    if order is None or not dataset or not task:
        raise ValueError(f"cannot determine task identity for {task_dir}")
    return int(order), str(dataset).lower(), str(task)


def _one_clean_test_status(audit: Mapping[str, Any], *, source: str) -> tuple[bool, str]:
    status = audit.get("status")
    if status is not None and status != "valid":
        return False, f"{source}: supervision audit status={status!r}"
    boundary = audit.get("evaluation_boundary", {})
    if not isinstance(boundary, dict) or "frozen_test_overlap_count" not in boundary:
        return False, f"{source}: missing evaluation_boundary.frozen_test_overlap_count"
    overlap = int(boundary.get("frozen_test_overlap_count", -1))
    membership = boundary.get("gallery_membership_valid", True)
    if overlap != 0:
        return False, f"{source}: frozen_test_overlap_count={overlap}"
    if membership is not True:
        return False, f"{source}: gallery_membership_valid is not true"
    return True, f"{source}: verified zero overlap"


def clean_test_status(
    summary: Mapping[str, Any],
    audit_file: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Require every available audit representation to prove Test isolation."""

    audits = summary.get("supervision_audits", {})
    summary_audit = audits.get("iterative") if isinstance(audits, dict) else None
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(summary_audit, dict):
        candidates.append(("summary", summary_audit))
    if isinstance(audit_file, dict):
        candidates.append(("audit file", audit_file))
    if not candidates:
        return False, "missing iterative supervision audit"
    reasons = []
    for source, audit in candidates:
        valid, reason = _one_clean_test_status(audit, source=source)
        if not valid:
            return False, reason
        reasons.append(reason)
    return True, "; ".join(reasons)


def completed_task_status(summary: Mapping[str, Any]) -> tuple[bool, str]:
    if summary.get("status") != "completed":
        return False, f"summary status={summary.get('status')!r}"
    stages = summary.get("stage_status")
    if not isinstance(stages, dict) or stages.get("iterative") != "completed":
        return False, f"iterative stage status={stages.get('iterative') if isinstance(stages, dict) else None!r}"
    return True, "task and iterative stage completed"


def _nonnegative_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-negative integer, found {value!r}")
    count = int(value)
    if count != value or count < 0:
        raise ValueError(f"{label} must be a non-negative integer, found {value!r}")
    return count


def supervision_class_diagnostics(summary: Mapping[str, Any]) -> dict[str, int | None]:
    """Extract class-support diagnostics without changing any AP estimate.

    Older fixtures/archived summaries may predate these audit fields.  A fully
    absent diagnostic block is represented by ``None`` values; malformed or
    partially specified count entries fail closed.
    """

    audits = summary.get("supervision_audits", {})
    audit = audits.get("iterative") if isinstance(audits, dict) else None
    if not isinstance(audit, dict):
        return {
            "min_attr_positive": None,
            "min_attr_negative": None,
            "single_class_attribute_count": None,
            "insufficient_balance_attribute_count": None,
            "joint_positive": None,
            "joint_negative": None,
        }

    attribute_counts = audit.get("attribute_counts")
    if attribute_counts is None:
        min_positive = min_negative = single_class = insufficient_balance = None
    else:
        if not isinstance(attribute_counts, dict) or not attribute_counts:
            raise ValueError("iterative attribute_counts must be a non-empty object")
        positives: list[int] = []
        negatives: list[int] = []
        for attribute, counts in attribute_counts.items():
            if not isinstance(counts, dict) or "positive" not in counts or "negative" not in counts:
                raise ValueError(
                    f"attribute_counts[{attribute!r}] requires positive and negative"
                )
            positives.append(_nonnegative_count(
                counts["positive"], label=f"attribute_counts[{attribute!r}].positive"
            ))
            negatives.append(_nonnegative_count(
                counts["negative"], label=f"attribute_counts[{attribute!r}].negative"
            ))
        min_positive = min(positives)
        min_negative = min(negatives)
        attribute_totals = {
            positive + negative for positive, negative in zip(positives, negatives)
        }
        if len(attribute_totals) != 1:
            raise ValueError(
                f"attribute class-count totals disagree: {sorted(attribute_totals)}"
            )
        single_class = sum(
            positive == 0 or negative == 0
            for positive, negative in zip(positives, negatives)
        )
        insufficient_balance = sum(
            positive < PROBEBANK_MIN_CLASS_SAMPLES
            or negative < PROBEBANK_MIN_CLASS_SAMPLES
            for positive, negative in zip(positives, negatives)
        )
        if insufficient_balance < single_class:
            raise AssertionError("single-class attributes must be a subset of insufficient balance")

    joint_counts = audit.get("joint_counts")
    if joint_counts is None:
        joint_positive = joint_negative = None
    else:
        if (
            not isinstance(joint_counts, dict)
            or "positive" not in joint_counts
            or "negative" not in joint_counts
        ):
            raise ValueError("iterative joint_counts requires positive and negative")
        joint_positive = _nonnegative_count(
            joint_counts["positive"], label="joint_counts.positive"
        )
        joint_negative = _nonnegative_count(
            joint_counts["negative"], label="joint_counts.negative"
        )
        if attribute_counts is not None:
            attribute_total = next(iter(attribute_totals))
            if joint_positive + joint_negative != attribute_total:
                raise ValueError(
                    "joint and attribute class-count totals disagree: "
                    f"{joint_positive + joint_negative} != {attribute_total}"
                )

    return {
        "min_attr_positive": min_positive,
        "min_attr_negative": min_negative,
        "single_class_attribute_count": single_class,
        "insufficient_balance_attribute_count": insufficient_balance,
        "joint_positive": joint_positive,
        "joint_negative": joint_negative,
    }


def budget_counts_from_task(
    task_dir: Path,
    summary: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None]:
    provenance_path = (
        task_dir
        / "vqa"
        / "iterative"
        / "split"
        / "budget_prefix_provenance.json"
    )
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        materialized = provenance.get("materialized", {})
        if not isinstance(materialized, dict):
            raise ValueError(f"invalid materialized budget provenance: {provenance_path}")
        logical = int(provenance["logical_budget"])
        train = int(materialized["train_count"])
        audit = int(materialized["audit_count"])
        observed = int(materialized["observed_total_labeled_count"])
        shortfall = int(materialized.get("logical_budget_shortfall", logical - observed))
        if min(logical, train, audit, observed) < 0:
            raise ValueError(f"negative budget count in {provenance_path}")
        if observed != train + audit or observed != logical or shortfall != 0:
            raise ValueError(
                "inconsistent logical/train/audit budget provenance in "
                f"{provenance_path}"
            )
        return logical, train, audit

    # Compatibility for historical/fixture outputs that predate materialized
    # prefix provenance.  Physical API call counts are not a valid fallback for
    # current runs, which always carry the provenance file above.
    for key in ("vqa_successful", "vqa_attempted"):
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(value), None, None
    return None, None, None


def collect_task_records(
    runs: Sequence[BudgetRun],
    *,
    ours_full_override: str | None,
    require_clean_test_audit: bool,
) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    seen: set[tuple[int, int]] = set()
    for run in runs:
        for metrics_path in find_metric_files(run.root):
            task_dir = metrics_path.parent
            summary_path = task_dir / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"task summary is missing: {summary_path}")
            summary = read_json(summary_path)
            completed, completed_reason = completed_task_status(summary)
            if not completed:
                raise ValueError(
                    f"incomplete task output at budget={run.budget}, {task_dir.name}: "
                    f"{completed_reason}"
                )
            order, dataset, task = parse_task_identity(task_dir, summary)
            identity = (run.budget, order)
            if identity in seen:
                raise ValueError(f"duplicate task order {order} at budget {run.budget}")
            seen.add(identity)

            payload = read_json(metrics_path)
            if not isinstance(payload, list) or not payload:
                raise ValueError(f"metrics must be a non-empty JSON list: {metrics_path}")
            metrics = tuple(row for row in payload if isinstance(row, dict))
            local_root = nearest_run_root(metrics_path, run.root)
            suite, suite_id = load_suite(local_root)
            method_id, resolution = resolve_ours_full_method(
                suite, metrics, ours_full_override
            )
            audit_path = task_dir / "supervision_audit" / "iterative.json"
            audit_file = read_json(audit_path) if audit_path.is_file() else None
            clean, clean_reason = clean_test_status(summary, audit_file)
            if require_clean_test_audit and not clean:
                raise ValueError(
                    f"clean Test audit failed for budget={run.budget}, order={order}: {clean_reason}"
                )
            diagnostics = supervision_class_diagnostics(summary)
            realized_budget, train_label_count, audit_label_count = (
                budget_counts_from_task(task_dir, summary)
            )
            if realized_budget is not None and realized_budget != run.budget:
                raise ValueError(
                    f"nominal budget={run.budget} but realized logical budget="
                    f"{realized_budget} for {task_dir.name}"
                )
            summary_hashes = summary.get("supervision_hashes", {})
            supervision_hash = (
                str(summary_hashes.get("iterative") or "")
                if isinstance(summary_hashes, dict)
                else ""
            )
            audit_hashes = []
            audits = summary.get("supervision_audits", {})
            if isinstance(audits, dict) and isinstance(audits.get("iterative"), dict):
                audit_hashes.append(audits["iterative"].get("supervision_hash"))
            if isinstance(audit_file, dict):
                audit_hashes.append(audit_file.get("supervision_hash"))
            conflicting_hashes = {
                str(value) for value in audit_hashes if value not in (None, "")
            }
            if supervision_hash:
                conflicting_hashes.add(supervision_hash)
            if len(conflicting_hashes) > 1:
                raise ValueError(
                    f"supervision hash mismatch across summary/audit for {task_dir.name}: "
                    f"{sorted(conflicting_hashes)}"
                )
            if not supervision_hash and conflicting_hashes:
                supervision_hash = next(iter(conflicting_hashes))
            records.append(
                TaskRecord(
                    budget=run.budget,
                    realized_budget=realized_budget,
                    train_label_count=train_label_count,
                    audit_label_count=audit_label_count,
                    order=order,
                    dataset=dataset,
                    task=task,
                    task_key=task_dir.name,
                    suite_id=suite_id,
                    method_id=method_id,
                    method_resolution=resolution,
                    supervision_hash=supervision_hash,
                    min_attr_positive=diagnostics["min_attr_positive"],
                    min_attr_negative=diagnostics["min_attr_negative"],
                    single_class_attribute_count=diagnostics["single_class_attribute_count"],
                    insufficient_balance_attribute_count=(
                        diagnostics["insufficient_balance_attribute_count"]
                    ),
                    joint_positive=diagnostics["joint_positive"],
                    joint_negative=diagnostics["joint_negative"],
                    clean_test_verified=clean,
                    metrics_path=metrics_path.resolve(),
                    metrics=metrics,
                )
            )
    return sorted(records, key=lambda row: (row.budget, row.order))


def flatten_metrics(records: Sequence[TaskRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for item in record.metrics:
            targets = item.get("metrics")
            if not isinstance(targets, dict):
                continue
            for target, values in targets.items():
                if not isinstance(values, dict):
                    continue
                for metric, value in values.items():
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        continue
                    rows.append(
                        {
                            "budget": record.budget,
                            "realized_budget": record.realized_budget,
                            "train_label_count": record.train_label_count,
                            "audit_label_count": record.audit_label_count,
                            "order": record.order,
                            "dataset": record.dataset,
                            "task": record.task,
                            "task_key": record.task_key,
                            "seed": item.get("seed"),
                            "method": item.get("method"),
                            "stage": item.get("stage"),
                            "split": item.get("split"),
                            "target": target,
                            "metric": metric,
                            "value": float(value),
                            "joint_rule": item.get("joint_rule"),
                            "family": item.get("family"),
                            "aggregation": item.get("aggregation"),
                            "supervision_hash": item.get("supervision_hash"),
                            "suite_id": record.suite_id,
                            "ours_full_method": record.method_id,
                            "ours_full_resolution": record.method_resolution,
                            "clean_test_verified": record.clean_test_verified,
                            "source_file": str(record.metrics_path),
                        }
                    )
    return sorted(
        rows,
        key=lambda row: (
            row["budget"], row["order"], str(row["method"]), str(row["split"]),
            -1 if row["seed"] is None else int(row["seed"]), str(row["target"]),
            str(row["metric"]),
        ),
    )


def extract_main_per_task_seed(records: Sequence[TaskRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        selected = [
            item for item in record.metrics
            if str(item.get("stage", "")).lower() == "iterative"
            and str(item.get("split", "")).lower() == "test"
            and str(item.get("method")) == record.method_id
        ]
        by_seed: dict[int, float] = {}
        for item in selected:
            seed = item.get("seed")
            if not isinstance(seed, (int, float)) or int(seed) != seed:
                raise ValueError(
                    f"Ours-Full row has a non-integer seed in {record.metrics_path}: {seed!r}"
                )
            joint = item.get("metrics", {}).get("joint", {})
            ap = joint.get("ap") if isinstance(joint, dict) else None
            if not isinstance(ap, (int, float)) or not math.isfinite(float(ap)):
                raise ValueError(
                    f"missing finite clean Test Joint AP for {record.task_key}, seed={seed}"
                )
            metric_hash = str(item.get("supervision_hash") or "")
            if record.supervision_hash and metric_hash != record.supervision_hash:
                raise ValueError(
                    f"stale Ours-Full metric supervision hash for {record.task_key}, "
                    f"seed={seed}: {metric_hash!r} != {record.supervision_hash!r}"
                )
            seed_int = int(seed)
            if seed_int in by_seed:
                raise ValueError(
                    f"duplicate Ours-Full Test row for {record.task_key}, seed={seed_int}"
                )
            by_seed[seed_int] = float(ap)
        if not by_seed:
            raise ValueError(
                f"no iterative Test rows for Ours-Full={record.method_id!r} in {record.metrics_path}"
            )
        for seed, ap in sorted(by_seed.items()):
            rows.append(
                {
                    "budget": record.budget,
                    "realized_budget": record.realized_budget,
                    "train_label_count": record.train_label_count,
                    "audit_label_count": record.audit_label_count,
                    "order": record.order,
                    "dataset": record.dataset,
                    "task": record.task,
                    "task_key": record.task_key,
                    "seed": seed,
                    "method": record.method_id,
                    "split": "test",
                    "target": "joint",
                    "metric": "ap",
                    "value": ap,
                    "suite_id": record.suite_id,
                    "supervision_hash": record.supervision_hash,
                    "min_attr_positive": record.min_attr_positive,
                    "min_attr_negative": record.min_attr_negative,
                    "single_class_attribute_count": record.single_class_attribute_count,
                    "insufficient_balance_attribute_count": (
                        record.insufficient_balance_attribute_count
                    ),
                    "joint_positive": record.joint_positive,
                    "joint_negative": record.joint_negative,
                    "clean_test_verified": record.clean_test_verified,
                    "source_file": str(record.metrics_path),
                }
            )
    return rows


def mean_by_task(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(int(row["budget"]), int(row["order"]))].append(row)
    output: list[dict[str, Any]] = []
    for (budget, order), group in sorted(grouped.items()):
        values = np.asarray([float(row["value"]) for row in group], dtype=np.float64)
        first = group[0]
        realized = {row.get("realized_budget") for row in group}
        train_counts = {row.get("train_label_count") for row in group}
        audit_counts = {row.get("audit_label_count") for row in group}
        supervision_hashes = {str(row.get("supervision_hash") or "") for row in group}
        diagnostic_fields = (
            "min_attr_positive",
            "min_attr_negative",
            "single_class_attribute_count",
            "insufficient_balance_attribute_count",
            "joint_positive",
            "joint_negative",
        )
        diagnostics: dict[str, Any] = {}
        for field in diagnostic_fields:
            values_for_field = {row.get(field) for row in group}
            if len(values_for_field) != 1:
                raise ValueError(
                    f"inconsistent {field} across seeds for budget={budget}, order={order}"
                )
            diagnostics[field] = next(iter(values_for_field))
        output.append(
            {
                "budget": budget,
                "realized_budget": next(iter(realized)) if len(realized) == 1 else None,
                "train_label_count": (
                    next(iter(train_counts)) if len(train_counts) == 1 else None
                ),
                "audit_label_count": (
                    next(iter(audit_counts)) if len(audit_counts) == 1 else None
                ),
                "order": order,
                "dataset": first["dataset"],
                "task": first["task"],
                "task_key": first["task_key"],
                "method": first["method"],
                "split": "test",
                "target": "joint",
                "metric": "ap",
                "seed_count": int(values.size),
                "mean_ap": float(values.mean()),
                "std_seed_ap": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min_seed_ap": float(values.min()),
                "max_seed_ap": float(values.max()),
                "supervision_hash": (
                    next(iter(supervision_hashes)) if len(supervision_hashes) == 1 else ""
                ),
                **diagnostics,
                "clean_test_verified": all(bool(row["clean_test_verified"]) for row in group),
            }
        )
    return output


def validate_panel_coverage(
    task_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
    *,
    allow_incomplete: bool,
) -> None:
    normalized_budgets = tuple(int(value) for value in budgets)
    by_budget: dict[int, set[int]] = defaultdict(set)
    for row in task_rows:
        by_budget[int(row["budget"])].add(int(row["order"]))
    expected = set(ALL36_ORDERS)
    invariant_messages: list[str] = []
    identities: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    for row in task_rows:
        identities[int(row["order"])].add(
            (str(row["dataset"]), str(row["task"]), str(row["task_key"]))
        )
    for order, values in sorted(identities.items()):
        if len(values) > 1:
            invariant_messages.append(
                f"order {order} changes task identity across budgets: {sorted(values)}"
            )
    if invariant_messages:
        raise ValueError("non-comparable budget panel; " + "; ".join(invariant_messages))

    if allow_incomplete:
        return

    messages: list[str] = []
    if normalized_budgets != EXPECTED_BUDGETS:
        messages.append(
            f"budgets must be exactly {list(EXPECTED_BUDGETS)}, found {list(normalized_budgets)}"
        )
    for budget in EXPECTED_BUDGETS:
        missing = sorted(expected - by_budget[budget])
        unexpected = sorted(by_budget[budget] - expected)
        if missing:
            messages.append(f"budget {budget}: missing All-36 orders {missing}")
        if unexpected:
            messages.append(f"budget {budget}: unexpected task orders {unexpected}")

    seeds_by_cell: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in seed_rows:
        seeds_by_cell[(int(row["budget"]), int(row["order"]))].add(int(row["seed"]))
    expected_seeds = set(EXPECTED_SEEDS)
    for budget in EXPECTED_BUDGETS:
        expected_train, expected_audit = EXPECTED_LABEL_COUNTS[budget]
        for order in ALL36_ORDERS:
            cell = (budget, order)
            if cell in seeds_by_cell and seeds_by_cell[cell] != expected_seeds:
                messages.append(
                    f"budget {budget}, order {order}: seeds "
                    f"{sorted(seeds_by_cell[cell])} != {list(EXPECTED_SEEDS)}"
                )
            rows = [
                row for row in task_rows
                if int(row["budget"]) == budget and int(row["order"]) == order
            ]
            if not rows:
                continue
            row = rows[0]
            if row.get("realized_budget") != budget:
                messages.append(
                    f"budget {budget}, order {order}: missing exact logical-budget provenance"
                )
            if (
                row.get("train_label_count") != expected_train
                or row.get("audit_label_count") != expected_audit
            ):
                messages.append(
                    f"budget {budget}, order {order}: train/audit counts "
                    f"{row.get('train_label_count')}/{row.get('audit_label_count')} != "
                    f"{expected_train}/{expected_audit}"
                )
            if not row.get("supervision_hash"):
                messages.append(
                    f"budget {budget}, order {order}: missing supervision hash"
                )
            missing_diagnostics = [
                field
                for field in (
                    "min_attr_positive",
                    "min_attr_negative",
                    "single_class_attribute_count",
                    "insufficient_balance_attribute_count",
                    "joint_positive",
                    "joint_negative",
                )
                if row.get(field) is None
            ]
            if missing_diagnostics:
                messages.append(
                    f"budget {budget}, order {order}: missing supervision diagnostics "
                    f"{missing_diagnostics}"
                )
            elif int(row["joint_positive"]) + int(row["joint_negative"]) != expected_train:
                messages.append(
                    f"budget {budget}, order {order}: joint class counts "
                    f"{row['joint_positive']}+{row['joint_negative']} != train {expected_train}"
                )
            if row.get("clean_test_verified") is not True:
                messages.append(
                    f"budget {budget}, order {order}: clean-Test isolation is unverified"
                )
    if messages:
        raise ValueError("incomplete All-36 budget runs; " + "; ".join(messages))


def cohort_groups(task_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    available = {int(row["order"]) for row in task_rows}
    groups = [
        {"group_type": "cohort", "group": "Formal-23", "orders": FORMAL23_ORDERS},
        {"group_type": "cohort", "group": "All-36", "orders": ALL36_ORDERS},
    ]
    dataset_orders: dict[str, set[int]] = defaultdict(set)
    for row in task_rows:
        order = int(row["order"])
        if order in ALL36_ORDERS:
            dataset_orders[str(row["dataset"])].add(order)
    for dataset, orders in sorted(dataset_orders.items()):
        groups.append(
            {"group_type": "dataset", "group": dataset, "orders": tuple(sorted(orders))}
        )
    return groups


def paired_bootstrap_curve(
    values_by_budget: Mapping[int, Mapping[int, float]],
    budgets: Sequence[int],
    *,
    repetitions: int,
    seed: int,
) -> tuple[list[int], dict[int, np.ndarray]]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    complete = sorted(
        set.intersection(*(set(values_by_budget[budget]) for budget in budgets))
    )
    if not complete:
        raise ValueError("no tasks are paired across all budgets")
    matrix = np.asarray(
        [[values_by_budget[budget][order] for budget in budgets] for order in complete],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[0], size=(repetitions, matrix.shape[0]))
    samples = matrix[indices].mean(axis=1)
    return complete, {budget: samples[:, position] for position, budget in enumerate(budgets)}


def quantile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def aggregate_macro_curves(
    task_rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    reference_budget: int | None,
) -> list[dict[str, Any]]:
    if not budgets:
        return []
    reference = (
        PRIOR_PLATEAU_BUDGET
        if reference_budget is None and PRIOR_PLATEAU_BUDGET in budgets
        else min(budgets) if reference_budget is None
        else int(reference_budget)
    )
    if reference not in budgets:
        raise ValueError(f"reference budget {reference} is not among {list(budgets)}")
    rows_by_order_budget = {
        (int(row["order"]), int(row["budget"])): row for row in task_rows
    }
    output: list[dict[str, Any]] = []
    for group_index, spec in enumerate(cohort_groups(task_rows)):
        declared_orders = tuple(int(order) for order in spec["orders"])
        if not declared_orders:
            continue
        values_by_budget: dict[int, dict[int, float]] = {}
        for budget in budgets:
            values_by_budget[budget] = {
                order: float(rows_by_order_budget[(order, budget)]["mean_ap"])
                for order in declared_orders
                if (order, budget) in rows_by_order_budget
            }
        complete_orders = set.intersection(
            *(set(values_by_budget[budget]) for budget in budgets)
        )
        # Partial pilots can have a dataset represented at early budgets but
        # no task from that dataset completed at every budget.  Such a group
        # has no paired estimand and must be omitted, not allowed to abort
        # otherwise valid paired cohort curves.
        if not complete_orders:
            continue
        paired_orders, bootstrap = paired_bootstrap_curve(
            values_by_budget,
            budgets,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + group_index,
        )
        paired_set = set(paired_orders)
        reference_samples = bootstrap[reference]
        previous: int | None = None
        for budget in budgets:
            point_values = np.asarray(
                [values_by_budget[budget][order] for order in paired_orders], dtype=np.float64
            )
            ci_low, ci_high = quantile_interval(bootstrap[budget])
            ref_delta_samples = bootstrap[budget] - reference_samples
            ref_low, ref_high = quantile_interval(ref_delta_samples)
            if previous is None:
                prev_delta = prev_low = prev_high = None
            else:
                delta_samples = bootstrap[budget] - bootstrap[previous]
                prev_delta = float(point_values.mean() - np.mean(
                    [values_by_budget[previous][order] for order in paired_orders]
                ))
                prev_low, prev_high = quantile_interval(delta_samples)
            seed_counts = [
                int(rows_by_order_budget[(order, budget)]["seed_count"])
                for order in paired_orders
            ]
            realized_values = [
                rows_by_order_budget[(order, budget)].get("realized_budget")
                for order in paired_orders
            ]
            finite_realized = [int(value) for value in realized_values if value is not None]
            single_class_values = [
                rows_by_order_budget[(order, budget)].get("single_class_attribute_count")
                for order in paired_orders
            ]
            diagnosed_single_class = [
                int(value) for value in single_class_values if value is not None
            ]
            single_class_task_count = (
                sum(value > 0 for value in diagnosed_single_class)
                if len(diagnosed_single_class) == len(paired_orders)
                else None
            )
            insufficient_values = [
                rows_by_order_budget[(order, budget)].get(
                    "insufficient_balance_attribute_count"
                )
                for order in paired_orders
            ]
            diagnosed_insufficient = [
                int(value) for value in insufficient_values if value is not None
            ]
            insufficient_task_count = (
                sum(value > 0 for value in diagnosed_insufficient)
                if len(diagnosed_insufficient) == len(paired_orders)
                else None
            )
            output.append(
                {
                    "group_type": spec["group_type"],
                    "group": spec["group"],
                    "budget": budget,
                    "task_count": len(paired_orders),
                    "declared_task_count": len(declared_orders),
                    "paired_task_count": len(paired_set),
                    "seed_count_min": min(seed_counts),
                    "seed_count_max": max(seed_counts),
                    "macro_test_joint_ap": float(point_values.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "reference_budget": reference,
                    "delta_vs_reference": float(point_values.mean() - np.mean(
                        [values_by_budget[reference][order] for order in paired_orders]
                    )),
                    "delta_vs_reference_ci95_low": ref_low,
                    "delta_vs_reference_ci95_high": ref_high,
                    "previous_budget": previous,
                    "delta_vs_previous": prev_delta,
                    "delta_vs_previous_ci95_low": prev_low,
                    "delta_vs_previous_ci95_high": prev_high,
                    "realized_budget_min": min(finite_realized) if finite_realized else None,
                    "realized_budget_max": max(finite_realized) if finite_realized else None,
                    "class_diagnostics_task_count": len(diagnosed_single_class),
                    "single_class_attribute_task_count": single_class_task_count,
                    "single_class_attribute_task_fraction": (
                        float(single_class_task_count / len(paired_orders))
                        if single_class_task_count is not None else None
                    ),
                    "insufficient_balance_attribute_task_count": insufficient_task_count,
                    "insufficient_balance_attribute_task_fraction": (
                        float(insufficient_task_count / len(paired_orders))
                        if insufficient_task_count is not None else None
                    ),
                    "method": sorted(
                        {str(rows_by_order_budget[(order, budget)]["method"]) for order in paired_orders}
                    )[0],
                    "split": "test",
                    "target": "joint",
                    "metric": "ap",
                }
            )
            previous = budget
    if not any(row["group_type"] == "cohort" for row in output):
        raise ValueError("no cohort tasks are paired across all budgets")
    return output


def load_legacy_curve(label: str, path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    if not source:
        raise ValueError(f"legacy curve is empty: {path}")
    fields = set(source[0])
    output: list[dict[str, Any]] = []
    if {"strategy", "total_vqa_labels", "macro_test_ap"} <= fields:
        for row in source:
            if str(row.get("strategy", "")).lower() != "iterative":
                continue
            output.append(
                {
                    "label": label,
                    "budget": int(float(row["total_vqa_labels"])),
                    "macro_test_joint_ap": float(row["macro_test_ap"]),
                    "task_count": int(float(row.get("n_tasks") or 0)) or None,
                    "source_file": str(path.resolve()),
                    "source_kind": "historical-context-only",
                }
            )
    elif {"order", "round", "split", "mean_ap"} <= fields:
        grouped: dict[int, list[float]] = defaultdict(list)
        orders: dict[int, set[int]] = defaultdict(set)
        for row in source:
            if str(row.get("split", "")).lower() != "test":
                continue
            round_index = int(float(row["round"]))
            total_vqa = 100 + 50 * round_index
            grouped[total_vqa].append(float(row["mean_ap"]))
            orders[total_vqa].add(int(float(row["order"])))
        # Archived studies may contain a later exploratory round for only a
        # subset of tasks.  Exclude those rows instead of drawing a curve whose
        # task panel silently changes with budget.
        complete_task_count = max((len(value) for value in orders.values()), default=0)
        for budget in sorted(grouped):
            if len(orders[budget]) != complete_task_count:
                continue
            output.append(
                {
                    "label": label,
                    "budget": budget,
                    "macro_test_joint_ap": float(np.mean(grouped[budget])),
                    "task_count": len(orders[budget]),
                    "source_file": str(path.resolve()),
                    "source_kind": "historical-context-only",
                }
            )
    else:
        raise ValueError(
            f"unsupported legacy curve schema at {path}; fields={sorted(fields)}"
        )
    return output


def render_curve_plot(
    rows: Sequence[Mapping[str, Any]],
    output_base: Path,
    *,
    group_type: str,
    title: str,
    legacy_rows: Sequence[Mapping[str, Any]] = (),
) -> None:
    selected = [row for row in rows if row["group_type"] == group_type]
    if not selected:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _render_curve_plot_pillow(selected, output_base, title, legacy_rows)
        return
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    for group in sorted({str(row["group"]) for row in selected}):
        group_rows = sorted(
            (row for row in selected if row["group"] == group),
            key=lambda row: int(row["budget"]),
        )
        task_count = min(int(row["task_count"]) for row in group_rows)
        declared_count = max(int(row["declared_task_count"]) for row in group_rows)
        display_group = (
            group
            if task_count == declared_count
            else f"{group} pilot ({task_count}/{declared_count} tasks)"
        )
        x = np.asarray([row["budget"] for row in group_rows], dtype=float)
        y = np.asarray([row["macro_test_joint_ap"] for row in group_rows], dtype=float)
        low = np.asarray([row["ci95_low"] for row in group_rows], dtype=float)
        high = np.asarray([row["ci95_high"] for row in group_rows], dtype=float)
        line, = axis.plot(x, y, marker="o", linewidth=2.0, label=display_group)
        axis.fill_between(x, low, high, color=line.get_color(), alpha=0.14)
    for label in sorted({str(row["label"]) for row in legacy_rows}):
        legacy = sorted(
            (row for row in legacy_rows if row["label"] == label),
            key=lambda row: int(row["budget"]),
        )
        axis.plot(
            [row["budget"] for row in legacy],
            [row["macro_test_joint_ap"] for row in legacy],
            linestyle="--", marker="x", linewidth=1.2, alpha=0.65,
            label=f"Historical only: {label}",
        )
    axis.set_title(title)
    axis.set_xlabel("Nominal VQA budget (images / task)")
    axis.set_ylabel("Macro clean-Test Joint AP")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_curve_plot_pillow(
    selected: Sequence[Mapping[str, Any]],
    output_base: Path,
    title: str,
    legacy_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Dependency-light plot fallback used when matplotlib is unavailable."""
    from PIL import Image, ImageDraw, ImageFont

    def load_font(size: int, *, bold: bool = False) -> Any:
        names = (
            ("DejaVuSans-Bold.ttf", "arialbd.ttf")
            if bold
            else ("DejaVuSans.ttf", "arial.ttf")
        )
        candidates = [
            *names,
            *(f"C:/Windows/Fonts/{name}" for name in names),
            *(
                f"/usr/share/fonts/truetype/dejavu/{name}"
                for name in names
                if name.startswith("DejaVu")
            ),
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    width, height = 1640, 1000
    left, top, right, bottom = 190, 110, 70, 145
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(28, bold=True)
    label_font = load_font(20)
    tick_font = load_font(17)
    legend_font = load_font(17)
    current_budgets = [float(row["budget"]) for row in selected]
    current_values = [
        float(value)
        for row in selected
        for value in (row["ci95_low"], row["ci95_high"])
    ]
    legacy_budgets = [float(row["budget"]) for row in legacy_rows]
    legacy_values = [float(row["macro_test_joint_ap"]) for row in legacy_rows]
    x_values = current_budgets + legacy_budgets
    y_values = current_values + legacy_values
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    y_min, y_max = min(y_values), max(y_values)
    padding = max(0.03, (y_max - y_min) * 0.12)
    y_min = max(0.0, y_min - padding)
    y_max = min(1.0, y_max + padding)
    if y_min == y_max:
        y_min, y_max = max(0.0, y_min - 0.05), min(1.0, y_max + 0.05)

    def xy(budget: float, value: float) -> tuple[int, int]:
        x = left + int((budget - x_min) / (x_max - x_min) * plot_width)
        y = top + int((y_max - value) / (y_max - y_min) * plot_height)
        return x, y

    draw.text((left, 30), title, fill=(20, 20, 20, 255), font=title_font)
    draw.line((left, top, left, top + plot_height), fill=(30, 30, 30, 255), width=3)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill=(30, 30, 30, 255), width=3,
    )
    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5.0
        _, y = xy(x_min, value)
        draw.line((left, y, left + plot_width, y), fill=(180, 180, 180, 100), width=1)
        label = f"{value:.2f}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (left - 20 - (box[2] - box[0]), y - (box[3] - box[1]) / 2),
            label,
            fill=(70, 70, 70, 255),
            font=tick_font,
        )
    for budget in sorted(set(current_budgets + legacy_budgets)):
        x, _ = xy(budget, y_min)
        label = f"{budget:g}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, top + plot_height + 18),
            label,
            fill=(70, 70, 70, 255),
            font=tick_font,
        )

    x_label = "Nominal VQA budget (images / task)"
    box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(
        ((width - (box[2] - box[0])) / 2, height - 55),
        x_label,
        fill=(40, 40, 40, 255),
        font=label_font,
    )

    y_label = "Macro clean-Test Joint AP"
    box = draw.textbbox((0, 0), y_label, font=label_font)
    label_image = Image.new(
        "RGBA", (box[2] - box[0] + 12, box[3] - box[1] + 12), (0, 0, 0, 0)
    )
    ImageDraw.Draw(label_image).text(
        (6 - box[0], 6 - box[1]), y_label, fill=(40, 40, 40, 255), font=label_font
    )
    label_image = label_image.rotate(90, expand=True)
    image.paste(
        label_image,
        (25, top + (plot_height - label_image.height) // 2),
        label_image,
    )

    palette = (
        (43, 108, 176, 255), (45, 146, 84, 255), (224, 122, 31, 255),
        (126, 87, 194, 255), (200, 70, 90, 255), (50, 160, 165, 255),
    )
    legend_y = top + 12
    for index, group in enumerate(sorted({str(row["group"]) for row in selected})):
        color = palette[index % len(palette)]
        group_rows = sorted(
            (row for row in selected if row["group"] == group),
            key=lambda row: int(row["budget"]),
        )
        task_count = min(int(row["task_count"]) for row in group_rows)
        declared_count = max(int(row["declared_task_count"]) for row in group_rows)
        display_group = (
            group
            if task_count == declared_count
            else f"{group} pilot ({task_count}/{declared_count} tasks)"
        )
        upper = [xy(float(row["budget"]), float(row["ci95_high"])) for row in group_rows]
        lower = [xy(float(row["budget"]), float(row["ci95_low"])) for row in reversed(group_rows)]
        if len(upper) + len(lower) >= 3:
            draw.polygon(upper + lower, fill=(*color[:3], 35))
        points = [xy(float(row["budget"]), float(row["macro_test_joint_ap"])) for row in group_rows]
        if len(points) > 1:
            draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        legend_x = left + plot_width - 455
        draw.line((legend_x, legend_y + 10, legend_x + 40, legend_y + 10), fill=color, width=5)
        draw.text(
            (legend_x + 52, legend_y),
            display_group,
            fill=(30, 30, 30, 255),
            font=legend_font,
        )
        legend_y += 34

    for label in sorted({str(row["label"]) for row in legacy_rows}):
        legacy = sorted(
            (row for row in legacy_rows if row["label"] == label),
            key=lambda row: int(row["budget"]),
        )
        points = [xy(float(row["budget"]), float(row["macro_test_joint_ap"])) for row in legacy]
        if len(points) > 1:
            draw.line(points, fill=(100, 100, 100, 180), width=3)
        for x, y in points:
            draw.line((x - 6, y - 6, x + 6, y + 6), fill=(100, 100, 100, 220), width=2)
            draw.line((x - 6, y + 6, x + 6, y - 6), fill=(100, 100, 100, 220), width=2)
        legend_x = left + plot_width - 455
        draw.text(
            (legend_x, legend_y),
            f"Historical only: {label}",
            fill=(90, 90, 90, 255),
            font=legend_font,
        )
        legend_y += 34

    output_base.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_base.with_suffix(".png"), format="PNG")
    image.save(output_base.with_suffix(".pdf"), format="PDF", resolution=150.0)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_report(
    runs: Sequence[BudgetRun],
    records: Sequence[TaskRecord],
    macro_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> str:
    lines = [
        "# VQA Budget Curve",
        "",
        "## Protocol",
        "",
        "- Metric: Ours-Full clean frozen-Test Joint AP.",
        "- Aggregation: first average seeds within each task, then take an unweighted task macro mean.",
        f"- Class-support diagnostic: ProbeBank requires at least {PROBEBANK_MIN_CLASS_SAMPLES} positive and {PROBEBANK_MIN_CLASS_SAMPLES} negative samples per attribute; insufficient balance means positive < {PROBEBANK_MIN_CLASS_SAMPLES} or negative < {PROBEBANK_MIN_CLASS_SAMPLES}. Single-class (zero positives or negatives) is retained as a stricter subset. These diagnostics do not alter AP aggregation.",
        f"- Uncertainty: {bootstrap_repetitions:,} paired task-bootstrap resamples (seed {bootstrap_seed}); the same sampled task indices are reused at every budget.",
        "- Historical curves, when shown, are context only and are never pooled with the current Formal-23 or All-36 estimates.",
        "",
        "## Input audit",
        "",
        "| Budget | Tasks | Ours-Full method ID | Logical VQA | Train | Audit | Tasks with >=1 attr below min balance | Tasks with >=1 single-class attr | Clean-Test audits | Root |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        subset = [record for record in records if record.budget == run.budget]
        methods = ", ".join(sorted({record.method_id for record in subset}))
        realized = [record.realized_budget for record in subset if record.realized_budget is not None]
        realized_text = (
            str(min(realized)) if realized and min(realized) == max(realized)
            else f"{min(realized)}-{max(realized)}" if realized
            else "unavailable"
        )
        train = [record.train_label_count for record in subset if record.train_label_count is not None]
        train_text = (
            str(min(train)) if train and min(train) == max(train)
            else f"{min(train)}-{max(train)}" if train
            else "unavailable"
        )
        audit = [record.audit_label_count for record in subset if record.audit_label_count is not None]
        audit_text = (
            str(min(audit)) if audit and min(audit) == max(audit)
            else f"{min(audit)}-{max(audit)}" if audit
            else "unavailable"
        )
        clean_count = sum(record.clean_test_verified for record in subset)
        diagnosed = [
            int(record.single_class_attribute_count)
            for record in subset
            if record.single_class_attribute_count is not None
        ]
        single_class_text = (
            f"{sum(value > 0 for value in diagnosed)}/{len(diagnosed)}"
            if diagnosed else "n/a"
        )
        diagnosed_insufficient = [
            int(record.insufficient_balance_attribute_count)
            for record in subset
            if record.insufficient_balance_attribute_count is not None
        ]
        insufficient_text = (
            f"{sum(value > 0 for value in diagnosed_insufficient)}/{len(diagnosed_insufficient)}"
            if diagnosed_insufficient else "n/a"
        )
        lines.append(
            f"| {run.budget} | {len(subset)} | `{methods}` | {realized_text} | "
            f"{train_text} | {audit_text} | {insufficient_text} | {single_class_text} | "
            f"{clean_count}/{len(subset)} | `{run.root}` |"
        )

    lines.extend(["", "## Main curves", ""])
    for cohort in ("Formal-23", "All-36"):
        subset = [
            row for row in macro_rows
            if row["group_type"] == "cohort" and row["group"] == cohort
        ]
        if not subset:
            continue
        best = max(subset, key=lambda row: float(row["macro_test_joint_ap"]))
        paired_count = int(subset[0]["task_count"])
        declared_count = int(subset[0]["declared_task_count"])
        cohort_title = (
            cohort
            if paired_count == declared_count
            else f"{cohort} pilot ({paired_count}/{declared_count} tasks)"
        )
        lines.extend([
            f"### {cohort_title}",
            "",
            "| Budget | Tasks | Macro AP | Paired 95% CI | Delta previous (95% CI) | Tasks with >=1 attr below min balance | Tasks with >=1 single-class attr |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in subset:
            delta = "n/a" if row["delta_vs_previous"] is None else (
                f"{float(row['delta_vs_previous']):+.4f} "
                f"[{fmt(row['delta_vs_previous_ci95_low'])}, {fmt(row['delta_vs_previous_ci95_high'])}]"
            )
            lines.append(
                f"| {row['budget']} | {row['task_count']} | {fmt(row['macro_test_joint_ap'])} | "
                f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {delta} | "
                f"{row['insufficient_balance_attribute_task_count'] if row['insufficient_balance_attribute_task_count'] is not None else 'n/a'} | "
                f"{row['single_class_attribute_task_count'] if row['single_class_attribute_task_count'] is not None else 'n/a'} |"
            )
        lines.extend([
            "",
            f"Descriptive peak: budget **{best['budget']}**, macro AP **{fmt(best['macro_test_joint_ap'])}**. This is not a significance claim by itself.",
            "",
        ])

    dataset = [row for row in macro_rows if row["group_type"] == "dataset"]
    if dataset:
        lines.extend([
            "## Dataset groups (All-36 membership)",
            "",
            "| Dataset | Budget | Tasks | Macro AP | 95% CI | Tasks with >=1 attr below min balance | Tasks with >=1 single-class attr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in dataset:
            lines.append(
                f"| {row['group']} | {row['budget']} | {row['task_count']} | "
                f"{fmt(row['macro_test_joint_ap'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | "
                f"{row['insufficient_balance_attribute_task_count'] if row['insufficient_balance_attribute_task_count'] is not None else 'n/a'} | "
                f"{row['single_class_attribute_task_count'] if row['single_class_attribute_task_count'] is not None else 'n/a'} |"
            )
        lines.append("")

    if legacy_rows:
        lines.extend([
            "## Historical context only",
            "",
            "The curves below use older task panels/protocols/method definitions. They are displayed only as prior evidence and must not be described as part of the current Formal-23 or All-36 result.",
            "",
            "| Curve | Tasks | Budget | Historical macro AP |",
            "|---|---:|---:|---:|",
        ])
        for row in legacy_rows:
            lines.append(
                f"| {row['label']} | {row.get('task_count') or '—'} | {row['budget']} | {fmt(row['macro_test_joint_ap'])} |"
            )
        lines.append("")

    cohort_rows = [row for row in macro_rows if row["group_type"] == "cohort"]
    if cohort_rows:
        by_cohort = {
            cohort: sorted(
                (row for row in cohort_rows if row["group"] == cohort),
                key=lambda row: int(row["budget"]),
            )
            for cohort in {str(row["group"]) for row in cohort_rows}
        }
        preferred = None
        for cohort in ("All-36", "Formal-23"):
            rows_for_cohort = by_cohort.get(cohort, [])
            if rows_for_cohort and int(rows_for_cohort[0]["task_count"]) == int(
                rows_for_cohort[0]["declared_task_count"]
            ):
                preferred = rows_for_cohort
                break
        if preferred is None:
            preferred = max(
                by_cohort.values(),
                key=lambda values: (
                    int(values[0]["task_count"])
                    / int(values[0]["declared_task_count"]),
                    int(values[0]["task_count"]),
                ),
            )
        cohort_name = str(preferred[0]["group"])
        task_count = int(preferred[0]["task_count"])
        declared_count = int(preferred[0]["declared_task_count"])
        analysis_name = (
            cohort_name
            if task_count == declared_count
            else f"{cohort_name} pilot ({task_count}/{declared_count} tasks)"
        )
        reference_budget = int(preferred[0]["reference_budget"])
        reference = next(
            (row for row in preferred if int(row["budget"]) == reference_budget),
            None,
        )
        last = preferred[-1]
        peak = max(preferred, key=lambda row: float(row["macro_test_joint_ap"]))
        post_reference = [
            row
            for row in preferred
            if int(row["budget"]) > reference_budget
            and row["delta_vs_previous"] is not None
        ]
        clearly_positive_steps = sum(
            float(row["delta_vs_previous_ci95_low"]) > 0.0
            for row in post_reference
        )
        lines.extend([
            "## Comparison with the prior ~150-label hypothesis",
            "",
            f"- Prior hypothesis: performance reaches an effective plateau near budget {PRIOR_PLATEAU_BUDGET}.",
        ])
        first = preferred[0]
        if reference is not None and int(first["budget"]) < reference_budget:
            gain_to_reference = -float(first["delta_vs_reference"])
            gain_to_reference_low = -float(first["delta_vs_reference_ci95_high"])
            gain_to_reference_high = -float(first["delta_vs_reference_ci95_low"])
            lines.append(
                f"- {analysis_name}: budget {first['budget']} to {reference_budget} changes macro AP by "
                f"{gain_to_reference:+.4f} (paired 95% CI "
                f"[{fmt(gain_to_reference_low)}, {fmt(gain_to_reference_high)}])."
            )
        if reference is not None and int(last["budget"]) > reference_budget:
            lines.append(
                f"- {analysis_name}: budget {reference_budget} to {last['budget']} changes macro AP by "
                f"{float(last['delta_vs_reference']):+.4f} "
                f"(paired 95% CI [{fmt(last['delta_vs_reference_ci95_low'])}, "
                f"{fmt(last['delta_vs_reference_ci95_high'])}])."
            )
        lines.append(
            f"- The descriptive peak among the observed budgets is budget {peak['budget']} "
            f"(macro AP {fmt(peak['macro_test_joint_ap'])})."
        )
        if post_reference:
            lines.append(
                f"- After budget {reference_budget}, {clearly_positive_steps}/{len(post_reference)} "
                "successive increments have a paired 95% CI strictly above zero."
            )
            if clearly_positive_steps == 0:
                lines.append(
                    f"- Current evidence therefore supports rapid improvement through about {reference_budget} "
                    "followed by diminishing, uncertain gains; it supports a plateau near that budget, "
                    "not a universal strict maximum at exactly that budget."
                )
            else:
                lines.append(
                    f"- At least one post-{reference_budget} increment is clearly positive, so the data do not "
                    f"support treating {reference_budget} as the saturation point for this panel."
                )
        if legacy_rows:
            historical_peaks = []
            for label in sorted({str(row["label"]) for row in legacy_rows}):
                values = [row for row in legacy_rows if row["label"] == label]
                best = max(values, key=lambda row: float(row["macro_test_joint_ap"]))
                historical_peaks.append(f"{label}: {best['budget']}")
            lines.append(
                "- Historical descriptive peak budgets (different panels/protocols): "
                + "; ".join(historical_peaks)
                + "."
            )
        full_panel = task_count == declared_count
        if not full_panel:
            lines.append(
                f"- Paper recommendation: defer the inclusion decision until the complete {cohort_name} "
                "panel is available; this pilot is descriptive only."
            )
        elif (
            reference is not None
            and int(first["budget"]) < reference_budget
            and not post_reference
        ):
            lines.append(
                f"- Paper recommendation: the gain through budget {reference_budget} can be reported, "
                f"but the plateau claim must wait for at least one budget above {reference_budget}."
            )
        elif reference is not None and int(first["budget"]) < reference_budget:
            gain_to_reference_low = -float(first["delta_vs_reference_ci95_high"])
            if gain_to_reference_low > 0.0 and clearly_positive_steps == 0:
                lines.append(
                    "- Paper recommendation: include the curve as evidence of budget efficiency and "
                    f"diminishing returns near {reference_budget}; do not claim an exact hard maximum."
                )
            elif gain_to_reference_low > 0.0:
                lines.append(
                    "- Paper recommendation: include the scaling curve, but revise the saturation claim "
                    f"because at least one increment after {reference_budget} is clearly positive."
                )
            else:
                lines.append(
                    "- Paper recommendation: do not use this as a main-paper budget-efficiency claim; "
                    "the gain up to the proposed plateau is not clearly positive."
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


LONG_FIELDS = (
    "budget", "realized_budget", "train_label_count", "audit_label_count",
    "order", "dataset", "task", "task_key", "seed",
    "method", "stage", "split", "target", "metric", "value", "joint_rule", "family",
    "aggregation", "supervision_hash", "suite_id", "ours_full_method",
    "ours_full_resolution", "clean_test_verified", "source_file",
)
SEED_FIELDS = (
    "budget", "realized_budget", "train_label_count", "audit_label_count",
    "order", "dataset", "task", "task_key", "seed",
    "method", "split", "target", "metric", "value", "suite_id", "supervision_hash",
    "min_attr_positive", "min_attr_negative", "single_class_attribute_count",
    "insufficient_balance_attribute_count",
    "joint_positive", "joint_negative",
    "clean_test_verified", "source_file",
)
TASK_FIELDS = (
    "budget", "realized_budget", "train_label_count", "audit_label_count",
    "order", "dataset", "task", "task_key", "method",
    "split", "target", "metric", "seed_count", "mean_ap", "std_seed_ap",
    "min_seed_ap", "max_seed_ap", "supervision_hash", "clean_test_verified",
    "min_attr_positive", "min_attr_negative", "single_class_attribute_count",
    "insufficient_balance_attribute_count",
    "joint_positive", "joint_negative",
)
MACRO_FIELDS = (
    "group_type", "group", "budget", "task_count", "declared_task_count",
    "paired_task_count", "seed_count_min", "seed_count_max", "macro_test_joint_ap",
    "ci95_low", "ci95_high", "reference_budget", "delta_vs_reference",
    "delta_vs_reference_ci95_low", "delta_vs_reference_ci95_high", "previous_budget",
    "delta_vs_previous", "delta_vs_previous_ci95_low", "delta_vs_previous_ci95_high",
    "realized_budget_min", "realized_budget_max", "method", "split", "target", "metric",
    "class_diagnostics_task_count", "single_class_attribute_task_count",
    "single_class_attribute_task_fraction",
    "insufficient_balance_attribute_task_count",
    "insufficient_balance_attribute_task_fraction",
)


def summarize(
    runs: Sequence[BudgetRun],
    output_dir: Path,
    *,
    ours_full_override: str | None = None,
    require_clean_test_audit: bool = True,
    allow_incomplete_panels: bool = False,
    allow_method_id_change: bool = False,
    allow_suite_id_change: bool = False,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_902,
    reference_budget: int | None = None,
    legacy_specs: Sequence[tuple[str, Path]] = (),
    render_plots: bool = True,
) -> dict[str, Any]:
    records = collect_task_records(
        runs,
        ours_full_override=ours_full_override,
        require_clean_test_audit=require_clean_test_audit,
    )
    methods = sorted({record.method_id for record in records})
    if len(methods) > 1 and not allow_method_id_change:
        raise ValueError(
            "Ours-Full method ID changes across inputs; this is not a comparable budget curve: "
            + ", ".join(methods)
        )
    suite_ids = sorted({record.suite_id for record in records})
    if len(suite_ids) > 1 and not allow_suite_id_change:
        raise ValueError(
            "suite ID changes across inputs; this is not a comparable budget curve: "
            + ", ".join(repr(value) for value in suite_ids)
        )
    if not allow_incomplete_panels and suite_ids == [""]:
        raise ValueError("strict All-36 summary requires a non-empty suite ID")
    long_rows = flatten_metrics(records)
    seed_rows = extract_main_per_task_seed(records)
    task_rows = mean_by_task(seed_rows)
    budgets = [run.budget for run in runs]
    validate_panel_coverage(
        task_rows, seed_rows, budgets, allow_incomplete=allow_incomplete_panels
    )
    macro_rows = aggregate_macro_curves(
        task_rows,
        budgets,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        reference_budget=reference_budget,
    )
    legacy_rows: list[dict[str, Any]] = []
    for label, path in legacy_specs:
        legacy_rows.extend(load_legacy_curve(label, path))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics_long.csv", LONG_FIELDS, long_rows)
    write_csv(output_dir / "per_task_seed.csv", SEED_FIELDS, seed_rows)
    write_csv(output_dir / "per_task_mean.csv", TASK_FIELDS, task_rows)
    write_csv(output_dir / "macro_curve.csv", MACRO_FIELDS, macro_rows)
    per_task_contract = {
        "schema_version": 1,
        "metric_contract": {
            "stage": "iterative",
            "split": "clean frozen test",
            "target": "joint",
            "metric": "average_precision",
        },
        "expected_budgets": list(EXPECTED_BUDGETS),
        "expected_seeds": list(EXPECTED_SEEDS),
        "strict_all36_complete": not allow_incomplete_panels,
        "supervision_diagnostics": {
            "source": "summary.supervision_audits.iterative",
            "probebank_min_class_samples": PROBEBANK_MIN_CLASS_SAMPLES,
            "insufficient_balance_attribute_definition": (
                f"positive < {PROBEBANK_MIN_CLASS_SAMPLES} or "
                f"negative < {PROBEBANK_MIN_CLASS_SAMPLES}"
            ),
            "single_class_attribute_definition": "positive == 0 or negative == 0",
            "affects_ap_aggregation": False,
        },
    }
    write_json(
        output_dir / "per_task_seed.json",
        {**per_task_contract, "aggregation": "none", "rows": seed_rows},
    )
    write_json(
        output_dir / "per_task_mean.json",
        {
            **per_task_contract,
            "aggregation": "arithmetic mean over seeds within each task-budget cell",
            "rows": task_rows,
        },
    )
    if legacy_rows:
        write_csv(
            output_dir / "legacy_curves.csv",
            ("label", "budget", "macro_test_joint_ap", "task_count", "source_file", "source_kind"),
            legacy_rows,
        )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_contract": {
            "stage": "iterative",
            "split": "clean frozen test",
            "target": "joint",
            "metric": "average_precision",
            "aggregation": "mean seeds within task, then unweighted task macro",
        },
        "supervision_diagnostics": {
            "source": "summary.supervision_audits.iterative",
            "probebank_min_class_samples": PROBEBANK_MIN_CLASS_SAMPLES,
            "insufficient_balance_attribute_definition": (
                f"positive < {PROBEBANK_MIN_CLASS_SAMPLES} or "
                f"negative < {PROBEBANK_MIN_CLASS_SAMPLES}"
            ),
            "single_class_attribute_definition": "positive == 0 or negative == 0",
            "insufficient_balance_macro_count": (
                "tasks with insufficient_balance_attribute_count > 0"
            ),
            "macro_count": "tasks with single_class_attribute_count > 0",
            "affects_ap_aggregation": False,
        },
        "bootstrap": {
            "unit": "task",
            "paired_across_budgets": True,
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
            "interval": "percentile 95%",
        },
        "ours_full_method_ids": methods,
        "suite_ids": suite_ids,
        "completeness": {
            "strict_all36_required": not allow_incomplete_panels,
            "strict_all36_complete": not allow_incomplete_panels,
            "expected_budgets": list(EXPECTED_BUDGETS),
            "expected_task_orders": list(ALL36_ORDERS),
            "expected_seeds": list(EXPECTED_SEEDS),
            "required_task_budget_cells": len(EXPECTED_BUDGETS) * len(ALL36_ORDERS),
            "required_seed_cells": (
                len(EXPECTED_BUDGETS) * len(ALL36_ORDERS) * len(EXPECTED_SEEDS)
            ),
            "observed_task_budget_cells": len(task_rows),
            "observed_seed_cells": len(seed_rows),
        },
        "prior_hypothesis": {
            "description": "performance plateaus near 150 VQA labels per task",
            "reference_budget": PRIOR_PLATEAU_BUDGET,
        },
        "runs": [
            {"budget": run.budget, "root": str(run.root)} for run in runs
        ],
        "rows": macro_rows,
        "legacy_context": legacy_rows,
        "legacy_context_is_excluded_from_current_estimates": True,
    }
    write_json(output_dir / "macro_curve.json", payload)
    (output_dir / "REPORT.md").write_text(
        build_report(
            runs,
            records,
            macro_rows,
            legacy_rows,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
        ),
        encoding="utf-8",
    )
    if render_plots:
        render_curve_plot(
            macro_rows,
            output_dir / "macro_curve",
            group_type="cohort",
            title="VQA budget vs Ours-Full clean-Test Joint AP",
            legacy_rows=legacy_rows,
        )
        render_curve_plot(
            macro_rows,
            output_dir / "dataset_curves",
            group_type="dataset",
            title="VQA budget curve by dataset (All-36)",
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-run", action="append", default=[], metavar="BUDGET=PATH",
        help="Completed run_probebank output root; repeat once per VQA budget.",
    )
    parser.add_argument(
        "--budget-runs-json", type=Path,
        help="Optional JSON list (or {runs: [...]}) with budget/path entries.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ours-full-method",
        help="Explicit method ID. By default it is resolved from each suite and output.",
    )
    parser.add_argument(
        "--legacy-curve", action="append", default=[], metavar="LABEL=CSV",
        help="Optional old 6/10-task curve for historical context only.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_902)
    parser.add_argument(
        "--reference-budget", type=int,
        help="Budget used for plateau deltas; defaults to 150 when present.",
    )
    parser.add_argument(
        "--allow-incomplete-panels", action="store_true",
        help="Allow partial dry runs; Formal-23/All-36 task counts remain explicit.",
    )
    parser.add_argument(
        "--allow-missing-clean-test-audit", action="store_true",
        help="Do not fail when zero-overlap test audit metadata is missing.",
    )
    parser.add_argument(
        "--allow-method-id-change", action="store_true",
        help="Permit different Ours-Full IDs across runs (not recommended).",
    )
    parser.add_argument(
        "--allow-suite-id-change", action="store_true",
        help="Permit different suite IDs across runs (not recommended).",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = load_budget_runs(args.budget_run, args.budget_runs_json)
    legacy_specs = [
        (label, Path(path).resolve())
        for label, path in (parse_assignment(value, "--legacy-curve") for value in args.legacy_curve)
    ]
    summarize(
        runs,
        args.output_dir.resolve(),
        ours_full_override=args.ours_full_method,
        require_clean_test_audit=not args.allow_missing_clean_test_audit,
        allow_incomplete_panels=args.allow_incomplete_panels,
        allow_method_id_change=args.allow_method_id_change,
        allow_suite_id_change=args.allow_suite_id_change,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        reference_budget=args.reference_budget,
        legacy_specs=legacy_specs,
        render_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
