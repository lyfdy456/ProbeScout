"""Run one deterministic 100+8x50 iterative-VQA trajectory per task.

The acquisition policy is the production iterative sampler.  Historical VQA
answers may be reused *after* a candidate is selected; every reused answer
still consumes one logical VQA-budget unit.  This avoids paying twice for the
same image without exposing unselected candidate labels to the sampler.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).with_name("run_retrieval_harness.py")
DEFAULT_STAGE = "iterative_vqa_100_50_v3_budget500"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def task_state(task: dict[str, Any], stage: str) -> Path:
    import run_retrieval_harness as harness

    harness.configure(task["dataset"], task["task"], task.get("joint_label"))
    return harness.ADAPTER.task_root / "split" / stage / "round_state.json"


def completed_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds") or []
    if payload.get("status") != "completed" or len(rounds) != 9:
        return None
    final = rounds[-1]
    if int(final.get("total_labeled", -1)) != 500:
        return None
    if int(final.get("n_train", -1)) != 420 or int(final.get("n_audit", -1)) != 80:
        return None
    return payload


def approved_preflight_rounds(
    approved_manifest: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    stage: str,
) -> dict[int, int]:
    """Return the approved round for each requested task, with strict identity checks."""
    if str(approved_manifest.get("stage")) != str(stage):
        raise ValueError(
            "approved preflight stage mismatch: "
            f"{approved_manifest.get('stage')!r} != {stage!r}"
        )
    summaries = approved_manifest.get("tasks")
    if not isinstance(summaries, list):
        raise ValueError("approved preflight tasks must be a list")
    if int(approved_manifest.get("task_count", -1)) != len(summaries):
        raise ValueError("approved preflight task_count does not match tasks")

    rows = approved_manifest.get("rows") or []
    row_counts: dict[tuple[int, str, str, int], int] = {}
    for row in rows:
        key = (
            int(row["order"]),
            str(row["dataset"]),
            str(row["task"]),
            int(row["round"]),
        )
        row_counts[key] = row_counts.get(key, 0) + 1

    summary_by_order: dict[int, dict[str, Any]] = {}
    for summary in summaries:
        order = int(summary["order"])
        if order in summary_by_order:
            raise ValueError(f"approved preflight has duplicate task order {order}")
        round_no = int(summary["round"])
        if round_no < 0:
            raise ValueError(
                f"approved preflight has a negative round for task order {order}"
            )
        key = (
            order,
            str(summary["dataset"]),
            str(summary["task"]),
            round_no,
        )
        observed_rows = row_counts.get(key, 0)
        if observed_rows <= 0 or int(summary.get("row_count", -1)) != observed_rows:
            raise ValueError(
                "approved preflight task summary does not match its rows: "
                f"order={order}, declared={summary.get('row_count')!r}, "
                f"observed={observed_rows}"
            )
        summary_by_order[order] = summary

    requested = {int(task["order"]): task for task in tasks}
    approved: dict[int, int] = {}
    for order, task in requested.items():
        summary = summary_by_order.get(order)
        if summary is None:
            continue
        if (
            str(summary.get("dataset")) != str(task["dataset"])
            or str(summary.get("task")) != str(task["task"])
        ):
            raise ValueError(
                "approved preflight task identity mismatch: "
                f"order={order}, approved="
                f"{summary.get('dataset')!r}/{summary.get('task')!r}, runtime="
                f"{task['dataset']!r}/{task['task']!r}"
            )
        approved[order] = int(summary["round"])
    return approved


def committed_preflight_result(
    task: dict[str, Any],
    *,
    state_path: Path,
    stage: str,
    approved_round: int,
) -> dict[str, Any] | None:
    """Describe a preflight batch already committed in the task state.

    A committed round is an immutable acquisition boundary.  Reusing the same
    outbound approval after that boundary must therefore be a no-op; otherwise
    the harness would select a later round that the approval did not cover.
    """
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read acquisition state {state_path}: {error}") from error
    if not isinstance(state, dict):
        raise ValueError(f"acquisition state must be a JSON object: {state_path}")

    expected_identity = {
        "stage": str(stage),
        "dataset": str(task["dataset"]),
        "task": str(task["task"]),
    }
    for field, expected in expected_identity.items():
        observed = state.get(field)
        if str(observed) != expected:
            raise ValueError(
                f"acquisition state {field} mismatch for order {task['order']}: "
                f"{observed!r} != {expected!r}"
            )

    rounds = state.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError(f"acquisition state rounds must be a list: {state_path}")
    observed_rounds: list[int] = []
    for index, record in enumerate(rounds):
        if not isinstance(record, dict):
            raise ValueError(
                f"acquisition state round {index} must be an object: {state_path}"
            )
        round_no = int(record.get("round", -1))
        observed_rounds.append(round_no)
        if round_no != index:
            raise ValueError(
                "acquisition state rounds are not contiguous: "
                f"expected={index}, observed={round_no}, state={state_path}"
            )
        if str(record.get("stage")) != str(stage):
            raise ValueError(
                f"acquisition state round {round_no} stage mismatch: "
                f"{record.get('stage')!r} != {stage!r}"
            )
        if record.get("complete") is not True:
            raise ValueError(
                f"acquisition state contains an uncommitted round record: "
                f"round={round_no}, state={state_path}"
            )

    if int(approved_round) not in observed_rounds:
        return None
    final = rounds[-1]
    return {
        "order": int(task["order"]),
        "dataset": task["dataset"],
        "task": task["task"],
        "status": "reused_preflight_batch_completed",
        "state_status": state.get("status"),
        "approved_preflight_round": int(approved_round),
        "rounds": len(rounds),
        "logical_vqa": final.get("total_labeled"),
        "train": final.get("n_train"),
        "audit": final.get("n_audit"),
        "physical_new_calls": sum(int(row.get("n_new", 0)) for row in rounds),
        "elapsed_seconds": 0.0,
        "state": str(state_path),
    }


def run_task(
    task: dict[str, Any],
    *,
    state_path: Path,
    stage: str,
    log_root: Path,
    allow_vqa: bool,
    vqa_config: str,
    vqa_workers: int,
    vqa_python: str | None,
    approved_preflight: Path | None,
    approved_preflight_sha256: str | None,
    committed_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    order = int(task["order"])
    task_id = f"{order:03d}_{task['dataset']}_{task['task']}"
    existing = completed_state(state_path)
    if existing is not None:
        return {
            "order": order,
            "dataset": task["dataset"],
            "task": task["task"],
            "status": "reused_completed",
            "state": str(state_path),
            "logical_vqa": 500,
            "train": 420,
            "audit": 80,
        }
    if committed_preflight is not None:
        return dict(committed_preflight)

    command = [
        str(Path(sys.executable).resolve()),
        "-u",
        str(HARNESS),
        "--dataset",
        task["dataset"],
        "--task",
        task["task"],
        "--backbone",
        "siglip",
        "--iterative-vqa",
        "--iterative-exclude-provider-content-rejections",
        "--iterative-stage",
        stage,
        "--iterative-max-rounds",
        "8",
        "--iterative-stop-at-labels",
        "500",
        "--iterative-max-labels",
        "500",
        "--iterative-resume",
        "--iterative-defer-round-evaluation",
    ]
    if task.get("joint_label"):
        command.extend(["--joint-label", str(task["joint_label"])])
    if allow_vqa:
        resolved_vqa_config = str(Path(vqa_config).resolve())
        command.extend(
            [
                "--auto-vqa",
                "--vqa-config",
                resolved_vqa_config,
                "--vqa-workers",
                str(vqa_workers),
            ]
        )
        if vqa_python:
            command.extend(["--vqa-python", str(Path(vqa_python).resolve())])
        if approved_preflight is None or not approved_preflight_sha256:
            raise ValueError("approved VQA preflight path/hash are required for external calls")
        command.extend(
            [
                "--vqa-preflight-manifest",
                str(approved_preflight.resolve()),
                "--vqa-preflight-sha256",
                str(approved_preflight_sha256),
                "--vqa-preflight-task-order",
                str(order),
            ]
        )

    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{task_id}.log"
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {command!r}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    state = completed_state(state_path)
    observed = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if result.returncode != 0:
        status = "failed"
    elif state is not None:
        status = "completed"
    elif observed.get("status") in {"preflight_batch_completed", "awaiting_vqa"}:
        status = str(observed["status"])
    else:
        status = "failed"
    final = (observed.get("rounds") or [{}])[-1]
    return {
        "order": order,
        "dataset": task["dataset"],
        "task": task["task"],
        "status": status,
        "returncode": int(result.returncode),
        "state_status": observed.get("status"),
        "rounds": len(observed.get("rounds") or []),
        "logical_vqa": final.get("total_labeled"),
        "train": final.get("n_train"),
        "audit": final.get("n_audit"),
        "physical_new_calls": sum(
            int(row.get("n_new", 0)) for row in observed.get("rounds") or []
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "state": str(state_path),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--orders", type=int, nargs="+", default=None)
    parser.add_argument("--allow-vqa", action="store_true")
    parser.add_argument(
        "--vqa-config",
        default=str(ROOT / "attribute_annotation" / "config.yaml"),
    )
    parser.add_argument("--vqa-workers", type=int, default=32)
    parser.add_argument("--vqa-python", default=None)
    parser.add_argument(
        "--export-vqa-preflight",
        type=Path,
        default=None,
        help="write a credential-free manifest of current cache-missing rows and exit",
    )
    parser.add_argument(
        "--approved-vqa-preflight",
        type=Path,
        default=None,
        help="preflight manifest approved for --allow-vqa",
    )
    parser.add_argument(
        "--approved-vqa-preflight-sha256",
        default=None,
        help="exact manifest_sha256 printed by --export-vqa-preflight",
    )
    parser.add_argument(
        "--overwrite-vqa-preflight",
        action="store_true",
        help="explicitly replace a divergent --export-vqa-preflight destination",
    )
    parser.add_argument(
        "--materialize-missing-attributes-from-manifest",
        action="store_true",
        help=(
            "before preflight export, create only missing qa/attributes.txt files "
            "after exact all-36 manifest/runtime validation; never invokes extraction"
        ),
    )
    parser.add_argument(
        "--overwrite-public-vqa-config",
        action="store_true",
        help="explicitly replace a divergent credential-free config during export",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.export_vqa_preflight and args.allow_vqa:
        raise SystemExit("--export-vqa-preflight is no-network and cannot use --allow-vqa")
    if args.allow_vqa and (
        args.approved_vqa_preflight is None
        or not args.approved_vqa_preflight_sha256
    ):
        raise SystemExit(
            "--allow-vqa requires --approved-vqa-preflight and "
            "--approved-vqa-preflight-sha256"
        )
    if not args.allow_vqa and (
        args.approved_vqa_preflight is not None
        or args.approved_vqa_preflight_sha256 is not None
    ):
        raise SystemExit("approved preflight arguments require --allow-vqa")
    if args.overwrite_vqa_preflight and args.export_vqa_preflight is None:
        raise SystemExit("--overwrite-vqa-preflight requires --export-vqa-preflight")
    if (
        args.materialize_missing_attributes_from_manifest
        and args.export_vqa_preflight is None
    ):
        raise SystemExit(
            "--materialize-missing-attributes-from-manifest requires "
            "--export-vqa-preflight"
        )
    if args.overwrite_public_vqa_config and args.export_vqa_preflight is None:
        raise SystemExit("--overwrite-public-vqa-config requires --export-vqa-preflight")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = list(manifest["tasks"])
    if args.orders:
        requested = set(args.orders)
        tasks = [task for task in tasks if int(task["order"]) in requested]
        missing = requested.difference(int(task["order"]) for task in tasks)
        if missing:
            raise SystemExit(f"orders absent from manifest: {sorted(missing)}")
    tasks.sort(key=lambda row: int(row["order"]))
    # ``run_retrieval_harness`` keeps the configured adapter in module globals.
    # Resolve task roots serially before worker threads launch subprocesses.
    state_paths = {
        int(task["order"]): task_state(task, args.stage)
        for task in tasks
    }

    source_vqa_config = Path(args.vqa_config)
    if not source_vqa_config.is_absolute():
        import run_retrieval_harness as harness

        source_vqa_config = harness.VQA_DIR / source_vqa_config
    source_vqa_config = source_vqa_config.resolve()
    resolved_vqa_config = source_vqa_config

    if args.export_vqa_preflight is not None:
        from vqa_budget_preflight import (
            build_preflight_manifest,
            materialize_public_vqa_config,
            materialize_task_attribute_files,
            validate_strict_cache_preflight_state,
            write_manifest,
        )

        strict_cache_results = validate_strict_cache_preflight_state(
            tasks,
            state_paths=state_paths,
            stage=args.stage,
        )
        write_json(
            args.output / "preflight_strict_cache_audit.json",
            {"network_used": False, "tasks": strict_cache_results},
        )

        public_config_path = (args.output / "public_vqa_config.yaml").resolve()
        public_config_result = materialize_public_vqa_config(
            source_vqa_config,
            public_config_path,
            overwrite=args.overwrite_public_vqa_config,
        )
        resolved_vqa_config = public_config_path
        write_json(
            args.output / "public_vqa_config_audit.json",
            {"network_used": False, **public_config_result},
        )

        if args.materialize_missing_attributes_from_manifest:
            attribute_results = materialize_task_attribute_files(tasks)
            write_json(
                args.output / "preflight_attribute_materialization.json",
                {
                    "network_used": False,
                    "tasks": attribute_results,
                    "created": sum(
                        row["status"] == "created" for row in attribute_results
                    ),
                    "reused_exact": sum(
                        row["status"] == "reused_exact" for row in attribute_results
                    ),
                },
            )

        preflight = build_preflight_manifest(
            tasks,
            state_paths=state_paths,
            stage=args.stage,
            vqa_config=resolved_vqa_config,
        )
        sealed = write_manifest(
            args.export_vqa_preflight,
            preflight,
            overwrite=args.overwrite_vqa_preflight,
        )
        print(
            f"VQA preflight written: {args.export_vqa_preflight.resolve()}\n"
            f"rows={sealed['row_count']} tasks={sealed['task_count']}\n"
            f"manifest_sha256={sealed['manifest_sha256']}\n"
            f"public_vqa_config={resolved_vqa_config}\n"
            "No network calls were made.",
            flush=True,
        )
        return

    committed_preflight_results: dict[int, dict[str, Any]] = {}
    if args.allow_vqa:
        from vqa_budget_preflight import (
            load_and_validate_manifest,
            validate_external_credential_source,
        )

        approved_manifest = load_and_validate_manifest(
            args.approved_vqa_preflight,
            args.approved_vqa_preflight_sha256,
        )
        approved_config_path = approved_manifest.get("provider", {}).get("config_path")
        if not approved_config_path:
            raise SystemExit("approved preflight lacks an explicit public config path")
        resolved_vqa_config = Path(approved_config_path).resolve()
        validate_external_credential_source(resolved_vqa_config)
        try:
            approved_rounds = approved_preflight_rounds(
                approved_manifest,
                tasks,
                stage=args.stage,
            )
            for task in tasks:
                order = int(task["order"])
                if order not in approved_rounds:
                    continue
                reused = committed_preflight_result(
                    task,
                    state_path=state_paths[order],
                    stage=args.stage,
                    approved_round=approved_rounds[order],
                )
                if reused is not None:
                    committed_preflight_results[order] = reused
        except ValueError as error:
            raise SystemExit(str(error)) from error

    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_task,
                task,
                state_path=state_paths[int(task["order"])],
                stage=args.stage,
                log_root=args.output / "logs",
                allow_vqa=args.allow_vqa,
                vqa_config=str(resolved_vqa_config),
                vqa_workers=args.vqa_workers,
                vqa_python=args.vqa_python,
                approved_preflight=args.approved_vqa_preflight,
                approved_preflight_sha256=args.approved_vqa_preflight_sha256,
                committed_preflight=committed_preflight_results.get(
                    int(task["order"])
                ),
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda row: int(row["order"]))
            write_json(
                args.output / "acquisition_state.json",
                {
                    "stage": args.stage,
                    "logical_budget_definition": "unique selected VQA images including permanent audit",
                    "cache_policy": "answers_reused_only_after_iterative_selection",
                    "tasks_requested": len(tasks),
                    "tasks_completed": sum(
                        row["status"] in {"completed", "reused_completed"}
                        for row in results
                    ),
                    "results": results,
                },
            )
            print(
                f"[{len(results)}/{len(tasks)}] {result['order']:03d} "
                f"{result['task']}: {result['status']} "
                f"budget={result.get('logical_vqa')}",
                flush=True,
            )

    failed = [row for row in results if row["status"] == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} acquisition tasks failed")


if __name__ == "__main__":
    main()
