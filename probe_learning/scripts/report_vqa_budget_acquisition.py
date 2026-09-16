"""Build a read-only accounting snapshot for iterative-VQA acquisition.

The acquisition controller intentionally separates logical budget from paid
provider work.  This reporter keeps those units separate and only consumes
immutable/append-only evidence:

* round state and the current frozen manifest for logical selection;
* stage-local ``*_results.jsonl`` files for terminal provider operations;
* controller logs for explicit application-level retries and tqdm timing;
* outbound preflight manifests for wave attribution; and
* rejection ledgers for non-budgeted replacements.

No provider is contacted and no acquisition state or cache file is modified.
When output paths are requested, only new report artifacts are written.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = "iterative_vqa_100_50_v3_budget500"

_COMMAND_HEADER = re.compile(r"^\[(?P<timestamp>[^]]+)]\s+(?P<command>\[.*])\s*$")
_OUTPUT_FILE = re.compile(
    r"(?:\[init]\s+output file:|\[done]\s+results saved to:)\s*(?P<path>.+?)\s*$"
)
_DONE = re.compile(r"\[done]\s+success:\s*(?P<success>\d+),\s*failed:\s*(?P<failed>\d+)")
_RETRY = re.compile(r"\[retry\s+\d+/\d+]", re.IGNORECASE)
_PROGRESS = re.compile(
    r"VQA labeling:.*?\[(?P<elapsed>\d+:\d{2}(?::\d{2})?)<"
)
_DATA_INSPECTION_FIELD = re.compile(
    r"[\"'](?:code|type)[\"']\s*:\s*[\"']data_inspection_failed[\"']",
    re.IGNORECASE,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _normalize_image(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def _task_image_key(task: dict[str, Any], image: Any) -> tuple[int, str]:
    return int(task["order"]), _normalize_image(image)


def _uniq(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(_normalize_image(value) for value in values if str(value).strip()))


def _stable_event_key(event: dict[str, Any]) -> str:
    explicit = event.get("event_sha256")
    if explicit:
        return str(explicit)
    payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    # Provider rows are historically local-naive while command headers carry
    # +0800.  For within-log attribution, comparing their local wall values is
    # the only faithful operation.
    return parsed.replace(tzinfo=None)


def _format_datetime(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _elapsed_text_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"unsupported tqdm elapsed time: {value!r}")


def _flag_value(command: list[Any], flag: str) -> str | None:
    command = [str(value) for value in command]
    try:
        index = command.index(flag)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _is_data_inspection_failure(row: dict[str, Any]) -> bool:
    if row.get("error_code") == "data_inspection_failed":
        return True
    error = row.get("error")
    if isinstance(error, dict):
        if error.get("code") == "data_inspection_failed" or error.get("type") == "data_inspection_failed":
            return True
        nested = error.get("error")
        return isinstance(nested, dict) and (
            nested.get("code") == "data_inspection_failed"
            or nested.get("type") == "data_inspection_failed"
        )
    text = str(error or "").strip()
    return text == "data_inspection_failed" or bool(_DATA_INSPECTION_FIELD.search(text))


def _is_success_row(row: dict[str, Any]) -> bool:
    answer = row.get("answer")
    return isinstance(answer, str) and bool(answer.strip()) and answer.strip() != "[FAILED]"


def _is_failed_row(row: dict[str, Any]) -> bool:
    return str(row.get("answer") or "").strip() == "[FAILED]"


def _parse_log(log_path: Path, task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    if not log_path.is_file():
        return [], []
    warnings: list[str] = []
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish() -> None:
        nonlocal current
        # ``--auto-vqa`` may fail closed at preflight validation before the
        # labeler starts.  Count only segments with concrete labeler evidence,
        # otherwise a safety rejection would be misreported as a paid batch.
        if current is not None and (
            current["result_paths"]
            or current["explicit_retries"]
            or current["vqa_elapsed_seconds"]
            or current["done"]
        ):
            current["result_paths"] = sorted(current["result_paths"])
            segments.append(current)
        current = None

    for line_no, line in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        match = _COMMAND_HEADER.match(line)
        if match:
            finish()
            try:
                command = ast.literal_eval(match.group("command"))
                if not isinstance(command, list):
                    raise ValueError("command is not a list")
            except (SyntaxError, ValueError) as error:
                warnings.append(f"{log_path}:{line_no}: cannot parse command header: {error}")
                command = []
            current = {
                "order": int(task["order"]),
                "dataset": str(task["dataset"]),
                "task": str(task["task"]),
                "log": str(log_path.resolve()),
                "line": int(line_no),
                "started_at_local": match.group("timestamp"),
                "started_datetime": _parse_datetime(match.group("timestamp")),
                "auto_vqa": "--auto-vqa" in [str(value) for value in command],
                "preflight_sha256": _flag_value(command, "--vqa-preflight-sha256"),
                "result_paths": set(),
                "explicit_retries": 0,
                "vqa_elapsed_seconds": 0.0,
                "done": False,
                "done_success": None,
                "done_failed": None,
                "row_timestamps": [],
            }
            continue
        if current is None:
            continue
        path_match = _OUTPUT_FILE.search(line)
        if path_match:
            current["result_paths"].add(_path_key(path_match.group("path").strip()))
        current["explicit_retries"] += len(_RETRY.findall(line))
        progress_match = _PROGRESS.search(line)
        if progress_match:
            current["vqa_elapsed_seconds"] = max(
                current["vqa_elapsed_seconds"],
                _elapsed_text_seconds(progress_match.group("elapsed")),
            )
        done_match = _DONE.search(line)
        if done_match:
            current["done"] = True
            current["done_success"] = int(done_match.group("success"))
            current["done_failed"] = int(done_match.group("failed"))
    finish()
    return segments, warnings


def _preflight_catalog(acquisition_root: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    hash_to_wave: dict[str, str] = {}
    waves: dict[str, dict[str, Any]] = {}
    for path in sorted(acquisition_root.glob("outbound_preflight_wave*.json")):
        data = _read_json(path)
        # Keep supplemental manifests (for example ``wave04_replacement01``)
        # distinct from their base wave.  Collapsing the name to ``wave04``
        # lets the last matching file silently overwrite the base preflight.
        wave = path.stem.removeprefix("outbound_preflight_").lower()
        digest = str(data.get("manifest_sha256") or "")
        if digest:
            if digest in hash_to_wave and hash_to_wave[digest] != wave:
                raise ValueError(f"preflight hash appears under two waves: {digest}")
            hash_to_wave[digest] = wave
        approved_keys = {
            (int(row["order"]), _normalize_image(row.get("logical_image")))
            for row in data.get("rows") or []
            if isinstance(row, dict) and row.get("logical_image") is not None
        }
        waves[wave] = {
            "wave": wave,
            "manifest": str(path.resolve()),
            "manifest_sha256": digest or None,
            "created_at_utc": data.get("created_at_utc"),
            "approved_task_count": int(data.get("task_count", 0)),
            "approved_row_count": int(data.get("row_count", len(approved_keys))),
            "approved_unique_task_images": approved_keys,
        }
    return hash_to_wave, waves


def _wave_for_hash(digest: str | None, hash_to_wave: dict[str, str]) -> str:
    if not digest:
        return "legacy_unsealed"
    return hash_to_wave.get(digest, f"unknown_preflight_{digest[:12]}")


def _evidence_files(
    tasks: list[dict[str, Any]], tasks_root: Path, stage: str, acquisition_root: Path
) -> list[Path]:
    paths = list(acquisition_root.glob("outbound_preflight_wave*.json"))
    for task in tasks:
        root = tasks_root / str(task["dataset"]) / str(task["task"])
        paths.append(root / "split" / stage / "round_state.json")
        paths.extend((root / "qa" / stage).glob("round_*/manifest.json"))
        paths.extend((root / "qa" / stage).glob("round_*/*_results.jsonl"))
        paths.extend((root / "qa" / stage).glob("round_*/provider_content_rejection_audit.json"))
        paths.append(
            acquisition_root
            / "logs"
            / f"{int(task['order']):03d}_{task['dataset']}_{task['task']}.log"
        )
    return sorted({path.resolve() for path in paths if path.is_file()}, key=str)


def _file_fingerprints(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    result = {}
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        result[str(path.resolve())] = (int(stat.st_size), int(stat.st_mtime_ns))
    return result


def _collect_task_state(
    task: dict[str, Any], tasks_root: Path, stage: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    task_root = tasks_root / str(task["dataset"]) / str(task["task"])
    state_path = task_root / "split" / stage / "round_state.json"
    qa_root = task_root / "qa" / stage
    warnings: list[str] = []
    if state_path.is_file():
        state = _read_json(state_path)
    else:
        state = {"rounds": [], "status": "missing_state"}
        warnings.append(f"missing state: {state_path}")
    rounds = list(state.get("rounds") or [])
    committed = _uniq(
        image
        for record in rounds
        for partition in ("train", "audit")
        for image in record.get(partition, [])
    )
    final_total = int((rounds[-1] if rounds else {}).get("total_labeled", len(committed)))
    if final_total != len(committed):
        warnings.append(
            f"order {task['order']}: final total_labeled={final_total} but committed unique={len(committed)}"
        )
    round_numbers = [int(record.get("round", -1)) for record in rounds]
    if round_numbers != list(range(len(rounds))):
        warnings.append(f"order {task['order']}: committed rounds are not contiguous: {round_numbers}")

    pending_manifest: dict[str, Any] = {}
    pending_path: Path | None = None
    pending_round = state.get("pending_round")
    if pending_round is not None:
        pending_path = qa_root / f"round_{int(pending_round):02d}" / "manifest.json"
        if pending_path.is_file():
            pending_manifest = _read_json(pending_path)
        else:
            warnings.append(f"missing pending manifest: {pending_path}")
    pending = _uniq(
        list(pending_manifest.get("train") or [])
        + list(pending_manifest.get("audit") or [])
    )
    overlap = sorted(set(committed).intersection(pending))
    if overlap:
        warnings.append(
            f"order {task['order']}: {len(overlap)} committed/pending images overlap"
        )

    manifests: list[dict[str, Any]] = []
    for path in sorted(qa_root.glob("round_*/manifest.json")):
        try:
            manifests.append({"path": path, "data": _read_json(path)})
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"cannot read {path}: {error}")
    return (
        {
            "task_root": task_root,
            "qa_root": qa_root,
            "state_path": state_path,
            "state": state,
            "rounds": rounds,
            "committed": committed,
            "pending": pending,
            "pending_manifest": pending_manifest,
            "pending_manifest_path": pending_path,
        },
        pending_manifest,
        manifests,
        warnings,
    )


def _iter_provider_events(state: dict[str, Any], manifests: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in state.get("rounds") or []:
        for event in record.get("provider_request_history") or []:
            if isinstance(event, dict):
                yield event
    for wrapper in manifests:
        for event in wrapper["data"].get("provider_request_history") or []:
            if isinstance(event, dict):
                yield event


def _collect_rejection_evidence(
    state: dict[str, Any], manifests: list[dict[str, Any]]
) -> tuple[set[str], set[tuple[str, str]]]:
    rejected = set(_uniq(state.get("provider_rejected_not_budgeted") or []))
    replacements: set[tuple[str, str]] = set()
    for record in list(state.get("rounds") or []) + [wrapper["data"] for wrapper in manifests]:
        rejected.update(_uniq(record.get("rejected_not_budgeted") or []))
        for old, new in (record.get("replacement_map") or {}).items():
            replacements.add((_normalize_image(old), _normalize_image(new)))
        for audit in record.get("content_rejection_audits") or []:
            if not isinstance(audit, dict):
                continue
            rejected.update(_uniq(audit.get("rejected_not_budgeted") or []))
            for old, new in (audit.get("replacement_map") or {}).items():
                replacements.add((_normalize_image(old), _normalize_image(new)))
    return rejected, replacements


def _assign_segment(
    candidates: list[dict[str, Any]], timestamp: datetime | None
) -> dict[str, Any] | None:
    if not candidates:
        return None
    if timestamp is not None:
        before = [
            segment
            for segment in candidates
            if segment.get("started_datetime") is not None
            and segment["started_datetime"] <= timestamp
        ]
        if before:
            return max(before, key=lambda segment: segment["started_datetime"])
    waves = {segment["wave"] for segment in candidates}
    return candidates[-1] if len(waves) == 1 else None


def _new_wave_row(wave: str) -> dict[str, Any]:
    return {
        "wave": wave,
        "manifest": None,
        "manifest_sha256": None,
        "created_at_utc": None,
        "approved_task_count": 0,
        "approved_row_count": 0,
        "approved_unique_task_images": set(),
        "provider_invocation_segments": 0,
        "completed_vqa_batches": 0,
        "incomplete_vqa_batches": 0,
        "explicit_application_retries_logged": 0,
        "provider_labeler_elapsed_seconds_logged": 0.0,
        "terminal_records": 0,
        "successful_terminal_records": 0,
        "failed_terminal_records": 0,
        "unknown_terminal_records": 0,
        "content_rejection_terminal_records": 0,
        "attempted_task_images": set(),
        "successful_task_images": set(),
        "content_rejected_task_images": set(),
        "row_timestamps": [],
        "segment_lower_bounds": [],
    }


def build_report(
    *,
    manifest_path: Path,
    stage: str,
    acquisition_root: Path,
    tasks_root: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    acquisition_root = acquisition_root.resolve()
    tasks_root = tasks_root.resolve()
    manifest = _read_json(manifest_path)
    tasks = sorted(list(manifest["tasks"]), key=lambda row: int(row["order"]))
    if len({int(task["order"]) for task in tasks}) != len(tasks):
        raise ValueError("task manifest contains duplicate order values")

    starting_files = _evidence_files(tasks, tasks_root, stage, acquisition_root)
    starting_fingerprints = _file_fingerprints(starting_files)
    hash_to_wave, preflight_waves = _preflight_catalog(acquisition_root)

    segments: list[dict[str, Any]] = []
    task_contexts: dict[int, dict[str, Any]] = {}
    task_manifests: dict[int, list[dict[str, Any]]] = {}
    all_warnings: list[str] = []
    provider_events: dict[str, dict[str, Any]] = {}
    source_wave_fallback: dict[str, set[str]] = defaultdict(set)

    for task in tasks:
        context, _, manifests, warnings = _collect_task_state(task, tasks_root, stage)
        task_contexts[int(task["order"])] = context
        task_manifests[int(task["order"])] = manifests
        all_warnings.extend(warnings)
        log_path = (
            acquisition_root
            / "logs"
            / f"{int(task['order']):03d}_{task['dataset']}_{task['task']}.log"
        )
        parsed, warnings = _parse_log(log_path, task)
        for segment in parsed:
            segment["wave"] = _wave_for_hash(segment.get("preflight_sha256"), hash_to_wave)
        segments.extend(parsed)
        all_warnings.extend(warnings)
        for event in _iter_provider_events(context["state"], manifests):
            # Event hashes from legacy migrations did not include task
            # identity, so equal image names in two tasks must remain distinct.
            provider_events.setdefault(
                f"{int(task['order'])}:{_stable_event_key(event)}",
                {"order": int(task["order"]), **event},
            )
    for event in provider_events.values():
        digest = event.get("preflight_manifest_sha256")
        wave = _wave_for_hash(str(digest) if digest else None, hash_to_wave)
        for source in event.get("provider_result_sources") or []:
            if isinstance(source, dict) and source.get("path"):
                source_wave_fallback[_path_key(source["path"])].add(wave)

    path_segments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        for path in segment["result_paths"]:
            path_segments[path].append(segment)
    for candidates in path_segments.values():
        candidates.sort(
            key=lambda segment: segment.get("started_datetime") or datetime.min
        )

    waves: dict[str, dict[str, Any]] = {
        wave: {**_new_wave_row(wave), **row}
        for wave, row in preflight_waves.items()
    }
    raw_rows: list[dict[str, Any]] = []
    malformed_result_rows = 0
    result_files = 0
    mapped_result_files: set[str] = set()
    unattributed_result_files: set[str] = set()
    task_result_stats: dict[int, Counter] = defaultdict(Counter)
    task_result_sets: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {"attempted": set(), "success": set(), "content": set()}
    )
    task_result_files: dict[int, list[str]] = defaultdict(list)

    for task in tasks:
        order = int(task["order"])
        qa_root = task_contexts[order]["qa_root"]
        for path in sorted(qa_root.glob("round_*/*_results.jsonl")):
            result_files += 1
            resolved = str(path.resolve())
            key = _path_key(path)
            task_result_files[order].append(resolved)
            candidates = path_segments.get(key, [])
            if candidates:
                mapped_result_files.add(resolved)
            else:
                fallback = source_wave_fallback.get(key, set())
                if len(fallback) != 1:
                    unattributed_result_files.add(resolved)
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for row_no, line in enumerate(handle, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        malformed_result_rows += 1
                        continue
                    if not isinstance(row, dict):
                        malformed_result_rows += 1
                        continue
                    image = _normalize_image(row.get("image"))
                    timestamp = _parse_datetime(row.get("timestamp"))
                    segment = _assign_segment(candidates, timestamp)
                    if segment is not None:
                        wave = segment["wave"]
                        segment["row_timestamps"].append(timestamp)
                    else:
                        fallback = source_wave_fallback.get(key, set())
                        wave = next(iter(fallback)) if len(fallback) == 1 else "unattributed"
                    if wave not in waves:
                        waves[wave] = _new_wave_row(wave)
                    success = _is_success_row(row)
                    failed = _is_failed_row(row)
                    content = failed and _is_data_inspection_failure(row)
                    record = {
                        "order": order,
                        "image": image,
                        "wave": wave,
                        "success": success,
                        "failed": failed,
                        "content_rejection": content,
                        "timestamp": timestamp,
                        "source": resolved,
                        "source_row": row_no,
                    }
                    raw_rows.append(record)
                    counter = task_result_stats[order]
                    counter["terminal_records"] += 1
                    counter["successful_terminal_records" if success else "failed_terminal_records" if failed else "unknown_terminal_records"] += 1
                    counter["content_rejection_terminal_records"] += int(content)
                    sets = task_result_sets[order]
                    sets["attempted"].add(image)
                    if success:
                        sets["success"].add(image)
                    if content:
                        sets["content"].add(image)
                    wave_row = waves[wave]
                    wave_row["terminal_records"] += 1
                    wave_row["successful_terminal_records" if success else "failed_terminal_records" if failed else "unknown_terminal_records"] += 1
                    wave_row["content_rejection_terminal_records"] += int(content)
                    wave_row["attempted_task_images"].add((order, image))
                    if success:
                        wave_row["successful_task_images"].add((order, image))
                    if content:
                        wave_row["content_rejected_task_images"].add((order, image))
                    if timestamp is not None:
                        wave_row["row_timestamps"].append(timestamp)

    task_segment_stats: dict[int, Counter] = defaultdict(Counter)
    for segment in segments:
        order = int(segment["order"])
        wave = segment["wave"]
        if wave not in waves:
            waves[wave] = _new_wave_row(wave)
        wave_row = waves[wave]
        wave_row["provider_invocation_segments"] += 1
        wave_row["completed_vqa_batches"] += int(segment["done"])
        wave_row["incomplete_vqa_batches"] += int(not segment["done"])
        wave_row["explicit_application_retries_logged"] += int(segment["explicit_retries"])
        wave_row["provider_labeler_elapsed_seconds_logged"] += float(segment["vqa_elapsed_seconds"])
        counter = task_segment_stats[order]
        counter["provider_invocation_segments"] += 1
        counter["completed_vqa_batches"] += int(segment["done"])
        counter["incomplete_vqa_batches"] += int(not segment["done"])
        counter["explicit_application_retries_logged"] += int(segment["explicit_retries"])
        counter["provider_labeler_elapsed_seconds_logged"] += float(segment["vqa_elapsed_seconds"])
        timestamps = [value for value in segment["row_timestamps"] if value is not None]
        start = segment.get("started_datetime")
        if timestamps and start is not None:
            lower_bound = (max(timestamps) - start).total_seconds()
            if lower_bound >= 0:
                segment["harness_to_last_provider_record_seconds_lower_bound"] = lower_bound
                wave_row["segment_lower_bounds"].append(lower_bound)
                counter["harness_to_last_provider_record_seconds_lower_bound"] += lower_bound

    per_task: list[dict[str, Any]] = []
    all_committed: set[tuple[int, str]] = set()
    all_pending: set[tuple[int, str]] = set()
    all_known_complete: set[tuple[int, str]] = set()
    all_declared_reused: set[tuple[int, str]] = set()
    all_rejected: set[tuple[int, str]] = set()
    all_replacements: set[tuple[int, str, str]] = set()

    for task in tasks:
        order = int(task["order"])
        context = task_contexts[order]
        state = context["state"]
        rounds = context["rounds"]
        pending_manifest = context["pending_manifest"]
        committed = {_task_image_key(task, image) for image in context["committed"]}
        pending = {_task_image_key(task, image) for image in context["pending"]}
        declared_reused = {
            _task_image_key(task, image)
            for record in rounds
            for image in record.get("reused") or []
        }
        declared_reused.update(
            _task_image_key(task, image) for image in pending_manifest.get("reused") or []
        )
        pending_complete = pending if pending_manifest.get("complete") is True else {
            _task_image_key(task, image) for image in pending_manifest.get("reused") or []
        }
        known_complete = committed | pending_complete
        result_sets = task_result_sets[order]
        provider_success = {(order, image) for image in result_sets["success"]}
        provider_attempted = {(order, image) for image in result_sets["attempted"]}
        selected = committed | pending
        provider_success_selected = selected.intersection(provider_success)
        provider_success_known_complete = known_complete.intersection(provider_success)
        cache_only_known_complete = known_complete.difference(provider_success)
        rejected, replacements = _collect_rejection_evidence(
            state, task_manifests[order]
        )
        rejected_keys = {(order, image) for image in rejected}
        replacement_keys = {(order, old, new) for old, new in replacements}
        result_counter = task_result_stats[order]
        segment_counter = task_segment_stats[order]
        logical_missing = len(pending.difference(pending_complete))
        row = {
            "order": order,
            "dataset": str(task["dataset"]),
            "task": str(task["task"]),
            "state_status": state.get("status"),
            "committed_rounds": len(rounds),
            "pending_round": state.get("pending_round"),
            "logical_committed_task_images": len(committed),
            "logical_pending_selected_task_images": len(pending),
            "logical_selected_current_task_images": len(selected),
            "logical_known_complete_current_task_images": len(known_complete),
            "logical_pending_missing_by_manifest": logical_missing,
            "strict_cache_hits_declared_at_final_selection_or_resume": len(declared_reused),
            "strict_cache_complete_selected_without_stage_provider_success": len(cache_only_known_complete),
            "strict_cache_declared_overlap_with_stage_provider_success": len(
                declared_reused.intersection(provider_success)
            ),
            "provider_unique_attempted_task_images": len(provider_attempted),
            "provider_unique_successful_task_images": len(provider_success),
            "provider_successful_current_selected_task_images": len(provider_success_selected),
            "provider_successful_known_complete_current_task_images": len(
                provider_success_known_complete
            ),
            "provider_terminal_records": int(result_counter["terminal_records"]),
            "provider_successful_terminal_records": int(result_counter["successful_terminal_records"]),
            "provider_failed_terminal_records": int(result_counter["failed_terminal_records"]),
            "provider_unknown_terminal_records": int(result_counter["unknown_terminal_records"]),
            "provider_explicit_application_retries_logged": int(
                segment_counter["explicit_application_retries_logged"]
            ),
            "provider_application_call_attempts_observed_lower_bound": int(
                result_counter["terminal_records"]
                + segment_counter["explicit_application_retries_logged"]
            ),
            "provider_invocation_segments": int(segment_counter["provider_invocation_segments"]),
            "completed_vqa_batches": int(segment_counter["completed_vqa_batches"]),
            "incomplete_vqa_batches": int(segment_counter["incomplete_vqa_batches"]),
            "provider_labeler_elapsed_seconds_logged": float(
                segment_counter["provider_labeler_elapsed_seconds_logged"]
            ),
            "harness_to_last_provider_record_seconds_lower_bound": float(
                segment_counter["harness_to_last_provider_record_seconds_lower_bound"]
            ),
            "content_rejection_terminal_records": int(
                result_counter["content_rejection_terminal_records"]
            ),
            "content_rejected_unique_task_images_raw": len(result_sets["content"]),
            "rejected_not_budgeted_unique_task_images_ledger": len(rejected_keys),
            "replacement_edges": len(replacement_keys),
            "rejected_not_budgeted_images": sorted(rejected),
            "replacement_map": [
                {"rejected": old, "replacement": new}
                for _, old, new in sorted(replacement_keys)
            ],
            "result_files": task_result_files[order],
            "state": str(context["state_path"].resolve()),
            "pending_manifest": (
                str(context["pending_manifest_path"].resolve())
                if context["pending_manifest_path"] is not None
                else None
            ),
        }
        per_task.append(row)
        all_committed.update(committed)
        all_pending.update(pending)
        all_known_complete.update(known_complete)
        all_declared_reused.update(declared_reused)
        all_rejected.update(rejected_keys)
        all_replacements.update(replacement_keys)

    provider_attempted_all = {
        (record["order"], record["image"]) for record in raw_rows if record["image"]
    }
    provider_success_all = {
        (record["order"], record["image"])
        for record in raw_rows
        if record["image"] and record["success"]
    }
    provider_content_all = {
        (record["order"], record["image"])
        for record in raw_rows
        if record["image"] and record["content_rejection"]
    }
    current_selected = all_committed | all_pending
    provider_success_current = current_selected.intersection(provider_success_all)
    provider_success_known_complete = all_known_complete.intersection(provider_success_all)
    cache_only_known_complete = all_known_complete.difference(provider_success_all)
    retry_total = sum(int(segment["explicit_retries"]) for segment in segments)
    terminal_total = len(raw_rows)
    incomplete_batches = sum(not segment["done"] for segment in segments)

    wave_rows: list[dict[str, Any]] = []
    for wave in sorted(
        waves,
        key=lambda value: (
            0 if re.fullmatch(r"wave\d+", value) else 1,
            int(value[4:]) if re.fullmatch(r"wave\d+", value) else value,
        ),
    ):
        row = waves[wave]
        timestamps = sorted(row["row_timestamps"])
        approved = row["approved_unique_task_images"]
        attempted = row["attempted_task_images"]
        wave_rows.append({
            "wave": wave,
            "manifest": row.get("manifest"),
            "manifest_sha256": row.get("manifest_sha256"),
            "created_at_utc": row.get("created_at_utc"),
            "approved_task_count": int(row["approved_task_count"]),
            "approved_row_count": int(row["approved_row_count"]),
            "approved_unique_task_images": len(approved),
            "provider_invocation_segments": int(row["provider_invocation_segments"]),
            "completed_vqa_batches": int(row["completed_vqa_batches"]),
            "incomplete_vqa_batches": int(row["incomplete_vqa_batches"]),
            "provider_unique_attempted_task_images": len(attempted),
            "provider_unique_successful_task_images": len(row["successful_task_images"]),
            "provider_terminal_records": int(row["terminal_records"]),
            "provider_successful_terminal_records": int(row["successful_terminal_records"]),
            "provider_failed_terminal_records": int(row["failed_terminal_records"]),
            "provider_unknown_terminal_records": int(row["unknown_terminal_records"]),
            "provider_explicit_application_retries_logged": int(
                row["explicit_application_retries_logged"]
            ),
            "provider_application_call_attempts_observed_lower_bound": int(
                row["terminal_records"] + row["explicit_application_retries_logged"]
            ),
            "content_rejection_terminal_records": int(
                row["content_rejection_terminal_records"]
            ),
            "content_rejected_unique_task_images": len(row["content_rejected_task_images"]),
            "provider_labeler_elapsed_seconds_logged": round(
                float(row["provider_labeler_elapsed_seconds_logged"]), 3
            ),
            "harness_to_last_provider_record_seconds_lower_bound": round(
                sum(row["segment_lower_bounds"]), 3
            ),
            "first_provider_record_at_local": _format_datetime(timestamps[0] if timestamps else None),
            "last_provider_record_at_local": _format_datetime(timestamps[-1] if timestamps else None),
            "provider_activity_span_seconds_including_idle_gaps": (
                round((timestamps[-1] - timestamps[0]).total_seconds(), 3)
                if len(timestamps) >= 2 else 0.0 if timestamps else None
            ),
            "approved_rows_terminally_observed_unique": len(approved.intersection(attempted)),
            "approved_rows_without_terminal_record_at_snapshot": len(approved.difference(attempted)),
            "terminally_observed_task_images_outside_wave_preflight": len(
                attempted.difference(approved)
            ) if approved else len(attempted),
        })

    ledger_statuses = Counter(str(event.get("status") or "unknown") for event in provider_events.values())
    ledger_declared = sum(
        int(event.get("request_count", len(event.get("requested") or [])))
        for event in provider_events.values()
    )
    provider_times = sorted(
        record["timestamp"] for record in raw_rows if record["timestamp"] is not None
    )
    ending_files = _evidence_files(tasks, tasks_root, stage, acquisition_root)
    ending_fingerprints = _file_fingerprints(ending_files)
    changed_files = sorted(
        path
        for path in set(starting_fingerprints) | set(ending_fingerprints)
        if starting_fingerprints.get(path) != ending_fingerprints.get(path)
    )

    caveats = [
        "Logical counts use task-image identities; the same physical image selected for two tasks counts twice.",
        "A terminal JSONL row is one completed vqa_label.process_image operation, not necessarily one HTTP request.",
        "Application call attempts are terminal JSONL rows plus logged [retry] events. Interrupted in-flight calls and OpenAI-client internal retries are not exposed, so this is a lower bound and exact HTTP requests remain unknown.",
        "strict_cache_hits_declared_at_final_selection_or_resume can include successful rows from an earlier interrupted provider invocation that became cache hits on resume.",
        "strict_cache_complete_selected_without_stage_provider_success is the cleaner cache-only estimate: state/manifest-complete selections minus stage-local successful provider evidence.",
        "provider_labeler_elapsed_seconds_logged sums rounded tqdm elapsed values; harness_to_last_provider_record is a lower bound that includes pre-provider harness work but excludes post-provider work.",
        "provider_activity_span_seconds_including_idle_gaps is an envelope, not active compute time.",
        "Preflight row counts are approved maxima. Runtime is allowed to send an exact-byte subset, so unsent approved rows are not failures.",
        "Legacy provider_request_history is incomplete and migration events may be proxies; raw JSONL plus logs are the primary request evidence.",
    ]
    if incomplete_batches:
        caveats.append(
            f"{incomplete_batches} logged VQA invocation(s) lack a [done] marker; their in-flight attempts make the request lower bound non-exact."
        )
    if changed_files:
        caveats.append(
            f"{len(changed_files)} evidence file(s) changed during the scan; rerun after acquisition is idle for a stable snapshot."
        )
    if all_warnings:
        caveats.append(f"{len(all_warnings)} structural/log warning(s) are listed under evidence_coverage.warnings.")

    return {
        "schema": "iterative-vqa-acquisition-accounting-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "stage": stage,
        "manifest": str(manifest_path),
        "tasks_root": str(tasks_root),
        "acquisition_root": str(acquisition_root),
        "summary": {
            "manifest_task_count": len(tasks),
            "tasks_with_state": sum(row["state_status"] != "missing_state" for row in per_task),
            "tasks_at_logical_500": sum(row["logical_committed_task_images"] == 500 for row in per_task),
            "logical_committed_task_images": len(all_committed),
            "logical_pending_selected_task_images": len(all_pending),
            "logical_selected_current_task_images": len(current_selected),
            "logical_known_complete_current_task_images": len(all_known_complete),
            "logical_pending_missing_by_manifest": sum(
                row["logical_pending_missing_by_manifest"] for row in per_task
            ),
            "strict_cache_hits_declared_at_final_selection_or_resume": len(all_declared_reused),
            "strict_cache_complete_selected_without_stage_provider_success": len(cache_only_known_complete),
            "strict_cache_declared_overlap_with_stage_provider_success": len(
                all_declared_reused.intersection(provider_success_all)
            ),
            "provider_result_files": result_files,
            "provider_result_files_mapped_to_log": len(mapped_result_files),
            "provider_terminal_records": terminal_total,
            "provider_successful_terminal_records": sum(record["success"] for record in raw_rows),
            "provider_failed_terminal_records": sum(record["failed"] for record in raw_rows),
            "provider_unknown_terminal_records": sum(
                not record["success"] and not record["failed"] for record in raw_rows
            ),
            "provider_unique_attempted_task_images": len(provider_attempted_all),
            "provider_unique_successful_task_images": len(provider_success_all),
            "provider_successful_current_selected_task_images": len(provider_success_current),
            "provider_successful_known_complete_current_task_images": len(
                provider_success_known_complete
            ),
            "provider_invocation_segments": len(segments),
            "completed_vqa_batches": sum(segment["done"] for segment in segments),
            "incomplete_vqa_batches": incomplete_batches,
            "provider_explicit_application_retries_logged": retry_total,
            "provider_application_call_attempts_observed_lower_bound": terminal_total + retry_total,
            "provider_exact_http_request_count": None,
            "provider_labeler_elapsed_seconds_logged": round(
                sum(float(segment["vqa_elapsed_seconds"]) for segment in segments), 3
            ),
            "harness_to_last_provider_record_seconds_lower_bound": round(
                sum(
                    float(segment.get("harness_to_last_provider_record_seconds_lower_bound", 0.0))
                    for segment in segments
                ), 3
            ),
            "first_provider_record_at_local": _format_datetime(provider_times[0] if provider_times else None),
            "last_provider_record_at_local": _format_datetime(provider_times[-1] if provider_times else None),
            "provider_activity_span_seconds_including_idle_gaps": (
                round((provider_times[-1] - provider_times[0]).total_seconds(), 3)
                if len(provider_times) >= 2 else 0.0 if provider_times else None
            ),
            "content_rejection_terminal_records": sum(
                record["content_rejection"] for record in raw_rows
            ),
            "content_rejected_unique_task_images_raw": len(provider_content_all),
            "rejected_not_budgeted_unique_task_images_ledger": len(all_rejected),
            "replacement_edges": len(all_replacements),
            "provider_request_ledger_unique_events": len(provider_events),
            "provider_request_ledger_declared_image_submissions": ledger_declared,
            "provider_request_ledger_statuses": dict(sorted(ledger_statuses.items())),
        },
        "waves": wave_rows,
        "tasks": per_task,
        "evidence_coverage": {
            "result_files_total": result_files,
            "result_files_mapped_to_log": len(mapped_result_files),
            "unattributed_result_files": sorted(unattributed_result_files),
            "malformed_result_rows": malformed_result_rows,
            "provider_segments_total": len(segments),
            "provider_segments_without_done_marker": incomplete_batches,
            "evidence_files_changed_during_scan": changed_files,
            "stable_snapshot": not changed_files,
            "warnings": sorted(set(all_warnings)),
        },
        "definitions": {
            "logical_budget_unit": "unique selected task-image, including permanent audit images",
            "provider_terminal_record": "one JSONL record emitted after process_image succeeds or exhausts vqa_label retries",
            "provider_unique_successful_task_image": "unique (task order, logical image) with a non-[FAILED] stage-local JSONL result",
            "provider_application_call_attempts_observed_lower_bound": "terminal JSONL records + explicit [retry] log events",
            "strict_cache_complete_selected_without_stage_provider_success": "state/manifest-complete selected task-images with no successful stage-local provider result",
            "content_rejected_not_budgeted": "terminal data_inspection_failed images recorded by the rejection ledger and excluded from logical budget",
        },
        "caveats": caveats,
    }


_CSV_FIELDS = [
    "order", "dataset", "task", "state_status", "committed_rounds", "pending_round",
    "logical_committed_task_images", "logical_pending_selected_task_images",
    "logical_selected_current_task_images", "logical_known_complete_current_task_images",
    "logical_pending_missing_by_manifest",
    "strict_cache_hits_declared_at_final_selection_or_resume",
    "strict_cache_complete_selected_without_stage_provider_success",
    "provider_unique_attempted_task_images", "provider_unique_successful_task_images",
    "provider_terminal_records", "provider_successful_terminal_records",
    "provider_failed_terminal_records", "provider_explicit_application_retries_logged",
    "provider_application_call_attempts_observed_lower_bound",
    "content_rejection_terminal_records", "rejected_not_budgeted_unique_task_images_ledger",
    "replacement_edges", "provider_invocation_segments", "completed_vqa_batches",
    "incomplete_vqa_batches", "provider_labeler_elapsed_seconds_logged",
    "harness_to_last_provider_record_seconds_lower_bound",
]


def _csv_text(report: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["tasks"])
    return buffer.getvalue()


def _minutes(seconds: Any) -> str:
    if seconds is None:
        return "n/a"
    return f"{float(seconds) / 60.0:.2f} min"


def _markdown_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Iterative-VQA acquisition accounting",
        "",
        f"Snapshot: `{report['generated_at_utc']}`  ",
        f"Stage: `{report['stage']}`  ",
        f"Stable during scan: **{report['evidence_coverage']['stable_snapshot']}**",
        "",
        "## Current logical state",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Manifest tasks | {summary['manifest_task_count']} |",
        f"| Tasks at committed budget 500 | {summary['tasks_at_logical_500']} |",
        f"| Logical committed task-images | {summary['logical_committed_task_images']} |",
        f"| Current pending selected task-images | {summary['logical_pending_selected_task_images']} |",
        f"| Current logical selected total | {summary['logical_selected_current_task_images']} |",
        f"| Pending missing by frozen manifests | {summary['logical_pending_missing_by_manifest']} |",
        f"| Cache-only complete selected (no stage-local provider success) | {summary['strict_cache_complete_selected_without_stage_provider_success']} |",
        "",
        "## Provider work observed",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Result JSONL files | {summary['provider_result_files']} |",
        f"| Terminal image operations | {summary['provider_terminal_records']} |",
        f"| Successful terminal operations | {summary['provider_successful_terminal_records']} |",
        f"| Failed terminal operations | {summary['provider_failed_terminal_records']} |",
        f"| Unique successful provider-labeled task-images | {summary['provider_unique_successful_task_images']} |",
        f"| Explicit application retries logged | {summary['provider_explicit_application_retries_logged']} |",
        f"| Observable application call attempts (lower bound) | {summary['provider_application_call_attempts_observed_lower_bound']} |",
        f"| Exact HTTP requests | unavailable |",
        f"| Content-rejection terminal operations | {summary['content_rejection_terminal_records']} |",
        f"| Rejected, non-budgeted unique task-images | {summary['rejected_not_budgeted_unique_task_images_ledger']} |",
        f"| Replacement edges | {summary['replacement_edges']} |",
        "",
        "## Timing",
        "",
        f"Logged VQA-labeler active time: **{_minutes(summary['provider_labeler_elapsed_seconds_logged'])}**.  ",
        f"Harness-start to last provider record lower bound: **{_minutes(summary['harness_to_last_provider_record_seconds_lower_bound'])}**.  ",
        f"Provider activity envelope (includes idle gaps): **{_minutes(summary['provider_activity_span_seconds_including_idle_gaps'])}**.",
        "",
        "## Waves",
        "",
        "| Wave | Approved | Terminal ops | Unique success | Retries | App attempts >= | Rejections | VQA time | Incomplete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["waves"]:
        lines.append(
            f"| {row['wave']} | {row['approved_row_count']} | {row['provider_terminal_records']} | "
            f"{row['provider_unique_successful_task_images']} | {row['provider_explicit_application_retries_logged']} | "
            f"{row['provider_application_call_attempts_observed_lower_bound']} | "
            f"{row['content_rejection_terminal_records']} | "
            f"{row['provider_labeler_elapsed_seconds_logged'] / 60.0:.2f} min | "
            f"{row['incomplete_vqa_batches']} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in report["caveats"])
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, default=ROOT / "dataset" / "tasks")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(
        manifest_path=args.manifest,
        stage=args.stage,
        acquisition_root=args.acquisition_root,
        tasks_root=args.tasks_root,
    )
    if args.output_json is not None:
        _write_atomic(
            args.output_json,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
    if args.output_csv is not None:
        _write_atomic(args.output_csv, _csv_text(report))
    if args.output_md is not None:
        _write_atomic(args.output_md, _markdown_text(report))
    summary = report["summary"]
    print(json.dumps({
        "read_only": True,
        "stable_snapshot": report["evidence_coverage"]["stable_snapshot"],
        "logical_committed": summary["logical_committed_task_images"],
        "logical_pending_selected": summary["logical_pending_selected_task_images"],
        "provider_terminal_records": summary["provider_terminal_records"],
        "provider_unique_successful_task_images": summary["provider_unique_successful_task_images"],
        "provider_application_call_attempts_observed_lower_bound": summary[
            "provider_application_call_attempts_observed_lower_bound"
        ],
        "rejected_not_budgeted": summary["rejected_not_budgeted_unique_task_images_ledger"],
        "replacement_edges": summary["replacement_edges"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
