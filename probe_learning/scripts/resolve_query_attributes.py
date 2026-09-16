"""Resolve query attributes against a task's modeled attribute schema.

This is the auditable bridge between query-image attribute discovery and the
fixed attributes modeled by ProbeBank.  It deliberately never reads
``gt_attribute.txt``: that file is a human-review reference, not supervision.

Resolution order:

1. ``<task>/attributes.txt`` (canonical, zero API calls).
2. An explicit ``--query-vqa-attributes`` file.
3. Existing ``<task>/qa/attributes.txt`` produced by the query VQA extractor.
4. The existing ``attribute_annotation/extract_attributes.py``
   pipeline, but only when ``--allow-api`` is explicitly supplied.

Every discovered attribute must map one-to-one to ``task.json.attributes``.
Matching is deterministic: normalized exact names plus aliases explicitly
declared in ``task.json.attribute_aliases``.  Missing, extra, duplicate, or
ambiguous attributes fail closed and no ``resolved_attributes.txt`` is emitted.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACTOR = (
    EXPERIMENT_ROOT / "attribute_annotation" / "extract_attributes.py"
)
DEFAULT_EXTRACTOR_CONFIG = (
    EXPERIMENT_ROOT / "attribute_annotation" / "config_attributes.yaml"
)
ATTRIBUTE_SPLIT_RE = re.compile(r"[,，;；\n]+")


class ResolutionError(RuntimeError):
    """A fail-closed attribute resolution error with stable reason codes."""

    def __init__(self, message: str, reason_codes: Iterable[str]):
        super().__init__(message)
        self.reason_codes = list(dict.fromkeys(reason_codes))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def normalize_attribute(value: str) -> str:
    """Normalize spelling only; do not make semantic/fuzzy guesses."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def parse_attribute_text(text: str) -> list[str]:
    """Parse the extractor/canonical ``{a, b}`` format deterministically."""
    value = str(text).strip().lstrip("\ufeff")
    if not value:
        raise ResolutionError("attribute file is empty", ["empty_attribute_file"])
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    attrs = [part.strip().strip("`'\"{} ") for part in ATTRIBUTE_SPLIT_RE.split(value)]
    attrs = [attr for attr in attrs if attr]
    if not attrs:
        raise ResolutionError("attribute file has no usable values", ["empty_attribute_list"])
    return attrs


def read_attributes(path: Path) -> list[str]:
    if not path.is_file():
        raise ResolutionError(f"attribute file does not exist: {path}", ["attribute_file_missing"])
    return parse_attribute_text(path.read_text(encoding="utf-8-sig"))


def load_task_schema(task_dir: Path) -> tuple[dict[str, Any], list[str]]:
    task_json = task_dir / "task.json"
    if not task_json.is_file():
        raise ResolutionError(f"task.json does not exist: {task_json}", ["task_json_missing"])
    try:
        task = json.loads(task_json.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResolutionError(f"cannot parse {task_json}: {exc}", ["task_json_invalid"]) from exc
    modeled = task.get("attributes")
    if not isinstance(modeled, list) or not modeled or any(not str(v).strip() for v in modeled):
        raise ResolutionError(
            "task.json.attributes must be a non-empty string list",
            ["modeled_attributes_invalid"],
        )
    modeled = [str(value).strip() for value in modeled]
    normalized = [normalize_attribute(value) for value in modeled]
    if len(normalized) != len(set(normalized)):
        raise ResolutionError(
            "task.json.attributes collide after normalization",
            ["modeled_attributes_ambiguous"],
        )
    return task, modeled


def _alias_lookup(
    task: dict[str, Any], modeled: list[str]
) -> tuple[dict[str, set[str]], dict[tuple[str, str], str], list[str]]:
    aliases = task.get("attribute_aliases", {})
    errors: list[str] = []
    if aliases is None:
        aliases = {}
    if not isinstance(aliases, dict):
        return {}, {}, ["attribute_aliases_invalid"]

    modeled_set = set(modeled)
    lookup: dict[str, set[str]] = {}
    match_type: dict[tuple[str, str], str] = {}
    for canonical in modeled:
        norm = normalize_attribute(canonical)
        lookup.setdefault(norm, set()).add(canonical)
        match_type[(norm, canonical)] = "canonical_exact"

    for canonical, values in aliases.items():
        if canonical not in modeled_set:
            errors.append("attribute_alias_target_unknown")
            continue
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            errors.append("attribute_aliases_invalid")
            continue
        for alias in values:
            norm = normalize_attribute(alias)
            if not norm:
                errors.append("attribute_aliases_invalid")
                continue
            lookup.setdefault(norm, set()).add(canonical)
            match_type[(norm, canonical)] = "declared_alias"
    return lookup, match_type, list(dict.fromkeys(errors))


def map_discovered_attributes(
    discovered: list[str], task: dict[str, Any], modeled: list[str]
) -> dict[str, Any]:
    """Strictly map discovered names to modeled keys and return an audit block."""
    lookup, match_types, schema_errors = _alias_lookup(task, modeled)
    mappings: list[dict[str, Any]] = []
    unmapped: list[str] = []
    ambiguous: list[dict[str, Any]] = []
    targets: dict[str, list[str]] = {}

    for raw in discovered:
        normalized = normalize_attribute(raw)
        candidates = sorted(lookup.get(normalized, set()))
        row: dict[str, Any] = {
            "raw": raw,
            "normalized": normalized,
            "modeled_attribute": None,
            "match_type": None,
        }
        if len(candidates) == 1:
            target = candidates[0]
            row["modeled_attribute"] = target
            row["match_type"] = match_types[(normalized, target)]
            targets.setdefault(target, []).append(raw)
        elif len(candidates) > 1:
            ambiguous.append({"raw": raw, "candidates": candidates})
        else:
            unmapped.append(raw)
        mappings.append(row)

    duplicate_targets = {
        target: raw_values for target, raw_values in targets.items() if len(raw_values) > 1
    }
    missing = [attribute for attribute in modeled if attribute not in targets]
    reason_codes = list(schema_errors)
    if unmapped:
        reason_codes.append("unmapped_discovered_attributes")
    if ambiguous:
        reason_codes.append("ambiguous_discovered_attributes")
    if duplicate_targets:
        reason_codes.append("duplicate_modeled_attribute_matches")
    if missing:
        reason_codes.append("missing_modeled_attributes")

    passed = not reason_codes
    return {
        "mappings": mappings,
        "mapped_attributes": list(modeled) if passed else [],
        "unmapped_discovered_attributes": unmapped,
        "ambiguous_discovered_attributes": ambiguous,
        "duplicate_modeled_attribute_matches": duplicate_targets,
        "missing_modeled_attributes": missing,
        "validation": {
            "status": "passed" if passed else "failed",
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "policy": (
                "strict one-to-one normalized exact/declared-alias mapping; "
                "no fuzzy or semantic guessing"
            ),
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _query_image_count(task_dir: Path) -> int:
    query_dir = task_dir / "query_pics"
    if not query_dir.is_dir():
        query_dir = task_dir / "query pics"
    if not query_dir.is_dir():
        return 0
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for path in query_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _run_query_vqa_extractor(
    task_dir: Path,
    discovery_dir: Path,
    extractor_script: Path,
    extractor_config: Path,
    python_executable: str,
) -> tuple[Path, dict[str, Any], float]:
    if not extractor_script.is_file():
        raise ResolutionError(
            f"query VQA extractor does not exist: {extractor_script}",
            ["query_vqa_extractor_missing"],
        )
    query_dir = task_dir / "query_pics"
    if not query_dir.is_dir():
        query_dir = task_dir / "query pics"
    if not query_dir.is_dir():
        raise ResolutionError(
            f"query image directory does not exist under {task_dir}",
            ["query_images_missing"],
        )
    discovery_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(extractor_script),
        "--config",
        str(extractor_config),
        "--query-dir",
        str(query_dir),
        "--out-dir",
        str(discovery_dir),
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(extractor_script.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    execution = {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if result.returncode != 0:
        raise ResolutionError(
            f"query VQA extractor failed with exit code {result.returncode}",
            ["query_vqa_extractor_failed"],
        )
    output = discovery_dir / "attributes.txt"
    if not output.is_file():
        raise ResolutionError(
            f"query VQA extractor did not create {output}",
            ["query_vqa_output_missing"],
        )
    return output, execution, elapsed


def resolve_task_attributes(
    task_dir: Path,
    output_dir: Path,
    *,
    query_vqa_attributes: Path | None = None,
    reuse_existing: bool = True,
    allow_api: bool = False,
    extractor_script: Path = DEFAULT_EXTRACTOR,
    extractor_config: Path = DEFAULT_EXTRACTOR_CONFIG,
    python_executable: str = sys.executable,
) -> tuple[dict[str, Any], Path]:
    """Resolve one task and always write ``audit.json`` before returning."""
    task_dir = Path(task_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.json"
    resolved_path = output_dir / "resolved_attributes.txt"
    started_wall = utc_now_iso()
    started = time.perf_counter()
    gt_reference = task_dir / "gt_attribute.txt"
    audit: dict[str, Any] = {
        "schema_version": 1,
        "task_dir": str(task_dir),
        "task_id": task_dir.name,
        "dataset": None,
        "started_at_utc": started_wall,
        "finished_at_utc": None,
        "elapsed_seconds": None,
        "source": None,
        "api_calls": {
            "invoked_this_run": False,
            "successful_operations": 0,
            "actual_requests_including_retries": 0,
            "note": "No paid API invocation occurred in this resolver run.",
        },
        "modeled_attributes": [],
        "raw_discovered_attributes": [],
        "mappings": [],
        "mapped_attributes": [],
        "unmapped_discovered_attributes": [],
        "ambiguous_discovered_attributes": [],
        "duplicate_modeled_attribute_matches": {},
        "missing_modeled_attributes": [],
        "validation": {"status": "failed", "reason_codes": ["resolution_not_completed"]},
        "gt_attribute_reference": {
            "path": str(gt_reference),
            "exists": gt_reference.is_file(),
            "consumed": False,
            "policy": "reference-only; never read by this resolver",
        },
        "artifacts": {"audit": str(audit_path), "resolved_attributes": None},
    }

    try:
        task, modeled = load_task_schema(task_dir)
        audit["task_id"] = str(task.get("task_id") or task_dir.name)
        audit["dataset"] = task.get("dataset")
        audit["modeled_attributes"] = modeled

        canonical = task_dir / "attributes.txt"
        existing_query_vqa = task_dir / "qa" / "attributes.txt"
        extractor_elapsed = 0.0
        extractor_execution = None
        if canonical.is_file():
            source_path = canonical
            source_kind = "canonical_attributes_txt"
        elif query_vqa_attributes is not None:
            source_path = Path(query_vqa_attributes).resolve()
            source_kind = "query_vqa_explicit_output"
        elif reuse_existing and existing_query_vqa.is_file():
            source_path = existing_query_vqa
            source_kind = "query_vqa_existing_output"
        elif not allow_api:
            raise ResolutionError(
                "canonical attributes are absent and query VQA API invocation is disabled; "
                "pass --query-vqa-attributes or explicitly opt in with --allow-api",
                ["query_vqa_required_but_api_disabled"],
            )
        else:
            query_count = _query_image_count(task_dir)
            minimum_operations = query_count + 1 if query_count else None
            audit["source"] = {
                "kind": "query_vqa_live_extraction",
                "path": str(output_dir / "query_vqa" / "attributes.txt"),
                "extractor_elapsed_seconds": None,
                "extractor_execution": None,
            }
            audit["api_calls"] = {
                "invoked_this_run": True,
                "successful_operations": None,
                "actual_requests_including_retries": None,
                "minimum_requests_if_successful": minimum_operations,
                "note": (
                    "The existing extractor owns retries and does not expose an exact HTTP "
                    "request count; a failed run may have made zero or more requests."
                ),
            }
            source_path, extractor_execution, extractor_elapsed = _run_query_vqa_extractor(
                task_dir=task_dir,
                discovery_dir=output_dir / "query_vqa",
                extractor_script=Path(extractor_script).resolve(),
                extractor_config=Path(extractor_config).resolve(),
                python_executable=python_executable,
            )
            source_kind = "query_vqa_live_extraction"
            successful_operations = query_count + 1
            audit["api_calls"] = {
                "invoked_this_run": True,
                "successful_operations": successful_operations,
                "actual_requests_including_retries": None,
                "minimum_requests": successful_operations,
                "note": (
                    "The existing extractor performs one operation per query image plus one "
                    "summary operation. Internal retry requests are not exposed, so the exact "
                    "HTTP request count is unavailable."
                ),
            }

        if source_path.resolve() == gt_reference.resolve():
            raise ResolutionError(
                "gt_attribute.txt is reference-only and cannot be used as an attribute source",
                ["gt_attribute_reference_forbidden"],
            )
        audit["source"] = {
            "kind": source_kind,
            "path": str(source_path),
            "extractor_elapsed_seconds": round(extractor_elapsed, 6),
            "extractor_execution": extractor_execution,
        }
        discovered = read_attributes(source_path)
        audit["raw_discovered_attributes"] = discovered
        audit.update(map_discovered_attributes(discovered, task, modeled))
        if audit["validation"]["status"] == "passed":
            _atomic_text(resolved_path, "{" + ", ".join(modeled) + "}\n")
            audit["artifacts"]["resolved_attributes"] = str(resolved_path)
    except ResolutionError as exc:
        audit["validation"] = {
            "status": "failed",
            "reason_codes": exc.reason_codes,
            "message": str(exc),
            "policy": "fail closed; no resolved attribute artifact was emitted",
        }
    except Exception as exc:  # keep the audit artifact even for unexpected failures
        audit["validation"] = {
            "status": "failed",
            "reason_codes": ["unexpected_resolution_error"],
            "message": f"{type(exc).__name__}: {exc}",
            "policy": "fail closed; no resolved attribute artifact was emitted",
        }
    finally:
        audit["finished_at_utc"] = utc_now_iso()
        audit["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _atomic_json(audit_path, audit)
    return audit, audit_path


def _default_output_dir(task_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return task_dir / "qa" / "query_attribute_resolution" / stamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve query-discovered attributes to task.json modeled keys"
    )
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--query-vqa-attributes",
        type=Path,
        help="reuse an explicit attributes.txt produced by the query VQA extractor",
    )
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="do not reuse <task>/qa/attributes.txt",
    )
    parser.add_argument(
        "--allow-api",
        action="store_true",
        help="explicitly allow the existing query VQA extractor to make paid API calls",
    )
    parser.add_argument("--extractor-script", type=Path, default=DEFAULT_EXTRACTOR)
    parser.add_argument("--extractor-config", type=Path, default=DEFAULT_EXTRACTOR_CONFIG)
    parser.add_argument("--vqa-python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task_dir = args.task_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir(task_dir)
    audit, audit_path = resolve_task_attributes(
        task_dir,
        output_dir,
        query_vqa_attributes=args.query_vqa_attributes,
        reuse_existing=not args.no_reuse_existing,
        allow_api=args.allow_api,
        extractor_script=args.extractor_script,
        extractor_config=args.extractor_config,
        python_executable=args.vqa_python,
    )
    print(json.dumps({
        "status": audit["validation"]["status"],
        "source": (audit.get("source") or {}).get("kind"),
        "mapped_attributes": audit.get("mapped_attributes", []),
        "audit": str(audit_path),
        "resolved_attributes": audit.get("artifacts", {}).get("resolved_attributes"),
    }, ensure_ascii=False))
    return 0 if audit["validation"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
