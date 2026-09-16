"""Auditable, credential-free preflight manifests for iterative VQA calls.

The preflight binds a frozen iterative selection to the exact local image
bytes, task attributes, expanded prompt, endpoint, model, and non-secret
request configuration.  It never loads credentials from the environment and
never writes credential fields to the manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


SCHEMA = "iterative-vqa-outbound-preflight-v1"
_EXACT_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "client_secret",
    "access_token",
    "refresh_token",
    "bearer_token",
}
_OUTBOUND_CONFIG_KEYS = (
    "base_url",
    "model",
    "prompt",
    "max_tokens",
    "timeout",
    "max_retries",
    "delay",
    "concurrency",
)


class PreflightError(ValueError):
    """Raised when a preflight cannot prove the exact outbound payload."""


def _is_secret_key(key: Any) -> bool:
    """Identify credential fields without misclassifying ``max_tokens``."""
    normalized = str(key).strip().lower().replace("-", "_")
    if normalized in _EXACT_SECRET_KEYS:
        return True
    return normalized.endswith(
        ("_api_key", "_access_token", "_refresh_token", "_password", "_secret")
    )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _redact_secrets(item)
            for key, item in value.items()
            if not _is_secret_key(key)
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _assert_no_secret_fields(value: Any, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_secret_key(key):
                raise PreflightError(
                    f"credential-like field is forbidden in preflight at {location}.{key}"
                )
            _assert_no_secret_fields(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{location}[{index}]")


def load_public_vqa_config(config_path: Path) -> dict[str, Any]:
    """Load only non-secret configuration; environment credentials are ignored."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise PreflightError(f"VQA config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(payload, dict):
        raise PreflightError(f"VQA config must be a mapping: {path}")
    public = _redact_secrets(payload)
    if not public.get("base_url") or not public.get("model"):
        raise PreflightError(f"VQA config lacks base_url/model: {path}")
    if not isinstance(public.get("prompt"), str) or not public["prompt"].strip():
        raise PreflightError(f"VQA config lacks a non-empty prompt: {path}")
    outbound = {key: public[key] for key in _OUTBOUND_CONFIG_KEYS if key in public}
    return {
        "config_path": str(path),
        "endpoint": str(public["base_url"]),
        "model": str(public["model"]),
        "prompt_template": str(public["prompt"]),
        "prompt_template_sha256": _sha256_bytes(str(public["prompt"]).encode("utf-8")),
        "config_sha256": _sha256_json(outbound),
        "redacted_config_sha256": _sha256_json(public),
    }


def validate_external_credential_source(
    config_path: Path, *, environ: dict[str, str] | None = None
) -> dict[str, str]:
    """Require a runtime-only credential and reject tracked literal secrets.

    Only the environment variable name is returned.  Credential values are
    never copied into logs, manifests, state, or exceptions.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise PreflightError(f"VQA config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(payload, dict):
        raise PreflightError(f"VQA config must be a mapping: {path}")

    nonempty_fields: list[str] = []

    def inspect(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{location}.{key}" if location else str(key)
                if _is_secret_key(key):
                    if item is not None and str(item).strip():
                        nonempty_fields.append(child)
                else:
                    inspect(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(item, f"{location}[{index}]")

    inspect(payload, "")
    if nonempty_fields:
        raise PreflightError(
            "external VQA is disabled while config contains non-empty credential "
            f"fields: {sorted(nonempty_fields)}; blank them and use an environment variable"
        )

    environment = os.environ if environ is None else environ
    for name in ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY"):
        if str(environment.get(name, "")).strip():
            return {
                "credential_source": "environment",
                "environment_variable": name,
            }
    raise PreflightError(
        "external VQA requires DASHSCOPE_API_KEY or DEEPSEEK_API_KEY in the runtime environment"
    )


def materialize_public_vqa_config(
    source_path: Path,
    destination_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically derive a credential-free runtime config from a source YAML."""
    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    if not source.is_file():
        raise PreflightError(f"VQA config not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(payload, dict):
        raise PreflightError(f"VQA config must be a mapping: {source}")
    redacted = _redact_secrets(payload)
    _assert_no_secret_fields(redacted, "public_vqa_config")
    # The runtime supplies images, output directory, and task attributes via
    # explicit command-line arguments.  Keep the public config minimal so old
    # task-local paths/attributes from a source YAML cannot leak into an
    # outbound approval artifact or be mistaken for runtime inputs.
    public = {
        key: redacted[key]
        for key in _OUTBOUND_CONFIG_KEYS
        if key in redacted
    }
    # Validate required outbound fields before writing anything.
    for key in ("base_url", "model", "prompt"):
        if key not in public or not str(public[key]).strip():
            raise PreflightError(f"VQA config lacks non-empty {key}: {source}")

    if destination.is_file():
        existing = yaml.safe_load(destination.read_text(encoding="utf-8-sig")) or {}
        if not isinstance(existing, dict):
            if not overwrite:
                raise PreflightError(
                    f"refusing to replace invalid public VQA config {destination}; "
                    "request an explicit overwrite"
                )
        else:
            _assert_no_secret_fields(existing, "existing_public_vqa_config")
            if existing == public:
                return {
                    "status": "reused_exact",
                    "source": str(source),
                    "path": str(destination),
                    "redacted_config_sha256": _sha256_json(public),
                }
            if not overwrite:
                raise PreflightError(
                    f"refusing to replace divergent public VQA config {destination}; "
                    "request an explicit overwrite"
                )
    elif destination.exists() and not overwrite:
        raise PreflightError(
            f"public VQA config destination is not a file: {destination}"
        )

    rendered = yaml.safe_dump(
        public,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    _atomic_write_text(destination, rendered)
    # Re-load what was written so serialization cannot silently alter types.
    observed = yaml.safe_load(destination.read_text(encoding="utf-8-sig")) or {}
    if observed != public:
        raise PreflightError(f"public VQA config round-trip mismatch: {destination}")
    return {
        "status": "created" if not overwrite else "created_or_replaced",
        "source": str(source),
        "path": str(destination),
        "redacted_config_sha256": _sha256_json(public),
    }


def expanded_prompt(prompt_template: str, image_id: str, attributes_text: str) -> str:
    """Mirror ``vqa_label.build_prompt`` without importing its API client."""
    prompt = prompt_template.replace("{image_id}", image_id)
    prompt = prompt.replace("{attributes}", attributes_text.strip())
    return prompt.strip()


def _parse_attribute_text(text: str) -> list[str]:
    value = str(text).strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    attributes = [part.strip() for part in value.split(",") if part.strip()]
    if not attributes:
        raise PreflightError("attribute text does not contain any attributes")
    return attributes


def _validate_attribute_text(
    text: str, expected_attributes: list[str], *, source: Path
) -> str:
    normalized = str(text).strip()
    observed = _parse_attribute_text(normalized)
    if observed != list(expected_attributes):
        raise PreflightError(
            f"VQA attributes differ from runtime modeled attributes at {source}: "
            f"observed={observed!r}, expected={list(expected_attributes)!r}"
        )
    return normalized


def _atomic_write_text(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


def materialize_task_attribute_files(
    tasks: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create missing ``qa/attributes.txt`` files without model/API calls.

    The task manifest and the configured adapter must agree exactly. Existing
    files are never replaced and must already contain the same ordered attrs.
    """
    import run_retrieval_harness as harness

    results: list[dict[str, Any]] = []
    for task_spec in sorted(tasks, key=lambda item: int(item["order"])):
        order = int(task_spec["order"])
        harness.configure(
            task_spec["dataset"], task_spec["task"], task_spec.get("joint_label")
        )
        adapter = harness.ADAPTER
        modeled_attributes = list(harness.ATTRS)
        declared_attributes = list(task_spec.get("attributes") or [])
        if declared_attributes != modeled_attributes:
            raise PreflightError(
                f"cannot materialize order {order}: manifest/runtime attribute mismatch: "
                f"{declared_attributes!r} != {modeled_attributes!r}"
            )
        target = adapter.task_root / "qa" / "attributes.txt"
        if target.is_file():
            _validate_attribute_text(
                target.read_text(encoding="utf-8-sig"),
                modeled_attributes,
                source=target,
            )
            status = "reused_exact"
            source = target
        else:
            canonical = adapter.task_root / "attributes.txt"
            if canonical.is_file():
                try:
                    text = _validate_attribute_text(
                        canonical.read_text(encoding="utf-8-sig"),
                        modeled_attributes,
                        source=canonical,
                    )
                except PreflightError:
                    # A legacy task-level file may intentionally contain a
                    # larger historical attribute set. It is not the outbound
                    # target; the all-36 manifest + configured runtime schema is.
                    text = "{" + ", ".join(modeled_attributes) + "}"
                    source = Path("all36-task-manifest+runtime-adapter")
                else:
                    source = canonical
            else:
                text = "{" + ", ".join(modeled_attributes) + "}"
                _validate_attribute_text(text, modeled_attributes, source=target)
                source = Path("all36-task-manifest+runtime-adapter")
            _atomic_write_text(target, text + "\n")
            status = "created"
        results.append(
            {
                "order": order,
                "dataset": str(task_spec["dataset"]),
                "task": str(task_spec["task"]),
                "status": status,
                "attributes": modeled_attributes,
                "path": str(target.resolve()),
                "source": str(source.resolve()) if source.is_absolute() else str(source),
            }
        )
    return results


def _manifest_digest(payload: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("manifest_sha256", None)
    return _sha256_json(unsigned)


def seal_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(payload)
    sealed.pop("manifest_sha256", None)
    _assert_no_secret_fields(sealed)
    sealed["manifest_sha256"] = _manifest_digest(sealed)
    return sealed


def validate_manifest_payload(
    payload: dict[str, Any], expected_sha256: str | None = None
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise PreflightError(f"unsupported preflight schema: {payload.get('schema')!r}")
    _assert_no_secret_fields(payload)
    declared = str(payload.get("manifest_sha256") or "")
    observed = _manifest_digest(payload)
    if not declared or declared != observed:
        raise PreflightError(
            f"preflight content hash mismatch: declared={declared!r}, observed={observed!r}"
        )
    if expected_sha256 is not None and observed != str(expected_sha256).lower():
        raise PreflightError(
            f"preflight approval hash mismatch: expected={expected_sha256!r}, observed={observed!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or int(payload.get("row_count", -1)) != len(rows):
        raise PreflightError("preflight row_count does not match rows")
    keys: set[tuple[int, str, str, int, str]] = set()
    for row in rows:
        key = (
            int(row["order"]),
            str(row["dataset"]),
            str(row["task"]),
            int(row["round"]),
            str(row["logical_image"]),
        )
        if key in keys:
            raise PreflightError(f"duplicate preflight row: {key}")
        keys.add(key)
    return payload


def load_and_validate_manifest(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise PreflightError(f"preflight manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise PreflightError(f"invalid preflight JSON: {manifest_path}: {error}") from error
    return validate_manifest_payload(payload, expected_sha256)


def write_manifest(
    path: Path, payload: dict[str, Any], *, overwrite: bool = False
) -> dict[str, Any]:
    sealed = seal_manifest(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            existing = load_and_validate_manifest(destination)
        except PreflightError as error:
            if not overwrite:
                raise PreflightError(
                    f"refusing to replace invalid existing preflight {destination}; "
                    "request an explicit overwrite"
                ) from error
        else:
            if existing["manifest_sha256"] == sealed["manifest_sha256"]:
                return existing
            if not overwrite:
                raise PreflightError(
                    f"refusing to replace divergent preflight {destination}; "
                    "request an explicit overwrite"
                )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return sealed


def _task_relative_path(task_root: Path, image_path: Path) -> str:
    return Path(os.path.relpath(image_path, task_root)).as_posix()


def _selection_metadata(round_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    train = set(map(str, round_manifest.get("train") or []))
    audit = set(map(str, round_manifest.get("audit") or []))
    overlap = train.intersection(audit)
    if overlap:
        raise PreflightError(f"train/audit overlap in frozen round: {sorted(overlap)[:5]}")
    role_by_image: dict[str, str] = {}
    for role, paths in (round_manifest.get("roles") or {}).items():
        for path in map(str, paths or []):
            if path in role_by_image:
                raise PreflightError(
                    f"image assigned to multiple acquisition roles: {path!r}"
                )
            role_by_image[path] = str(role)
    diagnostics = round_manifest.get("diagnostics") or {}
    backfill_reasons = diagnostics.get("backfill_reasons") or {}
    shared_backfill_reason = diagnostics.get("backfill_reason")
    metadata: dict[str, dict[str, Any]] = {}
    for path in train | audit:
        role = role_by_image.get(path)
        reason = None
        if role is None:
            if isinstance(backfill_reasons, dict):
                reason = backfill_reasons.get(path)
            if reason is None and isinstance(shared_backfill_reason, str):
                reason = shared_backfill_reason
        metadata[path] = {
            "partition": "train" if path in train else "audit",
            "acquisition_role": role or "backfill",
            "backfill_reason": reason,
        }
    return metadata


def _selection_contract_sha256(round_manifest: dict[str, Any]) -> str:
    """Hash stable selection identity while allowing retry bookkeeping to change."""
    metadata = _selection_metadata(round_manifest)
    contract = {
        "stage": str(round_manifest.get("stage")),
        "round": int(round_manifest.get("round", -1)),
        "train": list(map(str, round_manifest.get("train") or [])),
        "audit": list(map(str, round_manifest.get("audit") or [])),
        "roles": {
            str(role): list(map(str, paths or []))
            for role, paths in sorted((round_manifest.get("roles") or {}).items())
        },
        "selection_metadata": {
            path: metadata[path] for path in sorted(metadata)
        },
    }
    return _sha256_json(contract)


def _row_outbound_facts(
    *,
    order: int,
    dataset: str,
    task: str,
    round_no: int,
    logical_image: str,
    adapter: Any,
    modeled_attributes: list[str],
    attributes_text: str,
    public_config: dict[str, Any],
) -> dict[str, Any]:
    logical_key = str(adapter.vqa_to_key(str(logical_image)))
    image_path = Path(adapter.resolve_image_path(logical_key)).resolve()
    if not image_path.is_file():
        raise PreflightError(f"selected VQA image not found: {image_path}")
    prompt = expanded_prompt(
        public_config["prompt_template"], logical_key, attributes_text
    )
    return {
        "order": int(order),
        "dataset": str(dataset),
        "task": str(task),
        "round": int(round_no),
        "logical_image": logical_key,
        "task_relative_image_path": _task_relative_path(adapter.task_root, image_path),
        "absolute_image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "image_bytes": int(image_path.stat().st_size),
        "attributes": list(modeled_attributes),
        "attributes_text_sha256": _sha256_bytes(attributes_text.strip().encode("utf-8")),
        "expanded_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "endpoint": public_config["endpoint"],
        "model": public_config["model"],
        "config_path": public_config["config_path"],
        "config_sha256": public_config["config_sha256"],
        "redacted_config_sha256": public_config["redacted_config_sha256"],
        "prompt_template_sha256": public_config["prompt_template_sha256"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PreflightError(f"expected JSON object: {path}")
    return payload


def _strict_pending_selection(
    harness: Any,
    iterative_runner: Any,
    adapter: Any,
    stage: str,
    modeled_attributes: list[str],
    round_manifest: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    sources = iterative_runner._all_cached_sources(adapter, stage)
    vqa_map, _, audit = iterative_runner._load_iterative_cache(
        sources,
        adapter,
        adapter.task_root / "qa" / stage,
    )
    conflicts = audit.get("conflicts") or {}
    if conflicts:
        raise PreflightError(
            "strict VQA cache contains conflicting complete labels; refusing preflight: "
            f"count={len(conflicts)}, examples={sorted(conflicts)[:3]}"
        )
    committed_audit = iterative_runner._committed_cache_audit(
        state, vqa_map, audit
    )
    if not committed_audit.get("ok"):
        raise PreflightError(
            "committed iterative VQA cache is invalid; refusing outbound preflight: "
            f"earliest_round={committed_audit.get('earliest_invalid_committed_round')}, "
            f"invalid_rounds={len(committed_audit.get('invalid_committed_rounds') or [])}, "
            f"uncommitted_conflicts={len(committed_audit.get('uncommitted_conflicts') or [])}"
        )
    selected = list(
        dict.fromkeys(
            map(
                str,
                list(round_manifest.get("train") or [])
                + list(round_manifest.get("audit") or []),
            )
        )
    )
    missing = [
        path
        for path in selected
        if not harness.is_complete_binary_labels(
            vqa_map.get(path), modeled_attributes
        )
    ]
    saved = list(map(str, round_manifest.get("to_label") or []))
    diagnostic = {
        "schema": "preflight-strict-cache-recomputation-v1",
        "cache_source_count": len(sources),
        "cache_source_set_sha256": audit.get("source_set_sha256"),
        "priority_policy": audit.get("priority_policy"),
        "committed_cache_audit_schema": committed_audit.get("schema"),
        "pending_legacy_snapshot_migrations": len(
            committed_audit.get("migrations") or []
        ),
        "strict_complete_image_count": int(audit.get("complete_image_count", 0)),
        "shadowed_conflict_count": len(audit.get("shadowed_conflicts") or {}),
        "selected_count": len(selected),
        "saved_to_label_count": len(saved),
        "strict_missing_count": len(missing),
        "saved_matches_strict": set(saved) == set(missing),
        "strict_missing_sha256": _sha256_json(sorted(missing)),
    }
    return missing, diagnostic


def validate_strict_cache_preflight_state(
    tasks: Iterable[dict[str, Any]],
    *,
    state_paths: dict[int, Path],
    stage: str,
) -> list[dict[str, Any]]:
    """Read-only gate: reject invalid committed caches before artifact writes."""
    import run_retrieval_harness as harness
    import iterative_vqa_runner as iterative_runner

    results: list[dict[str, Any]] = []
    for task_spec in sorted(tasks, key=lambda item: int(item["order"])):
        order = int(task_spec["order"])
        state_path = Path(state_paths[order]).resolve()
        if not state_path.is_file():
            results.append({"order": order, "status": "no_state"})
            continue
        state = _read_json(state_path)
        status = str(state.get("status") or "unknown")
        if status in {
            "completed",
            "preflight_batch_completed",
            "ready_to_commit_refrozen",
        }:
            results.append({"order": order, "status": status})
            continue
        if status != "awaiting_vqa":
            raise PreflightError(
                f"task order {order} is not in an exportable acquisition state: {status}"
            )
        harness.configure(
            task_spec["dataset"], task_spec["task"], task_spec.get("joint_label")
        )
        adapter = harness.ADAPTER
        sources = iterative_runner._all_cached_sources(adapter, stage)
        vqa_map, _, audit = iterative_runner._load_iterative_cache(
            sources, adapter, adapter.task_root / "qa" / stage
        )
        conflicts = audit.get("conflicts") or {}
        if conflicts:
            raise PreflightError(
                f"task order {order} has {len(conflicts)} unresolved strict cache conflicts"
            )
        committed = iterative_runner._committed_cache_audit(state, vqa_map, audit)
        if not committed.get("ok"):
            raise PreflightError(
                f"task order {order} has invalid committed iterative cache: "
                f"earliest_round={committed.get('earliest_invalid_committed_round')}"
            )
        results.append(
            {
                "order": order,
                "status": status,
                "strict_complete_image_count": int(
                    audit.get("complete_image_count", 0)
                ),
                "cache_source_set_sha256": audit.get("source_set_sha256"),
                "pending_legacy_snapshot_migrations": len(
                    committed.get("migrations") or []
                ),
            }
        )
    return results


def build_preflight_manifest(
    tasks: Iterable[dict[str, Any]],
    *,
    state_paths: dict[int, Path],
    stage: str,
    vqa_config: Path,
) -> dict[str, Any]:
    """Aggregate all currently frozen, cache-missing task rows without networking."""
    import run_retrieval_harness as harness
    import iterative_vqa_runner as iterative_runner

    public_config = load_public_vqa_config(Path(vqa_config))
    rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for task_spec in sorted(tasks, key=lambda item: int(item["order"])):
        order = int(task_spec["order"])
        state_path = Path(state_paths[order]).resolve()
        if not state_path.is_file():
            continue
        state = _read_json(state_path)
        if state.get("status") == "completed":
            continue
        if state.get("status") != "awaiting_vqa" or state.get("pending_round") is None:
            continue
        round_no = int(state["pending_round"])
        harness.configure(
            task_spec["dataset"], task_spec["task"], task_spec.get("joint_label")
        )
        adapter = harness.ADAPTER
        modeled_attributes = list(harness.ATTRS)
        declared_attributes = list(task_spec.get("attributes") or [])
        if declared_attributes and declared_attributes != modeled_attributes:
            raise PreflightError(
                f"modeled attributes changed for order {order}: "
                f"manifest={declared_attributes!r}, runtime={modeled_attributes!r}"
            )
        attr_path = adapter.task_root / "qa" / "attributes.txt"
        if not attr_path.is_file():
            raise PreflightError(
                f"preflight refuses implicit attribute extraction; file missing: {attr_path}"
            )
        attributes_text = _validate_attribute_text(
            attr_path.read_text(encoding="utf-8-sig"),
            modeled_attributes,
            source=attr_path,
        )
        round_path = (
            adapter.task_root
            / "qa"
            / stage
            / f"round_{round_no:02d}"
            / "manifest.json"
        )
        round_manifest = _read_json(round_path)
        if int(round_manifest.get("round", -1)) != round_no:
            raise PreflightError(f"round mismatch in {round_path}")
        if str(round_manifest.get("stage")) != str(stage):
            raise PreflightError(f"stage mismatch in {round_path}")
        pending, strict_cache = _strict_pending_selection(
            harness,
            iterative_runner,
            adapter,
            stage,
            modeled_attributes,
            round_manifest,
            state,
        )
        if not pending:
            # A terminal-rejection replacement can be satisfied entirely by
            # strict cache reuse.  It needs no outbound approval and the runner
            # will commit it on resume; omitting it here keeps preflight scoped
            # to rows that can actually leave the machine.
            continue
        if len(set(pending)) != len(pending):
            raise PreflightError(f"duplicate to_label rows in frozen round: {round_path}")
        metadata = _selection_metadata(round_manifest)
        task_rows = []
        for logical_image in pending:
            if logical_image not in metadata:
                raise PreflightError(
                    f"pending image is absent from frozen train/audit selection: {logical_image}"
                )
            row = _row_outbound_facts(
                order=order,
                dataset=task_spec["dataset"],
                task=task_spec["task"],
                round_no=round_no,
                logical_image=logical_image,
                adapter=adapter,
                modeled_attributes=modeled_attributes,
                attributes_text=attributes_text,
                public_config=public_config,
            )
            row.update(metadata[logical_image])
            task_rows.append(row)
        task_rows.sort(key=lambda item: item["logical_image"])
        rows.extend(task_rows)
        task_summaries.append(
            {
                "order": order,
                "dataset": str(task_spec["dataset"]),
                "task": str(task_spec["task"]),
                "round": round_no,
                "row_count": len(task_rows),
                "attributes": modeled_attributes,
                "attributes_file": str(attr_path.resolve()),
                "attributes_text_sha256": _sha256_bytes(
                    attributes_text.encode("utf-8")
                ),
                "frozen_round_manifest": str(round_path.resolve()),
                "frozen_round_manifest_sha256": sha256_file(round_path),
                "selection_contract_sha256": _selection_contract_sha256(
                    round_manifest
                ),
                "strict_cache": strict_cache,
            }
        )
    rows.sort(
        key=lambda item: (
            int(item["order"]), int(item["round"]), str(item["logical_image"])
        )
    )
    payload = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "scope": "currently frozen cache-missing iterative VQA rows only",
        "network_used": False,
        "approval_semantics": (
            "runtime may send only an exact-byte subset of these rows; one approved "
            "batch per task invocation"
        ),
        "provider": {
            "endpoint": public_config["endpoint"],
            "model": public_config["model"],
            "config_path": public_config["config_path"],
            "config_sha256": public_config["config_sha256"],
            "redacted_config_sha256": public_config["redacted_config_sha256"],
            "prompt_template_sha256": public_config["prompt_template_sha256"],
        },
        "task_count": len(task_summaries),
        "row_count": len(rows),
        "tasks": task_summaries,
        "rows": rows,
    }
    return seal_manifest(payload)


_RUNTIME_FIELDS = (
    "order",
    "dataset",
    "task",
    "round",
    "logical_image",
    "task_relative_image_path",
    "absolute_image_path",
    "image_sha256",
    "image_bytes",
    "attributes",
    "attributes_text_sha256",
    "expanded_prompt_sha256",
    "endpoint",
    "model",
    "config_path",
    "config_sha256",
    "redacted_config_sha256",
    "prompt_template_sha256",
)


def validate_runtime_batch(
    preflight_path: Path,
    expected_manifest_sha256: str,
    *,
    order: int,
    dataset: str,
    task: str,
    stage: str,
    round_no: int,
    candidate_rels: Iterable[str],
    adapter: Any,
    modeled_attributes: list[str],
    attributes_file: Path,
    vqa_config: Path,
) -> dict[str, Any]:
    """Validate exact bytes/config immediately before a provider invocation.

    A subset is permitted so failed rows can be retried using the same approval;
    no row absent from the approved manifest can pass.
    """
    manifest = load_and_validate_manifest(preflight_path, expected_manifest_sha256)
    if str(manifest.get("stage")) != str(stage):
        raise PreflightError(
            f"preflight stage mismatch: {manifest.get('stage')!r} != {stage!r}"
        )
    attributes_path = Path(attributes_file).resolve()
    if not attributes_path.is_file():
        raise PreflightError(f"VQA attributes file missing at call time: {attributes_path}")
    attributes_text = _validate_attribute_text(
        attributes_path.read_text(encoding="utf-8-sig"),
        list(modeled_attributes),
        source=attributes_path,
    )
    public_config = load_public_vqa_config(Path(vqa_config))
    task_summaries = [
        summary
        for summary in manifest.get("tasks") or []
        if int(summary.get("order", -1)) == int(order)
        and str(summary.get("dataset")) == str(dataset)
        and str(summary.get("task")) == str(task)
        and int(summary.get("round", -1)) == int(round_no)
    ]
    if len(task_summaries) != 1:
        raise PreflightError(
            "preflight must contain exactly one task summary for this runtime batch"
        )
    summary = task_summaries[0]
    round_path = (
        Path(adapter.task_root)
        / "qa"
        / str(stage)
        / f"round_{int(round_no):02d}"
        / "manifest.json"
    ).resolve()
    if str(round_path) != str(summary.get("frozen_round_manifest")):
        raise PreflightError("frozen round manifest path changed after approval")
    round_manifest = _read_json(round_path)
    selection_hash = _selection_contract_sha256(round_manifest)
    if selection_hash != summary.get("selection_contract_sha256"):
        raise PreflightError("frozen train/audit/role selection changed after approval")
    current_selection = _selection_metadata(round_manifest)
    approved = {
        str(row["logical_image"]): row
        for row in manifest["rows"]
        if int(row["order"]) == int(order)
        and str(row["dataset"]) == str(dataset)
        and str(row["task"]) == str(task)
        and int(row["round"]) == int(round_no)
    }
    candidates = list(dict.fromkeys(map(str, candidate_rels)))
    if not candidates:
        raise PreflightError("refusing an empty VQA runtime batch")
    validated = []
    for logical_image in candidates:
        runtime = _row_outbound_facts(
            order=order,
            dataset=dataset,
            task=task,
            round_no=round_no,
            logical_image=logical_image,
            adapter=adapter,
            modeled_attributes=list(modeled_attributes),
            attributes_text=attributes_text,
            public_config=public_config,
        )
        approved_row = approved.get(runtime["logical_image"])
        if approved_row is None:
            raise PreflightError(
                f"runtime image was not approved for this task/round: {runtime['logical_image']}"
            )
        changed = [
            field
            for field in _RUNTIME_FIELDS
            if approved_row.get(field) != runtime.get(field)
        ]
        if changed:
            raise PreflightError(
                f"runtime payload changed for {runtime['logical_image']}: {changed}"
            )
        current_meta = current_selection.get(runtime["logical_image"])
        approved_meta = {
            key: approved_row.get(key)
            for key in ("partition", "acquisition_role", "backfill_reason")
        }
        if current_meta != approved_meta:
            raise PreflightError(
                f"selection metadata changed for {runtime['logical_image']}"
            )
        validated.append(runtime["logical_image"])
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "order": int(order),
        "dataset": str(dataset),
        "task": str(task),
        "round": int(round_no),
        "validated_rows": len(validated),
        "logical_images": validated,
    }
