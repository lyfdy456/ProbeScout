"""Build an interactive rank-normalized parallel-coordinate plot.

Each horizontal axis is one retrieval method. Each polyline is one image. The
coordinate on an axis is that image's descending-score rank percentile:
1 means highest ranked and 0 means lowest ranked. The tool reads the audited
ProbeBank caches, aligns every method by embedding_index/records_hash, supports
one seed or five-seed aggregation, and can reconstruct the formal Ours-Full
score from its saved train-only fit.

The generated HTML uses a dependency-free Canvas renderer. It embeds all rank
values, but initially draws a priority subset for responsiveness; the line-count
slider can be moved up to the full split size.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
LINEAR_ROOT = ROOT / "probe_learning"
SCRIPT_ROOT = LINEAR_ROOT / "scripts"
for path in (LINEAR_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_probebank_batch as B  # noqa: E402
import run_retrieval_harness as H  # noqa: E402
from iterative_vqa_runner import _frozen_split  # noqa: E402


DEFAULT_MANIFEST = ROOT / "configs/main17_tasks.json"
DEFAULT_SUITE = ROOT / "configs/probe_suite.json"
DEFAULT_OURS_FULL_ROOT = (
    ROOT / "outputs" / "formal23_iterative_selected8_r_softgate_front_minmax_allv2_v1"
)
OURS_FULL_ID = "r_softgate_minmax_all_gridinit_joint_bce_fixed_t"
DISPLAY_NAMES = {
    "image_prototype": "Image Prototype",
    "query_maxsim": "Query MaxSim",
    "img_text_fusion": "Image--Text Fusion",
    "text_prompt_ensemble": "Text Prompt Ensemble",
    "zscore_img_text_fusion": "Z-score Image--Text Fusion",
    "mlp_baseline": "MLP",
    "kfold_pu": "K-Fold",
    "pu_ranking": "Ours-PURA",
    "triplet_loss": "Triplet Loss",
    "attention_pooling": "Attention Pooling",
    "attribute_conditioned_attention": "Attribute-conditioned Attention",
    "nnpu": "nnPU",
    "dcpu": "DC-PU",
    OURS_FULL_ID: "Ours-Full",
}
EMBEDDING_BASELINES = [
    "image_prototype",
    "query_maxsim",
    "img_text_fusion",
    "text_prompt_ensemble",
    "zscore_img_text_fusion",
]
LEARNED_METHOD_ORDER = [
    "mlp_baseline",
    "kfold_pu",
    "triplet_loss",
    "attention_pooling",
    "attribute_conditioned_attention",
    "nnpu",
    "dcpu",
    "pu_ranking",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def portable_path(path: Path) -> str:
    """Use a project-relative path when possible; never expose the host root."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return resolved.name


def task_key(task: dict[str, Any]) -> str:
    return f"{int(task['order']):03d}_{task['dataset']}_{task['task']}"


def find_task(manifest: dict[str, Any], order: int) -> dict[str, Any]:
    matches = [row for row in manifest["tasks"] if int(row["order"]) == int(order)]
    if len(matches) != 1:
        raise ValueError(f"expected one task for order {order}, found {len(matches)}")
    return matches[0]


def resolve_attribute(requested: str, attributes: list[str]) -> str | None:
    if requested.lower() == "joint":
        return None
    exact = [attr for attr in attributes if attr.lower() == requested.lower()]
    if len(exact) == 1:
        return exact[0]
    slug_matches = [attr for attr in attributes if B.slug(attr) == B.slug(requested)]
    if len(slug_matches) == 1:
        return slug_matches[0]
    raise ValueError(
        f"unknown score key {requested!r}; use 'joint' or one of {attributes}"
    )


def source_root(suite: dict[str, Any], method: str) -> Path:
    roots = suite["probe_source_roots"]
    relative = roots.get(method, roots["__default__"])
    return (ROOT / relative).resolve()


def load_score_tensor(
    suite: dict[str, Any],
    task: dict[str, Any],
    methods: list[str],
    attributes: list[str],
    database_paths: list[str],
    stage: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[int]], list[dict[str, Any]]]:
    """Load and audit method x attribute x seed x image score arrays."""
    expected_hash = B.stable_hash(database_paths)
    tensors: dict[str, dict[str, np.ndarray]] = {}
    seeds_by_method: dict[str, list[int]] = {}
    audit: list[dict[str, Any]] = []
    for method in methods:
        method_tensors: dict[str, np.ndarray] = {}
        method_seeds: list[int] | None = None
        for attribute in attributes:
            entry = B.cache_dir(
                source_root(suite, method) / "task_isolated" / task_key(task),
                task["dataset"],
                task["task"],
                stage,
                method,
                attribute,
            )
            metadata_path = entry / "metadata.json"
            scores_path = entry / "scores.npz"
            if not metadata_path.is_file() or not scores_path.is_file():
                raise FileNotFoundError(
                    f"missing score cache for {task_key(task)}/{method}/{attribute}: {entry}"
                )
            metadata = load_json(metadata_path)
            required = {
                "dataset": task["dataset"],
                "task": task["task"],
                "supervision_stage": stage,
                "method": method,
                "canonical_attribute": attribute,
                "records_hash": expected_hash,
                "score_length": len(database_paths),
            }
            mismatches = {
                key: {"observed": metadata.get(key), "expected": value}
                for key, value in required.items()
                if metadata.get(key) != value
            }
            seeds = [int(value) for value in metadata.get("seeds", [])]
            with np.load(scores_path, allow_pickle=False) as payload:
                values = np.asarray(payload["scores"], dtype=np.float32)
            if values.shape != (len(seeds), len(database_paths)):
                mismatches["scores_shape"] = {
                    "observed": list(values.shape),
                    "expected": [len(seeds), len(database_paths)],
                }
            if mismatches:
                raise ValueError(
                    f"score cache identity mismatch at {entry}: "
                    + json.dumps(mismatches, ensure_ascii=False)
                )
            if method_seeds is None:
                method_seeds = seeds
            elif method_seeds != seeds:
                raise ValueError(
                    f"seed order differs across attributes for {method}: "
                    f"{method_seeds} vs {seeds}"
                )
            method_tensors[attribute] = values
            audit.append(
                {
                    "method": method,
                    "attribute": attribute,
                    "scores": str(scores_path.relative_to(ROOT)).replace("\\", "/"),
                    "metadata": str(metadata_path.relative_to(ROOT)).replace("\\", "/"),
                    "shape": list(values.shape),
                    "records_hash": expected_hash,
                    "seeds": seeds,
                }
            )
        tensors[method] = method_tensors
        seeds_by_method[method] = method_seeds or []
    return tensors, seeds_by_method, audit


def split_indices(adapter: Any, split: str) -> tuple[list[str], np.ndarray]:
    database = B.database_paths(adapter)
    gallery, test, train = _frozen_split(
        adapter, None, adapter.p2i, adapter.gt_by_attr
    )
    normalized = split.lower()
    if normalized in {"all", "gallery"}:
        paths = gallery
    elif normalized == "test":
        paths = test
    elif normalized == "train_pool":
        paths = train
    else:
        raise ValueError("split must be one of: all, gallery, test, train_pool")
    indices = np.asarray([adapter.p2i[path] for path in paths], dtype=np.int64)
    if normalized in {"all", "gallery"} and paths != database:
        raise RuntimeError("Gallery no longer matches full embedding order")
    return list(paths), indices


def method_scores_by_seed(
    tensors: dict[str, dict[str, np.ndarray]],
    seeds_by_method: dict[str, list[int]],
    method: str,
    selected_seeds: list[int],
    attributes: list[str],
    target_attribute: str | None,
    indices: np.ndarray,
) -> np.ndarray:
    positions = []
    for seed in selected_seeds:
        try:
            positions.append(seeds_by_method[method].index(seed))
        except ValueError as exc:
            raise ValueError(f"{method} does not contain seed {seed}") from exc
    rows = []
    for position in positions:
        if target_attribute is None:
            per_attribute = [
                np.asarray(tensors[method][attribute][position, indices], dtype=np.float64)
                for attribute in attributes
            ]
            rows.append(np.prod(np.vstack(per_attribute), axis=0))
        else:
            rows.append(
                np.asarray(
                    tensors[method][target_attribute][position, indices],
                    dtype=np.float64,
                )
            )
    return np.vstack(rows)


def embedding_baseline_scores(
    adapter: Any,
    attributes: list[str],
    target_attribute: str | None,
    indices: np.ndarray,
    backbone: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reconstruct the five deterministic baselines on the selected split.

    This mirrors ``run_retrieval_harness.compute_embedding_baselines``. The
    Z-score fusion is standardized separately on the selected split, matching
    the formal evaluation policy.
    """
    embeddings, embedding_meta = H.load_backbone_embeddings(adapter, backbone)
    embeddings = H.l2norm(embeddings)
    query_indices = np.asarray(adapter.query_idx, dtype=np.int64)
    if query_indices.size == 0:
        raise ValueError("embedding baselines require at least one query image")
    selected = embeddings[indices]
    prototype = H.l2norm(
        embeddings[query_indices].mean(axis=0, keepdims=True)
    ).squeeze()
    image_prototype = np.asarray(selected @ prototype, dtype=np.float32)
    query_maxsim = np.asarray(
        np.max(selected @ embeddings[query_indices].T, axis=1), dtype=np.float32
    )

    text_features = H.text_feats_for_backbone(backbone)
    fusion_by_attribute: dict[str, np.ndarray] = {}
    for position, attribute in enumerate(attributes):
        fused = H.l2norm(
            (0.5 * prototype + 0.5 * text_features[position]).reshape(1, -1)
        ).squeeze()
        fusion_by_attribute[attribute] = np.asarray(selected @ fused, dtype=np.float32)
    image_text_fusion = (
        np.mean(
            np.vstack([fusion_by_attribute[attr] for attr in attributes]), axis=0
        ).astype(np.float32)
        if target_attribute is None
        else fusion_by_attribute[target_attribute]
    )

    # Prompt-ensemble dictionaries are keyed by the adapter's stable ranking
    # slug (for example ``bmw``), not necessarily by the human-readable
    # canonical attribute (for example ``BMW brand identity``).  HICO's
    # Cat/Hugging task happened to use identical values and previously hid
    # this distinction.
    ranking_key = (
        H.JOINT_KEY
        if target_attribute is None
        else H.ranking_key_for_combo([target_attribute])
    )
    prompt_features = H.prompt_ensemble_feats_for_backbone(backbone)
    text_prompt_ensemble = np.asarray(
        selected @ prompt_features[ranking_key], dtype=np.float32
    )

    def split_zscore(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - float(np.mean(values))) / (float(np.std(values)) + 1e-6)

    zscore_fusion = np.asarray(
        0.5 * split_zscore(image_prototype)
        + 0.5 * split_zscore(text_prompt_ensemble),
        dtype=np.float32,
    )
    scores = {
        "image_prototype": image_prototype,
        "query_maxsim": query_maxsim,
        "img_text_fusion": image_text_fusion,
        "text_prompt_ensemble": text_prompt_ensemble,
        "zscore_img_text_fusion": zscore_fusion,
    }
    audit = {
        "kind": "deterministic_embedding_baseline_reconstruction",
        "backbone": backbone,
        "embedding_dim": int(embedding_meta["embedding_dim"]),
        "query_count": int(query_indices.size),
        "split_score_length": int(len(indices)),
        "zscore_policy": "selected-split-local unlabeled standardization",
        "text_model": H.BACKBONE_TEXT_MODEL.get(backbone),
    }
    return scores, audit


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    mask = values >= 0.0
    output[mask] = 1.0 / (1.0 + np.exp(-values[mask]))
    exp_values = np.exp(values[~mask])
    output[~mask] = exp_values / (1.0 + exp_values)
    return output


def reconstruct_ours_full_scores(
    tensors: dict[str, dict[str, np.ndarray]],
    seeds_by_method: dict[str, list[int]],
    methods: list[str],
    attributes: list[str],
    target_attribute: str | None,
    selected_seeds: list[int],
    indices: np.ndarray,
    task: dict[str, Any],
    ours_full_root: Path,
    member_weights: dict[str, float] | None = None,
    member_weights_by_attribute: dict[str, dict[str, float]] | None = None,
    attribute_weights: dict[str, float] | None = None,
    expected_stage: str | None = None,
) -> np.ndarray:
    result_path = ours_full_root / "tasks" / task_key(task) / "front_minmax_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing Ours-Full fit: {result_path}")
    payload = load_json(result_path)
    expected_task = task_key(task)
    fit_audit = payload.get("audit", {})
    if int(fit_audit.get("order", -1)) != int(task["order"]):
        raise ValueError(f"Ours-Full fit order mismatch at {result_path}")
    if fit_audit.get("task") != expected_task:
        raise ValueError(f"Ours-Full fit task mismatch at {result_path}")
    if list(fit_audit.get("attributes", [])) != list(attributes):
        raise ValueError(f"Ours-Full fit attribute mismatch at {result_path}")
    if expected_stage is not None:
        if fit_audit.get("stage") != expected_stage:
            raise ValueError(
                f"Ours-Full fit stage mismatch at {result_path}: "
                f"{fit_audit.get('stage')!r} != {expected_stage!r}"
            )
        source_entries = fit_audit.get("source_entries")
        if not isinstance(source_entries, list):
            raise ValueError(f"Ours-Full fit source audit is missing at {result_path}")
        expected_sources = {
            (method, attribute)
            for method in methods
            for attribute in attributes
        }
        observed_sources = {
            (str(row.get("method")), str(row.get("attribute")))
            for row in source_entries
            if isinstance(row, dict)
        }
        if observed_sources != expected_sources or len(source_entries) != len(
            expected_sources
        ):
            raise ValueError(f"Ours-Full fit source coverage drifted at {result_path}")
        if any(
            row.get("stage") != expected_stage
            or not bool(row.get("supervision_hash_match"))
            or row.get("supervision_hash") != row.get("current_supervision_hash")
            for row in source_entries
        ):
            raise ValueError(
                f"Ours-Full fit source stage/supervision audit failed at {result_path}"
            )
    fits = payload["audit"]["fits"]
    normalization_rows = payload["normalization_rows"]
    normalization: dict[tuple[int, str, str], tuple[float, float]] = {}
    for row in normalization_rows:
        key = (int(row["seed"]), row["method"], row["attribute"])
        if key in normalization:
            raise ValueError(f"duplicate Ours-Full normalization row {key}")
        normalization[key] = (float(row["train_min"]), float(row["train_max"]))
    output = []
    for seed in selected_seeds:
        if str(seed) not in fits or OURS_FULL_ID not in fits[str(seed)]:
            raise ValueError(f"Ours-Full fit lacks seed {seed}: {result_path}")
        model = fits[str(seed)][OURS_FULL_ID]
        if model.get("probe_aggregation") != "mean":
            raise ValueError(
                "only the saved mean probe aggregation is supported, got "
                f"{model.get('probe_aggregation')!r}"
            )
        gates = {}
        for attribute in attributes:
            normalized_members = []
            selected_members = list(
                model.get("selected_members_by_attr", {}).get(attribute, [])
            )
            if not selected_members:
                raise ValueError(
                    f"Ours-Full fit has no selected members for {attribute!r}"
                )
            unknown = [member for member in selected_members if member not in methods]
            if unknown:
                raise ValueError(
                    f"Ours-Full fit references unavailable members {unknown}"
                )
            for method in selected_members:
                position = seeds_by_method[method].index(seed)
                raw = np.asarray(
                    tensors[method][attribute][position, indices], dtype=np.float64
                )
                normalization_key = (seed, method, attribute)
                if normalization_key not in normalization:
                    raise ValueError(
                        f"missing Ours-Full normalization row {normalization_key}"
                    )
                low, high = normalization[normalization_key]
                span = high - low
                if span <= 0.0:
                    normalized = np.zeros_like(raw)
                else:
                    normalized = np.clip((raw - low) / span, 0.0, 1.0)
                normalized_members.append(normalized)
            member_matrix = np.vstack(normalized_members)
            active_member_weights = (
                member_weights_by_attribute.get(attribute)
                if member_weights_by_attribute is not None
                else member_weights
            )
            if active_member_weights is None:
                attribute_input = member_matrix.mean(axis=0)
            else:
                weights = np.asarray(
                    [active_member_weights.get(method, 0.0) for method in selected_members],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                    raise ValueError("Ours-Full member weights must be finite and non-negative")
                if float(np.sum(weights)) <= 0.0:
                    raise ValueError(
                        f"Ours-Full member weights sum to zero for {attribute!r}"
                    )
                # Preserve the original arithmetic-mean code path exactly for
                # equal weights.  This makes the default interactive weighted
                # fusion numerically auditable against the exported Ours-Full.
                if np.all(weights == weights[0]):
                    attribute_input = member_matrix.mean(axis=0)
                else:
                    attribute_input = np.average(
                        member_matrix,
                        axis=0,
                        weights=weights,
                    )
            theta = float(model["theta_by_attr"][attribute])
            temperature = float(model["temperature_by_attr"][attribute])
            if not np.isfinite(theta) or not np.isfinite(temperature) or temperature <= 0:
                raise ValueError(
                    f"invalid Ours-Full theta/temperature for seed={seed}, "
                    f"attribute={attribute}: theta={theta}, T={temperature}"
                )
            gates[attribute] = stable_sigmoid(
                (attribute_input - theta) / temperature
            )
        if target_attribute is None:
            if attribute_weights is None:
                output.append(np.prod(np.vstack([gates[attr] for attr in attributes]), axis=0))
            else:
                exponents = np.asarray(
                    [attribute_weights.get(attribute, 0.0) for attribute in attributes],
                    dtype=np.float64,
                )
                if (
                    not np.all(np.isfinite(exponents))
                    or np.any(exponents < 0.0)
                    or float(np.sum(exponents)) <= 0.0
                ):
                    raise ValueError(
                        "Ours-Full attribute weights must be finite, non-negative and not all zero"
                    )
                gated = np.vstack([gates[attr] for attr in attributes])
                output.append(np.prod(np.power(gated, exponents[:, None]), axis=0))
        else:
            output.append(gates[target_attribute])
    return np.vstack(output)


def normalized_rank(values: np.ndarray, tie_policy: str) -> np.ndarray:
    """Map scores to [0,1], with best=1 and worst=0."""
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count <= 1:
        return np.ones(count, dtype=np.float32)
    if tie_policy == "average":
        ranks = rankdata(values, method="average") - 1.0
    elif tie_policy == "stable":
        order = np.argsort(values, kind="stable")
        ranks = np.empty(count, dtype=np.float64)
        ranks[order] = np.arange(count, dtype=np.float64)
    else:
        raise ValueError("tie_policy must be 'average' or 'stable'")
    return np.asarray(ranks / (count - 1.0), dtype=np.float32)


def aggregate_seed_ranks(
    score_rows: np.ndarray,
    seed_mode: str,
    tie_policy: str,
) -> np.ndarray:
    if seed_mode == "seed":
        if len(score_rows) != 1:
            raise ValueError("seed mode expects exactly one selected seed")
        return normalized_rank(score_rows[0], tie_policy)
    if seed_mode == "mean-score":
        return normalized_rank(score_rows.mean(axis=0), tie_policy)
    if seed_mode == "mean-rank":
        ranks = np.vstack([
            normalized_rank(score_rows[index], tie_policy)
            for index in range(len(score_rows))
        ])
        return np.asarray(ranks.mean(axis=0), dtype=np.float32)
    raise ValueError(f"unknown seed mode: {seed_mode}")


def priority_order(ranks: np.ndarray, top_per_method: int, seed: int) -> np.ndarray:
    """Prioritize Top-K union and high-disagreement images, then fill randomly."""
    count, method_count = ranks.shape
    selected = np.zeros(count, dtype=bool)
    priority: list[int] = []

    def add(values: np.ndarray) -> None:
        for value in values:
            index = int(value)
            if not selected[index]:
                selected[index] = True
                priority.append(index)

    top_count = min(max(int(top_per_method), 0), count)
    if top_count:
        for method_index in range(method_count):
            top = np.argpartition(ranks[:, method_index], count - top_count)[
                count - top_count:
            ]
            top = top[np.argsort(-ranks[top, method_index], kind="stable")]
            add(top)
    disagreement = np.ptp(ranks, axis=1)
    disagreement_order = np.argsort(-disagreement, kind="stable")
    add(disagreement_order[: min(2000, count)])
    remaining = np.flatnonzero(~selected)
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    add(remaining)
    return np.asarray(priority, dtype=np.uint32)


def encode_array(values: np.ndarray, dtype: str) -> str:
    packed = np.asarray(values, dtype=dtype, order="C")
    return base64.b64encode(packed.tobytes(order="C")).decode("ascii")


def build_html(
    output: Path,
    ranks: np.ndarray,
    method_names: list[str],
    image_ids: list[str],
    order: np.ndarray,
    metadata: dict[str, Any],
    initial_lines: int,
) -> None:
    rows, columns = ranks.shape
    payload = {
        "rows": rows,
        "columns": columns,
        "methods": method_names,
        "images": image_ids,
        "rankFloat32": encode_array(ranks, "<f4"),
        "priorityUint32": encode_array(order, "<u4"),
        "initialLines": int(min(max(initial_lines, 1), rows)),
        "metadata": metadata,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_json = payload_json.replace("</", "<\\/")
    title = html.escape(
        f"Rank PCP — {metadata['task']} — {metadata['score_key']}", quote=True
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #ffffff;
  --fg: #172033;
  --muted: #64748b;
  --line: #94a3b8;
  --axis: #475569;
  --accent: #2563eb;
  --highlight: #dc2626;
  --control: #f8fafc;
  --border: #cbd5e1;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0b1220;
    --fg: #e5edf7;
    --muted: #9aa9bd;
    --line: #64748b;
    --axis: #cbd5e1;
    --accent: #60a5fa;
    --highlight: #fb7185;
    --control: #111b2d;
    --border: #334155;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 18px;
  background: var(--bg);
  color: var(--fg);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
.toolbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px 18px;
  align-items: end;
  margin-bottom: 10px;
}}
.field {{ display: grid; gap: 4px; min-width: 145px; }}
.field.wide {{ flex: 1 1 280px; }}
label {{ color: var(--muted); font-size: 12px; }}
select, input, button {{
  font: inherit;
  color: var(--fg);
  background: var(--control);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 8px;
}}
input[type=range] {{ width: 100%; padding: 0; }}
button {{ cursor: pointer; }}
.meta, .selection {{
  color: var(--muted);
  min-height: 20px;
  margin: 4px 0;
  overflow-wrap: anywhere;
}}
.plot {{
  position: relative;
  width: 100%;
  min-height: 480px;
  overflow-x: auto;
}}
canvas {{
  display: block;
  width: 100%;
  cursor: crosshair;
}}
.hint {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
</style>
</head>
<body>
<div class="toolbar">
  <div class="field wide">
    <label for="line-count">Visible image lines: <span id="line-value"></span></label>
    <input id="line-count" type="range" min="100" max="{rows}" step="100">
  </div>
  <div class="field">
    <label for="opacity">Line opacity: <span id="opacity-value"></span></label>
    <input id="opacity" type="range" min="0.005" max="0.20" step="0.005" value="0.025">
  </div>
  <div class="field">
    <label for="color-method">Color by rank</label>
    <select id="color-method"><option value="-1">Uniform</option></select>
  </div>
  <div class="field">
    <label for="top-percent">Highlight top-ranked % (boundary ties included)</label>
    <input id="top-percent" type="range" min="0" max="20" step="0.5" value="2">
  </div>
  <button id="reset" type="button">Reset</button>
</div>
<div id="meta" class="meta"></div>
<div id="selection" class="selection">Click near any method row to select the closest visible image.</div>
<div class="plot"><canvas id="pcp" role="img" aria-label="Rank-normalized parallel coordinate plot"></canvas></div>
<div class="hint">0 = lowest rank, 1 = highest rank. Coordinates are score-rank percentiles, or their mean across seeds.</div>
<script id="pcp-data" type="application/json">{payload_json}</script>
<script>
(() => {{
  "use strict";
  const data = JSON.parse(document.getElementById("pcp-data").textContent);
  const decode = (value, Type) => {{
    const binary = atob(value);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Type(bytes.buffer);
  }};
  const ranks = decode(data.rankFloat32, Float32Array);
  const priority = decode(data.priorityUint32, Uint32Array);
  const canvas = document.getElementById("pcp");
  const ctx = canvas.getContext("2d", {{ alpha: true }});
  const lineInput = document.getElementById("line-count");
  const opacityInput = document.getElementById("opacity");
  const colorInput = document.getElementById("color-method");
  const topInput = document.getElementById("top-percent");
  const lineValue = document.getElementById("line-value");
  const opacityValue = document.getElementById("opacity-value");
  const selection = document.getElementById("selection");
  const styles = getComputedStyle(document.documentElement);
  let selected = -1;
  let geometry = null;
  let drawScheduled = false;
  const rankOrderCache = new Map();

  data.methods.forEach((name, index) => {{
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = name;
    colorInput.appendChild(option);
  }});
  lineInput.max = String(data.rows);
  lineInput.step = data.rows < 1000 ? "10" : "100";
  lineInput.value = String(data.initialLines);
  document.getElementById("meta").textContent =
    data.metadata.task + " · " + data.metadata.split + " · " +
    data.metadata.score_key + " · " + data.metadata.seed_description +
    " · " + data.rows.toLocaleString() + " images";

  function rankAt(image, method) {{
    return ranks[image * data.columns + method];
  }}

  function palette(t) {{
    const hue = 225 - 190 * t;
    return "hsl(" + hue.toFixed(1) + " 72% 52%)";
  }}

  function rankOrder(method) {{
    if (!rankOrderCache.has(method)) {{
      const values = Uint32Array.from(
        {{length: data.rows}}, (_, index) => index
      );
      values.sort((a, b) => rankAt(b, method) - rankAt(a, method) || a - b);
      rankOrderCache.set(method, values);
    }}
    return rankOrderCache.get(method);
  }}

  function topCutoff(method, percent) {{
    if (percent <= 0) return Infinity;
    const count = Math.max(1, Math.ceil(data.rows * percent / 100));
    return rankAt(rankOrder(method)[count - 1], method);
  }}

  function resize() {{
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssWidth = Math.max(480, canvas.parentElement.clientWidth);
    const cssHeight = Math.max(480, 82 + data.columns * 72);
    canvas.style.width = cssWidth + "px";
    canvas.style.height = cssHeight + "px";
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    geometry = {{
      width: cssWidth,
      height: cssHeight,
      left: Math.min(220, Math.max(130, cssWidth * 0.22)),
      right: cssWidth - 24,
      top: 40,
      bottom: cssHeight - 32
    }};
    scheduleDraw();
  }}

  function visibleIndices() {{
    const count = Math.min(Number(lineInput.value), data.rows);
    return priority.subarray(0, count);
  }}

  function pathFor(indices) {{
    ctx.beginPath();
    const g = geometry;
    const spanX = g.right - g.left;
    const spanY = g.bottom - g.top;
    const denom = Math.max(data.columns - 1, 1);
    for (let p = 0; p < indices.length; p++) {{
      const image = indices[p];
      for (let method = 0; method < data.columns; method++) {{
        const x = g.left + rankAt(image, method) * spanX;
        const y = g.top + method * spanY / denom;
        if (method === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
    }}
  }}

  function drawAxes() {{
    const g = geometry;
    const spanY = g.bottom - g.top;
    const denom = Math.max(data.columns - 1, 1);
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = styles.getPropertyValue("--axis").trim();
    ctx.fillStyle = styles.getPropertyValue("--fg").trim();
    ctx.font = "12px system-ui, sans-serif";
    for (let method = 0; method < data.columns; method++) {{
      const y = g.top + method * spanY / denom;
      ctx.beginPath();
      ctx.moveTo(g.left, y);
      ctx.lineTo(g.right, y);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(data.methods[method], g.left - 12, y);
      for (const tick of [0, 0.25, 0.5, 0.75, 1]) {{
        const x = g.left + tick * (g.right - g.left);
        ctx.beginPath();
        ctx.moveTo(x, y - 3);
        ctx.lineTo(x, y + 3);
        ctx.stroke();
        if (method === data.columns - 1) {{
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillStyle = styles.getPropertyValue("--muted").trim();
          ctx.fillText(tick.toFixed(2), x, y + 7);
          ctx.fillStyle = styles.getPropertyValue("--fg").trim();
        }}
      }}
    }}
    ctx.restore();
  }}

  function draw() {{
    drawScheduled = false;
    if (!geometry) return;
    const g = geometry;
    ctx.clearRect(0, 0, g.width, g.height);
    const indices = visibleIndices();
    const opacity = Number(opacityInput.value);
    const colorMethod = Number(colorInput.value);
    const topPercent = Number(topInput.value);
    lineValue.textContent = indices.length.toLocaleString() + " / " + data.rows.toLocaleString();
    opacityValue.textContent = opacity.toFixed(3);

    ctx.save();
    ctx.lineWidth = 0.65;
    if (colorMethod < 0) {{
      pathFor(indices);
      ctx.strokeStyle = styles.getPropertyValue("--line").trim();
      ctx.globalAlpha = opacity;
      ctx.stroke();
    }} else {{
      const bins = Array.from({{length: 16}}, () => []);
      for (let i = 0; i < indices.length; i++) {{
        const image = indices[i];
        const bin = Math.min(15, Math.floor(rankAt(image, colorMethod) * 16));
        bins[bin].push(image);
      }}
      bins.forEach((values, bin) => {{
        if (!values.length) return;
        pathFor(values);
        ctx.strokeStyle = palette((bin + 0.5) / 16);
        ctx.globalAlpha = Math.min(0.45, opacity * 1.5);
        ctx.stroke();
      }});
      if (topPercent > 0) {{
        const cutoff = topCutoff(colorMethod, topPercent);
        const highlighted = [];
        for (let i = 0; i < indices.length; i++) {{
          const image = indices[i];
          if (rankAt(image, colorMethod) >= cutoff) highlighted.push(image);
        }}
        if (highlighted.length) {{
          pathFor(highlighted);
          ctx.strokeStyle = styles.getPropertyValue("--highlight").trim();
          ctx.globalAlpha = 0.18;
          ctx.lineWidth = 1.0;
          ctx.stroke();
        }}
      }}
    }}
    ctx.restore();
    drawAxes();
    if (selected >= 0) {{
      pathFor([selected]);
      ctx.strokeStyle = styles.getPropertyValue("--highlight").trim();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }}
  }}

  function scheduleDraw() {{
    if (!drawScheduled) {{
      drawScheduled = true;
      requestAnimationFrame(draw);
    }}
  }}

  function selectNearest(event) {{
    const rect = canvas.getBoundingClientRect();
    const g = geometry;
    if (!g || !rect.width || !rect.height) return;
    const x = (event.clientX - rect.left) * g.width / rect.width;
    const y = (event.clientY - rect.top) * g.height / rect.height;
    const method = Math.max(0, Math.min(
      data.columns - 1,
      Math.round((y - g.top) / (g.bottom - g.top) * Math.max(data.columns - 1, 1))
    ));
    const target = Math.max(0, Math.min(1, (x - g.left) / (g.right - g.left)));
    const indices = visibleIndices();
    let best = -1;
    let bestDistance = Infinity;
    for (let i = 0; i < indices.length; i++) {{
      const image = indices[i];
      const distance = Math.abs(rankAt(image, method) - target);
      if (distance < bestDistance) {{
        bestDistance = distance;
        best = image;
      }}
    }}
    selected = best;
    if (selected >= 0) {{
      const values = data.methods.map((name, index) =>
        name + "=" + rankAt(selected, index).toFixed(4)
      );
      selection.textContent =
        data.images[selected] + " · selected on " + data.methods[method] +
        " · " + values.join(" · ");
    }}
    scheduleDraw();
  }}

  [lineInput, opacityInput, colorInput, topInput].forEach(element => {{
    element.addEventListener("input", scheduleDraw);
    element.addEventListener("change", scheduleDraw);
  }});
  document.getElementById("reset").addEventListener("click", () => {{
    lineInput.value = String(data.initialLines);
    opacityInput.value = "0.025";
    colorInput.value = "-1";
    topInput.value = "2";
    selected = -1;
    selection.textContent = "Click near any method row to select the closest visible image.";
    scheduleDraw();
  }});
  canvas.addEventListener("click", selectNearest);
  let resizeTimer = 0;
  window.addEventListener("resize", () => {{
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 120);
  }});
  resize();
}})();
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique CSV list")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True, help="task order in manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--stage",
        choices=("iterative", "two_stage"),
        default="iterative",
        help="audited ProbeBank supervision stage",
    )
    parser.add_argument(
        "--split",
        choices=("all", "gallery", "test", "train_pool"),
        default="all",
    )
    parser.add_argument(
        "--score-key",
        default="joint",
        help="'joint' or one canonical attribute name/slug",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("seed", "mean-score", "mean-rank"),
        default="mean-rank",
    )
    parser.add_argument("--seed", type=int, default=0, help="used when --seed-mode=seed")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"))
    parser.add_argument("--tie-policy", choices=("average", "stable"), default="average")
    parser.add_argument("--backbone", choices=("siglip",), default="siglip")
    parser.add_argument(
        "--include-embedding-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-ours-full",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ours-full-root", type=Path, default=DEFAULT_OURS_FULL_ROOT)
    parser.add_argument("--initial-lines", type=int, default=12000)
    parser.add_argument("--top-per-method", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest.resolve())
    suite = load_json(args.suite.resolve())
    task = find_task(manifest, args.order)
    H.configure(task["dataset"], task["task"])
    adapter = H.ADAPTER
    attributes = list(H.ATTRS)
    target_attribute = resolve_attribute(args.score_key, attributes)
    database_paths = B.database_paths(adapter)
    paths, indices = split_indices(adapter, args.split)
    configured_methods = list(suite["learned_methods"])
    methods = [
        method for method in LEARNED_METHOD_ORDER if method in configured_methods
    ] + [
        method for method in configured_methods if method not in LEARNED_METHOD_ORDER
    ]
    tensors, seeds_by_method, score_audit = load_score_tensor(
        suite, task, methods, attributes, database_paths, args.stage
    )
    selected_seeds = [args.seed] if args.seed_mode == "seed" else list(args.seeds)
    ranks_by_method = []
    method_names = []
    embedding_audit = None
    if args.include_embedding_baselines:
        fixed_scores, embedding_audit = embedding_baseline_scores(
            adapter,
            attributes,
            target_attribute,
            indices,
            args.backbone,
        )
        for method in EMBEDDING_BASELINES:
            ranks_by_method.append(normalized_rank(fixed_scores[method], args.tie_policy))
            method_names.append(DISPLAY_NAMES[method])
    for method in methods:
        raw_scores = method_scores_by_seed(
            tensors,
            seeds_by_method,
            method,
            selected_seeds,
            attributes,
            target_attribute,
            indices,
        )
        ranks_by_method.append(
            aggregate_seed_ranks(raw_scores, args.seed_mode, args.tie_policy)
        )
        method_names.append(DISPLAY_NAMES.get(method, method))
    if args.include_ours_full:
        ours_full_scores = reconstruct_ours_full_scores(
            tensors,
            seeds_by_method,
            methods,
            attributes,
            target_attribute,
            selected_seeds,
            indices,
            task,
            args.ours_full_root.resolve(),
            expected_stage=(args.stage if args.stage != "iterative" else None),
        )
        ranks_by_method.append(
            aggregate_seed_ranks(ours_full_scores, args.seed_mode, args.tie_policy)
        )
        method_names.append(DISPLAY_NAMES[OURS_FULL_ID])
    rank_matrix = np.ascontiguousarray(np.column_stack(ranks_by_method), dtype=np.float32)
    order = priority_order(rank_matrix, args.top_per_method, args.sample_seed)
    seed_description = (
        f"seed {args.seed}"
        if args.seed_mode == "seed"
        else f"{args.seed_mode} over seeds {selected_seeds}"
    )
    metadata = {
        "schema_version": 1,
        "task_order": int(args.order),
        "task": task_key(task),
        "dataset": task["dataset"],
        "task_name": task["task"],
        "split": "gallery" if args.split == "all" else args.split,
        "score_key": "joint" if target_attribute is None else target_attribute,
        "joint_rule": "product" if target_attribute is None else None,
        "seed_mode": args.seed_mode,
        "selected_seeds": selected_seeds,
        "seed_description": seed_description,
        "tie_policy": args.tie_policy,
        "rank_definition": (
            "mean_of_seedwise_score_percentiles; 1=highest, 0=lowest"
            if args.seed_mode == "mean-rank"
            else "ascending_percentile_of_aggregated_score; 1=highest, 0=lowest"
        ),
        "image_count": len(paths),
        "method_count": len(method_names),
        "methods": method_names,
        "records_hash": B.stable_hash(database_paths),
        "manifest": portable_path(args.manifest),
        "suite": portable_path(args.suite),
        "includes_embedding_baselines": bool(args.include_embedding_baselines),
        "embedding_baselines_seed_independent": bool(args.include_embedding_baselines),
        "includes_ours_full": bool(args.include_ours_full),
    }
    build_html(
        args.output.resolve(),
        rank_matrix,
        method_names,
        paths,
        order,
        metadata,
        args.initial_lines,
    )
    audit_path = (
        args.audit_output.resolve()
        if args.audit_output
        else args.output.resolve().with_suffix(".audit.json")
    )
    audit = {
        **metadata,
        "html": portable_path(args.output),
        "rank_matrix_shape": list(rank_matrix.shape),
        "rank_matrix_bytes": int(rank_matrix.nbytes),
        "priority_order_length": int(len(order)),
        "embedding_baseline_source": embedding_audit,
        "score_sources": score_audit,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "html": str(args.output.resolve()),
                "audit": str(audit_path),
                "images": len(paths),
                "methods": method_names,
                "rank_matrix_shape": list(rank_matrix.shape),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
