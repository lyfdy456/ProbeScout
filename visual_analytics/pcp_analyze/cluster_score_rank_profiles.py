"""Cluster per-image method-rank profiles and build a manual review report.

The input is a standalone HTML produced by ``build_score_rank_pcp.py``.  This
script removes Ours-Full by default, clusters the remaining 13 method ranks in
two ways, and writes assignments, summaries, representative-image sheets, and
an interactive local review page.

Schemes
-------
absolute
    K-means after z-standardizing each of the 13 [0, 1] rank-percentile
    columns.  This gives each method equal scale while preserving whether an
    image is ranked high or low relative to that method's distribution.
shape
    K-means after subtracting each image's mean rank from all of its method
    coordinates.  This removes overall level and retains relative method
    preference.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
LINEAR_ROOT = ROOT / "probe_learning"
SCRIPT_ROOT = LINEAR_ROOT / "scripts"
for path in (LINEAR_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_retrieval_harness as H  # noqa: E402


PCP_PATTERN = re.compile(
    r'<script id="pcp-data" type="application/json">(.*?)</script>', re.DOTALL
)
SAMPLE_GROUPS = (
    ("typical", "典型样本"),
    ("boundary", "边界样本"),
    ("disagreement", "高方法分歧"),
    ("level_stratified", "整体排名分层样本"),
    ("random", "固定随机样本"),
    ("joint_positive", "Joint GT 正例"),
    ("joint_positive_all", "全部 Joint GT 正例"),
)


def read_pcp(path: Path, exclude_methods: set[str]) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    match = PCP_PATTERN.search(source)
    if match is None:
        raise ValueError(f"PCP payload not found in {path}")
    payload = json.loads(match.group(1))
    rows = int(payload["rows"])
    columns = int(payload["columns"])
    methods_all = list(payload["methods"])
    ranks_all = np.frombuffer(
        base64.b64decode(payload["rankFloat32"]), dtype="<f4"
    ).reshape(rows, columns)
    keep = [index for index, name in enumerate(methods_all) if name not in exclude_methods]
    methods = [methods_all[index] for index in keep]
    ranks = np.asarray(ranks_all[:, keep], dtype=np.float64)
    images = list(payload["images"])
    if ranks.shape != (rows, len(methods)) or len(images) != rows:
        raise ValueError("PCP matrix/image dimensions are inconsistent")
    if not np.isfinite(ranks).all():
        raise ValueError("PCP rank matrix contains non-finite values")
    if np.min(ranks) < -1e-6 or np.max(ranks) > 1.0 + 1e-6:
        raise ValueError("PCP rank matrix is outside [0,1]")
    return {
        "payload": payload,
        "methods": methods,
        "ranks": ranks,
        "images": images,
        "kept_indices": keep,
    }


def gt_vectors(metadata: dict[str, Any], images: list[str]) -> tuple[Any, dict[str, np.ndarray]]:
    H.configure(metadata["dataset"], metadata["task_name"])
    adapter = H.ADAPTER
    vectors = {
        attribute: np.asarray(
            [int(adapter.gt_by_attr[attribute].get(path, 0)) for path in images],
            dtype=np.uint8,
        )
        for attribute in H.ATTRS
    }
    vectors["joint"] = np.prod(
        np.vstack([vectors[attribute] for attribute in H.ATTRS]), axis=0
    ).astype(np.uint8)
    return adapter, vectors


def candidate_k_metrics(
    features: np.ndarray,
    sample_indices: np.ndarray,
    k_values: range,
) -> list[dict[str, Any]]:
    rows = []
    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=4096,
            n_init=10,
            max_iter=300,
            reassignment_ratio=0.01,
        ).fit(features)
        labels = model.labels_
        counts = np.bincount(labels, minlength=k)
        rows.append(
            {
                "k": int(k),
                "silhouette_sample": float(
                    silhouette_score(features[sample_indices], labels[sample_indices])
                ),
                "davies_bouldin": float(davies_bouldin_score(features, labels)),
                "calinski_harabasz": float(
                    calinski_harabasz_score(features, labels)
                ),
                "smallest_cluster": int(np.min(counts)),
                "largest_cluster": int(np.max(counts)),
            }
        )
    return rows


def stability_ari(features: np.ndarray, k: int) -> dict[str, float]:
    labelings = [
        KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,
            max_iter=500,
            algorithm="lloyd",
        ).fit_predict(features)
        for seed in range(5)
    ]
    values = [
        adjusted_rand_score(labelings[left], labelings[right])
        for left, right in itertools.combinations(range(len(labelings)), 2)
    ]
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def canonical_order(
    scheme: str,
    labels: np.ndarray,
    ranks: np.ndarray,
    features: np.ndarray,
    k: int,
) -> tuple[list[int], list[tuple[str, str]]]:
    raw_centers = np.vstack([ranks[labels == index].mean(axis=0) for index in range(k)])
    feature_centers = np.vstack(
        [features[labels == index].mean(axis=0) for index in range(k)]
    )
    fixed_mean = raw_centers[:, :5].mean(axis=1)
    learned_mean = raw_centers[:, 5:].mean(axis=1)
    delta = learned_mean - fixed_mean
    if scheme == "absolute":
        if k != 4:
            return list(np.argsort(raw_centers.mean(axis=1))), [
                (f"Absolute cluster {index}", f"绝对排名簇 {index}") for index in range(k)
            ]
        overall = raw_centers.mean(axis=1)
        low = int(np.argmin(overall))
        high = int(np.argmax(overall))
        middle = [index for index in range(k) if index not in {low, high}]
        embedding = min(middle, key=lambda index: delta[index])
        learned = max(middle, key=lambda index: delta[index])
        return [low, embedding, learned, high], [
            ("Consensus low", "一致低排名"),
            ("Embedding-favored", "嵌入基线偏好"),
            ("Learned-favored", "学习方法偏好"),
            ("Consensus high", "一致高排名"),
        ]
    if k != 3:
        order = list(np.argsort(feature_centers[:, 5:].mean(axis=1) - feature_centers[:, :5].mean(axis=1)))
        return order, [
            (f"Shape cluster {index}", f"排名形状簇 {index}") for index in range(k)
        ]
    shape_delta = feature_centers[:, 5:].mean(axis=1) - feature_centers[:, :5].mean(axis=1)
    embedding = int(np.argmin(shape_delta))
    learned = int(np.argmax(shape_delta))
    balanced = next(index for index in range(k) if index not in {embedding, learned})
    return [embedding, balanced, learned], [
        ("Embedding-favored shape", "嵌入基线偏好型"),
        ("Balanced shape", "方法均衡型"),
        ("Learned-favored shape", "学习方法偏好型"),
    ]


def fit_scheme(
    scheme: str,
    ranks: np.ndarray,
    features: np.ndarray,
    k: int,
    gt: dict[str, np.ndarray],
    sample_indices: np.ndarray,
) -> dict[str, Any]:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=30,
        max_iter=500,
        algorithm="lloyd",
    ).fit(features)
    raw_labels = model.labels_
    order, names = canonical_order(scheme, raw_labels, ranks, features, k)
    raw_to_canonical = {raw: canonical for canonical, raw in enumerate(order)}
    labels = np.asarray([raw_to_canonical[int(value)] for value in raw_labels], dtype=np.uint8)
    feature_centers = np.vstack([model.cluster_centers_[raw] for raw in order])
    raw_centers = np.vstack([ranks[labels == index].mean(axis=0) for index in range(k)])
    distances_all = model.transform(features)
    assigned_distance = distances_all[np.arange(len(features)), raw_labels]
    sorted_distances = np.partition(distances_all, kth=1, axis=1)[:, :2]
    margin = sorted_distances[:, 1] - sorted_distances[:, 0]
    counts = np.bincount(labels, minlength=k)
    overall_positive_rate = float(np.mean(gt["joint"]))
    clusters = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_ranks = ranks[mask]
        center = raw_centers[cluster_id]
        fixed_mean = float(np.mean(center[:5]))
        learned_mean = float(np.mean(center[5:]))
        joint_count = int(np.sum(gt["joint"][mask]))
        joint_rate = float(np.mean(gt["joint"][mask])) if np.any(mask) else 0.0
        top_indices = np.argsort(-center)[:4]
        bottom_indices = np.argsort(center)[:4]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": names[cluster_id][0],
                "label_zh": names[cluster_id][1],
                "size": int(counts[cluster_id]),
                "fraction": float(counts[cluster_id] / len(ranks)),
                "mean_rank": float(np.mean(center)),
                "fixed_mean": fixed_mean,
                "learned_mean": learned_mean,
                "learned_minus_fixed": learned_mean - fixed_mean,
                "mean_within_image_rank_std": float(
                    np.mean(np.std(cluster_ranks, axis=1))
                ),
                "joint_positive_count": joint_count,
                "joint_positive_rate": joint_rate,
                "joint_positive_enrichment": (
                    joint_rate / overall_positive_rate if overall_positive_rate > 0 else None
                ),
                "attribute_positive_count": {
                    key: int(np.sum(values[mask]))
                    for key, values in gt.items()
                    if key != "joint"
                },
                "centroid_raw": center.tolist(),
                "centroid_feature": feature_centers[cluster_id].tolist(),
                "top_methods": [int(index) for index in top_indices],
                "bottom_methods": [int(index) for index in bottom_indices],
            }
        )
    return {
        "scheme": scheme,
        "k": k,
        "labels": labels,
        "assigned_distance": assigned_distance,
        "margin": margin,
        "raw_centers": raw_centers,
        "feature_centers": feature_centers,
        "clusters": clusters,
        "metrics": {
            "silhouette_sample": float(
                silhouette_score(features[sample_indices], labels[sample_indices])
            ),
            "davies_bouldin": float(davies_bouldin_score(features, labels)),
            "calinski_harabasz": float(
                calinski_harabasz_score(features, labels)
            ),
            "inertia": float(model.inertia_),
            "stability_ari": stability_ari(features, k),
        },
    }


def representatives(
    result: dict[str, Any],
    ranks: np.ndarray,
    gt: dict[str, np.ndarray],
    per_group: int,
) -> list[dict[str, Any]]:
    rows = []
    labels = result["labels"]
    distance = result["assigned_distance"]
    margin = result["margin"]
    disagreement = np.std(ranks, axis=1)
    mean_rank = np.mean(ranks, axis=1)
    for cluster in result["clusters"]:
        cluster_id = cluster["cluster_id"]
        members = np.flatnonzero(labels == cluster_id)
        rng = np.random.default_rng(
            20260803 + cluster_id + (0 if result["scheme"] == "absolute" else 100)
        )
        sorted_by_level = members[np.argsort(mean_rank[members], kind="stable")]
        stratified_parts = np.array_split(sorted_by_level, 4)
        stratified = []
        per_part = max(1, per_group // 4)
        for part in stratified_parts:
            if len(part):
                chosen = rng.choice(part, min(per_part, len(part)), replace=False)
                stratified.extend(int(value) for value in chosen)
        selections = {
            "typical": members[np.argsort(distance[members], kind="stable")[:per_group]],
            "boundary": members[np.argsort(margin[members], kind="stable")[:per_group]],
            "disagreement": members[
                np.argsort(-disagreement[members], kind="stable")[:per_group]
            ],
            "level_stratified": np.asarray(stratified[:per_group], dtype=np.int64),
            "random": rng.choice(
                members, min(per_group, len(members)), replace=False
            ),
        }
        positives = members[gt["joint"][members] == 1]
        selections["joint_positive"] = positives[
            np.argsort(distance[positives], kind="stable")[:per_group]
        ]
        selections["joint_positive_all"] = positives[
            np.argsort(distance[positives], kind="stable")
        ]
        for group, indices in selections.items():
            for rank, index in enumerate(indices, start=1):
                rows.append(
                    {
                        "scheme": result["scheme"],
                        "cluster_id": cluster_id,
                        "cluster_label": cluster["label"],
                        "cluster_label_zh": cluster["label_zh"],
                        "sample_group": group,
                        "sample_rank": rank,
                        "image_index": int(index),
                        "distance": float(distance[index]),
                        "margin": float(margin[index]),
                        "mean_rank": float(np.mean(ranks[index])),
                        "rank_std": float(disagreement[index]),
                        "fixed_mean": float(np.mean(ranks[index, :5])),
                        "learned_mean": float(np.mean(ranks[index, 5:])),
                        "joint_gt": int(gt["joint"][index]),
                        "attribute_gt": {
                            key: int(values[index])
                            for key, values in gt.items()
                            if key != "joint"
                        },
                    }
                )
    return rows


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def resolve_image_sources(image_root: Path, image_ids: list[str]) -> list[Path]:
    """Resolve dataset image IDs, falling back to unique nested basenames.

    AwA2 records store bare filenames while the files themselves live below
    class directories.  Keep the ordinary relative-path fast path, and only
    scan recursively when at least one direct source is missing.  Ambiguous
    basenames are rejected instead of silently choosing the wrong image.
    """

    root = image_root.resolve()
    resolved: list[Path | None] = []
    unresolved_indices: list[int] = []
    for index, image_id in enumerate(image_ids):
        relative = Path(str(image_id).replace("\\", "/"))
        direct = (root / relative).resolve()
        try:
            direct.relative_to(root)
        except ValueError:
            direct = root / relative.name
        if direct.is_file():
            resolved.append(direct)
        else:
            resolved.append(None)
            unresolved_indices.append(index)

    if unresolved_indices:
        unique: dict[str, Path] = {}
        ambiguous: set[str] = set()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            basename = path.name
            if basename in ambiguous:
                continue
            if basename in unique:
                del unique[basename]
                ambiguous.add(basename)
            else:
                unique[basename] = path.resolve()

        missing: list[str] = []
        for index in unresolved_indices:
            basename = Path(image_ids[index]).name
            if basename in ambiguous:
                raise ValueError(
                    f"Image source basename {basename!r} is ambiguous under {root}"
                )
            source = unique.get(basename)
            if source is None:
                missing.append(image_ids[index])
            else:
                resolved[index] = source
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images are missing; first={missing[0]}"
            )

    return [path for path in resolved if path is not None]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(
    output: Path,
    image_sources: list[Path],
    image_ids: list[str],
    sample_rows: list[dict[str, Any]],
) -> None:
    columns = 4
    rows = 3
    cell_width = 260
    cell_height = 215
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "#111827")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for slot, row in enumerate(sample_rows[: columns * rows]):
        x0 = (slot % columns) * cell_width
        y0 = (slot // columns) * cell_height
        image_id = image_ids[row["image_index"]]
        source = image_sources[row["image_index"]]
        try:
            with Image.open(source) as original:
                tile = ImageOps.contain(original.convert("RGB"), (244, 164))
            x = x0 + (cell_width - tile.width) // 2
            y = y0 + 6 + (164 - tile.height) // 2
            canvas.paste(tile, (x, y))
        except Exception:
            draw.rectangle((x0 + 8, y0 + 8, x0 + 252, y0 + 168), fill="#374151")
            draw.text((x0 + 18, y0 + 78), "image unavailable", font=font, fill="#f9fafb")
        label = (
            f"{slot + 1:02d} {Path(image_id).name}\n"
            f"mean={row['mean_rank']:.3f} std={row['rank_std']:.3f} "
            f"joint={row['joint_gt']}"
        )
        draw.multiline_text((x0 + 8, y0 + 174), label, font=font, fill="#f9fafb", spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, optimize=True)


def build_review_html(
    output: Path,
    methods: list[str],
    image_ids: list[str],
    image_sources: list[Path],
    schemes: dict[str, dict[str, Any]],
    representative_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    group_labels = dict(SAMPLE_GROUPS)
    review_rows = []
    for row in representative_rows:
        image_id = image_ids[row["image_index"]]
        image_source = image_sources[row["image_index"]]
        relative_source = os.path.relpath(image_source, output.parent).replace("\\", "/")
        review_rows.append(
            {
                **row,
                "image_id": image_id,
                "image_url": quote(relative_source, safe="/._-"),
                "sample_group_label": group_labels[row["sample_group"]],
            }
        )
    public_schemes = {}
    for key, result in schemes.items():
        public_schemes[key] = {
            "scheme": key,
            "k": result["k"],
            "metrics": result["metrics"],
            "clusters": result["clusters"],
        }
    payload = {
        "methods": methods,
        "metadata": metadata,
        "schemes": public_schemes,
        "samples": review_rows,
        "sampleGroups": [
            {"value": value, "label": label} for value, label in SAMPLE_GROUPS
        ],
    }
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_text = payload_text.replace("</", "<\\/")
    title = html.escape(f"13-method rank clustering - {metadata['task']}")
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#172033; --muted:#64748b; --axis:#94a3b8; --line:#2563eb; --card:#f8fafc; --border:#cbd5e1; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0b1220; --fg:#e5edf7; --muted:#9aa9bd; --axis:#475569; --line:#60a5fa; --card:#111b2d; --border:#334155; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:18px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
h1 {{ margin:0 0 4px; font-size:20px; font-weight:600; }}
h2 {{ margin:18px 0 8px; font-size:16px; }}
.sub {{ color:var(--muted); margin-bottom:14px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin-bottom:12px; }}
.field {{ display:grid; gap:4px; min-width:180px; }}
label {{ color:var(--muted); font-size:12px; }}
select {{ padding:6px 8px; color:var(--fg); background:var(--card); border:1px solid var(--border); border-radius:6px; font:inherit; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; margin:8px 0 14px; }}
.stat {{ padding:10px; background:var(--card); border:1px solid var(--border); border-radius:7px; }}
.stat span {{ display:block; color:var(--muted); font-size:12px; }}
.stat strong {{ display:block; font-size:18px; margin-top:2px; }}
.profile {{ width:100%; overflow-x:auto; }}
svg {{ display:block; min-width:760px; width:100%; height:auto; }}
.axis {{ stroke:var(--axis); stroke-width:1; opacity:.65; }}
.zero {{ stroke:var(--axis); stroke-width:1.2; }}
.profile-line {{ fill:none; stroke:var(--line); stroke-width:2; }}
.profile-dot {{ fill:var(--line); }}
.label {{ fill:var(--fg); font:12px system-ui,sans-serif; }}
.value {{ fill:var(--muted); font:11px ui-monospace,monospace; }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:10px; }}
.tile {{ margin:0; background:var(--card); border:1px solid var(--border); border-radius:7px; overflow:hidden; }}
.tile img {{ width:100%; height:155px; display:block; object-fit:contain; background:#111827; }}
.tile figcaption {{ padding:8px; overflow-wrap:anywhere; }}
.tile .id {{ font-size:12px; }}
.tile .metrics {{ color:var(--muted); font-size:11px; margin-top:4px; }}
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:850px; }}
th,td {{ padding:7px 8px; text-align:right; border-bottom:1px solid var(--border); white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-weight:500; }}
@media (max-width:600px) {{ body {{ padding:12px; }} .field {{ min-width:100%; }} .tile img {{ height:135px; }} }}
</style>
</head>
<body>
<h1>13-method rank clustering</h1>
<div id="subtitle" class="sub"></div>
<div class="controls">
  <div class="field"><label for="scheme">聚类方案</label><select id="scheme"></select></div>
  <div class="field"><label for="cluster">聚类簇</label><select id="cluster"></select></div>
  <div class="field"><label for="sample-group">审阅图片</label><select id="sample-group"></select></div>
</div>
<div class="sub">先看“典型样本”理解簇中心；排名形状的“方法均衡型”请重点切换到“整体排名分层样本”，避免只看到共识高分图片。点击缩略图可打开原图。</div>
<div id="summary" class="summary"></div>
<div class="profile"><svg id="profile" viewBox="0 0 980 510" role="img" aria-label="Cluster centroid method-rank profile"></svg></div>
<h2 id="gallery-title"></h2>
<div id="gallery" class="gallery"></div>
<h2>全部簇概览</h2>
<div class="table-wrap"><table><thead><tr><th>簇</th><th>图片数</th><th>占比</th><th>平均排名</th><th>固定基线均值</th><th>学习方法均值</th><th>学习-固定</th><th>Joint 正例</th><th>正例富集倍数</th></tr></thead><tbody id="overview"></tbody></table></div>
<script id="cluster-data" type="application/json">{payload_text}</script>
<script>
(() => {{
  "use strict";
  const data = JSON.parse(document.getElementById("cluster-data").textContent);
  const schemeSelect = document.getElementById("scheme");
  const clusterSelect = document.getElementById("cluster");
  const groupSelect = document.getElementById("sample-group");
  const schemeLabels = {{absolute:"绝对排名聚类", shape:"排名形状聚类"}};
  Object.keys(data.schemes).forEach(key => {{ const option=document.createElement("option"); option.value=key; option.textContent=schemeLabels[key]; schemeSelect.appendChild(option); }});
  data.sampleGroups.forEach(row => {{ const option=document.createElement("option"); option.value=row.value; option.textContent=row.label; groupSelect.appendChild(option); }});
  document.getElementById("subtitle").textContent = data.metadata.task + " · " + data.metadata.image_count.toLocaleString() + " images · 13 methods · Ours-Full excluded";
  const fmt = value => Number(value).toFixed(3);
  function refreshClusters() {{
    const result=data.schemes[schemeSelect.value];
    clusterSelect.textContent="";
    result.clusters.forEach(cluster => {{ const option=document.createElement("option"); option.value=String(cluster.cluster_id); option.textContent=cluster.cluster_id+" · "+cluster.label_zh; clusterSelect.appendChild(option); }});
    render();
  }}
  function renderSummary(cluster) {{
    const stats=[
      ["图片数", cluster.size.toLocaleString()+" ("+(cluster.fraction*100).toFixed(1)+"%)"],
      ["整体平均排名", fmt(cluster.mean_rank)],
      ["固定 / 学习均值", fmt(cluster.fixed_mean)+" / "+fmt(cluster.learned_mean)],
      ["学习-固定", (cluster.learned_minus_fixed>=0?"+":"")+fmt(cluster.learned_minus_fixed)],
      ["方法内分歧", fmt(cluster.mean_within_image_rank_std)],
      ["Joint GT 正例", cluster.joint_positive_count+" · "+(cluster.joint_positive_rate*100).toFixed(3)+"%"]
    ];
    document.getElementById("summary").innerHTML=stats.map(row=>'<div class="stat"><span>'+row[0]+'</span><strong>'+row[1]+'</strong></div>').join("");
  }}
  function renderProfile(cluster, scheme) {{
    const svg=document.getElementById("profile");
    const values=scheme==="absolute"?cluster.centroid_raw:cluster.centroid_feature;
    const min=scheme==="absolute"?0:-0.4, max=scheme==="absolute"?1:0.4;
    const left=245, right=900, top=22, step=36;
    const x=value=>left+(Math.max(min,Math.min(max,value))-min)/(max-min)*(right-left);
    const points=[]; let parts=[];
    if (scheme==="shape") parts.push('<line class="zero" x1="'+x(0)+'" y1="8" x2="'+x(0)+'" y2="482"/>');
    data.methods.forEach((method,index)=>{{ const y=top+index*step; const px=x(values[index]); points.push(px+","+y); parts.push('<line class="axis" x1="'+left+'" y1="'+y+'" x2="'+right+'" y2="'+y+'"/><text class="label" x="'+(left-12)+'" y="'+(y+4)+'" text-anchor="end">'+method+'</text><text class="value" x="'+(right+12)+'" y="'+(y+4)+'">'+fmt(values[index])+'</text><circle class="profile-dot" cx="'+px+'" cy="'+y+'" r="3.2"/>'); }});
    parts.push('<polyline class="profile-line" points="'+points.join(" ")+'"/>');
    svg.innerHTML=parts.join("");
  }}
  function renderGallery(cluster, scheme) {{
    const group=groupSelect.value;
    const rows=data.samples.filter(row=>row.scheme===scheme&&row.cluster_id===cluster.cluster_id&&row.sample_group===group);
    const groupLabel=data.sampleGroups.find(row=>row.value===group).label;
    document.getElementById("gallery-title").textContent=cluster.label_zh+" · "+groupLabel+" ("+rows.length+")";
    document.getElementById("gallery").innerHTML=rows.map(row=>{{
      const attrGt=Object.entries(row.attribute_gt).map(pair=>pair[0]+" "+pair[1]).join(" · ");
      return '<figure class="tile"><a href="'+row.image_url+'" target="_blank"><img loading="lazy" src="'+row.image_url+'" alt="'+row.image_id+'"></a><figcaption><div class="id">'+row.image_id+'</div><div class="metrics">mean '+fmt(row.mean_rank)+' · std '+fmt(row.rank_std)+' · fixed '+fmt(row.fixed_mean)+' · learned '+fmt(row.learned_mean)+'<br>'+attrGt+' · joint '+row.joint_gt+'</div></figcaption></figure>';
    }}).join("") || '<div class="sub">该簇没有此类样本。</div>';
  }}
  function renderOverview(result) {{
    document.getElementById("overview").innerHTML=result.clusters.map(cluster=>'<tr><td>'+cluster.cluster_id+' · '+cluster.label_zh+'</td><td>'+cluster.size.toLocaleString()+'</td><td>'+(cluster.fraction*100).toFixed(1)+'%</td><td>'+fmt(cluster.mean_rank)+'</td><td>'+fmt(cluster.fixed_mean)+'</td><td>'+fmt(cluster.learned_mean)+'</td><td>'+(cluster.learned_minus_fixed>=0?'+':'')+fmt(cluster.learned_minus_fixed)+'</td><td>'+cluster.joint_positive_count+' ('+(cluster.joint_positive_rate*100).toFixed(3)+'%)</td><td>'+fmt(cluster.joint_positive_enrichment)+'×</td></tr>').join("");
  }}
  function render() {{
    const scheme=schemeSelect.value; const result=data.schemes[scheme]; const cluster=result.clusters[Number(clusterSelect.value)||0];
    renderSummary(cluster); renderProfile(cluster,scheme); renderGallery(cluster,scheme); renderOverview(result);
  }}
  schemeSelect.addEventListener("change",refreshClusters); clusterSelect.addEventListener("change",render); groupSelect.addEventListener("change",render);
  schemeSelect.value="absolute"; groupSelect.value="typical"; refreshClusters();
}})();
</script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcp-html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-method", action="append", default=["Ours-Full"])
    parser.add_argument("--absolute-k", type=int, default=4)
    parser.add_argument("--shape-k", type=int, default=3)
    parser.add_argument("--representatives-per-group", type=int, default=12)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    loaded = read_pcp(args.pcp_html.resolve(), set(args.exclude_method))
    payload = loaded["payload"]
    methods = loaded["methods"]
    ranks = loaded["ranks"]
    images = loaded["images"]
    if len(methods) != 13:
        raise ValueError(f"expected 13 methods after exclusions, got {len(methods)}")
    adapter, gt = gt_vectors(payload["metadata"], images)
    image_root = Path(adapter.raw_images_dir).resolve()
    image_sources = resolve_image_sources(image_root, images)
    if len(image_sources) != len(images):
        raise RuntimeError("Resolved image-source count does not match the PCP rows")

    sample_rng = np.random.default_rng(20260803)
    sample_indices = np.sort(
        sample_rng.choice(len(ranks), min(6000, len(ranks)), replace=False)
    )
    absolute_scaler = StandardScaler().fit(ranks)
    features_by_scheme = {
        "absolute": absolute_scaler.transform(ranks),
        "shape": ranks - ranks.mean(axis=1, keepdims=True),
    }
    requested_k = {"absolute": args.absolute_k, "shape": args.shape_k}
    results = {
        scheme: fit_scheme(
            scheme,
            ranks,
            features,
            requested_k[scheme],
            gt,
            sample_indices,
        )
        for scheme, features in features_by_scheme.items()
    }
    k_search = {
        scheme: candidate_k_metrics(features, sample_indices, range(3, 13))
        for scheme, features in features_by_scheme.items()
    }

    representative_rows = []
    for result in results.values():
        representative_rows.extend(
            representatives(result, ranks, gt, args.representatives_per_group)
        )
    for row in representative_rows:
        row["image_id"] = images[row["image_index"]]
        row["attribute_gt_json"] = json.dumps(
            row["attribute_gt"], ensure_ascii=False, sort_keys=True
        )
        row["image_source"] = portable_path(image_sources[row["image_index"]])

    assignments = []
    rank_std = np.std(ranks, axis=1)
    fixed_mean = np.mean(ranks[:, :5], axis=1)
    learned_mean = np.mean(ranks[:, 5:], axis=1)
    for index, image_id in enumerate(images):
        assignments.append(
            {
                "image_index": index,
                "image_id": image_id,
                "absolute_cluster": int(results["absolute"]["labels"][index]),
                "absolute_label": results["absolute"]["clusters"][
                    int(results["absolute"]["labels"][index])
                ]["label"],
                "absolute_distance": float(
                    results["absolute"]["assigned_distance"][index]
                ),
                "absolute_margin": float(results["absolute"]["margin"][index]),
                "shape_cluster": int(results["shape"]["labels"][index]),
                "shape_label": results["shape"]["clusters"][
                    int(results["shape"]["labels"][index])
                ]["label"],
                "shape_distance": float(results["shape"]["assigned_distance"][index]),
                "shape_margin": float(results["shape"]["margin"][index]),
                "mean_rank": float(np.mean(ranks[index])),
                "rank_std": float(rank_std[index]),
                "fixed_mean": float(fixed_mean[index]),
                "learned_mean": float(learned_mean[index]),
                "learned_minus_fixed": float(learned_mean[index] - fixed_mean[index]),
                **{f"gt_{key}": int(values[index]) for key, values in gt.items()},
            }
        )

    cluster_rows = []
    centroid_rows = []
    for scheme, result in results.items():
        for cluster in result["clusters"]:
            cluster_rows.append(
                {
                    "scheme": scheme,
                    **{
                        key: value
                        for key, value in cluster.items()
                        if key
                        not in {
                            "centroid_raw",
                            "centroid_feature",
                            "top_methods",
                            "bottom_methods",
                            "attribute_positive_count",
                        }
                    },
                    "attribute_positive_count_json": json.dumps(
                        cluster["attribute_positive_count"], ensure_ascii=False
                    ),
                    "top_methods": " | ".join(
                        methods[index] for index in cluster["top_methods"]
                    ),
                    "bottom_methods": " | ".join(
                        methods[index] for index in cluster["bottom_methods"]
                    ),
                }
            )
            for method_index, method in enumerate(methods):
                centroid_rows.append(
                    {
                        "scheme": scheme,
                        "cluster_id": cluster["cluster_id"],
                        "cluster_label": cluster["label"],
                        "cluster_label_zh": cluster["label_zh"],
                        "method_index": method_index,
                        "method": method,
                        "centroid_raw_rank": cluster["centroid_raw"][method_index],
                        "centroid_feature": cluster["centroid_feature"][method_index],
                    }
                )

    assignment_fields = list(assignments[0])
    write_csv(output / "image_assignments.csv", assignments, assignment_fields)
    write_csv(output / "cluster_summary.csv", cluster_rows, list(cluster_rows[0]))
    write_csv(output / "cluster_centroids.csv", centroid_rows, list(centroid_rows[0]))
    write_csv(
        output / "representative_images.csv",
        representative_rows,
        [
            "scheme",
            "cluster_id",
            "cluster_label",
            "cluster_label_zh",
            "sample_group",
            "sample_rank",
            "image_index",
            "image_id",
            "image_source",
            "distance",
            "margin",
            "mean_rank",
            "rank_std",
            "fixed_mean",
            "learned_mean",
            "joint_gt",
            "attribute_gt_json",
        ],
    )

    cross = np.zeros((args.absolute_k, args.shape_k), dtype=np.int64)
    for left, right in zip(
        results["absolute"]["labels"], results["shape"]["labels"], strict=True
    ):
        cross[int(left), int(right)] += 1
    cross_rows = [
        {
            "absolute_cluster": left,
            "absolute_label": results["absolute"]["clusters"][left]["label"],
            **{f"shape_{right}": int(cross[left, right]) for right in range(args.shape_k)},
        }
        for left in range(args.absolute_k)
    ]
    write_csv(output / "cross_scheme_contingency.csv", cross_rows, list(cross_rows[0]))

    correlation = np.corrcoef(ranks, rowvar=False)
    correlation_rows = [
        {"method": methods[row], **{methods[col]: correlation[row, col] for col in range(len(methods))}}
        for row in range(len(methods))
    ]
    write_csv(output / "method_rank_correlations.csv", correlation_rows, list(correlation_rows[0]))

    pca = {
        scheme: PCA(n_components=min(8, len(methods)), random_state=42)
        .fit(features)
        .explained_variance_ratio_.tolist()
        for scheme, features in features_by_scheme.items()
    }
    audit = {
        "schema_version": 1,
        "source_html": portable_path(args.pcp_html),
        "task": payload["metadata"]["task"],
        "dataset": payload["metadata"]["dataset"],
        "image_count": len(images),
        "method_count": len(methods),
        "methods": methods,
        "excluded_methods": list(args.exclude_method),
        "rank_source": payload["metadata"]["rank_definition"],
        "schemes": {
            scheme: {
                "k": result["k"],
                "metrics": result["metrics"],
                "clusters": result["clusters"],
                "k_search": k_search[scheme],
                "pca_explained_variance_ratio": pca[scheme],
                "selection_note": (
                    "Columns are z-standardized so every method receives equal scale. "
                    "K=4 retains the four actionable absolute patterns: consensus-low, "
                    "embedding-favored, learned-favored, and consensus-high."
                    if scheme == "absolute"
                    else "K=3 is the best sampled-silhouette solution and separates "
                    "embedding-favored, balanced, and learned-favored rank shapes."
                ),
            }
            for scheme, result in results.items()
        },
        "ground_truth_used_for_clustering": False,
        "ground_truth_used_only_for_post_hoc_description": True,
        "preprocessing": {
            "absolute": {
                "definition": "per-method z-standardization of the 13 raw rank columns",
                "column_mean": absolute_scaler.mean_.tolist(),
                "column_scale": absolute_scaler.scale_.tolist(),
            },
            "shape": {
                "definition": "subtract each image's mean across the 13 raw rank columns",
                "additional_column_scaling": False,
                "row_l2_normalization": False,
            },
        },
        "image_root": portable_path(image_root),
        "missing_images": 0,
    }
    (output / "cluster_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "cluster_data.npz",
        ranks=np.asarray(ranks, dtype=np.float32),
        image_ids=np.asarray(images),
        methods=np.asarray(methods),
        absolute_labels=results["absolute"]["labels"],
        shape_labels=results["shape"]["labels"],
    )

    sheets = output / "sheets"
    for scheme, result in results.items():
        for cluster in result["clusters"]:
            for sample_group in ("typical", "joint_positive"):
                selected = [
                    row
                    for row in representative_rows
                    if row["scheme"] == scheme
                    and row["cluster_id"] == cluster["cluster_id"]
                    and row["sample_group"] == sample_group
                ]
                if selected:
                    contact_sheet(
                        sheets
                        / f"{scheme}_cluster_{cluster['cluster_id']}_{sample_group}.jpg",
                        image_sources,
                        images,
                        selected,
                    )

    review_metadata = {
        "task": payload["metadata"]["task"],
        "image_count": len(images),
    }
    build_review_html(
        output / "cluster_review.html",
        methods,
        images,
        image_sources,
        results,
        representative_rows,
        review_metadata,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "review": str(output / "cluster_review.html"),
                "images": len(images),
                "methods": methods,
                "absolute_clusters": [
                    {key: cluster[key] for key in ("cluster_id", "label", "size")}
                    for cluster in results["absolute"]["clusters"]
                ],
                "shape_clusters": [
                    {key: cluster[key] for key in ("cluster_id", "label", "size")}
                    for cluster in results["shape"]["clusters"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
