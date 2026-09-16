"""Top-50 retrieval export for the unified frozen-test harness (dataset-agnostic).

For each method, given continuous scores over the full candidate library, export
per-ranking hardlinked image folders for visualization:
  - <slug>           : ranked by P(attribute)
  - <slug0>_<...>    : ranked by selected attribute-combination scores

Leakage-free: trained methods rank on the HELD-OUT pool (the VQA supervision subset
is excluded), and each stage gets one shared top_50/vqa_gt/<ranking>/ folder holding
that ranking's VQA-labeled supervision positives. Zero-shot embedding baselines rank
on the full library with no per-method vqa_gt copy.

Images are hardlinked from `raw_images_dir` (supplied by the dataset adapter); the
destination filename flattens any '/' in the relative path (SUN/CUB are nested).
Each ranking folder gets a manifest.json with per-image score and GT hit flag.

GT model: `gt_by_attr` = {attr: {relative_path: 0/1}}; ranking_spec gt entries are
("single", attr) or ("joint", attr0, attr1, ...) for any selected combination.
"""

import json
import os
import stat
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _robust_rmtree(path: Path, retries: int = 5, delay: float = 0.4):
    """Delete a directory tree, tolerating Windows read-only flags / transient locks.

    Windows raises PermissionError (WinError 5) when a file is read-only or briefly
    held by another process (indexer/AV/preview). Clear the read-only bit and retry
    a few times before giving up.
    """
    def _on_error(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for attempt in range(retries):
        try:
            # Python 3.12 renamed onerror -> onexc; support both.
            try:
                shutil.rmtree(path, onexc=_on_error)
            except TypeError:
                shutil.rmtree(path, onerror=lambda f, p, e: _on_error(f, p, e))
            return
        except (PermissionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
# Default raw-images root (Stanford Cars). Adapters pass their own raw_images_dir.
RAW_IMAGES = ROOT / "dataset" / "raw" / "stanford_cars" / "images"


def _hardlink(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def _flat(rel: str) -> str:
    """Flatten a (possibly nested) relative path into a single filename."""
    return rel.replace("/", "__").replace("\\", "__")


def _gt_hit(spec_gt, rel: str, gt_by_attr: dict):
    """GT hit flag for one ranking key given its gt spec and per-attr GT dicts."""
    kind = spec_gt[0]
    if kind == "joint":
        vals = [gt_by_attr.get(a, {}).get(rel) for a in spec_gt[1:]]
        if any(v is None for v in vals):
            return None
        return int(all(int(v) == 1 for v in vals))
    g = gt_by_attr.get(spec_gt[1], {}).get(rel)
    return None if g is None else int(g)


def _export_shared_vqa_gt(task_dir: Path, vqa_gt_by_key: dict, gt_by_attr: dict,
                          raw_images_dir: Path, ranking_spec: dict):
    """Shared vqa_gt folders for a stage.

    The VQA supervision positives are stage-level data, not method-level outputs.
    Export them once under top_50/vqa_gt/<ranking_subdir>/ to avoid many duplicate
    hardlink folders and repeated manifest writes.
    """
    out_root = task_dir / "top_50" / "vqa_gt"
    if out_root.exists():
        _robust_rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for key, info in ranking_spec.items():
        gt_paths = list(vqa_gt_by_key.get(key, []))
        if not gt_paths:
            continue
        dest = out_root / info["subdir"]
        dest.mkdir(parents=True, exist_ok=True)
        manifest = []
        for rank, rel in enumerate(gt_paths, 1):
            raw = raw_images_dir / rel
            dst = dest / f"{rank:02d}_{_flat(rel)}"
            if raw.is_file():
                _hardlink(raw, dst)
            manifest.append({
                "rank": rank,
                "image": rel,
                "gt_hit": _gt_hit(info["gt"], rel, gt_by_attr),
            })
        (dest / "manifest.json").write_text(
            json.dumps({"method": "shared_vqa_gt", "ranking": key,
                        "kind": "vqa_supervision_positives",
                        "count": len(manifest), "items": manifest},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"    [vqa_gt/{info['subdir']}] {len(manifest)} supervised positives")


def export_method_top50(
    task_dir: Path,
    method: str,
    scores: dict,
    cand_paths: list,
    gt_by_attr: dict,
    raw_images_dir: Path,
    ranking_spec: dict,
    topn: int = 50,
    exclude_set: set | None = None,
    vqa_gt_by_key: dict | None = None,
):
    """Export per-key rankings for one method.

    scores: {key: np.ndarray} aligned to cand_paths order; keys match ranking_spec.
    exclude_set: VQA supervision relative paths to drop from ranking (held-out).
    vqa_gt_by_key: {key: [relative_path,...]} supervision positives per ranking.
    """
    import numpy as np

    exclude_set = exclude_set or set()
    vqa_gt_by_key = vqa_gt_by_key or {}
    out_root = task_dir / "top_50" / method

    if exclude_set:
        keep_pos = np.array([i for i, rp in enumerate(cand_paths) if rp not in exclude_set],
                            dtype=np.int64)
    else:
        keep_pos = np.arange(len(cand_paths), dtype=np.int64)

    for key, info in ranking_spec.items():
        sub = info["subdir"]
        arr = scores[key]
        order = keep_pos[np.argsort(-arr[keep_pos])[:topn]]
        dest = out_root / sub
        if dest.exists():
            _robust_rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        manifest = []
        for rank, ci in enumerate(order, 1):
            rel = cand_paths[ci]
            raw = raw_images_dir / rel
            dst = dest / f"{rank:02d}_{_flat(rel)}"
            hit = _gt_hit(info["gt"], rel, gt_by_attr)
            if raw.is_file():
                _hardlink(raw, dst)
            manifest.append({"rank": rank, "image": rel, "score": float(arr[ci]), "gt_hit": hit})
        hits = sum(1 for m in manifest if m["gt_hit"] == 1)
        (dest / "manifest.json").write_text(
            json.dumps({"method": method, "ranking": key,
                        "ranking_pool": "held_out" if exclude_set else "full",
                        "precision_at_50": hits / max(len(manifest), 1),
                        "items": manifest}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"    [{method}/{sub}] top-{topn}  P@{topn}={hits/max(len(manifest),1):.3f}")


def export_all_top50(
    task_dir: Path,
    scores_by_method: dict,
    cand_paths: list,
    gt_by_attr: dict,
    raw_images_dir: Path,
    ranking_spec: dict,
    topn: int = 50,
    exclude_set: set | None = None,
    vqa_gt_by_key: dict | None = None,
    no_exclude_methods: set | None = None,
):
    """scores_by_method: {method: {key: arr}}; keys match ranking_spec.

    Methods in `no_exclude_methods` (zero-shot baselines) rank on the full library
    with no vqa_gt/; all other (trained) methods rank held-out with per-ranking vqa_gt/.
    """
    no_exclude_methods = no_exclude_methods or set()
    print(f"\n=== Exporting top-{topn} retrieval images to {task_dir / 'top_50'} ===")
    if vqa_gt_by_key:
        _export_shared_vqa_gt(task_dir, vqa_gt_by_key, gt_by_attr, raw_images_dir, ranking_spec)
    for method, scores in scores_by_method.items():
        if method in no_exclude_methods:
            export_method_top50(task_dir, method, scores, cand_paths, gt_by_attr,
                                raw_images_dir, ranking_spec, topn,
                                exclude_set=None, vqa_gt_by_key=None)
        else:
            export_method_top50(task_dir, method, scores, cand_paths, gt_by_attr,
                                raw_images_dir, ranking_spec, topn,
                                exclude_set=exclude_set, vqa_gt_by_key=vqa_gt_by_key)
