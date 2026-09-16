"""Dataset adapter for the unified frozen-test harness (run_retrieval_harness.py).



Goal: make ONE harness serve multiple datasets (Stanford Cars / SUN / CUB / ...).

The harness's training / split / metrics / top-50 logic is dataset-agnostic; only

a handful of things are dataset-specific, and they are all funneled through this

adapter:



  - feature/record artifacts (clip_embedding.npy, clip_patch_tokens.npy, records.csv)

  - raw VQA-answer jsonl + how its `image` field maps to the canonical key

  - ground-truth: a binary label per (attribute, canonical key)

  - the query prototype (embedding indices)

  - the raw-images dir used to hardlink top-50 results



Canonical key: ALWAYS `records.relative_path`. Every dataset's VQA `image` field is

normalized to this via `vqa_to_key`, and GT is keyed by it too.



The harness supports 2 to 5 attributes, each with a binary GT. For Cars these come

from the task registry's cars_mapping; for SUN from sun_mapping; for CUB from

CUB_ATTR_MAPPING.

"""



from __future__ import annotations



import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


import numpy as np

import pandas as pd



from src.utils import paths as P





@dataclass

class DatasetAdapter:
    dataset: str

    task: str

    attrs: list[str]                      # 2..5 attribute names (match VQA answer keys)

    key_slugs: list[str]                  # filesystem-safe short labels, aligned to attrs

    joint_label: str

    emb_path: Path

    patch_path: Path

    records: pd.DataFrame

    raw_jsonl_path: Path

    vqa_to_key: Callable[[str], str]      # VQA `image` field -> canonical key (relative_path)

    query_idx: list[int]                  # embedding indices for the query prototype

    raw_images_dir: Path                  # hardlink source root; file = raw_images_dir / relative_path
    gt_by_attr: dict[str, dict[str, int]] # attr -> {relative_path: 0/1}
    task_root: Path                       # new per-task dir: tasks/<group>/<task>/
    p2i: dict[str, int] = field(default_factory=dict)
    database_root: Path | None = None
    query_embedding_paths: dict[str, Path] = field(default_factory=dict)
    _resolved_image_paths: dict[str, Path] = field(default_factory=dict, repr=False)
    # Supervision strategies actually run for this dataset. Datasets that only have a
    # two_stage labeling pass (e.g. AwA2) restrict this to ["two_stage"].

    strategies: list[str] = field(default_factory=lambda: ["random", "two_stage"])



    def __post_init__(self):

        if not (2 <= len(self.attrs) <= 5):

            raise ValueError(f"harness needs 2 to 5 attributes, got {self.attrs}")

        if not self.p2i:

            self.p2i = dict(zip(self.records["relative_path"], self.records["embedding_index"]))



    # ---- derived new-layout paths ----

    @property

    def split_dir(self) -> Path:

        return self.task_root / "split"



    @property

    def stage_dirs(self) -> dict[str, Path]:

        """Supervision strategy -> output stage dir. random=onestage, two_stage=twostage."""

        return {"random": self.task_root / "onestage",

                "two_stage": self.task_root / "twostage"}



    @property

    def database_dir(self) -> Path:
        from src.utils import paths as _P
        if self.database_root is not None:
            return self.database_root
        return _P.database_dir(self.dataset)

    def resolve_image_path(self, image_id: str) -> Path:
        """Resolve one canonical records.relative_path to its raw image file."""
        key = self.vqa_to_key(str(image_id))
        if key in self._resolved_image_paths:
            return self._resolved_image_paths[key]
        if key not in self.p2i:
            raise KeyError(f"image ID is not present in logical database records: {image_id}")

        relative = Path(str(key).replace("\\", "/"))
        candidates = [self.raw_images_dir / relative]
        rows = self.records.loc[self.records["relative_path"].astype(str) == key]
        if len(rows) == 1 and "class_name" in rows.columns:
            class_name = str(rows.iloc[0].get("class_name", "")).strip()
            if class_name and class_name.lower() != "nan":
                candidates.append(self.raw_images_dir / class_name / relative.name)
        for path in candidates:
            if path.is_file():
                resolved = path.resolve()
                self._resolved_image_paths[key] = resolved
                return resolved

        matches = list(self.raw_images_dir.rglob(relative.name))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"cannot uniquely resolve {image_id!r} under {self.raw_images_dir}: "
                f"matches={len(matches)}"
            )
        resolved = matches[0].resolve()
        self._resolved_image_paths[key] = resolved
        return resolved

    def vqa_image_manifest(self, image_ids: list[str]) -> list[dict]:
        """Build explicit logical-ID/path rows consumed by the VQA labeler."""
        rows = []
        seen = set()
        for value in image_ids:
            image_id = self.vqa_to_key(str(value))
            if image_id in seen:
                continue
            seen.add(image_id)
            rows.append({
                "image_id": image_id,
                "relative_path": image_id,
                "image_path": str(self.resolve_image_path(image_id)),
                "embedding_index": int(self.p2i[image_id]),
            })
        return rows




def _slug(text: str, fallback: str) -> str:

    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    s = re.sub(r"_+", "_", s)

    return s or fallback





def _dedupe_slugs(slugs: list[str]) -> list[str]:
    seen, out = {}, []

    for s in slugs:

        if s in seen:

            seen[s] += 1

            out.append(f"{s}{seen[s]}")

        else:

            seen[s] = 0

            out.append(s)

    return out


def _load_query_ids(task_root: Path) -> list[str]:
    """Load the canonical query IDs for a task.

    ``query_ids.json`` is the sole source of truth.  Adapters may normalize an
    ID to their dataset-specific ``records.relative_path`` representation, but
    they must not infer query membership from copied files in ``query_pics/``.
    """
    query_ids = task_root / "query_ids.json"
    if not query_ids.is_file():
        raise FileNotFoundError(f"task is missing canonical query IDs: {query_ids}")
    payload = json.loads(query_ids.read_text(encoding="utf-8"))
    values = payload.get("image_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError(f"query_ids.json must contain a non-empty image_ids list: {query_ids}")
    result = [str(value).replace("\\", "/").strip() for value in values]
    if any(not value for value in result):
        raise ValueError(f"query_ids.json contains an empty image ID: {query_ids}")
    if len(result) != len(set(result)):
        raise ValueError(f"query_ids.json contains duplicate image IDs: {query_ids}")
    return result


def _query_indices_from_ids(
    task_root: Path,
    p2i: dict[str, int],
    normalize: Callable[[str], str] = lambda value: value,
) -> tuple[list[str], list[int]]:
    """Resolve every canonical query ID and reject partial/empty mappings."""
    query_ids = _load_query_ids(task_root)
    resolved = [normalize(value) for value in query_ids]
    missing = [source for source, target in zip(query_ids, resolved) if target not in p2i]
    if missing:
        raise KeyError(f"query IDs are missing from records.csv for {task_root}: {missing}")
    return resolved, [int(p2i[value]) for value in resolved]


def _task_config_path(task_root: Path) -> Path:
    """Resolve the canonical task config, retaining a legacy fallback."""
    canonical = task_root / "task.json"
    return canonical if canonical.is_file() else task_root / "task_meta.json"


# --------------------------------------------------------------------------- Cars
def _build_cars(task: str, joint_label: str | None) -> DatasetAdapter:
    cfg = P.get_cars_task(task)
    mapping = cfg["cars_mapping"]

    attrs = list(cfg["attributes"])

    if not (2 <= len(attrs) <= 5) or any(a not in mapping for a in attrs):

        # fall back to the GT-mapped attributes if the list carries extras

        attrs = list(mapping.keys())

    if not (2 <= len(attrs) <= 5):

        raise ValueError(f"Cars task {task} needs 2 to 5 GT-mapped attrs, got {attrs}")

    gt_types = [mapping[a]["gt_type"] for a in attrs]

    inverses = [mapping[a].get("inverse", False) for a in attrs]

    slugs = _dedupe_slugs([gt.replace("is_", "") for gt in gt_types])



    from src.evaluation.cars_eval import load_cars_ground_truth

    gt_lookup = load_cars_ground_truth()  # {basename: {is_bmw,is_sedan,is_convertible}}

    records = pd.read_csv(P.CARS_PROCESSED_DIR / "records.csv")



    gt_by_attr: dict[str, dict[str, int]] = {}
    for a, gt_type, inv in zip(attrs, gt_types, inverses):
        spec = mapping[a]
        d = {}
        for key in records["relative_path"]:
            ge = gt_lookup.get(key)
            if ge is None:
                continue
            if gt_type == "class_name_contains":
                needle = str(spec["contains"])
                v = int(needle.lower() in str(ge["class_name"]).lower())
            elif gt_type == "class_name_startswith":
                needle = str(spec["prefix"])
                v = int(str(ge["class_name"]).lower().startswith(needle.lower()))
            else:
                v = int(ge.get(gt_type, 0))
            d[key] = (1 - v) if inv else v
        gt_by_attr[a] = d


    p2i = dict(zip(records["relative_path"], records["embedding_index"]))

    def car_query_to_rel(name: str) -> str:
        if name in p2i:
            return name
        base = Path(name).name
        if base in p2i:
            return base
        m = re.match(r"query_\d+_(.+)$", base)
        if m and m.group(1) in p2i:
            return m.group(1)
        return name

    troot = P.task_root("cars", task)
    query_files, query_idx = _query_indices_from_ids(troot, p2i, car_query_to_rel)
    return DatasetAdapter(

        dataset="cars", task=task, attrs=attrs, key_slugs=slugs,

        joint_label=joint_label or cfg.get("joint_label") or " & ".join(slugs),

        emb_path=P.CARS_PROCESSED_DIR / "clip_embedding.npy",

        patch_path=P.CARS_PROCESSED_DIR / P.PATCH_TOKEN_FILE,

        records=records,

        raw_jsonl_path=troot / "qa" / cfg["raw_jsonl"],

        vqa_to_key=lambda s: s,

        query_idx=query_idx,

        raw_images_dir=P.CARS_ROOT / "images",

        gt_by_attr=gt_by_attr,

        task_root=troot,

        p2i=p2i,

    )





# --------------------------------------------------------------------------- SUN

def _build_sun(task: str, joint_label: str | None) -> DatasetAdapter:
    cfg = P.get_sun_task(task)
    sun_map = cfg["sun_mapping"]                 # attr -> {sun_idx, sun_name?, inverse}

    attrs = list(sun_map.keys())                 # GT-mapped attrs (expect 2)

    if not (2 <= len(attrs) <= 5):

        raise ValueError(f"SUN task {task} sun_mapping must have 2 to 5 attrs, got {attrs}")

    slugs = _dedupe_slugs([_slug(sun_map[a].get("sun_name", a), f"attr{i}")

                           for i, a in enumerate(attrs)])



    records = pd.read_csv(P.SUN_PROCESSED_DIR / "records.csv")
    from src.evaluation.sun_eval import load_sun_ground_truth, sun_gt_binary

    img_to_idx, labels_cv = load_sun_ground_truth()



    gt_by_attr: dict[str, dict[str, int]] = {}
    for a in attrs:
        spec = sun_map[a]
        inv = spec.get("inverse", False)
        d = {}
        if "sun_category" in spec:
            prefix = str(spec["sun_category"]).strip().replace("\\", "/").strip("/")
            for key in records["relative_path"]:
                g = str(key).replace("\\", "/").startswith(prefix + "/")
                d[key] = int((not g) if inv else g)
        elif "sun_any" in spec:
            indices = [int(s["sun_idx"]) for s in spec["sun_any"]]
            for key in records["relative_path"]:
                vals = [sun_gt_binary(img_to_idx, labels_cv, key, sidx, False)
                        for sidx in indices]
                vals = [v for v in vals if v is not None]
                if vals:
                    g = any(vals)
                    d[key] = int((not g) if inv else g)
        else:
            sidx = spec["sun_idx"]
            for key in records["relative_path"]:
                g = sun_gt_binary(img_to_idx, labels_cv, key, sidx, inv)
                if g is not None:
                    d[key] = int(g)
        gt_by_attr[a] = d


    # VQA `image` may be 'a__abbey__sun_x.jpg' or bare 'sun_x.jpg'; both -> 'a/abbey/sun_x.jpg'
    bare_to_rel = {Path(rp).name: rp for rp in records["relative_path"]}
    p2i = dict(zip(records["relative_path"], records["embedding_index"]))


    def vqa_to_key(s: str) -> str:
        if "__" in s:
            return s.replace("__", "/")
        return bare_to_rel.get(s, s)

    def sun_query_to_rel(name: str) -> str:
        if name in p2i:
            return name
        base = re.sub(r"^query_\d+_", "", Path(name).name, flags=re.IGNORECASE)
        return bare_to_rel.get(base, vqa_to_key(name))

    troot = P.task_root("sun", task)
    query_files, query_idx = _query_indices_from_ids(troot, p2i, sun_query_to_rel)
    return DatasetAdapter(
        dataset="sun", task=task, attrs=attrs, key_slugs=slugs,
        joint_label=joint_label or cfg.get("joint_label") or " & ".join(slugs),
        emb_path=P.SUN_PROCESSED_DIR / "clip_embedding.npy",
        patch_path=P.SUN_PROCESSED_DIR / P.PATCH_TOKEN_FILE,
        records=records,
        raw_jsonl_path=troot / "qa" / cfg["raw_jsonl"],
        vqa_to_key=vqa_to_key,
        query_idx=query_idx,
        raw_images_dir=P.SUN_ROOT / "images",
        gt_by_attr=gt_by_attr,
        task_root=troot,
        p2i=p2i,
    )




# --------------------------------------------------------------------------- CUB

def _build_cub(task: str, joint_label: str | None) -> DatasetAdapter:
    """CUB adapter. NOTE: requires precomputed CUB artifacts under PROCESSED_DIR
    (clip_embedding.npy, clip_patch_tokens.npy, records.csv, attrs.npy) and a task
    with canonical query IDs + a raw VQA jsonl. If artifacts are missing this raises.


    `task` registry for CUB is not yet defined; pass the 2 attributes + query via a

    task_meta.json sidecar similar to Cars, or extend this builder.

    """

    from src.evaluation.cub_eval import CUB_ATTR_MAPPING, _cub_label_for_attr  # noqa


    troot = P.task_root("cub", task)
    meta_path = _task_config_path(troot)
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"CUB task needs {meta_path} with {{attributes(2), raw_jsonl}}.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    proc = P.PROCESSED_DIR
    emb_path = proc / "clip_embedding.npy"
    if not emb_path.is_file():
        raise FileNotFoundError(
            f"CUB embeddings not found at {emb_path}. Precompute CUB features first "
            "(scripts/precompute_backbone_patch_tokens.py + the CLIP embed step) before using "
            "the CUB adapter.")
    CUB_ATTR_MAPPING.update({k: list(v) for k, v in meta.get("cub_attr_mapping", {}).items()})
    attrs = list(meta["attributes"])
    if not (2 <= len(attrs) <= 5) or any(a not in CUB_ATTR_MAPPING for a in attrs):

        raise ValueError(f"CUB task needs 2 to 5 attrs present in CUB_ATTR_MAPPING, got {attrs}")

    slugs = _dedupe_slugs([_slug(a, f"attr{i}") for i, a in enumerate(attrs)])



    records = pd.read_csv(proc / "records.csv")

    attrs_np = np.load(proc / "attrs.npy")

    p2i = dict(zip(records["relative_path"], records["embedding_index"]))



    gt_by_attr: dict[str, dict[str, int]] = {}

    for a in attrs:

        d = {}

        for key, idx in p2i.items():

            d[key] = _cub_label_for_attr(attrs_np, int(idx), a)

        gt_by_attr[a] = d



    from src.data.cub import image_field_to_relative_path as cub_to_rel

    basename_to_rel = {Path(rp).name: rp for rp in p2i}

    def cub_task_to_rel(name: str) -> str:
        if "/" in name:
            return name if name.startswith("images/") else f"images/{name}"
        if "__" in name:
            return cub_to_rel(name)
        return basename_to_rel.get(name, name)

    query_files, query_idx = _query_indices_from_ids(troot, p2i, cub_task_to_rel)
    raw_jsonl = meta.get("raw_jsonl", "")


    return DatasetAdapter(

        dataset="cub", task=task, attrs=attrs, key_slugs=slugs,

        joint_label=joint_label or meta.get("joint_label") or " & ".join(slugs),

        emb_path=emb_path,

        patch_path=proc / P.PATCH_TOKEN_FILE,

        records=records,

        raw_jsonl_path=troot / "qa" / raw_jsonl,

        vqa_to_key=cub_task_to_rel,
        query_idx=query_idx,

        raw_images_dir=P.CUB_ROOT,        # CUB relative_path already starts with 'images/'

        gt_by_attr=gt_by_attr,

        task_root=troot,

        p2i=p2i,

    )





# --------------------------------------------------------------------------- HICO

def _build_hico(task: str, joint_label: str | None) -> DatasetAdapter:
    """HICO adapter. GT attributes come from anno.mat image-level HOI labels,

    precomputed into records.csv (is_bicycle / is_jumping / is_bike_jump) by

    scripts/precompute_backbone_embeddings.py --dataset hico --backbone siglip.



    Task config via task_meta.json sidecar under the task dir:

      {attributes: [2 names in HICO_ATTR_MAPPING], query_files|query_indices, raw_jsonl}

    If query is omitted, defaults to the first few is_bike_jump positives so the

    two_stage prototype is meaningful.

    """

    import json as _json



    from src.evaluation.hico_eval import (
        HICO_ATTR_MAPPING,
        load_hico_ground_truth,
        load_hico_task_ground_truth,
    )


    proc = P.HICO_PROCESSED_DIR

    emb_path = proc / "clip_embedding.npy"
    records_path = proc / "records.csv"
    if not records_path.is_file():
        raise FileNotFoundError(
            f"HICO records not found at {records_path}. Run "
            "scripts/precompute_backbone_embeddings.py --dataset hico --backbone siglip --records-only first.")


    meta_path = _task_config_path(P.task_root("hico", task))
    meta = _json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    attrs = list(meta.get("attributes", ["bicycle", "jumping"]))
    hoi_mapping = meta.get("hico_hoi_mapping")
    valid_attrs = hoi_mapping if hoi_mapping else HICO_ATTR_MAPPING
    if not (2 <= len(attrs) <= 5) or any(a not in valid_attrs for a in attrs):
        raise ValueError(
            f"HICO task needs 2 to 5 attrs with official GT mappings, got {attrs}")
    slugs = _dedupe_slugs([_slug(a, f"attr{i}") for i, a in enumerate(attrs)])



    records = pd.read_csv(proc / "records.csv")

    gt_lookup = (load_hico_task_ground_truth(hoi_mapping)
                 if hoi_mapping else load_hico_ground_truth())
    p2i = dict(zip(records["relative_path"], records["embedding_index"]))
    bare2rel = {Path(rp).name: rp for rp in records["relative_path"]}

    def hico_query_to_rel(name: str) -> str:
        if name in p2i:
            return name
        base = Path(name).name
        if base in bare2rel:
            return bare2rel[base]
        m = re.match(r"query_\d+_(.+)$", base)
        if m:
            raw = m.group(1)
            return bare2rel.get(raw, raw)
        return name

    gt_by_attr: dict[str, dict[str, int]] = {}
    for a in attrs:
        if hoi_mapping:
            d = {key: int(gt_lookup[key][a]) for key in records["relative_path"]}
        else:
            col = HICO_ATTR_MAPPING[a]["gt_col"]
            inv = HICO_ATTR_MAPPING[a].get("inverse", False)
            d = {}
            for key in records["relative_path"]:
                v = int(gt_lookup[key][col])
                d[key] = (1 - v) if inv else v
        gt_by_attr[a] = d


    # Query membership is defined only by the canonical query_ids.json file.
    troot = P.task_root("hico", task)
    query_files, query_idx = _query_indices_from_ids(troot, p2i, hico_query_to_rel)


    # raw VQA jsonl: explicit name from meta, else auto-detect latest qa/*_results.jsonl
    raw_jsonl = meta.get("raw_jsonl", "")
    qa_dir = troot / "qa"
    if not raw_jsonl:

        cands = sorted(qa_dir.glob("*_results.jsonl"),

                       key=lambda p: p.stat().st_mtime, reverse=True) if qa_dir.is_dir() else []

        raw_jsonl = cands[0].name if cands else ""



    return DatasetAdapter(

        dataset="hico", task=task, attrs=attrs, key_slugs=slugs,

        joint_label=joint_label or meta.get("joint_label") or " & ".join(slugs),

        emb_path=emb_path,

        patch_path=proc / P.PATCH_TOKEN_FILE,

        records=records,

        raw_jsonl_path=qa_dir / raw_jsonl,

        vqa_to_key=lambda s: s if s in p2i else hico_query_to_rel(s),
        query_idx=query_idx,

        raw_images_dir=P.HICO_IMAGES_DIR,

        gt_by_attr=gt_by_attr,

        task_root=troot,

        p2i=p2i,

    )





# --------------------------------------------------------------------------- AwA2

def _build_awa2(task: str, joint_label: str | None) -> DatasetAdapter:
    """AwA2 adapter (e.g. zebra task).

    Preferred layout:
      dataset/raw/AwA2/processed/{records.csv, clip_embedding.npy, ...}  # shared DB
      tasks/task_awa2/<task>/processed/{query_records.csv, query_clip_embedding.npy, ...}

    The adapter exposes a virtual records table that appends this task's query rows
    after the shared database rows, preserving the harness' single-index assumption.
    Legacy task-scoped combined processed/{records.csv, clip_embedding.npy} remains
    supported as a fallback.


    Under the unified flow the harness holds out ~20% of the full-DB GT (minus query)

    as the frozen test set, then draws the supervision subset from the VQA-labeled

    images in the remaining pool — matching the real workflow of labeling a small

    budget and propagating to the rest. AwA2 only has a two_stage labeling pass.



    Attribute GT is class-level (AwA2 85-predicate matrix) resolved via class_id + the

    full-dataset predicates.json. The 2 attributes default to ``stripes`` ∩ ``hooves``

    (≈ zebra) and may be overridden by a task_meta.json sidecar {"attributes": [a, b]}.

    """

    import json as _json

    troot = P.task_root("awa2", task)
    proc = troot / "processed"
    shared_proc = P.AWA2_PROCESSED_DIR
    query_records_path = proc / "query_records.csv"
    query_clip_path = proc / "query_clip_embedding.npy"
    db_records = pd.read_csv(shared_proc / "records.csv")
    db_p2i = dict(zip(db_records["relative_path"], db_records["embedding_index"]))
    query_ids = _load_query_ids(troot)

    def shared_query_key(name: str) -> str | None:
        base = Path(name).name
        if base in db_p2i:
            return base
        if "__" in base:
            canonical = base.split("__", 1)[1]
            if canonical in db_p2i:
                return canonical
        numbered = re.sub(r"^query_\d+_", "", base, flags=re.IGNORECASE)
        if numbered in db_p2i:
            return numbered
        return None

    shared_query_files = [shared_query_key(name) for name in query_ids]
    use_shared_queries = bool(query_ids) and all(shared_query_files)
    if not use_shared_queries:
        missing = [name for name, resolved in zip(query_ids, shared_query_files) if resolved is None]
        raise KeyError(f"AwA2 query IDs are missing from records.csv: {missing}")
    use_shared_layout = query_records_path.is_file() and query_clip_path.is_file()

    if use_shared_queries:
        emb_path = shared_proc / "clip_embedding.npy"
        records = db_records.copy()
        if "is_query" not in records.columns:
            records["is_query"] = 0
        query_embedding_paths = {}
        shared_patch = shared_proc / P.PATCH_TOKEN_FILE
        legacy_patch = P.task_root("awa2", "task_zebra") / "processed" / P.PATCH_TOKEN_FILE
        patch_path = shared_patch if shared_patch.is_file() else legacy_patch
    elif use_shared_layout:
        emb_path = shared_proc / "clip_embedding.npy"
        if "is_query" not in db_records.columns:
            db_records["is_query"] = 0
        qrecords = pd.read_csv(query_records_path)
        db_n = len(db_records)
        qrecords = qrecords.copy()
        qrecords["embedding_index"] = np.arange(db_n, db_n + len(qrecords), dtype=np.int64)
        qrecords["dataset"] = "awa2"
        qrecords["class_id"] = -1
        qrecords["class_name"] = ""
        qrecords["is_query"] = 1
        records = pd.concat([db_records, qrecords[db_records.columns]], ignore_index=True)
        query_embedding_paths = {
            "clip": query_clip_path,
            "dinov2": proc / "query_dinov2_embedding.npy",
            "siglip": proc / "query_siglip_embedding.npy",
        }
        patch_path = proc / P.PATCH_TOKEN_FILE
    else:
        emb_path = proc / "clip_embedding.npy"
        records_path = proc / "records.csv"
        if not emb_path.is_file() or not records_path.is_file():
            raise FileNotFoundError(
                f"AwA2 features not found. Preferred: shared DB {shared_proc/'clip_embedding.npy'} "
                f"plus task query {query_clip_path}. Run scripts/precompute_backbone_embeddings.py (requires prepared records) --task {task}.")
        records = pd.read_csv(records_path)
        query_embedding_paths = {}
        patch_path = proc / P.PATCH_TOKEN_FILE

    preds = _json.loads((P.AWA2_PROCESSED_DIR / "predicates.json").read_text(encoding="utf-8"))
    pred_names = preds["predicate_names"]

    bin_mat = np.array(preds["binary_matrix"], dtype=np.int64)   # 50 classes x 85 predicates

    pidx = {p: i for i, p in enumerate(pred_names)}



    meta_path = _task_config_path(troot)
    meta = _json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    attrs = list(meta.get("attributes", ["stripes", "hooves"]))
    gt_mapping = meta.get("gt_mapping") or {a: {"predicate": a} for a in attrs}
    if not (2 <= len(attrs) <= 5):
        raise ValueError(f"AwA2 task needs 2 to 5 attributes, got {attrs}")
    for a in attrs:
        spec = gt_mapping.get(a)
        if not spec:
            raise ValueError(f"AwA2 task missing gt_mapping for attribute {a!r}")
        if "predicate" in spec and spec["predicate"] not in pidx:
            raise ValueError(
                f"AwA2 predicate {spec['predicate']!r} for {a!r} not in predicates.json")
        if "predicate" not in spec and "class_name" not in spec:
            raise ValueError(
                f"AwA2 gt_mapping for {a!r} needs 'predicate' or 'class_name', got {spec}")
    slugs = _dedupe_slugs([_slug(a, f"attr{i}") for i, a in enumerate(attrs)])


    if "is_query" not in records.columns:
        records["is_query"] = 0
    p2i = dict(zip(records["relative_path"], records["embedding_index"]))



    # class-level GT keyed by relative_path (skip query rows / unknown class)

    gt_by_attr: dict[str, dict[str, int]] = {a: {} for a in attrs}
    for _, row in records.iterrows():
        cid = int(row["class_id"])
        if int(row.get("is_query", 0)) == 1 or cid < 1:
            continue
        rp = row["relative_path"]
        for a in attrs:
            spec = gt_mapping[a]
            if "class_name" in spec:
                gt_by_attr[a][rp] = int(str(row.get("class_name", "")) == spec["class_name"])
            else:
                gt_by_attr[a][rp] = int(bin_mat[cid - 1, pidx[spec["predicate"]]])


    if use_shared_queries:
        query_idx = [int(db_p2i[q]) for q in shared_query_files]
    else:
        query_idx = [int(i) for i in records.loc[records["is_query"] == 1, "embedding_index"].tolist()]


    qa_dir = troot / "qa"

    raw_jsonl = meta.get("raw_jsonl", "")

    if not raw_jsonl:

        cands = sorted(qa_dir.glob("*_results.jsonl"),

                       key=lambda p: p.stat().st_mtime, reverse=True) if qa_dir.is_dir() else []

        raw_jsonl = cands[0].name if cands else ""



    bare2rel = {Path(rp).name: rp for rp in records["relative_path"]}

    return DatasetAdapter(

        dataset="awa2", task=task, attrs=attrs, key_slugs=slugs,
        joint_label=joint_label or meta.get("joint_label") or " & ".join(slugs),
        emb_path=emb_path,
        patch_path=patch_path,
        records=records,
        raw_jsonl_path=qa_dir / raw_jsonl,
        vqa_to_key=lambda s: s if s in p2i else bare2rel.get(Path(s).name, s),
        query_idx=query_idx,

        raw_images_dir=P.database_dir("awa2"),   # flat: database/<basename>

        gt_by_attr=gt_by_attr,
        task_root=troot,
        p2i=p2i,
        query_embedding_paths=query_embedding_paths,
        # Run whichever per-strategy VQA sources are available. Historically this
        # task only had two_stage labels, but qa/random/ can now be generated for

        # a true one-stage baseline.

        strategies=["random", "two_stage"],

    )


# --------------------------------------------------------------------------- CelebA
def _build_celeba(task: str, joint_label: str | None) -> DatasetAdapter:
    """CelebA adapter backed by official binary attributes in list_attr_celeba.txt.

    Task config via task_meta.json:
      {attributes, gt_mapping: attr -> {celeba_attr, inverse}, query_files, raw_jsonl}
    """
    troot = P.task_root("celeba", task)
    proc = P.CELEBA_PROCESSED_DIR
    emb_path = proc / "clip_embedding.npy"
    records_path = proc / "records.csv"
    if not records_path.is_file():
        raise FileNotFoundError(
            f"CelebA records not found at {records_path}. Run "
            "scripts/precompute_backbone_embeddings.py --dataset celeba --backbone siglip --records-only first.")

    meta_path = _task_config_path(troot)
    if not meta_path.is_file():
        raise FileNotFoundError(f"CelebA task needs {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    attrs = list(meta["attributes"])
    gt_mapping = meta["gt_mapping"]
    if not (2 <= len(attrs) <= 5) or any(a not in gt_mapping for a in attrs):
        raise ValueError(f"CelebA task needs 2 to 5 GT-mapped attrs, got {attrs}")
    slugs = _dedupe_slugs([_slug(a, f"attr{i}") for i, a in enumerate(attrs)])

    records = pd.read_csv(records_path)
    p2i = dict(zip(records["relative_path"], records["embedding_index"]))
    gt_by_attr = {a: {} for a in attrs}
    for _, row in records.iterrows():
        rp = row["relative_path"]
        for a in attrs:
            spec = gt_mapping[a]
            v = int(row[spec["celeba_attr"]])
            gt_by_attr[a][rp] = (1 - v) if spec.get("inverse", False) else v

    qfiles, query_idx = _query_indices_from_ids(troot, p2i, lambda value: Path(value).name)

    qa_dir = troot / "qa"
    raw_jsonl = meta.get("raw_jsonl", "")
    if not raw_jsonl:
        cands = sorted(qa_dir.glob("*_results.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True) if qa_dir.is_dir() else []
        raw_jsonl = cands[0].name if cands else ""

    return DatasetAdapter(
        dataset="celeba", task=task, attrs=attrs, key_slugs=slugs,
        joint_label=joint_label or meta.get("joint_label") or " & ".join(slugs),
        emb_path=emb_path,
        patch_path=proc / P.PATCH_TOKEN_FILE,
        records=records,
        raw_jsonl_path=qa_dir / raw_jsonl,
        vqa_to_key=lambda s: Path(s).name,
        query_idx=query_idx,
        raw_images_dir=P.CELEBA_IMAGES_DIR,
        gt_by_attr=gt_by_attr,
        task_root=troot,
        p2i=p2i,
        database_root=P.CELEBA_IMAGES_DIR,
        strategies=["random", "two_stage"],
    )


_BUILDERS = {"cars": _build_cars, "sun": _build_sun, "cub": _build_cub,
             "hico": _build_hico, "awa2": _build_awa2, "celeba": _build_celeba}




def build_adapter(dataset: str, task: str, joint_label: str | None = None) -> DatasetAdapter:

    if dataset not in _BUILDERS:

        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(_BUILDERS)}")

    return _BUILDERS[dataset](task, joint_label)

