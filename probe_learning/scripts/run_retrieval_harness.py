"""Shared feature, supervision, eight-probe training and acquisition utilities.

Use scripts/train_probes.py for the paper fixed-Validation protocol.
The CLI here supplies the iterative VQA acquisition controller.
"""


import sys
import json
import hashlib
import time
import argparse
import subprocess
import re
import shutil
import warnings
from itertools import combinations
from pathlib import Path


import numpy as np

import pandas as pd

import torch

from sklearn.metrics import (

    average_precision_score, f1_score, precision_score, recall_score, accuracy_score,

)

from sklearn.linear_model import LogisticRegression



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



from src.utils.paths import (
    TASKS_DIR, BMW_SEDAN_TASK,
    PATCH_TOKEN_FILES, CLIP_MODEL_NAME,
)
from src.data.cars import prepare_attribute_data
from src.data.dataset_adapter import build_adapter

from src.methods.catalog import METHOD_KINDS as PAPER_METHOD_KINDS, METHOD_LABELS as PAPER_METHOD_LABELS
from src.methods.mlp_probe import train_one_attribute, _seed_everything
from src.methods.pu_probe import cross_validate_probs, train_one_attribute_weighted

from src.methods.attn_probe import (
    encode_attribute_texts_for_backbone, prepare_attribute_patches, train_one_attribute_attn,
)
from src.methods.metric_attention_probes import (
    train_one_attribute_attention_pooling,
    train_one_attribute_cosine_margin_ranking,
    train_one_attribute_triplet_loss,
)
# --- additional PU variants wired into the harness for one-shot comparison ---

from src.methods.dcpu_probe import train_one_attribute_dcpu

from src.methods.nnpu_probe import train_one_attribute_nnpu

from src.methods.pu_ranking_probe import train_one_attribute_pu_ranking
from scripts.extract_sun_probing_labels import extract_labels, extract_label_scores

from scripts.generate_top50_cars import export_all_top50



# --------------------------------------------------------------------------- config

TEST_FRAC = 0.20

SPLIT_SEED = 42

LABEL_SIZE = 1000     # default VQA labeling budget L (images sent to the labeler)

SUP_SIZE = 1000       # default supervision budget S (subset of L actually trained on)

CEILING_COV = 0.98    # VQA ceiling shown only if labels cover >= this frac of the eval pool

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

TOPKS = [10, 50, 100]
U_CAP = 4000          # cap unlabeled pool size for ranking/nnpu/sapu (memory/time)

EPOCHS = 100



# ---- task-dependent globals (set by configure() at runtime via the adapter) ----

ADAPTER = None        # src.data.dataset_adapter.DatasetAdapter

DATASET = "cars"

TASK = BMW_SEDAN_TASK

ATTRS = []            # list of 2..5 attribute strings (match VQA answer keys)

ATTR_KEY = {}         # attr -> short slug ("bmw" ...)

KEYS = []             # slugs aligned to ATTRS

JOINT_LABEL = ""      # display label for the joint concept, e.g. "BMW 轿车"

STAGE_DIRS = {}       # {"random": <task>/onestage, "two_stage": <task>/twostage} (Path)

SPLIT_DIR = None      # <task>/split (Path)

RANKING_SPEC = {}     # consumed by generate_top50_cars.export_all_top50
JOINT_KEY = "joint"
MAX_RANKING_COMBOS = 15


# supervision strategy -> output stage subdir name (random=onestage, two_stage=twostage)

STAGE_NAME = {
    "random": "onestage",
    "two_stage": "twostage",
    "exploit300_coverage200": "exploit300_coverage200",
    # Kept separate from one-shot stages. The iterative runner owns its round
    # manifests and never writes into legacy supervision directories.
    "iterative_vqa_100_50_v1": "iterative_vqa_100_50_v1",
}




def configure(dataset: str, task: str, joint_label: str | None = None, *, adapter=None):
    """Build the dataset adapter and populate task-dependent globals."""
    global ADAPTER, DATASET, TASK, ATTRS, ATTR_KEY, KEYS
    global JOINT_LABEL, STAGE_DIRS, SPLIT_DIR, RANKING_SPEC
    global _CLIP_TEXT_FEATS, _CLIP_PROMPT_ENSEMBLE_FEATS, _CLIP_BINARY_TEXT_FEATS
    global _TEXT_FEATS_BY_BACKBONE, _PROMPT_ENSEMBLE_FEATS_BY_BACKBONE
    ADAPTER = adapter if adapter is not None else build_adapter(dataset, task, joint_label)
    DATASET = dataset

    TASK = task

    ATTRS = list(ADAPTER.attrs)

    KEYS = list(ADAPTER.key_slugs)

    if not (2 <= len(ATTRS) <= 5):

        raise ValueError(f"harness supports 2 to 5 attributes, got {ATTRS}")

    ATTR_KEY = dict(zip(ATTRS, KEYS))

    JOINT_LABEL = ADAPTER.joint_label

    STAGE_DIRS = {**ADAPTER.stage_dirs}
    for strat, stage in STAGE_NAME.items():
        STAGE_DIRS.setdefault(strat, ADAPTER.task_root / stage)
    SPLIT_DIR = ADAPTER.split_dir
    RANKING_SPEC = build_ranking_spec()
    _CLIP_TEXT_FEATS = None
    _CLIP_PROMPT_ENSEMBLE_FEATS = None
    _CLIP_BINARY_TEXT_FEATS = None
    _TEXT_FEATS_BY_BACKBONE.clear()
    _PROMPT_ENSEMBLE_FEATS_BY_BACKBONE.clear()




# Backward-compatible alias (Cars-only entrypoint).

def configure_task(task: str, joint_label: str | None = None):
    configure("cars", task, joint_label)


def backbone_slug(backbone: str) -> str:
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    return backbone.replace("+", "_")


def apply_backbone_output_layout(backbone: str, isolate_output: bool = False) -> Path:
    """Switch report/result dirs to task/embedding_runs/<backbone>/ when needed."""
    global STAGE_DIRS, SPLIT_DIR
    slug = backbone_slug(backbone)
    if backbone == "clip" and not isolate_output:
        return ADAPTER.task_root
    root = ADAPTER.task_root / BACKBONE_RUN_ROOT / slug
    STAGE_DIRS = {strat: root / stage for strat, stage in STAGE_NAME.items()}
    SPLIT_DIR = root / "split"
    return root


def _embedding_base_dir(adapter) -> Path:
    """Directory that holds this adapter's pooled embeddings."""
    return Path(adapter.emb_path).parent


def load_backbone_embeddings(adapter, backbone: str) -> tuple[np.ndarray, dict]:
    """Load one pooled embedding or a normalized concatenation of several backbones."""
    parts = BACKBONES[backbone]
    base_dir = _embedding_base_dir(adapter)
    arrays = []
    sources = {}
    n_rows = None
    for part in parts:
        path = base_dir / BACKBONE_FILE[part]
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {part} embedding for {adapter.dataset}/{adapter.task}: {path}\n"
                f"Create it with scripts/precompute_backbone_embeddings.py, or run with --backbone clip."
            )
        arr = np.load(path)
        qpath = getattr(adapter, "query_embedding_paths", {}).get(part)
        if qpath is not None:
            if not qpath.is_file():
                raise FileNotFoundError(f"missing query {part} embedding: {qpath}")
            qarr = np.load(qpath)
            arr = np.vstack([arr.astype(np.float32, copy=False), qarr.astype(np.float32, copy=False)])
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(
                f"backbone row mismatch in {base_dir}: expected {n_rows}, "
                f"got {arr.shape[0]} for {path.name}"
            )
        arrays.append(l2norm(arr.astype(np.float32, copy=False)))
        sources[part] = str(path) if qpath is None else {"database": str(path), "query": str(qpath)}
    expected_rows = len(adapter.records)
    if n_rows != expected_rows:
        raise ValueError(
            f"embedding/records length mismatch for {adapter.dataset}/{adapter.task}: "
            f"embeddings={n_rows}, records={expected_rows}"
        )
    emb = arrays[0] if len(arrays) == 1 else l2norm(np.concatenate(arrays, axis=1))
    meta = {
        "backbone": backbone,
        "backbone_parts": list(parts),
        "embedding_sources": sources,
        "embedding_dim": int(emb.shape[1]),
    }
    return emb.astype(np.float32, copy=False), meta


class PatchTokenRows:
    """Read DB patch tokens plus optional task-query rows without copying the DB."""

    def __init__(self, database: np.ndarray, query: np.ndarray | None = None):
        self.database = database
        self.query = query
        self.db_n = int(database.shape[0])
        extra = 0 if query is None else int(query.shape[0])
        self.shape = (self.db_n + extra, *database.shape[1:])
        self.dtype = database.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        if self.query is None:
            return self.database[index]
        if isinstance(index, slice):
            return self[np.arange(*index.indices(len(self)), dtype=np.int64)]
        arr = np.asarray(index)
        if arr.ndim == 0:
            idx = int(arr)
            if idx < self.db_n:
                return self.database[idx]
            return self.query[idx - self.db_n]
        flat = arr.astype(np.int64, copy=False).ravel()
        out = np.empty((len(flat), *self.shape[1:]), dtype=self.dtype)
        db_mask = flat < self.db_n
        if db_mask.any():
            out[db_mask] = self.database[flat[db_mask]]
        if (~db_mask).any():
            out[~db_mask] = self.query[flat[~db_mask] - self.db_n]
        return out.reshape((*arr.shape, *self.shape[1:]))


def load_backbone_patches(adapter, backbone: str):
    """Load patch tokens for attention methods in the selected backbone."""
    if backbone not in PATCH_TOKEN_FILES:
        raise FileNotFoundError(f"no patch-token file registered for backbone {backbone!r}")
    if backbone == "clip":
        path = Path(adapter.patch_path)
    else:
        path = _embedding_base_dir(adapter) / PATCH_TOKEN_FILES[backbone]
    if not path.is_file():
        raise FileNotFoundError(
            f"patch tokens required for attention methods but missing: {path}. "
            "Create them with scripts/precompute_backbone_patch_tokens.py."
        )
    database = np.load(path, mmap_mode="r")
    qpatch = None
    qpaths = getattr(adapter, "query_embedding_paths", {})
    qemb = qpaths.get(backbone)
    if qemb is not None:
        qpath = Path(qemb).with_name("query_" + PATCH_TOKEN_FILES[backbone])
        if not qpath.is_file():
            raise FileNotFoundError(
                f"missing task query patch tokens: {qpath}. "
                "Create them with scripts/precompute_backbone_patch_tokens.py --task <task>."
            )
        qpatch = np.load(qpath, mmap_mode="r")
    patches = PatchTokenRows(database, qpatch)
    expected_rows = len(adapter.records)
    if len(patches) > expected_rows and qpatch is None:
        # Some legacy AwA2 task patch files append task-query rows after the
        # shared database rows. Shared-query adapters only expose the DB rows.
        patches = PatchTokenRows(database[:expected_rows], None)
    if len(patches) != expected_rows:
        raise ValueError(
            f"patch/records length mismatch for {adapter.dataset}/{adapter.task}: "
            f"patches={len(patches)}, records={expected_rows}, source={path}"
        )
    sources = str(path) if qpatch is None else {"database": str(path), "query": str(qpath)}
    return patches, {
        "patch_backbone": backbone,
        "patch_sources": sources,
        "patch_shape": [int(x) for x in patches.shape],
        "patch_dtype": str(patches.dtype),
    }


def active_embedding_methods(backbone: str) -> list[str]:
    """Text-fusion zero-shot baselines need a matching image/text embedding space."""
    if backbone in ("clip", "siglip"):
        return list(EMB_METHODS)
    return [m for m in EMB_METHODS if m not in TEXT_EMB_METHODS]


def validate_methods_for_backbone(backbone: str, methods_to_run: list[str]) -> None:
    """Reject methods whose model assumptions are not a clean backbone swap."""
    incompatible = []
    attn_requested = [m for m in methods_to_run if TRAINED_METHODS.get(m) in ("attn", "attn_ens")]
    if backbone != "clip":
        incompatible += [m for m in methods_to_run if m in CLIP_ADAPTER_METHODS]
    if attn_requested and backbone not in ("clip", "siglip"):
        incompatible += attn_requested
    if incompatible:
        attn_note = "Attention methods are currently wired for clip and siglip patch-token backbones only."
        raise SystemExit(
            f"--backbone {backbone} is not compatible with all requested methods. "
            f"Remove incompatible methods: {sorted(set(incompatible))}. "
            f"{attn_note} CLIP/Tip adapters need CLIP text weights."
        )


# method name -> feature kind. "mlp"/"attn" return a single model; "mlp_ens"/"attn_ens"
# return a list of models (multi-boundary ensemble), scored by averaging.
TRAINED_METHODS = dict(PAPER_METHOD_KINDS)
EMB_METHODS = [
    "image_prototype",
    "img_text_fusion",
    "text_prompt_ensemble",
    "query_maxsim",
    "zscore_img_text_fusion",
]
BACKBONES = {
    "clip": ("clip",),
    "dinov2": ("dinov2",),
    "siglip": ("siglip",),
    "clip_dinov2": ("clip", "dinov2"),
    "siglip_dinov2": ("siglip", "dinov2"),
}
BACKBONE_FILE = {
    "clip": "clip_embedding.npy",
    "dinov2": "dinov2_embedding.npy",
    "siglip": "siglip_embedding.npy",
}
BACKBONE_TEXT_MODEL = {
    "clip": CLIP_MODEL_NAME,
    "siglip": "google/siglip-base-patch16-224",
}
BACKBONE_RUN_ROOT = "embedding_runs"
TEXT_EMB_METHODS = {
    "img_text_fusion",
    "text_prompt_ensemble",
    "zscore_img_text_fusion",
}
CLIP_ADAPTER_METHODS = set()
VQA_CEILING = "vqa_direct"   # directly threshold full-VQA 0/1 labels (intersection); label-quality ceiling
METHOD_LABELS = {**PAPER_METHOD_LABELS, "query_maxsim": "Query MaxSim", "text_prompt_ensemble": "Text Prompt Ensemble", "vqa_direct": "VQA direct"}
ALL_METHODS = [VQA_CEILING] + EMB_METHODS + list(TRAINED_METHODS.keys())

# Main-table method policy, updated 2026-07-06. Future methods are documented
# separately so the runnable harness does not accept placeholder method names.
MAIN_TABLE_METHOD_GROUPS = {"probes": tuple(TRAINED_METHODS)}
FUTURE_MAIN_TABLE_METHODS = []
MAIN_TABLE_METHODS = list(TRAINED_METHODS)
COMPACT_REPORT_METHODS = [
    (method, METHOD_LABELS.get(method, method))
    for method in MAIN_TABLE_METHODS
]


CALIBRATED_METHOD_BASE = {}

ZSCORE_JOINT_METHOD_BASE = {}


def base_method_for(method: str) -> str:
    return CALIBRATED_METHOD_BASE.get(method, ZSCORE_JOINT_METHOD_BASE.get(method, method))


def l2norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)

    return (x / np.where(n > 1e-12, n, 1.0)).astype(np.float32)





def device():

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")





def joint_slug() -> str:
    return "_".join(KEYS)




def combo_slug(attrs: tuple[str, ...] | list[str]) -> str:
    return "_".join(ATTR_KEY[a] for a in attrs)


def ranking_key_for_combo(attrs: tuple[str, ...] | list[str]) -> str:
    attrs = tuple(attrs)
    if len(attrs) == len(ATTRS) and tuple(attrs) == tuple(ATTRS):
        return JOINT_KEY
    return combo_slug(attrs)


def selected_attr_combos() -> list[tuple[str, ...]]:
    """Non-empty attribute combinations to report/export.

    Ordered by combination size, then original attribute order. If all non-empty
    combinations exceed MAX_RANKING_COMBOS, keep the earliest small combinations
    and force the full-attribute joint as the final combination.
    """
    combos = []
    for r in range(1, len(ATTRS) + 1):
        combos.extend(tuple(c) for c in combinations(ATTRS, r))
    if len(combos) <= MAX_RANKING_COMBOS:
        return combos
    full = tuple(ATTRS)
    kept = combos[:MAX_RANKING_COMBOS - 1]
    if full not in kept:
        kept.append(full)
    return kept[:MAX_RANKING_COMBOS]


def build_ranking_spec() -> dict:
    spec = {}
    for combo in selected_attr_combos():
        key = ranking_key_for_combo(combo)
        subdir = combo_slug(combo)
        if len(combo) == 1:
            spec[key] = {"subdir": subdir, "gt": ("single", combo[0]), "attrs": combo}
        else:
            spec[key] = {"subdir": subdir, "gt": ("joint", *combo), "attrs": combo}
    return spec


def combo_gt_from(gt_by_attr_vectors: dict[str, np.ndarray], attrs: tuple[str, ...] | list[str]) -> np.ndarray:
    vals = [np.asarray(gt_by_attr_vectors[a], dtype=int) for a in attrs]
    if not vals:
        return np.array([], dtype=int)
    return np.logical_and.reduce(vals).astype(int)


def joint_gt_from(gt_by_attr_vectors: dict[str, np.ndarray]) -> np.ndarray:
    return combo_gt_from(gt_by_attr_vectors, ATTRS)


def combo_score_from(scores_by_attr: dict[str, np.ndarray], attrs: tuple[str, ...] | list[str]) -> np.ndarray:
    vals = [np.asarray(scores_by_attr[a], dtype=np.float32) for a in attrs]
    if not vals:
        return np.array([], dtype=np.float32)
    out = np.ones_like(vals[0], dtype=np.float32)
    for v in vals:
        out = out * v
    return out


def joint_score_from(scores_by_attr: dict[str, np.ndarray]) -> np.ndarray:
    return combo_score_from(scores_by_attr, ATTRS)


def combo_pred_at_threshold(scores_by_attr: dict[str, np.ndarray],
                            attrs: tuple[str, ...] | list[str],
                            threshold: float = 0.5) -> np.ndarray:
    vals = [(np.asarray(scores_by_attr[a]) >= threshold) for a in attrs]
    if not vals:
        return np.array([], dtype=int)
    return np.logical_and.reduce(vals).astype(int)


def joint_pred_at_threshold(scores_by_attr: dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    return combo_pred_at_threshold(scores_by_attr, ATTRS, threshold)


def scores_by_ranking_key(scores_by_attr: dict[str, np.ndarray], joint_scores: np.ndarray | None = None) -> dict:
    out = {}
    for key, info in RANKING_SPEC.items():
        attrs = tuple(info.get("attrs") or info["gt"][1:])
        if len(attrs) == 1:
            out[key] = np.asarray(scores_by_attr[attrs[0]])
        elif key == JOINT_KEY and joint_scores is not None:
            out[key] = np.asarray(joint_scores)
        else:
            out[key] = combo_score_from(scores_by_attr, attrs)
    return out




# --------------------------------------------------------------------------- VQA labeling

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_CONTENT_FILTER_REGISTRY = (
    EXPERIMENT_ROOT / "configs" / "experiments" / "manual_vqa_content_filter_labels.json"
)
VQA_DIR = EXPERIMENT_ROOT / "GPT网页端VQA自动打标方案"

_VQA_DIR_CANDIDATES = [
    EXPERIMENT_ROOT / "GPT网页端VQA自动打标方案",
    EXPERIMENT_ROOT / "attribute_annotation",
]
VQA_DIR = next((p for p in _VQA_DIR_CANDIDATES if (p / "vqa_label.py").is_file()), VQA_DIR)


def _format_attr_list(attrs):
    return "{" + ", ".join(attrs) + "}"


def _validate_vqa_attributes_file(attr_txt):
    """Reject ambiguous multi-line attribute lists before invoking VQA."""
    text = Path(attr_txt).read_text(encoding="utf-8-sig").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and "," not in text:
        raise SystemExit(
            f"[auto-vqa] invalid attributes format in {attr_txt}: multiple attributes "
            "must be comma-separated, for example {attribute 1, attribute 2}."
        )
    return Path(attr_txt)


def _clean_attr_name(s):
    s = re.sub(r"\s+", " ", str(s)).strip()

    return s.strip("`'\"，,;； ")





def parse_attributes_from_answer_md(answer_path):

    """Extract the final VQA labeling attributes from qa/answer.md.



    Preference order:

      1. The last non-empty manual final list, e.g. 人为提取的attr：{a, b, c}.

      2. The first JSON object with attributes[*].name from Stage2.

    """

    answer_path = Path(answer_path)

    if not answer_path.is_file():

        return []

    text = answer_path.read_text(encoding="utf-8-sig")



    json_names = []

    decoder = json.JSONDecoder()

    for m in re.finditer(r"\{", text):

        try:

            obj, _ = decoder.raw_decode(text[m.start():])

        except json.JSONDecodeError:

            continue

        attrs = obj.get("attributes") if isinstance(obj, dict) else None

        if isinstance(attrs, list):

            names = []

            for a in attrs:

                if isinstance(a, dict):

                    name = _clean_attr_name(a.get("name", ""))

                else:

                    name = _clean_attr_name(a)

                if name:

                    names.append(name)

            if names:

                json_names = names

                break



    manual_patterns = [

        r"(?:人为|人工|手动)\s*提取\s*的?\s*attr(?:ibutes)?\s*[:：]\s*`?\{([^{}]*)\}`?",

        r"final\s+attributes\s*[:：]\s*`?\{([^{}]*)\}`?",

    ]

    manual = []

    for pat in manual_patterns:

        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):

            attrs = [_clean_attr_name(x) for x in re.split(r"[,，;；\n]+", m.group(1))]

            attrs = [a for a in attrs if a]

            if attrs:

                manual.append(attrs)

    if manual:

        chosen = manual[-1]

        if json_names:

            json_set = {a.lower() for a in json_names}

            overlap = sum(1 for a in chosen if a.lower() in json_set)

            if overlap == len(chosen):

                return chosen

            print(f"  [auto-vqa] ignored inconsistent manual attributes in {answer_path}: "

                  f"{chosen}; using Stage2 JSON attributes instead.")

            return json_names

        return chosen



    return json_names





def _load_direct_label_rows(path):

    """Load pre-parsed label rows from a JSON list or JSONL file."""

    p = Path(path)

    if p.suffix.lower() == ".json":

        obj = json.loads(p.read_text(encoding="utf-8"))

        if isinstance(obj, list):

            return obj

        raise ValueError(f"direct-label JSON must be a list: {p}")

    rows = []

    with open(p, encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(json.loads(line))

    return rows





def _direct_labels(entry, attrs):

    out = {}

    for attr in attrs:

        val = entry.get(attr)

        out[attr] = None if val is None else int(val)

    return out





def _direct_label_scores(labels):
    return {k: (None if v is None else float(v)) for k, v in labels.items()}


def is_complete_binary_labels(labels, attrs) -> bool:
    """Return whether ``labels`` is exactly one strict binary task label row.

    This deliberately rejects booleans (``bool`` is an ``int`` subclass),
    numeric strings, floats, missing attributes, and unexpected attributes.
    Callers must not silently coerce malformed provider/cache output into a
    negative label.
    """

    attrs = tuple(attrs)
    if not isinstance(labels, dict) or set(labels) != set(attrs):
        return False
    for attr in attrs:
        value = labels[attr]
        if isinstance(value, (bool, np.bool_)):
            return False
        if not isinstance(value, (int, np.integer)) or int(value) not in (0, 1):
            return False
    return True


def _strict_binary_labels(labels, attrs):
    if not is_complete_binary_labels(labels, attrs):
        return None
    return {attr: int(labels[attr]) for attr in attrs}


def _strict_labels_from_raw_answer(raw_answer, attrs):
    """Parse native provider JSON without the legacy integer coercions."""

    if not isinstance(raw_answer, str):
        return None, None, "answer_is_not_a_string", []
    start, end = raw_answer.find("{"), raw_answer.rfind("}")
    if start < 0 or end < start:
        return None, None, "answer_has_no_json_object", []
    try:
        payload = json.loads(raw_answer[start:end + 1])
    except json.JSONDecodeError as error:
        return None, None, f"answer_json_error:{error}", []
    attributes = payload.get("attributes") if isinstance(payload, dict) else None
    if not isinstance(attributes, dict):
        return None, None, "attributes_object_missing_or_invalid", []
    missing = sorted(set(attrs).difference(attributes))
    extras = sorted(set(attributes).difference(attrs))
    if missing:
        return None, None, f"current_attributes_missing:{missing}", extras
    native_labels = {}
    scores = {}
    for attr in attrs:
        info = attributes.get(attr)
        if not isinstance(info, dict):
            return None, None, f"attribute_record_invalid:{attr}", extras
        native_labels[attr] = info.get("present")
    labels = _strict_binary_labels(native_labels, attrs)
    if labels is None:
        return None, None, "present_values_are_not_strict_binary_integers", extras
    for attr in attrs:
        confidence = attributes[attr].get("confidence")
        if (isinstance(confidence, (bool, np.bool_))
                or not isinstance(confidence, (int, np.integer, float, np.floating))
                or not np.isfinite(float(confidence))):
            scores[attr] = float(labels[attr])
            continue
        confidence = min(max(float(confidence), 0.0), 1.0)
        scores[attr] = confidence if labels[attr] == 1 else 1.0 - confidence
    return labels, scores, None, extras


def _canonical_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vqa_source_descriptor(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": int(path.stat().st_size),
    }


def load_vqa_many_strict(jsonl_paths, vqa_to_key, attrs, *, source_priority=None):
    """Load VQA supervision with strict labels, provenance, and conflict audit.

    Only rows containing every current attribute and exact integer 0/1 values
    enter the returned map.  Identical complete rows may be repeated, but two
    distinct complete label vectors for one image at the same winning priority
    are reported as a conflict and that image is withheld from the map.  An
    optional explicit ``source_priority`` mapping may resolve lower-priority
    alternatives; every shadowed disagreement remains visible in the audit.
    Within one source, the last complete row wins (failed/incomplete retries do
    not erase an earlier complete answer).

    Returns ``(vqa_map, vqa_prob, audit)``.  The caller decides whether a
    reported conflict invalidates a committed trajectory or a future cache.
    """

    attrs = tuple(attrs)
    sources = []
    seen_paths = set()
    for raw_path in jsonl_paths or []:
        path = Path(raw_path)
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        sources.append(path)

    priority_map = source_priority or {}
    source_descriptors = []
    for path in sources:
        descriptor = _vqa_source_descriptor(path)
        spec = priority_map.get(descriptor["path"], priority_map.get(str(path), {}))
        if isinstance(spec, int):
            spec = {"priority": spec, "category": "explicit"}
        if not isinstance(spec, dict):
            spec = {}
        descriptor.update({
            "priority": int(spec.get("priority", 0)),
            "priority_category": str(spec.get("category", "equal-unresolved")),
        })
        source_descriptors.append(descriptor)
    complete_records = {}
    incomplete_by_image = {}
    invalid_rows = []
    shadowed_within_source = []

    for source_index, (path, descriptor) in enumerate(zip(sources, source_descriptors)):
        try:
            rows = _load_direct_label_rows(path)
        except Exception as error:
            invalid_rows.append({
                "source": descriptor,
                "row": None,
                "reason": f"source_parse_error:{type(error).__name__}:{error}",
            })
            continue
        for row_index, entry in enumerate(rows, start=1):
            if not isinstance(entry, dict):
                invalid_rows.append({
                    "source": descriptor,
                    "row": row_index,
                    "reason": "row_is_not_an_object",
                })
                continue
            image = entry.get("image")
            if not isinstance(image, str) or not image.strip():
                invalid_rows.append({
                    "source": descriptor,
                    "row": row_index,
                    "reason": "missing_or_invalid_image",
                })
                continue
            try:
                key = vqa_to_key(image)
            except Exception as error:
                invalid_rows.append({
                    "source": descriptor,
                    "row": row_index,
                    "image": image,
                    "reason": f"image_key_error:{type(error).__name__}:{error}",
                })
                continue
            if not isinstance(key, str) or not key:
                invalid_rows.append({
                    "source": descriptor,
                    "row": row_index,
                    "image": image,
                    "reason": "resolved_image_key_is_invalid",
                })
                continue

            provenance = {
                "source_index": source_index,
                "source_path": descriptor["path"],
                "source_sha256": descriptor["sha256"],
                "source_priority": descriptor["priority"],
                "source_priority_category": descriptor["priority_category"],
                "row": row_index,
                "raw_image": image,
            }
            if entry.get("answer") == "[FAILED]":
                incomplete_by_image.setdefault(key, []).append({
                    **provenance, "reason": "provider_failed",
                })
                continue
            try:
                if "answer" in entry:
                    (parsed_labels, parsed_scores, strict_error,
                     raw_extra_attributes) = _strict_labels_from_raw_answer(
                        entry["answer"], attrs,
                     )
                    if strict_error is not None:
                        incomplete_by_image.setdefault(key, []).append({
                            **provenance, "reason": strict_error,
                            "ignored_extra_raw_attributes": raw_extra_attributes,
                        })
                        continue
                else:
                    raw_extra_attributes = []
                    parsed_labels = {attr: entry.get(attr) for attr in attrs}
                    parsed_scores = {
                        attr: (float(entry[attr]) if attr in entry and isinstance(
                            entry[attr], (int, np.integer, float, np.floating)
                        ) and not isinstance(entry[attr], (bool, np.bool_)) else None)
                        for attr in attrs
                    }
            except Exception as error:
                incomplete_by_image.setdefault(key, []).append({
                    **provenance,
                    "reason": f"label_parse_error:{type(error).__name__}:{error}",
                })
                continue

            labels = _strict_binary_labels(parsed_labels, attrs)
            if labels is None:
                missing = ([attr for attr in attrs if attr not in parsed_labels]
                           if isinstance(parsed_labels, dict) else list(attrs))
                extra = (sorted(set(parsed_labels).difference(attrs))
                         if isinstance(parsed_labels, dict) else [])
                invalid = ({
                    attr: parsed_labels.get(attr)
                    for attr in attrs
                    if isinstance(parsed_labels, dict)
                    and (attr not in parsed_labels or not is_complete_binary_labels(
                        {attr: parsed_labels.get(attr)}, [attr]
                    ))
                } if isinstance(parsed_labels, dict) else {"labels": parsed_labels})
                incomplete_by_image.setdefault(key, []).append({
                    **provenance,
                    "reason": "incomplete_or_nonbinary_labels",
                    "missing_attributes": missing,
                    "extra_attributes": extra,
                    "invalid_values": invalid,
                })
                continue

            label_sha256 = _canonical_json_sha256(labels)
            record = {
                **provenance,
                "labels": labels,
                "label_sha256": label_sha256,
                "scores": parsed_scores,
                "ignored_extra_raw_attributes": raw_extra_attributes,
            }
            by_source = complete_records.setdefault(key, {})
            previous = by_source.get(source_index)
            if previous is not None and previous["label_sha256"] != label_sha256:
                shadowed_within_source.append({
                    "image": key,
                    "source_path": descriptor["path"],
                    "previous_row": previous["row"],
                    "previous_label_sha256": previous["label_sha256"],
                    "selected_row": row_index,
                    "selected_label_sha256": label_sha256,
                    "policy": "last_complete_row_wins",
                })
            by_source[source_index] = record

    vqa_map, vqa_prob = {}, {}
    provenance_by_image = {}
    all_complete_provenance_by_image = {}
    conflicts = {}
    shadowed_conflicts = {}
    for key in sorted(complete_records):
        records = list(complete_records[key].values())
        all_complete_provenance_by_image[key] = [
            {
                field: record[field]
                for field in (
                    "source_index", "source_path", "source_sha256", "row",
                    "raw_image", "label_sha256", "source_priority",
                    "source_priority_category", "ignored_extra_raw_attributes",
                )
            }
            for record in records
        ]
        winning_priority = max(record["source_priority"] for record in records)
        winning_records = [
            record for record in records
            if record["source_priority"] == winning_priority
        ]
        winning_variants = {}
        for record in winning_records:
            winning_variants.setdefault(record["label_sha256"], []).append(record)
        if len(winning_variants) != 1:
            conflicts[key] = {
                "winning_priority": winning_priority,
                "variants": [
                    {
                        "label_sha256": variant_hash,
                        "labels": variant_records[0]["labels"],
                        "provenance": [
                            {
                                field: record[field]
                                for field in (
                                    "source_index", "source_path", "source_sha256",
                                    "row", "raw_image",
                                )
                            }
                            for record in variant_records
                        ],
                    }
                    for variant_hash, variant_records in sorted(winning_variants.items())
                ],
            }
            continue
        chosen = winning_records[0]
        provenance_by_image[key] = [
            {
                field: record[field]
                for field in (
                    "source_index", "source_path", "source_sha256", "row",
                    "raw_image", "label_sha256", "source_priority",
                    "source_priority_category", "ignored_extra_raw_attributes",
                )
            }
            for record in winning_records
            if record["label_sha256"] == chosen["label_sha256"]
        ]
        disagreements = [
            record for record in records
            if record["label_sha256"] != chosen["label_sha256"]
        ]
        if disagreements:
            shadowed_conflicts[key] = {
                "selected": {
                    "labels": chosen["labels"],
                    "label_sha256": chosen["label_sha256"],
                    "priority": winning_priority,
                    "provenance": provenance_by_image[key],
                },
                "shadowed": [
                    {
                        "labels": record["labels"],
                        "label_sha256": record["label_sha256"],
                        "priority": record["source_priority"],
                        "priority_category": record["source_priority_category"],
                        "source_path": record["source_path"],
                        "source_sha256": record["source_sha256"],
                        "row": record["row"],
                    }
                    for record in disagreements
                ],
            }
        vqa_map[key] = dict(chosen["labels"])
        vqa_prob[key] = {
            attr: (
                None if chosen["scores"].get(attr) is None
                else float(chosen["scores"][attr])
            )
            for attr in attrs
        }

    canonical_labels = [
        {"image": key, "labels": vqa_map[key]}
        for key in sorted(vqa_map)
    ]
    audit = {
        "schema": "strict-vqa-cache-audit-v1",
        "attributes": list(attrs),
        "sources": source_descriptors,
        "source_set_sha256": _canonical_json_sha256(source_descriptors),
        "complete_image_count": len(vqa_map),
        "complete_record_count": int(sum(len(rows) for rows in complete_records.values())),
        "selected_labels_sha256": _canonical_json_sha256(canonical_labels),
        "provenance_by_image": provenance_by_image,
        "all_complete_provenance_by_image": all_complete_provenance_by_image,
        "incomplete_by_image": incomplete_by_image,
        "invalid_rows": invalid_rows,
        "conflicts": conflicts,
        "shadowed_conflicts": shadowed_conflicts,
        "shadowed_within_source": shadowed_within_source,
        "priority_policy": (
            "explicit-priority-then-last-complete-within-source-v1"
            if source_priority else "equal-priority-conflicts-unresolved-v1"
        ),
    }
    return vqa_map, vqa_prob, audit


def load_vqa(jsonl_path, vqa_to_key, attrs):
    """Parse VQA labels into (vqa_map, vqa_prob).



    Supports both raw VQA result JSONL rows with an ``answer`` field and legacy

    pre-parsed ``split/*_vqa.json`` rows shaped as ``{image, attr: 0/1}``.

    Missing file -> empty dicts.

    """

    vqa_map, vqa_prob = {}, {}
    p = Path(jsonl_path) if jsonl_path else None
    if p and p.is_file():
        rows = _load_direct_label_rows(p)
        invalid_rows = sum(not isinstance(entry, dict) for entry in rows)
        if invalid_rows:
            print(
                f"  [vqa] ignored {invalid_rows} non-record rows in {p}; "
                "strict supervision validation will still fail if any selected "
                "image lacks a complete label"
            )
        for e in rows:
            if not isinstance(e, dict):
                continue
            if e.get("answer") == "[FAILED]":
                continue
            key = vqa_to_key(e["image"])

            if "answer" in e:

                vqa_map[key] = extract_labels(e["answer"], attrs)

                vqa_prob[key] = extract_label_scores(e["answer"], attrs)

            else:

                labels = _direct_labels(e, attrs)

                vqa_map[key] = labels

                vqa_prob[key] = _direct_label_scores(labels)

    return vqa_map, vqa_prob


def load_vqa_many(jsonl_paths, vqa_to_key, attrs):
    """Load several VQA sources; later files override earlier rows for the same image."""
    vqa_map, vqa_prob = {}, {}
    for path in jsonl_paths or []:
        part_map, part_prob = load_vqa(path, vqa_to_key, attrs)
        vqa_map.update(part_map)
        vqa_prob.update(part_prob)
    return vqa_map, vqa_prob


def discover_task_vqa_sources(task_root):
    """Find raw JSONL and pre-parsed legacy labels that are safe to reuse."""
    task_root = Path(task_root)
    sources = []
    sources.extend(sorted((task_root / "qa").glob("**/*_results.jsonl")))
    sources.extend(sorted((task_root / "split").glob("**/*_vqa.json")))
    supervision = task_root / "supervision"
    sources.extend(sorted(supervision.glob("**/*_results.jsonl")))
    sources.extend(sorted(supervision.glob("**/*_vqa.json")))
    legacy_root = supervision / "legacy_vqa"
    if legacy_root.is_dir():
        for path in sorted(legacy_root.glob("**/*.json")):
            if path.name.startswith("_") or path.name in {
                "to_label.json", "reused_labels.json", "candidate_pool.json",
                "selected_train.json", "selected_audit.json", "manifest.json",
            }:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (isinstance(payload, list) and payload
                    and isinstance(payload[0], dict) and "image" in payload[0]):
                sources.append(path)
    return _existing_paths(sources)


def latest_jsonl(qa_dir):
    qa_dir = Path(qa_dir)
    if not qa_dir.is_dir():
        return None
    cands = sorted(qa_dir.glob("*_results.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def all_jsonl_results(qa_dir):
    qa_dir = Path(qa_dir)
    if not qa_dir.is_dir():
        return []
    return sorted(qa_dir.glob("*_results.jsonl"))


def _existing_paths(paths):
    out = []
    seen = set()
    for path in paths:
        if path is None:
            continue
        p = Path(path)
        if p.is_file() and str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def source_summary(paths):
    parts = []
    for path in paths or []:
        p = Path(path)
        parts.append(f"{p.parent.name}/{p.name}")
    return "+".join(parts) if parts else "(none)"




# Per-strategy VQA layout under each task's qa/: one folder per labeling pass.

#   qa/full/       -> a FULL-VQA pass (whole gallery labeled); both strategies share it.

#   qa/random/     -> a uniformly-random DB labeling pass (feeds the `random` strategy).

#   qa/two_stage/  -> a similarity-retrieval candidate labeling pass (feeds `two_stage`).
#   qa/exploit300_coverage200/ -> top-300 retrieval plus 200 MMR coverage candidates.
STRATEGY_QA_SUBDIRS = ("full", "random", "two_stage", "exploit300_coverage200")




def ensure_qa_subdirs(qa_dir):
    """Create per-strategy qa/ subdirectories (idempotent)."""
    for sub in STRATEGY_QA_SUBDIRS:
        (Path(qa_dir) / sub).mkdir(parents=True, exist_ok=True)




def resolve_vqa_sources(qa_dir, adapter_strats, split_dir=None):
    """Decide which VQA jsonl feeds each strategy (see STRATEGY_QA_SUBDIRS layout).



    - `qa/full/` present -> 'full' mode: every strategy draws its supervision subset S

      from the SAME full-VQA labels. This is the ONLY case where `random` vs `two_stage`

      differ meaningfully, because the labeled pool spans the whole gallery (random picks

      a uniform sample, two_stage picks the similarity-top). The VQA ceiling is available.

    - Otherwise 'per_strategy' mode: each strategy reads its OWN folder's jsonl
      (`qa/random/`, `qa/two_stage/`, or an active acquisition policy); a strategy with no jsonl
      is dropped. Labels are a sparse candidate set -> no ceiling. This avoids the

      degenerate case where both stages select from the same ~1000 retrieval candidates.



    Returns (mode, sources, full_jsonl), sources: strat -> list[jsonl Path].
    """
    qa_dir = Path(qa_dir)
    full_js = latest_jsonl(qa_dir / "full")
    if full_js:
        return "full", {s: [full_js] for s in adapter_strats}, full_js
    current_split = Path(split_dir) if split_dir is not None else qa_dir.parent / "split"
    root_split = qa_dir.parent / "split"
    legacy_random = _existing_paths([
        current_split / "onestage_vqa.json",
        root_split / "onestage_vqa.json",
    ])
    legacy_twostage = _existing_paths([
        current_split / "twostage_vqa.json",
        root_split / "twostage_vqa.json",
    ])
    legacy = {"random": legacy_random, "two_stage": legacy_twostage}
    sources = {}
    for s in adapter_strats:
        if s == "exploit300_coverage200":
            sources[s] = _existing_paths([
                *all_jsonl_results(qa_dir / "two_stage"),
                *legacy["two_stage"],
                *all_jsonl_results(qa_dir / s),
            ])
        else:
            sources[s] = _existing_paths([*all_jsonl_results(qa_dir / s), *legacy.get(s, [])])
    sources = {s: paths for s, paths in sources.items() if paths}
    if not sources:
        sources = {}
    return "per_strategy", sources, None




def select_label_candidates(strategies, train_set, emb_norm, p2i, query_idx, label_size):
    """Pick the union (across strategies) of the top-`label_size` images to VQA-label.

    No existing labels required (selects from the whole train pool)."""

    cand = set()
    for strat in strategies:
        if strat == "two_stage":
            ranked, _ = query_ranked_paths(train_set, emb_norm, p2i, query_idx)
        elif strat == "exploit300_coverage200":
            ranked = exploit300_coverage200_rank(train_set, emb_norm, p2i, query_idx, label_size)
        else:
            rng = np.random.default_rng(SPLIT_SEED)
            ranked = [train_set[i] for i in rng.permutation(len(train_set))]
        cand.update(ranked[:label_size])

    return sorted(cand)


def query_ranked_paths(paths, emb_norm, p2i, query_idx):
    proto = l2norm(emb_norm[query_idx].mean(axis=0, keepdims=True)).squeeze()
    sims = emb_norm @ proto
    ranked = sorted(paths, key=lambda rp: -sims[p2i[rp]])
    return ranked, sims


def mmr_coverage_paths(paths, emb_norm, p2i, sims, k, already=None, lam=0.65):
    already = set() if already is None else set(already)
    candidates = [rp for rp in paths if rp not in already]
    if k <= 0 or not candidates:
        return []
    idx = np.asarray([p2i[rp] for rp in candidates], dtype=np.int64)
    x = emb_norm[idx]
    qsim = sims[idx].astype(np.float32)
    max_sim_to_selected = np.zeros(len(candidates), dtype=np.float32)
    blocked = np.zeros(len(candidates), dtype=bool)
    selected = []
    for _ in range(min(k, len(candidates))):
        score = lam * qsim - (1.0 - lam) * max_sim_to_selected
        score[blocked] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break
        selected.append(candidates[j])
        blocked[j] = True
        sims_to_j = x @ x[j]
        max_sim_to_selected = np.maximum(max_sim_to_selected, sims_to_j.astype(np.float32))
    return selected


def exploit300_coverage200_rank(paths, emb_norm, p2i, query_idx, label_size):
    ranked, sims = query_ranked_paths(paths, emb_norm, p2i, query_idx)
    n_exploit = min(300, label_size)
    exploit = ranked[:n_exploit]
    coverage_pool = ranked[:min(2000, len(ranked))]
    coverage = mmr_coverage_paths(
        coverage_pool,
        emb_norm,
        p2i,
        sims,
        label_size - len(exploit),
        already=set(exploit),
        lam=0.65,
    )
    selected = exploit + coverage
    if len(selected) < label_size:
        used = set(selected)
        selected += [rp for rp in ranked if rp not in used][:label_size - len(selected)]
    return selected[:label_size]




def find_vqa_python(explicit=None):

    """Find a python interpreter that can import the VQA deps (openai/yaml/tqdm)."""

    cands = []

    if explicit:

        cands.append(explicit)

    cands += [sys.executable, "py -3.12", r"C:\Program Files\Python312\python.exe",

              r"C:\Program Files\Python311\python.exe"]

    for c in cands:

        argv = c.split() if (" -" in c) else [c]

        try:

            r = subprocess.run(argv + ["-c", "import openai,yaml,tqdm"],

                               capture_output=True, timeout=60)

            if r.returncode == 0:

                return argv

        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):

            continue

    raise SystemExit(

        "No python with VQA deps (openai/yaml/tqdm) found. Install them or pass "

        "--vqa-python <path-to-python>.")





def run_attribute_extraction(adapter, attr_config, vqa_python):

    """Run the query-image attribute extraction pipeline into task/qa/attributes.txt."""

    qa_root = adapter.task_root / "qa"

    query_dir = adapter.task_root / "query_pics"

    if not query_dir.is_dir() and (adapter.task_root / "query pics").is_dir():

        query_dir = adapter.task_root / "query pics"

    if not query_dir.is_dir():

        raise SystemExit(f"[auto-vqa] query_pics dir not found: {query_dir}")

    py = find_vqa_python(vqa_python)

    cfg = attr_config if Path(attr_config).is_absolute() else str(VQA_DIR / attr_config)

    cmd = py + ["extract_attributes.py", "--config", cfg,

                "--query-dir", str(query_dir), "--out-dir", str(qa_root)]

    print(f"  [auto-vqa] no qa/answer.md attributes; extracting attributes from query pics:\n"

          f"    {' '.join(cmd)}\n    (cwd={VQA_DIR})")

    subprocess.run(cmd, cwd=str(VQA_DIR), check=True)

    attr_txt = qa_root / "attributes.txt"
    if not attr_txt.is_file() or not attr_txt.read_text(encoding="utf-8").strip():
        raise SystemExit(f"[auto-vqa] attribute extraction did not create {attr_txt}")
    return _validate_vqa_attributes_file(attr_txt)




def ensure_vqa_attributes_file(adapter, attr_config, vqa_python):

    """Prepare qa/attributes.txt for VQA labeling.



    Existing attributes.txt wins. Otherwise use the task's answer.md final attributes.

    If answer.md has no usable final attributes, run the attribute extraction pipeline.

    """

    qa_root = adapter.task_root / "qa"

    qa_root.mkdir(parents=True, exist_ok=True)

    attr_txt = qa_root / "attributes.txt"
    if attr_txt.is_file() and attr_txt.read_text(encoding="utf-8").strip():
        _validate_vqa_attributes_file(attr_txt)
        print(f"  [auto-vqa] using existing attributes file: {attr_txt}")
        return attr_txt

    canonical_attr_txt = adapter.task_root / "attributes.txt"
    if canonical_attr_txt.is_file() and canonical_attr_txt.read_text(encoding="utf-8").strip():
        attrs = [_clean_attr_name(value) for value in re.split(
            r"[,，;；\n]+", canonical_attr_txt.read_text(encoding="utf-8-sig").strip("{} \n")
        )]
        attrs = [value for value in attrs if value]
        attr_txt.write_text(_format_attr_list(attrs), encoding="utf-8")
        print(f"  [auto-vqa] copied canonical attributes to: {attr_txt}")
        return _validate_vqa_attributes_file(attr_txt)

    attrs = parse_attributes_from_answer_md(qa_root / "answer.md")
    if attrs:

        attr_txt.write_text(_format_attr_list(attrs), encoding="utf-8")
        print(f"  [auto-vqa] wrote attributes.txt from answer.md: {attrs}")
        return _validate_vqa_attributes_file(attr_txt)


    return run_attribute_extraction(adapter, attr_config, vqa_python)





def run_vqa_pipeline(adapter, candidate_rels, vqa_config, vqa_python, workers,
                     output_dir=None, attr_config="config_attributes.yaml",
                     require_min_success=True,
                     apply_manual_content_filter_labels=True):
    """Auto-VQA: write the candidate basenames, then invoke vqa_label.py to label exactly

    those images (resumable). We pass EXPLICIT paths (images_dir = the flattened database,

    output_dir = the target qa subfolder, attributes = qa/attributes.txt) so this is robust

    to ambiguous task names. The config supplies api_key/base_url/model/prompt. Returns the

    jsonl path written under output_dir.

    """

    qa_root = adapter.task_root / "qa"

    out_dir = Path(output_dir) if output_dir else qa_root

    out_dir.mkdir(parents=True, exist_ok=True)

    attr_txt = ensure_vqa_attributes_file(adapter, attr_config, vqa_python)

    db_dir = adapter.database_dir

    if not Path(db_dir).is_dir():

        raise SystemExit(f"[auto-vqa] flattened database dir not found: {db_dir}")

    basenames = sorted({Path(rp).name for rp in candidate_rels})
    list_path = out_dir / "to_label.json"
    list_path.write_text(json.dumps(basenames, ensure_ascii=False), encoding="utf-8")
    image_manifest = adapter.vqa_image_manifest(list(candidate_rels))
    manifest_path = out_dir / "vqa_image_manifest.json"
    manifest_path.write_text(
        json.dumps(image_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


    py = find_vqa_python(vqa_python)

    cfg = vqa_config if Path(vqa_config).is_absolute() else str(VQA_DIR / vqa_config)

    cmd = py + ["vqa_label.py", "--config", cfg,

                "--images-dir", str(db_dir), "--output-dir", str(out_dir),
                "--attributes-file", str(attr_txt), "--images-manifest", str(manifest_path)]
    if workers:

        cmd += ["--workers", str(workers)]

    print(f"  [auto-vqa] labeling {len(basenames)} candidates -> running:\n    "
          f"{' '.join(cmd)}\n    (cwd={VQA_DIR})")
    subprocess.run(cmd, cwd=str(VQA_DIR), check=True)
    result_path = latest_jsonl(out_dir)
    if apply_manual_content_filter_labels:
        _apply_manual_content_filter_labels(
            adapter, result_path, list(adapter.attrs), out_dir,
        )
    strict_map, _, strict_audit = load_vqa_many_strict(
        [result_path] if result_path else [], adapter.vqa_to_key, list(adapter.attrs),
    )
    requested_keys = {adapter.vqa_to_key(relative_path) for relative_path in candidate_rels}
    successful = len(requested_keys.intersection(strict_map))
    if strict_audit["conflicts"]:
        examples = sorted(strict_audit["conflicts"])[:3]
        raise SystemExit(
            "[auto-vqa] conflicting complete binary labels in provider output "
            f"{result_path}: {examples}"
        )
    if require_min_success and successful < max(1, len(requested_keys) // 2):
        raise SystemExit(
            f"[auto-vqa] bulk labeling failure in {out_dir}: only "
            f"{successful}/{len(requested_keys)} candidates have complete strict "
            "binary labels; skipping this task."
        )
    return result_path


def _apply_manual_content_filter_labels(adapter, result_path, attrs, out_dir):
    """Apply reviewed labels only to provider content-filter failures.

    Every registry entry must contain explicit binary labels and review
    provenance. Network, rate-limit, parsing, and other failures remain
    unresolved and cannot enter training through this fallback.
    """
    if not result_path or not Path(result_path).is_file():
        return 0
    rows = _load_direct_label_rows(result_path)
    filtered = {
        str(row.get("image", "")).replace("\\", "/"): row
        for row in rows
        if row.get("answer") == "[FAILED]"
        and "data_inspection_failed" in str(row.get("error", ""))
    }
    if not filtered:
        return 0

    registry = {"entries": []}
    if MANUAL_CONTENT_FILTER_REGISTRY.is_file():
        registry = json.loads(MANUAL_CONTENT_FILTER_REGISTRY.read_text(encoding="utf-8"))
    already_applied = {
        str(row.get("image", "")).replace("\\", "/")
        for row in rows
        if row.get("label_source") == "manual_content_filter"
    }
    matching = {}
    for entry in registry.get("entries", []):
        if entry.get("dataset") != adapter.dataset or entry.get("task") != adapter.task:
            continue
        image = str(entry.get("image", "")).replace("\\", "/")
        if image in filtered:
            matching[image] = entry

    applied = []
    with Path(result_path).open("a", encoding="utf-8") as handle:
        for image, entry in sorted(matching.items()):
            if image in already_applied:
                continue
            labels = entry.get("labels", {})
            missing = [attr for attr in attrs if labels.get(attr) not in (0, 1)]
            if missing:
                raise ValueError(
                    f"manual content-filter label lacks binary attributes {missing}: {image}"
                )
            record = {
                "image": image,
                **{attr: int(labels[attr]) for attr in attrs},
                "label_source": "manual_content_filter",
                "fallback_reason": "provider_data_inspection_failed",
                "reviewed_by": entry.get("reviewed_by", "unspecified"),
                "review_note": entry.get("review_note", ""),
                "timestamp": entry.get("created_at"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            applied.append(record)

    unresolved = sorted(set(filtered) - set(matching))
    audit = {
        "schema_version": 1,
        "dataset": adapter.dataset,
        "task": adapter.task,
        "result_file": str(Path(result_path)),
        "eligibility_rule": "error contains data_inspection_failed",
        "content_filter_failures": len(filtered),
        "manual_labels_applied": len(applied),
        "manual_fraction_of_batch": len(applied) / max(
            1, len({row.get("image") for row in rows})
        ),
        "applied": applied,
        "unresolved_content_filter_images": unresolved,
    }
    audit_path = Path(out_dir) / "manual_content_filter_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if applied:
        print(
            f"  [manual-vqa] applied {len(applied)} reviewed content-filter labels; "
            f"audit={audit_path}"
        )
    return len(applied)




# --------------------------------------------------------------------------- scoring

def score_mlp(model, emb_subset):
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = next(model.buffers()).device
    X = torch.tensor(emb_subset, dtype=torch.float32).to(dev)
    with torch.no_grad():

        logits = model(X).cpu().numpy().ravel()

    return 1.0 / (1.0 + np.exp(-logits))





def _patch_chunk(patch_rows, start: int, end: int):
    """Return a patch-token chunk.

    `patch_rows` can be a normal ndarray or `(patch_memmap, indices)`. The latter
    avoids materializing a full CelebA-size gallery patch array in RAM.
    """
    if isinstance(patch_rows, tuple):
        patches, indices = patch_rows
        return patches[np.asarray(indices[start:end], dtype=np.int64)]
    return patch_rows[start:end]


def _patch_len(patch_rows) -> int:
    if isinstance(patch_rows, tuple):
        return len(patch_rows[1])
    return len(patch_rows)


def score_attn(model, patch_subset, text_q, chunk=512):
    return score_attn_many([model], patch_subset, text_q, chunk=chunk)[0]


def score_attn_many(models, patch_subset, text_q, chunk=1024):
    """Score several seed probes while transferring every patch chunk once.

    Patch stores are float16 memmaps.  Preserving that dtype until the CUDA copy
    halves host/PCIe traffic; conversion to the models' float32 dtype happens on
    device, so the model inputs and scoring definition remain float32.
    """
    if not models:
        return np.empty((0, _patch_len(patch_subset)), dtype=np.float32)
    dev = next(models[0].parameters()).device
    tq = None if text_q is None else text_q.to(dev)
    outputs = [[] for _ in models]
    with torch.inference_mode(), warnings.catch_warnings():
        # Read-only memmaps are safe here: neither this function nor the models
        # mutate their input tensor.  Suppress PyTorch's generic writeability warning.
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable.*",
        )
        for i in range(0, _patch_len(patch_subset), chunk):
            array = np.asarray(_patch_chunk(patch_subset, i, i + chunk))
            host = torch.from_numpy(array)
            X = host.to(device=dev, dtype=torch.float32)
            for output, model in zip(outputs, models):
                logits = model(X, tq).cpu().numpy().ravel()
                output.append((1.0 / (1.0 + np.exp(-logits))).astype(np.float32))
    return np.vstack([
        np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        for parts in outputs
    ])




def score_models(m, kind, emb_rows, patch_rows, text_q_attr):

    """Dispatch scoring by feature kind. Returns P(present) per row.

    Single-model: mlp/attn. Ensemble (multi-boundary): mlp_ens/attn_ens -> mean prob."""

    if kind == "mlp":

        return score_mlp(m, emb_rows)

    if kind == "mlp_ens":

        _, mean_probs, _ = multi_boundary_predict(m, emb_rows)

        return mean_probs

    if kind == "attn":

        return score_attn(m, patch_rows, text_q_attr)

    if kind == "attn_ens":

        _, mean_probs, _ = multi_boundary_attn_predict(m, patch_rows, text_q_attr)

        return mean_probs

    raise ValueError(f"unknown scoring kind {kind!r}")





def score_attr_models(models, emb_rows, patch_rows, text_q):

    """Score both task attributes with a trained per-attribute model dict."""

    out = {}

    n = len(emb_rows)

    for attr in ATTRS:

        if attr not in models:

            out[attr] = np.zeros(n, dtype=np.float32)

            continue

        model, kind = models[attr]

        out[attr] = score_models(model, kind, emb_rows, patch_rows, text_q[attr])

    return out





def _safe_logit(p):

    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)

    return np.log(p / (1.0 - p))





def _joint_calib_features(scores_by_attr):
    return np.column_stack([_safe_logit(scores_by_attr[a]) for a in ATTRS]).astype(np.float32)


def _safe_zscore(values):
    values = np.asarray(values, dtype=np.float64)
    sd = float(values.std())
    if sd <= 1e-8:
        return values - float(values.mean())
    return (values - float(values.mean())) / sd


def zscore_joint_score(scores_by_attr):
    vals = [_safe_zscore(scores_by_attr[a]) for a in ATTRS]
    if not vals:
        return np.array([], dtype=np.float32)
    return np.mean(np.vstack(vals), axis=0).astype(np.float32)




def fit_joint_calibrator(scores_by_attr, labeled_records, seed):

    """Learn M4 logistic composition from VQA-supervised joint labels.



    The calibrator changes only the joint ranking score. Single-attribute scores

    and their AP tables remain the underlying method's outputs.

    """

    keep = []

    labels = []

    for i, rec in enumerate(labeled_records):

        vals = [rec.get(a) for a in ATTRS]

        if any(v is None for v in vals):

            continue

        keep.append(i)

        labels.append(int(all(int(v) == 1 for v in vals)))

    y = np.asarray(labels, dtype=np.int64)

    if len(y) < 4 or len(np.unique(y)) < 2:

        return None

    X = _joint_calib_features(scores_by_attr)[np.asarray(keep, dtype=np.int64)]

    clf = LogisticRegression(

        solver="liblinear", C=1.0,

        max_iter=1000, random_state=seed,

    )

    clf.fit(X, y)

    return clf





def joint_score(scores_by_attr, calibrator=None, method: str | None = None):
    if calibrator is None:
        if method in ZSCORE_JOINT_METHOD_BASE:
            return zscore_joint_score(scores_by_attr)
        return joint_score_from(scores_by_attr)
    return calibrator.predict_proba(_joint_calib_features(scores_by_attr))[:, 1]




# --------------------------------------------------------------------------- training






def train_method(method, kind, labeled, sub_idx, U_idx, ctx, seed):
    """Return {attr: (model, kind)} for a trained method (skip attr with <2 classes)."""
    _seed_everything(seed)
    base_method = base_method_for(method)
    emb, patches, p2i, text_q, input_dim = (
        ctx["emb"], ctx["patches"], ctx["p2i"], ctx["text_q"], ctx["input_dim"])
    fixed_validation = "val_labeled" in ctx
    val_labeled = ctx.get("val_labeled")
    if fixed_validation:
        supported = {
            "mlp_baseline", "kfold_pu", "nnpu", "dcpu", "pu_ranking",
            "triplet_loss", "attention_pooling", "attribute_conditioned_attention",
        }
        if method not in supported or base_method not in supported or kind != TRAINED_METHODS[method]:
            raise ValueError(f"fixed validation is unsupported for method/kind: {method}/{kind}")
        if not isinstance(val_labeled, list) or not val_labeled:
            raise ValueError("fixed validation requires non-empty ctx['val_labeled'] records")
        fit_paths = [record.get("image") for record in labeled]
        val_paths = [record.get("image") for record in val_labeled]
        for label, image_paths in (("fit", fit_paths), ("validation", val_paths)):
            if not image_paths or any(not isinstance(path, str) or path not in p2i for path in image_paths):
                raise ValueError(f"fixed {label} contains missing or unknown image IDs")
            if len(set(image_paths)) != len(image_paths):
                raise ValueError(f"fixed {label} contains duplicate image IDs")
        if set(sub_idx) != set(fit_paths) or len(sub_idx) != len(fit_paths):
            raise ValueError("fixed fit labels must match the selected fit image IDs")
        fit_indices = {int(p2i[path]) for path in fit_paths}
        val_indices = {int(p2i[path]) for path in val_paths}
        if fit_indices & val_indices:
            raise ValueError("fixed validation overlaps supervised fitting rows")
        u_indices = np.asarray(U_idx)
        u_sample = np.asarray(ctx.get("U_sample", []))
        for name, indices in (("U_idx", u_indices), ("U_sample", u_sample)):
            if indices.ndim != 1 or (indices.size and (
                indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(emb))
            )):
                raise ValueError(f"fixed validation requires valid integer {name} indices")
            if set(indices.tolist()) & (fit_indices | val_indices):
                raise ValueError(f"fixed validation or fit rows overlap the {name} unlabeled pool")
        if not set(u_sample.tolist()).issubset(set(u_indices.tolist())):
            raise ValueError("fixed U_sample must be a subset of U_idx")
    models = {}
    is_mlp = kind in ("mlp", "mlp_ens")

    for attr in ATTRS:

        if is_mlp:
            X, y, matched_images = prepare_attribute_data(attr, labeled, emb, p2i)
        else:
            X, y, _ = prepare_attribute_patches(attr, labeled, patches, p2i, lambda s: s)
            matched_images = []
        validation_kwargs = {}
        if fixed_validation:
            if any(record.get(attr) not in (0, 1) for record in [*labeled, *val_labeled]):
                raise ValueError(f"fixed fit/validation labels must be complete binary values for {attr}")
            if is_mlp:
                X_val, y_val, matched_val = prepare_attribute_data(attr, val_labeled, emb, p2i)
            else:
                X_val, y_val, matched_val = prepare_attribute_patches(attr, val_labeled, patches, p2i, lambda s: s)
            if len(y) != len(labeled) or len(y_val) != len(val_labeled) or len(matched_val) != len(val_labeled):
                raise ValueError(f"fixed fit/validation feature preparation dropped rows for {attr}")
            validation_kwargs = {"validation_data": (X_val, y_val)}
        if len(np.unique(y)) < 2 or int(y.sum()) < 2 or int((y == 0).sum()) < 2:
            print(f"    [{method}/{attr}] SKIP (insufficient class balance: pos={int(y.sum())})")
            continue


        if base_method == 'mlp_baseline':
            m, _ = train_one_attribute(attr, X, y, input_dim, epochs=EPOCHS, seed=seed, **validation_kwargs)
        elif base_method == 'kfold_pu':
            cvp = cross_validate_probs(attr, X, y, input_dim, epochs=EPOCHS, seed=seed)

            w = np.where(y == 1, cvp, 1.0)

            m, _ = train_one_attribute_weighted(attr, X, y, w, input_dim, epochs=EPOCHS, seed=seed, **validation_kwargs)
        elif base_method == 'dcpu':
            m, _ = train_one_attribute_dcpu(attr, X, y, emb[ctx["U_sample"]], input_dim,
                                            epochs=EPOCHS, seed=seed, **validation_kwargs)
        elif base_method == 'nnpu':
            m, _ = train_one_attribute_nnpu(attr, X, y, emb[ctx["U_sample"]], input_dim,
                                            epochs=EPOCHS, seed=seed, **validation_kwargs)
        elif base_method == 'pu_ranking':
            m, _ = train_one_attribute_pu_ranking(attr, X, y, emb[ctx["U_sample"]], input_dim,
                                                  epochs=EPOCHS, seed=seed, **validation_kwargs)
        elif base_method == 'triplet_loss':
            m, _ = train_one_attribute_triplet_loss(
                attr, X, y, input_dim, epochs=EPOCHS, seed=seed, **validation_kwargs,
            )
        elif base_method == 'attention_pooling':
            m, _ = train_one_attribute_attention_pooling(
                attr, X, y, patch_dim=ctx["patch_dim"],
                epochs=EPOCHS, seed=seed, **validation_kwargs,
            )
        elif base_method == 'attribute_conditioned_attention':
            m, _ = train_one_attribute_attn(attr, X, y, text_q[attr],
                                            patch_dim=ctx["patch_dim"], text_dim=ctx["text_dim"],
                                            epochs=EPOCHS, seed=seed, **validation_kwargs)
        else:
            raise ValueError(f"Unknown paper probe: {method}")
        if fixed_validation:
            m._probe_training_metrics = {
                **_,
                "validation_policy": "explicit-fixed-holdout-v1",
                "fit_count": int(len(y)),
                "validation_count": int(len(y_val)),
                "validation_checkpoint_policy": (
                    "best-validation-auc" if len(np.unique(y_val)) > 1
                    else "first-epoch-single-class-validation"
                ),
            }
        models[attr] = (m, kind)
    return models


def train_method_reusing_base(method, kind, labeled, sub_idx, U_idx, ctx, seed, model_cache):
    """Train once per base probe; score-only variants reuse the exact models."""
    base_method = base_method_for(method)
    cache_key = (base_method, kind)
    if cache_key in model_cache:
        print(f"    [{method}] reuse base probe: {base_method}")
        return model_cache[cache_key]
    models = train_method(method, kind, labeled, sub_idx, U_idx, ctx, seed)
    model_cache[cache_key] = models
    return models








# --------------------------------------------------------------------------- subsets

def build_subset(strategy, seed, pool_paths, vqa_map, emb_norm, p2i, query_idx,

                 sup_size=SUP_SIZE, label_size=LABEL_SIZE):

    """Select the supervision subset S from the train pool (= gallery minus test).



    Only VQA-labeled images are eligible (training needs labels). We rank the labeled

    pool by `strategy`, take the labeling budget L = top `label_size`, then supervision

    S = top `sup_size` (so S ⊆ L). The extra L\\S images are treated as unlabeled during

    training ("over-labeled → pretend unlabeled"). When fewer than `label_size`/`sup_size`

    labeled images exist in the pool, S/L shrink to what's available.



    Returns (chosen_S, labeled_records_S, L_paths).

    """

    labeled_pool = [rp for rp in pool_paths if rp in vqa_map]

    if strategy == "two_stage":
        ranked, _ = query_ranked_paths(labeled_pool, emb_norm, p2i, query_idx)
    elif strategy == "exploit300_coverage200":
        ranked = exploit300_coverage200_rank(labeled_pool, emb_norm, p2i, query_idx, label_size)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(labeled_pool))
        ranked = [labeled_pool[i] for i in order]

    else:

        raise ValueError(strategy)

    L = ranked[:label_size]

    S = ranked[:min(sup_size, len(L))]

    labeled = [{"image": rp, **vqa_map[rp]} for rp in S]

    return S, labeled, L





# --------------------------------------------------------------------------- metrics

def rank_metrics(scores, yt):

    yt = np.asarray(yt)

    out = {"ap": float(average_precision_score(yt, scores)) if yt.sum() > 0 else float("nan")}

    order = np.argsort(-scores)

    total_pos = int(yt.sum())

    for k in TOPKS:

        topk = order[:k]

        hit = int(yt[topk].sum())

        out[f"p@{k}"] = hit / k

        out[f"r@{k}"] = hit / total_pos if total_pos > 0 else float("nan")

    return out





def clf_at_threshold(yt, preds):

    return {

        "acc@0.5": float(accuracy_score(yt, preds)),

        "prec@0.5": float(precision_score(yt, preds, zero_division=0)),

        "rec@0.5": float(recall_score(yt, preds, zero_division=0)),

        "f1@0.5": float(f1_score(yt, preds, zero_division=0)),

    }





def best_f1_clf(scores, yt):
    """Exact optimal-threshold metrics from one score sort.

    Evaluate only boundaries between distinct score values. This is both exact
    and substantially faster than repeatedly materializing predictions for a
    fixed threshold grid on a large gallery.
    """
    scores = np.asarray(scores, dtype=np.float64)
    yt = np.asarray(yt, dtype=np.int64)
    empty = {"acc@0.5": 0.0, "prec@0.5": 0.0, "rec@0.5": 0.0, "f1@0.5": 0.0}
    if not len(scores):
        return empty
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = yt[order]
    tp = np.cumsum(sorted_y, dtype=np.int64)
    fp = np.cumsum(1 - sorted_y, dtype=np.int64)
    total_pos = int(tp[-1])
    total_neg = int(len(yt) - total_pos)
    boundaries = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp_b = tp[boundaries].astype(np.float64)
    fp_b = fp[boundaries].astype(np.float64)
    fn_b = total_pos - tp_b
    denom = 2.0 * tp_b + fp_b + fn_b
    f1 = np.divide(2.0 * tp_b, denom, out=np.zeros_like(tp_b), where=denom > 0)
    j = int(np.argmax(f1))
    tn = total_neg - fp_b[j]
    predicted = tp_b[j] + fp_b[j]
    return {
        "acc@0.5": float((tp_b[j] + tn) / len(yt)),
        "prec@0.5": float(tp_b[j] / predicted) if predicted > 0 else 0.0,
        "rec@0.5": float(tp_b[j] / total_pos) if total_pos > 0 else 0.0,
        "f1@0.5": float(f1[j]),
    }




def best_clf_metrics(scores, yt):

    """Best-threshold (F1-maximizing) classification metrics, suffixed with _best."""

    b = best_f1_clf(scores, yt)

    return {"acc_best": b["acc@0.5"], "prec_best": b["prec@0.5"],

            "rec_best": b["rec@0.5"], "f1_best": b["f1@0.5"]}





def eval_method_on_test(scores_by_attr, joint_score, gt_test, joint_gt, is_prob):
    """scores_by_attr: {attr: arr over test}. Returns nested metric dict."""
    res = {}
    for key, info in RANKING_SPEC.items():
        attrs = tuple(info.get("attrs") or info["gt"][1:])
        if len(attrs) == 1:
            s = scores_by_attr[attrs[0]]
            yt = gt_test[attrs[0]]
            pred = (s >= 0.5).astype(int)
        elif key == JOINT_KEY:
            s = joint_score
            yt = joint_gt
            pred = combo_pred_at_threshold(scores_by_attr, attrs, 0.5)
        else:
            s = combo_score_from(scores_by_attr, attrs)
            yt = combo_gt_from(gt_test, attrs)
            pred = combo_pred_at_threshold(scores_by_attr, attrs, 0.5)
        m = rank_metrics(s, yt)
        m.update(clf_at_threshold(yt, pred))
        m.update(best_clf_metrics(s, yt))
        res[key] = m
    return res




# --------------------------------------------------------------------------- aggregation

def aggregate(per_seed_list):

    """per_seed_list: list of metric dicts (same structure). Return mean/std dict."""

    keys_path = []



    def walk(d, prefix):

        for k, v in d.items():

            if isinstance(v, dict):

                walk(v, prefix + [k])

            else:

                keys_path.append(prefix + [k])

    walk(per_seed_list[0], [])



    agg = {}

    for path in keys_path:

        vals = []

        for d in per_seed_list:

            cur = d

            for p in path:

                cur = cur[p]

            if cur == cur:  # not nan

                vals.append(cur)

        node = agg

        for p in path[:-1]:

            node = node.setdefault(p, {})

        if vals:

            node[path[-1]] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

        else:

            node[path[-1]] = {"mean": float("nan"), "std": float("nan"), "n": 0}

    return agg





# --------------------------------------------------------------------------- report

def fmt(cell, pct=True):

    if cell is None or cell["n"] == 0 or cell["mean"] != cell["mean"]:

        return "-"

    mean, std = cell["mean"], cell["std"]

    if pct:

        return f"{mean*100:.1f}±{std*100:.1f}"

    return f"{mean:.3f}±{std:.3f}"





def attr_table(agg, attr_key):

    lines = [

        "| 方法 | AP | P@50 | P@100 | R@50 | R@100 | F1@0.5 |",

        "|---|---|---|---|---|---|---|",

    ]

    for method in ALL_METHODS:

        a = agg.get(method, {}).get(attr_key)

        if a is None:

            continue

        # Full-VQA ceiling scores are binary 0/1 -> ranking metrics (AP/P@K/R@K) are

        # meaningless (massive ties); show "-" instead of misleading numbers.

        rk = (lambda key: "-") if method == VQA_CEILING else (lambda key: fmt(a.get(key)))

        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(

            METHOD_LABELS[method], rk("ap"), rk("p@50"), rk("p@100"),

            rk("r@50"), rk("r@100"), fmt(a.get("f1@0.5"))))

    return "\n".join(lines)







def joint_table(agg, key=JOINT_KEY):
    lines = ["| ?? | AP | P@50 | P@100 | R@50 | R@100 |", "|---|---|---|---|---|---|"]
    for method in ALL_METHODS:
        a = agg.get(method, {}).get(key)
        if a is None:
            continue
        # Full-VQA ceiling: binary scores -> all ranking columns meaningless -> "-".

        rk = (lambda key: "-") if method == VQA_CEILING else (lambda key: fmt(a.get(key)))

        lines.append("| {} | {} | {} | {} | {} | {} |".format(

            METHOD_LABELS[method], rk("ap"), rk("p@50"),

            rk("p@100"), rk("r@50"), rk("r@100")))

    return "\n".join(lines)



def clf_table(agg, key):
    """Classification metrics. Acc/Prec/Rec/F1@0.5 at fixed 0.5 (embedding: optimal);

    plus Prec/Rec/F1(最优阈值) = F1-maximizing threshold on the test scores."""

    lines = ["| 方法 | Acc@0.5 | Prec@0.5 | Rec@0.5 | F1@0.5 | Prec* | Rec* | F1(最优阈值) |",

             "|---|---|---|---|---|---|---|---|"]

    for method in ALL_METHODS:

        a = agg.get(method, {}).get(key)

        if a is None:

            continue

        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(

            METHOD_LABELS[method], fmt(a.get("acc@0.5")), fmt(a.get("prec@0.5")),

            fmt(a.get("rec@0.5")), fmt(a.get("f1@0.5")),

            fmt(a.get("prec_best")), fmt(a.get("rec_best")), fmt(a.get("f1_best"))))

    return "\n".join(lines)


def ranking_title(key: str, info: dict) -> str:
    attrs = tuple(info.get("attrs") or info["gt"][1:])
    if key == JOINT_KEY:
        return f"{JOINT_LABEL} (GT: {combo_slug(attrs)})"
    if len(attrs) == 1:
        return f"`{attrs[0]}` (GT: {key})"
    return f"{' & '.join(ATTR_KEY[a] for a in attrs)}"


def is_single_ranking(info: dict) -> bool:
    attrs = tuple(info.get("attrs") or info["gt"][1:])
    return len(attrs) == 1



def _metric_section(agg, title, note):
    """One metrics block (test or gallery): selected attr-combination ranking + clf tables."""
    L = [f"## {title}\n"]
    if note:
        L.append(note + "\n")
    for key, info in RANKING_SPEC.items():
        L.append(f"### {ranking_title(key, info)}\n")
        if is_single_ranking(info):
            L.append(attr_table(agg, key) + "\n")
        else:
            L.append(joint_table(agg, key) + "\n")
    L.append("### ?????@0.5 ?????embedding baseline ????????\n")
    for key, info in RANKING_SPEC.items():
        L.append(f"**{ranking_title(key, info)}**\n")
        L.append(clf_table(agg, key) + "\n")
    return L




def build_report_single(agg, agg_gallery, strat, meta):
    """Per-strategy report for one stage dir (onestage/twostage)."""

    test_pos_str = ", ".join(f"{k}={meta['test_pos'][k]}" for k in KEYS)

    gallery_pos_str = ", ".join(f"{k}={meta['gallery_pos'][k]}" for k in KEYS)

    L = []

    L.append(f"# ????? ???? ({DATASET}, {JOINT_LABEL}) - ?? `{strat}`?{STAGE_NAME[strat]}?\n")

    L.append("??????????? T??? `??\\T` ? VQA ?????????? S ???"

             "????????? GT ????????? (AP/P@K/R@K)?F1@0.5 ???\n")

    L.append("## ????\n")

    L.append(f"- ???gallery???-query??{meta['n_gallery']} ???? VQA ??? {meta['n_labeled']} ??")

    L.append(f"- ????? T?{meta['n_test']} ???? GT ???? {int(TEST_FRAC*100)}%, split seed={SPLIT_SEED}??"

             f"?? {test_pos_str}, {joint_slug()}={meta['test_joint']}?")

    L.append(f"- gallery GT ???{gallery_pos_str}, {joint_slug()}={meta['gallery_joint']}?")

    L.append(f"- ???? L={meta['label_size']}????? S={meta['sup_size']}????? L={meta['n_label_used']}?S={meta['n_sub']}"

             "??? `??\\T` ?????????????????")

    L.append(f"- U ????{U_CAP}??? ranking/nnPU/SAPU ????????????seeds={meta['seeds']}?")

    L.append(f"- VQA ??????test={meta['test_cov']:.2f}, gallery={meta['gallery_cov']:.2f}?"

             "??????????? `Full-VQA ????????` ??\n")



    L += _metric_section(

        agg, "??????held-out, ? seed mean?std?",

        "????????????????? seed ? mean?std (%)?"

        + ("" if meta.get("ceiling_test") else f"???? test ??? {meta['test_cov']:.2f}<{CEILING_COV}???????"))

    L += _metric_section(

        agg_gallery, "?? gallery ???? seed, ??-query, ?????",

        f"???????????? query ????? {meta['n_gallery']} ????"

        "?? seeds[0] ???????????? std???????????"

        + ("" if meta.get("ceiling_gallery") else f"???? gallery ??? {meta['gallery_cov']:.2f}<{CEILING_COV}???????"))

    return "\n".join(L)


def _fmt_scalar_pct(val) -> str:
    if val is None:
        return "-"
    val = float(val)
    if val != val:
        return "-"
    return f"{val * 100:.1f}"


def _compact_metric(agg: dict, method: str, key: str, metric: str) -> str:
    val = (((agg or {}).get(method) or {}).get(key) or {}).get(metric)
    if isinstance(val, dict):
        mean = val.get("mean")
        std = val.get("std")
        if mean is None:
            return "-"
        mean_s = _fmt_scalar_pct(mean)
        if mean_s == "-":
            return "-"
        if std is None or abs(float(std)) < 1e-12:
            return mean_s
        std_s = _fmt_scalar_pct(std)
        if std_s == "-":
            return mean_s
        return f"{mean_s}±{std_s}"
    if val is None:
        return "-"
    return _fmt_scalar_pct(val)


def _compact_target_label(key: str, info: dict) -> str:
    attrs = tuple(info.get("attrs") or info["gt"][1:])
    slugs = [ATTR_KEY[a] for a in attrs]
    if len(slugs) == 1:
        return slugs[0]
    return " & ".join(slugs)


def _compact_pool_table(agg: dict, pool_name: str) -> str:
    lines = [
        f"### {pool_name}",
        "",
        "| Target | Method | AP | F1* |",
        "|---|---|---:|---:|",
    ]
    for key, info in RANKING_SPEC.items():
        target = _compact_target_label(key, info)
        for method, label in COMPACT_REPORT_METHODS:
            lines.append(
                f"| {target} | {label} | "
                f"{_compact_metric(agg, method, key, 'ap')} | "
                f"{_compact_metric(agg, method, key, 'f1_best')} |"
            )
    return "\n".join(lines)


def build_compact_task_report(agg_by_strategy, agg_gallery_by_strategy, meta_by_strategy):
    """Task-root summary: key methods x selected attr combinations, AP and best-threshold F1."""
    lines = [
        f"# Method Comparison Summary ({DATASET}, {JOINT_LABEL})",
        "",
        "This compact summary keeps only AP and best-threshold F1 (F1*) for the key methods.",
        "Missing entries (`-`) mean that method or attribute-combination was not present in this run.",
        "",
        f"- attributes: {', '.join(KEYS)}",
        f"- exported combinations: {', '.join(RANKING_SPEC.keys())}",
        "",
    ]
    for strat in ["random", "two_stage"]:
        if strat not in agg_by_strategy:
            continue
        smeta = meta_by_strategy.get(strat, {})
        lines.append(f"## {strat} ({STAGE_NAME[strat]})")
        lines.append("")
        if smeta:
            lines.append(
                f"- seeds: {smeta.get('seeds')}  "
                f"L={smeta.get('label_size')} S={smeta.get('sup_size')}  "
                f"used L={smeta.get('n_label_used')} S={smeta.get('n_sub')}"
            )
            lines.append("")
        lines.append(_compact_pool_table(agg_by_strategy.get(strat, {}), "Test"))
        lines.append("")
        lines.append(_compact_pool_table(agg_gallery_by_strategy.get(strat, {}), "Gallery"))
        lines.append("")
    return "\n".join(lines)


DETAILED_REPORT_METHODS = [(key, key, METHOD_LABELS[key]) for key in TRAINED_METHODS]


def _metric_mean_value(agg: dict, method: str, key: str, metric: str) -> float | None:
    val = (((agg or {}).get(method) or {}).get(key) or {}).get(metric)
    if isinstance(val, dict):
        val = val.get("mean")
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _format_detailed_metric(agg: dict, method: str, key: str, metric: str) -> str:
    val = (((agg or {}).get(method) or {}).get(key) or {}).get(metric)
    if isinstance(val, dict):
        mean = val.get("mean")
        std = val.get("std")
        if mean is None:
            return "-"
        mean_s = _fmt_scalar_pct(mean)
        if mean_s == "-":
            return "-"
        if std is None or abs(float(std)) < 1e-12:
            return mean_s
        std_s = _fmt_scalar_pct(std)
        if std_s == "-":
            return mean_s
        return f"{mean_s}+/-{std_s}"
    if val is None:
        return "-"
    return _fmt_scalar_pct(val)


def _style_metric_row(agg: dict, key: str, metric: str) -> list[str]:
    numeric = {
        alias: _metric_mean_value(agg, method, key, metric)
        for alias, method, _ in DETAILED_REPORT_METHODS
    }
    non_upper = {
        alias: val for alias, val in numeric.items()
        if alias != "Upper" and val is not None
    }
    ordered = sorted(set(non_upper.values()), reverse=True)
    best = ordered[0] if ordered else None
    second = ordered[1] if len(ordered) > 1 else None

    cells = []
    for alias, method, _ in DETAILED_REPORT_METHODS:
        text = _format_detailed_metric(agg, method, key, metric)
        val = numeric[alias]
        if text == "-" or val is None:
            cells.append(text)
            continue
        if alias == "Upper":
            cells.append(f"**{text}**")
        elif best is not None and abs(val - best) <= 1e-12:
            cells.append(f"**{text}**")
        elif second is not None and abs(val - second) <= 1e-12:
            cells.append(f"<u>{text}</u>")
        else:
            cells.append(text)
    return cells


def _detailed_metric_table(key: str, metric: str,
                           agg_by_strategy: dict,
                           agg_gallery_by_strategy: dict) -> str:
    headers = [alias for alias, _, _ in DETAILED_REPORT_METHODS]
    title = "AP" if metric == "ap" else "F1*"
    lines = [
        f"#### {title}",
        "",
        "| Strategy | Split | " + " | ".join(headers) + " |",
        "| --- | --- | " + " | ".join(["---:"] * len(headers)) + " |",
    ]
    for strat in ["random", "two_stage"]:
        if strat not in agg_by_strategy and strat not in agg_gallery_by_strategy:
            continue
        for split_name, agg in [
            ("test", agg_by_strategy.get(strat, {})),
            ("gallery", agg_gallery_by_strategy.get(strat, {})),
        ]:
            cells = _style_metric_row(agg, key, metric)
            lines.append(f"| {strat} | {split_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_detailed_task_report(agg_by_strategy, agg_gallery_by_strategy, meta_by_strategy):
    """Task-root metric_analysis.md in the BMW-convertible detailed-table format."""
    lines = [
        "## Detailed Metric Tables",
        "",
        "Each target has separate AP and F1* tables. If `Upper` exists, it is always **bold**; "
        "among non-Upper methods, the best value is also **bold** and the second-best value is "
        "<u>underlined</u>. Values are percentages; test rows show mean+/-std when multiple "
        "seeds are available.",
        "",
        "Method aliases:",
        "",
    ]
    for alias, _, label in DETAILED_REPORT_METHODS:
        lines.append(f"- `{alias}`: {label}")
    lines.append("")

    for key, info in RANKING_SPEC.items():
        lines.append(f"### {ranking_title(key, info)}")
        lines.append("")
        lines.append(_detailed_metric_table(key, "ap", agg_by_strategy, agg_gallery_by_strategy))
        lines.append("")
        lines.append(_detailed_metric_table(key, "f1_best", agg_by_strategy, agg_gallery_by_strategy))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_methods_results(mr_dir, strat, agg, agg_gallery, per_seed, trained_scores,
                          gallery_scores, baseline_scores, test_set, gt_test, joint_gt,
                          gallery_paths, gt_gallery, joint_gt_gallery):
    """One <method>_results.json per method."""
    gt_block = {}
    for key, info in RANKING_SPEC.items():
        attrs = tuple(info.get("attrs") or info["gt"][1:])
        vals = gt_test[attrs[0]] if len(attrs) == 1 else combo_gt_from(gt_test, attrs)
        gt_block[key] = [int(x) for x in vals]

    # Shared (method-independent) gallery index + GT written ONCE; per-method files only
    # carry that method's scores + metrics (avoids duplicating large gallery paths).
    gt_block_g = {}
    for key, info in RANKING_SPEC.items():
        attrs = tuple(info.get("attrs") or info["gt"][1:])
        vals = gt_gallery[attrs[0]] if len(attrs) == 1 else combo_gt_from(gt_gallery, attrs)
        gt_block_g[key] = [int(x) for x in vals]
    (mr_dir / "_gallery_index.json").write_text(

        json.dumps({"gallery_images": list(gallery_paths), "gallery_gt": gt_block_g},

                   ensure_ascii=False), encoding="utf-8")

    for method in ALL_METHODS:

        if method not in agg and method not in agg_gallery:

            continue

        ts = trained_scores.get(method) or baseline_scores.get(method)

        rec = {

            "method": method, "label": METHOD_LABELS.get(method, method),

            "strategy": strat, "stage": STAGE_NAME[strat],

            "test_metrics": _jsonable(agg.get(method)),

            "gallery_metrics": _jsonable(agg_gallery.get(method)),

            "test_images": list(test_set),

            "test_gt": gt_block,

            "gallery_scores_ref": "_gallery_index.json",

        }

        if ts is not None:

            rec["test_scores"] = ts

        gs = gallery_scores.get(method)
        if gs is not None:
            # Learned probes store full-gallery scores by attribute slug and need to be
            # converted to ranking keys. Embedding baselines are already emitted by
            # ranking key (e.g. "joint", pairwise combinations), so preserve them.
            if any(k in RANKING_SPEC for k in gs):
                rec["gallery_scores"] = {
                    k: np.asarray(v) for k, v in gs.items() if k in RANKING_SPEC
                }
            else:
                rec["gallery_scores"] = scores_by_ranking_key(gs, gs.get(JOINT_KEY))
            rec["gallery_scores"] = {k: list(map(float, np.asarray(v)))
                                      for k, v in rec["gallery_scores"].items()}
        if per_seed.get(method):

            rec["per_seed_metrics"] = per_seed[method]

        (mr_dir / f"{method}_results.json").write_text(

            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")





# --------------------------------------------------------------------------- main

def main():
    global LABEL_SIZE, SUP_SIZE
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval harness experiments. For long runs, redirect logs under "
            "logs/<run_name>/ or archive any root-level .log files after completion."
        )
    )
    parser.add_argument("--dataset", default="cars", choices=["cars", "sun", "cub", "hico", "awa2", "celeba"],
                        help="dataset adapter to use")

    parser.add_argument("--task", default=BMW_SEDAN_TASK,

                        help="task name (Cars registry/sidecar, SUN registry, or CUB sidecar)")

    parser.add_argument("--joint-label", default=None,

                        help="display label for the joint concept, e.g. 'BMW 轿车'")

    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)

    parser.add_argument("--label-size", type=int, default=LABEL_SIZE,

                        help="VQA labeling budget L (images sent to the labeler, default 1000). "

                             "The VQA ceiling is shown automatically only when labels cover the "

                             "eval pool (i.e. you labeled (nearly) the whole DB).")

    parser.add_argument("--sup-size", type=int, default=SUP_SIZE,

                        help="supervision budget S, the subset of L actually trained on "

                             "(default 1000). S <= L; the extra L\\S is treated as unlabeled.")

    parser.add_argument("--smoke", action="store_true", help="1 seed, mlp_baseline only")

    parser.add_argument("--methods", default=None,
                        help="comma-separated subset of trained methods to run (default: all). "
                             f"choices: {','.join(TRAINED_METHODS)}")
    parser.add_argument("--backbone", default="clip", choices=sorted(BACKBONES),
                        help="pooled image embedding backbone. Non-CLIP backbones write under "
                             f"{BACKBONE_RUN_ROOT}/<backbone>/ to avoid overwriting legacy results.")
    parser.add_argument("--isolate-backbone-output", action="store_true",
                        help="also write CLIP results under embedding_runs/clip/ instead of the "
                             "legacy task-root onestage/twostage folders.")
    parser.add_argument("--embedding-only", action="store_true",
                        help="compute only VQA ceiling/embedding baselines; skip trained methods")
    parser.add_argument("--no-top50", action="store_true", help="skip top-50 export (already generated)")

    parser.add_argument("--top50-only", action="store_true", help="only (re)generate top-50, skip metrics")
    parser.add_argument("--merge-existing", action="store_true",
                        help="merge newly computed method metrics into an existing metrics.json instead "
                             "of replacing already-computed methods. Useful for incremental method runs.")
    parser.add_argument("--auto-vqa", action="store_true",

                        help="full end-to-end: if VQA labels are missing/insufficient, run the "

                             "VQA pipeline (vqa_label.py) on the L candidate budget, then continue. "

                             "Requires a config with an API key (see --vqa-config).")

    parser.add_argument("--auto-vqa-missing-strategies", action="store_true",
                        help="with --auto-vqa, also label any missing per-strategy source under "
                             "qa/<strategy>/ even when another strategy already has labels. "
                             "Useful for adding random/onestage labels to a task that already "
                             "has qa/two_stage/.")
    parser.add_argument("--strategies", default=None,
                        help="comma-separated supervision strategies to run "
                             "(default: adapter strategies, e.g. random,two_stage).")
    parser.add_argument("--vqa-config", default="config.yaml",

                        help="VQA config (path or name under the VQA folder); supplies api_key/"

                             "base_url/model/prompt. Default config.yaml.")

    parser.add_argument("--attr-config", default="config_attributes.yaml",

                        help="attribute-extraction config used by --auto-vqa when qa/attributes.txt "

                             "is missing and qa/answer.md has no final attributes. Default "

                             "config_attributes.yaml.")

    parser.add_argument("--vqa-python", default=None,

                        help="python interpreter for vqa_label.py (needs openai/yaml/tqdm); "

                             "auto-detected if omitted.")

    parser.add_argument("--vqa-workers", type=int, default=64, help="VQA concurrency (default 64)")
    parser.add_argument(
        "--vqa-preflight-manifest",
        default=None,
        help="approved outbound VQA preflight; validated immediately before each API batch",
    )
    parser.add_argument(
        "--vqa-preflight-sha256",
        default=None,
        help="approved canonical hash for --vqa-preflight-manifest",
    )
    parser.add_argument(
        "--vqa-preflight-task-order",
        type=int,
        default=None,
        help="task order bound into an approved budget-curve preflight",
    )
    parser.add_argument("--iterative-vqa", action="store_true",
                        help="run the isolated iterative 100+3x50 VQA policy; old stages are untouched")
    parser.add_argument(
        "--iterative-exclude-provider-content-rejections",
        action="store_true",
        help=(
            "exclude exact unresolved provider data_inspection_failed rows from "
            "uncommitted iterative selections and deterministically backfill them"
        ),
    )
    parser.add_argument("--iterative-stage", default="iterative_vqa_100_50_v1",
                        help="task-local stage name for --iterative-vqa (default iterative_vqa_100_50_v1)")
    parser.add_argument("--iterative-work-root", default=None,
                        help="isolated output/cache root for a batch-local iterative run")
    parser.add_argument("--iterative-max-rounds", type=int, default=3, choices=range(0, 9),
                        help="number of 50-image refinement rounds after round 0 (default 3; up to 8 / 500 total labels)")
    parser.add_argument("--iterative-initial-labels", type=int, default=100,
                        help="iterative round-0 candidate budget (currently fixed protocol: 100)")
    parser.add_argument("--iterative-round-labels", type=int, default=50,
                        help="iterative refinement candidate budget (currently fixed protocol: 50)")
    parser.add_argument("--iterative-train-per-round", type=int, default=40,
                        help="iterative refinement train count (currently fixed protocol: 40)")
    parser.add_argument("--iterative-audit-per-round", type=int, default=10,
                        help="iterative refinement permanent-audit count (currently fixed protocol: 10)")
    parser.add_argument("--iterative-max-labels", type=int, default=300,
                        help="hard per-task iterative label ceiling (default 300)")
    parser.add_argument("--iterative-stop-at-labels", type=int, default=250,
                        help="default iterative stopping point before optional round 4 (default 250)")
    parser.add_argument("--iterative-adaptive-budget", action="store_true",
                        help="allow audit-based early stopping at 200 labels or extension from 250 to 300 labels")
    parser.add_argument("--iterative-audit-min-gain", type=float, default=0.05,
                        help="minimum consecutive prequential audit P@10 gain needed to extend 250 to 300 labels")
    parser.add_argument("--iterative-resume", action="store_true",
                        help="resume an existing iterative stage from its round state")
    parser.add_argument("--iterative-dry-run", action="store_true",
                        help="write selection manifests and diagnostics, but never invoke VQA")
    parser.add_argument("--iterative-export-final-scores", action="store_true",
                        help="recompute fixed iterative/static final score arrays for fixed fusion; no VQA calls")
    parser.add_argument("--iterative-defer-round-evaluation", action="store_true",
                        help="run acquisition only and defer per-round model/static-control evaluation to a later budget sweep")
    args = parser.parse_args()

    if any(
        value is not None
        for value in (
            args.vqa_preflight_manifest,
            args.vqa_preflight_sha256,
            args.vqa_preflight_task_order,
        )
    ) and not args.iterative_vqa:
        raise SystemExit("VQA preflight is supported only with --iterative-vqa")
    if args.iterative_exclude_provider_content_rejections and not args.iterative_vqa:
        raise SystemExit(
            "--iterative-exclude-provider-content-rejections requires --iterative-vqa"
        )

    if args.iterative_vqa:
        # Import lazily so normal harness runs retain their established imports and paths.
        from iterative_vqa_runner import run_iterative_vqa
        run_iterative_vqa(args)
        return


    LABEL_SIZE = args.label_size

    SUP_SIZE = args.sup_size

    if SUP_SIZE > LABEL_SIZE:

        raise SystemExit(f"--sup-size ({SUP_SIZE}) must be <= --label-size ({LABEL_SIZE})")

    configure(args.dataset, args.task, args.joint_label)
    A = ADAPTER
    if args.strategies:
        requested_strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        allowed_strategies = set(A.strategies) | {"exploit300_coverage200"}
        unknown_strategies = [s for s in requested_strategies if s not in allowed_strategies]
        if unknown_strategies:
            raise SystemExit(
                f"unknown --strategies {unknown_strategies}; choose from {sorted(allowed_strategies)}")
        A.strategies = requested_strategies
    report_root = apply_backbone_output_layout(args.backbone, args.isolate_backbone_output)
    seeds = [0] if args.smoke else args.seeds
    print(f"=== frozen-test comparison ===  dataset={DATASET}  task={TASK}  "
          f"backbone={args.backbone}  CUDA={torch.cuda.is_available()}  seeds={seeds}")
    print(f"    attrs={ATTRS}  slugs={KEYS}  joint='{JOINT_LABEL}'")
    print(f"    output_root={report_root}")

    # --- load features / records / GT (via adapter)
    emb, backbone_meta = load_backbone_embeddings(A, args.backbone)
    emb_norm = l2norm(emb)
    records = A.records
    p2i = A.p2i

    input_dim = emb.shape[1]

    gt_by_attr = A.gt_by_attr



    # Which methods will run -> whether we need the (potentially huge) patch tokens.

    methods_to_run = [] if args.embedding_only else (["mlp_baseline"] if args.smoke else list(TRAINED_METHODS.keys()))
    if args.methods:
        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in requested if m not in TRAINED_METHODS]
        if unknown:
            raise SystemExit(f"unknown --methods {unknown}; choose from {list(TRAINED_METHODS)}")
        methods_to_run = requested
    validate_methods_for_backbone(args.backbone, methods_to_run)
    emb_methods = active_embedding_methods(args.backbone)
    need_patches = any(TRAINED_METHODS.get(m) in ("attn", "attn_ens") for m in methods_to_run)
    if need_patches:
        # mmap so a full-database patch file (GBs) is read lazily by index, not into RAM.
        patches, patch_meta = load_backbone_patches(A, args.backbone)
        patch_dim = int(patches.shape[-1])
        print(f"  patches {patches.shape} (mmap), emb {emb.shape}")
    else:
        patches = None
        patch_meta = {}
        patch_dim = 0
        print(f"  patches skipped (no attention method requested), emb {emb.shape}")


    # --- query rows are EXCLUDED from everything (they're the prototype, not targets).

    i2p = {int(i): rp for rp, i in p2i.items()}

    query_rp = {i2p[int(i)] for i in A.query_idx if int(i) in i2p}

    # gallery G = full database minus query images = the retrieval/eval universe.

    gallery_paths = [rp for rp in records["relative_path"] if rp not in query_rp]



    # --- per-strategy VQA sources under qa/{full,random,two_stage}/ (see resolve_vqa_sources).

    # full mode: both strategies share one full-VQA pass (random vs two_stage differ

    # because the labeled pool spans the gallery). per_strategy mode: each strategy reads

    # its own folder; strategies without a jsonl are dropped (no degenerate onestage).

    qa_dir = A.task_root / "qa"

    ensure_qa_subdirs(qa_dir)

    vqa_mode, vqa_sources, full_js = resolve_vqa_sources(qa_dir, list(A.strategies), SPLIT_DIR)
    vqa_by_strat = {s: load_vqa_many(paths, A.vqa_to_key, ATTRS) for s, paths in vqa_sources.items()}
    full_map, full_prob = (load_vqa(full_js, A.vqa_to_key, ATTRS) if full_js else ({}, {}))
    _src_str = ", ".join(f"{s}:{source_summary(paths)}"
                         for s, paths in vqa_sources.items()) or "(none)"
    print(f"  Gallery (DB - query): {len(gallery_paths)}  |  VQA mode: {vqa_mode}  |  sources: {{{_src_str}}}")



    def gt_of(attr, rp):

        return int(gt_by_attr[attr].get(rp, 0))



    # --- frozen stratified split: test T = TEST_FRAC of the gallery GT (seed=SPLIT_SEED),

    # carved FIRST and shared by all strategies/seeds. Train pool = gallery \ T. The

    # supervision subset is drawn from the VQA-labeled images in the pool (never from T).

    rng = np.random.default_rng(SPLIT_SEED)

    buckets = {}

    for rp in gallery_paths:

        buckets.setdefault(tuple(gt_of(a, rp) for a in ATTRS), []).append(rp)

    test_set = []

    for key, items in buckets.items():

        items = sorted(items)

        rng.shuffle(items)

        test_set += items[:int(round(TEST_FRAC * len(items)))]

    test_set = sorted(test_set)

    test_lookup = set(test_set)

    train_set = sorted(rp for rp in gallery_paths if rp not in test_lookup)



    test_idx = np.array([p2i[rp] for rp in test_set], dtype=np.int64)

    gallery_idx = np.array([p2i[rp] for rp in gallery_paths], dtype=np.int64)

    # Materialize test-set features once (patches is mmap -> avoid re-reading per method/attr).

    test_emb = emb[test_idx]

    test_patches = (patches, test_idx) if patches is not None else None


    def gt_vec(paths, attr):

        return np.array([gt_of(attr, rp) for rp in paths], dtype=int)



    gt_test = {a: gt_vec(test_set, a) for a in ATTRS}

    joint_gt = joint_gt_from(gt_test)

    gt_gallery = {a: gt_vec(gallery_paths, a) for a in ATTRS}

    joint_gt_gallery = joint_gt_from(gt_gallery)



    def usable_vqa_count(strat):

        vmap = vqa_by_strat.get(strat, ({}, {}))[0]

        return sum(1 for rp in train_set if rp in vmap)



    def strategy_needs_vqa(strat):
        cand = select_label_candidates([strat], train_set, emb_norm, p2i,
                                       list(A.query_idx), LABEL_SIZE)
        vmap = vqa_by_strat.get(strat, ({}, {}))[0]
        if not cand:
            return False
        if strat not in vqa_sources:
            return True
        return any(rp not in vmap for rp in cand)


    # --- auto-VQA (full end-to-end): by default, if NO strategy has enough labels,

    # label a two_stage candidate budget into qa/two_stage/. With

    # --auto-vqa-missing-strategies, also fill missing/partial per-strategy folders

    # such as qa/random/ without reusing another strategy's sparse labels.

    if args.auto_vqa and (
        not vqa_sources
        or args.auto_vqa_missing_strategies
        or any(strategy_needs_vqa(s) for s in A.strategies)
    ):
        if vqa_sources and vqa_mode == "full":
            targets = []
        elif vqa_sources:
            targets = [s for s in A.strategies if strategy_needs_vqa(s)]
        elif (not vqa_sources) and args.auto_vqa_missing_strategies:
            targets = list(A.strategies)
        else:
            targets = ["two_stage" if "two_stage" in A.strategies else list(A.strategies)[0]]
        for target in targets:
            cand = select_label_candidates([target], train_set, emb_norm, p2i,
                                           list(A.query_idx), LABEL_SIZE)
            cached = set(vqa_by_strat.get(target, ({}, {}))[0])
            missing = [rp for rp in cand if rp not in cached]
            print(f"  [auto-vqa] '{target}' candidates={len(cand)}, "
                  f"cached={len(cand) - len(missing)}, new={len(missing)} "
                  f"(usable labels now {usable_vqa_count(target)}/{min(LABEL_SIZE, len(train_set))}) "
                  f"-> qa/{target}/ ...")
            if missing:
                run_vqa_pipeline(A, missing, args.vqa_config, args.vqa_python, args.vqa_workers,
                                 output_dir=qa_dir / target, attr_config=args.attr_config)
        vqa_mode, vqa_sources, full_js = resolve_vqa_sources(qa_dir, list(A.strategies), SPLIT_DIR)
        vqa_by_strat = {s: load_vqa_many(paths, A.vqa_to_key, ATTRS) for s, paths in vqa_sources.items()}
        full_map, full_prob = (load_vqa(full_js, A.vqa_to_key, ATTRS) if full_js else ({}, {}))
        print(f"  [auto-vqa] reloaded: mode={vqa_mode}, strategies with labels={list(vqa_sources)}")


    # Active strategies = those (allowed by the adapter) that actually have labels.

    strategies = [s for s in A.strategies if s in vqa_sources]



    split_dir = SPLIT_DIR

    split_dir.mkdir(parents=True, exist_ok=True)



    # No labels for any strategy -> can't train. Emit per-strategy to-label lists and stop.
    if not strategies:
        for _strat in A.strategies:
            _cands = select_label_candidates([_strat], train_set, emb_norm, p2i,
                                             list(A.query_idx), LABEL_SIZE)
            (split_dir / f"to_label_{_strat}.json").write_text(
                json.dumps(_cands, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(
            f"No VQA labels found under {qa_dir}/{{{','.join(STRATEGY_QA_SUBDIRS)}}}/. Wrote "
            f"{LABEL_SIZE}-image to-label lists to {split_dir}/to_label_*.json; VQA-label them "
            f"and drop the *_results.jsonl into qa/full/ (full pass, both stages) or "
            f"qa/<strategy>/ (per-strategy), then re-run. Or pass --auto-vqa.")


    # Per-strategy supervision/coverage (labels can differ per strategy in per_strategy mode).

    def _cov(paths, vmap):

        return (sum(1 for rp in paths if rp in vmap) / len(paths)) if paths else 0.0

    per_strat_meta = {}

    for _strat in strategies:

        vmap = vqa_by_strat[_strat][0]

        lp = [rp for rp in train_set if rp in vmap]

        n_label = min(LABEL_SIZE, len(lp))

        n_sub = min(SUP_SIZE, n_label)

        tcov, gcov = _cov(test_set, vmap), _cov(gallery_paths, vmap)

        per_strat_meta[_strat] = {

            "source": source_summary(vqa_sources[_strat]),
            "n_labeled": sum(1 for rp in gallery_paths if rp in vmap),

            "n_label_used": n_label, "n_sub": n_sub,

            "test_cov": round(tcov, 4), "gallery_cov": round(gcov, 4),

            "ceiling_test": bool(vqa_mode == "full" and tcov >= CEILING_COV),

            "ceiling_gallery": bool(vqa_mode == "full" and gcov >= CEILING_COV),

        }

        if 0 < len(lp) < SUP_SIZE:

            print(f"  [budget/{_strat}] WARNING: only {len(lp)} labeled in train pool "

                  f"(< sup-size {SUP_SIZE}); supervision capped at {n_sub}.")



    # VQA ceiling is meaningful only in 'full' mode (full-gallery 0/1 labels).

    test_cov = _cov(test_set, full_map)

    gallery_cov = _cov(gallery_paths, full_map)

    ceiling_test = vqa_mode == "full" and test_cov >= CEILING_COV

    ceiling_gallery = vqa_mode == "full" and gallery_cov >= CEILING_COV



    meta = {
        "dataset": DATASET, "task": TASK, "attrs": ATTRS, "slugs": KEYS,
        **backbone_meta,
        **patch_meta,
        "output_root": str(report_root),
        "vqa_mode": vqa_mode,
        "label_size": LABEL_SIZE, "sup_size": SUP_SIZE,
        "n_gallery": len(gallery_paths), "n_test": len(test_set), "n_train": len(train_set),

        "test_pos": {ATTR_KEY[a]: int(gt_test[a].sum()) for a in ATTRS},

        "test_joint": int(joint_gt.sum()),

        "gallery_pos": {ATTR_KEY[a]: int(gt_gallery[a].sum()) for a in ATTRS},

        "gallery_joint": int(joint_gt_gallery.sum()), "seeds": seeds,

        "strategies": strategies, "per_strategy": per_strat_meta,

    }

    (split_dir / "test.json").write_text(json.dumps(test_set, ensure_ascii=False), encoding="utf-8")

    (split_dir / "gallery.json").write_text(json.dumps(gallery_paths, ensure_ascii=False), encoding="utf-8")

    (split_dir / "split_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for _strat in strategies:

        sm = per_strat_meta[_strat]

        print(f"  Split[{_strat}] -> gallery={len(gallery_paths)}, test={len(test_set)}, "

              f"train_pool={len(train_set)}, L={sm['n_label_used']}/S={sm['n_sub']} "

              f"(src={sm['source']}, cov test={sm['test_cov']:.2f}/gallery={sm['gallery_cov']:.2f}, "

              f"ceiling test={sm['ceiling_test']}/gallery={sm['ceiling_gallery']})")



    # supervision subsets per stage (canonical seed=SPLIT_SEED): the VQA-labeled images

    # each strategy trains on. random -> onestage_vqa.json, two_stage -> twostage_vqa.json.

    _qidx = list(A.query_idx)

    for _strat in strategies:

        _, _labeled, _ = build_subset(_strat, SPLIT_SEED, train_set, vqa_by_strat[_strat][0],

                                      emb_norm, p2i, _qidx, sup_size=SUP_SIZE, label_size=LABEL_SIZE)

        (split_dir / f"{STAGE_NAME[_strat]}_vqa.json").write_text(

            json.dumps(_labeled, ensure_ascii=False, indent=2), encoding="utf-8")



    # --- text queries (encode once only for attention methods / top-50 fallback).
    # Pooled-embedding baselines do not consume these hidden-state text queries.
    if need_patches or args.top50_only:
        print("  Encoding attribute texts ...")
        text_backbone = args.backbone if args.backbone in BACKBONE_TEXT_MODEL else "clip"
        text_q = encode_attribute_texts_for_backbone(ATTRS, text_backbone, BACKBONE_TEXT_MODEL[text_backbone])
        text_dim = list(text_q.values())[0].shape[-1]
    else:
        text_q = {a: torch.zeros(1, 1) for a in ATTRS}
        text_dim = 1


    query_idx = list(A.query_idx)



    if args.top50_only:
        generate_top50(strategies, methods_to_run, train_set, vqa_by_strat, emb, emb_norm, patches,
                       p2i, text_q, text_dim, input_dim, query_idx, gallery_paths, gallery_idx,
                       gt_by_attr, emb_methods=emb_methods, backbone=args.backbone)
        print("\nTop-50 regenerated (top50-only mode).")

        return



    # --- strategy/seed-independent baselines on TEST (compute once)

    emb_base, emb_raw = compute_embedding_baselines(
        emb_norm, test_idx, query_idx, text_q, gt_test, joint_gt, emb_methods, backbone=args.backbone)
    if ceiling_test:
        vqa_base, vqa_raw = compute_vqa_direct_baseline(test_set, full_map, gt_test, joint_gt, vqa_prob=full_prob)
        baseline_test_scores = {VQA_CEILING: vqa_raw, **emb_raw}
        f1_parts = [f"{k} F1@0.5={vqa_base[k]['f1@0.5']['mean']:.3f}"
                    for k in RANKING_SPEC if k in vqa_base]
        print("  [vqa ceiling/test] " + "  ".join(f1_parts))
    else:

        vqa_base, baseline_test_scores = None, dict(emb_raw)

        print(f"  [vqa ceiling/test] skipped (mode={vqa_mode}, coverage {test_cov:.2f} < {CEILING_COV})")



    # --- strategy/seed-independent baselines on GALLERY (compute once)

    emb_base_g, emb_raw_g = compute_embedding_baselines(
        emb_norm, gallery_idx, query_idx, text_q, gt_gallery, joint_gt_gallery, emb_methods, backbone=args.backbone)
    if ceiling_gallery:
        vqa_base_g, vqa_raw_g = compute_vqa_direct_baseline(gallery_paths, full_map, gt_gallery,
                                                            joint_gt_gallery, vqa_prob=full_prob)
        baseline_gallery_scores = {VQA_CEILING: vqa_raw_g, **emb_raw_g}
    else:
        vqa_base_g = None
        baseline_gallery_scores = dict(emb_raw_g)
        print(f"  [vqa ceiling/gallery] skipped (mode={vqa_mode}, coverage {gallery_cov:.2f} < {CEILING_COV})")


    results = {"meta": meta, "strategies": {}}

    agg_by_strategy = {}

    agg_gallery_by_strategy = {}

    # per-strategy, per-method test-set scores accumulated across seeds (for methods_results)

    test_scores_by_strategy = {}



    # Top-50 reuse: instead of a separate seed=42 retrain, we score the FULL candidate

    # library once using the SAME models trained for the metrics at the first seed. This

    # is reproducible (the top-50 ranking comes from a model that's actually in the run)

    # and avoids retraining every method again.

    TOP50_SEED = seeds[0]

    full_cand_emb = emb[gallery_idx]

    full_cand_patches = (patches, gallery_idx) if patches is not None else None
    trained_full_scores = {}   # strat -> method -> {attr: gallery scores @ TOP50_SEED}

    top50_chosen = {}          # strat -> supervision subset @ TOP50_SEED (for exclusion)

    method_times = {}          # method -> [per-seed wall-clock seconds]



    for strat in strategies:

        print(f"\n{'#'*70}\n# STRATEGY: {strat}\n{'#'*70}")
        per_seed = {m: [] for m in methods_to_run}
        test_acc = {m: {k: [] for k in RANKING_SPEC} for m in methods_to_run}
        vqa_map_s = vqa_by_strat[strat][0]

        for seed in seeds:

            print(f"\n--- {strat} | seed {seed} ---")

            chosen, labeled, _ = build_subset(strat, seed, train_set, vqa_map_s, emb_norm, p2i,

                                              query_idx, sup_size=SUP_SIZE, label_size=LABEL_SIZE)

            sub_set = set(chosen)

            if seed == TOP50_SEED:

                top50_chosen[strat] = chosen

            U_paths = [rp for rp in train_set if rp not in sub_set]

            U_idx = np.array([p2i[rp] for rp in U_paths], dtype=np.int64)

            u_rng = np.random.default_rng(1000 + seed)

            U_sample = U_idx if len(U_idx) <= U_CAP else U_idx[u_rng.choice(len(U_idx), U_CAP, replace=False)]
            ctx = {"emb": emb, "patches": patches, "p2i": p2i, "text_q": text_q,
                   "input_dim": input_dim, "text_dim": text_dim, "patch_dim": patch_dim,
                   "U_sample": U_sample}


            model_cache = {}
            for method in methods_to_run:
                kind = TRAINED_METHODS[method]

                print(f"\n  >>> {strat}/seed{seed}/{method} ({kind})")

                _t0 = time.perf_counter()

                models = train_method_reusing_base(
                    method, kind, labeled, chosen, U_idx, ctx, seed, model_cache,
                )
                calibrator = None

                if method in CALIBRATED_METHOD_BASE:

                    chosen_idx = np.array([p2i[rp] for rp in chosen], dtype=np.int64)

                    chosen_emb = emb[chosen_idx]

                    chosen_patches = np.asarray(patches[chosen_idx]) if patches is not None else None

                    sba_train = score_attr_models(models, chosen_emb, chosen_patches, text_q)

                    calibrator = fit_joint_calibrator(sba_train, labeled, seed)

                    status = "fit" if calibrator is not None else "skipped (single-class joint labels)"

                    print(f"  [joint-calib] {status}")

                # score test

                sba = score_attr_models(models, test_emb, test_patches, text_q)

                joint = joint_score(sba, calibrator, method=method)
                metrics = eval_method_on_test(sba, joint, gt_test, joint_gt, is_prob=True)
                per_seed[method].append(metrics)
                srk = scores_by_ranking_key(sba, joint)
                for key, score in srk.items():
                    test_acc[method][key].append(score)
                # reuse this seed's model to score the full library for top-50 export

                if seed == TOP50_SEED:

                    fs = score_attr_models(models, full_cand_emb, full_cand_patches, text_q)

                    fs[JOINT_KEY] = joint_score(fs, calibrator, method=method)
                    trained_full_scores.setdefault(strat, {})[method] = fs

                _dt = time.perf_counter() - _t0

                method_times.setdefault(method, []).append(_dt)

                print(f"  [time] {strat}/seed{seed}/{method} = {_dt:.1f}s"

                      + ("  (incl. full-gallery scoring)" if seed == TOP50_SEED else ""))



        # aggregate trained methods on TEST (multi-seed mean +/- std)

        agg = {m: aggregate(per_seed[m]) for m in methods_to_run if per_seed[m]}

        # merge single-value baselines (std 0): full-VQA ceiling + embedding baselines

        if vqa_base is not None:

            agg[VQA_CEILING] = vqa_base

        for m in emb_methods:
            agg[m] = emb_base[m]
        agg_by_strategy[strat] = agg

        results["strategies"][strat] = {"per_seed": per_seed}

        # mean test scores across seeds per trained method

        test_scores_by_strategy[strat] = {

            m: {k: np.mean(np.vstack(v), axis=0).tolist() for k, v in test_acc[m].items()}

            for m in methods_to_run if any(test_acc[m][k] for k in test_acc[m])}



        # GALLERY metrics (full DB - query) from the TOP50_SEED full-library scores (single seed).

        # Includes the supervision images (leaky), shown as the real retrieval deliverable.

        agg_g = {}

        for m in methods_to_run:

            fs = trained_full_scores.get(strat, {}).get(m)

            if not fs:

                continue

            sba_g = {a: np.asarray(fs[a]) for a in ATTRS}
            jg = np.asarray(fs.get(JOINT_KEY, joint_score_from(sba_g)))
            res_g = eval_method_on_test(sba_g, jg,
                                        gt_gallery, joint_gt_gallery, is_prob=True)
            agg_g[m] = _single_to_agg(res_g)

        for m in emb_methods:
            agg_g[m] = emb_base_g[m]
        if vqa_base_g is not None:

            agg_g[VQA_CEILING] = vqa_base_g

        agg_gallery_by_strategy[strat] = agg_g



    # --- reports (per stage: onestage=random, twostage=two_stage). Only the
    # strategies actually run for this dataset (A.strategies) produce reports.
    compact_meta_by_strategy = {}
    for strat in strategies:
        sdir = STAGE_DIRS[strat]
        agg_new = agg_by_strategy[strat]
        agg_g_new = agg_gallery_by_strategy.get(strat, {})
        agg = dict(agg_new)
        agg_g = dict(agg_g_new)
        # per-strategy meta view (flatten this strategy's label/coverage fields onto base).
        smeta = {**meta, **per_strat_meta[strat]}
        if args.merge_existing and (sdir / "metrics.json").is_file():
            existing = json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
            agg = {**existing.get("agg_test", {}), **agg_new}
            agg_g = {**existing.get("agg_gallery", {}), **agg_g_new}
            old_meta = existing.get("meta", {})
            smeta = {**old_meta, **smeta}
            old_methods = set(existing.get("agg_test", {}))
            print(f"  [merge-existing] {STAGE_NAME[strat]} kept {len(old_methods)} old methods, "
                  f"added/updated {len(agg_new)} methods")
        agg_by_strategy[strat] = agg
        agg_gallery_by_strategy[strat] = agg_g
        compact_meta_by_strategy[strat] = smeta
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "method_comparison.md").write_text(
            build_report_single(agg, agg_g, strat, smeta), encoding="utf-8")
        (sdir / "metrics.json").write_text(

            json.dumps({"meta": smeta, "strategy": strat, "stage": STAGE_NAME[strat],

                        "agg_test": _jsonable(agg), "agg_gallery": _jsonable(agg_g)},

                       ensure_ascii=False, indent=2), encoding="utf-8")

        mr = sdir / "methods_results"
        mr.mkdir(parents=True, exist_ok=True)
        gallery_scores_for_export = {
            **baseline_gallery_scores,
            **trained_full_scores.get(strat, {}),
        }
        write_methods_results(mr, strat, agg_new, agg_g_new, results["strategies"][strat]["per_seed"],
                              test_scores_by_strategy.get(strat, {}),
                              gallery_scores_for_export, baseline_test_scores,
                              test_set, gt_test, joint_gt,
                              gallery_paths, gt_gallery, joint_gt_gallery)
    (report_root / "metric_analysis.md").write_text(
        build_detailed_task_report(agg_by_strategy, agg_gallery_by_strategy, compact_meta_by_strategy),
        encoding="utf-8")
    print(f"\nReports + methods_results written to: "
          f"{', '.join(STAGE_NAME[s] for s in strategies)}/ and task root.")


    if method_times:

        print("\n--- per-method timing (mean s/seed-run over all runs) ---")

        ranked = sorted(method_times.items(), key=lambda kv: -np.mean(kv[1]))

        for m, ts in ranked:

            print(f"  {m:<26} {np.mean(ts):6.1f}s/run  (kind={TRAINED_METHODS[m]}, n={len(ts)})")



    # --- top-50 export, reusing the first-seed models' full-library scores (no retrain)

    if not args.no_top50:

        generate_top50(strategies, methods_to_run, train_set, vqa_by_strat, emb, emb_norm, patches,
                       p2i, text_q, text_dim, input_dim, query_idx, gallery_paths, gallery_idx,
                       gt_by_attr, trained_full_scores=trained_full_scores,
                       top50_chosen=top50_chosen, top50_seed=TOP50_SEED,
                       emb_methods=emb_methods, backbone=args.backbone)


    print("\nAll done.")







def compute_vqa_direct_baseline(test_set, vqa_map, gt_test, joint_gt, vqa_prob=None):

    """Full-VQA upper bound: directly use full-VQA labels as predictions."""

    sba = {}

    for attr in ATTRS:

        if vqa_prob is not None:

            sba[attr] = np.array(

                [float(vqa_prob.get(rp, {}).get(attr) or 0.0) for rp in test_set],

                dtype=np.float32)

        else:

            sba[attr] = np.array(

                [1.0 if vqa_map.get(rp, {}).get(attr) == 1 else 0.0 for rp in test_set],

                dtype=np.float32)

    joint = joint_score_from(sba)
    res = eval_method_on_test(sba, joint, gt_test, joint_gt, is_prob=True)
    raw = {k: v.tolist() for k, v in scores_by_ranking_key(sba, joint).items()}
    return _single_to_agg(res), raw




_TEXT_FEATS_BY_BACKBONE = {}
_PROMPT_ENSEMBLE_FEATS_BY_BACKBONE = {}
_SIGLIP_TEXT_MODEL_AND_TOKENIZER = None
_CLIP_TEXT_FEATS = None
_CLIP_PROMPT_ENSEMBLE_FEATS = None
_CLIP_BINARY_TEXT_FEATS = None




def clip_text_feats():
    """CLIP projected text embeddings (512-d, same space as emb) for the 2 attrs.

    Loaded once and cached (the model load is slow / network-flaky)."""

    global _CLIP_TEXT_FEATS
    if _CLIP_TEXT_FEATS is None:
        from transformers import CLIPModel, CLIPTokenizerFast
        try:
            model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).eval()
            tok = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
        except Exception:
            model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval()
            tok = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
        inp = tok(list(ATTRS), return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():

            tfeat = model.text_projection(model.text_model(**inp).pooler_output).cpu().numpy()

        _CLIP_TEXT_FEATS = l2norm(tfeat)

    return _CLIP_TEXT_FEATS


def _load_siglip_text_model():
    global _SIGLIP_TEXT_MODEL_AND_TOKENIZER
    if _SIGLIP_TEXT_MODEL_AND_TOKENIZER is not None:
        return _SIGLIP_TEXT_MODEL_AND_TOKENIZER
    from transformers import AutoTokenizer, SiglipModel
    model_name = BACKBONE_TEXT_MODEL["siglip"]
    try:
        model = SiglipModel.from_pretrained(model_name, local_files_only=True).eval()
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        model = SiglipModel.from_pretrained(model_name).eval()
        tok = AutoTokenizer.from_pretrained(model_name)
    _SIGLIP_TEXT_MODEL_AND_TOKENIZER = (model, tok)
    return _SIGLIP_TEXT_MODEL_AND_TOKENIZER


def _siglip_text_features(texts: list[str]) -> np.ndarray:
    model, tok = _load_siglip_text_model()
    max_len = int(model.config.text_config.max_position_embeddings)
    inp = tok(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    with torch.no_grad():
        feats = model.get_text_features(**inp)
        if not isinstance(feats, torch.Tensor):
            text_embeds = getattr(feats, "text_embeds", None)
            feats = text_embeds if text_embeds is not None else getattr(feats, "pooler_output", None)
            if feats is None:
                raise TypeError("SigLIP get_text_features returned no text_embeds/pooler_output")
    return l2norm(feats.cpu().numpy().astype(np.float32))


def text_feats_for_backbone(backbone: str) -> np.ndarray:
    """Projected text embeddings in the same space as the active image backbone."""
    if backbone == "clip":
        return clip_text_feats()
    if backbone != "siglip":
        raise ValueError(f"text-fusion baselines are not defined for backbone {backbone!r}")
    cached = _TEXT_FEATS_BY_BACKBONE.get(backbone)
    if cached is None:
        cached = _siglip_text_features(list(ATTRS))
        _TEXT_FEATS_BY_BACKBONE[backbone] = cached
    return cached


def _binary_prompt_variants(attr, present: bool):
    if present:
        return [
            f"a photo of {attr}",
            f"an image showing {attr}",
            f"a close-up photo showing {attr}",
            f"a photo where the main visual evidence is {attr}",
        ]
    return [
        f"a photo without {attr}",
        f"an image not showing {attr}",
        f"a photo where {attr} is absent",
        f"a photo lacking the visual attribute {attr}",
    ]


def clip_binary_text_feats(attr):
    """CLIP text classifier weights for binary absent/present attribute scoring.

    This adapts CLIP-Adapter/Tip-Adapter's multi-class text classifier to the
    harness' per-attribute binary setting. Row order is [absent, present].
    """
    global _CLIP_BINARY_TEXT_FEATS
    if _CLIP_BINARY_TEXT_FEATS is None:
        from transformers import CLIPModel, CLIPTokenizerFast
        try:
            model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).eval()
            tok = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
        except Exception:
            model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval()
            tok = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
        feats = {}
        with torch.no_grad():
            for a in ATTRS:
                rows = []
                for present in (False, True):
                    inp = tok(_binary_prompt_variants(a, present),
                              return_tensors="pt", padding=True, truncation=True)
                    tfeat = model.text_projection(model.text_model(**inp).pooler_output).cpu().numpy()
                    rows.append(l2norm(l2norm(tfeat).mean(axis=0, keepdims=True)).squeeze())
                feats[a] = np.stack(rows, axis=0).astype(np.float32)
        _CLIP_BINARY_TEXT_FEATS = feats
    return _CLIP_BINARY_TEXT_FEATS[attr]


def _combo_phrase(attrs):
    return " and ".join(str(a) for a in attrs)


def _prompt_variants(attrs):
    phrase = _combo_phrase(attrs)
    return [
        f"a photo of {phrase}",
        f"an image of {phrase}",
        f"a close-up photo showing {phrase}",
        f"a photo where the main visual evidence is {phrase}",
    ]


def clip_prompt_ensemble_feats():
    """Prompt-ensemble CLIP text features for each reported ranking key."""
    global _CLIP_PROMPT_ENSEMBLE_FEATS
    if _CLIP_PROMPT_ENSEMBLE_FEATS is not None:
        return _CLIP_PROMPT_ENSEMBLE_FEATS

    from transformers import CLIPModel, CLIPTokenizerFast
    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).eval()
        tok = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
    except Exception:
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval()
        tok = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)

    feats = {}
    with torch.no_grad():
        for key, info in RANKING_SPEC.items():
            attrs = tuple(info.get("attrs") or info["gt"][1:])
            inp = tok(_prompt_variants(attrs), return_tensors="pt", padding=True, truncation=True)
            tfeat = model.text_projection(model.text_model(**inp).pooler_output).cpu().numpy()
            feats[key] = l2norm(l2norm(tfeat).mean(axis=0, keepdims=True)).squeeze()
    _CLIP_PROMPT_ENSEMBLE_FEATS = feats
    return feats


def prompt_ensemble_feats_for_backbone(backbone: str) -> dict[str, np.ndarray]:
    """Prompt-ensemble text features for each ranking key in the active backbone space."""
    if backbone == "clip":
        return clip_prompt_ensemble_feats()
    if backbone != "siglip":
        raise ValueError(f"prompt text baselines are not defined for backbone {backbone!r}")
    cached = _PROMPT_ENSEMBLE_FEATS_BY_BACKBONE.get(backbone)
    if cached is not None:
        return cached

    feats = {}
    for key, info in RANKING_SPEC.items():
        attrs = tuple(info.get("attrs") or info["gt"][1:])
        tfeat = _siglip_text_features(_prompt_variants(attrs))
        feats[key] = l2norm(l2norm(tfeat).mean(axis=0, keepdims=True)).squeeze()
    _PROMPT_ENSEMBLE_FEATS_BY_BACKBONE[backbone] = feats
    return feats


def _zscore(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-6)


def _eval_ranking_scores(scores_by_key, gt_test):
    metrics = {}
    for key, info in RANKING_SPEC.items():
        attrs = tuple(info.get("attrs") or info["gt"][1:])
        yt = gt_test[attrs[0]] if len(attrs) == 1 else combo_gt_from(gt_test, attrs)
        score = np.asarray(scores_by_key[key], dtype=np.float32)
        metrics[key] = rank_metrics(score, yt)
        metrics[key].update(best_f1_clf(score, yt))
        metrics[key].update(best_clf_metrics(score, yt))
    return _single_to_agg(metrics)



def compute_embedding_baselines(
    emb_norm, test_idx, query_idx, text_q_unused, gt_test, joint_gt,
    emb_methods=None, backbone="clip",
):
    """Seed-independent baselines in the active CLIP or SigLIP embedding space.

    Query MaxSim uses the active backbone's image embeddings. Text Prompt
    Ensemble and both image/text fusions use the matching CLIP or SigLIP text
    encoder selected by ``backbone``; no CLIP feature is mixed into a SigLIP run.
    """
    emb_methods = set(emb_methods or EMB_METHODS)
    proto = l2norm(emb_norm[query_idx].mean(axis=0, keepdims=True)).squeeze()
    img_scores = emb_norm[test_idx] @ proto

    out = {}
    # Image Prototype: single score for all attrs; joint uses the same prototype score.
    raw_scores = {}
    if "image_prototype" in emb_methods:
        ip = {}
        for key, info in RANKING_SPEC.items():
            attrs = tuple(info.get("attrs") or info["gt"][1:])
            yt = gt_test[attrs[0]] if len(attrs) == 1 else combo_gt_from(gt_test, attrs)
            ip[key] = rank_metrics(img_scores, yt)
            ip[key].update(best_f1_clf(img_scores, yt))
            ip[key].update(best_clf_metrics(img_scores, yt))
        out["image_prototype"] = _single_to_agg(ip)
        raw_scores["image_prototype"] = {k: img_scores.tolist() for k in RANKING_SPEC}

    # Query MaxSim: fair multi-query adaptation of single-reference retrieval.
    if "query_maxsim" in emb_methods:
        qms_scores = np.max(emb_norm[test_idx] @ emb_norm[query_idx].T, axis=1)
        raw_scores["query_maxsim"] = {k: qms_scores.tolist() for k in RANKING_SPEC}
        out["query_maxsim"] = _eval_ranking_scores(raw_scores["query_maxsim"], gt_test)

    if not (TEXT_EMB_METHODS & emb_methods):
        return out, raw_scores

    tfeat = text_feats_for_backbone(backbone)

    # Img+Text Fusion
    if "img_text_fusion" in emb_methods:
        fusion_scores = {}
        for i, attr in enumerate(ATTRS):
            comb = l2norm((0.5 * proto + 0.5 * tfeat[i]).reshape(1, -1)).squeeze()
            fusion_scores[attr] = emb_norm[test_idx] @ comb
        itf = {}
        raw_scores["img_text_fusion"] = {}
        for key, info in RANKING_SPEC.items():
            attrs = tuple(info.get("attrs") or info["gt"][1:])
            score = fusion_scores[attrs[0]] if len(attrs) == 1 else np.mean(
                np.vstack([fusion_scores[a] for a in attrs]), axis=0)
            yt = gt_test[attrs[0]] if len(attrs) == 1 else combo_gt_from(gt_test, attrs)
            itf[key] = rank_metrics(score, yt)
            itf[key].update(best_f1_clf(score, yt))
            itf[key].update(best_clf_metrics(score, yt))
            raw_scores["img_text_fusion"][key] = score.tolist()
        out["img_text_fusion"] = _single_to_agg(itf)

    # Text Prompt Ensemble: zero-shot CLIP/SigLIP retrieval with the text encoder
    # paired to the active image backbone.
    prompt_feats = prompt_ensemble_feats_for_backbone(backbone)
    tpe_scores = {key: (emb_norm[test_idx] @ feat).astype(np.float32)
                  for key, feat in prompt_feats.items()}
    if "text_prompt_ensemble" in emb_methods:
        raw_scores["text_prompt_ensemble"] = {k: v.tolist() for k, v in tpe_scores.items()}
        out["text_prompt_ensemble"] = _eval_ranking_scores(tpe_scores, gt_test)

    # Z-score Img+Text Fusion: normalized late fusion to avoid scale bias between
    # visual prototypes and text prompts while keeping the baseline training-free.
    if "zscore_img_text_fusion" in emb_methods:
        zit_scores = {
            key: (0.5 * _zscore(img_scores) + 0.5 * _zscore(tpe_scores[key])).astype(np.float32)
            for key in RANKING_SPEC
        }
        raw_scores["zscore_img_text_fusion"] = {k: v.tolist() for k, v in zit_scores.items()}
        out["zscore_img_text_fusion"] = _eval_ranking_scores(zit_scores, gt_test)
    return out, raw_scores


def _single_to_agg(metric_dict):

    """Wrap single-value metrics as {mean:v, std:0, n:1} to match aggregate() output."""

    def wrap(d):

        r = {}

        for k, v in d.items():

            if isinstance(v, dict):

                r[k] = wrap(v)

            else:

                r[k] = {"mean": float(v) if v == v else float("nan"),

                        "std": 0.0, "n": 1 if v == v else 0}

        return r

    return wrap(metric_dict)





def _jsonable(agg):

    return agg







def generate_top50(strategies, methods_to_run, train_set, vqa_by_strat, emb, emb_norm, patches,
                   p2i, text_q, text_dim, input_dim, query_idx, gallery_paths, gallery_idx,
                   gt_by_attr, trained_full_scores=None, top50_chosen=None, top50_seed=42,
                   emb_methods=None, backbone="clip"):
    """Export top-50 retrieval images over the gallery (full DB - query)."""
    emb_methods = list(emb_methods or EMB_METHODS)
    emb_method_set = set(emb_methods)
    reuse = trained_full_scores or {}
    chosen_map = top50_chosen or {}

    SEED = top50_seed

    stage_for = STAGE_DIRS

    candidates = gallery_paths
    baseline_strat = "random" if "random" in strategies else strategies[0]
    print(f"\n{'#'*70}\n# TOP-50 EXPORT (seed={SEED} models, gallery={len(gallery_paths)})\n{'#'*70}")


    # embedding baseline scores over the gallery (zero-shot, no training)

    proto = l2norm(emb_norm[query_idx].mean(axis=0, keepdims=True)).squeeze()

    full = gallery_idx
    ip_scores = emb_norm[full] @ proto
    qms_scores = np.max(emb_norm[full] @ emb_norm[query_idx].T, axis=1)
    fuse = {}
    tpe = {}
    zit = {}
    if TEXT_EMB_METHODS & emb_method_set:
        tfeat = text_feats_for_backbone(backbone)
        for i, attr in enumerate(ATTRS):
            comb = l2norm((0.5 * proto + 0.5 * tfeat[i]).reshape(1, -1)).squeeze()
            fuse[attr] = emb_norm[full] @ comb
        prompt_feats = prompt_ensemble_feats_for_backbone(backbone)
        tpe = {key: (emb_norm[full] @ feat).astype(np.float32)
               for key, feat in prompt_feats.items()}
        zit = {key: (0.5 * _zscore(ip_scores) + 0.5 * _zscore(tpe[key])).astype(np.float32)
               for key in RANKING_SPEC}


    # Materialize full-library features only if we must retrain some method (fallback path).

    any_retrain = any(not (strat in reuse and m in reuse[strat])

                      for strat in strategies for m in methods_to_run)

    full_emb = emb[full] if any_retrain else None

    full_patches = ((patches, full) if (any_retrain and patches is not None) else None)


    for strat in strategies:

        vqa_map = vqa_by_strat[strat][0]

        # supervision subset (for leakage-free exclusion + vqa_gt/). Prefer the exact

        # subset used at top50_seed in the metrics run; else recompute deterministically.

        if strat in chosen_map:

            chosen = list(chosen_map[strat])

        else:

            chosen, _, _ = build_subset(strat, SEED, train_set, vqa_map, emb_norm, p2i, query_idx,

                                        sup_size=SUP_SIZE, label_size=LABEL_SIZE)

        sub_set = set(chosen)

        # per-ranking VQA supervision positives (kept out of ranking, shown in vqa_gt/)
        vqa_gt_by_key = {k: [] for k in RANKING_SPEC}
        for rp in chosen:
            lab = vqa_map.get(rp, {})
            for key, info in RANKING_SPEC.items():
                attrs = tuple(info.get("attrs") or info["gt"][1:])
                if all(lab.get(attr) == 1 for attr in attrs):
                    vqa_gt_by_key[key].append(rp)
        counts = ", ".join(f"{k}={len(vqa_gt_by_key[k])}" for k in RANKING_SPEC)
        print(f"  [top50] {strat} supervision={len(sub_set)}  vqa_gt: {counts}")


        scores_by_method = {}
        # Embedding baselines are independent of VQA strategy. Export them only once
        # (normally under onestage/random) to avoid duplicate image folders.
        if strat == baseline_strat:
            if "image_prototype" in emb_method_set:
                scores_by_method["image_prototype"] = {k: ip_scores for k in RANKING_SPEC}
            if "query_maxsim" in emb_method_set:
                scores_by_method["query_maxsim"] = {k: qms_scores for k in RANKING_SPEC}
            if "img_text_fusion" in emb_method_set:
                scores_by_method["img_text_fusion"] = {}
                for key, info in RANKING_SPEC.items():
                    attrs = tuple(info.get("attrs") or info["gt"][1:])
                    scores_by_method["img_text_fusion"][key] = (
                        fuse[attrs[0]] if len(attrs) == 1
                        else np.mean(np.vstack([fuse[a] for a in attrs]), axis=0)
                    )
            if "text_prompt_ensemble" in emb_method_set:
                scores_by_method["text_prompt_ensemble"] = tpe
            if "zscore_img_text_fusion" in emb_method_set:
                scores_by_method["zscore_img_text_fusion"] = zit
        else:
            for method in emb_methods:
                stale = stage_for[strat] / "top_50" / method
                if stale.exists():
                    shutil.rmtree(stale)


        # lazily build the training ctx only if a fallback retrain is needed this strategy

        ctx = None

        model_cache = {}
        for method in methods_to_run:
            if strat in reuse and method in reuse[strat]:

                fs = reuse[strat][method]

                attr_scores = {attr: np.asarray(fs[attr]) for attr in ATTRS}

                if JOINT_KEY in fs:

                    attr_scores[JOINT_KEY] = np.asarray(fs[JOINT_KEY])

            else:

                if ctx is None:

                    U_idx = np.array([p2i[rp] for rp in train_set if rp not in sub_set], dtype=np.int64)

                    u_rng = np.random.default_rng(1000 + SEED)

                    U_sample = (U_idx if len(U_idx) <= U_CAP
                                else U_idx[u_rng.choice(len(U_idx), U_CAP, replace=False)])
                    ctx = {"emb": emb, "patches": patches, "p2i": p2i, "text_q": text_q,
                           "input_dim": input_dim, "text_dim": text_dim, "patch_dim": patch_dim,
                           "U_sample": U_sample, "U_idx": U_idx}
                labeled = [{"image": rp, **vqa_map[rp]} for rp in chosen]

                print(f"\n  [top50] {strat}/{method} (retrain fallback)")

                models = train_method_reusing_base(
                    method, TRAINED_METHODS[method], labeled, chosen,
                    ctx["U_idx"], ctx, SEED, model_cache,
                )
                calibrator = None

                if method in CALIBRATED_METHOD_BASE:

                    chosen_idx = np.array([p2i[rp] for rp in chosen], dtype=np.int64)

                    chosen_emb = emb[chosen_idx]

                    chosen_patches = np.asarray(patches[chosen_idx]) if patches is not None else None

                    sba_train = score_attr_models(models, chosen_emb, chosen_patches, text_q)

                    calibrator = fit_joint_calibrator(sba_train, labeled, SEED)

                attr_scores = score_attr_models(models, full_emb, full_patches, text_q)

                attr_scores[JOINT_KEY] = joint_score(attr_scores, calibrator, method=method)
            scores_by_method[method] = scores_by_ranking_key(

                {a: attr_scores[a] for a in ATTRS}, attr_scores.get(JOINT_KEY))



        export_all_top50(stage_for[strat], scores_by_method,
                         candidates, gt_by_attr, ADAPTER.raw_images_dir, RANKING_SPEC,
                         topn=50, exclude_set=sub_set, vqa_gt_by_key=vqa_gt_by_key,
                         no_exclude_methods=set(emb_methods))




if __name__ == "__main__":

    main()
