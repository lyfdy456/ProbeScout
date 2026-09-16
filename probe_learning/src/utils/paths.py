"""Centralized paths and constants for the labeling-methods project."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Derive the experiment root from this file's location so the project is
# portable across machines/OSes (was a hardcoded macOS absolute path).
# Override with the EXPERIMENT_ROOT environment variable if needed.
EXPERIMENT_ROOT = Path(os.environ.get("EXPERIMENT_ROOT", PROJECT_DIR.parent))
TASKS_DIR = EXPERIMENT_ROOT / "dataset" / "tasks"

# ===================================================================
# Task layout (refactored 2026-07):
#   tasks/<dataset>/<task>/{task.json, query_ids.json, attributes.txt,
#                          supervision, evaluation}
# Raw and processed dataset artifacts live under dataset/raw, never in tasks.
# ===================================================================
DATASET_GROUPS = {
    "cars": "cars",
    "sun": "sun",
    "hico": "hico",
    "cub": "cub",
    "awa2": "awa2",
    "celeba": "celeba",
}

# Per-task folder name overrides (when the task token differs from the desired
# on-disk folder, e.g. SUN strips the redundant "sun_" prefix for cleaner names).
TASK_FOLDER_OVERRIDES = {
    "task_sun_pedestrians_on_zebra_crossing": "task_pedestrians_on_zebra_crossing",
}


def dataset_group(dataset: str) -> str:
    """Map a dataset key to its on-disk group folder name."""
    return DATASET_GROUPS.get(dataset, dataset)


def task_folder_name(task: str) -> str:
    """Resolve the on-disk per-task folder name for a (possibly legacy) task token."""
    return TASK_FOLDER_OVERRIDES.get(task, task)


def task_root(dataset: str, task: str) -> Path:
    """New per-task root: tasks/<group>/<task_folder>/."""
    return TASKS_DIR / dataset_group(dataset) / task_folder_name(task)


def database_dir(dataset: str) -> Path:
    """Return the canonical raw-image root for a dataset."""
    roots = {
        "cars": EXPERIMENT_ROOT / "dataset" / "raw" / "stanford_cars" / "images",
        "sun": EXPERIMENT_ROOT / "dataset" / "raw" / "SUN" / "images",
        "hico": EXPERIMENT_ROOT / "dataset" / "raw" / "HICO" / "images",
        "cub": EXPERIMENT_ROOT / "dataset" / "raw" / "CUB_200_2011" / "images",
        "awa2": EXPERIMENT_ROOT / "dataset" / "raw" / "AwA2" / "JPEGImages",
        "celeba": EXPERIMENT_ROOT / "dataset" / "raw" / "CelebA" / "img_celeba",
    }
    return roots[dataset]

CUB_ROOT = EXPERIMENT_ROOT / "dataset" / "raw" / "CUB_200_2011"
PROCESSED_DIR = CUB_ROOT / "processed"

MODEL_DIR = PROJECT_DIR / "models"

EMBEDDING_FILES: dict[str, tuple[str, int]] = {
    "clip": ("clip_embedding.npy", 512),
    "dinov2": ("dinov2_embedding.npy", 768),
    "siglip": ("siglip_embedding.npy", 768),
}

DEFAULT_TASK = "task_blue_shortbeak_perched_birds"
DEFAULT_ATTRIBUTES = ["dominant_vivid_blue_plumage", "perched_bird_pose", "short_conical_beak"]


SUN_ROOT = EXPERIMENT_ROOT / "dataset" / "raw" / "SUN"
SUN_PROCESSED_DIR = SUN_ROOT / "processed"

# ===================================================================
# HICO dataset (Human-Object Interaction; image-level HOI GT in anno.mat)
# ===================================================================
HICO_ROOT = EXPERIMENT_ROOT / "dataset" / "raw" / "HICO"
HICO_PROCESSED_DIR = HICO_ROOT / "processed"
HICO_IMAGES_DIR = HICO_ROOT / "images"
HICO_ANNO_MAT = HICO_ROOT / "anno.mat"

HICO_EMBEDDING_FILES: dict[str, tuple[str, int]] = {
    "clip": ("clip_embedding.npy", 512),
}

SUN_EMBEDDING_FILES: dict[str, tuple[str, int]] = {
    "clip": ("clip_embedding.npy", 512),
}

# ===================================================================
# AwA2 (Animals with Attributes 2; class-level 85-predicate GT)
# ===================================================================
AWA2_ROOT = EXPERIMENT_ROOT / "dataset" / "raw" / "AwA2"
AWA2_PROCESSED_DIR = AWA2_ROOT / "processed"   # full-dataset clip_embedding/records/predicates.json
AWA2_IMAGES_DIR = AWA2_ROOT / "JPEGImages"     # JPEGImages/<class>/<basename>.jpg

# ===================================================================
# CelebA (face attributes; official binary GT in Anno/list_attr_celeba.txt)
# ===================================================================
CELEBA_ROOT = EXPERIMENT_ROOT / "dataset" / "raw" / "CelebA"
CELEBA_PROCESSED_DIR = CELEBA_ROOT / "processed"
CELEBA_IMAGES_DIR = CELEBA_ROOT / "img_celeba"

SNOW_ROAD_TASK = "task_snow_road"
SNOW_ROAD_ATTRIBUTES = [
    "Dark asphalt road",
    "Ground covered in white snow",
    "Visible tire tracks or wet slush on road",
    "Overcast gray lighting",
    "Monochromatic or desaturated color palette",
]

MANMADE_DAMP_TASK = "task_manmade_damp"
MANMADE_DAMP_ATTRIBUTES = [
    "bright natural daylight illumination",
    "concrete structural material",
    "blue sky background",
    "large-scale water infrastructure",
    "body of water setting",
]

SNOW_ROAD_TWO_STAGE_TASK = "task_snow_road_two_stage"

ZEBRA_CROSSING_TASK = "task_sun_pedestrians_on_zebra_crossing"
ZEBRA_CROSSING_ATTRIBUTES = [
    "person actively walking across street on marked pedestrian crosswalk",
    "thick white or rainbow-colored parallel stripes marking pedestrian crosswalk",
    "grey asphalt road surface with visible texture and minor cracks",
]

ZEBRA_CROSSING_TWO_STAGE_TASK = "task_sun_pedestrians_on_zebra_crossing_two_stage"

ZEBRA_TWO_STAGE_1PCT_TASK = "task_zebra_two_stage_1pct"
ZEBRA_TWO_STAGE_2PCT_TASK = "task_zebra_two_stage_2pct"
ZEBRA_TWO_STAGE_3PCT_TASK = "task_zebra_two_stage_3pct"

FOREST_ROAD_TREE_CORRIDOR_TASK = "task_forest_road_tree_corridor"
FOREST_ROAD_TREE_CORRIDOR_ATTRIBUTES = [
    "road",
    "trees",
    "forest road",
    "road surface texture variation (asphalt, dirt, snow)",
]

SUN_TASK_REGISTRY: dict[str, dict] = {
    SNOW_ROAD_TASK: {
        "attributes": SNOW_ROAD_ATTRIBUTES,
        "raw_jsonl": "20260512_175748_sun_dataset_tongyi_snow_roads_results.jsonl",
        "sun_mapping": {
            "Dark asphalt road": {"sun_idx": 47, "sun_name": "asphalt", "inverse": False},
            "Ground covered in white snow": {"sun_idx": 70, "sun_name": "snow", "inverse": False},
            "Visible tire tracks or wet slush on road": {"sun_idx": 81, "sun_name": "moist/ damp", "inverse": False},
            "Overcast gray lighting": {"sun_idx": 75, "sun_name": "direct sun/sunny", "inverse": True},
        },
        "query_indices": [5520, 10660, 8573, 9435],
    },
    MANMADE_DAMP_TASK: {
        "attributes": MANMADE_DAMP_ATTRIBUTES,
        "raw_jsonl": "20260515_101942_sun_dataset_tongyi_water_infra_results.jsonl",
        "sun_mapping": {
            "concrete structural material": {"sun_idx": 53, "sun_name": "concrete", "inverse": False},
            "large-scale water infrastructure": {"sun_idx": 88, "sun_name": "man-made", "inverse": False},
            "bright natural daylight illumination": {"sun_idx": 75, "sun_name": "direct sun/sunny", "inverse": False},
            "body of water setting": {"sun_idx": 68, "sun_name": "still water", "inverse": False},
        },
        "query_indices": [11323, 13983, 6348, 4014],
    },
    SNOW_ROAD_TWO_STAGE_TASK: {
        "attributes": SNOW_ROAD_ATTRIBUTES,
        "raw_jsonl": "20260521_135105_sun_snow_road_5pct_qwen-vl-max_results.jsonl",
        "sun_mapping": {
            "Dark asphalt road": {"sun_idx": 47, "sun_name": "asphalt", "inverse": False},
            "Ground covered in white snow": {"sun_idx": 70, "sun_name": "snow", "inverse": False},
            "Visible tire tracks or wet slush on road": {"sun_idx": 81, "sun_name": "moist/ damp", "inverse": False},
            "Overcast gray lighting": {"sun_idx": 75, "sun_name": "direct sun/sunny", "inverse": True},
        },
        "query_indices": [5520, 10660, 8573, 9435],
    },
    ZEBRA_CROSSING_TASK: {
        "attributes": ZEBRA_CROSSING_ATTRIBUTES,
        "raw_jsonl": "20260519_133748_pedestrian_zebra_crossroadsun_retrieval_10pct_qwen-vl-max_results.jsonl",
        "sun_mapping": {
            "grey asphalt road surface with visible texture and minor cracks": {
                "sun_idx": 47, "sun_name": "asphalt", "inverse": False,
            },
            "thick white or rainbow-colored parallel stripes marking pedestrian crosswalk": {
                "sun_idx": 48, "sun_name": "pavement", "inverse": False,
            },
            "person actively walking across street on marked pedestrian crosswalk": {
                "sun_idx": 18, "sun_name": "socializing", "inverse": False,
            },
        },
        "query_indices": [12034, 3907, 3908, 3909, 3919],
    },
    ZEBRA_CROSSING_TWO_STAGE_TASK: {
        "attributes": ZEBRA_CROSSING_ATTRIBUTES,
        "raw_jsonl": "20260519_235051_sun_retrieval_top5pct_qwen-vl-max_results.jsonl",
        "sun_mapping": {
            "grey asphalt road surface with visible texture and minor cracks": {
                "sun_idx": 47, "sun_name": "asphalt", "inverse": False,
            },
            "thick white or rainbow-colored parallel stripes marking pedestrian crosswalk": {
                "sun_idx": 48, "sun_name": "pavement", "inverse": False,
            },
        },
        "query_indices": [12034, 3907, 3908, 3909, 3919],
    },
    ZEBRA_TWO_STAGE_1PCT_TASK: {
        "attributes": ZEBRA_CROSSING_ATTRIBUTES,
        "raw_jsonl": "20260530_174355_sun_full_qwen-vl-max_results.jsonl",
        "sun_mapping": {
            "grey asphalt road surface with visible texture and minor cracks": {
                "sun_idx": 47, "sun_name": "asphalt", "inverse": False,
            },
            "thick white or rainbow-colored parallel stripes marking pedestrian crosswalk": {
                "sun_idx": 48, "sun_name": "pavement", "inverse": False,
            },
        },
        "query_indices": [12034, 3907, 3908, 3909, 3919],
    },
    ZEBRA_TWO_STAGE_2PCT_TASK: {
        "attributes": ZEBRA_CROSSING_ATTRIBUTES,
        "raw_jsonl": "20260530_174355_sun_full_qwen-vl-max_results.jsonl",
        "sun_mapping": {
            "grey asphalt road surface with visible texture and minor cracks": {
                "sun_idx": 47, "sun_name": "asphalt", "inverse": False,
            },
            "thick white or rainbow-colored parallel stripes marking pedestrian crosswalk": {
                "sun_idx": 48, "sun_name": "pavement", "inverse": False,
            },
        },
        "query_indices": [12034, 3907, 3908, 3909, 3919],
    },
    ZEBRA_TWO_STAGE_3PCT_TASK: {
        "attributes": ZEBRA_CROSSING_ATTRIBUTES,
        "raw_jsonl": "20260530_174355_sun_full_qwen-vl-max_results.jsonl",
        "sun_mapping": {
            "grey asphalt road surface with visible texture and minor cracks": {
                "sun_idx": 47, "sun_name": "asphalt", "inverse": False,
            },
            "thick white or rainbow-colored parallel stripes marking pedestrian crosswalk": {
                "sun_idx": 48, "sun_name": "pavement", "inverse": False,
            },
        },
        "query_indices": [12034, 3907, 3908, 3909, 3919],
    },
    FOREST_ROAD_TREE_CORRIDOR_TASK: {
        "attributes": FOREST_ROAD_TREE_CORRIDOR_ATTRIBUTES,
        "raw_jsonl": "forest_road_tree_corridor_vqa.jsonl",
        "sun_mapping": {
            "road": {
                "sun_category": "f/forest_road", "sun_name": "forest_road_category", "inverse": False,
            },
            "trees": {
                "sun_idx": 40, "sun_name": "trees", "inverse": False,
            },
            "forest road": {
                "sun_category": "f/forest_road", "sun_name": "forest_road", "inverse": False,
            },
            "road surface texture variation (asphalt, dirt, snow)": {
                "sun_any": [
                    {"sun_idx": 47, "sun_name": "asphalt"},
                    {"sun_idx": 62, "sun_name": "dirt/soil"},
                    {"sun_idx": 70, "sun_name": "snow"},
                ],
                "sun_name": "asphalt_or_dirt_or_snow",
                "inverse": False,
            },
        },
        "query_indices": [5520, 5521, 5522, 5523, 5526, 5529, 5535],
        "joint_label": "forest road with tree-lined corridor",
    },
}


def _load_sun_task_sidecar(task: str) -> dict | None:
    """Load a per-task SUN config from task.json."""
    import json as _json

    task_dir = task_root("sun", task)
    meta = task_dir / "task.json"
    if not meta.is_file():
        return None
    cfg = _json.loads(meta.read_text(encoding="utf-8"))
    if not cfg.get("raw_jsonl"):
        qa = task_dir / "qa"
        cands = sorted(qa.glob("*_results.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True) if qa.is_dir() else []
        if cands:
            cfg["raw_jsonl"] = cands[0].name
    cfg.setdefault("raw_jsonl", "")
    return cfg


def get_sun_task(task: str) -> dict:
    """Retrieve SUN task config from registry or per-task sidecar."""
    if task in SUN_TASK_REGISTRY:
        return SUN_TASK_REGISTRY[task]
    for registry_task, folder_name in TASK_FOLDER_OVERRIDES.items():
        if task == folder_name and registry_task in SUN_TASK_REGISTRY:
            return SUN_TASK_REGISTRY[registry_task]
    side = _load_sun_task_sidecar(task)
    if side is not None:
        return side
    for suffix in ("_two_stage_test", "_two_stage", "_test"):
        if task.endswith(suffix):
            base = task[: -len(suffix)]
            if base in SUN_TASK_REGISTRY:
                return SUN_TASK_REGISTRY[base]
            side = _load_sun_task_sidecar(base)
            if side is not None:
                return side
    raise KeyError(task)


# ===================================================================
# Stanford Cars dataset
# ===================================================================
CARS_ROOT = EXPERIMENT_ROOT / "dataset" / "raw" / "stanford_cars"
CARS_PROCESSED_DIR = CARS_ROOT / "processed"
CARS_IMAGE_ROOT = CARS_ROOT / "images"
CARS_ANNOS_MAT = CARS_ROOT / "cars_annos.mat"

CARS_EMBEDDING_FILES: dict[str, tuple[str, int]] = {
    "clip": ("clip_embedding.npy", 512),
}

BMW_SEDAN_TASK = "task_bmw_sedan"
BMW_SEDAN_TWO_STAGE_TASK = "task_bmw_sedan_two_stage"
BMW_SEDAN_ATTRIBUTES = [
    "BMW kidney grille",
    "sedan body shape",
]
BMW_SEDAN_QUERY_FILES = [
    "002057.jpg", "002065.jpg", "002289.jpg", "002806.jpg", "002844.jpg",
]
_BMW_SEDAN_RAW_JSONL = "20260615_022349_candidate_images_qwen-vl-max_results.jsonl"
_BMW_CARS_MAPPING = {
    "BMW kidney grille": {"gt_type": "is_bmw", "inverse": False},
    "sedan body shape": {"gt_type": "is_sedan", "inverse": False},
}

BMW_CONVERTIBLE_TASK = "task_bmw_convertible"
BMW_CONVERTIBLE_ATTRIBUTES = [
    "BMW brand identity",
    "Convertible body type with soft-top roof",
]
BMW_CONVERTIBLE_QUERY_FILES = [
    "002158.jpg", "002466.jpg", "002488.jpg", "002911.jpg", "003069.jpg",
]
# Filled in after the 16k VQA labeling run finishes (timestamped jsonl in qa/).
_BMW_CONVERTIBLE_RAW_JSONL = "20260617_004909_candidate_images_qwen-vl-max_results.jsonl"
_BMW_CONVERTIBLE_CARS_MAPPING = {
    "BMW brand identity": {"gt_type": "is_bmw", "inverse": False},
    "Convertible body type with soft-top roof": {"gt_type": "is_convertible", "inverse": False},
}

CARS_TASK_REGISTRY: dict[str, dict] = {
    BMW_SEDAN_TASK: {
        "attributes": BMW_SEDAN_ATTRIBUTES,
        "raw_jsonl": _BMW_SEDAN_RAW_JSONL,
        "cars_mapping": _BMW_CARS_MAPPING,
        "query_files": BMW_SEDAN_QUERY_FILES,
        "subset": {"mode": "random", "fraction": 0.05, "seed": 42},
    },
    BMW_SEDAN_TWO_STAGE_TASK: {
        "attributes": BMW_SEDAN_ATTRIBUTES,
        "raw_jsonl": _BMW_SEDAN_RAW_JSONL,
        "cars_mapping": _BMW_CARS_MAPPING,
        "query_files": BMW_SEDAN_QUERY_FILES,
        "subset": {"mode": "two_stage", "fraction": 0.05},
    },
    BMW_CONVERTIBLE_TASK: {
        "attributes": BMW_CONVERTIBLE_ATTRIBUTES,
        "raw_jsonl": _BMW_CONVERTIBLE_RAW_JSONL,
        "cars_mapping": _BMW_CONVERTIBLE_CARS_MAPPING,
        "query_files": BMW_CONVERTIBLE_QUERY_FILES,
        "subset": {"mode": "random", "fraction": 0.05, "seed": 42},
    },
    BMW_CONVERTIBLE_TASK + "_two_stage": {
        "attributes": BMW_CONVERTIBLE_ATTRIBUTES,
        "raw_jsonl": _BMW_CONVERTIBLE_RAW_JSONL,
        "cars_mapping": _BMW_CONVERTIBLE_CARS_MAPPING,
        "query_files": BMW_CONVERTIBLE_QUERY_FILES,
        "subset": {"mode": "two_stage", "fraction": 0.05},
    },
}


def _load_cars_task_sidecar(task: str) -> dict | None:
    """Load a per-task config from dataset/tasks/<task>/task_meta.json.

    Lets new Cars tasks be added WITHOUT editing CARS_TASK_REGISTRY: drop a
    task_meta.json with {attributes, cars_mapping, query_files[, raw_jsonl,
    subset]} and it is discovered here. If raw_jsonl is absent/empty, the most
    recent qa/*_results.jsonl is auto-detected.
    """
    import json as _json

    task_dir = task_root("cars", task)
    meta = task_dir / "task.json"
    if not meta.is_file():
        return None
    cfg = _json.loads(meta.read_text(encoding="utf-8"))
    if not cfg.get("raw_jsonl"):
        qa = task_dir / "qa"
        cands = sorted(qa.glob("*_results.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True) if qa.is_dir() else []
        if cands:
            cfg["raw_jsonl"] = cands[0].name
    cfg.setdefault("raw_jsonl", "")
    cfg.setdefault("subset", {"mode": "random", "fraction": 0.05, "seed": 42})
    return cfg


def get_cars_task(task: str) -> dict:
    """Retrieve Stanford Cars task config.

    Resolution order: built-in CARS_TASK_REGISTRY -> per-task task_meta.json
    sidecar (also tries the base name if `task` ends with a known test suffix).
    Raises KeyError if neither is found.
    """
    if task in CARS_TASK_REGISTRY:
        return CARS_TASK_REGISTRY[task]
    side = _load_cars_task_sidecar(task)
    if side is not None:
        return side
    # allow harness-derived test dirs to resolve to their base task
    for suffix in ("_two_stage_test", "_two_stage", "_test"):
        if task.endswith(suffix):
            base = task[: -len(suffix)]
            if base in CARS_TASK_REGISTRY:
                return CARS_TASK_REGISTRY[base]
            side = _load_cars_task_sidecar(base)
            if side is not None:
                return side
    raise KeyError(task)


# CLIP patch-token features (attention-pooled methods + precompute scripts)
PATCH_TOKEN_FILE = "clip_patch_tokens.npy"
PATCH_TOKEN_FILES: dict[str, str] = {
    "clip": "clip_patch_tokens.npy",
    "siglip": "siglip_patch_tokens.npy",
}
PATCH_TOKEN_DIM = 768
PATCH_TOKEN_NUM_PATCHES = 49

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# Backward-compat re-export. The legacy per-method model directories and
# ``*_output_file`` helpers now live in ``legacy_paths`` (used only by the
# archived runners in scripts/_legacy/, superseded by the unified harness).
# Re-exported so existing ``from src.utils.paths import <NAME>`` keep working.
# ---------------------------------------------------------------------------
from .legacy_paths import *  # noqa: E402,F401,F403
