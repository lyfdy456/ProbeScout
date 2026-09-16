"""Legacy per-method model directories and ``*_output_file`` helpers.

These symbols are referenced ONLY by the archived CLI runners in
``scripts/_legacy/`` (the per-method / per-dataset training scripts that
predate, and are superseded by, the unified shared-test-set harness
``scripts/run_retrieval_harness.py``).

They live here, separate from the active :mod:`src.utils.paths`, so the main
paths module stays focused and readable. ``paths.py`` re-exports everything
below for backward compatibility, so existing ``from src.utils.paths import
<NAME>`` imports in the archived scripts keep working.

This module is intentionally self-contained (it does not import ``paths`` to
avoid a circular import).
"""

import os
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parents[2]
_EXPERIMENT_ROOT = Path(os.environ.get("EXPERIMENT_ROOT", _PROJECT_DIR.parent))
_TASKS_DIR = _EXPERIMENT_ROOT / "dataset" / "tasks"
_MODELS = _PROJECT_DIR / "models"
_DEFAULT_TASK = "task_blue_shortbeak_perched_birds"


def _out(task: str, name: str) -> Path:
    """Per-task predicted-labels output path (legacy flat layout)."""
    return _TASKS_DIR / task / name


# --- Stanford Cars per-method model dirs -----------------------------------
CARS_MODEL_DIR = _MODELS / "cars"
CARS_HIGH_CONF_MODEL_DIR = _MODELS / "cars_high_conf"
CARS_NEG_ONLY_MODEL_DIR = _MODELS / "cars_neg_only"
CARS_PU_MODEL_DIR = _MODELS / "cars_pu"
CARS_ATTN_MODEL_DIR = _MODELS / "cars_attn"
CARS_ATTN_PU_MODEL_DIR = _MODELS / "cars_attn_pu"
CARS_ATTN_RANKING_MODEL_DIR = _MODELS / "cars_attn_ranking"
CARS_ATTN_NNPU_MODEL_DIR = _MODELS / "cars_attn_nnpu"
CARS_ATTN_SAPU_MODEL_DIR = _MODELS / "cars_attn_sapu"


def label_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "vqa_probing_labels.jsonl")


def output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "vqa_predicted_labels.jsonl")


# --- MLP baseline / expanded / high-conf / neg-only ------------------------
EXPANDED_MODEL_DIR = _MODELS / "expanded"
HIGH_CONF_MODEL_DIR = _MODELS / "high_conf"
NEG_ONLY_MODEL_DIR = _MODELS / "neg_only"


def expanded_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "expanded_predicted_labels.jsonl")


# --- PU family (MLP) -------------------------------------------------------
PU_MODEL_DIR = _MODELS / "pu"
PU_RANKING_MODEL_DIR = _MODELS / "pu_ranking"
NNPU_MODEL_DIR = _MODELS / "nnpu"
SAPU_MODEL_DIR = _MODELS / "sapu"


def pu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "pu_predicted_labels.jsonl")


def pu_ranking_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "pu_ranking_predicted_labels.jsonl")


def nnpu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "nnpu_predicted_labels.jsonl")


def sapu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "sapu_predicted_labels.jsonl")


# --- SUN per-method model dirs ---------------------------------------------
SUN_HIGH_CONF_MODEL_DIR = _MODELS / "sun_high_conf"
SUN_NEG_ONLY_MODEL_DIR = _MODELS / "sun_neg_only"
SUN_PU_MODEL_DIR = _MODELS / "sun_pu"
SUN_PU_RANKING_MODEL_DIR = _MODELS / "sun_pu_ranking"
SUN_NNPU_MODEL_DIR = _MODELS / "sun_nnpu"
SUN_SAPU_MODEL_DIR = _MODELS / "sun_sapu"

# --- Attention-pooled MLP --------------------------------------------------
ATTN_MODEL_DIR = _MODELS / "attn"
SUN_ATTN_MODEL_DIR = _MODELS / "sun_attn"


def attn_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_predicted_labels.jsonl")


# --- Attention + PU combined ----------------------------------------------
ATTN_PU_MODEL_DIR = _MODELS / "attn_pu"
ATTN_RANKING_MODEL_DIR = _MODELS / "attn_ranking"
ATTN_NNPU_MODEL_DIR = _MODELS / "attn_nnpu"
ATTN_SAPU_MODEL_DIR = _MODELS / "attn_sapu"

SUN_ATTN_PU_MODEL_DIR = _MODELS / "sun_attn_pu"
SUN_ATTN_RANKING_MODEL_DIR = _MODELS / "sun_attn_ranking"
SUN_ATTN_NNPU_MODEL_DIR = _MODELS / "sun_attn_nnpu"
SUN_ATTN_SAPU_MODEL_DIR = _MODELS / "sun_attn_sapu"


def attn_pu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_pu_predicted_labels.jsonl")


def attn_ranking_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_ranking_predicted_labels.jsonl")


def attn_nnpu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_nnpu_predicted_labels.jsonl")


def attn_sapu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_sapu_predicted_labels.jsonl")


# --- Plan 1: Multi-Boundary PU ---------------------------------------------
MULTI_BOUNDARY_PU_MODEL_DIR = _MODELS / "multi_boundary_pu"
SUN_MULTI_BOUNDARY_PU_MODEL_DIR = _MODELS / "sun_multi_boundary_pu"
ATTN_MULTI_BOUNDARY_PU_MODEL_DIR = _MODELS / "attn_multi_boundary_pu"
SUN_ATTN_MULTI_BOUNDARY_PU_MODEL_DIR = _MODELS / "sun_attn_multi_boundary_pu"


def multi_boundary_pu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "multi_boundary_pu_predicted_labels.jsonl")


def attn_multi_boundary_pu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_multi_boundary_pu_predicted_labels.jsonl")


# --- Plan 3A: DC-PU --------------------------------------------------------
DCPU_MODEL_DIR = _MODELS / "dcpu"
SUN_DCPU_MODEL_DIR = _MODELS / "sun_dcpu"
ATTN_DCPU_MODEL_DIR = _MODELS / "attn_dcpu"
SUN_ATTN_DCPU_MODEL_DIR = _MODELS / "sun_attn_dcpu"


def dcpu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "dcpu_predicted_labels.jsonl")


def attn_dcpu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_dcpu_predicted_labels.jsonl")


# --- Plan 3B: PU-AUC -------------------------------------------------------
PU_AUC_MODEL_DIR = _MODELS / "pu_auc"
SUN_PU_AUC_MODEL_DIR = _MODELS / "sun_pu_auc"
ATTN_PU_AUC_MODEL_DIR = _MODELS / "attn_pu_auc"
SUN_ATTN_PU_AUC_MODEL_DIR = _MODELS / "sun_attn_pu_auc"


def pu_auc_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "pu_auc_predicted_labels.jsonl")


def attn_pu_auc_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_pu_auc_predicted_labels.jsonl")


# --- Plan 5A: Graph-Regularized PU -----------------------------------------
GRAPH_PU_MODEL_DIR = _MODELS / "graph_pu"
SUN_GRAPH_PU_MODEL_DIR = _MODELS / "sun_graph_pu"
ATTN_GRAPH_PU_MODEL_DIR = _MODELS / "attn_graph_pu"
SUN_ATTN_GRAPH_PU_MODEL_DIR = _MODELS / "sun_attn_graph_pu"


def graph_pu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "graph_pu_predicted_labels.jsonl")


def attn_graph_pu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "attn_graph_pu_predicted_labels.jsonl")


# --- Plan 5B: GNN-PU -------------------------------------------------------
GNN_PU_MODEL_DIR = _MODELS / "gnn_pu"
SUN_GNN_PU_MODEL_DIR = _MODELS / "sun_gnn_pu"


def gnn_pu_output_file(task: str = _DEFAULT_TASK) -> Path:
    return _out(task, "gnn_pu_predicted_labels.jsonl")
