"""Warm-start the eight actual ProbeBank learners, independently of fusion mode.

This is deliberately not a replacement sigmoid head or a differentiable proxy
for cached scores. It requires the original trained checkpoint. The caller pins
the immutable task/Val/base-bank identity and merges original replay labels with
confirmed attribute feedback before calling (an override replaces its old label).
Unknown attribute labels must not be manufactured from a Joint negative.

The native objective and optimizer family are retained. Feedback weights change
only confirmed-label terms; PU risk/balance, P-U ranking and sampled triplets keep
their native definitions. Fixed Val is used only to select a checkpoint. The
returned model is frozen before the separate staged/joint fusion-weight phase.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from src.methods.mlp_probe import MLP, _get_device, _seed_everything
from src.methods.attn_probe import AttnPoolMLPWithProj
from src.methods.metric_attention_probes import (
    LearnedQueryAttentionProbe, TripletProjectionProbe,
)
from src.methods.nnpu_probe import nnpu_loss
from src.methods.dcpu_probe import dcpu_loss
from src.methods.pu_ranking_probe import pu_ranking_loss
from src.methods.pu_probe import cross_validate_probs
from src.methods.fixed_validation import split_labeled_validation


PROTOCOL = "native-probe-warm-start-v1"
NATIVE_METHODS = (
    "mlp_baseline", "kfold_pu", "triplet_loss", "attention_pooling",
    "attribute_conditioned_attention", "nnpu", "dcpu", "pu_ranking",
)
ATTENTION_METHODS = {"attention_pooling", "attribute_conditioned_attention"}
PU_METHODS = {"nnpu", "dcpu", "pu_ranking"}
OBJECTIVES = {
    "mlp_baseline": "class-balanced-labeled-bce",
    "kfold_pu": "cold-fit-only-oof-confidence-weighted-labeled-bce",
    "triplet_loss": "class-balanced-labeled-bce-plus-supervised-triplets",
    "attention_pooling": "learned-query-attention-class-balanced-labeled-bce",
    "attribute_conditioned_attention": "text-query-attention-class-balanced-labeled-bce",
    "nnpu": "native-pnu-nonnegative-risk-plus-weighted-confirmed-labels",
    "dcpu": "native-pnu-nonnegative-risk-and-balance-plus-weighted-confirmed-labels",
    "pu_ranking": "class-balanced-labeled-bce-plus-native-positive-unlabeled-ranking",
}


@dataclass(frozen=True)
class NativeUpdateConfig:
    """One deterministic configuration shared by both system-two buttons."""

    epochs: int = 20
    lr: float = 1e-4
    batch_size: int = 64
    batch_size_u: int = 256
    weight_decay: float = 1e-4
    anchor_strength: float = 1e-3
    kfold_epochs: int = 20
    kfold_folds: int = 5
    triplets_per_step: int = 32

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "batch_size_u", "kfold_epochs", "kfold_folds", "triplets_per_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Native {name} must be a positive integer")
        if self.kfold_folds < 2:
            raise ValueError("Native KFold needs at least two folds")
        for name in ("lr", "weight_decay", "anchor_strength"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == "lr" and value == 0):
                raise ValueError(f"Invalid native {name}")


def _ids(values: Sequence[str], count: int, name: str) -> set[str]:
    if len(values) != count or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"Native {name} IDs must align with the feature rows")
    result = set(values)
    if len(result) != count:
        raise ValueError(f"Native {name} IDs contain duplicates; merge overrides first")
    return result


def _load_base(method: str, path: Path, expected_sha256: str,
               feature_width: int, text_query: torch.Tensor | None) -> tuple[nn.Module, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Native Probe update requires its original trained checkpoint: {path}")
    content = path.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    if checksum != expected_sha256:
        raise RuntimeError("Native base checkpoint checksum mismatch")
    # Never unpickle arbitrary objects or silently initialize when loading fails.
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("Native checkpoint must contain a ProbeBank state_dict")
    state = payload["state_dict"]
    if not state or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                        for value in state.values()):
        raise RuntimeError("Native checkpoint contains invalid parameter tensors")
    kind = "attn" if method in ATTENTION_METHODS else "mlp"
    if payload.get("kind") != kind:
        raise RuntimeError("Native checkpoint feature kind does not match its learner")
    if method == "triplet_loss":
        projection = state.get("projector.0.weight")
        if projection is None or projection.ndim != 2:
            raise RuntimeError("Native triplet checkpoint is missing its real projection")
        model = TripletProjectionProbe(feature_width, projection_dim=int(projection.shape[0]))
    elif method == "attention_pooling":
        # Published native banks use four heads. Head count is not inferable from
        # the shape of MultiheadAttention tensors; do not guess another count.
        if payload.get("num_heads", 4) != 4:
            raise RuntimeError("Native attention checkpoint must use the canonical four heads")
        model = LearnedQueryAttentionProbe(feature_width, num_heads=4)
    elif method == "attribute_conditioned_attention":
        if text_query is None or text_query.ndim != 2 or text_query.shape[0] != 1:
            raise ValueError("Native conditioned attention requires its fixed attribute text query")
        if payload.get("num_heads", 4) != 4:
            raise RuntimeError("Native attention checkpoint must use the canonical four heads")
        if payload.get("text_dim") != int(text_query.shape[1]):
            raise RuntimeError("Native text-query dimension does not match its checkpoint")
        model = AttnPoolMLPWithProj(feature_width, int(text_query.shape[1]), num_heads=4)
    else:
        model = MLP(feature_width)
    if payload.get("model_class") != type(model).__name__:
        raise RuntimeError("Native checkpoint is not the requested real learner architecture")
    width_key = "patch_dim" if kind == "attn" else "input_dim"
    if payload.get(width_key) != feature_width:
        raise RuntimeError("Native checkpoint input dimensions do not match current features")
    model.load_state_dict(state, strict=True)
    return model, checksum


def _forward(method: str, model: nn.Module, features: torch.Tensor,
             text_query: torch.Tensor | None) -> torch.Tensor:
    return (model(features, text_query) if method in ATTENTION_METHODS else model(features)).reshape(-1)


def native_labeled_objective(method: str, logits: torch.Tensor, labels: torch.Tensor,
                             weights: torch.Tensor, *, pos_weight: torch.Tensor,
                             unlabeled_logits: torch.Tensor | None = None,
                             pi: float = 0.5) -> torch.Tensor:
    """Native label/PU terms; uniform weights recover the existing helper loss.

For nnPU/DC-PU, feedback adds only the excess confirmed-P/N BCE. In particular,
the unlabeled correction and DC-PU balance are not relabeled or reweighted.
"""
    positive, negative = labels == 1, labels == 0
    if method in ("nnpu", "dcpu"):
        if unlabeled_logits is None or not unlabeled_logits.numel():
            raise ValueError("Native PU risk requires an isolated unlabeled pool")
        if positive.any() and negative.any():
            if method == "nnpu":
                base = nnpu_loss(logits[positive], logits[negative], unlabeled_logits, pi)
            else:
                base = dcpu_loss(logits[positive], logits[negative], unlabeled_logits, pi, 0.5)
        elif positive.any():
            base = F.binary_cross_entropy_with_logits(logits[positive], labels[positive])
            base = base + F.binary_cross_entropy_with_logits(unlabeled_logits, torch.zeros_like(unlabeled_logits))
        else:
            base = F.binary_cross_entropy_with_logits(logits[negative], labels[negative])
        for mask in (positive, negative):
            if mask.any():
                label_bce = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], reduction="none")
                base = base + ((weights[mask] - 1) * label_bce).mean()
        return base
    losses = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=None if method == "kfold_pu" else pos_weight, reduction="none",
    )
    loss = (losses * weights).mean()
    if method == "pu_ranking":
        if unlabeled_logits is None or not unlabeled_logits.numel():
            raise ValueError("Native P-U ranking requires an isolated unlabeled pool")
        if positive.any():
            loss = loss + pu_ranking_loss(logits[positive], unlabeled_logits)
    return loss


def update_native_probe(
    method: str, checkpoint_path: str | Path, X_fit: np.ndarray, y_fit: np.ndarray, *,
    sample_weights: np.ndarray,
    validation_data: tuple[np.ndarray, np.ndarray],
    fit_ids: Sequence[str], validation_ids: Sequence[str],
    expected_checkpoint_sha256: str,
    X_unlabeled: np.ndarray | None = None, unlabeled_ids: Sequence[str] = (),
    blocked_ids: Sequence[str] = (), text_query: torch.Tensor | None = None,
    config: NativeUpdateConfig | None = None, seed: int = 0,
) -> tuple[nn.Module, dict]:
    """Update one original method×attribute×seed; no fusion-mode argument.

Inputs contain only confirmed binary attribute labels, including original fit
replay. New feedback is merged by ID first. The original checkpoint is never
modified; an updated clone plus audit is returned for run-local persistence.
"""
    if method not in NATIVE_METHODS:
        raise ValueError(f"Unsupported native learner: {method}")
    config = config or NativeUpdateConfig()
    config.validate()
    if type(seed) is not int or seed < 0:
        raise ValueError("Native seed must be a non-negative integer")
    if validation_data is None:
        raise ValueError("Native update requires the fixed original VQA Val")
    X, Xv, y, yv = split_labeled_validation(X_fit, y_fit, seed, validation_data)
    expected_ndim = 3 if method in ATTENTION_METHODS else 2
    if X.ndim != expected_ndim:
        raise ValueError("Native learner received the wrong feature kind")
    fit_set = _ids(fit_ids, len(y), "fit")
    val_set = _ids(validation_ids, len(yv), "validation")
    if fit_set & (val_set | set(blocked_ids)):
        raise ValueError("Native fitting data overlaps Val/Test/Query")
    if set(np.unique(y)) != {0, 1} or set(np.unique(yv)) != {0, 1}:
        raise ValueError("Native fit and fixed Val must each contain both classes")
    weights = np.asarray(sample_weights, dtype=np.float32)
    if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Native feedback weights must be positive finite values aligned with fit rows")
    if method in PU_METHODS:
        U = np.asarray(X_unlabeled)
        if U.ndim != 2 or U.shape[1:] != X.shape[1:] or not len(U) or not np.isfinite(U).all():
            raise ValueError("Native PU learner requires finite, aligned unlabeled features")
    else:
        U = np.asarray(X_unlabeled) if X_unlabeled is not None else np.empty((0, X.shape[-1]))
    u_set = _ids(unlabeled_ids, len(U), "unlabeled")
    if u_set & (fit_set | val_set | set(blocked_ids)):
        raise ValueError("Native unlabeled pool overlaps labeled fit/Val/Test/Query")
    if text_query is not None and not torch.isfinite(text_query).all():
        raise ValueError("Native attribute text query is not finite")
    _seed_everything(seed)
    path = Path(checkpoint_path)
    model, checksum = _load_base(method, path, expected_checkpoint_sha256, int(X.shape[-1]), text_query)
    device = _get_device()
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    anchor = {key: value.detach().clone() for key, value in model.named_parameters()}
    oof = None
    if method == "kfold_pu":
        if len(y) < config.kfold_folds or min(int(np.sum(y == label)) for label in (0, 1)) < 2:
            raise ValueError("Native KFold needs at least two fit examples per class and enough total rows")
        # phi0 already saw old fit labels: warm-starting OOF folds from it would
        # leak each held-out fold. Only these auxiliary OOF models cold-start.
        oof = cross_validate_probs(
            "native-feedback", X, y, int(X.shape[-1]), n_folds=config.kfold_folds,
            epochs=config.kfold_epochs, lr=config.lr, batch_size=config.batch_size, seed=seed,
        )
        if oof.shape != y.shape or not np.isfinite(oof).all() or np.any((oof < 0) | (oof > 1)):
            raise RuntimeError("Native KFold returned invalid fit-only OOF confidence")
        weights = weights * np.where(y == 1, oof, 1).astype(np.float32)
        _seed_everything(seed)
    Xt, yt, wt = (torch.as_tensor(value, dtype=torch.float32) for value in (X, y, weights))
    Xvt = torch.as_tensor(Xv, dtype=torch.float32, device=device)
    Ut = torch.as_tensor(U, dtype=torch.float32) if method in PU_METHODS else None
    tq = None if text_query is None else text_query.detach().to(device)
    loader = DataLoader(TensorDataset(Xt, yt, wt), batch_size=config.batch_size, shuffle=True)
    pi = float(np.mean(y))
    pos_weight = torch.tensor(float(np.sum(y == 0)) / int(np.sum(y == 1)), device=device)
    optimizer_class = torch.optim.AdamW if method in ("triplet_loss", "attention_pooling") else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    best_auc, best_state, best_epoch = -1.0, None, None
    before_auc = _auc(method, model, Xvt, yv, tq)
    positives, negatives = Xt[yt == 1], Xt[yt == 0]
    triplet_active = method == "triplet_loss" and len(positives) >= 2 and len(negatives) >= 1
    for epoch in range(1, config.epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            logits = _forward(method, model, xb, tq)
            logits_u = None
            if Ut is not None:
                u_indices = torch.randint(0, len(Ut), (config.batch_size_u,))
                logits_u = model(Ut[u_indices].to(device)).reshape(-1)
            loss = native_labeled_objective(method, logits, yb, wb, pos_weight=pos_weight,
                                             unlabeled_logits=logits_u, pi=pi)
            if triplet_active:
                count = min(config.triplets_per_step, max(2, len(positives)))
                indices = torch.randint(0, len(positives), (count,))
                partners = torch.randint(0, len(positives) - 1, (count,))
                partners += (partners >= indices).to(partners.dtype)
                negatives_idx = torch.randint(0, len(negatives), (count,))
                loss = loss + F.triplet_margin_loss(
                    model.embedding(positives[indices].to(device)),
                    model.embedding(positives[partners].to(device)),
                    model.embedding(negatives[negatives_idx].to(device)), margin=0.5, p=2,
                )
            if config.anchor_strength:
                # Mean squared displacement controls scale across architectures.
                displacement = sum((parameter - anchor[name]).square().sum()
                                   for name, parameter in model.named_parameters())
                count = sum(parameter.numel() for parameter in model.parameters())
                loss = loss + config.anchor_strength * displacement / count
            if not torch.isfinite(loss):
                raise RuntimeError("Native learner produced a non-finite loss")
            optimizer.zero_grad()
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise RuntimeError("Native learner produced a non-finite gradient")
            optimizer.step()
        scheduler.step()
        auc = _auc(method, model, Xvt, yv, tq)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Native learner produced no updated checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("Native base checkpoint changed while training; discard this update")
    delta = sum(float((best_state[key] - value).square().sum()) for key, value in initial_state.items())
    audit = {
        "protocol": PROTOCOL, "method": method, "seed": seed, "config": asdict(config),
        "baseCheckpointSha256": checksum, "warmStarted": True,
        "fitCount": len(y), "valCount": len(yv), "unlabeledCount": len(U),
        "beforeValAuc": before_auc, "afterValAuc": best_auc, "selectedEpoch": best_epoch,
        "objective": OBJECTIVES[method], "optimizer": optimizer_class.__name__,
        "classPriorFitOnly": pi, "tripletActive": triplet_active,
        "feedbackWeightPolicy": "confirmed-label-terms-only; native-auxiliaries-unchanged",
        "replayPolicy": "original-fit-and-feedback-merged-by-id; overrides-replace-old-label",
        "validationPolicy": "fixed-vqa-val-checkpoint-selection-only",
        "kfoldPolicy": "cold-init-fit-only-oof; warm-start-final-probe" if oof is not None else None,
        "parameterDisplacementSquared": delta, "frozenForFusion": True,
        "fusionModeIndependent": True,
    }
    return model, audit


def _auc(method: str, model: nn.Module, Xv: torch.Tensor, yv: np.ndarray,
         text_query: torch.Tensor | None) -> float:
    model.eval()
    with torch.no_grad():
        scores = _forward(method, model, Xv, text_query).detach().cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("Native validation produced non-finite scores")
    return float(roc_auc_score(yv, scores))


def score_native_probe(method: str, model: nn.Module, full_features, *,
                       text_query: torch.Tensor | None = None, chunk_size: int = 1024) -> np.ndarray:
    """Full-gallery inference only; returns raw sigmoid probabilities, not z."""
    if method not in NATIVE_METHODS or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Invalid native scoring method/chunk size")
    device = next(model.parameters()).device
    tq = None if text_query is None else text_query.detach().to(device)
    if method == "attribute_conditioned_attention" and tq is None:
        raise ValueError("Native conditioned scoring needs the fixed attribute text query")
    model.eval()
    output = np.empty(len(full_features), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(full_features), chunk_size):
            features = torch.as_tensor(full_features[start:start + chunk_size], dtype=torch.float32, device=device)
            if not torch.isfinite(features).all():
                raise ValueError("Native scoring features are not finite")
            probabilities = torch.sigmoid(_forward(method, model, features, tq))
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Native scoring produced non-finite probabilities")
            output[start:start + len(features)] = probabilities.cpu().numpy()
    return output
