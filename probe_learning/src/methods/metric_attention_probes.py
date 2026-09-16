"""Additional ProbeBank probes for metric learning and patch-token attention.

The module adds three genuinely new per-attribute probes:

``cosine_margin_ranking``
    Learns a normalized attribute direction in the frozen pooled-embedding
    space.  Class-balanced BCE calibrates the binary score, while an explicit
    hinge margin requires labeled positives to outrank sampled unlabeled rows
    in cosine similarity.

``triplet_loss``
    Learns a normalized low-dimensional projection.  Class-balanced BCE is
    combined with supervised positive-positive-negative triplets constructed
    only from labeled examples, so unlabeled images are not silently treated as
    reliable negatives.

``attention_pooling``
    Learns one free attention query per attribute probe and pools frozen image
    patch tokens before a binary classifier.  Unlike
    ``attribute_conditioned_attention``, it never reads the attribute text;
    this makes the value of textual conditioning directly measurable.

The fourth requested method, ``attribute_conditioned_attention``, is the
explicit ProbeBank name for the existing text-query cross-attention model in
``attn_probe.py``.  It is registered by the retrieval harness without
duplicating that implementation.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.methods.mlp_probe import _get_device, _seed_everything
from src.methods.fixed_validation import split_labeled_validation


def cosine_margin_loss(
    positive_cosine: torch.Tensor,
    unlabeled_cosine: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Require every sampled positive cosine to exceed U by ``margin``."""
    if positive_cosine.numel() == 0 or unlabeled_cosine.numel() == 0:
        return positive_cosine.sum() * 0.0
    differences = positive_cosine.unsqueeze(1) - unlabeled_cosine.unsqueeze(0)
    return F.relu(float(margin) - differences).mean()


class CosineMarginRanker(nn.Module):
    """Normalized linear attribute direction with a learnable logit scale."""

    def __init__(self, input_dim: int, initial_scale: float = 10.0):
        super().__init__()
        self.attribute_direction = nn.Parameter(torch.randn(input_dim))
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(initial_scale))))
        self.bias = nn.Parameter(torch.zeros(()))

    def cosine_score(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=-1)
        direction = F.normalize(self.attribute_direction, dim=0)
        return x_norm @ direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp().clamp(1.0, 100.0)
        return (scale * self.cosine_score(x) + self.bias).unsqueeze(1)


class TripletProjectionProbe(nn.Module):
    """Projection trained by labeled triplets plus a binary logit head."""

    def __init__(self, input_dim: int, projection_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(projection_dim, 1)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(x))


class LearnedQueryAttentionProbe(nn.Module):
    """Attribute-specific learned-query attention over frozen patch tokens."""

    def __init__(self, patch_dim: int, num_heads: int = 4):
        super().__init__()
        if patch_dim % num_heads != 0:
            raise ValueError(f"patch_dim={patch_dim} must be divisible by num_heads={num_heads}")
        self.query = nn.Parameter(torch.empty(1, 1, patch_dim))
        nn.init.normal_(self.query, std=patch_dim ** -0.5)
        self.cross_attention = nn.MultiheadAttention(
            patch_dim, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(patch_dim)
        self.classifier = nn.Sequential(
            nn.Linear(patch_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        text_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del text_query  # Deliberately text-free; the conditioned variant uses it.
        query = self.query.expand(patch_tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attention(query, patch_tokens, patch_tokens)
        return self.classifier(self.norm(pooled[:, 0]))


def _split_labeled(X: np.ndarray, y: np.ndarray, seed: int, validation_data=None):
    return split_labeled_validation(X, y, seed, validation_data)


def _positive_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    return torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=device)


def _validation_auc(model: nn.Module, X_val: torch.Tensor, y_val: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X_val).detach().cpu().numpy().ravel()
    if len(np.unique(y_val)) < 2:
        return 0.0
    value = float(roc_auc_score(y_val, logits))
    return 0.0 if not np.isfinite(value) else value


def _final_metrics(
    attr: str,
    model: nn.Module,
    X_val: torch.Tensor,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X_val).detach().cpu().numpy().ravel()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "attribute": attr,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "train_pos_ratio": float(np.mean(y_train)),
        "val_accuracy": float(accuracy_score(y_val, predictions)),
        "val_f1": float(f1_score(y_val, predictions, zero_division=0)),
        "val_auc": (
            float(roc_auc_score(y_val, probabilities))
            if len(np.unique(y_val)) > 1 else 0.0
        ),
    }


def train_one_attribute_cosine_margin_ranking(
    attr: str,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    input_dim: int,
    margin: float = 0.2,
    lambda_rank: float = 1.0,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    batch_size_u: int = 256,
    seed: int = 42,
) -> tuple[CosineMarginRanker, dict]:
    """Fit class-balanced BCE plus positive-vs-U cosine margin ranking."""
    _seed_everything(seed)
    X_train, X_val, y_train, y_val = _split_labeled(X_labeled, y_labeled, seed)
    device = _get_device()
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    X_u_t = torch.as_tensor(X_unlabeled, dtype=torch.float32)
    model = CosineMarginRanker(input_dim).to(device)
    supervised = nn.BCEWithLogitsLoss(pos_weight=_positive_weight(y_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_auc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).squeeze(1)
            loss = supervised(logits, yb)
            positive = model.cosine_score(xb[yb == 1])
            if len(X_u_t):
                indices = torch.randint(0, len(X_u_t), (min(batch_size_u, len(X_u_t)),))
                unlabeled = model.cosine_score(X_u_t[indices].to(device))
                loss = loss + float(lambda_rank) * cosine_margin_loss(
                    positive, unlabeled, margin=margin,
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_auc = _validation_auc(model, X_val_t, y_val)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError("cosine-margin training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, _final_metrics(attr, model, X_val_t, y_train, y_val)


def train_one_attribute_triplet_loss(
    attr: str,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    input_dim: int,
    projection_dim: int = 128,
    margin: float = 0.5,
    lambda_triplet: float = 1.0,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    triplets_per_step: int = 32,
    seed: int = 42,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[TripletProjectionProbe, dict]:
    """Fit BCE plus supervised positive-positive-negative triplet loss."""
    _seed_everything(seed)
    X_train, X_val, y_train, y_val = _split_labeled(X_labeled, y_labeled, seed, validation_data)
    device = _get_device()
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    positives = torch.as_tensor(X_train[y_train == 1], dtype=torch.float32)
    negatives = torch.as_tensor(X_train[y_train == 0], dtype=torch.float32)
    triplet_active = len(positives) >= 2 and len(negatives) >= 1
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    model = TripletProjectionProbe(input_dim, projection_dim=projection_dim).to(device)
    supervised = nn.BCEWithLogitsLoss(pos_weight=_positive_weight(y_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_auc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = supervised(model(xb).squeeze(1), yb)
            if triplet_active:
                count = min(int(triplets_per_step), max(2, len(positives)))
                anchor_idx = torch.randint(0, len(positives), (count,))
                partner_idx = torch.randint(0, len(positives) - 1, (count,))
                partner_idx += (partner_idx >= anchor_idx).to(partner_idx.dtype)
                negative_idx = torch.randint(0, len(negatives), (count,))
                anchor = model.embedding(positives[anchor_idx].to(device))
                partner = model.embedding(positives[partner_idx].to(device))
                negative = model.embedding(negatives[negative_idx].to(device))
                metric_loss = F.triplet_margin_loss(
                    anchor, partner, negative, margin=float(margin), p=2,
                )
                loss = loss + float(lambda_triplet) * metric_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_auc = _validation_auc(model, X_val_t, y_val)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError("triplet training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics = _final_metrics(attr, model, X_val_t, y_train, y_val)
    metrics["triplet_active"] = bool(triplet_active)
    return model, metrics


def train_one_attribute_attention_pooling(
    attr: str,
    X_patches: np.ndarray,
    y_labeled: np.ndarray,
    patch_dim: int,
    num_heads: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[LearnedQueryAttentionProbe, dict]:
    """Train text-free learned-query attention pooling with balanced BCE."""
    _seed_everything(seed)
    X_train, X_val, y_train, y_val = _split_labeled(X_patches, y_labeled, seed, validation_data)
    device = _get_device()
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    model = LearnedQueryAttentionProbe(patch_dim, num_heads=num_heads).to(device)
    supervised = nn.BCEWithLogitsLoss(pos_weight=_positive_weight(y_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_auc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = supervised(model(xb).squeeze(1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_auc = _validation_auc(model, X_val_t, y_val)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError("attention-pooling training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, _final_metrics(attr, model, X_val_t, y_train, y_val)
