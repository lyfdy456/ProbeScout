"""K-fold confidence-weighted PU-style probe (``kfold_pu``).

Five stratified fold models produce out-of-fold positive probabilities for the
labeled examples.  Labeled positives receive those probabilities as training
weights, so suspected false positives are down-weighted; labeled negatives keep
weight 1.  A fresh class-compatible MLP is then trained with weighted BCE.
This method does not add unlabeled images to the final training set.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from src.methods.mlp_probe import MLP, _get_device, _seed_everything
from src.methods.fixed_validation import split_labeled_validation


def _train_fold_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> MLP:
    """Train a lightweight MLP for one CV fold (no val-based early stopping)."""
    _seed_everything(seed)
    device = _get_device()
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model = MLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def cross_validate_probs(
    attr: str,
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    n_folds: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """K-fold CV to obtain out-of-sample predicted probabilities for every sample."""
    device = _get_device()
    oof_probs = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model = _train_fold_model(
            X[train_idx], y[train_idx], input_dim, epochs, lr, batch_size,
            seed + fold_idx,
        )
        X_val_t = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_val_t).cpu().numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        oof_probs[val_idx] = probs
        fold_acc = ((probs >= 0.5).astype(int) == y[val_idx]).mean()
        print(f"    [CV fold {fold_idx}/{n_folds}] acc={fold_acc:.4f}")

    return oof_probs


def train_one_attribute_weighted(
    attr: str,
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    input_dim: int,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[MLP, dict]:
    """Train a binary MLP with per-sample weights. Returns (model, metrics_dict)."""
    from sklearn.model_selection import train_test_split

    _seed_everything(seed)

    if validation_data is None:
        X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=seed, stratify=y,
        )
    else:
        X_train, X_val, y_train, y_val = split_labeled_validation(X, y, seed, validation_data)
        w_train = np.asarray(sample_weights)
        if w_train.shape != y_train.shape or not np.all(np.isfinite(w_train)) or np.any(w_train < 0) or not np.any(w_train > 0):
            raise ValueError("explicit fit sample weights must be aligned, finite, non-negative, and nonzero")

    device = _get_device()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    w_train_t = torch.tensor(w_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = y_val

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t, w_train_t),
        batch_size=batch_size, shuffle=True,
    )

    model = MLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            per_sample_loss = criterion(model(xb), yb)
            loss = (per_sample_loss * wb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val_t).cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)
            try:
                val_auc = roc_auc_score(y_val_np, probs)
            except ValueError:
                val_auc = 0.0
            if np.isnan(val_auc):
                val_auc = 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            val_acc = accuracy_score(y_val_np, preds)
            print(f"  [{attr}] epoch {epoch:3d}  val_acc={val_acc:.4f}  val_auc={val_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_val_t).cpu().numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

    metrics = {
        "attribute": attr,
        "train_samples": len(y_train),
        "val_samples": len(y_val_np),
        "train_pos_ratio": float(y_train.mean()),
        "val_accuracy": float(accuracy_score(y_val_np, preds)),
        "val_f1": float(f1_score(y_val_np, preds, zero_division=0)),
        "val_auc": float(roc_auc_score(y_val_np, probs)) if len(np.unique(y_val_np)) > 1 else 0.0,
    }

    print(f"\n  [{attr}] Final val  acc={metrics['val_accuracy']:.4f}  "
          f"f1={metrics['val_f1']:.4f}  auc={metrics['val_auc']:.4f}")
    print(f"  [{attr}] {classification_report(y_val_np, preds, zero_division=0)}")

    return model, metrics
