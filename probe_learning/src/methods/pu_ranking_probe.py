"""Positive-unlabeled pairwise ranking probe (``pu_ranking``).

Each attribute MLP minimizes ``lambda_sup`` times class-weighted BCE on labeled
positives/negatives plus ``lambda_rank`` times
``softplus(-(score_positive-score_unlabeled))`` over randomly sampled U.  The
canonical ``pu_ranking`` (PURA in the paper) uses both terms.  Setting
``lambda_sup=0`` gives the loss-only ``pu_pairwise_only`` baseline with the same
architecture, sampler, optimizer, scheduler, and validation-AUC checkpointing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from src.methods.mlp_probe import MLP, _get_device, _seed_everything
from src.methods.fixed_validation import split_labeled_validation


def pu_ranking_loss(scores_P: torch.Tensor, scores_U: torch.Tensor) -> torch.Tensor:
    """Pairwise ranking: each positive should score higher than each U sample."""
    diff = scores_P.unsqueeze(1) - scores_U.unsqueeze(0)
    return F.softplus(-diff).mean()


def train_one_attribute_pu_ranking(
    attr: str,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    input_dim: int,
    lambda_sup: float = 1.0,
    lambda_rank: float = 1.0,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    batch_size_u: int = 256,
    seed: int = 42,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[MLP, dict]:
    """Train MLP with weighted BCE(P,N) and/or soft P-U pairwise ranking."""
    _seed_everything(seed)
    X_train, X_val, y_train, y_val = split_labeled_validation(X_labeled, y_labeled, seed, validation_data)

    device = _get_device()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_u_all = torch.tensor(X_unlabeled, dtype=torch.float32)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True,
    )

    model = MLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state = None
    n_u = len(X_u_all)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            u_idx = torch.randint(0, n_u, (batch_size_u,))
            x_u = X_u_all[u_idx].to(device)

            logits_pn = model(xb).squeeze(1)
            logits_u = model(x_u).squeeze(1)

            L_sup = criterion(logits_pn.unsqueeze(1), yb)

            pos_mask = yb.squeeze(1) == 1
            if pos_mask.any():
                L_rank = pu_ranking_loss(logits_pn[pos_mask], logits_u)
            else:
                L_rank = torch.tensor(0.0, device=device)

            # With a pure pairwise objective an all-negative labeled minibatch
            # has no defined P-U pair.  Skipping it avoids manufacturing a
            # negative-only loss while preserving the production sampler.
            if lambda_sup == 0.0 and not pos_mask.any():
                continue

            loss = lambda_sup * L_sup + lambda_rank * L_rank
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val_t).cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            try:
                val_auc = roc_auc_score(y_val, probs)
            except ValueError:
                val_auc = 0.0
            if np.isnan(val_auc):
                val_auc = 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            val_acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
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
        "val_samples": len(y_val),
        "train_pos_ratio": float(y_train.mean()),
        "val_accuracy": float(accuracy_score(y_val, preds)),
        "val_f1": float(f1_score(y_val, preds, zero_division=0)),
        "val_auc": float(roc_auc_score(y_val, probs)) if len(np.unique(y_val)) > 1 else 0.0,
        "objective": (
            "soft_pu_pairwise_only" if lambda_sup == 0.0
            else "labeled_bce_plus_soft_pu_pairwise"
        ),
        "uses_labeled_bce": bool(lambda_sup != 0.0),
        "uses_pairwise_ranking": bool(lambda_rank != 0.0),
        "labeled_negatives_used_in_training_loss": bool(lambda_sup != 0.0),
        "lambda_sup": float(lambda_sup),
        "lambda_rank": float(lambda_rank),
    }

    print(f"\n  [{attr}] Final val  acc={metrics['val_accuracy']:.4f}  "
          f"f1={metrics['val_f1']:.4f}  auc={metrics['val_auc']:.4f}")
    print(f"  [{attr}] {classification_report(y_val, preds, zero_division=0)}")

    return model, metrics
