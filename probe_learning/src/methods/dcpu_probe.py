"""DC-PU probe: doubly-constrained PU learning (nnPU + balance constraint)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from src.methods.mlp_probe import MLP, _get_device, _seed_everything
from src.methods.fixed_validation import split_labeled_validation


def dcpu_loss(
    scores_P: torch.Tensor,
    scores_N: torch.Tensor,
    scores_U: torch.Tensor,
    pi: float,
    lambda_balance: float,
) -> torch.Tensor:
    L_pos = F.binary_cross_entropy_with_logits(
        scores_P, torch.ones_like(scores_P))
    L_neg_clean = F.binary_cross_entropy_with_logits(
        scores_N, torch.zeros_like(scores_N))
    L_pos_as_neg = F.binary_cross_entropy_with_logits(
        scores_P, torch.zeros_like(scores_P))
    L_unl = F.binary_cross_entropy_with_logits(
        scores_U, torch.zeros_like(scores_U))

    neg_risk = L_unl - pi * L_pos_as_neg
    neg_risk_nn = torch.clamp(neg_risk, min=0.0)

    R_pu = L_pos + L_neg_clean + neg_risk_nn
    R_balance = torch.abs(L_pos - neg_risk_nn)

    return R_pu + lambda_balance * R_balance


def train_one_attribute_dcpu(
    attr: str,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    input_dim: int,
    pi: float | None = None,
    lambda_balance: float = 0.5,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    batch_size_u: int = 256,
    seed: int = 42,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[MLP, dict]:
    """Train MLP with DC-PU loss (nnPU + balance constraint)."""
    _seed_everything(seed)
    X_train, X_val, y_train, y_val = split_labeled_validation(X_labeled, y_labeled, seed, validation_data)
    if pi is None:
        # Keep the held-out validation labels outside all training statistics.
        pi = float(y_train.sum()) / len(y_train)

    device = _get_device()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_u_all = torch.tensor(X_unlabeled, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True,
    )

    model = MLP(input_dim).to(device)
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

            pos_mask = yb.squeeze(1) == 1
            neg_mask = ~pos_mask

            if pos_mask.any() and neg_mask.any():
                loss = dcpu_loss(logits_pn[pos_mask], logits_pn[neg_mask],
                                 logits_u, pi, lambda_balance)
            elif pos_mask.any():
                L_pos = F.binary_cross_entropy_with_logits(
                    logits_pn[pos_mask], torch.ones_like(logits_pn[pos_mask]))
                L_unl = F.binary_cross_entropy_with_logits(
                    logits_u, torch.zeros_like(logits_u))
                loss = L_pos + L_unl
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits_pn[neg_mask], torch.zeros_like(logits_pn[neg_mask]))

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
        "pi": pi,
    }

    print(f"\n  [{attr}] Final val  acc={metrics['val_accuracy']:.4f}  "
          f"f1={metrics['val_f1']:.4f}  auc={metrics['val_auc']:.4f}  pi={pi:.4f}")
    print(f"  [{attr}] {classification_report(y_val, preds, zero_division=0)}")

    return model, metrics
