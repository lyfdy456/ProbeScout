"""Class-weighted supervised MLP baseline for one binary attribute.

``mlp_baseline`` uses frozen image embeddings, a 256-128 two-hidden-layer MLP
with ReLU/dropout, positive-class-weighted BCE, Adam, cosine learning-rate
decay, and best validation-AUC checkpoint selection.  The same architecture is
reused by the PU and ReCAP families, whose files replace or augment this loss.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from src.data.cub import image_field_to_relative_path
from src.methods.fixed_validation import split_labeled_validation


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    """Seed model initialization, minibatch shuffling, and NumPy helpers."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def train_one_attribute(
    attr: str,
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[MLP, dict]:
    """Train a binary MLP for one attribute. Returns (model, metrics_dict)."""
    _seed_everything(seed)
    X_train, X_val, y_train, y_val = split_labeled_validation(X, y, seed, validation_data)

    device = _get_device()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = y_val

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )

    model = MLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
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


def _cub_path_to_image_name(rel_path: str) -> str:
    stripped = rel_path.removeprefix("images/")
    parts = stripped.split("/", 1)
    return f"{parts[0]}__{parts[1]}"


def _sun_path_to_image_name(rel_path: str) -> str:
    return rel_path.replace("/", "__")


def _cars_path_to_image_name(rel_path: str) -> str:
    return rel_path


PATH_CONVERTERS = {
    "cub": _cub_path_to_image_name,
    "sun": _sun_path_to_image_name,
    "cars": _cars_path_to_image_name,
}


def predict_remaining(
    models: dict[str, MLP],
    embeddings: np.ndarray,
    records: pd.DataFrame,
    labeled_paths: set[str],
    attributes: list[str],
    dataset: str = "cub",
) -> list[dict]:
    """Predict all attributes for images not in the labeled set."""
    device = next(iter(models.values())).net[0].weight.device
    path_to_name = PATH_CONVERTERS[dataset]

    remaining_indices = []
    remaining_rel_paths = []
    for _, row in records.iterrows():
        if row["relative_path"] not in labeled_paths:
            remaining_indices.append(int(row["embedding_index"]))
            remaining_rel_paths.append(row["relative_path"])

    print(f"\nPredicting on {len(remaining_indices)} remaining images ...")

    X_remain = torch.tensor(
        embeddings[remaining_indices], dtype=torch.float32
    ).to(device)

    predictions = {}
    for attr, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(X_remain).cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)
        predictions[attr] = preds

    results = []
    for i, rel_path in enumerate(remaining_rel_paths):
        image_name = path_to_name(rel_path)
        entry = {"image": image_name}
        for attr in attributes:
            entry[attr] = int(predictions[attr][i])
        results.append(entry)

    return results


def predict_remaining_with_probs(
    models: dict[str, MLP],
    embeddings: np.ndarray,
    records: pd.DataFrame,
    labeled_paths: set[str],
    attributes: list[str],
    dataset: str = "cub",
) -> list[dict]:
    """Like predict_remaining, but each attribute also gets an ``<attr>_prob`` field."""
    device = next(iter(models.values())).net[0].weight.device
    path_to_name = PATH_CONVERTERS[dataset]

    remaining_indices = []
    remaining_rel_paths = []
    for _, row in records.iterrows():
        if row["relative_path"] not in labeled_paths:
            remaining_indices.append(int(row["embedding_index"]))
            remaining_rel_paths.append(row["relative_path"])

    print(f"\nPredicting (with probs) on {len(remaining_indices)} remaining images ...")

    X_remain = torch.tensor(
        embeddings[remaining_indices], dtype=torch.float32
    ).to(device)

    pred_labels: dict[str, np.ndarray] = {}
    pred_probs: dict[str, np.ndarray] = {}
    for attr, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(X_remain).cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)
        pred_labels[attr] = preds
        pred_probs[attr] = probs

    results = []
    for i, rel_path in enumerate(remaining_rel_paths):
        image_name = path_to_name(rel_path)
        entry = {"image": image_name}
        for attr in attributes:
            entry[attr] = int(pred_labels[attr][i])
            entry[f"{attr}_prob"] = float(pred_probs[attr][i])
        results.append(entry)

    return results
