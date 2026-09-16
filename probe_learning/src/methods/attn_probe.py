"""Attention-pooled MLP probe: text-conditioned cross-attention over CLIP patch tokens."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.methods.fixed_validation import split_labeled_validation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from transformers import AutoProcessor, AutoTokenizer, CLIPModel, SiglipModel

from src.methods.mlp_probe import _seed_everything


class AttnPoolMLP(nn.Module):
    """Cross-attention pooling followed by a binary MLP classifier.

    The text embedding is used as the query to attend over image patch tokens,
    producing an attribute-specific image representation.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, patch_tokens: torch.Tensor, text_query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (batch, num_patches, embed_dim)
            text_query: (1, embed_dim) -- broadcast to batch
        Returns:
            logits: (batch, 1)
        """
        query = text_query.unsqueeze(1).expand(patch_tokens.size(0), -1, -1)  # (B, 1, D)
        pooled, _ = self.cross_attn(query, patch_tokens, patch_tokens)  # (B, 1, D)
        pooled = self.norm(pooled.squeeze(1))  # (B, D)
        return self.classifier(pooled)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_attribute_text(attr_text: str, model_name: str) -> torch.Tensor:
    """Encode an attribute description with CLIP text encoder, returning the
    hidden state before projection (768-d for ViT-B/32)."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    inputs = processor(text=[attr_text], return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        text_outputs = model.text_model(**inputs)
        # pooler_output is the hidden state of the EOT token after the final layer norm
        hidden = text_outputs.pooler_output  # (1, 512) for projected, but we want pre-projection

    # text_model outputs are 512-d (hidden_size of text_model for clip-vit-base-patch32)
    # The vision hidden_size is 768. We need to project text to 768-d.
    # Use the text_projection matrix transposed is not ideal; instead we use a
    # simple linear layer inside the model. But actually for clip-vit-base-patch32:
    #   text_model.config.hidden_size = 512
    #   vision_model.config.hidden_size = 768
    # So we'll use the text projection to get 512-d, then the AttnPoolMLP handles
    # dimension mismatch by projecting. Actually let's just return the raw hidden
    # and handle projection inside the model.

    return hidden.detach()  # (1, text_hidden_dim)


def encode_attribute_texts_batch(
    attr_texts: list[str], model_name: str,
) -> dict[str, torch.Tensor]:
    """Encode multiple attribute texts, returning {attr_text: (1, hidden_dim)} tensors."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    results = {}
    for text in attr_texts:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            text_outputs = model.text_model(**inputs)
            hidden = text_outputs.pooler_output
        results[text] = hidden.detach()

    return results


def encode_attribute_texts_for_backbone(
    attr_texts: list[str],
    backbone: str,
    model_name: str,
) -> dict[str, torch.Tensor]:
    """Encode attribute text queries for the patch-token attention backbone."""
    if backbone == "clip":
        return encode_attribute_texts_batch(attr_texts, model_name)
    if backbone != "siglip":
        raise ValueError(f"attention text queries are not wired for backbone {backbone!r}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = SiglipModel.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = SiglipModel.from_pretrained(model_name)
    model.eval()

    results = {}
    for text in attr_texts:
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=int(model.config.text_config.max_position_embeddings),
        )
        with torch.inference_mode():
            text_outputs = model.text_model(**inputs)
            hidden = getattr(text_outputs, "pooler_output", None)
            if hidden is None:
                hidden = text_outputs.last_hidden_state[:, -1, :]
        results[text] = hidden.detach()

    return results


class AttnPoolMLPWithProj(nn.Module):
    """AttnPoolMLP with a learnable projection from text_dim to patch_dim."""

    def __init__(self, patch_dim: int = 768, text_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, patch_dim)
        self.cross_attn = nn.MultiheadAttention(
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

    def forward(self, patch_tokens: torch.Tensor, text_query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (batch, num_patches, patch_dim)
            text_query: (1, text_dim)
        """
        projected = self.text_proj(text_query)  # (1, patch_dim)
        query = projected.unsqueeze(1).expand(patch_tokens.size(0), -1, -1)  # (B, 1, patch_dim)
        pooled, _ = self.cross_attn(query, patch_tokens, patch_tokens)  # (B, 1, patch_dim)
        pooled = self.norm(pooled.squeeze(1))
        return self.classifier(pooled)


def train_one_attribute_attn(
    attr: str,
    X_patches: np.ndarray,
    y: np.ndarray,
    text_query: torch.Tensor,
    patch_dim: int = 768,
    text_dim: int = 512,
    num_heads: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[AttnPoolMLPWithProj, dict]:
    """Train an attention-pooled MLP for one attribute.

    Args:
        X_patches: (N, num_patches, patch_dim) patch token arrays
        y: (N,) binary labels
        text_query: (1, text_dim) text embedding for this attribute
    """
    _seed_everything(seed)
    X_train, X_val, y_train, y_val = split_labeled_validation(X_patches, y, seed, validation_data)

    device = _get_device()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    text_q = text_query.to(device)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True,
    )

    model = AttnPoolMLPWithProj(patch_dim, text_dim, num_heads).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, text_q)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val_t, text_q).cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)
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
            val_acc = accuracy_score(y_val, preds)
            print(f"  [{attr}] epoch {epoch:3d}  val_acc={val_acc:.4f}  val_auc={val_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_val_t, text_q).cpu().numpy().ravel()
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
    }

    print(f"\n  [{attr}] Final val  acc={metrics['val_accuracy']:.4f}  "
          f"f1={metrics['val_f1']:.4f}  auc={metrics['val_auc']:.4f}")
    print(f"  [{attr}] {classification_report(y_val, preds, zero_division=0)}")

    return model, metrics


def prepare_attribute_patches(
    attr: str,
    labeled: list[dict],
    patch_tokens: np.ndarray,
    path_to_idx: dict[str, int],
    path_converter,
):
    """Build X_patches, y arrays for one attribute from patch tokens."""
    X_list, y_list, matched = [], [], []
    for entry in labeled:
        label = entry.get(attr)
        if label is None:
            continue
        rel_path = path_converter(entry["image"])
        idx = path_to_idx.get(rel_path)
        if idx is None:
            continue
        X_list.append(patch_tokens[idx])
        y_list.append(int(label))
        matched.append(entry["image"])

    return np.array(X_list), np.array(y_list), matched


def predict_remaining_attn(
    models: dict[str, AttnPoolMLPWithProj],
    text_queries: dict[str, torch.Tensor],
    patch_tokens: np.ndarray,
    records: pd.DataFrame,
    labeled_paths: set[str],
    attributes: list[str],
    path_converter,
) -> list[dict]:
    """Predict all attributes for images not in the labeled set using attention-pooled MLP."""
    device = _get_device()

    remaining_indices = []
    remaining_rel_paths = []
    for _, row in records.iterrows():
        if row["relative_path"] not in labeled_paths:
            remaining_indices.append(int(row["embedding_index"]))
            remaining_rel_paths.append(row["relative_path"])

    print(f"\nPredicting on {len(remaining_indices)} remaining images ...")

    X_remain = torch.tensor(
        patch_tokens[remaining_indices], dtype=torch.float32,
    ).to(device)

    predictions = {}
    for attr, model in models.items():
        model.eval()
        text_q = text_queries[attr].to(device)

        all_preds = []
        chunk_size = 512
        for i in range(0, len(X_remain), chunk_size):
            chunk = X_remain[i:i + chunk_size]
            with torch.no_grad():
                logits = model(chunk, text_q).cpu().numpy().ravel()
                probs = 1.0 / (1.0 + np.exp(-logits))
                preds = (probs >= 0.5).astype(int)
            all_preds.append(preds)
        predictions[attr] = np.concatenate(all_preds)

    results = []
    for i, rel_path in enumerate(remaining_rel_paths):
        image_name = path_converter(rel_path)
        entry = {"image": image_name}
        for attr in attributes:
            entry[attr] = int(predictions[attr][i])
        results.append(entry)

    return results
