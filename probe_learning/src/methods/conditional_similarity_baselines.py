"""CLAY and GeneCIS inference components for local retrieval tasks.

CLAY follows the official CVPR 2026 implementation: normalized text features
define a tangent-space condition subspace, image features are aligned to its
mean with two Householder reflections, and retrieval uses cosine similarity in
the projected tangent space.  The implementation below avoids materializing
the dense D x D projection matrix and scores large galleries in chunks.

GeneCIS follows the official CVPR 2023 Combiner architecture.  Its released
weights are tied to the released fine-tuned CLIP backbone and must not be used
with the project's existing CLIP-B/32 or SigLIP features.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLAY_SOURCE_URL = "https://github.com/kaist-ami/CLAY"
GENECIS_SOURCE_URL = "https://github.com/facebookresearch/genecis"


def l2_normalize_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(denom, 1e-12)


def _normalize_torch(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, p=2, dim=-1, eps=1e-12)


def log_map(mu: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Logarithmic map on the unit hypersphere at base point ``mu``."""
    mu = _normalize_torch(mu)
    values = _normalize_torch(values)
    dot = (values * mu).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(dot)
    tangent = values - dot * mu
    return theta * tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def householder_vector(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """Return unit ``u`` for H=I-2uu^T mapping source to target."""
    direction = source - target
    norm = direction.norm()
    if float(norm) < 1e-12:
        return None
    return direction / norm


def apply_householder(values: torch.Tensor, vector: torch.Tensor | None) -> torch.Tensor:
    if vector is None:
        return values
    return values - 2.0 * (values @ vector).unsqueeze(-1) * vector


@dataclass(frozen=True)
class ClaySubspace:
    mean: torch.Tensor
    basis: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])


def make_clay_subspace(text_features: torch.Tensor, max_rank: int = 50) -> ClaySubspace:
    """Build CLAY's manifold-aware textual subspace.

    The official code uses r=50.  Local binary condition banks can contain
    fewer than 50 prompts, so rank is capped at n_prompts-1 to avoid retaining
    arbitrary zero-singular-value directions.
    """
    text_features = _normalize_torch(text_features.float())
    if text_features.ndim != 2 or len(text_features) < 2:
        raise ValueError("CLAY needs at least two condition prompts")
    mean = _normalize_torch(text_features.mean(dim=0, keepdim=True))[0]
    tangent = log_map(mean, text_features)
    centered = tangent - tangent.mean(dim=0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    numerical_rank = int((singular > singular.max().clamp_min(1e-12) * 1e-6).sum())
    rank = min(int(max_rank), len(text_features) - 1, numerical_rank)
    if rank < 1:
        raise ValueError("CLAY condition prompt bank has zero numerical rank")
    return ClaySubspace(mean=mean, basis=vh[:rank].T.contiguous())


def _clay_alignment(
    image_mean: torch.Tensor,
    text_mean: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image_mean = _normalize_torch(image_mean)
    text_mean = _normalize_torch(text_mean)
    midpoint = image_mean + text_mean
    if float(midpoint.norm()) < 1e-9:
        raise ValueError("image and text means are antipodal; CLAY alignment is undefined")
    midpoint = _normalize_torch(midpoint)
    return householder_vector(image_mean, midpoint), householder_vector(midpoint, text_mean)


def _project_clay_chunk(
    values: torch.Tensor,
    text_mean: torch.Tensor,
    basis: torch.Tensor,
    first_reflection: torch.Tensor | None,
    second_reflection: torch.Tensor | None,
) -> torch.Tensor:
    values = apply_householder(values, first_reflection)
    values = apply_householder(values, second_reflection)
    tangent = log_map(text_mean, values)
    return _normalize_torch(tangent @ basis)


@torch.inference_mode()
def clay_scores(
    image_features: np.ndarray,
    query_indices: Iterable[int],
    text_features: np.ndarray | torch.Tensor,
    *,
    max_rank: int = 50,
    chunk_size: int = 8192,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, int]:
    """Score every image using the mean of projected query-image features."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    images_np = l2_normalize_np(image_features)
    query_indices = np.asarray(list(query_indices), dtype=np.int64)
    if len(query_indices) == 0:
        raise ValueError("CLAY needs at least one query image")
    if query_indices.min() < 0 or query_indices.max() >= len(images_np):
        raise IndexError("CLAY query index is outside the image feature table")

    text_tensor = torch.as_tensor(text_features, dtype=torch.float32, device=device)
    subspace = make_clay_subspace(text_tensor, max_rank=max_rank)
    image_mean = torch.as_tensor(images_np.mean(axis=0), dtype=torch.float32, device=device)
    first, second = _clay_alignment(image_mean, subspace.mean)

    query = torch.as_tensor(images_np[query_indices], dtype=torch.float32, device=device)
    query_projected = _project_clay_chunk(
        query, subspace.mean, subspace.basis, first, second,
    )
    prototype = _normalize_torch(query_projected.mean(dim=0, keepdim=True))[0]

    scores = np.empty(len(images_np), dtype=np.float32)
    for start in range(0, len(images_np), chunk_size):
        chunk = torch.as_tensor(
            images_np[start:start + chunk_size], dtype=torch.float32, device=device,
        )
        projected = _project_clay_chunk(
            chunk, subspace.mean, subspace.basis, first, second,
        )
        scores[start:start + len(chunk)] = (projected @ prototype).float().cpu().numpy()
    return scores, subspace.rank


class GeneCISCombiner(nn.Module):
    """Official GeneCIS/CLIP4CIR Combiner architecture."""

    def __init__(self, feature_dim: int, projection_dim: int = 2560, hidden_dim: int = 5120):
        super().__init__()
        self.text_projection_layer = nn.Linear(feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(feature_dim, projection_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, feature_dim)
        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        text_projected = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected = self.dropout2(F.relu(self.image_projection_layer(image_features)))
        raw = torch.cat((text_projected, image_projected), dim=-1)
        combined = self.dropout3(F.relu(self.combiner_layer(raw)))
        scalar = self.dynamic_scalar(raw)
        output = self.output_layer(combined) + scalar * text_features + (1 - scalar) * image_features
        return _normalize_torch(output)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        expected = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 128 * 1024 * 1024
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                total = f"/{expected / 2**20:.0f} MiB" if expected else ""
                print(f"[download] {downloaded / 2**20:.0f} MiB{total}", flush=True)
                next_report += 128 * 1024 * 1024
    if expected and downloaded != expected:
        raise IOError(f"incomplete download for {url}: expected {expected}, got {downloaded}")
    if downloaded == 0:
        raise IOError(f"empty download for {url}")
    temporary.replace(destination)
    return destination


def ordered_paths_hash(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_cache_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
