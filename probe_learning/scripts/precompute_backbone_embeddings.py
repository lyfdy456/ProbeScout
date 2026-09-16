"""Precompute non-CLIP pooled embeddings aligned to an existing records.csv.

This script is intentionally records-driven: it does not rebuild dataset metadata.
It reads the current processed records.csv, resolves each relative_path to an image,
and writes one additional embedding file beside the existing clip_embedding.npy.

Examples:
  uv run python scripts/precompute_backbone_embeddings.py --dataset cars --backbone dinov2
  uv run python scripts/precompute_backbone_embeddings.py --dataset celeba --backbone siglip
  uv run python scripts/precompute_backbone_embeddings.py --dataset awa2 --backbone dinov2
  uv run python scripts/precompute_backbone_embeddings.py --dataset awa2 --task task_zebra --backbone dinov2
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoModel, SiglipModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import paths as P  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

BACKBONE_SPECS = {
    "dinov2": {
        "model": "facebook/dinov2-base",
        "out": "dinov2_embedding.npy",
        "kind": "dinov2",
    },
    "siglip": {
        "model": "google/siglip-base-patch16-224",
        "out": "siglip_embedding.npy",
        "kind": "siglip",
    },
}


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dataset_paths(dataset: str, task: str | None) -> tuple[Path, Path, list[Path], bool]:
    """Return records/embedding dir, primary image root, and fallback roots."""
    if dataset == "cars":
        return P.CARS_PROCESSED_DIR, P.CARS_IMAGE_ROOT, [], False
    if dataset == "sun":
        return P.SUN_PROCESSED_DIR, P.SUN_ROOT / "images", [], False
    if dataset == "hico":
        return P.HICO_PROCESSED_DIR, P.HICO_IMAGES_DIR, [], False
    if dataset == "cub":
        return P.PROCESSED_DIR, P.CUB_ROOT, [], False
    if dataset == "celeba":
        return P.CELEBA_PROCESSED_DIR, P.CELEBA_IMAGES_DIR, [], False
    if dataset == "awa2":
        if task:
            troot = P.task_root("awa2", task)
            return troot / "processed", troot / "query_pics", [], True
        return P.AWA2_PROCESSED_DIR, P.database_dir("awa2"), [P.AWA2_IMAGES_DIR], False
    raise SystemExit(f"unsupported dataset: {dataset}")


def resolve_image(
    rel: str,
    image_root: Path,
    fallbacks: list[Path],
    class_name: str | None = None,
) -> Path:
    rel_path = Path(str(rel).replace("\\", "/"))
    candidates = [image_root / rel_path]
    clean_class = str(class_name or "").strip()
    if clean_class and clean_class.lower() != "nan":
        candidates.append(image_root / clean_class / rel_path.name)
    candidates += [root / rel_path for root in fallbacks]
    if clean_class and clean_class.lower() != "nan":
        candidates += [root / clean_class / rel_path.name for root in fallbacks]
    candidates += [root / rel_path.name for root in fallbacks]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"cannot resolve image {rel!r}; tried {candidates[:3]}")


def load_model(backbone: str, device: str):
    spec = BACKBONE_SPECS[backbone]
    model_name = spec["model"]
    if spec["kind"] == "dinov2":
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    elif spec["kind"] == "siglip":
        model = SiglipModel.from_pretrained(model_name).to(device).eval()
    else:
        raise ValueError(spec["kind"])
    return model


def preprocessing_spec(backbone: str, model) -> tuple[int, torch.Tensor, torch.Tensor]:
    if BACKBONE_SPECS[backbone]["kind"] == "dinov2":
        size = int(getattr(model.config, "image_size", 224))
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    else:
        size = int(model.config.vision_config.image_size)
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    return size, mean, std


def preprocess_path(path: Path, size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Load and resize one image; safe to call from a worker thread."""
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1)
    return (ten - mean) / std


def preprocess_images(
    backbone: str,
    model,
    images: list[Image.Image],
    device: str,
) -> torch.Tensor:
    """Preprocess already-open PIL images with the exact backbone transform.

    Patch-token extraction operates on in-memory images rather than paths.  It
    must use the same resize and normalization constants as pooled embedding
    extraction so the two feature files remain directly comparable.
    """
    size, mean, std = preprocessing_spec(backbone, model)
    tensors = []
    for image in images:
        resized = image.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append((ten - mean) / std)
    if not tensors:
        return torch.empty((0, 3, size, size), dtype=torch.float32, device=device)
    return torch.stack(tensors, dim=0).to(device)


def encode_paths(backbone: str, model, paths: list[Path], device: str,
                 executor: ThreadPoolExecutor) -> np.ndarray:
    size, mean, std = preprocessing_spec(backbone, model)
    tensors = list(executor.map(
        lambda path: preprocess_path(path, size, mean, std),
        paths,
    ))
    pixel_values = torch.stack(tensors, dim=0).to(device)
    with torch.inference_mode():
        if BACKBONE_SPECS[backbone]["kind"] == "dinov2":
            out = model(pixel_values=pixel_values)
            feats = getattr(out, "pooler_output", None)
            if feats is None:
                feats = out.last_hidden_state[:, 0]
        else:
            feats = model.get_image_features(pixel_values=pixel_values)
            if not isinstance(feats, torch.Tensor):
                image_embeds = getattr(feats, "image_embeds", None)
                feats = image_embeds if image_embeds is not None else getattr(feats, "pooler_output", None)
                if feats is None:
                    raise TypeError("SigLIP get_image_features returned no image_embeds/pooler_output")
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return feats.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cars", "sun", "cub", "hico", "awa2", "celeba"])
    ap.add_argument("--task", default=None, help="required for task-specific datasets such as AwA2")
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONE_SPECS))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel image loading/resizing workers")
    ap.add_argument("--limit", type=int, default=0, help="encode first N rows only for smoke testing")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing embedding file")
    args = ap.parse_args()

    proc_dir, image_root, fallbacks, query_only = dataset_paths(args.dataset, args.task)
    records_path = proc_dir / ("query_records.csv" if query_only else "records.csv")
    if not records_path.is_file():
        raise FileNotFoundError(
            f"{records_path.name} not found: {records_path}. "
            "For AwA2 task query embeddings, run scripts/precompute_backbone_embeddings.py (requires prepared records) first."
        )
    records = pd.read_csv(records_path)
    if args.limit > 0:
        records = records.head(args.limit).copy()
        print(f"[smoke] limited to {len(records)} rows")

    out_name = BACKBONE_SPECS[args.backbone]["out"]
    if query_only:
        out_name = "query_" + out_name
    if args.limit > 0:
        stem = Path(out_name).stem
        out_name = f"{stem}_limit{args.limit}.npy"
    out_path = proc_dir / out_name
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists; pass --overwrite to replace it")

    device = choose_device()
    print(f"dataset={args.dataset} task={args.task or '-'} backbone={args.backbone} device={device}")
    print(f"records={records_path} rows={len(records)} image_root={image_root}")

    model = load_model(args.backbone, device)

    sample_row = records.iloc[0]
    sample_path = resolve_image(
        str(sample_row["relative_path"]), image_root, fallbacks,
        sample_row.get("class_name"),
    )
    rels = records["relative_path"].astype(str).tolist()
    classes = (records["class_name"].astype(str).tolist()
               if "class_name" in records.columns else [None] * len(records))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        dim = encode_paths(args.backbone, model, [sample_path], device, executor).shape[1]
        emb = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=np.float32, shape=(len(records), dim),
        )
        cursor = 0
        for start in tqdm(range(0, len(rels), args.batch_size), desc=f"{args.backbone} embeddings"):
            end = start + args.batch_size
            batch_rels = rels[start:end]
            batch_classes = classes[start:end]
            batch_paths = [
                resolve_image(rel, image_root, fallbacks, class_name)
                for rel, class_name in zip(batch_rels, batch_classes)
            ]
            feats = encode_paths(args.backbone, model, batch_paths, device, executor)
            emb[cursor:cursor + len(batch_rels)] = feats
            cursor += len(batch_rels)
    emb.flush()
    print(f"saved {out_path} shape={emb.shape}")


if __name__ == "__main__":
    main()
