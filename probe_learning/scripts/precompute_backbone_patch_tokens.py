"""Precompute patch tokens for patch-attention backbones.

The output is aligned to the dataset's existing records.csv. For AwA2 task
queries, pass --task to write query_<backbone>_patch_tokens.npy beside the
task-level query_records.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import SiglipModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.precompute_backbone_embeddings import dataset_paths, preprocess_images, resolve_image  # noqa: E402
from src.utils import paths as P  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

PATCH_SPECS = {
    "siglip": {
        "model": "google/siglip-base-patch16-224",
        "out": P.PATCH_TOKEN_FILES["siglip"],
    },
}


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(backbone: str, device: str):
    spec = PATCH_SPECS[backbone]
    if backbone == "siglip":
        return SiglipModel.from_pretrained(spec["model"]).to(device).eval()
    raise ValueError(backbone)


def encode_patch_batch(backbone: str, model, images: list[Image.Image], device: str) -> np.ndarray:
    pixel_values = preprocess_images(backbone, model, images, device)
    with torch.inference_mode():
        if backbone == "siglip":
            outputs = model.vision_model(pixel_values=pixel_values)
            tokens = outputs.last_hidden_state
        else:
            raise ValueError(backbone)
    return tokens.detach().cpu().numpy()


def load_completed(progress_path: Path) -> set[int]:
    if not progress_path.is_file():
        return set()
    done: set[int] = set()
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "done":
            done.add(int(rec["start"]))
    return done


def append_done(progress_path: Path, start: int, end: int) -> None:
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"start": int(start), "end": int(end), "status": "done"}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cars", "sun", "cub", "hico", "awa2", "celeba"])
    ap.add_argument("--task", default=None, help="AwA2 task query patch tokens")
    ap.add_argument("--backbone", required=True, choices=sorted(PATCH_SPECS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--limit", type=int, default=0, help="encode first N rows only for smoke testing")
    ap.add_argument("--force", action="store_true", help="overwrite existing patch file/progress")
    args = ap.parse_args()

    proc_dir, image_root, fallbacks, query_only = dataset_paths(args.dataset, args.task)
    records_path = proc_dir / ("query_records.csv" if query_only else "records.csv")
    if not records_path.is_file():
        raise FileNotFoundError(f"records not found: {records_path}")
    records = pd.read_csv(records_path)
    if args.limit > 0:
        records = records.head(args.limit).copy()
        print(f"[smoke] limited to {len(records)} rows")

    out_name = PATCH_SPECS[args.backbone]["out"]
    if query_only:
        out_name = "query_" + out_name
    if args.limit > 0:
        out_name = f"{Path(out_name).stem}_limit{args.limit}.npy"
    out_path = proc_dir / out_name
    progress_path = proc_dir / f"{Path(out_name).stem}.progress.jsonl"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists; pass --force to replace it")
    if args.force and progress_path.exists():
        progress_path.unlink()

    device = choose_device()
    print(f"dataset={args.dataset} task={args.task or '-'} backbone={args.backbone} device={device}")
    print(f"records={records_path} rows={len(records)} image_root={image_root} dtype={args.dtype}")

    model = load_model(args.backbone, device)
    sample_row = records.iloc[0]
    sample_path = resolve_image(
        str(sample_row["relative_path"]), image_root, fallbacks,
        sample_row.get("class_name"),
    )
    with Image.open(sample_path) as im:
        sample = encode_patch_batch(args.backbone, model, [im.convert("RGB")], device)
    token_shape = tuple(int(x) for x in sample.shape[1:])
    dtype = np.float16 if args.dtype == "float16" else np.float32
    patches = np.lib.format.open_memmap(
        str(out_path),
        mode="w+",
        dtype=dtype,
        shape=(len(records), *token_shape),
    )

    completed = load_completed(progress_path)
    rels = records["relative_path"].astype(str).tolist()
    classes = (records["class_name"].astype(str).tolist()
               if "class_name" in records.columns else [None] * len(records))
    total_batches = (len(rels) + args.batch_size - 1) // args.batch_size
    for start in tqdm(range(0, len(rels), args.batch_size),
                      desc=f"{args.backbone} patch tokens",
                      total=total_batches):
        end = min(start + args.batch_size, len(rels))
        if start in completed:
            continue
        images = []
        try:
            for rel, class_name in zip(rels[start:end], classes[start:end]):
                with Image.open(resolve_image(rel, image_root, fallbacks, class_name)) as im:
                    images.append(im.convert("RGB"))
            tokens = encode_patch_batch(args.backbone, model, images, device)
            patches[start:end] = tokens.astype(dtype, copy=False)
            patches.flush()
            append_done(progress_path, start, end)
        finally:
            for im in images:
                im.close()

    patches.flush()
    print(f"saved {out_path} shape={patches.shape} dtype={patches.dtype}")


if __name__ == "__main__":
    main()
