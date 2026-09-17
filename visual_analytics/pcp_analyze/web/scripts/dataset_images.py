"""Resolve user-selected image folders; task arrays remain in the repository."""
import json
from pathlib import Path

DIRECTORIES = {"cars": "stanford_cars/images", "hico": "HICO/images", "celeba": "CelebA/img_celeba"}


def image_root(repo: Path, dataset: str) -> Path:
    settings = repo / ".probescout/settings.json"
    roots = json.loads(settings.read_text(encoding="utf-8")).get("imageRoots", {}) if settings.is_file() else {}
    selected = roots.get(dataset)
    return (repo / selected).resolve() if selected else repo / "dataset/raw" / DIRECTORIES[dataset]
