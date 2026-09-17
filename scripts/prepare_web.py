"""Prepare installed Main17 task packages and local images for the Web interface.

This command never downloads datasets, runs VQA, or changes frozen score arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "visual_analytics/pcp_analyze/web"
sys.path.insert(0, str(WEB / "scripts"))
from dataset_images import DIRECTORIES, image_root


def inside(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise RuntimeError(f"Path escapes its package root: {relative}")
    return path


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_package(dataset: str):
    pin = read(ROOT / "manifests/task_packages.json")["datasets"][dataset]
    path = ROOT / "dataset/tasks" / dataset / "manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Extract {dataset}.zip from ProbeScout-tasks into the repository root first")
    if sha(path) != pin["manifestSha256"]:
        raise RuntimeError(f"{dataset}: task package version differs from this source release")
    package = read(path)
    for relative, expected in package["files"].items():
        file = inside(ROOT, relative)
        if not file.is_file() or sha(file) != expected:
            raise RuntimeError(f"Missing or modified task asset: {relative}")
    catalog = read(path.parent / "catalog.json")
    print(f"{dataset}: verified {len(package['tasks'])} tasks and {len(package['files'])} assets", flush=True)
    return catalog


def image_jobs(catalog, without_images=False):
    jobs = []
    for dataset in catalog["datasets"]:
        raw = image_root(ROOT, dataset["id"])
        for task in dataset["tasks"]:
            bundle = inside(WEB / "public", task["dataRoot"].lstrip("/"))
            manifest = read(bundle / "manifest.json")
            ids = read(inside(bundle, manifest["files"]["imageIds"]["path"]))
            if not without_images:
                missing = [name for name in ids if not inside(raw, name).is_file()]
                if missing:
                    raise RuntimeError(f"{dataset['id']}: {len(missing)} images missing under {raw}; first: {missing[0]}")
                for query in manifest["query"]["images"]:
                    source = inside(raw, query["imageId"])
                    if sha(source) != query["sha256"]:
                        raise RuntimeError(f"Query image differs from the released dataset: {source}")
            jobs.append((raw, bundle, manifest, ids))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=tuple(DIRECTORIES), required=True,
                        help="Enable exactly these installed datasets (one or more)")
    parser.add_argument("--check-only", action="store_true", help="Verify packages/images without writing")
    parser.add_argument("--without-images", action="store_true",
                        help="Prepare scores/plots/training only; image galleries require original images")
    parser.add_argument("--atlas-workers", type=int, default=4)
    parser.add_argument("--force-atlases", action="store_true", help="Regenerate local thumbnail images")
    args = parser.parse_args()
    datasets = list(dict.fromkeys(args.dataset))
    catalogs = [verify_package(dataset) for dataset in datasets]
    catalog = {**catalogs[0], "datasets": [d for c in catalogs for d in c["datasets"]]}
    catalog["taskCount"] = sum(len(d["tasks"]) for d in catalog["datasets"])
    jobs = image_jobs(catalog, args.without_images)
    if args.check_only:
        print("Task packages and requested inputs verified.")
        return
    if not args.without_images:
        from export_web_data import build_atlases
        done = {}
        for raw, bundle, manifest, ids in jobs:
            config = manifest["thumbnails"]
            directory = inside(WEB / "public", str((bundle / config["directory"]).relative_to(WEB / "public")))
            identity = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if directory in done and done[directory] != identity:
                raise RuntimeError("Shared thumbnail directory has inconsistent image order")
            if directory not in done:
                args.webp_quality = config.get("webpQuality", 62)
                result = build_atlases(ids, raw, bundle, config, args)
                print(f"Thumbnails: {result['generatedAtlases']} generated, {result['reusedAtlases']} reused", flush=True)
                if result["missingSourceCount"]:
                    raise RuntimeError("Some images could not be decoded; fix them and use --force-atlases")
                done[directory] = identity
            for query in manifest["query"]["images"]:
                target = inside(bundle, query["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(inside(raw, query["imageId"]), target)
    target = WEB / "public/data/catalog.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(f"Enabled {catalog['taskCount']} tasks: {', '.join(datasets)}.")
    if args.without_images:
        print("Image previews are unavailable. Rerun without --without-images after adding original images.")
    print("Run npm run dev in visual_analytics/pcp_analyze/web; open http://localhost:3000")


if __name__ == "__main__":
    main()
