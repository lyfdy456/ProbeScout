"""User-defined tasks, kept separate from the immutable Main17 asset packages."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

DIRECTORIES = {"cars": "stanford_cars", "hico": "HICO", "celeba": "CelebA"}
PROTOCOL = "probescout-local-task-v2"
SUPPORTED_PROTOCOLS = {"probescout-local-task-v1", PROTOCOL}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_input(path, root):
    """Validate exact image IDs and complete binary supervision before any training."""
    from sklearn.model_selection import train_test_split
    path, root = Path(path).resolve(), Path(root).resolve()
    config = read(path)
    config.pop("query_text", None)  # Accepted only for older input files; text comes from attributes.
    if set(config) != {"name", "dataset", "query_images", "attributes", "labels"}:
        raise ValueError("task.json requires name, dataset, query_images, attributes, labels")
    if not isinstance(config["name"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", config["name"]):
        raise ValueError("name must be a lowercase slug, at most 48 characters")
    if config["dataset"] not in DIRECTORIES:
        raise ValueError("Supported datasets: cars, hico, celeba")
    attrs = config["attributes"]
    if not isinstance(attrs, list) or not 2 <= len(attrs) <= 5:
        raise ValueError("Use 2 to 5 attributes")
    for attr in attrs:
        if (not isinstance(attr, dict) or set(attr) != {"id", "name"}
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", str(attr["id"]))
                or attr["id"] in {"joint", "image_id"}
                or not isinstance(attr["name"], str) or not attr["name"].strip()):
            raise ValueError("Each attribute needs a safe id and a nonempty name")
    if len({a["id"] for a in attrs}) != len(attrs) or len({a["name"] for a in attrs}) != len(attrs):
        raise ValueError("Attribute IDs and names must be unique")
    processed = root / "dataset/raw" / DIRECTORIES[config["dataset"]] / "processed"
    records_path = processed / "records.csv"
    with records_path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    ids = [row["relative_path"] for row in records]
    if ([int(row["embedding_index"]) for row in records] != list(range(len(ids)))
            or len(set(ids)) != len(ids) or len(ids) < 101):
        raise ValueError("records.csv must retain the complete canonical embedding order")
    for value in ids:
        if "\\" in value or Path(value).is_absolute() or ".." in Path(value).parts or ":" in value:
            raise ValueError("Image IDs must be safe forward-slash relative paths")
    lookup = {name: row for row, name in enumerate(ids)}
    query = config["query_images"]
    if (not isinstance(query, list) or not query or any(not isinstance(q, str) or q not in lookup for q in query)
            or len(set(query)) != len(query)):
        raise ValueError("query_images must contain unique image IDs from records.csv")
    label_path = (path.parent / str(config["labels"])).resolve()
    if not label_path.is_relative_to(path.parent):
        raise ValueError("labels must refer to a CSV inside the task input folder")
    with label_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = ["image_id", *[a["id"] for a in attrs]]
        if reader.fieldnames != columns:
            raise ValueError(f"labels.csv columns must be {columns}, in this order")
        labels = []
        for line, item in enumerate(reader, 2):
            if (set(item) != set(columns) or item["image_id"] not in lookup
                    or item["image_id"] in query or any(item[a["id"]] not in ("0", "1") for a in attrs)):
                raise ValueError(f"Invalid labels.csv row {line}: use known non-query IDs and explicit 0/1 for every attribute")
            labels.append([lookup[item["image_id"]], *[int(item[a["id"]]) for a in attrs]])
    labels.sort(key=lambda item: item[0])
    if not labels or len({r[0] for r in labels}) != len(labels):
        raise ValueError("Supervision must be nonempty, with no duplicate images")
    truth = np.asarray([row[1:] for row in labels], dtype=np.uint8)
    joint = truth.prod(axis=1)
    if min(np.count_nonzero(joint == v) for v in (0, 1)) < 2:
        raise ValueError("Joint needs at least two positive and two negative labeled images")
    fit, val = train_test_split(np.arange(len(labels)), test_size=.2, random_state=0, stratify=joint)
    for a, attr in enumerate(attrs):
        if min(np.count_nonzero(truth[fit, a] == v) for v in (0, 1)) < 2 or set(truth[val, a]) != {0, 1}:
            raise ValueError(f"{attr['id']}: need at least two positives/negatives in fit and both classes in the fixed 20% Val; provide more labels")
    result = {"protocol": PROTOCOL, "config": {k: v for k, v in config.items() if k != "labels"},
              "recordsSha256": sha(records_path), "labels": labels,
              "validationRows": sorted(labels[int(i)][0] for i in val),
              "queryRows": [lookup[name] for name in query]}
    return result, ids, processed


def available(service, task_id):
    return getattr(service.task(task_id), "manifest", {}).get("localTask", {}).get("protocol") in SUPPORTED_PROTOCOLS


def settings(service, task_id):
    spec = service.task(task_id).manifest["localTask"]
    root = service.web_root.parents[2].resolve()
    path = (root / spec["path"]).resolve()
    allowed = (service.web_root.parent / "runtime/local-tasks").resolve()
    if not path.is_relative_to(allowed) or sha(path) != spec["sha256"]:
        raise RuntimeError("Local task snapshot is missing or changed")
    value = read(path)
    if value.get("taskId") != task_id or value.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise RuntimeError("Local task identity mismatch")
    return value


def adapter(service, task_id):
    import sys
    import pandas as pd
    root = service.web_root.parents[2]
    sys.path.insert(0, str(root / "probe_learning"))
    from src.data.dataset_adapter import DatasetAdapter
    value = settings(service, task_id)
    config = value["config"]
    processed = root / "dataset/raw" / DIRECTORIES[config["dataset"]] / "processed"
    if sha(processed / "records.csv") != value["recordsSha256"]:
        raise RuntimeError("Local task canonical records changed")
    for filename, expected in value.get("featureFiles", {}).items():
        stat = (processed / filename).stat()
        if {"bytes": stat.st_size, "mtimeNs": stat.st_mtime_ns} != expected:
            raise RuntimeError("Local task features were replaced; run the task command to create a new version")
    records = pd.read_csv(processed / "records.csv")
    if records["relative_path"].tolist() != list(service.bundle(task_id).image_ids):
        raise RuntimeError("Local task image order changed")
    attrs = config["attributes"]
    return DatasetAdapter(dataset=config["dataset"], task=service.task(task_id).task_name,
        attrs=[a["name"] for a in attrs], key_slugs=[a["id"] for a in attrs],
        joint_label=" and ".join(a["name"] for a in attrs), emb_path=processed/"siglip_embedding.npy",
        patch_path=processed/"siglip_patch_tokens.npy", records=records,
        raw_jsonl_path=service.web_root.parent / "runtime/local-tasks" / task_id / "input.json", vqa_to_key=lambda text: text,
        query_idx=value["queryRows"], raw_images_dir=processed.parent / ("img_celeba" if config["dataset"] == "celeba" else "images"),
        gt_by_attr={a["name"]: {} for a in attrs},
        task_root=service.web_root.parent / "runtime/local-tasks" / task_id)


def original_supervision(service, task_id, target_id):
    from tuning_supervision import OriginalSupervision
    value = settings(service, task_id)
    attrs = [a["id"] for a in value["config"]["attributes"]]
    rows = np.asarray([r[0] for r in value["labels"]], dtype=np.int64)
    matrix = np.asarray([r[1:] for r in value["labels"]], dtype=np.uint8)
    labels = matrix.prod(axis=1).astype(np.uint8) if target_id == "joint" else matrix[:, attrs.index(target_id)]
    mask = np.frombuffer(service.bundle(task_id).development_mask, dtype=np.uint8)[rows].astype(bool)
    from val_isolation import digest
    audit = {"recoveredSupervisionHash": digest(value["labels"]), "labelSource": "user-provided-supervision",
             "originalSupervisionCount": len(rows), "developmentCount": int(mask.sum()),
             "positiveCount": int(labels[mask].sum()), "negativeCount": int(mask.sum()-labels[mask].sum()),
             "excludedNonDevelopmentCount": int((~mask).sum())}
    return OriginalSupervision(rows[mask], labels[mask], rows, labels, audit)
