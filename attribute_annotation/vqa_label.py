"""
VQA Auto-Labeling Script (API Mode)

Uses OpenAI-compatible API (DeepSeek etc.) to perform visual question answering
on images. Sends each image with a prompt and saves structured replies as JSONL.

Usage:
    python vqa_label.py                    # normal run
    python vqa_label.py --dry-run          # list images without calling API
    python vqa_label.py --config other.yaml # use a different config file
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI
from tqdm import tqdm


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = EXPERIMENT_ROOT / "dataset" / "tasks"

# 多线程并发写入同一个 JSONL 时用于保护文件追加
_write_lock = threading.Lock()


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = config.get("api_key", "")
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[error] API key not found. Set 'api_key' in config.yaml or export DASHSCOPE_API_KEY.")
        sys.exit(1)
    config["api_key"] = api_key
    return config


def load_image_list(images_dir: str, only_basenames: set[str] | None = None) -> list[str]:
    img_dir = Path(images_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"image directory not found: {images_dir}")
    iterator = img_dir.rglob("*") if only_basenames is not None else img_dir.iterdir()
    paths = sorted(
        str(p.resolve()) for p in iterator
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        and (only_basenames is None or p.name in only_basenames)
    )
    if only_basenames is not None:
        found = [os.path.basename(path) for path in paths]
        duplicates = sorted({name for name in found if found.count(name) > 1})
        if duplicates:
            raise ValueError(f"requested image basenames are ambiguous: {duplicates[:10]}")
        missing = sorted(only_basenames - set(found))
        if missing:
            print(f"[warn] requested images not found: {len(missing)}; examples={missing[:5]}")
    return paths


def load_finished(output_file: str) -> set[str]:
    finished = set()
    if not os.path.exists(output_file):
        return finished
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("answer") != "[FAILED]":
                    finished.add(record["image"])
            except json.JSONDecodeError:
                continue
    return finished


def count_failed(output_file: str) -> int:
    """统计输出文件中 answer == '[FAILED]' 的记录数。"""
    if not os.path.exists(output_file):
        return 0
    failed = 0
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("answer") == "[FAILED]":
                    failed += 1
            except json.JSONDecodeError:
                continue
    return failed


def generate_output_file(config: dict) -> str:
    output_dir = config.get("output_dir", "results/")
    os.makedirs(output_dir, exist_ok=True)

    model_name = config.get("model", "unknown").replace("/", "_")
    dataset_name = Path(config["images_dir"]).resolve().name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{dataset_name}_{model_name}_results.jsonl"
    filepath = os.path.join(output_dir, filename)

    existing = sorted(
        Path(output_dir).glob(f"*_{dataset_name}_{model_name}_results.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if existing:
        latest = str(existing[0])
        finished = load_finished(latest)
        if finished:
            print(f"[init] resuming from: {latest} ({len(finished)} done)")
            return latest

    print(f"[init] output file: {filepath}")
    return filepath


def append_result(output_file: str, record: dict):
    with _write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(config: dict, image_name: str) -> str:
    prompt = config.get("prompt", "")
    attributes = config.get("attributes", "")
    prompt = prompt.replace("{image_id}", image_name)
    prompt = prompt.replace("{attributes}", attributes.strip())
    return prompt.strip()


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


def call_api(client: OpenAI, config: dict, image_path: str, prompt: str) -> str:
    b64 = encode_image_base64(image_path)
    mime = get_mime_type(image_path)
    data_url = f"data:{mime};base64,{b64}"

    response = client.chat.completions.create(
        model=config["model"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=config.get("max_tokens", 4096),
        timeout=config.get("timeout", 60),
    )
    return response.choices[0].message.content


def process_image(client: OpenAI, config: dict, image_path: str, output_file: str,
                  image_id: str | None = None) -> bool:
    image_id = image_id or os.path.basename(image_path)
    prompt = build_prompt(config, image_id)
    max_retries = config.get("max_retries", 3)

    for attempt in range(max_retries):
        try:
            reply = call_api(client, config, image_path, prompt)
            append_result(output_file, {
                "image": image_id,
                "model": config["model"],
                "answer": reply,
                "timestamp": datetime.now().isoformat(),
            })
            return True

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate" in error_str.lower()
            wait_time = (2 ** attempt) * (10 if is_rate_limit else 2)

            if attempt < max_retries - 1:
                print(f"  [retry {attempt + 1}/{max_retries}] {error_str[:80]}... waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                print(f"  [failed] {error_str[:120]}")
                append_result(output_file, {
                    "image": image_id,
                    "model": config["model"],
                    "answer": "[FAILED]",
                    "error": error_str[:500],
                    "timestamp": datetime.now().isoformat(),
                })
                return False


def _resolve_task_dir(task: str) -> Path:
    """Find a task dir under the new nested layout tasks/<group>/<task>/, falling back
    to the legacy flat tasks/<task>/. Task names are unique across dataset groups."""
    direct = TASKS_DIR / task
    if direct.is_dir():
        return direct
    matches = [p for p in TASKS_DIR.glob(f"*/{task}") if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"[error] ambiguous --task {task!r}: {[str(m) for m in matches]}")
    raise SystemExit(f"[error] task {task!r} not found under {TASKS_DIR}")


def _apply_overrides(config: dict, args) -> None:
    """Layer CLI overrides on top of config (priority: explicit flags > --task > config)."""
    if args.task:
        tdir = _resolve_task_dir(args.task)
        config["output_dir"] = str(tdir / "qa")
        attr_txt = tdir / "attributes.txt"
        if not attr_txt.is_file():
            attr_txt = tdir / "qa" / "attributes.txt"
        if not config.get("attributes") and attr_txt.is_file():
            config["attributes"] = attr_txt.read_text(encoding="utf-8").strip()
    if args.images_dir:
        config["images_dir"] = args.images_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.attributes_file:
        config["attributes"] = Path(args.attributes_file).read_text(encoding="utf-8").strip()
    if args.attributes:
        config["attributes"] = args.attributes
    if not config.get("attributes"):
        print("[warn] no 'attributes' set (config/--task/--attributes all empty); "
              "prompt {attributes} will be blank")


def main():
    parser = argparse.ArgumentParser(description="VQA Auto-Labeling (API Mode)")
    parser.add_argument("--config", default="config.yaml", help="config file path")
    parser.add_argument("--task", default=None,
                        help="task dir name (resolved under tasks/<group>/<task>); sets the "
                             "output directory and reads canonical attributes. Images still "
                             "come from --images-manifest or explicit --images-dir.")
    parser.add_argument("--images-dir", default=None, help="override images_dir")
    parser.add_argument("--output-dir", default=None, help="override output_dir")
    parser.add_argument("--attributes", default=None, help="override attributes string")
    parser.add_argument("--attributes-file", default=None,
                        help="read attributes string from this file (e.g. qa/attributes.txt)")
    parser.add_argument("--images-list", default=None,
                        help="JSON file with a list of basenames; only label those images "
                             "from images_dir (used by the harness to label just the L "
                             "candidate budget instead of the whole library)")
    parser.add_argument("--images-manifest", default=None,
                        help="JSON rows with explicit image_id and image_path from the "
                             "logical database; preferred over --images-list")
    parser.add_argument("--dry-run", action="store_true", help="list images without calling API")
    parser.add_argument("--workers", type=int, default=None, help="并发线程数 (>1 启用并发；默认读 config.concurrency 或 1)")
    args = parser.parse_args()

    config = load_config(args.config)
    _apply_overrides(config, args)
    if args.images_manifest:
        payload = json.loads(Path(args.images_manifest).read_text(encoding="utf-8-sig"))
        entries = []
        seen_ids = set()
        for row in payload:
            image_id = str(row["image_id"])
            image_path = str(Path(row["image_path"]).resolve())
            if image_id in seen_ids:
                raise SystemExit(f"[error] duplicate image_id in manifest: {image_id}")
            if not Path(image_path).is_file():
                raise SystemExit(f"[error] manifest image not found: {image_path}")
            seen_ids.add(image_id)
            entries.append({"image_id": image_id, "image_path": image_path})
        print(f"[init] loaded {len(entries)} logical database rows from {args.images_manifest}")
        images = [entry["image_path"] for entry in entries]
    else:
        only = None
        if args.images_list:
            only = set(json.loads(Path(args.images_list).read_text(encoding="utf-8-sig")))
            print(f"[init] restricting to {len(only)} basenames from {args.images_list}")
        images = load_image_list(config["images_dir"], only_basenames=only)
        entries = [{"image_id": os.path.basename(path), "image_path": path} for path in images]
    print(f"[init] found {len(images)} images in {config['images_dir']}")
    print(f"[init] model: {config['model']} @ {config['base_url']}")

    if args.dry_run:
        output_file = generate_output_file(config)
        done = load_finished(output_file)
        errors = count_failed(output_file)
        todo = [entry for entry in entries if entry["image_id"] not in done]
        print(f"[dry-run] done: {len(done)}, remaining: {len(todo)}, error: {errors}")
        for entry in todo[:20]:
            print(f"  {entry['image_id']}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    output_file = generate_output_file(config)
    done = load_finished(output_file)
    errors = count_failed(output_file)
    todo = [entry for entry in entries if entry["image_id"] not in done]
    print(f"[init] total: {len(images)}, done: {len(done)}, remaining: {len(todo)}, error: {errors}")

    if not todo:
        print("[init] all images already processed")
        return

    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    delay = config.get("delay", 1)
    workers = args.workers if args.workers is not None else config.get("concurrency", 1)
    workers = max(1, int(workers))
    success_count = 0
    failed_count = 0

    if workers == 1:
        pbar = tqdm(todo, desc="VQA labeling", unit="img")
        for entry in pbar:
            image_path = entry["image_path"]
            pbar.set_postfix(current=entry["image_id"][:30])

            ok = process_image(client, config, image_path, output_file, entry["image_id"])
            if ok:
                success_count += 1
            else:
                failed_count += 1

            time.sleep(delay)
        pbar.close()
    else:
        print(f"[init] concurrency enabled: {workers} workers")

        def worker(entry):
            ok = process_image(
                client, config, entry["image_path"], output_file, entry["image_id"]
            )
            if delay:
                time.sleep(delay)
            return ok

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, entry): entry for entry in todo}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="VQA labeling", unit="img")
            for fut in pbar:
                try:
                    ok = fut.result()
                except Exception as e:
                    ok = False
                    print(f"  [worker error] {str(e)[:120]}")
                if ok:
                    success_count += 1
                else:
                    failed_count += 1
            pbar.close()

    print(f"\n[done] success: {success_count}, failed: {failed_count}")
    print(f"[done] results saved to: {output_file}")


if __name__ == "__main__":
    main()
