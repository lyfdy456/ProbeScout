"""End-to-end 3-stage VQA attribute extraction from query pics.

Mirrors the manual workflow with vqa_prompts_v2.md:
  Stage 1  per-image detailed description (VLM, image+text)
  Stage 2  summarize shared attributes across descriptions (VLM, text-only -> JSON)
  Stage 3  auto-filter to a concise final attribute set -> {attr1, attr2, ...}

Outputs into the task's qa/ folder:
  - stage1_descriptions.json
  - stage2_summary.json
  - attributes.txt        (the final {a, b, c} string, ready to paste into a labeling config)
  - answer_auto.md        (human-readable summary mirroring answer.md)

Usage:
  python extract_attributes.py --config config_attributes.yaml
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI

from vqa_label import load_config, load_image_list, encode_image_base64, get_mime_type

_print_lock = threading.Lock()


def _safe_print(*a):
    with _print_lock:
        print(*a)


def call_vlm(client, model, prompt, image_path=None, max_tokens=2048, timeout=90, max_retries=4):
    """Single VLM call. If image_path is None -> text-only."""
    if image_path is not None:
        b64 = encode_image_base64(image_path)
        mime = get_mime_type(image_path)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = [{"type": "text", "text": prompt}]

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa
            last_err = str(e)
            import time
            time.sleep((2 ** attempt) * 2)
    raise RuntimeError(f"VLM call failed after {max_retries} tries: {last_err}")


def parse_json_block(text: str) -> dict:
    """Extract the first JSON object from a possibly fenced model reply."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in reply:\n{text[:300]}")
    return json.loads(t[start:end + 1])


def stage1_describe(client, cfg, images):
    """Per-image descriptions (concurrent)."""
    model = cfg["model"]
    prompt = cfg["prompt_stage1"].strip()
    workers = max(1, int(cfg.get("workers", 5)))
    descriptions = {}

    def work(idx_path):
        idx, path = idx_path
        qid = f"query_{idx + 1}"
        desc = call_vlm(client, model, prompt, image_path=path,
                        max_tokens=cfg.get("max_tokens", 2048),
                        timeout=cfg.get("timeout", 90),
                        max_retries=cfg.get("max_retries", 4))
        _safe_print(f"  [stage1] {qid} ({os.path.basename(path)}) -> {len(desc)} chars")
        return qid, os.path.basename(path), desc.strip()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, ip): ip for ip in enumerate(images)}
        for fut in as_completed(futs):
            qid, fname, desc = fut.result()
            descriptions[qid] = {"file": fname, "description": desc}
    # keep query order
    return {f"query_{i+1}": descriptions[f"query_{i+1}"] for i in range(len(images))}


def stage2_summarize(client, cfg, descriptions):
    model = cfg["model"]
    blocks = []
    for qid, d in descriptions.items():
        blocks.append(f"{qid} ({d['file']}):\n{d['description']}")
    desc_text = "\n\n".join(blocks)
    prompt = cfg["prompt_stage2"].replace("{descriptions}", desc_text)
    reply = call_vlm(client, model, prompt, image_path=None,
                     max_tokens=cfg.get("max_tokens", 2048),
                     timeout=cfg.get("timeout", 90),
                     max_retries=cfg.get("max_retries", 4))
    return parse_json_block(reply)


def stage3_filter(summary, cfg, num_queries):
    """Auto-select the final concise attribute set.

    Hard rules:
      - attribute must be present in EVERY query image
        (unclear_or_absent_images empty AND visible_in_images covers all queries)
      - verifiability must be 'high' (medium/low rejected)
    """
    attrs = summary.get("attributes", [])
    allow_subj = set(cfg.get("allow_subjectivity", ["low", "medium"]))
    allow_ver = set(cfg.get("allow_verifiability", ["high"]))
    max_attrs = int(cfg.get("max_attrs", 6))

    type_rank = {"concrete": 0, "visual_quality": 1, "abstract": 2}
    ver_rank = {"high": 0, "medium": 1, "low": 2}

    def present_in_all(a):
        if a.get("unclear_or_absent_images"):
            return False
        return len(set(a.get("visible_in_images", []))) >= num_queries

    kept = [
        a for a in attrs
        if present_in_all(a)
        and a.get("verifiability", "high") in allow_ver
        and a.get("subjectivity", "low") in allow_subj
    ]
    kept.sort(key=lambda a: (
        type_rank.get(a.get("type", "abstract"), 3),
        ver_rank.get(a.get("verifiability", "low"), 3),
    ))
    return kept[:max_attrs]


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = EXPERIMENT_ROOT / "dataset" / "tasks"


def _resolve_task_dir(task: str) -> Path:
    """Find a task dir under tasks/<group>/<task>/ (new layout) or tasks/<task>/ (legacy)."""
    direct = TASKS_DIR / task
    if direct.is_dir():
        return direct
    matches = [p for p in TASKS_DIR.glob(f"*/{task}") if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"[error] ambiguous --task {task!r}: {[str(m) for m in matches]}")
    raise SystemExit(f"[error] task {task!r} not found under {TASKS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="3-stage VQA attribute extraction")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_attributes.yaml")))
    parser.add_argument("--task", default=None,
                        help="task dir name (resolved under tasks/<group>/<task>); overrides "
                             "query_dir/out_dir to <task>/{query_pics, qa}")
    parser.add_argument("--query-dir", default=None, help="override query_dir")
    parser.add_argument("--out-dir", default=None, help="override out_dir")
    parser.add_argument("--reuse", action="store_true",
                        help="skip Stage1/2 API calls; re-filter existing stage2_summary.json")
    parser.add_argument("--from-stage1", action="store_true",
                        help="reuse existing stage1_descriptions.json; re-run Stage2 + Stage3 only")
    args = parser.parse_args()

    cfg = load_config(args.config, require_api_key=not args.reuse)
    # CLI overrides (priority: --query-dir/--out-dir > --task > config)
    if args.task:
        tdir = _resolve_task_dir(args.task)
        qp = tdir / "query_pics"
        if not qp.is_dir() and (tdir / "query pics").is_dir():
            qp = tdir / "query pics"  # legacy folder name with a space
        cfg["query_dir"] = str(qp)
        cfg["out_dir"] = str(tdir / "qa")
    if args.query_dir:
        cfg["query_dir"] = args.query_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    query_dir = cfg["query_dir"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] query_dir={query_dir}\n[init] out_dir={out_dir}")

    images = load_image_list(query_dir)
    print(f"[init] {len(images)} query images in {query_dir}")
    if not images:
        print("[error] no query images found")
        sys.exit(1)

    if args.reuse:
        summary = json.loads((out_dir / "stage2_summary.json").read_text(encoding="utf-8"))
        print("[reuse] loaded existing stage2_summary.json (no API calls)")
    else:
        print(f"[init] model={cfg['model']} @ {cfg['base_url']}")
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

        if args.from_stage1:
            descriptions = json.loads(
                (out_dir / "stage1_descriptions.json").read_text(encoding="utf-8"))
            print(f"[from-stage1] loaded {len(descriptions)} existing descriptions (no Stage1 calls)")
        else:
            print("\n=== Stage 1: per-image descriptions ===")
            descriptions = stage1_describe(client, cfg, images)
            (out_dir / "stage1_descriptions.json").write_text(
                json.dumps(descriptions, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n=== Stage 2: summarize shared attributes ===")
        summary = stage2_summarize(client, cfg, descriptions)
        (out_dir / "stage2_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  target: {summary.get('target_concept_summary', '')}")
    print(f"  raw attributes: {len(summary.get('attributes', []))}")

    print("\n=== Stage 3: auto-filter ===")
    kept = stage3_filter(summary, cfg, num_queries=len(images))
    names = [a["name"] for a in kept]
    final_str = "{" + ", ".join(names) + "}"

    (out_dir / "attributes.txt").write_text(final_str, encoding="utf-8")

    md = ["# 自动抽取的属性（3-stage VQA pipeline）\n",
          f"**target**: {summary.get('target_concept_summary','')}\n",
          f"**final attributes**: `{final_str}`\n", "## 入选属性\n"]
    for a in kept:
        md.append(f"- **{a['name']}** — {a.get('description','')}  "
                  f"`[{a.get('type','')}/{a.get('shared_strength','')}/"
                  f"ver={a.get('verifiability','')}/subj={a.get('subjectivity','')}]`")
    dropped = [a for a in summary.get("attributes", []) if a not in kept]
    if dropped:
        md.append("\n## 未入选（被筛掉）\n")
        for a in dropped:
            md.append(f"- {a['name']} `[{a.get('shared_strength','')}/"
                      f"ver={a.get('verifiability','')}/subj={a.get('subjectivity','')}]`")
    (out_dir / "answer_auto.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== FINAL ATTRIBUTES ===")
    print(final_str)
    print(f"\n[done] saved to {out_dir}/ (stage1_descriptions.json, stage2_summary.json, "
          f"attributes.txt, answer_auto.md)")


if __name__ == "__main__":
    main()
