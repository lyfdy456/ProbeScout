"""Validate the source publication boundary without loading research assets."""
from pathlib import Path
import ast
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".vendor", ".next", ".vinext", ".npm-cache", "build", "dist", ".wrangler"}
FORBIDDEN_PARTS = {"runtime", "users", "sessions", "human_feedback", "feedback_data", "case_replay"}
BINARY_ASSETS = {".npy", ".npz", ".pt", ".pth", ".safetensors", ".f32", ".f64", ".sqlite", ".sqlite3", ".db"}
CREDENTIAL_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})\b")


def credential_issues(name: str, text: str) -> list[str]:
    """Report locations, never credential values; check templates as well as code."""
    issues = []
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        issues.append("Private environment file must not be published")
    if CREDENTIAL_TOKEN.search(text):
        issues.append("Possible embedded credential")
    if Path(name).suffix in {".yaml", ".yml"} or name == ".env.example":
        for number, line in enumerate(text.splitlines(), 1):
            field = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*?)\s*(?:#.*)?$", line)
            if not field:
                continue
            key, value = field.groups()
            secret_key = key.lower() in {"api_key", "access_token", "secret", "password"} or key.lower().endswith(
                ("_api_key", "_access_token", "_secret", "_password")
            )
            if secret_key and value not in {"", '""', "''", "null", "~"}:
                issues.append(f"Credential field must be empty at line {number}")
    return issues


def main():
    issues = []
    counts = {"files": 0, "python": 0, "json": 0}
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in IGNORED]
        for name in files:
            path = Path(folder) / name
            rel = path.relative_to(ROOT)
            counts["files"] += 1
            if FORBIDDEN_PARTS.intersection(rel.parts) or path.suffix in BINARY_ASSETS or name in {"labels_snapshot.json", "annotations.json", ".cookie-secret"}:
                issues.append(f"Non-public payload present: {rel}")
            if path.stat().st_size > 100 * 1024 * 1024:
                issues.append(f"Large file belongs in the separate asset package: {rel}")
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeError:
                text = ""
            issues.extend(f"{issue}: {rel}" for issue in credential_issues(name, text))
            if path.suffix == ".py":
                counts["python"] += 1
                try:
                    ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(rel))
                except (SyntaxError, UnicodeError) as error:
                    issues.append(f"Python parse error in {rel}: {error}")
            if path.suffix == ".json":
                counts["json"] += 1
                try:
                    json.loads(path.read_text(encoding="utf-8-sig"))
                except (ValueError, UnicodeError) as error:
                    issues.append(f"JSON parse error in {rel}: {error}")
            if path.suffix in {".py", ".mjs", ".js", ".ts", ".tsx", ".json", ".toml", ".lock", ".yaml", ".yml"}:
                text = path.read_text(encoding="utf-8-sig")
                if path != Path(__file__).resolve() and re.search(r"(?:linear_probing|pVIS)[/\\]|[\"'](?:linear_probing|pVIS)[\"']|vlm auto attributes extraction", text):
                    issues.append(f"Old directory reference in executable/configuration source: {rel}")
    web = ROOT / "visual_analytics/pcp_analyze/web"
    source = ROOT / "probe_learning"
    if web.parents[2] / "probe_learning" != source:
        issues.append("Web-to-learning relative path invariant failed")
    package = json.loads((web / "package.json").read_text())
    lock = json.loads((web / "package-lock.json").read_text())
    if package["name"] != lock["name"] or package["name"] != lock["packages"][""]["name"]:
        issues.append("Web package/lock names differ")
    for group in ("dependencies", "devDependencies", "engines"):
        if package.get(group, {}) != lock["packages"][""].get(group, {}):
            issues.append(f"Web package/lock {group} differ")
    policy = json.loads((ROOT / "manifests/publication_policy.json").read_text())
    if policy["github"]["human_feedback_data"] or policy["hugging_face"]["human_feedback_data"]:
        issues.append("Human feedback publication must remain disabled")
    scope = json.loads((ROOT / "manifests/paper_main17.json").read_text())
    if len({t["task_id"] for t in scope["tasks"]}) != scope["task_count"] or scope["task_count"] != 17:
        issues.append("Main17 task manifest is invalid")
    tasks = json.loads((ROOT / "configs/main17_tasks.json").read_text())
    suite = json.loads((ROOT / "configs/probe_suite.json").read_text())
    if [t["task_id"] for t in tasks["tasks"]] != [t["task_id"] for t in scope["tasks"]]:
        issues.append("Training/export task order differs from Main17")
    if suite["learned_methods"] != scope["method_ids"]:
        issues.append("Training suite differs from the eight paper probes")
    print(json.dumps({"ok": not issues, "counts": counts, "issues": issues}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
