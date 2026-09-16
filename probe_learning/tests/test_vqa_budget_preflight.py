import json
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from vqa_budget_preflight import (  # noqa: E402
    PreflightError,
    build_preflight_manifest,
    expanded_prompt,
    load_and_validate_manifest,
    materialize_public_vqa_config,
    materialize_task_attribute_files,
    validate_external_credential_source,
    validate_runtime_batch,
    validate_strict_cache_preflight_state,
    write_manifest,
)


class FakeAdapter:
    def __init__(self, task_root: Path, images: dict[str, Path]):
        self.task_root = task_root
        self.images = images

    def vqa_to_key(self, value: str) -> str:
        return str(value).replace("\\", "/")

    def resolve_image_path(self, value: str) -> Path:
        return self.images[self.vqa_to_key(value)]


class VqaBudgetPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task_root = self.root / "tasks" / "demo" / "task_demo"
        self.qa_root = self.task_root / "qa"
        self.stage = "iterative_vqa_100_50_v3_budget500"
        self.round_root = self.qa_root / self.stage / "round_00"
        self.round_root.mkdir(parents=True)
        self.attrs = ["red object", "round shape"]
        (self.qa_root / "attributes.txt").write_text(
            "{red object, round shape}", encoding="utf-8"
        )
        image_root = self.root / "raw" / "demo"
        image_root.mkdir(parents=True)
        self.image_a = image_root / "a.jpg"
        self.image_b = image_root / "b.jpg"
        self.image_c = image_root / "c.jpg"
        self.image_a.write_bytes(b"jpeg-a")
        self.image_b.write_bytes(b"jpeg-b")
        self.image_c.write_bytes(b"jpeg-c")
        self.adapter = FakeAdapter(
            self.task_root,
            {"a.jpg": self.image_a, "b.jpg": self.image_b, "c.jpg": self.image_c},
        )
        self.round_manifest = {
            "round": 0,
            "stage": self.stage,
            "train": ["a.jpg"],
            "audit": ["b.jpg"],
            "roles": {"query_near": ["a.jpg"]},
            "to_label": ["a.jpg", "b.jpg"],
            "diagnostics": {"backfill_reason": "deterministic capacity fill"},
        }
        (self.round_root / "manifest.json").write_text(
            json.dumps(self.round_manifest), encoding="utf-8"
        )
        self.state_path = self.task_root / "split" / self.stage / "round_state.json"
        self.state_path.parent.mkdir(parents=True)
        self.state_path.write_text(
            json.dumps({"status": "awaiting_vqa", "pending_round": 0, "rounds": []}),
            encoding="utf-8",
        )
        self.secret = "fixture-secret-that-must-not-be-written"
        self.config_path = self.root / "vqa.yaml"
        self._write_config()
        self.task = {
            "order": 7,
            "dataset": "demo",
            "task": "task_demo",
            "joint_label": "red & round",
            "attributes": self.attrs,
        }
        self.strict_vqa_map = {}
        self.strict_conflicts = {}
        self.committed_cache_ok = True

    def tearDown(self):
        self.temp.cleanup()

    def _write_config(
        self, *, model="qwen-vl-max", prompt=None, max_tokens=256, api_key=None
    ):
        prompt = prompt or "Image {image_id}\nTarget attributes:\n{attributes}\nReturn JSON."
        api_key = self.secret if api_key is None else api_key
        self.config_path.write_text(
            "\n".join(
                [
                    f'api_key: "{api_key}"',
                    'base_url: "https://example.invalid/v1"',
                    f'model: "{model}"',
                    "prompt: |",
                    *[f"  {line}" for line in prompt.splitlines()],
                    f"max_tokens: {max_tokens}",
                    "timeout: 30",
                    "max_retries: 2",
                    "concurrency: 7",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _fake_harness(self):
        fake_harness = types.ModuleType("run_retrieval_harness")
        fake_harness.configure = lambda *args, **kwargs: None
        fake_harness.ADAPTER = self.adapter
        fake_harness.ATTRS = self.attrs
        fake_harness.is_complete_binary_labels = (
            lambda labels, attrs: isinstance(labels, dict)
            and set(labels) == set(attrs)
            and all(type(labels[attr]) is int and labels[attr] in (0, 1) for attr in attrs)
        )
        return fake_harness

    def _fake_iterative_runner(self):
        fake_runner = types.ModuleType("iterative_vqa_runner")
        fake_runner._all_cached_sources = lambda adapter, stage: []
        fake_runner._load_iterative_cache = lambda sources, adapter, qa_root: (
            dict(self.strict_vqa_map),
            {},
            {
                "source_set_sha256": "fixture-source-set",
                "complete_image_count": len(self.strict_vqa_map),
                "conflicts": dict(self.strict_conflicts),
                "shadowed_conflicts": {},
            },
        )
        fake_runner._committed_cache_audit = lambda state, vqa_map, audit: {
            "schema": "fixture-committed-cache-audit",
            "ok": self.committed_cache_ok,
            "earliest_invalid_committed_round": None if self.committed_cache_ok else 0,
            "invalid_committed_rounds": [] if self.committed_cache_ok else [{"round": 0}],
            "uncommitted_conflicts": [],
            "migrations": [],
        }
        return fake_runner

    def _build(self):
        fake_harness = self._fake_harness()
        fake_runner = self._fake_iterative_runner()
        with patch.dict(
            sys.modules,
            {
                "run_retrieval_harness": fake_harness,
                "iterative_vqa_runner": fake_runner,
            },
        ):
            return build_preflight_manifest(
                [self.task],
                state_paths={7: self.state_path},
                stage=self.stage,
                vqa_config=self.config_path,
            )

    def _materialize(self):
        fake_harness = self._fake_harness()
        with patch.dict(sys.modules, {"run_retrieval_harness": fake_harness}):
            return materialize_task_attribute_files([self.task])

    def _precheck(self):
        fake_harness = self._fake_harness()
        fake_runner = self._fake_iterative_runner()
        with patch.dict(
            sys.modules,
            {
                "run_retrieval_harness": fake_harness,
                "iterative_vqa_runner": fake_runner,
            },
        ):
            return validate_strict_cache_preflight_state(
                [self.task],
                state_paths={7: self.state_path},
                stage=self.stage,
            )

    def _write(self):
        path = self.root / "out" / "preflight.json"
        payload = write_manifest(path, self._build())
        return path, payload

    def _validate_runtime(self, path, payload, candidates):
        return validate_runtime_batch(
            path,
            payload["manifest_sha256"],
            order=7,
            dataset="demo",
            task="task_demo",
            stage=self.stage,
            round_no=0,
            candidate_rels=candidates,
            adapter=self.adapter,
            modeled_attributes=self.attrs,
            attributes_file=self.qa_root / "attributes.txt",
            vqa_config=self.config_path,
        )

    def test_export_is_credential_free_and_contains_auditable_rows(self):
        path, payload = self._write()
        raw = path.read_text(encoding="utf-8")
        self.assertNotIn("api_key", raw.lower())
        self.assertNotIn(self.secret, raw)
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        self.assertEqual(payload["row_count"], 2)
        self.assertEqual(payload["task_count"], 1)
        rows = {row["logical_image"]: row for row in payload["rows"]}
        self.assertEqual(rows["a.jpg"]["partition"], "train")
        self.assertEqual(rows["a.jpg"]["acquisition_role"], "query_near")
        self.assertEqual(rows["b.jpg"]["partition"], "audit")
        self.assertEqual(rows["b.jpg"]["acquisition_role"], "backfill")
        self.assertEqual(
            rows["b.jpg"]["backfill_reason"], "deterministic capacity fill"
        )
        for row in rows.values():
            self.assertEqual(row["attributes"], self.attrs)
            self.assertEqual(row["model"], "qwen-vl-max")
            self.assertEqual(len(row["image_sha256"]), 64)
            self.assertEqual(len(row["expanded_prompt_sha256"]), 64)
            self.assertTrue(Path(row["absolute_image_path"]).is_absolute())
            self.assertIn("raw/demo", row["task_relative_image_path"])
        loaded = load_and_validate_manifest(path, payload["manifest_sha256"])
        self.assertEqual(loaded["manifest_sha256"], payload["manifest_sha256"])

    def test_attribute_materialization_is_no_network_idempotent_and_fail_closed(self):
        attr_path = self.qa_root / "attributes.txt"
        attr_path.unlink()
        (self.task_root / "attributes.txt").write_text(
            "{red object, round shape, legacy extra attribute}", encoding="utf-8"
        )
        first = self._materialize()
        self.assertEqual(first[0]["status"], "created")
        self.assertEqual(
            attr_path.read_text(encoding="utf-8").strip(),
            "{red object, round shape}",
        )
        second = self._materialize()
        self.assertEqual(second[0]["status"], "reused_exact")
        attr_path.write_text("{different attribute}", encoding="utf-8")
        with self.assertRaisesRegex(PreflightError, "differ from runtime"):
            self._materialize()

    def test_prompt_expansion_exactly_matches_provider_labeler(self):
        vqa_dir = Path(__file__).resolve().parents[2] / "attribute_annotation"
        openai_stub = types.ModuleType("openai")
        openai_stub.OpenAI = object
        tqdm_stub = types.ModuleType("tqdm")
        tqdm_stub.tqdm = lambda value, **kwargs: value
        spec = importlib.util.spec_from_file_location(
            "fixture_vqa_label", vqa_dir / "vqa_label.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"openai": openai_stub, "tqdm": tqdm_stub}):
            spec.loader.exec_module(module)
        build_prompt = module.build_prompt
        template = "  Image {image_id}\nAttrs: {attributes}\n  "
        attributes = "  {red object, round shape}  "
        config = {"prompt": template, "attributes": attributes}
        self.assertEqual(
            expanded_prompt(template, "a.jpg", attributes),
            build_prompt(config, "a.jpg"),
        )

    def test_external_calls_reject_literal_credentials_and_require_runtime_env(self):
        with self.assertRaises(PreflightError) as captured:
            validate_external_credential_source(
                self.config_path, environ={"DASHSCOPE_API_KEY": "runtime-value"}
            )
        self.assertIn("non-empty credential fields", str(captured.exception))
        self.assertNotIn(self.secret, str(captured.exception))
        self.assertNotIn("runtime-value", str(captured.exception))

        self._write_config(api_key="")
        with self.assertRaisesRegex(PreflightError, "requires DASHSCOPE_API_KEY"):
            validate_external_credential_source(self.config_path, environ={})
        result = validate_external_credential_source(
            self.config_path,
            environ={"DASHSCOPE_API_KEY": "runtime-value"},
        )
        self.assertEqual(result["environment_variable"], "DASHSCOPE_API_KEY")
        self.assertNotIn("runtime-value", json.dumps(result))

    def test_public_config_is_secret_free_idempotent_and_guarded(self):
        destination = self.root / "acquisition" / "public_vqa_config.yaml"
        first = materialize_public_vqa_config(self.config_path, destination)
        self.assertEqual(first["status"], "created")
        raw = destination.read_text(encoding="utf-8")
        self.assertNotIn("api_key", raw.lower())
        self.assertNotIn(self.secret, raw)
        public = yaml.safe_load(raw)
        source = yaml.safe_load(
            self.config_path.read_text(encoding="utf-8")
        )
        self.assertEqual(set(public), {
            key for key in (
                "base_url", "model", "prompt", "max_tokens", "timeout",
                "max_retries", "delay", "concurrency",
            )
            if key in source
        })
        for key in (
            "base_url",
            "model",
            "prompt",
            "max_tokens",
            "timeout",
            "max_retries",
            "concurrency",
        ):
            self.assertEqual(public[key], source[key])
        second = materialize_public_vqa_config(self.config_path, destination)
        self.assertEqual(second["status"], "reused_exact")

        self._write_config(model="new-model")
        with self.assertRaisesRegex(PreflightError, "explicit overwrite"):
            materialize_public_vqa_config(self.config_path, destination)
        replaced = materialize_public_vqa_config(
            self.config_path, destination, overwrite=True
        )
        self.assertEqual(replaced["status"], "created_or_replaced")

    def test_runtime_allows_only_an_exact_approved_subset(self):
        path, payload = self._write()
        result = self._validate_runtime(path, payload, ["a.jpg"])
        self.assertEqual(result["validated_rows"], 1)
        self.assertEqual(result["logical_images"], ["a.jpg"])
        with self.assertRaisesRegex(PreflightError, "was not approved"):
            self._validate_runtime(path, payload, ["c.jpg"])

    def test_runtime_rejects_wrong_stage_and_changed_selection_contract(self):
        path, payload = self._write()
        with self.assertRaisesRegex(PreflightError, "stage mismatch"):
            validate_runtime_batch(
                path,
                payload["manifest_sha256"],
                order=7,
                dataset="demo",
                task="task_demo",
                stage="iterative_vqa_wrong_stage",
                round_no=0,
                candidate_rels=["a.jpg"],
                adapter=self.adapter,
                modeled_attributes=self.attrs,
                attributes_file=self.qa_root / "attributes.txt",
                vqa_config=self.config_path,
            )

        # Retry bookkeeping may change, but the train/audit/role contract may not.
        changed = dict(self.round_manifest)
        changed["roles"] = {"committee_disagreement": ["a.jpg"]}
        (self.round_root / "manifest.json").write_text(
            json.dumps(changed), encoding="utf-8"
        )
        with self.assertRaisesRegex(PreflightError, "selection changed"):
            self._validate_runtime(path, payload, ["a.jpg"])

    def test_runtime_fails_closed_when_image_bytes_change(self):
        path, payload = self._write()
        self.image_a.write_bytes(b"changed-after-approval")
        with self.assertRaisesRegex(PreflightError, "runtime payload changed"):
            self._validate_runtime(path, payload, ["a.jpg"])

    def test_runtime_fails_closed_when_attrs_model_or_prompt_change(self):
        path, payload = self._write()
        (self.qa_root / "attributes.txt").write_text(
            "{red object, square shape}", encoding="utf-8"
        )
        with self.assertRaisesRegex(PreflightError, "differ from runtime"):
            self._validate_runtime(path, payload, ["a.jpg"])

        (self.qa_root / "attributes.txt").write_text(
            "{red object, round shape}", encoding="utf-8"
        )
        self._write_config(model="different-model")
        with self.assertRaisesRegex(PreflightError, "runtime payload changed"):
            self._validate_runtime(path, payload, ["a.jpg"])

        self._write_config(prompt="Changed {image_id}: {attributes}")
        with self.assertRaisesRegex(PreflightError, "runtime payload changed"):
            self._validate_runtime(path, payload, ["a.jpg"])

        self._write_config(max_tokens=999)
        with self.assertRaisesRegex(PreflightError, "runtime payload changed"):
            self._validate_runtime(path, payload, ["a.jpg"])

    def test_manifest_hash_tamper_and_wrong_approval_are_rejected(self):
        path, payload = self._write()
        with self.assertRaisesRegex(PreflightError, "approval hash mismatch"):
            load_and_validate_manifest(path, "0" * 64)

        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["rows"][0]["image_bytes"] += 1
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(PreflightError, "content hash mismatch"):
            load_and_validate_manifest(path, payload["manifest_sha256"])

    def test_atomic_writer_requires_explicit_divergent_overwrite(self):
        path, payload = self._write()
        changed = dict(payload)
        changed["created_at_utc"] = "2099-01-01T00:00:00+00:00"
        with self.assertRaisesRegex(PreflightError, "explicit overwrite"):
            write_manifest(path, changed)
        replaced = write_manifest(path, changed, overwrite=True)
        loaded = load_and_validate_manifest(path, replaced["manifest_sha256"])
        self.assertEqual(loaded["created_at_utc"], changed["created_at_utc"])

    def test_export_ignores_stale_nonselected_to_label_row(self):
        self.round_manifest["to_label"].append("c.jpg")
        (self.round_root / "manifest.json").write_text(
            json.dumps(self.round_manifest), encoding="utf-8"
        )
        payload = self._build()
        self.assertEqual(
            [row["logical_image"] for row in payload["rows"]],
            ["a.jpg", "b.jpg"],
        )

    def test_export_recomputes_strict_missing_instead_of_trusting_to_label(self):
        self.strict_vqa_map = {"a.jpg": {"red object": 1, "round shape": 0}}
        self.round_manifest["to_label"] = ["a.jpg"]  # intentionally stale/wrong
        (self.round_root / "manifest.json").write_text(
            json.dumps(self.round_manifest), encoding="utf-8"
        )
        payload = self._build()
        self.assertEqual(
            [row["logical_image"] for row in payload["rows"]], ["b.jpg"]
        )
        strict = payload["tasks"][0]["strict_cache"]
        self.assertFalse(strict["saved_matches_strict"])
        self.assertEqual(strict["strict_missing_count"], 1)

    def test_export_rejects_any_strict_cache_conflict(self):
        self.strict_conflicts = {"a.jpg": {"variants": []}}
        with self.assertRaisesRegex(PreflightError, "conflicting complete labels"):
            self._build()

    def test_export_rejects_invalid_committed_cache_state(self):
        self.committed_cache_ok = False
        with self.assertRaisesRegex(PreflightError, "committed iterative VQA cache is invalid"):
            self._build()

    def test_read_only_precheck_rejects_invalid_committed_state(self):
        self.assertEqual(self._precheck()[0]["status"], "awaiting_vqa")
        self.committed_cache_ok = False
        with self.assertRaisesRegex(PreflightError, "invalid committed iterative cache"):
            self._precheck()


if __name__ == "__main__":
    unittest.main()
