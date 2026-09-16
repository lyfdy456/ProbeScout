from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


WEB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = WEB_ROOT / "scripts" / "export_web_data.py"
SCRIPT_DIR = str(MODULE_PATH.parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location("pcp_export_web_data_images", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


class ImageSourceResolverTests(unittest.TestCase):
    def test_direct_path_has_priority_without_recursive_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = root / "same.jpg"
            direct.write_bytes(b"direct")
            nested = root / "class-a" / "same.jpg"
            nested.parent.mkdir()
            nested.write_bytes(b"nested")

            resolver = EXPORT.ImageSourceResolver(root)

            self.assertEqual(resolver.resolve("same.jpg"), direct.resolve())
            self.assertIsNone(resolver._basename_index)

    def test_unique_nested_basename_is_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "beaver" / "beaver_10027.jpg"
            nested.parent.mkdir()
            nested.write_bytes(b"nested")

            resolver = EXPORT.ImageSourceResolver(root)

            self.assertEqual(
                resolver.resolve("beaver_10027.jpg"),
                nested.resolve(),
            )
            self.assertIsNotNone(resolver._basename_index)

    def test_duplicate_nested_basename_fails_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("class-a", "class-b"):
                path = root / directory / "duplicate.jpg"
                path.parent.mkdir()
                path.write_bytes(directory.encode("utf-8"))

            resolver = EXPORT.ImageSourceResolver(root)

            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolver.resolve("duplicate.jpg")

    def test_fixed_query_uses_nested_gallery_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            source = image_root / "otter" / "otter_10244.jpg"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"query-image")
            task_dir = root / "task"
            task_dir.mkdir()
            (task_dir / "query_ids.json").write_text(
                json.dumps({"image_ids": ["otter_10244.jpg"]}),
                encoding="utf-8",
            )
            output_dir = root / "output"

            query = EXPORT.export_fixed_query(
                output_dir=output_dir,
                image_root=image_root,
                image_ids=["otter_10244.jpg"],
                dataset="awa2",
                task_name="task_test",
                task_dir_override=task_dir,
                query_text_override=None,
            )

            self.assertIsNotNone(query)
            self.assertEqual(query["images"][0]["imageIndex"], 0)
            self.assertEqual(
                (output_dir / query["images"][0]["path"]).read_bytes(),
                b"query-image",
            )

    def test_atlas_uses_nested_gallery_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            first = image_root / "beaver" / "beaver_10027.jpg"
            second = image_root / "otter" / "otter_10244.jpg"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            Image.new("RGB", (20, 20), (220, 20, 20)).save(first)
            Image.new("RGB", (20, 20), (20, 20, 220)).save(second)
            output_dir = root / "output"
            args = SimpleNamespace(
                atlas_columns=2,
                atlas_rows=1,
                tile_width=16,
                tile_height=16,
                force_atlases=True,
                atlas_workers=1,
                webp_quality=80,
            )
            config = EXPORT.atlas_config(args, 2)

            status = EXPORT.build_atlases(
                ["beaver_10027.jpg", "otter_10244.jpg"],
                image_root,
                output_dir,
                config,
                args,
            )

            self.assertEqual(status["missingSourceCount"], 0)
            atlas = output_dir / "atlases" / "atlas-0000.webp"
            self.assertTrue(atlas.is_file())
            with Image.open(atlas) as image:
                self.assertEqual(image.size, (32, 16))


if __name__ == "__main__":
    unittest.main()
