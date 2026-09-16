from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "cluster_score_rank_profiles.py"
SPEC = importlib.util.spec_from_file_location("pcp_cluster_score_rank_profiles", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLUSTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLUSTER
SPEC.loader.exec_module(CLUSTER)


class ClusterImageSourceTests(unittest.TestCase):
    def test_nested_unique_basenames_are_resolved_in_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "class-a" / "first.jpg"
            second = root / "class-b" / "second.jpg"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            self.assertEqual(
                CLUSTER.resolve_image_sources(root, ["second.jpg", "first.jpg"]),
                [second.resolve(), first.resolve()],
            )

    def test_direct_path_wins_and_ambiguous_fallback_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = root / "same.jpg"
            left = root / "left" / "same.jpg"
            right = root / "right" / "same.jpg"
            direct.write_bytes(b"direct")
            left.parent.mkdir(parents=True)
            right.parent.mkdir(parents=True)
            left.write_bytes(b"left")
            right.write_bytes(b"right")
            self.assertEqual(
                CLUSTER.resolve_image_sources(root, ["same.jpg"]),
                [direct.resolve()],
            )
            direct.unlink()
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                CLUSTER.resolve_image_sources(root, ["same.jpg"])


if __name__ == "__main__":
    unittest.main()
