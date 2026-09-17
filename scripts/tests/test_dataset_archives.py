"""Original archive layouts, recovery and extraction boundaries."""
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_archives import archive_parts, import_images


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.files = {"000001.jpg": b"image one", "000002.jpg": b"image two"}
        self.queries = {"000001.jpg": hashlib.sha256(self.files["000001.jpg"]).hexdigest()}

    def tearDown(self):
        self.temporary.cleanup()

    def make(self, extension="zip", names=None):
        archive = self.root / ("images." + extension)
        names = names or {f"wrapper/car_ims/{k}": v for k, v in self.files.items()}
        if extension == "zip":
            with zipfile.ZipFile(archive, "w") as stream:
                for name, data in names.items():
                    stream.writestr(name, data)
        else:
            with tarfile.open(archive, "w:gz") as stream:
                for name, data in names.items():
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    stream.addfile(entry, io.BytesIO(data))
        return archive

    def extract(self, archive, **kwargs):
        return import_images(archive, self.root / "output", "cars", list(self.files), self.queries, **kwargs)

    def test_zip_and_tgz_strip_wrappers_without_a_second_image_copy(self):
        for extension in ("zip", "tgz"):
            archive = self.make(extension)
            output = self.extract(archive)
            for name, data in self.files.items():
                self.assertEqual((output / name).read_bytes(), data)
            self.assertFalse((output / "wrapper").exists())
            self.assertTrue(archive.is_file())
            before = (output / "000001.jpg").stat().st_mtime_ns
            self.assertEqual(self.extract(archive), output)
            self.assertEqual((output / "000001.jpg").stat().st_mtime_ns, before)

    def test_hico_preserves_train_and_test_folders(self):
        files = {"train2015/HICO_train2015_00000001.jpg": b"train",
                 "test2015/HICO_test2015_00000001.jpg": b"test"}
        archive = self.make(names={f"hico_20160224_det/images/{k}": v for k, v in files.items()})
        output = import_images(archive, self.root / "output", "hico", list(files), {})
        self.assertEqual({k: (output / k).read_bytes() for k in files}, files)

    def test_missing_or_duplicate_images_fail_before_extracting(self):
        for names in (
            {"cars_train/00001.jpg": b"wrong numbering"},
            {"a/000001.jpg": b"x", "b/000001.jpg": b"x", "000002.jpg": b"y"},
        ):
            with self.assertRaises(ValueError):
                self.extract(self.make(names=names))
            self.assertFalse((self.root / "output").exists())

    def test_archive_path_escape_and_tar_links_are_rejected(self):
        for extension in ("zip", "tgz"):
            with self.assertRaises(ValueError):
                self.extract(self.make(extension, {"../escape.jpg": b"outside"}))
        archive = self.root / "link.tar"
        with tarfile.open(archive, "w") as stream:
            member = tarfile.TarInfo("car_ims/000001.jpg")
            member.type = tarfile.SYMTYPE
            member.linkname = "../outside"
            stream.addfile(member)
        with self.assertRaises(ValueError):
            self.extract(archive)
        self.assertFalse((self.root / "output").exists())

    def test_query_bytes_reject_the_wrong_image_variant(self):
        with self.assertRaisesRegex(ValueError, "variant differs"):
            self.extract(self.make(names={"000001.jpg": b"aligned crop", "000002.jpg": b"other"}))

    def test_insufficient_space_does_not_write_images(self):
        archive = self.make()
        with patch("dataset_archives.shutil.disk_usage") as usage:
            usage.return_value.free = 100
            with self.assertRaisesRegex(ValueError, "Choose another"):
                self.extract(archive)
        self.assertFalse(list((self.root / "output").rglob("*.jpg")))

    def test_interrupted_import_reuses_finished_images(self):
        archive = self.make()
        stopped = []
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.extract(archive, progress=lambda _: stopped.append(True), cancelled=lambda: bool(stopped))
        first = next((self.root / "output").rglob("000001.jpg"))
        stamp = first.stat().st_mtime_ns
        output = self.extract(archive)
        self.assertEqual(first.stat().st_mtime_ns, stamp)
        self.assertEqual((output / "000002.jpg").read_bytes(), self.files["000002.jpg"])

    @unittest.skipUnless(os.environ.get("PROBESCOUT_TEST_7ZR"), "Set PROBESCOUT_TEST_7ZR to test real split 7z")
    def test_real_split_7z_with_unicode_paths_and_missing_volume(self):
        tool = os.environ["PROBESCOUT_TEST_7ZR"]
        source = self.root / "原图 with spaces" / "img_celeba"
        source.mkdir(parents=True)
        for name, data in self.files.items():
            (source / name).write_bytes(data)
        archive = self.root / "img_celeba.7z"
        subprocess.run([tool, "a", str(archive), str(source), "-v150b", "-y"],
                       stdout=subprocess.DEVNULL, check=True)
        first = Path(str(archive) + ".001")
        parts = archive_parts(first)
        self.assertGreater(len(parts), 1)
        output = self.extract(first, seven=tool)
        self.assertEqual((output / "000001.jpg").read_bytes(), self.files["000001.jpg"])
        parts[-1].unlink()
        with self.assertRaises(ValueError):
            self.extract(first, seven=tool)


if __name__ == "__main__":
    unittest.main()
