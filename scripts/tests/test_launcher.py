"""Installer boundaries and recovery; runs without scientific dependencies."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

SPEC = importlib.util.spec_from_file_location("launcher", Path(__file__).resolve().parents[1] / "launch.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.local = patch.object(launcher, "LOCAL", self.root)
        self.local.start()
        self.installer = launcher.Installer(no_browser=True)

    def tearDown(self):
        self.installer.logfile.close()
        self.local.stop()
        self.temp.cleanup()

    def serve(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_extract_rejects_traversal_before_writing(self):
        with self.assertRaises(ValueError):
            launcher.inside(self.root, "folder\\escape")
        for name in ("../escape", "C:/escape", "/escape", "file:stream"):
            archive = self.root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("first.txt", "should not be written")
                output.writestr(name, "bad")
            with self.assertRaises(ValueError):
                launcher.extract(archive, self.root / "destination")
            self.assertFalse((self.root / "destination/first.txt").exists())

    def test_extract_rejects_symlinks(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as output:
            entry = zipfile.ZipInfo("link")
            entry.external_attr = 0o120777 << 16
            output.writestr(entry, "outside")
        with self.assertRaises(ValueError):
            launcher.extract(archive, self.root / "destination")

    def test_changed_local_payload_is_not_reused(self):
        path = self.root / "asset"
        path.write_bytes(b"good")
        checksum = hashlib.sha256(b"good").hexdigest()
        self.assertTrue(self.installer.valid(path, checksum, 4))
        path.write_bytes(b"modified")
        self.assertFalse(self.installer.valid(path, checksum, 4))

    def test_download_resumes_and_falls_back_when_server_ignores_range(self):
        data = b"test asset content" * 400
        requested = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                requested.append(self.headers.get("Range"))
                offset = 123 if self.path == "/resume" else 0
                self.send_response(206 if offset else 200)
                self.send_header("Content-Length", str(len(data)-offset))
                if offset:
                    self.send_header("Content-Range", f"bytes {offset}-{len(data)-1}/{len(data)}")
                self.end_headers()
                self.wfile.write(data[offset:])
        url = self.serve(Handler)
        for route in ("resume", "restart"):
            target = self.root / route
            target.with_name(route + ".part").write_bytes(data[:123])
            self.installer.download(url + "/" + route, target, hashlib.sha256(data).hexdigest(), len(data))
            self.assertEqual(target.read_bytes(), data)
            # A verified destination must not make another request.
            self.installer.download(url + "/" + route, target, hashlib.sha256(data).hexdigest(), len(data))
        self.assertEqual(requested, ["bytes=123-", "bytes=123-"])

    def test_corrupt_download_is_never_promoted(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"bad")
        url = self.serve(Handler)
        target = self.root / "asset"
        with self.assertRaises(RuntimeError):
            self.installer.download(url, target, hashlib.sha256(b"good").hexdigest(), 4)
        self.assertFalse(target.exists())

    def test_asset_selection_excludes_other_datasets_and_common_scripts(self):
        files = [{"dataset_id": "cars", "path": "cars.npy"},
                 {"dataset_id": "hico", "path": "hico.npy"}, {"path": "restore.py"}]
        self.assertEqual(launcher.selected_files({"files": files}, "cars"), [files[0]])

    def test_setup_rejects_cross_origin_or_missing_token(self):
        url = self.serve(launcher.make_handler(self.installer, "test-only-token"))
        for headers in ({}, {"X-Setup-Token": "test-only-token", "Origin": "https://example.org"},
                        {"X-Setup-Token": "test-only-token", "Host": "example.org"}):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url + "/status", headers=headers))
            self.assertEqual(error.exception.code, 403)
        with urlopen(Request(url + "/status", headers={"X-Setup-Token": "test-only-token"})) as response:
            self.assertEqual(json.load(response)["edition"], "basic")

    def test_missing_images_do_not_install_or_save_settings(self):
        with self.assertRaises(ValueError):
            self.installer.start({"dataset": "cars", "edition": "basic", "imageRoot": str(self.root)})
        self.assertFalse((self.root / "settings.json").exists())
        self.assertFalse(self.installer.busy)

    def test_sharded_features_reassemble_and_reuse_verified_array(self):
        data = b"frozen patch array"
        parts = [data[:6], data[6:]]
        entries = [{"path": f"dataset/patch.npy.part-{i}", "bytes": len(part),
                    "sha256": hashlib.sha256(part).hexdigest(), "byte_offset": 0 if i == 0 else 6,
                    "dataset_id": "celeba"} for i, part in enumerate(parts)]
        assembly = {"path": "dataset/patch.npy", "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "parts": entries}
        features = {"files": entries, "reassemble": [assembly]}
        probes = {"files": [{"path": "models/probe.pt", "bytes": 3,
                             "sha256": hashlib.sha256(b"one").hexdigest(), "dataset_id": "celeba"}]}
        manifests = {"features": features, "probes": probes}
        payloads = {entries[i]["path"]: part for i, part in enumerate(parts)}
        payloads["models/probe.pt"] = b"one"
        downloaded = []
        def download(url, target, *_):
            downloaded.append(target.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            name = target.name.removesuffix("-manifest.json")
            if name in manifests:
                launcher.write(target, manifests[name])
            else:
                target.write_bytes(payloads[target.relative_to(self.root).as_posix()])
        with patch.object(launcher, "ROOT", self.root), patch.object(self.installer, "download", download):
            self.installer.full_assets("celeba")
            self.assertEqual((self.root / "dataset/patch.npy").read_bytes(), data)
            # Once the final NPY is verified, no shards need to be fetched again.
            for entry in entries:
                (self.root / entry["path"]).unlink()
            downloaded.clear()
            self.installer.full_assets("celeba")
            self.assertEqual(downloaded, ["features-manifest.json", "probes-manifest.json"])


if __name__ == "__main__":
    unittest.main()
