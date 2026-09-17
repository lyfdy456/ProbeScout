"""Local first-run setup and supervised Web launcher (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import webbrowser
import zipfile

from dataset_archives import archive_parts, import_images

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "visual_analytics/pcp_analyze/web"
LOCAL = ROOT / ".probescout"
DATASETS = {"cars": "stanford_cars/images", "hico": "HICO/images", "celeba": "CelebA/img_celeba"}


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root, relative):
    # Reject drive paths, NTFS streams and Windows backslashes on every OS.
    relative = str(relative)
    if PurePosixPath(relative).is_absolute() or ":" in relative or "\\" in relative or ".." in PurePosixPath(relative).parts:
        raise ValueError(f"Invalid package path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError(f"Path escapes package: {relative}")
    return result


def extract(archive, destination):
    with zipfile.ZipFile(archive) as bundle:
        entries = [(item, inside(destination, item.filename)) for item in bundle.infolist()]
        if any(stat.S_ISLNK(item.external_attr >> 16) for item, _ in entries):
            raise ValueError("Archive contains a symbolic link")
        for item, target in entries:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with bundle.open(item) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            os.replace(temporary, target)


def hf_url(pin, path):
    kind = "datasets/" if pin.get("repoType", "dataset") == "dataset" else ""
    return f"https://huggingface.co/{kind}{pin['repoId']}/resolve/{pin['revision']}/{quote(path, safe='/')}"


def selected_files(manifest, dataset):
    return [item for item in manifest["files"] if item.get("dataset_id") == dataset]


def available_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def supervise_windows_children():
    """The OS stops inherited child processes even if the console is closed."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    class BasicLimits(ctypes.Structure):
        _fields_ = [("processTime", ctypes.c_int64), ("jobTime", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t),
                    ("maximum", ctypes.c_size_t), ("active", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                    ("processMemory", ctypes.c_size_t), ("jobMemory", ctypes.c_size_t),
                    ("peakProcess", ctypes.c_size_t), ("peakJob", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess())):
        raise ctypes.WinError(ctypes.get_last_error())
    # The handle is intentionally held until process exit, when Windows closes it.
    return handle


class Installer:
    def __init__(self, edition="basic", no_browser=False):
        LOCAL.mkdir(parents=True, exist_ok=True)
        self.pins = read(ROOT / "manifests/launcher.json")
        self.tasks = read(ROOT / "manifests/task_packages.json")
        self.settings = read(LOCAL / "settings.json", {})
        self.verified = read(LOCAL / "verified.json", {})
        self.state = {"stage": "idle", "message": "Choose a dataset and its image folder.",
                      "edition": edition, "dataset": self.settings.get("dataset", "cars"),
                      "imageRoots": self.settings.get("imageRoots", {}),
                      "imageInputs": self.settings.get("imageInputs", {}), "sourceType": "folder",
                      "logs": [], "url": None}
        self.lock = threading.Lock()
        self.busy = False
        self.no_browser = no_browser
        self.children = set()
        self.closing = False
        self.logfile = (LOCAL / "launcher.log").open("a", encoding="utf-8", buffering=1)
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "UV_NO_PROGRESS": "1",
                    "UV_CACHE_DIR": str(LOCAL / "uv-cache"), "npm_config_cache": str(LOCAL / "npm-cache"),
                    "HF_HOME": str(LOCAL / "hf-cache")}
        self.uv = os.environ.get("PROBESCOUT_UV", str(LOCAL / "uv/uv.exe"))
        self.python = LOCAL / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def update(self, stage=None, message=None, **fields):
        with self.lock:
            if stage:
                self.state["stage"] = stage
            if message:
                self.state["message"] = message
            self.state.update(fields)
        if message:
            self.log(message)

    def log(self, message):
        line = str(message).rstrip()
        if not line:
            return
        with self.lock:
            self.state["logs"] = (self.state["logs"] + [line])[-100:]
            self.logfile.write(line + "\n")
        print(line, flush=True)

    def valid(self, path, checksum, size=None):
        if not path.is_file() or (size is not None and path.stat().st_size != size):
            return False
        stamp = [path.stat().st_size, path.stat().st_mtime_ns, checksum]
        key = str(path.resolve())
        if self.verified.get(key) == stamp:
            return True
        if sha(path) != checksum:
            return False
        self.verified[key] = stamp
        return True

    def download(self, url, target, checksum, size=None):
        if self.valid(target, checksum, size):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.is_file() else 0
                if offset and (size is None or offset == size) and self.valid(partial, checksum, size):
                    os.replace(partial, target)
                    return
                if size is not None and offset >= size:
                    partial.unlink()
                    offset = 0
                headers = {"User-Agent": "ProbeScout-launcher/1", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Download server returned an unexpected byte range")
                    if not append:
                        offset = 0
                    total = size or (int(response.headers.get("Content-Length", 0)) + offset)
                    last = 0
                    with partial.open("ab" if append else "wb") as output:
                        while chunk := response.read(4 * 1024 * 1024):
                            if self.closing:
                                raise RuntimeError("Launcher is closing")
                            output.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last > 1:
                                self.update(progress={"file": target.name, "bytes": offset, "total": total})
                                last = time.monotonic()
                if not self.valid(partial, checksum, size):
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(f"Checksum mismatch: {target.name}")
                os.replace(partial, target)
                self.valid(target, checksum, size)
                write(LOCAL / "verified.json", self.verified)
                return
            except (OSError, RuntimeError) as error:
                if self.closing or attempt == 2:
                    raise RuntimeError(f"Could not download {target.name}: {error}. Retry to resume.") from error
                self.log(f"Retry {attempt + 1}/2: {target.name} ({type(error).__name__})")

    def spawn(self, command, cwd=ROOT):
        if self.closing:
            raise RuntimeError("Launcher is closing")
        process = subprocess.Popen([str(v) for v in command], cwd=cwd, env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.children.add(process)
        def output():
            for line in process.stdout:
                if line.startswith("WARN Skipping file for "):
                    continue
                self.log(line)
        threading.Thread(target=output, daemon=True).start()
        return process

    def run(self, command, cwd=ROOT):
        process = self.spawn(command, cwd)
        result = process.wait()
        self.children.discard(process)
        if result:
            raise RuntimeError(f"{Path(str(command[0])).name} exited with code {result}. See the log below.")

    def stop_children(self):
        for process in list(self.children):
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            self.children.discard(process)

    def dependencies(self, edition):
        self.update("environment", "Installing the local Web environment…", progress=None)
        node_pin = self.pins["node"]
        node_dir = LOCAL / node_pin["directory"]
        self.node = node_dir / "node.exe"
        node_marker = LOCAL / "node.json"
        if read(node_marker) != node_pin["sha256"] or not self.node.is_file():
            archive = LOCAL / "downloads/node.zip"
            self.download(node_pin["url"], archive, node_pin["sha256"])
            extract(archive, LOCAL)
            write(node_marker, node_pin["sha256"])
        self.env["PATH"] = str(node_dir) + os.pathsep + self.env.get("PATH", "")
        if not self.python.is_file():
            self.run([self.uv, "venv", "--python", sys.executable, LOCAL / "venv"])
        requirements = ROOT / "scripts" / f"requirements-{'full' if edition == 'full' else 'web'}.txt"
        fingerprint = sha(requirements) + sha(ROOT / "scripts/requirements-web.txt")
        marker = LOCAL / f"dependencies-{edition}.json"
        if read(marker) != fingerprint:
            self.run([self.uv, "pip", "install", "--torch-backend", "cpu", "--python", self.python, "-r", requirements])
            write(marker, fingerprint)
        npm_marker = LOCAL / "npm.json"
        npm_hash = sha(WEB / "package-lock.json") + node_pin["sha256"]
        if read(npm_marker) != npm_hash or not (WEB / "node_modules/vinext/dist/cli.js").is_file():
            self.run([self.node, node_dir / "node_modules/npm/bin/npm-cli.js", "ci", "--no-audit", "--no-fund"], WEB)
            write(npm_marker, npm_hash)

    def install_task(self, dataset):
        self.update("tasks", f"Preparing {dataset} tasks…", progress=None)
        pin = self.tasks["datasets"][dataset]
        manifest = ROOT / f"dataset/tasks/{dataset}/manifest.json"
        installed = self.valid(manifest, pin["manifestSha256"])
        if installed:
            installed = all(self.valid(inside(ROOT, path), checksum)
                            for path, checksum in read(manifest)["files"].items())
        if not installed:
            archive = LOCAL / "downloads" / pin["archive"]
            self.download(hf_url(self.tasks, pin["archive"]), archive, pin["sha256"], pin["bytes"])
            extract(archive, ROOT)
        write(LOCAL / "verified.json", self.verified)

    def full_assets(self, dataset):
        for name in ("features", "probes"):
            self.update(name, f"Preparing {dataset} {name}…", progress=None)
            pin = self.pins[name]
            path = LOCAL / "downloads" / f"{name}-manifest.json"
            self.download(hf_url(pin, "asset_manifest.json"), path, pin["manifestSha256"])
            manifest = read(path)
            files = selected_files(manifest, dataset)
            if not files:
                raise RuntimeError(f"No {name} assets found for {dataset}")
            assemblies = [item for item in manifest.get("reassemble", [])
                          if any(part["path"] in {f["path"] for f in files} for part in item["parts"])]
            skip = set()
            for item in assemblies:
                if self.valid(inside(ROOT, item["path"]), item["sha256"], item["bytes"]):
                    skip.update(part["path"] for part in item["parts"])
            pending = [f for f in files if f["path"] not in skip
                       and not self.valid(inside(ROOT, f["path"]), f["sha256"], f["bytes"])]
            def remaining(item, suffix):
                temporary = inside(ROOT, item["path"] + suffix)
                present = temporary.stat().st_size if temporary.is_file() else 0
                return max(0, item["bytes"] - present)
            needed = sum(remaining(f, ".part") for f in pending) + sum(
                remaining(item, ".assembling") for item in assemblies if item["parts"][0]["path"] not in skip)
            if shutil.disk_usage(ROOT).free < needed + 2_000_000_000:
                raise RuntimeError(f"Not enough disk space: allow {needed / 1e9 + 2:.1f} GB for remaining {name} files.")
            for index, item in enumerate(pending, 1):
                self.update(message=f"{dataset} {name}: {index}/{len(pending)} — {Path(item['path']).name}")
                self.download(hf_url(pin, item["path"]), inside(ROOT, item["path"]), item["sha256"], item["bytes"])
            for item in assemblies:
                if item["parts"][0]["path"] in skip:
                    continue
                self.update(message="Reassembling CelebA patch features (extra disk space required)…", progress=None)
                target = inside(ROOT, item["path"])
                temporary = target.with_name(target.name + ".assembling")
                with temporary.open("wb") as output:
                    for part in item["parts"]:
                        if output.tell() != part["byte_offset"]:
                            raise RuntimeError("Patch part order differs from its manifest")
                        with inside(ROOT, part["path"]).open("rb") as source:
                            shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                if not self.valid(temporary, item["sha256"], item["bytes"]):
                    raise RuntimeError("Reassembled patch features failed checksum verification")
                os.replace(temporary, target)
                self.valid(target, item["sha256"], item["bytes"])
            write(LOCAL / "verified.json", self.verified)

    def start_web(self):
        self.update("starting", "Starting the Web interface…", progress=None)
        api_port, web_port = available_port(), available_port()
        while web_port == api_port:
            web_port = available_port()
        self.env.update(PROBESCOUT_PYTHON=str(self.python), PROBESCOUT_API_PORT=str(api_port),
                        PROBESCOUT_WEB_PORT=str(web_port))
        process = self.spawn([self.node, WEB / "scripts/dev.mjs"], WEB)
        url = f"http://127.0.0.1:{web_port}"
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Web startup failed. See the log below and retry.")
            try:
                with urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        self.update("ready", "Ready — open ProbeScout to browse and tune.", url=url)
                        if not self.no_browser:
                            webbrowser.open(url)
                        def watch():
                            code = process.wait()
                            if not self.closing and self.state.get("url") == url:
                                self.update("error", f"Web process stopped ({code}). Click Start to restart.", url=None)
                        threading.Thread(target=watch, daemon=True).start()
                        return
            except OSError:
                pass
            time.sleep(.5)
        raise RuntimeError("Web startup timed out. Check the log and retry.")

    def import_archive(self, dataset, archive, destination):
        self.update("extracting", "Inspecting the image archive and checking disk space…", progress=None)
        seven = None
        if str(archive).lower().endswith((".7z", ".7z.001")):
            pin = self.pins["sevenZip"]
            seven = LOCAL / "tools/7zr.exe"
            self.download(pin["url"], seven, pin["sha256"], pin["bytes"])
        catalog = read(ROOT / "dataset/tasks" / dataset / "catalog.json")
        expected, queries = None, {}
        for task in catalog["datasets"][0]["tasks"]:
            bundle = inside(WEB / "public", task["dataRoot"].lstrip("/"))
            manifest = read(bundle / "manifest.json")
            ids = read(inside(bundle, manifest["files"]["imageIds"]["path"]))
            if expected is not None and ids != expected:
                raise ValueError("Task packages disagree on image order.")
            expected = ids
            for query in manifest["query"]["images"]:
                if query["imageId"] in queries and queries[query["imageId"]] != query["sha256"]:
                    raise ValueError("Task packages disagree on query image bytes.")
                queries[query["imageId"]] = query["sha256"]
        if not expected:
            raise ValueError("Task package contains no images.")
        return import_images(archive, destination, dataset, expected, queries, seven=seven,
                             progress=lambda value: self.update(progress=value), log=self.log,
                             cancelled=lambda: self.closing)

    def start(self, options):
        dataset, edition = options.get("dataset"), options.get("edition")
        if dataset not in DATASETS or edition not in {"basic", "full"}:
            raise ValueError("Choose a dataset and edition")
        source_type = options.get("sourceType", "folder")
        if source_type not in {"folder", "archive"}:
            raise ValueError("Choose an image folder or archive.")
        selection = {"sourceType": source_type}
        folder = None
        if source_type == "archive":
            raw = str(options.get("archivePath", "")).strip().strip('"')
            archive = (ROOT / raw).resolve()
            archive_parts(archive)
            destination = (ROOT / (str(options.get("extractTo", "")).strip().strip('"') or "dataset/imported")).resolve()
            if destination.exists() and not destination.is_dir():
                raise ValueError("Choose a folder for extracted images.")
            selection.update(archivePath=str(archive), extractTo=str(destination))
        else:
            raw = str(options.get("imageRoot", "")).strip().strip('"') or str(ROOT / "dataset/raw" / DATASETS[dataset])
            folder = (ROOT / raw).resolve()
            if not folder.is_dir():
                raise ValueError(f"Image folder does not exist: {folder}")
            sample = {"cars": "000001.jpg", "hico": "train2015/HICO_train2015_00000001.jpg", "celeba": "000001.jpg"}[dataset]
            if not (folder / sample).is_file():
                raise ValueError(f"Select the folder containing {sample}. See the image layout below.")
            selection["imageRoot"] = str(folder)
        with self.lock:
            if self.busy:
                raise ValueError("Setup is already running")
            self.busy = True
            self.state.update(stage="environment", message="Preparing…", url=None, logs=[])
        roots = {**self.settings.get("imageRoots", {})}
        if folder is not None:
            roots[dataset] = str(folder)
        inputs = {**self.settings.get("imageInputs", {}), dataset: selection}
        self.settings.update(dataset=dataset, edition=edition, imageRoots=roots, imageInputs=inputs)
        write(LOCAL / "settings.json", self.settings)
        self.update(edition=edition, dataset=dataset, imageRoots=roots, imageInputs=inputs, sourceType=source_type)
        def install():
            try:
                self.stop_children()
                self.dependencies(edition)
                self.install_task(dataset)
                if source_type == "archive":
                    imported = self.import_archive(dataset, archive, destination)
                    roots[dataset] = str(imported)
                    inputs[dataset] = {**selection, "sourceType": "folder", "imageRoot": str(imported)}
                    self.settings.update(imageRoots=roots, imageInputs=inputs)
                    write(LOCAL / "settings.json", self.settings)
                    self.update(imageRoots=roots, imageInputs=inputs)
                self.update("images", "Checking original images and building thumbnails…", progress=None)
                self.run([self.python, ROOT / "scripts/prepare_web.py", "--dataset", dataset])
                if edition == "full":
                    self.full_assets(dataset)
                self.start_web()
            except Exception as error:
                self.stop_children()
                self.update("error", str(error), url=None, progress=None)
            finally:
                with self.lock:
                    self.busy = False
        threading.Thread(target=install, daemon=True).start()


def make_handler(installer, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, value, html=False):
            body = value.encode("utf-8") if html else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self, authenticated=True):
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected or self.headers.get("Origin", f"http://{expected}") != f"http://{expected}":
                self.reply(403, {"error": "Open setup using its local launcher link"})
                return False
            if authenticated and not secrets.compare_digest(self.headers.get("X-Setup-Token", ""), token):
                self.reply(403, {"error": "Reopen setup from the launcher window"})
                return False
            return True

        def do_GET(self):
            if not self.allowed(authenticated=self.path != "/"):
                return
            if self.path == "/":
                self.reply(200, (ROOT / "scripts/setup.html").read_text(encoding="utf-8"), html=True)
            elif self.path == "/status":
                with installer.lock:
                    data = {**installer.state, "busy": installer.busy}
                self.reply(200, data)
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.allowed():
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length < 16384:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if self.path == "/start":
                    installer.start(data)
                    self.reply(202, {"ok": True})
                elif self.path in {"/folder", "/archive"} and os.name == "nt":
                    # A user-clicked folder picker; no paths are interpolated into shell code.
                    choose_archive = self.path == "/archive"
                    picker = (
                        "$picker = New-Object System.Windows.Forms.OpenFileDialog; "
                        "$picker.Title = 'Select the downloaded image archive'; "
                        "$picker.Filter = 'Image archives|*.zip;*.tar;*.tgz;*.tar.gz;*.tar.bz2;*.tar.xz;*.7z;*.7z.001'; "
                        "$picker.CheckFileExists = $true; "
                        if choose_archive else
                        "$picker = New-Object System.Windows.Forms.FolderBrowserDialog; "
                        "$picker.Description = 'Select an image folder or extraction destination'; "
                        "$picker.ShowNewFolderButton = $true; "
                    )
                    selected = "$picker.FileName" if choose_archive else "$picker.SelectedPath"
                    result = subprocess.run(["powershell.exe", "-NoProfile", "-STA", "-Command",
                        "Add-Type -AssemblyName System.Windows.Forms; " + picker +
                        "if ($picker.ShowDialog() -eq 'OK') { "
                        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; " + selected + " }"],
                        capture_output=True, text=True, encoding="utf-8-sig", errors="replace",
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    self.reply(200, {"path": result.stdout.strip()})
                else:
                    self.reply(404, {"error": "Not found"})
            except (ValueError, OSError) as error:
                self.reply(400, {"error": str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=("basic", "full"), default="basic")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--configure", action="store_true", help="Show setup without starting the saved selection")
    args = parser.parse_args()
    job = supervise_windows_children()
    LOCAL.mkdir(parents=True, exist_ok=True)
    # Keep one owner of the installer, downloads and child processes per checkout.
    lockfile = (LOCAL / "launcher.lock").open("a+b")
    if os.name == "nt":
        import msvcrt
        lockfile.seek(0)
        try:
            msvcrt.locking(lockfile.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            endpoint = read(LOCAL / "endpoint.json", {}).get("url")
            if endpoint and not args.no_browser:
                webbrowser.open(endpoint)
            print("ProbeScout is already running. Use its setup page to switch datasets or editions.")
            return
    installer = Installer(args.edition, args.no_browser)
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(installer, token))
    endpoint = f"http://127.0.0.1:{server.server_port}/#{token}"
    write(LOCAL / "endpoint.json", {"url": endpoint})
    print(f"Setup: {endpoint}\nKeep this window open. Press Ctrl+C to stop ProbeScout.", flush=True)
    if not args.no_browser:
        webbrowser.open(endpoint)
    if installer.settings.get("edition") == args.edition and not args.configure:
        try:
            dataset = installer.settings["dataset"]
            selection = installer.settings.get("imageInputs", {}).get(dataset, {
                "sourceType": "folder", "imageRoot": installer.settings.get("imageRoots", {}).get(dataset, "")})
            installer.start({**selection, "dataset": dataset, "edition": args.edition})
        except ValueError as error:
            installer.update("error", str(error))
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        installer.closing = True
        installer.stop_children()
        server.server_close()
        lockfile.close()


if __name__ == "__main__":
    main()
