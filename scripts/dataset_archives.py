"""Import original image archives directly into a reusable local image folder."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile
import zlib

FORMATS = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".7z", ".7z.001")


def archive_parts(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Archive does not exist: {path}")
    if not path.name.lower().endswith(FORMATS):
        raise ValueError("Choose a ZIP, TAR, TGZ or 7z archive. For split 7z files, choose .7z.001.")
    if path.name.lower().endswith(".7z.001"):
        prefix = path.name[:-3]
        parts = sorted(path.parent.glob(prefix + "[0-9][0-9][0-9]"))
        if [int(p.suffix[1:]) for p in parts] != list(range(1, len(parts) + 1)):
            raise ValueError("A 7z volume is missing. Keep all numbered parts in the same folder.")
        return parts
    return [path]


def member_name(name):
    normalized = str(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    if (path.is_absolute() or ":" in normalized or ".." in path.parts
            or any(ord(c) < 32 for c in normalized)):
        raise ValueError(f"Unsafe archive path: {name}")
    return path.as_posix()


def target_path(root, name):
    path = (root / member_name(name)).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Image path escapes its destination: {name}")
    return path


@dataclass
class Entry:
    name: str
    size: int
    crc: str | None = None
    member: object = None


def run_seven(executable, arguments, progress=None, cancelled=lambda: False):
    """Run the pinned tool without a shell; also works with Unicode paths."""
    process = subprocess.Popen([str(executable), *map(str, arguments)], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    lines = []
    last = 0
    try:
        for line in process.stdout:
            if cancelled():
                raise RuntimeError("Image extraction stopped")
            if progress:
                percent = re.search(r"(\d+)%", line)
                if percent and time.monotonic() - last > .5:
                    progress({"file": "Extracting images", "bytes": int(percent[1]), "total": 100, "unit": "percent"})
                    last = time.monotonic()
                lines = (lines + [line])[-12:]
            else:
                lines.append(line)
        code = process.wait()
        if code:
            raise ValueError("7z archive could not be read. Check that all volumes are complete and "
                             "unencrypted. " + "".join(lines[-8:]).strip()[:700])
        return "".join(lines)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        process.stdout.close()


class ImageArchive:
    def __init__(self, path, seven=None, cancelled=lambda: False):
        self.path = Path(path)
        self.seven = seven
        self.cancelled = cancelled
        self.container = None
        self.kind = "7z" if self.path.name.lower().endswith((".7z", ".7z.001")) else "zip" if self.path.suffix.lower() == ".zip" else "tar"

    def __enter__(self):
        try:
            self.entries = self.list_entries()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self.container:
            self.container.close()

    def list_entries(self):
        entries = []
        if self.kind == "7z":
            text = run_seven(self.seven, ["l", "-slt", "-ba", "-sccUTF-8", "-p-", "--", self.path],
                             cancelled=self.cancelled)
            for block in re.split(r"\n\s*\n", text.strip()):
                fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
                if "Path" not in fields:
                    continue
                name = member_name(fields["Path"])
                attributes = fields.get("Attributes", "")
                if fields.get("Encrypted") == "+":
                    raise ValueError("Encrypted image archives are not supported.")
                if (fields.get("Symbolic Link") or fields.get("Hard Link") or "L" in attributes
                        or fields.get("Mode", "").startswith("l")):
                    raise ValueError("Image archives must not contain links.")
                if fields.get("Folder") == "+" or "D" in attributes.split(" ")[0]:
                    continue
                entries.append(Entry(name, int(fields["Size"]), fields.get("CRC") or None))
        elif self.kind == "zip":
            self.container = zipfile.ZipFile(self.path)
            for item in self.container.infolist():
                name = member_name(item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Image archives must not contain links.")
                if item.flag_bits & 1:
                    raise ValueError("Encrypted image archives are not supported.")
                if not item.is_dir():
                    entries.append(Entry(name, item.file_size, f"{item.CRC:08X}", item))
        else:
            self.container = tarfile.open(self.path, "r:*")
            for item in self.container:
                if self.cancelled():
                    raise RuntimeError("Image extraction stopped")
                name = member_name(item.name)
                if not item.isfile() and not item.isdir():
                    raise ValueError("Image archives must contain regular files and directories only.")
                if item.isfile():
                    entries.append(Entry(name, item.size, member=item))
        return entries

    @contextmanager
    def open(self, entry):
        stream = self.container.open(entry.member) if self.kind == "zip" else self.container.extractfile(entry.member)
        with stream:
            yield stream


def match_images(entries, expected):
    """Strip archive wrapper folders, keeping the exact published image IDs."""
    lookup = {}
    for name in expected:
        name = member_name(name)
        lookup.setdefault(PurePosixPath(name).name, []).append(name)
    matches = {}
    for entry in entries:
        for relative in lookup.get(PurePosixPath(entry.name).name, ()):
            if entry.name == relative or entry.name.endswith("/" + relative):
                if relative in matches:
                    raise ValueError(f"Archive contains more than one copy of {relative}.")
                matches[relative] = entry
    missing = set(expected) - matches.keys()
    if missing:
        raise ValueError(f"Archive does not match this dataset: {len(missing)} required images missing; "
                         f"first: {min(missing)}. Use car_ims, HICO-DET, or CelebA In-The-Wild as appropriate.")
    return matches


def file_stamp(path):
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def crc_file(path):
    checksum = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
    return f"{checksum:08X}"


def import_images(archive, destination, dataset, expected, queries, *, seven=None,
                  progress=lambda value: None, log=lambda value: None, cancelled=lambda: False):
    parts = archive_parts(archive)
    identity = {"version": 1, "dataset": dataset,
                "sources": [[str(p), *file_stamp(p)] for p in parts],
                "imageIdsSha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                "queries": queries}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    parent = Path(destination).resolve()
    base = parent / f"probescout-{dataset}-{key}"
    if not base.resolve().is_relative_to(parent) or base.is_symlink():
        raise ValueError("Import directory must stay inside the selected destination.")
    marker = base / "source.json"
    if base.exists() and (not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity):
        raise ValueError(f"Import directory already exists with different contents: {base}")
    with ImageArchive(parts[0], seven, cancelled) as source:
        matches = match_images(source.entries, expected)
        # This dataset-specific directory avoids overwriting any existing image collection.
        base.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        output = base / "images"
        output.mkdir(exist_ok=True)
        journal = base / "completed.jsonl"
        completed = {}
        if journal.is_file():
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    name, stamp = json.loads(line)
                    completed[name] = stamp
                except (ValueError, TypeError):
                    continue  # An interrupted last line can be reconstructed from the archive.
        pending = {}
        for name, entry in matches.items():
            target = target_path(output, name)
            if target.is_file() and completed.get(name) == file_stamp(target) and target.stat().st_size == entry.size:
                continue
            # 7z may have completed files before an interrupted process; its listing supplies their CRCs.
            if source.kind == "7z" and target.is_file() and target.stat().st_size == entry.size and entry.crc:
                if crc_file(target) == entry.crc:
                    completed[name] = file_stamp(target)
                    continue
            pending[name] = entry
        needed = sum(entry.size for entry in pending.values())
        free = shutil.disk_usage(parent).free
        if free < needed + 1024**3:
            raise ValueError(f"Extraction needs {needed / 1024**3:.2f} GiB plus 1 GiB free space; "
                             f"this drive has {free / 1024**3:.2f} GiB. Choose another extraction folder.")
        log(f"Image import: {len(pending)} to extract, {len(matches) - len(pending)} reused; "
            f"{needed / 1024**3:.2f} GiB additional images. Destination: {output}")
        with journal.open("a", encoding="utf-8") as receipts:
            def record(name):
                path = target_path(output, name)
                if path.stat().st_size != matches[name].size:
                    raise ValueError(f"Incomplete extracted image: {name}")
                if name in queries:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest != queries[name]:
                        raise ValueError(f"Image variant differs from the released dataset: {name}. "
                                         "Use the original images; CelebA requires In-The-Wild.")
                receipts.write(json.dumps([name, file_stamp(path)]) + "\n")

            if source.kind == "7z" and pending:
                groups = {}
                for name, entry in pending.items():
                    groups.setdefault(str(PurePosixPath(name).parent), []).append((name, entry))
                for group, rows in groups.items():
                    directory = output if group == "." else target_path(output, group)
                    directory.mkdir(parents=True, exist_ok=True)
                    listing = base / "selected-files.txt"
                    listing.write_text("".join(entry.name + "\n" for _, entry in rows), encoding="utf-8")
                    run_seven(seven, ["e", "-y", "-aoa", "-spd", "-p-", "-scsUTF-8", "-sccUTF-8",
                              "-bsp1", "-bso0", f"-o{directory}", f"-i@{listing}", "--", parts[0]],
                              progress, cancelled)
                    for name, _ in rows:
                        record(name)
                    receipts.flush()
            elif source.kind != "7z":
                done = 0
                last = 0
                for name, entry in pending.items():
                    if cancelled():
                        raise RuntimeError("Image extraction stopped")
                    target = target_path(output, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".extracting")
                    with source.open(entry) as input_file, temporary.open("wb") as image:
                        shutil.copyfileobj(input_file, image, 1024 * 1024)
                    if temporary.stat().st_size != entry.size:
                        raise ValueError(f"Incomplete archive image: {name}")
                    os.replace(temporary, target)
                    record(name)
                    done += entry.size
                    if time.monotonic() - last > .5:
                        progress({"file": f"Extracting {name}", "bytes": done, "total": needed})
                        receipts.flush()
                        last = time.monotonic()
        for name, checksum in queries.items():
            with target_path(output, name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                    raise ValueError(f"Query image differs from the released dataset: {name}")
        progress({"file": "Images ready", "bytes": len(matches), "total": len(matches), "unit": "images"})
        return output
