"""Build the two source-only Windows downloads from a committed Git revision."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/releases")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise SystemExit("Commit the reviewed source changes before packaging.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "archive", "--format=zip", "HEAD"], cwd=ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        for edition in ("Starter", "Full"):
            target = args.output / f"ProbeScout-{edition}-Windows.zip"
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
                for item in archive.infolist():
                    data = archive.read(item)
                    if item.filename == "start.bat" and edition == "Full":
                        data = data.replace(b"-Edition basic", b"-Edition full")
                    output.writestr("ProbeScout/" + item.filename, data)
                output.writestr("ProbeScout/EDITION.txt",
                    f"ProbeScout {edition}\nSource: {commit}\n"
                    "Extract this ZIP, then double-click start.bat.\n"
                    "Select your original image folder. Matching assets download on first launch.\n")
            results.append({"edition": edition, "file": target.name, "bytes": target.stat().st_size,
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {"commit": commit, "packages": results, "containsOriginalImages": False,
                "containsAssetCaches": False, "downloadsAssetsOnFirstLaunch": True}
    (args.output / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
