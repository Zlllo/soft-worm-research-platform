"""Archive bounded stage-0 evidence, excluding environments and whole checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    run, dest = a.run_dir.resolve(), a.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "reproduction.zip"
    if archive.exists():
        raise FileExistsError("existing evidence archive must be preserved")
    for name in ("summary.json", "muscle-mapping.json", "requirements-lock.txt", "stage0-probes.png", "stage0-probes.svg"):
        shutil.copy2(run / name, dest / name)
    files = {}
    for path in (ROOT / "analysis/neuromechanics").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            files[str(path.relative_to(ROOT))] = path
    for folder in ("src/thermotaxis/neuromechanics", "configs/neuromechanics"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(ROOT))] = path
    files["tests/test_neuromechanics_mapping.py"] = ROOT / "tests/test_neuromechanics_mapping.py"
    for path in run.iterdir():
        if path.is_file() and path.suffix in (".json", ".txt", ".png", ".svg") and path.name != "requirements-installed.txt":
            files["evidence/" + path.name] = path
    for path in (run / "logs").glob("*.log"):
        files["evidence/logs/" + path.name] = path
    for folder in (run / "outputs").iterdir():
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix in (".nml", ".xml", ".json", ".npz", ".log"):
                files["evidence/outputs/" + folder.name + "/" + path.name] = path
        for name in ("position_buffer.txt", "muscles_activity_buffer.txt"):
            path = folder / name
            if path.exists():
                files["evidence/outputs/" + folder.name + "/" + path.name] = path
    for folder in run.glob("build-*"):
        for path in folder.iterdir():
            if path.is_file():
                files["evidence/" + folder.name + "/" + path.name] = path
    for name in ("c302", "sibernetic"):
        files["licenses/" + name + "-LICENSE"] = run / "upstream" / name / "LICENSE"
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        for name, sha in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise RuntimeError("archive hash mismatch: " + name)
    (dest / "manifest.json").write_text(json.dumps(dict(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=manifest), indent=2))
    print(json.dumps({"archive":str(archive), "files":len(files), "bytes":archive.stat().st_size}))


if __name__ == "__main__":
    main()
