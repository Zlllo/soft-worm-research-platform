"""Archive the stepper's selected source, traces and checks without binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--previous", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    a = p.parse_args()
    base, dest = a.run_dir.resolve(), a.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    copies = {"summary.json":base/"audit/summary.json",
        "interface-checks.png":base/"audit/interface-checks.png",
        "interface-checks.svg":base/"audit/interface-checks.svg",
        "pytest.log":base/"pytest.log",
        "implementation-correction.json":base/"implementation-correction.json"}
    for name, path in copies.items():
        shutil.copy2(path, dest/name)
    files = {}
    for name in ["build_boyle_worker.py", "audit_boyle_worker.py", "archive_boyle_worker.py",
                 "run_boyle_probe.py", "build_boyle_source.py"]:
        key = "analysis/neuromechanics/"+name
        files[key] = ROOT/key
    for name in ["boyle_body.py", "boyle_output.py", "thermal_body.py", "__init__.py"]:
        key = "src/thermotaxis/neuromechanics/"+name
        files[key] = ROOT/key
    for name in ["test_boyle_body.py", "test_boyle_output.py", "test_neuromechanics_mapping.py", "test_connectome.py"]:
        key = "tests/"+name
        files[key] = ROOT/key
    for name in ["configs/neuromechanics/boyle-source-lock.json",
                 "analysis/connectome_pilot.py",
                 "src/thermotaxis/__init__.py", "src/thermotaxis/config.py",
                 "src/thermotaxis/connectome.py", "src/thermotaxis/connectome_policy.py",
                 "src/thermotaxis/neuromechanics/muscle_mapping.py",
                 "data/connectome/cook2019_herm.json", "data/connectome/herm_neurons.json",
                 "data/connectome/upstream_validation.yaml", "data/connectome/README.md",
                 "data/connectome/OPENWORM_LICENSE.txt",
                 "docs/reports/neuromechanics-stepper-2026-10-27.md"]:
        files[name] = ROOT/name
    evidence_paths = list(base.glob("*"))
    for folder in ("build", "build-v2", "audit"):
        evidence_paths.extend((base/folder).rglob("*"))
    for path in evidence_paths:
        if path.is_file() and path.suffix in (".npz", ".json", ".log", ".cc", ".patch", ".png", ".svg"):
            files["evidence/"+str(path.relative_to(base))] = path
    for name in ["WormSim/Model/worm.cc", "WormSim/README.md", "WormSim/sundials-2.3.0/LICENSE"]:
        files["upstream-reference/"+name] = a.source.resolve()/name
    for name in ["simdata.csv", "design.json", "source.patch"]:
        files["previous-accepted-state/"+name] = a.previous.resolve()/name
    files["README-reproduction.md"] = base/"README-reproduction.md"
    manifest = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
    archive = dest/"reproduction.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(files.items()):
            z.write(path, name)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("archive failed CRC check")
        for name,sha in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError("archive hash mismatch: "+name)
    (dest/"manifest.json").write_text(json.dumps(dict(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=manifest,
        selected_build="evidence/build-v2", superseded_build="evidence/build"), indent=2))
    print(json.dumps(dict(files=len(files), bytes=archive.stat().st_size)))


if __name__ == "__main__":
    main()
