"""Build the audited upstream macOS target with an explicit local Python ABI.

Apply recorded compatibility patches first. This script changes no source and
does not install system libraries. Its output names are caller-selected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Darwin":
        p.error("this build recipe is audited for macOS only")
    src, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "build.json").exists():
        raise FileExistsError("use a new build record directory")
    libdir = sysconfig.get_config_var("LIBDIR")
    # Relocatable bundled Python can retain its build-time /install prefix.
    if not libdir or not Path(libdir).is_dir():
        libdir = str(Path(sys.base_prefix) / "lib")
    version = sysconfig.get_config_var("LDVERSION")
    libs = f"-L{libdir} -Wl,-rpath,{libdir} -lpython{version} -framework OpenGL -framework GLUT -framework OpenCL"
    command = ["make", "-f", "makefile.OSX", "-j2", "all",
               f"PYTHONHEADERDIR={sysconfig.get_path('include')}",
               f"PYTHONLIBDIR={libdir}", f"LIBS={libs}"]
    t = time.monotonic()
    with (out / "build.log").open("w") as log:
        proc = subprocess.run(command, cwd=src, stdout=log, stderr=subprocess.STDOUT)
    diff = subprocess.check_output(["git", "diff", "--", "inc", "src"], cwd=src, text=True)
    (out / "applied.patch").write_text(diff)
    binary = src / "Release/Sibernetic"
    record = dict(command=command, wall_s=time.monotonic()-t, returncode=proc.returncode,
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip(),
                  applied_patch_sha256=hashlib.sha256(diff.encode()).hexdigest(),
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest() if proc.returncode == 0 else None)
    (out / "build.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
