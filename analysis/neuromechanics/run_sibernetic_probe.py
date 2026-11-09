"""Run a bounded upstream body probe and keep stdout, timing and failure state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import site
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", default="worm_alone_half_resolution")
    p.add_argument("--duration-s", type=float, default=.00005)
    p.add_argument("--timeout-s", type=float, default=90.)
    p.add_argument("--logstep", type=int, default=1)
    args = p.parse_args()
    if args.duration_s <= 0 or args.timeout_s <= 0 or args.logstep < 1:
        p.error("positive duration, timeout and logstep required")
    src, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "design.json").exists():
        raise FileExistsError("use a new output directory to preserve prior results")
    binary = src / "Release/Sibernetic"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(src), *site.getsitepackages()])
    command = [str(binary), "-no_g", "-f", args.case,
               f"timelimit={args.duration_s}", "-l_to",
               f"logstep={args.logstep}", f"lpath={out}"]
    design = dict(command=command, source=str(src), requested_duration_s=args.duration_s,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  configuration_sha256=hashlib.sha256((src / "configuration" / args.case).read_bytes()).hexdigest(),
                  pythonpath=env["PYTHONPATH"], scope="upstream body with prescribed muscle drive; no thermotaxis")
    (out / "design.json").write_text(json.dumps(design, indent=2))
    t = time.monotonic()
    with (out / "run.log").open("w") as log:
        try:
            run = subprocess.run(command, cwd=src, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=args.timeout_s)
            result = dict(returncode=run.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            result = dict(returncode=None, timed_out=True)
    result.update(wall_s=time.monotonic()-t,
                  output_files=[str(f.relative_to(out)) for f in sorted(out.rglob("*")) if f.is_file()],
                  scope=design["scope"])
    (out / "process.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if result["returncode"] != 0 or result["timed_out"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
