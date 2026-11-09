"""Run the pinned upstream C1 Syns example without GUI; record actual traces.

Run in the isolated environment described in the stage-0 report. This is a
neuron/muscle electrical example, not a locomotion or thermotaxis experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt-ms", type=float, default=.05)
    parser.add_argument("--engine", choices=("jneuroml", "neuron"), default="jneuroml")
    parser.add_argument("--disconnect-nmj", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "design.json").exists():
        raise FileExistsError("use a new output directory to preserve prior results")
    os.environ.setdefault("MPLCONFIGDIR", str(out / "mpl-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out / "cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import jdk4py
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent),
                                          str(jdk4py.JAVA.parent), os.environ["PATH"]])
    if args.engine == "neuron":
        import neuron
        os.environ["NEURON_HOME"] = str(Path(neuron.__file__).parent / ".data")
    import c302
    import c302.c302_Syns as example
    import numpy as np
    from neuroml.writers import NeuroMLWriter
    from pyneuroml import pynml

    design = dict(example="c302.c302_Syns.setup", parameter_set="C1", dt_ms=args.dt_ms,
                  engine=args.engine, c302_version=c302.__version__,
                  disconnect_AS2_MDL07=args.disconnect_nmj,
                  scope="upstream electrical neuron/muscle example; no body or feedback")
    (out / "design.json").write_text(json.dumps(design, indent=2))
    started = time.monotonic()
    *_, doc = example.setup("C1", generate=True, dt=args.dt_ms,
                           target_directory=str(out), verbose=False)
    if args.disconnect_nmj:
        network = doc.networks[0]
        removed = [x for x in network.continuous_projections
                   if x.presynaptic_population == "AS2" and x.postsynaptic_population == "MDL07"]
        if len(removed) != 1:
            raise RuntimeError("expected exactly one AS2 to MDL07 projection")
        network.continuous_projections.remove(removed[0])
        NeuroMLWriter.write(doc, str(out / "c302_C1_Syns.net.nml"))
    generation_s = time.monotonic() - started
    started = time.monotonic()
    if args.engine == "jneuroml":
        data = pynml.run_lems_with_jneuroml("LEMS_c302_C1_Syns.xml", nogui=True,
                    load_saved_data=True, plot=False, exec_in_dir=str(out),
                    exit_on_fail=False, verbose=False)
    else:
        # jNeuroML's Java launcher bypasses the wheel's relocation wrapper.
        # Generate only, then compile with the installed Python-wheel entry point.
        ok = pynml.run_lems_with_jneuroml_neuron("LEMS_c302_C1_Syns.xml", nogui=True,
                 only_generate_scripts=True, compile_mods=False, load_saved_data=False,
                 exec_in_dir=str(out), exit_on_fail=False, verbose=False)
        if not ok:
            raise RuntimeError("NEURON script generation failed")
        for command in ([str(Path(sys.executable).parent / "nrnivmodl")],
                        [sys.executable, "LEMS_c302_C1_Syns_nrn.py", "-nogui"]):
            subprocess.run(command, cwd=out, check=True)
        from pyneuroml.runners import reload_saved_data
        data = reload_saved_data("LEMS_c302_C1_Syns.xml", base_dir=str(out), plot=False)
    wall_s = time.monotonic() - started
    if not isinstance(data, dict) or "t" not in data:
        raise RuntimeError(f"simulation did not return traces: {data!r}")
    arrays = {k: np.asarray(v) for k, v in data.items()}
    t = arrays["t"]
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise RuntimeError("time series is empty or not increasing")
    if not all(np.isfinite(v).all() and len(v) == len(t) for v in arrays.values()):
        raise RuntimeError("invalid or inconsistent simulation traces")
    np.savez_compressed(out / "traces.npz", **arrays)
    stats = {k: {"min": float(v.min()), "max": float(v.max()),
                 "range": float(np.ptp(v))} for k, v in arrays.items() if k != "t"}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in out.iterdir() if p.is_file() and p.suffix in (".nml", ".xml", ".dat", ".npz")}
    result = dict(status="completed", generation_s=generation_s, simulation_wall_s=wall_s,
                  samples=len(t), first_time_s=float(t[0]), last_time_s=float(t[-1]),
                  simulated_s=float(t[-1]-t[0]), trace_stats=stats, files_sha256=hashes,
                  scope=design["scope"], units="jNeuroML/NeuroML output SI; time s, voltage V, calcium mol/m3")
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("files_sha256", "trace_stats")}))


if __name__ == "__main__":
    main()
