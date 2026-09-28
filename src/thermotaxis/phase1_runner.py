"""Reproducible result bundles for phase-1 thermotaxis runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
from typing import Optional

from . import __version__
from .phase1 import run_phase1_experiment
from .phase1_config import Phase1ExperimentConfig


def _git_revision(repository: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_phase1_result_bundle(
    config: Phase1ExperimentConfig, output_root: Path, repository: Path
) -> Path:
    result = run_phase1_experiment(config)
    run_id = result["config_sha256"][:12]
    run_dir = Path(output_root) / f"{config.name}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thermotaxis_core_version": __version__,
        "config_sha256": result["config_sha256"],
        "git_revision": _git_revision(Path(repository)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "purpose": config.purpose,
        "result_type": result["result_type"],
    }
    for name, payload in (
        ("config.resolved.json", config.to_dict()),
        ("manifest.json", manifest),
        ("result.json", result),
    ):
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return run_dir
