"""Command-line entry point for reproducible staged experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_config
from .phase1_config import load_phase1_config
from .phase1_ensemble import write_phase1_ensemble_bundle
from .phase1_runner import write_phase1_result_bundle
from .runner import write_result_bundle


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermotaxis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a versioned experiment config")
    validate.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run", help="write a reproducible phase-0 contract probe")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repository", type=Path, default=Path.cwd())
    phase1_validate = subparsers.add_parser(
        "phase1-validate", help="validate a phase-1 point-thermotaxis config"
    )
    phase1_validate.add_argument("--config", type=Path, required=True)
    phase1_run = subparsers.add_parser(
        "phase1-run", help="write a reproducible phase-1 trajectory and metric bundle"
    )
    phase1_run.add_argument("--config", type=Path, required=True)
    phase1_run.add_argument("--output-root", type=Path, required=True)
    phase1_run.add_argument("--repository", type=Path, default=Path.cwd())
    phase1_ensemble = subparsers.add_parser(
        "phase1-ensemble", help="write a paired multi-seed phase-1 summary bundle"
    )
    phase1_ensemble.add_argument("--config", type=Path, required=True)
    phase1_ensemble.add_argument("--replicates", type=_positive_int, required=True)
    phase1_ensemble.add_argument("--output-root", type=Path, required=True)
    phase1_ensemble.add_argument("--repository", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command.startswith("phase1-"):
        phase1_config = load_phase1_config(args.config)
        if args.command == "phase1-validate":
            print(
                f"valid phase1 schema={phase1_config.schema_version} "
                f"Re={phase1_config.reynolds_number:.6g}"
            )
            return 0
        if args.command == "phase1-ensemble":
            run_dir = write_phase1_ensemble_bundle(
                phase1_config, args.replicates, args.output_root, args.repository
            )
        else:
            run_dir = write_phase1_result_bundle(
                phase1_config, args.output_root, args.repository
            )
        print(run_dir)
        return 0
    config = load_config(args.config)
    if args.command == "validate":
        print(f"valid schema={config.schema_version} Re={config.reynolds_number:.6g}")
        return 0
    run_dir = write_result_bundle(config, args.output_root, args.repository)
    print(run_dir)
    return 0
