"""Command-line entry point for reproducible staged experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_config
from .runner import write_result_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermotaxis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a versioned experiment config")
    validate.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run", help="write a reproducible phase-0 contract probe")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repository", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        print(f"valid schema={config.schema_version} Re={config.reynolds_number:.6g}")
        return 0
    run_dir = write_result_bundle(config, args.output_root, args.repository)
    print(run_dir)
    return 0
