"""Command-line entry point."""

import argparse
import sys
from pathlib import Path

from da_mocka import __version__
from da_mocka.config import ContractError, load_config
from da_mocka.demo import make_demo
from da_mocka.io import load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="da-mocka")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate an entirely invented cohort")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--n-samples", type=int, default=2000)
    demo.add_argument("--seed", type=int, default=42)
    check = commands.add_parser("check-input", help="Validate local input contracts")
    check.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            make_demo(args.output, args.n_samples, args.seed)
            print("Artificial demo created.")
        else:
            data = load_dataset(load_config(args.config))
            print(
                f"Input valid: {len(data.phenotypes)} samples, "
                f"{data.microbiome.shape[1]} microbial features."
            )
    except ContractError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("Error: filesystem operation failed", file=sys.stderr)
        return 2
    return 0
