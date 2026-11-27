"""Command-line entry point."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from da_mocka import __version__
from da_mocka.config import ContractError, Stratification, load_config
from da_mocka.demo import make_demo
from da_mocka.generation import generate
from da_mocka.io import load_dataset


def _stratify_args(args, spec: Stratification) -> Stratification:
    changes = {}
    names = ("allocation", "min_stratum_size", "rare_strata", "missing_strata", "pooling_strength")
    for name in names:
        value = getattr(args, name)
        if value is not None:
            changes[name] = value
    if args.stratify is not None:
        changes["columns"] = tuple(args.stratify)
    if args.balance is not None:
        changes["balance"] = tuple(args.balance)
        if args.allocation is None:
            changes["allocation"] = "marginal"
    if args.include:
        include = dict(spec.include)
        for selection in args.include:
            name, sep, values = selection.partition("=")
            if not sep or not name or not values:
                raise ContractError("Expected --include COLUMN=VALUE1,VALUE2")
            values = values.removeprefix("[").removesuffix("]")
            items = tuple(item.strip() for item in values.split(","))
            if not all(items) or len(set(items)) != len(items):
                raise ContractError("Selected outcomes must be unique and nonempty")
            include[name] = items
        changes["include"] = include
    return replace(spec, **changes)


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
    gen = commands.add_parser("generate", help="Generate local unreviewed synthetic candidates")
    gen.add_argument("--config", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--n-samples", type=int)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--stratify", nargs="+")
    gen.add_argument("--include", action="append", metavar="COLUMN=VALUES")
    gen.add_argument(
        "--allocation", choices=["proportional", "balanced", "original", "custom", "marginal"]
    )
    gen.add_argument("--balance", nargs="+")
    gen.add_argument("--min-stratum-size", type=int)
    gen.add_argument("--rare-strata", choices=["error", "suppress"])
    gen.add_argument("--missing-strata", choices=["error", "exclude"])
    gen.add_argument("--pooling-strength", type=float)
    gen.add_argument("--engine", choices=["independent", "joint_latent"])
    gen.add_argument("--latent-rank", type=int)
    gen.add_argument("--shrinkage", type=float)
    gen.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            make_demo(args.output, args.n_samples, args.seed)
            print("Artificial demo created.")
        elif args.command == "generate":
            config = load_config(args.config)
            config = replace(config, stratification=_stratify_args(args, config.stratification))
            options = {
                name: value
                for name, value in (
                    ("name", args.engine),
                    ("rank", args.latent_rank),
                    ("shrinkage", args.shrinkage),
                    ("batch_size", args.batch_size),
                )
                if value is not None
            }
            config = replace(config, engine=replace(config.engine, **options))
            generate(config, args.output, args.n_samples, args.seed)
            print("Synthetic candidates created. Not reviewed for disclosure or release.")
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
