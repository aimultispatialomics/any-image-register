"""Command-line interface for any-image-register."""

import argparse

import numpy as np

from .difficulty import DifficultyFactors, describe
from .io import read_image
from .metrics import mutual_information, ncc


def _cmd_metrics(args):
    fixed = read_image(args.fixed)
    moving = read_image(args.moving)
    if fixed.shape != moving.shape:
        raise SystemExit(f"shape mismatch: {fixed.shape} vs {moving.shape} — warp first")
    print(f"NCC: {ncc(fixed, moving):.4f}")
    print(f"MI:  {mutual_information(fixed, moving):.4f} nats")


def _cmd_difficulty(args):
    factors = DifficultyFactors(
        modality_gap=args.modality_gap,
        deformation=args.deformation,
        resolution_gap=args.resolution_gap,
        tissue_artifacts=args.tissue_artifacts,
    )
    print(describe(factors))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="any-image-register")
    sub = parser.add_subparsers(dest="command", required=True)

    p_metrics = sub.add_parser("metrics", help="similarity between two aligned images")
    p_metrics.add_argument("fixed")
    p_metrics.add_argument("moving")
    p_metrics.set_defaults(func=_cmd_metrics)

    p_diff = sub.add_parser("difficulty", help="score a registration scenario")
    for name in ("modality_gap", "deformation", "resolution_gap", "tissue_artifacts"):
        p_diff.add_argument(f"--{name.replace('_', '-')}", type=int, default=0, choices=(0, 1, 2))
    p_diff.set_defaults(func=_cmd_difficulty)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
