#!/usr/bin/env python3
"""Fit the Phase 4 dmf forecasters on the P3-D2 dev pool. Logic: :mod:`rld.deck.forecast_fit`.

Examples::

    # the closed-form pair (seconds)
    python scripts/fit_dmf_forecasters.py --models dlinear_ols residual_interval
    # the TCNs, three seeds each, on the GPU (hours; launch in the background)
    python scripts/fit_dmf_forecasters.py --models tcn tcn_quantile --seeds 0 1 2 --device cuda
    # pad-v_z conformal calibration only, on an already-fitted interval model
    python scripts/fit_dmf_forecasters.py --calibrate-only --models tcn_quantile
    # toy-size pipeline check into a scratch directory
    python scripts/fit_dmf_forecasters.py --smoke --out /tmp/dmf_smoke

``--smoke`` writes its record into ``<out>/record/`` unless ``--record-dir`` is given, so a
smoke run can never overwrite ``results/forecast/``.
"""

import argparse
import sys
from pathlib import Path

import torch

from rld.deck.forecast_fit import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_RECORD_DIR,
    FORECASTERS,
    SMOKE,
    calibrate_only,
    fit_forecasters,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(FORECASTERS), choices=list(FORECASTERS))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="SGD seeds (default: 0 1 2)."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="SGD device."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="DataLoader workers (dmf default 16)."
    )
    parser.add_argument("--smoke", action="store_true", help="Toy-size pipeline check.")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Only (re)fit the pad-v_z conformal factors of already-fitted interval models.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    smoke = SMOKE if args.smoke else None
    record_dir = args.record_dir
    if record_dir is None:
        record_dir = args.out / "record" if args.smoke else DEFAULT_RECORD_DIR
    workers = args.workers if args.workers is not None else (0 if args.smoke else None)
    if args.calibrate_only:
        calibrate_only(
            models=args.models,
            corpus_root=args.corpus,
            out_root=args.out,
            record_dir=record_dir,
            smoke=smoke,
        )
        return 0
    fit_forecasters(
        models=args.models,
        seeds=args.seeds,
        corpus_root=args.corpus,
        out_root=args.out,
        record_dir=record_dir,
        device=args.device,
        num_workers=workers,
        smoke=smoke,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
