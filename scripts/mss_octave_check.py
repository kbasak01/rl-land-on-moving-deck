#!/usr/bin/env python3
"""Octave parity check for the MSS transfer arm, with every path outside third_party/.

A thin wrapper over :func:`rld.deck.mss_octave.run_parity`. dmf's own
``scripts/mss_octave_check.py`` resolves its m-files and the MSS clone relative to dmf's
tree, so it cannot be pointed at ``artifacts/mss/upstream``. See that module's docstring.

Writes ``<out-dir>/octave_parity.csv`` (dmf's row schema plus ``rld_vs_octave`` rows). If
Octave is absent it writes a one-row ``skipped`` table and exits 0, as dmf's script does.
Exits 1 if any row fails. These are simulated trajectories, not measurements of a real ship.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from rld.deck.mss import MSS_CONFIG, MSS_ROOT, MSS_UPSTREAM_SHA, load_mss_config
from rld.deck.mss_octave import OctaveUnavailableError, run_parity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", type=Path, default=MSS_CONFIG)
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=MSS_ROOT,
        help="Directory holding the MSS clone as upstream/; dmf's m-files are staged here.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/mss"))
    parser.add_argument("--n-steps", type=int, default=300, help="Samples per cell, 10 Hz full.")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Max relative deviation.")
    parser.add_argument("--octave", default="octave", help="Octave executable.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "octave_parity.csv"
    cfg = load_mss_config(args.config)
    print(f"[parity] MSS clone pinned at {MSS_UPSTREAM_SHA}", file=sys.stderr)
    try:
        table = run_parity(
            cfg,
            args.stage_dir.resolve(),
            n_steps=args.n_steps,
            tolerance=args.tolerance,
            octave=args.octave,
        )
    except OctaveUnavailableError as exc:
        print(f"[parity] {exc} -> SKIPPED", file=sys.stderr)
        pd.DataFrame([{"status": "skipped", "reason": str(exc)}]).to_csv(path, index=False)
        return 0
    table.to_csv(path, index=False)
    for check, sub in table.groupby("check", sort=False):
        print(
            f"[parity] {check:24s} rows {len(sub):3d}  worst rel deviation "
            f"{float(sub['rel_deviation'].max()):.3e}  failed {int((~sub['passed']).sum())}",
            file=sys.stderr,
        )
    n_fail = int((~table["passed"].astype(bool)).sum())
    print(
        f"[parity] wrote {path} ({len(table)} rows); tolerance {args.tolerance}; "
        f"{n_fail} row(s) failed",
        file=sys.stderr,
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
