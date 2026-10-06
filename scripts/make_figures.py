"""Render the README's success-vs-sea-state figures from committed CSVs (Phase 9). Argparse wrapper.

Needs nothing but the committed ``results/e07/matrix/`` CSVs: no checkpoints, no simulation.
See :mod:`rld.viz.curves`. Writes ``success_vs_seastate_id.png`` (the headline) and
``success_vs_seastate_shift.png`` (``unseen_seastate``, ``unseen_heading``, ``unseen_vessel``)
and ``closing_speed_ecdf_id.png`` (touchdown closing-speed distributions, ``id`` SS5 and SS6).
"""

import argparse
import sys
from pathlib import Path

from rld.viz.curves import plot_closing_speed_ecdf, plot_success_vs_seastate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    print(
        plot_success_vs_seastate(
            args.results_dir, args.out_dir / "success_vs_seastate_id.png", regimes=("id",)
        )
    )
    print(
        plot_success_vs_seastate(
            args.results_dir,
            args.out_dir / "success_vs_seastate_shift.png",
            regimes=("unseen_seastate", "unseen_heading", "unseen_vessel"),
            ylim=(35.0, 101.0),
            title=(
                "Landing success by sea state under shift. Simulation: Froude-scaled JONSWAP "
                "deck motion (λ = 1/25), aft pad"
            ),
        )
    )
    print(plot_closing_speed_ecdf(args.results_dir, args.out_dir / "closing_speed_ecdf_id.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
