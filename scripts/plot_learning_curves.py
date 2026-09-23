"""Plot learning curves: one run, or mean +- std across seeds (Phase 5). Argparse wrapper.

Usage::

    # one run
    python scripts/plot_learning_curves.py artifacts/runs/ppo_smoke/0 --out smoke.png

    # every seed of a method (each directory needs evals.csv)
    python scripts/plot_learning_curves.py artifacts/runs/ppo/{0,1,2,3,4} --out ppo.png

Panels: tune-pool success, training episode return, p95 touchdown closing speed; see
:mod:`rld.rl.curves`.
"""

import argparse
import sys
from pathlib import Path

from rld.rl.curves import plot_learning_curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", type=Path, nargs="+", help="run directories")
    parser.add_argument("--out", type=Path, required=True, help="output PNG")
    parser.add_argument("--title", default="", help="figure title")
    args = parser.parse_args()
    missing = [d for d in args.run_dirs if not (d / "evals.csv").exists()]
    if missing:
        parser.error(f"no evals.csv in {missing}")
    out = plot_learning_curves(args.run_dirs, args.out, args.title)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
