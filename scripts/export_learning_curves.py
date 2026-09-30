r"""Plot and tabulate the learning curves of one method's seeds (Phase 5 Gate 5).

Argparse wrapper around :mod:`rld.rl.curves`. Writes the figure through
:func:`~rld.rl.curves.plot_learning_curves` and, beside it, the numbers the figure shows as a
CSV, so the committed figure can be re-derived from committed data:

* ``panel = success``: tune-pool success (SS3-SS5 mean, deterministic policy) per evaluation
  step, mean and sample SD (``ddof = 1``) across seeds, via :func:`~rld.rl.curves.aggregate`;
* ``panel = p95_closing``: pooled p95 touchdown closing speed of the evaluation, same grid;
* ``panel = train_return``: training episode return, binned into 40 equal step bins.

Units: steps are env control steps (1/30 s model scale each); closing speed metres per
second model scale; success and return dimensionless. Simulation only.

Usage::

    python scripts/export_learning_curves.py artifacts/runs/ppo/{0,1,2,3,4} \
        --out results/e05/learning_curves_ppo
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rld.rl.curves import aggregate, binned_return, plot_learning_curves, read_evals, read_monitor


def curve_table(run_dirs: list[Path]) -> pd.DataFrame:
    """Return the three panels' mean and SD across seeds as one long table.

    Args:
        run_dirs: One run directory per seed of one method.

    Returns:
        Columns ``panel, steps, mean, sd, n_seeds``.
    """
    evals = [read_evals(d) for d in run_dirs]
    parts = []
    for panel, col in (("success", "success"), ("p95_closing", "pooled_td_closing_p95_m_s")):
        curves = [(e["steps"].to_numpy(float), e[col].to_numpy(float)) for e in evals]
        grid, mean, sd = aggregate(curves)
        parts.append(pd.DataFrame({"panel": panel, "steps": grid, "mean": mean, "sd": sd}))
    monitors = [read_monitor(d) for d in run_dirs]
    top = max(float(m["steps"].max()) if not m.empty else 1.0 for m in monitors)
    edges = np.linspace(0.0, top, 41)
    stack = np.vstack([binned_return(m, edges) for m in monitors])
    with warnings.catch_warnings():  # empty bins stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1) if len(run_dirs) > 1 else np.zeros_like(mean)
    centres = 0.5 * (edges[:-1] + edges[1:])
    parts.append(pd.DataFrame({"panel": "train_return", "steps": centres, "mean": mean, "sd": sd}))
    table = pd.concat(parts, ignore_index=True)
    table["n_seeds"] = len(run_dirs)
    return table


def main() -> None:
    """Write ``<out>.png`` and ``<out>.csv``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output stem (no extension)")
    args = ap.parse_args()
    runs = [Path(d) for d in args.run_dirs]
    plot_learning_curves(runs, args.out.with_suffix(".png"))
    curve_table(runs).to_csv(args.out.with_suffix(".csv"), index=False, float_format="%.6g")
    print(args.out.with_suffix(".png"), args.out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
