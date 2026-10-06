"""Render the README's landing GIFs from configs/viz/gifs.yaml (Phase 9, P9-D1). Argparse wrapper.

Every panel is re-flown from its trained checkpoint (gitignored ``artifacts/runs``) and checked,
column by column, against its committed episode row before anything is written; see
:mod:`rld.viz.gifs`. Writes ``<out-dir>/<slug>.{gif,png}`` and ``<out-dir>/manifest.csv``.

Usage::

    python scripts/make_gifs.py                      # every case
    python scripts/make_gifs.py --only id_ss6_10_rl_hard
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from rld.eval.envs import load_eval_configs
from rld.viz.gifs import load_gallery, render_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/viz/gifs.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures/gifs"))
    parser.add_argument("--only", nargs="*", default=None, help="case slugs to render")
    args = parser.parse_args()
    gallery = load_gallery(args.config)
    cases = [c for c in gallery.cases if not args.only or c.slug in args.only]
    if args.only and len(cases) != len(set(args.only)):
        parser.error(f"unknown slug in {args.only}")
    cfgs = load_eval_configs()
    manifest = args.out_dir / "manifest.csv"
    old = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()
    rows = []
    for case in cases:
        rows += render_case(
            case,
            gallery,
            cfgs,
            results_dir=args.results_dir,
            runs_dir=args.runs_dir,
            out_dir=args.out_dir,
        )
        print(f"{case.slug}: written", flush=True)
    new = pd.DataFrame(rows)
    if not old.empty:
        old = old[~old["slug"].isin(new["slug"])]
        new = pd.concat([old, new], ignore_index=True)
    order = {c.slug: i for i, c in enumerate(gallery.cases)}
    new = new.sort_values("slug", key=lambda s: s.map(order), kind="stable")
    new.to_csv(manifest, index=False, lineterminator="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
