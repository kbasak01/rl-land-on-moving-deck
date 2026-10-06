"""Collect per-stage wall clock into results/runtime_stages.csv (Phase 9). Argparse wrapper.

See :mod:`rld.runtimes`. ``--measure deck-stats report`` also times the two stages that never
recorded a wall clock, writing their outputs to a temporary directory. Rows measured earlier
are kept when ``--measure`` is not given again.
"""

import argparse
import csv
import sys
from pathlib import Path

from rld.runtimes import RuntimeRow, collect, measure, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results/runtime_stages.csv"))
    parser.add_argument("--measure", nargs="*", default=[], choices=["deck-stats", "report"])
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    repo = Path.cwd()
    measured = measure(repo, args.measure, args.workers) if args.measure else []
    if args.out.exists():
        with args.out.open(encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                if rec["source"].startswith("measured") and rec["stage"] not in args.measure:
                    measured.append(
                        RuntimeRow(
                            rec["stage"],
                            rec["target"],
                            rec["item"],
                            float(rec["wall_s"]),
                            rec["workers"],
                            rec["source"],
                            rec["committed"] == "True",
                        )
                    )
    rows = collect(repo)
    order = {"deck-stats": 0, "report": 1}
    measured.sort(key=lambda r: order.get(r.stage, 9))
    deck = [r for r in measured if r.stage == "deck-stats"]
    report = [r for r in measured if r.stage == "report"]
    write_csv(args.out, deck + rows + report)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
