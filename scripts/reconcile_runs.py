"""Mark runs and sweeps whose process died (host restart, kill -9) as failed / interrupted.

Argparse wrapper around :func:`rld.rl.reconcile.reconcile`; the rules are in that module.
In short: a run whose ``status.json`` says ``running``, whose writer process is gone and
whose status is older than 3 evaluation intervals becomes ``state: failed`` with
``error: "host terminated; pid dead since <status mtime>"`` and ``reconciled_at``; every
other field is untouched. A sweep file whose scheduler is dead becomes ``interrupted``, and
its running jobs follow their runs (``failed`` / ``done``). Nothing is deleted, moved or
re-run.

Usage::

    python scripts/reconcile_runs.py --dry-run     # print what would change
    python scripts/reconcile_runs.py               # apply it
"""

import argparse
import sys
from pathlib import Path

from rld.rl.config import RUNS_ROOT
from rld.rl.reconcile import DEFAULT_REASON, STALE_INTERVALS, format_changes, reconcile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT, help="root of all runs")
    parser.add_argument("--dry-run", action="store_true", help="print the changes only")
    parser.add_argument("--reason", default=DEFAULT_REASON, help="cause written into error")
    parser.add_argument(
        "--stale-intervals",
        type=int,
        default=STALE_INTERVALS,
        help="evaluation (runs) / poll (sweeps) intervals of silence required",
    )
    args = parser.parse_args()
    changes = reconcile(
        args.runs_root,
        dry_run=args.dry_run,
        reason=args.reason,
        stale_intervals=args.stale_intervals,
    )
    print(
        ("DRY RUN -- nothing written\n" if args.dry_run else "")
        + format_changes(changes, args.runs_root)
    )
    n_runs = len({c.item for c in changes if c.kind == "run"})
    n_sweeps = len({c.item for c in changes if c.kind == "sweep"})
    verb = "would change" if args.dry_run else "changed"
    print(f"{verb}: {n_runs} run(s), {n_sweeps} sweep(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
