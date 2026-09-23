"""Queue several training runs behind a worker-slot cap (Phase 5). Argparse wrapper.

``make sweep CFG=configs/rl/ppo.yaml SEEDS="0 1 2 3 4" MAX_WORKERS=34`` writes
``artifacts/runs/_sweeps/<id>.json``, starts a detached scheduler (its own session, log
beside the JSON) and returns immediately, printing the sweep file, the scheduler pid and
its log. The scheduler starts each run as slots free up; see :mod:`rld.rl.scheduler`.

``--run-scheduler <sweep.json>`` is the detached process's entry point; not for direct use.
"""

import argparse
import sys
from pathlib import Path

from rld.rl.config import RUNS_ROOT
from rld.rl.scheduler import SWEEPS_DIR, detach_scheduler, new_sweep, run_scheduler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, action="append", help="training config(s)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-workers", type=int, default=34, help="global worker-slot cap")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--poll-s", type=float, default=15.0, help="poll interval, host seconds")
    parser.add_argument("--run-scheduler", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_scheduler is not None:
        run_scheduler(args.run_scheduler, args.runs_root, args.poll_s)
        return 0
    if not args.config:
        parser.error("--config is required")
    jobs = [(str(c), s) for c in args.config for s in args.seeds]
    label = "-".join(c.stem for c in args.config)
    sweeps_dir = args.runs_root / SWEEPS_DIR.name
    path = new_sweep("sweep", jobs, args.max_workers, sweeps_dir=sweeps_dir, label=label)
    pid = detach_scheduler(path, args.runs_root, args.poll_s)
    print(f"sweep {path.stem}: {len(jobs)} jobs, max_workers {args.max_workers}")
    print(f"  sweep file     {path}")
    print(f"  scheduler pid  {pid}")
    print(f"  scheduler log  {path.with_suffix('.log')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
