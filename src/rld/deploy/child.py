"""Child entry points of :mod:`rld.deploy.pipeline`, one fresh process each (dmf's pattern).

* ``python -m rld.deploy.child bench --job JOB.json --out RESULT.json`` -- one latency
  configuration (:func:`rld.deploy.latency.run_job`), written with dmf's
  :func:`~dmf.deploy.bench.write_benchmark_json`;
* ``python -m rld.deploy.child parity --specs SPECS.json --out PARITY.csv`` -- every graph ×
  provider (:func:`rld.deploy.pipeline.parity_rows_for`);
* ``python -m rld.deploy.child e2e --specs SPECS.json --out E2E.json`` -- the per-step budget of
  each timed graph (:func:`rld.deploy.e2e.run_e2e`).

Each is started by the parent under :func:`dmf.deploy.providers.tf32_environment` (the e2e child
also with ``OMP_NUM_THREADS = MKL_NUM_THREADS = 1``). A module rather than ``python -c``, so a
failure names a file and a line.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the parser (not a user-facing CLI; every path is required)."""
    parser = argparse.ArgumentParser(description="One rld.deploy child task.")
    sub = parser.add_subparsers(dest="task", required=True)
    bench = sub.add_parser("bench")
    bench.add_argument("--job", type=Path, required=True)
    bench.add_argument("--out", type=Path, required=True)
    for name in ("parity", "e2e"):
        p = sub.add_parser(name)
        p.add_argument("--specs", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one task and write its output.

    Args:
        argv: Arguments (``None``: ``sys.argv``).

    Returns:
        0 on success; any refusal propagates as an exception and a non-zero status.
    """
    args = build_parser().parse_args(argv)
    if args.task == "bench":
        from dmf.deploy.bench import write_benchmark_json

        from rld.deploy.latency import job_from_payload, run_job

        job = job_from_payload(json.loads(args.job.read_text(encoding="utf-8")))
        write_benchmark_json((run_job(job),), args.out)
        return 0
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    if args.task == "parity":
        from rld.deploy.parity import parity_csv
        from rld.deploy.pipeline import parity_rows_for

        args.out.write_text(parity_csv(parity_rows_for(specs)), encoding="utf-8")
        return 0
    from rld.deploy.closed_loop import closed_loop_episodes
    from rld.deploy.e2e import run_e2e
    from rld.eval.envs import load_eval_configs

    cfgs = load_eval_configs()
    episodes = closed_loop_episodes()
    runs = []
    for spec in specs:
        timings, outcomes = run_e2e(
            cfgs, episodes, run_dir=Path(spec["run_dir"]), onnx_path=Path(spec["onnx_path"])
        )
        runs.append(
            {
                "method": spec["method"],
                "seed": spec["seed"],
                "architecture": spec["architecture"],
                "timings": [t.times_s for t in timings],
                "outcomes": [
                    {"ss": o.ss, "index": o.index, "outcome": o.outcome, "steps": o.steps}
                    for o in outcomes
                ],
            }
        )
        print(f"e2e {spec['method']}_s{spec['seed']}: {len(timings)} steps", flush=True)
    args.out.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
