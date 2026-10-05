"""``make bench``: selection, export, parity, closed loop, latency, e2e budget, H5 (P8-D1).

Stages, in the order the methodology requires (parity before timing; nothing else running on the
machine while anything is timed):

1. ``selection`` -- recompute the median seeds (P8-D1 §2) into ``selection.csv`` and refuse if
   they are not the exported seeds;
2. ``export`` -- the four graphs (P8-D1 §3) into ``artifacts/onnx/``;
3. ``parity`` -- numeric parity in a child process under ``tf32_environment()`` -> ``parity.csv``;
4. ``closed-loop`` -- PyTorch vs ONNX on 50 episodes -> ``closed_loop_parity.csv`` and
   ``closed_loop_episodes.csv``, and the post-hoc ``closed_loop_controls.csv`` (P8-D2);
5. ``latency`` -- the grid, one child per configuration, refused rows never launched ->
   ``latency.json``, ``latency.csv``, and ``h5.csv``;
6. ``e2e`` -- the per-step budget in a child process -> ``e2e_budget.csv``.

``run_info.json`` collects wall times, graph SHA-256s, the environment stamp, refused and failed
configurations and the e2e-vs-closed-loop outcome check; it is not byte-checked.

``--check`` re-exports into a temporary directory, re-runs selection, parity and the closed loop
against those graphs, and compares (P8-D1 §11): bytes for ``selection.csv``,
``closed_loop_parity.csv``, ``closed_loop_episodes.csv``, ``closed_loop_controls.csv`` and the
CPU rows of ``parity.csv``; for the GPU rows the verdicts, n, ``|y|max`` and threshold byte for
byte and the error below the threshold; the graph SHA-256s against ``run_info.json``. Latency,
e2e and H5 are measurements and are not re-checked. ``--check`` writes nothing under ``results/``.

Units: latencies milliseconds; times seconds model scale.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dmf.deploy.bench import environment_stamp, write_benchmark_json
from dmf.deploy.harness import CHILD_TIMEOUT_S
from dmf.deploy.parity import PARITY_SEEDS
from dmf.deploy.providers import tf32_environment

from rld.config import REPO_ROOT
from rld.deploy.export import DEFAULT_ONNX_DIR, EXPORTS, ExportedGraph, export_run, sha256_file
from rld.deploy.parity import (
    CPU_THREADS,
    GPU_PARITY_PROVIDERS,
    ORT_PROVIDERS,
    PARITY_COLUMNS,
    ParityRow,
    deterministic_provider,
    read_parity_csv,
)
from rld.deploy.selection import SELECTION_SOURCES, rank_seeds, read_seed_row, selection_csv

__all__ = [
    "DEFAULT_OUT_DIR",
    "RUN_ROOT",
    "STAGES",
    "TIMED_SEED",
    "compare_parity",
    "main",
    "parity_rows_for",
]

#: Where the committed outputs go.
DEFAULT_OUT_DIR: Path = REPO_ROOT / "results" / "latency"

#: The run tree.
RUN_ROOT: Path = REPO_ROOT / "artifacts" / "runs"

#: Stages, in order.
STAGES: tuple[str, ...] = ("selection", "export", "parity", "closed-loop", "latency", "e2e")

#: The timed seed of each architecture: the median seed (P8-D1 §2, §3).
TIMED_SEED: dict[str, int] = {"ppo": 4, "residual_ppo_forecast": 4}

#: The method whose graph scores H5 (P8-D1 §3).
H5_METHOD: str = "ppo"


def _log(message: str) -> None:
    """Print a progress line, flushed."""
    print(f"[bench {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _graph_specs(onnx_dir: Path) -> list[dict[str, Any]]:
    """Describe the four graphs (method, seed, run dir, graph path, architecture)."""
    specs = []
    for method, seed in EXPORTS:
        path = onnx_dir / f"{method}_s{seed}.onnx"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        specs.append(
            {
                "method": method,
                "seed": seed,
                "run_dir": str((RUN_ROOT / method / str(seed)).resolve()),
                "onnx_path": str(path),
                "architecture": str(meta["architecture"]),
                "onnx_sha256": sha256_file(path),
            }
        )
    return specs


# --------------------------------------------------------------------------- selection


def stage_selection() -> str:
    """Recompute the median seeds and return ``selection.csv``.

    Raises:
        RuntimeError: If a recomputed median seed is not :data:`TIMED_SEED`'s.
    """
    rankings = [rank_seeds(m, read_seed_row(SELECTION_SOURCES[m], m)) for m in TIMED_SEED]
    for r in rankings:
        if r.median_seed != TIMED_SEED[r.method]:
            raise RuntimeError(f"{r.method}: median seed {r.median_seed} != {TIMED_SEED[r.method]}")
        _log(f"selection {r.method}: worst->best {r.order}, median seed {r.median_seed}")
    return selection_csv(rankings)


# --------------------------------------------------------------------------- export


def stage_export(onnx_dir: Path) -> list[ExportedGraph]:
    """Export the four graphs."""
    graphs = [export_run(m, s, out_dir=onnx_dir) for m, s in EXPORTS]
    for g in graphs:
        _log(f"exported {g.onnx_path.name}: D={g.obs_dim}, sha256 {g.onnx_sha256[:12]}")
    return graphs


# --------------------------------------------------------------------------- parity


def _row_fields(
    spec: dict[str, Any], provider: str, realized: str, threads: int, draw: tuple[Any, ...]
) -> dict[str, Any]:
    """The identity fields of one parity row (``draw`` = (seed, x, n_tail, clipped, ref))."""
    return {
        "method": str(spec["method"]),
        "seed": int(spec["seed"]),
        "architecture": str(spec["architecture"]),
        "onnx_sha256": str(spec["onnx_sha256"]),
        "provider": provider,
        "provider_realized": realized,
        "threads": threads,
        "draw_seed": int(draw[0]),
        "n_tail": int(draw[2]),
        "clipped_fraction": float(draw[3]),
    }


def parity_rows_for(specs: Sequence[dict[str, Any]]) -> list[ParityRow]:
    """Run numeric parity for every graph and provider (the parity child's body).

    Args:
        specs: :func:`_graph_specs` entries.

    Returns:
        Rows in (graph, provider, threads, draw) order.
    """
    import torch
    from dmf.deploy.providers import disable_tf32

    from rld.deploy.export import folded_actor, normalizer_constants
    from rld.deploy.parity import check_ort, check_torch, open_session, parity_draw
    from rld.deploy.parity import reference_actions as ref_fn
    from rld.eval.envs import load_eval_configs
    from rld.rl.train import build_policy

    disable_tf32()
    cfgs = load_eval_configs()
    rows: list[ParityRow] = []
    for spec in specs:
        method, seed = str(spec["method"]), int(spec["seed"])
        onnx_path = Path(spec["onnx_path"])
        policy = build_policy(cfgs, run_dir=Path(spec["run_dir"]))
        constants = normalizer_constants(policy)
        module = folded_actor(policy)
        draws = []
        for draw_seed in PARITY_SEEDS:
            x, n_tail, clipped = parity_draw(constants, draw_seed)
            torch.set_num_threads(1)
            draws.append((draw_seed, x, n_tail, clipped, ref_fn(policy, x)))

        ort_plan = [("CPUExecutionProvider", t) for t in CPU_THREADS] + [
            (p, 1) for p in ORT_PROVIDERS if p != "CPUExecutionProvider"
        ]
        for provider, threads in ort_plan:
            try:
                session, realized = open_session(onnx_path, provider, threads)
            except ValueError as exc:
                note = str(exc).replace("\n", " ")[:300]
                rows.extend(
                    ParityRow(**_row_fields(spec, provider, "", threads, d), result=None, note=note)
                    for d in draws
                )
                _log(f"parity {onnx_path.name} {provider} t{threads}: NOT REALIZED")
                continue
            for d in draws:
                res = check_ort(session, d[1], d[4], provider, seed=int(d[0]))
                rows.append(
                    ParityRow(**_row_fields(spec, provider, realized, threads, d), result=res)
                )
            del session
        torch_plan = [("cpu", t) for t in CPU_THREADS] + [("cuda", 1)]
        for device, threads in torch_plan:
            provider = f"torch:{device}"
            for d in draws:
                try:
                    res = check_torch(module, d[1], d[4], device, seed=int(d[0]), threads=threads)
                except ValueError as exc:
                    rows.append(
                        ParityRow(
                            **_row_fields(spec, provider, "", threads, d),
                            result=None,
                            note=str(exc),
                        )
                    )
                    continue
                rows.append(
                    ParityRow(**_row_fields(spec, provider, provider, threads, d), result=res)
                )
        worst = max(
            (
                r.result.max_abs_err
                for r in rows
                if r.result and r.seed == seed and r.method == method
            ),
            default=float("nan"),
        )
        _log(f"parity {onnx_path.name}: worst max_abs_err over providers {worst:.3e}")
    torch.set_num_threads(1)
    return rows


def _run_child(args: list[str], *, env: dict[str, str], timeout_s: float) -> None:
    """Run ``python -m rld.deploy.child <args>`` and raise with its stderr on failure."""
    completed = subprocess.run(
        [sys.executable, "-m", "rld.deploy.child", *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"child {args[0]} exited {completed.returncode}\n{completed.stderr.strip()[-4000:]}"
        )
    for line in completed.stdout.splitlines():
        print(line, flush=True)


def stage_parity(specs: Sequence[dict[str, Any]]) -> str:
    """Run parity in a fresh child under ``tf32_environment()``; return ``parity.csv``."""
    with tempfile.TemporaryDirectory(prefix="rld-parity-") as tmp:
        spec_path = Path(tmp) / "specs.json"
        out_path = Path(tmp) / "parity.csv"
        spec_path.write_text(json.dumps(list(specs)), encoding="utf-8")
        _run_child(
            ["parity", "--specs", str(spec_path), "--out", str(out_path)],
            env=tf32_environment(),
            timeout_s=CHILD_TIMEOUT_S,
        )
        return out_path.read_text(encoding="utf-8")


def compare_parity(committed: str, fresh: str) -> list[str]:
    """Compare a fresh ``parity.csv`` with the committed one (P8-D1 §11).

    Args:
        committed: Committed CSV text.
        fresh: Re-run CSV text.

    Returns:
        Human-readable differences (empty when the check passes).
    """
    import csv
    import io

    a = list(csv.DictReader(io.StringIO(committed)))
    b = list(csv.DictReader(io.StringIO(fresh)))
    if len(a) != len(b):
        return [f"parity.csv: {len(a)} committed rows vs {len(b)} re-run rows"]
    errors: list[str] = []
    loose = {"max_abs_err", "mean_abs_err", "provider_realized"}
    for i, (ra, rb) in enumerate(zip(a, b, strict=True)):
        if deterministic_provider(ra["provider"]):
            if ra != rb:
                errors.append(f"parity.csv row {i + 2} ({ra['provider']}): bytes differ")
            continue
        for col in PARITY_COLUMNS:
            if col in loose:
                continue
            if ra[col] != rb[col]:
                errors.append(f"parity.csv row {i + 2} ({ra['provider']}): {col} differs")
        if rb["passed"] == "True" and not float(rb["max_abs_err"]) < float(rb["scaled_tolerance"]):
            errors.append(f"parity.csv row {i + 2}: re-run error not below threshold")
        if ra["provider"] not in GPU_PARITY_PROVIDERS:
            errors.append(f"parity.csv row {i + 2}: unexpected provider {ra['provider']}")
    return errors


# --------------------------------------------------------------------------- closed loop


def stage_closed_loop(specs: Sequence[dict[str, Any]], workers: int) -> tuple[str, str, str]:
    """Fly the closed-loop arms.

    Returns:
        ``(closed_loop_parity.csv, closed_loop_episodes.csv, closed_loop_controls.csv)``.
    """
    from rld.deploy.closed_loop import (
        closed_loop_episodes,
        closed_loop_tables,
        controls_table,
        run_closed_loop,
    )
    from rld.eval.envs import load_eval_configs

    # As e05/e06 (rld.eval.learned.main): one torch thread in every spawned runner worker.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    cfgs = load_eval_configs()
    episodes = closed_loop_episodes()
    graphs = [
        (str(s["method"]), int(s["seed"]), Path(s["run_dir"]), Path(s["onnx_path"])) for s in specs
    ]
    results = run_closed_loop(graphs, cfgs, episodes, workers=workers)
    summary, per_episode = closed_loop_tables(results)
    return summary, per_episode, controls_table(results)


# --------------------------------------------------------------------------- latency + H5


def _timed_specs(specs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The median-seed graph of each architecture."""
    return [s for s in specs if TIMED_SEED.get(str(s["method"])) == int(s["seed"])]


def stage_latency(
    specs: Sequence[dict[str, Any]],
    parity_csv_path: Path,
    *,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[list[Any], str, str, str]:
    """Run the grid (refused rows never launched) and score H5.

    Returns:
        ``(latency rows, latency.csv, h5.csv, verdict)``.
    """
    import dataclasses

    from rld.deploy.latency import h5_csv, latency_csv, latency_jobs, run_latency, score_h5

    parity_rows = read_parity_csv(parity_csv_path)
    timed = [
        (
            str(s["method"]),
            int(s["seed"]),
            str(s["architecture"]),
            Path(s["onnx_path"]),
            Path(s["run_dir"]),
        )
        for s in _timed_specs(specs)
    ]
    jobs = [
        dataclasses.replace(j, warmup_iters=warmup_iters, timed_iters=timed_iters)
        for j in latency_jobs(timed)
    ]
    rows = run_latency(jobs, parity_rows, log=_log)
    h5_arch = next(str(s["architecture"]) for s in _timed_specs(specs) if s["method"] == H5_METHOD)
    verdict, table = score_h5(rows, h5_arch)
    _log(f"H5 on {h5_arch}: {verdict}")
    return rows, latency_csv(rows), h5_csv(table), verdict


# --------------------------------------------------------------------------- e2e


def stage_e2e(specs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Run the e2e budget in a fresh single-threaded child; return its JSON payload."""
    env = {**tf32_environment(), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(prefix="rld-e2e-") as tmp:
        spec_path = Path(tmp) / "specs.json"
        out_path = Path(tmp) / "e2e.json"
        spec_path.write_text(json.dumps(_timed_specs(specs)), encoding="utf-8")
        _run_child(
            ["e2e", "--specs", str(spec_path), "--out", str(out_path)],
            env=env,
            timeout_s=CHILD_TIMEOUT_S,
        )
        payload: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    return payload


def e2e_check_outcomes(payload: dict[str, Any], episodes_csv: Path) -> dict[str, Any]:
    """Compare the e2e episodes' outcomes and steps with the closed-loop ONNX rows."""
    import csv

    with episodes_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, Any] = {}
    for run in payload["runs"]:
        ref = {
            (r["ss"], r["index"]): (r["outcome_onnx"], r["steps_onnx"])
            for r in rows
            if r["method"] == run["method"] and r["seed"] == str(run["seed"])
        }
        same = sum(
            1
            for o in run["outcomes"]
            if ref.get((o["ss"], str(o["index"]))) == (o["outcome"], str(o["steps"]))
        )
        out[f"{run['method']}_s{run['seed']}"] = {
            "n_episodes": len(run["outcomes"]),
            "n_outcome_and_steps_equal_closed_loop_onnx": same,
        }
    return out


# --------------------------------------------------------------------------- main


def _update_run_info(out_dir: Path, key: str, value: Any) -> None:
    """Merge one section into ``run_info.json``."""
    path = out_dir / "run_info.json"
    info: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    info[key] = value
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--stage", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--check", action="store_true", help="re-run and compare (writes nothing)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--workers", type=int, default=16, help="closed-loop runner processes")
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--timed-iters", type=int, default=2000)
    args = parser.parse_args(argv)
    default_iters = (args.warmup_iters, args.timed_iters) == (200, 2000)
    if not default_iters and args.out_dir.resolve() == DEFAULT_OUT_DIR.resolve():
        parser.error("non-default iteration counts may not write to results/latency")
    return args


def run_check(args: argparse.Namespace) -> int:
    """``--check`` (P8-D1 §11)."""
    out_dir: Path = args.out_dir
    failures: list[str] = []

    def compare(name: str, fresh: str) -> None:
        committed = (out_dir / name).read_text(encoding="utf-8")
        status = "byte-identical" if committed == fresh else "DIFFERS"
        if committed != fresh:
            failures.append(name)
        _log(f"check {name}: {status}")

    compare("selection.csv", stage_selection())
    info = json.loads((out_dir / "run_info.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="rld-onnx-check-") as tmp:
        graphs = stage_export(Path(tmp))
        recorded = {g["file"]: g["onnx_sha256"] for g in info["graphs"]}
        for g in graphs:
            ok = recorded.get(g.onnx_path.name) == g.onnx_sha256
            _log(f"check {g.onnx_path.name}: sha256 {'equal' if ok else 'DIFFERS'}")
            if not ok:
                failures.append(g.onnx_path.name)
        specs = _graph_specs(Path(tmp))
        errors = compare_parity(
            (out_dir / "parity.csv").read_text(encoding="utf-8"), stage_parity(specs)
        )
        _log(f"check parity.csv: {'OK' if not errors else 'DIFFERS'}")
        for e in errors:
            _log(f"  {e}")
        failures.extend(errors)
        summary, episodes, controls = stage_closed_loop(specs, args.workers)
        compare("closed_loop_parity.csv", summary)
        compare("closed_loop_episodes.csv", episodes)
        compare("closed_loop_controls.csv", controls)
    _log("check: " + ("OK" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested stages (or ``--check``).

    Args:
        argv: Arguments (``None``: ``sys.argv``).

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    if args.check:
        return run_check(args)
    out_dir: Path = args.out_dir
    onnx_dir: Path = args.onnx_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [s for s in STAGES if s in args.stage]
    _update_run_info(out_dir, "environment", environment_stamp())

    def timed_stage(name: str, fn: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        _log(f"stage {name} ...")
        value = fn()
        wall = time.perf_counter() - start
        _update_run_info(out_dir, f"wall_s_{name}", round(wall, 1))
        _log(f"stage {name} done in {wall:.1f} s")
        return value

    if "selection" in stages:
        text = timed_stage("selection", stage_selection)
        (out_dir / "selection.csv").write_text(text, encoding="utf-8")
    if "export" in stages:
        graphs = timed_stage("export", lambda: stage_export(onnx_dir))
        _update_run_info(
            out_dir,
            "graphs",
            [
                {
                    "file": g.onnx_path.name,
                    "method": g.method,
                    "seed": g.seed,
                    "architecture": g.architecture,
                    "obs_dim": g.obs_dim,
                    "onnx_sha256": g.onnx_sha256,
                    "model_sha256": g.model_sha256,
                    "vecnormalize_sha256": g.vecnormalize_sha256,
                }
                for g in graphs
            ],
        )
    specs = _graph_specs(onnx_dir)
    if "parity" in stages:
        text = timed_stage("parity", lambda: stage_parity(specs))
        (out_dir / "parity.csv").write_text(text, encoding="utf-8")
    if "closed-loop" in stages:
        summary, episodes, controls = timed_stage(
            "closed-loop", lambda: stage_closed_loop(specs, args.workers)
        )
        (out_dir / "closed_loop_parity.csv").write_text(summary, encoding="utf-8")
        (out_dir / "closed_loop_episodes.csv").write_text(episodes, encoding="utf-8")
        (out_dir / "closed_loop_controls.csv").write_text(controls, encoding="utf-8")
    if "latency" in stages:
        rows, lat_csv, h5_text, verdict = timed_stage(
            "latency",
            lambda: stage_latency(
                specs,
                out_dir / "parity.csv",
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
            ),
        )
        write_benchmark_json(
            tuple(r.result for r in rows if r.result is not None), out_dir / "latency.json"
        )
        (out_dir / "latency.csv").write_text(lat_csv, encoding="utf-8")
        (out_dir / "h5.csv").write_text(h5_text, encoding="utf-8")
        _update_run_info(
            out_dir,
            "latency",
            {
                "n_jobs": len(rows),
                "n_timed": sum(r.status == "timed" for r in rows),
                "refused": [r.job.label for r in rows if r.status == "refused"],
                "failed": {r.job.label: r.status for r in rows if r.status.startswith("failed")},
                "h5_verdict": verdict,
                "warmup_iters": args.warmup_iters,
                "timed_iters": args.timed_iters,
            },
        )
    if "e2e" in stages:
        from rld.deploy.e2e import EpisodeOutcome, StepTiming, e2e_csv

        payload = timed_stage("e2e", lambda: stage_e2e(specs))
        runs = [
            (
                str(r["method"]),
                int(r["seed"]),
                str(r["architecture"]),
                [StepTiming(t) for t in r["timings"]],
                [EpisodeOutcome(**o) for o in r["outcomes"]],
            )
            for r in payload["runs"]
        ]
        (out_dir / "e2e_budget.csv").write_text(e2e_csv(runs), encoding="utf-8")
        episodes_csv = out_dir / "closed_loop_episodes.csv"
        if episodes_csv.exists():
            _update_run_info(
                out_dir, "e2e_vs_closed_loop", e2e_check_outcomes(payload, episodes_csv)
            )
    _update_run_info(
        out_dir,
        "threads_env",
        {k: os.environ.get(k, "") for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    )
    return 0
