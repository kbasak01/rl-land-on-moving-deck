"""Run (or ``--check``) the P8-D3 closed-loop parity investigation.

Writes ``results/latency/closed_loop_investigation/``:

* ``outcomes.csv`` -- one row per arm × policy × episode (U for the base arms, ``id`` SS6 for the
  20 ``ulp_k`` arms);
* ``flips.csv`` -- flip sets of every base arm against ``torch`` and ``torch_fp64``, per set;
* ``noise_distribution.csv`` -- arm C;
* ``same_input.csv`` -- A1 and B;
* ``trace_summary.csv`` and ``traces_T.csv.gz`` -- A2;
* ``noopt_parity.csv`` -- P8-D1 §6 parity of the ``ORT_DISABLE_ALL`` session;
* ``verdict.csv`` -- P8-D3's readings applied mechanically;
* ``posthoc_same_input.csv``, ``posthoc_bias.csv``, ``posthoc_loo.csv`` -- added at the Gate 8
  review (P8-D6), after every P8-D3 result had been read; descriptive only;
* ``run_info.json`` -- validity checks, environment, wall times (not byte-checked).

``--check`` recomputes everything and compares bytes, excluding the ``onnx_cuda`` rows of
``outcomes.csv`` and ``flips.csv`` (P8-D3: CUDA kernel selection is not guaranteed
bit-reproducible) and ``run_info.json``. It writes nothing.

Units: as :mod:`rld.deploy.investigate`.
"""

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rld.config import REPO_ROOT
from rld.deploy import investigate as inv
from rld.deploy.closed_loop import COMMITTED_EPISODES, closed_loop_episodes, committed_lines
from rld.deploy.export import DEFAULT_ONNX_DIR, normalizer_constants, sha256_file
from rld.eval.episodes import EPISODES_DIR, read_list
from rld.eval.runner import EvalArm, episode_rows_csv, run_arms

__all__ = ["DEFAULT_OUT_DIR", "main", "run"]

#: Output directory (P8-D3).
DEFAULT_OUT_DIR: Path = REPO_ROOT / "results" / "latency" / "closed_loop_investigation"

#: SS6 indices of set T.
T_INDICES: tuple[int, ...] = tuple(range(12))

CPU_ARMS: tuple[str, ...] = tuple(a for a in inv.BASE_ARMS if a != "onnx_cuda")


def _log(message: str) -> None:
    print(f"[investigate {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _graphs() -> list[tuple[str, int, Path, Path]]:
    """The four graphs, checked against ``results/latency/run_info.json``."""
    info = json.loads((REPO_ROOT / "results" / "latency" / "run_info.json").read_text())
    recorded = {g["file"]: g["onnx_sha256"] for g in info["graphs"]}
    out = []
    for g in info["graphs"]:
        path = DEFAULT_ONNX_DIR / g["file"]
        if sha256_file(path) != recorded[g["file"]]:
            raise RuntimeError(f"{path} is not the committed graph")
        run_dir = (REPO_ROOT / "artifacts" / "runs" / g["method"] / str(g["seed"])).resolve()
        out.append((str(g["method"]), int(g["seed"]), run_dir, path))
    return out


def run(workers: int, cuda_workers: int) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Run every arm and analysis; return ``({file: bytes}, run_info)``."""
    from dmf.deploy.bench import environment_stamp
    from dmf.deploy.parity import PARITY_SEEDS

    from rld.deploy.parity import (
        PARITY_TOLERANCE,
        check_ort,
        parity_draw,
        reference_actions,
    )
    from rld.eval.envs import load_eval_configs
    from rld.rl.train import build_policy

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    info: dict[str, Any] = {"environment": environment_stamp()}
    cfgs = load_eval_configs()
    if cfgs.noise.enabled:
        raise RuntimeError("P8-D3 flies noise off")
    graphs = _graphs()
    e50 = closed_loop_episodes()
    e50_keys = {(e.ss, e.index) for e in e50}
    id_rows = read_list(EPISODES_DIR / "id.parquet")
    ss6 = [r for r in id_rows if r.ss == "SS6"]
    union = [r for r in id_rows if (r.ss, r.index) in e50_keys or r.ss == "SS6"]
    if len(ss6) != 200 or len(union) != 238:
        raise RuntimeError(f"episode sets: SS6 {len(ss6)}, U {len(union)}")
    sets = {
        "E50": [(e.ss, e.index) for e in e50],
        "SS6-200": [(e.ss, e.index) for e in ss6],
        "U": [(e.ss, e.index) for e in union],
    }

    # ---- flights (C, D, E)
    start = time.perf_counter()
    arms: list[EvalArm] = []
    labels: list[tuple[str, int, str, int]] = []
    for arm in CPU_ARMS:
        for method, seed, spec in inv.make_specs(graphs, arm):
            arms.append(EvalArm(spec, None, union))
            labels.append((method, seed, arm, len(union)))
    for arm in inv.ulp_arms():
        for method, seed, spec in inv.make_specs(graphs, arm):
            arms.append(EvalArm(spec, None, ss6))
            labels.append((method, seed, arm, len(ss6)))
    _log(f"flying {len(arms)} CPU arm-policies ({sum(n for *_, n in labels)} episodes)")
    rows = run_arms(arms, cfgs, workers=workers, chunk=25)
    _log("flying onnx_cuda")
    cuda_arms = [EvalArm(spec, None, union) for _, _, spec in inv.make_specs(graphs, "onnx_cuda")]
    cuda_rows = run_arms(cuda_arms, cfgs, workers=cuda_workers, chunk=25)
    for method, seed, _ in inv.make_specs(graphs, "onnx_cuda"):
        labels.append((method, seed, "onnx_cuda", len(union)))
    rows = rows + cuda_rows
    info["wall_s_flights"] = round(time.perf_counter() - start, 1)

    outcomes: inv.Outcomes = {}
    out_rows: list[list[str]] = []
    k = 0
    torch_match: dict[str, int] = {}
    for method, seed, arm, n in labels:
        chunk_rows = rows[k : k + n]
        k += n
        outcomes[(method, seed, arm)] = {
            (str(r["ss"]), int(r["index"])): str(r["outcome"]) for r in chunk_rows
        }
        out_rows.extend(inv.outcome_row(method, seed, arm, r) for r in chunk_rows)
        if arm == "torch":
            ref = committed_lines(COMMITTED_EPISODES[method], method, seed)
            lines = episode_rows_csv(chunk_rows).rstrip("\n").split("\n")[1:]
            torch_match[f"{method}_s{seed}"] = sum(
                1
                for r, line in zip(chunk_rows, lines, strict=True)
                if ref.get((str(r["regime"]), str(r["ss"]), str(r["index"]))) == line
            )
    info["torch_rows_byte_identical_to_committed_of_238"] = torch_match

    # ---- traces (A2, B) and same-input (A1)
    start = time.perf_counter()
    traces = inv.run_traces(graphs, ("torch", "onnx", "torch_folded"), e50, cfgs, workers=workers)
    mismatches = [
        f"{t.method}_s{t.seed}/{t.arm}/{t.ss}/{t.index}"
        for t in traces
        if outcomes[(t.method, t.seed, t.arm)][(t.ss, t.index)] != t.outcome
    ]
    info["trace_outcome_mismatches_vs_runner"] = mismatches
    if mismatches:
        raise RuntimeError(f"traced outcomes differ from the runner: {mismatches[:5]}")
    same = inv.same_input_rows(graphs, traces, cfgs)
    sb3 = [r for r in same if r[2] == "sb3"]
    info["sb3_reproduces_recorded_torch_actions"] = all(r[3] == r[4] for r in sb3)
    consts = {(m, s): normalizer_constants(build_policy(cfgs, run_dir=r)) for m, s, r, _ in graphs}
    summary, step_rows = inv.trace_summary_rows(traces, T_INDICES, cfgs, consts)
    info["wall_s_traces"] = round(time.perf_counter() - start, 1)

    # ---- C readings
    sensitive = inv.sensitive_sets(outcomes, sets["SS6-200"])
    flips = inv.flip_rows(outcomes, sets, sensitive)
    noise = inv.noise_rows(outcomes, sets["SS6-200"], sensitive)
    reading, verdict = inv.verdict_rows(same, summary, noise)
    info["reading"] = reading

    # ---- noopt parity (P8-D1 §6 on the ORT_DISABLE_ALL session)
    parity_rows: list[list[str]] = []
    for method, seed, run_dir, onnx_path in graphs:
        policy = build_policy(cfgs, run_dir=run_dir)
        session = inv._session(onnx_path, "onnx_noopt")
        constants = normalizer_constants(policy)
        for draw in PARITY_SEEDS:
            x, _, _ = parity_draw(constants, draw)
            res = check_ort(session, x, reference_actions(policy, x), "onnx_noopt", seed=draw)
            parity_rows.append(
                [
                    method,
                    str(seed),
                    str(draw),
                    repr(res.max_abs_err),
                    repr(res.scaled_tolerance),
                    str(res.passed),
                ]
            )
    info["noopt_parity_all_pass"] = all(r[5] == "True" for r in parity_rows)
    info["parity_tolerance"] = PARITY_TOLERANCE

    files = {
        "outcomes.csv": inv.write_csv(inv.OUTCOME_COLUMNS, out_rows).encode(),
        "flips.csv": inv.write_csv(inv.FLIP_COLUMNS, flips).encode(),
        "noise_distribution.csv": inv.write_csv(inv.NOISE_COLUMNS, noise).encode(),
        "same_input.csv": inv.write_csv(inv.SAME_INPUT_COLUMNS, same).encode(),
        "trace_summary.csv": inv.write_csv(inv.TRACE_SUMMARY_COLUMNS, summary).encode(),
        "traces_T.csv.gz": inv.trace_rows_gz(step_rows),
        "noopt_parity.csv": inv.write_csv(
            ("method", "seed", "draw_seed", "max_abs_err", "threshold", "passed"), parity_rows
        ).encode(),
        "verdict.csv": inv.write_csv(inv.VERDICT_COLUMNS, verdict).encode(),
    }
    # Post hoc (Gate 8 review, P8-D6): descriptive only, never read by the P8-D3 readings.
    files["posthoc_same_input.csv"] = inv.write_csv(
        inv.POSTHOC_SAME_INPUT_COLUMNS, inv.posthoc_same_input_rows(graphs, traces, cfgs)
    ).encode()
    files["posthoc_bias.csv"] = inv.write_csv(
        inv.POSTHOC_BIAS_COLUMNS, inv.posthoc_bias_rows(outcomes, sets["SS6-200"])
    ).encode()
    files["posthoc_loo.csv"] = inv.write_csv(
        inv.POSTHOC_LOO_COLUMNS, inv.posthoc_loo_rows(outcomes, sets["SS6-200"])
    ).encode()
    info["episodes"] = {name: len(v) for name, v in sets.items()}
    info["k_ulp"] = inv.K_ULP
    info["n_episode_rows"] = len(out_rows)
    return files, info


def _drop_cuda(data: bytes) -> bytes:
    """Remove ``onnx_cuda`` rows (column ``arm``) from a CSV's bytes."""
    lines = data.decode().split("\n")
    header = lines[0].split(",")
    i = header.index("arm")
    kept = [lines[0]] + [ln for ln in lines[1:] if ln and ln.split(",")[i] != "onnx_cuda"]
    return "\n".join(kept).encode()


def main(argv: Sequence[str] | None = None) -> int:
    """Run or check the investigation.

    Args:
        argv: Arguments (``None``: ``sys.argv``).

    Returns:
        Exit status (1 when ``--check`` finds a difference).
    """
    parser = argparse.ArgumentParser(description="P8-D3 closed-loop parity investigation.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cuda-workers", type=int, default=8)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    files, info = run(args.workers, args.cuda_workers)
    info["wall_s_total"] = round(time.perf_counter() - start, 1)
    out: Path = args.out_dir
    if args.check:
        bad = []
        for name, data in files.items():
            committed = (out / name).read_bytes()
            if name in {"outcomes.csv", "flips.csv"}:
                ok = _drop_cuda(committed) == _drop_cuda(data)
            else:
                ok = committed == data
            _log(f"check {name}: {'byte-identical' if ok else 'DIFFERS'}")
            if not ok:
                bad.append(name)
        _log("check: " + ("OK" if not bad else f"FAILED {bad}"))
        return 0 if not bad else 1
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    _log(f"reading: {info['reading']}; wrote {sorted(files)}")
    return 0
