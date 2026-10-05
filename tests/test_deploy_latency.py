"""Latency harness (P8-D1 §5, §8, §10): a refused provider is never timed; the grid; H5 scoring.

Units: latencies milliseconds.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from dmf.deploy.bench import BenchConfig, BenchResult
from dmf.deploy.harness import BENCH_N_WINDOWS, BENCH_WINDOW_SEED
from dmf.deploy.parity import ParityResult

from _deploy_helpers import tiny_policy
from rld.deploy.export import export_folded, folded_actor, normalizer_constants
from rld.deploy.latency import (
    LatencyRow,
    PolicyBenchJob,
    bench_inputs,
    benchmark_ort,
    benchmark_torch,
    latency_csv,
    latency_jobs,
    make_batches,
    run_latency,
    score_h5,
)
from rld.deploy.parity import ParityRow

ARCH = "mlp512x2-tanh-in25"


def _job(backend: str, batch: int = 1, device: str = "cpu", threads: int = 1) -> PolicyBenchJob:
    return PolicyBenchJob(
        "ppo",
        4,
        ARCH,
        Path("g.onnx"),
        Path("run"),
        backend,
        batch,
        device,
        intra_op_threads=threads,  # type: ignore[arg-type]
    )


def _parity(provider: str, threads: int, passed: bool) -> ParityRow:
    res = ParityResult(1e-7, 1e-8, 1000, 1e-4, passed, 1.0, 1.0, 1e-4, passed, 0, provider)
    return ParityRow("ppo", 4, ARCH, "sha", provider, provider, threads, 0, 200, 0.1, res)


def _result(job: PolicyBenchJob, p50: float, p99: float | None = None) -> BenchResult:
    realized = (
        (f"torch-eager:{job.resolved_device}",)
        if job.backend == "torch-eager"
        else (job.provider, "CPUExecutionProvider")
    )
    return BenchResult(
        label=job.label,
        p50_ms=p50,
        p90_ms=p50,
        p99_ms=p99 if p99 is not None else 2 * p50,
        mean_ms=p50,
        std_ms=0.0,
        throughput_windows_s=1e3 / p50,
        peak_host_mem_mb=1.0,
        peak_device_mem_mb=0.0,
        providers_realized=realized,
        intra_op_threads=job.intra_op_threads,
        tf32=False,
        config=job.bench_config(),
        env={},
    )


def test_grid_is_p8d1s() -> None:
    jobs = latency_jobs(
        [("ppo", 4, ARCH, Path("g"), Path("r")), ("rpf", 4, "B", Path("h"), Path("s"))]
    )
    assert len(jobs) == 44 and len({j.label for j in jobs}) == 44
    one = [j for j in jobs if j.architecture == ARCH]
    assert {(j.backend, j.resolved_device) for j in one if j.intra_op_threads == 1} == {
        ("torch-eager", "cpu"),
        ("torch-eager", "cuda"),
        ("ort-cpu", "cpu"),
        ("ort-cuda", "cuda"),
        ("ort-trt", "cuda"),
    }
    sweep = {(j.backend, j.intra_op_threads) for j in one if j.intra_op_threads > 1}
    assert sweep == {(b, t) for b in ("ort-cpu", "torch-eager") for t in (2, 4, 8)}
    assert {j.batch_size for j in one} == {1, 32}


def test_job_config_is_project4s_method() -> None:
    cfg = _job("ort-cuda").bench_config()
    assert (cfg.warmup_iters, cfg.timed_iters, cfg.synchronize) == (200, 2000, True)
    assert cfg.device == "cuda" and _job("ort-cpu").bench_config().device == "cpu"
    assert BenchConfig() == BenchConfig(200, 2000, 1, "cpu", True, 1)
    assert (_job("ort-cpu").n_inputs, _job("ort-cpu").input_seed) == (
        BENCH_N_WINDOWS,
        BENCH_WINDOW_SEED,
    )
    assert _job("torch-eager", device="cuda").label == f"{ARCH}-torch-eager-cuda-b1-t1"
    assert _job("ort-trt").label == f"{ARCH}-ort-trt-b1-t1"


def test_a_refused_provider_is_never_timed() -> None:
    jobs = latency_jobs([("ppo", 4, ARCH, Path("g"), Path("r"))])
    parity = (
        [_parity("CPUExecutionProvider", t, True) for t in (1, 2, 4, 8)]
        + [_parity("torch:cpu", t, True) for t in (1, 2, 4, 8)]
        + [
            _parity("CUDAExecutionProvider", 1, True),
            _parity("TensorrtExecutionProvider", 1, False),  # fails -> refused
            # torch:cuda has no parity row at all -> refused
        ]
    )
    launched: list[str] = []

    def runner(job: PolicyBenchJob) -> BenchResult:
        launched.append(job.label)
        return _result(job, 0.1)

    rows = run_latency(jobs, parity, runner=runner)
    refused = {r.job.label for r in rows if r.status == "refused"}
    assert refused == {
        j.label
        for j in jobs
        if j.backend == "ort-trt" or (j.backend == "torch-eager" and j.resolved_device == "cuda")
    }
    assert not refused & set(launched)
    assert all(r.result is None for r in rows if r.status == "refused")
    text = latency_csv(rows)
    trt_line = next(line for line in text.splitlines() if "-ort-trt-b1-" in line)
    assert ",refused," in trt_line and ",0." not in trt_line


def test_failed_child_is_not_a_number() -> None:
    def runner(job: PolicyBenchJob) -> BenchResult:
        raise RuntimeError("child exited 1\nValueError: requested X but the session realized Y")

    rows = run_latency([_job("ort-cpu")], [_parity("CPUExecutionProvider", 1, True)], runner=runner)
    assert rows[0].status.startswith("failed:") and rows[0].result is None


def _h5_rows(cpu: float, cuda: float | None, trt: float | None) -> list[LatencyRow]:
    rows = [LatencyRow(_job("ort-cpu"), "timed", _result(_job("ort-cpu"), cpu))]
    for backend, value in (("ort-cuda", cuda), ("ort-trt", trt)):
        job = _job(backend)
        rows.append(
            LatencyRow(job, "refused", None)
            if value is None
            else LatencyRow(job, "timed", _result(job, value))
        )
    tc = _job("torch-eager", device="cuda")
    rows.append(LatencyRow(tc, "timed", _result(tc, 0.01)))  # context only, never scored
    return rows


def test_h5_scoring() -> None:
    verdict, table = score_h5(_h5_rows(0.05, 0.40, 0.15), ARCH)
    assert verdict == "supported"
    assert [r[13] for r in table] == ["True", "True", "False"]  # torch-eager CUDA not scored
    assert float(table[0][11]) == pytest.approx(8.0)
    assert score_h5(_h5_rows(0.05, 0.40, 0.09), ARCH)[0] == "not supported"  # 1.8x < 2
    assert score_h5(_h5_rows(0.05, 0.40, None), ARCH)[0] == "supported"  # refused TRT not scored
    assert score_h5(_h5_rows(0.05, None, None), ARCH)[0].startswith("not scored")


def test_rank2_batches_mirror_dmf() -> None:
    x = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    b = make_batches(x, 4)
    assert len(b) == 2 and all(a.shape == (4, 3) for a in b)
    assert np.array_equal(make_batches(x, 32)[0][10], x[0])  # cycles
    with pytest.raises(ValueError):
        make_batches(x[None], 1)


def test_benchmark_functions_run(tmp_path: Path) -> None:
    policy = tiny_policy()
    constants = normalizer_constants(policy)
    path = export_folded(folded_actor(policy), constants.dim, tmp_path / "g.onnx")
    inputs = bench_inputs(constants)
    assert inputs.shape == (64, 25)
    cfg = BenchConfig(warmup_iters=3, timed_iters=10, batch_size=4, intra_op_threads=1)
    res = benchmark_ort(path, inputs, cfg, "x")
    assert res.providers_realized[0] == "CPUExecutionProvider" and res.p50_ms > 0
    assert res.tf32 is False and res.intra_op_threads == 1
    tres = benchmark_torch(folded_actor(policy), inputs, cfg, "y")
    assert tres.providers_realized == ("torch-eager:cpu",)
    with pytest.raises(ValueError, match="un-synchronised"):
        benchmark_torch(
            folded_actor(policy), inputs, BenchConfig(device="cuda", synchronize=False), "z"
        )
    json.dumps(res.env)
