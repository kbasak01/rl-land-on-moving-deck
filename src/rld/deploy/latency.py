"""Policy latency with Project 4's harness, rank 2, one subprocess per configuration (P8-D1 §5, §8).

Why a mirror and not a call
---------------------------
:func:`dmf.deploy.bench.benchmark_onnxruntime`, :func:`dmf.deploy.bench.benchmark_torch` and their
``_batches`` require ``(n, L, C)`` windows; the policy graph's input is ``(batch, D)``.
``third_party/`` is read-only. So :func:`benchmark_ort` and :func:`benchmark_torch` below are
dmf's two functions **line for line** with the rank check changed to rank 2 -- same 200 warmup
and 2 000 timed iterations, same host-to-host iteration, same pinned and read-back threads, same
``torch.cuda.synchronize()`` after every timed GPU iteration, same refusal of an unrealized
provider, same memory accounting -- and they return dmf's own :class:`~dmf.deploy.bench.BenchResult`
reduced by dmf's own :func:`~dmf.deploy.bench._stats`, stamped by
:func:`~dmf.deploy.bench.environment_stamp` and written by
:func:`~dmf.deploy.bench.write_benchmark_json`. P8-D1 §5 records the loop as method-identical.

Process boundary
----------------
Every configuration runs in a fresh ``python -m rld.deploy.child bench`` process under
:func:`dmf.deploy.providers.tf32_environment`, as :func:`dmf.deploy.harness.run_job_subprocess`
does, for dmf's three reasons (per-configuration ``ru_maxrss``, no CUDA/compile state carried
between rows, TensorRT engine builds isolated). Configurations run one at a time.

Refusal
-------
:func:`run_latency` reads the parity rows and **never launches** a configuration whose
(architecture, provider, threads) is refused (:func:`rld.deploy.parity.refused`); such a row is
written ``refused`` with no number. A child that fails (e.g. an unrealized provider) is written
``failed`` with its message, never as a number.

H5
--
:func:`score_h5` applies P3-D1 §8 as P8-D1 §10 reads it: ratio = p50(GPU provider, batch 1) /
p50(ORT CPU, batch 1, 1 thread) on the ``ppo`` graph, for every parity-passing ORT GPU provider;
supported iff every ratio >= 2. The torch-eager CUDA row is printed as context and not scored.

Units: latencies milliseconds, throughput policy evaluations (observations) per second, memory
mebibytes.
"""

import csv
import io
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from dmf.deploy.bench import (
    BYTES_PER_MIB,
    GPU_PROVIDERS,
    BenchConfig,
    BenchResult,
    _cuda_free_bytes,
    _peak_host_mem_mb,
    _stats,
    environment_stamp,
)
from dmf.deploy.export_onnx import INPUT_NAME
from dmf.deploy.harness import (
    BENCH_N_WINDOWS,
    BENCH_WINDOW_SEED,
    CHILD_TIMEOUT_S,
    DEVICE_BY_BACKEND,
    PROVIDERS_BY_BACKEND,
    bench_result_from_payload,
)
from dmf.deploy.providers import (
    disable_tf32,
    preload_gpu_libraries,
    session_providers,
    tf32_environment,
)
from dmf.typedefs import FloatArray
from torch import Tensor

from rld.deploy.export import NormalizerConstants
from rld.deploy.parity import CPU_THREADS, ParityRow, refused

__all__ = [
    "BACKENDS",
    "BATCH_SIZES",
    "CONTROL_PERIOD_MS",
    "H5_COLUMNS",
    "H5_THRESHOLD",
    "LATENCY_COLUMNS",
    "Backend",
    "LatencyRow",
    "PolicyBenchJob",
    "bench_inputs",
    "benchmark_ort",
    "benchmark_torch",
    "constants_from_sidecar",
    "h5_csv",
    "job_from_payload",
    "job_payload",
    "latency_csv",
    "latency_jobs",
    "make_batches",
    "run_job",
    "run_job_subprocess",
    "run_latency",
    "score_h5",
]

#: The backends timed (P8-D1 §8): torch eager is the reference row, on CPU and on CUDA.
Backend = Literal["torch-eager", "ort-cpu", "ort-cuda", "ort-trt"]

#: Report order.
BACKENDS: tuple[Backend, ...] = ("torch-eager", "ort-cpu", "ort-cuda", "ort-trt")

#: Batch sizes: 1 is the real-time case, 32 the throughput case.
BATCH_SIZES: tuple[int, ...] = (1, 32)

#: The control period, milliseconds model scale (30 Hz).
CONTROL_PERIOD_MS: float = 1000.0 / 30.0

#: H5's factor (P3-D1 §8), dimensionless.
H5_THRESHOLD: float = 2.0


def make_batches(inputs: FloatArray, batch_size: int) -> list[FloatArray]:
    """Slice inputs into fixed-size batches, cycling (dmf ``_batches``, rank 2).

    Args:
        inputs: ``(n, D)`` raw policy inputs.
        batch_size: Rows per batch.

    Returns:
        At least one contiguous float32 batch of shape ``(batch_size, D)``.

    Raises:
        ValueError: If ``inputs`` is not rank 2 or ``batch_size`` is not positive.
    """
    if inputs.ndim != 2:
        raise ValueError(f"inputs must have shape (n, D), got {inputs.shape}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    n = int(inputs.shape[0])
    n_batches = max(1, n // batch_size)
    out: list[FloatArray] = []
    for i in range(n_batches):
        idx = [(i * batch_size + j) % n for j in range(batch_size)]
        out.append(np.ascontiguousarray(inputs[idx], dtype=np.float32))
    return out


def benchmark_torch(
    model: Callable[[Tensor], Tensor] | torch.nn.Module,
    inputs: FloatArray,
    cfg: BenchConfig,
    label: str,
) -> BenchResult:
    """Benchmark the folded torch module, eager (dmf ``benchmark_torch``, rank 2, no compile).

    Args:
        model: The folded actor (moved to ``cfg.device``).
        inputs: ``(n, D)`` raw policy inputs to cycle through.
        cfg: Harness settings.
        label: Configuration name.

    Returns:
        The measured result; ``providers_realized`` = ``("torch-eager:<device>",)``.

    Raises:
        ValueError: On an unknown device, an un-synchronised CUDA request, or no CUDA device.
    """
    if cfg.device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {cfg.device!r}")
    if cfg.device == "cuda":
        if not cfg.synchronize:
            raise ValueError(
                "device='cuda' with synchronize=False would time kernel *launches*, not kernel "
                "completions; un-synchronised CUDA timing is refused"
            )
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but no CUDA device is available")

    disable_tf32()
    torch.set_num_threads(cfg.intra_op_threads)
    device = torch.device(cfg.device)
    module = model if not isinstance(model, torch.nn.Module) else model.eval().to(device)
    batches = make_batches(inputs, cfg.batch_size)
    on_cuda = cfg.device == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    def _one(batch: FloatArray) -> None:
        """Run one host-to-host iteration: NumPy in, NumPy out."""
        tensor = torch.from_numpy(batch).to(device)
        out = module(tensor)
        out.detach().cpu().numpy()

    with torch.no_grad():
        for i in range(cfg.warmup_iters):
            _one(batches[i % len(batches)])
        if on_cuda and cfg.synchronize:
            torch.cuda.synchronize(device)

        samples_s: list[float] = []
        for i in range(cfg.timed_iters):
            batch = batches[i % len(batches)]
            start = time.perf_counter()
            _one(batch)
            if on_cuda and cfg.synchronize:
                torch.cuda.synchronize(device)
            samples_s.append(time.perf_counter() - start)

    p50, p90, p99, mean, std = _stats(samples_s)
    peak_device = float(torch.cuda.max_memory_allocated(device)) / BYTES_PER_MIB if on_cuda else 0.0
    return BenchResult(
        label=label,
        p50_ms=p50,
        p90_ms=p90,
        p99_ms=p99,
        mean_ms=mean,
        std_ms=std,
        throughput_windows_s=cfg.batch_size / (mean / 1e3),
        peak_host_mem_mb=_peak_host_mem_mb(),
        peak_device_mem_mb=peak_device,
        providers_realized=(f"torch-eager:{cfg.device}",),
        intra_op_threads=torch.get_num_threads(),
        tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        config=cfg,
        env=environment_stamp(),
    )


def benchmark_ort(
    onnx_path: Path,
    inputs: FloatArray,
    cfg: BenchConfig,
    label: str,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> BenchResult:
    """Benchmark an ONNX graph under a provider (dmf ``benchmark_onnxruntime``, rank 2).

    Args:
        onnx_path: The exported, parity-checked graph.
        inputs: ``(n, D)`` raw policy inputs to cycle through.
        cfg: Harness settings.
        label: Configuration name.
        providers: Execution providers in priority order.

    Returns:
        The measured result.

    Raises:
        FileNotFoundError: If the graph is missing.
        ValueError: If ``providers`` is empty or the first one is not realized.
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"no exported graph at {onnx_path}")
    if not providers:
        raise ValueError("providers must name at least one execution provider")

    import onnxruntime as ort

    preload_gpu_libraries()
    disable_tf32()
    wants_gpu = bool(GPU_PROVIDERS & set(providers))
    free_before = _cuda_free_bytes() if wants_gpu else 0

    options = ort.SessionOptions()
    options.intra_op_num_threads = cfg.intra_op_threads
    options.inter_op_num_threads = cfg.intra_op_threads
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=session_providers(providers)
    )
    realized = tuple(str(p) for p in session.get_providers())
    if providers[0] not in realized:
        raise ValueError(
            f"requested {providers[0]!r} but the session realized {list(realized)}; ORT falls "
            f"back silently, so reporting this row would publish "
            f"{realized[0] if realized else 'no'} timings under the {providers[0]} label"
        )

    torch.set_num_threads(cfg.intra_op_threads)
    batches = make_batches(inputs, cfg.batch_size)
    on_gpu = bool(GPU_PROVIDERS & set(realized))
    sync = on_gpu and cfg.synchronize and torch.cuda.is_available()
    free_min = free_before

    for i in range(cfg.warmup_iters):
        session.run([], {INPUT_NAME: batches[i % len(batches)]})
    if sync:
        torch.cuda.synchronize()
    if wants_gpu:
        free_min = min(free_min, _cuda_free_bytes())

    samples_s = []
    for i in range(cfg.timed_iters):
        batch = batches[i % len(batches)]
        start = time.perf_counter()
        session.run([], {INPUT_NAME: batch})
        if sync:
            torch.cuda.synchronize()
        samples_s.append(time.perf_counter() - start)
    if wants_gpu:
        free_min = min(free_min, _cuda_free_bytes())

    p50, p90, p99, mean, std = _stats(samples_s)
    peak_device = max(0.0, float(free_before - free_min) / BYTES_PER_MIB) if wants_gpu else 0.0
    return BenchResult(
        label=label,
        p50_ms=p50,
        p90_ms=p90,
        p99_ms=p99,
        mean_ms=mean,
        std_ms=std,
        throughput_windows_s=cfg.batch_size / (mean / 1e3),
        peak_host_mem_mb=_peak_host_mem_mb(),
        peak_device_mem_mb=peak_device,
        providers_realized=realized,
        intra_op_threads=int(options.intra_op_num_threads),
        tf32=False,
        config=cfg,
        env=environment_stamp(),
    )


def constants_from_sidecar(onnx_path: Path) -> NormalizerConstants:
    """Read a graph's normaliser from its sidecar (an ORT job never loads the checkpoint).

    Args:
        onnx_path: The graph; its sidecar is ``<graph>.json``.

    Returns:
        The constants.
    """
    meta = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))["normalizer"]
    return NormalizerConstants(
        mean=np.asarray(meta["mean"], dtype=np.float64),
        var=np.asarray(meta["var"], dtype=np.float64),
        epsilon=float(meta["epsilon"]),
        clip_obs=float(meta["clip_obs"]),
    )


def bench_inputs(
    constants: NormalizerConstants, n: int = BENCH_N_WINDOWS, seed: int = BENCH_WINDOW_SEED
) -> FloatArray:
    """Draw the cycled benchmark inputs: ``N(mu, sqrt(var + eps)^2)`` raw observations.

    Args:
        constants: The graph's normaliser.
        n: Distinct inputs (dmf's 64).
        seed: dmf's ``BENCH_WINDOW_SEED`` (not a parity seed).

    Returns:
        ``(n, D)`` float32.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, constants.dim))
    out: FloatArray = (np.asarray(constants.mean) + constants.std * noise).astype(np.float32)
    return out


@dataclass(frozen=True)
class PolicyBenchJob:
    """One latency configuration.

    Attributes:
        method: Method label of the timed graph.
        seed: Training seed of the timed graph.
        architecture: The latency unit.
        onnx_path: The graph.
        run_dir: The run directory (torch rows load the checkpoint from it).
        backend: Which runtime.
        batch_size: Observations per call.
        device: ``"cpu"`` or ``"cuda"`` (torch rows; ORT rows take it from the backend).
        warmup_iters: Untimed iterations.
        timed_iters: Timed iterations.
        intra_op_threads: Pinned CPU threads.
        n_inputs: Distinct inputs cycled.
        input_seed: Seed of the input draw.
    """

    method: str
    seed: int
    architecture: str
    onnx_path: Path
    run_dir: Path
    backend: Backend
    batch_size: int = 1
    device: str = "cpu"
    warmup_iters: int = 200
    timed_iters: int = 2000
    intra_op_threads: int = 1
    n_inputs: int = BENCH_N_WINDOWS
    input_seed: int = BENCH_WINDOW_SEED

    @property
    def resolved_device(self) -> str:
        """Device this job runs on."""
        return DEVICE_BY_BACKEND.get(self.backend, self.device)

    @property
    def label(self) -> str:
        """Row label, e.g. ``"mlp512x2-tanh-in25-ort-cpu-b1-t1"`` (dmf's label form)."""
        parts = [self.architecture, self.backend]
        if self.backend not in DEVICE_BY_BACKEND:
            parts.append(self.resolved_device)
        parts.extend([f"b{self.batch_size}", f"t{self.intra_op_threads}"])
        return "-".join(parts)

    @property
    def provider(self) -> str:
        """The provider name parity was checked under (``torch:<device>`` for torch rows)."""
        if self.backend == "torch-eager":
            return f"torch:{self.resolved_device}"
        return PROVIDERS_BY_BACKEND[self.backend][0]

    def bench_config(self) -> BenchConfig:
        """Return dmf's :class:`BenchConfig` for this job, ``synchronize`` always True."""
        return BenchConfig(
            warmup_iters=self.warmup_iters,
            timed_iters=self.timed_iters,
            batch_size=self.batch_size,
            device=self.resolved_device,
            synchronize=True,
            intra_op_threads=self.intra_op_threads,
        )


def job_payload(job: PolicyBenchJob) -> dict[str, Any]:
    """Serialise a job for the child."""
    return {
        "method": job.method,
        "seed": job.seed,
        "architecture": job.architecture,
        "onnx_path": str(job.onnx_path),
        "run_dir": str(job.run_dir),
        "backend": job.backend,
        "batch_size": job.batch_size,
        "device": job.device,
        "warmup_iters": job.warmup_iters,
        "timed_iters": job.timed_iters,
        "intra_op_threads": job.intra_op_threads,
        "n_inputs": job.n_inputs,
        "input_seed": job.input_seed,
    }


def job_from_payload(payload: Mapping[str, Any]) -> PolicyBenchJob:
    """Rebuild a job from :func:`job_payload`."""
    backend: Backend = payload["backend"]
    return PolicyBenchJob(
        method=str(payload["method"]),
        seed=int(payload["seed"]),
        architecture=str(payload["architecture"]),
        onnx_path=Path(str(payload["onnx_path"])),
        run_dir=Path(str(payload["run_dir"])),
        backend=backend,
        batch_size=int(payload["batch_size"]),
        device=str(payload["device"]),
        warmup_iters=int(payload["warmup_iters"]),
        timed_iters=int(payload["timed_iters"]),
        intra_op_threads=int(payload["intra_op_threads"]),
        n_inputs=int(payload["n_inputs"]),
        input_seed=int(payload["input_seed"]),
    )


def run_job(job: PolicyBenchJob) -> BenchResult:
    """Measure one configuration in this process (the child's body).

    ORT jobs read the graph and its sidecar only, never the checkpoint (dmf's rule); torch jobs
    load the checkpoint and fold it.

    Args:
        job: The configuration.

    Returns:
        The result.
    """
    constants = constants_from_sidecar(job.onnx_path)
    inputs = bench_inputs(constants, max(job.n_inputs, job.batch_size), job.input_seed)
    cfg = job.bench_config()
    if job.backend == "torch-eager":
        from rld.deploy.export import folded_actor
        from rld.eval.envs import load_eval_configs
        from rld.rl.train import build_policy

        module = folded_actor(build_policy(load_eval_configs(), run_dir=job.run_dir))
        return benchmark_torch(module, inputs, cfg, job.label)
    return benchmark_ort(
        job.onnx_path, inputs, cfg, job.label, providers=PROVIDERS_BY_BACKEND[job.backend]
    )


def run_job_subprocess(
    job: PolicyBenchJob, python: str | None = None, timeout_s: float = CHILD_TIMEOUT_S
) -> BenchResult:
    """Run one configuration in a fresh child under ``tf32_environment()`` (dmf's pattern).

    Args:
        job: The configuration.
        python: Interpreter (default: this one).
        timeout_s: Seconds before the child is killed.

    Returns:
        The child's result.

    Raises:
        RuntimeError: If the child exits non-zero or writes nothing (stderr included).
    """
    with tempfile.TemporaryDirectory(prefix="rld-bench-") as tmp:
        job_path = Path(tmp) / "job.json"
        out_path = Path(tmp) / "result.json"
        job_path.write_text(json.dumps(job_payload(job)), encoding="utf-8")
        completed = subprocess.run(
            [
                python or sys.executable,
                "-m",
                "rld.deploy.child",
                "bench",
                "--job",
                str(job_path),
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=tf32_environment(),
        )
        if completed.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                f"benchmark child for {job.label} exited {completed.returncode}\n"
                f"{completed.stderr.strip()[-4000:]}"
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    result = bench_result_from_payload(payload["results"][0])
    return replace(result, label=job.label)


def latency_jobs(timed: Sequence[tuple[str, int, str, Path, Path]]) -> list[PolicyBenchJob]:
    """Return the P8-D1 §8 grid for each timed graph.

    Args:
        timed: ``(method, seed, architecture, onnx_path, run_dir)`` per architecture.

    Returns:
        Per architecture: every backend (torch eager on CPU and CUDA) × batch {1, 32} at 1
        thread, then the {2, 4, 8}-thread sweep of ``ort-cpu`` and torch-eager CPU at both
        batches -- 22 jobs per architecture.
    """
    jobs: list[PolicyBenchJob] = []
    for method, seed, arch, onnx_path, run_dir in timed:

        def job(
            backend: Backend,
            batch: int,
            device: str = "cpu",
            threads: int = 1,
            *,
            _m: str = method,
            _s: int = seed,
            _a: str = arch,
            _o: Path = onnx_path,
            _r: Path = run_dir,
        ) -> PolicyBenchJob:
            return PolicyBenchJob(
                _m, _s, _a, _o, _r, backend, batch, device, intra_op_threads=threads
            )

        for batch in BATCH_SIZES:
            jobs.append(job("torch-eager", batch, "cpu"))
            jobs.append(job("torch-eager", batch, "cuda"))
            jobs.append(job("ort-cpu", batch))
            jobs.append(job("ort-cuda", batch))
            jobs.append(job("ort-trt", batch))
        for threads in CPU_THREADS[1:]:
            for batch in BATCH_SIZES:
                jobs.append(job("ort-cpu", batch, threads=threads))
                jobs.append(job("torch-eager", batch, "cpu", threads=threads))
    return jobs


@dataclass(frozen=True)
class LatencyRow:
    """One configuration's outcome.

    Attributes:
        job: The configuration.
        status: ``"timed"``, ``"refused"`` (parity) or ``"failed: <reason>"``.
        result: The measurement (``None`` unless timed).
    """

    job: PolicyBenchJob
    status: str
    result: BenchResult | None


def run_latency(
    jobs: Sequence[PolicyBenchJob],
    parity_rows: Sequence[ParityRow] | Sequence[Mapping[str, str]],
    runner: Callable[[PolicyBenchJob], BenchResult] = run_job_subprocess,
    log: Callable[[str], None] | None = None,
) -> list[LatencyRow]:
    """Time every non-refused job, one at a time.

    Args:
        jobs: The grid.
        parity_rows: The parity record; a refused (architecture, provider, threads) is never
            launched.
        runner: How a job is measured (a fresh subprocess; tests inject a recorder).
        log: Progress sink.

    Returns:
        One row per job, in job order.
    """
    rows: list[LatencyRow] = []
    for job in jobs:
        threads = job.intra_op_threads if job.resolved_device == "cpu" else 1
        if refused(parity_rows, job.architecture, job.provider, threads):
            rows.append(LatencyRow(job, "refused", None))
            if log is not None:
                log(f"refused (parity): {job.label}")
            continue
        try:
            result = runner(job)
        except RuntimeError as exc:
            reason = str(exc).strip().splitlines()[-1] if str(exc).strip() else "child failed"
            rows.append(LatencyRow(job, f"failed: {reason}", None))
            if log is not None:
                log(f"FAILED: {job.label}: {reason}")
            continue
        if job.backend != "torch-eager" and result.providers_realized[0] != job.provider:
            rows.append(LatencyRow(job, "failed: realized provider differs", None))
            continue
        rows.append(LatencyRow(job, "timed", result))
        if log is not None:
            log(f"{job.label}: p50 {result.p50_ms:.4f} ms, p99 {result.p99_ms:.4f} ms")
    return rows


#: Columns of ``results/latency/latency.csv``.
LATENCY_COLUMNS: tuple[str, ...] = (
    "label",
    "method",
    "seed",
    "architecture",
    "backend",
    "device",
    "provider",
    "provider_realized",
    "batch_size",
    "threads",
    "status",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "mean_ms",
    "std_ms",
    "throughput_obs_s",
    "peak_host_mem_mib",
    "peak_device_mem_mib",
    "tf32",
    "warmup_iters",
    "timed_iters",
)


def _fmt(value: float) -> str:
    """Six significant digits; latency is not byte-checked."""
    return f"{value:.6g}"


def latency_csv(rows: Sequence[LatencyRow]) -> str:
    """Render ``latency.csv``."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(LATENCY_COLUMNS)
    for row in rows:
        job, res = row.job, row.result
        nums: list[str] = (
            [""] * 9
            if res is None
            else [
                _fmt(res.p50_ms),
                _fmt(res.p90_ms),
                _fmt(res.p99_ms),
                _fmt(res.mean_ms),
                _fmt(res.std_ms),
                _fmt(res.throughput_windows_s),
                _fmt(res.peak_host_mem_mb),
                _fmt(res.peak_device_mem_mb),
                str(res.tf32),
            ]
        )
        writer.writerow(
            [
                job.label,
                job.method,
                job.seed,
                job.architecture,
                job.backend,
                job.resolved_device,
                job.provider,
                "" if res is None else "|".join(res.providers_realized),
                job.batch_size,
                job.intra_op_threads if res is None else res.intra_op_threads,
                row.status,
                *nums,
                job.warmup_iters,
                job.timed_iters,
            ]
        )
    return buffer.getvalue()


#: Columns of ``results/latency/h5.csv``.
H5_COLUMNS: tuple[str, ...] = (
    "hypothesis",
    "architecture",
    "batch_size",
    "cpu_label",
    "cpu_p50_ms",
    "cpu_p99_ms",
    "other_label",
    "other_provider",
    "other_status",
    "other_p50_ms",
    "other_p99_ms",
    "ratio_p50",
    "ratio_p99",
    "scored",
    "threshold",
    "meets_threshold",
    "verdict",
)


def score_h5(rows: Sequence[LatencyRow], architecture: str) -> tuple[str, list[list[str]]]:
    """Score H5 on one architecture (P3-D1 §8; P8-D1 §10).

    Args:
        rows: The latency rows.
        architecture: The ``ppo`` graph's architecture.

    Returns:
        ``(verdict, table rows)``: one row per GPU configuration at batch 1 (ORT CUDA and ORT
        TensorRT scored; torch-eager CUDA as context), with ratios ``other / ORT CPU``.

    Raises:
        ValueError: If the ORT CPU batch-1 1-thread row was not timed.
    """

    def find(backend: str, device: str) -> LatencyRow | None:
        for r in rows:
            j = r.job
            if (
                j.architecture == architecture
                and j.backend == backend
                and j.resolved_device == device
                and j.batch_size == 1
                and j.intra_op_threads == 1
            ):
                return r
        return None

    cpu = find("ort-cpu", "cpu")
    if cpu is None or cpu.result is None:
        raise ValueError("H5 needs a timed ORT CPU batch-1 1-thread row")
    others = [
        (find("ort-cuda", "cuda"), True),
        (find("ort-trt", "cuda"), True),
        (find("torch-eager", "cuda"), False),
    ]
    scored_ratios: list[float] = []
    table: list[list[str]] = []
    for other, scored in others:
        if other is None:
            continue
        res = other.result
        is_scored = scored and res is not None
        if res is None:
            ratio50 = ratio99 = float("nan")
        else:
            ratio50 = res.p50_ms / cpu.result.p50_ms
            ratio99 = res.p99_ms / cpu.result.p99_ms
        if is_scored:
            scored_ratios.append(ratio50)
        table.append(
            [
                "H5",
                architecture,
                "1",
                cpu.job.label,
                _fmt(cpu.result.p50_ms),
                _fmt(cpu.result.p99_ms),
                other.job.label,
                other.job.provider,
                other.status,
                "" if res is None else _fmt(res.p50_ms),
                "" if res is None else _fmt(res.p99_ms),
                "" if res is None else _fmt(ratio50),
                "" if res is None else _fmt(ratio99),
                str(is_scored),
                _fmt(H5_THRESHOLD),
                "" if res is None else str(ratio50 >= H5_THRESHOLD),
            ]
        )
    if not scored_ratios:
        verdict = "not scored - no parity-passing GPU provider"
    elif all(r >= H5_THRESHOLD for r in scored_ratios):
        verdict = "supported"
    else:
        verdict = "not supported"
    for line in table:
        line.append(verdict)
    return verdict, table


def h5_csv(table: Sequence[Sequence[str]]) -> str:
    """Render ``h5.csv``."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(H5_COLUMNS)
    writer.writerows(table)
    return buffer.getvalue()
