"""Numeric parity of an exported policy graph, per provider, before anything is timed (P8-D1 §6).

A rank-2 mirror of :func:`dmf.deploy.parity.check_parity` (dmf's requires ``(n, L, C)`` windows):
the same scale-relative criterion ``max_abs_err < 1e-4 * max(1, |y|max)`` with ``|y|max`` read off
the reference, the unscaled verdict beside it, five draws with dmf's
:data:`~dmf.deploy.parity.PARITY_SEEDS`, the worst draw as the verdict, the realized provider read
back and refused if it is not the requested one, and :class:`dmf.deploy.parity.ParityResult` as the
per-draw record.

The draw
--------
1 000 raw observations per draw from the graph's own normaliser, ``s = sqrt(var + eps)`` per entry:
rows 0-799 from ``N(mu, s^2)`` and rows 800-999 from ``N(mu, (20 s)^2)``. The tail rows push most
entries past ``|z| = clip_obs = 10``, so the clip branch of the fold is exercised; the fraction of
clipped entries is recorded per draw.

The reference
-------------
:meth:`rld.rl.policy.LearnedPolicy.network_action`, batched, on the CPU, in its own arithmetic:
``VecNormalize.normalize_obs`` (float64) -> float32 -> SB3 ``predict(deterministic=True)`` -> clip
``[-1, 1]``. The normaliser is applied **outside** the network there and **inside** the graph here,
so the fold itself is under test.

Providers
---------
ORT ``CPUExecutionProvider`` (at every thread count the latency sweep times),
``CUDAExecutionProvider`` and ``TensorrtExecutionProvider``, plus the folded torch module on the
CPU (``torch:cpu``, at the sweep's thread counts) and on CUDA (``torch:cuda``): the torch-eager
reference rows are timed too, and a timed row whose arithmetic was not checked is not allowed
(dmf P7-D9).
TF32 is off on every path (:func:`dmf.deploy.providers.disable_tf32`, and the parity child runs
under :func:`dmf.deploy.providers.tf32_environment`).

Refusal
-------
An (architecture, provider) is **refused** when any exported seed of the architecture fails on that
provider, or has no parity row for it. :func:`refused` is what the latency driver reads; a refused
configuration is written as ``refused`` and never launched.

Units: observations in observation units (model scale); actions and errors in normalised action
units (dimensionless).
"""

import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from dmf.deploy.export_onnx import INPUT_NAME
from dmf.deploy.parity import PARITY_N_WINDOWS, ParityResult
from dmf.deploy.providers import disable_tf32, preload_gpu_libraries, session_providers
from dmf.typedefs import FloatArray

from rld.deploy.export import FoldedActor, NormalizerConstants
from rld.rl.policy import LearnedPolicy

__all__ = [
    "CPU_THREADS",
    "GPU_PARITY_PROVIDERS",
    "ORT_PROVIDERS",
    "PARITY_COLUMNS",
    "PARITY_TOLERANCE",
    "TAIL_FRACTION",
    "TAIL_SCALE",
    "TORCH_PROVIDERS",
    "ParityRow",
    "check_ort",
    "check_torch",
    "deterministic_provider",
    "open_session",
    "parity_csv",
    "parity_draw",
    "read_parity_csv",
    "reference_actions",
    "refused",
    "verdicts",
]

#: The relative coefficient, dimensionless (plan Phase 8 step 2; dmf P7-D3 form).
PARITY_TOLERANCE: float = 1e-4

#: Fraction of each draw taken from the wide tail, dimensionless (rows 800-999 of 1 000).
TAIL_FRACTION: float = 0.2

#: Standard-deviation multiplier of the tail rows, dimensionless.
TAIL_SCALE: float = 20.0

#: Thread counts of the CPU latency sweep (P8-D1 §8); parity is checked at each.
CPU_THREADS: tuple[int, ...] = (1, 2, 4, 8)

#: ORT providers checked, in report order.
ORT_PROVIDERS: tuple[str, ...] = (
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
)

#: Torch reference "providers" (dmf's ``torch:<device>`` label form).
TORCH_PROVIDERS: tuple[str, ...] = ("torch:cpu", "torch:cuda")

#: Providers whose kernels run on the GPU; their errors are not bit-reproducible between
#: sessions (TensorRT tactic selection), so ``--check`` compares their verdicts, not their digits.
GPU_PARITY_PROVIDERS: frozenset[str] = frozenset(
    {"CUDAExecutionProvider", "TensorrtExecutionProvider", "torch:cuda"}
)

#: Columns of ``results/latency/parity.csv``.
PARITY_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "architecture",
    "onnx_sha256",
    "provider",
    "provider_realized",
    "threads",
    "draw_seed",
    "n_obs",
    "n_tail",
    "clipped_fraction",
    "output_abs_max",
    "max_abs_err",
    "mean_abs_err",
    "tolerance",
    "scaled_tolerance",
    "passed",
    "passed_absolute",
    "graph_provider_passed",
    "note",
)


def deterministic_provider(provider: str) -> bool:
    """Whether a provider's parity digits are byte-checked (CPU paths only)."""
    return provider not in GPU_PARITY_PROVIDERS


def parity_draw(
    constants: NormalizerConstants,
    seed: int,
    n_obs: int = PARITY_N_WINDOWS,
    tail_fraction: float = TAIL_FRACTION,
    tail_scale: float = TAIL_SCALE,
) -> tuple[FloatArray, int, float]:
    """Draw one parity batch of raw observations.

    Args:
        constants: The graph's frozen normaliser.
        seed: Draw seed.
        n_obs: Observations per draw.
        tail_fraction: Fraction of rows from the wide tail (the last rows).
        tail_scale: Standard-deviation multiplier of the tail rows.

    Returns:
        ``(x, n_tail, clipped_fraction)``: ``x`` is ``(n_obs, D)`` float32 raw observations;
        ``clipped_fraction`` is the fraction of entries whose normalised value exceeds
        ``clip_obs`` in magnitude.
    """
    rng = np.random.default_rng(seed)
    n_tail = int(round(n_obs * tail_fraction))
    scale = np.ones((n_obs, 1), dtype=np.float64)
    scale[n_obs - n_tail :] = tail_scale
    std = constants.std
    noise = rng.standard_normal((n_obs, constants.dim))
    x = (np.asarray(constants.mean) + std * noise * scale).astype(np.float32)
    z = (x.astype(np.float64) - np.asarray(constants.mean)) / std
    clipped = float(np.mean(np.abs(z) > constants.clip_obs))
    return cast(FloatArray, x), n_tail, clipped


def reference_actions(policy: LearnedPolicy, x: FloatArray) -> FloatArray:
    """Return :meth:`LearnedPolicy.network_action` for a batch, in its own (CPU) arithmetic.

    Args:
        policy: The loaded policy (CPU model, frozen normaliser).
        x: ``(n, D)`` float32 raw policy inputs.

    Returns:
        ``(n, 3)`` float32 actions in ``[-1, 1]``.

    Raises:
        ValueError: If the policy has no normaliser.
    """
    if policy.normalizer is None:
        raise ValueError(f"{policy.name}: no normaliser")
    z = np.asarray(policy.normalizer.normalize_obs(np.asarray(x, dtype=np.float32)))
    action, _ = policy.model.predict(z.astype(np.float32), deterministic=True)
    out: FloatArray = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if out.shape != (x.shape[0], 3):
        raise ValueError(f"reference shape {out.shape} != ({x.shape[0]}, 3)")
    return out


def _result(
    reference: FloatArray, candidate: FloatArray, seed: int, provider: str, tolerance: float
) -> ParityResult:
    """Reduce one comparison to dmf's :class:`ParityResult` (dmf's arithmetic, verbatim)."""
    if candidate.shape != reference.shape:
        raise ValueError(f"{provider}: output {candidate.shape} != reference {reference.shape}")
    err = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    output_abs_max = float(np.abs(reference).max())
    scale = max(1.0, output_abs_max)
    max_abs_err = float(err.max())
    return ParityResult(
        max_abs_err=max_abs_err,
        mean_abs_err=float(err.sum()) / int(err.size),
        n_windows=int(reference.shape[0]),
        tolerance=tolerance,
        passed=max_abs_err < tolerance * scale,
        output_abs_max=output_abs_max,
        scale=scale,
        scaled_tolerance=tolerance * scale,
        passed_absolute=max_abs_err < tolerance,
        seed=seed,
        provider=provider,
    )


def open_session(onnx_path: Path, provider: str, threads: int = 1) -> tuple[Any, str]:
    """Open an ORT session for a parity check and read the realized provider back.

    Args:
        onnx_path: The graph.
        provider: Requested execution provider (the CPU provider is the fallback on GPU rows,
            as in dmf's ``PROVIDERS_BY_BACKEND``).
        threads: ``intra_op_num_threads`` = ``inter_op_num_threads``.

    Returns:
        ``(session, first realized provider)``.

    Raises:
        FileNotFoundError: If the graph is missing.
        ValueError: If the requested provider is not the one the session realized.
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"no exported graph at {onnx_path}")
    import onnxruntime as ort

    preload_gpu_libraries()
    disable_tf32()
    requested = (
        (provider,) if provider == "CPUExecutionProvider" else (provider, "CPUExecutionProvider")
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=session_providers(requested)
    )
    realized = tuple(str(p) for p in session.get_providers())
    if provider not in realized or realized[0] != provider:
        raise ValueError(
            f"requested {provider!r} for the parity check but the session realized "
            f"{list(realized)}; a parity row against a provider that did not load would certify "
            "the wrong runtime"
        )
    return session, realized[0]


def check_ort(
    session: Any,
    x: FloatArray,
    reference: FloatArray,
    provider: str,
    *,
    seed: int,
    tolerance: float = PARITY_TOLERANCE,
) -> ParityResult:
    """Compare an open ORT session with the reference (rank-2 ``check_parity``).

    Args:
        session: From :func:`open_session`.
        x: ``(n, D)`` float32 raw inputs.
        reference: ``(n, 3)`` reference actions.
        provider: The session's provider, recorded.
        seed: Draw seed, recorded.
        tolerance: Relative coefficient.

    Returns:
        The result.
    """
    (out,) = session.run([], {INPUT_NAME: np.ascontiguousarray(x, dtype=np.float32)})
    return _result(reference, np.asarray(out), seed, provider, tolerance)


def check_torch(
    module: FoldedActor,
    x: FloatArray,
    reference: FloatArray,
    device: str,
    *,
    seed: int,
    threads: int = 1,
    tolerance: float = PARITY_TOLERANCE,
) -> ParityResult:
    """Compare the folded torch module on ``device`` with the reference.

    Args:
        module: The folded actor (moved to ``device`` and back to the CPU).
        x: ``(n, D)`` float32 raw inputs.
        reference: ``(n, 3)`` reference actions.
        device: ``"cpu"`` or ``"cuda"``.
        seed: Draw seed, recorded.
        threads: ``torch.set_num_threads`` for the comparison.
        tolerance: Relative coefficient.

    Returns:
        The result, ``provider = "torch:<device>"``.

    Raises:
        ValueError: If CUDA is requested but unavailable.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but no CUDA device is available")
    disable_tf32()
    torch.set_num_threads(threads)
    module.to(device).eval()
    try:
        with torch.no_grad():
            out = module(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device))
            got = out.detach().cpu().numpy()
    finally:
        module.to("cpu")
    return _result(reference, got, seed, f"torch:{device}", tolerance)


@dataclass(frozen=True)
class ParityRow:
    """One (graph, provider, threads, draw) parity record.

    Attributes:
        method: Method label.
        seed: Training seed of the graph.
        architecture: The latency unit (``mlp512x2-tanh-in<D>``).
        onnx_sha256: SHA-256 of the graph.
        provider: Requested provider (or ``torch:<device>``).
        provider_realized: First realized provider (``""`` when the session failed).
        threads: CPU threads of the comparison.
        draw_seed: Seed of the draw (one of dmf's ``PARITY_SEEDS``).
        n_tail: Tail rows in the draw.
        clipped_fraction: Fraction of normalised entries beyond the clip.
        result: dmf's :class:`ParityResult` (``None`` when the provider did not load).
        note: Why there is no result, or ``""``.
    """

    method: str
    seed: int
    architecture: str
    onnx_sha256: str
    provider: str
    provider_realized: str
    threads: int
    draw_seed: int
    n_tail: int
    clipped_fraction: float
    result: ParityResult | None
    note: str = ""

    @property
    def passed(self) -> bool:
        """The verdict of this draw (False when the provider did not load)."""
        return self.result is not None and bool(self.result.passed)


def verdicts(rows: Iterable[ParityRow]) -> dict[tuple[str, int, str, int], bool]:
    """Worst-of-draws verdict per (method, seed, provider, threads)."""
    out: dict[tuple[str, int, str, int], bool] = {}
    for row in rows:
        key = (row.method, row.seed, row.provider, row.threads)
        out[key] = out.get(key, True) and row.passed
    return out


def refused(
    rows: Sequence[ParityRow] | Sequence[Mapping[str, str]],
    architecture: str,
    provider: str,
    threads: int,
) -> bool:
    """Whether (architecture, provider, threads) is refused for timing.

    Refused when **any** seed of the architecture fails a draw on that provider at that thread
    count, or when no parity row exists for it at all (an unchecked provider is never timed).

    Args:
        rows: Parity rows, as :class:`ParityRow` or as ``parity.csv`` dicts.
        architecture: The latency unit.
        provider: Requested provider (``torch:<device>`` for the torch rows).
        threads: CPU threads (GPU rows are checked at 1).

    Returns:
        True if the configuration must not be timed.
    """
    matched = 0
    for row in rows:
        if isinstance(row, ParityRow):
            arch, prov, thr, ok = row.architecture, row.provider, row.threads, row.passed
        else:
            arch, prov, thr = row["architecture"], row["provider"], int(row["threads"])
            ok = row["passed"] == "True"
        if arch == architecture and prov == provider and thr == threads:
            matched += 1
            if not ok:
                return True
    return matched == 0


def parity_csv(rows: Sequence[ParityRow]) -> str:
    """Render ``parity.csv`` deterministically (floats as ``repr``)."""
    worst = verdicts(rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(PARITY_COLUMNS)
    for row in rows:
        res = row.result
        nan = repr(float("nan"))
        writer.writerow(
            [
                row.method,
                row.seed,
                row.architecture,
                row.onnx_sha256,
                row.provider,
                row.provider_realized,
                row.threads,
                row.draw_seed,
                PARITY_N_WINDOWS if res is None else res.n_windows,
                row.n_tail,
                repr(row.clipped_fraction),
                nan if res is None else repr(res.output_abs_max),
                nan if res is None else repr(res.max_abs_err),
                nan if res is None else repr(res.mean_abs_err),
                repr(PARITY_TOLERANCE if res is None else res.tolerance),
                nan if res is None else repr(res.scaled_tolerance),
                row.passed,
                False if res is None else bool(res.passed_absolute),
                worst[(row.method, row.seed, row.provider, row.threads)],
                row.note,
            ]
        )
    return buffer.getvalue()


def read_parity_csv(path: Path) -> list[dict[str, str]]:
    """Read ``parity.csv`` as text rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
