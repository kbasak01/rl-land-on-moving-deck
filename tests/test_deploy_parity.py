"""Numeric parity (P8-D1 §6): the draw, the criterion, refusal, and the ``--check`` comparison.

Units: observations dimensionless (synthetic); errors in normalised action units.
"""

from pathlib import Path

import numpy as np
import pytest
from dmf.deploy.parity import PARITY_SEEDS, ParityResult

from _deploy_helpers import tiny_policy
from rld.deploy.export import export_folded, folded_actor, normalizer_constants
from rld.deploy.parity import (
    PARITY_TOLERANCE,
    ParityRow,
    check_ort,
    check_torch,
    open_session,
    parity_csv,
    parity_draw,
    reference_actions,
    refused,
    verdicts,
)
from rld.deploy.pipeline import compare_parity


def test_draw_is_seeded_and_reaches_the_clip() -> None:
    constants = normalizer_constants(tiny_policy())
    x1, n_tail, clipped = parity_draw(constants, seed=PARITY_SEEDS[0])
    x2, _, _ = parity_draw(constants, seed=PARITY_SEEDS[0])
    x3, _, _ = parity_draw(constants, seed=PARITY_SEEDS[1])
    assert x1.shape == (1000, 25) and x1.dtype == np.float32
    assert np.array_equal(x1, x2) and not np.array_equal(x1, x3)
    z = (x1.astype(np.float64) - constants.mean) / constants.std
    assert np.abs(z[:800]).max() < 10.0 < np.abs(z[800:]).max()
    assert n_tail == 200 and 0.05 < clipped < 0.2


def test_cpu_provider_passes_and_records_the_dmf_criterion(tmp_path: Path) -> None:
    policy = tiny_policy()
    constants = normalizer_constants(policy)
    path = export_folded(folded_actor(policy), constants.dim, tmp_path / "g.onnx")
    session, realized = open_session(path, "CPUExecutionProvider", threads=2)
    assert realized == "CPUExecutionProvider"
    x, _, _ = parity_draw(constants, seed=0)
    ref = reference_actions(policy, x)
    res = check_ort(session, x, ref, realized, seed=0)
    assert res.passed and res.passed_absolute
    assert res.scale == max(1.0, res.output_abs_max)
    assert res.scaled_tolerance == PARITY_TOLERANCE * res.scale
    t = check_torch(folded_actor(policy), x, ref, "cpu", seed=0, threads=1)
    assert t.passed and t.provider == "torch:cpu"


def test_a_wrong_graph_fails(tmp_path: Path) -> None:
    good, other = tiny_policy(seed=0), tiny_policy(seed=1)
    constants = normalizer_constants(good)
    path = export_folded(folded_actor(other), constants.dim, tmp_path / "bad.onnx")
    session, _ = open_session(path, "CPUExecutionProvider")
    x, _, _ = parity_draw(constants, seed=0)
    assert not check_ort(session, x, reference_actions(good, x), "CPU", seed=0).passed


def _row(arch: str, provider: str, threads: int, passed: bool, seed: int = 0) -> ParityRow:
    res = ParityResult(
        max_abs_err=1e-7 if passed else 1.0,
        mean_abs_err=1e-8,
        n_windows=1000,
        tolerance=1e-4,
        passed=passed,
        output_abs_max=1.0,
        scale=1.0,
        scaled_tolerance=1e-4,
        passed_absolute=passed,
        seed=0,
        provider=provider,
    )
    return ParityRow("m", seed, arch, "sha", provider, provider, threads, 0, 200, 0.1, res)


def test_refusal_rule() -> None:
    rows = [
        _row("A", "CPUExecutionProvider", 1, True),
        _row("A", "TensorrtExecutionProvider", 1, True, seed=0),
        _row("A", "TensorrtExecutionProvider", 1, False, seed=4),  # one seed fails
        _row("A", "CUDAExecutionProvider", 1, True),
    ]
    assert not refused(rows, "A", "CPUExecutionProvider", 1)
    assert refused(rows, "A", "TensorrtExecutionProvider", 1)  # any seed failing refuses
    assert refused(rows, "A", "CPUExecutionProvider", 4)  # never checked at 4 threads
    assert refused(rows, "B", "CUDAExecutionProvider", 1)  # never checked for this graph
    # The same rule read back from the CSV.
    import csv
    import io

    text_rows = list(csv.DictReader(io.StringIO(parity_csv(rows))))
    assert refused(text_rows, "A", "TensorrtExecutionProvider", 1)
    assert not refused(text_rows, "A", "CUDAExecutionProvider", 1)
    worst = verdicts(rows)
    assert worst[("m", 4, "TensorrtExecutionProvider", 1)] is False


def test_unloaded_provider_row_fails() -> None:
    row = ParityRow("m", 0, "A", "sha", "TensorrtExecutionProvider", "", 1, 0, 200, 0.1, None, "x")
    assert not row.passed
    assert refused([row], "A", "TensorrtExecutionProvider", 1)
    assert "nan" in parity_csv([row])


def test_check_compares_cpu_bytes_and_gpu_verdicts() -> None:
    cpu = _row("A", "CPUExecutionProvider", 1, True)
    gpu = _row("A", "TensorrtExecutionProvider", 1, True)
    committed = parity_csv([cpu, gpu])
    # GPU digits move between sessions: tolerated while below the threshold.
    moved = ParityResult(**{**gpu.result.__dict__, "max_abs_err": 3e-7})  # type: ignore[union-attr]
    fresh = parity_csv([cpu, ParityRow(**{**gpu.__dict__, "result": moved})])
    assert committed != fresh and compare_parity(committed, fresh) == []
    # A CPU digit moving is a failure.
    cpu_moved = ParityResult(**{**cpu.result.__dict__, "max_abs_err": 3e-7})  # type: ignore[union-attr]
    fresh_cpu = parity_csv([ParityRow(**{**cpu.__dict__, "result": cpu_moved}), gpu])
    assert compare_parity(committed, fresh_cpu)
    # A GPU verdict moving is a failure.
    bad = _row("A", "TensorrtExecutionProvider", 1, False)
    assert compare_parity(committed, parity_csv([cpu, bad]))


@pytest.mark.gpu
def test_gpu_providers_pass_with_tf32_off(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    policy = tiny_policy()
    constants = normalizer_constants(policy)
    path = export_folded(folded_actor(policy), constants.dim, tmp_path / "g.onnx")
    x, _, _ = parity_draw(constants, seed=0)
    ref = reference_actions(policy, x)
    session, realized = open_session(path, "CUDAExecutionProvider")
    assert realized == "CUDAExecutionProvider"
    assert check_ort(session, x, ref, realized, seed=0).passed
    assert check_torch(folded_actor(policy), x, ref, "cuda", seed=0).passed
