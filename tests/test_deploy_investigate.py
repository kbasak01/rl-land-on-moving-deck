"""P8-D3 investigation logic: the ulp and float64 arms, the noise statistics, the fixed readings.

Units: actions normalised in [-1, 1]; positions metres model scale.
"""

import numpy as np

from _deploy_helpers import tiny_policy
from rld.deploy import investigate as inv
from rld.deploy.export import normalizer_constants
from rld.deploy.parity import parity_draw


def _ulp_policy(k: int):  # type: ignore[no-untyped-def]
    p = tiny_policy()
    inv._reclass(p, inv._UlpRandomMixin, f"ulp_{k}")
    p.ulp_k = k  # type: ignore[attr-defined]
    return p


def test_ulp_arm_is_seeded_per_episode_and_one_ulp() -> None:
    torch_policy = tiny_policy()
    x, _, _ = parity_draw(normalizer_constants(torch_policy), seed=0)
    a, b = _ulp_policy(3), _ulp_policy(3)
    a.reset(11)
    b.reset(11)
    first = [a.network_action(o) for o in x[:20]]
    assert all(
        np.array_equal(u, v)
        for u, v in zip(first, [b.network_action(o) for o in x[:20]], strict=False)
    )
    a.reset(11)  # re-seeded at reset: the same sequence again
    assert all(np.array_equal(u, a.network_action(o)) for u, o in zip(first, x[:20], strict=False))
    c = _ulp_policy(4)
    c.reset(11)
    assert not all(
        np.array_equal(u, c.network_action(o)) for u, o in zip(first, x[:20], strict=False)
    )
    # The perturbation is tiny: actions within the parity criterion of the unperturbed path.
    for o in x[:50]:
        assert np.max(np.abs(a.network_action(o) - torch_policy.network_action(o))) < 1e-4
    # The nudge itself is exactly one ulp per entry.
    rng = np.random.default_rng(0)
    v = x[:1].astype(np.float32)
    up = rng.random(v.shape) < 0.5
    w = np.where(up, np.nextafter(v, np.float32(np.inf)), np.nextafter(v, np.float32(-np.inf)))
    assert np.all(w != v) and np.all(np.abs(w - v) <= np.spacing(np.abs(v)))


def test_fp64_arm_agrees_with_float32_to_rounding() -> None:
    import copy

    import torch

    p32 = tiny_policy()
    p64 = tiny_policy()
    c = normalizer_constants(p64)
    inv._reclass(p64, inv._Fp64NetworkMixin, "fp64")
    sb3 = p64.model.policy
    p64.fp64_net = copy.deepcopy(  # type: ignore[attr-defined]
        torch.nn.Sequential(sb3.mlp_extractor.policy_net, sb3.action_net)
    ).double()
    p64.fp64_mean, p64.fp64_std, p64.fp64_clip = c.mean, c.std, c.clip_obs  # type: ignore[attr-defined]
    x, _, _ = parity_draw(c, seed=2)
    for o in x[::97]:
        a64 = p64.network_action(o)
        assert a64.dtype == np.float64
        assert np.max(np.abs(a64 - p32.network_action(o))) < 1e-4


def test_rank_and_noise_rows() -> None:
    assert inv._rank(3, [1, 2, 3, 4]) == (2 + 0.5) / 4
    ss6 = [("SS6", i) for i in range(5)]
    base = dict.fromkeys(ss6, "success")
    outcomes: inv.Outcomes = {("m", 0, "torch"): dict(base)}
    onnx = dict(base)
    onnx[("SS6", 1)] = "hard_landing"
    outcomes[("m", 0, "onnx")] = onnx
    for k, arm in enumerate(inv.ulp_arms()):
        o = dict(base)
        if k % 2 == 0:
            o[("SS6", 1)] = "hard_landing"
        outcomes[("m", 0, arm)] = o
    sens = inv.sensitive_sets(outcomes, ss6)
    assert sens[("m", 0)] == {("SS6", 1)}
    rows = inv.noise_rows(outcomes, ss6, sens)
    per = rows[0]
    assert per[3] == "1" and per[7] == "1" and per[10] == "1" and per[11] == "0"
    assert rows[-1][0] == "pooled"


def _summary_row(flipped: bool, k0_err: float, cross8: str, cross4: str, td: str) -> list[str]:
    row = [""] * len(inv.TRACE_SUMMARY_COLUMNS)
    row[4] = "torch_vs_onnx"
    row[7] = str(flipped)
    row[11] = repr(k0_err)
    row[inv.TRACE_SUMMARY_COLUMNS.index("step_dp_gt_1e-08")] = cross8
    row[inv.TRACE_SUMMARY_COLUMNS.index("step_dp_gt_1e-04")] = cross4
    row[inv.TRACE_SUMMARY_COLUMNS.index("touchdown_step_torch")] = td
    return row


def _same(max_err: float) -> list[str]:
    return [
        "m",
        "0",
        "onnx",
        "100",
        "0",
        repr(max_err),
        "0",
        "5",
        repr(max_err),
        "0",
        "nan",
        repr(max_err),
    ]


def _noise(onnx: int, mx: int, outside: int) -> list[list[str]]:
    per = ["m", "0", "200", str(onnx), "", "0", "0", str(mx), "0", "3", "", "", "0", "", ""]
    pooled = [
        "pooled",
        "",
        "200",
        str(onnx),
        "",
        "0",
        "0",
        str(mx),
        "0",
        "3",
        "",
        str(outside),
        "0",
        "",
        "",
    ]
    return [per, pooled]


def test_fixed_readings() -> None:
    clean = [_summary_row(True, 1e-7, "5", "30", "50"), _summary_row(False, 6e-8, "4", "40", "60")]
    reading, _ = inv.verdict_rows([_same(5e-7)], clean, _noise(1, 3, 0))
    assert reading == "rounding sensitivity"
    assert inv.verdict_rows([_same(5e-4)], clean, _noise(1, 3, 0))[0] == "export defect"
    assert inv.verdict_rows([_same(5e-7)], clean, _noise(4, 3, 0))[0] == "export defect"
    assert inv.verdict_rows([_same(5e-7)], clean, _noise(1, 3, 2))[0] == "export defect"
    late = [_summary_row(True, 1e-7, "5", "", "50")]  # no 4-decade growth before touchdown
    assert inv.verdict_rows([_same(5e-7)], late, _noise(1, 3, 0))[0] == "inconclusive"


def _outcomes_for_posthoc() -> tuple[inv.Outcomes, list[tuple[str, int]]]:
    ss6 = [("SS6", i) for i in range(6)]
    base = dict.fromkeys(ss6, "success")
    out: inv.Outcomes = {("m", 0, "torch"): dict(base)}
    for arm in ("onnx", "onnx_cuda", "torch_folded", "torch_fp64"):
        out[("m", 0, arm)] = dict(base)
    out[("m", 0, "onnx")][("SS6", 1)] = "bounce"
    out[("m", 0, "onnx_cuda")][("SS6", 5)] = "hard_landing"  # outside S
    for k, arm in enumerate(inv.ulp_arms()):
        o = dict(base)
        o[("SS6", 1 if k % 2 else 2)] = "bounce" if k % 2 else "hard_landing"
        out[("m", 0, arm)] = o
    return out, ss6


def test_posthoc_bias_and_loo() -> None:
    outcomes, ss6 = _outcomes_for_posthoc()
    bias = inv.posthoc_bias_rows(outcomes, ss6)
    onnx = next(r for r in bias if r[0] == "m" and r[2] == "onnx" and r[3] == "torch")
    assert onnx[5:8] == ["1", "-1", "1"]  # one flip, success -1, success->bounce
    ulp = next(r for r in bias if r[0] == "m" and r[2] == "ulp")
    assert ulp[5] == "20" and ulp[13] == "20" and ulp[14] == "20"
    assert float(ulp[12]) == 0.5
    loo = inv.posthoc_loo_rows(outcomes, ss6)
    pooled = loo[-1]
    assert pooled[2:4] == ["20", "0"] and pooled[5:7] == ["1", "1"]
