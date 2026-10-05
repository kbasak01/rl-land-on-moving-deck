"""The ONNX override of ``network_action`` (P8-D1 §7): same action as torch; base and alpha kept.

Units: actions normalised in [-1, 1].
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _deploy_helpers import tiny_policy
from rld.deploy.export import export_folded, folded_actor, normalizer_constants
from rld.deploy.onnx_policy import (
    OnnxLearnedPolicy,
    OnnxNetworkMixin,
    OnnxResidualPolicy,
    attach_onnx,
)
from rld.deploy.parity import parity_draw
from rld.rl.policy import LearnedPolicy
from rld.rl.residual import ResidualPolicy, compose_residual


class _Base:
    """A deterministic stand-in base controller (not privileged, no feed)."""

    name = "stub_base"
    privileged = False
    needs_motion_feed = False

    def __init__(self) -> None:
        self.calls = 0

    def reset(self, seed: int, context: Any = None, motion_feed: Any = None) -> None:
        del seed, context, motion_feed
        self.calls = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        self.calls += 1
        return np.array([0.2, -0.1, -0.5], dtype=np.float32)


def _graph(policy: LearnedPolicy, tmp_path: Path) -> Path:
    dim = normalizer_constants(policy).dim
    return export_folded(folded_actor(policy), dim, tmp_path / "g.onnx")


def test_mixin_action_equals_torch_action(tmp_path: Path) -> None:
    torch_policy = tiny_policy()
    onnx_policy = attach_onnx(tiny_policy(), _graph(torch_policy, tmp_path))
    assert type(onnx_policy) is OnnxLearnedPolicy
    assert isinstance(onnx_policy, OnnxNetworkMixin) and isinstance(onnx_policy, LearnedPolicy)
    x, _, _ = parity_draw(normalizer_constants(torch_policy), seed=11)
    for obs in x[::50]:
        a_t = torch_policy.network_action(obs)
        a_o = onnx_policy.network_action(obs)
        assert a_o.shape == (3,) and a_o.dtype == np.float32
        np.testing.assert_allclose(a_o, a_t, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(onnx_policy.act(obs), a_o)


def test_residual_twin_keeps_base_and_alpha(tmp_path: Path) -> None:
    pure = tiny_policy()
    path = _graph(pure, tmp_path)
    base = _Base()
    residual = ResidualPolicy(
        "r",
        pure.model,
        pure.normalizer,
        Path(),
        Path(),
        0,
        base=base,
        alpha=0.3,  # type: ignore[arg-type]
    )
    twin = attach_onnx(residual, path)
    assert type(twin) is OnnxResidualPolicy and twin.base is base and twin.alpha == 0.3
    twin.reset(5)
    obs = normalizer_constants(pure).mean.astype(np.float32)
    expected = compose_residual(base.act(obs), twin.network_action(obs), 0.3)
    base.reset(5)
    np.testing.assert_array_equal(twin.act(obs), expected)
    assert base.calls == 1


def test_width_mismatch_refused(tmp_path: Path) -> None:
    path = _graph(tiny_policy(dim=31), tmp_path)
    with pytest.raises(ValueError, match="input width"):
        attach_onnx(tiny_policy(dim=25), path)
