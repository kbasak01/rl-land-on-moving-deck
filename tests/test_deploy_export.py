"""ONNX export (P8-D1 §4): the fold is ``network_action``, clip included; dynamic batch only.

Units: observations dimensionless (synthetic); actions normalised in [-1, 1].
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from dmf.deploy.export_onnx import graph_dims

from _deploy_helpers import tiny_policy
from rld.deploy.export import (
    EXPORTS,
    FoldedActor,
    export_folded,
    folded_actor,
    graph_name,
    normalizer_constants,
)
from rld.deploy.onnx_policy import cpu_session
from rld.deploy.parity import PARITY_TOLERANCE, parity_draw, reference_actions

#: The P8-D1 §6 criterion (|y|max <= 1 here). The synthetic sigma = 0.004 entry makes float32
#: rounding of mu reach ~1e-5 through random weights; the real graphs sit near 1e-6.
TOL = PARITY_TOLERANCE


def _torch(module: FoldedActor, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return module(torch.from_numpy(x)).numpy()


def test_fold_equals_network_action_including_the_clip() -> None:
    policy = tiny_policy()
    constants = normalizer_constants(policy)
    module = folded_actor(policy)
    x, n_tail, clipped = parity_draw(constants, seed=7)
    assert n_tail == 200 and clipped > 0.05  # the tail rows really reach the clip
    ref = reference_actions(policy, x)
    np.testing.assert_allclose(_torch(module, x), ref, rtol=0, atol=TOL)
    # Row by row through LearnedPolicy.network_action itself (batch 1), on clipped rows too.
    for i in (0, 1, 850, 999):
        np.testing.assert_allclose(
            _torch(module, x[i : i + 1])[0], policy.network_action(x[i]), atol=TOL
        )


def test_the_input_clip_is_in_the_graph() -> None:
    policy = tiny_policy()
    constants = normalizer_constants(policy)
    module = folded_actor(policy)
    x = (constants.mean + 50.0 * constants.std).astype(np.float32)[None, :]  # z = +50 everywhere
    at_clip = (constants.mean + 10.0 * constants.std).astype(np.float32)[None, :]
    np.testing.assert_allclose(_torch(module, x), _torch(module, at_clip), atol=TOL)
    np.testing.assert_allclose(_torch(module, x), reference_actions(policy, x), atol=TOL)


def test_the_output_clip_is_in_the_graph() -> None:
    policy = tiny_policy(bias=25.0)
    module = folded_actor(policy)
    x, _, _ = parity_draw(normalizer_constants(policy), seed=1)
    out = _torch(module, x)
    assert np.all(out <= 1.0) and np.isclose(out.max(), 1.0)
    np.testing.assert_allclose(out, reference_actions(policy, x), atol=TOL)


def test_folding_never_mutates_the_policy() -> None:
    policy = tiny_policy()
    before = {k: v.clone() for k, v in policy.model.policy.state_dict().items()}
    module = folded_actor(policy)
    module.to("cpu").float()
    for k, v in policy.model.policy.state_dict().items():
        assert torch.equal(v, before[k])
    assert policy.normalizer is not None and not policy.normalizer.training


def test_frozen_normalizer_required() -> None:
    policy = tiny_policy()
    assert policy.normalizer is not None
    policy.normalizer.training = True
    with pytest.raises(ValueError, match="frozen"):
        folded_actor(policy)


def test_export_is_batch_dynamic_only_and_matches(tmp_path: Path) -> None:
    policy = tiny_policy(dim=31)
    constants = normalizer_constants(policy)
    path = export_folded(folded_actor(policy), constants.dim, tmp_path / "g.onnx")
    assert graph_dims(path) == (("batch", 31), ("batch", 3))
    session = cpu_session(path)
    x, _, _ = parity_draw(constants, seed=3)
    ref = reference_actions(policy, x)
    for batch in (1, 7, 32, 1000):
        (out,) = session.run([], {"input": np.ascontiguousarray(x[:batch])})
        assert out.shape == (batch, 3)
        np.testing.assert_allclose(out, ref[:batch], rtol=0, atol=TOL)
    with pytest.raises(Exception):  # noqa: B017  (ORT raises its own error type on a bad width)
        session.run([], {"input": np.zeros((1, 25), dtype=np.float32)})


def test_export_is_byte_reproducible(tmp_path: Path) -> None:
    policy = tiny_policy()
    dim = normalizer_constants(policy).dim
    a = export_folded(folded_actor(policy), dim, tmp_path / "a.onnx").read_bytes()
    b = export_folded(folded_actor(policy), dim, tmp_path / "b.onnx").read_bytes()
    assert a == b


def test_exported_pairs_are_p8d1s() -> None:
    assert EXPORTS == (
        ("ppo", 0),
        ("ppo", 4),
        ("residual_ppo_forecast", 0),
        ("residual_ppo_forecast", 4),
    )
    assert graph_name("ppo", 4) == "ppo_s4.onnx"
