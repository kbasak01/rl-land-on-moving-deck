"""Phase 5 engineering: the independent evaluator reproduces the lockstep evaluator exactly.

:meth:`TunePoolEvaluator.evaluate` ships the policy and the frozen observation statistics to
each worker, which flies its share of the draw on its own. It must return the same episode
rows as :meth:`TunePoolEvaluator.evaluate_lockstep` (the evaluator the P5-D4 smoke run used)
-- every field, bit for bit -- with and without reset prefetch, in worker processes and
in-process. The policy is a PPO network with a randomised action head and the
normaliser has non-trivial statistics, so the episodes end in different ways and any
difference in the actions would show.

Units: steps are env control steps (1/30 s model scale each).
"""

from typing import Any

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from _rl_helpers import tiny_config
from conftest import BASE_SEED
from rld.rl.callbacks import TunePoolEvaluator, frozen_normalizer
from rld.rl.eval_workers import (
    normalizer_from_state,
    normalizer_state,
    policy_from_payload,
    policy_payload,
    split_shares,
)
from rld.rl.train import build_env_configs, eval_episodes, training_pools
from rld.rl.wrappers import EnvFactory
from test_rl_prefetch import _same

N_WORKERS = 3


@pytest.fixture(scope="module")
def setup() -> dict[str, Any]:
    cfg = tiny_config("ppo", eval__episodes_per_ss=3)
    cfgs = build_env_configs(cfg)
    _, tune = training_pools(cfg, cfgs)
    episodes = eval_episodes(cfg, cfgs)
    factory = EnvFactory(cfgs, tuple(tune), "aft", BASE_SEED, 0, None, None)
    probe = DummyVecEnv([factory])
    model = PPO(
        "MlpPolicy",
        probe,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=BASE_SEED,
        device="cpu",
    )
    gen = torch.Generator().manual_seed(BASE_SEED)
    with torch.no_grad():
        head = model.policy.action_net  # type: ignore[union-attr]
        head.weight.copy_(0.05 * torch.randn(head.weight.shape, generator=gen))
        head.bias.copy_(torch.tensor([0.0, 0.0, -0.6]))  # descend: touchdowns
    norm = VecNormalize(probe, norm_obs=True, norm_reward=True)
    rng = np.random.default_rng(BASE_SEED)
    norm.obs_rms.mean = rng.normal(0.0, 0.1, size=norm.obs_rms.mean.shape)
    norm.obs_rms.var = rng.uniform(0.5, 2.0, size=norm.obs_rms.var.shape)
    return {"cfgs": cfgs, "tune": tuple(tune), "episodes": episodes, "model": model, "norm": norm}


def _evaluator(setup: dict[str, Any], vec_env: str, prefetch: bool) -> TunePoolEvaluator:
    factories = [
        EnvFactory(setup["cfgs"], setup["tune"], "aft", BASE_SEED, rank, None, None, prefetch)
        for rank in range(N_WORKERS)
    ]
    return TunePoolEvaluator(factories, setup["episodes"], vec_env)  # type: ignore[arg-type]


def test_payloads_round_trip_exactly(setup: dict[str, Any]) -> None:
    model, norm = setup["model"], setup["norm"]
    policy = policy_from_payload(policy_payload(model.policy))
    for a, b in zip(model.policy.state_dict().values(), policy.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    frozen = normalizer_from_state(normalizer_state(norm))
    assert frozen.training is False and frozen.norm_reward is False
    ref = frozen_normalizer(norm)
    obs = np.random.default_rng(1).normal(size=(5, norm.obs_rms.mean.size)).astype(np.float32)
    assert np.array_equal(frozen.normalize_obs(obs), ref.normalize_obs(obs))
    x = np.asarray(frozen.normalize_obs(obs), dtype=np.float32)
    shipped, _ = policy.predict(x, deterministic=True)
    original, _ = model.predict(x, deterministic=True)
    assert np.array_equal(shipped, original)
    shares = split_shares(setup["episodes"], N_WORKERS)
    assert [e for s in shares for e in s] != [] and sum(map(len, shares)) == 9


@pytest.mark.pybullet
@pytest.mark.slow
@pytest.mark.parametrize(("vec_env", "prefetch"), [("subproc", True), ("subproc", False)])
def test_independent_evaluation_equals_lockstep(
    setup: dict[str, Any], vec_env: str, prefetch: bool
) -> None:
    evaluator = _evaluator(setup, vec_env, prefetch)
    try:
        rows, steps = evaluator.evaluate(setup["model"], setup["norm"])
        ref, ref_steps = evaluator.evaluate_lockstep(setup["model"], setup["norm"])
        again, _ = evaluator.evaluate(setup["model"], setup["norm"])  # workers reused
    finally:
        evaluator.close()
    assert len(rows) == len(ref) == 9
    _same(rows, ref, "rows")
    _same(again, ref, "again")
    assert steps == sum(int(r["steps"]) for r in rows) <= ref_steps
    # Discriminating: the episodes did not all end the same way.
    assert len({round(float(r["return"]), 6) for r in rows}) > 3
    assert len({r["outcome"] for r in rows}) >= 3, [r["outcome"] for r in rows]
    assert any(r["touchdown_contact"] for r in rows)


@pytest.mark.pybullet
@pytest.mark.slow
def test_in_process_workers_equal_lockstep(setup: dict[str, Any]) -> None:
    inproc = _evaluator(setup, "dummy", True)
    threads = torch.get_num_threads()
    try:
        rows, _ = inproc.evaluate(setup["model"], setup["norm"])
        ref, _ = inproc.evaluate_lockstep(setup["model"], setup["norm"])
    finally:
        inproc.close()
    assert torch.get_num_threads() == threads  # an in-process worker leaves torch alone
    _same(rows, ref, "rows")
