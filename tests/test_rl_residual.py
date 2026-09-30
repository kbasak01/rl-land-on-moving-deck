"""Phase 6: residual RL -- the composition, the zero initialisation, and the Gate 6 identity.

**Gate test.** A freshly built ``residual_ppo`` policy (the committed config, its
``action_net`` zero-initialised exactly as a training run builds it) flown through
:mod:`rld.eval.runner` on a sample of the frozen ``id`` list gives episode rows identical to
``pid_feedforward``'s in every column but ``method``, and bit-identical per-step actions. The
training-side :class:`~rld.rl.residual.ResidualActionWrapper`, driven by the same zeroed
network, reproduces the same actions and the same episode records. The test needs no
artifact outside the repository and is never skipped. (Reading ``results/episodes/`` here, in
a test, is allowed; ``rld.rl`` itself never does, ``tests/test_rl_leakage.py``.)

Units: actions in normalised units (``v_max`` = 1.5 m/s model scale); times seconds model scale.
"""

import functools
import math
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from _rl_helpers import FakeLanding, fresh_run_dir, phase6_config, tiny_config, tiny_raw
from conftest import REPO_ROOT
from rld.eval.envs import EvalConfigs, load_eval_configs, make_env, motion_for, pad_offset_for
from rld.eval.episodes import ListedEpisode, read_list
from rld.eval.runner import (
    EPISODE_COLUMNS,
    RECORD_COLUMNS,
    PolicySpec,
    callable_spec,
    controller_spec,
    run_list,
)
from rld.rl.config import train_config_from_dict
from rld.rl.residual import (
    ResidualActionWrapper,
    ResidualPolicy,
    build_base_controller,
    compose_residual,
    zero_init_action_net,
)
from rld.rl.train import (
    _build_model,
    build_policy,
    load_policy,
    run_needs_motion_feed,
)

ID_LIST = REPO_ROOT / "results" / "episodes" / "id.parquet"

#: Episodes per sea state in the gate sample (first ``n`` of each ``id`` cell, SS3-SS6).
PER_SS = 3

#: ``pid_feedforward``'s committed e01 rows, read to add its first ``bounce`` and first
#: ``hard_landing`` at ``id`` SS6 to the sample, so the identity covers loss classes too.
E01_EPISODES = REPO_ROOT / "results" / "e01" / "episodes.csv"

# --------------------------------------------------------------------------- pure pieces


def test_compose_residual_is_the_base_at_zero_and_clips() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        base = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        out = compose_residual(base, np.zeros(3, np.float32), 0.3)
        assert out.dtype == np.float32 and np.array_equal(out, base)
    out = compose_residual(np.array([0.9, -0.9, 0.0]), np.array([1.0, -1.0, 0.5]), 0.3)
    assert np.array_equal(out, np.array([1.0, -1.0, 0.15], dtype=np.float32))
    for bad in (np.array([np.nan, 0.0, 0.0]), np.zeros(2)):
        with pytest.raises(ValueError):
            compose_residual(np.zeros(3), bad, 0.3)


def test_zero_init_touches_only_the_action_net(tmp_path: Path) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = tiny_config("ppo", residual={"alpha": 0.3, "base": "pid_feedforward"})
    plain = _build_model(tiny_config("ppo"), DummyVecEnv([FakeLanding]), 0, tmp_path / "a")
    model = _build_model(cfg, DummyVecEnv([FakeLanding]), 0, tmp_path / "b")
    policy: Any = model.policy
    assert torch.count_nonzero(policy.action_net.weight) == 0
    assert torch.count_nonzero(policy.action_net.bias) == 0
    # Everything else is the seeded initialisation a pure PPO run gets.
    ref = plain.policy.state_dict()
    for name, value in policy.state_dict().items():
        if not name.startswith("action_net."):
            assert torch.equal(value, ref[name]), name
    assert np.allclose(policy.log_std.detach().numpy(), cfg.ppo.log_std_init)  # type: ignore[union-attr]
    obs = np.random.default_rng(1).uniform(-1, 1, (5, 3)).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert np.array_equal(action, np.zeros((5, 3), np.float32))
    zero_init_action_net(plain)  # idempotent helper works on any PPO model
    assert torch.count_nonzero(plain.policy.action_net.weight) == 0  # type: ignore[union-attr,arg-type]


class _StubBase:
    name = "stub"
    privileged = False
    needs_motion_feed = False

    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.calls = 0

    def reset(self, seed: int, context: Any = None, motion_feed: Any = None) -> None:
        del context, motion_feed
        self.seeds.append(seed)
        self.calls = 0

    def act(self, obs: Any) -> Any:
        del obs
        self.calls += 1
        return np.array([0.5, -0.25, 0.1], dtype=np.float32)


def test_residual_wrapper_composes_and_resets_the_base() -> None:
    base = _StubBase()
    env = ResidualActionWrapper(FakeLanding(), base, 0.3)  # type: ignore[arg-type]
    env.reset(seed=11)
    env.step(np.array([1.0, 1.0, -1.0], dtype=np.float32))
    assert np.array_equal(env.last_action, np.array([0.8, 0.05, -0.2], dtype=np.float32))
    assert np.array_equal(env.last_base_action, np.array([0.5, -0.25, 0.1], np.float32))
    env.reset(seed=12)
    assert base.seeds == [11, 12] and base.calls == 0


def test_residual_config_is_checked() -> None:
    raw = tiny_raw("ppo")
    ok = {**raw, "residual": {"alpha": 0.3, "base": "pid_feedforward"}}
    assert train_config_from_dict(ok).residual is not None
    bad = [
        ({"alpha": 0.0, "base": "pid_feedforward"}, "alpha"),
        ({"alpha": 0.3, "base": "oracle_gated"}, "privileged"),
        ({"alpha": 0.3, "base": "gated_forecast"}, "motion feed"),
        ({"alpha": 0.3, "base": "nope"}, "not a registered"),
        ({"alpha": 0.3}, "missing keys"),
    ]
    for block, message in bad:
        with pytest.raises(ValueError, match=message):
            train_config_from_dict({**raw, "residual": block})
    sac = tiny_raw("sac")
    with pytest.raises(ValueError, match="PPO-only"):
        train_config_from_dict({**sac, "residual": {"alpha": 0.3, "base": "pid_feedforward"}})


# --------------------------------------------------------------------------- the gate


class _Recorder:
    """Wrap a runner policy and keep every action it returns, per episode (in-process)."""

    def __init__(self, inner: Any, sink: list[list[np.ndarray]]) -> None:
        self.inner = inner
        self.sink = sink
        self.privileged = inner.privileged
        self.needs_motion_feed = inner.needs_motion_feed

    def reset(self, seed: int, context: Any = None, motion_feed: Any = None) -> None:
        if motion_feed is None:
            self.inner.reset(seed, context)
        else:
            self.inner.reset(seed, context, motion_feed)
        self.sink.append([])

    def act(self, obs: Any) -> Any:
        action = self.inner.act(obs)
        self.sink[-1].append(np.array(action, copy=True))
        return action


_SINKS: dict[str, list[list[np.ndarray]]] = {}


def _recording(name: str, build: Callable[[EvalConfigs], Any], cfgs: EvalConfigs) -> Any:
    return _Recorder(build(cfgs), _SINKS.setdefault(name, []))


def _pid_build(cfgs: EvalConfigs) -> Any:
    return controller_spec("pid_feedforward").build(cfgs)


def _sample() -> list[ListedEpisode]:
    import csv

    rows = read_list(ID_LIST)
    out: list[ListedEpisode] = []
    for ss in ("SS3", "SS4", "SS5", "SS6"):
        cell = [r for r in rows if r.ss == ss and r.pad == "aft"]
        assert len(cell) == 200
        out += cell[:PER_SS]
    with E01_EPISODES.open(newline="") as handle:
        losses = [
            r
            for r in csv.DictReader(handle)
            if r["method"] == "pid_feedforward"
            and (r["regime"], r["ss"], r["pad"]) == ("id", "SS6", "aft")
            and r["outcome"] != "success"
        ]
    for outcome in ("bounce", "hard_landing"):
        index = int(next(r["index"] for r in losses if r["outcome"] == outcome))
        out.append(next(r for r in rows if r.ss == "SS6" and r.index == index))
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


@pytest.fixture(scope="module")
def residual_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fresh_run_dir(phase6_config("residual_ppo"), tmp_path_factory.mktemp("runs"))


@pytest.mark.slow
@pytest.mark.pybullet
def test_gate_zeroed_residual_is_pid_feedforward(residual_run: Path) -> None:
    """Gate 6: zeroed residual through the runner == pid_feedforward, row for row."""
    _SINKS.clear()
    cfgs = load_eval_configs()
    episodes = _sample()
    assert len(episodes) == 4 * PER_SS + 2
    reference = controller_spec("pid_feedforward")
    pid_spec = PolicySpec(
        method=reference.method,
        privileged=reference.privileged,
        build=functools.partial(_recording, "pid", _pid_build),
        run_seed=reference.run_seed,
        needs_motion_feed=reference.needs_motion_feed,
    )
    res_build = functools.partial(build_policy, run_dir=residual_run, ckpt="final")
    pickle.dumps(res_build)  # callable_spec ships it to spawned workers
    assert run_needs_motion_feed(residual_run) is False
    res_spec = callable_spec(
        "residual_ppo", functools.partial(_recording, "res", res_build), run_seed=0
    )
    pid_rows = run_list(episodes, pid_spec, cfgs, workers=1)
    res_rows = run_list(episodes, res_spec, cfgs, workers=1)

    assert len(pid_rows) == len(res_rows) == len(episodes)
    compared = [c for c in EPISODE_COLUMNS if c != "method"]
    for p, r in zip(pid_rows, res_rows, strict=True):
        differing = [c for c in compared if not _same(p[c], r[c])]
        assert not differing, (p["ss"], p["index"], {c: (p[c], r[c]) for c in differing})
        assert r["method"] == "residual_ppo"
    # Not a trivial identity: the sample holds landings and two loss classes.
    assert {row["outcome"] for row in pid_rows} == {"success", "bounce", "hard_landing"}
    pid_actions, res_actions = _SINKS["pid"], _SINKS["res"]
    assert len(pid_actions) == len(res_actions) == len(episodes)
    for a, b in zip(pid_actions, res_actions, strict=True):
        stack_a, stack_b = np.stack(a), np.stack(b)
        assert stack_a.dtype == stack_b.dtype == np.float32
        assert np.array_equal(stack_a, stack_b)
    print(
        f"\ngate: {len(episodes)} id episodes (SS3-SS6 x {PER_SS} + 2 SS6 losses), "
        f"{sum(len(a) for a in pid_actions)} steps, rows and actions identical; outcomes "
        f"{sorted({str(r['outcome']) for r in pid_rows})}"
    )

    # The training path: ResidualActionWrapper driven by the same zeroed network.
    policy = load_policy(residual_run, "final", cfgs=cfgs)
    assert isinstance(policy, ResidualPolicy)
    first = episodes[0]
    env = ResidualActionWrapper(
        make_env(
            cfgs,
            motion_for(
                cfgs,
                first.vessel,
                first.ss,
                first.heading_deg,
                first.speed_kn,
                first.realization_seed,
            ),
            first.vessel,
            first.pad,
            first.episode_seed,
        ),
        build_base_controller("pid_feedforward", cfgs),
        0.3,
    )
    landing: Any = env.unwrapped
    try:
        for listed, expected, row in zip(episodes, pid_actions, pid_rows, strict=True):
            landing.set_motion(
                motion_for(
                    cfgs,
                    listed.vessel,
                    listed.ss,
                    listed.heading_deg,
                    listed.speed_kn,
                    listed.realization_seed,
                ),
                listed.pad,
                pad_offset_for(cfgs, listed.vessel, listed.pad),
            )
            obs, _ = env.reset(seed=listed.episode_seed)
            assert float(landing.record.t0_model_s) == listed.t0_model_s
            executed: list[np.ndarray] = []
            while True:
                a_res = policy.network_action(obs)  # frozen stats, batch 1, deterministic
                assert np.array_equal(a_res, np.zeros(3, np.float32))
                obs, _, terminated, truncated, _ = env.step(a_res)
                assert env.last_action is not None
                executed.append(env.last_action.copy())
                if terminated or truncated:
                    break
            assert np.array_equal(np.stack(executed), np.stack(expected))
            record = landing.record.as_row()
            for column in RECORD_COLUMNS:
                value = record[column]
                value = float("nan") if value is None else value
                if column == "n_contacts" and landing.record.contact_record is None:
                    value = 0
                assert _same(value, row[column]), (listed.ss, listed.index, column)
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.pybullet
def test_residual_policy_contract(residual_run: Path) -> None:
    policy = load_policy(residual_run)
    assert isinstance(policy, ResidualPolicy)
    assert policy.privileged is False and policy.needs_motion_feed is False
    assert policy.alpha == 0.3 and policy.base.name == "pid_feedforward"
    with pytest.raises(ValueError, match="ShipMotionFeed"):
        policy.reset(0, None, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="privileged"):
        policy.reset(0, context=object())  # type: ignore[arg-type]
