"""Phase 5 engineering: reset prefetch is bit-identical to the serial path.

A :class:`~rld.rl.wrappers.PoolSamplingEnv` with a standby environment prepares each next
episode in a background thread. Nothing it produces may differ from the serial wrapper: the
same realizations and episode seeds, and ``np.array_equal`` observations, identical rewards,
flags and infos, at every step. Both wrappers here run on the real
:class:`~rld.envs.landing_env.DeckLandingAviary` and are put through the same script:

* an explicit ``reset(seed=...)`` at the start and another mid-episode;
* random actions (std 0.37, so episodes end within tens of steps and there are many resets);
* curriculum stage changes mid-episode, while a prefetched episode drawn at the old stage is
  pending -- including a change and a change back before the next reset;
* a switch to queue mode (the evaluator's path), flown past the end of the queue into the
  idle replays.

Then the same comparison through ``SubprocVecEnv`` workers, where the background thread runs
while the worker waits on its pipe and the stage change arrives as an ``env_method``
broadcast.

Units: steps are env control steps (1/30 s model scale each).
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

from _rl_helpers import tiny_config
from conftest import BASE_SEED
from rld.control.tuning import TuningEpisode
from rld.rl.train import build_env_configs, training_pools
from rld.rl.wrappers import EnvFactory, PoolSamplingEnv

pytestmark = [pytest.mark.pybullet, pytest.mark.slow]

ACTION_STD = 0.37


def _same(a: Any, b: Any, path: str = "") -> None:
    """Assert two outputs are identical: arrays bit-for-bit, NaN equal to NaN."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{path}: keys differ"
        for key in a:
            _same(a[key], b[key], f"{path}.{key}")
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
        for f in dataclasses.fields(a):
            _same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype, f"{path}: dtype differs"
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), f"{path}: arrays differ"
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), f"{path}: {a!r} != {b!r}"
    elif isinstance(a, list | tuple):
        assert type(a) is type(b) and len(a) == len(b), f"{path}: sequences differ"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{path}[{i}]")
    else:
        assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
        assert bool(a == b), f"{path}: {a!r} != {b!r}"


@pytest.fixture(scope="module")
def pools() -> tuple[Any, Any]:
    cfg = tiny_config("ppo")
    cfgs = build_env_configs(cfg)
    train, _ = training_pools(cfg, cfgs)
    return cfgs, tuple(train)


def _factory(pools: tuple[Any, Any], prefetch: bool, rank: int = 0) -> EnvFactory:
    cfgs, train = pools
    return EnvFactory(cfgs, train, "aft", BASE_SEED, rank, "SS3", None, prefetch)


def _queue(train: tuple[Any, ...]) -> list[TuningEpisode]:
    """Three fixed episodes on train-pool realizations (queue mode needs pool members)."""
    specs = [s for s in train if s.sea_state == "SS4"][:2] + [
        s for s in train if s.sea_state == "SS5"
    ][:1]
    return [
        TuningEpisode(
            s.sea_state, i, "aft", s.vessel, s.heading_deg, s.speed_kn, s.seed, 1000 + 7 * i
        )
        for i, s in enumerate(specs)
    ]


def _script(env: PoolSamplingEnv, train: tuple[Any, ...]) -> list[Any]:
    """Drive one wrapper through the fixed script; return everything it produced."""
    rng = np.random.default_rng(BASE_SEED)
    events: dict[int, Callable[[], Any]] = {
        250: lambda: env.set_stage("SS4"),
        520: lambda: env.set_stage("SS5"),
        521: lambda: env.set_stage("SS4"),
        700: lambda: env.reset(seed=BASE_SEED + 9),
        900: lambda: env.set_stage("SS3"),
        1100: lambda: env.load_episode_queue(_queue(train)),
    }
    trace: list[Any] = [("reset", env.reset(seed=BASE_SEED + 1))]
    for t in range(1500):
        if t in events:
            out = events[t]()
            if out is not None:
                trace.append(("event", t, out))
            if t == 1100:  # queue mode starts with the next reset
                trace.append(("reset", env.reset()))
        action = np.clip(rng.normal(0.0, ACTION_STD, size=3), -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append(("step", t, obs, reward, terminated, truncated, info))
        if terminated or truncated:
            trace.append(("reset", env.reset()))
    return trace


def test_prefetch_is_bit_identical_to_serial(pools: tuple[Any, Any]) -> None:
    _, train = pools
    serial = _factory(pools, prefetch=False)()
    fetched = _factory(pools, prefetch=True)()
    assert isinstance(serial, PoolSamplingEnv) and isinstance(fetched, PoolSamplingEnv)
    assert not serial.prefetching and fetched.prefetching
    try:
        a = _script(serial, train)
        b = _script(fetched, train)
    finally:
        serial.close()
        fetched.close()
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        _same(x, y, f"trace[{i}]")

    resets = [e for e in a if e[0] == "reset"]
    terminal = [e[6] for e in a if e[0] == "step" and (e[4] or e[5])]
    # The script really exercised what it claims to.
    assert len(resets) >= 8
    assert {i["ss"] for i in terminal} >= {"SS3", "SS4", "SS5"}
    assert any(i["eval_idle"] for i in terminal) and any(i["eval_index"] == 2 for i in terminal)
    assert fetched.prefetch_counts["used"] >= 5
    assert fetched.prefetch_counts["discarded"] >= 3  # stage change, explicit seed, queue
    assert serial.prefetch_counts == {"used": 0, "discarded": 0}


def test_prefetch_matches_serial_through_subproc_workers(pools: tuple[Any, Any]) -> None:
    def run(prefetch: bool) -> list[Any]:
        venv = SubprocVecEnv([_factory(pools, prefetch, rank) for rank in range(2)])
        rng = np.random.default_rng(BASE_SEED + 3)
        try:
            venv.seed(BASE_SEED)
            trace: list[Any] = [venv.reset()]
            for t in range(300):
                if t == 150:
                    venv.env_method("set_stage", "SS5")
                action = np.clip(rng.normal(0.0, ACTION_STD, size=(2, 3)), -1, 1)
                obs, reward, dones, infos = venv.step(action.astype(np.float32))
                trace.append((obs, reward, dones, [dict(i) for i in infos]))
            counts = venv.get_attr("prefetch_counts")
        finally:
            venv.close()
        trace.append(counts if prefetch else None)
        return trace

    serial, fetched = run(False), run(True)
    counts = fetched.pop()
    serial.pop()
    for i, (x, y) in enumerate(zip(serial, fetched, strict=True)):
        _same(x, y, f"step[{i}]")
    assert all(c["used"] >= 2 for c in counts), counts
    ended = [i for _, _, d, infos in serial[1:] for k, i in enumerate(infos) if d[k]]
    assert {i["ss"] for i in ended} == {"SS3", "SS5"}
