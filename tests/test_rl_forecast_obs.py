"""Phase 6: the forecast observation block -- training/evaluation parity, causality, shape.

* **Parity.** The training-side :class:`~rld.rl.forecast_obs.ForecastObsWrapper` (inside
  :class:`~rld.rl.wrappers.EnvFactory`, reset prefetch **on**, queue mode) and the
  evaluation-side :class:`~rld.rl.forecast_obs.ForecastPolicy` (flown through
  :mod:`rld.eval.runner`, which builds and advances the feed) give the network the same
  31 numbers at every step of the same episodes, and so the same actions.
* **Causality.** Perturbing the ship's motion after the feed's clock leaves the block
  unchanged.
* **Shape and range.** 25 + 6 = 31 entries, finite, inside the observation box.
* The forecaster is the one ``gated_forecast`` uses, on one ORT thread.

These tests need the fitted ``residual_interval`` under ``artifacts/dmf/`` (``make
dmf-forecasters``, gitignored) and are skipped without it, like the forecast-gated
controller's own tests. Reading ``results/episodes/`` here, in a test, is allowed.

Units: pad ``z`` metres model scale, pad ``v_z`` metres per second model scale; leads seconds
full scale; clocks seconds model scale.
"""

import dataclasses
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _rl_helpers import fresh_run_dir, phase6_config
from conftest import REPO_ROOT
from rld.control.config import CONTROL_CONFIG_DIR, load_gated_forecast
from rld.control.tuning import TuningEpisode
from rld.deck.forecast import DEFAULT_MODEL_ROOT, ShipMotionFeed
from rld.eval.envs import EvalConfigs, load_eval_configs, motion_for
from rld.eval.episodes import ListedEpisode, read_list
from rld.eval.runner import EPISODE_COLUMNS, callable_spec, controller_spec, run_list
from rld.rl.forecast_obs import (
    FORECASTER_ROOT,
    ForecastPolicy,
    ResidualForecastPolicy,
    forecast_block,
    forecaster_dir,
    load_forecaster,
)
from rld.rl.policy import LearnedPolicy
from rld.rl.train import build_policy, load_policy, run_needs_motion_feed
from rld.rl.wrappers import PREPARED_INFO_KEY, EnvFactory

MODEL_DIR = forecaster_dir("residual_interval")
ID_LIST = REPO_ROOT / "results" / "episodes" / "id.parquet"

needs_forecaster = pytest.mark.skipif(
    not (MODEL_DIR / "meta.json").is_file(),
    reason=f"no fitted residual_interval under {MODEL_DIR} (make dmf-forecasters)",
)


def test_forecaster_is_gated_forecasts() -> None:
    gated = load_gated_forecast(CONTROL_CONFIG_DIR / "gated_forecast.yaml")
    assert forecaster_dir("residual_interval") == gated.forecaster_dir
    assert FORECASTER_ROOT.resolve() == DEFAULT_MODEL_ROOT.resolve()


def _first_per_ss(sea_states: tuple[str, ...], n: int) -> list[ListedEpisode]:
    rows = read_list(ID_LIST)
    out: list[ListedEpisode] = []
    for ss in sea_states:
        out += [r for r in rows if r.ss == ss and r.pad == "aft"][:n]
    return out


class _InputRecorder:
    """Keep the network input and the action of every ``act`` (in-process runner)."""

    def __init__(self, inner: Any, sink: list[list[tuple[np.ndarray, np.ndarray]]]) -> None:
        self.inner = inner
        self.sink = sink
        self.privileged = inner.privileged
        self.needs_motion_feed = inner.needs_motion_feed

    def reset(self, seed: int, context: Any = None, motion_feed: Any = None) -> None:
        self.inner.reset(seed, context, motion_feed)
        self.sink.append([])

    def act(self, obs: Any) -> Any:
        x = np.array(self.inner.policy_input(obs), copy=True)
        action = self.inner.act(obs)
        self.sink[-1].append((x, np.array(action, copy=True)))
        return action


_SINKS: dict[str, list[list[tuple[np.ndarray, np.ndarray]]]] = {}


def _recording(name: str, build: Callable[[EvalConfigs], Any], cfgs: EvalConfigs) -> Any:
    return _InputRecorder(build(cfgs), _SINKS.setdefault(name, []))


def _network_action(policy: LearnedPolicy, x: np.ndarray) -> np.ndarray:
    """The policy's action on an already-built network input (what the learner computes)."""
    assert policy.normalizer is not None
    z = np.asarray(policy.normalizer.normalize_obs(x.reshape(1, -1)), dtype=np.float32)
    action, _ = policy.model.predict(z, deterministic=True)
    return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)


@pytest.fixture(scope="module")
def forecast_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not (MODEL_DIR / "meta.json").is_file():
        pytest.skip(f"no fitted residual_interval under {MODEL_DIR}")
    return fresh_run_dir(phase6_config("ppo_forecast"), tmp_path_factory.mktemp("runs"))


@needs_forecaster
@pytest.mark.slow
@pytest.mark.pybullet
def test_training_wrapper_and_eval_policy_see_the_same_31_numbers(forecast_run: Path) -> None:
    _SINKS.clear()
    cfgs = load_eval_configs()
    episodes = _first_per_ss(("SS3", "SS4", "SS5"), 1)
    assert run_needs_motion_feed(forecast_run) is True
    build = functools.partial(build_policy, run_dir=forecast_run, ckpt="final")
    spec = callable_spec(
        "ppo_forecast",
        functools.partial(_recording, "eval", build),
        run_seed=0,
        needs_motion_feed=True,
    )
    rows = run_list(episodes, spec, cfgs, workers=1)
    evaluated = _SINKS["eval"]
    assert len(evaluated) == len(rows) == len(episodes)

    policy = load_policy(forecast_run, cfgs=cfgs)
    assert isinstance(policy, ForecastPolicy) and policy.needs_motion_feed is True
    session = policy.forecaster._session  # one ORT thread per worker
    assert session.get_session_options().intra_op_num_threads == 1
    run_cfg = phase6_config("ppo_forecast")
    queue = [
        TuningEpisode(
            r.ss,
            r.index,
            r.pad,
            r.vessel,
            r.heading_deg,
            r.speed_kn,
            r.realization_seed,
            r.episode_seed,
        )
        for r in episodes
    ]
    specs = tuple(e.realization for e in queue)
    factory = EnvFactory(
        cfgs,
        specs,
        "aft",
        0,
        0,
        None,
        None,
        prefetch=True,
        forecast_obs=run_cfg.forecast_obs,
    )
    env = factory()
    try:
        assert env.observation_space.shape == (31,)
        env.get_wrapper_attr("load_episode_queue")(queue)
        steps = 0
        for listed, expected in zip(episodes, evaluated, strict=True):
            obs, reset_info = env.reset()
            assert PREPARED_INFO_KEY not in reset_info  # consumed by the forecast wrapper
            landing: Any = env.unwrapped
            assert float(landing.record.t0_model_s) == listed.t0_model_s
            trained: list[tuple[np.ndarray, np.ndarray]] = []
            while True:
                assert obs.shape == (31,) and obs.dtype == np.float32
                assert np.all(np.isfinite(obs)) and env.observation_space.contains(obs)
                action = _network_action(policy, obs)
                trained.append((obs.copy(), action))
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            assert len(trained) == len(expected)
            for (x_train, a_train), (x_eval, a_eval) in zip(trained, expected, strict=True):
                assert np.array_equal(x_train, x_eval)
                assert np.array_equal(a_train, a_eval)
            assert info["episode_row"]["outcome"] == rows[episodes.index(listed)]["outcome"]
            steps += len(trained)
        counts = env.get_wrapper_attr("prefetch_counts")
        assert counts["used"] == len(episodes) - 1  # episodes 2 and 3 came from the standby
    finally:
        env.close()
    block = np.stack([x for x, _ in evaluated[0]])[:, 25:]
    assert np.abs(block[:, 0::2]).max() < 1.0 and np.abs(block[:, 1::2]).max() < 2.0
    assert np.ptp(block, axis=0).min() > 0.0  # the block moves with the deck
    print(f"\nforecast parity: {len(episodes)} episodes, {steps} steps, 31 entries identical")


@needs_forecaster
@pytest.mark.slow
@pytest.mark.pybullet
def test_zeroed_residual_forecast_policy_is_pid_feedforward(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    run = fresh_run_dir(phase6_config("residual_ppo_forecast"), tmp_path_factory.mktemp("runs"))
    cfgs = load_eval_configs()
    policy = load_policy(run, cfgs=cfgs)
    assert isinstance(policy, ResidualForecastPolicy)
    assert policy.needs_motion_feed is True and policy.privileged is False
    assert run_needs_motion_feed(run) is True
    episodes = _first_per_ss(("SS5", "SS6"), 1)
    spec = callable_spec(
        "residual_ppo_forecast",
        functools.partial(build_policy, run_dir=run, ckpt="final"),
        run_seed=0,
        needs_motion_feed=True,
    )
    res = run_list(episodes, spec, cfgs, workers=1)
    pid = run_list(episodes, controller_spec("pid_feedforward"), cfgs, workers=1)
    for a, b in zip(pid, res, strict=True):
        for column in EPISODE_COLUMNS:
            if column != "method":
                same = a[column] == b[column] or (a[column] != a[column] and b[column] != b[column])
                assert same, (a["ss"], column, a[column], b[column])


@needs_forecaster
@pytest.mark.slow
def test_forecast_policy_requires_the_feed(forecast_run: Path) -> None:
    policy = load_policy(forecast_run)
    with pytest.raises(ValueError, match="ShipMotionFeed"):
        policy.reset(0)
    with pytest.raises(RuntimeError, match="before reset"):
        policy.act(np.zeros(25, np.float32))


class _FutureTampered:
    """A deck source whose ship channels are perturbed at every time after ``cut_full_s``."""

    def __init__(self, source: Any, cut_full_s: float) -> None:
        self._source = source
        self.cut_full_s = cut_full_s

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def channels(self, t_full_s: Any) -> Any:
        ch = self._source.channels(t_full_s)
        t = np.atleast_1d(np.asarray(t_full_s, dtype=np.float64))
        bump = np.where(t > self.cut_full_s, 1.0, 0.0)
        changed = {
            f.name: getattr(ch, f.name) + bump
            for f in dataclasses.fields(ch)
            if f.name != "t_full_s"
        }
        return dataclasses.replace(ch, **changed)


@needs_forecaster
@pytest.mark.slow
def test_block_is_causal() -> None:
    cfgs = load_eval_configs()
    scale = cfgs.scaling.froude_scale()
    source = motion_for(cfgs, "frigate", "SS5", 180.0, 6.0, 0)
    t0, dt = 20.0, cfgs.landing.ctrl_dt_s
    horizon_steps = 45  # 1.5 s model of control steps
    cut_full = float(source.full_time_s(np.array([t0 + horizon_steps * dt]))[0])
    tampered = _FutureTampered(source, cut_full)
    later = np.array([cut_full + 0.5])
    assert not np.array_equal(
        source.channels(later).heave_m, tampered.channels(later).heave_m
    )  # the future really differs
    forecaster = load_forecaster(MODEL_DIR)
    feeds = [
        ShipMotionFeed.for_pad(src, "aft", cfgs.pads, scale)  # type: ignore[arg-type]
        for src in (source, tampered)
    ]
    for feed in feeds:
        feed.reset(t0)
    for k in range(horizon_steps + 1):
        blocks = []
        for feed in feeds:
            feed.advance_to(t0 + k * dt)
            blocks.append(forecast_block(forecaster, feed, (1.0, 2.0, 3.0)))
        assert blocks[0].shape == (6,) and blocks[0].dtype == np.float32
        assert np.all(np.isfinite(blocks[0]))
        assert np.array_equal(blocks[0], blocks[1]), k
