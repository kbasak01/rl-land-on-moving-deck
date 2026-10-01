"""The forecast observation block: dmf's point pad forecast at 1/2/3 s full-scale lead.

``ppo_forecast`` and ``residual_ppo_forecast`` append six numbers to the 25-entry
environment observation (31 in total):

    [z(1 s), v_z(1 s), z(2 s), v_z(2 s), z(3 s), v_z(3 s)]

-- the point forecast of the pad deck-point height (metres **model** scale, relative to the
ship's mean-position CG height) and vertical velocity (metres per second **model** scale) at
leads of 1, 2 and 3 s **full** scale (0.2, 0.4, 0.6 s model at ``lam = 1/25``), from
:func:`rld.deck.forecast.leads_full_s`. ``VecNormalize`` scales them with the rest of the
observation. The environment and :mod:`rld.envs.observation` are not touched.

The forecaster
--------------
``residual_interval`` (dmf's DLinear-OLS point plus fixed empirical residual quantiles; only
its 0.5-quantile point is read here) -- the model ``configs/control/gated_forecast.yaml``
uses, resolved the same way: ``artifacts/dmf/<name>`` under the repository root
(:func:`forecaster_dir`; a test pins the two paths equal). It runs on ONNX Runtime's CPU
provider with **one** intra-op and one inter-op thread (:func:`load_forecaster`), so a
worker's forecast never spreads over more than the core it steps on. The forecaster was fitted
on the dev pool (P4-D1): the DLinear-OLS point forecast, the only part this block reads, is
fitted on the train pool, so its forecasts are in-sample in training and out-of-sample on the
tune pool and the frozen lists (recorded in P6-D1 and P6-D6).

One block function, two call sites
----------------------------------
:func:`forecast_block` is the only code that turns a feed into the block. It reads a
past-only :class:`~rld.deck.forecast.ShipMotionFeed` (P4-D3: an ideal, noise-free,
zero-latency ship motion reference unit that never returns a sample later than its clock --
an extra sensor the non-forecast methods do not have):

* **training and the tune-pool evaluation** -- :class:`ForecastObsWrapper` takes a fresh
  feed at every reset, built from the active landing env's own source and pad
  (:func:`episode_feed`, the construction :mod:`rld.eval.runner` uses, lever-arm check
  included) and reset to ``t0``, and, before returning each observation, advances it to
  ``t0 + k * ctrl_dt`` where ``k`` is the env's executed control steps -- **the same float
  expression** the runner evaluates before its ``k``-th ``act``. Inside
  :class:`~rld.rl.wrappers.EnvFactory` the feed is built by the pool-sampling env's
  ``prepare_episode`` hook, i.e. on the reset-prefetch thread (its ~80 ms pre-fill would
  otherwise stall every lockstep worker at each reset); the wrapper checks it belongs to the
  env that flies and builds it itself when no prepared feed is handed up;
* **evaluation on the frozen lists** -- :class:`ForecastPolicy` (``needs_motion_feed =
  True``) receives the runner's feed in ``reset``; the runner advances the clock before every
  ``act`` and the policy appends the block itself.

So, episode for episode, the policy sees the same 31 numbers in both paths
(``tests/test_rl_forecast_obs.py``). With prefetch (:class:`~rld.rl.wrappers.PoolSamplingEnv`)
the wrapper reads ``unwrapped`` after the reset, which is the slot that now flies.

The block is clipped to :data:`FORECAST_BLOCK_BOUND` so that the observation box stays
finite (as :func:`rld.envs.observation.observation_space` promises for its own entries); the
bound is ~30x the largest pad excursion in the corpus and never binds in practice.

Units: pad ``z`` metres model scale, pad ``v_z`` metres per second model scale; leads seconds
**full** scale; the clock seconds **model** scale.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from dmf.typedefs import FloatArray
from gymnasium.spaces import Box
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from rld.config import REPO_ROOT
from rld.control.base import Controller, PrivilegedContext
from rld.deck.config import PadConfig
from rld.deck.scaling import FroudeScale
from rld.envs.observation import OBS_DTYPE
from rld.eval.envs import EvalConfigs
from rld.rl.config import ForecastObsConfig
from rld.rl.policy import LearnedPolicy
from rld.rl.residual import ResidualPolicy
from rld.rl.wrappers import PREPARED_INFO_KEY

if TYPE_CHECKING:
    from rld.deck.forecast import OnlineForecaster, ShipMotionFeed

__all__ = [
    "FORECASTER_ROOT",
    "FORECAST_BLOCK_BOUND",
    "ForecastObsWrapper",
    "ForecastPolicy",
    "ResidualForecastPolicy",
    "block_observation_space",
    "episode_feed",
    "forecast_block",
    "forecaster_dir",
    "load_forecaster",
]

#: Where the fitted dmf models live (``rld.deck.forecast.DEFAULT_MODEL_ROOT``, restated here
#: so that importing this module does not import torch; a test pins the two equal).
FORECASTER_ROOT: Path = REPO_ROOT / "artifacts" / "dmf"

#: Clip bound of every block entry, metres (``z``) and metres per second (``v_z``), model
#: scale. Keeps the observation box finite; never reached (module docstring).
FORECAST_BLOCK_BOUND: float = 10.0


def forecaster_dir(name: str) -> Path:
    """Return the fitted model directory of a forecaster name.

    Args:
        name: Artifact name, e.g. ``"residual_interval"``.

    Returns:
        ``<repo>/artifacts/dmf/<name>``, resolved -- the path
        :func:`rld.control.config.load_gated_forecast` resolves ``forecaster_dir`` to.
    """
    return (FORECASTER_ROOT / name).resolve()


def load_forecaster(model_dir: Path) -> "OnlineForecaster":
    """Load a fitted forecaster for the landing loop: ORT, CPU, one thread.

    Args:
        model_dir: The artifact directory; its files are checked against ``meta.json``.

    Returns:
        The :class:`~rld.deck.forecast.OnlineForecaster`.
    """
    from rld.deck.forecast import OnlineForecaster  # heavy (torch, ORT): only where used

    return OnlineForecaster(model_dir, backend="ort", intra_op_threads=1)


def forecast_block(
    forecaster: "OnlineForecaster", feed: "ShipMotionFeed", leads_full_s: Sequence[float]
) -> FloatArray:
    """Return the forecast observation block at the feed's clock -- the one shared function.

    Args:
        forecaster: The fitted forecaster.
        feed: A reset, advanced, past-only ship-motion feed; its clock is not moved.
        leads_full_s: Leads, seconds **full** scale (1, 2, 3).

    Returns:
        ``(2 * len(leads_full_s),)`` float32, ``[z, v_z]`` per lead in lead order, metres
        and metres per second **model** scale (point forecasts), clipped to
        :data:`FORECAST_BLOCK_BOUND`.

    Raises:
        ValueError: If the forecast is not finite or a lead is off the forecast grid.
    """
    from rld.deck.forecast import leads_full_s as point_leads

    block = point_leads(forecaster.forecast(feed), tuple(leads_full_s)).reshape(-1)
    if not np.all(np.isfinite(block)):
        raise ValueError(f"non-finite forecast block {block}")
    out: FloatArray = np.clip(block, -FORECAST_BLOCK_BOUND, FORECAST_BLOCK_BOUND).astype(OBS_DTYPE)
    return out


def block_observation_space(base: Box, block_size: int) -> Box:
    """Return the environment's observation box with the forecast block appended.

    Args:
        base: The environment's ``Box`` (25 entries, float32, finite).
        block_size: Entries appended (6).

    Returns:
        A float32 ``Box`` of ``base.shape[0] + block_size`` entries, the block bounded by
        ``+/- FORECAST_BLOCK_BOUND``.
    """
    bound = np.full(block_size, FORECAST_BLOCK_BOUND, dtype=OBS_DTYPE)
    low = np.concatenate([np.asarray(base.low, dtype=OBS_DTYPE), -bound])
    high = np.concatenate([np.asarray(base.high, dtype=OBS_DTYPE), bound])
    return Box(low=low, high=high, dtype=OBS_DTYPE)


def episode_feed(landing: Any, pads: PadConfig, scale: FroudeScale) -> "ShipMotionFeed":
    """Build the current episode's feed from a landing env just after ``reset``, clock at t0.

    The construction of :mod:`rld.eval.runner` (``_make_feed``): the env's own source and
    effective pad, a lever-arm check against the env, ``reset(t0)``.

    Args:
        landing: The :class:`~rld.envs.landing_env.DeckLandingAviary` that flies the episode
            (``motion``, ``pad``, ``pad_offset_m``, ``record.t0_model_s``).
        pads: Pad geometry.
        scale: Froude scale.

    Returns:
        The feed, clock at ``t0`` (seconds model scale), lookback filled from before it.

    Raises:
        RuntimeError: If the feed's lever arm is not the env's, or its clock is not ``t0``.
    """
    from rld.deck.forecast import ShipMotionFeed

    t0 = float(landing.record.t0_model_s)
    feed = ShipMotionFeed.for_pad(landing.motion, landing.pad, pads, scale)
    arm_model_m = float(scale.length(np.asarray(feed.r_pad_full_m[0], dtype=np.float64)))
    if not np.isclose(arm_model_m, float(landing.pad_offset_m[0]), rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            f"feed lever arm {arm_model_m} m != env pad lever arm {landing.pad_offset_m[0]} m"
        )
    feed.reset(t0)
    if feed.clock_model_s != t0:
        raise RuntimeError(f"feed clock {feed.clock_model_s!r} != t0 {t0!r} after reset")
    return feed


class ForecastObsWrapper(gym.Wrapper[FloatArray, FloatArray, FloatArray, FloatArray]):
    """Training-side forecast block: every observation gains :func:`forecast_block`.

    Wraps the pool-sampling env (or the residual wrapper on it) and sits under ``Monitor``.
    A fresh :class:`~rld.deck.forecast.ShipMotionFeed` per episode, advanced exactly as the
    evaluation runner advances it (module docstring).

    Attributes:
        leads_full_s: Leads, seconds **full** scale.
        forecaster: The worker's forecaster (one per worker, ORT, one thread).
    """

    def __init__(
        self,
        env: gym.Env[FloatArray, FloatArray],
        cfgs: EvalConfigs,
        fc: ForecastObsConfig,
        forecaster: "OnlineForecaster | None" = None,
    ) -> None:
        """Wrap ``env``.

        Args:
            env: The landing env stack (``unwrapped`` must be the landing env that flies).
            cfgs: The committed configs (pads, Froude scale).
            fc: The block's config.
            forecaster: An already-loaded forecaster (tests); loaded from
                ``fc.forecaster`` otherwise.
        """
        super().__init__(env)
        self.leads_full_s = tuple(fc.leads_full_s)
        self.forecaster = (
            forecaster if forecaster is not None else load_forecaster(forecaster_dir(fc.forecaster))
        )
        self._pads = cfgs.pads
        self._scale = cfgs.scaling.froude_scale()
        base_space = env.observation_space
        if not isinstance(base_space, Box):
            raise TypeError(f"expected a Box observation space, got {base_space}")
        self.observation_space = block_observation_space(base_space, fc.block_size)
        self._feed: ShipMotionFeed | None = None
        self._t0 = 0.0

    @property
    def feed(self) -> "ShipMotionFeed":
        """This episode's feed (tests)."""
        if self._feed is None:
            raise RuntimeError("ForecastObsWrapper: no episode started")
        return self._feed

    def _augment(self, obs: FloatArray) -> FloatArray:
        """Advance the feed to the env's control time and append the block.

        Args:
            obs: The environment observation, 25 entries.

        Returns:
            The observation with the block appended, float32.

        Raises:
            RuntimeError: If the feed's clock is not the env's control time.
        """
        landing: Any = self.env.unwrapped
        feed = self.feed
        dt = float(landing.cfg.ctrl_dt_s)
        k = int(landing.record.steps)
        t_ctrl = self._t0 + k * dt
        feed.advance_to(t_ctrl)
        if feed.clock_model_s != t_ctrl:
            raise RuntimeError(f"feed clock {feed.clock_model_s!r} != control time {t_ctrl!r}")
        block = forecast_block(self.forecaster, feed, self.leads_full_s)
        return np.concatenate([np.asarray(obs, dtype=OBS_DTYPE).reshape(-1), block])

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset the env, take (or build) this episode's feed at ``t0``, return 31 entries.

        Args:
            seed: Passed through.
            options: Passed through.

        Returns:
            ``(observation with the block, info)``.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        landing: Any = self.env.unwrapped
        prepared = info.pop(PREPARED_INFO_KEY, None)
        t0 = float(landing.record.t0_model_s)
        if prepared is None:
            prepared = episode_feed(landing, self._pads, self._scale)
        elif prepared.source is not landing.motion or prepared.clock_model_s != t0:
            raise RuntimeError("the prepared ship-motion feed is not the flying episode's")
        self._feed = prepared
        self._t0 = t0
        return self._augment(obs), info

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Step the env and append the block at the new control time.

        Args:
            action: Passed through.

        Returns:
            The env's step with the block appended to the observation.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), float(reward), bool(terminated), bool(truncated), info


class ForecastPolicy(LearnedPolicy):
    """Evaluation-side forecast block: ``pi`` reads ``[o, forecast_block(feed)]``.

    ``needs_motion_feed = True``: the runner builds, resets and advances the episode's
    past-only feed and hands it to :meth:`reset`. Not privileged.

    Attributes:
        forecaster_dir: The fitted model directory.
        leads_full_s: Leads, seconds **full** scale.
    """

    needs_motion_feed: bool = True

    def __init__(
        self,
        name: str,
        model: BaseAlgorithm,
        normalizer: VecNormalize | None,
        run_dir: Path,
        checkpoint: Path,
        steps: int,
        *,
        forecaster_dir: Path,
        leads_full_s: Sequence[float],
        forecaster: "OnlineForecaster | None" = None,
    ) -> None:
        """Hold the model, its frozen normaliser and the forecaster's location.

        Args:
            name: Method label.
            model: The SB3 model (CPU); its input is 25 + 2 * len(leads) entries.
            normalizer: Frozen normaliser, or ``None``.
            run_dir: The run directory.
            checkpoint: The checkpoint directory.
            steps: Training env steps at the checkpoint.
            forecaster_dir: The fitted model directory (loaded at the first ``reset``).
            leads_full_s: Leads, seconds **full** scale.
            forecaster: An already-loaded forecaster (tests).
        """
        super().__init__(name, model, normalizer, run_dir, checkpoint, steps)
        self._setup_forecast(forecaster_dir, leads_full_s, forecaster)

    def _setup_forecast(
        self,
        model_dir: Path,
        leads_full_s: Sequence[float],
        forecaster: "OnlineForecaster | None",
    ) -> None:
        """Attach the forecaster's location (shared with the combined class)."""
        self.forecaster_dir = Path(model_dir)
        self.leads_full_s = tuple(float(x) for x in leads_full_s)
        self._forecaster = forecaster
        self._feed: ShipMotionFeed | None = None

    @property
    def forecaster(self) -> "OnlineForecaster":
        """The forecaster, loaded on first use (ORT, CPU, one thread)."""
        if self._forecaster is None:
            self._forecaster = load_forecaster(self.forecaster_dir)
        return self._forecaster

    def reset(
        self,
        seed: int,
        context: PrivilegedContext | None = None,
        motion_feed: "ShipMotionFeed | None" = None,
    ) -> None:
        """Take this episode's feed (required), then the parent's reset without it.

        Args:
            seed: The episode seed.
            context: Must be ``None``.
            motion_feed: The episode's feed, reset to ``t0``. Required.

        Raises:
            ValueError: If no feed is passed, or a privileged context is.
        """
        if motion_feed is None:
            raise ValueError(
                f"{self.name} reads the forecast block and needs a ShipMotionFeed at every "
                "reset (needs_motion_feed=True)"
            )
        super().reset(seed, context, None)
        _ = self.forecaster  # load before the first act, not inside the timed loop
        self._feed = motion_feed

    def policy_input(self, obs: FloatArray) -> FloatArray:
        """Return ``[obs, forecast_block(feed)]``, float32, at the feed's (runner's) clock.

        Args:
            obs: The environment observation, 25 entries.

        Returns:
            ``(25 + 2 * len(leads),)`` float32.

        Raises:
            RuntimeError: Before :meth:`reset`.
        """
        if self._feed is None:
            raise RuntimeError(f"{self.name}: act() before reset(motion_feed=...)")
        block = forecast_block(self.forecaster, self._feed, self.leads_full_s)
        return np.concatenate([super().policy_input(obs), block])


class ResidualForecastPolicy(ForecastPolicy, ResidualPolicy):
    """``residual_ppo_forecast`` at evaluation: residual action, forecast-block input.

    ``act`` is :meth:`ResidualPolicy.act` (the base reads the raw 25 entries); the network's
    input is :meth:`ForecastPolicy.policy_input`. ``needs_motion_feed = True``.
    """

    def __init__(
        self,
        name: str,
        model: BaseAlgorithm,
        normalizer: VecNormalize | None,
        run_dir: Path,
        checkpoint: Path,
        steps: int,
        *,
        base: Controller,
        alpha: float,
        forecaster_dir: Path,
        leads_full_s: Sequence[float],
        forecaster: "OnlineForecaster | None" = None,
    ) -> None:
        """Hold the model, the base and the forecaster's location.

        Args:
            name: Method label.
            model: The SB3 PPO model (CPU).
            normalizer: Frozen normaliser, or ``None``.
            run_dir: The run directory.
            checkpoint: The checkpoint directory.
            steps: Training env steps at the checkpoint.
            base: The residual base controller.
            alpha: Residual scale, normalised units.
            forecaster_dir: The fitted model directory.
            leads_full_s: Leads, seconds **full** scale.
            forecaster: An already-loaded forecaster (tests).
        """
        LearnedPolicy.__init__(self, name, model, normalizer, run_dir, checkpoint, steps)
        self._setup_residual(base, alpha)
        self._setup_forecast(forecaster_dir, leads_full_s, forecaster)
