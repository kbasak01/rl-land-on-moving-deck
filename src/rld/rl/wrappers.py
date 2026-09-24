"""Pool sampling around the landing environment, and the per-worker environment factory.

:class:`PoolSamplingEnv` wraps one :class:`~rld.envs.landing_env.DeckLandingAviary` and
decides, at every ``reset``, which realization the next episode flies against. It has two
modes:

* **sampling** (training): a realization is drawn uniformly from the pool it was built with
  -- the P3-D2 **train** pool, restricted to the current curriculum sea state -- together
  with a fresh episode seed, from a per-worker :class:`numpy.random.SeedSequence`. The deck
  source is swapped with :meth:`DeckLandingAviary.set_motion` *before* ``reset``, so no
  episode straddles two realizations.
* **queue** (evaluation): :meth:`PoolSamplingEnv.load_episode_queue` hands the worker a list of
  fixed tune-pool episodes (realization + episode seed). Each reset pops the next one. When
  the queue is empty the worker replays its last episode and flags it ``eval_idle`` so the
  evaluator ignores it.

Reset prefetch
--------------
A reset costs about 90 ms against a 1.6 ms step, and a ``SubprocVecEnv`` steps in lockstep,
so every worker waits for whichever one is resetting. With ``standby`` set (a second landing
environment, :attr:`EnvFactory.prefetch`), the wrapper **prefetches**: right after an episode
starts, the next one is drawn and its environment reset on the standby instance in a
background thread, while the worker is idle waiting for actions. The next ``reset`` then
swaps the two instances and returns the prepared observation.

The worker's output is bit-identical to the serial path, because

* the next episode is drawn with exactly the state the serial path would draw it with: the
  sampler RNG (or the evaluation queue) is snapshotted before the early draw, and restored
  -- as if the draw never happened -- whenever the draw could have come out differently:
  the stage changes (:meth:`PoolSamplingEnv.set_stage`), a new queue is loaded, or the reset
  carries an explicit ``seed`` or ``options``;
* a :class:`~rld.envs.landing_env.DeckLandingAviary` reset with an explicit seed does not
  depend on what that instance flew before (P2 reset determinism), so which of the two
  instances flies an episode does not matter; ``tests/test_rl_prefetch.py`` checks the whole
  step sequence with ``np.array_equal`` against the serial path;
* each instance keeps its own deck-motion cache, so the background reset never shares a
  motion object with the episode being flown.

The pool is fixed at construction and every realization in it is checked against
:data:`rld.rl.config.FORBIDDEN_SEA_STATES`, so an SS6 realization cannot even be *held*.
Nothing here reads ``results/episodes/``.

The terminal info
-----------------
``timeout`` stays ``truncated=True`` (the wrapped env decides it; nothing here touches the
flags). At the terminal step the info dict gains flat keys -- the outcome, the realization,
the episode seed, the start state, the closing speed -- which ``Monitor`` logs to
``monitor/*.monitor.csv`` (the training-episode trail the reward-hacking audit reads), and
``episode_row``: ``EpisodeRecord.as_row()`` plus the realization fields, which the tune-pool
evaluator consumes.

Units: lengths metres model scale, speeds metres per second model scale, times seconds model
scale, headings degrees, ship speed knots full scale.
"""

import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from dmf.data.splits import RealizationKey
from dmf.sim.generate import RealizationSpec
from dmf.typedefs import FloatArray

from rld.control.tuning import TuningEpisode
from rld.deck.splits import key_for_spec
from rld.envs.platform import EpisodeMotionSource
from rld.eval.envs import EvalConfigs, make_env, motion_for, pad_offset_for
from rld.rl.config import FORBIDDEN_SEA_STATES

__all__ = [
    "MONITOR_KEYWORDS",
    "TRAIN_SALT",
    "EnvFactory",
    "PoolSamplingEnv",
]

#: Mixed into every training sampler seed sequence, so a training seed ``s`` cannot produce
#: the same draws as the same integer used as an episode or realization seed elsewhere.
TRAIN_SALT: int = 0x7A1B5E

#: Terminal-info keys ``Monitor`` writes to ``monitor/<rank>.monitor.csv``.
MONITOR_KEYWORDS: tuple[str, ...] = (
    "outcome",
    "termination_reason",
    "ss",
    "heading_deg",
    "speed_kn",
    "realization_seed",
    "episode_seed",
    "t0_model_s",
    "init_height_m",
    "init_lateral_m",
    "closing_speed_normal_m_s",
    "max_penetration_m",
    "detectors_disagree",
)


class _LandingLike(Protocol):
    """What the wrapper needs from the wrapped environment (the real one or a test fake)."""

    record: Any
    INIT_XYZS: Any

    def set_motion(
        self,
        motion: EpisodeMotionSource,
        pad: str | None = None,
        pad_offset_m: tuple[float, float, float] | None = None,
    ) -> None:
        """Point the environment at a new deck-motion source before the next reset."""
        ...


@dataclass(frozen=True)
class _Scheduled:
    """One episode to fly: its realization and its episode seed.

    Attributes:
        spec: The dmf realization.
        episode_seed: The environment's episode seed.
        index: Position in the evaluation draw (``-1`` for a training episode).
        idle: A replay of the last queued episode after the queue ran out (evaluation
            only; the evaluator ignores it).
    """

    spec: RealizationSpec
    episode_seed: int
    index: int = -1
    idle: bool = False


@dataclass
class _Prefetch:
    """One episode drawn early and being reset on the standby environment.

    Attributes:
        scheduled: The episode.
        stage: The sampling stage it was drawn at.
        queued: Whether it was drawn in queue mode.
        rng_state: Sampler RNG state *before* the draw (restored on discard).
        queue: Evaluation queue *before* the draw (restored on discard), or ``None``.
        thread: The background reset.
        result: ``(obs, info)`` of the standby reset once the thread has finished.
        error: What the background reset raised, re-raised when the episode is used.
    """

    scheduled: _Scheduled
    stage: str | None
    queued: bool
    rng_state: Mapping[str, Any]
    queue: list[_Scheduled] | None
    thread: threading.Thread | None = None
    result: tuple[FloatArray, dict[str, Any]] | None = None
    error: BaseException | None = field(default=None)


class PoolSamplingEnv(gym.Wrapper[FloatArray, FloatArray, FloatArray, FloatArray]):
    """Draw each episode's realization from a fixed pool, at the current curriculum stage.

    With a ``standby`` environment the next episode is prepared in a background thread
    (module docstring, "Reset prefetch"); the observable behaviour is unchanged.

    Attributes:
        pool: The realizations this worker may fly, grouped by sea state.
        stage: Sea state resets sample from, or ``None`` for all of the pool.
        deck_origin_m: Mean deck point, metres model scale, world frame (for the start
            state's ``init_height_m`` and ``init_lateral_m``).
    """

    def __init__(
        self,
        env: gym.Env[FloatArray, FloatArray],
        pool: Sequence[RealizationSpec],
        motion_factory: Callable[[RealizationSpec], EpisodeMotionSource],
        offset_factory: Callable[[str], tuple[float, float, float]],
        *,
        pad: str,
        seed: int,
        rank: int,
        stage: str | None,
        deck_origin_m: tuple[float, float, float],
        standby: gym.Env[FloatArray, FloatArray] | None = None,
    ) -> None:
        """Wrap ``env``.

        Args:
            env: The landing environment (it must provide ``set_motion`` and ``record``).
            pool: Realizations this worker may fly. Must not contain a forbidden sea state.
            motion_factory: ``spec -> deck-motion source`` (normally
                :func:`rld.eval.envs.motion_for` bound to the configs).
            offset_factory: ``vessel -> pad lever arm``, metres model scale.
            pad: Pad name.
            seed: The training seed.
            rank: This worker's index in its vector env.
            stage: Initial sea state to sample from, or ``None`` for the whole pool.
            deck_origin_m: Mean deck point, metres model scale, world frame.
            standby: A second, identically configured landing environment. Given, the next
                episode is reset on it in a background thread while the current one runs;
                ``None`` keeps every reset serial.

        Raises:
            ValueError: If the pool is empty or holds a forbidden sea state.
        """
        super().__init__(env)
        if not pool:
            raise ValueError("PoolSamplingEnv needs a non-empty pool")
        if bad := sorted({s.sea_state for s in pool} & set(FORBIDDEN_SEA_STATES)):
            raise ValueError(f"the pool holds {bad} realizations; SS6 is never trained on")
        self._landing: _LandingLike = env  # type: ignore[assignment]
        self.pool: dict[str, list[RealizationSpec]] = {}
        for spec in pool:
            self.pool.setdefault(spec.sea_state, []).append(spec)
        self._all = list(pool)
        self._motion_factory = motion_factory
        self._offset_factory = offset_factory
        # One environment instance per slot; slot ``self._active`` flies the current episode.
        self._slots: list[gym.Env[FloatArray, FloatArray]] = [env]
        if standby is not None:
            self._slots.append(standby)
        self._active = 0
        # One deck-motion cache per slot, so the background reset never shares a motion
        # object with the episode in flight.
        self._motions: list[dict[RealizationKey, EpisodeMotionSource]] = [{} for _ in self._slots]
        self._pending: _Prefetch | None = None
        #: Prefetched episodes used, and discarded because the draw could have changed.
        self.prefetch_counts: dict[str, int] = {"used": 0, "discarded": 0}
        self._pad = pad
        self._train_seed = int(seed)
        self._rank = int(rank)
        self._rng = np.random.default_rng(
            np.random.SeedSequence([self._train_seed, rank, TRAIN_SALT])
        )
        self.stage: str | None = None
        self.set_stage(stage)
        self.deck_origin_m = tuple(float(v) for v in deck_origin_m)
        self._queue: list[_Scheduled] | None = None
        self._current: _Scheduled | None = None

    # ------------------------------------------------------------------ control

    def set_stage(self, stage: str | None) -> None:
        """Set the sea state subsequent resets sample from (takes effect at the next reset).

        Args:
            stage: A sea state present in the pool, or ``None`` for the whole pool.

        Raises:
            ValueError: If ``stage`` is forbidden or has no realization in the pool.
        """
        if stage is not None:
            if stage in FORBIDDEN_SEA_STATES:
                raise ValueError(f"stage {stage} is never trained on")
            if stage not in self.pool:
                raise ValueError(f"the pool has no realization at {stage}")
        if self._pending is not None and self._pending.stage != stage:
            self._discard_pending()
        self.stage = stage

    def get_stage(self) -> str | None:
        """Return the sea state resets currently sample from."""
        return self.stage

    def load_episode_queue(self, episodes: Sequence[TuningEpisode]) -> None:
        """Switch to queue mode: the next resets fly exactly these episodes, in order.

        Args:
            episodes: Fixed episodes; each realization must be in this worker's pool.

        Raises:
            ValueError: If an episode's realization is not in the pool.
        """
        keys = {key_for_spec(s) for s in self._all}
        queue: list[_Scheduled] = []
        for ep in episodes:
            spec = ep.realization
            if key_for_spec(spec) not in keys:
                raise ValueError(f"episode realization {key_for_spec(spec)} is not in the pool")
            queue.append(_Scheduled(spec, int(ep.episode_seed), int(ep.index)))
        self._discard_pending()
        self._queue = queue

    def pool_keys(self) -> list[RealizationKey]:
        """Return every realization key this worker may fly, in pool order."""
        return [key_for_spec(s) for s in self._all]

    @property
    def prefetching(self) -> bool:
        """Return whether this wrapper prepares the next episode in the background."""
        return len(self._slots) > 1

    # ------------------------------------------------------------------ gym API

    def _next(self, seed: int | None) -> _Scheduled:
        """Choose the next episode.

        Args:
            seed: A reset seed. In sampling mode it re-seeds the sampler from
                ``(training seed, rank, seed)``; in queue mode it is ignored.

        Returns:
            The episode to fly.
        """
        if self._queue is not None:
            if self._queue:
                return self._queue.pop(0)
            if self._current is None:
                raise RuntimeError("queue mode with an empty queue and no episode flown yet")
            last = self._current
            return _Scheduled(last.spec, last.episode_seed, last.index, idle=True)
        if seed is not None:
            self._rng = np.random.default_rng(
                np.random.SeedSequence([self._train_seed, self._rank, int(seed), TRAIN_SALT])
            )
        candidates = self._all if self.stage is None else self.pool[self.stage]
        spec = candidates[int(self._rng.integers(0, len(candidates)))]
        return _Scheduled(spec, int(self._rng.integers(0, 2**31 - 1)))

    def _motion(self, spec: RealizationSpec, slot: int) -> EpisodeMotionSource:
        """Return the (cached) deck-motion source of one realization for one slot."""
        key = key_for_spec(spec)
        cache = self._motions[slot]
        if key not in cache:
            cache[key] = self._motion_factory(spec)
        return cache[key]

    def _reset_slot(
        self, slot: int, scheduled: _Scheduled, options: dict[str, Any] | None
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Point one slot's environment at an episode and reset it.

        Args:
            slot: Environment slot.
            scheduled: The episode.
            options: Passed through to the environment's ``reset``.

        Returns:
            The environment's ``(observation, info)``.
        """
        env = self._slots[slot]
        landing: _LandingLike = env  # type: ignore[assignment]
        spec = scheduled.spec
        landing.set_motion(self._motion(spec, slot), self._pad, self._offset_factory(spec.vessel))
        return env.reset(seed=scheduled.episode_seed, options=options)

    def _activate(self, slot: int) -> None:
        """Make ``slot`` the environment that steps (and that ``unwrapped`` reaches)."""
        self._active = slot
        self.env = self._slots[slot]
        self._landing = self._slots[slot]  # type: ignore[assignment]

    def _start_prefetch(self) -> None:
        """Draw the next episode now and reset the standby environment in the background.

        Skipped without a standby environment, and in queue mode once the queue is empty
        (an idle replay is only ever needed by a lockstep evaluator and stays serial).
        """
        if not self.prefetching or (self._queue is not None and not self._queue):
            return
        rng_state = copy.deepcopy(self._rng.bit_generator.state)
        queue = None if self._queue is None else list(self._queue)
        pending = _Prefetch(
            scheduled=self._next(None),
            stage=self.stage,
            queued=queue is not None,
            rng_state=rng_state,
            queue=queue,
        )
        slot = 1 - self._active

        def work() -> None:
            try:
                pending.result = self._reset_slot(slot, pending.scheduled, None)
            except BaseException as exc:  # re-raised in the main thread when used
                pending.error = exc

        pending.thread = threading.Thread(target=work, name="rld-reset-prefetch", daemon=True)
        self._pending = pending
        pending.thread.start()

    def _join_pending(self) -> _Prefetch | None:
        """Wait for the background reset, take it out of the wrapper and return it."""
        pending, self._pending = self._pending, None
        if pending is not None and pending.thread is not None:
            pending.thread.join()
        return pending

    def _discard_pending(self) -> None:
        """Drop a prefetched episode and restore the draw state as if it was never drawn."""
        pending = self._join_pending()
        if pending is None:
            return
        self.prefetch_counts["discarded"] += 1
        self._rng.bit_generator.state = pending.rng_state
        if pending.queued:
            self._queue = pending.queue

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Point the environment at the next realization, then reset it.

        Args:
            seed: See :meth:`_next`.
            options: Passed through.

        Returns:
            ``(observation, info)``; ``info`` gains ``ss`` and ``episode_seed``.

        Raises:
            BaseException: Whatever a background reset raised, when its episode is used.
        """
        pending = self._pending
        usable = (
            pending is not None
            and seed is None
            and options is None
            and pending.stage == self.stage
            and pending.queued == (self._queue is not None)
        )
        if usable:
            taken = self._join_pending()
            assert taken is not None
            if taken.error is not None:
                raise taken.error
            assert taken.result is not None
            self._activate(1 - self._active)
            self.prefetch_counts["used"] += 1
            scheduled = taken.scheduled
            obs, info = taken.result
        else:
            self._discard_pending()
            scheduled = self._next(seed)
            obs, info = self._reset_slot(self._active, scheduled, options)
        self._current = scheduled
        info = dict(info)
        info["ss"] = scheduled.spec.sea_state
        info["episode_seed"] = scheduled.episode_seed
        self._start_prefetch()
        return obs, info

    def close(self) -> None:
        """Wait for any background reset, then close every environment instance."""
        self._join_pending()
        for slot, env in enumerate(self._slots):
            if slot != self._active:
                env.close()
        super().close()

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Step the environment; at the terminal step, add the episode's flat record.

        Args:
            action: Normalised action in ``[-1, 1]^3``.

        Returns:
            The wrapped environment's step, unchanged except for extra info keys at the
            terminal step. ``terminated``/``truncated`` pass through untouched.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info)
            info.update(self._terminal_fields(info))
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _terminal_fields(self, info: dict[str, Any]) -> dict[str, Any]:
        """Return the flat per-episode fields added to the terminal info.

        Args:
            info: The wrapped environment's terminal info.

        Returns:
            The :data:`MONITOR_KEYWORDS` fields, ``eval_idle``, ``eval_index`` and
            ``episode_row``.
        """
        assert self._current is not None
        spec = self._current.spec
        row: dict[str, Any] = dict(self._landing.record.as_row())
        init = np.asarray(self._landing.INIT_XYZS, dtype=np.float64).reshape(-1, 3)[0]
        origin = self.deck_origin_m
        init_height = float(init[2] - origin[2])
        init_lateral = float(np.hypot(init[0] - origin[0], init[1] - origin[1]))
        row.update(
            {
                "ss": spec.sea_state,
                "vessel": spec.vessel,
                "heading_deg": float(spec.heading_deg),
                "speed_kn": float(spec.speed_kn),
                "realization_seed": int(spec.seed),
                "episode_seed": self._current.episode_seed,
                "index": self._current.index,
                "pad": self._pad,
                "detectors_disagree": bool(info.get("detectors_disagree", False)),
                "init_height_m": init_height,
                "init_lateral_m": init_lateral,
            }
        )
        return {
            "outcome": row["outcome"],
            "termination_reason": row["termination_reason"],
            "ss": spec.sea_state,
            "heading_deg": float(spec.heading_deg),
            "speed_kn": float(spec.speed_kn),
            "realization_seed": int(spec.seed),
            "episode_seed": self._current.episode_seed,
            "t0_model_s": row["t0_model_s"],
            "init_height_m": init_height,
            "init_lateral_m": init_lateral,
            "closing_speed_normal_m_s": row["closing_speed_normal_m_s"],
            "max_penetration_m": row["max_penetration_m"],
            "detectors_disagree": row["detectors_disagree"],
            "eval_idle": self._current.idle,
            "eval_index": self._current.index,
            "episode_row": row,
        }


@dataclass(frozen=True)
class EnvFactory:
    """A picklable zero-argument factory building one wrapped worker environment.

    ``SubprocVecEnv`` pickles the factory into each worker, which then imports PyBullet and
    builds its own client -- no connected client ever crosses a process boundary.

    Attributes:
        cfgs: The committed configs, reward weights already overridden.
        pool: Realizations this worker may fly (train pool, or tune pool for evaluation).
        pad: Pad name.
        seed: Training seed.
        rank: Worker index.
        stage: Initial sea state, or ``None``.
        monitor_dir: Directory for ``<rank>.monitor.csv``, or ``None`` for no Monitor.
        prefetch: Build a second landing environment and prepare each next episode on it
            in a background thread (:class:`PoolSamplingEnv`, "Reset prefetch").
    """

    cfgs: EvalConfigs
    pool: tuple[RealizationSpec, ...]
    pad: str
    seed: int
    rank: int
    stage: str | None
    monitor_dir: Path | None
    prefetch: bool = False

    def _motion(self, spec: RealizationSpec) -> EpisodeMotionSource:
        """Return the deck-motion source of one realization."""
        return motion_for(
            self.cfgs, spec.vessel, spec.sea_state, spec.heading_deg, spec.speed_kn, spec.seed
        )

    def _offset(self, vessel: str) -> tuple[float, float, float]:
        """Return the pad lever arm for ``vessel``, metres model scale."""
        return pad_offset_for(self.cfgs, vessel, self.pad)

    def __call__(self) -> gym.Env[FloatArray, FloatArray]:
        """Build the worker's environment: landing env(s) -> pool sampling -> Monitor."""
        from stable_baselines3.common.monitor import Monitor

        first = self.pool[0]
        base = make_env(self.cfgs, self._motion(first), first.vessel, self.pad, self.seed)
        standby = (
            make_env(self.cfgs, self._motion(first), first.vessel, self.pad, self.seed)
            if self.prefetch
            else None
        )
        wrapped: gym.Env[FloatArray, FloatArray] = PoolSamplingEnv(
            base,
            self.pool,
            self._motion,
            self._offset,
            pad=self.pad,
            seed=self.seed,
            rank=self.rank,
            stage=self.stage,
            deck_origin_m=self.cfgs.landing.platform.deck_origin_m,
            standby=standby,
        )
        if self.monitor_dir is not None:
            self.monitor_dir.mkdir(parents=True, exist_ok=True)
            wrapped = Monitor(
                wrapped,
                filename=str(self.monitor_dir / str(self.rank)),
                info_keywords=MONITOR_KEYWORDS,
            )
        return wrapped
