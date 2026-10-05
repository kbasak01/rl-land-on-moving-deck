"""The evaluation runner: one policy, one committed episode list, one row per episode.

Contract
--------
* **The list is the only source of episodes.** Every reset uses the listed episode seed on
  the listed realization, and the runner checks on **every** reset that the environment
  starts where the list says it does: ``t0_model_s``, the drone's initial position (read
  from the physics state) and the realization key must match exactly, or
  :class:`EpisodeListMismatchError` is raised. A list that no longer describes the environment
  is a protocol failure, not a warning.
* **Privilege comes from the spec, never from the policy.** A
  :class:`~rld.control.base.PrivilegedContext` (the true future deck trajectory) is built
  and passed to ``reset`` **only** when the :class:`PolicySpec` is privileged, and for a
  registry controller that flag is copied from :mod:`rld.control.registry`. A policy that
  declares itself privileged under a non-privileged spec is refused.
* **The ship-motion feed comes from the spec, too.** A past-only
  :class:`~rld.deck.forecast.ShipMotionFeed` is built and handed to ``reset`` **only** when
  the :class:`PolicySpec` has ``needs_motion_feed`` (copied from the registry). It is built
  fresh for every episode from the environment's own source and pad
  (``ShipMotionFeed.for_pad(env.motion, env.pad, pads, scale)``), ``reset`` to the episode
  start ``t0`` and advanced to ``t0 + k * ctrl_dt`` before the ``k``-th ``act``; after each
  advance the runner checks that the feed's clock is exactly the environment's control
  time ``t0 + env.record.steps * ctrl_dt`` (seconds model scale) and raises
  :class:`FeedClockError` otherwise. A feed is never shared between episodes, chunks or
  workers.
* **The static list cannot feed a forecaster.** :class:`~rld.envs.platform.StaticDeckMotion`
  has no vessel and no ship channels, and its start offsets go down to 0.51 s model, inside
  the 4.0 s model lookback. :func:`feed_skip_reason` names that; :func:`split_runnable`
  separates such episodes **with the reason recorded** (they become a ``skipped`` cell in
  the summary, never a silent drop), and :func:`run_chunk` refuses one outright.
* **Pad override (the pad-at-CG control arm).** ``pad_override`` re-points every episode at
  another pad of the same hull on the **same** listed episode: the start offset and the
  initial drone position do not depend on the pad (the plate is anchored at the deck origin
  with the pad's standing lever arm removed), so :func:`_check_start` holds unchanged, and it
  additionally checks that the environment is on the effective pad with that pad's lever
  arm. The row's ``pad`` column is the **effective** pad. Ground truth
  (``td_in_quiescent_window``) is rebuilt from the environment after the episode, so it reads
  the effective pad's ``v_z``.
* **Phase 7 arms (P7-D1).** An :class:`EvalArm` also carries ``motion_kind`` (the deck-motion
  model, :data:`~rld.eval.envs.MOTION_KINDS`; ``"jonswap"`` by default), ``cfgs_override``
  (a per-arm :class:`~rld.eval.envs.EvalConfigs` from :func:`~rld.eval.envs.with_noise` or
  :func:`~rld.eval.envs.with_lambda`; ``None`` keeps the run's configs) and ``start_check``:

  - ``"strict"`` (default) -- :func:`_check_start` as above: ``t0``, initial position,
    realization key and pad all equal the list's. The sinusoid, noise and pad-at-CG arms keep
    it: their start offset and initial state are the listed ones.
  - ``"lambda"`` -- the lambda arm (P7-D1 §3): at another Froude scale the environment
    **re-draws** ``t0`` from the same episode seed inside the new scale's window, so the
    check asserts the realization key, the initial position and the pad, and does not
    compare ``t0``; the row's ``t0_model_s`` records the start actually flown.

  Episode rows gain no column: an arm's condition is written at summary level by the
  caller. With every field at its default the task is exactly the pre-Phase 7 one.
* **Output does not depend on ``workers``.** The list is cut into contiguous chunks of
  ``chunk`` episodes whatever the worker count; each chunk builds one environment and one
  policy, re-points the environment with ``set_motion`` between episodes, and the rows are
  reassembled in list order. Because every episode is fully determined by its reset (P2-D7)
  and every controller clears its state in ``reset``, the rows do not depend on ``chunk``
  either; ``tests/test_eval_runner.py`` checks both.

The Phase 5 hook
----------------
:func:`controller_spec` builds a :class:`PolicySpec` for a registry controller. A learned
policy is run the same way through :func:`callable_spec`: pass a **picklable** factory
(a module-level function or a ``functools.partial`` of one) that builds the policy inside
the worker from the :class:`~rld.eval.envs.EvalConfigs`, and a ``run_seed`` naming the
training seed. :class:`CallablePolicy` adapts a bare ``obs -> action`` callable that has no
per-episode state. Learned policies are not privileged.

Per-episode rows
----------------
:data:`EPISODE_COLUMNS`: the list's columns, then the method (``method``, ``privileged``,
``run_seed``), then ``EpisodeRecord.as_row()`` verbatim, then the extras the metrics need --
``detectors_disagree``, ``closing_speed_world_z_m_s``, ``time_to_touchdown_s`` (contact
detector), ``effort_mean_sq``, ``action_jerk_mean`` and ``td_in_quiescent_window``. Missing
values (no touchdown) are NaN, never 0 and never False.

``td_in_quiescent_window`` is evaluation ground truth (:mod:`rld.eval.truth`): whether the
**true** deck satisfied the permissive quiescence predicate
(:class:`rld.control.quiescence.QuiescenceRule`) over the 12 control-rate samples starting
at the contact touchdown. The true trajectory is rebuilt with
:meth:`~rld.control.base.PrivilegedContext.from_env` **after** the episode has ended, for
every method alike; it is never passed to a non-privileged policy.

Units: times seconds model scale, lengths metres model scale, speeds metres per second model
scale, angles degrees; actions and the two action statistics are in normalised action units
(multiply by ``v_max`` = 1.5 m/s for a velocity setpoint).
"""

import csv
import io
import multiprocessing as mp
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from dmf.data.splits import realization_key
from dmf.typedefs import FloatArray

from rld.control.base import ControlSpec, PrivilegedContext
from rld.control.registry import entry, make_controller
from rld.envs.landing_env import ACTION_DIM, DeckLandingAviary
from rld.eval.envs import (
    STATIC_LABEL,
    EvalConfigs,
    MotionKind,
    make_env,
    motion_for,
    pad_offset_for,
)
from rld.eval.episodes import ListedEpisode
from rld.eval.truth import td_in_quiescent_window, touchdown_rule

if TYPE_CHECKING:
    # Type-only: rld.deck.forecast imports torch; it is imported inside the worker, and only
    # for a spec that needs the feed.
    from rld.deck.forecast import ShipMotionFeed

__all__ = [
    "DEFAULT_CHUNK",
    "DEFAULT_WORKERS",
    "EPISODE_COLUMNS",
    "LIST_ROW_COLUMNS",
    "RECORD_COLUMNS",
    "CallablePolicy",
    "EpisodeListMismatchError",
    "EvalArm",
    "EvalPolicy",
    "FeedClockError",
    "PolicySpec",
    "START_CHECKS",
    "SkippedEpisode",
    "StartCheck",
    "callable_spec",
    "controller_spec",
    "episode_rows_csv",
    "feed_skip_reason",
    "run_arms",
    "run_chunk",
    "run_list",
    "run_matrix",
    "split_runnable",
]

#: Why a feed-consuming policy is not flown on the static-pad list. Recorded verbatim in the
#: summary's ``skip_reason`` column and in the rendered table.
STATIC_FEED_SKIP_REASON: str = (
    "the static-pad list has no ship motion to feed a forecaster "
    "(StaticDeckMotion has no vessel or ship channels, and its t0 goes down to 0.51 s model, "
    "inside the 4.0 s model lookback)"
)

#: How a reset is checked against its listed episode (see the module docstring).
type StartCheck = Literal["strict", "lambda"]

#: Every :data:`StartCheck`.
START_CHECKS: tuple[str, ...] = ("strict", "lambda")

#: A runner task: ``(episodes, spec, configs, pad_override)``, optionally followed by
#: ``motion_kind`` and ``start_check`` (the 4-tuple is the pre-Phase 7 form, still accepted).
type RunTask = (
    tuple[Sequence[ListedEpisode], "PolicySpec", EvalConfigs, str | None]
    | tuple[Sequence[ListedEpisode], "PolicySpec", EvalConfigs, str | None, MotionKind, StartCheck]
)

#: Episodes per parallel task. Fixed independently of the worker count.
DEFAULT_CHUNK: int = 25

#: Default worker processes.
DEFAULT_WORKERS: int = 24

#: Columns copied from the episode list (``t0_model_s`` comes from the record and is
#: asserted equal to the list's).
LIST_ROW_COLUMNS: tuple[str, ...] = (
    "regime",
    "ss",
    "index",
    "pad",
    "vessel",
    "heading_deg",
    "speed_kn",
    "realization_seed",
    "episode_seed",
    "in_training_distribution",
    "init_x_m",
    "init_y_m",
    "init_z_m",
)

#: Columns of ``EpisodeRecord.as_row()``, in its order. Checked against the live record on
#: every episode, so a change in :mod:`rld.envs` cannot silently reshape the output.
RECORD_COLUMNS: tuple[str, ...] = (
    "outcome",
    "termination_reason",
    "t0_model_s",
    "steps",
    "return",
    "control_effort",
    "action_jerk",
    "dwell_s",
    "max_penetration_m",
    "tunnelled",
    "contact_events",
    "touchdown_contact",
    "touchdown_analytic",
    "td_t_episode_s",
    "td_t_episode_analytic_s",
    "rel_vz_normal_m_s",
    "rel_vz_world_z_m_s",
    "closing_speed_normal_m_s",
    "lateral_offset_m",
    "rel_tilt_deg",
    "abs_tilt_deg",
    "deck_tilt_deg",
    "n_contacts",
)

#: Extras computed by the runner.
_EXTRA_COLUMNS: tuple[str, ...] = (
    "detectors_disagree",
    "closing_speed_world_z_m_s",
    "time_to_touchdown_s",
    "effort_mean_sq",
    "action_jerk_mean",
    "td_in_quiescent_window",
)

#: Committed column order of every per-episode table.
EPISODE_COLUMNS: tuple[str, ...] = (
    *LIST_ROW_COLUMNS,
    "method",
    "privileged",
    "run_seed",
    *RECORD_COLUMNS,
    *_EXTRA_COLUMNS,
)


class EpisodeListMismatchError(AssertionError):
    """The environment did not start an episode where the committed list says it does."""


class FeedClockError(AssertionError):
    """The ship-motion feed's clock is not the environment's control time at an ``act``."""


class EvalPolicy(Protocol):
    """What the runner needs from anything it evaluates.

    :class:`rld.control.base.Controller` satisfies it; so does :class:`CallablePolicy`.
    """

    def reset(
        self,
        seed: int,
        context: PrivilegedContext | None = None,
        motion_feed: "ShipMotionFeed | None" = None,
    ) -> None:
        """Clear per-episode state.

        The runner calls ``reset(seed, context)`` for a spec without ``needs_motion_feed``
        and ``reset(seed, context, feed)`` for one with it, so a policy that never takes a
        feed may keep the two-argument form.

        Args:
            seed: The episode seed.
            context: The true deck trajectory, passed only under a privileged spec.
            motion_feed: The episode's past-only ship-motion feed, already reset to ``t0``,
                passed only under a spec with ``needs_motion_feed``.
        """
        ...

    def act(self, obs: FloatArray) -> FloatArray:
        """Return a ``(3,)`` action in ``[-1, 1]^3`` for one observation.

        Args:
            obs: The environment's observation.
        """
        ...


class CallablePolicy:
    """Adapt a stateless ``obs -> action`` callable (e.g. a Phase 5 policy) to the runner.

    Attributes:
        fn: The callable. It must hold no per-episode state; one that does must implement
            :class:`EvalPolicy` itself so ``reset`` can clear it.
        privileged: Always False: a learned policy never receives the true deck trajectory.
    """

    privileged: bool = False

    def __init__(self, fn: Callable[[FloatArray], FloatArray]) -> None:
        """Wrap ``fn``.

        Args:
            fn: Maps an observation to a ``(3,)`` action in normalised units.
        """
        self.fn = fn

    def reset(
        self,
        seed: int,
        context: PrivilegedContext | None = None,
        motion_feed: "ShipMotionFeed | None" = None,
    ) -> None:
        """Do nothing: the wrapped callable is stateless.

        Args:
            seed: The episode seed (unused).
            context: Must be ``None``.
            motion_feed: Must be ``None``: a bare callable has nowhere to keep a feed.

        Raises:
            ValueError: If a context or a feed is passed.
        """
        del seed
        if context is not None:
            raise ValueError("a CallablePolicy is never privileged")
        if motion_feed is not None:
            raise ValueError("a CallablePolicy does not consume a ShipMotionFeed")

    def act(self, obs: FloatArray) -> FloatArray:
        """Return ``fn(obs)``.

        Args:
            obs: The observation.

        Returns:
            The action, normalised units.
        """
        return self.fn(obs)


@dataclass(frozen=True)
class PolicySpec:
    """A picklable description of what to evaluate.

    Attributes:
        method: Method label in every table, e.g. ``"pid_feedforward"`` or ``"ppo"``.
        privileged: Whether the runner passes the true deck trajectory. For a registry
            controller this is the registry's flag.
        build: Picklable factory ``(EvalConfigs) -> EvalPolicy``, called once per chunk
            inside the worker.
        run_seed: Training seed of a learned policy; 0 for a deterministic controller.
        needs_motion_feed: Whether the runner builds, resets and advances a past-only
            :class:`~rld.deck.forecast.ShipMotionFeed` and hands it to ``reset``. For a
            registry controller this is the registry's flag.
    """

    method: str
    privileged: bool
    build: Callable[[EvalConfigs], EvalPolicy]
    run_seed: int = 0
    needs_motion_feed: bool = False


@dataclass(frozen=True)
class SkippedEpisode:
    """A listed episode a spec was **not** flown on, and why (never a silent drop).

    Attributes:
        episode: The listed episode.
        pad: The effective pad of the arm it was skipped in.
        reason: Why, verbatim into ``summary.csv``'s ``skip_reason``.
    """

    episode: ListedEpisode
    pad: str
    reason: str


@dataclass(frozen=True)
class EvalArm:
    """One (policy, pad, condition) arm: what to fly, on which pad, under which conditions.

    Attributes:
        spec: What to evaluate.
        pad_override: Replace every row's pad (``"cg"`` for the pad-at-CG control arm), or
            ``None`` to keep the listed pad (the frozen aft pad).
        episodes: The runnable listed episodes, in committed order.
        motion_kind: The deck-motion model the listed realizations are flown under
            (``"jonswap"``: the frozen lists as committed).
        cfgs_override: The arm's own configs (noise or lambda), or ``None`` for the run's.
        start_check: ``"strict"`` or ``"lambda"`` (module docstring).
    """

    spec: PolicySpec
    pad_override: str | None
    episodes: Sequence[ListedEpisode]
    motion_kind: MotionKind = "jonswap"
    cfgs_override: EvalConfigs | None = None
    start_check: StartCheck = "strict"


def _build_registered(name: str, config_path: Path | None, cfgs: EvalConfigs) -> EvalPolicy:
    """Build a registry controller from the same configs the environment was built from.

    Args:
        name: Registry name.
        config_path: Override of the committed YAML, or ``None``.
        cfgs: The evaluation's configs.

    Returns:
        The controller. Its ``privileged`` flag is checked against the spec by
        :func:`_check_privilege` in the worker.
    """
    spec = ControlSpec(
        landing=cfgs.landing,
        observation=cfgs.observation,
        scale=cfgs.scaling.froude_scale(),
    )
    return make_controller(name, spec, config_path)


def controller_spec(name: str, config_path: Path | None = None) -> PolicySpec:
    """Return the spec of a registered controller, with the registry's privileged flag.

    Args:
        name: Registry name, e.g. ``"oracle_gated"``.
        config_path: Override of the committed YAML (tests only; committed results use the
            registry's path).

    Returns:
        The :class:`PolicySpec`.
    """
    item = entry(name)
    return PolicySpec(
        method=name,
        privileged=item.privileged,
        build=partial(_build_registered, name, config_path),
        run_seed=0,
        needs_motion_feed=item.needs_motion_feed,
    )


def callable_spec(
    method: str,
    build: Callable[[EvalConfigs], EvalPolicy],
    run_seed: int,
    *,
    needs_motion_feed: bool = False,
) -> PolicySpec:
    """Return the spec of a learned (never privileged) policy -- the Phase 5 hook.

    Args:
        method: Method label, e.g. ``"ppo"``.
        build: Picklable factory building the policy inside the worker.
        run_seed: The training seed.
        needs_motion_feed: Whether the policy consumes the past-only ship-motion feed (a
            forecast-conditioned policy that runs its own forecaster).

    Returns:
        The :class:`PolicySpec`.
    """
    return PolicySpec(
        method=method,
        privileged=False,
        build=build,
        run_seed=run_seed,
        needs_motion_feed=needs_motion_feed,
    )


def _check_privilege(spec: PolicySpec, policy: EvalPolicy) -> None:
    """Refuse a policy whose declared flags differ from its spec's.

    Args:
        spec: The spec.
        policy: The built policy.

    Raises:
        ValueError: If the policy declares a ``privileged`` or ``needs_motion_feed`` flag
            that differs from the spec's.
    """
    declared = getattr(policy, "privileged", None)
    if declared is not None and bool(declared) != spec.privileged:
        raise ValueError(
            f"{spec.method}: policy declares privileged={declared}, spec says {spec.privileged}"
        )
    feed = getattr(policy, "needs_motion_feed", None)
    if feed is not None and bool(feed) != spec.needs_motion_feed:
        raise ValueError(
            f"{spec.method}: policy declares needs_motion_feed={feed}, spec says "
            f"{spec.needs_motion_feed}"
        )


def _check_start(
    env: DeckLandingAviary,
    info: Mapping[str, Any],
    row: ListedEpisode,
    pad: str,
    pad_offset_m: tuple[float, float, float],
    *,
    check_t0: bool = True,
) -> None:
    """Assert that the reset reproduced the listed episode exactly, on the effective pad.

    ``t0`` and the initial position are the list's under any pad: neither depends on the pad
    (the start offset is drawn from the motion source's window, the drone is placed relative
    to the deck origin), which is what lets the pad-at-CG arm fly the frozen lists.

    Args:
        env: The environment, immediately after ``reset``.
        info: The reset's info dict.
        row: The listed episode.
        pad: The effective pad (the listed one, or the override).
        pad_offset_m: That pad's standing lever arm, metres model scale.
        check_t0: Compare ``t0`` with the list's (``False`` only for the lambda arm, whose
            environment re-draws ``t0`` inside another scale's window, P7-D1 §3).

    Raises:
        EpisodeListMismatchError: On any difference in ``t0`` (when checked), initial
            position, realization, pad or pad lever arm.
    """
    t0 = float(env.record.t0_model_s)
    position = tuple(float(v) for v in np.asarray(env.pos[0], dtype=np.float64).reshape(3))
    key = realization_key(row.ss, row.heading_deg, row.speed_kn, row.vessel, row.realization_seed)
    if check_t0 and t0 != row.t0_model_s:
        raise EpisodeListMismatchError(f"{row.regime}/{row.ss}/{row.index}: t0 {t0!r} != listed")
    if position != row.init_xyz_m:
        raise EpisodeListMismatchError(
            f"{row.regime}/{row.ss}/{row.index}: init {position} != listed {row.init_xyz_m}"
        )
    if info["realization_key"] != key:
        raise EpisodeListMismatchError(
            f"{row.regime}/{row.ss}/{row.index}: key {info['realization_key']} != {key}"
        )
    if env.pad != pad or tuple(env.pad_offset_m) != tuple(pad_offset_m):
        raise EpisodeListMismatchError(
            f"{row.regime}/{row.ss}/{row.index}: env pad {env.pad!r} {env.pad_offset_m} != "
            f"effective {pad!r} {pad_offset_m}"
        )


def _none_to_nan(value: Any) -> Any:
    """Return NaN for ``None`` so every numeric column is numeric."""
    return float("nan") if value is None else value


def _episode_row(
    row: ListedEpisode,
    pad: str,
    spec: PolicySpec,
    env: DeckLandingAviary,
    actions: Sequence[FloatArray],
    td_quiescent: bool | float,
) -> dict[str, Any]:
    """Flatten one finished episode into its committed row.

    Args:
        row: The listed episode.
        pad: The effective pad, written to the ``pad`` column (the listed pad unless
            overridden).
        spec: The evaluated policy's spec.
        env: The environment, immediately after the episode ended.
        actions: The actions sent to ``env.step``, clipped to ``[-1, 1]^3`` as the
            environment clips them, normalised units.
        td_quiescent: ``td_in_quiescent_window`` from :mod:`rld.eval.truth`: a bool, or
            NaN when there was no contact touchdown.

    Returns:
        A dict with keys :data:`EPISODE_COLUMNS`, in order.

    Raises:
        RuntimeError: If the episode ended unclassified or the record's columns changed.
    """
    record = env.record.as_row()
    if tuple(record) != RECORD_COLUMNS:
        raise RuntimeError(f"EpisodeRecord.as_row() columns changed: {list(record)}")
    if record["outcome"] is None or record["termination_reason"] is None:
        raise RuntimeError(f"{row.regime}/{row.ss}/{row.index}: episode ended unclassified")
    record = {name: _none_to_nan(value) for name, value in record.items()}
    contact = env.record.contact_record
    if contact is None:
        record["n_contacts"] = 0
    stack = np.asarray(actions, dtype=np.float64).reshape(-1, ACTION_DIM)
    effort = float(np.mean(np.sum(stack * stack, axis=1)))
    jerk = (
        float(np.mean(np.linalg.norm(np.diff(stack, axis=0), axis=1)))
        if stack.shape[0] > 1
        else float("nan")
    )
    out: dict[str, Any] = {name: getattr(row, name) for name in LIST_ROW_COLUMNS}
    out["pad"] = pad
    out.update(method=spec.method, privileged=spec.privileged, run_seed=spec.run_seed)
    out.update(record)
    out.update(
        detectors_disagree=env.record.detectors_disagree(
            env.success.detector_disagreement_window_s
        ),
        closing_speed_world_z_m_s=(
            float("nan") if contact is None else float(contact.closing_speed_world_z_m_s)
        ),
        time_to_touchdown_s=float("nan") if contact is None else float(contact.t_episode_s),
        effort_mean_sq=effort,
        action_jerk_mean=jerk,
        td_in_quiescent_window=td_quiescent,
    )
    if tuple(out) != EPISODE_COLUMNS:
        raise RuntimeError(f"row columns {list(out)} != {list(EPISODE_COLUMNS)}")
    return out


def feed_skip_reason(listed: ListedEpisode, spec: PolicySpec) -> str | None:
    """Return why ``spec`` cannot be flown on ``listed``, or ``None`` if it can.

    Only a feed-consuming spec is ever skipped, and only on the static-pad list: there is no
    ship to feed the forecaster (:data:`STATIC_FEED_SKIP_REASON`).

    Args:
        listed: A listed episode.
        spec: What would be flown on it.

    Returns:
        The reason, verbatim for ``summary.csv``, or ``None``.
    """
    if spec.needs_motion_feed and listed.vessel == STATIC_LABEL:
        return STATIC_FEED_SKIP_REASON
    return None


def split_runnable(
    episodes: Sequence[ListedEpisode], spec: PolicySpec, pad_override: str | None = None
) -> tuple[list[ListedEpisode], list[SkippedEpisode]]:
    """Separate the episodes ``spec`` can fly from those it cannot, keeping the reason.

    Args:
        episodes: Listed episodes, in committed order.
        spec: What to evaluate.
        pad_override: The arm's pad override, or ``None`` (recorded on each skip).

    Returns:
        ``(runnable, skipped)``, each in committed order.
    """
    runnable: list[ListedEpisode] = []
    skipped: list[SkippedEpisode] = []
    for listed in episodes:
        reason = feed_skip_reason(listed, spec)
        if reason is None:
            runnable.append(listed)
        else:
            pad = listed.pad if pad_override is None else pad_override
            skipped.append(SkippedEpisode(listed, pad, reason))
    return runnable, skipped


def _make_feed(env: DeckLandingAviary, cfgs: EvalConfigs, t0: float) -> "ShipMotionFeed":
    """Build this episode's past-only ship-motion feed and reset it to ``t0``.

    A new feed per episode: it wraps this episode's source and pad and nothing else, so no
    history can leak between episodes, chunks or workers.

    Args:
        env: The environment, immediately after ``reset`` (its source and effective pad).
        cfgs: The committed configs.
        t0: The episode start, seconds model scale (``env.record.t0_model_s``).

    Returns:
        The feed, clock at ``t0``, lookback filled from before ``t0``.

    Raises:
        FeedClockError: If the feed's lever arm is not the environment's pad lever arm
            (they are built from the same pad table; a disagreement is a wiring bug), or its
            clock is not ``t0`` after the reset.
    """
    from rld.deck.forecast import ShipMotionFeed  # heavy (torch); only in feed workers

    scale = cfgs.scaling.froude_scale()
    feed = ShipMotionFeed.for_pad(env.motion, env.pad, cfgs.pads, scale)
    arm_model_m = float(scale.length(np.asarray(feed.r_pad_full_m[0], dtype=np.float64)))
    if not np.isclose(arm_model_m, float(env.pad_offset_m[0]), rtol=1e-12, atol=1e-12):
        raise FeedClockError(
            f"feed lever arm {arm_model_m} m != env pad {env.pad!r} lever arm "
            f"{env.pad_offset_m[0]} m (model)"
        )
    feed.reset(t0)
    if feed.clock_model_s != t0:
        raise FeedClockError(f"feed clock {feed.clock_model_s!r} != t0 {t0!r} after reset")
    return feed


def _advance_feed(feed: "ShipMotionFeed", env: DeckLandingAviary, t0: float, k: int) -> None:
    """Advance the feed to the ``k``-th control instant and check it is the env's.

    Args:
        feed: This episode's feed.
        env: The environment, before its ``k``-th ``step``.
        t0: The episode start, seconds model scale.
        k: The index of the ``act`` about to be made (0 for the reset observation).

    Raises:
        FeedClockError: If the environment has not executed exactly ``k`` control steps, or
            the feed's clock is not exactly ``t0 + k * ctrl_dt`` (seconds model scale).
    """
    dt = env.cfg.ctrl_dt_s
    if env.record.steps != k:
        raise FeedClockError(f"env executed {env.record.steps} control steps, runner at {k}")
    t_ctrl = t0 + env.record.steps * dt
    feed.advance_to(t0 + k * dt)
    if feed.clock_model_s != t_ctrl:
        raise FeedClockError(
            f"feed clock {feed.clock_model_s!r} s != env control time {t_ctrl!r} s (model) "
            f"at act {k}"
        )


def _unpack(
    task: RunTask,
) -> tuple[Sequence[ListedEpisode], PolicySpec, EvalConfigs, str | None, MotionKind, StartCheck]:
    """Return a task's six fields, defaulting the Phase 7 ones for a 4-tuple task."""
    if len(task) == 4:
        episodes, spec, cfgs, pad_override = task
        return episodes, spec, cfgs, pad_override, "jonswap", "strict"
    if len(task) == 6:
        return task
    raise ValueError(f"a runner task has 4 or 6 fields, got {len(task)}")


def run_chunk(task: RunTask) -> list[dict[str, Any]]:
    """Evaluate one policy on a contiguous chunk of listed episodes in one environment.

    Module-level with one packed argument so ``multiprocessing`` can pickle it.

    Args:
        task: ``(episodes, spec, configs, pad_override[, motion_kind, start_check])``.
            ``pad_override`` replaces every row's pad (the pad-at-CG control arm); ``None``
            keeps the listed pad. ``motion_kind`` (default ``"jonswap"``) is the deck-motion
            model; ``start_check`` (default ``"strict"``) how each reset is checked.

    Returns:
        One row per episode, in the order given.

    Raises:
        EpisodeListMismatchError: If any reset does not reproduce its listed episode.
        FeedClockError: If a feed's clock is not the environment's control time at an act.
        ValueError: If the policy's flags disagree with the spec, it returns a non-finite
            or wrongly shaped action, or a feed-consuming spec is handed an episode it
            cannot fly (:func:`feed_skip_reason`; use :func:`split_runnable` first).
    """
    episodes, spec, cfgs, pad_override, motion_kind, start_check = _unpack(task)
    if start_check not in START_CHECKS:
        raise ValueError(f"unknown start check {start_check!r}")
    if not episodes:
        return []
    for listed in episodes:
        reason = feed_skip_reason(listed, spec)
        if reason is not None:
            raise ValueError(f"{spec.method}: {listed.regime}/{listed.ss}/{listed.index}: {reason}")
    first = episodes[0]
    pad0 = first.pad if pad_override is None else pad_override
    env = make_env(
        cfgs,
        motion_for(
            cfgs,
            first.vessel,
            first.ss,
            first.heading_deg,
            first.speed_kn,
            first.realization_seed,
            kind=motion_kind,
            episode_seed=first.episode_seed,
        ),
        first.vessel,
        pad0,
        first.episode_seed,
    )
    policy = spec.build(cfgs)
    _check_privilege(spec, policy)
    max_steps = int(round(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz)) + 1
    rule = touchdown_rule(cfgs)
    stride = int(cfgs.landing.pyb_steps_per_ctrl)
    rows: list[dict[str, Any]] = []
    try:
        for listed in episodes:
            pad = listed.pad if pad_override is None else pad_override
            offset = pad_offset_for(cfgs, listed.vessel, pad)
            env.set_motion(
                motion_for(
                    cfgs,
                    listed.vessel,
                    listed.ss,
                    listed.heading_deg,
                    listed.speed_kn,
                    listed.realization_seed,
                    kind=motion_kind,
                    episode_seed=listed.episode_seed,
                ),
                pad,
                offset,
            )
            obs, info = env.reset(seed=listed.episode_seed)
            _check_start(env, info, listed, pad, offset, check_t0=start_check == "strict")
            t0 = float(env.record.t0_model_s)
            context = PrivilegedContext.from_env(env) if spec.privileged else None
            feed = _make_feed(env, cfgs, t0) if spec.needs_motion_feed else None
            if feed is None:
                policy.reset(listed.episode_seed, context)
            else:
                policy.reset(listed.episode_seed, context, feed)
            actions: list[FloatArray] = []
            for k in range(max_steps):
                if feed is not None:
                    _advance_feed(feed, env, t0, k)
                action = np.asarray(policy.act(obs), dtype=np.float64)
                if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
                    raise ValueError(f"{spec.method}: bad action {action!r}")
                actions.append(np.clip(action, -1.0, 1.0))
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            # Evaluation ground truth, built after the episode for every method alike, on the
            # effective pad (the env's pad and lever arm).
            contact = env.record.contact_record
            truth = td_in_quiescent_window(
                PrivilegedContext.from_env(env),
                rule,
                None if contact is None else float(contact.t_episode_s),
                stride,
            )
            rows.append(_episode_row(listed, pad, spec, env, actions, truth))
    finally:
        env.close()
    return rows


def run_arms(
    arms: Sequence[EvalArm],
    cfgs: EvalConfigs,
    *,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
) -> list[dict[str, Any]]:
    """Evaluate several (policy, pad, episodes) arms through one worker pool.

    Every arm's episodes are cut into contiguous chunks of ``chunk`` exactly as
    :func:`run_list` cuts them, so an arm's rows are identical to ``run_list`` on the same
    arguments, at any ``workers``. An arm with ``cfgs_override`` flies under those configs
    instead of ``cfgs``; its ``motion_kind`` and ``start_check`` travel in the task.

    Args:
        arms: What to evaluate, in output order.
        cfgs: The committed configs.
        workers: Processes; the rows do not depend on it.
        chunk: Episodes per task; the rows do not depend on it.

    Returns:
        Every arm's rows, arm after arm, each in its episodes' order.

    Raises:
        ValueError: If ``workers`` or ``chunk`` is not positive.
    """
    if workers < 1 or chunk < 1:
        raise ValueError(f"workers and chunk must be positive, got {workers}, {chunk}")
    tasks: list[RunTask] = [
        (
            arm.episodes[i : i + chunk],
            arm.spec,
            cfgs if arm.cfgs_override is None else arm.cfgs_override,
            arm.pad_override,
            arm.motion_kind,
            arm.start_check,
        )
        for arm in arms
        for i in range(0, len(arm.episodes), chunk)
    ]
    if workers == 1 or len(tasks) <= 1:
        results = [run_chunk(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            results = list(pool.imap(run_chunk, tasks, chunksize=1))
    return [row for rows in results for row in rows]


def run_list(
    episodes: Sequence[ListedEpisode],
    spec: PolicySpec,
    cfgs: EvalConfigs,
    *,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
    pad_override: str | None = None,
    motion_kind: MotionKind = "jonswap",
    start_check: StartCheck = "strict",
) -> list[dict[str, Any]]:
    """Evaluate one policy on a list of committed episodes.

    Args:
        episodes: Listed episodes, in committed order (one or several cells). A
            feed-consuming spec must not be handed static-pad episodes
            (:func:`split_runnable`).
        spec: What to evaluate.
        cfgs: The committed configs.
        workers: Processes; the rows do not depend on it.
        chunk: Episodes per task; the rows do not depend on it either.
        pad_override: Replace every row's pad (the pad-at-CG control arm), or ``None``.
        motion_kind: The deck-motion model (default ``"jonswap"``).
        start_check: ``"strict"`` (default) or ``"lambda"``.

    Returns:
        One row per episode (:data:`EPISODE_COLUMNS`), in the order of ``episodes``.

    Raises:
        ValueError: If ``workers`` or ``chunk`` is not positive.
    """
    return run_arms(
        [EvalArm(spec, pad_override, episodes, motion_kind, None, start_check)],
        cfgs,
        workers=workers,
        chunk=chunk,
    )


def run_matrix(
    episodes: Sequence[ListedEpisode],
    specs: Sequence[PolicySpec],
    cfgs: EvalConfigs,
    *,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
    pad_override: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate several policies on the same list through one worker pool.

    Equivalent to concatenating :func:`run_list` over ``specs`` -- the chunks are the same
    and are reassembled in (spec, list) order -- but keeps every worker busy across the
    boundaries between policies.

    Args:
        episodes: Listed episodes, in committed order.
        specs: What to evaluate, in output order.
        cfgs: The committed configs.
        workers: Processes; the rows do not depend on it.
        chunk: Episodes per task; the rows do not depend on it.
        pad_override: Replace every row's pad, or ``None``.

    Returns:
        ``len(specs) * len(episodes)`` rows: every episode for ``specs[0]``, then for
        ``specs[1]``, and so on.

    Raises:
        ValueError: If ``workers`` or ``chunk`` is not positive.
    """
    return run_arms(
        [EvalArm(spec, pad_override, episodes) for spec in specs],
        cfgs,
        workers=workers,
        chunk=chunk,
    )


def episode_rows_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialise per-episode rows deterministically.

    Columns :data:`EPISODE_COLUMNS`; floats as ``repr`` (shortest exact round trip, ``nan``
    for missing), booleans as ``True``/``False``, newline line endings.

    Args:
        rows: Runner rows.

    Returns:
        The CSV text, header included.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(EPISODE_COLUMNS)
    for row in rows:
        writer.writerow(
            [
                repr(float(v)) if isinstance(v, float) else v
                for v in (row[c] for c in EPISODE_COLUMNS)
            ]
        )
    return buffer.getvalue()
