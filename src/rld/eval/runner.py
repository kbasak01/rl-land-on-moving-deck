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
detector), ``effort_mean_sq`` and ``action_jerk_mean``. Missing values (no touchdown) are
NaN, never 0.

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
from typing import Any, Protocol

import numpy as np
from dmf.data.splits import realization_key
from dmf.typedefs import FloatArray

from rld.control.base import ControlSpec, PrivilegedContext
from rld.control.registry import entry, make_controller
from rld.envs.landing_env import ACTION_DIM, DeckLandingAviary
from rld.eval.envs import EvalConfigs, make_env, motion_for, pad_offset_for
from rld.eval.episodes import ListedEpisode

__all__ = [
    "DEFAULT_CHUNK",
    "DEFAULT_WORKERS",
    "EPISODE_COLUMNS",
    "LIST_ROW_COLUMNS",
    "RECORD_COLUMNS",
    "CallablePolicy",
    "EpisodeListMismatchError",
    "EvalPolicy",
    "PolicySpec",
    "callable_spec",
    "controller_spec",
    "episode_rows_csv",
    "run_chunk",
    "run_list",
    "run_matrix",
]

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


class EvalPolicy(Protocol):
    """What the runner needs from anything it evaluates.

    :class:`rld.control.base.Controller` satisfies it; so does :class:`CallablePolicy`.
    """

    def reset(self, seed: int, context: PrivilegedContext | None = None) -> None:
        """Clear per-episode state.

        Args:
            seed: The episode seed.
            context: The true deck trajectory, passed only under a privileged spec.
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

    def reset(self, seed: int, context: PrivilegedContext | None = None) -> None:
        """Do nothing: the wrapped callable is stateless.

        Args:
            seed: The episode seed (unused).
            context: Must be ``None``.

        Raises:
            ValueError: If a context is passed.
        """
        del seed
        if context is not None:
            raise ValueError("a CallablePolicy is never privileged")

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
    """

    method: str
    privileged: bool
    build: Callable[[EvalConfigs], EvalPolicy]
    run_seed: int = 0


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
    )


def callable_spec(
    method: str, build: Callable[[EvalConfigs], EvalPolicy], run_seed: int
) -> PolicySpec:
    """Return the spec of a learned (never privileged) policy -- the Phase 5 hook.

    Args:
        method: Method label, e.g. ``"ppo"``.
        build: Picklable factory building the policy inside the worker.
        run_seed: The training seed.

    Returns:
        The :class:`PolicySpec`.
    """
    return PolicySpec(method=method, privileged=False, build=build, run_seed=run_seed)


def _check_privilege(spec: PolicySpec, policy: EvalPolicy) -> None:
    """Refuse a policy that claims a privilege its spec does not grant, or vice versa.

    Args:
        spec: The spec.
        policy: The built policy.

    Raises:
        ValueError: If the policy declares a ``privileged`` flag that differs from the
            spec's.
    """
    declared = getattr(policy, "privileged", None)
    if declared is not None and bool(declared) != spec.privileged:
        raise ValueError(
            f"{spec.method}: policy declares privileged={declared}, spec says {spec.privileged}"
        )


def _check_start(env: DeckLandingAviary, info: Mapping[str, Any], row: ListedEpisode) -> None:
    """Assert that the reset reproduced the listed episode exactly.

    Args:
        env: The environment, immediately after ``reset``.
        info: The reset's info dict.
        row: The listed episode.

    Raises:
        EpisodeListMismatchError: On any difference in ``t0``, initial position or realization.
    """
    t0 = float(env.record.t0_model_s)
    position = tuple(float(v) for v in np.asarray(env.pos[0], dtype=np.float64).reshape(3))
    key = realization_key(row.ss, row.heading_deg, row.speed_kn, row.vessel, row.realization_seed)
    if t0 != row.t0_model_s:
        raise EpisodeListMismatchError(f"{row.regime}/{row.ss}/{row.index}: t0 {t0!r} != listed")
    if position != row.init_xyz_m:
        raise EpisodeListMismatchError(
            f"{row.regime}/{row.ss}/{row.index}: init {position} != listed {row.init_xyz_m}"
        )
    if info["realization_key"] != key:
        raise EpisodeListMismatchError(
            f"{row.regime}/{row.ss}/{row.index}: key {info['realization_key']} != {key}"
        )


def _none_to_nan(value: Any) -> Any:
    """Return NaN for ``None`` so every numeric column is numeric."""
    return float("nan") if value is None else value


def _episode_row(
    row: ListedEpisode,
    spec: PolicySpec,
    env: DeckLandingAviary,
    actions: Sequence[FloatArray],
) -> dict[str, Any]:
    """Flatten one finished episode into its committed row.

    Args:
        row: The listed episode.
        spec: The evaluated policy's spec.
        env: The environment, immediately after the episode ended.
        actions: The actions sent to ``env.step``, clipped to ``[-1, 1]^3`` as the
            environment clips them, normalised units.

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
    )
    if tuple(out) != EPISODE_COLUMNS:
        raise RuntimeError(f"row columns {list(out)} != {list(EPISODE_COLUMNS)}")
    return out


def run_chunk(
    task: tuple[Sequence[ListedEpisode], PolicySpec, EvalConfigs, str | None],
) -> list[dict[str, Any]]:
    """Evaluate one policy on a contiguous chunk of listed episodes in one environment.

    Module-level with one packed argument so ``multiprocessing`` can pickle it.

    Args:
        task: ``(episodes, spec, configs, pad_override)``. ``pad_override`` replaces every
            row's pad (the pad-at-CG control arm); ``None`` keeps the listed pad.

    Returns:
        One row per episode, in the order given.

    Raises:
        EpisodeListMismatchError: If any reset does not reproduce its listed episode.
        ValueError: If the policy's privilege disagrees with the spec, or it returns a
            non-finite or wrongly shaped action.
    """
    episodes, spec, cfgs, pad_override = task
    if not episodes:
        return []
    first = episodes[0]
    pad0 = first.pad if pad_override is None else pad_override
    env = make_env(
        cfgs,
        motion_for(
            cfgs, first.vessel, first.ss, first.heading_deg, first.speed_kn, first.realization_seed
        ),
        first.vessel,
        pad0,
        first.episode_seed,
    )
    policy = spec.build(cfgs)
    _check_privilege(spec, policy)
    max_steps = int(round(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz)) + 1
    rows: list[dict[str, Any]] = []
    try:
        for listed in episodes:
            pad = listed.pad if pad_override is None else pad_override
            env.set_motion(
                motion_for(
                    cfgs,
                    listed.vessel,
                    listed.ss,
                    listed.heading_deg,
                    listed.speed_kn,
                    listed.realization_seed,
                ),
                pad,
                pad_offset_for(cfgs, listed.vessel, pad),
            )
            obs, info = env.reset(seed=listed.episode_seed)
            _check_start(env, info, listed)
            context = PrivilegedContext.from_env(env) if spec.privileged else None
            policy.reset(listed.episode_seed, context)
            actions: list[FloatArray] = []
            for _ in range(max_steps):
                action = np.asarray(policy.act(obs), dtype=np.float64)
                if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
                    raise ValueError(f"{spec.method}: bad action {action!r}")
                actions.append(np.clip(action, -1.0, 1.0))
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            rows.append(_episode_row(listed, spec, env, actions))
    finally:
        env.close()
    return rows


def run_list(
    episodes: Sequence[ListedEpisode],
    spec: PolicySpec,
    cfgs: EvalConfigs,
    *,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
    pad_override: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one policy on a list of committed episodes.

    Args:
        episodes: Listed episodes, in committed order (one or several cells).
        spec: What to evaluate.
        cfgs: The committed configs.
        workers: Processes; the rows do not depend on it.
        chunk: Episodes per task; the rows do not depend on it either.
        pad_override: Replace every row's pad (the pad-at-CG control arm), or ``None``.

    Returns:
        One row per episode (:data:`EPISODE_COLUMNS`), in the order of ``episodes``.

    Raises:
        ValueError: If ``workers`` or ``chunk`` is not positive.
    """
    if workers < 1 or chunk < 1:
        raise ValueError(f"workers and chunk must be positive, got {workers}, {chunk}")
    tasks = [
        (episodes[i : i + chunk], spec, cfgs, pad_override) for i in range(0, len(episodes), chunk)
    ]
    if workers == 1 or len(tasks) <= 1:
        results = [run_chunk(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            results = list(pool.imap(run_chunk, tasks, chunksize=1))
    return [row for rows in results for row in rows]


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
    if workers < 1 or chunk < 1:
        raise ValueError(f"workers and chunk must be positive, got {workers}, {chunk}")
    tasks = [
        (episodes[i : i + chunk], spec, cfgs, pad_override)
        for spec in specs
        for i in range(0, len(episodes), chunk)
    ]
    if workers == 1 or len(tasks) <= 1:
        results = [run_chunk(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            results = list(pool.imap(run_chunk, tasks, chunksize=1))
    return [row for rows in results for row in rows]


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
