"""End-to-end control-step budget, measured inside real environment episodes (P8-D1 §9).

The ONNX policy of each timed architecture flies the closed-loop episodes
(:func:`rld.deploy.closed_loop.closed_loop_episodes`) in **one** process with every component at
one thread, and each control step's components are timed with ``time.perf_counter``:

* ``obs_build`` -- :meth:`DeckLandingAviary._computeObs` (the step's observation; the reset's for
  step 0);
* ``forecaster`` -- :func:`rld.rl.forecast_obs.forecast_block`: the DLinear-OLS window build, its
  ORT run (CPU, 1 thread) and the lead extraction (forecast methods only);
* ``policy_onnx`` -- the exported graph on the raw input (ORT CPU, batch 1, 1 thread), with the
  input assembly and the output clip of :meth:`OnnxNetworkMixin.network_action`;
* ``base_act`` -- ``pid_feedforward.act`` (residual methods only);
* ``compose`` -- :func:`rld.rl.residual.compose_residual` (residual methods only);
* ``deployment_sum`` -- the per-step sum of the five above, set against the 33.3 ms period.

Reported separately, as **simulation**, not deployment:

* ``feed_advance`` -- the ship-motion feed's ``advance_to`` (analytic dmf motion standing in for a
  ship motion sensor; forecast methods only);
* ``physics_step`` -- ``env.step`` minus its observation build (8 PyBullet substeps, contact
  polling, reward).

The step loop is :func:`rld.eval.runner.run_chunk`'s with the policy's ``act`` unrolled into
timed pieces, in ``ResidualPolicy.act``'s order (network, then base, then composition). One
untimed warmup episode (the first listed) is flown first. Each episode's outcome and step count is
kept so that the pipeline can check them against the closed-loop ONNX rows (same actions, same
episode).

Units: times milliseconds; the control period 1/30 s = 33.3 ms, model scale.
"""

import csv
import io
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from dmf.deploy.export_onnx import INPUT_NAME
from dmf.typedefs import FloatArray

from rld.deploy.latency import CONTROL_PERIOD_MS
from rld.deploy.onnx_policy import OnnxNetworkMixin, build_onnx_policy
from rld.envs.landing_env import ACTION_DIM
from rld.eval.envs import EvalConfigs, make_env, motion_for, pad_offset_for
from rld.eval.episodes import ListedEpisode
from rld.rl.forecast_obs import ForecastPolicy, episode_feed, forecast_block
from rld.rl.residual import ResidualPolicy, compose_residual

__all__ = [
    "DEPLOYMENT_COMPONENTS",
    "E2E_COLUMNS",
    "SIMULATION_COMPONENTS",
    "EpisodeOutcome",
    "StepTiming",
    "e2e_csv",
    "run_e2e",
]

#: Deployment components, in report order (seconds per step are stored).
DEPLOYMENT_COMPONENTS: tuple[str, ...] = (
    "obs_build",
    "forecaster",
    "policy_onnx",
    "base_act",
    "compose",
)

#: Simulation-side components, reported separately.
SIMULATION_COMPONENTS: tuple[str, ...] = ("feed_advance", "physics_step")

#: Columns of ``results/latency/e2e_budget.csv``.
E2E_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "architecture",
    "category",
    "component",
    "n_steps",
    "n_episodes",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "mean_ms",
    "max_ms",
    "control_period_ms",
    "p50_fraction_of_period",
    "p99_fraction_of_period",
)


@dataclass(frozen=True)
class StepTiming:
    """One control step's component times, seconds.

    Attributes:
        times_s: ``{component: seconds}`` (absent components are 0).
    """

    times_s: dict[str, float]

    @property
    def deployment_s(self) -> float:
        """The per-step deployment sum, seconds."""
        return sum(self.times_s.get(c, 0.0) for c in DEPLOYMENT_COMPONENTS)


@dataclass(frozen=True)
class EpisodeOutcome:
    """An e2e episode's outcome, for the consistency check against the closed loop.

    Attributes:
        ss: Sea state.
        index: Listed index.
        outcome: Outcome class.
        steps: Control steps executed.
    """

    ss: str
    index: int
    outcome: str
    steps: int


class _ObsTimer:
    """Wraps ``env._computeObs`` and keeps its last duration, seconds."""

    def __init__(self, inner: Callable[[], FloatArray]) -> None:
        self.inner = inner
        self.last_s = 0.0

    def __call__(self) -> FloatArray:
        start = time.perf_counter()
        out = self.inner()
        self.last_s = time.perf_counter() - start
        return out


def run_e2e(
    cfgs: EvalConfigs,
    episodes: Sequence[ListedEpisode],
    *,
    run_dir: Path,
    onnx_path: Path,
    warmup_episodes: int = 1,
) -> tuple[list[StepTiming], list[EpisodeOutcome]]:
    """Fly the ONNX policy on ``episodes`` and time every control step's components.

    Args:
        cfgs: The committed configs (noise must be off).
        episodes: The listed episodes, in order.
        run_dir: The run directory.
        onnx_path: Its exported graph.
        warmup_episodes: Leading episodes flown untimed (the first of ``episodes`` again).

    Returns:
        ``(per-step timings, per-episode outcomes)``; warmup episodes appear in neither.

    Raises:
        ValueError: If noise is enabled or an episode does not start where its list says.
    """
    import torch

    if cfgs.noise.enabled:
        raise ValueError("the e2e budget flies noise off (P8-D1 §9)")
    torch.set_num_threads(1)
    policy = build_onnx_policy(cfgs, run_dir=run_dir, onnx_path=onnx_path)
    session = cast(OnnxNetworkMixin, policy).onnx_session
    forecast = isinstance(policy, ForecastPolicy)
    residual = isinstance(policy, ResidualPolicy)
    scale = cfgs.scaling.froude_scale()
    first = episodes[0]
    env = make_env(
        cfgs,
        motion_for(
            cfgs, first.vessel, first.ss, first.heading_deg, first.speed_kn, first.realization_seed
        ),
        first.vessel,
        first.pad,
        first.episode_seed,
    )
    timer = _ObsTimer(env._computeObs)
    setattr(env, "_computeObs", timer)  # noqa: B010 (instance override of a method, by design)
    max_steps = int(round(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz)) + 1
    dt = float(env.cfg.ctrl_dt_s)
    timings: list[StepTiming] = []
    outcomes: list[EpisodeOutcome] = []
    plan = [episodes[0]] * warmup_episodes + list(episodes)
    try:
        for n, listed in enumerate(plan):
            record = n >= warmup_episodes
            env.set_motion(
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
            t0 = float(env.record.t0_model_s)
            position = tuple(float(v) for v in np.asarray(env.pos[0]).reshape(3))
            if t0 != listed.t0_model_s or position != listed.init_xyz_m:
                raise ValueError(f"{listed.ss}/{listed.index}: start differs from the list")
            feed = episode_feed(env, cfgs.pads, scale) if forecast else None
            if feed is not None:
                policy.reset(listed.episode_seed, None, feed)
            else:
                policy.reset(listed.episode_seed)
            obs_build_s = timer.last_s
            for k in range(max_steps):
                times: dict[str, float] = {"obs_build": obs_build_s}
                x = np.asarray(obs, dtype=np.float32).reshape(-1)
                if feed is not None:
                    fp = cast(Any, policy)
                    start = time.perf_counter()
                    feed.advance_to(t0 + k * dt)
                    times["feed_advance"] = time.perf_counter() - start
                    if feed.clock_model_s != t0 + env.record.steps * dt:
                        raise ValueError(f"feed clock {feed.clock_model_s} off the control grid")
                    start = time.perf_counter()
                    block = forecast_block(fp.forecaster, feed, fp.leads_full_s)
                    times["forecaster"] = time.perf_counter() - start
                else:
                    block = None
                start = time.perf_counter()
                x_in = x if block is None else np.concatenate([x, block])
                (out,) = session.run(
                    [], {INPUT_NAME: np.ascontiguousarray(x_in.reshape(1, -1), dtype=np.float32)}
                )
                a_net = np.clip(np.asarray(out, dtype=np.float32).reshape(-1), -1.0, 1.0)
                times["policy_onnx"] = time.perf_counter() - start
                if residual:
                    rp = cast(ResidualPolicy, policy)
                    start = time.perf_counter()
                    a_base = np.asarray(rp.base.act(obs))
                    times["base_act"] = time.perf_counter() - start
                    start = time.perf_counter()
                    action = compose_residual(a_base, a_net, rp.alpha)
                    times["compose"] = time.perf_counter() - start
                else:
                    action = a_net
                if action.shape != (ACTION_DIM,):
                    raise ValueError(f"bad action {action!r}")
                start = time.perf_counter()
                obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float64))
                step_s = time.perf_counter() - start
                obs_build_s = timer.last_s
                times["physics_step"] = step_s - obs_build_s
                if record:
                    timings.append(StepTiming(times))
                if terminated or truncated:
                    break
            if record:
                outcomes.append(
                    EpisodeOutcome(
                        listed.ss,
                        listed.index,
                        str(env.record.as_row()["outcome"]),
                        int(env.record.steps),
                    )
                )
    finally:
        env.close()
    return timings, outcomes


def _row(
    head: Sequence[Any], category: str, component: str, values_s: Sequence[float], n_ep: int
) -> list[str]:
    """One ``e2e_budget.csv`` row from per-step seconds."""
    ms = np.asarray(values_s, dtype=np.float64) * 1e3
    p50, p90, p99 = (float(v) for v in np.percentile(ms, (50.0, 90.0, 99.0)))
    return [
        *(str(h) for h in head),
        category,
        component,
        str(ms.size),
        str(n_ep),
        f"{p50:.6g}",
        f"{p90:.6g}",
        f"{p99:.6g}",
        f"{float(ms.mean()):.6g}",
        f"{float(ms.max()):.6g}",
        f"{CONTROL_PERIOD_MS:.6g}",
        f"{p50 / CONTROL_PERIOD_MS:.6g}",
        f"{p99 / CONTROL_PERIOD_MS:.6g}",
    ]


def e2e_csv(
    runs: Sequence[tuple[str, int, str, Sequence[StepTiming], Sequence[EpisodeOutcome]]],
) -> str:
    """Render ``e2e_budget.csv``.

    Args:
        runs: ``(method, seed, architecture, timings, outcomes)`` per timed architecture.

    Returns:
        The CSV text: per run, every component present, the deployment sum, then the
        simulation-side components.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(E2E_COLUMNS)
    for method, seed, arch, timings, outcomes in runs:
        head = (method, seed, arch)
        n_ep = len(outcomes)
        for comp in DEPLOYMENT_COMPONENTS:
            vals = [t.times_s[comp] for t in timings if comp in t.times_s]
            if vals:
                writer.writerow(_row(head, "deployment", comp, vals, n_ep))
        writer.writerow(
            _row(head, "deployment", "deployment_sum", [t.deployment_s for t in timings], n_ep)
        )
        for comp in SIMULATION_COMPONENTS:
            vals = [t.times_s[comp] for t in timings if comp in t.times_s]
            if vals:
                writer.writerow(_row(head, "simulation", comp, vals, n_ep))
    return buffer.getvalue()
