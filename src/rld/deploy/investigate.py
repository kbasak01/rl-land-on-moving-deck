"""Closed-loop parity investigation (P8-D3): what caused the three P8-D2 §4 outcome flips.

Arms (P8-D3 table), each the e05/e06 policy with only ``network_action`` replaced:

* ``torch`` -- SB3 float32, unchanged;
* ``onnx`` -- the graph on ORT CPU, 1 thread, default optimisations (P8-D2's arm);
* ``onnx_noopt`` -- ORT CPU, 1 thread, ``ORT_DISABLE_ALL``;
* ``onnx_cuda`` -- ORT CUDA, TF32 off;
* ``torch_folded`` -- the folded torch module, float32;
* ``torch_fp64`` -- a float64 reference (input, normaliser and actor in double);
* ``ulp_k`` (k = 1..20) -- SB3 float32 on the raw input with every entry nudged one float32 ulp
  up or down at random each step, ``default_rng(SeedSequence([20261005, k, episode_seed]))``
  re-created at every reset.

Analyses:

* **A1** (:func:`same_input_rows`) -- every recorded ``torch`` step of E50 re-evaluated at batch 1
  by SB3 (validity: bit-identical), ORT CPU, ORT CPU without optimisations and the folded module,
  with the input-clip and output-clip regimes flagged;
* **A2** (:func:`trace_summary_rows`) -- per-step divergence of ``onnx`` and ``torch_folded`` from
  ``torch`` on set T (``id`` SS6 index 0-11): first differing step, growth of the physics-state
  difference, largest single-step jump and its event;
* **C/D/E** (:func:`flip_rows`, :func:`noise_rows`) -- flip sets against ``torch`` and against
  ``torch_fp64`` on U = E50 ∪ ``id`` SS6, and the rounding-noise distribution on SS6;
* :func:`verdict_rows` applies P8-D3's fixed readings mechanically.

Units: positions metres and velocities metres per second, model scale (PyBullet physics state,
float64); tilt degrees; actions normalised in [-1, 1]; steps are 30 Hz control steps.
"""

import csv
import functools
import gzip
import io
import math
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from dmf.deploy.export_onnx import INPUT_NAME
from dmf.deploy.providers import disable_tf32, preload_gpu_libraries, session_providers
from dmf.typedefs import FloatArray

from rld.deploy.closed_loop import build_control_policy
from rld.deploy.export import folded_actor, normalizer_constants
from rld.deploy.onnx_policy import OnnxNetworkMixin, attach_onnx
from rld.envs.landing_env import ACTION_DIM
from rld.eval.envs import EvalConfigs, make_env, motion_for, pad_offset_for
from rld.eval.episodes import ListedEpisode
from rld.rl.forecast_obs import episode_feed
from rld.rl.policy import LearnedPolicy
from rld.rl.residual import ResidualPolicy, compose_residual

__all__ = [
    "BASE_ARMS",
    "K_ULP",
    "ULP_ENTROPY",
    "Trace",
    "build_variant",
    "flip_rows",
    "noise_rows",
    "same_input_rows",
    "trace_chunk",
    "trace_rows_gz",
    "trace_summary_rows",
    "ulp_arms",
    "verdict_rows",
    "write_csv",
]

#: Arms flown on U (P8-D3).
BASE_ARMS: tuple[str, ...] = (
    "torch",
    "onnx",
    "onnx_noopt",
    "onnx_cuda",
    "torch_folded",
    "torch_fp64",
)

#: Number of random-ulp arms (P8-D3: K = 20).
K_ULP: int = 20

#: First entropy word of every ``ulp_k`` generator.
ULP_ENTROPY: int = 20261005

#: Observation indices (``rld.envs.observation``): relative tilt (rad) and the contact flag.
REL_TILT_INDEX: int = 18
CONTACT_INDEX: int = 24

#: ‖Δp‖ thresholds whose first crossing step is reported, metres model scale.
DP_THRESHOLDS: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2)

#: Fit window of log10‖Δp‖ against step, metres model scale.
FIT_LO_M: float = 1e-8
FIT_HI_M: float = 1e-3


def ulp_arms() -> tuple[str, ...]:
    """``("ulp_1", ..., "ulp_20")``."""
    return tuple(f"ulp_{k}" for k in range(1, K_ULP + 1))


# --------------------------------------------------------------------------- variants


class _Fp64NetworkMixin:
    """``network_action`` in float64: input, normaliser and actor in double."""

    fp64_net: torch.nn.Module
    fp64_mean: FloatArray
    fp64_std: FloatArray
    fp64_clip: float

    def network_action(self, obs: FloatArray) -> FloatArray:
        """Return the float64 mean action clipped to ``[-1, 1]``, ``(3,)`` float64."""
        host = cast(Any, self)
        x = np.asarray(host.policy_input(obs), dtype=np.float64).reshape(1, -1)
        z = np.clip((x - self.fp64_mean) / self.fp64_std, -self.fp64_clip, self.fp64_clip)
        with torch.no_grad():
            out = self.fp64_net(torch.from_numpy(z)).numpy().reshape(-1)
        action: FloatArray = np.clip(out, -1.0, 1.0)
        return action


class _UlpRandomMixin:
    """SB3 float32 on the raw input nudged ±1 ulp per entry per step (seeded per episode)."""

    ulp_k: int
    _ulp_rng: np.random.Generator

    def reset(self, seed: int, context: Any = None, motion_feed: Any = None) -> None:
        """Re-create the generator from ``(ULP_ENTROPY, k, episode seed)``, then reset."""
        self._ulp_rng = np.random.default_rng(
            np.random.SeedSequence([ULP_ENTROPY, int(self.ulp_k), int(seed)])
        )
        cast(Any, super()).reset(seed, context, motion_feed)

    def network_action(self, obs: FloatArray) -> FloatArray:
        """Return SB3's deterministic action on the perturbed input, ``(3,)`` float32."""
        host = cast(Any, self)
        x = np.asarray(host.policy_input(obs), dtype=np.float32).reshape(1, -1)
        up = self._ulp_rng.random(x.shape) < 0.5
        x = np.where(
            up, np.nextafter(x, np.float32(np.inf)), np.nextafter(x, np.float32(-np.inf))
        ).astype(np.float32)
        z = np.asarray(host.normalizer.normalize_obs(x), dtype=np.float32)
        action, _ = host.model.predict(z, deterministic=True)
        out: FloatArray = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return out


def _session(onnx_path: Path, arm: str) -> Any:
    """ORT session of an ONNX arm, realized provider checked."""
    import onnxruntime as ort

    preload_gpu_libraries()
    disable_tf32()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if arm == "onnx_noopt":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers: tuple[str, ...] = ("CPUExecutionProvider",)
    elif arm == "onnx_cuda":
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    else:
        providers = ("CPUExecutionProvider",)
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=session_providers(providers)
    )
    if session.get_providers()[0] != providers[0]:
        raise ValueError(f"{arm}: realized {session.get_providers()}, wanted {providers[0]}")
    return session


def _reclass(policy: LearnedPolicy, mixin: type, tag: str) -> None:
    """Put ``mixin`` in front of the policy's class, in place."""
    base = type(policy)
    policy.__class__ = type(f"{tag}_{base.__name__}", (mixin, base), {})


def build_variant(cfgs: EvalConfigs, *, run_dir: Path, onnx_path: Path, arm: str) -> LearnedPolicy:
    """Picklable runner factory of every P8-D3 arm.

    Args:
        cfgs: The evaluation configs.
        run_dir: The run directory.
        onnx_path: The run's exported graph.
        arm: One of :data:`BASE_ARMS` or ``ulp_<k>``.

    Returns:
        The policy.

    Raises:
        ValueError: On an unknown arm.
    """
    from rld.rl.train import build_policy

    if arm == "torch_folded":
        return build_control_policy(cfgs, run_dir=run_dir, kind="torch_folded")
    policy = build_policy(cfgs, run_dir=run_dir)
    if arm == "torch":
        return policy
    if arm in {"onnx", "onnx_noopt", "onnx_cuda"}:
        twin = attach_onnx(policy, onnx_path)
        if arm != "onnx":
            cast(OnnxNetworkMixin, twin).onnx_session = _session(onnx_path, arm)
        return twin
    if arm == "torch_fp64":
        constants = normalizer_constants(policy)
        sb3 = cast(Any, policy.model.policy)
        net = torch.nn.Sequential(sb3.mlp_extractor.policy_net, sb3.action_net)
        import copy

        _reclass(policy, _Fp64NetworkMixin, "fp64")
        host = cast(_Fp64NetworkMixin, policy)
        host.fp64_net = copy.deepcopy(net).double().eval()
        host.fp64_mean = np.asarray(constants.mean, dtype=np.float64)
        host.fp64_std = constants.std
        host.fp64_clip = float(constants.clip_obs)
        return policy
    if arm.startswith("ulp_"):
        _reclass(policy, _UlpRandomMixin, arm)
        cast(_UlpRandomMixin, policy).ulp_k = int(arm.split("_", 1)[1])
        return policy
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------- traces


@dataclass
class Trace:
    """One traced episode.

    Attributes:
        method: Method label.
        seed: Training seed.
        arm: Arm name.
        ss: Sea state.
        index: Listed index.
        outcome: Outcome class.
        steps: Control steps executed.
        x: ``(T, D)`` float32 raw policy inputs.
        a_net: ``(T, 3)`` network actions (dtype as returned).
        pos: ``(T, 3)`` drone position **before** each act, float64, metres model scale.
        vel: ``(T, 3)`` drone velocity before each act, float64, metres per second.
        tilt_deg: ``(T,)`` relative tilt from the observation, degrees.
        contact: ``(T,)`` contact flag from the observation.
    """

    method: str
    seed: int
    arm: str
    ss: str
    index: int
    outcome: str = ""
    steps: int = 0
    x: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    a_net: FloatArray = field(default_factory=lambda: np.zeros((0, 3)))
    pos: FloatArray = field(default_factory=lambda: np.zeros((0, 3)))
    vel: FloatArray = field(default_factory=lambda: np.zeros((0, 3)))
    tilt_deg: FloatArray = field(default_factory=lambda: np.zeros(0))
    contact: FloatArray = field(default_factory=lambda: np.zeros(0))


def trace_chunk(task: tuple[Any, ...]) -> list[Trace]:
    """Fly and trace one arm on a chunk of episodes (``run_chunk``'s loop, ``act`` unrolled).

    ``ResidualPolicy.act`` is ``compose_residual(base.act(o), network_action(o), alpha)`` with
    the network evaluated first; the loop below does exactly that, so the executed actions are
    the runner's (checked against the runner rows by the caller).

    Args:
        task: ``(episodes, method, seed, run_dir, onnx_path, arm, cfgs)``.

    Returns:
        One :class:`Trace` per episode.
    """
    episodes, method, seed, run_dir, onnx_path, arm, cfgs = task
    policy = build_variant(cfgs, run_dir=run_dir, onnx_path=onnx_path, arm=arm)
    feed_needed = bool(getattr(policy, "needs_motion_feed", False))
    residual = isinstance(policy, ResidualPolicy)
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
    max_steps = int(round(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz)) + 1
    dt = float(env.cfg.ctrl_dt_s)
    out: list[Trace] = []
    try:
        for listed in episodes:
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
            if t0 != listed.t0_model_s:
                raise ValueError(f"{listed.ss}/{listed.index}: t0 differs from the list")
            feed = (
                episode_feed(env, cfgs.pads, cfgs.scaling.froude_scale()) if feed_needed else None
            )
            if feed is not None:
                policy.reset(listed.episode_seed, None, feed)
            else:
                policy.reset(listed.episode_seed)
            xs, nets, pos, vel, tilt, contact = [], [], [], [], [], []
            for k in range(max_steps):
                if feed is not None:
                    feed.advance_to(t0 + k * dt)
                xs.append(np.asarray(policy.policy_input(obs), dtype=np.float32).reshape(-1))
                pos.append(np.asarray(env.pos[0], dtype=np.float64).copy())
                vel.append(np.asarray(env.vel[0], dtype=np.float64).copy())
                tilt.append(math.degrees(float(obs[REL_TILT_INDEX])))
                contact.append(float(obs[CONTACT_INDEX]))
                a_net = np.asarray(policy.network_action(obs))
                nets.append(a_net.astype(np.float64))
                if residual:
                    rp = cast(ResidualPolicy, policy)
                    action = compose_residual(np.asarray(rp.base.act(obs)), a_net, rp.alpha)
                else:
                    action = a_net
                if np.asarray(action).shape != (ACTION_DIM,):
                    raise ValueError(f"bad action {action!r}")
                obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float64))
                if terminated or truncated:
                    break
            out.append(
                Trace(
                    method,
                    seed,
                    arm,
                    listed.ss,
                    listed.index,
                    str(env.record.as_row()["outcome"]),
                    int(env.record.steps),
                    np.stack(xs),
                    np.stack(nets),
                    np.stack(pos),
                    np.stack(vel),
                    np.asarray(tilt),
                    np.asarray(contact),
                )
            )
    finally:
        env.close()
    return out


def run_traces(
    graphs: Sequence[tuple[str, int, Path, Path]],
    arms: Sequence[str],
    episodes: Sequence[ListedEpisode],
    cfgs: EvalConfigs,
    *,
    workers: int,
    chunk: int = 10,
) -> list[Trace]:
    """Trace every (graph, arm) on ``episodes`` through a spawn pool; order is deterministic."""
    tasks = [
        (episodes[i : i + chunk], m, s, r, o, arm, cfgs)
        for m, s, r, o in graphs
        for arm in arms
        for i in range(0, len(episodes), chunk)
    ]
    with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
        results = list(pool.imap(trace_chunk, tasks, chunksize=1))
    return [t for chunk_traces in results for t in chunk_traces]


# --------------------------------------------------------------------------- A1


def same_input_rows(
    graphs: Sequence[tuple[str, int, Path, Path]], traces: Sequence[Trace], cfgs: EvalConfigs
) -> list[list[str]]:
    """A1 and B: re-evaluate each recorded ``torch`` input at batch 1 by every comparator.

    Args:
        graphs: ``(method, seed, run_dir, onnx_path)``.
        traces: The ``torch`` traces (E50).
        cfgs: The evaluation configs.

    Returns:
        Rows of ``same_input.csv``.
    """
    from rld.rl.train import build_policy

    rows: list[list[str]] = []
    for method, seed, run_dir, onnx_path in graphs:
        policy = build_policy(cfgs, run_dir=run_dir)
        constants = normalizer_constants(policy)
        folded = folded_actor(policy)
        sessions = {
            "onnx": _session(onnx_path, "onnx"),
            "onnx_noopt": _session(onnx_path, "onnx_noopt"),
        }
        mine = [t for t in traces if t.method == method and t.seed == seed and t.arm == "torch"]
        x = np.concatenate([t.x for t in mine])
        recorded = np.concatenate([t.a_net for t in mine]).astype(np.float32)
        z = (x.astype(np.float64) - constants.mean) / constants.std
        in_clip = np.any(np.abs(z) > constants.clip_obs, axis=1)
        with torch.no_grad():
            zc = torch.from_numpy(
                np.clip(z, -constants.clip_obs, constants.clip_obs).astype(np.float32)
            )
            mean = folded.action_net(folded.policy_net(zc)).numpy()
        out_clip = np.any(np.abs(mean) >= 1.0, axis=1)
        outputs: dict[str, np.ndarray] = {}
        assert policy.normalizer is not None
        sb3 = []
        for row in x:
            zz = np.asarray(policy.normalizer.normalize_obs(row.reshape(1, -1)), dtype=np.float32)
            a, _ = policy.model.predict(zz, deterministic=True)
            sb3.append(np.clip(np.asarray(a, dtype=np.float32).reshape(-1), -1.0, 1.0))
        outputs["sb3"] = np.stack(sb3)
        for name, session in sessions.items():
            outputs[name] = np.stack(
                [
                    np.clip(session.run([], {INPUT_NAME: row.reshape(1, -1)})[0].reshape(-1), -1, 1)
                    for row in x
                ]
            )
        with torch.no_grad():
            outputs["torch_folded"] = np.stack(
                [folded(torch.from_numpy(row.reshape(1, -1))).numpy().reshape(-1) for row in x]
            )
        for name, values in outputs.items():
            d = np.max(np.abs(values.astype(np.float64) - recorded.astype(np.float64)), axis=1)
            identical = np.all(values == recorded, axis=1)

            def _mx(mask: np.ndarray, d: np.ndarray = d) -> str:
                return repr(float(d[mask].max())) if mask.any() else "nan"

            rows.append(
                [
                    method,
                    str(seed),
                    name,
                    str(len(d)),
                    str(int(identical.sum())),
                    repr(float(d.max())),
                    repr(float(np.percentile(d, 99))),
                    str(int(in_clip.sum())),
                    _mx(in_clip),
                    str(int(out_clip.sum())),
                    _mx(out_clip),
                    _mx(~in_clip & ~out_clip),
                ]
            )
    return rows


SAME_INPUT_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "comparator",
    "n_steps",
    "n_bit_identical",
    "max_abs_d_action",
    "p99_abs_d_action",
    "n_steps_input_clip",
    "max_abs_d_input_clip",
    "n_steps_output_clip",
    "max_abs_d_output_clip",
    "max_abs_d_no_clip",
)


# --------------------------------------------------------------------------- A2


TRACE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "ss",
    "index",
    "pair",
    "outcome_torch",
    "outcome_other",
    "flipped",
    "steps_torch",
    "steps_other",
    "k0_first_action_diff",
    "abs_d_action_at_k0",
    *(f"step_dp_gt_{t:.0e}" for t in DP_THRESHOLDS),
    "fit_slope_log10_per_step",
    "fit_efold_steps",
    "fit_r2",
    "fit_n_points",
    "max_jump_decades",
    "max_jump_step",
    "max_jump_event",
    "touchdown_step_torch",
    "touchdown_in_fit_window",
    "d_pos_at_touchdown_m",
)


def _pair_arrays(a: Trace, b: Trace) -> dict[str, np.ndarray]:
    """Aligned per-step differences of two traces of the same episode."""
    n = min(len(a.pos), len(b.pos))
    return {
        "d_a": np.max(np.abs(a.a_net[:n] - b.a_net[:n]), axis=1),
        "d_pos": np.linalg.norm(a.pos[:n] - b.pos[:n], axis=1),
        "d_vel": np.linalg.norm(a.vel[:n] - b.vel[:n], axis=1),
        "d_tilt": np.abs(a.tilt_deg[:n] - b.tilt_deg[:n]),
        "contact_a": a.contact[:n],
        "contact_b": b.contact[:n],
    }


def trace_summary_rows(
    traces: Sequence[Trace],
    t_indices: Sequence[int],
    cfgs: EvalConfigs,
    constants_by: Mapping[tuple[str, int], Any],
) -> tuple[list[list[str]], list[list[str]]]:
    """A2: divergence of ``onnx`` and ``torch_folded`` from ``torch`` on set T.

    Args:
        traces: All traces (E50 × arms).
        t_indices: SS6 indices of set T.
        cfgs: The configs (unused beyond provenance; kept for symmetry).
        constants_by: ``{(method, seed): NormalizerConstants}`` for the clip flags.

    Returns:
        ``(summary rows, per-step rows)``.
    """
    del cfgs
    by = {(t.method, t.seed, t.arm, t.ss, t.index): t for t in traces}
    summary: list[list[str]] = []
    steps_rows: list[list[str]] = []
    keys = sorted({(t.method, t.seed) for t in traces}, key=lambda k: (k[0], k[1]))
    for method, seed in keys:
        consts = constants_by[(method, seed)]
        for index in t_indices:
            ref = by[(method, seed, "torch", "SS6", index)]
            z = (ref.x.astype(np.float64) - consts.mean) / consts.std
            in_clip = np.any(np.abs(z) > consts.clip_obs, axis=1)
            out_clip = np.any(np.abs(ref.a_net) >= 1.0, axis=1)
            for other_arm in ("onnx", "torch_folded"):
                other = by[(method, seed, other_arm, "SS6", index)]
                arr = _pair_arrays(ref, other)
                d_a, d_pos = arr["d_a"], arr["d_pos"]
                nz = np.nonzero(d_a > 0)[0]
                k0 = int(nz[0]) if nz.size else -1
                crossings = []
                for thr in DP_THRESHOLDS:
                    hit = np.nonzero(d_pos > thr)[0]
                    crossings.append(str(int(hit[0])) if hit.size else "")
                lo = np.nonzero(d_pos > FIT_LO_M)[0]
                hi = np.nonzero(d_pos > FIT_HI_M)[0]
                slope = efold = r2 = float("nan")
                n_fit = 0
                if lo.size:
                    end = int(hi[0]) if hi.size else len(d_pos) - 1
                    idx = np.arange(int(lo[0]), end + 1)
                    idx = idx[d_pos[idx] > 0]
                    n_fit = int(idx.size)
                    if n_fit >= 3:
                        yv = np.log10(d_pos[idx])
                        coef = np.polyfit(idx.astype(np.float64), yv, 1)
                        slope = float(coef[0])
                        pred = np.polyval(coef, idx)
                        ss_res = float(np.sum((yv - pred) ** 2))
                        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                        efold = 1.0 / (slope * math.log(10)) if slope > 0 else float("nan")
                with np.errstate(divide="ignore"):
                    logp = np.where(d_pos > 0, np.log10(np.maximum(d_pos, 1e-300)), np.nan)
                jumps = np.diff(logp)
                jump_step = -1
                jump = float("nan")
                event = "none"
                if np.any(np.isfinite(jumps)):
                    j = int(np.nanargmax(jumps))
                    jump, jump_step = float(jumps[j]), j + 1
                    s = jump_step
                    onset = (arr["contact_a"][s] > 0 and arr["contact_a"][s - 1] == 0) or (
                        arr["contact_b"][s] > 0 and arr["contact_b"][s - 1] == 0
                    )
                    if onset:
                        event = "contact_onset"
                    elif s - 1 < len(in_clip) and (in_clip[s - 1] or out_clip[s - 1]):
                        event = "clip_active"
                touch = np.nonzero(ref.contact > 0)[0]
                td = int(touch[0]) if touch.size else -1
                in_window = bool(
                    td >= 0 and lo.size and td <= (int(hi[0]) if hi.size else len(d_pos) - 1)
                )
                summary.append(
                    [
                        method,
                        str(seed),
                        "SS6",
                        str(index),
                        f"torch_vs_{other_arm}",
                        ref.outcome,
                        other.outcome,
                        str(ref.outcome != other.outcome),
                        str(ref.steps),
                        str(other.steps),
                        str(k0),
                        repr(float(d_a[k0])) if k0 >= 0 else "nan",
                        *crossings,
                        repr(slope),
                        repr(efold),
                        repr(r2),
                        str(n_fit),
                        repr(jump),
                        str(jump_step),
                        event,
                        str(td),
                        str(in_window),
                        repr(float(d_pos[td])) if 0 <= td < len(d_pos) else "nan",
                    ]
                )
                for s in range(len(d_a)):
                    steps_rows.append(
                        [
                            method,
                            str(seed),
                            str(index),
                            f"torch_vs_{other_arm}",
                            str(s),
                            repr(float(d_a[s])),
                            repr(float(d_pos[s])),
                            repr(float(arr["d_vel"][s])),
                            repr(float(arr["d_tilt"][s])),
                            str(int(arr["contact_a"][s])),
                            str(int(arr["contact_b"][s])),
                            str(bool(in_clip[s])) if s < len(in_clip) else "",
                            str(bool(out_clip[s])) if s < len(out_clip) else "",
                        ]
                    )
    return summary, steps_rows


TRACE_STEP_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "index",
    "pair",
    "step",
    "abs_d_action",
    "d_pos_m",
    "d_vel_m_s",
    "d_tilt_deg",
    "contact_torch",
    "contact_other",
    "input_clip_torch",
    "output_clip_torch",
)


def trace_rows_gz(rows: Sequence[Sequence[str]]) -> bytes:
    """Gzip the per-step trace CSV deterministically (``mtime = 0``)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(TRACE_STEP_COLUMNS)
    writer.writerows(rows)
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buffer.getvalue().encode("utf-8"))
    return raw.getvalue()


# --------------------------------------------------------------------------- C/D/E


OUTCOME_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "arm",
    "regime",
    "ss",
    "index",
    "episode_seed",
    "outcome",
    "steps",
    "rel_vz_normal_m_s",
    "closing_speed_normal_m_s",
    "rel_tilt_deg",
)


def outcome_row(method: str, seed: int, arm: str, row: Mapping[str, Any]) -> list[str]:
    """One ``outcomes.csv`` row from a runner row."""

    def f(v: Any) -> str:
        return repr(float("nan") if v is None else float(v))

    return [
        method,
        str(seed),
        arm,
        str(row["regime"]),
        str(row["ss"]),
        str(row["index"]),
        str(row["episode_seed"]),
        str(row["outcome"]),
        str(row["steps"]),
        f(row["rel_vz_normal_m_s"]),
        f(row["closing_speed_normal_m_s"]),
        f(row["rel_tilt_deg"]),
    ]


Outcomes = dict[tuple[str, int, str], dict[tuple[str, int], str]]


def _flips(
    a: Mapping[tuple[str, int], str],
    b: Mapping[tuple[str, int], str],
    keys: Sequence[tuple[str, int]],
) -> list[tuple[str, int]]:
    return [k for k in keys if k in a and k in b and a[k] != b[k]]


def _fmt_eps(
    eps: Sequence[tuple[str, int]],
    a: Mapping[tuple[str, int], str],
    b: Mapping[tuple[str, int], str],
) -> str:
    return " ".join(f"{ss}/{i}:{b[(ss, i)]}->{a[(ss, i)]}" for ss, i in eps)


FLIP_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "episode_set",
    "arm",
    "reference",
    "n_episodes",
    "n_flips",
    "n_flips_in_S",
    "success_arm",
    "success_reference",
    "flipped_episodes",
)


def flip_rows(
    outcomes: Outcomes,
    sets: Mapping[str, Sequence[tuple[str, int]]],
    sensitive: Mapping[tuple[str, int], set[tuple[str, int]]],
) -> list[list[str]]:
    """Flip sets of every base arm against ``torch`` and against ``torch_fp64``, per set."""
    rows: list[list[str]] = []
    policies = sorted({(m, s) for m, s, _ in outcomes})
    for method, seed in policies:
        for set_name, keys in sets.items():
            for ref_arm in ("torch", "torch_fp64"):
                ref = outcomes[(method, seed, ref_arm)]
                for arm in BASE_ARMS:
                    if arm == ref_arm or (method, seed, arm) not in outcomes:
                        continue
                    got = outcomes[(method, seed, arm)]
                    fl = _flips(got, ref, keys)
                    sens = sensitive.get((method, seed), set())
                    rows.append(
                        [
                            method,
                            str(seed),
                            set_name,
                            arm,
                            ref_arm,
                            str(len(keys)),
                            str(len(fl)),
                            str(sum(1 for e in fl if e in sens)),
                            str(sum(1 for k in keys if got.get(k) == "success")),
                            str(sum(1 for k in keys if ref.get(k) == "success")),
                            _fmt_eps(fl, got, ref),
                        ]
                    )
    return rows


NOISE_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "n_episodes",
    "onnx_flips",
    "ulp_flips_by_k",
    "ulp_flips_min",
    "ulp_flips_median",
    "ulp_flips_max",
    "onnx_percentile_rank",
    "n_sensitive_S",
    "onnx_flips_in_S",
    "onnx_flips_outside_S",
    "onnx_success_delta",
    "ulp_success_delta_by_k",
    "sensitive_episodes",
)


def sensitive_sets(
    outcomes: Outcomes, ss6: Sequence[tuple[str, int]]
) -> dict[tuple[str, int], set[tuple[str, int]]]:
    """S per policy: SS6 episodes that flip against ``torch`` under at least one ``ulp_k``."""
    out: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for method, seed in sorted({(m, s) for m, s, _ in outcomes}):
        ref = outcomes[(method, seed, "torch")]
        sens: set[tuple[str, int]] = set()
        for arm in ulp_arms():
            sens.update(_flips(outcomes[(method, seed, arm)], ref, ss6))
        out[(method, seed)] = sens
    return out


def _rank(value: int, dist: Sequence[int]) -> float:
    """Mid-rank percentile of ``value`` in ``dist`` (fraction below + half the ties)."""
    below = sum(1 for d in dist if d < value)
    ties = sum(1 for d in dist if d == value)
    return (below + 0.5 * ties) / len(dist)


def noise_rows(
    outcomes: Outcomes,
    ss6: Sequence[tuple[str, int]],
    sensitive: Mapping[tuple[str, int], set[tuple[str, int]]],
) -> list[list[str]]:
    """Arm C per policy, plus a pooled row (counts summed over policies per k)."""
    rows: list[list[str]] = []
    pooled_onnx = 0
    pooled_k = [0] * K_ULP
    pooled_in = pooled_out = pooled_s_size = 0
    pooled_sd = 0
    pooled_sdk = [0] * K_ULP
    policies = sorted({(m, s) for m, s, _ in outcomes})
    for method, seed in policies:
        ref = outcomes[(method, seed, "torch")]
        onnx = outcomes[(method, seed, "onnx")]
        fl = _flips(onnx, ref, ss6)
        counts = [len(_flips(outcomes[(method, seed, a)], ref, ss6)) for a in ulp_arms()]
        succ_ref = sum(1 for k in ss6 if ref[k] == "success")
        sd = sum(1 for k in ss6 if onnx[k] == "success") - succ_ref
        sdk = [
            sum(1 for k in ss6 if outcomes[(method, seed, a)][k] == "success") - succ_ref
            for a in ulp_arms()
        ]
        sens = sensitive[(method, seed)]
        n_in = sum(1 for e in fl if e in sens)
        rows.append(
            [
                method,
                str(seed),
                str(len(ss6)),
                str(len(fl)),
                "|".join(str(c) for c in counts),
                str(min(counts)),
                repr(float(np.median(counts))),
                str(max(counts)),
                repr(_rank(len(fl), counts)),
                str(len(sens)),
                str(n_in),
                str(len(fl) - n_in),
                str(sd),
                "|".join(str(v) for v in sdk),
                " ".join(f"{ss}/{i}" for ss, i in sorted(sens, key=lambda e: e[1])),
            ]
        )
        pooled_onnx += len(fl)
        pooled_k = [a + b for a, b in zip(pooled_k, counts, strict=True)]
        pooled_in += n_in
        pooled_out += len(fl) - n_in
        pooled_s_size += len(sens)
        pooled_sd += sd
        pooled_sdk = [a + b for a, b in zip(pooled_sdk, sdk, strict=True)]
    rows.append(
        [
            "pooled",
            "",
            str(len(ss6) * len(policies)),
            str(pooled_onnx),
            "|".join(str(c) for c in pooled_k),
            str(min(pooled_k)),
            repr(float(np.median(pooled_k))),
            str(max(pooled_k)),
            repr(_rank(pooled_onnx, pooled_k)),
            str(pooled_s_size),
            str(pooled_in),
            str(pooled_out),
            str(pooled_sd),
            "|".join(str(v) for v in pooled_sdk),
            "",
        ]
    )
    return rows


# --------------------------------------------------------------------------- readings


VERDICT_COLUMNS: tuple[str, ...] = ("reading", "condition", "value", "met")


def verdict_rows(
    same_input: Sequence[Sequence[str]],
    trace_summary: Sequence[Sequence[str]],
    noise: Sequence[Sequence[str]],
) -> tuple[str, list[list[str]]]:
    """Apply P8-D3's fixed readings mechanically.

    Returns:
        ``(reading, rows)``: ``"export defect"``, ``"rounding sensitivity"`` or
        ``"inconclusive"``, and one row per condition.
    """
    rows: list[list[str]] = []
    onnx_rows = [r for r in same_input if r[2] == "onnx"]

    def fl(v: str) -> float:
        return float(v) if v not in {"", "nan"} else float("nan")

    a1_max = max(fl(r[5]) for r in onnx_rows)
    clip_max = max(
        [fl(r[8]) for r in onnx_rows if r[8] != "nan"]
        + [fl(r[10]) for r in onnx_rows if r[10] != "nan"],
        default=float("nan"),
    )
    noclip_max = max(fl(r[11]) for r in onnx_rows if r[11] != "nan")
    onnx_pairs = [r for r in trace_summary if r[4] == "torch_vs_onnx"]
    flipped = [r for r in onnx_pairs if r[7] == "True"]
    k0_flipped = max((fl(r[11]) for r in flipped if r[11] != "nan"), default=float("nan"))
    k0_all = max((fl(r[11]) for r in onnx_pairs if r[11] != "nan"), default=float("nan"))
    # Growth: d_pos crossed 1e-8 and 1e-4 (>= 4 decades) before touchdown, for flipped episodes.
    i8 = TRACE_SUMMARY_COLUMNS.index("step_dp_gt_1e-08")
    i4 = TRACE_SUMMARY_COLUMNS.index("step_dp_gt_1e-04")
    td_i = TRACE_SUMMARY_COLUMNS.index("touchdown_step_torch")
    grow = [
        r[i8] != "" and r[i4] != "" and (r[td_i] == "-1" or int(r[i4]) <= int(r[td_i]))
        for r in flipped
    ]
    per_policy = [r for r in noise if r[0] != "pooled"]
    pooled = next(r for r in noise if r[0] == "pooled")
    over_max = [f"{r[0]} s{r[1]}" for r in per_policy if int(r[3]) > int(r[7])]
    outside = int(pooled[11])

    defect = [
        ("A1 max |da| > 1e-4 on any flown step", repr(a1_max), a1_max > 1e-4),
        (
            "A1 clip-regime max > 1e-5 and > 10x the unclipped max",
            f"{clip_max!r} vs {noclip_max!r}",
            bool(clip_max > 1e-5 and clip_max > 10 * noclip_max),
        ),
        ("A2 |da(k0)| > 1e-5 in a flipped episode", repr(k0_flipped), bool(k0_flipped > 1e-5)),
        (
            "C flips(onnx) > max_k flips(ulp_k) for some policy",
            ",".join(over_max) or "none",
            bool(over_max),
        ),
        ("C pooled ONNX flips outside S >= 2", str(outside), outside >= 2),
    ]
    rounding = [
        ("A1 max |da| <= 1e-5 in every regime", repr(a1_max), a1_max <= 1e-5),
        ("A2 |da(k0)| <= 1e-6 in flipped and control episodes", repr(k0_all), k0_all <= 1e-6),
        (
            "A2 d_pos grows >= 4 decades before touchdown (flipped)",
            str(sum(grow)) + f"/{len(grow)}",
            all(grow) and bool(grow),
        ),
        (
            "C every policy flips(onnx) <= max_k flips(ulp_k)",
            ",".join(over_max) or "none",
            not over_max,
        ),
        ("C pooled ONNX flips outside S <= 1", str(outside), outside <= 1),
    ]
    for cond, value, met in defect:
        rows.append(["export defect", cond, value, str(bool(met))])
    for cond, value, met in rounding:
        rows.append(["rounding sensitivity", cond, value, str(bool(met))])
    if any(m for _, _, m in defect):
        reading = "export defect"
    elif all(m for _, _, m in rounding):
        reading = "rounding sensitivity"
    else:
        reading = "inconclusive"
    rows.append(["READING", "P8-D3 fixed readings applied", reading, ""])
    return reading, rows


def write_csv(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a CSV deterministically."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


def make_specs(
    graphs: Sequence[tuple[str, int, Path, Path]], arm: str
) -> list[tuple[str, int, Any]]:
    """Runner specs of one arm for every graph."""
    from rld.eval.learned import inspect_run
    from rld.eval.runner import callable_spec

    out = []
    for method, seed, run_dir, onnx_path in graphs:
        run = inspect_run(method, run_dir)
        spec = callable_spec(
            method,
            functools.partial(build_variant, run_dir=run.run_dir, onnx_path=onnx_path, arm=arm),
            run_seed=seed,
            needs_motion_feed=run.needs_motion_feed,
        )
        out.append((method, seed, spec))
    return out
