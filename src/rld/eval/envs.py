"""The committed configs and the environment factory every evaluation worker uses.

One picklable bundle (:class:`EvalConfigs`) travels to each spawned worker, and one factory
(:func:`make_env`) builds a :class:`~rld.envs.landing_env.DeckLandingAviary` from it, so the
episode-list generator and the runner cannot construct the environment two different ways.
:func:`motion_for` maps a listed realization to its deck-motion source -- the dmf bridge for
a JONSWAP realization, :class:`~rld.envs.platform.StaticDeckMotion` for the static-pad list
-- and :func:`pad_offset_for` returns the pad's standing lever arm (zero for the static
fixture, whose deck point is the ship's CG by construction).

Motion kinds (Phase 7, P7-D1)
-----------------------------
:func:`motion_for` takes a ``kind`` (:data:`MOTION_KINDS`), defaulting to ``"jonswap"`` so
that every existing call builds exactly what it built before:

* ``"jonswap"`` -- the analytic dmf bridge (:class:`~rld.deck.bridge.JonswapDeckMotion`);
* ``"sinusoid"`` -- the H4 test leg: **the training builder itself**,
  :func:`rld.rl.motion.sinusoid_motion`, fed the realization's committed aft RMS from
  :func:`rld.rl.motion.committed_rms_full` and the listed **episode seed** for the phases
  (P7-D1 §1). The source has the JONSWAP source's window and full-scale lookback, so the
  listed start offset and initial state are reproduced and the strict start check holds;
* ``"mss"`` / ``"mss_corpus"`` -- the optional MSS transfer arm (P7-D1 §6), through
  ``deck-bridge-engineer``'s factory :func:`rld.deck.mss.mss_motion` (grid kinds ``mss`` and
  ``corpus``); the listed ``ss`` is the MSS label ``"SS5/mss:<grid>"`` (:func:`_mss_motion`).

Per-arm configs
---------------
:func:`with_noise` and :func:`with_lambda` return a copy of an :class:`EvalConfigs` with the
perception stand-in switched on at a stated sigma / latency (P7-D1 §4), or with another
Froude scale (P7-D1 §3), by ``dataclasses.replace``; the committed YAML files are not touched.

Units: lengths metres model scale, times seconds model scale, headings degrees, speeds knots
full scale. :mod:`rld.envs` is imported, never edited.
"""

import dataclasses
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dmf.config import SimConfig, load_sim
from dmf.sim.generate import RealizationSpec

from rld.deck.bridge import JonswapDeckMotion
from rld.deck.config import (
    MOTION_JONSWAP_CONFIG,
    PAD_CONFIG,
    SCALING_CONFIG,
    MotionConfig,
    PadConfig,
    ScalingConfig,
    load_motion,
    load_pads,
    load_scaling,
)
from rld.envs.config import (
    LANDING_CONFIG,
    NOISE_CONFIG,
    OBSERVATION_CONFIG,
    REWARD_CONFIG,
    SUCCESS_CONFIG,
    LandingConfig,
    NoiseConfig,
    ObservationConfig,
    RewardConfig,
    SuccessConfig,
    load_landing,
    load_noise,
    load_observation,
    load_reward,
    load_success,
)
from rld.envs.landing_env import DeckLandingAviary
from rld.envs.platform import EpisodeMotionSource, StaticDeckMotion, pad_offset_model_m

__all__ = [
    "MOTION_KINDS",
    "STATIC_LABEL",
    "EvalConfigs",
    "MotionKind",
    "load_eval_configs",
    "make_env",
    "motion_for",
    "pad_offset_for",
    "with_lambda",
    "with_noise",
]

#: Deck-motion models the runner can fly a listed realization under (P7-D1).
type MotionKind = Literal["jonswap", "sinusoid", "mss", "mss_corpus"]

#: Every :data:`MotionKind`, in documentation order.
MOTION_KINDS: tuple[str, ...] = ("jonswap", "sinusoid", "mss", "mss_corpus")

#: The MSS grid kind each MSS motion kind evaluates (P7-D1 §6: ``mss`` primary, ``corpus``
#: the attribution control).
_MSS_GRID: dict[str, str] = {"mss": "mss", "mss_corpus": "corpus"}

#: Label used for the regime, sea state and vessel of the static-pad list. It is also the
#: vessel field of :attr:`StaticDeckMotion.key`, so a row and its source agree.
STATIC_LABEL: str = "static"


@dataclass(frozen=True)
class EvalConfigs:
    """Every committed config an evaluation worker needs, as one picklable object.

    Attributes:
        sim: dmf's corpus config, full scale.
        scaling: Froude scale.
        pads: Pad geometry; ``radius_model_m`` is metres model scale.
        motion: Model-scale rates and the full-scale forecaster lookback.
        landing: The environment config.
        success: The frozen success criteria.
        observation: The observation config.
        noise: The perception stand-in config.
        reward: The reward weights (irrelevant to any evaluation metric, but the
            environment needs them to run).
    """

    sim: SimConfig
    scaling: ScalingConfig
    pads: PadConfig
    motion: MotionConfig
    landing: LandingConfig
    success: SuccessConfig
    observation: ObservationConfig
    noise: NoiseConfig
    reward: RewardConfig


def load_eval_configs(
    *,
    landing: Path = LANDING_CONFIG,
    success: Path = SUCCESS_CONFIG,
    observation: Path = OBSERVATION_CONFIG,
    noise: Path = NOISE_CONFIG,
    reward: Path = REWARD_CONFIG,
    scaling: Path = SCALING_CONFIG,
    pads: Path = PAD_CONFIG,
    motion: Path = MOTION_JONSWAP_CONFIG,
) -> EvalConfigs:
    """Load the committed configs, defaulting to the files under ``configs/``.

    Args:
        landing: ``configs/env/landing.yaml``.
        success: ``configs/env/success.yaml`` (frozen at Gate 3).
        observation: ``configs/env/observation.yaml``.
        noise: ``configs/env/noise.yaml``.
        reward: ``configs/env/reward.yaml``.
        scaling: ``configs/deck/scaling.yaml``.
        pads: ``configs/deck/pad.yaml``.
        motion: ``configs/deck/motion_jonswap.yaml``; its ``sim_config_path`` names dmf's
            corpus config.

    Returns:
        The :class:`EvalConfigs`.
    """
    motion_cfg = load_motion(motion)
    return EvalConfigs(
        sim=load_sim(motion_cfg.sim_config_path),
        scaling=load_scaling(scaling),
        pads=load_pads(pads),
        motion=motion_cfg,
        landing=load_landing(landing),
        success=load_success(success),
        observation=load_observation(observation),
        noise=load_noise(noise),
        reward=load_reward(reward),
    )


def motion_for(
    cfgs: EvalConfigs,
    vessel: str,
    ss: str,
    heading_deg: float,
    speed_kn: float,
    realization_seed: int,
    *,
    kind: MotionKind = "jonswap",
    episode_seed: int | None = None,
) -> EpisodeMotionSource:
    """Return the deck-motion source of one listed realization.

    Args:
        cfgs: The committed configs.
        vessel: dmf vessel config stem, or :data:`STATIC_LABEL` for the static pad.
        ss: Sea state label, e.g. ``"SS5"``.
        heading_deg: Encounter heading, degrees (180 head seas, 90 beam).
        speed_kn: Forward speed, knots, full scale.
        realization_seed: dmf realization seed ordinal within its cell.
        kind: The motion model (:data:`MOTION_KINDS`); ``"jonswap"`` by default.
        episode_seed: The listed episode seed; required for ``"sinusoid"`` (the three DOF
            phases are drawn from it, as in training), ignored otherwise.

    Returns:
        A :class:`~rld.deck.bridge.JonswapDeckMotion` (the analytic dmf bridge, evaluated on
        the physics grid), a :class:`~rld.envs.platform.StaticDeckMotion`, the matched
        :class:`~rld.deck.sinusoid.SinusoidDeckMotion`, or an MSS source.

    Raises:
        ValueError: On an unknown kind, a non-JONSWAP kind on the static pad (it has no ship
            to imitate), or a sinusoid without an episode seed.
    """
    if kind not in MOTION_KINDS:
        raise ValueError(f"unknown motion kind {kind!r}; expected one of {MOTION_KINDS}")
    if vessel == STATIC_LABEL:
        if kind != "jonswap":
            raise ValueError(f"the static pad has no ship motion to fly as {kind!r}")
        return StaticDeckMotion()
    spec = RealizationSpec(
        sea_state=ss,
        heading_deg=float(heading_deg),
        speed_kn=float(speed_kn),
        vessel=vessel,
        seed=int(realization_seed),
    )
    if kind == "sinusoid":
        if episode_seed is None:
            raise ValueError("a sinusoid source needs the listed episode seed (its phases)")
        return _sinusoid_motion(cfgs, spec, int(episode_seed))
    if kind in _MSS_GRID:
        return _mss_motion(cfgs, spec, _MSS_GRID[kind])
    return JonswapDeckMotion(
        spec,
        cfgs.sim,
        cfgs.scaling.froude_scale(),
        cfgs.pads,
        cfgs.motion.forecast_lookback_full_s,
    )


@functools.lru_cache(maxsize=4096)
def _committed_rms(
    ss: str, heading_deg: float, speed_kn: float, vessel: str, seed: int
) -> tuple[float, float, float]:
    """Return one realization's committed aft RMS, full scale (cached per worker process).

    Read by :func:`rld.rl.motion.committed_rms_full` -- the function the ``ppo_sinusoid``
    training run read its table with -- from ``results/deck_stats_seeds.csv``.
    """
    from dmf.data.splits import realization_key

    from rld.rl.motion import committed_rms_full  # imports this module; import lazily

    key = realization_key(ss, heading_deg, speed_kn, vessel, seed)
    return committed_rms_full([key])[key]


def _sinusoid_motion(
    cfgs: EvalConfigs, spec: RealizationSpec, episode_seed: int
) -> EpisodeMotionSource:
    """Build the H4 sinusoid test leg with the training builder (P7-D1 §1).

    Args:
        cfgs: The arm's configs.
        spec: The listed realization.
        episode_seed: The listed episode seed.

    Returns:
        ``rld.rl.motion.sinusoid_motion(cfgs, spec, episode_seed, rms)`` with ``rms`` the
        realization's committed aft RMS.
    """
    from rld.rl.motion import sinusoid_motion  # imports this module; import lazily

    rms = _committed_rms(
        spec.sea_state, float(spec.heading_deg), float(spec.speed_kn), spec.vessel, int(spec.seed)
    )
    return sinusoid_motion(cfgs, spec, episode_seed, rms)


def _mss_motion(cfgs: EvalConfigs, spec: RealizationSpec, grid: str) -> EpisodeMotionSource:
    """Build an MSS transfer source through ``rld.deck.mss.mss_motion`` (P7-D1 §6).

    An MSS list row carries ``ss = rld.deck.mss.mss_ss_label(grid_kind)`` (e.g.
    ``"SS5/mss:mss"``), so its realization key is the source's own key
    (:func:`rld.deck.mss.mss_key`), distinct from every dmf ``s175`` corpus key, and the
    runner's strict start check holds unchanged. The vessel stays ``"s175"``, so the pad
    lever arm and the feed work unchanged.

    Args:
        cfgs: The arm's configs.
        spec: The listed realization: vessel ``s175`` and an MSS sea-state label.
        grid: ``"mss"`` or ``"corpus"``, from the motion kind.

    Returns:
        The source.

    Raises:
        ValueError: If the row's sea-state label is not an MSS label of ``grid``.
    """
    from rld.deck.mss import mss_motion, parse_mss_ss_label  # dmf.mss; MSS arm only

    _, label_grid = parse_mss_ss_label(spec.sea_state)
    if label_grid != grid:
        raise ValueError(f"listed ss {spec.sea_state!r} is grid {label_grid!r}, flown as {grid!r}")
    return mss_motion(cfgs, grid, spec.heading_deg, spec.speed_kn, spec.seed)


def with_noise(
    cfgs: EvalConfigs,
    *,
    position_sigma_m: float,
    velocity_sigma_m_s: float,
    latency_ms: float,
    hold_freq_hz: float | None = None,
) -> EvalConfigs:
    """Return ``cfgs`` with the perception stand-in **enabled** at the given settings.

    Args:
        cfgs: The committed configs.
        position_sigma_m: Relative-position noise sigma, metres model scale.
        velocity_sigma_m_s: Relative-velocity noise sigma, metres per second model scale.
        latency_ms: Transport delay, milliseconds model scale; the environment quantises it
            **down** to whole control steps (33.4 ms -> 1 step at 30 Hz, 33 ms -> 0).
        hold_freq_hz: Sample-and-hold rate, hertz model scale; ``None`` keeps the committed
            value (30 Hz = the control rate = no hold).

    Returns:
        A copy with ``noise`` replaced; nothing else changes.
    """
    noise = dataclasses.replace(
        cfgs.noise,
        enabled=True,
        position_sigma_m=float(position_sigma_m),
        velocity_sigma_m_s=float(velocity_sigma_m_s),
        latency_ms=float(latency_ms),
        hold_freq_hz=cfgs.noise.hold_freq_hz if hold_freq_hz is None else float(hold_freq_hz),
    )
    return dataclasses.replace(cfgs, noise=noise)


def with_lambda(cfgs: EvalConfigs, lam_inverse: float) -> EvalConfigs:
    """Return ``cfgs`` at another Froude scale ``lambda = 1 / lam_inverse`` (P7-D1 §3).

    Everything that reads the scale -- the deck source, the pad lever arm, the controllers'
    ``ControlSpec`` and the ship-motion feed -- reads it from ``cfgs.scaling``, so this one
    replacement moves them all. The project lambda stays 1/25 (P1-D1); this is the
    sensitivity arm only.

    Args:
        cfgs: The committed configs.
        lam_inverse: ``1 / lambda``, dimensionless (15 or 40 in P7-D1).

    Returns:
        A copy with ``scaling.lam_inverse`` replaced.

    Raises:
        ValueError: If ``lam_inverse`` is not positive.
    """
    if not lam_inverse > 0.0:
        raise ValueError(f"lam_inverse must be positive, got {lam_inverse}")
    return dataclasses.replace(
        cfgs, scaling=dataclasses.replace(cfgs.scaling, lam_inverse=float(lam_inverse))
    )


def pad_offset_for(cfgs: EvalConfigs, vessel: str, pad: str) -> tuple[float, float, float]:
    """Return the pad's standing lever arm from the ship's CG, metres model scale.

    Args:
        cfgs: The committed configs.
        vessel: dmf vessel config stem, or :data:`STATIC_LABEL`.
        pad: Pad name, ``"aft"`` or ``"cg"``.

    Returns:
        ``(x, y, z)`` metres model scale, ship body frame; ``(0, 0, 0)`` for the static pad.
    """
    if vessel == STATIC_LABEL:
        return (0.0, 0.0, 0.0)
    return pad_offset_model_m(vessel, cfgs.pads, cfgs.scaling.froude_scale(), pad)


def make_env(
    cfgs: EvalConfigs,
    motion: EpisodeMotionSource,
    vessel: str,
    pad: str,
    episode_seed: int,
) -> DeckLandingAviary:
    """Build one landing environment from the committed configs.

    Args:
        cfgs: The committed configs.
        motion: The first episode's deck-motion source; re-point with
            :meth:`DeckLandingAviary.set_motion` between episodes rather than rebuilding.
        vessel: The source's vessel, for the pad lever arm.
        pad: Pad name.
        episode_seed: Constructor seed. Every evaluation reset passes an explicit seed, so
            this only affects a reset without one.

    Returns:
        The environment. The caller closes it.
    """
    return DeckLandingAviary(
        motion=motion,
        pad=pad,
        pad_radius_m=cfgs.pads.radius_model_m,
        pad_offset_m=pad_offset_for(cfgs, vessel, pad),
        cfg=cfgs.landing,
        success=cfgs.success,
        obs_cfg=cfgs.observation,
        noise_cfg=cfgs.noise,
        reward_cfg=cfgs.reward,
        episode_seed=episode_seed,
    )
