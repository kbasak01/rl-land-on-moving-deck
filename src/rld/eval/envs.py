"""The committed configs and the environment factory every evaluation worker uses.

One picklable bundle (:class:`EvalConfigs`) travels to each spawned worker, and one factory
(:func:`make_env`) builds a :class:`~rld.envs.landing_env.DeckLandingAviary` from it, so the
episode-list generator and the runner cannot construct the environment two different ways.
:func:`motion_for` maps a listed realization to its deck-motion source -- the dmf bridge for
a JONSWAP realization, :class:`~rld.envs.platform.StaticDeckMotion` for the static-pad list
-- and :func:`pad_offset_for` returns the pad's standing lever arm (zero for the static
fixture, whose deck point is the ship's CG by construction).

Units: lengths metres model scale, times seconds model scale, headings degrees, speeds knots
full scale. :mod:`rld.envs` is imported, never edited.
"""

from dataclasses import dataclass
from pathlib import Path

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
    "STATIC_LABEL",
    "EvalConfigs",
    "load_eval_configs",
    "make_env",
    "motion_for",
    "pad_offset_for",
]

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
) -> EpisodeMotionSource:
    """Return the deck-motion source of one listed realization.

    Args:
        cfgs: The committed configs.
        vessel: dmf vessel config stem, or :data:`STATIC_LABEL` for the static pad.
        ss: Sea state label, e.g. ``"SS5"``.
        heading_deg: Encounter heading, degrees (180 head seas, 90 beam).
        speed_kn: Forward speed, knots, full scale.
        realization_seed: dmf realization seed ordinal within its cell.

    Returns:
        A :class:`~rld.deck.bridge.JonswapDeckMotion` (the analytic dmf bridge, evaluated on
        the physics grid) or a :class:`~rld.envs.platform.StaticDeckMotion`.
    """
    if vessel == STATIC_LABEL:
        return StaticDeckMotion()
    spec = RealizationSpec(
        sea_state=ss,
        heading_deg=float(heading_deg),
        speed_kn=float(speed_kn),
        vessel=vessel,
        seed=int(realization_seed),
    )
    return JonswapDeckMotion(
        spec,
        cfgs.sim,
        cfgs.scaling.froude_scale(),
        cfgs.pads,
        cfgs.motion.forecast_lookback_full_s,
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
