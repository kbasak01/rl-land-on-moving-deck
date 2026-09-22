"""Shared pytest fixtures.

Carries the suite's base seed, the two paths every phase resolves against, the deck-motion
configs (Phase 1) and the landing-environment configs, the PyBullet client and the
environment factory (Phase 2).

The deck fixtures are the committed configs under ``configs/deck/``, loaded once per session
because they are frozen dataclasses: ``deck_scaling`` (lambda, g, the feasibility rule),
``deck_pads`` (pad offsets as fractions of the vessel's full-scale length), ``deck_motion_cfg``
(model-scale physics/control rates and the full-scale forecaster lookback) and ``sim_cfg``
(dmf's own corpus config, full scale). ``deck_source`` builds a bridge for one realization.

Every random draw in the test suite must come from a seeded generator derived from
:data:`BASE_SEED`. A test calling ``np.random.default_rng()`` with no seed is a
reproducibility hole and should be rejected in review.

Unit conventions follow the package convention in :mod:`rld`: seconds, metres, degrees at
boundaries, and model-scale versus full-scale named for every time and length.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
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

#: Base seed for the whole test suite. Individual streams are ``BASE_SEED + offset`` so
#: that two fixtures never silently share a stream.
BASE_SEED = 20260921

#: Repository root, resolved from this file so tests do not depend on the working
#: directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The ``deck-motion-forecast`` submodule. Its configs are resolved by ``dmf`` relative to
#: its own source tree, which is why it must be an editable install.
DMF_ROOT = REPO_ROOT / "third_party" / "deck-motion-forecast"


@pytest.fixture
def rng() -> np.random.Generator:
    """Return the suite's default seeded generator.

    Returns:
        A ``numpy.random.Generator`` seeded from :data:`BASE_SEED`, function-scoped, so
        test ordering cannot change any result.
    """
    return np.random.default_rng(BASE_SEED)


@pytest.fixture(scope="session")
def deck_motion_cfg() -> MotionConfig:
    """Return the committed JONSWAP motion config.

    Returns:
        ``configs/deck/motion_jonswap.yaml``: physics and control rates in hertz at **model**
        scale, forecaster rate in hertz at **full** scale.
    """
    return load_motion(MOTION_JONSWAP_CONFIG)


@pytest.fixture(scope="session")
def sim_cfg(deck_motion_cfg: MotionConfig) -> SimConfig:
    """Return dmf's corpus config, the one the committed corpus was generated from.

    Returns:
        The :class:`dmf.config.SimConfig`. ``spinup_s`` 120 s, ``duration_s`` 600 s and
        ``fs_hz`` 10 Hz are all **full** scale.
    """
    return load_sim(deck_motion_cfg.sim_config_path)


@pytest.fixture(scope="session")
def deck_scaling() -> ScalingConfig:
    """Return the committed Froude-scaling config (lambda = 1/25 by default)."""
    return load_scaling(SCALING_CONFIG)


@pytest.fixture(scope="session")
def deck_pads() -> PadConfig:
    """Return the committed pad geometry (``aft`` at -0.4 L, ``cg`` at the origin)."""
    return load_pads(PAD_CONFIG)


@pytest.fixture
def deck_source(
    sim_cfg: SimConfig,
    deck_scaling: ScalingConfig,
    deck_pads: PadConfig,
    deck_motion_cfg: MotionConfig,
) -> Callable[[RealizationSpec], JonswapDeckMotion]:
    """Return a factory building a JONSWAP deck-motion source for one realization.

    Returns:
        A callable ``(spec) -> JonswapDeckMotion`` wired to the committed configs, with the
        forecaster's 20 s full-scale lookback reserved.
    """

    def build(spec: RealizationSpec) -> JonswapDeckMotion:
        return JonswapDeckMotion(
            spec,
            sim_cfg,
            deck_scaling.froude_scale(),
            deck_pads,
            lookback_full_s=deck_motion_cfg.forecast_lookback_full_s,
        )

    return build


# ---------------------------------------------------------------------------------
# Phase 2: the landing environment.
#
# The env configs are the committed files under ``configs/env/`` and are session-scoped
# because they are frozen dataclasses. ``pyb_client`` is function-scoped and always torn
# down: a leaked DIRECT client survives the test that opened it and the next test's
# ``getContactPoints`` then reads someone else's world.
# ---------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def env_landing_cfg() -> LandingConfig:
    """Return ``configs/env/landing.yaml``.

    Rates in hertz model scale, episode geometry in seconds model scale, ``v_max`` in metres
    per second model scale, and the deck plate geometry in metres model scale.
    """
    return load_landing(LANDING_CONFIG)


@pytest.fixture(scope="session")
def env_success_cfg() -> SuccessConfig:
    """Return ``configs/env/success.yaml``, the frozen success criteria."""
    return load_success(SUCCESS_CONFIG)


@pytest.fixture(scope="session")
def env_obs_cfg() -> ObservationConfig:
    """Return ``configs/env/observation.yaml``."""
    return load_observation(OBSERVATION_CONFIG)


@pytest.fixture(scope="session")
def env_noise_cfg() -> NoiseConfig:
    """Return ``configs/env/noise.yaml`` (noise off by default)."""
    return load_noise(NOISE_CONFIG)


@pytest.fixture(scope="session")
def env_reward_cfg() -> RewardConfig:
    """Return ``configs/env/reward.yaml``."""
    return load_reward(REWARD_CONFIG)


@pytest.fixture
def pyb_client() -> Iterator[int]:
    """Yield a headless PyBullet client id, gravity and timestep set, torn down after.

    Gravity is gym-pybullet-drones' 9.8 m/s^2 and the timestep is the committed model-scale
    physics step, so a platform test sees the same world the environment does.
    """
    import pybullet as pyb

    client = int(pyb.connect(pyb.DIRECT))
    pyb.setGravity(0.0, 0.0, -9.8, physicsClientId=client)
    pyb.setRealTimeSimulation(0, physicsClientId=client)
    pyb.setTimeStep(1.0 / 240.0, physicsClientId=client)
    try:
        yield client
    finally:
        pyb.disconnect(physicsClientId=client)


@pytest.fixture
def static_motion() -> StaticDeckMotion:
    """Return the zero-motion deck fixture: a pad on flat water, model scale."""
    return StaticDeckMotion()


@pytest.fixture
def landing_env(
    deck_pads: PadConfig,
    deck_scaling: ScalingConfig,
    env_landing_cfg: LandingConfig,
    env_success_cfg: SuccessConfig,
    env_obs_cfg: ObservationConfig,
    env_noise_cfg: NoiseConfig,
    env_reward_cfg: RewardConfig,
) -> Iterator[Callable[..., DeckLandingAviary]]:
    """Return a factory building a landing environment wired to the committed configs.

    Every environment the factory builds is closed when the test ends, so a failing
    assertion cannot leak a PyBullet client into the next test.

    Returns:
        ``(motion, *, pad="aft", episode_seed=BASE_SEED, **overrides) -> DeckLandingAviary``.
    """
    built: list[DeckLandingAviary] = []

    def build(
        motion: EpisodeMotionSource,
        *,
        pad: str = "aft",
        pad_offset_m: tuple[float, float, float] | None = None,
        episode_seed: int = BASE_SEED,
        cfg: LandingConfig | None = None,
        success: SuccessConfig | None = None,
        noise_cfg: NoiseConfig | None = None,
    ) -> DeckLandingAviary:
        if pad_offset_m is None:
            # A JONSWAP or sinusoid source carries a real lever arm; the static test double
            # is the ship's CG by construction, so its offset is exactly zero.
            # dmf's RealizationKey is a plain tuple
            # (ss, heading_deg, speed_kn, vessel, seed_ordinal).
            vessel = str(motion.key[3])
            pad_offset_m = (
                (0.0, 0.0, 0.0)
                if vessel == "static"
                else pad_offset_model_m(vessel, deck_pads, deck_scaling.froude_scale(), pad)
            )
        env = DeckLandingAviary(
            motion=motion,
            pad=pad,
            pad_radius_m=deck_pads.radius_model_m,
            pad_offset_m=pad_offset_m,
            cfg=cfg if cfg is not None else env_landing_cfg,
            success=success if success is not None else env_success_cfg,
            obs_cfg=env_obs_cfg,
            noise_cfg=noise_cfg if noise_cfg is not None else env_noise_cfg,
            reward_cfg=env_reward_cfg,
            episode_seed=episode_seed,
        )
        built.append(env)
        return env

    try:
        yield build
    finally:
        for env in built:
            env.close()
