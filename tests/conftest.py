"""Shared pytest fixtures.

Carries the suite's base seed, the two paths every phase resolves against, and (from Phase 1)
the deck-motion configs. Phase 2 adds the PyBullet client fixture.

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

from collections.abc import Callable
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
