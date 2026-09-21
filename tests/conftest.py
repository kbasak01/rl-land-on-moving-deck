"""Shared pytest fixtures.

Nearly empty in Phase 0: it carries the suite's base seed and the two paths every later
phase resolves against, so that no test hard-codes either. Phase 1 adds the deck-motion
fixtures, Phase 2 the PyBullet client fixture.

Every random draw in the test suite must come from a seeded generator derived from
:data:`BASE_SEED`. A test calling ``np.random.default_rng()`` with no seed is a
reproducibility hole and should be rejected in review.

Unit conventions follow the package convention in :mod:`rld`: seconds, metres, degrees at
boundaries, and model-scale versus full-scale named for every time and length.
"""

from pathlib import Path

import numpy as np
import pytest

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
