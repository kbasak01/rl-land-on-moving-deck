"""Realization-level splits, inherited from dmf rather than re-invented.

CLAUDE.md non-negotiable 2: train and eval never share a realization key
``(vessel, ss, heading, speed, seed)``. Episodes are start offsets *inside* a realization, so
the split unit is the realization and nothing here ever sees a window. The four regimes and
their held-out axes are dmf's, built by :func:`dmf.data.splits.build_split`:

======================  ==========================  ==============================
regime                  test set                     question
======================  ==========================  ==============================
``id``                  top 20 % of seed ordinals    in-distribution
``unseen_seastate``     SS6                          extrapolation to rougher seas
``unseen_heading``      90 deg (beam)                worst-case roll
``unseen_vessel``       s175                         transfer across hull dynamics
======================  ==========================  ==============================

**The corpus is never generated, read or enumerated on disk here.** The split is a pure
function of ``dmf.sim.generate.realization_grid(cfg)`` -- the same grid the corpus would be
generated from -- reshaped into the manifest's column names (``ss``, ``heading``, ``speed``,
``vessel``, ``seed``) that ``build_split`` expects. Nothing under ``artifacts/`` is touched.

Note that the first three regimes are confined to the primary hull, so they cover the 1920
frigate realizations and drop s175 entirely; only ``unseen_vessel`` covers all 2304. A
"covers the whole grid" assertion would be wrong for three regimes out of four.

**The intersection pool (protocol P3-D2).** dmf's four regimes are four *separate* splits,
and each regime's train set overlaps another regime's test set: ``id`` tests frigate seed
ordinals 32-39, which ``unseen_seastate``/``unseen_heading``/``unseen_vessel`` train on (their
train is 0-33); and ``id``-val SS6 and 90 deg realizations sit in the ``unseen_seastate`` and
``unseen_heading`` test sets. A single policy trained on one regime's train set and evaluated
on all four would therefore violate CLAUDE.md non-negotiable 2. :func:`dev_pool` returns the
development data every method in this project uses instead: TRAIN is the intersection of all
four regimes' train partitions (frigate, SS3-SS5, headings 180/135/45 deg, all speeds, seed
ordinals 0-26: 729 realizations) and TUNE is the same cells at seed ordinals 27-31
(135 realizations), for PID gain tuning, RL hyperparameters and curriculum validation.
Evaluation stays on each regime's dmf test partition, unchanged. Both pools are *derived*
from :func:`build_all_splits`, never from hard-coded seed numbers.

Units: ``heading`` degrees, ``speed`` knots, ``seed`` a dimensionless ordinal within its
cell. No time or length quantity appears in this module, so no scale applies.
"""

from functools import reduce
from typing import Literal

import pandas as pd
from dmf.config import SimConfig
from dmf.data.splits import REGIMES, RealizationKey, Regime, Split, build_split, realization_key
from dmf.sim.generate import RealizationSpec, realization_grid

__all__ = [
    "META_COLUMNS",
    "PART_NAMES",
    "Part",
    "build_all_splits",
    "dev_pool",
    "key_for_spec",
    "part_specs",
    "realization_meta",
    "sorted_specs",
    "spec_for_key",
]

#: Column names ``dmf.data.splits.build_split`` requires, in manifest order. Deliberately
#: ``heading``/``speed``, not ``heading_deg``/``speed_kn``: this frame stands in for
#: ``manifest.parquet``, whose schema is dmf's.
META_COLUMNS: tuple[str, ...] = ("ss", "heading", "speed", "vessel", "seed")

#: The three partitions of a split.
Part = Literal["train", "val", "test"]

#: Tuple form of :data:`Part`, for iteration.
PART_NAMES: tuple[Part, ...] = ("train", "val", "test")


def realization_meta(cfg: SimConfig) -> pd.DataFrame:
    """Build the manifest-shaped metadata frame from the corpus grid, without the corpus.

    Args:
        cfg: dmf's corpus config. Its vessels, sea states, headings, speeds and per-vessel
            seed counts define the grid.

    Returns:
        One row per realization with columns :data:`META_COLUMNS` -- ``heading`` in degrees,
        ``speed`` in knots, ``seed`` a dimensionless ordinal -- in
        :func:`dmf.sim.generate.realization_grid` order (vessel, sea state, heading, speed,
        seed). 2304 rows for the committed grid.
    """
    specs = realization_grid(cfg)
    return pd.DataFrame(
        {
            "ss": [spec.sea_state for spec in specs],
            "heading": [float(spec.heading_deg) for spec in specs],
            "speed": [float(spec.speed_kn) for spec in specs],
            "vessel": [spec.vessel for spec in specs],
            "seed": [int(spec.seed) for spec in specs],
        }
    )


def build_all_splits(cfg: SimConfig, val_frac: float = 0.15) -> dict[Regime, Split]:
    """Build all four dmf regimes from the grid metadata.

    Args:
        cfg: dmf's corpus config.
        val_frac: Fraction of *training* realizations held out for validation, in (0, 1).
            Held out at realization granularity, by seed ordinal, exactly as dmf does it --
            there is no RNG anywhere in split construction.

    Returns:
        One :class:`dmf.data.splits.Split` per regime, keyed by regime name, in
        :data:`dmf.data.splits.REGIMES` order.
    """
    meta = realization_meta(cfg)
    return {regime: build_split(meta, regime, val_frac) for regime in REGIMES}


def key_for_spec(spec: RealizationSpec) -> RealizationKey:
    """Return the canonical realization key of one spec.

    Args:
        spec: Grid coordinates and seed ordinal.

    Returns:
        ``(ss, heading_deg, speed_kn, vessel, seed_ordinal)``, built through
        :func:`dmf.data.splits.realization_key` so it compares equal to a key built from a
        manifest row.
    """
    return realization_key(spec.sea_state, spec.heading_deg, spec.speed_kn, spec.vessel, spec.seed)


def spec_for_key(key: RealizationKey) -> RealizationSpec:
    """Return the realization spec of one canonical key -- the inverse of :func:`key_for_spec`.

    Args:
        key: ``(ss, heading_deg, speed_kn, vessel, seed_ordinal)``.

    Returns:
        The :class:`dmf.sim.generate.RealizationSpec`, which is what the deck-motion bridge
        and dmf's own seed derivation take. Heading is degrees, speed knots.
    """
    sea_state, heading_deg, speed_kn, vessel, seed = key
    return RealizationSpec(
        seed=int(seed),
        sea_state=str(sea_state),
        heading_deg=float(heading_deg),
        speed_kn=float(speed_kn),
        vessel=str(vessel),
    )


def sorted_specs(keys: frozenset[RealizationKey]) -> list[RealizationSpec]:
    """Return the specs of ``keys`` in a deterministic, scheduler-independent order.

    The order is ``(vessel, sea state, heading, speed, seed)`` -- the grid's own nesting --
    so that anything pooled or reduced over a set of realizations is pooled in the same
    order on every machine and at every worker count (plan addendum C6).

    Args:
        keys: Realization keys, e.g. one partition of a :class:`dmf.data.splits.Split`.

    Returns:
        The corresponding specs, sorted.
    """
    specs = [spec_for_key(key) for key in keys]
    return sorted(specs, key=lambda s: (s.vessel, s.sea_state, s.heading_deg, s.speed_kn, s.seed))


def part_specs(split: Split, part: Part) -> list[RealizationSpec]:
    """Return one partition of a split as sorted realization specs.

    Args:
        split: The split.
        part: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        The partition's specs in :func:`sorted_specs` order.

    Raises:
        ValueError: If ``part`` is not one of :data:`PART_NAMES`.
    """
    if part == "train":
        return sorted_specs(split.train_keys)
    if part == "val":
        return sorted_specs(split.val_keys)
    if part == "test":
        return sorted_specs(split.test_keys)
    raise ValueError(f"unknown partition {part!r}, expected one of {list(PART_NAMES)}")


def dev_pool(cfg: SimConfig) -> tuple[list[RealizationSpec], list[RealizationSpec]]:
    """Return the regime-safe development pool: the (train, tune) realizations (P3-D2).

    Derived from :func:`build_all_splits` at its default ``val_frac``, never from literal
    seed numbers:

    * ``train`` = the intersection over all four regimes of ``train_keys``;
    * ``tune`` = the intersection over all four regimes of ``train_keys | val_keys``, minus
      the union of every regime's ``test_keys``, minus ``train``.

    Neither pool shares a realization key with any regime's test partition, so one policy
    (or one set of PID gains) developed on these pools can be evaluated on all four regimes
    without breaking CLAUDE.md non-negotiable 2. For the committed grid this is the frigate,
    SS3-SS5, headings 180/135/45 deg, speeds 0/6/12 kn, with seed ordinals 0-26 (train,
    729 realizations) and 27-31 (tune, 135 realizations).

    Args:
        cfg: dmf's corpus config. Its grid (vessels, sea states, headings in degrees, speeds
            in knots, per-vessel seed counts) defines the splits.

    Returns:
        ``(train, tune)``, each a list of :class:`dmf.sim.generate.RealizationSpec` in
        :func:`sorted_specs` order. Heading degrees, speed knots, seed a dimensionless
        ordinal; no time or length quantity, so no scale applies.
    """
    splits = list(build_all_splits(cfg).values())
    train = reduce(frozenset.intersection, (s.train_keys for s in splits))
    development = reduce(frozenset.intersection, (s.train_keys | s.val_keys for s in splits))
    tested = reduce(frozenset.union, (s.test_keys for s in splits))
    tune = development - tested - train
    return sorted_specs(train), sorted_specs(tune)
