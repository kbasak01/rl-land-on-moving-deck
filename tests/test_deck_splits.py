"""Realization-level split integrity -- CLAUDE.md non-negotiable 2.

Every assertion here is about *which realizations* land where, never about windows: the split
unit is the realization key ``(vessel, ss, heading, speed, seed)`` and episodes are start
offsets inside one. The partition sizes are asserted as literals so that a change in dmf's
``TEST_SEED_FRAC`` or ``val_frac`` cannot slip through as "still disjoint, therefore fine".

Note the coverage counts: the first three regimes are confined to the frigate and cover 1920
realizations, dropping s175 entirely. Only ``unseen_vessel`` covers all 2304. A "covers the
whole grid" assertion would be wrong for three regimes out of four and is deliberately not
written.

Units: heading degrees, speed knots, seed a dimensionless ordinal. No scale applies.
"""

from pathlib import Path

import pytest
from dmf.config import SimConfig
from dmf.data.splits import (
    HELDOUT_HEADING_DEG,
    HELDOUT_SEA_STATE,
    HELDOUT_VESSEL,
    PRIMARY_VESSEL,
    REGIMES,
    Regime,
    assert_seed_disjoint,
)
from dmf.sim.encounter import is_encounter_monotonic, knots_to_m_s
from dmf.sim.generate import RealizationSpec, realization_grid
from numpy import linspace

from rld.deck.splits import (
    META_COLUMNS,
    PART_NAMES,
    build_all_splits,
    dev_pool,
    key_for_spec,
    part_specs,
    realization_meta,
    sorted_specs,
    spec_for_key,
)

#: Partition sizes of the committed grid, verified against ``build_split`` this phase.
#: ``covered`` is train + val + test, which is 1920 (frigate only) for every regime but
#: ``unseen_vessel``.
EXPECTED_SIZES: dict[Regime, tuple[int, int, int, int]] = {
    "id": (1296, 240, 384, 1920),
    "unseen_seastate": (1224, 216, 480, 1920),
    "unseen_heading": (1224, 216, 480, 1920),
    "unseen_vessel": (1632, 288, 384, 2304),
}


def test_meta_frame_is_the_grid_in_manifest_shape(sim_cfg: SimConfig) -> None:
    meta = realization_meta(sim_cfg)
    assert tuple(meta.columns) == META_COLUMNS
    assert len(meta) == 2304
    assert len(meta) == len(realization_grid(sim_cfg))
    # Per-vessel seed counts, not a single global count: s175 is the held-out hull.
    assert int((meta["vessel"] == PRIMARY_VESSEL).sum()) == 1920
    assert int((meta["vessel"] == HELDOUT_VESSEL).sum()) == 384
    assert sorted(meta["ss"].unique()) == ["SS3", "SS4", "SS5", "SS6"]
    assert sorted(meta["heading"].unique()) == [45.0, 90.0, 135.0, 180.0]
    assert sorted(meta["speed"].unique()) == [0.0, 6.0, 12.0]
    assert meta.duplicated().sum() == 0


def test_partition_sizes_are_the_committed_literals(sim_cfg: SimConfig) -> None:
    splits = build_all_splits(sim_cfg)
    assert tuple(splits) == REGIMES
    for regime, (n_train, n_val, n_test, n_covered) in EXPECTED_SIZES.items():
        split = splits[regime]
        sizes = (len(split.train_keys), len(split.val_keys), len(split.test_keys))
        assert sizes == (n_train, n_val, n_test), regime
        assert sum(sizes) == n_covered, regime


def test_every_regime_is_seed_disjoint(sim_cfg: SimConfig) -> None:
    for regime, split in build_all_splits(sim_cfg).items():
        assert_seed_disjoint(split)  # dmf's own three-property check
        assert split.regime == regime


def test_heldout_axes_appear_only_in_their_own_test_set(sim_cfg: SimConfig) -> None:
    splits = build_all_splits(sim_cfg)
    for regime, split in splits.items():
        for part in ("train", "val"):
            specs = part_specs(split, part)
            # The held-out hull is never trained or validated on, in any regime.
            assert not [s for s in specs if s.vessel == HELDOUT_VESSEL], (regime, part)
            if regime == "unseen_seastate":
                assert not [s for s in specs if s.sea_state == HELDOUT_SEA_STATE]
            if regime == "unseen_heading":
                assert not [s for s in specs if s.heading_deg == HELDOUT_HEADING_DEG]
    # ... and each held-out axis value appears *only* in its own regime's test set.
    vessel_test = {s.vessel for s in part_specs(splits["unseen_vessel"], "test")}
    assert vessel_test == {HELDOUT_VESSEL}
    ss_test = {s.sea_state for s in part_specs(splits["unseen_seastate"], "test")}
    assert ss_test == {HELDOUT_SEA_STATE}
    heading_test = {s.heading_deg for s in part_specs(splits["unseen_heading"], "test")}
    assert heading_test == {HELDOUT_HEADING_DEG}
    # The id regime holds out seed ordinals, not an axis value, so its test set spans the
    # whole frigate grid.
    id_test = part_specs(splits["id"], "test")
    assert {s.vessel for s in id_test} == {PRIMARY_VESSEL}
    assert {s.sea_state for s in id_test} == {"SS3", "SS4", "SS5", "SS6"}
    assert min(s.seed for s in id_test) == 32
    assert max(s.seed for s in id_test) == 39


def test_no_realization_is_shared_between_train_and_any_test_set(sim_cfg: SimConfig) -> None:
    splits = build_all_splits(sim_cfg)
    for regime, split in splits.items():
        assert not (split.train_keys | split.val_keys) & split.test_keys, regime


def test_both_encounter_non_monotonic_cells_survive_splitting(sim_cfg: SimConfig) -> None:
    """The 45 deg cells at 6 and 12 kn are kept, exactly as dmf keeps them.

    ``dmf.sim.encounter.is_encounter_monotonic`` reports the condition; it never filters. If
    a split silently dropped those cells, the awkward regime would vanish from the project
    without a decision being recorded.
    """
    band = linspace(sim_cfg.w_min_rad_s, sim_cfg.w_max_rad_s, 256)
    non_monotonic = [
        (heading, speed)
        for heading in sim_cfg.headings_deg
        for speed in sim_cfg.speeds_kn
        if not is_encounter_monotonic(band, knots_to_m_s(float(speed)), float(heading))
    ]
    assert sorted(non_monotonic) == [(45.0, 6.0), (45.0, 12.0)]
    for regime, split in build_all_splits(sim_cfg).items():
        everywhere = {
            (s.heading_deg, s.speed_kn) for part in PART_NAMES for s in part_specs(split, part)
        }
        for cell in non_monotonic:
            assert cell in everywhere, (regime, cell)


def test_key_and_spec_round_trip(sim_cfg: SimConfig) -> None:
    splits = build_all_splits(sim_cfg)
    for split in splits.values():
        for part in PART_NAMES:
            specs = part_specs(split, part)
            assert [spec_for_key(key_for_spec(s)) for s in specs] == specs
    # The keys a split carries are exactly the keys of the specs it yields.
    split = splits["unseen_vessel"]
    assert {key_for_spec(s) for s in part_specs(split, "test")} == set(split.test_keys)
    with pytest.raises(ValueError, match="unknown partition"):
        part_specs(split, "holdout")  # type: ignore[arg-type]


def test_sorted_specs_is_deterministic_and_grid_ordered(sim_cfg: SimConfig) -> None:
    split = build_all_splits(sim_cfg)["id"]
    specs = part_specs(split, "test")
    keys = [(s.vessel, s.sea_state, s.heading_deg, s.speed_kn, s.seed) for s in specs]
    assert keys == sorted(keys)
    # Set iteration order must not leak into the result.
    assert sorted_specs(frozenset(reversed(list(split.test_keys)))) == specs


def test_splitting_reads_no_corpus_file(
    sim_cfg: SimConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splits are a pure function of the grid: no Parquet, nothing under ``artifacts/``.

    Enforced by breaking the two ways a corpus could be reached -- pandas' Parquet reader and
    ``Path.open`` -- for the duration of the call.
    """
    import pandas as pd

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"split construction read a file: {args!r}")

    monkeypatch.setattr(pd, "read_parquet", explode)
    monkeypatch.setattr(Path, "open", explode)
    splits = build_all_splits(sim_cfg)
    assert len(splits) == len(REGIMES)


def test_val_fraction_is_honoured_and_validated(sim_cfg: SimConfig) -> None:
    split = build_all_splits(sim_cfg, val_frac=0.25)["id"]
    # Validation is carved by seed ordinal from the development pool, so its size is a
    # multiple of the number of cells (48 frigate cells).
    assert len(split.val_keys) % 48 == 0
    assert len(split.val_keys) > len(build_all_splits(sim_cfg, val_frac=0.15)["id"].val_keys)
    with pytest.raises(ValueError, match="val_frac"):
        build_all_splits(sim_cfg, val_frac=0.0)


def test_episodes_are_offsets_inside_a_realization_not_a_split_unit(
    sim_cfg: SimConfig,
) -> None:
    """A realization contributes to exactly one partition of a regime, whole.

    This is the structural reason a start-offset episode cannot leak: the split is a
    partition of realization keys, so every episode of a realization inherits that
    realization's partition.
    """
    split = build_all_splits(sim_cfg)["unseen_seastate"]
    membership: dict[RealizationSpec, set[str]] = {}
    for part in PART_NAMES:
        for spec in part_specs(split, part):
            membership.setdefault(spec, set()).add(part)
    assert all(len(parts) == 1 for parts in membership.values())
    assert len(membership) == 1920


# --- The intersection pool (protocol P3-D2) ---------------------------------------------------

#: Sizes of the regime-safe development pool for the committed grid: 3 sea states x 3
#: headings x 3 speeds x 27 seeds (train) and x 5 seeds (tune), frigate only.
EXPECTED_DEV_POOL_SIZES: tuple[int, int] = (729, 135)


def test_dev_pool_is_disjoint_from_every_regime_test_set(sim_cfg: SimConfig) -> None:
    """Neither pool shares a realization key with any regime's test partition."""
    train, tune = dev_pool(sim_cfg)
    train_keys = {key_for_spec(s) for s in train}
    tune_keys = {key_for_spec(s) for s in tune}
    for regime, split in build_all_splits(sim_cfg).items():
        assert not train_keys & split.test_keys, regime
        assert not tune_keys & split.test_keys, regime
        # Train is inside every regime's train partition; tune inside every train | val.
        assert train_keys <= split.train_keys, regime
        assert tune_keys <= split.train_keys | split.val_keys, regime
    assert not train_keys & tune_keys


def test_dev_pool_has_no_heldout_axis_value(sim_cfg: SimConfig) -> None:
    for pool in dev_pool(sim_cfg):
        assert {s.vessel for s in pool} == {PRIMARY_VESSEL}
        assert not [s for s in pool if s.vessel == HELDOUT_VESSEL]
        assert not [s for s in pool if s.sea_state == HELDOUT_SEA_STATE]
        assert not [s for s in pool if s.heading_deg == HELDOUT_HEADING_DEG]
        assert {s.sea_state for s in pool} == {"SS3", "SS4", "SS5"}
        assert {s.heading_deg for s in pool} == {45.0, 135.0, 180.0}
        assert {s.speed_kn for s in pool} == {0.0, 6.0, 12.0}


def test_dev_pool_seed_ranges_and_sizes_are_the_committed_literals(sim_cfg: SimConfig) -> None:
    train, tune = dev_pool(sim_cfg)
    assert {s.seed for s in train} == set(range(27))
    assert {s.seed for s in tune} == set(range(27, 32))
    assert (len(train), len(tune)) == EXPECTED_DEV_POOL_SIZES
    # No duplicates and size == 27 cells x n_seeds, so every cell appears with every seed:
    # a full factorial, no silently dropped cell.
    assert len(set(train)) == len(train)
    assert len(set(tune)) == len(tune)


def test_dev_pool_is_deterministic_and_grid_ordered(sim_cfg: SimConfig) -> None:
    train, tune = dev_pool(sim_cfg)
    for pool in (train, tune):
        keys = [(s.vessel, s.sea_state, s.heading_deg, s.speed_kn, s.seed) for s in pool]
        assert keys == sorted(keys)
        assert pool == sorted_specs(frozenset(reversed([key_for_spec(s) for s in pool])))
    assert dev_pool(sim_cfg) == (train, tune)
