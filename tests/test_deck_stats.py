"""Deck-point statistics: determinism, schema, and the feasibility rule.

The committed 192-row table is what Gate 1 reads, so these tests are about the properties that
make it *citable*: it does not depend on the worker count, its percentiles are of the absolute
value (which is what a landing budget and the feasibility rule mean), its cell-level numbers
are pooled in sorted spec order, and the aft and CG rows of a cell come from bit-identical
motion.

The full grid is 2304 realizations and takes minutes, so the behavioural tests run on a
deliberately small grid built with ``dataclasses.replace`` -- two sea states, two headings, one
speed, two seeds, a 20 s full-scale record. The committed artifacts themselves are checked for
shape by a separate test that skips when they have not been generated yet.

Units: ``*_model_*`` metres / m per s / m per s^2 at model scale; angles degrees
(scale-invariant); ``pitch_rate_std_dps`` is the dmf channel at **full** scale.
"""

import csv
import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from dmf.config import SimConfig
from dmf.sim.generate import RealizationSpec

from conftest import REPO_ROOT
from rld.deck.bridge import JonswapDeckMotion
from rld.deck.config import MotionConfig, PadConfig, ScalingConfig
from rld.deck.stats import (
    TILT_CRITERION_DEG,
    CellPadStats,
    cell_keys,
    compute_cell,
    compute_grid,
    feasibility_rows,
    rows_as_dicts,
)

#: Expected shape of the committed artifacts over the full grid.
EXPECTED_CELL_ROWS = 192
EXPECTED_SEED_ROWS = 4608


@pytest.fixture
def small_cfg(sim_cfg: SimConfig) -> SimConfig:
    """Return a two-cell-per-heading grid on a 20 s full-scale record, for fast tests."""
    by_name = {sea.name: sea for sea in sim_cfg.sea_states}
    return replace(
        sim_cfg,
        vessels=("frigate",),
        sea_states=(by_name["SS3"], by_name["SS6"]),
        headings_deg=(180.0, 90.0),
        speeds_kn=(12.0,),
        seeds_per_cell=2,
        seeds_per_cell_by_vessel=(("frigate", 2),),
        duration_s=20.0,
    )


def _csv_text(rows: list[CellPadStats]) -> str:
    """Serialise rows exactly as ``scripts/deck_stats.py`` does, minus provenance."""
    payload = rows_as_dicts(rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(payload[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(payload)
    return buffer.getvalue()


def test_grid_has_96_cells(sim_cfg: SimConfig) -> None:
    cells = cell_keys(sim_cfg)
    assert len(cells) == 96
    assert len(cells) == len(set(cells))
    assert 2 * len(cells) == EXPECTED_CELL_ROWS


def test_workers_1_and_4_are_byte_identical(
    small_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """The committed CSV must not depend on the worker count (plan addendum C6)."""
    serial_cells, serial_seeds = compute_grid(
        small_cfg, deck_scaling, deck_pads, physics_freq_hz=240.0, workers=1
    )
    parallel_cells, parallel_seeds = compute_grid(
        small_cfg, deck_scaling, deck_pads, physics_freq_hz=240.0, workers=4
    )
    assert _csv_text(serial_cells) == _csv_text(parallel_cells)
    assert rows_as_dicts(serial_seeds) == rows_as_dicts(parallel_seeds)
    # ... and not merely equal within tolerance: the floats are identical.
    assert serial_cells == parallel_cells


def test_row_counts_and_pad_pairing(
    small_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    cells, seeds = compute_grid(
        small_cfg, deck_scaling, deck_pads, physics_freq_hz=240.0, workers=1
    )
    n_cells = len(cell_keys(small_cfg))
    assert n_cells == 4
    assert len(cells) == 2 * n_cells
    assert len(seeds) == 2 * n_cells * 2  # two seeds per cell, two pads
    # Every cell appears once per pad, and the two rows share every cell key.
    for index in range(0, len(cells), 2):
        aft, cg = cells[index], cells[index + 1]
        assert (aft.pad, cg.pad) == ("aft", "cg")
        assert (aft.vessel, aft.ss, aft.heading_deg, aft.speed_kn) == (
            cg.vessel,
            cg.ss,
            cg.heading_deg,
            cg.speed_kn,
        )
        # Attitude columns are pad-independent, so they must be exactly equal.
        assert aft.roll_std_deg == cg.roll_std_deg
        assert aft.pitch_std_deg == cg.pitch_std_deg
        assert aft.tilt_p99_deg == cg.tilt_p99_deg
        assert aft.corr_pitch_heave == cg.corr_pitch_heave
        assert aft.te_peak_full_s == cg.te_peak_full_s


def test_cg_pad_is_the_pure_heave_control(
    small_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    cells, _seeds = compute_grid(
        small_cfg, deck_scaling, deck_pads, physics_freq_hz=240.0, workers=1
    )
    for row in cells:
        if row.pad == "cg":
            assert row.x_pad_full_m == 0.0
            assert row.x_pad_model_m == 0.0
            assert row.vz_rms_ratio_pad_over_cg == pytest.approx(1.0, abs=1e-15)
        else:
            assert row.x_pad_full_m == pytest.approx(-49.6)
            assert row.x_pad_model_m == pytest.approx(-1.984)
        assert row.lam == pytest.approx(0.04)
        assert row.sample_rate_full_hz == pytest.approx(48.0)
        assert row.tilt_p99_over_15deg == (row.tilt_p99_deg > TILT_CRITERION_DEG)
        assert row.te_peak_model_s == pytest.approx(row.te_peak_full_s * 0.2)


def test_percentiles_are_of_the_absolute_value_and_pooled_in_sorted_order(
    small_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """Recompute one cell independently, pooling seeds in sorted order, and compare exactly."""
    cell = ("frigate", "SS6", 180.0, 12.0)
    cells, _seeds = compute_cell(cell, small_cfg, deck_scaling, deck_pads, physics_freq_hz=240.0)
    aft = next(row for row in cells if row.pad == "aft")
    scale = deck_scaling.froude_scale()
    n = int(round(small_cfg.duration_s * scale.sqrt_lam * 240.0))
    t_model_s = np.arange(n, dtype=np.float64) / 240.0
    chunks = []
    for seed in range(small_cfg.seeds_for("frigate")):  # sorted spec order
        spec = RealizationSpec(
            seed=seed, sea_state="SS6", heading_deg=180.0, speed_kn=12.0, vessel="frigate"
        )
        source = JonswapDeckMotion(spec, small_cfg, scale, deck_pads)
        chunks.append(source.deck_point(t_model_s, "aft").state.velocity_m_s[:, 2])
    pooled = np.concatenate(chunks)
    assert aft.vz_p99_model_m_s == float(np.percentile(np.abs(pooled), 99.0))
    assert aft.vz_p95_model_m_s == float(np.percentile(np.abs(pooled), 95.0))
    assert aft.vz_std_model_m_s == float(np.std(pooled))
    assert aft.vz_rms_model_m_s == float(np.sqrt(np.mean(np.square(pooled))))
    # Per-seed spread brackets the pooled percentile.
    per_seed = [float(np.percentile(np.abs(chunk), 99.0)) for chunk in chunks]
    assert aft.vz_p99_model_m_s_seed_min == pytest.approx(min(per_seed))
    assert aft.vz_p99_model_m_s_seed_max == pytest.approx(max(per_seed))
    assert min(per_seed) <= aft.vz_p99_model_m_s <= max(per_seed) * 1.5


def test_feasibility_takes_the_worst_head_seas_speed_and_both_references(
    small_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    cells, _seeds = compute_grid(
        small_cfg, deck_scaling, deck_pads, physics_freq_hz=240.0, workers=1
    )
    both = feasibility_rows(cells, deck_scaling, reference="all")
    assert len(both) == 2
    assert sum(row.is_gate for row in both) == 1
    gate = next(row for row in both if row.is_gate)
    assert gate.reference_name == "cf2x_urdf_max_speed_kmh"
    assert gate.threshold_m_s == pytest.approx(2.0833, abs=1e-4)
    assert (gate.vessel, gate.ss, gate.heading_deg, gate.pad) == (
        "frigate",
        "SS6",
        180.0,
        "aft",
    )
    # The measured statistic is the one in the table for that cell.
    matching = next(
        row
        for row in cells
        if (row.vessel, row.ss, row.heading_deg, row.speed_kn, row.pad)
        == (gate.vessel, gate.ss, gate.heading_deg, gate.speed_kn, gate.pad)
    )
    assert gate.vz_p99_model_m_s == matching.vz_p99_model_m_s
    assert gate.verdict == ("PASS" if gate.vz_p99_model_m_s <= gate.threshold_m_s else "FAIL")
    # The rejected reading is reported too, and is far stricter.
    other = next(row for row in both if not row.is_gate)
    assert other.threshold_m_s == pytest.approx(0.0625)
    assert len(feasibility_rows(cells, deck_scaling, reference="gate")) == 1
    with pytest.raises(ValueError, match="reference must be"):
        feasibility_rows(cells, deck_scaling, reference="urdf")
    with pytest.raises(ValueError, match="no s175"):
        feasibility_rows(cells, deck_scaling, vessel="s175")


def test_worker_count_is_validated(
    small_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        compute_grid(small_cfg, deck_scaling, deck_pads, physics_freq_hz=240.0, workers=0)


def test_sinusoid_parameters_are_committed_per_realization(
    small_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """Phase 6's ``ppo_sinusoid`` and Phase 7's H4 cross read these from the seeds CSV."""
    _cells, seeds = compute_grid(
        small_cfg, deck_scaling, deck_pads, physics_freq_hz=240.0, workers=1
    )
    for row in seeds:
        assert row.sin_te_full_s == pytest.approx(row.te_peak_full_s)
        assert row.sin_heave_amp_m > 0.0
        assert row.sin_pitch_amp_deg > 0.0
        assert row.sin_roll_amp_deg > 0.0
        assert row.sin_episode_seed == 0
    # The two pad rows of one realization carry the same sinusoid parameters.
    assert seeds[0].sin_heave_amp_m == seeds[1].sin_heave_amp_m


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a committed CSV into dict rows."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_committed_artifacts_have_the_expected_shape(
    deck_motion_cfg: MotionConfig, sim_cfg: SimConfig
) -> None:
    """Shape and self-consistency of the artifacts Gate 1 reads, if they exist yet."""
    cells_path = REPO_ROOT / "results" / "deck_stats.csv"
    seeds_path = REPO_ROOT / "results" / "deck_stats_seeds.csv"
    feas_path = REPO_ROOT / "results" / "deck_feasibility.csv"
    if not cells_path.exists():
        pytest.skip("results/deck_stats.csv not generated yet: run `make deck-stats`")
    cells = _read_csv(cells_path)
    seeds = _read_csv(seeds_path)
    feasibility = _read_csv(feas_path)
    assert len(cells) == EXPECTED_CELL_ROWS
    assert len(seeds) == EXPECTED_SEED_ROWS
    assert len(feasibility) == 2
    assert {row["pad"] for row in cells} == {"aft", "cg"}
    assert {row["vessel"] for row in cells} == {"frigate", "s175"}
    assert {row["ss"] for row in cells} == {"SS3", "SS4", "SS5", "SS6"}
    assert sum(int(row["n_seeds"]) for row in cells if row["pad"] == "aft") == 2304
    # The sample rate is the model physics rate expressed full scale, not the corpus rate.
    expected_rate = deck_motion_cfg.physics_freq_hz * 0.2
    assert {float(row["sample_rate_full_hz"]) for row in cells} == {expected_rate}
    assert expected_rate != sim_cfg.fs_hz
    # Provenance travels with every row.
    for row in (cells[0], seeds[0], feasibility[0]):
        assert row["dmf_sha"] and row["timestamp_utc"] and row["lam"]
    assert sum(row["is_gate"] == "True" for row in feasibility) == 1
