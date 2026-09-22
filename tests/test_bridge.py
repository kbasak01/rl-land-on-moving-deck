"""Parity of the deck-motion bridge against ``dmf.sim.generate.simulate_realization``.

This is the first test of Phase 1 and the one everything else is built behind. It asserts a
**structural** property, not a numerical-accuracy one: that :mod:`rld.deck.bridge`
reproduces dmf's seed path -- ``realization_seed_sequence(spec).spawn(2)``, first child into
``sample_components``, then ``synthesize_motion`` on the **absolute full-scale** time grid --
exactly. Measured margins are about three orders of magnitude inside the tolerance (heave
~6e-8 against ~1.8e-4), so a pass means "same random draws, same time origin", and a failure
means the seed path or the time origin is wrong, never that the tolerance is tight.

The tolerance ``1e-4 * max(1, max|x|)`` is set by the corpus's float32 cast
(``dmf/sim/generate.py``, ``CORPUS_DTYPES``), not by physics. The ``max(1, ...)`` floor is
load-bearing: without it the head-seas roll column, whose own float32 quantum is ~1e-6,
would be held to an absolute 1e-4 of a signal of amplitude ~1e-2.

Units: full-scale seconds for ``t``; degrees and degrees per second for roll/pitch channels;
metres, metres per second and metres per second squared for heave. Nothing in this module is
model-scale.
"""

from collections.abc import Callable

import numpy as np
import pytest
from dmf.config import SimConfig
from dmf.data.splits import realization_key
from dmf.sim.generate import (
    RealizationSpec,
    realization_seed_sequence,
    simulate_realization,
)
from dmf.sim.response import synthesize_motion
from dmf.sim.spectra import sample_components

from conftest import BASE_SEED
from rld.deck.bridge import (
    CLEAN_CHANNELS,
    FORECAST_CHANNELS,
    DeckMotionSource,
    JonswapDeckMotion,
    dof_phasors,
    harmonic_sum,
)
from rld.deck.config import MotionConfig, PadConfig, ScalingConfig

#: Number of grid cells drawn for the parity sweep, out of the corpus's 96.
N_PARITY_CELLS = 20

#: Relative tolerance per channel, set by the float32 storage cast of the corpus.
PARITY_RTOL = 1e-4


def _grid_cells(cfg: SimConfig) -> list[tuple[str, str, float, float]]:
    """Return every (vessel, sea state, heading_deg, speed_kn) cell of the corpus grid."""
    return [
        (vessel, sea.name, float(heading), float(speed))
        for vessel in cfg.vessels
        for sea in cfg.sea_states
        for heading in cfg.headings_deg
        for speed in cfg.speeds_kn
    ]


def _covers_every_axis(cells: list[tuple[str, str, float, float]], cfg: SimConfig) -> bool:
    """Return True if ``cells`` touches every vessel, sea state, heading and speed."""
    return (
        {c[0] for c in cells} == set(cfg.vessels)
        and {c[1] for c in cells} == {s.name for s in cfg.sea_states}
        and {c[2] for c in cells} == set(map(float, cfg.headings_deg))
        and {c[3] for c in cells} == set(map(float, cfg.speeds_kn))
    )


def parity_specs(cfg: SimConfig) -> list[RealizationSpec]:
    """Draw the parity sweep's realizations deterministically from ``BASE_SEED``.

    Twenty of the 96 grid cells are drawn without replacement, redrawing from the same
    generator until the sample covers every vessel, sea state, heading and speed (so the
    coverage assertion is a documented property of the draw, not a lucky one). The two
    encounter-non-monotonic cells -- 45 deg at 6 kn and 12 kn, where ``dw_e/dw`` changes
    sign inside the synthesis band -- are then added explicitly if absent.

    Args:
        cfg: dmf's corpus-generation config, which defines the grid.

    Returns:
        One :class:`dmf.sim.generate.RealizationSpec` per selected cell, seed ordinal drawn
        from the same generator.
    """
    rng = np.random.default_rng(BASE_SEED)
    cells = _grid_cells(cfg)
    for _ in range(100):
        idx = rng.choice(len(cells), size=N_PARITY_CELLS, replace=False)
        chosen = [cells[int(i)] for i in idx]
        if _covers_every_axis(chosen, cfg):
            break
    else:  # pragma: no cover - 100 failures is impossible for this grid
        raise AssertionError("could not draw a grid-covering parity sample")
    for extra in (("frigate", "SS5", 45.0, 6.0), ("frigate", "SS5", 45.0, 12.0)):
        if extra not in chosen:
            chosen.append(extra)
    return [
        RealizationSpec(
            seed=int(rng.integers(cfg.seeds_for(vessel))),
            sea_state=sea,
            heading_deg=heading,
            speed_kn=speed,
            vessel=vessel,
        )
        for vessel, sea, heading, speed in chosen
    ]


def test_parity_sample_covers_the_grid(sim_cfg: SimConfig) -> None:
    specs = parity_specs(sim_cfg)
    assert len(specs) >= N_PARITY_CELLS
    cells = [(s.vessel, s.sea_state, s.heading_deg, s.speed_kn) for s in specs]
    assert _covers_every_axis(cells, sim_cfg)
    assert ("frigate", "SS5", 45.0, 6.0) in cells
    assert ("frigate", "SS5", 45.0, 12.0) in cells
    assert all(0 <= s.seed < sim_cfg.seeds_for(s.vessel) for s in specs)


@pytest.mark.slow
def test_bridge_parity_with_simulate_realization(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """The bridge reproduces every clean corpus channel on the corpus's own 10 Hz grid."""
    worst: dict[str, float] = dict.fromkeys(CLEAN_CHANNELS, 0.0)
    for spec in parity_specs(sim_cfg):
        frame = simulate_realization(spec, sim_cfg)
        t_full_s = frame["t"].to_numpy()
        assert t_full_s[0] == pytest.approx(sim_cfg.spinup_s), "corpus time is absolute"
        source = JonswapDeckMotion(spec, sim_cfg, deck_scaling.froude_scale(), deck_pads)
        channels = source.channels(t_full_s)
        for column in CLEAN_CHANNELS:
            want = frame[column].to_numpy().astype(np.float64)
            got = np.asarray(getattr(channels, CLEAN_CHANNELS[column]), dtype=np.float64)
            err = float(np.max(np.abs(got - want)))
            tol = PARITY_RTOL * max(1.0, float(np.max(np.abs(want))))
            worst[column] = max(worst[column], err / tol)
            assert err <= tol, (
                f"{spec} column {column!r}: max_abs_err {err:.3e} exceeds tol {tol:.3e}"
            )
    # Margins are recorded as a fraction of tolerance; parity is structural (see docstring).
    assert max(worst.values()) < 0.1, f"parity margins unexpectedly tight: {worst}"


@pytest.mark.slow
def test_bridge_is_memoryless_in_time(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """Evaluating on a sparse grid equals evaluating on the dense grid and slicing.

    ``synthesize_motion`` is a sum of harmonics in ``t`` with no state, which is what
    licenses "no interpolation, no finite differences": the bridge may be asked for the
    model physics grid directly. Asserted bit-identical, not merely close.
    """
    spec = RealizationSpec(
        seed=3, sea_state="SS5", heading_deg=180.0, speed_kn=12.0, vessel="frigate"
    )
    source = JonswapDeckMotion(spec, sim_cfg, deck_scaling.froude_scale(), deck_pads)
    dense = np.arange(sim_cfg.spinup_s, sim_cfg.spinup_s + 60.0, 0.1)
    sparse = dense[::7]
    full = source.channels(dense)
    part = source.channels(sparse)
    for field in CLEAN_CHANNELS.values():
        assert np.array_equal(
            np.asarray(getattr(part, field)), np.asarray(getattr(full, field))[::7]
        ), field


@pytest.mark.slow
def test_seven_channel_reconstruction_matches_synthesize_motion(
    sim_cfg: SimConfig, deck_source: Callable[[RealizationSpec], JonswapDeckMotion]
) -> None:
    """Dmf's own seven channels, rebuilt from public phasors, agree to ``rtol = 1e-12``.

    This is the guard on the two angular-acceleration channels dmf does not store
    (``roll_acc``, ``pitch_acc``). They are built as ``-w_e**2 * z_dof`` from
    :func:`dmf.sim.response.dof_transfer` and
    :func:`dmf.sim.encounter.encounter_frequency` -- the same phasor form
    ``synthesize_motion`` uses -- so the honest test is that the *other* seven channels,
    rebuilt the same way, reproduce ``synthesize_motion`` to round-off. If they do, the two
    new channels are dmf's analytic derivatives too, and no finite difference is needed.
    Both encounter-non-monotonic 45 deg cells are included, because the signed-``w_e`` /
    ``abs(w_e)`` handling they exercise is inherited by this reconstruction.
    """
    specs = [
        RealizationSpec(
            seed=1, sea_state="SS6", heading_deg=180.0, speed_kn=12.0, vessel="frigate"
        ),
        RealizationSpec(seed=2, sea_state="SS5", heading_deg=45.0, speed_kn=6.0, vessel="frigate"),
        RealizationSpec(seed=0, sea_state="SS5", heading_deg=45.0, speed_kn=12.0, vessel="s175"),
        RealizationSpec(seed=3, sea_state="SS3", heading_deg=90.0, speed_kn=0.0, vessel="s175"),
    ]
    t_full_s = np.linspace(sim_cfg.spinup_s, sim_cfg.spinup_s + 120.0, 2401)
    to_deg = float(np.degrees(1.0))
    for spec in specs:
        source = deck_source(spec)
        phasors = dof_phasors(source.components, source.vessel, spec.heading_deg, source.speed_m_s)
        w_e = phasors.w_e_rad_s
        rebuilt = harmonic_sum(
            np.stack(
                [
                    phasors.roll,
                    phasors.pitch,
                    phasors.heave,
                    1j * w_e * phasors.roll,
                    1j * w_e * phasors.pitch,
                    1j * w_e * phasors.heave,
                    -(w_e**2) * phasors.heave,
                ]
            ),
            w_e,
            t_full_s,
        )
        reference = synthesize_motion(
            source.components, source.vessel, spec.heading_deg, source.speed_m_s, t_full_s
        )
        want = [
            reference.roll_deg,
            reference.pitch_deg,
            reference.heave_m,
            reference.roll_rate_dps,
            reference.pitch_rate_dps,
            reference.heave_rate_m_s,
            reference.heave_acc_m_s2,
        ]
        scales = [to_deg, to_deg, 1.0, to_deg, to_deg, 1.0, 1.0]
        for index, (column, factor) in enumerate(zip(want, scales, strict=True)):
            np.testing.assert_allclose(rebuilt[:, index] * factor, column, rtol=1e-12, atol=0.0)


@pytest.mark.slow
def test_spawn_of_the_unused_imu_stream_is_load_bearing(
    sim_cfg: SimConfig, deck_source: Callable[[RealizationSpec], JonswapDeckMotion]
) -> None:
    """Skipping dmf's second ``spawn`` child changes the wave draw, so it cannot be skipped."""
    spec = RealizationSpec(
        seed=5, sea_state="SS4", heading_deg=135.0, speed_kn=6.0, vessel="frigate"
    )
    source = deck_source(spec)
    unspawned = sample_components(
        hs_m=1.9,
        tp_s=8.8,
        gamma=3.3,
        n_components=sim_cfg.n_components,
        w_min_rad_s=sim_cfg.w_min_rad_s,
        w_max_rad_s=sim_cfg.w_max_rad_s,
        rng=np.random.default_rng(realization_seed_sequence(spec)),
        jitter=sim_cfg.jitter_frequencies,
    )
    assert not np.allclose(source.components.phase_rad, unspawned.phase_rad)


def test_key_matches_dmf_realization_key(
    deck_source: Callable[[RealizationSpec], JonswapDeckMotion],
) -> None:
    spec = RealizationSpec(
        seed=11, sea_state="SS5", heading_deg=90.0, speed_kn=12.0, vessel="frigate"
    )
    assert deck_source(spec).key == realization_key("SS5", 90.0, 12.0, "frigate", 11)


def test_time_window_and_bounds(
    sim_cfg: SimConfig,
    deck_motion_cfg: MotionConfig,
    deck_source: Callable[[RealizationSpec], JonswapDeckMotion],
) -> None:
    spec = RealizationSpec(
        seed=0, sea_state="SS5", heading_deg=180.0, speed_kn=12.0, vessel="frigate"
    )
    source = deck_source(spec)
    lo, hi = source.t_model_window_s
    # 20 s full-scale lookback and a 600 s full-scale record, at lambda = 1/25.
    assert lo == pytest.approx(deck_motion_cfg.forecast_lookback_full_s * 0.2)
    assert (lo, hi) == pytest.approx((4.0, 120.0))
    # Model time 0 is the first committed full-scale sample; the window's end is the last.
    assert float(source.full_time_s(np.array(0.0))) == pytest.approx(sim_cfg.spinup_s)
    assert float(source.full_time_s(np.array(hi))) == pytest.approx(
        sim_cfg.spinup_s + sim_cfg.duration_s
    )
    # A full lookback from the earliest legal episode start still lands inside the record.
    earliest = source.full_time_s(np.array(lo)) - deck_motion_cfg.forecast_lookback_full_s
    assert earliest == pytest.approx(sim_cfg.spinup_s)
    assert source.episode_start_window_s(12.0) == pytest.approx((4.0, 108.0))
    with pytest.raises(ValueError, match="does not fit"):
        source.episode_start_window_s(200.0)
    for bad in (sim_cfg.spinup_s - 1.0, sim_cfg.spinup_s + sim_cfg.duration_s + 1.0):
        with pytest.raises(ValueError, match="committed record"):
            source.channels(np.array([bad]))


def test_deck_point_is_model_scale_and_cg_is_pure_heave(
    deck_scaling: ScalingConfig,
    deck_source: Callable[[RealizationSpec], JonswapDeckMotion],
) -> None:
    """The CG pad is exactly Froude-scaled heave; the aft pad adds the lever-arm terms."""
    spec = RealizationSpec(
        seed=4, sea_state="SS5", heading_deg=180.0, speed_kn=12.0, vessel="frigate"
    )
    source = deck_source(spec)
    lam = deck_scaling.lam
    t_model_s = np.arange(10.0, 14.0, 1.0 / 240.0)
    channels = source.channels(source.full_time_s(t_model_s))
    cg = source.deck_point(t_model_s, "cg")
    aft = source.deck_point(t_model_s, "aft")
    assert cg.state.position_m.shape == (t_model_s.size, 3)
    np.testing.assert_allclose(cg.state.position_m[:, 2], channels.heave_m * lam, rtol=1e-14)
    np.testing.assert_allclose(
        cg.state.velocity_m_s[:, 2], channels.heave_rate_m_s * lam**0.5, rtol=1e-14
    )
    np.testing.assert_allclose(cg.state.acceleration_m_s2[:, 2], channels.heave_acc_m_s2)
    assert np.all(cg.state.position_m[:, :2] == 0.0)
    # The aft pad's lever arm is -0.4 L = -49.6 m full scale = -1.984 m model scale. Its x
    # projection is foreshortened by cos(pitch), so it sits just inside that arm.
    x_mean = float(aft.state.position_m[:, 0].mean())
    assert -49.6 * lam < x_mean < -49.6 * lam * np.cos(np.radians(5.0))
    np.testing.assert_allclose(
        aft.state.position_m[:, 0], -49.6 * lam * np.cos(np.radians(channels.pitch_deg)), rtol=1e-12
    )
    # Both pads share the deck's attitude exactly; only the pad offset differs.
    np.testing.assert_array_equal(aft.state.euler_xyz_rad, cg.state.euler_xyz_rad)
    assert float(np.std(aft.state.velocity_m_s[:, 2])) > float(np.std(cg.state.velocity_m_s[:, 2]))


def test_source_satisfies_the_protocol(
    deck_source: Callable[[RealizationSpec], JonswapDeckMotion],
) -> None:
    spec = RealizationSpec(
        seed=0, sea_state="SS3", heading_deg=180.0, speed_kn=0.0, vessel="frigate"
    )
    source: DeckMotionSource = deck_source(spec)  # static conformance
    assert set(FORECAST_CHANNELS) <= set(CLEAN_CHANNELS)
    for member in ("key", "t_model_window_s", "full_time_s", "channels", "deck_point"):
        assert hasattr(source, member)
