"""Sinusoidal (Angelis-style) deck motion and its match to a JONSWAP realization.

Deliberately **not** marked ``slow``: the H4 motion-realism arm is this project's novelty
claim (plan D0.4), so its motion model must not hide behind the parity sweep in
``test_bridge.py``.

Everything is full scale unless a name says ``model``. Amplitudes are degrees (roll, pitch)
and metres (heave); periods and times are seconds; the reference cell is SS6, 180 deg, 12 kn,
whose peak encounter period is 9.401 s full = 1.880 s model at ``lam = 1/25``.
"""

import numpy as np
import pytest
from dmf.config import SimConfig
from dmf.sim.generate import RealizationSpec, realization_seed_sequence

from conftest import BASE_SEED
from rld.deck.bridge import DeckMotionSource, JonswapDeckMotion, load_vessel_cached
from rld.deck.config import (
    MOTION_SINUSOID_CONFIG,
    PadConfig,
    ScalingConfig,
    load_sinusoid,
)
from rld.deck.sinusoid import (
    SinusoidDeckMotion,
    matched_rms,
    matched_rms_from_channels,
    peak_encounter_period_s,
    sinusoid_params,
    sinusoid_seed_sequence,
)

#: The reference cell for every numeric expectation in this module.
REFERENCE_SPEC = RealizationSpec(
    seed=0, sea_state="SS6", heading_deg=180.0, speed_kn=12.0, vessel="frigate"
)

#: Its peak encounter period, seconds full scale, and the model-scale value at lam = 1/25.
REFERENCE_TE_FULL_S = 9.401238
REFERENCE_TE_MODEL_S = 1.880248


def _source(
    sim_cfg: SimConfig,
    deck_scaling: ScalingConfig,
    deck_pads: PadConfig,
    spec: RealizationSpec = REFERENCE_SPEC,
    episode_seed: int = 0,
    lookback_full_s: float = 20.0,
) -> SinusoidDeckMotion:
    """Build a matched sinusoid source for one realization, with the forecaster lookback."""
    scale = deck_scaling.froude_scale()
    params = sinusoid_params(spec, sim_cfg, scale, deck_pads, episode_seed=episode_seed)
    return SinusoidDeckMotion(
        params,
        sim_cfg,
        scale,
        deck_pads,
        length_full_m=load_vessel_cached(spec.vessel).length_m,
        lookback_full_s=lookback_full_s,
    )


def test_committed_config_names_only_implemented_rules(tmp_path: object) -> None:
    cfg = load_sinusoid(MOTION_SINUSOID_CONFIG)
    assert cfg.amplitude_rule == "sqrt2_times_matched_jonswap_rms"
    assert cfg.period_rule == "peak_encounter_period"
    assert cfg.phase_rule == "uniform_per_dof_from_episode_seed"
    # The sinusoid arm shares the JONSWAP arm's rates and forecaster lookback exactly.
    assert cfg.motion.physics_freq_hz == pytest.approx(240.0)
    assert cfg.motion.ctrl_freq_hz == pytest.approx(30.0)
    assert cfg.motion.forecast_lookback_full_s == pytest.approx(20.0)


def test_unimplemented_rule_is_rejected_not_ignored(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    text = MOTION_SINUSOID_CONFIG.read_text(encoding="utf-8").replace(
        "amplitude_rule: sqrt2_times_matched_jonswap_rms", "amplitude_rule: peak_to_peak"
    )
    bad = tmp_path / "motion_sinusoid.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="implements only"):
        load_sinusoid(bad)


def test_peak_encounter_period_is_the_reference_value() -> None:
    assert peak_encounter_period_s(12.4, 12.0, 180.0) == pytest.approx(
        REFERENCE_TE_FULL_S, abs=1e-5
    )
    assert peak_encounter_period_s(12.4, 12.0, 180.0) * 0.2 == pytest.approx(
        REFERENCE_TE_MODEL_S, abs=1e-5
    )
    # At rest there is no Doppler shift, so the encounter period is the peak period itself.
    assert peak_encounter_period_s(12.4, 0.0, 180.0) == pytest.approx(12.4)
    # Head seas shorten the encounter period; stern-quartering lengthens it.
    assert peak_encounter_period_s(9.7, 12.0, 180.0) < 9.7
    assert peak_encounter_period_s(9.7, 12.0, 45.0) > 9.7
    with pytest.raises(ValueError, match="tp_s must be positive"):
        peak_encounter_period_s(0.0, 12.0, 180.0)


def test_amplitudes_match_the_realization_rms_within_2_percent(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """Per DOF, the sinusoid's own RMS reproduces the matched realization's RMS.

    The rule is ``amplitude = sqrt(2) * RMS``, so this is a real check of the rule *and* of
    the record the RMS was taken over, not an identity: the sinusoid's RMS is measured from
    generated samples over the same 600 s window at the same 10 Hz full-scale rate.
    """
    scale = deck_scaling.froude_scale()
    target = matched_rms(REFERENCE_SPEC, sim_cfg, scale, deck_pads)
    source = _source(sim_cfg, deck_scaling, deck_pads)
    n = int(round(sim_cfg.duration_s * sim_cfg.fs_hz))
    t_full_s = sim_cfg.spinup_s + np.arange(n) / sim_cfg.fs_hz
    got = matched_rms_from_channels(source.channels(t_full_s))
    for value, want, name in zip(got, target, ("roll", "pitch", "heave"), strict=True):
        assert value == pytest.approx(want, rel=0.02), name
    # The rule itself, exactly: amplitude = sqrt(2) * RMS.
    assert source.params.heave_amp_m == pytest.approx(np.sqrt(2.0) * target[2], rel=1e-12)
    assert source.params.pitch_amp_deg == pytest.approx(np.sqrt(2.0) * target[1], rel=1e-12)
    assert source.params.roll_amp_deg == pytest.approx(np.sqrt(2.0) * target[0], rel=1e-12)


def test_dominant_period_is_within_2_percent_of_the_encounter_period(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """Measured from zero crossings of the generated pitch channel, not from the parameter."""
    source = _source(sim_cfg, deck_scaling, deck_pads)
    t_full_s = np.linspace(sim_cfg.spinup_s, sim_cfg.spinup_s + 300.0, 30_001)
    pitch = source.channels(t_full_s).pitch_deg
    sign_change = np.nonzero(np.diff(np.signbit(pitch)))[0]
    # Linear interpolation of each crossing time, then mean half-period.
    crossings = t_full_s[sign_change] - pitch[sign_change] * (
        t_full_s[sign_change + 1] - t_full_s[sign_change]
    ) / (pitch[sign_change + 1] - pitch[sign_change])
    measured_period_s = 2.0 * float(np.mean(np.diff(crossings)))
    assert measured_period_s == pytest.approx(REFERENCE_TE_FULL_S, rel=0.02)
    assert measured_period_s == pytest.approx(source.params.te_full_s, rel=1e-6)
    # Model scale: the period a PyBullet episode actually sees.
    assert deck_scaling.froude_scale().time(measured_period_s) == pytest.approx(
        REFERENCE_TE_MODEL_S, rel=0.02
    )


def test_phases_are_deterministic_in_the_episode_seed(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    scale = deck_scaling.froude_scale()
    rms = matched_rms(REFERENCE_SPEC, sim_cfg, scale, deck_pads)
    first = sinusoid_params(REFERENCE_SPEC, sim_cfg, scale, deck_pads, 7, rms_full=rms)
    again = sinusoid_params(REFERENCE_SPEC, sim_cfg, scale, deck_pads, 7, rms_full=rms)
    other_episode = sinusoid_params(REFERENCE_SPEC, sim_cfg, scale, deck_pads, 8, rms_full=rms)
    other_seed_spec = RealizationSpec(
        seed=1, sea_state="SS6", heading_deg=180.0, speed_kn=12.0, vessel="frigate"
    )
    other_realization = sinusoid_params(other_seed_spec, sim_cfg, scale, deck_pads, 7, rms_full=rms)
    assert first == again
    assert first.phase_rad != other_episode.phase_rad
    assert first.phase_rad != other_realization.phase_rad
    assert all(0.0 <= phase < 2.0 * np.pi for phase in first.phase_rad)
    # The phase stream has its own namespace, so it cannot alias dmf's wave draw.
    wave_entropy = realization_seed_sequence(REFERENCE_SPEC).entropy
    assert sinusoid_seed_sequence(REFERENCE_SPEC, 7).entropy != wave_entropy
    with pytest.raises(ValueError, match="episode_seed must be non-negative"):
        sinusoid_seed_sequence(REFERENCE_SPEC, -1)


def test_analytic_rates_and_accelerations_match_finite_differences(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """Rates are analytic derivatives of the sinusoid; FD lives only in this test."""
    source = _source(sim_cfg, deck_scaling, deck_pads)
    t_full_s = np.linspace(sim_cfg.spinup_s + 10.0, sim_cfg.spinup_s + 60.0, 501)
    step_s = 1e-3
    samples = [source.channels(t_full_s + k * step_s) for k in (-2, -1, 0, 1, 2)]

    def five_point(field: str) -> np.ndarray:
        values = [np.asarray(getattr(s, field)) for s in samples]
        return (-values[4] + 8.0 * values[3] - 8.0 * values[1] + values[0]) / (12.0 * step_s)

    for level, rate, acc in (
        ("roll_deg", "roll_rate_dps", "roll_acc_dps2"),
        ("pitch_deg", "pitch_rate_dps", "pitch_acc_dps2"),
        ("heave_m", "heave_rate_m_s", "heave_acc_m_s2"),
    ):
        np.testing.assert_allclose(
            getattr(samples[2], rate), five_point(level), rtol=1e-8, atol=1e-12
        )
        np.testing.assert_allclose(
            getattr(samples[2], acc), five_point(rate), rtol=1e-8, atol=1e-12
        )


def test_satisfies_the_deck_motion_source_protocol(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """Same interface as the JONSWAP source, **including** the full-scale lookback window.

    Without the lookback the Phase 4 forecast-conditioned arm could not run on the H4
    sinusoid leg, and H4 would compare two different observation spaces.
    """
    source: DeckMotionSource = _source(sim_cfg, deck_scaling, deck_pads)
    jonswap = JonswapDeckMotion(
        REFERENCE_SPEC, sim_cfg, deck_scaling.froude_scale(), deck_pads, lookback_full_s=20.0
    )
    assert source.key == jonswap.key
    assert source.t_model_window_s == jonswap.t_model_window_s == pytest.approx((4.0, 120.0))
    lo, _hi = source.t_model_window_s
    assert source.episode_start_window_s(12.0) == pytest.approx((4.0, 108.0))
    np.testing.assert_allclose(
        source.full_time_s(np.array([0.0, lo])), jonswap.full_time_s(np.array([0.0, lo]))
    )
    for bad in (sim_cfg.spinup_s - 1.0, sim_cfg.spinup_s + sim_cfg.duration_s + 1.0):
        with pytest.raises(ValueError, match="committed record"):
            source.channels(np.array([bad]))
    with pytest.raises(ValueError, match="does not fit"):
        source.episode_start_window_s(500.0)


def test_deck_point_uses_the_same_geometry_and_scaling_as_jonswap(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    source = _source(sim_cfg, deck_scaling, deck_pads)
    lam = deck_scaling.lam
    t_model_s = np.arange(10.0, 12.0, 1.0 / 240.0)
    channels = source.channels(source.full_time_s(t_model_s))
    cg = source.deck_point(t_model_s, "cg")
    aft = source.deck_point(t_model_s, "aft")
    np.testing.assert_allclose(cg.state.position_m[:, 2], channels.heave_m * lam, rtol=1e-14)
    assert np.all(cg.state.position_m[:, :2] == 0.0)
    np.testing.assert_allclose(
        aft.state.position_m[:, 0],
        -49.6 * lam * np.cos(np.radians(channels.pitch_deg)),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(aft.state.euler_xyz_rad, cg.state.euler_xyz_rad)
    assert float(np.std(aft.state.velocity_m_s[:, 2])) > float(np.std(cg.state.velocity_m_s[:, 2]))


def test_dofs_are_not_phase_locked_the_way_dmf_locks_them(
    sim_cfg: SimConfig, deck_scaling: ScalingConfig, deck_pads: PadConfig
) -> None:
    """The sinusoid's pitch/heave correlation is set by its drawn phases, not by a defect.

    dmf puts pitch in phase with heave where strip theory puts them in quadrature; at the
    reference cell the corpus record's ``corr(pitch, heave)`` is about +0.84. The sinusoid's
    correlation is exactly ``cos(phase_pitch - phase_heave)``, so the H4 arm is not carrying
    the same phase structure by construction. Recorded, not corrected (CLAUDE.md 7).
    """
    rng = np.random.default_rng(BASE_SEED + 21)
    scale = deck_scaling.froude_scale()
    t_full_s = sim_cfg.spinup_s + np.arange(6000) / sim_cfg.fs_hz
    jonswap_channels = JonswapDeckMotion(REFERENCE_SPEC, sim_cfg, scale, deck_pads).channels(
        t_full_s
    )
    defect_corr = float(np.corrcoef(jonswap_channels.pitch_deg, jonswap_channels.heave_m)[0, 1])
    assert defect_corr == pytest.approx(0.84, abs=0.02)
    seeds = [int(s) for s in rng.integers(0, 1000, size=6)]
    correlations = []
    for episode_seed in seeds:
        source = _source(sim_cfg, deck_scaling, deck_pads, episode_seed=episode_seed)
        # Exactly 60 encounter periods (564 s, inside the 600 s record), so the sample
        # correlation equals cos(dphi) to round-off instead of carrying an edge effect.
        span_s = 60.0 * source.params.te_full_s
        t_whole_periods_s = sim_cfg.spinup_s + np.linspace(0.0, span_s, 60_000, endpoint=False)
        channels = source.channels(t_whole_periods_s)
        measured = float(np.corrcoef(channels.pitch_deg, channels.heave_m)[0, 1])
        pitch_phase, heave_phase = source.params.phase_rad[1], source.params.phase_rad[2]
        assert measured == pytest.approx(float(np.cos(pitch_phase - heave_phase)), abs=1e-6)
        correlations.append(measured)
    assert min(abs(value) for value in correlations) < 0.9
