"""MSS transfer-arm deck motion (protocol P7-D1 section 6): ``rld.deck.mss``.

What is pinned here, and against what:

* the 10 Hz samples equal dmf's exported CSV rows: our values rendered with the export's
  ``%.6g`` are the CSV's strings **exactly**, so ``|error| <= half a unit in the 6th
  significant digit``. In memory they equal ``dmf.mss.export.generate_realization`` bit
  for bit;
* rates and accelerations are analytic derivatives. A central finite difference with
  ``h = 1e-4`` s full scale is the test oracle, never the implementation;
* the time window equals the record's (120 s start, 600 s, 10 Hz full scale) and the
  JONSWAP source's model window;
* Froude scaling at ``lam = 1/25`` against the same source at ``lam = 1``, plus a
  hand-computed aft/CG pad case with the S175 lever arm (-70 m full = -2.8 m model);
* the heave sign convention (MSS z-down -> corpus up) on a one-component long wave the hull
  must contour;
* row-invariance (lazy chunking is exact), key isolation, the factory;
* the forecast feed and the fitted forecasters run on it, and the landing env resets on it.

Tests that need the MSS clone (``artifacts/mss/upstream``) or the exported records
(``artifacts/mss/records``) skip with the reason when they are absent (``make mss-export``
builds both). The Octave test also needs GNU Octave. Everything is simulation.
"""

import csv
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from dmf.config import load_sim
from dmf.data.splits import realization_key
from dmf.mss.convert import mss_motion_to_frame
from dmf.mss.export import MSSRealizationSpec, generate_realization
from dmf.mss.synth import WaveGrid, synthesize_mss_motion
from dmf.sim.generate import RealizationSpec

from rld.config import REPO_ROOT
from rld.deck.bridge import CLEAN_CHANNELS, JonswapDeckMotion
from rld.deck.config import MOTION_JONSWAP_CONFIG, load_motion
from rld.deck.mss import (
    MSS_CONFIG,
    MSS_RECORDS_DIR,
    MSS_UPSTREAM_DIR,
    MSS_UPSTREAM_SHA,
    MssDeckMotion,
    load_mss_config,
    load_mss_vessel_cached,
    mss_key,
    mss_motion,
    mss_record_path,
    mss_ss_label,
    parse_mss_ss_label,
    record_time_full_s,
    resolve_mat_path,
)
from rld.deck.scaling import FroudeScale

DMF_CONFIG = (
    REPO_ROOT / "third_party" / "deck-motion-forecast" / "configs" / "mss" / "s175_ss5.yaml"
)
LAM = 1.0 / 25.0
SIX = ("roll", "pitch", "heave", "roll_rate", "pitch_rate", "heave_rate")
GRID_KINDS = ("mss", "corpus")


def _all_specs() -> list[MSSRealizationSpec]:
    cfg = yaml.safe_load(MSS_CONFIG.read_text())
    return [
        MSSRealizationSpec(kind, float(h), float(u), int(s))
        for kind in GRID_KINDS
        for h in cfg["headings_deg"]
        for u in cfg["speeds_kn"]
        for s in cfg["seeds"]
    ]


def _clone_present() -> bool:
    return (MSS_UPSTREAM_DIR / "HYDRO" / "vessels_shipx" / "s175" / "s175.mat").is_file()


def _records_present() -> bool:
    return (MSS_RECORDS_DIR / "manifest.csv").is_file()


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    if not _clone_present():
        pytest.skip(f"MSS clone absent at {MSS_UPSTREAM_DIR} (run `make mss-export`)")
    return load_mss_config()


@pytest.fixture(scope="module")
def records(cfg: dict[str, Any]) -> Path:
    if not _records_present():
        pytest.skip(f"MSS records absent under {MSS_RECORDS_DIR} (run `make mss-export`)")
    return MSS_RECORDS_DIR


def _source(
    cfg: dict[str, Any],
    spec: MSSRealizationSpec,
    deck_pads: Any,
    lam: float = LAM,
    lookback_full_s: float = 20.0,
) -> MssDeckMotion:
    return MssDeckMotion(spec, cfg, FroudeScale(lam=lam), deck_pads, lookback_full_s)


# --- provenance ---------------------------------------------------------------------------


def test_config_differs_from_dmf_only_in_mat_path() -> None:
    ours = yaml.safe_load(MSS_CONFIG.read_text())
    theirs = yaml.safe_load(DMF_CONFIG.read_text())
    assert ours["vessel"]["mat_path"] == "artifacts/mss/upstream/HYDRO/vessels_shipx/s175/s175.mat"
    assert theirs["vessel"]["mat_path"] == "mss/upstream/HYDRO/vessels_shipx/s175/s175.mat"
    ours["vessel"]["mat_path"] = theirs["vessel"]["mat_path"]
    assert ours == theirs


def test_clone_is_pinned_and_outside_third_party(cfg: dict[str, Any]) -> None:
    head = subprocess.run(
        ["git", "-C", str(MSS_UPSTREAM_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == MSS_UPSTREAM_SHA
    mat = resolve_mat_path(cfg).resolve()
    assert mat.is_relative_to((REPO_ROOT / "artifacts").resolve())
    assert not mat.is_relative_to((REPO_ROOT / "third_party").resolve())


# --- the 10 Hz record ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("spec", _all_specs(), ids=lambda s: f"{s.grid_kind}-{s.stem}")
def test_ten_hz_samples_equal_exported_csv(
    cfg: dict[str, Any], records: Path, spec: MSSRealizationSpec, deck_pads: Any
) -> None:
    """Every CSV cell is our analytic value rendered as the export renders it (``%.6g``)."""
    with mss_record_path(spec, records).open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    assert header == ["t", *SIX]
    t = record_time_full_s(cfg)
    assert len(body) == t.size == 6000
    ch = _source(cfg, spec, deck_pads).channels(t)
    ours = {"t": t, **{c: getattr(ch, CLEAN_CHANNELS[c]) for c in SIX}}
    for j, name in enumerate(header):
        rendered = ["%.6g" % v for v in ours[name]]  # noqa: UP031 - the export's float_format
        column = [row[j] for row in body]
        bad = [i for i, (a, b) in enumerate(zip(rendered, column, strict=True)) if a != b]
        assert not bad, f"{spec.stem} {name}: {len(bad)} rows differ, first {bad[:3]}"


@pytest.mark.parametrize(
    "spec",
    [
        MSSRealizationSpec("mss", 180.0, 0.0, 0),
        MSSRealizationSpec("mss", 135.0, 12.0, 2),
        MSSRealizationSpec("corpus", 180.0, 6.0, 1),
        MSSRealizationSpec("corpus", 135.0, 6.0, 0),
    ],
    ids=lambda s: f"{s.grid_kind}-{s.stem}",
)
def test_channels_bit_equal_dmf_generate_realization(
    cfg: dict[str, Any], spec: MSSRealizationSpec, deck_pads: Any
) -> None:
    frame, _ = generate_realization(spec, load_mss_vessel_cached(resolve_mat_path(cfg)), cfg)
    t = record_time_full_s(cfg)
    assert np.array_equal(frame["t"].to_numpy(), t)
    ch = _source(cfg, spec, deck_pads).channels(t)
    for name in SIX:
        assert np.array_equal(getattr(ch, CLEAN_CHANNELS[name]), frame[name].to_numpy()), name


def test_single_sample_and_slices_are_bit_identical_to_the_full_grid(
    cfg: dict[str, Any], deck_pads: Any
) -> None:
    """Row-invariance: the feed's one-sample reads and lazy chunks reproduce the full grid."""
    source = _source(cfg, MSSRealizationSpec("mss", 135.0, 6.0, 1), deck_pads)
    t = record_time_full_s(cfg)[:1500]
    full = source.channels(t)
    fields = [*CLEAN_CHANNELS.values(), "roll_acc_dps2", "pitch_acc_dps2"]
    rng = np.random.default_rng(20261001)
    for i in rng.integers(0, t.size, 25):
        one = source.channels(t[i : i + 1])
        for f in fields:
            assert np.array_equal(getattr(one, f), getattr(full, f)[i : i + 1]), (i, f)
    t_model = np.arange(3001) / 240.0 + 40.0
    eager = source.deck_point(t_model, "aft").state
    for bounds in ((0, 1), (0, 240), (240, 480), (480, 960), (1920, 3001), (1000, 1001)):
        part = source.deck_point_rows(t_model, "aft", *bounds).state
        for f in eager.__dataclass_fields__:
            assert np.array_equal(getattr(part, f), getattr(eager, f)[slice(*bounds)]), (
                bounds,
                f,
            )


# --- analytic rates -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [MSSRealizationSpec("mss", 180.0, 12.0, 0), MSSRealizationSpec("corpus", 135.0, 6.0, 2)],
    ids=lambda s: f"{s.grid_kind}-{s.stem}",
)
def test_rates_are_the_analytic_derivative(
    cfg: dict[str, Any], spec: MSSRealizationSpec, deck_pads: Any
) -> None:
    """Rates and accelerations match a fine central difference (the oracle, not the code).

    ``h = 1e-4`` s full scale. Truncation ``~ h**2 w_e**2 / 6 < 1e-7`` relative at the
    highest encounter frequency (~6.4 rad/s); round-off ``~ eps / h ~ 1e-12``. Tolerance
    ``1e-6`` of each channel's peak.
    """
    source = _source(cfg, spec, deck_pads)
    h = 1e-4
    t = np.sort(np.random.default_rng(7).uniform(121.0, 718.0, 400))
    mid = source.channels(t)
    up, dn = source.channels(t + h), source.channels(t - h)
    pairs = (
        ("roll_deg", "roll_rate_dps"),
        ("pitch_deg", "pitch_rate_dps"),
        ("heave_m", "heave_rate_m_s"),
        ("roll_rate_dps", "roll_acc_dps2"),
        ("pitch_rate_dps", "pitch_acc_dps2"),
        ("heave_rate_m_s", "heave_acc_m_s2"),
    )
    for value, rate in pairs:
        fd = (getattr(up, value) - getattr(dn, value)) / (2.0 * h)
        analytic = getattr(mid, rate)
        # Head-seas roll is ~1e-6 deg/s^2 of RAO-table residue (dmf reports 0.0000 deg RMS);
        # the relative check below is still meaningful at that magnitude.
        peak = float(np.max(np.abs(analytic)))
        assert peak > 0.0
        err = float(np.max(np.abs(fd - analytic)))
        assert err <= 1e-6 * peak, (rate, err, peak)


# --- window -------------------------------------------------------------------------------


def test_window_matches_the_record(cfg: dict[str, Any], deck_pads: Any, sim_cfg: Any) -> None:
    rec = cfg["record"]
    assert (rec["t_start_s"], rec["duration_s"], rec["fs_hz"]) == (120.0, 600.0, 10.0)
    t = record_time_full_s(cfg)
    assert (t[0], t.size) == (120.0, 6000)
    assert t[-1] == pytest.approx(719.9, abs=1e-9)
    lookback = load_motion(MOTION_JONSWAP_CONFIG).forecast_lookback_full_s
    assert lookback == 20.0
    source = _source(cfg, MSSRealizationSpec("mss", 180.0, 6.0, 0), deck_pads, lookback_full_s=20.0)
    assert source.t_model_window_s == pytest.approx((4.0, 120.0), abs=1e-12)
    assert source.episode_start_window_s(12.5) == pytest.approx((4.0, 107.5), abs=1e-12)
    jonswap = JonswapDeckMotion(
        RealizationSpec(sea_state="SS5", heading_deg=180.0, speed_kn=6.0, vessel="s175", seed=0),
        sim_cfg,
        FroudeScale(LAM),
        deck_pads,
        20.0,
    )
    assert source.t_model_window_s == jonswap.t_model_window_s
    assert source.episode_start_window_s(12.5) == jonswap.episode_start_window_s(12.5)
    assert np.array_equal(
        source.full_time_s(np.array([0.0, 4.0, 120.0])),
        jonswap.full_time_s(np.array([0.0, 4.0, 120.0])),
    )
    assert source.full_time_s(np.array([0.0, 120.0])) == pytest.approx([120.0, 720.0])
    for bad in (119.9, 720.1):
        with pytest.raises(ValueError, match="leave the MSS record"):
            source.channels(np.array([bad]))
    with pytest.raises(ValueError, match="leave the MSS record"):
        source.deck_point(np.array([120.01]), "aft")
    with pytest.raises(ValueError, match="does not fit"):
        source.episode_start_window_s(116.5)


# --- Froude scaling and sign conventions ----------------------------------------------------


def test_froude_scaling_and_hand_computed_pads(cfg: dict[str, Any], deck_pads: Any) -> None:
    spec = MSSRealizationSpec("mss", 135.0, 12.0, 1)
    model = _source(cfg, spec, deck_pads, lam=LAM)
    full = _source(cfg, spec, deck_pads, lam=1.0)
    t_model = np.linspace(4.0, 119.0, 401)
    t_full_rel = t_model / np.sqrt(LAM)  # full-scale seconds since the record start
    assert np.allclose(model.full_time_s(t_model), full.full_time_s(t_full_rel), rtol=0, atol=1e-9)
    ch = full.channels(full.full_time_s(t_full_rel))
    sq = np.sqrt(LAM)
    for pad, x_full in (("aft", -70.0), ("cg", 0.0)):
        m = model.deck_point(t_model, pad).state
        f = full.deck_point(t_full_rel, pad).state
        assert np.allclose(m.position_m, LAM * f.position_m, rtol=1e-12, atol=1e-14)
        assert np.allclose(m.velocity_m_s, sq * f.velocity_m_s, rtol=1e-12, atol=1e-14)
        assert np.allclose(m.acceleration_m_s2, f.acceleration_m_s2, rtol=1e-12, atol=1e-14)
        assert np.allclose(m.euler_xyz_rad, f.euler_xyz_rad, rtol=1e-12, atol=1e-15)
        assert np.allclose(
            m.angular_velocity_rad_s, f.angular_velocity_rad_s / sq, rtol=1e-12, atol=1e-14
        )
        # Hand-computed centreline pad (protocol P1-D2): z = heave + x sin(pitch),
        # v_z = heave_rate + x cos(pitch) pitch_rate, at full scale, then scaled once.
        pitch = np.radians(ch.pitch_deg)
        z_full = ch.heave_m + x_full * np.sin(pitch)
        vz_full = ch.heave_rate_m_s + x_full * np.cos(pitch) * np.radians(ch.pitch_rate_dps)
        assert np.allclose(m.position_m[:, 2], LAM * z_full, rtol=1e-11, atol=1e-13)
        assert np.allclose(m.velocity_m_s[:, 2], sq * vz_full, rtol=1e-11, atol=1e-13)
        # Froude number of the pad's peak vertical speed is scale-invariant.
        fr_m = FroudeScale(LAM).froude_number(
            float(np.max(np.abs(m.velocity_m_s[:, 2]))), LAM * 175.0
        )
        fr_f = FroudeScale(1.0).froude_number(float(np.max(np.abs(f.velocity_m_s[:, 2]))), 175.0)
        assert fr_m == pytest.approx(fr_f, rel=1e-12)
    # The lever arm is the one the plate anchor removes.
    from rld.envs.platform import pad_offset_model_m

    assert pad_offset_model_m("s175", deck_pads, FroudeScale(LAM), "aft") == pytest.approx(
        (-2.8, 0.0, 0.0)
    )
    assert deck_pads.spec("aft").r_pad_full_m(model.length_full_m) == pytest.approx(
        (-70.0, 0.0, 0.0)
    )


def test_heave_sign_is_up_on_a_contoured_long_wave(cfg: dict[str, Any]) -> None:
    """Hand case: a 1 m, 0.25 rad/s head-sea wave at zero speed; the hull contours it.

    MSS heave is positive **down** (SNAME); the corpus convention is positive **up**. So
    converted heave must follow the wave elevation (positive up): correlation ~ +1 and
    amplitude ~ 1 m/m (MSS's long-wave limit, within 10 %). A missing flip gives -1.
    """
    vessel = load_mss_vessel_cached(resolve_mat_path(cfg))
    one = WaveGrid(
        w_rad_s=np.array([0.25]),
        amplitude_m=np.array([1.0]),
        phase_rad=np.array([0.0]),
        source="mss",
    )
    t = np.linspace(0.0, 4 * 2 * np.pi / 0.25, 2001)
    motion = synthesize_mss_motion(vessel, one, 180.0, 0.0, t)
    frame = mss_motion_to_frame(motion)
    heave = frame["heave"].to_numpy()
    assert np.corrcoef(heave, motion.elevation_m)[0, 1] > 0.99
    assert float(np.max(np.abs(heave))) == pytest.approx(1.0, rel=0.10)
    assert np.array_equal(heave, -motion.eta[2])
    assert np.array_equal(frame["pitch"].to_numpy(), np.rad2deg(motion.eta[4]))
    assert np.array_equal(frame["roll"].to_numpy(), np.rad2deg(motion.eta[3]))


# --- identity and factory -------------------------------------------------------------------


def test_key_cannot_collide_with_dmf_corpus_keys(cfg: dict[str, Any], sim_cfg: Any) -> None:
    keys = {mss_key(s.grid_kind, s.heading_deg, s.speed_kn, s.seed) for s in _all_specs()}
    assert len(keys) == 36
    corpus_like = {
        realization_key(ss.name, s.heading_deg, s.speed_kn, "s175", s.seed)
        for ss in load_sim(
            REPO_ROOT / "third_party" / "deck-motion-forecast" / "configs" / "sim" / "corpus.yaml"
        ).sea_states
        for s in _all_specs()
    }
    assert keys.isdisjoint(corpus_like)
    assert mss_key("mss", 180.0, 0.0, 0) == ("SS5/mss:mss", 180.0, 0.0, "s175", 0)
    assert mss_key("corpus", 180.0, 0.0, 0) == ("SS5/mss:corpus", 180.0, 0.0, "s175", 0)
    assert all("/" not in ss.name for ss in sim_cfg.sea_states)
    assert parse_mss_ss_label(mss_ss_label("corpus")) == ("SS5", "corpus")
    for bad in ("SS5", "SS5/mss:", "SS5/mss:jonswap", "/mss:mss"):
        with pytest.raises(ValueError):
            parse_mss_ss_label(bad)
    with pytest.raises(ValueError):
        mss_ss_label("jonswap")


def test_factory_builds_the_arm_and_rejects_off_arm_realizations(cfg: dict[str, Any]) -> None:
    from rld.eval.envs import load_eval_configs

    cfgs = load_eval_configs()
    source = mss_motion(cfgs, "corpus", 135.0, 6.0, 1)
    assert source.key == ("SS5/mss:corpus", 135.0, 6.0, "s175", 1)
    assert source.scale.lam == pytest.approx(LAM)
    assert source.lookback_full_s == 20.0
    assert source.t_model_window_s == pytest.approx((4.0, 120.0))
    for bad in (("mss", 90.0, 6.0, 0), ("mss", 180.0, 3.0, 0), ("mss", 180.0, 6.0, 3)):
        with pytest.raises(ValueError, match="not in the MSS arm"):
            mss_motion(cfgs, *bad)
    with pytest.raises(ValueError, match="unknown grid_kind"):
        mss_motion(cfgs, "jonswap", 180.0, 6.0, 0)


# --- consumers: forecast feed, forecasters, landing env --------------------------------------


@pytest.mark.slow
def test_feed_and_forecasters_run_on_the_mss_source(
    cfg: dict[str, Any], records: Path, deck_pads: Any
) -> None:
    from rld.deck.forecast import DEFAULT_MODEL_ROOT, OnlineForecaster, ShipMotionFeed

    spec = MSSRealizationSpec("mss", 135.0, 6.0, 0)
    source = _source(cfg, spec, deck_pads)
    scale = FroudeScale(LAM)
    feed = ShipMotionFeed.for_pad(source, "aft", deck_pads, scale)
    assert feed.r_pad_full_m == (-70.0, 0.0, 0.0)
    assert feed.pad_frac == pytest.approx(-0.4)
    t0 = 4.0  # the earliest legal start: the whole 20 s full-scale lookback is in the record
    feed.reset(t0)
    window = feed.window()
    assert window.shape == (200, 6) and np.all(np.isfinite(window))
    k = np.arange(feed.origin_index - 199, feed.origin_index + 1)
    assert k[0] == 1201 and k[-1] == 1400  # 120.1 ... 140.0 s full scale
    exact = np.concatenate([feed.query_channels(np.array([kk / 10.0])) for kk in k])
    assert np.array_equal(window, exact)
    # Against the CSV: the feed's grid k/10 and the export's 120 + j/10 may differ by an
    # ulp in t, so the bound is the %.6g rounding plus 1e-9 absolute.
    with mss_record_path(spec, records).open(newline="") as handle:
        body = np.array([[float(v) for v in row[1:]] for row in list(csv.reader(handle))[1:]])
    rows = body[k - 1200]
    assert np.all(np.abs(window - rows) <= 5e-6 * np.abs(rows) + 1e-9)
    with pytest.raises(ValueError, match="causality"):
        feed.query_channels(np.array([feed.clock_full_s + 0.1]))
    feed.advance_to(t0 + 1.0)
    assert feed.origin_index == 1450

    ran = 0
    for name in ("dlinear_ols", "residual_interval", "tcn", "tcn_quantile"):
        meta = DEFAULT_MODEL_ROOT / name / "meta.json"
        if not meta.is_file():
            continue
        forecast = OnlineForecaster(DEFAULT_MODEL_ROOT / name).forecast(feed)
        assert np.all(np.isfinite(forecast.pad_vz_m_s)), name
        assert np.all(np.isfinite(forecast.pad_z_m)), name
        ran += 1
    if ran == 0:
        pytest.skip(f"no fitted forecaster under {DEFAULT_MODEL_ROOT}")


@pytest.mark.slow
@pytest.mark.pybullet
def test_landing_env_resets_and_steps_on_the_mss_source(cfg: dict[str, Any]) -> None:
    from rld.envs.platform import LazyDeckTrajectory, build_trajectory
    from rld.eval.envs import load_eval_configs, make_env, pad_offset_for

    cfgs = load_eval_configs()
    source = mss_motion(cfgs, "mss", 180.0, 12.0, 2)
    env = make_env(cfgs, source, "s175", "aft", 11)
    try:
        _, info = env.reset(seed=11)
        assert info["realization_key"] == mss_key("mss", 180.0, 12.0, 2)
        lo, hi = source.episode_start_window_s(cfgs.landing.total_len_s)
        assert lo <= env.record.t0_model_s <= hi
        assert tuple(env.pad_offset_m) == pad_offset_for(cfgs, "s175", "aft")
        for _ in range(5):
            env.step(np.zeros((1, 3)))
    finally:
        env.close()
    # The env's lazy trajectory equals the eager reference on an episode grid.
    grid = np.arange(3001) / 240.0 + 10.0
    lazy = LazyDeckTrajectory(source, "aft", cfgs.landing.platform, grid, (-2.8, 0.0, 0.0)).full()
    eager = build_trajectory(source, "aft", cfgs.landing.platform, grid, (-2.8, 0.0, 0.0))
    assert np.array_equal(lazy.position_m, eager.position_m)
    assert np.array_equal(lazy.quaternion, eager.quaternion)


# --- Octave parity ----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.octave
def test_octave_parity_outside_third_party(cfg: dict[str, Any], tmp_path: Path) -> None:
    """The dmf m-files, staged verbatim beside a link to our clone, agree with our source."""
    if shutil.which("octave") is None:
        pytest.skip("GNU Octave not installed")
    from rld.deck.mss_octave import DMF_MSS_DIR, OCTAVE_FILES, run_parity

    (tmp_path / "upstream").symlink_to(MSS_UPSTREAM_DIR.resolve(), target_is_directory=True)
    table = run_parity(cfg, tmp_path, n_steps=40)
    for name in OCTAVE_FILES:
        assert (tmp_path / name).read_bytes() == (DMF_MSS_DIR / name).read_bytes()
    assert bool(table["passed"].all())
    patch = table[table["check"] == "patch_equivalence"]
    assert len(patch) == 1 and float(patch["rel_deviation"].iloc[0]) == 0.0
    ours = table[table["check"] == "rld_vs_octave"]
    assert len(ours) == 24
    assert float(ours["rel_deviation"].max()) < 1e-9
