"""Phase 4 forecaster adapter: parity, causality, pad kinematics, leakage, cost, coverage.

What each group asserts, and what it needs:

* **Causality** (bridge only, always runs). The feed never reads the source later than its
  clock: a spy records every query against the clock; a source that corrupts every future
  sample leaves windows and forecasts bit-identical; running the clock backwards, reading
  ahead of it, or reading before ``reset`` raises.
* **Pad kinematics** (bridge only). The adapter's closed-form pad ``z``/``v_z``, fed the
  *true* future channels, equals the bridge's ``deck_point`` for the aft pad and the CG pad
  to 1e-9; the interval-arithmetic band is the exact image of the channel box; and a
  hand-computed case pins the sign (bow-up lowers the aft pad).
* **Parity (a)** (corpus + models, slow). Identical corpus windows through dmf's offline path
  (``DeckMotionDataset`` -> ``model.forward`` -> ``invert_norm``) and through the adapter
  (torch and ORT) agree to ``1e-4 * max(1, |y|)``; the saved ``norm_stats.npz`` equals a
  fresh dmf fit on the manifest's train keys.
* **Parity (b)** (corpus + models, slow). The adapter fed by :class:`ShipMotionFeed` from the
  analytic bridge agrees with dmf offline on the corpus window at the same origin, and the
  feed's grid times are the corpus ``t`` column bit for bit.
* **Leak** (committed record). The fitted train and tune keys, read back through
  ``results/forecast/fit_keys.json`` and verified against the manifest hashes, equal
  ``dev_pool`` and are disjoint from every realization in ``results/episodes/*.parquet``.
* **CPU cost** (models, slow). ``advance_to`` then ``forecast`` at 30 Hz model for 2000 steps,
  ORT CPU, one thread. Writes ``results/forecast/cost.csv``; recorded, not gated.
* **Band coverage** (corpus + interval models, slow). Empirical coverage of the 90 % band on
  the tune pool, roll, pitch, pad ``v_z`` (aft and CG) at 1/2/3 s full-scale leads. Writes
  ``results/forecast/coverage.csv``; reported, not gated.

Artifacts are read from ``artifacts/dmf/`` or ``$RLD_DMF_ARTIFACTS``; tests needing an
absent artifact skip. Results files are written only from **non-smoke** models.

Units: model-scale seconds for episode time, full-scale seconds for the 10 Hz grid; degrees,
metres and metres per second, full scale for channels, model scale for pad quantities.
"""

import csv
import json
import os
import platform
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from dmf.config import ModelConfig, SimConfig, load_data
from dmf.data.dataset import DeckMotionDataset
from dmf.data.normalize import NormStats, build_norm_stats, invert_norm
from dmf.data.splits import RealizationKey, Split, realization_key
from dmf.data.windows import window_spec_from_config
from dmf.models.heads import PredictiveDistribution
from dmf.train.registry import build_model

from conftest import REPO_ROOT
from rld.deck.bridge import (
    FORECAST_CHANNELS,
    DeckPointTrajectory,
    JonswapDeckMotion,
    MotionChannels,
    load_vessel_cached,
)
from rld.deck.config import PadConfig, ScalingConfig
from rld.deck.forecast import (
    CONFORMAL_FILE,
    DEFAULT_MODEL_ROOT,
    HI,
    LO,
    POINT,
    DeckForecast,
    OnlineForecaster,
    ShipMotionFeed,
    calibrate_band,
    forecast_from_channels,
    leads_full_s,
    pad_vertical_band,
)
from rld.deck.forecast_fit import (
    DEFAULT_CORPUS_ROOT,
    DMF_DATA_CONFIG,
    FORECASTERS,
    canonical_key_list,
    corpus_frame,
    key_list_sha256,
    load_norm_scale,
    pool_keys,
    sha256_file,
)
from rld.deck.scaling import FroudeScale
from rld.deck.splits import spec_for_key

MODEL_ROOT = Path(os.environ.get("RLD_DMF_ARTIFACTS", str(DEFAULT_MODEL_ROOT)))
CORPUS_ROOT = Path(os.environ.get("RLD_DMF_CORPUS", str(DEFAULT_CORPUS_ROOT)))
RECORD_DIR = REPO_ROOT / "results" / "forecast"
EPISODES_DIR = REPO_ROOT / "results" / "episodes"

#: Parity tolerance, dmf's scale-relative criterion (P7-D3), elementwise.
PARITY_TOL = 1e-4

#: Absolute tolerance of the perfect-forecast kinematics check, metres / metres per second.
KINEMATICS_TOL = 1e-9

#: dmf's 10 Hz full-scale grid offset: the corpus starts at spinup_s = 120 s = sample 1200.
SPINUP_SAMPLES = 1200

#: Cost-test geometry: 2000 control steps at 30 Hz model.
COST_STEPS = 2000
CTRL_HZ = 30.0

#: Coverage: window origins every 20 samples (2 s full); leads 1/2/3 s full (the observation
#: block) and 7/8/9.3 s full (1.4/1.6/1.86 s model, where the gated controllers' quiescence
#: window sits).
COVERAGE_ORIGIN_STRIDE = 20
COVERAGE_LEADS = (10, 20, 30, 70, 80, 93)
FEASIBILITY_LEADS = (70, 80, 93)

#: Permissive quiescence limits, model scale (P3-D3): degrees, degrees, m/s model.
LIM_ROLL_DEG, LIM_PITCH_DEG, LIM_VZ_M_S = 3.0, 2.0, 0.16

#: Pooled calibrated pad-v_z coverage must land here on the tune pool (the amendment's test).
CALIBRATED_COVERAGE = (0.89, 0.91)

INTERVAL_MODELS = ("residual_interval", "tcn_quantile")


# ---------------------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------------------


def _available(name: str) -> bool:
    """True when a model is fitted -- and, for an interval model, also pad-v_z calibrated.

    Between the TCN fit and ``--calibrate-only`` an interval model exists without its
    calibration; its tests skip (named in ``-rs``) rather than fail on a known gap.
    """
    meta_path = MODEL_ROOT / name / "meta.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta["head"] != "quantile" or CONFORMAL_FILE in meta["files"]


def _corpus_present() -> bool:
    return (CORPUS_ROOT / "manifest.parquet").is_file()


def _needs_model(name: str) -> None:
    if not _available(name):
        pytest.skip(f"no fitted (and, if interval, calibrated) {name} under {MODEL_ROOT}")


def _needs_corpus() -> None:
    if not _corpus_present():
        pytest.skip(f"no dmf corpus under {CORPUS_ROOT}")


@pytest.fixture(scope="module")
def scale(deck_scaling: ScalingConfig) -> FroudeScale:
    return deck_scaling.froude_scale()


@pytest.fixture(scope="module")
def dmf_train_stats(sim_cfg: SimConfig) -> NormStats:
    """Return dmf's own normalisation fit on the fitted train keys (loads 729 once)."""
    _needs_corpus()
    train_keys, tune = pool_keys(sim_cfg)
    data_cfg = load_data(DMF_DATA_CONFIG)
    split = Split(
        regime="id",
        train_keys=frozenset(train_keys),
        val_keys=frozenset(tune),
        test_keys=frozenset(),
    )
    ds = DeckMotionDataset(CORPUS_ROOT, split, "train", data_cfg, window_spec_from_config(data_cfg))
    return ds.norm_stats


def _source(
    key: RealizationKey, sim_cfg: SimConfig, scale: FroudeScale, pads: PadConfig
) -> JonswapDeckMotion:
    return JonswapDeckMotion(spec_for_key(key), sim_cfg, scale, pads, lookback_full_s=20.0)


def _t_model_for_origin(origin_index: int, offset_full_s: float = 0.05) -> float:
    """Model time whose latest grid sample is corpus row ``origin_index`` (+ spin-up)."""
    t_full = (SPINUP_SAMPLES + origin_index) / 10.0 + offset_full_s
    return (t_full - 120.0) * 0.2


class SpySource:
    """Wraps a source; records ``(max queried t_full, feed clock)`` for every read."""

    def __init__(self, inner: JonswapDeckMotion) -> None:
        """Wrap ``inner``; ``feed`` is attached after the feed is built."""
        self.inner = inner
        self.scale = inner.scale
        self.key = inner.key
        self.t_model_window_s = inner.t_model_window_s
        self.feed: ShipMotionFeed | None = None
        self.log: list[tuple[float, float]] = []

    def full_time_s(self, t_model_s: np.ndarray) -> np.ndarray:
        """Delegate."""
        return self.inner.full_time_s(t_model_s)

    def deck_point(self, t_model_s: np.ndarray, pad: str) -> DeckPointTrajectory:
        """Refuse: the feed reads channels only."""
        del t_model_s, pad
        raise AssertionError("the feed must not read deck_point")

    def channels(self, t_full_s: np.ndarray) -> MotionChannels:
        """Record the query against the clock, then delegate."""
        assert self.feed is not None
        self.log.append((float(np.max(t_full_s)), self.feed.clock_full_s))
        return self.inner.channels(t_full_s)


class FutureCorruptingSource(SpySource):
    """Returns wildly wrong channels for any time later than the feed's clock."""

    def channels(self, t_full_s: np.ndarray) -> MotionChannels:
        """Corrupt every sample later than the clock by +1000 in every channel."""
        assert self.feed is not None
        ch = self.inner.channels(t_full_s)
        future = np.asarray(t_full_s) > self.feed.clock_full_s
        bump = np.where(future, 1000.0, 0.0)
        return MotionChannels(
            t_full_s=ch.t_full_s,
            roll_deg=ch.roll_deg + bump,
            pitch_deg=ch.pitch_deg + bump,
            heave_m=ch.heave_m + bump,
            roll_rate_dps=ch.roll_rate_dps + bump,
            pitch_rate_dps=ch.pitch_rate_dps + bump,
            heave_rate_m_s=ch.heave_rate_m_s + bump,
            heave_acc_m_s2=ch.heave_acc_m_s2,
            roll_acc_dps2=ch.roll_acc_dps2,
            pitch_acc_dps2=ch.pitch_acc_dps2,
        )


def _run_feed(feed: ShipMotionFeed, t0: float, n_steps: int) -> Iterator[np.ndarray]:
    feed.reset(t0)
    yield feed.window()
    for n in range(1, n_steps + 1):
        feed.advance_to(t0 + n / CTRL_HZ)
        yield feed.window()


CAUSALITY_KEY: RealizationKey = realization_key("SS5", 180.0, 12.0, "frigate", 30)


# ---------------------------------------------------------------------------------------
# causality -- bridge only
# ---------------------------------------------------------------------------------------


def test_feed_never_reads_past_its_clock(
    sim_cfg: SimConfig, scale: FroudeScale, deck_pads: PadConfig
) -> None:
    spy = SpySource(_source(CAUSALITY_KEY, sim_cfg, scale, deck_pads))
    feed = ShipMotionFeed.for_pad(spy, "aft", deck_pads, scale)  # type: ignore[arg-type]
    spy.feed = feed
    for _ in _run_feed(feed, 17.123, 400):
        pass
    assert len(spy.log) > 50
    worst = max(t - clock for t, clock in spy.log)
    assert worst <= 0.0, f"a query ran {worst} s (full) ahead of the clock"


def test_future_perturbation_leaves_windows_and_forecasts_bit_identical(
    sim_cfg: SimConfig, scale: FroudeScale, deck_pads: PadConfig
) -> None:
    clean = SpySource(_source(CAUSALITY_KEY, sim_cfg, scale, deck_pads))
    dirty = FutureCorruptingSource(_source(CAUSALITY_KEY, sim_cfg, scale, deck_pads))
    feeds = []
    for src in (clean, dirty):
        feed = ShipMotionFeed.for_pad(src, "aft", deck_pads, scale)  # type: ignore[arg-type]
        src.feed = feed
        feeds.append(feed)
    # The corrupting source does corrupt -- a direct future read shows it (the test has teeth).
    feeds[1].reset(20.0)
    future = np.array([feeds[1].clock_full_s + 0.5])
    assert dirty.channels(future).roll_deg[0] - clean.inner.channels(future).roll_deg[0] == 1000.0

    forecaster = OnlineForecaster(MODEL_ROOT / "dlinear_ols") if _available("dlinear_ols") else None
    n_compared = 0
    for w_clean, w_dirty, step in zip(
        _run_feed(feeds[0], 20.0, 120), _run_feed(feeds[1], 20.0, 120), range(121), strict=True
    ):
        assert np.array_equal(w_clean, w_dirty)
        if forecaster is not None and step % 10 == 0:
            a, b = forecaster.forecast(feeds[0]), forecaster.forecast(feeds[1])
            assert np.array_equal(a.channels_full, b.channels_full)
            assert np.array_equal(a.pad_vz_m_s, b.pad_vz_m_s)
            n_compared += 1
    assert forecaster is None or n_compared == 13


def test_feed_raise_cases(sim_cfg: SimConfig, scale: FroudeScale, deck_pads: PadConfig) -> None:
    feed = ShipMotionFeed.for_pad(
        _source(CAUSALITY_KEY, sim_cfg, scale, deck_pads), "aft", deck_pads, scale
    )
    with pytest.raises(RuntimeError):
        feed.advance_to(10.0)
    with pytest.raises(RuntimeError):
        feed.window()
    feed.reset(10.0)
    feed.advance_to(10.5)
    with pytest.raises(ValueError, match="backwards"):
        feed.advance_to(10.49)
    with pytest.raises(ValueError, match="causality"):
        feed.query_channels(np.array([feed.clock_full_s + 1e-9]))
    # A read exactly at the clock is legal.
    assert feed.query_channels(np.array([feed.clock_full_s])).shape == (1, 6)
    # The lookback may not leave the committed record (t0 < 4.0 s model).
    with pytest.raises(ValueError):
        feed.reset(3.9)


def test_feed_grid_origin_and_incremental_updates(
    sim_cfg: SimConfig, scale: FroudeScale, deck_pads: PadConfig
) -> None:
    src = _source(CAUSALITY_KEY, sim_cfg, scale, deck_pads)
    feed = ShipMotionFeed.for_pad(src, "aft", deck_pads, scale)
    assert feed.r_pad_full_m == pytest.approx((-49.6, 0.0, 0.0))
    assert feed.full_s_per_model_s == pytest.approx(5.0)
    t0 = _t_model_for_origin(1000, offset_full_s=0.0)
    feed.reset(t0)
    # On-grid clock: the origin is the sample at (or, after rounding, just before) the clock.
    assert feed.origin_index in (SPINUP_SAMPLES + 999, SPINUP_SAMPLES + 1000)
    assert feed.t_origin_model_s <= feed.clock_model_s
    for n in range(1, 200):
        feed.advance_to(t0 + n / CTRL_HZ)
        assert feed.clock_full_s - feed.t_origin_full_s < 0.1
        assert feed.t_origin_full_s <= feed.clock_full_s
    # The incrementally built ring equals a fresh fill at the same clock.
    fresh = ShipMotionFeed.for_pad(src, "aft", deck_pads, scale)
    fresh.reset(feed.clock_model_s)
    assert np.array_equal(feed.window(), fresh.window())
    assert np.array_equal(feed.window_times_full_s, fresh.window_times_full_s)
    # A jump longer than the lookback refills the whole ring.
    feed.advance_to(feed.clock_model_s + 10.0)
    fresh.reset(feed.clock_model_s)
    assert np.array_equal(feed.window(), fresh.window())


# ---------------------------------------------------------------------------------------
# pad kinematics -- bridge only
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        realization_key("SS5", 180.0, 12.0, "frigate", 28),
        realization_key("SS5", 45.0, 6.0, "frigate", 3),  # encounter-non-monotonic cell
        realization_key("SS6", 90.0, 0.0, "frigate", 35),
        realization_key("SS4", 135.0, 12.0, "s175", 2),
    ],
)
@pytest.mark.parametrize("pad", ["aft", "cg"])
def test_perfect_forecast_pad_kinematics_match_the_bridge(
    key: RealizationKey, pad: str, sim_cfg: SimConfig, scale: FroudeScale, deck_pads: PadConfig
) -> None:
    src = _source(key, sim_cfg, scale, deck_pads)
    feed = ShipMotionFeed.for_pad(src, pad, deck_pads, scale)
    feed.reset(41.7)
    lead_full = np.arange(1, 151) / 10.0
    t_full = feed.t_origin_full_s + lead_full
    truth = feed.source.channels(t_full)  # the true future, read outside the feed on purpose
    six = np.stack(
        [
            truth.roll_deg,
            truth.pitch_deg,
            truth.heave_m,
            truth.roll_rate_dps,
            truth.pitch_rate_dps,
            truth.heave_rate_m_s,
        ],
        axis=-1,
    )
    band = np.repeat(six[:, :, None], 3, axis=-1)
    fc = forecast_from_channels(
        band,
        origin_channels_full=feed.window()[-1],
        t_origin_full_s=feed.t_origin_full_s,
        t_origin_model_s=feed.t_origin_model_s,
        full_s_per_model_s=feed.full_s_per_model_s,
        fs_full_hz=10.0,
        r_pad_full_m=feed.r_pad_full_m,
        scale=scale,
        model="truth",
        interval_levels=None,
    )
    bridge = src.deck_point(fc.t_model_s, pad).state
    np.testing.assert_allclose(
        fc.pad_z_m[:, POINT], bridge.position_m[:, 2], rtol=0, atol=KINEMATICS_TOL
    )
    np.testing.assert_allclose(
        fc.pad_vz_m_s[:, POINT], bridge.velocity_m_s[:, 2], rtol=0, atol=KINEMATICS_TOL
    )
    assert np.array_equal(fc.pad_z_m[:, LO], fc.pad_z_m[:, POINT])
    assert np.array_equal(fc.pad_vz_m_s[:, HI], fc.pad_vz_m_s[:, POINT])
    np.testing.assert_allclose(fc.roll_deg[:, POINT], truth.roll_deg, rtol=0, atol=0)
    np.testing.assert_allclose(fc.lead_model_s, lead_full * 0.2, rtol=1e-12)
    # band_at on the grid returns the grid values; the origin is the observed sample.
    got = fc.band_at("pad_vz_m_s", fc.t_model_s[[0, 9, 149]])
    np.testing.assert_allclose(got, fc.pad_vz_m_s[[0, 9, 149]], rtol=0, atol=1e-15)
    at_origin = src.deck_point(np.array([fc.t_origin_model_s]), pad).state
    np.testing.assert_allclose(
        fc.band_at("pad_z_m", np.array([fc.t_origin_model_s]))[0],
        at_origin.position_m[0, 2],
        rtol=0,
        atol=KINEMATICS_TOL,
    )


def test_pad_band_is_the_exact_image_of_the_channel_box(rng: np.random.Generator) -> None:
    x = -49.6
    for straddle in (False, True):
        band = np.zeros((1, 6, 3))
        centre = rng.normal(size=6) * np.array([2.0, 1.5, 0.8, 1.0, 1.0, 0.5])
        if straddle:
            centre[1] = 0.1
        half = np.abs(rng.normal(size=6)) * 0.4 + 0.2
        band[0, :, LO], band[0, :, POINT], band[0, :, HI] = centre - half, centre, centre + half
        z, vz = pad_vertical_band(band, x)
        # Dense sampling of the box (corners, edges and zero pitch included).
        heave = np.linspace(band[0, 2, LO], band[0, 2, HI], 5)
        pitch = np.unique(np.r_[np.linspace(band[0, 1, LO], band[0, 1, HI], 41), 0.0])
        pitch = pitch[(pitch >= band[0, 1, LO]) & (pitch <= band[0, 1, HI])]
        rate = np.linspace(band[0, 4, LO], band[0, 4, HI], 5)
        hrate = np.linspace(band[0, 5, LO], band[0, 5, HI], 5)
        zs = heave[:, None] + x * np.sin(np.radians(pitch))[None, :]
        vzs = (
            hrate[:, None, None]
            + (x * np.radians(rate))[None, :, None] * np.cos(np.radians(pitch))[None, None, :]
        )
        assert zs.min() == pytest.approx(z[0, LO], abs=1e-12)
        assert zs.max() == pytest.approx(z[0, HI], abs=1e-12)
        assert vzs.min() == pytest.approx(vz[0, LO], abs=1e-12)
        assert vzs.max() == pytest.approx(vz[0, HI], abs=1e-12)
        assert z[0, LO] <= z[0, POINT] <= z[0, HI]
        assert vz[0, LO] <= vz[0, POINT] <= vz[0, HI]


def test_pad_band_sign_convention_hand_case(scale: FroudeScale) -> None:
    """Bow-up 1 deg, no heave: the aft pad (x = -49.6 m) drops 49.6 sin(1 deg) = 0.8656394 m."""
    band = np.zeros((1, 6, 3))
    band[0, 1, :] = 1.0  # pitch, deg, bow-up
    band[0, 4, :] = 2.0  # pitch rate, deg/s, bow rising
    z, vz = pad_vertical_band(band, -49.6)
    assert z[0, POINT] == pytest.approx(-0.8656394, abs=1e-7)
    assert vz[0, POINT] == pytest.approx(-49.6 * np.cos(np.radians(1.0)) * np.radians(2.0))
    assert vz[0, POINT] < 0.0  # bow rising -> aft pad falling
    z_cg, vz_cg = pad_vertical_band(band, 0.0)
    assert z_cg[0, POINT] == 0.0 and vz_cg[0, POINT] == 0.0
    # Model units: z x lam = 0.04, v_z x sqrt(lam) = 0.2.
    fc = forecast_from_channels(
        np.repeat(band, 150, axis=0),
        origin_channels_full=band[0, :, POINT],
        t_origin_full_s=200.0,
        t_origin_model_s=16.0,
        full_s_per_model_s=5.0,
        fs_full_hz=10.0,
        r_pad_full_m=(-49.6, 0.0, 0.0),
        scale=scale,
        model="hand",
        interval_levels=None,
    )
    assert fc.pad_z_m[0, POINT] == pytest.approx(-0.8656394 * 0.04, abs=1e-8)
    assert fc.pad_vz_m_s[0, POINT] == pytest.approx(vz[0, POINT] * 0.2)
    block = leads_full_s(fc)
    assert block.shape == (3, 2)
    np.testing.assert_allclose(fc.lead_model_s[[9, 19, 29]], [0.2, 0.4, 0.6], rtol=1e-12)
    with pytest.raises(ValueError):
        leads_full_s(fc, (1.05,))
    with pytest.raises(ValueError):
        fc.band_at("pad_z_m", np.array([fc.t_end_model_s + 1e-6]))
    with pytest.raises(ValueError, match="centreline"):
        forecast_from_channels(
            band,
            origin_channels_full=band[0, :, POINT],
            t_origin_full_s=200.0,
            t_origin_model_s=16.0,
            full_s_per_model_s=5.0,
            fs_full_hz=10.0,
            r_pad_full_m=(-49.6, 1.0, 0.0),
            scale=scale,
            model="hand",
            interval_levels=None,
        )


# ---------------------------------------------------------------------------------------
# parity -- corpus + fitted models
# ---------------------------------------------------------------------------------------


def _offline_reference(
    name: str, keys: list[RealizationKey], stats: NormStats, n_origins: int
) -> tuple[np.ndarray, list[tuple[RealizationKey, int]], np.ndarray]:
    """Run dmf's offline path on corpus windows: dataset, ``forward``, ``invert_norm``.

    Returns:
        ``(bands (B, H, 6, 3), [(key, start_index)], raw corpus windows (B, L, 6))``, full
        scale, ``[lo, point, hi]`` = the 0.05/0.5/0.95 quantiles (or the point, thrice).
    """
    meta = json.loads((MODEL_ROOT / name / "meta.json").read_text(encoding="utf-8"))
    channels, saved_scale = load_norm_scale(MODEL_ROOT / name / "norm_stats.npz")
    assert channels == stats.channels
    if meta["smoke"]:  # a smoke fit's scale comes from its own handful of keys
        stats = build_norm_stats(saved_scale, channels, "dev_pool/train", 1)
    else:
        np.testing.assert_array_equal(saved_scale, stats.scale)  # the saved stats are dmf's
    data_cfg = load_data(DMF_DATA_CONFIG)
    spec = window_spec_from_config(data_cfg)
    split = Split(
        regime="id", train_keys=frozenset(), val_keys=frozenset(keys), test_keys=frozenset()
    )
    starts = np.arange(0, 6000 - spec.total_length + 1, spec.stride)
    origins = spec.lookback - 1 + starts[np.linspace(0, starts.size - 1, n_origins).astype(int)]
    val_ds = DeckMotionDataset(
        CORPUS_ROOT, split, "val", data_cfg, spec, stats=stats, origins=origins
    )
    cfg = ModelConfig(
        name=meta["dmf_name"],
        head=meta["head"],
        quantiles=tuple(meta["quantiles"]),
        params=meta["params"],
        label=meta["label"],
    )
    model = build_model(cfg, spec, 6, 6)
    model.load_state_dict(
        torch.load(MODEL_ROOT / name / "state_dict.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    triples = [val_ds[i] for i in range(len(val_ds))]
    where = [val_ds.describe_window(i) for i in range(len(val_ds))]
    x = torch.stack([t[0] for t in triples])
    mean = torch.stack([t[2] for t in triples])
    target_stats = stats.subset(val_ds.target_columns)
    with torch.no_grad():
        raw = model.forward(x)
    if meta["head"] == "quantile":
        dist = PredictiveDistribution(raw, "quantile", tuple(meta["quantiles"]))
        bands = invert_norm(dist.quantiles_at((0.05, 0.5, 0.95)), target_stats, mean).numpy()
    else:
        point = invert_norm(raw, target_stats, mean).numpy()
        bands = np.repeat(point[..., None], 3, axis=-1)
    frames = {key: corpus_frame(CORPUS_ROOT, key, FORECAST_CHANNELS) for key in keys}
    windows = np.stack(
        [
            frames[key][list(FORECAST_CHANNELS)].to_numpy(np.float64)[s : s + spec.lookback]
            for key, s in where
        ]
    )
    return bands.astype(np.float64), where, windows


#: Parity measurements collected across the parametrized tests; written at module teardown.
_PARITY_ROWS: list[dict[str, object]] = []


@pytest.fixture(scope="module", autouse=True)
def _parity_record() -> Iterator[None]:
    """Merge this run's non-smoke parity rows into ``results/forecast/parity.csv``."""
    yield
    real = [r for r in _PARITY_ROWS if not r["smoke_fit"]]
    if not real:
        return
    path = RECORD_DIR / "parity.csv"
    merged: dict[tuple[str, str], dict[str, object]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                merged[(row["model"], row["check"])] = dict(row)
    for row in real:
        merged[(str(row["model"]), str(row["check"]))] = row
    order = list(FORECASTERS)
    rows = sorted(merged.values(), key=lambda r: (order.index(str(r["model"])), str(r["check"])))
    _write_rows(path, rows)


def _parity(
    got: np.ndarray, ref: np.ndarray, *, model: str, check: str, smoke: bool, gated: bool = True
) -> float:
    """Compare at ``1e-4 * max(1, |ref|)`` elementwise, record, and assert when gated."""
    err = np.abs(got - ref)
    limit = PARITY_TOL * np.maximum(1.0, np.abs(ref))
    worst = float((err / limit).max())
    _PARITY_ROWS.append(
        {
            "model": model,
            "check": check,
            "n_windows": int(got.shape[0]) if got.ndim == 4 else 1,
            "max_abs_err": f"{float(err.max()):.3e}",
            "worst_err_over_tol": round(worst, 5),
            "tolerance": "1e-4*max(1,|y|) elementwise, full-scale corpus units",
            "gated": gated,
            "passed": worst < 1.0,
            "smoke_fit": smoke,
        }
    )
    if gated:
        assert worst < 1.0, f"{model}/{check}: worst error is {worst:.3f} x the tolerance"
    return worst


@pytest.mark.slow
@pytest.mark.parametrize("name", list(FORECASTERS))
def test_parity_a_adapter_matches_dmf_offline_on_corpus_windows(
    name: str, sim_cfg: SimConfig, dmf_train_stats: NormStats
) -> None:
    _needs_model(name)
    _needs_corpus()
    tune = pool_keys(sim_cfg)[1]
    keys = [tune[i] for i in (0, 44, 89, 134)]
    ref, _, windows = _offline_reference(name, keys, dmf_train_stats, n_origins=12)
    for backend in ("torch", "ort"):
        forecaster = OnlineForecaster(MODEL_ROOT / name, backend)  # type: ignore[arg-type]
        got = forecaster.predict_windows(windows)
        assert got.shape == ref.shape == (48, 150, 6, 3)
        _parity(got, ref, model=name, check=f"a_corpus_{backend}", smoke=forecaster.is_smoke)
        assert np.all(got[..., LO] <= got[..., POINT]) and np.all(got[..., POINT] <= got[..., HI])


@pytest.mark.slow
@pytest.mark.parametrize("name", list(FORECASTERS))
def test_parity_b_feed_from_the_bridge_matches_dmf_offline(
    name: str,
    sim_cfg: SimConfig,
    scale: FroudeScale,
    deck_pads: PadConfig,
    dmf_train_stats: NormStats,
) -> None:
    _needs_model(name)
    _needs_corpus()
    tune = pool_keys(sim_cfg)[1]
    keys = [tune[i] for i in (7, 100)]
    ref, where, _ = _offline_reference(name, keys, dmf_train_stats, n_origins=6)
    forecaster = OnlineForecaster(MODEL_ROOT / name, "ort")
    got, unquantised = [], []
    for key, start in where:
        origin = start + 199
        feed = ShipMotionFeed.for_pad(
            _source(key, sim_cfg, scale, deck_pads), "aft", deck_pads, scale
        )
        feed.reset(_t_model_for_origin(origin))
        assert feed.origin_index == SPINUP_SAMPLES + origin
        frame = corpus_frame(CORPUS_ROOT, key, FORECAST_CHANNELS)
        # The grid is the corpus grid, bit for bit ...
        assert np.array_equal(feed.window_times_full_s, frame["t"].to_numpy()[start : origin + 1])
        # ... and the analytic window is the stored window to within one float32 ulp.
        stored = frame[list(FORECAST_CHANNELS)].to_numpy()[start : origin + 1]
        assert stored.dtype == np.float32
        live = feed.window().astype(np.float32)
        assert np.all(np.abs(live - stored) <= np.spacing(np.abs(stored)))
        got.append(forecaster.forecast(feed).channels_full)
        # Measured, not gated: the same window without dmf's float32 storage cast.
        x = ((feed.window() - feed.window().mean(axis=0)) / forecaster.scale_full)[None]
        raw = forecaster.run_normalised(x.astype(np.float32)).astype(np.float64)
        if forecaster.head == "quantile":
            raw = raw[..., [forecaster.quantiles.index(q) for q in (0.05, 0.5, 0.95)]]
        else:
            raw = np.repeat(raw[..., None], 3, axis=-1)
        mean = feed.window().mean(axis=0)
        unquantised.append(raw[0] * forecaster.scale_full[None, :, None] + mean[None, :, None])
    smoke = forecaster.is_smoke
    _parity(np.stack(got), ref, model=name, check="b_bridge_feed_ort", smoke=smoke)
    ratio = _parity(
        np.stack(unquantised),
        ref,
        model=name,
        check="b_bridge_feed_ort_without_float32_cast (measured, not gated)",
        smoke=smoke,
        gated=False,
    )
    print(f"{name}: without dmf's float32 storage cast the worst error is {ratio:.3f} x tol")


# ---------------------------------------------------------------------------------------
# leakage -- committed record
# ---------------------------------------------------------------------------------------


def test_fitted_keys_are_dev_pool_and_disjoint_from_every_frozen_episode(
    sim_cfg: SimConfig,
) -> None:
    manifest_path = RECORD_DIR / "fit_manifest.json"
    keys_path = RECORD_DIR / "fit_keys.json"
    if not manifest_path.is_file():
        pytest.skip("no committed fit record yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["smoke"] is False
    assert sha256_file(keys_path) == manifest["keys_file_sha256"]
    doc = json.loads(keys_path.read_text(encoding="utf-8"))
    pools = {part: [realization_key(*row) for row in doc[part]] for part in ("train", "tune")}
    for part, keys in pools.items():
        assert len(keys) == manifest["pools"][part]["n"]
        assert key_list_sha256(keys) == manifest["pools"][part]["sha256"]
        assert canonical_key_list(keys) == json.dumps(doc[part], separators=(",", ":"))
    train, tune = pool_keys(sim_cfg)
    assert pools["train"] == train and len(train) == 729
    assert pools["tune"] == tune and len(tune) == 135
    assert not set(train) & set(tune)
    for model in manifest["models"].values():
        assert model["fitted_on"] == "dev_pool/train"
    lists = sorted(EPISODES_DIR.glob("*.parquet"))
    assert len(lists) == 5
    frozen: set[RealizationKey] = set()
    for path in lists:
        df = pd.read_parquet(path)
        frozen |= {
            realization_key(r.ss, r.heading_deg, r.speed_kn, r.vessel, r.realization_seed)
            for r in df.itertuples(index=False)
        }
    assert len(frozen) > 500
    assert not set(train) & frozen, "a fitted training realization is a frozen episode"
    assert not set(tune) & frozen, "a tune realization is a frozen episode"


# ---------------------------------------------------------------------------------------
# CPU cost -- recorded, not gated
# ---------------------------------------------------------------------------------------


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.slow
def test_cpu_cost_per_control_step(
    sim_cfg: SimConfig, scale: FroudeScale, deck_pads: PadConfig
) -> None:
    import onnxruntime as ort

    names = [n for n in FORECASTERS if _available(n)]
    if not names:
        pytest.skip(f"no fitted forecaster under {MODEL_ROOT}")
    rows: list[dict[str, object]] = []
    key = realization_key("SS5", 180.0, 12.0, "frigate", 29)
    for name in names:
        forecaster = OnlineForecaster(MODEL_ROOT / name, "ort", intra_op_threads=1)
        feed = ShipMotionFeed.for_pad(
            _source(key, sim_cfg, scale, deck_pads), "aft", deck_pads, scale
        )
        t0 = 10.0
        feed.reset(t0)
        for _ in range(20):  # warm-up, not timed
            forecaster.forecast(feed)
        adv, fct = np.empty(COST_STEPS), np.empty(COST_STEPS)
        for n in range(COST_STEPS):
            a = time.perf_counter()
            feed.advance_to(t0 + (n + 1) / CTRL_HZ)
            b = time.perf_counter()
            fc = forecaster.forecast(feed)
            c = time.perf_counter()
            adv[n], fct[n] = b - a, c - b
        assert isinstance(fc, DeckForecast)
        total = adv + fct
        for part, values in (("advance_to", adv), ("forecast", fct), ("total", total)):
            ms = values * 1e3
            assert np.all(np.isfinite(ms))
            rows.append(
                {
                    "model": name,
                    "smoke_fit": forecaster.is_smoke,
                    "part": part,
                    "backend": "onnxruntime-cpu",
                    "intra_op_threads": 1,
                    "steps": COST_STEPS,
                    "ctrl_hz_model": CTRL_HZ,
                    "control_period_ms": round(1e3 / CTRL_HZ, 3),
                    "p50_ms": round(float(np.percentile(ms, 50)), 4),
                    "p99_ms": round(float(np.percentile(ms, 99)), 4),
                    "mean_ms": round(float(ms.mean()), 4),
                    "max_ms": round(float(ms.max()), 4),
                    "onnxruntime": ort.__version__,
                    "cpu": platform.processor() or platform.machine(),
                    "host": platform.node(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
                }
            )
    real = [r for r in rows if not r["smoke_fit"]]
    for r in rows:
        print(f"{r['model']:>18} {r['part']:>10}: p50 {r['p50_ms']} ms, p99 {r['p99_ms']} ms")
    if real:
        _write_rows(RECORD_DIR / "cost.csv", real)


# ---------------------------------------------------------------------------------------
# pad-v_z conformal calibration -- adapter behaviour
# ---------------------------------------------------------------------------------------


def test_calibrate_band_with_unit_gamma_is_the_raw_box_bit_for_bit(
    rng: np.random.Generator,
) -> None:
    box = np.sort(rng.normal(size=(40, 150, 3)), axis=-1)
    assert np.array_equal(calibrate_band(box, np.ones(150)), box)
    shrunk = calibrate_band(box, np.full(150, 0.5))
    np.testing.assert_allclose(
        shrunk[..., HI] - shrunk[..., LO], 0.5 * (box[..., HI] - box[..., LO])
    )
    assert np.array_equal(shrunk[..., POINT], box[..., POINT])
    with pytest.raises(ValueError):
        calibrate_band(box, np.zeros(150))


def test_forecaster_applies_the_pad_gamma_and_refuses_an_uncalibrated_pad(
    sim_cfg: SimConfig, scale: FroudeScale, deck_pads: PadConfig, tmp_path: Path
) -> None:
    _needs_model("residual_interval")
    src = _source(CAUSALITY_KEY, sim_cfg, scale, deck_pads)
    forecaster = OnlineForecaster(MODEL_ROOT / "residual_interval")
    assert set(forecaster.calibrated_pads) == {"aft", "cg"}
    for pad in ("aft", "cg"):
        feed = ShipMotionFeed.for_pad(src, pad, deck_pads, scale)
        feed.reset(30.0)
        fc = forecaster.forecast(feed)
        assert fc.pad_vz_box_m_s is not None and fc.pad_vz_gamma is not None
        gamma = forecaster.pad_vz_gamma(deck_pads.spec(pad).r_pad_frac[0])
        assert np.array_equal(fc.pad_vz_gamma, gamma)
        assert np.array_equal(fc.pad_vz_m_s, calibrate_band(fc.pad_vz_box_m_s, gamma))
        # Only pad v_z is calibrated: pad z keeps the raw box, roll/pitch their native bands.
        raw = forecast_from_channels(
            fc.channels_full,
            origin_channels_full=fc.origin_channels_full,
            t_origin_full_s=fc.t_origin_full_s,
            t_origin_model_s=fc.t_origin_model_s,
            full_s_per_model_s=feed.full_s_per_model_s,
            fs_full_hz=10.0,
            r_pad_full_m=feed.r_pad_full_m,
            scale=scale,
            model="raw",
            interval_levels=None,
        )
        assert np.array_equal(fc.pad_z_m, raw.pad_z_m)
        assert np.array_equal(fc.pad_vz_box_m_s, raw.pad_vz_m_s)
        assert np.array_equal(fc.roll_deg, raw.roll_deg)
    # A pad at an uncalibrated lever arm raises; a point model does not care.
    odd = ShipMotionFeed(src, (-30.0, 0.0, 0.0), scale)
    odd.reset(30.0)
    with pytest.raises(ValueError, match="no pad-v_z conformal calibration"):
        forecaster.forecast(odd)
    if _available("dlinear_ols"):
        point = OnlineForecaster(MODEL_ROOT / "dlinear_ols").forecast(odd)
        assert point.pad_vz_gamma is None
        assert np.array_equal(point.pad_vz_m_s[:, LO], point.pad_vz_m_s[:, HI])
    # An interval model with no calibration file at all raises too.
    bare = tmp_path / "residual_interval"
    bare.mkdir()
    meta = json.loads((MODEL_ROOT / "residual_interval" / "meta.json").read_text("utf-8"))
    for name in meta["files"]:
        if name != CONFORMAL_FILE:
            (bare / name).write_bytes((MODEL_ROOT / "residual_interval" / name).read_bytes())
    meta["files"].pop(CONFORMAL_FILE, None)
    (bare / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    feed = ShipMotionFeed.for_pad(src, "aft", deck_pads, scale)
    feed.reset(30.0)
    with pytest.raises(ValueError, match="no pad-v_z conformal calibration"):
        OnlineForecaster(bare).forecast(feed)


# ---------------------------------------------------------------------------------------
# band coverage on the tune pool -- reported, not gated
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tune_predictions(sim_cfg: SimConfig) -> dict[str, dict[str, np.ndarray]]:
    """ORT predictions of every fitted interval model on the tune pool, cached per module.

    Windows start every :data:`COVERAGE_ORIGIN_STRIDE` samples in each of the 135 tune
    realizations; bands and truths are kept at :data:`COVERAGE_LEADS` only. Full scale.
    """
    _needs_corpus()
    tune = pool_keys(sim_cfg)[1]
    starts = np.arange(0, 6000 - 200 - 150 + 1, COVERAGE_ORIGIN_STRIDE)
    leads = np.asarray(COVERAGE_LEADS)
    series = {
        key: corpus_frame(CORPUS_ROOT, key, FORECAST_CHANNELS)[list(FORECAST_CHANNELS)].to_numpy(
            np.float64
        )
        for key in tune
    }
    common = {
        "ss": np.asarray([str(key[0]) for key in tune for _ in starts]),
        "fold": np.asarray([i % 2 for i in range(len(tune)) for _ in starts]),
        "truth": np.concatenate(
            [np.stack([series[k][s + 199 + leads] for s in starts]) for k in tune]
        ),
    }
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in INTERVAL_MODELS:
        if not _available(name):
            continue
        forecaster = OnlineForecaster(MODEL_ROOT / name, "ort", intra_op_threads=8)
        band = np.concatenate(
            [
                forecaster.predict_windows(np.stack([series[k][s : s + 200] for s in starts]))[
                    :, leads - 1
                ]
                for k in tune
            ]
        )
        gammas = {
            pad: forecaster.pad_vz_gamma(frac)[leads - 1]
            for pad, frac in forecaster.calibrated_pads.items()
        }
        out[name] = {
            **common,
            "band": band,
            "smoke": np.asarray(forecaster.is_smoke),
            **{f"gamma_{pad}": g for pad, g in gammas.items()},
        }
    if not out:
        pytest.skip("no interval model fitted")
    return out


def _pad_vz(band: np.ndarray, truth: np.ndarray, x: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw pad-v_z box and the true pad v_z, both m/s **model** scale."""
    _, box = pad_vertical_band(band, x)
    true = truth[..., 5] + x * np.cos(np.radians(truth[..., 1])) * np.radians(truth[..., 4])
    return box * 0.2, true * 0.2


def _aft_x(deck_pads: PadConfig) -> float:
    x = deck_pads.spec("aft").r_pad_full_m(float(load_vessel_cached("frigate").length_m))[0]
    assert x == pytest.approx(-49.6)
    return float(x)


@pytest.mark.slow
def test_band_coverage_on_the_tune_pool(
    tune_predictions: dict[str, dict[str, np.ndarray]], deck_pads: PadConfig
) -> None:
    pads = {"aft": _aft_x(deck_pads), "cg": 0.0}
    rows: list[dict[str, object]] = []
    for name, pred in tune_predictions.items():
        band, truth, ss = pred["band"], pred["truth"], pred["ss"]
        cases: list[tuple[str, str, str, np.ndarray, np.ndarray, str]] = [
            ("native", "-", "roll_deg", band[..., 0, :], truth[..., 0], "deg"),
            ("native", "-", "pitch_deg", band[..., 1, :], truth[..., 1], "deg"),
        ]
        for pad, x in pads.items():
            box, true = _pad_vz(band, truth, x)
            cases.append(("raw_box", pad, "pad_vz", box, true, "m/s model"))
            cal = calibrate_band(box, pred[f"gamma_{pad}"])
            cases.append(("conformal", pad, "pad_vz", cal, true, "m/s model"))
        for band_kind, pad, quantity, qb, qt, units in cases:
            for group in ["all", *sorted(set(ss.tolist()))]:
                m = np.ones(ss.size, dtype=bool) if group == "all" else ss == group
                for j, lead in enumerate(COVERAGE_LEADS):
                    lo, hi, true = qb[m, j, LO], qb[m, j, HI], qt[m, j]
                    inside = (true >= lo) & (true <= hi)
                    rows.append(
                        {
                            "model": name,
                            "smoke_fit": bool(pred["smoke"]),
                            "pool": "dev_pool/tune",
                            "band": band_kind,
                            "pad": pad,
                            "quantity": quantity,
                            "units": units,
                            "ss": group,
                            "lead_full_s": lead / 10.0,
                            "lead_model_s": round(lead / 10.0 * 0.2, 6),
                            "nominal": 0.90,
                            "coverage": round(float(inside.mean()), 5),
                            "mean_width": round(float((hi - lo).mean()), 6),
                            "median_width": round(float(np.median(hi - lo)), 6),
                            "n_windows": int(inside.size),
                            "calibrated_on_this_pool": band_kind == "conformal"
                            or name == "residual_interval",
                        }
                    )
    pooled = [r for r in rows if r["band"] == "conformal" and r["ss"] == "all"]
    for r in pooled:
        print(f"{r['model']:>18} {r['pad']:>3} {r['lead_full_s']:>4} s: {r['coverage']}")
        # A smoke calibration uses 3 tune keys; it is gated only on the real, 135-key fit.
        if not r["smoke_fit"]:
            lo_ok, hi_ok = CALIBRATED_COVERAGE
            assert lo_ok <= float(str(r["coverage"])) <= hi_ok, r
    real = [r for r in rows if not r["smoke_fit"]]
    if real:
        _write_rows(RECORD_DIR / "coverage.csv", real)


def _gamma_2fold(
    box: np.ndarray, true: np.ndarray, fold: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Band (B) calibrated on one half of the tune keys and applied to the other."""
    pt = box[..., POINT]
    lo_h = np.maximum(pt - box[..., LO], 1e-9)
    hi_h = np.maximum(box[..., HI] - pt, 1e-9)
    score = np.maximum((pt - true) / lo_h, (true - pt) / hi_h)
    lo, hi = np.empty_like(pt), np.empty_like(pt)
    for f in (0, 1):
        cal, test = fold != f, fold == f
        n = int(cal.sum())
        k = int(np.ceil((n + 1) * 0.9))
        gamma = np.sort(score[cal], axis=0)[k - 1]
        lo[test], hi[test] = pt[test] - gamma * lo_h[test], pt[test] + gamma * hi_h[test]
    return lo, hi


def _quantile_bands(
    box: np.ndarray, true: np.ndarray, fold: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Construction (C): 5/95 % residual quantiles of the point pad v_z, in-sample and 2-fold."""
    pt = box[..., POINT]
    resid = true - pt
    q = np.quantile(resid, [0.05, 0.95], axis=0)
    lo, hi = np.empty_like(pt), np.empty_like(pt)
    for f in (0, 1):
        cal, test = fold != f, fold == f
        qf = np.quantile(resid[cal], [0.05, 0.95], axis=0)
        lo[test], hi[test] = pt[test] + qf[0], pt[test] + qf[1]
    return {
        "C_residual_quantiles_in_sample": (pt + q[0], pt + q[1]),
        "C_residual_quantiles_2fold": (lo, hi),
    }


@pytest.mark.slow
def test_band_feasibility_record(
    tune_predictions: dict[str, dict[str, np.ndarray]], deck_pads: PadConfig
) -> None:
    """Pre-registration evidence (plan amendment 2026-09-23): can any band pass the rule?

    For each pad-v_z band construction, the fraction of tune-pool forecasts whose band lies
    wholly inside +-0.16 m/s model, and whose roll and pitch bands also lie inside 3 / 2 deg
    (one sample of the interval rule), beside the TRUE future's fraction -- the most any
    forecaster can pass. Leads 7 / 8 / 9.3 s full (1.4 / 1.6 / 1.86 s model).
    """
    pads = {"aft": _aft_x(deck_pads), "cg": 0.0}
    idx = [COVERAGE_LEADS.index(lead) for lead in FEASIBILITY_LEADS]
    rows: list[dict[str, object]] = []
    for name, pred in tune_predictions.items():
        band, truth, ss, fold = (
            pred["band"][:, idx],
            pred["truth"][:, idx],
            pred["ss"],
            pred["fold"],
        )
        ang_band = (
            (band[..., 0, LO] >= -LIM_ROLL_DEG)
            & (band[..., 0, HI] <= LIM_ROLL_DEG)
            & (band[..., 1, LO] >= -LIM_PITCH_DEG)
            & (band[..., 1, HI] <= LIM_PITCH_DEG)
        )
        ang_true = (np.abs(truth[..., 0]) <= LIM_ROLL_DEG) & (
            np.abs(truth[..., 1]) <= LIM_PITCH_DEG
        )
        for pad, x in pads.items():
            box, true = _pad_vz(band, truth, x)
            cal = calibrate_band(box, pred[f"gamma_{pad}"][idx])
            constructions = {
                "A_raw_box": (box[..., LO], box[..., HI]),
                "B_conformal_deployed": (cal[..., LO], cal[..., HI]),
                "B_conformal_2fold": _gamma_2fold(box, true, fold),
                **_quantile_bands(box, true, fold),
                "TRUTH_base_rate": (true, true),
            }
            for construction, (lo, hi) in constructions.items():
                covered = (true >= lo) & (true <= hi)
                vz_in = (lo >= -LIM_VZ_M_S) & (hi <= LIM_VZ_M_S)
                ang = ang_true if construction == "TRUTH_base_rate" else ang_band
                for group in ["all", *sorted(set(ss.tolist()))]:
                    m = np.ones(ss.size, dtype=bool) if group == "all" else ss == group
                    for j, lead in enumerate(FEASIBILITY_LEADS):
                        is_truth = construction == "TRUTH_base_rate"
                        rows.append(
                            {
                                "model": name,
                                "smoke_fit": bool(pred["smoke"]),
                                "construction": construction,
                                "pad": pad,
                                "ss": group,
                                "lead_full_s": lead / 10.0,
                                "lead_model_s": round(lead / 10.0 * 0.2, 6),
                                "pad_vz_coverage": ""
                                if is_truth
                                else round(float(covered[m, j].mean()), 5),
                                "median_width_m_s_model": round(
                                    float(np.median(hi[m, j] - lo[m, j])), 6
                                ),
                                "frac_vz_band_inside": round(float(vz_in[m, j].mean()), 5),
                                "frac_all3_inside": round(float((vz_in & ang)[m, j].mean()), 5),
                                "n_windows": int(m.sum()),
                                "limits": "roll 3 deg, pitch 2 deg, pad v_z 0.16 m/s model",
                            }
                        )
    real = [r for r in rows if not r["smoke_fit"]]
    assert rows
    if real:
        _write_rows(RECORD_DIR / "band_feasibility.csv", real)
