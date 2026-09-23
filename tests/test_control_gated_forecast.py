"""``gated_forecast`` / ``gated_forecast_tcn``: commit on the forecast band at touchdown.

Contract checks (the new-controller set): shape and bounds, determinism under reset,
static-pad landing, and ``privileged=False`` -- plus what keeps this controller honest:

* it **needs the feed** and refuses to run without it (and refuses a privileged context);
* its rule is **the shared** ``QuiescenceRule`` applied at ``oracle_gated``'s touchdown time:
  given a stub forecaster that returns the TRUE future with a zero-width band, it commits
  exactly when ``oracle_gated`` does, on the 40 windows ``test_control_oracle_gated.py``
  uses (the same generator, the same seed);
* the band enters as ``max(|lo|, |hi|)``: a band that pokes past one limit on either side
  blocks the commit even when the point forecast is quiet;
* a window past the forecast horizon does not qualify (False, not an error);
* ``strict`` loads with 18 samples and applies;
* its clock is the feed's: a runner that forgets ``advance_to`` is caught.

The real forecaster (``residual_interval``) is exercised once, on one JONSWAP episode,
marked slow and skipped when ``artifacts/dmf/`` (or ``$RLD_DMF_ARTIFACTS``) lacks it. The
stub feed stands in for :class:`rld.deck.forecast.ShipMotionFeed` wherever the deck is
synthetic or static: the real feed samples dmf's channels and cannot run on
:class:`~rld.envs.platform.StaticDeckMotion`.

Units: metres, metres per second and seconds model scale; degrees.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from dmf.sim.generate import RealizationSpec

from _control_helpers import (
    assert_actions_in_space,
    committed_spec,
    deck_normal,
    feed,
    quiescence_windows,
    synthetic_obs,
)
from rld.control.base import PrivilegedContext
from rld.control.config import load_gated, load_gated_forecast
from rld.control.gated_forecast import BAND_HI, BAND_LO, GatedForecast
from rld.control.obs_view import obs_layout, view
from rld.control.oracle import OracleGated
from rld.control.registry import entry, make_controller
from rld.deck.forecast import DEFAULT_MODEL_ROOT, HI, LO, DeckForecast
from rld.envs.landing_env import DeckLandingAviary, EpisodeRecord

NAMES = ("gated_forecast", "gated_forecast_tcn")
NAME = "gated_forecast"

#: Absolute model time of the synthetic episodes' start inside the record, s model.
T0 = 10.0

#: Half-width of the plateau each true-future sample is held on in the stub forecast, s
#: model: queries within it read the sample exactly; anything else reads a loud value.
PLATEAU = 1e-6

MODEL_ROOT = Path(os.environ.get("RLD_DMF_ARTIFACTS", str(DEFAULT_MODEL_ROOT)))


# ---------------------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------------------


class StubFeed:
    """The clock half of :class:`~rld.deck.forecast.ShipMotionFeed`, for synthetic decks."""

    def __init__(self, t0_model_s: float = T0) -> None:
        """Start the clock at ``t0_model_s``, s model (as if ``reset`` had been called)."""
        self.clock_model_s = float(t0_model_s)

    def reset(self, t0_model_s: float) -> None:
        """Set the clock to the episode start, s model."""
        self.clock_model_s = float(t0_model_s)

    def advance_to(self, t_model_s: float) -> None:
        """Move the clock forward to ``t_model_s``, s model; never backwards."""
        assert t_model_s >= self.clock_model_s, "clock cannot run backwards"
        self.clock_model_s = float(t_model_s)


class StubForecaster:
    """Returns a prescribed forecast; counts calls. ``interval_levels`` settable."""

    def __init__(self, make: Any, levels: tuple[float, float] | None = (0.05, 0.95)) -> None:
        """Wrap ``make(feed) -> DeckForecast``; report ``levels`` as the band levels."""
        self._make = make
        self._levels = levels
        self.calls = 0

    @property
    def interval_levels(self) -> tuple[float, float] | None:
        """The reported band levels (None mimics a point model)."""
        return self._levels

    def forecast(self, feed: Any) -> DeckForecast:
        """Return ``make(feed)`` and count the call."""
        self.calls += 1
        return self._make(feed)


def make_forecast(
    t_origin: float,
    t_nodes: np.ndarray,
    roll: np.ndarray,
    pitch: np.ndarray,
    vz: np.ndarray,
) -> DeckForecast:
    """A :class:`DeckForecast` with arbitrary lead nodes and ``(H, 3)`` bands.

    Args:
        t_origin: Forecast origin, s model (absolute).
        t_nodes: Absolute node times, s model, strictly increasing, all after the origin.
        roll: ``(H, 3)`` roll band, degrees.
        pitch: ``(H, 3)`` pitch band, degrees.
        vz: ``(H, 3)`` pad v_z band, m/s model.
    """
    t_nodes = np.asarray(t_nodes, dtype=np.float64)
    n = t_nodes.size
    assert np.all(np.diff(t_nodes) > 0.0) and t_nodes[0] > t_origin
    lead = t_nodes - t_origin
    return DeckForecast(
        model="stub",
        interval_levels=(0.05, 0.95),
        t_origin_model_s=float(t_origin),
        t_origin_full_s=0.0,
        lead_model_s=lead,
        lead_full_s=lead * 5.0,
        roll_deg=np.asarray(roll, dtype=np.float64),
        pitch_deg=np.asarray(pitch, dtype=np.float64),
        pad_z_m=np.zeros((n, 3)),
        pad_vz_m_s=np.asarray(vz, dtype=np.float64),
        channels_full=np.zeros((n, 6, 3)),
        origin_channels_full=np.zeros(6),
        r_pad_full_m=(0.0, 0.0, 0.0),
        lam=1.0 / 25.0,
    )


def _band(values: np.ndarray) -> np.ndarray:
    """Zero-width band ``[v, v, v]`` from ``(H,)`` values."""
    return np.repeat(np.asarray(values, dtype=np.float64)[:, None], 3, axis=1)


def plateau_forecast(
    t_origin: float,
    centres: np.ndarray,
    bands: tuple[np.ndarray, np.ndarray, np.ndarray],
    loud: tuple[float, float, float] = (10.0, -10.0, 0.5),
    t_end: float | None = None,
) -> DeckForecast:
    """A forecast holding sample ``j`` exactly on ``[c_j - PLATEAU, c_j + PLATEAU]``.

    Everywhere else -- midway between samples, before the first, and at ``t_end`` -- it is
    loud (far outside every limit), so a controller sampling anywhere but the control-rate
    instants at the predicted touchdown would read a violation. Mirrors the oracle test,
    whose context is loud between the control-rate physics samples.

    Args:
        t_origin: Origin, s model.
        centres: ``(n,)`` sample instants, s model (absolute).
        bands: ``(roll, pitch, vz)``, each ``(n, 3)`` ``[lo, point, hi]``.
        loud: The loud value per quantity.
        t_end: Last node, s model; default ``t_origin + 3.0`` (the real horizon).
    """
    c = np.asarray(centres, dtype=np.float64)
    spacing = float(c[1] - c[0]) if c.size > 1 else 1.0 / 30.0
    loud_row = [np.full(3, v) for v in loud]
    times: list[float] = [float(c[0] - 0.5 * spacing)]
    rows: list[list[np.ndarray]] = [loud_row]
    for j, cj in enumerate(c):
        sample = [np.asarray(b[j], dtype=np.float64) for b in bands]
        times += [float(cj - PLATEAU), float(cj + PLATEAU)]
        rows += [sample, sample]
        if j + 1 < c.size:
            times.append(float(0.5 * (cj + c[j + 1])))
            rows.append(loud_row)
    end = t_origin + 3.0 if t_end is None else t_end
    if end > times[-1]:
        times.append(float(end))
        rows.append(loud_row)
    stacked = [np.stack([r[q] for r in rows]) for q in range(3)]
    t = np.asarray(times)
    keep = t <= end + 1e-15
    return make_forecast(t_origin, t[keep], *(s[keep] for s in stacked))


def quiet_forecast(feed: Any, horizon_s: float = 3.0) -> DeckForecast:
    """A static deck's perfect forecast: zero everywhere, zero-width, 50 Hz, 3.0 s ahead."""
    t = feed.clock_model_s + np.arange(1, int(round(horizon_s * 50)) + 1) / 50.0
    zeros = np.zeros((t.size, 3))
    return make_forecast(feed.clock_model_s, t, zeros, zeros, zeros)


def build(
    forecaster: Any | None = None, config_path: Path | None = None, name: str = NAME
) -> GatedForecast:
    """A ``GatedForecast`` with a (stub) forecaster injected."""
    path = entry(name).config_path if config_path is None else config_path
    stub = StubForecaster(quiet_forecast) if forecaster is None else forecaster
    return GatedForecast(load_gated_forecast(path), committed_spec(), name=name, forecaster=stub)


def hover_obs(env_obs_cfg: Any, controller: GatedForecast) -> np.ndarray:
    hover = controller.gated.hover_height_m
    return synthetic_obs(env_obs_cfg, rel_position_w=(0.0, 0.0, -hover), height_m=hover)


def drive(
    controller: GatedForecast, stub_feed: StubFeed, obs_list: list[np.ndarray]
) -> list[np.ndarray]:
    """Act on observations the way the runner will: advance the feed, then act."""
    spec = controller.spec
    layout = obs_layout(spec.observation)
    t0 = stub_feed.clock_model_s
    actions = []
    for k, obs in enumerate(obs_list):
        stamped = obs.copy()
        stamped[layout.slices["time_fraction"]] = k * spec.ctrl_dt_s / spec.landing.episode_len_s
        stub_feed.advance_to(t0 + k * spec.ctrl_dt_s)
        actions.append(controller.act(stamped))
    return actions


def commits_on(env_obs_cfg: Any, controller: GatedForecast, forecast: DeckForecast) -> bool:
    """Reset with a stub feed at ``T0``, act once at hover, return whether it committed."""
    controller._forecaster = StubForecaster(lambda _feed: forecast)  # noqa: SLF001
    stub_feed = StubFeed()
    controller.reset(0, None, stub_feed)  # type: ignore[arg-type]
    drive(controller, stub_feed, [hover_obs(env_obs_cfg, controller)])
    return controller.committed


def window_centres(env_obs_cfg: Any, controller: GatedForecast) -> np.ndarray:
    """The absolute instants the controller samples at hover at ``t = 0`` (after a reset)."""
    obs = hover_obs(env_obs_cfg, controller)
    return controller.window_times_model_s(view(obs, obs_layout(env_obs_cfg)), 0.0)


# ---------------------------------------------------------------------------------------
# flags, configs, contract
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_privileged_flag_is_false_and_the_feed_is_needed(name: str) -> None:
    """Not privileged (registry and instance); needs the feed (registry and instance)."""
    assert entry(name).privileged is False
    assert entry(name).needs_motion_feed is True
    controller = make_controller(name)
    assert controller.privileged is False
    assert controller.needs_motion_feed is True
    assert controller.name == name


@pytest.mark.parametrize("name", NAMES)
def test_config_is_gated_yaml_plus_the_forecaster(name: str) -> None:
    """Every inherited key equals gated.yaml's; forecaster dir and the 90 % interval."""
    cfg = load_gated_forecast(entry(name).config_path)
    gated = load_gated(entry("gated").config_path)
    for field in dataclasses.fields(gated):
        assert getattr(cfg, field.name) == getattr(gated, field.name), field.name
    assert cfg.thresholds == "permissive"
    assert cfg.fallback_commit_s is None
    assert cfg.interval == (0.05, 0.95)
    model = "residual_interval" if name == "gated_forecast" else "tcn_quantile"
    assert cfg.forecaster_dir.is_absolute()
    assert cfg.forecaster_dir.parts[-3:] == ("artifacts", "dmf", model)


def test_band_columns_match_the_adapter() -> None:
    """The controller's lo/hi column indices are rld.deck.forecast's."""
    assert (BAND_LO, BAND_HI) == (LO, HI)


def test_interval_must_match_the_forecaster() -> None:
    """A forecaster whose band is not the config's 5/95 % (or a point model) is refused."""
    with pytest.raises(ValueError, match="interval"):
        build(StubForecaster(quiet_forecast, levels=(0.1, 0.9)))
    with pytest.raises(ValueError, match="interval"):
        build(StubForecaster(quiet_forecast, levels=None))


def test_refuses_to_run_without_a_feed_or_with_a_context(env_obs_cfg: Any) -> None:
    """No feed -> ValueError at reset; a privileged context -> ValueError; act first -> error."""
    controller = build()
    with pytest.raises(ValueError, match="ShipMotionFeed"):
        controller.reset(0)
    with pytest.raises(ValueError, match="ShipMotionFeed"):
        controller.reset(0, None, None)
    ctx = PrivilegedContext(
        t0_model_s=T0,
        physics_dt_s=1.0 / 240.0,
        t_episode_s=np.zeros(1),
        position_m=np.zeros((1, 3)),
        velocity_m_s=np.zeros((1, 3)),
        roll_deg=np.zeros(1),
        pitch_deg=np.zeros(1),
        normal=np.array([[0.0, 0.0, 1.0]]),
        realization_key=("SS5", 180.0, 0.0, "synthetic", 0),
    )
    with pytest.raises(ValueError, match="not privileged"):
        controller.reset(0, ctx, StubFeed())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="before reset"):
        build().act(hover_obs(env_obs_cfg, controller))


def test_clock_must_follow_the_feed(env_obs_cfg: Any) -> None:
    """A runner that does not advance the feed, or advances it twice, is caught.

    The observations carry a consistent ``time_fraction`` throughout, so it is the feed
    check -- not :class:`~rld.control.base.StepClock` -- that fires.
    """
    controller = build()
    obs = hover_obs(env_obs_cfg, controller)
    dt = controller.spec.ctrl_dt_s
    stub_feed = StubFeed()
    controller.reset(0, None, stub_feed)  # type: ignore[arg-type]
    drive(controller, stub_feed, [obs] * 20)  # advanced properly: fine

    # Forgot advance_to: k = 0 is at t0 and passes; k = 1 is caught.
    controller.reset(0, None, StubFeed())  # type: ignore[arg-type]
    feed(controller, [obs], dt, 12.0)
    stamped = obs.copy()
    stamped[obs_layout(env_obs_cfg).slices["time_fraction"]] = dt / 12.0
    with pytest.raises(RuntimeError, match="feed clock"):
        controller.act(stamped)

    # Advanced one step too far before k = 0.
    stub_feed = StubFeed()
    controller.reset(0, None, stub_feed)  # type: ignore[arg-type]
    stub_feed.advance_to(T0 + dt)
    with pytest.raises(RuntimeError, match="feed clock"):
        feed(controller, [obs], dt, 12.0)


def test_actions_stay_in_the_shared_space_on_arbitrary_observations(env_obs_cfg: Any) -> None:
    """Any observation inside the declared bounds gives a float32 action in the unit ball."""
    from rld.envs.observation import observation_space

    controller = build()
    space = observation_space(env_obs_cfg)
    rng = np.random.default_rng(3)
    obs_list = [
        rng.uniform(space.low.astype(np.float64), space.high.astype(np.float64)).astype(np.float32)
        for _ in range(500)
    ]
    stub_feed = StubFeed()
    controller.reset(0, None, stub_feed)  # type: ignore[arg-type]
    assert_actions_in_space(drive(controller, stub_feed, obs_list))


# ---------------------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------------------


def test_window_is_the_oracles_touchdown_law(env_obs_cfg: Any) -> None:
    """Samples at t0 + t + height / descent_rate + j / 30, j = 0..11 (oracle_gated's law)."""
    controller = build()
    controller.reset(0, None, StubFeed())  # type: ignore[arg-type]
    oracle = make_controller("oracle_gated", committed_spec())
    assert isinstance(oracle, OracleGated)
    obs = hover_obs(env_obs_cfg, controller)
    v = view(obs, obs_layout(env_obs_cfg))
    for t in (0.0, 1.5, 7.0):
        times = controller.window_times_model_s(v, t)
        assert times.shape == (12,)
        assert times[0] == T0 + oracle.predicted_touchdown_s(v, t)
        np.testing.assert_allclose(np.diff(times), 1.0 / 30.0, rtol=0, atol=1e-12)
    hover, rate = controller.gated.hover_height_m, controller.pid.descent_rate_m_s
    assert window_centres(env_obs_cfg, controller)[0] == pytest.approx(T0 + hover / rate)


def test_true_future_zero_width_band_commits_exactly_when_the_oracle_does(
    env_obs_cfg: Any,
) -> None:
    """On the oracle test's 40 windows: same verdict as oracle_gated and QuiescenceRule.

    The stub forecaster returns the TRUE future window at the control-rate instants from
    the predicted touchdown with ``lo = point = hi``, and loud values everywhere else; the
    oracle's context is the same window, loud between its control-rate physics samples.
    """
    spec = committed_spec()
    controller = build()
    oracle = make_controller("oracle_gated", spec)
    assert isinstance(oracle, OracleGated)
    assert controller.rule == oracle.rule
    rule = controller.rule
    n, stride = rule.n_samples, oracle.window_stride
    hover = controller.gated.hover_height_m
    t_td = hover / oracle.pid.descent_rate_m_s
    size, dt = 3001, 1.0 / 240.0
    obs = hover_obs(env_obs_cfg, controller)
    controller.reset(0, None, StubFeed())  # type: ignore[arg-type]
    centres = window_centres(env_obs_cfg, controller)
    verdicts = []
    for roll, pitch, vz in quiescence_windows(np.random.default_rng(20260922), rule.limits, n):
        expected = rule.verdict(roll, pitch, vz)

        # oracle_gated, exactly as test_control_oracle_gated.py builds it.
        roll_all, pitch_all, vz_all = np.full(size, 10.0), np.full(size, -10.0), np.full(size, 0.5)
        start = int(np.ceil(t_td / dt - 1e-9))
        index = start + stride * np.arange(n)
        roll_all[index], pitch_all[index], vz_all[index] = roll, pitch, vz
        normal = np.array([deck_normal(r, q) for r, q in zip(roll_all, pitch_all, strict=True)])
        context = PrivilegedContext(
            t0_model_s=T0,
            physics_dt_s=dt,
            t_episode_s=np.arange(size) * dt,
            position_m=np.tile([0.0, 0.0, 1.0], (size, 1)),
            velocity_m_s=np.column_stack([np.zeros(size), np.zeros(size), vz_all]),
            roll_deg=roll_all,
            pitch_deg=pitch_all,
            normal=normal,
            realization_key=("SS5", 180.0, 0.0, "synthetic", 0),
        )
        oracle.reset(0, context)
        feed(oracle, [obs], spec.ctrl_dt_s, 12.0)

        # gated_forecast: the same window is its (perfect, zero-width) forecast.
        forecast = plateau_forecast(T0, centres, (_band(roll), _band(pitch), _band(vz)))
        committed = commits_on(env_obs_cfg, controller, forecast)

        assert committed == oracle.committed == expected, (roll, pitch, vz)
        verdicts.append(expected)
    assert any(verdicts) and not all(verdicts)


def _quiet_bands(n: int, limits: Any, fraction: float = 0.5) -> list[np.ndarray]:
    """Zero-width bands at ``fraction`` of each limit, alternating sign, ``(n, 3)`` each."""
    sign = np.where(np.arange(n) % 2, -1.0, 1.0)
    return [
        _band(fraction * sign * lim)
        for lim in (limits.roll_deg, limits.pitch_deg, limits.pad_vz_m_s)
    ]


@pytest.mark.parametrize(
    ("quantity", "column", "sample"),
    [(2, HI, 6), (0, LO, 0), (1, HI, 11), (2, LO, 3)],
)
def test_band_poking_past_one_limit_blocks_the_commit(
    env_obs_cfg: Any, quantity: int, column: int, sample: int
) -> None:
    """Quiet point, one band edge 5 % past one limit at one sample: no commit.

    The control: the same band with that edge 5 % inside the limit commits, so it is the
    band edge -- ``max(|lo|, |hi|)`` -- that decides, not the point forecast.
    """
    controller = build()
    controller.reset(0, None, StubFeed())  # type: ignore[arg-type]
    centres = window_centres(env_obs_cfg, controller)
    lim = controller.limits
    limit = (lim.roll_deg, lim.pitch_deg, lim.pad_vz_m_s)[quantity]
    sign = 1.0 if column == HI else -1.0

    def with_edge(factor: float) -> DeckForecast:
        bands = [np.zeros((12, 3)) for _ in range(3)]  # point exactly quiet
        bands[quantity][sample, column] = sign * factor * limit
        return plateau_forecast(T0, centres, (bands[0], bands[1], bands[2]))

    assert commits_on(env_obs_cfg, controller, with_edge(0.95))
    assert not commits_on(env_obs_cfg, controller, with_edge(1.05))


def test_wide_symmetric_band_inside_the_limits_commits(env_obs_cfg: Any) -> None:
    """``lo = -0.9 limit``, ``hi = +0.9 limit`` around a zero point: every |edge| inside."""
    controller = build()
    controller.reset(0, None, StubFeed())  # type: ignore[arg-type]
    centres = window_centres(env_obs_cfg, controller)
    lim = controller.limits
    bands = [
        np.tile([-0.9 * x, 0.0, 0.9 * x], (12, 1))
        for x in (lim.roll_deg, lim.pitch_deg, lim.pad_vz_m_s)
    ]
    assert commits_on(env_obs_cfg, controller, plateau_forecast(T0, centres, tuple(bands)))


def test_window_past_the_horizon_does_not_qualify(env_obs_cfg: Any) -> None:
    """Last sample beyond t_end_model_s -> False (not an exception); just inside -> commit."""
    controller = build()
    controller.reset(0, None, StubFeed())  # type: ignore[arg-type]
    centres = window_centres(env_obs_cfg, controller)
    bands = _quiet_bands(12, controller.limits)
    short = plateau_forecast(T0, centres, tuple(bands), t_end=float(centres[-1]) - 1e-4)
    assert short.t_end_model_s < centres[-1]
    assert not commits_on(env_obs_cfg, controller, short)
    obs = hover_obs(env_obs_cfg, controller)
    assert controller._rule_fires(view(obs, obs_layout(env_obs_cfg)), 0.0) is False  # noqa: SLF001
    enough = plateau_forecast(T0, centres, tuple(bands), t_end=float(centres[-1]) + 1e-4)
    assert commits_on(env_obs_cfg, controller, enough)


def test_forecaster_is_consulted_only_when_the_rule_is(env_obs_cfg: Any) -> None:
    """Away from hover the rule is not consulted, so the forecaster is not run."""
    stub = StubForecaster(quiet_forecast)
    controller = build(stub)
    high = synthetic_obs(env_obs_cfg, rel_position_w=(0.0, 0.0, -0.9), height_m=0.9)
    stub_feed = StubFeed()
    controller.reset(0, None, stub_feed)  # type: ignore[arg-type]
    drive(controller, stub_feed, [high] * 5)
    assert stub.calls == 0 and not controller.committed
    drive_feed = StubFeed()
    controller.reset(0, None, drive_feed)  # type: ignore[arg-type]
    drive(controller, drive_feed, [hover_obs(env_obs_cfg, controller)])
    assert stub.calls == 1 and controller.committed


def _strict_config(tmp_path: Path) -> Path:
    """gated_forecast.yaml with ``thresholds: strict``, gains_from made absolute."""
    source = entry(NAME).config_path
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["thresholds"] = "strict"
    raw["gains_from"] = str((source.parent / raw["gains_from"]).resolve())
    path = tmp_path / "gated_forecast_strict.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_strict_loads_with_18_samples_and_applies(env_obs_cfg: Any, tmp_path: Path) -> None:
    """STRICT: 18 samples (0.6 s at 30 Hz), 1.5 deg / 1.0 deg / 0.08 m/s, and it bites.

    * 18 samples at 2.0 deg roll: quiet under permissive (3.0), loud under strict (1.5).
    * 18 samples quiet under strict except sample 15: permissive (12 samples) commits,
      strict does not -- the longer window is read.
    * 18 samples at half the strict limits: both commit.
    """
    strict = build(config_path=_strict_config(tmp_path))
    permissive = build()
    assert strict.gated.thresholds == "strict"
    assert strict.rule.n_samples == 18 and permissive.rule.n_samples == 12
    lim = strict.limits
    assert (lim.roll_deg, lim.pitch_deg) == (1.5, 1.0)
    assert lim.pad_vz_m_s == pytest.approx(0.08, abs=1e-12)
    strict.reset(0, None, StubFeed())  # type: ignore[arg-type]
    permissive.reset(0, None, StubFeed())  # type: ignore[arg-type]
    centres = window_centres(env_obs_cfg, strict)
    assert centres.shape == (18,)
    np.testing.assert_array_equal(window_centres(env_obs_cfg, permissive), centres[:12])

    def forecast(roll: np.ndarray, pitch: np.ndarray, vz: np.ndarray) -> DeckForecast:
        return plateau_forecast(T0, centres, (_band(roll), _band(pitch), _band(vz)))

    zeros = np.zeros(18)
    mid_roll = forecast(np.full(18, 2.0), zeros, zeros)
    assert commits_on(env_obs_cfg, permissive, mid_roll)
    assert not commits_on(env_obs_cfg, strict, mid_roll)

    late = np.full(18, 0.04)
    late[15] = 0.12  # > 0.08 strict, < 0.16 permissive; outside the 12-sample window anyway
    late_forecast = forecast(zeros, zeros, late)
    assert commits_on(env_obs_cfg, permissive, late_forecast)
    assert not commits_on(env_obs_cfg, strict, late_forecast)

    quiet = forecast(np.full(18, 0.75), np.full(18, -0.5), np.full(18, 0.04))
    assert commits_on(env_obs_cfg, strict, quiet)
    assert commits_on(env_obs_cfg, permissive, quiet)


# ---------------------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------------------


def run_episode_with_feed(
    env: DeckLandingAviary, controller: GatedForecast, seed: int, motion_feed: Any
) -> tuple[list[np.ndarray], EpisodeRecord]:
    """One episode, fed the way the runner will: feed reset to t0, advanced before act."""
    obs, _ = env.reset(seed=seed)
    t0 = float(env.record.t0_model_s)
    motion_feed.reset(t0)
    controller.reset(seed, None, motion_feed)
    dt = controller.spec.ctrl_dt_s
    actions: list[np.ndarray] = []
    max_steps = int(env.cfg.total_len_s * env.cfg.ctrl_freq_hz) + 1
    for k in range(max_steps):
        motion_feed.advance_to(t0 + k * dt)
        action = controller.act(obs)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return actions, env.record


@pytest.mark.pybullet
def test_deterministic_under_reset(landing_env: Any, static_motion: Any) -> None:
    """Same seed, same instance, another episode in between: bit-identical actions."""
    env = landing_env(static_motion)
    controller = build()
    first, _ = run_episode_with_feed(env, controller, 3, StubFeed())
    run_episode_with_feed(env, controller, 4, StubFeed())
    again, _ = run_episode_with_feed(env, controller, 3, StubFeed())
    np.testing.assert_array_equal(np.stack(first), np.stack(again))
    assert_actions_in_space(first)


@pytest.mark.pybullet
@pytest.mark.slow
def test_static_pad_landing_succeeds(landing_env: Any, static_motion: Any) -> None:
    """A static deck's (perfect) forecast is always quiescent: ten out of ten."""
    env = landing_env(static_motion)
    controller = build()
    for seed in range(10):
        _, record = run_episode_with_feed(env, controller, seed, StubFeed())
        assert record.outcome == "success", (seed, record.as_row())
        assert controller.commit_time_s is not None


@pytest.mark.pybullet
@pytest.mark.slow
def test_real_residual_interval_forecaster_runs_one_jonswap_episode(
    landing_env: Any, deck_source: Any, deck_pads: Any, deck_scaling: Any
) -> None:
    """The real feed and the real ORT forecaster, one SS4 episode: runs, stays in the space.

    A smoke run, not an evaluation: the outcome is not asserted, only that the episode
    ends in one of the outcome classes, every action is in the shared space, and the
    forecaster was consulted.
    """
    from rld.deck.forecast import OnlineForecaster, ShipMotionFeed

    model_dir = MODEL_ROOT / "residual_interval"
    if not (model_dir / "meta.json").is_file():
        pytest.skip(f"no fitted residual_interval under {MODEL_ROOT}")
    forecaster = OnlineForecaster(model_dir, backend="ort")

    class Counting:
        calls = 0

        @property
        def interval_levels(self) -> tuple[float, float] | None:
            return forecaster.interval_levels

        def forecast(self, f: Any) -> DeckForecast:
            Counting.calls += 1
            return forecaster.forecast(f)

    spec = RealizationSpec(
        sea_state="SS4", heading_deg=180.0, speed_kn=6.0, vessel="frigate", seed=27
    )
    motion = deck_source(spec)
    env = landing_env(motion, pad="aft")
    controller = build(Counting())
    motion_feed = ShipMotionFeed.for_pad(motion, "aft", deck_pads, deck_scaling.froude_scale())
    actions, record = run_episode_with_feed(env, controller, 7, motion_feed)
    assert_actions_in_space(actions)
    assert record.outcome in {"success", "hard_landing", "off_pad", "bounce", "crash", "timeout"}
    assert Counting.calls > 0
