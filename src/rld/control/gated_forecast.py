"""``gated_forecast``: hover above the deck, commit when the **forecast** touchdown window is quiet.

**Not privileged** (``privileged=False``; ``needs_motion_feed=True`` in the registry). It
never sees the episode's true future. What it consumes, and all it consumes beyond the
observation ``gated`` reads, is:

* the **past-only ship-motion history** of dmf's six clean channels (roll, pitch, heave and
  their rates, full scale), through the :class:`~rld.deck.forecast.ShipMotionFeed` the
  runner hands to ``reset`` -- an ideal (noise-free, zero-latency) ship motion reference
  unit. The feed's clock is advanced by the runner alone and it refuses to read past it, so
  nothing later than "now" reaches the controller; its 200-sample, 10 Hz **full-scale**
  lookback (20 s full = 4.0 s model) starts before the episode, because the ship was moving
  before the drone arrived (P2-D7, P4-D3);
* a **fitted dmf forecaster** run on that history
  (:class:`~rld.deck.forecast.OnlineForecaster`, ONNX Runtime on the CPU), trained on the
  P3-D2 dev pool, disjoint from every frozen evaluation realization (P4-D1). It returns, per
  lead, a ``[lo, point, hi]`` band on roll (degrees), pitch (degrees) and pad deck-point
  ``v_z`` (metres per second **model** scale), out to 3.0 s model (15 s full).

Two registry entries share this class and differ only in ``forecaster_dir``:
``gated_forecast`` (dmf's ``residual_interval``: DLinear-OLS point plus fixed empirical 5/95 %
residual quantiles) and ``gated_forecast_tcn`` (dmf's learned ``tcn_quantile`` fan).

The law
-------
Everything except the rule is ``gated``'s (:class:`~rld.control.gated.GatedBase`):
``pid_feedforward``'s tuned gains by reference, hover at ``hover_height_m``, commit only with
the lateral gate open and the drone within ``hover_tolerance_m`` of hover, drop the commit
only if the lateral gate closes, and with ``fallback_commit_s: null`` a never-quiet forecast
times out and is reported as such.

The rule (forecast window)
--------------------------
At episode time ``t`` (seconds model scale, from the controller's own step counter) the
controller predicts its touchdown with the **same law as** ``oracle_gated``,
:meth:`~rld.control.gated.GatedBase.predicted_touchdown_s`:
``t_td = t + max(height, 0) / descent_rate``. It forecasts from the feed and samples the band
at the absolute model times ``t0 + t_td + j / ctrl_freq``, ``j = 0 .. n - 1``, where ``n`` is
the shared rule's sample count (12 permissive, 18 strict, at 30 Hz) and ``t0`` is the feed's
clock at ``reset``. Between the forecast's own 50 Hz model grid points the band is linearly
interpolated (:meth:`~rld.deck.forecast.DeckForecast.band_at`): an interpolation of the
**forecast**, never of the deck (P4-D3).

Each sample hands ``max(|lo|, |hi|)`` of roll, pitch and pad ``v_z`` to **the shared**
:class:`rld.control.quiescence.QuiescenceRule` built by ``GatedBase`` -- dmf's interval rule:
the whole 90 % band must lie inside ``+/- limit``, at every sample. So the three gated
controllers apply one predicate and differ only in which samples they read: the observed
past (``gated``), the true future (``oracle_gated``, privileged), or the forecast band
(here). Thresholds are dmf's ``PERMISSIVE``/``STRICT``, Froude-converted (angles unchanged;
0.8 -> 0.16 m/s, 2.0 -> 0.4 s permissive; 0.4 -> 0.08 m/s, 3.0 -> 0.6 s strict at
``lam = 1/25``), applied to the pad deck-point ``v_z`` (P3-D3 deviation, continued).

A window whose last sample lies beyond the forecast's last lead (``t_end_model_s``) does
not qualify: the rule returns False rather than extrapolate.

The forecaster runs only when the rule is consulted (lateral gate open, at hover, not yet
committed); the decision is the same as forecasting every step, at less cost.

Clock discipline
----------------
The controller's episode time and the feed's clock must describe the same instant. At every
``act`` it checks ``|feed.clock_model_s - (t0 + k / ctrl_freq)| <= 0.5 / ctrl_freq`` and
raises otherwise -- the symptom of a runner that forgot ``feed.advance_to`` or advanced it
twice.

Units: degrees; metres, metres per second and seconds **model** scale unless marked full.

Configs: ``configs/control/gated_forecast.yaml``, ``configs/control/gated_forecast_tcn.yaml``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from dmf.typedefs import FloatArray

from rld.control.base import ControlSpec, PrivilegedContext
from rld.control.config import GatedForecastConfig, load_gated_forecast
from rld.control.gated import GatedBase
from rld.control.obs_view import ObsView

if TYPE_CHECKING:
    # Type-only: rld.deck.forecast imports torch; the forecaster is built on first use.
    from rld.deck.forecast import DeckForecast, ShipMotionFeed

__all__ = [
    "BAND_HI",
    "BAND_LO",
    "BandForecaster",
    "GatedForecast",
    "make_gated_forecast",
    "make_gated_forecast_tcn",
]

#: Column indices of ``lo`` and ``hi`` in a :class:`~rld.deck.forecast.DeckForecast` band.
#: Mirrors ``rld.deck.forecast.LO`` / ``HI`` without importing torch at registry import;
#: a test asserts they agree.
BAND_LO: int = 0
BAND_HI: int = 2

type _Quantity = Literal["roll_deg", "pitch_deg", "pad_vz_m_s"]

#: The quantities the rule reads from the band, in :meth:`QuiescenceRule.verdict` order.
_QUANTITIES: tuple[_Quantity, _Quantity, _Quantity] = ("roll_deg", "pitch_deg", "pad_vz_m_s")


class BandForecaster(Protocol):
    """What the controller needs from a forecaster (:class:`~rld.deck.forecast.OnlineForecaster`).

    Attributes:
        interval_levels: The band's quantile levels, dimensionless, e.g. ``(0.05, 0.95)``;
            None for a point model (refused by this controller).
    """

    @property
    def interval_levels(self) -> tuple[float, float] | None:
        """The band's quantile levels, or None for a point model."""
        ...

    def forecast(self, feed: "ShipMotionFeed") -> "DeckForecast":
        """Forecast from the feed's current window without moving its clock."""
        ...


class GatedForecast(GatedBase):
    """``gated_forecast``: commit when the forecast band at predicted touchdown is quiet.

    Not privileged; requires a :class:`~rld.deck.forecast.ShipMotionFeed` at every
    ``reset``. See the module docstring.

    Attributes:
        gated: The full config, forecaster directory and interval included.
    """

    name = "gated_forecast"
    privileged = False
    needs_motion_feed = True

    def __init__(
        self,
        cfg: GatedForecastConfig,
        spec: ControlSpec,
        *,
        name: str | None = None,
        forecaster: BandForecaster | None = None,
    ) -> None:
        """Build the controller. The forecaster is loaded once, on first use.

        Args:
            cfg: The gated-forecast config.
            spec: Committed environment configs.
            name: Registry name when registered under another name
                (``gated_forecast_tcn``); the class name otherwise.
            forecaster: An already-built forecaster (tests; a stub with the same
                interface). When None, :class:`~rld.deck.forecast.OnlineForecaster` is
                built from ``cfg.forecaster_dir`` with the ORT backend on first use, so
                constructing the controller (as the registry and the config hash do) does
                not need the artifacts.

        Raises:
            ValueError: If an injected forecaster's interval levels differ from
                ``cfg.interval``.
        """
        super().__init__(cfg, spec)
        self.gated: GatedForecastConfig = cfg
        if name is not None:
            self.name = name
        self._forecaster: BandForecaster | None = None
        if forecaster is not None:
            self._forecaster = self._checked(forecaster)
        self._feed: ShipMotionFeed | None = None
        self._t0_model_s = 0.0

    # ------------------------------------------------------------------ forecaster

    def _checked(self, forecaster: BandForecaster) -> BandForecaster:
        """Return the forecaster after checking its band levels against the config.

        Args:
            forecaster: A built forecaster.

        Returns:
            The same object.

        Raises:
            ValueError: If its ``interval_levels`` are not ``cfg.interval``.
        """
        levels = forecaster.interval_levels
        if not self.gated.interval_matches(levels):
            raise ValueError(
                f"{self.name}: config interval {self.gated.interval} but the forecaster's "
                f"band levels are {levels}"
            )
        return forecaster

    @property
    def forecaster(self) -> BandForecaster:
        """The forecaster, loaded from ``cfg.forecaster_dir`` on first access (ORT, CPU).

        Raises:
            FileNotFoundError: If the model directory is incomplete.
            ValueError: On an artifact hash mismatch or an interval mismatch.
        """
        if self._forecaster is None:
            from rld.deck.forecast import OnlineForecaster

            self._forecaster = self._checked(
                OnlineForecaster(self.gated.forecaster_dir, backend="ort")
            )
        return self._forecaster

    # ------------------------------------------------------------------ contract

    def reset(
        self,
        seed: int,
        context: PrivilegedContext | None = None,
        motion_feed: "ShipMotionFeed | None" = None,
    ) -> None:
        """Clear per-episode state and take this episode's ship-motion feed.

        Args:
            seed: Episode seed; recorded only.
            context: Must be ``None``: this controller is not privileged.
            motion_feed: The episode's feed, already ``reset`` to the episode start
                ``t0`` (seconds model scale). Required.

        Raises:
            ValueError: If ``motion_feed`` is None (the controller refuses to degrade
                silently into ``gated``), or a privileged context is passed.
            RuntimeError: If the feed has not been reset.
            FileNotFoundError: If the forecaster is loaded here for the first time and its
                directory is incomplete.
        """
        if context is not None:
            raise ValueError(
                f"{self.name} is not privileged and must never receive a PrivilegedContext"
            )
        if motion_feed is None:
            raise ValueError(
                f"{self.name} needs a ShipMotionFeed at every reset (needs_motion_feed=True): "
                "build one with ShipMotionFeed.for_pad(env.motion, env.pad, pads, scale), "
                "reset it to the episode start, and advance it before every act"
            )
        _ = self.forecaster  # load once, before the first act, so act() has no cold start
        self._t0_model_s = float(motion_feed.clock_model_s)
        self._feed = motion_feed
        super().reset(seed, None, None)

    # ------------------------------------------------------------------ the rule

    def _check_clock(self, t_s: float) -> None:
        """Raise unless the feed's clock is this step's instant, within half a step.

        Args:
            t_s: Episode time of the observation being acted on, seconds model scale.

        Raises:
            RuntimeError: If no feed (``act`` before ``reset``) or the clocks disagree.
        """
        if self._feed is None:
            raise RuntimeError(f"{self.name}: act() before reset(motion_feed=...)")
        expected = self._t0_model_s + t_s
        clock = float(self._feed.clock_model_s)
        if abs(clock - expected) > 0.5 * self.spec.ctrl_dt_s:
            raise RuntimeError(
                f"{self.name}: feed clock {clock:.6f} s but the controller is at t0 + t = "
                f"{expected:.6f} s (model): the runner must call feed.advance_to(t0 + k * "
                "ctrl_dt) exactly once before every act()"
            )

    def _vertical_m_s(self, v: ObsView, t_s: float) -> float:
        """Check the feed's clock, then run ``GatedBase``'s hover-and-commit.

        Args:
            v: The observation, world frame.
            t_s: Episode time, seconds model scale.

        Returns:
            Metres per second model scale, world ``z``, before feedforward.
        """
        self._check_clock(t_s)
        return super()._vertical_m_s(v, t_s)

    def window_times_model_s(self, v: ObsView, t_s: float) -> FloatArray:
        """Return the absolute model times at which the band is sampled.

        Args:
            v: The observation, world frame.
            t_s: Episode time, seconds model scale.

        Returns:
            ``t0 + t_td + j / ctrl_freq`` for ``j = 0 .. n - 1``, seconds model scale on the
            feed's (record) clock, ``(n,)``.
        """
        t_td_abs = self._t0_model_s + self.predicted_touchdown_s(v, t_s)
        j = np.arange(self.rule.n_samples, dtype=np.float64)
        return np.asarray(t_td_abs + j * self.rule.spacing_s, dtype=np.float64)

    def _rule_fires(self, v: ObsView, t_s: float) -> bool:
        """Return the shared rule's verdict on the forecast band at the predicted touchdown.

        Args:
            v: The observation, world frame.
            t_s: Episode time, seconds model scale.

        Returns:
            :meth:`QuiescenceRule.verdict` on ``max(|lo|, |hi|)`` of roll, pitch and pad
            ``v_z`` at :meth:`window_times_model_s`; False when the window's last sample
            lies beyond the forecast's last lead.

        Raises:
            RuntimeError: If no feed was given (``reset`` was never called).
        """
        if self._feed is None:
            raise RuntimeError(f"{self.name}: act() before reset(motion_feed=...)")
        times = self.window_times_model_s(v, t_s)
        forecast = self.forecaster.forecast(self._feed)
        if float(times[-1]) > forecast.t_end_model_s:
            return False
        worst = [
            np.maximum(np.abs(band[:, BAND_LO]), np.abs(band[:, BAND_HI]))
            for band in (forecast.band_at(q, times) for q in _QUANTITIES)
        ]
        return self.rule.verdict(worst[0], worst[1], worst[2])


def make_gated_forecast(config_path: Path, spec: ControlSpec) -> GatedForecast:
    """Registry factory for ``gated_forecast`` (dmf ``residual_interval``).

    Args:
        config_path: ``configs/control/gated_forecast.yaml`` or an override.
        spec: Committed environment configs.

    Returns:
        A fresh controller; call ``reset(seed, motion_feed=feed)`` before use.
    """
    return GatedForecast(load_gated_forecast(config_path), spec)


def make_gated_forecast_tcn(config_path: Path, spec: ControlSpec) -> GatedForecast:
    """Registry factory for ``gated_forecast_tcn`` (dmf ``tcn_quantile``).

    Args:
        config_path: ``configs/control/gated_forecast_tcn.yaml`` or an override.
        spec: Committed environment configs.

    Returns:
        A fresh :class:`GatedForecast` named ``gated_forecast_tcn``; call
        ``reset(seed, motion_feed=feed)`` before use.
    """
    return GatedForecast(load_gated_forecast(config_path), spec, name="gated_forecast_tcn")
