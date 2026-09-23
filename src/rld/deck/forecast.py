"""Online dmf forecasting for the landing loop: causal feed, CPU forecaster, pad bands.

Phase 4, plan Step 3, protocol P4-D3. Three public objects and two helpers:

* :class:`ShipMotionFeed` -- the **past** of dmf's six clean channels, sampled on dmf's
  absolute 10 Hz full-scale grid and held in a 200-sample ring buffer. Semantically an ideal
  (noise-free, zero-latency) ship motion reference unit: it sees the ship, never the future.
* :class:`OnlineForecaster` -- one fitted dmf model (``artifacts/dmf/<model>/``) run on one
  window: normalise exactly as dmf does, run ONNX Runtime on the CPU with one intra-op thread
  (or the torch model, for parity), and undo the normalisation.
* :class:`DeckForecast` -- per lead, ``[lo, point, hi]`` for roll, pitch, pad ``z`` and pad
  ``v_z``, in **model** units, plus the full-scale channel bands they came from.
* :func:`pad_vertical_band` -- the interval-arithmetic pad band, exposed so it can be tested
  against :func:`rld.deck.kinematics.deck_point_state`.
* :func:`leads_full_s` -- the point pad ``z``/``v_z`` at 1/2/3 s full-scale lead, the block
  the Phase 6 observation will carry (not wired into any observation yet).

Time: two scales, one grid
--------------------------
The dmf models take a 200-sample lookback at 10 Hz **full scale** (20 s full = 4.0 s model
at ``lam = 1/25``) and emit 150 leads at 0.1 s full scale (0.02 s model) out to 15 s full
(3.0 s model). The feed's grid is dmf's corpus grid, **absolute**: sample ``k`` is at
``t_full = k / 10`` seconds full scale, the exact float the corpus ``t`` column holds
(``np.arange(n) / fs_hz`` in ``dmf.sim.generate``), so an online window at an origin on that
grid is the corpus window at the same origin. Model time maps to full time through the
source's own ``full_time_s`` (``spinup_s + t_model / sqrt(lam)`` for the JONSWAP and
sinusoid sources); the feed reads that map rather than re-deriving it.

**Causality is structural.** The feed's clock is advanced only by :meth:`ShipMotionFeed.
advance_to`; it never moves backwards; and every read of the underlying source goes through
:meth:`ShipMotionFeed.query_channels`, which raises for any time later than the clock
(compared in full-scale seconds, exactly, with no tolerance). The forecast origin is the
latest grid sample at or before the clock.

History before the episode start is legitimate -- the ship was already moving before the
drone arrived -- so :meth:`ShipMotionFeed.reset` fills the whole lookback from before
``t0``. The committed episode start window ``(4.0, 107.5)`` s model reserves exactly that
lookback (P2-D7).

Sign conventions and the pad band
---------------------------------
dmf's channels (``docs/corpus_card.md`` of dmf): roll positive to **starboard**, pitch
positive **bow-up**, heave positive up; degrees, degrees per second, metres, metres per
second. World frame (:mod:`rld.deck.kinematics`, P1-D2): ``x`` bow, ``y`` port, ``z`` up, ZYX
composition. For a **centreline** pad at signed body offset ``x`` (negative aft; ``-0.4 L``
= -49.6 m on the frigate) the pad's vertical position and velocity are, exactly,

    z   = heave + x * sin(pitch)
    v_z = heave_rate + x * cos(pitch) * pitch_rate          (pitch_rate in rad/s)

-- roll-blind under ZYX. A bow-up pitch lowers an aft pad. Given per-channel bands
``[lo, hi]``, the band on ``z`` and ``v_z`` is the **exact image of the channel box** under
these maps (each channel enters once, so interval arithmetic is exact, not merely a bound);
it is conservative only with respect to the joint predictive distribution, which a box of
marginal quantiles cannot represent. Evaluated at full scale, then Froude-scaled once:
``z * lam``, ``v_z * sqrt(lam)``; angles unchanged. The CG pad (``x = 0``) gives
``z = heave``, ``v_z = heave_rate``: the control arm for dmf's roll/pitch-heave phase
defect, which enters the aft pad through the ``x`` terms and is carried, not corrected.

The point forecast of a quantile model is its **0.5 quantile** (dmf P5-D5). For
``residual_interval`` that is the DLinear-OLS point plus the median tune-pool residual.

The deployed pad-v_z band is conformal-calibrated (plan amendment 2026-09-23)
-------------------------------------------------------------------------------
The raw box above is exact for the channel box but, at the aft pad, far wider than the pad
``v_z`` actually varies: the box adds the heave-rate band and the ``x * pitch_rate`` band as
if independent, while dmf's in-phase roll/pitch-heave defect makes them partly cancel. For
an **interval** model the deployed ``pad_vz_m_s`` band is therefore the box rescaled about
its point by one split-conformal factor per lead, ``gamma_j``, fitted on the dev tune pool
to 90 % pad-``v_z`` coverage, per pad (:func:`calibrate_band`,
:func:`rld.deck.forecast_fit.calibrate_pad_vz`). The pad is identified from the feed's
lever arm as a **fraction of the vessel length** (``-0.4`` aft, ``0`` CG), so the s175's aft
pad (``-70`` m) uses the frigate-calibrated ``gamma`` -- a transfer under the
``unseen_vessel`` shift, like the forecaster itself, not a calibration. A pad fraction with
no calibration raises. ``pad_z_m`` keeps the **raw box** (no controller reads it), and roll
and pitch keep the model's native 5/95 % bands. Point models are untouched (zero width).
"""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from dmf.config import ModelConfig
from dmf.data.windows import WindowSpec
from dmf.deploy.export_onnx import INPUT_NAME, wrap_for_export
from dmf.typedefs import FloatArray
from torch import nn

from rld.config import REPO_ROOT
from rld.deck.bridge import FORECAST_CHANNELS, DeckMotionSource, MotionChannels, load_vessel_cached
from rld.deck.config import PadConfig
from rld.deck.scaling import FroudeScale

__all__ = [
    "BAND",
    "CONFORMAL_FILE",
    "DEFAULT_LEADS_FULL_S",
    "DEFAULT_MODEL_ROOT",
    "HI",
    "INTERVAL_LEVELS",
    "LO",
    "POINT",
    "Backend",
    "DeckForecast",
    "OnlineForecaster",
    "ShipMotionFeed",
    "forecast_from_channels",
    "leads_full_s",
    "calibrate_band",
    "pad_vertical_band",
]

#: Band column order on every ``(..., 3)`` array here.
BAND: tuple[str, str, str] = ("lo", "point", "hi")
LO, POINT, HI = 0, 1, 2

#: The band a quantile model reports: its 5 % and 95 % fan members (dmf's 90 % interval,
#: read off the fan, never interpolated).
INTERVAL_LEVELS: tuple[float, float] = (0.05, 0.95)

#: The observation block's leads, seconds **full** scale (0.2 / 0.4 / 0.6 s model).
DEFAULT_LEADS_FULL_S: tuple[float, ...] = (1.0, 2.0, 3.0)

#: Where :mod:`rld.deck.forecast_fit` writes the fitted models.
DEFAULT_MODEL_ROOT: Path = REPO_ROOT / "artifacts" / "dmf"

#: File holding the per-pad, per-lead conformal factors of an interval model.
CONFORMAL_FILE: str = "conformal_padvz.npz"

#: Tolerance when matching a feed's pad (fraction of vessel length) to a calibrated pad.
_PAD_FRAC_TOL: float = 1e-9

#: ``|pitch|`` above which the monotonicity of ``sin`` and the positivity of ``cos`` used by
#: the band no longer hold, degrees. Unreachable by a ship (dmf's worst |pitch| is ~10 deg).
_PITCH_LIMIT_DEG: float = 89.0

Backend = Literal["ort", "torch"]


def _six(ch: MotionChannels) -> FloatArray:
    """Stack dmf's six forecast channels, in ``FORECAST_CHANNELS`` order, shape ``(n, 6)``."""
    return np.stack(
        [
            ch.roll_deg,
            ch.pitch_deg,
            ch.heave_m,
            ch.roll_rate_dps,
            ch.pitch_rate_dps,
            ch.heave_rate_m_s,
        ],
        axis=-1,
    ).astype(np.float64)


def _centreline_x(r_pad_full_m: Sequence[float]) -> float:
    """Return a centreline pad's signed x lever arm, metres full scale, or raise."""
    x, y, z = (float(v) for v in r_pad_full_m)
    if y != 0.0 or z != 0.0:
        raise ValueError(
            f"the forecast pad band is derived for a centreline pad (y = z = 0); got "
            f"r_pad_full_m = {tuple(r_pad_full_m)}"
        )
    return x


class ShipMotionFeed:
    """Causal, clock-gated history of dmf's six clean channels, full scale, 10 Hz.

    Owned by the episode runner, which is the only caller of :meth:`reset` and
    :meth:`advance_to`. A controller holding the feed can read :meth:`window` and the
    origin, and can never see a sample later than the runner's clock.

    Attributes:
        source: The deck-motion source (JONSWAP or sinusoid) the ship is sampled from.
        r_pad_full_m: The pad's body-frame lever arm from the CG, metres **full** scale,
            centreline (``y = z = 0``). Carried here because it is a property of the
            episode, so one :class:`OnlineForecaster` can serve any episode.
        scale: The Froude scale the pad quantities are converted with.
        fs_full_hz: Sampling rate of the grid, hertz **full** scale (10).
        lookback_samples: Ring-buffer length, samples (200 = 20 s full).
    """

    def __init__(
        self,
        source: DeckMotionSource,
        r_pad_full_m: Sequence[float],
        scale: FroudeScale,
        *,
        fs_full_hz: float = 10.0,
        lookback_samples: int = 200,
    ) -> None:
        """Wrap a source; nothing is sampled until :meth:`reset`.

        Args:
            source: Any :class:`~rld.deck.bridge.DeckMotionSource` whose ``full_time_s`` is
                affine in model time.
            r_pad_full_m: Pad lever arm ``(x, 0, 0)``, metres full scale, x negative aft.
            scale: Froude scale. Must equal ``source.scale`` when the source has one.
            fs_full_hz: Grid rate, hertz full scale.
            lookback_samples: Buffer length, samples.

        Raises:
            ValueError: If the pad is off the centreline, the rates are not positive, the
                source's time map is not affine, or its Froude scale differs from ``scale``.
        """
        if fs_full_hz <= 0.0 or lookback_samples < 1:
            raise ValueError(f"bad grid: {fs_full_hz} Hz, {lookback_samples} samples")
        self.source = source
        self.r_pad_full_m: tuple[float, float, float] = (
            _centreline_x(r_pad_full_m),
            0.0,
            0.0,
        )
        self.scale = scale
        self.fs_full_hz = float(fs_full_hz)
        self.lookback_samples = int(lookback_samples)
        #: The pad's x lever arm as a fraction of the vessel length (-0.4 aft, 0 CG), or
        #: None when the source's vessel has no dmf config (the static test deck).
        self.pad_frac: float | None = None
        try:
            length_full_m = float(load_vessel_cached(str(source.key[3])).length_m)
        except FileNotFoundError:
            length_full_m = 0.0
        if length_full_m > 0.0:
            self.pad_frac = self.r_pad_full_m[0] / length_full_m
        source_scale = getattr(source, "scale", None)
        if isinstance(source_scale, FroudeScale) and source_scale.lam != scale.lam:
            raise ValueError(f"source lam {source_scale.lam} != feed lam {scale.lam}")
        probe = np.asarray(source.full_time_s(np.array([0.0, 1.0, 2.0])), dtype=np.float64)
        self._full_at_model_zero = float(probe[0])
        self._full_per_model = float(probe[1] - probe[0])
        if self._full_per_model <= 0.0 or not math.isclose(
            float(probe[2] - probe[1]), self._full_per_model, rel_tol=1e-12
        ):
            raise ValueError("source.full_time_s must be affine and increasing in model time")
        self._buffer: FloatArray = np.full((self.lookback_samples, 6), np.nan)
        self._head = 0
        self._k_last: int | None = None
        self._clock_model_s: float | None = None
        self._clock_full_s: float | None = None

    @classmethod
    def for_pad(
        cls,
        source: DeckMotionSource,
        pad: str,
        pads: PadConfig,
        scale: FroudeScale,
        **kwargs: Any,
    ) -> "ShipMotionFeed":
        """Build a feed for a named pad of the source's vessel.

        Args:
            source: The deck-motion source; its key's vessel sets the lever arm.
            pad: Pad name from ``configs/deck/pad.yaml`` (``"aft"`` or ``"cg"``).
            pads: Pad geometry, fractions of the vessel's full-scale length.
            scale: Froude scale.
            **kwargs: Forwarded to the constructor.

        Returns:
            The feed, with ``r_pad_full_m`` = the pad's offset in metres full scale
            (-49.6 m for the frigate's aft pad, 0 for ``cg``).
        """
        length_full_m = float(load_vessel_cached(str(source.key[3])).length_m)
        return cls(source, pads.spec(pad).r_pad_full_m(length_full_m), scale, **kwargs)

    # --- clock ------------------------------------------------------------------------

    def _last_k(self, t_full_s: float) -> int:
        """Return the largest grid index ``k`` with ``k / fs <= t_full_s`` (exact)."""
        k = math.floor(t_full_s * self.fs_full_hz)
        while (k + 1) / self.fs_full_hz <= t_full_s:
            k += 1
        while k / self.fs_full_hz > t_full_s:
            k -= 1
        return k

    def _grid_times(self, k_first: int, k_last: int) -> FloatArray:
        """Return ``k / fs`` for ``k`` in ``[k_first, k_last]``, seconds full scale."""
        return np.arange(k_first, k_last + 1, dtype=np.float64) / self.fs_full_hz

    def _to_full(self, t_model_s: float) -> float:
        """Map model seconds to the source's absolute full-scale seconds."""
        return float(np.asarray(self.source.full_time_s(np.array([float(t_model_s)])))[0])

    def reset(self, t0_model_s: float) -> None:
        """Start an episode: set the clock to ``t0`` and fill the lookback from before it.

        Args:
            t0_model_s: Episode start, seconds **model** scale from the start of the
                committed record.

        Raises:
            ValueError: If the lookback before ``t0`` leaves the committed record (the
                source refuses the times).
        """
        self._clock_model_s = float(t0_model_s)
        self._clock_full_s = self._to_full(t0_model_s)
        k_now = self._last_k(self._clock_full_s)
        self._buffer = self._samples(k_now - self.lookback_samples + 1, k_now)
        self._head = 0
        self._k_last = k_now

    def _samples(self, k_first: int, k_last: int) -> FloatArray:
        """Evaluate grid samples ``k_first..k_last``, **one source call per sample**.

        dmf's superposition is a BLAS matrix product whose summation order depends on how
        many times are evaluated together, so the same instant evaluated inside a block of
        200 and on its own differs by ~1e-16. One call per sample makes every sample's bits
        independent of its neighbours, so the window is a pure function of the clock -- the
        same whether the runner advanced every control step or reset straight to it.

        Returns:
            Shape ``(k_last - k_first + 1, 6)``, full scale, oldest first.
        """
        times = self._grid_times(k_first, k_last)
        return np.concatenate([self.query_channels(times[i : i + 1]) for i in range(times.size)])

    def advance_to(self, t_model_s: float) -> None:
        """Move the clock forward and append every grid sample that has become past.

        Args:
            t_model_s: New clock, seconds **model** scale. Equal to the current clock is a
                no-op.

        Raises:
            RuntimeError: If called before :meth:`reset`.
            ValueError: If ``t_model_s`` is earlier than the current clock.
        """
        if self._clock_model_s is None or self._k_last is None:
            raise RuntimeError("ShipMotionFeed.advance_to called before reset")
        if t_model_s < self._clock_model_s:
            raise ValueError(
                f"clock cannot run backwards: {t_model_s} s < {self._clock_model_s} s (model)"
            )
        self._clock_model_s = float(t_model_s)
        self._clock_full_s = self._to_full(t_model_s)
        k_now = self._last_k(self._clock_full_s)
        if k_now <= self._k_last:
            return
        n_new = k_now - self._k_last
        if n_new >= self.lookback_samples:
            self._buffer = self._samples(k_now - self.lookback_samples + 1, k_now)
            self._head = 0
        else:
            for row in self._samples(self._k_last + 1, k_now):
                self._buffer[self._head] = row
                self._head = (self._head + 1) % self.lookback_samples
        self._k_last = k_now

    def query_channels(self, t_full_s: FloatArray) -> FloatArray:
        """Read the six channels at past full-scale times -- the feed's only source read.

        Args:
            t_full_s: Absolute times, seconds **full** scale, each at or before the clock.

        Returns:
            Shape ``(n, 6)``, ``FORECAST_CHANNELS`` order: degrees, degrees, metres,
            degrees per second, degrees per second, metres per second, full scale.

        Raises:
            RuntimeError: Before :meth:`reset`.
            ValueError: If any requested time is later than the clock (the causality guard),
                or outside the source's committed record.
        """
        if self._clock_full_s is None:
            raise RuntimeError("ShipMotionFeed read before reset")
        t = np.atleast_1d(np.asarray(t_full_s, dtype=np.float64))
        if t.size and float(t.max()) > self._clock_full_s:
            raise ValueError(
                f"causality: requested t_full = {float(t.max())!r} s is ahead of the clock "
                f"{self._clock_full_s!r} s (full scale)"
            )
        return _six(self.source.channels(t))

    # --- read side ----------------------------------------------------------------------

    def window(self) -> FloatArray:
        """Return the current lookback, oldest first, as a copy.

        Returns:
            Shape ``(lookback_samples, 6)``, full scale, ``FORECAST_CHANNELS`` order; row
            ``-1`` is the forecast origin.

        Raises:
            RuntimeError: Before :meth:`reset`.
        """
        if self._k_last is None:
            raise RuntimeError("ShipMotionFeed read before reset")
        return np.concatenate((self._buffer[self._head :], self._buffer[: self._head]), axis=0)

    @property
    def clock_model_s(self) -> float:
        """The runner's clock, seconds **model** scale."""
        if self._clock_model_s is None:
            raise RuntimeError("ShipMotionFeed read before reset")
        return self._clock_model_s

    @property
    def clock_full_s(self) -> float:
        """The runner's clock, absolute seconds **full** scale."""
        if self._clock_full_s is None:
            raise RuntimeError("ShipMotionFeed read before reset")
        return self._clock_full_s

    @property
    def origin_index(self) -> int:
        """Absolute grid index ``k`` of the latest sample at or before the clock."""
        if self._k_last is None:
            raise RuntimeError("ShipMotionFeed read before reset")
        return self._k_last

    @property
    def t_origin_full_s(self) -> float:
        """Forecast origin ``k / fs``, absolute seconds **full** scale."""
        return self.origin_index / self.fs_full_hz

    @property
    def t_origin_model_s(self) -> float:
        """Forecast origin, seconds **model** scale (``<=`` the clock)."""
        return (self.t_origin_full_s - self._full_at_model_zero) / self._full_per_model

    @property
    def full_s_per_model_s(self) -> float:
        """Full-scale seconds per model second (``1 / sqrt(lam)`` = 5 at ``lam = 1/25``)."""
        return self._full_per_model

    @property
    def window_times_full_s(self) -> FloatArray:
        """Absolute full-scale times of :meth:`window`'s rows, seconds, oldest first."""
        k = self.origin_index
        return self._grid_times(k - self.lookback_samples + 1, k)


@dataclass(frozen=True)
class DeckForecast:
    """One forecast from one origin, model units for the pad, with its full-scale source.

    Every ``(..., 3)`` array is ``[lo, point, hi]`` (:data:`BAND`); ``lo = hi = point`` for a
    point model. Row ``j`` is lead ``j + 1`` samples of the 10 Hz full-scale grid.

    Attributes:
        model: Artifact name of the model, e.g. ``"residual_interval"``.
        interval_levels: ``(0.05, 0.95)`` for a quantile model, None for a point model.
        t_origin_model_s: The latest grid sample at or before the clock, seconds **model**.
        t_origin_full_s: The same instant, absolute seconds **full** scale.
        lead_model_s: Leads, seconds **model** scale, shape ``(H,)``: 0.02 ... 3.0 s.
        lead_full_s: Leads, seconds **full** scale, shape ``(H,)``: 0.1 ... 15.0 s.
        roll_deg: Roll band, degrees, shape ``(H, 3)`` (Froude-invariant).
        pitch_deg: Pitch band, degrees, shape ``(H, 3)``.
        pad_z_m: Pad vertical position band, metres **model** scale, relative to the ship's
            mean-position CG height, shape ``(H, 3)``.
        pad_vz_m_s: Pad vertical velocity band, metres per second **model** scale,
            ``(H, 3)``. For an interval model this is the **conformal-calibrated** box
            (module docstring); for a point model it has zero width.
        channels_full: The six channel bands, **full** scale, ``FORECAST_CHANNELS`` order,
            shape ``(H, 6, 3)``.
        origin_channels_full: The observed channels at the origin, full scale, ``(6,)``.
        r_pad_full_m: Pad lever arm the pad bands were computed for, metres full scale.
        lam: The Froude length scale the pad quantities were converted with.
        pad_vz_box_m_s: The raw interval-arithmetic box on pad ``v_z``, m/s model,
            ``(H, 3)``, before calibration (None when no calibration was applied).
        pad_vz_gamma: The per-lead conformal factors applied, ``(H,)``, or None.
    """

    model: str
    interval_levels: tuple[float, float] | None
    t_origin_model_s: float
    t_origin_full_s: float
    lead_model_s: FloatArray
    lead_full_s: FloatArray
    roll_deg: FloatArray
    pitch_deg: FloatArray
    pad_z_m: FloatArray
    pad_vz_m_s: FloatArray
    channels_full: FloatArray
    origin_channels_full: FloatArray
    r_pad_full_m: tuple[float, float, float]
    lam: float
    pad_vz_box_m_s: FloatArray | None = None
    pad_vz_gamma: FloatArray | None = None

    @property
    def t_model_s(self) -> FloatArray:
        """Absolute model time of each lead, seconds **model** scale, ``(H,)``."""
        return self.t_origin_model_s + self.lead_model_s

    @property
    def t_end_model_s(self) -> float:
        """Last forecast instant, seconds **model** scale (origin + 3.0 s at 1/25)."""
        return float(self.t_origin_model_s + self.lead_model_s[-1])

    def band_at(
        self,
        quantity: Literal["roll_deg", "pitch_deg", "pad_z_m", "pad_vz_m_s"],
        t_model_s: FloatArray,
    ) -> FloatArray:
        """Linearly interpolate one quantity's band at arbitrary model times.

        The grid is the origin (the observed sample, zero-width band) followed by the H
        forecast leads, i.e. 50 Hz model at ``lam = 1/25``; ``lo``, ``point`` and ``hi`` are
        interpolated independently. This interpolates the **forecast**, never the deck
        (P4-D3).

        Args:
            quantity: Which band.
            t_model_s: Absolute times, seconds **model** scale, inside
                ``[t_origin_model_s, t_end_model_s]``.

        Returns:
            Shape ``(n, 3)``, in the quantity's units (degrees, metres model, metres per
            second model).

        Raises:
            ValueError: If any time lies outside the forecast span.
        """
        t = np.atleast_1d(np.asarray(t_model_s, dtype=np.float64))
        grid = np.concatenate(([self.t_origin_model_s], self.t_model_s))
        if t.size and (float(t.min()) < grid[0] or float(t.max()) > grid[-1]):
            raise ValueError(
                f"times [{float(t.min())}, {float(t.max())}] s (model) leave the forecast "
                f"span [{grid[0]}, {grid[-1]}]"
            )
        band = getattr(self, quantity)
        origin = self._origin_value(quantity)
        values = np.concatenate((np.full((1, 3), origin), band), axis=0)
        return np.stack([np.interp(t, grid, values[:, i]) for i in range(3)], axis=-1)

    def _origin_value(self, quantity: str) -> float:
        """Return the observed value of ``quantity`` at the origin, model units."""
        ch = self.origin_channels_full
        if quantity == "roll_deg":
            return float(ch[0])
        if quantity == "pitch_deg":
            return float(ch[1])
        band = np.broadcast_to(ch[None, :, None], (1, 6, 3))
        z_full, vz_full = pad_vertical_band(band, self.r_pad_full_m[0])
        scale = FroudeScale(lam=self.lam)
        if quantity == "pad_z_m":
            return float(scale.length(z_full[0, POINT]))
        return float(scale.velocity(vz_full[0, POINT]))


def pad_vertical_band(
    channels_band_full: FloatArray, x_pad_full_m: float
) -> tuple[FloatArray, FloatArray]:
    """Map channel bands to the exact band of a centreline pad's ``z`` and ``v_z``.

    ``z = heave + x sin(pitch)`` and ``v_z = heave_rate + x cos(pitch) pitch_rate`` (module
    docstring). Each channel enters once, so the image of the channel box is the interval
    arithmetic below, exactly: ``sin`` is monotone and ``cos`` positive for ``|pitch| < 90``
    deg, ``cos`` peaks at 1 when the pitch band straddles zero, and the product
    ``(x * pitch_rate) * cos(pitch)`` takes its extremes at the corners.

    Args:
        channels_band_full: Shape ``(..., 6, 3)``, ``FORECAST_CHANNELS`` order, ``[lo, point,
            hi]`` with ``lo <= point <= hi``; degrees, metres, degrees per second, metres per
            second, **full** scale.
        x_pad_full_m: Signed centreline lever arm, metres **full** scale (negative aft).

    Returns:
        ``(z, v_z)``, each ``(..., 3)`` ``[lo, point, hi]``, metres and metres per second,
        **full** scale.

    Raises:
        ValueError: If any pitch bound reaches ``|pitch| >= 89`` deg.
    """
    band = np.asarray(channels_band_full, dtype=np.float64)
    heave, pitch_deg = band[..., 2, :], band[..., 1, :]
    q_deg, heave_rate = band[..., 4, :], band[..., 5, :]
    if np.any(np.abs(pitch_deg) >= _PITCH_LIMIT_DEG):
        raise ValueError("pitch band reaches +-89 deg; the pad band's monotonicity fails")
    x = float(x_pad_full_m)
    p = np.radians(pitch_deg)
    q = np.radians(q_deg)
    lo, pt, hi = LO, POINT, HI

    s_lo, s_hi = x * np.sin(p[..., lo]), x * np.sin(p[..., hi])
    z = np.stack(
        [
            heave[..., lo] + np.minimum(s_lo, s_hi),
            heave[..., pt] + x * np.sin(p[..., pt]),
            heave[..., hi] + np.maximum(s_lo, s_hi),
        ],
        axis=-1,
    )

    c_a, c_b = np.cos(p[..., lo]), np.cos(p[..., hi])
    straddles = (p[..., lo] <= 0.0) & (p[..., hi] >= 0.0)
    c_min = np.minimum(c_a, c_b)
    c_max = np.where(straddles, 1.0, np.maximum(c_a, c_b))
    a_1, a_2 = x * q[..., lo], x * q[..., hi]
    a_min, a_max = np.minimum(a_1, a_2), np.maximum(a_1, a_2)
    corners = np.stack([a_min * c_min, a_min * c_max, a_max * c_min, a_max * c_max], axis=-1)
    vz = np.stack(
        [
            heave_rate[..., lo] + corners.min(axis=-1),
            heave_rate[..., pt] + (x * q[..., pt]) * np.cos(p[..., pt]),
            heave_rate[..., hi] + corners.max(axis=-1),
        ],
        axis=-1,
    )
    return z, vz


def calibrate_band(box: FloatArray, gamma: FloatArray) -> FloatArray:
    """Rescale a ``[lo, point, hi]`` band's two half-widths about its point.

    ``lo' = point - gamma * (point - lo)`` and ``hi' = point + gamma * (hi - point)``,
    written as ``lo + (1 - gamma) * (point - lo)`` so that ``gamma = 1`` returns the box
    **bit for bit**. The point is untouched; ``gamma > 0`` keeps ``lo' <= point <= hi'``.

    Args:
        box: ``(..., H, 3)`` band, any units.
        gamma: ``(H,)`` dimensionless factors, positive.

    Returns:
        The calibrated band, same shape and units.

    Raises:
        ValueError: If ``gamma`` is not positive and finite or does not match ``H``.
    """
    band = np.asarray(box, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    if g.shape != band.shape[-2:-1] or not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError(f"gamma {g.shape} must be positive, finite, one per lead")
    shrink = 1.0 - g  # broadcasts over the leading axes of the (..., H) slices
    lo = band[..., LO] + shrink * (band[..., POINT] - band[..., LO])
    hi = band[..., HI] - shrink * (band[..., HI] - band[..., POINT])
    return np.stack([lo, band[..., POINT], hi], axis=-1)


def forecast_from_channels(
    channels_band_full: FloatArray,
    *,
    origin_channels_full: FloatArray,
    t_origin_full_s: float,
    t_origin_model_s: float,
    full_s_per_model_s: float,
    fs_full_hz: float,
    r_pad_full_m: Sequence[float],
    scale: FroudeScale,
    model: str,
    interval_levels: tuple[float, float] | None,
) -> DeckForecast:
    """Assemble a :class:`DeckForecast` from full-scale channel bands.

    Used by :meth:`OnlineForecaster.forecast`; public so that a test can feed it the true
    future and check the pad quantities against the bridge.

    Args:
        channels_band_full: ``(H, 6, 3)`` channel bands, full scale.
        origin_channels_full: ``(6,)`` observed channels at the origin, full scale.
        t_origin_full_s: Origin, absolute seconds full scale.
        t_origin_model_s: Origin, seconds model scale.
        full_s_per_model_s: The source's time factor (5 at ``lam = 1/25``).
        fs_full_hz: Lead spacing, hertz full scale.
        r_pad_full_m: Centreline pad lever arm, metres full scale.
        scale: Froude scale for the pad quantities.
        model: Model artifact name.
        interval_levels: The band's quantile levels, or None for a point model.

    Returns:
        The forecast; pad quantities in **model** units.
    """
    band = np.asarray(channels_band_full, dtype=np.float64)
    arm = (_centreline_x(r_pad_full_m), 0.0, 0.0)
    z_full, vz_full = pad_vertical_band(band, arm[0])
    n_leads = band.shape[0]
    lead_full = np.arange(1, n_leads + 1, dtype=np.float64) / fs_full_hz
    return DeckForecast(
        model=model,
        interval_levels=interval_levels,
        t_origin_model_s=float(t_origin_model_s),
        t_origin_full_s=float(t_origin_full_s),
        lead_model_s=lead_full / full_s_per_model_s,
        lead_full_s=lead_full,
        roll_deg=scale.angle(band[:, 0, :]),
        pitch_deg=scale.angle(band[:, 1, :]),
        pad_z_m=scale.length(z_full),
        pad_vz_m_s=scale.velocity(vz_full),
        channels_full=band,
        origin_channels_full=np.asarray(origin_channels_full, dtype=np.float64).copy(),
        r_pad_full_m=arm,
        lam=scale.lam,
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OnlineForecaster:
    """One fitted dmf model, run on one window at a time on the CPU.

    Attributes:
        model_dir: The artifact directory (``artifacts/dmf/<model>/``).
        backend: ``"ort"`` (deployment: ONNX Runtime, CPU EP) or ``"torch"`` (parity).
        meta: The directory's ``meta.json``.
        scale_full: Per-channel normalisation scale, full-scale corpus units, ``(6,)``.
    """

    def __init__(
        self,
        model_dir: Path,
        backend: Backend = "ort",
        *,
        intra_op_threads: int = 1,
        verify_files: bool = True,
    ) -> None:
        """Load a model directory written by :mod:`rld.deck.forecast_fit`.

        Args:
            model_dir: The directory.
            backend: ``"ort"`` or ``"torch"``.
            intra_op_threads: ORT intra-op threads; 1 for the landing loop (the cost the
                plan records), more only for offline batch scoring.
            verify_files: Check every file's SHA-256 against ``meta.json`` first.

        Raises:
            FileNotFoundError: If a file is missing.
            ValueError: On a hash mismatch, an unknown backend, or channels that are not
                dmf's six in order.
        """
        from rld.deck.forecast_fit import load_norm_scale  # local: keeps the import light

        self.model_dir = Path(model_dir)
        self.backend = backend
        self.meta: dict[str, Any] = json.loads(
            (self.model_dir / "meta.json").read_text(encoding="utf-8")
        )
        if verify_files:
            for name, digest in self.meta["files"].items():
                if _sha256(self.model_dir / name) != digest:
                    raise ValueError(f"{self.model_dir / name} does not match meta.json")
        channels, scale = load_norm_scale(self.model_dir / "norm_stats.npz")
        if channels != FORECAST_CHANNELS or tuple(self.meta["input_channels"]) != channels:
            raise ValueError(f"model channels {channels} are not {FORECAST_CHANNELS}")
        self.scale_full: FloatArray = scale
        self.head: str = str(self.meta["head"])
        self.quantiles: tuple[float, ...] = tuple(float(q) for q in self.meta["quantiles"])
        self.lookback = int(self.meta["lookback"])
        self.max_horizon = int(self.meta["max_horizon"])
        self.fs_full_hz = float(self.meta["fs_hz"])
        self._level_index: tuple[int, int, int] | None = None
        if self.head == "quantile":
            self._level_index = (
                self._index(INTERVAL_LEVELS[0]),
                self._index(0.5),
                self._index(INTERVAL_LEVELS[1]),
            )
        elif self.head != "point":
            raise ValueError(f"unsupported head {self.head!r}")
        self._gammas: dict[str, tuple[float, FloatArray]] = {}
        if self.head == "quantile" and (self.model_dir / CONFORMAL_FILE).is_file():
            with np.load(self.model_dir / CONFORMAL_FILE, allow_pickle=False) as data:
                for pad in (str(p) for p in data["pads"].tolist()):
                    self._gammas[pad] = (
                        float(data[f"x_frac_{pad}"]),
                        np.asarray(data[f"gamma_{pad}"], dtype=np.float64),
                    )
        self._session: Any = None
        self._torch: nn.Module | None = None
        if backend == "ort":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = int(intra_op_threads)
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                str(self.model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
            )
        elif backend == "torch":
            self._torch = self._build_torch()
        else:
            raise ValueError(f"unknown backend {backend!r}")

    @property
    def name(self) -> str:
        """Artifact name, e.g. ``"tcn_quantile"``."""
        return str(self.meta["model"])

    @property
    def is_smoke(self) -> bool:
        """True for a toy-size smoke fit, whose numbers must never be committed."""
        return bool(self.meta["smoke"])

    @property
    def interval_levels(self) -> tuple[float, float] | None:
        """``(0.05, 0.95)`` for a quantile model, None for a point model."""
        return INTERVAL_LEVELS if self.head == "quantile" else None

    def _index(self, level: float) -> int:
        """Return the fan index of a quantile level (exact match, never interpolated)."""
        for i, q in enumerate(self.quantiles):
            if abs(q - level) <= 1e-9:
                return i
        raise ValueError(f"level {level} not in the fan {self.quantiles}")

    def _build_torch(self) -> nn.Module:
        """Rebuild the torch model from ``meta.json`` and ``state_dict.pt`` (strict)."""
        from dmf.train.registry import build_model

        cfg = ModelConfig(
            name=str(self.meta["dmf_name"]),
            head=self.meta["head"],
            quantiles=self.quantiles,
            params=dict(self.meta["params"]),
            label=str(self.meta["label"]),
        )
        window = WindowSpec(lookback=self.lookback, horizons=(self.max_horizon,), stride=1)
        n_channels = len(FORECAST_CHANNELS)
        model = build_model(cfg, window, n_channels, n_channels, revin=bool(self.meta["revin"]))
        state = torch.load(self.model_dir / "state_dict.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        graph = wrap_for_export(model)
        graph.eval()
        return graph

    def normalise(self, windows_full: FloatArray) -> tuple[np.ndarray, FloatArray]:
        """Apply dmf's input pipeline: float32 storage cast, per-window de-mean, train scale.

        dmf's models only ever saw the corpus as **stored**, in float32
        (``dmf.sim.generate.CORPUS_DTYPES``), promoted to float64 by the dataset. The window is
        cast the same way here, so an online window on the corpus grid is the corpus window,
        bit for bit, and the online forecast equals dmf's offline forecast (parity (b)).
        Without the cast the difference is not negligible for every model: DLinear-OLS's
        remainder map has row L1 norms up to ~6.5e3 in normalised units (its design is
        rank-deficient and ridge-regularised, dmf P3-D19), so float32-ulp input differences
        move its output by up to 1.05e-3 deg -- measured at 8.7x (``dlinear_ols``) and 9.8x
        (``residual_interval``) dmf's 1e-4 parity tolerance over 12 tune-pool windows
        (``results/forecast/parity.csv``, the ``without_float32_cast`` rows). Physically that
        is a thousandth of a degree; the cast is about reproducing dmf, not about accuracy.

        Args:
            windows_full: ``(B, L, 6)`` full-scale windows, any float dtype.

        Returns:
            ``(x, mean)``: ``x`` float32 ``(B, L, 6)`` dimensionless, ``mean`` float64
            ``(B, 1, 6)`` in corpus units -- the same arithmetic as
            ``dmf.data.dataset.DeckMotionDataset.__getitem__``.
        """
        raw = np.asarray(windows_full, dtype=np.float32).astype(np.float64)
        mean = raw.mean(axis=1, keepdims=True)
        x = ((raw - mean) / self.scale_full[None, None, :]).astype(np.float32)
        return x, mean

    def run_normalised(self, x: np.ndarray) -> np.ndarray:
        """Run the model on normalised windows.

        Args:
            x: ``(B, L, 6)`` float32, dimensionless.

        Returns:
            ``(B, H, 6)`` or, for a quantile head, ``(B, H, 6, Q)`` sorted along ``Q``;
            float32, dimensionless.
        """
        batch = np.ascontiguousarray(x, dtype=np.float32)
        if self._session is not None:
            (out,) = self._session.run(None, {INPUT_NAME: batch})
            return np.asarray(out)
        assert self._torch is not None
        with torch.no_grad():
            out_t = self._torch(torch.from_numpy(batch))
        return np.asarray(out_t.numpy(), dtype=np.float32)

    def predict_windows(self, windows_full: FloatArray) -> FloatArray:
        """Forecast a batch of windows into full-scale channel bands.

        Args:
            windows_full: ``(B, L, 6)`` full scale, ``FORECAST_CHANNELS`` order.

        Returns:
            ``(B, H, 6, 3)`` ``[lo, point, hi]``, full scale, float64. For a point model
            the three columns are identical.
        """
        x, mean = self.normalise(windows_full)
        raw = self.run_normalised(x).astype(np.float64)
        if self._level_index is not None:
            raw = raw[..., list(self._level_index)]
        else:
            raw = np.repeat(raw[..., None], 3, axis=-1)
        return raw * self.scale_full[None, None, :, None] + mean[..., None]

    @property
    def calibrated_pads(self) -> dict[str, float]:
        """``{pad name: x as a fraction of vessel length}`` with a pad-v_z calibration."""
        return {pad: frac for pad, (frac, _) in self._gammas.items()}

    def pad_vz_gamma(self, pad_frac: float | None) -> FloatArray:
        """Return the per-lead conformal factors for a pad, identified by its lever arm.

        Args:
            pad_frac: The pad's x lever arm as a fraction of vessel length (-0.4, 0.0).

        Returns:
            ``(H,)`` dimensionless factors.

        Raises:
            ValueError: If this is not an interval model, or no calibrated pad matches.
        """
        if self.head != "quantile":
            raise ValueError(f"{self.name} is a point model; it has no pad-v_z band")
        for frac, gamma in self._gammas.values():
            if pad_frac is not None and abs(frac - pad_frac) <= _PAD_FRAC_TOL:
                return gamma
        raise ValueError(
            f"{self.name} has no pad-v_z conformal calibration for a pad at x = {pad_frac} L "
            f"(calibrated: {self.calibrated_pads}); run scripts/fit_dmf_forecasters.py "
            f"--calibrate-only --models {self.name}"
        )

    def forecast(self, feed: ShipMotionFeed) -> DeckForecast:
        """Forecast from the feed's current window.

        Args:
            feed: A reset feed; its clock is not moved.

        Returns:
            The :class:`DeckForecast` for the feed's pad, pad quantities in model units.
            For an interval model the pad-``v_z`` band is conformal-calibrated for the
            feed's pad; for a point model every band has zero width.

        Raises:
            ValueError: If this is an interval model and the feed's pad is not calibrated.
        """
        gamma = self.pad_vz_gamma(feed.pad_frac) if self.head == "quantile" else None
        window = feed.window()
        band = self.predict_windows(window[None])[0]
        raw = forecast_from_channels(
            band,
            origin_channels_full=window[-1],
            t_origin_full_s=feed.t_origin_full_s,
            t_origin_model_s=feed.t_origin_model_s,
            full_s_per_model_s=feed.full_s_per_model_s,
            fs_full_hz=self.fs_full_hz,
            r_pad_full_m=feed.r_pad_full_m,
            scale=feed.scale,
            model=self.name,
            interval_levels=self.interval_levels,
        )
        if gamma is None:
            return raw
        return replace(
            raw,
            pad_vz_m_s=calibrate_band(raw.pad_vz_m_s, gamma),
            pad_vz_box_m_s=raw.pad_vz_m_s,
            pad_vz_gamma=gamma,
        )


def leads_full_s(
    forecast: DeckForecast, leads_s: Sequence[float] = DEFAULT_LEADS_FULL_S
) -> FloatArray:
    """Return the point pad ``z`` and ``v_z`` at full-scale leads that lie on the grid.

    Args:
        forecast: A forecast.
        leads_s: Leads, seconds **full** scale; each must be a multiple of the 0.1 s grid
            (1, 2, 3 s -> lead indices 10, 20, 30 -> 0.2, 0.4, 0.6 s model).

    Returns:
        Shape ``(len(leads_s), 2)``: columns pad ``z`` (metres **model**) and pad ``v_z``
        (metres per second **model**), point forecasts.

    Raises:
        ValueError: If a lead is off the grid or beyond the horizon.
    """
    fs = 1.0 / float(forecast.lead_full_s[0])
    rows = []
    for lead in leads_s:
        j = round(float(lead) * fs)
        if not 1 <= j <= forecast.lead_full_s.size or abs(j / fs - float(lead)) > 1e-9:
            raise ValueError(f"lead {lead} s (full) is not on the forecast grid")
        rows.append((forecast.pad_z_m[j - 1, POINT], forecast.pad_vz_m_s[j - 1, POINT]))
    return np.asarray(rows, dtype=np.float64)
