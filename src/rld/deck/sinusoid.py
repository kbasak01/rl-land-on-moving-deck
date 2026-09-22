"""Angelis-style sinusoidal deck motion -- the H4 motion-realism arm (plan D0.4).

The pre-registered novelty test compares policies trained on JONSWAP motion with policies
trained on the *sinusoidal* platform motion used by the closest prior art. For that
comparison to be about **realism** rather than about amplitude or speed, the sinusoid is
matched to a specific JONSWAP realization:

* **amplitude**, per DOF: ``sqrt(2) x RMS`` of the matched realization over its whole
  committed record, so the two motions share a variance per channel. Degrees for roll and
  pitch, metres for heave, at **full** scale.
* **period**: the realization's peak encounter period
  ``T_e = 2*pi / |encounter_frequency(2*pi/Tp, U, beta)|``, seconds **full** scale. One
  period drives all three DOFs -- that single-frequency structure is exactly what H4 tests.
  (SS6, 180 deg, 12 kn: ``T_e = 9.401 s`` full = ``1.880 s`` model at ``lam = 1/25``.)
* **phase**: one phase per DOF drawn ``U(0, 2*pi)`` from the episode seed, so the three DOFs
  are *not* locked in phase the way dmf's known defect locks them. The phase stream is
  derived from its own namespace and never from dmf's, so a sinusoid episode cannot alias a
  JONSWAP wave draw.

The class implements the same :class:`rld.deck.bridge.DeckMotionSource` interface as
:class:`rld.deck.bridge.JonswapDeckMotion`, **including** the full-scale lookback window, so
Phase 4's forecast-conditioned arm runs on this leg unchanged and Phase 6's
``ppo_sinusoid`` is a config change rather than a code change.

Units and scales: ``channels`` is full scale (degrees, degrees per second, degrees per second
squared, metres, metres per second, metres per second squared, absolute full-scale seconds);
``deck_point`` is model scale in the world frame of :mod:`rld.deck.kinematics`. Rates and
accelerations are analytic derivatives of the sinusoid, never finite differences.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from dmf.config import SimConfig
from dmf.data.splits import RealizationKey
from dmf.sim.encounter import encounter_frequency, knots_to_m_s
from dmf.sim.generate import RealizationSpec
from dmf.typedefs import FloatArray

from rld.deck.bridge import (
    TIME_BOUND_SLACK_S,
    DeckPointTrajectory,
    JonswapDeckMotion,
    MotionChannels,
)
from rld.deck.config import PadConfig
from rld.deck.kinematics import deck_point_state, to_model_state
from rld.deck.scaling import FroudeScale
from rld.deck.splits import key_for_spec

__all__ = [
    "SINUSOID_SEED_NAMESPACE",
    "SinusoidDeckMotion",
    "SinusoidParams",
    "matched_rms",
    "matched_rms_from_channels",
    "peak_encounter_period_s",
    "sinusoid_params",
    "sinusoid_seed_sequence",
]

#: Namespace constant mixed into every sinusoid phase draw. Deliberately *not* dmf's
#: ``CORPUS_SEED_NAMESPACE``: the sinusoid arm must not be able to alias a wave draw.
SINUSOID_SEED_NAMESPACE: int = 0x726C64_53  # "rld" + sinusoid

#: Smallest encounter frequency, radians per second, for which a peak period is meaningful.
#: The corpus grid excludes following seas, where ``w_e`` can pass through zero.
MIN_ENCOUNTER_RAD_S: float = 1e-6


@dataclass(frozen=True)
class SinusoidParams:
    """One realization's matched sinusoid, all **full** scale.

    Attributes:
        key: The matched dmf realization's key ``(ss, heading_deg, speed_kn, vessel, seed)``.
        roll_amp_deg: Roll amplitude, degrees = ``sqrt(2) x`` the realization's roll RMS.
        pitch_amp_deg: Pitch amplitude, degrees.
        heave_amp_m: Heave amplitude, metres.
        te_full_s: Period, seconds, full scale -- the peak encounter period.
        phase_rad: ``(roll, pitch, heave)`` phases, radians in ``[0, 2*pi)``, from the
            episode seed.
        episode_seed: The episode seed the phases were drawn from, dimensionless.
    """

    key: RealizationKey
    roll_amp_deg: float
    pitch_amp_deg: float
    heave_amp_m: float
    te_full_s: float
    phase_rad: tuple[float, float, float]
    episode_seed: int

    @property
    def w_e_rad_s(self) -> float:
        """Angular frequency, radians per second, full scale: ``2*pi / te_full_s``."""
        return 2.0 * np.pi / self.te_full_s


def peak_encounter_period_s(tp_s: float, speed_kn: float, heading_deg: float) -> float:
    """Return the encounter period of the spectral peak, seconds full scale.

    Uses dmf's own Doppler relation, so following-sea sign handling and the
    encounter-non-monotonic cells behave exactly as they do in the corpus: the magnitude of
    the signed encounter frequency sets the period.

    Args:
        tp_s: Spectral peak period, seconds, full scale.
        speed_kn: Forward speed, knots.
        heading_deg: Encounter angle, degrees (180 head, 90 beam).

    Returns:
        ``2*pi / |w_e(2*pi/tp_s)|``, seconds, full scale. At SS6 (``tp_s = 12.4``), 180 deg
        and 12 kn this is 9.401 s.

    Raises:
        ValueError: If ``tp_s`` is not positive, or if the encounter frequency is so close to
            zero that no period is defined (the corpus grid excludes following seas, where
            that happens).
    """
    if tp_s <= 0.0:
        raise ValueError(f"tp_s must be positive, got {tp_s}")
    w_peak = 2.0 * np.pi / tp_s
    w_e = float(encounter_frequency(np.asarray([w_peak]), knots_to_m_s(speed_kn), heading_deg)[0])
    if abs(w_e) < MIN_ENCOUNTER_RAD_S:
        raise ValueError(
            f"encounter frequency {w_e} rad/s at heading {heading_deg} deg and {speed_kn} kn "
            "is indistinguishable from zero: no peak encounter period exists"
        )
    return float(2.0 * np.pi / abs(w_e))


def matched_rms_from_channels(channels: MotionChannels) -> tuple[float, float, float]:
    """Return the per-DOF RMS of an already-evaluated JONSWAP record, full scale.

    Args:
        channels: The matched realization's channels, full scale, over the span the RMS is
            defined on (the committed 600 s record, for a committed sinusoid).

    Returns:
        ``(roll_rms_deg, pitch_rms_deg, heave_rms_m)``, i.e. ``sqrt(mean(x**2))`` per
        channel, full scale.
    """
    return (
        float(np.sqrt(np.mean(np.square(channels.roll_deg)))),
        float(np.sqrt(np.mean(np.square(channels.pitch_deg)))),
        float(np.sqrt(np.mean(np.square(channels.heave_m)))),
    )


def matched_rms(
    spec: RealizationSpec, sim_cfg: SimConfig, scale: FroudeScale, pads: PadConfig
) -> tuple[float, float, float]:
    """Evaluate the matched JONSWAP realization and return its per-DOF RMS, full scale.

    The record is sampled at the corpus rate (``sim_cfg.fs_hz``, 10 Hz full scale) over the
    whole committed 600 s, which is the definition the committed sinusoid parameters in
    ``results/deck_stats_seeds.csv`` are computed from. The motion is band-limited to
    2.5 rad/s, so that rate is far above Nyquist and the RMS is rate-insensitive.

    Args:
        spec: The realization to match.
        sim_cfg: dmf's corpus config, full scale.
        scale: Froude scale (unused by the RMS itself; carried so the JONSWAP source is
            constructed exactly as elsewhere).
        pads: Pad geometry (likewise).

    Returns:
        ``(roll_rms_deg, pitch_rms_deg, heave_rms_m)``, full scale.
    """
    source = JonswapDeckMotion(spec, sim_cfg, scale, pads)
    n = int(round(sim_cfg.duration_s * sim_cfg.fs_hz))
    t_full_s = sim_cfg.spinup_s + np.arange(n, dtype=np.float64) / sim_cfg.fs_hz
    return matched_rms_from_channels(source.channels(t_full_s))


def sinusoid_seed_sequence(spec: RealizationSpec, episode_seed: int) -> np.random.SeedSequence:
    """Derive the phase stream for one (realization, episode) pair.

    Mirrors :func:`dmf.sim.generate.realization_seed_sequence`: ``blake2b`` of a canonical
    coordinate string mixed with a namespace constant, so a sinusoid episode is reproducible
    in isolation, independent of worker count and of the order episodes are generated in. The
    namespace is :data:`SINUSOID_SEED_NAMESPACE`, not dmf's, so this stream cannot collide
    with a wave draw.

    Args:
        spec: The matched realization.
        episode_seed: Episode seed, a dimensionless non-negative ordinal.

    Returns:
        A ``numpy.random.SeedSequence`` for the three DOF phases.

    Raises:
        ValueError: If ``episode_seed`` is negative.
    """
    if episode_seed < 0:
        raise ValueError(f"episode_seed must be non-negative, got {episode_seed}")
    key = (
        f"sinusoid|{spec.vessel}|{spec.sea_state}|{spec.heading_deg:.6f}|"
        f"{spec.speed_kn:.6f}|{spec.seed:d}|{episode_seed:d}"
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    words = [int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)]
    return np.random.SeedSequence([SINUSOID_SEED_NAMESPACE, *words])


def sinusoid_params(
    spec: RealizationSpec,
    sim_cfg: SimConfig,
    scale: FroudeScale,
    pads: PadConfig,
    episode_seed: int = 0,
    rms_full: tuple[float, float, float] | None = None,
) -> SinusoidParams:
    """Build the matched sinusoid parameters for one realization.

    Args:
        spec: The realization to match.
        sim_cfg: dmf's corpus config, full scale.
        scale: Froude scale.
        pads: Pad geometry.
        episode_seed: Episode seed the three phases are drawn from.
        rms_full: Optional pre-computed ``(roll_deg, pitch_deg, heave_m)`` RMS of the matched
            realization, full scale, to avoid re-evaluating a record the caller already has.
            Must be the RMS over the whole committed record if the result is to be committed.

    Returns:
        The :class:`SinusoidParams`, full scale.

    Raises:
        ValueError: If the sea state is unknown to ``sim_cfg`` or the cell has no peak
            encounter period.
    """
    sea = next((s for s in sim_cfg.sea_states if s.name == spec.sea_state), None)
    if sea is None:
        known = [s.name for s in sim_cfg.sea_states]
        raise ValueError(f"unknown sea state {spec.sea_state!r}, expected one of {known}")
    roll_rms, pitch_rms, heave_rms = (
        rms_full if rms_full is not None else matched_rms(spec, sim_cfg, scale, pads)
    )
    phases = np.random.default_rng(sinusoid_seed_sequence(spec, episode_seed)).uniform(
        0.0, 2.0 * np.pi, size=3
    )
    root_two = float(np.sqrt(2.0))
    return SinusoidParams(
        key=key_for_spec(spec),
        roll_amp_deg=root_two * roll_rms,
        pitch_amp_deg=root_two * pitch_rms,
        heave_amp_m=root_two * heave_rms,
        te_full_s=peak_encounter_period_s(sea.tp_s, spec.speed_kn, spec.heading_deg),
        phase_rad=(float(phases[0]), float(phases[1]), float(phases[2])),
        episode_seed=int(episode_seed),
    )


class SinusoidDeckMotion:
    """Sinusoidal deck motion matched to one dmf realization.

    Attributes:
        params: The matched sinusoid's amplitudes, period and phases, full scale.
        sim_cfg: dmf's corpus config -- the source of the committed time window, so that a
            sinusoid episode and its JONSWAP twin occupy the same span.
        scale: Froude scale, full to model.
        pads: Pad geometry; offsets are fractions of the vessel's full-scale length.
        length_full_m: The matched vessel's length between perpendiculars, metres full scale,
            which sets the pad lever arm.
        lookback_full_s: Reserved history before the earliest usable model time, seconds full
            scale. Present for the same reason as on the JONSWAP source: Phase 4's forecast
            arm must be able to look back 200 samples at 10 Hz full scale on this leg too.
    """

    def __init__(
        self,
        params: SinusoidParams,
        sim_cfg: SimConfig,
        scale: FroudeScale,
        pads: PadConfig,
        length_full_m: float,
        lookback_full_s: float = 0.0,
    ) -> None:
        """Store the matched parameters and the window they are valid on.

        Args:
            params: Matched sinusoid parameters, full scale.
            sim_cfg: dmf's corpus config.
            scale: Froude scale.
            pads: Pad geometry.
            length_full_m: Vessel length between perpendiculars, metres full scale.
            lookback_full_s: Reserved history, seconds full scale, non-negative.

        Raises:
            ValueError: If ``lookback_full_s`` is negative or leaves no usable window, or if
                ``length_full_m`` is not positive.
        """
        if lookback_full_s < 0.0:
            raise ValueError(f"lookback_full_s must be non-negative, got {lookback_full_s}")
        if lookback_full_s >= sim_cfg.duration_s:
            raise ValueError(
                f"lookback_full_s {lookback_full_s} leaves no usable window inside a "
                f"{sim_cfg.duration_s} s record"
            )
        if length_full_m <= 0.0:
            raise ValueError(f"length_full_m must be positive, got {length_full_m}")
        self.params = params
        self.sim_cfg = sim_cfg
        self.scale = scale
        self.pads = pads
        self.length_full_m = float(length_full_m)
        self.lookback_full_s = float(lookback_full_s)

    @property
    def key(self) -> RealizationKey:
        """Return the matched realization's key ``(ss, heading, speed, vessel, seed)``.

        The sinusoid inherits the key of the realization it matches, which is what keeps the
        H4 cross-evaluation on the *same* realization-level splits as the JONSWAP arm.
        """
        return self.params.key

    @property
    def t_model_window_s(self) -> tuple[float, float]:
        """Usable model-scale span, seconds, identical to the JONSWAP source's window.

        The sinusoid is analytic everywhere, but it is held to the same window so that an H4
        episode list is valid on both legs without re-derivation.
        """
        return (
            float(self.scale.to_model_time(self.lookback_full_s)),
            float(self.scale.to_model_time(self.sim_cfg.duration_s)),
        )

    def episode_start_window_s(self, episode_len_model_s: float) -> tuple[float, float]:
        """Return the model-scale window of legal episode start offsets, seconds.

        Args:
            episode_len_model_s: Episode length, seconds, model scale.

        Returns:
            ``(lo, hi)`` model-scale seconds, as on the JONSWAP source.

        Raises:
            ValueError: If the episode does not fit in the record.
        """
        lo, hi = self.t_model_window_s
        if not 0.0 < episode_len_model_s <= hi - lo:
            raise ValueError(
                f"episode of {episode_len_model_s} s (model) does not fit in the usable "
                f"window {(lo, hi)}"
            )
        return (lo, hi - episode_len_model_s)

    def full_time_s(self, t_model_s: FloatArray) -> FloatArray:
        """Map model-scale seconds to absolute full-scale seconds.

        Args:
            t_model_s: Times, seconds, model scale, from the start of the committed record.

        Returns:
            ``spinup_s + t_model_s / sqrt(lam)``, seconds, full scale, absolute -- the same
            convention as the JONSWAP source, so the two legs share a clock.
        """
        t_model = np.asarray(t_model_s, dtype=np.float64)
        return np.asarray(self.sim_cfg.spinup_s + self.scale.to_full_time(t_model))

    def channels(self, t_full_s: FloatArray) -> MotionChannels:
        """Evaluate the sinusoidal channels at absolute full-scale times.

        Args:
            t_full_s: Times, seconds, full scale, absolute, inside the committed record
                ``[spinup_s, spinup_s + duration_s]``.

        Returns:
            The :class:`rld.deck.bridge.MotionChannels`, full scale. Rates are
            ``A*w*cos(w*t + phase)`` and accelerations ``-A*w**2*sin(w*t + phase)``: analytic
            derivatives, never differences.

        Raises:
            ValueError: If any requested time lies outside the committed record.
        """
        t_full = np.asarray(t_full_s, dtype=np.float64)
        lo = self.sim_cfg.spinup_s - TIME_BOUND_SLACK_S
        hi = self.sim_cfg.spinup_s + self.sim_cfg.duration_s + TIME_BOUND_SLACK_S
        if t_full.size and (float(t_full.min()) < lo or float(t_full.max()) > hi):
            raise ValueError(
                f"full-scale times [{float(t_full.min())}, {float(t_full.max())}] leave the "
                f"committed record [{self.sim_cfg.spinup_s}, "
                f"{self.sim_cfg.spinup_s + self.sim_cfg.duration_s}]"
            )
        w = self.params.w_e_rad_s
        roll_phase, pitch_phase, heave_phase = self.params.phase_rad
        sin_roll = np.sin(w * t_full + roll_phase)
        sin_pitch = np.sin(w * t_full + pitch_phase)
        sin_heave = np.sin(w * t_full + heave_phase)
        cos_roll = np.cos(w * t_full + roll_phase)
        cos_pitch = np.cos(w * t_full + pitch_phase)
        cos_heave = np.cos(w * t_full + heave_phase)
        roll_amp = self.params.roll_amp_deg
        pitch_amp = self.params.pitch_amp_deg
        heave_amp = self.params.heave_amp_m
        return MotionChannels(
            t_full_s=t_full,
            roll_deg=roll_amp * sin_roll,
            pitch_deg=pitch_amp * sin_pitch,
            heave_m=heave_amp * sin_heave,
            roll_rate_dps=roll_amp * w * cos_roll,
            pitch_rate_dps=pitch_amp * w * cos_pitch,
            heave_rate_m_s=heave_amp * w * cos_heave,
            heave_acc_m_s2=-heave_amp * w**2 * sin_heave,
            roll_acc_dps2=-roll_amp * w**2 * sin_roll,
            pitch_acc_dps2=-pitch_amp * w**2 * sin_pitch,
        )

    def deck_point(self, t_model_s: FloatArray, pad: str) -> DeckPointTrajectory:
        """Evaluate one pad's world-frame trajectory at model-scale times.

        Args:
            t_model_s: Times, seconds, model scale, from the start of the committed record.
            pad: Pad name from ``configs/deck/pad.yaml``.

        Returns:
            The :class:`rld.deck.bridge.DeckPointTrajectory` at **model** scale in the world
            frame, computed by the same kinematics and the same single Froude scaling as the
            JONSWAP source, with the same full-scale lever arm.

        Raises:
            ValueError: If ``pad`` is unknown or a time leaves the committed record.
        """
        t_model = np.asarray(t_model_s, dtype=np.float64)
        t_full = self.full_time_s(t_model)
        ch = self.channels(t_full)
        state_full = deck_point_state(
            roll_deg=ch.roll_deg,
            pitch_deg=ch.pitch_deg,
            heave_m=ch.heave_m,
            roll_rate_dps=ch.roll_rate_dps,
            pitch_rate_dps=ch.pitch_rate_dps,
            heave_rate_m_s=ch.heave_rate_m_s,
            roll_acc_dps2=ch.roll_acc_dps2,
            pitch_acc_dps2=ch.pitch_acc_dps2,
            heave_acc_m_s2=ch.heave_acc_m_s2,
            r_pad_m=self.pads.spec(pad).r_pad_full_m(self.length_full_m),
        )
        return DeckPointTrajectory(
            pad=pad,
            t_model_s=t_model,
            t_full_s=t_full,
            state=to_model_state(state_full, self.scale),
        )
