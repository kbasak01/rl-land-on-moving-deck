"""dmf's quiescence thresholds, Froude-converted to the model-scale deck.

Source, not copy
----------------
The two threshold sets are **imported** from :mod:`dmf.eval.quiescence` --
:data:`~dmf.eval.quiescence.PERMISSIVE` (3.0 deg roll, 2.0 deg pitch, 0.8 m/s heave rate,
2.0 s sustain) and :data:`~dmf.eval.quiescence.STRICT` (1.5 deg, 1.0 deg, 0.4 m/s, 3.0 s),
all **full** scale -- and converted here through :class:`rld.deck.scaling.FroudeScale`, so a
change to either side flows through rather than being shadowed by a literal.

The conversion (plan D0.1's table)
----------------------------------
* angles unchanged (``lam ** 0``);
* heave rate, a velocity, times ``sqrt(lam)``: 0.8 -> 0.16 m/s, 0.4 -> 0.08 m/s at
  ``lam = 1/25``;
* sustain, a time, times ``sqrt(lam)``: 2.0 -> 0.4 s, 3.0 -> 0.6 s.

Documented deviation from dmf's rule
------------------------------------
dmf tests ``|heave rate|``, the **CG**'s vertical velocity. The controllers here test the
**pad deck-point** vertical velocity ``v_pad_z``, because the pad is what the drone lands on
and at an aft pad it differs from the CG's by the pitch lever-arm term (P1-D2: aft/CG
``v_z`` RMS ratio 0.685-7.378 over the grid). The threshold *value* is dmf's, converted;
only the quantity it is applied to changes. Recorded in ``docs/protocol.md`` P3-D3.

Sustain convention
------------------
dmf's P6-D2: a run of ``k`` consecutive in-limit samples lasts ``k / fs`` seconds, so the
minimum run is :func:`dmf.eval.quiescence.sustain_samples` ``(sustain_s, fs)``. Reused, not
re-derived: 0.4 s is 12 control samples at 30 Hz (96 physics samples at 240 Hz).

One predicate
-------------
:class:`QuiescenceRule` is the **only** commit predicate. ``gated`` applies it to the
observed past, ``oracle_gated`` to the true future, both at the control rate (Gate 3
remediation, 2026-09-22, ``docs/protocol.md`` P3-D3).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from dmf.eval.quiescence import PERMISSIVE, STRICT, QuiescenceThresholds, sustain_samples
from dmf.typedefs import FloatArray

from rld.deck.scaling import FroudeScale

__all__ = [
    "THRESHOLD_SETS",
    "ModelQuiescenceLimits",
    "QuiescenceRule",
    "ThresholdSetName",
    "froude_limits",
    "limits_for",
]

#: Threshold-set names a controller YAML may name.
type ThresholdSetName = Literal["permissive", "strict"]

#: dmf's threshold sets by name, full scale, imported rather than copied.
THRESHOLD_SETS: dict[str, QuiescenceThresholds] = {
    PERMISSIVE.name: PERMISSIVE,
    STRICT.name: STRICT,
}


@dataclass(frozen=True)
class ModelQuiescenceLimits:
    """Quiescence limits on the model-scale deck.

    Attributes:
        name: dmf's threshold-set name, ``"permissive"`` or ``"strict"``.
        roll_deg: Maximum absolute deck roll, degrees (scale-invariant).
        pitch_deg: Maximum absolute deck pitch, degrees (scale-invariant).
        pad_vz_m_s: Maximum absolute **pad deck-point** vertical velocity, metres per
            second **model** scale. dmf's CG heave-rate limit times ``sqrt(lam)``.
        sustain_s: Minimum quiescent duration, seconds **model** scale. dmf's sustain times
            ``sqrt(lam)``.
        source: dmf's full-scale thresholds this was converted from.
        lam: The length scale used, dimensionless.
    """

    name: str
    roll_deg: float
    pitch_deg: float
    pad_vz_m_s: float
    sustain_s: float
    source: QuiescenceThresholds
    lam: float

    def sustain_samples(self, rate_hz: float) -> int:
        """Return the minimum run length at a sampling rate (dmf's P6-D2 convention).

        Args:
            rate_hz: Sampling rate, hertz model scale.

        Returns:
            ``ceil(sustain_s * rate_hz)``, at least 1, dimensionless.
        """
        return sustain_samples(self.sustain_s, rate_hz)

    def holds(self, roll_deg: float, pitch_deg: float, pad_vz_m_s: float) -> bool:
        """Return whether one instantaneous deck state is inside all three limits.

        Args:
            roll_deg: Deck roll, degrees.
            pitch_deg: Deck pitch, degrees.
            pad_vz_m_s: Pad vertical velocity, metres per second model scale.

        Returns:
            True when ``|roll|``, ``|pitch|`` and ``|v_z|`` are all at or below their limits.
        """
        return (
            abs(roll_deg) <= self.roll_deg
            and abs(pitch_deg) <= self.pitch_deg
            and abs(pad_vz_m_s) <= self.pad_vz_m_s
        )


def froude_limits(thresholds: QuiescenceThresholds, scale: FroudeScale) -> ModelQuiescenceLimits:
    """Convert dmf's full-scale thresholds to the model-scale deck.

    Args:
        thresholds: dmf's thresholds: degrees, metres per second and seconds, **full**
            scale.
        scale: The Froude scale; ``lam = 1/25`` in the committed config.

    Returns:
        The :class:`ModelQuiescenceLimits`: angles unchanged, the rate and the sustain times
        ``sqrt(lam)``.
    """
    return ModelQuiescenceLimits(
        name=thresholds.name,
        roll_deg=float(scale.angle(thresholds.roll_deg)),
        pitch_deg=float(scale.angle(thresholds.pitch_deg)),
        pad_vz_m_s=float(scale.velocity(thresholds.heave_rate_mps)),
        sustain_s=float(scale.time(thresholds.sustain_s)),
        source=thresholds,
        lam=scale.lam,
    )


def limits_for(name: str, scale: FroudeScale) -> ModelQuiescenceLimits:
    """Return a named dmf threshold set, Froude-converted.

    Args:
        name: ``"permissive"`` or ``"strict"``.
        scale: The Froude scale.

    Returns:
        The converted limits.

    Raises:
        ValueError: If ``name`` is not one of dmf's sets.
    """
    if name not in THRESHOLD_SETS:
        raise ValueError(
            f"unknown threshold set {name!r}, expected one of {sorted(THRESHOLD_SETS)}"
        )
    return froude_limits(THRESHOLD_SETS[name], scale)


@dataclass(frozen=True)
class QuiescenceRule:
    """The one commit predicate shared by ``gated`` and ``oracle_gated``.

    A window of deck samples qualifies when it holds **exactly** :attr:`n_samples` samples,
    spaced :attr:`spacing_s` apart, and **every** sample is inside all three limits by
    :meth:`ModelQuiescenceLimits.holds`. The two controllers differ only in *which* samples
    they hand to :meth:`verdict`:

    * ``gated``: the trailing ``n_samples`` **observed** control-rate samples, ending now;
    * ``oracle_gated``: ``n_samples`` **true future** samples at the same control-rate
      spacing, starting at the predicted touchdown time.

    Both build the rule the same way (:class:`rld.control.gated.GatedBase`), at the control
    rate, so sample count, spacing, thresholds and the pad-``v_z`` quantity cannot drift
    apart. With ``permissive`` at ``lam = 1/25`` and 30 Hz: 12 samples, 1/30 s apart.

    Attributes:
        limits: The model-scale limits.
        rate_hz: Sampling rate of the window, hertz model scale (the control rate).
    """

    limits: ModelQuiescenceLimits
    rate_hz: float

    def __post_init__(self) -> None:
        """Validate the rate.

        Raises:
            ValueError: If ``rate_hz`` is not positive.
        """
        if self.rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {self.rate_hz}")

    @property
    def n_samples(self) -> int:
        """Samples a qualifying window holds, dimensionless (dmf's P6-D2 run length)."""
        return self.limits.sustain_samples(self.rate_hz)

    @property
    def spacing_s(self) -> float:
        """Spacing between the window's samples, seconds model scale."""
        return 1.0 / self.rate_hz

    def verdict(self, roll_deg: FloatArray, pitch_deg: FloatArray, pad_vz_m_s: FloatArray) -> bool:
        """Return whether one window of deck samples permits a commit.

        Args:
            roll_deg: Deck roll, degrees, dmf sign, ``(k,)``, oldest (or earliest) first.
            pitch_deg: Deck pitch, degrees, dmf sign, ``(k,)``.
            pad_vz_m_s: Pad deck-point vertical velocity, metres per second model scale,
                world ``z``, ``(k,)``.

        Returns:
            True when ``k == n_samples`` and every sample is inside all three limits. A
            shorter window -- history not yet long enough, or a future window that runs
            past the end of the trajectory -- is False.

        Raises:
            ValueError: If the channels differ in length, or ``k > n_samples`` (a caller
                handing over more than one window is a bug, not a verdict).
        """
        roll = np.asarray(roll_deg, dtype=np.float64).reshape(-1)
        pitch = np.asarray(pitch_deg, dtype=np.float64).reshape(-1)
        vz = np.asarray(pad_vz_m_s, dtype=np.float64).reshape(-1)
        if not roll.size == pitch.size == vz.size:
            raise ValueError(
                f"channel lengths differ: roll {roll.size}, pitch {pitch.size}, v_z {vz.size}"
            )
        if roll.size > self.n_samples:
            raise ValueError(f"window has {roll.size} samples, the rule takes {self.n_samples}")
        if roll.size < self.n_samples:
            return False
        return all(
            self.limits.holds(float(r), float(p), float(z))
            for r, p, z in zip(roll, pitch, vz, strict=True)
        )
