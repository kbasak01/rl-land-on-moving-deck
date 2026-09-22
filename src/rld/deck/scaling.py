"""Froude scaling between full-scale ship motion and model-scale (Crazyflie) motion.

The deck is scaled to the drone, not the drone to the deck (plan D0.1). With
``lam = L_model / L_full`` every quantity scales as ``lam`` raised to the exponent in
:data:`SCALE_EXPONENTS`:

===============  =========================  ==============
quantity          factor                     lam = 1/25
===============  =========================  ==============
length, heave     ``lam``                    x 0.04
time, period      ``sqrt(lam)``              x 0.2
linear velocity   ``sqrt(lam)``              x 0.2
linear accel.     ``1``                      x 1
angle             ``1``                      x 1
angular rate      ``1 / sqrt(lam)``          x 5
frequency         ``1 / sqrt(lam)``          x 5
===============  =========================  ==============

**Direction.** Every method on :class:`FroudeScale` converts **full scale to model scale**;
the inverse of the time map is spelled out separately as :meth:`FroudeScale.to_full_time`
because the bridge needs it on every call (it is handed model time and must ask dmf for
full-scale time). Argument names carry the scale: ``*_full_*`` in, model scale out.

``acceleration`` is scale-invariant only if ``g`` is the same at both scales. dmf uses
``g = 9.80665`` and gym-pybullet-drones ``9.8`` (``BaseAviary.py:74``), 0.068 % apart; that
is recorded in ``docs/protocol.md`` P1-D1 and is below every tolerance in Phase 1.
"""

from dataclasses import dataclass
from typing import TypeVar

from dmf.sim.encounter import GRAVITY_M_S2
from dmf.typedefs import FloatArray

__all__ = ["SCALE_EXPONENTS", "FroudeScale"]

#: Exponent of ``lam`` for each scaled quantity -- the table in plan D0.1, once, as data.
SCALE_EXPONENTS: dict[str, float] = {
    "length": 1.0,
    "time": 0.5,
    "velocity": 0.5,
    "acceleration": 0.0,
    "angle": 0.0,
    "angular_rate": -0.5,
    "frequency": -0.5,
}

#: Either a scalar or a float64 array; scaling is elementwise and shape-preserving.
Value = TypeVar("Value", float, FloatArray)


@dataclass(frozen=True)
class FroudeScale:
    """Froude similarity between a full-scale hull and its model-scale stand-in.

    Attributes:
        lam: Length scale ``L_model / L_full``, dimensionless, in ``(0, 1]``. ``1.0`` is the
            identity. Project default 0.04 (= 1/25), from ``configs/deck/scaling.yaml``.
        gravity_m_s2: Gravitational acceleration, metres per second squared, assumed equal
            at both scales. Used only by :meth:`froude_number`.
    """

    lam: float
    gravity_m_s2: float = GRAVITY_M_S2

    def __post_init__(self) -> None:
        """Validate the scale.

        Raises:
            ValueError: If ``lam`` is outside ``(0, 1]`` or ``gravity_m_s2`` is not
                positive.
        """
        if not 0.0 < self.lam <= 1.0:
            raise ValueError(f"lam must lie in (0, 1], got {self.lam}")
        if self.gravity_m_s2 <= 0.0:
            raise ValueError(f"gravity_m_s2 must be positive, got {self.gravity_m_s2}")

    @property
    def sqrt_lam(self) -> float:
        """Square root of the length scale, dimensionless. The time and velocity factor."""
        return float(self.lam**0.5)

    def factor(self, quantity: str) -> float:
        """Return the full-scale-to-model-scale multiplier for one named quantity.

        Args:
            quantity: A key of :data:`SCALE_EXPONENTS`.

        Returns:
            ``lam ** SCALE_EXPONENTS[quantity]``, dimensionless.

        Raises:
            ValueError: If ``quantity`` is not in :data:`SCALE_EXPONENTS`.
        """
        if quantity not in SCALE_EXPONENTS:
            raise ValueError(
                f"unknown scaled quantity {quantity!r}, expected one of {sorted(SCALE_EXPONENTS)}"
            )
        return float(self.lam ** SCALE_EXPONENTS[quantity])

    def _scaled(self, value: Value, quantity: str) -> Value:
        """Multiply ``value`` by the factor for ``quantity``, preserving shape and type."""
        return value * self.factor(quantity)

    def length(self, value_full_m: Value) -> Value:
        """Scale a length or displacement, metres full scale to metres model scale."""
        return self._scaled(value_full_m, "length")

    def time(self, value_full_s: Value) -> Value:
        """Scale a duration or period, seconds full scale to seconds model scale."""
        return self._scaled(value_full_s, "time")

    def velocity(self, value_full_m_s: Value) -> Value:
        """Scale a linear velocity, m/s full scale to m/s model scale."""
        return self._scaled(value_full_m_s, "velocity")

    def acceleration(self, value_full_m_s2: Value) -> Value:
        """Scale a linear acceleration, m/s^2 full scale to m/s^2 model scale (identity)."""
        return self._scaled(value_full_m_s2, "acceleration")

    def angle(self, value_full_deg: Value) -> Value:
        """Scale an angle, degrees full scale to degrees model scale (identity)."""
        return self._scaled(value_full_deg, "angle")

    def angular_rate(self, value_full_dps: Value) -> Value:
        """Scale an angular rate, deg/s full scale to deg/s model scale (x 1/sqrt(lam))."""
        return self._scaled(value_full_dps, "angular_rate")

    def frequency(self, value_full_hz: Value) -> Value:
        """Scale a frequency, Hz full scale to Hz model scale (x 1/sqrt(lam))."""
        return self._scaled(value_full_hz, "frequency")

    def to_model_time(self, t_full_s: Value) -> Value:
        """Convert full-scale time to model-scale time, seconds: ``t * sqrt(lam)``."""
        return self.time(t_full_s)

    def to_full_time(self, t_model_s: Value) -> Value:
        """Convert model-scale time to full-scale time, seconds: ``t / sqrt(lam)``.

        This is the direction the deck-motion bridge uses on every call: it is asked for a
        model-scale physics grid and must evaluate dmf at the corresponding full-scale
        times. The offset of the corpus record (dmf's 120 s spin-up) is **not** applied
        here; that belongs to the bridge, which knows the record it is reading.
        """
        return t_model_s / self.sqrt_lam

    def froude_number(self, speed_m_s: float, length_m: float) -> float:
        """Return ``v / sqrt(g * L)``, dimensionless, at whichever scale both inputs are.

        Args:
            speed_m_s: Speed, metres per second, model or full scale.
            length_m: Characteristic length, metres, at the **same** scale as ``speed_m_s``.

        Returns:
            The Froude number. Equal for a full-scale pair and its Froude-scaled
            model-scale pair, which is the invariant ``tests/test_scaling.py`` asserts.

        Raises:
            ValueError: If ``length_m`` is not positive.
        """
        if length_m <= 0.0:
            raise ValueError(f"length_m must be positive, got {length_m}")
        return float(speed_m_s / (self.gravity_m_s2 * length_m) ** 0.5)
