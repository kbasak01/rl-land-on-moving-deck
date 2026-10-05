"""The perception stand-in: the deck as a perception pipeline would report it (P7-D4).

What it stands in for
---------------------
Nothing in this project sees an image. The stand-in degrades the **deck** -- the analytic
:class:`~rld.envs.platform.PlatformSample` at the pad -- the way a deck-state estimator would
deliver it: a transport delay, a refresh rate that need not match the control rate, and
measurement noise on the pad position and velocity. It is a stand-in, it is labelled as one in
every table, and the required caveat in ``docs/protocol.md`` says so.

What is perceived (P7-D4, superseding the Phase 2 stand-in)
-----------------------------------------------------------
The perceived quantity is the deck sample, **not** the relative block. At control step ``k``
the perceived sample is the true deck sample of step ``k - L`` (``L`` the latency in whole
control steps), held at the refresh rate, with zero-mean Gaussian noise added to its pad
position (sigma_p) and pad velocity (sigma_v) **in the world frame**. Orientation, normal,
tilt, angular velocity, acceleration and the sample's time stamp are the delayed true ones;
they get no noise.

The environment then builds every deck-derived observation entry -- relative position,
relative velocity, deck normal, relative tilt and pad-plane clearance -- from the perceived
sample and the drone's **current** true state, through the unchanged
:func:`rld.envs.observation.build_observation`. The drone's own state is not perceived and
stays clean and current, so ``velocity + rel_velocity = v_pad(t - L) + noise``: a consistent,
stale deck estimate (P7-D4, review M1). The Phase 2 stand-in instead replaced only the six
relative entries with a delayed copy, which mixed timestamps (``v_pad(t - L) + v_drone(t) -
v_drone(t - L)``) and left the clearance, normal and tilt ideal.

Termination, reward, touchdown detection and every evaluation metric read the **true** deck;
only the observation reads the perceived one.

Off by default
--------------
``configs/env/noise.yaml`` ships ``enabled: false``. Phases 5 and 6 train clean; Phase 7
ablates sigma and latency on the frozen episode lists. With the stand-in disabled the
environment never calls it, so the observation path is the clean one, untouched.

Seeding
-------
The generator is a **spawned child** of the episode seed sequence, never a bare
``np.random.default_rng()`` and never the same generator that draws the episode start
offset or the initial drone state. If turning noise on consumed draws from the same stream,
the noisy and the clean run of the same seed would be different *episodes* and the Phase 7
ablation would stop being a paired comparison. Per control step it draws 3 position values
(if sigma_p > 0) and then 3 velocity values (if sigma_v > 0), in that order.

Units and scales
----------------
Sigmas are metres and metres per second **model** scale, world frame; latency is
milliseconds model scale; the hold rate is hertz model scale. At ``lam = 1/25`` a 33.4 ms
model latency is 167 ms full scale.
"""

import dataclasses
from collections import deque

import numpy as np

from rld.envs.config import NoiseConfig
from rld.envs.platform import PlatformSample

__all__ = ["PerceptionNoise", "copy_sample"]


def copy_sample(sample: PlatformSample) -> PlatformSample:
    """Return a copy of a deck sample that owns its arrays.

    :meth:`~rld.envs.platform.DeckTrajectory.sample` returns rows that are *views* into the
    trajectory's arrays. The delay line keeps samples for several control steps, so it keeps
    copies: nothing done to the trajectory's buffers can then reach a stored sample, and
    nothing done to a stored sample can reach the trajectory. Values are copied bit for bit.

    Args:
        sample: The deck sample, model scale, world frame.

    Returns:
        An equal :class:`PlatformSample` whose array fields are fresh float64 copies.
    """
    return dataclasses.replace(
        sample,
        position_m=np.array(sample.position_m, dtype=np.float64, copy=True),
        quaternion=np.array(sample.quaternion, dtype=np.float64, copy=True),
        euler_xyz_rad=np.array(sample.euler_xyz_rad, dtype=np.float64, copy=True),
        velocity_m_s=np.array(sample.velocity_m_s, dtype=np.float64, copy=True),
        angular_velocity_rad_s=np.array(sample.angular_velocity_rad_s, dtype=np.float64, copy=True),
        acceleration_m_s2=np.array(sample.acceleration_m_s2, dtype=np.float64, copy=True),
        normal=np.array(sample.normal, dtype=np.float64, copy=True),
    )


class PerceptionNoise:
    """Stateful per-episode perception of the deck sample: delay, hold, then noise.

    One instance per episode. Construct it from a generator spawned off the episode seed
    sequence and call :meth:`perceive` exactly once per control step, in order (the
    environment calls it from ``_computeObs``, once at reset and once per ``step``).

    Attributes:
        cfg: The noise settings.
        latency_steps: Transport delay in whole control steps, dimensionless. The
            configured millisecond latency is quantised **down**, so a latency shorter than
            one control step is exactly zero rather than silently rounded up.
        hold_steps: Control steps between refreshes, dimensionless, at least 1.
    """

    def __init__(self, cfg: NoiseConfig, rng: np.random.Generator, ctrl_freq_hz: float) -> None:
        """Build the per-episode noise state.

        Args:
            cfg: The noise settings, from ``configs/env/noise.yaml``.
            rng: A generator spawned off the episode seed sequence.
            ctrl_freq_hz: Control rate, hertz model scale, used to quantise the latency and
                the hold interval into control steps.

        Raises:
            ValueError: If ``ctrl_freq_hz`` is not positive.
        """
        if ctrl_freq_hz <= 0.0:
            raise ValueError(f"ctrl_freq_hz must be positive, got {ctrl_freq_hz}")
        self.cfg = cfg
        self._rng = rng
        self._ctrl_freq_hz = float(ctrl_freq_hz)
        self.latency_steps = int(cfg.latency_ms * 1e-3 * ctrl_freq_hz)
        self.hold_steps = max(1, int(round(ctrl_freq_hz / cfg.hold_freq_hz)))
        self._step = 0
        self._held: PlatformSample | None = None
        self._delay: deque[PlatformSample] = deque(maxlen=self.latency_steps + 1)

    def reset(self) -> None:
        """Clear the hold and the delay line. Call at the start of every episode."""
        self._step = 0
        self._held = None
        self._delay = deque(maxlen=self.latency_steps + 1)

    def perceive(self, deck: PlatformSample) -> PlatformSample:
        """Return the deck sample the observation is built from at this control step.

        The order is delay, then hold, then noise: a real pipeline delivers a stale sample
        and *then* the estimator's error is added to what was delivered, so the noise is not
        itself held constant across the hold interval.

        Args:
            deck: The **true** deck sample at the pad at the current control step, metres
                and metres per second model scale, world frame. Never modified.

        Returns:
            The perceived sample: the true sample of control step ``k - latency_steps``
            (before that many samples exist, the oldest one available), refreshed every
            ``hold_steps`` control steps, with ``N(0, sigma_p^2)`` added per world axis to
            ``position_m`` and ``N(0, sigma_v^2)`` per world axis to ``velocity_m_s``. Every
            other field is the delayed true value. With the stand-in disabled, ``deck``
            itself, untouched, and no state advances and no draw is made.
        """
        if not self.cfg.enabled:
            return deck

        # The deque holds `latency_steps + 1` samples, so its oldest entry is exactly the
        # sample `latency_steps` control steps ago once it has filled, and the oldest
        # sample available before that -- which is the right warm-up behaviour: at t = 0 no
        # older measurement exists, so the freshest one is all an estimator could report.
        self._delay.append(copy_sample(deck))
        delayed = self._delay[0]

        if self._held is None or self._step % self.hold_steps == 0:
            self._held = delayed
        self._step += 1

        # A fresh copy every step, so a caller can never mutate the held sample through it.
        out = copy_sample(self._held)
        position = out.position_m
        velocity = out.velocity_m_s
        if self.cfg.position_sigma_m > 0.0:
            position = position + self._rng.normal(0.0, self.cfg.position_sigma_m, size=3)
        if self.cfg.velocity_sigma_m_s > 0.0:
            velocity = velocity + self._rng.normal(0.0, self.cfg.velocity_sigma_m_s, size=3)
        # Only position and velocity move. The normal, quaternion, Euler triple and tilt are
        # attitude quantities, which P7-D4 leaves noise-free, and none of them depends on the
        # pad position or velocity, so the perturbed sample stays self-consistent.
        return dataclasses.replace(out, position_m=position, velocity_m_s=velocity)
