"""The perception stand-in: Gaussian noise, fixed latency and sample-and-hold.

What it stands in for
---------------------
Nothing in this project sees an image. The observation's relative-pad block is a *given*
relative pose, which is exactly the quantity a downward-looking camera plus a fiducial on
the deck would estimate, and this module degrades it the way such an estimator does: with
measurement noise, a transport delay, and a refresh rate that need not match the control
rate. It is a stand-in, it is labelled as one in every table, and the required caveat in
``docs/protocol.md`` says so.

It is applied to the **relative pad pose only** -- relative position and relative velocity.
The drone's own attitude, body rates and velocity are left clean, because those come from
an IMU and a state estimator in any plausible deployment, not from the camera.

Off by default
--------------
``configs/env/noise.yaml`` ships ``enabled: false``. Phases 5 and 6 train clean; Phase 7
ablates sigma and latency on the frozen episode lists.

Seeding
-------
The generator is a **spawned child** of the episode seed sequence, never a bare
``np.random.default_rng()`` and never the same generator that draws the episode start
offset or the initial drone state. If turning noise on consumed draws from the same stream,
the noisy and the clean run of the same seed would be different *episodes* and the Phase 7
ablation would stop being a paired comparison.

Units and scales
----------------
Sigmas are metres and metres per second **model** scale, latency is milliseconds model
scale, the hold rate is hertz model scale. At ``lam = 1/25`` a 33 ms model latency is
165 ms full scale.
"""

from collections import deque

import numpy as np
from dmf.typedefs import FloatArray

from rld.envs.config import NoiseConfig

__all__ = ["PerceptionNoise"]


class PerceptionNoise:
    """Stateful per-episode degradation of the relative-pad observation block.

    One instance per episode. Construct it from a generator spawned off the episode seed
    sequence and call :meth:`apply` once per control step, in order.

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
        self._held: FloatArray | None = None
        self._delay: deque[FloatArray] = deque(maxlen=self.latency_steps + 1)

    def reset(self) -> None:
        """Clear the hold and the delay line. Call at the start of every episode."""
        self._step = 0
        self._held = None
        self._delay = deque(maxlen=self.latency_steps + 1)

    def apply(self, relative_block: FloatArray) -> FloatArray:
        """Degrade one relative-pad block.

        The order is delay, then hold, then noise: a real pipeline delivers a stale sample
        and *then* the estimator's error is added to what was delivered, so the noise is not
        itself held constant across the hold interval.

        Args:
            relative_block: ``(6,)`` = relative position (metres model scale) followed by
                relative velocity (metres per second model scale), in whichever frame the
                observation builder uses.

        Returns:
            A new ``(6,)`` array, same units. The input is never modified in place.

        Raises:
            ValueError: If the block is not ``(6,)``.
        """
        block = np.asarray(relative_block, dtype=np.float64).reshape(-1)
        if block.size != 6:
            raise ValueError(f"relative block must have 6 entries, got {block.size}")
        if not self.cfg.enabled:
            return block.copy()

        # The deque holds `latency_steps + 1` samples, so its oldest entry is exactly the
        # sample `latency_steps` control steps ago once it has filled, and the oldest
        # sample available before that -- which is the right warm-up behaviour: at t = 0 no
        # older measurement exists, so the freshest one is all an estimator could report.
        self._delay.append(block.copy())
        delayed = self._delay[0]

        if self._held is None or self._step % self.hold_steps == 0:
            self._held = delayed.copy()
        self._step += 1

        out = self._held.copy()
        if self.cfg.position_sigma_m > 0.0:
            out[0:3] = out[0:3] + self._rng.normal(0.0, self.cfg.position_sigma_m, size=3)
        if self.cfg.velocity_sigma_m_s > 0.0:
            out[3:6] = out[3:6] + self._rng.normal(0.0, self.cfg.velocity_sigma_m_s, size=3)
        return out
