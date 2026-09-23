"""``pid_track_descend``: the naive baseline. Null the lateral error, then descend at a rate.

**Not privileged.** Reads only the observation.

The law
-------
All in the **world** frame, metres per second **model** scale, recomputed every control
step (30 Hz):

* **Lateral (x, y):** PI on ``e = (p_pad - p_drone)_xy`` --
  ``v_xy = kp * e + ki * integral(e dt)``, the integral term clamped per axis to
  ``integral_limit_m_s`` (anti-windup) and frozen while the drone is in deck contact (a
  landed drone must not wind up and slide), then the horizontal speed capped at
  ``lateral_speed_max_m_s``.
* **Vertical (z):** descend at the constant ``descent_rate_m_s`` **only once the lateral
  error is inside ``commit_radius_m``**; otherwise command ``v_z = 0`` (hold height). The
  gate has hysteresis: descent is released again only if the error grows past
  ``release_factor * commit_radius_m``, so it cannot chatter at the boundary.

Why the lateral gate is not optional: episodes start with a +/-0.3 m lateral spread against
a 0.26 m plate half-width, so ~13 % start outside the deck footprint (P2-D3). A pure
vertical descent from there is a deck-edge strike or a fall to the sea, which would score
this baseline badly for a geometric reason rather than a control one.

What it deliberately does not do: no feedforward of the deck's motion (that is
``pid_feedforward``), no flare, no timing. It descends into whatever the deck is doing, so
its touchdown closing speed is ``descent_rate + v_pad_z`` whenever the pad is rising.

Every gain is in ``configs/control/pid_track_descend.yaml``; the tuning procedure and the
final values are ``docs/protocol.md`` P3-D3.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dmf.typedefs import FloatArray

from rld.control.base import ControlSpec, PrivilegedContext, StepClock, reject_motion_feed
from rld.control.config import PidConfig, load_pid
from rld.control.obs_view import ObsLayout, ObsView, obs_layout, setpoint_to_action, view

if TYPE_CHECKING:
    from rld.deck.forecast import ShipMotionFeed

__all__ = ["PidTrackDescend", "TrackDescendBase", "make_pid_track_descend"]


class TrackDescendBase:
    """Shared machinery: the lateral PI, the lateral gate, the clock, the action conversion.

    Subclasses override :meth:`_feedforward_m_s` (deck-velocity feedforward) and
    :meth:`_vertical_m_s` (the descent decision). Not registered itself.

    Attributes:
        name: Registry name.
        privileged: Whether the controller reads privileged information.
        needs_motion_feed: Whether ``reset`` requires a
            :class:`~rld.deck.forecast.ShipMotionFeed`; False here and in every classical
            baseline, which raise if handed one.
        pid: The tracker and descent gains.
        spec: The environment facts the controller may know.
    """

    name: str = "track_descend_base"
    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(self, pid: PidConfig, spec: ControlSpec) -> None:
        """Build the controller. No state survives :meth:`reset` except the configs.

        Args:
            pid: Tracker and descent gains.
            spec: Committed environment configs (``v_max``, rates, layout, scale).
        """
        self.pid = pid
        self.spec = spec
        self._layout: ObsLayout = obs_layout(spec.observation)
        self._clock = StepClock(spec.ctrl_dt_s, spec.landing.episode_len_s)
        self._integral_m = np.zeros(2, dtype=np.float64)
        self._lateral_ok = False
        self._seed = 0
        self._ready = False

    # ------------------------------------------------------------------ contract

    def reset(
        self,
        seed: int,
        context: PrivilegedContext | None = None,
        motion_feed: "ShipMotionFeed | None" = None,
    ) -> None:
        """Clear every per-episode state.

        Args:
            seed: Episode seed; recorded only, the controller is deterministic.
            context: Ignored by non-privileged controllers.
            motion_feed: Must be ``None``: this controller does not consume the
                ship-motion feed. A subclass that does consumes it and passes ``None`` up.

        Raises:
            ValueError: If a feed is passed (a runner wiring bug).
        """
        del context
        reject_motion_feed(self.name, motion_feed)
        self._seed = int(seed)
        self._clock.reset()
        self._integral_m = np.zeros(2, dtype=np.float64)
        self._lateral_ok = False
        self._ready = True

    def act(self, obs: FloatArray) -> FloatArray:
        """Return the action for one observation.

        Args:
            obs: The observation vector.

        Returns:
            ``(3,)`` float32 action in ``[-1, 1]^3``.

        Raises:
            RuntimeError: If called before :meth:`reset`, or out of step with the
                environment (see :class:`~rld.control.base.StepClock`).
        """
        if not self._ready:
            raise RuntimeError(f"{self.name}: act() before reset()")
        return setpoint_to_action(self.setpoint_m_s(obs), self.spec.landing)

    def setpoint_m_s(self, obs: FloatArray) -> FloatArray:
        """Return the world-frame velocity setpoint, before normalisation, and advance.

        Public so Phase 6's residual wrapper can compose in metres per second if it wants
        to, and so tests can check the law without the norm cap.

        Args:
            obs: The observation vector.

        Returns:
            ``(3,)`` metres per second model scale, world frame, uncapped.
        """
        v = view(obs, self._layout)
        t = self._clock.tick(v.time_fraction)
        lateral = self._lateral_m_s(v)
        vertical = self._vertical_m_s(v, t)
        return np.array([lateral[0], lateral[1], vertical]) + self._feedforward_m_s(v)

    # ------------------------------------------------------------------ pieces

    @property
    def lateral_ok(self) -> bool:
        """Whether the lateral gate currently permits descent."""
        return self._lateral_ok

    @property
    def time_s(self) -> float:
        """Episode time of the next observation, seconds model scale."""
        return self._clock.steps * self.spec.ctrl_dt_s

    def _lateral_m_s(self, v: ObsView) -> FloatArray:
        """Run the lateral PI and update the lateral gate.

        Args:
            v: The observation, world frame.

        Returns:
            ``(2,)`` horizontal velocity command, metres per second model scale, world
            frame, speed-capped.
        """
        cfg = self.pid
        error = v.lateral_error_m
        distance = float(np.linalg.norm(error))
        if distance <= cfg.commit_radius_m:
            self._lateral_ok = True
        elif distance > cfg.release_factor * cfg.commit_radius_m:
            self._lateral_ok = False

        if cfg.ki_xy_per_s2 > 0.0:
            if not v.in_contact:
                self._integral_m = self._integral_m + error * self.spec.ctrl_dt_s
            bound = cfg.integral_limit_m_s / cfg.ki_xy_per_s2
            self._integral_m = np.clip(self._integral_m, -bound, bound)
        command = cfg.kp_xy_per_s * error + cfg.ki_xy_per_s2 * self._integral_m
        speed = float(np.linalg.norm(command))
        if speed > cfg.lateral_speed_max_m_s:
            command = command * (cfg.lateral_speed_max_m_s / speed)
        return np.asarray(command, dtype=np.float64)

    def _vertical_m_s(self, v: ObsView, t_s: float) -> float:
        """Return the vertical command before feedforward.

        Args:
            v: The observation, world frame.
            t_s: Episode time, seconds model scale.

        Returns:
            ``-descent_rate`` when the lateral gate is open, else 0; metres per second
            model scale, world ``z``.
        """
        del v, t_s
        return -self.pid.descent_rate_m_s if self._lateral_ok else 0.0

    def _feedforward_m_s(self, v: ObsView) -> FloatArray:
        """Return the deck-velocity feedforward term.

        Args:
            v: The observation, world frame.

        Returns:
            ``(3,)`` metres per second model scale; zero here.
        """
        del v
        return np.zeros(3, dtype=np.float64)


class PidTrackDescend(TrackDescendBase):
    """``pid_track_descend``: lateral PI, lateral-gated constant-rate descent. Not privileged.

    See the module docstring for the law. Gains: ``configs/control/pid_track_descend.yaml``.
    """

    name = "pid_track_descend"
    privileged = False


def make_pid_track_descend(config_path: Path, spec: ControlSpec) -> PidTrackDescend:
    """Registry factory.

    Args:
        config_path: ``configs/control/pid_track_descend.yaml`` or an override.
        spec: Committed environment configs.

    Returns:
        A fresh controller; call :meth:`~TrackDescendBase.reset` before use.
    """
    return PidTrackDescend(load_pid(config_path), spec)
