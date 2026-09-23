"""``pid_feedforward_lowvz_cut``: ``pid_feedforward_lowvz`` plus a post-contact throttle cut.

**Not privileged** (``privileged=False``) and **no ship-motion feed**
(``needs_motion_feed=False``). The rule reads only two blocks of the committed observation:
the ``in_contact`` flag and the ``deck_normal``. Everything lowvz reads is read as lowvz reads
it.

Why it exists
-------------
P5-D1 found that about 80 % of lowvz's bounces are rim-first rocking. The drone keeps close to
hover thrust after contact, the rim lifts by about 2 mm for more than the 50 ms contact-loss
grace, and the episode is scored ``bounce`` while the CoM is still closing on the deck. The
user's decision at the end of P5-D1 added this controller. It shows how much of a success gain
can come from post-contact thrust alone. It does **not** replace lowvz as the H1a reference.

The rule (``docs/protocol.md`` P5-D2, fixed before any episode was flown; no tuning budget)
------------------------------------------------------------------------------------------
* **Touchdown inference.** The observation's ``in_contact`` flag, as :class:`ObsView` reads it.
  This is the environment's contact-manifold detector at the last physics substep of the
  control step.
* **Latch.** The latch sets on the first observation with ``in_contact`` = 1 and holds until
  :meth:`reset`. It is never released. A contact gap longer than the 0.05 s grace ends the
  episode anyway, and the shorter gaps are when the down-force is needed.
* **Before the latch.** The setpoint is lowvz's, bit for bit, from the same code path.
* **After the latch.** The setpoint is lowvz's setpoint minus ``cut_speed_m_s`` along the
  observed unit deck normal, then the shared norm cap. With ``cut_speed_m_s`` = ``v_max`` =
  1.5 m/s, and in P5-D2's post-contact envelope, ``DSLPIDControl``'s collective falls below
  the level at which every motor sits at ``MIN_PWM``. The motors go to idle: 0.1126 N =
  42.6 % of weight, with zero attitude torque. That is the lowest thrust the shared action
  space can produce. Outside that envelope the collective stays at the floor, but one motor
  can sit slightly above idle.

Lowvz's clock, lateral integrator and lateral gate advance on every step, before and after the
latch, exactly as they do in lowvz.

Units and scales
----------------
Metres, metres per second and seconds are **model** scale (lambda = 1/25). Setpoints are
world-frame velocities in metres per second, before normalisation.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dmf.typedefs import FloatArray

from rld.control.base import ControlSpec, PrivilegedContext
from rld.control.config import LowvzCutConfig, load_lowvz_cut
from rld.control.feedforward import PidFeedforward
from rld.control.obs_view import view

if TYPE_CHECKING:
    from rld.deck.forecast import ShipMotionFeed

__all__ = ["LOWVZ_CUT_NAME", "PidFeedforwardLowvzCut", "make_pid_feedforward_lowvz_cut"]

#: Registry name.
LOWVZ_CUT_NAME = "pid_feedforward_lowvz_cut"


class PidFeedforwardLowvzCut(PidFeedforward):
    """``pid_feedforward_lowvz`` with a latched post-contact throttle cut. Not privileged.

    Attributes:
        ff: The full config: lowvz's gains, ``gains_from`` and ``cut_speed_m_s``.
    """

    name = LOWVZ_CUT_NAME
    privileged = False
    needs_motion_feed = False
    ff: LowvzCutConfig

    def __init__(self, cfg: LowvzCutConfig, spec: ControlSpec) -> None:
        """Build the controller.

        Args:
            cfg: Lowvz's gains plus the cut.
            spec: Committed environment configs.

        Raises:
            ValueError: If the observation layout has no ``in_contact`` block. Without that
                block the rule could never fire, and the controller would silently fly as
                lowvz.
        """
        super().__init__(cfg, spec, name=LOWVZ_CUT_NAME)
        if "in_contact" not in self._layout.slices:
            raise ValueError(
                f"{LOWVZ_CUT_NAME} infers touchdown from the in_contact observation block, "
                "which this observation config does not include"
            )
        self._latched = False
        self._latch_step: int | None = None

    def reset(
        self,
        seed: int,
        context: PrivilegedContext | None = None,
        motion_feed: "ShipMotionFeed | None" = None,
    ) -> None:
        """Clear every per-episode state, including the latch.

        Args:
            seed: Episode seed; recorded only, the controller is deterministic.
            context: Ignored; not privileged.
            motion_feed: Must be ``None``.

        Raises:
            ValueError: If a feed is passed (a runner wiring bug).
        """
        super().reset(seed, context, motion_feed)
        self._latched = False
        self._latch_step = None

    @property
    def latched(self) -> bool:
        """Whether touchdown has been inferred in this episode, so the cut is active."""
        return self._latched

    @property
    def latch_step(self) -> int | None:
        """The ``act`` index (0-based, 1/30 s model steps) at which the latch set, or None."""
        return self._latch_step

    def setpoint_m_s(self, obs: FloatArray) -> FloatArray:
        """Return lowvz's setpoint, or that setpoint minus the cut once contact has been seen.

        Args:
            obs: The observation vector.

        Returns:
            ``(3,)`` metres per second model scale, world frame, uncapped. The norm cap is
            applied afterwards by :meth:`act`, as for every controller.
        """
        step = self._clock.steps
        base = super().setpoint_m_s(obs)
        v = view(obs, self._layout)
        if not self._latched and v.in_contact:
            self._latched = True
            self._latch_step = step
        if not self._latched:
            return base
        normal = np.asarray(v.deck_normal, dtype=np.float64)
        return np.asarray(base - self.ff.cut_speed_m_s * normal, dtype=np.float64)


def make_pid_feedforward_lowvz_cut(config_path: Path, spec: ControlSpec) -> PidFeedforwardLowvzCut:
    """Registry factory for ``pid_feedforward_lowvz_cut``.

    Args:
        config_path: ``configs/control/pid_feedforward_lowvz_cut.yaml`` or an override.
        spec: Committed environment configs.

    Returns:
        A fresh controller; call ``reset`` before use.
    """
    return PidFeedforwardLowvzCut(load_lowvz_cut(config_path), spec)
