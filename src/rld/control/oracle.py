"""``oracle_gated``: the same hover-and-commit, timed on the **true future** deck motion.

**PRIVILEGED** (``privileged=True`` in the registry). It reads the episode's true future
deck-point trajectory through :class:`~rld.control.base.PrivilegedContext`, which no
deployed vehicle has. It is an **upper bound on commit timing** under this hover-and-commit
law and ``gated``'s own rule, and must never be reported as a deployable result. ``reset``
raises without a context, so it cannot silently run as something else.

Everything except the rule is ``gated``'s: ``pid_feedforward``'s tuned gains, hover at
``hover_height_m``, commit only with the lateral gate open and the drone at hover, and the
same Froude-converted dmf thresholds (pad deck-point ``v_z`` deviation included).

The rule (true future window)
-----------------------------
At episode time ``t`` the controller predicts its own touchdown time as

    t_td = t + max(height, 0) / descent_rate

seconds model scale -- ``height`` is the observation's clearance along the deck normal. The
prediction assumes the committed descent closes on the pad at exactly ``descent_rate``,
which is what the ``k_ff * v_pad`` feedforward is for; it ignores ``DSLPIDControl``'s
tracking lag and the transient from hover to descent speed, both of which make the real
touchdown slightly later.

It then applies **exactly** ``gated``'s predicate,
:class:`rld.control.quiescence.QuiescenceRule` (built once in
:class:`~rld.control.gated.GatedBase` at the control rate): the same sample count
(``sustain_samples(sustain, 30 Hz)`` = 12), the same 1/30 s spacing, the same thresholds and
the same quantities -- roll and pitch from the deck normal through
:func:`rld.control.obs_view.deck_angles_deg`, and the pad deck-point world ``v_z``. The only
difference is the samples: the **true future** deck states at ``t_td + j / 30`` s,
``j = 0 .. 11`` (every ``physics_freq / ctrl_freq`` = 8th physics sample of the trajectory
the plate is driven along, from the first at or after ``t_td``), rather than the observed
past. A window that would run past the end of the episode's trajectory does not qualify.

*History.* Until the Gate 3 remediation (2026-09-22) the oracle tested all 96 physics
samples at 240 Hz over ``[t_td, t_td + 0.4 s)``, strictly harder than the 12-sample rule it
was meant to bound; ``docs/protocol.md`` P3-D3 records the change.

Config: ``configs/control/oracle_gated.yaml``.
"""

from pathlib import Path

import numpy as np

from rld.control.base import ControlSpec, PrivilegedContext
from rld.control.config import GatedConfig, load_gated
from rld.control.gated import GatedBase
from rld.control.obs_view import ObsView, deck_angles_deg

__all__ = ["OracleGated", "make_oracle_gated"]


class OracleGated(GatedBase):
    """``oracle_gated``: commit when the true future touchdown window is quiescent.

    **Privileged**: requires a :class:`PrivilegedContext` at every ``reset``.
    """

    name = "oracle_gated"
    privileged = True

    def __init__(self, cfg: GatedConfig, spec: ControlSpec) -> None:
        """Build the controller.

        Args:
            cfg: The gated config (same schema as ``gated``).
            spec: Committed environment configs.

        Raises:
            ValueError: If the physics rate is not an integer multiple of the control rate,
                so control-rate samples would not fall on the physics grid.
        """
        super().__init__(cfg, spec)
        landing = spec.landing
        if landing.physics_freq_hz % landing.ctrl_freq_hz != 0:
            raise ValueError(
                f"physics rate {landing.physics_freq_hz} Hz is not a multiple of the control "
                f"rate {landing.ctrl_freq_hz} Hz"
            )
        self._stride = int(landing.pyb_steps_per_ctrl)
        self._context: PrivilegedContext | None = None

    @property
    def window_samples(self) -> int:
        """Samples in the future window, dimensionless: ``gated``'s count (12 at 0.4 s, 30 Hz)."""
        return self.rule.n_samples

    @property
    def window_stride(self) -> int:
        """Physics samples between window samples, dimensionless (8: 1/30 s at 240 Hz)."""
        return self._stride

    def reset(self, seed: int, context: PrivilegedContext | None = None) -> None:
        """Clear per-episode state and take this episode's true trajectory.

        Args:
            seed: Episode seed; recorded only.
            context: The episode's true deck trajectory, from
                :meth:`PrivilegedContext.from_env` after ``env.reset``. Required.

        Raises:
            ValueError: If ``context`` is ``None``. The oracle refuses to run blind rather
                than degrade silently into a different controller.
        """
        if context is None:
            raise ValueError(
                "oracle_gated is privileged and needs a PrivilegedContext at every reset; "
                "build one with PrivilegedContext.from_env(env) after env.reset"
            )
        self._context = context
        super().reset(seed, None)

    def predicted_touchdown_s(self, v: ObsView, t_s: float) -> float:
        """Return the touchdown time a commit now would produce.

        Args:
            v: The observation, world frame.
            t_s: Episode time, seconds model scale.

        Returns:
            ``t + max(height, 0) / descent_rate``, seconds model scale since the episode
            start.
        """
        return t_s + max(v.height_m, 0.0) / self.pid.descent_rate_m_s

    def _rule_fires(self, v: ObsView, t_s: float) -> bool:
        """Return the shared rule's verdict on the true window at the predicted touchdown.

        Args:
            v: The observation, world frame.
            t_s: Episode time, seconds model scale.

        Returns:
            :meth:`QuiescenceRule.verdict` on the :attr:`window_samples` true deck states
            at ``t_td + j / ctrl_freq``; False when the window runs past the trajectory.

        Raises:
            RuntimeError: If no context was given (``reset`` was never called).
        """
        if self._context is None:
            raise RuntimeError("oracle_gated: act() before reset(context=...)")
        window = self._context.window(
            self.predicted_touchdown_s(v, t_s), self.rule.n_samples, stride=self._stride
        )
        if not window.complete:
            return False
        angles = np.array([deck_angles_deg(n) for n in window.normal], dtype=np.float64)
        return self.rule.verdict(angles[:, 0], angles[:, 1], window.pad_vz_m_s)


def make_oracle_gated(config_path: Path, spec: ControlSpec) -> OracleGated:
    """Registry factory.

    Args:
        config_path: ``configs/control/oracle_gated.yaml`` or an override.
        spec: Committed environment configs.

    Returns:
        A fresh controller; call ``reset(seed, context)`` before use.
    """
    return OracleGated(load_gated(config_path), spec)
