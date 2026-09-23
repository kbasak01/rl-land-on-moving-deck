"""``pid_feedforward``: ``pid_track_descend`` plus deck-point velocity feedforward.

**The residual-RL base** (Phase 6: ``action = pid_feedforward + alpha * pi(o)``), so it is
kept clean and deterministic: no randomness, no hidden state beyond what ``reset`` clears.

**Not privileged relative to the observation** (``privileged=False``). The feedforward uses
the pad deck-point velocity ``v_pad = velocity + rel_velocity``, which is derivable from the
observation alone. Physically it is what **an ideal (noise-free, zero-latency) ship motion
reference unit plus state estimate** would provide: a ship's motion reference unit supplies
deck motion to a landing aid, and the vehicle estimates its own velocity. A real one adds
noise and latency that this observation does not model. It is not future information.

The law
-------
``pid_track_descend``'s setpoint (lateral PI; ``-descent_rate`` once the lateral gate is
open, 0 otherwise) plus ``k_ff * v_pad`` on **all three** axes, world frame, metres per
second model scale. So "hold height" now means "hold height *above the pad*", and a
committed descent closes on the pad at ``descent_rate`` rather than at
``descent_rate + v_pad_z``. What remains of the touchdown closing speed is tracking lag of
``DSLPIDControl`` and one control step of observation age against the deck's acceleration.

Gains: ``configs/control/pid_feedforward.yaml``; procedure and values in
``docs/protocol.md`` P3-D3.

``pid_feedforward_lowvz``
-------------------------
The **same class and the same law** with a different committed gain set,
``configs/control/pid_feedforward_lowvz.yaml``: the closing-speed reference for H1. Its
gains are the row of the existing ``pid_feedforward`` tuning log that
:func:`rld.control.tuning.select_lowvz` picks (trial 10). No trial and no tuning episode was
added to select it. It is registered under its own name so tables and ``controller.name``
tell the two apart. Not privileged.
"""

from pathlib import Path

from dmf.typedefs import FloatArray

from rld.control.base import ControlSpec
from rld.control.config import FeedforwardConfig, load_feedforward
from rld.control.obs_view import ObsView
from rld.control.pid import TrackDescendBase

__all__ = ["LOWVZ_NAME", "PidFeedforward", "make_pid_feedforward", "make_pid_feedforward_lowvz"]

#: Registry name of the low-closing-speed gain set of this same law.
LOWVZ_NAME = "pid_feedforward_lowvz"


class PidFeedforward(TrackDescendBase):
    """``pid_feedforward``: lateral PI + gated descent + ``k_ff * v_pad``. Not privileged.

    Attributes:
        ff: The full config, gains plus ``k_ff``.
    """

    name = "pid_feedforward"
    privileged = False

    def __init__(self, cfg: FeedforwardConfig, spec: ControlSpec, name: str | None = None) -> None:
        """Build the controller.

        Args:
            cfg: Gains and the feedforward gain.
            spec: Committed environment configs.
            name: Registry name, when this law is registered under another name with
                another gain set (``pid_feedforward_lowvz``); the class name otherwise.
        """
        super().__init__(cfg.pid, spec)
        self.ff = cfg
        if name is not None:
            self.name = name

    def _feedforward_m_s(self, v: ObsView) -> FloatArray:
        """Return ``k_ff * v_pad``.

        Args:
            v: The observation, world frame.

        Returns:
            ``(3,)`` metres per second model scale, world frame.
        """
        return self.ff.k_ff * v.deck_velocity_m_s


def make_pid_feedforward(config_path: Path, spec: ControlSpec) -> PidFeedforward:
    """Registry factory.

    Args:
        config_path: ``configs/control/pid_feedforward.yaml`` or an override.
        spec: Committed environment configs.

    Returns:
        A fresh controller; call ``reset`` before use.
    """
    return PidFeedforward(load_feedforward(config_path), spec)


def make_pid_feedforward_lowvz(config_path: Path, spec: ControlSpec) -> PidFeedforward:
    """Registry factory for ``pid_feedforward_lowvz``: this law, the low-closing-speed gains.

    Args:
        config_path: ``configs/control/pid_feedforward_lowvz.yaml`` or an override.
        spec: Committed environment configs.

    Returns:
        A fresh :class:`PidFeedforward` named ``pid_feedforward_lowvz``; call ``reset``
        before use.
    """
    return PidFeedforward(load_feedforward(config_path), spec, name=LOWVZ_NAME)
