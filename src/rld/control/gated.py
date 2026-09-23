"""``gated``: hover-track above the deck, commit to descent when the deck is quiescent **now**.

**Not privileged.** The rule reads the *current* deck state, which the observation carries
(roll and pitch from the deck normal, pad vertical velocity from
``velocity + rel_velocity``) -- what an ideal (noise-free, zero-latency) ship motion
reference unit plus state estimate would provide, like ``pid_feedforward``.

The law
-------
Lateral, feedforward and descent are ``pid_feedforward``'s, with **its tuned gains**
(``gains_from: pid_feedforward.yaml``); the gated controllers' own thresholds are dmf's and
are not tuned. The vertical channel has two modes, metres per second model scale, world
``z``, before the ``k_ff * v_pad_z`` feedforward:

* **hover** -- ``clip(hover_kp * (hover_height - height), +/-hover_speed_max)``: hold
  ``hover_height_m`` (0.3 m) of clearance above the pad. Starting 0.6-1.0 m above the mean
  deck, the drone first descends to it.
* **committed** -- ``-descent_rate``, as ``pid_feedforward``.

Commit happens at the first control step where **all** of these hold: the lateral gate is
open, the drone is within ``hover_tolerance_m`` of ``hover_height_m``, and the rule fires
(or ``fallback_commit_s`` has passed, if set). A commit is dropped, back to hover, only if
the lateral gate closes.

The rule (current state)
------------------------
Fires when ``|roll| <= R``, ``|pitch| <= P`` and ``|v_pad_z| <= V`` have held at every
control sample for the trailing ``sustain`` -- ``sustain_samples(sustain, 30 Hz)`` samples,
dmf's P6-D2 run-length convention. The predicate is
:class:`rld.control.quiescence.QuiescenceRule`, built once in :class:`GatedBase` at the
control rate and shared with ``oracle_gated``, which applies the **same** predicate to the
true future samples at the same spacing. So the two differ only in which samples they read:
"the deck has been quiet for ``sustain``; assume it stays quiet" against "the deck will be
quiet for ``sustain`` from touchdown". The history starts at the episode start; nothing
before it is visible. Roll and pitch come from the observed deck normal through
:func:`rld.control.obs_view.deck_angles_deg`; ``v_pad_z`` is world ``z`` of
``velocity + rel_velocity``.

Thresholds are dmf's ``PERMISSIVE`` (default) or ``STRICT``, imported, Froude-converted by
:mod:`rld.control.quiescence` (angles unchanged; 0.8 -> 0.16 m/s and 2.0 -> 0.4 s
permissive, 0.4 -> 0.08 m/s and 3.0 -> 0.6 s strict, at ``lam = 1/25``). **Deviation from
dmf:** the rate limit is applied to the **pad deck-point** ``v_z``, not dmf's CG heave
rate, because the pad is what the drone lands on.

With ``fallback_commit_s: null`` an episode whose deck never goes quiet is **not** rescued:
it times out, and the ``timeout`` fraction is reported as the cost of the rule.

Config: ``configs/control/gated.yaml``.
"""

from collections import deque
from pathlib import Path

import numpy as np

from rld.control.base import ControlSpec, PrivilegedContext
from rld.control.config import GatedConfig, load_gated
from rld.control.feedforward import PidFeedforward
from rld.control.obs_view import ObsView
from rld.control.quiescence import ModelQuiescenceLimits, QuiescenceRule, limits_for

__all__ = ["Gated", "GatedBase", "make_gated"]


class GatedBase(PidFeedforward):
    """Hover-and-commit machinery shared by ``gated`` and ``oracle_gated``.

    Subclasses implement :meth:`_rule_fires` and may override :meth:`_observe_deck`.

    Attributes:
        gated: The full config.
        limits: dmf's thresholds, Froude-converted to model scale.
        rule: The commit predicate, at the control rate. Shared by construction: both
            subclasses get it from here and only choose which samples to hand it.
    """

    name = "gated_base"
    privileged = False

    def __init__(self, cfg: GatedConfig, spec: ControlSpec) -> None:
        """Build the controller.

        Args:
            cfg: Hover, threshold and fallback settings plus ``pid_feedforward``'s gains.
            spec: Committed environment configs; ``spec.scale`` converts the thresholds.
        """
        super().__init__(cfg.gains, spec)
        self.gated = cfg
        self.limits: ModelQuiescenceLimits = limits_for(cfg.thresholds, spec.scale)
        self.rule = QuiescenceRule(self.limits, float(spec.landing.ctrl_freq_hz))
        self._committed = False
        self._commit_time_s: float | None = None

    def reset(self, seed: int, context: PrivilegedContext | None = None) -> None:
        """Clear every per-episode state, including the commit latch.

        Args:
            seed: Episode seed; recorded only.
            context: Ignored here; the oracle overrides this to require it.
        """
        del context
        super().reset(seed, None)
        self._committed = False
        self._commit_time_s = None

    @property
    def committed(self) -> bool:
        """Whether descent is currently committed."""
        return self._committed

    @property
    def commit_time_s(self) -> float | None:
        """Episode time of the most recent commit, seconds model scale, or ``None``."""
        return self._commit_time_s

    def _observe_deck(self, v: ObsView) -> None:
        """Update rule state from the current deck; called every control step.

        Args:
            v: The observation, world frame.
        """
        del v

    def _rule_fires(self, v: ObsView, t_s: float) -> bool:
        """Return whether the quiescence rule permits a commit now.

        Args:
            v: The observation, world frame.
            t_s: Episode time, seconds model scale.

        Returns:
            True to commit.
        """
        raise NotImplementedError

    def _vertical_m_s(self, v: ObsView, t_s: float) -> float:
        """Return the vertical command, hover or committed, before feedforward.

        Args:
            v: The observation, world frame.
            t_s: Episode time, seconds model scale.

        Returns:
            Metres per second model scale, world ``z``.
        """
        cfg = self.gated
        self._observe_deck(v)
        if self._committed and not self.lateral_ok:
            self._committed = False
        if not self._committed and self.lateral_ok:
            at_hover = abs(v.height_m - cfg.hover_height_m) <= cfg.hover_tolerance_m
            fallback = cfg.fallback_commit_s is not None and t_s >= cfg.fallback_commit_s
            if fallback or (at_hover and self._rule_fires(v, t_s)):
                self._committed = True
                self._commit_time_s = t_s
        if self._committed:
            return -self.pid.descent_rate_m_s
        correction = cfg.hover_kp_per_s * (cfg.hover_height_m - v.height_m)
        return float(np.clip(correction, -cfg.hover_speed_max_m_s, cfg.hover_speed_max_m_s))


class Gated(GatedBase):
    """``gated``: commit when the **current** deck has been quiescent for ``sustain``.

    Not privileged. See the module docstring.
    """

    name = "gated"
    privileged = False

    def __init__(self, cfg: GatedConfig, spec: ControlSpec) -> None:
        """Build the controller.

        Args:
            cfg: The gated config.
            spec: Committed environment configs.
        """
        super().__init__(cfg, spec)
        self._history: deque[tuple[float, float, float]] = deque(maxlen=self.rule.n_samples)

    def reset(self, seed: int, context: PrivilegedContext | None = None) -> None:
        """Clear every per-episode state, including the deck history.

        Args:
            seed: Episode seed; recorded only.
            context: Ignored; this controller is not privileged.
        """
        del context
        super().reset(seed, None)
        self._history.clear()

    @property
    def sustain_samples_needed(self) -> int:
        """Control samples the rule needs in a row, dimensionless (12 at 0.4 s, 30 Hz)."""
        return self.rule.n_samples

    def _observe_deck(self, v: ObsView) -> None:
        """Append the observed deck state to the trailing window.

        Args:
            v: The observation, world frame.
        """
        self._history.append((v.deck_roll_deg, v.deck_pitch_deg, float(v.deck_velocity_m_s[2])))

    def _rule_fires(self, v: ObsView, t_s: float) -> bool:
        """Return the shared rule's verdict on the trailing observed window.

        Args:
            v: Unused; the window was updated in :meth:`_observe_deck`.
            t_s: Unused.

        Returns:
            :meth:`QuiescenceRule.verdict` on the last :attr:`sustain_samples_needed`
            control samples, ending now; False while fewer have been observed.
        """
        del v, t_s
        if not self._history:
            return False
        window = np.asarray(self._history, dtype=np.float64)
        return self.rule.verdict(window[:, 0], window[:, 1], window[:, 2])


def make_gated(config_path: Path, spec: ControlSpec) -> Gated:
    """Registry factory.

    Args:
        config_path: ``configs/control/gated.yaml`` or an override.
        spec: Committed environment configs.

    Returns:
        A fresh controller; call ``reset`` before use.
    """
    return Gated(load_gated(config_path), spec)
