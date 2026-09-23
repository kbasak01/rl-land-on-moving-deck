"""Eval-side ground truth: was the TRUE deck quiescent at the moment the drone touched down?

The per-episode metric ``td_in_quiescent_window`` (Gate 3 remediation, results-skeptic M1)
answers one question for every episode that reached contact: did the **true** deck state
satisfy the permissive quiescence predicate over the 12 control-rate samples starting at the
touchdown time? It lets a table say whether a gated controller's landings -- or any other
method's -- actually happened in the window the rule is about, independently of what the
controller believed.

What is used, and what is not re-implemented
--------------------------------------------
* **The predicate** is :class:`rld.control.quiescence.QuiescenceRule`, imported, built
  exactly as :class:`rld.control.gated.GatedBase` builds it: dmf's ``permissive`` thresholds
  Froude-converted by :func:`rld.control.quiescence.limits_for`, at the control rate. With
  ``lam = 1/25`` and 30 Hz that is 12 samples, 1/30 s apart, ``|roll| <= 3.0 deg``,
  ``|pitch| <= 2.0 deg``, ``|v_pad_z| <= 0.16 m/s`` model scale. ``tests/test_eval_truth.py``
  asserts the rule equals ``gated``'s.
* **The true deck states** are the episode's trajectory rebuilt by
  :meth:`rld.control.base.PrivilegedContext.from_env` -- the same batched
  :func:`rld.envs.platform.build_trajectory` call on the environment's own physics grid, so
  the samples are the states the plate was driven through, bit for bit. This is
  **evaluation ground truth computed after the episode**; the runner builds it after the
  episode ends and never hands it to a non-privileged policy.
* **The samples** are those ``oracle_gated`` would read for the same start time:
  :meth:`PrivilegedContext.window` with ``stride = physics_freq / ctrl_freq`` (8), i.e. the
  first physics sample at or after ``t_td`` and every 8th after it. Roll and pitch come from
  the deck normal through :func:`rld.control.obs_view.deck_angles_deg`, as both gated
  controllers compute them; ``v_z`` is the pad deck-point world-``z`` velocity.

The touchdown time is the **contact** detector's (``ContactRecord.t_episode_s``), the one
the outcome is classified from. No contact: the value is NaN, never False.

Units: seconds model scale, degrees, metres per second model scale.
"""

import math

import numpy as np

from rld.control.base import PrivilegedContext
from rld.control.obs_view import deck_angles_deg
from rld.control.quiescence import QuiescenceRule, limits_for
from rld.eval.envs import EvalConfigs

__all__ = [
    "TD_QUIESCENCE_THRESHOLDS",
    "td_in_quiescent_window",
    "touchdown_rule",
]

#: dmf threshold set the touchdown audit applies (the gated controllers' committed set).
TD_QUIESCENCE_THRESHOLDS: str = "permissive"


def touchdown_rule(cfgs: EvalConfigs) -> QuiescenceRule:
    """Return the quiescence predicate the touchdown audit applies.

    Built as :class:`rld.control.gated.GatedBase` builds its rule, so the two are equal.

    Args:
        cfgs: The committed evaluation configs (Froude scale and control rate).

    Returns:
        ``QuiescenceRule(limits_for("permissive", scale), ctrl_freq_hz)``.
    """
    return QuiescenceRule(
        limits_for(TD_QUIESCENCE_THRESHOLDS, cfgs.scaling.froude_scale()),
        float(cfgs.landing.ctrl_freq_hz),
    )


def td_in_quiescent_window(
    context: PrivilegedContext,
    rule: QuiescenceRule,
    t_touchdown_s: float | None,
    stride: int,
) -> bool | float:
    """Return whether the true deck was quiescent over the window starting at touchdown.

    Args:
        context: The episode's true deck trajectory (evaluation ground truth).
        rule: The predicate, from :func:`touchdown_rule`.
        t_touchdown_s: Contact-detector touchdown time, seconds model scale since the
            episode start, or ``None``/NaN when the drone never touched down.
        stride: Physics samples between window samples, dimensionless
            (``physics_freq / ctrl_freq`` = 8, i.e. 1/30 s).

    Returns:
        ``rule.verdict`` on the ``rule.n_samples`` true deck states at
        ``t_td + j * stride * physics_dt``; NaN (a float) when there is no touchdown.

    Raises:
        RuntimeError: If the window runs past the end of the trajectory. A contact
            touchdown happens within the 12 s flight budget and the trajectory covers
            12.5 s, so this cannot happen on the committed configs; a silent False would
            mislabel the episode, so it is loud instead.
    """
    if t_touchdown_s is None or not math.isfinite(t_touchdown_s):
        return float("nan")
    window = context.window(float(t_touchdown_s), rule.n_samples, stride=stride)
    if not window.complete:
        raise RuntimeError(
            f"touchdown at {t_touchdown_s!r} s: the {rule.n_samples}-sample window runs past "
            f"the trajectory's end ({len(context)} samples)"
        )
    angles = np.array([deck_angles_deg(n) for n in window.normal], dtype=np.float64)
    return bool(rule.verdict(angles[:, 0], angles[:, 1], window.pad_vz_m_s))
