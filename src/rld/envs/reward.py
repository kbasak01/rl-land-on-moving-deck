"""The pre-registered reward: a fixed structure, with weights in ``configs/env/reward.yaml``.

Pre-registration
----------------
The *structure* below is fixed by the implementation plan (Phase 2, task 6) and does not
change. Only the weights are tuned, and only on ``id`` **validation** realizations in
Phase 5, never on anything in ``results/episodes/``. Recording that split here rather than
in a training script is deliberate: the reward is part of the environment every method is
judged in, so a weight change is an environment change, not a hyperparameter.

The structure
-------------
Per control step::

    r_t = w_progress * (d_{t-1} - d_t)                       progress toward the pad centre
        - w_vz       * max(0, closing - v_safe) * [h < gate] closing speed, only near deck
        - w_smooth   * ||a_t - a_{t-1}||^2                   smoothness
        - w_time                                             per-step cost

and once, at the end::

    + r_success | - r_hard_landing | - r_off_pad | - r_bounce | - r_crash
    and EXACTLY ZERO for `timeout`.

Why ``timeout`` carries no terminal term
-----------------------------------------
``timeout`` is a **truncation** (``truncated=True``), not a termination. The value function
must bootstrap through it, and a terminal penalty attached to a truncation is exactly the
bug that teaches a policy the clock is a cliff -- or, with the sign the other way, that
hovering until the clock runs out is safe. The environment returns ``truncated=True`` and
this module returns 0.0, and both halves are tested.

Units and scales
----------------
``d`` is metres **model** scale (drone centre to pad centre), the closing speed is metres
per second model scale along the deck normal, ``h`` is the signed clearance in metres model
scale, and the action is the dimensionless ``[-1, 1]^3`` command. The weights are
multipliers on those, so the reward itself is dimensionless.
"""

import numpy as np
from dmf.typedefs import FloatArray

from rld.envs.config import RewardConfig
from rld.envs.touchdown import Outcome

__all__ = ["shaping_reward", "terminal_reward"]


def shaping_reward(
    *,
    cfg: RewardConfig,
    distance_prev_m: float,
    distance_now_m: float,
    closing_speed_m_s: float,
    clearance_m: float,
    action: FloatArray,
    prev_action: FloatArray,
) -> float:
    """Return one control step's dense reward.

    Args:
        cfg: The weights.
        distance_prev_m: Distance from the drone centre to the pad centre at the previous
            control step, metres model scale.
        distance_now_m: The same distance now, metres model scale.
        closing_speed_m_s: ``max(0, -(v_drone - v_pad) . n_deck)``, metres per second model
            scale. Non-negative by construction: moving away is not negative closing.
        clearance_m: Signed clearance from the drone's lowest point to the pad plane,
            metres model scale. The closing-speed penalty is gated on it, so approaching
            fast while still high costs nothing -- the policy is penalised for arriving
            fast, not for moving fast.
        action: This step's action, dimensionless, in ``[-1, 1]^3``.
        prev_action: The previous step's action, dimensionless.

    Returns:
        The dense reward for this step, dimensionless.
    """
    progress = cfg.w_progress * (distance_prev_m - distance_now_m)
    gated = 1.0 if clearance_m < cfg.gate_height_m else 0.0
    vz_penalty = cfg.w_vz * max(0.0, closing_speed_m_s - cfg.v_safe_m_s) * gated
    delta = np.asarray(action, dtype=np.float64) - np.asarray(prev_action, dtype=np.float64)
    smooth_penalty = cfg.w_smooth * float(np.dot(delta, delta))
    return float(progress - vz_penalty - smooth_penalty - cfg.w_time)


def terminal_reward(cfg: RewardConfig, outcome: Outcome) -> float:
    """Return the one-off reward for an episode's outcome.

    Args:
        cfg: The weights. Penalties are stored as positive magnitudes and negated here.
        outcome: The classified outcome.

    Returns:
        The terminal reward, dimensionless. Exactly ``0.0`` for ``timeout``.

    Raises:
        ValueError: If the outcome is not one of the six classes.
    """
    table: dict[Outcome, float] = {
        "success": +cfg.r_success,
        "hard_landing": -cfg.r_hard_landing,
        "off_pad": -cfg.r_off_pad,
        "bounce": -cfg.r_bounce,
        "crash": -cfg.r_crash,
        "timeout": 0.0,
    }
    if outcome not in table:
        raise ValueError(f"unknown outcome {outcome!r}")
    return table[outcome]
