"""Reinforcement-learning quadrotor landing on a Froude-scaled heaving ship deck.

Everything in this package is **simulation only**. No deck motion, no vehicle state and no
result produced here is derived from, or validated against, real ship-deck measurements or
real flight. Deck motion comes from ``deck-motion-forecast`` (``dmf``), which synthesises
JONSWAP-driven roll, pitch and heave; the vehicle is gym-pybullet-drones' Crazyflie 2.x.

This top-level module deliberately imports nothing heavyweight (no torch, no pybullet, no
stable-baselines3) so that ``import rld`` stays cheap and can serve as a bootstrap smoke
check.

Unit and scale conventions, stated once here and repeated in every docstring that touches
them:

- **Scale.** The deck is Froude-scaled to the drone by ``lambda = L_model / L_full``
  (default 1/25). Every time and length quantity is named *model-scale* or *full-scale*;
  a bare number is a bug. Lengths scale by ``lambda``, times and linear velocities by
  ``sqrt(lambda)``, linear accelerations and angles by 1, angular rates by ``1/sqrt(lambda)``.
- **Angles** are **degrees** at every public boundary (YAML configs, dmf columns, public
  function signatures) and are converted to radians only inside kinematics and physics.
- **Angular rates** are **degrees per second** at public boundaries, **radians per second**
  inside physics.
- Linear displacements are **metres**, linear rates **metres per second**, linear
  accelerations **metres per second squared**.
- Times and periods are **seconds**. Sampling and control rates are **hertz** and are the
  only Hz quantities; wave and response frequencies are radians per second.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
