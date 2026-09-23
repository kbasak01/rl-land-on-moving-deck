"""Classical landing controllers: the baselines every learned method is printed beside.

Every controller implements :class:`rld.control.base.Controller` -- ``act(obs) -> action``
in the shared normalised action space ``[-1, 1]^3`` (a world-frame velocity setpoint scaled
by ``v_max``, P2-D2) and ``reset(seed, context=None)``. Gains and thresholds live in
``configs/control/<name>.yaml``; :mod:`rld.control.registry` maps names to factories,
config paths and the ``privileged`` flag.

Units: metres, metres per second and seconds are **model** scale unless a name says
``full``; angles are degrees at module boundaries and radians inside kinematics.
"""
