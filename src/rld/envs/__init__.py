"""Phase 2: the moving-platform landing environment.

Submodules are imported explicitly (``from rld.envs.landing_env import
DeckLandingAviary``), never re-exported here, exactly as :mod:`rld.deck` does it: importing
this package must not pull in PyBullet, because :mod:`rld.envs.config` is read by scripts
that never open a physics client.
"""

__all__: list[str] = []
