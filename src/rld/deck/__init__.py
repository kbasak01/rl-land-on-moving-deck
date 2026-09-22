"""Deck-motion bridge: dmf seakeeping motion, Froude-scaled to model-scale deck points.

Phase 1. Everything downstream -- the PyBullet platform body (Phase 2), the feedforward
controllers (Phase 3), the forecaster adapter (Phase 4) -- consumes this package's units
and frame conventions, so they are stated in each module's docstring and tested rather than
described.
"""

__all__: list[str] = []
