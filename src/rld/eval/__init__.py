"""Evaluation harness: frozen episode lists, the runner, per-cell metrics and statistics.

Owner ``eval-auditor``. Read-only over :mod:`rld.control` and :mod:`rld.rl`: nothing here
changes a controller or a policy to change a number; it only measures them, identically,
on the committed episode lists under ``results/episodes/``.
"""
