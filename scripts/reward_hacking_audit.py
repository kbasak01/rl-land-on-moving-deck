r"""Phase 5 reward-hacking audit: thin wrapper around :func:`rld.rl.audit.main`.

Evaluation-side entry point. It checks every frozen list against ``MANIFEST.csv``, reads the
``id`` list and hands its rows to :mod:`rld.rl.audit`, which never opens a frozen list itself
(``tests/test_rl_leakage.py``). The audit reads the training monitors of
``artifacts/runs/{ppo,sac}/{0..4}`` and the committed ``results/e05`` / ``results/e01*``
episode tables, re-flies the pre-stated sample in scratch (substep logs to
``--scratch-dir``, never under ``results/``), writes the CSVs of ``results/audit/`` and
prints their SHA-256.

Usage::

    .venv/bin/python scripts/reward_hacking_audit.py --out-dir results/audit \
        --scratch-dir /tmp/rld_audit_logs --workers 24
"""

from rld.eval.episodes import EPISODES_DIR, read_list
from rld.eval.learned import verify_lists
from rld.rl.audit import main

if __name__ == "__main__":
    verify_lists(EPISODES_DIR)
    raise SystemExit(main(read_list(EPISODES_DIR / "id.parquet")))
