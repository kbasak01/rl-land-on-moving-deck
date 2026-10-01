r"""Reward-hacking audit (Phases 5 and 6): thin wrapper around :func:`rld.rl.audit.main`.

Evaluation-side entry point. It checks every frozen list against ``MANIFEST.csv``, reads the
``id`` list and hands its rows to :mod:`rld.rl.audit`, which never opens a frozen list itself
(``tests/test_rl_leakage.py``).

* ``--phase 5`` (default): the training monitors of ``artifacts/runs/{ppo,sac}/{0..4}`` and
  the committed ``results/e05`` / ``results/e01*`` episode tables; CSVs to ``results/audit``.
* ``--phase 6``: ``artifacts/runs/{residual_ppo,ppo_forecast,residual_ppo_forecast,
  ppo_sinusoid}/{0..4}`` against ``results/e06`` (plus ``ppo``'s e05 rows and runs for the
  hard-landing comparison); CSVs to ``results/audit/e06``. Thresholds are pre-stated in
  ``results/audit/e06/README.md``. Phase 6 refuses to write into ``results/audit`` itself.

Both re-fly their pre-stated samples in scratch (substep logs to ``--scratch-dir``, never
under ``results/``), write the CSVs and print their SHA-256. Every CSV is independent of
``--scratch-dir``, ``--workers`` and ``--chunk``.

Usage::

    .venv/bin/python scripts/reward_hacking_audit.py --out-dir results/audit \
        --scratch-dir /tmp/rld_audit_logs --workers 24
    .venv/bin/python scripts/reward_hacking_audit.py --phase 6 --out-dir results/audit/e06 \
        --scratch-dir /tmp/rld_audit_logs_e06 --workers 24
"""

from rld.eval.episodes import EPISODES_DIR, read_list
from rld.eval.learned import verify_lists
from rld.rl.audit import main

if __name__ == "__main__":
    verify_lists(EPISODES_DIR)
    raise SystemExit(main(read_list(EPISODES_DIR / "id.parquet")))
