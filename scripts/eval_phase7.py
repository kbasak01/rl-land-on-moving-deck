"""Phase 7: fly the frozen matrix and the shift and ablation arms -> results/e07/ (P7-D1).

Argparse wrapper only; the logic, the arms and the ``--check`` / ``--refly`` modes live in
:mod:`rld.eval.phase7` (arm definitions in :mod:`rld.eval.arms`, scoring in
:mod:`rld.eval.hypotheses`). See ``python scripts/eval_phase7.py --help``.

Usage::

    python scripts/eval_phase7.py --arm all --workers 24     # fly what is missing, then score
    python scripts/eval_phase7.py --arm all --check          # verify everything; write nothing
    python scripts/eval_phase7.py --arm sinusoid --refly sinusoid --refly-workers 7
"""

import sys

from rld.eval.phase7 import main

if __name__ == "__main__":
    sys.exit(main())
