"""Evaluate learned runs' checkpoints on the frozen episode lists -> results/e05/ (Phase 5).

Argparse wrapper only; the logic, arguments and ``--render-only`` / ``--check`` modes live in
:mod:`rld.eval.learned`. See ``python scripts/eval_learned.py --help``.
"""

import sys

from rld.eval.learned import main

if __name__ == "__main__":
    sys.exit(main())
