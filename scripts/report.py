"""Render results/results.md from the committed Phase 7 CSVs (``--check``: compare bytes).

Argparse wrapper only; the renderer is :func:`rld.eval.report.render_results`
(:mod:`rld.eval.results_md`). See ``python scripts/report.py --help``.
"""

import sys

from rld.eval.results_md import main

if __name__ == "__main__":
    sys.exit(main())
