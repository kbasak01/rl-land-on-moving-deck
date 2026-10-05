"""ONNX export, parity, closed loop, latency, e2e budget and H5 -> results/latency/ (Phase 8).

Argparse wrapper only; the stages, arguments and ``--check`` mode live in
:mod:`rld.deploy.pipeline`. See ``python scripts/bench.py --help``.
"""

import sys

from rld.deploy.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
