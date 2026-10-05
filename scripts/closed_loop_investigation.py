"""P8-D3 closed-loop parity investigation -> results/latency/closed_loop_investigation/.

Argparse wrapper only; the arms, analyses and ``--check`` live in
:mod:`rld.deploy.investigation_run` and :mod:`rld.deploy.investigate`.
"""

import sys

from rld.deploy.investigation_run import main

if __name__ == "__main__":
    sys.exit(main())
