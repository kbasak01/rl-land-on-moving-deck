"""Environment-throughput measurement for Phase 0.

Separate from :mod:`rld.deploy`, which is the Phase 8 ONNX export, parity and inference
latency work built on ``dmf.deploy``. This package measures how fast the *simulator* runs,
which is what sizes the Phase 5 training budget; that one measures how fast a *policy*
runs, which is what H5 is scored on.
"""

from rld.bench.throughput import ThroughputResult, measure_throughput

__all__ = ["ThroughputResult", "measure_throughput"]
