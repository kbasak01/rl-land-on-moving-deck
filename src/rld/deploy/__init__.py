"""Phase 8: ONNX export, parity and latency of the learned policies (P8-D1).

* :mod:`rld.deploy.selection` -- which method and seed are exported (P8-D1 §1-§2);
* :mod:`rld.deploy.export` -- the actor with ``VecNormalize`` folded in, opset 18, dynamic batch;
* :mod:`rld.deploy.parity` -- numeric parity per (graph, provider), and the refusal rule;
* :mod:`rld.deploy.onnx_policy` -- the ONNX override of ``LearnedPolicy.network_action``;
* :mod:`rld.deploy.closed_loop` -- PyTorch vs ONNX on 50 frozen episodes;
* :mod:`rld.deploy.latency` -- a rank-2 mirror of Project 4's harness, one subprocess per
  configuration, and the H5 score;
* :mod:`rld.deploy.e2e` -- the per-step control budget inside full simulated episodes;
* :mod:`rld.deploy.pipeline` -- ``make bench`` and its ``--check``.

Everything is simulation. Latency is measured on a desktop RTX A4000 and i9-10980XE under WSL2;
no embedded target was measured.
"""
