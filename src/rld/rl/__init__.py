"""Phase 5-6 learned policies: SB3 PPO and SAC on the landing environment.

Modules: :mod:`rld.rl.config` (YAML -> dataclasses), :mod:`rld.rl.wrappers` (pool sampling,
the per-worker env factory), :mod:`rld.rl.curriculum` (SS3 -> SS4 -> SS5), :mod:`rld.rl.callbacks`
(tune-pool evaluation, checkpoints with VecNormalize statistics, ``status.json``),
:mod:`rld.rl.train` (the training entry point and the evaluation-time loader),
:mod:`rld.rl.tuning` (the pre-registered hyperparameter search), :mod:`rld.rl.scheduler`
(the detached run queue behind ``make sweep`` / ``make tune``), :mod:`rld.rl.resume`
(resumable checkpoints and what a resumed run does and does not reproduce),
:mod:`rld.rl.reconcile` (marking runs and sweeps whose process died as failed /
interrupted) and :mod:`rld.rl.curves` (learning-curve aggregation).

Nothing in this package reads ``results/episodes/``: training draws from the P3-D2 train
pool and every in-training score comes from the P3-D2 tune pool.
"""
