---
name: rl-experiment
description: How to configure, launch, monitor and log seeded RL training runs in this repo — SB3 PPO/SAC settings, VecNormalize, residual wrappers, curriculum, background launching with make train-bg, status files, budgets, and the reward-hacking audit. Use whenever starting, resuming, tuning or comparing training runs, or when a task mentions sweeps, seeds, hyperparameters or learning curves.
argument-hint: <config path, optional>
---

# RL experiment conventions

## Launching
- `make train-bg CFG=configs/rl/<method>.yaml SEED=<n>` → nohup process, logs to
  `artifacts/runs/<method>/<seed>/`, writes `status.json` {state, steps, fps, last_eval, started, pid}
  every eval interval. Never run training in the Claude Code foreground shell.
- Sweeps: `make sweep CFG=... SEEDS="0 1 2 3 4"`. Keep total parallel envs ≤ physical cores − 2.
- Resume only from a saved checkpoint + VecNormalize stats; record the resume in the run's status.

## Defaults (overridden only via P3-D1)
- PPO: MlpPolicy [256,256] tanh, n_envs 16, n_steps 1024, batch 4096, gamma 0.99, gae 0.95,
  clip 0.2, lr 3e-4 linear decay, device cpu. SAC: [256,256], buffer 1e6, batch 256, tau 0.005.
- `VecNormalize(norm_obs=True, norm_reward=True for PPO, clip_obs=10)`.
- Residual: action = base + α·π(o), α from config, actor last layer zero-initialised.
- Curriculum SS3→SS4→SS5, promote at ≥ 80 % success on training-validation episodes.

## Budgets and fairness
Equal env-step budget within each algorithm class, equal tuning trials across methods, tuning only
on `id` validation seeds. Record compute (wall-clock, CPU-hours) per run.

## Reward-hacking audit (after every method)
timeout rate · max contact penetration · analytic-vs-contact disagreement · success by initial-state
quintile · action saturation fraction. Write to `results/audit/<method>.csv`.
