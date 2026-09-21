# protocol.md — decision log

Every decision, threshold, deviation and gate result, dated, in order. Where this file and
`IMPLEMENTATION_PLAN.md` disagree, **this file is what happened.** Entries are append-only:
a later entry may supersede an earlier one, never edit it.

Format: `### P<phase>-D<n> — <title> (YYYY-MM-DD)` then *Decision*, *Reason*, *Evidence*.

---

## Phase 0

### P0-D1 — Pinned dependencies (YYYY-MM-DD)
- deck-motion-forecast submodule SHA: `e9fa15cc35312a5c1ecf0c42668a690b5061c090`
- gym-pybullet-drones submodule SHA: `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`
- Resolved pins: gymnasium ____, stable-baselines3 ____, pybullet 3.2.7 (sdist build), rliable ____
- Env throughput (steps/s): 1 env ____ · 8 envs ____ · 16 envs ____ → `results/env_throughput.csv`

## Phase 1

### P1-D1 — Froude scale and pad position (YYYY-MM-DD)
- λ = 1/25 (default) — confirmed / changed to ____ because ____
- r_pad = [−0.4·L, 0, 0] full scale — confirmed / changed because ____
- Feasibility rule result: SS6 head-seas deck-point v_z p99 = ____ m/s vs 25 % of CF2X max speed ____

## Phase 3

### P3-D1 — FROZEN EVALUATION PROTOCOL (YYYY-MM-DD)
**Nothing below changes after the first Phase 5 training run without a dated deviation entry.**
- Success criteria: (copy of `configs/env/success.yaml`, with its SHA-256)
- Episode lists: N = 200 per (regime, SS); generator seed ____; SHA-256 of each file ____
- Metrics: (list)
- Statistics: rliable IQM + optimality gap, stratified bootstrap, reps ____; paired bootstrap for H1–H4
- Training budget: PPO ____ env steps; SAC ____; seeds 0–4; tuning ≤ 20 trials/method on `id` validation
- Residual α = ____; curriculum promotion threshold ____
- Hypotheses with predicted direction and magnitude:
  - H1 ____
  - H2 ____
  - H3 ____
  - H4 ____
  - H5 ____

## Gates
| gate | date | result | note |
|---|---|---|---|
