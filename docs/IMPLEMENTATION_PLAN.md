# IMPLEMENTATION_PLAN.md — rl-land-on-moving-deck

Reinforcement-learning quadrotor landing on a heaving ship deck, driven by JONSWAP seakeeping
motion from `deck-motion-forecast` (Project 4), with a classical PID baseline, pure RL (PPO, SAC),
residual RL on top of the baseline, forecast-conditioned observations, and an ONNX policy
latency study. Single RTX A4000 + multi-core CPU. **Simulation only.**

This is the *original* plan. Once work starts, `docs/protocol.md` is the record of what actually
happened. Where they disagree, the protocol wins, exactly as in Project 4.

---

## 0. Design decisions that differ from the source document

The source document (Project 8 in *Novel, Single-GPU Engineering Projects…*) sketches a 9-day
project. Five decisions below change its scope; each is deliberate and each is written into the
README's limitations.

**D0.1 — The deck is Froude-scaled to the drone, not the drone to the deck.**
gym-pybullet-drones flies a Crazyflie 2.x (`cf2x`, ~27 g, ~92 mm). Putting it on a full-scale
frigate deck heaving metres at 10 s periods is not a landing problem, it is an absurdity. Deck
motion is therefore Froude-scaled by λ = L_model / L_full (default **λ = 1/25**, pre-registered):

| quantity | scale factor | λ = 1/25 |
|---|---|---|
| length, heave, lever arm | λ | ×0.04 |
| time, period | √λ | ×0.2 |
| linear velocity | √λ | ×0.2 |
| linear acceleration | 1 | ×1 |
| angle (roll, pitch) | 1 | ×1 |
| angular rate, frequency | 1/√λ | ×5 |

Scouting numbers (4 realizations/cell, frigate, 12 kn, pad 0.4·L aft of CG, **sign convention
unverified** — Phase 1 recomputes these properly on the full grid):

| sea state, heading | deck-point z std (model) | deck-point v_z std | v_z p99 |
|---|---|---|---|
| SS3, 180° | 1.4 cm | 0.085 m/s | 0.22 m/s |
| SS4, 180° | 3.9 cm | 0.20 m/s | 0.50 m/s |
| SS5, 180° | 7.7 cm | 0.36 m/s | 0.95 m/s |
| SS6, 180° | 13.2 cm | 0.50 m/s | 1.33 m/s |
| SS5, 90° (beam) | 3.2 cm | 0.10 m/s | 0.24 m/s |

This gives a real difficulty ladder against a 0.5 m/s touchdown limit. The vehicle itself is
*not* scaled (a real CF2X), so the claim is about the ratio of deck-motion bandwidth to vehicle
control bandwidth — which is reported in the README as a dimensionless number, not hidden.

**D0.2 — State-based observations, not vision.** The policy observes relative deck pose and
velocity (with a noise-and-latency model standing in for a vision estimator). Perception is out
of scope. The closest prior art (Angelis, Bauersfeld, Scaramuzza & Boukas, arXiv:2605.23717) is
*vision-based*; this project does **not** compete with it on perception and never claims to.

**D0.3 — The deck is 3-DOF.** `dmf` synthesises roll, pitch and heave only. Surge, sway and yaw
are absent. The deck-point trajectory is `p(t) = [0,0,heave] + R(roll,pitch)·r_pad`. Stated as a
limitation; optionally extended in §9.

**D0.4 — The project's novelty test is a pre-registered motion-realism experiment.**
The defensible gap versus prior art is *platform-motion realism*, so it is measured directly:
policies trained on **sinusoidal** platform motion (matched RMS and dominant period — the
Angelis-style motion model) are evaluated on **JONSWAP** motion, and vice versa (H4 below). If the
sinusoid-trained policy transfers fine, that is reported as the finding and the novelty claim is
withdrawn.

**D0.5 — Project 4 is consumed as-is, including its known defect.** `src/dmf/sim/response.py`
puts roll/pitch in phase with heave where strip theory puts them in quadrature (Project 4
Limitations). At an aft pad, deck-point vertical motion is `heave − x_pad·sin(pitch)`, so the
defect **changes the amplitude of the thing the drone lands on**. It is not fixed here (fixing it
would desynchronise the two projects). Instead: (a) a **pad-at-CG control arm** (x_pad = 0) removes
the lever-arm coupling and isolates the defect's effect; (b) an **optional MSS transfer arm**
evaluates on Project 4's strip-theory MSS records. Both are in Phase 7.

---

## 1. Deliverables and definition of done

1. `rld` Python package (`src/rld/`): deck-motion bridge, moving-platform PyBullet env, classical
   controllers, RL training, evaluation, ONNX deployment.
2. A frozen **evaluation protocol** (pre-registered in `docs/protocol.md` *before* any RL run).
3. Results for 7 methods × 4 regimes × 4 sea states, ≥ 5 training seeds for every learned method,
   with IQM and stratified-bootstrap 95 % CIs (rliable) and paired contrasts.
4. Success-rate-vs-sea-state curves; touchdown-velocity distributions; landing GIFs (PID vs pure
   RL vs residual RL at SS5).
5. ONNX-exported policies (observation normalisation folded into the graph), parity-checked,
   latency-benchmarked with Project 4's harness.
6. README in Project 4's register: every number traceable to a committed CSV; limitations first.
7. `make all` reproduces everything; `make test lint` green; every gate recorded.

**Done** = Gates 0–9 pass as written (or fail and are recorded as failed), `/full-audit` returns
no BLOCKING findings, README reviewed by `results-skeptic`.

---

## 2. Research questions and pre-registered hypotheses

Written into `docs/protocol.md` as P3-D1 **before Phase 5 starts**, with predicted directions and
magnitudes. Scored in Phase 7 regardless of outcome.

- **H1 (residual beats baseline).** Residual PPO ≥ PID+feedforward success rate on `id`, SS5,
  paired over identical episodes; predicted gain ≥ 10 points.
- **H2 (residual generalises better).** Under `unseen_seastate` (SS6), pure RL loses more success
  rate relative to its `id` value than residual RL does.
- **H3 (forecasts help, then stop helping).** Forecast-conditioned observations reduce p95
  touchdown relative vertical velocity on `id` SS5–SS6; the gain shrinks under `unseen_vessel`,
  where Project 4 showed forecaster skill degrades.
- **H4 (motion realism matters — the novelty test).** A policy trained on sinusoidal motion loses
  more success on JONSWAP than a JONSWAP-trained policy loses on sinusoids.
- **H5 (deployment).** At batch 1, the policy MLP runs faster on ORT CPU than on any GPU provider.

---

## 3. Repository layout

```
rl-land-on-moving-deck/
├── CLAUDE.md
├── .claude/{settings.json, agents/, skills/}
├── Makefile
├── pyproject.toml
├── third_party/
│   ├── deck-motion-forecast/     # git submodule, pinned commit, pip install -e
│   └── gym-pybullet-drones/      # git submodule, pinned commit, pip install -e
├── configs/
│   ├── deck/        scaling.yaml, pad.yaml, motion_{jonswap,sinusoid}.yaml
│   ├── env/         landing.yaml, observation.yaml, noise.yaml, success.yaml
│   ├── control/     pid_track_descend.yaml, pid_feedforward.yaml, gated.yaml, oracle.yaml
│   ├── rl/          ppo.yaml, sac.yaml, residual_ppo.yaml, curriculum.yaml
│   └── experiment/  e01_baselines.yaml … e07_shift.yaml
├── src/rld/
│   ├── deck/        bridge.py, scaling.py, kinematics.py, splits.py, sinusoid.py, forecast.py
│   ├── envs/        platform.py, landing_env.py, touchdown.py, observation.py, noise.py
│   ├── control/     base.py, pid.py, feedforward.py, gated.py, oracle.py, registry.py
│   ├── rl/          train.py, residual.py, wrappers.py, callbacks.py, curriculum.py
│   ├── eval/        episodes.py, metrics.py, stats.py, runner.py, report.py, gate.py
│   ├── deploy/      export_onnx.py, parity.py, bench.py   # thin wrappers over dmf.deploy
│   └── viz/         gifs.py, curves.py
├── scripts/         argparse wrappers only
├── tests/
├── results/         committed CSVs, figures, gate read-outs
├── artifacts/       gitignored: checkpoints, logs, tensorboard, ONNX, dmf corpus
└── docs/            IMPLEMENTATION_PLAN.md, protocol.md, findings.md, eval_protocol.md
```

Package name `rld`. Same architecture rules as `dmf`: logic in `src/`, scripts are wrappers,
YAML → dataclasses, no magic numbers.

---

## 4. Environment

- Python **3.12**, Ubuntu/WSL2, RTX A4000, same machine as Project 4.
- Reuse Project 4 pins where they overlap: `numpy==2.5.2`, `torch==2.13.0`, `onnx==1.22.0`,
  `onnxruntime-gpu==1.29.0`, `tensorrt-cu13==10.16.1.11`. Upstream gym-pybullet-drones (now at
  `learnsyslab/gym-pybullet-drones`, v2.2.0) declares `python ^3.12`, `torch ^2.13`, `numpy ^2.5`,
  `gymnasium ^1.3`, `stable-baselines3 ^2.9`, `pybullet ^3.2.7` — compatible.
- **Trap: pybullet 3.2.7 publishes no cp312 wheel**; it builds from sdist. Needs
  `build-essential` and `python3.12-dev`, takes several minutes. Phase 0 verifies the build.
- **Trap: `dmf` resolves `configs/sim/vessels` relative to its own source file**
  (`Path(__file__).parents[3]`). A non-editable `pip install git+…` puts it in site-packages where
  that path does not exist. Install dmf as a **git submodule + `pip install -e`**.
- Extra pins resolved in Phase 0 and recorded: `gymnasium`, `stable-baselines3`, `rliable`,
  `tensorboard`, `imageio[ffmpeg]`, `optuna` (optional).
- MLP policies train faster on CPU than GPU in SB3; set `device="cpu"` for PPO and measure SAC
  both ways once (Phase 5). The GPU is for the forecaster and the latency study.

---

## 5. Phase-by-phase build

Time estimates are working time, excluding unattended compute. Realistic total: **12–15 working
days + ~3–5 days of background compute** — longer than the source document's 9 days because of
the seeds and the pre-registration.

### Phase 0 — Bootstrap (0.5 day) · owner: main thread

Tasks
1. `git init`; add submodules for `deck-motion-forecast` (pin current `main` SHA) and
   `gym-pybullet-drones` (pin SHA). Record both SHAs in `docs/protocol.md` P0-D1.
2. `pyproject.toml` with pins above; `.venv`; `pip install -e third_party/deck-motion-forecast
   -e third_party/gym-pybullet-drones -e ".[dev]"`.
3. Copy the kit's `CLAUDE.md` and `.claude/` into place. `Makefile` with `PY/RUFF/MYPY/PYTEST`
   auto-detection copied from Project 4.
4. Smoke tests: `import dmf, gym_pybullet_drones, pybullet, stable_baselines3`;
   `simulate_realization` on one spec; `HoverAviary` 1 000 random steps headless;
   SB3 PPO trains `HoverAviary` for 20 k steps without error.
5. Measure env throughput: steps/s for 1, 8, 16 `SubprocVecEnv` workers (DIRECT mode). This number
   sets the Phase 5 budget.

**Gate 0**: `make test lint` green; pybullet imports on 3.12; env throughput recorded in
`results/env_throughput.csv`; both submodule SHAs recorded.

### Phase 1 — Deck-motion bridge (1.5 days) · owner: `deck-bridge-engineer`

Tasks
1. `deck/bridge.py`: `DeckMotionSource(spec: RealizationSpec, sim_cfg, t_model: ndarray)`.
   Reproduce `simulate_realization`'s seed path exactly — `realization_seed_sequence(spec).spawn(2)`,
   first child → `sample_components`, then `synthesize_motion` evaluated **directly on the
   full-scale time grid corresponding to the physics rate** (240 Hz model → 48 Hz full-scale at
   λ = 1/25). No interpolation, no finite differences: rates come from `synthesize_motion`.
2. `deck/scaling.py`: Froude scaling as a pure function with the table in D0.1; units in every
   docstring; angles stored in degrees at the boundary, radians inside kinematics.
3. `deck/kinematics.py`: deck-point pose, linear velocity and angular velocity from
   (roll, pitch, heave, rates) and `r_pad`. **Verify the sign convention against
   `docs/corpus_card.md` of dmf** (roll positive to starboard, pitch bow-up) and write it down.
4. `deck/splits.py`: build the four dmf regimes (`id`, `unseen_seastate`=SS6, `unseen_heading`=90°,
   `unseen_vessel`=S175) from `realization_grid(cfg)` metadata via `dmf.data.splits.build_split`,
   **without generating the corpus**. Split unit = realization seed. Episodes are start-time
   offsets inside a realization; a realization never contributes to two splits.
5. `deck/sinusoid.py`: Angelis-style sinusoidal motion, per-DOF amplitude matched to the JONSWAP
   realization's RMS and period matched to its peak encounter period. Same interface.
6. Phase-defect characterisation: for each sea state, report deck-point v_z RMS at the aft pad and
   at CG. Recorded, not fixed.

Tests (`tests/test_bridge.py`, `test_scaling.py`, `test_kinematics.py`, `test_deck_splits.py`)
- **Parity**: bridge evaluated at 10 Hz full-scale reproduces `simulate_realization` clean columns
  to `max_abs_err ≤ 1e-4 · max(1, |x|max)` (float32 corpus) on ≥ 20 specs across all cells.
- Scaling invariants: angles unchanged; `v/√(gL)` (Froude number) unchanged; periods × √λ.
- Kinematics: finite-difference of deck-point position matches analytic velocity (test-only FD).
- Split disjointness on (vessel, ss, heading, speed, seed); S175 appears only in `unseen_vessel`.
- Sinusoid matches target RMS within 2 %.

**Gate 1**: all tests pass; `results/deck_stats.csv` (full grid: z std, v_z std/p99, a_z p99 per
cell, aft pad and CG) committed; λ and `r_pad` confirmed or changed **with a recorded reason**
(P1-D*). Feasibility rule, pre-registered: SS6 head-seas deck-point v_z p99 must be ≤ 25 % of the
CF2X's commanded max speed in gym-pybullet-drones; if not, λ is reduced and recorded.

### Phase 2 — Moving-platform landing environment (2 days) · owner: `sim-env-engineer`

Tasks
1. `envs/platform.py`: the deck as a PyBullet body driven along the bridge trajectory.
   **Do not teleport it with `resetBasePositionAndOrientation` every step** — contact then sees zero
   platform velocity and touchdown impacts are wrong. Use a dynamic body with a `JOINT_FIXED`
   constraint to world, moved each physics step with `changeConstraint(pivot, orientation,
   maxForce=large)`, **or** mass-0 body plus explicit `resetBaseVelocity` — whichever passes the
   tracking and velocity tests below. Record the choice (P2-D1).
2. `envs/landing_env.py`: subclass of gym-pybullet-drones `BaseRLAviary` (or `BaseAviary` with a
   Gymnasium wrapper). Physics 240 Hz, control 30 Hz (pre-registered). Episode ≤ 12 s model time.
   Initial state: altitude 0.6–1.0 m above mean deck, lateral ±0.3 m, zero velocity, randomised
   episode start time within the realization's 600 s (full-scale) record.
3. **Action space** (shared by every method, so the comparison is fair): world-frame velocity
   setpoint `[vx, vy, vz]` ∈ [−1, 1]³ scaled to `v_max`, yaw held at 0, tracked by gym-pybullet-drones'
   `DSLPIDControl`. Residual RL adds a bounded correction `α·a_rl` to the baseline's setpoint.
4. **Observation**: drone state (attitude, body rates, velocity), relative deck-pad position and
   velocity in the drone frame, deck normal, time-since-start; optional forecast block (Phase 4).
   `envs/noise.py`: Gaussian noise + fixed latency + 30 Hz sample-and-hold on relative pose — the
   vision stand-in. Noise off by default for training; ablated in Phase 7.
5. `envs/touchdown.py`: analytic touchdown detection from recorded states **and** contact-point
   detection from PyBullet; both logged, disagreement counted. Outcome classes: `success`,
   `hard_landing`, `off_pad`, `bounce`, `crash`, `timeout`.
6. Reward (pre-registered structure, weights tuned only on `id` validation seeds): progress toward
   pad, penalty on relative vertical velocity near the deck, smoothness penalty, terminal bonus on
   `success`, terminal penalty on `crash/off_pad/hard_landing`. **`timeout` is a truncation, not a
   termination** (Gymnasium `truncated=True`) so value bootstrapping is correct.

Success criteria (pre-registered in `configs/env/success.yaml`, frozen at Gate 3):
- relative vertical velocity at first contact ≤ **0.5 m/s**,
- lateral offset from pad centre ≤ **0.10 m** (pad radius 0.15 m),
- relative tilt drone-vs-deck ≤ **15°**,
- remains in contact, on pad, for ≥ **0.5 s** after touchdown.

Tests
- Platform tracking error ≤ 1 mm and platform `getBaseVelocity` within 2 % of analytic deck-point
  velocity, over 10 s at SS6.
- `gymnasium.utils.env_checker.check_env` passes.
- Deterministic reset: same seed → bit-identical first 100 observations.
- A drone dropped from rest onto a static pad registers exactly one touchdown and classifies it.
- Scripted "descend at 0.3 m/s onto static pad" → 100/100 `success`.

**Gate 2**: tests pass; random-policy and hover-policy outcome distributions recorded
(`results/e00_env_sanity.csv`); PyBullet-vs-analytic touchdown disagreement rate < 1 %.

### Phase 3 — Classical baselines and evaluation protocol freeze (1.5 days) · owners: `controls-engineer`, `eval-auditor`

Controllers (`src/rld/control/`, one file each, registry entry each):
1. `pid_track_descend`: track pad x/y, descend at constant rate. The naive baseline.
2. `pid_feedforward`: as above plus deck-point velocity feedforward (privileged but realistic —
   a ship's motion reference unit provides it). **This is the baseline residual RL builds on.**
3. `gated`: hover-track at 0.3 m above deck, commit to descent when a quiescence rule fires
   (initially on true current deck state; forecaster version in Phase 4).
4. `oracle_gated`: gating with access to the **true future** deck motion — an upper bound on
   timing, never a deployable result.

Evaluation harness (`src/rld/eval/`, owner `eval-auditor`)
- Episode lists generated once from a fixed seed, written to `results/episodes/<regime>.parquet`
  (realization key + start offset + initial state). **Every method is evaluated on the identical
  episode list** → paired comparisons.
- N = 200 episodes per (regime, sea state) cell; seeds disjoint from any training seed.
- Metrics: success rate with Wilson 95 % CI; outcome-class breakdown; touchdown relative v_z
  (p50/p95), lateral error, tilt; time-to-touchdown; control effort; action jerk.

**Protocol freeze (P3-D1)** — before Gate 3 closes, write into `docs/protocol.md`: success
criteria, episode lists (hash), N, metrics, statistical tests (rliable IQM + stratified bootstrap;
paired bootstrap for H1–H4), training budget per method, number of seeds, tuning budget, and
H1–H5 with predicted magnitudes. **Nothing in this block changes after RL training starts without
a recorded, dated deviation.**

**Gate 3**: `pid_feedforward` ≥ 95 % success on static pad and on `id` SS3; success-vs-sea-state
table for all four controllers committed (`results/e01/`); protocol P3-D1 committed and hashed.
`results-skeptic` review folded in.

### Phase 4 — Forecaster integration (1.5 days + ~1 h compute) · owner: `deck-bridge-engineer`

1. Regenerate the dmf corpus (`make -C third_party/deck-motion-forecast data`, ~15 min) and fit
   `dlinear_ols` (closed-form; best cross-generator transfer in Project 4) and `tcn` (best `id`
   skill; worst transfer) on the `id` regime. Checkpoints go to `artifacts/dmf/`.
2. `deck/forecast.py`: online adapter. The forecaster wants a 200-sample, 10 Hz **full-scale**
   lookback (20 s full = 4 s model at λ = 1/25) of the 6 clean channels. Maintain a ring buffer
   subsampled from the bridge; feed history from before the episode start (the ship was moving
   before the drone arrived). Output: predicted deck-point z and v_z at leads 1, 2, 3 s full-scale
   (0.2, 0.4, 0.6 s model), converted through kinematics and Froude scaling.
3. Forecast block for the RL observation (normalised), and a `gated_forecast` controller using
   Project 4's **interval rule** (whole 90 % band inside limits), with limits Froude-scaled.
4. Latency budget: the forecaster runs at 10 Hz full-scale = 50 Hz model — faster than the 30 Hz
   control loop. Run it at the control rate on CPU (ORT) and record its cost per step.

Tests: online adapter reproduces dmf's offline predictions on identical windows (parity
`1e-4 · max(1,|y|)`); the adapter never reads samples after the current time (a causality test
that shifts the future and asserts identical output).

**Gate 4**: parity + causality pass; `gated_forecast` results on the Phase 3 episode lists
committed alongside `gated` and `oracle_gated`.

### Phase 5 — Pure RL: PPO and SAC (2 days + ~1–2 days compute) · owner: `rl-trainer`

1. `rl/train.py`: SB3 PPO and SAC; `SubprocVecEnv`; `VecNormalize` on observations (reward
   normalisation for PPO only); **statistics saved with every checkpoint and frozen at eval.**
2. Curriculum (`rl/curriculum.py`): SS3 → SS4 → SS5 by success-rate threshold on a held-out
   training-validation set; SS6 is never trained on (it is `unseen_seastate`).
3. Budget (pre-registered in P3-D1, sized from Phase 0 throughput): same env steps for every
   method; default 10 M for PPO, 2 M for SAC (off-policy), 5 training seeds each. Hyperparameter
   tuning: ≤ 20 trials per method on `id`-validation seeds only, identical budget for every method.
4. Runs launch **in the background** (`make train-bg`), write TensorBoard and a JSON status file;
   Claude Code monitors via `/sweep-status` rather than blocking a shell.
5. Reward-hacking audit after each method: fraction of `timeout` episodes (hovering to avoid
   penalties), penetration depth at contact, touchdowns classified differently by PyBullet vs
   analytic detection, success achieved by exploiting episode-start states.

**Gate 5**: learning curves (5 seeds, mean ± std) committed; `id` results on the frozen episode
list for PPO and SAC; hacking audit clean or findings recorded; seed-to-seed spread reported.

### Phase 6 — Residual RL and forecast-conditioned RL (1.5 days + ~1 day compute) · owner: `rl-trainer`

Methods (all 5 seeds, same budget as PPO):
1. `residual_ppo`: action = `pid_feedforward` setpoint + α·π(o), α pre-registered (default 0.3·v_max);
   policy initialised so that α·π ≈ 0 at start (zero-initialised last layer).
2. `ppo_forecast`: pure PPO with the Phase 4 forecast block in the observation.
3. `residual_ppo_forecast`: both.
4. `ppo_sinusoid`: pure PPO trained on sinusoidal motion (for H4).

**Gate 6**: all runs complete; `id` results committed; residual policies verified to reduce to the
baseline when the residual is zeroed (test).

### Phase 7 — Evaluation under shift and ablations (2 days + ~0.5 day compute) · owner: `eval-auditor`

Full matrix on frozen episode lists: 4 regimes × 4 sea states × all methods × all seeds.
Additional arms:
- **H4 realism cross**: {JONSWAP-trained, sinusoid-trained} × {JONSWAP, sinusoid} test motion.
- **Perception stand-in**: noise σ ∈ {0, 1, 2, 4} cm and latency ∈ {0, 33, 66} ms on relative pose.
- **Pad-at-CG control** (defect isolation, D0.5).
- **MSS transfer (optional, needs Octave)**: episodes on Project 4's MSS strip-theory records.
- **λ sensitivity**: λ ∈ {1/15, 1/25, 1/40} on the best two methods only.

Statistics (`eval/stats.py`): rliable IQM and optimality-gap with stratified bootstrap 95 % CIs
across seeds; paired bootstrap on per-episode success differences for H1–H4; Wilson CIs per cell.
**Every hypothesis gets a verdict line — supported, not supported, or inconclusive — with the
number.** No pooling of success rate across sea states in any headline.

**Gate 7**: `results/results.md` rendered from CSVs byte-reproducibly; H1–H5 scored in
`docs/findings.md`; `results-skeptic` review with no BLOCKING findings.

### Phase 8 — ONNX export and latency (0.5–1 day) · owner: `deploy-benchmarker`

1. Export the actor network with `VecNormalize` mean/var **folded into the graph** (and the
   residual's baseline stays outside — document the boundary). Opset 18, dynamic batch only.
2. Parity on every benchmarked provider: 1 000 random observations × 5 draws, scale-relative
   `1e-4`; TF32 disabled. Also: closed-loop parity — 50 episodes with the ONNX policy produce the
   same outcome classes as PyTorch.
3. Latency via Project 4's harness (`dmf.deploy.harness`, `providers.preload_gpu_libraries`): 200
   warmup + 2 000 timed, p50/p90/p99, batch 1 and 32, one CPU thread and a thread sweep.
4. End-to-end control-step budget: policy + forecaster + observation build, against the 33 ms
   control period.

**Gate 8**: parity passes on every timed provider (refused rows never timed); `results/latency*`
committed; H5 scored.

### Phase 9 — Documentation, figures, release (1 day) · owner: main thread + `results-skeptic`

README (Project 4 structure: headline figure → results tables with baselines → hypotheses →
limitations → reproduce), landing GIFs, `docs/findings.md`, `docs/protocol.md`, citations,
`THIRD_PARTY_NOTICES.md` (gym-pybullet-drones MIT, dmf MIT). `/full-audit` run last.

**Gate 9**: validation protocol (§7) all checked; no BLOCKING audit findings; `make all` dry-run
lists every stage; every README number traced to a committed file.

---

## 6. Evaluation protocol summary

Maintained in full in the `landing-protocol` skill and frozen in P3-D1. Headline table per regime:

| method | params | success % (IQM, 95 % CI) SS3/SS4/SS5/SS6 | p95 touchdown v_z | crash % | timeout % |

Always printed beside learned methods: `pid_track_descend`, `pid_feedforward`, `oracle_gated`.
Success without the outcome-class breakdown is never reported.

---

## 7. Validation protocol (run before calling it finished)

- [ ] Splits: no realization key in both a training and an evaluation set (test re-run on final configs).
- [ ] Episode lists: hash in results matches P3-D1 hash.
- [ ] Every learned method: 5 seeds present for every cell; no silently dropped seed.
- [ ] VecNormalize stats loaded frozen at eval (test asserts `training=False`, `norm_reward=False`).
- [ ] Reward-hacking audit re-run on final checkpoints.
- [ ] Touchdown analytic-vs-contact disagreement < 1 % on the final matrix.
- [ ] Bridge parity test green against the pinned dmf SHA.
- [ ] Forecaster causality test green.
- [ ] ONNX parity on every timed provider; closed-loop parity 50/50.
- [ ] Every README number greps to a committed CSV; `make report` re-renders `results.md` byte-identically.
- [ ] Every deviation from P3-D1 dated and justified in `docs/protocol.md`.
- [ ] README states: simulation only; 3-DOF deck; Froude-scaled; state-based; dmf phase defect.

---

## 8. Risk register

| risk | likelihood | mitigation |
|---|---|---|
| pybullet fails to build on 3.12 | low–med | build-essential + python3.12-dev; fallback: a 3.11 venv for the sim only is **not** acceptable (breaks dmf pins) — fix the build |
| Platform contact dynamics wrong | med | constraint-driven body; velocity-match test; analytic touchdown as ground truth |
| RL fails to learn SS5 | med | curriculum; residual RL still delivers; report honestly |
| Reward hacking (hover to timeout) | high | truncation semantics; timeout rate in every table |
| RL seed variance swamps differences | high | 5 seeds, IQM + CIs, paired episodes; claims only where CIs separate |
| dmf phase defect dominates results | med | pad-at-CG control; MSS arm |
| Compute overrun | med | Phase 0 throughput sizes budget; SAC budget smaller; λ sweep only on 2 methods |
| Forecaster adds latency / leaks future | low | causality test; per-step cost measured |
| H4 comes out null | med | that is a finding; README says so and drops the novelty claim |

---

## 9. Extension path toward a publishable contribution

1. Add sway/yaw with a directional-spreading sea (needs a dmf extension) → full 6-DOF deck.
2. Replace the noise stand-in with Project 1's pose regressor in the loop (rendered camera).
3. Conformal-gated commit: use Project 4's calibrated intervals as a safety filter, and measure the
   documented coverage collapse under shift *as a landing-failure rate*.
4. Head-to-head against a linear MPC from safe-control-gym on the same episodes.
