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

**Before you start (added after Gate 0, 2026-09-21).** Read `docs/protocol.md` P0-D1 and P0-D2.
Three things Phase 0 settled that this phase depends on: (a) `dmf` is an **editable** install at
SHA `e9fa15c` and `simulate_realization` is verified to run from it — that SHA is what the Phase 1
parity test is written against, and `tests/test_smoke.py` already contains a working call to copy;
(b) the environment is exactly pinned, so a parity tolerance that only holds on one numpy version
is a real result, not a nuisance; (c) the Phase 0 throughput number is `HoverAviary`-only and is a
**ceiling** — Phase 2 must re-measure with the deck body and the bridge in the loop before P3-D1
fixes any budget. Nothing in Phase 0 changes Phase 1's tasks or Gate 1.

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

**Before you start (added after Gate 1, 2026-09-21).** Read `docs/protocol.md` P1-D1, P1-D2 and
P1-D3. Five things Phase 1 measured that change this phase:

(a) **The action space cannot reuse `ActionType.VEL` verbatim.** `BaseRLAviary.py:95` caps commanded
speed at `SPEED_LIMIT` = 0.25 m/s, which is *below* the deck's own aft-pad `v_z` p99 at every sea
state from SS3 up (0.166 / 0.351 / 0.575 / 0.577 m/s at 180 deg, 12 kn). Set `v_max` as a project
parameter, >= 1.0 m/s, call `DSLPIDControl` directly, and record the value as P2-D*.

(b) **The frame and sign convention are fixed and tested — do not re-derive them.** World frame is
x = bow, y = **port**, z = up; hand `pybullet.getQuaternionFromEuler([+radians(roll_deg),
-radians(pitch_deg), 0])`, which is exactly `rld.deck.kinematics.euler_xyz_rad`. Composition is ZYX;
a `pybullet`-marked test already asserts our rotation matrix equals PyBullet's to 1e-12. Note the
**plan's own D0.5 and D0.1 are superseded by P1-D2** on this point, as is the `deck-scaling-physics`
skill's older text.

(c) **`rld.deck.bridge.JonswapDeckMotion` is the interface**, model scale, world frame: `deck_point(
t_model_s, pad)` -> position/velocity/acceleration/euler/omega/alpha/normal/tilt, and
`deck_points(t_model_s, pads)` for both pads from one channel evaluation (use it for the pad-at-CG
arm so both come from bit-identical motion). Model time 0 == full-scale absolute 120.0 s;
`t_model_window_s == (4.0, 120.0)`; `episode_start_window_s(12.0) == (4.0, 108.0)`. Times outside the
committed record raise.

(d) **The deck alone exceeds the 15 deg tilt criterion in 10 of 96 cells** (max 24.49 deg, frigate
SS6 90 deg), dominated by roll in beam and stern-quartering SS6. The success criterion must be on
*relative* tilt, and `configs/env/success.yaml` should say so explicitly.

(e) **Re-measure throughput with the deck body and the bridge in the loop before P3-D1.** P0-D2's
number is `HoverAviary`-only and is a ceiling. The bridge costs ~0.28 s per 28 800-sample
realization evaluation, so the per-step cost of evaluating deck motion is not free.

Sanity anchors for the platform tracking test: at lambda = 1/25 the frigate's aft pad sits
1.984 m aft of the deck reference point, `z` std is 4.43 cm at SS5 180 deg 12 kn and peak `|v_z|` is
~0.67 m/s at SS6; the CG pad is *exactly* pure heave in position, velocity and acceleration.

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

**Before you start (added after Gate 2, 2026-09-22).** Read `docs/protocol.md` P2-D1 through P2-D9.
Six things Phase 2 fixed or measured that change this phase:

(a) **The action space is settled and shared — do not invent a second one.** `Box(-1, 1, (3,))`,
world-frame velocity setpoint, `v_max` = **1.5 m/s** model scale, yaw commanded to **0**, and
`v_max` is a cap on the **speed**, not per axis (P2-D2). Every controller emits a setpoint through
`env.velocity_setpoint_m_s(action)` so PID, PPO and residual scale identically; that is what makes
the comparison fair.

(b) **A pure vertical descent is not a landing in this environment.** Episodes start with a lateral
spread of ±0.3 m against a plate half-width of 0.26 m, so roughly 13 % begin laterally outside the
deck footprint. `pid_track_descend` must null lateral error before committing to descent or it will
score badly for a geometric reason rather than a control one (P2-D3).

(c) **The success criteria live in `configs/env/success.yaml` and carry three frozen resolutions**
that the plan and the `landing-protocol` skill left ambiguous (P2-D5): `hard_landing` absorbs a
relative-tilt violation, the vertical-velocity criterion is the component **along the deck normal**
(both readings are recorded per episode), and divergence is `crash` with a `termination_reason`
column. P3-D1 must copy that file **and its SHA-256**, not restate the numbers.

(d) **Size the P3-D1 training budget from `results/env_throughput_landing.csv`, not from P0-D2.**
P0-D2's 7 575 steps/s is a `HoverAviary` ceiling. With the deck body and the bridge in the loop the
landing env runs 514 steps/s at 16 workers under a random policy and 4 610 steps/s with full-length
episodes — a **10.4×** spread, because a reset costs 88.5 ms of which 73 % is the analytic bridge
call and one worker's reset stalls a synchronous vector step. The budget is bracketed by those two
rows; quoting either alone is wrong (P2-D8).

(e) **The pad's standing lever arm is removed inside the environment** (P2-D3). `DeckPointState.
position_m` is measured from the vessel's mean-position CG, so the frigate's aft pad carries a
constant −1.984 m world-x offset (−2.800 m on the s175); the env subtracts it, so controllers see a
pad oscillating about `deck_origin_m` with identical world geometry for both pads and both hulls.
The lever-arm *motion* is untouched and asserted non-zero.

(f) **Episode geometry.** 12.0 s flight budget + 0.5 s dwell grace, so a touchdown at 11.9 s can
finish its 0.5 s dwell; start offsets come from `episode_start_window_s(12.5) == (4.0, 107.5)`
(P2-D7).

One caveat on what Gate 2 does and does not certify: only **1 of 400** random-policy episodes ever
reached contact, so the sanity sweep barely exercises the touchdown path — the scripted tests
(100/100 static descents, 0/30 disagreement on a moving deck) carry that evidence. Phase 3's
baselines are the first methods that will exercise it at volume, and that single random touchdown
was also the one episode that tunnelled (7.86 mm against a 5 mm threshold). Watch `tunnelling_n`
and the analytic-vs-contact disagreement rate in the Phase 3 tables.


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

**Before you start (added after Gate 3, 2026-09-22).** Read `docs/protocol.md` P3-D1 (FROZEN
revision 1), P3-D2 and P3-D4. Four things Phase 3 fixed that change this phase:

(a) **Task 1's "fit on the `id` regime" is superseded by P3-D2, and following it would leak.**
- *The leak.* dmf's `id` train partition (seeds 0–26, all four sea states, all four headings)
  contains SS6 and 90° realizations. Those are `unseen_seastate`-test and `unseen_heading`-test
  realizations, i.e. frozen evaluation episodes.
- *The fix.* Fit `dlinear_ols` and `tcn` **only** on `rld.deck.splits.dev_pool(sim_cfg)[0]`
  (729 realizations: frigate, SS3–SS5, 45/135/180°, seeds 0–26). Use `dev_pool()[1]` for any
  forecaster model selection.
- *What to record.* The pool restriction and the resulting training-set size, as P4-D*.
- *The test.* Assert that the fitted models' training keys are disjoint from every
  `results/episodes/` list.

(b) **`gated_forecast` is scored on the committed Phase 3 lists, beside `gated` and
`oracle_gated`.**
- *The lists.* `results/episodes/`, commit `0780aaa`, MANIFEST SHA-256 `e6f30e55…`. Do not
  regenerate them.
- *The rule.* It must use the shared `QuiescenceRule` (`src/rld/control/quiescence.py`), so its
  only difference from `gated` and `oracle_gated` is which deck samples it sees. That is predicted
  samples instead of the past (`gated`) or the true future (`oracle_gated`).
- *The benchmark.* The Phase 3 numbers are the bar. At SS6, `gated` times out on 35–55 % of
  episodes. `oracle_gated` is **not** a bound on success, and only 63–85 % of its SS5/SS6
  touchdowns fall in a truly quiescent window. So "forecast-gated approaches the oracle" is not
  the same claim as "lands well".
- *Metrics.* Report `td_in_quiescent_window` beside success.

(c) **First `controls-engineer` task: fix the stale "upper bound" labels** carried by P3-D4. They
are in `configs/control/oracle_gated.yaml`, the `registry.py` docstring and the
`PrivilegedContext` docstring. Replace them with "commit-timing oracle (privileged)". Because the
YAML's bytes are hashed into `results/e01/summary.csv`, re-render e01 in the same commit, or record
the new hash beside the old one.

(d) **Froude time is unchanged and still the trap.** The forecaster's 200-sample, 10 Hz lookback
is **full-scale** (20 s full = 4.0 s model). The episode start window `(4.0, 107.5)` s model
already reserves exactly that lookback (P2-D7).

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

**Before you start (added after Gate 4, 2026-09-23).**

(a) **Still open from P3-D4.** Check that `pid_feedforward_lowvz`'s low-speed bounces are not a
contact-solver artifact that a residual or pure policy could learn to exploit. Do this before
Phase 5 training starts.

(b) **Tuning pool.** Item 3's "`id`-validation seeds" means `dev_pool()[1]`, the tune pool (P3-D2).
Never use `id`-val, which contains frozen SS6 and 90° test realizations.

(c) **Evaluation interface.**
- `Controller.reset` is now `reset(seed, context=None, motion_feed=None)`.
- The runner hands a `ShipMotionFeed` only to registry entries with `needs_motion_feed=True`. Pure
  PPO and SAC must not take one: `CallablePolicy` rejects a feed.
- `make baselines` and `tune_controller.py --final` are pinned to the five Phase 3 controllers.
- `run_info.json` now records `git_sha` and `git_dirty`, so launch evaluations from a clean,
  committed tree.

(d) **Phase 4 results that bear on the training set-up.**
- Every gated controller fails only by `timeout`, and `pid_feedforward` is at or near the top on
  success in every cell (P4-D4).
- The hacking audit's timeout fraction (item 5) should therefore be read against `gated`'s timeout
  rate, since `gated` fails the same way: it times out on 32–55 % of SS6 episodes on frigate lists and 9–10 % on s175 (`results/e02`).
- The forecast observation block is **Phase 6** (P4-D4a records), not this phase.

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

**Before you start (added after Gate 5, 2026-09-30).**

(a) **Inherited settings.**
- Every Phase 6 PPO-family method inherits `configs/rl/ppo.yaml` unchanged (P5-D11: trial 14's
  hyperparameters and reward weights, `log_std_init` −2.80).
- Budget: 10 M steps, seeds 0–4. No budget cut (P5-D12). A run takes about 3.2 h, so 20 runs take
  roughly 16–20 h at 4 concurrent.
- Tuning trials used: PPO 20 of 20 (P5-D4, P5-D7, P5-D11). Phase 6 methods get no search of their
  own (P3-D1 §5).

(b) **H1a confound (P5-D1, P5-D3, P5-D14).**
- Pure PPO already cuts to idle after contact in 3 of 5 seeds. A residual policy can learn the
  same thing, and `pid_feedforward_lowvz_cut` is the control for it.
- Run audit check 6 (post-contact down-force) on every residual seed. Report it beside H1a.

(c) **When comparing PPO with the PID baselines** (results-skeptic minor 5):
- PPO lands with a two-phase descent: about −1.2 m/s, then −0.26 m/s relative to the deck. The
  PID tuning space (constant descent 0.08–0.60 m/s) cannot express that.
- 11 of `pid_feedforward`'s 19 SS6 losses are `bounce`. P5-D1 showed that class is driven by the
  50 ms grace rule and is unstable at 240 Hz.
- Carry both facts wherever an RL-vs-PID gap is stated.

(d) **Measurement caveats (P5-D14).**
- The recorded closing speed understates impact speed by about 7 %.
- Tunnelling is counted at any contact substep, whatever `success.yaml`'s comment says.
- Keep both caveats beside any closing-speed or success number near the limits.

(e) **Hover trap (P5-D9, P5-D14).** Report training timeouts at 1 % step resolution for every
Phase 6 run: the 10-bin view missed SAC's early hovering.

(f) **Infrastructure.**
- Keep the host awake. Runs resume from checkpoints (P5-D10), but a VM kill still loses up to
  1 M steps.
- Commit before launching, so each run records a clean `git_sha`.

**First task:** the `residual_ppo` wrapper. Its action is `pid_feedforward` + 0.3·π(o) through the
env's norm cap, with the last layer zero-initialised (P3-D1 §6). Its gate test: a zeroed residual
reproduces `pid_feedforward` exactly on a frozen-list sample.

Methods (all 5 seeds, same budget as PPO):
1. `residual_ppo`: action = `pid_feedforward` setpoint + α·π(o), α pre-registered (default 0.3·v_max);
   policy initialised so that α·π ≈ 0 at start (zero-initialised last layer).
2. `ppo_forecast`: pure PPO with the Phase 4 forecast block in the observation.
3. `residual_ppo_forecast`: both.
4. `ppo_sinusoid`: pure PPO trained on sinusoidal motion (for H4).

**Gate 6**: all runs complete; `id` results committed; residual policies verified to reduce to the
baseline when the residual is zeroed (test).

### Phase 7 — Evaluation under shift and ablations (2 days + ~0.5 day compute) · owner: `eval-auditor`

**Before you start (added after Gate 6, 2026-10-01).**

(a) **H4 is bounded at its pre-registered cell (P6-D6).**
- `ppo_sinusoid` and `ppo` are both 200/200 on JONSWAP `id` SS5 in every seed. So at `id` SS5,
  drop_sin − drop_jon ≤ 0 whatever the sinusoid leg shows, and H4 cannot be supported there.
- **User decision (2026-10-01, P6-D6): H4 stays at `id` SS5 as pre-registered.** Score it as
  written. The novelty claim is withdrawn in the README (D0.4; also P3-D1 §8 if the CI includes
  0), and the transfer is reported as the finding. The H4 cross is still flown in full and reported.

(b) **Evaluation-side work this phase needs.**
- *Sinusoid test motion.* `rld.eval.envs.motion_for` builds only JONSWAP and static motion. The H4
  cross needs a sinusoid test leg on the same lists, matching `rld.rl.motion`:
  - amplitudes √2 × the committed RMS;
  - the realization's peak encounter period;
  - per-episode phases from the episode seed.
  Record this definition in `docs/protocol.md` **before the first sinusoid flight**.
- *Feed methods.* `ppo_forecast` and `residual_ppo_forecast` need the runner's ship-motion feed.
  `rld.eval.learned` already passes it (P6-D4). The static list cannot feed them.
- *Relative-p95.* The paired relative-p95 statistic (P3-D1 §4) is not implemented yet.

(c) **Carry these into every table and verdict.**
- The P6-D1 forecast caveats: forecasts were in-sample in training, and the feed is an extra ideal
  sensor.
- The residual methods descend *harder* than their base (P6-D5, corrected): no residual seed cuts
  the throttle, and a `lowvz`-like descent was within authority. Score H1a exactly as written.
- From the Phase 6 "Before you start" note (c) and (d): the two-phase descent, the 50 ms
  bounce-grace rule, the ~7 % closing-speed understatement, and tunnelling counted at any contact
  substep.
- P6-D5's tunnelling bound: 3 / 1 / 0 / 3 SS6 successes per 1 000.
- All hard landings of the four Phase 6 methods, `ppo` and `pid_feedforward` in e06 are
  deck-tilt events (P6-D5). `sac`'s and `pid_track_descend`'s are mostly speed-driven: 352 of
  378 and 27 of 33 exceed 0.5 m/s.

(d) **λ sensitivity runs on "the best two methods".** Fix the rule that picks them (metric, cell,
tie-break) in `docs/protocol.md` before reading any Phase 7 number.

(e) **Labels.**
- The `ppo_sinusoid` learning-curve panels are on sinusoid motion. Label them in the Phase 9
  figures.
- `results/e06/success_vs_seastate.md`'s "not audited" line is superseded by P6-D5.

**First task:** the paired relative-p95 statistic in `rld.eval.stats`, with its unit test on a
synthetic case with a known r (P3-D1 §4). Then the sinusoid test-motion leg in the evaluation
runner.

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

**Before you start (added after Gate 7, 2026-10-04).**

(a) **H5 is the only open hypothesis.** It is scored here, at Gate 8, exactly as P3-D1 §8 writes it:
batch-1 p50 on ORT CPU (1 thread) vs every parity-passing GPU provider, ≥ 2×, Project 4's harness,
200 warmup + 2 000 timed, p99 reported (P7-D1 §7).

(b) **Which network.** Fix the exported policy in `docs/protocol.md` **before timing anything**.
H5 names "the exported policy MLP". The natural choice is `ppo`, the Phase 7 best learned method
(P7-D1 §3). Its five seeds share one architecture, so latency is per architecture, and parity is per
seed. State what is exported for the residual and forecast methods:
- the residual's `pid_feedforward` base stays outside the graph (plan step 1);
- the forecast methods also need the DLinear-OLS forecaster per step, which added about 1.2 ms per
  env step in training (P6-D1). That belongs in the end-to-end budget (step 4), not in H5.

(c) **Closed-loop parity.** Fly the 50 episodes from the frozen `id` list with noise off, the
configuration every scored number used. The P7-D4 stand-in is irrelevant here.

(d) **Carry these into the README (Phase 9), from Gate 7.** These are the `docs/findings.md` Phase 7
readings that are easy to overstate:
- H1b's 1.0-point margin and its out-of-distribution cell;
- plain `ppo` lies above `residual_ppo` at the H1b cell (unpaired);
- `pid_feedforward` unbeaten on the S175 hull and on MSS;
- the white-noise perception arm, whose σ is larger than the deck motion at SS3–SS4;
- unaudited tunnelled successes outside clean `id`.

(e) **Carry into the README (Phase 9), from Phase 8 (P8-D5, added 2026-10-05).** Outcome classes at
SS6 near the 15° tilt limit are not determined at float32 precision. About 6.6 % of SS6-200
policy-episodes are rounding-sensitive (53 of 800 flip under at least one of 20 one-ulp input
perturbations; P8-D4 §6). This qualifies the per-episode resolution of every committed SS6 success
count. Report closed-loop parity with both verdicts: P8-D1 §7 as written, not met; P8-D5's post-hoc
noise-floor criterion, met for ORT CPU; ORT CUDA not judged (it would fail (ii) for 2 of 4
policies; P8-D5 §3). H1b's 1.0-point margin at `id` SS6 is of the same order as the rounding-level per-seed shifts in SS6 success measured here: −2.5 to +1.5 points under one-ulp noise and −0.5 to +1.5 under the float64 reference, per seed of the exported policies. `residual_ppo` itself was not re-flown, and about 30 % of flip events (88 of 291 ulp flips) go through the bounce channel, the rule H1b is already noted to be fragile to.

**First task:** export the `ppo` actor (seed 0) with the `VecNormalize` mean and variance folded into
the graph, opset 18, dynamic batch, and pass the 1 000 × 5 parity on ORT CPU.


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
