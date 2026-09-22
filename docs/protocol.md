# protocol.md — decision log

Every decision, threshold, deviation and gate result, dated, in order. Where this file and
`IMPLEMENTATION_PLAN.md` disagree, **this file is what happened.** Entries are append-only:
a later entry may supersede an earlier one, never edit it.

Format: `### P<phase>-D<n> — <title> (YYYY-MM-DD)` then *Decision*, *Reason*, *Evidence*.

---

## Phase 0

### P0-D1 — Pinned dependencies (2026-09-21)

*Decision.* The environment is pinned exactly, `deck-motion-forecast`'s resolved set is reused
verbatim where it overlaps, and both third-party projects are installed editable from pinned
submodules.

- deck-motion-forecast submodule SHA: `e9fa15cc35312a5c1ecf0c42668a690b5061c090`
- gym-pybullet-drones submodule SHA: `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`
- Resolved pins (this project's own; the dmf-shared block is in `pyproject.toml`):
  gymnasium **1.3.0**, stable-baselines3 **2.9.0**, pybullet **3.2.7** (sdist build),
  rliable **1.2.0**, tensorboard **2.21.0**, imageio **2.37.4** + imageio-ffmpeg **0.6.0**.
  `optuna` is deliberately *not* installed; it sits in an optional `tune` extra because no
  committed result depends on it (plan §4 calls it optional).
- Inherited from dmf unchanged: numpy 2.5.2, scipy 1.18.1, pandas 3.0.5, pyarrow 25.0.1,
  pyyaml 6.0.3, torch 2.13.0, onnx 1.22.0, onnxruntime-gpu 1.29.0, tensorrt-cu13 10.16.1.11,
  matplotlib 3.11.1, rich 15.0.0, tqdm 4.70.0; dev: pytest 9.1.1, pytest-cov 7.1.0, ruff 0.16.4,
  mypy 2.3.1, types-PyYAML, pandas-stubs.
- Machine: Python 3.12.3, Ubuntu/WSL2 (kernel 6.6.114.1), Intel i9-10980XE (36 logical cores),
  62 GB RAM, RTX A4000 driver 596.71, gcc 13.3.0. `torch` resolved to **2.13.0+cu130** from the
  default PyPI index and `torch.cuda.is_available()` is True, matching dmf's reference machine.

*Evidence / notes.*

1. **pybullet built from sdist, as predicted.** No cp312 wheel exists; pip downloaded
   `pybullet-3.2.7.tar.gz` (80.5 MB) and produced
   `pybullet-3.2.7-cp312-cp312-linux_x86_64.whl` (99 MB) locally, now in pip's wheel cache.
   `build-essential` and `python3.12-dev` were already present. The whole three-package
   editable install took under 5 minutes end to end on a cold pip cache.
2. **rliable co-installs cleanly** with `numpy==2.5.2` / `pandas==3.0.5`, pulling arch 7.2.0,
   statsmodels 0.15.0, formulaic, patsy, seaborn and absl-py. No dmf pin had to be relaxed, so
   the contingency planned for Phase 0 (moving rliable to an optional extra) was not needed and
   CLAUDE.md non-negotiable 5 stands as written.
3. **Editable install verified functionally, not just by import.** `tests/test_smoke.py` runs
   `dmf.sim.generate.simulate_realization` on one frigate SS5 spec; `dmf` resolves
   `configs/sim/vessels` via `Path(__file__).parents[3]`, so that call passes only from an
   editable install. Likewise `pybullet` is checked by opening a DIRECT client and stepping it,
   not by `import pybullet` alone — an sdist build can import and fail to link.
4. **Layout note.** The Phase 0 throughput logic lives in `src/rld/bench/throughput.py` with
   `scripts/env_throughput.py` as its argparse wrapper. Plan §3 does not list a `bench/` package;
   it lists `deploy/bench.py` for Phase 8, which is a different measurement (policy inference
   latency, for H5). Keeping them apart avoids a Phase 8 collision.

### P0-D2 — Environment throughput and what it does and does not say (2026-09-21)

*Decision.* `results/env_throughput.csv` is the Phase 0 measurement that Phase 5's budget in P3-D1
will be sized from. The Phase 5 budget must be read from the **`vel` rows**, not the `rpm` rows.

*Measurement.* `HoverAviary` (cf2x, DIRECT/headless, pyb 240 Hz, ctrl 30 Hz), random actions,
20 000 timed environment steps per row after a 50-vector-step warmup, `OMP_NUM_THREADS=1`,
36 logical cores. A *step* is one control tick of one drone, so a 16-worker vector step is 16 steps.

| backend | n_envs | act | steps/s | per env | steps/episode | h per 10 M steps |
|---|---|---|---|---|---|---|
| DummyVecEnv   | 1  | rpm | 512  | 512  | 13.5  | 5.42 |
| SubprocVecEnv | 1  | rpm | 402  | 402  | 13.5  | 6.91 |
| SubprocVecEnv | 8  | rpm | 678  | 85   | 14.6  | 4.10 |
| SubprocVecEnv | 16 | rpm | 831  | 52   | 13.9  | 3.34 |
| DummyVecEnv   | 1  | vel | 1175 | 1175 | 243.9 | 2.36 |
| SubprocVecEnv | 1  | vel | 789  | 789  | 243.9 | 3.52 |
| SubprocVecEnv | 8  | vel | 4956 | 620  | 250.0 | 0.56 |
| SubprocVecEnv | 16 | vel | 7575 | 473  | 250.0 | **0.37** |

*Reason the `vel` rows are the ones to use.* The first run of this measurement showed `vel`
roughly twice as fast as `rpm`, which is backwards: `vel` adds a `DSLPIDControl` inner loop at the
physics rate. Counting episode boundaries explains it, and the counts are now a committed column.
Under a **random** policy a random RPM vector tumbles the Crazyflie out of bounds in about 14
steps, so the `rpm` rows spend most of their wall clock in `reset`, reloading the drone URDF; the
`vel` rows run ~244 steps, essentially to `EPISODE_LEN_SEC`. The `rpm` rows therefore measure how
fast a random policy crashes, not what an action space costs, and their near-flat scaling
(402 → 831 steps/s for 16× the workers) is reset serialisation, not step cost. The `vel` rows scale
789 → 7575 steps/s, 9.6× on 16 workers, and their episode length is the right order for a landing
episode (≤ 12 s model time at 30 Hz = 360 steps).

*What this number does not cover.* It is environment stepping only — no policy forward pass, no
PPO update, no `VecNormalize`. And it is `HoverAviary`, not the Phase 2 landing environment, which
adds a second PyBullet body (the deck), a constraint driven every physics step, and the deck-motion
bridge evaluation. Phase 2 must re-measure before P3-D1 fixes the budget; this row is the ceiling,
not the estimate. Recorded as the ceiling deliberately: 10 M steps at 7575 steps/s is 0.37 h of
stepping, so a 5-seed PPO sweep is not compute-bound at Phase 0's best case, and any budget
argument in P3-D1 that claims otherwise has to explain what got slower.

## Phase 1

### P1-D1 — Froude scale, pad position and the feasibility rule (2026-09-21)

*Decision.* **λ = 1/25 confirmed. `r_pad = [−0.4·L, 0, 0]` full scale confirmed.** Neither is changed.
The pre-registered feasibility rule passes with a 3.61× margin under the reference named below.

*The rule's denominator was ambiguous and is resolved here, before the number was read.* Phase 1's
rule reads "SS6 head-seas deck-point v_z p99 must be ≤ 25 % of the CF2X's commanded max speed in
gym-pybullet-drones". That phrase resolves two ways in the pinned submodule:

| reference | source | 25 % threshold | measured | verdict |
|---|---|---|---|---|
| `cf2x_urdf_max_speed_kmh` **(the gate)** | `cf2x.urdf:5 max_speed_kmh="30"` → 8.3333 m/s | **2.08333 m/s** | 0.577165 m/s | **PASS** (3.61× inside) |
| `baserlaviary_speed_limit` (rejected) | `BaseRLAviary.py:95 SPEED_LIMIT = 0.03·MAX_SPEED_KMH·(1000/3600)` → 0.25 m/s | 0.0625 m/s | 0.577165 m/s | FAIL (9.2× over) |

*Reason for choosing the URDF reading.* The `SPEED_LIMIT` reading is unsatisfiable, not merely
strict: `v_z ∝ √λ`, so satisfying 0.0625 m/s would need λ ≈ 1/11000 (a 1.1 cm frigate), and on the
committed data **every sea state from SS3 up fails it** (SS3 head seas aft = 0.1661 m/s, 2.66× over).
A rule no cell in the corpus can pass at any usable λ cannot be the rule. It also contradicts the
plan's own 0.5 m/s touchdown limit: a vehicle capped at 0.25 m/s could not null a 0.5 m/s deck.
Recorded so the rejected reading is auditable rather than invisible.

*Evidence.* `results/deck_feasibility.csv` (2 rows, `--threshold-reference all`), measured cell
frigate SS6 180° **12 kn**, aft pad — the worst of the three speeds in the cell the rule names
(0 kn 0.530188, 6 kn 0.565162, 12 kn 0.577165 m/s). The verdict is robust to the choice of cell:
the **globally worst aft-pad cell in the whole grid** is s175 SS6 180° 12 kn at 0.7002 m/s, still
2.98× inside the threshold, and the worst frigate cell anywhere is SS5 180° **6 kn** at 0.5790 m/s —
marginally above the SS6 cell the rule names. Nothing in the grid comes within 2.9× of the gate
threshold, so λ = 1/25 is not a near miss.

*Sizing numbers this phase fixes for later phases.*

- `results/deck_stats.csv` — 192 rows (96 cells × {aft, cg} pad), 48 columns, all 2304 realizations,
  40 seeds/cell frigate and 8 s175, sampled at the **model** physics rate 240 Hz (= 48 Hz full scale
  at λ = 1/25) over each 600 s full-scale record. SHA-256 prefix `3bf4793be4da5269`.
- `results/deck_stats_seeds.csv` — 4608 rows (per realization × pad), the audit trail behind the
  192-row table and the committed per-realization sinusoid parameters Phase 6 (`ppo_sinusoid`) and
  Phase 7 (H4 cross) consume. SHA-256 prefix `0c49556027cfc5e4`.
- `results/deck_feasibility.csv` — SHA-256 prefix `fef80986a03fcf42`.
- **Cell statistics are pooled-sample, not means over seeds.** Every realization is pooled in sorted
  spec order before percentiles are taken, because a percentile of a mean of percentiles is not a
  percentile. Per-realization values are in `deck_stats_seeds.csv`, so a mean-over-seeds table is
  recomputable without re-running. (The plan's column list said "means over seeds"; its Addendum C6
  said pooled. Pooled wins, and this sentence is why.)
- **Sampling rate is not load-bearing.** `vz_p99` at 240 Hz model vs at the corpus's 10 Hz full-scale
  rate differ by max 0.247 %, median 0.029 % over 192 rows (committed as
  `vz_p99_abs_at_corpus_fs_m_s`). The 240 Hz figure is the committed one because it is the grid the
  environment evaluates, not because 10 Hz is biased.
- **One project-wide λ across both hulls** — see P1-D3.
- **Angular acceleration's Froude factor is λ⁻¹** (625 at λ = 1/25), derived as angular_rate ÷ time.
  It is absent from plan D0.1's seven-row table and is deliberately kept out of `SCALE_EXPONENTS` so
  the table in code stays exactly the plan's; it is documented as derived in `to_model_state`.
- **Gravity differs between the two dependencies and is not reconciled**: dmf uses 9.80665 m/s²,
  gym-pybullet-drones 9.8 (`BaseAviary.py:74`), 0.068 % apart. "Acceleration ×1" assumes
  `g_model == g_full`. Below every tolerance in this phase; recorded, not fixed.

*Two findings handed to `sim-env-engineer` and `eval-auditor`, both from committed columns.*

1. **`BaseRLAviary`'s `ActionType.VEL` cannot be reused verbatim.** Its `SPEED_LIMIT` = 0.25 m/s is
   below the deck's own aft-pad `v_z` p99 at every sea state from SS3 up (0.166 / 0.351 / 0.575 /
   0.577 m/s at 180°, 12 kn). Phase 2's shared action space must set `v_max` as a project parameter,
   ≳ 1.0 m/s. Phase 1 reports the number; Phase 2 sets `v_max` and records it.
2. **The 15° relative-tilt success criterion is exceeded by the deck alone in 10 of 96 cells**, max
   24.492° (frigate SS6 90° 0 kn). The set is dominated by beam and stern-quartering SS6, i.e. by
   **roll**, not pitch. Those cells are structurally 0 % under the criterion as worded, so P3-D1 must
   either state that the criterion is on *relative* tilt (it is) or exclude them knowingly. Committed
   as `tilt_p99_deg` and `tilt_p99_over_15deg`.

### P1-D2 — Deck-point sign convention and rotation order: the plan and the skill were both wrong (2026-09-21)

*Decision.* The deck-point kinematics use a **single** world frame — `x` = bow, `y` = **port**,
`z` = up, right-handed, PyBullet's frame — with dmf's angle *signs* mapped into it rather than a
second SNAME frame. The composition is **ZYX** (`R = R_y(θ_w)·R_x(φ_w)`, ψ = 0), which is what
`pybullet.getQuaternionFromEuler` uses, with `φ_w = +radians(roll_deg)`, `θ_w = −radians(pitch_deg)`:

```
z_pad   = heave + x_pad·sin(pitch)                     # PLUS sign; x_pad signed, negative aft
ω       = (φ̇·cos θ_w,  θ̇,  −φ̇·sin θ_w)
α       = (φ̈·cos θ_w − φ̇·θ̇·sin θ_w,  θ̈,  −φ̈·sin θ_w − φ̇·θ̇·cos θ_w)
v_pad   = [0,0,heave_rate] + ω × p_offset
a_pad   = [0,0,heave_acc]  + α × p_offset + ω × (ω × p_offset)
n_world = (−sin(pitch)·cos(roll),  −sin(roll),  cos(pitch)·cos(roll))
```

A **starboard** pad therefore has `y < 0`. `configs/deck/pad.yaml` states this.

*What was wrong.* `docs/IMPLEMENTATION_PLAN.md` D0.5 writes the deck point as
`heave − x_pad·sin(pitch)`, and `.claude/skills/deck-scaling-physics/SKILL.md` wrote
`v_z = heave_rate − x_pad·cos(pitch)·pitch_rate` (twice — once under "Deck-point kinematics", again
under "Known dmf defect") and the composition `R_x(roll)·R_y(pitch)`. With `r_pad = [−0.4·L, 0, 0]`
and dmf's stated pitch-positive-bow-up, that minus sign makes an **aft** point rise when the bow
rises. The skill's three occurrences are corrected in this change; D0.1's and D0.5's text is
**superseded by this entry, not edited** — the plan stays the original plan.

*Why it is not cosmetic.* dmf's known in-phase defect (P8-D6 in dmf's own protocol; the missing
factor of `i` at `dmf/sim/response.py:229`) means pitch and heave are nearly in phase, so the sign
decides whether the lever-arm term **adds to or cancels** `heave_rate`. Measured on the committed
data, `corr(pitch, heave)` is **+0.812 min, +0.865 median, +0.948 max over all 96 cells** where strip
theory would put it near 0.

*Consequences — results, recorded, not fixed.*

| frigate 180° 12 kn | aft `v_z` p99 | CG `v_z` p99 | aft z std | aft/CG `v_z` RMS ratio |
|---|---|---|---|---|
| SS3 | 0.1661 | 0.0319 | 1.044 cm | 5.181 |
| SS4 | 0.3512 | 0.1037 | 2.528 cm | 3.384 |
| SS5 | 0.5746 | 0.2286 | 4.430 cm | 2.497 |
| SS6 | **0.5772** | 0.4612 | 4.880 cm | **1.275** |

(m/s, model scale.) Two structural results follow, and both were confirmed on 40 seeds after being
predicted on 6:

1. **The aft-pad difficulty ladder saturates between SS5 and SS6** (0.5746 → 0.5772 m/s, +0.5 %),
   because at SS6's long `Tp` = 12.4 s the in-phase pitch term increasingly cancels heave.
2. **The CG ladder is the clean monotone one** (0.0319 → 0.1037 → 0.2286 → 0.4612, each step
   2.2–3.3×), so D0.5's pad-at-CG control arm carries more of the difficulty ladder than the plan
   assumed. It is no longer only a defect control.

Cancellation is not a head-seas curiosity: **30 of 96 aft-pad cells have `vz_rms_ratio_pad_over_cg`
below 1**, i.e. the aft pad moves *less* vertically than the CG (min 0.685, s175 SS3 90° 6 kn). The
aft/CG ratio over aft rows is min 0.685, median 1.769, max 7.378.

*Evidence that the plan's own scouting table used the wrong sign.* Plan D0.1 reports SS5 180° z std
7.7 cm / `v_z` std 0.36 / p99 0.95 and SS6 13.2 cm / 0.50 / 1.33. Recomputing with the minus sign
reproduces those to three digits (SS5 7.90 / 0.367 / 0.976; SS6 13.04 / 0.495 / 1.289), where the
correct sign gives 4.43 cm / 0.222 / 0.575 and 4.88 cm / 0.226 / 0.577. D0.1 flagged its own sign as
"unverified"; it was wrong, and by a factor of ~2.2 in `v_z` p99 at SS6.

*How the sign and the order are pinned so they cannot regress.* Three hand-computed tests in
`tests/test_kinematics.py`, each asserted componentwise at `atol = 1e-9`, each with an
anti-assertion against the wrong value:

- **Case A (sign).** roll 0°, pitch 6°, heave 0.5 m, pitch_rate 2 °/s, heave_rate −0.2 m/s,
  `x_pad = −49.6 m` ⇒ `p_offset = (−49.328286010, 0, −5.184611778)`, `z_pad = −4.684611778 m`,
  `v_pad = (+0.180977092, 0, −1.921882010) m/s`; model scale `z = −0.187384471 m`,
  `v_z = −0.384376402 m/s`. The aft pad goes **down and forward** on bow-up. The wrong sign gives
  `+6.1846 m`, which the test excludes explicitly.
- **Case B (roll sign, lateral term).** roll 10°, roll_rate 3 °/s, pad at `(−49.6, −7.0, 0)`
  (starboard) ⇒ `p = (−49.6, −6.893654271, −1.215537244)`, i.e. the starboard pad is **below** the CG,
  with the port pad paired as the opposite assertion.
- **Case C (rotation order).** roll 30°, pitch 6°, centreline aft pad ⇒ `p_y == 0.0` **exactly**, and
  `p_z` equal to Case A's offset (roll-blind). The discarded `R_x·R_y` order gives
  `Δz = 0.694606270 m` and a spurious `p_y = +2.592305889 m`.

Plus a `pybullet`-marked test that `getMatrixFromQuaternion(getQuaternionFromEuler(euler))` equals
our rotation matrix to 1e-12, which is what makes Phase 2's platform orientation the same `R`.

*One correction to the plan's own arithmetic, for the record.* The approved plan stated Case A
`v_z = −1.9224` m/s full / `−0.38449` model. The correct value is **−1.921882010 / −0.384376402**
(`−0.2 + (−49.6)(cos 6°)(radians 2) = −0.2 − 1.721882010`), self-consistent with the plan's own
`p_x = −49.328286010` because the lever term equals `ω_y·p_x`. The implementing agent flagged the
disagreement with its derivation rather than adopting either value silently; the tests use the
correct one.

*Side effect of the correct order, asserted rather than discovered later.* With `y = z = 0` the
centreline pad is **exactly roll-blind** — `R_x` fixes `(x, 0, 0)`, so roll and roll_rate contribute
nothing to pad position or velocity. Beam seas is therefore the **easiest** pad-motion regime
(frigate SS5 aft `v_z` p99 = 0.2117 m/s at 90° vs 0.5746 at 180°) and simultaneously the **worst
tilt** regime (23.16° at SS6 90°). `unseen_heading` must not be described as simply "harder" in
Phase 5 or 7 without that qualifier. `configs/deck/pad.yaml` exposes `y_m`/`z_m` (default 0) for a
roll-sensitive pad if a later phase wants one.

### P1-D3 — One project-wide λ, not per-vessel (2026-09-21)

*Decision.* A **single** λ = 1/25 is used for both hulls. The lever arm is allowed to differ with
length (frigate −49.6 m, s175 −70.0 m full scale; −1.984 m and −2.800 m model).

*Reason.* λ is defined as `L_model / L_full`, so a literal reading gives each vessel its own λ and
therefore its own **time and velocity** rescaling. `unseen_vessel` would then measure transfer across
a second rescaling as well as across hull dynamics, and its result would not be interpretable as
either. One λ keeps the time base common and leaves exactly one thing varying.

*Consequence, now measured rather than assumed.* `unseen_vessel` is a harder regime in pad motion and
not only in modal parameters: the s175 owns the grid's worst aft-pad cell (0.7002 m/s vs the
frigate's worst 0.5790 m/s), largely because its lever arm is 41 % longer. The confound is committed
per cell in `deck_stats.csv` (`pad_x_m_full`), so Phase 7 can report it instead of discovering it.

## Phase 2

### P2-D1 — The platform driver is `kinematic`, measured against `constraint` (2026-09-22)

*Decision.* The deck plate is driven **kinematically**: a dynamic body (1 000 kg) whose weight
is cancelled by an external force at its centre of mass, with pose **and** velocity — linear
and angular — written from the analytic deck state every **physics** step (240 Hz model).
`configs/env/landing.yaml: platform.driver: kinematic`. The `constraint` driver is implemented
and stays selectable, so this entry can be re-measured rather than re-argued.

*This was measured, not chosen.* Both drivers were run over the same 10 s model-scale sweep at
the frigate SS6 180° 12 kn aft pad (the worst frigate cell the P1-D1 feasibility rule names),
driven at 240 Hz, with every quantity read from PyBullet **before** `stepSimulation` — the
state the contact solver is about to use, which is the only reading a touchdown can see.

| metric | gate | `constraint` | `kinematic` |
|---|---|---|---|
| body-origin pose error, max | ≤ 1 mm | **11.370 mm** ✗ | **0.000 mm** ✓ |
| plate-corner pose error, max | ≤ 1 mm | **14.754 mm** ✗ | 2.2e-16 m ✓ |
| orientation error, max | ≤ 1 mrad | **8.457 mrad** ✗ | 3.0e-8 rad ✓ |
| `getBaseVelocity` linear ratio, p99 | ≤ 2 % | **416 %** ✗ | 0.0 % ✓ |
| `getBaseVelocity` linear ratio, RMS | ≤ 2 % | **83.5 %** ✗ | 0.0 % ✓ |
| `getBaseVelocity` angular ratio, p99 | ≤ 2 % | **219 %** ✗ | 0.0 % ✓ |
| post-step drift vs the next analytic pose | ≤ 1 mm | 11.370 mm | 0.024 mm |
| conveyor: drone carried at 0.5 m/s | yes | **yes** (v error 0.0 %) | **yes** (v error 0.02 %) |

Velocity ratios are `‖v_pyb − v_analytic‖ / max(‖v_analytic‖, v_floor)` with
`v_floor = 0.02 × v_z RMS` of the same record, because the deck's velocity crosses zero.

*Why `constraint` loses, and why it cannot be tuned into winning.* `changeConstraint` is a
velocity-level servo, so the body trails its retargeted frame by a few physics steps: 11.4 mm
is about three steps of the pad's horizontal speed. The failure is **insensitive to every knob
it has**. A 24-point sweep over `maxForce ∈ {1e6, 1e9}`, `erp ∈ {default, 0.9, 1.0}`,
`numSolverIterations ∈ {50, 200}` and plate mass ∈ {1, 1000} kg returned
`pos_max = 1.137e-2`, `quat_max = 8.457e-3`, `vratio_p99 = 4.162` to **four identical digits in
all 24 rows**. `tests/test_platform.py::test_constraint_driver_fails_the_tracking_gate` asserts
the failure, so a PyBullet upgrade that changed it would force a re-measurement instead of
inheriting this verdict silently.

*The skill file was wrong and is corrected.* `.claude/skills/pybullet-moving-platform/SKILL.md`
called the constraint driver "Preferred" and offered reset-pose-plus-`resetBaseVelocity` as the
"Alternative". On this project's numbers that is backwards by an order of magnitude on every
gated metric. The skill is edited in place (as P1-D2 did for `deck-scaling-physics`), now carries
the measured numbers and the conveyor test, and says it was previously wrong; this entry is the
record of why.

*Two implementation details the numbers depend on, both non-obvious.*

1. **The plate's weight must be cancelled, and at its centre of mass in `WORLD_FRAME`.**
   PyBullet integrates gravity into a body's velocity *before* the constraint solver runs, so
   without cancellation the plate enters every contact solve at `v − g·dt` = 40.8 mm/s low —
   7 % of the deck's own peak |v_z| at SS6, straight into the touchdown relative velocity.
   Applying the force with `LINK_FRAME` and `posObj=[0,0,0]` looks equivalent and is not: in
   `WORLD_FRAME` `posObj` is a *world point*, and passing the origin there applies 9 800 N at a
   1 m lever arm, which produced 6.3e-3 rad of spurious rotation per step until it was found.
2. **Tracking must be measured at the plate's corners, not only its origin.** A plate rotated
   about its own body origin has *zero* origin position error; the 0.40 m lever turns 1 mrad
   into 0.4 mm, which is what makes the corner column above the binding one for `constraint`.

*The conveyor test is the decisive one, and it is why `getBaseVelocity` is not enough.* Under a
kinematic drive `getBaseVelocity` reads back exactly what was written, whether or not the
solver used it — a perfect 0.0 % that proves nothing. The test that does prove something is a
drone at rest on a plate translating at 0.5 m/s for 2 s:

| drive | drone x after 2 s (plate travels 1.000 m) | drone v_x |
|---|---|---|
| `kinematic` | +0.949 m | +0.4999 m/s |
| `constraint` | +0.946 m | +0.5000 m/s |
| `teleport` (pose only, the forbidden drive) | **−0.062 m**, then off the plate and falling | **−0.043 m/s** |

The ~5 % displacement shortfall on the two real drivers is the 0.2 s of friction spin-up at
μ_eff = 0.25 (2.45 m/s²), not a tracking error; the velocity is the tight end of the test.
`tests/test_platform.py` runs all three, and asserts the teleport drive **fails**.

### P2-D2 — The shared action space: `v_max` = 1.5 m/s, a norm cap, and yaw held at 0 (2026-09-22)

*Decision (user, 2026-09-22).* One action space for every method in the project — PID
baselines, PPO, SAC, residual, forecast-conditioned: `Box(-1, 1, (3,), float32)`, a **world-frame
velocity setpoint** scaled by `v_max = 1.5 m/s` (model scale), yaw commanded to **0**, tracked by
gym-pybullet-drones' `DSLPIDControl` at the 30 Hz control rate. Residual composition
(`base + α·π(o)`) belongs in a Phase 6 wrapper, never in the environment.

*Why 1.5 m/s.* 2.60× the frigate SS6 aft-pad `v_z` p99 (0.577 m/s, P1-D1), 2.14× the globally
worst cell (s175 SS6 180° 12 kn, 0.700 m/s), and 18 % of the CF2X URDF's own 8.333 m/s. P1-D1's
finding 1 stands: `BaseRLAviary`'s `ActionType.VEL` caps at `SPEED_LIMIT` = 0.25 m/s, below the
deck's own vertical velocity from SS3 up, so it is not reused — the environment subclasses
`BaseAviary` and calls `DSLPIDControl` itself.

*`v_max` is a cap on the **speed**, not on each axis.* A commanded vector longer than `v_max` is
rescaled to `v_max` rather than clipped per component. A literal `[−1,1]³ × v_max` would permit
`√3·v_max` = 2.598 m/s on a diagonal, i.e. a diagonal descent 73 % faster than a vertical one —
an anisotropy of the box, not of the vehicle, and one every method would inherit at once.
`configs/env/landing.yaml: v_max_is_norm_cap: true`.

*Yaw is commanded to 0, not to the current yaw — a deliberate deviation from upstream.*
`BaseRLAviary.py:220` passes `target_rpy = [0, 0, state[9]]`, which re-targets whatever yaw the
attitude loop has drifted to: a pure integrator with no restoring term. We pass `[0, 0, 0]`.
Tested two ways: a spy on `DSLPIDControl.computeControl` asserts the yaw target is literally
0.0 while the drone's actual yaw is non-zero, and under the scripted descent yaw stays below
5°. **Measured caveat:** under a *uniformly random* action stream the same drone still reaches
40.9° of yaw before it tumbles past the 60° crash threshold, because a random 1.5 m/s setpoint
drives large roll and pitch and the DSL yaw loop is the weakest of the three. That is a
property of the random policy, not of the yaw target, and it is recorded so the 40.9° is not
rediscovered as a regression.

### P2-D3 — The deck plate, and the pad lever arm that has to be removed from it (2026-09-22)

*Decision (user, 2026-09-22).* The platform is a Froude-scaled frigate flight-deck plate,
**0.80 m × 0.52 m × 0.02 m** model (20 m × 13 m × 0.5 m full at λ = 1/25), **the same size for
both hulls**. The pad is a 0.15 m-radius disc at the plate's **top surface**, which is the
plate's body origin and is the bridge's deck point. `deck_origin_m = [0, 0, 1.0]`.

*Why one plate for both hulls.* The s175 is a containership with no flight deck; a per-hull
plate would confound `unseen_vessel` with a geometry change, which is exactly what P1-D3
refused to do for λ. The lever arm is allowed to differ (P1-D3); the deck is not.

*Why the origin is on the top surface.* The box's collision and visual frames are offset
`−h_z` = −0.01 m in body z, so `deck_origin_m + DeckPointState.position_m` drives the body
directly. Without the offset the bridge's deck point would drive the plate's **mid-plane** and
every landing would be one half-thickness — 1 cm model, 25 cm full scale — too low.
`tests/test_platform.py::test_plate_origin_sits_on_the_top_surface` pins it against the body's
AABB.

*Why z = 1.0 m.* It lifts the mean deck a metre above PyBullet's ground plane, so a drone
released 0.6–1.0 m above the deck flies at z ≈ 1.6–2.0 m and falling to the sea is a distinct,
reachable `crash` rather than a soft landing on the ground plane.

*A bug this phase found, fixed, and now tests for: the pad's standing lever arm.*
`DeckPointState.position_m` measures a pad **from the vessel's mean-position CG**, so the
frigate's aft pad carries a constant −1.984 m offset in world x (−0.4 L at λ = 1/25) and the
s175's carries −2.800 m. Driving the plate with that offset left in put it **two metres aft of
where the drone is released**: every scripted descent fell past the deck and ended
`crash`/`below_deck` — 30 of 30 on the first moving-deck run — while the *static*-pad tests
passed perfectly, because `StaticDeckMotion`'s lever arm is zero. It would also have put the
frigate's and the s175's plates in different places in the world, turning P1-D3's named
lever-arm confound into a world-geometry confound as well.

`rld.envs.platform.pad_offset_model_m(vessel, pads, scale, pad)` returns the constant and
`build_trajectory` subtracts it, so the plate oscillates about `deck_origin_m`. **Only the
constant is removed**: the lever-arm-driven *motion*, `R(t)·r − r`, is untouched and is
asserted to be non-zero. After the fix the same scripted descent on an SS5 deck gives
19 `success` / 6 `hard_landing` / 5 `bounce` out of 30.

*Geometry stated in `success.yaml` rather than left to be rediscovered.* The CF2X collision
shape is a cylinder of radius 0.06 m and length 0.025 m (`cf2x.urdf:34`, mass 0.027 kg). The
lateral criterion is on the **drone centre**, so a landing at exactly 0.10 m puts the airframe
rim at 0.16 m — just outside the 0.15 m painted pad and far inside the plate. That is the
intended reading, and the file says so.

### P2-D4 — The e00 sanity episodes come from the `id` split's val partition (2026-09-22)

*Decision (user, 2026-09-22).* `results/e00_env_sanity.csv`'s episodes are drawn from
`build_all_splits(sim_cfg)["id"]`, partition **val**, with the draw seed **20260922** recorded
in every row of both artifacts. Those realizations are neither trained on nor part of the
Phase 3 frozen evaluation lists, so the sanity sweep cannot contaminate either.

*Shape.* 60 val realizations per sea state, 200 episodes per (policy, sea state) cell, so each
realization contributes several episodes at **different start offsets** — which is this
project's definition of an episode (CLAUDE.md non-negotiable 2), not a shortage of data.

*The two policies fly the identical draw.* The realization order and the episode seeds are a
function of the draw seed and the sea state and **not** of the policy, so the `hover` and
`random` rows of a sea state are a paired comparison. The label is mixed in through
SHA-256, not through `hash()`, which is salted per interpreter process and would have made a
single-worker run and a spawned worker disagree.

### P2-D5 — Three frozen-criterion ambiguities, resolved before they could be tuned (2026-09-22)

The plan's four success criteria do not determine an answer in three places. All three were
settled with the user **before** implementation, are written into `configs/env/success.yaml`
with their reasoning, and are frozen at Gate 3.

1. **`hard_landing` absorbs a relative-tilt violation.** Re-worded as "touched down on the pad
   but violated a touchdown-**quality** criterion — relative vertical velocity **or** relative
   tilt". Without it an on-pad, soft, 20°-tilt landing that dwells its 0.5 s matches **no**
   class, while the six-class list is documented as exhaustive. The list is unchanged;
   `rel_vz_normal_m_s` and `rel_tilt_deg` are both committed per episode so the two causes stay
   separable. `success.yaml: hard_landing_includes_relative_tilt: true`.
2. **`vz_component: deck_normal`.** Criterion 1 is on the closing speed into the surface,
   `max(0, −(v_drone − v_pad)·n̂_deck)`, using the same normal criterion 3 uses. **Both**
   readings are committed on every episode row (`rel_vz_normal_m_s`, `rel_vz_world_z_m_s`), so
   the whole evaluation is recomputable under the world-z reading without re-running anything.
   They differ by `1 − cos(tilt)` = 9.0 % at the grid's worst deck tilt (24.492°, frigate SS6
   90° 0 kn).
3. **Divergence is a `crash`, with a free-text `termination_reason`.** Bounds are ±3 m per axis
   relative to the pad, generous enough that only genuine divergence trips them. Every outcome
   row carries one of `ground_contact | off_plate_strike | tilt_gt_crash | below_deck |
   out_of_bounds | dwell_complete | release | time_limit`, so the `crash` bar stays
   decomposable in Phase 7 without a seventh class.

*Two further ambiguities the implementation had to settle, recorded here rather than left in
code.*

4. **The 60° `crash` threshold is on the drone's ABSOLUTE tilt**, unlike criterion 3's 15°
   relative tilt. It is a loss-of-control detector, not a landing-quality criterion: a
   quadrotor 60° from vertical cannot fly whatever the deck is doing. Both tilts are committed
   per episode (`rel_tilt_deg`, `abs_tilt_deg`), so either reading is recomputable.
   `success.yaml: crash_tilt_component: absolute`.
5. **`analytic_contact_margin_m = 0.001`, and it is PyBullet's own number, not a tuned one.**
   Bullet shrinks a convex collision shape by its margin and inflates the contact by the same
   amount, so a drone resting on the plate settles with its geometric cylinder 0.99 mm clear of
   the geometric plate while `contactDistance` reads ≈ 0. A purely geometric analytic detector
   triggering at exactly zero therefore **never fires on a soft landing** while the contact
   detector fires every time: measured at **4 of 100** scripted static-pad descents before this
   key existed. The analytic detector triggers at `clearance ≤ margin` instead. The margin lives
   on the **drone** body (the plate, built with `createMultiBody`, reports 0.0), and
   `tests/test_platform.py` asserts the config value equals `getDynamicsInfo(drone, −1)[11]`, so
   it cannot drift from the engine.

### P2-D6 — Both touchdown detectors, and what made them agree (2026-09-22)

*Decision.* First contact is detected twice on every episode and **both** records are committed
whether or not they agree: from PyBullet's contact manifold, and analytically from the recorded
drone state plus the analytic deck state. Both are polled at the **physics** rate (240 Hz), not
the control rate. Two detectors disagree when they fire more than one control step (33.3 ms
model) apart, or when exactly one fires.

*Measured disagreement rate: **0 of 800** episodes in `results/e00_env_sanity.csv`
(0.0000, gate < 1 %), 0 of 100 scripted static-pad descents (max gap 4.2 ms), 0 of 30 scripted
descents on a moving SS5 deck.*

*Three filters the rate depends on, each of which was measured failing without it.*

- **`normalForce > 1e-4 N`.** Bullet reports candidate points up to ~2 cm before the surfaces
  touch (`contactBreakingThreshold`), by a distance that grows with closing speed.
- **`clearance ≤ analytic_contact_margin_m`** on the analytic side — see P2-D5 item 5; without
  it, 4 of 100 soft landings had a contact record and no analytic record.
- **The analytic footprint test is widened by the drone's collision radius** (0.06 m). A drone
  whose *centre* is past the plate edge can still strike it with its rim, and PyBullet reports
  that contact. This was the **single** disagreement in the first clean 800-episode sweep: a
  random-policy episode with the drone centre at 0.453 m against a 0.40 m half-extent, correctly
  classified `crash`/`off_plate_strike`. Widening the analytic footprint took the rate to 0/800.
  The record's `on_plate` flag stays on the **centre**, so that classification is unchanged.

*One PyBullet call per substep, not two.* `getContactPoints(bodyA=drone)` is read once and
partitioned into deck contacts and ground contacts, saving 240 round trips per simulated second.
Drone-versus-ground contact is a `crash` (`termination_reason: ground_contact`) and is never a
landing.

*Tunnelling is counted, not prevented.* `contactDistance < −5 mm` at first contact sets a
per-episode flag. Measured: 1 of 800 sanity episodes (−7.86 mm, a random-policy impact), 0 of
100 scripted 0.3 m/s descents. The flag discriminates rather than always firing, which is what
makes it worth reporting.

### P2-D7 — Termination, truncation, and the dwell grace (2026-09-22)

*Decision.* `timeout` is the **only** truncation: `truncated=True`, `terminated=False`, and
**exactly zero** terminal reward. `success`, `hard_landing`, `off_pad`, `bounce` and `crash` are
all `terminated=True`. Getting this backwards — either half of it — teaches PPO that the clock
is a cliff and that hovering it out is safe; `tests/test_landing_env.py` pins both halves.

*Episode budget: 12.0 s flight + 0.5 s dwell grace, model scale.* The grace is simulated time
granted **after** the flight budget so that a touchdown at 11.9 s can still complete its 0.5 s
dwell; it is not flight time and no policy can use it. Start offsets are therefore drawn from
`motion.episode_start_window_s(12.5)` = **(4.0, 107.5) s model**, the 4.0 s being the Phase 4
forecaster's reserved lookback. The deck grid is exactly `12.5 × 240 + 1 = 3 001` samples, so
the last requested time is exactly `t0 + 12.5 s` and never past the committed dmf record — one
sample more would raise at the top of the start window.

*First contact does not end the episode.* Its record is frozen and decides
`success`/`hard_landing`/`off_pad`; the episode then continues for the 0.5 s dwell. Contact lost
for longer than `contact_loss_grace_s` = 0.05 s is a `release` → `bounce`. Shorter flicker is
ignored, because Bullet drops and re-adds contact points on a settling body: the first drop test
counted **2** rising edges on a single clean landing before the grace was applied to the counter.
`EpisodeRecord.contact_events` counts grace-aware events, so "exactly one touchdown" is a
measurement rather than a property of the bookkeeping.

*`step()` is overridden.* `BaseAviary.step()` runs `PYB_STEPS_PER_CTRL` physics substeps with no
per-substep hook, and both the deck drive and the contact poll must run at 240 Hz — driving the
deck at 30 Hz fails the 1 mm tracking test outright, and polling contacts at 30 Hz mis-times a
fast approach. The override mirrors the base loop and adds, per substep: *advance the deck →
`stepSimulation` → poll both detectors*. `_computeReward/_computeTerminated/_computeTruncated/
_computeInfo` stay pure reads of state recorded in that loop, so the base class's call order is
intact.

*Determinism.* Same seed → bit-identical observations, asserted as the environment's **1st**
reset and again as its **3rd**. The three places this breaks are all handled explicitly:
`DSLPIDControl.reset()` is called on every reset (its integral state otherwise survives),
the noise generator is a **spawned child** of the episode seed sequence, and the episode start
offset and the initial drone state come from two further spawned children — so enabling noise
cannot shift the episode, which is what keeps the Phase 7 noise ablation a paired comparison.

### P2-D8 — Throughput with the deck in the loop, and what P3-D1 must size from (2026-09-22)

*Measurement.* `results/env_throughput_landing.csv`, 8 rows, same schema and method as P0-D2
(20 000 → 6 000 timed environment steps per row after a 50-vector-step warmup,
`OMP_NUM_THREADS=1`, 36 logical cores, DIRECT). `results/env_throughput.csv` is **untouched**:
the landing rows go to a new file so the committed Gate 0 artifact keeps its shape.

| env | policy | backend | n_envs | steps/s | per env | steps/episode | h per 10 M |
|---|---|---|---|---|---|---|---|
| landing | random | DummyVecEnv | 1 | 271 | 271 | 42.6 | 10.26 |
| landing | random | SubprocVecEnv | 1 | 233 | 233 | 42.6 | 11.95 |
| landing | random | SubprocVecEnv | 8 | 437 | 55 | 45.1 | 6.36 |
| landing | random | SubprocVecEnv | 16 | 514 | 32 | 46.9 | 5.40 |
| landing | hold | DummyVecEnv | 1 | 558 | 558 | 375.0 | 4.98 |
| landing | hold | SubprocVecEnv | 1 | 442 | 442 | 375.0 | 6.29 |
| landing | hold | SubprocVecEnv | 8 | 2 986 | 373 | 375.0 | 0.93 |
| landing | hold | SubprocVecEnv | 16 | **4 610** | 288 | 375.0 | **0.60** |

*A second policy row had to be added, and this is why.* P0-D2's `random` contract makes the
landing environment **reset-bound**, not step-bound. A uniformly random 1.5 m/s setpoint tumbles
the drone past the 60° crash threshold in ~43 control steps (1.4 s model), and a reset costs
**88.5 ms** — of which **64.3 ms (73 %)** is the analytic deck-motion evaluation over the
episode's 3 001-sample physics grid. In a *synchronous* vector env one worker's reset stalls all
of them: at 16 workers the chance that some worker resets in a given vector step is
1 − (1 − 1/47)¹⁶ ≈ 29 %, so ~110 of 375 vector steps carry an 89 ms stall, which accounts for the
whole 11.7 s wall clock and for the flat 233 → 514 scaling. The `hold` rows (a constant zero
action, i.e. hover to the time limit) run full 375-step episodes and scale **442 → 4 610,
10.4× on 16 workers**.

*What P3-D1 should size from.* The in-episode step cost is **1.45 ms** (690 steps/s
single-process, deck driven at 240 Hz and both detectors polled at 240 Hz). The honest budget is
**bracketed** by the two policy rows: early training resets often and lives near the `random`
row; a policy that survives its episode lives near the `hold` row. Against P0-D2's ceiling
(`HoverAviary`, `vel`, 16 workers, 7 575 steps/s) the landing environment is **1.64× slower**
per step at 16 workers — that is the cost of a second body driven every physics step, a
240 Hz two-detector contact poll, and a 25-entry observation. A 10 M-step PPO run is 0.60 h of
stepping at best and 5.40 h at worst, so the sweep is still not compute-bound, but the gap
between those two numbers is a **reset** cost and the way to close it is longer episodes, not
more workers.

*An optimisation deliberately **not** taken, so it is not rediscovered as an oversight.* The
64 ms bridge call is paid in full even by an episode that crashes after 43 steps. Evaluating the
deck lazily in chunks would avoid that, but it would change the summation order of the
superposition (see the correction below), so it is left out; if Phase 5 wants it, it needs its
own numerical-equivalence entry.

*Correction to the approved plan's addendum: the batched bridge call is **not** bit-identical.*
The addendum states "Precompute the deck trajectory per episode (one batched bridge call) …
Bit-identical, and a test asserts it." Measured: **0 of 20** probed samples match bit for bit.
The superposition ends in a matrix product, and BLAS selects a different kernel — hence a
different summation order over the 299 wave components — for a `(1, 299)` operand than for a
`(3001, 299)` one. The measured agreement is **≤ 9.2e-17 m** and **≤ 2.0e-17 m/s** on quantities
of order 2 m and 0.4 m/s, i.e. **under one ULP of float64**. Nothing downstream changes: the
environment always evaluates the same 3 001-sample batch, so it is bit-deterministic, and the
determinism test passes. The claim that had to be weakened is the *equality*, not the
*reproducibility*, and `tests/test_landing_env.py` now asserts the measured tolerance.

### P2-D9 — Environment API, sanity outcomes, and corrections carried forward (2026-09-22)

*The API every later phase is written against.*

- **Action:** `Box(-1, 1, (3,), float32)`, dimensionless, world-frame velocity setpoint. See
  P2-D2 for the scaling and the norm cap.
- **Observation:** `Box(float32, (25,))`, finite bounds everywhere, clipped into them. Declared
  as a table (`rld.envs.observation.OBS_FIELDS` via `obs_fields(cfg)`) so Phase 4 appends its
  forecast block without renumbering anything:
  `attitude` (3, rad, world) · `body_rates` (3, rad/s) · `velocity` (3, m/s, drone-yaw) ·
  `rel_position` (3, m, drone-yaw) · `rel_velocity` (3, m/s, drone-yaw) · `deck_normal`
  (3, drone-yaw) · `rel_tilt` (1, rad) · `height` (1, m, signed clearance along the deck normal)
  · `time_fraction` (1, [0,1]) · `last_action` (3, [-1,1]) · `in_contact` (1, {0,1}).
  "drone-yaw" is the world frame rotated by −yaw about z, **not** the full body frame: a tilted
  drone's relative-position block must not swing with its attitude.
- **`last_action` and `in_contact` are in the observation on purpose.** The smoothness reward
  term depends on `a_t − a_{t−1}`, so without `a_{t−1}` the reward is not a function of the
  observed state and the MDP is not Markov; and the 0.5 s dwell criterion is not representable
  without a contact flag.
- `gymnasium.utils.env_checker.check_env` passes.

*Sanity outcomes (`results/e00_env_sanity.csv`, 4 rows; `…_episodes.csv`, 800 rows).* The
expected shape was pre-registered in the module docstring before the run, and is what came out:

| policy | SS | crash | off_pad | hard_landing | bounce | success | timeout | disagree | tunnel |
|---|---|---|---|---|---|---|---|---|---|
| hover | SS3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | 0/200 | 0 |
| hover | SS5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | 0/200 | 0 |
| random | SS3 | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/200 | 0 |
| random | SS5 | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/200 | 1 |

A random policy scores **zero** successes, which is the point of printing the breakdown: a
random policy that "succeeded" would mean the criteria or the contact logic were wrong. The
`crash` bar decomposes through `termination_reason`: 394 `tilt_gt_crash`, 3 `out_of_bounds`,
2 `below_deck`, 1 `off_plate_strike` — i.e. a random 1.5 m/s setpoint tumbles the vehicle long
before it reaches the deck, and only one of 400 random episodes touched the deck at all.

Artifacts: `results/e00_env_sanity.csv` (4 rows, 53 columns), `results/e00_env_sanity_episodes.csv`
(800 rows, 53 columns), `results/env_throughput_landing.csv` (8 rows), all with the standard
provenance block; the sweep takes 27 s on 24 workers and its output does not depend on
`--workers` (asserted at 1 and 2 workers).

*Corrections carried, not edited.*

1. **`rld.bench.throughput`'s docstring was wrong about when `DSLPIDControl` runs**, and P0-D2's
   reasoning quotes it. It said the `vel` path runs the PID "at the physics rate (240 Hz) rather
   than the control rate (30 Hz)". It does not: `BaseRLAviary._preprocessAction` calls
   `computeControl` **once per control step** and holds the RPMs across the eight physics
   substeps, and `DeckLandingAviary` does the same. The measured numbers in P0-D2 stand; only
   the explanation was wrong. The docstring now carries the correction inline, pointing here.
2. **The `pybullet-moving-platform` skill's driver preference is backwards** on this project's
   numbers — see P2-D1.
3. **The approved plan's addendum overstates batched-bridge equality** — see P2-D8.
4. **The plan's "drone dropped from rest onto a static pad" is not expressible in this action
   space.** Every action is a velocity setpoint tracked by a stabilising PID, so there is no
   "motors off". The test uses the maximum commanded descent (1.5 m/s) instead, which is a
   genuinely fast arrival: it registers exactly one touchdown, classifies it, and **does**
   exceed the 5 mm tunnelling threshold — the flag is asserted to equal the recorded
   penetration against the frozen threshold rather than asserted away.

*Open physics concerns, recorded rather than fixed.*

- **Reset cost dominates short episodes** (P2-D8). It is inherent to the architecture rule that
  deck motion is evaluated analytically on the physics grid with no interpolation.
- **Initial lateral spread exceeds the plate's beam.** `init.lateral_m` = ±0.3 m against a plate
  half-width of 0.26 m, so ~13 % of episodes start laterally *outside* the plate's footprint (in
  the air above the sea). That is legitimate — the drone must fly back over the pad — but it
  means a pure vertical descent from the initial state is not a landing, and any Phase 3
  baseline must null the lateral error first.
- **The `off_pad`/`crash` boundary is the plate edge, and the plate is not square.** A landing
  with the drone centre beyond 0.26 m in y (or 0.40 m in x) is a deck-edge impact and classifies
  as `crash`, not `off_pad`. Both bars are reported, so the split is visible.
- **The two gravities still differ** (dmf 9.80665, gym-pybullet-drones 9.8; P1-D1). The plate's
  weight cancellation reads gravity back from PyBullet, so it is exact in the client's own terms.

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
| 0 | 2026-09-21 | PASSED | `make test lint` green (6 tests, 53 s); pybullet 3.2.7 built from sdist and opens a DIRECT client on 3.12.3; `results/env_throughput.csv` written (8 rows, 1/8/16 SubprocVecEnv workers x rpm/vel); both submodule SHAs recorded in P0-D1. |
| 1 | 2026-09-21 | PASSED | `make test lint` green (66 tests, 62 s; ruff + ruff-format + mypy --strict clean). `results/deck_stats.csv` 192 rows = 96 cells x {aft, cg}, full grid (2 vessels x SS3-SS6 x 4 headings x 3 speeds), all 2304 realizations, with `z_std_model_m`, `vz_std_model_m_s`, `vz_p99_model_m_s`, `az_p99_model_m_s2` populated and no NaN; plus `deck_stats_seeds.csv` (4608 rows) and `deck_feasibility.csv` (2 rows). lambda = 1/25 and r_pad = -0.4*L confirmed in P1-D1. Feasibility rule PASS: frigate SS6 180 deg 12 kn aft vz p99 = 0.577165 m/s vs the 2.08333 m/s gate threshold, 3.61x inside; the rejected `SPEED_LIMIT` reading is recorded as FAIL beside it. Sign convention and ZYX rotation order corrected and pinned by three hand-computed cases (P1-D2); one project-wide lambda (P1-D3). |
| 2 | 2026-09-22 | PASSED | `make test lint` green (118 passed, 1 skipped, 107 s; ruff + ruff-format + mypy --strict clean). The 1 skip is by design: `tests/test_platform.py` gates only the configured driver and *measures* the other, and `test_constraint_driver_fails_the_tracking_gate` asserts the rejected one fails. Platform tracking, `kinematic` driver, 10 s model at frigate SS6 180 deg 12 kn aft: body-origin 0.000 mm and plate-corner 2.2e-16 m against the 1 mm gate, orientation 3.0e-8 rad, `getBaseVelocity` linear and angular ratio 0.0 % against the 2 % gate; `constraint` fails every one of those by an order of magnitude and cannot be tuned into passing (24-point sweep, four identical digits) -- P2-D1. `gymnasium.utils.env_checker.check_env` passes; reset determinism bit-identical as the env's 1st and 3rd reset; drop on a static pad registers exactly one touchdown; scripted 0.3 m/s descent on a static pad 100/100 `success`. `results/e00_env_sanity.csv` (4 rows) + `results/e00_env_sanity_episodes.csv` (800 rows) written from the `id` split's val partition, draw seed 20260922: hover 1.000 `timeout` at SS3 and SS5, random 1.000 `crash` at both, no successes under either -- outcome fractions sum to 1.000000 per row. **PyBullet-vs-analytic touchdown disagreement 0/800 = 0.0000** against the < 1 % gate, plus 0/100 on scripted static descents and 0/30 on a moving SS5 deck. `results/env_throughput_landing.csv` (8 rows) records the deck-in-the-loop throughput P3-D1 must size from (P2-D8); `results/env_throughput.csv` untouched. Two flags carried forward, both recorded rather than fixed: one of 800 sanity episodes tunnelled (7.86 mm penetration vs the 5 mm threshold) and it was the only random episode that ever reached contact, so the random arm exercises the touchdown path barely at all -- the scripted tests carry that load; and the plan's "dropped from rest" is not expressible in a velocity-setpoint action space, so the drop test uses the maximum commanded descent (P2-D9). |
