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

### P3-D1 — FROZEN EVALUATION PROTOCOL (2026-09-22)

**Status: revision 1 (2026-09-22), awaiting re-freeze at Gate 3.**
- The user approved the first draft and it was marked frozen.
- The first Gate 3 run then **failed** on the `results-skeptic` review: one BLOCKING finding (H1)
  and three MAJOR findings.
- No RL run has started. The block was revised with the user's decisions, as logged in §9.
- On re-freeze, nothing below changes after the first Phase 5 training run without a dated
  deviation entry.
- The SHA-256 of this block (from this heading to the line before P3-D2, UTF-8) is recorded in
  the Gate 3 row.

**1. Success criteria.** `configs/env/success.yaml`, SHA-256
`c7fbdcc4df48bd4a2cfe21811613e5e50e3e4868822fe6dddd13745d96692118`. The file (comments included)
is the authority; its keys, verbatim:

```
rel_vertical_velocity_max_m_s: 0.5          contact_dwell_min_s: 0.5
vz_component: deck_normal                   contact_loss_grace_s: 0.05
lateral_offset_max_m: 0.10                  crash_tilt_deg: 60.0
relative_tilt_max_deg: 15.0                 crash_tilt_component: absolute
hard_landing_includes_relative_tilt: true   tunnelling_penetration_m: 0.005
contact_normal_force_min_n: 1.0e-4          analytic_contact_margin_m: 0.001
detector_disagreement_window_s: 0.0333333333333333
```

The six outcome classes are matched in order `crash → off_pad → hard_landing → bounce → success →
timeout`. `timeout` is the only truncation. The P2-D5 resolutions stand.

**2. Episode lists.** These lists are committed in `0780aaa5dc4dd7e1c20fdeed48d405a1e45a4f80` on
their own commit, before any controller was evaluated on them.
- Generator seed **20260924**.
- Draw rule `balanced_round_robin`: per cell, the regime's dmf test realizations are shuffled and
  episodes are dealt round-robin, so the per-realization spread is ≤ 1. SS6 in `unseen_seastate`
  has 480 realizations, so 200 of them get one episode each.
- Aft pad.
- N = 200 per cell, 13 cells, plus a 200-episode static-pad list.
- The pad-at-CG and sinusoid arms (Phase 7) fly the **same** lists.
- Content SHA-256 is over a canonical CSV serialisation. `results/episodes/MANIFEST.csv` is the
  authority, with SHA-256
  `e6f30e55e478d39061b94e38de33959ca99b16e8cac38a3bd99b34f6543ad4a2`.
  `scripts/make_episodes.py --check` re-verifies both hashes of every list.

| file | cells | rows | file SHA-256 | content SHA-256 |
|---|---|---|---|---|
| id.parquet | SS3–SS6 | 800 | `919be99d…c1926dc64d` | `ea38d8a5…a8dde5769017` |
| unseen_seastate.parquet | SS6 | 200 | `b8cf656e…fd36524f` | `9fdd5ae9…b91d90a` |
| unseen_heading.parquet | SS3–SS6 | 800 | `1eba3df5…cbde0ee` | `2259c0c6…bb325499` |
| unseen_vessel.parquet | SS3–SS6 | 800 | `3b9293fe…20b87d26` | `3f84623d…a4cb0` |
| static.parquet | static | 200 | `90741add…0ca1c` | `7818bda5…be732` |

The pools are defined in P3-D2. Train is 729 realizations and tune is 135, both from `dev_pool()`,
and neither shares a realization with any list. `id` SS6 and every 90° episode are flagged
`in_training_distribution = false`, which is 0.4375 of `id`; 0.5625 is flagged true.

The regimes are **not independent samples**. Their test partitions share realizations:
- `id` ∩ `unseen_heading`: 96;
- `id` ∩ `unseen_seastate`: 37;
- `unseen_heading` ∩ `unseen_seastate`: 50.

This is not leakage, since every one of them is a test realization. But a regime-vs-regime
difference is not a comparison of independent draws, and it is never reported as one.

**3. Metrics, per (method, seed, regime, SS) cell.** Every cell reports:
- success rate with a Wilson 95 % CI and N;
- all six outcome-class fractions, which sum to 1;
- `termination_reason` counts;
- touchdown closing speed along the deck normal, p50 and p95, which is the criterion-1 quantity;
  the world-z reading is reported beside it;
- lateral offset p50 and p95;
- relative tilt p95;
- time to touchdown p50;
- control effort (mean ‖a‖², normalised units) and action jerk (mean ‖aₜ − aₜ₋₁‖);
- analytic-vs-contact detector disagreement, n and rate;
- tunnelling n;
- the `in_training_distribution` fraction.

**Primary metric:** success rate. **Primary secondary metric:** p95 closing speed.

**4. Statistics.**
- **Per cell and per seed:** a Wilson 95 % CI.
- **Across the 5 training seeds:** rliable IQM and optimality gap (from `rliable.metrics`), with a
  stratified-bootstrap 95 % CI. It uses **2 000** replicates, bootstrap seed **20260926**, and
  resamples runs within each task, never tasks.
- **Method contrasts (H1–H4):** a paired bootstrap on per-episode differences over the identical
  episode lists. It resamples seeds, then episodes, with **10 000** replicates and seed 20260926.
- **Closing-speed contrasts (H1 and H3)** use a paired **relative-p95** statistic:
  - *Pairing.* Each bootstrap replicate resamples seeds (per method), then resamples one set of
    episode indices shared by both methods.
  - *p95.* Within the replicate, each method's p95 closing speed is taken over **that method's
    touched-down episodes only**. An episode with no touchdown (`timeout`, or a `crash` before
    contact) has no closing speed; it is excluded from that method's p95, never imputed. Its
    cost is captured by the success criterion that each hypothesis also scores.
  - *Statistic.* r = 1 − p95(method) / p95(reference).
  - *Judged on.* The magnitude threshold ("≥ 15 %", "≥ 10 %") is judged on the **point
    estimate**. "Separates" requires the 95 % CI of r to exclude 0.
  - *Implementation.* Phase 7 implements this in `rld.eval.stats` with a unit test on a synthetic
    case with a known r. Today's `paired_bootstrap` (difference of means) is not this statistic.
- A difference "separates" only when the 95 % CI of the difference excludes 0. Overlapping CIs are
  never reported as a result.
- Success is never pooled across sea states in a headline. Classical baselines are deterministic
  and have one "seed".
- **Deviation from plan §5 (tooling, not substance):** `rliable.library` cannot be imported in the
  pinned environment (`arch` 7.2.0 fails at import against pandas 3.0.5). The stratified bootstrap
  is therefore implemented in `rld.eval.stats` and unit-tested. rliable's own implementation also
  draws from NumPy's global random state, so its intervals would not be reproducible.

**5. Training budget.** Sized from `results/env_throughput_landing.csv` (P2-D8). At 16 workers the
env runs at 514 steps/s under a policy that resets constantly (the `random` row) and 4 610 steps/s
under one that flies full episodes (the `hold` row). Both figures exclude the PPO update.

| method family | env steps / run | seeds | runs | total steps |
|---|---|---|---|---|
| PPO: `ppo`, `residual_ppo`, `ppo_forecast`, `residual_ppo_forecast`, `ppo_sinusoid` | **10 M** | 0–4 | 25 | 250 M |
| SAC: `sac` | **2 M** | 0–4 | 5 | 10 M |

Wall clock for the PPO family is 250 M steps at 4 610 steps/s = **15 h** best case, or at
514 steps/s = **135 h** worst case, as one stream. Two concurrent 16-worker runs on 36 cores
bring that to about **7.5–68 h**. Two facts support budgeting on the optimistic side:
- the residual methods start from `pid_feedforward`, which lands in about 4.5 s (≈ 135 steps);
- a crashing pure-PPO policy lives near the `random` row only early in training.

The ~135 h worst case is stated, not hidden. **Pre-registered budget rule:** the first PPO seed's
wall clock is reported. If the projected total exceeds 5 days, the budget may be cut **once**, by a
dated deviation that applies the same factor to every PPO-family method. The cut must be made
before any method's evaluation results are read.

**Hyperparameter tuning:**
- **PPO and SAC:** ≤ **20 trials** each, on the P3-D2 **tune pool only**. Each trial trains at
  **2 M** steps (PPO) or **0.5 M** steps (SAC) on one seed, and is scored by tune-pool success on
  SS3–SS5 with ties broken by p95 closing speed.
- **`residual_ppo`, `ppo_forecast`, `residual_ppo_forecast` and `ppo_sinusoid`** inherit PPO's
  tuned hyperparameters and get no search of their own. Any bias from this runs *against* the
  residual and forecast methods, since the settings were tuned for pure PPO.
- **The classical baselines** spent their 20 trials each in P3-D3. `pid_feedforward_lowvz` spent
  none: it is selected from the same log by a rule (§8, H1).
- **SAC's budget is not equal to PPO's** (2 M vs 10 M env steps), as plan §5 intends for an
  off-policy method. Every PPO-vs-SAC statement carries that confound, stated beside it.

**6. Seeds, α, curriculum.**
- **Seeds:** training seeds 0–4 for every learned method, and none is ever dropped.
- **Residual α:** **α = 0.3** in normalised action units (0.45 m/s at `v_max` = 1.5 m/s). The
  composed action is `clip(a_pid_ff + α·π(o))` through the env's norm cap, and the policy's last
  layer is zero-initialised.
- **Curriculum:** SS3 → SS4 → SS5. A stage is promoted when rolling tune-pool success is
  **≥ 0.80 over ≥ 100 episodes** at the current sea state. SS6 is never trained on.

**7. Baselines on the frozen lists (committed as `results/e01/`, read before the hypotheses below
were written).**
- `pid_feedforward` succeeds on 200/200 static, 200/200 at `id` SS3, 200/200 at SS4, 198/200 at
  SS5 (99.0 % [96.4, 99.7]) and 181/200 at SS6 (90.5 %). Its p95 closing speed at `id` SS5 is
  0.262 m/s.
- `pid_track_descend` scores 85.0 % at `id` SS5.
- `gated` and `oracle_gated` sit **below** `pid_feedforward` from SS5 up, and all of their losses
  are timeouts. `oracle_gated` is an upper bound on **commit timing under the quiescence rule**,
  not on success. It is labelled that way in every table.

**Consequence for the plan's H1.** Plan H1 predicts a residual gain of ≥ 10 points over
`pid_feedforward` at `id` SS5. The baseline sits at 99.0 %, so that gain is **arithmetically
impossible**. The hypotheses below are restated against the measured ceiling. The change is made
here, before any RL run, and this paragraph records why.

**8. Hypotheses (direction, magnitude, test).** Each is scored in Phase 7 as supported, not
supported or inconclusive, with the number.

- **H1 (residual RL improves on what gain-tuning alone can do).** Cell: `id` SS5.
  - *Why a second reference.* A PID that simply descends more slowly already lands softer:
    tuning trials 10, 5 and 15 of `pid_feedforward` cut pooled tune-pool p95 by 27–38 % at
    98–99 % success. So a closing-speed gain over the success-tuned `pid_feedforward` would not
    show that RL adds anything.
  - *The reference.* H1 is therefore scored against **`pid_feedforward_lowvz`**.
    - *Selection rule.* It is chosen from the existing 20-trial P3-D3 log: among trials whose tune
      success is ≥ best − 0.02, take the lowest pooled p95 closing speed, with the lowest trial
      index breaking ties.
    - *What it selects.* **Trial 10**: 178/180 tune success, pooled p95 0.153 m/s, descent
      0.111 m/s.
    - *When the rule was fixed.* It was worded after the tuning log existed, but before any
      frozen-list number for any candidate was seen.
    - *A wrong prediction, corrected.* The revision-1 text predicted "trial 15", a prediction
      carried over from the skeptic's review. The user kept the **rule** (2026-09-22) rather than
      re-tuning the cutoff to reach the predicted trial.
    - *Budget and reporting.* No tuning budget was added. It is printed beside every learned
      method.
  - **H1a (closing speed, scored).** `residual_ppo` lowers p95 closing speed relative to
    `pid_feedforward_lowvz` by **≥ 15 %**, as the relative-p95 point estimate from §4, with its
    95 % CI excluding 0.
  - **Non-inferiority.** The lower bound of the 95 % CI of success(`residual_ppo`) −
    success(`pid_feedforward_lowvz`) must be **≥ −2 points** (paired bootstrap).
  - *Scoring for H1a:*
    - **Supported:** point estimate ≥ 15 %, the r CI excludes 0, **and** non-inferiority holds.
    - **Not supported:** non-inferiority fails, **or** the r CI includes 0, **or** r < 0 (residual
      lands harder).
    - **Inconclusive:** r separates from 0 but its point estimate is below 15 %, with
      non-inferiority holding.
  - **H1b (success where the baseline has room, scored).** On `id` SS6, `residual_ppo` exceeds
    **`pid_feedforward`** in success by **≥ 5 points** (paired bootstrap).
    - **Supported:** point estimate ≥ 5 points and the CI excludes 0.
    - **Not supported:** the CI includes 0 or the difference is negative.
    - **Inconclusive:** a positive, separating difference below 5 points.
    - The success-tuned `pid_feedforward` is the right reference here, because it is the stronger
      of the two on success.
- **H2 (residual degrades less under sea-state shift).** For a method, "drop" = IQM success at
  `id` SS5 − IQM success at `unseen_seastate` SS6. Prediction: drop(`ppo`) − drop(`residual_ppo`)
  ≥ **10 points**. Test: bootstrap of the difference of drops. It resamples seeds, and resamples
  episodes independently within each cell, because the two cells are different episodes.
- **H3 (forecasts help, then help less).** On `id` SS5 and SS6, `ppo_forecast` lowers p95 closing
  speed relative to `ppo` by **≥ 10 %**, using the relative-p95 statistic of §4 with the same
  point-estimate / CI rule as H1a.
  - Under `unseen_vessel` (same SS), that relative gain is **≤ half** of its `id` value.
  - The same two tests are reported for `residual_ppo_forecast` vs `residual_ppo` as a secondary.
- **H4 (motion realism, the novelty test).** Cell: `id` SS5. Define:
  - drop_sin = success(`ppo_sinusoid` on sinusoid) − success(`ppo_sinusoid` on JONSWAP);
  - drop_jon = success(`ppo` on JONSWAP) − success(`ppo` on sinusoid).

  Prediction: drop_sin − drop_jon ≥ **10 points**, using the same episode lists for both test
  motions. **If the CI includes 0, the novelty claim is withdrawn in the README (D0.4).**
- **H5 (deployment).** At batch 1, the exported policy MLP's p50 latency on ORT CPU (1 thread) is
  lower than on every parity-passing GPU provider, by a factor of **≥ 2×**. Test: Project 4's
  harness, 200 warmup + 2 000 timed runs. Scored on p50, with p99 reported.

**9. Deviations and incidents before freeze.**
1. **A smoke check ran before the lists were committed (corrected in revision 1).**
   - *Timeline, from file mtimes and process start times, all local time on 2026-09-22.* The
     controller gain YAMLs were last written at **17:03:29**, and nothing under `src/rld/control/`
     or `configs/control/` changed after that until the Gate 3 remediation. The superseded
     uniform-draw lists were written at **16:58:28**. The eval-auditor's runner smoke check started
     at about **17:10**. The balanced lists overwrote the superseded ones at **17:24:00** and were
     committed as `0780aaa` at **19:11**. `results/e01/` was written at **19:19:48**.
   - *What was flown.* 4 controllers × static indices 0–7 and `id` SS3 indices 0–7 (64
     controller-episodes). The output went to the agent's terminal only. Nothing was written to
     `results/`, and nothing was tuned from it.
   - *Which lists.* The smoke check read the lists at about 17:10, so it read the **superseded**
     ones. Their `id` SS3 episodes differ from the committed ones in their realization
     assignment. **The static list was not changed by the redraw**: its file hash `90741add…` is
     identical in both. So **32 of those controller-episodes were committed static episodes**,
     flown before the commit. The earlier text of this item said "about 1 h" after the gains and
     "no committed episode was flown". Both were wrong.
   - *Consequence: none.* The gains were fixed 7 minutes before the smoke check, and the static
     pad has no deck motion.
2. **A hung smoke-check process.** A second smoke-check invocation, started at about 17:10, hung.
   Its parent process kept respawning worker processes for about 2.4 h, with its output going to
   a closed pipe. The main thread terminated that process tree only at about 19:35. It had read
   the superseded lists at start-up. No output of it was ever read or written.
3. **The plan's H1 is restated** (item 7 above).
4. **Revision 1 (2026-09-22), after the first Gate 3 run failed on the `results-skeptic` review.**
   The user chose each fix.
   - **B1 (H1 beatable by a slower PID).** H1 is split into H1a, scored against the rule-selected
     `pid_feedforward_lowvz`, and H1b, success at `id` SS6. The scoring is made explicit.
   - **M1 (`oracle_gated` stricter than `gated`).** The oracle now applies `gated`'s own 30 Hz,
     12-sample predicate to the true future window, and has been re-evaluated.
   - **M2.** This section was rewritten with exact times.
   - **M3.** The relative-p95 statistic, NaN handling and the judging rule are specified in §4.
   - **Minor corrections:** the `in_training_distribution` fraction; the "all timeouts" sentence
     in §7; the MANIFEST SHA-256; the SAC budget confound; the regime overlap.

### P3-D2 — One training pool for all four regimes: the intersection of dmf's train partitions (2026-09-22)

*Decision (user, 2026-09-22).* Every learned method and every tuned baseline draws its
realizations from **one** development pool, `rld.deck.splits.dev_pool(sim_cfg) -> (train, tune)`:

| pool | realizations | vessel | sea states | headings | speeds | seed ordinals |
|---|---|---|---|---|---|---|
| train | 729 | frigate | SS3, SS4, SS5 | 45°, 135°, 180° | 0, 6, 12 kn | 0–26 |
| tune | 135 | frigate | SS3, SS4, SS5 | 45°, 135°, 180° | 0, 6, 12 kn | 27–31 |

`train` is the intersection of the four regimes' `train_keys`; `tune` is the intersection of
their `train_keys ∪ val_keys`, minus every regime's `test_keys`, minus `train`. Both are
*derived* from `build_all_splits`, not hard-coded, and `tests/test_deck_splits.py` asserts the
sizes, the seed ranges, and disjointness from every regime's test set. `tune` is used for PID gain
tuning, RL hyperparameter trials and curriculum validation. Evaluation episodes come from each
regime's dmf **test** partition, unchanged.

*Reason.* dmf builds the four regimes as four separate splits, and they overlap one another. With
40 frigate seed ordinals, `id` holds out 32–39 for test and 27–31 for val, while
`unseen_seastate`, `unseen_heading` and `unseen_vessel` train on 0–33 and validate on 34–39. So:

- the three shift regimes' train sets contain `id`-test realizations (seeds 32–33); and
- `id`-val contains SS6 and 90° realizations that are `unseen_seastate`-test and
  `unseen_heading`-test.

One policy trained on `id`-train and scored on all four regimes would therefore evaluate on
realizations it was trained or tuned on. That breaks CLAUDE.md non-negotiable 2. Training one
policy per regime would fix this too, but at 4× the RL compute; the intersection gives the same
guarantee with one policy per method × seed.

*Consequence, stated so it is not rediscovered as an anomaly.* The `id` list keeps dmf's
definition (seeds 32–39, all four sea states, all four headings). Its SS6 cells and its 90° cells
are therefore **outside the training distribution** of every learned method. Every episode row
carries an `in_training_distribution` flag, so `id` is reportable both as dmf defines it and
restricted to trained-on cells.

*Correction to P2-D4, carried rather than edited.* P2-D4 says the e00 sanity episodes (`id`-val,
all headings) are "neither trained on nor part of the Phase 3 frozen evaluation lists". Under
dmf's regimes, the 90° realizations among those draws are `unseen_heading`-test realizations. No
harm follows, because `hover` and `random` are null policies that nothing was fitted to. But the
sentence is not true as written, and from Phase 3 on the null-policy sweep's pool is not the
tuning pool.

### P3-D3 — Baseline controller tuning (2026-09-22)

*Decision.* `pid_track_descend` and `pid_feedforward` are tuned **with the same budget, the
same search space on their shared parameters and the same episodes**, on the P3-D2 **tune pool
only**, so the naive baseline is not strawmanned against the residual-RL base. `gated` and
`oracle_gated` are **not** tuned. They inherit `pid_feedforward`'s gains by reference
(`gains_from: pid_feedforward.yaml`, never copied), and their thresholds are dmf's. The
procedure is `rld.control.tuning` + `scripts/tune_controller.py`, driven by
`configs/control/tuning.yaml`.

*Procedure.*

- **Pool.** `dev_pool(sim_cfg)[1]`: frigate, SS3–SS5, headings 45/135/180° (never 90°),
  0/6/12 kn, seed ordinals 27–31 (135 realizations). The aft pad. No episode is drawn from
  `results/episodes/` or from any regime's test partition.
- **Draw.** 60 episodes per sea state × SS3–SS5 = **180 episodes per trial**. The committed
  tuning seed is **20260923** (P2-D4's sanity draw is 20260922). It is mixed with a SHA-256
  label per sea state, so the draw is independent of the controller and the trial. Every
  trial of both controllers flies the **identical** 180 episodes (paired).
- **Budget.** **20 trials per controller, 20 spent by each** (3 600 episodes each). Trial 0
  is the hand-set point the controllers were written with. Trials 1–19 are the first 19 points
  of **one** scrambled Sobol sequence of dimension 5 (`scipy.stats.qmc.Sobol`, seed 20260923,
  `random_base2(5)` truncated). `pid_track_descend` reads coordinates 1–4 and ignores the
  fifth, so both controllers were evaluated at the **same** (kp, ki, radius, descent) points;
  `pid_feedforward` additionally varies `k_ff`.
- **Ranges** (linear; model scale): `kp_xy_per_s` [0.5, 3.0] (m/s)/m; `ki_xy_per_s2` [0, 1]
  (m/s)/(m·s); `commit_radius_m` [0.02, 0.10] m; `descent_rate_m_s` [0.08, 0.60] m/s;
  `k_ff` [0.5, 1.2] (feedforward only). Fixed, not searched: `integral_limit_m_s` 0.1,
  `lateral_speed_max_m_s` 0.5, `release_factor` 2.0.
- **Objective.** Mean success over SS3–SS5 (equal weight). Ties are broken by the lower
  **pooled p95 touchdown closing speed along the deck normal**, then by the lower trial index.
- **Reproducibility.** The measured columns of `results/e01/tuning_pid_feedforward.csv` are
  identical when re-run at 7 workers instead of 30. The `--final` run
  (`results/e01/tune_pool_final.csv`) reproduces both selected trials' numbers exactly.

*Final gains* (committed at full precision; `tests/test_control_pid_*.py` asserts the YAML equals
the log's `selected` row):

| controller | trial | kp_xy (1/s) | ki_xy (1/s²) | commit radius (m) | descent (m/s) | k_ff |
|---|---|---|---|---|---|---|
| `pid_track_descend` | 7 | 1.18205 | 0.11468 | 0.02868 | 0.25068 | — |
| `pid_feedforward` | 4 | 1.95602 | 0.95414 | 0.03446 | 0.20716 | 1.08174 |

*Tune-pool results at the final configs* (`results/e01/tune_pool_final.csv`, the same 180
episodes, n = 60 per cell; closing speed = p95 touchdown closing speed along the deck normal,
m/s model; lateral = p95 lateral offset, m model; ttd = mean time to touchdown, s model):

| controller | SS | success | crash | off_pad | hard | bounce | timeout | closing p95 | lateral p95 | ttd |
|---|---|---|---|---|---|---|---|---|---|---|
| `pid_track_descend` | SS3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.317 | 0.037 | 4.51 |
| `pid_track_descend` | SS4 | 0.983 | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 | 0.436 | 0.033 | 4.53 |
| `pid_track_descend` | SS5 | 0.917 | 0.000 | 0.000 | 0.017 | 0.067 | 0.000 | 0.480 | 0.040 | 4.62 |
| `pid_feedforward` | SS3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.220 | 0.031 | 4.45 |
| `pid_feedforward` | SS4 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.246 | 0.033 | 4.48 |
| `pid_feedforward` | SS5 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.252 | 0.030 | 4.56 |
| `gated` | SS3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.219 | 0.042 | 3.22 |
| `gated` | SS4 | 0.983 | 0.000 | 0.000 | 0.000 | 0.000 | 0.017 | 0.243 | 0.042 | 3.97 |
| `gated` | SS5 | 0.933 | 0.000 | 0.000 | 0.000 | 0.000 | 0.067 | 0.240 | 0.038 | 4.79 |
| `oracle_gated` (privileged) | SS3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.216 | 0.042 | 3.37 |
| `oracle_gated` (privileged) | SS4 | 0.983 | 0.000 | 0.000 | 0.000 | 0.000 | 0.017 | 0.236 | 0.046 | 3.86 |
| `oracle_gated` (privileged) | SS5 | 0.883 | 0.000 | 0.000 | 0.000 | 0.000 | 0.117 | 0.236 | 0.036 | 4.69 |

Touchdown-detector disagreement is 0 of 720 across these rows. Tunnelling occurred once
(`pid_track_descend`, SS5). Across all 40 trials (7 200 episodes) there were 2 disagreements,
both in `pid_feedforward` trials, at most 1 per trial. There were 76 tunnelling flags: 71 in
`pid_track_descend` trials and 5 in `pid_feedforward` trials. The worst trial is trial 1
(descent 0.586 m/s), with 25 of 180 for `pid_track_descend` and 3 for `pid_feedforward`.
Tunnelling tracks the closing speed. It is Phase 3's first volume test of P2-D6's flag and is
reported, not filtered.

*What these numbers are, and what they are not.*

1. **They are selection-biased.** The winners were chosen on this exact draw, so tune-pool
   success is an optimistic estimate. The frozen `id` lists (P3-D1) are the measurement.
   Nothing here is an evaluation result.
2. **Tuning did not improve `pid_feedforward`'s success. Only the tie-break moved.** Its
   hand-set trial 0 already scored 1.000, and **4 of 20 trials** (0, 2, 4, 17) tied at 1.000.
   The selection among them was made by the pre-registered tie-break, p95 closing speed
   0.245 m/s (trial 4) against 0.297–0.380 for the other three. The tune pool is **saturated**
   for this controller, so the ≥ 95 % Gate 3 criterion is not in doubt on the tune pool, but
   the pool cannot rank the top of the search space on success. `pid_track_descend` improved
   from 0.928 (trial 0) to 0.967 (trial 7), a single winner.
3. **The two failure modes of the naive controller bracket its descent rate.** Fast descents
   (≥ 0.46 m/s) lose to `hard_landing`, because the closing speed is `descent + v_pad,z`
   whenever the pad rises. Slow descents (≤ 0.15 m/s) lose to `bounce` (up to 0.47 at SS5),
   because without feedforward a landed drone commanded to −0.11 m/s in the world frame cannot
   follow a deck that falls faster, and it lifts off. `pid_feedforward`'s `k_ff·v_pad` removes
   both.
4. **`oracle_gated` is an upper bound on commit timing under this rule, not on success.** With
   `fallback_commit_s: null`, every failure of both gated controllers is a `timeout`: the deck
   was not quiescent in time. The oracle waits for a whole true 0.4 s window at its predicted
   touchdown, sampled at 240 Hz. The current-state rule fires on 12 control samples at 30 Hz.
   So the oracle can time out *more* often (SS5: 7/60 against 4/60), while landing more gently
   (pooled closing p95 0.233 against 0.238 m/s). Success does not require quiescence: an
   ungated feedforward descent lands 60/60 at SS5. Tables must not read `oracle_gated`'s
   success as a ceiling on the other controllers'.
5. **Quiescence base rate on the tune pool** (aft pad, model scale, permissive; fraction of
   start instants whose next 0.4 s is entirely inside all three limits): SS3 0.80 / 0.99 / 0.90,
   SS4 0.57 / 0.78 / 0.49, SS5 0.19 / 0.39 / 0.21, at 45° / 135° / 180°. The binding limit at SS5
   is the pad `v_z` in head seas (in limits 54 % of the time at 180°) and roll at 45° (51 %).
   The gated controllers' timeout fraction is therefore expected to rise sharply at SS6 (not
   measured here; SS6 is never in the tune pool) and under `strict`.

*The quiescence thresholds and their conversion.* The thresholds are **imported** from
`dmf.eval.quiescence` (`PERMISSIVE`, the default, and `STRICT`), not copied. They are converted
by `rld.control.quiescence.froude_limits` through `FroudeScale` at λ = 1/25 from
`configs/deck/scaling.yaml`:

| set | roll (°) | pitch (°) | rate, full → model (m/s) | sustain, full → model (s) | samples @30 Hz / @240 Hz |
|---|---|---|---|---|---|
| permissive | 3.0 | 2.0 | 0.8 → **0.16** | 2.0 → **0.4** | 12 / 96 |
| strict | 1.5 | 1.0 | 0.4 → **0.08** | 3.0 → **0.6** | 18 / 144 |

Angles are unchanged (λ⁰); the rate and the sustain are × √λ = 0.2. Run lengths use dmf's P6-D2
convention, `sustain_samples = ceil(sustain·fs)`, reused from dmf.
`tests/test_control_gated.py` asserts every number above, `source is dmf.PERMISSIVE`, the
identity at λ = 1, and the √λ law at λ = 1/100.

*Deviation from dmf's rule, recorded.* dmf applies the rate limit to the **CG heave rate**.
Here it is applied to the **pad deck-point vertical velocity** `v_pad,z`, because the pad is
what the drone lands on. At the aft pad that differs from the CG by the pitch lever-arm term:
the aft/CG `v_z` RMS ratio is 0.685–7.378 over the grid (P1-D2). The threshold value is
dmf's; only the quantity it is applied to changes.

*Controller definitions fixed by this entry* (docstrings in `src/rld/control/`):

- **Model time.** Each controller counts its own `act` calls since `reset`
  (`t = k / 30 Hz`). It cross-checks that count against the observation's float32
  `time_fraction` while the latter is below 1, and raises on a mismatch larger than half a
  step (a skipped reset or a double `act`). `time_fraction` alone saturates during the dwell
  grace.
- **Deck-point velocity is `velocity + rel_velocity`**, not `velocity − rel_velocity`: the
  environment defines `rel_velocity = v_pad − v_drone`. `tests/test_control_base.py` checks the
  sum against the env's analytic deck state (≤ 2e-6 m/s), and asserts that the difference is
  wrong.
- **Deck roll/pitch** are inverted from the observed normal via P1-D2's
  `n = (−sinθ·cosφ, −sinφ, cosθ·cosφ)`, degrees in dmf's signs; checked against the env ≤ 1e-4°.
- **`gated`** hovers at 0.3 m clearance (hand-set: `hover_kp_per_s` 2.0,
  `hover_speed_max_m_s` 0.4, `hover_tolerance_m` 0.05; not tuned). It commits once the lateral
  gate is open, it is at hover, and the trailing 0.4 s of control samples were all in limits.
- **`oracle_gated`** predicts touchdown at `t + height/descent_rate` (it assumes the
  feedforward closes on the pad at exactly the descent rate, and ignores tracking lag). It
  commits when the true `[t_td, t_td + 0.4 s)` window (96 physics samples) is all in limits.
  A window running past the trajectory's end does not qualify. `reset` without a
  `PrivilegedContext` raises. The context is rebuilt from the env's public
  `motion`/`pad`/`pad_offset_m`/`cfg`/`record.t0_model_s` through `build_trajectory`, and it
  is bit-identical to the driven trajectory (tested).
- **`fallback_commit_s: null`** for both gated controllers, so their timeouts are reported
  rather than rescued.

*Evidence.* `results/e01/tuning_pid_track_descend.csv`, `results/e01/tuning_pid_feedforward.csv`
(20 rows each, every trial's parameters and per-SS outcome breakdown, `selected` flag,
provenance block), `results/e01/tune_pool_final.csv` (4 rows). Config SHA-256 at the time of
writing: `pid_track_descend.yaml` `5d6aef55…`, `pid_feedforward.yaml` `91e7d5ad…`,
`gated.yaml` `da8eb578…`, `oracle_gated.yaml` `9665bd4c…`, `tuning.yaml` `c7d099d2…`.

*Amendment (2026-09-22, Gate 3 remediation, results-skeptic M1; the text above is left as
written).* The `oracle_gated` bullet above and item 4 describe the **superseded** oracle rule:
96 physics samples at 240 Hz over `[t_td, t_td + 0.4 s)`. That rule is strictly harder to satisfy
than the 12-sample, 30 Hz rule it was meant to bound. From this amendment, `oracle_gated` applies
**exactly** `gated`'s predicate, `rld.control.quiescence.QuiescenceRule`. The rule is built once in
`GatedBase` at the control rate and shared, so the two cannot drift apart:
- the same 12 samples, 1/30 s apart;
- the same Froude-converted thresholds;
- the same quantities: roll and pitch from the deck normal through `deck_angles_deg`, and the pad
  deck-point world `v_z`.

The only difference is the samples. The oracle reads the true future states at `t_td + j/30`,
`j = 0..11` (every 8th physics sample from the first at or after `t_td`). `gated` reads the
observed trailing 12. `window_is_quiescent` is removed, and `QuiescenceRule` is now the only
predicate. `tests/test_control_oracle_gated.py` asserts on 40 identical windows that both
controllers commit or neither does, that both agree with `QuiescenceRule.verdict`, and that
physics samples between the control-rate samples are not read. `privileged=True` is unchanged.
Every `oracle_gated` number produced before this amendment was made under the old rule, and is
superseded until re-run. That covers the `tune_pool_final.csv` rows and SS5 figures quoted above,
and `results/e01/` on the frozen lists. No tuning was involved, because the oracle has no tuned
parameters. At the same time, "MRU-equivalent" was replaced in code, configs and tests by "an
ideal (noise-free, zero-latency) ship motion reference unit plus state estimate" (skeptic minor
4). The edits to the config comments move `oracle_gated.yaml` to `e8bc4efd…` and
`pid_feedforward.yaml` to `beb988ad…`. The gain values are unchanged.

*Amendment (2026-09-22, Gate 3 remediation, results-skeptic B1): `pid_feedforward_lowvz`.* This
controller is a second gain set for **the same class and law** (`PidFeedforward`), registered
with `privileged=False` as the closing-speed reference for H1. Its gains come from the
**existing** 20-trial log `results/e01/tuning_pid_feedforward.csv` by the user's rule:
1. keep the trials whose tune success (`mean_success`, the mean over SS3–SS5) is ≥ best − 0.02;
2. of those, take the lowest pooled p95 closing speed (`pooled_td_closing_p95_m_s`);
3. break any remaining tie by the lowest trial index.

The rule was **worded after the tuning log existed, but before any frozen-list number for any
candidate was seen**. `rld.control.tuning.select_lowvz` implements it. The best success is 1.000,
so the cutoff is 0.98 and 11 trials qualify. The rule selects **trial 10**, with success
178/180 = 0.989 and pooled p95 0.153 m/s:
- its gains are kp_xy 1.4012840571813285 (1/s), ki_xy 0.8979697581380606 (1/s²), commit radius
  0.0711192973703146 m, descent 0.11144125740975142 m/s and k_ff 0.9547040911391378, with
  pid_feedforward's fixed keys;
- its outcomes are SS3 59/60, SS4 60/60 and SS5 59/60, and both losses are `bounce`;
- the runners-up are trial 5 (0.983, 0.173 m/s) and trial 15 (0.994, 0.179 m/s).

An earlier prediction that the rule would pick trial 15 was wrong. The user kept the rule as
worded. **Zero tuning budget was added: no new trial and no new episode.** The gains are in
`configs/control/pid_feedforward_lowvz.yaml`. `tests/test_control_pid_feedforward_lowvz.py` pins
trial 10 and asserts that the YAML equals that log row. The slow descent takes a mean of about
7.7 s to touch down against the 12 s budget, compared with about 4.5 s for `pid_feedforward`.

## Gates
| gate | date | result | note |
|---|---|---|---|
| 0 | 2026-09-21 | PASSED | `make test lint` green (6 tests, 53 s); pybullet 3.2.7 built from sdist and opens a DIRECT client on 3.12.3; `results/env_throughput.csv` written (8 rows, 1/8/16 SubprocVecEnv workers x rpm/vel); both submodule SHAs recorded in P0-D1. |
| 1 | 2026-09-21 | PASSED | `make test lint` green (66 tests, 62 s; ruff + ruff-format + mypy --strict clean). `results/deck_stats.csv` 192 rows = 96 cells x {aft, cg}, full grid (2 vessels x SS3-SS6 x 4 headings x 3 speeds), all 2304 realizations, with `z_std_model_m`, `vz_std_model_m_s`, `vz_p99_model_m_s`, `az_p99_model_m_s2` populated and no NaN; plus `deck_stats_seeds.csv` (4608 rows) and `deck_feasibility.csv` (2 rows). lambda = 1/25 and r_pad = -0.4*L confirmed in P1-D1. Feasibility rule PASS: frigate SS6 180 deg 12 kn aft vz p99 = 0.577165 m/s vs the 2.08333 m/s gate threshold, 3.61x inside; the rejected `SPEED_LIMIT` reading is recorded as FAIL beside it. Sign convention and ZYX rotation order corrected and pinned by three hand-computed cases (P1-D2); one project-wide lambda (P1-D3). |
| 2 | 2026-09-22 | PASSED | `make test lint` green (118 passed, 1 skipped, 107 s; ruff + ruff-format + mypy --strict clean). The 1 skip is by design: `tests/test_platform.py` gates only the configured driver and *measures* the other, and `test_constraint_driver_fails_the_tracking_gate` asserts the rejected one fails. Platform tracking, `kinematic` driver, 10 s model at frigate SS6 180 deg 12 kn aft: body-origin 0.000 mm and plate-corner 2.2e-16 m against the 1 mm gate, orientation 3.0e-8 rad, `getBaseVelocity` linear and angular ratio 0.0 % against the 2 % gate; `constraint` fails every one of those by an order of magnitude and cannot be tuned into passing (24-point sweep, four identical digits) -- P2-D1. `gymnasium.utils.env_checker.check_env` passes; reset determinism bit-identical as the env's 1st and 3rd reset; drop on a static pad registers exactly one touchdown; scripted 0.3 m/s descent on a static pad 100/100 `success`. `results/e00_env_sanity.csv` (4 rows) + `results/e00_env_sanity_episodes.csv` (800 rows) written from the `id` split's val partition, draw seed 20260922: hover 1.000 `timeout` at SS3 and SS5, random 1.000 `crash` at both, no successes under either -- outcome fractions sum to 1.000000 per row. **PyBullet-vs-analytic touchdown disagreement 0/800 = 0.0000** against the < 1 % gate, plus 0/100 on scripted static descents and 0/30 on a moving SS5 deck. `results/env_throughput_landing.csv` (8 rows) records the deck-in-the-loop throughput P3-D1 must size from (P2-D8); `results/env_throughput.csv` untouched. Two flags carried forward, both recorded rather than fixed: one of 800 sanity episodes tunnelled (7.86 mm penetration vs the 5 mm threshold) and it was the only random episode that ever reached contact, so the random arm exercises the touchdown path barely at all -- the scripted tests carry that load; and the plan's "dropped from rest" is not expressible in a velocity-setpoint action space, so the drop test uses the maximum commanded descent (P2-D9). |
