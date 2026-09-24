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

**Status: FROZEN 2026-09-22, revision 1, approved by the user.**
- The user approved the first draft; the first Gate 3 run then **failed** on the `results-skeptic`
  review, before any RL run.
- This revision, with the user's decisions logged in §9, was approved by the user on 2026-09-22.
- Nothing below changes after the first Phase 5 training run without a dated deviation entry.
- The SHA-256 of this block (from this heading to the line before P3-D2, UTF-8) is recorded in the
  Gate 3 row.

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

**7. Baselines on the frozen lists (`results/e01/`, revision-1 run, read before the hypotheses
below were written).**
- *Run.* 5 controllers × 2 800 episodes. The run is byte-identical at 24 and 7 workers. The three
  controllers whose code did not change reproduce the first run bit for bit.
- *No catastrophic failures.* No controller crashes or goes off the pad in any cell.
- *Detectors and tunnelling.* Detector disagreement is 5 / 14 000. Tunnelling is 14, all
  `pid_track_descend`, with a maximum penetration of 6.5 mm.
- **`pid_feedforward`.**
  - Success: 200/200 static, then 200/200, 200/200, 198/200 (99.0 % [96.4, 99.7]) and 181/200
    (90.5 %) at `id` SS3–SS6.
  - p95 closing speed at `id` SS5: 0.262 m/s.
  - Its losses are only `hard_landing` and `bounce`.
- **`pid_feedforward_lowvz`** (the H1a reference).
  - Success: 200, 196, 191 (95.5 % [91.7, 97.6]) and 170 (85.0 %) of 200 at `id` SS3–SS6.
  - p95 closing speed at `id` SS5: **0.188 m/s**. It is below `pid_feedforward`'s p95 in every
    cell.
  - It never times out; median time to touchdown is 7.6–8.0 s. Its losses are bounces and hard
    landings.
  - H1a's non-inferiority bound is therefore taken against 95.5 %.
- **`pid_track_descend`:** 85.0 % at `id` SS5.
- **`gated`.** It loses mostly to `timeout`, plus 14 bounces:
  - 4 at `id` SS6;
  - 2 at `unseen_seastate` SS6;
  - 1 at `unseen_heading` SS5;
  - 7 at `unseen_heading` SS6.

  It is below `pid_feedforward` from SS5 up in every regime.
- **`oracle_gated`** (privileged; `gated`'s own rule applied to the true future deck motion).
  - It loses only to `timeout`, apart from 1 bounce at `id` SS5.
  - Its success is within ±5 points of `gated`'s in every cell (unpaired reading). It is not
    above `gated` everywhere: 178 vs 179 at `id` SS5, and 178 vs 180 at `unseen_vessel` SS6.
  - It ties `pid_feedforward` at `unseen_heading` SS5 (199/200).
  - It is a **commit-timing oracle, not a bound on success or on landing quality**. Even so, only
    63–79 % of its SS5/SS6 touchdowns fall inside a truly quiescent window
    (`td_in_quiescent_window`). The cause is not verified; the likely one is that its touchdown
    prediction ignores tracking lag.
  - It is labelled that way in every table.
- **Quiescence at touchdown.** At SS6, only 17–26 % of `gated`'s touchdowns fall in a truly
  quiescent window, against 5–21 % for the ungated controllers. A quiet past 0.4 s is a weak
  predictor of a quiet next 0.4 s at this sea state.

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
    `pid_feedforward_lowvz` by **≥ 15 %** (0.188 → ≤ 0.160 m/s at `id` SS5), as the relative-p95
    point estimate from §4, with its 95 % CI excluding 0.
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

### P3-D4 — P3-D1 errata and Gate 3 re-review fold-in (2026-09-22)

*Why this is a separate entry.* The frozen P3-D1 block keeps its SHA-256
`21465588610e65f1d1253bd26e5ce5db1da938889c6042338e1de8d4d99f5ed2`. Every item below either
narrows a claim, corrects a statement of fact, or specifies a detail that was left open. None of
them changes a threshold, a prediction, an episode list or the success criteria. Where an item
disagrees with the P3-D1 text, this entry wins. It is dated before any Phase 5 run.

**MAJOR-1 of the second review: H1's title claimed too much.**
- *Narrowed title.* H1 is retitled **"residual RL lands softer than the lowest-closing-speed gain
  set in the P3-D3 log (constant-descent law), without losing success"**. The P3-D1 title
  "improves on what gain-tuning alone can do" is withdrawn.
- *What the reference is.* `pid_feedforward_lowvz` received **zero** tuning budget aimed at closing
  speed. It is the softest point of a search scored on success. All 20 trials use a
  constant-descent law. The search range goes down to 0.08 m/s, but only trial 10 (0.111 m/s)
  approaches it, and a constant-descent PID much slower than that would time out: `lowvz`
  already touches down as late as 10.65 s against the 12 s budget.
- *What H1a will and will not show.* A supported H1a shows "softer than the softest
  constant-descent PID in the success-tuned log". It does **not** show "softer than any PID":
  - a PID tuned for closing speed, or one with a two-stage (flare) descent, has not been ruled
    out;
  - the README says so wherever H1 appears.
- *Open option.* A pre-registered closing-speed PID search would close this gap. It remains
  available to the user before Phase 5 and is not taken here.

**Minor corrections to P3-D1:**
1. **§7 quiescence range.** The oracle's SS5/SS6 `td_in_quiescent_window` range is **63–85 %**
   (0.634–0.847), not 63–79 %.
2. **§2 overlap numbers were mislabelled.**
   - 96 / 37 / 50 are the realization overlaps of the **committed lists**.
   - The dmf **test partitions** themselves share 96 (`id` ∩ `unseen_heading`), 96
     (`id` ∩ `unseen_seastate`) and 120 (`unseen_heading` ∩ `unseen_seastate`).
3. **§2 wording.** The lists were committed before any controller was evaluated on them,
   **except the 32 static-pad controller-episodes of §9.1**.
4. **Timing.**
   - H1a and H1b were committed at `9343cb8`, **before** the revision-1 e01 run. They were written
     after the first e01 run.
   - The `lowvz` selection rule was fixed before any frozen-list number was seen for any candidate
     **other than trial 4** (`pid_feedforward`), which had already been evaluated in the first
     run.
5. **H1a outcomes made exclusive, H1 has no combined verdict.**
   - *Supported:* r ≥ 15 %, the CI of r excludes 0, and non-inferiority holds.
   - *Inconclusive:* 0 < r < 15 %, the CI excludes 0, and non-inferiority holds.
   - *Not supported:* every other case, including non-inferiority failing, the CI including 0, or
     r < 0.
   - H1a and H1b are scored and reported separately. No combined "H1" verdict is given.
6. **§4 pooling across seeds.** Within a bootstrap replicate, p95 is taken over the pooled
   resampled (seed, episode) touchdowns of each method. The point estimate is the same statistic
   on the unresampled pool of all seeds × episodes.
7. **H1b is scored outside the training distribution.** `id` SS6 is 0 % in-distribution, since
   SS6 is never trained on, so H1b is a sea-state-extrapolation result as well as an improvement
   claim.
8. **Sensitivity check.** Excluding non-touchdown episodes from p95 lets a policy lower its p95 by
   timing out on its worst deck windows. The −2-point non-inferiority margin caps this. Phase 7
   additionally reports, **as a sensitivity analysis, not a scored test**, the relative-p95 with
   timeouts ranked as the worst closing speed.
9. **Multiplicity.** No multiplicity correction is applied across H1a, H1b, H2, H3 and H4. Each is
   a separately pre-registered prediction, and the README states that no correction was made.

**P3-D3 tune-pool table, re-run with the corrected oracle and the `lowvz` row.** Source:
`results/e01/tune_pool_final.csv` at `d35f224`, tune pool only, seed 20260923, n = 60 per SS.
Success fractions:

| controller | SS3 | SS4 | SS5 | mean |
|---|---|---|---|---|
| pid_track_descend | 1.000 | 0.983 | 0.917 | 0.967 |
| pid_feedforward | 1.000 | 1.000 | 1.000 | 1.000 |
| pid_feedforward_lowvz | 0.983 | 1.000 | 0.983 | 0.989 |
| gated | 1.000 | 0.983 | 0.933 | 0.972 |
| oracle_gated (privileged) | 1.000 | 0.983 | 0.967 | 0.983 |

- This supersedes the oracle rows and the "oracle times out more than gated" claim in P3-D3
  item 4.
- `configs/control/pid_feedforward_lowvz.yaml` SHA-256:
  `373c2307857988a88f90cd39959e873b808034cddf7fc70fe7c14f323da34f4c`.

**Carried to Phase 4, not fixed in Phase 3.**
- *Stale "upper bound" labels.* They remain in `configs/control/oracle_gated.yaml` (line 1),
  and in the `src/rld/control/registry.py` and `PrivilegedContext` docstrings. They contradict
  P3-D1 §7.
  - *Why not now.* The YAML's bytes are hashed into the committed `results/e01/summary.csv`, so
    editing it now would leave that hash stale.
  - *Fix.* Relabel them "commit-timing oracle (privileged)" as the first `controls-engineer` task
    of Phase 4, together with `gated_forecast`.
  - *Guard.* The word "upper bound" must not appear in any results table or in the README for
    this controller.
- *`lowvz` bounces.* They happen at low closing speed (0.064–0.116 m/s at `id` SS5), which makes
  them lift-offs, not impacts. Before Phase 5, check that they are not a contact-solver artifact
  a residual policy could learn to exploit.
- *Unrecorded worker-count check.* The "byte-identical at 24 and 7 workers" check is not recorded
  in any committed artifact. The 7-worker outputs were compared in the session scratchpad
  (SHA-256 of `episodes.csv` `c5852090…`, identical to the committed file). Phase 7's
  `make eval` records the second worker count in `run_info.json`.

## Phase 4

### P4-D1 — Forecasters are fitted on the development pool, not on dmf's `id` train partition (2026-09-23)

*Decision (user, 2026-09-23).* `dlinear_ols`, `residual_interval`, `tcn` and `tcn_quantile` are
fitted on `rld.deck.splits.dev_pool(sim_cfg)[0]`:
- 729 realizations: frigate, SS3–SS5, 45/135/180°, 0/6/12 kn, seed ordinals 0–26;
- canonical key-list SHA-256 `346b50fa…`.

`dev_pool()[1]` is used for early stopping, seed selection and every calibration:
- 135 realizations, seed ordinals 27–31;
- SHA-256 `ea4a2cf2…`.

The full key lists and hashes are in `results/forecast/fit_keys.json` and
`results/forecast/fit_manifest.json`.

*Reason.* The kickoff prompt and plan task 1 said "fit on the `id` regime". dmf's `id` train
partition is frigate seeds 0–26 over **all** sea states and headings, 1 296 realizations. It
therefore contains SS6 and 90° realizations, and those are `unseen_seastate`-test and
`unseen_heading`-test episodes on the frozen lists (P3-D2). The plan's Phase 4 note (a) supersedes
the task text.

*How dmf is driven.* dmf selects training data by regime name only.
`rld.deck.forecast_fit` therefore uses dmf as a library:
- it builds a `dmf.data.splits.Split` whose `train_keys` / `val_keys` are the two pools (label
  `dev_pool`);
- it uses dmf's own dataset, fitting, export and parity code, unmodified.

The corpus is regenerated with dmf's `generate_corpus.py` into `artifacts/dmf/corpus`:
- 2 304 realizations, 1.11 GB, manifest SHA-256 `64ff4a17…`;
- the dmf submodule SHA is `e9fa15cc…`, unchanged;
- `third_party/` is untouched (`git status` clean).

*Test.* `tests/test_deck_forecast.py` reads the fitted keys back and checks them against the
manifest hashes. It asserts that both pools are disjoint from every realization in all five
`results/episodes/` lists.

### P4-D2 — Model set, seeds, selection and what is stored (2026-09-23)

**Models and configs.** Every model uses dmf's own configs: `configs/model/*.yaml` and
`configs/data/default.yaml`, with SHA-256s in the manifest.

| model | head | how it is fitted | train settings from |
|---|---|---|---|
| `dlinear_ols` | point | closed form | `e02_deep.yaml` |
| `residual_interval` | quantile | DLinear-OLS point plus empirical 5/95 % residual quantiles fitted on the tune pool; the band's width does not depend on the input | `e03_probabilistic.yaml` |
| `tcn` | point | AdamW, bf16, 60-epoch cap, batch 1024, lr 2e-3 | `e02_deep.yaml` |
| `tcn_quantile` | 9-level fan | same as `tcn` | `e03_probabilistic.yaml` |

**TCN seeds.** Seeds 0, 1 and 2 (dmf's minimum). Each seed gets its own data loader, shuffled by
that seed.
- *Selection rule, fixed here before any fit finished:* the deployed checkpoint is the seed with
  the lowest tune-pool loss (MSE for `tcn`, pinball loss for `tcn_quantile`).
- The losses of all seeds are recorded in the manifest.

**What is stored.** dmf does not store `NormStats`, and never saves DLinear-OLS. So each
`artifacts/dmf/<model>/` holds:
- `state_dict.pt`;
- `norm_stats.npz`, the per-channel train-partition std that dmf divides by. dmf de-means each
  window by its own mean;
- `model.onnx`, from `dmf.deploy.export_onnx`;
- `conformal_padvz.npz` (P4-D3);
- `meta.json`, which holds the SHA-256 of every file and is verified on load.

torch↔ORT parity is checked with `dmf.deploy.parity`. `residual_interval`'s tune loss is in-sample
by construction, and the manifest flags this.

### P4-D3 — Online adapter semantics (2026-09-23)

**The feed (`rld.deck.forecast.ShipMotionFeed`).**
- *What it is.* The past-only history of dmf's 6 clean channels, full scale. It is an ideal
  (noise-free, zero-latency) ship motion reference unit, **past only**.
- *Privilege.* It is not privileged **with respect to the future**: it never reads deck motion after
  the current time.
  - It is, however, an **extra ideal ship motion sensor** that `gated` and the PID controllers do
    not have. It supplies 4.0 s model of pre-episode history, plus heave, roll and pitch rates. The
    observation vector carries neither.
  - Comparisons between forecast-using and non-forecast methods (`gated_forecast*` against `gated`
    here; `ppo_forecast` against `ppo` in Phase 6) therefore compare different sensor suites as well
    as different decision rules.
  - Observation noise is `enabled: false` in every evaluation, and this feed is noise-free too.
  - *Correction (2026-09-23, results-skeptic M3).* An earlier version of this bullet said the feed
    gives "no information advantage over `gated`". That was wrong.
- *Clock.* The runner advances its clock with `advance_to` before each `act`. The source is never
  evaluated at a time later than the clock.
- *Grid.* Samples lie on dmf's absolute 10 Hz full-scale grid, bit-identical to the corpus `t`
  column. The 200-sample ring buffer is pre-filled from `[t0 − 4.0 s, t0]` model, i.e. before the
  episode (P2-D7).
- *Where it cannot run.* The static list has no vessel, so the feed cannot be built on it.

**The float32 cast.** The adapter reproduces dmf's pipeline including the corpus's float32 storage
cast.
- Without the cast, bridge-fed parity is 8.7× (`dlinear_ols`) and 9.8× (`residual_interval`) the
  `1e-4·max(1,|y|)` tolerance.
- The cause is the model's conditioning: DLinear-OLS `remainder` rows have L1 norms up to 6 512.
- The uncast figure is recorded, non-gated, in `results/forecast/parity.csv`.

**Pad quantities.** For a centreline pad, `z = heave + x·sin(pitch)` and
`v_z = heave_rate + x·cos(pitch)·pitch_rate`. They are converted to model units at z × λ and
v_z × √λ.

**The pad-v_z band: a conformal-calibrated box** (user decision, 2026-09-23).
- *Construction.*
  - Take the interval-arithmetic box from the channel bands, which is exact and conservative.
  - Rescale its half-width about the point by one split-conformal γ per lead.
  - Fit γ on the tune pool to 90 % pad-v_z coverage. dmf's conventions apply: the score, the
    `ceil((n+1)(1−α))` order statistic, α = 0.1.
  - Do this per model and per pad.
- *Why not the raw box.* The raw box assumes the heave-rate and lever-arm terms are independent.
  So it over-covers at the aft pad (0.91–0.99).
- *The γ values for `residual_interval`.* Aft γ = 0.61–0.76; CG γ = 1.00.
- *Pad matching.* Pads are matched by lever arm as a fraction of vessel length. The s175 aft pad
  therefore uses the frigate-calibrated γ, a transfer under the vessel shift like the forecaster
  itself.
- *What stays uncalibrated.* Roll and pitch keep the model's native 5/95 % bands. Pad z keeps the
  raw box; no controller reads it.
- *Coverage.* Calibrated pooled coverage is 0.896–0.903 at every lead and pad. But the band has one
  width for every sea state, so it under-covers at SS5 (0.81 aft, 0.74 CG at 9.3 s full). See
  `results/forecast/coverage.csv`.

**`gated_forecast` and `gated_forecast_tcn`.** These use the shared `QuiescenceRule` (P3-D3
amendment) unchanged:
- 12 samples, 1/30 s apart, from `t_td = t + h / descent_rate`, the oracle's law, which is now a
  shared helper.
- The band is read by linear interpolation of the **forecast** on its 50 Hz model grid. This is an
  interpolation of predictions, not of deck motion.
- `max(|lo|, |hi|)` per sample is handed to `verdict`. This is dmf's interval rule: the whole 90 %
  band inside ±limit.
- A window past the 3.0 s-model horizon is not quiescent.

**Rates and cost.**
- One new sample arrives every 0.02 s model, 1–2 per 30 Hz control step.
- On CPU ORT with 1 thread, a step costs p50 0.86 ms (`dlinear_ols`) and 1.04 ms
  (`residual_interval`), against the 33.3 ms control period (`results/forecast/cost.csv`).

### P4-D4a — Pre-registered expectation for the forecast-gated arms (2026-09-23, before any `gated_forecast` episode on the frozen lists)

**Evidence.** `results/forecast/band_feasibility.csv`: `residual_interval` on the tune pool, leads
7.0 / 8.0 / 9.3 s full, which is where `gated_forecast`'s window sits.
- At the **aft** pad, the calibrated pad-v_z band is 0.35–0.36 m/s model wide at those leads. The
  permissive limit ±0.16 leaves a window only 0.32 m/s wide. So the fraction of forecasts that pass
  is **0.000** at SS3, SS4 and SS5, under every construction tried.
- The true future passes all three limits 0.95 / 0.77 / 0.45 of the time (SS3 / SS4 / SS5).
- At the **CG**, the band passes 0.92 / 0.69 / 0.38, against a true base rate of 0.96 / 0.88 / 0.59.

**Expectations, recorded before the run:**
1. `gated_forecast` (`residual_interval`) at the **aft** pad times out on ≥ 95 % of episodes in
   every regime × sea-state cell. This is a property of a fixed-width band at 7–9 s full-scale
   leads, not of the deck. If it holds, it is reported as found and not fixed after the fact.
2. With the pad at the CG, the same controller commits. Its timeout fraction rises with sea
   state. No numeric prediction is made.
3. `gated_forecast_tcn` (`tcn_quantile`): no numeric prediction is made. Its band depends on the
   input and may pass in calm windows.

*Addendum (2026-09-23, 14:30, after the TCN fits and before any e02 episode).* The `tcn_quantile`
fits finished, and the selected checkpoint is seed 1. The same `band_feasibility.csv` analysis now
covers it, at a lead of 8.0 s full with the calibrated band.

| aft pad, lead 8.0 s full | SS3 | SS4 | SS5 |
|---|---|---|---|
| calibrated band, median width (m/s model) | 0.061 | 0.113 | 0.198 |
| pad-v_z coverage | 0.91 | 0.90 | 0.89 |
| all three quantities inside, `tcn_quantile` band | 0.93 | 0.57 | 0.17 |
| all three quantities inside, true future (base rate) | 0.95 | 0.77 | 0.45 |

- Unlike `residual_interval`, the TCN arm is therefore expected to commit at the aft pad.
- Its pass rate falls well below the base rate as sea state rises.
- This is evidence, not a new scored prediction. Expectation 3 stands as written.

*Erratum (2026-09-23, results-skeptic M2; the text above is left as written).* The heading's
"before any `gated_forecast` episode on the frozen lists" is **not true as written**.

What happened:
- A wiring smoke run at about 10:23 local (`run_info` 14:23 UTC, in the session scratchpad) flew
  366 frozen-list episodes. It took the first 3 episodes of each `id` and `unseen_vessel` cell,
  plus 3 static, for all 7 controllers × 2 pads.
- That included 24 aft `gated_forecast` episodes with the real `residual_interval` model, all
  timeouts. It also included a smoke-checkpoint `tcn_quantile`, whose numbers are meaningless.
- P4-D4a was written into the working tree in the same period. Its order relative to the smoke run
  is not established.
- P4-D4a was first committed in `9e8061f` at 14:25:57 local. The addendum's "14:30" stamp is wrong:
  it was written shortly before that commit.
- *Corrected at the Gate 4 review.* e02's episodes began at about 14:25:24 local, roughly 33 s
  **before** that commit. The start is `run_info` `timestamp_utc` 18:53:58, taken at the end of
  the run, minus `wall_s` 1 714. The run writes no result until it ends, so no e02 result existed
  before P4-D4a was committed.

What this means:
- Expectation 1 should be read as recorded after, or at best alongside, a 24-episode smoke run.
- It was grounded in the tune-pool band analysis (the fraction of forecasts whose band passes was
  0.000 at SS3–SS5), which predates both.
- Nothing was tuned after the smoke run. For the same episodes, the smoke rows and the e02 rows are
  identical in outcome, steps, touchdown time, touchdown speed and `td_in_quiescent_window`. This
  was checked by the results-skeptic against the smoke `run_info` and `episodes.csv` in the session
  scratchpad. That file is **not committed**, so the check cannot be traced to a committed artifact.

**Other records for P4-D2 and P4-D3.**
- *TCN training.* Every TCN fit ran to the 60-epoch cap without early stopping: best epochs 59/56/56
  for `tcn` and 59/59/59 for `tcn_quantile`. So the models may be under-trained against dmf's own
  budget, which was kept unchanged. Tune losses: `tcn` MSE 0.09920 / 0.09950 / 0.09911, selected
  seed 2; `tcn_quantile` pinball 0.05036 / 0.05034 / 0.05066, selected seed 1.
- *Seed sensitivity (user decision, 2026-09-23).* The top two `tcn_quantile` seeds differ by 0.03 %
  in tune loss, so the selection is effectively arbitrary. Seeds 0 and 2 are therefore evaluated as
  **secondary** arms, `gated_forecast_tcn_seed0` and `gated_forecast_tcn_seed2`, in
  `results/e02_tcn_seeds/`. The pre-registered selected seed stays the primary arm.
- *Coverage.* The coverage P4-D3 quotes (0.896–0.903) is in-sample, because the band was
  calibrated on the same tune pool. Its 2-fold out-of-sample counterpart is in
  `results/forecast/coverage.csv`.
- *Deferred.* Plan task 3's forecast block for the RL observation is **deferred to Phase 6**:
  - `rld.deck.forecast.leads_full_s` provides pad z and v_z at leads of 1/2/3 s full scale;
  - it is not wired into `rld.envs.observation`;
  - it is not a Gate 4 criterion.

### P4-D4 — Forecast-gated results on the frozen lists (2026-09-23)

**Sources.**
- `results/e02/` holds 7 controllers × {aft, cg} on all five frozen lists: 38 400 episodes, 1 714 s at
  24 workers.
  - It was launched from the working tree committed as `9e8061f`. That SHA is not recorded in its
    `run_info`, which predates the `git_sha` field.
  - Its summary was re-derived from the unchanged `episodes.csv`, adding the per-listed-episode
    quiet columns. The re-derivation ran from a dirty tree, later committed as `12fd9f5`
    (`run_info.resummarised`, `episodes_flown: 0`).
  - Between those commits `gated_forecast.py` changed only in docstrings and factories, and the
    forecaster file hashes are pinned per row.
- `results/e02_tcn_seeds/` holds the secondary seed arms, 10 400 new episodes at `12fd9f5` (clean
  tree), plus e02's rows carried.
- All numbers are simulation only, on a Froude-scaled Crazyflie model and dmf's synthetic JONSWAP
  deck motion. Nothing here concerns real flight or real deck data.

**Reproduction.**
- The e02 aft rows of the five Phase 3 controllers are byte-identical to `results/e01/episodes.csv`
  (14 000 of 14 000; `run_info.json["reference_check"]`).
- The episode-list MANIFEST is 10/10 OK.
- The P3-D1 block SHA-256 verifies before each run.

**Scoring of P4-D4a.**
1. **Expectation 1 held.** `gated_forecast` (`residual_interval`) at the aft pad: 2 600 of 2 600
   moving-deck episodes time out, with zero touchdowns in all 13 regime × SS cells.
2. **Expectation 2.** Recorded as "timeouts rise with sea state"; scored as non-decreasing. At the
   CG the same controller commits, and its success falls with sea state:
   - `id`: SS3 0.995, SS4 0.955, SS5 0.79, SS6 0.315;
   - `unseen_heading`: flat at SS3 and SS4 (1.000, 1.000).
3. **Expectation 3** made no prediction. `gated_forecast_tcn` commits at the aft pad (`id` success
   1.00 / 0.95 / 0.60 / 0.165, SS3 → SS6).

**Findings.** They are descriptive: per-cell Wilson 95 % CIs, compared over 26 moving-deck cells,
with no multiplicity correction and no pre-registered hypothesis. "Separated" means the two
intervals do not overlap.

**Outcome classes.** Every forecast-gated failure is a `timeout`, except for a few bounces:
- 2 bounces in e02 and 5 more in the seed arms, at most 0.005 in any cell;
- no crash, off-pad or hard landing anywhere.

This follows from `fallback_commit_s: null` (P3-D3) and the P3-D1 episode budget: a commit window
that never arrives is reported as a timeout, not rescued.

**Success: forecast gating lowers it.** Counts are cells with separated CIs:

| arm | worse than `gated` | better than `gated` | worse than `pid_feedforward` |
|---|---|---|---|
| `gated_forecast` | 19 of 26 | 0 | 20 of 26 |
| `gated_forecast_tcn` | 9 of 26 | 0 | 12 of 26 |

- For `gated_forecast_tcn`, the loss grows with sea state. For example, `id` SS6 aft success is
  0.165 against 0.63 for `gated`.
- `pid_feedforward`, ungated, has the highest success, or success within the CI of the highest, in
  every cell. At CG `unseen_heading` SS5, `oracle_gated` has 0.995 against its 0.990.

**Quiet landings: count them per listed episode, not per touchdown.**
- Per touchdown, `gated_forecast_tcn` looks close to `oracle_gated` (e.g. `id` SS6 aft 0.64 against
  0.68). But it touches down on only 33 of 200 episodes, against 129.
- Per **listed** episode (`quiet_landings_per_listed`), the separated-CI cells against `gated` are:

  | arm | more than `gated` | fewer than `gated` |
  |---|---|---|
  | `gated_forecast_tcn` | 5 of 26 (aft `id` SS4, aft `unseen_heading` SS5, CG `id` SS4, CG `id` SS5, CG `unseen_heading` SS5) | 1 (aft `unseen_vessel` SS6) |
  | `gated_forecast` | 2 | 13 |

- Against `oracle_gated`, both arms have fewer quiet landings in 12–17 cells and more in none.
- **Supported:** at SS4–SS5 the TCN arm converts some of `gated`'s non-quiet landings into quiet
  ones, and pays in timeouts.
- **Not supported:**
  - that forecast gating improves commit timing overall;
  - that it approaches the commit-timing oracle;
  - that it helps at SS6.

**Seed sensitivity** (secondary, `results/e02_tcn_seeds/`).
- Across `tcn_quantile` seeds 0, 1 (selected) and 2, the largest per-cell spread is 0.075 in
  success (aft `unseen_heading` SS6: seed 0 0.295, seed 1 0.22, seed 2 0.235) and 0.07 in quiet landings per listed
  episode (aft `unseen_heading` SS5).
- Every seed is worse than `gated` on success in 8–9 cells and better in none.
- Every seed has more quiet landings per listed episode than `gated` in 3–6 cells and fewer in 1.
- So the **direction** of the TCN-arm findings does not depend on which seed was selected, but the
  specific cells partly do.
  - Success: every seed is worse than `gated`, in 8–9 cells, and better in none.
  - Quiet landings: seed 2 reproduces only 3 of the 5 cells where the selected seed has more than
    `gated` (aft `id` SS4, CG `id` SS4, CG `id` SS5), and neither `unseen_heading` SS5 cell.

**The aft vs CG control.**
- At the CG the DLinear band passes and the arm commits. At the aft pad it never does.
- The aft band's excess width comes from the lever-arm term (P4-D3, `band_feasibility.csv`).
- Aft-pad v_z is where dmf's roll/pitch–heave phase defect enters. So the aft-pad null for the
  fixed-width forecaster is reported together with this caveat, as the CG arm requires.

**Distribution caveats.**
- The forecasters were fitted on frigate SS3–SS5 at 45/135/180°.
- The SS6, 90° and s175 cells are therefore outside the forecasters' training range, as they are for
  every learned method (P3-D2).
- The s175 aft pad uses the frigate-calibrated γ (P4-D3).

**Cost.** Per 30 Hz control step, CPU ORT with 1 thread, p50 / p99 (`results/forecast/cost.csv`):

| model | p50 | p99 |
|---|---|---|
| `dlinear_ols` | 0.94 ms | 1.20 ms |
| `residual_interval` | 1.15 ms | 1.49 ms |
| `tcn` | 1.60 ms | 2.02 ms |
| `tcn_quantile` | 1.88 ms | 3.08 ms |

That is against a 33.3 ms control period. The `cost.csv` regenerated at `12fd9f5` supersedes the
earlier P4-D3 figures. A feed `reset` (200 dmf evaluations) costs about 52 ms, once per episode.

### P4-D5 — `oracle_gated` relabel and its config hash (2026-09-23)

Plan note (c) and P3-D4 asked for this relabel. The stale "upper bound" labels were replaced by
"commit-timing oracle (privileged)":
- in `configs/control/oracle_gated.yaml` line 1;
- in the `registry.py` module docstring and description;
- in the `PrivilegedContext` and `oracle.py` docstrings;
- in one test docstring.

All of these are comment and docstring changes only. No value or behaviour changed.

**The hash.** `oracle_gated.yaml` SHA-256 moved:
- old: `e8bc4efd5615cea1a07f05f0368f226b682e11d78968b2a8d6b03c8f0862b4f1`, which remains in the 14
  `oracle_gated` rows of `results/e01/summary.csv` and is left as committed;
- new: `5daca06e217fc765611af98af2fc70f312ca4cbdafffe408d5827eae8138bce3`, carried by
  `results/e02/summary.csv`.

`resolved_gains_sha256` is unchanged at `dec9793d…`.

**Evidence the change is harmless.** `results/e02/run_info.json["reference_check"]` shows the 2 800
aft `oracle_gated` rows of e02 byte-identical to `results/e01/episodes.csv`, and likewise for all
five Phase 3 controllers (14 000 of 14 000 rows). The only provenance difference it lists is this
hash.

`results/e01/success_vs_seastate.md` was re-rendered with `--render-only`. Only the `oracle_gated`
label lines changed. A render-time guard (`rld.eval.report`) now refuses any output containing
"upper bound".

**The pad-at-CG control arm.** It evaluates `gated`, `oracle_gated`, both forecast-gated arms and
the three PID baselines on the **same** frozen episodes, with the pad at the CG. It is added here
because aft-pad v_z carries dmf's roll/pitch–heave phase defect (CLAUDE.md). It is a control, not
a new evaluation list.

## Phase 5

### P5-D1 — lowvz bounces: not a contact-solver artifact; ~80 % are rim-first rocking that the 50 ms contact-loss grace scores as `bounce` (2026-09-23)

*Question (P3-D4 carry-over, Phase 5 "Before you start" item a).* Are
`pid_feedforward_lowvz`'s low-closing-speed bounces a contact-solver artifact that a pure or
residual policy could exploit, or be unfairly penalised by? This is a diagnosis. **No environment
code, config or success criterion was changed.**

*What was flown.* The **P3-D2 tune pool only**; nothing from `results/episodes/`. Each arm flew:
- the committed P3-D3 tune draw: 180 episodes, SS3–SS5;
- plus 1 000 SS5 and 400 SS4 tune-pool episodes from a separate diagnostic seed (20260925);
- 1 580 episodes in total, logged at every physics substep.

The committed-physics arm reproduces `results/e01/tune_pool_final.csv` exactly on the tune draw.

The diagnostic-only physics arms are 480, 960 and 1920 Hz, `contactERP` = 0 (no
penetration-recovery push-off), and a 0.5 s contact-loss grace that measures contact gaps without
ending the episode. Scripts: `scripts/p5_bounce_check.py`, `scripts/p5_bounce_check_report.py`
and `scripts/p5_contact_probes.py`. Results and a short note: `results/p5_bounce_check/`.

*n examined.* 242 bounces were logged substep by substep:
- `lowvz`: 64 at 240 Hz, 86 at 480 Hz, 66 with `contactERP` = 0;
- `pid_feedforward`, same arms: 10, 10 and 6.

*Findings (lowvz, committed 240 Hz physics, 64/1 580 bounces).*
1. **Not a solver artifact.**
   - The solver leaves the contact point at a relative normal velocity of −0.3 mm/s (median),
     4.2 mm/s (p95) and 10.1 mm/s (max) at the end of the last contact substep. Zero restitution
     works as intended.
   - Bullet's multibody contact *does* have a Baumgarte push-off, ≈ `contactERP`·overlap/dt
     (18 mm/s per mm of overlap at 240 Hz; `contact_probes.csv`). Setting `contactERP` = 0
     removes it (median per-episode max 18.6 → 0.4 mm/s) and **does not change the result**:
     64 → 66 bounces, 60 of 64 unchanged pairwise, and 58 → 59 dwell gaps over 50 ms.
   - Penetration is ≤ 1.3 mm. No episode was flagged for tunnelling.
2. **Mechanism A, rim rocking: 51/64 (80 %).** 48 happen at impact and 3 later.
   - The drone touches down on its rim at a median relative tilt of 6.9° (IQR 5.6–7.8°), with one
     contact point.
   - The rim impulse and the attitude loop rotate it flat at about 1.6 rad/s, faster than the CoM
     falls. The rim lifts by at most a median 2.0 mm (max 3.9 mm).
   - The **CoM keeps closing on the deck through every one of these gaps**.
   - Contact is lost 8 ms after touchdown and the gap outlasts `contact_loss_grace_s` = 0.05 s, so
     the episode is scored `release` → `bounce`. P3-D4's description ("lift-offs, not impacts") is
     therefore wrong for these: they are post-impact rocking, not lift-offs.
3. **Mechanism B, unloaded rim lift-off: 13/64 (20 %).** These happen 4–500 ms after touchdown.
   - The deck accelerates down its normal at a median −2.1 m/s² (−1.7 to −3.7).
   - The drone holds 76 % of hover thrust. The contact carries only 8.7 mN (median; 1–22 mN) out of
     a 265 mN weight, and a small rotation lifts the rim.
   - This is physical. A drone at near-hover thrust on a deck dropping at ~0.2 g is nearly
     weightless on it.
4. **Down-force decides it, not the solver.** In controller-free rim-first drops (`contact_probes.csv`):
   - At thrust = weight, the contact gap is 229–325 ms at every rate from 240 to 1920 Hz, and also
     with `contactERP` = 0.
   - At 96 % of weight the gaps disappear. At 98 % one of 20 runs leaves a single 4 ms gap.
   - `lowvz` holds 96 % of weight in the rocking gaps: after contact its setpoint is still the deck
     feedforward minus 0.111 m/s, which barely unloads the rotors.
5. **The bounce rate follows the rocking time**, τ = 0.06 m · sin(rel. tilt) / closing speed.
   - Tune pool: 9/1 299 for τ ≤ 50 ms, then 7/136 (50–70 ms), 25/91 (70–100 ms) and 23/54 (> 100 ms).
   - The committed e01 frozen-list rows, read and not flown, show the same pattern: 11/2 267, then
     14/242, 43/138 and 42/152.
   - This is why `lowvz` bounces more than `pid_feedforward` (64 vs 10 here, 111 vs 52 in e01):
     halving the closing speed doubles τ.
6. **Substep sensitivity, recorded and not fixed.** Dwell gaps over 50 ms number
   58 / 78 / 112 / 116 at 240 / 480 / 960 / 1920 Hz, converging by about 960 Hz.
   - The committed 240 Hz physics *under*-counts rocking gaps by about 2× relative to converged
     physics. It does not create them: the mechanism and the CoM-closing signature are the same at
     every rate.
   - Individual episodes are not stable to the substep. Base → 480 Hz: 20 of 64 bounces stay
     bounces, and 66 successes become bounces.
   - Per-episode bounce labels are therefore sensitive to integration detail. Aggregate rates are
     conservative at 240 Hz.
7. **Detectors.** Every `release` is declared by the contact-manifold detector (normal force
   > 1e-4 N, 50 ms grace).
   - The analytic clearance agrees in 64/64: the lowest point was more than the 1 mm margin clear
     of the plate during the gap. So this is not a false loss from the force filter.
   - First-touchdown detector disagreement is 3/1 580.

*RL relevance.*
- **No solver exploit found.** The only energy-injecting term is Baumgarte push-off. It only ever
  adds separating velocity, which causes bounces rather than preventing them. It needs penetration
  that only harder impacts produce, and removing it changes nothing measurable.
- **What policies will be rewarded for.** `bounce` (−10 against +50) penalises **slow, tilted**
  touchdowns under a rule, not a physical lift-off. A policy can avoid it three ways:
  - (a) touch down flatter relative to the deck;
  - (b) touch down faster;
  - (c) hold a few percent of net down-force after contact, by commanding a strong descent once
    `in_contact` is set. This is the "throttle cut on touchdown" that `lowvz` lacks by design.

  (a) and (c) are legitimate landing behaviour, and (c) is learnable from the observation. They
  are not exploits.
- **Two consequences for H1a.** Both are recorded here before any Phase 5/6 run and neither
  changes P3-D1:
  - Part of any success gain over `lowvz` (4–5 % bounce at SS5 on this pool) can come from
    post-contact thrust behaviour rather than from the approach. H1a's non-inferiority margin is
    therefore easier to meet than "equal approach quality" would suggest.
  - (b) pushes against softer landings: a 15 % lower closing speed raises τ by ~18 % for the same
    tilt. This biases H1a *against* support. The Phase 7 README reports bounce rates beside
    closing speed for every method.
- **Nothing is changed.** Success criteria, grace, physics rate and `lowvz` all stay as frozen. If
  the user wants it, the options are:
  - a dated deviation of `contact_loss_grace_s`. A 0.1 s grace would reclassify 51 of the 58
    base-physics > 50 ms gaps;
  - a pre-Phase-5 change to 960 Hz physics;
  - a touchdown throttle-cut variant of `lowvz` as a separate baseline.

  None is adopted here.

*Decision (user, 2026-09-23, before any Phase 5 training run).*
- **Criteria stay frozen.** `contact_loss_grace_s` stays at 0.05 s and physics stays at 240 Hz.
  The rim-rocking mechanism, the physics-rate instability of individual `bounce` labels, and the
  two H1a consequences above are carried as recorded findings.
- **A throttle-cut baseline is added:** `pid_feedforward_lowvz_cut`, which is `lowvz` plus a
  post-contact down-force rule. Its rule is fixed in P5-D2 before any episode of it is flown. It
  gets no tuning budget, and it runs on the frozen lists before any RL run. It is printed beside
  `pid_feedforward_lowvz` in every table. It does **not** replace `lowvz` as the H1a reference
  (P3-D1 §8 is unchanged); it bounds how much of a success gain can come from post-contact thrust
  alone.

### P5-D2 — pid_feedforward_lowvz_cut: rule fixed before any episode (2026-09-23)

*What this entry fixes.* This entry fixes the post-contact rule of the throttle-cut baseline
that P5-D1's decision added. It was written at 19:30 local on 2026-09-23, on top of `b3fd2b5`,
**before any episode of this controller was flown** on any list, pool or static pad. The only
inputs were:
- physical reasoning about `DSLPIDControl`, which is the shared action space's tracker;
- the P5-D1 numbers already committed in `results/p5_bounce_check/`.

The rule has **no tuning budget**. No parameter was swept and no variant was flown or selected
by success. The rule below is the only one that was written.

*Config.* `configs/control/pid_feedforward_lowvz_cut.yaml`, SHA-256
`e5f19903a89165ecdc50218f2d626b7b41b03b566895f7be520f5d9a59e499bb`.
- *Pre-flight correction.* The first draft of this entry recorded `8cbf2e61…`. A YAML comment
  and the "why 1.5 m/s" paragraph below then claimed that any cut speed above 0.7 m/s gives the
  same idle state. The static check of the tracker's arithmetic, described below, showed that
  claim was false. Both texts were corrected **before any episode was flown**. The two keys and
  their values did not change.
- It holds two keys: `gains_from: pid_feedforward_lowvz.yaml` and `cut_speed_m_s: 1.5`.
- The gains are **loaded by reference**, not copied, so they are lowvz's trial-10 gains and
  cannot drift from them.
- `pid_feedforward_lowvz.yaml` is untouched (SHA-256 still `373c2307…`, as in P3-D4).
- Registered as `pid_feedforward_lowvz_cut` with `privileged=False` and `needs_motion_feed=False`.
- Code: `src/rld/control/lowvz_cut.py`.

**The rule.** All quantities are model scale.
1. **Touchdown inference.** The observation's `in_contact` flag, read through `ObsView` (> 0.5).
   This is the environment's contact-manifold detector: drone–deck normal force > 1e-4 N, taken
   at the **last physics substep** of the control step. There is no other cue and no other
   threshold.
   - *Rejected: a height cue.* A rocking gap's clearance (median 2.0 mm, max 3.9 mm; P5-D1)
     looks the same as the last pre-contact sample. At lowvz's 0.111 m/s descent that sample
     is 0–3.7 mm above the deck. A height threshold would therefore fire *before* contact in
     most episodes, which changes the approach and is not a post-contact rule.
   - *Rejected: a body-rate or tilt-rate signature.* It would need a threshold that could only
     be set by flying.
2. **Latch.** The latch sets on the first observation with `in_contact` = 1 and holds until
   `reset`. It is **never released**, for two reasons:
   - any contact gap longer than `contact_loss_grace_s` = 0.05 s ends the episode;
   - the shorter gaps are exactly the rim-rocking gaps in which down-force is needed.
3. **Before the latch.** The controller returns `pid_feedforward_lowvz`'s setpoint, **bit for
   bit**. The same code path computes it, and lowvz's clock, integrator and lateral gate advance
   as they do in lowvz. It continues to do so after the latch.
4. **After the latch.** The setpoint is `s_lowvz(o) − cut_speed_m_s · n̂(o)`, followed by the shared
   norm cap. n̂ is the observed unit deck normal in the world frame, and `cut_speed_m_s` = 1.5 m/s
   = `v_max_m_s`, the largest descent the shared action space can command.

*Why this command, from the tracker's arithmetic.*
- **How `DSLPIDControl` computes thrust.** The environment calls it with `target_pos` =
  `cur_pos`. Its thrust vector is therefore `t = m g ẑ + D ∘ (v_sp − v)`, with D = (0.2, 0.2,
  0.5) N/(m/s) and m g = 0.2646 N. The collective is `max(0, t · b_z)`. Each motor's PWM is the
  collective plus a mixer term of at most 6 400 PWM (torques are clipped at ±3 200 PWM), and is
  then clipped to `MIN_PWM` = 20 000.
- **The idle state.** When `t · b_z` ≤ **0.0754 N**, every motor sits at `MIN_PWM`. Thrust is
  then **0.1126 N = 42.6 % of weight** and the attitude torque is **zero**. That is "motors to
  idle", the lowest thrust the tracker can produce.
  - For a level drone this needs a velocity error of v_z ≤ −0.378 m/s.
  - The collective reaches the `MIN_PWM` floor at −0.304 m/s.
  - The target attitude flips at −0.529 m/s. This is harmless once every motor is at idle,
    because no torque is applied.
- **Why 1.5 m/s, and where it reaches idle.** 1.5 m/s is `v_max_m_s`, the full-scale descent of
  the action space. It was **not compared against other values by flying**. A static grid
  check, which evaluates the tracker's arithmetic after the norm cap and flies no episode,
  covers these post-contact states:
  - the setpoint is lowvz's, with the lateral PI command anywhere up to its 0.5 m/s cap and the
    gate open or closed;
  - |v_pad,z| ≤ 0.7 m/s (the grid's worst aft-pad `vz_p99` is 0.700 m/s) and |v_pad,x|,
    |v_pad,y| ≤ 0.3 m/s;
  - drone-vs-pad velocity ≤ 0.1 m/s along each axis. P5-D1's in-gap CoM closing speed is about
    0.08–0.10 m/s;
  - deck tilt ≤ 15°, and the drone's body axis within 15° of the deck normal.

  Results:
  - Over that grid, `t · b_z` ≤ **0.040 N**, so every motor is at idle.
  - At 1.0 m/s it is ≤ 0.065 N, so every motor is still at idle.
  - At 0.7 m/s it reaches 0.088 N. The collective is at the floor, but not every motor is at
    idle.
  - With a drone-vs-pad velocity up to 0.2 m/s at 1.5 m/s, it reaches 0.090 N, again with the
    collective at the floor but not every motor at idle.
  - The idle state is **not** unconditional. The norm cap shortens the vertical component when
    the pad falls fast and the lateral command is large. In unphysical corners (drone tilted 30°
    from the deck, falling 0.3 m/s faster than a pad at −0.7 m/s) one motor sits slightly above
    idle.

  `tests/test_control_pid_feedforward_lowvz_cut.py` checks the envelope through
  `DSLPIDControl.computeControl` itself: all four motors are at `MIN_PWM`. It also checks that
  post-latch RPMs in the environment are idle.
- **Why idle and not a partial cut.** A partial cut such as 80–90 % of weight would need a
  magnitude that depends on the drone's velocity, and that magnitude could only be chosen by
  flying. Idle is the saturating end of the tracker. It is also what "throttle cut on touchdown"
  means physically.
- **What idle buys.** At idle the drone's free acceleration along its body axis is −5.63 m/s².
  - The largest aft-pad deck acceleration p99 over the whole grid is 3.06 m/s²
    (`results/deck_stats.csv`), so a latched drone stays loaded against every cell's p99 deck
    acceleration.
  - P5-D1's unloaded lift-offs (mechanism B, deck at −2.1 m/s² median and −3.7 m/s² at most)
    are therefore within its authority.
  - With zero torque, the attitude loop stops fighting a tilted deck. A rim-first drone is laid
    flat by gravity about the loaded rim.
- **Why along the deck normal (the user's wording).** Because D is anisotropic, a setpoint along
  −n̂ tilts the tracker's *target* attitude slightly away from n̂. At idle no torque is applied,
  so this has no effect.

*Pre-registered expectations (written before any flight).*
1. **Pre-contact identity.** On any list, every first-contact quantity is identical per episode
   to `pid_feedforward_lowvz`:
   - touchdown time, closing speed, relative tilt, lateral offset, and both detectors' records;
   - every outcome decided at or before first contact: `crash` before contact, `off_pad`,
     `hard_landing` and `timeout`.

   Only `bounce`, `success` and post-contact `crash` can differ. Any other difference is a bug.
2. **Reach is limited by what the observation shows.** The flag reports only the last substep
   of each 1/30 s control step. In P5-D1's 48 at-impact rocking bounces, the gap opens 4.2 ms
   (6 cases) or 8.3 ms (42 cases) after touchdown, and release is scored 50 ms later. The flag
   therefore reads 1 at a control boundary before the gap only when touchdown falls in the last
   1–2 substeps of a step, which is about 1/8 or 2/8 of cases.
   - **Expected reach:** of lowvz's 64 committed-physics tune-pool bounces, about **25 (≈ 40 %)**
     are within the rule's reach:
     - 11.25 expected of the 48 at-impact bounces;
     - 1.75 of the 4 bounces whose gap opens 12.5–20.8 ms after touchdown;
     - all 12 whose gap opens ≥ 125 ms after touchdown.
   - The other ≈ 60 % open and are scored before the flag can read 1, and **no post-contact rule
     on this observation can reach them**.
   - *Prediction:* `lowvz_cut`'s bounce count is **about 0.6×** lowvz's, and not below about
     0.5×, on the same episodes.
3. **What this means for RL.** An MLP policy sees the same flag. Its purely post-contact gain is
   therefore bounded the same way. A policy that removes the remaining rocking bounces must do
   so **before** contact, by touching down flatter or faster, or by cutting thrust in
   anticipation. That is approach behaviour, which is what H1a scores.
4. **Side effects, expected and not failures:**
   - control effort and action jerk rise, because the action jumps at the latch;
   - post-contact lateral hold is by friction only, about 0.25 × 0.152 N ≈ 38 mN, compared with
     about 2–10 mN under lowvz's near-hover thrust;
   - no new `crash` is expected.

*What may happen next, and what may not.*
- **Allowed:**
  - one sanity check on the **P3-D3 tune pool only**: the committed 180-episode tune draw from
    `dev_pool(cfg)[1]`, seed 20260923, flown by both `lowvz` and `lowvz_cut`;
  - a per-episode check of pre-contact identity and of the latch.
- **Not allowed:**
  - changing the rule because of the check's numbers;
  - running any episode from `results/episodes/`. Those are the `eval-auditor`'s, after commit.
- **Bugs.** If an outright bug appears, such as a latch that never fires, the fix and its reason
  are recorded below this line, and nothing is tuned.

*Provenance.* `resolved_gains_sha256` (`rld.eval.controller_config`) for
`pid_feedforward_lowvz_cut` is `123b890f…`. It carries lowvz's gains, `gains_from` and
`cut_speed_m_s`. lowvz's own resolved hash is unchanged at `19b1f9c8…`, the value in
`results/e01/summary.csv`.

*Tune-pool sanity check (run 19:39–19:40 local, after the entry above was written; not an
evaluation).*
- **What was flown.** The committed P3-D3 tune draw: `dev_pool(cfg)[1]`, `tuning_seed`
  20260923, 60 episodes each at SS3–SS5, 180 in total. Both controllers flew every episode in
  the same environment.
  - The script is in the session scratchpad and is not committed.
  - No other episode was flown, and nothing came from `results/episodes/`.
- **lowvz** reproduces `results/e01/tune_pool_final.csv`: 59/60, 60/60 and 59/60. Its two
  losses are `bounce`, at SS3 #10 and SS5 #35.
- **`lowvz_cut`** also scores 59/60, 60/60 and 59/60, and **no episode changes outcome**. The
  same two episodes bounce.
  - The latch **never fired** in either of them. Both are P5-D1 at-impact rocking bounces
    whose gap opens 8.3 ms after touchdown (`bounce_triggers.csv`).
  - In both, the flag never read 1 at a control boundary before release was scored. This is
    the out-of-reach case of expectation 2, not a bug.
  - With n = 2 bounces this check says nothing about the ≈ 0.6× prediction. The frozen lists
    will test it.
- **Pre-contact identity.** Actions before the latch are bit-identical to lowvz's in
  **180/180** episodes. All 13 first-contact columns of `EpisodeRecord.as_row` are identical in
  **180/180**: touchdown times from both detectors, closing speed, relative and absolute tilt,
  lateral offset, deck tilt and contact count.
- **Latch timing.** The latch fired in 178/180 episodes: every success.
  - Median delay after touchdown: 20.8 ms.
  - 124 latched at the first control boundary after touchdown.
  - 54 latched at the second or third boundary (up to 95.8 ms). In these, a contact gap
    shorter than the grace hid the flag at the first boundary.
- **Idle.** All four motors were at idle RPM (9 440.3) in **2 610/2 610** post-latch control
  steps.
- **Side effects** (median per-episode ratio to lowvz): control effort **+55 %** and action jerk
  **+41 %**.
- **Rule unchanged.** No part of the rule was changed after the check.

### P5-D3 — pid_feedforward_lowvz_cut on the frozen lists (2026-09-23)

*What was flown.* `pid_feedforward_lowvz_cut` (registry name) through the e01 runner, exactly as
e01 flew its five controllers.
- *Lists and N.* Every frozen list, aft pad: static and `id`, `unseen_seastate`, `unseen_heading`
  and `unseen_vessel` at every listed SS. That is 14 cells × N = 200 = 2 800 episodes, with none
  skipped.
- *Timing.* Flown after P5-D2 and before any Phase 5 training run, per the P5-D1 decision.
- *Before the run.* `scripts/make_episodes.py --check` passed 10/10 (MANIFEST SHA-256
  `e6f30e55…`). Config SHA-256 was `e5f19903…` and the resolved hash `123b890f…`, both as
  recorded in P5-D2.
- *Command.*
  `scripts/eval_baselines.py --out-dir results/e01_lowvz_cut --controllers pid_feedforward_lowvz_cut --carry-from results/e02 --carry-methods pid_track_descend pid_feedforward pid_feedforward_lowvz gated oracle_gated --pad aft --workers 24`.
  It ran at `140d55c` in 133.5 s.

*Provenance notes.*
- **The five baselines were carried, not re-flown.** They were carried from `results/e02`, not
  from `results/e01`. `--carry-from results/e01` fails, because e01's `summary.csv` predates the
  Phase 4 `forecaster_files_sha256` column.
  - That first attempt did fly all 2 800 `lowvz_cut` episodes. It then raised in the carry step
    **before writing anything**, so no number from it was seen.
  - The run above is its only re-run, with nothing changed except the carry source.
  - The script's reference check compared every carried aft row with `results/e01/episodes.csv`
    (`c5852090…`): **14 000/14 000 byte-identical** (`run_info.json["reference_check"]`).
  - All paired statistics below read lowvz and `pid_feedforward` directly from
    `results/e01/episodes.csv`. `results/e01/` and the `make baselines` pin are untouched.
- **Dirty tree at run start.** `run_info.json` records `git_dirty = true`. The dirty paths are:
  - the user's two `.claude/skills/*.md` files;
  - this entry's uncommitted eval-side files: `src/rld/eval/report.py` (a display label only),
    `tests/test_eval_report.py` and the new `src/rld/eval/paired_outcomes.py`.

  Nothing under `src/rld/control/`, `src/rld/envs/` or `configs/` was dirty, and none of the
  dirty files is on the flight path.
- **Re-rendering.** The markdown was re-rendered with `--render-only` after the display label
  was added. Both it and every CSV re-render or re-score **byte-identically**.

*Artifacts, all in `results/e01_lowvz_cut/`.*

| file | SHA-256 |
|---|---|
| `episodes.csv` | `8fe15278…` |
| `summary.csv` | `180eeef9…` |
| `success_vs_seastate.md` | `e59f5008…` |
| `paired_vs_lowvz.csv` | `87cd0433…` |
| `p5d2_predictions.csv` | `da593380…` |
| `paired_vs_pid_feedforward.csv` | `e1234e05…` |

- *Tables.* `success_vs_seastate.md` prints all six controllers per cell, with the full outcome
  breakdown and touchdown audit. `oracle_gated` is marked privileged there.
- *Scoring.* The paired statistics come from `python -m rld.eval.paired_outcomes`. They use
  P3-D1 §4's paired bootstrap: 10 000 replicates, seed 20260926, one "seed" per deterministic
  controller, with episodes resampled and shared by both methods.

**Headline: success per cell (%, Wilson 95 % CI, k/200).** The paired difference is
`lowvz_cut − lowvz`, in points, with its paired-bootstrap 95 % CI. Success is never pooled
across sea states.

| cell | `lowvz_cut` | `lowvz` | `pid_feedforward` | cut − lowvz |
|---|---|---|---|---|
| static | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | +0.0 [+0.0, +0.0] |
| id SS3 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | +0.0 [+0.0, +0.0] |
| id SS4 | 98.5 [95.7, 99.5] 197 | 98.0 [95.0, 99.2] 196 | 100.0 [98.1, 100.0] 200 | +0.5 [+0.0, +1.5] |
| id SS5 | 98.0 [95.0, 99.2] 196 | 95.5 [91.7, 97.6] 191 | 99.0 [96.4, 99.7] 198 | **+2.5 [+0.5, +5.0]** |
| id SS6 | 88.0 [82.8, 91.8] 176 | 85.0 [79.4, 89.3] 170 | 90.5 [85.6, 93.8] 181 | **+3.0 [+1.0, +5.5]** |
| unseen_seastate SS6 | 89.5 [84.5, 93.0] 179 | 86.0 [80.5, 90.1] 172 | 90.0 [85.1, 93.4] 180 | **+3.5 [+1.0, +6.0]** |
| unseen_heading SS3 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | +0.0 [+0.0, +0.0] |
| unseen_heading SS4 | 99.0 [96.4, 99.7] 198 | 99.0 [96.4, 99.7] 198 | 100.0 [98.1, 100.0] 200 | +0.0 [+0.0, +0.0] |
| unseen_heading SS5 | 97.0 [93.6, 98.6] 194 | 96.5 [93.0, 98.3] 193 | 99.5 [97.2, 99.9] 199 | +0.5 [+0.0, +1.5] |
| unseen_heading SS6 | 83.0 [77.2, 87.6] 166 | 70.0 [63.3, 75.9] 140 | 77.0 [70.7, 82.3] 154 | **+13.0 [+8.5, +18.0]** |
| unseen_vessel SS3 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | +0.0 [+0.0, +0.0] |
| unseen_vessel SS4 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | +0.0 [+0.0, +0.0] |
| unseen_vessel SS5 | 99.5 [97.2, 99.9] 199 | 99.5 [97.2, 99.9] 199 | 99.5 [97.2, 99.9] 199 | +0.0 [+0.0, +0.0] |
| unseen_vessel SS6 | 97.0 [93.6, 98.6] 194 | 95.0 [91.0, 97.3] 190 | 98.5 [95.7, 99.5] 197 | **+2.0 [+0.5, +4.0]** |

- *Where the difference separates.* It separates from 0 (bold) in 5 of 14 cells. Where it does
  not, the CI touches 0 or the cell has no losses to remove.
- *Outcome breakdown.* Only `hard_landing` and `bounce` occur, for both controllers. There is no
  `crash`, `off_pad` or `timeout` in any cell. Counts, `lowvz_cut` / `lowvz`:

  | cell | `hard_landing` | `bounce` |
  |---|---|---|
  | id SS4 | 0 / 0 | 3 / 4 |
  | id SS5 | 0 / 0 | 4 / 9 |
  | id SS6 | 9 / 9 | 15 / 21 |
  | unseen_seastate SS6 | 7 / 7 | 14 / 21 |
  | unseen_heading SS4 | 0 / 0 | 2 / 2 |
  | unseen_heading SS5 | 0 / 0 | 6 / 7 |
  | unseen_heading SS6 | 24 / 24 | 10 / 36 |
  | unseen_vessel SS5 | 0 / 0 | 1 / 1 |
  | unseen_vessel SS6 | 0 / 0 | 6 / 10 |

  Every other cell is 200/200 success for both.
- *Which way outcomes changed.* Every one of the 50 changed outcomes is `bounce → success`.
  None goes `success → bounce`, and there is no new `crash`.
- *Closing speed.* p95 closing speed is identical to lowvz in every cell, by prediction 1: for
  example 0.188 m/s at `id` SS5.

**P5-D2 predictions, scored per episode on the identical lists** (`p5d2_predictions.csv`).
1. **Pre-contact identity: HELD.**
   - All 16 first-contact columns are text-identical (hence float64-identical) to lowvz in
     **2 800/2 800** episodes:
     - the 12 first-contact fields of `EpisodeRecord.as_row`;
     - `detectors_disagree`, `closing_speed_world_z_m_s`, `time_to_touchdown_s` and
       `td_in_quiescent_window`.
   - The 40 episodes whose outcome is decided at or before first contact (all `hard_landing`)
     have the identical outcome in **40/40**.
   - *Not pre-registered; reported, not scored.* `termination_reason` also agrees in 35/40. In
     5 SS6 `hard_landing` episodes (relative tilt 15.2–22.3° at contact), lowvz rocked off
     (`release`) and `lowvz_cut` stayed down until `dwell_complete`. The class is decided at
     contact, and only how the episode ended changed. That is post-contact behaviour, not a
     pre-contact difference, and not a bug.
   - *A check made stricter, then reverted.* My first scoring code also required
     `termination_reason` to match. I reverted to P5-D2's wording ("outcome") after seeing the
     5, and both numbers are in the CSV.
2. **Bounce ratio ≈ 0.6×, not below ≈ 0.5×: HELD.**
   - `lowvz_cut` bounces **61** times where lowvz bounces **111**. The ratio is **0.550**, with
     a paired stratified-bootstrap 95 % CI of **[0.458, 0.640]**. Episodes are resampled within
     each cell and shared by both methods, 10 000 replicates, seed 20260926.
   - *The decision rule.* The rule "point ≥ 0.5 and the CI contains 0.6" was written into
     `rld.eval.paired_outcomes` before the scoring was first run. The point is 0.050 above the
     floor, and the CI's lower end (0.458) lies below it.
   - *The removed fraction.* The cut removed 50/111 = 45 % of bounces, against P5-D2's expected
     reach of ≈ 40 %.
   - *Heterogeneity (post hoc, descriptive).* The per-cell ratio is 0.28 at `unseen_heading`
     SS6 (10/36) and 0.44–1.0 in every other cell with bounces. That one cell supplies 26 of the
     50 removed bounces, and without it the pooled ratio is 51/75 = 0.68.
   - *A candidate explanation, not verified.* At 90° heading the deck is roll-dominated, and
     P5-D1's mechanism B (a later, unloaded lift-off) is within the rule's reach. No
     substep-level log was taken on the frozen lists.
3. **Control effort and action jerk rise: HELD.**
   - The paired mean difference (cut − lowvz) has a 95 % CI above 0 in **13/13** moving-deck
     cells for both metrics, and also on the static pad.
   - *Operationalisation.* P5-D2 gave only a direction. The rule "CI above 0 in every
     moving-deck cell" was written before the scoring was first run.
   - *Size.*
     - `effort_mean_sq`, the committed per-cell metric (mean ‖a‖²): median per-episode ratio
       3.9–7.1× by cell.
     - `action_jerk_mean`: 1.32–2.40×.
     - The per-episode sums `control_effort` and `action_jerk` give median ratios of 1.55× and
       1.53× over all 2 800. That matches P5-D2's tune-pool "+55 %" if that figure used the
       sum; P5-D2 does not name the column.

**Not predicted by P5-D2.**
- **Tunnelling.** 31 `lowvz_cut` episodes exceed the 5 mm `tunnelling_penetration_m` threshold;
  lowvz has **0**. The maximum penetration is 7.03 mm, against 3.91 mm for lowvz.
  - *Where.* `id` SS5 1, `id` SS6 5, `unseen_seastate` SS6 7 and `unseen_heading` SS6 18.
  - *Outcomes.* 17 are `hard_landing`, 13 `success` and 1 `bounce`. Tunnelling does not enter
    the outcome classification, and no outcome is attributed to it.
  - *Likely cause, not verified.* A drone at idle thrust on a deck accelerating up into it
    loads the contact harder than lowvz's near-hover thrust does.
  - *Comparison.* It is of the same size as `pid_track_descend`'s e01 tunnelling (14, max
    6.5 mm).
  - *Consequence for RL.* It matters for Phase 5/6: a policy that cuts thrust after contact
    will show tunnelling counts that lowvz does not. The Phase 7 tables must report
    `tunnelling_n` beside success for every method, as P3-D1 §3 already requires.
- **Detector disagreement.** It is 1/2 800, at `unseen_seastate` SS6, the same episode as
  lowvz's, since first contact is identical.

**Post hoc, not part of any prediction: `lowvz_cut` vs `pid_feedforward`**
(`paired_vs_pid_feedforward.csv`, same bootstrap).
- It separates in **no** cell. The largest difference is `unseen_heading` SS6, +6.0 points
  [−1.0, +13.0]; `id` SS6 is −2.5 [−7.5, +2.5].
- A post-contact rule on lowvz's approach therefore does not demonstrably beat the
  success-tuned `pid_feedforward` anywhere on these lists.

**What this means for H1 (P3-D1 §8 and P3-D4 are unchanged).**
- **lowvz stays the H1a reference.** `lowvz_cut` is printed beside it.
- **The P5-D1 consequence is now quantified.** A purely post-contact thrust rule, with lowvz's
  approach bit for bit and no tuning, gains **+2.5 points [+0.5, +5.0]** of success over lowvz
  at `id` SS5 (H1a's cell), at the identical p95 closing speed. So a `residual_ppo` that
  matched lowvz's approach and learned only this cut would clear H1a's −2-point non-inferiority
  bound by a margin, but would show r = 0 on closing speed.
- **H1b's reference is untouched.** At `id` SS6, `lowvz_cut` (88.0 %) is below
  `pid_feedforward` (90.5 %), −2.5 [−7.5, +2.5] paired.

*Code (eval side only).*
- `src/rld/eval/paired_outcomes.py` is new: episode pairing, first-contact identity, the paired
  stratified ratio bootstrap, the per-cell paired table, and the `python -m` entry point.
  `scripts/` is outside the eval-auditor's write scope; the main thread may move the entry
  point there.
- `tests/test_eval_paired_outcomes.py` is new.
- `METHOD_LABELS` in `src/rld/eval/report.py` gains the `pid_feedforward_lowvz_cut` label.

## Gates
| gate | date | result | note |
|---|---|---|---|
| 0 | 2026-09-21 | PASSED | `make test lint` green (6 tests, 53 s); pybullet 3.2.7 built from sdist and opens a DIRECT client on 3.12.3; `results/env_throughput.csv` written (8 rows, 1/8/16 SubprocVecEnv workers x rpm/vel); both submodule SHAs recorded in P0-D1. |
| 1 | 2026-09-21 | PASSED | `make test lint` green (66 tests, 62 s; ruff + ruff-format + mypy --strict clean). `results/deck_stats.csv` 192 rows = 96 cells x {aft, cg}, full grid (2 vessels x SS3-SS6 x 4 headings x 3 speeds), all 2304 realizations, with `z_std_model_m`, `vz_std_model_m_s`, `vz_p99_model_m_s`, `az_p99_model_m_s2` populated and no NaN; plus `deck_stats_seeds.csv` (4608 rows) and `deck_feasibility.csv` (2 rows). lambda = 1/25 and r_pad = -0.4*L confirmed in P1-D1. Feasibility rule PASS: frigate SS6 180 deg 12 kn aft vz p99 = 0.577165 m/s vs the 2.08333 m/s gate threshold, 3.61x inside; the rejected `SPEED_LIMIT` reading is recorded as FAIL beside it. Sign convention and ZYX rotation order corrected and pinned by three hand-computed cases (P1-D2); one project-wide lambda (P1-D3). |
| 2 | 2026-09-22 | PASSED | `make test lint` green (118 passed, 1 skipped, 107 s; ruff + ruff-format + mypy --strict clean). The 1 skip is by design: `tests/test_platform.py` gates only the configured driver and *measures* the other, and `test_constraint_driver_fails_the_tracking_gate` asserts the rejected one fails. Platform tracking, `kinematic` driver, 10 s model at frigate SS6 180 deg 12 kn aft: body-origin 0.000 mm and plate-corner 2.2e-16 m against the 1 mm gate, orientation 3.0e-8 rad, `getBaseVelocity` linear and angular ratio 0.0 % against the 2 % gate; `constraint` fails every one of those by an order of magnitude and cannot be tuned into passing (24-point sweep, four identical digits) -- P2-D1. `gymnasium.utils.env_checker.check_env` passes; reset determinism bit-identical as the env's 1st and 3rd reset; drop on a static pad registers exactly one touchdown; scripted 0.3 m/s descent on a static pad 100/100 `success`. `results/e00_env_sanity.csv` (4 rows) + `results/e00_env_sanity_episodes.csv` (800 rows) written from the `id` split's val partition, draw seed 20260922: hover 1.000 `timeout` at SS3 and SS5, random 1.000 `crash` at both, no successes under either -- outcome fractions sum to 1.000000 per row. **PyBullet-vs-analytic touchdown disagreement 0/800 = 0.0000** against the < 1 % gate, plus 0/100 on scripted static descents and 0/30 on a moving SS5 deck. `results/env_throughput_landing.csv` (8 rows) records the deck-in-the-loop throughput P3-D1 must size from (P2-D8); `results/env_throughput.csv` untouched. Two flags carried forward, both recorded rather than fixed: one of 800 sanity episodes tunnelled (7.86 mm penetration vs the 5 mm threshold) and it was the only random episode that ever reached contact, so the random arm exercises the touchdown path barely at all -- the scripted tests carry that load; and the plan's "dropped from rest" is not expressible in a velocity-setpoint action space, so the drop test uses the maximum commanded descent (P2-D9). |
| 3 | 2026-09-22 | PASSED (2nd attempt) | First attempt FAILED on the `results-skeptic` review (BLOCKING B1: the restated H1 was beatable by a slower PID; MAJOR M1-M3), before any RL run; remediated with user decisions (P3-D1 revision 1 §9, P3-D3 amendments). Second attempt: `make test lint` green (233 passed, 1 skipped by design, 162 s; ruff + ruff-format + mypy --strict clean). `pid_feedforward` 200/200 on the static pad and 200/200 at `id` SS3 (Wilson [98.1, 100.0] each) against the >= 95 % gate, from `results/e01/episodes.csv` at `d35f224`. Success-vs-sea-state table `results/e01/success_vs_seastate.md` committed: 5 controllers x 14 cells x N = 200, re-renders byte-identically from `summary.csv`; no crash or off_pad anywhere; detector disagreement 5/14000; tunnelling 14 (all `pid_track_descend`, max 6.5 mm). Frozen lists `results/episodes/` committed at `0780aaa`, MANIFEST SHA-256 `e6f30e55e478d39061b94e38de33959ca99b16e8cac38a3bd99b34f6543ad4a2`, `--check` 10/10 OK. P3-D1 FROZEN revision 1, block SHA-256 **`21465588610e65f1d1253bd26e5ce5db1da938889c6042338e1de8d4d99f5ed2`** (from `### P3-D1 — FROZEN` to the line before `### P3-D2`, UTF-8). Second `results-skeptic` review: no BLOCKING; MAJOR-1 (H1 scope) and MINOR 1-9 folded in by P3-D4 errata without touching the frozen block; three items carried to Phase 4 (P3-D4). |
| 4 | 2026-09-23 | PASSED | Criteria checked against committed artifacts at `eeb87d5` plus the Gate 4 wording fixes. `make test lint` green: 320 passed, 1 skipped by design (P2-D1), 313 s; ruff, ruff-format and mypy --strict clean. `tests/test_deck_forecast.py` and `tests/test_control_gated_forecast.py` 56/56 with **0 skips**, artifacts present (a fresh clone without `artifacts/dmf/` would skip parity, so this run is the evidence). **Parity:** `results/forecast/parity.csv`, 18 gated checks over 6 model dirs, all pass at `1e-4·max(1,|y|)`, worst 0.041× tolerance; bridge-fed parity needs dmf's float32 cast (P4-D3). **Causality:** the feed-clock spy and the future-perturbation bit-identity tests pass. **Leakage:** 729 train / 135 tune forecaster keys vs 1 273 frozen-list realizations, overlap 0 (recomputed independently from `fit_keys.json`). **Frozen-list results:** `gated_forecast` and `gated_forecast_tcn` committed in `results/e02/` beside `gated`, `oracle_gated` and the three PID baselines, 26 cells × 200 each at aft and CG, with `td_in_quiescent_window`, quiet landings per listed episode and Wilson CIs; the static list is not run for feed controllers, reason recorded. Phase 3 aft rows byte-identical to e01 (14 000/14 000). Secondary seed arms in `results/e02_tcn_seeds/` (`12fd9f5`, clean). Oracle relabel done; P4-D5 records the hash. MANIFEST 10/10; P3-D1 block SHA-256 unchanged. **Results-skeptic:** first review MAJOR M1–M5 remediated (P4-D3 wording, P4-D4a erratum, P4-D5, M1 columns, seed arms); second review no BLOCKING and no MAJOR, 6 MINOR wording and provenance errors fixed in P4-D4/P4-D4a. Headline (P4-D4): pre-registered expectation 1 held, 2 600/2 600 aft `gated_forecast` timeouts; forecast gating lowers success against `gated` in 19 (DLinear) and 9 (TCN) of 26 cells, and improves it in none. |
