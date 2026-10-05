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

### P5-D4 — PPO smoke run: exploration noise at the SB3 default flips the drone; the smoke run counts as one tuning trial (2026-09-23)

*Run.* `make train-bg CFG=configs/rl/ppo_smoke.yaml SEED=0`, commit `51bee6f`. The only dirty
paths were the user's two `.claude/skills` files. Budget 200 k steps, done at 212 992, 739 s wall,
0.57 CPU-hours. It trained on the P3-D2 train pool and was evaluated every 20 k steps on the P3-D3
tune draw (180 episodes, SS3–SS5). No frozen-list episode was read. The curriculum stayed at SS3.

*Plumbing: healthy.*
- `status.json` reached `done` with no NaN.
- Each checkpoint holds both `model.zip` and `vecnormalize.pkl`, and `load_policy(run_dir,
  "final")` reloads the policy.
- All six outcome classes occurred across the evaluations.

*Learning: not healthy.*
- The PPO base config left `log_std_init` at SB3's default of 0, i.e. an initial action std of 1.0
  on a Box(−1, 1) action. The training policy was therefore close to the P2-D9 random policy,
  which crashes in 100 % of episodes.
- In the smoke run, 4 152 of 4 157 training episodes (99.9 %) ended in `crash` after about 50
  control steps. 4 088 of those crashes were `tilt_gt_crash`. Only 5 training episodes touched
  down.
- The deterministic evaluation policy did reach the deck. Its success peaked at 0.133 at 60 k
  steps and ended at 0.006, and its p95 closing speed rose from 0.40 to 0.98 m/s.
- Binned training return rose only from about −42 to −37, all of it on crash episodes.

*Throughput.* 288 steps/s overall and 467 excluding evaluation. The host was about 91 % idle, so
the run was stalled, not compute-bound. This goes to engineering; it is not a protocol matter.

*Decision (user, 2026-09-23).*
- The PPO base config sets `log_std_init = −1.0` (std ≈ 0.37).
- `tune_ppo.yaml` searches `log_std_init` uniformly in [−2.3, −0.7] (std ≈ 0.10–0.50).
- The base config was changed after a tune-pool training run was seen, so **the smoke run counts
  as one of PPO's 20 trials**. The PPO search draws at most 19 trials, trial 0 being the new base.
- SAC is unchanged: it keeps SB3's `log_std_init = −3` with automatic entropy tuning, and all 20
  of its trials.
- A second 200 k smoke run with the new setting must show touchdowns in training episodes before
  any tuning trial launches. That second smoke run is a check of the change, not a trial.

### P5-D5 — Reset speed-up, bit-identical (2026-09-23)

*Why.* User-approved (2026-09-23) as engineering before Phase 5 tuning. P5-D4's smoke run was
stalled, not compute-bound: 467 steps/s excluding evaluation on a host that was ~91 % idle. A
step costs 1.6 ms but a reset cost ~95 ms. With early-training episodes at ~50–70 control steps,
SubprocVecEnv lockstep made training reset-bound. P2-D8 set aside the fix, a lazily evaluated
deck, "unless Phase 5 wants it, with its own numerical-equivalence entry". This is that entry.
**No evaluation result, config, success criterion or episode changes.**

*Change 1: the deck trajectory is synthesised lazily, in chunks.*
- `rld.envs.platform.LazyDeckTrajectory` replaces the eager 3 001-sample `build_trajectory`
  call in `DeckLandingAviary._prepare_episode`.
- The grid is the same one as before. It is cut into chunks of 240, 240, 480, 960 and 1 081
  samples. A chunk is synthesised the first time any of its samples is read, then cached.
- A reset therefore synthesises only the chunk that holds sample 0.
- Any consumer can still read any sample at any time. `build_trajectory` itself is unchanged;
  it stays the eager reference, and `PrivilegedContext.from_env` still calls it.

*Why a plain slice would have changed bits, and what was done instead.* A plain slice of the
grid is **not** bit-identical to the full grid (measured):
- The two bridge angular accelerations reduce over the 299 wave components with a
  `(m, 299) @ (299, 2)` product. On the pinned OpenBLAS 0.3.34 (SkylakeX kernel), that product
  uses the small-matrix kernel while `m·2·299 ≤ 10⁶` (m ≤ 1 672) and the blocked kernel above
  that, and the two kernels differ by up to 8e-14.
- Evaluating 240-sample chunks directly changed 13 of 13 aft chunks.

What was done instead:
- **`rld.deck.bridge.harmonic_sum_rows`** (an additive, narrow bridge edit). It evaluates
  cos/sin only for the requested rows, then writes them into a reused operand with the **same
  shape and the same row positions** as the full-grid call. BLAS therefore runs the same
  kernel on the same shape, and each output row depends only on its own input row.
- **dmf's seven channels.** `synthesize_motion` is still called unmodified, on the row slice.
  Its `(m, 299) @ (299, 7)` product was measured row-invariant for every m ≥ 2 and not for
  m = 1 (a matrix-vector path; this is P2-D8's observation). `JonswapDeckMotion.deck_point_rows`
  therefore widens a one-row request to two rows.
- Time mapping, kinematics, Froude scaling and quaternions are per-sample.
- Sinusoid and static sources are elementwise in time. Any other source is evaluated eagerly as
  one full-grid chunk.
- **This equality depends on BLAS.** It is proven for the pinned stack and asserted by tests,
  not argued for any BLAS. A different OpenBLAS or CPU kernel would fail
  `tests/test_lazy_trajectory.py` instead of drifting silently.

*Change 2: the drone's visual mesh is not loaded in DIRECT mode.*
- ~21 of the ~23 ms of URDF loading per reset was PyBullet parsing `cf2.dae`, the drone's
  visual mesh.
- `DeckLandingAviary._housekeeping` now mirrors upstream `BaseAviary._housekeeping` line for
  line. The only difference is that the drone is loaded with
  `URDF_IGNORE_VISUAL_SHAPES | URDF_USE_INERTIA_FROM_FILE`. PyBullet then shows the collision
  cylinder as a proxy visual.
- The mirror applies only when `visual_shapes` is False. That is the default in DIRECT mode;
  with `gui=True` or `visual_shapes=True` the environment calls upstream's own method.
- A test pins the SHA-256 of upstream's method source at gym-pybullet-drones `7ebad1e`, so a
  submodule bump forces a re-review.
- `saveState`/`restoreState` and keeping bodies across resets were rejected. Both skip
  `resetSimulation`, so the broadphase and contact caches would carry a history that a fresh
  world does not have, and bit-identity could not be argued.

*Evidence that no bit changed.*

| check | result |
|---|---|
| Lazy vs eager trajectory, `np.array_equal`, every array (t, position, quaternion, and every `DeckPointState` field including accelerations), whole grid | Equal. `tests/test_lazy_trajectory.py`: 6 JONSWAP realizations (both vessels, SS3–SS6, headings 0/45/90/135/180) × aft/CG × t0 ∈ {bottom, top, random} of the start window × 3 chunk layouts (default; first chunk of 1 sample; a one-row tail), plus scrambled-order `sample(i)` reads. Sinusoid and static likewise. Scratch sweeps of 2 × 384 further grids: 0 mismatches. |
| `harmonic_sum_rows` vs `harmonic_sum` rows | Equal, including one-row ranges, both sides of the 1 672-row kernel switch, and ranges straddling a `TIME_CHUNK` block. |
| DIRECT world vs upstream world | Every body id, `getBodyInfo`, `getDynamicsInfo` per link, `getCollisionShapeData`, `getJointInfo`, base pose, base velocity and engine parameter is identical. |
| Episodes, new env vs pre-P5-D5 env (eager deck **and** visual mesh), SS5 moving deck, aft and CG | Observations, rewards, flags, driven deck states and whole episode rows are identical. Covers 3 scripted descents that touch down and 2 Gaussian (std 0.37) policies. |
| Determinism and `check_env` | Pass: the existing static-deck tests, plus a new moving-deck same-seed test. |
| `make env-sanity` → scratch dir vs `results/e00_env_sanity*.csv` | 4 + 800 rows. Every column is byte-identical except `steps_per_s`, `wall_s` and `timestamp_utc`, which are per-run by construction. |
| `make baselines` → scratch dir vs `results/e01/` | `episodes.csv` is **byte-identical** (`cmp`; 14 000/14 000 rows; the runner's own reference check reports `all_identical=True`). `summary.csv`: every metric column is identical. The committed file predates the Phase 4 schema, and `oracle_gated`'s `controller_config_sha256` is the P4-D5 relabel hash. All 70 re-run summary rows are byte-identical to the same rows of the post-relabel `results/e02/summary.csv`. The markdown differs only in its title, which is chosen by output directory name. |

*Speed* (one process, `OMP_NUM_THREADS=1`, 36-core host otherwise idle; frigate SS6 180° 12 kn
aft, the `rld.bench.throughput` landing factory; Gaussian policy std 0.37, i.e. P5-D4's initial
exploration; both runs 292 episodes × 68.5 steps, identical as bit-identity requires):

| | before (`HEAD` = `cbdb7cf`) | after |
|---|---|---|
| reset, mean / p99 | 95.5 / 104.3 ms | **11.9 / 13.3 ms** |
| reset, lazy deck only / URDF change only | — | 32.3 / 72.3 ms (mean) |
| single env, steps/s incl. resets | 330 | **466** |
| 16-worker SubprocVecEnv, steps/s | 685 | **1 754** (2.6×) |
| full-length hover episode (360 steps), ms | 665 | 666 |

- The per-step cost rises from 1.61 to 1.95 ms because later chunks are now paid for inside
  steps.
- A full-length episode breaks even: five chunk overheads (~20 ms) against the ~21 ms URDF
  saving. No regime got slower.
- This resolves P2-D9's open concern that reset cost dominates short episodes, without
  interpolating the corpus: the deck is still evaluated analytically on the physics grid.
- Not changed: `PrivilegedContext.from_env` (in `rld.control`) still does one eager full-grid
  build, which the eval runner pays once per episode for `td_in_quiescent_window`. That is
  evaluation-side cost, not training.
- Also not changed: the docstrings of `rld.bench.throughput` and `scripts/env_throughput.py`,
  which still say "bridge evaluated once per reset".

### P5-D6 — Training-pipeline speed-up: reset prefetch and independent eval workers, bit-identical (2026-09-24)

*What changed (src/rld/rl/ only).*
- **Reset prefetch.** `prefetch_reset: true` is the default. While the current episode runs, each
  training worker prepares its next episode on a second env instance in a background thread.
  - A prefetched episode drawn at an old curriculum stage, or overtaken by an explicit
    `reset(seed=...)`, is discarded. The sampler's RNG state is then restored to what it was
    before the draw.
  - *Tests.* `tests/test_rl_prefetch.py` flies a 1 500-step script on the real env, serial and
    prefetched. Across the whole run it has about 20 resets, 2 explicit seeds, mid-episode stage
    changes including SS5 → SS4, and a switch into queue mode. Every observation, reward, flag
    and info field is equal (`np.array_equal`). The counters show 14 prefetched episodes used and
    5 discarded.
  - The same test also passes through 2 `SubprocVecEnv` workers with an `env_method` stage
    broadcast.
- **Independent eval workers.** Each eval worker flies its share of the fixed P3-D3 tune draw on
  its own; the evaluation no longer runs in lockstep.
  - Worker *i* forwards *n* copies of its observation and takes row *i*. This keeps the float32
    MLP's batch-shape-dependent arithmetic exactly as it was in the lockstep batch.
  - *Verified.* On the four smoke-run checkpoints (180 episodes each), 0 of 24 480 fields differ
    from the lockstep evaluator. That holds with the P5-D5 env changes and without them, and
    with prefetch on and off.
  - The lockstep path is kept as `evaluate_lockstep` and tested against.
- **Forkserver preload** of the heavy imports. Worker PSS falls from 479 to 141 MB, the 16-worker
  pool from 7.7 to 2.4 GB, and start-up from 4.2 to 3.2 s.
- **Slot cost.** It is now `ceil(0.4 × busiest pool) + 1`: 8 per PPO run, down from 17, and 5 per
  SAC run (SAC not measured). The measured time-averaged use of a PPO run is about 6.4 cores. So
  `MAX_WORKERS = 34` fits four PPO runs, about 26 busy cores. When two runs' eval bursts coincide,
  those evaluations slow down; nothing else changes.
- **Bug fixed.** In `eval_episodes.csv`, each episode's own `steps` overwrote the training step.
  The training step is now `train_steps`.
  - `ppo_smoke/0/eval_episodes.csv` keeps the old column. Its training step is recoverable
    through `eval_index` → `evals.csv`.
  - `eval_env_steps` now counts steps actually flown, excluding idle replays.
- **Legacy config shim.** `load_policy` fills `ppo.log_std_init = 0.0` into a saved run's config
  when the key is absent. Those runs trained at SB3's default before P5-D4, and a loaded model's
  `log_std` comes from `model.zip` in any case.
  - The shim runs only at load; training configs must state the key.
  - `ppo_smoke/0` loads again; it was broken by `cbdb7cf`. A test checks that the weights are
    identical with and without the key.

*Throughput* (36-core host, measured with the host otherwise quiet, 16 workers, train pool SS3,
std 0.37 random policy):

| setup | steps/s |
|---|---|
| before (HEAD `cbdb7cf`) | 662–685 |
| P5-D5 env changes, prefetch off | 1 768 |
| P5-D5 env changes, prefetch on | 2 057 |

- With a PPO learner in the loop, P5-D5 plus prefetch runs at 1 861 steps/s rollout-only and 1 597
  including updates. The smoke run did 467 excluding eval.
- A 180-episode tune evaluation takes about 5 s, down from 22–33 s.

*A known numerical non-comparability (recorded, not changed).* The float32 MLP is not
batch-invariant: batch 1 and batch 16 differ by up to 3.6e-7.
- `LearnedPolicy.act`, the frozen-list evaluation path, runs at batch 1. The in-training tune
  evaluator uses the batch shape above.
- So a checkpoint's tune-pool numbers and its `make eval` numbers are not bit-comparable at the
  last float32 bit.
- They are never compared directly: tune-pool numbers select hyperparameters, and frozen-list
  numbers are the results.

### P5-D7 — Second smoke run still unhealthy; the 30 Hz loop tolerates exploration std ≲ 0.15; PPO base std 0.10 (2026-09-24)

*Run.* `make train-bg CFG=configs/rl/ppo_smoke_p5d4.yaml SEED=0` at commit `e075c51`. It uses
`log_std_init` −1.0 (std 0.37) and the P5-D5/D6 speed-ups, and is otherwise identical to the first
smoke run. It finished `done` at 212 992 steps in 309 s, at 1 430 steps/s excluding eval (the first
run: 467).

*Learning: still not healthy.*
- 1 928 of 1 952 training episodes (98.8 %) crashed, 1 824 of them by `tilt_gt_crash`. Only 24
  episodes ended in a touchdown class.
- Episodes did get longer: 69 control steps on average in the first quarter of training, 172 in
  the last, with 11.7 % timeouts in that last quarter.
- Deterministic tune-pool success peaked at 0.044 (140 k) and ended at 0.000.

*Diagnosis.* A scratch probe, not a trial, run on train-pool realizations only (frigate SS3,
180/135°, 6 kn, seeds 0–7, 48 episodes per cell). It flew zero-mean Gaussian action noise, with
and without a constant descent command of −0.2 (0.3 m/s).

| noise std | i.i.d., hover | AR(1) ρ = 0.9, hover | i.i.d. + descent |
|---|---|---|---|
| 0.05 | 100 % timeout | 100 % timeout | 25 % crash, 60 % off_pad, 15 % success |
| 0.10 | 100 % timeout | 94 % timeout | 29 % crash, 62 % off_pad, 8 % success |
| 0.20 | 54 % crash | 31 % crash | 42 % crash |
| 0.37 | 100 % crash | 98 % crash | 90 % crash |

- A second scratch probe held each command for 1/30 s, but ran `DSLPIDControl` at 240 Hz instead
  of the frozen 30 Hz.
  - At std 0.2 (hover) crashes fell from 54 % to 0 %.
  - At std 0.37 they fell only from 100 % to 85 %.
  - With the descent command and std 0.05 they went from 25 % to 19 %.
- **Reading.**
  - The frozen 30 Hz control loop (the attitude loop runs once per policy step) makes the drone
    lose attitude under white-noise velocity commands above about std 0.15 (≈ 0.2 m/s).
  - A std of 0.37 is too large even with a fast inner loop.
  - The crashes under a plain descent with little noise happen around contact. That is landing
    difficulty for the policy to learn, not exploration.

*Decision (user, 2026-09-24).*
- **The environment stays frozen:** 30 Hz control rate, P2-D2 action space.
- **The PPO base is `log_std_init = −2.3` (std 0.10).** The search range is [−3.0, −1.6]
  (std 0.05–0.20). This supersedes P5-D4's −1.0 and [−2.3, −0.7].
- **The second smoke run also led to a change of the base config, so it too is charged as a
  trial.** PPO: 2 charged + 18 drawn = 20 (`trials_used_before_search: 2`, `trials: 18`). SAC
  stays at 20 drawn: SB3's `log_std_init` is −3 (std 0.05), and the automatic entropy target
  −dim(A) = −3 corresponds to a pre-tanh std of about 0.09, inside the tolerated band.
- **The smoke configs keep the values their runs actually flew with.** `ppo_smoke.yaml` records
  0.0 and `ppo_smoke_p5d4.yaml` records −1.0. A third smoke run, `ppo_smoke_p5d7.yaml` at −2.3,
  must show touchdowns in training episodes before tuning launches.
- **Rejected by the user:** running `DSLPIDControl` at the 240 Hz physics rate. It would need a
  dated P3-D1 deviation and a re-run of every committed baseline.

### P5-D8 — Third smoke run learns; one update after the SS3→SS4 promotion collapses it (2026-09-24)

*Run.* `make train-bg CFG=configs/rl/ppo_smoke_p5d7.yaml SEED=0` at `ed5d7ca`, with
`log_std_init` −2.3. It finished `done` at 212 992 steps in 291 s, at 1 556 steps/s excluding
eval.

*Learning: healthy up to the promotion.*
- In training episodes, crashes fell to about 0 (5 `off_plate_strike`), and 560 of 753 episodes
  ended in a touchdown class. By training quarter, success went 0.011 → 0.043 → 0.181 → 0.234
  and mean return −2.4 → +8.5.
- Deterministic tune-pool success, SS3–SS5, rose monotonically from 20 k to 200 k: 0, 0, .011,
  .106, .206, .267, .322, .411, .567, .744. At 200 k, SS3 was .983, SS4 .767 and SS5 .483.
- The curriculum promoted SS3 → SS4 at 200 k, mid-rollout.

*The collapse.*
- At the final evaluation (212 992), one PPO update later, success was 0.000 at every SS:
  timeout .661, bounce .339.
- *Confirmed outside the training pipeline.* Flying the SS3 tune draw serially through
  `load_policy`:
  - the 200 k checkpoint scores 58/60;
  - the `final` checkpoint scores 0/60 (46 timeout, 13 bounce, 1 off_pad).
- *Swap test.*
  - final weights + 200 k VecNormalize stats → 0/60;
  - 200 k weights + final stats → 57/60.

  So the weights carry the collapse, not the normalisation.
- *What moved.* The update's logged learning rate was 5.09e-6 (the linear schedule; it is not
  negative). The largest weight change was about 1.5e-3. SB3 initialises the action head with
  gain 0.01, and its largest weight here is 9.5e-3, so a change of that size moves the
  deterministic mean action by an amount that matters at this action scale.
- *Not established.* Whether the SS3/SS4 mix in the update's rollout caused the collapse, or it
  is ordinary PPO instability, is not known.

*Why it matters.* P3-D1 §5 scores tuning trials on their **final** checkpoint. If collapses like
this recur, the scores are noisy.

*Decision (user, 2026-09-24).* Before tuning, run a 1 M-step diagnostic with the same config:
`configs/rl/ppo_diag_p5d7.yaml`, which differs only in budget, evaluation interval (25 k),
checkpoint interval and name. It is not a trial. Any config change that follows from it will be
charged as one, as in P5-D4 and P5-D7.

### P5-D9 — 1 M diagnostic: the collapse is hovering, not the curriculum; SAC device; tuning pre-registered (2026-09-24)

*Diagnostic run.* `make train-bg CFG=configs/rl/ppo_diag_p5d7.yaml SEED=0` at `73f7ea3`. It
finished `done` at 1 015 808 steps in 1 049 s (2.20 CPU-hours) and was evaluated every 25 k on
the P3-D3 tune draw.
- **Learning, then hovering.** Tune-pool success peaked at 0.27 (150 k). It then fell to 0.00 at
  200 k **with no promotion** (still SS3), with 83 % timeout.
- **The same shift in training.** By 25 k-step bins, the timeout share of training episodes went
  0.27 → 0.48 → 0.58 → 0.75 → 0.85 from 150 k to 250 k, while training return rose.
- **Reward economics that make hovering the better bet**, mean training return by outcome:
  - success +53.4;
  - bounce, off_pad and hard_landing −6.0 to −7.1;
  - timeout +1.05 (p10 −3.5, p90 +3.8).

  Attempting a landing therefore beats hovering only if p(success) exceeds about 12 %. The
  stochastic policy (std ≈ 0.10) converted only 26 of 489 touchdowns (5 %).
- **Recovery.** The policy climbed out of hovering on its own:
  - success 0.34 (400 k), 0.61 (450 k), 0.79 (475 k);
  - promoted SS3 → SS4 at 525 k (0.88) with **no collapse**, and SS4 → SS5 at 575 k (0.97);
  - then 0.96–1.00 at every evaluation from 600 k to 1 015 808 (final 0.994).
- **Final checkpoint, tune draw.**
  - 179/180 successes and 1 bounce;
  - p95 closing speed 0.340 m/s (p50 0.227);
  - median time to touchdown 2.0 s;
  - 0 tunnelling, 0 detector disagreement, 0 timeout.
  - For reference, P3-D4 table on the same draw: `pid_feedforward` 1.000 / 1.000 / 1.000 at p95
    0.22–0.25 m/s and 4.5 s; `pid_feedforward_lowvz` p95 0.13–0.17 m/s and 7.7 s. These are
    tune-pool numbers, not results.
- **Reading.**
  - P5-D8's post-promotion collapse was the same hover trap, which occurs with or without a
    promotion. A 2 M-step trial ends well after the recovery point observed here.
  - The reward-weight search (w_time up to 0.03, r_fail 5–30) spans both hover-favouring and
    landing-favouring settings, so the hover incentive is a hyperparameter property that the
    pre-registered search can select against.
  - The Phase 5 hacking audit must report the timeout fraction of training episodes, not only
    of evaluation episodes.

*SAC device (measured once, as the rl-trainer rule requires).* `scripts/p5_sac_device_bench.py`,
committed `configs/rl/sac.yaml`, 30 000 steps per device into scratch run directories:
- CPU 86.3 steps/s excluding eval, CUDA 87.0 steps/s. The two are equal because the run is
  bound by one gradient step per transition collected (UTD = 1), not by device compute.
- **Fixed: `device: cpu`** (unchanged), which leaves the GPU free.
- Expected SAC trial wall time is about 1.6 h (0.5 M steps). SAC tuning is the long pole.

*Tuning, pre-registered here before trial 1 (user decision, 2026-09-24: launch as registered).*

| method | config (SHA-256) | trials drawn | charged before | per-trial budget | Sobol seed | trial seed |
|---|---|---|---|---|---|---|
| PPO | `configs/rl/tune_ppo.yaml` (`77aaa7f11a6690f2b5806900c3e3917a3385d04a55f1f2e2faee9316f9fad95d`) | 18 | 2 (P5-D4, P5-D7) | 2 M | 20260927 | 0 |
| SAC | `configs/rl/tune_sac.yaml` (`3eded2be9a6ed7bed7981498dd5d9c93b52eb82ad627d64d81d651fa5d60abfe`) | 20 | 0 | 0.5 M | 20260928 | 0 |

- **Base configs.**
  - `configs/rl/ppo.yaml`: `08974a77077ffb342ce2cf9aa0593c12a09b65a10200ceaa844c2034a922fca9`.
  - `configs/rl/sac.yaml`: `d07803ca1c9ae6f9bb1ec3c7812183f2e075ba29eeb4cae7b80425ae34e385da`.
  - `configs/env/reward.yaml`: `d6bdb515f37b7253d60ba8eaf2cebfb5ccf8d59411006d58b7978a387e2481ec`, which fixes the reward's structure; only its
    weights are searched.
  - The evaluation draw is `configs/control/tuning.yaml` (`c7d099d2f8f85b34903d33860240f687481d5a1568d9f57dc85195168fcb43d7`), the P3-D3 tune
    draw: 60 episodes per SS, SS3–SS5, aft.
- **Search spaces**, as in the two YAML files:
  - PPO: learning rate, n_steps, batch, epochs, entropy coefficient, width, `log_std_init`
    [−3.0, −1.6], plus the reward weights;
  - SAC: learning rate, batch, tau, learning_starts, width, plus the same reward-weight ranges.
- **Score.**
  - Each trial is scored on its **final checkpoint**, on the tune draw: mean success over SS3–SS5.
  - Ties go to the lower pooled p95 touchdown closing speed, then the lower trial index.
  - A failed trial is listed and cannot win.
- **Nothing in tuning touches `results/episodes/`.** Trials train on the P3-D2 train pool with the
  curriculum, exactly as a final run would.
- **Commands.** Both use `MAX_WORKERS = 34` slots, counted globally; a slot costs 8 per PPO run and
  5 per SAC run (P5-D6).
  ```
  make tune CFG=configs/rl/tune_ppo.yaml
  make tune CFG=configs/rl/tune_sac.yaml
  ```
- **Outputs.** `results/tune/{ppo,sac}/trials.csv` and `selection.json`. The winners are written
  to `configs/rl/{ppo,sac}.yaml` in a dated entry before any final 5-seed run starts.

### P5-D10 — VM terminated mid-tuning; reconcile and resume added; SAC search re-run at 4 torch threads (2026-09-28)

*Incident.*
- The WSL VM was terminated at about 12:15 EDT on 2026-09-24. `last -x` shows no clean shutdown
  between the 2026-09-23 18:12 boot and the 2026-09-24 16:24 boot.
- Every live training process and both P5-D9 tuning schedulers died. The runs' logs end in
  normal training output, with no error.
- The cause on the Windows side (sleep, update restart, `wsl --shutdown`) is not known.
- *State at termination.*
  - **PPO:** trials 00–15 done; 16–17 never started. They were waiting for slots held by SAC,
    because the two schedulers were not FIFO across each other.
  - **SAC:** trials 00, 02, 03 and 06 done; 01, 04, 05, 07, 08 and 09 dead part-way
    (127 k–484 k of 500 k steps); 10–19 never started.

*Recovery code* (commit `f272cc5`; `make test` 438 passed, 1 skipped by design; `make lint`
clean).
- **Reconcile** (`scripts/reconcile_runs.py`). A run whose status says `running`, whose writing
  process is gone, and whose status is older than 3 × max(eval interval ÷ fps, heartbeat) is
  marked `failed`, with the reason and `reconciled_at`. Nothing else in the record changes.
  - Applied 2026-09-28T11:46:12Z: the six dead SAC trials became `failed` ("host terminated; pid
    dead since 2026-09-24T16:14:45–16:15:27Z"), and both sweeps became `interrupted`.
  - No run directory was deleted or moved. The pre-change JSONs are kept in scratch.
- **Checkpoint resume.** Periodic checkpoints now hold the resume state as well:
  - curriculum stage and window, counters, per-worker sampler RNGs, learner RNGs;
  - for SAC, the replay buffer.

  A `failed` run can be resumed in place, with `make train-bg … RESUME=<run_dir>` or a scheduler
  job, and the resume is recorded in `status.json` and `provenance.json`.
  - *Not bit-exact:* in-flight episodes are dropped, VecNormalize gets one extra update, PPO
    loses its partial rollout, and SAC loses the checkpoint step's own transitions. This is
    documented in `rld.rl.resume`.
  - Replaced CSV rows are first copied to `*.before_resume<k>.csv`.
  - Identical tiny PPO and SAC runs before and after this change give identical weights, evals
    and monitor rows, so the new checkpoint code does not change training.
- **Search resume.**
  - Re-invoking `make tune` skips `done` trials, starts never-run ones, and re-runs `failed` ones:
    fresh into `<seed>_r<k>` by default, or resumed if a search sets `resume_failed: true`. It
    refuses while any trial is live or unreconciled.
  - `--collect` scores each trial from its latest `done` attempt and lists superseded runs.
- **Scheduling.** All schedulers share one global FIFO by enqueue time, so the 2026-09-24
  starvation cannot recur. The slot cost is now `ceil(0.4 × busiest pool) + torch_threads`.
- **Bug fixed.** `collect_trials` would have crashed when scoring PPO trial 0, whose base config
  holds no reward weights (`float(None)`). Automatic scoring would have ended as
  `done_with_errors` with no `trials.csv`. Trial 0's row now reports the committed `reward.yaml`
  weights it actually trains with.

*SAC throughput* (`scripts/p5_sac_throughput_bench.py`; scratch run dirs, train pool only, no
score read; 15 k timed steps after a 2 k warm-up, one measurement per cell; steps/s run alone):

| width | CPU, 1 / 2 / 4 threads | CUDA, 1 / 2 / 4 threads |
|---|---|---|
| 256 | 80.0 / 96.1 / 103.3 | 71.8 / 71.7 / 75.0 |
| 512 | 32.3 / 47.7 / 64.4 | 67.0 / 75.6 / 72.5 |

- **Concurrent**, steps/s per run:
  - 4 × CPU at 4 threads: 41.6 at width 512, 73.6 at width 256;
  - 6 × CPU at 1 thread, width 512: 26.0;
  - CUDA totals are capped at about 190–200 steps/s summed over all concurrent SAC runs,
    whatever the width.
- **Why tuning ran slower.** The 14–58 steps/s seen during tuning is most plausibly contention
  with 4 concurrent PPO runs on 18 physical cores (36 logical). That mix was not reproduced.
- **P5-D9's CPU = GPU reading** was measured at width 256 and 1 thread only. At width 512 or
  with more threads it does not hold.

*Decision (user, 2026-09-28: follow the recommendations).*
- **Every SAC run, tuning and final, uses `device: cpu`, `torch_threads: 4`** (slot cost 8).
  - It is the fastest per run on CPU: 1.3× at width 256 and 2× at width 512.
  - The device is unchanged from P5-D9; only the thread count changes.
  - `configs/rl/sac.yaml` is now `c8eda091453c9ab020e9c3f24348061ffe9732d167d5c6e46158d5b28f08ae85`
    (superseding P5-D9's hash).
- **The SAC search is re-run in full as `configs/rl/tune_sac_v2.yaml`**
  (`4c2b6d7f8133e5386bde1274eab2711720ffce0f552dde7503bf89e7f41036d8`).
  - It has the same 20 Sobol points, checked by comparing the `--dry-run` output byte for byte
    with `tune_sac.yaml`'s.
  - It has the same budgets, seeds and scoring rule. Only `run_group` (`tune_sac_v2`) and
    `results_dir` (`results/tune/sac_v2`) differ, plus the base config's thread count.
  - Every SAC trial therefore runs under one runtime setting.
  - The `tune_sac` runs, 4 done and 6 failed, are kept on disk, **never scored**, and superseded
    by `tune_sac_v2`.
  - Failed trials re-run fresh (`resume_failed` defaults to false), so every trial is one
    uninterrupted run.
- **The PPO search completes in place.** `tune_ppo.yaml` is unchanged: trials 00–15 are skipped
  as done, and 16–17 run fresh. The code change was verified not to alter PPO training.
- **Launch order**, PPO first so the FIFO serves its two trials first:
  ```
  make tune CFG=configs/rl/tune_ppo.yaml
  make tune CFG=configs/rl/tune_sac_v2.yaml
  ```
- **Recommended to the user:** make sure the host does not sleep or restart during runs.
  Checkpoint resume limits a future loss to one checkpoint interval (PPO 1 M, SAC 250 k env
  steps). It does not prevent the loss.

### P5-D11 — PPO search scored; trial 14 becomes the PPO final config; PPO sweep queued (2026-09-28)

*Scoring.* The scheduler's `on_complete` hook ran `collect_trials` automatically when trial 17
finished. It wrote `results/tune/ppo/trials.csv` and `selection.json` under the P5-D9 rule: mean
tune-pool success over SS3–SS5 on the final checkpoint, then lower pooled p95 closing speed, then
lower trial index.
- Hashes (SHA-256):
  - `results/tune/ppo/trials.csv`: `2ead58510a7a8699858dbcf14068d6cb8d12501c8c94841c801e202fab38edf9`
  - `results/tune/ppo/selection.json`:
    `93a4d045080f897453cb2ea8caee167cd90ac01f89ff71e41f7ca56b91357188`
  - the resulting `configs/rl/ppo.yaml`:
    `26b6c49f8552f5fa626c19bab5222205eb83d294854734a23c3b0b1b3d70e14a`
- All 18 trials are `done`; none was superseded or resumed.
- **14 of 18 trials scored 180/180** (mean success 1.000), so the tie-break decided.
- Trials 6 and 4 scored 0.994 and trial 8 0.978.
- **Trial 15 hovers:** success 0.011, SS5 mean time to touchdown 8.35 s, bounce 0.333 of its
  few touchdowns. This is the P5-D9 hover trap, inside the search space.

*Winner: trial 14.*
- Mean success 1.000 and pooled p95 closing speed **0.233 m/s**. The next lowest among the 1.000
  trials were 0.315, 0.318 and 0.318 (trials 3, 5 and 9).
- At SS5: p95 0.241 m/s, mean time to touchdown 2.17 s, bounce 0.000.
- Hyperparameters:
  - learning rate 5.941708698045433e-4 (linear decay), n_steps 1024, batch 1024, 5 epochs;
  - entropy coefficient 0.009895217847079038;
  - width 512 (depth 2, tanh);
  - `log_std_init` −2.7972111202776433 (std 0.061).
- Reward weights:
  - w_progress 13.580398401245475, w_vz 5.847430455465423, w_smooth 0.04575471046550012,
    w_time 0.0007295555155724287;
  - r_success 29.54699269030243;
  - r_hard_landing = r_off_pad = r_bounce 6.8282349687069654;
  - r_crash 42.44193982332945.
- For reference only, P3-D4 table, same tune draw, SS5 p95: `pid_feedforward` 0.252 m/s,
  `pid_feedforward_lowvz` 0.169 m/s. Tune-pool numbers are not results.

*Final config.*
- `configs/rl/ppo.yaml` now carries trial 14's values, copied at full precision from
  `results/tune/ppo/configs/trial_14.yaml`.
- The two parsed configs are equal in every field except `method`, `run_group` and
  `total_steps` (10 M vs 2 M). `tests/test_rl_config.py::test_ppo_config_is_the_tuning_winner`
  pins this.
- **Carried difference.** Because the learning-rate schedule is linear to the budget, the final
  runs decay it over 10 M steps instead of 2 M. This is inherent in tuning at a smaller budget
  (P3-D1 §5) and is stated, not changed.
- **Every Phase 6 PPO-family method inherits these hyperparameters and reward weights**, as
  P3-D1 §5 requires.

*Launch.*
- `make sweep CFG=configs/rl/ppo.yaml SEEDS="0 1 2 3 4"`: 10 M steps per seed, slot cost 8.
- It runs behind the 16 SAC trials already queued (global FIFO, P5-D10), so it starts as the SAC
  search drains.
- P3-D1 §5's budget rule applies: the first PPO seed's wall clock is reported, and any budget cut
  is made before any evaluation result is read.

*Test fix, owned by the main thread.* `5376f2f` changed `sac.yaml` to 4 threads without
re-running the tests. `test_slot_cost_counts_torch_threads` then failed, because it assumed the
committed file had 1 thread. The test now sets both thread counts explicitly and checks the
committed value. `test_collect_scores_latest_done_and_lists_superseded` now expects trial 0's
effective weights: the committed `reward.yaml` overlaid with the base config's overrides.

### P5-D12 — PPO budget rule: no cut; SAC search scored; trial 10 becomes the SAC final config (2026-09-29)

*PPO budget rule (P3-D1 §5), applied before any evaluation result is read.*
- The first final PPO seed, `artifacts/runs/ppo/0` at `1d67488`, took **3.17 h** wall clock for
  10 010 624 steps (15.3 CPU-hours). Seeds 1–3 took 3.22, 3.13 and 3.07 h. Seed 4 was at 9.24 M
  steps and 2.42 h when this entry was written.
- **Projection for the PPO family** (25 runs × 10 M, P3-D1 §5): about 25 × 3.2 h = 80 h as a
  single stream, or about 20 h at the four concurrent runs the slot cap allows. That is far
  below the 5-day threshold. **No budget cut is made.**
- **Provenance note.** `git_dirty` is true on all five runs. For seeds 0–2 the dirty paths are
  only the user's two `.claude/skills` files. Seeds 3–4 also list the untracked
  `results/tune/sac_v2/{selection.json,trials.csv}`: SAC scoring wrote those files while those
  runs were queued. They are outputs, not code or config, and do not affect training.

*SAC scoring.* The scheduler's `on_complete` hook scored all 20 `tune_sac_v2` trials under the
P5-D9 rule. None was superseded or resumed.
- Hashes (SHA-256):
  - `results/tune/sac_v2/trials.csv`: `884b285511a4ec8217cda748c812b64449380149c34b500aa6f72f70c2faf57f`
  - `results/tune/sac_v2/selection.json`:
    `90d1e19faae45340a26b7f956dba7adadbedd36e0a52fbf4fd65462406d500b4`
- **Winner: trial 10.**
  - Mean tune-pool success **0.978**, pooled p95 closing speed 0.472 m/s.
  - At SS5: success 0.933, p95 0.503 m/s, mean time to touchdown 1.28 s, timeout 0.
  - Runners-up: trial 14 and trial 7 (0.950), trial 6 (0.944), trial 17 (0.939).
- **No SAC trial scored 180/180.** Six scored 0.000:
  - trials 2, 5, 12, 19 and 4 are **timeout-dominated** (0.83–0.97 at SS5); trial 4 has a
    mean time to touchdown of 8.1 s. That is the same hover trap as PPO trial 15.
  - trial 13 touched down (SS5 mean time to touchdown 4.5 s, timeout 0.05) but had no success.
- **Single-seed trial scores are noisy** (added at the Gate 5 review, 2026-09-30). The superseded
  `tune_sac` v1 run of trial 03 (same Sobol point, width 256, 1 torch thread) ended at 0.000 tune
  success after hovering (`artifacts/runs/tune_sac/trial_03/0/evals.csv`). Its v2 re-run scored
  0.872. The v1 runs of trials 00, 02 and 06 reproduced their v2 scores. Thread count alone
  changes the arithmetic, so the two runs are not bit-comparable, but the swing shows how much
  one seed can move a trial's score. v1 was never used for selection.
- **SAC lands harder than PPO on the tune draw.** The SAC winner's pooled p95 is 0.472 m/s,
  against 0.233 for the PPO winner. Tune-pool numbers are not results, and SAC's 2 M budget
  against PPO's 10 M is a confound (P3-D1 §5).
- Winner hyperparameters:
  - learning rate 2.8376762147927807e-4, batch 256, tau 0.01665660378704859;
  - learning_starts 10 000, width 512 (depth 2, ReLU);
  - `device: cpu`, `torch_threads: 4` (P5-D10).
- Winner reward weights:
  - w_progress 19.067129995673895, w_vz 2.5956556352380216, w_smooth 0.04927539336297006,
    w_time 0.010835396256297826;
  - r_success 93.97084654774517;
  - r_hard_landing = r_off_pad = r_bounce 16.795000918209553;
  - r_crash 13.287172988057137.

*Final config.*
- `configs/rl/sac.yaml` (`a46f94152e99cd325ec30eb7de6c25c79be05c2294519eddee235e09b8bf5159`)
  carries trial 10's values at full precision.
- Its parsed config equals `results/tune/sac_v2/configs/trial_10.yaml` except `method`,
  `run_group` and `total_steps` (2 M vs 0.5 M). This is pinned by
  `test_final_config_is_the_tuning_winner`, parametrised over PPO and SAC.
- **Expected wall clock:** at the 40–70 steps/s seen for width 512 under contention, each 2 M-step
  seed takes roughly 8–14 h. The five seeds share the 34 slots at a cost of 8 each.

*Launch.* `make sweep CFG=configs/rl/sac.yaml SEEDS="0 1 2 3 4"`.

### P5-D13 — PPO and SAC final checkpoints on the frozen id list (2026-09-30)

*What was flown.* The `final/` checkpoint of each of the ten finished Phase 5 runs, on
`results/episodes/id.parquet` (SS3–SS6, N = 200 per cell, aft pad): 10 runs × 800 = **8 000
episodes**, none skipped.
- *Runs.* `artifacts/runs/ppo/{0..4}` (`ppo`, 10 M steps, trained at `1d67488`; checkpoint at
  10 010 624 steps) and `artifacts/runs/sac/{0..4}` (`sac`, 2 M steps, trained at `14d7f32`;
  checkpoint at 2 000 000 steps). All ten are `done`, and `status.json` `resumed_from` is null
  for every one.
- *Policy path.* `rld.eval.runner.callable_spec(method, functools.partial(rld.rl.train.build_policy,
  run_dir=<run>, ckpt="final"), run_seed=<training seed>)`: the batch-1 `LearnedPolicy`,
  deterministic action, `VecNormalize` frozen (`training=False`, `norm_reward=False`), no
  ship-motion feed, not privileged. The seed label is the run's training seed, read from
  `provenance.json` and checked against `status.json`.
- *Runner.* The same chunked, parallel `run_matrix` path as e01 (chunk 25, 24 workers,
  `OMP_NUM_THREADS = MKL_NUM_THREADS = 1` in every worker); 150.0 s wall.
- *Before the run.* `scripts/make_episodes.py --check`: 10/10 hashes OK, MANIFEST SHA-256
  `e6f30e55…`. The `id` list is file `919be99d…`, content `ea38d8a5…`, as in P3-D1 §2.
- *Command.* `python -m rld.eval.learned --learned ppo=artifacts/runs/ppo/0,…,4
  sac=artifacts/runs/sac/0,…,4 --ckpt final --list id --out-dir results/e05 --workers 24`.
  The entry point is a module because `scripts/` is outside the eval-auditor's write scope
  (as for `rld.eval.paired_outcomes` in P5-D3); a thin `scripts/eval_learned.py` may call
  `rld.eval.learned.main`.

*Provenance.*
- *Code state.* HEAD `00bff09`, `git_dirty = true`. The dirty paths are the user's two
  `.claude/skills/*.md` files and this entry's eval-side files: the new
  `src/rld/eval/learned.py` and `tests/test_eval_learned.py`, and `src/rld/eval/report.py`
  (a public `CAVEATS` alias only). Nothing under `src/rld/` or `configs/` changed between
  `14d7f32` and HEAD. Since `1d67488` the only such change is `configs/rl/sac.yaml` (P5-D12),
  which the PPO runs do not read.
- *Run digests* (all in `run_info.json`, and per row in `summary.csv`). Every PPO run's
  `config.yaml` is `aa28e4f4…` and its `config_source.yaml` is `26b6c49f…` (=
  `configs/rl/ppo.yaml`, P5-D11). Every SAC run's are `04af3f42…` and `a46f9415…` (=
  `configs/rl/sac.yaml`, P5-D12). `final/model.zip` and `final/vecnormalize.pkl` were hashed
  before and after the flight, and did not change:

  | run | `model.zip` | `vecnormalize.pkl` |
  |---|---|---|
  | ppo 0 / 1 / 2 / 3 / 4 | `1a012394…` / `77e172f4…` / `785d0789…` / `ab715248…` / `0f6fa22d…` | `9ddb2a99…` / `a82172a5…` / `9a95795a…` / `97bb0f4a…` / `6e5d7599…` |
  | sac 0 / 1 / 2 / 3 / 4 | `3401491e…` / `8f724505…` / `9688985a…` / `c5c2be8f…` / `45a54640…` | `145883fa…` / `eb9ea77c…` / `aa95499e…` / `6462c1c4…` / `5d0066db…` |

- *Baselines carried, not re-flown.* Their `summary.csv` lines for the four `id` aft cells are
  copied line for line into `carried_summary_e01.csv` (20 lines: `pid_track_descend`,
  `pid_feedforward`, `pid_feedforward_lowvz`, `gated`, `oracle_gated`; source summary
  `2e5a3400…`, episodes `c5852090…`) and `carried_summary_e01_lowvz_cut.csv` (4 lines:
  `pid_feedforward_lowvz_cut`; source summary `180eeef9…`, episodes `8fe15278…`). Checked and
  recorded in `run_info.json["carried"]`:
  - every line is byte-identical to a source line (also re-checked with `grep -Fx`: 21/21 and
    5/5 including headers);
  - each source's run-wide provenance columns equal the live ones;
  - each source flew exactly the 800 listed `id` episodes (regime, SS, index, realization,
    episode seed, `t0`, initial position);
  - re-summarising each source's `episodes.csv` reproduces its summary's metric columns as
    text.
- *Re-derivation.* `python -m rld.eval.learned --check` recomputes `summary.csv`, `seeds.csv`,
  `aggregate.csv` and the markdown from `episodes.csv`: all four **byte-identical**. The markdown
  was re-rendered once after the run (renderer text only: the penetration column shown as a
  positive depth, and the note that a seed-bootstrap CI excludes episode sampling). No CSV
  changed; `run_info.json["re_rendered"]` records both hashes.

**Headline: `id`, aft pad, success per sea state.** Learned rows: rliable IQM across the 5
seeds (%), stratified-bootstrap 95 % CI (P3-D1 §4: 2 000 replicates, seed 20260926, seeds
resampled within the cell, one cell per task), then the per-seed range. Baselines: one
deterministic run, % [Wilson 95 % CI] k/200. Success is not pooled across sea states. SS6 is
outside every method's training distribution (`in_training_distribution` 0.000; 0.745–0.755 at
SS3–SS5, where the 90° headings are outside the dev pool).

| method | SS3 | SS4 | SS5 | SS6 (out of distribution) |
|---|---|---|---|---|
| `ppo` IQM [CI]; seed range | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 98.2 [98.0, 98.5]; 98.0–98.5 |
| `sac` IQM [CI]; seed range | 99.8 [98.8, 100.0]; 98.5–100.0 | 98.2 [96.3, 99.0]; 96.0–99.0 | 87.0 [81.7, 90.3]; 80.0–91.0 | 72.5 [60.2, 79.3]; 55.5–81.5 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200 | 96.0 [92.3, 98.0] 192 | 85.0 [79.4, 89.3] 170 | 80.0 [73.9, 85.0] 160 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | 99.0 [96.4, 99.7] 198 | 90.5 [85.6, 93.8] 181 |
| `pid_feedforward_lowvz` | 100.0 [98.1, 100.0] 200 | 98.0 [95.0, 99.2] 196 | 95.5 [91.7, 97.6] 191 | 85.0 [79.4, 89.3] 170 |
| `pid_feedforward_lowvz_cut` | 100.0 [98.1, 100.0] 200 | 98.5 [95.7, 99.5] 197 | 98.0 [95.0, 99.2] 196 | 88.0 [82.8, 91.8] 176 |
| `gated` | 100.0 [98.1, 100.0] 200 | 98.5 [95.7, 99.5] 197 | 89.5 [84.5, 93.0] 179 | 63.0 [56.1, 69.4] 126 |
| `oracle_gated` (privileged; commit-timing oracle) | 100.0 [98.1, 100.0] 200 | 98.5 [95.7, 99.5] 197 | 89.0 [83.9, 92.6] 178 | 64.5 [57.7, 70.8] 129 |

- *Per-seed Wilson CIs.* `ppo`: 200/200 [98.1, 100.0] for every seed at SS3–SS5; at SS6,
  197, 196, 196, 197, 196 (seeds 0–4), e.g. 98.0 [95.0, 99.2]. `sac`: SS5 178, 160, 182, 174,
  170, e.g. seed 1 80.0 [73.9, 85.0]; SS6 163, 111, 150, 146, 139, e.g. seed 1 55.5
  [48.6, 62.2]. Every cell is in `summary.csv` and the rendered table.
- *Optimality gap* (points, [CI]): `ppo` 0.0 at SS3–SS5, 1.8 [1.6, 2.0] at SS6; `sac` 0.4
  [0.0, 1.0], 2.1 [1.1, 3.2], 13.6 [10.6, 17.0], 29.1 [22.4, 37.5].
- *What the seed CI is.* It reflects seed-to-seed variation only. Where every `ppo` seed scores
  200/200 it collapses to [100.0, 100.0]; the episode-level uncertainty is then each seed's
  Wilson CI, [98.1, 100.0].
- **Budget confound.** `ppo` had 10 M env steps per seed and `sac` 2 M (P3-D1 §5). Every
  PPO-vs-SAC reading carries it. No method contrast is tested here, and none of the baseline
  comparisons above is a paired test: they are unpaired readings of the tables.

**Closing speed (p95 along the deck normal, m/s model scale; IQM of the per-seed p95s
[stratified-bootstrap CI]).** Descriptive; it is not P3-D1 §4's pooled relative-p95 statistic.

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` | 0.265 [0.263, 0.266] | 0.267 [0.264, 0.270] | 0.271 [0.267, 0.275] | 0.278 [0.274, 0.285] |
| `sac` | 0.392 [0.369, 0.433] | 0.437 [0.406, 0.478] | 0.595 [0.562, 0.619] | 0.668 [0.590, 0.778] |
| `pid_feedforward` | 0.219 | 0.226 | 0.262 | 0.264 |
| `pid_feedforward_lowvz` (H1a reference) | 0.127 | 0.139 | 0.188 | 0.181 |
| `pid_feedforward_lowvz_cut` | 0.127 | 0.139 | 0.188 | 0.181 |
| `pid_track_descend` | 0.308 | 0.412 | 0.525 | 0.502 |
| `gated` | 0.217 | 0.229 | 0.250 | 0.267 |
| `oracle_gated` (privileged) | 0.220 | 0.236 | 0.252 | 0.254 |

*Caveats added at the Gate 5 review (2026-09-30), from P5-D14.* The recorded closing speed
understates impact speed by about 7 %. Up to 41 SAC successes (1 / 3 / 13 / 24 at SS3–SS6, per
1 000 seed-episodes), 2 PPO successes (SS6) and 1 `pid_feedforward_lowvz_cut` success (SS6)
may depend on tunnelling overlap. Both can only raise SAC's SS5/SS6
success, by at most a few points. The same caveat is now a line in `success_vs_seastate.md`'s
header.

**Seed-to-seed spread (`seeds.csv`; sample SD, ddof = 1).**
- `ppo`: success SD 0.0 points at SS3–SS5 and 0.3 at SS6; p95 SD 0.001–0.005 m/s.
- `sac`: success SD 0.7, 1.3, 4.2 and 9.7 points at SS3–SS6; p95 SD 0.028–0.091 m/s.

**Outcome breakdown where the learned methods lose** (counts over 5 seeds × 200 per cell; every
cell's six fractions sum to 1; full per-seed tables in `success_vs_seastate.md`).
- `ppo` loses only at SS6: 15 `hard_landing` and 3 `bounce` in 1 000 seed-episodes (per seed
  3/4/3/2/3 hard, 0/0/1/1/1 bounce). No `crash`, `off_pad` or `timeout` in any cell.
- `sac`, summed over seeds:

  | SS | crash | off_pad | hard_landing | bounce | timeout |
  |---|---|---|---|---|---|
  | SS3 | 0 | 0 | 4 | 0 | 0 |
  | SS4 | 0 | 1 | 20 | 0 | 0 |
  | SS5 | 2 | 8 | 122 | 3 | 1 |
  | SS6 | 18 | 39 | 232 | 2 | 0 |

  The 20 crashes are `tilt_gt_crash` 16, `off_plate_strike` 3 and `below_deck` 1; 17 of them
  are seeds 2 and 4 at SS6. No baseline has a `crash` or an `off_pad` in any `id` cell
  (P3-D1 §7).

**Surprises and flags (reported, not acted on; the Step 2 audit owns the follow-up).**
1. **`ppo` at `id` SS6 is above every baseline's single run (unpaired).** It scores 98.0–98.5 %
   on a sea state it never trained on, against 90.5 % [85.6, 93.8] for `pid_feedforward`. This
   is not a tested contrast, and it is not H1b, which is about `residual_ppo`.
2. **`ppo` lands fast at a nearly fixed closing speed.** Its median time to touchdown is
   1.51–1.59 s at every SS, against 4.4–4.6 s for `pid_feedforward`, 3.2–4.7 s for `gated` and
   7.6–7.8 s for `lowvz`. Its p50 closing speed is 0.241–0.254 m/s in every cell, and its p95 is
   above `pid_feedforward`'s and `lowvz`'s in every cell.
3. **`sac` tunnels.** 264 of 4 000 episodes exceed the 5 mm threshold (3, 11, 72 and 178 at
   SS3–SS6), with a maximum depth of 15.67 mm. Compare P5-D3's `lowvz_cut`: 31 over all 14
   cells, max 7.03 mm. `ppo` tunnels 6 times (max 5.35 mm).
4. **`sac` seed 1 at SS6** is 55.5 %, 14 points below the next-lowest seed (69.5 %). Its
   losses are 80 `hard_landing`, 8 `off_pad` and 1 `crash`.
5. **Timeouts are almost absent**: 1 of 8 000 (`sac` seed 2, SS5). Neither final policy shows
   the P5-D9 hover trap on this list. `gated` times out on 35 % of `id` SS6.
6. **Detector disagreement** is 2 / 8 000 (`sac` SS6, seeds 0 and 4); `ppo` has 0 / 4 000.
7. **N and seeds.** Every one of the 40 learned cells has N = 200 and all 5 seeds; there is
   no missing seed and no short cell.

*Artifacts, all in `results/e05/` (SHA-256).*

| file | SHA-256 |
|---|---|
| `episodes.csv` (8 000 rows) | `92419dfc49d854dae84836d1f3721fde4cd254363772a13e6ba980f606fb6169` |
| `summary.csv` (40 rows) | `a2a0f50f77012a4488c6c28939a170725ed95453d2a9589f10612a7a1d6192f2` |
| `seeds.csv` (8 rows) | `cf61ce010b08535fabe96996cd5e8259ac04578cbf4bc95c8fcfe6a7883c1613` |
| `aggregate.csv` (24 rows) | `ded3cdd53290d025f60c81a029d8545dcaf6d99babe204dd03d842acfa2874b9` |
| `carried_summary_e01.csv` | `27da5b6f8db170cbcb0c794c039113018d82242b2beb9470a8149f8823b2d277` |
| `carried_summary_e01_lowvz_cut.csv` | `39101ccbf2df946ac838bb635f3e295d58d1008ef175987d4301f25cdcda9594` |
| `success_vs_seastate.md` | `d8a30a0f62d485806fbb2a0f72470ef72a984b80ef03f886cf4df6c8e957cb3b` (Gate 5 caveat line added and corrected at the re-review; as first committed: `6ae9bc3100539c0b3810071b9c731e9148074fa2b44c1d8893eb4c52aa9863f7`) |

`run_info.json` holds the non-deterministic facts (host, timings, git state) and is not hashed
here. No hypothesis is scored (Phase 7). P3-D1 is unchanged.

*Code (eval side only).* `src/rld/eval/learned.py` (new): run inspection and seed labelling,
the learned spec, the flight through `run_matrix`, per-seed summary, `seeds.csv`,
`aggregate.csv`, the baseline carry and its checks, the renderer, and `--render-only` and
`--check`. `tests/test_eval_learned.py` (new): two tiny SAC runs at training seeds 0 and 3 flown
end to end through `main` on a temporary dev-pool list, never a committed one. It checks seed
labels, worker independence, the carry checks, the privileged label and byte-identical
re-rendering, plus known-value tests of the spread and IQM.

### P5-D14 — Reward-hacking audit of the ten final Phase 5 runs (2026-09-30)

*Scope.* The audit covered:
- every training episode in the monitors (PPO 681 073, SAC 161 527);
- all 8 000 e05 rows;
- the e01 and e01_lowvz_cut `id` rows as baselines.

1 354 scratch re-flights (a 25-per-SS sample, plus every e05 tunnelled or disagreement episode)
added per-step logs through the committed env and `build_policy`. **Each re-flight reproduced its
committed row exactly, 1 354 of 1 354.** Nothing was re-trained and nothing in `results/e05` or
`results/e01*` changed.
- *Command.* `python scripts/reward_hacking_audit.py --out-dir results/audit --scratch-dir <dir>
  --workers 24`. The main thread re-ran it into scratch, and all 16 CSVs were byte-identical.
- *Hashes.*
  - `verdicts.csv`: `36ec2fdcd9ce7c9553c74b157e09dd6b4d94b883f8323c11000008c9a55b832a`
  - `README.md`: `cd04d8f387efe05b8039a896bdf7a581813a9032f99a4133b0497416ac5fa8fc`
    (after the Gate 5 corrections and the re-review fixes; as first committed in `d3af59c`:
    `e30e413db8845480d1a40888b4cc6395a94c03e2760b7c2aa91bf38622e2c2ed`)
  - `ppo.csv`: `47030581299b8f80898f734307a26372ded42bb9e7e47c72d6ec733a8ac49cbe`
  - `sac.csv`: `190dfb6e4769391d7b725bba86977cf118d4f9570d8e8756c0ea093ce1df5330`
- *Thresholds.* They were pre-stated in `results/audit/README.md`, whose section says it was
  written at 2026-09-30 08:49 EDT before any audit number. That timestamp is the agent's own
  record: the section was not committed separately before computation. No threshold was moved,
  and post-hoc readings are labelled as such beside the pre-stated verdicts.

*Verdicts*, PPO / SAC (`verdicts.csv`):

| check | PPO | SAC |
|---|---|---|
| 1a training timeouts, final 10 % of steps | clean (0.000) | clean (0.000) |
| 1b training timeouts, 10 bins | clean (max 0.040) | clean (max 0.269) |
| 1c e05 timeouts vs `gated` | clean (0 / 4 000) | clean (1 / 4 000) |
| 2a tunnelling (limit: `lowvz_cut` 0.75 %, 7.03 mm) | clean (0.15 %, 5.35 mm) | **finding** (6.6 %, 15.67 mm) |
| 2b tunnelling mechanism | finding: post-contact idle, 6 of 6 | inconclusive: impact 103, idle 119, other 42 of 264 |
| 2c outcome depends on tunnelling (pre-stated flag) | finding (2 of 6) | finding (44 of 264) |
| 3 detector disagreement | clean (0) | clean (0.2 % at SS6) |
| 4 success concentrated in easy start states | clean | clean |
| 5 pre-contact norm-cap saturation | clean (0.8 %) | **finding** (12.2 %; 86 % of it in the first 0.5 s) |
| 6 post-contact down-force (H1a confound) | clean by 0.008 (idle fraction 0.492), **3 of 5 seeds idle** | **finding** (0.871, all seeds) |
| 7 passive, deck-driven landings | clean (3.6 %) | clean (0 %) |
| 8 seed outliers | minor metrics only | SAC seed 1 (SS6 55.5 %) is not singled out by any check |

*Readings.*
- **Hover trap.** Neither final policy hovers.
  - Post hoc, at 1 % bins, four of five SAC seeds hovered between 20 k and 160 k steps (timeout
    0.93–1.00) and recovered within 40–60 k steps.
  - PPO's early timeouts (≤ 100 k steps) are most likely the untrained initial policy; this is
    not proven.
  - The pre-stated 10-bin check was too coarse to see either.
- **SAC tunnelling is driven mainly by impact speed, with the post-contact throttle cut adding to
  it.**
  - The tunnelling rate rises with closing speed: 1.5 % at 0.2–0.3 m/s, 24.6 % at 0.5–0.6, 61 %
    at 0.6–0.7, 92 % at 0.8–0.9.
  - Maximum depth comes a median 38 ms after first contact, and the motors are at idle then in
    59 % of cases.
- **Whether any success depends on penetration is not shown** (corrected at the Gate 5 review,
  2026-09-30; the audit first wrote "no success depends on penetration").
  - The audit's post-hoc argument was that no flagged success has an unloaded stretch longer
    than the 50 ms grace. That does not settle it: without the overlap the drone would separate
    during that stretch and have to close the gap again, so the counterfactual gap is at least
    as long and plausibly longer. 18 of the 41 SAC stretches are ≥ 25 ms, and one is exactly
    50 ms.
  - **Bound:** up to 41 SAC successes (1 / 3 / 13 / 24 at SS3–SS6, per 1 000 seed-episodes)
    and 2 PPO successes (SS6) may depend on the overlap; so may 1 `pid_feedforward_lowvz_cut`
    success (SS6), under the same frozen definition.
  - With the measurement-instant fact below, both biases can only raise SAC's SS5/SS6 success,
    by at most a few points. SAC is already below `pid_feedforward` there, so no reading there
    changes direction.
  - Of the non-success flagged episodes, 1 of SAC's is plausibly outcome-changing, and it is a
    bounce, which counts against the policy.
- **P5-D3's idle-thrust explanation for `lowvz_cut` tunnelling is now verified.** PPO's 6 and
  lowvz_cut's 6 `id` episodes are all of that kind.
- **PPO's fixed ~1.5 s touchdown is an active descent.** It descends at about −1.2 m/s, then holds
  −0.26 m/s relative to the deck for the last ~0.9 s. Closing speed does not depend on the deck's
  vertical velocity (R² ≤ 0.021), and it lands on a rising deck 48–54 % of the time, close to
  `pid_feedforward`.
- **H1a confound, carried to Phase 7.** A post-contact throttle cut is present in every SAC seed
  and in three of the five PPO seeds. Both methods bounce 0.2–0.3 % at SS6, against 7.5–10.5 % for
  lowvz and lowvz_cut. `pid_feedforward_lowvz_cut` (P5-D2) is the control for it.

*Two measurement facts, recorded and not changed. Both are frozen definitions applied to every
method alike.*
1. **The recorded touchdown closing speed understates impact speed.** It is read after the first
   contact substep's solver impulse, about 7 % below the value one substep earlier (median ratio
   0.92–0.94 for every method).
   - In the re-flight sample, 15 of 445 SAC successes (3.4 %) arrived above 0.5 m/s one substep
     before contact; PPO had 0 of 500 and `pid_feedforward` 0 of 99.
   - SAC's SS5/SS6 success therefore depends on the measurement instant by a few points.
   - This is P2-D5/P2-D6's frozen definition.
2. **`success.yaml`'s comment says penetration is flagged "at first contact", but the code flags
   it at any contact substep.**
   - The committed tunnelling counts, e01 and e05 alike, follow the code.
   - In 262 of 264 SAC cases the maximum depth comes after first contact.
   - The comment is wrong, not the counts. Both readings are carried; neither is changed here.

*Code.*
- `src/rld/rl/audit.py` (new) and `scripts/reward_hacking_audit.py` (new).
- `tests/test_rl_audit.py` (new): 17 tests, including an end-to-end re-flight that reproduces a
  committed e01 row.
- `tests/test_rl_leakage.py` forbids `rld.rl` from referencing the frozen lists, and it is kept
  unchanged. The script (evaluation-side) checks MANIFEST, reads `id.parquet`, and passes the rows
  into `rld.rl.audit`, which never opens a list.

## Phase 6

### P6-D1 — Phase 6 method definitions, fixed before any Phase 6 training step (2026-09-30)

*Status.* This entry was written and committed before any Phase 6 smoke or final run. P3-D1 is
unchanged; its block SHA-256 is still `21465588…`.

*Inherited, unchanged (P3-D1 §5, P5-D11, P5-D12).*
- All four methods take `configs/rl/ppo.yaml` verbatim: trial 14's hyperparameters and reward
  weights, `log_std_init` −2.80, 10 M env steps, seeds 0–4.
- They use the same P3-D2 train pool, the same curriculum, and the same tune-pool evaluation draw.
- `tests/test_rl_config.py::test_phase6_config_inherits_ppo` pins every final config equal to
  `ppo.yaml` except in `method`, `run_group` and the three component keys below.
- None of the four has a hyperparameter search of its own.

**`residual_ppo`** (`src/rld/rl/residual.py`).
- *Executed action.* `clip(a_pid_ff(o) + 0.3·π(o), −1, 1)`, then the env's own norm cap (P2-D2).
  α = 0.3 in normalised units is 0.45 m/s model scale.
- *The base.* `pid_feedforward` with the committed `configs/control/pid_feedforward.yaml`. It is
  built as the runner builds the baseline, reset every episode, and fed the raw observation.
- *The policy's input.* π reads the unchanged 25-entry observation. The base action is **not**
  appended, as P3-D1 writes "π(o)".
  - `pid_feedforward`'s lateral integrator is state the policy does not see.
- *Initialisation.* Only `action_net`'s weight and bias are zero-initialised. The value head, the
  MLPs and `log_std` are not.
  - The initial deterministic policy is therefore exactly `pid_feedforward`.
  - **Effective exploration std** is 0.3 × 0.061 = **0.018** normalised units. This follows from
    inheriting `ppo.yaml`'s `log_std_init`, and it is stated rather than re-tuned.
- *One composition path.* Training (`ResidualActionWrapper`), the tune-pool evaluation workers,
  and evaluation (`ResidualPolicy`) all call the same `compose_residual`.
- **Gate test.** `tests/test_rl_residual.py::test_gate_zeroed_residual_is_pid_feedforward` flies
  the zero-initialised committed network through `rld.eval.runner.run_list` on 14 frozen `id`
  episodes (2 220 steps).
  - The sample is SS3–SS6 × 3, plus `pid_feedforward`'s first SS6 `bounce` and first SS6
    `hard_landing` from e01.
  - Every episode column except `method` is identical to `pid_feedforward`, and every per-step
    float32 action is bit-identical.
  - The training-wrapper path reproduces the same actions.
  - It needs only committed files, so it cannot skip.

**Forecast observation block** (`ppo_forecast`, `residual_ppo_forecast`;
`src/rld/rl/forecast_obs.py`). This is the Phase 4 block deferred by P4-D4a.
- *Forecaster.* dmf `residual_interval` (DLinear-OLS), the same one as the primary
  `gated_forecast` arm. A test pins that.
- *Block.* `leads_full_s(forecast, (1, 2, 3) s full)`, i.e. 0.2 / 0.4 / 0.6 s model: point pad z
  (m) and pad v_z (m/s), model scale.
  - Layout `[z1, vz1, z2, vz2, z3, vz3]`, clipped to ±10 so the box stays finite. The clip never
    binds.
  - It is appended after the 25 env entries, giving 31. `VecNormalize` normalises it.
- *Feed.* It is computed from a past-only `ShipMotionFeed` advanced to the control time. Training
  builds the feed per episode in the wrapper; evaluation receives it from the runner
  (`needs_motion_feed=True`).
  - Train/eval parity is bit-identical on 3 `id` episodes, including episodes from the prefetch
    standby slot.
  - Perturbing the future leaves the block bit-identical.
- *Threads.* ORT runs with 1 intra-op and 1 inter-op thread in every worker.
- *Carried caveats.*
  1. The forecaster was fitted on the P3-D2 dev pool (P4-D1). Its forecasts are therefore
     **in-sample during training** and out-of-sample on the frozen lists, so the policy may learn
     to trust them more than test-time accuracy warrants.
  2. The feed is an extra ideal ship-motion sensor that `ppo` does not have (P4-D3). So `ppo_forecast`
     vs `ppo` (H3) compares sensor suites as well as information about the future.
- *Cost.* A forecast adds about 1.2 ms per env step (+73 % at p50, measured with one worker). The
  feed's 200-sample pre-fill runs on the reset-prefetch thread.

**`ppo_sinusoid`** (`motion: sinusoid`; `src/rld/rl/motion.py`).
- *Motion.* Every training episode, every periodic tune-pool evaluation episode, and therefore
  curriculum promotion and the learning curves, fly a `SinusoidDeckMotion`. It is built per
  episode and matched to the episode's P3-D2 realization:
  - amplitude √2 × the committed RMS in `results/deck_stats_seeds.csv` (aft rows), which matches
    `matched_rms` to rtol 1e-9;
  - the realization's peak encounter period;
  - phases drawn from the episode seed.
- *No JONSWAP.* No JONSWAP motion enters its training. The frozen-list evaluation on JONSWAP (Gate
  6) and the H4 cross (Phase 7) are evaluation-side.

**Smoke runs are pipeline checks, not tuning trials.**
- One per method: seed 0, 100 k steps, `method` = `run_group` = `<method>_smoke`.
- Nothing read from them may change a hyperparameter. A failure that seems to need such a change
  goes to the user and would be a dated deviation.

**Also recorded.**
- `load_policy` refuses to load if `pid_feedforward.yaml` or the forecaster's `meta.json` has
  changed since training. Their digests are in `provenance.json["components"]`.
- `config_to_dict` omits the new keys at their defaults, so the Phase 5 run configs and their
  hashes are unchanged.
- **Eval side, open.** `rld.eval.learned` and the audit's re-flight do not yet pass
  `needs_motion_feed` for the forecast runs, and they fail loudly until they do. That is the
  eval-auditor's work, before e06.
- **Slot cost, open.** The scheduler formula is unchanged, since there are no extra threads. But a
  forecast worker uses about 70 % more CPU per step. Process-tree CPU is measured on the smoke
  runs before the final forecast runs are queued.

### P6-D2 — Smoke runs: pipeline checks pass; final sweeps queued (2026-09-30)

*What ran.* Four smoke runs, all launched together with `make train-bg CFG=configs/rl/<m>_smoke.yaml
SEED=0`:
- seed 0 each, trained at `8736876` with `git_dirty` false;
- 114 688 steps each (7 rollouts);
- all `done`, none resumed.

Each `final/` holds `model.zip` and `vecnormalize.pkl`, and `load_policy` returns the expected
class:

| run | policy class | obs | feed | wall | CPU-h (avg cores) | tune-pool success, 6 evals (20 k … 114 688) | final stage |
|---|---|---|---|---|---|---|---|
| `residual_ppo_smoke` | `ResidualPolicy` | 25 | no | 315 s | 0.52 (5.9) | 1.000, 0.994, 0.994, 1.000, 1.000, 1.000 | SS5 |
| `ppo_forecast_smoke` | `ForecastPolicy` | 31 | yes | 413 s | 1.13 (9.8) | 0.000, 0.000, 0.000, 0.022, 0.089, 0.000 | SS3 |
| `residual_ppo_forecast_smoke` | `ResidualForecastPolicy` | 31 | yes | 407 s | 0.74 (6.5) | 1.000, 1.000, 0.994, 1.000, 1.000, 0.994 | SS5 |
| `ppo_sinusoid_smoke` | `LearnedPolicy` | 25 | no | 309 s | 0.76 (8.9) | 0.000, 0.000, 0.000, 0.000, 0.000, 0.006 | SS3 |

*Readings. These are pipeline facts, not results; tune-pool numbers are never results.*
- **The residual runs start where they should.** They begin at `pid_feedforward`'s level on the
  tune draw and promote SS3 → SS5 within the smoke budget, consistent with the zero-initialised
  residual.
- **The pure-PPO runs have not learned yet at 115 k steps.**
  - `ppo_forecast`'s final eval is 1.000 `timeout`. `ppo_sinusoid`'s is 0.617 `timeout`, 0.317
    `off_pad`, 0.050 `bounce` and 0.011 `crash`.
  - This matches Phase 5. `ppo` seed 0 first scored 0.700 at its first eval (200 k), and P5-D14
    found early timeouts in PPO training up to about 100 k steps.
  - Nothing was changed because of it. Per P6-D1, smoke runs are not tuning trials.
- **CPU.** The averages are dominated by evaluation (6 × 180 tune-pool episodes in 115 k steps,
  against one eval per 200 k in the final runs) and by 64 env workers sharing 36 cores. They
  overstate steady state: the Phase 5 `ppo` finals averaged 4.8 cores (15.3 CPU-h / 3.17 h).
  - The slot cost stays at 8 for every Phase 6 run.
  - Four forecast runs at once would be the worst case, estimated at about 30 cores on the 36-core
    host. Over-subscription would cost wall clock only; the runs are seeded and CPU-deterministic.
  - Measured wall clock is reported with the results.

*Launch.* The four sweeps are queued in this order on the global FIFO scheduler (34 slots, cost 8,
so 4 runs at once):
- `make sweep CFG=configs/rl/residual_ppo.yaml SEEDS="0 1 2 3 4"`
- then the same for `ppo_forecast`, `residual_ppo_forecast` and `ppo_sinusoid`.

That is 20 runs × 10 M steps, about 20–25 h. The sweep IDs are in `artifacts/runs/_sweeps/`.

### P6-D3 — Phase 6 sweeps complete; learning curves (2026-10-01)

*Runs.* All 20 runs are `done`:
- `artifacts/runs/{residual_ppo,ppo_forecast,residual_ppo_forecast,ppo_sinusoid}/{0..4}`, 10 010 624
  steps each;
- every one trained at `6cedd5d` with `git_dirty` false, and `resumed_from` null for all;
- no run failed, stalled or was dropped.

The sweep ran from 2026-09-30 17:56 to 2026-10-01 15:26 UTC, about 21.5 h, with 4 runs at a time.

| method | wall per seed | fps |
|---|---|---|
| `residual_ppo` | 3.23–3.28 h | 847–861 |
| `ppo_forecast` | 5.52–5.67 h | 491–504 |
| `residual_ppo_forecast` | 5.28–5.39 h | 516–526 |
| `ppo_sinusoid` | 2.15–2.54 h | 1 095–1 291 |

- The forecaster's per-step cost (P6-D1) makes the forecast runs about 1.7× slower.
- `ppo_sinusoid` seeds 3–4 ran with fewer neighbours, which is why they were faster.

*Curriculum.* Every run reached SS5.
- SS4 at 0.4 M steps and SS5 at 0.8 M steps, except `ppo_forecast` seeds 2–3 and `ppo_sinusoid`
  seed 4, which promoted at 0.6 M and 1.0 M.
- The early fall in the residual methods' training return (about 40.5 → 38.9 over the first 3 M
  steps) begins at these promotions: harder sea states give lower returns. Return then recovers
  to about 39.6.

*Learning curves* (`results/e06/`, 5 seeds, mean ± SD with ddof = 1, from
`scripts/export_learning_curves.py`; SHA-256):

| file | SHA-256 |
|---|---|
| `learning_curves_residual_ppo.csv` / `.png` | `ba279218163ac6b5708fca680496423023a62fdea005d5f0f5f8c1c46dd438ca` / `e08536585710dcc0ae0df1a6cbfe9da1076baca624a60bf62e8b7aeaae95875a` |
| `learning_curves_ppo_forecast.csv` / `.png` | `1721dcc620f2be5b905d0b7c46d1df5cdb9422985a266109a44951b9e6cc19ac` / `4041bfc366067200050172b84578d1a10bf72a30b469a118a785d815c38e1dbd` |
| `learning_curves_residual_ppo_forecast.csv` / `.png` | `0bb8291f63017d34158cd89e85df7a39ccbc21d37cb1b8cd46be694d2fd7bf82` / `2320b07de377e11a5fdc4daff7e68e9c0644c2261382119fc85ffbada44f605c` |
| `learning_curves_ppo_sinusoid.csv` / `.png` | `06f7e5397c78f1a6b969cf5e43f7e4d5282b2961fcf5b4548e4c075e787c54d5` / `480ec2e484ba287bb29af6d580835abaddbf41ee9044b7d8189cf20e3e17c354` |

Tune-pool readings, which are not results:
- the residual methods stay at about 1.000 tune success throughout;
- `ppo_forecast` reaches about 1.0 by about 1 M steps;
- the final evals are 0.994–1.000 for every run;
- `ppo_sinusoid`'s curve is on **sinusoid** motion (P6-D1).

### P6-D4 — Phase 6 final checkpoints on the frozen id list: e06 (2026-10-01)

*What was flown.* The `final/` checkpoint of each of the 20 Phase 6 runs on `results/episodes/id.parquet`
(SS3–SS6, N = 200 per cell, aft pad): **16 000 episodes**, none skipped.
- The path is the same as P5-D13: `rld.eval.learned`, chunked `run_matrix`, chunk 25, 24 workers,
  OMP/MKL = 1. The flight took 403.9 s.
- `make_episodes.py --check` ran first: 10/10 OK, MANIFEST `e6f30e55…`, `id` file `919be99d…`,
  content `ea38d8a5…`.
- *Policy path.* `build_policy` returns `ResidualPolicy` (`residual_ppo`), `ForecastPolicy`
  (`ppo_forecast`), `ResidualForecastPolicy` (`residual_ppo_forecast`) or `LearnedPolicy`
  (`ppo_sinusoid`).
  - Every policy is deterministic, with `VecNormalize` frozen, and none is privileged.
  - The two forecast methods receive the runner's past-only `ShipMotionFeed`
    (`needs_motion_feed` from `run_needs_motion_feed`).
  - A pre-flight `check_policies` asserts class, feed flag, privilege and input size (25 or 31).
- **`ppo_sinusoid` flies JONSWAP here.** These rows are its Gate 6 `id` result, **not** the H4
  cross, which is Phase 7.
- *Command.* `python -m rld.eval.learned --learned residual_ppo=…/0,…,4 ppo_forecast=…
  residual_ppo_forecast=… ppo_sinusoid=… --ckpt final --list id --out-dir results/e06 --workers 24
  --carry-learned results/e05=ppo,sac`.

*Provenance.*
- HEAD `7d6fc64`, `git_dirty` true. The dirty paths are only the eval-side `src/rld/eval/learned.py`
  and `tests/test_eval_learned.py`, committed unchanged right after as `0e49f39`. This is the same
  gap as P5-D13, and the audit's re-flights through committed code close it (P6-D5).
- 100 checkpoint files were hashed before and after the flight and did not change. The
  post-flight digests of the 20 runs are committed in `results/e06/summary.csv`
  (`run_model_sha256`, `run_vecnormalize_sha256`, `run_checkpoint_json_sha256`,
  `run_config_sha256`), and they match the files. Only the pre-flight digests are the agent's own
  record (corrected at `/phase-gate 6`, P6-D6). The files are the 20 runs'
  `model.zip`, `vecnormalize.pkl`, `checkpoint.json` and `config.yaml`, plus the 10 Phase 5
  checkpoints.
- **Carried rows.**
  - The six baselines come from e01 and e01_lowvz_cut, line for line, as in P5-D13.
  - `ppo` and `sac` come from e05 (`carried_learned_{summary,seeds,aggregate}_e05.csv`).
  - Every carried line is byte-identical to its source, and the sources flew the identical
    episodes.
  - Re-summarising the source episodes reproduces the source summaries.
  - e05 `--check` is still byte-identical after the eval-side change.
  - As a scratch check, 80 e05 rows re-flown with today's code are 80/80 byte-identical.
- **`--check`** (re-run by the main thread): `summary.csv`, `seeds.csv`, `aggregate.csv`, both
  carried-learned derived files and `success_vs_seastate.md` are all byte-identical.

**Headline: `id`, aft pad, success per sea state.**
- Learned rows show the rliable IQM across 5 seeds (%), with the stratified-bootstrap 95 % CI
  (P3-D1 §4: 2 000 replicates, seed 20260926), then the per-seed range.
- Baselines show % [Wilson 95 % CI] k/200.
- Success is not pooled across sea states.
- SS6 is outside every method's training distribution.

| method | SS3 | SS4 | SS5 | SS6 (out of distribution) |
|---|---|---|---|---|
| `residual_ppo` | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 99.5 [99.5, 99.8]; 99.5–100.0 | 96.2 [95.7, 97.8]; 95.5–98.5 |
| `ppo_forecast` (feed) | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 98.0 [97.5, 98.5]; 97.5–98.5 |
| `residual_ppo_forecast` (feed) | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 99.5 [99.5, 99.5]; 99.5–99.5 | 96.8 [96.5, 98.0]; 96.5–98.5 |
| `ppo_sinusoid` (JONSWAP `id`; not H4) | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 97.3 [96.7, 97.8]; 96.5–98.0 |
| `ppo` (carried, e05) | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 100.0 [100.0, 100.0]; 100.0–100.0 | 98.2 [98.0, 98.5]; 98.0–98.5 |
| `sac` (carried, e05; 2 M steps) | 99.8 [98.8, 100.0]; 98.5–100.0 | 98.2 [96.3, 99.0]; 96.0–99.0 | 87.0 [81.7, 90.3]; 80.0–91.0 | 72.5 [60.2, 79.3]; 55.5–81.5 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200 | 96.0 [92.3, 98.0] 192 | 85.0 [79.4, 89.3] 170 | 80.0 [73.9, 85.0] 160 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200 | 100.0 [98.1, 100.0] 200 | 99.0 [96.4, 99.7] 198 | 90.5 [85.6, 93.8] 181 |
| `pid_feedforward_lowvz` | 100.0 [98.1, 100.0] 200 | 98.0 [95.0, 99.2] 196 | 95.5 [91.7, 97.6] 191 | 85.0 [79.4, 89.3] 170 |
| `pid_feedforward_lowvz_cut` | 100.0 [98.1, 100.0] 200 | 98.5 [95.7, 99.5] 197 | 98.0 [95.0, 99.2] 196 | 88.0 [82.8, 91.8] 176 |
| `gated` | 100.0 [98.1, 100.0] 200 | 98.5 [95.7, 99.5] 197 | 89.5 [84.5, 93.0] 179 | 63.0 [56.1, 69.4] 126 |
| `oracle_gated` (privileged; commit-timing oracle) | 100.0 [98.1, 100.0] 200 | 98.5 [95.7, 99.5] 197 | 89.0 [83.9, 92.6] 178 | 64.5 [57.7, 70.8] 129 |

- *Per-seed SS6 counts* (of 200, seeds 0–4; per-seed Wilson CIs are in `success_vs_seastate.md`):
  - `residual_ppo` 192 / 197 / 193 / 192 / 191;
  - `ppo_forecast` 196 / 197 / 195 / 195 / 197;
  - `residual_ppo_forecast` 197 / 193 / 194 / 193 / 194;
  - `ppo_sinusoid` 196 / 195 / 194 / 195 / 193.
- *Optimality gap at SS6* (points): `residual_ppo` 3.5 [2.4, 4.2], `ppo_forecast` 2.0 [1.6, 2.4],
  `residual_ppo_forecast` 2.9 [2.2, 3.4], `ppo_sinusoid` 2.7 [2.3, 3.2].
- *What the seed CI is.* It reflects seed variation only. Where every seed scores 200/200 it
  collapses, and the episode-level uncertainty is then each seed's Wilson CI, [98.1, 100.0].
- **No contrast is tested here.** Every comparison below is an unpaired reading of the tables. H1b
  (`residual_ppo` vs `pid_feedforward` at SS6), H2 and H3 are Phase 7's paired tests.

**Closing speed** (p95 along the deck normal, m/s model scale). Learned rows show the IQM of the
per-seed p95s [CI]. These are descriptive, not P3-D1 §4's pooled relative-p95 statistic, and the
recorded value understates impact speed by about 7 % (P5-D14).

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `residual_ppo` | 0.269 [0.267, 0.271] | 0.271 [0.266, 0.273] | 0.277 [0.272, 0.279] | 0.283 [0.281, 0.287] |
| `ppo_forecast` | 0.268 [0.262, 0.271] | 0.269 [0.264, 0.271] | 0.272 [0.268, 0.275] | 0.281 [0.272, 0.295] |
| `residual_ppo_forecast` | 0.277 [0.274, 0.281] | 0.279 [0.275, 0.281] | 0.278 [0.276, 0.280] | 0.278 [0.274, 0.280] |
| `ppo_sinusoid` | 0.274 [0.270, 0.277] | 0.275 [0.271, 0.279] | 0.282 [0.278, 0.285] | 0.288 [0.282, 0.292] |
| `ppo` | 0.265 [0.263, 0.266] | 0.267 [0.264, 0.270] | 0.271 [0.267, 0.275] | 0.278 [0.274, 0.285] |
| `sac` | 0.392 [0.369, 0.433] | 0.437 [0.406, 0.478] | 0.595 [0.562, 0.619] | 0.668 [0.590, 0.778] |
| `pid_feedforward` | 0.219 | 0.226 | 0.262 | 0.264 |
| `pid_feedforward_lowvz` (H1a reference) | 0.127 | 0.139 | 0.188 | 0.181 |
| `pid_feedforward_lowvz_cut` | 0.127 | 0.139 | 0.188 | 0.181 |
| `pid_track_descend` | 0.308 | 0.412 | 0.525 | 0.502 |
| `gated` | 0.217 | 0.229 | 0.250 | 0.267 |
| `oracle_gated` (privileged) | 0.220 | 0.236 | 0.252 | 0.254 |

**Outcome breakdown where the new methods lose** (counts per 1 000 seed-episodes; every cell's six
fractions sum to 1; anything not listed is 0).

| method | SS5 | SS6 | tunnelling > 5 mm (of 4 000) | max depth |
|---|---|---|---|---|
| `residual_ppo` | 4 `hard_landing` | 24 `hard_landing`, 11 `bounce` | 7 | 6.17 mm |
| `ppo_forecast` | 0 | 18 `hard_landing`, 2 `bounce` | 10 | 6.21 mm |
| `residual_ppo_forecast` | 5 `hard_landing` | 25 `hard_landing`, 4 `bounce` | 3 | 5.65 mm |
| `ppo_sinusoid` | 0 | 25 `hard_landing`, 2 `bounce` | 11 | 7.12 mm |

- In all 80 new cells there are **0** `crash`, `off_pad` and `timeout`. Detector disagreement is
  0 / 16 000.
- *Post hoc.* Every `hard_landing` of the four new methods (and of `ppo`) breaks the 15°
  relative-tilt limit, not the 0.5 m/s limit; P2-D5 folds tilt into `hard_landing`. The largest
  closing speed in any of them is 0.390 m/s.
- Every residual SS5 loss is the same episode, `id` SS5 #45: 9 losses, one per seed except
  `residual_ppo` seed 4, which lands it. The deck tilt at touchdown is 16.9–21.7° across the 9
  (`results/audit/e06/hard_landing_tilt.csv`). `pid_feedforward` loses it too. *(Corrected at
  the Gate 6 review, P6-D6.)*

**Seed-to-seed spread** (`seeds.csv`; sample SD, ddof = 1).
- Success SD is 0.0 points at SS3–SS4 for all four new methods.
- At SS5 it is 0.22 for `residual_ppo` and 0.0 for the others.
- At SS6: `residual_ppo` 1.17, `ppo_forecast` 0.50, `residual_ppo_forecast` 0.82,
  `ppo_sinusoid` 0.57 (for comparison, `ppo` 0.27 and `sac` 9.65).
- p95 SD is 0.002–0.005 m/s, except `ppo_forecast` at SS6 with 0.012 (per-seed 0.271–0.295).
- No seed is an outlier.

**Readings.** These are unpaired and untested, and are reported, not acted on.
1. **At SS6, every seed of all four new methods (95.5–98.5 %) is above `pid_feedforward`'s single
   run** (90.5 % [85.6, 93.8]). Its Wilson upper bound is below every new-method seed's point
   estimate, but the episode-level CIs overlap: the lowest seed, 191/200, has Wilson [91.7, 97.6].
   The seed-bootstrap CI leaves out episode uncertainty. H1b is Phase 7's paired test. Every
   RL-vs-PID gap carries two facts from the Phase 6 "Before you start" note (c):
   - the learned methods land with a two-phase descent that the PID tuning space cannot express;
   - 11 of `pid_feedforward`'s 19 SS6 losses are `bounce`, a class driven by the 50 ms grace rule
     (P5-D1). The new methods' SS6 bounces are 2–11 per 1 000.
2. **None of the four is above `ppo`.** At SS6, the seed-bootstrap CIs of `residual_ppo` and
   `ppo_sinusoid` lie wholly below `ppo`'s, and `ppo_forecast`'s overlaps it. Phase 7 tests these
   contrasts.
3. **The residual policies no longer behave like their base.**
   - Median time to touchdown is 1.94–2.01 s, against 4.4–4.6 s for `pid_feedforward`.
   - p50 closing speed is 0.247–0.263 m/s, against about 0.19 m/s.
   - Their p95 is above `pid_feedforward`'s in every cell, and well above the H1a reference
     `pid_feedforward_lowvz` (0.188 m/s at SS5).
   - H1a is scored only in Phase 7, but this table gives no sign of the ≥ 15 % reduction H1a
     predicts. The post-contact throttle-cut confound is checked per residual seed in P6-D5.
4. **`ppo_sinusoid` on JONSWAP `id` is within about 1 point of `ppo`** (100 % at SS3–SS5, 97.3 % at
   SS6). Training on sinusoids alone shows no visible penalty on this list. That is not H4, which
   compares *drops* across both test motions in Phase 7, but it bears on it. **It already
   bounds H4 at its pre-registered cell: see P6-D6.**
5. **The forecast block brings no visible gain.** `ppo_forecast` matches `ppo` (both 100 % at
   SS5; 98.0 vs 98.2 at SS6), and its p95 is the same to within 0.003 m/s. Both P6-D1 caveats
   apply. H3 is Phase 7's.
6. **Tunnelling.** `ppo_sinusoid`'s 7.12 mm maximum is just above the 7.03 mm `lowvz_cut` reference
   that P5-D14 used as a limit. Its rates are 0.075–0.28 % of 4 000. The mechanism is in P6-D5.

*Artifacts, in `results/e06/` (SHA-256).*

| file | SHA-256 |
|---|---|
| `episodes.csv` (16 000 rows) | `eb832a557691f920d24fd5cde407df5ca35162f941ac705d1bcf76545803f358` |
| `summary.csv` (80 rows) | `7aa5c49b4e692432462cd53417bc9563f1fdefb9279fae969b2b26d13b5310a3` |
| `seeds.csv` (16 rows) | `02e816c7bc4e334ae7823e8662f583d51a303425b055f4479b8dda2fef56dc16` |
| `aggregate.csv` (48 rows) | `6974e3896efe504fabc3553510afa123d49c731d8dd3bd33ed285e9bbe550d79` |
| `success_vs_seastate.md` | `8d6ff5721a8bed2de1440da023023820e460f6a4051fb9f953ac167888200cdb` |
| `carried_summary_e01.csv` | `27da5b6f8db170cbcb0c794c039113018d82242b2beb9470a8149f8823b2d277` |
| `carried_summary_e01_lowvz_cut.csv` | `39101ccbf2df946ac838bb635f3e295d58d1008ef175987d4301f25cdcda9594` |
| `carried_learned_summary_e05.csv` | `a2a0f50f77012a4488c6c28939a170725ed95453d2a9589f10612a7a1d6192f2` |
| `carried_learned_seeds_e05.csv` | `cf61ce010b08535fabe96996cd5e8259ac04578cbf4bc95c8fcfe6a7883c1613` |
| `carried_learned_aggregate_e05.csv` | `ded3cdd53290d025f60c81a029d8545dcaf6d99babe204dd03d842acfa2874b9` |

`run_info.json` holds the non-deterministic facts and is not hashed here. No hypothesis is scored,
and P3-D1 is unchanged.

### P6-D5 — Reward-hacking audit of the twenty final Phase 6 runs (2026-10-01)

*Pre-statement, now provable.* The checks and thresholds were committed alone in
`results/audit/e06/README.md` at **`f66bca6`, 2026-10-01 14:44:22 EDT**. The first audit CSV was
written at 15:11 EDT.
- The thresholds are P5-D14's, unchanged, plus Phase 6 checks 1b-fine, 9 and 10 and the
  per-seed reporting of check 6.
- The README was only appended to afterwards (+328 lines, 0 removed; +339 / −0 from `f66bca6` after the P6-D6 errata).
- One slip is recorded in the README's own post-run section: its header reads "written 14:39–15:10
  EDT", and the 15:10 was an estimate typed before the commit.
- This closes P5-D14's gap, where the timestamp was only the agent's own record.

*Scope and reproduction.*
- `python scripts/reward_hacking_audit.py --phase 6 --out-dir results/audit/e06 --scratch-dir <dir>
  --workers 24`, at `1d57571`.
- **2 340 of 2 340 re-flights reproduce their committed e06 row** through committed code. They
  cover:
  - 2 200 sampled flights: 2 000 learned (25 per SS per run) plus 200 baseline
    (`pid_feedforward` 100, `pid_feedforward_lowvz_cut` 100; `reflight_counts.csv`);
  - every tunnelled episode and every hard landing;
  - 1 053 forecast-run flights on the runner's past-only feed.
- This also closes P6-D4's provenance gap.
- A second run with different worker and chunk settings gave 28 of 28 byte-identical CSVs.
- The Phase 5 audit, re-run into scratch after the code change, is 16 of 16 byte-identical. The
  committed `results/audit/*` did not change.

*Verdicts* (`verdicts.csv`), in the order `residual_ppo` / `ppo_forecast` / `residual_ppo_forecast` / `ppo_sinusoid`:

| check | `residual_ppo` | `ppo_forecast` | `residual_ppo_forecast` | `ppo_sinusoid` |
|---|---|---|---|---|
| 1a training timeouts, final 10 % | clean (0.000) | clean (0.000) | clean (0.000) | clean (0.000) |
| 1b training timeouts, 10 bins | clean (0.000) | clean (0.091) | clean (0.000) | clean (0.059) |
| 1b-fine, 1 % bins, bin 0 | clean (0.000) | **finding** (0.901) | clean (0.000) | **finding** (0.908) |
| 1b-fine, bins 1–99 | clean | **finding** (seed 2: 0.785 at 200–300 k) | clean | clean (0.430) |
| 1c e06 timeouts | clean (0) | clean (0) | clean (0) | clean (0) |
| 2a tunnelling rate; max depth (limit `lowvz_cut` 0.75 %, 7.03 mm) | clean (0.175 %; 6.17 mm) | clean (0.25 %; 6.21 mm) | clean (0.075 %; 5.65 mm) | **finding** (0.275 %; **7.12 mm**) |
| 2b mechanism, impact / idle / other | 0 / 1 / 6 | 0 / 7 / 3 | 0 / 2 / 1 | 0 / 8 / 3 |
| 2c outcome may depend on tunnelling | **finding** (3 of 7) | **finding** (1 of 10) | clean (0 of 3) | **finding** (3 of 11) |
| 3 detector disagreement | clean (0) | clean (0) | clean (0) | clean (0) |
| 4 success concentrated in easy start states (48 tests) | clean | clean | clean | clean |
| 5 pre-contact norm-cap saturation | clean (0.0 %) | clean (3.1 %) | clean (0.0 %) | clean (0.7 %) |
| 6 post-contact down-force, pooled (idle fraction; median setpoint m/s) | clean (0.055; −0.24) | clean (0.471; −0.33) | clean (0.055; −0.24) | **finding** (0.549; −0.37) |
| 7 passive, deck-driven landings | clean (1.2 %) | clean (4.0 %) | clean (0.6 %) | clean (3.0 %) |
| 8 seed outliers | minor | minor | minor | minor |
| 9 residual authority-limited (π_z at its limit on > 50 % of pre-contact steps) | no (46.5 %) | — | no (45.2 %) | — |
| 10a forecast block changes the action (> 0.015 m/s median) | — | yes (0.0455 m/s) | yes (0.0269 m/s) | — |
| 10b forecast block changes outcomes | — | not shown | not shown | — |

*Readings.*
- **H1a confound, check 6, per residual seed** (`downforce_per_seed.csv`; idle fraction, then median
  post-contact setpoint in m/s):
  - `residual_ppo` seeds 0–4: idle 0.115 / 0.036 / 0.064 / 0.048 / 0.014; setpoint −0.29 / −0.22 /
    −0.19 / −0.29 / −0.21.
  - `residual_ppo_forecast` seeds 0–4: idle 0.032 / 0.010 / 0.019 / 0.006 / 0.210; setpoint −0.23 /
    −0.19 / −0.27 / −0.20 / −0.31.
  - For reference, `pid_feedforward_lowvz_cut` gives 0.948 / −1.50 and the base `pid_feedforward`
    0.000 / −0.20.
  - **No residual seed has learned the post-contact throttle cut.**
  - Their SS6 bounces, 11 and 4 per 1 000, are below `pid_feedforward` (55), `lowvz_cut` (75) and
    `lowvz` (105). So their low bounce rate is not a throttle-cut effect. This is carried beside
    H1a, together with the "Before you start" (c) caveat: `bounce` is driven by the 50 ms
    contact-loss grace rule and is unstable at 240 Hz (P5-D1).
- **The pure methods repeat `ppo`'s split** (P5-D14): 3 of 5 seeds cut to idle after contact
  (`ppo_forecast` seeds 0–2; `ppo_sinusoid` seeds 0, 3, 4). Pooled, `ppo_sinusoid` crosses the
  check-6 threshold.
- **Hovering.**
  - The residual methods never time out in training.
  - The pure methods time out at 0.75–0.91 in their first 100 k steps, as `ppo` did. An untrained
    policy and a learned trap cannot be told apart there.
  - `ppo_forecast` seed 2 relapses: training timeouts go 0.75, 0.28, then **0.785 at 200–300 k
    steps**, then 0.15, then 0 from 400 k on.
  - No final policy hovers (1c).
- **The residual uses most of its authority.**
  - Before contact, |α·π| is 0.41 / 0.36 m/s at the median and 0.70 / 0.71 m/s at most
    (`residual_ppo` / `residual_ppo_forecast`).
  - π_z sits at its limit on 46.5 % / 45.2 % of pre-contact steps, just short of the pre-stated
    50 % mark.
  - The output clip and the norm cap never bind.
  - **This is how touchdown fell from 4.57 s to about 1.98 s:**
    - The residual holds −0.45 m/s through the approach, while the base commands +0.02 to
      −0.15 m/s.
    - About 1 s out its own contribution falls to about −0.06 to −0.07 m/s. That is still
      *downward*, on top of a base command of −0.19 to −0.21 m/s, so the last second is flown
      at −0.25 to −0.29 m/s (`residual_descent_profile.csv`).
    - That is `ppo`'s two-phase descent, rebuilt within the residual's bound.
    - *Post hoc (corrected at the Gate 6 review, P6-D6).* In the last second the residual pushes
      0.06–0.07 m/s **below** the base. A descent as slow as `pid_feedforward_lowvz`'s 0.111 m/s
      needed about +0.09 m/s, roughly 20 % of the ±0.45 m/s authority. It was within reach and
      was not learned; check 9 also finds the residual not authority-limited. An earlier version
      of this bullet said the authority bound made the H1a reference out of reach. That was
      wrong.
- **The forecast block is used, but no change in outcome is shown.**
  - Replacing the block with its training mean or with zeros moves the executed setpoint by a
    median 0.045 m/s (`ppo_forecast`) and 0.027 m/s (`residual_ppo_forecast`). That is above the
    pre-stated 0.015 m/s for every seed.
  - In 2 000 counterfactual flights, touchdown shifts by about one control step at the median, and
    the outcome changes once.
  - The paired success difference is −0.002 [−0.006, 0.000] and 0 [0, 0].
  - *Limit of the test.* The original sample is 500 of 500 successes, so it could only detect
    losses.
- **Tilt-only hard landings.**
  - All 101 new-method hard landings are tilt-only. So are `ppo`'s 15 and `pid_feedforward`'s 9.
  - The drone's own tilt is ≤ 5.8° in every one, against a deck tilt of 10.8–21.7°. In 77 of the
    101 the deck alone exceeds 15°.
  - All 125 fall on 27 listed episodes.
  - These are deck-tilt events with a nearly level drone. The frozen relative-tilt criterion
    (P2-D5) scores them as `hard_landing` for every method alike; it is not a new-method effect.
- **Tunnelling.**
  - All 31 tunnelled episodes are **post-contact**, none impact. Maximum depth comes 62–525 ms
    after contact, at closing speeds of 0.17–0.39 m/s.
  - 18 happen at idle motors, the P5-D14 mechanism.
  - 13 happen under low, non-idle thrust (0.44–0.68 of weight). That is `residual_ppo`'s main case,
    near the end of the 0.5 s dwell.
  - `ppo_sinusoid`'s 7.12 mm maximum, 0.09 mm over the reference, is seed 2, SS6 #58: idle motors
    88 ms after contact, in an episode that was already a tilt `hard_landing` at first contact.
  - **Bound, under P5-D14's rule.** Up to 7 SS6 successes may depend on tunnelling overlap:
    `residual_ppo` 3, `ppo_forecast` 1 and `ppo_sinusoid` 3, per 1 000 seed-episodes. Their
    unloaded stretches are 4–33 ms. That is at most 0.3 points of any cell, and no reading in P6-D4
    changes direction because of it.
- **Measurement caveats carried** (P5-D14): recorded closing speed is about 7 % below impact speed,
  and penetration is flagged at any contact substep.

*Code.*
- `src/rld/rl/audit.py`: an `AuditSpec` with `--phase {5,6}`; the feed for forecast runs; read-only
  hooks whose recorded base, π and input rebuild the executed action bit for bit or the audit
  stops; a scratch-only forecast-block replacement.
- `src/rld/rl/audit_phase6.py` (new).
- `tests/test_rl_audit_phase6.py` (new, 18 tests). `test_rl_leakage.py` still passes.

*Artifacts (SHA-256).*
- `results/audit/e06/README.md`: `2b2f8ad33b589fac429cb0d4f7bcdee1e6b4a6816923c186028525511739ee92` after the Gate 6 errata were appended (P6-D6). As first committed in `1d57571` it was `5561232b59bd40aa193787d7ccc1310810cc9c384d5d8f0672491396b45e6a47`.
- `verdicts.csv`: `dcdbcc9564fc5ae91faac0f9c5bc4003c7caa1da367d2c659f3e007515d81718`.
- `downforce_per_seed.csv`: `a727be5dbff6c311066f22fc54ba705f228ea43c8b0bb923fe548770adb0e1b2`.
- `residual_authority.csv`: `5e8c4de7b205da76b3df5d7847a994d2a42650630ae10986629aa0b2c23b8727`.
- `forecast_dependence.csv`: `25c93ac168a52ce7f38ca0886b4a98f8e7fdee611be11e4175657d3d1d76710f`.
- `tunnelling_episodes.csv`: `6eea307765581ed6794d68a82acfeffce9155e4ed1962c802ad148f2a3a030b2`.
- `hard_landing_tilt.csv`: `8470c63cc67bf570e304f4ec5a2f6d3b43d296f3938c863ab710bb374b24207d`.
- `training_bins_fine.csv`: `9520ed9559cf9ace966ef3b34fef02826199ed88362a8e248186b8baea736e6a`.
- The remaining 21 CSVs are hashed in the README's post-run section.

### P6-D6 — Gate 6 review fold-in; H4 is bounded at its pre-registered cell (2026-10-01)

*Review.* `results-skeptic` reviewed at `7caa2e6`. There is **no BLOCKING finding**, and no split
leakage. Two findings are MAJOR and nine MINOR. All are fixed in this commit or recorded below.

**M1. P6-D5 misread the residual's descent (fixed in P6-D5).**
- P6-D5 said the residual's 0.45 m/s authority made the H1a reference out of reach.
- `results/audit/e06/residual_descent_profile.csv` shows otherwise. In the last second the residual
  adds −0.06 to −0.07 m/s to a base command of −0.19 to −0.21 m/s.
- `lowvz`'s descent needed about +0.09 m/s, which was within authority. Check 9's pre-stated
  verdict was also "not authority-limited".
- The bullet is corrected in place and marked.

**M2. H4 is arithmetically bounded at its pre-registered cell (dated note; H4 is NOT changed).**
- *Definitions.* H4 (P3-D1 §8) is scored at `id` SS5:
  - drop_sin = success(`ppo_sinusoid` on sinusoid) − success(`ppo_sinusoid` on JONSWAP);
  - drop_jon = success(`ppo` on JONSWAP) − success(`ppo` on sinusoid);
  - prediction: drop_sin − drop_jon ≥ 10 points.
- *What e06 already fixes.* `ppo_sinusoid` on JONSWAP `id` SS5 is 200/200 in all 5 seeds, and so
  is `ppo` (e05).
- *The bound.* With x and s the two policies' sinusoid-leg success rates (%):
  - drop_sin = s − 100 ≤ 0;
  - drop_jon = 100 − x ≥ 0.
  - So drop_sin − drop_jon = s + x − 200 ≤ 0, **whatever the Phase 7 sinusoid leg shows**.
- *What follows.* H4 cannot be supported at `id` SS5. Under D0.4 (plan §0), a sinusoid-trained
  policy that transfers is reported as the finding, and the novelty claim is withdrawn in the
  README.
- *Why it is written down now.* This mirrors P3-D1 §7's H1-ceiling paragraph. It is recorded
  before any sinusoid-test-motion episode exists.
- *What is not done.* Nothing is re-scored here. H4 stays as pre-registered, and Phase 7 scores it
  as written.
- *If the cell moves.* Scoring H4 at another cell (for example `id` SS6, where `ppo_sinusoid` is
  97.3 % and `ppo` 98.2 %) or under another statistic would be a **dated deviation**. It is the
  user's decision, and it must be made before any sinusoid-leg result is read.
- **User decision (2026-10-01): H4 stays at `id` SS5, as pre-registered.** No deviation is
  recorded. It was decided before any sinusoid-test-motion episode existed. Phase 7 scores H4 as
  written. Because the point estimate is bounded at ≤ 0, H4 cannot be "supported". The README
  withdraws the novelty claim and reports the transfer as the finding. Two triggers are cited:
  - D0.4: the sinusoid-trained policy transfers, scoring 100 % on JONSWAP `id` SS5;
  - P3-D1 §8: the CI includes 0.
  If both sinusoid legs lose heavily, the CI could lie wholly below 0. §8's literal trigger
  would then not fire, but D0.4's still does, so the outcome is the same either way.
  *(The second trigger was added at `/phase-gate 6`.)*

**MINOR.**
1. *The `residual_ppo_forecast` zeroed-residual check is weaker than the gate test.*
   `tests/test_rl_forecast_obs.py` (2 episodes, rows only) can skip without the forecaster.
   - It ran with **0 skips** in the reviewer's run and in the Gate 6 `make test`. The suite's
     single skip is `test_platform.py:191`, P2-D1, by design.
   - The Gate 6 row records this.
2. *SS5 #45 sentence in P6-D4.* Corrected in place: 9 losses; `residual_ppo` seed 4 lands it; deck
   tilt 16.9–21.7°.
3. *"100 checkpoint files hashed" in P6-D4.* The post-flight digests are committed in
   `results/e06/summary.csv`; only the pre-flight digests are the agent's own record. The
   P6-D4 sentence says so. *(This item first claimed that no digest was committed. That was
   corrected at `/phase-gate 6`.)*
4. *Stale line in `results/e06/success_vs_seastate.md`* ("Phase 6 methods not audited … no bound
   exists"). It is superseded by P6-D5, whose tunnelling bound is 3 / 1 / 0 / 3 SS6 successes per
   1 000. The file is not re-rendered, so its committed hash stands. Phase 7's render carries the
   bound.
5. *Grace-rule caveat on the residual bounce comparison.* Added to P6-D5, and appended to
   `results/audit/e06/README.md` as an erratum.
6. *"Not supported" used early in the audit README.* An appended erratum reads it as "shows no sign
   of". The pre-stated section is untouched.
7. *P6-D4 reading 1 compared a Wilson bound with point estimates.* The episode-level CI overlap is
   now stated, and H1b is named as Phase 7's paired test.
8. *The `ppo_sinusoid` learning-curve figure is not labelled as sinusoid motion.* The tune-pool
   success and closing-speed panels of `results/e06/learning_curves_ppo_sinusoid.png` are on
   **sinusoid** deck motion (P6-D1, P6-D3). The figure is not regenerated, so its committed hash
   stands; Phase 9's figures carry the label.
9. *`src/rld/rl/forecast_obs.py` docstring.* It said "in-sample on the train and tune pools". The
   DLinear-OLS point forecast, the only part the block reads, is fitted on the **train** pool only
   (`results/forecast/fit_manifest.json`). So tune-pool evaluations, curriculum promotion included,
   saw out-of-sample forecasts. The docstring is corrected; this is a docstring-only change.
   P6-D1's "in-sample during training" stands.

**Notes carried without action.**
- The gate test uses an untrained network with a zeroed `action_net`. That is equivalent to a
  zeroed trained residual: zero weight and bias give exactly 0 for any finite input.
- The audit thresholds were pre-stated after e06's numbers were known. The README says so.
- The residual's executed exploration std, 0.018, is about 3.3× below `ppo`'s (P6-D1).
- `ppo_sinusoid`'s amplitudes and periods are *statistics* of the matched JONSWAP realizations, by
  design. "No JONSWAP motion" means no JONSWAP time series.

## Phase 7

### P7-D1 — Phase 7 arm definitions, fixed before any Phase 7 flight (2026-10-01)

*Status.* This entry is committed **alone**, before any Phase 7 episode is flown. That includes the
sinusoid test leg, the λ, noise, pad-at-CG and MSS arms, and the first full-matrix episode of a
learned method outside `id`. It specifies what P3-D1 left open. **It changes no threshold, no
prediction, no episode list and no success criterion.** The P3-D1 block SHA-256 stays
`21465588…`. The user's decisions are dated 2026-10-01 and were taken at the Phase 7 plan review.

**1. Sinusoid test motion (the H4 cross).**
- The test leg is built by `rld.rl.motion.sinusoid_motion`, the training builder, called from the
  evaluation runner. That makes the training and test definitions one function.
  - Amplitude per DOF: √2 × the realization's RMS over its whole record, full scale, read from the
    committed **aft** rows of `results/deck_stats_seeds.csv`.
  - Period: the realization's peak encounter period.
  - Phases: one per DOF, drawn U(0, 2π) from the listed **episode seed**.
- The leg flies the **same committed lists** as JONSWAP. A sinusoid source has the JONSWAP source's
  window and full-scale lookback, so the start offset and the initial state are the listed ones.
  The runner's strict start check holds unchanged.
- Every method flies the leg on all four `id` sea states, aft pad, the three always-printed
  baselines included. H4 is read at `id` SS5 only. The other cells are descriptive.
- The two forecast methods receive a past-only feed of the **sinusoid** ship motion. That is the
  motion they would sense.

**2. Estimators that P3-D1 §8 names but does not fully specify.**
- *Success of a learned method in a cell.* The rliable IQM over the 5 seeds of each seed's success
  rate. A deterministic baseline has one run, so its value is its success rate.
- *H2.* drop(m) = IQM success at `id` SS5 − IQM success at `unseen_seastate` SS6. Each replicate:
  - resamples seeds per method (shared across a method's two cells, since a seed is one policy);
  - resamples episodes **independently in each cell**, because the two cells are different
    episodes.

  The statistic is drop(`ppo`) − drop(`residual_ppo`).
- *H4.* Both test motions fly the same episode list. So each replicate resamples seeds per method,
  then **one** set of episode indices, shared by both motions and both methods. The statistic is
  drop_sin − drop_jon.
- *All contrasts.* 10 000 replicates, bootstrap seed 20260926, percentile 95 % CI. "Separates"
  means the CI excludes 0.
- *H1a/H3 relative-p95.* As P3-D1 §4 and P3-D4 #6. The P3-D4 #8 timeout-ranked variant is
  reported beside it as a sensitivity analysis, never scored.
- *H3, `unseen_vessel` part.* The half-rule is judged on point estimates: r(`unseen_vessel`,
  SS) ≤ 0.5 · r(`id`, SS), per sea state, with both CIs reported. If r(`id`, SS) ≤ 0, the part is
  scored "not applicable — no `id` gain to shrink".

**3. λ sensitivity arm.**
- *Selection rule (user, 2026-10-01).* "The best two methods" means the best **learned** method
  and the best **classical** (non-privileged) controller. Each is ranked by success at `id` SS6
  (aft, JONSWAP; IQM for learned methods) in the committed e05, e06, e01 and e01_lowvz_cut tables.
  Ties go to the lower p95 closing speed, then to the lower method name.
  - It selects **`ppo`** (IQM 0.9817, against `ppo_forecast` 0.9800) and **`pid_feedforward`**
    (181/200, against `pid_feedforward_lowvz_cut` 176/200).
  - These are Phase 5 and 6 numbers. No Phase 7 number exists yet.
- *λ.* {1/15, 1/40}, against the main matrix at 1/25. This is a sensitivity arm. **The project λ
  stays 1/25 (P1-D1, P1-D3).** 1/15 is outside the declared ladder. Its SS6 deck-point v_z p99 is
  reported against the P1-D1 feasibility threshold, for context only.
- *Flown.* `ppo` × seeds 0–4 and `pid_feedforward`, plus the always-printed `pid_track_descend`
  and `oracle_gated`. Lists: `id`, `unseen_seastate`, `unseen_heading`, `unseen_vessel`, aft pad.
  The static list is λ-independent and is not flown.
- **The episodes at λ ≠ 1/25 are not identical to the listed ones.**
  - The realization, the episode seed, the initial drone position and the pad are the listed ones.
    The initial state comes from its own spawned stream, independent of the motion window.
  - The start offset t0 is **re-drawn by the environment** from the same episode seed, inside the
    new λ's window. That window does not scale in proportion: the episode is 12 s model at every λ,
    but the record shrinks by √λ. At 1/40 some listed full-scale offsets fall outside it.
  - The runner's start check for this arm asserts the realization key, the initial position and
    the pad, and records t0. It does not compare t0.
  - Contrasts against λ = 1/25 therefore use the H2-style bootstrap: seeds resampled, episodes
    resampled **independently** per λ. A per-episode paired difference is never reported for
    this arm.

**4. Perception stand-in arm.**
- *Grid.* σ_p ∈ {0, 1, 2, 4} cm × latency ∈ {0, 1, 2} control steps, i.e. {0, 33.3, 66.7} ms
  model scale. The plan writes the latencies as {0, 33, 66} ms. 1 control step at 30 Hz is 33.3 ms,
  or 167 ms full scale.
  - *Configured values.* `PerceptionNoise` quantises latency **down** to whole steps, so a
    literal 33 ms would be **0** steps, identical to the clean condition. The configured values are
    therefore **33.4 ms and 66.7 ms**, which quantise to exactly 1 and 2 steps. A unit test asserts
    the step counts.
  - *Labels.* Every table states the effective latency in steps and in ms.
  - *Why recorded now.* The approved Phase 7 plan already read the grid as 0 / 1 / 2 steps. This
    item records that reading before any flight, so that the quantisation cannot silently drop a
    condition.
- *Velocity noise (user: position and velocity).* σ_v = σ_p / τ, with τ = 0.2 s model (1 s full
  scale), giving σ_v ∈ {0, 0.05, 0.10, 0.20} m/s model scale.
  - Rationale: a relative-velocity estimate smoothed over about 1 s full scale. It is a stated
    modelling choice, not a measured estimator. (Limitation recorded in P7-D6: the noise as
    implemented is white, i.i.d. per control step.)
  - The SS5 `id` deck v_z std is about 0.36 m/s for comparison (plan D0.1, scouting numbers).
    (corrected in P7-D6)
- *Hold.* 30 Hz, i.e. no hold.
- *Scope.* Noise and latency apply to the six relative-pad entries only (`configs/env/noise.yaml`). *(Superseded by the P7-D4 deviation: the stand-in now perceives the deck, and every deck-derived entry is perceived.)*
  The drone's own state is clean. **The forecast feed of `ppo_forecast` and
  `residual_ppo_forecast` stays ideal**, and that is stated beside them.
- *Flown.* Every method (all 6 learned × 5 seeds and the 6 baselines), all four `id` sea states,
  aft pad, JONSWAP. That is 11 non-clean conditions. The clean (0, 0) condition is the main
  matrix's rows.
- *Pairing.* Noise draws from its own spawned child of the episode seed. So t0 and the initial
  state are the listed ones under every condition, and the strict start check holds. Contrasts
  against clean are paired over identical episodes.

**5. Pad-at-CG control (D0.5).**
- Every learned method × seed and `pid_feedforward_lowvz_cut` fly every list with the pad
  overridden to `cg`. The static list is not flown, because its pad is the CG by construction.
- The other baselines' CG rows are carried line-for-line from `results/e02/`, which flew them on
  the identical lists.
- The start offset and the initial state do not depend on the pad, so the strict check holds.
- No hypothesis is scored at CG. The arm isolates the dmf roll/pitch-heave phase defect's lever-arm
  effect (aft − CG, per method and cell), and it is descriptive.

**6. MSS transfer arm (optional; user: run it).**
- *Records.* Project 4's MSS strip-theory records:
  - ITTC S-175, SS5 JONSWAP (Hs 3.3 m, Tp 9.7 s, γ 3.3);
  - headings {180°, 135°} × speeds {0, 6, 12} kn × seeds {0, 1, 2}, i.e. 18 realizations per grid
    kind;
  - grid kinds `mss` (primary, MSS spectrum and RAOs) and `corpus` (the attribution control: dmf's
    own wave field through the MSS transfer function).
- *Provenance.* MSS upstream is cloned at dmf's pinned SHA
  `98970f71a21cfe81e7e29abdcc1bb6741789cddc` into `artifacts/mss/upstream/`. That path is
  gitignored; `third_party/` is never edited. dmf's config is copied to
  `configs/deck/mss_s175_ss5.yaml` with only `mat_path` changed. The Octave parity check is run
  and its result recorded.
- *Motion.* Evaluated **analytically** on the physics grid from the same seeded wave grid as dmf's
  export. It is never interpolated from the 10 Hz CSV. Its 10 Hz samples are tested equal to the
  exported records. It is Froude-scaled at λ = 1/25, with the S175 lever arm.
- *Episodes.* A separate list directory `results/episodes_mss/` with its own MANIFEST.
  - Committed before any MSS flight.
  - Regimes `mss_transfer` (grid `mss`) and `mss_transfer_corpus` (grid `corpus`), N = 200 each.
  - Drawn `balanced_round_robin` over the 18 realizations, generator seed **20261001**.
  - The frozen `results/episodes/` and its MANIFEST `e6f30e55…` are not touched.
  - All of these realizations are outside every training distribution: S175 is never trained on.
- *Flown.* Every method, aft and CG.
- *Reporting.* The tables sit beside `unseen_vessel` SS5 restricted to the 180° and 135° headings.
  That comparison is descriptive: the motions are different realizations, and nothing is scored.
  These are another simulator's trajectories, **not measurements of a real ship**.

**7. H5.** H5 (ORT CPU vs GPU latency) needs Phase 8's ONNX export.
- *User decision (2026-10-01).* Phase 7 gives H5 a verdict line reading **"pending — scored at
  Gate 8"**.
- Gate 7's "H1–H5 scored" is read as H1a, H1b, H2, H3 and H4 scored and H5 explicitly deferred.
  This is a dated wording deviation, not a change to H5.

**8. Carried into every Phase 7 table and verdict.** These are the Phase 7 "Before you start"
items (c) of the plan, with no change:
- the P6-D1 forecast caveats;
- residual methods descend harder than their base, and no residual seed cuts the throttle (P6-D5);
- the two-phase descent, the 50 ms bounce grace, the ~7 % closing-speed understatement, and
  tunnelling counted at any contact substep;
- P6-D5's tunnelling bound;
- tilt-only hard landings.

### P7-D1a — Scoring details fixed before any Phase 7 number is read (2026-10-02)

*Status.* The Phase 7 flights (matrix, cg, sinusoid, λ, noise) started at **08:32:37 EDT** from
`6b83e5c`, in the background. This entry is committed while they run, **before any Phase 7
output has been read**. Nothing is read from `results/e07/` until the flights have finished,
and nothing from the run log beyond its arm start/done lines. Only the scoring code reads
`results/e07/`. The `eval-auditor` raised the open points at the code hand-back; the user
decided items 1–2 and 12, and the rest are readings of P3-D1. **No threshold or prediction
changes.** P3-D1's block SHA-256 stays `21465588…`.

1. **H2 and H4 verdict mapping (user, 2026-10-02).** This follows H1b's pattern, with the
   pre-registered 10-point magnitude.
   - **Supported:** the point estimate is ≥ 10 points **and** the 95 % CI's lower bound is > 0.
   - **Inconclusive:** the CI's lower bound is > 0, but the point estimate is < 10 points.
   - **Not supported:** every other case.
2. **H3 has no combined verdict (user, 2026-10-02).** This is the P3-D4 #5 precedent for H1.
   - Each part gets its own verdict: `ppo_forecast` vs `ppo` at `id` SS5 and at `id` SS6, and the
     `unseen_vessel` half-rule at each.
   - The `residual_ppo_forecast` vs `residual_ppo` secondary is scored the same way.
   - No conjunction row is written.
3. **Erratum to P7-D1 §2: IQM vs mean.**
   - P7-D1 §2 said a learned method's cell success is the IQM over seeds for every contrast. That
     over-reached. P3-D1 §4 fixes the H1–H4 method contrasts as a **paired bootstrap on per-episode
     differences**, a mean over the resampled seeds × episodes.
   - Only H2 names IQM explicitly ("IQM success").
   - So H1a's non-inferiority, H1b and H4 use the §4 mean-based paired bootstrap, and H2 uses IQM.
   - The IQM-based value of H1a, H1b and H4 is printed beside each as "post hoc, not scored".
   - Where P3-D1 and P7-D1 disagree, P3-D1 wins.
4. **H2 pairing inside a cell.** `ppo` and `residual_ppo` flew the identical list in each cell, so
   inside a cell they share the resampled episode indices. Between the two cells (`id` SS5 and
   `unseen_seastate` SS6), episodes are resampled independently, as P3-D1 states.
5. **H3's rule.** H3 uses H1a's point-estimate and CI rule on r: ≥ 10 % on the point estimate, and
   the CI of r excludes 0. It has **no** non-inferiority term, because P3-D1 gives H3 none.
6. **"The CI excludes 0".** For a predicted improvement this means the lower bound is > 0. A CI
   wholly below 0 is "not supported".
7. **D0.4's transfer trigger for H4 (P6-D6).** It counts as fired when `ppo_sinusoid`'s JONSWAP
   `id` SS5 success is ≥ `ppo`'s. Both are already 200/200 in every seed, from e05 and e06.
8. **P3-D4 #8 sensitivity.** Only `timeout` episodes are ranked as the worst closing speed (+∞). A
   `crash` without contact stays excluded. This is a sensitivity analysis and is never scored.
9. **p95 estimator.** NumPy `linear`, the estimator behind every committed p95 since Phase 3.
10. **What λ rescales.** Every component that reads the project scale is rebuilt at the arm's λ:
    - the deck source;
    - the pad lever arm;
    - the ship-motion feed;
    - `gated` / `oracle_gated`'s quiescence thresholds, which are dmf's full-scale thresholds
      Froude-converted (`limits_for(..., spec.scale)`).

    PID gains are model-scale constants and do not change. Learned policies do not change. That is
    the point of the arm.
11. **Bootstrap seed.** Every contrast uses seed 20260926 (P3-D1 §4). Replicate draws are therefore
    correlated across contrasts. This is stated, not corrected. No multiplicity correction is
    applied (P3-D4 #9).
12. **Storage (user, 2026-10-02).**
    - Episode-level CSVs are committed **gzipped, one file per condition**, as
      `results/e07/<arm>[/<condition>]/episodes.csv.gz`, about 100 MB in all.
    - The summaries, contrasts and `results.md` are re-derivable from a clone.
    - The SHA-256 of each uncompressed `episodes.csv` is recorded beside it.

### P7-D2 — Phase 7 flights: what was flown, provenance, and the byte-identity record (2026-10-02)

*Status.* Every Phase 7 flight is done and committed at `05361a5`. This entry records what was flown
and how it can be checked. It scores nothing (P7-D3 does), and it changes no threshold, list or
criterion. The P3-D1 block SHA-256 stays `21465588…`.

**1. What was flown.** All arms: 24 workers, chunk 25, OMP/MKL = 1, host with 36 logical CPUs.
Every flown cell has N = 200 listed episodes, and every learned cell has all 5 seeds (0–4).

| arm | conditions | lists (pad) | flown | carried line for line | episodes flown | flight wall |
|---|---|---|---|---|---|---|
| `matrix` | JONSWAP, λ = 1/25 | `id` SS3–SS6, `unseen_seastate` SS6, `unseen_heading` SS3–SS6, `unseen_vessel` SS3–SS6, `static` (aft) | 6 learned × 5 seeds | 5 baselines from `results/e01`, `pid_feedforward_lowvz_cut` from `results/e01_lowvz_cut` | 82 000 | 1 791.0 s |
| `cg` | pad overridden to CG | the 13 non-static cells | 6 learned × 5 seeds + `pid_feedforward_lowvz_cut` | 5 baselines from `results/e02` | 80 600 | 1 908.0 s |
| `sinusoid` | sinusoid test motion (P7-D1 §1) | `id` SS3–SS6 (aft) | 6 learned × 5 seeds + 6 baselines | – | 28 800 | 540.2 s |
| `lambda` | `lam15`, `lam40` | the 13 non-static cells (aft) | `ppo` × 5 seeds + `pid_feedforward`, `pid_track_descend`, `oracle_gated` | – | 20 800 each | 481.1 s, 482.2 s |
| `noise` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | 11 non-clean (σ_p, latency) conditions (P7-D1 §4) | `id` SS3–SS6 (aft) | 6 learned × 5 seeds + 6 baselines | clean column = `matrix` rows | 28 800 each, 316 800 in all | 612.4–792.7 s each |
| `mss` | `mss_transfer`, `mss_transfer_corpus` | `results/episodes_mss/` (aft and CG) | 6 learned × 5 seeds + 6 baselines | – | 28 800 | 3 409.0 s |

- **578 600 episodes flown** in all, plus the carried baseline rows.
- **Skips, with their reasons.** The only cells not run are `ppo_forecast` and
  `residual_ppo_forecast` × 5 seeds on the `static` list, 2 000 episodes. The recorded reason,
  verbatim from `results/e07/matrix/run_info.json`: "the static-pad list has no ship motion to feed a
  forecaster (StaticDeckMotion has no vessel or ship channels, and its t0 goes down to 0.51 s
  model, inside the 4.0 s model lookback)". They are printed as "not run" in every table. No seed,
  method or sea state was dropped.
- **Not flown by design** (P7-D1): the `static` list in the `cg` and `lambda` arms (its pad is the
  CG, and it has no motion to scale); regimes other than `id` in the `sinusoid` and `noise` arms;
  methods other than the four named in P7-D1 §3 in the `lambda` arm.
- **λ arm start offsets.** t0 differs from the listed value in 20 800 of 20 800 rows at each λ
  (`lambda_t0` in each `run_info.json`), as P7-D1 §3 states. The realization, episode seed, initial
  position and pad are the listed ones.
- **λ feasibility context** is in `results/e07/lambda/feasibility.csv` (all three λ pass P1-D1's
  rule; v_z p99 0.745 / 0.577 / 0.456 m/s model at 1/15 / 1/25 / 1/40 against 2.083 m/s).

**2. Provenance.** Times are EDT on 2026-10-02, from `artifacts/e07_eval.log`,
`artifacts/e07_mss.log` and the `run_info.json` files.

| arm | flown | code tree | `git_sha` in the condition's `run_info.json` | `git_dirty` |
|---|---|---|---|---|
| `matrix` | 08:32:37–09:05:35 | `6b83e5c` | `6b83e5c` | false |
| `cg` | 09:05:35–09:40:06 | `6b83e5c` | `fba9e51` | true |
| `sinusoid` | 09:40:06–09:50:02 | `6b83e5c` | `fba9e51` | true |
| `lambda` | 09:50:02–10:08:41 | `6b83e5c` | `fba9e51` | true |
| `noise` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | 10:08:41–12:25:23 | `6b83e5c` | `fba9e51` | true |
| `mss` | 12:26:18–13:25 | `3582735` | `3582735` | true |

- **Why four arms record `fba9e51`.** `artifacts/e07_run.sh` launched the five arms one process
  each, from `6b83e5c` (committed 08:32:20). `fba9e51` (P7-D1a, committed 08:35:41 while the matrix
  flew) touches `docs/protocol.md` only: `git diff 6b83e5c fba9e51 -- src scripts configs Makefile`
  is empty. So all five arms flew the `6b83e5c` code tree. The top-level
  `results/e07/run_info.json` lists `fba9e51` for the matrix too, because that log reads git state
  when an invocation ends.
- **Why HEAD stayed at `fba9e51`.** `7e15b35` and `3582735` were committed on the worktree branch
  at 09:49:09 and 09:49:26, and fast-forwarded into `phase-7-eval` at 12:25:49 (reflog). That was
  after the first five arms and before the MSS flight.
- **`git_dirty`.** Wherever it is true, the only dirty paths are the untracked `.claude/worktrees/`
  and `results/e07/`, the output directory. The matrix condition recorded false because its state
  was read before `results/e07/` existed.
- **After the flights** (all at `3582735`):
  - `eval_phase7.py --arm all --compress`, 13:25:35, ending at about 13:27;
  - `--arm hypotheses`, 13:36:18 (81.6 s);
  - then `scripts/report.py` (`results/results.md`, 2 297 lines);
  - commit `05361a5` at 13:52:34.

  The compression ran at 13:25, not 13:37; only the scoring and the report came at about 13:37.
- **Checkpoints.** For each flown run, `model.zip`, `vecnormalize.pkl`, `checkpoint.json` and
  `config.yaml` were hashed. That is 30 runs, or the 5 `ppo` runs in the λ arm.
  - They were hashed before and after every condition, and at the start and end of every
    invocation. The digests are unchanged in all 17 conditions and all 6 flight invocations
    (`checkpoints_unchanged: true`).
  - The digests in every e07 `summary.csv` (`run_model_sha256`, `run_vecnormalize_sha256`,
    `run_checkpoint_json_sha256`, `run_config_sha256`) are one value per (method, seed) across all
    arms.
  - They equal the digests committed in `results/e05/summary.csv` and `results/e06/summary.csv`
    for all 30 (method, seed) pairs.
- **Frozen lists.** Each non-MSS condition's `run_info.json` records MANIFEST `e6f30e55…`,
  "10/10 hashes OK", and each list's file and content hash. The MSS condition records the MSS
  MANIFEST below, "4/4 hashes OK".
- **Reference checks.** The matrix's `id` aft rows of the learned methods reproduce the committed
  Phase 5/6 rows byte for byte:
  - `results/e05/episodes.csv` (`ppo`, `sac`): 8 000 of 8 000;
  - `results/e06/episodes.csv` (four Phase 6 methods): 16 000 of 16 000.

  So P5-D14's and P6-D5's audits cover these rows unchanged. Each carried baseline file is
  line-for-line identical to its source, with provenance equal to the live configs
  (`carried` in `matrix/` and `cg/run_info.json`).

**3. MSS transfer lists** (P7-D1 §6), recorded the way P3-D1 §2 records the frozen ones.
- *When committed.* At `3582735`, on their own commit (09:49:26 on the worktree branch; on
  `phase-7-eval` at 12:25:49). That was before the first MSS flight at 12:26:18.
- *How drawn.* Generator seed **20261001**, `balanced_round_robin` over the 18 realizations of each
  grid kind (headings {180°, 135°} × speeds {0, 6, 12} kn × seeds {0, 1, 2}). Aft pad, N = 200 per
  list, 100 at each heading. `frac_in_training_distribution` is 0.0: S175 is never trained on.
- *Authority.* `results/episodes_mss/MANIFEST.csv`, SHA-256
  `49468e1997fda585900309b60d48fd4e8697105985920ff8a8be3a1a65721ba0`.
  `scripts/make_episodes.py --mss --check` re-verifies both hashes of each list.

| file | cell | rows | file SHA-256 | content SHA-256 |
|---|---|---|---|---|
| mss_transfer.parquet | SS5/mss:mss | 200 | `da02270e47f500bb1cb358027e4ef7c6ffd6594184f63b3f3e962c5cb2c97d66` | `a15ddc341eb889ed68605db8d3a367d5f461a691df17e30a0864568faadd4dbc` |
| mss_transfer_corpus.parquet | SS5/mss:corpus | 200 | `ec440f176ea3c75b6112e95495256e8a4dae943ed655c097f73f3e7235a260e6` | `e97acc761cfac166012b12f502248e8438c17ba73d1b39c786bde6921c0fd7cc` |

- **Spectrum match** (`results/mss/spectrum_match.csv`, SHA-256
  `88f36b5d29d2c46bbd28a1b1453acd19cf800da98b95d52562b1208379dc864c`, 18 realizations per grid kind).
  The target is Hs 3.3 m, Tz 7.547 s.
  - Mean Hs relative error: −1.84 % (± 4.34 % SD) for `mss`, +0.39 % (± 4.47 %) for `corpus`.
  - Mean Tz relative error: +3.45 % (± 3.76 %) for `mss`, +3.22 % (± 4.21 %) for `corpus`.
  - All four are within the 5 % predicate of the `mss-export` target.
- **Octave parity** (`results/mss/octave_parity.csv`, SHA-256
  `8f85a6dd609852a1a54b94644554a29f1f6630a838a9fbdb00d964b7aa1966d4`; Octave 8.4.0).
  - *Rows.* 73: one patch-equivalence row, then four (heading, speed) cells: 180°/0 kn, 180°/12 kn,
    135°/6 kn and 135°/12 kn.
  - *Per cell.* NumPy vs Octave on η and η̇ for all 6 DOF, and `rld` vs Octave on heave, roll,
    pitch and their rates.
  - *Result.* 0 failed. The worst relative deviation is 4.4 × 10⁻¹² against a tolerance of 10⁻².
- These are another simulator's strip-theory trajectories, **not measurements of a real ship**.

**4. Byte-identity checks** (`artifacts/e07_checks.log`; all exit 0; run at `3582735` before the
commit).
1. `scripts/report.py --check`: `results/results.md` is byte-identical.
2. A second `scripts/report.py` render, compared with `cmp` against a copy of the first: identical.
3. `scripts/eval_phase7.py --arm all --check`: all 17 conditions OK. For each condition:
   - the `.gz` is present and equals both `episodes.csv.sha256` and `run_info.json`;
   - `summary`, `seeds`, `aggregate`, `baselines_summary` and the carried files re-derive
     byte-identically from the episode rows;
   - and `lambda/feasibility.csv`, `contrasts.csv` and `hypotheses.csv` re-derive identically.

   The re-derivation uses `3582735`'s code. Five of the six arms were flown on `6b83e5c`'s. So
   `7e15b35`'s eval-side changes did not change a derived file.
4. `scripts/eval_learned.py --check` for `results/e05` (4/4) and `results/e06` (6/6).
5. **The 7-worker re-flight** (`--arm sinusoid --refly sinusoid --refly-workers 7
   --refly-per-cell 50`).
   - *What it flew.* The first 50 episodes of each of the 4 cells for all 36 runs and controllers:
     7 200 rows, at 7 workers, against the committed rows flown at 24 workers.
   - *Result.* The header is identical, and **7 200 of 7 200 rows are byte-identical**.
   - *Where recorded.* `results/e07/sinusoid/run_info.json` → `worker_count_checks` (17:52:10 UTC,
     315.9 s, git `3582735`).
   - *It also spans a code change.* The arm flew on `6b83e5c`'s tree.
   - **This closes the P3-D4 carry item** ("Phase 7's `make eval` records the second worker count
     in `run_info.json`"): a second worker count is now recorded in a committed artifact.
   - *Its scope.* One arm, a 25 % subset. It was run by hand: the Makefile `eval` target does not
     call `--refly`.

**5. Key artifacts (SHA-256).** For episode files, the authority is the SHA-256 of the
**uncompressed** `episodes.csv` (P7-D1a #12). The `.gz` hash is listed for convenience; the bytes
are deterministic for a given zlib.

| file | SHA-256 |
|---|---|
| `results/e07/hypotheses.csv` (17 rows) | `d5a82daafa0643f7d7ae9b55ed5d5a461a95fc1be7587828bc0aa7758346b0e5` |
| `results/e07/contrasts.csv` (861 rows) | `1d5f1ce0e7af0f850fc3c867db8214356ed4681506fc07c98545e6eac8319041` |
| `results/results.md` (2 297 lines) | `7fffab4d3b69d9177bdb16431c77c1136208865b410a33088bcde80eed77fe51` |
| `results/e07/lambda/feasibility.csv` | `49f480e7c7e8208cda1bdb989a5b633d34fd6f1dba23cf76c35da4390757ad5f` |

| condition | `summary.csv` | `aggregate.csv` | `episodes.csv` (uncompressed) | `episodes.csv.gz` |
|---|---|---|---|---|
| `matrix` | `1216bcd678635a5f93317b0f4701e837388154183343190b6bec9b367adb91a8` | `a97acead2561d41a65fdbef916e8531907fe07f821d52e52ff7a20b230b462f1` | `6d14b05af2dc195ed281fc3d5f454d923c6227b17d6cbfcf93697cab6ff11152` | `1ab1fc844e1fd7dcb65f7aa8eefa35fadd77b40dfed89ab7b534d41f407fa280` |
| `cg` | `383dd7df90a846b44cb16669cfc7b3927fc2416c9c6bf1bed9a8720623cdf454` | `3a6e1afed3ec70c2eb5648a3bfe6816121f527c01635e8413d1916c4f74116c2` | `e1df07c27217a53db0dfbcea24d158bc3412bda75c7f1e9a857bd0cb64583335` | `0b63bbbbf4eaa3207f2096cc51ca7600a84dca477ad4a7f54b60151fb736a5d0` |
| `sinusoid` | `9b7b683f2cd28a0d5f6c8dd23da9f1715bb149b07ede12aa0844a80d9cbcceca` | `d2442b0ffe47eb9405c2209c3c88d7d34d087a99446a82e0bc40f4a04cccb753` | `0b7b484eefeabc48adec267341175cf37dfea318fcb3e274b3f688ccd270a47e` | `c89725863df63836fffa4f5125e8488d2cab7f24ad3e136285ed9115c4f53b80` |
| `lambda/lam15` | `408cc19f2124de540e43a2955da97abe3cd730d69a478d0b67b019a2bc71daeb` | `db7e91da333767982ee81c9bc3f27e23619fbb64bb0ee8a9e62ce5a342e2cf2e` | `15206f5bb38a6b0e443bbcbf65df5b52a72266f8b37f7ab50a18743f67ea2152` | `813712cb5ff80064fc5a71df6d4e325ed91fd087384bf03b76477f255587a1cb` |
| `lambda/lam40` | `87f4c480424e703c4df7e0ccc45153b275b43c1221f7ad06ff71ca18996dbf17` | `3c6ebe2f409289c4d2312bbf0104140529e9a7fe387497f907fc35036fb3acc1` | `c35900f86920c7a70d53ba1c769f13dafaf2c3d6621969511b517c07b903c44a` | `0a6d25d556f4b19d5f1b37f2d714e967346287312d482ff508a2f491149d537e` |
| `noise/sigma1cm_lat0step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `d2edf45870e5124ee1cdeee9f9bc73169d54bda635c294d025584654d77fc56a` | `2c0063288b328c6b3243f073cf846f4e4594d64033086ea8c186f02fe24c5d68` | `9dcf6941d9222f2f99d70e6820b21410f0bc69f29c560f2562bbced45ef3da1f` | `03d091bfca35960639755de16edd41c2d2928b4b45b22facba0b748b09411c58` |
| `noise/sigma2cm_lat0step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `845704cdfea07b9333f7e4fb2dc38753e639e03d77f1665bc278dbdf2ddae7c3` | `9544d4296da55abeeab35292a425fb1114f81111ef6e02bcb2afb5ac15ea1f7d` | `08faedbfb979774c008d242cc6c7a7169a99df5d963d287bed8c14c29b0dd1e9` | `0de28870959ae8a042ef625b5ceb96a5b54b847d30e515810f6bdc674edf306f` |
| `noise/sigma4cm_lat0step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `48e4c707191c71ead6b0fc67bbc3ff5da7067b6758eb8e6eb4e2d092c4dd2394` | `f65076c685c3919722850561baa82ec7303d04e980c6dbc767591824fd4758a2` | `03a2b24849316097d62c5df23edbfd464d463b21ffb527583f4dd8e01a90003f` | `607f14d600ace4ecd5eb7805bb08229f681aa36382e7905827a8b4e65f82e251` |
| `noise/sigma0cm_lat1step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `8da8a13c439cba44dbe2adf2403a80fcea999ab55f76a7f4d4d5022a073eb8e2` | `90339ed2e85763ecda7f6019e295882e5b032e04104b25bf267131fbf249db0f` | `fcefd562858d8e5ba3badfce52136863ef5061ed540223eea8552b90b62bc2e1` | `16c5b2efdb2bb4173f2bbf3d8676895afd0cf5816e8fa00bf0660b0176a55fb6` |
| `noise/sigma1cm_lat1step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `da80f43784668c107631a03cf724e2ca02f86ec338b3a525712da3c582b797e2` | `025221b79f3b322960e933167143e3a659b4f740d96036c377580c70a32dc902` | `c1b9d831243114cc38810d5662f83bf69e387be9a2b1cb3ba232c2e6590d1fdb` | `4e9bad2d100a00acf668681bbc4e5e54d1226afa0cb14c85a4e9776b23e623aa` |
| `noise/sigma2cm_lat1step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `6ad4b167f1edbad579e6d26214394b45e2a3cf6d560956001381aa8d24dfd9b5` | `4d91e6b9244c5f67c19378b03ae5ae11fd56c1fc69bb8399decb3d2b74fb4b72` | `deeecd1564f0b1f5afd7fb9baa74603dbd512b305316820b2499dd826a443c96` | `f102d2293dfda7da80c899fba7becaaf18d23f6fe3518c9223c7f780bba107ed` |
| `noise/sigma4cm_lat1step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `77a7395f1a084abc06ebf5e452240fd8fa2f10af5a56ffe775dddcf8d1f0a906` | `06404f4965007afdb189d0fb362130a33d9c77fa4c1eacec96c8bdafd23baa49` | `efe71388722b018db48114c62025df3ca680dcd5f8377d40923172c7007f1158` | `7a5cd216622e614881220fb095443f9bc853c16691f44e151ae6a1592119cff6` |
| `noise/sigma0cm_lat2step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `8d8b11adff40515de64bc8b789ccf30bf1733ea5feca110235c7e575e2952682` | `33532960b5d862651a1ef45d1537fbc1a57cb62dfd3520151b5d08c1c921cba7` | `70fb5eb6427165d436f555543ae1e920329cf92c81e9cd07b13275279d504f38` | `0bfbe0ac215581556efe9b322fcf445d4ad1617762b09d6c79f6fe3d926a904d` |
| `noise/sigma1cm_lat2step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `d48f57050defabf19594f2fb4c87450db672ec600f1f82956a506484e2f1cf79` | `75e6a663cd5e92bec3f435ba2b74cbcfee3533f2970e1142be32f6eb0d3ae391` | `57106556cbe21a4e5002f3d8ed57321f8dd24438a295e0158dc1366652519f17` | `dd4c951bf3abb320e170b06b0c36e994d181637c291671e27cbc93ea124a68f7` |
| `noise/sigma2cm_lat2step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `84c32e9c20f94682a1b01455d8112c7875e44f0305fb8c4a0a56ba923b0ac458` | `cfd9a7ebe8536e9f5d95d6408e9afbd015302618cd0386bd4d18d08f22615ba8` | `136d251f7894afe565d1b56e8ef1ac9c6af3e82f5b72a8ae52668e0d732287c3` | `eae89ce387d2371e88143bea45cca1efe72544bbd274e0fb09ad0906718e8f0f` |
| `noise/sigma4cm_lat2step` (superseded by P7-D4; now `noise_superseded_p7d1/`; current hashes P7-D5 §3) | `2ee64db7e7cdbe5118e4fbe3c5b9d3b0ddd4b4b397ab96e025f9f59bbe37461f` | `e6bf7911b85f5e8b10f1db1e590ef0d3f6cc2955badc6a0ef7c794449471923c` | `42b114336793ac08e3e044d4226fefe763ec21029e4e9afbd7d34d1b72d89b8d` | `245d498584ba74e638a1db943b50d63d395a278f5f6cf2a3f526ca7e056178e6` |
| `mss` | `b76c4f487eeebcdc1fa8f1663ba377cb3e0d1f6f9d13facaf0e8e0fc746d46ff` | `3b5f7ff931daf65313f6cb8f89c1d32697d0d620d52c763a81f1fc11d88d3963` | `246e8688690e83dc07ef2fe13c399ba4e923a61315686e9a5802690bdd3fced5` | `e09f3a727029fc5b3609486d555aa2c7dad506b650a00d1b31782e14dd0467a8` |

`run_info.json` files hold the non-deterministic facts (times, workers, digests, the worker-count
check) and are not hashed here. `seeds.csv`, `baselines_summary.csv` and the carried files are
covered by the `--check` above and by `files_sha256` in each `run_info.json`.

### P7-D3 — Hypothesis scoring (2026-10-02)

*Status.* `rld.eval.hypotheses` scored the hypotheses at `3582735` (13:36 EDT). It read the
committed e07 episode rows and, for the baselines, `results/e01/episodes.csv`. The verdicts are
written in `results/e07/hypotheses.csv` (`d5a82daa…`), and every contrast behind them is in
`results/e07/contrasts.csv` (`1d5f1ce0…`).
- This entry copies those verdicts. **Nothing is re-scored here.**
- *Rules:* P3-D1 §8, P3-D4 #5–#9, P6-D6, P7-D1 §2 and P7-D1a.
- *Bootstrap:* 10 000 replicates, seed 20260926, percentile 95 % CI.
- *Units:* success in points, r in %.

**1. Verdicts, one line per scored part, as in `hypotheses.csv`.**

| H | part | cell (aft) | statistic | point [95 % CI] | prediction | verdict | rule applied |
|---|---|---|---|---|---|---|---|
| H1a | relative p95 + non-inferiority | `id` SS5 | r = 1 − p95(`residual_ppo`) / p95(`pid_feedforward_lowvz`); NI: mean paired success difference | r = **−47.0 %** [−69.6, −36.8]; p95 0.2758 vs 0.1877 m/s. NI: +4.1 [+1.2, +7.1], lower bound +1.2 ≥ −2 | r ≥ 15 %, CI of r excludes 0, NI holds | **not supported** | P3-D4 #5: r < 0 (residual lands harder) is "not supported" |
| H1b | success difference | `id` SS6 | mean paired success(`residual_ppo`) − success(`pid_feedforward`) | **+6.0** [+2.0, +10.2]; 965/1 000 vs 181/200 | ≥ +5 points, CI excludes 0 | **supported** | P3-D1 §8 H1b; P7-D1a #3, #6: point ≥ 5 and lower bound > 0 |
| H2 | drop difference | `id` SS5 → `unseen_seastate` SS6 | drop(`ppo`) − drop(`residual_ppo`), drop = IQM SS5 − IQM SS6 | **−1.5** [−4.5, +1.7]; drop(`ppo`) +2.7, drop(`residual_ppo`) +4.2 | ≥ +10 points | **not supported** | P7-D1a #1: lower bound ≤ 0 → "every other case" |
| H3 | primary: `id` SS5 | `id` SS5 | r = 1 − p95(`ppo_forecast`) / p95(`ppo`) | **−0.4 %** [−2.3, +2.3]; 0.2729 vs 0.2720 m/s | r ≥ 10 %, CI excludes 0 | **not supported** | P7-D1a #5, #6: CI includes 0 |
| H3 | primary: `id` SS6 | `id` SS6 | same | **−0.4 %** [−6.3, +3.3]; 0.2826 vs 0.2814 m/s | r ≥ 10 % | **not supported** | CI includes 0 |
| H3 | primary: `unseen_vessel` SS5 half-rule | `unseen_vessel` vs `id` SS5 | r(`unseen_vessel`) ≤ 0.5 · r(`id`), point estimates | r(uv) −0.7 % [−2.3, +1.4]; r(id) −0.4 % | half-rule | **not applicable — no id gain to shrink** | P7-D1 §2: r(`id`) ≤ 0 |
| H3 | primary: `unseen_vessel` SS6 half-rule | `unseen_vessel` vs `id` SS6 | same | r(uv) −0.4 % [−3.3, +3.2]; r(id) −0.4 % | half-rule | **not applicable — no id gain to shrink** | P7-D1 §2: r(`id`) ≤ 0 |
| H3 | secondary: `id` SS5 | `id` SS5 | r = 1 − p95(`residual_ppo_forecast`) / p95(`residual_ppo`) | **−0.8 %** [−2.4, +0.9]; 0.2781 vs 0.2758 m/s | r ≥ 10 % | **not supported** | CI includes 0 |
| H3 | secondary: `id` SS6 | `id` SS6 | same | **+1.9 %** [+0.4, +3.8]; 0.2785 vs 0.2838 m/s | r ≥ 10 % | **inconclusive** | 0 < r < 10 %, lower bound > 0 |
| H3 | secondary: `unseen_vessel` SS5 half-rule | `unseen_vessel` vs `id` SS5 | same | r(uv) −2.0 % [−3.6, −0.3]; r(id) −0.8 % | half-rule | **not applicable — no id gain to shrink** | P7-D1 §2: r(`id`) ≤ 0 |
| H3 | secondary: `unseen_vessel` SS6 half-rule | `unseen_vessel` vs `id` SS6 | same | r(uv) +0.04 % [−1.9, +2.6] ≤ 0.5 × r(id) = 0.5 × 1.88 % = 0.94 % | half-rule | **holds** (relabelled "not scored (no supported id gain)" in P7-D5) | P7-D1 §2, judged on point estimates |
| H4 | drop difference | `id` SS5 | drop_sin − drop_jon, mean paired success, one shared episode set | **+0.0** [+0.0, +0.0]; all four legs 1 000/1 000 | ≥ +10 points | **not supported** | P7-D1a #1: lower bound 0 is not > 0 |
| H5 | ORT CPU vs GPU p50 latency | batch 1 | — | — | ≥ 2× | **pending — scored at Gate 8** | P7-D1 §7 |

No combined H1 or H3 verdict exists (P3-D4 #5, P7-D1a #2).

**2. Rows that are not scored.**
- *Post hoc, not scored* (P7-D1a #3): the IQM-over-seeds versions of the mean-based contrasts.
  - H1a non-inferiority: +4.0 [+1.0, +7.0];
  - H1b: +5.7 [+1.8, +10.3];
  - H4: +0.0 [+0.0, +0.0].
- *Sensitivity, not scored* (P3-D4 #8, P7-D1a #8): relative-p95 with timeouts ranked worst.
  - All 9 rows (H1a and the 8 H3 relative-p95 rows) equal their scored counterparts exactly.
  - The reason: no episode in any of these cells timed out. `n_timeouts` is 0 for both methods in
    every row, and every learned episode touched down (1 000 of 1 000).
  - So the sensitivity analysis is **vacuous** on these cells. It neither supports nor weakens any
    verdict.

**3. H4, per P6-D6: the novelty claim is withdrawn.**
- *The sinusoid leg at `id` SS5.*
  - `ppo_sinusoid` on sinusoid motion: 1 000/1 000. `ppo` on sinusoid motion: 1 000/1 000.
  - On JONSWAP both are 1 000/1 000 (`results/e07/sinusoid/`, `results/e07/matrix/`).
  - So drop_sin = drop_jon = 0. P6-D6's bound s + x − 200 ≤ 0 holds with equality.
- **Both triggers fired, as the H4 note in `hypotheses.csv` records:**
  - *D0.4 / P7-D1a #7 (transfer).* `ppo_sinusoid`'s JONSWAP `id` SS5 success, 1.000, is ≥ `ppo`'s,
    1.000. It also fires under the post-hoc IQM, 1.000 ≥ 1.000.
  - *P3-D1 §8 (CI includes 0).* The CI is [0.0, 0.0]. Its lower bound is 0, not > 0
    (P7-D1a #6), so it does not exclude 0.
- **The CI is degenerate.** All 4 000 seed-episodes in the cell (2 methods × 2 motions × 1 000) are
  successes, so no bootstrap replicate can differ from 0. [0, 0] records a ceiling, not precision.
- *What the README reports instead* (D0.4). The transfer is the finding: a policy trained only on
  matched sinusoids lands as often on JONSWAP `id` SS3–SS5 as the JONSWAP-trained `ppo`, 100 % in
  every seed. At SS6 the two read 97.3 % and 98.2 % (IQM). That SS6 comparison is unpaired between
  methods and not tested.

**4. H5** is "pending — scored at Gate 8" (P7-D1 §7; the user's decision of 2026-10-01).

**5. No multiplicity correction** is applied across H1a, H1b, H2, H3 and H4 (P3-D4 #9).
- All contrasts share bootstrap seed 20260926, so their replicate draws are correlated (P7-D1a #11).
- H1b is the only supported part. Its lower bound, +2.0 points, is reported uncorrected.

**6. Caveats carried beside the verdicts they bear on** (P7-D1 §8).

*H1a (not supported; r ≈ −0.47).*
- **The sign follows from the descent the residual learned (P6-D5, corrected at P6-D6 M1).**
  - *Through the approach,* the residual holds about −0.45 m/s, while the base commands +0.02 to
    −0.15 m/s.
  - *In the last second* its own share falls to −0.06 to −0.07 m/s. That is still on top of a base
    command of −0.19 to −0.21 m/s.
  - *At touchdown.* The median executed vertical setpoint at the last control step before
    touchdown is −0.274 m/s (`results/audit/e06/residual_descent_profile.csv`, step 0).
  - *Against `lowvz`.* `pid_feedforward_lowvz` commands a constant 0.111 m/s descent relative to the
    deck. The residual's final commanded descent is therefore about 2.5× faster.
  - *Closing speed.* Its pooled p95 is 0.276 m/s, close to that final setpoint, against `lowvz`'s
    0.188 m/s. The ratio 0.276 / 0.188 = 1.47 is r = −0.47.
  - *What was within reach.* A `lowvz`-like final descent needed about +0.09 m/s of residual, about
    20 % of the ±0.45 m/s authority. It was not learned, and check 9 found the residual not
    authority-limited. Why it was not learned is not established.
- *Throttle cut and bounces.* No residual seed learned the post-contact throttle cut (P6-D5).
  Its 0 SS5 bounces against `lowvz`'s 9 are not a throttle-cut effect.
- *Non-inferiority.* It holds with room: +4.1 [+1.2, +7.1]. It does not enter the verdict once
  r < 0.
- *H1 is narrowed* (P3-D4 MAJOR-1). H1a asks for "softer than the softest constant-descent PID in
  the success-tuned log". The data answer that the residual lands harder than that PID, and harder
  than `pid_feedforward` too: 0.276 vs 0.262 m/s, an unpaired reading.
- *Measurement.* The ~7 % closing-speed understatement applies to every method alike (P5-D14). It
  does not change the sign of r.

*H1b (supported; +6.0 [+2.0, +10.2], 1.0 point above its line).*
- **Outside the training distribution** (P3-D4 #7). `id` SS6 is never trained on, so H1b is a
  sea-state-extrapolation result.
- **Two-phase descent.**
  - `residual_ppo` reaches touchdown in a median 1.94–2.01 s, against 4.4–4.6 s for
    `pid_feedforward` (P6-D4), with the residual's two-phase descent.
  - The PID tuning space (constant descent 0.08–0.60 m/s) cannot express that descent.
  - H1b therefore shows that this residual policy beats this PID at SS6. It does not show that it
    beats any PID.
- **Loss classes.**
  - `pid_feedforward`'s 19 SS6 losses are 11 `bounce` and 8 `hard_landing`. Its hard landings are
    tilt-only (P6-D5).
  - `residual_ppo`'s 35 per 1 000 are 24 `hard_landing` (tilt-only) and 11 `bounce`.
- **Tunnelling bound** (P6-D5). Up to 3 `residual_ppo` SS6 successes per 1 000 may depend on
  tunnelling overlap (corrected in P7-D5: the raw tunnelled-success count is 4; 3 is the audited
  subset).
  - The e07 rows are byte-identical to the e06 rows the audit covered (16 000/16 000), so the bound
    applies unchanged.
  - `pid_feedforward` has 0 tunnelled episodes in `results/e01`.
- **Could these two caveats move H1b across its 5-point line?** This is post-hoc arithmetic on
  point estimates from the committed counts. No CI is recomputed and nothing is re-scored.
  - *The margin.* 6.0 − 5.0 = **1.0 point**. One `pid_feedforward` episode is 0.5 point (1/200).
    One `residual_ppo` seed-episode is 0.1 point (1/1 000).
  - **Tunnelling alone cannot.** Remove all 3 bounded successes: 962/1 000 − 181/200 = 96.2 − 90.5
    = **+5.7**, still ≥ 5. Crossing would take 10 per 1 000, more than three times the bound
    (corrected in P7-D5: removing all 4 raw tunnelled successes gives +5.6; crossing takes 11).
  - **The bounce-grace rule can.** A bounce has already passed the crash, off-pad and
    hard-landing tests, and only the 0.5 s dwell failed. A rule that kept contact through the gap
    would at most turn it into a success.
    - Say k of `pid_feedforward`'s 11 bounces and j of `residual_ppo`'s 11 became successes. The
      difference would be 6.0 − 0.5k + 0.1j.
    - With j = 0, **k = 3 is enough**: 6.0 − 1.5 = 4.5 < 5.
    - If the same fraction f of each method's bounces converted, the difference is
      6.0 − 4.4f, which falls below 5 for f > 0.23, i.e. about one bounce in four.
    - All 22 converted gives 96.5 + 1.1 − (90.5 + 5.5) = **+1.6**.
    - Only `pid_feedforward`'s 11 converted gives **+0.5**.
  - **Combined.** With the 3 tunnelling-bounded successes removed (+5.7), k = 2 already crosses:
    5.7 − 1.0 = 4.7 (corrected in P7-D5: with all 4 removed, +5.6 and 4.6).
  - *Is such a change plausible?* For `lowvz` on the tune pool, P5-D1 found:
    - 80 % of bounces were rim rocking, with the CoM still closing;
    - a 0.1 s grace would have reclassified 51 of 58 gaps over 50 ms.

    That was measured on `lowvz` on the tune pool, not on `pid_feedforward` at `id` SS6, so it is
    only indicative.

    P5-D1 also found that the 240 Hz `bounce` label is unstable in both directions: converged
    physics roughly doubles the rocking gaps. The direction of a "true" correction is therefore not
    known.
  - **Conclusion.** The bounce-grace caveat **can** move H1b across its 5-point line, under a
    change of the frozen rule that P5-D1 shows is plausible. The tunnelling caveat cannot on its
    own.
    - The verdict stands as scored under the frozen criteria.
    - The README must state that H1b's margin over its threshold (1.0 point; 0.7 for the post-hoc
      IQM) is smaller than the effect of a plausible change to the bounce rule.

*H2 (not supported; −1.5 [−4.5, +1.7]).*
- The point estimate has the opposite sign to the prediction. `residual_ppo` drops 4.2 points and
  `ppo` 2.7, but the CI includes 0.
- The two cells share realizations (37 in the committed lists, P3-D1 §2/P3-D4 #2). The bootstrap
  resamples them independently, as P3-D1 specifies.
- The two-phase-descent and bounce-grace caveats apply to both methods.

*H3 (no primary part supported).*
- **P6-D1 caveats.** The forecaster's forecasts were in-sample during training, and the past-only
  ship-motion feed is an extra ideal sensor. Even with both advantages, the forecast block gave no
  measurable closing-speed gain:
  - primary r = −0.4 % at `id` SS5 and at SS6, both CIs including 0;
  - all four half-rule parts were scored "not applicable" or rest on no supported gain.
- **The secondary SS6 "inconclusive"** (+1.9 % [+0.4, +3.8]) is under a fifth of the predicted
  10 %.
- **The secondary `unseen_vessel` SS6 "holds"** is the scoring code's word for a met half-rule
  (relabelled "not scored (no supported id gain)" in P7-D5).
  - P7-D1 §2 fixes the rule but no verdict word for it, and this is outside the
    supported / not supported / inconclusive vocabulary of P3-D1 §8.
  - It shrinks a gain that is itself only "inconclusive", and r(uv SS6)'s own CI includes 0.
  - It is **not** read as support for H3's "then help less". There is no supported "help" part for
    it to shrink.

*H4 (not supported).* The H4-bounded note of P6-D6 applies, and the CI is degenerate (item 3).

*Every verdict.* These caveats also apply to every verdict:
- no multiplicity correction, and a shared bootstrap seed;
- tunnelling is flagged at any contact substep;
- closing speed is understated by about 7 %;
- in the `id` cells P6-D5 audited, every hard landing of the four Phase 6 methods, `ppo` and
  `pid_feedforward` is tilt-only. Hard landings outside `id` (H2's `unseen_seastate` SS6, H3's
  `unseen_vessel` cells) are not audited;
- simulation only, with λ = 1/25, a 3-DOF deck and dmf's roll/pitch–heave phase defect carried.

### P7-D4 — DEVIATION: the perception stand-in perceives the deck, and the noise arm is re-flown (2026-10-02)

*Status.*
- This is a dated deviation from P7-D1 §4's scope sentence and from the Phase 2 stand-in
  (`rld.envs.noise`, `DeckLandingAviary._computeObs`).
- The user decided it (option (b)) on 2026-10-02, after the Phase 7 `results-skeptic` review.
- It is committed **alone**, before the stand-in code changes and before any re-flight.
- It changes no success criterion, no episode list, no hypothesis and no threshold. P3-D1's block
  SHA-256 stays `21465588…`.
- No hypothesis reads the noise arm, so no verdict in P7-D3 can move.

**Why (review M1, M2).** The Phase 2 stand-in replaced only the six relative entries (relative
position and relative velocity) with a delayed, held, noisy copy. Everything else was computed from
the true deck at the current step: the drone's own velocity, the deck normal, the relative tilt and
the pad-plane clearance.
- *M1, mixed timestamps.* A controller that reconstructs deck velocity as own velocity + relative
  velocity (`rld.control.obs_view`) therefore received v_pad(t−L) + v_drone(t) − v_drone(t−L).
  That is positive feedback on the drone's own acceleration through `pid_feedforward`'s
  feedforward, with k_ff = 1.08.
  - The reviewer's scratch re-flight: with a consistently delayed deck velocity, 20 of 20
    episodes that crash as flown succeed (`pid_feedforward`, 2 steps, `id` SS3 and SS5).
  - So the 2-step-latency crash rates in the committed noise arm measure this inconsistency, not
    perception latency.
- *M2, ideal channels.* Under every noise condition, the clearance, the deck normal and the
  relative tilt stayed true and undelayed.

**The corrected stand-in (definition, fixed before any change):**
1. **What is perceived** is the *deck*: the full analytic deck sample at the pad (pad position,
   velocity, orientation and normal, as `PlatformSample`). The drone's own state is not perceived.
2. **Delay and hold.** At control step k the perceived sample is the true deck sample of step
   k − L.
   - L is the configured latency, quantised **down** to whole control steps. This is unchanged:
     33.4 / 66.7 ms → 1 / 2 steps (P7-D1 §4).
   - Warm-up is unchanged: before L samples exist, the oldest available sample is used.
   - The 30 Hz hold is unchanged (no hold).
3. **Noise.** Zero-mean Gaussian noise, independent per world axis, is added to the perceived pad
   position (σ_p) and pad velocity (σ_v = σ_p / 0.2 s). The sigmas and the stream are unchanged
   (a spawned child of the episode seed). It is now drawn in the **world** frame, not the
   drone-yaw frame. The distribution is the same, because the noise is isotropic in the horizontal
   plane and identical in z, but the realised values differ from the committed arm. Orientation
   and normal get no noise; P7-D1 never specified attitude noise.
4. **Every deck-derived observation entry** is computed from the perceived sample and the drone's
   **current** true state, through the unchanged `build_observation`:
   - relative position;
   - relative velocity;
   - the deck normal in the yaw frame;
   - the relative tilt;
   - the pad-plane clearance.

   The drone's own attitude, rates and velocity, time, last action and contact flag stay clean
   and current. With this, own velocity + relative velocity = v_pad(t−L) + noise: a consistent,
   stale deck estimate.
5. **What does not change.**
   - With noise disabled (training, and every non-noise arm), the observation path is untouched.
   - At L = 0 and σ = 0 with noise enabled, the observation is bit-identical to the clean one.
   - Termination, reward, touchdown detection and every evaluation metric still use the true state.

**Re-flight.**
- All 11 non-clean noise conditions are re-flown on the identical episodes: same lists, t0,
  initial state and noise stream, by the same rule as P7-D1 §4. Every one of them changes: the
  latency conditions through the delayed deck, and the σ-only conditions through the clearance,
  which now carries position noise.
- The committed arm, flown at `6b83e5c` and recorded in P7-D2, is **not deleted**. It moves
  unchanged to `results/e07/noise_superseded_p7d1/` and is labelled superseded wherever it is
  shown. Its SHA-256s stay in P7-D2.
- The new arm writes `results/e07/noise/`, and the scored hypotheses do not change.

**Also folded in from the same review, as text corrections with no re-flight:**
- M3: the `below_deck` bail-out is a fixed model-scale constant and pre-empts completed dwells.
- M4: per-cell tunnelled-success counts.
- MINOR m1–m8.

These are recorded in P7-D5 after they are made.

### P7-D5 — Review fold-in: the re-flown perception arm, superseding hashes, and text corrections (2026-10-02)

*Status.*
- This entry records what P7-D4 announced: the stand-in change, the re-flight, the review's
  text corrections, and the hashes that now stand in place of P7-D2's for the files that changed.
- It changes no success criterion, no episode list, no hypothesis, no threshold and **no
  verdict**. P3-D1's block SHA-256 stays `21465588…`. MANIFEST `e6f30e55…` is untouched.
- P7-D3's text is not edited in place. Its errata carry "(corrected in P7-D5)" or "(relabelled …
  in P7-D5)" markers, and the corrections are recorded here (item 8).

**1. What changed, by commit.** Times are EDT on 2026-10-02.

| commit | time | what |
|---|---|---|
| `3a25620` | 15:38:32 | P7-D4, committed alone, before any code change or re-flight. |
| `8ebfd94` | 16:45:18 | The P7-D4 stand-in (`sim-env-engineer`): `PerceptionNoise.perceive(PlatformSample)` in `src/rld/envs/noise.py`, `DeckLandingAviary._computeObs`, `observation.py`, `config.py`; `tests/test_noise.py` (19 tests). The same commit carries the move of `results/e07/noise/` to `results/e07/noise_superseded_p7d1/` (77 renames, bytes unchanged). `182cdea`'s message also names the move, but the renames are in `8ebfd94`. |
| `182cdea` | 16:45:18 | Eval-side review fixes. These are `tunnelled_success.csv` per condition (M4) and the m6 relabel in `hypotheses.csv` (item 5). `make eval`'s verify-only path is read-only and needs no checkpoint. The baselines appear in every appendix table, and the `unseen_vessel` subset table has Wilson CIs and a breakdown. The noise flight gets a preflight guard, and `rld.eval.superseded` hash-checks the superseded arm. `contrasts.csv` temporarily lost its 528 noise rows (333 rows), and `results.md` was re-rendered. |
| `91f3447` | 20:01:56 | The re-flown noise arm. `contrasts.csv` is back to 861 rows, `hypotheses.csv` is byte-unchanged from `182cdea`, and `results.md` has 2 772 lines. |
| this entry's commit | — | This entry, the `docs/findings.md` Phase 7 rewrite of the affected passages, and docstring-only fixes of the stale P7-D1 §4 scope wording in `src/rld/eval/envs.py` (`with_noise`) and `src/rld/eval/arms.py` (`NoiseSetting`, `_SIGMAS`). No derived file changes with them. |

**2. Re-flight provenance** (`artifacts/e07_noise_p7d4.log`, `artifacts/e07_noise_post.log`, each
condition's `run_info.json`).
- *What.* All 11 non-clean conditions, every method: 6 learned × 5 seeds and the 6 baselines. That
  is 28 800 episodes per condition and 316 800 in all, at 24 workers, chunk 25, OMP/MKL = 1, on a
  host with 36 logical CPUs. There were 0 skipped episodes.
- *When.* From 16:45:42 (`sigma1cm_lat0step` start, 20:45:42 UTC) to 19:11:32 (last condition
  written, 23:11:32 UTC). Condition walls were 688.4–792.2 s. The log ends `=== NOISE EXIT 0`.
- *Code tree.* `git_sha` is `182cdea` in all 11 `run_info.json`.
  - `git_dirty` is true, but the only dirty paths are the untracked `.claude/worktrees/` and the
    output directory `results/e07/noise/`. The first condition recorded only the former.
  - **Preflight guard** (`_noise_preflight`, `src/rld/eval/phase7.py`). It refuses to fly the noise
    arm if `PerceptionNoise` has no `perceive`, i.e. under the superseded stand-in. When writing
    into `results/`, it also refuses if `git status` shows any change under `src`, `configs`,
    `scripts`, `Makefile` or `pyproject.toml`. The flight passed it, so the tree it flew is
    `182cdea`'s.
- *Identical episodes.* Every condition records MANIFEST `e6f30e55…` with "10/10 hashes OK". Under
  the strict start check, each reset reproduced its listed t0 and initial state. Each condition
  re-derived its `summary`, `seeds`, `aggregate`, `baselines_summary` and `tunnelled_success`
  byte-identically from its rows (`rederived_byte_identical`).
- *Checkpoints unchanged.* In all 11 conditions, `checkpoint_digests_before` equals
  `checkpoint_digests_after` (30 runs), and the digests are the same across conditions.
  - The `run_model_sha256`, `run_vecnormalize_sha256`, `run_checkpoint_json_sha256` and
    `run_config_sha256` in every re-flown `summary.csv` equal the matrix's for all 30
    (method, seed) pairs.
  - P7-D2 §2 recorded that those equal the committed e05/e06 digests.
- *After the flight* (`artifacts/e07_noise_post.log`, ended 19:28; every step exit 0):
  - `--compress` (11 `.gz`);
  - `--arm hypotheses`, which left `hypotheses.csv` byte-unchanged from `182cdea`
    ("RC hyp-unchanged 0");
  - `scripts/report.py` (2 772 lines), a second render, and `scripts/report.py --check`
    (byte-identical);
  - `eval_phase7.py --arm all --check`: all 17 conditions OK, the 11 superseded conditions OK
    against P7-D2's hashes, and `lambda/feasibility`, `contrasts.csv` and `hypotheses.csv` OK;
  - `eval_learned.py --check` for `results/e05` (4/4) and `results/e06` (6/6).

**3. SHA-256s that supersede P7-D2 §5's.** For episode files the authority stays the SHA-256 of
the **uncompressed** `episodes.csv` (P7-D1a #12). Every other P7-D2 §5 hash (matrix, cg,
sinusoid, λ, MSS, feasibility) is unchanged.

| file | P7-D2 §5 | now | changed at |
|---|---|---|---|
| `results/e07/hypotheses.csv` (17 rows) | `d5a82daa…` | `27b4f6cea22893ffda2b24b088bffd02a299e2bb747909d7de93272e515c7f7d` | `182cdea` (m6 relabel), unchanged by `91f3447` |
| `results/e07/contrasts.csv` (861 rows) | `1d5f1ce0…` | `d373299e6bd4dab515cef058544700acffc66629eee20138caf7dbd06c74608c` | `91f3447` (the 528 noise rows re-derived from the new arm; 333 rows at `182cdea`) |
| `results/results.md` (2 772 lines) | `7fffab4d…` (2 297 lines) | `7f737727f3f43813ab60cf8282c5d296bbb52a281f7a64b298a7b5436ff678e7` | `91f3447` (`11e32e5e…` at `182cdea`) |

The re-flown noise arm (`results/e07/noise/`):

| condition | `summary.csv` | `aggregate.csv` | `episodes.csv` (uncompressed) | `episodes.csv.gz` | `tunnelled_success.csv` |
|---|---|---|---|---|---|
| `noise/sigma1cm_lat0step` | `0080b3e740b6f0cbb5a1c78537b6a6190a786512348c1d844f2e037381814fc5` | `1cac238a3cb9ed6125c263a1c20923a588ef8d611858fd34c726ac5847c12e4f` | `e10f6acb5c4a00b1055004595f1e63910ae34eb8a80c42850e997cdad1b16eb9` | `89cfe34ca4cb76a21dd7e53acf6a09c0202e614bcf7e531730cda3315cee5aee` | `64e9a3fcead830d27926e6609cc45cb022c3016500de5f840e6731198723f0d5` |
| `noise/sigma2cm_lat0step` | `719f4774902df7846c15311246da0a103873f8d92b5db17a6fd0a4e26c986767` | `e77a453bed522001d425e18ff6f505ff11b4ffc33872963ef22dfae3f47cdd6a` | `245d71fdf391c376ac4dbd0035c7add60ad4a6e9c91b52f7d0971707372aeabe` | `b7b7d994f3adc6d0e251d525b6bd2b455c27b4f0de03ab3738169bbf07df6296` | `5955b8bc175f00e47b120dd081d69ede5087741f02ea41727ac157adbe245448` |
| `noise/sigma4cm_lat0step` | `60dd25ee3dae5ae81b23f99a54e9363fe2d03614cc70c1aec6ac1a66e01ba952` | `b3ecf2556d5f53b9cefeecd5da3cc90c3bf9c015960b2774844eee6dd7058527` | `2c5b192535625fb5b49d3e1c100235751e6f1cf68daafc323f8f4104cbba47b5` | `4bf5732fb1fcb70483ca1945f80f705b474ae6c26d68fa3e27457fb45c7339f8` | `f8fbc6cf2b2cacaf60805eb62a1c9ff9ce2b2f175f2ebd77ee3bc1f97594cb83` |
| `noise/sigma0cm_lat1step` | `fd40e94179dd26f05c60deb182c1cedfd0067acd28c15f21f7b9bb0f656bf4ce` | `a1ce1f64c9a7e1188f696af69c221f24b8408e9d115a7e767a914efabbe1108c` | `4c7c33c6ddc230605802a0cb09960e2d258579285ead64b48177c38c96bc080b` | `bb832cfc79d6ffa3b3d0d9f24910821e8dc7c08c0362f845d2564361711e6117` | `4c961dc0092a87ef886ce93eea605bc33127f5870d2f47a05bdaaf30c0d37846` |
| `noise/sigma1cm_lat1step` | `23d3edca88286538778f18b2fc7858d26e460f599919eea58706feeaa62fd80f` | `2f4e96ac89b63ed4375c9913434f13ed36ca07affae28c153cebb14c25c42e14` | `ebae27b1c9e9df63261aafb32a74512f8dac66b12854f39c83901cd5f44f8ba5` | `24a880f2bd923d907827bee6808c8a73bd929f2349afe46c148670e48b02e0e4` | `8647206d5dd5b999cefc40ff0dce81885eef5395d3609d677f7ee9885f25fa8d` |
| `noise/sigma2cm_lat1step` | `bb8e03d4edf5eda7b5e8132a6a2a254a5da5c058f29f755c491d6a9b14ee4fe2` | `e69df22903835c8a9cb3309d7f64aff353c371a503001c746ac52ee929062c68` | `c6c27079192bf2f209c467dd7735cf2bb6800e2be0fca6644f9274124a70a29b` | `f8ed57fd08a1d9505b38f890b1d33c7630e1a07d11d5d4f7b5ba6cdc4ef1a3a5` | `c559ac3f7b5e768afcc84faff5a44e6f56c792a0954e6d412a8d4617143b9eec` |
| `noise/sigma4cm_lat1step` | `3241752e6e71b76b130ea57b56694cbfb9fa7ff3809457a508a742e841dea5ae` | `75efb59617cf236ad2801b700f511206b84cc3da30a172a9222bfcde971fe5eb` | `b31b41c24d86bd377af99021f1683fd6f03b853cd0e148f5d8b1c5798c53d5c0` | `2396714251f5fc4a11f2f1899eaa21790cc19e61c6ca58accd04f30df95fc180` | `1843785de4445b13e25995ddc3c04a2278b4839123f640fc63f1654bee552efc` |
| `noise/sigma0cm_lat2step` | `bd08e01a2be35f2336dcf03164631c86c78b1fce1a9a03854d8a329a5f72b23e` | `ab06f5fea7bb52c3dcef74d2efe802397bd91af7f67454f93955e6a8d1dedd6d` | `cc257b60c4e9aa9457c35e12f00be1d07dac603dd20dee0028e71915c3937162` | `bd3c7ce49ab7072666dc22eedcbf8d00af220def6b92f86760ea4d556b79c585` | `9e7d4a5d9b705c0580df90c93152354ab98e221622fbde2ab56e14255a3ba190` |
| `noise/sigma1cm_lat2step` | `2a54fcb72d02412c4fb090d72ab5eb919172cf7a1453ddbf2e1309d076d68d59` | `deb6f21922d54f68566db7b1b66e674aad056c802deb58e0f1ec1b3090862c0f` | `ff1a94c71b67274e6dc3d0a538c40682b11f19a797e55343d2349bc056034d6b` | `11147963104bbc45fda68040294591e4da7161f2282430f87f172f5bafe7aca0` | `14c47964b7aada79f450710b259ca6dd1e88ad224c7a309931487c49e69afcd2` |
| `noise/sigma2cm_lat2step` | `bc5f4066e19b996def0cfa82bcf46fbde0855aade76add56a2569eaf340e82bf` | `bd8d1f8abc22c493de8604eaf70f1f77bd2d6825829103c91d5a1fa45c301f21` | `4443b5e1c74c8b61a8634d2e1904290b8f6bb58f0b0adb1cc1049e4fc60de174` | `16eee42174e49411c920ecb5de88a5d59ebadb114d1eb9b00079ffbea6909ee3` | `c16c2cc055c2c61f30535b1a46575d365251a0690bd62208969007be3647f8b8` |
| `noise/sigma4cm_lat2step` | `8b4b6a83cec3b12e3faabee0c03b684fdd6c00cec57d5dc375c7143705f9bcaa` | `503ce7700d2bf3d45e5ec2e43326e492b1c499713928b2acc0f55e16b3a9e27f` | `82c988295d92e997f542ca33d2bcbc9687c9212cb4e104e4fe0defebd80d6791` | `c3a45246e6b084fe47578aeeb2c90b497596f38452d5fbe11b20b74d9f7ca28c` | `e5ff12d04ef745748d273c10a57bd456171f4380bcd862b4969303042531eee5` |

The per-condition `tunnelled_success.csv` of the other arms were new at `182cdea`, and their
episode rows are unchanged:

| file | SHA-256 |
|---|---|
| `results/e07/matrix/tunnelled_success.csv` | `45c6c7894cf490e7ad63412fd08770e2794d173b68cf3d5c2a650f7c65175dd1` |
| `results/e07/cg/tunnelled_success.csv` | `dc9ae9921cc8a0b561e7a378b17e6a02832eae0393371ff095362a6aaccd12f9` |
| `results/e07/sinusoid/tunnelled_success.csv` | `8f981cdbca25d7d4c17c359cd58a2e3e0507356c2ed87715903750a7fc7a75ef` |
| `results/e07/lambda/lam15/tunnelled_success.csv` | `8852d2330372ada7333ea2edf27a2ca23009fed2a7fc1e02870a27bb87b25785` |
| `results/e07/lambda/lam40/tunnelled_success.csv` | `948cce933b1d0ae43e80b76554e184b07c8f91bc75ced73d65488501b19db619` |
| `results/e07/mss/tunnelled_success.csv` | `130522fbeac30925e22cdbb24f9235266ae9bf20935d7a7758f4a69687255505` |

**4. The superseded arm.**
- It is in `results/e07/noise_superseded_p7d1/`: the 11 conditions flown at `6b83e5c`, bytes
  unchanged.
- P7-D2 §5's `noise/*` rows stay the authority for it. `rld.eval.superseded` checks the bytes
  against them on every `eval_phase7.py --check`, and they matched at 19:28 (item 2).
- It is never re-flown and never re-scored. `results/results.md` §4 names it as superseded and
  reports none of its numbers. `docs/findings.md` §4 cites three of its cells only as the
  artifact P7-D4 corrected.

**5. The m6 relabel** (`182cdea`, `rld.eval.hypotheses.verdict_half_rule`). H3's `unseen_vessel`
half-rule word "holds" / "fails" was outside P3-D1 §8's vocabulary.
- The rule text now written in `hypotheses.csv` for all four half-rule rows is: "judged on point
  estimates; not applicable if r(id) <= 0; not scored if the id part is not supported (no
  supported id gain to shrink); behind a supported id part: supported if r(unseen_vessel) <= 0.5
  r(id), else not supported".
- The only verdict word that changed: the secondary `unseen_vessel` SS6 row moved from "holds" to
  **"not scored (no supported id gain)"**. Its `id` SS6 part is "inconclusive" (+1.9 %).
- The other three half-rule rows stay "not applicable — no id gain to shrink". No number changed.

**6. The `below_deck` bail-out (review M3): diagnosis.** The frozen criteria do not change. This
is a documented scoring artifact.
- *The rule.* `src/rld/envs/landing_env.py` l.731 declares `crash` / `below_deck` when the drone's
  z < `deck_origin_z` − `bounds.below_deck_m`. That is 1.0 − 0.3 = 0.7 m
  (`configs/env/landing.yaml` l.107–109).
  - The 0.3 m is a fixed **model-scale** constant and is not λ-scaled. Deck excursions in model
    metres scale with λ, so at 1/15 they are 25/15 = 1.67× those at 1/25.
  - The check runs every physics substep. The outcome is classified at the end of the control
    step, and `crash` outranks `success` (P3-D1).
- *It fires on drones resting on, or tracking, a deck in a deep trough* (λ = 1/15, `unseen_vessel`
  SS6, `results/e07/lambda/lam15/episodes.csv.gz`).
  - #127 (`ppo` seeds 0, 1, 2, 4 and `pid_feedforward`): no contact. The S175 aft pad reaches
    0.51 m below the mean deck, so the drone crosses 0.7 m while about 0.2 m above the pad.
  - #114 and #133 (`pid_feedforward`): the drone rests on the pad as it sinks 0.31–0.32 m below
    the mean deck.
- *It can pre-empt a completed dwell in the same control step.*
  - Example: λ = 1/15, `pid_feedforward`, `unseen_vessel` SS6 #133. Touchdown at 4.079 s,
    closing speed 0.155 m/s, lateral offset 0.011 m, relative tilt 2.1°, and a recorded dwell of
    0.521 s.
  - The dwell reached 0.500 s at 4.579 s and the bound fired at 4.588 s, inside the control step
    that ends at 4.600 s. The episode is scored `crash`. Every other success criterion was met.
- *It also occurs at λ = 1/25, through deck roll rather than heave.*
  - The episode is `unseen_heading` SS6 #177 (frigate, 90°, 0 kn, seed 5). Matrix `ppo` seed 1
    crashes on it at the aft pad (`results/e07/matrix/episodes.csv.gz`, the C1 of that cell). At
    the CG, 3 more crash on it: `ppo` 1, `ppo_forecast` 0 and `ppo_sinusoid` 4
    (`results/e07/cg/episodes.csv.gz`). λ = 1/15 #20 (`ppo` seed 2) is the same case.
  - The pad sinks no more than 0.18–0.26 m. Instead the deck rolls to 24–32°, the drone slides
    across the plate and off its low edge, contact is lost, and the drone drops below 0.7 m.
  - Dwells were 0.40–0.48 s. These are failed landings under any label; the bound sets the label
    to `crash`.
- *Verification.*
  - The rows, reasons, contact counts and dwells are read from the committed episode files above.
  - Deck depths and roll come from the analytic deck source of each listed realization
    (`rld.eval.envs.motion_for`, with `with_lambda` at 1/15).
  - The drone-vs-plate picture comes from a post-hoc scratch re-flight of #133, #127, #20
    (λ = 1/15) and #177 (aft and CG, 1/25), traced per substep. Each re-flight reproduced its
    committed outcome, termination reason, step count and dwell. Nothing from it is committed.
- *Other `below_deck` rows* in the committed e07 files have no contact and were not traced. They
  are mostly `sac`: 1 in the matrix, 9 at CG, 3 on sinusoids and 1–53 per noise condition. Each
  4 cm noise condition also has 17–27 non-`sac` rows.
- *Effect on reported numbers.* One success is lost, `pid_feedforward` λ = 1/15 `unseen_vessel`
  SS6: 193/200 would be 194/200. No hypothesis reads the λ arm, and no λ contrast separates for
  `ppo` or `pid_feedforward`.

**7. Tunnelled successes (review M4).** Counted directly from `tunnelled_success.csv`
(`n_success_tunnelled`, pooled over seeds): successes whose penetration exceeded 5 mm at any
contact substep. The full per-cell tables are in `docs/findings.md` §2 (matrix) and §4 (noise
arm). `results/results.md` prints them in every outcome breakdown.
- *What each count measures.*
  - The direct count is every such success. It may, but need not, depend on the overlap.
  - P5-D14's and P6-D5's bounds are the subset of those successes that the audits judged possibly
    overlap-dependent: an unloaded stretch after contact (`possibly_dependent` in
    `results/audit/tunnelling_episodes.csv` and `results/audit/e06/tunnelling_episodes.csv`).
  - The audits examined exactly the successes the direct count lists, so the bound is a subset of
    it.
- *Matrix, `id`, per 1 000 (SS3/SS4/SS5/SS6) against the bound:*
  - `residual_ppo` 0/0/0/4 against 3 (SS6);
  - `ppo_forecast` 0/0/0/4 against 1 (SS6);
  - `ppo` 0/0/1/3 against 2 (SS6);
  - `sac` 3/7/23/50 = 83 against 41 (1/3/13/24);
  - `pid_feedforward_lowvz_cut` 0/0/1/2 per 200 against 1 (SS6);
  - `ppo_sinusoid` 0/0/0/3 against 3;
  - `residual_ppo_forecast` 0/0/1/0 against 0 at SS6.

  The direct counts exceed P6-D5's "may depend" bound at `id` SS6 for `residual_ppo` (4 vs 3),
  `ppo_forecast` (4 vs 1) and `ppo` (3 vs 2). Over `id` they exceed it for `sac` (83 vs 41) and
  `lowvz_cut` (3 vs 1).
- *Outside `id`* nothing is audited. The largest counts are at `unseen_heading` SS6 (`ppo` 25,
  `ppo_forecast` 22, `ppo_sinusoid` 19 per 1 000; `sac` 44) and `unseen_seastate` SS6 (`sac` 49;
  the pure PPO methods 7–8).
- *Noise arm (re-flown), unaudited.* At σ_p = 4 cm:
  - `ppo` 30–59 per 1 000 per cell, and `ppo_sinusoid` 18–57;
  - `pid_feedforward_lowvz_cut` 14–35 per 200 (35 of its 90 successes at SS6, 4 cm, 2 steps);
  - `sac` 3–122 per 1 000 per cell across all conditions.
- *Against the verdicts* (post hoc, point estimates, nothing re-scored):
  - H1b: removing all 4 gives +5.6 (item 8).
  - H2's cells hold at most 8 per 1 000 (`ppo`, `unseen_seastate` SS6), i.e. under 1 point per
    cell, against a 10-point prediction and a point estimate of −1.5.
  - H1a and H3 are p95 closing-speed statistics; a tunnelled success does not move a touchdown's
    closing speed.

**8. Errata to P7-D3 §6 (review m7).** H1b's tunnelling arithmetic used the bound, 3, as if it were
the raw count.
- The raw tunnelled-success count of `residual_ppo` at `id` SS6 is **4**. Removing all 4 gives
  961/1 000 − 181/200 = 96.1 − 90.5 = **+5.6**, still ≥ 5.
- "Crossing would take 10" should read **11**: at 10 removed the difference is exactly 5.0, which
  still meets "≥ 5". 11 is 3.7× the bound and 2.75× the raw count.
- "Combined": with all 4 removed (+5.6), k = 2 converted `pid_feedforward` bounces still cross
  (5.6 − 1.0 = 4.6), and k = 1 does not (5.1).
- P7-D3's conclusion stands: tunnelling alone cannot move H1b across its line, and the
  bounce-grace rule can.
- Markers are appended in P7-D3's H1b caveat and in its two "holds" mentions.

**9. Other text corrections in `docs/findings.md` (Phase 7).** No number in a committed CSV
changed for any of these.
- *M1.* Every latency sentence is rewritten from the re-flown arm.
  - At `id` SS5, 2 steps, σ_p = 0: `pid_feedforward` 92.5 % (superseded 0.0), `residual_ppo` IQM
    99.0 (superseded 0.2) and `ppo` IQM 100.0 (superseded 35.3). The seed means are 99.1 and 100.0
    (superseded 0.2 and 38.3).
  - The superseded arm appears only as the artifact P7-D4 corrected.
- *M2.* The sentence "its ship-motion feed is the only undelayed, noise-free deck-state channel" is
  replaced. Under P7-D4 every deck-derived observation entry is perceived. What stays ideal is the
  forecast methods' ship-motion feed and `oracle_gated`'s privileged context.
- *The new noise picture.* It is reported per sea state and per condition, beside the baselines,
  and no mechanism is asserted. Velocity noise through the feedforward is named only as a
  plausible, untested explanation.
- *m1.* At `unseen_heading` SS6 the pure-PPO tunnelling rate is 5.6 / 6.8 / 6.9 % (`ppo` /
  `ppo_forecast` / `ppo_sinusoid`). The earlier "1.4–2.0 %" pooled SS3–SS6.
- *m2.* "The PID baselines land softer than every learned method in every regime" was false.
  `pid_track_descend` lands harder than every PPO-family method in **12** of 14 cells; it is
  softer only at `unseen_heading` SS3 and `static`. The other five baselines are softer in all 14.
  (The review said 13 of 14; recomputing from `matrix/aggregate.csv` and
  `matrix/carried_summary_e01.csv` gives 12.)
- *m3.* The like-for-like clean tunnelling rate is 1.09 % (313 of the 28 800 clean `id` rows,
  learned and baselines). The earlier 1.4 % was all matrix rows of every regime (1 393 / 98 800).
- *m4.* The findings λ table now includes `pid_track_descend` and `oracle_gated`.
- *Detector disagreement in the re-flown arm.* It is above 1 % only at σ_p = 4 cm (1.21 / 1.28 /
  1.30 %). Of those disagreements, 8 / 8 / 4 per 28 800 are successes.
- *`below_deck`.* The earlier "only at λ = 1/15" is corrected (item 6).

**10. The stale `configs/env/noise.yaml` comment** is recorded here, not edited. Its SHA-256,
`7beb41da…`, is recorded as `noise_yaml_sha256` in every committed e07 summary.
- Its header says the stand-in is "Applied to the RELATIVE PAD POSE BLOCK ONLY (relative position
  and relative velocity)". Under P7-D4 the perceived quantity is the deck sample at the pad, and
  every deck-derived observation entry is built from it (P7-D4 items 1–4). The noise is drawn in
  the world frame.
- "At lambda = 1/25 a 33 ms model latency is 165 ms full scale" should read 33.3 ms → 166.7 ms.
  A configured 33 ms quantises to 0 steps (P7-D1 §4).
- "Sigma in {0, 1, 2, 4} cm and latency in {0, 33, 66} ms" is the plan's wording. The configured
  latencies are 0 / 33.4 / 66.7 ms = 0 / 1 / 2 steps (P7-D1 §4).
- "applied after the hold" and "The relative-pose block is refreshed at this rate" now apply to
  the perceived deck sample.
- Where the file and P7-D1 §4 / P7-D4 disagree, the protocol wins.

**11. No verdict moved.** `hypotheses.csv` is byte-identical before and after the re-flight. Its
only change since P7-D3 is the m6 word in item 5.
- H1a not supported; H1b supported (+6.0 [+2.0, +10.2]); H2 not supported.
- H3: primary not supported at `id` SS5 and SS6, half-rules not applicable; secondary not
  supported at SS5, inconclusive at SS6, half-rules not applicable (SS5) and not scored (SS6).
- H4 not supported, novelty claim withdrawn; H5 pending — scored at Gate 8.
- No hypothesis reads the noise or λ arm. The M3 artifact and the M4 counts do not move any
  scored statistic across a verdict boundary (items 6–8).

### P7-D6 — Re-review fold-in: noise-to-deck ratios, untested mechanisms, and text errata (2026-10-04)

*Status.*
- This entry records the text fixes from the `results-skeptic` re-review of `7be11a7` (MJ1,
  m1–m7 and two notes).
- Nothing is re-flown or re-scored. No results CSV changes. It changes no success criterion,
  no episode list, no hypothesis, no threshold and **no verdict**. P3-D1's block SHA-256 stays
  `21465588…`, and MANIFEST `e6f30e55…` is untouched.
- P7-D1, P7-D2 and P7-D3 are not edited in place. P7-D1 §4 and P7-D2 carry short appended
  markers, listed with each item.
- The only code change is one bullet in `rld.eval.results_md._NOISE_STANDIN`, the
  `results/results.md` §4 stand-in definition (item 7).

**1. Erratum to P7-D1 §4: noise magnitude against deck motion (review MJ1).**
- *What P7-D1 §4 said.* "The SS5 `id` deck v_z std is about 0.36 m/s for comparison (plan D0.1,
  scouting numbers)." `docs/findings.md` §4 inherited it as "σ_v = 0.2 m/s is more than half the
  SS5 deck v_z standard deviation".
- *Why it is wrong.* Plan D0.1's scouting table flagged its own sign as unverified. P1-D2 showed
  that it used the wrong sign: SS5 180° 12 kn is 0.367 m/s with the wrong sign and 0.222 m/s with
  the right one. It was also one head-seas cell, not the `id` sea state.
- *The committed values.* The rows are the aft rows of `results/deck_stats.csv` for the 12
  `id`-regime cells: frigate, headings 45 / 90 / 135 / 180°, speeds 0 / 6 / 12 kn. Each cell's
  `z_std_model_m` and `vz_std_model_m_s` is over its 40 realizations, at λ = 1/25, model scale.
  The deck SD of a sea state is the unweighted mean over its 12 cells.
  - Check: restricting to the 384 realizations the `id` list draws from (seeds 32–39, via
    `results/deck_stats_seeds.csv`) moves no value by more than 0.0008 m/s or 0.13 mm.
  - The scouting cell itself (SS5, 180°, 12 kn) is z SD 4.43 cm and v_z SD 0.2215 m/s in the
    committed row (P1-D2 quoted its recomputation as 0.222).

| sea state | deck z SD, cm (cell range) | deck v_z SD, m/s (cell range) | σ_p / z SD at σ_p = 1 / 2 / 4 cm | σ_v / v_z SD at σ_v = 0.05 / 0.10 / 0.20 m/s |
|---|---|---|---|---|
| SS3 | 1.04 (0.39–1.95) | 0.046 (0.015–0.084) | 0.96 / 1.93 / 3.86 | 1.08 / 2.16 / 4.31 |
| SS4 | 2.08 (1.23–3.42) | 0.086 (0.042–0.145) | 0.48 / 0.96 / 1.92 | 0.58 / 1.17 / 2.33 |
| SS5 | 3.41 (2.57–5.23) | 0.134 (0.070–0.227) | 0.29 / 0.59 / 1.17 | 0.37 / 0.74 / 1.49 |
| SS6 | 4.10 (2.71–5.39) | 0.147 (0.068–0.226) | 0.24 / 0.49 / 0.98 | 0.34 / 0.68 / 1.36 |

- *So* σ_v = 0.2 m/s is about 1.5× the deck v_z SD at SS5 and 4.3× at SS3. σ_p = 4 cm is 3.9×
  SS3's deck z SD and 1.2× SS5's. **SS3 has the worst noise-to-signal ratio** of the four sea
  states. The noise is per world axis; these ratios compare it with the vertical deck motion
  only.
- *Changed in `docs/findings.md` §4.*
  - The 0.36 m/s comparison is replaced by the table and the ratios.
  - The bold "**SS3 included**" is replaced by the SS3 ratio.
  - §8 item 5's "0 % even at SS3" now reads "0–0.5 % at every sea state", followed by the SS3
    ratio. The `pid_feedforward` counts at 4 cm are 0–1 of 200 per cell.
- *Marker.* "(corrected in P7-D6)" is appended to the P7-D1 §4 sentence.

**2. Untested mechanisms stated as fact (review m1).** `docs/findings.md` §4 says "No mechanism has
been tested". Two passages nevertheless asserted one, and no run noised or withheld the forecast
feed. Both now read "consistent with … (untested)":
- `residual_ppo_forecast` falling to 3.5–5.7 % "because its `pid_feedforward` base reads the
  perceived deck";
- `ppo_forecast`'s 4 cm robustness "comes from an ideal side channel" / "is not a property of the
  forecast block". This was in §4 and in §8 item 6.

**3. `oracle_gated` under noise (review m2).** "Ideal commit timing does not save it" was wrong.
Only the future trajectory is ideal; the timing and the gates read the perceived observation.
Line references verified at `7be11a7`:
- *Window placement.* The true-future window is placed at `predicted_touchdown_s` =
  t + max(clearance, 0) / descent rate (`src/rld/control/gated.py` l.153). The oracle calls it at
  `src/rld/control/oracle.py` l.145–147. Under P7-D4 the clearance is computed from the perceived
  (noisy) deck.
- *At-hover check.* A commit needs |clearance − 0.3 m| ≤ 0.05 m on the perceived clearance
  (`gated.py` l.182). `fallback_commit_s` is null in `configs/control/oracle_gated.yaml`.
- *Lateral gate.* The gate is set from the perceived lateral error (`src/rld/control/pid.py`
  l.170–175) and read at `gated.py` l.179–181.
  - The review cited `pid.py` l.168–172 and `gated.py` l.179–180. Lines 168–169 are the docstring
    and `cfg = self.pid`, and the gate itself runs to l.175. `gated.py` l.181 is the second read
    of `lateral_ok`.
- §4's "Two channels stay ideal" bullet is reworded to match.

**4. Latency alone (review m3).** "Latency alone costs almost nothing" was too strong. It now
reads "nothing measurable for any learned method; up to 15 points for the PID baselines".
- *Source.* `results/e07/contrasts.csv` `noise/sigma0cm_lat2step.pid_feedforward_lowvz.id.SS5`:
  +15.0 [+9.0, +21.0], clean 95.5 % → 80.5 % (`value_a1` 0.955, `value_a2` 0.805).
- *Learned methods.* None of the 48 latency-only learned contrasts separates. The largest
  learned point estimate is `sac` SS6 at 2 steps, +5.3 [−2.0, +11.2].
- §8 item 5's "latency alone moves almost nothing" is corrected the same way.

**5. "The pure PPO policies sit above every classical baseline" under noise (review m4).** The claim
is narrowed to `ppo` and `ppo_forecast`, and the overlaps are named.
- *How it was checked.* In each of the 9 σ_p > 0 conditions × 4 sea states (`aggregate.csv` IQM
  seed CI against `baselines_summary.csv` Wilson CIs, privileged baseline included), the seed-CI
  lower bound of `ppo` and of `ppo_forecast` is above the highest baseline Wilson upper bound.
- *`ppo_sinusoid` overlaps `pid_feedforward_lowvz_cut` in two cells:*
  - SS6, 4 cm, 2 steps: 66.7 [50.2, 74.5] against 45.0 [38.3, 51.9] (90/200), as the reviewer
    said;
  - SS5, 2 cm, 2 steps: 97.3 [93.8, 98.8] against 91.5 [86.8, 94.6] (183/200), which the review
    did not list.
- At σ_p = 0 with latency, SS3 (both latencies) and SS4 (1 step) tie at the 100 % ceiling.

**6. Tunnelled share of `pid_feedforward_lowvz_cut`'s 4 cm successes (review m5).**
- Added to the 4 cm success bullet in `docs/findings.md` §4, from
  `results/e07/noise/sigma4cm_lat*/tunnelled_success.csv` (`n_success_tunnelled` /
  `n_success`).
- *0 steps, SS3–SS6:* 26/83, 18/85, 20/93, 14/78.
- *1 step:* 25/85, 20/85, 17/86, 21/84.
- *2 steps:* 16/81, 19/80, 29/81, 35/90.
- That is 17.9–38.9 % of successes across the twelve 4 cm cells. The review's numbers all verify.

**7. White noise: erratum and limitation for P7-D1 §4's rationale (review m6).**
- *What P7-D1 §4 said.* σ_v was sized as "a relative-velocity estimate smoothed over about 1 s
  full scale". Such an estimator's error would be correlated over about 0.2 s model, i.e. about
  6 control steps at 30 Hz.
- *What is implemented.* `rld.envs.noise.PerceptionNoise.perceive` draws fresh N(0, σ²) per
  world axis at every 33.3 ms control step (hold 30 Hz, i.e. 1 step). It draws position and
  velocity independently. So the noise is **white, i.i.d. per control step**.
- **A correlated estimator error of the same σ was not tested.** This is a limitation of the
  arm, and the result may depend on it.
- *Where stated.* `docs/findings.md` §4 (the definition and §8 item 5) and the `results.md` §4
  stand-in definition (`src/rld/eval/results_md.py` `_NOISE_STANDIN`, one bullet added).
- *Descriptive, verified from `results/e07/noise/sigma4cm_lat0step/episodes.csv.gz`.* At SS3
  (4 cm, 0 steps):
  - all 120 crashes of `gated` and all 120 of `oracle_gated` are `tilt_gt_crash` with
    `n_contacts` = 0;
  - 88 of `pid_feedforward`'s 90 crashes are the same, and the other 2 are `off_plate_strike`
    with contact.

  Whether a correlated error would give the same in-air crashes was not tested.
- *Marker.* "(Limitation recorded in P7-D6 …)" is appended to P7-D1 §4's rationale sentence.

**8. P7-D2's superseded noise rows (review m7).**
- *Rows.* P7-D2 §1's and §2's `noise` rows, and its eleven §5 `noise/sigma…` hash rows.
- *Why marked.* They describe the arm flown at `6b83e5c`. That arm moved unchanged to
  `results/e07/noise_superseded_p7d1/` at `8ebfd94`, and the path they name,
  `results/e07/noise/`, now holds the re-flown arm's different bytes.
- *Marker.* Each row now carries "(superseded by P7-D4; now `noise_superseded_p7d1/`; current
  hashes P7-D5 §3)".
- *Still in force.* The §5 hashes stay the authority for the superseded bytes. They are
  hard-coded in `rld.eval.superseded`, which checks them on every `eval_phase7.py --check`.

**9. Notes, recorded and not changed.**
- *`hypotheses.csv` sources.* The `sources` column of the four H3 half-rule rows cites "P7-D4 (m6
  wording)". The relabel itself is recorded in P7-D5 §5; P7-D4 only announced it. The CSV is not
  changed: changing it would change a scored file's bytes for a citation.
- *`91f3447`'s message.* It says the noise arm was flown from `182cdea` with a "clean tree".
  All 11 `run_info.json` record `git_dirty` true. The dirty paths are untracked only:
  `.claude/worktrees/` everywhere, plus the output directory `results/e07/noise/` in 10 of 11
  (not in `sigma1cm_lat0step`, the first). The preflight guard checks only `src`, `configs`,
  `scripts`, `Makefile` and `pyproject.toml`, so "clean" meant "no tracked or untracked change
  under the code paths". P7-D5 §2 already states this correctly.

**10. Checks and hashes** (this entry's working tree, uncommitted):
- `results/results.md` is now 2 773 lines, SHA-256
  `6e84adba62676963072f76d76c69c93f119a51759dcf6b3d6b814ec2c798f2b7`. That supersedes P7-D5
  §3's `7f737727…`; the only diff is the one added §4 line.
- A second `scripts/report.py` render is byte-identical (`cmp`), and `scripts/report.py
  --check` reports byte-identical.
- `results/e07/contrasts.csv` (`d373299e…`), `results/e07/hypotheses.csv` (`27b4f6ce…`) and every
  other results CSV are unchanged.
- `scripts/eval_phase7.py --arm all --check` (read-only) exits 0 with 32 OK lines: 17 conditions,
  the superseded record and its 11 conditions against P7-D2's hashes, `lambda/feasibility`,
  `contrasts.csv` and `hypotheses.csv`. Afterwards `git status` shows only
  `docs/findings.md`, `docs/protocol.md`, `results/results.md` and
  `src/rld/eval/results_md.py` modified, plus the untracked `.claude/worktrees/`.
- `make lint` is clean (ruff, ruff-format, mypy --strict). `pytest tests/test_eval_*`: 222
  passed, 0 skipped.

## Gates
| gate | date | result | note |
|---|---|---|---|
| 0 | 2026-09-21 | PASSED | `make test lint` green (6 tests, 53 s); pybullet 3.2.7 built from sdist and opens a DIRECT client on 3.12.3; `results/env_throughput.csv` written (8 rows, 1/8/16 SubprocVecEnv workers x rpm/vel); both submodule SHAs recorded in P0-D1. |
| 1 | 2026-09-21 | PASSED | `make test lint` green (66 tests, 62 s; ruff + ruff-format + mypy --strict clean). `results/deck_stats.csv` 192 rows = 96 cells x {aft, cg}, full grid (2 vessels x SS3-SS6 x 4 headings x 3 speeds), all 2304 realizations, with `z_std_model_m`, `vz_std_model_m_s`, `vz_p99_model_m_s`, `az_p99_model_m_s2` populated and no NaN; plus `deck_stats_seeds.csv` (4608 rows) and `deck_feasibility.csv` (2 rows). lambda = 1/25 and r_pad = -0.4*L confirmed in P1-D1. Feasibility rule PASS: frigate SS6 180 deg 12 kn aft vz p99 = 0.577165 m/s vs the 2.08333 m/s gate threshold, 3.61x inside; the rejected `SPEED_LIMIT` reading is recorded as FAIL beside it. Sign convention and ZYX rotation order corrected and pinned by three hand-computed cases (P1-D2); one project-wide lambda (P1-D3). |
| 2 | 2026-09-22 | PASSED | `make test lint` green (118 passed, 1 skipped, 107 s; ruff + ruff-format + mypy --strict clean). The 1 skip is by design: `tests/test_platform.py` gates only the configured driver and *measures* the other, and `test_constraint_driver_fails_the_tracking_gate` asserts the rejected one fails. Platform tracking, `kinematic` driver, 10 s model at frigate SS6 180 deg 12 kn aft: body-origin 0.000 mm and plate-corner 2.2e-16 m against the 1 mm gate, orientation 3.0e-8 rad, `getBaseVelocity` linear and angular ratio 0.0 % against the 2 % gate; `constraint` fails every one of those by an order of magnitude and cannot be tuned into passing (24-point sweep, four identical digits) -- P2-D1. `gymnasium.utils.env_checker.check_env` passes; reset determinism bit-identical as the env's 1st and 3rd reset; drop on a static pad registers exactly one touchdown; scripted 0.3 m/s descent on a static pad 100/100 `success`. `results/e00_env_sanity.csv` (4 rows) + `results/e00_env_sanity_episodes.csv` (800 rows) written from the `id` split's val partition, draw seed 20260922: hover 1.000 `timeout` at SS3 and SS5, random 1.000 `crash` at both, no successes under either -- outcome fractions sum to 1.000000 per row. **PyBullet-vs-analytic touchdown disagreement 0/800 = 0.0000** against the < 1 % gate, plus 0/100 on scripted static descents and 0/30 on a moving SS5 deck. `results/env_throughput_landing.csv` (8 rows) records the deck-in-the-loop throughput P3-D1 must size from (P2-D8); `results/env_throughput.csv` untouched. Two flags carried forward, both recorded rather than fixed: one of 800 sanity episodes tunnelled (7.86 mm penetration vs the 5 mm threshold) and it was the only random episode that ever reached contact, so the random arm exercises the touchdown path barely at all -- the scripted tests carry that load; and the plan's "dropped from rest" is not expressible in a velocity-setpoint action space, so the drop test uses the maximum commanded descent (P2-D9). |
| 3 | 2026-09-22 | PASSED (2nd attempt) | First attempt FAILED on the `results-skeptic` review (BLOCKING B1: the restated H1 was beatable by a slower PID; MAJOR M1-M3), before any RL run; remediated with user decisions (P3-D1 revision 1 §9, P3-D3 amendments). Second attempt: `make test lint` green (233 passed, 1 skipped by design, 162 s; ruff + ruff-format + mypy --strict clean). `pid_feedforward` 200/200 on the static pad and 200/200 at `id` SS3 (Wilson [98.1, 100.0] each) against the >= 95 % gate, from `results/e01/episodes.csv` at `d35f224`. Success-vs-sea-state table `results/e01/success_vs_seastate.md` committed: 5 controllers x 14 cells x N = 200, re-renders byte-identically from `summary.csv`; no crash or off_pad anywhere; detector disagreement 5/14000; tunnelling 14 (all `pid_track_descend`, max 6.5 mm). Frozen lists `results/episodes/` committed at `0780aaa`, MANIFEST SHA-256 `e6f30e55e478d39061b94e38de33959ca99b16e8cac38a3bd99b34f6543ad4a2`, `--check` 10/10 OK. P3-D1 FROZEN revision 1, block SHA-256 **`21465588610e65f1d1253bd26e5ce5db1da938889c6042338e1de8d4d99f5ed2`** (from `### P3-D1 — FROZEN` to the line before `### P3-D2`, UTF-8). Second `results-skeptic` review: no BLOCKING; MAJOR-1 (H1 scope) and MINOR 1-9 folded in by P3-D4 errata without touching the frozen block; three items carried to Phase 4 (P3-D4). |
| 4 | 2026-09-23 | PASSED | Criteria checked against committed artifacts at `eeb87d5` plus the Gate 4 wording fixes. `make test lint` green: 320 passed, 1 skipped by design (P2-D1), 313 s; ruff, ruff-format and mypy --strict clean. `tests/test_deck_forecast.py` and `tests/test_control_gated_forecast.py` 56/56 with **0 skips**, artifacts present (a fresh clone without `artifacts/dmf/` would skip parity, so this run is the evidence). **Parity:** `results/forecast/parity.csv`, 18 gated checks over 6 model dirs, all pass at `1e-4·max(1,|y|)`, worst 0.041× tolerance; bridge-fed parity needs dmf's float32 cast (P4-D3). **Causality:** the feed-clock spy and the future-perturbation bit-identity tests pass. **Leakage:** 729 train / 135 tune forecaster keys vs 1 273 frozen-list realizations, overlap 0 (recomputed independently from `fit_keys.json`). **Frozen-list results:** `gated_forecast` and `gated_forecast_tcn` committed in `results/e02/` beside `gated`, `oracle_gated` and the three PID baselines, 26 cells × 200 each at aft and CG, with `td_in_quiescent_window`, quiet landings per listed episode and Wilson CIs; the static list is not run for feed controllers, reason recorded. Phase 3 aft rows byte-identical to e01 (14 000/14 000). Secondary seed arms in `results/e02_tcn_seeds/` (`12fd9f5`, clean). Oracle relabel done; P4-D5 records the hash. MANIFEST 10/10; P3-D1 block SHA-256 unchanged. **Results-skeptic:** first review MAJOR M1–M5 remediated (P4-D3 wording, P4-D4a erratum, P4-D5, M1 columns, seed arms); second review no BLOCKING and no MAJOR, 6 MINOR wording and provenance errors fixed in P4-D4/P4-D4a. Headline (P4-D4): pre-registered expectation 1 held, 2 600/2 600 aft `gated_forecast` timeouts; forecast gating lowers success against `gated` in 19 (DLinear) and 9 (TCN) of 26 cells, and improves it in none. |
| 5 | 2026-09-30 | PASSED | Criteria checked against committed artifacts at `b54493d`. `make test lint` green: 462 passed, 1 skipped by design (P2-D1), 368 s; ruff, ruff-format and mypy --strict clean. P3-D1 block SHA-256 unchanged (`21465588…`); MANIFEST `--check` 10/10. **Learning curves** (5 seeds, mean ± SD; `00bff09`): `results/e05/learning_curves_ppo.{png,csv}` (`500e795d0686d04507d452ed07ccd033d908d037c2f3bbc139ddb1d2216c3a45`, `a84124dd35eca77899e26822b3ff7a3b67bcbfea79d7d91199e45245a676ce9c`) and `learning_curves_sac.{png,csv}` (`e9c416e5be4c17eee23fc0d0e7d7dbd7332fd62632ffdb07bb6223e8d1365506`, `cc61fde31be366c14085794ac649097a90aa853e43eeed75295d115d443d394e`); the SD band is not clipped to [0, 1]. **`id` results** (P5-D13, `9707642`): 10 final checkpoints × 800 frozen `id` episodes, aft; IQM success SS3–SS6 PPO 100.0 / 100.0 / 100.0 / 98.2, SAC 99.8 / 98.2 / 87.0 / 72.5, six baselines carried byte-identically beside them; `--check` 4/4 byte-identical. e05 was flown with `src/rld/eval/learned.py` untracked (recorded in `run_info.json`); the audit's 1 248 learned re-flights through committed code reproduce their e05 rows exactly, which covers that provenance gap. **Hacking audit** (P5-D14, `d3af59c`, corrected `b54493d`): findings recorded, not clean — SAC tunnelling 6.6 % (max 15.7 mm, impact-speed driven), SAC pre-contact saturation, post-contact throttle cut in all SAC and 3/5 PPO seeds (H1a confound), and up to 41 SAC / 2 PPO successes that may depend on tunnelling overlap plus a ~7 % closing-speed measurement-instant understatement, both caveated in the e05 table; clean on final-policy hovering, detector disagreement, easy-start exploitation and passive landings; regenerates byte-identically. **Seed spread** (`results/e05/seeds.csv`): PPO success SD 0.0 points at SS3–SS5 and 0.3 at SS6; SAC 0.7 / 1.3 / 4.2 / 9.7 at SS3–SS6 (seed 1 at SS6 55.5 %). No run dropped or resumed; PPO budget rule applied with no cut (P5-D12). **Results-skeptic:** no BLOCKING; MAJOR M1 (overclaim that no success depends on penetration) remediated in `b54493d`; MINOR 1–3 fixed there, 4 and 7 recorded in this row, 5 carried to Phase 6's "Before you start", 6 needs no change. `/phase-gate 5` re-run at `1833810`: `results-skeptic` re-review confirms M1 and minors 1–5, 7 remediated, no BLOCKING or MAJOR; two new MINOR slips fixed in the next commit ("20 of the 41" corrected to 18 SAC unloaded stretches ≥ 25 ms; the overlap bound now also names 1 `pid_feedforward_lowvz_cut` SS6 success). |
| 6 | 2026-10-01 | PASSED | Criteria checked against committed artifacts at `9f2cf3f` plus this row's commit. `make test lint` green: 515 passed, 1 skipped by design (`test_platform.py:191`, P2-D1), 461 s; ruff, ruff-format and mypy --strict clean. P3-D1 block SHA-256 unchanged (`21465588…`); `make_episodes.py --check` OK on all 5 lists (MANIFEST `e6f30e55…`). **All runs complete** (P6-D3): 20 / 20 `done` (`residual_ppo`, `ppo_forecast`, `residual_ppo_forecast`, `ppo_sinusoid` × seeds 0–4), 10 010 624 steps each, trained at `6cedd5d` with `git_dirty` false, none resumed or dropped; inherited `ppo.yaml` unchanged (P6-D1, pinned by `test_phase6_config_inherits_ppo`); smoke runs were pipeline checks only (P6-D2). Learning curves, 5 seeds mean ± SD, in `results/e06/learning_curves_*` (`7d6fc64`). **`id` results committed** (P6-D4, `0e49f39`): 20 final checkpoints × 800 frozen `id` episodes, aft; IQM success SS3–SS6 `residual_ppo` 100.0 / 100.0 / 99.5 / 96.2, `ppo_forecast` 100.0 / 100.0 / 100.0 / 98.0, `residual_ppo_forecast` 100.0 / 100.0 / 99.5 / 96.8, `ppo_sinusoid` (JONSWAP; not H4) 100.0 / 100.0 / 100.0 / 97.3; `ppo`, `sac` and six baselines carried byte-identically; `--check` byte-identical for e06 and e05. **Zeroed residual reduces to the baseline (test):** `tests/test_rl_residual.py::test_gate_zeroed_residual_is_pid_feedforward` (14 frozen `id` episodes, 2 220 steps; every episode column but `method` identical to `pid_feedforward`, per-step actions bit-identical, training-wrapper path identical) and `tests/test_rl_forecast_obs.py::test_zeroed_residual_forecast_policy_is_pid_feedforward` (`residual_ppo_forecast`, 2 episodes) both **ran and passed, not skipped**, in this gate's `make test` and in a separate targeted run. **Hacking audit** (P6-D5, `1d57571`; thresholds pre-stated and committed alone at `f66bca6` before computation): findings recorded — early-training hovering in the pure methods (1 % bins; `ppo_forecast` seed 2 relapse at 200–300 k), `ppo_sinusoid` tunnelling max 7.12 mm vs 7.03 mm and pooled post-contact idle 0.549, up to 7 SS6 successes that may depend on tunnelling overlap; clean on final-policy hovering, detector disagreement, easy starts, saturation, passive landings; **no residual seed has the post-contact throttle cut** (H1a confound); 2 340 / 2 340 re-flights reproduce e06. **Results-skeptic:** no BLOCKING; MAJOR M1 (P6-D5 authority reading contradicted by the descent profile) corrected in place, MAJOR M2 (H4 arithmetically bounded at `id` SS5) recorded as a dated note with H4 unchanged (P6-D6); 9 MINOR fixed or recorded (P6-D6). Re-review at `9f2cf3f`: all remediated, no new BLOCKING or MAJOR; two new MINOR slips (this row's skip evidence; P6-D5's README line count) fixed in this commit. `/phase-gate 6` re-run at `9b1c322` (after the user's H4 decision): `make test` 515 passed, 1 skipped (P2-D1), 474 s; both zeroed-residual tests 2/2 passed, not skipped; lint clean; P3-D1 `21465588…` and all 5 lists OK; `--check` byte-identical for e06 (6/6) and e05 (4/4). A fresh `results-skeptic` review found no BLOCKING or MAJOR. It also flew the trained `residual_ppo/3` and `residual_ppo_forecast/4` with `action_net` zeroed in memory: 0 differing columns from `pid_feedforward` on 4 `id` episodes each. No sinusoid-test-motion episode exists anywhere. Three MINOR slips were fixed in the next commit: the Phase 7 note's hard-landing sentence is narrowed to the methods P6-D5 covers; the post-flight checkpoint digests are in `results/e06/summary.csv`; P6-D5's 2 200 = 2 000 learned + 200 baseline. Its NOTE was also acted on: both H4 withdrawal triggers are now cited, and the sinusoid test-leg definition, period included, is to be recorded before the first sinusoid flight. |
| 7 | 2026-10-04 | PASSED | Criteria checked against committed artifacts at `ccce147` plus this row's commit. `make test lint` green: 723 passed, 1 skipped by design (`test_platform.py:191`, P2-D1), 841 s; ruff, ruff-format and mypy --strict clean. P3-D1 block SHA-256 unchanged (`21465588…`); `make_episodes.py --check` 5/5 and `--mss --check` 2/2 (MSS MANIFEST `49468e19…`, P7-D2). **`results/results.md` rendered from CSVs byte-reproducibly:** `scripts/report.py --check` byte-identical and a second render `cmp`-identical (SHA-256 `6e84adba…`, P7-D6 §10); `eval_phase7.py --arm all --check` exit 0 (17 conditions, 11 superseded conditions against their P7-D2 hashes, feasibility, `contrasts.csv`, `hypotheses.csv`) with the tree left clean; e05/e06 `--check` byte-identical; sinusoid re-flight at 7 workers byte-identical (closes the P3-D4 carry item). **Hypotheses scored in `docs/findings.md`** exactly as pre-registered (P3-D1 §8, P3-D4, P7-D1, P7-D1a; `results/e07/hypotheses.csv`): H1a not supported (r = −0.470 [−0.696, −0.368]; `residual_ppo` lands harder than `pid_feedforward_lowvz`); H1b supported (+6.0 [+2.0, +10.2] points at `id` SS6, out of distribution, 1.0-point margin, fragile to the bounce-grace rule, P7-D3/P7-D5); H2 not supported (−1.5 [−4.5, +1.7]); H3 primary not supported at `id` SS5 and SS6, half-rule not applicable; H3 secondary SS6 inconclusive (+1.9 % [+0.4, +3.8]); H4 not supported (0.0 [0.0, 0.0], ceiling), novelty claim withdrawn (D0.4, P6-D6); H5 pending — scored at Gate 8 (P7-D1 §7, user decision). No multiplicity correction. **Flights:** 578 600 episodes, 6 arms (matrix, CG, sinusoid, λ, noise, MSS), every learned method × seeds 0–4, baselines beside every table (P7-D2). **Deviation P7-D4** (user, option b): the perception stand-in mixed timestamps (review M1) and left deck channels ideal (M2); it now perceives the deck, and the noise arm was re-flown from a clean `182cdea` (P7-D5); the superseded arm is kept byte-identical in `noise_superseded_p7d1/`. No verdict moved. **Results-skeptic:** first review at `2448d14` 0 BLOCKING, 4 MAJOR, 8 MINOR (P7-D4, P7-D5); re-review at `7be11a7` 0 BLOCKING, 1 MAJOR (MJ1, noise vs deck-motion reference wrong since P7-D1 §4), 7 MINOR (P7-D6); gate review at `ccce147` 0 BLOCKING, 0 MAJOR, 4 MINOR wording fixes to `findings.md` (CG CI overlap, plain `ppo` above `residual_ppo` at the H1b cell, H4 headline scoped to its ceiling cell, `sac` 17.6 mm penetration), fixed in this row's commit. |
