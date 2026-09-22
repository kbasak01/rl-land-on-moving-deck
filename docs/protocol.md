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
