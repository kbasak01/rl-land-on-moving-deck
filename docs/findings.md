# findings.md

Phase-by-phase record of what was found, including every claim withdrawn and what replaced it.
Written at each gate. Hypotheses are scored here exactly as pre-registered in P3-D1.

## Phase 9 — release: README, figures, GIFs (2026-10-05)

**Scope.** Nothing new is measured about landing. No criterion, list, number or verdict changes.
Definitions are in P9-D1.
- *H5 rendered.* H5 is rendered into `results/results.md` from `results/latency/h5.csv`
  (supported, 4.72× / 3.94×). `results/e07/hypotheses.csv` keeps its Phase 7 "pending" row.
- *Headline figures.* `results/figures/success_vs_seastate_{id,shift}.png` are drawn from the
  committed `results/e07/matrix/` CSVs only, and are byte-reproducible by `make figures`.
- *Landing GIFs.* Seven GIFs in `results/figures/gifs/` show `pid_feedforward | ppo |
  residual_ppo` on hand-picked `id` episodes, failures included.
  - Each panel is a re-flight that reproduces its committed episode row in every
    `RECORD_COLUMNS` value, compared as text.
  - `manifest.csv` holds the per-panel outcome and the k-of-5 seed counts the captions quote.
  - **They are illustrations, not evidence.**
- *Cosmetic defect, found while rendering.* The yellow pad disc (`DeckPlatform._spawn_pad_marker`)
  was never moved with the plate in any committed flight. It is visual only, with no collision
  shape, so no number is affected. The GIF code draws it on the plate (P9-D1 §2).
- *Bandwidth ratio.* The deck-to-vehicle bandwidth ratio promised in plan D0.1 was never
  measured. The README says so, and quotes only the committed speed ratios from
  `results/deck_feasibility.csv`.
- *Wall clock.* The per-stage wall clock is in `results/runtime_stages.csv`. Training rows are
  copied from gitignored status files and marked `committed = False`.
- *Touchdown closing-speed distributions* (a plan §1 deliverable). They are in
  `results/figures/closing_speed_ecdf_id.png`: ECDFs at `id` SS5 and SS6 over touched-down
  episodes, learned seeds pooled. Nothing in them is tested.
- *Label carried from P6-D6 #8, not met in a figure.* The tune-pool panels of
  `results/e06/learning_curves_ppo_sinusoid.png` are evaluated on **sinusoid** motion, the
  method's training motion, not on JONSWAP. That figure was not re-rendered with the label. The
  README states it instead.
- *Training distribution.* The training pool is the frigate, SS3–SS5, headings 45/135/180°
  (P3-D2). `id`'s 90° episodes and every `unseen_heading` and `unseen_vessel` cell are therefore
  outside it too, not only SS6. The figures shade only SS6, and their footnote says so.
- *Review.* The `results-skeptic` README review found 0 BLOCKING, 3 MAJOR and 19 MINOR issues,
  all wording or disclosure, folded in before commit. No number changed.
- *Pre-release audit* (`docs/audit_report.md`): 3 BLOCKING, 8 SHOULD FIX and 11 NOTE.
  - The BLOCKING items were all README errors, and they are fixed:
    - the two Phase 4 forecast-gated controllers were missing from the README tables;
    - a false S175 Wilson-interval sentence;
    - a false gallery-caption timing claim.
  - Both lists are itemised in P9-D2.
  - The §7 closed-loop parity item is met only under P8-D5. That is unchanged, and it is the
    Gate 9 decision.

## Phase 8 — ONNX export, parity and latency (2026-10-05)

**Scope.** Simulation only. The latencies are measurements of a desktop RTX A4000 and an
i9-10980XE under WSL2. **No embedded target was measured, and nothing here implies real flight.** Only ratios of rows measured on this machine are quoted. Sources:
`results/latency/`. Definitions are in P8-D1, and the full record is in P8-D2.

**What was exported.** `ppo` (the best pure RL method) and `residual_ppo_forecast` (the best
residual by the same rule; its CI overlaps `residual_ppo`'s, so this is a selection, not a result),
seeds 0 and 4. Seed 4 is the median seed of each at `id` SS6. Each graph is the deterministic
actor, a 512×2 tanh MLP, with the frozen `VecNormalize` folded in: the observation clip and the
action clip are inside the graph. The residual's `pid_feedforward` base, the α = 0.3 composition and
the DLinear-OLS forecaster stay outside it.

### Verdict (P8-D2 §7; `results/latency/h5.csv`)

| H | part | number | verdict |
|---|---|---|---|
| H5 | batch-1 p50, `ppo` graph: GPU provider / ORT CPU (1 thread) | ORT CUDA **4.72×** (0.217 vs 0.046 ms); ORT TensorRT **3.94×** (0.181 vs 0.046 ms); p99 5.57× and 6.97×; one measurement per configuration, no CI (a 20-iteration smoke read ORT CUDA 9.35×, P8-D2 addendum) | **supported** |

### 1. Parity

- **Numeric parity passes on every provider that was timed.** The check covered ORT CPU, CUDA and
  TensorRT and the folded torch module on CPU and CUDA, all four graphs, 1 000 observations × 5
  draws, with TF32 off. The worst max error is 9.5e-7 against a threshold of 1e-4
  (`parity.csv`). The draws push about 12 % of normalised entries past the ±10 clip, and the
  reference reaches the ±1 action clip, so both clips are exercised. **No configuration was
  refused.**
- **Closed-loop parity is not met as pre-registered.** 50 frozen `id` episodes, noise off, ORT CPU
  against PyTorch (`closed_loop_parity.csv`).
  - `residual_ppo_forecast` seed 0 has identical outcome classes in 50/50.
  - `ppo` seeds 0 and 4 and `residual_ppo_forecast` seed 4 have 49/50 each. Each differs in one
    SS6 episode, `success` ↔ `hard_landing`.
  - On the PyTorch side those episodes land at 12–16° relative tilt against the 15° limit.
- **Attribution (post hoc, P8-D2 §4).** The ONNX graph differs from PyTorch by ≤ 1e-6 per call. A
  one-ulp nudge of the PyTorch policy's input, with no ONNX involved, flips exactly the same three
  episodes in the same direction (`closed_loop_controls.csv`). Every arm's touchdown v_z drifts
  0.03–0.06 m/s from PyTorch's.
  - So at these SS6 episodes the closed loop is sensitive to float32 rounding, and the flips are
    consistent with no export defect.
  - The pre-registered criterion is still not met.
- **Gate 8 does not pass (user, 2026-10-05).** P8-D1 §7 stays as written; the flips were then
  investigated under a design fixed in advance (P8-D3, results in P8-D4,
  `results/latency/closed_loop_investigation/`).
- **The investigation finds rounding sensitivity, not an export defect.** That is the fixed P8-D3
  reading, and every condition was met.
  - *Export error on the inputs actually flown.* The graph matches SB3 to ≤ 9.5e-7 on all
    13 755 recorded steps of the 50 episodes, including both clip regimes (`same_input.csv`).
    ORT's graph optimisations change no output.
  - *Divergence.* The first act already differs, by ≤ 3.6e-7. The trajectory difference then grows
    exponentially, by e about every 2.4 control steps, to a median of 8 mm at touchdown
    (`trace_summary.csv`). Flipped episodes look like non-flipped ones.
  - *Noise floor.* Over the whole `id` SS6 cell (200 episodes × 4 policies), ONNX flips 16 of 800
    outcomes. Twenty random one-ulp input perturbations of the PyTorch path flip 10–21 each
    (median 15). Every ONNX flip is an episode that some perturbation also flips
    (`noise_distribution.csv`).
  - *Float64.* A float64 PyTorch reference differs from the committed float32 path on 9 of those
    800, i.e. 1.1 % (`flips.csv`).
  - *The 6.6 % is a different measure.* It is the union S of the 20 ulp draws: 53 of 800
    policy-episodes flip under at least one of them. That is the share of SS6 policy-episodes
    whose class this test shows to be not determined at float32 precision. (Attribution
    corrected at the Gate 8 review, m3.)
  - *What it means for the criterion, post hoc.* Only 2 of the 20 perturbed PyTorch paths would
    have passed §7 on its 50 episodes.
  - *What it cannot show.* It cannot rule out a defect on inputs that were never flown; the
    random-input parity above bounds that. Options for the gate are listed in P8-D4 §10 and are
    the user's.
- **Deviation P8-D5 (post hoc; user, option b, 2026-10-05).** Closed-loop parity is judged against
  the measured float32 noise floor on SS6-200. For each policy two things must hold: the runtime's
  flips are no more than the maximum of K = 20 one-ulp-perturbed PyTorch paths, and every runtime
  flip lies in the rounding-sensitive set S.
  - *ORT CPU: met for all four policies.* It flips 4, 4, 4 and 4, against maxima of 5, 6, 8 and 9,
    and all 16 flips lie in S (|S| = 9, 12, 16 and 16).
  - *ORT CUDA is descriptive only and not judged.* 3 of its 18 flips lie outside S, so (ii) would
    fail for 2 of 4 policies. It was scoped out after this was seen. A noise draw lands outside a
    leave-one-out S in only 9 of 291 flips, so P(≥ 3 of 18) ≈ 0.017 (`posthoc_loo.csv`). ORT
    TensorRT and torch-eager CUDA were never flown closed loop.
  - *The yardstick is lenient and biased* (post hoc, Gate 8 review; `posthoc_same_input.csv`,
    `posthoc_bias.csv`).
    - The floor is a per-step dithering perturbation, about 2–4× larger than the export error (median
      |Δa| on the recorded inputs).
    - It is biased toward fewer successes, through the bounce channel: 16 of the 18 draws that
      change pooled success lower it, and success→bounce happens 65 times against 19 for
      bounce→success. This is pooled: for `residual_ppo_forecast` s4 the ulp draws raise success
      (+16; 3 of 15 success-changing draws lower it).
    - On SS6-200, ONNX against `torch_folded` gives 18 flips and success −9, and against
      `torch_fp64` 16 flips and −6. For comparison, `torch_folded` and `torch_fp64` flip 10 and 9
      against `torch`.
  - *P8-D1 §7 as written stays **not met*** (3 of 4 policies at 49/50). Both verdicts are
    reported.
- **Carry to the README (Phase 9).** Outcome classes at SS6 near the 15° tilt limit are not
  determined at float32 precision. About 6.6 % of SS6-200 policy-episodes are rounding-sensitive
  (53 of 800 flip under at least one of 20 one-ulp perturbations). This qualifies the per-episode
  resolution of every committed SS6 success count. H1b's 1.0-point margin at `id` SS6 is of the same order as the rounding-level per-seed shifts in SS6 success measured here: −2.5 to +1.5 points under one-ulp noise and −0.5 to +1.5 under the float64 reference, per seed of the exported policies. `residual_ppo` itself was not re-flown, and about 30 % of flip events (88 of 291 ulp flips) go through the bounce channel, the rule H1b is already noted to be fragile to.

### 2. Latency and the control budget

- *Batch 1, one CPU thread* (`latency.csv`). ORT CPU runs the 25-input policy in 0.046 ms p50
  (0.082 ms p99). With 2–8 threads it takes 0.018–0.021 ms.
  - The GPU rows are slower: ORT CUDA 0.217 ms, ORT TensorRT 0.181 ms, torch-eager CUDA 0.404 ms.
    They include host-device copies by design (Project 4's host-to-host method). The likely
    causes of the GPU penalty were not measured separately: host↔device copies over PCIe and
    kernel launches dominating a tiny MLP, and WSL2's GPU paravirtualisation.
  - At batch 32, ORT CPU at 1 thread is still faster than the GPU rows: 0.156 against 0.243 (CUDA)
    and 0.196 ms (TensorRT). More CPU threads are faster still. *(Corrected in Phase 9: this said
    "the fastest ORT row".)*
- *End-to-end per control step, in full simulated episodes* (`e2e_budget.csv`), one thread, ONNX policy:
  - `ppo`: observation build + policy = 0.19 ms p50 and 0.30 ms p99, i.e. 0.9 % of the 33.3 ms
    period at p99.
  - `residual_ppo_forecast`: observation build + DLinear-OLS forecaster + policy + base + compose
    = 0.85 ms p50 and 1.13 ms p99, i.e. 3.4 %. The forecaster is the largest part (0.53 ms p50).
  - Physics stepping (1.5 ms p50, 13 ms p99) is simulation and is reported apart. So is the
    ship-motion feed (0.55 ms p50).
  - *Not in the deployed totals:* the velocity tracker `DSLPIDControl.computeControl` (setpoint to
    motor RPM). It runs inside `env.step`, so the e2e budget books it in the physics bucket, and
    the deployment sums above omit it. It was not re-timed (Gate 8 review m4).
- **What this does and does not say.** On this machine, a 512×2 MLP at batch 1 runs faster on
  one CPU thread than on any GPU provider *(corrected in Phase 9 from "fastest on one CPU
  thread"; 2–8 threads are faster still)*, and the timed per-step software stack (velocity tracker excluded) uses a few
  percent of the control period.
  It says nothing about any embedded computer's CPU or GPU, which were not measured.

## Phase 7 — evaluation under shift and ablations (2026-10-02)

**Scope.** Everything below is simulation:
- PyBullet, gym-pybullet-drones' Crazyflie 2.x;
- dmf's 3-DOF (heave, roll, pitch) JONSWAP deck motion, Froude-scaled at **λ = 1/25**
  (1 s model = 5 s full scale);
- state-based observations, with a noise-and-latency stand-in instead of vision;
- dmf's roll/pitch–heave phase defect, carried and not fixed.

The MSS records are another simulator's output, not ship measurements. No sentence here describes
real flight or real deck data.

**Sources.** Every number traces to a committed file under `results/e07/`, named beside it.
`results/results.md` renders the same numbers. Provenance and hashes are in P7-D2, and the scoring
record is in P7-D3. The perception arm was re-flown under P7-D4's stand-in; the re-flight, the
review corrections below and the hashes that supersede P7-D2's are recorded in P7-D5.

**Conventions.**
- *Learned rows* show the IQM over 5 seeds with its stratified-bootstrap 95 % CI, which reflects
  seed variation only. N = 5 × 200 = 1 000.
- *Baselines* show the rate with its Wilson 95 % CI and k/200.
- *Losses by class* follow the semicolon, as counts: C crash, O off_pad, H hard_landing, B bounce,
  T timeout. They are pooled over seeds for learned rows.
- `oracle_gated` is privileged: a commit-timing oracle that reads the true future deck motion.
- **A difference is claimed only where a paired or stated bootstrap CI excludes 0.** Anything else
  is called an unpaired reading or a non-result.

### Verdicts (P7-D3; `results/e07/hypotheses.csv`)

| H | part | number [95 % CI] | verdict |
|---|---|---|---|
| H1a | `residual_ppo` vs `pid_feedforward_lowvz`, relative p95 closing speed, `id` SS5 | r = −47.0 % [−69.6, −36.8] (0.276 vs 0.188 m/s); non-inferiority +4.1 [+1.2, +7.1] points | **not supported**: the residual lands *harder* |
| H1b | `residual_ppo` − `pid_feedforward` success, `id` SS6 | +6.0 [+2.0, +10.2] points | **supported**, caveated below |
| H2 | drop(`ppo`) − drop(`residual_ppo`), `id` SS5 → `unseen_seastate` SS6 | −1.5 [−4.5, +1.7] points | **not supported** |
| H3 primary | `ppo_forecast` vs `ppo`, relative p95, `id` SS5 / SS6 | −0.4 % [−2.3, +2.3] / −0.4 % [−6.3, +3.3] | **not supported** / **not supported** |
| H3 primary | `unseen_vessel` half-rule, SS5 / SS6 | — | **not applicable — no id gain to shrink** (both) |
| H3 secondary | `residual_ppo_forecast` vs `residual_ppo`, `id` SS5 / SS6 | −0.8 % [−2.4, +0.9] / +1.9 % [+0.4, +3.8] | **not supported** / **inconclusive** |
| H3 secondary | `unseen_vessel` half-rule, SS5 / SS6 | r(uv) −2.0 % [−3.6, −0.3] / +0.04 % [−1.9, +2.6] | **not applicable** / **not scored (no supported id gain)** (relabelled from "holds" at `182cdea`, no number changed; P7-D5) |
| H4 | drop_sin − drop_jon, `id` SS5 | +0.0 [+0.0, +0.0] (degenerate: every episode a success) | **not supported**; **novelty claim withdrawn** (D0.4 and P3-D1 §8 both fired) |
| H5 | ORT CPU vs GPU latency | — | **pending — scored at Gate 8** → scored at Gate 8, P8-D2 |

No multiplicity correction was applied (P3-D4 #9).

### 1. The hypotheses in plain words

- **Residual RL did not land softer (H1a).** P3-D4 narrowed the claim to "softer than the softest
  constant-descent PID in the tuning log", and the result is the opposite.
  - *The numbers.* `residual_ppo`'s pooled p95 closing speed at `id` SS5 is 0.276 m/s, 47 % above
    `pid_feedforward_lowvz`'s 0.188 m/s. The CI of r lies wholly below 0.
  - *The mechanism (P6-D5).* The residual pushes its final second *faster* than its own base: a
    median executed setpoint of −0.274 m/s at the last step before touchdown, against `lowvz`'s
    constant 0.111 m/s descent.
  - *It also lands harder than its unmodified base,* `pid_feedforward` (0.277 vs 0.262 m/s, IQM of
    per-seed p95 vs a single run; `results/e07/matrix/aggregate.csv`,
    `results/e07/matrix/carried_summary_e01.csv`). That is an unpaired reading; no paired test was
    run.
- **Residual RL beat the PID it is built on at SS6 (H1b), by a margin the bounce rule can erase**
  (see also Phase 8: rounding-level per-seed SS6 shifts of the same order as the margin).
  - *The numbers.* +6.0 [+2.0, +10.2] points at `id` SS6: 965/1 000 against 181/200. The line is
    5 points.
  - *What this cell is.* SS6 is outside every method's training distribution.
  - *What the PID cannot do.* The residual lands in about 2 s with a two-phase descent that the
    PID's constant-descent tuning space cannot express.
  - *The bounce rule.* 11 of `pid_feedforward`'s 19 losses are `bounce`s under the 50 ms grace
    rule. Three of those becoming successes would put the point estimate at 4.5, below the line.
  - *Tunnelling alone cannot move it.* `residual_ppo` has 4 successes per 1 000 at `id` SS6 with
    penetration > 5 mm (`matrix/tunnelled_success.csv`); P6-D5's audit judged 3 of them possibly
    overlap-dependent. Removing all 4 gives 96.1 − 90.5 = +5.6, still ≥ 5; crossing below the line
    would take 11 (P7-D3 §6, arithmetic corrected in P7-D5).
- **Residual RL did not generalise better across sea state (H2).** `ppo` lost 2.7 points and
  `residual_ppo` 4.2 between `id` SS5 and `unseen_seastate` SS6. The difference, −1.5
  [−4.5, +1.7], has the wrong sign and does not separate.
- **The forecast block does nothing measurable to closing speed (H3).**
  - `ppo_forecast` and `ppo` have the same p95 to within 0.4 % at `id` SS5 and SS6, and both CIs
    straddle 0. That holds although the forecast methods get an extra ideal sensor and were trained
    on in-sample forecasts (P6-D1).
  - The only separating forecast contrast is the secondary `residual_ppo_forecast` at SS6: +1.9 %,
    a fifth of the predicted 10 %.
  - In success, the forecast methods are within 3 points of their non-forecast twins in every
    clean cell of the matrix. That is an unpaired reading of the tables below; it was not tested.
- **At the scored `id` SS5 cell, where every method is at the ceiling, training on sinusoids cost
  nothing measurable (H4).**
  - `ppo_sinusoid`, trained only on matched sinusoids, scores 1 000/1 000 at `id` SS5 on both
    JONSWAP and sinusoid motion. So does `ppo`.
  - H4 cannot be supported, as P6-D6 foresaw, and the motion-realism novelty claim is withdrawn
    (D0.4). The transfer at that cell is the finding.
  - Outside the scored cell the picture is weaker for `ppo_sinusoid` (unpaired, untested). At `id`
    SS6 its seed CI, 97.3 [96.7, 97.8], lies below `ppo`'s 98.2 [98.0, 98.5]
    (`results/e07/matrix/aggregate.csv`). Under σ_p = 4 cm noise it is 6–11 points below `ppo` at
    every sea state, e.g. 69.7 vs 79.8 at SS3 with 0 steps (`results/e07/noise/sigma4cm_lat*/aggregate.csv`).
    *(Scope narrowed at the Gate 7 review.)*
- **H5** waits for Phase 8. → scored at Gate 8, P8-D2.

### 2. Success under shift, per regime (main matrix, aft pad, JONSWAP, λ = 1/25)

Source: `results/e07/matrix/aggregate.csv` and `summary.csv` for the learned rows;
`results/e07/matrix/carried_summary_e01.csv` and `carried_summary_e01_lowvz_cut.csv` for the
baselines.
- SS6 is outside every method's training distribution.
- The regimes share realizations (P3-D1 §2), so regime-vs-regime readings are not independent
  draws.
- `static` (all 12 methods 100 %; the forecast methods not run, see P7-D2) is in
  `results/results.md` §1.

Cell format: learned = IQM [95 % CI]; baselines = rate [Wilson 95 % CI] k/N; after the semicolon, losses by class (C crash, O off_pad, H hard_landing, B bounce, T timeout; counts of 1 000 for learned rows, of 200 for baselines; – = none).

#### `id`

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 98.2 [98.0, 98.5]; H15 B3 |
| `sac` (2 M steps) | 99.8 [98.8, 100.0]; H4 | 98.2 [96.3, 99.0]; O1 H20 | 87.0 [81.7, 90.3]; C2 O8 H122 B3 T1 | 72.5 [60.2, 79.3]; C18 O39 H232 B2 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.5 [99.5, 99.8]; H4 | 96.2 [95.7, 97.8]; H24 B11 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 98.0 [97.5, 98.5]; H18 B2 |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.5 [99.5, 99.5]; H5 | 96.8 [96.5, 98.0]; H25 B4 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 97.3 [96.7, 97.8]; H25 B2 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200; – | 96.0 [92.3, 98.0] 192/200; H4 B4 | 85.0 [79.4, 89.3] 170/200; H12 B18 | 80.0 [73.9, 85.0] 160/200; H17 B23 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 99.0 [96.4, 99.7] 198/200; H1 B1 | 90.5 [85.6, 93.8] 181/200; H8 B11 |
| `pid_feedforward_lowvz` | 100.0 [98.1, 100.0] 200/200; – | 98.0 [95.0, 99.2] 196/200; B4 | 95.5 [91.7, 97.6] 191/200; B9 | 85.0 [79.4, 89.3] 170/200; H9 B21 |
| `pid_feedforward_lowvz_cut` | 100.0 [98.1, 100.0] 200/200; – | 98.5 [95.7, 99.5] 197/200; B3 | 98.0 [95.0, 99.2] 196/200; B4 | 88.0 [82.8, 91.8] 176/200; H9 B15 |
| `gated` | 100.0 [98.1, 100.0] 200/200; – | 98.5 [95.7, 99.5] 197/200; T3 | 89.5 [84.5, 93.0] 179/200; T21 | 63.0 [56.1, 69.4] 126/200; B4 T70 |
| `oracle_gated` (privileged) | 100.0 [98.1, 100.0] 200/200; – | 98.5 [95.7, 99.5] 197/200; T3 | 89.0 [83.9, 92.6] 178/200; B1 T21 | 64.5 [57.7, 70.8] 129/200; T71 |

#### `unseen_seastate`

| method | SS6 |
|---|---|
| `ppo` | 97.3 [96.0, 98.2]; H23 B5 |
| `sac` (2 M steps) | 70.2 [62.2, 74.8]; C19 O46 H241 B3 |
| `residual_ppo` | 95.3 [94.0, 97.2]; H38 B7 |
| `ppo_forecast` (ideal feed) | 96.7 [96.2, 97.3]; H31 B2 |
| `residual_ppo_forecast` (ideal feed) | 95.8 [95.0, 96.5]; H35 B7 |
| `ppo_sinusoid` | 97.5 [96.5, 98.5]; H23 B2 |
| `pid_track_descend` | 81.0 [75.0, 85.8] 162/200; H15 B23 |
| `pid_feedforward` | 90.0 [85.1, 93.4] 180/200; H6 B14 |
| `pid_feedforward_lowvz` | 86.0 [80.5, 90.1] 172/200; H7 B21 |
| `pid_feedforward_lowvz_cut` | 89.5 [84.5, 93.0] 179/200; H7 B14 |
| `gated` | 64.5 [57.7, 70.8] 129/200; B2 T69 |
| `oracle_gated` (privileged) | 65.0 [58.2, 71.3] 130/200; T70 |

#### `unseen_heading`

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 90.7 [86.3, 93.3]; C1 O3 H77 B17 |
| `sac` (2 M steps) | 100.0 [100.0, 100.0]; – | 96.7 [95.8, 98.7]; O2 H28 | 84.2 [76.8, 87.0]; C2 O23 H139 B5 T1 | 46.5 [37.0, 55.2]; C86 O98 H343 B8 T3 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 85.7 [84.0, 89.0]; H107 B31 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 87.8 [84.7, 89.3]; O1 H113 B12 |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 87.5 [84.0, 90.3]; H102 B25 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 86.2 [84.8, 87.8]; H116 B21 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 97.5 [94.3, 98.9] 195/200; B5 | 72.5 [65.9, 78.2] 145/200; H22 B33 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 99.5 [97.2, 99.9] 199/200; B1 | 77.0 [70.7, 82.3] 154/200; H24 B22 |
| `pid_feedforward_lowvz` | 100.0 [98.1, 100.0] 200/200; – | 99.0 [96.4, 99.7] 198/200; B2 | 96.5 [93.0, 98.3] 193/200; B7 | 70.0 [63.3, 75.9] 140/200; H24 B36 |
| `pid_feedforward_lowvz_cut` | 100.0 [98.1, 100.0] 200/200; – | 99.0 [96.4, 99.7] 198/200; B2 | 97.0 [93.6, 98.6] 194/200; B6 | 83.0 [77.2, 87.6] 166/200; H24 B10 |
| `gated` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 98.0 [95.0, 99.2] 196/200; B1 T3 | 42.0 [35.4, 48.9] 84/200; B7 T109 |
| `oracle_gated` (privileged) | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 99.5 [97.2, 99.9] 199/200; T1 | 46.5 [39.7, 53.4] 93/200; T107 |

#### `unseen_vessel`

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; H1 | 99.3 [98.7, 99.5]; H7 B1 |
| `sac` (2 M steps) | 100.0 [99.7, 100.0]; H1 | 99.2 [98.5, 100.0]; H8 | 96.5 [95.7, 98.0]; O2 H30 B1 | 81.5 [79.0, 84.7]; C3 O23 H153 B3 T1 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.0 [98.7, 99.3]; H8 B2 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.3 [99.0, 99.5]; H6 B1 |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.2 [98.7, 99.5]; H6 B3 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.5 [99.0, 100.0]; H5 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200; – | 97.5 [94.3, 98.9] 195/200; B5 | 90.0 [85.1, 93.4] 180/200; H14 B6 | 81.5 [75.5, 86.3] 163/200; H20 B17 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 99.5 [97.2, 99.9] 199/200; B1 | 98.5 [95.7, 99.5] 197/200; H1 B2 |
| `pid_feedforward_lowvz` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 99.5 [97.2, 99.9] 199/200; B1 | 95.0 [91.0, 97.3] 190/200; B10 |
| `pid_feedforward_lowvz_cut` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 99.5 [97.2, 99.9] 199/200; B1 | 97.0 [93.6, 98.6] 194/200; B6 |
| `gated` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 93.5 [89.2, 96.2] 187/200; T13 | 90.0 [85.1, 93.4] 180/200; T20 |
| `oracle_gated` (privileged) | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 95.0 [91.0, 97.3] 190/200; T10 | 89.0 [83.9, 92.6] 178/200; T22 |


**Readings.** The learned-vs-baseline comparisons here are unpaired and untested unless a contrast
is named.
- **Sea states SS3–SS5 sit at the ceiling** for every PPO-family method in every regime.
  - The IQM is 100.0 with degenerate seed CIs [100.0, 100.0] in most cells.
  - So these cells cannot rank the PPO-family methods. They differ only at SS6, and SAC differs
    everywhere.
- **`pid_feedforward` ties the learned methods wherever it is not at SS6 of a frigate regime.**
  - It scores 99.0–100 % at SS3–SS5 in every regime.
  - At `unseen_vessel` SS6 it scores 98.5 % [95.7, 99.5] (197/200), against 99.0–99.5 % for the
    PPO family. Its Wilson interval contains every PPO-family point estimate.
  - **On the S175 hull (`unseen_vessel`), these tables show no sea state at which
    `pid_feedforward` is beaten.**
- **In the SS6 frigate cells the PPO family sits above the baselines in point estimate only.**
  The point estimates are:
  - `id`: 96.2–98.2 % against `pid_feedforward`'s 90.5 %;
  - `unseen_seastate`: 95.3–97.5 % against 90.0 %;
  - `unseen_heading`: 85.7–90.7 % against 77.0 %.

  Only `id` SS6 `residual_ppo` vs `pid_feedforward` (H1b) was tested. The rest are unpaired
  readings.
- **`unseen_heading` SS6 is the hardest cell for everyone.**
  - The PPO family drops to 85.7–90.7 %, mostly through `hard_landing` (77–116 per 1 000).
  - `gated` scores 42.0 % and `oracle_gated` 46.5 %, mostly `timeout`.
  - `pid_feedforward_lowvz_cut` (83.0 %) is above `pid_feedforward` (77.0 %) here. That is also
    unpaired; their Wilson intervals overlap.
- **Within the PPO family, plain `ppo` sits at or above the others; nothing here is tested.**
  - At `unseen_heading` SS6, `ppo` at 90.7 [86.3, 93.3] is above `residual_ppo` at 85.7
    [84.0, 89.0], but the seed CIs overlap. That is a non-result.
  - At `id` SS6, the H1b cell, `ppo` at 98.2 [98.0, 98.5] lies wholly above `residual_ppo` at 96.2
    [95.7, 97.8] and `ppo_sinusoid` at 97.3 [96.7, 97.8] (`results/e07/matrix/aggregate.csv`). This
    is an unpaired, untested reading. It bears on how H1b is read: residual RL beats its own PID base
    at SS6, while plain PPO sits above the residual in the same cell. *(Added at the Gate 7
    review.)*
- **SAC (2 M env steps against 10 M for the PPO family) is the weakest learned method everywhere
  past SS4.**
  - It falls to 46.5 % at `unseen_heading` SS6, with C86 and O98 per 1 000.
  - Its point estimate is below `pid_feedforward`'s in every SS5/SS6 cell of every regime.
  - At `id` SS5 and SS6 its seed CI lies wholly below `pid_feedforward`'s Wilson interval:
    [81.7, 90.3] against [96.4, 99.7], and [60.2, 79.3] against [85.6, 93.8].
  - **This is the clearest case of a classical baseline above an RL method.** It is still an
    unpaired reading; no paired SAC contrast was pre-registered.
- **`oracle_gated` is not a ceiling.** It is within a few points of `gated` in every cell, and both
  lose mainly to `timeout`. Its privilege buys commit timing, not success.

**Closing speed (unpaired reading; corrected at review m2).** Sources: `matrix/aggregate.csv`
(IQM of per-seed p95, m/s) and `carried_summary_e01.csv`, `carried_summary_e01_lowvz_cut.csv`.
Over the 14 cells (13 regime × SS cells and `static`):
- **Five baselines land softer than every PPO-family method in all 14 cells:** `pid_feedforward`,
  `pid_feedforward_lowvz`, `pid_feedforward_lowvz_cut`, `gated` and `oracle_gated`.
- **`pid_track_descend` is the exception.** It lands *harder* than every PPO-family method in 12
  of the 14 cells. It is softer than all of them only at `unseen_heading` SS3 (0.258 against
  0.263–0.278) and `static` (0.246 against 0.261–0.272).
- `sac` lands hardest of all in 13 of 14 cells. The exception is `unseen_vessel` SS5, where
  `pid_track_descend`'s 0.520 is above `sac`'s 0.461.
- At `id` SS5, for example: PPO family 0.271–0.282; `pid_feedforward` 0.262;
  `pid_feedforward_lowvz` 0.188; `pid_track_descend` 0.525; `sac` 0.595.

The only paired closing-speed tests are H1a and H3. The first finds the residual 47 % harder than
`lowvz`; the second finds no forecast effect.

**Tunnelled successes in the matrix (review M4).** Successes whose penetration exceeded 5 mm at any
contact substep, pooled over seeds, from `results/e07/matrix/tunnelled_success.csv`
(`n_success_tunnelled`; baselines carried from `results/e01` and `results/e01_lowvz_cut`). Counts
are per 1 000 seed-episodes for learned rows and per 200 for baselines. "us" is `unseen_seastate`,
"uh" `unseen_heading`, "uv" `unseen_vessel`.

| method | id SS3/4/5/6 | us SS6 | uh SS3/4/5/6 | uv SS3/4/5/6 | static | `id` audit bound (P5-D14 / P6-D5) |
|---|---|---|---|---|---|---|
| `ppo` | 0/0/1/3 | 8 | 0/0/0/25 | 0/0/0/2 | 0 | 2 (SS6) |
| `sac` (2 M steps) | 3/7/23/50 | 49 | 2/2/25/44 | 4/9/19/33 | 2 | 41 (1 / 3 / 13 / 24) |
| `residual_ppo` | 0/0/0/4 | 1 | 0/0/0/5 | 0/0/0/1 | 0 | 3 (SS6) |
| `ppo_forecast` (ideal feed) | 0/0/0/4 | 7 | 0/0/0/22 | 0/0/0/1 | not run | 1 (SS6) |
| `residual_ppo_forecast` (ideal feed) | 0/0/1/0 | 2 | 0/0/0/3 | 0/0/0/0 | not run | 0 (SS6) |
| `ppo_sinusoid` | 0/0/0/3 | 8 | 0/0/0/19 | 0/0/0/5 | 0 | 3 (SS6) |
| `pid_track_descend` | 0/0/0/0 | 0 | 0/0/0/2 | 0/0/0/0 | 0 | – |
| `pid_feedforward` | 0/0/0/0 | 0 | 0/0/0/0 | 0/0/0/0 | 0 | – |
| `pid_feedforward_lowvz` | 0/0/0/0 | 0 | 0/0/0/0 | 0/0/0/0 | 0 | – |
| `pid_feedforward_lowvz_cut` | 0/0/1/2 | 4 | 0/0/0/6 | 0/0/0/0 | 0 | 1 (SS6) |
| `gated` | 0/0/0/0 | 0 | 0/0/0/0 | 0/0/0/0 | 0 | – |
| `oracle_gated` (privileged) | 0/0/0/0 | 0 | 0/0/0/0 | 0/0/0/0 | 0 | – |

- **The two counts measure different things.**
  - The *direct count* is every success with penetration > 5 mm. Such a success may, but need
    not, depend on the overlap.
  - The *audit bound* is the subset that P5-D14 (`ppo`, `sac`, `pid_feedforward_lowvz_cut`) and
    P6-D5 (the four Phase 6 methods) judged possibly overlap-dependent: those with an unloaded
    stretch after contact (`possibly_dependent` in `results/audit/tunnelling_episodes.csv` and
    `results/audit/e06/tunnelling_episodes.csv`). The audits examined exactly the successes the
    direct count lists, so the bound is a subset of it, not a contradiction.
- **At `id` the direct count exceeds the bound** for `residual_ppo` (SS6: 4 against 3),
  `ppo_forecast` (SS6: 4 against 1), `ppo` (1 at SS5 + 3 at SS6 against 2), `sac` (83 against
  41) and `pid_feedforward_lowvz_cut` (1 + 2 against 1). `residual_ppo_forecast`'s single one is at
  SS5, outside P6-D5's SS6 bound. `ppo_sinusoid` equals its bound (3).
- **Outside `id` nothing is audited.** At `unseen_heading` SS6 the pure PPO methods have 25 / 22 /
  19 tunnelled successes per 1 000 (`ppo` / `ppo_forecast` / `ppo_sinusoid`), i.e. up to 2.5
  points of a cell. Their PPO-family lead over `pid_feedforward` there (85.7–90.7 % against
  77.0 %, unpaired) is larger than that.

### 3. Pad at CG: isolating the lever-arm route of dmf's phase defect (aft − CG)

The same listed episodes, with the pad moved to the ship's CG. The aft − CG success contrast is
paired over identical episodes, 10 000 replicates, in points
(`results/e07/contrasts.csv`, `cg.*`). Closing-speed p95 values are from `matrix/aggregate.csv` and
`cg/aggregate.csv` for the learned rows, and from `matrix/carried_summary_e01.csv`,
`cg/carried_summary_e02.csv` and `cg/baselines_summary.csv` for the baselines. Full tables,
outcome breakdowns included, are in `results/results.md` §2.

| method | `id` SS5 aft − CG (points) | `id` SS6 aft − CG (points) | p95 SS5 aft → CG (m/s) | p95 SS6 aft → CG (m/s) |
|---|---|---|---|---|
| `ppo` | +0.2 [+0.0, +0.8] | -0.5 [-1.7, +1.0] | 0.271 → 0.289 | 0.278 → 0.295 |
| `sac` (2 M steps) | +15.3 [+4.3, +24.7] **separates** | +23.8 [+11.3, +30.5] **separates** | 0.595 → 0.659 | 0.668 → 0.776 |
| `residual_ppo` | +0.3 [-0.2, +1.5] | +0.0 [-2.5, +2.8] | 0.277 → 0.287 | 0.283 → 0.305 |
| `ppo_forecast` (ideal feed) | +0.2 [+0.0, +1.2] | +0.3 [-0.8, +2.2] | 0.272 → 0.272 | 0.281 → 0.288 |
| `residual_ppo_forecast` (ideal feed) | +0.0 [-0.8, +0.7] | -0.2 [-2.0, +2.7] | 0.278 → 0.279 | 0.278 → 0.279 |
| `ppo_sinusoid` | +0.0 [+0.0, +0.5] | -0.2 [-1.3, +0.8] | 0.282 → 0.296 | 0.288 → 0.304 |
| `pid_track_descend` | -11.0 [-17.0, -5.5] **separates** | +3.0 [-4.0, +10.0] | 0.525 → 0.403 | 0.502 → 0.507 |
| `pid_feedforward` | +0.5 [-1.0, +2.0] | -1.5 [-6.0, +2.5] | 0.262 → 0.229 | 0.264 → 0.243 |
| `pid_feedforward_lowvz` | +0.5 [-3.0, +4.5] | +2.5 [-4.0, +9.0] | 0.188 → 0.131 | 0.181 → 0.144 |
| `pid_feedforward_lowvz_cut` | +1.5 [-1.0, +4.0] | +0.5 [-5.5, +6.5] | 0.188 → 0.131 | 0.181 → 0.144 |
| `gated` | -2.5 [-5.0, +0.0] | -3.0 [-6.5, +0.5] | 0.250 → 0.225 | 0.267 → 0.231 |
| `oracle_gated` (privileged) | -2.0 [-4.5, +0.0] | -1.0 [-5.0, +3.0] | 0.252 → 0.222 | 0.254 → 0.226 |

| method | `unseen_vessel` SS5 aft − CG (points) | `unseen_vessel` SS6 aft − CG (points) | p95 SS5 aft → CG (m/s) | p95 SS6 aft → CG (m/s) |
|---|---|---|---|---|
| `ppo` | +0.0 [-0.5, +0.0] | +0.0 [-1.5, +1.0] | 0.268 → 0.276 | 0.277 → 0.291 |
| `sac` (2 M steps) | +8.5 [+1.8, +16.3] **separates** | +17.0 [+8.2, +29.0] **separates** | 0.461 → 0.559 | 0.603 → 0.727 |
| `residual_ppo` | +0.0 [+0.0, +0.0] | -0.3 [-1.5, +0.7] | 0.273 → 0.281 | 0.278 → 0.301 |
| `ppo_forecast` (ideal feed) | +0.0 [+0.0, +0.0] | +0.2 [-1.0, +0.8] | 0.271 → 0.269 | 0.275 → 0.285 |
| `residual_ppo_forecast` (ideal feed) | +0.0 [+0.0, +0.0] | +0.3 [-0.8, +1.5] | 0.277 → 0.279 | 0.278 → 0.279 |
| `ppo_sinusoid` | +0.0 [+0.0, +0.0] | -0.2 [-1.0, +0.5] | 0.276 → 0.286 | 0.283 → 0.303 |
| `pid_track_descend` | -10.0 [-14.5, -6.0] **separates** | -13.5 [-20.0, -7.5] **separates** | 0.520 → 0.347 | 0.546 → 0.456 |
| `pid_feedforward` | -0.5 [-1.5, +0.0] | -0.5 [-2.5, +1.5] | 0.255 → 0.213 | 0.260 → 0.227 |
| `pid_feedforward_lowvz` | +2.0 [+0.0, +4.5] | +0.0 [-3.5, +4.0] | 0.155 → 0.122 | 0.159 → 0.141 |
| `pid_feedforward_lowvz_cut` | +2.0 [+0.0, +4.5] | +0.5 [-2.5, +4.0] | 0.155 → 0.122 | 0.159 → 0.141 |
| `gated` | -5.0 [-8.0, -2.0] **separates** | -1.0 [-4.5, +2.5] | 0.248 → 0.213 | 0.249 → 0.223 |
| `oracle_gated` (privileged) | -4.0 [-7.0, -1.5] **separates** | +0.5 [-3.5, +4.5] | 0.238 → 0.216 | 0.235 → 0.228 |

- **For the PPO family and the three `pid_feedforward` variants, moving the pad changes success by
  nothing measurable.**
  - None of the 65 PPO-family contrasts (5 methods × 13 cells) separates.
  - None of the 39 for `pid_feedforward`, `lowvz` and `lowvz_cut` separates.
  - Where both pads score 100 %, the CI is a degenerate [0, 0].
  - Through the lever arm, dmf's in-phase roll/pitch–heave defect does not move these methods'
    success. It is a non-result in 104 cells, not a demonstrated absence.
- **`pid_track_descend` lands better at CG.** The contrast is −11.0 [−17.0, −5.5] at `id` SS5, and
  −13.5 [−20.0, −7.5] at `unseen_vessel` SS6. That fits the calmer CG deck: its deck v_z p99 is
  0.232 against 0.344 m/s at frigate SS5 (`results/deck_stats.csv`).
- **Surprise 1: SAC is much worse at CG.**
  - The contrast is +23.8 [+11.3, +30.5] at `id` SS6, and separates in 5 of 13 cells. That is
    although the CG deck moves less.
  - Its CG losses at `id` SS6 are C48 O102 H327 per 1 000, against C18 O39 H232 aft.
  - It is not explained.
- **Surprise 2: at `id` SS6, the learned methods' p95 closing speed is higher at CG (for three of
  six it separates by seed CI), and most classical controllers land softer.** The exceptions are `pid_track_descend` (0.502 → 0.507 m/s) and `gated_forecast_tcn`
  (0.204 → 0.209), both slightly harder (`results/e02/summary.csv`). *(Corrected at the Phase 7
  re-review; the earlier wording said "every classical controller softer".)*
  - `ppo`'s p95 at `id` SS6 is 0.278 → 0.295 m/s, and `residual_ppo`'s 0.283 → 0.305.
  - `pid_feedforward`'s is 0.264 → 0.243.
  - This is an unpaired reading: no paired aft − CG closing-speed contrast was computed. The seed
    CIs of the p95 IQM separate between pads for `ppo` ([0.274, 0.285] vs [0.286, 0.304]),
    `residual_ppo` ([0.281, 0.287] vs [0.298, 0.311]) and `ppo_sinusoid` ([0.282, 0.292] vs
    [0.300, 0.311]) only. They overlap for `sac` ([0.590, 0.778] vs [0.710, 0.852]),
    `ppo_forecast` ([0.272, 0.295] vs [0.280, 0.300]) and `residual_ppo_forecast` ([0.274, 0.280]
    at both pads; points 0.278 vs 0.279) (`results/e07/{matrix,cg}/aggregate.csv`). *(Corrected at
    the Gate 7 review: the earlier text said no learned method's CIs overlapped.)*
  - All learned methods were trained at the aft pad only.
  - One untested explanation is that they learned aft-specific structure of the deck motion,
    including the defect's coupling of deck tilt to pad heave. It is only that: an explanation
    nobody has tested.

### 4. Perception stand-in (`id`, aft pad; P7-D4 definition, re-flown)

**What is perceived (P7-D4).** The stand-in perceives the *deck*.
- At control step k the observation is built from the true deck sample of step k − L.
  Zero-mean Gaussian noise, independent per world axis, is added to the pad position (σ_p) and
  the pad velocity (σ_v = σ_p / 0.2 s, i.e. 0.05 / 0.10 / 0.20 m/s model scale). The pad
  orientation and deck normal are delayed but get no noise.
- **Every deck-derived observation entry** is computed from that perceived sample and the
  drone's current true state: relative position, relative velocity, deck normal, relative tilt
  and pad-plane clearance. The drone's own state is clean and current.
- Latency is **0, 1 or 2 control steps**, i.e. 0 / 33.3 / 66.7 ms model or 0 / 167 / 333 ms full
  scale (`noise_latency_steps` and `noise_latency_ms_effective` in each
  `results/e07/noise/*/summary.csv`).
- **The noise is white** (P7-D6): drawn i.i.d. at every 33.3 ms control step, position and
  velocity independently (`rld.envs.noise.PerceptionNoise.perceive`). P7-D1 §4 sized σ_v as a
  relative-velocity estimate smoothed over about 1 s full scale, whose error would be correlated
  over about 6 control steps. **A correlated estimator error of the same σ was not tested.**
- **Noise against deck motion** (P7-D6). Deck SDs are the mean, over the 12 `id`-regime cells
  (frigate, headings 45 / 90 / 135 / 180°, speeds 0 / 6 / 12 kn), of the per-cell
  `z_std_model_m` and `vz_std_model_m_s` of the aft rows of `results/deck_stats.csv` (λ = 1/25,
  model scale; each cell over its 40 realizations). Restricting to the 384 realizations the `id`
  list draws from (seeds 32–39, `results/deck_stats_seeds.csv`) moves no value by more than
  0.0008 m/s or 0.13 mm. The noise is per world axis; the comparison is with the vertical deck
  motion only.

| sea state | deck z SD, cm (cell range) | deck v_z SD, m/s (cell range) | σ_p / z SD at σ_p = 1 / 2 / 4 cm | σ_v / v_z SD at σ_v = 0.05 / 0.10 / 0.20 m/s |
|---|---|---|---|---|
| SS3 | 1.04 (0.39–1.95) | 0.046 (0.015–0.084) | 0.96 / 1.93 / 3.86 | 1.08 / 2.16 / 4.31 |
| SS4 | 2.08 (1.23–3.42) | 0.086 (0.042–0.145) | 0.48 / 0.96 / 1.92 | 0.58 / 1.17 / 2.33 |
| SS5 | 3.41 (2.57–5.23) | 0.134 (0.070–0.227) | 0.29 / 0.59 / 1.17 | 0.37 / 0.74 / 1.49 |
| SS6 | 4.10 (2.71–5.39) | 0.147 (0.068–0.226) | 0.24 / 0.49 / 0.98 | 0.34 / 0.68 / 1.36 |

  - So the noise is largest against the deck motion at **SS3**: at 4 cm, σ_p is 3.9× and σ_v
    4.3× the deck SD there, against 1.2× and 1.5× at SS5. Under noise the calm sea state is not
    the easy case.
  - The earlier comparison with "about 0.36 m/s" (plan D0.1's scouting table, SS5 180° 12 kn)
    is withdrawn. P1-D2 showed that table used the wrong sign; the committed value of that one
    cell is 0.221 m/s, and the `id` SS5 mean is 0.134 m/s.
- **Two channels stay ideal.** The forecast methods' past-only ship-motion feed (P7-D1 §4), and
  `oracle_gated`'s privileged context, the true future deck trajectory
  (`PrivilegedContext.from_env`). Only the trajectory is ideal: where `oracle_gated` places its
  window, its at-hover check, its lateral gate and its tracking all read the perceived
  observation, like every other controller's (see below; P7-D6).
- **Superseded arm.** The arm first flown at `6b83e5c` under the Phase 2 stand-in is kept
  unchanged in `results/e07/noise_superseded_p7d1/`, and `scripts/eval_phase7.py --check`
  verifies its P7-D2 hashes. That stand-in delayed and noised only the six relative entries, so
  any latency mixed timestamps (P7-D4). Its numbers are used below only where they are named as
  the artifact P7-D4 corrected.

The re-flown arm (11 conditions, 16:45–19:12 EDT from `182cdea`; P7-D5) is shown in full, per
sea state and per condition. Sources: `results/e07/noise/<condition>/{aggregate,summary,baselines_summary}.csv`;
the clean column is the main matrix (`results/e07/matrix/aggregate.csv`, `summary.csv`,
`carried_summary_e01.csv`, `carried_summary_e01_lowvz_cut.csv`). The paired clean − noisy
contrasts are in `results/e07/contrasts.csv` (`noise/*`, 528 rows) and `results/results.md` §4.

Cell format: learned = IQM [95 % CI]; baselines = rate [Wilson 95 % CI] k/N; after the semicolon, losses by class (C crash, O off_pad, H hard_landing, B bounce, T timeout; counts of 1 000 for learned rows, of 200 for baselines; – = none).

#### `id` SS3: clean, σp alone, latency alone

| method | clean | σp 1 cm, 0 steps | σp 2 cm, 0 steps | σp 4 cm, 0 steps | σp 0, 1 step (33.3 ms) | σp 0, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.2 [98.7, 99.8]; H5 B3 | 79.8 [72.3, 89.7]; O44 H76 B77 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `sac` (2 M steps) | 99.8 [98.8, 100.0]; H4 | 92.5 [88.5, 97.5]; O8 H63 B2 | 64.0 [57.2, 81.3]; C7 O47 H271 B7 | 19.2 [11.7, 32.7]; C96 O228 H430 B32 T4 | 99.8 [98.2, 100.0]; H5 B1 | 100.0 [98.0, 100.0]; H6 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 77.0 [75.5, 81.2]; C2 O12 H60 B148 | 2.5 [2.2, 4.2]; C170 O311 H302 B187 T1 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 96.3 [93.7, 98.2]; O2 H11 B25 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 87.5 [81.5, 92.0]; O6 H31 B93 | 5.0 [3.0, 9.7]; C93 O313 H290 B246 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.0 [98.2, 99.5]; O1 H3 B7 | 69.7 [58.3, 80.8]; C1 O55 H88 B158 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200; – | 99.0 [96.4, 99.7] 198/200; B2 | 86.0 [80.5, 90.1] 172/200; B28 | 21.5 [16.4, 27.7] 43/200; B95 T62 | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 98.5 [95.7, 99.5] 197/200; B3 | 60.0 [53.1, 66.5] 120/200; O1 H5 B74 | 0.0 [0.0, 1.9] 0/200; C90 O34 H20 B13 T43 | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – |
| `pid_feedforward_lowvz` | 100.0 [98.1, 100.0] 200/200; – | 97.0 [93.6, 98.6] 194/200; B6 | 66.5 [59.7, 72.7] 133/200; H1 B66 | 7.5 [4.6, 12.0] 15/200; C8 O30 H46 B87 T14 | 99.5 [97.2, 99.9] 199/200; B1 | 98.0 [95.0, 99.2] 196/200; B4 |
| `pid_feedforward_lowvz_cut` | 100.0 [98.1, 100.0] 200/200; – | 99.0 [96.4, 99.7] 198/200; B2 | 90.5 [85.6, 93.8] 181/200; H1 B18 | 41.5 [34.9, 48.4] 83/200; C8 O30 H46 B19 T14 | 100.0 [98.1, 100.0] 200/200; – | 98.5 [95.7, 99.5] 197/200; B3 |
| `gated` | 100.0 [98.1, 100.0] 200/200; – | 96.5 [93.0, 98.3] 193/200; B7 | 16.5 [12.0, 22.3] 33/200; B31 T136 | 0.0 [0.0, 1.9] 0/200; C120 T80 | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – |
| `oracle_gated` (privileged) | 100.0 [98.1, 100.0] 200/200; – | 97.5 [94.3, 98.9] 195/200; B5 | 30.5 [24.5, 37.2] 61/200; H2 B66 T71 | 0.0 [0.0, 1.9] 0/200; C120 T80 | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – |

#### `id` SS3: σp with latency

| method | clean | σp 1 cm, 1 step (33.3 ms) | σp 2 cm, 1 step (33.3 ms) | σp 4 cm, 1 step (33.3 ms) | σp 1 cm, 2 steps (66.7 ms) | σp 2 cm, 2 steps (66.7 ms) | σp 4 cm, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.2 [99.0, 99.8]; H2 B5 | 79.7 [76.7, 87.5]; C1 O40 H78 B71 | 100.0 [100.0, 100.0]; – | 99.5 [98.3, 100.0]; B7 | 80.0 [74.5, 87.8]; C1 O49 H71 B71 |
| `sac` (2 M steps) | 99.8 [98.8, 100.0]; H4 | 92.5 [88.8, 97.0]; O3 H71 | 61.5 [55.2, 76.2]; C7 O60 H289 B6 T2 | 18.2 [13.0, 37.0]; C114 O217 H406 B34 T7 | 92.2 [88.0, 96.5]; O5 H73 | 61.8 [52.7, 78.8]; C4 O50 H297 B6 | 20.5 [12.2, 33.7]; C111 O228 H390 B44 T8 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 80.5 [76.3, 83.5]; C3 O8 H43 B144 | 2.7 [1.3, 6.0]; C142 O329 H295 B201 | 100.0 [100.0, 100.0]; – | 78.3 [74.8, 83.2]; C2 O8 H42 B162 | 3.3 [2.2, 5.3]; C125 O311 H302 B226 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 95.7 [93.3, 97.8]; O4 H15 B25 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 97.0 [95.8, 98.2]; C1 O3 H12 B14 |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 99.8 [99.5, 100.0]; C1 B1 | 83.3 [79.0, 91.5]; C1 O5 H39 B111 | 4.3 [2.8, 8.2]; C95 O312 H298 B245 | 100.0 [99.7, 100.0]; C1 | 84.8 [82.5, 91.3]; C1 O8 H24 B106 | 4.2 [2.5, 9.8]; C77 O317 H313 B240 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 98.8 [98.5, 99.0]; O2 H2 B8 | 72.7 [57.2, 82.2]; O51 H94 B144 | 100.0 [100.0, 100.0]; – | 99.3 [98.0, 99.8]; O1 H1 B7 | 72.8 [61.0, 85.0]; C3 O42 H97 B129 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200; – | 97.0 [93.6, 98.6] 194/200; B6 | 85.5 [80.0, 89.7] 171/200; B29 | 19.5 [14.6, 25.5] 39/200; B92 T69 | 97.0 [93.6, 98.6] 194/200; B6 | 81.0 [75.0, 85.8] 162/200; B38 | 16.5 [12.0, 22.3] 33/200; B90 T77 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 98.5 [95.7, 99.5] 197/200; B3 | 63.0 [56.1, 69.4] 126/200; O1 H8 B65 | 0.5 [0.1, 2.8] 1/200; C80 O30 H15 B16 T58 | 99.5 [97.2, 99.9] 199/200; B1 | 63.0 [56.1, 69.4] 126/200; C1 H8 B65 | 0.0 [0.0, 1.9] 0/200; C79 O20 H20 B11 T70 |
| `pid_feedforward_lowvz` | 100.0 [98.1, 100.0] 200/200; – | 97.0 [93.6, 98.6] 194/200; B6 | 71.5 [64.9, 77.3] 143/200; B57 | 8.5 [5.4, 13.2] 17/200; C13 O37 H38 B85 T10 | 94.5 [90.4, 96.9] 189/200; B11 | 61.5 [54.6, 68.0] 123/200; H1 B76 | 8.5 [5.4, 13.2] 17/200; C3 O38 H42 B87 T13 |
| `pid_feedforward_lowvz_cut` | 100.0 [98.1, 100.0] 200/200; – | 98.5 [95.7, 99.5] 197/200; B3 | 92.5 [88.0, 95.4] 185/200; B15 | 42.5 [35.9, 49.4] 85/200; C13 O37 H38 B17 T10 | 97.5 [94.3, 98.9] 195/200; B5 | 90.5 [85.6, 93.8] 181/200; H1 B18 | 40.5 [33.9, 47.4] 81/200; C3 O38 H42 B23 T13 |
| `gated` | 100.0 [98.1, 100.0] 200/200; – | 99.0 [96.4, 99.7] 198/200; B2 | 11.5 [7.8, 16.7] 23/200; B39 T138 | 0.0 [0.0, 1.9] 0/200; C118 T82 | 97.0 [93.6, 98.6] 194/200; B6 | 13.0 [9.0, 18.4] 26/200; H1 B39 T134 | 0.0 [0.0, 1.9] 0/200; C118 T82 |
| `oracle_gated` (privileged) | 100.0 [98.1, 100.0] 200/200; – | 98.0 [95.0, 99.2] 196/200; B4 | 28.0 [22.2, 34.6] 56/200; C1 H1 B56 T86 | 0.0 [0.0, 1.9] 0/200; C125 T75 | 98.0 [95.0, 99.2] 196/200; B4 | 29.5 [23.6, 36.2] 59/200; C1 H2 B59 T79 | 0.0 [0.0, 1.9] 0/200; C136 T64 |

#### `id` SS4: clean, σp alone, latency alone

| method | clean | σp 1 cm, 0 steps | σp 2 cm, 0 steps | σp 4 cm, 0 steps | σp 0, 1 step (33.3 ms) | σp 0, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.5 [99.2, 99.8]; B5 | 81.5 [75.3, 89.3]; O42 H68 B69 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `sac` (2 M steps) | 98.2 [96.3, 99.0]; O1 H20 | 89.7 [86.0, 93.3]; O7 H93 B4 | 59.8 [52.0, 76.3]; C9 O58 H296 B11 | 16.5 [13.5, 30.0]; C113 O230 H426 B31 T5 | 98.0 [96.8, 99.2]; H20 | 97.7 [96.3, 98.7]; O1 H23 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 81.2 [75.2, 84.3]; C2 O9 H49 B136 | 3.0 [0.8, 5.3]; C152 O311 H310 B194 T2 | 100.0 [99.7, 100.0]; B1 | 100.0 [99.3, 100.0]; B2 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 95.2 [91.8, 98.3]; O8 H19 B23 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 84.5 [79.7, 92.0]; C4 O6 H34 B104 | 4.3 [2.3, 10.2]; C85 O349 H296 B215 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 98.8 [98.0, 99.5]; O2 H4 B6 | 73.5 [62.0, 84.0]; C4 O49 H91 B126 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `pid_track_descend` | 96.0 [92.3, 98.0] 192/200; H4 B4 | 95.5 [91.7, 97.6] 191/200; H3 B6 | 79.0 [72.8, 84.1] 158/200; H3 B39 | 22.5 [17.3, 28.8] 45/200; B96 T59 | 95.5 [91.7, 97.6] 191/200; H4 B5 | 95.5 [91.7, 97.6] 191/200; H5 B4 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 97.5 [94.3, 98.9] 195/200; B5 | 66.0 [59.2, 72.2] 132/200; O1 H5 B62 | 0.0 [0.0, 1.9] 0/200; C80 O28 H19 B11 T62 | 100.0 [98.1, 100.0] 200/200; – | 99.5 [97.2, 99.9] 199/200; B1 |
| `pid_feedforward_lowvz` | 98.0 [95.0, 99.2] 196/200; B4 | 96.5 [93.0, 98.3] 193/200; B7 | 65.0 [58.2, 71.3] 130/200; O1 H2 B67 | 5.5 [3.1, 9.6] 11/200; C13 O39 H37 B89 T11 | 96.5 [93.0, 98.3] 193/200; B7 | 88.5 [83.3, 92.2] 177/200; B23 |
| `pid_feedforward_lowvz_cut` | 98.5 [95.7, 99.5] 197/200; B3 | 98.5 [95.7, 99.5] 197/200; B3 | 85.5 [80.0, 89.7] 171/200; O1 H2 B26 | 42.5 [35.9, 49.4] 85/200; C13 O39 H37 B15 T11 | 98.5 [95.7, 99.5] 197/200; B3 | 98.5 [95.7, 99.5] 197/200; B3 |
| `gated` | 98.5 [95.7, 99.5] 197/200; T3 | 94.5 [90.4, 96.9] 189/200; B4 T7 | 15.5 [11.1, 21.2] 31/200; H2 B20 T147 | 0.0 [0.0, 1.9] 0/200; C121 T79 | 97.5 [94.3, 98.9] 195/200; B1 T4 | 98.0 [95.0, 99.2] 196/200; T4 |
| `oracle_gated` (privileged) | 98.5 [95.7, 99.5] 197/200; T3 | 95.5 [91.7, 97.6] 191/200; B6 T3 | 27.0 [21.3, 33.5] 54/200; H1 B50 T95 | 0.0 [0.0, 1.9] 0/200; C125 T75 | 99.0 [96.4, 99.7] 198/200; T2 | 99.0 [96.4, 99.7] 198/200; T2 |

#### `id` SS4: σp with latency

| method | clean | σp 1 cm, 1 step (33.3 ms) | σp 2 cm, 1 step (33.3 ms) | σp 4 cm, 1 step (33.3 ms) | σp 1 cm, 2 steps (66.7 ms) | σp 2 cm, 2 steps (66.7 ms) | σp 4 cm, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 99.2 [98.7, 99.5]; H2 B7 | 79.8 [72.2, 89.7]; O37 H77 B82 | 100.0 [100.0, 100.0]; – | 99.3 [98.7, 99.8]; H2 B5 | 80.2 [73.5, 84.0]; C2 O38 H83 B83 |
| `sac` (2 M steps) | 98.2 [96.3, 99.0]; O1 H20 | 88.5 [85.2, 95.3]; C1 O8 H91 B4 | 62.0 [57.3, 75.7]; C5 O46 H297 B5 | 19.3 [14.8, 35.5]; C104 O241 H395 B29 T4 | 87.5 [84.7, 94.3]; C2 O6 H105 B1 | 57.2 [48.7, 71.7]; C8 O66 H325 B13 | 18.3 [12.8, 33.5]; C119 O222 H414 B33 T1 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 99.7 [99.2, 100.0]; H1 B3 | 79.3 [75.0, 85.7]; C3 O4 H48 B145 | 2.3 [1.2, 5.2]; C141 O340 H298 B193 | 99.8 [99.2, 100.0]; H1 B2 | 79.5 [76.7, 81.8]; C4 O11 H36 B155 | 4.7 [3.0, 6.2]; C121 O334 H305 B194 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 96.2 [95.0, 98.0]; O7 H10 B20 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 95.3 [93.7, 97.2]; O8 H14 B24 |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; H1 | 85.7 [79.2, 91.2]; C3 O8 H31 B105 | 4.7 [3.8, 9.7]; C88 O307 H324 B222 | 100.0 [100.0, 100.0]; – | 85.8 [81.0, 88.7]; C1 O11 H36 B98 | 5.7 [4.7, 9.5]; C113 O298 H308 B216 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.3 [98.2, 100.0]; H2 B6 | 72.7 [64.2, 78.5]; O43 H111 B124 | 100.0 [100.0, 100.0]; – | 99.2 [97.8, 99.8]; H1 B9 | 70.8 [59.0, 79.5]; O60 H106 B132 |
| `pid_track_descend` | 96.0 [92.3, 98.0] 192/200; H4 B4 | 96.0 [92.3, 98.0] 192/200; H2 B6 | 82.5 [76.6, 87.1] 165/200; H2 B33 | 22.0 [16.8, 28.2] 44/200; B88 T68 | 94.0 [89.8, 96.5] 188/200; H3 B9 | 78.5 [72.3, 83.6] 157/200; H3 B40 | 24.0 [18.6, 30.4] 48/200; B83 T69 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 98.5 [95.7, 99.5] 197/200; B3 | 62.0 [55.1, 68.4] 124/200; H4 B72 | 0.5 [0.1, 2.8] 1/200; C82 O32 H13 B10 T62 | 98.5 [95.7, 99.5] 197/200; B3 | 64.5 [57.7, 70.8] 129/200; C1 O1 H9 B60 | 0.0 [0.0, 1.9] 0/200; C84 O28 H18 B10 T60 |
| `pid_feedforward_lowvz` | 98.0 [95.0, 99.2] 196/200; B4 | 90.0 [85.1, 93.4] 180/200; B20 | 65.5 [58.7, 71.7] 131/200; H1 B68 | 6.5 [3.8, 10.8] 13/200; C12 O28 H38 B99 T10 | 89.0 [83.9, 92.6] 178/200; B22 | 56.5 [49.6, 63.2] 113/200; O1 H4 B82 | 6.5 [3.8, 10.8] 13/200; C11 O46 H40 B84 T6 |
| `pid_feedforward_lowvz_cut` | 98.5 [95.7, 99.5] 197/200; B3 | 96.0 [92.3, 98.0] 192/200; B8 | 88.5 [83.3, 92.2] 177/200; H1 B22 | 42.5 [35.9, 49.4] 85/200; C12 O28 H38 B27 T10 | 99.0 [96.4, 99.7] 198/200; B2 | 88.5 [83.3, 92.2] 177/200; O1 H4 B18 | 40.0 [33.5, 46.9] 80/200; C11 O46 H40 B17 T6 |
| `gated` | 98.5 [95.7, 99.5] 197/200; T3 | 94.5 [90.4, 96.9] 189/200; B5 T6 | 9.5 [6.2, 14.4] 19/200; H2 B26 T153 | 0.0 [0.0, 1.9] 0/200; C135 O1 T64 | 95.0 [91.0, 97.3] 190/200; B3 T7 | 13.5 [9.4, 18.9] 27/200; B30 T143 | 0.0 [0.0, 1.9] 0/200; C120 T80 |
| `oracle_gated` (privileged) | 98.5 [95.7, 99.5] 197/200; T3 | 96.0 [92.3, 98.0] 192/200; B4 T4 | 25.5 [20.0, 32.0] 51/200; C1 B47 T101 | 0.0 [0.0, 1.9] 0/200; C131 T69 | 96.0 [92.3, 98.0] 192/200; B4 T4 | 26.0 [20.4, 32.5] 52/200; H2 B61 T85 | 0.0 [0.0, 1.9] 0/200; C121 T79 |

#### `id` SS5: clean, σp alone, latency alone

| method | clean | σp 1 cm, 0 steps | σp 2 cm, 0 steps | σp 4 cm, 0 steps | σp 0, 1 step (33.3 ms) | σp 0, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 98.2 [96.8, 99.3]; H5 B14 | 79.2 [72.7, 84.7]; O48 H66 B97 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `sac` (2 M steps) | 87.0 [81.7, 90.3]; C2 O8 H122 B3 T1 | 77.8 [73.0, 83.3]; C5 O27 H180 B8 | 50.2 [44.7, 65.7]; C16 O95 H345 B10 T1 | 15.8 [14.3, 26.0]; C163 O240 H371 B33 T9 | 87.3 [81.8, 88.3]; C3 O15 H121 B1 | 84.3 [78.3, 89.3]; C6 O14 H142 |
| `residual_ppo` | 99.5 [99.5, 99.8]; H4 | 98.8 [98.2, 99.0]; H11 B2 | 79.7 [73.0, 85.2]; C1 O10 H66 B131 | 2.7 [2.2, 4.3]; C140 O343 H299 B188 | 99.3 [98.7, 99.8]; H4 B3 | 99.0 [99.0, 99.3]; H6 B3 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 100.0 [100.0, 100.0]; – | 95.8 [94.3, 97.2]; O2 H24 B16 | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 |
| `residual_ppo_forecast` (ideal feed) | 99.5 [99.5, 99.5]; H5 | 98.8 [98.5, 99.5]; C1 O1 H7 B2 | 82.2 [76.5, 86.7]; C1 O9 H50 B120 | 3.5 [2.5, 8.5]; C83 O312 H313 B245 | 99.5 [99.2, 99.5]; H5 B1 | 99.5 [99.2, 99.5]; H5 B1 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.2 [96.7, 100.0]; H7 B6 | 70.0 [56.2, 76.8]; C3 O54 H118 B143 | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – |
| `pid_track_descend` | 85.0 [79.4, 89.3] 170/200; H12 B18 | 83.0 [77.2, 87.6] 166/200; H12 B22 | 71.0 [64.4, 76.8] 142/200; H11 B47 | 22.0 [16.8, 28.2] 44/200; H4 B90 T62 | 86.0 [80.5, 90.1] 172/200; H10 B18 | 84.5 [78.8, 88.9] 169/200; H11 B20 |
| `pid_feedforward` | 99.0 [96.4, 99.7] 198/200; H1 B1 | 93.5 [89.2, 96.2] 187/200; H3 B10 | 56.0 [49.1, 62.7] 112/200; H5 B83 | 0.0 [0.0, 1.9] 0/200; C71 O34 H20 B13 T62 | 96.5 [93.0, 98.3] 193/200; H1 B6 | 92.5 [88.0, 95.4] 185/200; H2 B13 |
| `pid_feedforward_lowvz` | 95.5 [91.7, 97.6] 191/200; B9 | 88.5 [83.3, 92.2] 177/200; B23 | 58.0 [51.1, 64.6] 116/200; O2 H4 B78 | 9.0 [5.8, 13.8] 18/200; C5 O34 H39 B94 T10 | 88.5 [83.3, 92.2] 177/200; B23 | 80.5 [74.5, 85.4] 161/200; H1 B38 |
| `pid_feedforward_lowvz_cut` | 98.0 [95.0, 99.2] 196/200; B4 | 95.5 [91.7, 97.6] 191/200; B9 | 83.0 [77.2, 87.6] 166/200; O2 H4 B28 | 46.5 [39.7, 53.4] 93/200; C7 O32 H39 B19 T10 | 95.5 [91.7, 97.6] 191/200; B9 | 94.0 [89.8, 96.5] 188/200; H1 B11 |
| `gated` | 89.5 [84.5, 93.0] 179/200; T21 | 81.0 [75.0, 85.8] 162/200; B8 T30 | 3.0 [1.4, 6.4] 6/200; C1 B15 T178 | 0.0 [0.0, 1.9] 0/200; C131 T69 | 88.0 [82.8, 91.8] 176/200; B4 T20 | 87.0 [81.6, 91.0] 174/200; B6 T20 |
| `oracle_gated` (privileged) | 89.0 [83.9, 92.6] 178/200; B1 T21 | 86.0 [80.5, 90.1] 172/200; B3 T25 | 16.5 [12.0, 22.3] 33/200; C1 B38 T128 | 0.0 [0.0, 1.9] 0/200; C135 T65 | 89.5 [84.5, 93.0] 179/200; T21 | 88.0 [82.8, 91.8] 176/200; B1 T23 |

#### `id` SS5: σp with latency

| method | clean | σp 1 cm, 1 step (33.3 ms) | σp 2 cm, 1 step (33.3 ms) | σp 4 cm, 1 step (33.3 ms) | σp 1 cm, 2 steps (66.7 ms) | σp 2 cm, 2 steps (66.7 ms) | σp 4 cm, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 99.3 [99.0, 99.8]; H3 B3 | 98.5 [98.2, 99.2]; H3 B11 | 73.7 [68.3, 85.7]; C1 O36 H98 B109 | 99.8 [99.2, 100.0]; H1 B2 | 98.5 [97.3, 99.3]; H1 B15 | 76.5 [67.5, 84.3]; O44 H93 B100 |
| `sac` (2 M steps) | 87.0 [81.7, 90.3]; C2 O8 H122 B3 T1 | 76.5 [72.0, 81.7]; C10 O31 H188 B3 | 51.5 [43.8, 61.5]; C17 O98 H350 B13 T1 | 13.7 [8.3, 29.7]; C130 O291 H374 B29 T7 | 76.3 [69.0, 84.0]; C5 O27 H198 B4 T3 | 47.5 [42.5, 57.7]; C21 O74 H401 B13 T1 | 12.5 [10.7, 24.5]; C153 O274 H382 B32 T3 |
| `residual_ppo` | 99.5 [99.5, 99.8]; H4 | 99.0 [98.2, 99.5]; H9 B2 | 77.7 [74.8, 82.5]; C1 O7 H65 B146 | 3.0 [1.8, 4.7]; C152 O350 H283 B184 | 98.5 [97.2, 99.0]; H11 B6 | 75.2 [73.5, 77.2]; C4 O13 H80 B150 | 3.2 [1.0, 4.0]; C144 O322 H301 B205 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.7 [99.0, 100.0]; H2 B2 | 95.5 [93.8, 97.2]; O7 H18 B20 | 100.0 [100.0, 100.0]; – | 99.8 [99.5, 100.0]; B2 | 96.2 [94.5, 97.8]; O7 H11 B20 |
| `residual_ppo_forecast` (ideal feed) | 99.5 [99.5, 99.5]; H5 | 98.8 [98.5, 99.3]; H6 B5 | 83.5 [80.8, 87.5]; C1 O3 H46 B110 | 5.2 [3.8, 8.7]; C104 O318 H288 B232 | 98.3 [98.0, 99.2]; H6 B9 | 81.7 [80.2, 87.0]; O6 H51 B115 | 5.3 [3.3, 6.8]; C102 O311 H300 B235 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 97.2 [96.7, 97.8]; O1 H7 B20 | 71.0 [53.5, 82.2]; C2 O53 H98 B152 | 100.0 [100.0, 100.0]; – | 97.3 [93.8, 98.8]; H9 B23 | 69.2 [52.7, 79.5]; C1 O53 H116 B156 |
| `pid_track_descend` | 85.0 [79.4, 89.3] 170/200; H12 B18 | 82.0 [76.1, 86.7] 164/200; H14 B22 | 74.0 [67.5, 79.6] 148/200; H10 B42 | 22.0 [16.8, 28.2] 44/200; H6 B97 T53 | 83.5 [77.7, 88.0] 167/200; H13 B20 | 72.5 [65.9, 78.2] 145/200; H10 B45 | 20.5 [15.5, 26.6] 41/200; H7 B95 T57 |
| `pid_feedforward` | 99.0 [96.4, 99.7] 198/200; H1 B1 | 95.0 [91.0, 97.3] 190/200; B10 | 61.0 [54.1, 67.5] 122/200; H9 B69 | 0.0 [0.0, 1.9] 0/200; C86 O29 H18 B16 T51 | 90.0 [85.1, 93.4] 180/200; H3 B17 | 61.5 [54.6, 68.0] 123/200; H8 B69 | 0.5 [0.1, 2.8] 1/200; C71 O32 H22 B12 T62 |
| `pid_feedforward_lowvz` | 95.5 [91.7, 97.6] 191/200; B9 | 83.5 [77.7, 88.0] 167/200; H2 B31 | 59.5 [52.6, 66.1] 119/200; O1 H7 B73 | 6.5 [3.8, 10.8] 13/200; C7 O46 H39 B82 T13 | 70.5 [63.8, 76.4] 141/200; B59 | 56.5 [49.6, 63.2] 113/200; H3 B84 | 12.0 [8.2, 17.2] 24/200; C5 O42 H49 B69 T11 |
| `pid_feedforward_lowvz_cut` | 98.0 [95.0, 99.2] 196/200; B4 | 96.0 [92.3, 98.0] 192/200; H2 B6 | 88.0 [82.8, 91.8] 176/200; O1 H7 B16 | 43.0 [36.3, 49.9] 86/200; C7 O46 H39 B9 T13 | 93.0 [88.6, 95.8] 186/200; B14 | 91.5 [86.8, 94.6] 183/200; H3 B14 | 40.5 [33.9, 47.4] 81/200; C5 O42 H49 B12 T11 |
| `gated` | 89.5 [84.5, 93.0] 179/200; T21 | 82.5 [76.6, 87.1] 165/200; B5 T30 | 5.5 [3.1, 9.6] 11/200; B12 T177 | 0.0 [0.0, 1.9] 0/200; C121 T79 | 76.5 [70.2, 81.8] 153/200; H3 B14 T30 | 2.0 [0.8, 5.0] 4/200; B8 T188 | 0.0 [0.0, 1.9] 0/200; C126 T74 |
| `oracle_gated` (privileged) | 89.0 [83.9, 92.6] 178/200; B1 T21 | 85.0 [79.4, 89.3] 170/200; B5 T25 | 16.5 [12.0, 22.3] 33/200; H1 B29 T137 | 0.0 [0.0, 1.9] 0/200; C118 T82 | 83.5 [77.7, 88.0] 167/200; B8 T25 | 12.0 [8.2, 17.2] 24/200; H2 B36 T138 | 0.0 [0.0, 1.9] 0/200; C136 O1 T63 |

#### `id` SS6: clean, σp alone, latency alone

| method | clean | σp 1 cm, 0 steps | σp 2 cm, 0 steps | σp 4 cm, 0 steps | σp 0, 1 step (33.3 ms) | σp 0, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|
| `ppo` | 98.2 [98.0, 98.5]; H15 B3 | 98.5 [97.3, 99.0]; H15 B2 | 96.2 [94.5, 97.7]; O2 H20 B17 | 75.7 [69.3, 81.5]; O46 H128 B71 | 98.3 [97.7, 99.2]; H13 B3 | 98.3 [97.7, 98.8]; H12 B5 |
| `sac` (2 M steps) | 72.5 [60.2, 79.3]; C18 O39 H232 B2 | 66.8 [58.7, 73.3]; C28 O63 H232 B13 T1 | 40.3 [33.8, 51.3]; C41 O129 H389 B22 T2 | 10.5 [4.2, 17.8]; C226 O301 H318 B33 T13 | 74.0 [62.0, 78.8]; C21 O38 H217 B6 | 67.2 [59.0, 76.3]; C17 O37 H264 B7 T2 |
| `residual_ppo` | 96.2 [95.7, 97.8]; H24 B11 | 95.0 [95.0, 95.7]; H38 B10 | 74.5 [72.0, 78.3]; C2 O3 H108 B138 | 3.2 [2.5, 6.7]; C148 O310 H326 B176 | 96.7 [94.8, 97.5]; H24 B12 | 96.3 [94.5, 97.8]; H26 B11 |
| `ppo_forecast` (ideal feed) | 98.0 [97.5, 98.5]; H18 B2 | 97.7 [97.5, 98.3]; H18 B4 | 96.7 [95.8, 97.3]; H28 B6 | 89.8 [89.2, 92.2]; O6 H66 B25 | 98.2 [97.7, 98.8]; H16 B2 | 98.5 [98.2, 98.5]; H16 |
| `residual_ppo_forecast` (ideal feed) | 96.8 [96.5, 98.0]; H25 B4 | 95.2 [93.8, 96.3]; H38 B11 | 75.7 [69.7, 83.0]; C3 O6 H110 B120 | 5.5 [4.0, 11.2]; C107 O324 H312 B189 | 97.2 [96.3, 97.5]; H16 B14 | 95.8 [95.0, 97.8]; H25 B13 |
| `ppo_sinusoid` | 97.3 [96.7, 97.8]; H25 B2 | 97.5 [97.2, 97.8]; H24 B1 | 95.7 [91.8, 96.8]; H38 B13 | 68.5 [52.5, 74.3]; C1 O61 H168 B113 | 98.0 [97.2, 98.5]; H21 | 97.7 [97.2, 98.3]; H20 B3 |
| `pid_track_descend` | 80.0 [73.9, 85.0] 160/200; H17 B23 | 77.5 [71.2, 82.7] 155/200; H14 B31 | 64.0 [57.1, 70.3] 128/200; H20 B52 | 24.5 [19.1, 30.9] 49/200; H5 B92 T54 | 78.5 [72.3, 83.6] 157/200; H21 B22 | 80.0 [73.9, 85.0] 160/200; H25 B15 |
| `pid_feedforward` | 90.5 [85.6, 93.8] 181/200; H8 B11 | 86.0 [80.5, 90.1] 172/200; H7 B21 | 59.5 [52.6, 66.1] 119/200; H13 B68 | 0.5 [0.1, 2.8] 1/200; C96 O37 H15 B12 T39 | 93.5 [89.2, 96.2] 187/200; H6 B7 | 88.5 [83.3, 92.2] 177/200; H8 B15 |
| `pid_feedforward_lowvz` | 85.0 [79.4, 89.3] 170/200; H9 B21 | 79.0 [72.8, 84.1] 158/200; H7 B35 | 59.0 [52.1, 65.6] 118/200; H13 B69 | 6.0 [3.5, 10.2] 12/200; C5 O41 H47 B81 T14 | 81.0 [75.0, 85.8] 162/200; H7 B31 | 78.0 [71.8, 83.2] 156/200; H9 B35 |
| `pid_feedforward_lowvz_cut` | 88.0 [82.8, 91.8] 176/200; H9 B15 | 88.0 [82.8, 91.8] 176/200; H7 B17 | 84.0 [78.3, 88.4] 168/200; H13 B19 | 39.0 [32.5, 45.9] 78/200; C5 O41 H47 B15 T14 | 90.5 [85.6, 93.8] 181/200; H7 B12 | 91.5 [86.8, 94.6] 183/200; H9 B8 |
| `gated` | 63.0 [56.1, 69.4] 126/200; B4 T70 | 51.0 [44.1, 57.8] 102/200; B10 T88 | 2.0 [0.8, 5.0] 4/200; B3 T193 | 0.0 [0.0, 1.9] 0/200; C132 T68 | 63.0 [56.1, 69.4] 126/200; B7 T67 | 60.5 [53.6, 67.0] 121/200; H1 B9 T69 |
| `oracle_gated` (privileged) | 64.5 [57.7, 70.8] 129/200; T71 | 61.5 [54.6, 68.0] 123/200; B3 T74 | 9.5 [6.2, 14.4] 19/200; H1 B10 T170 | 0.0 [0.0, 1.9] 0/200; C125 O1 T74 | 64.0 [57.1, 70.3] 128/200; T72 | 64.0 [57.1, 70.3] 128/200; B2 T70 |

#### `id` SS6: σp with latency

| method | clean | σp 1 cm, 1 step (33.3 ms) | σp 2 cm, 1 step (33.3 ms) | σp 4 cm, 1 step (33.3 ms) | σp 1 cm, 2 steps (66.7 ms) | σp 2 cm, 2 steps (66.7 ms) | σp 4 cm, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|---|
| `ppo` | 98.2 [98.0, 98.5]; H15 B3 | 98.2 [97.3, 98.5]; H14 B6 | 95.7 [94.3, 96.8]; O3 H23 B18 | 75.8 [69.8, 84.2]; C1 O42 H120 B73 | 98.2 [96.8, 99.0]; H15 B5 | 95.0 [94.5, 96.8]; O1 H25 B20 | 73.2 [68.5, 79.5]; O44 H143 B77 |
| `sac` (2 M steps) | 72.5 [60.2, 79.3]; C18 O39 H232 B2 | 62.3 [61.0, 68.8]; C27 O68 H257 B9 | 37.8 [34.2, 51.7]; C47 O126 H396 B20 T2 | 9.5 [6.2, 16.2]; C223 O288 H352 B22 T10 | 59.3 [52.3, 65.3]; C29 O61 H307 B11 T2 | 39.7 [31.3, 52.5]; C42 O125 H408 B15 T1 | 9.8 [6.7, 15.2]; C195 O317 H341 B25 T17 |
| `residual_ppo` | 96.2 [95.7, 97.8]; H24 B11 | 95.2 [94.3, 95.8]; H41 B8 | 72.7 [70.8, 75.7]; C2 O8 H108 B152 | 3.3 [2.3, 7.0]; C157 O312 H308 B182 | 94.5 [93.2, 95.3]; H41 B15 | 73.2 [67.2, 77.3]; C1 O9 H115 B149 | 3.7 [3.0, 5.2]; C156 O332 H294 B179 |
| `ppo_forecast` (ideal feed) | 98.0 [97.5, 98.5]; H18 B2 | 97.7 [97.5, 98.3]; H20 B2 | 96.7 [95.3, 97.7]; O1 H31 B2 | 90.7 [87.8, 93.2]; O5 H59 B30 | 98.0 [96.8, 98.5]; H21 B1 | 97.2 [96.7, 98.2]; H23 B4 | 90.3 [87.5, 91.7]; O8 H66 B27 |
| `residual_ppo_forecast` (ideal feed) | 96.8 [96.5, 98.0]; H25 B4 | 95.8 [94.8, 96.3]; H34 B9 | 80.7 [76.8, 87.2]; C2 O4 H82 B99 | 4.0 [3.2, 9.3]; C108 O306 H341 B192 | 95.2 [94.0, 96.7]; H29 B18 | 79.5 [72.8, 82.3]; C1 O6 H78 B132 | 4.0 [2.5, 11.2]; C111 O312 H298 B223 |
| `ppo_sinusoid` | 97.3 [96.7, 97.8]; H25 B2 | 97.0 [96.5, 98.2]; H25 B3 | 96.5 [93.7, 97.8]; O1 H33 B6 | 71.0 [52.7, 74.3]; C3 O56 H155 B120 | 97.2 [96.7, 97.8]; H21 B7 | 93.7 [92.0, 94.7]; O3 H41 B21 | 66.7 [50.2, 74.5]; C1 O57 H155 B144 |
| `pid_track_descend` | 80.0 [73.9, 85.0] 160/200; H17 B23 | 78.5 [72.3, 83.6] 157/200; H18 B25 | 66.5 [59.7, 72.7] 133/200; H21 B46 | 23.5 [18.2, 29.8] 47/200; H7 B85 T61 | 80.0 [73.9, 85.0] 160/200; H21 B19 | 66.5 [59.7, 72.7] 133/200; H24 B43 | 21.5 [16.4, 27.7] 43/200; H6 B92 T59 |
| `pid_feedforward` | 90.5 [85.6, 93.8] 181/200; H8 B11 | 88.0 [82.8, 91.8] 176/200; H9 B15 | 54.0 [47.1, 60.8] 108/200; H19 B73 | 0.5 [0.1, 2.8] 1/200; C81 O28 H20 B10 T60 | 86.5 [81.1, 90.6] 173/200; H7 B20 | 48.5 [41.7, 55.4] 97/200; H23 B80 | 0.0 [0.0, 1.9] 0/200; C75 O38 H14 B9 T64 |
| `pid_feedforward_lowvz` | 85.0 [79.4, 89.3] 170/200; H9 B21 | 77.5 [71.2, 82.7] 155/200; H10 B35 | 50.0 [43.1, 56.9] 100/200; H18 B82 | 6.5 [3.8, 10.8] 13/200; C12 O33 H47 B86 T9 | 73.5 [67.0, 79.1] 147/200; H7 B46 | 43.5 [36.8, 50.4] 87/200; H19 B94 | 8.5 [5.4, 13.2] 17/200; C11 O37 H45 B85 T5 |
| `pid_feedforward_lowvz_cut` | 88.0 [82.8, 91.8] 176/200; H9 B15 | 91.0 [86.2, 94.2] 182/200; H10 B8 | 80.0 [73.9, 85.0] 160/200; H18 B22 | 42.0 [35.4, 48.9] 84/200; C12 O33 H47 B15 T9 | 91.5 [86.8, 94.6] 183/200; H7 B10 | 79.5 [73.4, 84.5] 159/200; H19 B22 | 45.0 [38.3, 51.9] 90/200; C11 O37 H45 B12 T5 |
| `gated` | 63.0 [56.1, 69.4] 126/200; B4 T70 | 51.5 [44.6, 58.3] 103/200; B7 T90 | 1.5 [0.5, 4.3] 3/200; B4 T193 | 0.0 [0.0, 1.9] 0/200; C122 O1 T77 | 48.5 [41.7, 55.4] 97/200; H1 B9 T93 | 1.5 [0.5, 4.3] 3/200; T197 | 0.0 [0.0, 1.9] 0/200; C116 O1 T83 |
| `oracle_gated` (privileged) | 64.5 [57.7, 70.8] 129/200; T71 | 62.0 [55.1, 68.4] 124/200; B4 T72 | 7.5 [4.6, 12.0] 15/200; H2 B10 T173 | 0.0 [0.0, 1.9] 0/200; C125 O1 T74 | 60.5 [53.6, 67.0] 121/200; B2 T77 | 5.0 [2.7, 9.0] 10/200; B15 T175 | 0.0 [0.0, 1.9] 0/200; C121 O2 T77 |


- **Latency alone costs nothing measurable for any learned method, and up to 15 points for the
  PID baselines (review M1, corrected by P7-D4; wording corrected in P7-D6).**
  - At 2 steps (66.7 ms model, 333 ms full scale) with σ_p = 0, at `id` SS5
    (`noise/sigma0cm_lat2step`):
    - `pid_feedforward` 92.5 % [88.0, 95.4] (185/200);
    - `residual_ppo` 99.0 [99.0, 99.3];
    - `ppo` 100.0 [100.0, 100.0].
  - Few clean − noisy contrasts separate under latency alone: 2 of 48 at 1 step and 5 of 48 at
    2 steps (`contrasts.csv`, `noise/sigma0cm_lat1step.*` and `noise/sigma0cm_lat2step.*`). All
    of them are PID baselines:
    - `pid_feedforward` at SS5: +2.5 [+0.5, +5.0] points at 1 step, +6.5 [+3.0, +10.5] at 2 steps;
    - `pid_feedforward_lowvz`: +7.0 [+2.5, +11.5] at SS5 and 1 step; +2.0 to +15.0 at SS3–SS6 and
      2 steps, lost mostly to `bounce` (e.g. SS5 H1 B38 against B9 clean). The largest is SS5 at
      2 steps, +15.0 [+9.0, +21.0] (95.5 → 80.5 %).
    - No learned method, `gated` or `oracle_gated` separates under latency alone. The largest
      learned point estimate is `sac`'s +5.3 [−2.0, +11.2] at SS6, 2 steps.
  - *The superseded arm* read 0.0 % / 0.2 % / 35.3 % for the same three cells
    (`results/e07/noise_superseded_p7d1/sigma0cm_lat2step/`). Those numbers measured the Phase 2
    stand-in's mixed-timestamp deck velocity, v_pad(t−L) + v_drone(t) − v_drone(t−L), fed through
    `pid_feedforward`'s k_ff = 1.08 feedforward (P7-D4). They did not measure perception latency.
    The earlier readings "latency is what breaks the controllers" and "classical baselines beat
    the learned methods under latency" are **withdrawn**.
- **Position-and-velocity noise degrades every method; latency adds little on top of it.**
  - At σ_p = 4 cm (σ_v = 0.2 m/s), all 48 clean − noisy contrasts separate in each of the three
    4 cm conditions (`contrasts.csv`).
  - At 4 cm and 0 steps, SS3 / SS4 / SS5 / SS6 (`noise/sigma4cm_lat0step`):
    - `ppo_forecast` 96.3 / 95.2 / 95.8 / 89.8 % (its feed stays ideal; see below);
    - `ppo` 79.8 / 81.5 / 79.2 / 75.7, and `ppo_sinusoid` 69.7 / 73.5 / 70.0 / 68.5;
    - `pid_feedforward_lowvz_cut` 41.5 / 42.5 / 46.5 / 39.0 (83 / 85 / 93 / 78 successes, of
      which 26 / 18 / 20 / 14 are tunnelled; 18–39 % across the twelve 4 cm cells, and 35 of 90
      at SS6, 2 steps; `tunnelled_success.csv`, table below);
    - `pid_track_descend` 21.5 / 22.5 / 22.0 / 24.5, and `sac` 19.2 / 16.5 / 15.8 / 10.5;
    - `pid_feedforward_lowvz` 5.5–9.0, `residual_ppo_forecast` 3.5–5.5, `residual_ppo` 2.5–3.2;
    - `pid_feedforward` 0.0 / 0.0 / 0.0 / 0.5 (1/200 at SS6), and `gated` and `oracle_gated`
      0.0 at every sea state. SS3 is where the noise is largest against the deck motion
      (σ_p 3.9×, σ_v 4.3× the deck SD; table above).
  - The 1- and 2-step 4 cm conditions sit within a few points of these, e.g. `ppo` 73.2–80.2 and
    `ppo_forecast` 90.3–97.0. No noisy-vs-noisy contrast was computed, so that is a reading of the
    tables only.
  - *How they fail differs* (4 cm, 0 steps, SS3):
    - `pid_feedforward`: crashes, off-pad, timeouts (C90 O34 H20 B13 T43 of 200);
    - `gated` / `oracle_gated`: crashes and timeouts (C120 T80 each). All 120 crashes of each
      are `tilt_gt_crash` with 0 contacts, i.e. tilt past 60° in the air; so are 88 of
      `pid_feedforward`'s 90 (the other 2 are `off_plate_strike`). Read from
      `noise/sigma4cm_lat0step/episodes.csv.gz` (`termination_reason`, `n_contacts`); descriptive
      only. Whether a correlated estimator error of the same σ gives the same in-air crashes was
      not tested;
    - `pid_track_descend`: bounces and timeouts (B95 T62);
    - `residual_ppo`: all classes (C170 O311 H302 B187 T1 per 1 000);
    - `ppo`: off-pad, hard landings, bounces (O44 H76 B77).
    - Across the three 4 cm conditions, `pid_feedforward`, `pid_track_descend`, `gated` and
      `oracle_gated` time out in 39–83 of 200 per cell. The learned methods time out in at most
      17 of 1 000 (`sac`), and the PPO family in at most 2.
  - *At 2 cm* the split is already visible across the three latencies:
    - the pure PPO methods stay at 93.7–100 %;
    - `residual_ppo` falls to 72.7–81.2 %, `pid_feedforward` to 48.5–66.0 %;
    - `gated` falls to 1.5–16.5 % and `oracle_gated` to 5.0–30.5 %.
  - *At 1 cm* every PPO-family cell is ≥ 94.5 %. `gated` already loses up to 14.5 points (SS6:
    51.0 / 51.5 / 48.5 % at 0 / 1 / 2 steps, against 63.0 % clean).
  - **No mechanism has been tested.** One plausible, *untested* explanation for the collapse of
    the controllers built on `pid_feedforward`:
    - those are `pid_feedforward`, its two `lowvz` variants, the gated pair and the residual
      methods, whose base it is;
    - velocity noise passes through the feedforward, which commands k_ff = 1.08 × the perceived
      pad velocity;
    - σ_v = 0.2 m/s is 1.5× the `id` SS5 deck v_z SD (0.134 m/s) and 4.3× SS3's (0.046 m/s);
      table above (corrected in P7-D6).

    `pid_track_descend` has no feedforward and still falls to 21.5–24.5 %. So that explanation
    could at best be partial. No run separated position noise from velocity noise.
- **Under noise, `ppo` and `ppo_forecast` sit above every classical baseline; `ppo_sinusoid`
  does in all but two cells** (unpaired reading; narrowed in P7-D6).
  - At 4 cm the lowest `ppo` seed-CI bound in any cell is 67.5 %, and `ppo_forecast`'s 87.5 %.
    The highest Wilson upper bound of any classical baseline is 53.4 %
    (`pid_feedforward_lowvz_cut`, SS5, 0 steps).
  - In every one of the 9 σ_p > 0 conditions × 4 sea states, the seed-CI lower bound of `ppo`
    and of `ppo_forecast` is above the highest Wilson upper bound of the six baselines in that
    cell (`aggregate.csv`, `baselines_summary.csv`).
  - `ppo_sinusoid`'s CI overlaps `pid_feedforward_lowvz_cut`'s in two cells: SS6, 4 cm, 2 steps,
    66.7 [50.2, 74.5] against 45.0 [38.3, 51.9] (90/200); and SS5, 2 cm, 2 steps, 97.3 [93.8,
    98.8] against 91.5 [86.8, 94.6] (183/200).
  - No method-vs-method contrast was computed in this arm, and the intervals are of different
    kinds (seed bootstrap vs Wilson).
  - Under latency alone, no learned controller separates from its clean value, and the PID
    baselines lose up to 15 points (above). The SS3 cells, and SS4 at 1 step, are at the 100 %
    ceiling for the pure PPO methods and the best baseline alike.
- **`ppo_forecast`'s noise robustness is not evidence that forecasting helps (review M2).**
  - Under P7-D4 every deck-derived observation entry is perceived. What stays ideal is the
    forecast methods' ship-motion feed and `oracle_gated`'s privileged context.
  - `ppo_forecast` keeps 89.8–97.0 % at 4 cm, against `ppo`'s 73.2–81.5 %. It is the only pure
    policy with an undelayed, noise-free view of the ship's motion.
  - `residual_ppo_forecast` has the same feed and still falls to 3.5–5.7 %. That is consistent
    with its `pid_feedforward` base reading the perceived deck (untested).
  - `oracle_gated` is 0.0 % at every 4 cm cell, like `gated`. Its privileged context is the true
    future deck trajectory, but its commit timing and its gates read the perceived observation
    (corrected in P7-D6):
    - the true-future window is placed at `predicted_touchdown_s`, t + max(clearance, 0) /
      descent rate, from the perceived (noisy) clearance (`src/rld/control/gated.py` l.153,
      called at `src/rld/control/oracle.py` l.145–147);
    - a commit also needs the at-hover check on the perceived clearance (`gated.py` l.182,
      ±5 cm of the 0.3 m hover height) and the lateral gate on the perceived lateral error
      (`src/rld/control/pid.py` l.170–175, read at `gated.py` l.179–181).
  - The `ppo_forecast` result is consistent with the policy leaning on its ideal side channel
    (untested: no run noised or withheld the feed). It is not evidence for the forecast block.
- **Measurement reliability under noise.**
  - *Detector disagreement* exceeds Gate 2's 1 % only in the three σ_p = 4 cm conditions:
    349 / 369 / 375 of 28 800 (1.21 / 1.28 / 1.30 %) at 0 / 1 / 2 steps. Every other condition
    is ≤ 0.15 % (`disagreement_n` in `summary.csv` and `baselines_summary.csv`; `detectors_disagree`
    in `episodes.csv.gz`).
  - The disagreements in those three conditions are almost all in failed episodes: 8 / 8 / 4 are
    successes, out of 28 800 each. So the breach barely touches the success rates. (The superseded
    arm had 3–13 such successes per 28 800 in its 6 conditions above 1 %.)
  - *Tunnelling* (penetration > 5 mm at any contact substep, `tunnelling_n`) is 1.22–11.97 % of
    episodes per condition. The like-for-like clean figure is 1.09 % (313 of the 28 800 clean
    `id` rows, learned and baselines: `matrix/episodes.csv.gz`, `results/e01/episodes.csv`,
    `results/e01_lowvz_cut/episodes.csv`). It rises with σ_p: 2.0–2.2 % at 1 cm, 5.4–5.7 % at
    2 cm, 11.4–12.0 % at 4 cm, and 1.2 % at σ_p = 0 with 1–2 steps.
- **Tunnelled successes (review M4)**, per condition and sea state, counted directly from
  `results/e07/noise/<condition>/tunnelled_success.csv` (`n_success_tunnelled`, pooled over seeds;
  clean from `matrix/tunnelled_success.csv`). Entries are SS3/SS4/SS5/SS6, per 1 000 for learned
  rows and per 200 for baselines.
  - Such a success had penetration > 5 mm at some contact substep. It **may, but need not**,
    depend on the overlap. No audit covers the noise arm: P6-D5's bound applies to the clean `id`
    rows only (§2).

| method | clean | 1 cm, 0 st | 2 cm, 0 st | 4 cm, 0 st | 0 cm, 1 st | 1 cm, 1 st | 2 cm, 1 st | 4 cm, 1 st | 0 cm, 2 st | 1 cm, 2 st | 2 cm, 2 st | 4 cm, 2 st |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` | 0/0/1/3 | 0/0/1/5 | 2/0/4/10 | 31/31/37/58 | 0/0/2/10 | 0/0/0/13 | 0/2/7/13 | 33/30/34/59 | 0/0/0/8 | 0/0/3/8 | 0/0/3/11 | 32/37/37/43 |
| `sac` (2 M steps) | 3/7/23/50 | 22/26/59/91 | 91/115/122/96 | 73/73/82/49 | 3/17/26/58 | 24/30/65/93 | 95/121/109/106 | 84/86/64/52 | 3/10/27/52 | 28/39/49/78 | 94/94/106/103 | 82/70/57/48 |
| `residual_ppo` | 0/0/0/4 | 0/0/2/1 | 2/8/7/6 | 1/1/5/5 | 0/0/0/2 | 0/0/1/5 | 3/2/7/12 | 3/3/3/3 | 0/0/0/4 | 0/0/0/4 | 5/6/6/14 | 4/5/3/4 |
| `ppo_forecast` (ideal feed) | 0/0/0/4 | 0/0/1/9 | 0/0/2/4 | 6/12/5/24 | 0/0/0/7 | 0/0/0/7 | 0/0/3/9 | 6/4/9/23 | 0/0/0/4 | 0/0/2/4 | 0/0/1/7 | 8/3/15/20 |
| `residual_ppo_forecast` (ideal feed) | 0/0/1/0 | 0/0/0/0 | 4/0/1/6 | 2/3/2/4 | 0/0/0/2 | 0/0/1/4 | 1/3/6/8 | 0/1/5/2 | 0/0/0/4 | 0/0/1/7 | 4/3/5/8 | 2/5/5/8 |
| `ppo_sinusoid` | 0/0/0/3 | 0/0/2/7 | 1/0/3/9 | 18/24/28/41 | 0/0/1/7 | 0/0/1/9 | 2/2/4/12 | 28/25/24/51 | 0/0/1/1 | 0/0/1/5 | 2/1/3/7 | 28/22/29/57 |
| `pid_track_descend` | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/1/0 | 0/0/0/1 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/1 | 0/0/0/0 | 0/0/0/0 |
| `pid_feedforward` | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/1/1 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/1 | 0/0/0/0 |
| `pid_feedforward_lowvz` | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/1 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/1/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/2/0/0 |
| `pid_feedforward_lowvz_cut` | 0/0/1/2 | 0/0/0/2 | 3/1/3/11 | 26/18/20/14 | 0/0/2/2 | 0/0/1/0 | 1/1/1/5 | 25/20/17/21 | 0/0/0/2 | 0/0/0/0 | 1/2/6/5 | 16/19/29/35 |
| `gated` | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| `oracle_gated` (privileged) | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |

  - *Where they matter.* At 4 cm `ppo` has 30–59 tunnelled successes per 1 000 per cell, up to
    5.9 points of a cell whose success is 73–82 %, and `ppo_sinusoid` has 18–57.
    `pid_feedforward_lowvz_cut` has 14–35 of 200: at SS6, 4 cm, 2 steps that is 35 of its 90
    successes. `sac` has 3–122 per 1 000 per cell across conditions.
  - At σ_p = 0 with 1–2 steps the counts stay near the clean ones (`ppo` SS6 10 and 8, against 3
    clean).

### 5. Sinusoid test motion (the H4 cross; `id`, aft)

Each listed episode was flown on the matched sinusoid from the training builder. Source:
`results/e07/sinusoid/{aggregate,summary,baselines_summary}.csv`; the JONSWAP − sinusoid contrasts
are `contrasts.csv` `sinusoid.*`.

Cell format: learned = IQM [95 % CI]; baselines = rate [Wilson 95 % CI] k/N; after the semicolon, losses by class (C crash, O off_pad, H hard_landing, B bounce, T timeout; counts of 1 000 for learned rows, of 200 for baselines; – = none).

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.0 [98.3, 99.3]; H4 B7 |
| `sac` (2 M steps) | 99.3 [99.0, 99.8]; H6 | 96.5 [91.5, 97.8]; O1 H44 B1 | 83.2 [74.5, 87.7]; C3 O13 H161 B2 | 61.0 [50.5, 73.0]; C27 O66 H280 B14 T1 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 98.7 [97.3, 99.5]; H5 B10 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 97.8 [96.7, 98.5]; O7 H11 B5 |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 98.7 [97.5, 99.7]; H9 B5 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 99.3 [98.5, 100.0]; H6 B1 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200; – | 93.5 [89.2, 96.2] 187/200; H2 B11 | 64.5 [57.7, 70.8] 129/200; H42 B29 | 60.0 [53.1, 66.5] 120/200; H50 B30 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 99.5 [97.2, 99.9] 199/200; B1 | 99.0 [96.4, 99.7] 198/200; B2 | 94.0 [89.8, 96.5] 188/200; H2 B10 |
| `pid_feedforward_lowvz` | 99.0 [96.4, 99.7] 198/200; B2 | 97.5 [94.3, 98.9] 195/200; B5 | 95.0 [91.0, 97.3] 190/200; B10 | 84.5 [78.8, 88.9] 169/200; H5 B26 |
| `pid_feedforward_lowvz_cut` | 99.0 [96.4, 99.7] 198/200; B2 | 98.5 [95.7, 99.5] 197/200; B3 | 97.0 [93.6, 98.6] 194/200; B6 | 88.5 [83.3, 92.2] 177/200; H5 B18 |
| `gated` | 100.0 [98.1, 100.0] 200/200; – | 75.5 [69.1, 80.9] 151/200; T49 | 29.0 [23.2, 35.6] 58/200; T142 | 2.0 [0.8, 5.0] 4/200; T196 |
| `oracle_gated` (privileged) | 100.0 [98.1, 100.0] 200/200; – | 75.0 [68.6, 80.5] 150/200; T50 | 28.5 [22.7, 35.1] 57/200; T143 | 2.0 [0.8, 5.0] 4/200; T196 |


- **No PPO-family method loses measurable success on sinusoids.**
  - At SS6, four of the five score at or above their JONSWAP rate: `ppo` 99.0 against 98.2,
    `ppo_sinusoid` 99.3 against 97.3. `ppo_forecast` scores 97.8 against 98.0.
  - None of the 20 PPO-family JONSWAP − sinusoid contrasts (5 methods × 4 SS) separates.
  - The two motions do not rank these policies.
- **The quiescence-gated controllers collapse on sinusoids.**
  - `gated` scores 2.0 % at SS6, with 196/200 `timeout`. The contrast is +61.0 [+54.0, +68.0].
  - A constant-amplitude sinusoid never offers the quiet window they wait for.
  - `pid_track_descend` also loses 20.5 [+13.5, +28.0] points at SS5, and `sac` 11.5 [+0.7, +19.3]
    at SS6.
  - So motion realism matters for these controllers. That is a descriptive, post-hoc reading, not
    H4.

### 6. λ sensitivity (`ppo` and `pid_feedforward`, with `pid_track_descend` and `oracle_gated`; aft)

Sources: `results/e07/lambda/{lam15,lam40}/{aggregate,summary,baselines_summary}.csv`,
`contrasts.csv` (`lambda/*`) and `lambda/feasibility.csv`; the λ = 1/25 column is the main matrix
(`matrix/aggregate.csv`, `matrix/summary.csv`, `matrix/carried_summary_e01.csv`).

**These contrasts against 1/25 are unpaired-episode** (P7-D1 §3). The start offset is re-drawn
inside each λ's window in 20 800 of 20 800 rows. So the bootstrap resamples seeds, then episodes
independently per λ. Full tables with outcome breakdowns are in `results/results.md` §5.

- **Neither `ppo` nor `pid_feedforward` separates from its λ = 1/25 value in any of the 52
  contrasts** (2 methods × 13 cells × 2 λ). Many are degenerate [0, 0] at the 100 % ceiling.
- *SS6, all four flown methods* (cell format as in §2; losses by class at 1/15 / 1/25 / 1/40):

| method | regime (SS6) | λ = 1/15 | λ = 1/25 | λ = 1/40 | losses 1/15 / 1/25 / 1/40 |
|---|---|---|---|---|---|
| `ppo` | `id` | 97.2 [96.5, 98.3] | 98.2 [98.0, 98.5] | 98.0 [96.8, 98.5] | H22 B5 / H15 B3 / H17 B5 |
| `pid_track_descend` | `id` | 73.0 [66.5, 78.7] 146/200 | 80.0 [73.9, 85.0] 160/200 | 86.5 [81.1, 90.6] 173/200 | H31 B23 / H17 B23 / H6 B21 |
| `pid_feedforward` | `id` | 89.0 [83.9, 92.6] 178/200 | 90.5 [85.6, 93.8] 181/200 | 93.0 [88.6, 95.8] 186/200 | H8 B14 / H8 B11 / H7 B7 |
| `oracle_gated` (privileged) | `id` | 56.5 [49.6, 63.2] 113/200 | 64.5 [57.7, 70.8] 129/200 | 73.0 [66.5, 78.7] 146/200 | T87 / T71 / T54 |
| `ppo` | `unseen_seastate` | 97.0 [96.7, 98.3] | 97.3 [96.0, 98.2] | 96.8 [96.5, 97.3] | H21 B6 / H23 B5 / H23 B8 |
| `pid_track_descend` | `unseen_seastate` | 74.0 [67.5, 79.6] 148/200 | 81.0 [75.0, 85.8] 162/200 | 83.5 [77.7, 88.0] 167/200 | H30 B22 / H15 B23 / H9 B24 |
| `pid_feedforward` | `unseen_seastate` | 90.5 [85.6, 93.8] 181/200 | 90.0 [85.1, 93.4] 180/200 | 93.0 [88.6, 95.8] 186/200 | H9 B10 / H6 B14 / H6 B8 |
| `oracle_gated` (privileged) | `unseen_seastate` | 60.5 [53.6, 67.0] 121/200 | 65.0 [58.2, 71.3] 130/200 | 69.5 [62.8, 75.5] 139/200 | T79 / T70 / T61 |
| `ppo` | `unseen_heading` | 86.7 [85.5, 87.7] | 90.7 [86.3, 93.3] | 92.8 [91.7, 95.2] | C1 O5 H108 B20 / C1 O3 H77 B17 / O1 H47 B20 |
| `pid_track_descend` | `unseen_heading` | 68.5 [61.8, 74.5] 137/200 | 72.5 [65.9, 78.2] 145/200 | 74.0 [67.5, 79.6] 148/200 | H32 B31 / H22 B33 / H20 B32 |
| `pid_feedforward` | `unseen_heading` | 72.0 [65.4, 77.8] 144/200 | 77.0 [70.7, 82.3] 154/200 | 80.5 [74.5, 85.4] 161/200 | H26 B30 / H24 B22 / H15 B24 |
| `oracle_gated` (privileged) | `unseen_heading` | 38.0 [31.6, 44.9] 76/200 | 46.5 [39.7, 53.4] 93/200 | 46.0 [39.2, 52.9] 92/200 | T124 / T107 / T108 |
| `ppo` | `unseen_vessel` | 98.8 [98.5, 99.0] | 99.3 [98.7, 99.5] | 99.8 [99.5, 100.0] | C4 H7 B1 / H7 B1 / H1 B1 |
| `pid_track_descend` | `unseen_vessel` | 74.5 [68.0, 80.0] 149/200 | 81.5 [75.5, 86.3] 163/200 | 85.0 [79.4, 89.3] 170/200 | H36 B15 / H20 B17 / H8 B22 |
| `pid_feedforward` | `unseen_vessel` | 96.5 [93.0, 98.3] 193/200 | 98.5 [95.7, 99.5] 197/200 | 98.5 [95.7, 99.5] 197/200 | C3 B4 / H1 B2 / H2 B1 |
| `oracle_gated` (privileged) | `unseen_vessel` | 81.5 [75.5, 86.3] 163/200 | 89.0 [83.9, 92.6] 178/200 | 90.0 [85.1, 93.4] 180/200 | T37 / T22 / T20 |

  - The points tend towards higher success at 1/40, where deck velocities are √(25/40) ≈ 0.79×
    those at 1/25.
  - Every `ppo` and `pid_feedforward` CI includes 0. For example `ppo` at `unseen_heading` SS6:
    1/25 − 1/15 is +4.0 [−2.3, +9.7]. For `pid_feedforward` at `id` SS6, 1/25 − 1/40 is −2.5
    [−8.0, +3.0].
- **Only the always-printed baselines separate, and only against 1/15.**
  - `pid_track_descend` separates in 3 of 26 cells: `id` SS5 +8.0 [+0.5, +16.0], `unseen_vessel`
    SS4 +4.5 [+0.5, +8.5] and SS5 +8.5 [+1.5, +15.0].
  - `oracle_gated` separates in 2: `unseen_heading` SS5 +4.0 [+1.0, +7.0] and `unseen_vessel` SS6
    +7.5 [+0.5, +14.5].
- **`below_deck` crashes are a scoring artifact of a fixed bound (review M3; diagnosed in P7-D5).**
  - *The rule.* The bail-out fires when the drone's height is below `deck_origin_z` − 0.3 m, i.e.
    0.7 m (`src/rld/envs/landing_env.py` l.731; `configs/env/landing.yaml`
    `bounds.below_deck_m: 0.3`). The 0.3 m is a fixed model-scale constant and is **not
    λ-scaled**, while deck excursions in model metres scale with λ: 25/15 = 1.67× larger at 1/15
    than at 1/25. `crash` outranks `success` (P3-D1), and the episode is classified at the end of
    the control step.
  - *λ = 1/15, `unseen_vessel` SS6: 7 crashes,* all `below_deck` (`lambda/lam15/episodes.csv.gz`).
    - #127: 4 `ppo` seeds (0, 1, 2, 4) and `pid_feedforward` never touch down. The S175 aft pad
      goes 0.51 m below the mean deck, so a drone tracking it about 0.2 m above crosses 0.7 m in
      the air.
    - #114 and #133 (`pid_feedforward`): the drone is resting on the pad as it sinks 0.31–0.32 m
      below the mean deck.
    - **#133 is a pre-empted success.** Touchdown at 4.08 s, closing speed 0.155 m/s, lateral
      offset 0.011 m, relative tilt 2.1°. The dwell reached 0.500 s inside the control step in
      which the bound fired, and the recorded dwell is 0.521 s. Without the artifact,
      `pid_feedforward`'s 193/200 (96.5 %) in this cell would read 194/200 (97.0 %).
  - *λ = 1/25 too, by a second route: deck roll.*
    - The episode is `unseen_heading` SS6 #177 (frigate, 90°, beam seas): matrix `ppo` seed 1 at
      the aft pad (its C1 in §2), and at the CG `ppo` 1, `ppo_forecast` 0 and `ppo_sinusoid` 4
      (`matrix/episodes.csv.gz`, `cg/episodes.csv.gz`). λ = 1/15 `unseen_heading` SS6 #20 (`ppo`
      seed 2) is the same case.
    - Here the pad never sinks more than 0.18–0.26 m. Instead the deck rolls to 24–32°, the drone
      slides across the plate and off its low edge, and it drops below 0.7 m.
    - Contact was lost before the drop, and the dwell was 0.40–0.48 s. These landings were failing
      under any label; the bound only decides that the label is `crash`.
  - *Evidence.* The deck depths are computed analytically from the listed realizations. The
    drone-vs-plate traces come from a post-hoc scratch re-flight of 5 of these episodes (#133,
    #127, #20 at 1/15; #177 aft and CG at 1/25). Each re-flight reproduced its committed outcome,
    step count and dwell. Nothing from it is committed, and the committed rows are unchanged.
  - Every other `below_deck` row in the committed e07 files has no contact (`n_contacts` = 0). They
    are mostly `sac` (1 in the matrix, 9 at CG, 3 on sinusoids, 1–53 per noise condition), plus
    17–27 non-`sac` rows in each 4 cm noise condition. They were not traced.
  - **The frozen criteria do not change.** This is a documented scoring artifact.
- *Feasibility.* The P1-D1 rule passes at every λ: v_z p99 0.745 / 0.577 / 0.456 m/s against
  2.083 m/s. The project λ stays 1/25.

### 7. MSS strip-theory records vs `unseen_vessel`, SS5, headings 180°/135° (described only)

These are records from another simulator (MSS, ITTC S-175, SS5 JONSWAP), **not measurements of a
real ship**. MSS's spectrum match and Octave parity pass (P7-D2 §3). Sources:
- `results/e07/mss/{aggregate,summary,baselines_summary}.csv` for MSS;
- the subset rows of `results/e07/matrix/episodes.csv.gz`, `results/e01/episodes.csv` and
  `results/e01_lowvz_cut/episodes.csv` for the `unseen_vessel` comparison.

| method | `mss_transfer` aft (N 1 000 / 200) | `mss_transfer_corpus` aft | `unseen_vessel` SS5, 180°/135° only (dmf, aft; N 485 / 97) |
|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 484/485; H1 |
| `sac` (2 M steps) | 97.8 [96.5, 99.3]; O1 H20 | 97.8 [94.7, 99.7]; H25 B1 | 475/485; H9 B1 |
| `residual_ppo` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 485/485 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 485/485 |
| `residual_ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 485/485 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 485/485 |
| `pid_track_descend` | 91.5 [86.8, 94.6] 183/200; H8 B9 | 96.0 [92.3, 98.0] 192/200; B8 | 81/97; H12 B4 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200; – | 100.0 [98.1, 100.0] 200/200; – | 97/97 |
| `pid_feedforward_lowvz` | 99.5 [97.2, 99.9] 199/200; B1 | 99.0 [96.4, 99.7] 198/200; B2 | 96/97; B1 |
| `pid_feedforward_lowvz_cut` | 99.5 [97.2, 99.9] 199/200; B1 | 99.0 [96.4, 99.7] 198/200; B2 | 96/97; B1 |
| `gated` | 100.0 [98.1, 100.0] 200/200; – | 99.0 [96.4, 99.7] 198/200; T2 | 89/97; T8 |
| `oracle_gated` (privileged) | 100.0 [98.1, 100.0] 200/200; – | 99.0 [96.4, 99.7] 198/200; T2 | 91/97; T6 |

- Every PPO-family method lands every MSS episode, and so does `pid_feedforward`. The MSS records
  do not separate any method from the strongest classical baseline.
- The point estimates of `gated`, `oracle_gated` and `pid_track_descend` are higher on MSS than on
  the dmf S175 subset.
  - The realizations, wave fields and transfer functions differ, so nothing is attributed.
  - The dmf subset has 97 episodes, below the protocol's 200. It is descriptive only, and
    `results/results.md` gives no Wilson CI for it.
- The CG-pad MSS tables (`results/results.md` §6) show the same picture, with `pid_track_descend`
  at 200/200 on both lists.

### 8. What a skeptic should ask about

1. **Ceilings and degenerate CIs.**
   - The PPO family is at 100 % (seed CI [100.0, 100.0]) in nearly every SS3–SS5 cell of every arm.
   - H4's CI is [0.0, 0.0] because all 4 000 seed-episodes in its cell succeed.
   - Many descriptive contrasts are [0, 0] for the same reason.
   - These intervals record a ceiling, not precision. The seed-bootstrap IQM CIs also leave out
     episode-level uncertainty: a 200/200 seed has a Wilson interval of [98.1, 100.0].
2. **H1b's margin is 1.0 point.** The frozen 50 ms bounce rule decides more than that
   (P7-D3 §6).
3. **The timeout-ranked sensitivity is vacuous.** No timeouts occur in any scored closing-speed
   cell.
4. **Learned methods' p95 is higher at CG (separating for `ppo`, `residual_ppo` and `ppo_sinusoid`
   only), and SAC fails more there**, although the CG deck is calmer
   (§3). This is not explained.
5. **The perception arm was re-flown (P7-D4).** The first arm's 2-step-latency collapse
   (`pid_feedforward`, `gated` and `oracle_gated` crashing in 200 of 200) was an artifact of the
   Phase 2 stand-in's mixed timestamps. It is withdrawn. Under P7-D4, latency alone moves no
   learned method measurably and costs the PID baselines up to 15 points (`pid_feedforward_lowvz`,
   SS5, 2 steps). σ_p = 4 cm (σ_v = 0.2 m/s) takes `pid_feedforward`, `gated` and `oracle_gated`
   to 0–0.5 % at every sea state. SS3 has the worst noise-to-signal ratio: there σ_p is 3.9× and
   σ_v 4.3× the deck SD (§4, P7-D6). The noise is white, i.i.d. per control step; a correlated
   estimator error of the same σ was not tested. Why noise does this was **not tested**.
6. **`ppo_forecast`'s noise robustness is consistent with an ideal side channel (untested)**
   (§4). Under P7-D4 every deck-derived observation entry is perceived. Only the forecast
   methods' ship-motion feed and `oracle_gated`'s privileged future trajectory stay ideal. No run
   noised or withheld the feed. Do not read it as a forecasting result.
7. **Tunnelling outside `id` is unaudited.**
   - P6-D5's bound (3 / 1 / 0 / 3 per 1 000) covers `id` SS6 only. The direct counts, which are a
     superset, are in §2 (matrix) and §4 (noise arm).
   - In the matrix, every pure-PPO tunnelling episode in `unseen_heading` is at SS6: 5.6 / 6.8 /
     6.9 % of that cell's episodes for `ppo` / `ppo_forecast` / `ppo_sinusoid`, with 25 / 22 / 19
     tunnelled successes per 1 000 (`matrix/summary.csv` `tunnelling_n`;
     `matrix/tunnelled_success.csv`). At `unseen_seastate` SS6 it is 1.5–2.0 %. (The earlier
     "1.4–2.0 % of `unseen_heading`" pooled SS3–SS6; corrected at review m1.)
   - The deepest PPO-family penetration is 8.81 mm (`ppo_forecast`, `unseen_heading` SS6). That is
     above the 7.03 mm `lowvz_cut` reference P5-D14 used as a limit (`matrix/summary.csv`,
     `max_penetration_m`). The deepest of any method is `sac` at 17.6 mm (`unseen_heading` SS6,
     seed 4), above P5-D14's 15.7 mm `id` maximum. *(Corrected at the Gate 7 review.)*
   - `sac` tunnels in 0.2 % (`static`) to 19.5 % (`unseen_seastate`) of episodes, by regime.
8. **The noise arm degrades the touchdown detectors** past Gate 2's 1 % in the 3 σ_p = 4 cm
   conditions (1.21–1.30 %). Only 4–8 of each condition's 28 800 disagreeing episodes are
   successes (§4). Tunnelling there reaches 11.4–12.0 %, against 1.09 % for the clean `id` rows.
9. **`below_deck` crashes are a scoring artifact** of a fixed 0.3 m model-scale bound (§6, P7-D5).
   They occur at λ = 1/15 (deep troughs; one pre-empted success, `pid_feedforward`
   `unseen_vessel` SS6 #133) and at λ = 1/25 (`unseen_heading` SS6 #177, a drone sliding off a
   deck rolled about 32°). The earlier "only at λ = 1/15" was wrong.
10. **The H3 secondary `unseen_vessel` SS6 half-rule** is "not scored (no supported id gain)". It
    was relabelled from "holds" at `182cdea`, and no number changed (P7-D5). The half-rule is met
    on point estimates, but the gain it would shrink is itself only "inconclusive". It is not
    support for H3.

### 9. Caveats carried into every reading (P7-D1 §8)

- **Forecast methods.** Their forecasts were in-sample in training, and the ship-motion feed is an
  extra ideal sensor (P6-D1).
- **Residual descent.** The residual methods descend harder than their base. No residual seed cuts
  the throttle after contact, and a `lowvz`-like descent was within authority (P6-D5, P6-D6 M1).
- **Descent law.** The learned methods use a two-phase descent that the PID tuning space cannot
  express.
- **Bounce label.** `bounce` is decided by a 50 ms contact-loss grace and is unstable at 240 Hz
  (P5-D1).
- **Closing speed** is understated by about 7 %, for every method alike (P5-D14).
- **Tunnelling** is flagged at any contact substep (P5-D14). P6-D5's bound is 3 / 1 / 0 / 3 SS6
  successes per 1 000 for `residual_ppo` / `ppo_forecast` / `residual_ppo_forecast` /
  `ppo_sinusoid`. That bound is the audited subset of the direct tunnelled-success counts, which
  are 4 / 4 / 0 / 3 at `id` SS6 (§2; `matrix/tunnelled_success.csv`).
- **Hard landings.** In the audited `id` cells, every hard landing of the four Phase 6 methods,
  `ppo` and `pid_feedforward` is tilt-only. `sac`'s and `pid_track_descend`'s are mostly
  speed-driven.
- **SAC budget.** SAC had 2 M env steps against 10 M for the PPO family.
