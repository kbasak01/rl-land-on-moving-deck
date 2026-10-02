# findings.md

Phase-by-phase record of what was found, including every claim withdrawn and what replaced it.
Written at each gate. Hypotheses are scored here exactly as pre-registered in P3-D1.

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
record is in P7-D3.

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
| H3 secondary | `unseen_vessel` half-rule, SS5 / SS6 | r(uv) −2.0 % [−3.6, −0.3] / +0.04 % [−1.9, +2.6] | **not applicable** / **holds** (point-estimate rule; see P7-D3 §6) |
| H4 | drop_sin − drop_jon, `id` SS5 | +0.0 [+0.0, +0.0] (degenerate: every episode a success) | **not supported**; **novelty claim withdrawn** (D0.4 and P3-D1 §8 both fired) |
| H5 | ORT CPU vs GPU latency | — | **pending — scored at Gate 8** |

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
- **Residual RL beat the PID it is built on at SS6 (H1b), by a margin the bounce rule can erase.**
  - *The numbers.* +6.0 [+2.0, +10.2] points at `id` SS6: 965/1 000 against 181/200. The line is
    5 points.
  - *What this cell is.* SS6 is outside every method's training distribution.
  - *What the PID cannot do.* The residual lands in about 2 s with a two-phase descent that the
    PID's constant-descent tuning space cannot express.
  - *The bounce rule.* 11 of `pid_feedforward`'s 19 losses are `bounce`s under the 50 ms grace
    rule. Three of those becoming successes would put the point estimate at 4.5, below the line.
    Tunnelling alone, at most 3 residual successes per 1 000, cannot move it (P7-D3 §6 has the
    arithmetic).
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
- **Training on sinusoids cost nothing measurable (H4).**
  - `ppo_sinusoid`, trained only on matched sinusoids, scores 1 000/1 000 at `id` SS5 on both
    JONSWAP and sinusoid motion. So does `ppo`.
  - H4 cannot be supported, as P6-D6 foresaw, and the motion-realism novelty claim is withdrawn
    (D0.4). The transfer is the finding.
- **H5** waits for Phase 8.

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
- **Within the PPO family no method stands out.** At `unseen_heading` SS6, `ppo` at 90.7
  [86.3, 93.3] is above `residual_ppo` at 85.7 [84.0, 89.0], but the seed CIs overlap. That is a
  non-result.
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

**Closing speed. The PID baselines land softer than every learned method in every regime, as an
unpaired reading.** Sources: `matrix/aggregate.csv` (IQM of per-seed p95, m/s) and
`carried_summary_e01.csv`. At `id` SS5:
- PPO family 0.271–0.282;
- `pid_feedforward` 0.262;
- `pid_feedforward_lowvz` 0.188;
- `sac` 0.595;
- `pid_track_descend` 0.525.

The only paired closing-speed tests are H1a and H3. The first finds the residual 47 % harder than
`lowvz`; the second finds no forecast effect.

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
- **Surprise 2: every learned method lands harder at CG, every classical controller softer.**
  - `ppo`'s p95 at `id` SS6 is 0.278 → 0.295 m/s, and `residual_ppo`'s 0.283 → 0.305.
  - `pid_feedforward`'s is 0.264 → 0.243.
  - This is an unpaired reading: no paired aft − CG closing-speed contrast was computed. The
    learned methods' seed CIs do not overlap between pads (`ppo` [0.274, 0.285] vs
    [0.286, 0.304]).
  - All learned methods were trained at the aft pad only.
  - One untested explanation is that they learned aft-specific structure of the deck motion,
    including the defect's coupling of deck tilt to pad heave. It is only that: an explanation
    nobody has tested.

### 4. Perception stand-in (`id`, aft pad)

σ_p is noise on the relative position, and σ_v = σ_p / 0.2 s on the relative velocity. Latency is
**0, 1 or 2 control steps**, i.e. 0 / 33.3 / 66.7 ms model or 0 / 167 / 333 ms full scale
(`noise_latency_steps` and `noise_latency_ms_effective` in each `results/e07/noise/*/summary.csv`).
The noise is applied to the six relative-pad entries only. **The forecast methods' ship-motion
feed stays ideal and undelayed.**

The table shows the clean condition, the three pure-noise conditions and the two pure-latency
conditions. The six mixed conditions, and every paired clean − noisy contrast, are in
`results/results.md` §4 and Appendix B, and in `contrasts.csv` (`noise/*`). Sources:
`results/e07/noise/<condition>/{aggregate,summary,baselines_summary}.csv`.

Cell format: learned = IQM [95 % CI]; baselines = rate [Wilson 95 % CI] k/N; after the semicolon, losses by class (C crash, O off_pad, H hard_landing, B bounce, T timeout; counts of 1 000 for learned rows, of 200 for baselines; – = none).

#### `id` SS5

| method | clean | σp 1 cm, 0 steps | σp 2 cm, 0 steps | σp 4 cm, 0 steps | σp 0, 1 step (33.3 ms) | σp 0, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|
| `ppo` | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 98.3 [97.0, 99.7]; H5 B11 | 78.2 [74.3, 85.8]; C1 O38 H84 B84 | 85.5 [78.5, 91.7]; O4 H8 B135 | 35.3 [25.2, 56.2]; C15 O188 H299 B115 |
| `sac` (2 M steps) | 87.0 [81.7, 90.3]; C2 O8 H122 B3 T1 | 78.8 [75.2, 84.7]; C5 O27 H166 B8 T1 | 54.8 [48.5, 70.3]; C23 O86 H302 B10 T2 | 16.2 [9.5, 26.5]; C165 O238 H388 B31 T4 | 34.7 [30.5, 50.3]; C71 O174 H352 B18 T5 | 11.8 [6.3, 16.5]; C187 O311 H366 B13 T9 |
| `residual_ppo` | 99.5 [99.5, 99.8]; H4 | 99.2 [98.7, 99.5]; H7 B2 | 80.3 [76.7, 84.2]; O7 H68 B121 | 3.3 [1.8, 6.2]; C157 O295 H306 B204 T1 | 79.5 [63.2, 87.0]; C1 O1 H12 B220 | 0.2 [0.0, 0.5]; C809 O110 H63 B16 |
| `ppo_forecast` (ideal feed) | 100.0 [100.0, 100.0]; – | 100.0 [99.7, 100.0]; B1 | 100.0 [99.7, 100.0]; H1 | 95.8 [95.2, 96.5]; O4 H19 B19 | 99.8 [99.5, 100.0]; B2 | 99.7 [99.2, 100.0]; O1 H1 B2 |
| `residual_ppo_forecast` (ideal feed) | 99.5 [99.5, 99.5]; H5 | 98.8 [98.5, 99.0]; H10 B2 | 84.3 [80.7, 88.0]; O4 H49 B102 | 3.8 [2.3, 8.5]; C87 O324 H309 B232 | 96.7 [95.2, 99.3]; H5 B26 | 9.2 [4.3, 18.3]; C560 O89 H110 B135 |
| `ppo_sinusoid` | 100.0 [100.0, 100.0]; – | 100.0 [100.0, 100.0]; – | 98.5 [98.0, 99.3]; O1 H8 B5 | 78.5 [67.5, 82.3]; C2 O49 H115 B70 | 84.2 [71.5, 92.8]; O21 H27 B120 | 16.8 [8.2, 24.3]; C57 O316 H364 B98 |
| `pid_track_descend` | 85.0 [79.4, 89.3] 170/200; H12 B18 | 81.5 [75.5, 86.3] 163/200; H13 B24 | 68.5 [61.8, 74.5] 137/200; H9 B54 | 18.5 [13.7, 24.5] 37/200; H8 B94 T61 | 84.0 [78.3, 88.4] 168/200; H13 B19 | 82.0 [76.1, 86.7] 164/200; H13 B23 |
| `pid_feedforward` | 99.0 [96.4, 99.7] 198/200; H1 B1 | 96.5 [93.0, 98.3] 193/200; B7 | 63.5 [56.6, 69.9] 127/200; O1 H7 B65 | 0.0 [0.0, 1.9] 0/200; C89 O23 H16 B7 T65 | 96.5 [93.0, 98.3] 193/200; C1 H1 B5 | 0.0 [0.0, 1.9] 0/200; C200 |
| `pid_feedforward_lowvz` | 95.5 [91.7, 97.6] 191/200; B9 | 88.5 [83.3, 92.2] 177/200; B23 | 60.5 [53.6, 67.0] 121/200; H6 B73 | 7.5 [4.6, 12.0] 15/200; C7 O34 H45 B89 T10 | 94.0 [89.8, 96.5] 188/200; H2 B10 | 71.0 [64.4, 76.8] 142/200; C2 O1 B55 |
| `pid_feedforward_lowvz_cut` | 98.0 [95.0, 99.2] 196/200; B4 | 95.5 [91.7, 97.6] 191/200; B9 | 84.0 [78.3, 88.4] 168/200; H6 B26 | 43.5 [36.8, 50.4] 87/200; C7 O34 H45 B17 T10 | 96.5 [93.0, 98.3] 193/200; H2 B5 | 89.0 [83.9, 92.6] 178/200; C2 O1 B19 |
| `gated` | 89.5 [84.5, 93.0] 179/200; T21 | 86.0 [80.5, 90.1] 172/200; B2 T26 | 5.5 [3.1, 9.6] 11/200; H1 B9 T179 | 0.0 [0.0, 1.9] 0/200; C109 O1 T90 | 88.5 [83.3, 92.2] 177/200; B1 T22 | 0.0 [0.0, 1.9] 0/200; C200 |
| `oracle_gated` (privileged) | 89.0 [83.9, 92.6] 178/200; B1 T21 | 86.5 [81.1, 90.6] 173/200; B3 T24 | 14.5 [10.3, 20.0] 29/200; H1 B28 T142 | 0.0 [0.0, 1.9] 0/200; C107 T93 | 88.5 [83.3, 92.2] 177/200; B1 T22 | 0.0 [0.0, 1.9] 0/200; C200 |

#### `id` SS6

| method | clean | σp 1 cm, 0 steps | σp 2 cm, 0 steps | σp 4 cm, 0 steps | σp 0, 1 step (33.3 ms) | σp 0, 2 steps (66.7 ms) |
|---|---|---|---|---|---|---|
| `ppo` | 98.2 [98.0, 98.5]; H15 B3 | 98.7 [98.0, 99.0]; H12 B2 | 96.0 [93.8, 97.7]; O1 H25 B15 | 78.7 [75.2, 84.8]; O34 H119 B53 | 90.0 [86.8, 91.5]; O1 H20 B84 | 36.8 [27.7, 50.5]; C11 O157 H303 B147 |
| `sac` (2 M steps) | 72.5 [60.2, 79.3]; C18 O39 H232 B2 | 61.7 [54.0, 70.2]; C24 O64 H287 B5 T1 | 41.0 [38.2, 52.3]; C50 O123 H371 B19 T1 | 12.0 [9.2, 20.2]; C219 O273 H339 B25 T8 | 27.3 [19.3, 38.0]; C132 O221 H339 B19 T6 | 6.3 [4.2, 9.7]; C275 O342 H288 B10 T18 |
| `residual_ppo` | 96.2 [95.7, 97.8]; H24 B11 | 95.3 [95.0, 96.7]; H33 B11 | 76.5 [71.2, 79.7]; O6 H117 B118 | 3.2 [2.0, 5.3]; C152 O343 H319 B151 | 83.5 [68.8, 89.0]; C1 H46 B147 | 0.5 [0.0, 1.8]; C809 O107 H63 B14 |
| `ppo_forecast` (ideal feed) | 98.0 [97.5, 98.5]; H18 B2 | 98.2 [98.0, 99.2]; H15 B1 | 96.3 [96.0, 97.7]; C1 O1 H25 B7 | 90.8 [88.0, 92.8]; O6 H59 B29 | 98.0 [96.2, 98.5]; H21 B3 | 96.7 [95.0, 97.0]; O1 H21 B15 |
| `residual_ppo_forecast` (ideal feed) | 96.8 [96.5, 98.0]; H25 B4 | 95.0 [93.0, 96.3]; H41 B11 | 79.5 [73.5, 85.0]; O6 H92 B109 | 4.8 [2.5, 9.8]; C93 O313 H313 B224 | 93.7 [92.7, 96.2]; O1 H32 B26 | 8.5 [4.0, 15.3]; C563 O98 H111 B135 |
| `ppo_sinusoid` | 97.3 [96.7, 97.8]; H25 B2 | 97.3 [97.0, 98.7]; H22 B2 | 95.3 [92.7, 96.0]; H41 B12 | 73.5 [66.7, 80.5]; C2 O57 H151 B55 | 81.0 [71.5, 86.5]; C1 O15 H61 B126 | 19.5 [9.5, 21.5]; C54 O313 H356 B106 |
| `pid_track_descend` | 80.0 [73.9, 85.0] 160/200; H17 B23 | 77.5 [71.2, 82.7] 155/200; H19 B26 | 68.5 [61.8, 74.5] 137/200; H14 B49 | 23.0 [17.7, 29.3] 46/200; H4 B81 T69 | 77.0 [70.7, 82.3] 154/200; H18 B28 | 82.5 [76.6, 87.1] 165/200; H16 B19 |
| `pid_feedforward` | 90.5 [85.6, 93.8] 181/200; H8 B11 | 88.0 [82.8, 91.8] 176/200; H7 B17 | 56.5 [49.6, 63.2] 113/200; O1 H17 B69 | 0.0 [0.0, 1.9] 0/200; C89 O30 H15 B12 T54 | 89.0 [83.9, 92.6] 178/200; H9 B13 | 0.0 [0.0, 1.9] 0/200; C195 O4 B1 |
| `pid_feedforward_lowvz` | 85.0 [79.4, 89.3] 170/200; H9 B21 | 76.0 [69.6, 81.4] 152/200; H12 B36 | 58.0 [51.1, 64.6] 116/200; O2 H16 B66 | 7.0 [4.2, 11.4] 14/200; C13 O44 H43 B76 T10 | 78.5 [72.3, 83.6] 157/200; H10 B33 | 59.5 [52.6, 66.1] 119/200; C6 H17 B58 |
| `pid_feedforward_lowvz_cut` | 88.0 [82.8, 91.8] 176/200; H9 B15 | 85.0 [79.4, 89.3] 170/200; H12 B18 | 81.0 [75.0, 85.8] 162/200; O2 H16 B20 | 35.5 [29.2, 42.3] 71/200; C13 O44 H43 B19 T10 | 83.5 [77.7, 88.0] 167/200; H10 B23 | 77.5 [71.2, 82.7] 155/200; C6 H17 B22 |
| `gated` | 63.0 [56.1, 69.4] 126/200; B4 T70 | 55.0 [48.1, 61.7] 110/200; B2 T88 | 3.0 [1.4, 6.4] 6/200; B5 T189 | 0.0 [0.0, 1.9] 0/200; C110 O1 T89 | 58.5 [51.6, 65.1] 117/200; B9 T74 | 0.0 [0.0, 1.9] 0/200; C200 |
| `oracle_gated` (privileged) | 64.5 [57.7, 70.8] 129/200; T71 | 61.0 [54.1, 67.5] 122/200; B3 T75 | 6.5 [3.8, 10.8] 13/200; H1 B11 T175 | 0.0 [0.0, 1.9] 0/200; C105 T95 | 63.0 [56.1, 69.4] 126/200; B2 T72 | 0.0 [0.0, 1.9] 0/200; C200 |


- **Latency is what breaks the controllers.**
  - At 2 steps (66.7 ms model, 333 ms full scale) with no position noise, `ppo` falls from 100 % to
    35.3 % [25.2, 56.2] at SS5. The clean − noisy contrast is +64.7 [+45.0, +75.5].
  - `ppo_sinusoid` falls to 16.8 %.
  - `residual_ppo` falls to 0.2 %, and **81 % of its episodes crash** (tilt > 60°).
  - `pid_feedforward`, `gated` and `oracle_gated` crash in **200 of 200 at every sea state, SS3
    included**. The crash reason is tilt > 60° (`n_reason_tilt_gt_crash` in
    `noise/sigma0cm_lat2step/baselines_summary.csv`).
  - At 1 step all of them still fly: `pid_feedforward` 96.5 % at SS5.
  - So a 66.7 ms delay of the relative-pad state destabilises the `pid_feedforward` gain set. The
    residual inherits that through its base.
  - This was not diagnosed beyond the termination reason.
- **Classical baselines beat the learned methods under latency** (unpaired reading of the table).
  - At 2 steps, SS5: `pid_track_descend` 82.0 % [76.1, 86.7] and `pid_feedforward_lowvz_cut`
    89.0 % [83.9, 92.6], against `ppo` 35.3 % [25.2, 56.2], `ppo_sinusoid` 16.8 % [8.2, 24.3] and
    `residual_ppo` 0.2 %.
  - At 1 step, `pid_feedforward` (96.5 % [93.0, 98.3]) is above `ppo` (85.5 % [78.5, 91.7]) and
    `residual_ppo` (79.5 % [63.2, 87.0]).
  - No method-vs-method contrast was computed in this arm. The intervals are of different kinds
    (Wilson vs seed bootstrap), and they do not overlap in the 2-step case.
- **`ppo_forecast` barely degrades under latency**: 99.7 % at 2 steps, SS5; 96.7 % at SS6.
  - **This is not evidence that forecasting helps.** Its ship-motion feed is the only undelayed,
    noise-free deck-state channel in the arm, and no other method has it.
  - `residual_ppo_forecast` has the same feed and still collapses (9.2 %), because its PID base
    reads the delayed state.
  - Read this as a measure of how much a policy leans on an ideal side channel. It is not a
    property of the forecast block.
- **Position noise alone is survivable at 1–2 cm and not at 4 cm.**
  - At 4 cm, SS5: `ppo` 78.2 %, `ppo_sinusoid` 78.5 %, `ppo_forecast` 95.8 %.
  - The residual methods fall to 3–4 %, `pid_feedforward` and the gated controllers to 0 %, and
    `pid_feedforward_lowvz_cut` to 43.5 %.
  - Under noise, `pid_feedforward`, `pid_track_descend` and the gated controllers start to *time
    out* (T54–T95 of 200 at 4 cm, SS5–SS6). The learned methods time out in at most 8 of 1 000.
- **Measurement reliability degrades under noise.**
  - Detector disagreement reaches 1.3–2.3 % of a condition's 28 800 episodes in 6 of the 11
    conditions: every 2-step condition with σ_p ≤ 2 cm, and every 4 cm condition. Gate 2's
    threshold was < 1 % (`disagreement_n`).
  - Tunnelling is 2.0–13.7 % of episodes per condition (`tunnelling_n`), against 1.4 % in the
    clean matrix.
  - Outcome labels in the noisy conditions rest on a less consistent touchdown detection than the
    clean ones.

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

### 6. λ sensitivity (`ppo` and `pid_feedforward`; aft)

Sources: `results/e07/lambda/{lam15,lam40}/{aggregate,summary,baselines_summary}.csv`,
`contrasts.csv` (`lambda/*`) and `lambda/feasibility.csv`.

**These contrasts against 1/25 are unpaired-episode** (P7-D1 §3). The start offset is re-drawn
inside each λ's window in 20 800 of 20 800 rows. So the bootstrap resamples seeds, then episodes
independently per λ. Full tables with outcome breakdowns are in `results/results.md` §5.

- **Neither `ppo` nor `pid_feedforward` separates from its λ = 1/25 value in any of the 52
  contrasts** (2 methods × 13 cells × 2 λ). Many are degenerate [0, 0] at the 100 % ceiling.
- *Point estimates at SS6* (1/15 / 1/25 / 1/40):

  | regime | `ppo` | `ppo` breakdown at 1/15 / 1/40 | `pid_feedforward` | `pid_feedforward` breakdown at 1/15 / 1/40 |
  |---|---|---|---|---|
  | `id` | 97.2 / 98.2 / 98.0 | H22 B5 / H17 B5 | 89.0 / 90.5 / 93.0 | H8 B14 / H7 B7 |
  | `unseen_heading` | 86.7 / 90.7 / 92.8 | C1 O5 H108 B20 / O1 H47 B20 | 72.0 / 77.0 / 80.5 | H26 B30 / H15 B24 |

  - The points tend towards higher success at 1/40, where deck velocities are √(25/40) ≈ 0.79×
    those at 1/25.
  - Every CI includes 0. For example `ppo` at `unseen_heading` SS6: 1/25 − 1/15 is +4.0
    [−2.3, +9.7]. For `pid_feedforward` at `id` SS6, 1/25 − 1/40 is −2.5 [−8.0, +3.0].
- **Only the always-printed baselines separate, and only against 1/15.**
  - `pid_track_descend` separates in 3 cells, e.g. +8.0 [+0.5, +16.0] at `id` SS5.
  - `oracle_gated` separates in 2.
- **Surprise: crashes appear only at λ = 1/15 on the S175 hull.**
  - At `unseen_vessel` SS6 there are 4 `ppo` seed-episodes and 3 `pid_feedforward` episodes, plus 1
    `ppo` at `unseen_heading` SS6.
  - All end by `below_deck` (`n_reason_below_deck`).
  - 1/15 is outside the declared ladder. Whether these are fast-rising-deck impacts or a contact
    artifact was not diagnosed.
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
4. **Learned methods land harder at CG, and SAC fails more there**, although the CG deck is calmer
   (§3). This is not explained.
5. **A 2-step delay crashes `pid_feedforward`, `gated` and `oracle_gated` in 100 % of episodes,
   SS3 included** (§4). This was not diagnosed beyond "tilt > 60°".
6. **`ppo_forecast`'s latency robustness comes from an ideal side channel** (§4). Do not read it as
   a forecasting result.
7. **Tunnelling outside `id` is unaudited.**
   - P6-D5's bound (3 / 1 / 0 / 3 per 1 000) covers `id` SS6 only.
   - In the matrix, the pure PPO methods tunnel in 1.4–2.0 % of `unseen_heading` and
     `unseen_seastate` episodes, with a deepest penetration of 8.81 mm (`ppo_forecast`,
     `unseen_heading`).
   - That is above the 7.03 mm `lowvz_cut` reference P5-D14 used as a limit
     (`matrix/summary.csv`, `tunnelling_n`, `max_penetration_m`).
   - `sac` tunnels in 0.2 % (`static`) to 19.5 % (`unseen_seastate`) of episodes, by regime.
8. **The noise arm degrades the touchdown detectors** past Gate 2's 1 % in 6 of 11 conditions
   (§4).
9. **`below_deck` crashes appear only at λ = 1/15** (§6).
10. **The H3 secondary "holds"** is a half-rule met on point estimates, shrinking a gain that is
    itself only "inconclusive". It is not support for H3 (P7-D3 §6).

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
  `ppo_sinusoid`.
- **Hard landings.** In the audited `id` cells, every hard landing of the four Phase 6 methods,
  `ppo` and `pid_feedforward` is tilt-only. `sac`'s and `pid_track_descend`'s are mostly
  speed-driven.
- **SAC budget.** SAC had 2 M env steps against 10 M for the PPO family.
