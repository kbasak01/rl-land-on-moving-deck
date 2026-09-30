# e05 — PPO and SAC final checkpoints on the frozen `id` list (aft pad): success versus sea state, with the classical baselines carried beside them

- Simulation only (PyBullet); no real flight and no real deck data.
- Deck motion is dmf's 3-DOF (heave, roll, pitch) JONSWAP response, Froude-scaled to a Crazyflie at lambda = 1/25 (1 s model = 5 s full scale).
- State-based observations; the perception-noise stand-in is disabled (`configs/env/noise.yaml: enabled: false`); not vision.
- dmf's roll/pitch-heave phase defect (~90 deg) is carried, not fixed; this table is the aft pad, which is sensitive to it (P1-D2).
- Success = all four frozen criteria (`configs/env/success.yaml`); rates in %, Wilson 95 % CI in brackets, then k/N. Success is never pooled across sea states.
- Learned rows: each training seed's checkpoint flown as the batch-1 `LearnedPolicy` (deterministic action, `VecNormalize` statistics frozen, no ship-motion feed, not privileged) on the identical listed episodes. `seed` is the training seed.
- **Unequal training budgets:** `ppo` 10 000 000 env steps per seed (checkpoint `final`, 10 010 624 steps); `sac` 2 000 000 env steps per seed (checkpoint `final`, 2 000 000 steps). P3-D1 §5 sets them so; every PPO-vs-SAC reading carries that confound.
- `IQM` is rliable's interquartile mean of the per-seed success rates of the cell (with 5 seeds, the mean of the middle 3); its CI is the stratified bootstrap of P3-D1 §4 (2 000 replicates, seed 20260926, seeds resampled within the cell, cells never resampled). One cell is one task: nothing is pooled across sea states. That CI reflects seed-to-seed variation only, not episode sampling: when every seed scores the same it collapses to a point (e.g. [100.0, 100.0]), and each seed's Wilson CI is then the episode-level uncertainty.
- Baselines are **carried, not re-run**: their rows are line-for-line copies of the committed summaries named in each label (`carried_summary_*.csv`). They are deterministic and have one run (`seed` `–`).
- **`id` SS6 is outside every method's training distribution**: no learned method trained on SS6 (curriculum SS3 → SS5) and the baselines were tuned on SS3–SS5; the `in-dist` column is 0.000 there. At SS3–SS5, `in-dist` < 1 because 90 deg-heading episodes are outside the development pool (P3-D2).
- `oracle_gated` is privileged: it reads the true future deck motion. It is a commit-timing oracle, not a bound on success, and never a deployable result.
- **Measurement caveats from the reward-hacking audit (P5-D14; `results/audit/`)**, both frozen definitions applied to every method alike. (1) The recorded closing speed is read after the first contact substep's solver impulse, about 7 % below the speed one substep earlier; in a re-flight sample 15 of 445 `sac` successes (3.4 %) arrived above 0.5 m/s one substep before contact (`ppo` 0 of 500). (2) Up to 41 `sac` successes (1 / 3 / 13 / 24 at SS3–SS6, per 1 000 seed-episodes) had an unloaded stretch while tunnelled, and whether they would have lost contact for more than the 50 ms grace without the overlap is not shown (`ppo`: 2 at SS6; `pid_feedforward_lowvz_cut`: 1 at SS6). Both can only raise `sac`'s success, by at most a few points at SS5/SS6.
- No hypothesis (H1–H5) is scored here and no method contrast is tested; the spread and aggregates are descriptive.
- Rendered from `summary.csv`, `seeds.csv`, `aggregate.csv` and `carried_summary_*.csv` by `rld.eval.learned`; do not edit by hand.

## id, aft pad: success per sea state

| method | seed | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|---|
| ppo (learned; 10 000 000 env steps per seed) | IQM (5 seeds) | **100.0** [100.0, 100.0] | **100.0** [100.0, 100.0] | **100.0** [100.0, 100.0] | **98.2** [98.0, 98.5] |
| ppo | 0 | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) |
| ppo | 1 | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) |
| ppo | 2 | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) |
| ppo | 3 | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) |
| ppo | 4 | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) |
| ppo | min–max over seeds | 100.0–100.0 | 100.0–100.0 | 100.0–100.0 | 98.0–98.5 |
| sac (learned; 2 000 000 env steps per seed) | IQM (5 seeds) | **99.8** [98.8, 100.0] | **98.2** [96.3, 99.0] | **87.0** [81.7, 90.3] | **72.5** [60.2, 79.3] |
| sac | 0 | 99.5 [97.2, 99.9] (199/200) | 98.5 [95.7, 99.5] (197/200) | 89.0 [83.9, 92.6] (178/200) | 81.5 [75.5, 86.3] (163/200) |
| sac | 1 | 100.0 [98.1, 100.0] (200/200) | 97.0 [93.6, 98.6] (194/200) | 80.0 [73.9, 85.0] (160/200) | 55.5 [48.6, 62.2] (111/200) |
| sac | 2 | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 91.0 [86.2, 94.2] (182/200) | 75.0 [68.6, 80.5] (150/200) |
| sac | 3 | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 87.0 [81.6, 91.0] (174/200) | 73.0 [66.5, 78.7] (146/200) |
| sac | 4 | 98.5 [95.7, 99.5] (197/200) | 96.0 [92.3, 98.0] (192/200) | 85.0 [79.4, 89.3] (170/200) | 69.5 [62.8, 75.5] (139/200) |
| sac | min–max over seeds | 98.5–100.0 | 96.0–99.0 | 80.0–91.0 | 55.5–81.5 |
| pid_track_descend [baseline, carried from `results/e01`] | – | 100.0 [98.1, 100.0] (200/200) | 96.0 [92.3, 98.0] (192/200) | 85.0 [79.4, 89.3] (170/200) | 80.0 [73.9, 85.0] (160/200) |
| pid_feedforward [baseline, carried from `results/e01`] | – | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 90.5 [85.6, 93.8] (181/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) | 95.5 [91.7, 97.6] (191/200) | 85.0 [79.4, 89.3] (170/200) |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 98.0 [95.0, 99.2] (196/200) | 88.0 [82.8, 91.8] (176/200) |
| gated [baseline, carried from `results/e01`] | – | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 89.5 [84.5, 93.0] (179/200) | 63.0 [56.1, 69.4] (126/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 89.0 [83.9, 92.6] (178/200) | 64.5 [57.7, 70.8] (129/200) |

### id, aft pad: aggregates across training seeds (`aggregate.csv`)

| method | SS | runs | IQM success % [95 % CI] | optimality gap, points [95 % CI] | IQM of per-seed p95 closing speed, m/s [95 % CI] |
|---|---|---|---|---|---|
| ppo | SS3 | 5 | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.265 [0.263, 0.266] |
| ppo | SS4 | 5 | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.267 [0.264, 0.270] |
| ppo | SS5 | 5 | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.271 [0.267, 0.275] |
| ppo | SS6 | 5 | 98.2 [98.0, 98.5] | 1.8 [1.6, 2.0] | 0.278 [0.274, 0.285] |
| sac | SS3 | 5 | 99.8 [98.8, 100.0] | 0.4 [0.0, 1.0] | 0.392 [0.369, 0.433] |
| sac | SS4 | 5 | 98.2 [96.3, 99.0] | 2.1 [1.1, 3.2] | 0.437 [0.406, 0.478] |
| sac | SS5 | 5 | 87.0 [81.7, 90.3] | 13.6 [10.6, 17.0] | 0.595 [0.562, 0.619] |
| sac | SS6 | 5 | 72.5 [60.2, 79.3] | 29.1 [22.4, 37.5] | 0.668 [0.590, 0.778] |

The p95 IQM is descriptive: each seed's p95 is over its touched-down episodes only, and it is not P3-D1 §4's pooled relative-p95 contrast statistic.

### id, aft pad: seed-to-seed spread (`seeds.csv`)

| method | SS | success % by seed (0, 1, 2, 3, 4) | min | max | mean | SD | p95 closing speed m/s by seed | min | max | mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | SS3 | 100.0 / 100.0 / 100.0 / 100.0 / 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.265 / 0.266 / 0.263 / 0.263 / 0.265 | 0.263 | 0.266 | 0.265 | 0.001 |
| ppo | SS4 | 100.0 / 100.0 / 100.0 / 100.0 / 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.269 / 0.269 / 0.264 / 0.270 / 0.263 | 0.263 | 0.270 | 0.267 | 0.003 |
| ppo | SS5 | 100.0 / 100.0 / 100.0 / 100.0 / 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.271 / 0.276 / 0.267 / 0.274 / 0.269 | 0.267 | 0.276 | 0.271 | 0.004 |
| ppo | SS6 | 98.5 / 98.0 / 98.0 / 98.5 / 98.0 | 98.0 | 98.5 | 98.2 | 0.3 | 0.282 / 0.275 / 0.278 / 0.287 / 0.274 | 0.274 | 0.287 | 0.279 | 0.005 |
| sac | SS3 | 99.5 / 100.0 / 100.0 / 100.0 / 98.5 | 98.5 | 100.0 | 99.6 | 0.7 | 0.384 / 0.411 / 0.364 / 0.381 / 0.443 | 0.364 | 0.443 | 0.397 | 0.031 |
| sac | SS4 | 98.5 / 97.0 / 99.0 / 99.0 / 96.0 | 96.0 | 99.0 | 97.9 | 1.3 | 0.404 / 0.481 / 0.410 / 0.430 / 0.472 | 0.404 | 0.481 | 0.439 | 0.035 |
| sac | SS5 | 89.0 / 80.0 / 91.0 / 87.0 / 85.0 | 80.0 | 91.0 | 86.4 | 4.2 | 0.555 / 0.619 / 0.589 / 0.576 / 0.620 | 0.555 | 0.620 | 0.592 | 0.028 |
| sac | SS6 | 81.5 / 55.5 / 75.0 / 73.0 / 69.5 | 55.5 | 81.5 | 70.9 | 9.7 | 0.625 / 0.731 / 0.572 / 0.647 / 0.802 | 0.572 | 0.802 | 0.675 | 0.091 |

SD is the sample SD across seeds (ddof = 1), in points for success. A p95 statistic is over the seeds with a touchdown (`p95_n_seeds_finite` in the CSV).

### id, aft pad: outcome breakdown and touchdown audit

| method | seed | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p50 / p95 (m/s) | lat p50 / p95 (m) | tilt p95 (deg) | t_td p50 (s) | effort | jerk | disagree | tunnel | max depth (mm) | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | 0 | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.245 / 0.265 | 0.245 / 0.266 | 0.013 / 0.026 | 3.8 | 1.53 | 0.240 | 0.214 | 0/200 | 0 | 2.58 | 0.755 |
| ppo | 0 | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.244 / 0.269 | 0.245 / 0.270 | 0.013 / 0.026 | 5.1 | 1.53 | 0.244 | 0.213 | 0/200 | 0 | 3.86 | 0.750 |
| ppo | 0 | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.244 / 0.271 | 0.244 / 0.274 | 0.013 / 0.027 | 7.8 | 1.54 | 0.247 | 0.219 | 0/200 | 0 | 4.60 | 0.745 |
| ppo | 0 | SS6 | 200 | 0.000 | 0.000 | 0.015 | 0.000 | 0.985 | 0.000 | 0.247 / 0.282 | 0.249 / 0.283 | 0.013 / 0.034 | 11.9 | 1.56 | 0.247 | 0.224 | 0/200 | 0 | 5.00 | 0.000 |
| ppo | 1 | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.246 / 0.266 | 0.246 / 0.265 | 0.012 / 0.025 | 3.6 | 1.52 | 0.261 | 0.207 | 0/200 | 0 | 2.43 | 0.755 |
| ppo | 1 | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.247 / 0.269 | 0.247 / 0.269 | 0.013 / 0.025 | 4.4 | 1.53 | 0.263 | 0.206 | 0/200 | 0 | 2.61 | 0.750 |
| ppo | 1 | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.247 / 0.276 | 0.249 / 0.276 | 0.012 / 0.023 | 8.8 | 1.53 | 0.266 | 0.208 | 0/200 | 0 | 4.57 | 0.745 |
| ppo | 1 | SS6 | 200 | 0.000 | 0.000 | 0.020 | 0.000 | 0.980 | 0.000 | 0.245 / 0.275 | 0.247 / 0.289 | 0.014 / 0.031 | 12.1 | 1.57 | 0.266 | 0.210 | 0/200 | 1 | 5.13 | 0.000 |
| ppo | 2 | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.242 / 0.263 | 0.243 / 0.264 | 0.010 / 0.019 | 2.8 | 1.55 | 0.222 | 0.208 | 0/200 | 0 | 2.15 | 0.755 |
| ppo | 2 | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.241 / 0.264 | 0.242 / 0.264 | 0.010 / 0.020 | 4.3 | 1.55 | 0.225 | 0.209 | 0/200 | 0 | 2.88 | 0.750 |
| ppo | 2 | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.243 / 0.267 | 0.245 / 0.274 | 0.013 / 0.026 | 8.5 | 1.56 | 0.230 | 0.213 | 0/200 | 1 | 5.28 | 0.745 |
| ppo | 2 | SS6 | 200 | 0.000 | 0.000 | 0.015 | 0.005 | 0.980 | 0.000 | 0.244 / 0.278 | 0.245 / 0.282 | 0.015 / 0.037 | 11.6 | 1.57 | 0.231 | 0.217 | 0/200 | 0 | 4.97 | 0.000 |
| ppo | 3 | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.246 / 0.263 | 0.246 / 0.263 | 0.008 / 0.016 | 3.2 | 1.51 | 0.238 | 0.192 | 0/200 | 0 | 2.38 | 0.755 |
| ppo | 3 | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.247 / 0.270 | 0.247 / 0.269 | 0.008 / 0.016 | 3.6 | 1.52 | 0.240 | 0.192 | 0/200 | 0 | 3.55 | 0.750 |
| ppo | 3 | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.249 / 0.274 | 0.250 / 0.275 | 0.009 / 0.019 | 7.0 | 1.52 | 0.243 | 0.196 | 0/200 | 0 | 4.67 | 0.745 |
| ppo | 3 | SS6 | 200 | 0.000 | 0.000 | 0.010 | 0.005 | 0.985 | 0.000 | 0.254 / 0.287 | 0.255 / 0.288 | 0.011 / 0.034 | 11.2 | 1.55 | 0.242 | 0.196 | 0/200 | 1 | 5.13 | 0.000 |
| ppo | 4 | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.244 / 0.265 | 0.244 / 0.265 | 0.010 / 0.021 | 4.4 | 1.55 | 0.237 | 0.214 | 0/200 | 0 | 2.74 | 0.755 |
| ppo | 4 | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.243 / 0.263 | 0.244 / 0.263 | 0.011 / 0.021 | 5.7 | 1.55 | 0.238 | 0.214 | 0/200 | 0 | 2.89 | 0.750 |
| ppo | 4 | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.244 / 0.269 | 0.245 / 0.267 | 0.010 / 0.022 | 7.9 | 1.57 | 0.240 | 0.216 | 0/200 | 0 | 4.50 | 0.745 |
| ppo | 4 | SS6 | 200 | 0.000 | 0.000 | 0.015 | 0.005 | 0.980 | 0.000 | 0.244 / 0.274 | 0.247 / 0.277 | 0.013 / 0.029 | 12.1 | 1.59 | 0.238 | 0.219 | 0/200 | 3 | 5.35 | 0.000 |
| sac | 0 | SS3 | 200 | 0.000 | 0.000 | 0.005 | 0.000 | 0.995 | 0.000 | 0.305 / 0.384 | 0.303 / 0.384 | 0.021 / 0.038 | 5.0 | 1.19 | 0.543 | 0.416 | 0/200 | 0 | 4.61 | 0.755 |
| sac | 0 | SS4 | 200 | 0.000 | 0.000 | 0.015 | 0.000 | 0.985 | 0.000 | 0.315 / 0.404 | 0.315 / 0.403 | 0.022 / 0.049 | 6.7 | 1.18 | 0.613 | 0.426 | 0/200 | 1 | 6.32 | 0.750 |
| sac | 0 | SS5 | 200 | 0.000 | 0.015 | 0.090 | 0.005 | 0.890 | 0.000 | 0.342 / 0.555 | 0.341 / 0.555 | 0.028 / 0.073 | 9.3 | 1.25 | 0.666 | 0.446 | 0/200 | 14 | 9.13 | 0.745 |
| sac | 0 | SS6 | 200 | 0.000 | 0.045 | 0.135 | 0.005 | 0.815 | 0.000 | 0.358 / 0.625 | 0.357 / 0.627 | 0.035 / 0.095 | 17.3 | 1.27 | 0.686 | 0.471 | 1/200 | 24 | 8.94 | 0.000 |
| sac | 1 | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.343 / 0.411 | 0.343 / 0.411 | 0.041 / 0.061 | 3.7 | 1.35 | 0.744 | 0.431 | 0/200 | 0 | 3.84 | 0.755 |
| sac | 1 | SS4 | 200 | 0.000 | 0.000 | 0.030 | 0.000 | 0.970 | 0.000 | 0.337 / 0.481 | 0.337 / 0.481 | 0.036 / 0.060 | 5.9 | 1.35 | 0.746 | 0.444 | 0/200 | 1 | 5.02 | 0.750 |
| sac | 1 | SS5 | 200 | 0.000 | 0.010 | 0.185 | 0.005 | 0.800 | 0.000 | 0.377 / 0.619 | 0.377 / 0.620 | 0.036 / 0.080 | 8.6 | 1.42 | 0.717 | 0.471 | 0/200 | 18 | 8.08 | 0.745 |
| sac | 1 | SS6 | 200 | 0.005 | 0.040 | 0.400 | 0.000 | 0.555 | 0.000 | 0.431 / 0.731 | 0.430 / 0.704 | 0.042 / 0.094 | 15.0 | 1.59 | 0.708 | 0.485 | 0/200 | 41 | 12.63 | 0.000 |
| sac | 2 | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.263 / 0.364 | 0.262 / 0.364 | 0.042 / 0.066 | 7.8 | 1.34 | 0.521 | 0.423 | 0/200 | 2 | 5.50 | 0.755 |
| sac | 2 | SS4 | 200 | 0.000 | 0.000 | 0.010 | 0.000 | 0.990 | 0.000 | 0.288 / 0.410 | 0.289 / 0.408 | 0.037 / 0.068 | 7.7 | 1.41 | 0.561 | 0.433 | 0/200 | 3 | 7.82 | 0.750 |
| sac | 2 | SS5 | 200 | 0.005 | 0.005 | 0.075 | 0.000 | 0.910 | 0.005 | 0.305 / 0.589 | 0.307 / 0.589 | 0.029 / 0.065 | 11.4 | 1.59 | 0.571 | 0.445 | 0/200 | 16 | 9.90 | 0.745 |
| sac | 2 | SS6 | 200 | 0.050 | 0.045 | 0.155 | 0.000 | 0.750 | 0.000 | 0.344 / 0.572 | 0.344 / 0.579 | 0.033 / 0.093 | 13.9 | 1.67 | 0.605 | 0.460 | 0/200 | 45 | 10.59 | 0.000 |
| sac | 3 | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.306 / 0.381 | 0.306 / 0.383 | 0.021 / 0.035 | 3.7 | 1.32 | 0.524 | 0.394 | 0/200 | 0 | 3.52 | 0.755 |
| sac | 3 | SS4 | 200 | 0.000 | 0.005 | 0.005 | 0.000 | 0.990 | 0.000 | 0.325 / 0.430 | 0.325 / 0.430 | 0.020 / 0.046 | 6.2 | 1.29 | 0.621 | 0.411 | 0/200 | 0 | 4.63 | 0.750 |
| sac | 3 | SS5 | 200 | 0.000 | 0.000 | 0.130 | 0.000 | 0.870 | 0.000 | 0.362 / 0.576 | 0.365 / 0.571 | 0.027 / 0.061 | 9.7 | 1.31 | 0.681 | 0.442 | 0/200 | 13 | 9.74 | 0.745 |
| sac | 3 | SS6 | 200 | 0.000 | 0.035 | 0.235 | 0.000 | 0.730 | 0.000 | 0.418 / 0.647 | 0.410 / 0.657 | 0.035 / 0.094 | 13.8 | 1.38 | 0.713 | 0.477 | 0/200 | 36 | 15.43 | 0.000 |
| sac | 4 | SS3 | 200 | 0.000 | 0.000 | 0.015 | 0.000 | 0.985 | 0.000 | 0.314 / 0.443 | 0.314 / 0.443 | 0.020 / 0.043 | 6.6 | 1.35 | 0.760 | 0.465 | 0/200 | 1 | 6.05 | 0.755 |
| sac | 4 | SS4 | 200 | 0.000 | 0.000 | 0.040 | 0.000 | 0.960 | 0.000 | 0.323 / 0.472 | 0.322 / 0.475 | 0.020 / 0.048 | 7.1 | 1.39 | 0.786 | 0.469 | 0/200 | 6 | 6.85 | 0.750 |
| sac | 4 | SS5 | 200 | 0.005 | 0.010 | 0.130 | 0.005 | 0.850 | 0.000 | 0.353 / 0.620 | 0.353 / 0.617 | 0.025 / 0.063 | 10.0 | 1.57 | 0.783 | 0.485 | 0/200 | 11 | 13.19 | 0.745 |
| sac | 4 | SS6 | 200 | 0.035 | 0.030 | 0.235 | 0.005 | 0.695 | 0.000 | 0.395 / 0.802 | 0.394 / 0.834 | 0.027 / 0.087 | 16.3 | 1.61 | 0.801 | 0.501 | 1/200 | 32 | 15.67 | 0.000 |
| pid_track_descend [baseline, carried from `results/e01`] | – | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.238 / 0.308 | 0.238 / 0.309 | 0.021 / 0.040 | 3.9 | 4.62 | 0.024 | 0.003 | 0/200 | 0 | 2.77 | 0.755 |
| pid_track_descend [baseline, carried from `results/e01`] | – | SS4 | 200 | 0.000 | 0.000 | 0.020 | 0.020 | 0.960 | 0.000 | 0.254 / 0.412 | 0.254 / 0.412 | 0.021 / 0.042 | 4.8 | 4.59 | 0.024 | 0.003 | 0/200 | 0 | 4.99 | 0.750 |
| pid_track_descend [baseline, carried from `results/e01`] | – | SS5 | 200 | 0.000 | 0.000 | 0.060 | 0.090 | 0.850 | 0.000 | 0.303 / 0.525 | 0.303 / 0.525 | 0.019 / 0.037 | 7.5 | 4.54 | 0.024 | 0.003 | 0/200 | 4 | 6.50 | 0.745 |
| pid_track_descend [baseline, carried from `results/e01`] | – | SS6 | 200 | 0.000 | 0.000 | 0.085 | 0.115 | 0.800 | 0.000 | 0.287 / 0.502 | 0.290 / 0.502 | 0.021 / 0.040 | 13.9 | 4.65 | 0.025 | 0.003 | 0/200 | 2 | 6.34 | 0.000 |
| pid_feedforward [baseline, carried from `results/e01`] | – | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.219 | 0.193 / 0.219 | 0.017 / 0.032 | 4.4 | 4.50 | 0.026 | 0.007 | 0/200 | 0 | 2.99 | 0.755 |
| pid_feedforward [baseline, carried from `results/e01`] | – | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.189 / 0.226 | 0.189 / 0.226 | 0.016 / 0.034 | 5.1 | 4.49 | 0.030 | 0.010 | 0/200 | 0 | 3.25 | 0.750 |
| pid_feedforward [baseline, carried from `results/e01`] | – | SS5 | 200 | 0.000 | 0.000 | 0.005 | 0.005 | 0.990 | 0.000 | 0.192 / 0.262 | 0.192 / 0.262 | 0.016 / 0.033 | 8.4 | 4.41 | 0.036 | 0.014 | 0/200 | 0 | 3.46 | 0.745 |
| pid_feedforward [baseline, carried from `results/e01`] | – | SS6 | 200 | 0.000 | 0.000 | 0.040 | 0.055 | 0.905 | 0.000 | 0.193 / 0.264 | 0.195 / 0.264 | 0.015 / 0.031 | 13.4 | 4.55 | 0.038 | 0.014 | 0/200 | 0 | 3.54 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.103 / 0.127 | 0.103 / 0.127 | 0.015 / 0.031 | 4.5 | 7.68 | 0.011 | 0.005 | 0/200 | 0 | 2.15 | 0.755 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.980 | 0.000 | 0.103 / 0.139 | 0.103 / 0.140 | 0.015 / 0.032 | 5.2 | 7.65 | 0.013 | 0.008 | 0/200 | 0 | 1.97 | 0.750 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.045 | 0.955 | 0.000 | 0.110 / 0.188 | 0.110 / 0.185 | 0.014 / 0.032 | 9.2 | 7.61 | 0.018 | 0.011 | 0/200 | 0 | 2.86 | 0.745 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | SS6 | 200 | 0.000 | 0.000 | 0.045 | 0.105 | 0.850 | 0.000 | 0.106 / 0.181 | 0.108 / 0.182 | 0.017 / 0.034 | 13.9 | 7.77 | 0.019 | 0.011 | 0/200 | 0 | 3.84 | 0.000 |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.103 / 0.127 | 0.103 / 0.127 | 0.015 / 0.031 | 4.5 | 7.68 | 0.070 | 0.009 | 0/200 | 0 | 2.25 | 0.755 |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.015 | 0.985 | 0.000 | 0.103 / 0.139 | 0.103 / 0.140 | 0.015 / 0.032 | 5.2 | 7.65 | 0.072 | 0.011 | 0/200 | 0 | 2.28 | 0.750 |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.980 | 0.000 | 0.110 / 0.188 | 0.110 / 0.185 | 0.014 / 0.032 | 9.2 | 7.61 | 0.075 | 0.015 | 0/200 | 1 | 5.14 | 0.745 |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | SS6 | 200 | 0.000 | 0.000 | 0.045 | 0.075 | 0.880 | 0.000 | 0.106 / 0.181 | 0.108 / 0.182 | 0.017 / 0.034 | 13.9 | 7.77 | 0.072 | 0.014 | 0/200 | 5 | 6.31 | 0.000 |
| gated [baseline, carried from `results/e01`] | – | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.217 | 0.193 / 0.216 | 0.024 / 0.044 | 4.7 | 3.18 | 0.044 | 0.008 | 0/200 | 0 | 1.90 | 0.755 |
| gated [baseline, carried from `results/e01`] | – | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.192 / 0.229 | 0.192 / 0.229 | 0.021 / 0.045 | 4.9 | 3.28 | 0.046 | 0.011 | 0/200 | 0 | 2.37 | 0.750 |
| gated [baseline, carried from `results/e01`] | – | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.895 | 0.105 | 0.193 / 0.250 | 0.192 / 0.249 | 0.019 / 0.039 | 6.3 | 3.74 | 0.043 | 0.014 | 0/200 | 0 | 2.80 | 0.745 |
| gated [baseline, carried from `results/e01`] | – | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.630 | 0.350 | 0.194 / 0.267 | 0.194 / 0.267 | 0.016 / 0.038 | 7.8 | 4.66 | 0.036 | 0.014 | 0/200 | 0 | 3.08 | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.220 | 0.193 / 0.219 | 0.024 / 0.044 | 4.6 | 3.19 | 0.043 | 0.008 | 0/200 | 0 | 1.90 | 0.755 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.194 / 0.236 | 0.194 / 0.237 | 0.021 / 0.044 | 4.4 | 3.29 | 0.046 | 0.011 | 0/200 | 0 | 2.15 | 0.750 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.890 | 0.105 | 0.191 / 0.252 | 0.191 / 0.252 | 0.019 / 0.042 | 5.1 | 3.60 | 0.044 | 0.015 | 0/200 | 0 | 2.43 | 0.745 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.645 | 0.355 | 0.190 / 0.254 | 0.191 / 0.253 | 0.019 / 0.043 | 5.9 | 4.80 | 0.037 | 0.014 | 0/200 | 0 | 2.57 | 0.000 |

`v_n` is the touchdown closing speed along the deck normal (criterion 1), `v_z` the same along world z, both over touched-down episodes only; `lat` the lateral offset from the pad centre; `tilt` the drone-versus-deck-normal tilt; `t_td` the time to first contact; `effort` the mean ||a||^2 and `jerk` the mean ||a_t - a_(t-1)|| in normalised action units; `disagree` the analytic-vs-contact touchdown detector disagreements; `tunnel` the episodes whose penetration exceeded 5 mm; `max depth` the deepest penetration in the cell, in mm (the negated most-negative separation, `max_penetration_m` in the CSV).

### id, aft pad: termination reasons (counts)

| method | seed | SS | N | ground_contact | off_plate_strike | tilt_gt_crash | below_deck | out_of_bounds | dwell_complete | release | time_limit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | 0 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 0 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 0 | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 0 | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 199 | 1 | 0 |
| ppo | 1 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 1 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 1 | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 1 | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 197 | 3 | 0 |
| ppo | 2 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 2 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 2 | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 2 | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 199 | 1 | 0 |
| ppo | 3 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 3 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 3 | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 3 | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 199 | 1 | 0 |
| ppo | 4 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 4 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 4 | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| ppo | 4 | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 199 | 1 | 0 |
| sac | 0 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 0 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 0 | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 197 | 3 | 0 |
| sac | 0 | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 194 | 6 | 0 |
| sac | 1 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 1 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 1 | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 197 | 3 | 0 |
| sac | 1 | SS6 | 200 | 0 | 0 | 1 | 0 | 0 | 195 | 4 | 0 |
| sac | 2 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 2 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 2 | SS5 | 200 | 0 | 0 | 1 | 0 | 0 | 198 | 0 | 1 |
| sac | 2 | SS6 | 200 | 0 | 0 | 10 | 0 | 0 | 186 | 4 | 0 |
| sac | 3 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 3 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 199 | 1 | 0 |
| sac | 3 | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 199 | 1 | 0 |
| sac | 3 | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 197 | 3 | 0 |
| sac | 4 | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 4 | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| sac | 4 | SS5 | 200 | 0 | 0 | 0 | 1 | 0 | 198 | 1 | 0 |
| sac | 4 | SS6 | 200 | 0 | 3 | 4 | 0 | 0 | 189 | 4 | 0 |
| pid_track_descend [baseline, carried from `results/e01`] | – | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| pid_track_descend [baseline, carried from `results/e01`] | – | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 195 | 5 | 0 |
| pid_track_descend [baseline, carried from `results/e01`] | – | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 173 | 27 | 0 |
| pid_track_descend [baseline, carried from `results/e01`] | – | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 173 | 27 | 0 |
| pid_feedforward [baseline, carried from `results/e01`] | – | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| pid_feedforward [baseline, carried from `results/e01`] | – | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| pid_feedforward [baseline, carried from `results/e01`] | – | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 199 | 1 | 0 |
| pid_feedforward [baseline, carried from `results/e01`] | – | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 188 | 12 | 0 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 196 | 4 | 0 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 191 | 9 | 0 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) [baseline, carried from `results/e01`] | – | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 175 | 25 | 0 |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 197 | 3 | 0 |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 196 | 4 | 0 |
| pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) [baseline, carried from `results/e01_lowvz_cut`] | – | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 182 | 18 | 0 |
| gated [baseline, carried from `results/e01`] | – | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| gated [baseline, carried from `results/e01`] | – | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 197 | 0 | 3 |
| gated [baseline, carried from `results/e01`] | – | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 179 | 0 | 21 |
| gated [baseline, carried from `results/e01`] | – | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 126 | 4 | 70 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | SS3 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | SS4 | 200 | 0 | 0 | 0 | 0 | 0 | 197 | 0 | 3 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | SS5 | 200 | 0 | 0 | 0 | 0 | 0 | 178 | 1 | 21 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion [baseline, carried from `results/e01`] | – | SS6 | 200 | 0 | 0 | 0 | 0 | 0 | 129 | 0 | 71 |
