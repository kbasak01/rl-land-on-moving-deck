# Results — Phase 7: evaluation under shift and ablations (simulation only)

- **Simulation only** (PyBullet, gym-pybullet-drones Crazyflie 2.x). No sentence here describes real flight or real deck data.
- Deck motion: dmf's 3-DOF (heave, roll, pitch) JONSWAP response, Froude-scaled to the drone at **lambda = 1/25** (1 s model = 5 s full scale), except where the lambda arm says otherwise. Surge, sway and yaw are absent.
- State-based observations; the perception stand-in (noise, latency) is the only sensor model and is off except in section 4. Not vision.
- dmf's roll/pitch–heave phase defect (~90°) is carried, not fixed; the aft pad is sensitive to it, and the pad-at-CG arm (section 2) is its control.
- Success = all four frozen criteria (`configs/env/success.yaml`). Success is never pooled across sea states. Learned rows: IQM over 5 training seeds with the stratified-bootstrap 95 % CI (2 000 replicates, seed 20260926) and the per-seed range; baselines: rate with the Wilson 95 % CI and k/N. Every success table is followed by the six-class outcome breakdown.
- `oracle_gated` is a **commit-timing oracle (privileged)**: it reads the true future deck motion. It is never a deployable result.
- `pid_track_descend`, `pid_feedforward` and `oracle_gated` are in every method table.
- Carried into every table and verdict (P7-D1 §8):
  - Forecast methods: the forecaster's forecasts were in-sample during training, and the past-only ship-motion feed is an extra ideal sensor the other methods lack (P6-D1).
  - The residual methods descend harder than their base; no residual seed learned the post-contact throttle cut, and a lowvz-like descent was within the residual's authority (P6-D5, corrected at P6-D6 M1).
  - The learned methods land with a two-phase descent that the PID tuning space cannot express.
  - `bounce` is decided by the 50 ms contact-loss grace rule and is unstable at 240 Hz (P5-D1).
  - Recorded closing speed understates impact speed by about 7 % (read after the first contact substep's impulse; P5-D14), for every method alike.
  - Penetration is flagged at any contact substep, not only at first contact (P5-D14).
  - Up to 3 / 1 / 0 / 3 SS6 successes per 1 000 of residual_ppo / ppo_forecast / residual_ppo_forecast / ppo_sinusoid may depend on tunnelling overlap (P6-D5); ppo 2, sac 41 (all SS), pid_feedforward_lowvz_cut 1 (P5-D14).
  - Every hard landing of the four Phase 6 methods, ppo and pid_feedforward in e06 is a deck-tilt event (relative tilt > 15 deg with a nearly level drone); sac's and pid_track_descend's are mostly speed-driven (P6-D5).
  - SAC trained for 2 M env steps against 10 M for the PPO family (P3-D1 §5).
  - The regimes share realizations (P3-D1 §2); a regime-vs-regime difference is not a comparison of independent draws.
- Rendered from the CSVs under `results/e07/` by `rld.eval.report.render_results` (`scripts/report.py`); do not edit by hand. `scripts/report.py --check` re-renders and compares bytes.

## 1. Main matrix: aft pad, JONSWAP, lambda = 1/25

Learned rows: IQM over 5 training seeds [stratified-bootstrap 95 % CI]; per-seed range; N = seeds × episodes. Baselines: rate [Wilson 95 % CI] k/N, carried line for line from `results/e01` and `results/e01_lowvz_cut`. p95 for learned rows is the IQM of per-seed p95s (descriptive, not the relative-p95 statistic).

- Cells not run (recorded, never dropped): 200 of 200 not run: the static-pad list has no ship motion to feed a forecaster (StaticDeckMotion has no vessel or ship channels, and its t0 goes down to 0.51 s model, inside the 4.0 s model lookback)

### `id`: success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.2** [98.0, 98.5]; seeds 98.0–98.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **99.8** [98.8, 100.0]; seeds 98.5–100.0; N 5×200=1000 | **98.2** [96.3, 99.0]; seeds 96.0–99.0; N 5×200=1000 | **87.0** [81.7, 90.3]; seeds 80.0–91.0; N 5×200=1000 | **72.5** [60.2, 79.3]; seeds 55.5–81.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.5** [99.5, 99.8]; seeds 99.5–100.0; N 5×200=1000 | **96.2** [95.7, 97.8]; seeds 95.5–98.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.0** [97.5, 98.5]; seeds 97.5–98.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.5** [99.5, 99.5]; seeds 99.5–99.5; N 5×200=1000 | **96.8** [96.5, 98.0]; seeds 96.5–98.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **97.3** [96.7, 97.8]; seeds 96.5–98.0; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 96.0 [92.3, 98.0] 192/200 | 85.0 [79.4, 89.3] 170/200 | 80.0 [73.9, 85.0] 160/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 90.5 [85.6, 93.8] 181/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 | 95.5 [91.7, 97.6] 191/200 | 85.0 [79.4, 89.3] 170/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 98.0 [95.0, 99.2] 196/200 | 88.0 [82.8, 91.8] 176/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 89.5 [84.5, 93.0] 179/200 | 63.0 [56.1, 69.4] 126/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 89.0 [83.9, 92.6] 178/200 | 64.5 [57.7, 70.8] 129/200 |

### `id`: p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.265 [0.263, 0.266] · 0.0 · 0.0 | 0.267 [0.264, 0.270] · 0.0 · 0.0 | 0.271 [0.267, 0.275] · 0.0 · 0.0 | 0.278 [0.274, 0.285] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.392 [0.369, 0.433] · 0.0 · 0.0 | 0.437 [0.406, 0.478] · 0.0 · 0.0 | 0.595 [0.562, 0.619] · 0.2 · 0.1 | 0.668 [0.590, 0.778] · 1.8 · 0.0 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.269 [0.267, 0.271] · 0.0 · 0.0 | 0.271 [0.266, 0.273] · 0.0 · 0.0 | 0.277 [0.272, 0.279] · 0.0 · 0.0 | 0.283 [0.281, 0.287] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.268 [0.262, 0.271] · 0.0 · 0.0 | 0.269 [0.264, 0.271] · 0.0 · 0.0 | 0.272 [0.268, 0.275] · 0.0 · 0.0 | 0.281 [0.272, 0.295] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.277 [0.274, 0.281] · 0.0 · 0.0 | 0.279 [0.275, 0.281] · 0.0 · 0.0 | 0.278 [0.276, 0.280] · 0.0 · 0.0 | 0.278 [0.274, 0.280] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.274 [0.270, 0.277] · 0.0 · 0.0 | 0.275 [0.271, 0.279] · 0.0 · 0.0 | 0.282 [0.278, 0.285] · 0.0 · 0.0 | 0.288 [0.282, 0.292] · 0.0 · 0.0 |
| `pid_track_descend` | 0.308 · 0.0 · 0.0 | 0.412 · 0.0 · 0.0 | 0.525 · 0.0 · 0.0 | 0.502 · 0.0 · 0.0 |
| `pid_feedforward` | 0.219 · 0.0 · 0.0 | 0.226 · 0.0 · 0.0 | 0.262 · 0.0 · 0.0 | 0.264 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.127 · 0.0 · 0.0 | 0.139 · 0.0 · 0.0 | 0.188 · 0.0 · 0.0 | 0.181 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.127 · 0.0 · 0.0 | 0.139 · 0.0 · 0.0 | 0.188 · 0.0 · 0.0 | 0.181 · 0.0 · 0.0 |
| `gated` | 0.217 · 0.0 · 0.0 | 0.229 · 0.0 · 1.5 | 0.250 · 0.0 · 10.5 | 0.267 · 0.0 · 35.0 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.220 · 0.0 · 0.0 | 0.236 · 0.0 · 1.5 | 0.252 · 0.0 · 10.5 | 0.254 · 0.0 · 35.5 |

### `id`: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 15 (1.5) | 3 (0.3) | 982 (98.2) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 4 (0.4) | 0 (0.0) | 996 (99.6) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 1 (0.1) | 20 (2.0) | 0 (0.0) | 979 (97.9) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 2 (0.2) | 8 (0.8) | 122 (12.2) | 3 (0.3) | 864 (86.4) | 1 (0.1) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 18 (1.8) | 39 (3.9) | 232 (23.2) | 2 (0.2) | 709 (70.9) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 4 (0.4) | 0 (0.0) | 996 (99.6) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 24 (2.4) | 11 (1.1) | 965 (96.5) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 18 (1.8) | 2 (0.2) | 980 (98.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 0 (0.0) | 995 (99.5) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 25 (2.5) | 4 (0.4) | 971 (97.1) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 25 (2.5) | 2 (0.2) | 973 (97.3) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 4 (2.0) | 192 (96.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 12 (6.0) | 18 (9.0) | 170 (85.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 17 (8.5) | 23 (11.5) | 160 (80.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 1 (0.5) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 11 (5.5) | 181 (90.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 191 (95.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 21 (10.5) | 170 (85.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 15 (7.5) | 176 (88.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 197 (98.5) | 3 (1.5) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 179 (89.5) | 21 (10.5) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 126 (63.0) | 70 (35.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 197 (98.5) | 3 (1.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 178 (89.0) | 21 (10.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 129 (64.5) | 71 (35.5) |

### `unseen_seastate`: success per sea state

| method | SS6 |
|---|---|
| `ppo` (learned, PPO) | **97.3** [96.0, 98.2]; seeds 95.5–98.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **70.2** [62.2, 74.8]; seeds 58.5–76.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **95.3** [94.0, 97.2]; seeds 93.5–98.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **96.7** [96.2, 97.3]; seeds 96.0–97.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **95.8** [95.0, 96.5]; seeds 95.0–96.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **97.5** [96.5, 98.5]; seeds 96.5–98.5; N 5×200=1000 |
| `pid_track_descend` | 81.0 [75.0, 85.8] 162/200 |
| `pid_feedforward` | 90.0 [85.1, 93.4] 180/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 86.0 [80.5, 90.1] 172/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 89.5 [84.5, 93.0] 179/200 |
| `gated` | 64.5 [57.7, 70.8] 129/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 65.0 [58.2, 71.3] 130/200 |

### `unseen_seastate`: p95 closing speed (m/s), crash %, timeout %

| method | SS6: p95 · crash · timeout |
|---|---|
| `ppo` (learned, PPO) | 0.280 [0.272, 0.285] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.679 [0.615, 0.789] · 1.9 · 0.0 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.283 [0.277, 0.288] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.278 [0.273, 0.294] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.278 [0.274, 0.280] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.288 [0.285, 0.293] · 0.0 · 0.0 |
| `pid_track_descend` | 0.497 · 0.0 · 0.0 |
| `pid_feedforward` | 0.250 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.152 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.152 · 0.0 · 0.0 |
| `gated` | 0.243 · 0.0 · 34.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.240 · 0.0 · 35.0 |

### `unseen_seastate`: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 5 (0.5) | 972 (97.2) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 19 (1.9) | 46 (4.6) | 241 (24.1) | 3 (0.3) | 691 (69.1) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 38 (3.8) | 7 (0.7) | 955 (95.5) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 31 (3.1) | 2 (0.2) | 967 (96.7) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 35 (3.5) | 7 (0.7) | 958 (95.8) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 2 (0.2) | 975 (97.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 15 (7.5) | 23 (11.5) | 162 (81.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 14 (7.0) | 180 (90.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 21 (10.5) | 172 (86.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 14 (7.0) | 179 (89.5) | 0 (0.0) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 129 (64.5) | 69 (34.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 130 (65.0) | 70 (35.0) |

### `unseen_heading`: success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **90.7** [86.3, 93.3]; seeds 85.5–93.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **96.7** [95.8, 98.7]; seeds 95.5–99.5; N 5×200=1000 | **84.2** [76.8, 87.0]; seeds 75.0–87.5; N 5×200=1000 | **46.5** [37.0, 55.2]; seeds 35.5–56.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **85.7** [84.0, 89.0]; seeds 83.5–90.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **87.8** [84.7, 89.3]; seeds 84.0–89.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **87.5** [84.0, 90.3]; seeds 83.0–91.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **86.2** [84.8, 87.8]; seeds 84.5–88.5; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 | 72.5 [65.9, 78.2] 145/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 77.0 [70.7, 82.3] 154/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 96.5 [93.0, 98.3] 193/200 | 70.0 [63.3, 75.9] 140/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 97.0 [93.6, 98.6] 194/200 | 83.0 [77.2, 87.6] 166/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 | 42.0 [35.4, 48.9] 84/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 46.5 [39.7, 53.4] 93/200 |

### `unseen_heading`: p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.263 [0.262, 0.265] · 0.0 · 0.0 | 0.265 [0.263, 0.267] · 0.0 · 0.0 | 0.272 [0.268, 0.275] · 0.0 · 0.0 | 0.285 [0.275, 0.292] · 0.1 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.378 [0.356, 0.405] · 0.0 · 0.0 | 0.471 [0.440, 0.480] · 0.0 · 0.0 | 0.584 [0.550, 0.692] · 0.2 · 0.1 | 0.695 [0.612, 0.994] · 8.6 · 0.3 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.268 [0.267, 0.271] · 0.0 · 0.0 | 0.270 [0.267, 0.273] · 0.0 · 0.0 | 0.276 [0.272, 0.277] · 0.0 · 0.0 | 0.281 [0.273, 0.288] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.268 [0.262, 0.271] · 0.0 · 0.0 | 0.268 [0.261, 0.273] · 0.0 · 0.0 | 0.270 [0.266, 0.276] · 0.0 · 0.0 | 0.305 [0.284, 0.362] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.278 [0.275, 0.281] · 0.0 · 0.0 | 0.278 [0.274, 0.280] · 0.0 · 0.0 | 0.277 [0.275, 0.279] · 0.0 · 0.0 | 0.278 [0.271, 0.281] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.273 [0.269, 0.277] · 0.0 · 0.0 | 0.274 [0.269, 0.277] · 0.0 · 0.0 | 0.277 [0.274, 0.284] · 0.0 · 0.0 | 0.287 [0.281, 0.299] · 0.0 · 0.0 |
| `pid_track_descend` | 0.258 · 0.0 · 0.0 | 0.313 · 0.0 · 0.0 | 0.370 · 0.0 · 0.0 | 0.454 · 0.0 · 0.0 |
| `pid_feedforward` | 0.205 · 0.0 · 0.0 | 0.212 · 0.0 · 0.0 | 0.222 · 0.0 · 0.0 | 0.230 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.111 · 0.0 · 0.0 | 0.117 · 0.0 · 0.0 | 0.130 · 0.0 · 0.0 | 0.139 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.111 · 0.0 · 0.0 | 0.117 · 0.0 · 0.0 | 0.130 · 0.0 · 0.0 | 0.139 · 0.0 · 0.0 |
| `gated` | 0.204 · 0.0 · 0.0 | 0.210 · 0.0 · 0.0 | 0.219 · 0.0 · 1.5 | 0.218 · 0.0 · 54.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.204 · 0.0 · 0.0 | 0.210 · 0.0 · 0.0 | 0.220 · 0.0 · 0.5 | 0.215 · 0.0 · 53.5 |

### `unseen_heading`: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 1 (0.1) | 3 (0.3) | 77 (7.7) | 17 (1.7) | 902 (90.2) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 2 (0.2) | 28 (2.8) | 0 (0.0) | 970 (97.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 2 (0.2) | 23 (2.3) | 139 (13.9) | 5 (0.5) | 830 (83.0) | 1 (0.1) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 86 (8.6) | 98 (9.8) | 343 (34.3) | 8 (0.8) | 462 (46.2) | 3 (0.3) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 107 (10.7) | 31 (3.1) | 862 (86.2) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 113 (11.3) | 12 (1.2) | 874 (87.4) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 102 (10.2) | 25 (2.5) | 873 (87.3) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 116 (11.6) | 21 (2.1) | 863 (86.3) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 22 (11.0) | 33 (16.5) | 145 (72.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 24 (12.0) | 22 (11.0) | 154 (77.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 24 (12.0) | 36 (18.0) | 140 (70.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 24 (12.0) | 10 (5.0) | 166 (83.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 196 (98.0) | 3 (1.5) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 84 (42.0) | 109 (54.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 93 (46.5) | 107 (53.5) |

### `unseen_vessel`: success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **99.3** [98.7, 99.5]; seeds 98.5–99.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **99.2** [98.5, 100.0]; seeds 98.5–100.0; N 5×200=1000 | **96.5** [95.7, 98.0]; seeds 95.5–98.5; N 5×200=1000 | **81.5** [79.0, 84.7]; seeds 78.5–85.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.0** [98.7, 99.3]; seeds 98.5–99.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.3** [99.0, 99.5]; seeds 99.0–99.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.2** [98.7, 99.5]; seeds 98.5–99.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.5** [99.0, 100.0]; seeds 99.0–100.0; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 | 90.0 [85.1, 93.4] 180/200 | 81.5 [75.5, 86.3] 163/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 98.5 [95.7, 99.5] 197/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 95.0 [91.0, 97.3] 190/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 97.0 [93.6, 98.6] 194/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 93.5 [89.2, 96.2] 187/200 | 90.0 [85.1, 93.4] 180/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 95.0 [91.0, 97.3] 190/200 | 89.0 [83.9, 92.6] 178/200 |

### `unseen_vessel`: p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.263 [0.262, 0.265] · 0.0 · 0.0 | 0.266 [0.264, 0.269] · 0.0 · 0.0 | 0.268 [0.266, 0.272] · 0.0 · 0.0 | 0.277 [0.272, 0.285] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.373 [0.346, 0.415] · 0.0 · 0.0 | 0.415 [0.383, 0.440] · 0.0 · 0.0 | 0.461 [0.443, 0.468] · 0.0 · 0.0 | 0.603 [0.536, 0.635] · 0.3 · 0.1 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.268 [0.264, 0.270] · 0.0 · 0.0 | 0.269 [0.267, 0.270] · 0.0 · 0.0 | 0.273 [0.270, 0.274] · 0.0 · 0.0 | 0.278 [0.273, 0.283] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.267 [0.261, 0.270] · 0.0 · 0.0 | 0.269 [0.266, 0.271] · 0.0 · 0.0 | 0.271 [0.268, 0.273] · 0.0 · 0.0 | 0.275 [0.272, 0.284] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.278 [0.275, 0.281] · 0.0 · 0.0 | 0.277 [0.273, 0.281] · 0.0 · 0.0 | 0.277 [0.273, 0.280] · 0.0 · 0.0 | 0.278 [0.273, 0.280] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.272 [0.268, 0.276] · 0.0 · 0.0 | 0.274 [0.270, 0.279] · 0.0 · 0.0 | 0.276 [0.272, 0.279] · 0.0 · 0.0 | 0.283 [0.277, 0.286] · 0.0 · 0.0 |
| `pid_track_descend` | 0.300 · 0.0 · 0.0 | 0.388 · 0.0 · 0.0 | 0.520 · 0.0 · 0.0 | 0.546 · 0.0 · 0.0 |
| `pid_feedforward` | 0.211 · 0.0 · 0.0 | 0.228 · 0.0 · 0.0 | 0.255 · 0.0 · 0.0 | 0.260 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.120 · 0.0 · 0.0 | 0.138 · 0.0 · 0.0 | 0.155 · 0.0 · 0.0 | 0.159 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.120 · 0.0 · 0.0 | 0.138 · 0.0 · 0.0 | 0.155 · 0.0 · 0.0 | 0.159 · 0.0 · 0.0 |
| `gated` | 0.216 · 0.0 · 0.0 | 0.227 · 0.0 · 0.0 | 0.248 · 0.0 · 6.5 | 0.249 · 0.0 · 10.0 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.218 · 0.0 · 0.0 | 0.224 · 0.0 · 0.0 | 0.238 · 0.0 · 5.0 | 0.235 · 0.0 · 11.0 |

### `unseen_vessel`: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 0 (0.0) | 999 (99.9) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 7 (0.7) | 1 (0.1) | 992 (99.2) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 0 (0.0) | 999 (99.9) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 8 (0.8) | 0 (0.0) | 992 (99.2) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 0 (0.0) | 2 (0.2) | 30 (3.0) | 1 (0.1) | 967 (96.7) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 3 (0.3) | 23 (2.3) | 153 (15.3) | 3 (0.3) | 817 (81.7) | 1 (0.1) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 8 (0.8) | 2 (0.2) | 990 (99.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 1 (0.1) | 993 (99.3) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 3 (0.3) | 991 (99.1) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 0 (0.0) | 995 (99.5) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 14 (7.0) | 6 (3.0) | 180 (90.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 20 (10.0) | 17 (8.5) | 163 (81.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 2 (1.0) | 197 (98.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 187 (93.5) | 13 (6.5) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 180 (90.0) | 20 (10.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 190 (95.0) | 10 (5.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 178 (89.0) | 22 (11.0) |

### `static`: success per sea state

| method | static |
|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | not run (0/200; reason above) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | not run (0/200; reason above) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 |

### `static`: p95 closing speed (m/s), crash %, timeout %

| method | static: p95 · crash · timeout |
|---|---|
| `ppo` (learned, PPO) | 0.261 [0.261, 0.264] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.363 [0.314, 0.392] · 0.0 · 0.0 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.269 [0.266, 0.270] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | – |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | – |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.272 [0.267, 0.277] · 0.0 · 0.0 |
| `pid_track_descend` | 0.246 · 0.0 · 0.0 |
| `pid_feedforward` | 0.203 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.109 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.109 · 0.0 · 0.0 |
| `gated` | 0.204 · 0.0 · 0.0 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.204 · 0.0 · 0.0 |

### `static`: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | static | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | static | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | static | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | static | not run |  |  |  |  |  |  |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | static | not run |  |  |  |  |  |  |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | static | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `pid_track_descend` | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |

## 2. Pad at CG (control arm for dmf's phase defect; P7-D1 §5; descriptive)

Same listed episodes with the pad at the ship's CG. Learned methods and `pid_feedforward_lowvz_cut` flown here; the other baselines' CG rows carried from `results/e02`. aft − CG is the paired success contrast on identical episodes (`contrasts.csv`, shared episodes, 10 000 replicates), in points. No hypothesis is scored at CG.

### `id`, pad at CG: success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.8** [99.5, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **98.7** [97.8, 99.0]; seeds 97.5–99.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **99.7** [99.2, 100.0]; seeds 99.0–100.0; N 5×200=1000 | **94.5** [92.5, 98.2]; seeds 92.5–99.0; N 5×200=1000 | **71.7** [66.0, 82.7]; seeds 65.5–83.5; N 5×200=1000 | **48.7** [44.0, 64.7]; seeds 43.5–69.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.2** [99.0, 99.5]; seeds 99.0–99.5; N 5×200=1000 | **96.2** [95.0, 98.2]; seeds 95.0–98.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.8** [99.2, 100.0]; seeds 99.0–100.0; N 5×200=1000 | **97.7** [96.2, 98.3]; seeds 95.5–98.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.5** [99.2, 99.8]; seeds 99.0–100.0; N 5×200=1000 | **97.0** [95.0, 98.3]; seeds 94.5–98.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **97.5** [97.0, 98.3]; seeds 97.0–98.5; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 96.0 [92.3, 98.0] 192/200 | 77.0 [70.7, 82.3] 154/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 92.0 [87.4, 95.0] 184/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 95.0 [91.0, 97.3] 190/200 | 82.5 [76.6, 87.1] 165/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 96.5 [93.0, 98.3] 193/200 | 87.5 [82.2, 91.4] 175/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 92.0 [87.4, 95.0] 184/200 | 66.0 [59.2, 72.2] 132/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 91.0 [86.2, 94.2] 182/200 | 65.5 [58.7, 71.7] 131/200 |

### `id`, pad at CG: p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.264 [0.261, 0.266] · 0.0 · 0.0 | 0.271 [0.270, 0.274] · 0.0 · 0.0 | 0.289 [0.281, 0.293] · 0.0 · 0.0 | 0.295 [0.286, 0.304] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.396 [0.351, 0.432] · 0.0 · 0.0 | 0.493 [0.447, 0.527] · 0.0 · 0.0 | 0.659 [0.604, 0.733] · 0.9 · 0.0 | 0.776 [0.710, 0.852] · 4.8 · 0.4 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.269 [0.267, 0.271] · 0.0 · 0.0 | 0.277 [0.274, 0.279] · 0.0 · 0.0 | 0.287 [0.283, 0.292] · 0.0 · 0.0 | 0.305 [0.298, 0.311] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.269 [0.262, 0.272] · 0.0 · 0.0 | 0.270 [0.263, 0.273] · 0.0 · 0.0 | 0.272 [0.266, 0.278] · 0.0 · 0.0 | 0.288 [0.280, 0.300] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.277 [0.274, 0.280] · 0.0 · 0.0 | 0.278 [0.275, 0.280] · 0.0 · 0.0 | 0.279 [0.274, 0.280] · 0.0 · 0.0 | 0.279 [0.274, 0.280] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.274 [0.269, 0.278] · 0.0 · 0.0 | 0.283 [0.277, 0.289] · 0.0 · 0.0 | 0.296 [0.291, 0.304] · 0.0 · 0.0 | 0.304 [0.300, 0.311] · 0.0 · 0.0 |
| `pid_track_descend` | 0.263 · 0.0 · 0.0 | 0.307 · 0.0 · 0.0 | 0.403 · 0.0 · 0.0 | 0.507 · 0.0 · 0.0 |
| `pid_feedforward` | 0.204 · 0.0 · 0.0 | 0.210 · 0.0 · 0.0 | 0.229 · 0.0 · 0.0 | 0.243 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.111 · 0.0 · 0.0 | 0.121 · 0.0 · 0.0 | 0.131 · 0.0 · 0.0 | 0.144 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.111 · 0.0 · 0.0 | 0.121 · 0.0 · 0.0 | 0.131 · 0.0 · 0.0 | 0.144 · 0.0 · 0.0 |
| `gated` | 0.205 · 0.0 · 0.0 | 0.212 · 0.0 · 1.0 | 0.225 · 0.0 · 7.5 | 0.231 · 0.0 · 32.0 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.205 · 0.0 · 0.0 | 0.216 · 0.0 · 1.0 | 0.222 · 0.0 · 9.0 | 0.226 · 0.0 · 34.5 |

### `id`, pad at CG: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 0 (0.0) | 998 (99.8) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 12 (1.2) | 3 (0.3) | 985 (98.5) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 1 (0.1) | 996 (99.6) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 1 (0.1) | 47 (4.7) | 2 (0.2) | 950 (95.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 9 (0.9) | 33 (3.3) | 225 (22.5) | 5 (0.5) | 728 (72.8) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 48 (4.8) | 102 (10.2) | 327 (32.7) | 1 (0.1) | 518 (51.8) | 4 (0.4) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 3 (0.3) | 992 (99.2) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 27 (2.7) | 9 (0.9) | 964 (96.4) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 1 (0.1) | 997 (99.7) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 21 (2.1) | 4 (0.4) | 974 (97.4) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 2 (0.2) | 995 (99.5) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 11 (1.1) | 968 (96.8) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 0 (0.0) | 999 (99.9) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 1 (0.1) | 976 (97.6) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 7 (3.5) | 192 (96.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 19 (9.5) | 27 (13.5) | 154 (77.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 10 (5.0) | 184 (92.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 8 (4.0) | 190 (95.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 28 (14.0) | 165 (82.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 5 (2.5) | 193 (96.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 18 (9.0) | 175 (87.5) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 184 (92.0) | 15 (7.5) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 132 (66.0) | 64 (32.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 182 (91.0) | 18 (9.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 131 (65.5) | 69 (34.5) |

### `id`: aft − CG

| method | SS | aft | CG | aft − CG, points [95 % CI] |
|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.8** [99.5, 100.0]; seeds 99.5–100.0; N 5×200=1000 | +0.2 [+0.0, +0.8] |
| `ppo` (learned, PPO) | SS6 | **98.2** [98.0, 98.5]; seeds 98.0–98.5; N 5×200=1000 | **98.7** [97.8, 99.0]; seeds 97.5–99.0; N 5×200=1000 | -0.5 [-1.7, +1.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | **99.8** [98.8, 100.0]; seeds 98.5–100.0; N 5×200=1000 | **99.7** [99.2, 100.0]; seeds 99.0–100.0; N 5×200=1000 | +0.2 [-0.8, +1.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | **98.2** [96.3, 99.0]; seeds 96.0–99.0; N 5×200=1000 | **94.5** [92.5, 98.2]; seeds 92.5–99.0; N 5×200=1000 | +3.7 [-0.3, +6.8] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | **87.0** [81.7, 90.3]; seeds 80.0–91.0; N 5×200=1000 | **71.7** [66.0, 82.7]; seeds 65.5–83.5; N 5×200=1000 | +15.3 [+4.3, +24.7] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | **72.5** [60.2, 79.3]; seeds 55.5–81.5; N 5×200=1000 | **48.7** [44.0, 64.7]; seeds 43.5–69.5; N 5×200=1000 | +23.8 [+11.3, +30.5] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | **99.5** [99.5, 99.8]; seeds 99.5–100.0; N 5×200=1000 | **99.2** [99.0, 99.5]; seeds 99.0–99.5; N 5×200=1000 | +0.3 [-0.2, +1.5] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | **96.2** [95.7, 97.8]; seeds 95.5–98.5; N 5×200=1000 | **96.2** [95.0, 98.2]; seeds 95.0–98.5; N 5×200=1000 | +0.0 [-2.5, +2.8] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.8** [99.2, 100.0]; seeds 99.0–100.0; N 5×200=1000 | +0.2 [+0.0, +1.2] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | **98.0** [97.5, 98.5]; seeds 97.5–98.5; N 5×200=1000 | **97.7** [96.2, 98.3]; seeds 95.5–98.5; N 5×200=1000 | +0.3 [-0.8, +2.2] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | **99.5** [99.5, 99.5]; seeds 99.5–99.5; N 5×200=1000 | **99.5** [99.2, 99.8]; seeds 99.0–100.0; N 5×200=1000 | +0.0 [-0.8, +0.7] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | **96.8** [96.5, 98.0]; seeds 96.5–98.5; N 5×200=1000 | **97.0** [95.0, 98.3]; seeds 94.5–98.5; N 5×200=1000 | -0.2 [-2.0, +2.7] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | +0.0 [+0.0, +0.5] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | **97.3** [96.7, 97.8]; seeds 96.5–98.0; N 5×200=1000 | **97.5** [97.0, 98.3]; seeds 97.0–98.5; N 5×200=1000 | -0.2 [-1.3, +0.8] |
| `pid_track_descend` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS4 | 96.0 [92.3, 98.0] 192/200 | 99.5 [97.2, 99.9] 199/200 | -3.5 [-6.5, -0.5] |
| `pid_track_descend` | SS5 | 85.0 [79.4, 89.3] 170/200 | 96.0 [92.3, 98.0] 192/200 | -11.0 [-17.0, -5.5] |
| `pid_track_descend` | SS6 | 80.0 [73.9, 85.0] 160/200 | 77.0 [70.7, 82.3] 154/200 | +3.0 [-4.0, +10.0] |
| `pid_feedforward` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS5 | 99.0 [96.4, 99.7] 198/200 | 98.5 [95.7, 99.5] 197/200 | +0.5 [-1.0, +2.0] |
| `pid_feedforward` | SS6 | 90.5 [85.6, 93.8] 181/200 | 92.0 [87.4, 95.0] 184/200 | -1.5 [-6.0, +2.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 98.0 [95.0, 99.2] 196/200 | 98.5 [95.7, 99.5] 197/200 | -0.5 [-3.0, +1.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 95.5 [91.7, 97.6] 191/200 | 95.0 [91.0, 97.3] 190/200 | +0.5 [-3.0, +4.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 85.0 [79.4, 89.3] 170/200 | 82.5 [76.6, 87.1] 165/200 | +2.5 [-4.0, +9.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 98.5 [95.7, 99.5] 197/200 | 99.0 [96.4, 99.7] 198/200 | -0.5 [-2.5, +1.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 98.0 [95.0, 99.2] 196/200 | 96.5 [93.0, 98.3] 193/200 | +1.5 [-1.0, +4.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 88.0 [82.8, 91.8] 176/200 | 87.5 [82.2, 91.4] 175/200 | +0.5 [-5.5, +6.5] |
| `gated` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `gated` | SS4 | 98.5 [95.7, 99.5] 197/200 | 99.0 [96.4, 99.7] 198/200 | -0.5 [-1.5, +0.0] |
| `gated` | SS5 | 89.5 [84.5, 93.0] 179/200 | 92.0 [87.4, 95.0] 184/200 | -2.5 [-5.0, +0.0] |
| `gated` | SS6 | 63.0 [56.1, 69.4] 126/200 | 66.0 [59.2, 72.2] 132/200 | -3.0 [-6.5, +0.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 98.5 [95.7, 99.5] 197/200 | 99.0 [96.4, 99.7] 198/200 | -0.5 [-1.5, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 89.0 [83.9, 92.6] 178/200 | 91.0 [86.2, 94.2] 182/200 | -2.0 [-4.5, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 64.5 [57.7, 70.8] 129/200 | 65.5 [58.7, 71.7] 131/200 | -1.0 [-5.0, +3.0] |

### `unseen_seastate`, pad at CG: success per sea state

| method | SS6 |
|---|---|
| `ppo` (learned, PPO) | **96.7** [96.2, 97.0]; seeds 96.0–97.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **50.0** [44.5, 60.0]; seeds 44.0–61.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **96.0** [94.3, 97.3]; seeds 94.0–97.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **96.8** [95.5, 97.5]; seeds 95.0–97.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **95.3** [94.7, 96.7]; seeds 94.5–97.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **97.3** [96.2, 98.3]; seeds 96.0–98.5; N 5×200=1000 |
| `pid_track_descend` | 83.5 [77.7, 88.0] 167/200 |
| `pid_feedforward` | 92.0 [87.4, 95.0] 184/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 88.5 [83.3, 92.2] 177/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 92.5 [88.0, 95.4] 185/200 |
| `gated` | 63.5 [56.6, 69.9] 127/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 65.5 [58.7, 71.7] 131/200 |

### `unseen_seastate`, pad at CG: p95 closing speed (m/s), crash %, timeout %

| method | SS6: p95 · crash · timeout |
|---|---|
| `ppo` (learned, PPO) | 0.290 [0.284, 0.301] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.769 [0.756, 0.897] · 6.7 · 0.1 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.297 [0.294, 0.303] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.287 [0.278, 0.311] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.277 [0.274, 0.279] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.301 [0.295, 0.303] · 0.0 · 0.0 |
| `pid_track_descend` | 0.466 · 0.0 · 0.0 |
| `pid_feedforward` | 0.234 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.143 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.143 · 0.0 · 0.0 |
| `gated` | 0.230 · 0.0 · 34.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.219 · 0.0 · 34.5 |

### `unseen_seastate`, pad at CG: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 19 (1.9) | 15 (1.5) | 966 (96.6) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 67 (6.7) | 91 (9.1) | 328 (32.8) | 2 (0.2) | 511 (51.1) | 1 (0.1) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 36 (3.6) | 5 (0.5) | 959 (95.9) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 2 (0.2) | 28 (2.8) | 4 (0.4) | 966 (96.6) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 38 (3.8) | 7 (0.7) | 955 (95.5) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 4 (0.4) | 973 (97.3) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 24 (12.0) | 167 (83.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 10 (5.0) | 184 (92.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 15 (7.5) | 177 (88.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 7 (3.5) | 185 (92.5) | 0 (0.0) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 127 (63.5) | 69 (34.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 131 (65.5) | 69 (34.5) |

### `unseen_seastate`: aft − CG

| method | SS | aft | CG | aft − CG, points [95 % CI] |
|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | **97.3** [96.0, 98.2]; seeds 95.5–98.5; N 5×200=1000 | **96.7** [96.2, 97.0]; seeds 96.0–97.0; N 5×200=1000 | +0.7 [-1.3, +2.3] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | **70.2** [62.2, 74.8]; seeds 58.5–76.5; N 5×200=1000 | **50.0** [44.5, 60.0]; seeds 44.0–61.5; N 5×200=1000 | +20.2 [+10.5, +27.8] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | **95.3** [94.0, 97.2]; seeds 93.5–98.0; N 5×200=1000 | **96.0** [94.3, 97.3]; seeds 94.0–97.5; N 5×200=1000 | -0.7 [-2.2, +1.2] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | **96.7** [96.2, 97.3]; seeds 96.0–97.5; N 5×200=1000 | **96.8** [95.5, 97.5]; seeds 95.0–97.5; N 5×200=1000 | -0.2 [-1.5, +1.7] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | **95.8** [95.0, 96.5]; seeds 95.0–96.5; N 5×200=1000 | **95.3** [94.7, 96.7]; seeds 94.5–97.0; N 5×200=1000 | +0.5 [-1.8, +2.7] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | **97.5** [96.5, 98.5]; seeds 96.5–98.5; N 5×200=1000 | **97.3** [96.2, 98.3]; seeds 96.0–98.5; N 5×200=1000 | +0.2 [-1.5, +2.0] |
| `pid_track_descend` | SS6 | 81.0 [75.0, 85.8] 162/200 | 83.5 [77.7, 88.0] 167/200 | -2.5 [-9.0, +4.0] |
| `pid_feedforward` | SS6 | 90.0 [85.1, 93.4] 180/200 | 92.0 [87.4, 95.0] 184/200 | -2.0 [-6.5, +2.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 86.0 [80.5, 90.1] 172/200 | 88.5 [83.3, 92.2] 177/200 | -2.5 [-8.0, +3.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 89.5 [84.5, 93.0] 179/200 | 92.5 [88.0, 95.4] 185/200 | -3.0 [-8.0, +2.0] |
| `gated` | SS6 | 64.5 [57.7, 70.8] 129/200 | 63.5 [56.6, 69.9] 127/200 | +1.0 [-2.5, +4.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 65.0 [58.2, 71.3] 130/200 | 65.5 [58.7, 71.7] 131/200 | -0.5 [-4.5, +3.5] |

### `unseen_heading`, pad at CG: success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **90.3** [85.5, 93.7]; seeds 85.0–94.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **96.2** [94.5, 97.8]; seeds 94.0–98.0; N 5×200=1000 | **80.7** [73.0, 85.3]; seeds 71.0–86.0; N 5×200=1000 | **44.2** [38.2, 51.2]; seeds 37.5–52.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **86.0** [83.3, 90.3]; seeds 83.0–92.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **87.0** [85.2, 88.8]; seeds 84.5–89.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **88.3** [85.2, 90.5]; seeds 84.5–91.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **86.8** [85.7, 88.3]; seeds 85.5–88.5; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 70.5 [63.8, 76.4] 141/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 76.0 [69.6, 81.4] 152/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 95.5 [91.7, 97.6] 191/200 | 70.5 [63.8, 76.4] 141/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 97.0 [93.6, 98.6] 194/200 | 81.0 [75.0, 85.8] 162/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 41.0 [34.4, 47.9] 82/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 44.5 [37.8, 51.4] 89/200 |

### `unseen_heading`, pad at CG: p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.265 [0.262, 0.265] · 0.0 · 0.0 | 0.265 [0.263, 0.266] · 0.0 · 0.0 | 0.271 [0.266, 0.272] · 0.0 · 0.0 | 0.287 [0.277, 0.297] · 0.1 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.389 [0.348, 0.416] · 0.0 · 0.0 | 0.475 [0.439, 0.492] · 0.0 · 0.0 | 0.594 [0.563, 0.635] · 0.9 · 0.1 | 0.710 [0.651, 0.981] · 9.9 · 0.5 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.268 [0.265, 0.272] · 0.0 · 0.0 | 0.271 [0.270, 0.274] · 0.0 · 0.0 | 0.278 [0.276, 0.280] · 0.0 · 0.0 | 0.282 [0.276, 0.287] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.268 [0.262, 0.270] · 0.0 · 0.0 | 0.269 [0.262, 0.273] · 0.0 · 0.0 | 0.271 [0.267, 0.277] · 0.0 · 0.0 | 0.311 [0.292, 0.352] · 0.1 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.279 [0.275, 0.280] · 0.0 · 0.0 | 0.278 [0.273, 0.281] · 0.0 · 0.0 | 0.278 [0.273, 0.281] · 0.0 · 0.0 | 0.277 [0.271, 0.278] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.273 [0.269, 0.276] · 0.0 · 0.0 | 0.273 [0.269, 0.278] · 0.0 · 0.0 | 0.278 [0.275, 0.282] · 0.0 · 0.0 | 0.290 [0.284, 0.305] · 0.1 · 0.0 |
| `pid_track_descend` | 0.265 · 0.0 · 0.0 | 0.329 · 0.0 · 0.0 | 0.386 · 0.0 · 0.0 | 0.489 · 0.0 · 0.0 |
| `pid_feedforward` | 0.204 · 0.0 · 0.0 | 0.211 · 0.0 · 0.0 | 0.226 · 0.0 · 0.0 | 0.234 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.113 · 0.0 · 0.0 | 0.120 · 0.0 · 0.0 | 0.131 · 0.0 · 0.0 | 0.138 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.113 · 0.0 · 0.0 | 0.120 · 0.0 · 0.0 | 0.131 · 0.0 · 0.0 | 0.138 · 0.0 · 0.0 |
| `gated` | 0.204 · 0.0 · 0.0 | 0.211 · 0.0 · 0.0 | 0.218 · 0.0 · 1.5 | 0.224 · 0.0 · 54.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.204 · 0.0 · 0.0 | 0.214 · 0.0 · 0.0 | 0.224 · 0.0 · 0.5 | 0.219 · 0.0 · 55.5 |

### `unseen_heading`, pad at CG: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 1 (0.1) | 3 (0.3) | 80 (8.0) | 16 (1.6) | 900 (90.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 3 (0.3) | 35 (3.5) | 1 (0.1) | 961 (96.1) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 9 (0.9) | 33 (3.3) | 159 (15.9) | 0 (0.0) | 798 (79.8) | 1 (0.1) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 99 (9.9) | 114 (11.4) | 329 (32.9) | 9 (0.9) | 444 (44.4) | 5 (0.5) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 113 (11.3) | 21 (2.1) | 866 (86.6) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 1 (0.1) | 0 (0.0) | 116 (11.6) | 13 (1.3) | 870 (87.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 99 (9.9) | 20 (2.0) | 881 (88.1) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 1 (0.1) | 0 (0.0) | 118 (11.8) | 12 (1.2) | 869 (86.9) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 27 (13.5) | 32 (16.0) | 141 (70.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 26 (13.0) | 22 (11.0) | 152 (76.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 191 (95.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 25 (12.5) | 34 (17.0) | 141 (70.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 25 (12.5) | 13 (6.5) | 162 (81.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 197 (98.5) | 3 (1.5) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 82 (41.0) | 109 (54.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 89 (44.5) | 111 (55.5) |

### `unseen_heading`: aft − CG

| method | SS | aft | CG | aft − CG, points [95 % CI] |
|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS6 | **90.7** [86.3, 93.3]; seeds 85.5–93.5; N 5×200=1000 | **90.3** [85.5, 93.7]; seeds 85.0–94.0; N 5×200=1000 | +0.3 [-1.8, +2.5] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | **96.7** [95.8, 98.7]; seeds 95.5–99.5; N 5×200=1000 | **96.2** [94.5, 97.8]; seeds 94.0–98.0; N 5×200=1000 | +0.5 [-1.7, +3.3] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | **84.2** [76.8, 87.0]; seeds 75.0–87.5; N 5×200=1000 | **80.7** [73.0, 85.3]; seeds 71.0–86.0; N 5×200=1000 | +3.5 [-1.7, +9.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | **46.5** [37.0, 55.2]; seeds 35.5–56.0; N 5×200=1000 | **44.2** [38.2, 51.2]; seeds 37.5–52.0; N 5×200=1000 | +2.3 [-3.0, +6.8] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | **85.7** [84.0, 89.0]; seeds 83.5–90.5; N 5×200=1000 | **86.0** [83.3, 90.3]; seeds 83.0–92.0; N 5×200=1000 | -0.3 [-3.0, +2.5] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | **87.8** [84.7, 89.3]; seeds 84.0–89.5; N 5×200=1000 | **87.0** [85.2, 88.8]; seeds 84.5–89.5; N 5×200=1000 | +0.8 [-2.5, +3.5] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | **87.5** [84.0, 90.3]; seeds 83.0–91.0; N 5×200=1000 | **88.3** [85.2, 90.5]; seeds 84.5–91.0; N 5×200=1000 | -0.8 [-3.5, +1.8] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | **86.2** [84.8, 87.8]; seeds 84.5–88.5; N 5×200=1000 | **86.8** [85.7, 88.3]; seeds 85.5–88.5; N 5×200=1000 | -0.7 [-3.5, +2.2] |
| `pid_track_descend` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS5 | 97.5 [94.3, 98.9] 195/200 | 99.0 [96.4, 99.7] 198/200 | -1.5 [-4.0, +0.5] |
| `pid_track_descend` | SS6 | 72.5 [65.9, 78.2] 145/200 | 70.5 [63.8, 76.4] 141/200 | +2.0 [-3.5, +7.5] |
| `pid_feedforward` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS5 | 99.5 [97.2, 99.9] 199/200 | 99.0 [96.4, 99.7] 198/200 | +0.5 [-1.0, +2.0] |
| `pid_feedforward` | SS6 | 77.0 [70.7, 82.3] 154/200 | 76.0 [69.6, 81.4] 152/200 | +1.0 [-4.5, +6.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 99.0 [96.4, 99.7] 198/200 | 100.0 [98.1, 100.0] 200/200 | -1.0 [-2.5, +0.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 96.5 [93.0, 98.3] 193/200 | 95.5 [91.7, 97.6] 191/200 | +1.0 [-3.0, +5.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 70.0 [63.3, 75.9] 140/200 | 70.5 [63.8, 76.4] 141/200 | -0.5 [-7.5, +6.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 99.0 [96.4, 99.7] 198/200 | 100.0 [98.1, 100.0] 200/200 | -1.0 [-2.5, +0.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 97.0 [93.6, 98.6] 194/200 | 97.0 [93.6, 98.6] 194/200 | +0.0 [-3.5, +3.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 83.0 [77.2, 87.6] 166/200 | 81.0 [75.0, 85.8] 162/200 | +2.0 [-3.0, +7.0] |
| `gated` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `gated` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `gated` | SS5 | 98.0 [95.0, 99.2] 196/200 | 98.5 [95.7, 99.5] 197/200 | -0.5 [-1.5, +0.0] |
| `gated` | SS6 | 42.0 [35.4, 48.9] 84/200 | 41.0 [34.4, 47.9] 82/200 | +1.0 [-1.5, +3.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 99.5 [97.2, 99.9] 199/200 | 99.5 [97.2, 99.9] 199/200 | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 46.5 [39.7, 53.4] 93/200 | 44.5 [37.8, 51.4] 89/200 | +2.0 [+0.0, +4.5] |

### `unseen_vessel`, pad at CG: success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.3** [99.0, 99.5]; seeds 99.0–99.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.5** [96.8, 100.0]; seeds 96.5–100.0; N 5×200=1000 | **88.0** [82.0, 93.5]; seeds 81.0–93.5; N 5×200=1000 | **64.5** [56.0, 71.2]; seeds 53.0–72.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.3** [99.0, 99.5]; seeds 99.0–99.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.2** [99.0, 99.8]; seeds 99.0–100.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.8** [98.5, 99.3]; seeds 98.5–99.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.7** [99.5, 100.0]; seeds 99.5–100.0; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 95.0 [91.0, 97.3] 190/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 97.5 [94.3, 98.9] 195/200 | 95.0 [91.0, 97.3] 190/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 97.5 [94.3, 98.9] 195/200 | 96.5 [93.0, 98.3] 193/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 91.0 [86.2, 94.2] 182/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 88.5 [83.3, 92.2] 177/200 |

### `unseen_vessel`, pad at CG: p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.264 [0.261, 0.265] · 0.0 · 0.0 | 0.271 [0.268, 0.274] · 0.0 · 0.0 | 0.276 [0.272, 0.280] · 0.0 · 0.0 | 0.291 [0.283, 0.296] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.372 [0.317, 0.395] · 0.0 · 0.0 | 0.443 [0.401, 0.474] · 0.0 · 0.0 | 0.559 [0.506, 0.652] · 0.2 · 0.0 | 0.727 [0.663, 0.768] · 1.8 · 0.1 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.269 [0.265, 0.273] · 0.0 · 0.0 | 0.273 [0.270, 0.276] · 0.0 · 0.0 | 0.281 [0.280, 0.283] · 0.0 · 0.0 | 0.301 [0.296, 0.306] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.268 [0.263, 0.271] · 0.0 · 0.0 | 0.268 [0.260, 0.271] · 0.0 · 0.0 | 0.269 [0.263, 0.272] · 0.0 · 0.0 | 0.285 [0.278, 0.289] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.278 [0.273, 0.280] · 0.0 · 0.0 | 0.278 [0.274, 0.281] · 0.0 · 0.0 | 0.279 [0.276, 0.281] · 0.0 · 0.0 | 0.279 [0.273, 0.280] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.272 [0.269, 0.277] · 0.0 · 0.0 | 0.278 [0.271, 0.283] · 0.0 · 0.0 | 0.286 [0.281, 0.292] · 0.0 · 0.0 | 0.303 [0.296, 0.306] · 0.0 · 0.0 |
| `pid_track_descend` | 0.252 · 0.0 · 0.0 | 0.283 · 0.0 · 0.0 | 0.347 · 0.0 · 0.0 | 0.456 · 0.0 · 0.0 |
| `pid_feedforward` | 0.205 · 0.0 · 0.0 | 0.207 · 0.0 · 0.0 | 0.213 · 0.0 · 0.0 | 0.227 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.110 · 0.0 · 0.0 | 0.113 · 0.0 · 0.0 | 0.122 · 0.0 · 0.0 | 0.141 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.110 · 0.0 · 0.0 | 0.113 · 0.0 · 0.0 | 0.122 · 0.0 · 0.0 | 0.141 · 0.0 · 0.0 |
| `gated` | 0.203 · 0.0 · 0.0 | 0.208 · 0.0 · 0.0 | 0.213 · 0.0 · 1.5 | 0.223 · 0.0 · 9.0 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.203 · 0.0 · 0.0 | 0.208 · 0.0 · 0.0 | 0.216 · 0.0 · 1.0 | 0.228 · 0.0 · 11.5 |

### `unseen_vessel`, pad at CG: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 2 (0.2) | 993 (99.3) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 16 (1.6) | 0 (0.0) | 984 (98.4) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 2 (0.2) | 12 (1.2) | 109 (10.9) | 0 (0.0) | 877 (87.7) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 18 (1.8) | 78 (7.8) | 260 (26.0) | 5 (0.5) | 638 (63.8) | 1 (0.1) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 1 (0.1) | 993 (99.3) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 5 (0.5) | 1 (0.1) | 993 (99.3) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 10 (1.0) | 1 (0.1) | 989 (98.9) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 0 (0.0) | 997 (99.7) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 6 (3.0) | 190 (95.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 1 (0.5) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 197 (98.5) | 3 (1.5) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 182 (91.0) | 18 (9.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 177 (88.5) | 23 (11.5) |

### `unseen_vessel`: aft − CG

| method | SS | aft | CG | aft − CG, points [95 % CI] |
|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS5 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [-0.5, +0.0] |
| `ppo` (learned, PPO) | SS6 | **99.3** [98.7, 99.5]; seeds 98.5–99.5; N 5×200=1000 | **99.3** [99.0, 99.5]; seeds 99.0–99.5; N 5×200=1000 | +0.0 [-1.5, +1.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [-0.5, +0.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | **99.2** [98.5, 100.0]; seeds 98.5–100.0; N 5×200=1000 | **98.5** [96.8, 100.0]; seeds 96.5–100.0; N 5×200=1000 | +0.7 [-1.2, +3.2] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | **96.5** [95.7, 98.0]; seeds 95.5–98.5; N 5×200=1000 | **88.0** [82.0, 93.5]; seeds 81.0–93.5; N 5×200=1000 | +8.5 [+1.8, +16.3] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | **81.5** [79.0, 84.7]; seeds 78.5–85.5; N 5×200=1000 | **64.5** [56.0, 71.2]; seeds 53.0–72.5; N 5×200=1000 | +17.0 [+8.2, +29.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | **99.0** [98.7, 99.3]; seeds 98.5–99.5; N 5×200=1000 | **99.3** [99.0, 99.5]; seeds 99.0–99.5; N 5×200=1000 | -0.3 [-1.5, +0.7] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | **99.3** [99.0, 99.5]; seeds 99.0–99.5; N 5×200=1000 | **99.2** [99.0, 99.8]; seeds 99.0–100.0; N 5×200=1000 | +0.2 [-1.0, +0.8] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | **99.2** [98.7, 99.5]; seeds 98.5–99.5; N 5×200=1000 | **98.8** [98.5, 99.3]; seeds 98.5–99.5; N 5×200=1000 | +0.3 [-0.8, +1.5] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | **99.5** [99.0, 100.0]; seeds 99.0–100.0; N 5×200=1000 | **99.7** [99.5, 100.0]; seeds 99.5–100.0; N 5×200=1000 | -0.2 [-1.0, +0.5] |
| `pid_track_descend` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS4 | 97.5 [94.3, 98.9] 195/200 | 100.0 [98.1, 100.0] 200/200 | -2.5 [-5.0, -0.5] |
| `pid_track_descend` | SS5 | 90.0 [85.1, 93.4] 180/200 | 100.0 [98.1, 100.0] 200/200 | -10.0 [-14.5, -6.0] |
| `pid_track_descend` | SS6 | 81.5 [75.5, 86.3] 163/200 | 95.0 [91.0, 97.3] 190/200 | -13.5 [-20.0, -7.5] |
| `pid_feedforward` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS5 | 99.5 [97.2, 99.9] 199/200 | 100.0 [98.1, 100.0] 200/200 | -0.5 [-1.5, +0.0] |
| `pid_feedforward` | SS6 | 98.5 [95.7, 99.5] 197/200 | 99.0 [96.4, 99.7] 198/200 | -0.5 [-2.5, +1.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | +0.5 [+0.0, +1.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 99.5 [97.2, 99.9] 199/200 | 97.5 [94.3, 98.9] 195/200 | +2.0 [+0.0, +4.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 95.0 [91.0, 97.3] 190/200 | 95.0 [91.0, 97.3] 190/200 | +0.0 [-3.5, +4.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | +0.5 [+0.0, +1.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 99.5 [97.2, 99.9] 199/200 | 97.5 [94.3, 98.9] 195/200 | +2.0 [+0.0, +4.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 97.0 [93.6, 98.6] 194/200 | 96.5 [93.0, 98.3] 193/200 | +0.5 [-2.5, +4.0] |
| `gated` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `gated` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `gated` | SS5 | 93.5 [89.2, 96.2] 187/200 | 98.5 [95.7, 99.5] 197/200 | -5.0 [-8.0, -2.0] |
| `gated` | SS6 | 90.0 [85.1, 93.4] 180/200 | 91.0 [86.2, 94.2] 182/200 | -1.0 [-4.5, +2.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 95.0 [91.0, 97.3] 190/200 | 99.0 [96.4, 99.7] 198/200 | -4.0 [-7.0, -1.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 89.0 [83.9, 92.6] 178/200 | 88.5 [83.3, 92.2] 177/200 | +0.5 [-3.5, +4.5] |

## 3. Sinusoid test motion (the H4 cross; P7-D1 §1), `id`, aft pad

Each listed episode flown on the matched sinusoid built by the training builder (`rld.rl.motion.sinusoid_motion`: amplitude √2 × committed aft RMS, peak encounter period, phases from the episode seed); same start offsets and initial states. H4 is read at `id` SS5 only (section 7); the other cells are descriptive. JONSWAP − sinusoid is the paired success contrast, in points.

### `id`, sinusoid motion: success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.0** [98.3, 99.3]; seeds 98.0–99.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **99.3** [99.0, 99.8]; seeds 99.0–100.0; N 5×200=1000 | **96.5** [91.5, 97.8]; seeds 89.5–98.0; N 5×200=1000 | **83.2** [74.5, 87.7]; seeds 72.5–88.5; N 5×200=1000 | **61.0** [50.5, 73.0]; seeds 47.0–76.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.7** [97.3, 99.5]; seeds 97.0–99.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **97.8** [96.7, 98.5]; seeds 96.5–98.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.7** [97.5, 99.7]; seeds 97.0–100.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.3** [98.5, 100.0]; seeds 98.5–100.0; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 93.5 [89.2, 96.2] 187/200 | 64.5 [57.7, 70.8] 129/200 | 60.0 [53.1, 66.5] 120/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 99.0 [96.4, 99.7] 198/200 | 94.0 [89.8, 96.5] 188/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 99.0 [96.4, 99.7] 198/200 | 97.5 [94.3, 98.9] 195/200 | 95.0 [91.0, 97.3] 190/200 | 84.5 [78.8, 88.9] 169/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 99.0 [96.4, 99.7] 198/200 | 98.5 [95.7, 99.5] 197/200 | 97.0 [93.6, 98.6] 194/200 | 88.5 [83.3, 92.2] 177/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 75.5 [69.1, 80.9] 151/200 | 29.0 [23.2, 35.6] 58/200 | 2.0 [0.8, 5.0] 4/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 75.0 [68.6, 80.5] 150/200 | 28.5 [22.7, 35.1] 57/200 | 2.0 [0.8, 5.0] 4/200 |

### `id`, sinusoid motion: p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.265 [0.261, 0.267] · 0.0 · 0.0 | 0.269 [0.267, 0.270] · 0.0 · 0.0 | 0.274 [0.272, 0.276] · 0.0 · 0.0 | 0.282 [0.279, 0.291] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.421 [0.398, 0.435] · 0.0 · 0.0 | 0.472 [0.417, 0.522] · 0.0 · 0.0 | 0.599 [0.542, 0.655] · 0.3 · 0.0 | 0.734 [0.617, 0.859] · 2.7 · 0.1 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.270 [0.268, 0.271] · 0.0 · 0.0 | 0.271 [0.269, 0.275] · 0.0 · 0.0 | 0.278 [0.272, 0.280] · 0.0 · 0.0 | 0.284 [0.277, 0.290] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.269 [0.263, 0.272] · 0.0 · 0.0 | 0.269 [0.265, 0.271] · 0.0 · 0.0 | 0.277 [0.273, 0.281] · 0.0 · 0.0 | 0.302 [0.289, 0.310] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.278 [0.275, 0.280] · 0.0 · 0.0 | 0.278 [0.275, 0.281] · 0.0 · 0.0 | 0.277 [0.273, 0.280] · 0.0 · 0.0 | 0.279 [0.276, 0.280] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.275 [0.270, 0.278] · 0.0 · 0.0 | 0.276 [0.271, 0.279] · 0.0 · 0.0 | 0.279 [0.278, 0.283] · 0.0 · 0.0 | 0.290 [0.285, 0.292] · 0.0 · 0.0 |
| `pid_track_descend` | 0.346 · 0.0 · 0.0 | 0.439 · 0.0 · 0.0 | 0.585 · 0.0 · 0.0 | 0.656 · 0.0 · 0.0 |
| `pid_feedforward` | 0.224 · 0.0 · 0.0 | 0.245 · 0.0 · 0.0 | 0.265 · 0.0 · 0.0 | 0.249 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.129 · 0.0 · 0.0 | 0.148 · 0.0 · 0.0 | 0.174 · 0.0 · 0.0 | 0.161 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.129 · 0.0 · 0.0 | 0.148 · 0.0 · 0.0 | 0.174 · 0.0 · 0.0 | 0.161 · 0.0 · 0.0 |
| `gated` | 0.219 · 0.0 · 0.0 | 0.233 · 0.0 · 24.5 | 0.226 · 0.0 · 71.0 | 0.211 · 0.0 · 98.0 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.219 · 0.0 · 0.0 | 0.242 · 0.0 · 25.0 | 0.224 · 0.0 · 71.5 | 0.207 · 0.0 · 98.0 |

### `id`, sinusoid motion: outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 4 (0.4) | 7 (0.7) | 989 (98.9) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 0 (0.0) | 994 (99.4) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 1 (0.1) | 44 (4.4) | 1 (0.1) | 954 (95.4) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 3 (0.3) | 13 (1.3) | 161 (16.1) | 2 (0.2) | 821 (82.1) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 27 (2.7) | 66 (6.6) | 280 (28.0) | 14 (1.4) | 612 (61.2) | 1 (0.1) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 10 (1.0) | 985 (98.5) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 7 (0.7) | 11 (1.1) | 5 (0.5) | 977 (97.7) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 9 (0.9) | 5 (0.5) | 986 (98.6) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 1 (0.1) | 993 (99.3) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 11 (5.5) | 187 (93.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 42 (21.0) | 29 (14.5) | 129 (64.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 50 (25.0) | 30 (15.0) | 120 (60.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 10 (5.0) | 188 (94.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 26 (13.0) | 169 (84.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 18 (9.0) | 177 (88.5) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 151 (75.5) | 49 (24.5) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 58 (29.0) | 142 (71.0) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 150 (75.0) | 50 (25.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 57 (28.5) | 143 (71.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) |

### `id`: JONSWAP − sinusoid

| method | SS | JONSWAP | sinusoid | JONSWAP − sinusoid [95 % CI] |
|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS6 | **98.2** [98.0, 98.5]; seeds 98.0–98.5; N 5×200=1000 | **99.0** [98.3, 99.3]; seeds 98.0–99.5; N 5×200=1000 | -0.8 [-2.7, +1.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | **99.8** [98.8, 100.0]; seeds 98.5–100.0; N 5×200=1000 | **99.3** [99.0, 99.8]; seeds 99.0–100.0; N 5×200=1000 | +0.5 [-1.0, +1.5] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | **98.2** [96.3, 99.0]; seeds 96.0–99.0; N 5×200=1000 | **96.5** [91.5, 97.8]; seeds 89.5–98.0; N 5×200=1000 | +1.7 [-0.8, +6.3] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | **87.0** [81.7, 90.3]; seeds 80.0–91.0; N 5×200=1000 | **83.2** [74.5, 87.7]; seeds 72.5–88.5; N 5×200=1000 | +3.8 [-2.0, +11.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | **72.5** [60.2, 79.3]; seeds 55.5–81.5; N 5×200=1000 | **61.0** [50.5, 73.0]; seeds 47.0–76.0; N 5×200=1000 | +11.5 [+0.7, +19.3] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | **99.5** [99.5, 99.8]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | -0.5 [-1.5, +0.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | **96.2** [95.7, 97.8]; seeds 95.5–98.5; N 5×200=1000 | **98.7** [97.3, 99.5]; seeds 97.0–99.5; N 5×200=1000 | -2.5 [-5.3, +1.2] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | **98.0** [97.5, 98.5]; seeds 97.5–98.5; N 5×200=1000 | **97.8** [96.7, 98.5]; seeds 96.5–98.5; N 5×200=1000 | +0.2 [-2.0, +2.7] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | **99.5** [99.5, 99.5]; seeds 99.5–99.5; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | -0.5 [-1.5, +0.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | **96.8** [96.5, 98.0]; seeds 96.5–98.5; N 5×200=1000 | **98.7** [97.5, 99.7]; seeds 97.0–100.0; N 5×200=1000 | -1.8 [-4.3, +1.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | **97.3** [96.7, 97.8]; seeds 96.5–98.0; N 5×200=1000 | **99.3** [98.5, 100.0]; seeds 98.5–100.0; N 5×200=1000 | -2.0 [-4.5, +0.2] |
| `pid_track_descend` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS4 | 96.0 [92.3, 98.0] 192/200 | 93.5 [89.2, 96.2] 187/200 | +2.5 [-1.5, +6.5] |
| `pid_track_descend` | SS5 | 85.0 [79.4, 89.3] 170/200 | 64.5 [57.7, 70.8] 129/200 | +20.5 [+13.5, +28.0] |
| `pid_track_descend` | SS6 | 80.0 [73.9, 85.0] 160/200 | 60.0 [53.1, 66.5] 120/200 | +20.0 [+11.5, +28.5] |
| `pid_feedforward` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS4 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | +0.5 [+0.0, +1.5] |
| `pid_feedforward` | SS5 | 99.0 [96.4, 99.7] 198/200 | 99.0 [96.4, 99.7] 198/200 | +0.0 [-1.5, +1.5] |
| `pid_feedforward` | SS6 | 90.5 [85.6, 93.8] 181/200 | 94.0 [89.8, 96.5] 188/200 | -3.5 [-8.5, +1.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | +1.0 [+0.0, +2.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 98.0 [95.0, 99.2] 196/200 | 97.5 [94.3, 98.9] 195/200 | +0.5 [-2.0, +3.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 95.5 [91.7, 97.6] 191/200 | 95.0 [91.0, 97.3] 190/200 | +0.5 [-3.5, +4.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 85.0 [79.4, 89.3] 170/200 | 84.5 [78.8, 88.9] 169/200 | +0.5 [-5.5, +6.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | +1.0 [+0.0, +2.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 98.5 [95.7, 99.5] 197/200 | 98.5 [95.7, 99.5] 197/200 | +0.0 [-2.5, +2.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 98.0 [95.0, 99.2] 196/200 | 97.0 [93.6, 98.6] 194/200 | +1.0 [-2.0, +4.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 88.0 [82.8, 91.8] 176/200 | 88.5 [83.3, 92.2] 177/200 | -0.5 [-6.0, +5.0] |
| `gated` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `gated` | SS4 | 98.5 [95.7, 99.5] 197/200 | 75.5 [69.1, 80.9] 151/200 | +23.0 [+17.0, +29.0] |
| `gated` | SS5 | 89.5 [84.5, 93.0] 179/200 | 29.0 [23.2, 35.6] 58/200 | +60.5 [+53.5, +67.5] |
| `gated` | SS6 | 63.0 [56.1, 69.4] 126/200 | 2.0 [0.8, 5.0] 4/200 | +61.0 [+54.0, +68.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 98.5 [95.7, 99.5] 197/200 | 75.0 [68.6, 80.5] 150/200 | +23.5 [+17.5, +29.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 89.0 [83.9, 92.6] 178/200 | 28.5 [22.7, 35.1] 57/200 | +60.5 [+53.0, +67.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 64.5 [57.7, 70.8] 129/200 | 2.0 [0.8, 5.0] 4/200 | +62.5 [+56.0, +69.0] |

## 4. Perception stand-in (P7-D1 §4), `id`, aft pad

σp on relative position, σv = σp / 0.2 s on relative velocity (0 / 0.05 / 0.10 / 0.20 m/s), latency 0 / 1 / 2 control steps (configured 0 / 33.4 / 66.7 ms; 1 step = 33.3 ms model = 167 ms full scale), no hold (30 Hz). Applied to the six relative-pad entries only; the forecast methods' ship-motion feed stays ideal. The clean column is the main matrix's rows. Cells: success (IQM or rate) with the clean − noisy paired contrast in points [95 % CI] below it.
Conditions written: 11 of 11.

### `id` SS3: success % by condition (clean − noisy, points [95 % CI])

| method | σp 0 cm / 0 step (0.0 ms) | σp 1 cm / 0 step (0.0 ms) | σp 2 cm / 0 step (0.0 ms) | σp 4 cm / 0 step (0.0 ms) | σp 0 cm / 1 step (33.3 ms) | σp 1 cm / 1 step (33.3 ms) | σp 2 cm / 1 step (33.3 ms) | σp 4 cm / 1 step (33.3 ms) | σp 0 cm / 2 step (66.7 ms) | σp 1 cm / 2 step (66.7 ms) | σp 2 cm / 2 step (66.7 ms) | σp 4 cm / 2 step (66.7 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 99.3<br>+0.7 [+0.0, +1.5] | 83.5<br>+16.5 [+8.2, +24.0] | 81.0<br>+19.0 [+13.8, +29.0] | 83.8<br>+16.2 [+8.5, +22.3] | 82.5<br>+17.5 [+10.5, +24.5] | 57.8<br>+42.2 [+26.5, +51.3] | 31.2<br>+68.8 [+45.5, +85.7] | 31.5<br>+68.5 [+46.3, +79.0] | 29.0<br>+71.0 [+50.7, +82.2] | 26.0<br>+74.0 [+53.3, +79.3] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 99.8 | 94.0<br>+5.8 [+0.3, +10.0] | 66.8<br>+33.0 [+20.5, +41.0] | 23.0<br>+76.8 [+60.5, +85.2] | 40.2<br>+59.7 [+38.2, +70.2] | 37.0<br>+62.8 [+38.8, +75.2] | 32.3<br>+67.5 [+56.0, +80.7] | 12.2<br>+87.7 [+71.0, +94.3] | 14.8<br>+85.0 [+71.5, +92.5] | 13.7<br>+86.2 [+77.5, +90.8] | 16.5<br>+83.3 [+74.5, +93.8] | 6.0<br>+93.8 [+85.0, +98.5] |
| `residual_ppo` (learned, residual on pid_feedforward) | 100.0 | 99.8<br>+0.2 [+0.0, +0.8] | 80.0<br>+20.0 [+12.2, +26.2] | 4.0<br>+96.0 [+92.2, +99.0] | 80.2<br>+19.8 [+6.5, +43.5] | 73.2<br>+26.8 [+18.0, +37.7] | 31.5<br>+68.5 [+57.3, +76.0] | 0.0<br>+100.0 [+99.2, +100.0] | 0.2<br>+99.8 [+99.2, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 100.0<br>+0.0 [+0.0, +0.0] | 96.2<br>+3.8 [+1.3, +7.7] | 100.0<br>+0.0 [+0.0, +0.0] | 100.0<br>+0.0 [+0.0, +0.5] | 99.0<br>+1.0 [+0.0, +3.0] | 91.7<br>+8.3 [+5.3, +13.2] | 100.0<br>+0.0 [+0.0, +0.0] | 98.8<br>+1.2 [+0.0, +2.5] | 97.0<br>+3.0 [+1.3, +6.3] | 90.0<br>+10.0 [+5.7, +16.2] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 85.0<br>+15.0 [+9.3, +20.2] | 3.8<br>+96.2 [+91.8, +98.2] | 97.0<br>+3.0 [+0.3, +7.8] | 88.2<br>+11.8 [+4.3, +17.3] | 51.5<br>+48.5 [+33.5, +56.3] | 0.7<br>+99.3 [+98.0, +100.0] | 6.7<br>+93.3 [+79.7, +97.5] | 3.8<br>+96.2 [+86.5, +99.0] | 1.5<br>+98.5 [+94.8, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 99.7<br>+0.3 [+0.0, +1.3] | 79.5<br>+20.5 [+15.5, +30.0] | 82.2<br>+17.8 [+9.7, +28.8] | 82.8<br>+17.2 [+13.0, +30.8] | 74.5<br>+25.5 [+20.5, +37.5] | 48.0<br>+52.0 [+45.8, +65.2] | 13.3<br>+86.7 [+74.5, +94.2] | 11.7<br>+88.3 [+79.5, +94.7] | 16.8<br>+83.2 [+77.3, +93.2] | 12.3<br>+87.7 [+81.7, +93.7] |
| `pid_track_descend` | 100.0 | 98.5<br>+1.5 [+0.0, +3.5] | 81.5<br>+18.5 [+13.5, +24.0] | 20.5<br>+79.5 [+73.5, +85.0] | 99.5<br>+0.5 [+0.0, +1.5] | 97.5<br>+2.5 [+0.5, +5.0] | 83.0<br>+17.0 [+12.0, +22.5] | 23.0<br>+77.0 [+71.0, +82.5] | 100.0<br>+0.0 [+0.0, +0.0] | 97.0<br>+3.0 [+1.0, +5.5] | 81.0<br>+19.0 [+13.5, +24.5] | 19.5<br>+80.5 [+75.0, +85.5] |
| `pid_feedforward` | 100.0 | 99.0<br>+1.0 [+0.0, +2.5] | 66.0<br>+34.0 [+27.5, +40.5] | 0.5<br>+99.5 [+98.5, +100.0] | 99.5<br>+0.5 [+0.0, +1.5] | 85.5<br>+14.5 [+10.0, +19.5] | 9.0<br>+91.0 [+87.0, +94.5] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 | 96.5<br>+3.5 [+1.0, +6.5] | 70.5<br>+29.5 [+23.5, +36.0] | 4.5<br>+95.5 [+92.5, +98.0] | 97.0<br>+3.0 [+1.0, +5.5] | 78.5<br>+21.5 [+16.0, +27.5] | 38.0<br>+62.0 [+55.0, +68.5] | 0.5<br>+99.5 [+98.5, +100.0] | 81.5<br>+18.5 [+13.5, +24.0] | 15.0<br>+85.0 [+80.0, +90.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 | 97.5<br>+2.5 [+0.5, +5.0] | 93.5<br>+6.5 [+3.5, +10.0] | 37.0<br>+63.0 [+56.5, +69.5] | 98.0<br>+2.0 [+0.5, +4.0] | 95.0<br>+5.0 [+2.0, +8.0] | 79.0<br>+21.0 [+15.5, +27.0] | 6.0<br>+94.0 [+90.5, +97.0] | 93.0<br>+7.0 [+3.5, +10.5] | 67.0<br>+33.0 [+26.5, +39.5] | 4.0<br>+96.0 [+93.0, +98.5] | 0.0<br>+100.0 [+100.0, +100.0] |
| `gated` | 100.0 | 97.5<br>+2.5 [+0.5, +5.0] | 16.0<br>+84.0 [+79.0, +89.0] | 0.0<br>+100.0 [+100.0, +100.0] | 98.5<br>+1.5 [+0.0, +3.5] | 82.0<br>+18.0 [+13.0, +23.5] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 | 97.0<br>+3.0 [+1.0, +5.5] | 29.0<br>+71.0 [+64.5, +77.0] | 0.0<br>+100.0 [+100.0, +100.0] | 99.0<br>+1.0 [+0.0, +2.5] | 80.0<br>+20.0 [+14.5, +26.0] | 2.0<br>+98.0 [+96.0, +99.5] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] |

### `id` SS4: success % by condition (clean − noisy, points [95 % CI])

| method | σp 0 cm / 0 step (0.0 ms) | σp 1 cm / 0 step (0.0 ms) | σp 2 cm / 0 step (0.0 ms) | σp 4 cm / 0 step (0.0 ms) | σp 0 cm / 1 step (33.3 ms) | σp 1 cm / 1 step (33.3 ms) | σp 2 cm / 1 step (33.3 ms) | σp 4 cm / 1 step (33.3 ms) | σp 0 cm / 2 step (66.7 ms) | σp 1 cm / 2 step (66.7 ms) | σp 2 cm / 2 step (66.7 ms) | σp 4 cm / 2 step (66.7 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 98.7<br>+1.3 [+0.0, +2.7] | 81.7<br>+18.3 [+10.2, +24.0] | 81.3<br>+18.7 [+12.5, +24.2] | 87.8<br>+12.2 [+6.5, +18.2] | 82.2<br>+17.8 [+10.0, +23.8] | 56.5<br>+43.5 [+27.7, +53.2] | 30.0<br>+70.0 [+47.2, +81.5] | 31.7<br>+68.3 [+47.7, +78.5] | 29.8<br>+70.2 [+50.2, +81.0] | 23.8<br>+76.2 [+54.5, +82.5] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 98.2 | 89.5<br>+8.7 [+2.7, +13.3] | 63.2<br>+35.0 [+18.2, +43.8] | 20.3<br>+77.8 [+62.3, +84.7] | 41.3<br>+56.8 [+38.3, +67.0] | 36.2<br>+62.0 [+41.5, +72.3] | 31.0<br>+67.2 [+52.5, +77.0] | 12.8<br>+85.3 [+72.3, +90.5] | 16.5<br>+81.7 [+74.2, +90.2] | 17.7<br>+80.5 [+73.5, +89.2] | 12.0<br>+86.2 [+78.3, +95.0] | 6.5<br>+91.7 [+84.8, +96.5] |
| `residual_ppo` (learned, residual on pid_feedforward) | 100.0 | 99.8<br>+0.2 [+0.0, +0.8] | 79.7<br>+20.3 [+13.8, +27.8] | 3.3<br>+96.7 [+94.0, +98.3] | 79.7<br>+20.3 [+7.5, +44.7] | 74.7<br>+25.3 [+18.2, +31.3] | 34.8<br>+65.2 [+54.3, +74.7] | 0.2<br>+99.8 [+99.0, +100.0] | 0.0<br>+100.0 [+99.5, +100.0] | 0.0<br>+100.0 [+99.5, +100.0] | 0.0<br>+100.0 [+99.5, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 100.0<br>+0.0 [+0.0, +0.0] | 96.8<br>+3.2 [+0.7, +6.3] | 100.0<br>+0.0 [+0.0, +0.0] | 100.0<br>+0.0 [+0.0, +0.0] | 99.0<br>+1.0 [+0.0, +2.5] | 93.7<br>+6.3 [+3.5, +11.3] | 99.8<br>+0.2 [+0.0, +1.3] | 99.0<br>+1.0 [+0.0, +3.0] | 96.7<br>+3.3 [+0.7, +6.8] | 90.2<br>+9.8 [+5.8, +14.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 84.2<br>+15.8 [+9.7, +20.8] | 5.2<br>+94.8 [+92.0, +97.0] | 98.7<br>+1.3 [+0.0, +6.2] | 92.2<br>+7.8 [+4.0, +14.2] | 48.7<br>+51.3 [+32.7, +57.8] | 0.5<br>+99.5 [+97.7, +100.0] | 7.2<br>+92.8 [+82.0, +98.2] | 3.3<br>+96.7 [+83.3, +98.5] | 1.3<br>+98.7 [+95.0, +100.0] | 0.0<br>+100.0 [+99.5, +100.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 99.0<br>+1.0 [+0.0, +2.7] | 80.3<br>+19.7 [+15.5, +25.8] | 82.8<br>+17.2 [+8.5, +32.0] | 83.3<br>+16.7 [+12.5, +27.8] | 74.7<br>+25.3 [+16.5, +41.8] | 49.2<br>+50.8 [+44.3, +64.3] | 15.7<br>+84.3 [+76.2, +93.8] | 15.8<br>+84.2 [+76.7, +92.3] | 15.5<br>+84.5 [+79.2, +92.3] | 15.7<br>+84.3 [+76.3, +93.5] |
| `pid_track_descend` | 96.0 | 93.5<br>+2.5 [-1.0, +6.0] | 78.5<br>+17.5 [+11.5, +23.5] | 22.5<br>+73.5 [+67.0, +79.5] | 95.5<br>+0.5 [-2.0, +3.0] | 95.5<br>+0.5 [-2.5, +3.5] | 73.5<br>+22.5 [+16.5, +28.5] | 22.0<br>+74.0 [+68.0, +80.0] | 96.0<br>+0.0 [-1.5, +1.5] | 93.0<br>+3.0 [-0.0, +6.5] | 80.0<br>+16.0 [+10.5, +22.0] | 21.5<br>+74.5 [+68.0, +80.5] |
| `pid_feedforward` | 100.0 | 99.5<br>+0.5 [+0.0, +1.5] | 57.0<br>+43.0 [+36.5, +50.0] | 0.5<br>+99.5 [+98.5, +100.0] | 99.5<br>+0.5 [+0.0, +1.5] | 82.0<br>+18.0 [+13.0, +23.5] | 13.0<br>+87.0 [+82.0, +91.5] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] | 0.0<br>+100.0 [+100.0, +100.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 98.0 | 95.0<br>+3.0 [+0.0, +6.0] | 65.0<br>+33.0 [+26.0, +40.0] | 7.0<br>+91.0 [+87.0, +95.0] | 95.5<br>+2.5 [-1.0, +6.0] | 77.5<br>+20.5 [+14.5, +26.5] | 39.0<br>+59.0 [+52.0, +66.0] | 0.0<br>+98.0 [+96.0, +99.5] | 76.5<br>+21.5 [+15.5, +28.0] | 17.0<br>+81.0 [+75.5, +86.0] | 0.5<br>+97.5 [+95.0, +99.5] | 0.0<br>+98.0 [+96.0, +99.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 98.5 | 97.0<br>+1.5 [-1.0, +4.0] | 90.5<br>+8.0 [+3.5, +12.5] | 42.5<br>+56.0 [+49.0, +63.0] | 98.0<br>+0.5 [-2.0, +3.0] | 91.0<br>+7.5 [+3.0, +12.0] | 78.5<br>+20.0 [+14.0, +26.0] | 2.5<br>+96.0 [+93.0, +98.5] | 94.0<br>+4.5 [+1.0, +8.5] | 67.5<br>+31.0 [+24.0, +38.0] | 2.0<br>+96.5 [+94.0, +99.0] | 0.0<br>+98.5 [+96.5, +100.0] |
| `gated` | 98.5 | 94.5<br>+4.0 [+1.5, +7.0] | 7.5<br>+91.0 [+87.0, +94.5] | 0.0<br>+98.5 [+96.5, +100.0] | 97.5<br>+1.0 [+0.0, +2.5] | 82.5<br>+16.0 [+11.0, +21.5] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] |
| `oracle_gated` — commit-timing oracle (privileged) | 98.5 | 96.5<br>+2.0 [+0.5, +4.0] | 25.5<br>+73.0 [+67.0, +79.0] | 0.0<br>+98.5 [+96.5, +100.0] | 97.0<br>+1.5 [+0.0, +3.5] | 84.0<br>+14.5 [+10.0, +19.5] | 1.5<br>+97.0 [+94.5, +99.0] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] | 0.0<br>+98.5 [+96.5, +100.0] |

### `id` SS5: success % by condition (clean − noisy, points [95 % CI])

| method | σp 0 cm / 0 step (0.0 ms) | σp 1 cm / 0 step (0.0 ms) | σp 2 cm / 0 step (0.0 ms) | σp 4 cm / 0 step (0.0 ms) | σp 0 cm / 1 step (33.3 ms) | σp 1 cm / 1 step (33.3 ms) | σp 2 cm / 1 step (33.3 ms) | σp 4 cm / 1 step (33.3 ms) | σp 0 cm / 2 step (66.7 ms) | σp 1 cm / 2 step (66.7 ms) | σp 2 cm / 2 step (66.7 ms) | σp 4 cm / 2 step (66.7 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | 100.0 | 100.0<br>+0.0 [+0.0, +0.5] | 98.3<br>+1.7 [+0.0, +3.7] | 78.2<br>+21.8 [+13.7, +27.5] | 85.5<br>+14.5 [+7.8, +22.3] | 86.5<br>+13.5 [+6.3, +23.0] | 82.7<br>+17.3 [+10.0, +24.3] | 56.2<br>+43.8 [+30.3, +53.3] | 35.3<br>+64.7 [+45.0, +75.5] | 33.5<br>+66.5 [+48.8, +75.5] | 33.7<br>+66.3 [+51.0, +74.2] | 27.7<br>+72.3 [+57.0, +78.2] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 87.0 | 78.8<br>+8.2 [+2.7, +12.5] | 54.8<br>+32.2 [+18.0, +38.5] | 16.2<br>+70.8 [+60.7, +79.5] | 34.7<br>+52.3 [+37.5, +59.2] | 33.0<br>+54.0 [+44.0, +62.8] | 25.0<br>+62.0 [+52.3, +70.0] | 9.7<br>+77.3 [+67.8, +83.2] | 11.8<br>+75.2 [+67.2, +84.3] | 11.0<br>+76.0 [+67.7, +84.2] | 11.2<br>+75.8 [+69.3, +84.5] | 4.7<br>+82.3 [+75.7, +87.7] |
| `residual_ppo` (learned, residual on pid_feedforward) | 99.5 | 99.2<br>+0.3 [+0.0, +1.5] | 80.3<br>+19.2 [+14.0, +24.5] | 3.3<br>+96.2 [+92.8, +98.5] | 79.5<br>+20.0 [+11.8, +36.7] | 72.7<br>+26.8 [+18.3, +34.0] | 32.5<br>+67.0 [+56.8, +76.3] | 0.2<br>+99.3 [+98.2, +100.0] | 0.2<br>+99.3 [+98.2, +100.0] | 0.2<br>+99.3 [+98.2, +100.0] | 0.0<br>+99.5 [+98.3, +100.0] | 0.0<br>+99.5 [+98.5, +100.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 100.0 | 100.0<br>+0.0 [+0.0, +0.5] | 100.0<br>+0.0 [+0.0, +0.5] | 95.8<br>+4.2 [+2.3, +6.3] | 99.8<br>+0.2 [+0.0, +0.8] | 99.8<br>+0.2 [+0.0, +0.8] | 98.5<br>+1.5 [+0.3, +2.8] | 90.5<br>+9.5 [+6.3, +13.0] | 99.7<br>+0.3 [+0.0, +1.3] | 99.2<br>+0.8 [+0.0, +2.2] | 96.8<br>+3.2 [+1.3, +5.5] | 88.2<br>+11.8 [+8.3, +16.3] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 99.5 | 98.8<br>+0.7 [+0.0, +1.7] | 84.3<br>+15.2 [+10.2, +20.3] | 3.8<br>+95.7 [+90.5, +98.0] | 96.7<br>+2.8 [+0.0, +5.5] | 90.3<br>+9.2 [+4.5, +16.7] | 47.2<br>+52.3 [+38.5, +57.0] | 0.7<br>+98.8 [+97.3, +100.0] | 9.2<br>+90.3 [+80.8, +95.7] | 6.7<br>+92.8 [+84.2, +97.3] | 1.7<br>+97.8 [+95.2, +99.7] | 0.0<br>+99.5 [+98.5, +100.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 100.0 | 100.0<br>+0.0 [+0.0, +0.0] | 98.5<br>+1.5 [+0.0, +2.8] | 78.5<br>+21.5 [+16.0, +32.7] | 84.2<br>+15.8 [+7.2, +28.3] | 82.8<br>+17.2 [+10.5, +31.8] | 76.8<br>+23.2 [+18.2, +31.8] | 50.0<br>+50.0 [+43.8, +64.8] | 16.8<br>+83.2 [+75.2, +92.2] | 17.5<br>+82.5 [+74.7, +91.2] | 21.8<br>+78.2 [+72.3, +90.0] | 15.5<br>+84.5 [+79.0, +91.8] |
| `pid_track_descend` | 85.0 | 81.5<br>+3.5 [-1.0, +8.0] | 68.5<br>+16.5 [+10.5, +22.5] | 18.5<br>+66.5 [+60.0, +73.5] | 84.0<br>+1.0 [-3.0, +5.0] | 81.5<br>+3.5 [-1.0, +8.0] | 65.5<br>+19.5 [+14.0, +25.5] | 17.5<br>+67.5 [+60.5, +74.0] | 82.0<br>+3.0 [-1.0, +7.0] | 81.0<br>+4.0 [-0.5, +9.0] | 72.5<br>+12.5 [+6.0, +19.0] | 19.0<br>+66.0 [+59.0, +73.0] |
| `pid_feedforward` | 99.0 | 96.5<br>+2.5 [+0.0, +5.0] | 63.5<br>+35.5 [+28.5, +42.5] | 0.0<br>+99.0 [+97.5, +100.0] | 96.5<br>+2.5 [+0.5, +5.0] | 77.5<br>+21.5 [+16.0, +27.5] | 10.0<br>+89.0 [+84.5, +93.0] | 0.0<br>+99.0 [+97.5, +100.0] | 0.0<br>+99.0 [+97.5, +100.0] | 0.0<br>+99.0 [+97.5, +100.0] | 0.0<br>+99.0 [+97.5, +100.0] | 0.0<br>+99.0 [+97.5, +100.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 95.5 | 88.5<br>+7.0 [+2.0, +12.0] | 60.5<br>+35.0 [+27.5, +42.5] | 7.5<br>+88.0 [+83.0, +92.5] | 94.0<br>+1.5 [-2.5, +5.5] | 70.5<br>+25.0 [+18.0, +32.0] | 36.0<br>+59.5 [+52.5, +66.5] | 0.5<br>+95.0 [+91.5, +98.0] | 71.0<br>+24.5 [+17.5, +31.5] | 23.0<br>+72.5 [+65.5, +79.0] | 0.0<br>+95.5 [+92.5, +98.0] | 0.0<br>+95.5 [+92.5, +98.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 98.0 | 95.5<br>+2.5 [-1.0, +6.0] | 84.0<br>+14.0 [+8.5, +19.5] | 43.5<br>+54.5 [+47.5, +61.5] | 96.5<br>+1.5 [-1.5, +5.0] | 91.0<br>+7.0 [+3.0, +11.5] | 77.0<br>+21.0 [+15.0, +27.0] | 3.5<br>+94.5 [+91.0, +97.5] | 89.0<br>+9.0 [+4.5, +14.0] | 67.5<br>+30.5 [+23.5, +37.5] | 7.0<br>+91.0 [+86.5, +95.0] | 0.0<br>+98.0 [+96.0, +99.5] |
| `gated` | 89.5 | 86.0<br>+3.5 [+1.0, +6.5] | 5.5<br>+84.0 [+79.0, +89.0] | 0.0<br>+89.5 [+85.5, +93.5] | 88.5<br>+1.0 [+0.0, +2.5] | 60.0<br>+29.5 [+23.0, +36.0] | 0.0<br>+89.5 [+85.5, +93.5] | 0.0<br>+89.5 [+85.5, +93.5] | 0.0<br>+89.5 [+85.5, +93.5] | 0.0<br>+89.5 [+85.5, +93.5] | 0.0<br>+89.5 [+85.5, +93.5] | 0.0<br>+89.5 [+85.5, +93.5] |
| `oracle_gated` — commit-timing oracle (privileged) | 89.0 | 86.5<br>+2.5 [+0.0, +5.0] | 14.5<br>+74.5 [+68.5, +80.0] | 0.0<br>+89.0 [+84.5, +93.0] | 88.5<br>+0.5 [-1.5, +2.5] | 63.0<br>+26.0 [+20.0, +32.5] | 0.0<br>+89.0 [+84.5, +93.0] | 0.0<br>+89.0 [+84.5, +93.0] | 0.0<br>+89.0 [+84.5, +93.0] | 0.0<br>+89.0 [+84.5, +93.0] | 0.0<br>+89.0 [+84.5, +93.0] | 0.0<br>+89.0 [+84.5, +93.0] |

### `id` SS6: success % by condition (clean − noisy, points [95 % CI])

| method | σp 0 cm / 0 step (0.0 ms) | σp 1 cm / 0 step (0.0 ms) | σp 2 cm / 0 step (0.0 ms) | σp 4 cm / 0 step (0.0 ms) | σp 0 cm / 1 step (33.3 ms) | σp 1 cm / 1 step (33.3 ms) | σp 2 cm / 1 step (33.3 ms) | σp 4 cm / 1 step (33.3 ms) | σp 0 cm / 2 step (66.7 ms) | σp 1 cm / 2 step (66.7 ms) | σp 2 cm / 2 step (66.7 ms) | σp 4 cm / 2 step (66.7 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | 98.2 | 98.7<br>-0.5 [-1.7, +0.8] | 96.0<br>+2.2 [+0.0, +4.8] | 78.7<br>+19.5 [+12.5, +25.2] | 90.0<br>+8.2 [+5.0, +12.8] | 86.5<br>+11.7 [+8.0, +17.7] | 82.7<br>+15.5 [+11.2, +20.0] | 53.5<br>+44.7 [+33.5, +53.0] | 36.8<br>+61.3 [+47.7, +71.3] | 33.8<br>+64.3 [+49.0, +71.8] | 32.8<br>+65.3 [+47.2, +74.8] | 29.0<br>+69.2 [+58.7, +77.2] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 72.5 | 61.7<br>+10.8 [+3.8, +15.8] | 41.0<br>+31.5 [+20.0, +36.7] | 12.0<br>+60.5 [+47.7, +66.0] | 27.3<br>+45.2 [+36.0, +51.7] | 22.5<br>+50.0 [+39.8, +56.7] | 15.8<br>+56.7 [+46.0, +62.3] | 5.8<br>+66.7 [+55.2, +73.0] | 6.3<br>+66.2 [+54.7, +73.0] | 7.7<br>+64.8 [+55.8, +72.8] | 4.8<br>+67.7 [+55.7, +75.3] | 3.0<br>+69.5 [+57.5, +75.8] |
| `residual_ppo` (learned, residual on pid_feedforward) | 96.2 | 95.3<br>+0.8 [-1.7, +3.5] | 76.5<br>+19.7 [+15.2, +26.3] | 3.2<br>+93.0 [+90.0, +95.8] | 83.5<br>+12.7 [+7.2, +27.3] | 73.7<br>+22.5 [+15.5, +31.5] | 31.8<br>+64.3 [+52.5, +72.0] | 0.3<br>+95.8 [+93.5, +98.3] | 0.5<br>+95.7 [+93.2, +98.0] | 0.2<br>+96.0 [+93.5, +98.5] | 0.0<br>+96.2 [+93.8, +98.7] | 0.0<br>+96.2 [+93.8, +98.7] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 98.0 | 98.2<br>-0.2 [-2.0, +1.0] | 96.3<br>+1.7 [+0.0, +3.2] | 90.8<br>+7.2 [+3.8, +11.0] | 98.0<br>+0.0 [-1.2, +2.2] | 97.2<br>+0.8 [-0.5, +2.5] | 96.2<br>+1.8 [+0.3, +4.5] | 86.8<br>+11.2 [+7.5, +15.0] | 96.7<br>+1.3 [-0.2, +3.7] | 95.8<br>+2.2 [+0.5, +4.3] | 93.0<br>+5.0 [+2.0, +8.3] | 82.0<br>+16.0 [+11.8, +20.2] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 96.8 | 95.0<br>+1.8 [-0.7, +5.5] | 79.5<br>+17.3 [+11.2, +24.8] | 4.8<br>+92.0 [+86.5, +95.3] | 93.7<br>+3.2 [+0.5, +5.7] | 87.7<br>+9.2 [+3.3, +14.5] | 47.0<br>+49.8 [+35.5, +56.0] | 1.2<br>+95.7 [+93.5, +98.2] | 8.5<br>+88.3 [+81.2, +93.5] | 4.5<br>+92.3 [+82.5, +95.2] | 1.7<br>+95.2 [+92.7, +97.7] | 0.0<br>+96.8 [+94.8, +99.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 97.3 | 97.3<br>+0.0 [-2.0, +1.3] | 95.3<br>+2.0 [-0.2, +5.5] | 73.5<br>+23.8 [+16.2, +32.0] | 81.0<br>+16.3 [+10.3, +26.7] | 80.7<br>+16.7 [+11.3, +28.5] | 71.8<br>+25.5 [+17.7, +34.2] | 45.8<br>+51.5 [+45.8, +65.0] | 19.5<br>+77.8 [+73.0, +89.0] | 19.0<br>+78.3 [+71.3, +88.8] | 18.2<br>+79.2 [+70.5, +88.8] | 18.0<br>+79.3 [+74.5, +85.0] |
| `pid_track_descend` | 80.0 | 77.5<br>+2.5 [-3.0, +8.0] | 68.5<br>+11.5 [+4.5, +18.5] | 23.0<br>+57.0 [+49.0, +65.0] | 77.0<br>+3.0 [-3.0, +9.0] | 77.0<br>+3.0 [-3.0, +9.0] | 66.0<br>+14.0 [+6.0, +22.0] | 24.5<br>+55.5 [+47.0, +63.5] | 82.5<br>-2.5 [-9.0, +4.0] | 80.5<br>-0.5 [-6.5, +5.5] | 65.5<br>+14.5 [+7.0, +22.0] | 21.0<br>+59.0 [+51.0, +67.0] |
| `pid_feedforward` | 90.5 | 88.0<br>+2.5 [-2.0, +7.0] | 56.5<br>+34.0 [+26.5, +41.5] | 0.0<br>+90.5 [+86.5, +94.5] | 89.0<br>+1.5 [-3.0, +6.0] | 70.5<br>+20.0 [+13.5, +27.0] | 8.0<br>+82.5 [+77.0, +87.5] | 0.0<br>+90.5 [+86.5, +94.5] | 0.0<br>+90.5 [+86.5, +94.5] | 0.0<br>+90.5 [+86.5, +94.5] | 0.0<br>+90.5 [+86.5, +94.5] | 0.0<br>+90.5 [+86.5, +94.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 85.0 | 76.0<br>+9.0 [+2.5, +15.5] | 58.0<br>+27.0 [+19.0, +35.0] | 7.0<br>+78.0 [+72.0, +84.0] | 78.5<br>+6.5 [+0.5, +12.5] | 72.5<br>+12.5 [+6.0, +19.5] | 27.0<br>+58.0 [+50.5, +65.5] | 0.0<br>+85.0 [+80.0, +89.5] | 59.5<br>+25.5 [+18.0, +33.0] | 16.5<br>+68.5 [+61.5, +75.0] | 0.0<br>+85.0 [+80.0, +89.5] | 0.0<br>+85.0 [+80.0, +89.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 88.0 | 85.0<br>+3.0 [-2.5, +8.5] | 81.0<br>+7.0 [+0.5, +13.5] | 35.5<br>+52.5 [+45.0, +59.5] | 83.5<br>+4.5 [-1.0, +10.0] | 86.5<br>+1.5 [-4.0, +7.5] | 70.0<br>+18.0 [+11.0, +25.0] | 6.0<br>+82.0 [+76.5, +87.5] | 77.5<br>+10.5 [+4.5, +17.0] | 51.0<br>+37.0 [+29.5, +44.5] | 2.0<br>+86.0 [+81.0, +90.5] | 0.0<br>+88.0 [+83.5, +92.0] |
| `gated` | 63.0 | 55.0<br>+8.0 [+4.5, +12.0] | 3.0<br>+60.0 [+53.0, +66.5] | 0.0<br>+63.0 [+56.5, +69.5] | 58.5<br>+4.5 [+1.5, +8.0] | 38.0<br>+25.0 [+19.0, +31.5] | 0.0<br>+63.0 [+56.5, +69.5] | 0.0<br>+63.0 [+56.5, +69.5] | 0.0<br>+63.0 [+56.5, +69.5] | 0.0<br>+63.0 [+56.5, +69.5] | 0.0<br>+63.0 [+56.5, +69.5] | 0.0<br>+63.0 [+56.5, +69.5] |
| `oracle_gated` — commit-timing oracle (privileged) | 64.5 | 61.0<br>+3.5 [+1.0, +6.5] | 6.5<br>+58.0 [+51.5, +65.0] | 0.0<br>+64.5 [+58.0, +71.0] | 63.0<br>+1.5 [-0.5, +4.0] | 44.0<br>+20.5 [+15.0, +26.5] | 0.5<br>+64.0 [+57.5, +70.5] | 0.0<br>+64.5 [+58.0, +71.0] | 0.0<br>+64.5 [+58.0, +71.0] | 0.0<br>+64.5 [+58.0, +71.0] | 0.0<br>+64.5 [+58.0, +71.0] | 0.0<br>+64.5 [+58.0, +71.0] |

## 5. lambda sensitivity (P7-D1 §3), aft pad

`ppo` (the best learned method at `id` SS6) and `pid_feedforward` (the best classical controller), plus the always-printed `pid_track_descend` and `oracle_gated`. The project lambda stays 1/25. At lambda ≠ 1/25 the realization, episode seed, initial position and pad are the listed ones, but **t0 is re-drawn** by the environment inside that lambda's window, so the episodes are not identical; the contrasts resample episodes independently per lambda (points [95 % CI]).

### `id`: success at lambda 1/15, 1/25, 1/40

| method | SS | 1/15 | 1/25 | 1/40 | 1/25 − 1/15 | 1/25 − 1/40 |
|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS5 | **99.8** [99.5, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.2 [+0.0, +0.8] | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS6 | **97.2** [96.5, 98.3]; seeds 96.5–98.5; N 5×200=1000 | **98.2** [98.0, 98.5]; seeds 98.0–98.5; N 5×200=1000 | **98.0** [96.8, 98.5]; seeds 96.5–98.5; N 5×200=1000 | +1.0 [-1.5, +3.5] | +0.2 [-2.0, +2.8] |
| `pid_track_descend` | SS3 | 99.0 [96.4, 99.7] 198/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +1.0 [+0.0, +2.5] | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS4 | 91.5 [86.8, 94.6] 183/200 | 96.0 [92.3, 98.0] 192/200 | 99.0 [96.4, 99.7] 198/200 | +4.5 [+0.0, +9.5] | -3.0 [-6.0, +0.0] |
| `pid_track_descend` | SS5 | 77.0 [70.7, 82.3] 154/200 | 85.0 [79.4, 89.3] 170/200 | 87.5 [82.2, 91.4] 175/200 | +8.0 [+0.5, +16.0] | -2.5 [-9.5, +4.5] |
| `pid_track_descend` | SS6 | 73.0 [66.5, 78.7] 146/200 | 80.0 [73.9, 85.0] 160/200 | 86.5 [81.1, 90.6] 173/200 | +7.0 [-1.5, +15.0] | -6.5 [-13.5, +0.5] |
| `pid_feedforward` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS5 | 98.5 [95.7, 99.5] 197/200 | 99.0 [96.4, 99.7] 198/200 | 99.0 [96.4, 99.7] 198/200 | +0.5 [-1.5, +2.5] | +0.0 [-2.0, +2.0] |
| `pid_feedforward` | SS6 | 89.0 [83.9, 92.6] 178/200 | 90.5 [85.6, 93.8] 181/200 | 93.0 [88.6, 95.8] 186/200 | +1.5 [-4.5, +7.5] | -2.5 [-8.0, +3.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 96.5 [93.0, 98.3] 193/200 | 98.5 [95.7, 99.5] 197/200 | 99.5 [97.2, 99.9] 199/200 | +2.0 [-1.0, +5.0] | -1.0 [-3.0, +1.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 83.5 [77.7, 88.0] 167/200 | 89.0 [83.9, 92.6] 178/200 | 92.0 [87.4, 95.0] 184/200 | +5.5 [-1.0, +12.0] | -3.0 [-8.5, +2.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 56.5 [49.6, 63.2] 113/200 | 64.5 [57.7, 70.8] 129/200 | 73.0 [66.5, 78.7] 146/200 | +8.0 [-1.5, +17.5] | -8.5 [-17.5, +0.5] |

#### `id`, lambda 1/15: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 0 (0.0) | 998 (99.8) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 22 (2.2) | 5 (0.5) | 973 (97.3) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 1 (0.5) | 198 (99.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 9 (4.5) | 183 (91.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 28 (14.0) | 18 (9.0) | 154 (77.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 31 (15.5) | 23 (11.5) | 146 (73.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 2 (1.0) | 197 (98.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 14 (7.0) | 178 (89.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 193 (96.5) | 7 (3.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 167 (83.5) | 33 (16.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 113 (56.5) | 87 (43.5) |

#### `id`, lambda 1/40: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 17 (1.7) | 5 (0.5) | 978 (97.8) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 18 (9.0) | 175 (87.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 21 (10.5) | 173 (86.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 7 (3.5) | 186 (93.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 184 (92.0) | 16 (8.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 146 (73.0) | 54 (27.0) |

### `unseen_seastate`: success at lambda 1/15, 1/25, 1/40

| method | SS | 1/15 | 1/25 | 1/40 | 1/25 − 1/15 | 1/25 − 1/40 |
|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | **97.0** [96.7, 98.3]; seeds 96.5–99.0; N 5×200=1000 | **97.3** [96.0, 98.2]; seeds 95.5–98.5; N 5×200=1000 | **96.8** [96.5, 97.3]; seeds 96.5–97.5; N 5×200=1000 | +0.3 [-3.5, +3.0] | +0.5 [-2.5, +3.2] |
| `pid_track_descend` | SS6 | 74.0 [67.5, 79.6] 148/200 | 81.0 [75.0, 85.8] 162/200 | 83.5 [77.7, 88.0] 167/200 | +7.0 [-1.0, +15.0] | -2.5 [-10.0, +5.0] |
| `pid_feedforward` | SS6 | 90.5 [85.6, 93.8] 181/200 | 90.0 [85.1, 93.4] 180/200 | 93.0 [88.6, 95.8] 186/200 | -0.5 [-6.5, +5.5] | -3.0 [-8.5, +2.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 60.5 [53.6, 67.0] 121/200 | 65.0 [58.2, 71.3] 130/200 | 69.5 [62.8, 75.5] 139/200 | +4.5 [-5.0, +14.0] | -4.5 [-13.5, +4.5] |

#### `unseen_seastate`, lambda 1/15: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 6 (0.6) | 973 (97.3) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 30 (15.0) | 22 (11.0) | 148 (74.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 10 (5.0) | 181 (90.5) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 121 (60.5) | 79 (39.5) |

#### `unseen_seastate`, lambda 1/40: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 8 (0.8) | 969 (96.9) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 24 (12.0) | 167 (83.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 8 (4.0) | 186 (93.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 139 (69.5) | 61 (30.5) |

### `unseen_heading`: success at lambda 1/15, 1/25, 1/40

| method | SS | 1/15 | 1/25 | 1/40 | 1/25 − 1/15 | 1/25 − 1/40 |
|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.5] |
| `ppo` (learned, PPO) | SS6 | **86.7** [85.5, 87.7]; seeds 85.0–88.0; N 5×200=1000 | **90.7** [86.3, 93.3]; seeds 85.5–93.5; N 5×200=1000 | **92.8** [91.7, 95.2]; seeds 91.5–96.0; N 5×200=1000 | +4.0 [-2.3, +9.7] | -2.2 [-8.0, +2.0] |
| `pid_track_descend` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS5 | 95.5 [91.7, 97.6] 191/200 | 97.5 [94.3, 98.9] 195/200 | 99.0 [96.4, 99.7] 198/200 | +2.0 [-1.5, +5.5] | -1.5 [-4.0, +1.0] |
| `pid_track_descend` | SS6 | 68.5 [61.8, 74.5] 137/200 | 72.5 [65.9, 78.2] 145/200 | 74.0 [67.5, 79.6] 148/200 | +4.0 [-5.0, +13.0] | -1.5 [-10.0, +7.0] |
| `pid_feedforward` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS5 | 99.5 [97.2, 99.9] 199/200 | 99.5 [97.2, 99.9] 199/200 | 99.5 [97.2, 99.9] 199/200 | +0.0 [-1.5, +1.5] | +0.0 [-1.5, +1.5] |
| `pid_feedforward` | SS6 | 72.0 [65.4, 77.8] 144/200 | 77.0 [70.7, 82.3] 154/200 | 80.5 [74.5, 85.4] 161/200 | +5.0 [-3.5, +13.5] | -3.5 [-12.0, +4.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 95.5 [91.7, 97.6] 191/200 | 99.5 [97.2, 99.9] 199/200 | 99.5 [97.2, 99.9] 199/200 | +4.0 [+1.0, +7.0] | +0.0 [-1.5, +1.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 38.0 [31.6, 44.9] 76/200 | 46.5 [39.7, 53.4] 93/200 | 46.0 [39.2, 52.9] 92/200 | +8.5 [-1.0, +18.0] | +0.5 [-9.5, +10.0] |

#### `unseen_heading`, lambda 1/15: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 1 (0.1) | 5 (0.5) | 108 (10.8) | 20 (2.0) | 866 (86.6) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 5 (2.5) | 191 (95.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 32 (16.0) | 31 (15.5) | 137 (68.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 26 (13.0) | 30 (15.0) | 144 (72.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 191 (95.5) | 9 (4.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 76 (38.0) | 124 (62.0) |

#### `unseen_heading`, lambda 1/40: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 47 (4.7) | 20 (2.0) | 932 (93.2) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 20 (10.0) | 32 (16.0) | 148 (74.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 15 (7.5) | 24 (12.0) | 161 (80.5) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 92 (46.0) | 108 (54.0) |

### `unseen_vessel`: success at lambda 1/15, 1/25, 1/40

| method | SS | 1/15 | 1/25 | 1/40 | 1/25 − 1/15 | 1/25 − 1/40 |
|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS4 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `ppo` (learned, PPO) | SS5 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | +0.0 [-0.5, +0.0] | +0.0 [-0.5, +0.0] |
| `ppo` (learned, PPO) | SS6 | **98.8** [98.5, 99.0]; seeds 98.5–99.0; N 5×200=1000 | **99.3** [98.7, 99.5]; seeds 98.5–99.5; N 5×200=1000 | **99.8** [99.5, 100.0]; seeds 99.5–100.0; N 5×200=1000 | +0.5 [-1.3, +2.3] | -0.5 [-2.0, +0.3] |
| `pid_track_descend` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_track_descend` | SS4 | 93.0 [88.6, 95.8] 186/200 | 97.5 [94.3, 98.9] 195/200 | 99.0 [96.4, 99.7] 198/200 | +4.5 [+0.5, +8.5] | -1.5 [-4.0, +1.0] |
| `pid_track_descend` | SS5 | 81.5 [75.5, 86.3] 163/200 | 90.0 [85.1, 93.4] 180/200 | 92.5 [88.0, 95.4] 185/200 | +8.5 [+1.5, +15.0] | -2.5 [-8.0, +3.0] |
| `pid_track_descend` | SS6 | 74.5 [68.0, 80.0] 149/200 | 81.5 [75.5, 86.3] 163/200 | 85.0 [79.4, 89.3] 170/200 | +7.0 [-1.0, +15.0] | -3.5 [-11.0, +3.5] |
| `pid_feedforward` | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `pid_feedforward` | SS5 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 100.0 [98.1, 100.0] 200/200 | -0.5 [-1.5, +0.0] | -0.5 [-1.5, +0.0] |
| `pid_feedforward` | SS6 | 96.5 [93.0, 98.3] 193/200 | 98.5 [95.7, 99.5] 197/200 | 98.5 [95.7, 99.5] 197/200 | +2.0 [-1.0, +5.0] | +0.0 [-2.5, +2.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | +0.0 [+0.0, +0.0] | +0.5 [+0.0, +1.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 92.5 [88.0, 95.4] 185/200 | 95.0 [91.0, 97.3] 190/200 | 97.5 [94.3, 98.9] 195/200 | +2.5 [-2.0, +7.0] | -2.5 [-6.5, +1.0] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 81.5 [75.5, 86.3] 163/200 | 89.0 [83.9, 92.6] 178/200 | 90.0 [85.1, 93.4] 180/200 | +7.5 [+0.5, +14.5] | -1.0 [-7.0, +5.0] |

#### `unseen_vessel`, lambda 1/15: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 4 (0.4) | 0 (0.0) | 7 (0.7) | 1 (0.1) | 988 (98.8) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 11 (5.5) | 186 (93.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 23 (11.5) | 14 (7.0) | 163 (81.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 36 (18.0) | 15 (7.5) | 149 (74.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 3 (1.5) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 193 (96.5) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 185 (92.5) | 15 (7.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 163 (81.5) | 37 (18.5) |

#### `unseen_vessel`, lambda 1/40: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 1 (0.1) | 998 (99.8) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 10 (5.0) | 185 (92.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 22 (11.0) | 170 (85.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 1 (0.5) | 197 (98.5) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 195 (97.5) | 5 (2.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 180 (90.0) | 20 (10.0) |

### P1-D1 feasibility rule at each lambda (context only; P7-D1 §3)

Deck-point v_z p99, model scale, worst head-seas SS6 frigate cell, aft pad, against 0.25 × the CF2X URDF speed. 1/15 is outside the declared ladder; it is reported, not adopted.

| lambda | cell | v_z p99 (m/s) | threshold (m/s) | verdict |
|---|---|---|---|---|
| 1/15 | frigate SS6 180 deg 12 kn aft | 0.7450 | 2.0833 | PASS |
| 1/25 | frigate SS6 180 deg 12 kn aft | 0.5772 | 2.0833 | PASS |
| 1/40 | frigate SS6 180 deg 12 kn aft | 0.4562 | 2.0833 | PASS |

## 6. MSS transfer (optional; P7-D1 §6; descriptive)

Episodes on another simulator's strip-theory trajectories (MSS, ITTC S-175, SS5), **not measurements of a real ship**. `mss_transfer` uses MSS's spectrum and RAOs; `mss_transfer_corpus` dmf's own wave field through MSS's transfer function (the attribution control). Beside them, the matrix's `unseen_vessel` SS5 restricted to the 180° and 135° headings (different realizations; descriptive only).

### `mss_transfer`, aft pad: success

| method | SS5/mss:mss |
|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **97.8** [96.5, 99.3]; seeds 96.0–100.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `pid_track_descend` | 91.5 [86.8, 94.6] 183/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 99.5 [97.2, 99.9] 199/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 99.5 [97.2, 99.9] 199/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 |

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5/mss:mss | 1000 | 0 (0.0) | 1 (0.1) | 20 (2.0) | 0 (0.0) | 979 (97.9) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 9 (4.5) | 183 (91.5) | 0 (0.0) |
| `pid_feedforward` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `gated` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |

### `mss_transfer_corpus`, aft pad: success

| method | SS5/mss:corpus |
|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **97.8** [94.7, 99.7]; seeds 93.5–100.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `pid_track_descend` | 96.0 [92.3, 98.0] 192/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 99.0 [96.4, 99.7] 198/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 99.0 [96.4, 99.7] 198/200 |
| `gated` | 99.0 [96.4, 99.7] 198/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 99.0 [96.4, 99.7] 198/200 |

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 25 (2.5) | 1 (0.1) | 974 (97.4) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 8 (4.0) | 192 (96.0) | 0 (0.0) |
| `pid_feedforward` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `gated` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) |

### `mss_transfer`, cg pad: success

| method | SS5/mss:mss |
|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **97.8** [95.8, 100.0]; seeds 95.5–100.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 99.5 [97.2, 99.9] 199/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 99.5 [97.2, 99.9] 199/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 |

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5/mss:mss | 1000 | 0 (0.0) | 1 (0.1) | 21 (2.1) | 0 (0.0) | 978 (97.8) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `gated` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |

### `mss_transfer_corpus`, cg pad: success

| method | SS5/mss:corpus |
|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **97.5** [95.3, 99.5]; seeds 95.0–99.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 99.5 [97.2, 99.9] 199/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 99.5 [97.2, 99.9] 199/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 |

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5/mss:corpus | 1000 | 0 (0.0) | 1 (0.1) | 25 (2.5) | 0 (0.0) | 974 (97.4) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `gated` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |

### `unseen_vessel` SS5, headings 180° and 135° only (matrix, aft)

| method | success, pooled over seeds |
|---|---|
| `ppo` (learned, PPO) | 99.8 (484/485 seed-episodes) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 97.9 (475/485 seed-episodes) |
| `residual_ppo` (learned, residual on pid_feedforward) | 100.0 (485/485 seed-episodes) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 100.0 (485/485 seed-episodes) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 100.0 (485/485 seed-episodes) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 100.0 (485/485 seed-episodes) |
| `pid_track_descend` | 83.5 (81/97 seed-episodes) |
| `pid_feedforward` | 100.0 (97/97 seed-episodes) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 99.0 (96/97 seed-episodes) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 99.0 (96/97 seed-episodes) |
| `gated` | 91.8 (89/97 seed-episodes) |
| `oracle_gated` — commit-timing oracle (privileged) | 93.8 (91/97 seed-episodes) |

## 7. Hypotheses (P3-D1 §8, P3-D4, P6-D6, P7-D1 §2, P7-D1a)

Every verdict is computed by `rld.eval.hypotheses` from the pre-registered rule named in its row. Rates and differences in points, r in %; 10 000 bootstrap replicates, seed 20260926 for every contrast (replicates correlated across contrasts; P7-D1a #11), percentile 95 % CI. No multiplicity correction (P3-D4 #9). H1a's non-inferiority, H1b and H4 use P3-D1 §4's paired bootstrap on per-episode differences (a mean over resampled seeds × episodes); their IQM-over-seeds values sit beside them as *post hoc, not scored*. H2 uses IQM success (P7-D1a #3). Each part of H1 and of H3 has its own verdict; there is no combined H1 or H3 verdict (P3-D4 #5, P7-D1a #2). Rows not labelled `scored` are not scored.

| H | part | role | cell | point [95 % CI] | threshold | verdict | rule | caveats |
|---|---|---|---|---|---|---|---|---|
| H1a | relative p95 + non-inferiority | scored | id SS5, aft | -47.0 [-69.6, -36.8] NI lower bound +1.2 | ≥ +15.0 | **not supported** | P3-D1 §8 H1a; P3-D4 #5; P7-D1a #3 (NI: mean paired bootstrap), #6 | P6-D5-residual-descent, two-phase-descent, bounce-grace-50ms, closing-speed-7pct, tunnelling-any-substep, P6-D5-tunnelling-bound, tilt-only-hard-landings, H1-narrowed, no-multiplicity, shared-bootstrap-seed |
| H1a | non-inferiority, IQM over seeds | post hoc, not scored | id SS5, aft | +4.0 [+1.0, +7.0] | – | **not scored** | P7-D1a #3: the IQM-based value, printed beside the scored mean | – |
| H1a | relative p95 with timeouts ranked worst | sensitivity (P3-D4 #8; not scored) | id SS5, aft | -47.0 [-69.6, -36.8] | – | **not scored** | P3-D4 #8; P7-D1a #8 (only timeouts ranked worst) | – |
| H1b | success difference | scored | id SS6, aft | +6.0 [+2.0, +10.2] | ≥ +5.0 | **supported** | P3-D1 §8 H1b; P7-D1a #3 (mean paired bootstrap), #6 | H1b-out-of-distribution, P6-D5-residual-descent, two-phase-descent, bounce-grace-50ms, closing-speed-7pct, tunnelling-any-substep, P6-D5-tunnelling-bound, tilt-only-hard-landings, H1-narrowed, no-multiplicity, shared-bootstrap-seed |
| H1b | success difference, IQM over seeds | post hoc, not scored | id SS6, aft | +5.7 [+1.8, +10.3] | – | **not scored** | P7-D1a #3: the IQM-based value, printed beside the scored mean | – |
| H2 | drop difference | scored | id SS5 -> unseen_seastate SS6, aft | -1.5 [-4.5, +1.7] | ≥ +10.0 | **not supported** | P3-D1 §8 H2 (prediction, IQM, test); P7-D1a #1 (mapping, user 2026-10-02), #3 (IQM), #4 (pairing) | regimes-overlap, two-phase-descent, bounce-grace-50ms, no-multiplicity, shared-bootstrap-seed |
| H3 | primary: id SS5 relative p95 | scored | id SS5, aft | -0.4 [-2.3, +2.3] | ≥ +10.0 | **not supported** | P3-D1 §8 H3; P7-D1a #2 (own verdict), #5, #6 | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed |
| H3 | primary: id SS6 relative p95 | scored | id SS6, aft | -0.4 [-6.3, +3.3] | ≥ +10.0 | **not supported** | P3-D1 §8 H3; P7-D1a #2 (own verdict), #5, #6 | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed |
| H3 | primary: unseen_vessel SS5 half-rule | scored | unseen_vessel SS5 vs id SS5, aft | -0.7 [-2.3, +1.4] | r(unseen_vessel) ≤ 0.5 × r(id) | **not applicable — no id gain to shrink** | P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict) | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed, regimes-overlap |
| H3 | primary: unseen_vessel SS6 half-rule | scored | unseen_vessel SS6 vs id SS6, aft | -0.4 [-3.3, +3.2] | r(unseen_vessel) ≤ 0.5 × r(id) | **not applicable — no id gain to shrink** | P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict) | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed, regimes-overlap |
| H3 | secondary: id SS5 relative p95 | secondary (pre-registered, reported beside) | id SS5, aft | -0.8 [-2.4, +0.9] | ≥ +10.0 | **not supported** | P3-D1 §8 H3; P7-D1a #2 (own verdict), #5, #6 | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed |
| H3 | secondary: id SS6 relative p95 | secondary (pre-registered, reported beside) | id SS6, aft | +1.9 [+0.4, +3.8] | ≥ +10.0 | **inconclusive** | P3-D1 §8 H3; P7-D1a #2 (own verdict), #5, #6 | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed |
| H3 | secondary: unseen_vessel SS5 half-rule | secondary (pre-registered, reported beside) | unseen_vessel SS5 vs id SS5, aft | -2.0 [-3.6, -0.3] | r(unseen_vessel) ≤ 0.5 × r(id) | **not applicable — no id gain to shrink** | P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict) | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed, regimes-overlap |
| H3 | secondary: unseen_vessel SS6 half-rule | secondary (pre-registered, reported beside) | unseen_vessel SS6 vs id SS6, aft | +0.0 [-1.9, +2.6] | r(unseen_vessel) ≤ 0.5 × r(id) | **holds** | P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict) | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed, regimes-overlap |
| H4 | drop difference | scored | id SS5, aft | +0.0 [+0.0, +0.0] | ≥ +10.0 | **not supported** | P3-D1 §8 H4 (prediction, test); P6-D6 (scored as written at id SS5); P7-D1a #1 (mapping, user 2026-10-02), #3 (mean paired bootstrap), #7 (D0.4) | H4-bounded, no-multiplicity, shared-bootstrap-seed |
| H4 | drop difference, IQM over seeds | post hoc, not scored | id SS5, aft | +0.0 [+0.0, +0.0] | – | **not scored** | P7-D1a #3: the IQM-based value, printed beside the scored mean | – |
| H5 | ORT CPU vs GPU p50 latency | pending | batch 1 | – | – | **pending — scored at Gate 8** | P3-D1 §8 H5; P7-D1 §7 (user decision 2026-10-01: deferred to Gate 8) | – |

### Numbers behind each verdict (`hypotheses.csv` notes)

- H1a, relative p95 + non-inferiority: p95 residual_ppo 0.2758 m/s, lowvz 0.1877 m/s; success difference (mean, scored) +0.0410 [+0.0120, +0.0710]; IQM over seeds (post hoc, not scored) +0.0400 [+0.0100, +0.0700]
- H1a, relative p95 with timeouts ranked worst: timeouts: residual_ppo 0, lowvz 0
- H1b, success difference: IQM over seeds (post hoc, not scored) +0.0567 [+0.0183, +0.1033]
- H2, drop difference: drop(ppo) +0.0267, drop(residual_ppo) +0.0417
- H3, primary: id SS5 relative p95: p95 ppo_forecast 0.2729 m/s, ppo 0.2720 m/s
- H3, primary: id SS6 relative p95: p95 ppo_forecast 0.2826 m/s, ppo 0.2814 m/s
- H3, primary: unseen_vessel SS5 half-rule: r(id SS5) -0.0036 [-0.0228, +0.0227]; r(unseen_vessel SS5) -0.0065 [-0.0234, +0.0144]
- H3, primary: unseen_vessel SS6 half-rule: r(id SS6) -0.0043 [-0.0628, +0.0329]; r(unseen_vessel SS6) -0.0042 [-0.0325, +0.0322]
- H3, secondary: id SS5 relative p95: p95 residual_ppo_forecast 0.2781 m/s, residual_ppo 0.2758 m/s
- H3, secondary: id SS6 relative p95: p95 residual_ppo_forecast 0.2785 m/s, residual_ppo 0.2838 m/s
- H3, secondary: unseen_vessel SS5 half-rule: r(id SS5) -0.0081 [-0.0244, +0.0089]; r(unseen_vessel SS5) -0.0195 [-0.0358, -0.0034]
- H3, secondary: unseen_vessel SS6 half-rule: r(id SS6) +0.0188 [+0.0042, +0.0382]; r(unseen_vessel SS6) +0.0004 [-0.0194, +0.0260]
- H4, drop difference: drop_sin +0.0000 (sin 1.0000, jon 1.0000); drop_jon +0.0000 (jon 1.0000, sin 1.0000). Novelty claim withdrawn: True. Triggers: P3-D1 §8 (CI does not exclude 0, i.e. lower bound <= 0, P7-D1a #6) True; D0.4 (ppo_sinusoid's JONSWAP id SS5 success 1.0000 >= ppo's 1.0000) True; with IQM over seeds (post hoc) 1.0000 >= 1.0000 True. IQM over seeds (post hoc, not scored) +0.0000 [+0.0000, +0.0000]

### Caveat keys

- `P6-D5-residual-descent`: The residual methods descend harder than their base; no residual seed learned the post-contact throttle cut, and a lowvz-like descent was within the residual's authority (P6-D5, corrected at P6-D6 M1).
- `two-phase-descent`: The learned methods land with a two-phase descent that the PID tuning space cannot express.
- `bounce-grace-50ms`: `bounce` is decided by the 50 ms contact-loss grace rule and is unstable at 240 Hz (P5-D1).
- `closing-speed-7pct`: Recorded closing speed understates impact speed by about 7 % (read after the first contact substep's impulse; P5-D14), for every method alike.
- `tunnelling-any-substep`: Penetration is flagged at any contact substep, not only at first contact (P5-D14).
- `P6-D5-tunnelling-bound`: Up to 3 / 1 / 0 / 3 SS6 successes per 1 000 of residual_ppo / ppo_forecast / residual_ppo_forecast / ppo_sinusoid may depend on tunnelling overlap (P6-D5); ppo 2, sac 41 (all SS), pid_feedforward_lowvz_cut 1 (P5-D14).
- `tilt-only-hard-landings`: Every hard landing of the four Phase 6 methods, ppo and pid_feedforward in e06 is a deck-tilt event (relative tilt > 15 deg with a nearly level drone); sac's and pid_track_descend's are mostly speed-driven (P6-D5).
- `H1-narrowed`: H1 reads 'lands softer than the lowest-closing-speed gain set in the P3-D3 log (constant-descent law), without losing success' -- not 'softer than any PID' (P3-D4 MAJOR-1).
- `no-multiplicity`: No multiplicity correction across H1a, H1b, H2, H3 and H4 (P3-D4 #9).
- `shared-bootstrap-seed`: Every contrast uses bootstrap seed 20260926 (P3-D1 §4), so replicate draws are correlated across contrasts; stated, not corrected (P7-D1a #11).
- `H1b-out-of-distribution`: id SS6 is outside every method's training distribution, so H1b is a sea-state extrapolation result (P3-D4 #7).
- `regimes-overlap`: The regimes share realizations (P3-D1 §2); a regime-vs-regime difference is not a comparison of independent draws.
- `P6-D1-forecast`: Forecast methods: the forecaster's forecasts were in-sample during training, and the past-only ship-motion feed is an extra ideal sensor the other methods lack (P6-D1).
- `H4-bounded`: H4 is arithmetically bounded at <= 0 at id SS5: ppo_sinusoid and ppo are both 200/200 on JONSWAP id SS5 in every seed (P6-D6); the user kept H4 at id SS5.

## Appendix A. Per-seed success with Wilson 95 % CI (main matrix, aft)

### `id`

| method | seed | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|---|
| `ppo` (learned, PPO) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 |
| `ppo` (learned, PPO) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 |
| `ppo` (learned, PPO) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 |
| `ppo` (learned, PPO) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 |
| `ppo` (learned, PPO) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0 | 99.5 [97.2, 99.9] 199/200 | 98.5 [95.7, 99.5] 197/200 | 89.0 [83.9, 92.6] 178/200 | 81.5 [75.5, 86.3] 163/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 1 | 100.0 [98.1, 100.0] 200/200 | 97.0 [93.6, 98.6] 194/200 | 80.0 [73.9, 85.0] 160/200 | 55.5 [48.6, 62.2] 111/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 2 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 91.0 [86.2, 94.2] 182/200 | 75.0 [68.6, 80.5] 150/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 3 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 87.0 [81.6, 91.0] 174/200 | 73.0 [66.5, 78.7] 146/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 4 | 98.5 [95.7, 99.5] 197/200 | 96.0 [92.3, 98.0] 192/200 | 85.0 [79.4, 89.3] 170/200 | 69.5 [62.8, 75.5] 139/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 96.0 [92.3, 98.0] 192/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 98.5 [95.7, 99.5] 197/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 96.5 [93.0, 98.3] 193/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 96.0 [92.3, 98.0] 192/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 95.5 [91.7, 97.6] 191/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 98.5 [95.7, 99.5] 197/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 96.5 [93.0, 98.3] 193/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 97.0 [93.6, 98.6] 194/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 96.5 [93.0, 98.3] 193/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 97.0 [93.6, 98.6] 194/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 97.0 [93.6, 98.6] 194/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 96.5 [93.0, 98.3] 193/200 |

### `unseen_seastate`

| method | seed | SS6 |
|---|---|---|
| `ppo` (learned, PPO) | 0 | 97.5 [94.3, 98.9] 195/200 |
| `ppo` (learned, PPO) | 1 | 97.5 [94.3, 98.9] 195/200 |
| `ppo` (learned, PPO) | 2 | 95.5 [91.7, 97.6] 191/200 |
| `ppo` (learned, PPO) | 3 | 98.5 [95.7, 99.5] 197/200 |
| `ppo` (learned, PPO) | 4 | 97.0 [93.6, 98.6] 194/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0 | 76.5 [70.2, 81.8] 153/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 1 | 58.5 [51.6, 65.1] 117/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 2 | 71.5 [64.9, 77.3] 143/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 3 | 69.5 [62.8, 75.5] 139/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 4 | 69.5 [62.8, 75.5] 139/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0 | 95.0 [91.0, 97.3] 190/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 1 | 93.5 [89.2, 96.2] 187/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 2 | 95.5 [91.7, 97.6] 191/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 3 | 95.5 [91.7, 97.6] 191/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 4 | 98.0 [95.0, 99.2] 196/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0 | 96.5 [93.0, 98.3] 193/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 1 | 97.0 [93.6, 98.6] 194/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 2 | 97.5 [94.3, 98.9] 195/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 3 | 96.5 [93.0, 98.3] 193/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 4 | 96.0 [92.3, 98.0] 192/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0 | 96.5 [93.0, 98.3] 193/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 1 | 95.0 [91.0, 97.3] 190/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 2 | 95.0 [91.0, 97.3] 190/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 3 | 96.5 [93.0, 98.3] 193/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 4 | 96.0 [92.3, 98.0] 192/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0 | 98.5 [95.7, 99.5] 197/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 1 | 96.5 [93.0, 98.3] 193/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 2 | 97.5 [94.3, 98.9] 195/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 3 | 96.5 [93.0, 98.3] 193/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 4 | 98.5 [95.7, 99.5] 197/200 |

### `unseen_heading`

| method | seed | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|---|
| `ppo` (learned, PPO) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 91.0 [86.2, 94.2] 182/200 |
| `ppo` (learned, PPO) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 85.5 [80.0, 89.7] 171/200 |
| `ppo` (learned, PPO) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 93.5 [89.2, 96.2] 187/200 |
| `ppo` (learned, PPO) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 93.0 [88.6, 95.8] 186/200 |
| `ppo` (learned, PPO) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 88.0 [82.8, 91.8] 176/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 86.0 [80.5, 90.1] 172/200 | 53.5 [46.6, 60.3] 107/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 1 | 100.0 [98.1, 100.0] 200/200 | 96.5 [93.0, 98.3] 193/200 | 75.0 [68.6, 80.5] 150/200 | 35.5 [29.2, 42.3] 71/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 2 | 100.0 [98.1, 100.0] 200/200 | 97.0 [93.6, 98.6] 194/200 | 86.0 [80.5, 90.1] 172/200 | 46.0 [39.2, 52.9] 92/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 3 | 100.0 [98.1, 100.0] 200/200 | 96.5 [93.0, 98.3] 193/200 | 87.5 [82.2, 91.4] 175/200 | 56.0 [49.1, 62.7] 112/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 4 | 100.0 [98.1, 100.0] 200/200 | 95.5 [91.7, 97.6] 191/200 | 80.5 [74.5, 85.4] 161/200 | 40.0 [33.5, 46.9] 80/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 86.0 [80.5, 90.1] 172/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 83.5 [77.7, 88.0] 167/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 85.0 [79.4, 89.3] 170/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 90.5 [85.6, 93.8] 181/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 86.0 [80.5, 90.1] 172/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 88.5 [83.3, 92.2] 177/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 84.0 [78.3, 88.4] 168/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 86.0 [80.5, 90.1] 172/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 89.5 [84.5, 93.0] 179/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 89.0 [83.9, 92.6] 178/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 89.0 [83.9, 92.6] 178/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 87.5 [82.2, 91.4] 175/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 86.0 [80.5, 90.1] 172/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 83.0 [77.2, 87.6] 166/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 91.0 [86.2, 94.2] 182/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 86.5 [81.1, 90.6] 173/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 86.5 [81.1, 90.6] 173/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 84.5 [78.8, 88.9] 169/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 85.5 [80.0, 89.7] 171/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 88.5 [83.3, 92.2] 177/200 |

### `unseen_vessel`

| method | seed | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|---|
| `ppo` (learned, PPO) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `ppo` (learned, PPO) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `ppo` (learned, PPO) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `ppo` (learned, PPO) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 99.5 [97.2, 99.9] 199/200 |
| `ppo` (learned, PPO) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 96.5 [93.0, 98.3] 193/200 | 80.0 [73.9, 85.0] 160/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 1 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 97.0 [93.6, 98.6] 194/200 | 78.5 [72.3, 83.6] 157/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 85.5 [80.0, 89.7] 171/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 95.5 [91.7, 97.6] 191/200 | 83.0 [77.2, 87.6] 166/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 4 | 99.5 [97.2, 99.9] 199/200 | 98.5 [95.7, 99.5] 197/200 | 96.0 [92.3, 98.0] 192/200 | 81.5 [75.5, 86.3] 163/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 1 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 2 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 3 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 4 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 |

### `static`

| method | seed | static |
|---|---|---|
| `ppo` (learned, PPO) | 0 | 100.0 [98.1, 100.0] 200/200 |
| `ppo` (learned, PPO) | 1 | 100.0 [98.1, 100.0] 200/200 |
| `ppo` (learned, PPO) | 2 | 100.0 [98.1, 100.0] 200/200 |
| `ppo` (learned, PPO) | 3 | 100.0 [98.1, 100.0] 200/200 |
| `ppo` (learned, PPO) | 4 | 100.0 [98.1, 100.0] 200/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0 | 100.0 [98.1, 100.0] 200/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 1 | 100.0 [98.1, 100.0] 200/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 2 | 100.0 [98.1, 100.0] 200/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 3 | 100.0 [98.1, 100.0] 200/200 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 4 | 100.0 [98.1, 100.0] 200/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0 | 100.0 [98.1, 100.0] 200/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 1 | 100.0 [98.1, 100.0] 200/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 2 | 100.0 [98.1, 100.0] 200/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 3 | 100.0 [98.1, 100.0] 200/200 |
| `residual_ppo` (learned, residual on pid_feedforward) | 4 | 100.0 [98.1, 100.0] 200/200 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0 | not run |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 1 | not run |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 2 | not run |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 3 | not run |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 4 | not run |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0 | not run |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 1 | not run |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 2 | not run |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 3 | not run |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 4 | not run |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0 | 100.0 [98.1, 100.0] 200/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 1 | 100.0 [98.1, 100.0] 200/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 2 | 100.0 [98.1, 100.0] 200/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 3 | 100.0 [98.1, 100.0] 200/200 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 4 | 100.0 [98.1, 100.0] 200/200 |

## Appendix B. Perception stand-in: outcome breakdown per condition

### `noise/sigma1cm_lat0step`: σp 1 cm / 0 step (0.0 ms), σv 0.05 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 12 (1.2) | 2 (0.2) | 986 (98.6) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 1 (0.1) | 4 (0.4) | 49 (4.9) | 1 (0.1) | 945 (94.5) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 2 (0.2) | 93 (9.3) | 3 (0.3) | 902 (90.2) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 5 (0.5) | 27 (2.7) | 166 (16.6) | 8 (0.8) | 793 (79.3) | 1 (0.1) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 24 (2.4) | 64 (6.4) | 287 (28.7) | 5 (0.5) | 619 (61.9) | 1 (0.1) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 1 (0.1) | 998 (99.8) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (0.2) | 998 (99.8) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 7 (0.7) | 2 (0.2) | 991 (99.1) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 33 (3.3) | 11 (1.1) | 956 (95.6) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 15 (1.5) | 1 (0.1) | 984 (98.4) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 10 (1.0) | 2 (0.2) | 988 (98.8) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 41 (4.1) | 11 (1.1) | 948 (94.8) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 22 (2.2) | 2 (0.2) | 976 (97.6) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 11 (5.5) | 187 (93.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 13 (6.5) | 24 (12.0) | 163 (81.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 19 (9.5) | 26 (13.0) | 155 (77.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 17 (8.5) | 176 (88.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 23 (11.5) | 177 (88.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 12 (6.0) | 36 (18.0) | 152 (76.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 191 (95.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 12 (6.0) | 18 (9.0) | 170 (85.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 189 (94.5) | 5 (2.5) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 172 (86.0) | 26 (13.0) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 110 (55.0) | 88 (44.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 193 (96.5) | 3 (1.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 173 (86.5) | 24 (12.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 122 (61.0) | 75 (37.5) |

### `noise/sigma2cm_lat0step`: σp 2 cm / 0 step (0.0 ms), σv 0.1 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (0.6) | 994 (99.4) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 11 (1.1) | 988 (98.8) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 11 (1.1) | 984 (98.4) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 25 (2.5) | 15 (1.5) | 959 (95.9) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 6 (0.6) | 35 (3.5) | 261 (26.1) | 12 (1.2) | 686 (68.6) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 3 (0.3) | 45 (4.5) | 280 (28.0) | 10 (1.0) | 662 (66.2) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 23 (2.3) | 86 (8.6) | 302 (30.2) | 10 (1.0) | 577 (57.7) | 2 (0.2) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 50 (5.0) | 123 (12.3) | 371 (37.1) | 19 (1.9) | 436 (43.6) | 1 (0.1) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 2 (0.2) | 12 (1.2) | 37 (3.7) | 142 (14.2) | 807 (80.7) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 7 (0.7) | 55 (5.5) | 144 (14.4) | 794 (79.4) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 7 (0.7) | 68 (6.8) | 121 (12.1) | 804 (80.4) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 6 (0.6) | 117 (11.7) | 118 (11.8) | 759 (75.9) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 0 (0.0) | 999 (99.9) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 1 (0.1) | 1 (0.1) | 25 (2.5) | 7 (0.7) | 966 (96.6) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 2 (0.2) | 8 (0.8) | 33 (3.3) | 103 (10.3) | 854 (85.4) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 1 (0.1) | 5 (0.5) | 40 (4.0) | 106 (10.6) | 848 (84.8) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 4 (0.4) | 49 (4.9) | 102 (10.2) | 845 (84.5) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 6 (0.6) | 92 (9.2) | 109 (10.9) | 793 (79.3) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 1 (0.1) | 996 (99.6) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 2 (0.2) | 4 (0.4) | 6 (0.6) | 988 (98.8) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 1 (0.1) | 8 (0.8) | 5 (0.5) | 986 (98.6) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 41 (4.1) | 12 (1.2) | 947 (94.7) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 37 (18.5) | 163 (81.5) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 41 (20.5) | 157 (78.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 54 (27.0) | 137 (68.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 14 (7.0) | 49 (24.5) | 137 (68.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 2 (1.0) | 5 (2.5) | 61 (30.5) | 132 (66.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 1 (0.5) | 7 (3.5) | 78 (39.0) | 114 (57.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 1 (0.5) | 7 (3.5) | 65 (32.5) | 127 (63.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 1 (0.5) | 17 (8.5) | 69 (34.5) | 113 (56.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 1 (0.5) | 0 (0.0) | 58 (29.0) | 141 (70.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 69 (34.5) | 130 (65.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 73 (36.5) | 121 (60.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 2 (1.0) | 16 (8.0) | 66 (33.0) | 116 (58.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 1 (0.5) | 0 (0.0) | 12 (6.0) | 187 (93.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 18 (9.0) | 181 (90.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 26 (13.0) | 168 (84.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 2 (1.0) | 16 (8.0) | 20 (10.0) | 162 (81.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 31 (15.5) | 32 (16.0) | 135 (67.5) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 21 (10.5) | 15 (7.5) | 164 (82.0) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 9 (4.5) | 11 (5.5) | 179 (89.5) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 6 (3.0) | 189 (94.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 1 (0.5) | 0 (0.0) | 2 (1.0) | 68 (34.0) | 58 (29.0) | 71 (35.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 46 (23.0) | 51 (25.5) | 101 (50.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 28 (14.0) | 29 (14.5) | 142 (71.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 11 (5.5) | 13 (6.5) | 175 (87.5) |

### `noise/sigma4cm_lat0step`: σp 4 cm / 0 step (0.0 ms), σv 0.2 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 1 (0.1) | 46 (4.6) | 77 (7.7) | 37 (3.7) | 839 (83.9) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 39 (3.9) | 88 (8.8) | 45 (4.5) | 828 (82.8) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 1 (0.1) | 38 (3.8) | 84 (8.4) | 84 (8.4) | 793 (79.3) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 34 (3.4) | 119 (11.9) | 53 (5.3) | 794 (79.4) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 88 (8.8) | 216 (21.6) | 391 (39.1) | 43 (4.3) | 253 (25.3) | 9 (0.9) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 105 (10.5) | 207 (20.7) | 414 (41.4) | 32 (3.2) | 233 (23.3) | 9 (0.9) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 165 (16.5) | 238 (23.8) | 388 (38.8) | 31 (3.1) | 174 (17.4) | 4 (0.4) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 219 (21.9) | 273 (27.3) | 339 (33.9) | 25 (2.5) | 136 (13.6) | 8 (0.8) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 132 (13.2) | 334 (33.4) | 282 (28.2) | 212 (21.2) | 40 (4.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 147 (14.7) | 320 (32.0) | 298 (29.8) | 198 (19.8) | 37 (3.7) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 157 (15.7) | 295 (29.5) | 306 (30.6) | 204 (20.4) | 37 (3.7) | 1 (0.1) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 152 (15.2) | 343 (34.3) | 319 (31.9) | 151 (15.1) | 35 (3.5) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 5 (0.5) | 11 (1.1) | 26 (2.6) | 958 (95.8) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 4 (0.4) | 13 (1.3) | 16 (1.6) | 967 (96.7) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 4 (0.4) | 19 (1.9) | 19 (1.9) | 958 (95.8) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 6 (0.6) | 59 (5.9) | 29 (2.9) | 906 (90.6) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 95 (9.5) | 326 (32.6) | 295 (29.5) | 238 (23.8) | 46 (4.6) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 86 (8.6) | 282 (28.2) | 337 (33.7) | 241 (24.1) | 54 (5.4) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 87 (8.7) | 324 (32.4) | 309 (30.9) | 232 (23.2) | 48 (4.8) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 93 (9.3) | 313 (31.3) | 313 (31.3) | 224 (22.4) | 57 (5.7) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 1 (0.1) | 59 (5.9) | 93 (9.3) | 70 (7.0) | 777 (77.7) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 2 (0.2) | 50 (5.0) | 101 (10.1) | 52 (5.2) | 795 (79.5) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 2 (0.2) | 49 (4.9) | 115 (11.5) | 70 (7.0) | 764 (76.4) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 2 (0.2) | 57 (5.7) | 151 (15.1) | 55 (5.5) | 735 (73.5) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 95 (47.5) | 41 (20.5) | 64 (32.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 84 (42.0) | 45 (22.5) | 71 (35.5) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 94 (47.0) | 37 (18.5) | 61 (30.5) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 81 (40.5) | 46 (23.0) | 69 (34.5) |
| `pid_feedforward` | SS3 | 200 | 90 (45.0) | 23 (11.5) | 7 (3.5) | 14 (7.0) | 1 (0.5) | 65 (32.5) |
| `pid_feedforward` | SS4 | 200 | 78 (39.0) | 35 (17.5) | 17 (8.5) | 11 (5.5) | 1 (0.5) | 58 (29.0) |
| `pid_feedforward` | SS5 | 200 | 89 (44.5) | 23 (11.5) | 16 (8.0) | 7 (3.5) | 0 (0.0) | 65 (32.5) |
| `pid_feedforward` | SS6 | 200 | 89 (44.5) | 30 (15.0) | 15 (7.5) | 12 (6.0) | 0 (0.0) | 54 (27.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 10 (5.0) | 44 (22.0) | 43 (21.5) | 85 (42.5) | 9 (4.5) | 9 (4.5) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 10 (5.0) | 36 (18.0) | 48 (24.0) | 82 (41.0) | 14 (7.0) | 10 (5.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 7 (3.5) | 34 (17.0) | 45 (22.5) | 89 (44.5) | 15 (7.5) | 10 (5.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 13 (6.5) | 44 (22.0) | 43 (21.5) | 76 (38.0) | 14 (7.0) | 10 (5.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 10 (5.0) | 44 (22.0) | 43 (21.5) | 20 (10.0) | 74 (37.0) | 9 (4.5) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 10 (5.0) | 36 (18.0) | 48 (24.0) | 11 (5.5) | 85 (42.5) | 10 (5.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 7 (3.5) | 34 (17.0) | 45 (22.5) | 17 (8.5) | 87 (43.5) | 10 (5.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 13 (6.5) | 44 (22.0) | 43 (21.5) | 19 (9.5) | 71 (35.5) | 10 (5.0) |
| `gated` | SS3 | 200 | 110 (55.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 90 (45.0) |
| `gated` | SS4 | 200 | 106 (53.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 94 (47.0) |
| `gated` | SS5 | 200 | 109 (54.5) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 90 (45.0) |
| `gated` | SS6 | 200 | 110 (55.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 89 (44.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 107 (53.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 93 (46.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 109 (54.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 91 (45.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 107 (53.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 93 (46.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 105 (52.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 95 (47.5) |

### `noise/sigma0cm_lat1step`: σp 0 cm / 1 step (33.3 ms), σv 0.0 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 4 (0.4) | 5 (0.5) | 198 (19.8) | 793 (79.3) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 4 (0.4) | 6 (0.6) | 172 (17.2) | 818 (81.8) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 4 (0.4) | 8 (0.8) | 135 (13.5) | 853 (85.3) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 20 (2.0) | 84 (8.4) | 895 (89.5) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 45 (4.5) | 106 (10.6) | 389 (38.9) | 18 (1.8) | 436 (43.6) | 6 (0.6) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 40 (4.0) | 115 (11.5) | 381 (38.1) | 17 (1.7) | 443 (44.3) | 4 (0.4) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 71 (7.1) | 174 (17.4) | 352 (35.2) | 18 (1.8) | 380 (38.0) | 5 (0.5) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 132 (13.2) | 221 (22.1) | 339 (33.9) | 19 (1.9) | 283 (28.3) | 6 (0.6) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 225 (22.5) | 769 (76.9) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 239 (23.9) | 758 (75.8) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 1 (0.1) | 1 (0.1) | 12 (1.2) | 220 (22.0) | 766 (76.6) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 1 (0.1) | 0 (0.0) | 46 (4.6) | 147 (14.7) | 806 (80.6) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (0.2) | 998 (99.8) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 3 (0.3) | 976 (97.6) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 35 (3.5) | 965 (96.5) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 1 (0.1) | 0 (0.0) | 0 (0.0) | 22 (2.2) | 977 (97.7) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 26 (2.6) | 969 (96.9) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 32 (3.2) | 26 (2.6) | 941 (94.1) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 3 (0.3) | 15 (1.5) | 21 (2.1) | 146 (14.6) | 815 (81.5) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 1 (0.1) | 27 (2.7) | 20 (2.0) | 144 (14.4) | 808 (80.8) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 21 (2.1) | 27 (2.7) | 120 (12.0) | 832 (83.2) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 1 (0.1) | 15 (1.5) | 61 (6.1) | 126 (12.6) | 797 (79.7) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 6 (3.0) | 191 (95.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 13 (6.5) | 19 (9.5) | 168 (84.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 18 (9.0) | 28 (14.0) | 154 (77.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 1 (0.5) | 0 (0.0) | 1 (0.5) | 5 (2.5) | 193 (96.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 13 (6.5) | 178 (89.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 191 (95.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 10 (5.0) | 188 (94.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 10 (5.0) | 33 (16.5) | 157 (78.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 5 (2.5) | 193 (96.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 10 (5.0) | 23 (11.5) | 167 (83.5) | 0 (0.0) |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 195 (97.5) | 4 (2.0) |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 177 (88.5) | 22 (11.0) |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 117 (58.5) | 74 (37.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 194 (97.0) | 4 (2.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 177 (88.5) | 22 (11.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 126 (63.0) | 72 (36.0) |

### `noise/sigma1cm_lat1step`: σp 1 cm / 1 step (33.3 ms), σv 0.05 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 12 (1.2) | 16 (1.6) | 126 (12.6) | 846 (84.6) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 6 (0.6) | 14 (1.4) | 101 (10.1) | 879 (87.9) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 1 (0.1) | 7 (0.7) | 23 (2.3) | 109 (10.9) | 860 (86.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 1 (0.1) | 2 (0.2) | 43 (4.3) | 96 (9.6) | 858 (85.8) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 49 (4.9) | 128 (12.8) | 391 (39.1) | 18 (1.8) | 409 (40.9) | 5 (0.5) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 43 (4.3) | 143 (14.3) | 398 (39.8) | 14 (1.4) | 398 (39.8) | 4 (0.4) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 74 (7.4) | 189 (18.9) | 383 (38.3) | 15 (1.5) | 335 (33.5) | 4 (0.4) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 170 (17.0) | 227 (22.7) | 352 (35.2) | 14 (1.4) | 235 (23.5) | 2 (0.2) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 6 (0.6) | 7 (0.7) | 41 (4.1) | 220 (22.0) | 726 (72.6) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 8 (0.8) | 7 (0.7) | 37 (3.7) | 195 (19.5) | 753 (75.3) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 4 (0.4) | 8 (0.8) | 55 (5.5) | 200 (20.0) | 733 (73.3) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 7 (0.7) | 8 (0.8) | 83 (8.3) | 173 (17.3) | 729 (72.9) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (0.2) | 998 (99.8) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 8 (0.8) | 971 (97.1) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 6 (0.6) | 2 (0.2) | 5 (0.5) | 95 (9.5) | 892 (89.2) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 3 (0.3) | 2 (0.2) | 4 (0.4) | 76 (7.6) | 915 (91.5) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 5 (0.5) | 4 (0.4) | 14 (1.4) | 81 (8.1) | 896 (89.6) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 6 (0.6) | 3 (0.3) | 38 (3.8) | 71 (7.1) | 882 (88.2) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 3 (0.3) | 19 (1.9) | 45 (4.5) | 136 (13.6) | 797 (79.7) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 3 (0.3) | 27 (2.7) | 45 (4.5) | 115 (11.5) | 810 (81.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 2 (0.2) | 30 (3.0) | 45 (4.5) | 121 (12.1) | 802 (80.2) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 1 (0.1) | 28 (2.8) | 77 (7.7) | 111 (11.1) | 783 (78.3) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 7 (3.5) | 191 (95.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 17 (8.5) | 20 (10.0) | 163 (81.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 17 (8.5) | 29 (14.5) | 154 (77.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 4 (2.0) | 0 (0.0) | 0 (0.0) | 25 (12.5) | 171 (85.5) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 3 (1.5) | 0 (0.0) | 2 (1.0) | 31 (15.5) | 164 (82.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 4 (2.0) | 0 (0.0) | 3 (1.5) | 38 (19.0) | 155 (77.5) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 5 (2.5) | 0 (0.0) | 12 (6.0) | 42 (21.0) | 141 (70.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 43 (21.5) | 157 (78.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 45 (22.5) | 155 (77.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 59 (29.5) | 141 (70.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 47 (23.5) | 145 (72.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 18 (9.0) | 182 (91.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 18 (9.0) | 182 (91.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 19 (9.5) | 173 (86.5) | 0 (0.0) |
| `gated` | SS3 | 200 | 2 (1.0) | 0 (0.0) | 0 (0.0) | 34 (17.0) | 164 (82.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 2 (1.0) | 0 (0.0) | 1 (0.5) | 22 (11.0) | 165 (82.5) | 10 (5.0) |
| `gated` | SS5 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 23 (11.5) | 120 (60.0) | 56 (28.0) |
| `gated` | SS6 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 12 (6.0) | 76 (38.0) | 111 (55.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 8 (4.0) | 0 (0.0) | 0 (0.0) | 31 (15.5) | 160 (80.0) | 1 (0.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 26 (13.0) | 168 (84.0) | 5 (2.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 5 (2.5) | 0 (0.0) | 0 (0.0) | 32 (16.0) | 126 (63.0) | 37 (18.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 1 (0.5) | 0 (0.0) | 1 (0.5) | 26 (13.0) | 88 (44.0) | 84 (42.0) |

### `noise/sigma2cm_lat1step`: σp 2 cm / 1 step (33.3 ms), σv 0.1 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 32 (3.2) | 50 (5.0) | 92 (9.2) | 826 (82.6) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 1 (0.1) | 23 (2.3) | 47 (4.7) | 98 (9.8) | 831 (83.1) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 1 (0.1) | 29 (2.9) | 42 (4.2) | 101 (10.1) | 827 (82.7) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 15 (1.5) | 72 (7.2) | 86 (8.6) | 827 (82.7) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 62 (6.2) | 144 (14.4) | 452 (45.2) | 17 (1.7) | 319 (31.9) | 6 (0.6) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 67 (6.7) | 180 (18.0) | 398 (39.8) | 27 (2.7) | 324 (32.4) | 4 (0.4) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 131 (13.1) | 183 (18.3) | 390 (39.0) | 24 (2.4) | 262 (26.2) | 10 (1.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 190 (19.0) | 282 (28.2) | 326 (32.6) | 18 (1.8) | 179 (17.9) | 5 (0.5) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 80 (8.0) | 87 (8.7) | 192 (19.2) | 314 (31.4) | 327 (32.7) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 52 (5.2) | 93 (9.3) | 201 (20.1) | 302 (30.2) | 352 (35.2) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 78 (7.8) | 109 (10.9) | 194 (19.4) | 291 (29.1) | 328 (32.8) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 72 (7.2) | 98 (9.8) | 228 (22.8) | 264 (26.4) | 338 (33.8) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 13 (1.3) | 987 (98.7) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (1.0) | 990 (99.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 1 (0.1) | 1 (0.1) | 12 (1.2) | 986 (98.6) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 3 (0.3) | 28 (2.8) | 12 (1.2) | 957 (95.7) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 42 (4.2) | 31 (3.1) | 103 (10.3) | 285 (28.5) | 539 (53.9) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 16 (1.6) | 50 (5.0) | 110 (11.0) | 294 (29.4) | 530 (53.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 51 (5.1) | 47 (4.7) | 135 (13.5) | 260 (26.0) | 507 (50.7) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 28 (2.8) | 43 (4.3) | 169 (16.9) | 258 (25.8) | 502 (50.2) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 4 (0.4) | 57 (5.7) | 109 (10.9) | 111 (11.1) | 719 (71.9) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 5 (0.5) | 58 (5.8) | 110 (11.0) | 107 (10.7) | 720 (72.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 4 (0.4) | 51 (5.1) | 81 (8.1) | 110 (11.0) | 754 (75.4) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 8 (0.8) | 54 (5.4) | 121 (12.1) | 100 (10.0) | 717 (71.7) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 34 (17.0) | 166 (83.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 50 (25.0) | 147 (73.5) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 12 (6.0) | 57 (28.5) | 131 (65.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 13 (6.5) | 55 (27.5) | 132 (66.0) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 76 (38.0) | 17 (8.5) | 17 (8.5) | 66 (33.0) | 18 (9.0) | 6 (3.0) |
| `pid_feedforward` | SS4 | 200 | 74 (37.0) | 18 (9.0) | 27 (13.5) | 52 (26.0) | 26 (13.0) | 3 (1.5) |
| `pid_feedforward` | SS5 | 200 | 79 (39.5) | 15 (7.5) | 20 (10.0) | 58 (29.0) | 20 (10.0) | 8 (4.0) |
| `pid_feedforward` | SS6 | 200 | 83 (41.5) | 13 (6.5) | 30 (15.0) | 53 (26.5) | 16 (8.0) | 5 (2.5) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 1 (0.5) | 0 (0.0) | 14 (7.0) | 109 (54.5) | 76 (38.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 2 (1.0) | 12 (6.0) | 108 (54.0) | 78 (39.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 2 (1.0) | 3 (1.5) | 11 (5.5) | 112 (56.0) | 72 (36.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 1 (0.5) | 22 (11.0) | 123 (61.5) | 54 (27.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 1 (0.5) | 0 (0.0) | 14 (7.0) | 27 (13.5) | 158 (79.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 2 (1.0) | 12 (6.0) | 29 (14.5) | 157 (78.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 2 (1.0) | 3 (1.5) | 11 (5.5) | 30 (15.0) | 154 (77.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 1 (0.5) | 22 (11.0) | 37 (18.5) | 140 (70.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 88 (44.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 0 (0.0) | 110 (55.0) |
| `gated` | SS4 | 200 | 92 (46.0) | 0 (0.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 107 (53.5) |
| `gated` | SS5 | 200 | 90 (45.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 0 (0.0) | 108 (54.0) |
| `gated` | SS6 | 200 | 97 (48.5) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 0 (0.0) | 102 (51.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 119 (59.5) | 0 (0.0) | 2 (1.0) | 13 (6.5) | 4 (2.0) | 62 (31.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 110 (55.0) | 1 (0.5) | 1 (0.5) | 8 (4.0) | 3 (1.5) | 77 (38.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 115 (57.5) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 0 (0.0) | 81 (40.5) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 100 (50.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 1 (0.5) | 96 (48.0) |

### `noise/sigma4cm_lat1step`: σp 4 cm / 1 step (33.3 ms), σv 0.2 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 3 (0.3) | 141 (14.1) | 162 (16.2) | 93 (9.3) | 601 (60.1) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 2 (0.2) | 123 (12.3) | 186 (18.6) | 102 (10.2) | 587 (58.7) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 6 (0.6) | 127 (12.7) | 177 (17.7) | 109 (10.9) | 581 (58.1) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 4 (0.4) | 114 (11.4) | 225 (22.5) | 112 (11.2) | 545 (54.5) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 189 (18.9) | 278 (27.8) | 340 (34.0) | 28 (2.8) | 153 (15.3) | 12 (1.2) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 193 (19.3) | 257 (25.7) | 359 (35.9) | 27 (2.7) | 155 (15.5) | 9 (0.9) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 252 (25.2) | 304 (30.4) | 291 (29.1) | 24 (2.4) | 116 (11.6) | 13 (1.3) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 362 (36.2) | 298 (29.8) | 235 (23.5) | 17 (1.7) | 67 (6.7) | 21 (2.1) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 571 (57.1) | 250 (25.0) | 143 (14.3) | 34 (3.4) | 2 (0.2) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 593 (59.3) | 258 (25.8) | 109 (10.9) | 37 (3.7) | 3 (0.3) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 597 (59.7) | 262 (26.2) | 115 (11.5) | 24 (2.4) | 2 (0.2) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 621 (62.1) | 251 (25.1) | 103 (10.3) | 21 (2.1) | 4 (0.4) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 10 (1.0) | 25 (2.5) | 55 (5.5) | 910 (91.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 6 (0.6) | 25 (2.5) | 40 (4.0) | 929 (92.9) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 14 (1.4) | 34 (3.4) | 48 (4.8) | 904 (90.4) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 13 (1.3) | 70 (7.0) | 48 (4.8) | 869 (86.9) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 420 (42.0) | 334 (33.4) | 173 (17.3) | 65 (6.5) | 8 (0.8) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 401 (40.1) | 312 (31.2) | 201 (20.1) | 78 (7.8) | 8 (0.8) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 418 (41.8) | 343 (34.3) | 167 (16.7) | 65 (6.5) | 7 (0.7) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 443 (44.3) | 303 (30.3) | 171 (17.1) | 72 (7.2) | 11 (1.1) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 15 (1.5) | 183 (18.3) | 238 (23.8) | 111 (11.1) | 453 (45.3) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 19 (1.9) | 199 (19.9) | 223 (22.3) | 94 (9.4) | 465 (46.5) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 13 (1.3) | 177 (17.7) | 217 (21.7) | 124 (12.4) | 469 (46.9) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 10 (1.0) | 175 (17.5) | 274 (27.4) | 114 (11.4) | 427 (42.7) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 91 (45.5) | 46 (23.0) | 63 (31.5) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 86 (43.0) | 44 (22.0) | 70 (35.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 96 (48.0) | 35 (17.5) | 63 (31.5) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 89 (44.5) | 49 (24.5) | 56 (28.0) |
| `pid_feedforward` | SS3 | 200 | 198 (99.0) | 2 (1.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 198 (99.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) |
| `pid_feedforward` | SS5 | 200 | 199 (99.5) | 0 (0.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 134 (67.0) | 26 (13.0) | 18 (9.0) | 15 (7.5) | 1 (0.5) | 6 (3.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 115 (57.5) | 44 (22.0) | 18 (9.0) | 13 (6.5) | 0 (0.0) | 10 (5.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 129 (64.5) | 27 (13.5) | 22 (11.0) | 10 (5.0) | 1 (0.5) | 11 (5.5) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 116 (58.0) | 35 (17.5) | 21 (10.5) | 18 (9.0) | 0 (0.0) | 10 (5.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 134 (67.0) | 26 (13.0) | 18 (9.0) | 4 (2.0) | 12 (6.0) | 6 (3.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 115 (57.5) | 44 (22.0) | 18 (9.0) | 8 (4.0) | 5 (2.5) | 10 (5.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 131 (65.5) | 25 (12.5) | 22 (11.0) | 4 (2.0) | 7 (3.5) | 11 (5.5) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 116 (58.0) | 35 (17.5) | 21 (10.5) | 6 (3.0) | 12 (6.0) | 10 (5.0) |
| `gated` | SS3 | 200 | 199 (99.5) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |

### `noise/sigma0cm_lat2step`: σp 0 cm / 2 step (66.7 ms), σv 0.0 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 22 (2.2) | 203 (20.3) | 361 (36.1) | 92 (9.2) | 322 (32.2) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 27 (2.7) | 191 (19.1) | 337 (33.7) | 109 (10.9) | 336 (33.6) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 15 (1.5) | 188 (18.8) | 299 (29.9) | 115 (11.5) | 383 (38.3) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 11 (1.1) | 157 (15.7) | 303 (30.3) | 147 (14.7) | 382 (38.2) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 98 (9.8) | 244 (24.4) | 459 (45.9) | 25 (2.5) | 170 (17.0) | 4 (0.4) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 99 (9.9) | 274 (27.4) | 447 (44.7) | 17 (1.7) | 159 (15.9) | 4 (0.4) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 187 (18.7) | 311 (31.1) | 366 (36.6) | 13 (1.3) | 114 (11.4) | 9 (0.9) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 275 (27.5) | 342 (34.2) | 288 (28.8) | 10 (1.0) | 67 (6.7) | 18 (1.8) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 833 (83.3) | 88 (8.8) | 60 (6.0) | 17 (1.7) | 2 (0.2) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 829 (82.9) | 103 (10.3) | 54 (5.4) | 13 (1.3) | 1 (0.1) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 809 (80.9) | 110 (11.0) | 63 (6.3) | 16 (1.6) | 2 (0.2) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 809 (80.9) | 107 (10.7) | 63 (6.3) | 14 (1.4) | 7 (0.7) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (0.4) | 996 (99.6) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 1 (0.1) | 1 (0.1) | 2 (0.2) | 996 (99.6) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 21 (2.1) | 15 (1.5) | 963 (96.3) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 604 (60.4) | 95 (9.5) | 82 (8.2) | 123 (12.3) | 96 (9.6) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 604 (60.4) | 105 (10.5) | 83 (8.3) | 117 (11.7) | 91 (9.1) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 560 (56.0) | 89 (8.9) | 110 (11.0) | 135 (13.5) | 106 (10.6) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 563 (56.3) | 98 (9.8) | 111 (11.1) | 135 (13.5) | 93 (9.3) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 69 (6.9) | 373 (37.3) | 350 (35.0) | 61 (6.1) | 147 (14.7) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 50 (5.0) | 350 (35.0) | 376 (37.6) | 74 (7.4) | 150 (15.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 57 (5.7) | 316 (31.6) | 364 (36.4) | 98 (9.8) | 165 (16.5) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 54 (5.4) | 313 (31.3) | 356 (35.6) | 106 (10.6) | 171 (17.1) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 5 (2.5) | 192 (96.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 13 (6.5) | 23 (11.5) | 164 (82.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 16 (8.0) | 19 (9.5) | 165 (82.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 199 (99.5) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 198 (99.0) | 0 (0.0) | 2 (1.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 195 (97.5) | 4 (2.0) | 0 (0.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 36 (18.0) | 163 (81.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 46 (23.0) | 153 (76.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 2 (1.0) | 1 (0.5) | 0 (0.0) | 55 (27.5) | 142 (71.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 6 (3.0) | 0 (0.0) | 17 (8.5) | 58 (29.0) | 119 (59.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 13 (6.5) | 186 (93.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 11 (5.5) | 188 (94.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 2 (1.0) | 1 (0.5) | 0 (0.0) | 19 (9.5) | 178 (89.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 6 (3.0) | 0 (0.0) | 17 (8.5) | 22 (11.0) | 155 (77.5) | 0 (0.0) |
| `gated` | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |

### `noise/sigma1cm_lat2step`: σp 1 cm / 2 step (66.7 ms), σv 0.05 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 23 (2.3) | 200 (20.0) | 314 (31.4) | 110 (11.0) | 353 (35.3) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 22 (2.2) | 209 (20.9) | 307 (30.7) | 110 (11.0) | 352 (35.2) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 15 (1.5) | 170 (17.0) | 311 (31.1) | 140 (14.0) | 364 (36.4) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 19 (1.9) | 188 (18.8) | 294 (29.4) | 130 (13.0) | 369 (36.9) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 94 (9.4) | 246 (24.6) | 484 (48.4) | 20 (2.0) | 151 (15.1) | 5 (0.5) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 120 (12.0) | 258 (25.8) | 435 (43.5) | 15 (1.5) | 169 (16.9) | 3 (0.3) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 205 (20.5) | 312 (31.2) | 344 (34.4) | 17 (1.7) | 114 (11.4) | 8 (0.8) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 307 (30.7) | 308 (30.8) | 301 (30.1) | 12 (1.2) | 66 (6.6) | 6 (0.6) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 855 (85.5) | 98 (9.8) | 36 (3.6) | 11 (1.1) | 0 (0.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 837 (83.7) | 100 (10.0) | 52 (5.2) | 10 (1.0) | 1 (0.1) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 809 (80.9) | 112 (11.2) | 60 (6.0) | 17 (1.7) | 2 (0.2) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 831 (83.1) | 99 (9.9) | 56 (5.6) | 12 (1.2) | 2 (0.2) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 1 (0.1) | 1 (0.1) | 0 (0.0) | 9 (0.9) | 989 (98.9) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 1 (0.1) | 0 (0.0) | 12 (1.2) | 987 (98.7) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 1 (0.1) | 2 (0.2) | 6 (0.6) | 991 (99.1) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 4 (0.4) | 20 (2.0) | 19 (1.9) | 957 (95.7) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 633 (63.3) | 103 (10.3) | 105 (10.5) | 99 (9.9) | 60 (6.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 605 (60.5) | 111 (11.1) | 114 (11.4) | 100 (10.0) | 70 (7.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 566 (56.6) | 118 (11.8) | 111 (11.1) | 123 (12.3) | 82 (8.2) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 600 (60.0) | 107 (10.7) | 136 (13.6) | 86 (8.6) | 71 (7.1) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 45 (4.5) | 393 (39.3) | 367 (36.7) | 71 (7.1) | 124 (12.4) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 68 (6.8) | 383 (38.3) | 332 (33.2) | 63 (6.3) | 154 (15.4) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 54 (5.4) | 365 (36.5) | 328 (32.8) | 82 (8.2) | 171 (17.1) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 43 (4.3) | 332 (33.2) | 368 (36.8) | 80 (8.0) | 177 (17.7) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 10 (5.0) | 186 (93.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 14 (7.0) | 24 (12.0) | 162 (81.0) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 15 (7.5) | 24 (12.0) | 161 (80.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 199 (99.5) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 199 (99.5) | 0 (0.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 17 (8.5) | 9 (4.5) | 10 (5.0) | 134 (67.0) | 30 (15.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 21 (10.5) | 5 (2.5) | 9 (4.5) | 131 (65.5) | 34 (17.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 27 (13.5) | 8 (4.0) | 15 (7.5) | 104 (52.0) | 46 (23.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 34 (17.0) | 7 (3.5) | 32 (16.0) | 94 (47.0) | 33 (16.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 17 (8.5) | 9 (4.5) | 10 (5.0) | 30 (15.0) | 134 (67.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 21 (10.5) | 5 (2.5) | 9 (4.5) | 30 (15.0) | 135 (67.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 27 (13.5) | 8 (4.0) | 15 (7.5) | 15 (7.5) | 135 (67.5) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 34 (17.0) | 7 (3.5) | 32 (16.0) | 25 (12.5) | 102 (51.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS6 | 200 | 199 (99.5) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 199 (99.5) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |

### `noise/sigma2cm_lat2step`: σp 2 cm / 2 step (66.7 ms), σv 0.1 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 18 (1.8) | 226 (22.6) | 324 (32.4) | 115 (11.5) | 317 (31.7) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 22 (2.2) | 241 (24.1) | 314 (31.4) | 96 (9.6) | 327 (32.7) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 22 (2.2) | 205 (20.5) | 279 (27.9) | 130 (13.0) | 364 (36.4) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 18 (1.8) | 201 (20.1) | 304 (30.4) | 119 (11.9) | 358 (35.8) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 117 (11.7) | 259 (25.9) | 442 (44.2) | 16 (1.6) | 160 (16.0) | 6 (0.6) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 159 (15.9) | 295 (29.5) | 408 (40.8) | 14 (1.4) | 116 (11.6) | 8 (0.8) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 227 (22.7) | 324 (32.4) | 320 (32.0) | 13 (1.3) | 105 (10.5) | 11 (1.1) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 347 (34.7) | 327 (32.7) | 256 (25.6) | 13 (1.3) | 48 (4.8) | 9 (0.9) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 881 (88.1) | 73 (7.3) | 41 (4.1) | 5 (0.5) | 0 (0.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 884 (88.4) | 78 (7.8) | 31 (3.1) | 6 (0.6) | 1 (0.1) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 851 (85.1) | 97 (9.7) | 39 (3.9) | 12 (1.2) | 1 (0.1) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 866 (86.6) | 87 (8.7) | 41 (4.1) | 6 (0.6) | 0 (0.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 3 (0.3) | 4 (0.4) | 28 (2.8) | 965 (96.5) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 4 (0.4) | 31 (3.1) | 965 (96.5) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 1 (0.1) | 1 (0.1) | 6 (0.6) | 24 (2.4) | 968 (96.8) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 4 (0.4) | 28 (2.8) | 37 (3.7) | 931 (93.1) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 660 (66.0) | 144 (14.4) | 119 (11.9) | 56 (5.6) | 21 (2.1) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 619 (61.9) | 168 (16.8) | 124 (12.4) | 68 (6.8) | 21 (2.1) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 647 (64.7) | 141 (14.1) | 109 (10.9) | 84 (8.4) | 19 (1.9) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 634 (63.4) | 159 (15.9) | 120 (12.0) | 70 (7.0) | 17 (1.7) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 67 (6.7) | 382 (38.2) | 329 (32.9) | 70 (7.0) | 152 (15.2) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 66 (6.6) | 376 (37.6) | 316 (31.6) | 98 (9.8) | 144 (14.4) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 52 (5.2) | 352 (35.2) | 326 (32.6) | 75 (7.5) | 195 (19.5) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 63 (6.3) | 357 (35.7) | 327 (32.7) | 75 (7.5) | 178 (17.8) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 38 (19.0) | 162 (81.0) | 0 (0.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 36 (18.0) | 160 (80.0) | 0 (0.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 14 (7.0) | 41 (20.5) | 145 (72.5) | 0 (0.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 16 (8.0) | 53 (26.5) | 131 (65.5) | 0 (0.0) |
| `pid_feedforward` | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 161 (80.5) | 16 (8.0) | 14 (7.0) | 9 (4.5) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 168 (84.0) | 16 (8.0) | 10 (5.0) | 5 (2.5) | 1 (0.5) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 159 (79.5) | 17 (8.5) | 8 (4.0) | 16 (8.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 168 (84.0) | 17 (8.5) | 10 (5.0) | 4 (2.0) | 0 (0.0) | 1 (0.5) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 161 (80.5) | 16 (8.0) | 14 (7.0) | 1 (0.5) | 8 (4.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 168 (84.0) | 16 (8.0) | 10 (5.0) | 2 (1.0) | 4 (2.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 159 (79.5) | 17 (8.5) | 8 (4.0) | 2 (1.0) | 14 (7.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 168 (84.0) | 17 (8.5) | 10 (5.0) | 0 (0.0) | 4 (2.0) | 1 (0.5) |
| `gated` | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |

### `noise/sigma4cm_lat2step`: σp 4 cm / 2 step (66.7 ms), σv 0.2 m/s

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 31 (3.1) | 281 (28.1) | 282 (28.2) | 96 (9.6) | 310 (31.0) | 0 (0.0) |
| `ppo` (learned, PPO) | SS4 | 1000 | 36 (3.6) | 304 (30.4) | 289 (28.9) | 82 (8.2) | 289 (28.9) | 0 (0.0) |
| `ppo` (learned, PPO) | SS5 | 1000 | 28 (2.8) | 268 (26.8) | 296 (29.6) | 97 (9.7) | 311 (31.1) | 0 (0.0) |
| `ppo` (learned, PPO) | SS6 | 1000 | 33 (3.3) | 268 (26.8) | 290 (29.0) | 110 (11.0) | 299 (29.9) | 0 (0.0) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 244 (24.4) | 349 (34.9) | 314 (31.4) | 14 (1.4) | 71 (7.1) | 8 (0.8) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 273 (27.3) | 350 (35.0) | 270 (27.0) | 26 (2.6) | 70 (7.0) | 11 (1.1) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 369 (36.9) | 329 (32.9) | 212 (21.2) | 22 (2.2) | 54 (5.4) | 14 (1.4) |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 452 (45.2) | 316 (31.6) | 166 (16.6) | 13 (1.3) | 36 (3.6) | 17 (1.7) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 925 (92.5) | 63 (6.3) | 11 (1.1) | 1 (0.1) | 0 (0.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 929 (92.9) | 54 (5.4) | 15 (1.5) | 2 (0.2) | 0 (0.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 939 (93.9) | 53 (5.3) | 8 (0.8) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 936 (93.6) | 53 (5.3) | 11 (1.1) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 1 (0.1) | 13 (1.3) | 24 (2.4) | 68 (6.8) | 894 (89.4) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 16 (1.6) | 21 (2.1) | 60 (6.0) | 903 (90.3) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 25 (2.5) | 40 (4.0) | 57 (5.7) | 878 (87.8) | 0 (0.0) |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 1 (0.1) | 31 (3.1) | 83 (8.3) | 64 (6.4) | 821 (82.1) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 854 (85.4) | 111 (11.1) | 31 (3.1) | 4 (0.4) | 0 (0.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 862 (86.2) | 104 (10.4) | 26 (2.6) | 7 (0.7) | 1 (0.1) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 865 (86.5) | 98 (9.8) | 34 (3.4) | 3 (0.3) | 0 (0.0) | 0 (0.0) |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 858 (85.8) | 102 (10.2) | 35 (3.5) | 5 (0.5) | 0 (0.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 81 (8.1) | 394 (39.4) | 334 (33.4) | 69 (6.9) | 122 (12.2) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 91 (9.1) | 392 (39.2) | 295 (29.5) | 72 (7.2) | 150 (15.0) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 88 (8.8) | 395 (39.5) | 297 (29.7) | 74 (7.4) | 146 (14.6) | 0 (0.0) |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 69 (6.9) | 386 (38.6) | 298 (29.8) | 71 (7.1) | 176 (17.6) | 0 (0.0) |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 95 (47.5) | 39 (19.5) | 66 (33.0) |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 86 (43.0) | 43 (21.5) | 70 (35.0) |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 96 (48.0) | 38 (19.0) | 60 (30.0) |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 85 (42.5) | 42 (21.0) | 68 (34.0) |
| `pid_feedforward` | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward` | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `gated` | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 200 (100.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
