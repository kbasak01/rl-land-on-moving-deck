# Results — Phase 7: evaluation under shift and ablations (simulation only)

- **Simulation only** (PyBullet, gym-pybullet-drones Crazyflie 2.x). No sentence here describes real flight or real deck data.
- Deck motion: dmf's 3-DOF (heave, roll, pitch) JONSWAP response, Froude-scaled to the drone at **lambda = 1/25** (1 s model = 5 s full scale), except where the lambda arm says otherwise. Surge, sway and yaw are absent.
- State-based observations; the perception stand-in (noise, latency) is the only sensor model and is off except in section 4. Not vision. Section 4 uses the P7-D4 stand-in: the deck is perceived (pad position and velocity delayed and noisy; orientation and normal delayed, no noise); the drone's own state is current and clean.
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
- `success ∧ tunnelled (> 5 mm)` in every outcome breakdown counts the successes (pooled over seeds) whose penetration exceeded `tunnelling_penetration_m` = 5 mm at any contact substep (`tunnelled_success.csv` per condition; review M4). Such a success may, but need not, depend on tunnelling overlap. P6-D5's bound (successes that may depend on the overlap, under P5-D14's rule) is the narrower subset of these it audited in `id`.
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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 1 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 15 (1.5) | 3 (0.3) | 982 (98.2) | 0 (0.0) | 3 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 4 (0.4) | 0 (0.0) | 996 (99.6) | 0 (0.0) | 3 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 1 (0.1) | 20 (2.0) | 0 (0.0) | 979 (97.9) | 0 (0.0) | 7 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 2 (0.2) | 8 (0.8) | 122 (12.2) | 3 (0.3) | 864 (86.4) | 1 (0.1) | 23 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 18 (1.8) | 39 (3.9) | 232 (23.2) | 2 (0.2) | 709 (70.9) | 0 (0.0) | 50 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 4 (0.4) | 0 (0.0) | 996 (99.6) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 24 (2.4) | 11 (1.1) | 965 (96.5) | 0 (0.0) | 4 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 18 (1.8) | 2 (0.2) | 980 (98.0) | 0 (0.0) | 4 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 0 (0.0) | 995 (99.5) | 0 (0.0) | 1 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 25 (2.5) | 4 (0.4) | 971 (97.1) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 25 (2.5) | 2 (0.2) | 973 (97.3) | 0 (0.0) | 3 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 4 (2.0) | 192 (96.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 12 (6.0) | 18 (9.0) | 170 (85.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 17 (8.5) | 23 (11.5) | 160 (80.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 1 (0.5) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 11 (5.5) | 181 (90.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 191 (95.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 21 (10.5) | 170 (85.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 15 (7.5) | 176 (88.0) | 0 (0.0) | 2 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 197 (98.5) | 3 (1.5) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 179 (89.5) | 21 (10.5) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 126 (63.0) | 70 (35.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 197 (98.5) | 3 (1.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 178 (89.0) | 21 (10.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 129 (64.5) | 71 (35.5) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 5 (0.5) | 972 (97.2) | 0 (0.0) | 8 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 19 (1.9) | 46 (4.6) | 241 (24.1) | 3 (0.3) | 691 (69.1) | 0 (0.0) | 49 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 38 (3.8) | 7 (0.7) | 955 (95.5) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 31 (3.1) | 2 (0.2) | 967 (96.7) | 0 (0.0) | 7 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 35 (3.5) | 7 (0.7) | 958 (95.8) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 2 (0.2) | 975 (97.5) | 0 (0.0) | 8 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 15 (7.5) | 23 (11.5) | 162 (81.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 14 (7.0) | 180 (90.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 21 (10.5) | 172 (86.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 14 (7.0) | 179 (89.5) | 0 (0.0) | 4 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 129 (64.5) | 69 (34.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 130 (65.0) | 70 (35.0) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 1 (0.1) | 3 (0.3) | 77 (7.7) | 17 (1.7) | 902 (90.2) | 0 (0.0) | 25 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 2 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 2 (0.2) | 28 (2.8) | 0 (0.0) | 970 (97.0) | 0 (0.0) | 2 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 2 (0.2) | 23 (2.3) | 139 (13.9) | 5 (0.5) | 830 (83.0) | 1 (0.1) | 25 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 86 (8.6) | 98 (9.8) | 343 (34.3) | 8 (0.8) | 462 (46.2) | 3 (0.3) | 44 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 107 (10.7) | 31 (3.1) | 862 (86.2) | 0 (0.0) | 5 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 113 (11.3) | 12 (1.2) | 874 (87.4) | 0 (0.0) | 22 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 102 (10.2) | 25 (2.5) | 873 (87.3) | 0 (0.0) | 3 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 116 (11.6) | 21 (2.1) | 863 (86.3) | 0 (0.0) | 19 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 22 (11.0) | 33 (16.5) | 145 (72.5) | 0 (0.0) | 2 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 24 (12.0) | 22 (11.0) | 154 (77.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 24 (12.0) | 36 (18.0) | 140 (70.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 24 (12.0) | 10 (5.0) | 166 (83.0) | 0 (0.0) | 6 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 196 (98.0) | 3 (1.5) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 84 (42.0) | 109 (54.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 93 (46.5) | 107 (53.5) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 0 (0.0) | 999 (99.9) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 7 (0.7) | 1 (0.1) | 992 (99.2) | 0 (0.0) | 2 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 0 (0.0) | 999 (99.9) | 0 (0.0) | 4 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 8 (0.8) | 0 (0.0) | 992 (99.2) | 0 (0.0) | 9 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 0 (0.0) | 2 (0.2) | 30 (3.0) | 1 (0.1) | 967 (96.7) | 0 (0.0) | 19 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 3 (0.3) | 23 (2.3) | 153 (15.3) | 3 (0.3) | 817 (81.7) | 1 (0.1) | 33 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 8 (0.8) | 2 (0.2) | 990 (99.0) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 1 (0.1) | 993 (99.3) | 0 (0.0) | 1 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 3 (0.3) | 991 (99.1) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 0 (0.0) | 995 (99.5) | 0 (0.0) | 5 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 14 (7.0) | 6 (3.0) | 180 (90.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 20 (10.0) | 17 (8.5) | 163 (81.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 2 (1.0) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 187 (93.5) | 13 (6.5) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 180 (90.0) | 20 (10.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 190 (95.0) | 10 (5.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 178 (89.0) | 22 (11.0) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | static | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | static | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 2 |
| `residual_ppo` (learned, residual on pid_feedforward) | static | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | static | not run |  |  |  |  |  |  |  |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | static | not run |  |  |  |  |  |  |  |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | static | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | static | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 0 (0.0) | 998 (99.8) | 0 (0.0) | 3 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 12 (1.2) | 3 (0.3) | 985 (98.5) | 0 (0.0) | 12 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 1 (0.1) | 996 (99.6) | 0 (0.0) | 3 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 1 (0.1) | 47 (4.7) | 2 (0.2) | 950 (95.0) | 0 (0.0) | 7 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 9 (0.9) | 33 (3.3) | 225 (22.5) | 5 (0.5) | 728 (72.8) | 0 (0.0) | 21 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 48 (4.8) | 102 (10.2) | 327 (32.7) | 1 (0.1) | 518 (51.8) | 4 (0.4) | 40 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 3 (0.3) | 992 (99.2) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 27 (2.7) | 9 (0.9) | 964 (96.4) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 1 (0.1) | 997 (99.7) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 21 (2.1) | 4 (0.4) | 974 (97.4) | 0 (0.0) | 8 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 2 (0.2) | 995 (99.5) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 11 (1.1) | 968 (96.8) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 0 (0.0) | 999 (99.9) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 1 (0.1) | 976 (97.6) | 0 (0.0) | 5 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 7 (3.5) | 192 (96.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 19 (9.5) | 27 (13.5) | 154 (77.0) | 0 (0.0) | 1 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 10 (5.0) | 184 (92.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 8 (4.0) | 190 (95.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 28 (14.0) | 165 (82.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 5 (2.5) | 193 (96.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 18 (9.0) | 175 (87.5) | 0 (0.0) | 1 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 184 (92.0) | 15 (7.5) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 132 (66.0) | 64 (32.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 182 (91.0) | 18 (9.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 131 (65.5) | 69 (34.5) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 19 (1.9) | 15 (1.5) | 966 (96.6) | 0 (0.0) | 15 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 67 (6.7) | 91 (9.1) | 328 (32.8) | 2 (0.2) | 511 (51.1) | 1 (0.1) | 39 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 36 (3.6) | 5 (0.5) | 959 (95.9) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 2 (0.2) | 28 (2.8) | 4 (0.4) | 966 (96.6) | 0 (0.0) | 9 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 38 (3.8) | 7 (0.7) | 955 (95.5) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 4 (0.4) | 973 (97.3) | 0 (0.0) | 5 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 24 (12.0) | 167 (83.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 10 (5.0) | 184 (92.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 15 (7.5) | 177 (88.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 7 (3.5) | 185 (92.5) | 0 (0.0) | 4 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 127 (63.5) | 69 (34.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 131 (65.5) | 69 (34.5) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 1 (0.1) | 3 (0.3) | 80 (8.0) | 16 (1.6) | 900 (90.0) | 0 (0.0) | 21 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 3 (0.3) | 35 (3.5) | 1 (0.1) | 961 (96.1) | 0 (0.0) | 8 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 9 (0.9) | 33 (3.3) | 159 (15.9) | 0 (0.0) | 798 (79.8) | 1 (0.1) | 26 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 99 (9.9) | 114 (11.4) | 329 (32.9) | 9 (0.9) | 444 (44.4) | 5 (0.5) | 49 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 113 (11.3) | 21 (2.1) | 866 (86.6) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 1 (0.1) | 0 (0.0) | 116 (11.6) | 13 (1.3) | 870 (87.0) | 0 (0.0) | 22 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 99 (9.9) | 20 (2.0) | 881 (88.1) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 1 (0.1) | 0 (0.0) | 118 (11.8) | 12 (1.2) | 869 (86.9) | 0 (0.0) | 18 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 27 (13.5) | 32 (16.0) | 141 (70.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 26 (13.0) | 22 (11.0) | 152 (76.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 191 (95.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 25 (12.5) | 34 (17.0) | 141 (70.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 25 (12.5) | 13 (6.5) | 162 (81.0) | 0 (0.0) | 3 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 197 (98.5) | 3 (1.5) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 82 (41.0) | 109 (54.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 89 (44.5) | 111 (55.5) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 2 (0.2) | 993 (99.3) | 0 (0.0) | 1 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 4 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 16 (1.6) | 0 (0.0) | 984 (98.4) | 0 (0.0) | 1 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 2 (0.2) | 12 (1.2) | 109 (10.9) | 0 (0.0) | 877 (87.7) | 0 (0.0) | 9 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 18 (1.8) | 78 (7.8) | 260 (26.0) | 5 (0.5) | 638 (63.8) | 1 (0.1) | 28 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 1 (0.1) | 993 (99.3) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 5 (0.5) | 1 (0.1) | 993 (99.3) | 0 (0.0) | 4 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 10 (1.0) | 1 (0.1) | 989 (98.9) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 0 (0.0) | 997 (99.7) | 0 (0.0) | 4 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 6 (3.0) | 190 (95.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 1 (0.5) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) | 0 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 197 (98.5) | 3 (1.5) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 182 (91.0) | 18 (9.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 177 (88.5) | 23 (11.5) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 4 (0.4) | 7 (0.7) | 989 (98.9) | 0 (0.0) | 12 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 0 (0.0) | 994 (99.4) | 0 (0.0) | 6 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 1 (0.1) | 44 (4.4) | 1 (0.1) | 954 (95.4) | 0 (0.0) | 9 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 3 (0.3) | 13 (1.3) | 161 (16.1) | 2 (0.2) | 821 (82.1) | 0 (0.0) | 53 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 27 (2.7) | 66 (6.6) | 280 (28.0) | 14 (1.4) | 612 (61.2) | 1 (0.1) | 60 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 10 (1.0) | 985 (98.5) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 7 (0.7) | 11 (1.1) | 5 (0.5) | 977 (97.7) | 0 (0.0) | 10 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 9 (0.9) | 5 (0.5) | 986 (98.6) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 1 (0.1) | 993 (99.3) | 0 (0.0) | 14 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 11 (5.5) | 187 (93.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 42 (21.0) | 29 (14.5) | 129 (64.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 50 (25.0) | 30 (15.0) | 120 (60.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 10 (5.0) | 188 (94.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 26 (13.0) | 169 (84.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 18 (9.0) | 177 (88.5) | 0 (0.0) | 0 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 151 (75.5) | 49 (24.5) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 58 (29.0) | 142 (71.0) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 150 (75.0) | 50 (25.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 57 (28.5) | 143 (71.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 |

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

## 4. Perception stand-in (P7-D4 definition), `id`, aft pad

**Stand-in definition (P7-D4, a dated deviation from P7-D1 §4 and the Phase 2 stand-in).** What is perceived is the **deck**: the analytic deck sample at the pad. At control step k the perceived sample is the true sample of step k − L, with L the configured latency quantised down to whole control steps (warm-up: the oldest available sample; no hold, 30 Hz).

- Pad **position and velocity**: delayed by L and **noisy** — zero-mean Gaussian, independent per world axis, σp and σv = σp / 0.2 s (0 / 0.05 / 0.10 / 0.20 m/s model scale), drawn from a spawned child of the episode seed.
- The noise is **white**: drawn i.i.d. at every 33.3 ms control step, position and velocity independently. A correlated estimator error of the same σ (P7-D1 §4's rationale, a velocity estimate smoothed over about 1 s full scale, i.e. about 6 control steps) was **not tested** (P7-D6).
- Pad **orientation and deck normal**: delayed by L, **no noise**.
- Every deck-derived observation entry — relative position, relative velocity, deck normal, relative tilt and pad-plane clearance — is computed from the perceived deck and the drone's **current, true** state.
- The drone's **own state** (attitude, rates, velocity), time, last action and contact flag: **current and clean**. Own velocity + relative velocity is therefore a consistent, stale deck velocity plus noise.
- The forecast methods' ship-motion feed stays ideal (P7-D1 §4). Termination, reward, touchdown detection and every metric use the true state.

Grid (P7-D1 §4): σp 0 / 1 / 2 / 4 cm × latency 0 / 1 / 2 control steps. 1 control step = 33.3 ms model = 166.7 ms full scale at lambda = 1/25. The configured 33.4 / 66.7 ms quantise down to exactly 1 / 2 steps. The clean (0 cm, 0 step) column is the main matrix's rows.

| condition | σp (cm) | σv (m/s) | latency configured (ms) | effective (control steps) | effective (ms, model) | effective (ms, full scale) | source |
|---|---|---|---|---|---|---|---|
| `noise/sigma1cm_lat0step` | 1 | 0.05 | 0 | 0 | 0.0 | 0.0 | flown (`summary.csv`) |
| `noise/sigma2cm_lat0step` | 2 | 0.1 | 0 | 0 | 0.0 | 0.0 | flown (`summary.csv`) |
| `noise/sigma4cm_lat0step` | 4 | 0.2 | 0 | 0 | 0.0 | 0.0 | flown (`summary.csv`) |
| `noise/sigma0cm_lat1step` | 0 | 0 | 33.4 | 1 | 33.3 | 166.7 | flown (`summary.csv`) |
| `noise/sigma1cm_lat1step` | 1 | 0.05 | 33.4 | 1 | 33.3 | 166.7 | flown (`summary.csv`) |
| `noise/sigma2cm_lat1step` | 2 | 0.1 | 33.4 | 1 | 33.3 | 166.7 | flown (`summary.csv`) |
| `noise/sigma4cm_lat1step` | 4 | 0.2 | 33.4 | 1 | 33.3 | 166.7 | flown (`summary.csv`) |
| `noise/sigma0cm_lat2step` | 0 | 0 | 66.7 | 2 | 66.7 | 333.3 | flown (`summary.csv`) |
| `noise/sigma1cm_lat2step` | 1 | 0.05 | 66.7 | 2 | 66.7 | 333.3 | flown (`summary.csv`) |
| `noise/sigma2cm_lat2step` | 2 | 0.1 | 66.7 | 2 | 66.7 | 333.3 | flown (`summary.csv`) |
| `noise/sigma4cm_lat2step` | 4 | 0.2 | 66.7 | 2 | 66.7 | 333.3 | flown (`summary.csv`) |

*Superseded (P7-D4):* `results/e07/noise_superseded_p7d1/` keeps, unchanged, the noise arm flown at `6b83e5c` under the Phase 2 stand-in. That stand-in delayed and noised only the six relative entries, so a 1–2 step latency mixed timestamps, and it left the clearance, deck normal and relative tilt ideal. Its SHA-256s are in P7-D2 §5 and `scripts/eval_phase7.py --check` verifies them. Its numbers are not reported here and are not comparable with this section.

Conditions written: 11 of 11.

Cells: success — learned: IQM over 5 seeds [stratified-bootstrap 95 % CI] and N; baselines: rate [Wilson 95 % CI] k/N — then the clean − noisy paired contrast in points [95 % CI]. Per-condition success tables, outcome breakdowns and tunnelled successes: Appendix B.

### `id` SS3: success % by condition (clean − noisy, points [95 % CI])

| method | σp 0 cm / 0 step (0.0 ms) | σp 1 cm / 0 step (0.0 ms) | σp 2 cm / 0 step (0.0 ms) | σp 4 cm / 0 step (0.0 ms) | σp 0 cm / 1 step (33.3 ms) | σp 1 cm / 1 step (33.3 ms) | σp 2 cm / 1 step (33.3 ms) | σp 4 cm / 1 step (33.3 ms) | σp 0 cm / 2 step (66.7 ms) | σp 1 cm / 2 step (66.7 ms) | σp 2 cm / 2 step (66.7 ms) | σp 4 cm / 2 step (66.7 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0] N 1000 | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.2** [98.7, 99.8] N 1000<br>+0.8 [+0.0, +2.0] | **79.8** [72.3, 89.7] N 1000<br>+20.2 [+10.3, +29.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.2** [99.0, 99.8] N 1000<br>+0.8 [+0.0, +1.7] | **79.7** [76.7, 87.5] N 1000<br>+20.3 [+12.2, +25.3] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.5** [98.3, 100.0] N 1000<br>+0.5 [+0.0, +2.0] | **80.0** [74.5, 87.8] N 1000<br>+20.0 [+12.0, +26.7] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **99.8** [98.8, 100.0] N 1000 | **92.5** [88.5, 97.5] N 1000<br>+7.3 [+1.7, +12.5] | **64.0** [57.2, 81.3] N 1000<br>+35.8 [+19.0, +45.0] | **19.2** [11.7, 32.7] N 1000<br>+80.7 [+67.0, +89.0] | **99.8** [98.2, 100.0] N 1000<br>+0.0 [-0.7, +1.2] | **92.5** [88.8, 97.0] N 1000<br>+7.3 [+1.7, +12.5] | **61.5** [55.2, 76.2] N 1000<br>+38.3 [+23.3, +47.0] | **18.2** [13.0, 37.0] N 1000<br>+81.7 [+62.8, +88.0] | **100.0** [98.0, 100.0] N 1000<br>-0.2 [-0.7, +1.3] | **92.2** [88.0, 96.5] N 1000<br>+7.7 [+2.3, +13.0] | **61.8** [52.7, 78.8] N 1000<br>+38.0 [+21.0, +48.5] | **20.5** [12.2, 33.7] N 1000<br>+79.3 [+66.0, +87.8] |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0] N 1000 | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **77.0** [75.5, 81.2] N 1000<br>+23.0 [+17.2, +27.5] | **2.5** [2.2, 4.2] N 1000<br>+97.5 [+95.0, +99.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **80.5** [76.3, 83.5] N 1000<br>+19.5 [+14.5, +25.0] | **2.7** [1.3, 6.0] N 1000<br>+97.3 [+93.3, +99.2] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **78.3** [74.8, 83.2] N 1000<br>+21.7 [+15.7, +27.0] | **3.3** [2.2, 5.3] N 1000<br>+96.7 [+93.8, +98.5] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0] N 1000 | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **96.3** [93.7, 98.2] N 1000<br>+3.7 [+1.2, +7.2] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **95.7** [93.3, 97.8] N 1000<br>+4.3 [+1.7, +7.5] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **97.0** [95.8, 98.2] N 1000<br>+3.0 [+1.2, +5.0] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0] N 1000 | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **87.5** [81.5, 92.0] N 1000<br>+12.5 [+7.7, +19.2] | **5.0** [3.0, 9.7] N 1000<br>+95.0 [+90.0, +97.7] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.8** [99.5, 100.0] N 1000<br>+0.2 [+0.0, +0.8] | **83.3** [79.0, 91.5] N 1000<br>+16.7 [+8.5, +22.7] | **4.3** [2.8, 8.2] N 1000<br>+95.7 [+91.3, +97.8] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **84.8** [82.5, 91.3] N 1000<br>+15.2 [+8.3, +19.3] | **4.2** [2.5, 9.8] N 1000<br>+95.8 [+90.0, +98.3] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0] N 1000 | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.0** [98.2, 99.5] N 1000<br>+1.0 [+0.0, +2.3] | **69.7** [58.3, 80.8] N 1000<br>+30.3 [+18.5, +42.8] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **98.8** [98.5, 99.0] N 1000<br>+1.2 [+0.2, +2.5] | **72.7** [57.2, 82.2] N 1000<br>+27.3 [+17.3, +42.2] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.3** [98.0, 99.8] N 1000<br>+0.7 [+0.0, +2.3] | **72.8** [61.0, 85.0] N 1000<br>+27.2 [+14.7, +39.8] |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200<br>+1.0 [+0.0, +2.5] | 86.0 [80.5, 90.1] 172/200<br>+14.0 [+9.5, +19.0] | 21.5 [16.4, 27.7] 43/200<br>+78.5 [+72.5, +84.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 97.0 [93.6, 98.6] 194/200<br>+3.0 [+1.0, +5.5] | 85.5 [80.0, 89.7] 171/200<br>+14.5 [+10.0, +19.5] | 19.5 [14.6, 25.5] 39/200<br>+80.5 [+75.0, +86.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 97.0 [93.6, 98.6] 194/200<br>+3.0 [+1.0, +5.5] | 81.0 [75.0, 85.8] 162/200<br>+19.0 [+13.5, +24.5] | 16.5 [12.0, 22.3] 33/200<br>+83.5 [+78.5, +88.5] |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200<br>+1.5 [+0.0, +3.5] | 60.0 [53.1, 66.5] 120/200<br>+40.0 [+33.0, +46.5] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 98.5 [95.7, 99.5] 197/200<br>+1.5 [+0.0, +3.5] | 63.0 [56.1, 69.4] 126/200<br>+37.0 [+30.5, +43.5] | 0.5 [0.1, 2.8] 1/200<br>+99.5 [+98.5, +100.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 99.5 [97.2, 99.9] 199/200<br>+0.5 [+0.0, +1.5] | 63.0 [56.1, 69.4] 126/200<br>+37.0 [+30.0, +44.0] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] 200/200 | 97.0 [93.6, 98.6] 194/200<br>+3.0 [+1.0, +5.5] | 66.5 [59.7, 72.7] 133/200<br>+33.5 [+27.0, +40.0] | 7.5 [4.6, 12.0] 15/200<br>+92.5 [+88.5, +96.0] | 99.5 [97.2, 99.9] 199/200<br>+0.5 [+0.0, +1.5] | 97.0 [93.6, 98.6] 194/200<br>+3.0 [+1.0, +5.5] | 71.5 [64.9, 77.3] 143/200<br>+28.5 [+22.0, +35.0] | 8.5 [5.4, 13.2] 17/200<br>+91.5 [+87.5, +95.0] | 98.0 [95.0, 99.2] 196/200<br>+2.0 [+0.5, +4.0] | 94.5 [90.4, 96.9] 189/200<br>+5.5 [+2.5, +9.0] | 61.5 [54.6, 68.0] 123/200<br>+38.5 [+31.5, +45.5] | 8.5 [5.4, 13.2] 17/200<br>+91.5 [+87.5, +95.0] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200<br>+1.0 [+0.0, +2.5] | 90.5 [85.6, 93.8] 181/200<br>+9.5 [+5.5, +14.0] | 41.5 [34.9, 48.4] 83/200<br>+58.5 [+52.0, +65.5] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 98.5 [95.7, 99.5] 197/200<br>+1.5 [+0.0, +3.5] | 92.5 [88.0, 95.4] 185/200<br>+7.5 [+4.0, +11.5] | 42.5 [35.9, 49.4] 85/200<br>+57.5 [+50.5, +64.5] | 98.5 [95.7, 99.5] 197/200<br>+1.5 [+0.0, +3.5] | 97.5 [94.3, 98.9] 195/200<br>+2.5 [+0.5, +5.0] | 90.5 [85.6, 93.8] 181/200<br>+9.5 [+5.5, +14.0] | 40.5 [33.9, 47.4] 81/200<br>+59.5 [+52.5, +66.0] |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 96.5 [93.0, 98.3] 193/200<br>+3.5 [+1.0, +6.5] | 16.5 [12.0, 22.3] 33/200<br>+83.5 [+78.0, +88.5] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 99.0 [96.4, 99.7] 198/200<br>+1.0 [+0.0, +2.5] | 11.5 [7.8, 16.7] 23/200<br>+88.5 [+84.0, +92.5] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 97.0 [93.6, 98.6] 194/200<br>+3.0 [+1.0, +5.5] | 13.0 [9.0, 18.4] 26/200<br>+87.0 [+82.0, +91.5] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200<br>+2.5 [+0.5, +5.0] | 30.5 [24.5, 37.2] 61/200<br>+69.5 [+63.0, +76.0] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 98.0 [95.0, 99.2] 196/200<br>+2.0 [+0.5, +4.0] | 28.0 [22.2, 34.6] 56/200<br>+72.0 [+66.0, +78.0] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 98.0 [95.0, 99.2] 196/200<br>+2.0 [+0.5, +4.0] | 29.5 [23.6, 36.2] 59/200<br>+70.5 [+64.0, +76.5] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] |

### `id` SS4: success % by condition (clean − noisy, points [95 % CI])

| method | σp 0 cm / 0 step (0.0 ms) | σp 1 cm / 0 step (0.0 ms) | σp 2 cm / 0 step (0.0 ms) | σp 4 cm / 0 step (0.0 ms) | σp 0 cm / 1 step (33.3 ms) | σp 1 cm / 1 step (33.3 ms) | σp 2 cm / 1 step (33.3 ms) | σp 4 cm / 1 step (33.3 ms) | σp 0 cm / 2 step (66.7 ms) | σp 1 cm / 2 step (66.7 ms) | σp 2 cm / 2 step (66.7 ms) | σp 4 cm / 2 step (66.7 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0] N 1000 | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.5** [99.2, 99.8] N 1000<br>+0.5 [+0.0, +1.3] | **81.5** [75.3, 89.3] N 1000<br>+18.5 [+10.3, +25.8] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **99.2** [98.7, 99.5] N 1000<br>+0.8 [+0.0, +2.0] | **79.8** [72.2, 89.7] N 1000<br>+20.2 [+10.3, +28.7] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.3** [98.7, 99.8] N 1000<br>+0.7 [+0.0, +1.8] | **80.2** [73.5, 84.0] N 1000<br>+19.8 [+14.5, +27.2] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **98.2** [96.3, 99.0] N 1000 | **89.7** [86.0, 93.3] N 1000<br>+8.5 [+4.3, +12.8] | **59.8** [52.0, 76.3] N 1000<br>+38.3 [+22.2, +47.5] | **16.5** [13.5, 30.0] N 1000<br>+81.7 [+68.2, +86.5] | **98.0** [96.8, 99.2] N 1000<br>+0.2 [-1.7, +1.5] | **88.5** [85.2, 95.3] N 1000<br>+9.7 [+2.5, +14.3] | **62.0** [57.3, 75.7] N 1000<br>+36.2 [+22.3, +43.0] | **19.3** [14.8, 35.5] N 1000<br>+78.8 [+62.7, +84.2] | **97.7** [96.3, 98.7] N 1000<br>+0.5 [-1.3, +2.2] | **87.5** [84.7, 94.3] N 1000<br>+10.7 [+3.5, +15.3] | **57.2** [48.7, 71.7] N 1000<br>+41.0 [+26.5, +50.8] | **18.3** [12.8, 33.5] N 1000<br>+79.8 [+65.3, +86.2] |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0] N 1000 | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **81.2** [75.2, 84.3] N 1000<br>+18.8 [+14.0, +25.5] | **3.0** [0.8, 5.3] N 1000<br>+97.0 [+94.0, +99.3] | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **99.7** [99.2, 100.0] N 1000<br>+0.3 [+0.0, +1.2] | **79.3** [75.0, 85.7] N 1000<br>+20.7 [+13.7, +26.5] | **2.3** [1.2, 5.2] N 1000<br>+97.7 [+94.5, +99.3] | **100.0** [99.3, 100.0] N 1000<br>+0.0 [+0.0, +0.8] | **99.8** [99.2, 100.0] N 1000<br>+0.2 [+0.0, +1.0] | **79.5** [76.7, 81.8] N 1000<br>+20.5 [+15.8, +25.5] | **4.7** [3.0, 6.2] N 1000<br>+95.3 [+92.8, +97.7] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0] N 1000 | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **95.2** [91.8, 98.3] N 1000<br>+4.8 [+1.5, +9.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **96.2** [95.0, 98.0] N 1000<br>+3.8 [+1.5, +6.3] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **95.3** [93.7, 97.2] N 1000<br>+4.7 [+2.2, +7.3] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0] N 1000 | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **84.5** [79.7, 92.0] N 1000<br>+15.5 [+8.0, +21.7] | **4.3** [2.3, 10.2] N 1000<br>+95.7 [+89.7, +98.2] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **85.7** [79.2, 91.2] N 1000<br>+14.3 [+8.3, +21.5] | **4.7** [3.8, 9.7] N 1000<br>+95.3 [+89.8, +97.3] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **85.8** [81.0, 88.7] N 1000<br>+14.2 [+9.5, +20.2] | **5.7** [4.7, 9.5] N 1000<br>+94.3 [+89.8, +96.5] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0] N 1000 | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **98.8** [98.0, 99.5] N 1000<br>+1.2 [+0.0, +2.7] | **73.5** [62.0, 84.0] N 1000<br>+26.5 [+15.7, +38.3] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.3** [98.2, 100.0] N 1000<br>+0.7 [+0.0, +2.2] | **72.7** [64.2, 78.5] N 1000<br>+27.3 [+19.5, +36.7] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.2** [97.8, 99.8] N 1000<br>+0.8 [+0.0, +2.5] | **70.8** [59.0, 79.5] N 1000<br>+29.2 [+19.0, +42.2] |
| `pid_track_descend` | 96.0 [92.3, 98.0] 192/200 | 95.5 [91.7, 97.6] 191/200<br>+0.5 [-2.5, +3.5] | 79.0 [72.8, 84.1] 158/200<br>+17.0 [+11.5, +23.0] | 22.5 [17.3, 28.8] 45/200<br>+73.5 [+67.0, +79.5] | 95.5 [91.7, 97.6] 191/200<br>+0.5 [-1.0, +2.5] | 96.0 [92.3, 98.0] 192/200<br>+0.0 [-3.0, +3.0] | 82.5 [76.6, 87.1] 165/200<br>+13.5 [+8.0, +19.0] | 22.0 [16.8, 28.2] 44/200<br>+74.0 [+67.5, +80.5] | 95.5 [91.7, 97.6] 191/200<br>+0.5 [+0.0, +1.5] | 94.0 [89.8, 96.5] 188/200<br>+2.0 [-1.0, +5.0] | 78.5 [72.3, 83.6] 157/200<br>+17.5 [+12.0, +23.5] | 24.0 [18.6, 30.4] 48/200<br>+72.0 [+65.5, +78.0] |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200<br>+2.5 [+0.5, +5.0] | 66.0 [59.2, 72.2] 132/200<br>+34.0 [+27.5, +40.5] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] | 100.0 [98.1, 100.0] 200/200<br>+0.0 [+0.0, +0.0] | 98.5 [95.7, 99.5] 197/200<br>+1.5 [+0.0, +3.5] | 62.0 [55.1, 68.4] 124/200<br>+38.0 [+31.5, +45.0] | 0.5 [0.1, 2.8] 1/200<br>+99.5 [+98.5, +100.0] | 99.5 [97.2, 99.9] 199/200<br>+0.5 [+0.0, +1.5] | 98.5 [95.7, 99.5] 197/200<br>+1.5 [+0.0, +3.5] | 64.5 [57.7, 70.8] 129/200<br>+35.5 [+29.0, +42.0] | 0.0 [0.0, 1.9] 0/200<br>+100.0 [+100.0, +100.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 98.0 [95.0, 99.2] 196/200 | 96.5 [93.0, 98.3] 193/200<br>+1.5 [-1.5, +5.0] | 65.0 [58.2, 71.3] 130/200<br>+33.0 [+26.0, +40.0] | 5.5 [3.1, 9.6] 11/200<br>+92.5 [+88.5, +96.0] | 96.5 [93.0, 98.3] 193/200<br>+1.5 [-1.5, +4.5] | 90.0 [85.1, 93.4] 180/200<br>+8.0 [+4.0, +12.5] | 65.5 [58.7, 71.7] 131/200<br>+32.5 [+26.0, +39.5] | 6.5 [3.8, 10.8] 13/200<br>+91.5 [+87.0, +95.5] | 88.5 [83.3, 92.2] 177/200<br>+9.5 [+5.0, +14.0] | 89.0 [83.9, 92.6] 178/200<br>+9.0 [+4.5, +13.5] | 56.5 [49.6, 63.2] 113/200<br>+41.5 [+34.5, +48.5] | 6.5 [3.8, 10.8] 13/200<br>+91.5 [+87.0, +95.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 98.5 [95.7, 99.5] 197/200 | 98.5 [95.7, 99.5] 197/200<br>+0.0 [-2.5, +2.5] | 85.5 [80.0, 89.7] 171/200<br>+13.0 [+8.0, +18.0] | 42.5 [35.9, 49.4] 85/200<br>+56.0 [+48.5, +63.0] | 98.5 [95.7, 99.5] 197/200<br>+0.0 [-2.5, +2.5] | 96.0 [92.3, 98.0] 192/200<br>+2.5 [-0.5, +5.5] | 88.5 [83.3, 92.2] 177/200<br>+10.0 [+5.5, +15.0] | 42.5 [35.9, 49.4] 85/200<br>+56.0 [+48.5, +63.0] | 98.5 [95.7, 99.5] 197/200<br>+0.0 [-2.5, +2.5] | 99.0 [96.4, 99.7] 198/200<br>-0.5 [-3.0, +1.5] | 88.5 [83.3, 92.2] 177/200<br>+10.0 [+5.5, +15.0] | 40.0 [33.5, 46.9] 80/200<br>+58.5 [+51.5, +65.5] |
| `gated` | 98.5 [95.7, 99.5] 197/200 | 94.5 [90.4, 96.9] 189/200<br>+4.0 [+1.5, +7.0] | 15.5 [11.1, 21.2] 31/200<br>+83.0 [+77.5, +88.0] | 0.0 [0.0, 1.9] 0/200<br>+98.5 [+96.5, +100.0] | 97.5 [94.3, 98.9] 195/200<br>+1.0 [+0.0, +2.5] | 94.5 [90.4, 96.9] 189/200<br>+4.0 [+1.5, +7.0] | 9.5 [6.2, 14.4] 19/200<br>+89.0 [+84.5, +93.0] | 0.0 [0.0, 1.9] 0/200<br>+98.5 [+96.5, +100.0] | 98.0 [95.0, 99.2] 196/200<br>+0.5 [+0.0, +1.5] | 95.0 [91.0, 97.3] 190/200<br>+3.5 [+1.0, +6.0] | 13.5 [9.4, 18.9] 27/200<br>+85.0 [+80.0, +89.5] | 0.0 [0.0, 1.9] 0/200<br>+98.5 [+96.5, +100.0] |
| `oracle_gated` — commit-timing oracle (privileged) | 98.5 [95.7, 99.5] 197/200 | 95.5 [91.7, 97.6] 191/200<br>+3.0 [+1.0, +5.5] | 27.0 [21.3, 33.5] 54/200<br>+71.5 [+65.0, +77.5] | 0.0 [0.0, 1.9] 0/200<br>+98.5 [+96.5, +100.0] | 99.0 [96.4, 99.7] 198/200<br>-0.5 [-1.5, +0.0] | 96.0 [92.3, 98.0] 192/200<br>+2.5 [+0.5, +5.0] | 25.5 [20.0, 32.0] 51/200<br>+73.0 [+67.0, +79.0] | 0.0 [0.0, 1.9] 0/200<br>+98.5 [+96.5, +100.0] | 99.0 [96.4, 99.7] 198/200<br>-0.5 [-1.5, +0.0] | 96.0 [92.3, 98.0] 192/200<br>+2.5 [+0.5, +5.0] | 26.0 [20.4, 32.5] 52/200<br>+72.5 [+66.5, +78.5] | 0.0 [0.0, 1.9] 0/200<br>+98.5 [+96.5, +100.0] |

### `id` SS5: success % by condition (clean − noisy, points [95 % CI])

| method | σp 0 cm / 0 step (0.0 ms) | σp 1 cm / 0 step (0.0 ms) | σp 2 cm / 0 step (0.0 ms) | σp 4 cm / 0 step (0.0 ms) | σp 0 cm / 1 step (33.3 ms) | σp 1 cm / 1 step (33.3 ms) | σp 2 cm / 1 step (33.3 ms) | σp 4 cm / 1 step (33.3 ms) | σp 0 cm / 2 step (66.7 ms) | σp 1 cm / 2 step (66.7 ms) | σp 2 cm / 2 step (66.7 ms) | σp 4 cm / 2 step (66.7 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0] N 1000 | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **98.2** [96.8, 99.3] N 1000<br>+1.8 [+0.3, +3.8] | **79.2** [72.7, 84.7] N 1000<br>+20.8 [+14.3, +28.2] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.3** [99.0, 99.8] N 1000<br>+0.7 [+0.0, +1.5] | **98.5** [98.2, 99.2] N 1000<br>+1.5 [+0.3, +2.7] | **73.7** [68.3, 85.7] N 1000<br>+26.3 [+14.5, +33.2] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.8** [99.2, 100.0] N 1000<br>+0.2 [+0.0, +1.0] | **98.5** [97.3, 99.3] N 1000<br>+1.5 [+0.3, +3.3] | **76.5** [67.5, 84.3] N 1000<br>+23.5 [+15.0, +33.0] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **87.0** [81.7, 90.3] N 1000 | **77.8** [73.0, 83.3] N 1000<br>+9.2 [+3.8, +14.0] | **50.2** [44.7, 65.7] N 1000<br>+36.8 [+22.0, +42.8] | **15.8** [14.3, 26.0] N 1000<br>+71.2 [+60.7, +76.5] | **87.3** [81.8, 88.3] N 1000<br>-0.3 [-3.3, +3.8] | **76.5** [72.0, 81.7] N 1000<br>+10.5 [+4.8, +15.2] | **51.5** [43.8, 61.5] N 1000<br>+35.5 [+26.5, +42.2] | **13.7** [8.3, 29.7] N 1000<br>+73.3 [+58.7, +80.5] | **84.3** [78.3, 89.3] N 1000<br>+2.7 [-2.8, +8.5] | **76.3** [69.0, 84.0] N 1000<br>+10.7 [+3.2, +17.3] | **47.5** [42.5, 57.7] N 1000<br>+39.5 [+29.5, +45.5] | **12.5** [10.7, 24.5] N 1000<br>+74.5 [+62.8, +79.0] |
| `residual_ppo` (learned, residual on pid_feedforward) | **99.5** [99.5, 99.8] N 1000 | **98.8** [98.2, 99.0] N 1000<br>+0.7 [+0.0, +2.2] | **79.7** [73.0, 85.2] N 1000<br>+19.8 [+13.3, +27.5] | **2.7** [2.2, 4.3] N 1000<br>+96.8 [+94.2, +98.5] | **99.3** [98.7, 99.8] N 1000<br>+0.2 [+0.0, +1.0] | **99.0** [98.2, 99.5] N 1000<br>+0.5 [+0.0, +2.0] | **77.7** [74.8, 82.5] N 1000<br>+21.8 [+15.7, +27.2] | **3.0** [1.8, 4.7] N 1000<br>+96.5 [+93.7, +98.7] | **99.0** [99.0, 99.3] N 1000<br>+0.5 [+0.0, +1.3] | **98.5** [97.2, 99.0] N 1000<br>+1.0 [+0.0, +3.0] | **75.2** [73.5, 77.2] N 1000<br>+24.3 [+19.7, +29.0] | **3.2** [1.0, 4.0] N 1000<br>+96.3 [+94.2, +99.0] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0] N 1000 | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **95.8** [94.3, 97.2] N 1000<br>+4.2 [+2.0, +6.7] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.7** [99.0, 100.0] N 1000<br>+0.3 [+0.0, +1.5] | **95.5** [93.8, 97.2] N 1000<br>+4.5 [+2.0, +7.2] | **100.0** [99.7, 100.0] N 1000<br>+0.0 [+0.0, +0.5] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.8** [99.5, 100.0] N 1000<br>+0.2 [+0.0, +0.8] | **96.2** [94.5, 97.8] N 1000<br>+3.8 [+1.5, +6.5] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **99.5** [99.5, 99.5] N 1000 | **98.8** [98.5, 99.5] N 1000<br>+0.7 [+0.0, +1.7] | **82.2** [76.5, 86.7] N 1000<br>+17.3 [+11.7, +24.0] | **3.5** [2.5, 8.5] N 1000<br>+96.0 [+90.7, +98.0] | **99.5** [99.2, 99.5] N 1000<br>+0.0 [+0.0, +0.5] | **98.8** [98.5, 99.3] N 1000<br>+0.7 [+0.0, +1.5] | **83.5** [80.8, 87.5] N 1000<br>+16.0 [+10.8, +20.5] | **5.2** [3.8, 8.7] N 1000<br>+94.3 [+90.2, +96.8] | **99.5** [99.2, 99.5] N 1000<br>+0.0 [+0.0, +0.5] | **98.3** [98.0, 99.2] N 1000<br>+1.2 [+0.0, +2.3] | **81.7** [80.2, 87.0] N 1000<br>+17.8 [+11.5, +22.0] | **5.3** [3.3, 6.8] N 1000<br>+94.2 [+91.0, +97.0] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0] N 1000 | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **99.2** [96.7, 100.0] N 1000<br>+0.8 [+0.0, +3.5] | **70.0** [56.2, 76.8] N 1000<br>+30.0 [+21.7, +43.5] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **97.2** [96.7, 97.8] N 1000<br>+2.8 [+1.2, +4.7] | **71.0** [53.5, 82.2] N 1000<br>+29.0 [+17.5, +46.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **100.0** [100.0, 100.0] N 1000<br>+0.0 [+0.0, +0.0] | **97.3** [93.8, 98.8] N 1000<br>+2.7 [+0.7, +6.7] | **69.2** [52.7, 79.5] N 1000<br>+30.8 [+19.8, +47.0] |
| `pid_track_descend` | 85.0 [79.4, 89.3] 170/200 | 83.0 [77.2, 87.6] 166/200<br>+2.0 [-2.5, +6.5] | 71.0 [64.4, 76.8] 142/200<br>+14.0 [+7.5, +20.5] | 22.0 [16.8, 28.2] 44/200<br>+63.0 [+55.5, +70.0] | 86.0 [80.5, 90.1] 172/200<br>-1.0 [-4.5, +2.5] | 82.0 [76.1, 86.7] 164/200<br>+3.0 [-1.5, +7.5] | 74.0 [67.5, 79.6] 148/200<br>+11.0 [+5.0, +17.0] | 22.0 [16.8, 28.2] 44/200<br>+63.0 [+56.0, +70.0] | 84.5 [78.8, 88.9] 169/200<br>+0.5 [-3.5, +4.5] | 83.5 [77.7, 88.0] 167/200<br>+1.5 [-3.0, +6.0] | 72.5 [65.9, 78.2] 145/200<br>+12.5 [+7.0, +18.0] | 20.5 [15.5, 26.6] 41/200<br>+64.5 [+57.5, +71.0] |
| `pid_feedforward` | 99.0 [96.4, 99.7] 198/200 | 93.5 [89.2, 96.2] 187/200<br>+5.5 [+2.5, +9.0] | 56.0 [49.1, 62.7] 112/200<br>+43.0 [+36.5, +50.0] | 0.0 [0.0, 1.9] 0/200<br>+99.0 [+97.5, +100.0] | 96.5 [93.0, 98.3] 193/200<br>+2.5 [+0.5, +5.0] | 95.0 [91.0, 97.3] 190/200<br>+4.0 [+1.0, +7.0] | 61.0 [54.1, 67.5] 122/200<br>+38.0 [+31.0, +45.0] | 0.0 [0.0, 1.9] 0/200<br>+99.0 [+97.5, +100.0] | 92.5 [88.0, 95.4] 185/200<br>+6.5 [+3.0, +10.5] | 90.0 [85.1, 93.4] 180/200<br>+9.0 [+5.0, +13.5] | 61.5 [54.6, 68.0] 123/200<br>+37.5 [+30.5, +44.5] | 0.5 [0.1, 2.8] 1/200<br>+98.5 [+96.5, +100.0] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 95.5 [91.7, 97.6] 191/200 | 88.5 [83.3, 92.2] 177/200<br>+7.0 [+2.5, +12.0] | 58.0 [51.1, 64.6] 116/200<br>+37.5 [+29.5, +45.5] | 9.0 [5.8, 13.8] 18/200<br>+86.5 [+81.5, +91.0] | 88.5 [83.3, 92.2] 177/200<br>+7.0 [+2.5, +11.5] | 83.5 [77.7, 88.0] 167/200<br>+12.0 [+6.5, +18.0] | 59.5 [52.6, 66.1] 119/200<br>+36.0 [+28.5, +43.5] | 6.5 [3.8, 10.8] 13/200<br>+89.0 [+84.5, +93.0] | 80.5 [74.5, 85.4] 161/200<br>+15.0 [+9.0, +21.0] | 70.5 [63.8, 76.4] 141/200<br>+25.0 [+18.5, +31.5] | 56.5 [49.6, 63.2] 113/200<br>+39.0 [+31.5, +46.5] | 12.0 [8.2, 17.2] 24/200<br>+83.5 [+78.0, +88.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 98.0 [95.0, 99.2] 196/200 | 95.5 [91.7, 97.6] 191/200<br>+2.5 [-1.0, +6.0] | 83.0 [77.2, 87.6] 166/200<br>+15.0 [+9.5, +20.5] | 46.5 [39.7, 53.4] 93/200<br>+51.5 [+44.0, +58.5] | 95.5 [91.7, 97.6] 191/200<br>+2.5 [-0.5, +5.5] | 96.0 [92.3, 98.0] 192/200<br>+2.0 [-1.0, +5.5] | 88.0 [82.8, 91.8] 176/200<br>+10.0 [+5.0, +15.0] | 43.0 [36.3, 49.9] 86/200<br>+55.0 [+48.0, +62.0] | 94.0 [89.8, 96.5] 188/200<br>+4.0 [+0.0, +8.0] | 93.0 [88.6, 95.8] 186/200<br>+5.0 [+1.5, +9.0] | 91.5 [86.8, 94.6] 183/200<br>+6.5 [+2.0, +11.0] | 40.5 [33.9, 47.4] 81/200<br>+57.5 [+50.5, +64.5] |
| `gated` | 89.5 [84.5, 93.0] 179/200 | 81.0 [75.0, 85.8] 162/200<br>+8.5 [+5.0, +12.5] | 3.0 [1.4, 6.4] 6/200<br>+86.5 [+81.5, +91.0] | 0.0 [0.0, 1.9] 0/200<br>+89.5 [+85.5, +93.5] | 88.0 [82.8, 91.8] 176/200<br>+1.5 [-0.5, +4.0] | 82.5 [76.6, 87.1] 165/200<br>+7.0 [+3.5, +10.5] | 5.5 [3.1, 9.6] 11/200<br>+84.0 [+79.0, +89.0] | 0.0 [0.0, 1.9] 0/200<br>+89.5 [+85.5, +93.5] | 87.0 [81.6, 91.0] 174/200<br>+2.5 [+0.0, +5.0] | 76.5 [70.2, 81.8] 153/200<br>+13.0 [+8.5, +17.5] | 2.0 [0.8, 5.0] 4/200<br>+87.5 [+83.0, +92.0] | 0.0 [0.0, 1.9] 0/200<br>+89.5 [+85.5, +93.5] |
| `oracle_gated` — commit-timing oracle (privileged) | 89.0 [83.9, 92.6] 178/200 | 86.0 [80.5, 90.1] 172/200<br>+3.0 [+0.5, +6.0] | 16.5 [12.0, 22.3] 33/200<br>+72.5 [+66.0, +78.5] | 0.0 [0.0, 1.9] 0/200<br>+89.0 [+84.5, +93.0] | 89.5 [84.5, 93.0] 179/200<br>-0.5 [-2.5, +1.0] | 85.0 [79.4, 89.3] 170/200<br>+4.0 [+1.0, +7.0] | 16.5 [12.0, 22.3] 33/200<br>+72.5 [+66.5, +78.5] | 0.0 [0.0, 1.9] 0/200<br>+89.0 [+84.5, +93.0] | 88.0 [82.8, 91.8] 176/200<br>+1.0 [-1.0, +3.0] | 83.5 [77.7, 88.0] 167/200<br>+5.5 [+2.0, +9.0] | 12.0 [8.2, 17.2] 24/200<br>+77.0 [+71.0, +83.0] | 0.0 [0.0, 1.9] 0/200<br>+89.0 [+84.5, +93.0] |

### `id` SS6: success % by condition (clean − noisy, points [95 % CI])

| method | σp 0 cm / 0 step (0.0 ms) | σp 1 cm / 0 step (0.0 ms) | σp 2 cm / 0 step (0.0 ms) | σp 4 cm / 0 step (0.0 ms) | σp 0 cm / 1 step (33.3 ms) | σp 1 cm / 1 step (33.3 ms) | σp 2 cm / 1 step (33.3 ms) | σp 4 cm / 1 step (33.3 ms) | σp 0 cm / 2 step (66.7 ms) | σp 1 cm / 2 step (66.7 ms) | σp 2 cm / 2 step (66.7 ms) | σp 4 cm / 2 step (66.7 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | **98.2** [98.0, 98.5] N 1000 | **98.5** [97.3, 99.0] N 1000<br>-0.3 [-1.5, +1.5] | **96.2** [94.5, 97.7] N 1000<br>+2.0 [-0.2, +4.5] | **75.7** [69.3, 81.5] N 1000<br>+22.5 [+15.5, +30.5] | **98.3** [97.7, 99.2] N 1000<br>-0.2 [-1.5, +1.0] | **98.2** [97.3, 98.5] N 1000<br>+0.0 [-1.2, +1.5] | **95.7** [94.3, 96.8] N 1000<br>+2.5 [+0.5, +4.8] | **75.8** [69.8, 84.2] N 1000<br>+22.3 [+13.5, +29.8] | **98.3** [97.7, 98.8] N 1000<br>-0.2 [-1.3, +1.2] | **98.2** [96.8, 99.0] N 1000<br>-0.0 [-1.0, +1.7] | **95.0** [94.5, 96.8] N 1000<br>+3.2 [+0.3, +5.5] | **73.2** [68.5, 79.5] N 1000<br>+25.0 [+17.3, +31.8] |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **72.5** [60.2, 79.3] N 1000 | **66.8** [58.7, 73.3] N 1000<br>+5.7 [-0.7, +10.5] | **40.3** [33.8, 51.3] N 1000<br>+32.2 [+21.7, +39.0] | **10.5** [4.2, 17.8] N 1000<br>+62.0 [+53.0, +68.3] | **74.0** [62.0, 78.8] N 1000<br>-1.5 [-6.7, +4.0] | **62.3** [61.0, 68.8] N 1000<br>+10.2 [-1.7, +14.8] | **37.8** [34.2, 51.7] N 1000<br>+34.7 [+20.2, +41.8] | **9.5** [6.2, 16.2] N 1000<br>+63.0 [+51.5, +69.5] | **67.2** [59.0, 76.3] N 1000<br>+5.3 [-2.0, +11.2] | **59.3** [52.3, 65.3] N 1000<br>+13.2 [+5.7, +19.0] | **39.7** [31.3, 52.5] N 1000<br>+32.8 [+23.5, +39.3] | **9.8** [6.7, 15.2] N 1000<br>+62.7 [+50.8, +69.5] |
| `residual_ppo` (learned, residual on pid_feedforward) | **96.2** [95.7, 97.8] N 1000 | **95.0** [95.0, 95.7] N 1000<br>+1.2 [-0.5, +3.3] | **74.5** [72.0, 78.3] N 1000<br>+21.7 [+16.8, +26.8] | **3.2** [2.5, 6.7] N 1000<br>+93.0 [+89.3, +95.7] | **96.7** [94.8, 97.5] N 1000<br>-0.5 [-2.3, +2.2] | **95.2** [94.3, 95.8] N 1000<br>+1.0 [-1.2, +3.8] | **72.7** [70.8, 75.7] N 1000<br>+23.5 [+18.7, +28.3] | **3.3** [2.3, 7.0] N 1000<br>+92.8 [+89.3, +95.5] | **96.3** [94.5, 97.8] N 1000<br>-0.2 [-2.5, +2.7] | **94.5** [93.2, 95.3] N 1000<br>+1.7 [-1.0, +5.3] | **73.2** [67.2, 77.3] N 1000<br>+23.0 [+17.8, +30.2] | **3.7** [3.0, 5.2] N 1000<br>+92.5 [+89.5, +95.3] |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **98.0** [97.5, 98.5] N 1000 | **97.7** [97.5, 98.3] N 1000<br>+0.3 [-1.2, +1.5] | **96.7** [95.8, 97.3] N 1000<br>+1.3 [-0.3, +3.3] | **89.8** [89.2, 92.2] N 1000<br>+8.2 [+4.7, +11.0] | **98.2** [97.7, 98.8] N 1000<br>-0.2 [-1.5, +1.0] | **97.7** [97.5, 98.3] N 1000<br>+0.3 [-1.0, +1.5] | **96.7** [95.3, 97.7] N 1000<br>+1.3 [-0.2, +3.2] | **90.7** [87.8, 93.2] N 1000<br>+7.3 [+3.5, +11.3] | **98.5** [98.2, 98.5] N 1000<br>-0.5 [-1.7, +0.8] | **98.0** [96.8, 98.5] N 1000<br>+0.0 [-1.2, +2.0] | **97.2** [96.7, 98.2] N 1000<br>+0.8 [-0.7, +2.2] | **90.3** [87.5, 91.7] N 1000<br>+7.7 [+4.7, +11.7] |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **96.8** [96.5, 98.0] N 1000 | **95.2** [93.8, 96.3] N 1000<br>+1.7 [-0.5, +4.5] | **75.7** [69.7, 83.0] N 1000<br>+21.2 [+13.3, +28.5] | **5.5** [4.0, 11.2] N 1000<br>+91.3 [+85.3, +94.5] | **97.2** [96.3, 97.5] N 1000<br>-0.3 [-2.0, +2.0] | **95.8** [94.8, 96.3] N 1000<br>+1.0 [-0.7, +3.7] | **80.7** [76.8, 87.2] N 1000<br>+16.2 [+9.2, +22.2] | **4.0** [3.2, 9.3] N 1000<br>+92.8 [+87.0, +96.0] | **95.8** [95.0, 97.8] N 1000<br>+1.0 [-1.3, +3.2] | **95.2** [94.0, 96.7] N 1000<br>+1.7 [-0.7, +4.3] | **79.5** [72.8, 82.3] N 1000<br>+17.3 [+12.8, +25.2] | **4.0** [2.5, 11.2] N 1000<br>+92.8 [+85.7, +96.3] |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **97.3** [96.7, 97.8] N 1000 | **97.5** [97.2, 97.8] N 1000<br>-0.2 [-1.7, +1.0] | **95.7** [91.8, 96.8] N 1000<br>+1.7 [-0.8, +6.0] | **68.5** [52.5, 74.3] N 1000<br>+28.8 [+20.8, +44.3] | **98.0** [97.2, 98.5] N 1000<br>-0.7 [-2.2, +0.5] | **97.0** [96.5, 98.2] N 1000<br>+0.3 [-2.0, +2.0] | **96.5** [93.7, 97.8] N 1000<br>+0.8 [-1.3, +4.3] | **71.0** [52.7, 74.3] N 1000<br>+26.3 [+20.3, +44.5] | **97.7** [97.2, 98.3] N 1000<br>-0.3 [-2.0, +1.2] | **97.2** [96.7, 97.8] N 1000<br>+0.2 [-1.8, +1.7] | **93.7** [92.0, 94.7] N 1000<br>+3.7 [+1.0, +6.8] | **66.7** [50.2, 74.5] N 1000<br>+30.7 [+21.2, +46.5] |
| `pid_track_descend` | 80.0 [73.9, 85.0] 160/200 | 77.5 [71.2, 82.7] 155/200<br>+2.5 [-4.0, +9.0] | 64.0 [57.1, 70.3] 128/200<br>+16.0 [+8.5, +23.5] | 24.5 [19.1, 30.9] 49/200<br>+55.5 [+47.5, +63.5] | 78.5 [72.3, 83.6] 157/200<br>+1.5 [-4.5, +7.5] | 78.5 [72.3, 83.6] 157/200<br>+1.5 [-5.0, +8.0] | 66.5 [59.7, 72.7] 133/200<br>+13.5 [+6.0, +21.0] | 23.5 [18.2, 29.8] 47/200<br>+56.5 [+48.5, +64.5] | 80.0 [73.9, 85.0] 160/200<br>+0.0 [-5.0, +5.0] | 80.0 [73.9, 85.0] 160/200<br>+0.0 [-6.0, +6.5] | 66.5 [59.7, 72.7] 133/200<br>+13.5 [+6.5, +20.5] | 21.5 [16.4, 27.7] 43/200<br>+58.5 [+50.0, +66.5] |
| `pid_feedforward` | 90.5 [85.6, 93.8] 181/200 | 86.0 [80.5, 90.1] 172/200<br>+4.5 [-1.5, +10.5] | 59.5 [52.6, 66.1] 119/200<br>+31.0 [+23.5, +38.5] | 0.5 [0.1, 2.8] 1/200<br>+90.0 [+85.5, +94.0] | 93.5 [89.2, 96.2] 187/200<br>-3.0 [-7.5, +1.5] | 88.0 [82.8, 91.8] 176/200<br>+2.5 [-3.0, +8.0] | 54.0 [47.1, 60.8] 108/200<br>+36.5 [+29.0, +44.0] | 0.5 [0.1, 2.8] 1/200<br>+90.0 [+85.5, +94.0] | 88.5 [83.3, 92.2] 177/200<br>+2.0 [-3.0, +7.0] | 86.5 [81.1, 90.6] 173/200<br>+4.0 [-2.0, +10.0] | 48.5 [41.7, 55.4] 97/200<br>+42.0 [+34.5, +49.5] | 0.0 [0.0, 1.9] 0/200<br>+90.5 [+86.5, +94.5] |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 85.0 [79.4, 89.3] 170/200 | 79.0 [72.8, 84.1] 158/200<br>+6.0 [+0.0, +12.0] | 59.0 [52.1, 65.6] 118/200<br>+26.0 [+18.0, +34.0] | 6.0 [3.5, 10.2] 12/200<br>+79.0 [+72.5, +85.0] | 81.0 [75.0, 85.8] 162/200<br>+4.0 [-2.5, +10.5] | 77.5 [71.2, 82.7] 155/200<br>+7.5 [+0.5, +14.5] | 50.0 [43.1, 56.9] 100/200<br>+35.0 [+27.0, +42.5] | 6.5 [3.8, 10.8] 13/200<br>+78.5 [+72.5, +84.0] | 78.0 [71.8, 83.2] 156/200<br>+7.0 [+0.5, +13.5] | 73.5 [67.0, 79.1] 147/200<br>+11.5 [+4.5, +18.5] | 43.5 [36.8, 50.4] 87/200<br>+41.5 [+33.5, +49.5] | 8.5 [5.4, 13.2] 17/200<br>+76.5 [+70.5, +82.5] |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 88.0 [82.8, 91.8] 176/200 | 88.0 [82.8, 91.8] 176/200<br>+0.0 [-5.0, +5.0] | 84.0 [78.3, 88.4] 168/200<br>+4.0 [-2.0, +10.0] | 39.0 [32.5, 45.9] 78/200<br>+49.0 [+41.0, +57.0] | 90.5 [85.6, 93.8] 181/200<br>-2.5 [-8.0, +2.5] | 91.0 [86.2, 94.2] 182/200<br>-3.0 [-8.0, +2.0] | 80.0 [73.9, 85.0] 160/200<br>+8.0 [+2.0, +14.5] | 42.0 [35.4, 48.9] 84/200<br>+46.0 [+38.0, +54.0] | 91.5 [86.8, 94.6] 183/200<br>-3.5 [-8.0, +1.0] | 91.5 [86.8, 94.6] 183/200<br>-3.5 [-8.5, +1.0] | 79.5 [73.4, 84.5] 159/200<br>+8.5 [+2.5, +14.5] | 45.0 [38.3, 51.9] 90/200<br>+43.0 [+34.5, +51.0] |
| `gated` | 63.0 [56.1, 69.4] 126/200 | 51.0 [44.1, 57.8] 102/200<br>+12.0 [+7.5, +16.5] | 2.0 [0.8, 5.0] 4/200<br>+61.0 [+54.0, +67.5] | 0.0 [0.0, 1.9] 0/200<br>+63.0 [+56.5, +69.5] | 63.0 [56.1, 69.4] 126/200<br>+0.0 [-3.0, +2.5] | 51.5 [44.6, 58.3] 103/200<br>+11.5 [+7.0, +16.5] | 1.5 [0.5, 4.3] 3/200<br>+61.5 [+55.0, +68.0] | 0.0 [0.0, 1.9] 0/200<br>+63.0 [+56.5, +69.5] | 60.5 [53.6, 67.0] 121/200<br>+2.5 [-0.5, +5.5] | 48.5 [41.7, 55.4] 97/200<br>+14.5 [+10.0, +19.5] | 1.5 [0.5, 4.3] 3/200<br>+61.5 [+54.5, +68.0] | 0.0 [0.0, 1.9] 0/200<br>+63.0 [+56.5, +69.5] |
| `oracle_gated` — commit-timing oracle (privileged) | 64.5 [57.7, 70.8] 129/200 | 61.5 [54.6, 68.0] 123/200<br>+3.0 [-0.0, +6.5] | 9.5 [6.2, 14.4] 19/200<br>+55.0 [+48.0, +61.5] | 0.0 [0.0, 1.9] 0/200<br>+64.5 [+58.0, +71.0] | 64.0 [57.1, 70.3] 128/200<br>+0.5 [-1.0, +2.0] | 62.0 [55.1, 68.4] 124/200<br>+2.5 [-1.0, +6.0] | 7.5 [4.6, 12.0] 15/200<br>+57.0 [+50.0, +64.0] | 0.0 [0.0, 1.9] 0/200<br>+64.5 [+58.0, +71.0] | 64.0 [57.1, 70.3] 128/200<br>+0.5 [-1.5, +2.5] | 60.5 [53.6, 67.0] 121/200<br>+4.0 [+1.0, +7.5] | 5.0 [2.7, 9.0] 10/200<br>+59.5 [+53.0, +66.0] | 0.0 [0.0, 1.9] 0/200<br>+64.5 [+58.0, +71.0] |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 0 (0.0) | 998 (99.8) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 22 (2.2) | 5 (0.5) | 973 (97.3) | 0 (0.0) | 10 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 1 (0.5) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 9 (4.5) | 183 (91.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 28 (14.0) | 18 (9.0) | 154 (77.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 31 (15.5) | 23 (11.5) | 146 (73.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 2 (1.0) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 14 (7.0) | 178 (89.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 193 (96.5) | 7 (3.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 167 (83.5) | 33 (16.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 113 (56.5) | 87 (43.5) | 0 |

#### `id`, lambda 1/40: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 2 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 17 (1.7) | 5 (0.5) | 978 (97.8) | 0 (0.0) | 20 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 18 (9.0) | 175 (87.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 21 (10.5) | 173 (86.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 7 (3.5) | 186 (93.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 184 (92.0) | 16 (8.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 146 (73.0) | 54 (27.0) | 0 |

### `unseen_seastate`: success at lambda 1/15, 1/25, 1/40

| method | SS | 1/15 | 1/25 | 1/40 | 1/25 − 1/15 | 1/25 − 1/40 |
|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | **97.0** [96.7, 98.3]; seeds 96.5–99.0; N 5×200=1000 | **97.3** [96.0, 98.2]; seeds 95.5–98.5; N 5×200=1000 | **96.8** [96.5, 97.3]; seeds 96.5–97.5; N 5×200=1000 | +0.3 [-3.5, +3.0] | +0.5 [-2.5, +3.2] |
| `pid_track_descend` | SS6 | 74.0 [67.5, 79.6] 148/200 | 81.0 [75.0, 85.8] 162/200 | 83.5 [77.7, 88.0] 167/200 | +7.0 [-1.0, +15.0] | -2.5 [-10.0, +5.0] |
| `pid_feedforward` | SS6 | 90.5 [85.6, 93.8] 181/200 | 90.0 [85.1, 93.4] 180/200 | 93.0 [88.6, 95.8] 186/200 | -0.5 [-6.5, +5.5] | -3.0 [-8.5, +2.5] |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 60.5 [53.6, 67.0] 121/200 | 65.0 [58.2, 71.3] 130/200 | 69.5 [62.8, 75.5] 139/200 | +4.5 [-5.0, +14.0] | -4.5 [-13.5, +4.5] |

#### `unseen_seastate`, lambda 1/15: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 6 (0.6) | 973 (97.3) | 0 (0.0) | 12 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 30 (15.0) | 22 (11.0) | 148 (74.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 10 (5.0) | 181 (90.5) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 121 (60.5) | 79 (39.5) | 0 |

#### `unseen_seastate`, lambda 1/40: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 8 (0.8) | 969 (96.9) | 0 (0.0) | 28 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 24 (12.0) | 167 (83.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 8 (4.0) | 186 (93.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 139 (69.5) | 61 (30.5) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 1 (0.1) | 5 (0.5) | 108 (10.8) | 20 (2.0) | 866 (86.6) | 0 (0.0) | 8 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 5 (2.5) | 191 (95.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 32 (16.0) | 31 (15.5) | 137 (68.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 26 (13.0) | 30 (15.0) | 144 (72.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 191 (95.5) | 9 (4.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 76 (38.0) | 124 (62.0) | 0 |

#### `unseen_heading`, lambda 1/40: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 3 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 47 (4.7) | 20 (2.0) | 932 (93.2) | 0 (0.0) | 66 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 20 (10.0) | 32 (16.0) | 148 (74.0) | 0 (0.0) | 2 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 15 (7.5) | 24 (12.0) | 161 (80.5) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 92 (46.0) | 108 (54.0) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 4 (0.4) | 0 (0.0) | 7 (0.7) | 1 (0.1) | 988 (98.8) | 0 (0.0) | 0 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 11 (5.5) | 186 (93.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 23 (11.5) | 14 (7.0) | 163 (81.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 36 (18.0) | 15 (7.5) | 149 (74.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 3 (1.5) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 193 (96.5) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 185 (92.5) | 15 (7.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 163 (81.5) | 37 (18.5) | 0 |

#### `unseen_vessel`, lambda 1/40: outcome breakdown

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 1 (0.1) | 998 (99.8) | 0 (0.0) | 8 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 10 (5.0) | 185 (92.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 22 (11.0) | 170 (85.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 1 (0.5) | 197 (98.5) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 199 (99.5) | 1 (0.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 195 (97.5) | 5 (2.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 180 (90.0) | 20 (10.0) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5/mss:mss | 1000 | 0 (0.0) | 1 (0.1) | 20 (2.0) | 0 (0.0) | 979 (97.9) | 0 (0.0) | 15 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 9 (4.5) | 183 (91.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `gated` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 25 (2.5) | 1 (0.1) | 974 (97.4) | 0 (0.0) | 10 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 8 (4.0) | 192 (96.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `gated` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5/mss:mss | 1000 | 0 (0.0) | 1 (0.1) | 21 (2.1) | 0 (0.0) | 978 (97.8) | 0 (0.0) | 10 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5/mss:mss | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `gated` | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5/mss:mss | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |

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

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5/mss:corpus | 1000 | 0 (0.0) | 1 (0.1) | 25 (2.5) | 0 (0.0) | 974 (97.4) | 0 (0.0) | 8 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5/mss:corpus | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `gated` | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5/mss:corpus | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |

### `unseen_vessel` SS5, headings 180° and 135° only (matrix, aft)

Per seed: success [Wilson 95 % CI] k/N (a baseline is one deterministic run, seed `–`). Then the six-class breakdown pooled over seeds, with the tunnelled successes. These are different realizations from the MSS lists; descriptive, nothing is scored.

| method | seed | success [Wilson 95 % CI] k/N |
|---|---|---|
| `ppo` (learned, PPO) | 0 | 100.0 [96.2, 100.0] 97/97 |
| `ppo` (learned, PPO) | 1 | 100.0 [96.2, 100.0] 97/97 |
| `ppo` (learned, PPO) | 2 | 100.0 [96.2, 100.0] 97/97 |
| `ppo` (learned, PPO) | 3 | 99.0 [94.4, 99.8] 96/97 |
| `ppo` (learned, PPO) | 4 | 100.0 [96.2, 100.0] 97/97 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0 | 95.9 [89.9, 98.4] 93/97 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 1 | 100.0 [96.2, 100.0] 97/97 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 2 | 100.0 [96.2, 100.0] 97/97 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 3 | 94.8 [88.5, 97.8] 92/97 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 4 | 99.0 [94.4, 99.8] 96/97 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo` (learned, residual on pid_feedforward) | 1 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo` (learned, residual on pid_feedforward) | 2 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo` (learned, residual on pid_feedforward) | 3 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo` (learned, residual on pid_feedforward) | 4 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 1 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 2 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 3 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 4 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 1 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 2 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 3 | 100.0 [96.2, 100.0] 97/97 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 4 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 1 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 2 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 3 | 100.0 [96.2, 100.0] 97/97 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 4 | 100.0 [96.2, 100.0] 97/97 |
| `pid_track_descend` | – | 83.5 [74.9, 89.6] 81/97 |
| `pid_feedforward` | – | 100.0 [96.2, 100.0] 97/97 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | – | 99.0 [94.4, 99.8] 96/97 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | – | 99.0 [94.4, 99.8] 96/97 |
| `gated` | – | 91.8 [84.6, 95.8] 89/97 |
| `oracle_gated` — commit-timing oracle (privileged) | – | 93.8 [87.2, 97.1] 91/97 |

| method | N (seed-episodes) | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | 485 | 0 (0.0) | 0 (0.0) | 1 (0.2) | 0 (0.0) | 484 (99.8) | 0 (0.0) | 0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 485 | 0 (0.0) | 0 (0.0) | 9 (1.9) | 1 (0.2) | 475 (97.9) | 0 (0.0) | 11 |
| `residual_ppo` (learned, residual on pid_feedforward) | 485 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 485 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 485 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 485 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 485 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 485 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 485 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 485 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | 97 | 0 (0.0) | 0 (0.0) | 12 (12.4) | 4 (4.1) | 81 (83.5) | 0 (0.0) | 0 |
| `pid_feedforward` | 97 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 97 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 97 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (1.0) | 96 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 97 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (1.0) | 96 (99.0) | 0 (0.0) | 0 |
| `gated` | 97 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 89 (91.8) | 8 (8.2) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | 97 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 91 (93.8) | 6 (6.2) | 0 |

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
| H3 | primary: unseen_vessel SS5 half-rule | scored | unseen_vessel SS5 vs id SS5, aft | -0.7 [-2.3, +1.4] | r(unseen_vessel) ≤ 0.5 × r(id) | **not applicable — no id gain to shrink** | P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict); P7-D4 (m6 wording) | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed, regimes-overlap |
| H3 | primary: unseen_vessel SS6 half-rule | scored | unseen_vessel SS6 vs id SS6, aft | -0.4 [-3.3, +3.2] | r(unseen_vessel) ≤ 0.5 × r(id) | **not applicable — no id gain to shrink** | P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict); P7-D4 (m6 wording) | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed, regimes-overlap |
| H3 | secondary: id SS5 relative p95 | secondary (pre-registered, reported beside) | id SS5, aft | -0.8 [-2.4, +0.9] | ≥ +10.0 | **not supported** | P3-D1 §8 H3; P7-D1a #2 (own verdict), #5, #6 | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed |
| H3 | secondary: id SS6 relative p95 | secondary (pre-registered, reported beside) | id SS6, aft | +1.9 [+0.4, +3.8] | ≥ +10.0 | **inconclusive** | P3-D1 §8 H3; P7-D1a #2 (own verdict), #5, #6 | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed |
| H3 | secondary: unseen_vessel SS5 half-rule | secondary (pre-registered, reported beside) | unseen_vessel SS5 vs id SS5, aft | -2.0 [-3.6, -0.3] | r(unseen_vessel) ≤ 0.5 × r(id) | **not applicable — no id gain to shrink** | P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict); P7-D4 (m6 wording) | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed, regimes-overlap |
| H3 | secondary: unseen_vessel SS6 half-rule | secondary (pre-registered, reported beside) | unseen_vessel SS6 vs id SS6, aft | +0.0 [-1.9, +2.6] | r(unseen_vessel) ≤ 0.5 × r(id) | **not scored (no supported id gain)** | P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict); P7-D4 (m6 wording) | P6-D1-forecast, closing-speed-7pct, tunnelling-any-substep, no-multiplicity, shared-bootstrap-seed, regimes-overlap |
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

Learned methods: one row per training seed. Baselines (carried from `results/e01` and `results/e01_lowvz_cut`): one deterministic run, seed `–`.

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
| `pid_track_descend` | – | 100.0 [98.1, 100.0] 200/200 | 96.0 [92.3, 98.0] 192/200 | 85.0 [79.4, 89.3] 170/200 | 80.0 [73.9, 85.0] 160/200 |
| `pid_feedforward` | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 90.5 [85.6, 93.8] 181/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | – | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 | 95.5 [91.7, 97.6] 191/200 | 85.0 [79.4, 89.3] 170/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | – | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 98.0 [95.0, 99.2] 196/200 | 88.0 [82.8, 91.8] 176/200 |
| `gated` | – | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 89.5 [84.5, 93.0] 179/200 | 63.0 [56.1, 69.4] 126/200 |
| `oracle_gated` — commit-timing oracle (privileged) | – | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 89.0 [83.9, 92.6] 178/200 | 64.5 [57.7, 70.8] 129/200 |

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
| `pid_track_descend` | – | 81.0 [75.0, 85.8] 162/200 |
| `pid_feedforward` | – | 90.0 [85.1, 93.4] 180/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | – | 86.0 [80.5, 90.1] 172/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | – | 89.5 [84.5, 93.0] 179/200 |
| `gated` | – | 64.5 [57.7, 70.8] 129/200 |
| `oracle_gated` — commit-timing oracle (privileged) | – | 65.0 [58.2, 71.3] 130/200 |

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
| `pid_track_descend` | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 | 72.5 [65.9, 78.2] 145/200 |
| `pid_feedforward` | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 77.0 [70.7, 82.3] 154/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | – | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 96.5 [93.0, 98.3] 193/200 | 70.0 [63.3, 75.9] 140/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | – | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 97.0 [93.6, 98.6] 194/200 | 83.0 [77.2, 87.6] 166/200 |
| `gated` | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 | 42.0 [35.4, 48.9] 84/200 |
| `oracle_gated` — commit-timing oracle (privileged) | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 46.5 [39.7, 53.4] 93/200 |

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
| `pid_track_descend` | – | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 | 90.0 [85.1, 93.4] 180/200 | 81.5 [75.5, 86.3] 163/200 |
| `pid_feedforward` | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 98.5 [95.7, 99.5] 197/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 95.0 [91.0, 97.3] 190/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 97.0 [93.6, 98.6] 194/200 |
| `gated` | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 93.5 [89.2, 96.2] 187/200 | 90.0 [85.1, 93.4] 180/200 |
| `oracle_gated` — commit-timing oracle (privileged) | – | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 95.0 [91.0, 97.3] 190/200 | 89.0 [83.9, 92.6] 178/200 |

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
| `pid_track_descend` | – | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward` | – | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | – | 100.0 [98.1, 100.0] 200/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | – | 100.0 [98.1, 100.0] 200/200 |
| `gated` | – | 100.0 [98.1, 100.0] 200/200 |
| `oracle_gated` — commit-timing oracle (privileged) | – | 100.0 [98.1, 100.0] 200/200 |

## Appendix B. Perception stand-in (P7-D4): success, p95, outcome breakdown and tunnelled successes per condition

### `noise/sigma1cm_lat0step` (σp 1 cm, σv 0.05 m/s, latency 0 step = 0.0 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.5** [97.3, 99.0]; seeds 97.0–99.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **92.5** [88.5, 97.5]; seeds 87.5–98.5; N 5×200=1000 | **89.7** [86.0, 93.3]; seeds 84.5–94.5; N 5×200=1000 | **77.8** [73.0, 83.3]; seeds 72.5–84.0; N 5×200=1000 | **66.8** [58.7, 73.3]; seeds 55.0–76.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **98.8** [98.2, 99.0]; seeds 98.0–99.0; N 5×200=1000 | **95.0** [95.0, 95.7]; seeds 95.0–96.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **97.7** [97.5, 98.3]; seeds 97.5–98.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **98.8** [98.5, 99.5]; seeds 98.5–99.5; N 5×200=1000 | **95.2** [93.8, 96.3]; seeds 93.5–96.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **97.5** [97.2, 97.8]; seeds 97.0–98.0; N 5×200=1000 |
| `pid_track_descend` | 99.0 [96.4, 99.7] 198/200 | 95.5 [91.7, 97.6] 191/200 | 83.0 [77.2, 87.6] 166/200 | 77.5 [71.2, 82.7] 155/200 |
| `pid_feedforward` | 98.5 [95.7, 99.5] 197/200 | 97.5 [94.3, 98.9] 195/200 | 93.5 [89.2, 96.2] 187/200 | 86.0 [80.5, 90.1] 172/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 97.0 [93.6, 98.6] 194/200 | 96.5 [93.0, 98.3] 193/200 | 88.5 [83.3, 92.2] 177/200 | 79.0 [72.8, 84.1] 158/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 99.0 [96.4, 99.7] 198/200 | 98.5 [95.7, 99.5] 197/200 | 95.5 [91.7, 97.6] 191/200 | 88.0 [82.8, 91.8] 176/200 |
| `gated` | 96.5 [93.0, 98.3] 193/200 | 94.5 [90.4, 96.9] 189/200 | 81.0 [75.0, 85.8] 162/200 | 51.0 [44.1, 57.8] 102/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 97.5 [94.3, 98.9] 195/200 | 95.5 [91.7, 97.6] 191/200 | 86.0 [80.5, 90.1] 172/200 | 61.5 [54.6, 68.0] 123/200 |

### `noise/sigma1cm_lat0step` (σp 1 cm, σv 0.05 m/s, latency 0 step = 0.0 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.306 [0.302, 0.315] · 0.0 · 0.0 | 0.319 [0.310, 0.324] · 0.0 · 0.0 | 0.305 [0.299, 0.315] · 0.0 · 0.0 | 0.316 [0.311, 0.325] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.513 [0.450, 0.542] · 0.0 · 0.0 | 0.543 [0.510, 0.571] · 0.0 · 0.0 | 0.596 [0.561, 0.650] · 0.5 · 0.0 | 0.682 [0.644, 0.702] · 2.8 · 0.1 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.308 [0.302, 0.323] · 0.0 · 0.0 | 0.321 [0.314, 0.327] · 0.0 · 0.0 | 0.318 [0.314, 0.320] · 0.0 · 0.0 | 0.321 [0.318, 0.323] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.284 [0.279, 0.288] · 0.0 · 0.0 | 0.292 [0.290, 0.295] · 0.0 · 0.0 | 0.292 [0.284, 0.296] · 0.0 · 0.0 | 0.306 [0.298, 0.317] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.310 [0.302, 0.313] · 0.0 · 0.0 | 0.309 [0.303, 0.315] · 0.0 · 0.0 | 0.310 [0.305, 0.316] · 0.1 · 0.0 | 0.314 [0.305, 0.317] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.314 [0.310, 0.319] · 0.0 · 0.0 | 0.323 [0.316, 0.331] · 0.0 · 0.0 | 0.317 [0.313, 0.327] · 0.0 · 0.0 | 0.328 [0.324, 0.332] · 0.0 · 0.0 |
| `pid_track_descend` | 0.305 · 0.0 · 0.0 | 0.418 · 0.0 · 0.0 | 0.505 · 0.0 · 0.0 | 0.501 · 0.0 · 0.0 |
| `pid_feedforward` | 0.251 · 0.0 · 0.0 | 0.250 · 0.0 · 0.0 | 0.301 · 0.0 · 0.0 | 0.275 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.160 · 0.0 · 0.0 | 0.182 · 0.0 · 0.0 | 0.189 · 0.0 · 0.0 | 0.191 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.160 · 0.0 · 0.0 | 0.182 · 0.0 · 0.0 | 0.189 · 0.0 · 0.0 | 0.191 · 0.0 · 0.0 |
| `gated` | 0.251 · 0.0 · 0.0 | 0.249 · 0.0 · 3.5 | 0.270 · 0.0 · 15.0 | 0.288 · 0.0 · 44.0 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.239 · 0.0 · 0.0 | 0.255 · 0.0 · 1.5 | 0.267 · 0.0 · 12.5 | 0.262 · 0.0 · 37.0 |

### `noise/sigma1cm_lat0step` (σp 1 cm, σv 0.05 m/s, latency 0 step = 0.0 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 1 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 15 (1.5) | 2 (0.2) | 983 (98.3) | 0 (0.0) | 5 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 8 (0.8) | 63 (6.3) | 2 (0.2) | 927 (92.7) | 0 (0.0) | 22 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 7 (0.7) | 93 (9.3) | 4 (0.4) | 896 (89.6) | 0 (0.0) | 26 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 5 (0.5) | 27 (2.7) | 180 (18.0) | 8 (0.8) | 780 (78.0) | 0 (0.0) | 59 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 28 (2.8) | 63 (6.3) | 232 (23.2) | 13 (1.3) | 663 (66.3) | 1 (0.1) | 91 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 11 (1.1) | 2 (0.2) | 987 (98.7) | 0 (0.0) | 2 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 38 (3.8) | 10 (1.0) | 952 (95.2) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 18 (1.8) | 4 (0.4) | 978 (97.8) | 0 (0.0) | 9 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 1 (0.1) | 1 (0.1) | 7 (0.7) | 2 (0.2) | 989 (98.9) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 38 (3.8) | 11 (1.1) | 951 (95.1) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 24 (2.4) | 1 (0.1) | 975 (97.5) | 0 (0.0) | 7 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 6 (3.0) | 191 (95.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 12 (6.0) | 22 (11.0) | 166 (83.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 14 (7.0) | 31 (15.5) | 155 (77.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 10 (5.0) | 187 (93.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 21 (10.5) | 172 (86.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 23 (11.5) | 177 (88.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 35 (17.5) | 158 (79.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 191 (95.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 17 (8.5) | 176 (88.0) | 0 (0.0) | 2 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 189 (94.5) | 7 (3.5) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 8 (4.0) | 162 (81.0) | 30 (15.0) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 102 (51.0) | 88 (44.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 191 (95.5) | 3 (1.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 172 (86.0) | 25 (12.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 123 (61.5) | 74 (37.0) | 0 |

### `noise/sigma2cm_lat0step` (σp 2 cm, σv 0.1 m/s, latency 0 step = 0.0 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **99.2** [98.7, 99.8]; seeds 98.5–100.0; N 5×200=1000 | **99.5** [99.2, 99.8]; seeds 99.0–100.0; N 5×200=1000 | **98.2** [96.8, 99.3]; seeds 96.5–99.5; N 5×200=1000 | **96.2** [94.5, 97.7]; seeds 94.0–98.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **64.0** [57.2, 81.3]; seeds 56.5–85.5; N 5×200=1000 | **59.8** [52.0, 76.3]; seeds 52.0–81.5; N 5×200=1000 | **50.2** [44.7, 65.7]; seeds 43.5–72.5; N 5×200=1000 | **40.3** [33.8, 51.3]; seeds 33.0–54.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **77.0** [75.5, 81.2]; seeds 75.0–83.0; N 5×200=1000 | **81.2** [75.2, 84.3]; seeds 73.5–85.0; N 5×200=1000 | **79.7** [73.0, 85.2]; seeds 71.5–85.5; N 5×200=1000 | **74.5** [72.0, 78.3]; seeds 71.0–80.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **96.7** [95.8, 97.3]; seeds 95.5–97.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **87.5** [81.5, 92.0]; seeds 79.0–93.5; N 5×200=1000 | **84.5** [79.7, 92.0]; seeds 79.0–93.5; N 5×200=1000 | **82.2** [76.5, 86.7]; seeds 75.5–88.0; N 5×200=1000 | **75.7** [69.7, 83.0]; seeds 67.0–86.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **99.0** [98.2, 99.5]; seeds 98.0–99.5; N 5×200=1000 | **98.8** [98.0, 99.5]; seeds 98.0–99.5; N 5×200=1000 | **99.2** [96.7, 100.0]; seeds 96.0–100.0; N 5×200=1000 | **95.7** [91.8, 96.8]; seeds 90.5–97.0; N 5×200=1000 |
| `pid_track_descend` | 86.0 [80.5, 90.1] 172/200 | 79.0 [72.8, 84.1] 158/200 | 71.0 [64.4, 76.8] 142/200 | 64.0 [57.1, 70.3] 128/200 |
| `pid_feedforward` | 60.0 [53.1, 66.5] 120/200 | 66.0 [59.2, 72.2] 132/200 | 56.0 [49.1, 62.7] 112/200 | 59.5 [52.6, 66.1] 119/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 66.5 [59.7, 72.7] 133/200 | 65.0 [58.2, 71.3] 130/200 | 58.0 [51.1, 64.6] 116/200 | 59.0 [52.1, 65.6] 118/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 90.5 [85.6, 93.8] 181/200 | 85.5 [80.0, 89.7] 171/200 | 83.0 [77.2, 87.6] 166/200 | 84.0 [78.3, 88.4] 168/200 |
| `gated` | 16.5 [12.0, 22.3] 33/200 | 15.5 [11.1, 21.2] 31/200 | 3.0 [1.4, 6.4] 6/200 | 2.0 [0.8, 5.0] 4/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 30.5 [24.5, 37.2] 61/200 | 27.0 [21.3, 33.5] 54/200 | 16.5 [12.0, 22.3] 33/200 | 9.5 [6.2, 14.4] 19/200 |

### `noise/sigma2cm_lat0step` (σp 2 cm, σv 0.1 m/s, latency 0 step = 0.0 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.360 [0.352, 0.361] · 0.0 · 0.0 | 0.375 [0.359, 0.385] · 0.0 · 0.0 | 0.350 [0.338, 0.354] · 0.0 · 0.0 | 0.364 [0.354, 0.381] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.697 [0.563, 0.861] · 0.7 · 0.0 | 0.717 [0.571, 0.834] · 0.9 · 0.0 | 0.716 [0.619, 0.770] · 1.6 · 0.1 | 0.814 [0.723, 0.878] · 4.1 · 0.2 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.369 [0.359, 0.386] · 0.2 · 0.0 | 0.372 [0.353, 0.389] · 0.2 · 0.0 | 0.375 [0.371, 0.385] · 0.1 · 0.0 | 0.369 [0.353, 0.384] · 0.2 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.311 [0.303, 0.323] · 0.0 · 0.0 | 0.324 [0.321, 0.325] · 0.0 · 0.0 | 0.309 [0.305, 0.320] · 0.0 · 0.0 | 0.328 [0.324, 0.337] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.355 [0.341, 0.357] · 0.0 · 0.0 | 0.351 [0.344, 0.358] · 0.4 · 0.0 | 0.359 [0.350, 0.373] · 0.1 · 0.0 | 0.356 [0.352, 0.359] · 0.3 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.370 [0.362, 0.383] · 0.0 · 0.0 | 0.381 [0.369, 0.385] · 0.0 · 0.0 | 0.368 [0.352, 0.385] · 0.0 · 0.0 | 0.379 [0.368, 0.386] · 0.0 · 0.0 |
| `pid_track_descend` | 0.318 · 0.0 · 0.0 | 0.395 · 0.0 · 0.0 | 0.511 · 0.0 · 0.0 | 0.515 · 0.0 · 0.0 |
| `pid_feedforward` | 0.302 · 0.0 · 0.0 | 0.305 · 0.0 · 0.0 | 0.310 · 0.0 · 0.0 | 0.338 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.214 · 0.0 · 0.0 | 0.223 · 0.0 · 0.0 | 0.230 · 0.0 · 0.0 | 0.217 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.214 · 0.0 · 0.0 | 0.223 · 0.0 · 0.0 | 0.230 · 0.0 · 0.0 | 0.217 · 0.0 · 0.0 |
| `gated` | 0.320 · 0.0 · 68.0 | 0.286 · 0.0 · 73.5 | 0.308 · 0.5 · 89.0 | 0.323 · 0.0 · 96.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.315 · 0.0 · 35.5 | 0.314 · 0.0 · 47.5 | 0.292 · 0.5 · 64.0 | 0.263 · 0.0 · 85.0 |

### `noise/sigma2cm_lat0step` (σp 2 cm, σv 0.1 m/s, latency 0 step = 0.0 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 3 (0.3) | 992 (99.2) | 0 (0.0) | 2 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (0.5) | 995 (99.5) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 14 (1.4) | 981 (98.1) | 0 (0.0) | 4 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 2 (0.2) | 20 (2.0) | 17 (1.7) | 961 (96.1) | 0 (0.0) | 10 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 7 (0.7) | 47 (4.7) | 271 (27.1) | 7 (0.7) | 668 (66.8) | 0 (0.0) | 91 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 9 (0.9) | 58 (5.8) | 296 (29.6) | 11 (1.1) | 626 (62.6) | 0 (0.0) | 115 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 16 (1.6) | 95 (9.5) | 345 (34.5) | 10 (1.0) | 533 (53.3) | 1 (0.1) | 122 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 41 (4.1) | 129 (12.9) | 389 (38.9) | 22 (2.2) | 417 (41.7) | 2 (0.2) | 96 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 2 (0.2) | 12 (1.2) | 60 (6.0) | 148 (14.8) | 778 (77.8) | 0 (0.0) | 2 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 2 (0.2) | 9 (0.9) | 49 (4.9) | 136 (13.6) | 804 (80.4) | 0 (0.0) | 8 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 1 (0.1) | 10 (1.0) | 66 (6.6) | 131 (13.1) | 792 (79.2) | 0 (0.0) | 7 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 2 (0.2) | 3 (0.3) | 108 (10.8) | 138 (13.8) | 749 (74.9) | 0 (0.0) | 6 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 2 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 28 (2.8) | 6 (0.6) | 966 (96.6) | 0 (0.0) | 4 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 6 (0.6) | 31 (3.1) | 93 (9.3) | 870 (87.0) | 0 (0.0) | 4 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 4 (0.4) | 6 (0.6) | 34 (3.4) | 104 (10.4) | 852 (85.2) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 1 (0.1) | 9 (0.9) | 50 (5.0) | 120 (12.0) | 820 (82.0) | 0 (0.0) | 1 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 3 (0.3) | 6 (0.6) | 110 (11.0) | 120 (12.0) | 761 (76.1) | 0 (0.0) | 6 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 1 (0.1) | 3 (0.3) | 7 (0.7) | 989 (98.9) | 0 (0.0) | 1 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 2 (0.2) | 4 (0.4) | 6 (0.6) | 988 (98.8) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 7 (0.7) | 6 (0.6) | 987 (98.7) | 0 (0.0) | 3 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 38 (3.8) | 13 (1.3) | 949 (94.9) | 0 (0.0) | 9 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 28 (14.0) | 172 (86.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 39 (19.5) | 158 (79.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 11 (5.5) | 47 (23.5) | 142 (71.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 20 (10.0) | 52 (26.0) | 128 (64.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 1 (0.5) | 5 (2.5) | 74 (37.0) | 120 (60.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 1 (0.5) | 5 (2.5) | 62 (31.0) | 132 (66.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 83 (41.5) | 112 (56.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 13 (6.5) | 68 (34.0) | 119 (59.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 66 (33.0) | 133 (66.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 1 (0.5) | 2 (1.0) | 67 (33.5) | 130 (65.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 2 (1.0) | 4 (2.0) | 78 (39.0) | 116 (58.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 13 (6.5) | 69 (34.5) | 118 (59.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 18 (9.0) | 181 (90.5) | 0 (0.0) | 3 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 1 (0.5) | 2 (1.0) | 26 (13.0) | 171 (85.5) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 2 (1.0) | 4 (2.0) | 28 (14.0) | 166 (83.0) | 0 (0.0) | 3 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 13 (6.5) | 19 (9.5) | 168 (84.0) | 0 (0.0) | 11 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 31 (15.5) | 33 (16.5) | 136 (68.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 20 (10.0) | 31 (15.5) | 147 (73.5) | 0 |
| `gated` | SS5 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 15 (7.5) | 6 (3.0) | 178 (89.0) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 4 (2.0) | 193 (96.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 66 (33.0) | 61 (30.5) | 71 (35.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 50 (25.0) | 54 (27.0) | 95 (47.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 38 (19.0) | 33 (16.5) | 128 (64.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 10 (5.0) | 19 (9.5) | 170 (85.0) | 0 |

### `noise/sigma4cm_lat0step` (σp 4 cm, σv 0.2 m/s, latency 0 step = 0.0 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **79.8** [72.3, 89.7]; seeds 70.5–91.5; N 5×200=1000 | **81.5** [75.3, 89.3]; seeds 75.0–91.0; N 5×200=1000 | **79.2** [72.7, 84.7]; seeds 71.5–85.5; N 5×200=1000 | **75.7** [69.3, 81.5]; seeds 66.5–84.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **19.2** [11.7, 32.7]; seeds 11.0–36.5; N 5×200=1000 | **16.5** [13.5, 30.0]; seeds 13.0–35.0; N 5×200=1000 | **15.8** [14.3, 26.0]; seeds 14.0–30.5; N 5×200=1000 | **10.5** [4.2, 17.8]; seeds 2.5–20.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **2.5** [2.2, 4.2]; seeds 2.0–5.0; N 5×200=1000 | **3.0** [0.8, 5.3]; seeds 0.5–6.0; N 5×200=1000 | **2.7** [2.2, 4.3]; seeds 2.0–5.0; N 5×200=1000 | **3.2** [2.5, 6.7]; seeds 2.5–8.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **96.3** [93.7, 98.2]; seeds 93.5–98.5; N 5×200=1000 | **95.2** [91.8, 98.3]; seeds 91.0–98.5; N 5×200=1000 | **95.8** [94.3, 97.2]; seeds 94.0–97.5; N 5×200=1000 | **89.8** [89.2, 92.2]; seeds 89.0–93.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **5.0** [3.0, 9.7]; seeds 3.0–11.0; N 5×200=1000 | **4.3** [2.3, 10.2]; seeds 2.0–12.5; N 5×200=1000 | **3.5** [2.5, 8.5]; seeds 2.5–10.5; N 5×200=1000 | **5.5** [4.0, 11.2]; seeds 4.0–13.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **69.7** [58.3, 80.8]; seeds 58.0–82.0; N 5×200=1000 | **73.5** [62.0, 84.0]; seeds 59.0–85.5; N 5×200=1000 | **70.0** [56.2, 76.8]; seeds 53.0–78.0; N 5×200=1000 | **68.5** [52.5, 74.3]; seeds 48.5–74.5; N 5×200=1000 |
| `pid_track_descend` | 21.5 [16.4, 27.7] 43/200 | 22.5 [17.3, 28.8] 45/200 | 22.0 [16.8, 28.2] 44/200 | 24.5 [19.1, 30.9] 49/200 |
| `pid_feedforward` | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.5 [0.1, 2.8] 1/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 7.5 [4.6, 12.0] 15/200 | 5.5 [3.1, 9.6] 11/200 | 9.0 [5.8, 13.8] 18/200 | 6.0 [3.5, 10.2] 12/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 41.5 [34.9, 48.4] 83/200 | 42.5 [35.9, 49.4] 85/200 | 46.5 [39.7, 53.4] 93/200 | 39.0 [32.5, 45.9] 78/200 |
| `gated` | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 |

### `noise/sigma4cm_lat0step` (σp 4 cm, σv 0.2 m/s, latency 0 step = 0.0 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.427 [0.402, 0.441] · 0.0 · 0.0 | 0.432 [0.409, 0.453] · 0.0 · 0.0 | 0.425 [0.407, 0.443] · 0.0 · 0.0 | 0.443 [0.438, 0.468] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.873 [0.737, 0.940] · 9.6 · 0.4 | 0.884 [0.771, 0.942] · 11.3 · 0.5 | 0.923 [0.780, 1.006] · 16.3 · 0.9 | 0.931 [0.773, 1.022] · 22.6 · 1.3 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.590 [0.553, 0.649] · 17.0 · 0.1 | 0.548 [0.486, 0.597] · 15.2 · 0.2 | 0.573 [0.505, 0.620] · 14.0 · 0.0 | 0.558 [0.526, 0.614] · 14.8 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.364 [0.353, 0.375] · 0.0 · 0.0 | 0.365 [0.356, 0.369] · 0.0 · 0.0 | 0.353 [0.339, 0.370] · 0.0 · 0.0 | 0.374 [0.368, 0.391] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.543 [0.506, 0.593] · 9.3 · 0.0 | 0.581 [0.531, 0.668] · 8.5 · 0.0 | 0.595 [0.542, 0.630] · 8.3 · 0.0 | 0.579 [0.535, 0.609] · 10.7 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.444 [0.429, 0.456] · 0.1 · 0.0 | 0.449 [0.427, 0.467] · 0.4 · 0.0 | 0.442 [0.435, 0.457] · 0.3 · 0.0 | 0.455 [0.433, 0.471] · 0.1 · 0.0 |
| `pid_track_descend` | 0.270 · 0.0 · 31.0 | 0.331 · 0.0 · 29.5 | 0.424 · 0.0 · 31.0 | 0.443 · 0.0 · 27.0 |
| `pid_feedforward` | 0.579 · 45.0 · 21.5 | 0.634 · 40.0 · 31.0 | 0.730 · 35.5 · 31.0 | 0.798 · 48.0 · 19.5 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.356 · 4.0 · 7.0 | 0.376 · 6.5 · 5.5 | 0.373 · 2.5 · 5.0 | 0.368 · 2.5 · 7.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.356 · 4.0 · 7.0 | 0.376 · 6.5 · 5.5 | 0.373 · 3.5 · 5.0 | 0.368 · 2.5 · 7.0 |
| `gated` | – · 60.0 · 40.0 | – · 60.5 · 39.5 | – · 65.5 · 34.5 | – · 66.0 · 34.0 |
| `oracle_gated` — commit-timing oracle (privileged) | – · 60.0 · 40.0 | – · 62.5 · 37.5 | – · 67.5 · 32.5 | 0.480 · 62.5 · 37.0 |

### `noise/sigma4cm_lat0step` (σp 4 cm, σv 0.2 m/s, latency 0 step = 0.0 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 44 (4.4) | 76 (7.6) | 77 (7.7) | 803 (80.3) | 0 (0.0) | 31 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 42 (4.2) | 68 (6.8) | 69 (6.9) | 821 (82.1) | 0 (0.0) | 31 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 48 (4.8) | 66 (6.6) | 97 (9.7) | 789 (78.9) | 0 (0.0) | 37 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 46 (4.6) | 128 (12.8) | 71 (7.1) | 755 (75.5) | 0 (0.0) | 58 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 96 (9.6) | 228 (22.8) | 430 (43.0) | 32 (3.2) | 210 (21.0) | 4 (0.4) | 73 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 113 (11.3) | 230 (23.0) | 426 (42.6) | 31 (3.1) | 195 (19.5) | 5 (0.5) | 73 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 163 (16.3) | 240 (24.0) | 371 (37.1) | 33 (3.3) | 184 (18.4) | 9 (0.9) | 82 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 226 (22.6) | 301 (30.1) | 318 (31.8) | 33 (3.3) | 109 (10.9) | 13 (1.3) | 49 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 170 (17.0) | 311 (31.1) | 302 (30.2) | 187 (18.7) | 29 (2.9) | 1 (0.1) | 1 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 152 (15.2) | 311 (31.1) | 310 (31.0) | 194 (19.4) | 31 (3.1) | 2 (0.2) | 1 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 140 (14.0) | 343 (34.3) | 299 (29.9) | 188 (18.8) | 30 (3.0) | 0 (0.0) | 5 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 148 (14.8) | 310 (31.0) | 326 (32.6) | 176 (17.6) | 40 (4.0) | 0 (0.0) | 5 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 2 (0.2) | 11 (1.1) | 25 (2.5) | 962 (96.2) | 0 (0.0) | 6 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 8 (0.8) | 19 (1.9) | 23 (2.3) | 950 (95.0) | 0 (0.0) | 12 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 2 (0.2) | 24 (2.4) | 16 (1.6) | 958 (95.8) | 0 (0.0) | 5 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 6 (0.6) | 66 (6.6) | 25 (2.5) | 903 (90.3) | 0 (0.0) | 24 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 93 (9.3) | 313 (31.3) | 290 (29.0) | 246 (24.6) | 58 (5.8) | 0 (0.0) | 2 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 85 (8.5) | 349 (34.9) | 296 (29.6) | 215 (21.5) | 55 (5.5) | 0 (0.0) | 3 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 83 (8.3) | 312 (31.2) | 313 (31.3) | 245 (24.5) | 47 (4.7) | 0 (0.0) | 2 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 107 (10.7) | 324 (32.4) | 312 (31.2) | 189 (18.9) | 68 (6.8) | 0 (0.0) | 4 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 1 (0.1) | 55 (5.5) | 88 (8.8) | 158 (15.8) | 698 (69.8) | 0 (0.0) | 18 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 4 (0.4) | 49 (4.9) | 91 (9.1) | 126 (12.6) | 730 (73.0) | 0 (0.0) | 24 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 3 (0.3) | 54 (5.4) | 118 (11.8) | 143 (14.3) | 682 (68.2) | 0 (0.0) | 28 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 1 (0.1) | 61 (6.1) | 168 (16.8) | 113 (11.3) | 657 (65.7) | 0 (0.0) | 41 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 95 (47.5) | 43 (21.5) | 62 (31.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 96 (48.0) | 45 (22.5) | 59 (29.5) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 90 (45.0) | 44 (22.0) | 62 (31.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 92 (46.0) | 49 (24.5) | 54 (27.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 90 (45.0) | 34 (17.0) | 20 (10.0) | 13 (6.5) | 0 (0.0) | 43 (21.5) | 0 |
| `pid_feedforward` | SS4 | 200 | 80 (40.0) | 28 (14.0) | 19 (9.5) | 11 (5.5) | 0 (0.0) | 62 (31.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 71 (35.5) | 34 (17.0) | 20 (10.0) | 13 (6.5) | 0 (0.0) | 62 (31.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 96 (48.0) | 37 (18.5) | 15 (7.5) | 12 (6.0) | 1 (0.5) | 39 (19.5) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 8 (4.0) | 30 (15.0) | 46 (23.0) | 87 (43.5) | 15 (7.5) | 14 (7.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 13 (6.5) | 39 (19.5) | 37 (18.5) | 89 (44.5) | 11 (5.5) | 11 (5.5) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 5 (2.5) | 34 (17.0) | 39 (19.5) | 94 (47.0) | 18 (9.0) | 10 (5.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 5 (2.5) | 41 (20.5) | 47 (23.5) | 81 (40.5) | 12 (6.0) | 14 (7.0) | 1 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 8 (4.0) | 30 (15.0) | 46 (23.0) | 19 (9.5) | 83 (41.5) | 14 (7.0) | 26 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 13 (6.5) | 39 (19.5) | 37 (18.5) | 15 (7.5) | 85 (42.5) | 11 (5.5) | 18 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 7 (3.5) | 32 (16.0) | 39 (19.5) | 19 (9.5) | 93 (46.5) | 10 (5.0) | 20 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 5 (2.5) | 41 (20.5) | 47 (23.5) | 15 (7.5) | 78 (39.0) | 14 (7.0) | 14 |
| `gated` | SS3 | 200 | 120 (60.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 80 (40.0) | 0 |
| `gated` | SS4 | 200 | 121 (60.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 79 (39.5) | 0 |
| `gated` | SS5 | 200 | 131 (65.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 69 (34.5) | 0 |
| `gated` | SS6 | 200 | 132 (66.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 68 (34.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 120 (60.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 80 (40.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 125 (62.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 75 (37.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 135 (67.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 65 (32.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 125 (62.5) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 74 (37.0) | 0 |

### `noise/sigma0cm_lat1step` (σp 0 cm, σv 0 m/s, latency 1 step = 33.3 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.3** [97.7, 99.2]; seeds 97.5–99.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **99.8** [98.2, 100.0]; seeds 97.5–100.0; N 5×200=1000 | **98.0** [96.8, 99.2]; seeds 96.5–99.5; N 5×200=1000 | **87.3** [81.8, 88.3]; seeds 79.5–88.5; N 5×200=1000 | **74.0** [62.0, 78.8]; seeds 58.0–79.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **99.3** [98.7, 99.8]; seeds 98.5–100.0; N 5×200=1000 | **96.7** [94.8, 97.5]; seeds 94.5–97.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.2** [97.7, 98.8]; seeds 97.5–99.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.5** [99.2, 99.5]; seeds 99.0–99.5; N 5×200=1000 | **97.2** [96.3, 97.5]; seeds 96.0–97.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.0** [97.2, 98.5]; seeds 97.0–98.5; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 95.5 [91.7, 97.6] 191/200 | 86.0 [80.5, 90.1] 172/200 | 78.5 [72.3, 83.6] 157/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 100.0 [98.1, 100.0] 200/200 | 96.5 [93.0, 98.3] 193/200 | 93.5 [89.2, 96.2] 187/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 99.5 [97.2, 99.9] 199/200 | 96.5 [93.0, 98.3] 193/200 | 88.5 [83.3, 92.2] 177/200 | 81.0 [75.0, 85.8] 162/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 100.0 [98.1, 100.0] 200/200 | 98.5 [95.7, 99.5] 197/200 | 95.5 [91.7, 97.6] 191/200 | 90.5 [85.6, 93.8] 181/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 97.5 [94.3, 98.9] 195/200 | 88.0 [82.8, 91.8] 176/200 | 63.0 [56.1, 69.4] 126/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 89.5 [84.5, 93.0] 179/200 | 64.0 [57.1, 70.3] 128/200 |

### `noise/sigma0cm_lat1step` (σp 0 cm, σv 0 m/s, latency 1 step = 33.3 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.270 [0.267, 0.274] · 0.0 · 0.0 | 0.285 [0.280, 0.290] · 0.0 · 0.0 | 0.295 [0.291, 0.300] · 0.0 · 0.0 | 0.311 [0.299, 0.314] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.395 [0.376, 0.439] · 0.0 · 0.0 | 0.448 [0.426, 0.481] · 0.0 · 0.0 | 0.564 [0.536, 0.596] · 0.3 · 0.0 | 0.690 [0.619, 0.791] · 2.1 · 0.0 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.277 [0.272, 0.279] · 0.0 · 0.0 | 0.280 [0.278, 0.282] · 0.0 · 0.0 | 0.296 [0.291, 0.304] · 0.0 · 0.0 | 0.302 [0.296, 0.309] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.268 [0.265, 0.271] · 0.0 · 0.0 | 0.274 [0.268, 0.277] · 0.0 · 0.0 | 0.280 [0.273, 0.282] · 0.0 · 0.0 | 0.291 [0.285, 0.303] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.281 [0.279, 0.284] · 0.0 · 0.0 | 0.286 [0.282, 0.287] · 0.0 · 0.0 | 0.293 [0.289, 0.296] · 0.0 · 0.0 | 0.292 [0.289, 0.293] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.278 [0.274, 0.281] · 0.0 · 0.0 | 0.292 [0.286, 0.295] · 0.0 · 0.0 | 0.299 [0.298, 0.303] · 0.0 · 0.0 | 0.310 [0.307, 0.313] · 0.0 · 0.0 |
| `pid_track_descend` | 0.315 · 0.0 · 0.0 | 0.407 · 0.0 · 0.0 | 0.494 · 0.0 · 0.0 | 0.514 · 0.0 · 0.0 |
| `pid_feedforward` | 0.233 · 0.0 · 0.0 | 0.255 · 0.0 · 0.0 | 0.312 · 0.0 · 0.0 | 0.308 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.146 · 0.0 · 0.0 | 0.178 · 0.0 · 0.0 | 0.242 · 0.0 · 0.0 | 0.242 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.146 · 0.0 · 0.0 | 0.178 · 0.0 · 0.0 | 0.242 · 0.0 · 0.0 | 0.242 · 0.0 · 0.0 |
| `gated` | 0.229 · 0.0 · 0.0 | 0.253 · 0.0 · 2.0 | 0.288 · 0.0 · 10.0 | 0.319 · 0.0 · 33.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.229 · 0.0 · 0.0 | 0.253 · 0.0 · 1.0 | 0.288 · 0.0 · 10.5 | 0.269 · 0.0 · 36.0 |

### `noise/sigma0cm_lat1step` (σp 0 cm, σv 0 m/s, latency 1 step = 33.3 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 2 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 13 (1.3) | 3 (0.3) | 984 (98.4) | 0 (0.0) | 10 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 1 (0.1) | 994 (99.4) | 0 (0.0) | 3 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 20 (2.0) | 0 (0.0) | 980 (98.0) | 0 (0.0) | 17 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 3 (0.3) | 15 (1.5) | 121 (12.1) | 1 (0.1) | 860 (86.0) | 0 (0.0) | 26 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 21 (2.1) | 38 (3.8) | 217 (21.7) | 6 (0.6) | 718 (71.8) | 0 (0.0) | 58 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 4 (0.4) | 3 (0.3) | 993 (99.3) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 24 (2.4) | 12 (1.2) | 964 (96.4) | 0 (0.0) | 2 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 16 (1.6) | 2 (0.2) | 982 (98.2) | 0 (0.0) | 7 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 1 (0.1) | 994 (99.4) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 16 (1.6) | 14 (1.4) | 970 (97.0) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 1 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 0 (0.0) | 979 (97.9) | 0 (0.0) | 7 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 5 (2.5) | 191 (95.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 10 (5.0) | 18 (9.0) | 172 (86.0) | 0 (0.0) | 1 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 21 (10.5) | 22 (11.0) | 157 (78.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 6 (3.0) | 193 (96.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 7 (3.5) | 187 (93.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 193 (96.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 23 (11.5) | 177 (88.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 31 (15.5) | 162 (81.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 9 (4.5) | 191 (95.5) | 0 (0.0) | 2 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 12 (6.0) | 181 (90.5) | 0 (0.0) | 2 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 195 (97.5) | 4 (2.0) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 176 (88.0) | 20 (10.0) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 126 (63.0) | 67 (33.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 179 (89.5) | 21 (10.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 128 (64.0) | 72 (36.0) | 0 |

### `noise/sigma1cm_lat1step` (σp 1 cm, σv 0.05 m/s, latency 1 step = 33.3 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **99.3** [99.0, 99.8]; seeds 99.0–100.0; N 5×200=1000 | **98.2** [97.3, 98.5]; seeds 97.0–98.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **92.5** [88.8, 97.0]; seeds 88.5–97.0; N 5×200=1000 | **88.5** [85.2, 95.3]; seeds 85.0–97.5; N 5×200=1000 | **76.5** [72.0, 81.7]; seeds 71.5–83.0; N 5×200=1000 | **62.3** [61.0, 68.8]; seeds 60.5–72.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **99.7** [99.2, 100.0]; seeds 99.0–100.0; N 5×200=1000 | **99.0** [98.2, 99.5]; seeds 98.0–99.5; N 5×200=1000 | **95.2** [94.3, 95.8]; seeds 94.0–96.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **97.7** [97.5, 98.3]; seeds 97.5–98.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **99.8** [99.5, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **98.8** [98.5, 99.3]; seeds 98.5–99.5; N 5×200=1000 | **95.8** [94.8, 96.3]; seeds 94.5–96.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **97.0** [96.5, 98.2]; seeds 96.5–98.5; N 5×200=1000 |
| `pid_track_descend` | 97.0 [93.6, 98.6] 194/200 | 96.0 [92.3, 98.0] 192/200 | 82.0 [76.1, 86.7] 164/200 | 78.5 [72.3, 83.6] 157/200 |
| `pid_feedforward` | 98.5 [95.7, 99.5] 197/200 | 98.5 [95.7, 99.5] 197/200 | 95.0 [91.0, 97.3] 190/200 | 88.0 [82.8, 91.8] 176/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 97.0 [93.6, 98.6] 194/200 | 90.0 [85.1, 93.4] 180/200 | 83.5 [77.7, 88.0] 167/200 | 77.5 [71.2, 82.7] 155/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 98.5 [95.7, 99.5] 197/200 | 96.0 [92.3, 98.0] 192/200 | 96.0 [92.3, 98.0] 192/200 | 91.0 [86.2, 94.2] 182/200 |
| `gated` | 99.0 [96.4, 99.7] 198/200 | 94.5 [90.4, 96.9] 189/200 | 82.5 [76.6, 87.1] 165/200 | 51.5 [44.6, 58.3] 103/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 98.0 [95.0, 99.2] 196/200 | 96.0 [92.3, 98.0] 192/200 | 85.0 [79.4, 89.3] 170/200 | 62.0 [55.1, 68.4] 124/200 |

### `noise/sigma1cm_lat1step` (σp 1 cm, σv 0.05 m/s, latency 1 step = 33.3 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.309 [0.308, 0.316] · 0.0 · 0.0 | 0.331 [0.323, 0.333] · 0.0 · 0.0 | 0.321 [0.314, 0.331] · 0.0 · 0.0 | 0.331 [0.323, 0.342] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.518 [0.448, 0.613] · 0.0 · 0.0 | 0.575 [0.497, 0.589] · 0.1 · 0.0 | 0.636 [0.593, 0.701] · 1.0 · 0.0 | 0.695 [0.620, 0.749] · 2.7 · 0.0 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.310 [0.304, 0.317] · 0.0 · 0.0 | 0.316 [0.314, 0.324] · 0.0 · 0.0 | 0.335 [0.328, 0.336] · 0.0 · 0.0 | 0.334 [0.333, 0.337] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.286 [0.284, 0.291] · 0.0 · 0.0 | 0.293 [0.289, 0.299] · 0.0 · 0.0 | 0.294 [0.289, 0.298] · 0.0 · 0.0 | 0.306 [0.300, 0.321] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.308 [0.301, 0.315] · 0.1 · 0.0 | 0.316 [0.311, 0.324] · 0.0 · 0.0 | 0.320 [0.314, 0.328] · 0.0 · 0.0 | 0.316 [0.312, 0.322] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.317 [0.309, 0.324] · 0.0 · 0.0 | 0.335 [0.328, 0.338] · 0.0 · 0.0 | 0.323 [0.320, 0.325] · 0.0 · 0.0 | 0.339 [0.335, 0.351] · 0.0 · 0.0 |
| `pid_track_descend` | 0.302 · 0.0 · 0.0 | 0.419 · 0.0 · 0.0 | 0.533 · 0.0 · 0.0 | 0.507 · 0.0 · 0.0 |
| `pid_feedforward` | 0.254 · 0.0 · 0.0 | 0.278 · 0.0 · 0.0 | 0.357 · 0.0 · 0.0 | 0.308 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.168 · 0.0 · 0.0 | 0.190 · 0.0 · 0.0 | 0.263 · 0.0 · 0.0 | 0.224 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.168 · 0.0 · 0.0 | 0.190 · 0.0 · 0.0 | 0.263 · 0.0 · 0.0 | 0.224 · 0.0 · 0.0 |
| `gated` | 0.261 · 0.0 · 0.0 | 0.266 · 0.0 · 3.0 | 0.317 · 0.0 · 15.0 | 0.324 · 0.0 · 45.0 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.274 · 0.0 · 0.0 | 0.267 · 0.0 · 2.0 | 0.288 · 0.0 · 12.5 | 0.286 · 0.0 · 36.0 |

### `noise/sigma1cm_lat1step` (σp 1 cm, σv 0.05 m/s, latency 1 step = 33.3 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 3 (0.3) | 994 (99.4) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 14 (1.4) | 6 (0.6) | 980 (98.0) | 0 (0.0) | 13 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 3 (0.3) | 71 (7.1) | 0 (0.0) | 926 (92.6) | 0 (0.0) | 24 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 1 (0.1) | 8 (0.8) | 91 (9.1) | 4 (0.4) | 896 (89.6) | 0 (0.0) | 30 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 10 (1.0) | 31 (3.1) | 188 (18.8) | 3 (0.3) | 768 (76.8) | 0 (0.0) | 65 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 27 (2.7) | 68 (6.8) | 257 (25.7) | 9 (0.9) | 639 (63.9) | 0 (0.0) | 93 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 3 (0.3) | 996 (99.6) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 9 (0.9) | 2 (0.2) | 989 (98.9) | 0 (0.0) | 1 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 41 (4.1) | 8 (0.8) | 951 (95.1) | 0 (0.0) | 5 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 20 (2.0) | 2 (0.2) | 978 (97.8) | 0 (0.0) | 7 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 1 (0.1) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 998 (99.8) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 0 (0.0) | 999 (99.9) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 5 (0.5) | 989 (98.9) | 0 (0.0) | 1 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 34 (3.4) | 9 (0.9) | 957 (95.7) | 0 (0.0) | 4 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 1 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 25 (2.5) | 3 (0.3) | 972 (97.2) | 0 (0.0) | 9 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 6 (3.0) | 192 (96.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 14 (7.0) | 22 (11.0) | 164 (82.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 18 (9.0) | 25 (12.5) | 157 (78.5) | 0 (0.0) | 1 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 10 (5.0) | 190 (95.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 15 (7.5) | 176 (88.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 20 (10.0) | 180 (90.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 31 (15.5) | 167 (83.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 10 (5.0) | 35 (17.5) | 155 (77.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 8 (4.0) | 192 (96.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 6 (3.0) | 192 (96.0) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 10 (5.0) | 8 (4.0) | 182 (91.0) | 0 (0.0) | 0 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 189 (94.5) | 6 (3.0) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 165 (82.5) | 30 (15.0) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (3.5) | 103 (51.5) | 90 (45.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 192 (96.0) | 4 (2.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 170 (85.0) | 25 (12.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 124 (62.0) | 72 (36.0) | 0 |

### `noise/sigma2cm_lat1step` (σp 2 cm, σv 0.1 m/s, latency 1 step = 33.3 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **99.2** [99.0, 99.8]; seeds 99.0–100.0; N 5×200=1000 | **99.2** [98.7, 99.5]; seeds 98.5–99.5; N 5×200=1000 | **98.5** [98.2, 99.2]; seeds 98.0–99.5; N 5×200=1000 | **95.7** [94.3, 96.8]; seeds 94.0–97.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **61.5** [55.2, 76.2]; seeds 54.5–79.0; N 5×200=1000 | **62.0** [57.3, 75.7]; seeds 56.5–81.0; N 5×200=1000 | **51.5** [43.8, 61.5]; seeds 42.0–64.0; N 5×200=1000 | **37.8** [34.2, 51.7]; seeds 33.5–57.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **80.5** [76.3, 83.5]; seeds 76.0–83.5; N 5×200=1000 | **79.3** [75.0, 85.7]; seeds 74.5–87.5; N 5×200=1000 | **77.7** [74.8, 82.5]; seeds 74.5–83.0; N 5×200=1000 | **72.7** [70.8, 75.7]; seeds 70.0–77.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **99.7** [99.0, 100.0]; seeds 99.0–100.0; N 5×200=1000 | **96.7** [95.3, 97.7]; seeds 95.0–98.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **83.3** [79.0, 91.5]; seeds 79.0–93.0; N 5×200=1000 | **85.7** [79.2, 91.2]; seeds 77.0–92.5; N 5×200=1000 | **83.5** [80.8, 87.5]; seeds 80.5–89.0; N 5×200=1000 | **80.7** [76.8, 87.2]; seeds 76.0–88.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **98.8** [98.5, 99.0]; seeds 98.5–99.0; N 5×200=1000 | **99.3** [98.2, 100.0]; seeds 98.0–100.0; N 5×200=1000 | **97.2** [96.7, 97.8]; seeds 96.5–98.0; N 5×200=1000 | **96.5** [93.7, 97.8]; seeds 92.5–98.0; N 5×200=1000 |
| `pid_track_descend` | 85.5 [80.0, 89.7] 171/200 | 82.5 [76.6, 87.1] 165/200 | 74.0 [67.5, 79.6] 148/200 | 66.5 [59.7, 72.7] 133/200 |
| `pid_feedforward` | 63.0 [56.1, 69.4] 126/200 | 62.0 [55.1, 68.4] 124/200 | 61.0 [54.1, 67.5] 122/200 | 54.0 [47.1, 60.8] 108/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 71.5 [64.9, 77.3] 143/200 | 65.5 [58.7, 71.7] 131/200 | 59.5 [52.6, 66.1] 119/200 | 50.0 [43.1, 56.9] 100/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 92.5 [88.0, 95.4] 185/200 | 88.5 [83.3, 92.2] 177/200 | 88.0 [82.8, 91.8] 176/200 | 80.0 [73.9, 85.0] 160/200 |
| `gated` | 11.5 [7.8, 16.7] 23/200 | 9.5 [6.2, 14.4] 19/200 | 5.5 [3.1, 9.6] 11/200 | 1.5 [0.5, 4.3] 3/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 28.0 [22.2, 34.6] 56/200 | 25.5 [20.0, 32.0] 51/200 | 16.5 [12.0, 22.3] 33/200 | 7.5 [4.6, 12.0] 15/200 |

### `noise/sigma2cm_lat1step` (σp 2 cm, σv 0.1 m/s, latency 1 step = 33.3 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.359 [0.350, 0.371] · 0.0 · 0.0 | 0.383 [0.345, 0.400] · 0.0 · 0.0 | 0.349 [0.343, 0.353] · 0.0 · 0.0 | 0.374 [0.362, 0.386] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.663 [0.594, 0.802] · 0.7 · 0.2 | 0.666 [0.572, 0.777] · 0.5 · 0.0 | 0.718 [0.637, 0.763] · 1.7 · 0.1 | 0.780 [0.687, 0.861] · 4.7 · 0.2 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.370 [0.361, 0.386] · 0.3 · 0.0 | 0.367 [0.353, 0.381] · 0.3 · 0.0 | 0.382 [0.373, 0.397] · 0.1 · 0.0 | 0.381 [0.367, 0.386] · 0.2 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.313 [0.308, 0.318] · 0.0 · 0.0 | 0.323 [0.316, 0.333] · 0.0 · 0.0 | 0.318 [0.312, 0.327] · 0.0 · 0.0 | 0.330 [0.327, 0.337] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.357 [0.345, 0.368] · 0.1 · 0.0 | 0.353 [0.347, 0.357] · 0.3 · 0.0 | 0.361 [0.355, 0.370] · 0.1 · 0.0 | 0.365 [0.355, 0.373] · 0.2 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.372 [0.362, 0.377] · 0.0 · 0.0 | 0.381 [0.376, 0.384] · 0.0 · 0.0 | 0.374 [0.359, 0.389] · 0.0 · 0.0 | 0.389 [0.384, 0.404] · 0.0 · 0.0 |
| `pid_track_descend` | 0.308 · 0.0 · 0.0 | 0.392 · 0.0 · 0.0 | 0.493 · 0.0 · 0.0 | 0.513 · 0.0 · 0.0 |
| `pid_feedforward` | 0.323 · 0.0 · 0.0 | 0.315 · 0.0 · 0.0 | 0.351 · 0.0 · 0.0 | 0.356 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.231 · 0.0 · 0.0 | 0.234 · 0.0 · 0.0 | 0.273 · 0.0 · 0.0 | 0.245 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.231 · 0.0 · 0.0 | 0.234 · 0.0 · 0.0 | 0.273 · 0.0 · 0.0 | 0.245 · 0.0 · 0.0 |
| `gated` | 0.330 · 0.0 · 69.0 | 0.316 · 0.0 · 76.5 | 0.333 · 0.0 · 88.5 | 0.304 · 0.0 · 96.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.312 · 0.5 · 43.0 | 0.310 · 0.5 · 50.5 | 0.332 · 0.0 · 68.5 | 0.310 · 0.0 · 86.5 |

### `noise/sigma2cm_lat1step` (σp 2 cm, σv 0.1 m/s, latency 1 step = 33.3 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 5 (0.5) | 993 (99.3) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 7 (0.7) | 991 (99.1) | 0 (0.0) | 2 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 3 (0.3) | 11 (1.1) | 986 (98.6) | 0 (0.0) | 7 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 3 (0.3) | 23 (2.3) | 18 (1.8) | 956 (95.6) | 0 (0.0) | 13 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 7 (0.7) | 60 (6.0) | 289 (28.9) | 6 (0.6) | 636 (63.6) | 2 (0.2) | 95 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 5 (0.5) | 46 (4.6) | 297 (29.7) | 5 (0.5) | 647 (64.7) | 0 (0.0) | 121 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 17 (1.7) | 98 (9.8) | 350 (35.0) | 13 (1.3) | 521 (52.1) | 1 (0.1) | 109 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 47 (4.7) | 126 (12.6) | 396 (39.6) | 20 (2.0) | 409 (40.9) | 2 (0.2) | 106 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 3 (0.3) | 8 (0.8) | 43 (4.3) | 144 (14.4) | 802 (80.2) | 0 (0.0) | 3 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 3 (0.3) | 4 (0.4) | 48 (4.8) | 145 (14.5) | 800 (80.0) | 0 (0.0) | 2 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 1 (0.1) | 7 (0.7) | 65 (6.5) | 146 (14.6) | 781 (78.1) | 0 (0.0) | 7 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 2 (0.2) | 8 (0.8) | 108 (10.8) | 152 (15.2) | 730 (73.0) | 0 (0.0) | 12 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 2 (0.2) | 996 (99.6) | 0 (0.0) | 3 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 31 (3.1) | 2 (0.2) | 966 (96.6) | 0 (0.0) | 9 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 1 (0.1) | 5 (0.5) | 39 (3.9) | 111 (11.1) | 844 (84.4) | 0 (0.0) | 1 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 3 (0.3) | 8 (0.8) | 31 (3.1) | 105 (10.5) | 853 (85.3) | 0 (0.0) | 3 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 1 (0.1) | 3 (0.3) | 46 (4.6) | 110 (11.0) | 840 (84.0) | 0 (0.0) | 6 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 2 (0.2) | 4 (0.4) | 82 (8.2) | 99 (9.9) | 813 (81.3) | 0 (0.0) | 8 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 2 (0.2) | 2 (0.2) | 8 (0.8) | 988 (98.8) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 6 (0.6) | 992 (99.2) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 1 (0.1) | 7 (0.7) | 20 (2.0) | 972 (97.2) | 0 (0.0) | 4 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 33 (3.3) | 6 (0.6) | 960 (96.0) | 0 (0.0) | 12 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 29 (14.5) | 171 (85.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 33 (16.5) | 165 (82.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 10 (5.0) | 42 (21.0) | 148 (74.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 21 (10.5) | 46 (23.0) | 133 (66.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 1 (0.5) | 8 (4.0) | 65 (32.5) | 126 (63.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 4 (2.0) | 72 (36.0) | 124 (62.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 69 (34.5) | 122 (61.0) | 0 (0.0) | 1 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 19 (9.5) | 73 (36.5) | 108 (54.0) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 57 (28.5) | 143 (71.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 68 (34.0) | 131 (65.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 1 (0.5) | 7 (3.5) | 73 (36.5) | 119 (59.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 18 (9.0) | 82 (41.0) | 100 (50.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 15 (7.5) | 185 (92.5) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 22 (11.0) | 177 (88.5) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 1 (0.5) | 7 (3.5) | 16 (8.0) | 176 (88.0) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 18 (9.0) | 22 (11.0) | 160 (80.0) | 0 (0.0) | 5 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 39 (19.5) | 23 (11.5) | 138 (69.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 26 (13.0) | 19 (9.5) | 153 (76.5) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 12 (6.0) | 11 (5.5) | 177 (88.5) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 3 (1.5) | 193 (96.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 1 (0.5) | 0 (0.0) | 1 (0.5) | 56 (28.0) | 56 (28.0) | 86 (43.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 1 (0.5) | 0 (0.0) | 0 (0.0) | 47 (23.5) | 51 (25.5) | 101 (50.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 29 (14.5) | 33 (16.5) | 137 (68.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 10 (5.0) | 15 (7.5) | 173 (86.5) | 0 |

### `noise/sigma4cm_lat1step` (σp 4 cm, σv 0.2 m/s, latency 1 step = 33.3 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **79.7** [76.7, 87.5]; seeds 76.0–90.0; N 5×200=1000 | **79.8** [72.2, 89.7]; seeds 71.5–91.0; N 5×200=1000 | **73.7** [68.3, 85.7]; seeds 67.0–90.0; N 5×200=1000 | **75.8** [69.8, 84.2]; seeds 67.5–87.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **18.2** [13.0, 37.0]; seeds 12.5–44.0; N 5×200=1000 | **19.3** [14.8, 35.5]; seeds 13.5–42.0; N 5×200=1000 | **13.7** [8.3, 29.7]; seeds 8.0–35.5; N 5×200=1000 | **9.5** [6.2, 16.2]; seeds 6.0–18.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **2.7** [1.3, 6.0]; seeds 1.0–7.5; N 5×200=1000 | **2.3** [1.2, 5.2]; seeds 1.0–6.0; N 5×200=1000 | **3.0** [1.8, 4.7]; seeds 1.5–5.0; N 5×200=1000 | **3.3** [2.3, 7.0]; seeds 2.0–8.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **95.7** [93.3, 97.8]; seeds 93.0–98.0; N 5×200=1000 | **96.2** [95.0, 98.0]; seeds 95.0–98.0; N 5×200=1000 | **95.5** [93.8, 97.2]; seeds 93.5–97.5; N 5×200=1000 | **90.7** [87.8, 93.2]; seeds 87.5–93.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **4.3** [2.8, 8.2]; seeds 2.5–9.5; N 5×200=1000 | **4.7** [3.8, 9.7]; seeds 3.5–12.0; N 5×200=1000 | **5.2** [3.8, 8.7]; seeds 3.5–10.0; N 5×200=1000 | **4.0** [3.2, 9.3]; seeds 3.0–11.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **72.7** [57.2, 82.2]; seeds 53.0–84.5; N 5×200=1000 | **72.7** [64.2, 78.5]; seeds 63.5–79.5; N 5×200=1000 | **71.0** [53.5, 82.2]; seeds 50.0–84.5; N 5×200=1000 | **71.0** [52.7, 74.3]; seeds 45.5–74.5; N 5×200=1000 |
| `pid_track_descend` | 19.5 [14.6, 25.5] 39/200 | 22.0 [16.8, 28.2] 44/200 | 22.0 [16.8, 28.2] 44/200 | 23.5 [18.2, 29.8] 47/200 |
| `pid_feedforward` | 0.5 [0.1, 2.8] 1/200 | 0.5 [0.1, 2.8] 1/200 | 0.0 [0.0, 1.9] 0/200 | 0.5 [0.1, 2.8] 1/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 8.5 [5.4, 13.2] 17/200 | 6.5 [3.8, 10.8] 13/200 | 6.5 [3.8, 10.8] 13/200 | 6.5 [3.8, 10.8] 13/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 42.5 [35.9, 49.4] 85/200 | 42.5 [35.9, 49.4] 85/200 | 43.0 [36.3, 49.9] 86/200 | 42.0 [35.4, 48.9] 84/200 |
| `gated` | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 |

### `noise/sigma4cm_lat1step` (σp 4 cm, σv 0.2 m/s, latency 1 step = 33.3 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.416 [0.406, 0.440] · 0.1 · 0.0 | 0.437 [0.423, 0.457] · 0.0 · 0.0 | 0.433 [0.420, 0.443] · 0.1 · 0.0 | 0.452 [0.427, 0.475] · 0.1 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.929 [0.761, 1.028] · 11.4 · 0.7 | 0.892 [0.786, 0.933] · 10.4 · 0.4 | 0.891 [0.783, 0.941] · 13.0 · 0.7 | 0.981 [0.837, 1.137] · 22.3 · 1.0 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.562 [0.513, 0.618] · 14.2 · 0.0 | 0.581 [0.544, 0.680] · 14.1 · 0.0 | 0.572 [0.534, 0.658] · 15.2 · 0.0 | 0.588 [0.528, 0.622] · 15.7 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.366 [0.356, 0.378] · 0.0 · 0.0 | 0.366 [0.358, 0.375] · 0.0 · 0.0 | 0.355 [0.338, 0.369] · 0.0 · 0.0 | 0.377 [0.365, 0.386] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.570 [0.512, 0.586] · 9.5 · 0.0 | 0.552 [0.513, 0.623] · 8.8 · 0.0 | 0.599 [0.540, 0.621] · 10.4 · 0.0 | 0.560 [0.526, 0.640] · 10.8 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.433 [0.420, 0.456] · 0.0 · 0.0 | 0.451 [0.444, 0.462] · 0.0 · 0.0 | 0.441 [0.432, 0.448] · 0.2 · 0.0 | 0.465 [0.449, 0.503] · 0.3 · 0.0 |
| `pid_track_descend` | 0.293 · 0.0 · 34.5 | 0.350 · 0.0 · 34.0 | 0.492 · 0.0 · 26.5 | 0.428 · 0.0 · 30.5 |
| `pid_feedforward` | 0.512 · 40.0 · 29.0 | 0.541 · 41.0 · 31.0 | 0.514 · 43.0 · 25.5 | 0.549 · 40.5 · 30.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.382 · 6.5 · 5.0 | 0.373 · 6.0 · 5.0 | 0.382 · 3.5 · 6.5 | 0.361 · 6.0 · 4.5 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.382 · 6.5 · 5.0 | 0.373 · 6.0 · 5.0 | 0.382 · 3.5 · 6.5 | 0.361 · 6.0 · 4.5 |
| `gated` | – · 59.0 · 41.0 | 0.601 · 67.5 · 32.0 | – · 60.5 · 39.5 | 0.826 · 61.0 · 38.5 |
| `oracle_gated` — commit-timing oracle (privileged) | – · 62.5 · 37.5 | – · 65.5 · 34.5 | – · 59.0 · 41.0 | 0.867 · 62.5 · 37.0 |

### `noise/sigma4cm_lat1step` (σp 4 cm, σv 0.2 m/s, latency 1 step = 33.3 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 1 (0.1) | 40 (4.0) | 78 (7.8) | 71 (7.1) | 810 (81.0) | 0 (0.0) | 33 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 37 (3.7) | 77 (7.7) | 82 (8.2) | 804 (80.4) | 0 (0.0) | 30 |
| `ppo` (learned, PPO) | SS5 | 1000 | 1 (0.1) | 36 (3.6) | 98 (9.8) | 109 (10.9) | 756 (75.6) | 0 (0.0) | 34 |
| `ppo` (learned, PPO) | SS6 | 1000 | 1 (0.1) | 42 (4.2) | 120 (12.0) | 73 (7.3) | 764 (76.4) | 0 (0.0) | 59 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 114 (11.4) | 217 (21.7) | 406 (40.6) | 34 (3.4) | 222 (22.2) | 7 (0.7) | 84 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 104 (10.4) | 241 (24.1) | 395 (39.5) | 29 (2.9) | 227 (22.7) | 4 (0.4) | 86 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 130 (13.0) | 291 (29.1) | 374 (37.4) | 29 (2.9) | 169 (16.9) | 7 (0.7) | 64 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 223 (22.3) | 288 (28.8) | 352 (35.2) | 22 (2.2) | 105 (10.5) | 10 (1.0) | 52 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 142 (14.2) | 329 (32.9) | 295 (29.5) | 201 (20.1) | 33 (3.3) | 0 (0.0) | 3 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 141 (14.1) | 340 (34.0) | 298 (29.8) | 193 (19.3) | 28 (2.8) | 0 (0.0) | 3 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 152 (15.2) | 350 (35.0) | 283 (28.3) | 184 (18.4) | 31 (3.1) | 0 (0.0) | 3 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 157 (15.7) | 312 (31.2) | 308 (30.8) | 182 (18.2) | 41 (4.1) | 0 (0.0) | 3 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 4 (0.4) | 15 (1.5) | 25 (2.5) | 956 (95.6) | 0 (0.0) | 6 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 7 (0.7) | 10 (1.0) | 20 (2.0) | 963 (96.3) | 0 (0.0) | 4 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 7 (0.7) | 18 (1.8) | 20 (2.0) | 955 (95.5) | 0 (0.0) | 9 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 5 (0.5) | 59 (5.9) | 30 (3.0) | 906 (90.6) | 0 (0.0) | 23 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 95 (9.5) | 312 (31.2) | 298 (29.8) | 245 (24.5) | 50 (5.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 88 (8.8) | 307 (30.7) | 324 (32.4) | 222 (22.2) | 59 (5.9) | 0 (0.0) | 1 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 104 (10.4) | 318 (31.8) | 288 (28.8) | 232 (23.2) | 58 (5.8) | 0 (0.0) | 5 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 108 (10.8) | 306 (30.6) | 341 (34.1) | 192 (19.2) | 53 (5.3) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 51 (5.1) | 94 (9.4) | 144 (14.4) | 711 (71.1) | 0 (0.0) | 28 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 43 (4.3) | 111 (11.1) | 124 (12.4) | 722 (72.2) | 0 (0.0) | 25 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 2 (0.2) | 53 (5.3) | 98 (9.8) | 152 (15.2) | 695 (69.5) | 0 (0.0) | 24 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 3 (0.3) | 56 (5.6) | 155 (15.5) | 120 (12.0) | 666 (66.6) | 0 (0.0) | 51 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 92 (46.0) | 39 (19.5) | 69 (34.5) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 88 (44.0) | 44 (22.0) | 68 (34.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 97 (48.5) | 44 (22.0) | 53 (26.5) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 85 (42.5) | 47 (23.5) | 61 (30.5) | 0 |
| `pid_feedforward` | SS3 | 200 | 80 (40.0) | 30 (15.0) | 15 (7.5) | 16 (8.0) | 1 (0.5) | 58 (29.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 82 (41.0) | 32 (16.0) | 13 (6.5) | 10 (5.0) | 1 (0.5) | 62 (31.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 86 (43.0) | 29 (14.5) | 18 (9.0) | 16 (8.0) | 0 (0.0) | 51 (25.5) | 0 |
| `pid_feedforward` | SS6 | 200 | 81 (40.5) | 28 (14.0) | 20 (10.0) | 10 (5.0) | 1 (0.5) | 60 (30.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 13 (6.5) | 37 (18.5) | 38 (19.0) | 85 (42.5) | 17 (8.5) | 10 (5.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 12 (6.0) | 28 (14.0) | 38 (19.0) | 99 (49.5) | 13 (6.5) | 10 (5.0) | 1 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 7 (3.5) | 46 (23.0) | 39 (19.5) | 82 (41.0) | 13 (6.5) | 13 (6.5) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 12 (6.0) | 33 (16.5) | 47 (23.5) | 86 (43.0) | 13 (6.5) | 9 (4.5) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 13 (6.5) | 37 (18.5) | 38 (19.0) | 17 (8.5) | 85 (42.5) | 10 (5.0) | 25 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 12 (6.0) | 28 (14.0) | 38 (19.0) | 27 (13.5) | 85 (42.5) | 10 (5.0) | 20 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 7 (3.5) | 46 (23.0) | 39 (19.5) | 9 (4.5) | 86 (43.0) | 13 (6.5) | 17 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 12 (6.0) | 33 (16.5) | 47 (23.5) | 15 (7.5) | 84 (42.0) | 9 (4.5) | 21 |
| `gated` | SS3 | 200 | 118 (59.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 82 (41.0) | 0 |
| `gated` | SS4 | 200 | 135 (67.5) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 64 (32.0) | 0 |
| `gated` | SS5 | 200 | 121 (60.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 79 (39.5) | 0 |
| `gated` | SS6 | 200 | 122 (61.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 77 (38.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 125 (62.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 75 (37.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 131 (65.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 69 (34.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 118 (59.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 82 (41.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 125 (62.5) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 74 (37.0) | 0 |

### `noise/sigma0cm_lat2step` (σp 0 cm, σv 0 m/s, latency 2 step = 66.7 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.3** [97.7, 98.8]; seeds 97.5–99.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **100.0** [98.0, 100.0]; seeds 97.0–100.0; N 5×200=1000 | **97.7** [96.3, 98.7]; seeds 96.0–99.0; N 5×200=1000 | **84.3** [78.3, 89.3]; seeds 76.5–89.5; N 5×200=1000 | **67.2** [59.0, 76.3]; seeds 56.5–78.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.3, 100.0]; seeds 99.0–100.0; N 5×200=1000 | **99.0** [99.0, 99.3]; seeds 99.0–99.5; N 5×200=1000 | **96.3** [94.5, 97.8]; seeds 94.5–98.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **98.5** [98.2, 98.5]; seeds 98.0–98.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.5** [99.2, 99.5]; seeds 99.0–99.5; N 5×200=1000 | **95.8** [95.0, 97.8]; seeds 95.0–98.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **97.7** [97.2, 98.3]; seeds 97.0–98.5; N 5×200=1000 |
| `pid_track_descend` | 100.0 [98.1, 100.0] 200/200 | 95.5 [91.7, 97.6] 191/200 | 84.5 [78.8, 88.9] 169/200 | 80.0 [73.9, 85.0] 160/200 |
| `pid_feedforward` | 100.0 [98.1, 100.0] 200/200 | 99.5 [97.2, 99.9] 199/200 | 92.5 [88.0, 95.4] 185/200 | 88.5 [83.3, 92.2] 177/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 98.0 [95.0, 99.2] 196/200 | 88.5 [83.3, 92.2] 177/200 | 80.5 [74.5, 85.4] 161/200 | 78.0 [71.8, 83.2] 156/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 98.5 [95.7, 99.5] 197/200 | 98.5 [95.7, 99.5] 197/200 | 94.0 [89.8, 96.5] 188/200 | 91.5 [86.8, 94.6] 183/200 |
| `gated` | 100.0 [98.1, 100.0] 200/200 | 98.0 [95.0, 99.2] 196/200 | 87.0 [81.6, 91.0] 174/200 | 60.5 [53.6, 67.0] 121/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 100.0 [98.1, 100.0] 200/200 | 99.0 [96.4, 99.7] 198/200 | 88.0 [82.8, 91.8] 176/200 | 64.0 [57.1, 70.3] 128/200 |

### `noise/sigma0cm_lat2step` (σp 0 cm, σv 0 m/s, latency 2 step = 66.7 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.280 [0.277, 0.282] · 0.0 · 0.0 | 0.305 [0.300, 0.315] · 0.0 · 0.0 | 0.330 [0.324, 0.333] · 0.0 · 0.0 | 0.338 [0.326, 0.349] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.395 [0.374, 0.442] · 0.0 · 0.0 | 0.450 [0.427, 0.481] · 0.0 · 0.0 | 0.591 [0.548, 0.620] · 0.6 · 0.0 | 0.674 [0.605, 0.731] · 1.7 · 0.2 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.287 [0.284, 0.290] · 0.0 · 0.0 | 0.296 [0.293, 0.303] · 0.0 · 0.0 | 0.335 [0.331, 0.343] · 0.0 · 0.0 | 0.332 [0.327, 0.336] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.275 [0.270, 0.276] · 0.0 · 0.0 | 0.285 [0.281, 0.289] · 0.0 · 0.0 | 0.295 [0.290, 0.302] · 0.0 · 0.0 | 0.308 [0.299, 0.316] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.293 [0.291, 0.294] · 0.0 · 0.0 | 0.297 [0.293, 0.301] · 0.0 · 0.0 | 0.324 [0.318, 0.332] · 0.0 · 0.0 | 0.312 [0.308, 0.318] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.289 [0.285, 0.293] · 0.0 · 0.0 | 0.318 [0.316, 0.327] · 0.0 · 0.0 | 0.334 [0.331, 0.347] · 0.0 · 0.0 | 0.346 [0.342, 0.350] · 0.0 · 0.0 |
| `pid_track_descend` | 0.316 · 0.0 · 0.0 | 0.423 · 0.0 · 0.0 | 0.500 · 0.0 · 0.0 | 0.536 · 0.0 · 0.0 |
| `pid_feedforward` | 0.251 · 0.0 · 0.0 | 0.285 · 0.0 · 0.0 | 0.355 · 0.0 · 0.0 | 0.354 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.156 · 0.0 · 0.0 | 0.207 · 0.0 · 0.0 | 0.294 · 0.0 · 0.0 | 0.304 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.156 · 0.0 · 0.0 | 0.207 · 0.0 · 0.0 | 0.294 · 0.0 · 0.0 | 0.304 · 0.0 · 0.0 |
| `gated` | 0.258 · 0.0 · 0.0 | 0.273 · 0.0 · 2.0 | 0.324 · 0.0 · 10.0 | 0.382 · 0.0 · 34.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.258 · 0.0 · 0.0 | 0.266 · 0.0 · 1.0 | 0.316 · 0.0 · 11.5 | 0.288 · 0.0 · 35.0 |

### `noise/sigma0cm_lat2step` (σp 0 cm, σv 0 m/s, latency 2 step = 66.7 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 12 (1.2) | 5 (0.5) | 983 (98.3) | 0 (0.0) | 8 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 0 (0.0) | 994 (99.4) | 0 (0.0) | 3 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 0 (0.0) | 1 (0.1) | 23 (2.3) | 0 (0.0) | 976 (97.6) | 0 (0.0) | 10 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 6 (0.6) | 14 (1.4) | 142 (14.2) | 0 (0.0) | 838 (83.8) | 0 (0.0) | 27 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 17 (1.7) | 37 (3.7) | 264 (26.4) | 7 (0.7) | 673 (67.3) | 2 (0.2) | 52 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (0.2) | 998 (99.8) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 3 (0.3) | 991 (99.1) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 26 (2.6) | 11 (1.1) | 963 (96.3) | 0 (0.0) | 4 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.1) | 999 (99.9) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 16 (1.6) | 0 (0.0) | 984 (98.4) | 0 (0.0) | 4 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 5 (0.5) | 1 (0.1) | 994 (99.4) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 25 (2.5) | 13 (1.3) | 962 (96.2) | 0 (0.0) | 4 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 1 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 20 (2.0) | 3 (0.3) | 977 (97.7) | 0 (0.0) | 1 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 5 (2.5) | 4 (2.0) | 191 (95.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 11 (5.5) | 20 (10.0) | 169 (84.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 25 (12.5) | 15 (7.5) | 160 (80.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 13 (6.5) | 185 (92.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 15 (7.5) | 177 (88.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 23 (11.5) | 177 (88.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 38 (19.0) | 161 (80.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 35 (17.5) | 156 (78.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 11 (5.5) | 188 (94.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 9 (4.5) | 8 (4.0) | 183 (91.5) | 0 (0.0) | 2 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 196 (98.0) | 4 (2.0) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 174 (87.0) | 20 (10.0) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 9 (4.5) | 121 (60.5) | 69 (34.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 200 (100.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 198 (99.0) | 2 (1.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 176 (88.0) | 23 (11.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 128 (64.0) | 70 (35.0) | 0 |

### `noise/sigma1cm_lat2step` (σp 1 cm, σv 0.05 m/s, latency 2 step = 66.7 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.8** [99.2, 100.0]; seeds 99.0–100.0; N 5×200=1000 | **98.2** [96.8, 99.0]; seeds 96.5–99.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **92.2** [88.0, 96.5]; seeds 87.5–97.0; N 5×200=1000 | **87.5** [84.7, 94.3]; seeds 84.5–96.0; N 5×200=1000 | **76.3** [69.0, 84.0]; seeds 66.0–86.5; N 5×200=1000 | **59.3** [52.3, 65.3]; seeds 49.0–68.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.8** [99.2, 100.0]; seeds 99.0–100.0; N 5×200=1000 | **98.5** [97.2, 99.0]; seeds 97.0–99.0; N 5×200=1000 | **94.5** [93.2, 95.3]; seeds 93.0–95.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.0** [96.8, 98.5]; seeds 96.5–98.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **100.0** [99.7, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **98.3** [98.0, 99.2]; seeds 98.0–99.5; N 5×200=1000 | **95.2** [94.0, 96.7]; seeds 94.0–97.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **97.2** [96.7, 97.8]; seeds 96.5–98.0; N 5×200=1000 |
| `pid_track_descend` | 97.0 [93.6, 98.6] 194/200 | 94.0 [89.8, 96.5] 188/200 | 83.5 [77.7, 88.0] 167/200 | 80.0 [73.9, 85.0] 160/200 |
| `pid_feedforward` | 99.5 [97.2, 99.9] 199/200 | 98.5 [95.7, 99.5] 197/200 | 90.0 [85.1, 93.4] 180/200 | 86.5 [81.1, 90.6] 173/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 94.5 [90.4, 96.9] 189/200 | 89.0 [83.9, 92.6] 178/200 | 70.5 [63.8, 76.4] 141/200 | 73.5 [67.0, 79.1] 147/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 97.5 [94.3, 98.9] 195/200 | 99.0 [96.4, 99.7] 198/200 | 93.0 [88.6, 95.8] 186/200 | 91.5 [86.8, 94.6] 183/200 |
| `gated` | 97.0 [93.6, 98.6] 194/200 | 95.0 [91.0, 97.3] 190/200 | 76.5 [70.2, 81.8] 153/200 | 48.5 [41.7, 55.4] 97/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 98.0 [95.0, 99.2] 196/200 | 96.0 [92.3, 98.0] 192/200 | 83.5 [77.7, 88.0] 167/200 | 60.5 [53.6, 67.0] 121/200 |

### `noise/sigma1cm_lat2step` (σp 1 cm, σv 0.05 m/s, latency 2 step = 66.7 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.313 [0.302, 0.319] · 0.0 · 0.0 | 0.342 [0.329, 0.352] · 0.0 · 0.0 | 0.345 [0.336, 0.349] · 0.0 · 0.0 | 0.355 [0.344, 0.363] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.518 [0.464, 0.561] · 0.0 · 0.0 | 0.554 [0.504, 0.576] · 0.2 · 0.0 | 0.637 [0.564, 0.692] · 0.5 · 0.3 | 0.704 [0.647, 0.725] · 2.9 · 0.2 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.327 [0.321, 0.331] · 0.0 · 0.0 | 0.336 [0.327, 0.347] · 0.0 · 0.0 | 0.358 [0.354, 0.366] · 0.0 · 0.0 | 0.352 [0.342, 0.358] · 0.0 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.293 [0.286, 0.295] · 0.0 · 0.0 | 0.304 [0.301, 0.306] · 0.0 · 0.0 | 0.304 [0.299, 0.307] · 0.0 · 0.0 | 0.323 [0.316, 0.331] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.313 [0.311, 0.315] · 0.1 · 0.0 | 0.323 [0.316, 0.329] · 0.0 · 0.0 | 0.337 [0.331, 0.342] · 0.0 · 0.0 | 0.334 [0.327, 0.346] · 0.0 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.324 [0.316, 0.332] · 0.0 · 0.0 | 0.352 [0.341, 0.363] · 0.0 · 0.0 | 0.348 [0.342, 0.353] · 0.0 · 0.0 | 0.363 [0.361, 0.370] · 0.0 · 0.0 |
| `pid_track_descend` | 0.314 · 0.0 · 0.0 | 0.401 · 0.0 · 0.0 | 0.535 · 0.0 · 0.0 | 0.537 · 0.0 · 0.0 |
| `pid_feedforward` | 0.265 · 0.0 · 0.0 | 0.299 · 0.0 · 0.0 | 0.407 · 0.0 · 0.0 | 0.335 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.171 · 0.0 · 0.0 | 0.213 · 0.0 · 0.0 | 0.294 · 0.0 · 0.0 | 0.285 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.171 · 0.0 · 0.0 | 0.213 · 0.0 · 0.0 | 0.294 · 0.0 · 0.0 | 0.285 · 0.0 · 0.0 |
| `gated` | 0.271 · 0.0 · 0.0 | 0.287 · 0.0 · 3.5 | 0.339 · 0.0 · 15.0 | 0.370 · 0.0 · 46.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.274 · 0.0 · 0.0 | 0.276 · 0.0 · 2.0 | 0.331 · 0.0 · 12.5 | 0.324 · 0.0 · 38.5 |

### `noise/sigma1cm_lat2step` (σp 1 cm, σv 0.05 m/s, latency 2 step = 66.7 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 2 (0.2) | 997 (99.7) | 0 (0.0) | 3 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 15 (1.5) | 5 (0.5) | 980 (98.0) | 0 (0.0) | 8 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 0 (0.0) | 5 (0.5) | 73 (7.3) | 0 (0.0) | 922 (92.2) | 0 (0.0) | 28 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 2 (0.2) | 6 (0.6) | 105 (10.5) | 1 (0.1) | 886 (88.6) | 0 (0.0) | 39 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 5 (0.5) | 27 (2.7) | 198 (19.8) | 4 (0.4) | 763 (76.3) | 3 (0.3) | 49 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 29 (2.9) | 61 (6.1) | 307 (30.7) | 11 (1.1) | 590 (59.0) | 2 (0.2) | 78 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 2 (0.2) | 997 (99.7) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 11 (1.1) | 6 (0.6) | 983 (98.3) | 0 (0.0) | 0 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 41 (4.1) | 15 (1.5) | 944 (94.4) | 0 (0.0) | 4 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 2 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 1 (0.1) | 978 (97.8) | 0 (0.0) | 4 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 1 (0.1) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 999 (99.9) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 6 (0.6) | 9 (0.9) | 985 (98.5) | 0 (0.0) | 1 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 29 (2.9) | 18 (1.8) | 953 (95.3) | 0 (0.0) | 7 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 1 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 21 (2.1) | 7 (0.7) | 972 (97.2) | 0 (0.0) | 5 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 9 (4.5) | 188 (94.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 13 (6.5) | 20 (10.0) | 167 (83.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 21 (10.5) | 19 (9.5) | 160 (80.0) | 0 (0.0) | 1 |
| `pid_feedforward` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1 (0.5) | 199 (99.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 17 (8.5) | 180 (90.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 20 (10.0) | 173 (86.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 11 (5.5) | 189 (94.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 22 (11.0) | 178 (89.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 59 (29.5) | 141 (70.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 46 (23.0) | 147 (73.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 5 (2.5) | 195 (97.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 198 (99.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 14 (7.0) | 186 (93.0) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 10 (5.0) | 183 (91.5) | 0 (0.0) | 0 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 6 (3.0) | 194 (97.0) | 0 (0.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 190 (95.0) | 7 (3.5) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 14 (7.0) | 153 (76.5) | 30 (15.0) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 9 (4.5) | 97 (48.5) | 93 (46.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 196 (98.0) | 0 (0.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 4 (2.0) | 192 (96.0) | 4 (2.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 8 (4.0) | 167 (83.5) | 25 (12.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (1.0) | 121 (60.5) | 77 (38.5) | 0 |

### `noise/sigma2cm_lat2step` (σp 2 cm, σv 0.1 m/s, latency 2 step = 66.7 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **99.5** [98.3, 100.0]; seeds 98.0–100.0; N 5×200=1000 | **99.3** [98.7, 99.8]; seeds 98.5–100.0; N 5×200=1000 | **98.5** [97.3, 99.3]; seeds 97.0–99.5; N 5×200=1000 | **95.0** [94.5, 96.8]; seeds 94.5–97.5; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **61.8** [52.7, 78.8]; seeds 52.0–84.0; N 5×200=1000 | **57.2** [48.7, 71.7]; seeds 47.5–75.0; N 5×200=1000 | **47.5** [42.5, 57.7]; seeds 40.5–62.0; N 5×200=1000 | **39.7** [31.3, 52.5]; seeds 28.0–57.5; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **78.3** [74.8, 83.2]; seeds 74.0–84.0; N 5×200=1000 | **79.5** [76.7, 81.8]; seeds 76.5–82.0; N 5×200=1000 | **75.2** [73.5, 77.2]; seeds 73.5–77.5; N 5×200=1000 | **73.2** [67.2, 77.3]; seeds 65.5–78.0; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **100.0** [100.0, 100.0]; seeds 100.0–100.0; N 5×200=1000 | **99.8** [99.5, 100.0]; seeds 99.5–100.0; N 5×200=1000 | **97.2** [96.7, 98.2]; seeds 96.5–98.5; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **84.8** [82.5, 91.3]; seeds 82.0–94.0; N 5×200=1000 | **85.8** [81.0, 88.7]; seeds 80.5–89.0; N 5×200=1000 | **81.7** [80.2, 87.0]; seeds 79.5–89.5; N 5×200=1000 | **79.5** [72.8, 82.3]; seeds 70.0–83.0; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **99.3** [98.0, 99.8]; seeds 97.5–100.0; N 5×200=1000 | **99.2** [97.8, 99.8]; seeds 97.5–100.0; N 5×200=1000 | **97.3** [93.8, 98.8]; seeds 93.0–99.0; N 5×200=1000 | **93.7** [92.0, 94.7]; seeds 91.5–95.0; N 5×200=1000 |
| `pid_track_descend` | 81.0 [75.0, 85.8] 162/200 | 78.5 [72.3, 83.6] 157/200 | 72.5 [65.9, 78.2] 145/200 | 66.5 [59.7, 72.7] 133/200 |
| `pid_feedforward` | 63.0 [56.1, 69.4] 126/200 | 64.5 [57.7, 70.8] 129/200 | 61.5 [54.6, 68.0] 123/200 | 48.5 [41.7, 55.4] 97/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 61.5 [54.6, 68.0] 123/200 | 56.5 [49.6, 63.2] 113/200 | 56.5 [49.6, 63.2] 113/200 | 43.5 [36.8, 50.4] 87/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 90.5 [85.6, 93.8] 181/200 | 88.5 [83.3, 92.2] 177/200 | 91.5 [86.8, 94.6] 183/200 | 79.5 [73.4, 84.5] 159/200 |
| `gated` | 13.0 [9.0, 18.4] 26/200 | 13.5 [9.4, 18.9] 27/200 | 2.0 [0.8, 5.0] 4/200 | 1.5 [0.5, 4.3] 3/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 29.5 [23.6, 36.2] 59/200 | 26.0 [20.4, 32.5] 52/200 | 12.0 [8.2, 17.2] 24/200 | 5.0 [2.7, 9.0] 10/200 |

### `noise/sigma2cm_lat2step` (σp 2 cm, σv 0.1 m/s, latency 2 step = 66.7 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.360 [0.354, 0.363] · 0.0 · 0.0 | 0.391 [0.375, 0.401] · 0.0 · 0.0 | 0.369 [0.355, 0.387] · 0.0 · 0.0 | 0.394 [0.382, 0.402] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.702 [0.582, 0.788] · 0.4 · 0.0 | 0.716 [0.631, 0.845] · 0.8 · 0.0 | 0.780 [0.675, 0.805] · 2.1 · 0.1 | 0.822 [0.745, 0.835] · 4.2 · 0.1 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.383 [0.352, 0.390] · 0.2 · 0.0 | 0.380 [0.374, 0.387] · 0.4 · 0.0 | 0.405 [0.391, 0.425] · 0.4 · 0.0 | 0.402 [0.386, 0.417] · 0.1 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.319 [0.311, 0.327] · 0.0 · 0.0 | 0.329 [0.319, 0.339] · 0.0 · 0.0 | 0.330 [0.321, 0.334] · 0.0 · 0.0 | 0.336 [0.327, 0.340] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.361 [0.354, 0.371] · 0.1 · 0.0 | 0.372 [0.354, 0.379] · 0.1 · 0.0 | 0.371 [0.365, 0.378] · 0.0 · 0.0 | 0.377 [0.372, 0.384] · 0.1 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.374 [0.367, 0.383] · 0.0 · 0.0 | 0.397 [0.393, 0.409] · 0.0 · 0.0 | 0.387 [0.377, 0.402] · 0.0 · 0.0 | 0.405 [0.400, 0.413] · 0.0 · 0.0 |
| `pid_track_descend` | 0.309 · 0.0 · 0.0 | 0.424 · 0.0 · 0.0 | 0.482 · 0.0 · 0.0 | 0.527 · 0.0 · 0.0 |
| `pid_feedforward` | 0.331 · 0.5 · 0.0 | 0.317 · 0.5 · 0.0 | 0.414 · 0.0 · 0.0 | 0.365 · 0.0 · 0.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.209 · 0.0 · 0.0 | 0.245 · 0.0 · 0.0 | 0.329 · 0.0 · 0.0 | 0.286 · 0.0 · 0.0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.209 · 0.0 · 0.0 | 0.245 · 0.0 · 0.0 | 0.329 · 0.0 · 0.0 | 0.286 · 0.0 · 0.0 |
| `gated` | 0.296 · 0.0 · 67.0 | 0.356 · 0.0 · 71.5 | 0.316 · 0.0 · 94.0 | 0.265 · 0.0 · 98.5 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.329 · 0.5 · 39.5 | 0.338 · 0.0 · 42.5 | 0.321 · 0.0 · 69.0 | 0.281 · 0.0 · 87.5 |

### `noise/sigma2cm_lat2step` (σp 2 cm, σv 0.1 m/s, latency 2 step = 66.7 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 7 (0.7) | 993 (99.3) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 2 (0.2) | 5 (0.5) | 993 (99.3) | 0 (0.0) | 0 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 15 (1.5) | 984 (98.4) | 0 (0.0) | 3 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 1 (0.1) | 25 (2.5) | 20 (2.0) | 954 (95.4) | 0 (0.0) | 11 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 4 (0.4) | 50 (5.0) | 297 (29.7) | 6 (0.6) | 643 (64.3) | 0 (0.0) | 94 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 8 (0.8) | 66 (6.6) | 325 (32.5) | 13 (1.3) | 588 (58.8) | 0 (0.0) | 94 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 21 (2.1) | 74 (7.4) | 401 (40.1) | 13 (1.3) | 490 (49.0) | 1 (0.1) | 106 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 42 (4.2) | 125 (12.5) | 408 (40.8) | 15 (1.5) | 409 (40.9) | 1 (0.1) | 103 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 2 (0.2) | 8 (0.8) | 42 (4.2) | 162 (16.2) | 786 (78.6) | 0 (0.0) | 5 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 4 (0.4) | 11 (1.1) | 36 (3.6) | 155 (15.5) | 794 (79.4) | 0 (0.0) | 6 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 4 (0.4) | 13 (1.3) | 80 (8.0) | 150 (15.0) | 753 (75.3) | 0 (0.0) | 6 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 1 (0.1) | 9 (0.9) | 115 (11.5) | 149 (14.9) | 726 (72.6) | 0 (0.0) | 14 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 1000 (100.0) | 0 (0.0) | 0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 2 (0.2) | 998 (99.8) | 0 (0.0) | 1 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 0 (0.0) | 23 (2.3) | 4 (0.4) | 973 (97.3) | 0 (0.0) | 7 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 1 (0.1) | 8 (0.8) | 24 (2.4) | 106 (10.6) | 861 (86.1) | 0 (0.0) | 4 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 1 (0.1) | 11 (1.1) | 36 (3.6) | 98 (9.8) | 854 (85.4) | 0 (0.0) | 3 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 6 (0.6) | 51 (5.1) | 115 (11.5) | 828 (82.8) | 0 (0.0) | 5 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 1 (0.1) | 6 (0.6) | 78 (7.8) | 132 (13.2) | 783 (78.3) | 0 (0.0) | 8 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 0 (0.0) | 1 (0.1) | 1 (0.1) | 7 (0.7) | 991 (99.1) | 0 (0.0) | 2 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 0 (0.0) | 1 (0.1) | 9 (0.9) | 990 (99.0) | 0 (0.0) | 1 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 0 (0.0) | 0 (0.0) | 9 (0.9) | 23 (2.3) | 968 (96.8) | 0 (0.0) | 3 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 0 (0.0) | 3 (0.3) | 41 (4.1) | 21 (2.1) | 935 (93.5) | 0 (0.0) | 7 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 38 (19.0) | 162 (81.0) | 0 (0.0) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 40 (20.0) | 157 (78.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 10 (5.0) | 45 (22.5) | 145 (72.5) | 0 (0.0) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 24 (12.0) | 43 (21.5) | 133 (66.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS3 | 200 | 1 (0.5) | 0 (0.0) | 8 (4.0) | 65 (32.5) | 126 (63.0) | 0 (0.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 1 (0.5) | 1 (0.5) | 9 (4.5) | 60 (30.0) | 129 (64.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 8 (4.0) | 69 (34.5) | 123 (61.5) | 0 (0.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 23 (11.5) | 80 (40.0) | 97 (48.5) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 76 (38.0) | 123 (61.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0 (0.0) | 1 (0.5) | 4 (2.0) | 82 (41.0) | 113 (56.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 84 (42.0) | 113 (56.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 19 (9.5) | 94 (47.0) | 87 (43.5) | 0 (0.0) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 18 (9.0) | 181 (90.5) | 0 (0.0) | 1 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 0 (0.0) | 1 (0.5) | 4 (2.0) | 18 (9.0) | 177 (88.5) | 0 (0.0) | 2 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 3 (1.5) | 14 (7.0) | 183 (91.5) | 0 (0.0) | 6 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 19 (9.5) | 22 (11.0) | 159 (79.5) | 0 (0.0) | 5 |
| `gated` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 1 (0.5) | 39 (19.5) | 26 (13.0) | 134 (67.0) | 0 |
| `gated` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 30 (15.0) | 27 (13.5) | 143 (71.5) | 0 |
| `gated` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 8 (4.0) | 4 (2.0) | 188 (94.0) | 0 |
| `gated` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 3 (1.5) | 197 (98.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 1 (0.5) | 0 (0.0) | 2 (1.0) | 59 (29.5) | 59 (29.5) | 79 (39.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 61 (30.5) | 52 (26.0) | 85 (42.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 0 (0.0) | 0 (0.0) | 2 (1.0) | 36 (18.0) | 24 (12.0) | 138 (69.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 15 (7.5) | 10 (5.0) | 175 (87.5) | 0 |

### `noise/sigma4cm_lat2step` (σp 4 cm, σv 0.2 m/s, latency 2 step = 66.7 ms model): success per sea state

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| `ppo` (learned, PPO) | **80.0** [74.5, 87.8]; seeds 74.0–90.0; N 5×200=1000 | **80.2** [73.5, 84.0]; seeds 71.0–85.5; N 5×200=1000 | **76.5** [67.5, 84.3]; seeds 64.5–87.5; N 5×200=1000 | **73.2** [68.5, 79.5]; seeds 66.5–82.0; N 5×200=1000 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | **20.5** [12.2, 33.7]; seeds 10.0–38.0; N 5×200=1000 | **18.3** [12.8, 33.5]; seeds 12.0–38.5; N 5×200=1000 | **12.5** [10.7, 24.5]; seeds 10.5–30.0; N 5×200=1000 | **9.8** [6.7, 15.2]; seeds 6.0–17.0; N 5×200=1000 |
| `residual_ppo` (learned, residual on pid_feedforward) | **3.3** [2.2, 5.3]; seeds 2.0–6.0; N 5×200=1000 | **4.7** [3.0, 6.2]; seeds 2.5–6.5; N 5×200=1000 | **3.2** [1.0, 4.0]; seeds 0.5–4.0; N 5×200=1000 | **3.7** [3.0, 5.2]; seeds 3.0–5.5; N 5×200=1000 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | **97.0** [95.8, 98.2]; seeds 95.5–98.5; N 5×200=1000 | **95.3** [93.7, 97.2]; seeds 93.5–97.5; N 5×200=1000 | **96.2** [94.5, 97.8]; seeds 94.5–98.0; N 5×200=1000 | **90.3** [87.5, 91.7]; seeds 86.5–92.0; N 5×200=1000 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | **4.2** [2.5, 9.8]; seeds 2.5–11.5; N 5×200=1000 | **5.7** [4.7, 9.5]; seeds 4.5–11.0; N 5×200=1000 | **5.3** [3.3, 6.8]; seeds 3.0–7.0; N 5×200=1000 | **4.0** [2.5, 11.2]; seeds 2.5–13.5; N 5×200=1000 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | **72.8** [61.0, 85.0]; seeds 59.5–86.5; N 5×200=1000 | **70.8** [59.0, 79.5]; seeds 58.5–80.0; N 5×200=1000 | **69.2** [52.7, 79.5]; seeds 49.5–80.0; N 5×200=1000 | **66.7** [50.2, 74.5]; seeds 45.0–76.5; N 5×200=1000 |
| `pid_track_descend` | 16.5 [12.0, 22.3] 33/200 | 24.0 [18.6, 30.4] 48/200 | 20.5 [15.5, 26.6] 41/200 | 21.5 [16.4, 27.7] 43/200 |
| `pid_feedforward` | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.5 [0.1, 2.8] 1/200 | 0.0 [0.0, 1.9] 0/200 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 8.5 [5.4, 13.2] 17/200 | 6.5 [3.8, 10.8] 13/200 | 12.0 [8.2, 17.2] 24/200 | 8.5 [5.4, 13.2] 17/200 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 40.5 [33.9, 47.4] 81/200 | 40.0 [33.5, 46.9] 80/200 | 40.5 [33.9, 47.4] 81/200 | 45.0 [38.3, 51.9] 90/200 |
| `gated` | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 |
| `oracle_gated` — commit-timing oracle (privileged) | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 | 0.0 [0.0, 1.9] 0/200 |

### `noise/sigma4cm_lat2step` (σp 4 cm, σv 0.2 m/s, latency 2 step = 66.7 ms model): p95 closing speed (m/s), crash %, timeout %

| method | SS3: p95 · crash · timeout | SS4: p95 · crash · timeout | SS5: p95 · crash · timeout | SS6: p95 · crash · timeout |
|---|---|---|---|---|
| `ppo` (learned, PPO) | 0.427 [0.418, 0.445] · 0.1 · 0.0 | 0.453 [0.429, 0.472] · 0.2 · 0.0 | 0.444 [0.427, 0.460] · 0.0 · 0.0 | 0.461 [0.444, 0.477] · 0.0 · 0.0 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | 0.902 [0.752, 0.974] · 11.1 · 0.8 | 0.939 [0.759, 1.026] · 11.9 · 0.1 | 0.953 [0.819, 1.043] · 15.3 · 0.3 | 0.953 [0.845, 1.014] · 19.5 · 1.7 |
| `residual_ppo` (learned, residual on pid_feedforward) | 0.545 [0.518, 0.586] · 12.5 · 0.0 | 0.587 [0.572, 0.651] · 12.1 · 0.0 | 0.587 [0.537, 0.615] · 14.4 · 0.0 | 0.611 [0.550, 0.650] · 15.6 · 0.0 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | 0.360 [0.353, 0.375] · 0.1 · 0.0 | 0.369 [0.364, 0.384] · 0.0 · 0.0 | 0.359 [0.356, 0.368] · 0.0 · 0.0 | 0.377 [0.366, 0.391] · 0.0 · 0.0 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | 0.562 [0.508, 0.640] · 7.7 · 0.0 | 0.542 [0.503, 0.564] · 11.3 · 0.0 | 0.607 [0.534, 0.649] · 10.2 · 0.0 | 0.604 [0.558, 0.645] · 11.1 · 0.0 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | 0.444 [0.426, 0.458] · 0.3 · 0.0 | 0.445 [0.435, 0.480] · 0.0 · 0.0 | 0.464 [0.453, 0.470] · 0.1 · 0.0 | 0.479 [0.466, 0.508] · 0.1 · 0.0 |
| `pid_track_descend` | 0.284 · 0.0 · 38.5 | 0.331 · 0.0 · 34.5 | 0.461 · 0.0 · 28.5 | 0.421 · 0.0 · 29.5 |
| `pid_feedforward` | 0.546 · 39.5 · 35.0 | 0.646 · 42.0 · 30.0 | 0.620 · 35.5 · 31.0 | 0.716 · 37.5 · 32.0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | 0.415 · 1.5 · 6.5 | 0.389 · 5.5 · 3.0 | 0.470 · 2.5 · 5.5 | 0.411 · 5.5 · 2.5 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | 0.415 · 1.5 · 6.5 | 0.389 · 5.5 · 3.0 | 0.470 · 2.5 · 5.5 | 0.411 · 5.5 · 2.5 |
| `gated` | – · 59.0 · 41.0 | – · 60.0 · 40.0 | – · 63.0 · 37.0 | 0.632 · 58.0 · 41.5 |
| `oracle_gated` — commit-timing oracle (privileged) | – · 68.0 · 32.0 | – · 60.5 · 39.5 | 0.973 · 68.0 · 31.5 | 0.772 · 60.5 · 38.5 |

### `noise/sigma4cm_lat2step` (σp 4 cm, σv 0.2 m/s, latency 2 step = 66.7 ms model): outcome breakdown (counts, pooled over seeds; % of N)

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | success ∧ tunnelled (> 5 mm) |
|---|---|---|---|---|---|---|---|---|---|
| `ppo` (learned, PPO) | SS3 | 1000 | 1 (0.1) | 49 (4.9) | 71 (7.1) | 71 (7.1) | 808 (80.8) | 0 (0.0) | 32 |
| `ppo` (learned, PPO) | SS4 | 1000 | 2 (0.2) | 38 (3.8) | 83 (8.3) | 83 (8.3) | 794 (79.4) | 0 (0.0) | 37 |
| `ppo` (learned, PPO) | SS5 | 1000 | 0 (0.0) | 44 (4.4) | 93 (9.3) | 100 (10.0) | 763 (76.3) | 0 (0.0) | 37 |
| `ppo` (learned, PPO) | SS6 | 1000 | 0 (0.0) | 44 (4.4) | 143 (14.3) | 77 (7.7) | 736 (73.6) | 0 (0.0) | 43 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS3 | 1000 | 111 (11.1) | 228 (22.8) | 390 (39.0) | 44 (4.4) | 219 (21.9) | 8 (0.8) | 82 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS4 | 1000 | 119 (11.9) | 222 (22.2) | 414 (41.4) | 33 (3.3) | 211 (21.1) | 1 (0.1) | 70 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS5 | 1000 | 153 (15.3) | 274 (27.4) | 382 (38.2) | 32 (3.2) | 156 (15.6) | 3 (0.3) | 57 |
| `sac` (learned, SAC, 2 M env steps (PPO family: 10 M)) | SS6 | 1000 | 195 (19.5) | 317 (31.7) | 341 (34.1) | 25 (2.5) | 105 (10.5) | 17 (1.7) | 48 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS3 | 1000 | 125 (12.5) | 311 (31.1) | 302 (30.2) | 226 (22.6) | 36 (3.6) | 0 (0.0) | 4 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS4 | 1000 | 121 (12.1) | 334 (33.4) | 305 (30.5) | 194 (19.4) | 46 (4.6) | 0 (0.0) | 5 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS5 | 1000 | 144 (14.4) | 322 (32.2) | 301 (30.1) | 205 (20.5) | 28 (2.8) | 0 (0.0) | 3 |
| `residual_ppo` (learned, residual on pid_feedforward) | SS6 | 1000 | 156 (15.6) | 332 (33.2) | 294 (29.4) | 179 (17.9) | 39 (3.9) | 0 (0.0) | 4 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 1 (0.1) | 3 (0.3) | 12 (1.2) | 14 (1.4) | 970 (97.0) | 0 (0.0) | 8 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 0 (0.0) | 8 (0.8) | 14 (1.4) | 24 (2.4) | 954 (95.4) | 0 (0.0) | 3 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 0 (0.0) | 7 (0.7) | 11 (1.1) | 20 (2.0) | 962 (96.2) | 0 (0.0) | 15 |
| `ppo_forecast` (learned, + past-only ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 0 (0.0) | 8 (0.8) | 66 (6.6) | 27 (2.7) | 899 (89.9) | 0 (0.0) | 20 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS3 | 1000 | 77 (7.7) | 317 (31.7) | 313 (31.3) | 240 (24.0) | 53 (5.3) | 0 (0.0) | 2 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS4 | 1000 | 113 (11.3) | 298 (29.8) | 308 (30.8) | 216 (21.6) | 65 (6.5) | 0 (0.0) | 5 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS5 | 1000 | 102 (10.2) | 311 (31.1) | 300 (30.0) | 235 (23.5) | 52 (5.2) | 0 (0.0) | 5 |
| `residual_ppo_forecast` (learned, residual + ship-motion feed (extra ideal sensor)) | SS6 | 1000 | 111 (11.1) | 312 (31.2) | 298 (29.8) | 223 (22.3) | 56 (5.6) | 0 (0.0) | 8 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS3 | 1000 | 3 (0.3) | 42 (4.2) | 97 (9.7) | 129 (12.9) | 729 (72.9) | 0 (0.0) | 28 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS4 | 1000 | 0 (0.0) | 60 (6.0) | 106 (10.6) | 132 (13.2) | 702 (70.2) | 0 (0.0) | 22 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS5 | 1000 | 1 (0.1) | 53 (5.3) | 116 (11.6) | 156 (15.6) | 674 (67.4) | 0 (0.0) | 29 |
| `ppo_sinusoid` (learned, trained on sinusoid motion only) | SS6 | 1000 | 1 (0.1) | 57 (5.7) | 155 (15.5) | 144 (14.4) | 643 (64.3) | 0 (0.0) | 57 |
| `pid_track_descend` | SS3 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 90 (45.0) | 33 (16.5) | 77 (38.5) | 0 |
| `pid_track_descend` | SS4 | 200 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 83 (41.5) | 48 (24.0) | 69 (34.5) | 0 |
| `pid_track_descend` | SS5 | 200 | 0 (0.0) | 0 (0.0) | 7 (3.5) | 95 (47.5) | 41 (20.5) | 57 (28.5) | 0 |
| `pid_track_descend` | SS6 | 200 | 0 (0.0) | 0 (0.0) | 6 (3.0) | 92 (46.0) | 43 (21.5) | 59 (29.5) | 0 |
| `pid_feedforward` | SS3 | 200 | 79 (39.5) | 20 (10.0) | 20 (10.0) | 11 (5.5) | 0 (0.0) | 70 (35.0) | 0 |
| `pid_feedforward` | SS4 | 200 | 84 (42.0) | 28 (14.0) | 18 (9.0) | 10 (5.0) | 0 (0.0) | 60 (30.0) | 0 |
| `pid_feedforward` | SS5 | 200 | 71 (35.5) | 32 (16.0) | 22 (11.0) | 12 (6.0) | 1 (0.5) | 62 (31.0) | 0 |
| `pid_feedforward` | SS6 | 200 | 75 (37.5) | 38 (19.0) | 14 (7.0) | 9 (4.5) | 0 (0.0) | 64 (32.0) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 3 (1.5) | 38 (19.0) | 42 (21.0) | 87 (43.5) | 17 (8.5) | 13 (6.5) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 11 (5.5) | 46 (23.0) | 40 (20.0) | 84 (42.0) | 13 (6.5) | 6 (3.0) | 2 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 5 (2.5) | 42 (21.0) | 49 (24.5) | 69 (34.5) | 24 (12.0) | 11 (5.5) | 0 |
| `pid_feedforward_lowvz` (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 11 (5.5) | 37 (18.5) | 45 (22.5) | 85 (42.5) | 17 (8.5) | 5 (2.5) | 0 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS3 | 200 | 3 (1.5) | 38 (19.0) | 42 (21.0) | 23 (11.5) | 81 (40.5) | 13 (6.5) | 16 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS4 | 200 | 11 (5.5) | 46 (23.0) | 40 (20.0) | 17 (8.5) | 80 (40.0) | 6 (3.0) | 19 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS5 | 200 | 5 (2.5) | 42 (21.0) | 49 (24.5) | 12 (6.0) | 81 (40.5) | 11 (5.5) | 29 |
| `pid_feedforward_lowvz_cut` (lowvz + latched post-contact throttle cut (P5-D2); not the H1a reference) | SS6 | 200 | 11 (5.5) | 37 (18.5) | 45 (22.5) | 12 (6.0) | 90 (45.0) | 5 (2.5) | 35 |
| `gated` | SS3 | 200 | 118 (59.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 82 (41.0) | 0 |
| `gated` | SS4 | 200 | 120 (60.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 80 (40.0) | 0 |
| `gated` | SS5 | 200 | 126 (63.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 74 (37.0) | 0 |
| `gated` | SS6 | 200 | 116 (58.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 83 (41.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS3 | 200 | 136 (68.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 64 (32.0) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS4 | 200 | 121 (60.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 79 (39.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS5 | 200 | 136 (68.0) | 1 (0.5) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 63 (31.5) | 0 |
| `oracle_gated` — commit-timing oracle (privileged) | SS6 | 200 | 121 (60.5) | 2 (1.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 77 (38.5) | 0 |
