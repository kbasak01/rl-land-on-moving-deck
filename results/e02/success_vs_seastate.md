# e02 — baselines, gated and forecast-gated controllers, aft pad and pad at CG: success versus sea state (frozen episode lists)

- Simulation only (PyBullet); no real flight and no real deck data.
- Deck motion is dmf's 3-DOF (heave, roll, pitch) JONSWAP response, Froude-scaled to a Crazyflie at lambda = 1/25 (1 s model = 5 s full scale).
- State-based observations; the perception-noise stand-in is disabled (`configs/env/noise.yaml: enabled: false`); not vision.
- dmf's roll/pitch-heave phase defect (~90 deg) is carried, not fixed. The aft pad (primary) is sensitive to it; the pad-at-CG control arm flies the same frozen episodes with the pad at the ship's CG, where the pad's v_z is the heave rate and the defect does not enter (P1-D2). The static-pad list has no lever arm, so its aft and CG rows fly the identical deck.
- Success = all four frozen criteria (`configs/env/success.yaml`); rates in %, Wilson 95 % CI in brackets, then k/N. Success is never pooled across sea states.
- `in-dist` is the fraction of the cell's episodes whose grid cell is in the development pool (P3-D2); `id` SS6 and every 90 deg episode are outside it.
- `td n` is the number of episodes that touched down (contact detector).
- `quiet | td` (the column `quiet \| td`) is, **of the touched-down episodes only**, the fraction whose TRUE deck satisfied the permissive quiescence predicate (`rld.control.quiescence.QuiescenceRule`: 12 samples 1/30 s apart, |roll| <= 3.0 deg, |pitch| <= 2.0 deg, |pad v_z| <= 0.16 m/s model) starting at the contact touchdown; evaluation ground truth, never a controller input; Wilson 95 % CI in brackets, then k/`td n`. `–` when nothing touched down. It is conditional on touching down: a controller that rarely touches down can score high here while landing quietly no more often than another.
- `quiet landings / listed` is the number of quiet touchdowns divided by the number of listed episodes in the cell (timeouts count as not quiet); Wilson 95 % CI in brackets, then k/N. This is the per-episode rate to compare across controllers.
- `oracle_gated` is privileged: it reads the true future deck motion. It bounds commit timing under the gated rule, not success, and is never a deployable result.
- `gated_forecast` and `gated_forecast_tcn` are not privileged: they read the observation plus the past-only history of dmf's six clean ship channels (an ideal noise-free, zero-latency ship motion reference unit that never reaches past the runner's clock) and a dmf forecaster fitted on the P3-D2 dev pool (P4-D1, P4-D3).
- Pad: `aft` is the frozen primary; `cg` is the pad-at-CG control arm, flown on the identical listed episodes (same t0, realization and initial state).
- Cells not run (never dropped; listed here and marked in the tables):
  - gated_forecast, pad aft, static static: 200 of 200 not run: the static-pad list has no ship motion to feed a forecaster (StaticDeckMotion has no vessel or ship channels, and its t0 goes down to 0.51 s model, inside the 4.0 s model lookback)
  - gated_forecast_tcn, pad aft, static static: 200 of 200 not run: the static-pad list has no ship motion to feed a forecaster (StaticDeckMotion has no vessel or ship channels, and its t0 goes down to 0.51 s model, inside the 4.0 s model lookback)
  - gated_forecast, pad cg, static static: 200 of 200 not run: the static-pad list has no ship motion to feed a forecaster (StaticDeckMotion has no vessel or ship channels, and its t0 goes down to 0.51 s model, inside the 4.0 s model lookback)
  - gated_forecast_tcn, pad cg, static static: 200 of 200 not run: the static-pad list has no ship motion to feed a forecaster (StaticDeckMotion has no vessel or ship channels, and its t0 goes down to 0.51 s model, inside the 4.0 s model lookback)
- Rendered from `summary.csv` by `rld.eval.report`; do not edit by hand.

## id — aft pad (primary, as frozen in P3-D1)

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 96.0 [92.3, 98.0] (192/200) | 85.0 [79.4, 89.3] (170/200) | 80.0 [73.9, 85.0] (160/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 90.5 [85.6, 93.8] (181/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) | 95.5 [91.7, 97.6] (191/200) | 85.0 [79.4, 89.3] (170/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 89.5 [84.5, 93.0] (179/200) | 63.0 [56.1, 69.4] (126/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 89.0 [83.9, 92.6] (178/200) | 64.5 [57.7, 70.8] (129/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | 100.0 [98.1, 100.0] (200/200) | 95.0 [91.0, 97.3] (190/200) | 60.0 [53.1, 66.5] (120/200) | 16.5 [12.0, 22.3] (33/200) |

### id — aft pad (primary, as frozen in P3-D1): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.238 / 0.308 | 0.309 | 0.040 | 3.9 | 4.62 | 0/200 | 0 | 200 | 0.940 [0.898, 0.965] (188/200) | 0.940 [0.898, 0.965] (188/200) | 0.755 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.020 | 0.020 | 0.960 | 0.000 | 0.254 / 0.412 | 0.412 | 0.042 | 4.8 | 4.59 | 0/200 | 0 | 200 | 0.725 [0.659, 0.782] (145/200) | 0.725 [0.659, 0.782] (145/200) | 0.750 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.060 | 0.090 | 0.850 | 0.000 | 0.303 / 0.525 | 0.525 | 0.037 | 7.5 | 4.54 | 0/200 | 4 | 200 | 0.315 [0.255, 0.382] (63/200) | 0.315 [0.255, 0.382] (63/200) | 0.745 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.085 | 0.115 | 0.800 | 0.000 | 0.287 / 0.502 | 0.502 | 0.040 | 13.9 | 4.65 | 0/200 | 2 | 200 | 0.100 [0.066, 0.149] (20/200) | 0.100 [0.066, 0.149] (20/200) | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.219 | 0.219 | 0.032 | 4.4 | 4.50 | 0/200 | 0 | 200 | 0.950 [0.910, 0.973] (190/200) | 0.950 [0.910, 0.973] (190/200) | 0.755 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.189 / 0.226 | 0.226 | 0.034 | 5.1 | 4.49 | 0/200 | 0 | 200 | 0.720 [0.654, 0.778] (144/200) | 0.720 [0.654, 0.778] (144/200) | 0.750 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.005 | 0.005 | 0.990 | 0.000 | 0.192 / 0.262 | 0.262 | 0.033 | 8.4 | 4.41 | 0/200 | 0 | 200 | 0.290 [0.232, 0.356] (58/200) | 0.290 [0.232, 0.356] (58/200) | 0.745 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.040 | 0.055 | 0.905 | 0.000 | 0.193 / 0.264 | 0.264 | 0.031 | 13.4 | 4.55 | 0/200 | 0 | 200 | 0.080 [0.050, 0.126] (16/200) | 0.080 [0.050, 0.126] (16/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.103 / 0.127 | 0.127 | 0.031 | 4.5 | 7.68 | 0/200 | 0 | 200 | 0.940 [0.898, 0.965] (188/200) | 0.940 [0.898, 0.965] (188/200) | 0.755 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.980 | 0.000 | 0.103 / 0.139 | 0.140 | 0.032 | 5.2 | 7.65 | 0/200 | 0 | 200 | 0.740 [0.675, 0.796] (148/200) | 0.740 [0.675, 0.796] (148/200) | 0.750 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.045 | 0.955 | 0.000 | 0.110 / 0.188 | 0.185 | 0.032 | 9.2 | 7.61 | 0/200 | 0 | 200 | 0.280 [0.222, 0.346] (56/200) | 0.280 [0.222, 0.346] (56/200) | 0.745 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.045 | 0.105 | 0.850 | 0.000 | 0.106 / 0.181 | 0.182 | 0.034 | 13.9 | 7.77 | 0/200 | 0 | 200 | 0.115 [0.078, 0.167] (23/200) | 0.115 [0.078, 0.167] (23/200) | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.217 | 0.216 | 0.044 | 4.7 | 3.18 | 0/200 | 0 | 200 | 0.950 [0.910, 0.973] (190/200) | 0.950 [0.910, 0.973] (190/200) | 0.755 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.192 / 0.229 | 0.229 | 0.045 | 4.9 | 3.28 | 0/200 | 0 | 197 | 0.741 [0.676, 0.797] (146/197) | 0.730 [0.665, 0.787] (146/200) | 0.750 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.895 | 0.105 | 0.193 / 0.250 | 0.249 | 0.039 | 6.3 | 3.74 | 0/200 | 0 | 179 | 0.430 [0.360, 0.503] (77/179) | 0.385 [0.320, 0.454] (77/200) | 0.745 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.630 | 0.350 | 0.194 / 0.267 | 0.267 | 0.038 | 7.8 | 4.66 | 0/200 | 0 | 130 | 0.169 [0.114, 0.243] (22/130) | 0.110 [0.074, 0.161] (22/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.220 | 0.219 | 0.044 | 4.6 | 3.19 | 0/200 | 0 | 200 | 0.985 [0.957, 0.995] (197/200) | 0.985 [0.957, 0.995] (197/200) | 0.755 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.194 / 0.236 | 0.237 | 0.044 | 4.4 | 3.29 | 0/200 | 0 | 197 | 0.909 [0.860, 0.941] (179/197) | 0.895 [0.845, 0.930] (179/200) | 0.750 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.890 | 0.105 | 0.191 / 0.252 | 0.252 | 0.042 | 5.1 | 3.60 | 0/200 | 0 | 179 | 0.771 [0.704, 0.826] (138/179) | 0.690 [0.623, 0.750] (138/200) | 0.745 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.645 | 0.355 | 0.190 / 0.254 | 0.253 | 0.043 | 5.9 | 4.80 | 0/200 | 0 | 129 | 0.682 [0.598, 0.756] (88/129) | 0.440 [0.373, 0.509] (88/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.755 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.750 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.745 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.217 | 0.216 | 0.044 | 4.6 | 3.19 | 0/200 | 0 | 200 | 0.995 [0.972, 0.999] (199/200) | 0.995 [0.972, 0.999] (199/200) | 0.755 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.950 | 0.050 | 0.191 / 0.222 | 0.221 | 0.044 | 4.9 | 3.43 | 0/200 | 0 | 190 | 0.916 [0.868, 0.948] (174/190) | 0.870 [0.816, 0.910] (174/200) | 0.750 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.600 | 0.400 | 0.188 / 0.229 | 0.230 | 0.036 | 4.6 | 4.72 | 0/200 | 0 | 120 | 0.750 [0.666, 0.819] (90/120) | 0.450 [0.383, 0.519] (90/200) | 0.745 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.165 | 0.835 | 0.186 / 0.204 | 0.203 | 0.037 | 5.7 | 5.62 | 0/200 | 0 | 33 | 0.636 [0.466, 0.778] (21/33) | 0.105 [0.070, 0.155] (21/200) | 0.000 |

## unseen_seastate — aft pad (primary, as frozen in P3-D1)

| method | SS6 |
|---|---|
| pid_track_descend | 81.0 [75.0, 85.8] (162/200) |
| pid_feedforward | 90.0 [85.1, 93.4] (180/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 86.0 [80.5, 90.1] (172/200) |
| gated | 64.5 [57.7, 70.8] (129/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 65.0 [58.2, 71.3] (130/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | 0.0 [0.0, 1.9] (0/200) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | 21.0 [15.9, 27.2] (42/200) |

### unseen_seastate — aft pad (primary, as frozen in P3-D1): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.075 | 0.115 | 0.810 | 0.000 | 0.282 / 0.497 | 0.498 | 0.039 | 14.3 | 4.49 | 1/200 | 1 | 200 | 0.120 [0.082, 0.172] (24/200) | 0.120 [0.082, 0.172] (24/200) | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.030 | 0.070 | 0.900 | 0.000 | 0.193 / 0.250 | 0.247 | 0.036 | 13.1 | 4.46 | 1/200 | 0 | 200 | 0.105 [0.070, 0.155] (21/200) | 0.105 [0.070, 0.155] (21/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.035 | 0.105 | 0.860 | 0.000 | 0.107 / 0.152 | 0.155 | 0.035 | 13.8 | 7.66 | 1/200 | 0 | 200 | 0.080 [0.050, 0.126] (16/200) | 0.080 [0.050, 0.126] (16/200) | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.010 | 0.645 | 0.345 | 0.195 / 0.243 | 0.244 | 0.040 | 7.2 | 4.93 | 0/200 | 0 | 131 | 0.214 [0.152, 0.292] (28/131) | 0.140 [0.099, 0.195] (28/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.650 | 0.350 | 0.195 / 0.240 | 0.241 | 0.035 | 5.5 | 4.37 | 0/200 | 0 | 130 | 0.662 [0.577, 0.737] (86/130) | 0.430 [0.363, 0.499] (86/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.210 | 0.790 | 0.186 / 0.218 | 0.220 | 0.029 | 6.5 | 6.59 | 0/200 | 0 | 42 | 0.595 [0.445, 0.730] (25/42) | 0.125 [0.086, 0.178] (25/200) | 0.000 |

## unseen_heading — aft pad (primary, as frozen in P3-D1)

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 97.5 [94.3, 98.9] (195/200) | 72.5 [65.9, 78.2] (145/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 77.0 [70.7, 82.3] (154/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 96.5 [93.0, 98.3] (193/200) | 70.0 [63.3, 75.9] (140/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) | 42.0 [35.4, 48.9] (84/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 46.5 [39.7, 53.4] (93/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 96.5 [93.0, 98.3] (193/200) | 22.0 [16.8, 28.2] (44/200) |

### unseen_heading — aft pad (primary, as frozen in P3-D1): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.232 / 0.258 | 0.258 | 0.040 | 4.0 | 4.71 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.240 / 0.313 | 0.313 | 0.040 | 4.5 | 4.64 | 0/200 | 0 | 200 | 0.920 [0.874, 0.950] (184/200) | 0.920 [0.874, 0.950] (184/200) | 0.000 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.025 | 0.975 | 0.000 | 0.264 / 0.370 | 0.368 | 0.041 | 7.0 | 4.64 | 0/200 | 0 | 200 | 0.400 [0.335, 0.469] (80/200) | 0.400 [0.335, 0.469] (80/200) | 0.000 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.110 | 0.165 | 0.725 | 0.000 | 0.285 / 0.454 | 0.463 | 0.044 | 18.1 | 4.54 | 0/200 | 5 | 200 | 0.060 [0.035, 0.102] (12/200) | 0.060 [0.035, 0.102] (12/200) | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.192 / 0.205 | 0.205 | 0.031 | 4.7 | 4.70 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.212 | 0.212 | 0.033 | 5.2 | 4.54 | 0/200 | 0 | 200 | 0.915 [0.868, 0.946] (183/200) | 0.915 [0.868, 0.946] (183/200) | 0.000 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.995 | 0.000 | 0.191 / 0.222 | 0.221 | 0.031 | 7.9 | 4.62 | 0/200 | 0 | 200 | 0.405 [0.339, 0.474] (81/200) | 0.405 [0.339, 0.474] (81/200) | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.120 | 0.110 | 0.770 | 0.000 | 0.188 / 0.230 | 0.231 | 0.033 | 17.7 | 4.51 | 0/200 | 0 | 200 | 0.055 [0.031, 0.096] (11/200) | 0.055 [0.031, 0.096] (11/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.103 / 0.111 | 0.111 | 0.035 | 4.6 | 8.00 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.010 | 0.990 | 0.000 | 0.102 / 0.117 | 0.118 | 0.033 | 5.4 | 7.85 | 0/200 | 0 | 200 | 0.900 [0.851, 0.934] (180/200) | 0.900 [0.851, 0.934] (180/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.035 | 0.965 | 0.000 | 0.104 / 0.130 | 0.130 | 0.035 | 7.3 | 7.82 | 0/200 | 0 | 200 | 0.355 [0.292, 0.423] (71/200) | 0.355 [0.292, 0.423] (71/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.120 | 0.180 | 0.700 | 0.000 | 0.102 / 0.139 | 0.139 | 0.034 | 18.7 | 7.78 | 0/200 | 0 | 200 | 0.050 [0.027, 0.090] (10/200) | 0.050 [0.027, 0.090] (10/200) | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.047 | 4.5 | 3.25 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.210 | 0.209 | 0.047 | 5.3 | 3.23 | 0/200 | 0 | 200 | 0.885 [0.833, 0.922] (177/200) | 0.885 [0.833, 0.922] (177/200) | 0.000 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.980 | 0.015 | 0.192 / 0.219 | 0.220 | 0.043 | 6.2 | 3.50 | 0/200 | 0 | 197 | 0.508 [0.438, 0.577] (100/197) | 0.500 [0.431, 0.569] (100/200) | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.035 | 0.420 | 0.545 | 0.189 / 0.218 | 0.218 | 0.036 | 7.3 | 5.36 | 0/200 | 0 | 91 | 0.253 [0.175, 0.351] (23/91) | 0.115 [0.078, 0.167] (23/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.047 | 4.5 | 3.25 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.210 | 0.210 | 0.049 | 5.3 | 3.25 | 0/200 | 0 | 200 | 0.985 [0.957, 0.995] (197/200) | 0.985 [0.957, 0.995] (197/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.995 | 0.005 | 0.197 / 0.220 | 0.221 | 0.042 | 5.5 | 3.61 | 0/200 | 0 | 199 | 0.789 [0.727, 0.840] (157/199) | 0.785 [0.723, 0.836] (157/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.465 | 0.535 | 0.191 / 0.215 | 0.215 | 0.034 | 6.3 | 5.60 | 0/200 | 0 | 93 | 0.634 [0.533, 0.725] (59/93) | 0.295 [0.236, 0.362] (59/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.047 | 4.5 | 3.25 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.213 | 0.213 | 0.046 | 5.2 | 3.31 | 0/200 | 0 | 200 | 0.935 [0.892, 0.962] (187/200) | 0.935 [0.892, 0.962] (187/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.965 | 0.030 | 0.195 / 0.220 | 0.221 | 0.044 | 5.8 | 3.90 | 0/200 | 0 | 194 | 0.701 [0.633, 0.761] (136/194) | 0.680 [0.612, 0.741] (136/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.220 | 0.780 | 0.187 / 0.220 | 0.219 | 0.029 | 5.6 | 5.63 | 0/200 | 0 | 44 | 0.409 [0.277, 0.556] (18/44) | 0.090 [0.058, 0.138] (18/200) | 0.000 |

## unseen_vessel — aft pad (primary, as frozen in P3-D1)

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 97.5 [94.3, 98.9] (195/200) | 90.0 [85.1, 93.4] (180/200) | 81.5 [75.5, 86.3] (163/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 98.5 [95.7, 99.5] (197/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 95.0 [91.0, 97.3] (190/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 93.5 [89.2, 96.2] (187/200) | 90.0 [85.1, 93.4] (180/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 95.0 [91.0, 97.3] (190/200) | 89.0 [83.9, 92.6] (178/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) | 0.0 [0.0, 1.9] (0/200) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 71.5 [64.9, 77.3] (143/200) | 16.0 [11.6, 21.7] (32/200) |

### unseen_vessel — aft pad (primary, as frozen in P3-D1): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.235 / 0.300 | 0.300 | 0.040 | 4.0 | 4.56 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.025 | 0.975 | 0.000 | 0.253 / 0.388 | 0.388 | 0.040 | 4.5 | 4.69 | 0/200 | 0 | 200 | 0.815 [0.755, 0.863] (163/200) | 0.815 [0.755, 0.863] (163/200) | 0.000 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.070 | 0.030 | 0.900 | 0.000 | 0.292 / 0.520 | 0.521 | 0.038 | 4.5 | 4.56 | 0/200 | 2 | 200 | 0.540 [0.471, 0.608] (108/200) | 0.540 [0.471, 0.608] (108/200) | 0.000 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.100 | 0.085 | 0.815 | 0.000 | 0.312 / 0.546 | 0.546 | 0.039 | 8.0 | 4.67 | 0/200 | 0 | 200 | 0.200 [0.150, 0.261] (40/200) | 0.200 [0.150, 0.261] (40/200) | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.211 | 0.210 | 0.032 | 4.4 | 4.53 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.228 | 0.228 | 0.034 | 4.4 | 4.55 | 0/200 | 0 | 200 | 0.815 [0.755, 0.863] (163/200) | 0.815 [0.755, 0.863] (163/200) | 0.000 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.995 | 0.000 | 0.195 / 0.255 | 0.255 | 0.033 | 5.8 | 4.49 | 0/200 | 0 | 200 | 0.525 [0.456, 0.593] (105/200) | 0.525 [0.456, 0.593] (105/200) | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.005 | 0.010 | 0.985 | 0.000 | 0.193 / 0.260 | 0.260 | 0.033 | 8.8 | 4.57 | 1/200 | 0 | 200 | 0.205 [0.155, 0.266] (41/200) | 0.205 [0.155, 0.266] (41/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.102 / 0.120 | 0.120 | 0.033 | 4.5 | 7.78 | 0/200 | 0 | 200 | 0.985 [0.957, 0.995] (197/200) | 0.985 [0.957, 0.995] (197/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.104 / 0.138 | 0.138 | 0.035 | 4.4 | 7.76 | 0/200 | 0 | 200 | 0.830 [0.772, 0.876] (166/200) | 0.830 [0.772, 0.876] (166/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.995 | 0.000 | 0.108 / 0.155 | 0.155 | 0.034 | 4.8 | 7.75 | 0/200 | 0 | 200 | 0.500 [0.431, 0.569] (100/200) | 0.500 [0.431, 0.569] (100/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.050 | 0.950 | 0.000 | 0.110 / 0.159 | 0.158 | 0.033 | 7.9 | 7.90 | 0/200 | 0 | 200 | 0.190 [0.142, 0.250] (38/200) | 0.190 [0.142, 0.250] (38/200) | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.216 | 0.216 | 0.048 | 4.5 | 3.17 | 0/200 | 0 | 200 | 0.995 [0.972, 0.999] (199/200) | 0.995 [0.972, 0.999] (199/200) | 0.000 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.192 / 0.227 | 0.228 | 0.046 | 4.9 | 3.25 | 0/200 | 0 | 200 | 0.815 [0.755, 0.863] (163/200) | 0.815 [0.755, 0.863] (163/200) | 0.000 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.935 | 0.065 | 0.194 / 0.248 | 0.247 | 0.043 | 5.0 | 3.40 | 0/200 | 0 | 187 | 0.604 [0.533, 0.672] (113/187) | 0.565 [0.496, 0.632] (113/200) | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.900 | 0.100 | 0.189 / 0.249 | 0.248 | 0.038 | 6.5 | 4.23 | 1/200 | 0 | 180 | 0.256 [0.197, 0.324] (46/180) | 0.230 [0.177, 0.293] (46/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.218 | 0.218 | 0.048 | 4.5 | 3.17 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.224 | 0.223 | 0.044 | 4.5 | 3.24 | 0/200 | 0 | 200 | 0.910 [0.862, 0.942] (182/200) | 0.910 [0.862, 0.942] (182/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.950 | 0.050 | 0.194 / 0.238 | 0.238 | 0.043 | 5.0 | 3.51 | 0/200 | 0 | 190 | 0.847 [0.789, 0.892] (161/190) | 0.805 [0.745, 0.854] (161/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.890 | 0.110 | 0.193 / 0.235 | 0.235 | 0.035 | 5.4 | 4.81 | 0/200 | 0 | 178 | 0.781 [0.715, 0.835] (139/178) | 0.695 [0.628, 0.755] (139/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | – / – | – | – | – | – | 0/200 | 0 | 0 | – [–, –] (0/0) | 0.000 [0.000, 0.019] (0/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.217 | 0.216 | 0.048 | 4.5 | 3.17 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.193 / 0.223 | 0.223 | 0.042 | 5.0 | 3.42 | 0/200 | 0 | 198 | 0.919 [0.873, 0.950] (182/198) | 0.910 [0.862, 0.942] (182/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.715 | 0.285 | 0.194 / 0.225 | 0.223 | 0.038 | 5.1 | 4.27 | 0/200 | 0 | 143 | 0.839 [0.770, 0.890] (120/143) | 0.600 [0.531, 0.665] (120/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.160 | 0.840 | 0.192 / 0.214 | 0.214 | 0.028 | 7.1 | 6.69 | 0/200 | 0 | 32 | 0.344 [0.204, 0.517] (11/32) | 0.055 [0.031, 0.096] (11/200) | 0.000 |

## static — aft pad (primary, as frozen in P3-D1)

| method | static |
|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) |
| gated | 100.0 [98.1, 100.0] (200/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 100.0 [98.1, 100.0] (200/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | not run (0/200; see above) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | not run (0/200; see above) |

### static — aft pad (primary, as frozen in P3-D1): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.232 / 0.246 | 0.246 | 0.040 | 4.2 | 4.60 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.203 | 0.203 | 0.031 | 4.2 | 4.53 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.101 / 0.109 | 0.109 | 0.031 | 4.1 | 7.77 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.045 | 4.5 | 3.14 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.045 | 4.5 | 3.14 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |

## id — pad at CG (control arm for dmf's phase defect; same frozen episodes)

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 96.0 [92.3, 98.0] (192/200) | 77.0 [70.7, 82.3] (154/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 92.0 [87.4, 95.0] (184/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 95.0 [91.0, 97.3] (190/200) | 82.5 [76.6, 87.1] (165/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 92.0 [87.4, 95.0] (184/200) | 66.0 [59.2, 72.2] (132/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 91.0 [86.2, 94.2] (182/200) | 65.5 [58.7, 71.7] (131/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | 99.5 [97.2, 99.9] (199/200) | 95.5 [91.7, 97.6] (191/200) | 79.0 [72.8, 84.1] (158/200) | 31.5 [25.5, 38.2] (63/200) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 85.0 [79.4, 89.3] (170/200) | 23.5 [18.2, 29.8] (47/200) |

### id — pad at CG (control arm for dmf's phase defect; same frozen episodes): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.229 / 0.263 | 0.262 | 0.045 | 4.6 | 4.70 | 0/200 | 0 | 200 | 0.965 [0.930, 0.983] (193/200) | 0.965 [0.930, 0.983] (193/200) | 0.755 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.995 | 0.000 | 0.230 / 0.307 | 0.307 | 0.040 | 5.0 | 4.58 | 0/200 | 0 | 200 | 0.835 [0.777, 0.880] (167/200) | 0.835 [0.777, 0.880] (167/200) | 0.750 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.005 | 0.035 | 0.960 | 0.000 | 0.262 / 0.403 | 0.405 | 0.037 | 8.3 | 4.56 | 0/200 | 0 | 200 | 0.395 [0.330, 0.464] (79/200) | 0.395 [0.330, 0.464] (79/200) | 0.745 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.095 | 0.135 | 0.770 | 0.000 | 0.291 / 0.507 | 0.507 | 0.043 | 13.7 | 4.61 | 0/200 | 4 | 200 | 0.120 [0.082, 0.172] (24/200) | 0.120 [0.082, 0.172] (24/200) | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.204 | 0.204 | 0.030 | 4.6 | 4.56 | 0/200 | 0 | 200 | 0.970 [0.936, 0.986] (194/200) | 0.970 [0.936, 0.986] (194/200) | 0.755 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.210 | 0.211 | 0.031 | 5.4 | 4.55 | 0/200 | 0 | 200 | 0.840 [0.783, 0.884] (168/200) | 0.840 [0.783, 0.884] (168/200) | 0.750 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.015 | 0.985 | 0.000 | 0.190 / 0.229 | 0.230 | 0.032 | 7.5 | 4.40 | 0/200 | 0 | 200 | 0.420 [0.354, 0.489] (84/200) | 0.420 [0.354, 0.489] (84/200) | 0.745 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.030 | 0.050 | 0.920 | 0.000 | 0.193 / 0.243 | 0.245 | 0.033 | 12.9 | 4.53 | 0/200 | 0 | 200 | 0.125 [0.086, 0.178] (25/200) | 0.125 [0.086, 0.178] (25/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.102 / 0.111 | 0.111 | 0.033 | 4.6 | 7.72 | 0/200 | 0 | 200 | 0.965 [0.930, 0.983] (193/200) | 0.965 [0.930, 0.983] (193/200) | 0.755 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.015 | 0.985 | 0.000 | 0.102 / 0.121 | 0.121 | 0.031 | 5.0 | 7.83 | 0/200 | 0 | 200 | 0.835 [0.777, 0.880] (167/200) | 0.835 [0.777, 0.880] (167/200) | 0.750 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.010 | 0.040 | 0.950 | 0.000 | 0.102 / 0.131 | 0.132 | 0.033 | 8.8 | 7.56 | 0/200 | 0 | 200 | 0.405 [0.339, 0.474] (81/200) | 0.405 [0.339, 0.474] (81/200) | 0.745 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.035 | 0.140 | 0.825 | 0.000 | 0.106 / 0.144 | 0.145 | 0.033 | 13.2 | 7.85 | 0/200 | 0 | 200 | 0.130 [0.090, 0.184] (26/200) | 0.130 [0.090, 0.184] (26/200) | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.205 | 0.204 | 0.051 | 4.7 | 3.18 | 0/200 | 0 | 200 | 0.955 [0.917, 0.976] (191/200) | 0.955 [0.917, 0.976] (191/200) | 0.755 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.193 / 0.212 | 0.213 | 0.044 | 5.0 | 3.20 | 0/200 | 0 | 198 | 0.843 [0.786, 0.887] (167/198) | 0.835 [0.777, 0.880] (167/200) | 0.750 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.920 | 0.075 | 0.191 / 0.225 | 0.225 | 0.042 | 6.1 | 3.55 | 0/200 | 0 | 185 | 0.519 [0.447, 0.590] (96/185) | 0.480 [0.412, 0.549] (96/200) | 0.745 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.660 | 0.320 | 0.194 / 0.231 | 0.232 | 0.035 | 8.3 | 4.77 | 0/200 | 0 | 136 | 0.206 [0.146, 0.281] (28/136) | 0.140 [0.099, 0.195] (28/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.192 / 0.205 | 0.204 | 0.051 | 4.7 | 3.18 | 0/200 | 0 | 200 | 0.995 [0.972, 0.999] (199/200) | 0.995 [0.972, 0.999] (199/200) | 0.755 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.192 / 0.216 | 0.215 | 0.044 | 4.9 | 3.23 | 0/200 | 0 | 198 | 0.944 [0.903, 0.969] (187/198) | 0.935 [0.892, 0.962] (187/200) | 0.750 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.910 | 0.090 | 0.191 / 0.222 | 0.222 | 0.043 | 5.3 | 3.43 | 0/200 | 0 | 182 | 0.819 [0.756, 0.868] (149/182) | 0.745 [0.680, 0.800] (149/200) | 0.745 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.655 | 0.345 | 0.191 / 0.226 | 0.226 | 0.034 | 5.3 | 4.77 | 0/200 | 0 | 131 | 0.695 [0.611, 0.767] (91/131) | 0.455 [0.387, 0.524] (91/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.995 | 0.005 | 0.192 / 0.204 | 0.204 | 0.050 | 4.7 | 3.21 | 0/200 | 0 | 199 | 1.000 [0.981, 1.000] (199/199) | 0.995 [0.972, 0.999] (199/200) | 0.755 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.955 | 0.045 | 0.192 / 0.212 | 0.211 | 0.044 | 4.7 | 3.32 | 0/200 | 0 | 191 | 0.969 [0.933, 0.986] (185/191) | 0.925 [0.880, 0.954] (185/200) | 0.750 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.790 | 0.210 | 0.190 / 0.221 | 0.221 | 0.041 | 5.2 | 4.01 | 0/200 | 0 | 158 | 0.810 [0.742, 0.864] (128/158) | 0.640 [0.571, 0.703] (128/200) | 0.745 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.315 | 0.685 | 0.191 / 0.227 | 0.228 | 0.031 | 5.9 | 5.78 | 0/200 | 0 | 63 | 0.508 [0.388, 0.627] (32/63) | 0.160 [0.116, 0.217] (32/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.192 / 0.205 | 0.205 | 0.050 | 4.7 | 3.18 | 0/200 | 0 | 200 | 0.990 [0.964, 0.997] (198/200) | 0.990 [0.964, 0.997] (198/200) | 0.755 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.193 / 0.216 | 0.215 | 0.044 | 4.9 | 3.29 | 0/200 | 0 | 197 | 0.964 [0.928, 0.983] (190/197) | 0.950 [0.910, 0.973] (190/200) | 0.750 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.850 | 0.150 | 0.193 / 0.219 | 0.220 | 0.038 | 4.9 | 4.14 | 0/200 | 0 | 170 | 0.800 [0.734, 0.853] (136/170) | 0.680 [0.612, 0.741] (136/200) | 0.745 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.235 | 0.765 | 0.192 / 0.209 | 0.208 | 0.034 | 5.0 | 5.64 | 0/200 | 0 | 47 | 0.660 [0.517, 0.778] (31/47) | 0.155 [0.111, 0.212] (31/200) | 0.000 |

## unseen_seastate — pad at CG (control arm for dmf's phase defect; same frozen episodes)

| method | SS6 |
|---|---|
| pid_track_descend | 83.5 [77.7, 88.0] (167/200) |
| pid_feedforward | 92.0 [87.4, 95.0] (184/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 88.5 [83.3, 92.2] (177/200) |
| gated | 63.5 [56.6, 69.9] (127/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 65.5 [58.7, 71.7] (131/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | 33.5 [27.3, 40.3] (67/200) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | 30.0 [24.1, 36.7] (60/200) |

### unseen_seastate — pad at CG (control arm for dmf's phase defect; same frozen episodes): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.045 | 0.120 | 0.835 | 0.000 | 0.283 / 0.466 | 0.481 | 0.039 | 13.3 | 4.57 | 0/200 | 0 | 200 | 0.155 [0.111, 0.212] (31/200) | 0.155 [0.111, 0.212] (31/200) | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.030 | 0.050 | 0.920 | 0.000 | 0.193 / 0.234 | 0.234 | 0.032 | 12.5 | 4.42 | 0/200 | 0 | 200 | 0.130 [0.090, 0.184] (26/200) | 0.130 [0.090, 0.184] (26/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.040 | 0.075 | 0.885 | 0.000 | 0.106 / 0.143 | 0.145 | 0.034 | 14.3 | 7.58 | 0/200 | 0 | 200 | 0.125 [0.086, 0.178] (25/200) | 0.125 [0.086, 0.178] (25/200) | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.635 | 0.345 | 0.190 / 0.230 | 0.231 | 0.043 | 8.3 | 4.36 | 0/200 | 0 | 131 | 0.275 [0.206, 0.357] (36/131) | 0.180 [0.133, 0.239] (36/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.655 | 0.345 | 0.186 / 0.219 | 0.219 | 0.041 | 5.3 | 4.33 | 0/200 | 0 | 131 | 0.679 [0.595, 0.753] (89/131) | 0.445 [0.378, 0.514] (89/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.335 | 0.665 | 0.187 / 0.214 | 0.215 | 0.035 | 5.1 | 5.30 | 0/200 | 0 | 67 | 0.567 [0.448, 0.679] (38/67) | 0.190 [0.142, 0.250] (38/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.300 | 0.700 | 0.192 / 0.211 | 0.212 | 0.031 | 4.4 | 6.31 | 0/200 | 0 | 60 | 0.567 [0.441, 0.684] (34/60) | 0.170 [0.124, 0.228] (34/200) | 0.000 |

## unseen_heading — pad at CG (control arm for dmf's phase defect; same frozen episodes)

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 70.5 [63.8, 76.4] (141/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 76.0 [69.6, 81.4] (152/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 95.5 [91.7, 97.6] (191/200) | 70.5 [63.8, 76.4] (141/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 41.0 [34.4, 47.9] (82/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 44.5 [37.8, 51.4] (89/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) | 25.0 [19.5, 31.4] (50/200) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) | 28.5 [22.7, 35.1] (57/200) |

### unseen_heading — pad at CG (control arm for dmf's phase defect; same frozen episodes): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.236 / 0.265 | 0.265 | 0.040 | 4.1 | 4.69 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.246 / 0.329 | 0.332 | 0.040 | 4.9 | 4.63 | 0/200 | 0 | 200 | 0.915 [0.868, 0.946] (183/200) | 0.915 [0.868, 0.946] (183/200) | 0.000 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.010 | 0.990 | 0.000 | 0.266 / 0.386 | 0.389 | 0.042 | 6.8 | 4.66 | 0/200 | 0 | 200 | 0.405 [0.339, 0.474] (81/200) | 0.405 [0.339, 0.474] (81/200) | 0.000 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.135 | 0.160 | 0.705 | 0.000 | 0.295 / 0.489 | 0.497 | 0.042 | 17.6 | 4.53 | 0/200 | 5 | 200 | 0.050 [0.027, 0.090] (10/200) | 0.050 [0.027, 0.090] (10/200) | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.204 | 0.205 | 0.029 | 4.2 | 4.71 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.211 | 0.213 | 0.031 | 4.7 | 4.58 | 0/200 | 0 | 200 | 0.915 [0.868, 0.946] (183/200) | 0.915 [0.868, 0.946] (183/200) | 0.000 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.010 | 0.990 | 0.000 | 0.190 / 0.226 | 0.226 | 0.033 | 8.2 | 4.55 | 0/200 | 0 | 200 | 0.410 [0.344, 0.479] (82/200) | 0.410 [0.344, 0.479] (82/200) | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.130 | 0.110 | 0.760 | 0.000 | 0.194 / 0.234 | 0.234 | 0.036 | 17.8 | 4.53 | 0/200 | 0 | 200 | 0.050 [0.027, 0.090] (10/200) | 0.050 [0.027, 0.090] (10/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.101 / 0.113 | 0.113 | 0.032 | 3.9 | 8.01 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.102 / 0.120 | 0.119 | 0.033 | 4.9 | 7.86 | 0/200 | 0 | 200 | 0.900 [0.851, 0.934] (180/200) | 0.900 [0.851, 0.934] (180/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.045 | 0.955 | 0.000 | 0.105 / 0.131 | 0.133 | 0.032 | 7.6 | 7.83 | 0/200 | 0 | 200 | 0.355 [0.292, 0.423] (71/200) | 0.355 [0.292, 0.423] (71/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.125 | 0.170 | 0.705 | 0.000 | 0.103 / 0.138 | 0.140 | 0.033 | 18.1 | 7.80 | 0/200 | 0 | 200 | 0.050 [0.027, 0.090] (10/200) | 0.050 [0.027, 0.090] (10/200) | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.204 | 0.204 | 0.047 | 4.7 | 3.23 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.211 | 0.211 | 0.049 | 5.3 | 3.22 | 0/200 | 0 | 200 | 0.885 [0.833, 0.922] (177/200) | 0.885 [0.833, 0.922] (177/200) | 0.000 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.189 / 0.218 | 0.217 | 0.049 | 6.9 | 3.48 | 0/200 | 0 | 197 | 0.477 [0.409, 0.547] (94/197) | 0.470 [0.402, 0.539] (94/200) | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.045 | 0.410 | 0.545 | 0.190 / 0.224 | 0.224 | 0.033 | 9.7 | 5.40 | 0/200 | 0 | 91 | 0.231 [0.156, 0.327] (21/91) | 0.105 [0.070, 0.155] (21/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.204 | 0.204 | 0.047 | 4.7 | 3.23 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.214 | 0.214 | 0.049 | 5.5 | 3.23 | 0/200 | 0 | 200 | 0.985 [0.957, 0.995] (197/200) | 0.985 [0.957, 0.995] (197/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.995 | 0.005 | 0.197 / 0.224 | 0.224 | 0.043 | 5.7 | 3.62 | 0/200 | 0 | 199 | 0.759 [0.695, 0.813] (151/199) | 0.755 [0.691, 0.809] (151/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.445 | 0.555 | 0.189 / 0.219 | 0.219 | 0.035 | 5.6 | 5.60 | 0/200 | 0 | 89 | 0.652 [0.548, 0.743] (58/89) | 0.290 [0.232, 0.356] (58/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.204 | 0.204 | 0.047 | 4.7 | 3.23 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.213 | 0.213 | 0.049 | 5.3 | 3.24 | 0/200 | 0 | 200 | 0.945 [0.904, 0.969] (189/200) | 0.945 [0.904, 0.969] (189/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.980 | 0.020 | 0.191 / 0.219 | 0.219 | 0.043 | 5.1 | 3.61 | 0/200 | 0 | 196 | 0.750 [0.685, 0.805] (147/196) | 0.735 [0.670, 0.791] (147/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.250 | 0.745 | 0.194 / 0.219 | 0.220 | 0.027 | 7.0 | 6.75 | 0/200 | 0 | 51 | 0.216 [0.125, 0.346] (11/51) | 0.055 [0.031, 0.096] (11/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.204 | 0.204 | 0.047 | 4.7 | 3.23 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.214 | 0.214 | 0.047 | 4.9 | 3.26 | 0/200 | 0 | 200 | 0.925 [0.880, 0.954] (185/200) | 0.925 [0.880, 0.954] (185/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.980 | 0.020 | 0.193 / 0.221 | 0.221 | 0.041 | 5.3 | 3.78 | 0/200 | 0 | 196 | 0.622 [0.553, 0.687] (122/196) | 0.610 [0.541, 0.675] (122/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.285 | 0.715 | 0.190 / 0.216 | 0.216 | 0.032 | 6.2 | 4.80 | 0/200 | 0 | 57 | 0.421 [0.302, 0.550] (24/57) | 0.120 [0.082, 0.172] (24/200) | 0.000 |

## unseen_vessel — pad at CG (control arm for dmf's phase defect; same frozen episodes)

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 95.0 [91.0, 97.3] (190/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 97.5 [94.3, 98.9] (195/200) | 95.0 [91.0, 97.3] (190/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 91.0 [86.2, 94.2] (182/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 88.5 [83.3, 92.2] (177/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | 100.0 [98.1, 100.0] (200/200) | 97.5 [94.3, 98.9] (195/200) | 89.5 [84.5, 93.0] (179/200) | 33.5 [27.3, 40.3] (67/200) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 97.0 [93.6, 98.6] (194/200) | 44.5 [37.8, 51.4] (89/200) |

### unseen_vessel — pad at CG (control arm for dmf's phase defect; same frozen episodes): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.232 / 0.252 | 0.252 | 0.038 | 4.1 | 4.54 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.232 / 0.283 | 0.283 | 0.041 | 4.3 | 4.65 | 0/200 | 0 | 200 | 0.940 [0.898, 0.965] (188/200) | 0.940 [0.898, 0.965] (188/200) | 0.000 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.237 / 0.347 | 0.347 | 0.041 | 5.1 | 4.58 | 0/200 | 0 | 200 | 0.785 [0.723, 0.836] (157/200) | 0.785 [0.723, 0.836] (157/200) | 0.000 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.020 | 0.030 | 0.950 | 0.000 | 0.290 / 0.456 | 0.459 | 0.041 | 8.3 | 4.64 | 0/200 | 0 | 200 | 0.265 [0.209, 0.330] (53/200) | 0.265 [0.209, 0.330] (53/200) | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.192 / 0.205 | 0.205 | 0.033 | 4.6 | 4.54 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.207 | 0.206 | 0.034 | 5.0 | 4.56 | 0/200 | 0 | 200 | 0.945 [0.904, 0.969] (189/200) | 0.945 [0.904, 0.969] (189/200) | 0.000 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.192 / 0.213 | 0.213 | 0.032 | 5.2 | 4.46 | 0/200 | 0 | 200 | 0.785 [0.723, 0.836] (157/200) | 0.785 [0.723, 0.836] (157/200) | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.005 | 0.005 | 0.990 | 0.000 | 0.191 / 0.227 | 0.228 | 0.032 | 8.1 | 4.56 | 0/200 | 0 | 200 | 0.290 [0.232, 0.356] (58/200) | 0.290 [0.232, 0.356] (58/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.102 / 0.110 | 0.110 | 0.033 | 4.0 | 7.82 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.995 | 0.000 | 0.101 / 0.113 | 0.113 | 0.034 | 4.4 | 7.80 | 0/200 | 0 | 200 | 0.965 [0.930, 0.983] (193/200) | 0.965 [0.930, 0.983] (193/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.025 | 0.975 | 0.000 | 0.103 / 0.122 | 0.123 | 0.032 | 5.6 | 7.72 | 0/200 | 0 | 200 | 0.760 [0.696, 0.814] (152/200) | 0.760 [0.696, 0.814] (152/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.050 | 0.950 | 0.000 | 0.104 / 0.141 | 0.142 | 0.033 | 8.9 | 7.81 | 0/200 | 0 | 200 | 0.240 [0.186, 0.304] (48/200) | 0.240 [0.186, 0.304] (48/200) | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.203 | 0.203 | 0.049 | 4.4 | 3.14 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.208 | 0.209 | 0.049 | 4.4 | 3.24 | 0/200 | 0 | 200 | 0.945 [0.904, 0.969] (189/200) | 0.945 [0.904, 0.969] (189/200) | 0.000 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.192 / 0.213 | 0.214 | 0.044 | 5.2 | 3.21 | 0/200 | 0 | 197 | 0.802 [0.741, 0.852] (158/197) | 0.790 [0.728, 0.841] (158/200) | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.910 | 0.090 | 0.193 / 0.223 | 0.223 | 0.039 | 6.0 | 3.91 | 0/200 | 0 | 182 | 0.313 [0.250, 0.384] (57/182) | 0.285 [0.227, 0.351] (57/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.203 | 0.203 | 0.049 | 4.4 | 3.14 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.208 | 0.209 | 0.047 | 4.3 | 3.24 | 0/200 | 0 | 200 | 0.995 [0.972, 0.999] (199/200) | 0.995 [0.972, 0.999] (199/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.191 / 0.216 | 0.215 | 0.044 | 4.7 | 3.26 | 0/200 | 0 | 198 | 0.944 [0.903, 0.969] (187/198) | 0.935 [0.892, 0.962] (187/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.885 | 0.115 | 0.193 / 0.228 | 0.228 | 0.039 | 5.6 | 4.35 | 0/200 | 0 | 177 | 0.757 [0.689, 0.814] (134/177) | 0.670 [0.602, 0.731] (134/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.203 | 0.203 | 0.049 | 4.4 | 3.14 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.975 | 0.025 | 0.190 / 0.208 | 0.208 | 0.046 | 4.3 | 3.26 | 0/200 | 0 | 195 | 1.000 [0.981, 1.000] (195/195) | 0.975 [0.943, 0.989] (195/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.895 | 0.105 | 0.191 / 0.212 | 0.213 | 0.043 | 4.9 | 3.42 | 0/200 | 0 | 179 | 0.961 [0.921, 0.981] (172/179) | 0.860 [0.805, 0.901] (172/200) | 0.000 |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.335 | 0.665 | 0.191 / 0.219 | 0.218 | 0.036 | 4.8 | 5.78 | 0/200 | 0 | 67 | 0.716 [0.599, 0.810] (48/67) | 0.240 [0.186, 0.304] (48/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.203 | 0.203 | 0.049 | 4.4 | 3.14 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.208 | 0.209 | 0.046 | 4.5 | 3.25 | 0/200 | 0 | 200 | 0.965 [0.930, 0.983] (193/200) | 0.965 [0.930, 0.983] (193/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.970 | 0.030 | 0.192 / 0.214 | 0.214 | 0.041 | 5.2 | 3.61 | 0/200 | 0 | 194 | 0.912 [0.864, 0.945] (177/194) | 0.885 [0.833, 0.922] (177/200) | 0.000 |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.445 | 0.555 | 0.195 / 0.213 | 0.213 | 0.031 | 7.6 | 5.96 | 0/200 | 0 | 89 | 0.494 [0.393, 0.596] (44/89) | 0.220 [0.168, 0.282] (44/200) | 0.000 |

## static — pad at CG (control arm for dmf's phase defect; same frozen episodes)

| method | static |
|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) |
| gated | 100.0 [98.1, 100.0] (200/200) |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | 100.0 [98.1, 100.0] (200/200) |
| gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band) | not run (0/200; see above) |
| gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band) | not run (0/200; see above) |

### static — pad at CG (control arm for dmf's phase defect; same frozen episodes): outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | td n | quiet \| td | quiet landings / listed | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.232 / 0.246 | 0.246 | 0.040 | 4.2 | 4.60 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.203 | 0.203 | 0.031 | 4.2 | 4.53 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.101 / 0.109 | 0.109 | 0.031 | 4.1 | 7.77 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| gated | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.045 | 4.5 | 3.14 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
| oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true future deck motion | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.045 | 4.5 | 3.14 | 0/200 | 0 | 200 | 1.000 [0.981, 1.000] (200/200) | 1.000 [0.981, 1.000] (200/200) | 0.000 |
