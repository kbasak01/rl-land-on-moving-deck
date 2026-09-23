# e01 — classical baselines: success versus sea state (frozen episode lists)

- Simulation only (PyBullet); no real flight and no real deck data.
- Deck motion is dmf's 3-DOF (heave, roll, pitch) JONSWAP response, Froude-scaled to a Crazyflie at lambda = 1/25 (1 s model = 5 s full scale).
- State-based observations; the perception-noise stand-in is disabled (`configs/env/noise.yaml: enabled: false`); not vision.
- dmf's roll/pitch-heave phase defect (~90 deg) is carried, not fixed; this table is the aft pad, which is sensitive to it (P1-D2).
- Success = all four frozen criteria (`configs/env/success.yaml`); rates in %, Wilson 95 % CI in brackets, then k/N. Success is never pooled across sea states.
- `in-dist` is the fraction of the cell's episodes whose grid cell is in the development pool (P3-D2); `id` SS6 and every 90 deg episode are outside it.
- `quiet td` is, of the touched-down episodes, the fraction whose TRUE deck satisfied the permissive quiescence predicate (`rld.control.quiescence.QuiescenceRule`: 12 samples 1/30 s apart, |roll| <= 3.0 deg, |pitch| <= 2.0 deg, |pad v_z| <= 0.16 m/s model) starting at the contact touchdown; evaluation ground truth, never a controller input. `–` when nothing touched down.
- `oracle_gated` is privileged: it reads the true future deck motion. It bounds commit timing under the gated rule, not success, and is never a deployable result.
- Rendered from `summary.csv` by `rld.eval.report`; do not edit by hand.

## id

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 96.0 [92.3, 98.0] (192/200) | 85.0 [79.4, 89.3] (170/200) | 80.0 [73.9, 85.0] (160/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 90.5 [85.6, 93.8] (181/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) | 95.5 [91.7, 97.6] (191/200) | 85.0 [79.4, 89.3] (170/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 89.5 [84.5, 93.0] (179/200) | 63.0 [56.1, 69.4] (126/200) |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | 100.0 [98.1, 100.0] (200/200) | 98.5 [95.7, 99.5] (197/200) | 89.0 [83.9, 92.6] (178/200) | 64.5 [57.7, 70.8] (129/200) |

### id: outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | quiet td | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.238 / 0.308 | 0.309 | 0.040 | 3.9 | 4.62 | 0/200 | 0 | 0.940 | 0.755 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.020 | 0.020 | 0.960 | 0.000 | 0.254 / 0.412 | 0.412 | 0.042 | 4.8 | 4.59 | 0/200 | 0 | 0.725 | 0.750 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.060 | 0.090 | 0.850 | 0.000 | 0.303 / 0.525 | 0.525 | 0.037 | 7.5 | 4.54 | 0/200 | 4 | 0.315 | 0.745 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.085 | 0.115 | 0.800 | 0.000 | 0.287 / 0.502 | 0.502 | 0.040 | 13.9 | 4.65 | 0/200 | 2 | 0.100 | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.219 | 0.219 | 0.032 | 4.4 | 4.50 | 0/200 | 0 | 0.950 | 0.755 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.189 / 0.226 | 0.226 | 0.034 | 5.1 | 4.49 | 0/200 | 0 | 0.720 | 0.750 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.005 | 0.005 | 0.990 | 0.000 | 0.192 / 0.262 | 0.262 | 0.033 | 8.4 | 4.41 | 0/200 | 0 | 0.290 | 0.745 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.040 | 0.055 | 0.905 | 0.000 | 0.193 / 0.264 | 0.264 | 0.031 | 13.4 | 4.55 | 0/200 | 0 | 0.080 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.103 / 0.127 | 0.127 | 0.031 | 4.5 | 7.68 | 0/200 | 0 | 0.940 | 0.755 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.980 | 0.000 | 0.103 / 0.139 | 0.140 | 0.032 | 5.2 | 7.65 | 0/200 | 0 | 0.740 | 0.750 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.045 | 0.955 | 0.000 | 0.110 / 0.188 | 0.185 | 0.032 | 9.2 | 7.61 | 0/200 | 0 | 0.280 | 0.745 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.045 | 0.105 | 0.850 | 0.000 | 0.106 / 0.181 | 0.182 | 0.034 | 13.9 | 7.77 | 0/200 | 0 | 0.115 | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.217 | 0.216 | 0.044 | 4.7 | 3.18 | 0/200 | 0 | 0.950 | 0.755 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.192 / 0.229 | 0.229 | 0.045 | 4.9 | 3.28 | 0/200 | 0 | 0.741 | 0.750 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.895 | 0.105 | 0.193 / 0.250 | 0.249 | 0.039 | 6.3 | 3.74 | 0/200 | 0 | 0.430 | 0.745 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.020 | 0.630 | 0.350 | 0.194 / 0.267 | 0.267 | 0.038 | 7.8 | 4.66 | 0/200 | 0 | 0.169 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.220 | 0.219 | 0.044 | 4.6 | 3.19 | 0/200 | 0 | 0.985 | 0.755 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.015 | 0.194 / 0.236 | 0.237 | 0.044 | 4.4 | 3.29 | 0/200 | 0 | 0.909 | 0.750 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.890 | 0.105 | 0.191 / 0.252 | 0.252 | 0.042 | 5.1 | 3.60 | 0/200 | 0 | 0.771 | 0.745 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.645 | 0.355 | 0.190 / 0.254 | 0.253 | 0.043 | 5.9 | 4.80 | 0/200 | 0 | 0.682 | 0.000 |

## unseen_seastate

| method | SS6 |
|---|---|
| pid_track_descend | 81.0 [75.0, 85.8] (162/200) |
| pid_feedforward | 90.0 [85.1, 93.4] (180/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 86.0 [80.5, 90.1] (172/200) |
| gated | 64.5 [57.7, 70.8] (129/200) |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | 65.0 [58.2, 71.3] (130/200) |

### unseen_seastate: outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | quiet td | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.075 | 0.115 | 0.810 | 0.000 | 0.282 / 0.497 | 0.498 | 0.039 | 14.3 | 4.49 | 1/200 | 1 | 0.120 | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.030 | 0.070 | 0.900 | 0.000 | 0.193 / 0.250 | 0.247 | 0.036 | 13.1 | 4.46 | 1/200 | 0 | 0.105 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.035 | 0.105 | 0.860 | 0.000 | 0.107 / 0.152 | 0.155 | 0.035 | 13.8 | 7.66 | 1/200 | 0 | 0.080 | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.010 | 0.645 | 0.345 | 0.195 / 0.243 | 0.244 | 0.040 | 7.2 | 4.93 | 0/200 | 0 | 0.214 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.650 | 0.350 | 0.195 / 0.240 | 0.241 | 0.035 | 5.5 | 4.37 | 0/200 | 0 | 0.662 | 0.000 |

## unseen_heading

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 97.5 [94.3, 98.9] (195/200) | 72.5 [65.9, 78.2] (145/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 77.0 [70.7, 82.3] (154/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 99.0 [96.4, 99.7] (198/200) | 96.5 [93.0, 98.3] (193/200) | 70.0 [63.3, 75.9] (140/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 98.0 [95.0, 99.2] (196/200) | 42.0 [35.4, 48.9] (84/200) |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 46.5 [39.7, 53.4] (93/200) |

### unseen_heading: outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | quiet td | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.232 / 0.258 | 0.258 | 0.040 | 4.0 | 4.71 | 0/200 | 0 | 1.000 | 0.000 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.240 / 0.313 | 0.313 | 0.040 | 4.5 | 4.64 | 0/200 | 0 | 0.920 | 0.000 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.025 | 0.975 | 0.000 | 0.264 / 0.370 | 0.368 | 0.041 | 7.0 | 4.64 | 0/200 | 0 | 0.400 | 0.000 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.110 | 0.165 | 0.725 | 0.000 | 0.285 / 0.454 | 0.463 | 0.044 | 18.1 | 4.54 | 0/200 | 5 | 0.060 | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.192 / 0.205 | 0.205 | 0.031 | 4.7 | 4.70 | 0/200 | 0 | 1.000 | 0.000 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.212 | 0.212 | 0.033 | 5.2 | 4.54 | 0/200 | 0 | 0.915 | 0.000 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.995 | 0.000 | 0.191 / 0.222 | 0.221 | 0.031 | 7.9 | 4.62 | 0/200 | 0 | 0.405 | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.120 | 0.110 | 0.770 | 0.000 | 0.188 / 0.230 | 0.231 | 0.033 | 17.7 | 4.51 | 0/200 | 0 | 0.055 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.103 / 0.111 | 0.111 | 0.035 | 4.6 | 8.00 | 0/200 | 0 | 1.000 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.010 | 0.990 | 0.000 | 0.102 / 0.117 | 0.118 | 0.033 | 5.4 | 7.85 | 0/200 | 0 | 0.900 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.035 | 0.965 | 0.000 | 0.104 / 0.130 | 0.130 | 0.035 | 7.3 | 7.82 | 0/200 | 0 | 0.355 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.120 | 0.180 | 0.700 | 0.000 | 0.102 / 0.139 | 0.139 | 0.034 | 18.7 | 7.78 | 0/200 | 0 | 0.050 | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.047 | 4.5 | 3.25 | 0/200 | 0 | 1.000 | 0.000 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.210 | 0.209 | 0.047 | 5.3 | 3.23 | 0/200 | 0 | 0.885 | 0.000 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.980 | 0.015 | 0.192 / 0.219 | 0.220 | 0.043 | 6.2 | 3.50 | 0/200 | 0 | 0.508 | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.035 | 0.420 | 0.545 | 0.189 / 0.218 | 0.218 | 0.036 | 7.3 | 5.36 | 0/200 | 0 | 0.253 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.047 | 4.5 | 3.25 | 0/200 | 0 | 1.000 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.210 | 0.210 | 0.049 | 5.3 | 3.25 | 0/200 | 0 | 0.985 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.995 | 0.005 | 0.197 / 0.220 | 0.221 | 0.042 | 5.5 | 3.61 | 0/200 | 0 | 0.789 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.465 | 0.535 | 0.191 / 0.215 | 0.215 | 0.034 | 6.3 | 5.60 | 0/200 | 0 | 0.634 | 0.000 |

## unseen_vessel

| method | SS3 | SS4 | SS5 | SS6 |
|---|---|---|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) | 97.5 [94.3, 98.9] (195/200) | 90.0 [85.1, 93.4] (180/200) | 81.5 [75.5, 86.3] (163/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 98.5 [95.7, 99.5] (197/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 99.5 [97.2, 99.9] (199/200) | 95.0 [91.0, 97.3] (190/200) |
| gated | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 93.5 [89.2, 96.2] (187/200) | 90.0 [85.1, 93.4] (180/200) |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | 100.0 [98.1, 100.0] (200/200) | 100.0 [98.1, 100.0] (200/200) | 95.0 [91.0, 97.3] (190/200) | 89.0 [83.9, 92.6] (178/200) |

### unseen_vessel: outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | quiet td | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.235 / 0.300 | 0.300 | 0.040 | 4.0 | 4.56 | 0/200 | 0 | 1.000 | 0.000 |
| pid_track_descend | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.025 | 0.975 | 0.000 | 0.253 / 0.388 | 0.388 | 0.040 | 4.5 | 4.69 | 0/200 | 0 | 0.815 | 0.000 |
| pid_track_descend | SS5 | 200 | 0.000 | 0.000 | 0.070 | 0.030 | 0.900 | 0.000 | 0.292 / 0.520 | 0.521 | 0.038 | 4.5 | 4.56 | 0/200 | 2 | 0.540 | 0.000 |
| pid_track_descend | SS6 | 200 | 0.000 | 0.000 | 0.100 | 0.085 | 0.815 | 0.000 | 0.312 / 0.546 | 0.546 | 0.039 | 8.0 | 4.67 | 0/200 | 0 | 0.200 | 0.000 |
| pid_feedforward | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.211 | 0.210 | 0.032 | 4.4 | 4.53 | 0/200 | 0 | 1.000 | 0.000 |
| pid_feedforward | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.228 | 0.228 | 0.034 | 4.4 | 4.55 | 0/200 | 0 | 0.815 | 0.000 |
| pid_feedforward | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.995 | 0.000 | 0.195 / 0.255 | 0.255 | 0.033 | 5.8 | 4.49 | 0/200 | 0 | 0.525 | 0.000 |
| pid_feedforward | SS6 | 200 | 0.000 | 0.000 | 0.005 | 0.010 | 0.985 | 0.000 | 0.193 / 0.260 | 0.260 | 0.033 | 8.8 | 4.57 | 1/200 | 0 | 0.205 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.102 / 0.120 | 0.120 | 0.033 | 4.5 | 7.78 | 0/200 | 0 | 0.985 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.104 / 0.138 | 0.138 | 0.035 | 4.4 | 7.76 | 0/200 | 0 | 0.830 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.005 | 0.995 | 0.000 | 0.108 / 0.155 | 0.155 | 0.034 | 4.8 | 7.75 | 0/200 | 0 | 0.500 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.050 | 0.950 | 0.000 | 0.110 / 0.159 | 0.158 | 0.033 | 7.9 | 7.90 | 0/200 | 0 | 0.190 | 0.000 |
| gated | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.216 | 0.216 | 0.048 | 4.5 | 3.17 | 0/200 | 0 | 0.995 | 0.000 |
| gated | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.192 / 0.227 | 0.228 | 0.046 | 4.9 | 3.25 | 0/200 | 0 | 0.815 | 0.000 |
| gated | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.935 | 0.065 | 0.194 / 0.248 | 0.247 | 0.043 | 5.0 | 3.40 | 0/200 | 0 | 0.604 | 0.000 |
| gated | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.900 | 0.100 | 0.189 / 0.249 | 0.248 | 0.038 | 6.5 | 4.23 | 1/200 | 0 | 0.256 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS3 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.218 | 0.218 | 0.048 | 4.5 | 3.17 | 0/200 | 0 | 1.000 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS4 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.191 / 0.224 | 0.223 | 0.044 | 4.5 | 3.24 | 0/200 | 0 | 0.910 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS5 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.950 | 0.050 | 0.194 / 0.238 | 0.238 | 0.043 | 5.0 | 3.51 | 0/200 | 0 | 0.847 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | SS6 | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.890 | 0.110 | 0.193 / 0.235 | 0.235 | 0.035 | 5.4 | 4.81 | 0/200 | 0 | 0.781 | 0.000 |

## static

| method | static |
|---|---|
| pid_track_descend | 100.0 [98.1, 100.0] (200/200) |
| pid_feedforward | 100.0 [98.1, 100.0] (200/200) |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | 100.0 [98.1, 100.0] (200/200) |
| gated | 100.0 [98.1, 100.0] (200/200) |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | 100.0 [98.1, 100.0] (200/200) |

### static: outcome breakdown and touchdown audit

| method | SS | N | crash | off_pad | hard_landing | bounce | success | timeout | v_n p50 / p95 (m/s) | v_z p95 (m/s) | lat p95 (m) | tilt p95 (deg) | t_td p50 (s) | disagree | tunnel | quiet td | in-dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pid_track_descend | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.232 / 0.246 | 0.246 | 0.040 | 4.2 | 4.60 | 0/200 | 0 | 1.000 | 0.000 |
| pid_feedforward | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.190 / 0.203 | 0.203 | 0.031 | 4.2 | 4.53 | 0/200 | 0 | 1.000 | 0.000 |
| pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8)) | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.101 / 0.109 | 0.109 | 0.031 | 4.1 | 7.77 | 0/200 | 0 | 1.000 | 0.000 |
| gated | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.045 | 4.5 | 3.14 | 0/200 | 0 | 1.000 | 0.000 |
| oracle_gated (privileged — the gated rule applied to the true future deck motion (commit-timing oracle)) | static | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.193 / 0.204 | 0.204 | 0.045 | 4.5 | 3.14 | 0/200 | 0 | 1.000 | 0.000 |
