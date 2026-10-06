# Pre-release audit (`/full-audit`, Gate 9 input)

**Audited:** `phase-9-release` at `bbdabc6`, clean tree, on 2026-10-05.

**Method:**
1. `make test lint`.
2. Every §7 checkbox of `docs/IMPLEMENTATION_PLAN.md`, checked against committed artifacts.
3. Every number in `README.md` traced to a committed file.
4. `report.py --check`.
5. The deviation and hypothesis record in `docs/protocol.md`.
6. A full adversarial review by `results-skeptic` (read-only, whole project).

Nothing was fixed during the audit.

**Classification used.**
- **BLOCKING:** a released sentence is false, or a CLAUDE.md non-negotiable is breached.
- **SHOULD FIX:** an overclaim, a gap in a checklist item, or a reproducibility gap that does not
  make a released claim false.
- **NOTE:** disclosed, harmless, or for the record.

The `results-skeptic` review classed every finding as SHOULD FIX or NOTE. This audit re-classes
three of them (B1–B3) as BLOCKING under the definition above, and says why in each case.

## Result

**BLOCKING: 3. SHOULD FIX: 8. NOTE: 11.**

No leakage, dropped seed, reworded hypothesis or verdict-changing simulator exploit was found. No
released sentence implies real flight, real deck data or an embedded latency target. Every number
traces to a committed file. The three blocking items are wording or omission errors in
`README.md` and `results/results.md`. Each needs a sentence or a table row, not a re-flight.

## BLOCKING

**B1. Two flown methods are missing from the release documents** (CLAUDE.md non-negotiable 6:
"Never drop a seed, a method, or a sea state from a table").
- *What is missing.* `gated_forecast` and `gated_forecast_tcn` were flown on the frozen lists at
  Gate 4. They have 26 cells × 200 each, aft and CG, in `results/e02/` (P4-D4). They appear in no
  `README.md` table or sentence, and nowhere in `results/results.md`.
- *A count is wrong as a result.* README line 7 says "six classical controllers".
- *The omission does not flatter the learned methods.* Forecast gating lowered success against
  `gated` in 19 (DLinear) and 9 (TCN) of 26 cells, and raised it in none (P4-D4, Gate 4 row).
- *Smallest fix.* Add the two methods to the README methods list with that result and a pointer
  to `results/e02/success_vs_seastate.md`, and correct the count. Optionally carry them into
  `results.md`.

**B2. The README's S175 statements are false as worded** (README lines 289–291).
- "Its Wilson interval contains every PPO-family point estimate" fails at `unseen_vessel` SS5.
  Every PPO-family method scores 100.0 there, above `pid_feedforward`'s 199/200 Wilson upper
  bound of 99.9. The source, `docs/findings.md` §2, scopes the sentence to SS6.
- "It scores 99.0–100 % at SS3–SS5" sits in the S175 bullet, but the S175 values are 100 / 100 /
  99.5. The 99.0 belongs to findings' "in every regime".
- *Smallest fix.* Scope the Wilson sentence to SS6 and write "99.5–100 %". The headline "not
  beaten on S175" stays true: no contrast separates, and it is unpaired.

**B3. A gallery caption contradicts the committed data** (README lines 116–117).
- The caption says "on #168 above, the timing is the other way round".
- On #168 the RL policies also touch down first: `ppo` at 1.60 s and `residual_ppo` at 2.25 s,
  against the PID's 5.25 s (`results/figures/gifs/manifest.csv`). What reverses is the *outcome*,
  not the timing.
- *Smallest fix.* "On #168 the same early RL touchdown goes with RL landing and the PID failing."

## SHOULD FIX

1. **§7 "closed-loop parity 50/50" is not met as written.** 3 of 4 exported policies matched 49/50
   (`results/latency/closed_loop_parity.csv`). The item passes only under the post-hoc, dated,
   user-approved deviation P8-D5, and only for ORT CPU. Gate 9 asks for "§7 all checked", so this
   item can be recorded only as "met under P8-D5", the way the Gate 8 row is.
2. **§7 "bridge parity test green against the pinned dmf SHA": nothing pins the SHA.**
   - `tests/test_bridge.py` compares the bridge with whatever dmf is installed.
   - By hand: the submodule HEAD is `e9fa15cc…`, which equals P0-D1; the submodule tree is clean;
     `dmf.__file__` resolves inside `third_party/`.
   - So the item is true today but not automated. Fix: one test asserting the submodule HEAD and
     the import path.
3. **The noise-arm detector disagreement cites incomplete sources** (README lines 565–567).
   - The quoted 1.21 / 1.28 / 1.30 % is learned plus baselines: 349 / 369 / 375 of 28 800, from
     `summary.csv` + `baselines_summary.csv`. The README cites `summary.csv` only, which gives
     1.31–1.43 % over 24 000.
   - The matrix figure beside it, 17 / 82 000, is learned-only.
   - Fix: cite both files and state the denominators.
4. **`docs/findings.md` is described as "the phase-by-phase record, including every claim this
   project withdrew"** (README lines 23–25 and 666). It covers Phases 7–9 only. Earlier withdrawals
   are only in `docs/protocol.md`, for example P5-D14 M1 and P6-D5 M1. Reword, or add Phase 0–6
   summaries.
5. **"Gates 0–2 and 4–7 passed as written" (README line 14) needs two qualifications.**
   - Gate 7's "H1–H5 scored" was met with H5 deferred by a dated decision (P7-D1 §7).
   - Gate 1's ambiguous denominator was resolved in P1-D1.
6. **Reproducibility overclaims.**
   - README line 658 says "every flown stage has a `--check` mode". `env_sanity.py`,
     `deck_stats.py`, `reward_hacking_audit.py` and `eval_baselines.py` do not have one
     (`eval_baselines.py` has `--reference-dir`).
   - README line 651 lists `make test` among targets with "no simulation", but it runs about
     14 min of simulation.
   - `make all` does not regenerate several committed result sets, and some of their scripts have
     no `make` target, so plan §1 item 7 ("`make all` reproduces everything") is not met as
     written. The sets are:
     - `e01_lowvz_cut`, `e02`, `e05` and `e06`;
     - `results/audit/` and the learning curves;
     - `results/mss/`, `results/forecast/*` and the throughput CSVs.
   - Fix: targets, or a README table naming the script behind each directory.
7. **The Phase 9 README review is not itemised in any committed file.** P9-D1 and findings give
   only counts (3 MAJOR, 19 MINOR), so its fold-in cannot be checked item by item. Itemise it in
   the Gate 9 row.
8. **The P9-D1 "written before the gallery was rendered" claim cannot be shown by git.** The entry
   and the GIFs were committed together (`bbdabc6`, `f95e388`). Reword it as an assertion, or
   accept it as one.

## NOTE

- **§7 touchdown disagreement.**
  - Matrix overall: 17 / 82 000 = 0.021 %, under 1 %. Baselines: 5 / 14 000 and 1 / 2 800.
  - The worst single cell is 3.5 % (`sac` seed 0, `unseen_heading` SS6). The σ_p = 4 cm conditions
    are 1.21–1.30 %.
  - Both are disclosed in the README. The item passes on the overall reading.
- **§7 reward-hacking audit.** `results/audit/` (P5-D14) and `results/audit/e06/` (P6-D5) audited
  the same `final/` checkpoints that every later phase flew, and Phase 7 recorded
  `checkpoints_unchanged: true`.
  - The audits were not re-run for Gate 9. Nothing was retrained.
  - They cover clean `id` only. Tunnelled successes outside it are unaudited, which is disclosed.
- **Deviations from P3-D1.**
  - The P3-D1 block SHA-256 recomputes to `21465588…`, as recorded at Gate 3.
  - `configs/env/success.yaml` hashes to `c7fbdcc4…`, and `MANIFEST.csv` to `e6f30e55…`.
  - The deviations that affect evaluation are dated and justified: P7-D4 (perception stand-in) and
    P8-D5 (post-hoc closed-loop parity rule, affecting P8-D1, not P3-D1). P3-D4 errata predate
    Phase 5.
  - The hypothesis wording in `hypotheses.csv`, `h5.csv`, findings and the README matches P3-D1
    §8. P7-D1a's scoring details were committed before any Phase 7 output was read (P7-D2).
- **README line 673** links this file. It did not exist until this audit wrote it.
- **README line 150** labels the deck-motion table "`id` regime". `results/deck_stats.csv` pools
  all 40 seeds per cell. P7-D6 showed that restricting to the `id` seeds moves no value by more
  than 0.13 mm or 0.0008 m/s.
- **README line 251.** The 14-cell closing-speed claim includes `static`, where only 3 of the 5
  PPO-family methods were flown (the forecast methods cannot run without a ship).
- **H5.** `torch:cuda` passed parity but is "context, not scored", as defined in P8-D1. Its 8.8×
  ratio would not change the verdict.
- **H1b's reference is a PID family that cannot express a two-phase descent.** This is disclosed.
  No two-phase classical baseline was tried.
- **P3-D1 §9's pre-commit smoke check** on 32 static episodes is disclosed and harmless.
- **The headline figure shows success only.** The outcome classes are in the tables directly
  below it.
- **Cosmetic pad-marker defect** (P9-D1 §2). The yellow disc never moved with the plate in any
  committed flight. It has no collision shape and no number is affected. The GIF code draws it on
  the plate.

## §7 validation protocol, checkbox by checkbox

| §7 item | verdict | evidence |
|---|---|---|
| Splits: no realization key in both training and evaluation (test re-run on final configs) | **PASS** | `tests/test_deck_splits.py` and `tests/test_rl_leakage.py` pass in this audit's `make test`. The skeptic independently rebuilt all 2 800 list keys against the dev pool (frigate, SS3–SS5, 45/135/180°, ordinals 0–31): 0 overlap. Training pools come from `dev_pool` (`src/rld/rl/train.py:351-378`). |
| Episode lists: hash in results matches P3-D1 hash | **PASS** | `scripts/make_episodes.py --check`: 5/5 lists, content and file OK. `MANIFEST.csv` SHA `e6f30e55…`. `results/episodes/` is unchanged since `0780aaa`. |
| Every learned method: 5 seeds present for every cell; no silently dropped seed | **PASS** | Seeds 0–4 in every learned cell of all 17 current `results/e07/**/summary.csv`; 0 cells short. `artifacts/runs/` holds exactly seeds 0–4 per method, all `done`. No dedicated test (SHOULD FIX-adjacent; covered by the CSVs). |
| VecNormalize stats loaded frozen at eval (`training=False`, `norm_reward=False`) | **PASS** | `tests/test_rl_env.py::test_vecnormalize_saved_reloaded_and_frozen` and `tests/test_rl_e2e.py::test_load_policy_is_a_controller_with_frozen_stats` pass. |
| Reward-hacking audit re-run on final checkpoints | **PASS** (not re-run for Gate 9; see NOTE) | `results/audit/README.md` (P5-D14) and `results/audit/e06/README.md` (P6-D5) audit the `final/` checkpoints. No checkpoint has changed since. |
| Touchdown analytic-vs-contact disagreement < 1 % on the final matrix | **PASS** (overall; see NOTE) | 17 / 82 000 = 0.021 % (`results/e07/matrix/summary.csv`). Worst cell 3.5 %. |
| Bridge parity test green against the pinned dmf SHA | **PASS, not automated** (SHOULD FIX 2) | `tests/test_bridge.py` passes. Submodule HEAD `e9fa15cc…` = P0-D1. No test pins it. |
| Forecaster causality test green | **PASS** | `tests/test_deck_forecast.py::test_future_perturbation_leaves_windows_and_forecasts_bit_identical` and `tests/test_rl_forecast_obs.py::test_block_is_causal` pass. |
| ONNX parity on every timed provider; closed-loop parity 50/50 | **Numeric PASS; closed loop FAIL as written** (SHOULD FIX 1) | `results/latency/parity.csv` 220/220 `passed`, worst max_abs_err 9.5e-7 against 1e-4. `closed_loop_parity.csv`: 1 of 4 policies at 50/50. Met only under post-hoc P8-D5 (ORT CPU). |
| Every README number greps to a committed CSV; `make report` re-renders `results.md` byte-identically | **PASS** (numbers trace); see B2/B3 for two false *statements* | `scripts/report.py --check`: byte-identical. Traceability table below: 0 untraceable data numbers. |
| Every deviation from P3-D1 dated and justified in `docs/protocol.md` | **PASS** | P3-D1 block SHA `21465588…` unchanged. P7-D4 and P8-D5 are dated and justified (see NOTE). |
| README states: simulation only; 3-DOF deck; Froude-scaled; state-based; dmf phase defect | **PASS** | README Limitations: "Simulation only." (l.508), "Froude scaling at λ = 1/25" (l.511), "A 3-DOF deck." (l.521), "State-based observations." (l.523), "dmf's phase defect is carried, not fixed." (l.528). |

**Test and lint (this audit).** `make test lint` at `bbdabc6`: 775 passed, 1 skipped (by design, `test_platform.py:191`, P2-D1), 842 s; ruff, ruff-format and mypy --strict clean (92 source files).

## README traceability

Every numeric token in `README.md` was extracted, leaving out code fences, URLs, paths and inline
code. Each was searched for, in this order, in `results/results.md` (rendered byte-reproducibly
from the CSVs), the committed CSVs and JSONs under `results/`, and `docs/findings.md` /
`docs/protocol.md` (which name the CSV behind each number). Numbers that did not match literally
were recomputed from their source.

| class | count | resolution |
|---|---|---|
| found literally in `results/results.md` | 317 | traced |
| found literally in a committed CSV/JSON under `results/` | 22 | traced |
| found in `docs/findings.md` / `docs/protocol.md` only | 21 | recomputed from the CSV each cites; all match (rows below) |
| found nowhere literally | 17 | 9 are bibliographic (years, arXiv ids); 8 recomputed, all match (rows below) |
| **untraceable** | **0** | — |

The recomputed numbers:

| README number(s) | source and derivation | matches |
|---|---|---|
| 0.577 m/s, 8.33 m/s, 0.069, 0.38, 1.15× | `results/deck_feasibility.csv` gate row: `vz_p99_model_m_s` 0.57716, `v_max_m_s` 8.3333; ratios to 8.33, 1.5 and 0.5 m/s | yes |
| deck z SD 1.04 / 2.08 / 3.41 / 4.10 cm; v_z SD 0.046 / 0.086 / 0.134 / 0.147 m/s | `results/deck_stats.csv`, frigate aft rows, mean over the 12 cells per SS: 0.0104 / 0.0208 / 0.0341 / 0.0410 m; 0.0464 / 0.0857 / 0.1344 / 0.1468 m/s | yes |
| noise ratios 3.86 / 1.92 / 1.17 / 0.98 and 4.31 / 2.33 / 1.49 / 1.36 | σ_p = 4 cm and σ_v = 0.2 m/s over the deck SDs above (findings §4 table) | yes |
| 0.018, 0.021, 0.082, 0.086, 0.156, 0.196, 0.243 ms | `results/latency/latency.csv` p50/p99 of the `mlp512x2-tanh-in25` rows (rounded) | yes |
| 0.19 / 0.30, 0.85 / 1.13, 0.53 ms; 0.9 %, 3.4 % | `results/latency/e2e_budget.csv` `deployment_sum` and `forecaster` rows | yes |
| 1.21 / 1.28 / 1.30 % | `results/e07/noise/sigma4cm_lat{0,1,2}step/{summary,baselines_summary}.csv`: 349 / 369 / 375 of 28 800 | yes (citation incomplete: SHOULD FIX 3) |
| 1.45 / 1.77 / 3.73 s; 16.3°, 16.5°, 21.9° etc. in captions | `results/figures/gifs/manifest.csv` `td_t_episode_s`, `rel_tilt_deg` (rounded) | yes |
| 12.4 %, 23.4 %, 6.0 %, 5.5 %; percentile ranges 0.21–0.29, 0.14–0.26 m/s | `results/e07/matrix/episodes.csv.gz`, `results/e01/episodes.csv`, `results/e01_lowvz_cut/episodes.csv` via `rld.viz.curves.closing_speed_table` (`tests/test_viz.py` pins 23.4 %) | yes |
| wall clock 3.6 min, 15 s, 8.8 min, 18.5 s, 2.1 min, 4.55 h, 4.88 h, 4.7 min, 11.9 min, 2.3 s; training 15.2 / 60.4 / 16.3 / 27.9 / 26.7 / 11.9 h; tuning 10.4 / 60.8 / 36.2 h; 0.7 h | `results/runtime_stages.csv`, summed per stage (`tests/test_viz.py::test_runtime_table_matches_its_committed_sources` re-collects the committed-source rows) | yes |

Independently, the `results-skeptic` recomputed the README's `id` and shift tables, the p95
values, the contrast counts (65/0, 39/0, 20/0, 52/0, 3, 2), the 14-cell closing-speed claim, the
GIF k-of-5 counts and the closed-loop 49/50 results from the CSVs. All match.

## What the gate needs

Gate 9 asks for §7 all checked, no BLOCKING audit findings, a `make all` dry run listing every
stage, and every README number traced.
- **Met:** the dry run lists deck-stats, env-sanity, baselines, dmf-forecasters, eval, bench,
  bench-investigate, report and figures. All README numbers are traced.
- **Not met:** B1–B3 must be fixed first.
- **Decision for the user:** the §7 closed-loop item is met only under P8-D5. Recording it that
  way, as at Gate 8, is a decision for the user, not for this audit.

## Post-audit resolution (2026-10-05, after this audit)

The audit above is unchanged. It records the state at `bbdabc6`. Fixes were made afterwards and
are itemised in P9-D2:

| finding | resolution |
|---|---|
| B1 | `gated_forecast` and `gated_forecast_tcn` are added to the README methods list, the `id` table and the SS6 shift table, from `results/e02/summary.csv`; the count now reads "six classical and two forecast-gated controllers" (P9-D2 §3). `results/results.md` and findings §2 carry a pointer to them (P9-D2 §7). |
| B2 | The range is corrected to 99.5–100 %. The Wilson sentence went through two wrong corrections (P9-D2 §3, §4). It now states the facts per sea state: at SS5 every PPO estimate is one episode above its 199/200 and just above its Wilson interval [97.2, 99.91]; at SS6 four of five are inside it and `ppo_sinusoid` (0.995 > 0.99489) is just above. |
| B3 | The #168 caption now says the *outcome*, not the timing, reverses. |
| SF1 | Not changed. The §7 closed-loop item is met only under P8-D5. **User decision (2026-10-05): accepted** for Gate 9 as "met under P8-D5, ORT CPU only" (P9-D2). |
| SF2 | `tests/test_release.py` pins both submodule commits and the import paths. |
| SF3 | The noise-arm disagreement cites `summary.csv` + `baselines_summary.csv` with its denominators. |
| SF4 | `docs/findings.md` is described as covering Phases 7–9; the protocol covers every phase. |
| SF5 | The README status states the Gate 1 and Gate 7 qualifications. |
| SF6 | The `--check` and `make test` claims are corrected, and a README table names what regenerates each directory outside `make all`. |
| SF7 | Itemised in P9-D2. |
| SF8 | P9-D1's ordering claim is labelled an assertion. |
| NOTEs | The deck-table label and the static-cell caveat are fixed; the rest need no change. |

**BLOCKING remaining after the fixes: 0.** SHOULD FIX remaining: 0 (SF1 accepted by the user under P8-D5).

**After the first `/phase-gate 9` attempt** (2026-10-06): that attempt failed on B2, which was
only partly fixed. It is now fully fixed, together with the gate review's MAJOR and MINOR items
(P9-D2 §3). BLOCKING remaining: 0.

**After the second `/phase-gate 9` attempt** (2026-10-06): that attempt failed on four false README
sentences, one of them the attempt-1 B2 fix. All four are fixed, with the attempt-2 MINOR items
and one more false sentence found in a full-precision sentence check (P9-D2 §4). BLOCKING
remaining: 0.

**After the third `/phase-gate 9` attempt** (2026-10-06): that attempt failed on one false sentence
in `docs/findings.md` §2 ("ties the learned methods"). It is fixed, together with four more
imprecise §2/README sentences found by an unrounded sentence check and the attempt-3 MINOR items
(P9-D2 §5). BLOCKING remaining: 0.

**After the fourth `/phase-gate 9` attempt** (2026-10-06): that attempt's review swept all of
findings Phases 7–9 at full precision. It found two false Phase 7 sentences, one mis-scoped one and
four minor imprecisions, all in `docs/findings.md`, and all fixed (P9-D2 §6). BLOCKING
remaining: 0.
