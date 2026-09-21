---
name: landing-protocol
description: The evaluation protocol for deck landing — success criteria, outcome classes, episode lists, metrics, statistics (Wilson CIs, rliable IQM, paired bootstrap), hypothesis scoring, and reporting rules. Use whenever writing or reviewing evaluation code, controllers, reward, results tables, figures, the README, or any sentence containing a success rate, touchdown velocity, confidence interval, or comparison between methods.
---

# Landing evaluation protocol

The frozen version lives in `docs/protocol.md` P3-D1. If this file and P3-D1 disagree, P3-D1 wins.

## Success (all four, read from `configs/env/success.yaml`)
- relative vertical velocity at first contact ≤ 0.5 m/s (model scale)
- lateral offset from pad centre ≤ 0.10 m (pad radius 0.15 m)
- relative tilt drone vs deck normal ≤ 15°
- stays in contact, on pad, ≥ 0.5 s after touchdown

## Outcome classes (exhaustive, mutually exclusive, first match wins)
`crash` (tilt > 60° or ground/deck-edge impact) → `off_pad` → `hard_landing` (v_z limit exceeded)
→ `bounce` (contact lost within 0.5 s) → `success` → `timeout` (truncation).

## Episodes
- Generated once per (regime, sea state) from a fixed seed, N = 200, committed to
  `results/episodes/*.parquet`, hash recorded in P3-D1.
- Every method and every training seed runs on the identical list → paired comparisons.
- Evaluation realizations are disjoint from all training and tuning realizations.

## Metrics per cell
success rate (Wilson 95 % CI, N), outcome-class fractions, touchdown relative v_z p50/p95,
lateral error p50/p95, relative tilt p95, time to touchdown, control effort, action jerk,
analytic-vs-contact touchdown disagreement count.

## Statistics
- Per cell, per seed: Wilson CI.
- Across ≥ 5 training seeds: rliable IQM + optimality gap, stratified bootstrap (≥ 2 000 reps) 95 % CI.
- Method contrasts (H1–H4): paired bootstrap over per-episode outcomes, seeds pooled by
  resampling seeds then episodes.
- Say "separates" only when the 95 % CI of the difference excludes 0.

## Reporting rules
- Baselines `pid_track_descend`, `pid_feedforward`, `oracle_gated` (marked privileged) in every table.
- Never pool success across sea states in a headline. Never drop a seed, method or cell.
- Hypotheses: supported / not supported / inconclusive + the number. Post-hoc stats labelled post-hoc.
- Every number in README traces to a committed CSV.
- Required caveats: simulation only; Froude-scaled (λ stated); 3-DOF deck; state-based with a noise
  stand-in, not vision; dmf roll/pitch–heave phase defect.
