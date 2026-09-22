---
name: deck-scaling-physics
description: Reference for Froude scaling of ship motion to drone scale, deck-point kinematics from roll/pitch/heave with a pad lever arm, dmf's seed path and units, the known dmf phase defect, and sinusoidal-motion matching. Use whenever touching src/rld/deck/, converting between full-scale and model-scale time or velocity, computing pad position or velocity, or interpreting deck-motion statistics.
---

# Deck scaling and kinematics

## Froude scaling (λ = L_model / L_full, default 1/25)
length ×λ · time ×√λ · velocity ×√λ · acceleration ×1 · angle ×1 · angular rate ×1/√λ · frequency ×1/√λ.
To sample model time `t_m` at the physics rate, evaluate dmf on `t_full = t_m / √λ`.
Invariant to test: Froude number `v / sqrt(g·L)` unchanged.

## dmf facts (read-only dependency)
- Channels: roll, pitch [deg]; heave [m]; roll_rate, pitch_rate [deg/s]; heave_rate [m/s]; heave_acc [m/s²].
- Conventions (dmf `docs/corpus_card.md`): roll positive to starboard, pitch positive bow-up. Verify.
- Seed path: `realization_seed_sequence(spec).spawn(2)`; child 0 → `sample_components(hs, tp, gamma,
  n_components, w_min, w_max, rng, jitter)`; child 1 is the IMU noise stream (unused here).
- `synthesize_motion(components, vessel, heading_deg, speed_m_s, t_s)` accepts any time array and
  differentiates analytically. `simulate_realization` discards `spinup_s` = 120 s: an episode at
  corpus time t corresponds to synthesis time t (the corpus `t` column is absolute).
- Vessels: frigate L = 124 m (trained), S175 (held out, `unseen_vessel`). Headings 180/135/90/45°;
  45° is encounter-non-monotonic — keep dmf's handling.
- Regimes: `id`, `unseen_seastate` (SS6), `unseen_heading` (90° beam), `unseen_vessel` (S175).

## Deck-point kinematics (3-DOF deck, finite angles) — see protocol P1-D2
One world frame: `x` = bow, `y` = **port**, `z` = up, right-handed (PyBullet's). Map dmf's angle
*signs* into it; do not carry a second SNAME frame. `φ_w = +radians(roll)`, `θ_w = −radians(pitch)`,
ψ = 0. Composition is **ZYX**: `R = R_y(θ_w)·R_x(φ_w)` — what `getQuaternionFromEuler` uses.
Default `r_pad = [−0.4·L, 0, 0]` full scale, **signed**, so an aft pad has `x < 0` and a **starboard**
pad has `y < 0`.

    z_pad   = heave + x_pad·sin(pitch)          # PLUS sign; no cos(roll) under ZYX
    ω       = (φ̇·cos θ_w,  θ̇,  −φ̇·sin θ_w)
    v_pad   = [0,0,heave_rate] + ω × p_offset
    n_world = (−sin(pitch)·cos(roll), −sin(roll), cos(pitch)·cos(roll))

The centreline pad is **exactly roll-blind** under ZYX. Write sign conventions into docstrings and
test against hand-computed cases (`tests/test_kinematics.py` Cases A/B/C).

## Known dmf defect
dmf roll/pitch are in phase with heave where strip theory puts them in quadrature (~90° error).
Aft-pad `v_z = heave_rate + x_pad·cos(pitch)·pitch_rate` (**plus**, `x_pad` signed negative aft)
adds components whose relative phase is wrong, so aft-pad amplitude is biased. Measured
`corr(pitch, heave)` = +0.812/+0.865/+0.948 (min/median/max over 96 cells) where strip theory would
give ~0. Because they are nearly in phase, the lever arm **cancels** heave at an aft pad in 30 of 96
cells, and the aft ladder saturates SS5→SS6 while the CG ladder stays monotone (protocol P1-D2).
Always report aft pad and CG pad side by side. Never fix it here.

## Sinusoidal motion (H4 arm)
Per DOF: amplitude = √2 · RMS of the matched JONSWAP realization, period = its peak encounter
period, random phase per DOF from the episode seed. Same `DeckMotionSource` interface.
