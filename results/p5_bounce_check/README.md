# P5-D1: `pid_feedforward_lowvz` bounces, contact-solver check

This is a diagnosis only (`docs/protocol.md` P5-D1). Everything was flown on the **P3-D2 tune pool only** (`dev_pool(cfg)[1]`: frigate, aft pad, seed ordinals 27–31) and nothing came from `results/episodes/`. No environment code, config or success criterion was changed. Units are model scale (λ = 1/25).

## Episodes
Each arm flew 1 580 episodes:
- the committed P3-D3 tune draw: `tuning_seed` 20260923, 60 episodes each at SS3, SS4 and SS5;
- an extra diagnostic draw from the same pool (seed 20260925): 1 000 at SS5 and 400 at SS4.

The committed-physics arm reproduces `results/e01/tune_pool_final.csv` exactly on the 180-episode tune draw: 59/60, 60/60 and 59/60 successes, and both losses are `bounce`.

## Files
| file | content |
|---|---|
| `arms_summary.csv` | Outcomes, contact-gap counts in the 0.5 s dwell window, solver push-off and penetration, per (controller, physics arm). |
| `paired_vs_base.csv` | Same-episode outcome transitions: base → 480 Hz and base → `contactERP` = 0. |
| `bounce_triggers.csv` | One row per bounce (242 rows): what the drone, deck and contact did across the gap that triggered `release`, plus a mechanism label. |
| `tau_bins.csv` | Bounce rate against τ = 0.06 m · sin(rel. tilt) / closing speed, the rim-rocking time. |
| `contact_probes.csv` | Controller-free PyBullet probes: (1) Baumgarte push-off against overlap, rate and `contactERP`; (2) rim-first drops against rate, tilt, speed and thrust. |

The scripts are `scripts/p5_bounce_check.py` (runs), `scripts/p5_bounce_check_report.py` (these CSVs) and `scripts/p5_contact_probes.py`. The per-substep logs (~1 GB) are not committed.

## Physics arms (all diagnostic, none committed)
- **`base`:** the committed physics, 240 Hz.
- **`hz480`:** 480 Hz physics.
- **`erp0`:** 240 Hz with `contactERP` = 0, so there is no penetration-recovery push-off.
- **`grace_probe*`:** the same physics at 240, 480, 960 and 1920 Hz, and at 240 and 960 Hz with `contactERP` = 0. The contact-loss grace is set to 0.5 s, so no episode ends on a release and the longest contact gap can be measured.

## Result, `pid_feedforward_lowvz`
| arm | bounces / 1 580 | dwell gaps > 50 ms | median of the per-episode max contact-point push-off (mm/s) |
|---|---|---|---|
| 240 Hz (committed) | 64 | 58 | 18.6 |
| 240 Hz, `contactERP` = 0 | 66 | 59 | 0.4 |
| 480 Hz | 86 | 78 | 19.5 |
| 960 Hz | – | 112 | 19.6 |
| 960 Hz, `contactERP` = 0 | – | 103 | 0.2 |
| 1920 Hz | – | 116 | 19.1 |

Mechanism of the 64 committed-physics bounces, taken from the gap that triggered `release`:

- **Rim rocking (51).** 48 happen at impact and 3 later in the dwell.
  - The drone touches down on its rim at a median relative tilt of 6.9° (IQR 5.6–7.8°), with 1 contact point.
  - The rim impulse and the attitude loop rotate the drone flat at about 1.6 rad/s (median). That is faster than the CoM is falling, so the rim lifts by at most ~2 mm (median 2.0 mm, max 3.9 mm).
  - The drone's **CoM keeps closing on the deck through the whole gap**: the mean CoM relative normal velocity is < 0 in all 51.
  - Median values: gap starts 8 ms after touchdown, closing speed 0.096 m/s, thrust 0.254 N (96 % of weight).
- **Unloaded rim lift-off (13).** These happen 4–500 ms after touchdown.
  - The deck accelerates down its normal at a median −2.1 m/s² (range −1.7 to −3.7).
  - The commanded thrust is 0.20 N (76 % of weight). The contact carries only a median 8.7 mN (1–22 mN) in the 20 ms before the gap.
  - A small rotation then lifts the rim. The CoM recedes at up to 21 mm/s (median of per-bounce maxima).

The solver's own contribution is the contact-point relative normal velocity left at the end of the last contact substep. Across the 64 bounces it is −0.3 mm/s (median), 4.2 mm/s (p95) and 10.1 mm/s (max).

## Reading
1. **Not a solver artifact.** The bounces do not come from restitution, ERP push-off, penetration recovery, friction or the force filter:
   - Turning push-off off (`contactERP` = 0) leaves 60 of 64 bounces unchanged, pairwise.
   - The penetration seen is ≤ 1.3 mm.
   - At every release the analytic clearance also showed the lowest point more than the 1 mm margin clear of the plate (64/64).
2. **The number of gaps does depend on the substep.** Gaps over 50 ms go 58 → 78 → 112 → 116 at 240 / 480 / 960 / 1920 Hz, converging by about 960 Hz. So 240 Hz *under*-counts them by about 2×; it does not create them. Individual episodes are not stable to the substep: of the 64 bounces at 240 Hz, 20 are still bounces at 480 Hz, and 66 successes at 240 Hz become bounces.
3. **About 80 % of these "bounces" are scored by the 50 ms `contact_loss_grace_s`, not caused by a lift-off.**
   - The bounce rate climbs with τ: 9/1 299 for τ ≤ 50 ms, 7/136 for 50–70 ms, 25/91 for 70–100 ms and 23/54 for > 100 ms.
   - The frozen-list e01 rows, read and not flown, show the same pattern: ≤ 1 % for τ ≤ 50 ms, then 14/242, 43/138 and 42/152.
4. **Down-force decides it.** In the controller-free probes, a tilted rim-first touchdown at thrust = weight leaves a contact gap of 229–325 ms at every rate from 240 to 1920 Hz, and also with `contactERP` = 0. At 96 % of weight the gaps disappear; at 98 % one case leaves a single 4 ms gap. `lowvz` keeps near-hover thrust after contact by design (the setpoint is the feedforward).
