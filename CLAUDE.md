# CLAUDE.md — rl-land-on-moving-deck

## What this project is

RL quadrotor landing on a heaving ship deck in gym-pybullet-drones. Deck motion comes from
`deck-motion-forecast` (Project 4, package `dmf`, submodule in `third_party/`), Froude-scaled to a
Crazyflie. Methods: PID baselines, pure RL (PPO/SAC), residual RL on the PID+feedforward baseline,
forecast-conditioned RL, a sinusoid-trained policy for the motion-realism test. ONNX policy
latency study at the end.

Full spec: `docs/IMPLEMENTATION_PLAN.md`. **Read the phase you are in before writing code, and read
`docs/protocol.md` for everything already decided.** Where they disagree, the protocol wins.

## Non-negotiables

1. **Simulation only.** No sentence anywhere may imply real-world flight or real deck data.
2. **Realization-level splits, inherited from dmf.** Train and eval never share a realization key
   (vessel, ss, heading, speed, seed). Episodes are start offsets *inside* a realization.
3. **Frozen evaluation.** Every method is evaluated on the identical committed episode lists in
   `results/episodes/`. The success criteria and P3-D1 protocol do not change after RL training
   starts without a dated deviation in `docs/protocol.md`.
4. **Baselines always printed.** `pid_track_descend`, `pid_feedforward` and `oracle_gated` sit beside
   every learned method in every table.
5. **Five seeds minimum for every learned method.** Report IQM with 95 % stratified-bootstrap CIs
   (rliable). A difference inside overlapping CIs is not a result.
6. **Report what you find.** If PID beats PPO, if H4 is null, if the CPU wins — it goes in the
   README body. Never drop a seed, a method, or a sea state from a table.
7. **dmf is read-only.** Never edit `third_party/`. Its known roll/pitch phase defect is carried and
   controlled for (pad-at-CG arm), not fixed.

## Architecture rules

- Logic in `src/rld/`; `scripts/` are argparse wrappers; YAML in `configs/` → dataclasses.
- Deck motion is evaluated **analytically** through `dmf.sim.response.synthesize_motion` on the
  physics time grid. No interpolation of the 10 Hz corpus, no finite-difference rates.
- Every controller implements `Controller.act(obs) -> setpoint[3]` (`src/rld/control/base.py`).
  New controller = file + config + registry entry + test. Use `/new-controller`.
- All methods share one action space: world-frame velocity setpoint tracked by `DSLPIDControl`.
- Units in every docstring. Angles in **degrees** at module boundaries, radians inside kinematics.
  Model-scale vs full-scale is stated for every time and length quantity.

## Commands

```
make test | make lint              # pytest; ruff + mypy
make deck-stats                    # Phase 1 deck-point statistics
make baselines                     # Phase 3 classical controllers on frozen episodes
make dmf-forecasters               # Phase 4: dmf corpus + dlinear_ols + tcn
make train-bg CFG=configs/rl/…     # background training run; status in artifacts/runs/*/status.json
make eval                          # full evaluation matrix → results/
make bench                         # ONNX export, parity, latency
make report                        # re-render results/results.md from CSVs
make all
```

**Never run training in the foreground.** Launch with `make train-bg`, then check `/sweep-status`.

## Gates

Each phase ends with a gate in the plan. Run `/phase-gate N`. Do not start phase N+1 until gate N
passes. Never relax a threshold silently; record any change in `docs/protocol.md` with its reason.

## Specialists — delegate rather than doing their work in the main thread

| Agent | Owns |
|---|---|
| `deck-bridge-engineer` | `src/rld/deck/`: dmf bridge, Froude scaling, kinematics, splits, sinusoid, forecaster adapter — Gates 1, 4 |
| `sim-env-engineer` | `src/rld/envs/`: PyBullet platform, landing env, touchdown, noise — Gate 2 |
| `controls-engineer` | `src/rld/control/`: PID, feedforward, gated, oracle — Gate 3 (controllers) |
| `rl-trainer` | `src/rld/rl/`: PPO, SAC, residual, curriculum, sweeps — Gates 5, 6 |
| `eval-auditor` | `src/rld/eval/`, episode lists, statistics — Gates 3 (protocol), 7. **Read-only over `rl/` and `control/`** |
| `deploy-benchmarker` | `src/rld/deploy/`: ONNX export, parity, latency — Gate 8 |
| `results-skeptic` | Adversarial read-only review before every gate from Phase 3 on |

## Known traps in this project

- **Teleported platforms.** `resetBasePositionAndOrientation` each step gives the contact solver a
  zero-velocity deck. Drive the platform with a constraint (or set velocity explicitly) and test it.
- **Timeouts are truncations.** `truncated=True`, not `terminated=True`, or PPO learns to hover.
- **VecNormalize at eval.** Load saved stats, set `training=False, norm_reward=False`. Export the
  normalisation *into* the ONNX graph.
- **Froude time.** 1 s full-scale = √λ s model-scale (0.2 s at λ = 1/25). The dmf forecaster's
  10 Hz / 200-sample lookback is **full-scale** time.
- **Aft pad and the dmf phase defect.** Deck-point v_z at an aft pad depends on the roll/pitch-heave
  phase that dmf gets wrong by ~90°. Always carry the pad-at-CG control.
- **pybullet on 3.12** builds from sdist; `dmf` must be an **editable** install (it resolves
  vessel configs relative to its source tree).
- **SB3 MLP on GPU is slower than CPU.** Train PPO with `device="cpu"`.
- **Success without outcome classes** hides hovering and crashes. Always print the breakdown.

## Style

Type hints, mypy strict, ruff (Project 4's config). Google docstrings with units. No notebooks
defining logic.
