# KICKOFF_PROMPTS.md — prompts to drive Claude Code through the build

Use one Claude Code session per phase. Start each in **plan mode** (Shift+Tab until "plan mode"),
approve the plan, then let it execute. Run `/clear` after each gate passes so the next phase starts
with a clean context and re-reads CLAUDE.md, the plan and the protocol.

Before the first prompt, run `/agents` and confirm all seven agents are listed, and type `/` to
confirm `/phase-gate`, `/new-controller`, `/full-audit` and `/sweep-status` appear.

---

## Prompt 0 — Bootstrap (Phase 0)

```
Read CLAUDE.md, docs/IMPLEMENTATION_PLAN.md (all of §0–§5 Phase 0) and docs/protocol.md.

Execute Phase 0 exactly:
1. Write pyproject.toml reusing deck-motion-forecast's pins where they overlap (read
   third_party/deck-motion-forecast/pyproject.toml), plus gymnasium, stable-baselines3, rliable,
   tensorboard, imageio[ffmpeg]. Package name `rld`, src layout, ruff/mypy/pytest config copied
   from dmf.
2. Write the Makefile with dmf's PY/RUFF/MYPY/PYTEST auto-detection and stub targets for every
   command listed in CLAUDE.md (stubs print "not implemented: phase N").
3. Give me the exact shell commands to create the venv and install
   -e third_party/deck-motion-forecast -e third_party/gym-pybullet-drones -e ".[dev]".
   I will run pip myself. Flag that pybullet builds from sdist on 3.12.
4. After I confirm install, write tests/test_smoke.py covering the Phase 0 smoke tests and
   scripts/env_throughput.py, run them, and write results/env_throughput.csv.
5. Fill in P0-D1 in docs/protocol.md with the submodule SHAs (git submodule status) and resolved pins.
Then run /phase-gate 0.
```

## Prompt 1 — Deck-motion bridge (Phase 1)

```
Read CLAUDE.md, Phase 1 of the plan, and docs/protocol.md. Delegate all of Phase 1 to the
deck-bridge-engineer agent.

Order of work: (1) the parity test against dmf.sim.generate.simulate_realization at 10 Hz — it must
fail first, then pass; (2) scaling.py and its invariant tests; (3) kinematics.py with the sign
convention verified from dmf's docs/corpus_card.md and a hand-computed test; (4) splits.py from
realization_grid metadata via dmf.data.splits.build_split; (5) sinusoid.py; (6) scripts/deck_stats.py
→ results/deck_stats.csv over the full grid, aft pad and CG pad side by side.

Then evaluate the pre-registered λ feasibility rule and record P1-D1. Do not change λ or r_pad
without writing the reason into docs/protocol.md. Finish with /phase-gate 1.
```

## Prompt 2 — Landing environment (Phase 2)

```
Read CLAUDE.md, Phase 2 of the plan, docs/protocol.md (especially P1-D1 for λ and the bridge
interface). Delegate Phase 2 to the sim-env-engineer agent.

Build and test in this order: platform body with the tracking and velocity-match tests (record the
chosen driving method as P2-D1); touchdown detection with both analytic and contact paths; the
landing env with the shared velocity-setpoint action space; observation and noise modules; reward
with timeout as truncation; configs/env/success.yaml with the plan's criteria. Run check_env, the
determinism test and the scripted static-pad test. Write results/e00_env_sanity.csv (random and hover
policies, 200 episodes each at id SS3 and SS5). Finish with /phase-gate 2.
```

## Prompt 3 — Baselines and protocol freeze (Phase 3)

```
Read CLAUDE.md, Phase 3 of the plan and docs/protocol.md.

Step A — delegate to controls-engineer: implement pid_track_descend, pid_feedforward, gated,
oracle_gated using /new-controller for each. Tune pid_feedforward on id validation seeds only, with a
stated trial budget; record gains and procedure in docs/protocol.md.

Step B — delegate to eval-auditor: generate and commit the frozen episode lists
(results/episodes/*.parquet, N=200 per regime × sea state), write the evaluation runner and
statistics module, evaluate all four controllers → results/e01/.

Step C — draft P3-D1 in docs/protocol.md: success criteria + SHA-256, episode-list hashes, metrics,
statistics, training budgets sized from results/env_throughput.csv, seeds, tuning budget, residual α,
curriculum threshold, and H1–H5 with predicted direction and magnitude. Show me the draft and WAIT
for my approval before marking it frozen.

Then /phase-gate 3.
```

## Prompt 4 — Forecaster integration (Phase 4)

```
Read CLAUDE.md, Phase 4 of the plan and docs/protocol.md. Delegate to deck-bridge-engineer.

1. Give me the commands to regenerate the dmf corpus and fit dlinear_ols and tcn on the id regime
   inside third_party/deck-motion-forecast, writing checkpoints to artifacts/dmf/. I will run them.
2. Implement src/rld/deck/forecast.py: ring buffer at 10 Hz full-scale, 200-sample lookback, history
   from before episode start, outputs deck-point z and v_z at 1/2/3 s full-scale leads in model units.
3. Tests: parity with dmf's offline predictions on identical windows; causality (perturb the future,
   assert identical output); per-step CPU cost measured.
4. Ask controls-engineer to add gated_forecast (interval rule, Froude-scaled PERMISSIVE and STRICT
   limits) via /new-controller; ask eval-auditor to evaluate it on the frozen episodes next to gated
   and oracle_gated.
Finish with /phase-gate 4.
```

## Prompt 5 — Pure RL (Phase 5)

```
Read CLAUDE.md, Phase 5 of the plan and P3-D1 in docs/protocol.md. Delegate to rl-trainer.

1. Implement src/rld/rl/ (train.py, wrappers.py, callbacks.py, curriculum.py) and the make train-bg
   and make sweep targets that write status.json. Test VecNormalize freeze-at-eval.
2. Run a 200k-step PPO smoke run in the background and show me its status and learning curve.
3. If the smoke run is healthy, give me the exact sweep commands for PPO and SAC, 5 seeds each, at
   the P3-D1 budgets. I will launch them. Do not tune on evaluation episodes.
While runs train, stop and wait. I will return with /sweep-status.
```

**Resume prompt after training finishes:**

```
All Phase 5 runs are done (/sweep-status shows it). Have rl-trainer run the reward-hacking audit on
every run and write results/audit/. Have eval-auditor evaluate the final checkpoints of every seed
on the frozen id episode lists and write results/e05/. Commit learning curves (mean ± std over 5
seeds). Then /phase-gate 5.
```

## Prompt 6 — Residual and forecast-conditioned RL (Phase 6)

```
Read CLAUDE.md, Phase 6 of the plan and docs/protocol.md. Delegate to rl-trainer.
Implement residual_ppo (zero-initialised last layer; test that a zeroed residual reproduces
pid_feedforward exactly), ppo_forecast, residual_ppo_forecast and ppo_sinusoid. 100k-step smoke run
of each in the background, then give me the sweep commands at the P3-D1 budget, 5 seeds each.
Wait for me after launching.
```

Resume: same as Phase 5 resume, writing `results/e06/`, then `/phase-gate 6`.

## Prompt 7 — Evaluation under shift (Phase 7)

```
Read CLAUDE.md, Phase 7 of the plan and P3-D1. Delegate to eval-auditor.
Run the full matrix (4 regimes × 4 sea states × every method × every seed) on the frozen episodes,
plus the H4 realism cross, perception noise/latency arms, pad-at-CG control, and the λ sensitivity
arm on the two best methods. Compute rliable IQM + CIs and paired contrasts. Score H1–H5 in
docs/findings.md exactly as pre-registered. Render results/results.md from CSVs and verify it
re-renders byte-identically. Then ask results-skeptic for a full review and show me its findings
before running /phase-gate 7.
```

Optional MSS arm: *"Give me the commands to export Project 4's MSS records (Octave required) and add
an `mss_transfer` evaluation arm on them."*

## Prompt 8 — ONNX and latency (Phase 8)

```
Read CLAUDE.md, Phase 8 of the plan. Delegate to deploy-benchmarker.
Export the best residual and best pure-RL policies (seed 0, plus the seed with median IQM) to ONNX
with VecNormalize folded in, reusing dmf.deploy's harness and GPU-library preloading. Parity on every
provider before timing; closed-loop parity on 50 episodes; latency at batch 1 and 32, 1 thread plus a
thread sweep; end-to-end control-step cost against 33 ms. Score H5. Then /phase-gate 8.
```

## Prompt 9 — Release (Phase 9)

```
Read CLAUDE.md, Phase 9, docs/findings.md and docs/protocol.md.
Write README.md in deck-motion-forecast's register: headline landing GIF + success-vs-sea-state
figure, results tables with all baselines, hypotheses with verdicts, limitations (simulation only,
Froude λ, 3-DOF deck, state-based, dmf phase defect, relation to Angelis et al.), reproduce section
with measured wall-clock per stage. Generate GIFs (PID vs PPO vs residual PPO at id SS5, same
episode). Add THIRD_PARTY_NOTICES.md. Then run /full-audit and show me the BLOCKING count.
```

---

## Recovery prompts

**Gate failed:**
```
Gate N failed on <criteria>. Do not start phase N+1. Propose the smallest fix for each failing
criterion, implement it after I approve, re-run /phase-gate N. If a threshold must change, draft a
dated deviation entry for docs/protocol.md and wait for my approval.
```

**Surprising result:**
```
<method> shows <number>, which is better than I expected. Before anything is written down, ask
results-skeptic to try to break it: leakage, reward hacking, touchdown-detector disagreement, and
whether pid_feedforward was tuned with a comparable budget.
```

**Context getting long mid-phase:** run `/compact focus on phase N state, open tasks, and decisions
recorded in protocol.md`, or finish the current task, update `docs/protocol.md`, and `/clear`.
