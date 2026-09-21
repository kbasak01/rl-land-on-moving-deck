---
name: phase-gate
description: Run the phase gate check for a given build phase before moving on. Verifies the phase's acceptance criteria against committed artifacts and reports pass/fail per criterion.
argument-hint: <phase-number 0-9>
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
---

Run the gate check for phase $ARGUMENTS of `docs/IMPLEMENTATION_PLAN.md`.

1. Read the phase section and its **Gate** criteria. Read `docs/protocol.md` for any recorded
   change to those criteria; the protocol wins.
2. Run `make test` and `make lint`. Report failures by test name only.
3. Check each criterion against **committed artifacts** (files in `results/`, test outcomes, recorded
   SHAs, hashes) — not against what the code claims. Unverified until an artifact shows it.
4. For phases 3 and above, delegate a review to the `results-skeptic` agent and fold in its findings.
5. Table: criterion | measured value | PASS / FAIL / UNVERIFIED.

Then exactly one of:

- **GATE PASSED** — one-line justification; append a dated `Gate $ARGUMENTS passed` entry to
  `docs/protocol.md`; write a short "Before you start" note at the head of the next phase if this
  phase measured anything that changes it; name the first task of the next phase.
- **GATE FAILED** — failing criteria and the smallest fix for each. Do not start the next phase.
  Do not propose relaxing a threshold; if one genuinely must change, say so explicitly and record the
  change and reason in `docs/protocol.md` as a dated deviation.

Be concise. This is a checkpoint, not a report.
