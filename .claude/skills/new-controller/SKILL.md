---
name: new-controller
description: Scaffold a new landing controller (classical or learned wrapper) with its file, YAML config, registry entry and tests.
argument-hint: <controller_name> <one-line description>
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

Create controller `$ARGUMENTS` following the architecture rules in CLAUDE.md.

1. `src/rld/control/<name>.py` implementing `Controller` (`act(obs) -> np.ndarray[3]` in the shared
   normalised velocity-setpoint space, `reset(seed)`); docstring with units and whether it is
   privileged.
2. `configs/control/<name>.yaml` — every gain and threshold lives here, none in code.
3. Registry entry in `src/rld/control/registry.py` with `privileged: bool`.
4. `tests/test_control_<name>.py`: output shape and bounds; determinism under reset; static-pad
   landing succeeds; privileged flag correct.
5. Run `make test lint`. Report files created and test status. Do not evaluate on the frozen
   episode lists — that is the `eval-auditor`'s job.
