# rl-land-on-moving-deck -- see docs/IMPLEMENTATION_PLAN.md for what each phase produces.
#
# Interpreter and tooling are resolved to .venv when it exists and to PATH otherwise. This
# is inherited from Project 4, where `make eval` and `make lint` exited 127 on a clean
# (unactivated) shell and would have failed a gate for an environment reason unrelated to
# the phase under test. Overridable: `make PY=python RUFF=ruff MYPY=mypy PYTEST=pytest`.
PY     ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
RUFF   ?= $(shell [ -x .venv/bin/ruff ]   && echo .venv/bin/ruff   || echo ruff)
MYPY   ?= $(shell [ -x .venv/bin/mypy ]   && echo .venv/bin/mypy   || echo mypy)
PYTEST ?= $(shell [ -x .venv/bin/pytest ] && echo .venv/bin/pytest || echo pytest)

# Training config for `make train-bg CFG=configs/rl/ppo.yaml` (Phase 5).
CFG ?= configs/rl/ppo.yaml
# Throughput measurement (Phase 0). STEPS is per (vec_cls, n_envs, act) row.
STEPS ?= 6000
THROUGHPUT_CSV ?= results/env_throughput.csv

.PHONY: test lint format throughput \
        deck-stats baselines dmf-forecasters train-bg eval bench report all

# --- implemented ------------------------------------------------------------------

test:   ; $(PYTEST)
lint:   ; $(RUFF) check src tests scripts && $(RUFF) format --check src tests scripts && $(MYPY) src
format: ; $(RUFF) format src tests scripts && $(RUFF) check --fix src tests scripts

# Phase 0: PyBullet env steps/s at 1/8/16 SubprocVecEnv workers, RPM and VEL actions.
# This number sizes the Phase 5 training budget recorded in P3-D1.
throughput: ; $(PY) scripts/env_throughput.py --out $(THROUGHPUT_CSV) --steps $(STEPS)

# --- not yet implemented ----------------------------------------------------------
# One stub per command in CLAUDE.md. Each exits 0 so that `make all` lists every stage
# (Gate 9 reads `make all` as the stage list). Replace the stub in its own phase; do not
# add a second target beside it.

# Phase 1 -- deck-bridge-engineer: scripts/deck_stats.py -> results/deck_stats.csv
deck-stats:      ; @echo "not implemented: phase 1"
# Phase 3 -- controls-engineer + eval-auditor: four classical controllers -> results/e01/
baselines:       ; @echo "not implemented: phase 3"
# Phase 4 -- deck-bridge-engineer: dmf corpus + dlinear_ols + tcn -> artifacts/dmf/
dmf-forecasters: ; @echo "not implemented: phase 4"
# Phase 5 -- rl-trainer: background training run; status in artifacts/runs/*/status.json
train-bg:        ; @echo "not implemented: phase 5 (CFG=$(CFG))"
# Phase 7 -- eval-auditor: full evaluation matrix on the frozen episode lists -> results/
eval:            ; @echo "not implemented: phase 7"
# Phase 8 -- deploy-benchmarker: ONNX export, parity, latency -> results/latency*
bench:           ; @echo "not implemented: phase 8"
# Phase 7/9 -- eval-auditor: re-render results/results.md from the committed CSVs
report:          ; @echo "not implemented: phase 7"

# The whole project in the order the methodology requires. Stubs today; each phase
# replaces its own line's target.
all: deck-stats baselines dmf-forecasters eval bench report
