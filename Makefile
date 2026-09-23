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
# Worker processes for the Phase 1 deck-statistics sweep (split over the 96 grid cells).
WORKERS ?= 24
THROUGHPUT_CSV ?= results/env_throughput.csv
# Phase 2 throughput: the same measurement with the deck body and the motion bridge in the
# loop. Written to its own file so the committed Gate 0 artifact is left exactly as it was.
THROUGHPUT_LANDING_CSV ?= results/env_throughput_landing.csv

.PHONY: test lint format throughput throughput-landing env-sanity \
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

# Phase 1 -- deck-bridge-engineer: deck-point statistics over all 96 grid cells at the model
# physics rate, plus the pre-registered lambda feasibility verdict under both candidate
# denominators. Parallel over cells; the CSVs do not depend on WORKERS.
deck-stats: ; $(PY) scripts/deck_stats.py --out results/deck_stats.csv --seeds-out results/deck_stats_seeds.csv --feasibility-out results/deck_feasibility.csv --workers $(WORKERS) --threshold-reference all

# Phase 2 -- sim-env-engineer: random and hover policies over 200 episodes per (policy, sea
# state) on the `id` split's val partition, with the six outcome-class fractions and the
# analytic-vs-contact touchdown disagreement rate Gate 2 reads. Parallel over chunks of
# episodes; the CSVs do not depend on WORKERS.
env-sanity: ; $(PY) scripts/env_sanity.py --out results/e00_env_sanity.csv --episodes-out results/e00_env_sanity_episodes.csv --workers $(WORKERS)

# Phase 2 -- sim-env-engineer: throughput with the deck body and the motion bridge in the
# loop. P0-D2's HoverAviary number is the ceiling; this is the estimate P3-D1 sizes the
# training budget from.
throughput-landing: ; $(PY) scripts/env_throughput.py --env landing --policies random hold --out $(THROUGHPUT_LANDING_CSV) --steps $(STEPS)
# Phase 3 -- eval-auditor: the four classical controllers on the frozen episode lists
# (results/episodes/, checked against MANIFEST.csv first) -> results/e01/{episodes,summary}.csv
# and success_vs_seastate.md rendered from the CSV. Parallel over chunks of episodes; the CSVs
# do not depend on WORKERS (volatile facts go to results/e01/run_info.json).
baselines: ; $(PY) scripts/eval_baselines.py --out-dir results/e01 --workers $(WORKERS)
# Phase 4 -- deck-bridge-engineer: the dmf corpus, then the four forecasters fitted on the
# P3-D2 dev pool (729 train / 135 tune realizations), driven as a library by
# rld.deck.forecast_fit. Everything is written under artifacts/dmf/ (gitignored) plus the
# committed record results/forecast/fit_manifest.json + fit_keys.json. Nothing runs inside
# third_party/ (it is read-only). The corpus (2304 realizations, ~1.1 GB, ~20 s at 24
# workers) is regenerated only if its manifest is absent. The TCN fits take hours on the GPU:
# launch in the background with output redirected to a log (no tee: /bin/sh has no
# pipefail, and a tee would turn a failed fit into a green make).
DMF_ROOT    ?= third_party/deck-motion-forecast
DMF_CORPUS  ?= artifacts/dmf/corpus
DMF_MODELS  ?= dlinear_ols residual_interval tcn tcn_quantile
DMF_SEEDS   ?= 0 1 2
DMF_DEVICE  ?= cuda
$(DMF_CORPUS)/manifest.parquet:
	$(PY) $(DMF_ROOT)/scripts/generate_corpus.py --config $(DMF_ROOT)/configs/sim/corpus.yaml --out $(DMF_CORPUS) --workers $(WORKERS)
dmf-forecasters: $(DMF_CORPUS)/manifest.parquet
	$(PY) scripts/fit_dmf_forecasters.py --corpus $(DMF_CORPUS) --out artifacts/dmf --models $(DMF_MODELS) --seeds $(DMF_SEEDS) --device $(DMF_DEVICE)
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
all: deck-stats env-sanity baselines dmf-forecasters eval bench report
