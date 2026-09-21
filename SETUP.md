# SETUP.md — standing up rl-land-on-moving-deck for Claude Code

This kit contains everything Claude Code needs before the first prompt: the plan, project memory,
seven subagents, eight skills, permissions and a lint hook, the decision-log template, and the
prompt sequence.

```
rl-land-on-moving-deck-kit/
├── SETUP.md                     ← you are here
├── CLAUDE.md                    project memory (loaded every session)
├── .claude/
│   ├── settings.json            permissions, env vars, ruff PostToolUse hook (needs jq)
│   ├── agents/                  7 subagents
│   └── skills/                  8 skills (4 slash commands, 4 auto-loaded references)
└── docs/
    ├── IMPLEMENTATION_PLAN.md   the sequential build plan, phases 0–9 with gates
    ├── KICKOFF_PROMPTS.md       one prompt per phase + resume and recovery prompts
    ├── protocol.md              decision-log template (P0-D1, P1-D1, P3-D1 frozen protocol)
    └── findings.md              findings template
```

## 1. Create the repo (≈10 min, you run these)

```bash
sudo apt-get install -y build-essential python3.12-dev python3.12-venv jq git-lfs
mkdir rl-land-on-moving-deck && cd rl-land-on-moving-deck && git init

git submodule add https://github.com/kbasak01/deck-motion-forecast third_party/deck-motion-forecast
git submodule add https://github.com/learnsyslab/gym-pybullet-drones third_party/gym-pybullet-drones
git submodule status            # record both SHAs in docs/protocol.md P0-D1

cp -r /path/to/rl-land-on-moving-deck-kit/{CLAUDE.md,.claude,docs} .
mkdir -p src/rld configs scripts tests results artifacts
printf "artifacts/\n.venv/\n__pycache__/\n*.egg-info/\nruns/\n" > .gitignore
git add -A && git commit -m "Phase 0: scaffold from kit"
```

Then open Claude Code in the repo root and paste **Prompt 0** from `docs/KICKOFF_PROMPTS.md`.
Claude writes `pyproject.toml` and the Makefile and hands you the install commands:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e third_party/deck-motion-forecast
pip install -e third_party/gym-pybullet-drones     # pybullet builds from source here, several minutes
pip install -e ".[dev]"
```

Why submodules + editable installs: dmf finds its vessel YAMLs relative to its own source tree, so a
plain `pip install git+…` breaks it; and pinning both SHAs makes every result reproducible.

## 2. What each piece does

**Subagents** (`.claude/agents/`, invoked by delegation — "delegate this to X" — or automatically
from their descriptions):

| agent | phases | writes |
|---|---|---|
| deck-bridge-engineer | 1, 4 | `src/rld/deck/` |
| sim-env-engineer | 2 | `src/rld/envs/` |
| controls-engineer | 3, 4 | `src/rld/control/` |
| rl-trainer | 5, 6 | `src/rld/rl/` |
| eval-auditor | 3, 7 | `src/rld/eval/`, `results/` — read-only over rl/ and control/ |
| deploy-benchmarker | 8 | `src/rld/deploy/` |
| results-skeptic | 3–9 | nothing (read-only reviewer) |

**Skills** (`.claude/skills/`):

| skill | how it runs | purpose |
|---|---|---|
| `/phase-gate N` | you type it | gate check against committed artifacts |
| `/new-controller name desc` | you type it | scaffold controller + config + registry + test |
| `/sweep-status [method]` | you type it | table of background training runs |
| `/full-audit` | you type it | pre-release validation + adversarial review |
| landing-protocol | auto-loaded | success criteria, metrics, statistics, reporting rules |
| deck-scaling-physics | auto-loaded | Froude scaling, kinematics, dmf facts, phase defect |
| pybullet-moving-platform | auto-loaded | platform driving, contact, touchdown traps |
| rl-experiment | auto-loaded (or `/rl-experiment`) | launching, VecNormalize, budgets, hacking audit |

**settings.json**: allows make/pytest/ruff/mypy/git-read; asks before commit, push, pip, rm, kill;
denies any write to `third_party/` (dmf stays pristine) and `artifacts/`. The PostToolUse hook runs
`ruff check` on every edited `.py` file and never blocks.

## 3. Operating rhythm

1. One session per phase, plan mode first, `/clear` after the gate.
2. **Long compute runs outside the chat.** Training is launched with `make train-bg` (or you paste
   the sweep commands into tmux). Claude Code's shell commands time out after minutes; a 10 M-step
   PPO run does not. Come back with `/sweep-status` and the resume prompt.
3. You approve P3-D1 (the frozen protocol) personally. That approval is what makes the later
   results credible.
4. Commit at every gate. Claude will ask before each commit.

## 4. Schedule (working days; compute runs overnight)

| phase | work | compute |
|---|---|---|
| 0 bootstrap | 0.5 | — |
| 1 deck bridge | 1.5 | minutes |
| 2 landing env | 2 | minutes |
| 3 baselines + protocol freeze | 1.5 | ~1 h |
| 4 forecaster integration | 1.5 | ~1 h (dmf corpus + fits) |
| 5 PPO + SAC, 5 seeds | 2 | 1–2 days |
| 6 residual / forecast / sinusoid, 5 seeds | 1.5 | ~1 day |
| 7 shift evaluation | 2 | ~0.5 day |
| 8 ONNX + latency | 0.5–1 | ~1 h |
| 9 docs + audit | 1 | — |
| **total** | **~14** | **~3–5 days** |

Fast path if interviews are near (< 3 weeks): Phases 0–3, then `ppo` and `residual_ppo` only in
Phases 5–6 with 5 seeds, Phase 7 on `id` + `unseen_seastate` + the H4 cross, skip Phase 8 until later.
Record the reduced scope as a dated deviation — do not present it as the full study.

## 5. Your validation pass before finalising

Run `/full-audit`, then personally check three things the audit cannot judge for you:
(1) watch the SS5 GIFs frame by frame for platform jitter or drone tunnelling;
(2) re-read P3-D1 against `docs/findings.md` and confirm no hypothesis was reworded;
(3) read the README limitations aloud as if presenting to a reviewer who has read Angelis et al.
