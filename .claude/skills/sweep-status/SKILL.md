---
name: sweep-status
description: Summarise the state of all background training runs from their status files and logs.
argument-hint: <method filter, optional>
disable-model-invocation: true
allowed-tools: Read, Glob, Bash
---

Summarise training runs under `artifacts/runs/` (filter: `$ARGUMENTS` if given).

For each run read `status.json` and the tail of its log, and check the PID is alive. Produce one
table: method | seed | state (running/done/failed/stalled) | steps / budget | fps | last eval success
| ETA. A run is **stalled** if `status.json` has not updated in > 3 eval intervals.

For failed or stalled runs, show the last 20 log lines and the likely cause. Do not restart or kill
anything; recommend the command and let the user decide.
