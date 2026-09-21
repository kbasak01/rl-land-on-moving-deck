---
name: full-audit
description: Run the complete pre-release validation protocol (plan section 7) plus an adversarial review, and produce docs/audit_report.md.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash, Write
---

Run the pre-release audit.

1. `make test lint`. Then walk every checkbox in §7 of `docs/IMPLEMENTATION_PLAN.md`, verifying each
   against committed artifacts, and mark PASS / FAIL / UNVERIFIED with evidence.
2. Trace every number in `README.md` to a committed file (grep the CSVs). List untraceable numbers.
3. Verify `make report` re-renders `results/results.md` byte-identically.
4. Check `docs/protocol.md`: every deviation from P3-D1 is dated and justified; hypotheses were not
   reworded after results.
5. Delegate to `results-skeptic` for a full adversarial review.
6. Write `docs/audit_report.md`: BLOCKING / SHOULD FIX / NOTE, checklist table, traceability table.

Do not fix anything during the audit. End with the count of BLOCKING findings.
