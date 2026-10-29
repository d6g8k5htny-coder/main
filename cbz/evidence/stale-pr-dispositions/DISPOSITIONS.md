# Stale open PR disposition evidence

Target `main` at `e6f2653841053004d213b3647a7e14e799f503b2`. These are recommendations from the documented rules in `pr_evidence.py`, not dispositions. Layer A fields are lexical signals, not certification.

| PR | Base | Recommendation | Reason | Merge onto main | Signals |
|---|---|---|---|---|---|
| #344 | `main` | **LAYER-A-REVIEW-FIRST** | lexical Layer A signals in its diff; review them before any integration | clean | float on added lines of 1 certificate-like file(s) |
| #345 | `main` | **INTEGRATE-CANDIDATE** | clean merge, no Layer A signals, a PASS review at the current head | clean | none |
| #346 | `codex/exact-population-certificate-20261009-root01a11cbf` | **STACK-DEPENDENT** | stacked on open PR #343; disposition the stack from the bottom | 5 conflict(s) | none |
| #347 | `main` | **QUARANTINE-ARCHIVE** | conflicts with the target in 1 path(s) | 1 conflict(s) | added skips: 2 |
| #349 | `codex/cusp-analytic-guards-20261009-root01a11cbf` | **STACK-DEPENDENT** | stacked on open PR #346; disposition the stack from the bottom | 5 conflict(s) | float on added lines of 5 other .py file(s), for review |
| #358 | `main` | **QUARANTINE-ARCHIVE** | conflicts with the target in 2 path(s) | 2 conflict(s) | none |
| #385 | `codex/global-cusp-h0-validation-20261009-root01a11cbf` | **STACK-DEPENDENT** | stacked on open PR #350; disposition the stack from the bottom | clean | none |
| #389 | `main` | **INTEGRATE-CANDIDATE-UNREVIEWED** | clean merge and no Layer A signals, but no PASS review at the current head | clean | float on added lines of 3 other .py file(s), for review |
| #499 | `main` | **REPAIR-OR-ARCHIVE** | 2 check run(s) at the head did not pass | clean | none |
| #503 | `main` | **INTEGRATE-CANDIDATE-UNREVIEWED** | clean merge and no Layer A signals, but no PASS review at the current head | clean | float on added lines of 2 other .py file(s), for review |
