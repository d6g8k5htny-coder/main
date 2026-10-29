# Stale open PR disposition evidence

Target `main` at `81acc060341e8a31cc4cd4f9523ef71fb6c06175`. These are recommendations from the documented rules in `pr_evidence.py`, not dispositions. Layer A fields are lexical signals, not certification.

| PR | Base | Recommendation | Reason | Merge onto main | Signals |
|---|---|---|---|---|---|
| #333 | `main` | **INTEGRATE-CANDIDATE** | clean merge, no Layer A signals, a PASS review at the current head | clean | none |
| #336 | `codex/spectral-amplitude-crossover-20261008` | **STACK-DEPENDENT** | stacked on branch codex/spectral-amplitude-crossover-20261008, which no open PR owns and which is not on the target | 5 conflict(s) | none |
| #338 | `main` | **INTEGRATE-CANDIDATE** | clean merge, no Layer A signals, a PASS review at the current head | clean | none |
| #340 | `codex/stationary-quartic-cusp-20261009` | **STACK-DEPENDENT** | stacked on open PR #336; disposition the stack from the bottom | 5 conflict(s) | none |
| #341 | `main` | **INTEGRATE-CANDIDATE** | clean merge, no Layer A signals, a PASS review at the current head | clean | none |
| #342 | `main` | **REPAIR-OR-ARCHIVE** | 4 check run(s) at the head did not pass | clean | float on added lines of 2 other .py file(s), for review |
| #344 | `main` | **LAYER-A-REVIEW-FIRST** | lexical Layer A signals in its diff; review them before any integration | clean | float on added lines of 1 certificate-like file(s) |
| #345 | `main` | **INTEGRATE-CANDIDATE** | clean merge, no Layer A signals, a PASS review at the current head | clean | none |
| #346 | `codex/exact-population-certificate-20261009-root01a11cbf` | **STACK-DEPENDENT** | stacked on open PR #343; disposition the stack from the bottom | 5 conflict(s) | none |
| #347 | `main` | **LAYER-A-REVIEW-FIRST** | lexical Layer A signals in its diff; review them before any integration | clean | added skips: 2 |
| #349 | `codex/cusp-analytic-guards-20261009-root01a11cbf` | **STACK-DEPENDENT** | stacked on open PR #346; disposition the stack from the bottom | 5 conflict(s) | float on added lines of 5 other .py file(s), for review |
| #358 | `main` | **QUARANTINE-ARCHIVE** | conflicts with the target in 2 path(s) | 2 conflict(s) | none |
| #363 | `main` | **REPAIR-OR-ARCHIVE** | 1 check run(s) at the head did not pass | clean | float on added lines of 1 other .py file(s), for review |
| #385 | `codex/global-cusp-h0-validation-20261009-root01a11cbf` | **STACK-DEPENDENT** | stacked on open PR #350; disposition the stack from the bottom | clean | none |
| #389 | `main` | **QUARANTINE-ARCHIVE** | the PR itself says it is not for merge | clean | none |
| #407 | `main` | **INTEGRATE-CANDIDATE** | clean merge, no Layer A signals, a PASS review at the current head | clean | none |
| #429 | `main` | **INTEGRATE-CANDIDATE** | clean merge, no Layer A signals, a PASS review at the current head | clean | none |
| #458 | `main` | **INTEGRATE-CANDIDATE** | clean merge, no Layer A signals, a PASS review at the current head | clean | none |
| #464 | `main` | **REPAIR-OR-ARCHIVE** | 2 check run(s) at the head did not pass | clean | float on added lines of 1 other .py file(s), for review |
| #482 | `main` | **REPAIR-OR-ARCHIVE** | 1 check run(s) at the head did not pass | clean | float on added lines of 1 other .py file(s), for review |
| #484 | `main` | **QUARANTINE-ARCHIVE** | conflicts with the target in 1 path(s) | 1 conflict(s) | none |
| #499 | `main` | **REPAIR-OR-ARCHIVE** | 2 check run(s) at the head did not pass | clean | none |
| #503 | `main` | **INTEGRATE-CANDIDATE-UNREVIEWED** | clean merge and no Layer A signals, but no PASS review at the current head | clean | float on added lines of 2 other .py file(s), for review |
| #624 | `grok16/v20-planner-source-carrier-20261025` | **STACK-DEPENDENT** | stacked on branch grok16/v20-planner-source-carrier-20261025, which no open PR owns and which is not on the target | clean | none |
| #629 | `grok16/v20-planner-source-carrier-20261025` | **STACK-DEPENDENT** | stacked on branch grok16/v20-planner-source-carrier-20261025, which no open PR owns and which is not on the target | clean | float on added lines of 3 other .py file(s), for review |
| #630 | `main` | **INTEGRATE-CANDIDATE-UNREVIEWED** | clean merge and no Layer A signals, but no PASS review at the current head | clean | none |
