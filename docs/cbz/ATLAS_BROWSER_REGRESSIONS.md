# Exact hosted Atlas browser regressions

This engineering runner closes an execution-coverage gap for two existing UI
regressions. It does not certify mathematics, all regression coverage, or CBZ-0.
Actual browser execution on this contribution is pending hosted qualification.

## Scope and execution

`tools/run_atlas_browser_regressions.py` loads only the exact focus-ring and
long-reader-selection method IDs declared in `EXPECTED_IDS`. It refuses an empty,
duplicate, missing, extra or renamed inventory. Success requires exactly two
successful methods, no failures/errors/skips, no expected failures/unexpected
successes, and unchanged clean tracked source identity before/after execution.
The focus test now explicitly enables Chromium's sandbox, matching the reader
test and existing hosted browser policy. No UI assertion or bound is changed.

The existing public-shop `browser-smoke` job supplies the hash-pinned
`tests/browser-requirements.txt` dependencies and packaged Chrome. No permission,
action pin, dependency pin, browser flag disabling security, or browser download
is introduced. Existing browser checks remain before the new runner steps.

After installing the existing pinned requirements in that supported environment:

```
python -B tools/run_atlas_browser_regressions.py --output browser-evidence/atlas-regressions/normal
python -B -O tools/run_atlas_browser_regressions.py --output browser-evidence/atlas-regressions/optimized
```

Each output directory must be new; reuse fails without overwriting old evidence.
Each mode records a result JSON and the test-produced screenshots/measurements.
CI separately retains complete stdout/stderr, runs both modes even if one fails,
and returns failure if either fails. The existing always-upload browser artifact
preserves these files. Runtime metadata records Python/Playwright and each chosen
browser launcher path, version output and launcher-file SHA256 (a launcher may be
a wrapper, so this is not a claim to hash every underlying browser binary).
Git commit/tree and tracked cleanliness are local observations, not cryptographic
attestation or hostile-process containment. A missing dependency, denied browser
launch, assertion failure or source drift is failure; do not relax the sandbox or
reinterpret an optional skip as success. The prior local socket denial is not
retried by this contribution.

## Self-verification

```
python -B -S -m unittest tests.test_atlas_browser_regressions_runner -v
python -B -O -S -m unittest tests.test_atlas_browser_regressions_runner -v
```

Fifteen controls run without launching a browser. They cover inventory, actual
unittest success/failure/error/subtest/skip/expected-failure/unexpected-success,
missing execution, report/exit consistency, source drift, old-output retention,
explicit sandbox parameters and workflow wiring. Six actual runner-source mutants
must fail their designated assertions with zero errors or skips: bypass inventory,
force green, ignore source drift, overwrite old evidence, return wrong exit and
retain a success report after exception. The unchanged control must pass. Finite
controls establish these contracts, not universal test-runner soundness.

## Canonical registration addendum for the integrator

Shared canonical files remain owned by the current baseline writer. Compose these
facts into their current versions rather than restoring the historical STATUS:

- GOVERNANCE/TOOLING: `tools/run_atlas_browser_regressions.py`; purpose: fixed-ID,
  zero-skip execution of two Atlas UI regressions; criterion: G5 execution coverage
  and G11 documented tooling; usage/self-verification: exact commands above;
  maintenance owner: none, self-contained; verification: 15 controls and 6 source
  mutants per mode locally passed; actual hosted browser result remains pending.
- STATUS, How to verify: run both control commands, then both real runner commands
  in the existing pinned browser-smoke environment; report the original result
  JSON/logs and checked source identity. This does not close five separate
  production-record applicability/live-source tests.
- DECISIONS: add the narrow execution-coverage runner because plain unittest can
  exit zero after skipping required browser methods. Preserve all UI assertions,
  source history, original failure logs and supported browser sandbox. No new UI
  functionality, mathematical claim or final baseline promotion is introduced.
