# Exact hosted Atlas browser regressions

This engineering runner closes an execution-coverage gap for two existing UI
regressions. It does not certify mathematics, all regression coverage, or CBZ-0.
The corrected contribution was executed in the hosted browser job at the source cut below; the composed handoff still requires its own fresh run.

## Scope and execution

`tools/run_atlas_browser_regressions.py` loads only the exact focus-ring and
long-reader-selection method IDs declared in `EXPECTED_IDS`. It refuses an empty,
duplicate, missing, extra or renamed inventory. Success requires exactly two
successful methods, no failures/errors/skips, no expected failures/unexpected
successes, and unchanged clean tracked source identity before/after execution.
Both tests explicitly enable Chromium's sandbox and choose the explicit
`ATLAS_BROWSER_EXECUTABLE` override first, then packaged `google-chrome`, then
`chromium`. This matches the existing reader selection and hosted browser policy.
No UI assertion or bound is changed. The first hosted PR #702 attempt at source
`625bfc89975efa1fd725076bc2b0961af2768279` failed: focus selected Chromium and
reported “No usable sandbox”; reader passed with packaged Chrome. That failure
is not erased by this launcher correction. Corrected actual browser execution passed at PR head `c0205ca9e8683b2847008f712582e183f723e186` in both modes, with two successes and no skips (job `123718493908`, run `40950918974`); no local denied launch was retried. This does not qualify the newer composed candidate.

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

Seventeen controls run without launching a browser. They cover inventory, actual
unittest success/failure/error/subtest/skip/expected-failure/unexpected-success,
missing execution, report/exit consistency, source drift, old-output retention,
explicit sandbox parameters, workflow wiring, ten launcher-selection cases and
runner identity correspondence to each selected test launcher. Identity controls
stub version probing and never launch a browser. Six actual runner-source mutants
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
  maintenance owner: none, self-contained; verification: 17 runner controls (32 combined browser-contract controls) and 6
  runner-source mutants per mode locally passed in the fresh recovery checkout; actual hosted browser execution passed in both modes at the recorded `c0205ca9` cut; the composed source still needs fresh execution.
- STATUS, How to verify: run both control commands, then both real runner commands
  in the existing pinned browser-smoke environment; report the original result
  JSON/logs and checked source identity. This does not close five separate
  production-record applicability/live-source tests.
- DECISIONS: add the narrow execution-coverage runner because plain unittest can
  exit zero after skipping required browser methods. Preserve all UI assertions,
  source history, original failure logs and supported browser sandbox. No new UI
  functionality, mathematical claim or final baseline promotion is introduced.
