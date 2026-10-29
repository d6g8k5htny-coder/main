# Exact hosted Atlas browser regressions

This engineering runner closes an execution-coverage gap for two existing UI
regressions. It does not certify mathematics, all regression coverage, or CBZ-0.
Actual browser execution passed on the corrected historical source documented below.
A later composition requires its own hosted qualification; historical success is
not transferred to an untested tree.

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
is not erased by this launcher correction. Corrected actual browser execution passed in both modes on the source below;
no local denied launch was retried.

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

Thirty controls run without launching a browser. The original seventeen are retained; thirteen additional controls qualify the mutation grader’s lifecycle and assertion-origin checks. They cover inventory, actual
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
  maintenance owner: none, self-contained; historical verification: 17 runner controls (32 combined browser-contract controls) and 6
  runner-source mutants per mode locally passed in the fresh recovery checkout; actual hosted execution passed on the historical corrected tree below,
  with new composition qualification still pending.
- STATUS, How to verify: run both control commands, then both real runner commands
  in the existing pinned browser-smoke environment; report the original result
  JSON/logs and checked source identity. This does not close five separate
  production-record applicability/live-source tests.
- DECISIONS: add the narrow execution-coverage runner because plain unittest can
  exit zero after skipping required browser methods. Preserve all UI assertions,
  source history, original failure logs and supported browser sandbox. No new UI
  functionality, mathematical claim or final baseline promotion is introduced.

## Retained hosted execution evidence

The original failed run 40936994120 and corrected run 40950918974 are retained
in [the evidence manifest](evidence/atlas-browser-20261029/manifest.json).
Both downloaded artifact ZIPs were independently hashed against their recorded
SHA-256 identities before extracting these text records. Screenshot member hashes
are retained in the manifest; screenshots themselves remain in the hosted ZIPs.

Corrected source `c0205ca9e8683b2847008f712582e183f723e186` was tested through
merge `8bd88e5952e4a67ff394e1cab3c272e7ace2f961`, tree
`c9e3a5656fc95feef3bd360af3ea393846afd1e8`, in browser-smoke job
123718493908. Its normal and optimized result records each contain exactly two
successful expected method IDs, zero errors, failures or skips, and optimize flags
0 and 1 respectively. Both used `/opt/google/chrome/google-chrome`, version
156.0.8078.47, with the existing sandbox-enabled tests.

Corrected artifact 12931733641 has SHA-256
`2bcd2bd3ceec0641c56f52bab6159e141e825a10042d9276bd716e7f12c1df36`.
Original artifact 12924209403 has SHA-256
`e6ac7e329f63c55dc1188f6dddeacfb3a72b1f60a2327dcaac657cdaee0b3f8a`.
The original focus failure remains in its result and stderr records. These are
engineering observations only, not mathematical evidence or final G5 completion.
The later composition onto Main `1e2be44c69e08e47472041ac329e1b27f7dec793`
preserves this evidence as historical; it still needs a new hosted execution.


## 2026-10-29 assertion-origin amendment

At parent tree `6786ec405cef50e0fef73c7fcd9ca3945c555230`, the six-mutant
meta-test accepted setup-only AssertionErrors as kills. An independent normal and
optimized reproduction is retained in `evidence/browser-origin-20261029/`.
This demonstrates a future-use grading defect; it does not establish that any
historical finite kill or actual browser result was false.

The test-only grader now captures actual exception/code identity before unittest
formats/clears traceback data. Admission requires the selected synchronous body
exactly once, the complete terminal population, no fixture/error/skip/expected
failure, and only the selected body plus explicitly trusted unittest assertion
helper code on the escaping failure path. Subject, helper, callback, forged-file,
setup, teardown and cleanup assertions are refused. Deferred/non-None body returns
and missing/duplicate terminal success events are refused. Each production-source
mutant first runs its unmodified designated baseline. All original assertions,
production runner/browser sources, workflow bytes and sandbox policy are unchanged.

Current scoped controls: 30 methods per mode, or 45 when combined with the two
existing browser-contract modules; six original production mutants per mode.
Six additional grader safeguard mutations are checked by an unchanged trusted
copy of the repaired raw-origin classifier, not by their own mutated classifier:

```
python -B docs/cbz/evidence/browser-origin-20261029/check_grader_mutations.py
python -B -O docs/cbz/evidence/browser-origin-20261029/check_grader_mutations.py
```

The deferred-body mutation deliberately removes both static and runtime refusal;
its unawaited-coroutine warning is retained in stderr as an adverse observation,
not accepted as a kill. The required kill is a genuine self-test assertion.
These are finite engineering controls in a trusted interpreter with frozen tests;
this is not hostile-process containment or an exhaustive proof of unittest.
New composed-tree full regression/actual-browser execution remains pending.
Canonical tooling registration should include this amendment and its self-tests,
not relabel historical source receipts as current-source qualification.


## Exact code-object identity correction

The subsequent reviewer found that Python CodeType objects can be structurally
`==` despite distinct identities, including after `co_filename` replacement.
At published parent `8d24c37ff52c012294d71bbcbf08bd24fae4e22a`, set membership
through a frozenset used equality and admitted such a cloned assertion helper.
The new clone regression first failed on that parent. Approved code references
are now retained in a tuple and checked with explicit `is`, never equality or
hash-set membership. All prior controls remain; current counts are31 module
methods /46 combined methods per mode, plus seven grader safeguard mutants.
Evidence is in `evidence/browser-code-identity-20261029/`. Earlier30/45-method
logs and six-mutant observations remain historical at their original source.
This closes the exact-provenance contract defect, not a mathematical proof gap;
no evidence shows earlier actual finite kills or browser outputs were false.
