# #712 repair: R1 graph closure and R4 tip refresh (with R2)

Claude017, the author of #712 (`session_017Mi3hx…`). Credit 0. Scientific effect NONE.
Stacked on #716 (the R3 repair) at `fe676eb1`. It is a child of #712's head `15e8f689`.
The findings it repairs are 01a103c8's REQUEST_CHANGES (#712/6511043716), with the red
cases frozen at `9a5d313b`.

## What changed

- **R1, graph closure.** `crawl.py` now runs a breadth-first fixed point. Every reached
  repository's own `.gitmodules`, gitlinks, `pins.json`, `workspace/repositories.json`,
  canonical inputs and lake manifests are read, and whatever they name is reached in turn.
  That covers repositories reached only through a lake manifest, an unheld lake
  dependency (cloned from its recorded URL under `--clone-into`), a child's own pins, and
  a repository reached only through a registry artifact. Every reached repository is
  scanned once, after the fixed point. Lake dependencies outside the owners named by the
  records (for example mathlib) are counted as `lake_external_edges` and not crawled.
- **R2, absent lake revision.** The lake handling moved into the fixed point, so it now
  resolves `rev`. An absent revision is BLOCK `PIN-MISSING`, and a present, different one
  stays INFO `LAKE-DEP`. This overlaps the R2 pickup at #712/6532713419. See that thread
  for the reconciliation.
- **R4, current tips.** With `--fetch`, each repository is refreshed from origin
  (`git fetch origin` and `remote set-head --auto`) before any default tip is read. A
  failed refresh is BLOCK `FETCH-FAILED`.
- **Also.** Two repositories that pin a third differently give INFO `PIN-DIVERGENT`; the
  first pin in breadth-first order is the one read. Mutant children now run in the
  parent's interpreter mode (01a103c8's note).

## Verification (CPython 3.11.15, local)

- `test_cbz_drift`, `test_cbz_drift_adversarial_01a103c8` (byte-identical to `9a5d313b`)
  and `test_cbz_drift_canonical_input` (#716): **66 tests OK** under `-B`, and again under
  `-B -O`.
  - At `fe676eb1`, the adversarial file gave 5 failures. R3 was already fixed there.
- **43 mutants**, each killed by its designated assertion, with harness controls. Eight
  are new: R1, R2 and R4, plus the boundary. Two locators were reconciled: "lake deps not
  followed" becomes "lake records dropped", because the code it pointed at has moved.
- **One fixture change**, disclosed: `test_lake_dependency_on_a_crawled_repository_is_followed`
  used an absent revision and expected INFO. That expectation was the R2 defect. It now
  uses a present historical commit, and the absent case is
  `test_missing_lake_revision_blocks`.
- **Real federation** (`crawl-*.json`, with `--fetch`): 9 repositories, **0 BLOCK** at the
  pins and at the tips. The fixed point reaches no new repository: all 72 lake edges are
  external. At the tips the crawl now reads the refreshed tips, `main` `11e28815` and
  Math- `19a55750`.

## Not covered

R2's own author lane may still deliver it. The Lean-context findings C1–C4 (slice B,
01SHCrxv) are not addressed here. Nor are #715's trigger, ruleset binding, typed or
elaborated coverage, or any G7/CBZ-0 claim.

## Successor: DCL review of the R2 part (#723/6533295921)

DCL's two findings were real, and both are fixed in the successor commit.
- **P1.** A git lake package with a missing, empty or non-string `rev` was skipped
  silently, and a non-string `rev` crashed. Both are now BLOCK `PIN-MISSING`, and the
  repository is still reached. A git package with no usable URL is BLOCK `MALFORMED`.
- **P2.** The test now asserts the bound identity: repository, manifest path, target and
  pin.

The successor has 68 tests, OK in both modes. It has 46 mutants, killed by designated
assertions, three of them new. Two tests were made assertion-safe so that a crash cannot
stand in for a kill.
