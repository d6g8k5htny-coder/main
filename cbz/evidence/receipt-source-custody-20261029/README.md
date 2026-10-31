# Exact source custody for Main formal receipts

The old producer could execute modified owned bytes, then name the unchanged
Git HEAD in a kernel-checked receipt. The actual RED specimen changed only a
Lean comment and its manifest hash: Lean passed, but receipt manifest7b544201
was absent from named commit5494cfe2. That historical specimen is **not admissible
commit-qualified evidence**. Its receipt, raw output and input patch are retained
under `red/`; early unimplemented-control/first-failure logs are retained too
and are not counted as passing controls or qualifying mutation kills.

The repaired producer binds owned source bytes and executable modes to raw Git
blobs before Lean and after the final source/version check, using the same HEAD.
It includes the manifest, declared package files, actual loaded producer/helper,
live consumer, consumer tests and parent aggregate workflow. Replacement objects
and ambient Git repository/index/config overrides cannot substitute another tree.
A different package argument is rejected before writes. Valid-package custody
refusal archives prior receipt/logs before leaving the current path without
success. Unrelated repository edits and generated `.lake` outputs are permitted.
The finite claim is pre/post endpoint custody in a trusted runner, not continuous
atomic isolation from a malicious concurrent writer.

Observed on sourcece4277c8: nine affected test modules passed219/219 in each
CPython3.11.16 mode, zero skips/failures/errors; both real Lean executions passed.
The source-identical mergeaa0d59ad incorporates actual Main22567 and received a
fresh real gate: build, leanchecker,31 target axiom checks,45 compiled constants,
365 dependency closure, zero axioms,11 separately bound IR names and10 genuine
negative controls. `SOURCE_BINDING.json` records exact cuts and owned hashes;
the receipt and every bound raw log are in
`../kernel/main-source-custody-20261029/`. All94 peer composition evidence hashes
were retained and verified. Existing earlier composition evidence remains intact.

Reproduce from a Git working clone with official Lean4.34.1 available,
CPython3.11.16 first in PATH and `LEAN_NUM_THREADS=2`:

```sh
python3 -B -S -m unittest tests.test_formal_source_custody -v
python3 -B -O -S -m unittest tests.test_formal_source_custody -v
python3 -B -S tools/formal_gate_check.py --run-lean
```

The custody module runs20 real-Git endpoint cases, one mandatory-binding test
and a nine-source-mutant test. Each mutant runs the complete22 internal controls,
requires exactly its declared assertion failures and zero errors/skips; the
unmutated baseline must pass first. The shared compiled-auditor harness is also
self-tested. Its seven policy mutants must all be killed. The formal gate's
existing runtime controls verify both endpoint calls and reject HEAD drift.

The full219 affected-run commands are preserved in `final/fresh-focused-results.json`;
local historical runtime paths in that record are not prerequisites. The normal
and optimized raw logs preserve all counts. Independent source review on the
source-identical5d25cadf production files found no remaining issue and ran the
custody, synthetic and workflow controls in both modes. Its first accidental
launch against a conflict-bearing tree is explicitly unqualified in that review.

Current-head hosted checks, full top-level regression and final federation gates
remain integration obligations. No theorem/bound, scientific alignment, model
hypothesis, analytic enclosure or parent mathematical claim was promoted.
