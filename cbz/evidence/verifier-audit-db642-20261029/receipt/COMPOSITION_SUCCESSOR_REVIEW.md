# Final Main successor composition review

Reviewer: OpenAI / Codex, agent `/root/main_composition_review`, 2026-10-29. Separate engineering review of the delta from the previously accepted `da739673`; this is not independent scientific alignment or a fresh Lean/kernel result.

**Ready-to-publish on reviewed source: YES. Ready-to-merge on reviewed source: YES.** No P0/P1/P2/P3 actionable finding was identified. Required hosted qualification on this exact combined source remains separate and must succeed before actual integration. Neither the predecessor review nor standalone PR689/PR692 hosted results supply that qualification.

## Exact reviewed source

| Identity | Value |
|---|---|
| Worktree | `/workspace/scratch/db6428847655/cbz/worktrees/main-composition` |
| Successor merge | `627b4049884d0b64ce879838361eb0c02e03a31f` |
| Tree | `a7e0ee448e512df80b69883d8af86879b325f241` |
| First native parent | `da7396732233dbd66745c13b160a9a104f91341e` |
| Second native parent | `50575e9823b883f7d766e0d9a45008c5e49c2fc6` |
| Manifest SHA-256 | `34767f4d89e0d1b5135a7425fc617a471238ea2a64da7e3ad78080490d31c914` |

I read the appended brief ruling, `PR689_505_DESCRIPTION.md`, successor report/evidence, full 13-path successor diff, relevant helper/tests and workflow, and the exact second-parent delta. Direct Git inspection confirms these identities and a clean worktree. The review stops at frozen `50575e98`; it does not silently include later PR689 changes.

## Findings and source judgments

No actionable findings. These are the specific compatibility and integration decisions, with source locations:

| Area | Source location | Assessment |
|---|---|---|
| Kernel population and closure | `tools/formal_compiled_inventory.py:41`, `:76`, `:90` | Generated Lean reads the package modules' compiled constant tables, resolves each kernel constant, records kind/provenance, and closes every root through type/value dependencies. Inductives additionally traverse constructors; missing constants/references fail. The implementation no longer relies on the source theorem regex as compiled population evidence. |
| Complete admission checks | `tools/formal_compiled_inventory.py:175`, `:198`, `:235` | Strict parsing distinguishes malformed output from a well-formed rejected inventory. Admission checks required modules, empty root, module-system exclusion, per-module counts, duplicate names, all target theorems, exact non-target records, auxiliary provenance/parent constraints, forbidden axioms and the target-audit subset cross-check. The existing clean source contract is 31 target theorems plus 14 non-target kernel records. |
| IR ownership is separate | `tools/formal_compiled_inventory.py:99`, `:280`, `:357`; `formal/manifest.json:464` | Extra compiler ownership names are emitted, counted and matched against the separate 11-record binding. They are not counted as kernel constants or claimed to have kernel-checkable bodies. Unknown, duplicate and mismatched ownership records fail. |
| Live gate integration | `tools/formal_gate_check.py:163`, `:243`, `:333`, `:374` | Helper, inventory tests/document, workflow and shell regression are manifest-bound. Compiled validation runs after fresh build/leanchecker and the retained target audit, before types, all exact negative controls and final source/runtime checks. Only then is the receipt written with both compiled results and the actual workflow attempt. Exceptions fail the gate; no inventory failure branch mints a success receipt. |
| Final-receipt lifecycle | `.github/workflows/formal-verification.yml:64`; `tests/test_formal_workflow_shell.py:42` | Digest-checked pinned installation precedes both-mode gate/diagnostic/compiled controls. Those controls precede the dedicated final kernel execution, which precedes live receipt binding. This prevents the final binding from simply selecting the last successful test execution. Existing bash failure propagation, actions and permissions are retained. |
| Mutation/specimen evidence | `tests/test_formal_compiled_inventory.py:326`, `:355` | Seven Python policy mutants run against the actual audit tests. Real specimens cover clean inventory, unused admitted definition, private admitted theorem, opaque admission, custom axiom and native admission. Specimen tests use a temporary copied package; they do not alter checked proof sources. Their local absence is explicitly skipped rather than credited as execution. |
| Original receipt/history obligations | `tools/required_formal_check.py:50`; `tools/atlas_formal_evidence_build.py:47`, `:69`; `tools/formal_gate_check.py:307`, `:379` | Strict live binding and both historical validators are unchanged from accepted `da739673`. Entire previous attempt custody still occurs before runtime preflight. Against exact `50575e98`, the producer's only difference is serialization of the real attempt field; no compiled, diagnostic or lifecycle check was disabled to compose it. |

The manifest resolution is mechanical and compatible: every non-file field and every bound path equals `50575e98`; only the actual merged producer and gate-test hashes differ. The compiled helper/test, workflow/shell test and compiled/formal documentation are byte-identical to that frozen input. Exact file equality is custody evidence; I separately inspected their behavior rather than accepting equality as a substitute for review.

## Evidence checked without rerunning suites

The retained successor logs support **192 run, 190 passed, two explicit Lean-dependent skips in each normal and optimized mode**. The full normal log supports **1613 run, 1588 passed, 25 explicit skips, zero failures/errors**, with recorded exit 0. Its unittest terminal result is at `composition-successor/composition-full-discovery.log:1638` and `:1640`. These are the new combined-source results, not the predecessor's 146/1567 counts or the source author's standalone execution.

I independently performed read-only checks of:

- All candidate changed-file hashes and six exact-successor file identities.
- All 48 proof/historical/fixture-data/canonical preservation files against both native inputs.
- Every current manifest-bound file hash, complete file-key set, and equality of all non-file manifest fields with `50575e98`.
- All 33 contribution checksums.
- Literal original test-method bodies against both parents across eight relevant test files. Every original method remains. The sole predecessor-body change is the positive runtime event/log test adding `Inventory.lean` and `compiled-inventory.log`; removing those two additions restores its old body exactly. All successor test bodies are unchanged.
- Every recorded focused/full test count and OK/skip summary against its raw log, and `git diff --check da739673 HEAD`.

These checks passed. No broad or focused suite, kernel command, remote action or installer was rerun. There was no unresolved concrete source doubt requiring another behavioral experiment.

## Explicit boundaries and behaviors set aside

1. **Fresh compiled and hosted qualification remains open.** The combined-source real-package and real-inventory tests explicitly skipped locally. The 45/11 inventory and 365 closure described in the source documentation are the frozen reviewed contract/prior observations, not newly measured compiled output from this session. Hosted execution must validate the pinned toolchain, actual generated Lean program, real negative specimens and dedicated final receipt on this combined source. A real failure must not be waived on the strength of Python results or standalone505 green.
2. **This is a dependency-free package check, not arbitrary future-package compatibility.** Admission intentionally fails closed for unregistered declarations/modules, unsupported module-system behavior or names that cannot be represented in the current manifest binding. The checker observes private/quoted/numeric names losslessly; the existing binding format need not auto-admit them. Changing the package or its reviewed inventory requires a new source review.
3. **Target audit versus full closure has the correct direction.** Every target-reported axiom must occur in the full closure; extra allowed axioms used only by non-target definitions need not appear in the target audit. Equality of those two sets is not required for complete package closure. Forbidden closure axioms remain rejected.
4. **The receipt contains two distinct transcript digests.** `compiled_inventory.sha256` at `tools/formal_compiled_inventory.py:354` hashes the parsed lines joined by LF, without a terminal newline. `logs['compiled-inventory.log']` binds the actual stored log bytes at `tools/formal_gate_check.py:382`. The former is not interpreted as the raw-log digest; the latter retains exact receipt/log custody. No existing consumer equates those fields in this composition.
5. **Predecessor limitations that remain unchanged:** source-only failure before entering execution does not archive a prior receipt; the workflow still gates binding on actual success. Local history preservation is not a claim that the unchanged upload step publishes all historical attempt directories. Immutable historical bridge metadata remains historical bytes and does not assert that today's live source validator ran on an old manifest.
6. **Broader scientific and environment closure:** semantic alignment, other formal packages, conventional research proof closure, remaining environmental/data skips, full optimized discovery, all Layer A and CBZ-0 remain unestablished. The prior exclusion of compiled inventory is superseded only for this explicitly authorized package-source delta; it is not replaced by a claim of observed combined kernel success.

No boundary above is used to overlook a concrete broken behavior in the reviewed current source. Only this report was written outside the checkout; no source, index, HEAD, remote state or existing evidence was changed, and no subagent was used.
