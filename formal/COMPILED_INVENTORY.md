# Main compiled-environment inventory

Scope: the dependency-free `UniversalLaw` formal package only. This is not a
federation CBZ certificate, an analytic coefficient proof, or an alignment review.
The 31 target statements and approved axiom set are unchanged.

## Tool registry and verification

`tools/formal_compiled_inventory.py` generates a pinned-Lean program and parses
its complete output. It is adapted from Math- `formal/gate_helper.py` at
`8151f592918a845bf19eb8eafaf8444540a32734`; no cross-repository runtime import
or new dependency exists. Owner: Main formal gate maintainers; self-contained.

Run from repository root with the pinned toolchain available:

```sh
python -B -S -m unittest tests.test_formal_compiled_inventory tests.test_formal_gate tests.test_formal_negative_diagnostics -v
python -O -B -S -m unittest tests.test_formal_compiled_inventory tests.test_formal_gate tests.test_formal_negative_diagnostics -v
python -B -S tools/formal_gate_check.py --run-lean
```

Real-Lean tests explicitly skip if `lake` is absent; such a run is NOT evidence
of real execution. The gate itself fails without its pinned runtime. The test
suite includes protocol/name/module/count/binding/axiom controls, seven mutated
Python auditors (all must be killed), real clean build, unused admitted def,
private hidden admitted theorem, opaque admission, custom axiom, and native
proof specimens. These are scoped tests, not a formal proof of Python/Lean
implementation correctness.

Each auditor mutant must run the complete unfiltered control population and
produce exactly its declared assertion-failure identities, with zero test errors
or skips. Startup errors, incomplete runs and unrelated failures are rejected.
The unmutated controls must pass first; four additional contract tests exercise
these rules. An outer `unittest -k` selection does not filter the internal mutant
controls. The pinned threshold is seven of seven declared policy mutants.

## Admission contract

After a fresh build and `leanchecker`, execution enumerates module tables in the
imported environment, including private, unused, and compiler-generated kernel
constants. Every user theorem must exactly match the existing target registry;
every other kernel constant must match `compiled_inventory` in the manifest.
There is no routine auto-admission: source changes require review of the exact
binding diff. The current clean package has 45 kernel constants: 31 user
theorems and 14 definitions (including four generated `fact` definitions).
All 45 roots are traversed transitively, including types and values, using the
pinned Lean `CollectAxioms` rules. Missing references fail. The clean closure
contains 365 visited constants and no axioms. The target axiom audit is retained
and cross-checked against the aggregate closure.

Lean 4.34.1 `Lean/Environment.lean` defines `extraConstNames` as code-generator
ownership-map entries **not** present in `ConstantInfo`. Main has 11 such IR
names in Ledger. They are enumerated as `EXTRA` records, counted and matched
exactly to `codegen_inventory`; they are not falsely claimed to possess
kernel-checkable theorem bodies. Logical evidence never relies on running
those IR bodies (`native_decide` axioms are rejected). All actual kernel
constants, including generated constants, remain in the all-constant closure.

Each name is losslessly encoded by UTF-8 string or numeric components. The
parser rejects unknown output, malformed/missing/duplicate records, module or
constant drift, unknown names, forbidden axioms, and incomplete counts.
Source regex checks are preflight conveniences, not compiled coverage evidence.

The receipt records compiled counts, closure size, axiom set, and transcript
SHA-256; `compiled-inventory.log` is also in the receipt log digest map. The
manifest binds helper, tests, gate, and this document. Prior receipt history is
retained by the existing gate. No bound or failing test was removed or weakened.

## Exact source-commit custody

The receipt producer compares every manifest-bound input and declared unbound
package file (including the manifest) with raw blobs and executable modes in
one immutable Git HEAD before Lean runs and after the final source/version
checks. The final check must use the same captured HEAD. Actual loaded producer
and compiled-helper bytes are matched to the declared source copies. The
manifest also binds the live receipt consumer, its tests and aggregate workflow.
Git replacements, alternate repository/index environment variables and content
filters cannot substitute another source tree. A mismatched package argument is
rejected before filesystem writes; a valid-package source refusal archives prior
receipt/logs before making any current success impossible. Unrelated repository
edits and generated `.lake` outputs are outside the owned-input scope.

Run `python3 -B -S -m unittest tests.test_formal_source_custody -v`, then repeat
with `-O`. Real-Git controls exercise dirty/refreshed/staged/hidden source bytes,
manifest metadata, missing inputs, executable modes, symlinks, repository roots,
HEAD drift, replacement objects, loaded code and failed-rerun receipt handling.
These checks establish pre/post endpoint custody in the trusted runner; they do
not claim continuous atomic isolation from a malicious concurrent writer.

The custody test module includes a declared nine-source-mutant inventory
(threshold9/9), with complete unfiltered controls and exact named assertion
failures under the shared strict mutation contract. It plants missing initial
and final checks as well as blob/mode/HEAD/manifest/loaded-code/inventory/Git
replacement defects. Startup failures, test errors, skips and unrelated failures
are not kills. The module also checks the mandatory live-consumer/test bindings.
