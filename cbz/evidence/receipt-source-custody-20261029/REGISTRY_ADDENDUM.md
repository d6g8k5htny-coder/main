# Canonical registry admission input

Updated existing producer `tools/formal_gate_check.py`: exact owned-source/HEAD
custody before and after Lean execution, raw Git blobs plus executable modes,
actual loaded producer/helper identity, manifest argument equality and explicit
package-argument ownership. No new service or runtime. `formal/manifest.json`
now also binds live consumer/tests, custody tests and parent aggregate workflow.
The existing formal workflow runs custody controls before the final kernel lane.
Maintenance owner: none, self-contained under the existing formal gate.

Usage from a Git working clone:

- Source-only metadata check: `python3 -B -S tools/formal_gate_check.py`.
- Real gate: `LEAN_NUM_THREADS=2 python3 -B -S tools/formal_gate_check.py --run-lean`, official pinned Lean4.34.1.
- Self-verification: `python3 -B -S -m unittest tests.test_formal_source_custody -v`, repeat with `-O`.
- Source-custody mutant inventory:9, threshold9/9; complete22 internal controls, exact declared assertion failures, no errors/skips, passing baseline. Existing compiled inventory:7, threshold7/7 under the same strict classifier, plus its four classification controls.

Observed219/219 affected tests per mode atce4277c8; fresh real producer at
source-identicalaa0d59ad after incorporating current Main22567. Raw source-bound
receipts/logs, actual old false-source RED and exact new refusal are retained.
Kernel/axiom evidence remains Main's literal31 skeleton targets and all45 compiled
constants, with all hypotheses intact. Serves criteria4,6,7,9,10; does not establish
statement alignment, semantic full-tree compliance or final federation CBZ-0.

Trust scope: trusted pinned runtime and pre/post source endpoint identity. It does
not claim OS-level continuous isolation against a malicious concurrent writer.
Source-only/immutable historical readers still support their documented data
projections; commit-qualified real execution requires a Git working clone.
