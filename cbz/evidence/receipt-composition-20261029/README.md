# Receipt and strict-kernel gate composition evidence

Read REVIEW.md for the exact source cut, changes, observed results and limits;
REGISTRY_ADDENDUM.md is input to the canonical single writer. SOURCE_BINDING.json
binds the tested controls. Raw build/kernel logs and receipt are in
`../kernel/main-composition-20261029/`. Earlier RED and unchanged-constituent
evidence remain in the PR692 contribution directory and history.

Reproduce the focused commands from the repository root with official pinned
Lean4.34.1 available and CPython3.11.16 first in PATH. Use `LEAN_NUM_THREADS=2`.
Run these eight modules with `python3 -B -S -m unittest discover -s tests -p NAME -v`,
then repeat with `-O`:

- test_required_formal_check.py
- test_formal_gate.py
- test_atlas_formal_evidence_build.py
- test_atlas_formal_source_bridge.py
- test_formal_workflow_shell.py
- test_required_formal_binding_control.py
- test_formal_negative_diagnostics.py
- test_formal_compiled_inventory.py

The last module includes all seven strict reason-bound source mutants and four
kill-classification controls; no external mutation driver is needed for that
gating check. The formal gate test executes the real build/kernel/axiom/rejection
path when the runtime is available. A missing-runtime skip is not a pass.

`fresh-focused-results.json` preserves actual argv and local execution paths;
those historical paths are not prerequisites. Current source/manifest checks
and final hosted/federation gates must be rerun at integration. No scientific
alignment, analytic model theorem or federation CBZ-0 is established here.
