# CBZ tooling

See [GOVERNANCE.md](GOVERNANCE.md) for purposes, verification and limitations.
Use Python 3.11 or newer; the new inventory tools have no third-party packages.

From the root:

    python -B -S -m unittest discover -s tests -p 'test_cbz_*.py' -v
    python -B -O -S -m unittest discover -s tests -p 'test_cbz_*.py' -v
    python -B -S tests/run_scanner_mutations.py --evidence-dir /tmp/NEW-CBZ-CUT
    python -B -S tools/cbz_source_audit.py --manifest cbz-source-manifest.json

The evidence directory must be new; prior cuts are never overwritten.
The scanner returns 1 for review findings or 2 for not-established/refusal.
Neither is a G1 or CBZ success. Its manifest covers the four current main-side
Lean files; using it for another tree requires an explicit complete manifest.
Build caches are not source exports; audit an immutable clean source view.

Existing main formal execution, with exact Lean 4.34.1 available:

    python -B -S tools/formal_gate_check.py
    python -B -S tools/formal_gate_check.py --run-lean

The existing hosted formal-verification lane performs those actual operations.
The new CBZ source-auditor workflow tests the inventory instrument itself;
it deliberately makes no full Layer A certification claim.
