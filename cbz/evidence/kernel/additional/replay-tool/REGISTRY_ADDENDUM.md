Tool: bounded compiled proof-closure replay generator and transcript reader.

Files: replay_tool.py, test_replay_tool.py, run_mutations.py,
run_lean_selftest.py, README.md. Maintenance owner: none — self-contained.

Purpose: close the gap between package-specific target checks and rechecking
all compiled owned constants plus complete type/value dependencies in an empty
Lean kernel environment. The existing pinned collector is copied verbatim;
the implementation uses Lean4.34.1's own Kernel.Environment.replay.

Usage: README.md contains portable exact generation, transcript consistency,
unit-test, Python source-mutation and actual Lean self-test commands. It requires
the caller to bind actual execution, source/object/dependency/runtime identity;
the parser does not authenticate execution and makes no semantic/model claim.

Self-verification: 31 tests in normal and optimized Python3.11.16; all three
declared Python validator-source mutants killed in both modes. Actual pinned
Lean fixture replay separately checks one positive, six adverse controls and
six generated-Lean auditor source mutants. No violating fixture is retained
as a source module. Raw evidence directories and SHA256SUMS are supplied.
The actual Lean harness also passed in both normal and optimized Python3.11.16
modes on identical final source/runtime/collector identities.

CBZ criteria served: 3 (literal proof qualification), 6 (bounded kernel/axiom
evidence), 9 (documented verified tool), 11 (reproducible cold-start commands).
It is a component gate. Its success does not declare G0–G11 or CBZ‑0 globally.
