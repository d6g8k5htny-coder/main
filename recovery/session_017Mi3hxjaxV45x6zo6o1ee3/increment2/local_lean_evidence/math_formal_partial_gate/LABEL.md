# LABEL: LOCAL, NON-AUTHORITATIVE, PARTIAL RUN, NOT A GATE PASS

These files are the untracked outputs of a local, deliberately partial run of the Math- `formal/` gate in the
Claude Code cloud container of session `session_017Mi3hxjaxV45x6zo6o1ee3` on 12 October 2026, 22:45–22:47 UTC,
at Math- commit `40440ed8bcb80913426042c80f6facedc6a4965c`, with the pinned toolchain (elan 4.2.4,
`leanprover/lean4:v4.34.1`, Lake 5.0.0, Mathlib `d13f23b723b8a846827a245b89c10fc7d3f11612`).

The full gate (`formal/gate.py --execute`) had been attempted first and its `leanchecker` step was killed by the
container's memory cgroup (exit 137, 13.49 GB anonymous RSS against a 14,343,688,192-byte limit; see
`toolchain_readiness/math_gate_execute.log` in checkpoint 1). This tree comes from a second run in which only the
leanchecker step was replaced by a logged SKIP (`leanchecker.log` says so); build, axiom audit, type checks and
negative controls ran. **This is not a gate pass**, `receipt.json` is **invalid as a receipt**, and the local
copy of the modified gate (`toolchain_readiness/gate_partial_LOCAL_DIAGNOSTIC_ONLY.py`, checkpoint 1) is not the
project gate and must not be installed. The hosted `formal-verification` runs are the only receipts.
Scientific effect NONE; organizational-independence credit 0. Preserved at the owner's request (main#307
comments 6136210973 and 6136304987); `<SCRATCH>` replaces the container's private scratch path.
