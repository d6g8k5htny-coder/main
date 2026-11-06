# CBZ mission-tool registry

**State: component registry; the final federation toolchain is incomplete.**

This file lists runnable mission-relevant tools present at Main revision
`04989724b5142f8eea8d41946d0367d0372d845a`. Each entry states purpose,
command, verification and scope. A component passing in isolation does not
complete a CBZ gate or the final G0–G11 matrix.

## Formal gate checker

- **Purpose:** run the Main formal checks, including the pinned Lean toolchain,
  exact source checks and retained negative controls documented by the formal
  contract.
- **Command:** `python -B tools/formal_gate_check.py --run-lean`
- **Verification:** follow
  [docs/FORMAL_VERIFICATION.md](docs/FORMAL_VERIFICATION.md) and
  [docs/FORMAL_REQUIRED_CHECKS.md](docs/FORMAL_REQUIRED_CHECKS.md); retain the
  actual build, kernel, axiom and negative-control output.
- **Scope:** Main's documented formal component only. It does not run the full
  federation, prove scientific alignment or substitute for G0–G11.

## G11 bounded census

- **Purpose:** report Python stubs, code-comment markers, no-op workflow steps
  and missing static documentation/test references in one tracked Git tree.
- **Commands:**

  ```sh
  python3 -B -S -m unittest tests.test_g11_census
  python3 -B -S -O -m unittest tests.test_g11_census
  python3 -B -S tools/cbz/g11_mutation_selftest.py
  python3 -B -S -O tools/cbz/g11_mutation_selftest.py
  python3 -B -S tools/cbz/g11_census.py --rev <commit> --json census.json
  ```

- **Verification:** the unit suite and mutation self-test must pass in both
  modes; a census exit of 0 means only that this implementation found no
  production finding in its declared paths and rules.
- **Scope:** static bounded triage. It does not establish runtime behavior,
  semantic placeholder absence, Layer A clearance or G11 admission. Full
  limits are in [docs/cbz/G11_AUDITOR.md](docs/cbz/G11_AUDITOR.md).

## Federation drift crawler

- **Purpose:** discover the federation from the Workspace seed and check pin,
  registry, topology and literal Lean-interface consistency.
- **Commands:**

  ```sh
  cd tests
  python3 -B -m unittest test_cbz_drift
  python3 -B -O -m unittest test_cbz_drift
  cd ..
  python3 -B -S tools/cbz_drift/crawl.py \
    --seed d6g8k5htny-coder/Universal-Law-Workspace \
    --search <checkout-directory> --fetch --at pins --json drift.json
  ```

- **Verification:** both synthetic-federation suites, including the registered
  source mutants, must pass. A current crawl must retain BLOCK and INFO output.
- **Scope:** source-level pin and interface drift under the documented parser.
  It does not elaborate Lean, expand macros, prove declarations or complete
  G2–G4. See [tools/cbz_drift/README.md](tools/cbz_drift/README.md).

## Empirical statement check (G12 component)

- **Purpose:** exact-arithmetic sampled checks for registered Lean statement
  classes and finite discrete models.
- **Commands:**

  ```sh
  python3 -B -S -m unittest tests.test_cbz_statement_fuzz -v
  python3 -B -O -S -m unittest tests.test_cbz_statement_fuzz -v
  python3 -B -S tools/cbz/statement_fuzz.py --package formal --selftest --json report.json
  ```

- **Verification:** the full registered self-test and mutation contract must
  pass in both modes; unsupported constructs remain named and fail closed.
- **Scope:** bounded empirical corroboration only. PASS and CORROBORATED are
  not kernel proof or scientific acceptance. G12 is registered context, not a
  replacement for the required final G0–G11 pass.

## Restored bounded replay controls

- **Purpose:** rerun the retained Python validator and three registered mutants
  in both interpreter modes for the preserved additional kernel evidence.
- **Command:** from `cbz/evidence/kernel/additional/replay-tool/`, run
  `python3.11 -B -S run_mutations.py --output <new-directory>`.
- **Verification:** preserve the generated result and logs; compare their
  source binding with the README and retained historical outcome.
- **Scope:** this validates the documented Python replay tool. It neither
  authenticates a historical execution nor performs a Lean kernel replay.

## Missing final orchestrator

No documented command in the current default runs G0 through G11 over one
common federation snapshot and emits one source-bound receipt. Therefore tool
criterion 9, cold-start criterion 11 and the final-pass requirement remain
unawarded. A future orchestrator must fail closed, preserve every failure,
record exact sources and tool versions, and carry its own adverse tests and
mutations before admission.
