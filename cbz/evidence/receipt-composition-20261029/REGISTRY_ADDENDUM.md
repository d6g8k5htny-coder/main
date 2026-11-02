# Serial integration registry input

Scope: Main arithmetic skeleton only, literal 31 target statements and all 45 compiled constants. No theorem, source claim, mathematical bound, toolchain pin or approved axiom set changed. The exact historical Atlas evidence remains immutable and scientifically unpromoted.

Updated existing tools:

- `tools/formal_gate_check.py`: producer attempt serialization plus #689 full rejection-transcript authentication, complete mutation inventory, prior-attempt preservation and all-constant audit integration. Source-only: `python3 -B -S tools/formal_gate_check.py`. Real gate: `python3 -B -S tools/formal_gate_check.py --run-lean` with official Lean4.34.1 and `LEAN_NUM_THREADS=2`.
- `tools/required_formal_check.py`: live producer/consumer workflow-attempt equality; existing CLI/workflow route. Local/null identity cannot satisfy hosted binding. Self-verification: `python3 -B -S -m unittest discover -s tests -p test_required_formal_check.py -v`, repeat `-O`; removal of only attempt comparison must cause the three named methods' 11 intended assertion failures.
- `tools/atlas_formal_evidence_build.py` and `tools/atlas_formal_source_bridge.py`: fixed historical source profiles are read only as hash-bound data; one immutable pre-attempt receipt is reconstructed through a separate helper, not live receipt binding. Runnable current-checkout fixture reproduction: affected `test_atlas_formal_evidence_build.py` and `test_atlas_formal_source_bridge.py` modules in normal/optimized modes. Historical CLI requires the exact external source projection and snapshot already documented in #692 registration addendum; current-root refusal is intentional.

New bounded tool from #689:

- `tools/formal_compiled_inventory.py`: generate the pinned-Lean complete module constant inventory and audit the lossless protocol, exact ownership/manifest counts, all-constant transitive type/value axiom closure and separately bound IR ownership entries. It does not certify mathematical statement alignment or execute IR to prove bounds. Invoked by the real formal gate; exact usage, trust boundary and tests are in `formal/COMPILED_INVENTORY.md`. Maintenance owner: none, self-contained under the existing formal gate.
- Self-verification: `python3 -B -S -m unittest tests.test_formal_compiled_inventory tests.test_formal_gate tests.test_formal_negative_diagnostics -v`, repeat with `-O`. Real Lean controls require installed pinned toolchain; absent runtime skips are not passes.
- Declared policy mutation inventory: seven distinct source mutants, threshold 7/7 (100%). Each must run all 26 internal policy controls, with zero errors/skips and exactly its declared assertion-failure identities. Baseline controls must pass first. Four added harness-contract tests reject test errors, skips, incomplete populations, survivors, empty contracts and unrelated failures. A separate original-condition harness mutant is killed by six intended assertions in these four tests in each Python mode. Outer CLI `-k` filtering does not shrink the internal population.

Serves CBZ criteria4,6,7,9,10. Each actual execution qualifies only its exact source cut; current hosted checks and final federation G0-G11 remain root integration obligations. No parent theorem, analytic enclosure, statement alignment or CBZ-0 is certified by this addendum.
