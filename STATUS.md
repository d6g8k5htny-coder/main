# Federation baseline state

This workspace develops mathematical candidates/formal companions in `Math-`, research/navigation and operational tools in `main`, federation pins in `Universal-Law-Workspace`, and associated infrastructure. The owner-directed mission is Clean Baseline Zero (CBZ-0). [GOVERNANCE.md](GOVERNANCE.md) replaces earlier procedural freezes, review-family quorums and approval/custody ceremonies. Historical mathematical/test outcomes remain facts.

## Current baseline

**Partial Baseline — consolidation is running; CBZ-0 is NOT established.** Observation cut: 2026-10-29 17:35 UTC. No final G0–G11 run has completed; none of the twelve criteria is qualified as part of a combined final baseline. This is a progress checkpoint, not a terminal/resource-impossibility declaration.

| Gate | Actual component observation | Final result |
|---|---|---|
| G0 | Math: 244 formal and 76 receipt-consumer tests passed per Python mode; compiled controls distinct | Not qualified |
| G1 | Complete source census and bounded lexical/AST probes exist; no semantic certification inferred | Not qualified |
| G2 | Literal axiom reports for 116 Math core constants passed; three approved axioms only | Not qualified |
| G3 | Fresh main arithmetic and Math core builds passed | Not qualified |
| G4 | Fresh main recheck passed; unchanged Math recheck passed with LEAN_NUM_THREADS=2 | Not qualified |
| G5 | Main: 1,541 tests per mode with 24 skips; 19 skipped methods passed separately. Prepared Node: 763/763, zero skips | Not qualified |
| G6 | Downstream: 24/24 and replica: 24/24 semantic mutants detected per mode | Not qualified |
| G7 | Workspace local/network historical-pin checks passed their existing contract | Current federation consistency not qualified |
| G8 | Canonical surfaces being reconciled to observed evidence | Not qualified |
| G9 | Historical fixture fetch preconditions identified | Final clone/tag test not run |
| G10 |158 fully contained branches pruned and absences read back; distinct deliveries remain | Not passed |
| G11 | Census tool: 38 tests per mode and both source mutations passed; independent review repaired/re-reviewed | Final federation admission not qualified |

## What is closed and verified

94 literal formal declarations are CLOSED at their exact kernel-checked source cuts; no stronger model or parent claim is closed. [PROOF_INDEX.md](PROOF_INDEX.md) records the admission boundary. The main arithmetic execution is bound to `c3377cc8c0fb35844dc7613976d6e746fdc05600`; Math core source is `8151f592918a845bf19eb8eafaf8444540a32734`. They concern literal formal statements only, not a constructed Gaussian/Palm model or a concrete coefficient/enclosure identification.

Completed engineering changes:

- main#685 governance/ledger merged at `81acc060341e8a31cc4cd4f9523ef71fb6c06175` after 8/8 exact-head checks passed.
- Five missing exact historical replica fixtures restored in meta-framework, integrated at `0f5894f81a5462c10ee5ec8f8109f466e712e47e`. Unchanged67 tests,24 mutations and77 central contracts pass per mode; no checker/test/bound relaxed.
- Math recheck recovered from observed exit137/OOM using `LEAN_NUM_THREADS=2`; same source/checks passed. Original failure retained; the complete gate subsequently passed every admission control.
- Verified full-history bundles before guarded cleanup. Removed main45,Math40,governance30,query6,meta37 contained branches; all 158 absences independently read back. Open-PR heads/bases, protected and workflow-used refs excluded.

## Quarantine boundary

The strict CLOSED domain contains only the 94 literal declarations in PROOF_INDEX.md. Conventional prose, finite numerical controls, loose Lean packages and historical ACCEPT labels remain outside it. The final per-source terminal index is still being assembled; its absence is not represented as a completed quarantine.

Known residue: non-core determinant/Cap auditors omit hidden declarations; main receipt producer/consumer lacks complete attempt binding; archived trial tests contain live write probes and need isolation. Original sources/failures are retained. Disjoint repairs/dispositions are coordinated in [main#684](https://github.com/d6g8k5htny-coder/main/issues/684), with main#619 the active board. No human or old family-review approval is required.

## Tools and how to verify

Use complete history, Python3.11 (observed3.11.17), repository-pinned Lean4.34.1 and exact dependency locks. Python3.12 does not satisfy scripts explicitly requiring3.11. Preserve TLS/proxy trust and credentials. Run in the stated repository root; record `git rev-parse HEAD` and `git status --porcelain` before/after. The new source-census tool scans twelve sibling clones under the supplied root; output must be outside those clone directories.

Math core:

```bash
python3.11 -B -S -m unittest discover -s formal/tests -v
python3.11 -B -O -S -m unittest discover -s formal/tests -v
python3.11 -B -S formal/gate.py
cd formal
lake exe cache get
cd ..
LEAN_NUM_THREADS=2 python3.11 -B -S formal/gate.py --execute
```

The execute command builds fresh local objects, rechecks the kernel, audits axioms/elaborated types/all116 compiled constants and adversarial controls. Logs/receipt: `Math-/formal/.lake/formal-evidence/`. Approved axioms exactly `propext`, `Classical.choice`, `Quot.sound`. Source-only PASS and supplemental literal116 printing cannot replace execution/kernel success.

Main pilot and regression:

```bash
python3.11 -B -S tools/formal_gate_check.py --run-lean
python3.11 -B -S -m unittest discover -s tests -v
python3.11 -B -O -S -m unittest discover -s tests -v
python3.11 -B -S tools/public_shop_check.py
python3.11 -B -S tools/museum_check.py
```

The Node launchers validate pinned fixture bytes. Raw Node discovery without fixture variables is incomplete. Historic main commits `71400b94f6cb354a8cf7aba73ffede2138a64efa`, `a12c178c0f857a130cf434e9efd44233a038195b` and `f2e432ea5c86742e480c66624775bc9103343314` must be available; shallow clones need exact fetches. Optional browser/pytest and all five actual restricted-container tests have passed separately in both modes. Five real-record cases remain unexercised because their data/interface prerequisites are absent; retain those gaps, never fabricate records to pass them. The archived trial suite reproduced 87 failures in each isolated mode; it is not green. The restricted-container run used a verified source archive and a Docker-managed 64 MiB output tmpfs, with all original security/resource assertions enforced; this is local runtime evidence, not a GitHub-hosted run.

Source census (a byte/binding checker, **not a mathematical certifier**):

```bash
python3.11 -B -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3.11 -B -S tools/cbz_inventory.py generate --root /workspace/cbz --output /workspace/cbz-inventory.json
python3.11 -B -S tools/cbz_inventory.py verify --root /workspace/cbz --manifest /workspace/cbz-inventory.json
```

It binds tracked/index file union, commits, actual bytes and classification fields. It cannot authenticate kernel execution, decide semantic completeness or grant CBZ green. [TOOLING.md](TOOLING.md) lists scope; [INVENTORY.md](INVENTORY.md) lists discovery/gaps.

## Decisions and invariants

[DECISIONS.md](DECISIONS.md) is append-only; [GOVERNANCE.md](GOVERNANCE.md) contains Layer B rules/tool registry. [BASELINE.md](BASELINE.md) binds snapshots/cuts. Raw private evidence stays local; public evidence is under `cbz/evidence/`.

Never present placeholders/stubs as closed, inject unverified axioms, certify mathematical bounds with floats, weaken bounds/tests, conceal vacuity, invent passing verification or trust untested auditors. Preserve failures and exact boundaries. Quarantined logical dependencies must not enter closed theorems.
