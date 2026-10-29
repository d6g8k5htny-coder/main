# Federation baseline state

This workspace develops mathematical candidates/formal companions in `Math-`, research/navigation and operational tools in `main`, federation pins in `Universal-Law-Workspace`, and associated infrastructure. The owner-directed mission is Clean Baseline Zero (CBZ-0). [GOVERNANCE.md](GOVERNANCE.md) replaces earlier procedural freezes, review-family quorums and approval/custody ceremonies. Historical mathematical/test outcomes remain facts.

## Current baseline

**Partial Baseline — consolidation is running; CBZ-0 is NOT established.** Observation cut: 2026-10-29 21:19 UTC; observations are tied to the exact source cuts stated below. No final G0–G11 run has completed; none of the twelve criteria is qualified as part of a combined final baseline. This is a progress checkpoint, not a terminal/resource-impossibility declaration.

| Gate | Actual component observation | Final result |
|---|---|---|
| G0 | Historical core controls passed; lexical and A3 mutation graders have identified false-kill paths and their repairs are pending | Not qualified |
| G1 | A historical source census and bounded lexical/AST probes exist; full current-source classification and auditor qualification remain incomplete | Not qualified |
| G2 | Literal axiom reports for 116 Math core constants passed; three approved axioms only | Not qualified |
| G3 | Fresh main arithmetic and complete Math core builds passed | Not qualified |
| G4 | Fresh main recheck and complete Math gate passed with LEAN_NUM_THREADS=2 | Not qualified |
| G5 | Main composed-source run: 1,801 tests per mode, 25 skips each; two Lean-dependent and five restricted-container methods passed separately at the recorded equal-tree source cut | Not qualified |
| G6 | Downstream: 24/24 and replica: 24/24 semantic mutants detected per mode | Not qualified |
| G7 | Workspace local/network historical-pin checks passed their existing contract | Current federation consistency not qualified |
| G8 | Canonical surfaces being reconciled to observed evidence | Not qualified |
| G9 | Historical fixture fetch preconditions identified | Final clone/tag test not run |
| G10 |158 fully contained branches pruned and absences read back; distinct deliveries remain | Not passed |
| G11 | Census: 75 tests and nine source mutants per mode. Replay helper: 31 tests and three Python mutants per mode, plus actual Lean guard controls/mutations | Final federation admission not qualified |

## What is closed and verified

367 literal declaration records are CLOSED in their exact kernel-checked package contexts: main arithmetic31, Math core63, specification78, smooth-source116, C29713 and determinant66. [PROOF_INDEX.md](PROOF_INDEX.md) and its exact signature/source/axiom records preserve every binder and hypothesis. Main source is `c3377cc8c0fb35844dc7613976d6e746fdc05600`; Math core/additional source is `8151f592918a845bf19eb8eafaf8444540a32734`; determinant source is `74565f013575b7b05ecfc4034cd5d638ee355353`. Package and source path are part of identity: determinant reuses 13 names from the separate core. These counts are not independent discoveries. No broader Gaussian/Palm construction, coefficient/enclosure identification or parent theorem is closed by these records.

Actual additional empty-kernel replays covered specification901 owned/65,210 dependencies, smooth222/55,236, C297131/38,531 and determinant157/58,421. Per-theorem axiom/type reports and actual adverse controls passed. The replay helper and its generated Lean audit have self-tests and source mutations. See the exact commands in [additional reproduction](cbz/evidence/kernel/additional/HOW_TO_REPLAY.md) and [determinant reproduction](cbz/evidence/kernel/determinant/HOW_TO_REPLAY.md). Defining any of the remaining specification Props does not assert it is universally true. Model satisfiability/premise discharge is not inferred.

Completed engineering changes:

- main#685 governance/ledger merged at `81acc060341e8a31cc4cd4f9523ef71fb6c06175` after 8/8 exact-head checks passed.
- main#693 manifest totals/duplicate-path repair merged at `27c89b05bd2503c580b692f57146d608a7f0f668` after 8/8 exact-head checks. Superseded draft #482 is closed; its source is preserved.
- Five missing exact historical replica fixtures restored in meta-framework, integrated at `0f5894f81a5462c10ee5ec8f8109f466e712e47e`. Unchanged67 tests,24 mutations and77 central contracts pass per mode; no checker/test/bound relaxed.
- Math recheck recovered from observed exit137/OOM using `LEAN_NUM_THREADS=2`; same source/checks passed. Original failure retained; the complete gate subsequently passed every admission control.
- Verified full-history bundles before guarded cleanup. Removed main45,Math40,governance30,query6,meta37 contained branches; all 158 absences independently read back. Open-PR heads/bases, protected and workflow-used refs excluded.

## Quarantine boundary

The strict CLOSED domain contains only the 367 context-bound literal records. [The adopted catalog](cbz/quarantine/catalog.json) and [readable boundaries](cbz/quarantine/BOUNDARIES.md) give all §11 fields for 307 listed QUARANTINED source/application records: source/model applications, conventional arguments, the archived trial suite and five real-record coverage gaps. They have no ongoing owner or maintenance. Remaining numerical-package mapping is being reconciled, so complete federation obligation coverage is not yet claimed.

The boundaries preserve actual smooth/Fourier constructions but withhold unconstructed critical-point/bar measures, Morse/Palm/Kac–Rice/once-counting/elder-pairing bridges, global small-bar converses, concrete coefficient identification and unproved source-to-model alignment. Conditional geometry keeps its regular/interior/Morse premises. Conventional quartic/cusp/fold arguments remain source-bound prose. Historical float-based WP/THM023 verifiers are approximate diagnostics, with no mathematical bound certified by them; Q-3010/Q-3011 preserve the exact decertification and finite-grid limits. Main PR336/340/429 source is retained at annotated tags `cbz-quarantine-q-8336`, `cbz-quarantine-q-8340`, `cbz-quarantine-q-8429`. Closed declarations retain their explicit hypotheses; quarantined assertions are not promoted as proved axioms. A complete cross-federation dependency and isolation audit has not passed.

Known residue: determinant compiled inventory/private support is not yet qualified at its current source; Cap auditor repairs remain in progress; main receipt attempt/source-custody repair is not yet integrated. The 99-module source514 build passed, but its independent compiled census failed on a generated metadata name that also has a kernel declaration; the required ownership and replay repair is pending. Archived trial tests reproduced 87 failures per mode under write/network isolation. Original sources/failures are retained. Disjoint repairs/dispositions are coordinated in [main#684](https://github.com/d6g8k5htny-coder/main/issues/684), with main#619 the active board. No human or old family-review approval is required.

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

Replay-helper self-verification, from `cbz/evidence/kernel/additional/replay-tool`:

```bash
python3.11 -B -S -m unittest -v test_replay_tool.py
python3.11 -B -O -S -m unittest -v test_replay_tool.py
python3.11 -B -S run_mutations.py --output /new/replay-tool-verification
```

Its [README](cbz/evidence/kernel/additional/replay-tool/README.md) supplies the exact installed-Lean/collector arguments for actual Lean controls and six generated-auditor mutations in both Python modes. Supplied transcript text does not authenticate an execution. Historical PR/citation tools are self-verified with the exact ordinary/optimized commands in their [README](cbz/evidence/stale-pr-dispositions/README.md): 44 focused tests and 49 intended source/harness mutant kills per mode. Their recommendations impose no staging/approval rules and describe a recorded historical cut, not current PR state.

## Decisions and invariants

[DECISIONS.md](DECISIONS.md) is append-only; [GOVERNANCE.md](GOVERNANCE.md) contains Layer B rules/tool registry. [BASELINE.md](BASELINE.md) binds snapshots/cuts. Raw private evidence stays local; public evidence is under `cbz/evidence/`.

Never present placeholders/stubs as closed, inject unverified axioms, certify mathematical bounds with floats, weaken bounds/tests, conceal vacuity, invent passing verification or trust untested auditors. Preserve failures and exact boundaries. Quarantined logical dependencies must not enter closed theorems.

## Current integration checkpoint

The integration candidate combines the receipt source-custody and compiled-inventory changes from main #701, the bounded lexical/AST prechecks from #698, and the current main source. Eight additional dormant engineering boundaries Q-9101–Q-9108 and their source histories are recorded in [runtime-archive-dispositions.json](cbz/quarantine/runtime-archive-dispositions.json); their boundaries are included in the central catalog. Existing kernel and test observations are tied to their recorded source cuts; the composed candidate has not completed the final gate matrix. Local execution encountered process-creation failures, so interrupted runs supply no passing evidence. The catalog count includes those eight records and the five paper and three carrier boundaries listed below; complete federation coverage is not yet claimed.

### Additional source-bound records and correction

Five Math paper applications Q-7522–Q-7526 retain the strict source hypotheses and their unproved analytic/model bridges; eight runtime experiments Q-9101–Q-9108 and three historical V20 carrier proposals Q-9109–Q-9111 retain their source and failed or unexecuted controls. Their individual titles and exact boundaries are listed in [PROOF_INDEX.md](PROOF_INDEX.md); the 307-row catalog includes these 16 records, 102 separately identified legacy V20 application records and five related external interfaces. The Q-8429 catalog and readable boundary now identify the seven actual documents and preserve the bound C·16^4·K^8·(1+L)^2·exp(-c/η); [the correction evidence](cbz/quarantine/Q8429-native-source-correction.json) retains the original context. The broader claim remains quarantined.

The hosted main unittest run observed 1,801 tests and exit 0 in each of ordinary and optimized modes, with the same 25 skips in each. A separate hosted job downloaded both original artifacts and verified their ZIP and log hashes; it lists every skipped test. The composed main formal job passed on its recorded tested merge; that formal artifact has not been independently downloaded. The skips and remaining federation suites leave G5 unqualified. See [ordinary](cbz/evidence/main-hosted-full-normal-20261029.json), [optimized](cbz/evidence/main-hosted-full-optimized-20261029.json), [skip observation](cbz/evidence/main-regression-skip-observation/result.json) and [formal](cbz/evidence/main-hosted-formal-20261029.json).

### Mutation-grader qualification limit

Source inspection found that both the lexical and A3 mutation graders accept any nonzero exit and an expected test failure even when foreign failures are present; neither requires the specific bypass reason. Their historical 66 and 23 mutant outcomes do not by themselves establish auditor soundness. Test-first repairs and fresh reason-bound controls are pending, so G0 and federation-wide G1 remain unqualified.

### Legacy V20 source records

The retained V20 dataset contains 104 source records and 104 corresponding obligation entries. All pairs were compared exactly with the source-bound contribution. The catalog records 102 application claims as quarantined; two definition-only records are supporting vocabulary and are not propositions asserted universally. The original dataset did not authenticate its source references. All 33 distinct pinned source blobs have since been read and matched to their tree-reported byte counts; no author-provided expected digest or source-statement alignment is established. Potential overlap with other rows has not been resolved. Row counts therefore do not represent independent theorems or new discoveries. [Provenance and limits](cbz/quarantine/legacy-v20-provenance.json) and the original [dataset](cbz/quarantine/legacy-v20-dataset.json) preserve the exact boundaries.

Five historical V20 external interfaces—conditional C² Gaussian kernels, uniform Gaussian envelopes, exact Kac–Rice applicability, off-diagonal exhaustion and Morse/elder measurability—are quarantined with their original applicability boundaries. They are retained as source/model obligations, not injected as kernel axioms; they may overlap other rows.

### Historical Math graph coverage

The retained Math graph at commit `8151f592918a845bf19eb8eafaf8444540a32734` has 49 nodes and 55 edges. Its complete records and edges match the preserved contribution exactly. In the current 307-row catalog, 35 node IDs occur literally and 14 do not; this is a text observation, not evidence of equivalent semantic coverage. Sixteen source groups and seven alignment gaps also need source-specific reconciliation. The original graph's historical review labels do not certify kernel closure. [Graph source](cbz/quarantine/current-math-graph-source.json) and [coverage observations](cbz/quarantine/math-graph-coverage-observations.json) record the exact cut and limits.

### Main regression skips at the recorded source cut

Both modes skipped the same 25 tests: five require a restricted container, two require a browser, two require Lean, one requires pinned cross-repository composition inputs, ten require pytest, and five require real-record fields or live freshness not present in the data. The five real-record boundaries remain Q-9002–Q-9006. Other historical separately run controls retain their original source and environment cuts; current-cut replacement coverage is not inferred merely from those older results. [Exact tests, reasons and artifact verification](cbz/evidence/main-regression-skip-observation/result.json) are recorded. Full G5 remains unqualified.

The two Lean-dependent methods skipped by broad unittest discovery were separately observed passing in both Python modes in the composed formal job: `test_actual_clean_and_omitted_constants` and `test_real_package_executes_with_receipt`. The tested PR merge has the exact same source tree as the composed candidate; the broad regression's successor adds only its workflow. This supplements the skip record without changing the full-suite result or establishing complete G5 coverage. [Evidence](cbz/evidence/main-regression-skip-observation/lean-controls.json).

The five restricted-container methods skipped by broad discovery were separately observed passing in both modes in the composed architecture job. A later diagnostic downloaded the original artifact, verified its ZIP SHA-256, and printed each of the ten passing test lines. The original broad run still has 25 skips per mode; no full G5 pass follows. [Evidence](cbz/evidence/main-regression-skip-observation/sandbox-controls.json), [observer workflow](cbz/evidence/main-regression-skip-observation/sandbox-observer-workflow.yml.txt).

### Pending conditional pair-law countermodel

A candidate Lean proof on [Math branch `cbz/lifetime-pair-law-countermodel-20261029`](https://github.com/d6g8k5htny-coder/Math-/tree/cbz/lifetime-pair-law-countermodel-20261029), initial head `5a470bfd739aba22e32b91587d5dfc031a7de3fc`, tests the existing interface with constant probability one. Its proposed result is conditional on supplied field, elder-interface and frame objects; it would not identify that artificial law with the geometric elder event or establish existence of those objects. Hosted run [40957238766](https://github.com/d6g8k5htny-coder/Math-/actions/runs/40957238766) is queued. The candidate is not CLOSED, kernel-verified, or included in the 367 declaration records.

### Main evidence delivery integration

Main [#704](https://github.com/d6g8k5htny-coder/main/pull/704) merged at `1e2be44c69e08e47472041ac329e1b27f7dec793`. The candidate preserves its delivery metadata and full history. At this comparison cut, 375 changed evidence paths matched exactly by Git blob and mode; the candidate's `.gitattributes` retained all main rules plus one exact preserved-log rule. This is a tree comparison, not a final gate. [Evidence](cbz/evidence/main704-reconciliation-20261029.json).

### Atlas browser controls in the composed candidate

The candidate incorporates [main #702](https://github.com/d6g8k5htny-coder/main/pull/702), including its fixed-ID runner, unchanged UI assertions and sandboxed browser workflow. Its previous hosted run at `c0205ca9e8683b2847008f712582e183f723e186` observed both browser tests passing in normal and optimized modes without skips; it does not qualify this new composed source. The separate #690 admission delivery remains open and its changes are not implied by this integration. [Exact commands and scope](docs/cbz/ATLAS_BROWSER_REGRESSIONS.md). After installing the existing pinned browser requirements, run both `python -B tools/run_atlas_browser_regressions.py --output NEW_NORMAL_DIR` and `python -B -O tools/run_atlas_browser_regressions.py --output NEW_OPTIMIZED_DIR`; self-tests use `python -B -S -m unittest tests.test_atlas_browser_regressions_runner -v` and the same command with `-O`.

### Correction to lifetime-interface boundary wording

Five lifetime-related quarantine rows previously said their abstract predicates hold for every interface inhabitant. That was an incorrect universal assertion: these are predicates on supplied objects, and the source itself warns that the pair-law interface permits counterexamples. The original full wording is retained in the catalog alongside the corrected boundary. A separately recorded Gaussian-field constructor is not an elder/pair-law binding. [Correction evidence](cbz/quarantine/lifetime-predicate-wording-correction.json). No parent theorem is promoted.

A source-level [conditional constant-law countermodel](cbz/quarantine/lifetime-pair-law-countermodel.md) explains why the abstract pair-law interface does not imply the planar lower rate even when its upper predicate holds. The field, elder interface and frame remain supplied hypotheses; the manuscript is not a kernel receipt and Q-1048 remains quarantined.

### Archived axis-envelope precision observation

[Source inspection and exact-rational comparisons](cbz/quarantine/axis-envelope-precision-observation.md) identify that the archived point-certificate script constructs its decimal threshold and point offset before raising mpmath precision and reuses the point in both later runs. Under a fresh default 53-bit context, both differ from their stated decimals in the recorded directions. The archived process has not been rerun; no exact point violation, opposite inequality or recovered Q-5026 packet is claimed.
