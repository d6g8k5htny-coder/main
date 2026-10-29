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

The historical 367 literal declaration records are CLOSED in their exact kernel-checked package contexts (two later conditional pair-law declarations are separately indexed below): main arithmetic31, Math core63, specification78, smooth-source116, C29713 and determinant66. [PROOF_INDEX.md](PROOF_INDEX.md) links to the original source and available printed type and axiom records; consult the pinned source for complete statements. Ninety-four records lack embedded full signatures, and 45 of the other 273 printed types contain ellipses. Main source is `c3377cc8c0fb35844dc7613976d6e746fdc05600`; Math core/additional source is `8151f592918a845bf19eb8eafaf8444540a32734`; determinant source is `74565f013575b7b05ecfc4034cd5d638ee355353`. Package and source path are part of identity: determinant reuses 13 names from the separate core. These counts are not independent discoveries. No broader Gaussian/Palm construction, coefficient/enclosure identification or parent theorem is closed by these records.

Actual additional empty-kernel replays covered specification901 owned/65,210 dependencies, smooth222/55,236, C297131/38,531 and determinant157/58,421. Per-theorem axiom/type reports and actual adverse controls passed. The replay helper and its generated Lean audit have self-tests and source mutations. See the exact commands in [additional reproduction](cbz/evidence/kernel/additional/HOW_TO_REPLAY.md) and [determinant reproduction](cbz/evidence/kernel/determinant/HOW_TO_REPLAY.md). Defining any of the remaining specification Props does not assert it is universally true. Model satisfiability/premise discharge is not inferred.

Completed engineering changes:

- main#685 governance/ledger merged at `81acc060341e8a31cc4cd4f9523ef71fb6c06175` after 8/8 exact-head checks passed.
- main#693 manifest totals/duplicate-path repair merged at `27c89b05bd2503c580b692f57146d608a7f0f668` after 8/8 exact-head checks. Superseded draft #482 is closed; its source is preserved.
- Five missing exact historical replica fixtures restored in meta-framework, integrated at `0f5894f81a5462c10ee5ec8f8109f466e712e47e`. Unchanged67 tests,24 mutations and77 central contracts pass per mode; no checker/test/bound relaxed.
- Math recheck recovered from observed exit137/OOM using `LEAN_NUM_THREADS=2`; same source/checks passed. Original failure retained; the complete gate subsequently passed every admission control.
- Verified full-history bundles before guarded cleanup. Removed main45,Math40,governance30,query6,meta37 contained branches; all 158 absences independently read back. Open-PR heads/bases, protected and workflow-used refs excluded.

## Quarantine boundary

The strict CLOSED domain contains the 367 historical context-bound literal records and the two separately indexed, branch-verified conditional pair-law declarations. [The adopted catalog](cbz/quarantine/catalog.json) and [readable boundaries](cbz/quarantine/BOUNDARIES.md) describe 307 records labelled QUARANTINED: source/model applications, conventional arguments, the archived trial suite and five real-record coverage gaps. Some test records have engineering or fixture blockers rather than mathematical blockers. Full §11 proof-quarantine compliance and isolation, and A5 treatment of those tests, remain unverified. Remaining numerical-package mapping is being reconciled, so complete federation obligation coverage is not yet claimed.

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

The Node launchers validate pinned fixture bytes. Raw Node discovery without fixture variables is incomplete. Historic main commits `71400b94f6cb354a8cf7aba73ffede2138a64efa`, `a12c178c0f857a130cf434e9efd44233a038195b` and `f2e432ea5c86742e480c66624775bc9103343314` must be available; shallow clones need exact fetches. At earlier source cuts, optional browser and pytest cases were observed separately in both modes; the browser results were at `c0205ca9e8683b2847008f712582e183f723e186`, and do not qualify the composed candidate. All five actual restricted-container tests were separately observed in both modes at their recorded cut. Five real-record cases remain unexercised because their data/interface prerequisites are absent; retain those gaps, never fabricate records to pass them. The archived trial suite reproduced 87 failures in each isolated mode; it is not green. The restricted-container run used a verified source archive and a Docker-managed 64 MiB output tmpfs, with all original security/resource assertions enforced; this is local runtime evidence, not a GitHub-hosted run.

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

### Kernel-verified conditional pair-law countermodel

At Math branch commit `5a470bfd739aba22e32b91587d5dfc031a7de3fc`, hosted [run 40957238766](https://github.com/d6g8k5htny-coder/Math-/actions/runs/40957238766) built the pinned specification, compiled the constant-one proof and passed `leanchecker`. Both declared theorems report only `propext`, `Classical.choice` and `Quot.sound`. They retain supplied field, elder-interface and frame hypotheses; they do not identify the artificial law with the geometric event or construct an elder interface. The two [separately indexed CLOSED declarations](cbz/evidence/lifetime-countermodel/observation.json) are additional to the historical 367 records and are not yet merged to Math main. The artifact ZIP has not independently been downloaded.

### Main evidence delivery integration

Main [#704](https://github.com/d6g8k5htny-coder/main/pull/704) merged at `1e2be44c69e08e47472041ac329e1b27f7dec793`. The candidate preserves its delivery metadata and full history. At this comparison cut, 375 changed evidence paths matched exactly by Git blob and mode; the candidate's `.gitattributes` retained all main rules plus one exact preserved-log rule. This is a tree comparison, not a final gate. [Evidence](cbz/evidence/main704-reconciliation-20261029.json).

### Atlas browser controls in the composed candidate

The candidate incorporates [main #702](https://github.com/d6g8k5htny-coder/main/pull/702), including its fixed-ID runner, unchanged UI assertions and sandboxed browser workflow. Its previous hosted run at `c0205ca9e8683b2847008f712582e183f723e186` observed both browser tests passing in normal and optimized modes without skips; it does not qualify this new composed source. The separate #690 admission delivery remains open and its changes are not implied by this integration. [Exact commands and scope](docs/cbz/ATLAS_BROWSER_REGRESSIONS.md). After installing the existing pinned browser requirements, run both `python -B tools/run_atlas_browser_regressions.py --output NEW_NORMAL_DIR` and `python -B -O tools/run_atlas_browser_regressions.py --output NEW_OPTIMIZED_DIR`; self-tests use `python -B -S -m unittest tests.test_atlas_browser_regressions_runner -v` and the same command with `-O`.

### Correction to lifetime-interface boundary wording

Five lifetime-related quarantine rows previously said their abstract predicates hold for every interface inhabitant. That was an incorrect universal assertion: these are predicates on supplied objects, and the source itself warns that the pair-law interface permits counterexamples. The original full wording is retained in the catalog alongside the corrected boundary. A separately recorded Gaussian-field constructor is not an elder/pair-law binding. [Correction evidence](cbz/quarantine/lifetime-predicate-wording-correction.json). No parent theorem is promoted.

The [conditional constant-law countermodel](cbz/quarantine/lifetime-pair-law-countermodel.md) now has the separate kernel evidence above; its field, elder interface and frame remain supplied hypotheses, and Q-1048 remains quarantined.

### Archived axis-envelope precision observation

[Source inspection and exact-rational comparisons](cbz/quarantine/axis-envelope-precision-observation.md) identify that the archived point-certificate script constructs its decimal threshold and point offset before raising mpmath precision and reuses the point in both later runs. Under a fresh default 53-bit context, both differ from their stated decimals in the recorded directions. The archived process has not been rerun; no exact point violation, opposite inequality or recovered Q-5026 packet is claimed.

At the recorded candidate cut, an ad hoc comparison found the same 307 unique quarantine IDs in the catalog, proof-index rows and readable headings, with each exact boundary text present, and the same 367 unique CLOSED IDs in the index and literal evidence records. This is a narrow index consistency observation, not a final G8 or completeness audit. [Evidence](cbz/evidence/index-id-alignment-20261029.json).

The [same source-level argument](cbz/quarantine/lifetime-pair-law-countermodel.md) also shows that the permitted constant-zero pair law defeats the original upper predicate, conditionally on supplied field, elder-interface and frame objects. This makes the universal-interface limitation explicit; Q-1045 remains quarantined.

The 14 historical graph nodes without literal catalog-ID mentions consist of three D1 reading-rule components, eight D5 review/checking components and three engineering or inventory records. Twelve referenced blobs were read at the pinned Math tree and matched their tree-reported byte lengths; their original graph edges and linked parent-row mentions are recorded. Review/checker labels do not prove a theorem or establish complete semantic coverage. [Observation](cbz/quarantine/math-graph-unmentioned-source-observation.json).

The central literal-record JSON links the 63 Math core and 31 main arithmetic declarations to their pinned source and type logs; those 94 entries do not embed complete source headers or signatures. Pretty-printed logs can abbreviate terms, so the pinned source remains necessary to inspect the full literal statement. This is a documentation and evidence-link correction, not a change to the recorded proof outcomes.

An actual test-first run against the historical grader/loader source (`9d52d680`) observed 15 assertion failures and no errors in each Python mode, including three poisoned-bytecode loader controls per mode. The raw job log is retained at [auditor A8 RED](cbz/evidence/auditor-grader-a8/red/9d52d68086f9-job123737390295.log). Later source-only loader and grader repairs are candidates awaiting fresh execution; this RED confirms defects and does not qualify G0.

### Cross-repository pin discrepancy

Universal-Law-Workspace [PR #31](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace/pull/31) actually passed its historical eight-pin network resolution but failed `--require-current`: at that run, the main and Math pins differed from their then-observed default-branch tips. Main has advanced since that run. [Exact run and hashes](cbz/evidence/ulw-pin-discrepancy-20261029.json). Freeze the final source heads, update the gitlinks and manifest together, and rerun the current-pin checks; G7 is not qualified.

### Temporary split-delivery limitation

At main commit `1e2be44c69e08e47472041ac329e1b27f7dec793`, #704 delivered historical replay documentation, manifests and evidence, but the four replay-tool Python sources and tests were deliberately deferred to the composed foundation candidate with their governance registry. Consequently the historical README's runnable-tool commands cannot be followed from that main commit alone; see [the split-delivery metadata](cbz/evidence/additional-delivery-20261029.json). The commands above describe the composed candidate, which contains those source files and still requires integration and fresh verification. The pending #705 determinant evidence split uses the same four-file deferral. Preserve historical checksums and source bytes when reconciling these deliveries.

### Auditor and Node candidate handoff

The auditor repair remains on branch `cbz/auditor-grader-a8-20261029`, observed head `78bbd4b388b86ddd9bc886b42f87e1b79e295b1f`; main #707's current Node candidate head is `9651b9c8f86fae3a696ceeb836a4e3c6c17a11a2`. Their new source and tests have not been qualified by a completed current-head run. A delegated repair turn was automatically stopped because its content was flagged for possible cybersecurity risk. Preserve its previous evidence and do not infer a passing gate or completed repair.

### Unavailable local numerical and classification work

The execution environment is currently unavailable. A previously observed local commit `4d416a851ecb1d474141bbfc60173927949124ce` contained a prepared 27-record numerical mapping and a 367-record source-classification update; its push did not complete, and the commit is not available through the examined public GitHub refs. The published `cbz/classifications.json` contains 94 records. The 367 literal declaration records and their published evidence remain available separately. Neither the 27 numerical records nor the local classification file has been recovered; restore the execution environment and preserve those exact bytes, or independently regenerate and verify a new mapping from published source evidence. No claim of permanent loss or completed numerical disposition is made.

Among the other 273 embedded printed signatures, 45 visibly contain Lean's `⋯` abbreviation at this cut. The count identifies visible abbreviations only; it is not a full semantic or vacuity audit.

### Further formal qualification in progress

A historical [main #358](https://github.com/d6g8k5htny-coder/main/pull/358) job printed empty axiom lists for 28 `UniversalLaw.P15` targets not named in the historical 367-record index or the two newer pair-law records. The exact three source blobs match its tested merge. [Artifact-verifier PR #711](https://github.com/d6g8k5htny-coder/main/pull/711) is prepared to check the original ZIP, receipt, logs and source bindings; its checks remain pending. The 28 targets are not added to the CLOSED count, and no general logarithm, cover semantics or model application is inferred.

Math [#539](https://github.com/d6g8k5htny-coder/Math-/pull/539) has a candidate repair for the recorded 99-module census failure. At head `785a71f0b8267d6d11b736dc5f621b1265bd7198`, preparatory workflow steps succeeded and the 99-module kernel replay was still running at the observed cut. Its helper self-tests use the affected historical grader and cache-sensitive loaders, so those step results do not establish auditor soundness. The replay driver separately reads and compiles the helper source bytes; assess any completed kernel observation against its raw output and stated assumptions. Math [#537](https://github.com/d6g8k5htny-coder/Math-/pull/537) has an observed kernel pass at tested merge `b64ad0825c44669366ff13f5a990c80fa9510dc2` and a separate receipt-log integrity pass; further environment and grader repairs are unqualified. The kernel observation assumes a non-adversarial execution interval and does not authenticate a transient edit-and-restore or cached dependency object provenance.

Source-cut pytest-dependent and pinned graph-composition diagnostic workflows are queued; their exact commands, pins and limitations are registered in [TOOLING.md](TOOLING.md). Neither result has been observed, so the associated skips remain unqualified.

The [P15 target observation](cbz/evidence/historical-p15-target-observation.json) preserves all 28 historical empty-axiom output lines and their exact source Git blobs; a literal-name comparison found no overlap with the historical 367-record index; the two newer pair-law names are also distinct. This is source-cut evidence, not artifact authentication or new CLOSED admission.

All three D2 companion jobs at the historical Math #537 tested merge completed successfully. Two broad companion suites each ran 1,066 tests per mode with one declared skip per mode; their artifacts have not been independently downloaded here. These results do not qualify the newer custody/grader candidate or complete G5. [Job observations](cbz/evidence/math537-d2-companion-observation.json).

### Math main advanced

Math [#492](https://github.com/d6g8k5htny-coder/Math-/pull/492) merged at `9f8dbb4d18b9f3ebd9ffac6b0caa42df9f343966` after six exact-head checks completed successfully. Its four changed paths concern the elder-cusp engineering verdict protocol; this does not promote an informal mathematical result. Candidate integrations and the workspace's current pins must be reconciled with this newer Math main before final verification. The previous `95eb` observations remain historical source cuts.

A [later non-atomic observation](cbz/evidence/public-default-heads-20261029.json) records the default-branch heads of all ten identified public repositories, with main, Math and workspace re-read at that cut. Two private infrastructure repositories, local worktrees and final pin consistency remain outside this observation. It is not a final baseline vector.

ULW #31 has since advanced to candidate `9d4ec7bdc79295f4e67944be352d8643d3cc7bff`, updating the main and Math gitlinks and matching manifest/README pins to a separately observed set of eight public tips. Its current-head tests and strict pin checks are queued. The observation is non-atomic, and neither the candidate nor final G7 is qualified until those checks complete and final source heads are frozen.

A separate [constant-zero Lean candidate](https://github.com/d6g8k5htny-coder/Math-/tree/cb1fe3d189d94ee207750fc005eadf0596fb9cef) formalizes the Q-1045 source-level contradiction with an explicit positive radius. It is not yet kernel-verified or included in the 369 CLOSED records; its [hosted run 40960912254](https://github.com/d6g8k5htny-coder/Math-/actions/runs/40960912254) is queued, and all field/elder/frame hypotheses remain explicit.

A separate [367-record reconstruction candidate](https://github.com/d6g8k5htny-coder/main/blob/43c011a79eced964b923d3fc18c3641560e79a64/cbz/reconstruction-367/README.md) uses the published historical 94+207+66 source and replay bindings to build a new classification file; it does not recover the missing local file. Hosted run [40960669819](https://github.com/d6g8k5htny-coder/main/actions/runs/40960669819) is queued, so neither its tests nor its generated mapping and inventory roundtrip have passed yet. The two later pair-law records are outside this historical reconstruction and need separate bindings.

The historical pair-law job did not run the specification package's fixed-tree inventory checker. That checker would reject the extra proof file in the packet root; integration is being prepared with the verified source outside that packet and with the existing checker retained. The scoped kernel result remains tied to its original source cut and does not imply the full package regression passed.

All 127 paths in the historical additional-proof source manifest matched their Git blob IDs in Math main `9f8dbb4d18b9f3ebd9ffac6b0caa42df9f343966` at a recorded tree comparison. This confirms source identity for those paths only; it does not establish a new Lean run, dependency-cache provenance or a final gate. [Observation](cbz/evidence/additional-current-tree-observation.json).

The 17 source/configuration paths named by the historical determinant receipt have identical Git blobs at its checked Math commit and at Math main `9f8dbb4d18b9f3ebd9ffac6b0caa42df9f343966`. This is a scoped source-tree comparison, not a current Lean replay, artifact authentication or compiled-object provenance check. [Observation](cbz/evidence/determinant-current-tree-observation.json).

[Math #543](https://github.com/d6g8k5htny-coder/Math-/pull/543) (head `9e89980bfa4f281e00550f481202033775f35485`) prepares integration of the exact same verified pair-law source outside the fixed-tree statement packet. Its workflow retains the packet inventory, checks source bytes, recompiles, compares the two axiom reports and invokes leanchecker. Current-head execution remains pending. The separate zero-law trial also places an extra file in the fixed-tree packet and requires the same integration correction before it can land.

The LM006 outer-test entry has an observed two-mode adverse failure: on its test-first source, a synthetic valid cached test module ran in place of the intended source, causing the selected control to fail. The separate repair candidate remains unverified and has not been promoted. [Exact job observations](cbz/evidence/lm006-outer-cache-red-observation.json).

Math #533's LM006 jobs completed successfully in both Python modes (run `40957720395`, jobs `123742046685` and `123742046345`) at tested merge `24bba37a9d86f2081746984c24b6309eb4e1ed3b`, with parents Math base `95eb8daaa02e89be1f816bc7f58eac812dd1b469` and PR head `5d2dfd44dc25ebe4d3186e08192d115eb107d0d5`; the merge tree differs from that PR head. Their ZIP payloads have not been independently inspected here. The separate outer-entry poisoned-cache control has an observed RED and its repair remains unverified, so these job successes do not qualify that later change or a mathematical bound.

A prior Math core custody mutation suite failed because its `git-replacement-refs` mutant survived: the mutation changed a Git protection variable from `1` to `0`, but Git documents that any set value enables the protection. The original RED is preserved; a revised mutation that removes the override still needs both-mode execution at its exact source. [Observation](cbz/evidence/math-core-git-replacement-mutant-red.json).

The 94 historical Math-core and main-arithmetic declaration records directly reference 11 unique source paths; their Git blobs and regular-file modes match between each historical checked tree and the later observed default tree. This does not cover all package/dependency inputs or constitute a new Lean run. [Observation](cbz/evidence/core-current-source-tree-observation.json).

Independent review found that #705's original tree and the earlier #704 additional-evidence split each omitted four files named in their checksum manifests. [The repair candidate](https://github.com/d6g8k5htny-coder/main/pull/705), observed at head `30184a6c7337099f910809116523112394e6504b`, contains both sets with the original four Git blob identities. Its workflows for all four manifests, historical tests, actual Lean controls, adverse examples, and the separate source-only Python mutation-grader candidate are queued. The proposed grader does not repair the separate Lean self-test limitation. No check has yet been observed passing for this head, and #705 remains unmerged; the canonical documents require reconciliation before integration.

### Replay-helper mutation-grader qualification limit

Source review found that the historical bounded replay helper's Python mutant classifier accepts a named failure plus an `AssertionError` appearing anywhere in the same failing transcript, without excluding other failures/errors or binding the assertion to the selected guard. Its test loader can also read existing bytecode. These are source-level findings, not a newly executed adverse test. The old mutation counts do not establish auditor soundness; targeted controls and any fix are pending. The historical Lean executions and literal source/axiom records remain preserved at their stated cuts. [Exact source review](cbz/evidence/replay-helper-grader-source-review.json).

The same source review found that the historical replay mutation runner's baseline condition does not check a nonzero test count, and that the separate Lean self-test normally imports the helper. A zero-test baseline and stale-bytecode path therefore need actual adverse controls; these are not claimed as executed failures yet.

### Archived replay self-test text

An independent readback identifies the exact historical Python and Lean report and log blobs, including a final-normal Lean report with a different self-test digest from the delivered source. These are archived-text observations, not authenticated process execution or a G0/G6 pass. [Evidence and limits](cbz/evidence/historical-replay-text-observation-20261029.json).

### Further scoped Lean candidates

Math branch `cbz/lifetime-zero-law-integration-20261029` at `36440828402102a840bc6bdffc2e1238ce6e4bea` contains an abstract zero-pair-law countermodel for supplied interfaces and positive length. Run `40964012873` is queued. Math branch `cbz/lifetime-e2-e4-implication-20261029` at `f19b254efae612ad5d4f010908cd615ca8c708fb` proposes that the literal lower predicate E2 implies E4, which itself assumes the literal parent upper predicate; run `40964002205` is queued. Both preserve the fixed packet inventory and propose packet, build, axiom and kernel checks. Neither is yet kernel-verified or proves the intended geometric law or interface existence.
