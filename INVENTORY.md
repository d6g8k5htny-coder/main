# CBZ discovery inventory

Initial observations span2026-10-29 from16:49UTC, non-atomic historical cuts, **not current counts after cleanup**. Public summary: `cbz/evidence/initial-public-inventory.json`; full private source/board evidence retained locally under `/workspace/cbz/discovery/federation`.

-12 repositories:10 public,2 private infrastructure;3 public archived.
-2,114 branches,25 tags,3,663 Git refs;69 open PRs,23 open issues initially;8 Workspace submodules.
-94 board/delivery containers,8,284 exact body hashes independently recomputed; pagination/count/thread-ID reconciliation complete.
-446 unique primary non-PR heads compared:163 contained candidates,290 distinct branches,1 unrelated history. After extra protection/workflow/open-base exclusions,158 pruned with leases and absences read back.
-3 unresolved native review threads: main PR340,341,346. Textual PASS/CLOSED words were not inferred acceptance.
-1,199 discovery artifact digests verified. Ref snapshot SHA256 `7d31b4024091bc11a8e70abfc34c07dfbad370d4516d8c72604491becbfaa0a0`.

Math8151 inventory:4,019 tracked files,162 Lean,182 workflows,286 exact workflow run blocks. Core:9 Lean files/116 constants, not the whole tree. Separate federation census captured13,654 entries at its own cut. Lexical/AST signals require classification, not automatic semantic verdicts.

## Tooling-gap queue

| Gate | Existing path | Remaining gap |
|---|---|---|
| G0 | Math formal tests/compiled controls | Final complete run; supplemental auditor behavioral/source-mutation tests |
| G1 | Compiled inventory/full source census | Exact terminal classification and semantic boundaries; no lexical substitution |
| G2 | Transitive closure/literal116 reports | Bind fresh final evidence and admitted successors |
| G3/G4 | Pinned Lean/lake/leanchecker | Finish all controls with tested thread limit2; preserve OOM failure |
| G5 | Existing root/auxiliary/fixture suites | Math and isolated trial observed; trial87failures/1skip per mode retained dormant, five main real-record gaps explicit; rerun final admitted source |
| G6 | Core controls/downstream24/replica24 | Final explicit inventory/threshold/rejection reasons |
| G7 | Workspace local/network manifests | Reconcile historical snapshot roles versus live pins |
| G8 | Exact evidence/canonical docs | Final source/data/doc consistency |
| G9 | Existing commands/fixture fetches | Baseline-tag fresh clone and complete documented runner |
| G10 | Native refs/PRs/Git/bundles/leases | Terminal distinct deliveries and current enumeration |
| G11 | Tool tests/registry | Complete reviewed census admission and register all mission changes |

Disjoint source repairs are coordinated in main#684. Missing tooling is a build task, not an owner dependency. Unsupported API errors were preserved with successful inventory fallbacks. No unexecuted job is green.

### Historical Math graph coverage

The retained Math graph at commit `8151f592918a845bf19eb8eafaf8444540a32734` has 49 nodes and 55 edges. Its complete records and edges match the preserved contribution exactly. In the current 307-row catalog, 35 node IDs occur literally and 14 do not; this is a text observation, not evidence of equivalent semantic coverage. Sixteen source groups and seven alignment gaps also need source-specific reconciliation. The original graph's historical review labels do not certify kernel closure. [Graph source](cbz/quarantine/current-math-graph-source.json) and [coverage observations](cbz/quarantine/math-graph-coverage-observations.json) record the exact cut and limits.

### Main regression skips at the recorded source cut

Both modes skipped the same 25 tests: five require a restricted container, two require a browser, two require Lean, one requires pinned cross-repository composition inputs, ten require pytest, and five require real-record fields or live freshness not present in the data. The five real-record boundaries remain Q-9002–Q-9006. Other historical separately run controls retain their original source and environment cuts; current-cut replacement coverage is not inferred merely from those older results. [Exact tests, reasons and artifact verification](cbz/evidence/main-regression-skip-observation/result.json) are recorded. Full G5 remains unqualified.

The two Lean-dependent methods skipped by broad unittest discovery were separately observed passing in both Python modes in the composed formal job: `test_actual_clean_and_omitted_constants` and `test_real_package_executes_with_receipt`. The tested PR merge has the exact same source tree as the composed candidate; the broad regression's successor adds only its workflow. This supplements the skip record without changing the full-suite result or establishing complete G5 coverage. [Evidence](cbz/evidence/main-regression-skip-observation/lean-controls.json).

### Later qualification and recovery limits

The lexical and A3 mutation graders have confirmed false-kill and bytecode-loader adverse failures; repairs and G0 remain unqualified. The local environment is unavailable, and the prepared 27 numerical boundaries and 367-entry source classification at unpublished local commit `4d416a851ecb1d474141bbfc60173927949124ce` have not been recovered from examined remote refs. The published classification file has 94 entries; literal declaration records remain separately available. The five restricted-container and two Lean-dependent methods were separately observed passing in both modes at the recorded composed source cut, without changing the original 25 broad-suite skips. [Details](STATUS.md).
