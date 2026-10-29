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
| G5 | Existing root/auxiliary/fixture suites | Complete Math/isolated trial; resolve10 main skip coverage cases; rerun final admitted source |
| G6 | Core controls/downstream24/replica24 | Final explicit inventory/threshold/rejection reasons |
| G7 | Workspace local/network manifests | Reconcile historical snapshot roles versus live pins |
| G8 | Exact evidence/canonical docs | Final source/data/doc consistency |
| G9 | Existing commands/fixture fetches | Baseline-tag fresh clone and complete documented runner |
| G10 | Native refs/PRs/Git/bundles/leases | Terminal distinct deliveries and current enumeration |
| G11 | Tool tests/registry | Complete reviewed census admission and register all mission changes |

Disjoint source repairs are coordinated in main#684. Missing tooling is a build task, not an owner dependency. Unsupported API errors were preserved with successful inventory fallbacks. No unexecuted job is green.
