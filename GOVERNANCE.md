# CBZ-0 governance

Effective 2026-10-29, by the owner's **MASTER DIRECTIVE — CRYSTALLINE
BASELINE PROTOCOL**. This is the current operational authority throughout
the federation. Earlier owner mandates, freezes, family-review gates,
approval forms and custody ceremonies are historical where they conflict.
Platform permissions and denials remain binding. Historical mathematical
statements, test failures, authorship and verification outcomes are facts;
this governance change does not rewrite them.

## Mission and authority

The sole mission is Clean Baseline Zero (CBZ-0), not additional research.
Agents may build, repair, replace, integrate, archive or remove operational
material needed for that mission. No owner question, selection, approval,
external review quorum or waiting period is required. Resolve ambiguity
conservatively for mathematical rigor and aggressively for consolidation.
Coordinate independent tasks without blocking on a peer's approval.

Priority: Layer A; CBZ-0; reduced maintenance; cold-start legibility;
auditor soundness; speed; aesthetics. Select work by closure, unblocking
and maintenance benefit, less effort and ten times the rigor risk. On
conflicts preserve the mathematically stronger valid result; prefer the
simpler operational structure; combine compatible work. Archive a genuinely
incompatible alternative with its exact boundary and evidence.

## Layer A: immutable

1. No logical placeholders or stubbed theorems presented as closed.
2. No unverified axiom injection.
3. No floating-point certification of mathematical bounds: exact rational
   or justified interval/real reasoning only.
4. Never loosen a mathematical bound to pass a check.
5. Preserve failing tests; repair or isolate them with a written boundary.
   Consolidation requires evidence of redundancy under a stricter proof.
6. Mark conclusions relying on unsatisfiable hypotheses in the proof index.
7. Every status statement must describe observed source/build/audit state.
8. Certifying tools require their own tests, including adverse mutations;
   they must not skip cases silently or manufacture a passing result.

Kernel verification establishes only its exact formal statement and
declared axiom dependencies. Prose review, finite algebra checks, hashes,
merges and engineering CI cannot establish kernel closure or a stronger
mathematical interpretation.

## Layer B: operational statutes

- **B-001 (2026-10-29):** Canonical federation authority is this file;
  `STATUS.md` is the complete cold-start entry. `PROOF_INDEX.md` records
  exact CLOSED/kernel evidence or QUARANTINED boundaries. `TOOLING.md`
  registers mission tools; `BASELINE.md` binds snapshots, commits and gates.
  Rationale: replace dispersed procedural approval chains with reproducible
  evidence. Scope: all in-scope repositories.
- **B-002 (2026-10-29):** Decisions are append-only in `DECISIONS.md`.
  Date, phase, layer, action, rationale, reversibility and evidence are
  required. Discovery evidence is timestamped and source-bound; a changing
  federation is not represented as an atomic observation.
- **B-003 (2026-10-29):** Root owns shared canonical files and integration;
  delegated discovery writes only its assigned evidence directory. Agents
  may make autonomous bounded decisions and finish without peer approval.
  Rationale: avoid concurrent overwrite while preserving parallel work.
- **B-004 (2026-10-29):** Snapshot before recoverable/destructive operations;
  one extra verification cycle before irreversible published history work.
  No silent deletion. Preserve removed material in a documented tag,
  recoverable archive or bundle. No history rewrite is needed merely for
  aesthetic cleanup.
- **B-005 (2026-10-29):** A gate is PASS only after its command ran and its
  output and exit were observed at the recorded source. Source-only checks
  and historical hosted receipts must not substitute for fresh execution.
  Final CBZ gates must all be observed in one final run.
- **B-006 (2026-10-29):** An auditor certifies only the properties and
  domain its implementation and adverse tests establish. Explicitly
  report unsupported claims or cases; never call them proven by a lexical
  scan. Quarantine classifications do not erase detected hazards.

## Quarantine

Every proof quarantine records an ID, title, exact source location, Layer A
status, exact proven/unproven boundary, mathematical blocking reason,
attempts, resumption cost, isolation, owner `none` and maintenance `zero`.
Closed formal code must not import a quarantined logical dependency.
Quarantine is dormant until deliberately reopened. No new research is
authorized merely because a quarantined question exists.

## Execution and termination

Discover all repositories/refs/PRs/proofs/CI/admin material and tool gaps;
write the inventory. Finish or quarantine obligations, integrate compatible
work and archive obsolete ceremony. Run G0 through G11; repair discrepancies
at source. Publish the canonical handoff and perform a fresh-clone test
using `STATUS.md` alone. Each iteration must make a measurable change;
after two no-op iterations diagnose and change strategy.

Do not declare CBZ-0 until all twelve owner criteria are observed together:
clean Git, no dangling deliveries, all proofs closed/quarantined, no Layer A
violations, all regressions green, Lean/kernel/axioms green, mutation target
met, pins consistent, tools documented/verified, canonical surfaces truthful,
cold-start reproducibility and complete decisions/no administrative residue.
A resource-limited terminal state must explicitly be a Partial Baseline,
with precise isolated residue and unmet criteria; it must never be called
CBZ-0.

## Tool registry

- **CBZ source census (2026-10-29):** `tools/cbz_inventory.py`, with
  `tests/test_cbz_inventory.py` and exact usage/schema in
  `docs/cbz/INVENTORY_TOOL_PLAN.md`. Generate/verify a complete offline
  tracked/index-source census and immutable declaration/evidence bindings.
  Run `python3.11 -B -S -m unittest discover -s tests -p
  test_cbz_inventory.py -v`, then repeat with `-O`. Observed 75/75 each mode,
  including nine source mutations killed by intended assertions. The RF312
  base-plus-overlay effective map has exact checked-tree coverage and each
  regular Git blob's declared SHA256 is verified. Actual 94-binding capture
  and verify passed with all 45 effective RF312 files checked. The earlier
  12-repository capture/verify passed 13,654 entries. Review findings were repaired and
  re-reviewed. Owner:none, self-contained. This is **not** a Layer A semantic
  certifier or kernel-run authenticator; its output explicitly grants neither
  semantic completeness nor CBZ green. Criteria1,3,8,11.
- **Bounded proof-closure replay (2026-10-29):**
  `cbz/evidence/kernel/additional/replay-tool/replay_tool.py` and its README,
  tests, Python mutation runner and actual Lean self-test runner. The frozen
  determinant evidence contains the same self-contained tool version.
  Generates the pinned official-kernel trust-zero replay and checks transcript
  consistency; supplied text is not execution authentication. Exact commands
  are in the README and both package HOW_TO_REPLAY.md guides. Observed 31 tests
  and three Python validator mutants per mode; actual Lean positive/six adverse
  controls/six generated-auditor mutants passed in both modes. Owner none,
  self-contained; criteria 3,6,9,11. Approved axioms remain exactly the three
  immutable names above. Raw receipts are retained unchanged.
- **PR/citation evidence tools (2026-10-29):**
  `cbz/evidence/stale-pr-dispositions/{pr_evidence,mutation_check}.py` and
  adjacent `main-obligation-census/build_census.py`. Exact commands and scope
  are in their README/REGISTRY_ADDENDUM. Observed 35+9 tests and 49 intended
  source/harness mutant kills per Python mode; actual network acquisition and
  26-head historical regeneration passed. They do not certify mathematics,
  independence or actual PR disposition. Obsolete staging text is historical.
  Owner none, self-contained; criteria 2,3,10,11,12.
- Existing core compiled auditors, fixture launchers and recovery commands
  are catalogued in `TOOLING.md`. A scope-limited or historical success is
  never promoted to a federation certificate.

Mission tools must be finished, documented and self-verified before admission.
Tools serve closure; no general infrastructure project is authorized merely
for elegance.

## Additional statutes

- **B-007 (2026-10-29):** Ref cleanup requires a verified recoverable snapshot,
  positive ancestor evidence, exact SHA leases, exclusions for open-PR heads
  and bases/protected/workflow refs, and observed readback. Rationale: avoid
  deleting new work that arrives after a non-atomic inventory. Scope: ref
  cleanup throughout the federation.
- **B-008 (2026-10-29):** Use tested `LEAN_NUM_THREADS=2` for Math kernel
  replay in this16GiB runtime. Preserve the original default-parallelism
  exit137; never skip the recheck. Rationale: the same source and kernel
  checks pass with bounded resource concurrency.
- **B-009 (2026-10-29):** Immutable source bindings may name an ancestor
  commit if actual current proof bytes remain identical. Explicit historical
  quarantine bindings cannot discharge CLOSED proofs. Final inventory output
  is external to scanned source trees; no file is expected to hash itself.
  Rationale: exact identity without impossible self-reference.
- **B-010 (2026-10-29):** Execute archived live-write probes only in a
  clean, credential-free network-disabled scratch namespace; retain original
  assertions and all failures. Mutating fixtures require separate fresh
  normal/optimized source copies. No network or permission denial is bypassed.
  Rationale: observe actual tests without granting old probes ambient write
  authority.
- **B-011 (2026-10-29):** G6 threshold is100% of every admitted tool's
  explicit source-bound mutant inventory. A kill requires the intended test
  assertion failure, not import/syntax/launch failure; stale mutation anchors
  fail. Retain positive harness controls and original RED evidence. Scope:
  mission certifiers and their declared mutation campaigns.
- **B-012 (2026-10-29):** Literal theorem identity includes repository,
  checked source cut, package/source path and declaration. Identical names in
  isolated kernel packages are not conflated. A new normalized observation
  receipt may index actual raw runs, but never rewrites old receipts or claims
  authentication. The explicit replay census format binds every source file,
  target/path, axiom list and raw artifact at immutable current bindings.
  Rationale: reproducible context and evidence without self-hash cycles.
- **B-013 (2026-10-29):** A quarantine covers the exact missing mathematical
  application/assertion, not the universal truth of vocabulary definitions.
  Historical tests and all original failures remain visible. Trial is dormant;
  five unsupported real-record methods remain unexercised. No active regression
  claim may disguise those observations as a green historical full suite.
  Rationale: terminal boundaries preserve facts rather than fabricate data.

- **B-014 (2026-10-29):** The active agent entry contains current CBZ instructions only. Superseded family-review/owner-approval requirements remain recoverable historical evidence and impose no task gate. Keep useful source/test/privacy guidance, exact current board routing and immutable verification requirements. Rationale: remove operative contradictory ceremony rather than repeatedly satisfy it.

### Composed verification components

The bounded Lean lexical scanner (`tools/cbz/layer_a_scan.py`) and Python floating-bound signal collector (`tools/cbz/a3_float_census.py`) are source prechecks, not semantic certificates. Their exact scope, commands, negative controls and observed mutation evidence are in `docs/cbz/CANONICAL_AUDITORS.md`. The source-custody and compiled-inventory checks are registered in `formal/COMPILED_INVENTORY.md` and the formal workflow. Component results at prior commits do not certify a new composed head. Owner: none; verification and limitations are recorded at those paths.

### Mutation-grader qualification limit

Source inspection found that both the lexical and A3 mutation graders accept any nonzero exit and an expected test failure even when foreign failures are present; neither requires the specific bypass reason. Their historical 66 and 23 mutant outcomes do not by themselves establish auditor soundness. Test-first repairs and fresh reason-bound controls are pending, so G0 and federation-wide G1 remain unqualified.

### Historical main regression skip observation

The archived [one-shot workflow](cbz/evidence/main-regression-skip-observation/workflow.yml.txt) downloaded the two specified hosted artifacts, checked their GitHub-reported ZIP digests and the original log digests, and listed the skipped test lines. The actual positive observation, exact source/run and limitations are in [result.json](cbz/evidence/main-regression-skip-observation/result.json). It is a historical diagnostic, not a Layer A auditor or a full G5 certificate; it has no dedicated adversarial test suite and must not be promoted as a certifier without one. Maintenance owner: none. To reproduce while the original GitHub artifacts remain available, run the exact archived workflow with `actions: read` or apply its listed hash and unzip commands to those exact artifact IDs.

### Atlas browser regression runner

`tools/run_atlas_browser_regressions.py` runs exactly two existing browser methods with zero skips in the pinned sandboxed browser environment. [Usage and self-verification](docs/cbz/ATLAS_BROWSER_REGRESSIONS.md) records source tests, historical actual runs, mutations and limits. It is an engineering test aid, not a mathematical auditor; current composed-head verification remains pending. Owner: none; criteria 5 and 11.

### Historical source-cut diagnostic workflows

The [sandbox artifact observer](cbz/evidence/main-regression-skip-observation/sandbox-observer-workflow.yml.txt) downloaded one recorded GitHub Actions ZIP, checked its SHA-256 and printed ten passing method lines; [result and limits](cbz/evidence/main-regression-skip-observation/sandbox-controls.json). Run its archived workflow with `actions: read` while the original artifact exists. It has no dedicated adversarial tests and is not an auditor or G5 certificate. Owner: none.

Two one-shot source-cut tests are queued: [pytest-dependent controls](https://github.com/d6g8k5htny-coder/main/blob/ba4077c2b3101cd969319b342e30c11b725a5f87/.github/workflows/cbz-pytest-controls.yml) and [pinned graph-composition controls](https://github.com/d6g8k5htny-coder/main/blob/8b03af28ccfe3276b8a8957b108ba541f17953d4/.github/workflows/cbz-graph-composition.yml). Their exact workflows give dependencies and commands; no completed result, independent self-test or gate admission is asserted. They are engineering diagnostics, not mathematical certifiers. Owner: none.

### Scoped pair-law Lean workflow

The archived [workflow](https://github.com/d6g8k5htny-coder/Math-/blob/5a470bfd739aba22e32b91587d5dfc031a7de3fc/.github/workflows/cbz-lifetime-pair-law-countermodel.yml) builds `Specifications.Lifetime`, compiles `CounterexampleE4.lean`, prints axioms and runs `leanchecker` in the pinned Math package. [Observed run and exact commands](cbz/evidence/lifetime-countermodel/observation.json). It is a source-scoped proof execution, not a general auditor; the GitHub artifact has not independently been downloaded and no final federation gate follows. Maintenance owner: none; criteria 3 and 6. Re-run on the exact branch cut or after integration with newly recorded source and results.

The separate [zero-law proof workflow](https://github.com/d6g8k5htny-coder/Math-/blob/cb1fe3d189d94ee207750fc005eadf0596fb9cef/.github/workflows/cbz-lifetime-pair-law-countermodel.yml) extends the scoped pair-law run by compiling and checking `CounterexampleZero.lean`. Its exact source and commands are at the linked commit; its hosted result is pending, and it is not an admitted general auditor. Owner: none; criteria 3 and 6.
