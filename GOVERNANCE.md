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

No new certifying tool has yet been admitted by this mission. Existing tools
are being inventoried against their actual tests and execution coverage.
Mission tools must be finished, documented with exact usage and verification,
and registered here and in `TOOLING.md` before final admission. Tools are a
means to closure, never a reason to open a general infrastructure project.


## Registered source inventory instrument


### cbz_source_audit

Path: tools/cbz_source_audit.py. Purpose: reconcile a hash-pinned local .lean
inventory and flag bounded lexical hazards without silently dropping sources.
It is an inventory/review tool, not a full Layer A certifier. Every report
leaves G1 and CBZ-0 NOT_ESTABLISHED.

Run: python -B -S tools/cbz_source_audit.py --manifest cbz-source-manifest.json

Manifest paths and SHA-256 values pin every Lean source in the selected local
root. Symlinks, unreadable/missing files, malformed UTF-8, unlisted Lean sources,
hash mismatches and resource-limit failures are explicit refusals. Nested Lean
comments and literal strings are masked; unsupported syntax is reviewed.
Optional output creates a new regular file outside the audited root and
refuses overwriting earlier evidence or inputs.

Verification: 36 real-CLI tests; tests/test_cbz_mutation_harness.py has 14
real-process controls. tests/run_scanner_mutations.py applies 39 source mutants
to the scanner. Its declared threshold is 39/39 killed with a passing unchanged
suite, valid CLI smoke, full case count, assertion failures and no harness/test
errors or skips. Startup crashes and invalid JSON are not successful rejection
evidence. The original 37/39 cut remains historical evidence; the final local
cut reached 39/39 after fixing uncovered tests and output/resource defects.

Limits: no grammar/elaboration/kernel proof, complete dependency audit,
semantic bound/nonvacuity proof, historical deletion comparison, concurrent
snapshot atomicity or status-truth certification. These require their actual
independent evidence. A clear lexical scan cannot make G1 green.

### formal_gate_check

Existing tool: tools/formal_gate_check.py.
Source-only mode checks the SIDE24 arithmetic manifest and exact source bytes.
With --run-lean it requires pinned Lean, fresh build, leanchecker, target axiom
audit and actual negative controls. Its scope is the registered 31 arithmetic
targets, not an analytic coefficient theorem or all federation mathematics.

## Ownership and maintenance

Tools are standard-library Python except documented Lean execution.
Maintenance is self-contained; any subsequent edit repeats the relevant
tests and updates this registry. No future reserved reviewer, human approval
queue or background monitor is installed here.
