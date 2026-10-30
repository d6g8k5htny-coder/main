# CBZ-0 governance

Effective 2026-10-29, by the owner's **MASTER DIRECTIVE — CRYSTALLINE
BASELINE PROTOCOL**. This is the current operational authority throughout
the federation. Earlier owner mandates, freezes, family-review gates,
approval forms and custody ceremonies are historical where they conflict.
Platform permissions and denials remain binding. Historical mathematical
statements, test failures, authorship and verification outcomes are facts;
this governance change does not rewrite them.

## Current owner source

The latest complete master is preserved [verbatim in this repository](governance/MASTER_DIRECTIVE_20261029_SWARM.txt)
and [in main#686](https://github.com/d6g8k5htny-coder/main/issues/686#issuecomment-6511339492):
36,617 UTF-8 bytes, SHA256
`83ac6552fc6406e95213a1cfd6e8636095215e11aac6e3db473cd42c22275a8b`.
Its team-model sections 1.5, 7.4 and 8 supersede older exclusive-owner and
monitor-specific integration restrictions. Historical decisions retain their
original timestamps and outcomes; the latest directive controls future action.

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
- **B-003 (updated 2026-10-30):** Agents share accountability for CBZ-0.
  Ownership identifies the actual writer and current source scope; it is a
  coordination hint, not an exclusive lock or approval gate. Preserve compatible
  active work, communicate exact file/ref scopes and use guarded updates.
  Reassign or absorb blocked work explicitly in the decision ledger.
  Canonical documents remain single coherent surfaces; no agent identity,
  including a shared `/root` label, confers exclusive integration authority.
  Rationale: implement the owner's dynamic team model without silent overwrite.
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

## Team structure and coordination

**B-007 (owner expansion dated 2026-10-29):** Operate as a fluid team, not a
sequential coordinator queue. These are mandates, not permanent seats:

| Role | Mandate | Required handoff |
|---|---|---|
| Discovery | Enumerate current topology, obligations, refs, deliveries and gaps | Exact source cut, coverage and unknowns |
| Proof closure | Finish or precisely quarantine existing obligations without new research | Exact statement, hypotheses, kernel/axiom evidence and boundary |
| Tool building | Build the narrow tool needed for closure; include adverse self-tests | Source, commands, exits, supported domain and residuals |
| Verification | Execute the actual gate population and retain failures | Tested composition, raw outcomes and limits |
| Integration | Compose compatible work with guarded, recoverable mutations | Actual integrated source, checks and remaining discrepancies |
| Documentation | Keep canonical surfaces consistent with observed evidence | Source-bound status, runnable commands and team reconstruction |

An agent may change roles or absorb blocked work under the owner's ranking
function. Roles and a shared `/root` alias confer neither exclusive file
ownership nor scientific or organizational independence. Automated checks
and self-review do not become evidence of a broader mathematical claim.

### Conflict protocol

1. Register the incompatibility in [CONFLICTS.md](CONFLICTS.md), with both
   positions, actual actors, source cuts, constraints and affected paths.
2. Apply the hierarchy: mathematically stricter valid content wins for
   Layer A; simpler compliant structure wins for Layer B; combine compatible
   contributions; retain the stronger verified alternative and precisely
   quarantine a genuinely irreconcilable one.
3. Log the resolution, rationale and evidence in [DECISIONS.md](DECISIONS.md)
   before calling it resolved. Preserve the losing work's recovery path and
   all observed failures; no vote, silence or shared account supplies proof.
4. When a conflict class recurs, add a dated Layer B rule here that prevents it.

Agent questions may coordinate work for one work cycle; they are not
approval gates. If a peer is unresponsive or its output is unusable, proceed
with the best source-bound judgment, record the assumption and take useful
work. Do not wait for a human or coordinator. A real platform denial remains
binding; an expected-head guard is race protection, not a permission lock.

### Shared state, handoffs and convergence

Publish current task intent and source changes in the existing ledger and
task-local record. Prepare concurrent reversible changes in isolated branches;
re-read the current tree before mutation and compose rather than overwrite.
A handoff records from-agent, to-agent or available role, task, exact state,
what was done, what remains, discoveries, tools and evidence. The receiver's
actual action is separate from a delivery notification.

At each of the four phase boundaries, reconcile the inventory, conflict
registry and gate evidence, then re-rank work. Phase convergence is an
evidence synchronization point, not a provider quorum or waiting period.
Private or local working files must be represented by an accessible,
appropriately scoped source/evidence handoff rather than invisible state.

### Reconstituting the team

A cold-start agent begins with [STATUS.md](STATUS.md), the exact owner source,
this role table and the decision/conflict records. It discovers the current
federation from pins, manifests and registry edges; assigns the highest-value
closable tasks to available roles; uses isolated reversible work; records
handoffs; and converges on the same final G0-G11 run. The current bounded team observation is in [STATUS.md](STATUS.md#team-and-reconstitution).
The final baseline must record the actual team composition and evidence, not
only this role template.

## Required addendum tools: admission boundaries

The owner's compatible intent-locked addendum requires both an empirical
randomized/pathological falsification harness and a dynamic federation drift
crawler before final baseline lock. Discovery follows active root pins,
dependency manifests and workspace registry graphs, including topology changes.
Neither tool is admitted merely by this requirement or by a green badge.

- G0/G5/G6 require meaningful positive/adverse self-tests and source-bound
  mutation grading. Invalid executions cannot count as kills, and a
  truth-preserving mutation cannot be assigned an unjustified false verdict.
- New CLOSED claims require applicable empirical validation at the exact source
  in addition to kernel, axiom and statement evidence. Empirical success never
  proves a mathematical bound; decisive numerical violations require exact
  arithmetic or justified intervals. Missing or unsupported coverage is explicit.
- G7 includes contextual AST signatures, type definitions and theorem statements
  across declared federation relationships. Lexical equality alone does not
  establish semantic agreement; missing topology cannot silently pass.
- G8/G9/G11 require truthful coverage and limitations, documented runnable
  commands, self-verification and tool registration. The final G0-G11 run remains
  required; separate proposed gate numbers do not replace it.

Implementation and repair continue through the existing mission main#684 and
tool/component task records. This governance change records requirements, not
completed implementations, theorem closure or CBZ-0.

## Restored bounded replay controls

The four Python sources under
`cbz/evidence/kernel/additional/replay-tool/` are restored byte-for-byte from
public commit `260271793f0d53e4348390c5e8708f3280ac4964`, matching their
published SHA256SUMS. The existing README defines use, coverage and limitations;
the Python validator checks transcript consistency and authenticates no execution.

Run `python3.11 -B -S run_mutations.py --output /new/output` from that directory
using Python 3.11.16. The frozen suite contains 31 baseline tests; the unchanged runner requires both
baselines to pass and all three intended validator mutant kills in each
normal/optimized mode. The
`bounded-replay-python-controls` job in
`.github/workflows/cbz-p15-historical-readback.yml` installs the pinned runtime
and retains complete results on success or failure. The existing P15 job is
unchanged. This job covers the restored Python controls, not actual Lean replay,
whole-federation G0 or a mathematical proof. Actual Lean controls use the separate
documented harness and pinned Lean 4.34.1.

This CI binding resolves an observed local Python 3.9 unittest-header mismatch;
the failed local mutation-grader attempt remains evidence. No assertion or
expected verdict was weakened. Registration records how to verify the restored
source; passing qualification is recorded at its exact run/source, never inferred
from this text. Maintenance owner: none, self-contained. Serves criteria 6, 9,
11 within the documented bounded scope.
