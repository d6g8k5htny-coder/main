# Coordinated milestone audits

Dylan requested occasional whole-program audits after substantial progress,
reusable coverage records and appropriately budgeted full audits. This extends
the [current workflow](OP-WORKFLOW-20260930.md). The
[coordination proposal](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5942391227)
asks the affected owners to choose the next stopping point. **This document
neither starts an audit nor freezes current work.**

The goal is a stable, inspectable account of what the identified program cut
establishes and what remains open. No audit guarantees that later findings will
be immaterial. Preserve earlier evidence and issue a successor when new facts
invalidate a dependency, review or conclusion.

## Milestones, depth and cost

Ordinary task verification continues. A pass, timer or commit count alone is
not an audit trigger. Coordinators propose an audit after substantive evidence:
a repaired dependency chain integrated, a result family with reviewed interfaces,
or a significant shared numerical/formal/custody capability. Cite actual evidence
and explain why the milestone justifies its cost.

- **Delta audit:** normally compare the preceding baseline with the proposed cut
  across the declared program. Cover changed/new/deleted work and the transitive
  affected dependency and consumer closure, including unchanged consumers whose
  inputs, hypotheses, tools or semantic interfaces changed. Compare both old and
  new graphs; justify where impact traversal stops. Equal top-level proof bytes
  or a path diff alone is insufficient. Unknown prior coverage is uncovered work.
- **Full audit:** a fresh end-to-end substantive examination of the declared
  current corpus and interfaces. Consider it after foundational model/statement
  changes, unreliable baselines, contradictory evidence, repeated escapes, major
  integration/publication, or accumulated cross-cutting changes. Record why delta
  coverage is insufficient. Reconsider at successive substantial milestones,
  without requiring an expensive full audit after every continuation.

Before work, agree reviewer effort, token/compute budget where measurable,
expensive replays, depth per area and a scope-reconsideration checkpoint. Report
actual costs when available; say unavailable otherwise. A budget limit means
staging or a partial report, never silently calling sampled work full coverage.
Reuse authenticated unchanged evidence when appropriate, distinguishing fresh
source review, fresh execution, historical execution and a digest. Unchanged
million-node runs need not be repeated merely to update a timestamp.

## Agree and freeze a specific cut

Use the existing coordination issue/PR and its current disposition. Identify
coordinator, actual active writers/integrators or custodians for each area, and
reviewers. Refresh live claims and handoffs; old rows and silence do not release
an active owner. Record the milestone condition, repositories/branches/paths and
source collections, exclusions, depth/budget, remaining work to finish or park,
and start/release/abort conditions. Open research can be parked with a finding
and fixed identity; an audit need not wait for every theorem to close.

Each affected owner must actually acknowledge the final scope, controlled refs,
bounded handoff and merge pause. Link each reply and identify the actual performer,
not just the shared account. Resolve competing claims first. An unattended area
needs an identified custodian who verifies its state. An offered assignment,
protocol review or unanswered notice is not agreement to a freeze.

After acknowledgments, capture the final commit/path/hash manifest and dependency
graph, confirm they match the agreed scope, then announce the exact frozen cut.
A changed scope/candidate requires affected owners' renewed agreement. Never
call partial acknowledgment a program-wide freeze.

During the agreed interval, **no in-scope merges, branch refreshes or source
mutations enter the audited cut**. Other work may continue on isolated branches
or queue without changing frozen inputs. An outside change to a consumed input
requires impact assessment. Unexpected drift/merge pauses the affected release
claim: preserve the snapshot, reconcile ownership, and agree a successor cut or
finish an explicitly historical audit. This is cooperative coordination, not a
technical lock, universal notification or restart of stopped loops.

## Bind the program and its sources

Use existing repository inventories, source maps and
[release custody tools](../docs/RELEASE_OPERATIONS.md#download-and-reproduce).
The collector's ten-repository allowlist is a discovery aid, not proof of whole
program coverage. Explicitly account for relevant research repositories, result
families, shared engines, formal layers and published consumers. Explain inactive
or excluded areas. Include authorized Drive/Dropbox/Library dependencies only
within existing custody/sharing permissions. Record access failures and missing
payloads; do not publish private material or widen access to finish the audit.

For each object bind repository/full commit, exact path/blob, bytes and SHA-256.
For non-Git material bind version/immutable identity where available, captured
byte hash and custody location. Bind consumed sections/extraction rules, hypotheses,
parameter domains and interfaces, original review bodies, logs, manifests and
configuration/toolchain identities. Distinguish source and execution-host commits.
Preserve failed runs and their amendments.

Reuse the existing dependency graph; an audit's evidence can contain a bounded
edge crosswalk without becoming another scientific register. Identify unmapped
edges and unresolved imports. Delta coverage follows impacted consumers and the
upstream premises needed to assess them; gaps encountered on that closure remain
explicit. An unchanged proof is reusable only at unchanged reviewed bytes,
hypotheses and consumed dependency scope.

## Review, findings and repairs

Allocate disjoint substantive scopes plus an integration/consistency read. Seek
fresh nonauthor/provider-distinct readers when useful. Record actual agent,
provider, authorship, prior review and shared-context exposure for each scope.
Author replays are not nonauthor reviews; same-provider review and shared accounts
earn no organizational-independence credit. Missing required human review or
formal statement alignment remains missing.

Each finding records exact source, affected consumers, severity, evidence or
counterexample, owner, disposition and repair verification. Separate source/custody
or missing coverage; mathematical statement/proof/hypothesis/dependency; numerical
certification or empirical limitations; formal kernel evidence versus statement
alignment; and engineering/integration/documentation defects. Keep scientific
acceptance distinct from all of these. Green CI, scalar Lean evidence, enclosure
arithmetic, a merge or audit-complete label cannot substitute for another predicate.
Existing scientific flags, explicit stops and HOLD/AMEND findings retain meaning.

Disposition each finding as repaired-and-verified, unresolved/blocking, deferred
with consequences, or inapplicable with evidence. Renaming a defect nonblocking
does not resolve it. State which conclusions cannot be relied on. Consequential
repairs receive a separate reviewer and affected-closure verification at successor
bytes; retain the original failure and review. Prepare repairs outside the frozen
cut, then obtain an acknowledged successor cut. Never silently amend old evidence.

## Release and next baseline

Coordinator and assigned reviewers reconcile agreed coverage, findings, affected
consumers, original execution receipts, lineage and final identities. Required
checks bind the actual release candidate/tested commit. Unchanged mathematical
reviews can carry forward only after documented dependency comparison.

A clean release requires complete agreed coverage, verified repairs, no unresolved
blocking finding for the declared use, and explicit limits for deferred/excluded
areas. Otherwise publish a **partial or blocked audit**, separating usable from
unusable conclusions. It can seed delta work only for evidenced coverage; gaps
and unresolved findings carry forward.

Record affected owners' release/abort acknowledgment before resuming paused
integration. Expiry is not implicit clean release: publish the partial state and
explicit abort/resume disposition under the agreed conditions. Name the next
baseline identity, inherited gaps and next milestone/risk for reconsideration.
Queued changes belong to subsequent work, not the frozen conclusion.

## One discoverable record per audit

Keep the report beside the audited work or in existing `audits/` or `reviews/`,
linked from the relevant current disposition and [workspace guide](../docs/WORKSPACE.md).
Use existing R17 surfaces where applicable. Add no global status register, scheduler
or per-pass checklist service. The report contains:

1. Audit identity/date, coordinator, trigger, delta/full depth, intended use and
   predecessor report/manifest hash, or explicitly no established predecessor
2. Scope/exclusions, owner acknowledgment links, frozen source identities,
   dependency crosswalk and drift checks
3. Coverage by object/section/domain/edge: fresh work, exact reused evidence and
   rationale, samples, exclusions and missing work
4. Actual reviewer lineage/exposure, linked verdicts and original execution
   receipts, including failed or unavailable checks
5. Findings, affected conclusions, dispositions and successor repair verification
6. Planned/actual cost and depth, limits, release/abort acknowledgment, next
   baseline and next milestone/full-audit reconsideration trigger

Use one canonical inventory and one immutable handoff with the existing
[delivery builder](../docs/RELEASE_OPERATIONS.md#compact-research-deliveries).
Verify membership and bytes and link one completion receipt. Reference predecessor
IDs/hashes instead of duplicating payloads unnecessarily; private evidence stays
private. A later receipt need not rebuild the archive to include itself.

The existing [fixed-target completion audit](../docs/research-translation/20260930/COMPLETION_AUDIT.md)
and [source-recovery audit](../docs/public-math/recovery-20260927/LIBRARY_PUBLICATION_AND_ORIGIN_AUDIT.md)
retain their bounded scopes. Neither becomes a whole-program baseline by citation.
Hashes establish identity, not proof correctness or permanent finality.
