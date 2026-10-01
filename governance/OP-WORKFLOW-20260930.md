# Working together with less overhead

Effective 30 September 2026 under the
[owner's delegation](OP-AUTONOMY-20260923-v2.1.md) and current explicit request
to improve the reader experience and remove workflow bottlenecks. This guide
replaces conflicting operational routing and repeated permission requirements.
Historical evidence and scientific acceptance conditions keep their meaning.
The [earlier stop](OWNER_STOP_20260927.md) is retained; this request resumes its
specified work, not stopped schedulers or other agents' background loops.

## Start with the task

Read the current source, the relevant existing PR and its latest substantive
findings. For a continuation, refresh the target's live head and ownership and
read changed or uncached dependencies. Reuse verified bytes by exact identity
and extraction rule. An old checkpoint requires bounded revalidation, not a
whole-workspace crawl. Read broader history only to resolve a concrete gap.

Keep [the public home](../docs/site/index.html) focused on understandable
mathematics and useful interactions. Contributor commands and detailed custody
records belong in [the workspace guide](../docs/WORKSPACE.md), with links from
the reader experience where they help a reader investigate a claim.

## One owner and one current disposition per object

Claim the exact scope and intended paths in the relevant existing PR or work
discussion; check current contenders before writing. If the task uses R17 Work
Events, link that claim ID in the same discussion. Existing lease ledgers and
handoffs are as-of views: an old or missing row does not erase a live peer claim.
Do not create another global ownership or scientific-status register.

The named integration owner controls branch refresh, readiness changes and merge
as well as the claimed edits. Other contributors use disjoint scopes or isolated
proposals. Mutate another owner's branch only after an explicit scoped handoff,
release, or confirmed lease expiry followed by current-state reconciliation.
Review work can proceed without taking over the writer's branch. Release a
finished claim. An offered task becomes active only when an actual pickup occurs.

Maintain one short current disposition in the PR body or a linked comment:
current head, owner, completed review scopes, unresolved finding IDs, validation
and next action. Update it on material change; link detailed evidence instead
of copying it. The [PR template](../.github/pull_request_template.md) supplies
this shape. Closed [main #95](https://github.com/d6g8k5htny-coder/main/issues/95)
is formal-rollout history; current work stays in its relevant PR. Neither issue
closure nor an outdated review thread resolves an outstanding obligation.

This is cooperative coordination. It does not install a distributed lock or
claim that every agent has acknowledged a notice. A competing mutation stops
the affected update and requires reconciliation, not a second competing write.

## Match verification to the change

| Change | Applicable local verification |
|---|---|
| Presentation and navigation | Links, relevant UI/control tests, keyboard behavior and changed layouts. Existing mathematical claims retain their scope. |
| Engineering, custody or workflow | Affected behavior and meaningful failure cases; reproduce a repaired defect against the original and amended source. |
| Mathematical statement, proof or dependency | Exact source identity, hypotheses, substantive review of the changed scope and every applicable scientific predicate. |

Mixed changes satisfy each affected row. These tiers select local work; they
do not skip required hosted checks or turn a proof edit into a cosmetic change.
Run focused checks while developing, then one complete applicable local run on
the frozen candidate. Repeat affected checks after changes, failures or newly
identified concerns; do not repeat an unchanged full run just to restate its
result, update a timestamp or produce another manifest.

Reuse a review only for its exact unchanged proof bytes, hypotheses and consumed
dependency slice. A base-only or unrelated change needs a bounded comparison of
that slice and fresh applicable integration checks, not a repeated mathematical
argument. Changed reviewed bytes, dependencies or scope invalidate the affected
review and require a successor disposition. Regenerating an inventory cannot
transfer an old review to new proof bytes.

Record actual author/reviewer exposure and separate technical verdict from
organizational independence. Same-provider review can be useful technical work
but earns zero organizational-independence credit. An author replay is not a
nonauthor review. Required analytic review, formal alignment and scientific
acceptance remain distinct from engineering checks and each other.

## Delegated owner review

On 30 September 2026, Dylan Roy explicitly authorized Codex to review on his
behalf, asked that the work be identifiable under his name for later inspection,
and reaffirmed autonomous execution with his retrospective direction. This
records that instruction; it does not claim he personally performed a review.

Use **Dylan Roy — delegated AI review** as the owner-facing attribution, followed
immediately by the actual performer, provider, agent and reviewed scope. The
review principal is Dylan Roy; an AI executor remains an AI executor. In any
machine-consumed review, `reviewer` and lineage fields identify the actual
performer. A separate principal annotation must never replace the executor with
`Human` or Dylan's name to satisfy an independence check. Authorization,
account ownership and the displayed name do not supply human-review evidence.

Record whether the entry is a fresh source review, an execution audit or a digest
of earlier reviews. A digest links the exact earlier sources and verdicts without
claiming they were rerun or expanding their scope. Historical authorship and
review records are not renamed. Same-provider or author exposure remains visible;
delegation itself earns zero organizational-independence credit and cannot fill
an independent-human-review or formal-alignment requirement.

Mark Dylan's personal reading **PENDING** until he supplies a response about the
identified material. This is an optional retrospective reading aid, not another
permission queue: authorized work and integration proceed when their applicable
evidence is ready. Later acknowledgements, corrections and scoped decisions link
his actual response; an acknowledgement alone is not proof acceptance. The
[review form](../.github/ISSUE_TEMPLATE/review-record.yml) records these distinctions.
Keep short digests beside the relevant reviews, with a pointer from the
[workspace guide](../docs/WORKSPACE.md); do not create another status register or
infer that an unread digest has been approved.

## Integrate once the evidence is ready

Immediately before merging, reread the current head/base, complete diff, latest
reviews, unresolved threads, ownership and required checks. Use
`expected_head_sha` for merge and a supported expected-old-ref guard for branch
updates; do not force past a mismatch. If another refresh produced the same
reviewed tree, compare its actual parents and affected source identities and
adopt it rather than publishing a duplicate refresh. Current integration checks
still bind to their actual tested commit.

Follow [the required formal checks](../docs/FORMAL_REQUIRED_CHECKS.md) and
[release operations](../docs/RELEASE_OPERATIONS.md). A base change can change
the tested merge; old CI is not rebound to it. Missing, skipped or failed checks
are not success. Unresolved engineering AMENDs survive peer merges. Record an
unexpected merge and its actual review chronology honestly, then repair or
reconcile it. Do not bypass protection or simulate another account's approval.

## Deliver one useful record

For occasional coordinated audits after substantial progress, use the
[milestone-audit protocol](OP-MILESTONE-AUDITS-20261001.md). It adds agreed
snapshots, recorded coverage and dependency-aware delta/full audit choices;
it does not trigger a freeze, full replay or scheduler after every pass.

Use one canonical delivery inventory, one immutable handoff per checkpoint and
one completion receipt. Derived summaries link these identities. Reference
predecessor deliveries by immutable ID/hash; copy their payloads only when needed
for a self-contained reproducibility package. Preserve exact source bytes,
historical failures, review scope and required original execution receipts.

Freeze the inventory after inputs stabilize and verify membership and bytes.
Use [the reusable delivery builder](../tools/research_delivery.py) with an explicit
file list and expected identities; its `build` and `verify` commands validate
archive membership and the canonical handoff without extracting or executing
payloads. Reuse existing tools where applicable; do not build a new delivery
framework for each continuation. When publishing to Drive, verify the uploaded
bytes and destination, register the bundle once in the existing R17 surfaces and
release the same claim. Scratch files and ordinary edits do not need separate
registrations or status banners. A later completion receipt may link an archived
handoff without rebuilding that archive to include its own upload receipt.

These are operating rules, not claims of new automation. No new scheduler or
enforcement service is installed by this document. The command-line delivery
helper runs only when explicitly invoked. Exact hashes
establish identity, tests establish their stated coverage, and a completed task
establishes its own acceptance test; none alone establishes a theorem.
