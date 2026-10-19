# Working together with less overhead

Effective 30 September 2026 under the
[owner's delegation](OP-AUTONOMY-20260923-v2.1.md) and current explicit request
to improve the reader experience and remove workflow bottlenecks. This guide
replaces conflicting operational routing and repeated permission requirements.
Historical evidence and scientific acceptance conditions keep their meaning.
The [earlier stop](OWNER_STOP_20260927.md) is no longer in effect by the
[owner decision of 5 October](OWNER_DECISION_20261005_CURSOR.md); its record stays
unchanged, and Cursor agents work on request.

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

For already-authorized, low-impact work with no live owner, an eligible worker
may begin the smallest disjoint stage after reading current claims and publishing
an actual task-local pickup naming the source, scope, exclusions, checks and a
finite checkpoint. Availability notes, offers and shared agent labels are not
pickups; such routine work needs no extra coordinator acknowledgement unless a
specific task or existing rule requires one. This does not authorize an occupied
role, private-access expansion, security-sensitive change, protected integration
or an action stopped by a denial or unresolved finding.

A coordinator may prospectively authorize a named receiver's exact bounded
stage in the existing task discussion. Only when that record expressly permits
pickup to activate that ordinary stage may the receiver personally accept and
pick it up after fresh source and contention checks without another routine ACK.
An offer or the prospective record alone permits no mutation. Unresolved prior
custody or transfer of an occupied reservation still requires separate receiver
acknowledgement, fresh coordinator reconciliation and explicit activation as
described below; silence, expiry and shared identities never supply release.

For an unresolved integration reservation, consult the case-specific prospective
reassignment implemented in [#276’s amendment](https://github.com/d6g8k5htny-coder/main/pull/276#issuecomment-6038812718)
and [activation](https://github.com/d6g8k5htny-coder/main/pull/276#issuecomment-6038906820).
That precedent records the authority and exact future scope affected in the existing
PR, identifies an eligible actual receiver and finite checkpoint, and activates only
after the receiver’s personal acknowledgement and a fresh coordinator confirmation
following reconciliation of current refs, claims, findings and reported in-flight
work. The proposal alone permits no mutation. Preserve the predecessor’s actual
work, attribution and own release as UNKNOWN wherever unresolved; age, silence or
a shared alias proves neither expiry nor release. Crossed work, competing claims,
explicit denials or unresolved findings stop the affected action for reconciliation.
This pointer supplies no blanket reassignment, unseen-claim override, security or
check waiver, or scientific acceptance; the existing source-preservation and
protected-integration requirements still apply.

The same steps apply whenever a session stops with any role unfinished (author
custody, a review pickup, a planning phase or an integration reservation), including
an owner-directed account or model switch. The successor claim reuses the existing
task and claim identifiers, names the actual successor session or invocation, and
is cross-linked to the predecessor's claim with the predecessor's closeout kept
UNKNOWN until its own release is found. A successor's checkpoint keeps the handoff's
artifact list and identities and records, per artifact, what the successor can
actually reach: published with its identity; recoverable as verified bytes through an
authorized route (those bytes are reused, matched by identity, not regenerated);
unavailable to the successor (not presumed destroyed; its existence stays UNKNOWN
until found); or unknown. Reconstruction happens only where it is necessary and
authorized, and a reconstructed artifact or execution is a new candidate under the
successor's provenance, never the predecessor's. Capability is observed, never
inferred from the account or model family: an existing read-only observation or the
least consequential relevant check comes first, an already-authorized necessary write
establishes its own outcome, no write is made only to demonstrate capability, and a
successful read is not evidence of write capability; the observation is restated when
it changes. Three situations are kept apart: a technically unavailable tool or
platform (labeled as such; retry only when restored, and say so), an explicit denial
(stop the affected action; an owner-imposed stop is resolved only by the owner's
recorded decision, a permission or security denial only by a supported platform
clearance or the required handoff to the user, and an owner decision does not stand
in for that clearance; the denied step is never routed to another executor, a new
session is not clearance, and neither a restored tool nor an already-permitted
alternative route is leave to evade the denial), and a lane whose usage is exhausted
(the coordinator reassigns; closeout UNKNOWN). Each successor's personal
acknowledgement and activation stand alone; an unidentified second successor does not
delay the first. Retained stops are carried by name. None of this adds a register, a
scheduler, a permission gate, a forced release or independence credit.

The confirmations above assume a coordinator who answers. When no coordinator
or Chief-of-Staff record has appeared on the current board for 24 hours, an
ordinary stage that waits only for a routine coordinator confirmation or
activation may be confirmed by a peer instead. The peer is an eligible agent of
a different provider from the acting worker, is neither the object's author,
predecessor nor receiver, has not reviewed it, and reads the same current refs,
claims, findings and reported in-flight work; it records in the task thread
what it read and that it concurs, and the record names itself a fallback
confirmation. The receiver's own acknowledgement, fresh source and contention
checks, the predecessor's UNKNOWN closeout and its preserved work and
attribution are unchanged.

The fallback covers ordinary engineering, documentation, review and integration
stages only. It cannot approve a manual review route, lift an owner stop, cure a
platform or permission denial, change protections or private access, activate a
security-sensitive stage or supply scientific acceptance; those keep their
existing owner or coordinator steps. A coordinator record posted later
supersedes a fallback confirmation for anything not yet landed. The fallback
repairs an absent confirmer; it is not a second coordinator and earns no
independence credit.

Four checks bracket every claim, mutation and verdict:

1. Before claiming, check whether the requested output has already landed on
   the default branch; if it has, do not reselect the work.
2. Right after claiming, and again before substantive work, reread the
   task-local and board claims and stop on overlap. There is no fixed wait.
   Each claim has one canonical pickup record, named in the claim; board or
   task-local mirrors cross-link it and carry no priority of their own.
   "Earliest created_at wins" compares canonical records only, applies only to
   genuinely equivalent scopes, after the already-landed check; a claimant
   withdraws only its own claim. If which record is canonical, or their order,
   is ambiguous or tied, both actors stop the affected mutation and reconcile
   in the existing thread; this does not reassign claims already released.
3. Before mutating, recheck current state and use an expected-head guard. For
   unversioned configuration such as automation prompts, use one serial writer
   with a readback after each write.
4. Before posting a verdict, reread the native review collection and stop if an
   equivalent review already exists.

A lease end is a checkpoint, not proof of release: an unknown predecessor goes
through explicit reassignment, successor acknowledgement and fresh activation,
with its closeout kept UNKNOWN. Reuse an existing collision notice where
possible. If none exists, or it cannot reach the affected actors, post one
source-linked warning containing materially new conflict evidence; avoid
duplicate notices. Review
carry-over to a new head also needs unchanged hypotheses and consumed
dependencies, the current tested composition and checks, and the actual
Ready-triggered review, not just matching blobs. The checklist records the
reviewed head, the PR-reported base and the actually tested merge commit and its
parents as separate identities, never inferred from a green badge. A resolved
thread is UI state, not a finding disposition; AMEND history and successor
reviews keep both identities.

Maintain one short current disposition in the PR body or a linked comment:
current head, owner, completed review scopes, unresolved finding IDs, validation
and next action. Update it on material change; link detailed evidence instead
of copying it. The [PR template](../.github/pull_request_template.md) supplies
this shape. Closed [main #95](https://github.com/d6g8k5htny-coder/main/issues/95)
is formal-rollout history; current work stays in its relevant PR. Neither issue
closure nor an outdated review thread resolves an outstanding obligation.

At source pickup, run a complete by-path contender check of the paths you will
touch against open pull requests; shared generated files and consumed
dependencies count as overlap. Before any mutation, revalidate changed or new
pull request heads. An exact-head, complete, paginated inventory may be reused,
refreshing only changed entries and pull requests merged since the candidate
base; a full repository recrawl is not required before every small docs edit.
Link a compact overlap record from the pull request instead of copying a long
list. Unresolved ownership still stops the affected write.

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

Blind reconstruction, adversarial attack and derived evidence profiles follow
the [closure-evidence protocol](OP-CLOSURE-EVIDENCE-20261006.md).

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

Marking a pull request ready can start an automatic review. Wait for that review
to finish and reconcile its findings before merging; an earlier automatic review
of the same head does not replace it. If it cannot finish, say so on the pull
request and settle the next step there instead of merging past it.

Where `public-intake` is required, verify its actual trusted workflow and event
identity and its current head, as CONTRIBUTING requires, as well as its
success. A check name and head alone cannot establish the Ready-event binding.
On an ordinary pending-required-check 405, re-read current refs, ownership, and
protection and check state. One retry at the same expected head is appropriate
only after the missing requirement has actually succeeded and no other stop
applies. This does not waive the separate Ready-triggered review, approve
security changes, or authorize retrying platform denials.

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

Before publishing or registering a delivery, apply the
[destination-audience preflight](../docs/RELEASE_OPERATIONS.md#destination-audience-preflight).
Verify each receiving destination separately, including the register or message
that receives metadata. A private archive does not establish a private register.
Check outgoing fields against that destination's current audience and existing
authorization. An unknown or unverifiable audience pauses only the affected
write; continue unrelated work.

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
