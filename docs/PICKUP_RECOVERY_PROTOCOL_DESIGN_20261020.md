# Pickup recovery: a protocol design and a fence test (20 October 2026)

Status: a design note and one test, written as the isolated design stage that the
coordinator's disposition
[#275/6281287486](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6281287486)
named for proposal
[#275/6276723458](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6276723458).
It changes no rule, activates nothing and releases nobody's claim. The
[current workflow](../governance/OP-WORKFLOW-20260930.md) governs, including the
coordinator-fallback section landed by
[main#450](https://github.com/d6g8k5htny-coder/main/pull/450) and the usage-limit route landed
by [main#453](https://github.com/d6g8k5htny-coder/main/pull/453); this note changes neither.
Author: Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), as delegated
AI work for Dylan Roy. Scientific effect NONE.

## 1. The problem, with its evidence

A pickup is a comment that names a performer, an exact source and a scope. Some performers
are stateless hourly invocations under a shared label (for example the xAI workers
GROK-H01 to H04). When such an invocation ends, its pickup keeps its claim, but no execution
holds it any longer, and the next invocation under the same label states, correctly, that it
holds no prior assignment. Five cases between 15 and 19 October:

| Pickup | Scope | Claimed | Outcome |
|---|---|---|---|
| [389/6188676525](https://github.com/d6g8k5htny-coder/main/pull/389#issuecomment-6188676525) | main#389 main text, slice M1 | 15 Oct 09:15Z | no delivery; ended by owner authority on 19 Oct 21:53Z |
| [389/6191845893](https://github.com/d6g8k5htny-coder/main/pull/389#issuecomment-6191845893) | main#389 main text, slice M2 | 15 Oct 12:17Z | same |
| [389/6182815146](https://github.com/d6g8k5htny-coder/main/pull/389#issuecomment-6182815146) | main#389 supplement draft 9 | 15 Oct 02:11Z | same |
| [389/6184901584](https://github.com/d6g8k5htny-coder/main/pull/389#issuecomment-6184901584) | main#389 supplement draft 9 | 15 Oct 05:08Z | same |
| [419/6248600576](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6248600576) | main#447 source review | 18 Oct 10:09Z | open as of 20 Oct 02:00Z |

The cost: the main#389 head was frozen for four days under the practice that an open read
freezes the head, and main#447's mathematical review could not start because other lanes
deferred to the pickup. The resolution of the first four needed a direct question to the
owner ([389/6278216270](https://github.com/d6g8k5htny-coder/main/pull/389#issuecomment-6278216270)).

## 2. What the coordinator corrected

The proposal's four clauses are withdrawn as written. The design below keeps these
constraints from the disposition:

1. A later invocation under a shared label is not the earlier performer. Its statement that
   it holds nothing neither releases the earlier claim nor proves that the earlier execution
   ended. Historical reassignment is recorded as authorized supersession or recovery, with
   the predecessor's closeout preserved as UNKNOWN; it is never described as the predecessor's
   own release.
2. A release-on-exit term is usable only when the exact invocation and a trustworthy
   termination event are observable; otherwise it is an unfulfilled condition, not a
   timeout. An unavailable executor is not known to be dead.
3. A new invocation reconciles the unresolved earlier claims it could conflict with, but does
   not release them as its own. A lookback window is a convenience, not a cutoff.
4. A claim binds an exact execution identity, a canonical task and scope, and a generation.
   Recovery requires explicit applicable authority, the successor's personal acceptance and a
   current-state reconciliation. An expected-head guard detects stale source; it does not by
   itself prevent every stale write, and any fencing claim needs technical enforcement and a
   test.
5. Immutable pinned-source reviews are preferred to freezing an author's whole branch. A
   review does not carry to new bytes without an exact-source or composition assessment. An
   existing explicit freeze stands until its owner or an authorized disposition resolves it.

## 3. What is technically prevented, and what is only discouraged

`tests/test_pickup_fence.py` builds a bare repository, lets a second worker (B) move a
branch, and records which pushes by the first worker (A) succeed, from a stale view and from
a fresh one. The three lease cases of the nonauthor read
[456/6281763561](https://github.com/d6g8k5htny-coder/main/pull/456#issuecomment-6281763561)
are included. Standard library and `git` only; it runs in about a second.

| Write by A | Result | Enforced by |
|---|---|---|
| stale view: `git push origin work` after the branch moved | rejected (`[rejected] ... fetch first`) | the transport's fast-forward rule |
| stale view: `git push --force-with-lease=refs/heads/work:<old>` after the branch moved | rejected (`stale info`) | the explicit lease: the remote ref must equal the stated old value |
| stale view: `git push --force-with-lease origin work`, no expected value, after a fetch that A never integrated | **accepted**, B's commit is overwritten | nothing: the implicit lease checks the remote-tracking ref, which the fetch already moved |
| the same push with `--force-if-includes` | rejected | the implicit lease with `--force-if-includes`: A's view must contain the remote ref |
| fresh view: `git push --force-with-lease=refs/heads/work:<current>`, a fast-forward | accepted | the lease matches; the push would also pass without it, being a fast-forward |
| fresh view: the same explicit current lease, a deliberate non-fast-forward rewrite of B's commit | **accepted**, B's commit is replaced | nothing: the lease fences stale writers, not deliberate conflicting writes |
| `git push --force origin work` | **accepted**, B's commit is overwritten | nothing |
| B records a claim on the branch; A pushes anyway | prevents nothing | nothing |

So a guarded push (the `ls-remote`, then `--force-with-lease=branch:old`, then `ls-remote`
pattern already used on main#389) prevents a stale worker from overwriting a moved branch,
provided every worker uses it, and uses the explicit form
`--force-with-lease=<ref>:<verified old head>` or the implicit form together with
`--force-if-includes`. The implicit form alone fences nothing once a fetch has moved the
remote-tracking ref, which a background fetch does without the worker's knowledge. And the
lease fences stale writers only: a worker with a fresh view who deliberately rewrites the
other worker's commit, naming the current head in the lease, is not stopped. That is a
conflicting write the lease does not prevent; on an unprotected branch it remains discouraged
only, like a bare `--force`, and only blocking force pushes prevents it. A claim record
prevents nothing. On GitHub, branch protection can forbid force pushes altogether, which turns
the deliberate rewrite and the bare `--force` from "discouraged" into "prevented" without any
worker's cooperation; the merge endpoint accepts a `sha` parameter (the expected head) that
fails the merge when the head has moved; and the update-branch endpoint accepts
`expected_head_sha` in the same way. None of the GitHub guards is exercised by the test, which
has no remote; they are stated from the documentation.

What no transport guard prevents: two performers reviewing the same pinned source (duplicate
work, not a conflicting write), and a successor acting on a scope whose predecessor is not
known to have stopped (a custody conflict, not a byte conflict). Those are protocol matters.

## 4. The six scenarios

In each scenario, A is the pickup's performer, B is a later actor, and the source is pinned by
an exact commit in the pickup.

**S1. Simultaneous claimants.** A and B post pickups for the same scope within minutes, each
after a contender check that did not see the other. Records: two pickups, both naming the
head. Who acts: by the workflow, the earlier pickup holds the scope; the later one withdraws
in the task thread, or the two split the scope explicitly there. Prevented: nothing at the
transport level, since neither has written. Discouraged: continuing both. Design: the pickup
states its contender-check time; the later claimant, on seeing the earlier record, posts a
withdrawal or a split before any write. A reviewer's delivery that overlaps an earlier
pickup is still a delivery; the coordinator decides which review binds.

**S2. The original worker returns after supersession.** A's pickup was superseded by an
authorized recovery record (as the four main#389 reservations were), and a later execution
under A's label returns with partial work. Records: the pickup, the supersession, the return.
Who acts: the returning execution is not known to be A (correction 1), so it may post its
partial work as data and may post a fresh pickup, but it does not resume A's claim. Prevented:
a push to a branch whose head moved, if guarded. Discouraged: treating the return as A's
continuation. Design: a returning execution states that its continuity is unverified, links
A's record, and takes a fresh pickup only if the scope is still open.

**S3. A delayed callback across expiry.** A's delivery arrives after a recorded supersession,
for the superseded head. Records: supersession, then A's delivery. Who acts: the delivery
stands as a review of the head it names (correction 5: a review binds pinned bytes); if the
successor's work is on the same head, the coordinator records both and decides which binds;
if the head moved, A's delivery is a review of old bytes and carries forward only with an
exact-source or composition assessment. Prevented: nothing (a comment). Design: every
delivery names the exact head it checked; a delivery for a superseded head is not rejected,
and is not promoted either.

**S4. A crash before release.** A stops without a delivery, checkpoint or release; nothing
further ever appears. Records: the pickup only. Who acts: no one may release A (corrections
1–2). The coordinator, or the owner, or after an unanswered 24-hour request the peer fallback
of section 5 point 4, records a supersession after a reconciliation, which the successor
accepts personally (correction 4); until then the scope is held. Prevented: a
successor's stale push, if guarded. Design: a pickup carries a checkpoint time; a missed
checkpoint is the trigger for the coordinator's reconciliation, not for release. The
proposal's "release-on-exit" is admissible only where the platform reports termination, which
the hourly invocations do not; for them the trigger is the missed checkpoint.

**S5. Ambiguous claim order.** Two pickups, each citing a contender check that predates the
other's post, with timestamps seconds apart or under clock skew. Records: the two posts with
server timestamps, which have one-second resolution, so a tie is possible. Who acts: when
the server timestamps order the posts, the later poster withdraws, as in S1; when which
record is canonical, or their order, is ambiguous or tied, both actors stop the affected
mutation and reconcile in the existing thread, and the reconciliation does not reassign
claims already released (the [current workflow](../governance/OP-WORKFLOW-20260930.md),
"Four checks bracket every claim", item 2). Design: as S1, with the tie rule; the contender check is re-run
immediately after posting, which detects the overlap. It does not close the window in which
both were posted, and nothing does; the tie rule is what handles it.

**S6. Stale-head mutation.** B refreshes or amends a branch whose pickup names an earlier
head; A, with a stale view, tries to push. Records: the pickup names the head; the branch
moved. Prevented: A's push, if A uses a guarded push in the explicit lease form or if force
pushes are forbidden on the branch (section 3). Discouraged only: A's bare `--force`, and a
deliberate rewrite of B's refresh from a fresh view. Design: guarded pushes for every
shared branch, and branch protection against force pushes where the host allows it; the
pickup's head is what a review binds, so B's refresh does not invalidate A's review of the
old head, it makes it a review of old bytes (correction 5).

## 5. A protocol consistent with the corrections

1. **Identity.** A pickup names the execution (session or invocation identifier), not only a
   label, and its generation under that label where the platform reuses labels. Two executions
   under one label are two performers.
2. **Checkpoint, not lease-expiry.** A pickup states a checkpoint time. A missed checkpoint is
   a trigger for the coordinator to reconcile; it is not a release, and no performer other
   than the coordinator, the owner or the peer fallback of point 4 records the supersession.
3. **Reconciliation on start.** An invocation lists, before claiming, the open pickups under
   its label from any time, and says for each whether it is the same execution (continuing),
   has the work (returning, continuity unverified) or neither. It releases none of them.
4. **Supersession record.** Authority, successor acceptance, and a current-state check, in
   one record that preserves the predecessor's closeout as UNKNOWN. The successor's pickup
   links it. The authority is the coordinator or the owner; or, exactly as the
   [current workflow](../governance/OP-WORKFLOW-20260930.md) ("The confirmations above assume
   a coordinator whose record reaches the board") provides, a peer, once the specific
   confirmation has been requested in the task thread and has gone unanswered for 24 hours
   from that request: an eligible agent of a different provider from the acting worker, who is
   neither the object's author, predecessor nor receiver, has not reviewed it, reads the same
   current refs, claims, findings and in-flight work, records what it read and that it concurs,
   names the record a fallback confirmation, states which condition applied, and mirrors it on
   the board. The fallback covers ordinary engineering, documentation, review and integration
   stages only; it cannot lift an owner stop, cure a platform or permission denial, change
   protections or supply scientific acceptance; a coordinator record posted later supersedes it
   for anything not yet landed; and it earns no independence credit. Nothing here widens or
   narrows that section.
5. **Reviews bind pinned sources.** A read names the exact commit; the author's branch is not
   frozen by an open read. A later revision does not inherit the read; it gets a byte-diff
   readback against the read head, or a fresh read, as today. A freeze is explicit, owned and
   lifted by its owner.
6. **Guarded writes.** Every push to a shared branch uses the explicit lease
   `--force-with-lease=<ref>:<verified old head>`, or the implicit lease together with
   `--force-if-includes`, and force pushes are forbidden on protected branches. The host's own
   writes carry the same guard: on GitHub, the merge endpoint's `sha` and the update-branch
   endpoint's `expected_head_sha` fail the write when the head has moved. The lease and the
   API guards prevent a stale write; only the prohibition of force pushes prevents a
   deliberate conflicting write from a fresh view (section 3). Everything else in the protocol
   discourages.

Point 5 is the one that would have cut the head freeze of the main#389 case from four days to
zero: the author's later revisions would have gone out as new pinned heads, each with its own
read, and the open reads would have remained reads of the heads they named. It would not have
shortened the stalled stage of section 6: the four reads would still have had no delivery
until the supersession, about four days.

## 6. Measurement plan

All measures come from board and PR records, which carry server timestamps.

- **Stalled-stage duration:** for each pickup, the time from its post to its terminal event,
  which is the first of: its delivery; the supersession record; the performer's own release or
  withdrawal (the S1 and S5 withdrawals included), with the reason recorded. A pickup with no
  terminal event at the measurement time is right-censored there: it enters the count of open
  pickups with its age, and not the distribution. Reported as a distribution over terminated
  pickups, the count above 24 hours, and the open count with ages. The five cases above give
  4 days, 4 days, 4 days, 4 days (terminal event: the owner-authorized end) and, for the fifth,
  censored at more than 40 hours.
- **Duplicate work:** the number of deliveries for one pinned source and scope by more than
  one performer without a recorded split.
- **Owner interventions:** the number of supersessions or releases that needed the owner
  rather than the coordinator. The five cases needed one owner intervention for four of them.
- **Safety requirement:** zero successful conflicting writes, counted as pushes that replaced
  a commit another performer had pushed to the same branch without a recorded handoff. With
  branch protection against force pushes that applies to administrators and to every other
  bypass actor (no force-push bypass list), this count is enforced to zero on the protected
  branches; a protection that exempts administrators or a bypass role enforces it only for the
  actors it binds, and the exempt actors' pushes are audited like any other. On this
  repository every lane pushes through one shared principal, so the exemption question is
  decided once, for that principal. With guarded pushes alone the count is enforced against
  stale writers only (section 3).
  On every branch, protected or not, and on branches since deleted, it is audited from the
  host's repository activity record,
  `GET /repos/{owner}/{repo}/activity?activity_type=force_push`, which records each force push
  with its before and after commits; the read
  [456/6281763561](https://github.com/d6g8k5htny-coder/main/pull/456#issuecomment-6281763561)
  found such records in this repository for an unprotected branch and for a deleted one. How
  long the host retains them was not established, so the count is taken during each
  measurement window, not only at its end.

A comparison runs over two windows, before and after adoption, on the same set of
coordinating lanes. No live lease-expiry experiment is part of this plan.

## 7. Reproduce

    python3 -m unittest tests.test_pickup_fence -v
    python3 -O -m unittest tests.test_pickup_fence

Both runs pass (eight tests) on a checkout with `git` on the path; the test writes only inside
a temporary directory.
