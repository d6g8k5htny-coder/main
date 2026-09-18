# `engine/bridge/` — the repository-side half of a PROPOSED, NOT DEPLOYED contract

On 2026-09-18 a Drive handoff (`drive/deltas/2026-09-18/DG-EXEC-20260918-49291487/`,
mirrored byte-exact, disposition **"PROPOSED EXECUTION CONTRACT / NOT DEPLOYED"**)
described how Drive-led, GitHub-executed research would work: a Drive work order
→ source and authorization checks → an isolated branch → a bounded, reproducible
run → an exact receipt → technical review → Drive acceptance. This directory
holds the two record shapes that loop needs on the git side, their validators,
and a checker. It holds no work orders and no receipts, because the contract is
not deployed. Committing this code does not deploy it.

> **Under the contract, work proceeds only under an explicitly authorized work
> order. The authorization is the owner-side Drive record the order names; this
> repository pins that record, cannot verify it, and confers nothing. An order
> it holds is PREPARED-only until the owner boundary says otherwise.** A
> receipt is a record that something ran, not evidence that it ran correctly
> and not a verdict. Nothing in this directory moves a claim, premise,
> obligation, gate or grade. The five validity premises of Theorem D1 v2.2(2)
> are OPEN, `D3-LEMMA-RN-UNIF` is not closed, and no record here can say
> otherwise.

## What is here

```
work_order.py     q0.bridge.work_order/v1  — the record and its validator (no writer)
run_receipt.py    q0.bridge.run_receipt/v1 — the record, its validator, the append-only store
common.py         canonical JSON, digests, identifier shapes, the strict loader, shared by both
orders/           committed work orders, named <task_id>.json        (empty)
receipts/         stored run receipts, named <idempotency_key>.json  (empty)
examples/         one example of each, record_kind EXAMPLE, authorizing nothing
```

```bash
python3 tools/bridge_check.py                  # validate orders, receipts, examples; non-zero on any problem
python3 -m pytest -q tests/test_bridge.py      # the negative controls
```

## What this does not do

- **No enforcement.** Nothing here protects a branch, restricts a push, checks a
  permission or runs a forbidden-path test. A validator that returns an empty
  list has found a well-formed file, and that is all.
- **No branch protection, no rulesets, no least-privilege CI settings.** Those
  are server-side administration by the owner (contract §4, §7 step 5). None of
  it is performed here; the only CI change is one added step that runs
  `tools/bridge_check.py` on the (empty) record directories.
- **No credentials.** No token, key or Drive credential is read, stored or
  expected by this code. A declared provider, model or session id in a receipt
  is attribution metadata; only `actor.authenticated_principal_id` names an
  authenticated identity, and this code does not authenticate it.
- **No verification of anything owner-side.** `VERIFIED_BY_OWNER_BOUNDARY` is
  transcribed from the Drive record an order names; the checker prints it as a
  NOTE and cannot check that the record exists or says that.
- **No acceptance.** `ACCEPTED` is an owner-side transaction in Drive. This
  repository cannot verify an acceptance: a receipt claiming it is refused
  unless it names a Drive-id-shaped record (id and SHA-256), and with one the
  claim is transcribed unverified and printed as a NOTE. Whether that record
  exists and accepted anything is checked at the owner boundary, not here.
- **No status change.** `scientific_status_change_authorized` on an order is
  always `false`; `scientific_status_change` on a receipt is always
  `UNCHANGED`. Any other value is refused (FW-NO-RECEIPT-PROMOTION). A merge is
  not a promotion; a green run is a run.
- **No competing task queue.** The R17 Work Events register
  (`registers/json/work_events.json`, append-only) remains the claim and
  disposition log. An order's `claim.work_events_event_id` points into it; it
  does not replace it.
- **No independence.** `review.independence_credit` is 0 unless an
  organizational-independence record id is given (a Drive-id-shaped id or a
  `reviews/` `REV-…` id; never a placeholder, never a "same-provider" anything,
  never on a `NOT_REVIEWED` verdict), and even then the validator records it;
  it awards nothing, and the independence-requiring gate stays open.

## The state model

A receipt's `status` is one of six words:

| status | meaning | what it requires |
|---|---|---|
| `NOT_RUN` | nothing executed; a template is not a receipt | — |
| `CANNOT_VERIFY` | something may have run, the evidence to say what is missing | — |
| `PREPARED` | order resolved and capsule assembled; no command ran | — |
| `EXECUTED` | commands ran at a named commit | `repository_id`, `commit_sha`, `tree_sha`, `dirty_worktree` (a boolean), `environment_identity`, real `commands` (no `:`, no blank, no placeholder), one exit code per command, test counts that count ≥ 1 test, ≥ 1 negative control that is a statement, a coverage statement, an authenticated principal, start/end with end ≥ start |
| `RECORDED` | receipt and evidence reached Drive **and were read back** | everything above plus `delivery_id`, `receipt_file_id`, `readback_sha256`, `readback_utc` |
| `ACCEPTED` | an owner-side acceptance is *claimed*, naming a Drive record | everything above plus `accepted_by_drive_record {drive_id, sha256}`; never verified here, transcribed as a NOTE |

Missing evidence is `NOT_RUN` or `CANNOT_VERIFY`, never PASS: a receipt with
`tests_passed: null` or `tests_passed: 0` cannot carry `PASS_TECHNICAL`; a
receipt with a nonzero exit code cannot claim `tests_failed: 0`; a receipt
that did not execute, or whose worktree was dirty, cannot pass; a receipt whose
commands, negative controls, principal, environment or coverage are
placeholders (`n/a`, `none`, `unknown`, `-`, `:`, a single character) is not
an execution. An execution can succeed while archival is pending (`EXECUTED`
without a Drive return), and a recorded candidate can still need review.

A work order's `status` is `PREPARED` or `EXECUTABLE`. `EXECUTABLE` requires
`authorization.verification_status: VERIFIED_BY_OWNER_BOUNDARY`, a Drive id in
`authorizing_record_drive_id`, a Drive id or Drive URL in `source_ref`, a
pinned base commit, at least one governing source with all five identity
fields, a Work Events claim and lease, a non-empty scope with allowed
commands, acceptance tests and required negative controls, and all three
limits. `NOT_VERIFIED` is PREPARED-only. An order must not be created or
edited by the PR it authorizes: `authorized_outside_this_repository` must be
`true`, and a reference that names `engine/bridge/orders/` in any spelling, any
top-level path of this repository, a file name, a GitHub URL or a pull request
is refused as the order's authority.

Scope paths are literal repository-relative paths, compared casefolded: an
absolute path, `~`, `..`, a glob, an empty segment or a control character is
refused, not normalized. Two surface lists apply:

- **Protected, unconditionally**: `engine/bridge/` (all of it), `engine/lanes/`,
  `engine/receipts/`, `drive/`, `tools/`, `tests/test_bridge.py`, `quarantine/`,
  `CLAUDE.md`, `AGENTS.md`. No reference unlocks these: an order that could
  authorize writing an order, a receipt, the checker or its negative controls
  could authorize itself.
- **Policy/status, conditionally**: `.github/`, `governance/`, `registers/`,
  `claims/graph.json`, `docs/OPEN_PROBLEMS.md`. Refused unless
  `public_disclosure.approved` and a `policy_change_authorized_by` Drive id are
  both present, and then recorded, not granted.

`allowed_commands` may not contain a git or gh mutation (`git push`, `git
commit`, …) or a command that lexically writes into a surface (a redirect,
`open(`, `tee`, `cp`, `sed -i`, … preceding a surface path). That is a
heuristic that can only refuse; it does not establish that a command is safe.

## Receipt against order

`tools/bridge_check.py` ties every receipt to the order its digest names: same
`task_id` and `repository_id`; every executed command present verbatim in the
order's `allowed_commands`; every output path within the order's
`allowed_paths` and on no surface; and no `EXECUTED`/`RECORDED`/`ACCEPTED`
receipt against an order that is not `EXECUTABLE` and
`VERIFIED_BY_OWNER_BOUNDARY` — a receipt that says something ran under a
PREPARED-only order records a run the contract did not permit, and the checker
fails on it. Agreement is consistency, not evidence: a receipt that agrees with
its order is still not evidence that the run was correct.

## The idempotency key

```
idempotency_key = sha256( canonical_json([task_id, work_order_digest, tested_commit, run_id]) )
```

`tested_commit` is `execution.commit_sha` (`null` when none). The receipt file
is `receipts/<idempotency_key>.json`, created with `open(path, "x")` and never
opened for writing again. An identical redelivery writes nothing. A
*conflicting* redelivery is held beside the first as
`<key>.NEEDS_RECONCILIATION.<held-digest-prefix>.json`, a
`q0.bridge.held_delivery/v1` wrapper carrying the conflicting receipt verbatim.
The first receipt is never replaced; reconciliation is an owner-side task the
wrapper does not perform. `work_order_digest` is the SHA-256 of an order's
canonical body; `body_sha256` is the same for a receipt; both are recomputed by
the checker so an edit after the fact is a schema violation.

The store writes to exactly one place inside this repository, `receipts/`;
any other destination must lie entirely outside the repository (tests use
temporary directories), and `orders/`, `examples/`, `engine/lanes/`,
`engine/receipts/`, `claims/`, `registers/`, `drive/`, `governance/`, `docs/`
and `.github/` are refused by name as well.

## Frozen in history, canonical on disk

A digest establishes a record's identity; git history is its freeze. The
checker fails on any `.json` under `orders/`, `receipts/` or `examples/` that
any commit reachable from HEAD modified, deleted or type-changed
(`git log --diff-filter=MDT`), so a rewrite that was committed is caught
exactly as an uncommitted one is; the working-tree-versus-HEAD comparison
remains as a local convenience. A record that must change gets a successor
under a new name, never an edit. History that was force-pushed away is not
history this checker can read; that is what server-side protection, which
this repository does not hold, is for.

The bytes on disk are the record: a file whose text is not the canonical
serialisation (sorted keys, two-space indent, trailing newline) of what it
parses to, a file with a duplicate key (the first `"status"` in the text is
what another reader sees) or with NaN, a file with any extension but lowercase
`.json`, a subdirectory, or any file other than `README.md` and records in the
live directories is a problem, never skipped.

## How a receipt reaches Drive

Not through this code. The contract (§6) separates the research execution job
from a **separately authorized records role** that imports immutable results,
appends events and reads the delivery back before treating it as recorded.
`store_receipt` writes a file under `receipts/`; the `drive_return` block is
filled in by that role from what Drive returned, and `RECORDED` is refused
without a readback. No Drive write credential is held or used here, and
research execution jobs do not edit canonical status cells.

## The first implementation work order: what is the owner's, and not done here

Contract §7 lists six steps. As of 2026-09-18 this repository does **step 4
only** (an `AGENTS.md` pointer and machine-readable order/receipt schemas with
tests). The rest are the owner's, require an admin-capable session or a Drive
write route, and are **NOT done** here:

1. an owner-authorized write-capable connection and a separate admin session;
2. the **access audit**: collaborators, invitations, installed apps, deploy
   keys, token scopes, rulesets, Actions permissions; the initial authenticated
   worker allowlist;
3. the fail-closed CI repair (the mirrored `ci_fail_closed.patch` is a design
   input, stored and not applied; the live workflow gained only the one
   `bridge_check.py` step) and manifest-coverage repair;
5. **branch protection / rulesets** and least-privilege CI settings, tested
   against the forbidden paths;
6. the **smoke task with Drive readback**, after which — and only after which —
   the execution environment may be recorded as installed.

Until those are done, "model-only execution enforced" is not a sentence this
repository can write. The checker is `tools/bridge_check.py`; its negative
controls are `tests/test_bridge.py`.
