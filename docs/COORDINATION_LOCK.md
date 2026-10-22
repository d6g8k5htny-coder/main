# Coordination audit: bounded report, no pickup authority

`tools/coordination_lock_check.py` reads the existing public node-status view at
one exact Git commit and reports its owner, freshness, source identities, stops
and missing inputs. It writes no ownership record and grants no authority. Every
JSON result includes `advisory_only: true`, `authority_granted: false`,
`universal_lock: false`, `native_reconciliation_required: true` and
`scientific_effect: "NONE"`.

This is the bounded, disjoint contribution claimed in
[475/6353027897](https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6353027897).
It does not implement the genuine distributed lease or cryptographic fence under
[351/6294319918](https://github.com/d6g8k5htny-coder/main/issues/351#issuecomment-6294319918):
R01 governs, product LOCKED, source HOLD, writer UNKNOWN and runtime NOT RUN
remain that source's recorded disposition. The existing semantic work's owner
and stops remain with
[333/6206436926](https://github.com/d6g8k5htny-coder/main/issues/333#issuecomment-6206436926)
and its
[current continuation](https://github.com/d6g8k5htny-coder/main/issues/333#issuecomment-6351948287).
No report, queue, clock or absence of a record transfers those scopes or removes
permission holds.

## Run the report

Run from the repository root, with the pinned commit already available locally:

```sh
python3 -B -S tools/coordination_lock_check.py report \
  --commit FULL_40_CHARACTER_LOWERCASE_COMMIT \
  --node eng.hard-gate --claimant actual-session-id
```

Omit `--claimant`, or pass an empty value, for observation without a claimant
comparison. `--now 2026-10-22T22:00:00Z` supplies an explicit clock for a
reproducible audit or test; otherwise the process reads the actual current UTC
time. An explicit clock is an input to this report, never evidence of authority.
The timestamp format is strictly `YYYY-MM-DDTHH:MM:SSZ`, with a real calendar
date. Naive times and alternate offsets are refused.

The command emits one ASCII JSON line. Strings are JSON-escaped so record text
cannot add multiline workflow commands. Exit 0 means a usable `OBSERVED`
report without recorded blockers; it never means PICKUP, a passing lease,
permission to write, or successful native reconciliation. Exit 2 accompanies
`HOLD`, `CONFLICT` or `EXPLICIT_HANDOFF_REQUIRED`.

| Observation | Reported consequence |
| --- | --- |
| Missing record, null owner or malformed/unknown schema | `HOLD` |
| Requested claimant differs from recorded owner | `CONFLICT`; recorded owner retained |
| Owner has `released_url`, with the same or no claimant | `EXPLICIT_HANDOFF_REQUIRED`; recorded owner retained |
| Observation older than one hour, recorded stale/unknown, or a future observation | Blocking freshness finding; recorded owner retained |
| Stops, missing inputs, refresh errors or open carried-forward findings | Blocking findings retained |
| Fresh, valid owner with no recorded blockers | `OBSERVED`; authority still false |

When findings coexist, a valid different-owner comparison reports `CONFLICT`,
and a valid released-owner comparison reports `EXPLICIT_HANDOFF_REQUIRED`;
additional freshness, stop and missing-input blockers remain in the same report.
Malformed records report `HOLD`. Available raw owner, source, stop and
missing-input fields can still appear on an invalid record and must be read
alongside its validation blockers.

`status` is `fresh`, `stale` or `unknown` for this observation, and
`recorded_status` preserves the recorded freshness flag. The signed
`observed_age_seconds` makes a future observation visible as a negative age;
such a report has `status: "unknown"`. One hour is a fixed checkpoint for
observational freshness, not an ownership expiry or a permission rule. At
exactly 3,600 seconds, a recorded fresh observation remains fresh. A stale owner
requires current task-local reconciliation and any required explicit handoff.
Repeated audits, including repeated matching claimant IDs, cannot manufacture
authority or resolve a duplicate pickup.

## Source and input bounds

The only data path is `docs/site/node-status.json` as a regular Git blob in the
supplied commit. The working-tree copy is ignored, even if replaced or dirty.
The reader checks the exact commit resolution, tree path and regular-file mode,
Git blob digest and byte count. A usable source identity includes the repository,
commit, path, blob ID, SHA-256 and pinned public source URL, together with the
record's `sources` and `source_rev`. Record source revisions must agree with
their pinned URLs and the primary source revision. Referenced sources are
reported, not downloaded or revalidated by this command.

The repository root and configured `origin` must identify
`d6g8k5htny-coder/main` in one of the tool's closed HTTPS/SSH forms. This is a
local configuration guard; it does not authenticate a remote or establish that
the pinned record is the latest native task state. Ambient `GIT_*` variables,
replacement objects, lazy fetching and transport protocols cannot substitute
another source. The command needs no token, performs no network reads, executes
no mathematical payloads, and mutates no Git objects or records.

Limits are a 1 MiB blob, 256 map records, 256 entries per record list, 16,384
characters per validated text field, 20 characters per JSON integer token and
ten seconds per local Git command. Floating-point tokens are unsupported by
the v0 record contract. Duplicate JSON keys, nonstandard numeric constants, invalid UTF-8, parse
anomalies, unsupported fields and unavailable source objects produce a hold.
The tool supports a stricter bounded profile of v0 of
[the existing schema](../architecture/node-status.schema.json), including source
URL/revision equality and a resolution URL for resolved carried-forward findings.
Those checks are observational only: they amend no schema, accept no status and
create no additional authority register.

## What workflow concurrency guarantees

The `audit` job in `.github/workflows/coordination-lock.yml` uses the stable
repository-local group `ul-coordination-node-${{ inputs.node_id }}` with
`cancel-in-progress: false` and a five-minute job timeout. GitHub treats group
names case-insensitively. Participating audit jobs
for the same node share that group; a newly queued audit does not cancel its
running predecessor. Default concurrency retains at most one running and one
pending job, and a later pending job replaces an earlier pending one. Ordering
is not guaranteed. See
[GitHub's concurrency documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

This job serializes its own participating audits within this repository. It
does not serialize or fence writers, other tasks, external sessions, other
repositories or jobs using another group. Duplicate pickups outside this job
are not prevented. GitHub cancellation is asynchronous: server condition
checks, runner messages, signals and forced termination are separate stages,
and some conditions can keep work running. See
[GitHub's cancellation documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation).
Cancellation, failure, replacement of a pending audit and timeout release no
owner or PICKUP. An audit is a report each time it runs.

The unresolved
[node-status freshness follow-up #490](https://github.com/d6g8k5htny-coder/main/issues/490)
means this derived view cannot substitute for fresh native task-local
reconciliation. Claims, reviews, handoffs and releases continue in the current
[agent message board](https://github.com/d6g8k5htny-coder/main/issues/475) and the
existing task discussion under the
[current workflow](../governance/OP-WORKFLOW-20260930.md),
[owner mandate](../governance/OWNER_MANDATE_20261015.md) and
[standing delegation](../governance/OP-AUTONOMY-20260923-v2.1.md).
Current required checks, protection rules and deployment requirements retain
their existing meaning. This audit deploys nothing and grants no scientific
acceptance.
