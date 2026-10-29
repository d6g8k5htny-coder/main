# Actual anonymous public-source observation

Candidate30326e7adc311a04261da6b4c70d48bed6265c35, toolSHA256
 a9a6f24af9917543bfe2010c530ae4fecc8da532291bfc2565efbbbce5334cd2.
One actual --check invocation, no --write; GITHUB_TOKEN/GH_TOKEN explicitly
removed from the process environment. No transport/security overrides or retries
through alternative routes. Public HTTPS calls succeeded without unknown results.

Observation21:04:47–21:05:33UTC on2026-10-29. Math default-head cut
95eb8daaa02e89be1f816bc7f58eac812dd1b469. Exit1 is CORRECT:
- eng.d0-packet-allowlist-fix: fresh
- eng.hard-gate: stale (record currently says fresh)
- math.lifetime-remainder: fresh
- math.side24-coefficient: fresh

All11Git-source observations independently matched already-fetched publicGit
objects using git ls-tree at exact pinned/current commits, including kind/mode.
The12th observation is mutable comment6351719961; its updated_at matches the
record and earlier nativeAPI observation2026-10-22T21:28:51Z. It cannot be verified
by Git. No mathematical claim or role ownership follows.

Input-before identity comes from the strict tool's actual captured-byte manifest,
not a reconstructed claim of a separate pre-launch shell hash command. Allfour
hashes equal post-run files and committed30326e7a bytes; input_drift false. Source
working tree was clean after run. Raw JSON and independent reconciliation retained.

## Proposed reconciliation — NOT performed

After review, adopt exactly the observed status disposition, retain all immutable
sources/source_rev, owner:null, next_action/missing_input/stops and NONE scientific
fields. Set hardgate status stale; the other3remain fresh. Use the actual completed
observation cut as observed_at, document its non-atomic interval and retained
source manifest; no claim of freshness beyond that observation. Keep the published
node-status projection byte-for-byte equivalent to the record values via existing
projection/release workflow. Never repin the changed directory to force fresh.

The original live-fresh test is still unconditional skip. Eventual replacement
must check a NONEMPTY genuine fresh subset against real bound current observations,
reject unknown/error/stale members of that subset, and also retain/check hardgate's
actual stale state. It must bind record snapshot hashes and include complete source
coverage. Cached supplied reports can exercise parsing/binding but cannot masquerade
as live acquisition. Offline tests require a separately bounded hosted live gate
or explicit unexecuted status; absence of network cannot become a green skip.

Keep --check's stronger all-sources-fresh meaning distinct: it legitimately remains
nonzero for stale hardgate even after truthful data reconciliation. A claim-truth
mode/test would check declared disposition equality and nonempty freshness coverage,
not silently weaken this existing all-fresh exit contract. No fullG5 or Q9006 terminal
closure is claimed by this observation/addendum.

## Historical observation versus present claim

Original observed_at is2026-10-22T17:00:35Z for d0/hardgate/SIDE24 and
2026-10-22T21:57:11Z for lifetime. Their fresh labels are dated historical
observations, not evidence of present freshness. This run does not reconstruct
or invalidate those historical observations; it establishes the2026-10-29 cut.
Hardgate's historical fresh value must not be presented as a supported current
fresh claim. Any refreshed projection must carry the newly observed stale status
and actual new observation time while preserving the older source pins. No
claim is made about when between historical and present cuts the directory changed.
