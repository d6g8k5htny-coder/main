# Infrastructure hardening scope — 2026-09-19

This change supplements the migration at commit
`a38ad1edd04ce2451274811731819ea4e4661ceb`. It does not establish complete Drive
migration, scientific correctness, claim promotion, independent review,
acceptance, or production bridge deployment. Frozen exports are unchanged.

Required CI checkers now run directly: a missing script or nonzero exit fails
the step. Both workflows request read-only repository contents, full history
for append-only checks, and no persisted checkout credential. Scheduled runs
also refuse tracked, staged, new and ignored changes under their four existing
governed path scopes, even when an earlier step fails. This is an end-state
check, not a filesystem sandbox or protection against transient writes.

The strict manifest checker rejects malformed/duplicate JSON keys, non-finite
numbers, invalid digests/sizes, ambiguous booleans, duplicate destinations,
symlinks, nonlocal paths and mismatched bytes. The baseline explicitly pins
the identities of all **114 existing manifests**. Deleted or rewritten pinned
manifests fail; additional manifests are scanned. Changing the baseline itself
still requires review and is not prevented by server-side controls here.

At the base commit, these manifests contain **589 stored entries** and **164
explicit exclusions**. The 589 entries have 589 distinct local paths and 585
Drive ids; 471 are marked exact and 118 are reading copies. Exclusions are
reported individually, never counted as verified or treated as authorized
omissions. Integrity of a reading copy does not turn it into original bytes.

The inventory has 4,456 rows, 3,714 file rows and 2,059 digest-bearing file ids.
1,617 of those ids lack a stored-manifest entry in this scan. This is a coverage
join, **not a missing-file count**: other carriers/provenance paths may cover
them. The preserved source map has 11,649 archive-member, 4,020 payload and 137
exception rows, and 238 dated path-change entries. No bulk reimport follows
from these counts, particularly for private, legacy or vault material.

A read-only live register export on 2026-09-19 has the same 44 tab names/order,
headers and row counts as the dated repository source. Its only 25 changed
cells were cached Review Queue `Age days` values, each +1. The dated export is
preserved. Later coordination events for this task are new records and are
not silently folded into the old snapshot. Source-map native metadata was
checked for its 10 tabs; all native source-map cell contents were not freshly
exported, so this is not a whole-Drive or whole-source-map freshness claim.

The [operator-mediated bridge pilot](../engine/bridge/PILOT.md) adds preflight
and raw-readback checks using the existing schemas. Actual run/custody records
belong at the owner boundary. Work execution, archival, technical review and
acceptance stay separate. No work order, source record or receipt is invented
inside this PR to authorize the PR itself.
