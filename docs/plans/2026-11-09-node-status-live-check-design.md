# Q-9006 live freshness companion design

Status: test-first preparation only. No production implementation, workflow, ref,
commit or public disposition is authorized by this artifact. Ordinary unittest
discovery and the existing named dormant real-record test remain unchanged.

The source basis is Main 1902aac113761974c4569f78acb6a8baadabb7f1:
- tests/test_living_project_validation.py blob 87d63d104ab29387ddb3f7236d76b2e835e205d8,
  RealRecordTests.test_real_records_fresh_against_live_source_head, line 453.
- tools/node_status_refresh.py blob 5fa7c8b6647938a0aca538a7735edede65de9ed5:
  ContentObserver, refresh_record and the cited-object-content-v1 contract.
- tools/validate_node_status.py blob e9d44b7620536b389a35ff6fad8aef2b0464b5d2;
  architecture/node-status.schema.json blob 2bad95d30bcb89ea686d1ac5a50d52113dd730be.

Purpose: execute the existing freshness semantics against actual public HTTPS
observations in a separate mandatory read-only G5 companion, without making
ordinary offline discovery internet-dependent. Freshness means unchanged cited
Git object type/mode/OID and unchanged discussion timestamp, not equality of an
old immutable source pin with a moving default-branch commit. It means nothing
about mathematical validity.

Approaches considered:
- Network inside every ordinary discovery: rejected; restricted-network jobs
  and repeated normal/optimized discovery would acquire an anonymous API dependency.
- Consume a caller-supplied "fresh" transcript: rejected; that alone authenticates
  neither the current source nor actual HTTPS execution.
- A small companion using the existing default production ContentObserver:
  selected by the parent. Direct offline helper tests replace only the external
  observation boundary and never earn actual live coverage.

Future production surface: tools/node_status_live_check.py, standard library
only, exposing LiveCheckError(ValueError),
evaluate_fresh_records(root, *, observer, identity) -> dict, and
run_live(root, *, identity) -> dict. The helper reads and production-validates the
complete four-file real record set, requires a nonempty complete fresh cohort,
uses the existing refresh_record with one supplied observer, and checks exact
observation URL multiplicities/counters and captured head identities. The
bounded four required filenames are eng.d0-packet-allowlist-fix.json,
eng.hard-gate.json, math.lifetime-remainder.json and
math.side24-coefficient.json. Additional files may not be silently ignored:
reject an unexpected record set until the selected scope is reviewed.
Current inputs contain three fresh candidates and one deliberately stale
eng.hard-gate record. Report that stale declaration separately; do not relabel it
or require its source to become fresh.

Capture schema, graph and record bytes before validation and compare their
identities after observation. Record bytes, paths, source pins and ownership
fields are never rewritten. Reject missing/invalid inputs, no fresh candidates,
a stale or unknown positive candidate, errors, malformed/incomplete observation
coverage, inconsistent heads, and any input drift. The direct unit controls pin record, schema and graph byte drift.

The helper report contains records_total, fresh_candidates, fresh_checked,
fresh_nodes, nonfresh_records, input_sha256, identity, observed_heads,
source_observation_count, source_observations, scientific_effect=NONE and
cbz_green=false. identity contains repository, checked_commit, run_id,
run_attempt and integer python_optimize (0 or 1, never bool).
Only d6g8k5htny-coder/main and a full immutable checked commit are accepted.
The future CLI must derive and compare the actual selected source commit with
the hosted context; unit-supplied identity is not proof of a hosted run.

run_live requires a nonempty GITHUB_TOKEN before constructing ContentObserver().
It constructs that real default class exactly once per mode and delegates to
the real helper. There are no API/transcript/fetcher CLI inputs, anonymous
fallbacks, retries or transport changes. Offline tests may replace that external
constructor only to verify wiring; they remain synthetic engineering controls.

Later hosted execution will use a dedicated read-only job at the same source
and run attempt as the prepared regression composition, with normal and
optimized invocations, bounded runtime, original input/tool/source digests,
head/time observations, real counters and raw failure retention. A failed,
denied, rate-limited or unavailable observation is nonzero, not skipped or green.
The current observer caches heads/commit roots/trees per instance; no cache is
shared across modes, and mutable discussion reads remain separately observed.
Keep its token out of offline and restricted-container tests.

Non-goals: no Q-9002 through Q-9005 owner, artifact or dependency-schema adoption;
no fabricated records, mathematical owner or scientific assertion; no source
pin update, new transport, generic orchestrator, gate weakening or 100% claim.
Authentic historical assignment/release provenance remains historical
operational-role evidence, not mathematical ownership.

The existing dormant named test/history is preserved until actual companion
qualification and an explicit documented suite split. Offline test success is
not live success. Publishing these tests first is expected to fail because the
production companion is absent; no RED or GREEN result is claimed here.
