# v19 local API, events and effect boundaries

`launch_v19.py` binds127.0.0.1:8776. Existing workspace/conductor routes remain
available. New owner actions require the local `X-UL-Token`, matching loopback
Host/Origin, bounded JSON and exact action fields. The token is embedded only in
the locally served page. This is a single-owner boundary, not remote identities.
COOP same-origin, COEP require-corp and CORP same-origin enable the Worker/SAB
experiment without external frontend assets.

| Route | Contract |
| --- | --- |
| GET `/frontier` | New control surface; existing workspace stays at `/` |
| GET `/api/frontier/state` | Math, conductor/IR, evidence, suggestions/learning, branches, annotations, jobs, urgency and capabilities |
| GET `/api/frontier/events` | SSE `frontier` events; current snapshots, event cursor and keepalives |
| GET `/api/frontier/positions` | Validated ULP1 binary frame, canonical graph identity, non-synthetic104-node positions |
| GET `/api/frontier/stream` | Same-origin RFC6455 read-only binary stream; no application commands accepted |
| POST `/api/frontier/plan` | `{target_ids:[...],budget:20}`; exact records, three registered milestones or `scalar-variance-bounds` |
| POST `/api/frontier/claim` | `{task_id,agent_id:"local-owner",strategy?,lease_seconds?}`; scoped global reservation |
| POST `/api/frontier/failure` | `{claim_id,diagnosis,evidence?}`; report classification, not mathematical verdict |
| POST `/api/frontier/run-math` | `{event_id?}`; fixed scalar pilot only, durable dedup/reservation; no submitted profile/source/command |
| POST `/api/frontier/belief` | Exact proposition/scope/polarity/source; optional input fingerprint/record ID/evidence; reported support/refutation |
| POST `/api/frontier/redteam` | Fixed generic Gaussian moment probe: optional power2/4 and bounded rational claimed upper bound |
| POST `/api/frontier/learning` | `{epochs:40,seed?:0}`; bounded pure-Python experiment, no graph installation |
| POST `/api/frontier/branch` | `{name,edits:[{kind,target,replacement}],base_hash?}`; typed validated pending branch |
| POST `/api/frontier/branch/stage` | `{branch_id}`; stages ordinary pending proposals without approval |
| POST `/api/frontier/source-update` | Existing source ID, exact old/new SHA and URL; stores a proposed update, no fetch/install |
| POST `/api/frontier/annotations` | Annotation set/delete, exact record ID and text; no mathematical status fields |
| POST `/api/frontier/annotations/merge` | `{state:...}`; bounded annotation-only replica state |

Unknown fields, unsupported effects, malformed rational values, attempted submitted
code and forged scientific annotation fields fail closed. A finite red-team
search returning no witness is inconclusive. All refutations remain at their
explicit generic law rather than being applied to a different conditioned field.

Conductor claims/artifacts, runtime jobs, evolution and annotations have independent
hash-linked event journals. Reported evidence is an append-only SQLite table with
body hashes; release holds the write fence and creates a validated stable backup.
Native jobs use a separate process lease, relative authoritative paths and capped
recovery. Event handling deduplicates request IDs; active identical requests can
alias one job. A technical retry reserves new effort, and old reservations are
never refunded by creating another blueprint.

`job.verification` is the recorded historical execution verdict. For a retained
completed job, `job.current_verification` is the current verifier result. A
historical pass cannot mask a current blocked/invalid/stale verdict. UI shows
these axes separately. Old profile history stays readable and unverified; a
current version has distinct task/blueprint fingerprints within the same budget.

A semantic IR self-consistency check does not authenticate a caller's source.
Only the actual coordinator is used by these endpoints; there is no replacement
IR submission route. Cryptographic commitments and visualization cannot create
mathematical acceptance. The old explicit approval gate and all original status
axes remain independent.
