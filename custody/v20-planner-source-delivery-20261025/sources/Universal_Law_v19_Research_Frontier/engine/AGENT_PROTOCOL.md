# v17 agent protocol update

Agents propose changes through the local inbox or token-protected proposal API.
Their output remains quarantined until explicit owner approval. Approval changes
only the working branch, computes its exact downstream impact and persists
versioned jobs in the approval transaction. Actual local diagnostic workers
produce artifact-hashed replay receipts; they cannot certify mathematics.
Durable leases, retries, escalation, stale-result rejection and restart recovery
are implemented. See `API_AND_EVENTS_V17.md` and `adapters/WORKER_PROTOCOL.md` for
the complete event and checker protocol. Source and certificate candidates use
separate custody/applicability gates. Formal kernel adapters remain unconfigured.

The following v16 protocol is preserved for historical context; its four-event
list is superseded by the complete v17 event specification.

---

# Reactive mathematical-agent protocol (v16)

The conductor implements three deterministic local agent roles:

1. `schema_agent`: checks the working dataset under the v15 semantic schema without trusting stored stale fingerprints. Its success means the *structure* is valid.
2. `dependency_agent`: uses recursive fingerprints and source-explicit premises to compute all affected obligations and affected primary results.
3. `proof_diagnostic_agent`: enumerates unresolved import scopes, unauthenticated mandatory corrections and side-condition counts per affected result, without trying to claim a proof.

A future proof agent can participate by writing inbox JSON; its proposed changes are **not authoritative**. The coordinator requires explicit approval before branch application. Future kernel adapters must supply the exact formalized theorem, statement-alignment evidence, imported assumptions and source bytes, checker binary/version, full artifact hash and a reproducible checker result. No v16 API endpoint can set checker-passed status.

## Event types

- `PROPOSAL_CREATED`: untrusted change and transitive-impact preview.
- `PROPOSAL_REJECTED`: rejection, no effect on mathematical branch.
- `PROPOSAL_APPROVED`: working-branch mutation accepted for inspection (not mathematical acceptance).
- `CHECK_RUN`: bounded local schema/lineage/diagnostic processing and unresolved-proof report.

Events are append-only in `runtime/events.jsonl`, with monotonic sequence numbers, `prev_hash` and `event_hash` over canonical UTF-8 JSON. The coordinator verifies every link on restart and rejects inconsistent journals. Snapshots are derived by deterministic replay. `runtime/current_state.json` is an atomic blackboard projection. The browser consumes the same state over `/api/state`, `/api/events` (SSE), and `/api/mermaid`.

## Fail-closed rules

- Rejected, invalid or unapproved edits do not modify the effective branch.
- No new record IDs or changed admission labels without a separate schema migration.
- An approved edit recomputes transitive fingerprints; unmodified branches retain their old fingerprints.
- An external import changed from unvalidated to validated cannot be asserted through a text-edit endpoint.
- A journal with a stale hash/sequence refuses replay; arbitrary event types refuse replay.
- A successful `CHECK_RUN` is never a theorem checker certificate.
- A proposed counterexample or contradiction is a claim requiring validation, not an automatic mathematical refutation.

## Determinism boundaries

Effective mathematical data, transitive affected record IDs, and Mermaid source are deterministic for a fixed ordered event sequence and baseline. Event timestamps and randomly assigned proposal IDs are not predetermined; however, recorded events replay into the same branch. The 3D projection is a seeded deterministic layout derived from the current local obligation neighborhood, not a physics engine.
