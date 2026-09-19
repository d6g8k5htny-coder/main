# Operator-mediated pilot

`pilot.py` adds input consistency and exact readback checks to the existing
v1 schemas. It is a library used by an executing session and a separately
authorized records role. It has no scheduler, shell executor, network client,
credential, permission grant, or status writer. Production deployment remains
open. This does not establish scientific truth, independent review, acceptance,
or an authenticated and isolated execution service.

The direct owner instruction for infrastructure work is external to this
repository. A research work order cannot authorize edits to the bridge itself.
The actual pilot order must be authored and archived at the owner boundary
before its pinned copy is used here; the shipped examples authorize nothing.

1. The executing session reads the live authority, claim and lease, current
   source versions and exact bytes, repository identity/head/tree/cleanliness,
   authenticated account and expected output head. It constructs a
   `BoundarySnapshot` from these observations. These are trusted caller inputs;
   a JSON file containing the same assertions is not authentication.
2. Call `preflight(order, snapshot, source_bytes, now)` immediately before
   executing the order's commands. It refuses a changed source, wrong commit,
   dirty checkout, unapproved principal, changed/inactive/expired claim, stale
   output head, or observation more than five minutes old. The fetching role
   checks native source revisions and extraction rules. This is not a lock
   against concurrent remote changes.
3. The host runs the explicitly authorized commands, retains real exit codes
   and logs, and calls `executed_receipt`. Its counts are **command-level
   acceptance checks**, not the number of pytest cases. Nonzero exits remain
   failures. Outputs are byte-hashed under logical capsule paths within the
   order's scope; the capsule can live outside the checkout. The host must
   report actual checkout dirtiness after execution.
4. The records role stores the EXECUTED stage with `store_receipt`, uploads
   that immutable canonical JSON and every output, and downloads their raw
   bytes. It rechecks the output head, then calls `recorded_receipt`. The
   receipt's Drive id and readback hash refer to the original EXECUTED stage,
   avoiding a self-referential hash. A custody record must map each logical
   output path to its actual Drive file id, byte count and raw readback hash.
5. Store the resulting RECORDED stage in a **separate archival-stage store**,
   upload it and read it back too. Preserve both stages. Within either store,
   identical retries are no-ops; changed content at the same idempotency key
   is held for reconciliation, never overwritten. A failure can be RECORDED;
   neither stage means ACCEPTED. Both remain NOT_REVIEWED with zero credit.

The host owns resource limits. This adapter checks observed elapsed time
against the order but cannot kill processes, bound memory, deny networking,
or constrain arbitrary test code. An operator-mediated pilot must disclose
the actual host limits and any unenforced limits. A production executor must
implement those controls and owner identity/authorization before claiming the
proposed contract is deployed. Server-side branch protection, access audits,
independent review and owner acceptance are separate open steps.

Verification: `python -m pytest -q tests/test_bridge_pilot.py`. Negative controls
cover stale/missing sources and observations, changed head/claim/principal,
dirty inputs, failures, invalid receipt evidence, changed/missing raw readback,
duplicate deliveries, conflicting retries and immutable archival stages.
