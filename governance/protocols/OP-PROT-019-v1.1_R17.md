# OP-PROT-019-v1.1 — Drive navigation, quarantine and review throughput

AI-DRIVE-AUTONOMY-R17 · effective 2026-09-17 · operational policy, not a mathematical verdict.

Dylan Roy directly authorized this overhaul and replacement of conflicting operational rules. R17 supersedes R16 discovery budgets, compulsory full-Drive preflight, blanket same-provider technical-review exclusions, recursive registration of drafts, and mandatory whole-package re-download for every task. Earlier protocols remain historical evidence and apply to scientific object definitions, exact extraction, and unchanged theorem-specific predicates where consistent. This authorization is not a proof, an independence credit, or approval of an unresolved theorem.

## 1. One entry point
Use the existing [Research Home](https://docs.google.com/document/d/180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8/edit) or the register's **Start Here** tab. Root **00_START_HERE** contains current routing. Old routing and checkpoint documents are reference history. Do not recursively read their predecessor protocols before working.

The [research register](https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit) has five primary surfaces:
- **Start Here**: compact rules, destinations and exact next reads.
- **Review Queue**: exact review objects and current next actions.
- **File Catalog**: searchable metadata; not proof or exhaustive current membership.
- **Quarantine Index**: exclusions, reasons, successors and restoration information.
- **Work Events**: append-only coordination and output receipts.

Existing scientific registers keep their stable sheet IDs, object identities and formulas. Use domain filter views rather than three competing writable register copies. Historic tabs and records remain available. Root contains folders only; direct links use Drive IDs, not mutable paths.

## 2. Honest call budget and caching
Known entry: read Start Here A1:H24, the needed review rows and recent Work Events in one multi-range request where supported. A new worker who needs the full policy also reads this file once. Read the actual exact source before reasoning. A known continuation needs one live bounded state read plus any changed or uncached source bytes. A first-time root search adds discovery calls. Claim, upload, verification and commit calls are additional; **one navigation read is not one-call proof verification**.

Reuse previously verified bytes by exact digest and extraction rule. Native Docs require revision-aware extraction; Drive modifiedTime alone is not a body digest. Before publishing, freshly resolve the stable target key, current source revision/hash, active claims and current output head. On mismatch, invalidate only affected cached dependencies and reconcile. A 120-minute-old handoff triggers bounded revalidation, not an automatic whole-Drive crawl. Missing fields, future timestamps, bad digests or conflicting pointers invalidate fast continuation.

A handoff is immutable per session/checkpoint. Store a JSON envelope with payload and payload_sha256; hash the payload only, using the bundled deterministic restricted JSON encoding. No self-referential hash field. Required payload: protocol version, checkpoint UTC, task ID, claim ID, stable target key, source identities (Drive ID, exact SHA-256, byte count, extraction rule; revision if native), output head, verified boundary and next action. A spreadsheet coordinate is only a hint. The digest establishes identity, not truth or authorization; compare against live routing and retained source bytes.

For inventory maintenance, use a modified-time delta plus targeted folder/ID verification. This connector does not expose a Drive changes cursor. A modified-time search can miss removals or access loss, so it cannot certify global completeness. Reconcile a relevant subtree on a missing file/move conflict; broader inventory is a maintenance task, never every worker's entry cost. Batch independent reads and bounded register writes. One delivery manifest may cover many immutable payloads; do not manufacture one global registration transaction per scratch file.

## 3. Claims and concurrency
Append one UUID event to **Work Events** with session, actual provider, exact target key, source digest, event type, lease expiry and claim reference. Use AppendCells on this plain log. Never sort, insert into, delete, or overwrite its data rows. Filter views may reorder presentation. Repeated identical event IDs are idempotent; conflicting reuse is an error. Use the confirmed append position to order contenders, not a self-reported timestamp.

Read back the matching target events before starting material work. Among unexpired, unreleased valid claims, the earliest append position controls primary publication. Default lease is 120 minutes; renew on material progress. A heartbeat must name the same claim and session and arrive before expiry; an expired claim requires a new claim. An unsuccessful contender switches to an explicit independent method, falsification/review branch or another task. Related topic does not imply duplicate object.

Sheets batch updates are atomic within one request, **not a compare-and-swap lock across clients**. Sharding and append logs do not eliminate every race. This is cooperative coordination, not a hard distributed mutex. Legacy Active Work Claims and advisory lanes remain collision inputs during migration; do not declare a known live older claim absent just because it is not in Work Events.

Publish immutable candidate outputs first. Immediately before changing a current pointer, recheck source/head/claim; use Docs requiredRevisionId for native text writes. Record the expected old head and proposed new head in PUBLISH events, and read back the resulting pointer. Multiple successors of one old head are a publication conflict: hold only that target, preserve both branches, and append a reconciliation selecting or merging them with reasons. Never resolve it by silently overwriting proof bytes. If a strict machine-enforced lock is required, a transaction-capable service is still needed; none is represented as installed here.

## 4. Technical review without provider deadlock
Record four separate dimensions: exact object correctness verdict; scope/dependency verdict; reviewer authorship/exposure; organizational independence.

A fresh nonauthor session of any provider may review an exact object, including the author's provider. It must disclose source exposure and perform substantive reconstruction, counterexample search or meaningful execution. Same provider is **zero organizational independence**, not a prohibition on useful technical review. Different provider alone does not establish independence if it coauthored the target or reused the same reasoning.

An author/coauthor can do internal verification and fix defects, but cannot call that self-review peer review. When all active instances coauthored the exact object, seek a fresh nonauthor instance; meanwhile, continue explicit internal checks, decomposition and other unblocked tasks. Merely starting the same topic does not permanently disqualify a model from reviewing a different object/version. Existing independent verdicts retain their exact scope; do not award a second credit to the same lineage.

Technical statuses: READY, IN_REVIEW, PASS_TECHNICAL, AMEND, FAIL, CANNOT_VERIFY, or NEEDS_RECONCILIATION. Independent status is a separate field. A task may finish its technical review while an external-independence predicate remains open. Never relabel an independence-required theorem terminal solely because its technical review passed.

A valid review records: source ID/hash/bytes and extraction, author and reviewer session/provider, exposure, precise hypotheses, reconstructed argument, actual execution and negative controls (if applicable), findings by criterion, unresolved dependencies, verdict and reproducible output. A source search or hash match alone is not mathematical review. No confidence voting.

## 5. Aging and automatic pickup
Review Queue is the active pickup surface; legacy dispatch rows remain provenance and scientific gates. Record request time separately from last substantive review activity. A refresh, rename or metadata touch does not reset review age. If the historical request time is unknown, mark age UNKNOWN and reconcile; do not substitute a file modification date.

On each research session entry, select an eligible overdue review before starting duplicate work, unless the user specified another exact task or an unfinished claim should resume.
- At 7 days: prioritize a qualified nonauthor technical reviewer regardless of provider.
- At 14 days: split a large packet into exact, independently checkable obligations; perform an available technical pass.
- At 30 days: escalate in the queue with the named missing capability or dependency. Continue other work; no global stop.
- Never turn age into automatic approval, invalidation, or extra independence.

The installed daily automation uses this same pickup rule and attempts one bounded review or reconciliation per run. It does not summon or impersonate another provider. Capability failures produce a named blocker, not a fictitious review. Notify the owner only on a completed substantive result or actionable new blocker. An existing verdict must be reconciled before repeating the same review. July queue items are *unresolved review obligations*, not necessarily work no one has ever reviewed.

## 6. Quarantine and restoration
**90_QUARANTINE_AND_TRIAGE** is the visible directory for nonauthoritative material. Existing package quarantines and the research vault remain linked in Quarantine Index.

| Classification | Evidence required | Action |
|---|---|---|
| EXACT_DUPLICATE | Matching raw digest/bytes, or declared native-body equivalence with format limitations | Choose an existing keeper; preserve IDs; move surplus copy; record keeper and rollback |
| SUPERSEDED | Explicit numbered successor or source-backed retirement | History/archive, not a claim of mathematical falsity |
| DEFECTIVE_SCOPE | Concrete failed statement, counterexample or reproducible invalid certificate chain | Exclude the affected claim; preserve valid unrelated content; link repair and re-review requirement |
| UNVERIFIED / CONFLICT | Missing identity, unresolved custody or conflicting heads | Isolate from active consumption pending resolution |
| LEGACY_INSPIRATION | Existing legacy classification | Inspiration only until rederived and reviewed |

Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal. Before a move, verify membership/parents, intended destination and affected links/manifests. Keep a move record with source parent, destination, original name, reason, identity and keeper/successor. Moving or renaming must not be represented as a scientific verdict.

Frozen artifacts or archive members that cannot be moved independently receive a **logical quarantine** keyed by carrier ID plus relative path and hash. Their bytes and old manifests remain intact; the active catalog excludes the affected certification claim. A new erratum/successor and fresh review restore eligibility. Do not alter a frozen Merkle tree to make history disappear. No permanent deletion in this workflow.

## 7. Draft, certify, review
Draft freely in **06_SANDBOX_FRONTIER** under a task/session folder. Drafts carry no canonical authority; no per-edit global manifest churn. Save expensive checkpoints and named negative results so work is not lost.

To publish: freeze a candidate outside the sandbox; compute exact identities; re-fetch/read back; produce one package manifest with dependency identities and a handoff. Register the delivery bundle once in the existing Artifact Index/Evidence Lineage, and any actual frozen scientific objects in Frozen Objects/Identity Drift Watch. Put the exact review obligation in Review Queue and append its Work Event. A bundle can contain many payloads without pretending every member is already reviewed.

Use DRAFT -> CANDIDATE_VERIFIED -> READY_FOR_REVIEW -> REVIEWED/AMEND. COMPLETE describes the particular task's acceptance test, not automatic theorem approval. A task remains PARTIAL/CANNOT_VERIFY when a required receipt or check is absent. Promotion follows the exact scientific predicates, including any retained independence requirement.

## 8. Evaluation of v1.0
| Proposal | Verdict and installed correction |
|---|---|
| Fast handoff is deterministically equivalent after one matching hash | REJECT as stated. It takes two described reads and omits mutable claim/head, source availability and policy checks. R17 adds scoped revalidation, explicit call accounting and noncircular payload hashing. |
| Three domain registers and staging eliminate collisions | REJECT guarantee. Domain filter views and one append-only log reduce collisions; guarded immutable publication and conflict reconciliation address remaining races. |
| Sandbox alone prevents leakage or Merkle corruption | REJECT guarantee; ADOPT two phases with an explicit promotion gate, exact identity and quarantine exclusion. |
| Mark COMPLETE immediately after registration | AMEND. Registration is custody, not proof/review completion. Use the distinct states above. |

Technical sources: [Google Sheets batchUpdate](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/batchUpdate), [AppendCells request](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/request#AppendCellsRequest), [Drive change tracking](https://developers.google.com/workspace/drive/api/guides/manage-changes). R17's policy choices are our design, not Google guarantees.

The accompanying local reference validator and adversarial tests demonstrate protocol predicates on specified inputs. They are not a deployed enforcement service and do not prove race freedom for Google Drive.
