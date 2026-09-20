# Derived research checkpoint

`tools/research_frontier.py` reads the current claim graph and selected existing
operation trials and run receipts. It reports recorded dependencies, exact input
identities, and recorded failures or non-runs. It is a derived view, not a new
governing register or a mathematical status writer. Drive remains the governing
record. The implementation and its tests are coauthor work at zero organizational
independence credit.

```bash
python tools/research_frontier.py self-check
python tools/research_frontier.py snapshot
python tools/research_frontier.py snapshot --output /absolute/outside-repository/frontier-001.json
python tools/research_frontier.py check /absolute/outside-repository/frontier-001.json
python tools/research_frontier.py diff /absolute/outside-repository/frontier-001.json /absolute/outside-repository/frontier-002.json
```

`self-check` derives and validates the current view, prints counts, and exits
nonzero on rejected input. `snapshot` prints canonical JSON by default. Only an
explicit `--output` writes a file; its parent must already exist, the destination
must be outside both the inspected repository and the implementation tree, and
an existing file or symlink is never overwritten. No source, status, register,
receipt or CI file is written. Bytecode writes are disabled by the command.
For another checkout, put `--root /absolute/repository` before the subcommand.

The initial current-scope replay at implementation time finds 39 nodes, 29 edges,
51 unresolved dependency paths and **zero recorded diagnostics** among four
trials and four receipts. Zero means none of the selected records report one;
it does not mean no operation has ever failed or been refused.

## Exact scope and identity

The payload lists each selected file's relative path, SHA-256 and byte count:

* `claims/graph.json` and `engine/operations/REGISTRY.json` are required.
* `engine/operations/trials/*.json` selects direct trial records.
* `engine/receipts/*/*.json` selects direct records one lane below receipts.

The implementation and existing validators are pinned separately, by their
observed file bytes. Source membership and bytes are reread before completing
capture, and Git identity is compared before and after. This detects changes
during the observed reads; it is not an atomic lock or authenticated execution.

The checkpoint includes Git HEAD, its tree, dirty state and a SHA of porcelain
status metadata. **The tree describes HEAD.** Dirty metadata does not identify
all uncommitted contents; only the explicitly listed source bytes are pinned.
The implementation identity names the code hosting this command even when
`--root` inspects another checkout. Git history, deleted/unpersisted attempts,
live Drive, bridge receipts, nested receipt folders, the slack registry and
numeric certificate files are outside this version's scope. Vault, legacy and
quarantine bodies are never opened.

JSON duplicate keys and non-finite numbers are rejected before projection.
Graph structure, selected field types, references, duplicate edges and cycles
are checked. Unknown top-level graph formats and `depends_*` variants are
refused; the payload enumerates exactly which node fields and edge types it
interprets. Other node fields are pinned as source bytes but not interpreted.
This does not replace `tools/claims_check.py` or its scientific firewalls.

Receipts pass `engine.receipt.validate_receipt_object`, including body and
argument digests. Trials pass the existing `tools.operations_check.check_trial`
against the current catalog, including exact replay of their recorded identity
steps. Catalog row hashes are recomputed. Historical catalog hashes are refused
as unsupported by this bounded view; their records are not silently discarded.
The full register-to-catalog transcription check remains the existing operations
checker's job. The checkpoint does not confer mathematical validity on a record.

## Dependencies and diagnostics

Every edge retains its recorded type: `depends_on` or `sub_obligations`. The view
retains all paths through diamonds. A route to a premise with the exact token
`OPEN` or `NOT_CLOSED` is listed separately for `status_frozen_v2_2` and
`status_register_note`. A note-layer `CLOSED` never hides a frozen-layer `OPEN`.
Other words, review independence fields and proposals are carried without being
turned into numerical scores or declarations of satisfaction. Paths establish
recorded reachability only: there is no inferred AND/OR composition, theorem
admission, minimum cut, scheduling priority or completeness of proof dependencies.
The finite path budget fails closed rather than returning a partial view.

The diagnostic view uses only these existing recorded outcomes:

* Trial `NOT_RUN` with its recorded reason, or `IDENTITY_NOT_REPRODUCED` with its
  failure and failed steps.
* Receipt `FAILED`, `UNAVAILABLE`, `SKIPPED` or `DRY_RUN`, preserving its error
  and notes.
* Receipt notes beginning exactly `total() refused: ` or
  `certified_enclosure() refused: `, the existing cover runner's refusal forms.

Every diagnostic points to its source path and exact field; that path resolves
to the byte/hash manifest. Node IDs identify entries of `claims/graph.json` in
the `claims` or `premises` map according to their `kind`. Original node source
citations are preserved too. Caveats such as “total() refuses while pending,”
`does_not_establish` text and missing run records are **not** refusal events.
An input-format rejection exits with an error; this command does not persist it
as a new event. These are unsigned records of asserted execution, not independent
authentication that an execution occurred.

## Check and comparison guarantees

The envelope hashes the UTF-8 payload serialized with sorted keys, compact
separators, literal Unicode and no non-finite values. The digest is stored
outside the payload, avoiding self-reference. The local observation timestamp
is explicitly untrusted. A file is immutable only in the writer's limited
sense that it refuses a second write; neither its hash nor Git identity is a
signature, trusted clock or authority grant.

`check` validates the envelope, scope, authority restrictions and dependency
projection, then re-derives the view from **current** sources. Every field except
the local observation time must match, including source membership and exact
bytes, implementation hashes and Git metadata. A historical checkpoint therefore
fails current-source replay after inputs or Git state change; this does not by
itself mean the original computation was wrong.

`diff` validates two checkpoint envelopes and their internal dependency
projections, then compares the closed node/edge/path/diagnostic fields. Only
changed nodes and added/removed list entries are shown. Source and Git identities
are compared separately, so a whitespace-only source edit is visible without
being called a mathematical change. Unprojected changes appear only as source
identity changes. Diff does not replay historical sources, authenticate either
snapshot or decide semantic equivalence.

Tests in `tests/test_research_frontier.py` mutate temporary inputs: missing
references, cycles, repeated edges, unsupported dependency forms, changed status
types, duplicate JSON keys, stale catalog/source identities, invalid receipt
hashes, altered checkpoint projections, output overwrites and symlinked paths.
They also preserve both diamond routes and keep hypothetical refusals out of
the diagnostic view. Passing these checks establishes neither mathematical
promotion, organizational independence, novelty nor whole-program completeness.
