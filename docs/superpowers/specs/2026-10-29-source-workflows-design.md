# Reliable source freshness and reuse

The owner requests useful project-wide improvement, peer assistance and continued
implementation without repeated permission questions. These engineering repairs
close two reproduced gaps on Main `4ef1867175dbe1072a54a02e95f80c4e2c70a4ff`.
Implementation choices below are the agents' calls under that delegation.

## Operational freshness

The refresher compares a URL's snapshot commit to the latest commit that changed
its path. An unchanged file last modified at A and pinned at a later snapshot B
is consequently marked stale. The public SIDE24 proof has identical pinned and
current blob `44b66f04f89fcd87383b3603fa69f1feb64cdddd`, despite different last-change
and snapshot commits. The preserved pre-fix reproduction returns stale.

Compare pinned/current Git object type and SHA instead. Resolve the current
default-branch commit once per source observation, then walk its fixed Git tree
and the pinned commit's tree nonrecursively. Reject malformed, truncated,
ambiguous, missing or unsupported object responses; these remain unknown.
A typed transient observation binds the URL, pin, current commit and object
identities. Keep existing injected scalar/tuple fetchers compatible. Never write
the moving commit into source_rev, change schemas or reinterpret scientific
validity. Discussion timestamp and owner-release behavior remain as before.

Alternatives considered: retaining the commit comparison perpetuates the false
warning; ancestry-based inference complicates renamed/reverted paths without
answering whether the referenced object changed. Git object comparison answers
the actual path-content question directly. HTTPS GitHub metadata is operational
identity evidence, not a new mathematical source-custody certificate.

## Acquired excerpt to source reference

The admitted acquired-text result currently provides a source link, preview and
excerpt hash, but no path into the existing reading-list/citation flow. Add the
existing Save source control only when the excerpt's exact repository, commit
and path match a validated source entry in that same admitted observation.
Delegate storage, deduplication, export, removal/undo and citation to the existing
research desk. Its source identity contains the whole-file Git blob/byte count;
the excerpt SHA stays in the excerpt disclosure and is never a whole-file hash.

The discovery renderer gets narrow source-decoration and post-render callbacks.
Populate the desk's validated source map before initializing discovery. Refresh
save-button states after replacement/appending and existing list mutations. An
unmatched excerpt has no fabricated save identity. Existing query collection,
paging, URL/history, focus, source admission and hidden text-search saving rules
remain. No new storage schema or semantic/equation-to-theorem join is introduced.

Alternatives considered: a second storage/controller creates competing ownership
and data semantics; requiring a manual file-path search leaves the broken reader
journey. Reusing the one existing controller is the smallest coherent repair.

## Scope and evidence

Semantic paths: tools/node_status_refresh.py, tests/test_node_status_refresh.py,
docs/site/library-discovery.mjs, the research-desk initialization callback and
focused logic/browser regressions. Open PR501 changes only a canonical import
key in research-desk.mjs; its collector/publication-transition work is untouched.
The complete 42-PR path inventory and actual browser/source observations are
retained in the private execution evidence. Canonical asset rewrites follow the
site guide; a fresh nonauthor engineering review checks the frozen whole branch.

Acceptance: snapshot B with unchanged blob/tree is fresh; changed object is
stale; unavailable/malformed evidence is unknown; source pins and record content
are preserved in report-only mode. On desktop/phone, text query → save source →
reading list → exact pinned citation works, duplicate excerpts share save state,
and appended/history-restored rows remain usable. Missing/mismatched source
identities cannot be saved. Existing focused controls and required hosted checks
must pass before ordinary guarded integration and actual public/offline replay.

No mathematical statement, review, acceptance, import applicability, certificate,
status record, workflow, protection or existing Drive item is changed. Same-family
separate engineering review is allowed; scientific effect NONE, organizational
independence credit0. Remaining literature, scientific review, physical-device/AT
and large-graph performance gaps are not declared solved by this repair.
