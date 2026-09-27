# Reconnaissance memo — OP-WITHDRAWAL-20260921

Scope: provenance-safe support withdrawal and loss-only dependency validation,
not a scientific theorem or a novelty claim. Inspected on 2026-09-21.

Primary sources: W3C PROV-DM (https://www.w3.org/TR/prov-dm/), GitHub secure-use
reference (https://docs.github.com/en/actions/reference/security/secure-use),
and GitHub workflow syntax
(https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).
The official actions/checkout v4.2.2 ref was independently fetched through the
GitHub connector and resolved to 11bd71901bbe5b1630ceea73d27597364c9af683.
This is a verified existing pin, not a claim that v4 is the latest release.

PROV-DM separates activity termination from entity invalidation and derivation.
That distinction is reused to separate scheduling from support and truth.
GitHub's guidance supports immutable action references, minimal permissions and
avoiding privilege-bearing execution of untrusted contributions. The dedicated
pilot workflow uses contents: read and persist-credentials: false; it neither
changes repository settings nor gives the checker Drive credentials.

No external source proves the research dependency map complete, supplies the
research-specific verdicts, or makes a model review independent. The implementation
therefore retains source statuses, records coverage as a separate obligation and
marks legacy objects unmigrated. The AND/OR support model and loss monotonicity are
standard reasoning mechanisms; no claim of a new general architecture is made.

Disposition: proceed with a scoped operational pilot and nonauthor review. Final
refutation, restoration, live graph migration and automatic cross-system commits
are outside v1. Existing frozen-source and scientific gates remain unchanged.
