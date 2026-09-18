# Governance canon (reading copies)

This directory holds the operator-issued protocols that govern the research
program, mirrored from the Drive's `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES`,
`10_AXIOMATIC_CORE_SPINE/00.1_GOVERNANCE_CANON` and the R17 release bundle.

> **These files are reading copies. None of them is the object.**
>
> Not one artifact in this repository that mirrors a Drive object is
> byte-identical to it. `OP-PROT-019-v1.1_R17.md` is the sharp case: it has the
> *same byte count* as the object the register names — 14,073 — and a different
> SHA-256 (`efcfdd5c…` here against the declared `04987ba4…`), so a byte-count
> check on this file confirms the wrong bytes. Reviewing "the protocol" from this
> directory is not reviewing the object `registers/json/review_queue.json` names.
>
> Every digest, byte count, extraction rule and transformation is recorded in
> [`PROVENANCE.json`](PROVENANCE.json) and enforced by
> `python3 tools/provenance_check.py`, which also refuses to let any file in the
> repository describe one of these copies as verbatim or byte-exact.
>
> This was found on 2026-09-18 by the nonauthor technical review of `RV-OPS-R17`
> ([`reviews/records/REV-OPS-R17-001.json`](../reviews/records/REV-OPS-R17-001.json),
> finding 1) — a review of this repository's own migration, which found a real
> defect in it. `OP-PROT-019` §2 is the clause it violated: *"The digest
> establishes identity, not truth or authorization."*

Reading order for a new contributor (human or model):

1. `protocols/OP-PROT-019-v1.1_R17.md` — current operational policy (entry,
   claims, review without provider deadlock, quarantine, draft/certify/review).
2. `protocols/OP-PROT-012.md` — autonomous evidence-gated governance and the
   autonomy classes 0–5. **Read the scope note below before relying on it.**
3. `protocols/OP-GDN-002.md` — the coupled mathematical–architectural advancement
   invariant and the required transition record.
4. `protocols/OP-CNS-001-R0.2.md` — preservation, artifact identities, automation
   scope, closure discipline, master manifest.
5. `GIT_ADAPTATION.md` — how each construct maps onto git.

Historical protocols (OP-PROT-001..011, -013..018) are kept under
`protocols/history/` as provenance; they apply only where R17 says they are
retained (scientific definitions, exact extraction rules, theorem-specific
predicates).

**Scope note on OP-PROT-012.** The Drive object this reading copy comes from
(`1hBQR7Pa10DpVeOxTCLv1qIozLT-bU_hgZKo6ksODCuo`) carries the title *"HISTORICAL
OP-PROT-012 — APPROVAL ROUTING SUPERSEDED BY R17"*. The reading order above
listed it second without that qualifier, which reads as though the whole
protocol were current. The supersession as the title states it is scoped to
**approval routing**, and R17 retains parts of the historical protocols — so
this is flagged, not resolved. Which clauses survive R17 is an operator
question, and nothing here decides it. Treat the autonomy classes 0–5 and the
independence predicate as live only where R17 or an operator decision says so.

Authority note: the owner (Dylan Roy) is the single final authority for
canonical promotion, external release, permanent deletion and machine-root
replacement. Nothing in this repository changes that.
