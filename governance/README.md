# Governance canon (reading copies)

This directory holds the operator-issued protocols that govern the research
program, mirrored from the Drive's `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES`,
`10_AXIOMATIC_CORE_SPINE/00.1_GOVERNANCE_CANON` and the R17 release bundle.

> **These files are reading copies unless [`PROVENANCE.json`](PROVENANCE.json)
> says otherwise. None of them is the object.**
>
> As of 2026-09-18, three of the seven mirrored artifacts in this repository are
> byte-identical to their Drive objects, each verified by a full SHA-256 match —
> and it matters *who* declares the digest. `protocols/OP-PROT-019-v1.1_R17.md`
> (14,073 bytes, `04987ba4…`) matches the digest a **register** declares:
> `registers/json/review_queue.json` row `RV-OPS-R17`. `docs/FULL_DOCS_MATH_READ.md`
> (`448adee5…`) and `docs/R17_IMPLEMENTATION_REPORT.md` (`7b2f6cc1…`) match
> digests declared only by the **accessibility source map** and inventory
> (`drive/source_map/Files.csv`, `Payloads.csv`, `drive/inventory.jsonl`); the
> registers name neither object nor digest for them. The other four — the three
> native Google Docs protocols in this directory and the register export — have
> no payload digest anywhere in the corpus, so their exactness is unverifiable
> and they remain reading copies. For two of those the repair found known
> content divergences: `OP-CNS-001-R0.2.md` omits the export's first line
> (*"Live rule: 00_LIVE_GOVERNANCE - Independent-Eyes Rule (OP-PROT-011)."*)
> and `OP-PROT-012.md` carries " at time of export" inserted into its Status
> line with quotes flattened to ASCII. Unverifiable is not the same as
> unaltered.
>
> `OP-PROT-019-v1.1_R17.md` was the sharp case. Until 2026-09-18 the copy here
> had the *same byte count* as the object — 14,073 — and a different SHA-256
> (`efcfdd5c…` against the declared `04987ba4…`), so a byte-count check on it
> confirmed the wrong bytes. The whole difference was one blank line inserted
> after the title and one trailing newline dropped. The raw object was
> downloaded, hashed to the declared digest, and written back; the repair and
> every byte-level difference it found are in
> [`../docs/PORT_FIDELITY_REPAIR.md`](../docs/PORT_FIDELITY_REPAIR.md). A digest
> match is identity, not review: the object enters at the status its register
> row carries.
>
> Every digest, byte count, extraction rule, transformation and outcome is
> recorded in [`PROVENANCE.json`](PROVENANCE.json) and enforced by
> `python3 tools/provenance_check.py`, which also refuses to let any file in the
> repository describe a non-exact copy as verbatim or byte-exact.
>
> The defect was found on 2026-09-18 by the nonauthor technical review of
> `RV-OPS-R17`
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
