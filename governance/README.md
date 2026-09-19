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

The earlier protocols are not one class. Only two carry a HISTORICAL retitle on
the Drive: `OP-PROT-011` (`HISTORICAL OP-PROT-011 — INDEPENDENCE GUIDANCE;
TECHNICAL REVIEW NOW R17`) and `OP-PROT-012` (`HISTORICAL OP-PROT-012 — APPROVAL
ROUTING SUPERSEDED BY R17`). The register export the repository ships labels the
later ones live, verbatim from `registers/json/artifact_index.json`: OP-PROT-013
"ACTIVE — OPERATIONAL ADJUNCT TO OP-PROT-012"; OP-PROT-014-v1.0 "CURRENT ACTIVE
OPERATIONAL PROTOCOL"; OP-PROT-015-v1.1 "ACTIVE"; OP-PROT-016-v1.0 "ACTIVE
CONTROL"; OP-PROT-017-v1.0 "PUBLISHED / ADOPTION-READY" (its reference
implementation OP-PROT-017-REF-v1.0 "COMPLETE / BYTE-EXACT"); OP-PROT-018-v1.0
"ACTIVE CONTROL PROTOCOL"; OP-RECON-20260916-v1.0 "ACTIVE / RAW READBACK
VERIFIED". None of those seven is held in this repository yet; they are indexed
in `drive/inventory.jsonl` and queued for mirroring. R17 itself is the control
plane (`autonomy_control` row CONTROL_PLANE_VERSION = AI-DRIVE-AUTONOMY-R17:
"R17 supersedes R16 operational entry/budget/provider locks. Scientific
predicates and exact frozen identities remain."). Until 2026-09-19 this paragraph
called OP-PROT-013..018 historical, which no register row says.

`protocols/history/` holds provenance for the operator relays of July 2026:
**as of 2026-09-19, four objects** — the three that exist in the Drive canon
lane as digest-bearing markdown (OP-PROT-001 `1v4Cv5xBvZRl8Ing9y34xN_xgFg-_K2q5`,
OP-PROT-003 `1pEGZoTdCUe9hu5tE_O8nY2N84STQNXhQ`, OP-PROT-005
`1x1JzVd5CMpP6Iqus6FSMHO6ncDrxHxUo`), stored byte-exact with SHA-256 and byte
count equal to their `drive/inventory.jsonl` rows, and a reading copy
(`exact: false`) of the Cross-Model Review Ledger v1.0 Doc; see
`protocols/history/README.md` and its `_MANIFEST.jsonl`, which
`tools/verify_manifests.py` checks. (Until 2026-09-18 this paragraph said they
were kept here; until 2026-09-19 it said the directory was empty.) One name is
ambiguous: "OP-PROT-003" denotes two different Drive objects — the
Operating-Philosophy relay mirrored here (`1pEGZoTd…`, 4,702 B, "close easiest
items first, retire settled facts permanently") and the majority-rule record
`OP-PROT-003 — Superseded Majority-Rule Record and Review Guidance`
(`1pVAG4d0h7kPv0dA0BM_yuWL3gDrrIAjpI1uhTQoAHP0`, a native Doc in
`05_FOUNDATIONS_AND_PROTOCOL_HISTORY`), the one OP-PROT-009 §9 supersedes and
`registers/json/relations.json` REL-OPP008-002 links. Mirroring changes nothing
about which protocol governs.

**Scope note on OP-PROT-012.** The Drive object this reading copy comes from
(`1hBQR7Pa10DpVeOxTCLv1qIozLT-bU_hgZKo6ksODCuo`) carries the title *"HISTORICAL
OP-PROT-012 — APPROVAL ROUTING SUPERSEDED BY R17"* (retitled 2026-09-17). The
register says what R17 retired: `autonomy_control` AUTONOMOUS_DECISION_BUDGET =
"RETIRED — R17 evidence predicates" ("Current owner-authorized operations have
no 50-decision counter gate"), and CONTROL_PLANE_VERSION = "R17 supersedes R16
operational entry/budget/provider locks. Scientific predicates and exact frozen
identities remain." No source enumerates which of OP-PROT-012's clauses survive
beyond that sentence, so the reading copy's own header no longer asserts that
its autonomy classes "remain the governing description" (it did until
2026-09-19). Treat the autonomy classes 0–5 and the independence predicate as
live only where R17 or an operator decision says so.

Authority note: the owner (Dylan Roy) is the single final authority for
canonical promotion, external release, permanent deletion and machine-root
replacement. Nothing in this repository changes that. The sentence is
`OP-CNS-001-R0.2`'s and agrees with the Canon and the Formalization Board. The
governance canon on the Drive (`10_AXIOMATIC_CORE_SPINE/00.1_GOVERNANCE_CANON`)
also holds `HISTORICAL OP-PROT-011 — INDEPENDENCE GUIDANCE; TECHNICAL REVIEW NOW
R17`, whose SUPERSESSION NOTICE of 2026-07-25 reads that OP-PROT-011's
"statement that theorem promotion is Dylan-only, and its reservation of
machine-root replacement to Dylan are superseded. Those actions are now governed
by executable evidence predicates under OP-PROT-012. Permanent deletion remains
prohibited and external release remains default-disabled until an autonomous
release predicate exists". The register records that chain as operator
decisions, not as an open question: `operator_decisions` row
`OD-OP-PROT-011-20260724-001` (Active FALSE; "Superseded 2026-07-25 for approval
and terminal routing by OP-PROT-012. Retained as independent-eyes evidence
guidance"), row `OD-OP-PROT-012-CROSSFAMILY-20260725-001` (source "CL-GOV-222;
authenticated conversation 2026-07-25"; "Direct operator answer; not inferred
from Drive content"), and `artifact_index` row OP-PROT-011 "RETAINED AS
TECHNICAL PREDICATE / SUPERSEDED AS LIVE APPROVAL ROUTER BY OP-PROT-012" — after
which R17 (2026-09-17) retitled OP-PROT-012's approval routing HISTORICAL in
turn. So the live approval router is R17; OP-PROT-011 is retained as a
technical-predicate and independent-eyes layer; and the owner-authority
sentence of `OP-CNS-001-R0.2`, the Canon and the Formalization Board stands.
Until 2026-09-19 this note (and `docs/FINDINGS_2026-09-18.md` §4.5) called the
OP-PROT-011-versus-012 question open for the operator; the operator had
answered it on 2026-07-25 and the register carries the answer.
