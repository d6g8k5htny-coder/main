# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 — PEER_REVIEW_SUBMISSION_PACKAGES` (partial)

Drive folder id `185P0tWR23btObvqZu9PgoBt4I13xA-H4` (126 inventory items, 120 files). This directory
mirrors one object: the errata note appended to the sealed second edition of the pre-peer-review
manuscript in `PKG-03`. `_MANIFEST.jsonl` is checked by `tools/verify_manifests.py`. **Mirroring is
not review, endorsement, submission or promotion.** Ported 2026-09-19.

## Coverage of the lane

| top-level subfolder | inventory items (files) | objects mirrored here |
|---|---:|---:|
| `PKG-01 — U2D_CONDITIONAL_UPPER_D1_v2_2` | 43 (43) | 0 — not ported by this lane |
| `00_MASTER_INDEX_AND_ROUTING` | 30 (30) | 0 — not ported by this lane |
| `PKG-05 — SIDE24_3D_FIXED_SCOPE_v2026-08` | 22 (22) | 0 — not ported by this lane |
| `PKG-02 — CERTIFIED_RUNG_r0p05_BRICK` | 9 (9) | 0 — not ported by this lane |
| `PKG-04 — LPW_LOCAL_PATH_LOWER_REVIEW_BUNDLE` | 9 (9) | 0 — not ported by this lane |
| `PKG-03 — PREPEER_MANUSCRIPT_v2_SEP13_WITH_ERRATA` | 7 (7) | 1 |
| `(files directly in the lane folder)` | 6 (0) | 0 — not ported by this lane |

## What is here

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact |
|---|---|---|---:|---|---|
| `17iWY1VlUXsSMGcg1BjJ1iRXlnaoX5Uwg` | 01_ERRATA_AND_CLARIFICATIONS_2026-09-13.md | `PKG-03 — PREPEER_MANUSCRIPT_v2_SEP13_WITH_ERRATA/01_ERRATA_AND_CLARIFICATIONS_2026-09-13.md` | 3,381 | `3df5b5fcb176ed37e34038cb1425bf3010d841c4b0f3dd1d380935e0c4bd24a1` | true |

The sealed second edition itself (body `df39e64904a845da…` per the errata) is **not** mirrored here;
only the errata note is, byte-exact against the inventory.

## The banners, verbatim

* "Appended to the sealed second edition (body df39e64904a845da…; the sealed file remains unchanged). Two clarification items from the author-side reconciliation (LPW_Review_Reconciliation_2026-09-12)."
* "## 1. The "ratified upper chain" is the 3D theorem — scope correction … AO48-OPR-045 ratifies the compact-positive-mark estimate sup(1 − p_r) ≤ C r³ and the ν₃,₂₄(ℓ) = c₃,₂₄ℓ^{−1/3}(1+o(1)) theorem for the normalized periodized Bargmann–Fock field on the side-24 **THREE-torus**, with the explicit firewall "this ratification concerns the SIDE24 3D track only. The 2D q0 program is untouched: … no cross-track inference." … the 2D lower wall (Proposition 3.26, conditional liminf form) on one side, and — on the other side — a MATCHING 2D UPPER THEOREM which is a separate open dependency".
* "## 3. Standing firewall", and under that heading: "This errata note changes no LS-CTL Boolean,
  eligibility predicate, theorem status, RP status, AO48 operator record, or q0 package status. Any
  status change requires separate operator adjudication." Until 2026-09-20 the heading and the
  paragraph were quoted as one span joined by an em dash: "## 3. Standing firewall — This errata
  note changes no LS-CTL Boolean, eligibility predicate, theorem status, RP status, AO48 operator
  record, or q0 package status. Any status change requires separate operator adjudication."

This is the source object behind CLAUDE.md rule 4 and the `tools/claims_check.py` firewall that fails
the build on any composition of the 2D upper/lower tracks with the 3D lifetime track.

## How these bytes got here

Every object was fetched through the Drive connector (`download_file_content`, base64) and the
base64 was decoded from the session transcript straight to disk, so no model retyped any byte.
For a raw file the SHA-256 and byte count were recomputed from disk and had to equal the
`drive/inventory.jsonl` row (2026-09-17 snapshot) or the bytes were not stored; every stored raw
file here passed. A native Google Doc has no payload digest anywhere in the corpus: its text export
is stored as `<title>.export.txt` with `exact: false` (a reading copy), and where the export carries
full-line `BEGIN_*BODY` / `END_*BODY` markers the marker-delimited body digest is recorded and
compared with the digest the registers declare for that Drive id. `_MANIFEST.jsonl` holds one row
per object and `tools/verify_manifests.py` re-checks every digest in CI. Archives are not extracted
and nothing was executed; `<name>.zip.members.txt` is a derived member listing, not a Drive object.

## What this directory does not establish

Nothing here submits, reviews, endorses or promotes the manuscript or any package in this lane. The
errata note is mirrored as bytes; the corrections it states are the source's, transcribed nowhere here
as a status change. The sealed manuscript is not present, so nothing about its body is verified here.
No 2D bound is composed with the 3D track and no original prize problem is solved.


## 2026-09-20 — the digest-bearing remainder: 71 files byte-exact

71 of this lane's 81 unheld digest-bearing items are stored byte-exact, across
PKG-01 through PKG-05, and the 10 left, all over the 65,536-byte store limit,
carry tree-only rows with the inventory's own digest. **Zero mismatches in 71
files.** The lane's digest-bearing gap is now zero.

Each file was fetched through the Drive connector and decoded to disk from the
session transcript, or from the file the harness spills an oversize tool result
to, so no model retyped a byte; a file was written only when its SHA-256 and
byte count already equalled the ones the 2026-09-17 inventory declares, and
every stored file was re-hashed from disk again by the process that wrote its
manifest row.

**NO EXTERNAL RELEASE IS APPROVED.** Holding a submission package submits
nothing, releases nothing and approves nothing. External release is an operator
decision under `governance/`, and the owner is the single final authority for it.
The packages describe themselves in their own words — readiness, sealing,
referee checklists, what is not claimed — and those words are the source's, not
this repository's.

### What this pass does not establish

Nothing about any claim, premise, obligation, gate or theorem, and nothing about
whether any package is complete, correct or fit to send. Every number above
counts files, bytes and digests. A byte-exact copy says these are the bytes the
2026-09-17 inventory declares for that Drive id and nothing more. Assembling a
package is not submitting it, and this repository has submitted nothing.


## 2026-09-20 — 38 reading copies, completing the lane

Every digest-bearing object of this lane was stored byte-exact earlier the same
day. The 38 native Google Docs that remained are now held as text exports.
**They are renderings, not the objects.**

This is a weaker thing than a byte-exact copy, and the difference is not a matter
of degree. A byte-exact copy here was written only because its SHA-256 and byte
count already equalled the ones `drive/inventory.jsonl` declares. **That rule
cannot apply to a native Google Doc**: the corpus declares no payload digest for
one anywhere, so nothing can prove an export and a re-fetch could differ. Each
row carries `exact: false` and `inventory_sha256: null`; its `sha256` and `bytes`
are of the export, computed at store time. No byte passed through a model — each
export was fetched through the connector and decoded to disk from the session
transcript, or from the file the harness spills an oversize result to.

### A routing pointer that leads into the vault

Two addenda in `00_MASTER_INDEX_AND_ROUTING` route a reader to the manifest of
the `99_DO_NOT_OPEN` vault. `24_ADDENDUM — DO_NOT_OPEN vault routing` names one
manifest id under the heading
`Authority for what was vaulted (OUTSIDE vault — read this, not vault contents)`.
`25_ADDENDUM — DO_NOT_OPEN live manifest update` repoints to a different id under
`LIVE MANIFEST (OUTSIDE vault — USE THIS)` and records the first as
`superseded`, explaining the reason on its face:
`Updates 24_ADDENDUM manifest pointer (Drive MCP cannot rewrite Doc bodies)`.

**The id 24 names is no longer outside the vault.** In the 2026-09-17 inventory
it sits at
`99_DO_NOT_OPEN — SUPERSEDED_MIRRORS_DEAD_ENDS_AND_TRAP_COPIES/ZZ_SUPERSEDED — DO_NOT_OPEN_MANIFEST shell (pre-entry-rules)`.
So 24, read alone, points a reader into the vault it tells them not to browse —
which is what the vault's own title calls a trap copy. Both documents are held
here unchanged and **the contradiction is recorded, not repaired**: 24's body is
the source's, and rewriting it is not this repository's to do. 25 is the later
document and says so itself.

What this repository holds reflects that. The live manifest 25 names **is** held,
as a reading copy, at its own inventory path outside the vault. The superseded
one 24 names is **not held and not indexed**, and neither are the other four
vault objects — `00_STOP — leave this vault; read the external MANIFEST` and
three further `ZZ_SUPERSEDED` stubs and copies. Those five are the whole of this
repository's remaining native-Doc gap, and they stay at zero:
`tools/quarantine_check.py` refuses any manifest row that stores bytes from that
lane.

### What this pass does not establish

Nothing about any claim, premise, obligation, gate, closure or theorem, and — for
this class — **nothing about the objects' bytes either.** Assembling a package is
not submitting it. A submission package held here is not submitted, not accepted,
not under review by anyone, and not approved for external release; external
release remains the operator's decision alone. A document in this lane that
records a review earns **zero independence credit** where its author line is the
same provider, and the gates requiring independence stay open. No original prize
problem is solved. The five validity premises of Theorem D1 v2.2(2) remain OPEN
and `D3-LEMMA-RN-UNIF` remains not closed.
