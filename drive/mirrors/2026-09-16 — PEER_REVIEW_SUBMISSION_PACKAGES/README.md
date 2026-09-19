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
* "## 3. Standing firewall — This errata note changes no LS-CTL Boolean, eligibility predicate, theorem status, RP status, AO48 operator record, or q0 package status. Any status change requires separate operator adjudication."

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
