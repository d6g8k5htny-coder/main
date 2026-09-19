# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-09-15 — KIMI FINAL INTAKE — UPPER2D STAGE E + H5 + ASSEMBLY` (partial)

Drive folder id `1bRmeImIMT-6kc-zBNZVoWW1aYNPBPLWs` (56 inventory items, 47 files). This directory
mirrors the current D1 assembly of the 2D upper track and the two 2026-09-15 Anthropic-side proposals
that sit next to it: 4 byte-exact markdown files, 51,562 bytes.
`_MANIFEST.jsonl` in each subdirectory is checked by `tools/verify_manifests.py`. **Mirroring is not
review, replay, endorsement or promotion.** Ported 2026-09-19.

## Coverage of the lane

| top-level subfolder | inventory items (files) | objects mirrored here |
|---|---:|---:|
| `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15` | 31 (28) | 2 |
| `(files directly in the lane folder)` | 7 (1) | 0 — not ported by this lane |
| `03_H5_PROMOTION_AND_RUNG_CERTIFICATES` | 6 (6) | 0 — not ported by this lane |
| `04_STAGE_E_REVIEWS_AND_DEFECT_FINDINGS` | 5 (5) | 0 — not ported by this lane |
| `01_CURRENT_ASSEMBLY_AND_STATE` | 3 (3) | 2 |
| `02_H4_EVENT_LEVEL_REPAIR` | 2 (2) | 0 — not ported by this lane |
| `06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES` | 2 (2) | 0 — not ported by this lane |

## What is here

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | frozen body: bytes / SHA-256 / rule / matches register |
|---|---|---|---:|---|---|---|
| `1v4z492iAzk5NcOrR47IJHGkIgfsRACpC` | D1_ASSEMBLY_v2_2.md | `01_CURRENT_ASSEMBLY_AND_STATE/D1_ASSEMBLY_v2_2.md` | 20,078 | `7ca114f0b38680d8bb987c097de10f3faf884ae3b05c3ca47215af5df081c174` | true | 18,311 / `490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6` / OP-PROT-017, leading blank lines stripped (= CL-REG-001 marker_strip_LF here) / **true** |
| `1oFNNK4D6BEVuBjTL9jERKynXxnNL_f4t` | D1_ASSEMBLY_v2_2_REGISTER_NOTE.md | `01_CURRENT_ASSEMBLY_AND_STATE/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` | 11,880 | `3ebc3c5ad911b2116fd18767e45260a421438d81df0b72c799c081d5886a97d3` | true | 10,752 / `c1d5e95d97e019574d6b6b09e6e606fd6e0cbc1fe3aa2e0a54953de6e4f2c470` / OP-PROT-017, leading blank lines stripped (= CL-REG-001 marker_strip_LF here) / **true** |
| `1RQE2B3EY9IP5MGZyeAtXBpX4ahY5AhfC` | D1_ASSEMBLY_v2_3_DRAFT.md | `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/D1_v2_3_DRAFT/D1_ASSEMBLY_v2_3_DRAFT.md` | 16,141 | `15053f8bf3ad103d87c4f3bed82c0d5897d86533dbbe2cb365b24bdc0cce7d69` | true | 14,734 / `2ec888daeb9e5b51e3f1cf6b1e502db8a3a9fb93b9ac81dd14c8e3e00da8dec4` / OP-PROT-017 / no declared digest to compare |
| `18R0wwSFHa--ZMdLRV3ElMO0T5u5YD3lk` | H5_ZBAND_CONSUMPTION_2026-09-15.md | `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/H5_ZBAND/H5_ZBAND_CONSUMPTION_2026-09-15.md` | 3,463 | `9445bdd3beba72c7533152aed08b115cccb20a107f6f0bb57c65b97eda6827a9` | true | — |

The frozen-body digests of `D1_ASSEMBLY_v2_2.md` (`490ad6b2…`, 18,311 B) and
`D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` (`c1d5e95d…`, 10,752 B) are the ones `claims/graph.json`,
`docs/RESEARCH_MAP.md` and `docs/FULL_DOCS_MATH_READ.md` cite; they are reproduced from the stored bytes
under the rule the files themselves name (`marker_strip_LF` per CL-REG-001: text strictly between the
`BEGIN_FROZEN_BODY` / `END_FROZEN_BODY` lines, surrounding whitespace stripped, one trailing LF). The
draft's body digest is first computed here; no register declares one. The whole-file digests are the
inventory's.

## The status banners, verbatim

* `D1_ASSEMBLY_v2_2.md`: "**Artifact:** D1-ASM-20260915-v2.2 … This v2.2 is the current strongest form and supersedes all prior status lines." §5: "**Strongest conditional theorem:** Theorem D1 v2.2(2) — 1 − q(r, 6/5) ≤ C·r³ for all 0 < r ≤ r₀ (r₀ existential), conditional on five named validity premises." §3: "**OPEN — VALIDITY premises of Theorem (2):** OBL-D1-PROMOTE (H5 lane + D1; sub-obligations OBL-H5-JETMOD / OBL-H5-ZBAND / OBL-H5-REMOTE-THRESHOLD); D3-LEMMA-RN-UNIF (rung + uniform parts; foundations); PERC-DECAY (B1.dir-far, B2-far, B4.rem; percolation lane); OBL-B1-BRANCH(loop|B1) (branch-control lane; constant-level); **the B4.loc dam-line tube certificate** (… asserted-not-established — scope ruling V2 …)". `claims/graph.json` adds, of the rung: "D3-LEMMA-RN-UNIF is used at the rung only and is precisely stated, NOT closed."
* `D1_ASSEMBLY_v2_2_REGISTER_NOTE.md`: "**Scope:** register actions only — the frozen v2.2 body (490ad6b2…) is NOT altered; everything below takes effect at the next issuance (v2.3)."
* `D1_ASSEMBLY_v2_3_DRAFT.md`: "**Agent:** Claude (Anthropic family) acting for the assembly role while Kimi is dark (2026-09-15 → 09-30). **Status:** PROPOSED." — "v2.2 remains the last Kimi-issued form; this v2.3 becomes the current strongest form only upon operator promotion." — §5: "conditional on **TWO** named validity premises: OBL-D1-PROMOTE, D3-LEMMA-RN-UNIF."
* `H5_ZBAND_CONSUMPTION_2026-09-15.md`: "AUTHOR: Claude (Anthropic) · CLASS: OBL-DISCHARGE (H5 promotion lane, sub-obligation) · STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: OBL-H5-ZBAND OPEN → DISCHARGED (consumption grade) upon promotion; no theorem statement changes; no frozen carrier edited."

The v2.3 draft and the H5_ZBAND note are **proposals**. The status this repository carries for Theorem
D1 v2.2(2) is the register's: five validity premises OPEN, `D3-LEMMA-RN-UNIF` not closed
(`docs/OPEN_PROBLEMS.md`, `claims/graph.json`). Mirroring the draft promotes nothing and discharges
nothing (CLAUDE.md rule 1); only an operator applying the exact licensing predicate moves a gate.

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

Nothing here verifies, promotes, closes, discharges or reclassifies any claim, premise or obligation.
A digest match is identity of bytes; a body digest equal to the one the claim graph cites is identity
of the body, not a verdict on the theorem. The two-premise form of the v2.3 draft is not the
repository's status for D1(2); OBL-H5-ZBAND is not discharged here; the gate files (`d1_falsify_v3.py`,
`d1_falsify_v4.py`) and every carrier the ledgers pin are not mirrored here and were not re-executed.
No 2D bound is composed with the 3D track, no original prize problem is solved and no independence
credit is awarded.

## 2026-09-19 — `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/RN_UNIF_2026-09-16/`

`CL-RNU-001_RN-UNIF_ENGINE_STATUS_AND_CLOSURE_PLAN_2026-09-16.md` (12,291 B,
SHA-256 `488b1b0c…`, Drive `16o9u_KgysJv9lnR93PNXCqc-MdrW9QL6`), byte-exact
against the inventory. Its header: "STATUS: PROPOSED · AUTHORITY: none ·
CANONICAL IMPACT: NONE — the lemma is NOT closed by this document." It is the
source of the `chi2_grad_bound` slack record in `research/slack/registry.py`
("`chi2_grad_bound` = 1.57e14 with χ² = 1.94e-6"; "True |∇χ²| at (5,0) … 1.563e-5
(vs the engine's bound 1.57e14 — 1e19 slack)"; "the adaptive polar certifier is
defined and never invoked"; "its driver is likewise unwritten"). D3-LEMMA-RN-UNIF
stays OPEN; nothing here closes it.
