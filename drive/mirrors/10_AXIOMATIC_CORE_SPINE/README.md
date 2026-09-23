# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/10_AXIOMATIC_CORE_SPINE`

Drive lane of the axiomatic core spine, folder id `1aNpP5Q2et5sAJY9O6E06bAu_PxWTQP_d`,
**41 inventory items** in the 2026-09-17 snapshot. It has two sub-folders:
`00.1_GOVERNANCE_CANON` (`1Cta8_0E60KYBJAMSt9-aBJJM5lIGqLE_`, 19 items) and
`00.2_CANONICAL_MASTER` (`1zOOSrLkoRg0QtbonGaVhMFzZzrDnsBwO`, 21 items).

All 41 appear exactly once across the manifests under this directory: **31 stored,
10 tree-only** (the ten folders, which have no bytes). Of the 31 stored, **21 are
byte-exact** — their SHA-256 and byte count equal what `drive/inventory.jsonl`
declares for that Drive id — and **10 are reading copies**: text exports of native
Google Docs, for which no payload digest exists anywhere in the corpus.
**2,614,033 bytes** are stored.

Sub-folder ids: `1_e7V938L5bKbni1jDI6DbPrpbj25bdAk` (00.1/00),
`1_YLj4gSNbZO0h7irUX3ofW7bZYv__XGh` (00.1/01),
`1CvRve9avU92nkgj0gRBHsRY5naowE1xE` (00.1/02),
`1p999Kg9MlbpEqLjvxUcDw68wiLOh3j3u` (00.1/03),
`15-aST-r0OApeFXoQ_Rh-4TELox2pcG2A` (00.2/00),
`1sFp2nn0iCwZghm210BEtbCkuL8Eh6AcX` (00.2/01),
`1_tO4ilmY68ACpFmSNWNBoWBaF_vU4VlQ` (00.2/02).

## What is here

| sub-path | stored | exact | reading copies | bytes |
|---|---:|---:|---:|---:|
| `00.1_GOVERNANCE_CANON` (its own six native Docs) | 6 | 0 | 6 | 33,266 |
| `00.1_GOVERNANCE_CANON/00_CURRENT_RECONCILED_SPECIFICATION` | 1 | 1 | 0 | 30,769 |
| `00.1_GOVERNANCE_CANON/01_PRIOR_DIVERGENT_V1_1_LINEAGES` | 2 | 2 | 0 | 90,169 |
| `00.1_GOVERNANCE_CANON/02_PDF_EXPORTS_AND_ORIGIN_DOCUMENTS` | 2 | 2 | 0 | 170,279 |
| `00.1_GOVERNANCE_CANON/03_OPERATOR_GUIDANCE_AND_RECOVERY` | 3 | 1 | 2 | 27,407 |
| `00.2_CANONICAL_MASTER/00_CANONICAL_MASTERS_AND_SOURCE` | 6 | 5 | 1 | 2,216,260 |
| `00.2_CANONICAL_MASTER/01_ADJUDICATIONS_AND_ERRATA` | 4 | 3 | 1 | 23,117 |
| `00.2_CANONICAL_MASTER/02_FREEZES_AND_MACHINE_CONTRACTS` | 7 | 7 | 0 | 22,766 |
| lane root (`_MANIFEST.jsonl`: the ten folder rows) | 0 | 0 | 0 | 0 |
| **total** | **31** | **21** | **10** | **2,614,033** |

Nine `_MANIFEST.jsonl` files, verified by `tools/verify_manifests.py`. Each
sub-directory that holds files has its own README; this one carries the
lane-level account.

The ten reading copies are text exports of native Google Docs. The Drive-reported
size the inventory records for a native Doc is not a payload digest and does not
equal an export byte count, so each reading-copy row carries `exact: false`,
`inventory_sha256: null`, and a digest of the export that was computed here for
the first time. The 21 byte-exact rows account for 2,561,865 bytes; the ten
reading copies for 52,168.

No id in this lane appears in `drive/deltas/2026-09-18/PATH_CHANGES.jsonl`, so
nothing here moved after the 2026-09-17 snapshot and every byte-exact row was
checked against a digest no path change contradicts. The lane's ids do appear
elsewhere in the deltas: 31 of the 41 are named by a row of
`drive/deltas/2026-09-18/07_MODEL_ACCESSIBILITY_extras/Reading_Links.csv`, which
records where a reading copy of each object was published rather than any change
to the object, and which declares no digest for anything. Nothing in this lane is under `99_DO_NOT_OPEN`
(`1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`) and none of it was opened.

## The controlling status banners, verbatim

### `00.1_GOVERNANCE_CANON`

**`GATE_FRAMEWORK_MASTER — CURRENT-v1.2-RECONCILED`** carries
"Status:** current successor specification" and, under a heading reading
"Supersedes without overwriting:", lists the v1.0 PDF, the uploaded v1.1
Markdown and the C095 executable-specification line. Its
"Established machinery:** one mandatory SCHEMA preflight, 16 numbered gates,
five typed schema profiles, canonical dependency hashes, and explicit
conditional-debt propagation." is fenced by its own §0.2:
"The machine certifies **tag consistency and arithmetic-contract consistency**.
It does not, by itself, certify tag-to-content fidelity." and by §0.3:
"A gate PASS never means “the theorem is true.”"

**The two prior v1.1 lineages disagree with each other, and the repository does
not choose between them.** The C095 executable lineage also calls itself
"Status:** current successor specification" and records
"Set:** one mandatory schema preflight, 16 numbered gates, five schema field
groups", while the uploaded standalone lineage records
"Status:** current unified set — **13 established gates + 1 schema field**, plus
2 candidate gates staged pending retrodiction (§11, explicitly *not* counted in
the established set)." Three files in this lane therefore describe themselves as
the current gate specification with two different gate counts. Which governs is
a source-register question for an operator; nothing was reconciled here.

**`AO48-OPR-022`** — "STATUS:            RELAY — the text below the rule is the
operator's, unaltered."; "AUTHORITY:         operator (relayed); none added by
the relaying line"; "CANONICAL IMPACT:  NONE by the relay itself; the guidance's
own force is the operator's to assert".

**`OP-GDN-002`** — "Canonical impact: Governance only. Mathematical truth remains
evidence-governed."

**`OP-DIR RECOVERY` (CL-REC-038-v1.0)** — "CANONICAL IMPACT: NONE · AUTHORITY:
none (the recovered text is operator-authored; this wrapper is not)".

**`GP-COR-184-v1.0`** and **`OP-PROT-008-R0.1`** each record
"Canonical mathematical impact: NONE"; `OP-PROT-008-R0.1` adds
"Revision status: ACTIVE PRACTICE FRAMEWORK under OP-PROT-011".
**`OP-GDN-003`** records "Status: ACTIVE NON-VOTING CONTINUITY GUIDANCE, revised
2026-07-24". **`OP-CNS-001-R0.2`** records "Status: OPERATOR-DIRECTED SUBSTANTIVE
ARCHITECTURE; ballot framing removed 2026-07-24". **`OP-RES-001`** records
"STATUS: OPERATOR-APPROVED / SETTLED FOR THE DECLARED RECORDS-INTEGRITY SCOPE."

**`HISTORICAL OP-PROT-011`** opens with a supersession notice of 2026-07-25:
"OP-PROT-012 is now the sole live approval and autonomous-decision routing rule",
and "OP-PROT-011 remains active only as evidence-lineage and independent-eyes
guidance."

### `00.2_CANONICAL_MASTER`

**`00_LEAF_CARD — 00.2 Canonical Master`** — header
"CANONICAL IMPACT: NONE · AUTHORITY: none"; body
"No file in this folder ever establishes closure, promotion, or a numerical
constant."; "Presentation quality and recency are not authority"; and
"Named decimals are not theorem constants; no certified numerical coefficient
exists for the bounding constant or the −1/3 law coefficients (Canon)."

**`Q0_MASTER.md`** states its own live theorem state:
"Qualitative cubic rate (root Q0_CUBIC_RATE_EXISTENCE_C101):** there exists a
finite C_Q0 with 0 ≤ 1 − q(r, 6/5) ≤ C_Q0·r³ for 0 < r ≤ 0.025; hence
**q(r, 6/5) → 1** (root Q0_LIMIT_C101). Grade: internal derived-exact existence;
CORE-CLOSED. SARD-G internal: PROGRAM-GRADE-CLOSED; external:
SPECIALIST-REVIEW-PENDING." and, in the same block,
"NOT-CLAIMED / KILLED (binding):** the numerical displays 4.3, 4.35, 0.8411,
0.84, 0.8501, 0.97, 0.99, 1.01 are not theorem constants (C094 adjudications;
C101 E-9; Q0-SHARP KILLED at C108). 0.946 is a rung-tagged measurement
(PAIR_DEFECT_DIRECT_R0025_B1200), never a truth constant."

**`Q0_C092_FINAL_MASTER.md`** — "Core closure:** zero open nodes." fenced by its
own §0: "“Finished” does **not** mean that every mathematically interesting
extension has been solved."

**`Q0_C108_PORTFOLIO_CLOSE.md`** — "The C093 successor portfolio is terminally
closed."; its project table gives `Q0-SHARP` the terminal state KILLED, and lists
"independent specialist acceptance of SARD-G" as the named external dependency of
Q0-REFEREE, Q0-IV and Q0-B. Of Theorem B's coefficient it records
"No numerical value of \(C_*\) is claimed."

**`MASTER_SET_ERRATA`** — "Live successor statement: existence of a finite C_Q0
only (C101)." and "the lower theorem is NOT PROMOTED pending the
marked-repulsion certificate under the determinant-weighted typed six-pin
pair-Palm law".

**`Adjudication_C094_C108_vs_C024_C047`** — "This instance concurs on every
disposition examined, with the receipts above." and "The single external crux is
unchanged since C046: independent specialist acceptance of SARD-G."

**`CL-CLS-055`** — "Canonical impact: EXACTLY the narrow closure below; nothing
broader."

The five freeze contracts are transcribed in
[`00.2_CANONICAL_MASTER/02_FREEZES_AND_MACHINE_CONTRACTS/README.md`](00.2_CANONICAL_MASTER/02_FREEZES_AND_MACHINE_CONTRACTS/README.md).

## Where a mirrored banner and a register disagree

`Q0_MASTER.md` carries, for Theorem B, "Conditional-MS form PROVEN-HERE; full
Gaussian form program-grade, external review pending." That label is inside the
mirrored bytes and is not transcribed here as current status.
`registers/json/automation_config.json` row `THEOREM_B_CURRENT_STATUS` reads
`CANDIDATE_UNCONDITIONAL_PROVEN_HERE_RETRACTED_EXACT_JACOBIAN_PROVED_CONDITIONAL_B0_PROVED`
under the banner "Do not cite Q0_C104 or Q0_MASTER historical PROVEN-HERE labels
as current proof; preserve exact Jacobian and conditional B0; five analytic
bridges remain open" (GP-AUD-187; OQ-011; HB-043). The register's own words
govern the routing; the mirrored label is held as bytes under that banner.

`registers/json/artifact_index.json` row `Q0_MASTER` records its status as
"CURRENT CONSOLIDATED SPINE" with authority
"Supersedes-for-use 488-file archive under stated precedence". Both are the
register's words, transcribed and not weighed.

`registers/json/file_catalog.json` records the Drive id
`1TO2x7UbCCXTjq0M_pnCvdubbBzJ0EP9w` — `MASTER_SET_ERRATA` in
`01_ADJUDICATIONS_AND_ERRATA` — with routing status "METADATA_ONLY". The
2026-09-17 inventory nevertheless declares a SHA-256 and byte count for that id,
and the bytes fetched here matched both, so the object is stored byte-exact.
Reconciling the catalog's routing word with the inventory's digest is a
source-register question for an operator; nothing was re-routed here.

## Two things this port does not resolve

* `drive/mirrors/16_THEMATIC_RESEARCH_TRACKS/README.md` records that
  `_CHARTER_T5.md` cites a `MASTER_SET_ERRATA.md` at Drive id
  `1vEtbWZgf3jmTaLqe63oRNjul77iAHW9E`, which is in no inventory row and no
  register tab, and notes a *different* id under this lane's path. This port
  stores that different id's bytes. It does not follow that the charter's
  pointer should be re-based onto it; the two are recorded as distinct ids and
  nothing was re-based.
* The three gate-framework lineages above disagree about the established gate
  count. The disagreement is transcribed, not adjudicated.

## What this does not establish

Mirroring is not review, replay, endorsement, promotion or acceptance. Every
status word above is transcribed from the mirrored bytes or from a named source
register, and none of it was decided here; a document that calls itself a
certificate, a freeze, a closure or a terminal state is not made one by being
copied into this repository.

A SHA-256 match is identity of bytes, not truth and not authorization. The 21
byte-exact rows establish that this directory holds the same bytes the
2026-09-17 inventory declares for those ids, and nothing further. The ten
reading copies are not the objects: a text export of a native Google Doc loses
the object's own bytes and carries no declared digest, and the two stored PDFs
are byte-exact renderings, not frozen bodies — their text is not quoted anywhere
in these READMEs, because a compressed PDF stream is not a quotable byte.

Nothing here closes, discharges, reclassifies or grades any claim, premise or
obligation. The five validity premises of Theorem D1 v2.2(2) and
`D3-LEMMA-RN-UNIF` are untouched by anything in this lane. No finite decimal
(4.3, 4.35, 0.8411, 0.84, 0.8501, 0.946, 0.97, 0.99, 1.01) is promoted; the
sources themselves kill or withhold every one of them. Nothing here composes the
2D upper/lower tracks with the 3D lifetime track, and nothing here enters the
prize reconnaissance track in either direction. No carrier was run and no
computation was reproduced. Nothing stored under this directory is imported,
executed, scheduled or tested by anything in this repository, and
`claims/graph.json` names none of these Drive ids.
