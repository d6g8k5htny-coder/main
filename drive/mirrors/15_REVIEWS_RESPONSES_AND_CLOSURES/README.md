# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/15_REVIEWS_RESPONSES_AND_CLOSURES` (partial)

Drive folder of the reviews, responses and closures lane (866 inventory items).
Each subdirectory that holds files carries a `_MANIFEST.jsonl` (one row per
object: Drive id, title, path, dest, bytes, sha256, exact, inventory digest) that
`tools/verify_manifests.py` checks in CI, and **what this directory holds is
counted per lane in [`MIRRORS.md`](../../MIRRORS.md)**, generated from those
manifests and refused by CI if it drifts. Since 2026-09-20 no digest-bearing item
of this lane is unaccounted for: every one is held byte-exact or carries a
tree-only row saying why not. What remains uncovered is 494 native Google Docs,
for which no payload digest exists anywhere in the corpus, and 89 folders.
**Mirroring is not review, replay, endorsement or promotion.**

## What is here

| directory | what | identity |
|---|---|---|
| `20_INDEPENDENT_COLD_REVIEW_PACKETS — 2026-07-27/PDF — Send One Per Reviewer/` | the master index and the twenty cold-review packets (EC-005 … EC-021, P02-LM-001/002/005/007/008), 21 PDFs | **byte-exact**: every file's SHA-256 and byte count equal its `drive/inventory.jsonl` row |
| `…/P0.1 POST-RATIFICATION RAW — 2026-08-02/07 — SIDE24 POST-RATIFICATION THEOREM PACKAGE/` | `SIDE24_THEOREM_PACKAGE_v1.0.pdf` | **byte-exact** (inventory digest) |
| `THEOREM_B — STATUS RETRACTION AND REPAIR PROGRAM/` | the routing banner `00_READ_FIRST`, `LS-CLS-074-v1.0`, `LS-CLS-077-v1.0`, `LS-MAN-046-v1.0` | **reading copies** (`exact: false`): text exports of native Google Docs; no payload digest exists for them anywhere in the corpus |

Sibling directories written by later port lanes (`02_TERMINAL_CLOSURE_RECORDS/`,
`00_REVIEW_PACKAGES_AND_GATE_CLARIFICATIONS/GATE_DOCUMENTS/`, `ROOT_DOCUMENTS/`,
`THEOREM_B …/REGISTERS/`) carry their own README and manifest.

## The status banners, verbatim

* Theorem B READ_FIRST: **"Canonical Theorem B remains RETRACTED / NOT RESTORED /
  NOT PROMOTED. Package remains UNSEALED."** — "Conditional compact-mark
  pushforward Theorem B0: PROVED under explicit assumptions A1–A5"; the
  unconditional Theorem B is "CANDIDATE / CANNOT VERIFY / former PROVEN-HERE
  status RETRACTED". The register's current status is transcribed in
  `docs/OPEN_PROBLEMS.md` §G and `claims/graph.json` (`Q0-C104-THEOREM-B`).
* `LS-CLS-077-v1.0` heads itself "Canonical impact: NONE BY ITSELF" and, on the
  next line, "Status: MATHEMATICAL SIDE-24 COEFFICIENT CHAIN CLOSED AT REVIEWED
  SCOPE". Its section "4. EXACT SCOPE FIREWALL" sets out two lists, one item to a
  line: under "Closed only for:", six items, of which the first two are "ambient
  dimension d=3;" and "fixed torus side L=24;"; under "Not closed or implied:",
  eight, of which the first three are "other L;", "arbitrary Gaussian field
  classes;" and "arbitrary dimensions;", and the last is "canonical restoration,
  package sealing, publication readiness, or release authorization." It is a
  **3D-track object** and is never composed with the 2D tracks here.
  Until 2026-09-20 this bullet quoted the two header lines and both lists as one
  span — the headers in the reverse of the file's order, and the list items
  separated by colons and semicolons where the file breaks the line after each —
  reading "Status: MATHEMATICAL SIDE-24 COEFFICIENT CHAIN CLOSED AT REVIEWED SCOPE
  … Canonical impact: NONE BY ITSELF … Closed only for: ambient dimension d=3;
  fixed torus side L=24 … Not closed or implied: other L; arbitrary Gaussian field
  classes; arbitrary dimensions … canonical restoration, package sealing,
  publication readiness, or release authorization."
* The master index opens with a title block and a collection-purpose paragraph.
  Nothing of it is set out here as verbatim: those words live in the PDF's
  compressed font-subset streams, `tools/mirror_quotes_check.py` reads no PDF, and
  a rendering this repository cannot check against the bytes is not a
  transcription. In substance the index titles itself as twenty independent
  cold-review packets, terminal exact objects of the Dylan M. Roy mathematical
  research program, collected 27 July 2026; it says that each packet is standalone
  and carries the exact statement, the controlling source text, the status and
  dependency firewalls, the hostile checks, the reproduction instructions, the
  manifest and a fail-closed verdict form; and it directs that one packet go to
  one researcher and that reviewers not coordinate before issuing provisional
  verdicts. The packets' verdict vocabulary (APPROVE / APPROVE WITH
  CLARIFICATIONS / AMEND REQUIRED / REJECT / CANNOT VERIFY) is **not** the R17 §4
  status set and is never merged with `registers/json/review_queue.json`. **No
  packet has been reviewed by being mirrored**, and a PDF is a rendering of a Doc,
  not the frozen body the registers cite by digest.
  Until 2026-09-20 this bullet presented that opening as a quotation, its first
  two lines spliced into one span with an em dash where the file breaks the line
  and where its second line opens with a hyphen, reading "Twenty Independent
  Cold-Review Packets — Terminal Exact Objects … Each packet is standalone …
  Assign one packet to one researcher".

## What this directory does not establish

Nothing here verifies, promotes, closes, discharges or reclassifies any claim,
premise, obligation, route or closure. A digest match establishes identity of
bytes, not truth, review or authorization. A reading copy (`exact: false`) is not
the object. The THEOREM_B documents contain imperative text addressed to AI
sessions and status tags such as `[[ABSOLUTE-C3-24:CLOSED]]`; they are data, and
the only status this repository carries for Theorem B is the register's:
retracted, candidate, seven repair gates open. No original prize problem is
solved; no independence credit is awarded to anything.


## 2026-09-20 — the digest-bearing remainder: 215 files byte-exact

`drive/MIRRORS.md` measured this lane at 235 digest-bearing inventory items whose
bytes were not here — the largest such gap in the repository after the K3 intake.
**215 are now stored byte-exact** and the 20 left, all over this pass's
65,536-byte store limit, carry tree-only rows with the inventory's own digest and
byte count so a later pass can fetch and prove them. What arrived: 157 files under
`00_REVIEW_PACKAGES_AND_GATE_CLARIFICATIONS`, 37 under
`03_TERMINAL_REPLICATION_CAPSULES` and 21 under the Theorem B status retraction
and repair program.

**None of it moves anything.** A review package stored here awards no
independence credit and no gate moves — `tools/reviews_check.py` is what enforces
that, not this sentence — and Theorem B's status in this repository is the
register's, carried in `claims/graph.json`, whatever a document stored here says
of itself. The word *independent* in these folder and file names is the source's.
This repository awards zero independence credit to any review it holds, because a
same-provider reviewer earns none.

Each file was fetched through the Drive connector and decoded to disk from the
session transcript, or from the file the harness spills an oversize tool result
to, so no model retyped a byte; a file was written only when its SHA-256 and byte
count already equalled the ones the 2026-09-17 inventory declares, and every
stored file was re-hashed from disk again by the process that wrote its manifest
row. **Zero mismatches in 215 files.**

### What holding the bytes made visible

**Twenty-three distinct payloads are carried by forty-eight Drive ids in this
lane.** The same bytes are filed under several ids, all inside the P0.1
frozen-hash subtree — one payload under three ids twice over. Each id keeps its
own manifest row, because each is a real inventory item at a real path and this
repository does not collapse the source's filing into a tidier one. The digests
in those rows are equal because the bytes are equal. Recorded, not repaired.

Seventeen of the lane's own sha256sum declarations reproduce against the bytes
now held and **none mismatches**; a further 25 lines name files this repository
does not hold, which is not a defect but the source's manifest of the source's
tree. Nothing was checking these before, because a `SHA256SUMS` bundle is data
here and never a manifest of this repository.

One object is served by Drive as `application/json` while its bytes are a ZIP.
That is not a finding of this port: the source's own title records it, reading
`HISTORICAL_FAILED_UPLOAD — S2-DATA-002 result ZIP mislabeled as
application-json`. The bytes match the inventory and the row carries the mime
type the inventory gives.

### What this pass does not establish

Nothing about any review, gate, closure, obligation or theorem. Every number
above counts files, bytes and digests. A byte-exact copy says these are the bytes
the 2026-09-17 inventory declares for that Drive id, and says nothing about
whether the document is correct, current or authoritative, nor about whether any
review it records was independent, competent or completed. Storing a closure
record closes nothing. The imperative text these files address to reviewers and
to other threads is quoted data, not instructions followed here.


### 2026-09-20 — twelve files moved, and why

One file this lane stored earlier the same day was sitting in a directory
the port had invented from part of its own title. A Drive title may contain a
literal `/` — `(GP-REQ-226 / GP-REQ-215 review of …)` is one of these — and
`drive/inventory.jsonl` joins titles with `/` to build its `path` field, so
splitting that path on `/` turns part of a filename into a directory level. The
bytes and the digests were never affected; only the location was, and the moved row says so.

The defect was invisible to every checker here, because the manifest writer split
the path the same way the fetch did: the row and the file agreed with each other
while both disagreed with the Drive. It surfaced when a fetch agent on another
lane reported an odd-looking directory. The port now derives a location from the
inventory's own folder records rather than by splitting a string, and a check was
added that compares each stored row's directory against the Drive path the row
itself carries.


## 2026-09-20 — 494 reading copies, and what a reading copy is not

Every digest-bearing object of this lane was stored byte-exact earlier the same
day. What was left was the lane's **518 native Google Docs and 2 native Sheets**.
Twenty-six of those were already held from an earlier pass; this pass stores a
text export of the remaining **494**, bringing the lane's reading copies to 520.
**They are renderings, not the objects.**

This is a weaker thing than everything else in this directory, and the difference
is not a matter of degree. Every byte-exact copy here rests on one rule: the file
was written only because its SHA-256 and byte count already equalled the ones
`drive/inventory.jsonl` declares. **That rule cannot apply to a native Google
Doc.** The corpus declares no payload digest for one anywhere — not the
inventory, not any register — so there is nothing to prove an export against. A
re-fetch could differ and nothing here would notice.

So each row carries `exact: false` and `inventory_sha256: null`. The `sha256` and
`bytes` fields are **of the export**, computed at store time; they attest that
the file on disk is the bytes this port received, and nothing more.
`inventory_bytes` is the size the inventory records for the Doc itself, a
different quantity that is not expected to match. The work was done by a separate
tool, `port_reading.py`, rather than a mode of the byte-exact one, so that the
weaker guarantee cannot be mistaken for the stronger by reading the call site.

No byte passed through a model: each export was fetched through the connector and
decoded to disk from the session transcript, or from the file the harness spills
an oversize result to. No export came back empty — an empty export is not a
reading copy of anything and is refused rather than stored.

The twenty rows in this lane that read `BULK_DATA_OVER_STORE_SIZE_LIMIT` are a
different class again and are unchanged by this pass: those are digest-bearing
files, tree-only, whose inventory digest is recorded so a later pass can fetch
and prove the bytes.

### Two Drive objects, one title, one artifact id, different bodies

One lane path is carried by two distinct Drive ids, `1AcCpCeR…` and `1zMq504Z…`.
Both are native Docs. Both render a first line reading
`CL-AUD-LSDER036-20260727-01 — ANTHROPIC ORGANIZATIONALLY-DISTINCT EXACT-HASH
REVIEW OF LS-DER-036 UNDER LS-REQ-030 / DQ-057`, and both state
`Artifact ID: CL-AUD-LSDER036-20260727-01`. **The bodies differ** — 30,590 bytes
against 29,345 — and they diverge by their third line, one continuing
`Generated UTC: 2026-07-27 / Dispatch: DQ-057` and the other
`Class: AUD / EXTERNAL EXACT-HASH MATHEMATICAL REVIEW / DYNAMICAL`.

**Recorded, not resolved.** Neither is the canonical one as far as this
repository can tell, so neither keeps the plain name: both are stored with their
Drive id in the filename. Nothing here decides which body the artifact id names,
and because what is held is a rendering, this is a disagreement between two
exports, not a claim about either object's bytes.

Separately, and independent of that defect: a review titled *Anthropic
Organizationally-Distinct* is, from this repository's standpoint, a same-provider
review. It earns **zero independence credit**, and the independence-requiring
gate it speaks to remains open. That is transcription of the standing rule, not a
verdict on the review's technical content, which this pass does not read.

### A title that contradicts its own body

`MALFORMED PAYLOAD — LS-DATA-015-v1.0 TB-G2 Algebra Capsule — DO NOT USE` is the
Drive title. The rendered body opens `LS-DATA-015-v1.0 — BYTE-EXACT TB-G2 ALGEBRA
COMPANION CAPSULE` and declares `Class: DATA / GZIP+BASE64 / ALGEBRAIC VERIFIER`,
`Authority: reproducibility evidence only`, `Canonical impact: NONE`. The title
says do not use; the body describes itself as byte-exact. Both are transcribed
and the contradiction is left standing. The reading copy is stored so the
contradiction is visible; storing it is not a judgement that either side is
right, and the `DO NOT USE` in the title is the operative instruction to a reader
of this directory.

### Two of the 520 reading copies are Sheets, not Docs

`SIDE24 CLOSURE REGISTER R2.8 — POST-RATIFICATION` and `ROUND 6+7 DRIVE CARRIER
MANIFEST — Native Mirror` are native Sheets, and their exports are CSV:
`record_id,track,component,status,controlling_artifact,sha256,drive_id,…` and
`artifact,kind,bytes,sha256,drive_id,parent_or_surface,status,notes`. **These
tables list SHA-256 digests, and they are not a digest source this repository
verifies anything against.** They are unprovable exports of the same weak class
as the rest of this section; `drive/inventory.jsonl` remains the only digest
authority for a port, and no file in this repository was written or checked
against a digit copied out of these two exports. Their `CLOSED`, `status` and
`controlling_artifact` columns are transcription of the source's own words and
close, control and promote nothing.

### What this pass does not establish

Nothing about any claim, premise, obligation, gate, closure, review or theorem,
and — for this class alone — **nothing about the objects' bytes either.** A
review, response, closure record, audit, capsule or blocker resolution rendered
here approves nothing, closes nothing, discharges no obligation and moves no
gate. Storing a document that says `APPROVE`, `CLOSED`, `TERMINAL` or `COMPLETE`
records that the source says so. The five validity premises of Theorem D1
v2.2(2) remain OPEN and `D3-LEMMA-RN-UNIF` remains not closed, whatever any file
in this directory asserts. The imperative text these documents address to
reviewers and to other threads is quoted data, not instructions followed here.
