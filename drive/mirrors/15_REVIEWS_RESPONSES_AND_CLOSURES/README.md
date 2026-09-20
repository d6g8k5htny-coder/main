# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/15_REVIEWS_RESPONSES_AND_CLOSURES` (partial)

Drive folder of the reviews, responses and closures lane (866 inventory items).
This directory mirrors a curated part of it. Each subdirectory that holds files
carries a `_MANIFEST.jsonl` (one row per object: Drive id, title, path, dest,
bytes, sha256, exact, inventory digest) that `tools/verify_manifests.py` checks in
CI. **Mirroring is not review, replay, endorsement or promotion.**

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
