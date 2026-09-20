# Index of `02_LEGACY_Q0_ARCHIVE — INSPIRATION ONLY / REVERIFY FROM FIRST PRINCIPLES` (metadata only — no bytes stored)

Drive root lane, folder id `13guGw-6_jgWwczOlvgvR83fBlJiGHz6X`; 346 items in the 2026-09-17
inventory (274 files, 72 folders, 266 of the files carrying a digest, 3,846,890 B in total).
**Not one of those bytes is stored here.** This directory holds a `_MANIFEST.jsonl` with one
`tree-only` row per inventory item and this README. The directory name replaces the lane's `/`
with `∕` (U+2215) because a filesystem refuses the original character; the rows carry the real
Drive path.

Created 2026-09-20.

## The controlling banners, verbatim

The lane's own name is its banner: **INSPIRATION ONLY / REVERIFY FROM FIRST PRINCIPLES.**

`CLAUDE.md` rule 10: "**Nothing under `legacy/`, `quarantine/` or the vault may be cited as
evidence.** Drive paths beginning `02_LEGACY_Q0_ARCHIVE` carry zero evidentiary authority."

And, from the FS2 v2.0 zone notice of the neighbouring lane (mirrored byte-exact at
`drive/mirrors/03_PERSONAL_AND_EARLIER_RESEARCH — ZERO EVIDENTIARY AUTHORITY/`), on what this
lane is **not**:

> "`02_LEGACY_Q0_ARCHIVE` says LEGACY but is **not** this stratum — it holds superseded *q0* work
> under the **current** method. Zero of the 17 live there. Different ontology, different rules."

Those two are the only banners this repository can show you verbatim: CLAUDE.md is in the tree and
the zone notice is mirrored byte-exact one lane over.

## The lane's archive map — read as an export, not stored, so summarised and not quoted

The lane's own archive map, `README — CODE DATA AND CALIBRATION ARCHIVE MAP`
(id `1xwyLI2SsF39PUbuCwCUEF8-VfVFkEXQdzQBia9JAEGU`, updated 2026-07-22), was read on 2026-09-20 as
a text export and **not** stored, and this lane stores no bytes at all, so nothing in this
repository carries its words and nothing below can be checked against a stored byte. What follows
is therefore this repository's own summary of what that export said, in this repository's words
and without quotation marks — not a transcription, and not usable as the map's wording:

* It presents itself as a description of the archive's current non-destructive organisation,
  grouping files by operational role ahead of any later duplicate, supersession, corruption or
  deletion adjudication, and says that placing a file in a folder is a navigation and provenance
  decision rather than a claim that the file is mathematically valid or current.
* Its numbered cautions include that equal byte size is not duplicate proof — duplicate status
  needs byte equality or matching cryptographic hashes after a raw download; that similar
  filenames do not establish supersession; and that pickle files should not be loaded casually,
  but in isolated, version-pinned environments, with recovered symbols and tables compared
  against companion scripts and outputs.
* It describes one of its folders as holding a contextual review queue for code and data
  artifacts that cannot yet be classified safely.
* It asserts that nothing was overwritten and no file permanently deleted.

None of that is evidence of anything, and a summary of an unstored export is weaker still: rule 10
governs, and the map is an object in the lane it describes.

## What the index covers, as the audit groups it

| group | files | bytes (not stored) | folders |
|---|---:|---:|---:|
| `02_LEGACY_MANUSCRIPTS_AND_PROOFS/00_MASTER_SET_V3.2` (10 md) | 10 | 215,993 | 1 |
| `02_LEGACY_MANUSCRIPTS_AND_PROOFS/01_SUPPORTING_MANUSCRIPTS_PDF` (30 PDF) | 30 | 2,478,755 | 6 |
| `02_LEGACY_MANUSCRIPTS_AND_PROOFS/02_CYCLE_DOCUMENTS_AND_FREEZES` (64 cycle documents) | 64 | 347,618 | 18 |
| `02_LEGACY_MANUSCRIPTS_AND_PROOFS/03_PROMPTS_AND_RESEARCH_CONTEXT` | 1 | 31,128 | 1 |
| `03_LEGACY_CODE_DATA_AND_CALIBRATION/00_PYTHON_INSTRUMENTS` (19 instruments) | 19 | 140,474 | 5 |
| `03_LEGACY_CODE_DATA_AND_CALIBRATION/01_JSON_SNAPSHOTS_AND_RESULTS` | 141 | 513,244 | 33 |
| `03_LEGACY_CODE_DATA_AND_CALIBRATION/02_BINARY_AND_TABLES` (the pickle) | 1 | 10,997 | 2 |
| `03_LEGACY_CODE_DATA_AND_CALIBRATION/03_ARCHIVE_INDEX_AND_VALIDATION_RECEIPTS` (7 native receipts) | 7 | 24,824 | 1 |
| `04_REFERENCE_IMAGES_AND_MISC/00_FIGURES_AND_CHARTS` | 1 | 83,857 | 1 |
| lane and section folders | — | — | 4 |
| **total** | **274** | **3,846,890** | **72** |

(The audit's "139 JSON" counts the `application/json` MIME rows; the 141 files in
`01_JSON_SNAPSHOTS_AND_RESULTS` also include one `application/octet-stream` object titled
`lb r0.70 full records.json` and one native Google Doc, `application/vnd.google-apps.document`,
titled `ACTION CARD — LOCATE C047 OBSERVED UPDATE V1 AND VERIFY V2 SUPERSESSION CHAIN`. Both
numbers are right for their scope; the manifest carries the per-file MIME type so neither has to
be guessed. The manifest holds exactly one `text/plain` row in the whole lane, and it is
`AClaude prompt 7 5-07-2026 backstory.txt` under
`02_LEGACY_MANUSCRIPTS_AND_PROOFS/03_PROMPTS_AND_RESEARCH_CONTEXT`, not under this path. Whether
the octet-stream object parses as JSON is not something this repository has checked: its bytes are
not stored.)

Each row carries: `id`, `title`, the 2026-09-17 snapshot `drive_path`, `inventory_bytes`,
`inventory_sha256`, `mimeType`, `access_status`, and any 2026-09-18 delta status. 238 inventory
paths Drive-wide were already stale after the 2026-09-18 restructure, so every path here is
labelled as the snapshot path, not a live one.

## What was deliberately not ported

Everything, as bytes. Specifically, and following the audit's do-not-port list:

* **The Master Set v3.2** (10 md, ~216 KB). Its `00 INVENTORY AND TRACE.md`
  (`1nC_eJ-FLlcUx8CMY-cRlJPrtjivz-V9P`, 37,800 B, digest `3b65dc59…`) states a two-sided rate law
  and "CLOSED" lemma labels **under the superseded method**; stored bodies could be mistaken for
  current status and would sit against the rule-4 firewall.
* **The 19 Python instruments.** `tools/carriers_verify.py` refuses to bind them and a negative
  control proves it. Nothing in this lane is importable or executable from this repository,
  because none of its bytes are here to import or execute. Two checkers do read this directory on
  every CI run: `tools/verify_manifests.py` runs from the repository root with no path argument
  and opens this `_MANIFEST.jsonl`, and `tools/mirror_quotes_check.py` reads both it and this
  README. Neither reads a Drive payload from this lane, because there is none to read.
* **The 30 supporting PDFs**, including "AClaude File 4 of 6 07-05-2026 File4 Pinning Lemma Pair
  Palm.pdf" (`1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA`). A PDF rendering is not a frozen body, and
  `registers/json/relations.json` REL-036/037/087 rest evidence on this object while its
  2026-09-17 path begins `02_LEGACY_Q0_ARCHIVE`. The export must not be edited; the audit proposes
  a `registers/KNOWN_FINDINGS.json` entry for the owner to resolve at the source, and **this lane
  wrote no such entry** — `registers/` is owned elsewhere and untouched here.
* **The 139 JSON snapshots** and **`bigraded tables.pkl`** — the archive map's own pickle caution,
  summarised above.
* **The 7 native receipts**, not even as reading copies: this lane is index-only by instruction.

## What this does not establish

An index row is metadata, not a mirror: it records that a Drive object with that id, title, size
and digest was listed in the 2026-09-17 inventory under that path, and nothing else. It is not
review, replay, endorsement, revalidation or promotion, and it confers no authority on the object.
Nothing under this lane may be cited as evidence (`CLAUDE.md` rule 10); the lane banner requires
that any idea taken from it be rebuilt from first principles under the current method. Every
status word appearing in a row's title — CLOSED, FINAL, MASTER, CERTIFICATE, PROOF — is the
source's own word, transcribed, and moves nothing here. No reading copy of any object in this lane
is stored: the archive map was read as a text export on 2026-09-20 and that export was not kept,
which is why its content appears above as this repository's summary and not as a quotation. A PDF
rendering is not a frozen body. No gate, hold, exclusion or independence credit
changes; the five validity premises of Theorem D1 v2.2(2) stay OPEN, `D3-LEMMA-RN-UNIF` is not
closed, nothing composes the 2D tracks with the 3D lifetime track, and no original prize problem
is solved.
