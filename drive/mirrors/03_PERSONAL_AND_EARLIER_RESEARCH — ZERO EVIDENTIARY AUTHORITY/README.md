# Mirror of `03_PERSONAL_AND_EARLIER_RESEARCH — ZERO EVIDENTIARY AUTHORITY` (the zone notice only)

Drive root lane (23 inventory items) that holds 14 of the 17 pre-current-method
legacy files. This directory holds the lane's zone notice byte-exact and nothing
else — none of the legacy files is mirrored, by design.

| file | identity |
|---|---|
| `00_READ_FIRST — QUARANTINE NOTICE — THIS ZONE HAS ZERO EVIDENTIARY AUTHORITY (FS2 v2.0).md` (2,414 B) | **byte-exact**: SHA-256 `4a11a66b…` and byte count equal the `drive/inventory.jsonl` row (id `1WReBwzGQlyCkZlVeeBWPDuxJkAdXs72_`) |

## The notice, verbatim

"THIS ENTIRE ZONE IS PRE-CURRENT-METHOD LEGACY RESEARCH. ZERO EVIDENTIARY
AUTHORITY." — "Effective 2026-07-29 · Authority: Dylan Roy · GOOGLE DRIVE FRESH
START 2.0" — "**LEGACY MATERIAL MAY GENERATE A QUESTION. LEGACY MATERIAL MAY NOT
SUPPLY THE ANSWER.**" The three rules: "1. **Do not quote** a number, parameter,
coefficient, or derivation from this zone as fact. 2. **Do not treat confident
language here as evidence.** … 3. **Do not carry a result from here into active
q0 work** — in any form, at any confidence level." And the name collision the
notice itself flags: "`02_LEGACY_Q0_ARCHIVE` says LEGACY but is **not** this
stratum — it holds superseded *q0* work under the **current** method. Zero of
the 17 live there. Different ontology, different rules."

## What this is not

The notice is mirrored so that the rule is quoted from its own carrier. Nothing
under this lane may be cited as evidence (`CLAUDE.md` rule 10), and this
repository holds none of its research files.

## 2026-09-20 — the four remaining FS2 v2.0 notices, and an index of the whole lane

Lane folder id `1q8aCmqiM0YaqcEi-QpKrridXyQR5CXnB`; 23 items in the 2026-09-17 inventory.
The zone notice above was already mirrored and is untouched. Added today, byte-exact against
`drive/inventory.jsonl`, each in its own Drive-relative subdirectory with its own
`_MANIFEST.jsonl`:

| file | id | identity |
|---|---|---|
| `00_IMPORTED_TEXT_CORPUS/00_NATIVE_GOOGLE_DOCS/00_READ_FIRST — QUARANTINE NOTICE — THIS FOLDER HAS ZERO EVIDENTIARY AUTHORITY (FS2 v2.0).md` (2,550 B) | `1BadaFtAe3Xau0mpJOjNSoSuZXJXXNFV5` | **byte-exact**, SHA-256 `8467a505…` |
| `00_IMPORTED_TEXT_CORPUS/01_DOCX_SOURCE_FILES/00_READ_FIRST — QUARANTINE NOTICE — THIS FOLDER HAS ZERO EVIDENTIARY AUTHORITY (FS2 v2.0).md` (2,069 B) | `1S13uIR0rmX0aKXDgdghjFOQMJNp1966N` | **byte-exact**, SHA-256 `6db50873…` |
| `01_FRAMEWORK_AND_METHOD_RESEARCH/00_READ_FIRST — QUARANTINE NOTICE — THIS FOLDER HAS ZERO EVIDENTIARY AUTHORITY (FS2 v2.0).md` (2,521 B) | `14Z2c2U9GkGVaxEdhVOJroyWZ0dkWUj39` | **byte-exact**, SHA-256 `fcb9c0a5…` |
| `02_PROMPTS_REQUESTS_AND_REUSABLE_CONTEXT/00_READ_FIRST — PROVENANCE NOTICE — LEGACY PROCESS DOCUMENTS (FS2 v2.0).md` (1,759 B) | `1Dq0WG6ui7d7XNt8Hou5rNM6ExBsMK2Xm` | **byte-exact**, SHA-256 `6bd73743…` |

The root `_MANIFEST.jsonl` gained a `tree-only` index row for each of the other 22 inventory
items, so the root manifest now carries all 23 of the lane's inventory items, one row each: id,
title, snapshot path, inventory bytes and digest where one exists, access status. Across the
lane's five manifests there are 27 rows for those 23 ids, not 23: each of the four notices in the
table above also has its `stored` row in its own subdirectory manifest, and its root row is a
`tree-only` cross-reference carrying `not_stored_reason: MIRRORED_IN_SUBDIRECTORY` and naming the
path that does hold the bytes. No delta row in `drive/deltas/2026-09-18/` touches any of them.

### The banners, verbatim

All four notices carry the zone's authority line and its rule, but each states the
zero-evidentiary-authority header in its own words, so the header is quoted per file. The two
`00_IMPORTED_TEXT_CORPUS` notices head:

> "## EVERY FILE IN THIS FOLDER IS PRE-CURRENT-METHOD LEGACY. ZERO EVIDENTIARY AUTHORITY."

`01_FRAMEWORK_AND_METHOD_RESEARCH` heads "## PRE-CURRENT-METHOD LEGACY RESEARCH. ZERO EVIDENTIARY
AUTHORITY." and `02_PROMPTS_REQUESTS_AND_REUSABLE_CONTEXT` heads "## TWO PRE-CURRENT-METHOD LEGACY
FILES ARE REGISTERED IN THIS FOLDER. ZERO EVIDENTIARY AUTHORITY." All four then carry these two
lines identically:

> "**Effective 2026-07-29 · Authority: Dylan Roy · GOOGLE DRIVE FRESH START 2.0**"
>
> "> **LEGACY MATERIAL MAY GENERATE A QUESTION. LEGACY MATERIAL MAY NOT SUPPLY THE ANSWER.**"

and each closes by explaining why the notice exists at all. All four open that closing paragraph
with the same sentence and reach the same conclusion:

> "**Note:** the connector that executed FS2 could not insert warnings inside the files
> themselves."
>
> "This notice is the warning."

What sits between and after those two sentences differs by file, so the shared wording stops
there. The two `00_IMPORTED_TEXT_CORPUS` notices put a sentence between them, "The files above
carry no internal warning."; `01_FRAMEWORK_AND_METHOD_RESEARCH` puts nothing between them and
adds a pointer after the second, "This notice is the warning. See
`FS2_LEGACY_QUARANTINE_MANIFEST_v2.0.csv`.";
`02_PROMPTS_REQUESTS_AND_REUSABLE_CONTEXT` carries the two sentences and nothing else, breaking
the line after "inside the files themselves." All four do end on the same line:

> "END NOTICE"

The native-Docs notice on Text3: "It asserts *"100% understanding"*, *"fully grasped"*, and
*"100% tested"* throughout, and states parameter values as settled. Those are **self-issued
markers in a generative-era artifact**, not a verification record." — "**Every numerical value,
parameter, coupling and derivative in it is UNVERIFIED.**" — and on the pairs: "The two `Text3`
copies differ by ~12 KB and the two `Texts` copies by ~666 B. **They are not established as
identical** — one of each pair may be a partial conversion. SHA-256 has not been computed. Do not
deduplicate before hashing."

The .docx notice: "These are the authoritative originals of the legacy stratum. Preserve them: no
deletion, no overwrite, no in-place edit. SHA-256 has not been computed for any of them and is not
claimed."

The framework/method notice names its own trap: "*Complexity Minimization* (2025-12-15) states a
**helicity-barrier coefficient** and a claimed linkage to a **Stelle-gravity constant** with
confidence… Three weeks later, `1Cdi88eYQmhcg9wZzLUPX63LGYf3Yai_UEdkIcqKaSM4` (2026-01-04)
independently searched for that coefficient, found no verifiable result in standard databases, and
recorded it **[CONTRADICTED]** *and* **[UNVERIFIED]**… **Read the 2026-01-04 verification record
BEFORE this folder's Complexity Minimization report.** Reading them in the other order imports a
coefficient that its own successor retracted." On the Λ-Dialectic document it adds: "It does not
thereby acquire evidentiary authority. Do not import "epistemic dissipation" as a measured
quantity or Constructal Law as an established objective function."

The provenance notice: "`HISTORICAL_PROVENANCE`. These are **process / prompt documents, not
scientific claims**… That said: they are pre-current-method and hold **zero evidentiary
authority**. Do not treat a procedure described here as the current operating protocol." And on
the third copy: "Equal size is a strong duplicate signal but **not proof**. SHA-256 was not
computed this session and is not claimed. Hash before deduplicating."

### What was deliberately not ported

The lane's corpus bodies: the four `.docx` originals (Text3.docx 1,840,507 B — also over this
port's 1,500,000-byte single-file ceiling — Texts.docx 1,059,835 B, Text2.docx 468,349 B, Request
Refined for Future Use.docx 40,915 B) and the eight native Docs (Text2, Text3 ×2, Texts ×2,
Complexity Minimization, Evaluating the Λ-Dialectic Framework, Request Refined for Future Use;
2,082,778 B, no digests). Not as bytes and not as reading copies: the zone notice forbids quoting
"a number, parameter, coefficient, or derivation from this zone as fact" and says "**Do not carry
a result from here into active q0 work** — in any form, at any confidence level." They appear only
as index rows.

### What this does not establish

Mirroring a warning is not review, replay, endorsement or promotion, and it carries no result out
of the zone. A digest match proves the bytes on disk are the bytes the inventory declared for that
Drive id and nothing more. An index row is metadata, not a mirror; a reading copy would not be the
object, and a PDF rendering is not a frozen body. Nothing under this lane may be cited as evidence
(`CLAUDE.md` rule 10). Per the zone notice's own distinction, `02_LEGACY_Q0_ARCHIVE` is a
different stratum with different rules and is indexed separately under
`drive/mirrors/02_LEGACY_Q0_ARCHIVE — INSPIRATION ONLY ∕ REVERIFY FROM FIRST PRINCIPLES`. No
status, gate, hold or independence credit moves; the five validity premises of D1 v2.2(2) stay
OPEN and `D3-LEMMA-RN-UNIF` is not closed. Nothing stored here is imported or executed, and
nothing here feeds a claim, a bound, a carrier or a grade. It is read, though:
`tools/verify_manifests.py` runs from the repository root with no path argument, so every CI run
opens each stored file in this directory and re-hashes it against its manifest, and
`tools/mirror_quotes_check.py` reads these same bytes as the corpus the quotations above are
checked against. Both only read, hash and compare.
