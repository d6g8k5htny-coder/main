# Port fidelity repair, 2026-09-18

What was done about RV-OPS-R17 finding 1 — that no artifact in this repository
mirroring a Drive object was byte-identical to it — and, more importantly,
what was *not* done, and why. Every number below is reproducible from the
repository by the command named beside it.

The rule this pass ran under, from OP-PROT-019 §2: *the digest establishes
identity, not truth or authorization.* A repository file was overwritten with
downloaded bytes **only** when those bytes hashed to a full 64-hex SHA-256 that
the corpus declares for that object. A prefix match, a byte-count match, or a
transform that "looks right" was not sufficient to write anything. Where no
such digest exists the file was left exactly as it was and the difference is
recorded here instead.

---

## 1. Method

For each of the seven records in `governance/PROVENANCE.json`, in order:

1. The raw bytes of `source_drive_id` were fetched with the Drive connector's
   `download_file_content` (no `exportMimeType`, so native objects came back as
   the connector's default `text/plain` or `text/csv` export and stored files
   came back as themselves). Each result is a JSON object whose `content` field
   is base64; the bytes were decoded from the recorded tool result with a
   script, never retyped, and written under the session scratchpad only.
2. The declared digest was established. Two records held only a 16-hex prefix;
   `drive/source_map/Files.csv`, `Payloads.csv`, `Archive_Members.csv` and
   `drive/inventory.jsonl` were searched for a 64-hex digest beginning with
   each prefix. Both resolved to exactly one full value.
3. The raw bytes were hashed and compared with the declared digest, and with
   the repository file byte for byte.
4. On a full-digest match the file was overwritten with the raw bytes and the
   result re-hashed on disk. Otherwise the file was not touched.

None of the seven downloads was a reading **volume** (no `BEGIN SOURCE`
banners), so `engine/rn_engine/reconstruct.extract()` was not needed.

## 2. The seven outcomes

| # | repository file | declared SHA-256 | raw download | outcome |
|---|---|---|---|---|
| 1 | `governance/protocols/OP-PROT-019-v1.1_R17.md` | `04987ba4…7390` (full; `review_queue.json` row `RV-OPS-R17`) | 14,073 B, `04987ba4…7390` | **REPLACED_BYTE_EXACT** |
| 2 | `governance/protocols/OP-PROT-012.md` | none in corpus | 9,732 B, `5cd6a68e…4b04` (`text/plain` export) | NO_DIGEST_DECLARED |
| 3 | `governance/protocols/OP-GDN-002.md` | none in corpus | 7,546 B, `f20b3b99…18fc` (`text/plain` export) | NO_DIGEST_DECLARED |
| 4 | `governance/protocols/OP-CNS-001-R0.2.md` | none in corpus | 9,102 B, `0044b40c…d5f9` (`text/plain` export) | NO_DIGEST_DECLARED |
| 5 | `docs/FULL_DOCS_MATH_READ.md` | `448adee5…be8d` (full; resolved from prefix `448adee54df714e2`) | 16,108 B, `448adee5…be8d` | **REPLACED_BYTE_EXACT** |
| 6 | `docs/R17_IMPLEMENTATION_REPORT.md` | `7b2f6cc1…779f` (full; resolved from prefix `7b2f6cc16c510529`) | 5,761 B, `7b2f6cc1…779f` | **REPLACED_BYTE_EXACT** |
| 7 | `registers/source/GP-REG-032_v1.2_export_2026-09-17.md` | none in corpus | 9,277 B, `43dc36ee…9142` (`text/csv` of the first sheet only) | NO_DIGEST_DECLARED |

Three of seven are now byte-identical to their objects. Four remain reading
copies. The full digests, byte counts, mime types and dates are in
`governance/PROVENANCE.json` under `raw_download_*` and `raw_export_*`.

```
sha256sum governance/protocols/OP-PROT-019-v1.1_R17.md docs/FULL_DOCS_MATH_READ.md docs/R17_IMPLEMENTATION_REPORT.md
# 04987ba47b58be623a3b2a5a8a21236f43d4393cc04009e87f4a62b59a787390
# 448adee54df714e2f768bac25990a0aae987eb9472385f273bcdabdc5db1be8d
# 7b2f6cc16c510529c5659fe8315d6642bf97b1c4792f670afc479590480c779f
python3 tools/provenance_check.py
```

## 3. What the three replaced files had been

Recorded so the transform this repository applied is on file, not just the
fact that it was undone.

### 3.1 `OP-PROT-019-v1.1_R17.md` — the sharp case

Object: 14,073 bytes, `04987ba4…7390`. Repository copy before repair: 14,073
bytes, `efcfdd5cde32e7bd706f432689f5b022b0a2a8ec289164eed170a61d7a0071d9`.
Same length, different digest — so the two differ by a same-length edit, and
it is exactly this:

| offset | object | repository copy |
|---|---|---|
| 74 (end of the title line) | `\n` then `AI-DRIVE-AUTONOMY-R17 …` | `\n\n` then `AI-DRIVE-AUTONOMY-R17 …` — one blank line inserted |
| end of file | `… Google Drive.\n\n` | `… Google Drive.\n` — one trailing newline dropped |

One byte inserted at offset 74, one byte removed at the end. Every byte in
between was shifted by one, which is why 13,724 of 14,073 byte positions
compared unequal while the content was otherwise untouched. No BOM, no CRLF,
no character substitution. Verified by
`copy == object[:74] + b"\n" + object[74:-1]`.

This is what a markdown-aware writer does to a file whose second line follows
the heading without a blank line, and it is invisible to a byte-count check.

### 3.2 `FULL_DOCS_MATH_READ.md` and `R17_IMPLEMENTATION_REPORT.md`

Both were the object with an HTML provenance comment prepended and nothing
else changed: `copy[386:] == object` and `copy[408:] == object` respectively.

| file | header at commit `10abdc1` | header at commit `a3c37e7` (on disk before repair) | copy | object |
|---|---|---|---|---|
| `FULL_DOCS_MATH_READ.md` | 3 lines, 239 B | 5-line comment + blank line, 386 B | 16,494 B, `0e7c859b…` | 16,108 B, `448adee5…` |
| `R17_IMPLEMENTATION_REPORT.md` | 4 lines, 262 B | 6-line comment + blank line, 408 B | 6,169 B, `15d57d16…` | 5,761 B, `7b2f6cc1…` |

(The +239 / +262 figures quoted in earlier findings were the deltas at the
first commit; the headers were rewritten and lengthened in `a3c37e7`, which is
why the on-disk deltas at repair time were +386 / +408. Either way the delta
was exactly the header.)

The headers carried a Drive id, a byte count, a truncated digest, a date and a
one-line description. None of that is lost: `governance/PROVENANCE.json`
records the Drive id, the now-full digest, the byte count, the extraction rule
and — new — a `description` field holding the one-line summary. No header was
re-added; a header is what made the files non-exact in the first place.

## 4. The four files not replaced, byte for byte

### 4.1 Why none of them could be

Records 2, 3, 4 and 7 are native Google objects (three Docs, one Sheet). The
source map has no payload digest for any of them, and a native object has no
stable bytes to digest: what the connector returns is an *export*, and the
digest of an export identifies that export on that day, nothing more. So no
download could satisfy the rule, and no replacement was made. The raw export
digests are recorded as `raw_export_sha256` so a future run can at least tell
whether the export has changed.

The exports themselves all have the same shape: a UTF-8 BOM (`EF BB BF`),
CRLF line endings, and every blank run doubled — the same display transform
`docs/ENGINE_RECOVERY.md` §1.3 documents for reading volumes. Undoing that
transform (strip BOM, CRLF → LF, halve blank runs) is a hypothesis, not a
verification; it is reported below only to say precisely how each repository
body relates to the export.

### 4.2 `OP-PROT-012.md`

Export: 9,732 B, BOM, 198 CRLF lines. Repository copy: 642-byte HTML header +
9,485-byte body. Relative to the de-transformed export the body differs in
**eight lines**, and not only in form:

| export line | repository body |
|---|---|
| `Status: ACTIVE — sole live approval and decision-routing rule` | `Status: ACTIVE at time of export — sole live …` — **the phrase " at time of export" was inserted into the object's text** |
| six lines containing `’` `“` `”` (U+2019, U+201C, U+201D) | the same lines with ASCII `'` and `"` |
| (no trailing newline) | trailing `\n` |

The inserted phrase is an editorial annotation placed inside the body rather
than in the header, which is exactly the kind of change a reading copy must
not make silently. It stands until someone with a verifiable source replaces
the file; it was not "fixed" here because there is no digest to check a fix
against, and rewriting it would be reconstruction. The object's own Drive
title, `HISTORICAL OP-PROT-012 — APPROVAL ROUTING SUPERSEDED BY R17`, is
recorded in the provenance notes and is the more important caveat.

### 4.3 `OP-GDN-002.md`

Export: 7,546 B, BOM, 111 CRLF lines. Repository copy: 217-byte HTML header +
7,397-byte body. The body **is** the de-transformed export plus one trailing
newline, byte for byte. That is the cleanest of the four, and it still
establishes nothing about the object: the same transform is demonstrably lossy
on other Drive exports (`ENGINE_RECOVERY.md` §1.4), and there is no digest.

### 4.4 `OP-CNS-001-R0.2.md`

Export: 9,102 B, BOM, 193 CRLF lines. Repository copy: 217-byte HTML header +
8,784-byte body. Relative to the de-transformed export the body:

* **omits the export's first two lines**: `Live rule: 00_LIVE_GOVERNANCE —
  Independent-Eyes Rule (OP-PROT-011).` and the blank line after it, so the
  repository copy begins at the record's own title;
* **omits the export's last line**, which is a single character `2` (a stray
  character at the end of the Doc), and ends with a newline instead.

Everything between is the de-transformed export. The dropped "Live rule" line
is content from the object; it is reported, not restored.

### 4.5 `GP-REG-032_v1.2_export_2026-09-17.md`

The connector's default export of a Sheets object is `text/csv` of the **first
sheet only** (`RESEARCH HOME · R17`, 30 rows, 9,277 B). The repository file is
a 2,209,031-byte markdown rendering of the whole workbook. The two are not the
same object at any level and were not compared further. Fidelity for the
registers is enforced downstream by `tools/registers_import.py --check`, which
fails if `registers/json` or `registers/csv` drift from this export. The file
lives under `registers/source/`, which this pass could not edit in any case.

## 5. What this does NOT establish

* **Byte-exactness is identity, not review.** A restored file is now the same
  bytes as the object the register names. That says nothing about whether the
  object is correct, current, authorized or reviewed. Each of the three enters
  at the status its register row carries: `OP-PROT-019-v1.1_R17` is row
  `RV-OPS-R17`, technical status `READY`, independence `EXTERNAL_REVIEW_OPEN`;
  the two reports are author-side documents whose status labels are as their
  sources wrote them. No status changed today.
* **A matching export digest is not an object digest.** The four
  `raw_export_sha256` values identify one export on 2026-09-18. They cannot be
  used to claim exactness later; they can only detect that the export changed.
* **A clean de-transform is not verification.** `OP-GDN-002.md`'s body reducing
  exactly to the export is a fact about two exports agreeing, not about the
  object; `byte_exact` stays `false`.
* **Nothing mathematical.** No claim, obligation or premise anywhere in the
  repository changes because a protocol file is now byte-identical to its
  source.
* **The content divergences in 4.2 and 4.4 are reported, not judged.** Whether
  the inserted phrase in `OP-PROT-012.md` or the dropped "Live rule" line in
  `OP-CNS-001-R0.2.md` matter is for the owner of those reading copies to
  decide against a verifiable source.

## 6. Consequences for other files (not edited by this pass)

* `tests/test_provenance.py` encodes the pre-repair state as fact. As of this
  repair `python3 -m pytest -q tests/test_provenance.py` reports **5 failed,
  6 passed**, and every failure is a test whose premise is now false rather
  than a defect in the record or the checker (`tools/provenance_check.py`
  exits 0 with `artifacts=7 byte_exact=3 count_only_traps=0 problems=0`):
  * `test_nothing_currently_claims_byte_exactness` — three records now do;
  * `test_the_byte_count_trap_is_recorded_not_hidden` — the trap is repaired,
    so `byte_exact` is `true` and the declared digest equals `repo_sha256`;
  * `test_rejects_byte_exact_claimed_without_a_full_digest` — it flips
    `byte_exact` on the first `docs/` record expecting a truncated digest, and
    that record now holds the full digest, which matches;
  * `test_rejects_byte_exact_claimed_against_a_mismatching_digest` — it flips
    `byte_exact` on `OP-PROT-019-v1.1_R17.md` expecting a mismatch, and the
    digest now matches;
  * `test_rejects_a_reintroduced_verbatim_claim` — it plants "ported verbatim"
    about `FULL_DOCS_MATH_READ.md`, which is now genuinely byte-exact, so the
    checker correctly lets the claim stand; the fixture needs a non-exact
    artifact such as `OP-PROT-012.md`.

  The file's own docstring says these are to be updated with the matching
  digest when a port becomes exact, not deleted. They are the test owner's to
  update; this pass owns neither the tests nor the checker.
* `docs/README.md` lines 8–9 still describe the two reports as "reading copy
  (not byte-exact)".
* `docs/FINDINGS_2026-09-18.md` §1.1 states "0 of 18" and the +239 B delta;
  both were true when written and are superseded by this document.
* `reviews/records/REV-OPS-R17-001.*` record the finding as it was found and
  should stay as they are.
