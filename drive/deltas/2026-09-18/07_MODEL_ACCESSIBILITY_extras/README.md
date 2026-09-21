# 07_MODEL_ACCESSIBILITY — publication extras

Copies of objects from one Drive lane, with one `_MANIFEST.jsonl` in this directory. Binding is not review,
replay, endorsement or promotion. Every status word below is transcribed from its source; none of it is
decided here, and nothing in this directory moves a claim, a premise, a gate or a grade.

## 2026-09-20 — Reading_Links.csv and a metadata index of the publication's own objects

### (a) The Drive lane and its folder ids

The lane is the Drive folder **`07_MODEL_ACCESSIBILITY — READING COPIES AND SOURCE MAP`**, id
`1j5vfg98shKLuNVEIUam9KD5TlfdW3kOH`, created 2026-09-17T19:19:27.040Z. `ACCESSIBILITY_HANDOFF.json` names its
five parts under `"folders"`:

| key in the handoff | Drive folder id | live title | contents |
|---|---|---|---|
| `root` | `1j5vfg98shKLuNVEIUam9KD5TlfdW3kOH` | 07_MODEL_ACCESSIBILITY — READING COPIES AND SOURCE MAP | the four folders below plus the START HERE Sheet |
| `text` | `1zhTR6CIDpn6mCu85asOKy5rVWJZ5OEr-` | 01_READING_VOLUMES | 276 text volumes (the source's count) |
| `data` | `1olfvsitpUhxAFD7ltZDzjMBZwGFsZFEA` | 02_DATA_READING_SHEETS | 30 native Sheets (enumerated 2026-09-20) |
| `extracted` | `1WqDPoSMBP_GhExk9HVUTXch1zFfVWltD` | 03_EXTRACTED_DOCUMENTS_AND_IMAGES | 86 files (enumerated 2026-09-20) |
| `audit` | `1QsWOSItihvtQ0Fk9BU1dcPJZniWHczFh` | 04_AUDIT_AND_EXCEPTIONS | 16 files (enumerated 2026-09-20) |

The START HERE Sheet is `1hO3MPQwtAiEjCGdOjZJTbB8fKIvt_3GU6KgxPGTPiLg`, 1,543,180 bytes, created
2026-09-17T23:53:54.606Z, modified 2026-09-18T00:17:50.066Z. It was not fetched. 276 + 30 + 86 = 392, the
published-copy total the sources state.

One observation from the enumeration, recorded and not repaired. The four `HISTORICAL WORKBOOK REGISTRY_*`
sheets all sit in `02_DATA_READING_SHEETS` and all four carry the same `Verification` string in
`Reading_Copies.csv` — *"NATIVE WORKBOOK STRUCTURE READ — historical formulas not recalculated"* — but three are
typed `Kind = xlsx` and `HISTORICAL WORKBOOK REGISTRY_NAVIGATION_EXPORT [0538452c5e1e]` is typed `Kind = binary`.
That single row is why the `Kind` tallies (sheet 26 + xlsx 3 = 29; binary 70 + docx 17 = 87) do not match the
folder counts (30 and 86) while the totals still reach 392. `Kind` may be describing the source carrier's format
rather than the copy's folder, in which case this is not a defect at all; either way nothing is corrected here.
`Reading_Copies.csv` is an export and this repository does not edit it.

**None of these objects has a `drive/inventory.jsonl` row.** The publication was created after the 2026-09-17
snapshot it indexes, so the corpus declares no digest for any of them. The one exception is
`ACCESSIBILITY_COMPLETION_REPORT.md`, whose SHA-256 is declared in `registers/json/work_events.json`, column
`Source SHA-256`, on the `PUBLISH-ACCESS-2026-09-17T23:57:33.664Z` and `RELEASE-ACCESS-2026-09-17T23:57:33.664Z`
events — positions 26 and 27 of that tab's 35-entry `rows` array counting from zero, which are its 27th and 28th
rows counting from one and sheet rows 28 and 29. No other row in the tab carries the digest.

### (b) Controlling status banners, verbatim

From `ACCESSIBILITY_COMPLETION_REPORT.md` (stored in this directory). The first two quotations are its section
*Authority and continuation*; the third is from *Result*; the fourth is the opening line of *Remaining source
exceptions*; the last is the `Historical workbooks` row of the *Result* table, quoted between its cell pipes:

> This work changed accessibility copies and navigation. Original evidence, quarantine holds, mathematical claims
> and scientific gates retain their prior status. Sharing permissions were not broadened.

> A failed source requires a valid replacement carrier or separately labeled recovery evidence; this report does
> not close any research question.

> Automated text APIs omit the native equation contents; the original documents govern mathematical layout. This
> is an accessibility qualification, not evidence that the visible native equations are absent.

> The exception table contains 137 entries of different kinds; it is not a list of 137 failed conversions.

> Historical workbooks | 4 | Native structure/metadata checked; historical formulas were not independently
> recalculated or scientifically validated

**The scoping sentence the task calls the completion report's** is in fact the closing line of
`EXTERNAL_RECON_ACCESSIBILITY.md` (also stored here). Quoted verbatim, from that file and not from the report:

> Disposition: existing supported platform functions reused and validated for this task. No complete external
> mathematical match was sought or inferred because this work concerns storage accessibility, not substantive
> research.

and, from the same file, the last sentence of its *Currency and limits* paragraph:

> No scientific gate, theorem claim, independent review or quarantine disposition changes follow from this
> reconnaissance.

From `Start_Here.csv` (stored here), rows *Current research status* and *Source preservation*:

> Use Research Home for current scientific status. This map is a source-access snapshot; draft, historical,
> quarantine and review scopes remain attached to their sources.

> Original files, source identities, scientific claims and sharing permissions were preserved. Copies remain
> private to the owner.

From `ACCESSIBILITY_HANDOFF.json` (stored here):

> "status": "COMPLETE_FOR_AUDITED_SNAPSHOT_WITH_DOCUMENTED_SOURCE_EXCEPTIONS"

> "claim_status": "RELEASED"

> "preservation": "Original evidence, scientific gates, quarantine and sharing retain their existing status."

> "unresolved_source_exceptions": "See Exceptions.csv or Exceptions tab. Do not interpret caches or DOCX layout
> qualifications as failed conversions."

`COMPLETE_FOR_AUDITED_SNAPSHOT_WITH_DOCUMENTED_SOURCE_EXCEPTIONS`, `RELEASED`, `PUBLISHED / VERIFIED` and
`COMPLETE_WITH_SOURCE_EXCEPTIONS` are the source's and the register's own words about accessibility copies and
folder metadata. They say nothing about any claim, premise or obligation, and nothing here paraphrases them into
one. The handoff's `resume_instructions` are imperative text written by another session for a future session;
they are carried as data and none of them was executed.

### (c) The per-key comparison rule in ACCESSIBILITY_VERIFICATION.json, verbatim

`ACCESSIBILITY_VERIFICATION.json` (stored here, 231,542 bytes) states what was compared per reading copy. It is
**character counts and non-blank-line fidelity, not bytes**, except for the 45 records typed `"raw"`. A complete
text record, verbatim — the file pretty-prints it, and the indentation and the key the `text` object files it
under are the file's own:

```json
    "FOUNDATION_HISTORY_001": {
      "key": "FOUNDATION_HISTORY_001",
      "id": "152ov2BKa2ZI07tgK_Pehzm6SpRw9Nhw7YXYkaSkshMM",
      "url": "https://docs.google.com/document/d/152ov2BKa2ZI07tgK_Pehzm6SpRw9Nhw7YXYkaSkshMM/edit?usp=drivesdk",
      "source_sha256": "f952c356bef82467a49ac55b29a059883c54deb9f2cbd74ff1e3f598eb016fa8",
      "source_chars": 36699,
      "readback_chars": 37310,
      "nonblank_line_fidelity": true
    }
```

Note that `source_chars` and `readback_chars` differ and the record still records `"nonblank_line_fidelity": true`:
the comparison normalises display whitespace. Of the 276 text records, 268 carry `"nonblank_line_fidelity": true`
and 8 carry `false`. A complete binary record, verbatim, again as the file lays it out:

```json
    {
      "key": "89a0d969e2b4fba196e425137d6b5d6c0dc0ca4ac04f469d6e4fa7e61e52b28d",
      "type": "docx",
      "id": "11Zjyeh1CQusa8eP0BfdVUZaB1t5EoEnhZPTqwUCO9BU",
      "url": "https://docs.google.com/document/d/11Zjyeh1CQusa8eP0BfdVUZaB1t5EoEnhZPTqwUCO9BU/edit?usp=drivesdk",
      "source_chars": 1141923,
      "read_chars": 1225024,
      "text_equal_ignoring_whitespace": true,
      "readable": true,
      "math_nodes": 0,
      "qualification": "Native conversion is for reading; original document governs layout. OMML companions preserve equation structure where present."
    }
```

The 90 binary records are typed `"raw"` (45), `"docx"` (41) and `"xlsx"` (4). The `native_boundaries` records
state their own scope verbatim:

> "Native first four and last three rows; complete local workbook cell values verified separately."

> "Native first five and last four rows; source cells checked locally, large-sheet whole-grid checks reused from
> saved verification."

and the source map's own visual check states:

> "Artifact previews inspected for all seven sheets. Browser had no authenticated Google session; native values
> and formatting checked through connected API."

The report's own reading rule, verbatim: *"Consult **Exceptions** and **Reading Copies → Verification** before
relying on a conversion."*

### (d) Where the ten published files live in this repository

`ACCESSIBILITY_HANDOFF.json` lists ten objects under `"publication"`. **Four of them sit in
`drive/source_map/`:** `Files.csv`, `Archive_Members.csv`, `Payloads.csv` and `Exceptions.csv`. The other six —
`ACCESSIBILITY_COMPLETION_REPORT.md`, `EXTERNAL_RECON_ACCESSIBILITY.md`, `ACCESSIBILITY_VERIFICATION.json`,
`Start_Here.csv`, `Reading_Copies.csv` and (as of 2026-09-20) `Reading_Links.csv` — are in this directory.

**`drive/inventory.jsonl` is derived from `Files.csv`**: it is a field projection of that table's 4,456 rows,
keyed by Drive ID, in a different row order. It is not an independent enumeration of the Drive and carries no
digest of its own in any corpus record.

### (e) What was ported on 2026-09-20, and what was not

Ported (rows appended to `_MANIFEST.jsonl`; no existing row or file was touched):

| file | bytes | exact | why |
|---|---:|---|---|
| `Reading_Links.csv` | 1,482,152 | false | the 6,034 source-to-copy links; named explicitly by this lane's instructions. Row 8 of the manifest records the same id tree-only under an earlier lane's 300,000-byte limit; that row stands unchanged and this is an addition to it, not a correction of it. |
| `PUBLICATION_INDEX.jsonl` | — | false | 22 metadata-only rows: the five folder ids, the START HERE Sheet, and the 16 files of `04_AUDIT_AND_EXCEPTIONS` with sizes and creation times. Not a Drive object. The audit gap proposed `drive/source_map/PUBLICATION_INDEX.jsonl`, which is outside this lane's directory, so it is written here. |

Deliberately not ported:

- **The 392 reading copies themselves** (`01_READING_VOLUMES`, `02_DATA_READING_SHEETS`,
  `03_EXTRACTED_DOCUMENTS_AND_IMAGES`): counts only, never fetched. They are derived conversions whose scope the
  source narrows itself (*"only 45 raw copies were byte-compared"*; *"Copies remain private to the owner"*), and
  several are copies of quarantined or legacy sources — two are titled `READING COPY  DO NOT USE` and one
  `READING COPY Copy of Evaluating the Λ-Dialectic Framework … [HISTORICAL OR QUARANTINED]`. Mirroring them under
  `drive/` would launder zero-authority material into a lane that checkers consume.
- **The four `HISTORICAL WORKBOOK REGISTRY_*` sheets and the seven `PICKLE STRUCTURE` sheets** of
  `02_DATA_READING_SHEETS` (up to 8,984,667 B each): superseded register states and opcode renderings. Never as
  register data, never as evidence.
- **`Drive_Structure_Audit_Package.zip`** (`13FNy_E-vptvkKOzoe5IBkxYtoDBuhyel`, 1,188,339 B): its members have no
  inventory rows, so a mirror would be an unindexed carrier. Metadata only, in `PUBLICATION_INDEX.jsonl`.
- **The START HERE Sheet**: metadata only. Its 2026-09-18 tabs belong to the delta lane.
- **`Files.csv`, `Archive_Members.csv`, `Payloads.csv`, `Exceptions.csv`**: already carried in `drive/source_map/`;
  that directory is owned elsewhere and was not touched.
- The register **consumer map** and the `governance/GIT_ADAPTATION.md` rewording named by the audit's ranks 7 and
  4: owned outside this lane.

### (f) What this does not establish

Copying bytes into this directory is not review, replay, reproduction, endorsement or promotion. Nothing here
verifies any of the 392 reading copies against its source, and nothing here validates the publication's own
verification — the source itself states that only 45 raw copies were byte-compared, that text comparisons
normalise whitespace, and that historical workbook checks are structural. No status label moves: the five validity
premises of Theorem D1 v2.2(2) remain OPEN and `D3-LEMMA-RN-UNIF` is not closed. `PUBLISHED / VERIFIED`,
`COMPLETE_WITH_SOURCE_EXCEPTIONS`, `RELEASED` and every other status word above is the source's own and refers to
what the source says it refers to — for this lane, accessibility copies and folder metadata — never to a claim, a
premise or an obligation. The per-folder counts are a 2026-09-20 enumeration through the Drive connector for two
of the three copy folders and the audit folder; `01_READING_VOLUMES` was not enumerated and its 276 is the
source's own figure, corroborated only by other statements of the same source. Nothing in this directory is
imported, executed or read by CI beyond `tools/verify_manifests.py` checking byte counts and digests. Dylan Roy
remains the single final authority for canonical promotion, external release, permanent deletion and machine-root
replacement.

### (g) Reading copies are not the objects

`exact: false` in `_MANIFEST.jsonl` means the stored bytes are a reading copy, not the object: either a
`text/plain` export of a native Google Doc, or a raw file for which no corpus record declares a digest, so the
digest stored beside it was first computed here and is corroborated by no second pass. A PDF rendering of a
document is never a frozen body, and no export in this repository may stand in for one. Rows written before
2026-09-20 in this manifest label raw publication fetches `exact: true`; the `Reading_Links.csv` row added on
2026-09-20 uses `exact: false` for the same class of object, because no corpus-declared digest corroborates the
digest. `PUBLICATION_INDEX.jsonl`, the other row added that day, is a third case: it is not a copy of a Drive
object at all but this repository's own metadata index, so its `exact: false` records that there is no object
for its bytes to be exact to. The earlier rows are left exactly as they were written.
