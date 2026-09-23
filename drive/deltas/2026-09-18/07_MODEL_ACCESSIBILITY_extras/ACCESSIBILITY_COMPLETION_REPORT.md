# Drive accessibility completion report

Verified at 2026-09-17T23:54:59.287Z.

[Open the source map](https://docs.google.com/spreadsheets/d/1hO3MPQwtAiEjCGdOjZJTbB8fKIvt_3GU6KgxPGTPiLg/edit?usp=drivesdk) · [Reading folders](https://drive.google.com/drive/folders/1j5vfg98shKLuNVEIUam9KD5TlfdW3kOH) · [Research Home](https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit)

## Result

The accessibility publication for the inventoried 17 September 2026 snapshot is complete, with the source exceptions below explicitly retained. The source map covers 4,456 Drive items (3,714 files and 742 folders), 77 original archive carriers, 11,649 archive-member occurrences and 4,020 distinct payloads. It provides 6,034 source-to-reading-copy links and 392 published reading copies.

| Published material | Count | Verification scope |
|---|---:|---|
| Text volumes | 276 | Full readable text compared to local source; whitespace-only normalization |
| Word reading copies | 41 | All readable; 33 native text comparisons passed; eight equation-bearing copies require linked companions |
| Exact extracted files | 45 | Full downloaded bytes matched source SHA-256 |
| Generated data sheets | 26 | 23 full local XLSX cell checks plus native first/last checks; three large sheets fully compared in saved chunk receipts |
| Historical workbooks | 4 | Native structure/metadata checked; historical formulas were not independently recalculated or scientifically validated |

The interrupted x9_series repair is complete. Its final five chunks restored 74,332 missing cells and passed full readback with zero differences. Earlier saved receipts complete the remaining ranges. The three large data sheets contain 183,590, 235,798 and 256,561 rows including headers. Their pickle sources were inspected as opcode data without deserialization.

Eight Word copies now carry verified notices linking to six equation-companion volumes. These retain all 973 original OMML equation structures. Automated text APIs omit the native equation contents; the original documents govern mathematical layout. This is an accessibility qualification, not evidence that the visible native equations are absent.

## Use the source map

1. Filter **Files** by title or original path, then open **First reading copy**.
2. For every copy of a source, filter **Reading Links** by the same Source Drive ID.
3. For a ZIP or nested archive, filter **Archive Members** by carrier title. Use its payload hash in **Payloads** to find all reader keys.
4. Reading volumes label source blocks with their SHA-256, original link and full path. Search the hash within the document.
5. Consult **Exceptions** and **Reading Copies → Verification** before relying on a conversion.

All six lookup tables have filters and frozen headers/identifier columns. All 261,229 exported map cells matched their specification; 344 native header, middle and boundary cells matched after import. Artifact previews were inspected on all seven tabs. The browser lacked a signed-in Google session, so native UI rendering was not inspected; connected-API formatting and values were verified.

## Remaining source exceptions

The exception table contains 137 entries of different kinds; it is not a list of 137 failed conversions.

| Type | Entries | Meaning |
|---|---:|---|
| EMPTY_NATIVE_BODY | 8 | Original native carrier had no readable body |
| READ_FAILED | 5 | Source retrieval remained unsuccessful |
| BINARY_UNRENDERED | 1 | Binary payload lacked a readable conversion |
| ARCHIVE_READ_FAILURE | 3 | Archive content could not be fully read |
| ENCODED_BLOCK_FAILURE | 15 | Embedded encoded block did not decode successfully |
| COMPILED_CACHE_RETAINED | 64 | Compiled caches retained as original evidence; not treated as source text |
| DOCX_LAYOUT_AND_EQUATIONS_REQUIRE_SOURCE | 41 | Original Word layout remains authoritative |

Each entry retains source identity, links and details. Overlapping occurrences are not additional independent sources. Empty or damaged evidence was not reconstructed into asserted fact.

## Authority and continuation

This work changed accessibility copies and navigation. Original evidence, quarantine holds, mathematical claims and scientific gates retain their prior status. Sharing permissions were not broadened.

The publication and release are recorded in the coupled registers. The durable handoff includes file IDs, source hashes, verification receipts, flat CSV exports and the source-map construction specification. Future maintenance should use modified-time deltas and targeted source checks against this snapshot, and should update only affected derived copies. A failed source requires a valid replacement carrier or separately labeled recovery evidence; this report does not close any research question.

