# Systematic source census — 27 September 2026

**Source custody and accessibility only. Scientific effect: NONE.** This supplements the [earlier reconciliation](README.md) and the existing public catalog, not the scientific-status system.

## Established result

Nineteen additional Dropbox sources (ten archives and nine loose files, 56,619,882 bytes) were acquired. Together with retained earlier downloads, the scanner indexed **3,720 file occurrences / 1,553 distinct SHA-256 values**. This is a bounded acquired corpus, **not an account-wide Dropbox census**.

| R2 availability measure | Before this pass | After byte matching |
|---|---:|---:|
| Unmatched rows | 299 | 260 |
| Unmatched distinct hashes | 286 | 256 |

The 39 reconciled rows represent 30 hashes. Twenty-one are the master index and twenty original cold-review DOCX packets inside `Peer_Review_Packets_20_COMPLETE.zip`. A fresh [Drive archive download](https://drive.google.com/file/d/1AjE8jt4fEX0PVWVyBg6qFWVM3NLx3Hiz/view) is byte-identical to Dropbox: 3,511,071 bytes, SHA-256 `3aadcd2fcd3edda0e37f203f5cc0ee5c8970600f7c8ab1fa14a060d5390db44c`. All 21 selected member identities match. This repairs member-level discovery; it does not imply new mathematical acceptance or justify another canonical archive copy.

Nine additional historical SIDE24 files resolve eighteen duplicate-named R2 rows. The original 55-entry package manifest verifies its eight accompanying payloads. The manuscript labels itself the July 30, 2026 noncanonical submission draft. Current Drive contains similarly named documents of different byte sizes. The recovered historical variants are retained separately; they do not replace current proof sources or reproduce the whole 55-entry package.

## Earlier-research intake

The following exact archives were not located in the checked Drive source-map snapshot/targeted title search; exact blob probes in `main` returned 404. Owner-wide default-branch code searches for `master_findings_round15` and `CHPPF` were also empty. These observations do not establish absence from every remote object or branch.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `research_archive_round3.zip` | 10,360 | `579baba1da329c696689f78113793ac3e9a3d85857308e3c5c3a0b70e6f1a58c` |
| `research_archive_round15.zip` | 88,544 | `8895f46c6906a10a460cacb30950589ae77ec9bc5b6bd26513105bd48929f402` |
| `CHPPF_v1.0.1_RC.zip` | 11,767 | `689a6b980ce96528a7d3a2fe29d95cb1e3bd1d87c8f6c4920639ec027c672b03` |

All 24 original members are preserved as individually readable **historical, noncontrolling** intake alongside the three ZIPs. The round archives are internally dated December 18, 2025, despite September 27, 2026 Dropbox uploads. Their derived gravity/correlation claims and mixed round-number headings are not silently adopted or rewritten. CHPPF's performance notebook explicitly uses synthetic data; empty notebook execution outputs and example scripts are not device-trial evidence.

Consensus and a primary APS/arXiv check verified the Lee et al. 2020 bibliographic anchor (DOI `10.1103/PhysRevLett.124.101101`). This is not validation of the archived two-template model, derived mass bound, or current experimental frontier. No legacy notebook or program was executed.

## Reusable inspection tool

[Tool, tests and usage](../../../tools/source_census_20260927/README.md).

The scanner uses signatures rather than filename extensions, handles nested ZIP/TAR/gzip and inert Office XML/math/embedded objects, reads text and binary plists, and optionally extracts PDF text and image metadata. It records unsupported formats rather than treating them as successfully understood. Cloud authentication/acquisition remains outside the tool.

The final census contains 3,267 text occurrences, 70 PDFs, 24 Office documents, 110 binary plists, 44 ZIP containers, one image and 204 unsupported binaries. The unsupported cases are chiefly compiled Python and Notes.app application/interface resources. `Transfer.zip` contains an application bundle; it is not thereby a recovered research-note export. No application was launched or republished.

Hash caching avoided 2,167 repeated parses on the cold pass. A warm replay parsed zero payloads anew and reproduced the inventory except for cache-hit flags, plus identical keyword-hit output. No text truncation or read/container error remained in this acquired corpus. Those results are not full visual review: PDF/Office extraction can lose equation structure, and unsupported binary contents remain semantically unread.

Twenty inert-fixture tests passed in normal and optimized Python. Three deliberate regressions were rejected in both modes, six mutation runs total. Script and test Git blobs were read back against the locally tested bytes. The dedicated PR workflow uses Python 3.11 and standard-library fixtures, not cloud credentials or old research test suites. The full pre-existing repository suite was not run locally; read actual hosted results separately. No nonauthor or independent mathematical acceptance is claimed.

## Remaining queue and evidence

The remaining 256 R2 hashes divide into **242 metadata-only Drive locators** and **14 without a locator in that particular snapshot**. Unacquired loose files, other unenumerated folders and full visual review remain open. The Drive source-map comparison is historical metadata, not a fresh all-Drive census. GitHub comparison is targeted, not every branch.

`TRANSVERSE_CONTACT_ASYMPTOTIC.md` remains unrecovered. Its sole name hit in this acquired corpus is a previous audit referring to it, not the source file. No missing RN carrier, later surrogate or figure was substituted for it. No Vault99 original was opened.

[Drive findings, manifests, remaining queue, historical variants and expanded earlier-research intake](https://drive.google.com/drive/folders/1rVNAeUv3OMeTGTuAd9ul5s6jh2R-9Axt). The final integration receipt there records actual upload/readback and merge state. Detailed private inventories are not bulk-published here.

[Coordination on the existing issue](https://github.com/d6g8k5htny-coder/main/issues/86#issuecomment-5859046490). No source body, theorem verdict, premise, prize, permission or `lemma_closed` value is changed by this pass.
