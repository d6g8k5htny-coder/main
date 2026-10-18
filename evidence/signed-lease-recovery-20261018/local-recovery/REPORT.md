# Exact accepted-target local recovery

Result: **RECOVERED_7_OF_7_EXACT_BYTE_TARGETS**. Finite read-only local recovery completed at 2026-10-18T00:34:15.143446+00:00. All seven requested accepted target identities were found as existing local project files in authorized roots. The original files were read inertly and preserved unchanged. No preimage patching, text replay, source reconstruction, project import, or project execution occurred.

| Opaque ID | Target | Bytes | LF count | SHA-256 |
|---|---|---:|---:|---|
| RECOVERED-01 | Complete historical inventory carrier | 284340 | 7526 | `f3cf37022a189db63e1bd37ea8bb99ee741f542e72d83ac9df5110136cb0b2b9` |
| RECOVERED-02 | Required predecessor manifest | 17576 | 150 | `13d285a52c7d5b373601d6e1a1f6ab3798ae2b260fcfa545a6be4ed5815e74f4` |
| RECOVERED-03 | Accepted S01 contract | 101664 | 268 | `c7ce3f45bca751031d206566c229e5de870bc60c9392aa5e00462a0114be0f9f` |
| RECOVERED-04 | Accepted S01 tests | 179658 | 2769 | `5eaa431716e8b28f0df20fb6d3da9a6e875e60fb0c4c1d57ead1654270c810e5` |
| RECOVERED-05 | Complete accepted PLAN | 74178 | 299 | `315d703aadb51bc2b5bd7412c3a4fb91162132d9f439abb1fd199ab06344a143` |
| RECOVERED-06 | Successor manifest | 3317 | 35 | `9272448f83347d950541748dabb733ce541dcc1a7635a793e74fe7dacd30f63f` |
| RECOVERED-07 | Accepted budget | 6311 | 156 | `9db25a5e4d1c43e17d96fe92de4fa4f0ee33c8e0c13faae506dfe51090241d2c` |

The complete historical inventory carrier was recovered as 7,526 LF / 284,340 bytes. The predecessor manifest is 150 LF / 17,576 bytes; the successor manifest is 35 LF / 3,317 bytes. The exact accepted tests have 2,769 LF and the contract has 268 LF. These observations verify complete raw-byte identities only. No manifest payload coverage or scientific claim follows from finding their manifests.

Source identity and copy checks: 26 selected candidate files were fully read and hashed; 19 candidate paths matched one of the seven requested identities, including duplicate retained accepted sources. Seven unique copies were created with O_EXCL and O_NOFOLLOW. All source path components were checked as real directories, every selected source and copy was a regular single-link file, and each recovery copy was frozen to mode 0444. Each recovered source was reread and matched its first raw hash, and each whole recovery copy literally equaled the reread source. All five supplied intake/reference inputs matched their expected SHA-256 before and after. The recovery directory is mode 0700.

Search scope: SEARCH-01 used rg filename discovery over the two authorized research roots, yielding 433,948 paths and 31,640 keyword-selected metadata rows. SEARCH-02 was a distinct ignored-file-inclusive named packet pass, completed within its 90-second/150,000-path bound in 2.113 seconds with 7,729 metadata rows. Both searches completed with return code 0 and no stderr. Private filesystem locators and commands with local roots are confined to private source tables. Public-safe metadata uses opaque identifiers. This is no claim about existence or absence outside the authorized roots.

Evidence boundary: this recovery provides all seven requested exact target byte strings. It does not newly validate every predecessor/successor payload, accepted source semantics, current source composition, the whole current method inventory, current ownership, custody authentication, or design/runtime readiness. A separately reported matching budget recovery through another authorized route is consistent with the local match but is not used as authority here.

Dependent work remains **HOLD**: implementation LOCKED; harness NOT_PREPARED; partition NOT_DESIGNED; schedule UNADMITTED; CONTRACT_RED NOT_QUALIFIED; organizational independence 0; scientific effect NONE. No foreign owner was released, no future writer/reviewer was reserved, and no source/product/harness/workflow/ref/PR state was written.

Operations were bounded to inert local metadata, raw file reads/copy/hash, JSON/text handling, and rg filename discovery. There was no network, download, Git CLI, Node, Lean, model, container, browser, dependency resolution, test, project/helper/controller/checker execution/import, old failed runtime artifact/log/image/protection/436 retrieval, or session-log search. Only this new recovery child was written.

Personal release: this actor releases only its finite READ_ONLY_LOCAL_DATA recovery allocation after final metadata sealing. No other allocation or project gate is released.
