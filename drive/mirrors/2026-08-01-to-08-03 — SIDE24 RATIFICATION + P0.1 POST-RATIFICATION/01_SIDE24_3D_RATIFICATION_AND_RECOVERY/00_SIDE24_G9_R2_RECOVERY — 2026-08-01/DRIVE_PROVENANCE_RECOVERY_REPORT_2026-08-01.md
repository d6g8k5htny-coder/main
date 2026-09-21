# SIDE24 / EC-019 / C030-C031 provenance recovery report

Date: 2026-08-01 (America/Chicago)

Scope: read-only Google Drive discovery and retrieval, followed by local integrity checks. No Drive file or folder was created, edited, moved, renamed, or deleted during this sweep.

## Controlling findings

1. `AO48-REC-034` is present and fully retrievable at Drive ID `12pnNmP7Uznn8p6km2c5r74qhxlShC3MQ`.
2. `AO48-AUD-030` is present and fully retrievable at Drive ID `1M9lVO7FYuZcf-NzJQTo2VaFg2xmnXIC3`.
3. The alleged-missing R2 archive is not missing. `SIDE24_PRE_REVIEW_2026-07-31.zip` is present at Drive ID `19ThSFctrr7lz0B7YNuaqLFOQaGx8E_UR`; its companion checksum is exact, the downloaded archive hashes correctly, and `unzip -t` reports no errors.
4. The complete ten-file C030/C031 LB-RATE upload set is now present in Drive folder `1RvyeNGThLxtGsnTLREhgw_bT5NEpw8XM`. All ten files were downloaded; all six JSON files parse successfully. The four markdown files with older July counterparts are byte-identical to those counterparts.
5. The full G.9 supplement and `verifier/v8/verify.py` do not surface as standalone Drive files. They are, however, recovered from the operator-supplied V5 archive in the shared workspace. Their hashes match the V5 manifest, and verifier v8 reruns successfully: 30/30 checks, `ALL_ASSERTIONS_PASS`, exit status 0.
6. `AO48-REC-035`, created after `AO48-REC-034`, records the C030/C031 filing and two V3.4 relays. Its statement that R2 “remains their upload blocker” is contradicted by direct Drive listing and byte-level recovery of R2.

## R2 archive recovery

| Object | Drive ID / URL | MIME | Bytes | Created UTC | Modified UTC | Parent |
|---|---|---:|---:|---|---|---|
| `SIDE24_PRE_REVIEW_2026-07-31.zip` | [`19ThSFctrr7lz0B7YNuaqLFOQaGx8E_UR`](https://drive.google.com/file/d/19ThSFctrr7lz0B7YNuaqLFOQaGx8E_UR/view?usp=drivesdk) | `application/zip` | 2,615,005 | 2026-07-31 13:03:25.653 | 2026-07-31 19:56:27.975 | `1A1oAXQAwJl66QNs2NtEb0-TSBhwHyMun` |
| checksum companion | [`1l2xaXz2rGCUER6O0MSwkZhvrmSXzcFX3`](https://drive.google.com/file/d/1l2xaXz2rGCUER6O0MSwkZhvrmSXzcFX3/view?usp=drivesdk) | `text/plain` | 99 | 2026-07-31 13:03:30.221 | 2026-07-31 19:56:34.222 | same |
| parent folder, `SIDE24_PROFESSOR_PRE_REVIEW_PACKAGE_2026-07-31` | [`1A1oAXQAwJl66QNs2NtEb0-TSBhwHyMun`](https://drive.google.com/drive/folders/1A1oAXQAwJl66QNs2NtEb0-TSBhwHyMun) | Drive folder | — | 2026-07-31 13:02:59.921 | 2026-07-31 13:02:59.921 | `PKG-SIDE24-001 — External Peer-Review Capsule`, ID `1youJIIAXXRLGdwH396kNheVrnYdqYpx0` |

Companion SHA-256 and independently recomputed SHA-256:

`14eaf7d403cd8c103576807719e8ee031cad1fc5bba58f1c37574a790849e44b`

Materialization method: authenticated Drive `fetch` with raw-file download produced a time-limited file reference; its bytes were streamed with `curl` into the local recovery directory. No Drive mutation occurred. Local recovered path:

`/workspace/scratch/4c1cbf90475f/work/drive_recovery/SIDE24_PRE_REVIEW_2026-07-31.zip`

Integrity: `unzip -t` completed with “No errors detected.”

## C030/C031 ten-file set

Parent folder: [`C030-C031 LB-RATE CYCLE SET — operator upload 2026-08-01 — AO48 relay`](https://drive.google.com/drive/folders/1RvyeNGThLxtGsnTLREhgw_bT5NEpw8XM), Drive ID `1RvyeNGThLxtGsnTLREhgw_bT5NEpw8XM`, created 2026-08-01 23:50:34.755 UTC, parent `My Drive` (`0AGEUF_sx7o_MUk9PVA`).

| File | Drive ID / URL | MIME | Bytes | Created UTC | SHA-256 of downloaded raw bytes |
|---|---|---|---:|---|---|
| `C030_Observed_Update.json` | [`1simrccXs6H6ojLxfjZ58aIkaVESmZjpt`](https://drive.google.com/file/d/1simrccXs6H6ojLxfjZ58aIkaVESmZjpt/view?usp=drivesdk) | `application/json` | 1,269 | 23:50:47.231 | `03490fe16e4e797f83792b2da758939874fd98a0b07741d62eb0e029d64af2f9` |
| `C030_Freeze.md` | [`107s211Ai5klT6jkOFUQcLdyaQlpJO4Hw`](https://drive.google.com/file/d/107s211Ai5klT6jkOFUQcLdyaQlpJO4Hw/view?usp=drivesdk) | `text/markdown` | 2,673 | 23:51:06.284 | `ae20f4e3a239b17a6b747eba881d18cc7cf8e51e4d6a86eaefc40d9504fdef4b` |
| `C030_CountingLemmas_Package.md` | [`1LuA6EOn6VCOrQcAKywWmMiLZDbihcw7G`](https://drive.google.com/file/d/1LuA6EOn6VCOrQcAKywWmMiLZDbihcw7G/view?usp=drivesdk) | `text/markdown` | 3,442 | 23:51:26.887 | `aed6683edaed6d483704a63cd321fcb300e329a26bf6c143042ead58a60bfc7b` |
| `c030_lemmas.json` | [`1vGzLfLPiM5so9uFg-sViCzdDhEy57R78`](https://drive.google.com/file/d/1vGzLfLPiM5so9uFg-sViCzdDhEy57R78/view?usp=drivesdk) | `application/json` | 2,418 | 23:51:48.316 | `5afabb604b35b28ce3f89a99859aa08887d7c4579d563ecd86deb4635afbcf19` |
| `c030_assembly.json` | [`1MqScq98RMS0g6N1fNsVj8JLC2r48R7YQ`](https://drive.google.com/file/d/1MqScq98RMS0g6N1fNsVj8JLC2r48R7YQ/view?usp=drivesdk) | `application/json` | 131 | 23:51:54.191 | `49e197e978d950ac356ec96866760a7d0d8337341415e501b6ac7905dcd69990` |
| `c030_uncond.json` | [`1R7ZD74LtD8kRQIUE60px7-XZ6h6NoXgA`](https://drive.google.com/file/d/1R7ZD74LtD8kRQIUE60px7-XZ6h6NoXgA/view?usp=drivesdk) | `application/json` | 307 | 23:52:00.966 | `2f0f5802c9577c0e6dc7db910c66b378b6fabd40ee9b543487cc6d682b642e39` |
| `C031_Freeze.md` | [`1JY0476_mMC7gT3D6b1S7RZV9ELjM7R10`](https://drive.google.com/file/d/1JY0476_mMC7gT3D6b1S7RZV9ELjM7R10/view?usp=drivesdk) | `text/markdown` | 1,739 | 23:52:13.421 | `e165821b1479f619af8ebdb1438460edfebe88f44c189553ca86f73f8fe1afbe` |
| `C031_LBRATE_Integration.md` | [`1oSF2QBCE_4IA_DiHc3wmTUy6qaGubILS`](https://drive.google.com/file/d/1oSF2QBCE_4IA_DiHc3wmTUy6qaGubILS/view?usp=drivesdk) | `text/markdown` | 13,690 | 23:53:18.765 | `e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e` |
| `c031_verify.json` | [`1zfsIDNzNlk4sym_H_KENAFH10ydni8IZ`](https://drive.google.com/file/d/1zfsIDNzNlk4sym_H_KENAFH10ydni8IZ/view?usp=drivesdk) | `application/json` | 1,603 | 23:53:34.934 | `83313c96d1610130451a7e936245a06e80d8cc84df00a20ae9e46e7aeef2a271` |
| `C031_Observed_Update.json` | [`1IxX384UzSknv5Oy_VgXTM6Sq-lOvXIAf`](https://drive.google.com/file/d/1IxX384UzSknv5Oy_VgXTM6Sq-lOvXIAf/view?usp=drivesdk) | `application/json` | 1,331 | 23:53:46.509 | `5df4205f2f496f28668c7f1ea22e2a1af9cde00a72cb39a6956b20b32c995063` |

All modification timestamps equal the displayed creation timestamps. All ten were raw-download retrievable. All six JSON files pass `jq -e .`.

Byte-identity cross-checks against older July Drive carriers:

| Current relay file | Older Drive carrier | Result |
|---|---|---|
| `C030_CountingLemmas_Package.md` | `C030 CountingLemmas Package.md`, ID [`1b41EtrlKrs2EdcBvHWKHBJjc077nz6K6`](https://drive.google.com/file/d/1b41EtrlKrs2EdcBvHWKHBJjc077nz6K6/view) | byte-identical |
| `C030_Freeze.md` | `C030 Freeze.md`, ID [`1bwiVzrfNPRU61XmeqYYxLOcCaXV9f00M`](https://drive.google.com/file/d/1bwiVzrfNPRU61XmeqYYxLOcCaXV9f00M/view) | byte-identical |
| `C031_LBRATE_Integration.md` | `C031 LBRATE Integration.md`, ID [`1BeXbrjY6fnpvuj7ksYAcnaBdozKjGqki`](https://drive.google.com/file/d/1BeXbrjY6fnpvuj7ksYAcnaBdozKjGqki/view) | byte-identical |
| `C031_Freeze.md` | `C031 Freeze.md`, ID [`1ilxPY5KhfRkW1TM1lC8wKNijV6eY1GAA`](https://drive.google.com/file/d/1ilxPY5KhfRkW1TM1lC8wKNijV6eY1GAA/view) | byte-identical |

This strengthens `AO48-REC-035`: its stated concern that byte identity was uncertified is now discharged for the four markdown files that have directly located July counterparts. It remains unadjudicated for the six JSON files because no older raw carriers were located by the available search path.

## AO48 / EC-019 core provenance

| Role | Drive ID / URL | MIME | Bytes | Created UTC | Parent | Raw/content retrieval | SHA-256 of raw download where applicable |
|---|---|---|---:|---|---|---|---|
| AO48 audit of EC-019 T1 | [`AO48-AUD-030`](https://drive.google.com/file/d/1M9lVO7FYuZcf-NzJQTo2VaFg2xmnXIC3/view?usp=drivesdk), `1M9lVO7FYuZcf-NzJQTo2VaFg2xmnXIC3` | `text/markdown` | 11,310 | 2026-07-25 13:08:06.402 | `04_REVIEW_AND_FALSIFIERS`, `1LkaebDSyKPC4UhjQAtn5GMMsBNXV7jIE` | yes | `3c6493b6979da59e322bfb0e83d9356f70b287e81a48e4b545ae90835ecc9e91` |
| audited source | [`GP-DER-161-v1.0`](https://docs.google.com/document/d/1CYvwdsUtDYJ6ynOMXeRy13lzFm7gCuAOSwxMtgoZJAE), `1CYvwdsUtDYJ6ynOMXeRy13lzFm7gCuAOSwxMtgoZJAE` | native Google Doc | 7,150 reported | 2026-07-23 16:18:00.273 | same | text retrievable | native object; no stable raw SHA supplied |
| regenerated receipt | [`AO48-REC-034`](https://drive.google.com/file/d/12pnNmP7Uznn8p6km2c5r74qhxlShC3MQ/view?usp=drivesdk), `12pnNmP7Uznn8p6km2c5r74qhxlShC3MQ` | `text/markdown` | 7,453 | 2026-08-01 23:44:30.540 | My Drive | yes | `b8ee5feea01db10c34a9d4ac7e6ccac82c2232d7aa69536a3bd6e377282f6116` |
| terminal readiness capsule | [`LCR-CAP-001-v1.0`](https://docs.google.com/document/d/1hFmy3qq2ApkNDay9P-2MeLHHSwX0vFIzmoR1XVwEMck), `1hFmy3qq2ApkNDay9P-2MeLHHSwX0vFIzmoR1XVwEMck` | native Google Doc | 20,093 reported | 2026-07-24 19:54:42.651 | `1EJ5MiazceP6zzFUDyxmLmNdS5fgSb_xR` | text retrievable | native object |
| distinct-family exact replay / closure receipt | [`GP-AUD-EC019-PATHB-20260730-v1.0`](https://docs.google.com/document/d/1jfhA9_Wxckmla5F4732upP6pnLKH4eNNH-IjAj4EbP4), `1jfhA9_Wxckmla5F4732upP6pnLKH4eNNH-IjAj4EbP4` | native Google Doc | 7,565 reported | 2026-07-31 00:03:07.263 | `1mwgprwS1Q3sWCT8hdoA2Ib3TmMu_A7HH` | text retrievable | native object |
| terminal operator decision | [`AD-009`](https://docs.google.com/document/d/1JFEScO_517U325GU6Bz65bn8d_XHGLGhnjXCIQXxMik), `1JFEScO_517U325GU6Bz65bn8d_XHGLGhnjXCIQXxMik` | native Google Doc | 6,780 reported | 2026-07-24 20:29:31.396 | `1Y3hlYvcTLaTc7FVQ8h7rYkipy9I1CgOJ` | text retrievable | native object |

Exact search for `AO48-AUD-030` returned precisely two results: the audit carrier and `AO48-REC-034`. The broader `EC-019 T1` query reached the 100-result cap, so the EC-019 row above is a controlling-lineage table, not a claim that only these EC-019 references exist.

## G.9 / G.9.1 / Q = 1 / sandbox-line provenance

| Object | Drive ID / URL | Bytes | Created UTC | Parent | Retrieval / status | Raw SHA-256 |
|---|---|---:|---|---|---|---|
| `AO48-AUD-033`, G.9.1 independent review and Q = 1 proof | [`135ZKlWDnAuMXZlWK7gN3Scra67ShrcL9`](https://drive.google.com/file/d/135ZKlWDnAuMXZlWK7gN3Scra67ShrcL9/view?usp=drivesdk) | 10,154 | 2026-08-01 23:04:47.393 | My Drive | fully retrievable | `895adbf10e3ebabe457a8aa293aed8be19562dd657d151622ce753de2aee83da` |
| prior Part-G independent audit | [`P9_PARTG_AUDIT_2026-08-01.md`](https://drive.google.com/file/d/1eXFPTCOibZoVSWzhcMkLsUdGc9E0kqIj/view?usp=drivesdk) | 4,577 | 2026-08-01 22:08:20.831 | `08-01-2026`, `1LaUTgU-CAuLtteSUy3X0h4VzlinBLux4` | fully retrievable | `c17c269da7f99d4bbca1b23bd7ff0c69391caa55ddd303ad63bed32c33cfddec` |
| endpoint/capture companion audit | [`LEMMA_A4_AUDIT_2026-08-01.md`](https://drive.google.com/file/d/15i5An5oEMdZQqHgKKdWuTYUSXK6pQB9P/view?usp=drivesdk) | 3,913 | 2026-08-01 21:55:18.322 | same | fully retrievable | `84ba0863f78a6e3b03d5c152e1704eecdcc677788ebdff30197ac895ef5ae833` |
| V3.4 RP-C/RP-S closure proof relay | [`1sl4moxxKEOdeWV-TqV7WtE7komxsujOb`](https://drive.google.com/file/d/1sl4moxxKEOdeWV-TqV7WtE7komxsujOb/view?usp=drivesdk) | 35,310 | 2026-08-01 23:56:34.466 | My Drive | fully retrievable; relay says byte identity not certified | `40faad08824e4e77520143078ed606c8681532cc77ae9c9d42551baa5277b400` |
| V3.4 disposition + checks relay | [`1QJ_H5A-xg2TINB7CqHBopjvY4-AVTJdJ`](https://drive.google.com/file/d/1QJ_H5A-xg2TINB7CqHBopjvY4-AVTJdJ/view?usp=drivesdk) | 8,157 | 2026-08-01 23:57:18.745 | My Drive | fully retrievable; relay says byte identity not certified | `d262021d7c0819ba09a6680c29379e3d731e982705a939bd60b8361e3382fbb1` |
| ingest index | [`AO48-REC-035`](https://drive.google.com/file/d/1XqQOv2G8Q9lFczNqlwXTdkCL0OToVJTo/view?usp=drivesdk) | 5,137 | 2026-08-01 23:58:11.363 | My Drive | fully retrievable | `f8359b1cf389c899f0b8393281d9f2b513770bfe0c60952e6455bfdd11ddef6a` |

No standalone Drive result was found for the full source `SIDE24_GAP_FILL_SUPPLEMENT.md` or `verifier/v8/verify.py`. Searches for `full G.9 supplement` and `verifier/v8/verify.py` returned only references such as `AO48-AUD-033` and `AO48-REC-034`, not the carriers themselves.

Local recovery from the latest operator attachment resolves the information layer:

| Recovered local carrier | Bytes | SHA-256 | Manifest check | Execution |
|---|---:|---|---|---|
| `SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md` | 105,980 | `05967f14a8b8d6a9e2637210807872ba4f30810987cb7f3017e10b186d07f6da` | matches `SIDE24_v5_2026-08-02/09_PACKAGE_MANIFEST_V5.csv` | source read successfully; Part G.9 begins at line 1688 |
| `SIDE24_gap_fill/verifier/v8/verify.py` | 5,365 | `c95a86504e8b62ade4bd7d20956415db2cfe95cc5897815dae3411c12af709c5` | matches the same manifest | 30/30 checks pass under the recovered V5 audit environment; exit 0 |

Local roots:

- `/workspace/scratch/4c1cbf90475f/work/v5_audit_unpacked/SIDE24_gap_fill/`
- `/workspace/scratch/4c1cbf90475f/work/drive_recovery/`

Status precision: verifier v8 validates the V5 state in which RP-C/RP-S were explicitly still open. The V3.4 closure is the later successor and carries its own audit-pending status; v8 does not independently certify V3.4.

## Reconciliation conclusion

The carrier problem is smaller than reported. R2 is present and exact; the C030/C031 set is filed and locally integrity-checked; the full G.9 source and v8 verifier are recovered from the current operator attachment with manifest-matching hashes and a clean v8 rerun. What remains unresolved is not mathematical loss but Drive publication/identity for the two sandbox-source files and independent audit of the later V3.4 closure successor.
