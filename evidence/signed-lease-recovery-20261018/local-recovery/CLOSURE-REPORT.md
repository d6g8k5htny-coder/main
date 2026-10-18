# Complete retained packet recovery and original-packet discrepancy

Result: **EXACT_REQUESTED_TARGETS_RECOVERED; P01/P02/P04/P05/P06 CLOSED; P03 HOLD**. This append-only completion record extends the earlier seven-target report SHA-256 `3c247289d89dab87d0cc5e13f0089dc8c74dfae48cb1d562bec0dfee225f7d31`. Its then-unperformed whole-packet manifest validation was subsequently authorized and completed here. Completion time: 2026-10-18T00:36:52.067985+00:00.

| Packet ID | Manifest payloads | Files incl. manifest | Dirs incl. root | Bytes incl. manifest | Literal manifest SHA-256 | Result |
|---|---:|---:|---:|---:|---|---|
| P01 | 150 | 151 | 51 | 7691429 | `13d285a52c7d5b373601d6e1a1f6ab3798ae2b260fcfa545a6be4ed5815e74f4` | CLOSED; full copied |
| P02 | 35 | 36 | 16 | 1243100 | `9272448f83347d950541748dabb733ce541dcc1a7635a793e74fe7dacd30f63f` | CLOSED; full copied |
| P03 | 26 | 27 | 19 | 1359621 | `28dcec337cc962c9880f3edd577ab8e50f269019ea0727029dbb03779c0e720c` | HOLD: one payload hash mismatch |
| P04 | 4 | 5 | 1 | 144407 | `d3fe23a71695c19e7ebd03ef2b8b641001306a9a58d83c280b8df18e1737387f` | CLOSED; full copied |
| P05 | 2 | 3 | 1 | 6963 | `b314615f371103c14faf54b3cdf21bbc5ae4fbf363d0bf65c5bb50db9c87eec4` | CLOSED; full copied |
| P06 | 26 | 27 | 19 | 1361893 | `28dcec337cc962c9880f3edd577ab8e50f269019ea0727029dbb03779c0e720c` | CLOSED retained subpacket within P01 |

P01 is the required complete predecessor; P02 is the accepted F03 successor. P04 is the complete accepted S01 distinct review packet, with its initial attribution drafts preserved. P05 is the complete append-only attribution correction packet. P06 is an existing complete retained accepted S01 author packet nested within P01. These roles identify documentary context; they do not add a new review verdict.

All 150 predecessor payloads and all 35 successor payloads match the complete literal manifest SHA-256 entries. Their exact file sets are the manifest entries plus the manifest itself, with no missing, unexpected or duplicate member. All source files are regular, nonsymlink, single-link mode 0444; all source directories are real mode 0700. Safe member resolution rejected absolute, dot, parent, backslash, duplicate and self-manifest paths. Every original was reread after copying with the same raw hash. Complete P01, P02, P04 and P05 packets were copied using O_EXCL/O_NOFOLLOW into private recovery containers, with mode 0444 files and mode 0700 directories; every whole copy readback literally equaled its original.

The full LP controls, intents, seeds, portable and target DDL, dependency-file inventory, source inventory, retained source histories, forward and inverse evidence are preserved within the complete closed packets. No text patch was applied and no authored helper, checker, controller or project source was imported or executed.

**Preserved negative:** P03 is the original adjacent S01 author packet. Its 26-entry manifest is itself the expected 2,660-byte SHA-256 `28dcec337cc962c9880f3edd577ab8e50f269019ea0727029dbb03779c0e720c`, and its file set is closed. One declared member, identified privately as P03-MISMATCH-01, is currently empty: 0 bytes / SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The declared accepted forward-patch identity is 2,272 bytes / SHA-256 `ef1694dbb4878c9b36a08eb3b98a6393b01057c5de373bc5b8ab008e9a0fabe7`. Current empty bytes and the literal manifest were retained as negative evidence. No cause is inferred; P03 was neither repaired nor copied as a qualified whole packet.

P06 supplies the existing complete retained S01 packet: all 26 own-manifest payloads, including the full 2,272-byte accepted forward patch, match their exact declared hashes. Its file set is closed and all 27 original files equal the retained copies in P01. The accepted S01 tests and contract also remain independently recovered by their exact requested 5eaa and c7ce whole-source identities. This is recovery of existing raw bytes, not reconstruction from the published preimage or substitution into the discrepant P03 source.

Only private source tables contain original filesystem locators. The public-safe summaries use opaque IDs and complete raw-byte hashes. No additional filename/content search or session-log access occurred during this adjacent-packet extension. The earlier bounded search and seven unique recovery copies remain preserved unchanged.

Dependent work remains HOLD: implementation LOCKED; harness NOT_PREPARED; partition NOT_DESIGNED; schedule UNADMITTED; CONTRACT_RED NOT_QUALIFIED; organizational independence 0; custody authentication UNKNOWN; scientific effect NONE. This work supplies concrete exact source packets for a separately assigned review, without assigning that review or a future writer.

Personal release: only this actor's finite READ_ONLY_LOCAL_DATA recovery allocation is complete and released. No foreign allocation, ownership claim, source/design gate or runtime gate is released. All source originals remain unchanged.
