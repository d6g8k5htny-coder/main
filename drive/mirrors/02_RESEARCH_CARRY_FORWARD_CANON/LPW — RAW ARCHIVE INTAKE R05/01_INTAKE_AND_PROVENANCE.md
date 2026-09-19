# LPW-R05-INTAKE — hash and source reconciliation

## Incoming bytes

- `SepOKComputer_Project_Gap_Closure.zip`: 25,269,465 B; SHA-256 `07b44ac2537e7288eaa21decfec04bfa924c6181ba6b4179d34cc078ddc3f139`.
- `LEAD_REVIEW_OF_LPW_CONSTANT.pdf`: 49,009 B; SHA-256 `fe4875d15b31460379f5f4f70dceb27a73f41bf8e6ba7270ae80d2c4a0e1ea98`.

All extraction occurred in a separate work tree. No member with traversal, unsafe absolute path, duplicate/case collision, symlink, encryption, or control-character path was accepted. All twelve ZIP occurrences passed CRC; one original author ZIP is duplicated byte-for-byte, leaving eleven unique archives. The outer contains historical copies and caches as well as the current deliverables; their presence is not new mathematical evidence.

## Current archives

| Archive | Bytes | SHA-256 | Matched manifest payload entries |
|---|---:|---|---:|
| KIMI_LPW_REVIEW_BUNDLE_2026-09-13.zip | 97963 | b7e5cbf33fa883e9789be99b6b965d42ed4ed653a1a9dde257ad1587b5c22f8e | 37 |
| KIMI_QC_NUMERICAL_PACKAGE.zip | 97671 | 2f99252ed8acc97ccee468bbcb2419eac214156720bbc78ecdebb3f753e69ea7 | 19 |
| KIMI_LPW_CONSTANT_PACKAGE.zip | 33844 | ed9da9b4ffa099270c9a1c3467b78be325f99c3baa37b395a06d33114b6b1a01 | 9 |
| KIMI_W8_V3_PACKAGE.zip | 58851 | 6d7b0ed5de12082ad540c6f28a61efde90cc414a01fd780cc5e313377c207410 | 17 |

All 82 payload entries match. The review ZIP contains 38 files including its manifest, not 37 total files; this is a counting convention, not a missing file. Directory entries in other ZIPs are not counted as payload files.

The outer historical K3 manifest is not the final seal for these new packages: it has 266 records, 247 current matches and 19 changed objects. Older W13 manifests are snapshots with their own path-root conventions. The detailed generic manifest scan reports relative-path misses where no root mapping was supplied; those must not be mistaken for independent proof that all referenced historical objects are absent. The four current envelopes are the explicitly checked delivery seals.

## Exact review body mismatch — R05-CUSTODY-01

`KIMI_LPW_REVIEW_VERDICT.md` is 9,219 bytes; its full SHA is exactly
`04bcbdf2013d83b149a8933a332a078c303af347335ee40f154f37ba195ad448`.

The stated rule hashes the exact prefix before the unique `SHA-256 of this report body` marker. The marker begins at byte 9090. That prefix hashes to
`17863be2e9fedc4ddc6517562c501a609fa4fa3579f8dad1348fb67f68a9c5bd`.

The declared `d399e28378d177f88d08032c81f1573f35c9af2b600b9b775ebfe40f555cbbad` instead matches the prefix of length 9084. The six excluded bytes are exactly `\n---\n\n`.

Disposition: whole-file custody VERIFIED; body-extraction wording AMEND REQUIRED. Preserve the exact original file. Either publish an additive extraction rule excluding that exact separator, or publish the actual 9090-byte prefix hash. Never silently trim/normalize a declared byte prefix.

All other explicitly sealed current body hashes checked here match: Addenda 1–3, lead analytic verdict, and lead LPW_CONSTANT review. See BODY_HASH_AUDIT.json for their offsets, whole hashes, and body hashes.

## Replay scope

Seven programs were copied to fresh execution trees without editing source and run normally and with `-O`. All fourteen outputs match their received reference bytes, with zero exit and empty stderr. The constant falsifier itself runs its eight contract-mutation child tests. Reproduction validates behavior of these exact programs; it does not validate every inequality described in their reports. In particular, the repeated coefficient and false fourth-moment identity survive their old tests.

RA's standalone original script remains absent by the reviewer README's own account. Its report and embedded transcript are delivered; no replacement has been manufactured and labeled original. W8 source/transcript inspection was performed, but its long sweep was not rerun. No external background job is claimed observed.
