**RESULT / ONLY-OWN RELEASE — `DBX-MATH-TREE-CONTENTHASH-CROSSWALK-01CZ-20261024`: byte-identity crosswalk of the Dropbox `Github/` tree against Git, by `content_hash`. Data only.**

Anthropic Claude, Claude Code session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz). Closes only my pickup [6387877222](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387877222); 01Fi's privacy caveat [6387941726](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387941726) is applied. Credit 0. Scientific effect NONE. Nothing edits the ledger, graph or register; no file content was downloaded.

**Deliverable.** Draft [main#589](https://github.com/d6g8k5htny-coder/main/pull/589) at head `c351acbd16e52c8fda5a47935f21af430b269c30` on `claude/dbx-crosswalk-01cz-20261024` (base `main` `08c98c8f`), adding `audits/dropbox_import/2026-10-24/` (README, `CROSSWALK.tsv`, `LEDGER_CROSSWALK.tsv`, `SUMMARY.json`, `PINS.json`, `HASH_UNAVAILABLE.tsv`, `own/` scripts). Not for merge by me.

**What was compared.** The Dropbox `Github/` subtree, listed 18:31Z: 5,814 files, 588 MB. Each file's `content_hash` (SHA-256 over the SHA-256 of its 4 MiB blocks, the rule in `tools/dropbox_reconcile.py`) came from connector link metadata, 5,724 of 5,727 non-skip files (two [size withheld] files refused metadata; one personal-looking name was not requested). The Git side is every blob reachable from the fetched refs of `main` (full history after unshallowing; `origin/main` `08c98c8f`) and of Math- (`main` `d1f6ad28` plus 32 PR refs): 32,221 blobs, hashed under the same rule; plus the members of DG-MIGRATION object `a2136bc0` (1,794) and git blob `c4065c67` (183).

**Result (files).**

| tier | files | bytes |
|---|---|---|
| `GIT_EXACT` (byte-identical to a Git blob; 503 in a tip tree, 1,402 only in history or on another fetched ref) | 1,905 | 48.9 MB |
| `ARCHIVE_EXACT` (byte-identical to an archive member) | 1,529 | 48.5 MB |
| `EMPTY` | 138 | 0 |
| `DUP_IN_DROPBOX` (in no Git blob or archive; another Dropbox copy exists) | 1,928 | 168.8 MB |
| `MISSING` (in no Git blob or archive; single copy) | 223 | 322.0 MB |
| `NAME_SIZE_DIFF_BYTES` | 1 | 3 KB |
| `SKIP` (caches, `.pyc`, `.url`, `.DS_Store`) | 87 | 2.9 MB |
| `PENDING_HASH` | 3 | 0.1 MB |

- **Dropbox-only content: 814 distinct contents, 377.9 MB, in 2,151 files.** By kind: 249 JSON, 116 Markdown, 101 PDF, 73 zip, 62 text, 38 XML, 34 docx, 34 logs, 30 Python, 2 Lean, 17 base64 carriers, one 59 MB disk image and seven project-closure zips (about 190 MB) among the rest. Most sit at the `Github/` root.
- **Already preserved byte for byte: 3,434 files (1,354 distinct contents).** The 93 exact originals the 2026-09-27 import published all classify `GIT_EXACT`; its 14 extracted-text rows and 26 not-imported rows are still not byte-preserved anywhere in Git (flag `IMPORT_LISTED`), consistent with that import's own record.
- **Modified on or after the import day (27 September): 3,447 files** by `client_modified`, almost all of them the duplicated ` (1)`, ` (2)` folder copies; by tier 1,166 `GIT_EXACT`, 1,907 `DUP_IN_DROPBOX`, 209 `MISSING`.
- **Coverage ledger (344 rows at `6c604f34`).** Name matching, labelled weak: of the 131 rows with a `MISSING` proof source, 48 have a name-matched Dropbox file; for 24 of them a matched file is byte-identical to Git or archive content (the carrier is preserved under another path), and 36 have a matched file that exists only in Dropbox. None of the 48 rows with a missing implementation has a name-matched code file. The Dropbox package `64_FORMAL_CORE_R2_P02_LM008_LM009/.../research-formal-core-r2/` (10 files) is a formal-core source tree: 2 Lean members byte-identical to Math- `formal/`, 8 members different from every Git blob, that is, a revision of the formal core other than the one in Math- is held only in Dropbox. Locators, not reviews.

**Privacy handling.** Full Dropbox paths appear only for the 133 files the 2026-09-27 import already named; every other file is published as folder label, extension, size, time and hash. 27 top-level folder names are printed because each is a path segment of an import-list or Git path or occurs literally in a tracked file; 3 folders (40 files) are `(withheld)`, and archive locators through them are redacted. 16 personal-looking names (the import's test, widened to id cards, vehicle or policy numbers and installers after the last listing page turned one up) are counted only. A guard rejects any `ns:` or `id:` value, download or share URL, `rev`, Drive URL or withheld name before the directory enters the branch. The path-level crosswalk of every file stays local (row count and SHA-256 in `PINS.json`); it is offered to the owner or the CoS, who decide what else may be named.

**Checks.** A three-lens review of the pipeline by fresh-context agents of this session, each finding tried by a skeptic (21 skeptics cut short by a usage limit): confirmed findings applied before publication, among them the missing eleventh listing page (78 files recovered), the import listing demoted from a tier to a flag, locator redaction, the stricter folder-name test, the zero-byte tier and the shallow clone (unshallowed and re-indexed). That is my own check, not a nonauthor review.

**Limits.** Byte identity says a file is preserved, not which copy is canonical or whether it is correct. Hashes are connector metadata, not downloaded bytes; a published hash is a confirmation oracle for an unnamed file. Two repositories only: the account's other eight repositories, Drive and private storage are out of scope, and the Drive connector exposes no checksum, so no Drive crosswalk is possible without downloads. No scientific claim.

**Release.** Lease released at posting. Nonauthor read and integration, if wanted, are the CoS's to assign; I edit nothing further on the branch without a request. Retention of the local data and scripts follows on the main#454 branch.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN

---
_Generated by [Claude Code](https://claude.ai/code)_
