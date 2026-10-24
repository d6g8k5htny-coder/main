## Change and purpose

Data only: a byte-identity crosswalk of the owner's Dropbox `Github/` subtree (5,814 files, listed 24 October 2026 at 18:31Z) against every Git blob of `main` and `Math-` and the members of the two reassembled archives (DG-MIGRATION object `a2136bc0`, git blob `c4065c67`), using the Dropbox `content_hash` rule that `tools/dropbox_reconcile.py` implements. Adds `audits/dropbox_import/2026-10-24/` (README, `CROSSWALK.tsv` with per-file rows only for already-named paths or already-public bytes, `CROSSWALK_AGGREGATE.tsv` for the rest, `LEDGER_CROSSWALK.tsv`, `SUMMARY.json`, `PINS.json`, `HASH_UNAVAILABLE.tsv`, `own/` scripts). Nothing else changes; the 2026-09-27 import directory is untouched. A reader can see which Dropbox files are already preserved byte for byte in Git or an archive, which exist only in Dropbox (814 distinct contents, 377.9 MB), which were modified on or after the import day (3,447), and which coverage-ledger rows with a `MISSING` proof source have a name-matched Dropbox carrier. Work discussion: [main#539 / 6387877222](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387877222) (pickup), under dot's storage-tooling findings [539/6387294856](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387294856) and 01Fi's privacy caveat [539/6387941726](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387941726). Scope: engineering/data. Scientific effect NONE.

This is the single-commit rebuild that the CoS disposed for main#590 H1 ([539/6389430715](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6389430715), option (a), not squash): one commit on a fresh branch from `main` `a3870738`, so that no pre-rounding totals and no per-file rows for unnamed files exist anywhere in the mergeable history. The tree is the reviewed #590 tip tree `d09bf1de` plus the nonblocking N1 of review 5639987376: `HASH_UNAVAILABLE.tsv` withholds the sizes of the two unnamed files (folder and extension only), with the matching README sentence and the two `own/` scripts that write them. Four files differ from `d09bf1de`; the other fourteen are byte-identical.

## Current disposition

Head and integration owner: none; draft by Anthropic Claude session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz), not for merge by its author. A separate nonauthor read and a separate integrator decide, if the CoS wants it landed, after the current MAIN-LOCK order.
Claim / handoff: `DBX-MATH-TREE-CONTENTHASH-CROSSWALK-01CZ-20261024`, released with the RESULT on main#539 ([539/6388834520](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6388834520)).
Successor of: draft main#589 (closed, superseded per Dylan's F2 decision) and draft main#590 (closed, superseded per the CoS H1 disposition); both branches stay in place, no remote deleted. Retained stops: none. Unavailable artifacts: the path-level crosswalk of every file is held locally by the author (row count and SHA-256 in `PINS.json`) and offered to the owner or the CoS, who decide what else may be named.
Completed review scopes and source identities: nonauthor exact-head review 5639738563 (OpenAI lane, GTP) at `c351acbd`: AMEND_SOURCE_SCOPED, A1–A3 applied; nonauthor delta review 5639778197 (agent 8) at `6c01a832`: F1, F3 applied, F1a–c rounding applied, F2 answered by rebuild; nonauthor delta review 5639987376 (agent 8) at `6a9c049c`: AMEND_SCOPED, tip verified, H1 answered by this single commit, N1 applied, N2 (body counts) corrected here. A fresh nonauthor delta read, not by agent 8, is requested per the CoS disposition.
Unresolved finding IDs and next action: none open; the per-file hashes of unpublished content stay local pending an owner or CoS publication decision (A1 option 2).

## Verification

- Hashes: Dropbox `content_hash` read from connector link metadata for 5,724 of the 5,727 non-skip files (two [size withheld] files refused link metadata and one was not requested; `HASH_UNAVAILABLE.tsv` lists the two refusals by folder and extension); no file content downloaded; collection fails closed (`hash_receipt` in `SUMMARY.json`: 0 rejected, 0 conflicts).
- Consistency checks run locally: the 93 `PUBLISHED_EXACT_ORIGINAL` rows of the 2026-09-27 `MANIFEST.csv` all classify `GIT_EXACT`; its 14 `PUBLISHED_EXTRACTED_TEXT` rows all classify `IMPORT_SEEN`; the two previously verified anchors (`[dropbox file name withheld]`, `[dropbox file name withheld]`) classify `GIT_EXACT` and `ARCHIVE_EXACT`; the 138 empty files all match the empty blob.
- Publication guard in `own/build_pr.py` (local): the output contains no `ns:` namespace id, no `id:` value, no download URL, no `rev`, no Drive URL, no withheld folder name; full Dropbox paths appear only for the 133 files the 2026-09-27 import already named; 16 personal-looking names are counted and not listed; every published byte total is rounded to 100 KB and no aggregate cell describes fewer than three files.
- History check: `git log a3870738..HEAD` is one commit; `git diff d09bf1de HEAD^{tree}` touches `HASH_UNAVAILABLE.tsv`, `README.md`, `own/build_pr.py`, `own/readme_gen.py` only.
- Hosted checks: none required for a data-only directory; whatever runs on this branch is linked from the PR checks tab.

## Evidence and limits

- Canonical record: the RESULT on main#539 and `audits/dropbox_import/2026-10-24/README.md`.
- Byte identity says a file is preserved somewhere in Git or an archive; it says nothing about canonical copies or correctness. Name matching to the coverage ledger is weak evidence and is labelled as such.
- The Git side is the refs of the session clones at the recorded commits (`PINS.json`); the Drive connector exposes no checksum, so no Drive crosswalk is made here.
- Authorship: Anthropic Claude (01Cz), source-exposed author; no review credit claimed. credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
