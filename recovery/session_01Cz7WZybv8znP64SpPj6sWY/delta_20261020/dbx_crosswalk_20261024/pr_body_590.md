## Change and purpose

Data only: a byte-identity crosswalk of the owner's Dropbox `Github/` subtree (5,736 files, listed 24 October 2026 at 18:31Z) against every Git blob of `main` and `Math-` and the members of the two reassembled archives (DG-MIGRATION object `a2136bc0`, git blob `c4065c67`), using the Dropbox `content_hash` rule that `tools/dropbox_reconcile.py` implements. Adds `audits/dropbox_import/2026-10-24/` (README, `CROSSWALK.tsv` with per-file rows only for already-named paths or already-public bytes, `CROSSWALK_AGGREGATE.tsv` for the rest with no cell below three files, `LEDGER_CROSSWALK.tsv`, `SUMMARY.json` with byte totals rounded to 100 KB, `PINS.json`, `HASH_UNAVAILABLE.tsv`, `own/` scripts). This PR is the rebuild of draft #589 on a fresh branch from `main` `a3870738`, per Dylan's F2 decision relayed in [589/6389084026](https://github.com/d6g8k5htny-coder/main/pull/589#issuecomment-6389084026): no commit in this history ever carried per-file hash, size or time rows for unnamed files. Nothing else changes; the 2026-09-27 import directory is untouched. A reader can see which Dropbox files are already preserved byte for byte in Git or an archive, which exist only in Dropbox (772 distinct contents, 338.7 MB), which were modified on or after the import day, and which coverage-ledger rows with a `MISSING` proof source have a name-matched Dropbox carrier. Work discussion: [main#539 / 6387877222](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387877222) (pickup), under dot's storage-tooling findings [539/6387294856](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387294856) and 01Fi's privacy caveat [539/6387941726](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387941726). Scope: engineering/data. Scientific effect NONE.

## Current disposition

Head and integration owner: none; draft by Anthropic Claude session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz), not for merge by its author. A separate nonauthor read and a separate integrator decide, if the CoS wants it landed.
Claim / handoff: `DBX-MATH-TREE-CONTENTHASH-CROSSWALK-01CZ-20261024`, released with the RESULT [539/6388834520](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6388834520); this PR is its amended carrier, superseding #589 (closed, branch retained).
Successor of: none; retained stops: none; unavailable artifacts: the path-level crosswalk of every file is held locally by the author (row count and SHA-256 in `PINS.json`) and offered to the owner or the CoS, who decide what else may be named.
Completed review scopes and source identities, all on #589 and all applied here: nonauthor exact-head review 5639738563 (OpenAI lane, GTP) at c351acbd, AMEND_SOURCE_SCOPED, A1-A3; nonauthor delta reads 5639778197 and 5639799500 (Grok Bot agent 8) at 6c01a832 and 3e134717, AMEND_SCOPED, F1 and F3 applied, F1a-c applied in the rounding commit; F2 answered by this fresh history. The tree at this head is byte-identical to #589 head 37c29627. A fresh outside delta read at this head is the CoS's to assign.
Unresolved finding IDs and next action: none open for the author; the per-file hashes of unpublished content stay local pending an owner or CoS publication decision (A1 option 2).

## Verification

- Hashes: Dropbox `content_hash` read from connector link metadata for 5,724 of 5,727 non-skip files (two [size withheld] files refused link metadata, listed in `HASH_UNAVAILABLE.tsv`; one personal-looking name was not requested); no file content downloaded.
- Consistency checks run locally: the 93 `PUBLISHED_EXACT_ORIGINAL` rows of the 2026-09-27 `MANIFEST.csv` all classify `GIT_EXACT`; its 14 `PUBLISHED_EXTRACTED_TEXT` rows and the 26 `NOT_IMPORTED.csv` rows are not byte-preserved in Git (flag `IMPORT_LISTED`); the two previously verified anchors (`[dropbox file name withheld]`, `[dropbox file name withheld]`) classify `GIT_EXACT` and `ARCHIVE_EXACT`; the 138 empty files are tier `EMPTY`.
- Publication guard in `own/build_pr.py`, run before the directory is copied into the branch: no `ns:` namespace id, no `id:` value, no download or share URL, no `rev`, no Drive URL, no withheld folder name outside `SUMMARY.json`; per-file rows only for the 133 files the 2026-09-27 import already named or for bytes already public (`GIT_EXACT`, `ARCHIVE_EXACT`, `EMPTY`); aggregate cells never describe fewer than three files; every published byte total is rounded to 100 KB; 16 personal-looking names counted and not listed; collection fails closed with harvested observations validated and counted (5,724 accepted, 0 conflicts).
- Hosted checks: none required for a data-only directory; whatever runs on this branch is linked from the PR checks tab.

## Evidence and limits

- Canonical record: the RESULT on main#539 (linked from the PR once posted) and `audits/dropbox_import/2026-10-24/README.md`.
- Byte identity says a file is preserved somewhere in Git or an archive; it says nothing about canonical copies or correctness. Name matching to the coverage ledger is weak evidence and is labelled as such.
- The Git side is the refs of the session clones at the recorded commits (`PINS.json`); the Drive connector exposes no checksum, so no Drive crosswalk is made here.
- Authorship: Anthropic Claude (01Cz), source-exposed author; no review credit claimed. credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
