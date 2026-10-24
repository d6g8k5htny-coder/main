# Recovery delta, 18–24 October 2026: session 01Cz, recovery rows §12, §5 and §18, the main#446 consumer note, the main#460 reviews and composition read, and the 21–24 October units

Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. This is a retention copy for the owner's recovery request, [main#307 / 6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988).
- It continues the 17–18 October delta, `recovery/session_01Cz7WZybv8znP64SpPj6sWY/delta_20261018/` (branch `claude/recovery-01cz-delta-20261018`, draft main#444), and does not repeat it.
- It is kept separate because main#444 is already near the 500-file limit of the `public-intake` check.

- **Not for merge.** Nothing here is a proof, an accepted result or a status change. Scientific effect NONE. Organizational-independence credit 0.
- **What it holds:** this session's own posted texts, comparison and verification results, check scripts and logs behind the units in the table below, plus the scripts of their multi-agent runs.
- **What it leaves out:**
  - Copies of other lanes' material: the four carriers, Math- files, comment texts and API dumps.
  - Own files that are superseded or redundant: an earlier draft, partial and subset results, text dumps, duplicate scripts, sent bodies with the footer and API readbacks.
  - `s12_d5_ledger_comparison/SOURCES.json` lists each excluded file by local path, size and SHA-256.
  - The two docx carriers, their text layers and the pre-push check's scratch contain Drive ids and are not included.
- **Placeholders:** local path prefixes are replaced by `<SCRATCH>`, `<SESSION_TMP>` and `<SESSION_DIR>`. `MANIFEST.json` records the pre-substitution SHA-256 of each affected file.
- **Redaction:** `s12_d5_ledger_comparison/prepost_check_results.json` is marked `hand_edited` with `sha256_before_hand_edit`. In it, product names were redacted before the path substitution.
- **Integrity:** every file except `MANIFEST.json` and `SHA256SUMS` is listed in `MANIFEST.json` with its size and SHA-256.

## Unit

| directory | claim and subject | public record | outcome |
|---|---|---|---|
| `s12_d5_ledger_comparison/` | `S12B-D5-LEDGER-COMPARISON-01CZ-20261019`, after the late-released `S12-D5-LEDGER-COMPARISON-01CZ-20261018`: ledger comparison of the July D5 historical carriers (C100, LS-DER-024, LS-DER-033, FlatSaddle) with the P2 punctured-pin proof (recovery row §12) | main#419: pickup 6251517975, late release 6279524549, fresh pickup 6279655509, result and release 6280394188; main#275 pointers 6251521112, 6279528124, 6279660527, 6280395619 | CONSISTENT AT THE FINAL POWERS, with intermediate-ledger defects recorded; posted and released |
| `s5_side24_v33_v34_binding/` | `S5-SIDE24-V33-V34-BINDING-01CZ-20261020`: custody binding record for the SIDE24 V3.3 checkpoint and V3.4 replay archives (recovery row §5) | main#419: pickup 6280913839, checkpoint 6281760247, result and release 6281985456; main#275 pointers 6280916731, 6281988200 | both archives bound byte-exact; no standalone V3.3-layer eigenfloor table; posted and released |
| `s12_d5_ledger_comparison/cursor_check/` | the summoner's check of the xAI Cursor agent's PASS_SCOPED on §2(b) of the §12 record, with one numeric correction to that record | main#454: request 6282011803, verdict 6282139686 (xAI), pointer 6282637458; main#419: check and correction 6282634857; main#275 pointer 6282637162 | verdict confirmed on substance; "0.42 at t = 0.4" corrected to 0.479 ± 0.001 |
| `s18_side24_custody_xlink/` | `S18-SIDE24-CUSTODY-XLINK-01CZ-20261020`: custody cross-link record for the SIDE24 custody items behind gap rows G142–G147 (recovery row §18) | main#419: pickup 6282065819, checkpoint 6282579128, result and release 6283335403; main#275 pointers 6282068463, 6283336911 | all objects byte-exact on the hardening branch; receipt replays from both identities; four of five status texts outdated or wrong as worded; posted and released |
| `h3n_m446_consumer_note/` | `M446-CONSUMER-NOTE-01CZ-20261020`: the Route B downstream consumer note (M1/M2) for the landed H3 closure package of main#446 | main#419: pickup 6283686275, checkpoint 6284218586, note and author-stage release 6284294166; pointers main#446 6284295602, main#275 6283688331 and 6284295984 | note posted; one nonauthor read requested |
| `m460_nonauthor_review/` | `M460-NONAUTHOR-REVIEW-01CZ-20261020`: nonauthor engineering review of draft main#460 (test-only R03 pin) at exact head `65a7ab8d` | main#460: pickup 6285652622, verdict 6286384129 (edited once at 08:05Z), native COMMENT review 5582356017; main#419 pointers 6285653023, 6286402021 | PASS_SCOPED with limits; the integrator then held on R03's hosted-comparison predicate (460/6286492695) |
| `m460_review2/` | `M460-REVIEW2-01CZ-20261020`: fresh nonauthor engineering review of draft main#460 at exact head `a68ce39c`, after the writer added hosted R03 controls | main#460: pickup 6286773707, verdict 6287637143, native COMMENT review 5583098843; main#419 pointers 6286776203, 6287643104 | PASS_SCOPED with one limit (L1: the hosted control asserts the R03 witnesses but does not record their observed outcomes), for R03's owner and the integrator |
| `m460_composition/` | `M460-COMPOSITION-01CZ-20261020`: the bounded composition read of draft main#460 at exact head `b5b6eb20` (merge of `a68ce39c` with `main` `7e95ddcc`) that the integrator requested in 460/6288136767 | main#460: pickup 6288150010, pointer 6288265710, verdict 6288840431, native COMMENT review 5583815803; main#419 pointers 6288153935, 6288849308 | PASS_SCOPED: the test blob and the PR's patch are unchanged by the refresh, main#461's base delta touches nothing the test or its workflow reads, and the `a68ce39c` review (L1, N2) applies unchanged; posted and released |
| `c2_335_characterization/` | `C2-CONTRACT-CHARACTERIZATION-01CZ-20261021`: data support for Math- issue 335. It records the genuine baseline and per-mutant outcomes of the C2 torus-transfer omission controls on Math- `main` `27606578` | Math- issue 335: pickup 6328350180, record and release 6328636311; main#475 pointers 6328354409, 6328643308 | all five mutants exit 1 through the final `RESULTS.json` comparison in both modes, and each changes a field set that matches its seeded defect; an uncaught exception also exits 1, and the workflow's exit-code-only check would accept it; released |
| `rvdq023_locators/` | the RV-DQ-023 preflight locators for LS-DER-021, item 2 of 419/6250496071, in the locator-only role of 419/6320832078 | main#475: post 6327850237 | delivered; dot recorded it complete (475/6328045583); the native binding and the third-family preflight stay open |
| `m481_pickup_release/` | `M481-FINAL-INTEGRATION-01CZ-20261021`: the main#481 final-integration pickup, released without a merge after this session's permission control refused a pre-merge read | main#481: pickup 6327996637, release 6328022731, diagnostic 6328105706; main#475: 6328001560, 6328029033, 6328123320 | no mutation; main#481 later merged by another lane as `b7f72a71` |
| `crawl_20261021/` | the read-only crawl of six repositories as of 22:18Z on 21 October: 99 open PRs and issues, all in main, Math-, meta-framework and query- (governance- and Universal-Law-Workspace had none open) | — | as-of ledger; not a register or an assignment |
| `cov_exact_engines_20261022/` | `COV-EXACT-UNASSESSED-ENGINES-01CZ-20261022`: locators and a static trace of the engines that the `cov_exact.py` defect disposition (main#447 comment 6329640342) lists as not assessed, classified under the #447 author's rule | main#475: pickup 6329729396, record and release 6330437266, run-setup note to 01Fi 6330494323 | delivered and released; data only |
| `m322_ledger_20261024/` | `AUD-LEGACY-MUTANT-01-LEDGER-01CZ-20261022`: static classification ledger of the 82 Math- workflows whose text contains `returncode` (control predicate, accepted exit statuses, baseline stderr handling, skeptic re-derivation, repair status), for Math- issue 322 | Math-322: pickup 6331271545 (22 Oct), lapse and re-pickup 6387837249, record and release 6388271177 (24 Oct); main#475 pointer 6331271890; main#539 mirrors 6387837482 and 6388276823 | delivered and released; data only |
| `dbx_crosswalk_20261024/` | `DBX-MATH-TREE-CONTENTHASH-CROSSWALK-01CZ-20261024`: byte-identity crosswalk of the owner's Dropbox `Github/` tree (5,814 files) against every fetched Git blob of main and Math- and two reassembled archives, by Dropbox `content_hash`; published subset in draft main#594 (head `06c4a1a0`, one commit from `a3870738`), after drafts main#589 and main#590, both closed as superseded | main#539: pickup 6387877222, record and release 6388834520, statuses 6389025058, 6389079863, 6389101569, 6389504892, fact note 6389412490; main#589: pointer 6388837881, author responses 6388931004, 6389024430, 6389079709, close 6389101272; main#590: close 6389024617, CI note 6389114519, author response 6389501107 | delivered and released; data only; path-level data held locally; nonauthor read of main#594 PASS_WITH_FINDINGS (review 5640045547, agent 13), author response 6389618132; ready for a separate integrator |
| `drive_crosswalk_20261024/` | `DRIVE-SNAPSHOT-BYTE-CROSSWALK-01CZ-20261024`: the SHA-256 digests of the 2026-09-17 Drive accessibility publication (2,080 files and 11,649 archive members, `drive/` tree at `a01c72f1`) matched without download to every fetched Git blob of main and Math-, to the Dropbox `Github/` hashes of main#594 and to two reassembled archives, with a connector liveness listing and a name join to the DRV-H2/H3 maps and the coverage ledger | main#539: pickup 6389922273, record and release (see `posts.json`) | delivered and released; data only; id-level data held locally |
| `codex_stall_20261024/` | the Codex-stall crawl of 22–24 October: items whose next step is owed by an OpenAI lane, each re-read by a skeptic where the usage limit allowed, as a complement to the CoS crawls | main#539: report 6388850664 | delivered; NOT A RESULT; no pickup |
| `workflows/` | the scripts of the §12 comparison run, the four-lens check of that result post and its re-check, the §5 lens run, the §2(b) summoner check and its pre-post check, the §18 lens run and its two pre-post checks, the consumer-note verification run and its pre-post check, the two main#460 review runs, the main#460 composition-read run and its pre-post check, the RV-DQ-023 locator run, the crawl run, the C2 record's pre-post check, and the cov_exact record's three-lens run and `lambda_q` refutation panel | — | method record |

## Reproducing

- **Interpreter:** CPython 3.11, with mpmath 1.3.0, sympy 1.14.0 and numpy 2.4.6 where imported; the agents ran Python with `-I`.
- **Carriers.** The carriers are not included; the two docx files contain Drive ids.
  - Rebuild each from the hardening branch `a01c72f1` DG-MIGRATION objects (`drive/deltas/2026-09-19/DG-MIGRATION-20260919/`) with `s11_d1v23_receipt_replay/own/rebuild.py` from the 17–18 October delta. The objects are `24eb9d55` (C100), `ce1eafee` (LS-DER-024), `e373caf4` (LS-DER-033) and `3875016e` (FlatSaddle).
  - Take P2 (blob `d8acf0bc`), its two committed review records and the D5 reconciliation (blob `4062d76b`) from Math-.
  - `LS-DER-024.txt` and `LS-DER-033.txt` are the docx text layers: `word/document.xml` with each `</w:p>` replaced by a newline and every other tag removed, without unescaping. This reproduces the hashes in `SOURCES.json`.
  - The scripts take their inputs as command-line arguments. The agents ran them on copies in `<SCRATCH>/s12/src/` under the names listed in `SOURCES.json`.
- **Results.**
  - `journal_results.json` holds the four carrier comparisons and their eight adversarial checks, from the run in `workflows/s12-d5-ledger-comparison-*.js`.
  - `prepost_check_results.json` and `recheck_results.json` hold the two checks of the post.
  - The `work/*/` directories are those agents' own scripts and logs.
- **Posted texts.**
  - The `*_posted.txt` files give the comment ids, and only those GitHub comments are authoritative.
  - `ptr275.md`, `ptr275_late.md` and `ptr275_b.md` include the attribution footer; the other posted texts omit it.
  - `post.sh` posts a comment and reads it back. It is kept as a method record and must not be run by anyone else.

- **s5.**
  - The two archives are not included. Rebuild them from the hardening branch `a01c72f1` DG-MIGRATION objects `4040d654` (V3.3) and `ba2a296b` (V3.4) with `s11_d1v23_receipt_replay/own/rebuild.py` from the 17–18 October delta; V3.4 also equals hardening blob `b2901c1d`.
  - `own/ledger_check.py <V33.zip> <V34.zip> <out.json>` reads both archives as data and checks their internal ledgers. `manifest_V33_V34.json` is its output: every top-level member's path, size and SHA-256. `diff_V33_V34.tsv` lists each member as unchanged, changed or added between V3.3 and V3.4.
  - One member name has the shape of a Drive id and is written `<drive-id>` in both files; it is the nested V5 export (6,062,706 B, `119ffbfa…`). Renaming it in the archives would break their own ledgers.
  - `lens_results.json` holds the run's four lens reports and their adversarial verifications. One product label in it is redacted; it is marked `hand_edited` with `sha256_before_hand_edit`.
  - The agents' scratch is not included: some of it holds copies of other lanes' files and unredacted Drive-id tokens. `SOURCES.json` lists each scratch directory by file count, bytes and a listing hash.

- **s12 `cursor_check/`.**
  - `check_results.json` holds the text, numeric and adversarial checks of the xAI verdict; `prepost_results.json` holds the pre-post check of `check_corr.md`.
  - `work/num/` and `work/adv/` are those checkers' scripts and logs. Some take the record's `bfcheck.py` (the unit's `work/lsder024/bfcheck.py`) as a command-line argument and reuse its definitions. They need numpy, which the agents installed locally and which is not included.
  - The verdict and pickup texts are another lane's comments and are not copied; read them by comment id.
- **s18.**
  - `own/replay_<blob>_out.txt` and `_err.txt` are the outputs of running each arithmetic_ledgers.py identity (hardening blobs `68da0f8c`, `cf1434d6`) in isolation, with Python 3.11.17, SymPy 1.14.0 and mpmath 1.3.0. Each `out.txt` equals the receipt `d72a3f9d` byte for byte. The scripts themselves are repository blobs and are not copied.
  - `own/rechecks.txt` records the summoner's own re-checks: `main`'s tree, the 23 September merge and revert, the Math- branch and PR heads, the G146 commits, the two body hashes, and the run-log and V2-manifest lines.
  - `lens_results.json` holds the five lens reports and their adversarial verifications, including the re-runs after a container restart. `prepost_results.json` holds the two pre-post checks of the result. Model and product labels in them are redacted and marked `hand_edited`.
  - The agents' scratch is not included; `SOURCES.json` lists each scratch directory by file count, bytes and a listing hash.

- **h3n.**
  - `own/recon_receipt.py` re-serializes `main`'s `RECEIPT_h3.json` (blob `96e95d17`) and removes its band records. `own/recon_receipt_out.txt` shows the two earlier recorded versions (4,482 B `ff788042…`, 2,717 B `7545a8c6…`) reproduced byte for byte.
  - `own/a2136bc0_members_check.txt` lists the certificate programs, transcripts and H2 libraries found in DG-MIGRATION object `a2136bc0`, rebuilt from the hardening branch with `s11_d1v23_receipt_replay/own/rebuild.py` from the 17–18 October delta. The archive is not included.
  - `note.md` and the JSON results carry recorded redactions of product labels (`hand_edited`); the posted comments are authoritative.

- **m460.** `lens_results.json` holds the three lens reports (diff, mutation, hosted) and their adversarial verifications. The agents' archive copies, mutant copies and run logs are not included; `SOURCES.json` lists them by listing hash. `verdict.md` is the comment as edited at 08:05Z; the first posting rendered a raw U+3000 inside one code span, and GitHub stripped it.

- **m460_review2.** `lens_results.json` holds the three lens reports (code, replay, hosted) and their adversarial verifications. Two verifiers finished after the verdict was posted; `followup.txt` records why no follow-up note was needed. The agents' archive copies, variant copies, run logs and a container-image root filesystem are not included; `SOURCES.json` lists them by listing hash.

- **m460_composition.** `lens_results.json` holds the four lens reports (identity, base delta, replay, hosted) and their adversarial verifications; `prepost_results.json` holds the two pre-post checks (accuracy, rules) of `verdict.md`. The agents' archive copies, run logs and scratch are not included; `SOURCES.json` lists them by listing hash. Model and product labels in the posted texts, the results and the pre-post workflow script are redacted and marked `hand_edited`; the posted comments are authoritative.

- **c2_335_characterization.**
  - `own/run_all.py <src> <out>` runs the 14 children (baseline, controls and five mutants, in `-B -S` and `-B -O -S`) from the packet directory. `own/results.json` is its output, with each stream's size and SHA-256.
  - `own/field_diff.py <packet> <mutant>` lists the `RESULTS.json` fields a mutant changes. `own/fields/` holds its five outputs.
  - `own/bogus_unknown_mutant.stderr` is the crash counterexample.
  - The workflow's first `run` step, which was run unchanged, is not copied. Extract it from workflow blob `819f1046` (1,463 B, SHA-256 `59dd4e1a…`). Rebuild the tree with `git archive` of Math- `27606578`. `SOURCES.json` lists the tree, the child streams and that excerpt.
  - `record.md` is the posted text. `prepost_results.json` holds the two checks applied before posting.
  - Model labels in the pre-post workflow script are redacted and marked `hand_edited`.
- **rvdq023_locators.**
  - `post475.md` is the posted text.
  - `own/scripts_apply.py <export> <begin> <end> <out>` applies the frozen-body rule. `own/scripts_extract.py <dir>` tests candidate rules against the frozen targets. `own/vd_check.py <draft>` checks a draft's blob ids and SHA-256 values against local git.
  - Not included, and listed in `SOURCES.json` by listing hash: the lens reports, the agents' verification and work scratch, the mirrored exports, the extracted candidate body and the board scan.
- **m481_pickup_release.**
  - Posted texts only. One product label in `pickup.md` is redacted and marked `hand_edited`; the posted comment is authoritative.
  - The refused step was a read-only GitHub `GET`, refused by this session's own permission control. No write or merge call was sent.
- **crawl_20261021.**
  - `ledger.md` is the synthesis, and `items.json` holds the 99 per-item records, as of 22:18Z on 21 October. Both are snapshots: re-read GitHub before acting on any line.
  - Model and product labels in them, and in the crawl workflow script, are redacted and marked `hand_edited`.
- **cov_exact_engines_20261022.**
  - `record.md`, `pickup.md` and `note_01fi.md` are the texts as sent. The `*_posted.txt` files are GitHub readbacks. Each readback has one more blank line before the footer than the sent text, and is otherwise identical.
  - `own/scan.py <root> <out.json>` is the static scanner. It reads every `.py` file under `<root>` as text, and parses with `ast` (never importing or running) only those whose text names `cov_exact` or `f08c1c5f`.
  - `own/scan_reduced.json` is its output with the quoted source text dropped: the `text` fields and the quoted halves of `load_lines` and `lattice_attr` are removed, and `lattice_attr` is renamed `lattice_attr_lines`.
  - `own/lamq_check.py` and `own/lamq_check2.py` are this session's own re-implementations of one lattice sum (lattice at 40 digits, and at 100 digits summed at 40). Their outputs are beside them.
  - `prepost_verifier_report.md` is the fresh-context checker's report on an earlier draft. `lens_results.json` and `lamq_refutation_results.json` hold the lens, skeptic and refutation results. Their confirmed corrections are in the posted record. They quote single source lines as evidence; no engine file or block is reproduced.
  - The local items named in the record's last Limits bullet are retained here, apart from the sources.
  - Not included, and listed in `SOURCES.json`:
    - the extracted archive sources and the two archives (DG-MIGRATION object `a2136bc0`, and git blob `c4065c67`);
    - 01Fi's note and the contender-check scan, which are other lanes' text;
    - the full scan with quoted source lines;
    - the agents' scratch, transcripts and journals;
    - own working lists.
- **m322_ledger_20261024.**
  - `pickup.md`, `ptr475.md`, `status.md`, `mirror539.md`, `record.md` and `mirror.md` are the texts as sent; `posts.json` maps them to their comment ids. The `*_posted.txt` files are GitHub readbacks, each with one more blank line before the footer than the sent text and otherwise identical.
  - `own/ledger_rows_final.json` and `own/ledger_table.md` are the 82-row ledger as posted (blob column from `git ls-tree` at Math- `d1f6ad28`). `own/ledger_raw.json` holds every reader's full classification and every skeptic's verdict with their line-numbered evidence; the one accepted skeptic correction (`downstream-gate.yml`, stderr label) is applied only in the final rows.
  - `own/prepass.py` and `own/prepass.json` are the heuristic first pass over the 81 files (text patterns only); `own/status_map.json` is the board-held repair and claim status joined into the table; `own/rc_files.txt`, `own/lastcommit.tsv` and `own/shell_candidates.txt` are the working lists.
  - `workflow_results.json` is extracted from the batch run's journal (12 readers, 12 skeptics). The run's script is `workflows/aud-legacy-mutant-ledger-wf_c4d925e1-36a.js`.
  - Not included, and listed in `SOURCES.json`: the 81 extracted workflow copies (byte-identical to the Math- blobs in the table), the readers' scratch notes, other lanes' text, intermediate build files and the agents' transcripts.
- **dbx_crosswalk_20261024.**
  - `pickup.md`, `result.md`, `pr_pointer.md`, the `amend*.md`, `status539*.md` and `supersede*.md` files, `close590.md`, `ci590.md` and `not569.md` are the texts as sent, and the matching `*_posted.txt` files their readbacks; `pr_body.md`, `pr_body_v2.md`, `pr_body_590.md` and `pr_body_594.md` are the PR descriptions as sent. `posts.json` maps the posts to their comment ids; `carriers.json` records the three draft PRs (main#589, main#590, main#594), their heads, the three nonauthor reviews and the owner and CoS decisions, by id. `published/` holds the README, `SUMMARY.json`, `PINS.json` and `HASH_UNAVAILABLE.tsv` of main#594 head `06c4a1a0` (byte totals rounded to 100 KB, the two unnamed sizes withheld; the large TSVs are only in the PR).
  - The retained PR descriptions (`pr_body*.md`), `result.md`, `result_posted.txt` and `workflow_results.json` carry `[size withheld]` where the texts gave the exact size of the two hash-refused unnamed files (review 5640045547, F1), each recorded as `hand_edited`; the live descriptions were edited the same way, the posted RESULT comment is a record and was not edited, and the copies in this branch's commits `9057f7f5` and `2cc5de93` carry that size. The copies of `README.md` and `SUMMARY.json` in this branch's earlier commits, through `9057f7f5`, are the pre-rounding `6c01a832` versions with exact totals, the exposure that reviews 5639778197 (F1a–c) and 5639987376 (H1) name on the carrier PRs. This session does not rewrite a pushed branch, so they stay in this branch's history; the owner or the CoS decide its disposal.
  - `workflow_results.json` holds the one-line returns of the 61 hash-retrieval subagents and the pipeline review's lens findings and skeptic verdicts. In it, Dropbox namespace ids, quoted Dropbox paths and unnamed Dropbox file names are withheld (`hand_edited` in `MANIFEST.json`); 21 skeptics stopped at a usage limit, so their findings stay UNVERIFIED there.
  - Not included, and listed in `SOURCES.json`: the path-level inventory, hashes and crosswalk (private Dropbox paths; offered to the owner or the CoS), the raw listing pages and subagent TSVs, the Git and archive indexes (rebuildable), the pre-copy build directory and the agents' transcripts.
- **drive_crosswalk_20261024.**
  - `pickup.md` and `result.md` are the texts as sent; the `*_posted.txt` files their readbacks; `posts.json` maps them to their comment ids. `published/` is the guard-passed directory (README, `DRIVE_CROSSWALK.tsv`, `DRIVE_AGGREGATE.tsv`, `LEDGER_DRIVE.tsv`, `ARCHIVE_CARRIERS.tsv`, `SUMMARY.json`, `PINS.json`, `own/`): per-file lines only where the Drive title is already on main or the bytes are already public, digests only for public bytes, no Drive id, URL or owner detail (checked with `tools/privacy/drive_id_scan.py` from main), `03_PERSONAL` counted only.
  - `workflow_results.json` holds the one-line returns of the ten listing agents and the review lenses' findings and skeptic verdicts. Not included, and listed in `SOURCES.json`: the id-level crosswalk and live listing, the local copies of the 2026-09-17 publication, and the agents' transcripts.
- **codex_stall_20261024.**
  - `report.md` is the text as sent; `report_posted.txt` its readback. `workflow_results.json` holds the six readers' candidate lists and the eight skeptic verdicts that ran; the readers' notes and transcripts are listed in `SOURCES.json`, not included.
- **Posted texts** in the 21 October units, and the `*_posted.txt` files of `cov_exact_engines_20261022/`, `m322_ledger_20261024/`, `dbx_crosswalk_20261024/` and `codex_stall_20261024/`, are GitHub readbacks, apart from that one recorded redaction. Each such file ends with one newline that the posted comment does not have.

## Caveats

- Drafts are kept as written. Only the bodies posted on GitHub are authoritative; read them back by comment id.
- The output logs record runs in this container and nothing more: they are not project or CI runs.
- Remaining obligations: none held by this session for rows §12, §5 or §18. For §12, an xAI check of §2(b) is now on record (PASS_SCOPED, with one correction), and the D5 `PROOF_INDEX` cross-links stay with the owner route. For §18, the Math- bindings, the G143 binding or restatement, a Git landing of the G147 T6 certificate, and the hardening README and register corrections are owner-routed.
