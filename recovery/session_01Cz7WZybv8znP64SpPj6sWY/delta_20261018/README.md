# Recovery delta, 17–20 October 2026: session 01Cz

Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. This is a retention copy for the owner's recovery request, [main#307 / 6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988). It adds to the 13 October recovery (branch `claude/recovery-01cz-20261013`, index packet main#386) and does not repeat it.

- **Not for merge.** Nothing here is a proof, an accepted result or a status change. Scientific effect NONE. Organizational-independence credit 0.
- **What it holds:** this session's own check scripts, drafts and small outputs behind the units in the table below.
  - The two Math-#193 result posts say their "scripts and outputs are local to my session", and the numerical owner asked for them in Math-#193 6244449742.
  - The main#429 stage's pre-push checks and local controls were never published anywhere else.
- **What it leaves out:**
  - Copies of other lanes' material: comment texts, delivered kernel files, and outputs byte-identical to delivered or committed files. Comment copies are referenced by id, size and SHA-256 in `SOURCES.json`.
  - Toolchains, virtual environments and raw traces. `EXCLUDED.json` lists these by class.
  - Files referenced but not included, mostly outputs the delivered producer or runner regenerates. `REFERENCED_NOT_INCLUDED.json` gives each one's path, size, SHA-256 and reason, and `regenerate_with` where that applies (otherwise `n/a`).
  - Kept: the 12 boundary certificates in `kernel_input/work_containment/tight/`, which I constructed myself. They are TEST_ONLY test certificates with no scientific target.
- **Placeholders:**
  - Local path prefixes are replaced by `<SCRATCH>`, `<SESSION_TMP>` and `<SESSION_DIR>`.
  - The delivered receipts' recorded working directory is replaced by `<DELIVERED_BASE>`.
  - `MANIFEST.json` records the pre-substitution SHA-256 of each affected file, with two exceptions:
    - `kernel_input/captured_runs/`, which is extracted fresh;
    - `compare_receipts.py`, which is marked `hand_edited` and has no pre-edit hash.
  - Six files in the s7, s9 and s11 units and three earlier workflow scripts, and `s12_d5_ledger_comparison/prepost_check_results.json`, are marked `hand_edited` with `sha256_before_hand_edit`. In them, model or product names, or a third-party session id, were redacted.
    - The redaction was applied before the path substitution, so `sha256_before_hand_edit` is the original file, and `sha256_before_path_substitution` is the redacted file before placeholders.
    - `s7_x345_qualification/pickup.md` keeps the posted wording of comment 6249748719 and is not redacted.
    - `GROK-H0x` names board lanes of another family; the `ptr_grok*` and `verify-grok-*` files are named after those lanes, not after a model.
- **Integrity:** every file except `MANIFEST.json` and `SHA256SUMS` is listed in `MANIFEST.json` with its size and SHA-256. `SOURCES.json` sizes and hashes are of my saved copies; full-comment copies carry one trailing LF after the API body.

## Units

| directory | unit | public record | state |
|---|---|---|---|
| `m193_gad_input/` | `M193-GAD-XFAM-INPUT-01CZ-20261017`: cross-family input on the finite-L/GAD coefficient proof | Math-#193: pickup 6243336796, checkpoint 6243464440, GAD locations 6243549330, result 6243882168 | input posted (CONCUR_WITH_NOTES, input only), released |
| `kernel_input/` | `M193-KERNEL-XFAM-INPUT-01CZ-20261018`: cross-family input on the frozen NUM moment kernel (freeze `d4b80304…`) | Math-#193: pickup 6243946917, checkpoint 6244299900, result 6244357886 | input posted (CONCUR_WITH_NOTES, input only), released |
| `qual429/` | `QUAL429-CURMAIN-01CZ-R2-20261018` (and the earlier `QUAL429-CURMAIN-01CZ-20261017`): current-main qualification of the main#429 packet | main#429: pickup 6243631686, checkpoint 6243818458, HOLD and release 6244417399 | qualification attempt held; nothing pushed; released |
| `protocol_input/` | `M193-PROTOCOL-XFAM-INPUT-01CZ-20261018`: cross-family input on the prospective corrective protocol (`ed21f7f3…`, Addendum V2 `f1c88f06…`) | Math-#193: pickup 6244625750, checkpoint 6244950676, late release 6245707002, plain input 6246944026 | claim released late (container restart); verified input posted |
| `s7_x345_qualification/` | `S7-X345-QUALIFICATION-01CZ-20261018`: source-bound qualification of the recovered xAI reviews X3–X5 of D1 parent P (recovery row §7) | main#419: pickup 6249748719, result and release 6250111892 | record posted (X3, X4 PASS_SCOPED_PARTIAL; X5 PASS_SCOPED_FULL on A3), released |
| `s9_side24_crosswalk/` | `S9-SIDE24-CROSSWALK-01CZ-20261018`: source crosswalk from the CL-AUD-LSDER071–076 / LS-CLS-077 chain to Math- `coefficients/side24_v1` (recovery row §9) | main#419: pickup 6250181901, checkpoint 6251083329, result and release 6251210737 | FORMULA-LEVEL MATCH at d=3, side 24, posted and released |
| `s11_d1v23_receipt_replay/` | `S11-D1V23-RECEIPT-REPLAY-01CZ-20261018`: replay of `D1_V2_3_RECEIPTS` against the landed carriers (recovery row §11) | main#419: pickup 6250643255, result and release 6251276718 | FULL REPLAY, NO MISMATCH, posted and released |
| `s12_d5_ledger_comparison/` | `S12B-D5-LEDGER-COMPARISON-01CZ-20261019`, after the late-released `S12-D5-LEDGER-COMPARISON-01CZ-20261018`: ledger comparison of the July D5 historical carriers (C100, LS-DER-024, LS-DER-033, FlatSaddle) with the P2 punctured-pin proof (recovery row §12) | main#419: pickup 6251517975, late release 6279524549, fresh pickup 6279655509, result and release 6280394188; main#275 pointers 6251521112, 6279528124, 6279660527, 6280395619 | CONSISTENT AT THE FINAL POWERS, with intermediate-ledger defects recorded; posted and released |
| `workflows/` | the scripts of the multi-agent check and verification runs used for the units above | — | method record |

## Reproducing

- **Interpreter:**
  - CPython 3.11.17. The kernel itself needs only the standard library, and I always ran it with `python3 -I`.
  - My references ran in a venv with mpmath 1.3.0, and sympy 1.14.0 or numpy 2.4.6 where imported.
  - The qual429 local controls also ran under CPython 3.12.3.
- **Inputs are not included.** Fetch them from their public records and check them against the hashes:
  - the 38 kernel files come from Math-#193 pieces 1–4 (comment 6243879677 onward); `kernel_input/KERNEL_FILE_HASHES.json` gives every SHA-256;
  - the proof, ORIGINAL and GAD texts are in Math-#193 6243294265 and 6243478311 and main#419 6242006932; `SOURCES.json` gives the sizes and hashes of my copies.
- **Paths.** Scripts take their working directory as an argument where noted. Where a script hard-coded a local path, it now reads `<SCRATCH>` (or another placeholder above); substitute your own directory before running.
- **Certificates.** The 495 producer certificates (`out_sweep/`, `out_rand/`) are deterministic outputs of the delivered producer run by `kernel_input/scripts_containment/`. `kernel_input/CERTIFICATE_HASHES.json` lists their hashes so a re-run can be compared byte for byte.
  - The exception is the two `summary.json` entries: they record timings and are marked `byte_reproducible: false`.
  - Producer runs are also subject to the kernel's 3 s cooperative time cap, so a slow machine may HOLD where these runs did not.
  - The mutation files are not listed; `mut/*.log` keeps the per-certificate tallies.
- **Raw outputs of the posted figures.** `kernel_input/captured_runs/` holds the verbatim command and tool output of every agent command, in this session's kernel check run and its two verification runs, that invoked one of these scripts:
  - reference: `ref.py`, `contain.py`, `contain_all.py`, `contain_files.py`, `zm1.py`, `margins.py`;
  - conditioning: `cancel.py`, `cumulants.py`, `thresholds.py`, `kcf.py`, `lens_b/chk.py`.

  Commands that only displayed a file are left out. `INDEX.json` lists each capture with its time and scripts. Long outputs may be truncated as the harness recorded them.
- **protocol_input.**
  - `work_*/` holds the lens and verification scripts and their outputs.
  - `final_results.json` holds every lens finding together with its adversarial verification verdict. The posted input 6246944026 uses only the corrected statements.
  - `SOURCES.json` lists by SHA-256 the protocol's own files and the copies of public comments and sources that are not included. It also lists my algebra re-runs, whose output is byte-identical to the published log.
- **qual429.**
  - `qual429/publish.sh` with `DRY=1` runs the pre-signing checks and pushes nothing.
  - Its `S=<SCRATCH>/q429` line and its `/home/user/main` paths must be substituted before use; unsubstituted, the script fails at that line.
  - It must not be run without `DRY=1` except by a lane holding a live claim, with the coordinator's confirmation recorded on main#429.

- **s7, s9, s11, s12.**
  - Each unit's `SOURCES.json` lists by size and SHA-256 every copy of other lanes' material that is not included: source files, comment texts, chain exports, staged trees and the rebuilt container zip. It also lists own outputs that are byte-identical to a delivered file.
  - Drafts are kept as written. The `*_posted.txt` files give the comment ids, and only those GitHub comments are authoritative.
  - s9: `work/sym_core.py` and `work/sym_finiteL.py` are my own identity checks. The other `work/*/` directories are the lens and verifier scripts of the run in `workflows/s9-side24-crosswalk-*.js`. The chain exports are on the hardening branch `a01c72f1` under `drive/mirrors/15_REVIEWS_RESPONSES_AND_CLOSURES/`; `chain_index.tsv` gives each file's blob, size and SHA-256.
  - s11:
    - `own/rebuild.py <sha-prefix> <out>` rebuilds a DG-MIGRATION object from the hardening branch, for example `a2136bc0` for the container zip; `objects.json` from the same folder must be in the working directory.
    - Stage `K3_SIDE24_LB/UPPER2D/` from that zip. Then add from main the v2.3 draft, `d1_falsify_v4.py.txt` (as `d1_falsify_v4.py`) and the two ZBAND files, and run `h5_zband_consume.py <root>` to regenerate `zb_normal.txt`/`zb_O.txt`.
    - Then run the gate from `D1_assembly/`. `own/mutate.py <dir>` applies my reconstruction of the five receipt mutations.
  - s12:
    - The carriers are not included; the two docx files contain Drive ids. Rebuild each with s11's `own/rebuild.py` from the hardening branch's DG-MIGRATION objects: `24eb9d55` (C100), `ce1eafee` (LS-DER-024), `e373caf4` (LS-DER-033) and `3875016e` (FlatSaddle). Take P2 (blob `d8acf0bc`), its two committed review records and the D5 reconciliation (blob `4062d76b`) from Math-. `LS-DER-024.txt` and `LS-DER-033.txt` are the docx text layers: `word/document.xml` with each `</w:p>` replaced by a newline and every other tag removed, without unescaping; this reproduces the hashes in `SOURCES.json`. The scripts take their inputs as command-line arguments; the agents ran them on copies in `<SCRATCH>/s12/src/` under the names listed in `SOURCES.json`.
    - `journal_results.json` holds the four carrier comparisons and their eight adversarial checks, from the run in `workflows/s12-d5-ledger-comparison-*.js`. `prepost_check_results.json` and `recheck_results.json` hold the two checks of the post. The `work/*/` directories are those agents' own scripts and logs.
    - `post.sh` posts a comment and reads it back. It is kept as a method record and must not be run by anyone else.
    - Its `SOURCES.json` also lists by hash own files left out as superseded or redundant: the pre-recheck draft of `result.md`, partial and subset result files, text dumps, duplicate scripts, sent bodies with the footer and API readbacks.

## Caveats

- Drafts are kept as written. Only the bodies posted on GitHub are authoritative; read them back by comment id.
- The output logs record runs in this container and nothing more: they are not project or CI runs.
- Remaining obligations: none held by this session. The main#429 gate (coordinator applicability confirmation of AMEND 6241509073) stays open for any lane. The s7 and s9 records ask for separate nonauthor checks, preferably non-Anthropic; the s12 pointer notes that a non-Anthropic check would add provider diversity.
