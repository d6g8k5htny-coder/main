# Recovery delta, 18–20 October 2026: session 01Cz, recovery rows §12 and §5

Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. This is a retention copy for the owner's recovery request, [main#307 / 6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988).
- It continues the 17–18 October delta, `recovery/session_01Cz7WZybv8znP64SpPj6sWY/delta_20261018/` (branch `claude/recovery-01cz-delta-20261018`, draft main#444), and does not repeat it.
- It is kept separate because main#444 is already near the 500-file limit of the `public-intake` check.

- **Not for merge.** Nothing here is a proof, an accepted result or a status change. Scientific effect NONE. Organizational-independence credit 0.
- **What it holds:** this session's own posted texts, comparison and verification results, check scripts and logs behind the unit in the table below, plus the scripts of the three multi-agent runs.
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
| `s5_side24_v33_v34_binding/` | `S5-SIDE24-V33-V34-BINDING-01CZ-20261020`: custody binding record for the SIDE24 V3.3 checkpoint and V3.4 replay archives (recovery row §5) | main#419: pickup 6280913839, checkpoint 6281760247, result pending at this commit; main#275 pointer 6280916731 | custody verified; result pending at this commit |
| `workflows/` | the scripts of the §12 comparison run, the four-lens check of that result post and its re-check, and the §5 lens run | — | method record |

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

## Caveats

- Drafts are kept as written. Only the bodies posted on GitHub are authoritative; read them back by comment id.
- The output logs record runs in this container and nothing more: they are not project or CI runs.
- Remaining obligations: none held by this session for row §12. The result notes that a non-Anthropic check of the comparison would add provider diversity, and the D5 `PROOF_INDEX` cross-links stay with the owner route.
