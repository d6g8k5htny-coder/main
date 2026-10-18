# Recovery delta, 17–18 October 2026: session 01Cz

Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. This is a retention copy for the owner's recovery request, [main#307 / 6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988). It adds to the 13 October recovery (branch `claude/recovery-01cz-20261013`, index packet main#386) and does not repeat it.

- **Not for merge.** Nothing here is a proof, an accepted result or a status change. Scientific effect NONE. Organizational-independence credit 0.
- **What it holds:** this session's own check scripts, drafts and small outputs behind three units of work.
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
- **Integrity:** every file except `MANIFEST.json` and `SHA256SUMS` is listed in `MANIFEST.json` with its size and SHA-256. `SOURCES.json` sizes and hashes are of my saved copies; full-comment copies carry one trailing LF after the API body.

## Units

| directory | unit | public record | state |
|---|---|---|---|
| `m193_gad_input/` | `M193-GAD-XFAM-INPUT-01CZ-20261017`: cross-family input on the finite-L/GAD coefficient proof | Math-#193: pickup 6243336796, checkpoint 6243464440, GAD locations 6243549330, result 6243882168 | input posted (CONCUR_WITH_NOTES, input only), released |
| `kernel_input/` | `M193-KERNEL-XFAM-INPUT-01CZ-20261018`: cross-family input on the frozen NUM moment kernel (freeze `d4b80304…`) | Math-#193: pickup 6243946917, checkpoint 6244299900, result 6244357886 | input posted (CONCUR_WITH_NOTES, input only), released |
| `qual429/` | `QUAL429-CURMAIN-01CZ-R2-20261018` (and the earlier `QUAL429-CURMAIN-01CZ-20261017`): current-main qualification of the main#429 packet | main#429: pickup 6243631686, checkpoint 6243818458, HOLD and release 6244417399 | qualification attempt held; nothing pushed; released |
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
- **qual429.**
  - `qual429/publish.sh` with `DRY=1` runs the pre-signing checks and pushes nothing.
  - Its `S=<SCRATCH>/q429` line and its `/home/user/main` paths must be substituted before use; unsubstituted, the script fails at that line.
  - It must not be run without `DRY=1` except by a lane holding a live claim, with the coordinator's confirmation recorded on main#429.

## Caveats

- Drafts are kept as written. Only the bodies posted on GitHub are authoritative; read them back by comment id.
- The output logs record runs in this container and nothing more: they are not project or CI runs.
- Remaining obligations: none held by this session. The main#429 gate (coordinator applicability confirmation of AMEND 6241509073) stays open for any lane.
