# RN-UNIF recovery index — 2026-10-25

**Scientific effect: NONE.** **Review status: REVIEW_REQUIRED.**
This packet holds byte copies and an index, nothing more. It accepts no theorem and
changes no `STATUS.md`, `Math-/PROOF_INDEX.md`, `GRAPH.json` or landing claim. It
does not discharge D3-LEMMA-RN-UNIF, `SYM-Fw-jet`, `ENV-RESCOV`, `ALLCELL-FDZ-Q4`
or OBL-H5-JETMOD. Nothing here was executed or replayed, so every numerical
statement below is the source file's own claim.

## Why this exists

On 25 October 2026 Dylan Roy asked for everything that bears on the RN-UNIF
recovery checklist ([main#86/6398744152](https://github.com/d6g8k5htny-coder/main/issues/86#issuecomment-6398744152))
to be available to every model family on GitHub, Google Drive or Dropbox, with
links. In this session he also said the Drive links need not be kept off
GitHub. The CoS relays the same current directive in
[539/6398833595](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6398833595).

## Headline

**None of the five checklist originals has been found.** These are:

- the `CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip`;
- the original `rnu_env.py` and `rnu_white.py`;
- the certification cell log and queue;
- the all-cell input tables;
- the completed far-region bound.

The searches are recorded in:

- [86/6398780395](https://github.com/d6g8k5htny-coder/main/issues/86#issuecomment-6398780395) (Claude 01Cz);
- [539/6398781202](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6398781202) (Grok agent 13);
- [86/6399770804](https://github.com/d6g8k5htny-coder/main/issues/86#issuecomment-6399770804) (Claude S, the session that assembled this packet).

There are dated negatives at the time:

- **16 Sep census:** `rnu_env.py` and `rnu_white.py` were recorded as absent from the Drive hub (`records_2026-09-16/`).
- **17 Sep lookup:** a bounded lookup by the RN5 lane did not return the v5 zip, `rnu_t4.py` or `rnu_spine.py` ([`RN5_NEAR_MOMENT_REPAIR.md` line 138](https://github.com/d6g8k5htny-coder/main/blob/main/incoming/claude-audit-package-recovery-20260928/QUARANTINED_Q-RN5-MOMENT-004/RN5_NEAR_MOMENT_REPAIR.md#L138), on main).
- **22 Sep hunt:** a Drive and local hunt found none of `rnu_env.py`, the v5 zip or `allcell_fdz_enclosures.json` (`status_walls_2026-09-22/`).

These are dated negatives. They do not prove that the files never existed or never reached Drive.

This packet does two things:

- It puts the nearest existing material on GitHub as plain files. Until now
  that material sat inside zip files or only in Git history.
- It gives every other known location as a link.

## What is in this packet

| Folder | Files | What it is | Where the bytes come from |
|---|---:|---|---|
| `kimi_D3_percolation_2026-09-14/` | 21 | The D3 engine directory of 14–15 Sep. See the breakdown below. | Members of `CLOSE-20260917-b9c2_PROOFS_CODE_AND_VERIFICATION.zip` under `intake/rn_source/K3_SIDE24_LB/UPPER2D/D3_percolation/`. The zip is git blob `c4065c67` in main's history at `b040bf0c`. Each file has the same SHA-256 as the member of the same name at `K3_SIDE24_LB/UPPER2D/D3_percolation/` in the 15 Sep operator zip `09152026OKComputer_Project_Gap_Closure.zip`, per the 17 Sep Drive inventory (`drive/source_map/Archive_Members.csv` at `b040bf0c`). |
| `records_2026-09-16/` | 3 | The 16 Sep census of the Drive `RN_UNIF_2026-09-16` hub, the code census, and the whitened env-form note that records `rnu_env.py` and `rnu_white.py` as absent. | Main's history at `b040bf0c`, under `drive/mirrors/2026-09-16 — HOLD_NOT_FOR_SUBMISSION/` |
| `rn5_round5_2026-09-17/` | 2 | `README.md` and `DELIVERY_MANIFEST.json` of OpenAI's round-5 bundle | Members of `RN5_REPAIR_AND_ERRATUM_BUNDLE.zip` (git blob `0002d3a7`, at `b040bf0c`) |
| `status_packet_2026-09-22/` | 7 | The first 22 Sep `docs/math_status` packet. Its paths name the `drive_peer_review_triage/code_prototypes/` receipts. | `docs/math_status/` at `4b3e47d1`, on the `chatgpt/c101-source-bound-successor-20260925` branch. The blobs are the same as Cursor's `355a9f25`. |
| `status_walls_2026-09-22/` | 3 | The 22 Sep evening RN-UNIF walls, where `SYM-Fw-jet` is first named, and their 23 Sep revision | `docs/math_status/STATUS_RN_UNIF.md` and `STATUS.md` at `6dba415e` (squash of Cursor's `b6069117`); `STATUS_RN_UNIF.md` at `64c3c176` |

**The D3 engine folder holds:**

- **RN-UNIF engine:** `d3_rn_unif.py` and `d3_rn_unif_dbg.py`.
- **Base D3 percolation run:** `d3_perc.py`, `D3_PERCOLATION.md`, `FREEZE.txt` and the `t_*` transcripts.
- **Remote amendment v1:** `d3_amend.py`, `D3_REMOTE_AMENDMENT.md`, `FREEZE_AMENDMENT.txt` and `ta_*`.
- **Remote amendment v2:** `d3_amend_v2.py`, `FREEZE_AMENDMENT_v2.txt` and `ta2_*`.
- **Falsifier:** `d3_falsifier.py` and `tf_*`.
- **Perc-decay transcripts:** `pd_*`.

The file headers say "Agent D3, U2D-UPPER". The attribution to the Kimi lane comes from the carrier: the operator zip in the Drive KIMI FINAL INTAKE folder.

`d3_falsifier.py` and `tf_*` are the PERC-DECAY versions, and those are pinned by `FREEZE_PERC_DECAY.txt`. `FREEZE.txt` and `FREEZE_AMENDMENT_v2.txt` pin earlier falsifier versions that are not included here.

**The directory's other four files are already on main:**

- [`D3_REMOTE_AMENDMENT_v2.md`](https://github.com/d6g8k5htny-coder/main/blob/main/incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/UPPER2D/D3_percolation/D3_REMOTE_AMENDMENT_v2.md)
- [`FREEZE_PERC_DECAY.txt`](https://github.com/d6g8k5htny-coder/main/blob/main/incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/UPPER2D/D3_percolation/FREEZE_PERC_DECAY.txt)
- [`d3_perc_decay.py.txt`](https://github.com/d6g8k5htny-coder/main/blob/main/incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_perc_decay.py.txt)
- [`PERC_DECAY.md`](https://github.com/d6g8k5htny-coder/main/blob/main/incoming/claude-audit-package-recovery-20260928/06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES/PERC_DECAY.md)

**Two members of the operator zip's directory are not vendored**, because `.pyc` is not an intake suffix. Both are in the operator zip on Drive and in the Dropbox `__pycache__/` folder below:

- `__pycache__/d3_amend.cpython-312.pyc`: 34,835 B, SHA-256 `420eef1534926c9b455ea569e025f24268ec9b9140ed1c192b82da0830126f34`
- `__pycache__/d3_perc.cpython-312.pyc`: 64,378 B, SHA-256 `eb65dad9ec6558777d03feab0ed20e042213cc5c433a37c4ccacf8b4a7423551`

**Renames.** None of these changes a byte:

- Each `.py` file has `.txt` appended, because the intake lane accepts only text suffixes. Nothing here is meant to be run from this folder.
- The intake lane protects the bare file name `STATUS`, so:
  - `docs/math_status/STATUS.md` at `4b3e47d1`, which is the 21 Sep RN-UNIF note, is stored as `status_packet_2026-09-22/STATUS_RN_UNIF_2026-09-21.md`;
  - the same file at `6dba415e`, with its 22 Sep evening section, is stored as `status_walls_2026-09-22/STATUS_RN_UNIF_2026-09-21_with_evening_walls.md`;
  - the 23 Sep `STATUS_RN_UNIF.md` is stored as `STATUS_RN_UNIF_2026-09-23.md`.

`SOURCE_MAP.json` gives every file's source commit, source path, archive member,
git blob id, byte count and SHA-256. For every file, it also lists the 17 Sep
Drive-inventory rows that hold the same bytes, from both `Archive_Members.csv`
and `Files.csv`.

## Checklist index

### 1. `CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip` or its delivery message

**ABSENT.**

- **The report that names its contents:** [CL-RNU-003 (17 Sep)](https://github.com/d6g8k5htny-coder/main/blob/main/incoming/claude-audit-package-recovery-20260928/QUARANTINED_Q-RN5-MOMENT-004/CL-RNU-003_PIECE1_RUN_PIECE2_CHARACTERISED_2026-09-17.md), quarantined under `Q-RN5-MOMENT-004`. Its Drive original is [`CL-RNU-003_PIECE1_RUN_PIECE2_CHARACTERISED_2026-09-17.md`](https://drive.google.com/file/d/1aCa-QG9CSrNUB9SUFKISifghSf-41fRy/view).
- **A 25 Sep navigation note, not a copy:** it classifies the report's v5 import as STILL_CONDITIONAL (PR #100) and is [here](https://docs.google.com/document/d/1GA5EjY29yD9Md4XNfKDyVwQHZ0qcnaPFDwtkPas-xDM/edit).
- **Earliest Git copy:** `round5/intake/cl.txt` inside [`RN5_REPAIR_AND_ERRATUM_BUNDLE.zip`](https://github.com/d6g8k5htny-coder/main/blob/a38ad1edd04ce2451274811731819ea4e4661ceb/drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_REPAIR_AND_ERRATUM_BUNDLE.zip), committed 19 Sep. It is byte-identical to the report (SHA-256 `59b8f002…`). The zip is also [on Drive](https://drive.google.com/file/d/1g5UP_KYlvPmx7LFwjGk6qK5QZLnL1hLj/view).
- **Dated negatives:**
  - 17 Sep, [`RN5_NEAR_MOMENT_REPAIR.md` line 138](https://github.com/d6g8k5htny-coder/main/blob/main/incoming/claude-audit-package-recovery-20260928/QUARANTINED_Q-RN5-MOMENT-004/RN5_NEAR_MOMENT_REPAIR.md#L138);
  - 22 Sep, `status_walls_2026-09-22/STATUS_RN_UNIF.md` ("Absent historical objects").
- **Likely location:** the v3 bundle was delivered in chat. The v5 bundle is most likely in the owner's own claude.ai conversation of 16–17 Sep, or on a device that saved the attachment.

### 2. Original `rnu_env.py`, `rnu_white.py`, `rnu_run1.txt`, `rnu_diag.py`, `rnu_land.py`, `rnu_fdcheck.py`, `rnu_dcheck.py`

**ABSENT.**

- **The Drive hub set, on main:** [this folder](https://github.com/d6g8k5htny-coder/main/tree/main/incoming/claude-audit-package-recovery-20260928/05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/RN_UNIF_2026-09-16), with CL-RNU-001 among the files. The hub's 13th Drive file, CL-RNU-003 PIECE1, is under the quarantine folder.
- **The hub on Drive:** the live folder is [`RN_UNIF_2026-09-16`](https://drive.google.com/drive/folders/1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG), and its parent is [`05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15`](https://drive.google.com/drive/folders/1J7Ly5v-XQWTT_AqqAGfsr1kdXmwyxvE3).
- **Dated negatives:**
  - `records_2026-09-16/FULL_ZIP_AND_CODE_CENSUS.md` §2.1 hashes 14 local extract files at about 21:45 CT on 16 Sep. That is 12 of the hub's files plus two copies of CL-GROK-CLOSE-001 from the parent folder.
  - `records_2026-09-16/MATH_PUSH_WHITENED_ENV_FORM_2_4.md` line 17 says "still ABSENT on Drive hub and local extract". Its Drive copy is [here](https://drive.google.com/file/d/1pnnSC2E-FY5tjLa4z60UOSjOGdiKcAIe/view).
  - `status_walls_2026-09-22/STATUS_RN_UNIF.md` records the 22 Sep hunt.
- **A later reconstruction, not the original:** `output/rn_math/rnu_white.py` (13,344 B) inside the b9c2 zips. See [CLOSE b9c2](https://github.com/d6g8k5htny-coder/main/blob/b040bf0c30f33a9de220d19692e8dbcad9a1c5aa/drive/mirrors/2026-09-16%20%E2%80%94%20HOLD_NOT_FOR_SUBMISSION/CLOSE-20260917-b9c2_PROOFS_CODE_AND_VERIFICATION.zip) and [MATH b9c2 on Drive](https://drive.google.com/file/d/1TJr1awRjB0D9niq1Jyi6XGnJKuvW9Wmx/view).

### 3. Original certification outputs: `certify_cells.jsonl`, `certify_queue.json`, transcripts, manifests, final freeze record

**ABSENT.**

- **The only record:** the fingerprint text in CL-RNU-003, "1,170 cells closed, 561 base cells pending, max certified bound 0.679737". That was progress at the time of writing, not a completed run.

### 4. Original covariance/mean derivative bounds through order four, all-cell enclosures, 24-jet, `ENV-RESCOV`, `ALLCELL-FDZ-Q4`

**ABSENT as originals.**

- **Origin of the file name:** 01B6's [RECOVERY_MATRIX](https://github.com/d6g8k5htny-coder/main/blob/d85e665dc46b380fb36bdd90772b853db1fdb424/recovery/session_01B6fUVkWEba9KPrZjfHujFq/historical_recovery/results_20261013/RECOVERY_MATRIX.md), row G05, records that the file name `allcell_fdz_enclosures.json` (blocker `ALLCELL-FDZ-Q4`) first appears on 22 Sep (main `b6069117`), with no source. The obligation itself is already listed on 17 Sep, in §6 of the b9c2 `RN_WHITENED_JET_THEOREM`. dot's checklist treats that file name as a later search alias.
- **Later partial substitute only:** [`CLOSE-20260917-b9c2_RN_FIELD_PROOF.md`](https://github.com/d6g8k5htny-coder/main/blob/c1b3b222e80909f23be3e952d69ee1e6e1780c54/drive/mirrors/2026-09-16%20%E2%80%94%20HOLD_NOT_FOR_SUBMISSION/CLOSE-20260917-b9c2_RN_FIELD_PROOF.md). It reports enclosures of B, F, z and D at r = 1/20.
- **Near-miss, for a math reader:** the table `phase1_cells.jsonl`, which has four-entry jets. It is in the 15 Sep operator zip, and its Drive reading copy (owner-only) is [here](https://docs.google.com/spreadsheets/d/1e1mh8H6O8UBRDPdMbGXGn29X1RKaYtMnrJTKEb2CC_U/edit). It sits in a W8 lambda-phase-2 tree and is believed not to be the RN-UNIF input.

### 5. Original full far-region proof or code (`SYM-Fw-jet`, `κ_far`, D3-LEMMA-RN-UNIF, Piece 1, 0.6798)

**The completed bound is ABSENT.** The nearest existing code is in this packet.

- **Where 0.6798 comes from:** `kimi_D3_percolation_2026-09-14/d3_rn_unif.py.txt`, lines 2144–2145, set `KAP_BUDGET = mpf('0.68')` and `CELL_MARGIN = mpf('0.0002')`. Line 2167 passes a cell only when `sup <= budget - CELL_MARGIN`.
- **What the engine defines:** `kappa_far_point` (line 788) and `run_certification` (line 2184).
- **No completion record:** the operator zip has no run transcript for this engine.
- **Amendment v2:** `FREEZE_AMENDMENT_v2.txt` (line 47) and `ta2_normal.txt` report `kappa_far = 0.677284905`, "certified <= 0.68" at the R2 floor, at the axis-worst station. This was not replayed here. `FREEZE_AMENDMENT_v2.txt` line 49 leaves D3-LEMMA-RN-UNIF "still NAMED for the zone uniformity", and `D3_REMOTE_AMENDMENT_v2.md` line 67 says "NAMED for the ZONE UNIFORMITY".
- **Later alternatives** (neither authenticates the older expression):
  - [RN3 `RN_JOINT_PROOF`](https://github.com/d6g8k5htny-coder/main/blob/c1b3b222e80909f23be3e952d69ee1e6e1780c54/drive/mirrors/2026-09-16%20%E2%80%94%20HOLD_NOT_FOR_SUBMISSION/RN3-20260917-b9c2_RN_JOINT_PROOF.md) claims < 0.12058 at r = 1/20. This is author-side, and its review is still open.
  - [CLOSE `RN_FIELD_PROOF`](https://github.com/d6g8k5htny-coder/main/blob/c1b3b222e80909f23be3e952d69ee1e6e1780c54/drive/mirrors/2026-09-16%20%E2%80%94%20HOLD_NOT_FOR_SUBMISSION/CLOSE-20260917-b9c2_RN_FIELD_PROOF.md) says "The full far budget κ_far<0.6798 is not proved".
- **`SYM-Fw-jet`:** the name first appears, without a definition, as the fourth ordered blocker in `status_walls_2026-09-22/STATUS_RN_UNIF.md` (22 Sep evening): ENV-RESCOV → ALLCELL-FDZ-Q4 → LOGQ-TAIL → SYM-Fw-jet → CH-LIFT.
  - 01B6's RECOVERY_MATRIX row G07 records that no definition has been located.
  - 01B6's [FINAL_ASSESSMENT](https://github.com/d6g8k5htny-coder/main/blob/d85e665dc46b380fb36bdd90772b853db1fdb424/recovery/session_01B6fUVkWEba9KPrZjfHujFq/historical_recovery/results_20261013/FINAL_ASSESSMENT.md#L44), item 5, believes the `code_prototypes/` receipts would define it. See open lead 1.

## Other places (access noted per link)

**GitHub (public):**

- Earlier recovery packets on main: [`claude-audit-package-recovery-20260928`](https://github.com/d6g8k5htny-coder/main/tree/main/incoming/claude-audit-package-recovery-20260928) and [`claude-bundle-v3-native-copies-20260928`](https://github.com/d6g8k5htny-coder/main/tree/main/incoming/claude-bundle-v3-native-copies-20260928).
- 01Cz's search data and shared custody: branch [`claude/shared-custody-01cz-20261025`](https://github.com/d6g8k5htny-coder/main/tree/claude/shared-custody-01cz-20261025/custody/claude-01cz-20261025), Draft [main#603](https://github.com/d6g8k5htny-coder/main/pull/603).
- 01B6's 13 Oct historical recovery: [FINAL_ASSESSMENT](https://github.com/d6g8k5htny-coder/main/blob/d85e665dc46b380fb36bdd90772b853db1fdb424/recovery/session_01B6fUVkWEba9KPrZjfHujFq/historical_recovery/results_20261013/FINAL_ASSESSMENT.md).

**Google Drive.** Links marked "anyone with the link" open for any reader. Links marked "owner-only" need the owner's Drive connector.

- [KIMI FINAL INTAKE folder](https://drive.google.com/drive/folders/1bRmeImIMT-6kc-zBNZVoWW1aYNPBPLWs), anyone with the link. It holds the operator zip [`09152026OKComputer_Project_Gap_Closure.zip`](https://drive.google.com/file/d/1vSI-evINWskhXVyiZ0slt-rLT74sPpXH/view).
- [START HERE index](https://docs.google.com/document/d/1npDbkQ0yhBegJS5J6cr23-0BtROPvbv__zqJB9Ae61M/edit), anyone with the link.
- [Absent-carrier classification](https://drive.google.com/file/d/1NuKMa4cF4-pl8SLhpD61uchh9LVBSMlf/view), owner-only.
- Codex's 25 Oct handoff, folder ["other phone ChatGPT"](https://drive.google.com/drive/folders/14NFE6ng7qqL4hkbwSeNdHmhOjV0BbYAB), owner-only. Its own index says it holds no original. Its audit is main#605.
- 01Cz's shared custody folder: owner-only on Drive. Use the GitHub branch above instead.

**Dropbox.** This is the owner's account. Locations are given as paths only, because Dropbox share links carry access keys.

- `/2026/Research 07-13-26/Research topology 07-14-26/Github/UPPER2D/UPPER2D/D3_percolation/`: loose copies of the 25 D3 files, plus `__pycache__/`.
- `/Shared_Custody_Claude01Cz_20261025/`: 01Cz's shared copies.

## Open leads (contents not yet reported by any lane)

1. **The `code_prototypes/` folder of the workspace `drive_peer_review_triage`.** The workspace was in use from at least 16 Sep. On 25 Oct, Grok agent 13 reported reading its `extracts/` folder ([539/6398781202](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6398781202)). The 16–22 Sep records name these files in `code_prototypes/`:
   - `RN_UNIF_WHITE_BOX_GRAD_BOUND_V1_RECEIPT.json`, `rnu_white_grad_box_envelope_receipt.json`, `rnu_white_hess_calibrate_v1_receipt.json`, `piece2_first_cell_failclosed_receipt.json` and `rnu_chi2_white_exact_grad_receipt.json`. These are in `status_packet_2026-09-22/math_console_snapshot.json`, lines 7–11.
   - `rnu_white_box_grad_bound_v1.py`, in `status_packet_2026-09-22/STATUS_RN_UNIF_2026-09-21.md` line 26 and `math_console.py.txt` line 232.
   - `rnu_whitened_env_form_stub.py` and `RN_UNIF_ENV_FORM_SMOKE_RECEIPT.json`, in `records_2026-09-16/MATH_PUSH_WHITENED_ENV_FORM_2_4.md` lines 23–24.

   Per 01B6, `RN_UNIF_LOGQ_HESS_SKELETON_RECEIPT.json` and `RN_UNIF_LOGQ_CH_COMPOSE_CERT_GATE_V1_RECEIPT.json` belong to the same set. This is asked on [539/6399789021](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6399789021) (this session).
2. **Six zips that exist only in Dropbox,** under `/2026/Research 07-13-26/Research topology 07-14-26/Github/`. None has a Drive or Git twin by size:
   - `09152026OKComputer_Project_Gap_Closure (1).zip`: 31,236,319 B, which is 1,088,034 B larger than the 15 Sep operator zip;
   - `OKComputer_Project_Gap_Closure.zip`;
   - `OKComputer_09132026Project_Gap_Closure.zip`;
   - `Kimi09132026OKComputer_Project_Gap_Closure.zip`;
   - `09142025OKComputer_Project_Gap_Closure.zip`;
   - `Kimi 3.zip`.
3. **[`DG-SEARCH-RN-FAMILIES-20260920-v1.zip`](https://drive.google.com/file/d/1Fs2boHnKAmdXHENcu5DZIaDNqyQNpRYC/view)** (7,260,226 B, 20 Sep, anyone with the link). Its full text matches the checklist names, most likely as a search index.
4. **Unsearched places:** Drive trash, Dropbox deleted files, Gmail spam, and the owner's claude.ai conversations of 15–17 Sep.

The archives in leads 2 and 3 are given as locations, not copies, for two reasons:

- They cannot pass this intake lane, which accepts text suffixes only, at most 256 KiB per file and 2 MiB per package.
- The session that assembled this packet could not download them, because its network policy refuses the Dropbox and Drive download hosts.

## Screening

Privacy, per `governance/OP-PRIVACY-20260927.md`:

- **What the 36 copied files contain:** no e-mail addresses, phone numbers, street addresses, home-directory or personal paths, share-link access keys or credentials, and no URLs.
- **Workspace paths:** the files do contain agent workspace paths, such as `/workspace/drive_peer_review_triage/...` and `/tmp/...`. These were already public at the source commits.
- **Drive and Dropbox locations:** the Drive links in this file, the Drive file and folder ids inside `records_2026-09-16/FULL_ZIP_AND_CODE_CENSUS.md`, and the Dropbox paths are published under the owner's instruction above. The Drive links carry no `resourcekey`. The Dropbox locations are paths, not share links.
- **Intake rules checked locally:**
  - safe paths, allowed suffixes, per-file and total size;
  - UTF-8 text and strict JSON;
  - add-only files in one package;
  - no protected status file names;
  - pinned sources reachable from a public branch.

## Provenance

Assembled by Anthropic Claude Opus 5.5 (Claude Code), session
`session_01S16677s1SGeAJmN3QMz4ip`, at Dylan Roy's instruction on 25 October
2026. Same-account work, so no independence credit. This packet changes no
authorship:

- **D3 directory:** the headers say "Agent D3, U2D-UPPER". The attribution to the Kimi lane comes from the carrier.
- **Records:** two of the three name the "agent 5" executor. `FULL_CODE_CENSUS.md` names no author.
- **RN5 files:** OpenAI, per `DELIVERY_MANIFEST.json`.
- **Status packet:**
  - `README.md` and `PACKET.json` are by Cursor Agent (`355a9f25`, with Dylan Roy as co-author), squashed as `4b3e47d1` by Dylan Roy.
  - The other five files are transcriptions of 21–22 Sep workspace notes. `STATUS_MATH_PUSH_2026-09-21.md` is by the Chief of Staff lane.
- **Status walls:** by Cursor Agent (`b6069117`, `64c3c176`), squashed as `6dba415e` by Dylan Roy.

Rollback: revert the one commit.
