# Claude audit package recovery — 2026-09-28

**Scientific effect: NONE.** **Review status: REVIEW_REQUIRED.**
Byte custody only. This packet does not accept any theorem, change `STATUS.md`,
`Math-/PROOF_INDEX.md`, `GRAPH.json` or any landing claim, or flip `lemma_closed`.
It does not discharge ENV-RESCOV, ALLCELL-FDZ-Q4, SYM-Fw-jet or OBL-H5-JETMOD.

## What this is

On 2026-09-21, main PR #2 ("Port the Drive research program to git and make it
active", merge `b040bf0c30f33a9de220d19692e8dbcad9a1c5aa`) added a Drive mirror to
`main`. On 2026-09-23, commit `f35eef1b50d9dd86b6cab907551bf3a09a4b0f2a` reverted all
of it: 3,879 files, with no stated reason. That removed the Anthropic/Claude audit
package that surrounds the RN-UNIF and JETMOD carriers the Math- downstream gate records
as `BLOCKED_ABSENT` / `OPEN_HISTORICAL`. The bytes stayed in `main`'s history but
were not on any current branch of the ten account repositories as of 2026-09-28.

This packet restores 37 files that bear on those holes, byte for byte from
`b040bf0c`. It also adds two originals that were only ever on Drive, downloaded on
2026-09-28, each matching its recorded inventory digest and size. It is a targeted
recovery, not a re-landing of the whole mirror:

| Folder here | Original location | Files |
|---|---|---|
| `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/` | `b040bf0c:drive/mirrors/2026-09-15 — KIMI FINAL INTAKE — UPPER2D STAGE E + H5 + ASSEMBLY/05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/`, plus `D1_v2_3_DRAFT/d1_falsify_v4.py.txt` from Drive | 30 + 1 |
| `06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES/` | `…/06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES/` | 3 |
| `HOLD_NOT_FOR_SUBMISSION_2026-09-16/` | `b040bf0c:drive/mirrors/2026-09-16 — HOLD_NOT_FOR_SUBMISSION/` (three JETMOD files) | 3 |
| `QUARANTINED_Q-RN5-MOMENT-004/` | CL-RNU-003 progress report from Drive; its successor `RN5_NEAR_MOMENT_REPAIR.md` from `b040bf0c`; `QUARANTINE_NOTE.md` | 1 + 1 + 1 |

`SOURCE_MAP.json` gives every file's original path and Git blob id. Two kinds of
rename were made so the intake lane accepts the files, and neither changes a byte:

- `.py` and `.jsonl` files carry an added `.txt` suffix. Nothing here is executed.
- `MATH_PUSH_RESULTS_2026-09-16 — H5-JETMOD + RN-UNIF (team + local).export.txt`
  is renamed to `MATH_PUSH_RESULTS_2026-09-16_H5-JETMOD_RN-UNIF.export.txt`.

The files re-hash to their original Git blob ids.

The `_MANIFEST.jsonl.txt` files record the Google Drive file id of each original.

## What the recovered bytes establish about the missing carriers

1. **`CL_ANTHROPIC_BUNDLE_*` was a chat delivery.** `CL-MIRROR-001_MANIFEST.sha256.txt`,
   lines 4–7: the 46 native copies it lists are in "the local bundle
   `CL_ANTHROPIC_BUNDLE_2026-09-16_v3.zip` (delivered in chat …)". It adds that they are
   "also present byte-for-byte inside the operator's own
   `09152026OKComputer_Project_Gap_Closure.zip`". The file records a SHA-256 for all 46.
   The v5 bundle of 2026-09-17 is most likely a later delivery through the same channel.
   Its likely location is the owner's claude.ai conversations of 2026-09-15 to 2026-09-17,
   not Drive or Dropbox; both came up empty in the D0 custody audit (Math- #65) and in the
   Dropbox gap hunt (`trial/portable/DROPBOX_GAP_HUNT_20260927/`).
2. **`rnu_env.py` was never uploaded to Drive.** `CL-RNU-001_…2026-09-16.md`, lines 6–7,
   lists it among the receipts "in this folder". The folder's Drive manifest
   (`RN_UNIF_2026-09-16/_MANIFEST.jsonl.txt`) has 13 rows, and `rnu_env.py` is not one of
   them. The same line names six more receipts that are on neither Drive nor Git:
   `rnu_run1.txt`, `rnu_diag.py`, `rnu_white.py`, `rnu_land.py`, `rnu_fdcheck.py`,
   `rnu_dcheck.py`, and their transcripts. The sibling scripts that did reach Drive are
   restored here: `rnu_ds3`, `rnu_ds3_scalar_SUPERSEDED`, `rnu_meanfix`,
   `rnu_chi2_white_v2`, `rnu_t4_push` and `rnu_execute`.
3. **Two Drive originals that were never mirrored to Git are now included.** Both were
   downloaded from Drive on 2026-09-28 and match the `inventory_sha256` and
   `inventory_bytes` recorded in the mirror's `_MANIFEST.jsonl` rows:
   - `d1_falsify_v4.py` (Drive id `1uTcWaYLtJUszT7iBEWzI1Xa6J7_nw9E6`; 20,558 bytes;
     SHA-256 `1bfb5dbc88b842893b5b0365680ccaa9b4893b6c028a950c194d2c1c739e62a6`, the digest
     `D1_V2_3_RECEIPTS.txt` line 3 gives for the D1 v2.3 gate). The mirror left it out only
     as out of scope for that lane. Nothing is executed here.
   - `CL-RNU-003_PIECE1_RUN_PIECE2_CHARACTERISED_2026-09-17.md` (Drive id
     `1aCa-QG9CSrNUB9SUFKISifghSf-41fRy`; 7,407 bytes; SHA-256 `59b8f002…6d53`). It is
     **quarantined** under exclusion `Q-RN5-MOMENT-004` (`DEFECTIVE_SCOPE`: its Section 3
     near-integrand certification claim). It sits in its own folder, next to its successor
     and `QUARANTINE_NOTE.md`. The exclusion stays in force.
4. **`CHART_SIDE_JETMOD_PLAN.md` exists.** It is in `main`'s object history as blob
   `9446daa2471b21f9eba68d2ebd2fd3b88c553cd2` (11,663 bytes) and is restored here. It is
   a plan, not the certified 24-jet band enclosure, so OBL-H5-JETMOD stays open.
5. **`allcell_fdz_enclosures.json` is still unexplained.** Nothing restored here names
   where it came from.

## Screening

Privacy, per `governance/OP-PRIVACY-20260927.md`:

- no e-mail addresses, phone numbers, street addresses, share-link access keys, local
  user paths or credentials were found;
- every Drive path is in lane `01_ACTIVE_RESEARCH_PACKAGES`;
- the Git-sourced bytes were already public in `main`'s history;
- the two Drive originals came from anonymous public download links, and the only URLs
  they contain are to public academic references.

Intake rules checked locally before pushing: safe paths, allowed suffixes, per-file
and total size, UTF-8 text, strict JSON.

## Provenance

Recovered by Anthropic Claude (Claude Code) on 2026-09-28 at Dylan Roy's explicit
instruction in the session that found the gap. Same-account authorship: no
independence credit. The files' own authors are as stated inside them; this packet
changes no authorship.
