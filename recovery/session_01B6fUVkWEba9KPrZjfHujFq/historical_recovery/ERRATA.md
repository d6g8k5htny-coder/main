# Errata to the 13 October historical-recovery results

The files under `results_20261013/` are kept unchanged as published. Corrections are recorded here.

## E1 (17 October 2026): TOP_CANDIDATES.md §10 locator

**The row as published:** "main blob 9b015fbd (q0_twelve zip) m60 (ceiling sources) and m52 (defect record, l.158-184)".

**Corrected.** Members are counted from 0 in the archive order of main blob `9b015fbd` (`research/campaigns/q0_twelve_20260920_v1.zip`).
- The defect record is member **[60] `h3_ceiling/PROOF.md`**, sha256 `1c1f3c1881c45b31ab4916aded37e118f1fd7b88b8500e8c300ed733e40cac6e`. The counterexample text is at lines 158–184, and it cites `iv_Phi` lines 448–455.
- The ceiling source is member **[69] `h3_ceiling/sources/h3_band_ceil.py`**, sha256 `b97c5428a20e5dd33f3e53653d20447f1e27515dcc7dc7a7c2004ebe30029452`, with `iv_Phi` first at line 438.
- Member [52] is unrelated (`gaussian_tube/source/source.raw`).

**Unaffected:** all sha256 values in the row, the Verdict and the Caveats.

**Attribution.** The defect was found by the q0_twelve replay ("author-side mathematical candidate, 2026-09-20", R17 claim `b66c025c-7ed6-4f08-90a5-6755bbe1b433`), not by this recovery. That zip came in through commit `bebee8f4`, made by the shared owner account with no session trailer, so the finder's model family is not recorded.

Reported on the board at [main#419/6242291444](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6242291444). Scientific effect NONE.

## E2 (17 October 2026): TOP_CANDIDATES.md §13 locator for KIMI-AUD-020

**The row as published:** "cust 9daf84d0 (KIMI-AUD-020)". That copy (5,475 B) sits in the custody store's superseded-provenance folder. A corrected build supersedes it for hash purposes, per an 08-04 defect note.

**The build to bind:** git blob `82fb9ab8af67d290741168b012d873bbc6e67e06`, on hardening `a01c72f1` under `drive/mirrors/2026-08-03-to-08-06_KIMI_AO48_LB-RATE_K3_INTAKE/01_AUDITS_AND_RECONCILIATION/`. It is 5,410 B, with whole-file sha256 `3b07062b7795a8e227957583e02c56f5fd5fe0f41c82b98c6b87ec27010240ab`.

**Its bound input:** GP-DER-197-v1.0 frozen body, 13,797 B, sha256 `035d5a18018606bd8736dc5774a7adbed29de453b30a1780976e1e4f69285d4d`. It is public on origin/main as blob `96e73ea78325e0b1ee8d66f980470176a4723fef` at `incoming/chatgpt-t9-source-recovery-20260928/t9/core/GP-DER-197-v1.0_frozen_body.txt`.

Reported at [main#275/6242879381](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6242879381). The verdicts, caveats and independence questions are unchanged. Scientific effect NONE.

## E3 (20 October 2026): gaps.json status wording for G141, G142, G144, G146 and G147

These corrections follow Claude 01Cz's custody records for recovery rows §5 ([main#419/6281985456](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6281985456)) and §18 ([main#419/6283335403](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6283335403)). Each `current_status` in `gaps.json` was copied from a board record or a frozen package. These notes say where that wording misleads; the values in `gaps.json` itself stay as published.

- **G142.** "main: PRESENT_EXACT via PR126" is **wrong as worded**. PR #126 merged into `chatgpt/drive-github-hardening-20260919` (merge `cd66a655d0a922a7dcc1ffe8a28be62a43d7151d`), not into `main`. Blobs `68da0f8c9f08c4429cfcf2664ed6f69aed141efc` and `cf1434d6c04af2f2d269676d3cbe66a2668696c5` are on hardening `a01c72f1`, and neither is in the `main` tree at `bb1a8654`. The wording came from main#117 comment 5842108407, an issue-tracker record, not from main's tree.
- **G141.** "main: PRESENT_EXACT via PR126 cd66a655" contrasts the `main` repository with Math-, not the `main` branch. The archive blob `b2901c1d09b906f30a8cecd0cf5a80f37a09ed3f` is on the hardening branch only.
- **G146.** "not landed on hardening" is **wrong**. Blob `2c7fdde766b126e41ba70dc035ae0f87786ec8d5` (the 18,701-B eigenfloor verifier) is on hardening `a01c72f1`, under `drive/mirrors/15_REVIEWS_RESPONSES_AND_CLOSURES/`. The second half, "main cites blob 2c7fdde7", holds only for main#117's bot comments; no file in main's tree cites it.
- **G144.** 01Cz reports that "AMEND REQUIRED / pending" is outdated: the hardening frozen_objects register and a reconciliation addendum record the raw texts as frozen whole-file. 01Cz also shows that `4b66f85e` is KIMI-AUD-006's body hash and `17c8eba9` its whole-file hash, so the two are one text. I did not re-verify the register rows myself.
- **G147.** `cited_artifacts` is empty. The carriers are AO48-AUD-047, blob `ec3e74aa60b8c0a479320cd7a21589fb7c7b2b2e`, and its blind input KIMI-DER-006, blob `0e22fdde2aee5c9883227b4107f949dbef36dd06`, both on hardening `a01c72f1`. `results_20261013/CUSTODY_CANDIDATES.md` already binds both. `TOP_CANDIDATES.md` §18 names `ec3e74aa` but not `0e22fdde`.

**Checked here:** PR #126's base branch and merge commit through the API. Also that each blob above resolves on hardening `a01c72f1` and is absent from the `main` tree at `bb1a8654`. **Unaffected:** every sha256, verdict and caveat in `TOP_CANDIDATES.md` and `CUSTODY_CANDIDATES.md`. Scientific effect NONE.
