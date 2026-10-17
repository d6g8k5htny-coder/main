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
