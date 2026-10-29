# Archived axis-envelope precision boundary

This is a source inspection and exact-rational calculation, not a run of the
archived program or a certificate of the opposite inequality. The source is
[Math- commit `e877242cf68fb24c605066f16dd07f513ee75a1d`,
`certify_axis_break.py`](https://github.com/d6g8k5htny-coder/Math-/blob/e877242cf68fb24c605066f16dd07f513ee75a1d/imports/upper2d_stage_e_20260926/raw/STAGE_E/certify_axis_break.py)
(Git blob `782267adff257a4e7f19816c75c1a100402f3eb8`). It calls its point
violation “fully certified.” Its `MBWD_C` and `Z_HI` decimal strings
are converted to `mpf` before importing `c1_grid`; `TH` and `DL`
are also converted before that import. The pinned
[`c1_grid.py`](https://github.com/d6g8k5htny-coder/Math-/blob/e877242cf68fb24c605066f16dd07f513ee75a1d/imports/upper2d_stage_e_20260926/raw/C1_alpha_intensity/c1_grid.py)
(Git blob `ce87be04704c3cd30bb0e6d1e17f6716601ece68`) sets
`mp.dps = 80`. The script constructs `xs, ys` before the subsequent
120- and 160-decimal-digit station calculations.

Under a fresh default 53-bit, nearest-rounded mpmath context, the exact
rounding results and comparisons are:

- The stored `MBWD_C` is \(4876433689373267/295147905179352825856\),
  strictly less than the intended \(16522/10^9\), because
  \(4876433689373267\cdot10^9-
  16522\cdot295147905179352825856=-388792832<0\).
- The stored `DL` is \(7205759403792794/144115188075855872\),
  strictly greater than \(1/20\), because
  \(20\cdot7205759403792794-144115188075855872=8>0\).

These are integer comparisons; they do not use floating-point arithmetic
to certify a bound. The actual archived mpmath process was not rerun here,
and a nondefault pre-initialized context would need separate examination.
The two later precision runs reuse the precomputed point and cannot by
themselves bound its input-rounding error or the decimal-threshold error.
Thus their agreement does not establish the advertised exact decimal point
violation or its stated interval certification. This observation does not
prove the original inequality, disprove it, or replace an outward-rounded
interval calculation with exact input construction.

The prior locally prepared Q-5026 packet was not available at this source
cut; this note does not reconstruct its full fields or claim recovery of
its unpublished bytes. Preserve the historical script and its failures.
