# LPW_CONSTANT v3 — RECEIPTS ERRATA (2026-09-15)

Errata to the frozen v3 record (certifier 4a015174…, RECEIPT.json 60549daf…,
report 76016d7b…; all frozen carriers untouched — this dated file is the
corrected record). Independent-review verdicts on the three changed interfaces
stand confirmed (PASS WITHIN SCOPE / PASS WITHIN SCOPE / AMEND REQUIRED
upstream); the certified pair arithmetic is unaffected by every item below.

## Erratum E1 — carrier hash table (3 wrong values in the record files)

The following values were recorded incorrectly in RECEIPT.json (`carriers`
table, lines 46/47/52) and, for the first, in LPW_CONSTANT_V3_REPORT.md prose
(carriers paragraph). ACTUAL values re-verified from bytes on 2026-09-15:

| carrier | file | WRONG (as recorded) | ACTUAL (verified from bytes) |
|---|---|---|---|
| v1_report | ../LPW_CONSTANT_REPORT.md | 702788dc1c00e96910dc21d2b8d66ddaa3b82a8c3c6ca3f9a4343fe104b2ba29 | 289d9f40e725bd51e2405d9de88b3afd428a338f76a709d6d882f1c395f50b22 |
| v1_transcript | ../out_normal.txt | d0cb92594d267a134ee93e4e676603118ce3041a60bf02b2102dc138f8ee720b | d0cb92595bad7e1b7266e1a9043a54caff380ed075e8f4e358b0ae04146c573f |
| h3_rung | ../../UPPER2D/H3_closure/H3_RUNG_FLOOR.md | 6347275d74d3b2a0047b2be02cb5c9f740b380d63831008ffdc093166d43c85d | 6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa |

Root cause: the receipt/report carrier tables were authored from notes before
the certifier's pins were finalized; the certifier's cnv enforcement rejected
the wrong values at first run and the in-script pins were corrected, but the
receipt/report tables were not resynced.

Impact: NONE load-bearing. The certifier's runtime pins (lpw_constant_v3.py
lines 99, 101, 111) are and always were the ACTUAL values above, enforced
fail-closed by cnv (exit 2 on mismatch) at every run, including all 17
mutation runs and both clean modes. The wrong strings existed only in the
human-readable record.

## Erratum E2 — lever-2 net factor arithmetic

RECEIPT.json (levers.LEVER2_delta_widening.certified_effect) and the report
CHANGES table record "net ×959.96", inconsistent with their own factors.
Corrected (exact Fraction arithmetic, re-verified):

  δ⁴ gain:        (1024·δ)⁴ = (1024·14587/2621440)⁴ = 1054.1540231547…
  weight ratio:   W/65 = (1550318853/26214400)/65    = 0.9098457061…
  net:            1054.1540231547 × 0.9098457061     = 959.117511… → **×959.12**

The certified c = 9.8040863135804911886149013570e-23 and the combined gain
c_v3/c_v2 = (9/10000)/1e-21 × 959.1175 = 8.6320576e20 (≈ ×8.63e20) are exact
and unaffected.

## Hygiene note H1 (record; no action required)

v3 does not re-ck the thin-box ⊂ JBOX containment at the new δ (v1 checked it
at δ = 1/1024). It holds analytically: the thin box
{|q + 10r| ≤ 2δr, |A − 2| ≤ δ, |B| ≤ δ, |D₃| ≤ δ} lies in
JBOX = [−(10+2δ)R, 0] × [2−δ, 2+δ] × [−δ, δ]² for every r ≤ R whenever
10 − 2δ > 0, i.e. δ < 5; v3's δ = 14587/2621440 ≈ 5.56e-3.

## Appendix (for the record; outside v3 receipt scope) — h3_band_ceil.py

Observed in UPPER2D/H3_closure/: h3_band_ceil.py (37909 bytes, 2026-09-15
15:21) with transcripts ceil_normal.txt / ceil_err1.txt, postdating H3's
08:29 MANIFEST (hence unmanifested). Its header identifies it as H3's answer
to the dispatched ceiling task ("LPW_CONSTANT v4 / OBL-D1-PROMOTE normalizer
band: CERTIFIED uniform normalizer CEILING E[G_r] ≤ U for ALL r in (0, 0.05]").
Its transcript certifies per-cell ceilings including the v3-relevant cell
"CC6 (0,0.0025] CERTIFIED: E[G_r] ≤ 3.66282864761194", then CK_FAILs on cell
(0.03, 0.04] (its own U ≤ 4 consumption cap) — i.e., H3 work-in-progress, not
a frozen consumable carrier. v3 does not consume it.

Body hash of this document (SHA256 over the body excluding this hash line):
7ccd618af7cee784619c06ca4a521656e5896baa685eacdd42f91f4ad3fbb681
