# ADDENDUM 3 to KIMI_LPW_REVIEW_VERDICT.md — two narrow printed-review corrections (2026-09-13)

The sealed verdict (body d399e28378d177f88d08032c81f1573f35c9af2b600b9b775ebfe40f555cbbad) and
Addenda 1–2 remain unchanged in the record. This third addendum corrects two printed-review items
identified by the author-side's Return-04 package (zip sha256 60784ea9…, 47/47 manifest files
verified). Neither touches the LPW endorsement (cf58f72e…e1a5) or any computed bound.

## 1. The endpoint comparison was reversed (owned)

Addendum 2's preliminary quantitative-verification paragraph printed "31/250 > 1/8". The correct
relation is 31/250 = 0.124 < 0.125 = 1/8. The proof it garbled is unaffected and is restated
correctly here: the planar endpoint satisfies λ_min(Γ_{0,∞}) > 1/8 (all ten exact Sylvester minors
of Γ_{0,∞} − I/8 positive, independently verified by both sides), and the infinite-image periodic
perturbation ‖Γ_{0,24} − Γ_{0,∞}‖_op < 10^{−102} then gives λ_min(Γ_{0,24}) > 1/8 − 10^{−102} >
31/250 = 0.124. The downstream arithmetic 31/250 − 32/1600 = 13/125 > 1/10 is and was correct.

## 2. The Q2 y-row explanation dropped a positive remainder (owned)

My lead analytic verdict's explanation for the average-y-gradient row wrote the error as
"(r²/8)·11 + 11r³/48 ≈ 11r²/8" — a displayed sum whose positive cubic term then vanishes from the
stated upper bound. The bound 11r²/8 is correct, but that explanation was sloppy. The candidate's
original argument (valid, and now the governing statement) is the centered integral form: with
h = r/2 and g(t) = f_y(t, 0),

    (g(h) + g(−h))/2 − g(0) = (1/2) ∫_0^h (h − s)[g″(s) + g″(−s)] ds,

so, with the established uniform Gaussian L² bound ‖g″(s)‖_2 ≤ 11,

    ‖(g(h) + g(−h))/2 − g(0)‖_2 ≤ 11 ∫_0^h (h − s) ds = 11h²/2 = 11r²/8,

with no extra cubic term anywhere. The row bound and the summed L² estimate (‖V_r − V_0‖ ≤ 2r) are
intact; only my one-line justification of that row is corrected.

SHA-256 of this addendum body (text after this line is excluded):
823a7d895fc15f206d8ddbc917f87bd16d6c5d3507b4d147c8edbf7872130d99
