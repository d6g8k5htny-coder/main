# G+V spectrum, File-4 P3 match, periodization

Scientific effect: NONE on STATUS.

## G+V Gram, exact

Order {f, f_t, f_s, f_tt, f_ts, f_ttt}. The matrix splits into four blocks:

    (f, f_tt)   = [[1,-1],[-1,3]]     eigenvalues 2±√2
    (f_t, f_ttt)= [[1,-3],[-3,15]]    eigenvalues 8±√58
    (f_s)       = 1
    (f_ts)      = 1

    det = 2 · 6 · 1 · 1 = 12
    λ_min = 8−√58 ≈ 0.384226894

Char poly: λ^6 − 22λ^5 + 113λ^4 − 220λ^3 + 196λ^2 − 80λ + 12.

This is the one-point conditioner for the cubic 2×2 identity.

## File-4 Theorem P3, convention check only

P3(i): c_30 = 12 ℓ / r^3 + O(r^2) = 2κ + O(r^2) under File-1 ℓ = κ r^3 / 6.
P3(iii): free leading coefficients are c_12 = f_tss and c_03 = f_sss.

That is the same pair whose Schur we computed. P3 is not being accepted as a public theorem here; the constant match is a consistency check (V4.1).

## Periodization

One-point jet identities on the L=24 torus differ from infinite BF by image terms of size exp(−L^2/2) = exp(−288) ≈ 8.4×10^{−126}.
L=2π images are ~ 2.7×10^{−9}.
SIDE24 leading cubic 2×2 and G+V spectrum are the infinite-BF values.

## Not closed

A3 type-convergence. D5 microdisk. File-3 ND. SARD-G A1 (closed-section endpoint hit still AMEND on PR122).
