# D1 A1–A7 interface table — 2026-09-26 round 3

**Scientific effect:** NONE.
**Does not accept Theorem A.**
**Does not edit STATUS.md.**

Parent source (immutable): `Math-/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`
SHA256 recorded in Math- PR64: `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`

Congruence erratum (not yet on Math- default branch): [Math- PR64](https://github.com/d6g8k5htny-coder/Math-/pull/64), still **open draft**, mergeable_state `behind`.
Correct factor: `D_r = diag(r^{-1/2}, I)`, so `det(D_r H_i D_r) = det(H_i)/r`.

A1–A7 below follow the load-bearing interfaces named on [main#63](https://github.com/d6g8k5htny-coder/main/issues/63) and the parent section map.

| ID | Parent locus | What it claims | Wrong-D used? | This-round check | Disposition |
|---|---|---|---|---|---|
| A1 | §2 | Periodic Bargmann–Fock Fourier coefficients `a_n > 0`; finite distinct derivative functionals linearly independent; smooth version with finite `C^q` `L^p` jets | No | Standard for this covariance. No numerical tail enclosed. | Qualitative positivity OK. Not a numerical certificate. |
| A2 | §3 | Contact map `T_r` with `|det T_r| = 12 r^{-(d+3)}`; `Sigma_r` uniformly PD on compact frames; transverse Hessian density bound (3.5) | No | The `12 r^{-(d+3)}` is the same cubic Jacobian already checked: `f_xxx(0)=12k` plus pin scaling `r^{-(d+3)}`. `T_r` matrix itself is not expanded in the extracted text. `lambda_min(Sigma_r)` not evaluated. | Exact power counting consistent with the fold ledger. Frame matrix not independently reconstructed. Do not ACCEPT. |
| A3 | §5 | Congruence to `diag(±6k, A_0)`; `Z_r/r^2 → (6k)^2 E[det(A_0)^2 1{A_0<0}]`; floor (5.5) | **Yes — displayed `diag(sqrt(r),I)`** | Sympy `m=1,2,3`: `D_r=diag(r^{-1/2},I)` gives `K_{00}=alpha`, off-diagonals `sqrt(r) beta`, `det(DHD)=det(H)/r`. Parent `D_old` instead gives `det = r det(H)`. Uniform integrability of (5.3) and `z_0>0` were not re-proved. | **Erratum linear algebra ACCEPT.** Analytic A3 (UI, type indicator, `z_*`) still **AMEND**. A3 reviewers must read parent + PR64 together. |
| A4 | §4 | Uniform `L^p` bounds on `C^q` norms under `Q`; residual `g_r` independent of `A_r`; `M_3,M_4 ≤ K_0 (J_r + \|A_r\|_F)` | No | Gaussian regression + Fourier summability is the usual argument. No rate. | Plausible scaffolding. Not closed. |
| A5 | §6 (6.1) | Canceled-pivot: `|det H_M| ≤ r (h/2) det B` on typed maxima, no `1/alpha` | No (uses physical `H` and (5.3)) | `m=1`: `det H = r\|alpha\| B - r^2 beta^2`, so the mixed square *subtracts* and the bound holds. `m≥2`: needs the written PD-monotonicity argument; not re-derived here. | `m=1` OK. Higher `m` still source-bound, not independently closed. |
| A6 | §6 (6.2) | Two soft factors kept: `W_r ≤ (r^2 h^2/4) λ_1[λ_1+(3/2)rh] ∏_{j≥2} λ_j(λ_j+rh)` | Mentions wrong D in passing; estimates use physical `H` | Power counting: two `r` factors from the two `det H ~ r` give `r^2` in `W_r`, which is what §7 integrates. Dropping the second soft factor changes the radius power. | Bookkeeping consistent with erratum-normalized `det H / r`. Not an ACCEPT of the typed-weight law. |
| A7 | §7 (7.8) | `Q^W(G_r^c) ≤ C r^3`, hence Theorem A | Uses A3 floor (5.5) | Numerator estimates (7.3)–(7.7) produce `E[W_r 1_{failure}] ≤ C r^5` (plus higher). Divide by `Z_r ≥ z_* r^2` to get `O(r^3)`. If A3 floor fails, A7 fails. Corank intersections are included with nonnegative powers; that part is carefully written. | **Not closed.** Blocked on A3 analytic floor plus independent check of the `r^5` integrals. |

## What this round actually closes

1. The congruence *factor* is `D_r = diag(r^{-1/2}, I)`, not `diag(sqrt(r), I)`. Confirmed by sympy for transverse dimensions 1, 2, 3.
2. Math- PR64 is the correct consumer note. It is still an **open draft** and is **behind** `Math-` main. It is not in the default tree. Anyone reviewing D1 from `main` alone will miss it.
3. Section 6 canceled-pivot bound holds for `m=1` by direct expansion.

## What this round does not close

- Theorem A.
- A3 uniform integrability and `z_* > 0`.
- A2 reconstruction of `T_r`.
- A5 for `m≥2`.
- A7 `O(r^3)` selection.
- D2 lifetime remainder (still imports this parent).
- SIDE24 persistence reading.

## Operational next step

Merge-or-rebase Math- PR64 onto `Math-` default so the erratum is in-tree, then run A3 as a dedicated nonauthor note that cites parent bytes + erratum bytes together. Do not treat this table as that note.
