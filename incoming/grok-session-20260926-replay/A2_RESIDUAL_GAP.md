# A2 residual — single missing estimate

**Object:** GROK-HEAVY-A2-RESIDUAL-20260926-v1
**Scientific effect:** NONE. Does not accept Theorem A. Does not flip STATUS.
**Sources:** `Math-/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` blob `dfed3b8d318a3ab1950957f393307733a4bef3f2` §3–§4, §7.

## What A2 is

Parent §3–§4 under the regression law Q:

- (3.5) `density_Q(A) <= C_0 exp(-c_0 ||A||_F^2)`
- (4.2) `f = mu_r + B_r.(A_r - E A_r) + g_r` with `g_r` independent of `A_r` under Q
- (4.3) `M3, M4 <= K_0(J_r + ||A_r||_F)`, `J_r = 1 + ||g_r||_{C^4}`, uniform `L^p(Q)` moments of `J_r`

Section 7 integrates the typed weight against Q, not Q^W. Transfer of (4.3) to the reweighted law is not required for the depth-failure numerator.

## Certified this session (CAS)

Axial block on ordered `(f(a), f_x(a), f(c), f_x(c))`:

    det T_ax = 12 r^{-4}.

Each transverse pair:

    det T_tr = r^{-1}.

Hence `|det T_r| = 12 r^{-(d+3)}`, parent (3.2).

Depth-failure integral (7.3) is an identity:

    int_0^{D r U^2} lambda (lambda + E r U) d lambda
      = r^3 [ (D^3/3) U^6 + (E D^2/2) U^5 ].

Convention match with File 4 pinning: parent target `U3 = 12k` equals File-4 `c_{30}` when `kappa = 6k` and `ell = k r^3`.

## Single missing estimate

Both (3.5) and (4.3) need a uniform spectral gap for the joint Gramian of the contact frame and the transverse Hessian at M:

    inf_{0 < r <= r_*} inf_{b in B, k in K, frames R}
        lambda_min( Cov_Q( U_r ⊕ vec A_r ) ) >= c_* > 0.

Parent argues contact-limit positive-definiteness (no duplicated functionals at r=0), compactness of frames, and continuity in r. That is an existence sketch. No explicit minorant is displayed.

Operational form consumed by (7.4):

    sup_{r,b,k,R} E_Q[ J_r^{2m+6} (1 + ||A_r||_F)^{2m+6} ] < infinity.

This follows from (3.5)+(4.3) only after the Gramian gap exists.

## What this is not

- Not A3 floor `Z_r = Theta(r^2)` / type-convergence (5.4)–(5.5).
- Not cap implication `G_r =>` elder pairing.
- Not File-3 Condition (ND).
- Not Theorem A.
