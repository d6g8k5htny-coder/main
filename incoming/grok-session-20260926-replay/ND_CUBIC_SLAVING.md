# Cubic-leading slaving in File 3's inner W-block

**Object:** GROK-HEAVY-ND-SLAVING-20260926-v1
**Scientific effect:** NONE. Does not accept or refute Condition (ND). Does not accept Theorem A.

## Nested rows the file actually writes

Layer one, pair block (4.2):

    V+ = (f(x_s)+f(x_m))/2,     V- = (f(x_s)-f(x_m))/r^3,
    G_t^- = (f_t(x_s)-f_t(x_m))/r,   G_t^+ sum-normalized by r^2 (ladder hedge in source),
    H+ = (D²f(x_s)+D²f(x_m))/2,   H- = (D²f(x_s)-D²f(x_m))/r.

Layer two, inner point y = x_s + r² z (6.1):

    W_0 = [f(y) - Π_pinned f(y)] / r^6,
    W_i = [f_i(y) - Π_pinned f_i(y)] / r^4,   i = t,s.

Pinned 3-jet: t³ and t²s. Free 3-jet: ts² and s³. Hessian of y is not inverted (§3.2).

`det T_inner(r) = c r^{-14}` (6.2). Area element r^4. Appendix A/B stencils are **not** in the 18-page file.

## Euler identity on the free cubic

Displacement δ = r² z. After subtracting pinned Taylor through order 2 and the pinned 3-jet,

    R = (1/2) f_{tss} z_t z_s² + (1/6) f_{sss} z_s³.

R is homogeneous of degree 3, so Euler's theorem gives

    3 W_0 = z_t W_t + z_s W_s

identically in (f_{tss}, f_{sss}, z). The three inner coordinates are linearly dependent at the scaling (6.1) uses. Any covariance that treats (W_0, W_t, W_s) as three independent Gaussian coordinates is singular at this order.

Fourth-order raw remainder is O(|δ|^4) = O(r^8). After /r^6 that is O(r²) and vanishes in the r→0 limit of (6.1). It does not restore a third independent direction off the axis. The file's remark that W_0(0) contains fourth-order polynomials is the **axis stratification** (drop W_0, 14-frame), not a 15th independent coordinate on a generic z.

## Where the 2-block itself dies

    W_t = (1/2) f_{tss} z_s²,
    W_s = f_{tss} z_t z_s + (1/2) f_{sss} z_s².

Both vanish for all free cubics if and only if z_s = 0. That axis is the candidate exceptional set of Appendix B.

## Bargmann–Fock check (public kernel C(u)=exp(-|u|²/2))

At a point, Cov(f_{tss}, f_{sss}) = diag(3, 15).

    Var(W_0^{cubic}) = z_s^4 (5 z_s² + 9 z_t²) / 12

which is positive iff z_s ≠ 0. The 3×3 Gram of (W_0, W_t, W_s) has rank 2 off-axis and rank 0 on-axis. Matches Euler.

## What this does not do

It does not compute det Σ_2(0). The pair Hessian block and cross-covariances with Ξ_1 are not included. Appendix A is absent, so the exact 15-row stencil is absent. The finding is only: **at the file's written scaling the inner 3-block is at most 2-dimensional**. Condition (ND) as a 15-independent-coordinate statement is not visible at cubic leading order. A 14-frame restatement (drop W_0 everywhere, not only on Appendix B) is the natural repair to check next. That repair is not executed here.
