# Cubic slaving of the inner W-block (File 3 (6.1), BF)

Scientific effect: NONE. Does not accept or kill Condition (ND). Does not flip STATUS.

## Nested rows, as written in File 3

Layer one (pair, scale r), (4.2):

    V+ = (f(x_s)+f(x_m))/2
    V- = (f(x_s)-f(x_m))/r^3
    G_i^± as in (4.2b)
    H+ = (D²f(x_s)+D²f(x_m))/2
    H- = (D²f(x_s)-D²f(x_m))/r

Layer two (inner, scale r²), (6.1):

    W_0 = [f(y) − Π_pinned f(y)] / r^6
    W_i = [∂_i f(y) − Π_pinned ∂_i f(y)] / r^4    i ∈ {t,s}

Pinned 3-jet (Remark 4.2 + §6.2): t³ and t²s.
Free 3-jet: ts² and s³.

## Leading inner polynomial

Displacement y = x_s + r² z. The free-cubic remainder is

    R(z) = (1/2) f_tss z_t z_s² + (1/6) f_sss z_s³.

Then at this order W_0 = R, W_t = ∂R/∂z_t, W_s = ∂R/∂z_s, hence the Euler identity

    3 W_0 = z_t W_t + z_s W_s

holds identically on the free 3-jet. Any 3×3 Gram of (W_0, W_t, W_s) built from these two coefficients has rank ≤ 2.

Computed (C(u)=exp(−|u|²/2)):

    Cov(f_tss, f_sss) = diag(3, 15)
    Var R = z_s⁴ (5 z_s² + 9 z_t²) / 12
          > 0  ⇔  z_s ≠ 0.

On the fold axis z_s = 0 the whole cubic W-block vanishes. That is the visible Appendix-B candidate.

## Fourth-order cannot rescue the 15th coordinate at this scaling

Order-4 Taylor content at displacement r² is O(r⁸). After /r⁶ it is O(r²) → 0 in the r→0 limit of (6.1). So the “fourth-order terms in W_0(0)” of Lemma 6.1 are the stratified (axis) 14-frame, not a third independent O(1) direction off-axis.

## What this does and does not say about (ND)

File 3 §7.3 asks det Σ₂(0)(z,θ) > 0 on a 15-frame that includes all three W’s. At the file’s own leading scaling those three coordinates are linearly dependent wherever the cubic-free jet is used. So the 15-count overcounts by 1 at cubic order.

This is **not** a certified kill of the route. File 3 already allows a stratified 14-frame on Appendix B, and Remark 7.2 allows integrable vanishing. The honest next statement is: restate (ND) on the 14-frame (W_t, W_s) off {z_s=0}, plus a separate axis frame. That restatement is not in the public vault.

## Public inputs used

- File 3 §§3–7 (attached).
- Bargmann–Fock C(u)=exp(−|u|²/2), exact 1-point jet Gramian through order 4: rank 15, min eigenvalue ≈ 0.202 (Lemma H at one point, this kernel).
- Ladgham–Lachièze-Rey, SPA 166 (2023) 104221: method citation for single-scale blow-up only. Not a two-scale 15-frame.

## Still missing

Appendix A Table 4.1 (14 pair rows) and Appendix B exceptional curves, as written. Without those stencils the full Σ₂(0) is not a matrix we can invert.

No STATUS flip. D2/D3 unchanged. File-1 Theorem B still PROVEN-MODULO (ND).
