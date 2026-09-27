# Closed scoped identity: BF (W_t, W_s) given pair G+V

Scientific effect: NONE on STATUS. This is not File-3 Condition (ND), not File-1 Theorem A, not GitHub D1.

## Object (exact)

Field: centered Bargmann–Fock on R², C(u)=exp(−|u|²/2).
Pair G+V limit functionals: {f, f_t, f_s, f_tt, f_ts, f_ttt} at the collision point.
Free cubics: f_tss, f_sss.
Inner coordinates as in File 3 (6.1) at cubic order:

    W_t = (z_s²/2) f_tss
    W_s = z_t z_s f_tss + (z_s²/2) f_sss

## Theorem (scoped)

    Cov(f_tss, f_sss | G+V) = diag(2, 6)

    det Gram(W_t, W_s | G+V) = (3/4) z_s^8

Hence the 2×2 is positive definite on {z_s ≠ 0} and vanishes exactly on the fold axis.

Independent numerical match at five (z_t,z_s) points, including signs. Adding f_tts or f_ss to the conditioner does not change the Schur.

Euler: 3 W_0 = z_t W_t + z_s W_s identically. W_0 is not an independent coordinate at this order.

## Axis (separate scaling)

At File 3’s written /r^6 and /r^4, the 4-jet remainder is O(r²) → 0. The same scaling does not rescue the axis.

If the axis frame is renormalized (value /r^8, gradients /r^6), then at z_s=0

    W_t^ax = (z_t³/6) f_tttt
    W_s^ax = (z_t³/6) f_ttts

    Gram(W_t^ax, W_s^ax | G+V) = diag((2/3) z_t^6, (1/6) z_t^6)
    det = (1/9) z_t^{12}

Positive for z_t ≠ 0. The origin z=0 is the excluded collision y=x_S.

This axis identity is a candidate stratified frame. It is not File 3 Appendix B (that appendix was never shipped).

## What is not closed

- File 3 (ND) as det Σ₂(0)>0 on the unshipped 15-stencil.
- Uniformity in the Gaussian field class beyond this kernel.
- Finite-r (not just r→0) invertibility of the raw two-scale 15-frame.
- GitHub D1: A3 floor is still type-convergence (5.4)–(5.5), not (3.5).
- D5 pin: PR82 cubic det H_M≡0 is an algebraic identity, not a Q-measure rate.

## Ledger

D2/D3/D4/D6 scoped ACCEPT unchanged. D1 AMEND. D5 AMEND. File-1 Theorem B still PROVEN-MODULO (ND).
