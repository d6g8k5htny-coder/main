# File 3 cubic W-block after pair H-

**Object:** GROK-HEAVY-FILE3-CUBIC-LEFTOVER-20260926-v1
**Scientific effect:** NONE. Does not accept or kill Condition (ND). Does not flip STATUS.
**Sources:** File 3 §4.2, §6.1–6.2, §7.3 (attached manuscript). Bargmann–Fock C(u)=exp(-|u|^2/2) Wick moments.

## Exact cubic identities (CAS)

Free remainder after pair pins t^3 and t^2 s:

    R = (1/2) f_tss z_t z_s^2 + (1/6) f_sss z_s^3

    W_0 = R,  W_t = ∂_{z_t} R = (z_s^2/2) f_tss,
    W_s = ∂_{z_s} R = z_s (f_sss z_s + 2 f_tss z_t)/2.

Euler: 3 W_0 = z_t W_t + z_s W_s identically.

At the layer-one limit, H^-_{ss} = f_tss. Therefore two linear relations among File 3's listed 15 coordinates:

    W_t = (z_s^2 / 2) H^-_{ss}
    W_0 = (z_s / 3) W_s + (z_t z_s^2 / 6) H^-_{ss}

Rank drop 2 at cubic order, not 1. Unconditional 3×3 Gram of (W_0,W_t,W_s) has determinant 0 and rank 2.

## Bargmann–Fock 2×2 of (W_t, W_s) before killing H-

Cov(f_tss, f_sss) = diag(3, 15).

    det Gram(W_t, W_s) = 45 z_s^8 / 16.

Positive iff z_s ≠ 0. Rank 2 off the fold axis. This is the visible Appendix-B curve, obtained without Appendix B.

## Leftover after removing f_tss (pair frame)

    leftover W_t = 0
    leftover W_s = (z_s^2 / 2) f_sss
    leftover W_0 = (z_s^3 / 6) f_sss = (z_s / 3) leftover W_s

    Var(leftover W_s) = 15 z_s^4 / 4 > 0 iff z_s ≠ 0.

The only new cubic functional is the pure-transverse f_sss evaluation. An honest inverted inner block given the pair frame is 1-dimensional off {z_s=0}.

## 13-frame candidate (not a theorem)

Listed 15 overcounts by 2 at File 3's own (6.1) scaling. A candidate restatement:

- drop W_0 and W_t from the inverted block (they lie in span{W_s, H^-_{ss}});
- keep leftover W_s;
- on {z_s=0} drop the whole cubic W-block (14-frame already contemplated by File 3 §6.5).

This restatement is **not** in the vault. Appendix A/B were never drafted (File 3 p.18). Fourth-order content after /r^6 is O(r^2) and does not restore a third O(1) coordinate.

## What this does not do

Does not evaluate det Σ_2(0). Does not kill the integrable-vanishing fork (Remark 7.2). Does not accept File-1 Theorem A or B. Does not touch GitHub D1/D5.
