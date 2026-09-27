# Bargmann–Fock off-axis 2×2 and the axis obstruction

**Object:** GROK-HEAVY-BF-WTWS-20260926-v1
**Scientific effect:** NONE. This is not File 3 Condition (ND). It is not Lemma I.

Kernel: C(u)=exp(-|u|²/2), derivatives at a single point. Independently replayed this session.

## Pair G+V Schur

Conditioning set: {f, f_t, f_s, f_tt, f_ts, f_ttt}. Adding f_tts or f_ss does not change the next line.

    Cov(f_tss, f_sss | G+V) = diag(2, 6).

## Off-axis cubic map (Remark 4.2 pins t³ and t²s)

    W_t = (1/2) z_s² f_tss,
    W_s = z_t z_s f_tss + (1/2) z_s² f_sss.

    Gram(W_t, W_s | G+V) = [[ z_s^4/2 , z_s³ z_t ], [ z_s³ z_t , z_s²(3 z_s² + 4 z_t²)/2 ]],
    det = (3/4) z_s^8.

Positive definite iff z_s ≠ 0. Both eigenvalues scale as z_s² · (quadratic in z). This is the 2×2 that (6.3) inverts if Hessians stay outside the inverted block.

## Axis, written scaling

On {z_s=0} the cubic map is the zero map. Fourth-order raw gradient remainder at displacement r² z_t e_t is O(r^6). After File 3's /r^4 that is O(r²)→0:

    W_t = (r² z_t³ / 6) f_tttt + o(r²),    W_s = (r² z_t³ / 6) f_ttts + o(r²).

BF: Var(f_tttt)=105, Var(f_ttts)=15, 4-jet 5×5 rank 5, det 7962624. A 4-jet 14-frame at the *same* normalization does not produce an O(1) axis block. Axis needs a different power of r (drop W, or divide by r^6).

## What is not claimed

- Not det Σ_2(0) on a 15-stencil (Appendices A/B never shipped).
- Not uniform positivity up to the axis.
- Not Q^W transfer.
- If pair Hessians are inverted, H⁻_ss slaved W_t and the leftover is 1-D (GCJA C2). That is a different matrix from (6.3).
