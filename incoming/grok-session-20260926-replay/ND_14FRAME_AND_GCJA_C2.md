# 14-frame leftover and GCJA C2 slaving

**Scientific effect:** NONE. Does not accept or kill File-3 Condition (ND).
Does not change STATUS.

## Certified this session (BF, C(u)=exp(-|u|²/2))

Unconditional 3-jet: `Var f_tss=3`, `Var f_sss=15`, cross 0.

Free cubic remainder before Hessian pin:

    R = (1/2) f_tss z_t z_s² + (1/6) f_sss z_s³
    3 W_0 = z_t W_t + z_s W_s     (Euler, residual < 2e-15 on a grid)
    det Gram(W_0,W_t,W_s) = 0 at every z
    det Gram(W_t,W_s) = 45 z_s^8 / 16     (vanishes iff z_s=0)
    Var R = z_s^4 (9 z_t² + 5 z_s²) / 12

After pair `H^-` pins the e_t 3-jet `(f_ttt, f_tts, f_tss)`:

    W_t leftover ≡ 0 at cubic order
    W_s leftover = (z_s² / 2) f_sss
    W_0 leftover = (z_s³ / 6) f_sss
    rank(W-block | pair Hess) ≤ 1
    Var W_s = (15/4) z_s^4 ,  Var W_t = 0 ,  Var W_0 = (5/12) z_s^6   (unconditional f_sss)

Fourth-and-higher jets after File-3 `/r^6` and `/r^4` are O(r²)→0. They do not restore a third O(1) W-coordinate.

## Alignment with GCJA (already in the attached stack)

GCJA Corollary C2: when value and ∂_s share a single pure-transverse leading stratum σ=(0,k*), the leading 2×2 is **rank-1** and the slaving ratio *is* Euler. Quote: “K4’s degeneracy is thus a THEOREM output, not an anomaly.”

GCJA C1 (D(2,2), γ=2, BF): `Var(f_sss|V)=6`, `Var(f_ttss|V)=4`, cross 0. Different from the unconditional 15 because V is the visible staircase, not the raw 3-jet.

File-3 §7.3 asks `det Σ_2(0)>0` on a 15-frame that treats `(W_0,W_t,W_s)` as independent *and* keeps `H^-`. Those two demands fight. GCJA already resolved the fight by allowing the limit form to degenerate and tracking the leading-order *scaling*.

## Honest restatement (not a STATUS row)

Off `{z_s=0}`: drop `W_0` (Euler) and drop either `W_t` or `H^-_{ss}` (same monomial `f_tss`). The surviving transverse observable is `W_s ∝ z_s² f_sss`, variance ≍ z_s^4.

On `{z_s=0}`: cubic W-block is identically 0. That is the visible Appendix-B axis. File-3 already permits a 14-frame there; the axis needs its own leading stratum (GCJA: ∂_t takes σ=(2,2) via H2(a=1)).

(ND) as a *positivity of a 15-determinant* is the wrong sentence. The load-bearing sentence is GCJA’s: the leading form `Σ_∞(z)` is not the zero matrix, with an integrable singularity on a curve if needed (File-3 Remark 7.2).

## Still missing

Appendix A Table 4.1 (exact pair-block limits). Without it the pair 12-block map 10-jet → Ξ_1(0) is not unique, so a numerical 14×14 cannot be certified as *the* File-3 matrix.

This note does not close Lemma I and does not flip D1.
