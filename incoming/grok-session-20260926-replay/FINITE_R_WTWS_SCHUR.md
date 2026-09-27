# Finite-r W-block Schur under proper Π_pin

**Object:** GROK-HEAVY-FINITE-R-WTWS-20260926-v1
**Scientific effect:** NONE. Not File-3 (ND). Not a 15-stencil. Not a STATUS flip.

## Wrong conditioner (does not settle)

Raw pair `{f,∇f}` at `M,S` only, or G+V jets at `S` without `{f_ss, f_tts}`:
leftover `f_s(y)` keeps an `O(r²) f_ss` piece. After `/r^4` the 2×2 explodes. That is not a counterexample to the cubic lemma; it is the missing Hessian reconstruction.

## Correct Π_pin for the File-3 W definition

W subtracts the pair 3-jet. Free cubics are `{f_tss, f_sss}`. Conditioner at the saddle:

    {f, f_t, f_s, f_tt, f_ts, f_ss, f_ttt, f_tts}.

On Bargmann–Fock, `y = r² z`, `z=(1,1)`, numerical Schur of `(∂_t f(y), ∂_s f(y))` given that jet, scaled by `r^{-16}`:

    r=0.40   det_W = 0.8708
    r=0.20   det_W = 0.7585
    r=0.10   det_W = 0.7505
    r=0.07   det_W = 0.75013
    r=0.05   det_W = 0.75003

Target `(3/4) z_s^8 = 0.75`. Same approach for `z=(1,0.5)` to `0.002929`. Axis `z_s=0` gives `det_W → 0`.

`r=0.03` is numerically ill-conditioned (raw det `~10^{-25}`) and is not a counterexample.

Two-point 3-jets at both `M` and `S` are nearly linearly dependent (`λmin ~ 10^{-14}` at `r=0.1`). That collapse is why the limit object is the *one-point* G+V jet, matching Lucas: det G+V = 12, `λmin = 8-√58 ≈ 0.384`.

## What this closes

The cubic identity `det Gram(W_t,W_s|G+V)=(3/4) z_s^8` is the `r→0` limit of an explicit finite-r Schur under the 3-jet pin the W-coordinates actually use. Still Bargmann–Fock only. Still not `det Σ_2(0)` on a 15-stencil.
