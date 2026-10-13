def rep(p, old, new):
    s=open(p,encoding='utf-8').read()
    assert s.count(old)==1, (p, old[:80], s.count(old))
    open(p,'w',encoding='utf-8').write(s.replace(old,new))
A='ALIGNMENT.md'; R='README.md'
k="for L7, the strict differentiability of the transverse gradient along the ridge (from `f ∈ C²` near `D`),"
K1=(" the identification of `G` with the slice gradient on a neighbourhood of each ridge point `(x, h x)`, `x ∈ [−2r, 2r]` (`hG`);"
    " the negative definiteness `(DG x ∘ inr) v v ≤ −δ‖v‖²` along the ridge for `x ∈ [−2r, 2r]` (`hneg`), whereas"
    " `transverse_hessian_le` gives `H ≤ −(λ − 5rm)` only on the cap (`x ∈ [−2r, r/2]`) and only for the slice Hessians `H`,"
    " which no declaration identifies with `DG x ∘ inr`")
Q=(" (since v1.8, `ridge_capHyp_source` derives `hG` from the joint derivative on an open neighbourhood, and `hneg` from"
   " `transverse_hessian_le_D` on the collar with `DG ∘ inr = H` along the ridge; `hGs` stays a source-data hypothesis"
   " on the collar);")
rep(A,k,k+K1+Q)
rep(A,"Since v1.8 the extension of L4 to the collar and the composition of the L7 lemmas with the cap are proved; since v1.9 also L6, L9 and L12, so that `hconv` is derived (`ridge_capHyp_C3`), and since v2.0 also SP's `f_xxx ≥ 7/4` (`ridge_capHyp_C4`, from L2's second statement and (H2), with SP's block bound `n` on the derivative of `f_xxx` on `D`), so that no analytic step of L1–L20 is assumed; and L9–L12 (`‖h'‖` and the `F''` bound giving `hconv`; proved since v1.9 from the `C³` data; the input `f_xxx ≥ 7/4` is proved since v2.0) |",
"No analytic step of L1–L20 is assumed: of SP's steps L1–L12, the last input, L4's `f_xxx ≥ 7/4` on `D`, is proved since v2.0 (`ridge_capHyp_C4`, from L2's second statement and (H2)). The source data listed in this row stay hypotheses. Since v2.1 (`ridge_capHyp_D`) the quantitative ones are needed on `D` only: the block bound `m`; `|f_xxx| ≤ m` on the pin segment, which the block bound `hT` excludes; SP's block bound `n` on the derivative of `f_xxx`; (H1) and (H2). On a collar of width `ε₀`, only qualitative data are assumed: `Df`, `D2f`, `T3` and its `x`-derivative with some bound `K`, `DG` (strict), `H`, `Hx` and `Hr`. |")
rep(A,"Since v1.8, Lean composes L1–L5, L7, L8 and L11 with the cap",
"Since v1.8 the extension of L4 to the collar and the composition of the L7 lemmas with the cap are proved. Since v1.8, Lean composes L1–L5, L7, L8 and L11 with the cap")
rep(A,"the whole route is composed from SP's source data there","the route up to L12 (`hconv`) is composed from SP's source data there")
rep(A,"the mean value inequality bounds the right side using `T[(1, η), (1, η), (1, h')]`, which is the same expression SP bounds.",
"the mean value inequality bounds the right side using `T[(1, η), (1, η), (1, h'(s))]` with `η = h'(x₁)` frozen. This is a mixed form, not CAP (11)'s `T[v, v, v]` with `v = (1, h')`; the two agree at `s = x₁`. SP's CAP (12) block estimate bounds both by the same constant, `f_xxx − m((1 + u)³ − 1)` (`trilinear_lower`, `l12_lower`).")
rep(A,"`ridge_capHyp_source` (v1.8) performs this composition, deriving the premises on `U = (−2r − ε, 2r + ε)`.",
      "`ridge_capHyp_source` (v1.8) performs this composition, deriving the premises on `U = (−2r − ε, 2r + ε)` and `hG`, `hneg` on `[−2r, 2r]`; `hGs` stays a source-data hypothesis on the collar (I3).")
rep(A,"`rn ≤ 1/20` ⇒ `ψ ≥ 2 − 9rn/2 ≥ 7/4` on `D` |","`rn ≤ 1/20` ⇒ `ψ ≥ 7/4` on `D` (the proof bounds `ψ ≥ 2 − 9rn/2`, and the statement exports `7/4`) |")
rep(R,"they give `f_xxx ≥ 2 − 9rn/2 ≥ 7/4` on `D` by SP's two-segment estimate (`fxxx_lower`).",
      "they give `f_xxx ≥ 7/4` on `D` by SP's two-segment estimate (`fxxx_lower`; the proof bounds `f_xxx ≥ 2 − 9rn/2`, and the statement exports `7/4`).")
rep(R,"then `f_xxx ≥ 2 − 9rn/2 ≥ 7/4` on `D`. Move first in `x`",
      "then `f_xxx ≥ 7/4` on `D` (the proof bounds `f_xxx ≥ 2 − 9rn/2`, and the statement exports `7/4`). Move first in `x`")
rep(R,"- the source data themselves. Since v2.0 every analytic step of L1–L20 is proved: L1–L12, L13 and L16–L20, with L7's\n  existence, uniqueness and regularity of `h`. Since v2.1 `ridge_capHyp_D` takes, as hypotheses:",
      "- the source data themselves. Since v2.0 every analytic step of L1–L20 that `ridge_capHyp_C4` uses is proved: L1–L13\n  and L16–L20, with L7's existence, uniqueness and regularity of `h`. L14 (the critical-point count in `D`) and L15\n  (Morse indices) are not formalized and are not used by the cap theorem. Since v2.1 `ridge_capHyp_D` takes, as\n  hypotheses:")
rep(R,"L3–L4 on SP's full domain:","L3–L4 on a collar of SP's `D`:")
print('ok')
