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
rep(A,"Lean does not prove this step. Since v1.8 the extension of L4 to the collar and the composition of the L7 lemmas with the cap are proved; and L9–L12",
      "Lean does not prove this step; and L9–L12")
rep(A,"These are checked with exact constants by SP's script, not in Lean. Since v1.8, Lean composes",
      "These are checked with exact constants by SP's script, not in Lean. Since v1.8 the extension of L4 to the collar and the composition of the L7 lemmas with the cap are proved. Since v1.8, Lean composes")
rep(A,"`ridge_capHyp_source` (v1.8) performs this composition, deriving the premises on `U = (−2r − ε, 2r + ε)`.",
      "`ridge_capHyp_source` (v1.8) performs this composition, deriving the premises on `U = (−2r − ε, 2r + ε)` and `hG`, `hneg` on `[−2r, 2r]`; `hGs` stays a source-data hypothesis on the collar (I3).")
rep(A,"the whole route is composed from SP's source data there","the route up to L12 (`hconv`) is composed from SP's source data there")
rep(R,"L3–L4 on SP's full domain:","L3–L4 on a collar of SP's `D`:")
print('ok')
