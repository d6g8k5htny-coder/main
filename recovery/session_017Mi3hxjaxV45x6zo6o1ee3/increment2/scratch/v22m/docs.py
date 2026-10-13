def rep(p, old, new):
    s=open(p,encoding='utf-8').read()
    assert s.count(old)==1, (p, old[:100])
    open(p,'w',encoding='utf-8').write(s.replace(old,new))
R='README.md'; A='ALIGNMENT.md'; T='test_gate.py'
# README narrative
old="  - Because (H1) is strict, some `ε > 0` keeps `8r(m + Kε)² < λ`.\n"
rep(R,old,old+"\nSince v2.2, `ridge_capHyp_E` assumes only that `f` is `C⁴` on an open `U ⊇ D`, together with SP's quantitative bounds on `D` written with the iterated derivatives `D^k f = iteratedFDeriv ℝ k f`:\n"
"- every derivative datum of `ridge_capHyp_D` is a Fréchet derivative of `f`, and `T3`, `T4` and `Φ4` are slices of `D³f` and `D⁴f`;\n"
"- the collar width `ε₀` comes from `collar_subset_open`: `D` is compact, so a closed thickening of it lies in `U`;\n"
"- `K` bounds `‖D⁴f‖` on the compact collar, and `block_of_norm` turns that norm bound into the block bound.\n")
# README rows
old_row="| `h1Toy_D` | `h1ToyF` meets every hypothesis of `ridge_capHyp_D` (`T4 = 0`, `K = 0`, `ε₀ = 1/2`). The theorem gives a ridge on `(−2 − ε, 2 + ε)` for some `ε > 0`, and the cap |"
rep(R,old_row,old_row+"\n"
"| `block_of_norm` | the block bound from the norm (v2.2). A trilinear form `B` on `ℝ × E` with `‖B‖ ≤ K` satisfies the block bound with constant `K`. Replacing one slot at a time by `sᵢ(1, 0)` leaves three terms, each with a transverse argument `(0, yᵢ)` |\n"
"| `collar_subset_open` | a closed collar `[−2r − ε₀, 2r + ε₀] × B̄(0, 2r)` with `ε₀ > 0` inside any open `U ⊇ D` (v2.2) |\n"
"| `ridge_capHyp_E` | the cap from `f ∈ C⁴` on an open `U ⊇ D` and SP's quantitative bounds on `D` (v2.2). It concludes the ridge on `(−2r − ε, 2r + ε)` for some `ε > 0`, `h(∓r/2) = 0`, the cap and (C2⁺). No derivative datum and no bound off `D` is assumed |\n"
"| `h1Toy_E` | `h1ToyF` meets every hypothesis of `ridge_capHyp_E` with `U = ℝ²`: `D³f[a, b, c] = 2a₁b₁c₁` and `D⁴f = 0`, so `m = 2` and `n = 0`. The theorem gives a ridge on `(−2 − ε, 2 + ε)` for some `ε > 0`, and the cap |")
# README not proved
rep(R,"  (Morse indices) are not formalized and are not used by the cap theorem. Since v2.1 `ridge_capHyp_D` takes, as\n"
"  hypotheses:\n"
"  - the quantitative bounds on SP's `D` only: the third-derivative block bound `m`, `|f_xxx| ≤ m` on the pin segment,\n"
"    the block bound `n` on the derivative of `f_xxx`, (H1) and (H2);\n"
"  - qualitative data on some collar `[−2r − ε₀, 2r + ε₀] × B̄(0, 2r)`: the derivatives `Df`, `DG`, `H`, `D2f`, the\n"
"    third derivatives and their `x`-derivatives, with some bound `K` on the latter.\n\n"
"  The continuity step from `D` to the collar is proved (`block_bound_collar`). What stays informal is that SP's\n"
"  `f ∈ C⁴` near `D` supplies these data and some `K`, by compactness of a closed collar;\n",
"  (Morse indices) are not formalized and are not used by the cap theorem. Since v2.2 `ridge_capHyp_E` takes, as\n"
"  hypotheses, only SP's §2 setting:\n"
"  - `f ∈ C⁴` on an open neighbourhood `U` of `D` (SP: `f ∈ C⁴` near `D`);\n"
"  - the quantitative bounds on SP's `D`: the third-derivative block bound `m`, `|f_xxx| ≤ m` on the pin segment,\n"
"    the two fourth-order blocks `n`, (H1) and (H2);\n"
"  - the critical pins and the gap `r³/6`.\n\n"
"  The derivative data, a collar inside `U` and the bound `K` on it are derived (`collar_subset_open`,\n"
"  `block_of_norm`, `block_bound_collar`). That the local field on `G_r` satisfies SP's §2 setting is SP's input and\n"
"  is not formalized;\n")
rep(R,'- prints the axioms of all 121 declarations,','- prints the axioms of all 125 declarations,')
# ALIGNMENT SP row
rep(A,"with the continuity step to the collar proved (`ridge_capHyp_D`). Inputs that remain: the source data themselves (I3). |",
"with the continuity step to the collar proved (`ridge_capHyp_D`). Since v2.2, the collar data, the collar and its bound `K` are derived from `f ∈ C⁴` on an open neighbourhood of `D` (`ridge_capHyp_E`). Inputs that remain: the source data themselves, that is SP's §2 setting on `D` (I3). |")
# ALIGNMENT §2 rows after ridge_capHyp_D row
anchor="| `sp_curved_side_constants` | `r > 0`, `m ≥ 2`, `δ > r m (8m - 5)`, `0 ≤ ω ≤ 2 m r²`"
rep(A,anchor,
"| `block_of_norm` | `B` a trilinear form on `ℝ × E` (a `ContinuousMultilinearMap` on `Fin 3`) with `‖B‖ ≤ K` ⇒ `|B[a, b, c] − s₁s₂s₃ B[e, e, e]| ≤ K((|s₁| + ‖y₁‖)(|s₂| + ‖y₂‖)(|s₃| + ‖y₃‖) − |s₁||s₂||s₃|)` for `a = (s₁, y₁)`, `b = (s₂, y₂)`, `c = (s₃, y₃)`, `e = (1, 0)` | I3's \"some `K`\": a norm bound gives the block form | Three-term telescoping (`a → s₁e`, then `b → s₂e`, then `c → s₃e`). Each term has one transverse argument `(0, yᵢ)` and is bounded with `‖(s, y)‖ ≤ |s| + ‖y‖`. The three bounds sum to the right side exactly. |\n"
"| `collar_subset_open` | `E` proper, `0 < r`, `U` open, `D ⊆ U` ⇒ `∃ ε₀ > 0` with `[−2r − ε₀, 2r + ε₀] × B̄(0, 2r) ⊆ U` | SP's \"near `D`\" | `D` is compact, so some closed thickening of `D` lies in `U`. In the max metric each collar point is within `ε₀` of a point of `D`. |\n"
"| `ridge_capHyp_E` | `U` open with `D ⊆ U`, and `f` `C⁴` on `U`. On `D`: the block bound `m` on `D³f`; `|D⁴f[e, e, e, e]| ≤ n` and `|D⁴f[(0, v), e, e, e]| ≤ n‖v‖`. On the pin segment, `|D³f(x, 0)[e, e, e]| ≤ m`. Also `D²f(M)[(0, v), (0, v)] ≤ −λ‖v‖²`, (H1) `8rm² < λ`, (H2) `rn ≤ 1/20`, `Df(M) = Df(S) = 0` and the gap `r³/6` ⇒ `∃ ε > 0` with `ridge_capHyp_C4`'s conclusion on `(−2r − ε, 2r + ε)` | SP §2 (`f ∈ C⁴` near `D`; (H1), (H2), `M₃`, `M₄` on `D`); L1–L20 | `D^k f = iteratedFDeriv ℝ k f` and `e = (1, 0)`. `ridge_capHyp_D` is applied with `Df = fderiv f`, `D2f = fderiv (fderiv f)`, `T3 p a b c = D³f(p)[c, a, b]`, `T4 p a b c = D⁴f(p)[e, c, a, b]` and `Φ4 p w = D⁴f(p)[w, e, e, e]`; `DG`, `H`, `Hx` and `Hr` are built from `D2f` and its derivative. `ε₀` comes from `collar_subset_open`, and `K = max C 0`, where `‖D⁴f‖ ≤ C` on the compact collar. The block bound is invariant under a common permutation of the three slots, so the slot order in `T3` and `T4` does not matter. Nothing is assumed off `D` except `C⁴` on `U`. |\n"
+anchor)
# ALIGNMENT §3 sentence
old3="The theorem chooses `ε` itself and yields a ridge on `(−2 − ε, 2 + ε)` and the cap."
rep(A,old3,old3+" `h1Toy_E` (v2.2) meets every hypothesis of `ridge_capHyp_E` with `U = ℝ²`. `h1ToyF` is a polynomial, `D³f[a, b, c] = 2a₁b₁c₁`, `D⁴f = 0` and `D²f[(0, v), (0, v)] = −40v²`, so `m = 2`, `n = 0` and `λ = 40`; Lean computes these iterated derivatives from the Fréchet derivatives. The theorem yields a ridge on `(−2 − ε, 2 + ε)` for some `ε > 0`, and the cap.")
# ALIGNMENT I3 main text
rep(A,"What stays informal is that `f ∈ C⁴` near `D` supplies the derivative data and some `K`, by compactness of a closed collar.",
"Since v2.2 the remaining qualitative step is proved too (`ridge_capHyp_E`). `f ∈ C⁴` on an open `U ⊇ D` supplies every derivative datum, a collar inside `U` (`collar_subset_open`) and the bound `K` on `‖D⁴f‖` over the compact collar (`block_of_norm`).")
rep(A,"On a collar of width `ε₀`, only qualitative data are assumed: `Df`, `D2f`, `T3` and its `x`-derivative with some bound `K`, `DG` (strict), `H`, `Hx` and `Hr`. |",
"On a collar of width `ε₀`, `ridge_capHyp_D` assumes only qualitative data: `Df`, `D2f`, `T3` and its `x`-derivative with some bound `K`, `DG` (strict), `H`, `Hx` and `Hr`. Since v2.2 (`ridge_capHyp_E`) these are derived, and the only assumption off `D` is `f ∈ C⁴` on an open neighbourhood of `D`. |")
rep(A,"| SP §2 (H1), L6, L7, L9–L12; CAP §§2–5 | These are checked with exact constants by SP's script. Since v2.1,",
"| SP §2 (H1), L6, L7, L9–L12; CAP §§2–5 | These are checked with exact constants by SP's script. Since v2.2, Lean derives the collar data, the collar and `K` from `f ∈ C⁴` near `D` (`ridge_capHyp_E`). Since v2.1,")
# ALIGNMENT §5
rep(A,"its 113 v2.0 declarations are unchanged. A same-provider read,",
"its 113 v2.0 declarations are unchanged. v2.2 adds three §2 rows (`block_of_norm` to `ridge_capHyp_E`), the `h1Toy_E` sentence of §3 and a further revision of the SP row and I3. The statements of its 121 v2.1 declarations are unchanged. The only edits to v2.1 lines are in the header comment and in the docstrings of `ridge_differentiable` and `fxxx_lower`, which take up the wording residuals of the K1, K4 and K5 reads. A same-provider read,")
rep(T,'self.assertEqual(len(m["targets"]), 121)','self.assertEqual(len(m["targets"]), 125)')
print("ok")
