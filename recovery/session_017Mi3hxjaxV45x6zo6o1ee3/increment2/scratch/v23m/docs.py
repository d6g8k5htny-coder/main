def rep(p, old, new):
    s=open(p,encoding='utf-8').read()
    assert s.count(old)==1, (p, old[:100])
    open(p,'w',encoding='utf-8').write(s.replace(old,new))
R='README.md'; A='ALIGNMENT.md'; T='test_gate.py'
old="- `K` bounds `‖D⁴f‖` on the compact collar, and `block_of_norm` turns that norm bound into the block bound.\n"
rep(R,old,old+"\nSince v2.3, `quarticToy_E` also instantiates `ridge_capHyp_E` away from the degenerate face. `quarticToyF = x³/3 − x/4 − 20y² + (x² − 1/4)²/480` keeps both pins critical and the gap at `1/6`, while `f_xxx = 2 + x/20` varies and `D⁴f = 1/20 ≠ 0`. So (H2) holds with equality (`rn = 1/20`), `fxxx_lower` runs with `n > 0`, and the collar bound `K` is positive.\n")
old_row="| `h1Toy_E` | `h1ToyF` meets every hypothesis of `ridge_capHyp_E` with `U = ℝ²`: `D³f[a, b, c] = 2a₁b₁c₁` and `D⁴f = 0`, so `m = 2` and `n = 0`. The theorem gives a ridge on `(−2 − ε, 2 + ε)` for some `ε > 0`, and the cap |"
rep(R,old_row,old_row+"\n| `quarticToyF`, `quarticToy_E` | a pin-preserving quartic perturbation of `h1ToyF` that meets every hypothesis of `ridge_capHyp_E` with `U = ℝ²`, `m = 81/40`, `n = 1/20` and `λ = 40` (v2.3). `D³f[a, b, c] = (2 + x/20)a₁b₁c₁` and `D⁴f[a, b, c, d] = a₁b₁c₁d₁/20`, so (H2) is sharp and (H1) reads `32.805 < 40`. The theorem gives the ridge and the cap |")
rep(R,'- prints the axioms of all 125 declarations,','- prints the axioms of all 127 declarations,')
old3="The theorem yields a ridge on `(−2 − ε, 2 + ε)` for some `ε > 0`, and the cap."
rep(A,old3,old3+" `quarticToy_E` (v2.3) does the same away from the degenerate face. `quarticToyF = h1ToyF + (x² − 1/4)²/480` keeps both pins critical and the gap at `1/6`; Lean computes `D³f[a, b, c] = (2 + x/20)a₁b₁c₁` and `D⁴f[a, b, c, d] = a₁b₁c₁d₁/20`. With `r = 1`, `m = 81/40` (the maximum of `|f_xxx|` on the pin segment), `n = 1/20` and `λ = 40`, (H2) holds with equality and (H1) reads `8m² = 32.805 < 40`. So `fxxx_lower` is used with `n > 0` and the collar bound `K` is positive, which answers the optional non-vacuity remarks of K4 (finding 8) and K6 (a).")
rep(A,"a collar inside `U` (`collar_subset_open`) and the bound `K` on `‖D⁴f‖` over the compact collar (`block_of_norm`).",
"a collar inside `U` (`collar_subset_open`) and the bound `K` on `‖D⁴f‖` over the compact collar (by compactness; `block_of_norm` turns it into the block bound).")
rep(A,"which take up the wording residuals of the K1, K4 and K5 reads. A same-provider read,",
"which take up the wording residuals of the K1, K4 and K5 reads. v2.3 adds the `quarticToy_E` sentence of §3 (two declarations, `quarticToyF` and `quarticToy_E`). Its 125 v2.2 statements are unchanged. Its only edits to v2.2 lines are the two optional wording points deferred from the K6 reads: I3's credit for `K` and the `ridge_differentiable` docstring, which now names `ridge_capHyp_source` as the declaration that performs the composition. It also adds one header line. A same-provider read,")
rep(T,'self.assertEqual(len(m["targets"]), 125)','self.assertEqual(len(m["targets"]), 127)')
print("ok")
