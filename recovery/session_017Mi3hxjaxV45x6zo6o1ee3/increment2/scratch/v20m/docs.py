def rep(p, old, new):
    s=open(p,encoding='utf-8').read()
    assert s.count(old)==1, (p, old[:90])
    open(p,'w',encoding='utf-8').write(s.replace(old,new))
R='README.md'; A='ALIGNMENT.md'; L='CapFirstExit.lean'
rep(R,"What remains an input is `f_xxx ≥ 7/4` on the collar (SP's L4, from L2 and (H2)), together with the source data.",
"Since v2.0, `ridge_capHyp_C4` also derives that last analytic input, `f_xxx ≥ 7/4`:\n"
"- L2's second statement: some point of the pin interval has `f_xxx ≥ 2`. This comes from a one-sided Hermite bound, by contradiction (`third_deriv_exists_two`).\n"
"- L4: (H2) `rn ≤ 1/20`, with a bound `n` on the derivative of `f_xxx`, gives `f_xxx ≥ 2 − 3rn ≥ 7/4` on the collar (`fxxx_lower`).\n\n"
"So the hypotheses of `ridge_capHyp_C4` are SP's (H1), (H2), the critical pins, the gap `r³/6` and the derivative data with the bounds `m` and `n`. No step of L1–L20 remains a hypothesis.")
old_row="| `h1Toy_C3` | `h1ToyF` meets every hypothesis of `ridge_capHyp_C3`: `D²f = diag(2x, −40)`, and only `f_xxx = 2` is nonzero among the third derivatives. The theorem gives its ridge and its cap without a supplied `hconv` |"
new_rows=old_row+"\n"+\
"| `hermite_gap_le_concave`, `third_deriv_exists_two` | L1 from the concave half alone (`g''' ≤ m` weakly gives `g(a) − g(c) ≤ m(c − a)³/12`), and L2's second statement (v2.0). If `g'''` is continuous on the pin interval and the gap is `r³/6`, then `g''' ≥ 2` somewhere there. Otherwise its maximum `M' < 2` would give a gap `≤ M'r³/12 < r³/6` |\n"+\
"| `fxxx_lower` | L4's `f_xxx ≥ 7/4` (v2.0): if `‖D f_xxx‖ ≤ n` on the collar, `f_xxx(x₀, 0) ≥ 2` and (H2) `rn ≤ 1/20`, then `f_xxx ≥ 2 − 3rn ≥ 7/4` (SP: `2 − 5rn`) |\n"+\
"| `ridge_capHyp_C4` | the cap from SP's hypotheses (v2.0): `ridge_capHyp_C3` with `f_xxx ≥ 7/4` replaced by a derivative `Φ4` of `f_xxx` with `‖Φ4‖ ≤ n` and (H2) `rn ≤ 1/20` |\n"+\
"| `h1Toy_C4` | `h1ToyF` meets every hypothesis of `ridge_capHyp_C4` (`f_xxx = 2` constant, `n = 0`). The theorem gives its ridge and cap with neither `hconv` nor `f_xxx ≥ 7/4` supplied |"
rep(R,old_row,new_rows)
rep(R,"- SP's L4 bound `f_xxx ≥ 7/4` on `D`, which SP derives from L2's second statement (some point has `f_xxx = 2`) and\n"
"  (H2) `rn ≤ 1/20` with the fourth-derivative bound `n`. It is a hypothesis of `ridge_capHyp_C3`. Since v1.9, L6 and\n"
"  L9–L12 are proved (`ridge_hconv`, `ridge_capHyp_C3`), and so are L1–L5, L7 (existence, uniqueness and regularity of\n"
"  `h`), L8, L11, L13 and L16–L20;\n","")
rep(R,"- the source data themselves. `ridge_capHyp_source` and `ridge_capHyp_C3` take, as hypotheses, the derivatives `Df`,\n"
"  `DG`, `H`, `D2f` and the third derivatives, with the bounds `m`, on the collar `[−2r − ε, 2r + ε] × B̄(0, 2r)`.",
"- the source data themselves. Since v2.0 every analytic step of L1–L20 is proved: L1–L12, L13 and L16–L20, with L7's\n"
"  existence, uniqueness and regularity of `h`. `ridge_capHyp_C4` takes, as hypotheses, the derivatives `Df`, `DG`,\n"
"  `H`, `D2f`, the third derivatives and the derivative of `f_xxx`, with the bounds `m` and `n`, on the collar\n"
"  `[−2r − ε, 2r + ε] × B̄(0, 2r)`.")
rep(R,'- prints the axioms of all 108 declarations,','- prints the axioms of all 113 declarations,')
rep(L,"  `ridge_capHyp_C3` derives `hconv`; SP's `f_xxx ≥ 7/4` (L4, from L2 and (H2)) stays an input.",
"  `ridge_capHyp_C3` derives `hconv`; since v2.0 `ridge_capHyp_C4` also derives SP's\n  `f_xxx ≥ 7/4` (L4, from L2's second statement and (H2)).")
rep('test_gate.py','self.assertEqual(len(m["targets"]), 108)','self.assertEqual(len(m["targets"]), 113)')
rep(A,"Inputs that remain: SP's `f_xxx ≥ 7/4` (L4, from L2 and (H2)) and the source data themselves (I3). |",
"Since v2.0, L2's second statement and L4's `f_xxx ≥ 7/4` are proved from (H2) (`ridge_capHyp_C4`). Inputs that remain: the source data themselves (I3). |")
s=open(A,encoding='utf-8').read()
anchor='| `ridge_source_facts`, `ridge_capHyp_C3` |'
i=s.index(anchor); j=s.index('\n',i)
rows="\n"+\
"| `hermite_gap_le_concave` | `g' = G` on `[a, c]`, `G(a) = G(c) = 0`, `G − (m/2)t²` concave ⇒ `g(a) − g(c) ≤ m(c − a)³/12` | SP L1 (one-sided) | `hermite_gap_le` uses only the lower bound on `G`, which the concave half gives. |\n"+\
"| `third_deriv_exists_two` | `g' = G`, `G' = G'`, `G'' = φ` continuous on `[−r/2, r/2]`, critical pins, gap `r³/6` ⇒ `∃ x₀ ∈ [−r/2, r/2], φ(x₀) ≥ 2` | SP L2, second statement (\"by continuity some `x_0` has `f_xxx(x_0, 0) = 2`\") | By contradiction. The maximum `M' < 2` of `φ` makes `G − (M'/2)t²` concave, and then the gap is `≤ M'r³/12 < r³/6`. Lean proves `≥ 2`, which is all L4 needs. |\n"+\
"| `fxxx_lower` | `ψ` with derivative of norm `≤ n` on the collar `× B̄(0, 2r)` (`0 < ε ≤ r/2`), `ψ(x₀, 0) ≥ 2` with `x₀` in the pin interval, `rn ≤ 1/20` ⇒ `ψ ≥ 7/4` on the collar | SP L4 (\"`f_xxx ≥ 2 − 5rn ≥ 2 − 1/4 = 7/4` on `D`\") | One mean value step on the convex collar. In the product (max) norm the displacement is at most `max(3r, 2r) = 3r`, so `f_xxx ≥ 2 − 3rn`. |\n"+\
"| `ridge_capHyp_C4` | `ridge_capHyp_C3`'s hypotheses with `f_xxx ≥ 7/4` replaced by: `Φ4 p` the derivative of `p ↦ f_xxx(p)` with `‖Φ4‖ ≤ n` on the collar, and (H2) `rn ≤ 1/20` ⇒ the same conclusion | SP §2 normalization, (H1), (H2), L1–L20 | The cap from SP's hypotheses, with no analytic step assumed. `n` bounds only the derivative of `f_xxx`, which is all SP uses of `M_4`. |"
s=s[:j]+rows+s[j:]
open(A,'w',encoding='utf-8').write(s)
rep(A,"and the theorem yields its ridge and cap with no `hconv` supplied.",
"and the theorem yields its ridge and cap with no `hconv` supplied. `h1Toy_C4` (v2.0) meets every hypothesis of `ridge_capHyp_C4`, with `f_xxx = 2` constant, so `Φ4 = 0` and (H2) holds with `n = 0`.")
rep(A,"and of the analytic inputs only SP's `f_xxx ≥ 7/4` on `D` remains (L4, from L2's second statement and (H2));",
"and since v2.0 also SP's `f_xxx ≥ 7/4` (`ridge_capHyp_C4`, from L2's second statement and (H2) with a bound `n` on the derivative of `f_xxx`), so that no analytic step of L1–L20 is assumed;")
rep(A,"proved since v1.9 from the `C³` data, with `f_xxx ≥ 7/4` as input) |",
"proved since v1.9 from the `C³` data; the input `f_xxx ≥ 7/4` is proved since v2.0) |")
rep(A,"Since v1.9, Lean proves L6, L9 and L12 with SP's constants and derives `hconv` (`ridge_capHyp_C3`); `f_xxx ≥ 7/4` stays an input.",
"Since v2.0, Lean also proves L2's second statement and L4's `f_xxx ≥ 7/4` from (H2) (`ridge_capHyp_C4`). Since v1.9, Lean proves L6, L9 and L12 with SP's constants and derives `hconv` (`ridge_capHyp_C3`).")
rep(A,"its 93 v1.8 declarations are unchanged.",
"its 93 v1.8 declarations are unchanged. v2.0 adds four §2 rows (`hermite_gap_le_concave` to `ridge_capHyp_C4`), the `h1Toy_C4` sentence of §3 and a further revision of the SP row and I3; its 108 v1.9 declarations are unchanged.")
print("docs ok")
