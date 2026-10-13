def rep(p, old, new):
    s=open(p,encoding='utf-8').read()
    assert s.count(old)==1, (p, old[:90])
    open(p,'w',encoding='utf-8').write(s.replace(old,new))
R='README.md'; A='ALIGNMENT.md'
# README: narrative
old="So the hypotheses of `ridge_capHyp_C4` are SP's (H1), (H2), the critical pins, the gap `r³/6` and the derivative data with the bounds `m` (on the collar) and `n` (on `D`). No step of L1–L20 remains a hypothesis.\n"
rep(R,old,old+"\nSince v2.1, the quantitative bounds enter on `D` only:\n"
"- `ridge_capHyp_C5` derives six derivative-bound hypotheses of `ridge_capHyp_C4` from the block bound `m`, and from `|f_xxx| ≤ m` on the pin segment:\n"
"  - L3's inputs `‖∂_x D_y² f‖ ≤ m` and `‖∂_ρ D_y² f‖ ≤ m‖y‖` (`hessian_x_bound`, `hessian_r_bound`);\n"
"  - the convex and concave envelopes of `∂_y f(·, 0)` and `∂_x f(·, 0)`, which are the inputs of L5–L6 and L1–L2 (`axis_convexConcave`).\n"
"- `ridge_capHyp_D` assumes the bounds `m`, `n`, (H1) and (H2) on SP's `D` only. On a collar of some width `ε₀ > 0` it assumes the derivatives, and some bound `K` on the `x`-derivative of the third derivatives (SP: `f ∈ C⁴` near `D`).\n"
"  - The continuity step `block_bound_collar` gives the block bound `m + Kε` on the collar of width `ε`.\n"
"  - Because (H1) is strict, some `ε > 0` keeps `8r(m + Kε)² < λ`.\n")
# README: table rows
old_row="| `h1Toy_C4` | `h1ToyF` meets every hypothesis of `ridge_capHyp_C4` (`f_xxx = 2` constant, `n = 0`). The theorem gives its ridge and cap with neither `hconv` nor `f_xxx ≥ 7/4` supplied |"
rep(R,old_row,old_row+"\n"
"| `hessian_x_bound`, `hessian_r_bound` | L3's inputs from the block bound (v2.1). `∂_x D_y² f (x, 0)[v, w] = T3(x, 0)[(0, v), (0, w), (1, 0)]` and `∂_ρ D_y² f (x, sy)[v, w] = T3(x, sy)[(0, v), (0, w), (0, y)]`. The identification uses derivatives within an interval, so collar endpoints and the sphere are allowed. So `‖∂_x D_y² f‖ ≤ m` and `‖∂_ρ D_y² f‖ ≤ m‖y‖` |\n"
"| `axis_convexConcave` | the envelope inputs (v2.1). If `|∂_x² (Df(·, 0) u)| ≤ B` on `[a, c]`, then `Df(·, 0) u ± (B/2)t²` is convex, respectively concave. Take `u = (0, v)`, `B = m‖v‖` for `w` (L5–L6), and `u = (1, 0)`, `B = m` for `G0` (L1–L2) |\n"
"| `block_nonneg`, `block_bound_collar` | the continuity step of I3 (v2.1). Assume the block bound `m` on `D`, and some bound `K` on the `x`-derivative of `T3` in block form on a collar of width `ε₀`. Then the block bound `m + Kε` holds on the collar of width `ε ≤ ε₀`, by the mean value inequality from the nearest point of `D` |\n"
"| `ridge_capHyp_C5` | `ridge_capHyp_C4` with `hHxb`, `hHrb`, `hwcv`, `hwcc`, `hG0cv` and `hG0cc` derived from the block bound and `|f_xxx| ≤ m` on the pin segment (v2.1) |\n"
"| `ridge_capHyp_D` | the cap from SP's bounds on `D` (v2.1). The bounds `m`, `n`, (H1) and (H2) are assumed on `D`, with qualitative derivative data and some `K ≥ 0` on a collar of width `ε₀`. It concludes the ridge on `(−2r − ε, 2r + ε)`, `h(∓r/2) = 0`, the cap and (C2⁺), for some `ε ∈ (0, ε₀]` |\n"
"| `h1Toy_D` | `h1ToyF` meets every hypothesis of `ridge_capHyp_D` (`T4 = 0`, `K = 0`, `ε₀ = 1/2`). The theorem gives a ridge on `(−2 − ε, 2 + ε)` for some `ε > 0`, and the cap |")
# README: not proved
rep(R,"- the source data themselves. Since v2.0 every analytic step of L1–L20 is proved: L1–L12, L13 and L16–L20, with L7's\n"
"  existence, uniqueness and regularity of `h`. `ridge_capHyp_C4` takes, as hypotheses:\n"
"  - the derivatives `Df`, `DG`, `H`, `D2f` and the third derivatives, with the bounds `m`, on the collar\n"
"    `[−2r − ε, 2r + ε] × B̄(0, 2r)`;\n"
"  - the derivative of `f_xxx`, with its block bound `n`, on `D`.\n\n"
"  SP has `f ∈ C⁴` near `D` and bounds the third derivatives on `D`. On a thin enough collar the bound `m + η` holds by continuity, and (H1) stays strict for\n"
"  small `η`; Lean does not prove this step;\n",
"- the source data themselves. Since v2.0 every analytic step of L1–L20 is proved: L1–L12, L13 and L16–L20, with L7's\n"
"  existence, uniqueness and regularity of `h`. Since v2.1 `ridge_capHyp_D` takes, as hypotheses:\n"
"  - the quantitative bounds on SP's `D` only: the third-derivative block bound `m`, `|f_xxx| ≤ m` on the pin segment,\n"
"    the block bound `n` on the derivative of `f_xxx`, (H1) and (H2);\n"
"  - qualitative data on some collar `[−2r − ε₀, 2r + ε₀] × B̄(0, 2r)`: the derivatives `Df`, `DG`, `H`, `D2f`, the\n"
"    third derivatives and their `x`-derivatives, with some bound `K` on the latter.\n\n"
"  The continuity step from `D` to the collar is proved (`block_bound_collar`). What stays informal is that SP's\n"
"  `f ∈ C⁴` near `D` supplies these data and some `K`, by compactness of a closed collar;\n")
rep(R,'- prints the axioms of all 113 declarations,','- prints the axioms of all 121 declarations,')
# ALIGNMENT: SP row
rep(A,"Since v2.0, L2's second statement and L4's `f_xxx ≥ 7/4` are proved from (H2) (`ridge_capHyp_C4`). Inputs that remain: the source data themselves (I3). |",
"Since v2.0, L2's second statement and L4's `f_xxx ≥ 7/4` are proved from (H2) (`ridge_capHyp_C4`). Since v2.1, the six derivative-bound hypotheses are derived from the block bound (`ridge_capHyp_C5`), and the quantitative bounds are assumed on `D` only, with the continuity step to the collar proved (`ridge_capHyp_D`). Inputs that remain: the source data themselves (I3). |")
# ALIGNMENT: §2 rows
s=open(A,encoding='utf-8').read()
anchor='| `ridge_capHyp_C4` |'
i=s.index(anchor); j=s.index('\n',i)
rows="\n"+\
"| `hessian_x_bound` | `y ↦ ∂_y f(t, y)` with derivative `H(t, 0)` at `0`, `Df` with derivative `D2f(t, 0)` at `(t, 0)` for `t ∈ J`, `J` uniquely differentiable at `x ∈ J`, `T3(x, 0)` the derivative of `D2f(·) a b` at `(x, 0)`, `|T3(x, 0)[(0, v), (0, w), (1, 0)]| ≤ m‖v‖‖w‖`, `t ↦ H(t, 0)` with derivative `Hx'` at `x` ⇒ `‖Hx'‖ ≤ m` | SP L3 (\"`‖∂_x D_y² f‖ ≤ M₃`\") | The proof identifies `Hx' v w` with the `T3` block by uniqueness of derivatives within `J` (`transverse_hessian_eq` on `J`). So collar endpoints are allowed. |\n"+\
"| `hessian_r_bound` | the same along the radius `τ ↦ (x, τy)`, `τ ∈ [0, 1]`: `|T3(x, sy)[(0, v), (0, w), (0, y)]| ≤ m‖v‖‖w‖‖y‖`, `τ ↦ H(x, τy)` with derivative `Hr'` at `s` ⇒ `‖Hr'‖ ≤ m‖y‖` | SP L3 (\"`‖D_y³ f‖ ≤ M₃`\" along rays) | Derivatives are taken within `[0, 1]`, so `‖y‖ = 2r` is allowed. |\n"+\
"| `axis_convexConcave` | `Df` with derivative `D2f(t, 0)` and `D2f(·)(1, 0) u` with derivative `T3(t, 0)(1, 0) u` at `(t, 0)`, `|T3(t, 0)[(1, 0), u, (1, 0)]| ≤ B` on `[a, c]` ⇒ `Df(·, 0) u + (B/2)t²` convex and `Df(·, 0) u − (B/2)t²` concave on `[a, c]` | SP L1 (`|g_0'''| ≤ m`), L5–L6 (`‖w''‖ ≤ m`) | `u = (0, v)`, `B = m‖v‖` gives `hwcv`, `hwcc`. `u = (1, 0)`, `B = m` gives `hG0cv`, `hG0cc` (SP: `|f_xxx| ≤ M₃`). |\n"+\
"| `block_nonneg`, `block_bound_collar` | `K ≥ 0`, `0 ≤ ε ≤ ε₀`; the block bound `m` on `D = [−2r, 2r] × B̄(0, 2r)`; on the collar of width `ε₀`, `T4 p a b c` the derivative of `t ↦ T3(t, p₂) a b c` at `p₁`, with the block bound `K` ⇒ the block bound `m + Kε` on the collar of width `ε` | I3's continuity step (SP bounds `M₃` on `D`; `f ∈ C⁴` near `D`) | Each collar point `(x, y)` is within `ε` of a point `(x₀, y) ∈ D`. The mean value inequality along `x` on the convex interval of width `ε₀` adds at most `Kε` times the block weight. |\n"+\
"| `ridge_capHyp_C5` | `ridge_capHyp_C4`'s hypotheses without `hHxb`, `hHrb`, `hG0cv`, `hG0cc`, `hwcv`, `hwcc`, and with `|T3(x, 0)[(1, 0), (1, 0), (1, 0)]| ≤ m` on the pin segment ⇒ the same conclusion | SP §2, L1–L20 | The six hypotheses are derived: `hHxb`, `hHrb` by the first two rows; `hwcv`, `hwcc` from the block bound by `axis_convexConcave`; `hG0cv`, `hG0cc` from the pin-segment bound. |\n"+\
"| `ridge_capHyp_D` | `0 < ε₀`, `K ≥ 0`; on the collar of width `ε₀`: `Df`, `D2f`, `T3`, `T4` (as above) with the block bound `K`, `DG` (strict), `H`, `Hx`, `Hr`; on `D`: the block bound `m`, `Φ4` with its two blocks `≤ n`; `|f_xxx| ≤ m` on the pin segment; (H1) `8rm² < λ`, (H2) `rn ≤ 1/20`; critical pins; gap `r³/6` ⇒ `∃ ε ∈ (0, ε₀]` with `ridge_capHyp_C4`'s conclusion on `(−2r − ε, 2r + ε)` | SP §2 (H1), (H2), `M₃`, `M₄` on `D`, `f ∈ C⁴` near `D`; L1–L20 | Since (H1) is strict, `ε` is chosen with `8r(m + Kε)² < λ`, `ε ≤ ε₀` and `ε ≤ r/2`, and `ridge_capHyp_C5` is applied with `m + Kε`. `K` is any finite bound, which SP's `C⁴` gives on a compact collar. |"
s=s[:j]+rows+s[j:]
open(A,'w',encoding='utf-8').write(s)
# ALIGNMENT: §3 toy sentence
rep(A,"`h1Toy_C4` (v2.0) meets every hypothesis of `ridge_capHyp_C4`, with `f_xxx = 2` constant, so `Φ4 = 0`, both blocks vanish and (H2) holds with `n = 0`.",
"`h1Toy_C4` (v2.0) meets every hypothesis of `ridge_capHyp_C4`, with `f_xxx = 2` constant, so `Φ4 = 0`, both blocks vanish and (H2) holds with `n = 0`. `h1Toy_D` (v2.1) meets every hypothesis of `ridge_capHyp_D` with `ε₀ = 1/2`: `T3` is constant, so `T4 = 0` and `K = 0`, and `|f_xxx| = 2 = m` on the pin segment. The theorem chooses `ε` itself and yields a ridge on `(−2 − ε, 2 + ε)` and the cap.")
# ALIGNMENT: I3
rep(A,"`m` as a bound on the third derivatives, which gives the convexity pairs for `G0 = ∂_x f(·, 0)` (L1's input `|g_0'''| ≤ m`) and for `w` (L5's input `‖w''‖ ≤ m`);",
"`m` as a bound on the third derivatives, which gives the convexity pairs for `G0 = ∂_x f(·, 0)` (L1's input `|g_0'''| ≤ m`) and for `w` (L5's input `‖w''‖ ≤ m`); since v2.1 these pairs and L3's bounds on `∂_x D_y² f` and `D_y³ f` are derived from the block bound and `|f_xxx| ≤ m` on the pin segment (`ridge_capHyp_C5`);")
rep(A,"SP has `f ∈ C⁴` near `D` and bounds the third derivatives on `D`; on a thin enough collar a bound `m + η` holds by continuity and (H1) stays strict for small `η`. Lean does not prove this step.",
"SP has `f ∈ C⁴` near `D` and bounds the third derivatives on `D`. Since v2.1 Lean proves the continuity step (`block_bound_collar`, `ridge_capHyp_D`). With the block bound `m` on `D`, and some bound `K` on the `x`-derivative of the third derivatives on a collar of width `ε₀`, the bound `m + Kε` holds on the collar of width `ε`, and (H1) stays strict for small `ε`. What stays informal is that `f ∈ C⁴` near `D` supplies the derivative data and some `K`, by compactness of a closed collar.")
rep(A,"| SP §2 (H1), L6, L7, L9–L12; CAP §§2–5 | These are checked with exact constants by SP's script. Since v2.0,",
"| SP §2 (H1), L6, L7, L9–L12; CAP §§2–5 | These are checked with exact constants by SP's script. Since v2.1, Lean derives the six derivative bounds from the block bound and proves the continuity step from `D` to the collar (`ridge_capHyp_C5`, `ridge_capHyp_D`). Since v2.0,")
# ALIGNMENT: §5 history
rep(A,"its 108 v1.9 declarations are unchanged. A same-provider read,",
"its 108 v1.9 declarations are unchanged. v2.1 adds six §2 rows (`hessian_x_bound` to `ridge_capHyp_D`, seven theorems), the `h1Toy_D` sentence of §3 and a further revision of the SP row and I3; its 113 v2.0 declarations are unchanged. A same-provider read,")
rep('test_gate.py','self.assertEqual(len(m["targets"]), 113)','self.assertEqual(len(m["targets"]), 121)')
print('ok')
