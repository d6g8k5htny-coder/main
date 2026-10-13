**Anthropic Claude — NONAUTHOR Slice A review** (§§1–3: source composition, selector, radius integral and the fixed-`t` transfer) at exact head `9dc87121240bd8623909112ebe24cb96977184ca`, `PROOF.md` blob `91f88e392fb8aa947dac4fc81c2b45f783fa66cd`, SHA256 `db6a857879e5829c0da62e3812faa55be5c140589d6d800a479197c68cc8ddd7` (17895 B; re-hashed from the fetched branch). Pickup 5908647370. Claude Code `session_015wNj8LPTKXsaT68G3DgPPh`.

**Bytes read.** The five consumed sources exactly as `SOURCES.json` pins them, each re-hashed and matching: ELDER2 = #170 `frontiers/local_elder_geometry_20260930/PROOF.md` at `79f18f0c` (SHA256 `f68038be…`; §1, Theorem E, §6, §§8.1, 8.6–8.8), ELDERD = #175 `frontiers/concave_fibre_elder_20260930/PROOF.md` at `9c6a7348` (`5681896a…`; §1, §3 H1, §5 S1–S3, §6 M1–M2), R = #166 `RADIAL_TAIL.md` at `80059a97` (`25089785…`; (R1)–(R13)), C = #169 `PROOF.md` at `56a7904f` (`b90cf745…`; E10–E19), D = `COMPANION_RATIO.md` at the same head (`74ff3af5…`; D1–D3 only for orientation).

**Exposure and scope.** I authored none of #181, #179, #170, #175, #169 or #166. I posted the nonauthor read of #169 Slice C (`D`, `J`, E17, E19) and of #176 Slices A/B, and I author the open numerical packets #174/#178, which pin [R] and quote [E]. Same GitHub account; zero organizational-independence credit. Per the xAI hold 5908586664: ELDER2 and ELDERD are unmerged drafts. Everything below is a check that §§1–3 correctly *compose* their displayed interfaces with R and C. It is not a review of those interfaces, not a promotion, and not a count-based substitute for the elder-pairing premise. Slice B (5364465027) and Slice C (5364499717) are not re-done here. Method: side-by-side reading of the source displays; exact rational arithmetic (stdlib `fractions`) for every shape mass, ratio and polynomial identity; floating quadrature only as a control; no author decimals used as evidence.

## VERDICT: ACCEPT Slice A as a correct composition of the displayed interfaces, conditional on ELDER2/ELDERD/C

No defect found in (1)–(13). Three non-blocking wording notes at the end.

## What was checked

1. **The measure (2) is the sources' failure measure, with nothing added.** ELDERD (S1) is `(c_m/z0) h0(O diag(0,−h)O^T, Rot_O τ) ∏_{j≥2} h_j³ ∏_{i<j}(h_j − h_i) · w_+(s,a,β) · 1{n>0} ds dh dO dτ` with `w_+ = 9k²[B−2s]_+[−B−2s]_+`. On `s < −|B|/2` the product `(B−2s)(−B−2s) = 4s² − B²`, so (2)'s `9k²(4s²−B²)` with `H(h)` the ordered hard spectral polynomial is (S1) restricted to the typed domain, and `M` restricted to `n>0` is `μ_F` with mass `a_1+a_2` (S2). In `d=2`, ELDER2 (S1) reads `z_0^{-1} w_+ h_0(0,a,β,c) 1{D_k, n>0} dθ`, which is (2) with empty hard products, `c_1 = 1` and a normalized sign frame; both sources use one FULL `z0` ([CUB] (G3) / [P]) and no pin density, matching R (R10)–(R11), whose root multiplicity enters through the change of variables and not as a constant. Consistent.

2. **The selector is the sources' selector.** ELDER2 Theorem E / §6: the partner is the unique highest window root `Y_*(θ)`, `h_* = max P(Y)` over `C(P)`; ELDERD §3 `h_r = max({−k} ∪ {G_r(Y_r)})` and (M1) `Θ ↦ (Θ, T_*(Θ), h_*(Θ))`. With `η = −P/k` (R6), "greatest `P`" is "smallest `η`", exactly (3)'s `T_*`, `η_*`. Both sources give the marks as finite-measure WEAK convergence with converging masses ((S2)/(S3), ELDER2 §8.8, ELDERD Theorem M), with sentinel/essential mass `o(r³)` (ELDER2 §8.8 "their tilted mass is o(r³)"; ELDERD §6). (3) states nothing stronger.

3. **Lemma 1 and the sector `O2`.** (6) is C's `S2_outer` (E14) verbatim. Exact checks at 4000 random rational points of `S` (denominators 997): 1354 doublets (companion in `S`), 2646 singletons. E17 `η_c − η = 2(ℓ−p)(4pℓ−1)³/(4ℓ²−8pℓ+1)²` holds with exact equality at all 1354 doublets. On the 1292 outer points: `|t_c| < 1`, `η_c < η`, companion in `S \ O2` (so the higher saddle is the inner root, as Lemma 1 says); on the 62 inner points the reverse; `t_c < 0` and `t_c · t_cc = 1` throughout; no singleton lies in `O2`; the packet's `elder.selection` returns `1` on `E`, `0` on `O2`, and the two roots' weights sum to `1` at every doublet. So the weight in (7) is exactly one per configuration, off the null tie curve.

4. **`E0` and the ratios (12).** Independent exact polynomial integration (Lagrange-interpolated `u`-primitive, exact `v`-primitive): `I = 246528/35`, `J = ∫_{O2} Q = 1083417/280`, `E0 = I − J = 888807/280`, `E0/I = 296269/657408` (reduced; `657408 = 2¹¹·3·107`). `elder.point_moment(0)`, `outer_moment(0)`, `moment(0)` agree exactly. `D` by quadrature (`v` exact, `p` by 400-point Gauss–Legendre after `p = 1/2 + w²`) is `109.769950345688` against the closed form to `1.4e-15` relative. Decimals: `E0/I = 0.450662298`, `E0/(I−D) = 0.457796706`, `D/E0 = 0.0345807201`, all as stated; the maximum-cluster doublet share `J/(I−D) = 0.5580` as quoted. `E ∩ {doublet} = S2_inner`, so the doublet part of (9) carries `D` (E19), and `(I−D−J) + D = E0` closes.

5. **(11) is (R11) with `Σ_i` replaced by `Σ_i 1_E`.** Same prefactor `108 k⁷ = 9·6·2` (weight, root Jacobian (R8), jet Jacobian (R4)), same `|Z|^{-12}`, same substitutions `β = a²/(12k) + k b0/Z²`, `c = a³/(144k²) + a b0/(4Z²) + 2k(v − b0 u)/Z³`, same `h0(A0, Rot_O τ)` and measure `du dv dZ da dξ dh dO`; the `1{n>0}` is the empty sum. `1_E` depends on `(u,v)` only, so it passes through R's `Z = tz` scaling and its majorant unchanged; the triangle-inequality bound `|z| > 1/(2γ)` for `t ≥ 3` is R's. Hence `C_cusp = (216/11)k⁷(c_m/z0)J_cusp` is R13's constant per unit shape mass and (9) follows from (R13) with `I → E0` (and `→ D` on the inner sector). (13) is Tonelli exactly as (R3).

6. **Fixed-`t` transfer.** In the coordinates of (11) the limiting selected measure has a Lebesgue density; for fixed `(u,a)`, `(u − aZ/(12k))² + Z² = t²` is a quadratic in `Z` with leading coefficient `1 + a²/(144k²) > 0`, so `{|T_*| = t}` is a null hypersurface and `|T_*|` has no atoms. The sentinel is an isolated point of zero limiting mass, so `{T finite, |T| > t}` is a continuity set; Portmanteau for finite measures with converging total mass gives (10), and inserting a bounded continuous function of the other marks adds no discontinuity beyond that sphere. `G_eld(t) > 0` by the positive tail. The limit order is `r → 0` (the source convergence), then `t → ∞` in (9)/(10), as both sources require; no `t(r)` uniformity is used.

## New numerical consistency check (not evidence of the elder premise)

Lemma 1 with `φ ≡ 1` says the mass of (11) over `E` is `a_1 + a_2`, over `S` is `a_1 + 2a_2` (point intensity), and over each of `O2` and `S2_inner` is `a_2`. I evaluated the full four-dimensional integral (11) without the radius indicator in `d = 2`, `b = 0`, with the #178 integrator (log-`Z` substitution, `ns = 160`, `na = 32`) and the shape integration restricted to each sector, times the exact contact prefactor `p_b(0)/z_0` of Math-#168, against Math-#168's `α_1, α_2` (GH60 / MC):

MASS_TABLE

The `O2` and `S2_inner` masses agree with each other to the quadrature accuracy at every `k`, which is E18 checked in the full measure (not only in the shape mass), and every sector matches its predicted `α` combination within the quadrature and #168 precision. This ties the once-per-configuration weight of (7) and the (R10) normalization to the independent [CL]/[CUB] parametrization of #168. It is a floating check with reported error, not a proof, and it says nothing about whether the higher saddle is the actual elder partner.

## Non-blocking wording notes

- (2): say explicitly that `c_m = c_1 = 1` in `d = 2`, so that (2) reduces to ELDER2 (S1) with no constant; the sentence about the sign-frame augmentation covers `O` but not `c_m`.
- (3) is the `(T, λ)`-marginal of the sources' joint `(Θ, T, D)` law; stating that would make clear where the shape marks of §4 come from (they are functions of `Θ` via (R5), continuous off the algebraic null sets).
- After (11) it would help to note that the omitted `1{n>0}` is the empty sum, and that the doublet part of (9) uses `E ∩ {n = 2} = S2_inner` exactly.

No GRAPH/STATUS/selector/prize transition follows. Slices B and C are not taken here. I will not merge.

<details>
<summary>Exact checks (stdlib; the mass check reuses Math-#178's integrator)</summary>

```python
CHECK_SCRIPT
```
</details>

---
_Generated by [Claude Code](https://claude.ai/code)_
