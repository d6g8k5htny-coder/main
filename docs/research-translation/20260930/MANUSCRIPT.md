# Short-lifetime H₀ persistence of a periodized Gaussian field

**Standalone draft exposition, 30 September 2026.** This draft assembles the source-reviewed planar statement, its mechanism and its coefficient. It is not a new proof acceptance, a complete journal submission or a replacement for the full proofs linked below. The mathematical source cut is Math- `fa2e9909d8cca40d38b7dc2eadda6766bd0788e1`. The [proof appendices](APPENDICES.md) expand the scalar derivations and identify the load-bearing imported proofs; the [literature comparison](LITERATURE.md) checks specific related theorems and their distinct observables. The [paragraph crosswalk](CROSSWALK.md) specifies the repaired source chain and remaining assembly work; [SOURCES.json](SOURCES.json) binds the principal sources. Earlier manuscript work exists: live research discussions refer to a V3 manuscript and referee map. Their complete text was not recovered for this draft, so this exposition makes no claim to be the project's first manuscript or to revise those private documents.

## Abstract

For a smooth Gaussian function on a compact surface, a finite superlevel H₀ persistence bar records a connected component born at a local maximum and its eventual merger into an older component. We consider the centered, variance-one Gaussian field on the square torus of side 24 whose covariance is the normalized periodization of `exp(−|z|²/2)`. The expected number of finite bars per unit area and per unit lifetime has a density satisfying `ν(ℓ) = c₂,₂₄ ℓ^(−1/3) + O(1)` as `ℓ ↓ 0`. Here `0.07340691930603427103 < c₂,₂₄ < 0.07340691930603427104`. The coefficient enclosure evaluates a specific contact-jet integral; its persistence interpretation uses the separate global elder-selection and marked Kac–Rice arguments. The remainder is existential: no numerical error constant or usable lifetime cutoff has been established by this chain. We explain why a local separating cap decides the global elder event, how determinant weighting produces the radial intensity, and why removing the birth and gap restrictions needs a separate estimate. A numerical experiment can compare the same observable after discretization, but does not prove the continuum asymptotic.

## 1. The field and what is counted

**[M1]** Let `X = ℝ²/(24ℤ²)` with its flat metric and area `|X| = 576`. Let `f` be the centered stationary Gaussian field with covariance

```text
K₂₄(z) = Σ[n∈ℤ²] exp(−|z+24n|²/2)
         / Σ[n∈ℤ²] exp(−|24n|²/2).
```

This normalization gives `Var f(x) = 1`. The exact field is periodic; it is not replaced by a nonperiodic field. Its Fourier variances are

```text
a_n = exp(−2π²|n|²/24²) / Σ[j∈ℤ²] exp(−2π²|j|²/24²).
```

Every `a_n` is positive and the coefficients decay rapidly enough for a smooth version with finite moments of all derivative suprema. The full positive spectrum also gives the finite-jet nondegeneracy used below. The square torus does not have arbitrary rotational symmetry; orientation is retained in the conditional laws and integrated at the end.

**[M2]** Decrease the level `t` through the superlevel sets `{f ≥ t}`. A local maximum creates a component. When components merge, ordinary elder pairing retains the component with the higher birth value and kills the younger one. A finite bar has birth `b`, death `s < b`, and lifetime `ℓ = b − s`. Almost surely the relevant field is Morse with distinct critical values; the death of an H₀ bar is an index-one merging saddle. Index counts negative Hessian eigenvalues, so a maximum has index two. The single essential class born at the global maximum is excluded. Saddles producing other topological changes are not counted as H₀ deaths merely because they are saddles.

For a Borel set `A ⊂ (0,∞)`, define the expected lifetime measure

```text
μ(A) = (1/576) E Σ[finite ordinary superlevel H₀ bars i] 1{ℓ_i ∈ A}.
```

Its density is denoted `ν`. This is an intensity, not the lifetime distribution of a bar drawn uniformly from each realization: there is no division by the random number of bars. All birth heights and all spatial separations are included.

**[M3]** A second measure counts every ordered maximum/index-one-saddle pair with positive height gap, whether or not the saddle kills that maximum. Write its density `ν_cand`. The selected density `ν_eld` is exactly `ν`. These roles distinguish the two endpoints, so no factor `1/2` enters the pair count. A candidate pair rejected by the elder selector is not an additional bar.

## 2. The assembled planar statement

**[M4]** The repaired parent leading law, the bounded-remainder theorem and the SIDE24 coefficient evaluation give the following statement for the field and measures just defined. There are constants `C < ∞` and `ℓ_* > 0` such that suitable density versions satisfy, for `0 < ℓ ≤ ℓ_*`,

```text
|ν_cand(ℓ) − c₂,₂₄ ℓ^(−1/3)| ≤ C,
0 ≤ ν_cand(ℓ) − ν_eld(ℓ) ≤ C,
|ν_eld(ℓ) − c₂,₂₄ ℓ^(−1/3)| ≤ C,

0.07340691930603427103 < c₂,₂₄ < 0.07340691930603427104.
```

The statement concerns expected densities. It does not assert that every realization has this shape. The coefficient interval is an arithmetic enclosure of the coefficient expression, not a confidence interval for a measured histogram. Its interpretation as the bar-intensity coefficient uses the parent chain in §§3–6 below. There is no additional blanket pairing hypothesis in that source chain.

**[M5]** Integration gives a directly measurable consequence:

```text
(1/576) E N_eld(0,t] = (3/2)c₂,₂₄ t^(2/3) + O(t).
```

More generally, for `q > −2/3`,

```text
(1/576) E Σ[finite bars with ℓ_i ≤ t] ℓ_i^q
  = c₂,₂₄ t^(q+2/3)/(q+2/3) + O(t^(q+1)).
```

The positive leading density gives the inverse-lifetime integrability threshold `p < 2/3` for `ℓ^(−p)` near zero, with logarithmic divergence at equality. The `O(1)` density remainder does not supply a second coefficient or show that the remainder converges. Neither `C` nor `ℓ_*` is evaluated here.

## 3. Why local geometry can determine a global death

**[M6]** Fix an orientation `u` and a transverse unit vector. Place pins at `M = −ru/2` and `S = ru/2`, with heights `f(M)=b`, `f(S)=b−kr³`, `k>0`, and zero gradients. There are six observed scalars in dimension two. Their continuous Gaussian regression law is `Q`, and the correct Kac–Rice weight is

```text
W_r = |det H_M det H_S| 1{H_M < 0, index(H_S)=1},
Z_r = E_Q W_r,                      dQ^W = (W_r/Z_r)dQ.
```

`Z_r` is the full typed determinant normalizer. It includes no pin density, coordinate Jacobian or adjacency restriction. Let `p_r(b,k,u)` be the `Q^W` probability that `S` is the actual global elder partner of `M`.

**[M7]** The deterministic cap theorem supplies a sufficient event for that global statement. In axial/transverse coordinates use the cylinder `D=[−2r,2r]×[−2r,2r]`, and let `M_j` be the maximum of the absolute coordinate partial derivatives of total order `j` on `D`. Put `λ=−f_yy(M)`. The sufficient event is

```text
G_r = { λ > [4/(3k)] r M₃²,    r M₄ ≤ 3k/10 }.
```

The torus chart is embedded when `r < 24/(4√2)`; the asymptotic radius is further reduced as necessary. On `G_r`, strict transverse concavity produces a ridge of transverse maxima and a closed cap containing `M`. Every cap-boundary exit has height at most `s=b−kr³`, while a path through `S` reaches a point higher than `b` with minimum exactly `s`. Thus

```text
d_f(M) = sup[paths γ from M to a point of height > b] min_t f(γ(t)) = s.
```

The cap bounds every possible excursion into the rest of the torus. On the Morse, distinct-critical-value locus, this maximin value identifies the unique elder death saddle `S`. The ridge need not be a gradient trajectory, and no Morse–Smale assumption is used. The essential maximum has no older endpoint and cannot occur on this good event.

**[M8]** The probability estimate is made under `Q^W`, not under the unweighted regression law. On a compact birth interval `B` and a compact gap interval `K ⊂ (0,∞)`, the contact regression gives

```text
Z_r/r² → z₀(b,k,u)
        = (6k)² E[A₀² 1{A₀<0}],             0 < z_* ≤ Z_r/r² ≤ z^*.
```

Here `A₀` is the scalar transverse Hessian under the contact regression. Its law is nondegenerate. The endpoint Hessians are scaled by the congruence `diag(r^(−1/2),1)`. The repaired argument uses joint convergence in law of these scaled Hessians, obtained from vanishing in-probability errors and convergence in law of the transverse block. It does not identify the limiting block on an unstated common probability space.

For the planar soft eigenvalue `λ>0`, the typed determinant product retains both factors

```text
W_r ≤ (r² M₃²/4) λ(λ+3rM₃/2).
```

Regressing the field further on the endpoint transverse Hessian leaves an independent residual with uniformly bounded derivative moments. Integrating the displayed weight over the shallow-curvature failure region yields `E_Q[W_r;G_r^c] ≤ C r⁵`; a separate large-`λ` branch and the fourth-derivative exception are also retained. Division by the full `r²` normalizer gives `1−p_r ≤ Q^W(G_r^c) ≤ C r³`, uniformly on `B×K` and over orientations. This is the compact-mark selection estimate. Its constant is not asserted uniform as `k ↓ 0` or `|b| → ∞`.

## 4. The intensity calculation and the exponent

**[M9]** A global elder mark is Borel but need not be continuous. The repaired marked Kac–Rice argument first works on compact pair domains away from the spatial diagonal. It constructs the conditional law of the entire field in `C²(X)`, identifies two finite measures on location times field space using continuous cylinder functions, and extends their equality to bounded Borel marks by a functional monotone-class argument. Height disintegration then produces the full pin density times `E_Q W_r` for candidates and the same expression times `p_r` for selected pairs. No second determinant or second normalizer is introduced. Exhaustion toward the diagonal is justified by the locally integrable radial bound below.

**[M10]** Use midpoint, directed separation `h=ru`, birth `b`, and scaled gap `k`. If `π_r` denotes the density of the nonsingular contact observation coordinates, their exact transformation from the six original pins has determinant `12r^(−5)`. The target is `(b−kr³/2,−kr²,0,12k,0,0)`. The factors in the planar intensity are

| Factor | Contribution |
|---|---:|
| Polar separation | `r dr dσ(u)` |
| Height change `(b,k) ↦ (b,b−kr³)` | `r³ db dk` |
| Pin density | `12r^(−5) π_r` |
| Typed determinant normalizer | `Z_r`, of order `r²` |

Here `dσ` is ordinary arc length on `S¹`, whose total mass is `2π`. Write `𝒜_r` for the intensity amplitude (the source denotes it `A_r`; it is distinct from the transverse Hessian `A₀` in §3). Multiplication gives

```text
r 𝒜_r(b,k,u) dr db dk dσ(u),       𝒜_r = 12π_r Z_r/r².
```

Since `ℓ=kr³`, the change of variable is `r dr/dℓ = ℓ^(−1/3)/(3k^(2/3))`. This is the origin of the exponent `−1/3`. On compact positive-length intervals `B,K`,

```text
c_(B,K) = ∫[B×K×S¹] 𝒜₀(b,k,u)/(3k^(2/3)) db dk dσ(u),
ν_cand^(B,K) ∼ ν_eld^(B,K) ∼ c_(B,K) ℓ^(−1/3),
0 ≤ ν_cand^(B,K) − ν_eld^(B,K) ≤ C_(B,K) ℓ^(2/3).
```

The last estimate uses `r³=ℓ/k` and `k` bounded away from zero. This coefficient belongs to a spatially and mark-restricted population. It is not `c₂,₂₄`.

## 5. Removing the restrictions and controlling the remainder

**[M11]** The unrestricted leading limit uses the unnormalized product `π_r Z_r`, avoiding division by a potentially small normalizer. On a fixed radius band independent of `b,k`, the parent proves

```text
0 ≤ 𝒜_r(b,k,u) ≤ C(1+|b|+k)^4 exp[−c(b²+k²)].
```

After multiplication by `k^(−2/3)`, this is integrable both at `k=0` and at infinity. At lifetime `ℓ`, the near-pair restriction `r≤r₀` is precisely `k≥ℓ/r₀³`. For every fixed `b,k>0,u`, `𝒜_r→𝒜₀` and `p_r→1`; dominated convergence gives the common unrestricted near-pair coefficient. Fixed far separations have a uniformly nonsingular original pin covariance, giving a bounded density contribution. This establishes the leading law for all finite bars without exporting the compact selection constant to all marks.

**[M12]** The bounded remainder requires stronger estimates than that dominated-convergence argument. D2 couples the centered contact regressions to second order and exploits a quadratic determinant cancellation: if a block matrix has off-diagonal entries `√r β`, its determinant is affine in `r`, even when the transverse block is singular. After the inertia filter and expectation, D2 obtains a polynomial-Gaussian majorant `H(b,k)` with

```text
𝒜_r ≤ (k+r)² H,        𝒜₀ ≤ k² H,
|𝒜_r−𝒜₀| ≤ r(k+r)H.
```

For the unnormalized cap-loss intensity it additionally proves `B_r ≤ (r/k)³ H` when `r/k≤1`, and always `𝒜_r(1−p_r)≤B_r≤𝒜_r`. These are all-mark intensity bounds, not a globally uniform probability estimate.

In the lifetime integral the lower cutoff `a=ℓ/r₀³` is retained. The candidate error monomials are `k^(−2/3)rk=ℓ^(1/3)` and `k^(−2/3)r²=ℓ^(2/3)k^(−4/3)`. Their integrals from `a` give a scaled error `O(ℓ^(1/3))`, hence an unscaled `O(1)` remainder. For selection loss D2 splits at `η=ℓ^(1/6)`, uses the intensity bound below `η` and the cap-loss bound above it, and obtains the same order. Adding the bounded far contribution proves §2. Replacing the lower cutoff by zero in the second monomial would create a divergent integral and invalidate this calculation.

## 6. The exact coefficient and its evaluation

**[M13]** At a contact point write `G=∇f`, `V_u=H_f u`, `t_u=∂_u³f`, and let `A_u` be the scalar transverse Hessian. Define

```text
τ_u² = Var(t_u | G=0),
D_u  = E[A_u² 1{A_u<0} | V_u=0].
```

The covariance is even, so the odd derivative vector `(G,t_u)` is independent of the even derivatives. Integrating out birth and positive gap gives the exact expression

```text
c₂,₂₄ = Γ(7/6)/(24^(1/3)√π)
         × ∫[S¹] p_G(0) p_(V_u)(0) τ_u^(4/3) D_u dσ(u).
```

The `24` in the prefactor is a numerical change-of-variable factor valid for any side length; the side length enters through the actual covariance of the jets. The angular measure remains unnormalized. These conventions are necessary to use the quoted enclosure.

**[M14]** SIDE24 first evaluates a nonperiodic reference contact expression. For `exp(−|z|²/2)`, the planar conditional transverse variance is `8/3`, so its negative-half second moment is `4/3`; also `τ²=6` and `p_G(0)p_V(0)=(2π)^(−2)/√3`. It then bounds all omitted periodic images through derivative order six, uniformly in orientation, compares the full joint covariance and its Schur complements, and integrates Gaussian density bounds over the negative cone. This proves a relative difference below `10^(−106)` between the periodic and reference coefficient expressions. The periodic coefficient is not declared equal to the reference one.

Outward rational interval arithmetic, including explicit remainders for the special functions, yields the interval in §2. Those arithmetic and perturbation results have their own scoped review record. They acquire a bar-intensity meaning by composition with the repaired parent; the arithmetic replay alone proves neither elder selection nor Kac–Rice.

## 7. A comparison that the experiment can make

**[M15]** A sample-based comparison should count finite ordinary superlevel H₀ bars from each realization, exclude its essential component, and divide the aggregate bin counts by the number of realizations, area `576`, and bin width. For a bin `[a,b)`, the leading prediction per unit area is

```text
∫[a,b] c₂,₂₄ ℓ^(−1/3)dℓ = (3/2)c₂,₂₄(b^(2/3)−a^(2/3)).
```

Using this integrated prediction avoids assigning the singular curve a representative value at the zero endpoint. Realizations, rather than the correlated bars within one realization, are the replication units. The [experiment protocol](EXPERIMENT.md) specifies the generator, filtration conventions, deterministic controls and refinement comparisons. The [executed 32-field pilot](../../../experiments/periodic_h0/results/pilot32/RESULTS.md) reports its actual spectral and grid choices and preserves every bin. The short-bin counts drift under grid refinement, so its coefficient comparison is inconclusive at these resolutions. The [eight-field refinement through 1024²](../../../experiments/periodic_h0/results/refinement8/RESULTS.md) adds two diagonal controls: some larger bins stabilize numerically, but the shortest bins remain resolution-sensitive. The [deterministic approximation note](../../../experiments/periodic_h0/APPROXIMATION.md) supplies a quadratic spatial bound conditional on certified inputs; those evaluated enclosures are still missing. Neither numerical result certifies nor refutes the continuum statement.

A finite spectral truncation changes the field and a finite grid changes the filtration. Agreement across resolutions is evidence about those computed approximations; this draft supplies no bound transferring the small-lifetime density of those approximations to the exact continuum density. Nor does the coefficient enclosure locate a lifetime range in which the unknown remainder is small. A visible discrepancy or a stable histogram can motivate further work without confirming or refuting an asymptotic statement outside a certified regime.

## 8. Scope and remaining work

**[M16]** The main statement assembles D1, D2 and SIDE24 at their reviewed consumption scopes. The planar refined-selector packet is a separate result: it identifies the actual local partner in a rare cubic sector and a compact rejected-candidate coefficient. Counting two additional saddle witnesses does not supply a persistence-pair measure, and a rejected candidate is not a replacement bar. The unrestricted rejected population has a positive far-separation lower bound in the separate Theorem U record; this is compatible with D2's bounded upper bound and rules out exporting the compact `O(ℓ^(2/3))` difference to all pairs.

The next quantitative obligation for this observable is an explicit remainder constant and a justified lifetime window, together with a continuum-to-computation error analysis. Other dimensions, refined selector laws, other covariances, higher homology and joint volume/lifetime limits require their own stated source chains. This draft does not consume live far-elder rate candidates or higher-dimensional partner PRs. It uses only the fixed planar model above.

The [expanded appendices](APPENDICES.md) now provide the corrected regression/normalizer argument, scalar failure integral, Borel extension, mark/radius integration and coefficient normalization in reading order. The full cap/ridge proof and interval implementation remain explicit source imports; a publication version still needs those complete supplementary proofs, a broader literature/priority check, and documented human mathematical review. The [theorem-level comparison](LITERATURE.md) is a bounded primary-source read, not exhaustive novelty clearance. The crosswalk identifies exactly what the present draft imports and what it has not checked. Existing model reviews, even when their providers differ, share an account and carry no organizational-independence credit. Neither those reviews nor source hashes, software tests, a numerical fit or this editorial assembly establish human refereeing or journal acceptance.
