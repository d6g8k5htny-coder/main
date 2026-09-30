# Planar proof appendices for the short-lifetime exposition

These appendices expand the derivations behind [MANUSCRIPT.md](MANUSCRIPT.md). They assemble existing arguments at the fixed source cut below and expose the imported propositions. They are not a newly accepted theorem, a reconstruction of an unseen V3 manuscript, or a claim that all original proof text has been reproduced. In particular, the quantitative ridge construction is consumed as a stated deterministic result; its full proof remains in CAP §§2–5. The interval-arithmetic implementation remains in the SIDE24 package.

## A. Source contract and notation

All project sources below are at Math- commit `fa2e9909d8cca40d38b7dc2eadda6766bd0788e1`. Full paths, byte counts and SHA256 hashes, checked against the pinned Git objects, are recorded in [APPENDIX_SOURCES.json](APPENDIX_SOURCES.json); these are the same principal identities as [SOURCES.json](SOURCES.json). Each section states the exact portion it consumes.

| Label | Source and pinned Git blob |
|---|---|
| P | [Uniform matrix-cap parent](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md), `dfed3b8d318a3ab1950957f393307733a4bef3f2` |
| CAP | [Marked cylinder cap](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md), `0633aca3c2a2882b0de4399da0a75d64c2e6b2e1` |
| E1 | [Congruence erratum](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md), `213594d6ca6a86fb938110f4d166d9ce275a02d0` |
| E2 | [Borel Kac–Rice replacement v1.1](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/d1_section9_borel_repair_20260925/REPAIR.md), `fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a` |
| REC | [D1 reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md), `75da2597971510f843f8d90c743950cb8c177342` |
| R | [Bounded remainder](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md), `247b3ecf80bfbe896948d5d489b2d5842a81c481` |
| S24 | [SIDE24 coefficient](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/coefficients/side24_v1/PROOF.md), `44b66f04f89fcd87383b3603fa69f1feb64cdddd` |

P is always read with E1, E2 and REC §5. E1 corrects the applied congruence to `diag(r^(−1/2),1)`; W1 in REC corrects the mode of convergence; the cap radius includes `r<24/(4√2)`. The historical author-time headers do not erase the later scoped reviews, and those reviews do not acquire a larger scope through this assembly. [CROSSWALK.md](CROSSWALK.md) records that review history.

Throughout, `f` is the exact centered variance-one Gaussian field on `X=ℝ²/(24ℤ²)` with the covariance in manuscript M1. Constants are finite but unevaluated. Compact-mark constants may depend on compact `B⊂ℝ` and `K⊂(0,∞)`, with `k_-=inf K>0`; all-mark constants below may depend on the fixed field but not on `b,k,r,u`. The axial direction is `u∈S¹`; an orthogonal transverse direction supplies coordinates `(x,y)`. `A_i=f_yy(i)` denotes a scalar transverse Hessian; `𝒜_r` denotes an intensity amplitude. `Q` is continuous Gaussian regression at the two heights and four zero gradient coordinates. The full normalizer is always

```text
W_r=|det H_M det H_S| 1{H_M<0, index(H_S)=1},
Z_r=E_Q W_r,                         dQ^W=(W_r/Z_r)dQ.
```

No density or Jacobian is hidden in `Z_r`.

## B. Contact regression and the full normalizer

**Sources: P §§2–5, with E1 and REC W1.** Put `a=−r/2`, `c=r/2`, and impose `f(a)=b`, `f(c)=b−kr³`. Replace the original observation vector `(f(a),f_x(a),f(c),f_x(c),f_y(a),f_y(c))` by

```text
U₀=(f(a)+f(c))/2,              U₁=(f(c)−f(a))/r,
U₂=(f_x(c)−f_x(a))/r,
U₃=(6/r²)[f_x(a)+f_x(c)−2(f(c)−f(a))/r],
V₀=(f_y(a)+f_y(c))/2,          V₁=(f_y(c)−f_y(a))/r.
```

The exact transformation has absolute determinant `12r^(−5)` and exact target

```text
v_r=(b−kr³/2, −kr², 0, 12k, 0, 0).
```

Centered difference formulas converge to `(f,f_x,f_xx,f_xxx,f_y,f_xy)` at the midpoint. The positive Fourier spectrum gives full rank for this list and for the list with `f_yy` appended: a zero-variance linear combination would annihilate every Fourier mode, hence the corresponding distribution of derivative evaluations would vanish. Its coefficients must then vanish. Continuity in `r` and the orthonormal frame, followed by compactness of the frame space, gives a uniform covariance gap near `r=0`. This is an existential gap, not a numerical spectral enclosure. Gaussian regression therefore gives uniform finite moments of derivative suprema on compact marks and a nondegenerate limiting law for `A_M`.

Write the endpoint Hessian as

```text
H_i = [[rα_i, rβ_i], [rβ_i, A_i]].
```

The exact pins imply

```text
|α_i|, |β_i| ≤ M₃/2,        |A_S−A_M| ≤ rM₃,
|α_M+6k|, |α_S−6k| ≤ rM₄/2,
det H_i/r = α_i A_i − rβ_i².                         (B1)
```

Here `M_j` is the maximum absolute coordinate partial derivative of order `j` on the cap cylinder. These identities use integral remainders, not a simultaneous vector Rolle argument.

With `D=diag(r^(−1/2),1)`,

```text
D H_i D = [[α_i, √r β_i], [√r β_i, A_i]],
det(D H_i D)=det(H_i)/r.
```

Its difference from `diag(∓6k,A_M)` tends to zero in probability, while `A_M` converges in law to the contact variable `A₀`. Slutsky's theorem gives joint convergence in law to `diag(−6k,A₀)` and `diag(6k,A₀)`. Since `k>0` and `A₀` has a nondegenerate Gaussian density, these limiting matrices are nonsingular almost surely. The limiting type indicator is `1{A₀<0}`. Equation (B1) and derivative moments supply uniform integrability of `W_r/r²`, so

```text
Z_r/r² → z₀(b,k,u)=(6k)² E[A₀² 1{A₀<0}].             (B2)
```

The right-hand side is positive and continuous: a nondegenerate Gaussian assigns positive mass to every negative open interval. A compactness/subsequence argument makes the convergence uniform on `B×K×S¹`. Consequently, for sufficiently small `r`,

```text
0<z_*≤Z_r/r²≤z^*<∞.                                 (B3)
```

Nothing in this step bounds `z_*` uniformly over all positive `k` or all births.

## C. The separating cap and the planar weighted failure integral

**Sources: CAP §§1–5; P §§4,6–8; REC §5.** The deterministic input is as follows. For a `C⁴` function on the embedded cylinder `D=[−2r,2r]²` with the exact pins, let `λ=−f_yy(M)`. If

```text
λ > [4/(3k)] rM₃²,                 rM₄≤3k/10,          (C1)
```

CAP supplies a transverse-critical ridge and the closed cap `C=[−2r,r/2]×[−2r,2r]`. Throughout `C`, `f≤b`; its boundary has value at most `s=b−kr³`, with equality on the right face only at `S`. The ridge continues from `M` through `S` to a point of height greater than `b`, with minimum exactly `s`. These are the precise outputs consumed from CAP's ridge derivative and all-face boundary estimates. CAP's separate historical fixed-axis probability bound is not consumed.

Every path from `M` to a point above birth must leave `C`, so its minimum is at most `s`. The ridge path shows the reverse inequality for the maximin. Thus `d_f(M)=s`. A Morse function with distinct critical values has a unique saddle at this death value; it is `S`. The older endpoint rules out an essential class. The chart condition is `2√2r<24/2`, explaining the stated embedding restriction.

To estimate failure under the tilted law, put `h=M₃`. On maximum support, `A_M=−λ<0`, and the negative Schur complement gives the canceled-pivot bound `|det H_M|≤rhλ/2`. Together with (B1) at `S` and `|A_S+λ|≤rh`, this yields

```text
W_r ≤ (r²h²/4) λ(λ+3rh/2).                           (C2)
```

Regress the full field once more on `A_M`. The centered residual `g_r` is independent of that scalar; `J=1+||g_r||_(C⁴)` has every finite moment uniformly bounded on compact marks. The regression coefficients give `h,M₄≤K₀(J+λ)`. The scalar conditional density is bounded and has a Gaussian tail.

Depth failure in (C1) is contained in `λ≤Dr(J+λ)²`, with `D` uniform on compact marks. Since `(J+λ)²≤2J²+2λ²`, this implies the union

```text
λ≤4DrJ²                 or                 λ>1/(4Dr). (C3)
```

On the first branch, `h≤CJ²`. Independence allows integration in `λ` before averaging `J`. Retaining both soft factors in (C2),

```text
E_Q[W_r; near branch]
 ≤ Cr² E[J⁴ ∫₀^(4DrJ²) λ(λ+CrJ²)dλ]
 ≤ Cr⁵ E[J¹⁰] ≤ C'r⁵.                               (C4)
```

The second branch is not omitted. Weighted Markov and the available joint moments give

```text
E_Q[(W_r/r²); λ>1/(4Dr)]
 ≤ (4Dr)⁴ E_Q[(W_r/r²)λ⁴] ≤ Cr⁴.
```

Likewise `E_Q[(W_r/r²); rM₄>3k_-/10]≤Cr⁴`. Multiplying these two bounds by `r²` and combining with (C4) gives a failure numerator `O(r⁵)+O(r⁶)`. Division by the full floor (B3) proves

```text
1−p_r ≤ Q^W(G_r^c) ≤ Cr³.                            (C5)
```

There is no Cauchy–Schwarz step losing a square root, and no assumption of independence between derivative maxima and the endpoint Hessian. Independence is used only for the specifically regressed residual.

## D. Genericity, the Borel mark and the pair measure

**Sources: P §8; E2 replacement §§9.1–9.4.** The pinned genericity argument is made at each fixed positive `r,b,k,u`. Away from the pins, the Gaussian maps detecting a degenerate critical point or an equal critical value have more output coordinates than parameters and nondegenerate finite-dimensional covariance on compact separated domains. A mesh argument with a bounded derivative norm makes their zero probability vanish; countable exhaustion and then removal of the derivative bound prove almost-sure absence of such zeros. The two pinned Hessians have densities and the pinned heights differ. Hence `Q`, and by absolute continuity `Q^W`, are almost surely Morse with distinct critical values. No common probability-one set over uncountably many regression laws is asserted.

For fixed or variable `M`, the strict event `d_f(M)>h` is described by countably many polygonal paths with rational interior vertices in torus charts, strict height clearance above `h`, and an endpoint higher than `f(M)`. Continuous paths with strict clearance can be approximated by such paths. Equality `d_f(M)=f(S)` is consequently a Borel mark on the generic locus; the essential convention is `d_f(M)=−∞`.

Let `D₀` be a compact ordered-pair domain separated from the spatial diagonal and put `G(x,y)=(∇f(x),∇f(y))`. Define two measures on `D₀×C²(X)`:

```text
μ(E)=E Σ[t∈D₀:G(t)=0] 1_E(t,f),
η(E)=∫[D₀] p_(G(t))(0) E[|det H_x det H_y|1_E(t,f) | G(t)=0]dt.
```

E2 uses the unweighted Gaussian Kac–Rice theorem to make both finite. The conditional field is the regular regression kernel

```text
f − Cov(f,G(t))Cov(G(t))^(−1)G(t)
  + Cov(f,G(t))Cov(G(t))^(−1)z.
```

Its law is continuous in the observation value, with the required local bounds. For bounded nonnegative continuous cylinder weights, the weighted theorem gives equality of the two integrals. Derivative evaluations through order two on a countable dense subset of `X`, together with location coordinates, generate the Borel sigma-algebra. The class of bounded functions with equal integrals is a vector space containing constants and closed under bounded pointwise limits. The functional monotone-class theorem therefore gives `μ=η` for every bounded Borel mark.

This reproduces E2's separate measure-extension argument; it does not call the elder indicator lower semicontinuous. The external theorem numbers and their applicability are checked in [LITERATURE.md](LITERATURE.md), entry L1. Height disintegration then gives full-pin density times `Z_r` for candidates and times `Z_r p_r` for elder pairs. The underlying measure identity, rather than arbitrary height-null-set choices, is what is used. Exhaustion toward the diagonal follows by nonnegative convergence with the integrability proved next.

## E. Exact lifetime pushforward and unrestricted leading term

**Sources: P §§10–15 and REC §§6–7.** The midpoint/directed-separation map has Jacobian one. Polar separation contributes `r dr dσ(u)`; `(b,k)↦(b,b−kr³)` contributes `r³`; the observation change contributes `12r^(−5)`; and `Z_r` contributes the remaining `r²`. Thus the per-area candidate pair measure is

```text
r𝒜_r db dk dr dσ(u),           𝒜_r=12π_r(v_r)Z_r/r².   (E1)
```

The selected measure has the additional factor `p_r`. Here `σ(S¹)=2π`; roles are ordered, so there is no half-factor. Transverse-frame changes leave the original observations and determinant weight unchanged, making the intensity a function of `u` alone without assuming isotropy.

For `ℓ=kr³`, `r dr/dℓ=ℓ^(−1/3)/(3k^(2/3))`. On compact positive-length `B,K`, whenever `ℓ<k_-r_*³`,

```text
ν_cand^(B,K)(ℓ)=ℓ^(−1/3)∫[B×K×S¹] 𝒜_(ℓ/k)^(1/3)/(3k^(2/3)),
c_(B,K)=∫[B×K×S¹] 𝒜₀/(3k^(2/3)).                    (E2)
```

The measures in these integrals are `db dk dσ`. Uniform contact convergence and (C5) give the same positive leading coefficient for selected pairs. Their difference is bounded by `Cℓ^(2/3)∫k^(−5/3)`, finite on `K`. This is a compact population statement.

For all marks, choose one small `r₀≤1` from the covariance gap, independent of `b,k`. The target has `|v_r|²≥c(b²+k²)`. Gaussian regression gives polynomial target growth of conditional derivative moments. Without dividing by `k` or `Z_r`, P (13.4) gives

```text
0≤𝒜_r≤H(b,k)=C(1+|b|+k)^4 exp[−c(b²+k²)].            (E3)
```

The function `Hk^(−2/3)` is integrable. At fixed lifetime the near region is exactly `k≥ℓ/r₀³`, so the scaled near density is

```text
ℓ^(1/3)ν_cand^near(ℓ)
 =∫ 1{k≥ℓ/r₀³}𝒜_(ℓ/k)^(1/3)/(3k^(2/3)).            (E4)
```

The elder integrand has factor `p_r≤1`. Pointwise in every fixed positive `k`, compact selection gives `p_r→1`; (E3) supplies domination for both populations. Their common limit is

```text
c₂,₂₄=∫[ℝ×(0,∞)×S¹] 𝒜₀/(3k^(2/3)).                (E5)
```

On `dist(x,y)≥r₀`, the original full-pin covariance is uniformly positive. At heights `(b,b−ℓ)`, `0<ℓ≤1`, its density is at most `Ce^(−cb²)` and conditional determinant moments grow at most polynomially in `b`. The height Jacobian is one here. Integrating the compact separation domain gives `0≤ν_eld^far≤ν_cand^far≤C`. Adding near and far proves the leading law. Almost-sure Morse theory identifies the full selected pair count with exactly the finite ordinary H₀ bar count.

## F. Derivation of the bounded unrestricted remainder

**Sources: R §§2–7, equations (R2)–(R18), and P (14.1).** This section retains the mark cutoffs that are essential to R; it does not assign a rate to the dominated convergence in §E.

The centered observations satisfy `U_r−U₀=O(r²)` in every finite moment and the corresponding covariance differences are `O(r²)`, uniformly in frame. In particular the fourth axial row is `f_xxx+(r²/40)f_xxxxx+O(r⁴)`. Couple the regression laws through one unconditioned field `F`:

```text
F_r=F+C_rΣ_r^(−1)(v_r−U_r),
F₀=F+C₀Σ₀^(−1)(v₀−U₀).
```

Covariance inversion and the exact targets imply, for finite `p,q`,

```text
||F_r−F₀||_(L^p(C^q)) ≤ Cr²P,            P=1+|b|+k,
|π_r(v_r)−π₀(v₀)|≤Cr²P²exp[−c(b²+k²)].              (F1)
```

The latter bound follows by interpolation of covariance and target; the `12k` coordinate retains coercivity along the interpolation.

For a symmetric matrix define `F_j(H)=|det H|1{index(H)=j}`, zero at singular matrices. The determinant telescoping inequality bounds changes in `F_j` by a polynomial norm factor times the matrix perturbation: when inertia differs, a singular matrix on the connecting segment bounds the one contributing endpoint. More strongly, for the planar block

```text
K_t=[[α,√t β],[√t β,A]],             det K_t=αA−tβ²,
|F_j(K_r)−F_j(K₀)|≤rβ².                              (F2)
```

If the contributing inertia changes, the determinant vanishes at an intermediate `t`; its affine form proves the same bound. Thus the `√r` off-diagonal entries cost `r`, even at `A=0`.

Apply (F2) to each congruence-scaled endpoint, then use (B1) and (F1) to compare its diagonal block with `diag(∓6k,A₀)`. A random envelope `T` has `E T^p≤C_pP^p`, endpoint error `O(rT²)` and determinant size `O((k+r)T²)`. Multiplication, expectation and the density perturbation in (F1) give, with a polynomial-Gaussian `H` of possibly larger degree,

```text
𝒜_r≤(k+r)²H,       𝒜₀≤k²H,
|𝒜_r−𝒜₀|≤r(k+r)H.                                  (F3)
```

For selection define `ℬ_r=12π_r E_Q[(W_r/r²)1{G_r^c}]`. Always `𝒜_r(1−p_r)≤ℬ_r≤𝒜_r`. To keep all-mark constants controlled, regress on the appended vector `(U_r,A_M)`. Its joint density obeys

```text
π_r(v_r) density(A_M=A | U_r=v_r)
 ≤ Cexp[−c(b²+k²+A²)].                               (F4)
```

The remaining residual is independent of this vector. Put `δ=r/k`; the scalar depth split is now `λ≤4DδT²` or `λ>1/(4Dδ)`, with `T=P+J`. Integration of the same two soft factors gives terms `δ³` and `rδ²=kδ³`, the latter absorbed into the polynomial in `P`. The far branch and fourth-derivative exception cost `δ⁴≤δ³` when `δ≤1`. This proves the unnormalized bound

```text
ℬ_r≤δ³H(b,k)                    when δ≤1.            (F5)
```

No global lower normalizer is used in (F3)–(F5).

Set `a=ℓ/r₀³`. The omitted contact mass below `a` is at most `C∫₀^a k^(4/3)dk=Ca^(7/3)`. For the remaining candidate difference, (F3) yields the exact monomials

```text
k^(−2/3)rk=ℓ^(1/3),
k^(−2/3)r²=ℓ^(2/3)k^(−4/3).
```

After integrating `b,u`, the first costs `Cℓ^(1/3)`. The second costs at most `Cℓ^(2/3)(a^(−1/3)+1)=O(ℓ^(1/3))`. The contact omission is smaller. Therefore `|ℓ^(1/3)ν_cand^near−c₂,₂₄|≤Cℓ^(1/3)`.

For loss, set `η=ℓ^(1/6)` and take `ℓ` small enough that `a<η<1`. In `a≤k≤η`, use `ℬ_r≤𝒜_r≤2(k²+r²)H`, giving scaled loss

```text
C[η^(7/3)+ℓ^(2/3)a^(−1/3)]
 =O(ℓ^(7/18))+O(ℓ^(1/3)).
```

For `k≥η`, `δ=ℓ^(1/3)k^(−4/3)≤ℓ^(1/9)≤1`, so (F5) applies. Its scaled integral is bounded by `Cℓ∫_η^1 k^(−14/3)dk` plus a Gaussian tail, hence `O(ℓη^(−11/3))=O(ℓ^(7/18))`. Since `7/18>1/3`, division by `ℓ^(1/3)` proves bounded near selection loss. Adding the bounded far densities from §E gives R's three inequalities

```text
|ν_cand−c₂,₂₄ℓ^(−1/3)|≤C,
0≤ν_cand−ν_eld≤C,
|ν_eld−c₂,₂₄ℓ^(−1/3)|≤C.                             (F6)
```

The lower cutoff `a` is indispensable: extending the `k^(−4/3)` integral to zero makes it diverge. Neither (F6) nor its proof evaluates the constants or makes the unrestricted rejected density tend to zero.

## G. The coefficient, arithmetic scope and integrated observable

**Sources: P §15; S24 §§1–5; R §8.** Let `G=∇f`, `V_u=H_fu`, `t_u=∂_u³f`, and `A_u` be the transverse scalar. Put `τ_u²=Var(t_u|G=0)` and `D_u=E[A_u²1{A_u<0}|V_u=0]`. Odd/even covariance parity splits the contact density. Birth disintegration in (E5) gives `p_(V_u)(0)D_u`; the scalar gap integral is

```text
144∫₀∞ k^(4/3)φ_τ(12k)dk
 = Γ(7/6)τ^(4/3)/(24^(1/3)√π).
```

For example, substitution `t=12k` contributes `144·12^(−7/3)`; the positive-half normal moment contributes `2^(−1/3)τ^(4/3)Γ(7/6)/√π`. Their product is the displayed factor. Thus

```text
c₂,₂₄=Γ(7/6)/(24^(1/3)√π)
       ∫[S¹] p_G(0)p_(V_u)(0)τ_u^(4/3)D_u dσ(u).      (G1)
```

The prefactor's integer 24 comes from the gamma/Jacobian calculation, not from substituting the torus side into a variable. All jet covariances in (G1) remain those of the exact periodic field.

S24 computes the reference covariance for `exp(−|z|²/2)`: `τ²=6`, `p_G(0)p_V(0)=(2π)^(−2)/√3`, and the conditional scalar variance is `8/3`, giving `D=4/3`. Its all-direction derivative bound through order six controls the full omitted periodic image sum. Multiplicative covariance comparison passes to conditional Schur complements; Gaussian density bounds control the cone expectation. It proves `|c₂,₂₄/c₂,ref−1|<10^(−106)`. The outward arithmetic then encloses (G1) between `0.07340691930603427103` and `0.07340691930603427104`. The interval implementation and special-function remainder proofs are imported from S24, not independently rerun or re-certified by this appendix.

Finally, integration of (F6) gives the per-area expected bin mass, for `0≤a<b≤ℓ_*`,

```text
μ_eld((a,b])=(3/2)c₂,₂₄(b^(2/3)−a^(2/3))+O(b−a).      (G2)
```

The same error constant works throughout this window because the density remainder is bounded there. Equation (G2) is an exact consequence of the source remainder, but `ℓ_*` and its error constant are still unknown. Its use as a visual reference curve for a discrete experiment does not certify that experiment's finite bins as asymptotic.

## H. What is complete here and what remains imported

The appendix now provides the scalar compact failure integral, the corrected contact-normalizer argument, the Borel measure-extension mechanism, the complete mark/radius integration yielding the exponent and bounded remainder, and the coefficient's normalization in reading order. It preserves the exact global elder event and the full normalizer throughout.

The load-bearing quantitative cap/ridge theorem and its constants are explicitly imported from CAP §§1–5; the Fourier/regression estimates are assembled from P and R rather than newly independently reviewed; the S24 interval algorithm remains an external appendix candidate. A publication version must decide whether to reproduce those full source proofs or include stable, complete supplementary texts. This draft adds no theorem acceptance, no numerical lifetime window, no continuum/grid error certificate and no human referee verdict. It consumes none of the live Math- #188/#190/#191 candidates or unseen V3 text. The assembly has same-provider author exposure and zero organizational-independence credit.
