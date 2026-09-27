# Shrinking witness pairs at fixed distance from the pins

**Object:** RN-SHRINKING-WITNESS-M2-20260926.
**Disposition:** additive reconnaissance for the closed fixed-η interface. It does not edit `frontiers/remote_window_20260924/PROOF.md`, does not change the seven fixed-ρ / fixed-η acceptances, and does not accept a sharp collision kernel, a pin-neighborhood estimate, an intermediate-annulus bridge, elder pairing, or a global RN / 24-jet theorem.

**Amendment note (2026-09-27).** Review comment [5850462018](https://github.com/d6g8k5htny-coder/main/pull/125#issuecomment-5850462018) on head `00c9d08a` returned AMEND with four items. This revision displays the uniform moment input as Lemma 1 with its complete compact parameter set (item 1), rewords the two sentences that overstated what is shown about the shrinking-pair order (item 2), restores the `#76` route row in `docs/RESEARCH_INDEX.md` and adds a separate row (item 3), renames the fifth axial derivative to `ω`, makes the Bargmann–Fock check rational, fixes the `δ_*` wording, and states the scope of the lock tests (item 4). The §3 bound and its proof are otherwise unchanged. Scientific effect of the amendment: NONE.

Reviewed outer boundary: Math- `frontiers/remote_window_20260924/PROOF.md` at `191ea7d541a486736ba7bbddfd4eac25a6c4567b`, 18355 bytes, SHA256 `a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7`. The accepted pair statement is (15)–(16) there: for each fixed `m≥2` and fixed `η>0`, one endpoint weight and one endpoint normalizer give

```text
E[T_{r,j,m,η}] = (k r^3)^m ∫_{D_{ρ,η,m}} Λ_{j,m} + O(r (k r^3)^m),
P(an unordered η-separated m-set) ≤ E[T]/m!.
```

For `m=2` the fixed-gap probability is `O(r^6)`. This note starts exactly where that compactness stops: both witnesses stay in the fixed remote region `D_ρ`, while their mutual separation `η` is allowed to tend to zero with `r`.

## 1. Divided-difference frame

Choose `δ_*∈(0,ρ/4]` and `r_*∈(0,ρ]` small enough for Lemma 1 below, then keep them fixed for the rest of the note; separations in `[δ_*,ρ/4]` are covered by the accepted formula (15) with `η=δ_*`. For `r≤r_*` every point of `D_ρ` is at least `ρ/2` from both pins. On the pair region

```text
0 < |x-y| ≤ δ_*,    x,y ∈ D_ρ,
```

write `δ=|y-x|`, `u=(y-x)/δ`, and

```text
G_0 = ∇f(x),    G_1 = δ^{-1}(∇f(y)-∇f(x)).
```

The linear map `(G_0,G_1) ↦ (∇f(x), ∇f(y)) = (G_0, G_0+δ G_1)` has Jacobian determinant `δ^d`. Therefore, wherever the law is nondegenerate,

```text
p_{∇f(x),∇f(y)}(0,0) = δ^{-d} p_{G_0,G_1}(0,0).
```

The contact limit is `(∇f(x), H_x u)`. Those `2d` functionals are linearly independent: the gradient functionals have order one, and `H ↦ H u` is a surjective family of order-two functionals. Together with the pin jet `U_0` at a site at least `ρ/2` away, the reviewed Fourier argument applies.

**Lemma 1 (uniform conditional moments).** Fix an integer `q≥4`, an exponent `p<∞`, and a compact height set `T` containing a neighborhood of the birth interval. Let

```text
Θ = { θ=(r,x,u,δ,t) : 0<r≤r_*, x∈D_ρ, |u|=1, 0<δ≤δ_*, x+δu∈D_ρ, t∈T },
```

and let `O_θ=(U_r, G_0, G_1, f(x))` be the observation vector, with `y=x+δu`. Then, after `δ_*` and `r_*` are taken small enough,

1. the covariance `Σ_obs(θ)` of `O_θ` satisfies `λ_min(Σ_obs(θ)) ≥ c_*/2` for every `θ∈Θ`, with `c_*>0` depending on `d,L,ρ,B,K` only, and
2. for the field conditioned on `O_θ=(v_r,0,0,t)` with `v_r` in the fixed compact endpoint target set,

```text
sup_{θ∈Θ} E[ (1+||f||_{C^q(D_ρ)})^p | O_θ=(v_r,0,0,t) ] ≤ C(d,L,ρ,δ_*,q,p,B,K) < ∞.
```

*Proof.* Every entry of `Σ_obs(θ)`, and every cross-covariance `Cov(D^a f(z), O_θ)` for `|a|≤q` and `z∈D_ρ`, is a smooth function of the sites because the kernel is smooth; the divided difference has the continuous extension `Cov(D^a f(z), G_1) = δ^{-1}(Cov(D^a f(z),∇f(y)) - Cov(D^a f(z),∇f(x))) → Cov(D^a f(z), H_x u)` as `δ→0`, and `U_r→U_0` as `r→0`. So `Σ_obs` and the cross-covariances extend continuously to the compact closure `\bar Θ` (where `δ=0` means `G_1=H_x u` and `r=0` means `U_r=U_0`). On `\bar Θ` the extended observation consists of the pin jet `U_0` at sites at least `ρ/2` away together with `(∇f(x), H_x u, f(x))`, which are `2d+1` linearly independent functionals at `x`; by the reviewed Fourier positivity argument (PROOF.md, the nondegeneracy step for (5)), its covariance is positive definite at every point of `\bar Θ`, hence bounded below by some `c_*>0` on `\bar Θ` by compactness and continuity. Shrinking `δ_*` and `r_*` makes the finite-parameter covariance at least `c_*/2`, which is item 1.

For item 2, the conditional law is Gaussian with mean `m_θ(z) = Cov(f(z),O_θ) Σ_obs(θ)^{-1} (v_r,0,0,t)` and covariance `K_θ(z,w) = K(z,w) - Cov(f(z),O_θ) Σ_obs(θ)^{-1} Cov(O_θ,f(w))`. Since `||Σ_obs(θ)^{-1}|| ≤ 2/c_*`, the target is bounded, and the cross-covariances are bounded in `C^q` uniformly on `\bar Θ`, the conditional mean is bounded in `C^q(D_ρ)` uniformly over `Θ`. The conditional covariance operator is dominated by the unconditional one, so for every fixed derivative `Var(D^a f(z) | O_θ) ≤ Var(D^a f(z))`, and the centered conditional field is a smooth Gaussian process on `D_ρ` whose canonical metric is dominated by the unconditional metric. Sobolev embedding on the smooth compact region `D_ρ` gives `||g||_{C^q} ≤ C ||g||_{W^{q+d,2}}`, the second moment of that Sobolev norm is a finite sum of pointwise variances integrated over `D_ρ`, each at most its unconditional value, and Fernique's theorem (equivalently Borell–TIS for the supremum of each derivative) turns the uniform second-moment bound into a uniform `p`-th moment bound. Adding the bounded mean gives the display. ∎

Lemma 1 is the whole uniform input used below; the reviewed estimates (8)–(10) are transferred to the enlarged conditioning only through it. In particular the conditional density of `(G_0,G_1)` at the origin, given `U_r=v_r`, is bounded by a constant that depends on `d,L,ρ,δ_*,B,K` and not on `δ` or on `η`. Hence

```text
p_{∇f(x),∇f(y) | U_r}(0,0) ≤ C δ^{-d}.
```

The same enlarged observation `(U_r, G_0, G_1, f(x))` remains uniformly nondegenerate. Its contact limit adjoins the value `f(x)` to `(U_0, ∇f(x), H_x u)`. The Bargmann–Fock model makes the mechanism explicit: the covariance of `(f_x, f_y, f_{xx}, f_{xy})` is `diag(1,1,3,1)`, and the conditional variance of `∂_u^3 f` given `∇f=0` and `H u=0` equals `6`. The periodized field uses the same jet-independence argument rather than this model covariance. Conditioning on a height in the compact mark interval therefore does not collapse the divided-difference covariance.

## 2. Determinant and endpoint scaling

Let `|u|=1`. The smallest singular value of a symmetric matrix satisfies `σ_min(H) ≤ ||H u||`, so

```text
|det H| ≤ ||H u|| ||H||_F^{d-1}.
```

On the conditional field with `G_1=0`,

```text
H_x u = -(δ/2) ∇_u(H_x) u + O(δ^2),
```

and the same expansion at `y` in the direction `-u` gives `||H_y u||`. The third- and fourth-derivative factors have conditional `L^p` norms bounded uniformly over the parameter set `Θ` of Lemma 1: pair location, direction, separation `δ`, pin distance `r`, and height target in `T`. Lemma 1(2) with `q=4` is exactly that statement. Therefore

```text
E[ |det H_x|^p |det H_y|^p | U_r, ∇f(x)=∇f(y)=0, f(x)=t ]^{1/p} ≤ C_p δ^2.
```

The endpoint weight uses the reviewed axial identities, which are consequences of the pins alone: `α_M=-6k+O(r M_4)` and `α_S=6k+O(r M_4)`. The fourth-derivative moments remain uniform under the extra divided-difference and height conditioning by Lemma 1(2). The filtered-determinant comparison of the reviewed note, (8)–(10) there, then yields

```text
E[(W_r/r^2)^p | U_r, ∇f(x)=∇f(y)=0, f(x)=t] ≤ C_p.
```

Hölder’s inequality gives the product bound

```text
E[ W_r F_j(H_x) F_j(H_y) | U_r, ∇f(x)=∇f(y)=0, f(x)=t ] ≤ C r^2 δ^2,
```

uniformly for `t` in a fixed neighborhood of the compact birth interval. The index filter is estimated by `F_j ≤ |det|`. One endpoint weight appears, once.

The full normalizer stays the endpoint-only `Z_r`. The accepted limit `Z_r/r^2=z_0+O(r)` with `inf z_0>0` is not recomputed under the remote pair conditioning. For small `r`, `Z_r ≥ c r^2`.

## 3. Expected-count bound

Let `I=(b-k r^3, b)` and let `T_{r,j,2}(η)` be the ordered number of index-`j` pairs in `D_ρ` whose heights lie in `I` and whose separation lies in `[η,δ_*]`. For each fixed `η>0` the two sites are distinct, so the conditional gradient covariance is positive definite and the marked Kac–Rice formula used for (15) applies:

```text
E[T_{r,j,2}(η)]
 = Z_r^{-1} ∫_{η≤|x-y|≤δ_*}
   p_{∇f(x),∇f(y)|U_r}(0,0)
   E[ W_r F_j(H_x) F_j(H_y) 1_{f(x)∈I} 1_{f(y)∈I}
      | U_r, ∇f(x)=∇f(y)=0 ] dx dy.
```

Drop the second height indicator and disintegrate the first:

```text
E[ W_r F_j(H_x) F_j(H_y) 1_{f(x)∈I} | gradients zero ]
 ≤ ∫_I p_{f(x)|gradients, pins}(t)
   E[W_r F_j(H_x) F_j(H_y) | gradients zero, f(x)=t] dt.
```

Section 1 bounds the marginal conditional density of `f(x)`, so the integral over `I` contributes at most `C k r^3`. Section 2 bounds the integrand by `C r^2 δ^2`. The gradient density contributes `δ^{-d}`. Division by `Z_r` cancels the endpoint factor `r^2`. The resulting integrand is at most

```text
C k r^3 \, δ^{2-d}.
```

In polar coordinates on the separation,

```text
∫_{η≤|h|≤δ_*} |h|^{2-d} dh
 = |S^{d-1}| ∫_η^{δ_*} s^{2-d} s^{d-1} ds
 = |S^{d-1}| ∫_η^{δ_*} s \, ds
 = |S^{d-1}| (δ_*^2-η^2)/2.
```

The exponent equals `1` for every `d≥2`, so the integral remains bounded as `η↓0`. Integrating over `x∈D_ρ` gives a constant multiple of `|D_ρ|`. Pairs with separation greater than `δ_*` are estimated by the accepted formula (15), which is `O(r^6)`.

**Bound.** There exist `r_*>0` and `C<∞`, depending on `d,L,ρ,δ_*,B,K` but not on `η`, such that for every `r≤r_*`, every mark in the fixed compact sets, every frame, every index `j`, and every `η∈(0,δ_*]`,

```text
E_{Q_r^W}[T_{r,j,2}(η)] ≤ C k r^3.
```

The same bound holds for the improper integral over `0<|x-y|≤δ_*`. Markov’s inequality gives

```text
P(an unordered pair of index j in the window, separation at least η)
 ≤ E[T_{r,j,2}(η)] / 2
 ≤ C k r^3.
```

The unnormalized numerator, before division by `Z_r`, is `O(k r^5)`. That is the same power as the one-point numerator (14), with one endpoint product `W_r` and one height window. A second independent window factor `(k r^3)` is available only while `η` stays fixed, because only then is `f(y)-f(x)` of order one under the critical-point conditioning. Along a segment with both axial derivatives zero, the exact degree-four expansion gives

```text
f(y)-f(x) = -μ δ^3/12 - ν δ^4/24 - ω δ^5/80
```

when the axial derivative at the first point is chosen so that it also vanishes at distance `δ`; here `μ,ν,ω` are the third, fourth and fifth axial derivatives at `x` (`ω` is not the remote radius `ρ`). The conditional height gap is therefore of order `δ^3`. Once `δ^3` is smaller than the window `k r^3`, the second height indicator is no longer an independent factor `k r^3`. This note discards that indicator rather than evaluating it, so the `r^3` above is an artifact of the discarded window and is not the true order of the shrinking-pair count.

## 4. The fixed-η kernel does not extend by sending η to zero

Formula (15) cannot be passed to the diagonal by dominated convergence. Three separate mechanisms stop it.

1. **Covariance.** The raw joint covariance of `(∇f(x),∇f(y))` has smallest eigenvalue tending to zero as `δ→0`. Its density blows up as `δ^{-d}`. The constant in the fixed-η nondegeneracy statement depends on `η` and is not claimed to be uniform down to zero. The divided-difference coordinate is what restores a uniform spectral gap.

2. **Jacobian.** The factor `δ^{-d}` is the change-of-variable determinant above. Leaving it inside an `η`-independent kernel overstates the density by that Jacobian.

3. **Index.** The leading axial curvatures have opposite signs. If the axial derivative vanishes at both ends of the segment and its third derivative is `μ`, then

```text
α_x = -(δ/2) μ + O(δ^2),    α_y = +(δ/2) μ + O(δ^2).
```

Whenever `μ` stays away from zero and the transverse Hessian stays a definite distance from the singular set, the indices differ by one. Same-index pairs are a thinner subset: either the transverse curvature crosses zero between the two points, or `μ` itself is of order `δ` and a fourth derivative determines the sign. The positive fixed-η density `Λ_{j,2}` therefore has no continuous strictly positive extension to `δ=0`. The upper bound in Section 3 does not need that extension, because it uses `F_j≤|det|` and the integrable majorant `δ^{2-d}`.

A sharp asymptotic kernel, with a nonzero leading coefficient, needs one more normalized jet beyond `(G_0,G_1)`. The natural extra coordinate is the transverse curvature in the directions orthogonal to `u`, or equivalently the scaled third derivative `δ^{-1}(H_x u)` after the gradient divided difference has been set to zero. That coordinate measures the codimension-one crossing which carries the same-index mass. It is not required for the `O(k r^3)` bound.

## 5. Bargmann–Fock diagnostic, not an input

On the unpinned field with covariance `exp(-|z|^2/2)` in dimension two, direct sampling of the conditional 2-jet is consistent with the scaling above and is stricter for the same-index product. As `δ` ran through `1/2, 1/4, ..., 1/32`, the product `p(∇f(x),∇f(y)=0) δ^2` stayed near `1.5×10^{-2}`, and `E[|det H_x det H_y|]/δ^2` increased toward about `4`. The same-index truncated moment `E[|det H_x det H_y| 1_{equal index}]/δ^5` increased from about `1.2` at `δ=0.4` to about `2.6` at `δ=1/80`, with the index-`1` contribution dominant over indices `0` and `2`. A pure `δ^5` limit is not identified. The conditional standard deviation of `f(y)-f(x)` tracked `0.204 δ^3`.

These figures are diagnostics for that model field. They are not values of `Λ_{j,2}` for the periodized pinned law, and the bound in Section 3 does not use them. They illustrate why the method of this note does not reproduce the fixed-η order `O(r^6)` once `η(r)→0`: after the second height indicator is dropped, the separation integral converges to a positive constant rather than to another factor `r^3`. The true order for shrinking pairs is not established here. Retaining the second indicator with the scaled gap `δ^{-3}(f(y)-f(x))` is the route to a sharper order; that computation is not carried out in this note.

## 6. Complement

The bound covers ordered index-`j` pairs in the fixed region `D_ρ`, inside the between-pin height window, at every mutual separation down to zero. The probability statement is the Markov bound from that expectation.

Still open, and untouched by this note:

- the mesoscopic neighborhood `x=ry` of the pins;
- pin collision, a witness within `O(r)` of `M` or `S`;
- the intermediate annulus, including `r ≪ |x| ≪ ρ`;
- remote critical points with no shrinking height window;
- legacy all-cell, 24-jet, and fixed-`r` inner-wedge selectors;
- a matching lower bound, a numerical constant, and a Poisson or factorial-moment limit for separations tending to zero;
- elder selection and any global RN closure.

The fixed-η statement (15)–(16) remains the outer boundary for separations bounded below by a positive constant independent of `r`. Once that constant is allowed to shrink, the order of the pair probability is not established here: this note proves an upper bound `O(k r^3)` only, by dropping `1_{f(y)∈I}` and using `F_j ≤ |det|`, and says nothing about whether the true order is `r^3`, `r^6`, or in between.

## 7. Checks

`tests/test_d5_shrinking_witness.py` locks the rational identities used above: the Jacobian `δ^d`, the singular-value determinant comparison, the axial expansions and the height-gap coefficients `-1/12`, `-1/24`, and `-1/80`, the same-sign fourth-derivative window, the radial integral `(δ_*^2-η^2)/2`, and the Bargmann–Fock conditional variance `6`. These are locks on algebra that the tests themselves restate; they guard against transcription drift and cannot fail if the analytic argument of Sections 1–3 is wrong. The file does not sample the periodized field and does not evaluate the pinned contact kernel.
