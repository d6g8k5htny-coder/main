# Shrinking witness pairs at fixed distance from the pins

**Object:** RN-SHRINKING-WITNESS-M2-20260926.
**Disposition:** additive reconnaissance for the closed fixed-η interface. It does not edit `frontiers/remote_window_20260924/PROOF.md`, does not change the seven fixed-ρ / fixed-η acceptances, and does not accept a sharp collision kernel, a pin-neighborhood estimate, an intermediate-annulus bridge, elder pairing, or a global RN / 24-jet theorem.

Reviewed outer boundary: Math- `frontiers/remote_window_20260924/PROOF.md` at `191ea7d541a486736ba7bbddfd4eac25a6c4567b`, 18355 bytes, SHA256 `a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7`. The accepted pair statement is (15)–(16) there: for each fixed `m≥2` and fixed `η>0`, one endpoint weight and one endpoint normalizer give

```text
E[T_{r,j,m,η}] = (k r^3)^m ∫_{D_{ρ,η,m}} Λ_{j,m} + O(r (k r^3)^m),
P(an unordered η-separated m-set) ≤ E[T]/m!.
```

For `m=2` the fixed-gap probability is `O(r^6)`. This note starts exactly where that compactness stops: both witnesses stay in the fixed remote region `D_ρ`, while their mutual separation `η` is allowed to tend to zero with `r`.

## 1. Divided-difference frame

Fix `0<δ_*≤ρ/4` and `0<r_*≤ρ`. For `r≤r_*` every point of `D_ρ` is at least `ρ/2` from both pins. On the pair region

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

The contact limit is `(∇f(x), H_x u)`. Those `2d` functionals are linearly independent: the gradient functionals have order one, and `H ↦ H u` is a surjective family of order-two functionals. Together with the pin jet `U_0` at a site at least `ρ/2` away, the reviewed Fourier argument applies. The covariance is continuous on the compact set `D_ρ × S^{d-1}`, so its smallest eigenvalue has a positive lower bound `c_*`.

The Taylor remainder of the smooth field gives `G_1 - H_x u = O(δ)` in every fixed `L^p`, uniformly for `x∈D_ρ` and `|u|=1`. For `δ_*` small and `r≤r_*`, the conditional covariance of `(G_0,G_1)` given the endpoint pins `U_r=v_r` stays bounded below by `c_*/2`. The conditional density of `(G_0,G_1)` at the origin is consequently bounded by a constant that depends on `d,L,ρ,δ_*,B,K` and not on `δ` or on `η`. Hence

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

and the same expansion at `y` in the direction `-u` gives `||H_y u||`. The third- and fourth-derivative factors have conditional `L^p` norms bounded uniformly in the pair location, the direction, and the height target in the compact mark set. Gaussian regression supplies those bounds: the observation covariance is uniformly positive definite, the targets `(v_r,0,0,t)` stay bounded, and conditioning does not increase variances of fixed jets. Therefore

```text
E[ |det H_x|^p |det H_y|^p | U_r, ∇f(x)=∇f(y)=0, f(x)=t ]^{1/p} ≤ C_p δ^2.
```

The endpoint weight uses the reviewed axial identities, which are consequences of the pins alone: `α_M=-6k+O(r M_4)` and `α_S=6k+O(r M_4)`. The fourth-derivative moments remain uniform under the extra divided-difference and height conditioning by the same regression bound. The filtered-determinant comparison of the reviewed note, (8)–(10) there, then yields

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
f(y)-f(x) = -μ δ^3/12 - ν δ^4/24 - ρ δ^5/80
```

when the axial derivative at the first point is chosen so that it also vanishes at distance `δ`. The conditional height gap is therefore of order `δ^3`. Once `δ^3` is smaller than the window `k r^3`, the second height is fixed by the first and does not produce another factor `r^3`.

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

These figures are diagnostics for that model field. They are not values of `Λ_{j,2}` for the periodized pinned law, and the bound in Section 3 does not use them. They do show why an attempt to keep the fixed-η order `O(r^6)` fails once `η(r)→0`: after the two close critical points share one height window, the separation integral converges to a positive constant rather than to another factor `r^3`.

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

The fixed-η statement (15)–(16) remains the outer boundary for separations bounded below by a positive constant independent of `r`. Its `O(r^6)` pair probability is not claimed, and is not true on the strength of this note, once that constant is allowed to shrink.

## 7. Checks

`tests/test_d5_shrinking_witness.py` locks the rational identities used above: the Jacobian `δ^d`, the singular-value determinant comparison, the axial expansions and the height-gap coefficients `-1/12`, `-1/24`, and `-1/80`, the same-sign fourth-derivative window, the radial integral `(δ_*^2-η^2)/2`, and the Bargmann–Fock conditional variance `6`. It does not sample the periodized field and does not evaluate the pinned contact kernel.
