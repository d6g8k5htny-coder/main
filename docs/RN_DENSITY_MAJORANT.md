# A conditional SIDE24 density–determinant majorant

For the normalized side-24 torus field, the fixed-axis pins at radius
`r=1/20` and birth `b=6/5`, the following explicit envelope replaces a
spatially unstable nine-pin Hessian calculation by two scalar premises.
It does **not** prove those premises on any region.

Let `G=(f(M),fx(M),fy(M),f(S),fx(S),fy(S))`, with
`M=(-1/40,0)`, `S=(1/40,0)` and pinned values
`c=(b,0,0,b-ell,0,0)`, where `ell=1/48000`.
Let `Y=(f(y),fx(y),fy(y))` have conditional mean `mu` and covariance
`Sigma` given `G=c`. For every point of a region and every
`v in [b-ell,b]`, suppose, under this one common Gaussian law,

```
det(Sigma) >= D > 0,
q(v,y) = ((v,0,0)-mu)^T Sigma^-1 ((v,0,0)-mu) >= Q >= 0.
```

The covariance is the actual conditional covariance, hence positive
semidefinite; positive determinant makes it positive definite. Interval
entry ranges chosen independently are not an alternative Gaussian law.
Then the typed RN integrand obeys

```
0 <= I(y) <= ell * 512 * (8+Q)^3 * exp(-Q/2)
             / (Z_lo * (2*pi)^(3/2) * sqrt(D)),
Z_lo = 15518583475065571 / 2000000000000000000.
```

Here the integrand includes the product of the **three original Hessian
determinants**, divided by the typed two-pin normalization `Z`, and the
three-jet density integrated over the full height window. Any required
type/event indicators in this numerator take values in `[0,1]` and may be
dropped for this upper bound. This is an integrand upper bound; multiplying
by a certified region area or forming a valid partition is separate work.

## Imported source contracts

`density_majorant.source_binding` checks these original bytes on every
call. The non-executing `research.rn.n6_inputs.authenticated_inputs` first
checks current inventory, source context/access, coverage holds, exclusions,
and the N6 archive's repository dependency closure. Selected historical
inner members are also checked against current excluded hashes/paths before
their bodies are read. Archive paths with symlink ancestors are refused.
The returned receipt retains those live identities. All archived code is
read as data; no archived program is executed by this majorant checker.

* `research/campaigns/h3_rn_n6_20260920_v1.zip`, 571735 bytes,
  SHA-256 `73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21`.
  Its manifest, the energy proofs/checker/result, and N6 proof/implementation
  are individually size/hash pinned in the module. In particular
  `pin_energy/PROOF_v2.md` and `pin_energy/result-v2.json` establish the
  existing six-pin energy premise
  `c^T Cov(G)^-1 c <= 1266466/160083 < 8`.
  The source treats the normalized limiting pin basis at zero separately;
  this application uses the nonsingular actual law at `r=1/20`.
* The same archive's `rn_n6/side24_taylor.py:moment_cap_certificate` and
  `rn_n6/PROOF.md` give normalized torus derivative moment caps
  `m2 <= 2` and `m4 <= 4`, with the full image tail and positive normalizer.
  Consequently all original Hessian coordinates satisfy `Var(X)<=4`:
  pure derivatives have variance `m4`, the mixed derivative has variance
  `m2^2` by kernel separability. These are bounds, not planar identities.
  Briefly, the source bounds every paired image tail through order 18 by
  `2^-141/(1-2^-846) < 2^-140 < 1`, using `exp(1)>2`,
  Hermite coefficient sums `<2^56`, and `24^18<2^90`; division by the
  normalizer, which is at least one, preserves the caps. The constant
  image gives the usual integer Gaussian derivative moment, but the
  nonzero images are retained.
* `drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/`'s
  `RN5_REPAIR_AND_ERRATUM_BUNDLE.zip`, 140170 bytes,
  SHA-256 `28c1c385406452a706f11aea1646a87585b002841005d57febdf32222c368c6e`.
  The field definition and the fixed-r `H3_RUNG_FLOOR.md` are individually
  pinned. The latter supplies the imported premise `Z >= Z_lo` above.
  This work does not rerun or reprove that floor, and does not replace it
  by the weaker-at-this-radius whole-band floor.

The source manifest is custody, not a new mathematical proof or operator
acceptance. The source pinning cannot validate user-supplied `D,Q`.

## Regression energy and its variance tradeoff

The field is centered before conditioning. The six-pin span and the
residual three-jet span are orthogonal in Gaussian `L2`. The Schur
complement identity therefore makes the full nine-observation pin energy
exactly `E6+q`, with `E6<8` and `q>=0`.

For any one original Hessian coordinate `X`, write `s=Var(X)<=4`, and let
`a` be its explained variance after projecting onto the joint observation
span. Gaussian regression and Cauchy–Schwarz give, under the canonical
conditional Gaussian law at these exact pin values,

```
0 <= a <= s,       conditional variance = s-a,
conditional mean squared <= a*(E6+q) <= a*T,    T=8+q.
```

No positive residual variance is needed: deterministic conditional
coordinates are allowed. If `s=0`, the coordinate is identically zero.
Otherwise set `lambda=a/s` in `[0,1]`. The exact noncentral Gaussian sixth
moment and positivity of all its coefficients imply

```
E[X^6 | G=c,Y=(v,0,0)] / s^3
 <= 15(1-lambda)^3 + 45*T*lambda*(1-lambda)^2
    + 15*T^2*lambda^2*(1-lambda) + T^3*lambda^3.
```

This is a Bernstein cubic with control values
`(15,15*T,5*T^2,T^3)`. For `T>=8`, all are at most `T^3`.
For an exact polynomial check, `T^3` minus its first three controls,
in ascending powers of `q`, has coefficients respectively
`(497,192,24,1)`, `(392,177,24,1)`, and `(192,112,19,1)`.
Thus `E|X|^6 <= s^3*T^3 <= 64*T^3`.

Keeping explained variance and residual variance coupled is essential.
The separate estimates `mean^2<=4*T` and `variance<=4` alone do **not**
prove this sixth-moment bound.

## From coordinates to determinants

In the coordinate order `(xx,yy,xy)`, write
`H=((A,C),(C,B))`. Pointwise,

```
|det H| <= |A*B|+C^2 <= (A^2+B^2)/2+C^2.
```

Minkowski in `L3`, followed by the sixth-moment bound, gives
`||det H||_3 <= (1/2+1/2+1)*4*T = 8*T`.
Hölder with exponents `(3,3,3)` therefore gives

```
E[|det H_M det H_S det H_y| | all nine pins] <= 512*T^3.
```

There is no independence assumption between sites, Hessians, or entries.
The joint conditional Hessian covariance may be singular. Neither
determinant signs nor a saddle/max event is inferred from this inequality.

## Density, full-mark integration, and monotonicity

The conditional three-jet density at `(v,0,0)` is
`exp(-q/2)/((2*pi)^(3/2)*sqrt(det Sigma))`.
Multiplying this density by the preceding moment bound uses the same
`q`, before replacing it by a lower bound. For
`f(q)=(8+q)^3 exp(-q/2)`,

```
f'(q) = -1/2 * (q+8)^2 * (q+2) * exp(-q/2) < 0   (q>=0).
```

Hence `q>=Q` permits `f(q)<=f(Q)`. Uniformity over the **whole** closed
height window allows its length `ell` to be factored once from the
integral. A separate conditional height-window mass must not be
multiplied in again. The central affine mark is
`t=v-(b-1/96000)`, spanning `[-1/96000,1/96000]`.

If a deterministic invertible three-jet coordinate change `Y'=T_y Y`
is used, the Mahalanobis energy is unchanged and
`p_Y(w)=|det T_y| p_Y'(T_y w)`. Use `D` for the transformed covariance
and a matching `jacobian_upper>=|det T_y|`, once. This is distinct from
a polar or other spatial area Jacobian. An unmatched transformed
determinant, Jacobian, order, or target is an invalid application.

## Exact implementation and use

The future import route is `research.rn.density_majorant`; the scratch
module has the same callable interface and currently uses the shared
eligibility binder from the exact runtime repository checkout:

```python
from fractions import Fraction as F
from research.rn.density_majorant import density_moment_majorant

result = density_moment_majorant(
    F(1, 10**20), F(100), repo_root="/absolute/path/to/research-main")
upper = result["integrand_upper"]
```

Those example numbers are hypothetical premises, not a wedge result.
`majorant_interval` encloses the explicit majorant expression; its lower
endpoint is **not** a typed integrand lower bound. Only
`typed_integrand_range=[0,upper]` encloses that integrand, conditionally.

Inputs are exact `int`/`Fraction`, with at most 4096 bits per numerator or
denominator; booleans/floats are refused. Precision is an integer in
128..1024. The implementation uses certified repository `pi`, `sqrt`,
and `exp`, rounding only outward. For bounded resources it replaces `Q`
by `min(Q,2048)` in the whole decreasing function. This can weaken the
upper bound but never strengthens it, underflows to zero, or asserts a
spatial premise. An unresolved positive denominator or failed custody
check raises an exception and returns no successful bound.

The supplied scalar premises have no self-promotion path. All results
retain `authority=NONE`, `field_certified=false`, `wedge_certified=false`,
`spatial_cover=false`, zero organizational independence credit, and no
scientific status change. Full annulus coverage, all radii/angles, RN
closure, q0, and event-0 remain outside this theorem's conclusion.
