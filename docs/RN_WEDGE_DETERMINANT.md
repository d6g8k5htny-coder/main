# A three-jet determinant bound on one SIDE24 wedge

This calculation concerns the normalized SIDE24 Gaussian law, the original six
pins at `(−1/40,0)` and `(1/40,0)`, and values
`(6/5,0,0,6/5−1/48000,0,0)`. The target set is only
`rho∈[1/10,11/100]`, `turns∈[−1/1024,1/1024]` at fixed separation `r=1/20`.
The height interval is the complete `[6/5−1/48000,6/5]`. There is no H3 replay,
whole-annulus bound, all-separation result, source promotion or independence
credit. Prior design exposure and authorship are disclosed.

## Authenticated input and mathematical premise

The caller injects the freshly authenticated authored N6 adapter from
`research/campaigns/h3_rn_n6_20260920_v1.zip`, SHA-256
`73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21`.
Its eligible member `rn_n6/side24_taylor.py` is 23,681 bytes, SHA-256
`32c4933c2ab543f738bd54f98066b351a334e377277e8ebcab5879438f418598`.
The production `n6_inputs.checked_n6()` checks the archive manifest, exclusion
metadata, source eligibility and exact original repository dependencies before
module-cache reuse. This module introduces no cache or archive loader. The
original historical RN5 programs are data and are not executed.

The mathematical source premise is the positive-definite normalized periodized
Gaussian kernel. Its derivative Gram matrices are positive semidefinite; six-pin
conditioning is orthogonal Gaussian projection. The authored N6 routine checks
strict six-pin interval LDL pivots and encloses the conditional derivative mean
and covariance. Kernel derivatives include normalized nonzero images. Its
proved stationary derivative variance caps through order18 bound Taylor tails.

## Taylor polynomials retain their common coordinates

For each target `beta∈{(0,0),(1,0),(0,1)}`, let `P_beta(h)` be the order-six
Taylor polynomial of `D^beta f(center+h)` under the same six-pin conditioning.
The sum uses all `alpha` with `|alpha|≤6` and coefficient
`h^alpha/alpha!`. Before rectangle evaluation, all products contributing to the
same two-variable covariance monomial are added with their signs intact. Thus
the covariance polynomial has degree≤12. Its coefficients are outward rational
intervals enclosing the corresponding actual common-law coefficients.

Choose an exact nonsingular lower-triangular rational matrix `T`, obtained by
inverting an approximate Cholesky factor of the conditional center three-jet.
The approximation is used only to choose coordinates. The covariance is
actually transformed as `T Cov(P) Tᵀ`; it is never replaced by the identity.
Form the symmetric determinant polynomial

`a*d*f + 2*b*c*e − a*e² − d*c² − f*b²`

before any rectangle evaluation. This degree≤36 polynomial preserves the
essential shared-coordinate cancellations. Equal powers are aggregated using
outward interval arithmetic throughout. On a centered rectangle with halfwidths
`hx,hy`, a nonconstant monomial with both powers even lies in
`[0,hx^a hy^b]`; every other nonconstant monomial lies in the corresponding
symmetric interval. The constant monomial is exactly1. Summing these products
therefore encloses the determinant of the actual Taylor covariance for every
point in the rectangle, despite interval-entry LDL refusal.

## Covariance remainder and determinant perturbation

The authenticated N6 remainder function supplies unconditional L2 error bounds
`eps_beta` for the order-six Taylor remainder. Conditional centered remainders
have L2 norm at most `eps_beta` because Gaussian projection contracts norm.
Conditional mean error is at most `sqrt(E) eps_beta`, where
`E=cᵀA6⁻¹c` is the checked six-pin energy.

For transformed coordinate `i`, let `eta_i=Σ_j |T_ij| eps_j`. For each spatial
point let `P` denote its actual transformed polynomial covariance, and choose
`sigma_i=sqrt(sup P_ii)`. Covariance Cauchy–Schwarz gives the uniform entry error

`e_ij = sigma_i eta_j + sigma_j eta_i + eta_i eta_j`.

Let `M_ij` be an absolute upper bound on polynomial covariance entry `P_ij`.
The difference of the determinants is at most

`Σ_permutations [ Π_i(M_i,perm(i)+e_i,perm(i)) − Π_i M_i,perm(i) ]`.

Each bracket is exactly the sum of the absolute upper bounds for all nonempty
error products in that determinant term. Consequently it retains every
linear, quadratic and cubic remainder term. It does not assume the whole
interval matrix hull is PSD. Add this symmetric error interval to the evaluated
determinant polynomial, then divide by the **square** of the exact `det T`.
This encloses the raw conditional three-jet determinant `D`. If its lower
endpoint is not strictly positive, the checker refuses. Otherwise the actual
source covariance is positive definite: it is already PSD by the common-law
kernel/projection argument, and now its determinant is positive.

## Full-height Mahalanobis lower bound

The same mean polynomial plus `sqrt(E) eps_fx` bounds `mu_fx`. The scalar
variance polynomial plus `2 sigma_fx eps_fx+eps_fx²` bounds `Var(fx)`.
Its upper endpoint must be positive. Set

`Q = inf(mu_fx²) / sup Var(fx)`.

For each positive-definite actual three-jet covariance `G` and target
`(height,0,0)`, covariance Cauchy–Schwarz gives
`(target−mu)ᵀG⁻¹(target−mu) >= mu_fx²/G_fx,fx >= Q`.
This holds for every height, so in particular it is uniform over the complete
required mark interval. No three-jet mean/variance values from different
points are treated as a single law: separate extrema merely weaken this valid
scalar inequality.

## Finite spatial scope and exact area

The twelve equal closed x strips partition the containing rectangle
`[9999/100000,11/100] × [−7/10000,7/10000]`. Their shared boundaries have no
gaps. With `theta=2*pi*turns`, `|theta|≤pi/512<22/(7*512)`.
The exact rational checks use `|sin theta|≤|theta|` and
`cos theta≥1−theta²/2` to show the entire target wedge lies in this rectangle.

The target is one polar wedge. Its exact area is
`pi*( (11/100)²−(1/10)² )*(1/512) = 21*pi/5120000`.
Auxiliary Cartesian areas are never added to that area. A uniform integrand
majorant, if separately established with the same source law and any explicit
imported normalization premise, may be multiplied by this one polar area.

At order6 and 256 bits all12 auxiliary strips pass. The common determinant
lower bound exceeds `8/10^25`; the common Mahalanobis lower bound exceeds103.
The reported exact rational endpoints are the proof values; decimal summaries
are diagnostics. The single wider containing rectangle was tried and refused,
and the earlier interval-entry LDL refusals are retained as failed attempts.

The implementation validates exact endpoints, positive widths, the fixed pilot
domain, `1≤order≤6`, `128≤bits≤512`, and `1≤pieces≤256` before computation.
Input or resource failure is not a positive certificate. There are no hidden
pending pieces when the full report returns successfully.
