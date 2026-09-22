# A full-height RN sector successor

This numbered author-side successor uses the existing normalized SIDE24
two-dimensional Gaussian field, six-pin law, fixed radius `r=1/20`, birth
`b=6/5` and horizontal pin axis. Its new work is a full-covariance energy
inequality and an exact finite polar partition. It does not reprove the
imported fixed-radius H3 normalization or six-pin energy result.

## Exact mathematical target

With turns measured as fractions of a full revolution, let

`W = {(rho*cos(2*pi*t), rho*sin(2*pi*t)):
       1/10 <= rho <= 3/25, -1/128 <= t <= 1/128}`.

The old wedge has radii `[1/10,11/100]` and turns
`[-1/1024,1/1024]`. Its verified full-height result, with byte identity pinned
in `check.py`, supplies the strict integral upper bound
`33/2560000000000`. The new geometry excludes the old wedge's interior and
partitions the remaining part of W into 16 cells with disjoint interiors.
All boundaries have area zero. The entire height interval is
`[57599/48000,6/5]`; the centered mark interval is `[-1/96000,1/96000]`.

The exact whole-sector area is `11*pi/160000`. Its ratio to the old wedge's
area is `352/21`. The new increment has area `331*pi/5120000`.
These are area statements, not fractions of the required full annulus.

## Per-cell certificate

For each cell, `geometry/geometry.py` constructs a rational Cartesian
rectangle containing the entire cell. Its proof establishes containment
using exact elementary bounds for pi, sine and cosine, validates occupancy
of every elementary polar atom, and checks the retained N6 source domain.
No sample points establish containment or numerical inequalities.

The N6 adapter is authenticated against current source-admission metadata
before use and before every rectangle. The original common-law Taylor
polynomials and rigorous L2 remainders are retained. Each cell requires a
strictly positive lower bound D for the actual three-jet covariance
determinant. Actual covariance is positive semidefinite by the Gaussian
field construction, hence is positive definite when D is positive.

The new full-height energy argument is in `projection/PROOF.md`. It first
tries a fixed rational projection. If that does not prove the requested
threshold, it preserves the full three-coordinate quadratic form through
the common adjugate polynomial. Covariance, mean, mixed and determinant
perturbations are bounded in the same transformed coordinates. Joint
spatial/height Bernstein coefficients enclose every point of the full
height interval. A failed target only retains a separately proved lower
bound; it does not silently grant the requested threshold.

For actual Mahalanobis energy q at height `(v,0,0)`, the cell output gives
`q >= Q >= 0`. A threshold of 103 is requested by the recorded run. The
actual returned Q and full-adjugate margins are recorded for every cell.
The original-coordinate determinant is restored once by division by
`det(T)^2`. This coordinate determinant is unrelated to the polar area
Jacobian.

## Density bound and integration

The unchanged sharp stationary-moment theorem is supplied verbatim in
`inputs/sharp_variance/PROOF.md`. With normalized torus moments m2 and m4,
it retains the image terms and proves
`(m4+m2^2)^3 < C = 64000001/1000000`.
Exactly 64 is not used as a bound. The imported constants are

- `E6 <= 1266466/160083 < 8`;
- `Z_lo = 15518583475065571/2000000000000000000`;
- height length `ell = 1/48000`.

For the established cell premises D and Q, the upper bound is evaluated
with outward rational interval arithmetic as

`C_cell >= ell*C*(8+Q)^3*exp(-Q/2)
                 /(Z_lo*(2*pi)^(3/2)*sqrt(D))`.

The function `(8+q)^3*exp(-q/2)` is decreasing for q >= 0. Its moment
factor and density factor are combined before substituting the lower
energy bound. The nonnegative integrand is enclosed by `[0,C_cell]`;
the lower endpoint of an intermediate expression is not an integrand
lower bound.

For a polar cell `[r0,r1] x [t0,t1]`, integration of the Jacobian
`2*pi*rho` gives area `pi*(r1^2-r0^2)*(t1-t0)`. Therefore

`integral_W I < 33/2560000000000
              + (22/7)*sum_cells C_cell*(r1^2-r0^2)*(t1-t0)`.

The old bound enters once, after the increment. The Cartesian containing
rectangles may overlap: their areas are never integrated. The rational
pi bound is proved in the geometry notes. Exact arithmetic checks the
reported total and any simpler displayed bound. Diagnostic decimal
conversions have no role in certification.

## Execution and scope

The completed 16-cell runs establish Q >= 103 in every cell. The largest
new-cell integrand cap is strictly below `7/10000000`. Exact rational
aggregation gives a whole-sector upper bound strictly below
`7/100000000000`; the recorded approximate value is
`6.732393130703628e-11`. The prior wedge's uniform cap is below `1/1000000`,
so that same uniform cap holds throughout the whole expanded sector.
Normal and optimized certificate bytes agree exactly.

`check.py` authenticates the 36 declared repository files and the prior
result before calculating, checks newly imported repository code against
that declared set, and checks source and local implementation identities
again afterward. It only writes to a fresh external output directory.
Every cell is retained, including an inconclusive cell. Any pending cell
prevents a complete-cover total. `summarize` is an internal exact
composition helper, not a general validator for arbitrary supplied JSON
or a substitute for the authenticated numerical run.

The alternate 32-cell geometry has executable construction and geometry
tests, but is not a numerical coverage claim unless a separate complete
run is supplied. The signed-turn width in each recorded result is the
scope of that run. No full annulus, all radii, other pin orientations,
weighted-Palm/event identification, q0 closure or scientific status
promotion follows from this bounded sector result. Same-provider tests,
reviews and replays have zero organizational-independence credit.
