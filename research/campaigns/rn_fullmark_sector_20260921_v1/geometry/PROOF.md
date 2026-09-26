# Exact geometry for bounded RN annular-sector extensions

This module proves a finite integration partition and rational containing
rectangles. It supplies no RN integrand bound. Source covariance, determinant,
full-mark Mahalanobis, normalization and event hypotheses remain the caller's
responsibility. No full annulus, all-radius result or scientific status changes.

## Domains and finite polar partition

For T equal to1/128 or1/64, define

`W_T = {(rho*cos(2*pi*t),rho*sin(2*pi*t)): 1/10<=rho<=3/25, -T<=t<=T}`.

The existing certified wedge W_old uses radii `[1/10,11/100]` and turns
`[-1/1024,1/1024]`. The new module supports the whole W_T and, preferentially,
its increment outside W_old. Sets differing on their boundary are equivalent
for the integrals here; all constructed cells are closed and shared boundaries
have area zero.

Set `r_i=(20+i)/200`, i=0,...,4. Angular boundaries are
`t_j=-T+j/256`, j=0,...,N, where N=4 for T=1/128 and N=8 for T=1/64.
The four radial shells and N angular intervals give16 or32 cells. No angular
cell crosses zero: it is a boundary in both grids.

For the incremental partition, change only the two angular cells adjoining
zero in each of the first two radial shells. End the negative one at
`-1/1024` and begin the positive one at `1/1024`. This removes W_old's interior
without adding cells. The outer two shells are unchanged. The increment is
the union of three disjoint-interior parameter rectangles:

1. `[11/100,3/25] x [-T,T]`;
2. `[1/10,11/100] x [-T,-1/1024]`;
3. `[1/10,11/100] x [1/1024,T]`.

The polar map is injective on this positive-radius domain, except for no
additional identifications beyond shared boundaries: total angular span is
strictly below a full turn. Hence a partition with disjoint parameter-cell
interiors also has disjoint spatial-cell interiors. There is no origin or
periodic-turn seam inside this bounded signed-turn domain.

`validate_partition` checks coverage independently of the constructor: collect
every submitted radial/angular endpoint and the domain/exclusion boundaries,
then form the elementary open grid atoms. Every required atom must have exactly
one covering cell; atoms in the excluded old wedge must have zero. Exact
Fractions make these finite comparisons exact. Thus equal-area gaps and
overlaps cannot cancel unnoticed. The independent exact area sum is a second
check, not a substitute for atom occupancy.

## Rational Cartesian containing rectangles

For a nonnegative angular interval `0<=a<=t<=b`, put
`L=6a`, `U=(44/7)b`. Since `3<pi<22/7`, the actual angle lies between L and U.
Every supported cell has `0<=L<=U<1/10`. On this range sine is increasing,
cosine is decreasing, and the alternating Taylor inequalities give

`1-U²/2 <= cos(theta) <= 1-L²/2+L⁴/24`,

`L-L³/6 <= sin(theta) <= U`.

All displayed lower bounds are nonnegative and the cosine lower bound is
strictly positive. Therefore the complete polar cell
`[r0,r1] x [a,b]` lies in the rational Cartesian rectangle

`[r0*(1-U²/2), r1*(1-L²/2+L⁴/24)]`

`x [r0*(L-L³/6), r1*U]`.

For negative turns `[-b,-a]`, evenness of cosine leaves the x interval
unchanged; oddness of sine sends `[y0,y1]` to `[-y1,-y0]`. The implementation
performs exactly this reflection. It refuses cells crossing zero rather than
quietly applying a one-sided bound to them.

For completeness, the pi bracket has elementary exact witnesses. With
`pi=4*integral_0^1 1/(1+x²) dx`, the eight-term alternating geometric expansion
has a strictly positive remainder. Its integrated rational lower bound is
`4*sum_{k=0}^7 (-1)^k/(2k+1)>3`. Also

`22/7-pi = integral_0^1 x^4(1-x)^4/(1+x²) dx > 0`.

Polynomial division gives quotient `x^6-4x^5+5x^4-4x²+4` and remainder -4;
integrating the quotient gives22/7. Tests verify these rational identities.
The sine/cosine inequalities follow from their alternating series, whose
term magnitudes decrease on the stated interval. No floating-point
trigonometric call or sampling is used as containment evidence.

## Area and no double counting

The polar Jacobian is `2*pi*rho` because t is measured in turns. Integrating
rho exactly once gives the cell area

`pi*(r1²-r0²)*(t1-t0)`.

Consequently the full areas are `11*pi/160000` for T=1/128 and
`11*pi/80000` for T=1/64. The removed old wedge has area
`21*pi/5120000`, giving incremental areas

`331*pi/5120000` and `683*pi/5120000` respectively.

Each report also gives the strict rational area upper obtained by multiplying
its exact pi coefficient by22/7. The actual pi coefficient remains the
primary value; the rational upper is not mislabeled as the exact area.

Suppose a caller proves a nonnegative integrand bound C_i on each auxiliary
Cartesian containing rectangle B_i. The valid budget is

`integral_{new region} I <= pi*sum_i C_i*(r1_i²-r0_i²)*(t1_i-t0_i)`.

The polar cells, not B_i, determine integration ownership. Overlapping B_i
therefore cause no double counting. Never sum B_i's Cartesian areas as if
they formed the target partition. For incremental mode the old wedge may be
added once using its original certificate. For whole-sector mode the old
wedge is already included and must not be added again. No numerical C_i is
asserted by this geometry module.

## Compatibility with the bounded N6 source domain

Exact enumeration for all four supported constructions (two angular widths,
whole/incremental) proves every auxiliary halfwidth satisfies
`hx<3/1000` and `hy<1/500`, in particular both are below1. Every rectangle
has `0<x<=3/25`, `|y|<=33/2800`, and its entire x interval is strictly above
the fixed right pin x=1/40. Hence neither a center nor its rectangle hits a
fixed pin; center coordinate magnitudes are strictly below17.

Coordinate displacement from any rectangle point to either fixed pin
`(+/-1/40,0)` is at most29/200, strictly below the source kernel's allowed
reduced image-tail radius18. The center-to-pin covariance arguments and
zero-displacement center derivative covariances therefore fit that domain.
For order-six Taylor transport the authenticated N6 module's stationary
derivative caps through order18 and halfwidth limit1 are unchanged. Gaussian
projection variance contraction and the fixed six-pin energy control remain
the existing analytic premises; they are not established by a box-size check.

These checks justify the new geometry as an admissible input domain for a
numbered source-law successor. They do not establish that interval LDL,
determinant positivity, energy lower bounds or a useful integrand cap succeed
there. The original wedge's hardcoded domain must not be silently widened in
a frozen artifact. Source authentication and numerical certificates remain
the parent runner's responsibility.

## Interface and verification

`build_sector(Fraction(1,128), incremental=True)` and its1/64 counterpart
return dictionaries with a `sectors` list. Each cell carries exact Fraction
`r0,r1,t0,t1`, four-Fraction `containing_rectangle=(x0,x1,y0,y1)` and
`area_pi_coefficient`. The report includes exact aggregate areas and source
range caps. `validate_report` recomputes geometry, partition and aggregates.
Only the two explicitly bounded candidates are admitted. Floats, bools used
as exact numbers, malformed coordinates, invalid reflection/area, oversized
inputs and changed scope fields are refused. The module imports no repository
code or external package and writes nothing.

Run `python -B -m unittest -v test_geometry` and the same with `-O`.
Eighteen tests include missing cells, equal-area gaps/overlaps, duplicate old
wedge ownership, wrong reflection, doubled area/Jacobian, malformed inputs,
source-domain caps and explicit pi-bracket witnesses. These are geometry
checks, with zero organizational-independence credit.
