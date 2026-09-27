# A uniform SIDE24 rectangle enclosure

`research.rn.side24_cell` extends the source-bound point calculation to a
nonzero spatial rectangle by an explicit L² increment argument. It returns
an upper for the complete typed RN integrand, including the full height window
and division by the separately imported H3 floor. It provides no complete
annulus cover, remote-budget assembly or all-small-r theorem; it changes no
scientific status and earns zero organizational independence credit.

Prep order is [RN_SIDE24.md](RN_SIDE24.md), then
[RN_SIDE24_DENSITY.md](RN_SIDE24_DENSITY.md), then this note. The ABSENT /
REFUSED / OPEN walls for this step are under
[Prep path and honesty walls](#prep-path-and-honesty-walls).
`field_certified` stays false.

At the declared square

\[
C=[1-1/4000,1+1/4000]^2
\]

the three portable moment witnesses replay and give a uniform upper below
\(559/10^8=0.00000559\). The actual stored upper is an exact rational,
approximately \(5.5890032754\cdot10^{-6}\). This is a statement for every
point of C and every height mark, conditional on the imported H3 premise.
The proof below, not point or corner sampling, supplies those quantifiers.

## Field and source binding

The field, six pins, normalization, Hessian order and mark law are exactly
those of [RN_SIDE24.md](RN_SIDE24.md). In particular r=1/20, b=6/5,
\(M=(-1/40,0)\), \(S=(1/40,0)\), \(\ell=1/48000\), and the fixed six-pin
vector is \(c=(b,0,0,b-\ell,0,0)\). The full moving mark is
\(v=b-\ell/2+t\), \(|t|\leq1/96000\).

The adapter verifies the same source document, archive and three archive-member
hashes before constructing its center law. No frozen executable is run. The
H3 floor, exact identity, imported status, and density/window factorization are
as documented in [RN_SIDE24_DENSITY.md](RN_SIDE24_DENSITY.md).

RN5's written covariance-remainder argument, equations (3)–(4), motivates the
construction. This implementation uses only a center value and an integral
first-derivative bound. It does not reproduce the source's degree-six Taylor
polynomial, higher-order remainder machinery or spatial-box numerical results.
It makes no novelty or numerical-dominance claim over those existing sources.

## Fixed six-pin conditioning

Let O be the six fixed pin functionals, with covariance A, and let P be the
orthogonal projection in the centered Gaussian L² space onto their span.
Positive interval LDL pivots certify A as nonsingular. For a field functional
X, its six-pin-conditioned law has deterministic mean

\[
m_X=\operatorname{Cov}(X,O)A^{-1}c
\]

and centered residual \(F_X=(I-P)X\). Its covariance is that of this residual.
The normalized Fourier derivative Gram representation already established for
the point adapter guarantees a coherent PSD joint Gaussian law. No claim is
made that every independent matrix entry in an interval hull is a field law.

At the rational center z of the rectangle, assemble and condition these twelve
targets in this exact order:

\[
(H_M^{xx},H_M^{yy},H_M^{xy},
  H_S^{xx},H_S^{yy},H_S^{xy},
  f(z),f_x(z),f_y(z),H_z^{xx},H_z^{yy},H_z^{xy}).
\]

The existing 18-dimensional raw covariance supplies the six observations and
twelve targets. The fixed six-pin covariance is inverted only through the
certified LDL solves at this center. Spatial variation is handled after that
operation, so raw interval widths are not amplified by repeatedly enclosing
the ill-conditioned fixed-pin inverse.

## Uniform L² and mean increments

Write \(y=z+\delta\), with \(|\delta_j|\leq h_j\). For each moving
derivative of multi-index a, of total order at most two, define

\[
\epsilon_a=\sum_{j=1}^2 h_j
 \sqrt{\operatorname{Var}(\partial^{a+e_j}f)}.
\]

Along the segment from z to y, the fundamental theorem of calculus and
Minkowski's inequality give

\[
\|\partial^a f(y)-\partial^a f(z)\|_2\leq\epsilon_a.
\]

The derivative variances are stationary, so this is valid on the whole
segment, not just its endpoints. Orthogonal projection is a contraction;
therefore the same upper bounds the centered **conditional** increment
\(F_a(y)-F_a(z)\). Fixed Hessian functionals at M and S have increment zero.
Every appearance of the moving height, gradient and Hessian uses this same y;
there are no independent copies of a spatial point or a spurious interval
subtraction for a self-covariance.

Set \(U=c^TA^{-1}O\) and \(Q=\operatorname{Var}U=c^TA^{-1}c\).
Cauchy–Schwarz then yields

\[
|m_a(y)-m_a(z)|
=|\operatorname{Cov}(\partial^a f(y)-\partial^a f(z),U)|
\leq\sqrt Q\,\epsilon_a.
\]

The code computes Q as an LDL sum of squares, with a positive variance
denominator for every term. It bounds each moving mean by its center interval
plus this symmetric error. Conditional centered variation and conditional
mean variation are different quantities; neither is omitted.

## Covariance transport

For each centered target at z, let \(\sigma_i\) be an upper bound on its
conditional standard deviation, obtained from the center covariance diagonal.
Writing \(F_i(y)=F_i(z)+R_i(y)\), with \(\|R_i(y)\|_2\leq\epsilon_i\),
gives

\[
|\operatorname{Cov}(F_i(y),F_j(y))-
  \operatorname{Cov}(F_i(z),F_j(z))|
\leq\sigma_i\epsilon_j+\sigma_j\epsilon_i+\epsilon_i\epsilon_j.
\]

This follows by expanding the covariance into its two linear-increment terms
and one product-increment term and applying Cauchy–Schwarz to each. The source
covariance is enclosed by adding this symmetric error to every center entry;
symmetric entries remain identical. The fixed-pair covariance block has zero
transport error. Enlarging the family to an interval entry hull spends
tightness but asserts no independence among entries or Hessian blocks.

## Third-derivative variance and normalized image tail

The only new kernel derivative beyond the point adapter is \(k^{(6)}(0)\).
All stationary derivative variances of total order at most three are products
\((-1)^{|a|}k^{(2a_1)}(0)k^{(2a_2)}(0)\).
For order six,

\[
\mathrm{He}_6(x)=x^6-15x^4+45x^2-15.
\]

The finite image numerator is
\(-15+2\mathrm{He}_6(24)e^{-288}\). For \(|j|\geq2\), the pair of
images is bounded in absolute value by
\(2\cdot76(24j)^6e^{-(24j)^2/2}\), since 76 is the sum of the absolute
Hermite coefficients and \(|24j|\geq1\). Consecutive ratios decrease and
are at most \(q=(3/2)^6e^{-1440}<1\). Thus the complete omitted numerator
has absolute value at most

\[
\frac{2\cdot76\cdot48^6e^{-1152}}{1-q}.
\]

The finite numerator plus this symmetric tail is divided by the same full
positive normalized image sum as the point adapter. No image is discarded and
no planar moment is substituted. At 512 bits the tests strictly distinguish
\(-k^{(6)}(0)\) from 15 and the mixed third-derivative variance from 3.
All lower kernel orders retain their existing normalized image tails.

## Conditioning on the moving three-jet

Extract the cell-wide six-pin mean and covariance of Y=(f(y),fx(y),fy(y)),
the nine Hessians H, and their cross-covariance from the transported twelve
targets. Condition H on
\(Y=(b-\ell/2+t,0,0)\) using the existing interval Gaussian solve, passing
the nonzero prior means explicitly. This requires positive Y pivots throughout
the interval enclosure. For every fixed y, the resulting covariance is
independent of t and the mean is affine in the same t.

Require positive pivots for each 3-by-3 Hessian marginal enclosure. No strict
positive-definiteness requirement is imposed on the whole nine-Hessian block.
A failed pivot is an inconclusive sufficient calculation, not a counterexample
to the field law. The current implementation refuses that cell.

Three generic moment certificates enclose the M4, S4 and y2 moments on the
**complete** mark interval, and the separate replay verifier checks their
coefficient derivation and Bernstein partitions. Their common law identifier
includes the entire declared rectangle, coordinate convention and mark law.
Hölder gives a uniform determinant-factor upper.

The transported six-pin Y law also feeds the existing gradient-density and
conditional-height-window helpers. The height/gradient covariance is retained.
Multiply their uniform density/window upper by the uniform Hölder upper and
divide by the pinned positive H3 floor. Maxima may occur at different points;
their product is still an upper. The result is `[0,U]` for the actual
nonnegative typed integrand; no positive lower bound is claimed for it.

## API, failure and coverage boundary

- `transported_law(Box, bits=192)` returns source binding, center law, six-pin
  pivots, Q, stationary derivative variances, L² errors, covariance error
  arrays, and the enclosed twelve-target family.
- `cell_laws(Box, bits=192)` adds three `FamilyLaw` objects, their marginal
  pivots, Y pivots, full affine mark laws and the density/window upper.
- `spatial_cell(Box, bits=192)` adds the replayed portable certificates,
  rational moment caps, Hölder upper, `integrand_upper` and `integrand_range`.

Boxes use exact `research.cover.Box` endpoints. Singleton boxes are allowed
for point-limit tests; a geometric cover must separately reject zero-area
partition cells. Endpoints, precision, side lengths and input sizes are
bounded. Boxes containing a fixed pin are refused. The math adapter does not
require a rectangle to lie inside the annulus: a cover region must separately
classify or account for its geometric intersection.

A failed covariance pivot or exhausted arithmetic budget cannot produce an
accepted upper. A cover controller must refine or retain a pending cell, keep
all boundary contributions, and verify a complete exact partition with zero
pending cells before reporting a total. A local rectangle result alone does
not supply these facts. Upper-sum admission and the existing driver's
two-sided range-width target are separate policies; the absence of a positive
typed lower bound is not an obstacle to a valid complete upper sum.

This low-order transport is locally feasible but is not a practical whole
annulus method. For example, a square of half-width 1/1000 at (1,1) fails the
sufficient marginal pivot check, while half-width 1/4000 succeeds. Diagnostics
near the inner edge required much smaller cells. Those refusals do not disprove
the true field covariance. The source RN5 degree-six centered expansion and
separate marginal whitening remain the route to investigate for useful
near-axis widths; a full annulus cost is not inferred from the present pilot.

## Validation and trust

```bash
python -m pytest -q -p no:cacheprovider tests/test_rn_side24_cell.py
```

The tests replay a nonzero rectangle's three witnesses, check exact source and
mark context, verify positive pivots, compare the singleton limit and selected
member points with the direct point adapter, and check nested transport bounds.
Semantic mutants dropping all spatial remainders or dropping the regression
energy fail to enclose an actual separately recomputed endpoint mean. Larger
unresolved cells and invalid/resource-limited inputs refuse without a cap.

Point and corner comparisons are implementation diagnostics. The interval
kernel, Gaussian projection, integral increment and covariance arguments above
establish the continuum claim. The producer, replay, point and cell paths share
certified interval primitives and source conventions; technical cross-checks
do not manufacture organizational independence or change source status.

## Prep path and honesty walls

This note is step 3. It uses the point law in [RN_SIDE24.md](RN_SIDE24.md) and
the density/window factor plus the imported H3 floor in
[RN_SIDE24_DENSITY.md](RN_SIDE24_DENSITY.md). It does not replace either note.
The lane pointer is [RESEARCH_MAP.md](RESEARCH_MAP.md) §3. The matching
command is in [RESEARCH_EXECUTION.md](RESEARCH_EXECUTION.md).

**Where the rectangle and budget are checked.** The larger rectangle, its
four-cell split and the local budget are not in this note. They are checked by
[`tools/rn_side24_spatial_check.py`](../tools/rn_side24_spatial_check.py)
against the stored candidate
[`research/rn/candidates/side24_spatial_20260920_v1.json`](../research/rn/candidates/side24_spatial_20260920_v1.json).
The checker fixes
`RECTANGLE = Box(F(1999, 2000), F(2001, 2000), F(1999, 2000), F(2001, 2000))`
and `LOCAL_BUDGET = F(3, 500000000000)`. Its replay expects one retained
`INCONCLUSIVE` parent and four `BOUNDED` child cells, and a complete local
cover whose integral upper is at most `3/500000000000`. The spatial claim of
this note stays the declared square C above. The rectangle, the four-cell
partition and the budget are the checker's engineering replay. They are not an
enlarged claim of this note, and they are not discharge. The checker scope
records `near_annulus_covered: false`, `independence_credit: 0` and
`h3_floor_status: EXPLICIT_IMPORTED_HYPOTHESIS`: the H3 floor is imported and
is not re-proved here. SIDE24 source-of-truth carriers for the objects marked
**ABSENT** below stay **ABSENT**.

**Frozen importer errata.** `docs/OPEN_PROBLEMS.md` §A5 (lines 116-120) still
credits this note with the rectangle, its four-cell split and the `6e-12`
budget. That file is byte-frozen by a pinned campaign archive, so the scoped
reading is recorded beside it, as E1 in
[OPEN_PROBLEMS_FROZEN_ERRATA_20260925.md](OPEN_PROBLEMS_FROZEN_ERRATA_20260925.md).
The downstream map is
[DOWNSTREAM_RN_CROSSWALK_20260925.md](DOWNSTREAM_RN_CROSSWALK_20260925.md).
These pages are navigation only. The spatial claim of this note stays the
declared square C.

**ABSENT** from this note: a complete annulus cover, a remote-budget assembly,
an all-small-r theorem, and a positive lower bound on the typed integrand.
`field_certified` stays false. A local rectangle does not supply those absent
objects, and this note does not invent a source of truth for them.

**REFUSED**, as already stated above: boxes that contain a fixed pin, a failed
covariance pivot, and an exhausted arithmetic budget. A failed pivot or an
exhausted budget cannot produce an accepted upper. The square of half-width
1/1000 at (1,1) fails the sufficient marginal pivot check, while half-width
1/4000 succeeds. Those refusals do not disprove the field covariance.

**OPEN:** the declared square is the whole spatial claim of this note. The
complete annulus cover, remote-budget assembly, and all-small-r theorem stay
open. Piece 2 of `D3-LEMMA-RN-UNIF` stays OPEN, as
[RESEARCH_MAP.md](RESEARCH_MAP.md) §3 already records. `D3-LEMMA-RN-UNIF`
stays OPEN. `OBL-H5-JETMOD` stays OPEN. `lemma_closed`,
`certified_C_H`, and `prizes_solved` stay false. A green run of
`tools/rn_side24_spatial_check.py` is an engineering check. It is not
discharge. Independence credit stays zero. SIDE24 has no new source-of-truth
carrier after 2026-08-06. Carriers for the objects marked **ABSENT** above
stay **ABSENT**. Quarantine is not a source of truth.
`inventable_attempt_accepted` stays false.
