# Exact interval conditioning for a declared Gaussian law

`research/rn/conditioning.py` computes containing affine means and Schur
covariances for a fixed Gaussian law, using rational interval LDL factorization
and triangular solves. It accepts at most 18 joint coordinates, including the
SIDE24 adapter's nine conditioning coordinates and nine Hessian coordinates.

The numerical inclusion statement is a candidate mathematical implementation.
It supplies no identification with the actual RN field, joint-PSD existence
proof, spatial cover, RN lemma closure, scientific status change, or
organizational independence credit. Matching source-law labels check
consistency; they do not authenticate the source. Field identity comes from the
separate source adapter and its proof obligations.

## Interface and exact quantifier

Write the joint Gaussian vector as `(X,Y)` and provide

\[
A=\operatorname{Cov}(X,X),\qquad
C=\operatorname{Cov}(Y,X),\qquad
B=\operatorname{Cov}(Y,Y).
\]

The rows of `C` are target coordinates and its columns are conditioning
coordinates. `conditioning_order` and `target_order` are explicit, unique,
disjoint coordinate labels. All three caller declarations `covariance_law`,
`mean_law`, and `conditioning_law` must identify the same law.

```python
from research.rn.conditioning import condition_gaussian, interval_ldlt

conditioned = condition_gaussian(
    A, C, B, c0, c1,
    conditioning_order=pin_order,
    target_order=hessian_order,
    covariance_law=law_id,
    mean_law=law_id,
    conditioning_law=law_id,
    mark_domain=(t_min, t_max),
    round_bits=192,
)
```

Each numerical input is an integer, `Fraction`, integer-ratio string, or an
`Interval` with exact rational endpoints. Floats, bools, exponent strings,
dimension mismatches, and asymmetric interval blocks are refused. Prior means
default to zero. The optional `prior_conditioning_intercept`,
`prior_conditioning_slope`, `prior_target_intercept`, and
`prior_target_slope` supply affine prior means in the same scalar mark.

The result contains `.intercept`, `.slope`, `.covariance`, `.gain`, and the
conditioning `.pivots`. The target mean is enclosed by `intercept + slope*t`.
`.mean(t)` checks the declared closed mark domain before evaluating it.
`interval_ldlt(matrix).solve(rhs)` exposes the lower-level bounded solve, and
the factor's `.diagonal` and `.pivots` are aliases.

The quantifier is **every fixed joint PSD Gaussian member** of the supplied
interval blocks, with covariance independent of the mark. For each such
member, the same fixed covariance is used for its mean and Schur complement
at every mark in the domain. Repeated uncertain values are not assumed
independent. The returned interval box may include extra, non-PSD matrices;
it is an enclosure, not a declaration that every returned box member is a
Gaussian covariance. Existence of a joint PSD member is not proved by this
module. Source labels alone cannot establish existence or mark independence.

## Inclusion argument

For a symmetric conditioning matrix, the ordinary LDL recurrences are

\[
d_j=A_{jj}-\sum_{k<j}L_{jk}^2d_k,
\qquad
L_{ij}=\frac{A_{ij}-\sum_{k<j}L_{ik}L_{jk}d_k}{d_j},\quad i>j.
\]

The implementation evaluates these recurrences using the public `Interval`
operations. Inductively, every exact member's factors lie in the computed
factor intervals. It admits a pivot only if its lower endpoint exceeds the
requested nonnegative `pivot_floor` (zero by default). Thus all admitted exact
pivots are positive, and congruence with their positive diagonal matrix proves
that every symmetric member of the conditioning box is positive definite.
No midpoint, numerical eigenvalue, square root, or explicit inverse enters
this proof.

To solve `Ax=b`, it encloses the forward solve `Ly=b`, divisions `z_j=y_j/d_j`,
and backward solve `Lᵀx=z`. Each denominator excludes zero. Induction through
these recurrences proves inclusion even though the factor intervals contain
dependent entries. Dropping those dependencies may widen the result, but
does not weaken containment.

For each target row, solving `A*x=C[row]ᵀ` gives a containing row of
`G=C*A⁻¹`, since each admitted matrix `A` is symmetric. The Gaussian
conditioning identities then give

\[
\Sigma_{Y\mid X}=B-GC^T,
\qquad
\mu_{Y\mid X}(t)=m_Y(t)+G\,[c_0+c_1t-m_X(t)].
\]

The coefficient operations enclose the affine intercept and slope separately.
Each Schur entry is computed in both symmetric orientations, and their
intervals are intersected. Both enclose the same exact entry, so this
intersection preserves inclusion and produces identical symmetric output
entries. A negative upper endpoint for a conditional variance is refused
because it excludes a joint PSD member.

## Bounded arithmetic and failure meaning

The default precision is 192 significant bits, configurable from 128 through
1024. Small exact rationals are retained exactly. Larger operation results are
widened using `Interval.round_out`; it rounds the lower endpoint down and the
upper endpoint up using exact integers. This is a tightness/resource choice,
not a correctness assumption. Numerator and denominator sizes are checked
against 16,384 bits, and the arithmetic operation budget is 200,000.

If an interval pivot reaches zero, the calculation returns no bound and raises
`ConditioningInconclusive`. This can reflect interval overestimation even when
a particular point matrix is SPD. A very small exact positive pivot can still
be admitted; no hidden floating-point condition-number cutoff is used. The
caller may choose an explicit positive `pivot_floor`. Resource exhaustion
raises `ConditioningResourceLimit`, also without a bound. Increasing precision
may improve an enclosure but is not guaranteed to resolve dependency width.

## Negative controls

The tests construct an exact correlated joint law as `Y=T*X+R`, with independent
Gaussian residual `R`, and recover its exact affine gain and residual covariance.
A separately implemented rational Gauss-Jordan inverse checks the solve and
Schur results. Direct conditioning agrees with two-stage conditioning, including
the nonzero affine intermediate means.

Other controls reject zero/negative pivots, uncertain ill-conditioned blocks,
mixed law labels, wrong coordinate orders, dimensions and inexact inputs.
Flipped Schur signs, a cross-block transpose, and midpoint substitution fail
the exact reference comparisons. A small interval family is checked against
192 exact matrix-corner members, and precision/resource controls check that
outward rounding retains the exact solution. These tests support the stated
implementation behavior; they do not provide independent mathematical review
or discharge a research obligation.
