# Two-axis Hermite–Gaussian envelope — author proof, review pending

2026-09-19. Technical state: **CANDIDATE / NOT REVIEWED**. Independence credit:
**0**. No scientific status, premise, obligation, or grade changes. In particular,
this does not establish the full 24-jet set, six-pin displacement maps, jet
normalizations, any required r-band certificate, or `OBL-H5-JETMOD`.

## Kernel and exact normalization

For nonnegative integers a,b, define

`k_ab(x,y) = (-1)^(a+b) He_a(x) He_b(y) exp(-(x²+y²)/2)`.

Here He denotes the **probabilists'** Hermite polynomial, with He_0=1,
He_1=x, and He_(n+1)=x He_n − n He_(n−1). Differentiating the recurrence
inductively gives He'_n=n He_(n−1), hence

`(He_n(x) exp(-x²/2))' = -He_(n+1)(x) exp(-x²/2)`.

Induction in each variable proves that k_ab is the mixed derivative of the
unit separable Gaussian. This agrees algebraically with `kplane` in
`engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py` (C_KERN=1); the frozen body is neither
modified nor executed. Its other routines and point-tail assertions are not
adopted. The normalization is spelled out because physicists' H_n uses a
different scale; general reference: [NIST DLMF §18.9](https://dlmf.nist.gov/18.9).

## Global envelope, with both axes retained

For m≥0 and t≥0, the nonnegative exponential series implies
`exp(t) ≥ t^m/m!`. Substitute t=x²/4 to obtain

`|x|^(2m) exp(-x²/4) ≤ M_(2m) := 4^m m!`.

The elementary inequality `2s ≤ 1+s²` for s≥0 yields

`|x|^(2m+1) exp(-x²/4) ≤ M_(2m+1) := (M_(2m)+M_(2m+2))/2`.

Write `He_n(x)=Σ_j c_(n,j) x^j` and let
`A_n=Σ_j |c_(n,j)| M_j`. The triangle inequality proves
`|He_n(x)| exp(-x²/4) ≤ A_n`. Multiply this statement for x and y and retain
the remaining Gaussian factor:

**`|k_ab(x,y)| ≤ A_a A_b exp(-(x²+y²)/4)` for every (x,y)∈R².**

All constants are exact nonnegative rationals. A_0=1, A_1=5/2, A_2=5,
A_3=51/2. The amplitude is the product for the actual pair (a,b); it is
not obtained by substituting the total degree a+b into a one-axis estimate.
No sampling, asymptotic approximation, fitted exponent, or float enters the
proof. The constants are deliberately loose and need not be optimal.

## Finite evaluation and infinite tail

Interval Horner evaluation encloses each polynomial over its whole input
interval. Interval squaring and `research.interval.exp` enclose the Gaussian;
multiplication preserves containment, including intervals crossing zero.
Dependency can widen an interval without invalidating containment.

The pair `(A=A_a A_b, B=1/4, valid_from=0)` fits the Gaussian branch of the
existing `tail_bound`. That routine uses an upper bound R on the **Euclidean
radius of the entire displacement box**, and requires `L(N+1)>R`. The shell
count 8m and closed geometric series cover every omitted shell m>N.
`band_enclosure` adds the positive tail on both sides of the finite sum.
A square `[-17,17]²` cannot be assigned R=17.

The new evaluator's arithmetic flag is true. The envelope's review flag
remains **false** under the existing API contract, which requires a human
reviewed argument. Consequently ordinary downstream records remain
NON-CERTIFYING until a legitimate review is supplied. Tests check the
implementation and negative controls; they do not supply that review.

## Remaining work

Bind the actual jet definitions and normalization powers, six-pin geometry,
and authoritative band endpoints; obtain the required review; then produce
finite per-band records with uniform tails and apply the existing falsifier.
These are separate tasks, not conclusions of this component.
