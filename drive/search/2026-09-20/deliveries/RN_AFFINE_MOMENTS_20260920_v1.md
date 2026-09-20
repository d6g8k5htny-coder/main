# Source-bound affine Gaussian moment operations

This is an **unreviewed algebraic candidate**, implemented in
`research/rn/gaussian_moments.py` and exposed through
`engine/operations/rn_applicability.py`. It changes no scientific status.
RN5's exact source is Drive `1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5`, SHA-256
`ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383`.
The fresh reconnaissance memo and raw custody receipt accompany this delivery.

## Mathematical interface and derivation

Fix one Gaussian conditional marginal `X(t)` with mean `m(t)=a+b t`
and constant symmetric positive-semidefinite rational covariance `S`.
The coordinate order is **(xx, yy, xy)** and `D=X1 X2-X3²`.
This identifies a two-dimensional symmetric Hessian; it is not the alternate
(xx, xy, yy) ordering. Neither order says anything about a three-dimensional
Hessian. Rational inputs identify an exact synthetic law unless a separate
field-identification argument is supplied.

For a multi-index alpha, choose i with alpha_i>0 and write beta=alpha-e_i.
Gaussian integration by parts gives, with `M_0=1`,

`M_alpha(t) = m_i(t) M_beta(t) + sum_j beta_j S_ij M_(beta-e_j)(t)`.

Indeed apply `E[(X_i-m_i) f(X)]=sum_j S_ij E[partial_j f(X)]`
to `f(X)=X^beta`; terms with beta_j=0 are absent. This identity holds also
for degenerate Gaussians by a linear representation of a standard Gaussian
(or a vanishing positive-diagonal perturbation). Induction on total degree
shows that every `M_alpha` is a polynomial of degree at most |alpha|.
The determinant expansion is

`E[D^k]=sum_(j=0)^k (-1)^j binom(k,j) M_(k-j,k-j,2j)`.

Thus k=2 and k=4 give degree at most four and eight, respectively, as RN5
already states. The new implementation makes those correlated, non-centered
polynomials executable over exact fractions. It claims no new Gaussian
moment theorem. See [Mamis, version 6](https://arxiv.org/html/2202.00189v6)
for the Gaussian moment/Stein setting.

There is a separate second-moment check. Let Q12=Q21=1/2, Q33=-1 and all
other entries zero, so `D=X^T Q X`. Write `X=m+Z`. The odd centered terms
vanish; Isserlis gives `E[(Z^T Q Z)^2]=tr(QS)^2+2tr(QSQS)`, while
`E[(m^T Q Z)^2]=m^T Q S Q m`. Consequently

`E[D²]=(m^T Q m+tr(QS))²+2tr(QSQS)+4m^T Q S Q m`.

The tests also expand `m+LZ` directly in independent standard-normal
coordinates and integrate each monomial, checking fourth moments without
using the Stein recurrence or the Q formula. Nonzero means, cross covariance,
singular laws and the independent RN5 counterexample are all exercised.

## A bound over the entire mark interval

For `p(t)=sum c_j t^j` on [l,h], substitute `t=l+(h-l)u` and call the new
power coefficients `a_j`. For n=degree(p), convert to Bernstein coefficients

`b_k=sum_(j=0)^k a_j binom(k,j)/binom(n,j)`.

The identity `u^j=sum_(k=j)^n binom(k,j)/binom(n,j) B_(k,n)(u)` follows
by expanding the binomial theorem. On [0,1], the Bernstein basis functions
are nonnegative and sum to one, so p lies between min(b_k) and max(b_k).
The implementation returns the repository's exact `Interval`. Dyadic
subdivision evaluates **both** children recursively and takes their hull;
every point of the original closed interval is included. Endpoint or grid
sampling is not the argument. This elementary convex-hull property is also
described in [Hamadneh, Section 2](https://onlinelibrary.wiley.com/doi/10.1155/2022/9156188).

RN5's cap `c0+sum_(j>=1)|c_j| H^j`, with H=max(|l|,|h|), remains valid.
We take the smaller of this cap and the Bernstein upper bound. This cannot
weaken the coefficient cap. The explicit development witness
`m=(-1+t,-1-t,0), S=0, t in [-1,1]` has `E[D^4]=(1-t²)^4`.
The old cap is 16; subdivided Bernstein gives 1, attained at t=0.
The candidate report includes a separate strictly improved, positive-definite
example with `S=I/1000`. These examples demonstrate a capability, not its
benefit on actual RN field covariances.

For three possibly dependent Hessian blocks A,B,C under one common law,
Hölder gives `E|ABC| <= (EA4 EB4)^(1/4) (EC2)^(1/2)`. Taking fourth powers
keeps the comparison rational. Any of the three assignments of the second
moment is valid, so their minimum is valid. There is no need for independence
between blocks. The historical fourth-moment substitution is retained only
as a falsified inference, never an enabled strategy. Likewise Cauchy–Schwarz
requires `(E[|D| 1_A])² <= E[D²] P(A)`, not `E[D²] P(A)²`.

## Applicability, memory and evaluation

The callable wrapper checks source bytes, dimension, coordinate order,
normalization, the conditioning-law identifier, the complete mark domain
and the evidence tier against the expected context. Mean and covariance
must identify the same conditional law. Conditioning changes both; see
[GPML Appendix A, A.6](https://gaussianprocess.org/gpml/chapters/RWA.pdf).
The result fingerprints **all** numerical law inputs. A reused name cannot
retrieve a stale polynomial. The interface checks consistency of declared
provenance; it does not authenticate that a caller's law describes the field.
Retirement records concern the invalid inferences, not names of otherwise
valid mathematical functions.

The pre-implementation plan reserves development IDs 0–3 and evaluation IDs
100–115, with five marks and two moment degrees. It called the latter held-out,
but a read-only context-review pass found a material split error: the generator
has period 35, so IDs 105–108 duplicate development inputs 0–3 exactly.
The corrected report retains all measured cases, explicitly rejects the
held-out label, and checks overlap using a canonical numerical-input identity
that excludes case IDs and context labels. It is a preselected synthetic
evaluation with four disclosed development duplicates. The final report compares
fresh computation at each mark with reuse of an affine polynomial, under
the same explicit one-million-unit budget per arm per case. All exact
values agree. Total counted costs are 64,896 for recomputation and 38,245
for reuse in this finite harness. Acquisition of the polynomials, cache
accesses and instrumented applicability checks are charged. The report
states the precise cost model and its clarification: division and PSD
checks were omitted from the short initial description and are charged
in both arms. Bit complexity, hashing, parsing, external oracle arithmetic
and wall time are excluded equally. This is not a net research-efficiency
claim, and the canonical registry's utility remains UNMEASURED.

Reproduce with `python tools/rn_moment_report.py --check
research/rn/candidates/affine_moments_20260920.json`. Tests include refusal
of stale sources, changed units/domains, wrong dimension/order, mixed laws,
unsupported evidence tiers and retired inference variants.

## Scope remaining

There is no RN field identification, interval-valued covariance family,
spatial cover, remote normalizer proof, weighted-Palm transfer or all-small-r
bound here. In particular, `E0[W S]/E0[W]` cannot be replaced by `E0[S]`.
The D1 v2.2 premises, including the uniform RN obligation, stay as the source
register states. Historical August K3 failures supply negative controls;
they do not override the later September LPW lower construction. The 2D
upper/lower and 3D lifetime tracks remain separate. No original prize is
closed, no independent review is earned, and no status is promoted.
