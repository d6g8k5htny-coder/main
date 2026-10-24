# Predeclared exact cusp-population falsification contract

Draft prepared outside the repository by OpenAI/Codex
`population_contract_design`, under integration owner `root`. The mathematical
author of the consumed two-control source is `universality_review`; this agent
contributes the finite test contract and has read the source and supplied route.
All are source-exposed members of the same provider/team. Organizational
independence credit is zero. This is a deterministic interval experiment with
no RNG, seed, numerical field draw, blind review or new theorem admission.

## Source and object frozen before execution

Consume the actual 782-line
`experiments/universality/two_parameter_cusp_population/PROOF.md`, SHA256
`9e8caa52b6448e23e465e646ca98d6e42e0cd71442ad4118b59145ee849e1326`.
The contract designer freshly reproduced this identity and read actual lines
1-260, 261-520 and 521-782. Historical tests and reviews earn no execution
credit in this run. Record this contract's hash and all consumed Python source
hashes in the run before evaluating the population. If these frozen inputs
change during execution, retain the attempt as `INCONCLUSIVE_SOURCE_DRIFT`.

Freeze these existing interval-source hashes as well:

| Repository source | SHA256 |
| --- | --- |
| `experiments/periodic_h0/finite_certificate.py` | `0674a2621dff6c040cbb24cc0b6e6df7847541ebc171ebdb28070afa05e04e6b` |
| `experiments/periodic_h0/gaussian_tail.py` | `07e55a62ed0bb6da6a9395762e2e0c033126f83b46abd48db4c24df710779db8` |
| `experiments/periodic_h0/mean_adaptive_confidence.py` | `d83b4d530d00aafab1ebf5aa262869911db94d3b088309df3d878089285922a5` |
| `experiments/periodic_h0/finite_count_loss.py` (imported dependency) | `85fca300f2dcb1af6102a3d82205e80cf5cb7f345b945186f2deb14d6b7f4317` |

Before the full run, freeze a manifest of the four new source files below,
this proof and these four interval dependencies. Record the actual hashes in
`RUN.json` and check them again at the end. The contract hash lives in the
manifest, avoiding a self-referential hash in this document. No claim is made
that the not-yet-written implementation already has a frozen identity.

Fix spatial dimension `d=2`, period `L=1`, `w_i=3^-i`, unnormalized field
height, the superlevel H0 elder rule, and exclusion of the essential class.
Haar translation is part of the analytic field law and is integrated out; it
is not simulated. Fix `A=2^-16`, `u0=3*A^2`, `v0=2*A^3`, and full symmetric
rectangle density `c0=1/(24*A^5)`. Set counts to zero on the null discriminant
and tie controls, as the consumed proof does. Full-rectangle probabilities
must retain their mass; do not condition on the existence of a finite bar.

For the positive-control chart put `a=A*z`, `h=a*theta`, `c=-a`. Its three
roots are `c-h`, `c+h`, `-2*c`; `u=3*a^2+h^2`,
`v=2*a*(a^2-h^2)`. Freeze `z>0`, `0<theta<1`,
`z^2*(3+theta^2)<3`. The selected lifetime is `ell=4*a*h^3` and
`tau=ell/A^4=4*z^4*theta^3`. Combining both control signs gives measure
`(1/3)*z^4*theta*(9-theta^2) dz dtheta`. The whole-rectangle finite-bar
mass is exactly `1/5`; normalized lifetime support is `0<tau<9/4`.

At every rational root-control witness, evaluate the polynomial at all three
critical roots and determine the younger outer maximum by comparing its
actual height to the other maximum. The positive chart must have
`P(2*a)-P(-a-h)=a^4*(1-theta)*(3+theta)^3/4 > 0` and selected gap
`P(-a-h)-P(-a+h)=4*a*h^3`. Reflection must exchange the younger side and
preserve the gap. This finite critical-height selector consumes the source's
global-selection proof; it is not an independent whole-field H0 algorithm.

## Quantities and two separately derived interval routes

Define `m(tau)=P(0<ell/A^4<=tau)`. The direct route integrates the control
measure, with no call to the CDF primitive:

```
m(tau) = (1/15) integral_0^1 theta*(9-theta^2) *
         min((3/(3+theta^2))^(5/2),
             (tau/(4*theta^3))^(5/4)) dtheta.
```

For `0<tau<9/4`, the separate closed-CDF route brackets the unique
`s in (0,1)` satisfying `36*s^3=tau*(3+s^2)^2` and encloses

```
(1/15) * [3 - 3*(1-s^2)/(1+s^2/3)^(3/2)
 + (tau/4)^(5/4) * ((36/7)*(s^(-7/4)-1)
                    - 4*(1-s^(1/4)))].
```

Set `m(0)=0` and `m(tau)=1/5` for `tau>=9/4`. This saturation also checks
the sign multiplicity, rectangle normalization and support convention.

Use only `fractions.Fraction` for mathematical values and interval endpoints.
Reuse the existing `finite_certificate.sqrt_bounds`,
`gaussian_tail.exp_neg_bounds`, and `mean_adaptive_confidence.log_bounds`
with their actual source identities. Any required cube-root enclosure uses
rational bisection of a monotone cubic. Nested outward square-root enclosures
supply quarter powers; do not use float powers, fitted slopes or numerical
quadrature to certify an endpoint. `pi_bounds` is available for separately
scoped physical-coordinate work, but this contract does not evaluate physical
positions or claim that it used that function.
The two routes have separate analytic derivations but may share the root
bracket and outward elementary primitives. Their agreement earns no blindness,
organizational independence or independence of the shared interval engine.

## Fixed schedule and numerical budgets

Freeze eight rows, `j=1,...,8`, with `tau_j=2^(-3*j)` and `n_j=4^j`.
Then `n_j*tau_j^(2/3)=1` exactly. For each row report rational intervals for
both routes' `m(tau_j)`, both normalized masses `n_j*m(tau_j)`, their
intersection, and the exact iid void probability
`V_j=(1-m(tau_j))^n_j`. The iid-copy law is an analytic premise; no iid
samples are produced. Enclose `V_j` using outward log/exponential arithmetic,
or exact rational exponentiation if it stays within the same fixed budget.

Bracket the split root in a dyadic interval of width at most `2^-96` by
96-step rational bisection of `g(r)=36*r^3/(3+r^2)^2`. Use outward elementary
operations with at least 128 bits and enforce total arithmetic inflation
at most `2^-22` for each normalized direct integral. For these rows the
split has `rhi<1/2`. A failed bracket, positivity or arithmetic-width gate
is inconclusive; no uncertified endpoint enters an acceptance predicate.
The `2^-22` arithmetic gate is an actual accumulated exact rational bound:
sum the positive quadrature weights times point/coefficient interval widths
for both the lower and upper quadrature contributions, including root-power
and coefficient uncertainty. Record that sum separately from discretization
and root-sliver widths and replay-check it. A precision setting alone is not
evidence for this gate. A sufficient auxiliary check is `rhi>2^-9`, giving
combined lower/upper quadrature weights at most `2/rhi<1024`; intervals of
full width at most `2^-40` at every weighted evaluation then contribute less
than `2^-30`. Actual accumulated arithmetic width remains the acceptance gate.

Direct core: put `theta=rlo*x` on `x in [0,1]`; use exactly `2^16`
Darboux panels. The normalized integrand is increasing, with range at most
`3/10`, since `n_j*rlo^2<=1/2`. Its enclosure width is at most
`3/(10*2^16)`. Record the monotonicity and range gates.

Direct tail: put `theta=rhi*t`. On each octave
`[2^k,min(2^(k+1),1/rhi)]`, use exactly 512 panels with midpoint lower and
trapezoid upper sums. Its normalized integrand is
`C*(9*t^(-11/4)-rhi^2*t^(-3/4))`, where
`C=n_j*tau_j^(5/4)/(15*4^(5/4))*rhi^(-7/4)<=1/30`.
It is convex; `f''<=99/32*t^(-19/4)`. The summed discretization width is
at most `297/(512*512^2)`. Enclose the intervening root sliver by
`[0,(3/5)*n_j*(rhi-rlo)]`. Total direct normalized width must be
strictly less than `2^-17`.

Declare at most 80,000 direct integrand evaluations per row, with midpoint
and endpoint enclosure computations counted explicitly. Compute each unique
node once and reuse its enclosure in adjacent panels: a cached use is not
another evaluation, while any actual repeated computation counts again.
At row 8 the plan uses at most `65,537+9*(513+512)=74,762` evaluations.
This function-evaluation count is separate from arithmetic inflation, which
counts both weighted lower/upper uses of every enclosure. The closed-CDF
normalized enclosure must have width at most `2^-16`. The direct/CDF
intervals must overlap. The reported void enclosure and target Poisson
enclosure must each have full width at most `2^-16`.

No adaptive extra rows, changed thresholds, increased panels, increased
root iterations or relaxed acceptance threshold belong to this run. Any
exhausted evaluation/precision budget is `INCONCLUSIVE_BUDGET`, preserving
partial intervals and counts. A successor contract may predeclare a changed
budget and link this result; it must not relabel the failed attempt as PASS.
The full eight-row runner has a 900-second wall-clock budget, enforced with
a monotonic timer. Runtime exhaustion is `INCONCLUSIVE_BUDGET`; record the
completed work and elapsed time without calling that attempt a full run.

## Predictions and predeclared falsification predicates

Let `B=9/(28*cuberoot(2))`, enclosed by rational bisection, and
`E(tau)=(9/8)*tau^(2/3)+tau^(7/12)`. All eight rows test the prediction
`abs(m(tau)/tau^(2/3)-B)<=E(tau)`. Record the complete predicted interval,
the observed certified interval and their distance; no regression estimates
replace this declared coefficient and error envelope.
For certified intervals `B in [Blo,Bhi]`, `E in [Elo,Ehi]` and normalized
mass `Lambda in [Llo,Lhi]`, PASS requires containment in the common inner
envelope: `Llo>=Bhi-Elo` and `Lhi<=Blo+Elo`. A certified falsifier requires
disjointness from the outer envelope `[Blo-Ehi,Bhi+Ehi]`. Neither containment
in that outer interval nor interval overlap alone proves the prediction.

At `j=8`, `E(tau_8)<=41/524288`. Verify the mass gate
`n_8*m(tau_8)<1/3`. It gives
`abs((1-m)^n_8-exp(-n_8*m))<1/(16*n_8)=1/1048576`.
Let the target be `exp(-B)`. The declared endpoint error allowance is

```
41/524288 + 1/1048576 + 2*(2^-16)
  = 115/1048576 < 2^-13.
```

The row-8 acceptance predicate is the explicit rational endpoint test
`max(abs(Vlo-target_hi),abs(Vhi-target_lo))<2^-13`.
Interval overlap alone is not enough. If certified intervals have minimum
distance greater than `2^-13`, the declared finite void prediction is
`FALSIFIED`. If intervals straddle that boundary or fail their width gate,
the result is `INCONCLUSIVE_PRECISION`.

If a row's certified normalized-mass interval is disjoint from the outer
prediction envelope, label that coefficient prediction `FALSIFIED`
after source, soundness and negative-control gates pass. Freeze new-lemma
work on the affected assertion and retain the counterinterval. If the
observed interval is neither disjoint from the outer envelope nor contained
in the common inner envelope, report `INCONCLUSIVE_PRECISION`. The explicit
inner endpoint inequalities yield that row's PASS.

Disjoint CDF and direct intervals are `FAIL_ROUTE_CONSISTENCY`, not a PASS
for either route. Invalid input, a failed arithmetic guarantee or an
undetected negative control is `FAIL_IMPLEMENTATION`. Such results provide
no certified scientific counterexample until their soundness issue is
resolved. A PASS means this fixed non-Gaussian population survived these
eight declared coefficient tests and the one declared void test. It does
not test general Gaussian one-third universality, all A3 populations, Hk,
same-field spatial Poisson statistics or scientific acceptance.

## Required negative controls and independent scope statuses

Freeze exact controls `a=A/4` with `theta=1/4,1/2,3/4` and their reflected
signs. They check root stationarity, order, Hessian signs, actual elder
selection and the displayed gap. Boundary controls `theta=0,1,3` must be
classified as discriminant or tie and excluded under the declared convention.
The adversarial control `a=A/4,theta=2` has ordered roots `(-3*a,a,2*a)`
and negative `v`. All these fixtures lie inside the frozen control rectangle.
The actual younger-right lifetime divided by `a^4` is `3/4`; the fixed-left
formula divided by `a^4` falsely gives `32`. The critical-height selector
must reject that wrong formula. These fixtures still earn no independent
whole-field H0 selector execution credit.
Also detect a missing factor two by comparing its total mass `1/10` with
the exact `1/5`, and reject bar-conditioned mass `1` as a rectangle mass.
Reject non-Fraction mathematical inputs and reversed/nonenclosing intervals.

Check rational analytic guard witnesses separately: `R=1/8`, `r1=1/32`,
`a*=r1/4`, `delta=r1^4/64`, `B0<=r1^2/2`, `D0<=r1`,
`B1<=33*r1`, `D1<=65`, `g*>=r1^3/64`. Use the proposed C-infinity
cutoff defined by `q(t)=exp(-1/t)` for `t>0` and zero otherwise,
`psi(t)=q(1-t)/(q(t)+q(1-t))`, flat beyond 0 and 1, with
`abs(psi')<=8`; the radial cutoff gives `abs(grad chi)<=64/r1`.
The monotone surgery uses
`c(r)=r/4+(1-r/4)*(1-psi((r-r1^2)/(R^2/2-r1^2)))`.
These analytic premises require substantive review; exact arithmetic only
checks their consequent rational inequalities. In particular

```
B1*u0+D1*v0 <= 101441/2^47 < 2^-22 <= g*/2,
B0*u0+D0*v0 <= 1537/2^52 < 2^-28 = delta/4.
```

Also check `R^2<w_2`, rectangle root-containment inequalities and wedge
coverage. Their scope status is `PASS_RATIONAL_GUARDS` only after actual
evaluation; the analytic cutoff derivative bounds and global selector
retain their separate review status. Constructing/evaluating the complete
surgery field and running an independent global H0 selector remain
`NOT_RUN`. Existing Fourier/grid/nodal APIs do not accept this surgery field.
The run must display these statuses even when exact-population tests pass.

## Minimal delivery and frozen-run disposition

The claimed implementation scope is exactly four source paths in
`experiments/universality/two_parameter_cusp_population/`:
`CONTROLLED_FALSIFICATION.md`, `population_certificate.py`,
`test_population_certificate.py`, `run_population_certificate.py`, and
three result paths in `experiments/universality/results/exact_population_1/`:
`RUN.json`, `CERTIFICATE.json`, `RESULTS.md`.

`RUN.json` records UTC start/end, command and exit code, frozen contract and
consumed source hashes, runtime, actual route evaluations, declared parameters,
performer/exposure and overall disposition. `CERTIFICATE.json` contains every
rational endpoint, eight rows, route/width/budget gates, predicted envelopes,
row-8 endpoint comparison, control outcomes and separate scope statuses.
`RESULTS.md` presents the actual intervals, failures or inconclusive gates and
the precise surviving/falsified scope. No field draw, selector execution,
independence, historical replay or broader mathematical closure may be inferred
from these artifacts. Preserve a failed original attempt before any successor.

Before a full run, freeze and hash the contract, obtain fresh substantive
source review of the new interval methods, and run focused tests that target
soundness and the specified negative controls. Run one full eight-row command
on the frozen candidate; changes require a successor source identity and
affected revalidation. Integration and hosted checks are engineering custody,
not proof or scientific promotion. The standing 8 October owner authorization
permits this ordinary project work without renewed owner approval.
