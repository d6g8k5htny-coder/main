# LPW R05 full-tail replay and quadratic covariance modulus — candidate v1

The source's linear covariance modulus is reproduced with exact rational
intervals, and its ten Hermite profiles give a stronger quadratic modulus:

\[
\|\Gamma_r-\Gamma_0\|_{op}\leq\|\Gamma_r-\Gamma_0\|_F
 < {451\over250}\,r^2,\qquad 0<r\leq 10^{-5}.
\]

Conditional on the inherited endpoint bound
`lambda_min(Gamma_0) >= 31/250`, this yields the simple uniform floor
`lambda_min(Gamma_r) >= 0.1239999998196` on the closed interval
`0 <= r <= 1/100000`. This statement applies to the covariance model defined
by the exact ten source symbols below. Its identification with the program's
six-pin transform is inherited, not independently reconstructed here.

This is source-exposed author/coauthor work with zero organizational
independence. It changes no scientific status. It does not establish
`K <= 9432`, a conditional density or field norm, an LPW probability
headline, the endpoint eigenfloor, a matching upper theorem, or any prize
closure. It is ordinary mathematics plus exact finite replay, not a proof
assistant formalization. R05-Q2 has more than this covariance calculation:
the finite two-dimensional amplitude sums are deliberately not replayed.

## Source custody and coordination

The checker verifies complete raw bytes before computation and never executes
either source program. Source paths are beneath
`drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/LPW — RAW ARCHIVE INTAKE R05/`.

| Source | Drive ID | Bytes | SHA-256 |
|---|---|---:|---|
| `03_RAYLEIGH_REPAIR_AND_INTERVAL_CERTIFICATE.md` | `1ngaO6hzeIXdCYWMwKl8JYPPZTeryGFTx` | 9628 | `840c75a7825c67b8d99a536394beb18976475fe9bd0e474ed5f80c375004ec81` |
| `checks/interval_repair.py` | `1cdE15_2dd0VLHJnDGcFQ5RDQJhTkXfr-` | 6843 | `b37150f0eb79ff5d11b9e8b60f80afd94c677f8150aa701573ce24b337f24069` |
| `raw_reports/lpw_constant.py` | `1fRWIm7IsufB2xkrCxaExFp2TFdcd4oaG` | 25380 | `e258322cfbb71dbb8665c2d0c0517399dca5ca5913e9ddf70eb7247ba74d3035` |

R17 target `PARALLEL-LPW-TAIL-PROFILE-20260920-v1`, claim
`f9be87ff-51ff-4c49-961a-dfd821b8b410`, Work Events row 67, was confirmed
before implementation. The repository is read only for this worker. All
outputs belong to the root task's scratch delivery; the existing repository
task remains the sole integration writer. The result separately records the
source root and interval implementation root, including hashes of all three
interval package files. `--repo` overrides the source root only.

## Full-lattice moments, with no planar substitution

Put `h=pi/12`, `a=h^2/2`, `Z1=sum_{n in Z} exp(-a*n^2)`. The full normalized
mass at `(n,m)` is `exp(-a*(n^2+m^2))/Z1^2`; frequency is `(hn,hm)`.
For nonnegative integer `p`, let `b_p(n)=exp(-a*n^2)*(hn)^p`, `n>=1`.
Then

\[
{b_p(n+1)\over b_p(n)}
=e^{-a(2n+1)}(1+1/n)^p=\rho_p(n).
\]

Both factors are nonincreasing for `n>=1`, and the exponential is strictly
decreasing. More explicitly,
`rho_p(n+1)/rho_p(n) = exp(-2a)*(n*(n+2)/(n+1)^2)^p < 1`.
Exact interval evaluation at `n0=71` proves `rho_p(n0)<1/2` for all
`p=0,...,12`. Consequently the entire omitted two-sided tail satisfies

\[
\sum_{|n|\ge71}e^{-an^2}(h|n|)^p
\le {2 e^{-a71^2}(71h)^p\over1-\rho_p(71)}.
\]

The finite sums use `|n|<=70`. The full normalization has interval
`[Ztr.lo, (Ztr+T0).hi]`. For positive even `p`, if `Qtr` is the finite
moment numerator, the normalized moment is enclosed by
`[(Qtr/Zall).lo, ((Qtr+Tp)/Ztr).hi]`. The smaller normalization belongs in
the upper quotient. The exact zeroth normalized moment is one. Odd absolute
moments use Cauchy–Schwarz between neighboring enclosing even moments.
Neither `1,3,15` nor another planar Gaussian moment is substituted.

The separate two-dimensional amplitude-tail contribution is also enclosed.
Shell `max(|n|,|m|)=s` has `8s` points, squared radius at least `s^2`, and
frequency norm at most `h*sqrt(2)*s`. For `p=3,4`, its unnormalized majorant is
`B_p(s)=8s*exp(-a*s^2/2)*(1+h*sqrt(2)*s)^p`. The consecutive ratio is

\[
{s+1\over s}\,e^{-a(2s+1)/2}
\left({1+c(s+1)\over1+cs}\right)^p,\quad c=h\sqrt2>0.
\]

Every factor decreases for `s>=1`; the last is `1+c/(1+cs)`. Exact evaluation
gives ratio `<1` at 71, so the full missed sum is at most
`B_p(71)/(1-ratio(71))`. Dividing by the lower endpoint of `Ztr` preserves
an upper bound for the normalized square-root masses. This verifies these
tails, not the uncomputed finite two-dimensional amplitude sum.

## The ten exact profiles and the new slope bounds

The source assigns derivative powers
`(0,0),(1,0),(0,1),(2,0),(1,1),(3,0),(0,2),(2,1),(1,2),(0,3)`
to `g0,...,g9`, respectively. Write `theta=k1*r/2` and `sinc(0)=1`.

| i | `g_i(theta)` | Uniform magnitude bound `P_i` on `r<=R` | `d_i` in `|g_i'(theta)|<=d_i*|theta|` |
|---:|---|---|---:|
| 0 | `cos(theta)+(theta/2)sin(theta)` | `1+k1^2*R^2/8` | 1 |
| 1 | `(3/2)sinc(theta)-(1/2)cos(theta)` | 2 | 1 |
| 2 | `cos(theta)` | 1 | 1 |
| 3 | `-(1/2)sinc(theta)` | 1/2 | 1/6 |
| 4 | `-sinc(theta)` | 1 | 1/3 |
| 5 | `(theta*cos(theta)-sin(theta))/(2*theta^3)` | 1/6 | 1/30 |
| 6 | -1 | 1 | 0 |
| 7 | -1/2 | 1/2 | 0 |
| 8 | -1/2 | 1/2 | 0 |
| 9 | -1/6 | 1/6 | 0 |

These are global in real `theta`; the origin singularities are removable.
The elementary identities

\[
\operatorname{sinc}(\theta)=\int_0^1\cos(t\theta)\,dt,
\qquad
g_5(\theta)=-\tfrac12\int_0^1t^2\operatorname{sinc}(t\theta)\,dt
\]

hold by integration for nonzero theta and by continuity at zero. From
`|sin u|<=|u|`, differentiating the first under its compact integral gives
`|sinc'(theta)| <= |theta|*integral_0^1 t^2 dt = |theta|/3`.
Differentiating the second gives
`|g5'(theta)| <= (|theta|/6)*integral_0^1 t^4 dt = |theta|/30`.
Also `|g5| <= (1/2)*integral_0^1 t^2 dt = 1/6`.

For g0, the derivative is `(theta*cos(theta)-sin(theta))/2`, of magnitude
at most `|theta|`; its magnitude is at most `1+theta^2/2`. For g1,
`(3/2)*(|theta|/3)+(1/2)*|theta|=|theta|` bounds its derivative. The remaining
rows follow directly from the same integral and elementary sine bounds.
Stronger cancellations exist for g0 and g1; they are unnecessary here.

The source's old linear calculation used `|g5|<=1/4`, `|g5'|<=5/4` and
global constant derivative bounds. Its exact coefficient is replayed as
`L1<19.072`; the result JSON retains every entry and every interval endpoint.

## Quadratic covariance integration and spectral floor

The source Fourier symbols include the derivative phase. Covariance entries
vanish if total derivative parities differ; symmetric k2 integration also
annuls an odd k2 power. These are exactly the two zero filters in the
implementation. Signs of retained entries do not matter for absolute bounds.

For a retained `(i,j)` entry with summed powers `(x,y)`, differentiation
contributes `(k1/2)*(g_i' g_j+g_i g_j')`. The new bounds imply

\[
|\partial_r\Gamma_{ij}(r)|
\le {r\over4}\,\mathbb E\bigl[|k_1|^{x+2}|k_2|^y
(d_iP_j+P_id_j)\bigr].
\]

The profile magnitude polynomials use the fixed endpoint `R`, so integrating
from zero to r gives `|Gamma_ij(r)-Gamma_ij(0)| <= C_ij*r^2`, where

\[
C_{ij}={1\over8}\mathbb E\bigl[|k_1|^{x+2}|k_2|^y
(d_iP_j+P_id_j)\bigr].
\]

The `1/8` is essential: one half comes from `theta=k1*r/2`, another from
the derivative chain rule, and another from `integral_0^r t dt=r^2/2`.
The product spectral law factors the moments in k1 and k2. Finite polynomial
majorants against Gaussian decay justify absolute summation and
differentiation by dominated convergence on the entire compact r interval.
Full moment enclosures through order 12 more than cover every required order.

The replay encloses `sqrt(sum_ij C_ij^2)` with upper endpoint approximately
`1.8037985575223484` (display only) and proves it is strictly below `451/250`.
At `R=1/100000`, the old replay budget is approximately `0.00019071156`,
while the new budget is approximately `0.000000000180379856`. Their exact
ratio exceeds one million. This compares two certified upper budgets, not
the unknown actual error or an empirical convergence rate.

Weyl's inequality and the imported endpoint lower bound give the claimed
floor `31/250-(451/250)/10^10=0.1239999998196`. Any principal block inherits
this floor. For the Schur complement, conditional on the source covariance
being the positive definite block matrix, the standard identity
`y^T S y=min_x (x,y)^T Gamma (x,y)` gives the same lower bound. No regression
mean or conditional field norm has been recalculated.

## Reproduce and falsify

From the repository root, with Python 3.11, these commands run replay,
certificate comparison, tests, normal/optimized equivalence and exact
analytic counterexamples:

```sh
python -B research/parallel/lpw/lpw_modulus.py
python -B -O research/parallel/lpw/lpw_modulus.py
python -B -m pytest -q -p no:cacheprovider tests/test_lpw_modulus.py
```

The portable integration derives the repository root from the module path and
verifies `candidate.json` by default. It removes only the two machine-specific
root-path metadata fields from the original report; mathematical data, source
identities, retained imports and scope are unchanged. `--repo` still selects
the source root only, while interval implementation identities come from the
checkout containing this module. The original scratch-delivery and read-only
worker statements above record the source author's historical scope.

The negative controls reject understated `19` and `1.803` coefficient
ceilings through actual CLI overrides, normal and `-O`; altered raw source;
omitted tails; moved tail start; reversed normalization; discarded valid
parity entries; halved quadratic integration factor; unsupported `1/31`
g5 slope; authority/independence changes; and a Boolean masquerading as
integer zero. The parser also refuses duplicate keys and noninteger JSON
numeric literals. Exact semantic counterexamples at theta=1/10 refute g5
slope `1/31` and sinc slope `1/4`; a single cosine covariance refutes halving
the integration coefficient. These point counterexamples falsify proposed
bounds; they are not being used to prove the uniform positive result.

The checker uses explicit exceptions, not assert, for acceptance decisions.
It does not trust supplied interval endpoints: a supplied data-only JSON
certificate must match a fresh typed reconstruction of every field. Source
hash matching is custody only. The interval library is shared with the
repository, and the implementation is exposed to the original derivation.
No independent-code or blind-review credit is claimed.
