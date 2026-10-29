# D2 Schur and endpoint positivity: exact formal scope

**Scientific effect NONE.** Dylan Roy — delegated AI work. Actual author OpenAI /
GPT-6 Astra Pro, session `lean-d2-schur-expansion-20261005`. This is a continuation
of that provider's informal derivation, not a nonauthor or blind review.
Alignment remains PENDING_INDEPENDENT_REVIEW; source progress is not an observed
kernel run. No source record, scientific register, prize or Boolean is advanced.

## Exact source interface

At Math- commit `ec61d0a4e1d7c29554c0af93fca983092327b618`, consume
`reviews/side24_d2_certified_small_L_claude_20261005/NOTE.md` section2,
Git blob `852f7403a8a06a1ef061e363df9502ae16f7a341`, SHA256
`407828facb8dde26a036a08a5b3978eb74d6d8a508c68e2758c291e16e911588`.
The exact Schur cancellation and endpoint observation were recorded separately
in Math-#302 comment6005199919. That mutable comment is attribution/context;
the complete formulas and Lean definitions here are the bound proof input.

The variables are arbitrary real scalars `m2`, `m4`, `m6`, `q`. The intended
interpretation is raw one-coordinate spectral moments and
`q=cos(theta)^2*sin(theta)^2`. The interpretation is NOT constructed in Lean.
The source covariance uses the raw spectral Gram of quadratic monomials, NOT
centered spectral covariance and NOT conditioning on the field height.

Set `k4=m4-3*m2^2`. Definitions in D2Schur.lean are

```
alpha = m4 - 2*k4*q
gamma = m2^2 + 2*k4*q
D = m2^2*m4 + k4*(m4+m2^2)*q
N = m2^2*(m4^2-m2^4)
S = alpha - (gamma^3+(2*gamma+alpha)*k4^2*q*(1-4*q))/D
Delta = m6-m4^2/m2
T = (1-4*q)*Delta + 4*q*(Delta+9*m2*(m4-m2^2))/4
```

The first14 declarations prove identities/inequalities of these expressions.
The last2 are positive reference and singular-boundary controls. No additional
rejection-run count is claimed for those2 ordinary audited declarations.

## Declaration-by-declaration mapping

| Declaration | Statement and hypotheses |
|---|---|
| d2_schur_numerator | `alpha*D-gamma^3-(2gamma+alpha)*k4^2*q*(1-4q)=N`; no denominator hypothesis |
| d2_det_axis | `D(0)=m2^2*m4` |
| d2_det_diagonal | `D(1/4)=(m4-m2^2)*(m4+3m2^2)/4` |
| d2_det_interpolation | D is the convex interpolation of those endpoints when q is in the intended interval; the equality holds for every real q |
| d2_schur_ratio | `S=N/D` requires `D!=0` |
| d2_numerator_pos | `N>0` from `m2>0` and `m4>m2^2` |
| d2_det_pos | `D>0` from those moment conditions and `0<=q<=1/4` |
| d2_schur_pos | `S>0` under the same conditions |
| d2_det_le_endpoint_max | D at `0<=q<=1/4` is at most the larger endpoint, without moment positivity hypotheses |
| d2_schur_endpoint_lower | `N/max(D(0),D(1/4))<=S` under strict moment conditions and the q interval |
| d2_tau_cumulant_eq | The original sixth/fourth cumulant formula equals T, with `m2!=0` |
| d2_tau_axis | `T(0)=Delta` |
| d2_tau_diagonal | `T(1/4)=(Delta+9*m2*(m4-m2^2))/4` |
| d2_tau_pos | T is positive from `m2>0`, `m4>m2^2`, `Delta>0`, `0<=q<=1/4` |
| d2_gaussian_control | At `m2=1,m4=3,m6=15`, `D=3`, `S=8/3`, `T=6`, for every real q |
| d2_degenerate_control | At `m2=m4=m6=1,q=1/4`, D=0, S=2 but N/D=0; axis T=0. The unguarded quotient identity and strict positivity at the moment boundary are false |

The positive lower-bound value follows from N>0 and positive endpoint maximum;
this is not a computation of its value for any period. Empty/nonreal/negative
moment inputs are not silently treated as probability laws. The mathematical
proofs are for arbitrary reals satisfying the written hypotheses; rational
Python controls are only falsification and implementation checks.

## Integration and verification

The existing47 theorem proof files, gate.py, pinned Lean4.34.1, mathlib commit
`d13f23b723b8a846827a245b89c10fc7d3f11612`, dependencies, old rejection controls,
original archives and historical alignment records remain unchanged.
The new16 targets extend the exact inventory to63. The predecessor weight test
checks its original slice40:47 instead of claiming to be the terminal suffix;
this new test checks the exact remaining suffix and exact total63. No old
weight inequality or test case is removed. The executable gate still requires
all declared sources/hashes and transitive axiom reports and all5 real negative
controls. The full type log remains part of the execution evidence.

```
python -B -S -m unittest discover -s formal/tests -v
python -B -O -S -m unittest discover -s formal/tests -v
python -B -S formal/gate.py
python -B -S formal/gate.py --execute
```

A source-only result does not run Lean. This author environment has no Lean
installation and cannot resolve GitHub DNS. Hosted build/recheck/axiom/negative
execution on the actual candidate is therefore required before a kernel claim.
Nonauthor exact-scope review is also required, independently of execution.

## Explicit exclusions and preserved obligations

Not formalized: construction or covariance identification of the random field;
actual matrix inverse/Schur complement or Gram rotation; derivation/realizability
of the moment assumptions; `q`'s trigonometric representation; finite-L tails;
interval arithmetic or quadrature; coefficient formula(15.2), Gamma normalization
or lifetime laws; higher dimensions; persistence matching; CapI4 transport;
any whole-package independent alignment or the earlier #281 lineage issue.
The source-bound numerical implementation remains unchanged. These algebraic
companions may assist its future review but do not retroactively validate it.
