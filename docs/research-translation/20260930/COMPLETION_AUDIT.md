# What remains to close, and for which theorem

[Manuscript](MANUSCRIPT.md) · [Full source crosswalk](CROSSWALK.md) · [Proof appendices](APPENDICES.md) · [Source comparison](COMPLETION_SOURCES.json)

**Completion-directive audit, 1 October 2026.** The useful instruction is to
finish a specified theorem through its actual dependencies. It does not follow
that every open item in the research program is a dependency of that theorem.
The fixed planar manuscript already states an actual persistence intensity,
using a global elder mark. This audit makes its selection argument and error
accounting explicit. It neither promotes a scientific status nor certifies the
whole research program as complete.

The eight principal proof/repair/review sources in APPENDIX_SOURCES.json are
byte-identical at the manuscript cut `fa2e9909d8cca40d38b7dc2eadda6766bd0788e1`
and the checked Math- cut `124c37d9218120245098df453cf88d3e54b8c9ca`.
The current proof index still records the regional shrinking-witness node as
open. Both statements can hold: the manuscript's route does not consume that
regional node. Exact paths, blobs, sizes and SHA256 digests are in the source
comparison. This is a bounded dependency audit, not a new full-depth review of
every imported proof or a replacement for the existing status graph.

## 1. One fixed target

Call the existing manuscript M1–M5 statement **T24-planar**. This name identifies
the assembly target; it does not rename a source theorem.

Let `X=R²/(24 Z²)`, with area 576, and let `f` be the centered stationary
Gaussian field with covariance

```text
K24(z) = sum[n in Z²] exp(-|z+24n|²/2)
         / sum[n in Z²] exp(-|24n|²/2).
```

Use its smooth version, ordinary superlevel H₀ persistence, and decreasing
level. Count finite bars only; exclude the single essential component.
Index means the number of negative Hessian eigenvalues. A maximum has index
two; an index-one saddle is counted only when it merges distinct components.
The higher-born component survives. The maximum whose bar dies is the
**younger** maximum, not the surviving elder maximum.

For Borel `A subset (0,infinity)`, the observable is

```text
mu(A) = (1/576) E sum[finite bars i] 1{birth_i-death_i in A}.
```

The target is a version of its density satisfying, for some finite `C` and
positive `ell_*`,

```text
|nu(ell) - c_(2,24) ell^(-1/3)| <= C,       0 < ell <= ell_*;
mu((0,t]) = (3/2)c_(2,24)t^(2/3) + O(t).
```

The coefficient is the exact torus contact integral P (15.2), specialized to
`d=2,L=24`, with the SIDE24 enclosure

```text
0.07340691930603427103 < c_(2,24) < 0.07340691930603427104.
```

This is an expected intensity per area, not an almost-sure histogram law or
a per-realization probability distribution. All births, directions and spatial
separations are included. `L=24` is fixed before lifetime tends to zero. No
uniformity in dimension, covariance or volume is asserted. The reference
coefficient is not substituted for the exact torus coefficient. Positivity and
finiteness follow from P's nondegenerate contact law and integrable Gaussian
majorant; SIDE24 supplies the displayed enclosure. No numerical `C` or
`ell_*`, higher homology, simultaneous infinite-volume limit, or new remainder
coefficient is part of this target.

Smoothness, the specific finite-jet rank conditions and almost-sure Morse/
distinct-critical-value properties are inputs proved for this field in P,
not generic assumptions silently imposed on every Gaussian field. Exact
degeneracies are handled there. Near-degenerate layers need the weighted
estimates below, even when the exact degenerate event has probability zero.

## 2. The dependency map and what each edge carries

The labels below refer to the immutable sources in APPENDICES A and the
repaired consumption contract in REC §7. “Imported, scoped” means that the
existing proof and its stated review contract are consumed. It does not create
a new `PROVED` flag, add independence credit or formalize that proof in Lean.

| Node | Inputs and exact conclusion used downstream | Disposition in this assembly |
|---|---|---|
| Model/regression | P §§2–4: smooth full-field law, independent derivative functionals, nonsingular pin/contact observations and finite derivative moments | Imported, scoped |
| Typed normalizer | P §§4–7 + E1 + REC W1: full `Z_r=E_Q W_r`, compact-mark `Z_r asymp r²`, determinant-weighted soft-layer bound | Imported, scoped; no global floor in all marks |
| Global selector | CAP §§1–5 + P §8: good cap implies the actual global elder death equals the pinned saddle | Imported, scoped; all cap hypotheses retained |
| Marked measure | E2 replacing P §9: full-field Borel elder mark in the Kac–Rice identity on separated domains, then exhaustion | Imported, scoped; not continuity of the indicator |
| Radial/contact intensity | P §§10–15 + REC §§6–7: exact Jacobians, domination, coefficient P (15.2) | Imported, scoped |
| All-mark remainder | R §§2–7: target-uniform unnormalized bounds, small-gap split, fixed-far bound; R (R1) | Imported, scoped; uses global selector/marked measure, not a uniform compact-mark probability bound |
| Coefficient evaluation | S24: same P (15.2) coefficient, derivative-covariance/Schur/cone transfer and outward enclosure | Imported arithmetic; parent interpretation remains a separate edge |
| T24-planar | Marked measure + radial/contact intensity + R (R1) + S24 | Existing manuscript statement, unchanged |
| Cumulative law | Integrate the density estimate in T24-planar; `ell^(-1/3)` and bounded remainder are integrable | Consequence; no reverse differentiation |

In particular the selector edge enters the loss bound, the Borel edge identifies
the selected pair measure with the bar measure, and the S24 edge evaluates the
coefficient. None of these edges can be replaced by a numerical fit.

The following are **not consumed by T24-planar**, for the explicit reasons
given. “Not consumed” does not close or supersede them for other theorems.

| Separate object | Why no dependency edge enters this target |
|---|---|
| `math.rn-region.witness-collision` | The selected pair identity counts every bar once directly, while the cap bounds rejected candidates. It does not count or identify a shrinking collection of additional witnesses. |
| D4 fixed-rho/fixed-eta RN count | R §7 uses P §14's unconditioned two-point height-density bound at one fixed separation cutoff, not a shrinking conditional remote-count estimate. |
| D5 growing normalized annulus/count and C6 factorial/cluster laws | R integrates the pair amplitude over all small separations via the exact gap variable and all-mark bounds. It does not replace this integral by a witness count. |
| Historical CH-LIFT/Piece-2/24-jet certificates | R explicitly supplies no original RN/24-jet certificate. Its new contact regression, determinant and density estimates are the named inputs instead. No absent historical carrier is declared recovered. |
| An identified limiting cluster law | A first-moment pair measure with a Borel selector does not need a distributional cluster limit. |
| Numerical asymptotic window or experiment confidence bound | An existential continuum statement does not depend on the finite-cutoff experiment or on a numerical value of its error constant. |

The existing [proof index](https://github.com/d6g8k5htny-coder/Math-/blob/124c37d9218120245098df453cf88d3e54b8c9ca/PROOF_INDEX.md)
and [downstream graph](https://github.com/d6g8k5htny-coder/Math-/blob/124c37d9218120245098df453cf88d3e54b8c9ca/frontiers/downstream_gate_20260925/GRAPH.json)
retain those separate obligations. This map audits one consumer, not their
global retirement.

## 3. The actual pair-to-bar bridge

On the Morse, distinct-critical-value locus, let `C_f` be all ordered
maximum/index-one-saddle pairs with positive height difference. Define
`e_f(M,S)=1` exactly when `S` is the ordinary elder death of `M`. Each finite
H₀ bar has one birth maximum and one killing saddle, and each such selected
ordered pair contributes one finite bar. Thus, before taking expectations,

```text
sum[finite bars i] 1{ell_i in A}
  = sum[(M,S) in C_f] e_f(M,S) 1{f(M)-f(S) in A}.
```

This proves the accounting direction from every bar to a pair. It does **not**
assert that every small bar satisfies a local normal form. Far pairs and cap
failures remain in the identity and are bounded in R.

The converse sufficient event is genuinely global. CAP gives a cap containing
`M` on which `f<=b`, every boundary exit has value at most `s`, and a ridge
path through `S` reaches a point of height greater than `b` with minimum `s`.
Every path to a point above birth must exit that cap. Its maximum possible
minimum is therefore exactly `s`; distinct critical values identify `S`.
The interior bound `f<=b` is essential: the retained
[cap-assumption ablation](cap-assumption-ablation/PROOF.md) shows why boundary
and pin data alone do not suffice. Branch adjacency is not substituted for
the global elder event.

Here is the general measure argument used after E2's Borel disintegration.
Suppose on a mark space `Theta` the candidate intensity is
`r A(r,theta) dr dtheta`, `0<=p<=1` is the conditional selected fraction,
and `k(theta)>0` with exact lifetime `ell=k(theta)r³`. Suppose also that
`0<=A(1-p)<=B`, where `B` is a measurable nonnegative loss amplitude. Define
the selected, rejected and dominating loss measures by pushing forward
`r A p dr dtheta`, `r A(1-p) dr dtheta` and `r B dr dtheta`. Then for
**every Borel lifetime set** `E`,

```text
mu_cand(E) = mu_eld(E) + mu_rej(E),
0 <= mu_rej(E) <= mu_B(E).                            (B1)
```

This additive identity remains meaningful for extended nonnegative measures;
it does not subtract two infinite masses. To prove it, multiply the pointwise inequality by
`r 1{k(theta)r³ in E}`, integrate and use Tonelli. Where these measures have
densities, the substitution for each fixed mark yields, almost everywhere,

```text
nu_eld(ell) = ell^(-1/3) integral A(r,theta)p(r,theta)
                                      /[3 k(theta)^(2/3)] dtheta,
0 <= nu_cand(ell)-nu_eld(ell)
   <= ell^(-1/3) integral B(r,theta)/[3 k(theta)^(2/3)] dtheta,
r=(ell/k(theta))^(1/3),                              (B2)
```

with the actual radius-domain indicator retained. Integrability must be
proved for the application; in the present one it is R (R11)–(R18).
An inequality of measures on all Borel sets gives the density inequality
almost everywhere by Radon–Nikodym uniqueness. Source statements using
canonical density versions retain their own version convention.

This is the required bridge in a form that exposes both topology and weighting.
The numerator is never replaced by an unweighted collision probability.

## 4. Error conversion, with the Jacobian retained

The following conversions are uniform only on a fixed compact mark set with
`0<k_-<=k<=k_+`, bounded other parameters, and a bounded contact pin density.
Their use over all marks needs the separate majorant in §5.

For a weighted conditional numerator `N_r=O(r^beta)`, the spatial, gap and pin
Jacobian factors multiply to `r^(-1)`. The radial error is therefore
`O(r^(beta-1)) dr`. Exact pushforward gives

```text
r^(beta-1) dr/dell = (1/3) k^(-beta/3) ell^((beta-3)/3).
```

For a radial density error `O(r^alpha) dr`, the corresponding lifetime density
is `O(ell^((alpha-2)/3))`. These two formulas describe different input objects;
the powers must not be interchanged.

| Input object | Weighted numerator | Radial density | Lifetime density | Cumulative order near zero |
|---|---:|---:|---:|---:|
| Principal typed weight | `r²` | `r` | `ell^(-1/3)` | `t^(2/3)` |
| Compact cap failure | `O(r⁵)` | `O(r⁴)` | `O(ell^(2/3))` | `O(t^(5/3))` |
| Generic weighted error, beta>0 | `O(r^beta)` | `O(r^(beta-1))` | `O(ell^((beta-3)/3))` | `O(t^(beta/3))` |
| Amplitude error `A_r-A_0=O(r)` on compact marks | `O(r³)` after bounded pin factors | `O(r²)` | `O(1)` | `O(t)` |

For compact cap failure, `Z_r>=z_* r²` and
`E_Q[W_r;G_r^c]<=C r⁵` give

```text
Q_r^W(G_r^c) <= (C/z_*) r³.
```

Multiplying the principal lifetime density by this uniform failure order
gives the same `O(ell^(2/3))` compact selection loss as the table. Dividing by
`Z_r` twice, dropping a determinant factor, or using this compact conclusion
as an unrestricted estimate would be incorrect.

## 5. The unrestricted error ledger already available

For the fixed small spatial cutoff `r0`, put `a=ell/r0³` and
`eta=ell^(1/6)`. R supplies an integrable polynomial-Gaussian majorant
`H(b,k)=C(1+|b|+k)^N exp[-c(b²+k²)]`. Its all-mark estimates are

```text
0<=A_r<=(k+r)² H,      0<=A_0<=k² H,
|A_r-A_0|<=r(k+r)H,
A_r(1-p_r)<=B_r<=A_r,
B_r<=(r/k)³H when r/k<=1.
```

These are unnormalized amplitude bounds. They avoid requiring a global
positive floor for `Z_r/r²` as `k` tends to zero.

| Source/error | Bound for ell^(1/3) times density error | Bound after dividing by ell^(1/3) |
|---|---:|---:|
| R §6: omitted contact marks `0<k<a` | `O(a^(7/3))` | `O(ell²)` at fixed r0 |
| R §6: contact-amplitude difference, `k>=a` | `O(ell^(1/3)+ell^(2/3)a^(-1/3))` | `O(1)` |
| R §7: loss on `a<=k<=eta` | `O(eta^(7/3)+ell^(2/3)a^(-1/3))` | `O(ell^(1/18)+1)` |
| R §7: loss on `k>=eta` | `O(ell eta^(-11/3))` | `O(ell^(1/18))` |
| P §14 / R §7: separation `>=r0` | `O(ell^(1/3))` | `O(1)` |

The lower cutoff `a` is not disposable: integrating `k^(-4/3)` to zero would
diverge. The large-mark Gaussian tail, scalar far-eigenvalue branch and
fourth-derivative exception are retained in R §§4–7. This ledger recovers the
bounded remainder, not a second coefficient or a vanishing remainder.

The limit order is: fix this model and r0 within the embedded chart, apply
the all-mark bounds, then let ell decrease to zero. The near/far partition is
exhaustive; no separate growing-annulus limit or shrinking remote cutoff is
used. Periodic images enter the **exact** field covariance throughout and its
coefficient transfer in S24; they are not discarded spatial configurations.
The SIDE24 exponentially small covariance perturbation is not an error that
vanishes with ell and is not a simultaneous L-to-infinity theorem.

## 6. Corrections required before using the outside directive

**A cumulative sandwich cannot be differentiated without more information.**
For `0<t<=1/10`, define

```text
F(t)=(9/2)t^(2/3)+t^(5/3)sin(1/t),     F(0)=0.
F'(t)=t^(-1/3)[3-cos(1/t)+(5/3)t sin(1/t)] > 0.
```

It is a positive measure's cumulative function, with
`F(t)=(9/2)t^(2/3)+O(t^(5/3))`, but `t^(1/3)F'(t)` has no limit.
Even this small cumulative error does not establish the claimed leading
density. Use (B1) for all Borel sets together with density bounds, or prove
a separate differentiation/Tauberian hypothesis. Density estimates do imply
the cumulative law by integration; the converse is the dangerous step.

**A value-only Taylor remainder does not control the inverse Jacobian.**
For `0<a<k`,

```text
ell(r)=k r³+a r⁴sin(1/r),
ell'(r)=r²[3k-a cos(1/r)+4ar sin(1/r)].
```

For small r this is increasing and has value remainder `O(r⁴)`, but the
relative derivative error does not tend to zero. A derived geometric
normal-form substitution needs, for example, uniform `|R|<=Cr⁴` and
`|partial_r R|<=Cr³`, with a positive lower bound on k. P instead defines
`k=(f(M)-f(S))/r³` as an integration coordinate; its lifetime Jacobian is
exact. The field's Taylor expansion is used elsewhere and is not a
justification for differentiating an unspecified big-O term.

**Global factorial bounds do not identify singleton clusters.** Let
`N_r=2` with probability `r³/2` and zero otherwise. Then
`E N_r=r³`, `E(N_r)_2=r³`, and every higher factorial moment is zero.
Nevertheless `P(N_r>=2)=r³/2`, and the positive-count law is always
`delta_2`, not `delta_1`. This refutes a singleton inference, not the valid
r-dependent compound-Poisson approximation already recorded in C6.
Indeed put `q=P(N>0)` and `J~law(N|N>0)`. The count is exactly a compound
Bernoulli count. Replacing its Bernoulli(q) cluster count by Poisson(q)
changes total variation by at most `q(1-exp(-q))<=q²`. If `E N=O(r³)`,
then `q<=E N` and the error is `O(r⁶)`. This does not identify or prove
convergence of J, or create independent spatial clusters. Alternating the
single-jump law `N=1` with probability r³ and the double-jump law above
along successively smaller radius bands preserves those upper moment orders
while preventing a unique positive-count limit.

**Positive Fourier coefficients require independent functionals.** Two
copies of the same derivative functional give a singular covariance, and
points separated by a period are the same torus point. For distinct points
modulo the torus and linearly independent derivative distributions, zero
variance implies every Fourier coefficient of their linear combination is
zero. Fourier uniqueness makes that distribution zero; independence then
forces its coefficients to vanish. Coincident-point contact limits must use
the rescaled, independent jet list, as P does. Qualitative rank away from
collisions is not a quantitative shrinking-scale eigenvalue estimate.

**Small lifetime does not unconditionally mean small spatial separation.**
The equivalence is uniform only inside a specified canonical positive-gap
mark regime. The all-pair formula retains distant small-height-gap pairs.
For T24-planar their density is bounded, rather than asserted absent.

## 7. A precise conditional interface for the collision frontier

This elementary reduction specifies a sufficient input; it does not claim
that the Gaussian field satisfies it. For a measurable nonnegative integer
witness count `N_r` and the **same** nonnegative typed weight `W_r`,

```text
1{N_r>=2} <= N_r(N_r-1)/2,
Q_r^W(N_r>=2) <= E_Q[W_r N_r(N_r-1)]/(2 Z_r).         (C1)
```

The first inequality holds for every integer value (zero for 0 and 1,
equality at 2); multiplication by W and integration proves the second.
If a specified bad selector event E satisfies the separately proved
deterministic implication `E subset {N_r>=2}` on typed support, and

```text
E_Q[W_r N_r(N_r-1)] <= C r^(2+beta),    beta>0,       (C2)
```

uniformly on the compact marks of §4, then
`Q_r^W(E)<=C r^beta/(2z_*)`. Its compact lifetime-density contribution is
`O(ell^((beta-1)/3))`, strictly lower order than `ell^(-1/3)`, by §4.
The cumulative contribution is `O(t^((beta+2)/3))`.
This closes the elementary reduction from an appropriately weighted
factorial estimate to a loss estimate. The geometric implication, the exact
shrinking witness count, and (C2) for that count remain the substantive
field-specific obligations. An estimate for another count cannot be
substituted without a deterministic comparison under the same law and weight.

## 8. What to work on next

The directive improves the task rule: choose an exact consumer theorem,
identify a missing conclusion with its conditioning and weight, then close
that conclusion or prove why it is not consumed. Do not add a numerical
experiment merely to support a step requiring an analytic proof.

For the **regional shrinking-witness theorem**, first specify the exact
witness count/region and event, the original tilted law, all parameter
ranges, and the intended downstream inequality. Then attack its collision
strata or exhibit a valid replacement argument. Its open graph node must
remain open until that precise proof exists. T24-planar's separate route
does not answer this research question.

For **T24-planar**, the immediate assembly need is a continuous complete proof
in reading order, particularly the full cap construction and repaired Borel
argument; the present manuscript and appendices still import their full
sources. Mathematical validity, manuscript self-containment, numerical
applicability, and external human assessment are separate completion axes.
Publication or a human review is not a logical premise of a theorem, while
the project's retained acceptance predicates are not silently removed by
this audit.

Later peer work must be consumed at exact reviewed versions. The live check
found no open main PR at entry; Math- contained active higher-order and
constant-evaluation proposals. Their advertised stronger results are not
needed for this fixed target and are not adopted here from PR titles or
green checks. Preserve authors' branch ownership and source-specific reviews.

There is deliberately no `OPEN NODES: 0` completion certificate here. A future
certificate must identify its theorem and actual consumed closure records,
and must not encode a desired outcome as an observed one. A machine can check
identities, edge references and status consistency; that alone does not
prove the analytic statements on those edges.

**Attribution:** Dylan Roy — delegated AI review; actual analysis and assembly
by OpenAI/Codex. Source exposure and same-provider collaboration give zero
organizational-independence credit. Dylan's personal reading is PENDING.
