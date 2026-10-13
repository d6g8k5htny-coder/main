# Full periodic field and global H0 selector: frozen execution contract

Predeclared by OpenAI/Codex root `01a11cbf-068d-7102-861e-75814e715c98`
on 9 October 2026, before implementation or fixture execution. This successor
uses the complete 451-line planning draft by `/root/population_contract_audit`,
26545 bytes, SHA-256 `dbc0c02aa0cbb7534b4965835e35e49a93fac25ef0d6f012a86476c5ad5062ec`.
Root read the complete preceding 448-line cut and the final counter/name delta;
the author freshly read all 451 lines. Distinct bounded draft math/ABI review
found no must-fix in the 448-line cut; the final delta changes only test names
and the primitive-counter scope. This accepts an execution design, not a
successful field evaluation, selector run, proof admission or scientific result.

Exact additive ownership is native issue307/comment6087579322 and R17 Work Events
row1359/event`f87f892a-7031-4624-9cba-592b081181ea`; working end20:30 UTC,
lease21:15 UTC. Standing owner authorization permits the work and supersedes
older permission checkpoints. Tests and mathematical acceptance remain required.
Source exposure is explicit; organizational and blind independence credit remain
zero. Earlier sources, results, failures and receipts remain unchanged.

**Goal:** instantiate the explicit smooth periodic field and derive its actual
superlevel H0 barcode from certified level-set connectivity on 18 fixed
controls, then apply the separate population convention.

**Architecture:** a small field module supplies outward value enclosures and
the bound model identity. A separate selector uses exact polynomial root and
component certificates; it receives only `(u,v)`, never a chosen side or gap.
The driver compares literal expectations only after selection and verification.

**Dependencies:** Python standard library, exact `fractions.Fraction`, the
existing `finite_certificate.sqrt_bounds` and `pi_bounds`, and
`gaussian_tail.exp_neg_bounds`. No RNG, grid persistence, external algebra
package, installation, population integration, or Lean kernel is required.

## 1. Consumed identities and acceptance boundary

The draft author read actual `PROOF.md` lines 1–220, 221–440, 441–620 and
621–782, all 270 controlled-contract lines, the earlier 288-line and final
287-line analytic-guard sources, and the actual 286-line population test file.
The new diagnosis concerns
the field and geometry in Sections 2–3, rather than a new verdict on every
population asymptotic or process claim.

Freeze these consumed inputs before importing the new executable modules:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `PROOF.md` | 38350 | `9e8caa52b6448e23e465e646ca98d6e42e0cd71442ad4118b59145ee849e1326` |
| `CONTROLLED_FALSIFICATION.md` | 15296 | `a23f88801e678dd1af484cac8f5564801d9d410d835b560a280c0ea6dfc8996e` |
| `ANALYTIC_GUARDS.md` | 10494 | `2f87cafba115a314e44d04d784701aebcebdc123feced7f22540063fa7570db9` |
| `finite_certificate.py` | 14843 | `0674a2621dff6c040cbb24cc0b6e6df7847541ebc171ebdb28070afa05e04e6b` |
| `gaussian_tail.py` | 4688 | `07e55a62ed0bb6da6a9395762e2e0c033126f83b46abd48db4c24df710779db8` |

The first three inputs are in the existing cusp-population directory; the
last two are in `experiments/periodic_h0/`. Record this draft's successor
contract and every new module/test/driver identity when those files exist.
Freeze any additional imported local source rather than silently extending
the dependency list. No implementation hash exists at this drafting stage.

Root has reported bounded acceptance of the final `2f87` guard source, after
a fresh final-source review recording `PASS_BOUNDED_ANALYTIC_GUARDS`.
The final outside review identity is 22303 bytes, SHA-256
`ef831e2778e805f277f0e1ec1185554e812f75adde815d05c60016d6e4761b08`.
That accepted analytic source closes no evaluator/selector execution scope.
Before execution, root must bind the accepted exact guard source and its
fresh nonauthor review by immutable source/review identity and accepted-premise
metadata. The outside review's host-specific path is not an import dependency;
the driver carries the bound acceptance metadata into the receipt.
An unavailable accepted analytic premise leaves
execution `NOT_RUN`. Source drift or a changed contract withholds all
field/selector acceptance as `INCONCLUSIVE_SOURCE_DRIFT`.

Claimed additive source names: `surgery_field.py`,
`global_h0_selector.py`, `test_surgery_field.py`, `test_global_h0_selector.py`, and
`test_global_h0_runner.py`, with `run_global_h0_selector.py` beside this
`GLOBAL_SELECTOR_CONTRACT.md`.
Keep the completed population files and `exact_population_1` results intact.

## 2. Exact object and scalar interval rules

Keep `d=2`, `L=1`, `R=1/8`, `r1=1/32`, `A=2^-16`,
`u0=3/2^32`, `v0=1/2^47`, `a_star=1/128`, `delta=2^-26`, `b=4/9`.
Controls accepted by the public selector satisfy the strict open rectangle
`-u0<u<u0`, `-v0<v<v0`. All mathematical inputs and endpoints have exact
`Fraction` type; reject floats, bools, reversed intervals and invalid domains.
Integer counts and monotonic clock values have their explicitly separate types.

For a torus input, reduce each rational coordinate to `[-1/2,1/2)` exactly.
For that representative define

```
s=sqrt(2/3)*sin(pi*y1), z=sqrt(2/9)*sin(pi*y2), r=s^2+z^2,
q(t)=exp(-1/t) for t>0, q(t)=0 otherwise,
psi(t)=q(1-t)/(q(t)+q(1-t)),
chi(s,z)=psi((r-r1^2/4)/(3*r1^2/4)),
c(r)=1-(1-r/4)*psi((r-r1^2)/(R^2/2-r1^2)),
F_(u,v)(y)=b-s^2*c(r)-z^2+chi(s,z)*(u*s^2/2+v*s).
```

The full formula equals the periodic seed outside the modified chart and
on neighborhoods of all seams. Chart gradients use `(s,z)` norms. The
chart Jacobian is nonsingular on the support; no numeric physical-gradient
lower bound is inferred. Translation preserves the barcode, so use `Y=0`
without a translation draw. The field evaluator must execute the full
formula, including both transitions, rather than only its core polynomial.

Use 128-bit outward dyadic rounding in the new transcendental operations.
The consumed exponential engine's fixed internal 256-bit precision is reused
unchanged and rounded outward at the new module boundary; it is not retuned.
Signed multiplication takes all four endpoint products; squares handle an
interval crossing zero; division requires a certified positive denominator.
All polynomial divisions, gcds, signs and root certificates use exact
unrounded rational arithmetic. Keep centered heights `h=F-b` in the selector.

For rational `X` intervals with `|X|<=2`, enclose sine using the fixed 64-term
polynomial `sum_(k=0)^63 (-1)^k*x^(2*k+1)/(2*k+1)!`. Evaluate it by
`T0=X`, `T_(k+1)=-T_k*X^2/((2*k+2)*(2*k+3))` and sum the 64 outward
terms, then add symmetric error `2^129/129!`. The recurrence multipliers
are at most `2/3`; do not use rounded Horner evaluation that amplifies early
rounding errors by repeated multiplication with `X^2`.
This is Taylor's formula through degree 128, whose even coefficient is zero.
Check the consumed `pi_hi<4` and the
representative bound to certify `|pi*y_i|<2`. Do not replace this remainder
by an observed approximation error or increase the term count in this run.

For `q(t)`, return `[0,0]` if `t<=0`. For `t>0` with `1/t<=4096`, use the
consumed exponential enclosure. For `1/t>4096`, use `[0,2^-4096]`, justified
by `exp(1)>2`. The exponential primitive must never receive an out-of-domain
argument. Use monotonic endpoint bounds for interval `q` and descending
`psi`; exact outside-step plateaus are `[1,1]` and `[0,0]`. The positive
denominator follows from one step argument being at least `1/2`; verify its
interval lower bound is positive. A zero enclosure lower bound cannot be
replaced by an invented denominator. Intersect bounds with a proved range
such as `[0,1]` only when that analytic range applies.

## 3. Small ABI and required returned values

Use `Q=fractions.Fraction`, `Interval=tuple[Q,Q]` and ascending-power
`Polynomial=tuple[Fraction,...]`. A root object contains its square-free
factor polynomial, sorted ordinal, rational isolator or exact rational value,
and multiplicity in the original polynomial. JSON stores Fractions as
canonical rational strings. Do not add a generic expression framework.

`Budget` is one attempt context owned by the driver. It tracks the shared
monotonic deadline, total field/primitive calls, per-fixture polynomial
evaluations, and actual branch/refinement depths. Tests may reduce ceilings;
no public control may increase a frozen ceiling. Its clock is not a
mathematical Fraction. Every executed operation charges its specified counter.
Its constructor is `Budget(start: float, *, seconds: int=900,
polynomial_limit: int=100000, field_limit: int=4096,
primitive_limit: int=32768, root_splits: int=256)`.
`begin_fixture(fixture_id: str)` opens each ID once and resets only its
polynomial counter; verification charges the same fixture. `snapshot()`
returns actual counters and retained field records without resetting them.
Low-level exhaustion raises `Inconclusive(status: str, phase: str, detail: str)`;
the selector/driver retains partial records and the allowed inconclusive status.

Required public interfaces:

```
torus_field_bounds(u: Q, v: Q, y: tuple[Q,Q], *, budget: Budget) -> Interval
chart_field_bounds(u: Q, v: Q, s: Interval, z: Interval, *, budget: Budget) -> Interval
guard_certificate() -> dict
isolate_roots(p: Polynomial, domain: Interval, *, budget: Budget) -> dict
select_h0(u: Q, v: Q, *, budget: Budget) -> dict
verify_h0(u: Q, v: Q, certificate: dict, *, budget: Budget) -> dict
```

The signatures and ordinary tuple/dict records target Python 3.12; record the
actual interpreter, and treat an unavailable compatible runtime as setup
blocked rather than inferring its version from the name `python3`.
The chart evaluator accepts a box only if its whole enclosure lies in
`s^2+z^2<=R^2`; point anchors use singleton intervals. Its record identifies
the explicit model, branch and source identities. `guard_certificate` binds
the reviewed analytic premises and replays their rational consequents; it
does not claim to prove C-infinity smoothness by finite sampling.

`isolate_roots` returns square-free/gcd factors, exact Sturm chains, root
counts, endpoint variation certificates, isolators, multiplicities and work
counts. `select_h0` receives no roots, theta, sign label, expected side,
barcode, population selector, or gap formula. It returns all critical data,
levels, sign intervals, incidence witnesses, event batches and elder graph,
the actual barcode, the population projection and disposition.
`verify_h0` checks the supplied algebra and incidence rather than calling
`select_h0` or trusting its PASS flags. It returns named check results and
rejects a changed root multiplicity, omitted component or swapped elder edge.

Neither new executable module imports `population_certificate`, its runner,
`critical_fixture`, a population CDF, or a lifetime/gap oracle. The driver may
read literal comparison expectations only after a complete selection and
verification certificate exists. Known expected lifetimes are absent from
the selector's inputs and selection phase.

Sturm variation discards zero entries at nonroot interval endpoints. Handle
an endpoint or midpoint that is exactly a root by retaining that exact point
and dividing its factor from the square-free polynomial before counting the
remaining roots. Do not count that point in both children. Retain the exact
factor identities and original multiplicity, with a new chain for the
quotient; all remaining endpoint variation requests must have nonroot endpoints.

Freeze these record keys for test/implementation agreement; IDs are strings,
indices/counts are strict integers and numeric endpoints are Fractions:

| Record | Required keys |
| --- | --- |
| Root set | `status`, `polynomial`, `domain`, `factorization`, `sturm_chains`, `roots`, `checks`, `work` |
| Root | `factor`, `ordinal`, `isolator`, `exact_value`, `multiplicity`, `branch_splits` |
| Selection | `schema_version`, `model_id`, `u`, `v`, `status`, `guard`, `critical_roots`, `old_criticals`, `critical_heights`, `height_groups`, `levels`, `incidence`, `events`, `actual_barcode`, `population_projection`, `checks`, `work` |
| Critical height | `root_ordinal`, `centered_interval`, `exact_value`, `multiplicity`, `axial_inertia`, `transverse_inertia`, `group_id` |
| Height group | `id`, `root_ordinals`, `centered_interval`, `equality_witness` |
| Level | `id`, `centered_height`, `root_certificate`, `components` |
| Component | `id`, `boundary_root_ordinals`, `interior_witness`, `witness_polynomial_value` |
| Incidence edge | `upper_level`, `lower_level`, `upper_component`, `lower_component`, `witness`, `lower_polynomial_value` |
| Event batch | `group_id`, `births`, `continuations`, `merges` |
| Actual barcode | `status`, `essential`, `finite`, `tie_policy` |
| Finite bar | `birth_root`, `death_group`, `birth_centered_interval`, `death_centered_interval`, `lifetime_interval`, `lifetime_exact`, `tied_birth` |
| Population projection | `status`, `count`, `reason`, `actual_finite_count` |
| Verification | `status`, `checks`, `work` |

`schema_version=1`, `model_id="explicit_periodic_cusp_v1"` and
`tie_policy="smallest_axial_root_ordinal_survives"` are fixed. Root, selection
and actual-barcode success is `CERTIFIED`; verifier success is `PASS`. An unavailable
exact value/equality witness is null, never a guessed zero. The guard record
contains the consumed analytic-source/review identities, model parameters,
named derivative/perturbation/barrier bounds and their rational checks.
Budget field records retain route, `(u,v)`, input point/box, used branches,
returned interval and bound model identity. The runner serializes these
records alongside the selection certificates. A Sturm-chain record contains
`factor`, `sequence`, `left`, `right`, `left_variations`, `right_variations`
and `root_count`; factorization records contain `factor` and `multiplicity`.
Each work record contains `polynomial_evaluations`, `gcd_steps`,
`polynomial_divisions`, `max_branch_splits`, `field_calls` and
`primitive_requests`, with per-fixture versus whole-attempt scope labeled.

## 4. Critical completeness and connectivity algorithm

First bind the full field to the accepted guard source. Replay its constants:
`|psi'|<=4`, `|grad chi|<=32/(3*r1)<64/r1`, the four perturbation bounds,
`|grad F0|>=r1^3/16>r1^3/64` on the cutoff annulus, strict root containment,
gradient exclusion and height barrier. These are analytic dependencies with
rational checks, not conclusions of the root list or evaluator anchors.

The resulting full critical list consists of the roots of
`p(s)=s^3-u*s-v` on `[-a_star,a_star]`, with transverse coordinate zero,
and the unchanged old saddles at `2/9,-2/9` and minimum at `-4/9`.
Verify their old-point locations/inertia and that none is a superlevel birth.
In the core the Hessian is `diag(u-3*s^2,-2*(1+s^2/4))`.
Use exact `gcd(p,p')` to retain repeated-root multiplicities. A zero axial
Hessian is recorded as degenerate, not relabeled a Morse saddle or maximum.

Evaluate each centered critical height from the actual field polynomial
`P(s)=-s^4/4+u*s^2/2+v*s`. Stationary polynomial reduction
`P mod p=(u*s^2+3*v*s)/4` is allowed as an independently checked evaluation
identity. It does not select a pair. Order height groups only when strict
interval separation or an exact equality certificate is available.
For rational roots, evaluate exactly. For the irrational tied outer roots,
`s^2-u=0` and `P mod(s^2-u)=u^2/4` certify equality. Overlapping intervals
alone never certify equality; other unresolved comparisons are inconclusive.

Use top level `h_top=delta/2` and bottom level `h_bottom=-delta/2`, whose
full-field separator is `b-delta/2=4/9-2^-27`. Both are regular and above
the exterior barrier `b-3*delta/4`. Between adjacent descending critical
height groups with enclosures `[L_i,U_i]`, `[L_next,U_next]`, require
`L_i>U_next` and use exact rational `h=(L_i+U_next)/2`.

At every regular level independently isolate all roots of `P(s)-h` on
`[-r1/2,r1/2]`. Evaluate the original, unnormalized polynomial's signs in
the complementary intervals and retain its positive closed intervals.
Normalization of a root polynomial must not silently reverse these signs.
The boundary barrier excludes an omitted outside-core part. Each positive
axial interval lifts to exactly one full-field component: transverse fibers
`z^2 <= (P(s)-h)/(1+s^2/4)` contract to their centers within the superlevel.

For consecutive descending regular levels, give each upper component a
rational interior witness with certified positive polynomial sign. Locate
that witness in a unique lower component. Connectedness and nested
superlevels then certify containment of the entire upper component.
A lower component with zero predecessors is a birth, one is continuation,
and multiple predecessors produce a merge. Link each event to the intervening
critical-height group and its critical roots; retain all witnesses.

Process equal-height groups simultaneously. The elder rule keeps the class
with the greatest actual birth height. If births tie exactly, retain the
smallest sorted axial root ordinal as the representative and record that
label convention. Derive birth/death expressions from this graph, retain the
essential class separately, and report positive finite lifetimes. A multiple
cubic root is handled by observed incidence; no fictitious Morse event or
zero-length finite bar is inserted. At the bottom level require one component.
The complete critical list/no-other-maxima certificate covers all later
global levels. No sampled torus grid is used for this argument.

## 5. Fixed fixtures and post-selection expectations

There are exactly 18 ordered controls. For the first 14 let `a=2^-18`,
use theta in `(1/4,1/2,3/4,0,1,3,2)` in that order and signs `(1,-1)`
for each theta, and set `u=(3+theta^2)*a^2`,
`v=sign*2*(1-theta^2)*a^3`. This reproduces the existing literal fixtures
without calling their selector. Rows 15–18 are

```
15: (u,v)=(-u0/2, v0/2)
16: (u,v)=(0, v0/2)
17: (u,v)=(u0/2, v0/2)
18: (u,v)=(u0/2, 0)
```

The first three extra controls are nonwedge; the last is the additional
irrational-position tie. These duplicate neither a random draw nor an added
population schedule. Every control lies strictly inside the original rectangle.

Post-selection finite lifetimes for theta `1/4,1/2,3/4,2` are respectively
`2^-76`, `2^-73`, `27/2^76`, `3/2^74`, for both signs.
Theta `0,3` yields a degenerate field with no positive finite bar.
Theta `1` has an actual tied finite bar `2^-70`; row 18 has one of
`9/2^68`. Both retain an essential class. Rows 15–17 have no finite bar.
On actual selected non-tie bars, reflection must exchange the younger side.
The hostile theta `2`, sign `1` bar comes from the right; the wrong
fixed-left lifetime `2^-67` must be rejected after graph construction.

Compute the population projection only after retaining the actual barcode:
the discriminant `4*u^3-27*v^2=0` and the line `u>0,v=0` have zero projected
count under the frozen law. Ties therefore retain an actual finite bar and
a separate `EXCLUDED_TIE`/zero population count. Never describe that zero
projection as absence of a topological merger.

Retain the previously derived grid-obstruction certificate for the existing
theta `1/2`, sign `1` fixture, without running a grid: at
`q_star=b+9*a^4/128` the continuous field has two components, while an
origin-aligned 1024-by-1024 vertex superlevel is empty. If `s=0`, the full
formula gives `F-b=-z^2<=0`. Otherwise its representative has
`|y1|>=2^-10`; `sin(pi*|y1|)>=2*|y1|` on `[0,1/2]` and
`sqrt(2/3)>1/2` give `|s|>=2^-10`. The physical model has `r<=8/9<1`
and the explicit surgery satisfies `c(r)>=r/4`. Therefore

```
F-b <= -s^4/4+u0*s^2/2+v0*|s|
    <= s^4*(-1/4+3/2^13+1/2^17) < 0.
```

At the existing fixture, the younger centered height is `9*a^4/64` and
the saddle is `-23*a^4/64`, putting `q_star-b` strictly between them.
This proves the target component discrepancy. It does not assert that every
other discrete bar is absent and earns no grid-campaign credit.

## 6. Full-field anchors and acceptance gates

For every fixture evaluate the chart point set

```
(0,0), (+/-r1/4,0), (+/-r1/2,0), (+/-3*r1/4,0),
(+/-r1,0), (+/-2*r1,0), (+/-3*r1,0),
(0,r1/2), (0,3*r1/4), (r1/2,r1/2)
```

Here `+/-` expands to two separate points. These 16 anchors cover the exact
core, both chi boundaries, chi transition, both surgery branches, its
transition and seed collar. Add the certified new-root chart boxes, with
singleton zero transverse coordinate, for actual critical evaluations.

Also call the torus evaluator at the eight physical points
`(0,0),(0,1/2),(1/2,0),(1/2,1/2),(1/4,0),(0,1/4),
(-1/2,1/8),(1/8,-1/2)` for every fixture. Re-evaluate each with integer
translation `(1,-1)` and require identical canonical input representatives
and returned intervals. This adds exact periodic-seam/exterior checks, not
translation sampling. At the three old criticals the intervals contain their
known exact seed values; collar anchors contain `b-s^2-z^2`.

Require every retained field interval to be ordered, correctly sourced and
have full width at most `2^-104`. Core algebraic height enclosures and
evaluated field enclosures must overlap under the separately certified exact
core identity. Strict critical-height order, positive interval signs and
component incidence come from exact certificates, not interval overlap.
Failure to meet a width/sign/separation gate is inconclusive; an exact
disagreement with a post-selection expectation is a retained falsifier of
this implementation/model prediction only after all soundness gates pass.

## 7. Frozen computational ceilings and failure dispositions

| Quantity | Ceiling or fixed target |
| --- | --- |
| Transcendental outward precision | 128 bits; no precision increase |
| Sine polynomial | 64 terms and fixed stated remainder |
| Root isolator target full width | `2^-160`, or an exact rational root |
| Isolation/refinement | at most 256 midpoint splits along any root branch |
| Polynomial evaluations | 100000 per fixture, counting verifier replay |
| Full-field evaluator calls | 4096 for the whole attempt |
| Elementary interval primitive requests | 32768 for the whole attempt |
| Whole 18-fixture command | 900 seconds, including writes and flush |

One polynomial evaluated at one rational endpoint or interval is one
evaluation. A Sturm variation calculation evaluates several polynomials and
charges each one. Count actual repeated computations; a cached certificate
reuse is separately recorded. Gcd and exact polynomial-division counts are
recorded, as are branch depths and every root refinement. Primitive requests
include square root, pi, sine and exponential calls even if a dependency's
internal cache serves them; those four request classes are the entire primitive
counter scope. Rational interval additions/multiplications do not charge that
counter. This is a conservative workload counter, separate
from polynomial and field calls. Budgets include certificate verification,
post-source hashing and serialization; no isolated helper gets a new clock.

These ceilings are frozen before implementation and execution; their empirical adequacy is NOT_RUN at this cut. Fixed
fixture precision is not a claim of termination for every control arbitrarily
near a fold or tie. Never guess a sign, conflate overlap with equality, or
increase the budget: retain `INCONCLUSIVE_PRECISION` for unresolved algebra,
`INCONCLUSIVE_BUDGET` for exhausted counts/time, and
`INCONCLUSIVE_SOURCE_DRIFT` for changed inputs. Reject invalid ABI inputs.
`FAIL_IMPLEMENTATION` covers a contradictory Sturm chain, invalid enclosure,
missing critical/component certificate or a malformed accepted guard certificate.

## 8. Test-first handoff and complete-command receipt

The test author uses the ABI and literal controls, without importing or
recreating the old pairing implementation. Required meaningful cases cover
outward scalar/field bounds and seams, gcd multiplicities, irrational exact
tie batching, quartic component incidence, hostile younger-right pairing,
actual-versus-projected tied bars, and certificate tampering. Reduced budget
and source-drift tests must withhold nested barcode acceptance while retaining
computed records. At a driver-wide drift or final budget failure, retain the
previous computed selection, verification, barcode and projection statuses
in named `computed_status_before_withdrawal` fields; assign the corresponding
inconclusive disposition to all their acceptance statuses. Numeric bars and
exclusion reasons remain recorded, but no nested `CERTIFIED`/`PASS` survives
as accepted. A producer cannot pass by supplying precomputed fixtures
or expected bars in place of the root/component algorithm.

The later implementation author supplies the minimal two modules and driver.
Fresh normal and optimized focused tests plus full changed-source review are
required before the single frozen fixture attempt. No such tests or fixture execution had run at this contract cut. Avoid a generic topology framework, additional exponent lemmas,
archive project, or repetition of the completed eight-row population run.

The driver freezes source bytes before scientific imports and checks them
again after computation. Its outputs are three exclusively created files in
the fresh `experiments/universality/results/global_selector_1/`:
`RUN.json`, `CERTIFICATE.json`, `RESULTS.md`. Preserve any existing attempt;
never overwrite or retry its directory. Canonical JSON rejects duplicate
keys, floats and nonfinite values; RUN binds the other two byte identities
without a self-hash cycle. Keep actual field-call/primitive/root/polynomial
counts, phase times and all partial certificates.
Serialize monotonic durations and UTC timestamps as canonical strings;
the live clock's float type does not enter the exact JSON certificate.

From the repository root the declared command is
`python3 -B -S experiments/universality/two_parameter_cusp_population/run_global_h0_selector.py`,
with `python3` resolved to the recorded compatible runtime.
The complete declared command must also run under an outside subprocess
recorder with a 900-second hard timeout and exact stdout/stderr/exit/walltime
evidence. A driver's final pre-write check cannot certify its own later flush.
No PASS delivery occurs unless the actual whole command finishes under 900
seconds with exit zero. Timeout or later failure preserves written artifacts
and receives an external inconclusive disposition; it does not overwrite an
earlier artifact status to manufacture a clean attempt.

Success means full-formula interval evaluations at all declared anchors and
an executed, separately implemented global connectivity selector for these
18 fixtures, conditional on the explicitly bound analytic guard proofs.
Report those two scopes separately. All Gaussian universality, random/blind
campaign, same-field spatial Poisson, higher-homology, Lean/kernel and
scientific-promotion scopes remain NOT_RUN. Source-exposed same-team review
and algorithmic separation supply no organizational-independence credit.
