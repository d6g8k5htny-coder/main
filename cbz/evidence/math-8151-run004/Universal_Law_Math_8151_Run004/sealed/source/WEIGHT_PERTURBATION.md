# Same-law weight perturbation and stable normalization

Scientific effect: NONE. Dylan Roy — delegated AI work. New proof author and
implementer: OpenAI / GPT-6 Astra Pro, session `lean-weight-perturbation-20261005`.
No parent theorem or independent alignment status is promoted.

## Purpose and source identity

A bound on an unnormalized weight is not automatically a bound on its normalized
probability law. This module supplies the missing abstract error-propagation
interface: use a positive reference normalizer and a genuine L1 error, then derive
a lower normalizer for the perturbed weight and an error bound for every measurable
event. Concrete field estimates and identification of the two weights remain the
application's responsibility.

Consumed construction: `formal/ResearchFormalCoreR1/WeightedLaw.lean` in
`d6g8k5htny-coder/Math-` at `5b012335b279d1f93b930e7b906da09164332c47`,
SHA256 `f31dfd2cd048cc0c0446c54c46d89c9cdd5d7bb9d76e9d8d16b7c034f351d736`.
All six earlier Lean modules and their forty theorem proofs are unchanged.
The new module imports WeightedLaw directly, not the later generality proposals.

Pinned mathlib is `d13f23b723b8a846827a245b89c10fc7d3f11612`;
`MeasureTheory.abs_integral_le_integral_abs` and `integral_sub` are in
`Mathlib/MeasureTheory/Integral/Bochner/Basic.lean`, and
`MeasureTheory.setIntegral_le_integral` is in the companion `Bochner/Set.lean`.
Lean stays at 4.34.1. No new dependency, custom axiom, or gate exemption is added.

## Exact seven-target scope

Every declaration is in `ResearchFormalCoreR1`.

| Target | Established statement |
|---|---|
| `weightPerturbation_integral` | Integrable W,V give absolute difference of integrals at most the integral of their absolute difference. Signed weights are allowed here. |
| `weightPerturbation_setIntegral` | The same bound holds for set integrals over any set A; the bound uses the total L1 error. |
| `weightPerturbation_normalizer_lower` | Reference integral at least c and L1 error at most delta imply perturbed integral at least c-delta. No positivity is needed for this inequality alone. |
| `weightPerturbation_event_integral_bounds` | A nonnegative integrable weight has every set integral between zero and its total integral. |
| `weightPerturbation_quotient_bound` | For positive z,t, 0<=a<=z, and both abs(a-b), abs(z-t) <=delta, the ratio error is at most 2delta/t. No sign premise on b is needed for this scalar statement. |
| `weightedLaw_event_perturbation` | With integrable a.e.-nonnegative W,V, integral(V)>=c>0 and L1 error<=delta<=c/2, conclude integral(W)>=c/2, both weighted laws are probability measures, and every measurable-event error is <=2delta/c. |
| `weightedLaw_event_perturbation_r2` | With r>0, reference integral>=cr^2 and L1 error<=eta*r^2, eta<=c/2, conclude integral(W)>=(cr^2)/2, both probability properties and every measurable-event error<=2eta/c. No r<=1 assumption. |

The ambient measure is arbitrary: it need not have finite mass or be a probability
measure. W and V always use the SAME underlying measure, and A is the SAME event
in both laws. Its dependence on either weight is unrestricted. Integrability is
an actual premise; no totalized divergent integral is used. Nonnegativity is only
almost everywhere, so values on null sets are irrelevant. The error-budget
hypotheses themselves imply delta>=0 (and eta>=0 in the scaled theorem).

## Mathematical argument and interpretation

Write a=int_A W, b=int_A V, z=int W, t=int V and Delta=int |W-V|.
Integral subtraction and the triangle inequality give abs(z-t)<=Delta and
abs(a-b)<=Delta. In particular z>=c-delta>=c/2>0. The existing weighted-law
construction then supplies the probability properties and event-ratio identities.
For the ratio bound use the exact identity

    a/z - b/t = ((a-b) + (a/z)*(t-z))/t.

Since 0<=a/z<=1, the numerator's absolute value is at most 2delta. Division by
t>=c gives the stated result uniformly over measurable A. The constant 2 is a
convenient valid bound; optimality is NOT claimed. The r^2 theorem is an algebraic
specialization with positive r, so the common vanishing scale cancels.

This is a uniform measurable-event bound. It does not instantiate mathlib's total
variation API or use a particular convention for the total variation norm.
The bound may exceed one; no sharpness or nontriviality at every budget is claimed.
A family application needs a fixed positive c and an actually controlled relative
budget eta(r). The module proves neither that eta(r) tends to zero nor that a
particular field has such a uniform bound.

## Nonclaims and review

No concrete Gaussian/typed Palm law, coupling of different base measures,
convergence of a density, reference normalizer floor, determinant approximation,
QS transfer, good-event complement, or persistence theorem is supplied. This
module does not replace a proof under mu_r with one under mu_0. Such applications
must first provide a legitimate common-measure representation and all displayed
estimates. A small absolute error alone is insufficient near a zero normalizer.
An absolute value AFTER integrating W-V cannot replace integrating |W-V|.

The earlier forty-target provenance and withheld alignment record remain exactly
as recorded. This seven-target addition requires its own nonauthor statement read;
it does not convert earlier withheld lineage into a full-package accepted record.
Any future complete alignment record must bind the new manifest and scope and
satisfy the unchanged lineage validator, including disclosed contributors.

## Controls

Sixteen focused Python methods cover exact signed integral inequalities, event
restriction, scalar ratios including negative b, all events on finite models,
nonunit and zero ambient masses, null-set changes, the half-budget endpoint and
r>1 scale cancellation. Countermodels reject omitting the normalizer, cancellation
inside a signed integral, substituting a different base law, signed-density
clipping, and normalizing a zero weight. These finite controls are not a Lean
formalization of the finite models or a proof for arbitrary measures.

The old generality test keeps its exact 36:40 inventory slice; the new test binds
the exact 40:47 extension and total 47. The unchanged gate independently checks
all declarations, hashes, build, leanchecker, transitive axioms and the original
five executable rejection controls. A supplied proof is not a successful run;
actual exact-commit execution and review evidence remain separate records.
