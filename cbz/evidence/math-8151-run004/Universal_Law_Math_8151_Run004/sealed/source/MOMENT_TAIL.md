# Actual moment-to-event-tail composition

Scientific effect: NONE. Dylan Roy — delegated AI work; actual performer OpenAI /
GPT-6 Astra Pro, 2026-10-05. This is an additive successor to the frozen 29-target
package in Math-#273, not a claim that the stack is merged or this new scope aligned.

## Source identity and design

Consumed predecessor: `d6g8k5htny-coder/Math-` at
`dce1b197be3431673289905259be38b450e9b898`. The original four Lean modules, all
29 theorem statements/proofs, gate and toolchain remain unchanged.
`WeightedLaw.lean` SHA256 is
`f31dfd2cd048cc0c0446c54c46d89c9cdd5d7bb9d76e9d8d16b7c034f351d736`.

The scalar P02-LM-009 source is the preserved fifth-power threshold, exponent40
identity and fourth-to-cubic ledger in `formal/ResearchFormalCoreR1/ProbabilityCompanionsV2.lean`
at that predecessor, SHA256
`338452a0c2ba24d016059354d543cc995b8cb29c9d4a41d98bb93499f5f8cb8f`.
The actual measure inequality is mathlib's
`MeasureTheory.mul_meas_ge_le_integral_of_nonneg` in
`Mathlib/MeasureTheory/Integral/Bochner/Basic.lean`, pinned mathlib
`d13f23b723b8a846827a245b89c10fc7d3f11612`; Lean remains v4.34.1.
This note gives a new explicit composition of those statements. It does not claim
a recovered full P02-LM-009 probability proof or a Gaussian supremum theorem.

## Exact derivation

For positive r and epsilon, define A = {x : epsilon < r R(x)^5}.
Its measurable-set property is proved from Measurable R. On A,

    0 < epsilon/r < R^5, hence (epsilon/r)^8 <= R^40.

A genuinely integrable fortieth power with integral at most M gives, by Markov,

    mu(A) <= M / (epsilon/r)^8 = (M/epsilon^8) r^8.

No nonnegativity of R is needed: membership in A supplies the positive threshold;
the fortieth power is nonnegative everywhere. Strict-boundary and negative-R
examples are tested. The generic Markov lemma is on a finite measure, not only
a probability; finite measure makes conversion of event masses to real numbers
monotone without turning an infinite mass into zero.

If the same probability measure and weight additionally satisfy W in L2,
W >= 0 almost everywhere, integral W^2 <= cW r^4, and cZ r^2 <= integral W,
with cW >= 0 and cZ > 0, the already constructed weighted probability law obeys

    P_W(A) <= C r^4,  C = (sqrt(cW)/cZ) sqrt(M/epsilon^8).

On 0 < r <= 1 this implies P_W(A) <= C r^3. Probability of P_W is proved as
part of each conclusion. W may depend arbitrarily on R and A; no independence
or replacement of a conditioned law by another law is introduced.

## Seven new audited targets (36 total)

| Suffix in ResearchFormalCoreR1 | Exact coverage |
|---|---|
| p02_lm009_markov_event | Event-contained-in-threshold Markov bound under actual integrability/nonnegativity and finite measure. |
| p02_lm009_badJet_measurable | Measurability of the strict bad event. |
| p02_lm009_badJet_subset | Event implies the fortieth-power threshold, including signed R. |
| p02_lm009_moment40_tail | Supplied integral fortieth-moment bound yields the actual eighth-order event tail. |
| p02_lm009_moment40_weighted_r4 | Actual weighted-law probability property and the stronger fourth-order bound. |
| p02_lm009_moment40_weighted_r3 | Cubic consequence on 0 < r <= 1. |
| p02_lm009_moment40_family_r3 | A single nonnegative C for 0 < r <= r0, only from uniformly supplied family premises with 0 < r0 <= 1. |

The family theorem explicitly quantifies possibly different mu_r,W_r,R_r under a
common measurable space, with the SAME mu_r for every moment/normalizer/event
at each r. M,cW,cZ,epsilon are fixed outside the r quantifier. It does not deduce
uniformity from unrelated pointwise constants. The witness C is displayed above.

## Retained open obligations

The input moment bound M, the weight second-moment bound, lower normalizer,
measurability/model construction and their constant uniformity must still be
proved for the concrete Gaussian field and typed Palm interpretation. No bound
on a derivative supremum, GT5 repair, Kac-Rice identification, full good event,
P0.2 or parent persistence theorem is supplied. The bad event here is exactly
one strict scalar threshold, not every component of the program's good event.
The older modules' exclusion notes remain true descriptions of those modules;
this new module supplies only the explicit moment-to-event implication.

## Verification and audit

The unchanged gate must build/recheck/audit all36 targets and its five existing
negative controls. A proof file or Python pass is not a kernel receipt. Twelve
focused tests include243 exact finite tail cases and972 weighted-event cases.
Countermodels expose omission of epsilon^-8, substitution of the20th for40th
moment, an incorrect r^16 rate, and strict-boundary/null-atom cases. They are
falsification checks, not continuum proof or an independent alignment review.
The prior weighted-law test now binds precisely targets20:29; the extension
binds targets29:36 and total36, with the unchanged gate still enforcing complete
source inventory. Every old proof byte remains frozen. Any successful audit
must state actual identity/exposure and exact new manifest/scope, not silently
transfer a previous29-target review to36 targets.

## Initial compiler readback

The first hosted run37315764990 failed before kernel evidence. The threshold
subset proof needed an explicit `change` from set membership to its inequality
goal; a `simpa only` list did not reduce that membership. The compiler-only
repair states this goal explicitly and removes a redundant `ring` after
`field_simp`. Every theorem type is unchanged. The original output is preserved
in `evidence/moment-tail-initial-build-failure.log`; it is not a success receipt.
