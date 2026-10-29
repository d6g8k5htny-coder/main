# Same-law measure-theoretic transfer companions

Scientific effect: NONE. Author: Dylan Roy — delegated AI work; actual performer
OpenAI / GPT-6 Astra Pro, ChatGPT session of 2026-10-05. These are new companions,
not edits to GP-FOR-192, its original modules, or an existing analytic proof.
Statement alignment is PENDING_INDEPENDENT_REVIEW. A supplied proof is not a claim
of successful Lean execution; only the actual exact-commit receipt can establish it.

## Source and dependency identity

The informal interface is P02-LM-008, LCR-CAP-024-v1.0, §§2–3 and §§6–8:
`d6g8k5htny-coder/Math-`, commit
`22876edaaec054d3ab8b0b16668ce3ff8cbf8c73`, path
`imports/hardening_ebedb780/P02-LM-008/proof.md.export.txt`.
This is a read-only correspondence reference, not a newly authenticated historical
review. The new source module consumes the unchanged scalar companions, whose
SHA-256 remains `338452a0c2ba24d016059354d543cc995b8cb29c9d4a41d98bb93499f5f8cb8f`.
Lean stays at `leanprover/lean4:v4.34.1`; mathlib stays at
`d13f23b723b8a846827a245b89c10fc7d3f11612`. Its
`MeasureTheory.integral_mul_le_Lp_mul_Lq_of_nonneg` in
`Mathlib/MeasureTheory/Integral/Bochner/Basic.lean` is specialized to exponents 2,2.

## Exact mathematical interface

Let μ be one probability measure, W an almost-everywhere nonnegative real function
with `MemLp W 2 μ`, and A a measurable event. Write

    q = (μ A).toReal, n = ∫_A W dμ, z = ∫ W dμ, M₂ = ∫ W² dμ.

The new declarations have the following coverage:

| Declaration suffix | Established implication |
|---|---|
| `p02_lm008_integral_cs` | Actual Bochner-integral Cauchy–Schwarz for nonnegative L² functions on a measure space, with no probability-measure restriction. |
| `p02_lm008_event_cs` | n ≤ √M₂ √q, using the measurable event indicator under the identical μ. |
| `p02_lm008_event_numerator` | M₂ ≤ cW r⁴ and cW ≥ 0 imply n ≤ √cW r² √q. |
| `p02_lm008_measure_transfer` | With r > 0, cZ > 0, cW ≥ 0, M₂ ≤ cW r⁴ and cZ r² ≤ z, derive z > 0 and n/z ≤ (√cW/cZ)√q. |
| `p02_lm009_measure_transfer_r8_to_r3` | Also q ≤ K r⁸, K ≥ 0 and r ≤ 1 imply n/z ≤ ((√cW/cZ)√K)r³, via the stronger intermediate fourth-order bound. |
| `p02_lm008_sqrt_counterexample` | Exact scalar sharpness/negative witness for omitting √q. |
| `p02_lm008_upper_normalizer_counterexample` | Exact scalar witness showing an upper normalizer bound cannot replace the required lower one. |

The common μ and W occur in the theorem types, not just in documentation. No
independence of A and W is assumed. Null/full events and zero weights on A are
allowed; the positive-normalizer hypothesis excludes the identically zero weight
from the ratio theorem. `MemLp` supplies genuine square-integrability rather than
using Lean's totalized integral to interpret a divergent second moment as zero.

## Deliberate remaining obligations

These are pointwise-in-r theorems. They do not construct a random field, a
conditioned law, a normalized `withDensity` probability measure, or a typed Palm
law. In particular PT1 (construction of the reweighted probability law) in the
informal source is not formalized here. The ratio is the interface being bounded.

A concrete application must still prove its own L²/measurability assumptions,
second-moment estimate, lower normalizer and event tail under this exact law and
weight, together with constant uniformity. GP214's upper bound supplies none of
the lower-normalizer premise. No Gaussian or GT5 assertion, family asymptotic,
exponential/super-polynomial corollary, P02-LM-006, P02-LM-007, P0.2, or parent
persistence theorem is closed by these companions. Historical source labels and
reviews are preserved, not promoted or transferred to this new code.

## Controls and execution

The existing gate is unchanged. It builds the complete registered package,
rechecks it with leanchecker, prints types and transitive axioms for all 20
declarations, and runs its original five rejection controls. All seven new
declarations are included in the same target/axiom inventory; the two counterexample
declarations are additional kernel propositions, not two extra rejection runs.

`tests/test_measure_bridge.py` adds 11 focused tests. Its 216 exact three-point
cases check the squared Cauchy–Schwarz and normalized bounds, dependent events,
zero weights and null/full events. Separate rational models expose removal of the
event square root, upper-for-lower normalization, first-for-second-moment and
cross-law substitution. These tests check finite models and source wiring only;
they cannot replace Lean execution or a semantic alignment review.

Review requests must identify the actual provider/family/agent and the exact new
manifest/scope. An OpenAI author replay or same-account commentary is not an
independent alignment acceptance. Existing gate/schema/toolchain and all original
proof bytes remain unchanged.
