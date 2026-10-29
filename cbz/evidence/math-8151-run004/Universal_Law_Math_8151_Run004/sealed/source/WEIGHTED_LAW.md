# Constructed weighted probability law

Scientific effect: NONE. Dylan Roy — delegated AI work; actual performer OpenAI /
GPT-6 Astra Pro, 2026-10-05. This is an additive successor to Math-#272, not a
rewrite or acceptance of its proof. Independent statement alignment is pending.

## Design and exact source correspondence

The source is P02-LM-008 (LCR-CAP-024-v1.0), §§2–3 and PT1 in §4, at
`d6g8k5htny-coder/Math-`, commit
`22876edaaec054d3ab8b0b16668ce3ff8cbf8c73`, path
`imports/hardening_ebedb780/P02-LM-008/proof.md.export.txt`.

The consumed formal bridge is Math-#272 at
`270167900b71539d0e2135131a3a96f1f04289f6`, module
`formal/ResearchFormalCoreR1/MeasureBridge.lean`, SHA-256
`28ac130db490c3c3e2fbfabf3ac5be43c854ab1468bf573bf5a8c0dd2c50955f`.
It is preserved byte-for-byte. The dependency is a kernel-checked candidate,
not a claim that #272 is merged or independently aligned.

For one measure μ and real function W, the definition is

    weightedLaw μ W = ofReal(∫ W dμ)⁻¹ • μ.withDensity (ofReal ∘ W).

This is a measure by construction, including countable additivity. Its probability
property is proved rather than assumed, under W integrable, W ≥ 0 almost everywhere,
and ∫ W dμ > 0. No finiteness or probability hypothesis on μ is needed for this
construction theorem. The applications additionally take μ to be a probability
measure and derive integrability from `MemLp W 2 μ`.

## Nine new audited declarations

| Suffix in ResearchFormalCoreR1 | Exact coverage |
|---|---|
| weightedLaw_apply | Measurable-event ENNReal mass equals inverse ofReal normalizer times ofReal set integral. |
| weightedLaw_isProbabilityMeasure | Integrability, a.e. nonnegativity and positive normalizer imply total mass one. |
| weightedLaw_event_real | Under those conditions, real event mass is (∫_A W dμ)/(∫ W dμ). |
| weightedLaw_absolutelyContinuous | Original-law null sets remain null; no independence or W > 0 assumption. |
| weightedLaw_congr_ae | Null-set changes of the weight leave the entire measure unchanged. |
| weightedLaw_zero | The zero weight yields zero measure, exposing the need for positive normalization. |
| weightedLaw_one | Unit weight on a probability space recovers μ. |
| p02_lm008_probability_transfer | Constructs a probability law and proves its event bound from the same moment/lower-normalizer assumptions as #272. |
| p02_lm009_probability_transfer_r8_to_r3 | Constructs that law and composes the supplied eighth-order event bound with the cubic transfer. |

The new definition is consumed transitively by these audited targets. The package
now has 29 theorem targets; the original 20 theorem statements and proof bytes are
unchanged. The older bridge scope's PT1 exclusion describes that earlier module;
this new module fills the abstract probability-law construction only.

## Essential hypotheses and nonclaims

`ofReal` discards negative values. Without a.e. nonnegativity its density integral
need not equal the Bochner normalizer: weights (-1,3) on two equal atoms give
normalizer 1 but clipped normalized total mass 3/2. A zero normalizer gives no
probability theorem. Real integral hypotheses never replace integrability with
an inequality on Lean's totalized integral of a divergent function.

This does not identify the constructed law with the program's concrete typed
Palm/conditioned field law. Concrete model correspondence, moment bounds, lower
normalizers, event tails, constant uniformity, GT5, P0.2, and the parent persistence
theorem remain separate obligations. No new hypothesis independence is asserted.
Historical analytical reviews are not inherited by these new formal statements.

## Verification and handoff

Lean 4.34.1, mathlib d13f23b723b8a846827a245b89c10fc7d3f11612, the original gate,
all five executable rejection controls, and the hosted workflow are unchanged.
The same gate must build, recheck, extract types, and audit all 29 targets.
The two prior scalar counterexamples remain target propositions, not rejection runs.

Thirteen focused Python tests include 208 exact rational event cases, null atoms,
a.e.-equal representatives, constant and rescaled weights, finite partitions,
zero-normalizer and signed-weight failures. These are finite falsification/wiring
checks, not a proof of countable additivity or a substitute for the Lean kernel.
The prior bridge-inventory test now checks its exact retained 13:20 slice; this
successor checks the complete 20:29 extension and total 29, so no target is dropped.

Review work on #272 stays isolated from this successor. Any helper must report its
actual model/provider/runtime and claimed scope. A dispatch, automated code review,
or successful build does not supply independent semantic alignment by itself.
