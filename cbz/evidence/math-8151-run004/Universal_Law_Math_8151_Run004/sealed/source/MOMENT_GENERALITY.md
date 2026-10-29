# Finite-measure and interval-only companions

Scientific effect: NONE. Executable adaptation, tests and packaging: Dylan Roy —
delegated AI work; actual performer OpenAI / GPT-6 Astra Pro, 2026-10-05.
Proposal contributors: Grok Bot agents 6 (L6 / MT-N2) and 7 (L7 / MT-N1).
Their underlying providers/models are not established by their role labels.
Source progress is proof supplied; only an exact execution receipt establishes
kernel evidence. Source-to-statement alignment requires a separate actual review.

## Consumed sources and authorship

All references below are in `d6g8k5htny-coder/Math-`.

- Integration base: `a44db48f10ea27c93595840118457a39a5bbbea4`.
- Retained `formal/ResearchFormalCoreR1/MomentTail.lean`: SHA256
  `87fe3ee90afc52d8c2652881f1ae195d1b0b15a2ffc18c84d6166d1a4d0de2aa`.
- Audit motivating the two generalizations: #276,
  `reviews/moment_tail_alignment_20261005/REVIEW.md` at
  `0cb17c1b836651144486d55788eef8d7cf532081`, notes MT-N1 and MT-N2.
- Agent 6 proposal, #277 at `d1267a6106e8180eeb75fa008ce09de9ad8065ca`:
  `reviews/lean_cycle2_20261005/agent6/MomentTail_finite_MT_N2.lean.txt`.
- Agent 7 proposal, #278 at `cf58e5b812699cd7016441c0b477b86fecb7017f`:
  `reviews/lean_cycle2_20261005/agent7/MomentTail_family_interval_MT_N1.lean.txt`.

Those immutable proposal files remain UNCOMPILED historical proposals; their PR
checks do not compile text files. This new module preserves all four proposed
theorem types and the first three proofs, apart from comments/packaging. The final
specialization proof uses a direct conjunction pair instead of repeatedly
projecting and reconstructing the same hypothesis tuple. No original source or
proposal is overwritten, and no contributor is silently relabeled as OpenAI.

## Exact four-target extension: 36 to 40

All names below are in `ResearchFormalCoreR1`.

| Target | Exact conclusion and retained requirements |
|---|---|
| `p02_lm009_moment40_tail_finite` | For a finite measure, positive r and epsilon, integrable R^40 and its integral bounded by M, the real mass of `{epsilon < r R^5}` is at most `(M/epsilon^8)r^8`. |
| `p02_lm009_moment40_tail_of_finite` | Recovers the probability-measure version as a specialization of the finite-measure companion. |
| `p02_lm009_moment40_family_r3_interval` | Produces one nonnegative C for every 0<r<=r0, assuming probability and all moment/normalizer hypotheses only on that interval. The constant is chosen before r. |
| `p02_lm009_moment40_family_r3_of_interval` | Recovers the earlier globally-probability family interface from the interval-only theorem. |

The finite-measure statement is about MASS, not normalized probability. There is
no division by total measure mass, and event masses may exceed one. The zero
measure is allowed. Signed R, strict threshold boundaries, and null atoms are
retained. Integrability is explicit; no divergent totalized integral is used.
The ambient measure is finite, not arbitrary or merely sigma-finite.

The interval theorem assumes 0<r0<=1, positive epsilon and cZ, nonnegative M and
cW, and for each 0<r<=r0: probability of mu_r, nonnegative L2 weight W_r,
measurable R_r, integrable R_r^40, its integral <=M, weight second moment
<=cW*r^4, and the LOWER normalizer cZ*r^2<=integral W_r. It concludes the
weighted-law probability property and the cubic bound, with witness

    C = (sqrt(cW)/cZ) * sqrt(M/epsilon^8).

All constants are outside the r quantifier. Mu_r can be the zero measure outside
the interval. The two specialization targets deliberately have the old stronger
premises; they show compatibility, not two further generalizations. No theorem
asserts that every pointwise collection of constants admits one uniform constant.

## Nonclaims and review boundaries

No concrete Gaussian/typed Palm model, supremum moment, lower normalizer, global
good-event complement, P0.2 or parent persistence theorem is proved. The earlier
36 source theorem statements/proofs and their scope notes remain unchanged.
MT-N1/N2 remain accurate descriptions of the earlier theorem types; only the new
companions remove those restrictions. This is not a theorem-wide acceptance.

A reviewer must compare these exact statements to the two original proposals and
the retained formal sources, disclose actual lineage and any proposer exposure,
and distinguish new execution evidence from prior 36-target acceptance. Unknown
proposal-provider identity does not create cross-provider or organizational
independence. Full-package alignment must be explicit, not a relabeled old JSON.

## Tests and execution contract

Twelve focused standard-library Python tests include 729 rational finite-measure
cases (total mass 0, 1/2 and 2), scaling, strict/signed/null edges, a countermodel
to extra normalization, an interval family that is zero outside its domain, and
a diagnostic for r-dependent transfer constants. These are source-wiring and
finite-model checks, not kernel proofs or a formalization of the test models.

The previous moment-tail inventory test now binds its exact retained 29:36 slice;
the new test binds the exact 36:40 extension and total 40. The unchanged gate
independently checks the complete declaration inventory, builds and rechecks the
package, extracts all target types and transitive axioms, and runs the same five
rejection controls. Lean 4.34.1, mathlib
`d13f23b723b8a846827a245b89c10fc7d3f11612`, dependency pins, workflows and scientific
registers are unchanged. Source-only tests do not imply Lean execution.
