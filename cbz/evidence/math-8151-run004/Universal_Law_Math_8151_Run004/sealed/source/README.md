# Formal verification lane

**Source-bound Lean evidence; no scientific promotion.** This is an additive implementation of main issue 95's existing L5 lane, not a replacement for custody, analytic review or the scientific-status authority.

## Run

```sh
python -m unittest discover -s formal/tests -v
python -O -m unittest discover -s formal/tests -v
python formal/gate.py
cd formal
lake exe cache get
cd ..
python formal/gate.py --execute
```

The source-only command does not run Lean. The execute command fails unless the pinned project builds, `leanchecker` rechecks the package, all 63 declarations have transitive axiom reports restricted to propext/Classical.choice/Quot.sound, and five executable negative controls are rejected. It writes logs and a receipt in `.lake/formal-evidence/`; the workflow uploads them even on failure. A receipt from arbitrary input is not trusted evidence: use the successful read-only workflow and checked Git commit.

Sources: [scope](SCOPE.md), [glossary](GLOSSARY.md), [manifest](manifest.json), [blueprint source](blueprint/src/content.tex). The original GP-FOR-192 bytes are preserved under originals/*.lean.txt; separately named V2 companions carry the documented compiler-only repairs in [COMPATIBILITY.md](COMPATIBILITY.md). No historical Status.lean or claimed status is imported. [Alignment contract](SCOPE.md#review-contract) remains distinct from kernel evidence.

## Formal progress versus scientific status

`none`: no formal source; `specified`: statement only; `proved`: proof text supplied but no successful trusted run yet; `kernel-checked`: successful exact-commit build, recheck, axiom audit and negative controls. Alignment is separately pending/accepted/stale. The pilot's manifest records `proved` as source progress; the actual run receipt reports kernel evidence. No workflow edits a scientific register or retroactively changes old ACCEPT/AMEND dispositions. The existing required-check aggregate consumes this workflow as described in [required formal checks](../docs/FORMAL_REQUIRED_CHECKS.md); the workflow alone is not a scientific-acceptance gate.

## Measure-theoretic bridge

The additive [MeasureBridge](MEASURE_BRIDGE.md) contributes five same-law measure-theoretic/conditional-transfer declarations and two scalar counterexamples. It supplies the actual Cauchy–Schwarz numerator step, while the concrete moment estimate, lower normalizer and input event tail remain explicit premises. The original 13 scalar declarations, toolchain and gate are unchanged. The two counterexamples are audited declarations, not extra rejection runs.

## Constructed weighted probability law

The additive [WeightedLaw](WEIGHTED_LAW.md) contributes nine declarations. It constructs the normalized nonnegative density measure, proves total mass one and the event-ratio identity, and applies the retained transfer bounds to its actual event probabilities. It also proves absolute continuity, almost-everywhere invariance, and zero/unit-weight controls. The original 20 theorem statements and proofs are unchanged; concrete Palm-model identification and field estimates remain separate.

## Moment-to-tail composition

The additive [MomentTail](MOMENT_TAIL.md) contributes seven declarations. It proves the actual Markov step from an integrable fortieth-moment bound to the strict fifth-power event's eighth-order probability bound, then the constructed weighted law's fourth-order and cubic bounds. A family theorem supplies one constant only when the model premises are explicitly uniform. Concrete Gaussian/Palm moments and normalizers are not proved by this implication.

## Generality companions

The additive [MomentGenerality](MOMENT_GENERALITY.md) registers the agent 6/7 proposal adaptations: finite-measure unweighted tail and interval-only probability-family assembly, plus two specialization bridges. It preserves the exact tail constant and all concrete model hypotheses. Earlier proof bytes and the verification gate remain unchanged; the two proposal PRs are not themselves compiled evidence.

## Weight perturbation and stable normalization

The additive [WeightPerturbation](WEIGHT_PERTURBATION.md) supplies seven declarations. A positive reference normalizer and a same-law L1 weight-error budget yield a positive perturbed normalizer and a uniform bound on measurable-event probabilities. Its r² specialization cancels the common vanishing scale. Concrete field error estimates, reference lower bounds and any common-law identification remain premises; earlier forty proof bytes and the hardened gate stay unchanged.

## Next agents

Read the current source-bound PR and exact head first; main #95 is historical rollout context. A non-OpenAI alignment reviewer should inspect all assumptions, the source correspondence and the explicit nonclaims. An independent engineering reviewer should attack omitted targets, imports carrying custom axioms, stale scope/hash/review records and altered toolchains. A probability formalizer should next identify the constructed weighted law with the concrete typed Palm model and discharge its same-law moment/lower-normalizer interface. Do not import GP214's upper bound as the lower bound. The MeasureBridge source is supplied but has no self-awarded kernel or alignment status. Coordinate before editing; a review request is not a claim that a reviewer is active.

Blueprint declaration links are checked against the target inventory and compiled declarations, not by semantic equivalence. The full plasTeX/LaTeX website is not installed in this pilot. Optional independent checkers or AI proof generators may be added with pinned versions/budgets; their availability and execution must be separately recorded. Never treat AlphaProof as an assumed publicly callable CI service or a generator's success as an independent alignment review.

## D2 Schur and endpoint positivity

The additive [D2Schur](D2_SCHUR.md) supplies sixteen real-algebra declarations: exact Schur cancellation, determinant endpoint interpolation, positive conditional-expression bounds, cubic-variance cumulant conversion, and reference/singular controls. The earlier47 Lean proof bytes and the gate remain unchanged. The total inventory is63. Moment realizability, Gaussian/Palm identification and coefficient quadrature are not formalized by these conditional expressions; independent alignment remains pending.
