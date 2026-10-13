# Proof packet contract (LVS phase 4): `specified` → `proved`

Read `DESIGN.md` first (vocabulary, environment, honesty rules); this file adds the proof-packet rules.
Scientific effect NONE. A kernel-checked proof of a registered Prop is evidence about that Prop only;
it never makes the informal claim accepted, reviewed or "verified". Credit 0 (Anthropic / Claude).

## Where
* Statement packet (READ-ONLY for proof workers): `<SCRATCH>/repos/math/specifications/lean_statements_20261010/`
  — modules `Specifications/{Field,Lifetime,Side24,RN,P15}.lean`, register `REGISTER.json`.
  Never edit a statement module: a proof that needs the statement changed is a FINDING (report it),
  because changing a statement stales the register and the review lane.
* Proof packet (the place to write): `<SCRATCH>/repos/math/specifications/lean_statement_proofs_20261011/`
  — `lakefile.toml` (lean_lib `SpecificationProofs`, path dependency on the statement packet, Mathlib
  pinned), root `SpecificationProofs.lean`, modules `SpecificationProofs/<Family>.lean`.
  `.lake/packages` is a symlink to the fully built Mathlib; **never run `lake update`, never touch
  `.lake/packages`, never run lake inside `scratchpad/mathf`**.
* Build/check a module quickly (10–40 s each; Mathlib is prebuilt):
  `cd <proof packet> && export PATH="$HOME/.elan/bin:$PATH" && lake env lean -DwarningAsError=true SpecificationProofs/<Family>.lean`
  (the root import list must include your module before `lake build` builds it; `lake build` of the
  whole packet is ~1–3 min because Lake rehashes the Mathlib tree; prefer `lake env lean` per file).
  Axiom check: append `#print axioms UniversalLaw.Proofs.<Family>.<name>` to a scratch copy and run it,
  or use `lake env lean` on a scratch file importing your module after a `lake build`.

## Theorem shape (the gate enforces this)
* Every theorem's TYPE is exactly a registered Prop constant applied to its parameters:
  `theorem rhoStar_below_six_sevenths : UniversalLaw.Spec.P15.RhoStarBelowSixSevenths := by …`
  or, for parametrised Props, `theorem foo (X : Type) [Fintype X] … : UniversalLaw.Spec.P15.SomeProp X … := …`
  with all parameters universally quantified (no extra hypotheses beyond the Prop's own parameters —
  if the Prop needs an extra hypothesis to be true, that is a FINDING about the statement, not a
  theorem with a smuggled premise).
* Namespace `UniversalLaw.Proofs.<Family>`, one per module. Modules import only `Specifications.<X>`
  modules and Mathlib.
* Helper lemmas are allowed (`theorem`/`lemma` with other types) but must be `private` or live in a
  `namespace UniversalLaw.Proofs.<Family>.Aux` block; the gate's inventory lists only the non-aux
  theorems and requires each to have a registered Prop as type (checked by `#check` + head-constant
  parse in the environment audit).
* No `sorry`, no `axiom`, no `native_decide`, no `decide` on ℝ (it does not reduce), no `unsafe`,
  no `set_option maxHeartbeats` above 400000 without a comment saying why, no `Classical` tricks that
  change the statement. Allowed transitive axioms: propext, Classical.choice, Quot.sound only.
* Docstring per theorem: `Proves:` (the registered Prop and the claim ids it serves, from REGISTER.json),
  `Method:` (one line), `Does not establish:` (the informal claim, review, acceptance; what the Prop
  leaves out as its own docstring says).

## Targets (tractability order; prove what you can, report precisely what you could not)
Family P15 (module P15.lean, statement module `Specifications/P15.lean`):
  demand-one counterexample: `TwoCoordGeneratorPrices`, `DemandOneRefutation`, `DemandOneBoundary`,
  `LogOnePlusBelowIdentity`, `LogRatioBound`, `CapTestAtTwoThirds`; ρ*: `RhoStarBelowSixSevenths`,
  `SixSeventhsCertificate`, `HStarExceedsOne`, `PhiAtPStar`, `HazardStarPositive`; algebra:
  `RatioExponentBound`, `OddMajorityIdentity`, `CapacityWorstCase`; benchmark instance:
  `Benchmark816Palette`, `Benchmark816WholeGround`, `Benchmark816MinimalForbiddenCounts`,
  `Benchmark816UnionBound`, `Benchmark816IsRealized`, `Benchmark816Price`, `Benchmark816RationalPrice`
  (large finite objects: `decide` on Finset of 2454 elements will NOT work; use `norm_num`/`simp`
  with the structure's data, or report infeasible); structure: `LocalDecomposability`,
  `LocalObstructionIsFullBlock`, `MinimalForbiddenCharacterisation`, `FullBlockCoverIff`,
  `CoverCostHazardBound`, `ElementaryPaletteFormula`, `TriangleNeedsThree`, `WholeGroundChromaticNumber`
  (genuine combinatorics; attempt only after the easier ones; partial progress is not committed).
  Hard analysis (attempt last, likely not now): `RhoStarEnclosure` (20 digits), `CoordinatewiseConcavity`,
  `HazardProductBound`, `FullTransformedPriceBudget`, `SharpnessOfRhoStar`, `RestrictedTransformedPriceBudget`.
Family Side24 (module Side24.lean): `ExponentialBounds` (e^{288/125} > 10 etc.), `SuccessiveTermRatio`,
  `LatticeImageSumBound` (if it is the final 1458 e^{-288} bound over a tsum — may need tsum estimates),
  `ConeMomentD2Algebra`, `TraceMoments`?, `ReferenceOddAndPinBlocks` (matrix eigen facts — maybe),
  `ReferenceCoefficientIdentity` (algebraic identity between cRef and the parent formula? check),
  `ElementaryConeIntegral` (∫_0^a (a-z)^2 e^{-z/2} dz/2 = a^2-4a+8-8e^{-a/2}: interval integral by FTC — doable),
  `RayleighSquareLaw`/`ConeMomentD1Identity`/`ConeMomentD2Identity`/`TraceMoments` need Gaussian moment
  lemmas — attempt only if Mathlib has `ProbabilityTheory.gaussianReal` moment lemmas you confirm by grep.
Family RN (module RN.lean): `HolderCountBoundN1`, `HolderConsequenceN2`, `ConditionalMeanIdentityN3`,
  `UniformCounterexampleN4`, `ExponentialTailLayerCakeN5` (Mathlib: integral_mul_le_Lp_mul_Lq_of_nonneg,
  ProbabilityTheory.cond, MeasureTheory.lintegral_eq_lintegral_meas_lt / layercake).
Family Lifetime/Field: nothing is expected to be provable (interface-level statements); do not spend time.

## Reporting (structured output)
For each attempted target: name, status (PROVED | NOT_PROVED), the Lean declaration name if proved,
axioms reported by `#print axioms`, and for NOT_PROVED the exact obstacle (missing Mathlib lemma, a
statement defect = FINDING with the proposed statement fix, or out of time). Never report PROVED
without having seen `lake env lean` exit 0 with warnings as errors on the module AND `#print axioms`
within the allowed set.
