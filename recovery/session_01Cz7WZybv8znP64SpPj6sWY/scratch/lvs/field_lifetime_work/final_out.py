import json, pathlib
decls = json.load(open("decls.json"))
for o in decls:
    if "Lifetime" in o["module"] and o["does_not_claim"].startswith("(re-declaration"):
        o["does_not_claim"] = f"Re-declaration (same text and anchor) of UniversalLaw.Spec.Field.{o['name']} inside Specifications/Lifetime.lean so that the module imports only Mathlib; claims nothing beyond that entry."
M = "Mathlib/"
mathlib = [
 ("EuclideanSpace", "Analysis/InnerProductSpace/PiL2.lean:114"),
 ("EuclideanSpace.single", "Analysis/InnerProductSpace/PiL2.lean:300"),
 ("PiLp CoeFun (coordinate access `x i`)", "Analysis/Normed/Lp/PiLp.lean:92"),
 ("WithLp (structure, current API)", "Analysis/Normed/Lp/WithLp.lean:50"),
 ("OrthonormalBasis", "Analysis/InnerProductSpace/PiL2.lean:393"),
 ("OrthonormalBasis.instFunLike (frame vectors `R i`)", "Analysis/InnerProductSpace/PiL2.lean:409"),
 ("Real.exp", "Analysis/Complex/Exponential.lean:80"),
 ("tsum / `∑'` syntax", "Topology/Algebra/InfiniteSum/Defs.lean:142,160"),
 ("Summable (to_additive of Multipliable)", "Topology/Algebra/InfiniteSum/Defs.lean:114"),
 ("Finset sum syntax `∑ i, f i`", "Algebra/BigOperators/Group/Finset/Defs.lean:181"),
 ("ContDiff", "Analysis/Calculus/ContDiff/Defs.lean:1068"),
 ("ContDiff with `n : ℕ` (precedent contDiff_infty)", "Analysis/Calculus/ContDiff/Defs.lean:1115"),
 ("`ℕ∞ω` scoped notation = WithTop ℕ∞", "Analysis/Calculus/ContDiff/FTaylorSeries.lean:116"),
 ("iteratedFDeriv", "Analysis/Calculus/ContDiff/FTaylorSeries.lean:804"),
 ("fderiv", "Analysis/Calculus/FDeriv/Defs.lean:160"),
 ("Matrix.vecCons / `![v, w]` syntax", "Data/Fin/VecNotation.lean:61"),
 ("Submodule", "Algebra/Module/Submodule/Defs.lean:41"),
 ("Module.finrank", "LinearAlgebra/Dimension/Finrank.lean:62"),
 ("IsCompl", "Order/Disjoint.lean:320"),
 ("iInf syntax `⨅`", "Order/SetNotation.lean:71"),
 ("ConditionallyCompleteLinearOrder ℝ (for iInf on ℝ)", "Algebra/Order/Archimedean/Real/Basic.lean:136"),
 ("Set.Icc", "Order/Interval/Set/Defs.lean:76"),
 ("Set.Ioi (to_dual of Set.Iio)", "Order/Interval/Set/Defs.lean:36-37"),
 ("Set.ncard", "Data/Set/Card.lean:613"),
 ("Set.Finite", "Basic/Finite/Defs.lean:185"),
 ("MeasureTheory.Measure", "MeasureTheory/Measure/MeasureSpaceDef.lean:77"),
 ("MeasureTheory.IsProbabilityMeasure", "MeasureTheory/Measure/Typeclasses/Probability.lean:64"),
 ("MeasureTheory.Measure.map", "MeasureTheory/Measure/Map.lean:99"),
 ("MeasurableSet", "MeasureTheory/MeasurableSpace/Defs.lean:64"),
 ("Measurable", "MeasureTheory/MeasurableSpace/Defs.lean:493"),
 ("MeasureTheory.Integrable", "MeasureTheory/Function/L1Space/Integrable.lean:59"),
 ("MeasureTheory.IntegrableOn", "MeasureTheory/Integral/IntegrableOn.lean:92"),
 ("Bochner integral syntax `∫ x, f x ∂μ` / `∫ x in s, f x`", "MeasureTheory/Integral/Bochner/Basic.lean:168,174,177"),
 ("ae syntax `∀ᵐ ω ∂μ, p ω`", "MeasureTheory/OuterMeasure/AE.lean:57"),
 ("ProbabilityTheory.gaussianReal", "Probability/Distributions/Gaussian/Real.lean:222"),
 ("NNReal (def as subtype; anonymous constructor)", "Basic/NNReal/Defs.lean:58"),
 ("Real.toNNReal (fallback if the anonymous constructor fails)", "Basic/NNReal/Defs.lean:155"),
 ("Asymptotics.IsEquivalent", "Analysis/Asymptotics/AsymptoticEquivalent.lean:76"),
 ("nhdsWithin", "Topology/Defs/Filter.lean:138"),
 ("Real rpow `Pow ℝ ℝ`", "Analysis/SpecialFunctions/Pow/Real.lean:38"),
 ("abs syntax `|x|`", "Algebra/Order/Group/Unbundled/Abs.lean:43"),
 ("Path", "Topology/Path.lean:60"),
 ("Path.instFunLike (`γ t`)", "Topology/Path.lean:67"),
 ("unitInterval", "Topology/UnitInterval.lean:32"),
 ("instance-implicit structure field precedent (TopCat.str)", "Topology/Category/TopCat/Basic.lean:38"),
 ("Mathlib root module (import Mathlib)", "Mathlib.lean (8529 imports; package root ../Mathlib.lean)"),
]
res = {
 "item_id": "spec-field-lifetime",
 "summary": "Wrote the two statement-only Mathlib modules of the lifetime track: Specifications/Field.lean (namespace UniversalLaw.Spec.Field, 26 declarations: periodized covariance K_L as a quotient of tsum theta sums, Gaussian-field interface with gaussianReal 0 v laws and covariance identity, critical points, Hessian via iteratedFDeriv, Sylvester-inertia index predicates, nondegenerate maxima and index-(d-1) saddles, pins, torus distance, fundamental domain, height/scaled gap, candidate pairs, Morse-distinct predicate, superlevel-bar/elder interface with the Section 8 maximin clause, finite bars, pair counts and the density-version predicate) and Specifications/Lifetime.lean (namespace UniversalLaw.Spec.Lifetime, 38 declarations: the same 26 re-declared verbatim plus 12 new: window/unrestricted populations, PairLawInterface (p_r(b,k,R) as a given [0,1]-valued function of r, b, k and an OrthonormalBasis frame), ParentUniformSelectionA (1.1), ParentCompactWindowDensityB (1.2), IsUnrestrictedLeadingCoefficient and ParentUnrestrictedLeadingDensityC (1.3), RemainderBoundedR1 (R1), PlanarElderLowerE2 (E2), PlanarTwoSidedRateE4 (E4), PlanarDensityGapE11 (E11)). All quantifiers kept explicit: fixed d >= 2 and L > 0 as hypotheses, compact B and K, all frames, existential r_*, C, c > 0, ell_* quantified outside the universal quantifiers; versions as existential density functions pinned by an integral identity; ~ via Asymptotics.IsEquivalent on nhdsWithin 0 (Set.Ioi 0); bounded remainders as explicit |..| <= C. Every anchor is a verbatim substring of the byte-pinned sources (checked against sha256-verified bytes). Nothing was compiled: the pinned Mathlib oleans for the analysis modules do not exist yet (1947/8529 built). Scientific effect NONE; formal progress of everything: specified; no claim promoted; alignment PENDING_INDEPENDENT_REVIEW.",
 "files": [
  "<SCRATCH>/repos/math/specifications/lean_statements_20261010/Specifications/Field.lean",
  "<SCRATCH>/repos/math/specifications/lean_statements_20261010/Specifications/Lifetime.lean",
  "<SCRATCH>/lvs/out/spec-field-lifetime.json",
  "<SCRATCH>/lvs/field_lifetime_work/static_check.py"
 ],
 "file_identities": {
  "Field.lean": {"bytes": 26885, "sha256": "9faa80870035a3e90f74844bbef96c7d303eead8741bea92a12d77197cf318d5"},
  "Lifetime.lean": {"bytes": 47342, "sha256": "b92378c93ba3b938a0e8921690f2f7e17a413cdca4f7f20c26abba529920c57f"}
 },
 "declarations": decls,
 "mathlib_identifiers": [{"name": n, "file_line": M + f} for n, f in mathlib],
 "checks_run": [
  "Read DESIGN.md completely; read SOURCES.json and the three cited sources in full (lifetime-parent 40261 B, lifetime-remainder 17734 B, elder-lower-density-gap 10229 B); read LANDING_CLAIMS.json, the lifetime nodes of GRAPH.json and PROOF_INDEX.md for claim ids.",
  "static_check.py (scratchpad, NON-CERTIFYING lexical check, no Lean): re-hashed every SOURCES.json local copy (sha256 and byte count match); every `Anchor: \"...\"` line of every declaration is a verbatim byte substring of the source named by the sha256 in its `Source:` line (64/64 pass); no anchor contains a comment delimiter; lexical refusal list (theorem, lemma, example, instance, axiom, sorry, native_decide, unsafe, opaque, partial, macro, notation, set_option as case-sensitive tokens; verified/proved/closed/certified as case-insensitive substrings) finds nothing in either module; docstring delimiters balanced; exactly one namespace per module with matching end; imports are exactly [Mathlib]; every def/abbrev/structure carries a docstring with the four labelled lines Source:, Anchor:, Interface:, Does not claim:.",
  "Shared-block identity: the 26 shared declarations of Field.lean (24543 bytes from `/-- The ambient Euclidean space` to `end`) occur byte-identically in Lifetime.lean (only the namespace line differs).",
  "Every Mathlib identifier and syntax used was confirmed by grep in the pinned source tree <SCRATCH>/mathf/.lake/packages/mathlib/Mathlib (file:line recorded in mathlib_identifiers); house precedent for `import Mathlib` taken from Math- formal/ResearchFormalCoreR1/AlgebraV2.lean and ProbabilityCompanionsV2.lean.",
  "Checked that no lake process was started and no path outside Specifications/Field.lean, Specifications/Lifetime.lean, lvs/out and the scratchpad work directory was touched; git status in the Math- clone shows only the untracked specifications/ directory; nothing committed or pushed.",
  "NOT run: Lean elaboration (`lake env lean`) of either module. The pinned Mathlib oleans for Analysis/InnerProductSpace/PiL2, Analysis/Calculus/ContDiff/Defs, Analysis/SpecialFunctions/Pow/Real, Analysis/Asymptotics/AsymptoticEquivalent, Probability/Distributions/Gaussian/Real, MeasureTheory/Integral/Bochner/Basic and Topology/Path do not exist yet (1947 of 8529 oleans built when checked), so the modules are unelaborated text."
 ],
 "uncertainties": [
  "Neither module has been elaborated by Lean; every statement below is a residual elaboration risk, not a confirmed error.",
  "`import Mathlib` is used (Math- formal/ house style); it requires the full Mathlib cache in CI. If the packet prefers narrow imports, the defining files listed in mathlib_identifiers are the minimum set (plus their transitive closure).",
  "ContDiff ℝ n g with n : ℕ relies on the coercion ℕ → WithTop ℕ∞ (precedent: contDiff_infty, Analysis/Calculus/ContDiff/Defs.lean:1115).",
  "The variance argument `⟨covQuadForm d L x a, quad_nonneg x a⟩ : ℝ≥0` relies on the anonymous constructor unfolding `NNReal := { r : ℝ // 0 ≤ r }` (Basic/NNReal/Defs.lean:58); fallback is `Real.toNNReal (covQuadForm d L x a)` (Basic/NNReal/Defs.lean:155), harmless given the quad_nonneg field.",
  "GaussianFieldInterface bundles `Ω : Type` with instance-implicit fields `[mΩ : MeasurableSpace Ω]` and `[isProb : IsProbabilityMeasure P]` (precedent TopCat.str); later fields (`P : Measure Ω`, `Measure.map`, `∫ … ∂P`) rely on these being local instances inside the structure declaration. The structure lives in Type 1.",
  "PairLawInterface carries parameters `G` and `I` that no field mentions (documentation of what the law refers to); Lean accepts unused structure parameters but a linter may warn.",
  "Coordinate access `x i` on EuclideanSpace in FundDomain relies on the PiLp CoeFun (Analysis/Normed/Lp/PiLp.lean:92) seeing through the abbrevs `E` and `EuclideanSpace`.",
  "`Module.finrank ℝ V` for `V : Submodule ℝ (E d)` relies on the CoeSort coercion; `iteratedFDeriv ℝ 2 g x ![v, w]` relies on the ContinuousMultilinearMap coercion to functions on `Fin 2 → E d`.",
  "HessianHasInertia encodes 'exactly j negative and d-j positive eigenvalues' by Sylvester inertia (complementary negative-definite and positive-definite subspaces of dimensions j and d-j); equivalence with the eigenvalue count needs symmetry of the Hessian, true for C^2 functions but not stated.",
  "torusDist is an iInf over Z^d in ℝ (ConditionallyCompleteLinearOrder); it is bounded below by 0 so not junk; `0 < torusDist` encodes 'distinct torus points'.",
  "pairCount uses Set.ncard (0 on infinite sets); IsLifetimeDensityVersion therefore demands almost-sure finiteness and integrability of the counts and integrability of ν on each Borel A ⊆ (0,∞); a version exists only if these hold, which is part of what the source asserts. If the counts were not a.e. measurable no version could exist (statement false, not vacuous).",
  "The elder relation is pinned (iff) only on the Morse distinct-value locus for smooth L-periodic g, exactly as the source; off that locus ElderPartner is unconstrained. Statements are parametrised by the interface object (the 'any object satisfying the interface' reading), so an object pathological off a P-null set does not change expectations, but the identification of the interface object with the source's elder rule is explicitly not claimed.",
  "IsEssential is characterised as 'no strictly higher point' (global maximum), and deathHeight by `h < d_f(M) ↔ ∃ y, g M < g y ∧ ∃ γ : Path M y, ∀ t, h < g (γ t)`; this equals the source's sup–min for continuous g on compact paths; both clauses are stated under IsSmooth ∧ IsLPeriodic hypotheses so the interface is satisfiable (the maximin function itself realises it).",
  "RemainderBoundedR1 pins the coefficient c through IsUnrestrictedLeadingCoefficient (the leading-term role of (1.3)) because the explicit integral (15.2) is not transcribed; the statement is vacuous if no such c exists. The alternative `∃ c > 0` form is self-contained but drops the identification 'c is exactly the parent's (15.2)'. Reviewer choice; recorded in the docstring.",
  "ParentCompactWindowDensityB and PlanarDensityGapE11 quantify the distance cutoff rStar existentially (the source uses Theorem A's r_*), and c_{B,K} existentially (not the Gaussian integral (11.3)); the difference bound of (1.2) is stated for 0 < ℓ ≤ ℓStar with ℓStar existential, which is the 'as ell decreases to zero' reading.",
  "Frames are `OrthonormalBasis (Fin d) ℝ (E d)`; since p is a primitive interface field the axial vector u = R e_1 and the pins pinM/pinS (defined in Field.lean) are not consumed by any Prop; they are documentation objects.",
  "The source names 'Theorem A/B/C/R' appear capitalised in docstrings; if the packet checker refuses `theorem` case-insensitively these docstrings need rewording (the lowercase token never appears).",
  "The 26 shared declarations are duplicated in Lifetime.lean under UniversalLaw.Spec.Lifetime; REGISTER.json must list both copies, and a drift check between the two copies (byte identity modulo the namespace line) should be added to check.py; the duplication is deliberate per the task.",
  "Anchors containing markdown `**` (p_r clause) and the exact-spacing (E11) and pin lines were checked byte-for-byte; the register must copy them unchanged.",
  "Item id for the out JSON was not given; `spec-field-lifetime` was used (file lvs/out/spec-field-lifetime.json).",
  "Claim-id labels of PROOF_INDEX style (D1-parent-lifetime-Theorem-A-(1.1), D2-lifetime-remainder-Theorem-R-(R1), D1-elder-lower-(E2)/(E4), D1-elder-lower-density-gap-(E11)) are constructed from PROOF_INDEX.md's D-labels and the sources' equation tags; the landing ids lifetime-remainder and side24-coefficient and the graph ids math.uniform-matrix-cap-lifetime, math.lifetime-remainder, math.side24-coefficient are verbatim. ELDER_LOWER_AND_DENSITY_GAP.md has no landing claim or graph node; its PROOF_INDEX object id OA-ELDER-LOWER-DENSITY-GAP-20260928-v1 is used.",
  "side24-coefficient is attached only to the two Theorem C declarations because its landing row names Theorem C / eq. 15.2 as 'required for persistence interpretation'; this records the dependency direction, not any coefficient value."
 ],
 "gaps": [
  "The explicit Gaussian/angular representation (15.2) (and (13.6), (11.3)) of c_{d,L} / c_{B,K} is not transcribed; the coefficient enters only through its leading-term role.",
  "The regression law Q_{r,b,k,R}, the typed weight W_r = |det H_M det H_S| 1{H_M<0, index(H_S)=d-1}, the full normalizer Z_r and Q^W are not transcribed; p_r(b,k,R) is a primitive interface field.",
  "The marked Kac-Rice representation (Sections 9-10, radial ledger (10.2)) is not transcribed; densities are pinned only by the integral identity of IsLifetimeDensityVersion.",
  "Not stated: the cumulative/moment corollaries (12.1)-(12.3) and (R19)-(R20); the off-diagonal bound (14.1); the normalizer floor (5.5); the parent's Sections 2-8 interfaces (positive spectrum, contact frame, moments, boundary layer, Borel genericity); the marked-cylinder-cap deterministic support; the reading-rule amendments (congruence erratum, Section 9 replacement v1.1, W1, embedding radius r < L/(4 sqrt 2)); (E3), (E7)-(E10) of the planar source.",
  "No Lean elaboration, no kernel receipt, no `#check`/`#print axioms` run; formal progress is `specified` only after the packet's --execute mode elaborates the modules.",
  "REGISTER.json rows for these 64 declarations, the root Specifications.lean imports, check.py's refusal list and drift check, README/SCOPE text and the workflow are other items; nothing outside the two modules and the out JSON was written to the packet."
 ]
}
out = pathlib.Path("<SCRATCH>/lvs/out/spec-field-lifetime.json")
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("written", out, len(out.read_bytes()), "bytes;", len(decls), "declarations;", len(res["mathlib_identifiers"]), "mathlib ids")
