"""Compose the fixer's structured result for item spec_field_lifetime and write it under both the
orchestrator's spelling and the author's earlier spelling. Scientific effect: NONE."""
import json, hashlib, pathlib
SP = pathlib.Path("<SCRATCH>")
PKT = SP / "repos/math/specifications/lean_statements_20261010"
FIELD, LIFE = PKT / "Specifications/Field.lean", PKT / "Specifications/Lifetime.lean"
decls = json.loads((SP / "lvs/fixer_fl/decls.json").read_text(encoding="utf-8"))


def ident(p: pathlib.Path):
    b = p.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


fid, lid = ident(FIELD), ident(LIFE)
M = "Mathlib/"
mathlib = [
    ("EuclideanSpace", "Analysis/InnerProductSpace/PiL2.lean:114"),
    ("EuclideanSpace.single (abbrev, [DecidableEq ι] variable at :296)", "Analysis/InnerProductSpace/PiL2.lean:300"),
    ("PiLp CoeFun instance (coordinate access `x i`)", "Analysis/Normed/Lp/PiLp.lean:92"),
    ("WithLp (structure, current API)", "Analysis/Normed/Lp/WithLp.lean:50"),
    ("OrthonormalBasis (structure; variable (ι 𝕜 E) order)", "Analysis/InnerProductSpace/PiL2.lean:393"),
    ("OrthonormalBasis.instFunLike", "Analysis/InnerProductSpace/PiL2.lean:409"),
    ("Real.exp", "Analysis/Complex/Exponential.lean:80"),
    ("Summable (to_additive of Multipliable)", "Topology/Algebra/InfiniteSum/Defs.lean:114"),
    ("tsum notation `∑' i, f i`", "Topology/Algebra/InfiniteSum/Defs.lean:160"),
    ("Finset sum syntax `∑ i, f i` (bigsum)", "Algebra/BigOperators/Group/Finset/Defs.lean:181"),
    ("ContDiff", "Analysis/Calculus/ContDiff/Defs.lean:1068"),
    ("ContDiff with `n : ℕ` (precedent contDiff_infty)", "Analysis/Calculus/ContDiff/Defs.lean:1115"),
    ("`ℕ∞ω` = WithTop ℕ∞ (module docstring)", "Analysis/Calculus/ContDiff/FTaylorSeries.lean:102"),
    ("iteratedFDeriv", "Analysis/Calculus/ContDiff/FTaylorSeries.lean:804"),
    ("iteratedFDeriv ℝ 2 f x ![v, w] application precedent", "Analysis/InnerProductSpace/Laplacian.lean:175"),
    ("fderiv (irreducible_def)", "Analysis/Calculus/FDeriv/Defs.lean:160"),
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
    ("Measure.instOuterMeasureClass [MeasurableSpace α]", "MeasureTheory/Measure/MeasureSpaceDef.lean:95"),
    ("MeasureTheory.IsProbabilityMeasure", "MeasureTheory/Measure/Typeclasses/Probability.lean:64"),
    ("MeasureTheory.Measure.map (irreducible_def, [MeasurableSpace α])", "MeasureTheory/Measure/Map.lean:99"),
    ("MeasurableSet", "MeasureTheory/MeasurableSpace/Defs.lean:64"),
    ("Measurable", "MeasureTheory/MeasurableSpace/Defs.lean:493"),
    ("MeasureTheory.Integrable ({_ : MeasurableSpace α}; first conjunct AEStronglyMeasurable)", "MeasureTheory/Function/L1Space/Integrable.lean:59"),
    ("MeasureTheory.AEStronglyMeasurable (named in a Does-not-claim line)", "MeasureTheory/Function/StronglyMeasurable/AEStronglyMeasurable.lean:66"),
    ("MeasureTheory.IntegrableOn", "MeasureTheory/Integral/IntegrableOn.lean:92"),
    ("MeasureTheory.integral ({_ : MeasurableSpace α})", "MeasureTheory/Integral/Bochner/Basic.lean:158"),
    ("Bochner integral notations `∫ x, f x ∂μ` / `∫ x in s, f x`", "MeasureTheory/Integral/Bochner/Basic.lean:168,171,174"),
    ("MeasureTheory.ae", "MeasureTheory/OuterMeasure/AE.lean:49"),
    ("ae notation `∀ᵐ x ∂μ, p x`", "MeasureTheory/OuterMeasure/AE.lean:57"),
    ("OuterMeasureClass (α : outParam)", "MeasureTheory/OuterMeasure/Defs.lean:64"),
    ("ProbabilityTheory.gaussianReal (μ : ℝ) (v : ℝ≥0)", "Probability/Distributions/Gaussian/Real.lean:222"),
    ("NNReal (subtype; anonymous constructor)", "Basic/NNReal/Defs.lean:58"),
    ("Real.toNNReal (fallback if the anonymous constructor fails)", "Basic/NNReal/Defs.lean:155"),
    ("Asymptotics.IsEquivalent (l u v)", "Analysis/Asymptotics/AsymptoticEquivalent.lean:76"),
    ("nhdsWithin", "Topology/Defs/Filter.lean:138"),
    ("Real rpow instance `Pow ℝ ℝ`", "Analysis/SpecialFunctions/Pow/Real.lean:38"),
    ("abs syntax `|x|`", "Algebra/Order/Group/Unbundled/Abs.lean:43"),
    ("Path", "Topology/Path.lean:60"),
    ("Path.instFunLike (`γ t`)", "Topology/Path.lean:67"),
    ("unitInterval", "Topology/UnitInterval.lean:32"),
    ("TopCat.str (square-bracket structure field precedent)", "Topology/Category/TopCat/Basic.lean:38"),
    ("ModuleCat `attribute [instance]` precedent (why a square-bracket field needs re-registration; `attribute` is a refused token here, hence `haveI`)", "Algebra/Category/ModuleCat/Basic.lean:64"),
    ("Mathlib root module (import Mathlib)", "Mathlib.lean (package root)"),
    ("haveI term-level parser (core Lean 4.34.1, not Mathlib)", "lean-toolchain/lean-4.34.1-linux/src/lean/Lean/Parser/Term.lean:578-579"),
]
res = {
    "item_id": "spec_field_lifetime",
    "role": "FIXER (verifier verdict AMEND applied)",
    "scientific_effect": "NONE",
    "scientific_status_authority": False,
    "alignment_status": "PENDING_INDEPENDENT_REVIEW",
    "organizational_independence_credit": 0,
    "summary": (
        "Applied the verifier's AMEND to the two statement-only Mathlib modules of the lifetime track, in the assigned paths only. "
        "Blocking fix: PlanarTwoSidedRateE4 now carries the parent upper input as an explicit hypothesis, exactly as the source states Corollary E2 "
        "('If the precise Theorem A upper bound in [LP] is consumed, then in d=2 ...'): `ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi → (... c r^3 ≤ 1 - p ≤ C r^3)`; "
        "docstring, Interface and Does-not-claim reworded, Anchor unchanged and verbatim. Minor fixes applied: (2) PlanarDensityGapE11 likewise carries the consumed parent upper input as a hypothesis, "
        "with a new parameter `PL : PairLawInterface 2 L G I` — on the evidence of the source (elder lines 43-46 and 165-168: '(E2)/(E4) imply (E11)', 'the upper consumes the already identified parent upper theorem', the upper input identified in Corollary E2 as 'the precise Theorem A upper bound in [LP]') the consumed input is (1.1), not (1.2) as the finding and the author's Does-not-claim said; "
        "the lower half is thereby stated under a hypothesis the source does not need for it, which the docstring and Does-not-claim record; "
        "(4) pinM/pinS are kept as auxiliaries but attached to no claim (claim_ids []) and their Does-not-claim says so; (5) PairLawInterface's Does-not-claim names the phantom parameters G, I; "
        "(6) IsLifetimeDensityVersion's Does-not-claim now says measurability of the pair count is REQUIRED (through Integrable, hence AEStronglyMeasurable), not disclaimed; "
        "(7) the three unhedged docstring remarks (HessianHasInertia eigenvalue count, torusDist vanishing, K_L(0) = 1) are hedged as 'intended ...; not shown here'; "
        "(8) the sha256 wording of the author JSON changed to 'sha256-matched' and the result written under the orchestrator's item id; "
        "(9) IsLifetimeDensityVersion opens with the inlined term-level `haveI := G.mΩ` (core-Lean 4.34.1 probe of the exact shape passes, see checks_run), so a bundled sample space with a square-bracket MeasurableSpace field no longer depends on unification-only resolution. "
        "Not applied (with reasons): (3) the existential-cutoff form of B/E11 is kept — the source states Theorem B for one cutoff r_* and the author's Does-not-claim already records that it is not tied to the r_* of (1.1); the proposed ∀-small-cutoff form says MORE than the anchor. "
        "The shared block of Lifetime.lean was regenerated byte-identically from Field.lean (Field lines 41-378 == Lifetime lines 69-406, 25499 B). "
        f"New identities: Field.lean {fid['bytes']} B sha256 {fid['sha256']}; Lifetime.lean {lid['bytes']} B sha256 {lid['sha256']}. "
        "Declaration counts unchanged: Field 26 (10 def_prop, 2 structure, 14 auxiliary), Lifetime 38 (21 def_prop, 3 structure, 14 auxiliary); 64/64 anchors verbatim and occurring exactly once in the sha256-matched sources; "
        "packet check.py scan_module passes on both modules; the author's static_check.py passes (0 failures); no Lean elaboration of the modules was possible (pinned Mathlib oleans still building; no lake started in scratchpad/mathf). "
        "Scientific effect NONE; formal progress of every declaration: specified (contingent on the packet's not-yet-run --execute elaboration); no claim promoted; alignment PENDING_INDEPENDENT_REVIEW."
    ),
    "files": [
        str(FIELD), str(LIFE),
        str(SP / "lvs/out/spec_field_lifetime.json"),
        str(SP / "lvs/out/spec-field-lifetime.json"),
        str(SP / "lvs/fixer_fl/apply_fixes.py"), str(SP / "lvs/fixer_fl/apply_fixes2.py"),
        str(SP / "lvs/fixer_fl/packet_scan.py"), str(SP / "lvs/fixer_fl/gen_decls.py"), str(SP / "lvs/fixer_fl/compose_out.py"),
        str(SP / "lvs/fixer_fl/probe/HaveIProp.lean"),
        str(SP / "lvs/fixer_fl/Field.before.lean"), str(SP / "lvs/fixer_fl/Lifetime.before.lean"),
    ],
    "file_identities": {
        "Specifications/Field.lean": {**fid, "declarations": 26, "before": {"bytes": 26885, "sha256": "9faa80870035a3e90f74844bbef96c7d303eead8741bea92a12d77197cf318d5"}},
        "Specifications/Lifetime.lean": {**lid, "declarations": 38, "before": {"bytes": 47342, "sha256": "b92378c93ba3b938a0e8921690f2f7e17a413cdca4f7f20c26abba529920c57f"}},
    },
    "verifier_findings_disposition": [
        {"finding": "blocking: PlanarTwoSidedRateE4 unconditional", "disposition": "APPLIED as proposed (hypothesis `ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi →`; docstring opening reworded; Anchor unchanged)."},
        {"finding": "minor: PlanarDensityGapE11 upper-half dependency", "disposition": "APPLIED with a correction: the consumed parent upper input is (1.1) (source lines 43-46, 165-168), not (1.2); E11 gains the parameter `PL : PairLawInterface 2 L G I` and the hypothesis `ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi →`; the lower half's extra hypothesis is disclosed."},
        {"finding": "minor: existential cutoff rStar in B/E11", "disposition": "NOT APPLIED: the source states Theorem B for its one cutoff r_*; the ∀-small-cutoff form says more than the anchor; the author's Does-not-claim ('that the cutoff r_* is the one of (1.1)') is kept and repeated in E11."},
        {"finding": "minor: pinM/pinS dead definitions labelled as claim statements", "disposition": "APPLIED: kept as auxiliaries (auxiliaries are not registered by check.py), claim_ids set to [] in this result, Does-not-claim says they are consumed by no statement."},
        {"finding": "minor: PairLawInterface phantom G, I", "disposition": "APPLIED (kept, documented): Does-not-claim adds 'that p depends on G or I in any way (no field mentions them ...)'."},
        {"finding": "minor: IsLifetimeDensityVersion Does-not-claim wording on count measurability", "disposition": "APPLIED as proposed."},
        {"finding": "minor: three unhedged docstring remarks", "disposition": "APPLIED as proposed (hedged; 'not shown here')."},
        {"finding": "minor: sha256 wording and item-id spelling in the author JSON", "disposition": "APPLIED: 'sha256-matched'; written as lvs/out/spec_field_lifetime.json and, identically, under the old spelling."},
        {"finding": "minor: bundled MeasurableSpace field not a global instance", "disposition": "APPLIED (option 1): `haveI := G.mΩ` opens the body of IsLifetimeDensityVersion; own core-Lean probe of the exact multi-line Prop shape passes (inst-implicit, FunLike/outParam and {_ : MS α} uses), control without haveI fails on the inst-implicit use as the verifier reported. The comment on the field and the docstring say so without the words 'instance'/'theorem' (the author's whole-text scanner flags them)."},
    ],
    "declarations": decls,
    "mathlib_identifiers": [{"name": n, "file_line": (M + fl if not fl.startswith(("Mathlib.lean", "lean-toolchain")) else fl)} for n, fl in mathlib],
    "checks_run": [
        "Read DESIGN.md completely; read the verifier report (lvs/out/spec_field_lifetime.verify.json) and the author report; re-read the source regions the findings cite (elder-lower-density-gap lines 1-60 and 140-185, lifetime-parent lines 11-52, lifetime-remainder lines 8-28) from the sha256-matched local copies before deciding each fix.",
        "apply_fixes.py / apply_fixes2.py: every edit is an exact-string replacement that asserts its occurrence count (1, or 2 for the pinM/pinS line) and the shared block of Lifetime.lean is regenerated from Field.lean after asserting that the two pre-existing shared blocks were identical; diffs reviewed (Field: 8 hunks, all in docstrings/comments plus the one `haveI` line; Lifetime: the same 8 plus header, PairLawInterface Does-not-claim, E4, E11).",
        "Author's static_check.py (whole-text lexical scan: refused tokens theorem/lemma/example/instance/axiom/sorry/native_decide/unsafe/opaque/partial/macro/notation/set_option; the four refused vocabulary words of DESIGN section 0 as case-insensitive substrings; docstring balance; one namespace; imports exactly Mathlib; four labelled lines; anchors verbatim in the sha256-matched source): FAILURES 0 on both modules (an intermediate wording with 'instance'/'theorem' in prose and a docstring line starting with 'structure' at column 0 was flagged by it and reworded before the final state).",
        "Verifier's scan.py (comment-stripped refusal regex incl. syntax/elab/attribute/deriving/mutual/axioms/decide; bracket balance (), [], {}, ⟨⟩; imports/open/namespace): forbidden hits in code NONE for both modules; brackets balanced (Field 141/141, Lifetime 257/257 parentheses).",
        "Packet check.py imported and run as the checker will: load_sources (23 sources, bytes/sha256/git-blob re-matched) then scan_module on Field and Lifetime: OK; Field 26 declarations (10 def_prop, 2 structure, 14 auxiliary), Lifetime 38 (21 def_prop, 3 structure, 14 auxiliary); every registered-kind declaration carries exactly one Source:/Anchor:/Interface:/Does not claim: line, Source lines parse and name a registered source, anchors verbatim; additionally each anchor occurs exactly once in its source (12/12 Field, 24/24 Lifetime registered-kind; verifier's anchors.py: 26/26 and 38/38 including auxiliaries, 0 non-verbatim).",
        "Shared-block identity: Field.lean lines 41-378 are byte-identical to Lifetime.lean lines 69-406 (25499 B).",
        "Mathlib identifiers re-grepped in <SCRATCH>/mathf/.lake/packages/mathlib/Mathlib with file:line (list in mathlib_identifiers; no new Mathlib identifier is introduced by the fixes; AEStronglyMeasurable, named in prose, confirmed at AEStronglyMeasurable.lean:66; `haveI` confirmed as a core builtin_term_parser at Lean/Parser/Term.lean:578-579 of the pinned toolchain source).",
        "Core-Lean 4.34.1 probe lvs/fixer_fl/probe/HaveIProp.lean (lean --version = Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release); no Mathlib, no lake, nothing in scratchpad/mathf touched): a `def … : Prop :=` whose body opens with `haveI := G.mΩ` on its own line followed by a multi-line conjunction elaborates with the bundled square-bracket field used through an inst-implicit argument, a FunLike/outParam class and a `{_ : MS α}` argument; `#print` shows the inlined term and `#print axioms` reports no axioms; the control without haveI fails on the inst-implicit use with 'failed to synthesize MS G.Ω' (as the verifier reported) while the FunLike and implicit uses pass.",
        "Honesty re-read of the two changed Props against the source: E4 = source Corollary E2 'If the precise Theorem A upper bound in [LP] is consumed, then in d=2 c r^3 <= 1-p_r(b,k,R) <= C r^3' -> hypothesis A at d = 2 for the same windows, then the two-sided bound with c > 0, C, r_* > 0 outside the ∀ over r, b, k, frames; E11 = 'The uniform positive lower and finite upper bounds for A_r, and (E2)/(E4), imply ... (E11). The lower follows directly from (E2); the upper consumes the already identified parent upper theorem' -> hypothesis A (the upper input of (E4)), then the two-sided density gap with existential cutoff, c > 0, C, ell_* > 0 and versions pinned by the integral identity. Non-vacuity check: `p ≡ 1` satisfies ParentUniformSelectionA for any C, and under it E4's conclusion fails, so E4 is not a tautology; the hypotheses of A and of E4/E11 are jointly satisfiable (2 ≤ 2, any L > 0, bLo ≤/< bHi, 0 < kLo ≤/< kHi).",
        "Read-only `git -C repos/math status --porcelain --untracked-files=all`: only untracked packet files and the workflow (the whole packet is untracked on branch claude/lean-specifications-20261010 at 6c020d6a); nothing committed or pushed; no lake process started; no path outside Specifications/Field.lean, Specifications/Lifetime.lean, lvs/out/spec_field_lifetime.json, lvs/out/spec-field-lifetime.json and the fixer scratch directory lvs/fixer_fl was written.",
        "NOT run: Lean elaboration of either module (the pinned Mathlib oleans for the analysis/probability modules do not exist yet and no lake may be started in scratchpad/mathf); the packet's --execute mode; the REGISTER.json rows (infra item: REGISTER.json currently has claims: []).",
    ],
    "uncertainties": [
        "Neither module has been elaborated by Lean against the pinned Mathlib; every statement below is a residual elaboration risk, not a confirmed error. The core-Lean probe covers the `haveI` shape only, not the Mathlib notations.",
        "`haveI := G.mΩ` followed by a newline and the conjunction relies on the core term parser `withPosition (\"haveI \" letConfig letDecl) optSemicolon termParser` (Term.lean:578-579) accepting the newline-separated body at the same column; the probe exercises exactly this layout.",
        "E11's hypothesis is (1.1) on my reading of elder lines 43-46/165-168 ('the already identified parent upper theorem' = 'the precise Theorem A upper bound in [LP]' of Corollary E2); the verifier's finding named (1.2). The conclusion's upper half coincides with the third line of (1.2), so a reviewer may prefer (1.2) or both as the hypothesis; the Does-not-claim records the dependency as (1.1) and the author's earlier text said (1.2).",
        "Stating E11's lower half under the (1.1) hypothesis is a hypothesis the source does not need for that half (it derives it from (E2) alone); disclosed in the docstring and Does-not-claim; an independent reviewer may record a narrower-scope verdict on E11 for that reason (verdicts are theirs to write, not ours).",
        "ContDiff ℝ n g with n : ℕ relies on the coercion ℕ → WithTop ℕ∞ (precedent contDiff_infty, Defs.lean:1115).",
        "The variance argument `⟨covQuadForm d L x a, quad_nonneg x a⟩ : ℝ≥0` relies on the anonymous constructor unfolding `NNReal := { r : ℝ // 0 ≤ r }` (Basic/NNReal/Defs.lean:58); fallback `Real.toNNReal` (:155).",
        "PairLawInterface still carries the parameters `G` and `I` that no field mentions (documented in Interface and Does-not-claim); a Mathlib-style linter may warn about unused structure parameters.",
        "Coordinate access `x i` on EuclideanSpace in FundDomain relies on the PiLp CoeFun seeing through the abbrevs `E` and `EuclideanSpace`; `Module.finrank ℝ V` for a Submodule relies on CoeSort; `iteratedFDeriv ℝ 2 g x ![v, w]` relies on the ContinuousMultilinearMap coercion.",
        "HessianHasInertia encodes the eigenvalue count by Sylvester inertia; equivalence with the eigenvalue count needs symmetry of the Hessian (C^2), now stated as 'intended ...; not shown here'.",
        "pairCount uses Set.ncard (0 on infinite sets); IsLifetimeDensityVersion demands almost-sure finiteness and integrability of the counts (hence their a.e. strong measurability) and integrability of ν on each Borel A ⊆ (0,∞); if the counts were not a.e. measurable no version could exist (statement false, not vacuous).",
        "The elder relation is pinned (iff) only on the Morse distinct-value locus for smooth L-periodic g, as the source; off that locus ElderPartner is unconstrained.",
        "The claim pointers (`claim_ids`) are free-text labels mixing the inventory object ids (`math.uniform-matrix-cap-lifetime / D1-THM-A` etc.) with the author's descriptive labels; the register rows (infra item) decide the actual claim_id values.",
    ],
    "gaps": [
        "The explicit Gaussian/angular representation (15.2) (and (13.6), (11.3)) of c_{d,L} / c_{B,K} is not transcribed; the coefficient enters only through its leading-term role.",
        "The regression law Q_{r,b,k,R}, the typed weight W_r, the full normalizer Z_r and Q^W are not transcribed; p_r(b,k,R) is a primitive interface field.",
        "The marked Kac-Rice representation (Sections 9-10, radial ledger (10.2)) is not transcribed; densities are pinned only by the integral identity of IsLifetimeDensityVersion.",
        "Not stated: (12.1)-(12.3), (R19)-(R20), (14.1), (5.5), the parent's Sections 2-8 interfaces, the marked-cylinder-cap support, the reading-rule amendments, (E3), (E7)-(E10).",
        "No Lean elaboration, no kernel receipt, no `#check`/`#print axioms` of the modules; formal progress is `specified` only after the packet's --execute mode elaborates them.",
        "REGISTER.json rows for these 64 declarations (and the transcription of the (1.1) dependency of E4/E11 into not_established/next_step), the root imports, README/SCOPE text and the workflow belong to other items; nothing outside the two modules, the out JSONs and the fixer scratch directory was written.",
    ],
}
for name in ("spec_field_lifetime.json", "spec-field-lifetime.json"):
    (SP / "lvs/out" / name).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("written", [str(SP / "lvs/out" / n) for n in ("spec_field_lifetime.json", "spec-field-lifetime.json")])
# vocabulary self-check on the result text
import re
blob = json.dumps(res, ensure_ascii=False)
for w in ("sha256-verified", "kernel-checked", "ALIGNED AT", "MISALIGNED"):
    print(f"  {w!r} in result:", w in blob)
print("  words 'verified'/'proved'/'closed'/'certified' used about a theorem? bare-word hits:",
      [w for w in ("verified", "proved", "closed", "certified") if re.search(r"(?<![A-Za-z0-9_-])" + w + r"(?![A-Za-z0-9_])", blob)])
