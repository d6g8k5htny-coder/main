"""Build lvs/out/spec_rn.json (fixer result for item spec_rn) from the final module bytes, the
author's claim-id map and the fixer's check results. Scientific effect: NONE. The script refuses
to write if any of the four honesty words of DESIGN section 0 appears in the output text."""
import hashlib, json, pathlib, re, sys

B = pathlib.Path("<SCRATCH>")
PKT = B / "repos/math/specifications/lean_statements_20261010"
RN = PKT / "Specifications/RN.lean"
W = B / "lvs/fixer_rn"
OUT = B / "lvs/out/spec_rn.json"
raw = RN.read_bytes(); txt = raw.decode("utf-8")
SHA = hashlib.sha256(raw).hexdigest(); NB = len(raw)
MODULE = "Specifications/RN.lean (namespace UniversalLaw.Spec.RN)"
SRC = json.loads((PKT / "sources/SOURCES.json").read_text())
by_sha = {s["sha256"]: s["id"] for s in SRC["sources"]}
author = json.load(open(B / "lvs/out/spec-rn.json"))
claim_map = {d["name"]: d["claim_ids"] for d in author["declarations"]}
claim_map["windowCountIndexSum"] = claim_map["RemoteCollisionSecondFactorialD"]

decl_re = re.compile(r"^(?:noncomputable\s+)?(def|abbrev|structure)\s+(\S+)", re.M)
decls = []
for dm in decl_re.finditer(txt):
    kw, name = dm.group(1), dm.group(2)
    before = txt[:dm.start()].rstrip(); ds = before[before.rfind("/--"):]
    sig = txt[dm.start():txt.find(":=", dm.start())] if kw != "structure" else ""
    kind = "structure" if kw == "structure" else ("def_prop" if sig.rstrip().endswith(": Prop") else kw)
    sha = re.search(r"sha256 ([0-9a-f]{64})", ds).group(1)
    anchor = re.search(r'^Anchor: "(.*)"\s*$', ds, flags=re.M).group(1)
    dnc = re.search(r"^Does not claim: (.*?) -/\s*$", ds, flags=re.S | re.M).group(1).strip()
    decls.append(dict(name=name, kind=kind, module=MODULE, source_id=by_sha[sha], anchor=anchor,
                      does_not_claim=dnc, formal_progress="specified", claim_ids=claim_map[name]))
assert len(decls) == 52, len(decls)
kinds = {}
for d in decls: kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
assert kinds == {"abbrev": 1, "def": 23, "def_prop": 27, "structure": 1}, kinds

ML = [(n, fl) for n, fl in [
 ("EuclideanSpace", "Mathlib/Analysis/InnerProductSpace/PiL2.lean:114"),
 ("EuclideanSpace.single", "Mathlib/Analysis/InnerProductSpace/PiL2.lean:300"),
 ("PiLp.single", "Mathlib/Analysis/Normed/Lp/PiLp.lean:152"),
 ("PiLp CoeFun (coordinate access `p i`)", "Mathlib/Analysis/Normed/Lp/PiLp.lean:92"),
 ("OrthonormalBasis", "Mathlib/Analysis/InnerProductSpace/PiL2.lean:393"),
 ("OrthonormalBasis.instFunLike (frame vectors `R i`)", "Mathlib/Analysis/InnerProductSpace/PiL2.lean:409"),
 ("WithLp.measurableSpace", "Mathlib/Analysis/Normed/Lp/MeasurableSpace.lean:26"),
 ("PiLp.borelSpace", "Mathlib/Analysis/Normed/Lp/MeasurableSpace.lean:52"),
 ("measureSpaceOfInnerProductSpace (volume on EuclideanSpace)", "Mathlib/MeasureTheory/Measure/Haar/OfBasis.lean:309"),
 ("Real.measureSpace", "Mathlib/MeasureTheory/Measure/Haar/OfBasis.lean:319"),
 ("MeasureTheory.Measure", "Mathlib/MeasureTheory/Measure/MeasureSpaceDef.lean:77"),
 ("MeasureTheory.Measure.instFunLike (`μ s`)", "Mathlib/MeasureTheory/Measure/MeasureSpaceDef.lean:90"),
 ("MeasureTheory.MeasureSpace / volume", "Mathlib/MeasureTheory/Measure/MeasureSpaceDef.lean:356"),
 ("MeasureTheory.IsProbabilityMeasure", "Mathlib/MeasureTheory/Measure/Typeclasses/Probability.lean:64"),
 ("MeasurableSet", "Mathlib/MeasureTheory/MeasurableSpace/Defs.lean:64"),
 ("MeasureTheory.Measure.restrict", "Mathlib/MeasureTheory/Measure/Restrict.lean:50"),
 ("MeasureTheory.Measure.withDensity", "Mathlib/MeasureTheory/Measure/WithDensity.lean:39"),
 ("ProbabilityTheory.cond", "Mathlib/Probability/ConditionalProbability.lean:76"),
 ("integral notation `∫ x, f x ∂μ`", "Mathlib/MeasureTheory/Integral/Bochner/Basic.lean:168"),
 ("set integral notation `∫ x in s, f x`", "Mathlib/MeasureTheory/Integral/Bochner/Basic.lean:177"),
 ("MeasureTheory.Integrable", "Mathlib/MeasureTheory/Function/L1Space/Integrable.lean:59"),
 ("MeasureTheory.ae / notation3 `∀ᵐ ω ∂μ` (line corrected from the author's :49)", "Mathlib/MeasureTheory/OuterMeasure/AE.lean:57"),
 ("ENNReal.toReal", "Mathlib/Basic/ENNReal/Basic.lean:228"),
 ("ENNReal.ofReal", "Mathlib/Basic/ENNReal/Basic.lean:231"),
 ("Set.indicator (to_additive of Set.mulIndicator)", "Mathlib/Algebra/Notation/Indicator.lean:52"),
 ("Set.Icc", "Mathlib/Order/Interval/Set/Defs.lean:76"),
 ("Set.Iic", "Mathlib/Order/Interval/Set/Defs.lean:44"),
 ("Set.Ioi (to_dual of Set.Iio)", "Mathlib/Order/Interval/Set/Defs.lean:36"),
 ("Nat.ceil (def :121) / notation `⌈x⌉₊` (:136; the author's entry gave only :121)", "Mathlib/Algebra/Order/Floor/Defs.lean:121,136"),
 ("Real.log", "Mathlib/Analysis/SpecialFunctions/Log/Basic.lean:44"),
 ("Real.exp", "Mathlib/Analysis/Complex/Exponential.lean:80"),
 ("Real.rpow / `noncomputable instance : Pow ℝ ℝ` (line corrected from the author's :35)", "Mathlib/Analysis/SpecialFunctions/Pow/Real.lean:38"),
 ("Filter.Tendsto", "Mathlib/Order/Filter/Defs.lean:422"),
 ("Filter.atTop", "Mathlib/Order/Filter/AtTopBot/Defs.lean:39"),
 ("nhdsWithin", "Mathlib/Topology/Defs/Filter.lean:138"),
 ("ContinuousOn", "Mathlib/Topology/Defs/Filter.lean:191"),
 ("Summable (to_additive of Multipliable, default unconditional filter)", "Mathlib/Topology/Algebra/InfiniteSum/Defs.lean:114"),
 ("tsum notation `∑' n, f n`", "Mathlib/Topology/Algebra/InfiniteSum/Defs.lean:160"),
 ("Set.ncard", "Mathlib/Data/Set/Card.lean:613"),
 ("Nat.descFactorial", "Mathlib/Data/Nat/Factorial/Basic.lean:332"),
 ("Nat.factorial", "Mathlib/Data/Nat/Factorial/Basic.lean:37"),
 ("Submodule", "Mathlib/Algebra/Module/Submodule/Defs.lean:41"),
 ("IsCompl", "Mathlib/Order/Disjoint.lean:320"),
 ("Module.finrank", "Mathlib/LinearAlgebra/Dimension/Finrank.lean:62"),
 ("iteratedFDeriv", "Mathlib/Analysis/Calculus/ContDiff/FTaylorSeries.lean:804"),
 ("fderiv", "Mathlib/Analysis/Calculus/FDeriv/Defs.lean:160"),
 ("ContDiff", "Mathlib/Analysis/Calculus/ContDiff/Defs.lean:1068"),
 ("vector notation `![v, w]`", "Mathlib/Data/Fin/VecNotation.lean:72"),
 ("Finset sum notation `∑ i, f i` and `∑ j ∈ s, f j` (syntax `bigsum`)", "Mathlib/Algebra/BigOperators/Group/Finset/Defs.lean:181"),
 ("Finset.sum (to_additive of `protected def prod`; new in windowCountIndexSum)", "Mathlib/Algebra/BigOperators/Group/Finset/Defs.lean:68"),
 ("Finset.range (new in windowCountIndexSum)", "Mathlib/Data/Finset/Range.lean:53"),
 ("iInf notation `⨅`", "Mathlib/Order/SetNotation.lean:71"),
 ("Set.image notation `''`", "Mathlib/Data/Set/Operations.lean:144"),
 ("abs notation `|x|`", "Mathlib/Algebra/Order/Group/Unbundled/Abs.lean:43"),
 ("Fin.instOfNat (needs `[NeZero d]`; Lean core)", "~/.elan/toolchains/leanprover--lean4---v4.34.1/src/lean/Init/Data/Fin/Basic.lean:245"),
]]
mathlib_identifiers = [{"name": n, "file_line": fl} for n, fl in ML]

files = [str(RN), str(OUT)] + [str(W / f) for f in [
    "apply_fixes.py", "apply_fixes2.py", "check_rn_fixed.py", "gen_json.py",
    "RN_scratch_full.lean", "RN_scratch_full.out",
    "EmptyInterface_baseline.lean", "EmptyInterface_baseline.out", "EmptyInterface_gated.lean", "EmptyInterface_gated.out",
    "VacuityO_baseline.lean", "VacuityO_baseline.out", "VacuityO_amended.lean", "VacuityO_amended.out"]]
for f in files:
    assert pathlib.Path(f).exists() or f == str(OUT), f

summary = (
 f"FIXER pass on the statement-only Mathlib module {RN} (namespace UniversalLaw.Spec.RN; now {NB} bytes, sha256 {SHA}, "
 "52 declarations = 27 `def … : Prop` + 1 interface `structure` + 24 auxiliary definitions, import Mathlib only). "
 "Both blocking findings of the verifier were confirmed by re-running its scratch files unchanged (both elaborate, exit 0) and were repaired: "
 "(1) PinnedLawInterface had no inhabitant for any L > 0 because `Q_pinned` prescribed the two pins for EVERY r > 0 while `periodic` identifies "
 "them at r = L along a coordinate frame; the three r-dependent fields `Q_pinned`, `Z_pos`, `QW_eq` are now gated on the pin regime "
 "`0 < r → r < L →` (the module's reading of the sources' clause \"For small r>0 the pins are\", quoted in the field docstring; for 0 < r < L the "
 "pins are distinct modulo L Z^d since ‖S − M‖ = r while every nonzero lattice vector has norm ≥ L), the structure docstring and its "
 "Interface/Does-not-claim lines explain the gate and still assert no inhabitant; the verifier's emptiness argument no longer elaborates "
 "against the gated structure (negative control, exit 1) while its baseline copy still does (exit 0). "
 "(2) RareClusterPoissonObstructionO held for every (μ, a, A) because (H2) `a r^3 ≤ P(N ≥ 2)` was demanded for ALL r > 0; the hypothesis block is now "
 "`∃ r0 > 0, ∀ 0 < r ≤ r0, (pmf, summable mean, H1, H2)`, matching the source's \"for all sufficiently small r\"; the redundant `a ≤ A` hypothesis "
 "(derived in the source: 'These premises imply 2a<=A') is dropped and said so; the verifier's tautology argument no longer elaborates against the "
 "amended Prop (exit 1) while its baseline still does (exit 0). Minor findings applied: Does-not-claim lines of N1–N5 now name the result/proof and "
 "nonauthor review/acceptance with the HOLD_WITH_DOMAIN transcription note, Theorem O's names the result and its proof; Corollary D now uses the "
 "source's own count `N(E) = sum_j N_j(E)` through the new auxiliary `windowCountIndexSum` (∑ j ∈ Finset.range (d+1), windowCount …; anchor "
 "\"N(E)   = sum_j N_j(E).\" verbatim) and discloses that it is not `windowCountAll`; the four Props built on `torusWindowCount` (F4, Q, Theta, D5-f) "
 "disclose that their count includes degenerate critical points and dominates the sources' all-index N. "
 "A further drift found by the fixer and not in the verifier report: Specifications/Field.lean was amended at 20:29Z by the Field/Lifetime fixer, "
 "after the RN author and verifier ran, so the author's 'nine shared blocks byte-identical' claim had become false for four docstrings; the module now "
 "adopts Field.lean's more cautious wording for HessianHasInertia and torusDist ('intended …; not shown here') and keeps its own `Does not claim:` "
 "lines for pinM/pinS (Field.lean's new line says the pins are consumed by no statement there, which is false inside RN.lean); the header states the "
 "exact scope: nine bodies byte-identical, seven docstrings identical, two Does-not-claim lines module-specific. "
 "Mechanical checks re-run on the final bytes: the packet's own check.py scanner (scan_module) accepts the module (refusal list clean, column-0 "
 "commands, namespace, `import Mathlib`, 27 def_prop + 1 structure, four labelled lines with verbatim anchors); all 52 declarations (auxiliaries "
 "included) carry the four labels with Source lines matching SOURCES.json and anchors that are verbatim byte substrings (52/52); none of the four "
 "honesty words of DESIGN section 0 appears in the module; every Mathlib identifier is confirmed by grep in the pinned tree (55 entries: the author's 53 with three line numbers corrected, plus Finset.sum and Finset.range). NON-CERTIFYING elaboration: a copy differing from the module ONLY in its import lines "
 "(13 narrowed imports in place of `import Mathlib`, whose olean is still absent) — with NO calculus stand-ins, since the FDeriv/ContDiff oleans are "
 "now built — elaborates with plain lean 4.34.1 (commit 5045d005) on the pinned oleans with zero diagnostics, so hessian, IsCriticalPoint and the "
 "`smooth` field are now elaborated too. Scientific effect NONE; formal progress of every declaration remains `specified` and nothing stronger; no "
 "claim is promoted, discharged or reclassified; alignment PENDING_INDEPENDENT_REVIEW. Register note: check.py classifies 27 declarations as "
 "def_prop (the 25 headline Props plus the Prop-valued shared predicates IsCriticalPoint and HessianHasInertia) and 1 as structure; those 28, not the "
 "24 auxiliaries, are the REGISTER.json statement rows this module requires.")

checks_run = [
 "Read DESIGN.md completely; read the verifier report lvs/out/spec_rn.verify.json and the author report lvs/out/spec-rn.json; read Specifications/RN.lean in full; re-read the source clauses cited by the findings in the sha256-matched local copies (remote_window PROOF.md section 1 'For small r>0 the pins are'; dimension-lift PROOF.md line 18 and palm route PROOF.md line 58, same clause; c6-rare-cluster PROOF.md sections 1 and 3, Theorem O 'for all sufficiently small r' and 'These premises imply 2a<=A'; remote_collision PROOF.md lines 22-23 `N(E) = sum_j N_j(E)`); confirmed in sources/landing-claims/LANDING_CLAIMS.json that rn-count-interface and rn-fixed-remote-window carry HOLD_WITH_DOMAIN and rn-fixed-annulus-window REVIEWED_SCOPED.",
 "Reproduced both blocking findings before editing: the verifier's lvs/verify_rn/EmptyInterface.lean and VacuityO.lean, copied unchanged to lvs/fixer_rn/*_baseline.lean, elaborate with plain lean 4.34.1 on the built oleans (exit 0, no error lines), i.e. the unamended interface derives False for 0 < L and the unamended Theorem O Prop holds for every input.",
 "Applied the edits with lvs/fixer_rn/apply_fixes.py (23 exact-match replacements, each asserted to occur exactly once) and apply_fixes2.py (4 replacements); no other file was touched; no git commit; no lake process started.",
 "Mechanical re-check lvs/fixer_rn/check_rn_fixed.py on the final bytes: (a) the packet's own check.py `scan_module` imported and run on RN.lean: no refused token, column-0 commands within the allowed set, imports exactly [Mathlib], namespace UniversalLaw.Spec.RN only, 52 declarations of kinds def_prop 27 / structure 1 / auxiliary 24, each registered-kind declaration with a docstring immediately before it carrying exactly one Source:, Anchor:, Interface:, Does not claim: line, Source matching a registered pinned source and Anchor verbatim; (b) the same four-label and verbatim-anchor check extended to all 52 declarations (52/52), SOURCES.json local copies re-hashed (sha256 and byte counts match); (c) the nine shared declarations: bodies byte-identical to Specifications/Field.lean 9/9, docstrings identical 7/9 with pinM/pinS differing only in the Does-not-claim line; (d) no occurrence of the four honesty words of DESIGN section 0 anywhere in the module (the repository label NON-CERTIFYING is not one of them); (e) the three gated field texts are byte-identical between the module and the negative-control file. FAILURES: [].",
 "Mathlib identifier confirmation by grep in the pinned source tree (mathlib d13f23b723b8a846827a245b89c10fc7d3f11612): the new `Finset.range` (Mathlib/Data/Finset/Range.lean:53 `def range (n : ℕ) : Finset ℕ`), `Finset.sum` via to_additive of `protected def prod` (Mathlib/Algebra/BigOperators/Group/Finset/Defs.lean:68) and the `∑ x ∈ s, f x` form of the `bigsum` syntax (Defs.lean:181); the three corrected lines: `notation3 \"∀ᵐ \"` at Mathlib/MeasureTheory/OuterMeasure/AE.lean:57, `noncomputable instance : Pow ℝ ℝ` at Mathlib/Analysis/SpecialFunctions/Pow/Real.lean:38, `notation \"⌈\" a \"⌉₊\"` at Mathlib/Algebra/Order/Floor/Defs.lean:136 (def Nat.ceil at :121). No other identifier was added.",
 "NON-CERTIFYING elaboration of lvs/fixer_rn/RN_scratch_full.lean with plain `lean` (Lean (version 4.34.1, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release); LEAN_PATH = the eight package lib dirs of scratchpad/mathf/.lake/packages; no lake process started; 7964 of the Mathlib oleans built at run time, Mathlib.olean itself absent): the scratch differs from the module only in its import block (13 narrowed imports: the author's 11 plus Mathlib.Analysis.Calculus.ContDiff.Defs and Mathlib.Data.Finset.Range) and contains NO stand-ins, so hessian (`iteratedFDeriv ℝ 2 g x ![v, w]`), IsCriticalPoint (`fderiv ℝ g x = 0`) and the `smooth` field (`ContDiff ℝ n (F ω)`) are elaborated for the first time. Result: exit 0, zero error or warning lines, 6.4 s. This is evidence about elaboration only; it is not a build of the packet module, not a kernel receipt, and raises no formal-progress label.",
 "Negative controls (executed): lvs/fixer_rn/EmptyInterface_gated.lean = the verifier's emptiness file with only the three fields gated exactly as in the module: lean exit 1 (errors at the `I.Q_pinned L 0 1 _ hL` application: the hypothesis `L < L` is missing and unavailable). lvs/fixer_rn/VacuityO_amended.lean = the verifier's tautology file with the Prop body replaced by the module's amended body (asserted byte-identical to the module's): lean exit 1 (`intro` finds no third binder; the `a ≤ A` hypothesis and the all-r antecedent are gone). Their baselines exit 0 in the same environment, so each control measures the amendment and not the environment.",
 f"Final file identity: {RN} {NB} bytes sha256 {SHA}; the author's 74036-byte sha256 3e10b6fa… version is superseded.",
 "This result JSON was scanned for the four honesty words of DESIGN section 0 before writing; zero hits.",
]

uncertainties = [
 "The packet module itself (`import Mathlib`) is still not elaborated: Mathlib.olean was absent at run time (7964 oleans built). The scratch run covers every declaration body with narrowed imports; a `check.py --execute` run (lake build, #check, #print axioms) is still required before any receipt records `specified`.",
 "Consistency of the gated PinnedLawInterface is a remark in its docstring, not a statement: a Dirac mass at one smooth L-periodic sample path with the prescribed pin values, a nondegenerate maximum at M and an index-(d-1) saddle at S, with W = 1 there, appears to satisfy every field for 0 < r < L, but no inhabitant is constructed or asserted anywhere. The verifier's emptiness proof fails against the gate (negative control); that shows the specific contradiction is gone, not that none remains.",
 "The pin regime 0 < r < L is the module's reading of the sources' unspecified 'small r'; the sources never name a threshold. The Props keep `∃ rStar > 0` unconstrained, so an instantiation may take rStar ≥ L, where the interface fields say nothing about Q_r; this only makes the Props harder to satisfy and does not affect their non-vacuity.",
 "RareClusterPoissonObstructionO is now stated under hypotheses on 0 < r ≤ r0 only; for A < 2a the hypotheses are unsatisfiable (the source's own remark), so the Prop is vacuous for those parameters by the mathematics, not by a transcription defect. Non-triviality for A ≥ 2a is not checked mechanically.",
 "Corollary D uses the index sum `windowCountIndexSum` (the source's definition) while torusWindowCount (F4, Q, Theta, D5-f) still counts every critical point off the pins, degenerate ones included; the Palm/Fourier/rare-cluster wording 'all indices' is ambiguous between the two, the two agree off the degenerate locus, and the difference is disclosed in each affected docstring rather than resolved.",
 "Specifications/Field.lean changed under this item (20:29Z, Field/Lifetime fixer). The nine shared bodies are byte-identical; two docstrings (pinM, pinS) deliberately differ in their Does-not-claim line because Field.lean's new line ('consumed by no statement of this packet') is false inside RN.lean. Lifetime.lean's header also says 'verbatim' about Field.lean's shared objects; whether that claim still holds was not checked here (outside this item's paths).",
 "check.py classifies IsCriticalPoint and HessianHasInertia as def_prop (Prop-valued definitions), so the register must carry 28 statement rows for this module (27 def_prop + 1 structure), not 23 as a headline-only count would suggest; the item JSON keeps kind values def/abbrev for the 24 auxiliaries, which are not register statements.",
 "Design difference from Field.lean/Lifetime.lean retained: PinnedLawInterface takes Ω and [MeasurableSpace Ω] as parameters, so every RN Prop has the extra arguments (Ω : Type) [MeasurableSpace Ω]; the register rows must record this signature.",
 "Reading choices retained from the author and disclosed in docstrings: PalmRouteFactorialMomentsQ lets r_* depend on q; D5eIntermediateShellsI3I4 takes its side conditions from dimension-lift (1.6); `collar` is defined from the dimension-lift source; D5bMicrodisk punctures the disk at the pin; nearPinPairCount counts ordered pairs whose FIRST witness is near a pin; FixedRemoteMeanMeasureA2 bundles the kernel's continuity and two-sided bounds with display (2); FourierCutoffPlanarSecondFactorialF4 is an implication from the G_2 hypothesis; torus Borel sets are represented in the fundamental domain and scaledAt charts are not reduced modulo the lattice.",
 "The verifier's count '22 Props' differs from check.py's 27 def_prop only by classification (the verifier excluded the two shared Prop-valued predicates and counted the headline statements differently); no declaration was lost. Declaration total is 52 (author 51 + windowCountIndexSum).",
]

gaps = [
 "Not stated from rn-count-interface: the Kac-Rice numerator (N6)/(N7), the support discussion of Section 5, the one-p sharpness construction N_r=ceil(r^(-3/p))1_Er, the completion test of Section 6.",
 "Not stated from rn-fixed-remote-window: the observation transform U_r and target v_r, the coupling (5), the weight expansion (10)-(11), the Kac-Rice formula (12), the contact kernel formula (13), the unnormalized numerator bound (14), the multi-witness Section 6 displays (15)-(16).",
 "Not stated from the annulus sources: the six-pin regression (A3)-(A5), normalizer (A8), contact vector (A10)-(A12), cutoff (A13)-(A14), density (A17), Hessian bounds (A18)-(A20), intensities (A21)-(A26); for the bridge: (4)-(20) and the R1-R7 review interfaces as separate statements.",
 "Not stated from d5-dimension-lift: display (1.4), Corollaries F_d and M_d, the five devices D1-D5, and the proofs; from c6-fourier-cutoff: Theorem F (F1)-(F2), Theorem W in general d (F3), Proposition B (F20); from c6-palm-route: Corollary P and Sections 3-7; from c6-rare-cluster: Theorem C (compound Poisson, (4.1)), Section 5 factorial matching, Theorem S, the replica limit (7.1), and the remark `2a ≤ A`; from remote-collision: Theorem C (near-pair count (C)), Corollary E and Corollary F.",
 "No inhabitant of PinnedLawInterface and no non-triviality witness for RareClusterPoissonObstructionO is constructed; both are remarks, not declarations (a construction would be a proof and belongs to the formal lane, not to this statement packet).",
 "REGISTER.json rows (28 statement rows for this module under check.py's classification), README/SCOPE text, check.py coverage and the root Specifications.lean import are owned by other workers; this item delivered the module and this result file only.",
 "No `#check`/`#print axioms` receipts and no lake build exist for this module; the scratch elaboration is NON-CERTIFYING evidence about elaboration only and carries no formal_progress beyond `specified`.",
]

out = {
 "item_id": "spec_rn",
 "role": "FIXER (after the adversarial verifier's AMEND verdict)",
 "scientific_effect": "NONE",
 "scientific_status_authority": False,
 "alignment_status": "PENDING_INDEPENDENT_REVIEW",
 "organizational_independence_credit": 0,
 "summary": summary,
 "files": files,
 "file_identities": {"Specifications/RN.lean": {"bytes": NB, "sha256": SHA, "declarations": 52, "def_prop": 27, "structure": 1, "auxiliary": 24}},
 "verifier_findings_disposition": [
  {"finding": "blocking 1: PinnedLawInterface uninhabited for 0 < L (Q_pinned for all r > 0)", "disposition": "confirmed by re-running the verifier's scratch (exit 0); applied: gate Q_pinned, Z_pos, QW_eq on `0 < r → r < L →`, docstrings/Interface/Does-not-claim updated; negative control: the emptiness argument fails against the gated structure"},
  {"finding": "blocking 2: RareClusterPoissonObstructionO tautology (H2 for all r > 0)", "disposition": "confirmed by re-running the verifier's scratch (exit 0); applied: hypotheses under `∃ r0 > 0, ∀ 0 < r ≤ r0`, `a ≤ A` dropped and disclosed, Does-not-claim names the result and its proof; negative control: the tautology argument fails against the amended Prop"},
  {"finding": "minor: Does-not-claim lines of N1–N5 and Theorem O", "disposition": "applied as proposed"},
  {"finding": "minor: windowCountAll vs the source's N(E) = sum_j N_j(E) in Corollary D; caveat for the torusWindowCount Props", "disposition": "applied: new auxiliary windowCountIndexSum used in Corollary D (Finset.range confirmed by grep); caveat added to F4, Q, Theta, D5-f and to Corollary D"},
  {"finding": "minor: item JSON wording and kind values", "disposition": "applied: no honesty word in this JSON; kinds kept for the auxiliaries with an explicit register note (28 statement rows: 27 def_prop + 1 structure under check.py's own classification, which the verifier's '22 Props' undercounts)"},
  {"finding": "minor: three unelaborated calculus bodies; three line numbers", "disposition": "the calculus oleans are now built, so the scratch elaboration covers all 52 bodies with no stand-ins (exit 0); the three line numbers are corrected in mathlib_identifiers"},
  {"finding": "not in the verifier report: Field.lean drifted at 20:29Z, breaking the 'nine shared blocks byte-identical' claim for four docstrings", "disposition": "applied: HessianHasInertia and torusDist docstrings adopt Field.lean's cautious wording; pinM/pinS keep module-specific Does-not-claim lines (Field.lean's new line is false here); header states the exact scope (bodies 9/9, docstrings 7/9)"},
 ],
 "declarations": decls,
 "mathlib_identifiers": mathlib_identifiers,
 "checks_run": checks_run,
 "uncertainties": uncertainties,
 "gaps": gaps,
}
text = json.dumps(out, indent=1, ensure_ascii=False)
hits = re.findall(r"\b(verified|proved|closed|certified|proven)\b", text, re.I)
if hits:
    print("REFUSING TO WRITE: honesty words in output:", hits); sys.exit(1)
OUT.write_text(text + "\n", encoding="utf-8")
print("wrote", OUT, len(text.encode()), "bytes; declarations", len(decls), kinds, "; mathlib ids", len(mathlib_identifiers))
