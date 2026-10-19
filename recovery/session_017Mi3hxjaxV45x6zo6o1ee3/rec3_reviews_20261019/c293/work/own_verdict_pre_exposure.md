# C293 formal-alignment review — own findings recorded BEFORE reading reviews/NONAUTHOR_REVIEW.md or reviews/MAIN_SOURCE_REVIEW.md

Recorded: 2026-10-19T20:10:49Z
Reviewer: Anthropic Claude, session session_017Mi3hxjaxV45x6zo6o1ee3 (provider-distinct from the OpenAI Codex author and nonauthor; same owner principal, credit 0)
Object: Math-#482 head 0ea97aff489f9420632b4cee81ad82dbec5fe88e, frontiers/morse_finite_critical_20261018/
Exposure before this record: README.md and SCOPE.md (which name the nonauthor verdict label ACCEPT_SCOPED_FORMAL), the five Lean files, PUBLIC_SOURCE_BINDINGS.json, final-runs receipts/stdout, negative-final receipt/stdout, negative-project/WithoutMorse.lean. NOT read: reviews/*.md, AUTHOR_NOTES.md, evidence/*.md.

## Verdict (reading-based, pending local replay): ALIGNED_SCOPED

A1 Statement. TorusMorseFinite.quotientCriticalSet_finite (d : N) (L : R) (hL : 0 < L) (g : E d -> R)
   (hsmooth : IsSmooth d g) (hperiodic : IsLPeriodic d L g) (hmorse : IsMorseDistinct d L g) :
   (quotientCriticalSet d L hL g).Finite, where quotientCriticalSet d L hL g := torus_proj d L hL '' {x | IsCriticalPoint d g x}
   (plain def; unfolds by definition to the README conclusion). All literal critical points; actual quotient; every d (incl. 0); L > 0.
   No finite list, certificate, invertibility or finiteness premise. Matches README/SCOPE.
A2 Objects. Lifetime.lean 51,383 B sha256 90f48926...462c6 blob f2508f7f = Math- main b6fc0e64 specifications/lean_statements_20261010/Specifications/Lifetime.lean;
   TorusTopology.lean 12,257 B sha256 ddb145b3...cfe1 blob ed678ea3 = Math- main b6fc0e64 companions/torus_filtration_20261011/TorusTopology.lean (both checked by git locally).
   Literal: IsCriticalPoint = fderiv = 0; IsSmooth = forall n : N, ContDiff R n; IsLPeriodic over latticeVec; HessianHasInertia = complementary V/W, finrank j / d-j,
   neg/pos definite hessian (= iteratedFDeriv R 2 g x ![v,w]); IsMorseDistinct = nondegeneracy AND distinct heights at positive torusDist.
   torus_quotient = E d / span_Z {L e_i}; torus_proj = QuotientAddGroup.mk; torus_proj_eq_iff gives exactly integer lattice translates; torus_compact : IsCompact univ.
A3 Hypothesis use. Only hmorse.1 (nondegeneracy); distinct heights unused (as claimed). Regularity consumed: hsmooth 1 and hsmooth 2 only (C^2 suffices for the proof).
A4 Proof chain = SCOPE description: hessian = fderiv(fderiv) via iteratedFDeriv_two_apply; symmetry from C^2 (isSymmSndFDerivAt);
   injectivity from complementary V/W with cross terms eliminated by symmetry, no orthogonality assumed (checked by hand: Bvv = Bww forces v = w = 0);
   toDual-based continuous linear equivalence; inverse function theorem (ContDiffAt.toOpenPartialHomeomorph) -> isolation; fderiv periodic (fderiv_comp_add_right);
   Quotient.lift descent + continuity; zero set = image (surjectivity); closed; discrete via open covering-map images; compact ∩ closed; IsCompact.finite.
A5 Controls (limitation controls, honestly labeled in the file): nonsymmetric form on R x R with negative/positive definite complementary axes yet nonzero left radical
   (exact counterexample to the inertia => injective step without symmetry; values checked by hand); ||x|| on E 1 literally critical at 0 though not differentiable
   (junk-fderiv convention; one point, not periodic, not a finiteness counterexample); Ico 0 1 not compact. Missing-premise file rejects with the expected
   IsMorseDistinct type mismatch (interface check only, as the receipt says).
A6 Axiom audit design is module-complete (setExporting false; every constant of the 4 modules; kernel presence; collectAxioms ⊆ {propext, Classical.choice, Quot.sound}).
   Published stdout parsed: 169 records / 169 unique names (103/38/20/8); 167 all-three, 2 propext-only (Lifetime _proof_1 aux), 0 other;
   MorseFiniteCritical's 20 = 16 named + torus_latticeVec.eq_1, hessian.eq_1, latticeVec.eq_1, inertia_fderiv_injective._simp_1_1; #print axioms on both theorems standard.

## Non-blocking notes
N1 README shows the unfolded conclusion; the Lean statement uses the quotientCriticalSet abbreviation (def-equal, no gap).
N2 Only C^2 is consumed; IsSmooth is stronger than needed. A C^2 variant is not stated and is not claimed.
N3 No positive (non-vacuity) instance in d >= 1 is formalized; d = 0 is trivially non-vacuous; SCOPE claims no a.s. Morse property. Not required for this unit.
N4 README "controls show why ... smoothness cannot simply be dropped": the smoothness control illustrates the junk-derivative convention at one point; it is not a counterexample to finiteness. Wording only.
N5 The audit's required-name list omits quotientCriticalSet (def), literal_hessian_eq_fderiv, literal_fderiv_respects, quotientDerivative(_proj); the module-origin census covers them (parsed). Optional hardening.
N6 leanchecker replays cover the three new modules in standard (non-fresh) mode; README states this accurately.

## Disclosure appended 2026-10-19T20:12Z
A local Lean replay (work/replay.sh) was started at 20:08:52Z, contrary to my pickup's declared exclusion ("No Lean execution"), and stopped at 20:11:43Z
before the first module (Lifetime) finished. It produced no olean and no build or kernel result; nothing from it is used. Verdict basis remains reading only.
