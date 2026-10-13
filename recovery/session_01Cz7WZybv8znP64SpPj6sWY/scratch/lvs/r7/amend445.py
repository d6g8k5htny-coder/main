#!/usr/bin/env python3
"""Amendments 445-A..D of the classification review (Math-#445 6138311227) to the frontier section of README.md and SCOPE.md
of the statement packet, on the frontier branch worktree at 9fb0a4a0. Docs only; identical edits in both documents, each
asserted to occur exactly once (the 36 vocabulary row reasons exactly 36 times) per document. Scientific effect NONE."""
import hashlib, subprocess, sys
from pathlib import Path
S = Path(sys.argv[1]); WT = S / 'lvs/r7/wt445'; PK = WT / 'specifications/lean_statements_20261010'
BASE = '9fb0a4a06900da5cc3f9126f30e117a747df1587'
assert subprocess.run(['git', '-C', str(WT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() == BASE
assert subprocess.run(['git', '-C', str(WT), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip() == ''
def once(t, old, new):
    assert t.count(old) == 1, (old[:70], t.count(old)); return t.replace(old, new)
def exactly(t, old, new, n):
    assert t.count(old) == n, (old[:70], t.count(old), n); return t.replace(old, new)

VOC_OLD = 'Predicate with explicit non-instance arguments, consumed only as a hypothesis; its universal closure is false or empty, so it names an object, not a result.'
VOC_NEW = ('Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted '
           'universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded).')
EDITS = [
 # 445-A: vocabulary class
 ("- **vocabulary** (36): a `def … : Prop` with explicit non-instance arguments that the modules consume only as a\n"
  "  hypothesis (`IsCriticalPoint d g x →`, `IsUnitTuple d q u →`, …). Its universal closure is false or empty, so no\n"
  "  theorem of its type is a target; it is counted among the 141 because the register inventories every `Prop`-valued\n"
  "  definition. Of the 105 target Props, 77 are proved and 28 remain.\n",
  "- **vocabulary** (36): a reusable `def … : Prop` with explicit non-instance arguments, used by the modules inside\n"
  "  definitions, as a hypothesis (`IsCriticalPoint d g x →`, `IsUnitTuple d q u →`, …) and in conclusions\n"
  "  (`IsUnrestrictedLeadingCoefficient` is the conclusion of C at Lifetime.lean:544, `IsLifetimeDensityVersion` an\n"
  "  existential conclusion of B at Lifetime.lean:502-503, `IsRealizedFamily` is concluded for `benchmark816` at\n"
  "  P15.lean:521, `NegDefCone` supplies the indicator set at Side24.lean:683). Its unrestricted universal closure is\n"
  "  false or empty, so a theorem of exactly its type is not a target; instantiated or conditional theorems about it are\n"
  "  not excluded by this grouping. It is counted among the 141 because the register inventories every `Prop`-valued\n"
  "  definition. Under this grouping, a planning convention of this section and not a proof that other meaningful\n"
  "  targets are exhausted, 105 Props are targets, 77 are proved and 28 remain.\n"),
 # 445-C: interface-blocked class
 ("  violates the conclusion is named below. No cleverer proof exists; the route is a numbered successor interface\n"
  "  carrying the missing field (a Kac-Rice intensity, the law of `p`, the conditional Hessian law), followed by\n"
  "  successor Props. Nothing here says the informal source is wrong: the source's hypotheses are exactly what the\n"
  "  interface does not transcribe.\n",
  "  violates the conclusion is named below; these are non-kernel mathematical judgements of a single reader. They refute\n"
  "  the unrestricted universal closure for the given outer objects and nothing more: the Side24 scaling inhabitant shows\n"
  "  that universal validity of either Prop would force the interface empty, without constructing an inhabitant; the\n"
  "  Lifetime `p ≡ 0` / `p ≡ 1` refutations hold for given admissible `G`, `I` and do not construct those; the RN\n"
  "  refutations are read on a common path space with a parameter-indexed family, finite counted sets and the stated\n"
  "  region and window hypotheses, and the preserved RN probe supplies conditional Dirac, count and null-set reductions,\n"
  "  not a formal countermodel receipt. So a theorem of exactly the Prop's type is not available without new fields; the\n"
  "  route is a numbered successor interface carrying the missing field (a Kac-Rice intensity, the law of `p`, the\n"
  "  conditional Hessian law), followed by successor Props. Nothing here says the informal source is wrong: the source's\n"
  "  hypotheses are exactly what the interface does not transcribe.\n"),
 # 445-C: Mathlib-blocked class
 ("- **Mathlib-blocked** (4): the Prop's truth value is fixed by the interface, and the statement follows the pinned\n"
  "  source, but the proof needs Gaussian-field or density theory that Mathlib `d13f23b7` does not contain\n"
  "  (effort estimated in weeks each).\n",
  "- **Mathlib-blocked** (4): the interface fixes the Prop's truth value and no interface counterexample was identified;\n"
  "  the statement follows the pinned source, and a proof is expected to need substantial Gaussian-field and density\n"
  "  formalisation beyond the primitives Mathlib `d13f23b7` has (an effort estimate of weeks each, not an impossibility\n"
  "  theorem). Provability is conditional on the corresponding informal claims, as the round-4 triage qualified it; the\n"
  "  machinery named in a row is the proposed proof route, not a conjunct of the target's conclusion.\n"),
 # 445-B: the Side24 conditioning variable
 ("Missing field: the conditional law of the Hessian block given the gradient under `K24`, which pins `coneMoment`. |",
  "Missing field: the conditional law of the transverse Hessian block conditioned on the pin `V_u = H_f u = 0` under `K24`, which pins `coneMoment`. |"),
 # 445-C: B row
 ("Theorem B (1.2) of the pinned source in full: density versions (a Kac-Rice representation), a.s. finiteness and integrability of the pair counts, "
  "the `c l^(-1/3)` asymptotics with `c > 0`, and the `O(l^(2/3))` gap. The `gaussian` field (Lifetime.lean:175-177) fixes the law, so the truth value "
  "does not depend on the inhabitant; none of the Gaussian-field machinery exists in Mathlib `d13f23b7`. |",
  "Theorem B (1.2) as the Prop states it: versions characterised by the integral identity `IsLifetimeDensityVersion` (Lifetime.lean:398; no Kac-Rice "
  "representation and no coefficient formula is transcribed), the `c l^(-1/3)` asymptotics with `c > 0`, and the `O(l^(2/3))` gap. The `gaussian` field "
  "(Lifetime.lean:175-177) fixes the law, so the truth value does not depend on the inhabitant. Expected proof route (dependencies, not conclusion "
  "conjuncts): a Kac-Rice representation, a.s. finiteness and integrability of the pair counts, substantial Gaussian-field formalisation beyond the "
  "primitives of Mathlib `d13f23b7`; conditional on the informal claim. |"),
 # 445-C: C row
 ("Theorem C (1.3): density versions, a.s. Morse genericity (so the elder relation is pinned a.s.), the off-diagonal domination and the small-`l` "
  "asymptotics with a positive constant; absent from the pinned Mathlib. |",
  "Theorem C (1.3) as the Prop states it: some `c` with `IsUnrestrictedLeadingCoefficient d L G I c` (Lifetime.lean:522-544), i.e. versions with the "
  "small-`l` asymptotics and a positive constant. Expected proof route (dependencies, not conclusion conjuncts): a.s. Morse genericity (so the elder "
  "relation is pinned a.s.) and the off-diagonal domination, Gaussian-field formalisation beyond the pinned Mathlib; conditional on the informal claim. |"),
 # 445-C: E11 row
 ("The hypothesis is satisfied by `p = 1` (probe Lifetime-B_1), so a theorem would be exactly the unconditional two-sided planar density gap for every "
  "`(G, I)` at `d = 2`: the (E2) planar lower argument plus the compact-window `O(l^(2/3))` upper bound, both Gaussian-field analysis absent from the "
  "pinned Mathlib. |",
  "The `p`-hypothesis is inert: the free `p = 1` satisfies it (probe Lifetime-B_1), so a theorem would be the unconditional two-sided planar density gap "
  "for every `(G, I)` at `d = 2`. Expected proof route: the (E2) planar lower argument plus the compact-window `O(l^(2/3))` upper bound, Gaussian-field "
  "analysis beyond the pinned Mathlib; conditional on the informal claims. |"),
 # 445-D (and C): R1 row
 ("Theorem R (R1): the hypothesis gives only `nu ~ c l^(-1/3)`, i.e. an `o(l^(-1/3))` error, strictly weaker than the `O(1)` remainder claimed; density "
  "versions are unique a.e., so re-choosing them cannot supply the bound; the remainder analysis of the source has no Mathlib counterpart. |",
  "Theorem R (R1), in its conditional form: for every `c` with the leading asymptotics (the hypothesis), versions exist with an `O(1)` remainder (the "
  "definition chooses the versions existentially, Lifetime.lean:562-568); vacuous if no such `c` exists. Leading asymptotics alone does not imply an "
  "essentially `O(1)` remainder; re-choosing versions can remove null-set exceptions but cannot repair failure of the bound on sets of positive measure. "
  "The content is the source's remainder analysis, which has no Mathlib counterpart. |"),
 # 445-C: F4 row qualification
 ("satisfies the first-moment hypothesis `G_2` (`E N = r^3`) while `E N(N-1) >= r^2 - r^3` exceeds `C r^3 log(1/r)` for small `r`. |",
  "satisfies the first-moment hypothesis `G_2` (`E N = r^3`) while `E N(N-1) >= r^2 - r^3` exceeds `C r^3 log(1/r)` for small `r`; here `n_r` is the exact "
  "finite total `torusWindowCount` (with a zero-count branch), uniformly over the antecedent's marks and frames. |"),
]
for name in ('README.md', 'SCOPE.md'):
    p = PK / name; t = p.read_text(encoding='utf-8'); before = t
    for old, new in EDITS: t = once(t, old, new)
    t = exactly(t, VOC_OLD, VOC_NEW, 36)
    assert 'consumed only as a hypothes' not in t and 'No cleverer proof exists' not in t and 'given the gradient' not in t
    assert 'unique a.e., so re-choosing' not in t and 'none of the Gaussian-field machinery exists' not in t
    assert t.count('36 vocabulary + 24 interface-blocked + 4 Mathlib-blocked = 64; 64 + 77 = 141.') == 1
    p.write_text(t, encoding='utf-8')
    print(name, len(before.encode()), '->', len(t.encode()), hashlib.sha256(t.encode()).hexdigest())
print('done')
