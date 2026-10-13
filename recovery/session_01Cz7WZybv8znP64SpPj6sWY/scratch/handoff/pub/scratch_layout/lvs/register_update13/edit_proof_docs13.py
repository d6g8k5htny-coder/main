#!/usr/bin/env python3
"""Round-13 hand edit of the proof packet's README.md and SCOPE.md for round 6 of the proof packet: the module
SpecificationProofs/Side24Images.lean (uncommitted, new, imported last by SpecificationProofs.lean) with three inventory
theorems

    UniversalLaw.Proofs.Side24Images.imageBound               : UniversalLaw.Spec.Side24.ImageBound
    UniversalLaw.Proofs.Side24Images.entrywiseCovarianceBound : UniversalLaw.Spec.Side24.EntrywiseCovarianceBound
    UniversalLaw.Proofs.Side24Images.covarianceOrdering       : UniversalLaw.Spec.Side24.CovarianceOrdering

on top of Math- main 4afcca6a (the merge of PR #439). Edits: inventory 75 -> 78 theorems over 74 -> 77 Props (the other
67 -> 64 Props without a theorem); one receipt-paragraph sentence: the three theorems have NO hosted receipt yet (added
after PR head c6ce0256, whose run 38195810463, actual tested merge 1320c5b2... with base be014794..., receipt theorem_count
75, does not name them; their local checks at the pin, like any local run, are evidence about no commit; the next hosted
run is their first receipt); the "why no theorem here" row 'Side24 lattice sums | ImageBound' and the CovarianceOrdering /
EntrywiseCovarianceBound part of the row 'Side24 covariance and density' leave those rows (NegDefCone stays, as
vocabulary) and are superseded by one 'formerly listed here (round 6)' row each, in the shape of round 5's rows (theorem
name, 'no hosted receipt yet' instead of a receipt, the BLOCKED_MISSING_MATHLIB (weeks) reason superseded by the theorem,
the route from the module docstring, what was found not load-bearing, and a 'Not established:' list from the theorem
docstrings); the triage-count sentence (7 -> 4 BLOCKED_MISSING_MATHLIB without a theorem; 28 + 24 + 4 + 8 = 64, asserted
against the table); the Layout listing; the SCOPE Side24 family row (modules, what is established, what is not, the
receipt status). Derived from register_update10/edit_proof_docs10.py (round 5's pending edits) and
register_update11/edit_proof_docs11.py (round 5's binding, committed at 4afcca6a): exact string replacements over the
COMMITTED text (read-only `git show 4afcca6a:...`), each with an asserted occurrence count of one, refusing unexpected text
and a worktree that differs from the commit.

Scientific effect: NONE. Documentation only: no label, disposition or status changes; claim-level labels stay none /
specified; 'proved' is not added (it is the label of a committed proof package's theorem entry) and 'kernel-checked' is
not added (a receipt word); alignment PENDING_INDEPENDENT_REVIEW; same-account author, organizational-independence credit
0. No triage verdict is altered: the round-4 verdict BLOCKED_MISSING_MATHLIB on the three Props stays the historical
verdict (transcribed in the register); only the README's previous informal reasons are superseded by the theorems, as
round 5 did for its PROVABLE_WITH_WORK Props, and reason (ii) for CovarianceOrdering ('Mathlib has
isSymmSndFDerivAt_iff_iteratedFDeriv only') is corrected as an informal reason, not a statement defect:
ContDiffAt.iteratedFDeriv_comp_perm is present at Mathlib d13f23b7 (Mathlib/Analysis/Analytic/IteratedFDeriv.lean, for a
function C^omega at the point; Kinf d is analytic). Never touches check.py, tests, the register or Lean. A theorem is
evidence about its registered Prop for the formal objects only, never about an informal claim; a local kernel check on an
unverified worktree is evidence about no commit. Every mention of a hosted run names the PR head and the actual tested
merge (C422-E1), asserted per paragraph; no run other than 38195810463 is cited for the round-6 theorems.

usage: python3 -B -S edit_proof_docs13.py      (R13_PDOCS_OUT=<dir> writes the two docs there instead of the worktree)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

CLONE = Path(os.environ['LVS_SCRATCH']) / 'repos/math_r5'  # scrubbed for publication: as-run value was the session scratch root
PACKET_REL = 'specifications/lean_statement_proofs_20261011'
PACKET = CLONE / PACKET_REL
OUT = Path(os.environ['R13_PDOCS_OUT']) if os.environ.get('R13_PDOCS_OUT') else PACKET
HEAD = '4afcca6a79a8a0acef251ae73988a7914c8080e7'
# FACTS (coordinator-verified; the last hosted receipt the register binds, as bound by edit_proof_docs11.py at 4afcca6a)
RUN10, C10, M10F, BASE4F = '38195810463', 'c6ce0256', '1320c5b217e491d301ea140e5cc25b07ff5a21b7', 'be01479435f98f34b802258ee9ec3c3578f37169'
SHA10 = 'd8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff'
M10, BASE4 = M10F[:8], BASE4F[:8]
RUN9, C9, M9, BASE3 = '38187538744', 'be014794', '79d3964c', '12e072d6'
RUN8, C8, M8 = '38180705952', '48bfcbae', '9833d6d9'
assert BASE4F.startswith(C9)
MODULE = 'SpecificationProofs/Side24Images.lean'
THEOREMS = (('imageBound', 'ImageBound'), ('entrywiseCovarianceBound', 'EntrywiseCovarianceBound'),
            ('covarianceOrdering', 'CovarianceOrdering'))
ROUND6_PROPS = tuple(p for _, p in THEOREMS)
N_REGISTERED, N_COVERED, N_THEOREMS = 141, 77, 78
TRIAGE = {'predicates': 28, 'BLOCKED_INTERFACE': 24, 'BLOCKED_MISSING_MATHLIB': 4, 'P15 predicates not re-examined': 8}


def git(*args):
    return subprocess.run(['git', '-C', str(CLONE), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout


def pristine(name: str) -> str:
    head = git('rev-parse', 'HEAD').strip()
    if head != HEAD:
        sys.exit(f'REFUSED: clone HEAD is {head}, expected {HEAD}')
    text = git('show', f'{HEAD}:{PACKET_REL}/{name}')
    if (PACKET / name).read_text(encoding='utf-8') != text:
        sys.exit(f'REFUSED: worktree {name} differs from {HEAD[:8]} (already edited?)')
    return text


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f'REFUSED: {label}: expected exactly one occurrence of the old text, found {n}')
    return text.replace(old, new)


# ---- the module this edit describes: present, uncommitted, imported last by the root, with the three theorems of
# exactly the registered types, and the docstring facts the new text paraphrases
_mod = (PACKET / MODULE).read_text(encoding='utf-8')
assert git('ls-files', f'{PACKET_REL}/{MODULE}').strip() == '', 'Side24Images.lean is expected uncommitted'
_root = [line for line in (PACKET / 'SpecificationProofs.lean').read_text(encoding='utf-8').splitlines() if line.strip()]
assert _root[-1] == 'import SpecificationProofs.Side24Images' and len(_root) == 14, _root[-2:]
for _thm, _prop in THEOREMS:
    assert _mod.count(f'\ntheorem {_thm} : UniversalLaw.Spec.Side24.{_prop} :=') == 1, _thm
for _needle in ('theorem ib_iteratedFDeriv_tsum_of_isOpen', 'hasFDerivAt_tsum_of_isPreconnected',
                'helper of this module, not an inventory theorem and not a Mathlib result',
                '`ThetaSummable d` is not used', 'any\nbound for `q > 6`', 'ContDiffAt.iteratedFDeriv_comp_perm',
                'theorem co_C24_hermitian', 'theorem co_kinf_analytic (d : ℕ) : ContDiff ℝ ω (Kinf d)',
                '1458 (76 · 24^6 + 15) e^(-288) < imageConstE', '|svecScale a · svecScale b · (-1)^|b|| ≤ 2',
                '`|x ⬝ (C_24 - C_ref) x| ≤ 26 imageConstE (x ⬝ x)`, and\n`78 imageConstE ≤ epsilon`',
                'Side24Covariance.referenceCovarianceLowerBound', 'theorem co_card_le {d : ℕ} (hd : d ≤ 3) : Fintype.card (JointIndex d) ≤ 13',
                "main's `UniversalLaw.Side24.covariance_relative_bound` is neither imported, cited\nnor relied on",
                'spectral-norm bound `20E`', '`60E < epsilon` (not reproduced', 'the passage to marginal and\nconditional covariances (not specified)',
                'the round-4 triage verdict on `ImageBound` (BLOCKED_MISSING_MATHLIB), which is\ntranscribed in the register and not altered here',
                'any enclosure of the coefficient `c_{d,24}`'):
    assert _needle in _mod, _needle
assert 'ContDiffBump' not in _mod   # the cutoff workaround was not used
# isSymmSndFDerivAt is named only in the module's maintainer remark, quoting the README's superseded reason (ii)
assert _mod.count('isSymmSndFDerivAt') == 1 and '("Mathlib\nhas `isSymmSndFDerivAt_iff_iteratedFDeriv` only")' not in _mod
assert '("Mathlib has `isSymmSndFDerivAt_iff_iteratedFDeriv` only")' in _mod.replace('\n', ' ')

PENDING_NOTE = 'no hosted receipt yet, see above'

# ---- README
OLD_RECEIPT_TAIL = (
    'does not name them; their local `lake env lean -DwarningAsError=true` checks at the pin, with `#print axioms`\n'
    'within `propext`, `Classical.choice`, `Quot.sound`, like any local run, are evidence about no commit.\n\n| group |')
NEW_RECEIPT_TAIL = (
    'does not name them; their local `lake env lean -DwarningAsError=true` checks at the pin, with `#print axioms`\n'
    'within `propext`, `Classical.choice`, `Quot.sound`, like any local run, are evidence about no commit. The 3 theorems\n'
    'of the round-6 module `Side24Images` (`imageBound`, `entrywiseCovarianceBound`, `covarianceOrdering`) have **no\n'
    f'hosted receipt yet**: they were added after PR head `{C10}`, whose hosted run {RUN10} (actual tested merge\n'
    f'`{M10}…` with base `{BASE4}…`; receipt theorem_count 75) does not name them; their local checks at the pin\n'
    '(`lake build`, `lake env lean -DwarningAsError=true`, `#check` of each against its registered Prop, `#print axioms`\n'
    'within `propext`, `Classical.choice`, `Quot.sound`), on an unverified worktree, like any local run, are evidence\n'
    'about no commit; the next hosted `lean-specification-proofs.yml` run is their first receipt.\n\n| group |')

OLD_IMAGE_ROW = (
    '| Side24 lattice sums | `ImageBound` | BLOCKED_MISSING_MATHLIB (weeks); not a statement defect (`K24` is `C^∞`, so Lean\'s total '
    '`iteratedFDeriv` is the genuine derivative, and `e^{-288} < 10^{-125}` leaves the strict inequality room). The exact missing piece '
    'is the exchange of `iteratedFDeriv` with the `tsum` over `Fin d → ℤ` near `0`: Mathlib `d13f23b7` has only the global '
    '`iteratedFDeriv_tsum`, `iteratedFDeriv_tsum_apply` and `contDiff_tsum_of_eventually` (`Analysis/Calculus/SmoothSeries.lean`), which '
    'need a summable bound `‖iteratedFDeriv 𝕜 k (f i) x‖ ≤ v k i` for every `x`, which the translates `x ↦ Kinf d (x + 24 n)` do not '
    'have (their supremum over `x` does not depend on `n`), and its only local (open-set) result, `hasFDerivAt_tsum_of_isPreconnected`, '
    'is first order. A cutoff workaround (`ContDiffBump` times each term, transfer back by `Filter.EventuallyEq.iteratedFDeriv`, the '
    'normalisation `1 / thetaSum24 d 0` by `iteratedFDeriv_const_smul_apply`) is a multi-week development on top of the derivative and '
    'lattice theorems now in `Side24Analysis`. |\n')
LATTICE_FORMER_ROW_END = (
    'a different route from the source\'s shell count `729 ∑ j⁹ e^{-288 j²}`, which stays unspecified. |\n')
IMAGE_ROW = (
    '| Side24 lattice sums, formerly listed here (round 6) | — | `ImageBound` has a theorem in `Side24Images` since round 6 '
    f'(`Side24Images.imageBound`; {PENDING_NOTE}). The BLOCKED_MISSING_MATHLIB (weeks) reason this table gave for it is superseded by '
    'the theorem; the round-4 triage verdict itself is not altered (it is transcribed in the register as the historical verdict). '
    'The piece the reason named as missing, the exchange of `iteratedFDeriv` with the `tsum` over `Fin d → ℤ` near `0`, is supplied '
    'inside the module, not by Mathlib: the helper `Aux.ib_iteratedFDeriv_tsum_of_isOpen` (not an inventory theorem and not a Mathlib '
    'result) is the local version of `iteratedFDeriv_tsum` on an open preconnected set, with summable bounds required only on that '
    'set, by the same induction as the global lemma, from the first-order local `hasFDerivAt_tsum_of_isPreconnected`; the cutoff '
    'workaround the previous reason described (`ContDiffBump`, `Filter.EventuallyEq.iteratedFDeriv`) was not needed. Route (the module '
    'docstring): on the unit ball the translates `z ↦ Kinf d (z + 24 n)` have summable derivative bounds (the round-4 operator-norm '
    'pairing bound, `pairingSum q r ≤ 76 (1 + r)⁶`, `(1 + r)⁶ ≤ 720 e^{1 + r}`, `r - r²/4 ≤ 1`, the shift bound '
    '`exp(-‖z + v‖²/4) ≤ exp(‖z‖²/4) exp(-‖v‖²/8)` and the product form of the Gaussian over `Fin d → ℤ`), so, for `q ≤ 6`, '
    '`D^q K24(0) = θ(0)⁻¹ ∑_n D^q Kinf(24 n)`; splitting off `n = 0`, the difference with `D^q Kinf(0)` is '
    '`θ(0)⁻¹ (∑_{n ≠ 0} D^q Kinf(24 n) - (θ(0) - 1) D^q Kinf(0))`, bounded through the packet\'s theorems '
    '`Side24Analysis.derivativeBoundAtLeastOne` (`76`), `derivativeBoundAtZero` (`15`), `latticeImageSumBound` (`1458 e^{-288}`) and '
    '`Side24Elementary.exponentialBounds` by `1458 (76·24⁶ + 15) e^{-288} < imageConstE`; the Prop\'s hypothesis `ThetaSummable d` is '
    'not used. Not established: display (2) of the source as the source\'s object (the alignment of `E d`, `Kinf`, `K24`, '
    '`thetaSum24`, `IsUnitTuple` and Mathlib\'s `iteratedFDeriv` with the source\'s objects is PENDING_INDEPENDENT_REVIEW); any bound '
    'for `q > 6` or `d ∉ {2, 3}`; differentiability of `K24` as a statement (it is used inside the proof, on the unit ball, through '
    'the helper); any enclosure of the coefficient `c_{d,24}`. |\n')

OLD_COV_PROPS = ('| Side24 covariance and density | `NegDefCone` (predicate), `CovarianceOrdering`, `EntrywiseCovarianceBound` | ')
NEW_COV_PROPS = '| Side24 covariance and density | `NegDefCone` (predicate) | '
OLD_COV_REASON = (
    ' `CovarianceOrdering` and `EntrywiseCovarianceBound`: BLOCKED_MISSING_MATHLIB (weeks), both true as stated. Two of the four pieces '
    'the triage listed as missing are theorems since round 4 (`latticeImageSumBound` for the `ℤ^d` tail '
    '`∑_{n ≠ 0} \\|n\\|⁶ e^{-288 \\|n\\|²} ≤ 1458 e^{-288}`; `derivativePairingBound` and its corollaries for the constants `76` and `15` '
    'that the entrywise constant `2 imageConstE` needs). The exact pieces still missing from Mathlib `d13f23b7` are (i) local term-wise '
    'iterated differentiation of `thetaSum24` at `0` with bounds summable only near `0` (the same gap as `ImageBound`: '
    '`iteratedFDeriv_tsum` needs bounds uniform over all `x`, the only local result `hasFDerivAt_tsum_of_isPreconnected` is first order, '
    'and the hypothesis `ThetaSummable d` gives pointwise summability only) and, for the `IsHermitian` half of `PosSemidef` in '
    '`CovarianceOrdering`, (ii) symmetry of `iteratedFDeriv` under permutation of the two direction blocks beyond order two (Mathlib has '
    '`isSymmSndFDerivAt_iff_iteratedFDeriv` only). The display-(3) ordering would even survive the crude bound '
    '`norm_iteratedFDeriv_comp_le` (`60 E\' < epsilon` with a factor of about 8 to spare). |\n')
COV_ROUND5_ROW_END = (
    'and no minor, no eigenvalue `8 ± sqrt 58` and no comparison of an eigenvalue with `1/3` is computed here. |\n')
COV_ROW = (
    '| Side24 covariance and density, formerly listed here (round 6) | — | `EntrywiseCovarianceBound` and `CovarianceOrdering` have '
    'theorems in `Side24Images` since round 6 (`Side24Images.entrywiseCovarianceBound`, `Side24Images.covarianceOrdering`; '
    f'{PENDING_NOTE}). The BLOCKED_MISSING_MATHLIB (weeks) reasons this table gave for them are superseded by the theorems; the round-4 '
    'triage verdicts themselves are not altered (they are transcribed in the register as the historical verdicts). Of the two pieces '
    'the reasons named as still missing from Mathlib `d13f23b7`, (i) the local term-wise iterated differentiation of `thetaSum24` at '
    '`0` is supplied inside the module by the same helper as for `ImageBound` (`Aux.ib_iteratedFDeriv_tsum_of_isOpen`, see the '
    'lattice-sums row of round 6), and (ii), symmetry of `iteratedFDeriv` under permutation of the two direction blocks beyond order '
    'two, is needed only for the analytic `Kinf d` (the exponential of a polynomial), and that case is present: Mathlib `d13f23b7` has '
    '`ContDiffAt.iteratedFDeriv_comp_perm` (`Mathlib/Analysis/Analytic/IteratedFDeriv.lean`) for a function that is `C^ω` at the point '
    '(the general version for finitely smooth functions beyond order two is indeed absent at the pin); the previous reason (ii) '
    '("Mathlib has `isSymmSndFDerivAt_iff_iteratedFDeriv` only") was therefore not load-bearing and is corrected here, as a correction '
    'of an informal reason, not a statement defect. Route (the module docstring): `EntrywiseCovarianceBound` from `imageBound`, the '
    'two entries sharing the factor `svecScale a · svecScale b · (-1)^\\|b\\|` of absolute value at most `2`, their contractions running '
    'along at most six unit frame vectors; for `CovarianceOrdering`, `C24` is symmetric (the helper `Aux.co_C24_hermitian`, stated for '
    'every `d`, `u` and `R`, not an inventory theorem) because each contraction of `K24` at `0` of total order at most `6` (each joint '
    'coordinate carries at most three directions) is `θ(0)⁻¹ ∑_n` of contractions of the '
    'analytic `Kinf d` at `24 n`, invariant under the block swap, and odd total orders vanish by `n ↦ -n` with the evenness of '
    '`Kinf d`; `Cref` is symmetric with '
    '`Cref - I/3` positive semidefinite by the round-5 theorem `Side24Covariance.referenceCovarianceLowerBound`; Cauchy-Schwarz over '
    'at most `13` joint coordinates gives `\\|x ⬝ (C24 - Cref) x\\| ≤ 26 imageConstE (x ⬝ x)`, and `78 imageConstE ≤ epsilon` by '
    '`norm_num`; the Prop\'s hypothesis `ThetaSummable d` is passed on to `imageBound`, which does not use it. Not established: the '
    'Section 3 entrywise sentence and display (3) of the source as the source\'s objects (the alignment of `jetDirs`, `svecScale`, '
    '`jointCov`, `C24`, `Cref` and Mathlib\'s `Matrix.PosSemidef`, `IsHermitian` included, with the source\'s objects is '
    'PENDING_INDEPENDENT_REVIEW); that `C24` or `Cref` is the covariance of any random vector (no field is constructed); the passage '
    'to marginal and conditional covariances (not specified); the spectral-norm bound `20E` (not specified) and the source\'s route '
    'through it and `60E < epsilon` (not reproduced); symmetry of `Cref d u R` for a general `u` or frame; main\'s '
    '`UniversalLaw.Side24.covariance_relative_bound` is neither imported, cited nor relied on; any enclosure of the coefficient '
    '`c_{d,24}`. |\n')

RECEIPT5_NOTE = f'first receipted by run {RUN10}: PR head `{C10}`, actual tested merge `{M10}…` with base `{BASE4}…`, see above'
README_EDITS = [
    ('inventory counts',
     'Inventory at this commit: 75 theorems, 74 of the 141 registered `def … : Prop` declarations (`TriangleNeedsThree`\n'
     'is proved twice). The other 67 fall into the groups below.',
     f'Inventory at this commit: {N_THEOREMS} theorems, {N_COVERED} of the {N_REGISTERED} registered `def … : Prop` declarations '
     '(`TriangleNeedsThree`\n'
     f'is proved twice). The other {N_REGISTERED - N_COVERED} fall into the groups below.'),
    ('round-6 receipt status', OLD_RECEIPT_TAIL, NEW_RECEIPT_TAIL),
    ('lattice row: ImageBound leaves the why-no-theorem rows', OLD_IMAGE_ROW, ''),
    ('lattice round-6 row', LATTICE_FORMER_ROW_END, LATTICE_FORMER_ROW_END + IMAGE_ROW),
    ('covariance row: Props column', OLD_COV_PROPS, NEW_COV_PROPS),
    ('covariance row: superseded reasons', OLD_COV_REASON, ' |\n'),
    ('covariance round-6 row', COV_ROUND5_ROW_END, COV_ROUND5_ROW_END + COV_ROW),
    ('triage totals',
     'Across the whole table the triage\'s counts for the 67 Props without a theorem are 28 predicates (these 26 plus `IsUnitTuple`, '
     '`NegDefCone`), 24 BLOCKED_INTERFACE (these 22 plus `CoefficientEnclosure`, `ReferenceComparison`), 7 BLOCKED_MISSING_MATHLIB (these 4 '
     'plus `ImageBound`, `CovarianceOrdering`, `EntrywiseCovarianceBound`) and the 8 P15 predicates it did not re-examine; its 2 '
     'PROVABLE_WITH_WORK Props (`ReferenceCovarianceLowerBound`, `ConeMomentD2Identity`) have theorems since round 5 '
     f'({RECEIPT5_NOTE}).',
     'Across the whole table the triage\'s counts for the 64 Props without a theorem are 28 predicates (these 26 plus `IsUnitTuple`, '
     '`NegDefCone`), 24 BLOCKED_INTERFACE (these 22 plus `CoefficientEnclosure`, `ReferenceComparison`), 4 BLOCKED_MISSING_MATHLIB (these 4 '
     'Lifetime Props) and the 8 P15 predicates it did not re-examine; its 2 PROVABLE_WITH_WORK Props (`ReferenceCovarianceLowerBound`, '
     f'`ConeMomentD2Identity`) have theorems since round 5 ({RECEIPT5_NOTE}), and 3 of its 7 BLOCKED_MISSING_MATHLIB Props (`ImageBound`, '
     f'`CovarianceOrdering`, `EntrywiseCovarianceBound`) have theorems since round 6 ({PENDING_NOTE}).'),
    ('layout listing',
     '                                       P15Interpolation; Side24 into Side24Elementary, Side24Analysis,\n'
     '                                       Side24Interfaces, Side24Covariance, Side24ConeMoment; RN into\n'
     '                                       RNProbability, RNCluster (each list in import order; the root\n'
     '                                       imports the eight modules of rounds 1-3, then Side24Analysis,\n'
     '                                       Side24Interfaces, RNCluster (round 4) and Side24Covariance,\n'
     '                                       Side24ConeMoment (round 5))\n',
     '                                       P15Interpolation; Side24 into Side24Elementary, Side24Analysis,\n'
     '                                       Side24Interfaces, Side24Covariance, Side24ConeMoment,\n'
     '                                       Side24Images; RN into RNProbability, RNCluster (each list in\n'
     '                                       import order; the root imports the eight modules of rounds 1-3,\n'
     '                                       then Side24Analysis, Side24Interfaces, RNCluster (round 4),\n'
     '                                       Side24Covariance, Side24ConeMoment (round 5) and Side24Images\n'
     '                                       (round 6))\n'),
]

# ---- SCOPE
SCOPE_EDITS = [
    ('Side24 family: modules',
     '| Side24 (`Specifications.Side24`, modules Side24Elementary, Side24Analysis, Side24Interfaces, Side24Covariance, Side24ConeMoment) |',
     '| Side24 (`Specifications.Side24`, modules Side24Elementary, Side24Analysis, Side24Interfaces, Side24Covariance, Side24ConeMoment, '
     'Side24Images) |'),
    ('Side24 family: image bound',
     'the lattice image sum (summable and `≤ 1458 e^{-288}` for `d ≤ 3`);',
     'the lattice image sum (summable and `≤ 1458 e^{-288}` for `d ≤ 3`); display (2), the image bound '
     '`\\|D^q K24(0) u - D^q Kinf(0) u\\| < imageConstE` for `d = 2` or `3`, every `q ≤ 6` and every unit tuple `u`, through '
     '`D^q K24(0) = θ(0)⁻¹ ∑_n D^q Kinf(24 n)`, by the term-wise derivative on the unit ball (the local exchange of `iteratedFDeriv` and `tsum` '
     'is the module\'s helper `Aux.ib_iteratedFDeriv_tsum_of_isOpen`, not an inventory theorem and not a Mathlib result; the hypothesis '
     '`ThetaSummable d` is not used);'),
    ('Side24 family: entrywise bound and ordering',
     'stated for every `0 < d`); and, for a GIVEN `I : ConeMomentInterface`,',
     'stated for every `0 < d`); the entrywise bound `\\|C24 a b - Cref a b\\| ≤ 2 imageConstE` and display (3), '
     '`C24 - (1 - epsilon) • Cref` and `(1 + epsilon) • Cref - C24` positive semidefinite in Mathlib\'s sense, `IsHermitian` included, '
     'both at `u = R 0` in an orthonormal frame `R` for `d = 2` or `3`, as the Props state them (the symmetry of `C24` through the '
     'term-wise derivative, from Mathlib\'s `ContDiffAt.iteratedFDeriv_comp_perm` for the analytic `Kinf d`, with the odd total orders '
     'vanishing by `n ↦ -n` and the evenness of `Kinf d`; `\\|x ⬝ (C24 - Cref) x\\| ≤ 26 imageConstE (x ⬝ x)` and '
     '`78 imageConstE ≤ epsilon`, not the source\'s `20E` and `60E < epsilon`); and, for a GIVEN `I : ConeMomentInterface`,'),
    ('Side24 family: formal objects and receipt status',
     '(`E d`, `Kinf`, `latticeVec`, `derivContraction`, `gaussDensity`, `jointCov`, `Cref`, `coneIntegrand`, the interface structures); '
     'the reference covariance lower bound and the `m = 2` cone moment (round 5) are first receipted by run '
     f'{RUN10} (PR head `{C10}`, actual tested merge `{M10}…` with base `{BASE4}…`; receipt theorem_count 75) |',
     '(`E d`, `Kinf`, `K24`, `thetaSum24`, `latticeVec`, `derivContraction`, `gaussDensity`, `jointCov`, `Cref`, `C24`, `coneIntegrand`, '
     'the interface structures); the reference covariance lower bound and the `m = 2` cone moment (round 5) are first receipted by run '
     f'{RUN10} (PR head `{C10}`, actual tested merge `{M10}…` with base `{BASE4}…`; receipt theorem_count 75); the image bound, the '
     'entrywise bound and display (3) (round 6, module Side24Images) have no hosted receipt yet: they were added after PR head '
     f'`{C10}`, that run does not name them, and the next hosted `lean-specification-proofs.yml` run is their first receipt |'),
    ('Side24 family: not established, the three Props leave',
     '`ImageBound`, `CovarianceOrdering`, `EntrywiseCovarianceBound` (no local term-wise iterated differentiation of the lattice sum at '
     'Mathlib `d13f23b7`); that `ConeMomentInterface`',
     'any image bound for `q > 6` or `d ∉ {2, 3}`; differentiability of `K24` as a statement (used only inside the proofs, on the '
     'unit ball); the spectral-norm bound `20E` (not specified) and the source\'s route through it and '
     '`60E < epsilon` (not reproduced; main\'s `UniversalLaw.Side24.covariance_relative_bound` is neither imported, cited nor relied '
     'on); the passage to marginal and conditional covariances (not specified); that `ConeMomentInterface`'),
    ('Side24 family: not established, C24 not a covariance',
     'that any `gradCov`/`pinCov` matrix or `Cref` is the covariance of a random object;',
     'that any `gradCov`/`pinCov` matrix, `Cref` or `C24` is the covariance of a random object;'),
]


def unescaped_pipes(line: str) -> int:
    return len(re.findall(r'(?<!\\)\|', line))


def table_rows(text: str, header: str) -> list:
    lines = text.split('\n')
    i = lines.index(header)
    rows = []
    for line in lines[i + 2:]:
        if not line.startswith('|'):
            break
        rows.append(line)
    return rows


README_HEADER = '| group | Props (names without `UniversalLaw.Spec.<Module>.`) | why no theorem here |'
SCOPE_HEADER = '| family | the theorems establish | not established |'


def check_readme(text: str, orig: str) -> None:
    rows = table_rows(text, README_HEADER)
    assert len(rows) == len(table_rows(orig, README_HEADER)) + 1, len(rows)   # -1 lattice row, +2 round-6 rows
    for row in rows:                                                 # the table's three cells survive (escaped pipes only)
        assert unescaped_pipes(row) == 4, row[:90]
    groups = [row.split(' | ')[0][2:] for row in rows]
    assert groups.count('Side24 lattice sums, formerly listed here (round 6)') == 1
    assert groups.count('Side24 covariance and density, formerly listed here (round 6)') == 1
    assert 'Side24 lattice sums' not in groups and 'Side24 covariance and density' in groups
    assert groups.index('Side24 lattice sums, formerly listed here (round 6)') == groups.index('Side24 lattice sums, formerly listed here') + 1
    assert (groups.index('Side24 covariance and density, formerly listed here (round 6)')
            == groups.index('Side24 covariance and density, formerly listed here (round 5)') + 1)
    # the "why no theorem here" rows list none of the three round-6 Props; count their Props against the inventory
    open_count = 0
    for row in rows:
        group, props = row.split(' | ')[0][2:], row.split(' | ')[1]
        if 'formerly listed here' in group:
            assert props == '—', group
            continue
        for p in ROUND6_PROPS:
            assert p not in props, (group, p)
        if group.startswith('RN beyond'):
            m = re.search(r'\| (\d+) Props in three classes, per the round-4 triage\. \(a\) (\d+) BLOCKED_INTERFACE.*\(b\) (\d+) Lifetime '
                          r'Props BLOCKED_MISSING_MATHLIB.*\(c\) (\d+) predicates with free arguments', row)
            assert m, 'RN/Lifetime/Field row shape'
            total, a, b, c = map(int, m.groups())
            assert total == a + b + c == 52 and (a, b, c) == (22, 4, 26), m.groups()
            open_count += total
        else:
            names = [n for n in re.findall(r'`([A-Za-z]+)`', props) if not n.endswith('Interface')]
            open_count += len(names)
    assert open_count == N_REGISTERED - N_COVERED == sum(TRIAGE.values()) == 64, (open_count, TRIAGE)
    assert 'counts for the 64 Props without a theorem are 28 predicates' in text
    assert '24 BLOCKED_INTERFACE (these 22 plus' in text and '4 BLOCKED_MISSING_MATHLIB (these 4 Lifetime Props)' in text
    assert 'the 8 P15 predicates it did not re-examine' in text
    # the superseded informal reasons are gone; the three Props are named only in round-6 / triage-history text
    for gone in ('multi-week development', 'Mathlib has `isSymmSndFDerivAt_iff_iteratedFDeriv` only)', 'both true as stated',
                 'The exact pieces still missing from Mathlib', 'The exact missing piece is the exchange'):
        assert gone not in text, gone
    assert text.count('formerly listed here (round 6)') == 2 and text.count('formerly listed here (round 5)') == 2
    assert text.count(PENDING_NOTE) == 3                       # two round-6 rows and the triage totals
    assert text.count('**no\nhosted receipt yet**') == 1
    assert 'Side24Images.imageBound' in text and 'Side24Images.entrywiseCovarianceBound' in text
    assert 'Side24Images.covarianceOrdering' in text
    assert 'is their first receipt' in text and text.count('first receipt') == orig.count('first receipt') + 1
    assert text.count(RECEIPT5_NOTE) == orig.count(RECEIPT5_NOTE) == 3


def check_scope(text: str, orig: str) -> None:
    rows = table_rows(text, SCOPE_HEADER)
    assert len(rows) == len(table_rows(orig, SCOPE_HEADER)) == 4
    for row in rows:
        assert unescaped_pipes(row) == 4, row[:90]
    side24 = [row for row in rows if row.startswith('| Side24 ')]
    assert len(side24) == 1
    established, not_established = side24[0].split(' | ')[1], side24[0].split(' | ')[2]
    assert 'Side24Images' in side24[0].split(' | ')[0]
    for needle in ('display (2), the image bound', 'the entrywise bound', 'display (3)', 'no hosted receipt yet'):
        assert needle in established, needle
    for p in ROUND6_PROPS:                                          # no longer listed as not established
        assert p not in not_established, p
    assert 'nothing here encloses the coefficient' in not_established    # no enclosure of c_{d,24}: kept verbatim
    assert 'no local term-wise iterated differentiation' not in text


def main() -> None:
    for name, edits in (('README.md', README_EDITS), ('SCOPE.md', SCOPE_EDITS)):
        text = pristine(name)
        orig = text
        assert 'Side24Images' not in orig and 'round 6' not in orig and 'no hosted receipt' not in orig, name
        for label, old, new in edits:
            text = replace_once(text, old, new, f'{name}: {label}')
        # labels: this edit adds neither the label 'proved' nor the receipt word 'kernel-checked'
        assert text.count('kernel-checked') == orig.count('kernel-checked'), name
        assert len(re.findall(r'\bproved\b', text)) == len(re.findall(r'\bproved\b', orig)), name
        assert text.count('Kernel-checked') == orig.count('Kernel-checked'), name
        for para in text.split('\n\n'):      # C422-E1: a hosted run is named with its PR head and its tested merge
            if RUN10 in para:
                assert C10 in para and M10 in para and BASE4 in para, (name, para[:120])
            if RUN9 in para:
                assert C9 in para and M9 in para and BASE3 in para, (name, para[:120])
            if RUN8 in para:
                assert C8 in para and M8 in para, (name, para[:120])
        assert set(re.findall(r'\b38\d{9}\b', text)) == set(re.findall(r'\b38\d{9}\b', orig)), name   # no new run is cited
        assert text.count(RUN10) == orig.count(RUN10) + (1 if name == 'README.md' else 0), name     # README: once more, as "does not name them"
        for phrase in ('Scientific effect', 'NONE', 'PENDING_INDEPENDENT_REVIEW'):
            assert phrase in text, (name, phrase)
        assert not re.findall(r'<<[^<>\n]*>>', text), name                                         # check.py's PLACEHOLDER
        if name == 'README.md':
            check_readme(text, orig)
        else:
            check_scope(text, orig)
        (OUT / name).write_text(text, encoding='utf-8')
        print(f'{name}: {len(edits)} replacement(s) applied -> {OUT / name}')


if __name__ == '__main__':
    main()
