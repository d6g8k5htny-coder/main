#!/usr/bin/env python3
"""Round-4 integration: merge the verified lane deliverables into three proof-packet modules.
Scientific effect: NONE. Slices each lane file by its scope markers, drops the reproductions the
lanes flagged for the integrator and the textually identical helpers, repoints call sites, and
rewrites the inventory theorems that now consume sibling inventory theorems."""
from pathlib import Path
import re, sys

W = Path('<SCRATCH>/lvs/round4/work')

def read(name):
    return (W / (name + '.lean')).read_text(encoding='utf-8')

def between(text, start, end):
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[i + len(start):j].strip('\n')

def aux_of(text):
    return between(text, '\nnamespace Aux\n', '\nend Aux\n')

def aux_until(text, stop):
    return between(text, '\nnamespace Aux\n', '\n' + stop)

def inventory_of(text, ns):
    return between(text, '\nend Aux\n', '\nend ' + ns + '\n')

def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        sys.exit(f'{label}: expected exactly one occurrence, found {n}:\n{old[:120]}')
    return text.replace(old, new)

def remove_block(text, first_line, label):
    """Remove the block starting at `first_line` (a docstring opener) up to and including the
    following blank line."""
    i = text.find(first_line)
    if i < 0 or text.find(first_line, i + 1) >= 0:
        sys.exit(f'{label}: block start not unique or missing: {first_line!r}')
    j = text.index('\n\n', i)
    return text[:i] + text[j + 2:]

def section(title, body):
    return f'/-! ## {title} -/\n\n{body}\n'

# ----------------------------------------------------------------------------- Side24Analysis
NS_A = 'UniversalLaw.Proofs.Side24Analysis'
th, dp, d1, d0 = read('ThetaSummable'), read('DerivativePairingBound'), read('DerivativeBoundAtLeastOne'), read('DerivativeBoundAtZero')
rb, rp, ir, li, dc = read('ReferenceOddAndPinBlocks'), read('ReferencePinDensityProduct'), read('IntegrandRatioBounds'), read('LatticeImageSumBound'), read('DensityComparison')

aux_th = aux_of(th)
aux_dp = aux_of(dp)
aux_d1 = aux_until(d1, '/-! ### Reproduction of the pairing-bound route')
aux_d0 = aux_until(d0, '/-! ### Reproduction of the pairing bound: smoothness')
aux_rb = aux_of(rb)
aux_rp = aux_until(rp, '/-! ### Reproduction of the two block identities')
aux_ir = aux_of(ir)
aux_li = aux_of(li)
aux_dc = aux_of(dc)
for name in ('d1_contDiff_Kinf', 'd1_derivativePairingBound', 'd0_contDiff_Kinf', 'd0_derivativePairingBound',
             'rp_contDiff_Kinf', 'rp_blockIdentities', 'rp_order_two'):
    for blob, lbl in ((aux_d1, 'd1'), (aux_d0, 'd0'), (aux_rp, 'rp')):
        if name in blob:
            sys.exit(f'{lbl}: reproduction {name} still present')

# dc: the two helpers identical to ir_epsilon_pos / ir_one_sub_epsilon_pos go, together with
# dc_epsilon_lt_one whose only consumer was dc_one_sub_epsilon_pos; call sites are repointed.
aux_dc = remove_block(aux_dc, '/-- `0 < epsilon`. -/\ntheorem dc_epsilon_pos', 'dc_epsilon_pos')
aux_dc = remove_block(aux_dc, '/-- `epsilon < 1`. -/\ntheorem dc_epsilon_lt_one', 'dc_epsilon_lt_one')
aux_dc = remove_block(aux_dc, '/-- `0 < 1 - epsilon`. -/\ntheorem dc_one_sub_epsilon_pos', 'dc_one_sub_epsilon_pos')
aux_dc = replace_once(aux_dc, '/-! ### (a) Elementary facts about `epsilon = 10^(-108)` -/',
                      '/-! ### (a) Elementary facts about `epsilon = 10^(-108)`\n\n'
                      '`0 < epsilon` and `0 < 1 - epsilon` are `ir_epsilon_pos` and `ir_one_sub_epsilon_pos` above\n'
                      '(the `dc_` copies were removed at integration); only `0 < 1 + epsilon` is added here. -/', 'dc section (a)')
aux_dc = replace_once(aux_dc, '  have := dc_epsilon_pos\n  linarith', '  have := ir_epsilon_pos\n  linarith', 'dc_one_add_epsilon_pos repoint')
for name in ('dc_epsilon_pos', 'dc_epsilon_lt_one', 'dc_one_sub_epsilon_pos'):
    if name in aux_dc:
        sys.exit(f'dc: {name} still referenced')

inv_th = inventory_of(th, NS_A)
inv_dp = inventory_of(dp, NS_A)
inv_rb = inventory_of(rb, NS_A)
inv_ir = inventory_of(ir, NS_A)
inv_li = inventory_of(li, NS_A)
inv_dc = inventory_of(dc, NS_A)
inv_dc = replace_once(inv_dc, 'have ha : (0 : ℝ) < 1 - epsilon := Aux.dc_one_sub_epsilon_pos',
                      'have ha : (0 : ℝ) < 1 - epsilon := Aux.ir_one_sub_epsilon_pos', 'densityComparison repoint')

inv_d1 = '''/-! ## The Section 2 derivative bound at `|x| ≥ 1` -/

/-- Proves: `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne` (register row `side24-coefficient`,
Section 2 of SIDE24-COEFFICIENT-D23-20260924-v1, "At |x|>=1 this is at most 76 |x|^6 e^(-|x|^2/2)").
Method: the Section 2 pairing bound `|D^q Kinf (x) u| ≤ exp(-‖x‖²/2) · pairingSum q ‖x‖` is the
inventory theorem `derivativePairingBound` of this module (the Hermite-type recursion
`b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q` for the operator norms, `Aux.dp_recursion`, with
`ContinuousMultilinearMap.le_opNorm` at a unit tuple); the hypothesis-form helper
`Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound` combines it with the elementary
inequality `pairingSum q r ≤ 76 r^6` for `1 ≤ r`, `q ≤ 6` (`Aux.d1_pairingSum_le_of_one_le`:
`interval_cases`, `Finset.sum_range_succ`, `Nat.factorial`, `norm_num`, and the monomial bounds
`r^k ≤ r^6` from `pow_le_pow_right₀`, `one_le_pow₀`), by `mul_le_mul_of_nonneg_left` with
`0 ≤ exp(-‖x‖²/2)`.
Does not establish: the informal Section 2 display or its review and acceptance; the alignment of the
formal objects (`Kinf`, `IsUnitTuple`, `pairingSum`, Mathlib's `iteratedFDeriv ℝ q`) with the source's
(PENDING_INDEPENDENT_REVIEW); any identification with the main-branch targets
`UniversalLaw.Side24.pairing_sum_six` and `pairing_sum_le_six`, which kernel-check the coefficient
arithmetic `1+15+45+15=76` and `76` bounding the smaller orders there, not here; validity for `q > 6` or
for `‖x‖ < 1` (the Prop restricts to `q ≤ 6` and `1 ≤ ‖x‖`, as the source does); any enclosure of the
coefficient `c_{d,24}`; the disposition of `side24-coefficient`, which is REVIEWED_SCOPED as transcribed
in the register, not derived here. -/
theorem derivativeBoundAtLeastOne : UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne :=
  Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound derivativePairingBound'''

inv_d0 = '''/-! ## The Section 2 derivative bound at the origin -/

/-- Proves: `UniversalLaw.Spec.Side24.DerivativeBoundAtZero` (register row `side24-coefficient`,
Section 2 of SIDE24-COEFFICIENT-D23-20260924-v1, "contraction has absolute value at most 15").
Method: the hypothesis-form helper `Aux.d0_derivativeBoundAtZero_of_derivativePairingBound`
specialises the pairing bound `DerivativePairingBound` to `x = 0`, where `‖0‖ = 0`,
`exp (-(0 ^ 2) / 2) = 1` and `pairingSum q 0 ≤ 15` for `q ≤ 6` (`Aux.d0_pairingSum_zero_le`:
the values are `1, 0, 1, 0, 3, 0, 15`, by `interval_cases`, `Finset.sum_range_succ`, `norm_num`);
the pairing bound itself is the inventory theorem `derivativePairingBound` of this module
(`Aux.dp_recursion`, `Aux.dp_pairing_all`, `Aux.dp_abs_le_opNorm`).
Does not establish: the informal Section 2 display or its review and acceptance; the alignment of
the formal objects (`Kinf`, `IsUnitTuple`, `pairingSum`, Mathlib's `iteratedFDeriv ℝ q` at `0`) with
the source's (PENDING_INDEPENDENT_REVIEW); the double-factorial identity `5!! = 15` as such, which
is kernel-checked on main (`UniversalLaw.Side24.sixth_moment_double_factorial`) and is neither
re-proved nor cited here (only `pairingSum 6 0 = 15` is computed); validity for `q > 6` (the Prop
restricts to `q ≤ 6`, as the source does); any value of a contraction (only an upper bound on its
absolute value); any enclosure of the coefficient `c_{d,24}`; the disposition of
`side24-coefficient`, which is REVIEWED_SCOPED as transcribed in the register, not derived here. -/
theorem derivativeBoundAtZero : UniversalLaw.Spec.Side24.DerivativeBoundAtZero :=
  Aux.d0_derivativeBoundAtZero_of_derivativePairingBound derivativePairingBound'''

inv_rp = '''/-! ## The Section 1 reference pin-density product -/

/-- Proves: `UniversalLaw.Spec.Side24.ReferencePinDensityProduct` (register row `side24-coefficient`,
Section 1 of SIDE24-COEFFICIENT-D23-20260924-v1, "p_G(0) p_V(0) = (2pi)^(-d) / sqrt(3)").
Method: the hypothesis form `Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks` takes
the inventory theorem `referenceOddAndPinBlocks` of this module and uses only its first conjunct
`gradCov d (Kinf d) R = 1` and fourth conjunct `pinCov d (Kinf d) (R 0) R = diagonal (3, 1, …, 1)`
(both from the contractions `Aux.rb_order_two`, `Aux.rb_order_four` in an orthonormal frame); the
common core `Aux.rp_referencePinDensityProduct_of_blockIdentities` rewrites the two blocks and
finishes with the density algebra `Aux.rp_density_algebra` (`Matrix.det_one`, `Matrix.det_diagonal`,
`Finset.prod_ite_eq'`, `Matrix.mulVec_zero`, `Real.rpow_add`, `Real.rpow_neg`, `Real.sqrt_eq_rpow`).
Does not establish: the informal Section 1 display or its review and acceptance; the alignment of
the formal objects (`gaussDensity`, `gradCov`, `pinCov`, `derivContraction` through Mathlib's
`iteratedFDeriv ℝ` at `0`, `Orthonormal ℝ R`) with the source's (PENDING_INDEPENDENT_REVIEW); the
determinant identity `det diag(3,1,…,1) = 3` as the main-branch target
`UniversalLaw.Side24.pin_determinant_and_joint_dimension`, which is kernel-checked on main and is
neither cited nor re-used here (the determinant is recomputed by `Matrix.det_diagonal` in
`Aux.rp_det_pinDiagonal`); that `gaussDensity` is the density of any random vector or that any field
is constructed; the remaining conjuncts of `ReferenceOddAndPinBlocks` (`Var(t) = 15`,
`Cov(t, G) = (-3, 0, …, 0)`, `tau^2 = 6`), which this theorem does not use; the value for
`d ∉ {2, 3}` (the Prop restricts to `d = 2 ∨ d = 3`, as the source does); any enclosure of the
coefficient `c_{d,24}`; the disposition of `side24-coefficient`, which is REVIEWED_SCOPED as
transcribed in the register, not derived here. -/
theorem referencePinDensityProduct : UniversalLaw.Spec.Side24.ReferencePinDensityProduct :=
  Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks referenceOddAndPinBlocks'''

DOC_A = '''/-!
# SIDE24 analysis proofs: the theta family, the Section 2 derivative bounds and lattice sum,
the Section 1 reference blocks and pin density, the Section 4 ratio and density comparison
(statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1, register
row `side24-coefficient`) reviewed, accepted or "verified", it does not identify the formal
definitions with the informal objects (that alignment is PENDING_INDEPENDENT_REVIEW), it does
not enclose the coefficient `c_{d,24}`, and it promotes, closes, discharges or reclassifies
nothing. The disposition of `side24-coefficient` (REVIEWED_SCOPED) is transcribed in the
register, not derived here.

Provenance: the nine `Prop`s proved here were selected by the round-4 triage (workflow
`wf_a3531fd8-891`) of the registered `Prop`s still without a theorem; each proof was written and
verified in its own lane (one file per `Prop`) and the lane files were merged into this module,
with the developments the lanes reproduced from one another removed, call sites repointed to the
sibling inventory theorems, and the lanes' helper tags kept. Helper lemmas live in
`UniversalLaw.Proofs.Side24Analysis.Aux`, tagged by the `Prop` they serve (`th_`, `dp_`, `d1_`,
`d0_`, `rb_`, `rp_`, `ir_`, `li_`, `dc_`); the gate inventories only the nine theorems outside
that namespace. The mathematics, in the order of the module:

* `ThetaSummable d` (`th_`): `‖24 n‖^2 = 576 ∑ n_i^2`, the shift bound
  `exp(-‖z + v‖^2/2) ≤ exp(‖z‖^2/2) exp(-‖v‖^2/4)`, `Real.exp_sum`, the one-dimensional
  summability of `m ↦ exp(-c m^2)` on `ℤ` (`HurwitzKernelBounds.summable_f_int`) and a
  `Fintype.piFinset` product bound for `n ↦ ∏ i, g (n i)` over `Fin d → ℤ`;
  `Summable.of_nonneg_of_le` assembles the `Prop` for every `d`.
* `DerivativePairingBound` (`dp_`): no closed form for the iterated Fréchet derivative of the
  Gaussian is used; the operator norms `b_q = ‖iteratedFDeriv ℝ q (Kinf d) x‖` satisfy the
  Hermite-type recursion `b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q` (`norm_iteratedFDeriv_fderiv`,
  `fderiv ℝ (Kinf d) y = Kinf d y • (-(innerSL ℝ y))`, the Leibniz bound
  `norm_iteratedFDeriv_smul_le`), `b_0 = Kinf d x`, `b_1 = Kinf d x ‖x‖`, and the orders `q ≤ 6`
  match the unfolded `pairingSum`; `ContinuousMultilinearMap.le_opNorm` at a unit tuple.
* `DerivativeBoundAtLeastOne` (`d1_`) and `DerivativeBoundAtZero` (`d0_`): consequences of the
  pairing bound through hypothesis-form helpers, with `pairingSum q r ≤ 76 r^6` for `r ≥ 1` and
  `pairingSum q 0 ≤ 15`.
* `ReferenceOddAndPinBlocks` (`rb_`): the contractions of `Kinf d` at `0` are evaluated by
  peeling one direction at a time (`iteratedFDeriv_succ_apply_right`,
  `iteratedFDeriv_clm_apply_const_apply`) with explicit `HasFDerivAt` closed forms of the
  intermediate functions, giving `-⟪a, b⟫`, the three pairings at order four and `-15 ⟪u, u⟫^3`
  at order six; in an orthonormal frame the five blocks follow.
* `ReferencePinDensityProduct` (`rp_`): the two block identities of the previous `Prop` and the
  density algebra `(2π)^(-d/2) (2π)^(-d/2) 3^(-1/2) = (2π)^(-d) / √3`.
* `IntegrandRatioBounds` (`ir_`): pure real arithmetic in `epsilon = 10^(-108)`; Bernoulli's
  inequality, exponent monotonicity of the real power and two exact rational inequalities of
  degree five decided by `norm_num`.
* `LatticeImageSumBound` (`li_`): a uniform bound on the finite partial sums via a power-mean
  inequality, `Real.exp_sum`, the `Fintype.piFinset` factorisation and one-dimensional sums over
  `ℤ` compared with the geometric series through the registered `SuccessiveTermRatio`
  (`SpecificationProofs.Side24Elementary.successiveTermRatio`, whence the import); the bound
  obtained is `432 exp(-288) ≤ 1458 exp(-288)`.
* `DensityComparison` (`dc_`): real matrix analysis; Loewner antitonicity of the inverse and
  monotonicity of the determinant (through `CFC.sqrt`, `Matrix.PosSemidef.conjTranspose_mul_mul_same`,
  `Matrix.IsHermitian.det_eq_prod_eigenvalues`), determinant scaling and real-power bookkeeping.

Remarks recorded by the lanes, for the maintainers (nothing below is edited here): the
proof-packet README rows "Side24 derivative bounds" ("Mathlib has no closed form for the iterated
Fréchet derivatives of a Gaussian"), "Side24 lattice sums" ("Mathlib's Jacobi-theta results are
one-dimensional") and "Side24 covariance and density" ("the evaluation of the Fréchet derivatives
is not available"; `IntegrandRatioBounds` grouped as "built on the derivative contractions") give
informal reasons that these proofs show are not load-bearing for the `Prop`s proved here; the
statement module's docstring for `ThetaSummable` reads "Does not claim: that the family is summable
(true, but not shown here)". Each is a remark about an informal reason or a statement-packet text,
not about any claim; no register label changes.
-/'''

side24_analysis = '\n'.join([
    'import Specifications.Side24',
    'import SpecificationProofs.Side24Elementary',
    '',
    DOC_A,
    '',
    f'namespace {NS_A}',
    '',
    'open UniversalLaw.Spec.Side24',
    'open scoped ContDiff',
    'open scoped RealInnerProductSpace',
    'open scoped MatrixOrder',
    'open scoped Pointwise',
    '',
    'namespace Aux',
    '',
    section('`th_`: the theta family (`ThetaSummable`)', aux_th),
    section('`dp_`: the derivative pairing bound (`DerivativePairingBound`)', aux_dp),
    section('`d1_`: the derivative bound at `|x| ≥ 1` (`DerivativeBoundAtLeastOne`)', aux_d1),
    section('`d0_`: the derivative bound at the origin (`DerivativeBoundAtZero`)', aux_d0),
    section('`rb_`: the Section 1 reference blocks (`ReferenceOddAndPinBlocks`)', aux_rb),
    section('`rp_`: the reference pin-density product (`ReferencePinDensityProduct`)', aux_rp),
    section('`ir_`: the Section 4 integrand-ratio bounds (`IntegrandRatioBounds`)', aux_ir),
    section('`li_`: the lattice-image sum bound (`LatticeImageSumBound`)', aux_li),
    section('`dc_`: the Section 4 Gaussian density comparison (`DensityComparison`)', aux_dc),
    'end Aux',
    '',
    inv_th, '', inv_dp, '', inv_d1, '', inv_d0, '', inv_rb, '', inv_rp, '', inv_ir, '', inv_li, '', inv_dc,
    '',
    f'end {NS_A}',
    '',
])

# ----------------------------------------------------------------------------- Side24Interfaces
NS_I = 'UniversalLaw.Proofs.Side24Interfaces'
c1, tm, rl = read('ConeMomentD1Identity'), read('TraceMoments'), read('RayleighSquareLaw')
aux_c1 = aux_of(c1)
aux_tm = aux_of(tm)
aux_rl = aux_of(rl)
aux_tm = remove_block(aux_tm, '/-- Second moment of `N(0, v)`: `∫ t^2 = v`, from `variance_fun_id_gaussianReal` rewritten\n'
                      'through `variance_eq_integral` (the mean is `0`). -/\ntheorem tm_second_moment', 'tm_second_moment')
aux_tm = replace_once(aux_tm, '/-! ### (b) Moments of the centred real Gaussian `gaussianReal 0 v` -/',
                      '/-! ### (b) Moments of the centred real Gaussian `gaussianReal 0 v`\n\n'
                      'The second moment `∫ t^2 = v` is `c1_second_moment_gaussianReal` above (the `tm_` copy was\n'
                      'removed at integration). -/', 'tm section (b)')
aux_rl = remove_block(aux_rl, "/-- The triple `(s, x, y)` is a.e.-measurable: its image measure under `P` is the product\n"
                      "Gaussian law (`sxy_law`); were the triple not a.e.-measurable, `Measure.map` would be a Dirac\n"
                      "mass (`Measure.map_of_not_aemeasurable_of_ne_zero`), which gives a singleton measure `1`,\n"
                      "whereas the product of non-degenerate Gaussians has `NullSingletonClass`. -/\n"
                      "theorem rl_aemeasurable_triple", 'rl_aemeasurable_triple')
aux_rl = replace_once(aux_rl, '  (rl_aemeasurable_triple I).snd', '  (tm_aemeasurable_triple I).snd', 'rl_aemeasurable_xy repoint')
aux_rl = replace_once(aux_rl, '  have hT := rl_aemeasurable_triple I', '  have hT := tm_aemeasurable_triple I', 'rl_law_xy repoint')
aux_rl = replace_once(aux_rl, '/-! ### (c) The interface: measurability and the law of `(x, y)` -/',
                      '/-! ### (c) The interface: measurability and the law of `(x, y)`\n\n'
                      'A.e.-measurability of the triple `(s, x, y)` is `tm_aemeasurable_triple` above (the `rl_` copy\n'
                      'was removed at integration). -/', 'rl section (c)')
for name in ('tm_second_moment', 'rl_aemeasurable_triple'):
    if name in aux_tm or name in aux_rl:
        sys.exit(f'interfaces: {name} still referenced')
inv_c1 = inventory_of(c1, NS_I)
inv_tm = inventory_of(tm, NS_I)
inv_rl = inventory_of(rl, NS_I)
inv_tm = replace_once(inv_tm, 'there: `∫ t^2 = 5/3` (`Aux.tm_second_moment`, from `variance_fun_id_gaussianReal`),',
                      'there: `∫ t^2 = 5/3` (`Aux.c1_second_moment_gaussianReal`, shared with `ConeMomentD1Identity`,\nfrom `variance_fun_id_gaussianReal`),', 'traceMoments doc')
inv_tm = replace_once(inv_tm, '    rw [e, Aux.tm_second_moment]', '    rw [e, Aux.c1_second_moment_gaussianReal]', 'traceMoments repoint')
inv_rl = replace_once(inv_rl, '`sxy_law`, after `Aux.rl_aemeasurable_triple` derives a.e.-measurability of the triple from the',
                      '`sxy_law`, after `Aux.tm_aemeasurable_triple`, shared with `TraceMoments`, derives\na.e.-measurability of the triple from the', 'rayleigh doc')
if 'tm_second_moment' in inv_tm or 'rl_aemeasurable_triple' in inv_rl:
    sys.exit('interfaces inventory still references a removed helper')

DOC_I = '''/-!
# SIDE24 interface proofs: the `m = 1` cone moment identity, the trace moments and the law of
`R^2 = x^2 + y^2` over `ConeMomentInterface` (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here), applied to its own parameter `I : ConeMomentInterface`. A kernel-checked
proof of such a `Prop` is evidence about that `Prop` only: it does not make the informal Layer 0
note (SIDE24-COEFFICIENT-D23-20260924-v1, register row `side24-coefficient`) reviewed, accepted
or "verified", it does not identify the formal definitions with the informal objects (that
alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`, and it
promotes, closes, discharges or reclassifies nothing. Nothing here constructs an inhabitant of
`ConeMomentInterface` or derives its law fields from any field; the theorems hold for every
inhabitant and use only the law fields `A_law`, `sxy_law` and the instance `isProb`.

Provenance: the three `Prop`s proved here were selected by the round-4 triage (workflow
`wf_a3531fd8-891`) of the registered `Prop`s still without a theorem; each proof was written and
verified in its own lane and the lane files were merged into this module, with the helpers the
lanes proved twice removed (one copy kept, call sites repointed) and the lanes' tags kept. Helper
lemmas live in `UniversalLaw.Proofs.Side24Interfaces.Aux`, tagged `c1_`, `tm_`, `rl_`; the gate
inventories only the three theorems outside that namespace. The mathematics:

* The common step: a map whose image measure under a probability measure is a non-degenerate
  Gaussian (or a product of them) is a.e.-measurable, because at this Mathlib revision the image
  measure of a non-a.e.-measurable map under a nonzero measure is a Dirac measure
  (`Measure.map_of_not_aemeasurable_of_ne_zero`), which charges a singleton, while the Gaussian
  law has `NullSingletonClass` (`c1_aemeasurable_of_map_eq_gaussian` for `A`,
  `tm_aemeasurable_triple` for `(s, x, y)`). Integrals are then transported along the laws by
  `integral_map`, and probabilities by `Measure.map_apply_of_aemeasurable`.
* `ConeMomentD1Identity I` (`c1_`): the negative half of the second moment of
  `gaussianReal 0 (8/3)` is `(8/3)/2 = 4/3` (`gaussianReal_map_neg`, `integral_add_compl`,
  `integral_Ici_eq_integral_Ioi`, `variance_fun_id_gaussianReal`).
* `TraceMoments I` (`tm_`): the first marginal of `sxy_law` is `gaussianReal 0 (5/3)`
  (`Measure.map_fst_prod`); `∫ t^2 = 5/3`, `∫ t^4 = 3 (5/3)^2 = 25/3` (`iteratedDeriv_mgf_zero`
  with the explicit fourth derivative of `exp (c t^2/2)` at `0`), `∫ exp(-t^2/2) = √(3/8)`
  (`integral_gaussianReal_eq_integral_smul`, `integral_gaussian`).
* `RayleighSquareLaw I` (`rl_`): the second marginal of `sxy_law` is
  `gaussianReal 0 1 ⊗ gaussianReal 0 1` (`Measure.map_snd_prod`); the measure of the closed disk of
  radius `√a` is computed by hand (Mathlib has no Rayleigh or chi-squared law): `prod_withDensity`,
  `withDensity_apply`, `ofReal_integral_eq_lintegral_ofReal`, polar coordinates
  (`integral_comp_polarCoord_symm`, `setIntegral_prod_mul`), the angular integral `2π` and the
  radial integral by the fundamental theorem of calculus; the right-hand side
  `∫_0^a exp(-z/2)/2 dz = 1 - exp(-a/2)` is the same elementary integral.

Remark recorded by the lanes, for the maintainers (not edited here): the proof-packet README row
"Side24 interface statements" gives as its informal reason that the Gaussian moment identities
"need the conditional law `A_u | V_u = 0` built in Mathlib's Gaussian measure theory"; for the three
`Prop`s proved here the interface's law fields suffice. This is a remark about an informal reason,
not about any claim; no register label changes.
-/'''

side24_interfaces = '\n'.join([
    'import Specifications.Side24',
    '',
    DOC_I,
    '',
    f'namespace {NS_I}',
    '',
    'open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24',
    '',
    'namespace Aux',
    '',
    section('`c1_`: the `m = 1` cone moment identity (`ConeMomentD1Identity`)', aux_c1),
    section('`tm_`: the trace moments (`TraceMoments`)', aux_tm),
    section('`rl_`: the law of `R^2` (`RayleighSquareLaw`)', aux_rl),
    'end Aux',
    '',
    inv_c1, '', inv_tm, '', inv_rl,
    '',
    f'end {NS_I}',
    '',
])

# ----------------------------------------------------------------------------- RNCluster
NS_R = 'UniversalLaw.Proofs.RNCluster'
rc = read('RareClusterPoissonObstructionO')
aux_rc = aux_of(rc)
inv_rc = inventory_of(rc, NS_R)
DOC_R = '''/-!
# RN: kernel-checked proof of the abstract rare-cluster Poisson obstruction
(`RareClusterPoissonObstructionO`, statement module `Specifications.RN`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/RN.lean` (statement packet `specifications/lean_statements_20261010`, which is
not modified here), applied to its own parameters `μ a A`. A kernel-checked proof of such a
`Prop` is evidence about that `Prop` only: it does not make the informal Layer 0 note
(OA-C6-RARE-CLUSTER-20260929-v1, Theorem O of c6_rare_cluster_laws_20260929/PROOF.md) reviewed,
accepted or "verified", it does not identify the formal objects (`pmfTV`, `poissonPMF`, the
family `μ`) with the source's (that alignment is PENDING_INDEPENDENT_REVIEW), it says nothing
about whether (H1) or (H2) hold for any Gaussian window count, and it promotes, closes,
discharges or reclassifies nothing. The Radon-Nikodym uniformity lemma `D3-LEMMA-RN-UNIF` and the
five validity premises of Theorem D1 v2.2(2) are untouched and remain OPEN.

Provenance: this `Prop` was selected by the round-4 triage (workflow `wf_a3531fd8-891`) of the
registered `Prop`s still without a theorem; the proof was written and verified in its own lane
and is placed here unchanged. Helper lemmas live in `UniversalLaw.Proofs.RNCluster.Aux` (prefix
`rc_`); the gate inventories only the one theorem outside that namespace.

The route is the source's, with the thresholds rearranged so that no `A^2` appears:
* the antecedents at `r = r0` give `2 a r0^3 ≤ 2 (1 - μ 0 - μ 1) ≤ ∑ n μ n ≤ A r0^3`, hence
  `2a ≤ A` and in particular `A > 0`;
* `rStar := min r0 (min (a/2) (min (1/(a + 2A)) (1/2)))`;
* upper half, witness `lam = 0`: `poissonPMF 0 = δ_0`, so `pmfTV (μ r) (poissonPMF 0) = 1 - μ r 0
  = ∑_{n ≥ 1} μ r n ≤ ∑ n μ r n ≤ A r^3`;
* lower half, `lam ≥ 0`: half the `l^1` distance dominates the difference of the masses of every
  tail event `{n ≥ k}` (for two summable families of total mass one). For `lam ≤ r^2` the tail
  `{n ≥ 2}` gives `pmfTV ≥ (1 - μ 0 - μ 1) - (1 - e^{-lam} - lam e^{-lam}) ≥ a r^3 - lam^2 ≥
  a r^3 - r^4 ≥ (a/2) r^3`; for `lam > r^2` the tail `{n ≥ 1}` gives `pmfTV ≥ (1 - e^{-lam}) -
  (1 - μ 0) ≥ (1 - e^{-r^2}) - A r^3 ≥ r^2/2 - A r^3 ≥ (a/2) r^3`, using `x/2 ≤ 1 - e^{-x}` on
  `[0, 1/2]` and `r (a + 2A) ≤ 1`.
Total mass one of `poissonPMF lam` for `lam ≥ 0` is Mathlib's
`ProbabilityTheory.hasSum_one_poissonMeasure`.

Remark recorded by the lane, for the maintainers: a theorem of this `Prop` kernel-checks only the
abstract implication (H1) ∧ (H2) ⇒ (3.1), which is what the source claims for Theorem O; the
hypotheses imply `2a ≤ A`, so the `Prop` is not vacuous. The claim row keeps its register labels.
-/'''
rn_cluster = '\n'.join([
    'import Specifications.RN',
    '',
    DOC_R,
    '',
    f'namespace {NS_R}',
    '',
    'open UniversalLaw.Spec.RN',
    '',
    'namespace Aux',
    '',
    aux_rc,
    '',
    'end Aux',
    '',
    inv_rc,
    '',
    f'end {NS_R}',
    '',
])

for name, text in (('Side24Analysis', side24_analysis), ('Side24Interfaces', side24_interfaces), ('RNCluster', rn_cluster)):
    if '#' in re.sub(r'/-.*?-/', '', text, flags=re.S) and re.search(r'^\s*#', text, re.M):
        sys.exit(f'{name}: a `#` command survived')
    if re.search(r'\n\n\n', text):
        text = re.sub(r'\n{3,}', '\n\n', text)
    (W / (name + '.lean')).write_text(text, encoding='utf-8')
    print(name, len(text.splitlines()), 'lines')
