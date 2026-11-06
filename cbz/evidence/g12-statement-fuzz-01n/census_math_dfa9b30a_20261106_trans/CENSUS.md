# G12 census of Math- at `dfa9b30a` with the transcendental enclosure backend (6 November 2026)

**What this is.** The census of [`census_math_dfa9b30a_20261106/`](../census_math_dfa9b30a_20261106/CENSUS.md) (the
finite-vector backend, main#755) re-taken with `tools/cbz/statement_fuzz.py` extended by the **transcendental
enclosure backend**: `Real.exp` of a rational is an exact exponential monomial and sums, products and integer powers
of such values stay in the exact normal form Σ c·e^q (equalities decided by linear independence of the e^q over ℚ,
Lindemann–Weierstrass); `Real.log`, `Real.pi` / `π`, `Real.sin` / `cos` / `tan` / `sinh` / `cosh` / `arctan`,
`Real.exp` of irrational arguments and nested radicals are enclosed by rational intervals from series with proved
remainder bounds (outward rounding only), a closed comparison is refined 64 → 128 → 256 → 512 → 1024 bits and
reported UNSUPPORTED when still undecided, and a sampled instance the enclosures do not decide at 256 bits is
skipped and counted (`undecided`). Same source archive, same 16 groups, same command (`run_census.sh`, seed 1,
64 samples); `verdict_changes_20261106_finvec_to_trans.json` lists every verdict (and every WEAK flag) that moved.

**Result: 3,782 declarations — 40 PASS · 229 CORROBORATED · 0 FAIL · 3,513 UNSUPPORTED**
(finite-vector census: 17 · 203 · 0 · 3,562). **49 verdicts moved, 49 of them from
UNSUPPORTED**: 23 to PASS and 26 to CORROBORATED; **0 to FAIL** — every one of these statements is kernel-proved, so a FAIL
would have been a reader defect. No previously decided verdict changed. By group: `specifications/lean_statement_proofs_20261011` 34, `frontiers/source_derivative_gaussian_20261025` 10, `reviews/w3_side24_source_proposal_20261008` 3, `frontiers/source_lifetime_realization_20261017` 1, `frontiers/source_smooth_realization_20261011` 1.
4 closed verdicts needed more than 64 bits (the 29- and 32-digit enclosures of `e` and `exp x_U` in `P15Interpolation`,
decided at 128 bits). 15 declarations are UNSUPPORTED *because of* the enclosure layer rather than a missing
construct: one closed equality undecided to 1024 bits (`fixture_seam_phase`), eleven sampled identities between trigonometric
values whose instances are undecided except at degenerate arguments (`phase_axis`, `phase_lattice_shift`, `phase_spatial_sub`,
`atom_phase_neg`, `srcQTheta_eq`, `pow_three_eq_exp` …, reported "sampling cannot corroborate the statement" with the counts),
and two whose hypotheses are themselves undecided identities (`fixture_inertia_pos` / `_neg`); enclosures can refute such an
equality but never confirm it, so these stay UNSUPPORTED, never PASS or CORROBORATED. 5 CORROBORATED verdicts carry the
WEAK flag — the same 5 as in the finite-vector census; the new `li_geom_tail`-style statements over ℕ reach 28 instances
(not WEAK) because `Real.exp` arguments up to 10⁶ in absolute value are enclosed and the pool's larger ℕ values are counted
as Infeasible, never as holding.

| Group | Declarations | PASS | CORROBORATED | FAIL | UNSUPPORTED | finite-vector census (P · C · U) |
|---|---:|---:|---:|---:|---:|---|
| `formal` | 63 | 3 | 56 | 0 | 4 | 3 · 56 · 4 |
| `frontiers/source_derivative_gaussian_20261025` | 2225 | 9 | 103 | 0 | 2113 | 8 · 94 · 2123 |
| `frontiers/source_lifetime_realization_20261017` | 182 | 0 | 2 | 0 | 180 | 0 · 1 · 181 |
| `specifications/lean_statement_proofs_20261011` | 559 | 25 | 20 | 0 | 514 | 3 · 8 · 548 |
| `frontiers/source_smooth_realization_20261011` | 122 | 0 | 2 | 0 | 120 | 0 · 1 · 121 |
| `frontiers/determinant_moment_20261025` | 85 | 0 | 24 | 0 | 61 | 0 · 24 · 61 |
| `frontiers/compact_morse_elder_adapter_20261018` | 121 | 0 | 2 | 0 | 119 | 0 · 2 · 119 |
| `frontiers/regular_superlevel_geometry_20261020` | 23 | 0 | 0 | 0 | 23 | 0 · 0 · 23 |
| `frontiers/asymmetric_elder_bridge_20261017` | 31 | 0 | 2 | 0 | 29 | 0 · 2 · 29 |
| `companions/torus_filtration_20261011` | 130 | 1 | 0 | 0 | 129 | 1 · 0 · 129 |
| `specifications/lean_statements_20261010` | 0 | 0 | 0 | 0 | 0 | 0 · 0 · 0 |
| `reviews/w3_side24_source_proposal_20261008` | 77 | 0 | 10 | 0 | 67 | 0 · 7 · 70 |
| `companions/d2_square_schur_20261006` | 4 | 0 | 2 | 0 | 2 | 0 · 2 · 2 |
| `companions/d2_atom_integral_20261006` | 6 | 0 | 0 | 0 | 6 | 0 · 0 · 6 |
| `companions/d2_moment_bridge_20261006` | 16 | 0 | 3 | 0 | 13 | 0 · 3 · 13 |
| `frontiers/cap_first_exit_lean_20261002` | 138 | 2 | 3 | 0 | 133 | 2 · 3 · 133 |

## What the transcendental backend reads, and what stays outside

`Real.exp q` for rational q is the exact monomial e^q; `+ - *`, integer powers, `Real.sinh` / `Real.cosh` of rationals
and the inverse of a monomial stay exact, so `Real.exp 2 = Real.exp 1 ^ 2`, `Real.exp 1 * Real.exp (-1) = 1` and
`Real.cosh 1 ^ 2 - Real.sinh 1 ^ 2 = 1` are PASS with no enclosure, and `Real.log (c · e^q) = log |c| + q` is read
through the monomial. Everything else is an enclosure `[lo, hi]` with exact rational endpoints: e from Σ 1/k! with the
tail bound 1/(N!·N); e^r for |r| ≤ 1/2 from the Taylor series with the Lagrange remainder 2·|r|^(N+1)/(N+1)!; log from
`q = 2^k·m`, `log m = 2·atanh((m−1)/(m+1))` with the geometric tail bound; π by Machin's formula from the alternating
arctan series (the first omitted term bounds the remainder); sin and cos after reduction modulo 2π (π itself an
interval) from their Taylor series with the remainder |y|^N/N!, widened to ±1 where a critical point may lie inside;
`Real.sqrt` of an enclosure or of an irrational algebraic value by integer square roots. Every series term is rounded
outward to a bounded number of significant bits, so the endpoints stay small at 1024 bits and for `e^n` with |n| up
to 10⁶ (e^(10⁶) ≈ 2^(1.44·10⁶) is within the result caps; beyond, Infeasible). Mathlib's totalisations hold: `Real.log 0 = 0`,
`Real.log (-x) = Real.log x`, `x / 0 = 0`, `Real.sqrt` of a non-positive value is 0. A comparison the enclosures do
not decide raises *Undecided*: a closed statement is refined up to 1024 bits and then UNSUPPORTED (the reason names
the operator, the precision and the two enclosures); a sampled instance is retried at 256 bits, then skipped and
counted; a statement whose undecided instances outnumber the decided ones is UNSUPPORTED, not weakly corroborated.
A side of `∧` / `∨` / `→` that is undecided does not decide the connective by itself (`A ∨ B` is true when B is).

Outside, fail-closed and named: `Real.rpow` (a real exponent), `Real.arcsin` / `arccos` / `Gamma` and every function
the backend does not model (`unknown function`), `Real.exp` beyond |q| ≤ 10⁶, `min`/`max` of overlapping enclosures
in an exact equality, and — by construction — any *equality* between two values that are not both exponential
polynomials: enclosures can separate them (FAIL) but never confirm equality, so `Real.cos Real.pi = -1`,
`Real.sin 1 ^ 2 + Real.cos 1 ^ 2 = 1`, `Real.arctan 1 = Real.pi / 4` are UNSUPPORTED with that reason, never PASS.

## Why UNSUPPORTED (classes, all groups)

| Class | Declarations | finite-vector census |
|---|---:|---:|
| binder outside the interpreter | 2177 | 2177 |
| proposition outside the interpreter | 215 | 215 |
| definition result type outside the interpreter | 197 | 197 |
| unparsed statement | 153 | 153 |
| unknown function | 148 | 134 |
| other | 106 | 105 |
| existential quantifier | 97 | 97 |
| structural conclusion | 92 | 92 |
| unknown name | 91 | 116 |
| definition outside the interpreter | 72 | 71 |
| Mathlib notation | 71 | 71 |
| hypothesis outside the interpreter | 51 | 51 |
| enclosures do not decide the statement | 15 | 0 |
| unbounded quantifier inside the statement | 12 | 12 |
| ascription to an unsupported type | 4 | 2 |
| quantifier over an unsupported type | 4 | 4 |
| application outside the interpreter | 3 | 3 |
| real exponent (Real.rpow) | 3 | 0 |
| no sample satisfied the hypotheses | 2 | 2 |

The remaining binder types are structures and function spaces (`SmoothSample 2 L`, `E d`, `IsUnit (…)`,
`RealizedData X`, `ℝ → ℝ`, `superlevelBar_E d → ℝ`, `(Fin 2 → ℝ) → ℝ`, `SourceMode 2` …), existential quantifiers,
structural conclusions and `∃`/`∑'`/`*ᵥ` notation — none is a target of this backend. The `transcendental function`
class now holds only the functions the backend does not model.

## Every PASS / CORROBORATED statement by group (new ones marked)

**`formal`**
- CORROBORATED: `ResearchFormalCoreR1.ec005_fold_gap`
- CORROBORATED: `ResearchFormalCoreR1.ec008_factor_expansion`
- CORROBORATED: `ResearchFormalCoreR1.ec010_generic`
- CORROBORATED: `ResearchFormalCoreR1.ec010_transverse`
- CORROBORATED: `ResearchFormalCoreR1.ec011_scalar_cancellation`
- CORROBORATED: `ResearchFormalCoreR1.ec014_contact_power`
- CORROBORATED: `ResearchFormalCoreR1.d2_schur_numerator`
- CORROBORATED: `ResearchFormalCoreR1.d2_det_axis`
- CORROBORATED: `ResearchFormalCoreR1.d2_det_diagonal`
- CORROBORATED: `ResearchFormalCoreR1.d2_det_interpolation`
- CORROBORATED: `ResearchFormalCoreR1.d2_schur_ratio`
- CORROBORATED: `ResearchFormalCoreR1.d2_numerator_pos`
- CORROBORATED: `ResearchFormalCoreR1.d2_det_pos`
- CORROBORATED: `ResearchFormalCoreR1.d2_schur_pos`
- CORROBORATED: `ResearchFormalCoreR1.d2_det_le_endpoint_max`
- CORROBORATED: `ResearchFormalCoreR1.d2_schur_endpoint_lower`
- CORROBORATED: `ResearchFormalCoreR1.d2_tau_cumulant_eq`
- CORROBORATED: `ResearchFormalCoreR1.d2_tau_axis`
- CORROBORATED: `ResearchFormalCoreR1.d2_tau_diagonal`
- CORROBORATED: `ResearchFormalCoreR1.d2_tau_pos`
- CORROBORATED: `ResearchFormalCoreR1.d2_gaussian_control`
- PASS: `ResearchFormalCoreR1.d2_degenerate_control`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_integral_cs`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_event_cs`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_event_numerator`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_measure_transfer`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_measure_transfer_r8_to_r3`
- PASS: `ResearchFormalCoreR1.p02_lm008_sqrt_counterexample`
- PASS: `ResearchFormalCoreR1.p02_lm008_upper_normalizer_counterexample`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_moment40_tail_finite`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_moment40_tail_of_finite`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_markov_event`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_badJet_subset`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_moment40_tail`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_moment40_weighted_r4`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_moment40_weighted_r3`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_r2_cancel`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_cross_multiplied`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_quotient_bound`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_bad_event_threshold`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_power40_identity`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_r4_le_r3`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_palm_r4_to_r3`
- CORROBORATED: `ResearchFormalCoreR1.weightPerturbation_integral`
- CORROBORATED: `ResearchFormalCoreR1.weightPerturbation_setIntegral`
- CORROBORATED: `ResearchFormalCoreR1.weightPerturbation_normalizer_lower`
- CORROBORATED: `ResearchFormalCoreR1.weightPerturbation_event_integral_bounds`
- CORROBORATED: `ResearchFormalCoreR1.weightPerturbation_quotient_bound`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_event_perturbation`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_event_perturbation_r2`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_apply`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_isProbabilityMeasure`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_event_real`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_absolutelyContinuous`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_congr_ae`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_zero`
- CORROBORATED: `ResearchFormalCoreR1.weightedLaw_one`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_probability_transfer`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_probability_transfer_r8_to_r3`

**`frontiers/source_derivative_gaussian_20261025`**
- CORROBORATED: `AdaptiveFoldBoundaryPath.side_drop_le`
- CORROBORATED: `AdaptiveFoldTransverseRidge.abs_neg_div_le_of_le_neg`
- CORROBORATED: `I4GaussianMajorant.exp_mul_abs_le` — new
- CORROBORATED: `I4SourceActualFoldCandidate.planar_quadratic_complete_square`
- CORROBORATED: `SourceActualFoldH0.fold_interval_endpoints`
- CORROBORATED: `I4SourceActualMorseLocalModel.point2_decomp`
- PASS: `I4SourceActualMorseLocalModel.fixture_scale_ne` — new
- CORROBORATED: `I4SourceActualMorseLocalModel.fixture_point_sq_pos`
- CORROBORATED: `I4SourceActualQuotientSaddleCollar.sc64_weighted_positive`
- CORROBORATED: `I4SourceActualQuotientSaddleCollar.sc64_positive_cone_iff`
- CORROBORATED: `I4SourceActualQuotientSaddleCollar.sc64_negative_cone_iff`
- CORROBORATED: `I4SourceActualSaddleUppercuts.ug65_square_mono`
- CORROBORATED: `I4SourceActualSaddleUppercuts.ug65_square_of_abs_le`
- CORROBORATED: `I4SourceActualSaddleUppercuts.ug65_reflect_involutive`
- CORROBORATED: `I4SourceAdaptiveFoldChart.chart_endpointM`
- CORROBORATED: `I4SourceAdaptiveFoldChart.chart_endpointS`
- CORROBORATED: `SourceCompactFoldWeightedProbability.sourceCompactCutoffA_nonneg`
- CORROBORATED: `SourceCompactFoldWeightedProbability.varying_cutoff_failure_inclusion`
- CORROBORATED: `SourceCompactGapNormalizer.gaussianCompactLowerConstant_pos` — new
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_one_sixth`
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_eq_confluent`
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_norm_le`
- CORROBORATED: `I4SourceConfluentLimits.average_norm_le`
- CORROBORATED: `I4SourceConfluentPins.rawTransform_confluentTransform`
- CORROBORATED: `I4SourceConfluentPins.confluentTransform_rawTransform`
- CORROBORATED: `I4SourceEndpointEstimates.sourcePhysicalCoordinates_inverse`
- CORROBORATED: `I4SourceEndpointHessianWitness.sourceC_ne_zero` — new
- CORROBORATED: `I4SourceEndpointHessianWitness.pinMultiplier_value` — new
- CORROBORATED: `SourceFoldWeightedProbability.sourceScale_pos`
- CORROBORATED: `SourceFoldWeightedProbability.sourceCutoffEps_pos`
- CORROBORATED: `SourceFoldWeightedProbability.sourceCutoffA_nonneg`
- CORROBORATED: `SourceFoldWeightedProbability.cutoff_failure_inclusion`
- CORROBORATED: `I4SourceFramedConfluentCovariance.average_abs_le`
- CORROBORATED (WEAK): `I4SourceFramedConfluentCovariance.ct58_lower_inverse`
- CORROBORATED (WEAK): `I4SourceFramedConfluentCovariance.ct58_frame_delta_four`
- CORROBORATED: `I4SourceFramedConfluentCovariance.compact_quad_smul`
- CORROBORATED: `I4SourceFramedConfluentCovariance.compact_square_nonneg`
- CORROBORATED: `I4SourceFramedConfluentCovariance.compact_square_eq_zero`
- PASS: `I4SourceGapConditionalMorseDistinct.source_directionX_eq`
- PASS: `I4SourceGapConditionalMorseDistinct.source_directionY_eq`
- CORROBORATED: `I4SourceGapConditionalMorseDistinct.constraintSquare_pos_iff`
- CORROBORATED: `I4SourceIntrinsicH0WeightedProbability.planar_det_neg_of_signs`
- CORROBORATED: `I4ActualSourceMorse.sourceC_ne_zero` — new
- PASS: `I4ActualSourceMorse.morseDirectionX_eq`
- PASS: `I4ActualSourceMorse.morseDirectionY_eq`
- PASS: `I4SourcePinCurvatureBounds.directionX_norm_le_one`
- PASS: `I4SourcePinCurvatureBounds.directionY_norm_le_one`
- CORROBORATED: `I4SourcePinCurvatureBounds.axis_point_sub_norm_le`
- CORROBORATED: `I4SourcePinCurvatureBounds.endpoint_det_abs_le`
- CORROBORATED: `I4SourcePinCurvatureBounds.curvature_slab_of_upper`
- CORROBORATED: `I4ActualQuarticTransfer.corrected_axial_constant`
- CORROBORATED: `I4SourcePinPreservingQuartic.euclidean_two_norm_sq`
- CORROBORATED: `I4SourcePinPreservingQuartic.euclidean_two_norm_le_abs_sum`
- CORROBORATED: `I4SourcePinPreservingQuartic.euclidean_two_norm_le_two_mul`
- PASS: `I4SourcePinPreservingQuartic.coordinateDirection_zero`
- PASS: `I4SourcePinPreservingQuartic.coordinateDirection_one`
- CORROBORATED: `I4SourcePinPreservingQuartic.coordinate_decomposition`
- CORROBORATED: `I4SourcePinPreservingQuartic.one_le_quarticFstar`
- CORROBORATED: `I4SourcePinPreservingQuartic.quarticWidth_pos`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_raw_regularity`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_scaled_error_budget`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_adaptive_tolerance`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_quarticWidth_tolerance`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_curvature_threshold`
- CORROBORATED: `I4SourcePinPreservingQuartic.certificate_width_geometry`
- CORROBORATED: `I4SourcePinPreservingQuartic.plane_decompose`
- CORROBORATED (WEAK): `I4SourcePinPreservingQuartic.coordinateDirection_norm_le_one`
- CORROBORATED: `I4SourcePinPreservingQuartic.quartic_strip_in_ball_two`
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_one_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_pos`
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_sum_abs_eq`
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_coord_abs_le_sub_one`
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_coord_abs_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_pos`
- CORROBORATED: `I4QuarticCaptureAlgebra.captureFstar_le_radius`
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_radius_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_le_one_sixteenth`
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_radius_le_one_sixtyfourth`
- CORROBORATED: `I4QuarticCaptureAlgebra.captureSigma_pos`
- CORROBORATED: `I4QuarticCaptureAlgebra.captureSigma_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.four_captureWidth_le_captureSigma`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_three_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_two_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_add_six_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_add_five_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.strip_coordinate_bounds`
- CORROBORATED: `I4QuarticCaptureAlgebra.quartic_transverse_forcing_bound`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_four_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.quartic_transverse_remainder_bound`
- CORROBORATED: `I4QuarticCaptureAlgebra.quartic_axial_remainder_bound`
- CORROBORATED: `I4QuarticCaptureAlgebra.width_identity`
- CORROBORATED: `I4SourceRemoteJetWitness.annihilator_pin_values` — new
- CORROBORATED: `I4SourceRemoteJetWitness.sourceC_ne_zero` — new
- CORROBORATED: `I4SourceTwoRemoteFirstJetWitness.sourceC_ne_zero` — new
- CORROBORATED: `ResearchFormalCoreR1.dm2_scale`
- CORROBORATED: `ResearchFormalCoreR1.dm2_factors`
- CORROBORATED: `ResearchFormalCoreR1.dm2_untyped`
- CORROBORATED: `ResearchFormalCoreR1.egm_coordinate_bound`
- CORROBORATED: `ResearchFormalCoreR1.egm_factor_bound`
- CORROBORATED: `ResearchFormalCoreR1.egm_untyped`
- CORROBORATED: `ResearchFormalCoreR1.enb_maximum_factor`
- CORROBORATED: `ResearchFormalCoreR1.enb_saddle_iff`
- CORROBORATED: `ResearchFormalCoreR1.gm8_finite_radius_power_bound`
- CORROBORATED: `ResearchFormalCoreR1.gm8_sqrt_pow`
- CORROBORATED: `ResearchFormalCoreR1.nfloor_contact_value`
- CORROBORATED: `I4Weights.frequency_l1_le_product`
- CORROBORATED: `I4Weights.sourceA_pos` — new
- CORROBORATED: `TorusFiltration.superlevel_mono`
- CORROBORATED: `TorusFiltration.superlevel_neg_height`
- CORROBORATED: `TorusFiltration.elder_control_birth`
- CORROBORATED: `TorusFiltration.elder_control_death`

**`frontiers/source_lifetime_realization_20261017`**
- CORROBORATED: `I4Weights.frequency_l1_le_product`
- CORROBORATED: `I4Weights.sourceA_pos` — new

**`specifications/lean_statement_proofs_20261011`**
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.quarter_pow_identity`
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.quarter_ratio_le`
- PASS: `UniversalLaw.Proofs.P15Budget.Aux.log_benchmark_le` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.two_lt_exp_one` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.three_exp_sub_two_pos` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.exp_two_eq` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.exp_neg_two_eq` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.exp_neg_three_eq` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.log_three_exp_sub_two_lt_two` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.taylor_sum_eq`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.exp_eleven_sixths_gt` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.three_exp_sub_two_lt` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.log_three_exp_sub_two_lt` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Hazard.Aux.neg_log_one_sub_ge` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_exp_one_gt` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_exp_one_lt` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_exp_xU_ge` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_exp_xL_le` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_log_lt` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_lt_log` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_hStar_pos` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.cc_affineExp_pos` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.f6_one_sub_exp_neg_mem` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Structure.Aux.neg_log_one_sub_gt` — new
- CORROBORATED: `UniversalLaw.Proofs.RNCluster.Aux.rc_half_le_one_sub_exp` — new
- CORROBORATED: `UniversalLaw.Proofs.RNCluster.Aux.rc_poisson_tail_two_le_sq` — new
- CORROBORATED: `UniversalLaw.Proofs.RNProbability.Aux.one_le_log_one_div` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_geom_tail` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_pow6_succ_le` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_exp_succ_le` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_sum_cube_le`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_lower`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_upper`
- CORROBORATED: `UniversalLaw.Proofs.Side24ConeMoment.Aux.cm_sq_add_sq_pos`
- CORROBORATED: `UniversalLaw.Proofs.Side24ConeMoment.Aux.cm_quad_pos`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.ten_lt_taylor_sum`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.ten_lt_exp` — new
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.pow_ten_lt_exp` — new
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.exp_neg_lt` — new
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.big_lt_exp` — new
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.ratio_const_lt_half` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Elementary.Aux.pow_succ_le`
- CORROBORATED: `UniversalLaw.Proofs.Side24Elementary.Aux.exp_succ_le` — new
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.sqrt_three_eighths`
- CORROBORATED: `UniversalLaw.Proofs.Side24Images.Aux.ib_poly_gauss_le` — new

**`frontiers/source_smooth_realization_20261011`**
- CORROBORATED: `I4Weights.frequency_l1_le_product`
- CORROBORATED: `I4Weights.sourceA_pos` — new

**`frontiers/determinant_moment_20261025`**
- CORROBORATED: `ResearchFormalCoreR1.ec005_fold_gap`
- CORROBORATED: `ResearchFormalCoreR1.ec008_factor_expansion`
- CORROBORATED: `ResearchFormalCoreR1.ec010_generic`
- CORROBORATED: `ResearchFormalCoreR1.ec010_transverse`
- CORROBORATED: `ResearchFormalCoreR1.ec011_scalar_cancellation`
- CORROBORATED: `ResearchFormalCoreR1.ec014_contact_power`
- CORROBORATED: `ResearchFormalCoreR1.dm2_scale`
- CORROBORATED: `ResearchFormalCoreR1.dm2_factors`
- CORROBORATED: `ResearchFormalCoreR1.dm2_untyped`
- CORROBORATED: `ResearchFormalCoreR1.egm_coordinate_bound`
- CORROBORATED: `ResearchFormalCoreR1.egm_factor_bound`
- CORROBORATED: `ResearchFormalCoreR1.egm_untyped`
- CORROBORATED: `ResearchFormalCoreR1.enb_maximum_factor`
- CORROBORATED: `ResearchFormalCoreR1.enb_saddle_iff`
- CORROBORATED: `ResearchFormalCoreR1.gm8_finite_radius_power_bound`
- CORROBORATED: `ResearchFormalCoreR1.gm8_sqrt_pow`
- CORROBORATED: `ResearchFormalCoreR1.nfloor_contact_value`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_r2_cancel`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_cross_multiplied`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_quotient_bound`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_bad_event_threshold`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_power40_identity`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_r4_le_r3`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_palm_r4_to_r3`

**`frontiers/compact_morse_elder_adapter_20261018`**
- CORROBORATED: `TorusFiltration.elder_control_birth`
- CORROBORATED: `TorusFiltration.elder_control_death`

**`frontiers/asymmetric_elder_bridge_20261017`**
- CORROBORATED: `TorusFiltration.elder_control_birth`
- CORROBORATED: `TorusFiltration.elder_control_death`

**`companions/torus_filtration_20261011`**
- PASS: `TorusFiltration.ordinaryH0Control_parabola_le`

**`reviews/w3_side24_source_proposal_20261008`**
- CORROBORATED (WEAK): `W3Scratch.srcE_sq`
- CORROBORATED (WEAK): `W3Scratch.srcQ_mem_Icc`
- CORROBORATED: `W3Scratch.moments_of_cumulants`
- CORROBORATED: `W3Scratch.cumulants_of_moments`
- CORROBORATED: `W3Scratch.gaussian_cumulants_iff`
- CORROBORATED: `W3Scratch.k4_sq_div`
- CORROBORATED: `W3Scratch.Side24.sq_pos_of_ne`
- CORROBORATED: `W3Scratch.Side24.weight_pos` — new
- CORROBORATED: `W3Scratch.Side24.abs_sub_sq_le` — new
- CORROBORATED: `W3Scratch.Side24.cosD_term_bound` — new

**`companions/d2_square_schur_20261006`**
- CORROBORATED: `D2SquareSchur.weighted_square_identity`
- CORROBORATED: `D2SquareSchur.weighted_square_lower`

**`companions/d2_moment_bridge_20261006`**
- CORROBORATED: `D2MomentBridge.residual_sq_expand`
- CORROBORATED: `D2MomentBridge.cubicResidual_eq_zero_iff`
- CORROBORATED: `D2MomentBridge.square_gap_integral`

**`frontiers/cap_first_exit_lean_20261002`**
- CORROBORATED: `CapFirstExit.sp_curved_side_constants`
- CORROBORATED: `CapFirstExit.l12_constants`
- CORROBORATED: `CapFirstExit.l9_arith`
- PASS: `CapFirstExit.toy_f_M`
- PASS: `CapFirstExit.toy_f_S`

## Reproduce

From a checkout of main at this PR's head, with the Math- archive at `dfa9b30a` extracted to `<scratch>/math`
(`git archive dfa9b30a formal frontiers companions specifications reviews`), run
`CENSUS_SCRATCH=<scratch> bash cbz/evidence/g12-statement-fuzz-01n/census_math_dfa9b30a_20261106_trans/run_census.sh`;
the 16 reports are byte-identical to these after replacing the local path in `lean_dir` by
`<git archive dfa9b30a>`. `SHA256SUMS` covers every file here.
