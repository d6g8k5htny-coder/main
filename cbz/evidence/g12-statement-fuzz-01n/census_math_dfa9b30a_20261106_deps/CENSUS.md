# G12 census of Math- at `dfa9b30a` with dependency definitions (`--defs-from`; 6 November 2026)

**What this is.** The census of [`census_math_dfa9b30a_20261106_trans/`](../census_math_dfa9b30a_20261106_trans/CENSUS.md)
(the transcendental backend, main#757) re-taken with `tools/cbz/statement_fuzz.py --lean-dir <group> --defs-from <dependency>…`:
each group now pools the definitions, carrier types and type abbreviations of the groups its `import` lines name, transitively
(the table below is derived from the import lines of the 16 groups; `run_census.sh` carries it), under the same resolution
rule as before — a statement's own module first, then the pool when exactly one definition carries the qualified name,
otherwise UNSUPPORTED "ambiguous definition". A dependency's theorems are never targets; every `import` of a target module is
accounted for (group file, dependency file, or a library prefix) and the unresolved ones are listed in each report
(`unresolved_imports`: modules the archive does not carry, such as `Lifetime`, `RegularPointChart`). Two resource caps were
added with it, both Infeasible (counted, never a verdict, never a float): a `∑`/`∏` over more than 4,096 indices, and an
instance whose exact results exceed 20,000,000 bits in total (`binomTail 10^6 65535 c`-style sums of binomials; the baseline
census re-run with the caps alone is identical to the stored receipts per target). Same source archive, same 16 groups,
seed 1, 64 samples; `verdict_changes_20261106_trans_to_deps.json` lists every verdict (and every WEAK flag) that moved.

**Result: 3,782 declarations — 66 PASS · 273 CORROBORATED · 0 FAIL · 3,443 UNSUPPORTED**
(transcendental census: 40 · 233 · 0 · 3,509). **66 verdicts moved, 66 of them from
UNSUPPORTED**: 26 to PASS and 40 to CORROBORATED; **0 to FAIL** — every one of these statements is kernel-proved, so a FAIL
would have been a reader defect. No previously decided verdict changed. By group: `specifications/lean_statement_proofs_20261011` 48, `reviews/w3_side24_source_proposal_20261008` 13, `companions/d2_moment_bridge_20261006` 3, `companions/d2_square_schur_20261006` 2.
The gains are the statements about the specification's constants and functions — `pStar = 1 − e⁻¹`, `hStar`, `rhoStar`,
`epsilon = 10⁻¹⁰⁸`, `phi`, `binomTail`, `pairingSum`, `imageConstant`, the `d2Schur`/`d2DetV` family — which the proofs
groups import from `specifications/lean_statements_20261010` and `formal/` and which the census could not see before.
6 CORROBORATED verdicts carry the WEAK flag (5 before).

| Group | Declarations | PASS | CORROBORATED | FAIL | UNSUPPORTED | transcendental census (P · C · U) |
|---|---:|---:|---:|---:|---:|---|
| `formal` | 63 | 3 | 56 | 0 | 4 | 3 · 56 · 4 |
| `frontiers/source_derivative_gaussian_20261025` | 2225 | 9 | 104 | 0 | 2112 | 9 · 104 · 2112 |
| `frontiers/source_lifetime_realization_20261017` | 182 | 0 | 4 | 0 | 178 | 0 · 4 · 178 |
| `specifications/lean_statement_proofs_20261011` | 559 | 49 | 44 | 0 | 466 | 25 · 20 · 514 |
| `frontiers/source_smooth_realization_20261011` | 122 | 0 | 3 | 0 | 119 | 0 · 3 · 119 |
| `frontiers/determinant_moment_20261025` | 85 | 0 | 24 | 0 | 61 | 0 · 24 · 61 |
| `frontiers/compact_morse_elder_adapter_20261018` | 121 | 0 | 2 | 0 | 119 | 0 · 2 · 119 |
| `frontiers/regular_superlevel_geometry_20261020` | 23 | 0 | 0 | 0 | 23 | 0 · 0 · 23 |
| `frontiers/asymmetric_elder_bridge_20261017` | 31 | 0 | 2 | 0 | 29 | 0 · 2 · 29 |
| `companions/torus_filtration_20261011` | 130 | 1 | 0 | 0 | 129 | 1 · 0 · 129 |
| `specifications/lean_statements_20261010` | 0 | 0 | 0 | 0 | 0 | 0 · 0 · 0 |
| `reviews/w3_side24_source_proposal_20261008` | 77 | 1 | 22 | 0 | 54 | 0 · 10 · 67 |
| `companions/d2_square_schur_20261006` | 4 | 0 | 4 | 0 | 0 | 0 · 2 · 2 |
| `companions/d2_atom_integral_20261006` | 6 | 0 | 0 | 0 | 6 | 0 · 0 · 6 |
| `companions/d2_moment_bridge_20261006` | 16 | 1 | 5 | 0 | 10 | 0 · 3 · 13 |
| `frontiers/cap_first_exit_lean_20261002` | 138 | 2 | 3 | 0 | 133 | 2 · 3 · 133 |

## Dependency definitions pooled per group (from the groups' `import` lines, transitive)

| Group | `--defs-from` |
|---|---|
| `formal` | — |
| `frontiers/source_derivative_gaussian_20261025` | `companions/torus_filtration_20261011`, `frontiers/asymmetric_elder_bridge_20261017`, `frontiers/determinant_moment_20261025`, `frontiers/source_lifetime_realization_20261017`, `frontiers/source_smooth_realization_20261011`, `specifications/lean_statements_20261010` |
| `frontiers/source_lifetime_realization_20261017` | `frontiers/source_smooth_realization_20261011` |
| `specifications/lean_statement_proofs_20261011` | `specifications/lean_statements_20261010` |
| `frontiers/source_smooth_realization_20261011` | — |
| `frontiers/determinant_moment_20261025` | — |
| `frontiers/compact_morse_elder_adapter_20261018` | `companions/torus_filtration_20261011`, `frontiers/asymmetric_elder_bridge_20261017`, `frontiers/source_lifetime_realization_20261017`, `frontiers/source_smooth_realization_20261011`, `specifications/lean_statements_20261010` |
| `frontiers/regular_superlevel_geometry_20261020` | — |
| `frontiers/asymmetric_elder_bridge_20261017` | `companions/torus_filtration_20261011`, `specifications/lean_statements_20261010` |
| `companions/torus_filtration_20261011` | `specifications/lean_statements_20261010` |
| `specifications/lean_statements_20261010` | — |
| `reviews/w3_side24_source_proposal_20261008` | `formal` |
| `companions/d2_square_schur_20261006` | `formal` |
| `companions/d2_atom_integral_20261006` | `companions/d2_moment_bridge_20261006`, `companions/d2_square_schur_20261006`, `formal` |
| `companions/d2_moment_bridge_20261006` | `formal` |
| `frontiers/cap_first_exit_lean_20261002` | — |


## Why UNSUPPORTED (classes, all groups)

| Class | Declarations | transcendental census |
|---|---:|---:|
| binder outside the interpreter | 2177 | 2177 |
| definition result type outside the interpreter | 251 | 197 |
| proposition outside the interpreter | 203 | 215 |
| unparsed statement | 153 | 153 |
| other | 110 | 106 |
| definition outside the interpreter | 108 | 72 |
| existential quantifier | 97 | 97 |
| structural conclusion | 92 | 92 |
| Mathlib notation | 71 | 71 |
| unknown function | 63 | 138 |
| hypothesis outside the interpreter | 51 | 51 |
| enclosures do not decide the statement | 23 | 21 |
| unbounded quantifier inside the statement | 12 | 12 |
| unknown name | 10 | 91 |
| real exponent (Real.rpow) | 8 | 3 |
| quantifier over an unsupported type | 5 | 4 |
| ascription to an unsupported type | 4 | 4 |
| application outside the interpreter | 3 | 3 |
| no sample satisfied the hypotheses | 2 | 2 |

The remaining binder types are structures and function spaces (`SmoothSample 2 L`, `E d`, `IsUnit (…)`,
`RealizedData X`, `ℝ → ℝ`, `superlevelBar_E d → ℝ`, `(Fin 2 → ℝ) → ℝ`, `SourceMode 2` …), existential quantifiers,
structural conclusions and `∃`/`∑'`/`*ᵥ` notation — none is a target of this tool yet. `unknown name` / `unknown function`
now hold only names the archive does not define anywhere in the group's dependency closure (modules the archive lacks,
`fderiv`, `iteratedDeriv`, `Metric.ball`, structure projections …).

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
- CORROBORATED: `I4GaussianMajorant.exp_mul_abs_le`
- CORROBORATED: `I4SourceActualFoldCandidate.planar_quadratic_complete_square`
- CORROBORATED: `SourceActualFoldH0.fold_interval_endpoints`
- CORROBORATED: `I4SourceActualMorseLocalModel.point2_decomp`
- PASS: `I4SourceActualMorseLocalModel.fixture_scale_ne`
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
- CORROBORATED: `SourceCompactGapNormalizer.gaussianCompactLowerConstant_pos`
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_one_sixth`
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_eq_confluent`
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_norm_le`
- CORROBORATED: `I4SourceConfluentLimits.average_norm_le`
- CORROBORATED: `I4SourceConfluentPins.rawTransform_confluentTransform`
- CORROBORATED: `I4SourceConfluentPins.confluentTransform_rawTransform`
- CORROBORATED: `I4SourceEndpointEstimates.sourcePhysicalCoordinates_inverse`
- CORROBORATED: `I4SourceEndpointHessianWitness.sourceC_ne_zero`
- CORROBORATED: `I4SourceEndpointHessianWitness.pinMultiplier_value`
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
- CORROBORATED: `I4ActualSourceMorse.sourceC_ne_zero`
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
- CORROBORATED: `I4SourceRemoteJetWitness.annihilator_pin_values`
- CORROBORATED: `I4SourceRemoteJetWitness.sourceC_ne_zero`
- CORROBORATED: `I4SourceTwoRemoteFirstJetWitness.sourceC_ne_zero`
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
- CORROBORATED: `I4Weights.productDecay_eq_exp`
- CORROBORATED: `I4Weights.frequency_l1_le_product`
- CORROBORATED: `I4Weights.sourceA_pos`
- CORROBORATED: `TorusFiltration.superlevel_mono`
- CORROBORATED: `TorusFiltration.superlevel_neg_height`
- CORROBORATED: `TorusFiltration.elder_control_birth`
- CORROBORATED: `TorusFiltration.elder_control_death`

**`frontiers/source_lifetime_realization_20261017`**
- CORROBORATED: `I4ThetaBridge.shifted_gaussian_le`
- CORROBORATED: `I4Weights.productDecay_eq_exp`
- CORROBORATED: `I4Weights.frequency_l1_le_product`
- CORROBORATED: `I4Weights.sourceA_pos`

**`specifications/lean_statement_proofs_20261011`**
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.le_cappedHazard_of` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.phi_le_div` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.quarter_pow_identity`
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.quarter_ratio_le`
- PASS: `UniversalLaw.Proofs.P15Budget.Aux.log_benchmark_le`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.two_lt_exp_one`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.three_exp_sub_two_pos`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.exp_two_eq`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.exp_neg_two_eq`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.exp_neg_three_eq`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.log_three_exp_sub_two_lt_two`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.one_lt_hStar` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.taylor_sum_eq`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.exp_eleven_sixths_gt`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.three_exp_sub_two_lt`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.log_three_exp_sub_two_lt`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.seven_sixths_lt_hStar` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.rhoStar_lt_six_sevenths` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.pStar_pos` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.pStar_gt_half` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.pStar_lt_one` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Elementary.Aux.binomTail_succ_left` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Elementary.Aux.binomTail_antitone_step` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Elementary.Aux.binomTail_le_of_le` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Elementary.Aux.binomTail_pos` — new
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.binomTail_three_one` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Elementary.Aux.binomTail_odd_chain` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Hazard.Aux.neg_log_one_sub_ge`
- CORROBORATED: `UniversalLaw.Proofs.P15Hazard.Aux.le_phi_self` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_exp_one_gt`
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_exp_one_lt`
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_exp_xU_ge`
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_exp_xL_le`
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_log_lt`
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_lt_log`
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_hStar_pos`
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_rhoStar_lt_upper` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_lower_lt_rhoStar` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f3_upper_lt_six_sevenths` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.cc_affineExp_pos`
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.f6_one_sub_exp_neg_mem`
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.f7_phi_nonneg` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.f7_phi_le_one` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.f7_phi_mem` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.f7_one_sub_exp_neg_phi_nonneg` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.f8_prod_phi_le_one` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Interpolation.Aux.bi_prod_phi_nonneg` — new
- PASS: `UniversalLaw.Proofs.P15Interpolation.Aux.f2_rhoStar_nonneg` — new
- CORROBORATED: `UniversalLaw.Proofs.P15Structure.Aux.neg_log_one_sub_gt`
- CORROBORATED: `UniversalLaw.Proofs.RNCluster.Aux.rc_poissonPMF_zero` — new
- CORROBORATED: `UniversalLaw.Proofs.RNCluster.Aux.rc_poissonPMF_one` — new
- CORROBORATED: `UniversalLaw.Proofs.RNCluster.Aux.rc_half_le_one_sub_exp`
- CORROBORATED: `UniversalLaw.Proofs.RNCluster.Aux.rc_poisson_tail_two_le_sq`
- CORROBORATED: `UniversalLaw.Proofs.RNProbability.Aux.one_le_log_one_div`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.d1_pairingSum_le_of_one_le` — new
- CORROBORATED (WEAK): `UniversalLaw.Proofs.Side24Analysis.Aux.d0_pairingSum_zero_le` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_epsilon_pos` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_epsilon_lt` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_one_sub_epsilon_pos` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_one_le_one_add_epsilon` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_lower_rational` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_upper_rational` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentA_two` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentB_two` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentA_three` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentB_three` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_geom_tail`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_pow6_succ_le`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_exp_succ_le`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_prod_exp` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_sum_cube_le`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_imageTerm_nonneg` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_imageTerm_le` — new
- PASS: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_one_add_epsilon_pos` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_lower`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_upper`
- CORROBORATED: `UniversalLaw.Proofs.Side24ConeMoment.Aux.cm_sq_add_sq_pos`
- CORROBORATED: `UniversalLaw.Proofs.Side24ConeMoment.Aux.cm_quad_pos`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.ten_lt_taylor_sum`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.ten_lt_exp`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.pow_ten_lt_exp`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.exp_neg_lt`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.big_lt_exp`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.ratio_const_lt_half`
- CORROBORATED: `UniversalLaw.Proofs.Side24Elementary.Aux.pow_succ_le`
- CORROBORATED: `UniversalLaw.Proofs.Side24Elementary.Aux.exp_succ_le`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.sqrt_three_eighths`
- CORROBORATED: `UniversalLaw.Proofs.Side24Images.Aux.ib_pairingSum_mono` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Images.Aux.ib_pairingSum_le` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Images.Aux.ib_poly_gauss_le`
- CORROBORATED: `UniversalLaw.Proofs.Side24Images.Aux.ib_intNormSq_ge_one` — new
- PASS: `UniversalLaw.Proofs.Side24Images.Aux.ib_final_constant` — new
- PASS: `UniversalLaw.Proofs.Side24Images.Aux.co_margin` — new

**`frontiers/source_smooth_realization_20261011`**
- CORROBORATED: `I4Weights.productDecay_eq_exp`
- CORROBORATED: `I4Weights.frequency_l1_le_product`
- CORROBORATED: `I4Weights.sourceA_pos`

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
- CORROBORATED: `W3Scratch.d2Alpha_eq_src` — new
- CORROBORATED: `W3Scratch.d2Gamma_eq_src` — new
- CORROBORATED (WEAK): `W3Scratch.srcE_sq`
- CORROBORATED: `W3Scratch.d2DetV_eq_alpha_gamma_q` — new
- CORROBORATED: `W3Scratch.d2DetV_eq_srcDet` — new
- CORROBORATED: `W3Scratch.d2DetV_eq_alpha_gamma_e` — new
- CORROBORATED: `W3Scratch.d2DetV_eq_det_varV` — new
- CORROBORATED: `W3Scratch.d2DetV_eq_note_linear` — new
- PASS: `W3Scratch.d2DetV_eq_alpha_gamma_e_needs_unit` — new
- CORROBORATED (WEAK): `W3Scratch.srcQ_mem_Icc`
- CORROBORATED: `W3Scratch.moments_of_cumulants`
- CORROBORATED: `W3Scratch.cumulants_of_moments`
- CORROBORATED: `W3Scratch.gaussian_cumulants_iff`
- CORROBORATED: `W3Scratch.k4_sq_div`
- CORROBORATED: `W3Scratch.d2Tau_eq_srcTau` — new
- CORROBORATED: `W3Scratch.d2Schur_eq_srcSchur` — new
- CORROBORATED: `W3Scratch.d2Numerator_eq_comment` — new
- CORROBORATED: `W3Scratch.src_delta_cumulant` — new
- CORROBORATED: `W3Scratch.note_section2_every_direction` — new
- CORROBORATED: `W3Scratch.Side24.sq_pos_of_ne`
- CORROBORATED: `W3Scratch.Side24.weight_pos`
- CORROBORATED: `W3Scratch.Side24.abs_sub_sq_le`
- CORROBORATED: `W3Scratch.Side24.cosD_term_bound`

**`companions/d2_square_schur_20261006`**
- CORROBORATED: `D2SquareSchur.weighted_square_identity`
- CORROBORATED: `D2SquareSchur.weighted_square_lower`
- CORROBORATED: `D2SquareSchur.schur_no_upper` — new
- CORROBORATED: `D2SquareSchur.schur_from_lower_inputs` — new

**`companions/d2_moment_bridge_20261006`**
- CORROBORATED: `D2MomentBridge.residual_sq_expand`
- CORROBORATED: `D2MomentBridge.residual_integral_eq_delta` — new
- CORROBORATED: `D2MomentBridge.delta_nonneg` — new
- CORROBORATED: `D2MomentBridge.cubicResidual_eq_zero_iff`
- CORROBORATED: `D2MomentBridge.square_gap_integral`
- PASS: `D2MomentBridge.zero_atom_counterexample` — new

**`frontiers/cap_first_exit_lean_20261002`**
- CORROBORATED: `CapFirstExit.sp_curved_side_constants`
- CORROBORATED: `CapFirstExit.l12_constants`
- CORROBORATED: `CapFirstExit.l9_arith`
- PASS: `CapFirstExit.toy_f_M`
- PASS: `CapFirstExit.toy_f_S`

## Reproduce

From a checkout of main at this PR's head, with the Math- archive at `dfa9b30a` extracted to `<scratch>/math`
(`git archive dfa9b30a formal frontiers companions specifications reviews`), run
`CENSUS_SCRATCH=<scratch> bash cbz/evidence/g12-statement-fuzz-01n/census_math_dfa9b30a_20261106_deps/run_census.sh`;
the 16 reports are byte-identical to these after replacing the local path in `lean_dir` by
`<git archive dfa9b30a>`. `SHA256SUMS` covers every file here.
