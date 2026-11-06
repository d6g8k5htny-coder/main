# G12 census of Math- at `dfa9b30a` with the finite-vector backend (6 November 2026)

**What this is.** The census of [`census_math_dfa9b30a_20261104/`](../census_math_dfa9b30a_20261104/CENSUS.md) re-taken
with `tools/cbz/statement_fuzz.py` extended by the **finite-vector backend**: binders `(j : Fin n → ℝ)`,
`(i : Fin n)`, `(M : Matrix (Fin m) (Fin n) ℝ)`, `EuclideanSpace ℝ (Fin n)` and type abbreviations of them are
sampled as exact vectors, indices and matrices; `![…]` / `!![…]` literals, `∑`/`∏` over finite index sets,
`.det`, `‖·‖` (sup norm on the Pi type, Euclidean norm on `EuclideanSpace`), `Pi.single`, `dotProduct` /
`⬝ᵥ`, `•`, pointwise `+ - * /`, the Pi order, `↑i` / `(i : ℕ)` casts and implicit dimension parameters
(`def gm8Radius {n : ℕ} (j : Fin n → ℝ)`) are read exactly. Same source archive, same 16 groups, same
command (`run_census.sh`, seed 1, 64 samples); `verdict_changes_20261104_to_20261106.json` lists every
verdict that moved.

**Result: 3,782 declarations — 17 PASS · 203 CORROBORATED · 0 FAIL · 3562 UNSUPPORTED**
(4 November: 7 · 128 · 0 · 3647). **85 verdicts moved, all from UNSUPPORTED**: 75 to CORROBORATED and
10 to PASS; **none to FAIL** — every one of these statements is kernel-proved, so a FAIL would have been a reader
defect. No verdict that was decided on 4 November changed. By group: `frontiers/source_derivative_gaussian_20261025` 69, `frontiers/determinant_moment_20261025` 6, `specifications/lean_statement_proofs_20261011` 4, `frontiers/compact_morse_elder_adapter_20261018` 2, `frontiers/asymmetric_elder_bridge_20261017` 2, `frontiers/source_lifetime_realization_20261017` 1, `frontiers/source_smooth_realization_20261011` 1.
Among the gains: the determinant-moment package's `egm_coordinate_bound`, `egm_untyped`, `dm2_untyped`, `enb_saddle_iff`,
`gm8_finite_radius_power_bound` (implicit `{n}`), `nfloor_contact_value`; the quartic-capture algebra
(`rawRadius_*`, `captureWidth_*`, `quartic_*_remainder_bound` with a `∀ i, |U i| ≤ R` hypothesis over `Fin 8`);
Euclidean-norm identities (`euclidean_two_norm_sq`, `…_le_abs_sum`), `Pi.single` decompositions
(`plane_decompose`, `point2_decomp`, `coordinateDirection_zero`), `frequency_l1_le_product` (`1 + ∑ v i ≤ ∏ (1 + v i)`),
`li_sum_cube_le` (`(∑ a i)³ ≤ 9 ∑ a i³` for `d ≤ 3`), the exact Taylor sums `taylor_sum_eq` and `ten_lt_taylor_sum`
(`∑ j ∈ Finset.range 6, (11/6)^j / j! = 197/32 + 26081/933120`, decided in ℚ) and the elder-interval propositions
(`elderActive`, a Prop-valued definition now inlined). 5 CORROBORATED verdicts carry the WEAK flag (fewer than a
quarter of the requested instances; e.g. `ct58_lower_inverse`, whose hypotheses `e 0² + e 1² = 1 ∧ …` a sampler
rarely meets).

| Group | Declarations | PASS | CORROBORATED | FAIL | UNSUPPORTED | 4 Nov (P · C · U) |
|---|---:|---:|---:|---:|---:|---|
| `formal` | 63 | 3 | 56 | 0 | 4 | 3 · 56 · 4 |
| `frontiers/source_derivative_gaussian_20261025` | 2225 | 8 | 94 | 0 | 2123 | 0 · 33 · 2192 |
| `frontiers/source_lifetime_realization_20261017` | 182 | 0 | 1 | 0 | 181 | 0 · 0 · 182 |
| `specifications/lean_statement_proofs_20261011` | 559 | 3 | 8 | 0 | 548 | 1 · 6 · 552 |
| `frontiers/source_smooth_realization_20261011` | 122 | 0 | 1 | 0 | 121 | 0 · 0 · 122 |
| `frontiers/determinant_moment_20261025` | 85 | 0 | 24 | 0 | 61 | 0 · 18 · 67 |
| `frontiers/compact_morse_elder_adapter_20261018` | 121 | 0 | 2 | 0 | 119 | 0 · 0 · 121 |
| `frontiers/regular_superlevel_geometry_20261020` | 23 | 0 | 0 | 0 | 23 | 0 · 0 · 23 |
| `frontiers/asymmetric_elder_bridge_20261017` | 31 | 0 | 2 | 0 | 29 | 0 · 0 · 31 |
| `companions/torus_filtration_20261011` | 130 | 1 | 0 | 0 | 129 | 1 · 0 · 129 |
| `specifications/lean_statements_20261010` | 0 | 0 | 0 | 0 | 0 | 0 · 0 · 0 |
| `reviews/w3_side24_source_proposal_20261008` | 77 | 0 | 7 | 0 | 70 | 0 · 7 · 70 |
| `companions/d2_square_schur_20261006` | 4 | 0 | 2 | 0 | 2 | 0 · 2 · 2 |
| `companions/d2_atom_integral_20261006` | 6 | 0 | 0 | 0 | 6 | 0 · 0 · 6 |
| `companions/d2_moment_bridge_20261006` | 16 | 0 | 3 | 0 | 13 | 0 · 3 · 13 |
| `frontiers/cap_first_exit_lean_20261002` | 138 | 2 | 3 | 0 | 133 | 2 · 3 · 133 |

## What the vector backend reads, and what stays outside

A vector binder is sampled from structured values (the zero vector, the ones, unit vectors, the pool along the
diagonal, alternating signs, constants), hypothesis-boundary components (`v 0 ∈ Set.Ioo a b` gives a, b and
their ±1 / ±10⁻⁶ neighbours for component 0) and random pool/rational entries; an index binder from
0 … n-1; a ℕ binder that is the dimension of a later vector binder from the small pool {0, 1, 2, 3, 4} (a
`Fin 0` index has no value, so those samples are rejected, never counted). Big operators bind as Mathlib's
`r:67` (`∑ i, f i ≤ 1` is `(∑ i, f i) ≤ 1`, `∑ i, a i + b i` is `(∑ i, a i) + b i`); their index set comes
from the binder type, a `Finset.range k` domain (a ℕ index) or `Finset.univ`, or the vectors the index applies
to, and is otherwise UNSUPPORTED. Arithmetic in `Fin n` wraps in Lean (`(2 : Fin 3) + 1 = 0`) and is refused,
as is a numeral at `Fin n` that is not below n; `(i : ℕ)` / `↑i` arithmetic is read. Products of vectors,
matrix norms, `*ᵥ` / `ᵥ*`, `Matrix.of`, orders on `EuclideanSpace`, determinants above 6×6 (Infeasible) and
`∑'` (tsum) stay UNSUPPORTED, named.

## Why UNSUPPORTED (classes, all groups)

| Class | Declarations |
|---|---:|
| binder outside the interpreter | 2177 |
| proposition outside the interpreter | 215 |
| definition result type outside the interpreter | 197 |
| unparsed statement | 153 |
| unknown function | 134 |
| unknown name | 116 |
| other | 105 |
| existential quantifier | 97 |
| structural conclusion | 92 |
| Mathlib notation | 71 |
| definition outside the interpreter | 71 |
| transcendental function | 60 |
| hypothesis outside the interpreter | 51 |
| unbounded quantifier inside the statement | 12 |
| quantifier over an unsupported type | 4 |
| application outside the interpreter | 3 |
| ascription to an unsupported type | 2 |
| no sample satisfied the hypotheses | 2 |

The remaining binder types are structures and function spaces (`SmoothSample 2 L` 322, `E d` 90,
`IsUnit (…)` 87, `RealizedData X` 70, `ℝ → ℝ` 56, `superlevelBar_E d → ℝ` 54, `(Fin 2 → ℝ) → ℝ` 51,
`SourceMode 2` 47 …), the transcendental functions (`Real.exp` 38, `π` 23), existential quantifiers (97),
structural conclusions (92) and `∃`/`∑'`/`*ᵥ` notation — none is a sampling target of this backend.

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
- CORROBORATED: `I4SourceActualFoldCandidate.planar_quadratic_complete_square`
- CORROBORATED: `SourceActualFoldH0.fold_interval_endpoints` — new
- CORROBORATED: `I4SourceActualMorseLocalModel.point2_decomp` — new
- CORROBORATED: `I4SourceActualMorseLocalModel.fixture_point_sq_pos` — new
- CORROBORATED: `I4SourceActualQuotientSaddleCollar.sc64_weighted_positive`
- CORROBORATED: `I4SourceActualQuotientSaddleCollar.sc64_positive_cone_iff` — new
- CORROBORATED: `I4SourceActualQuotientSaddleCollar.sc64_negative_cone_iff` — new
- CORROBORATED: `I4SourceActualSaddleUppercuts.ug65_square_mono`
- CORROBORATED: `I4SourceActualSaddleUppercuts.ug65_square_of_abs_le`
- CORROBORATED: `I4SourceActualSaddleUppercuts.ug65_reflect_involutive` — new
- CORROBORATED: `I4SourceAdaptiveFoldChart.chart_endpointM` — new
- CORROBORATED: `I4SourceAdaptiveFoldChart.chart_endpointS` — new
- CORROBORATED: `SourceCompactFoldWeightedProbability.sourceCompactCutoffA_nonneg`
- CORROBORATED: `SourceCompactFoldWeightedProbability.varying_cutoff_failure_inclusion`
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_one_sixth` — new
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_eq_confluent` — new
- CORROBORATED: `I4SourceCompactTargetMoments.sourceGapPins_norm_le` — new
- CORROBORATED: `I4SourceConfluentLimits.average_norm_le` — new
- CORROBORATED: `I4SourceConfluentPins.rawTransform_confluentTransform` — new
- CORROBORATED: `I4SourceConfluentPins.confluentTransform_rawTransform` — new
- CORROBORATED: `I4SourceEndpointEstimates.sourcePhysicalCoordinates_inverse` — new
- CORROBORATED: `SourceFoldWeightedProbability.sourceScale_pos`
- CORROBORATED: `SourceFoldWeightedProbability.sourceCutoffEps_pos`
- CORROBORATED: `SourceFoldWeightedProbability.sourceCutoffA_nonneg`
- CORROBORATED: `SourceFoldWeightedProbability.cutoff_failure_inclusion`
- CORROBORATED: `I4SourceFramedConfluentCovariance.average_abs_le`
- CORROBORATED (WEAK): `I4SourceFramedConfluentCovariance.ct58_lower_inverse` — new
- CORROBORATED (WEAK): `I4SourceFramedConfluentCovariance.ct58_frame_delta_four` — new
- CORROBORATED: `I4SourceFramedConfluentCovariance.compact_quad_smul` — new
- CORROBORATED: `I4SourceFramedConfluentCovariance.compact_square_nonneg` — new
- CORROBORATED: `I4SourceFramedConfluentCovariance.compact_square_eq_zero` — new
- PASS: `I4SourceGapConditionalMorseDistinct.source_directionX_eq` — new
- PASS: `I4SourceGapConditionalMorseDistinct.source_directionY_eq` — new
- CORROBORATED: `I4SourceGapConditionalMorseDistinct.constraintSquare_pos_iff` — new
- CORROBORATED: `I4SourceIntrinsicH0WeightedProbability.planar_det_neg_of_signs`
- PASS: `I4ActualSourceMorse.morseDirectionX_eq` — new
- PASS: `I4ActualSourceMorse.morseDirectionY_eq` — new
- PASS: `I4SourcePinCurvatureBounds.directionX_norm_le_one` — new
- PASS: `I4SourcePinCurvatureBounds.directionY_norm_le_one` — new
- CORROBORATED: `I4SourcePinCurvatureBounds.axis_point_sub_norm_le` — new
- CORROBORATED: `I4SourcePinCurvatureBounds.endpoint_det_abs_le`
- CORROBORATED: `I4SourcePinCurvatureBounds.curvature_slab_of_upper`
- CORROBORATED: `I4ActualQuarticTransfer.corrected_axial_constant`
- CORROBORATED: `I4SourcePinPreservingQuartic.euclidean_two_norm_sq` — new
- CORROBORATED: `I4SourcePinPreservingQuartic.euclidean_two_norm_le_abs_sum` — new
- CORROBORATED: `I4SourcePinPreservingQuartic.euclidean_two_norm_le_two_mul` — new
- PASS: `I4SourcePinPreservingQuartic.coordinateDirection_zero` — new
- PASS: `I4SourcePinPreservingQuartic.coordinateDirection_one` — new
- CORROBORATED: `I4SourcePinPreservingQuartic.coordinate_decomposition` — new
- CORROBORATED: `I4SourcePinPreservingQuartic.one_le_quarticFstar` — new
- CORROBORATED: `I4SourcePinPreservingQuartic.quarticWidth_pos` — new
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_raw_regularity`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_scaled_error_budget`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_adaptive_tolerance`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_quarticWidth_tolerance` — new
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_curvature_threshold`
- CORROBORATED: `I4SourcePinPreservingQuartic.certificate_width_geometry`
- CORROBORATED: `I4SourcePinPreservingQuartic.plane_decompose` — new
- CORROBORATED (WEAK): `I4SourcePinPreservingQuartic.coordinateDirection_norm_le_one` — new
- CORROBORATED: `I4SourcePinPreservingQuartic.quartic_strip_in_ball_two` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_one_le` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_pos` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_sum_abs_eq` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_coord_abs_le_sub_one` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.rawRadius_coord_abs_le` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_pos` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.captureFstar_le_radius` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_radius_le` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_le` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_le_one_sixteenth` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.captureWidth_radius_le_one_sixtyfourth` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.captureSigma_pos` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.captureSigma_le` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.four_captureWidth_le_captureSigma` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_three_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_two_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_add_six_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_add_five_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.strip_coordinate_bounds`
- CORROBORATED: `I4QuarticCaptureAlgebra.quartic_transverse_forcing_bound` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_four_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.quartic_transverse_remainder_bound` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.quartic_axial_remainder_bound` — new
- CORROBORATED: `I4QuarticCaptureAlgebra.width_identity` — new
- CORROBORATED: `ResearchFormalCoreR1.dm2_scale`
- CORROBORATED: `ResearchFormalCoreR1.dm2_factors`
- CORROBORATED: `ResearchFormalCoreR1.dm2_untyped` — new
- CORROBORATED: `ResearchFormalCoreR1.egm_coordinate_bound` — new
- CORROBORATED: `ResearchFormalCoreR1.egm_factor_bound`
- CORROBORATED: `ResearchFormalCoreR1.egm_untyped` — new
- CORROBORATED: `ResearchFormalCoreR1.enb_maximum_factor`
- CORROBORATED: `ResearchFormalCoreR1.enb_saddle_iff` — new
- CORROBORATED: `ResearchFormalCoreR1.gm8_finite_radius_power_bound` — new
- CORROBORATED: `ResearchFormalCoreR1.gm8_sqrt_pow`
- CORROBORATED: `ResearchFormalCoreR1.nfloor_contact_value` — new
- CORROBORATED: `I4Weights.frequency_l1_le_product` — new
- CORROBORATED: `TorusFiltration.superlevel_mono` — new
- CORROBORATED: `TorusFiltration.superlevel_neg_height` — new
- CORROBORATED: `TorusFiltration.elder_control_birth` — new
- CORROBORATED: `TorusFiltration.elder_control_death` — new

**`frontiers/source_lifetime_realization_20261017`**
- CORROBORATED: `I4Weights.frequency_l1_le_product` — new

**`specifications/lean_statement_proofs_20261011`**
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.quarter_pow_identity`
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.quarter_ratio_le`
- PASS: `UniversalLaw.Proofs.P15Elementary.Aux.taylor_sum_eq` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.li_sum_cube_le` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_lower`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_upper`
- CORROBORATED: `UniversalLaw.Proofs.Side24ConeMoment.Aux.cm_sq_add_sq_pos` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24ConeMoment.Aux.cm_quad_pos`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.ten_lt_taylor_sum` — new
- CORROBORATED: `UniversalLaw.Proofs.Side24Elementary.Aux.pow_succ_le`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.sqrt_three_eighths`

**`frontiers/source_smooth_realization_20261011`**
- CORROBORATED: `I4Weights.frequency_l1_le_product` — new

**`frontiers/determinant_moment_20261025`**
- CORROBORATED: `ResearchFormalCoreR1.ec005_fold_gap`
- CORROBORATED: `ResearchFormalCoreR1.ec008_factor_expansion`
- CORROBORATED: `ResearchFormalCoreR1.ec010_generic`
- CORROBORATED: `ResearchFormalCoreR1.ec010_transverse`
- CORROBORATED: `ResearchFormalCoreR1.ec011_scalar_cancellation`
- CORROBORATED: `ResearchFormalCoreR1.ec014_contact_power`
- CORROBORATED: `ResearchFormalCoreR1.dm2_scale`
- CORROBORATED: `ResearchFormalCoreR1.dm2_factors`
- CORROBORATED: `ResearchFormalCoreR1.dm2_untyped` — new
- CORROBORATED: `ResearchFormalCoreR1.egm_coordinate_bound` — new
- CORROBORATED: `ResearchFormalCoreR1.egm_factor_bound`
- CORROBORATED: `ResearchFormalCoreR1.egm_untyped` — new
- CORROBORATED: `ResearchFormalCoreR1.enb_maximum_factor`
- CORROBORATED: `ResearchFormalCoreR1.enb_saddle_iff` — new
- CORROBORATED: `ResearchFormalCoreR1.gm8_finite_radius_power_bound` — new
- CORROBORATED: `ResearchFormalCoreR1.gm8_sqrt_pow`
- CORROBORATED: `ResearchFormalCoreR1.nfloor_contact_value` — new
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_r2_cancel`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_cross_multiplied`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_quotient_bound`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_bad_event_threshold`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_power40_identity`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_r4_le_r3`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_palm_r4_to_r3`

**`frontiers/compact_morse_elder_adapter_20261018`**
- CORROBORATED: `TorusFiltration.elder_control_birth` — new
- CORROBORATED: `TorusFiltration.elder_control_death` — new

**`frontiers/asymmetric_elder_bridge_20261017`**
- CORROBORATED: `TorusFiltration.elder_control_birth` — new
- CORROBORATED: `TorusFiltration.elder_control_death` — new

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
`CENSUS_SCRATCH=<scratch> bash cbz/evidence/g12-statement-fuzz-01n/census_math_dfa9b30a_20261106/run_census.sh`;
the 16 reports are byte-identical to these after replacing the local path in `lean_dir` by
`<git archive dfa9b30a>`. `SHA256SUMS` covers every file here.
