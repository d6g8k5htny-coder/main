# G12 census of every Lean declaration in Math- at `dfa9b30a` (4 November 2026)

Unit `CBZ0-G12-CENSUS-01N-20261104`. Performer: Anthropic Claude, Claude Code session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N). Credit 0. Scientific effect NONE.

**What this is.** The empirical statement check (`tools/cbz/statement_fuzz.py` as landed by main#726, #742 and #743, plus the census mode, the reader repairs and the review repairs of this PR) run in `--lean-dir` mode over every `.lean` file of every Lean group in Math- at `dfa9b30a27ea8f8ee647bb067549fa53c7d06ef6` (the tip after Math-#536), so that every `theorem`/`lemma` is a target whether or not a manifest registers it. Raw evidence snapshots (`/raw`, `raw-` directories: 1,739 files of producer runs) are excluded; byte-identical files within a group are evaluated once. The owner's addendum rule — no closure marked CLOSED without the empirical layer — needs this picture before the index is written: per declaration, the layer either **applies** (PASS / CORROBORATED / FAIL) or **does not** (UNSUPPORTED, with the construct named). Nothing here is kernel evidence or a CLOSED decision.

**Totals.** 3782 declarations in 16 groups: **7 PASS · 128 CORROBORATED · 0 FAIL · 3647 UNSUPPORTED**; 2 of the corroborations are marked WEAK (fewer than a quarter of the requested instances satisfied their hypotheses — `W3Scratch.srcE_sq` with `c ^ 2 + s ^ 2 = 1` is hit at four points only). Zero FAIL across the 135 decided statements, kernel-proved groups included, is the reader-soundness check; every UNSUPPORTED names what stopped it.

**Two reader defects this census found, fixed before these receipts were taken.** The first run corroborated `Measurable (sourceA L)` (with `sourceA : ℝ → ℝ → ℝ` a function on the reals) and passed `Integrable (fun z => |z| ^ 8) (gaussianReal 0 1)`: structural predicates whose arguments the finite model does not carry were taken as true. Now a conclusion that is only a structural predicate is UNSUPPORTED for every statement (not only when a model binder is present), a structural predicate holds only for a table, a subset, a lambda on the carrier or a model measure, and a function-valued definition is applied to its extra arguments or reported as a value the interpreter does not carry. 42 verdicts of the first run were withdrawn to UNSUPPORTED by these repairs; regression tests cover both cases.

**Three reader defects Codex's review of #752 found (752/4471516437, 4471516451, 4471516464), fixed before these receipts were taken.** (1) A function returned by a definition (`def f : ℕ → ℝ := fun n => …`) bound its parameters at the codomain type, so `n - 1` was real subtraction where Lean truncates; parameters now take the domain types of the arrow type (an explicit binder type wins) and arrow types are split at bracket depth 0 only (`Set (Fin d → ℤ)` is not a function type). (2) A group's definitions were pooled by short name, first module wins; they are now kept by Lean-qualified name and resolved from the statement's namespace outward and through its `open`s — the module's own first, then the group's when exactly one definition (up to identical text) carries the name, UNSUPPORTED "ambiguous definition" otherwise. `namespace`/`section`/`end` are tracked as scopes, so declaration names are the ones Lean gives them. (3) The splitter recognised declaration keywords at column 0 only; it now also starts a declaration at a keyword that opens its line at any indentation or follows the `in` of an `open … in` / `set_option … in` prefix, and any `theorem`/`lemma` keyword it still cannot separate (a declaration beginning mid-line) is recorded as an UNSUPPORTED target "declaration not reached by the parser", so every declaration in a file is accounted for. Exposure in this corpus, measured with the tool as it was: 11 short-name collisions across modules (10 in `source_derivative_gaussian`, 1 in `lean_statements`) touched **no** decided statement; **no** decided statement applied a returned function; **no** declaration was unreached. Re-running everything: **0 of 3,782 verdicts changed**; 695 targets carry their proper qualified names (the cap packet's `CapHyp.*` are `CapFirstExit.CapHyp.*`, which is why every one of the 367 records of the #695 index now binds by exact name, below); 14 UNSUPPORTED reasons are more specific (a definition written with its namespace prefix is now found and its unmodelled result type named, instead of "unknown function"). The 12 groups without such names are byte-identical to the first run.

| Group | Disposition context (from the hub, 1–4 Nov) | Lean files | Declarations | PASS | CORROBORATED | FAIL | UNSUPPORTED |
|---|---|---:|---:|---:|---:|---:|---:|
| `formal` | registered package (`ResearchFormalCoreR1`), Mathlib real analysis and measure theory | 9 (0 copies) | 63 | 3 | 56 | 0 | 4 |
| `frontiers/source_derivative_gaussian_20261025` | landed via Math-#514; raw evidence snapshots excluded | 114 (10 copies) | 2225 | 0 | 33 | 0 | 2192 |
| `frontiers/source_lifetime_realization_20261017` | C291: CLOSED candidate on its own receipts (Claude 54, T5) | 19 (0 copies) | 182 | 0 | 0 | 0 | 182 |
| `specifications/lean_statement_proofs_20261011` | specification proofs | 15 (0 copies) | 559 | 1 | 6 | 0 | 552 |
| `frontiers/source_smooth_realization_20261011` | CLOSED in the catalog | 14 (0 copies) | 122 | 0 | 0 | 0 | 122 |
| `frontiers/determinant_moment_20261025` | landed via Math-#536 (determinant) | 14 (0 copies) | 85 | 0 | 18 | 0 | 67 |
| `frontiers/compact_morse_elder_adapter_20261018` | vendored copies of other groups (T5: not an obligation) | 11 (0 copies) | 121 | 0 | 0 | 0 | 121 |
| `frontiers/regular_superlevel_geometry_20261020` | QP-C54-T5-004: quarantine pending kernel replay | 9 (0 copies) | 23 | 0 | 0 | 0 | 23 |
| `frontiers/asymmetric_elder_bridge_20261017` | QP-C54-T5-002: quarantine pending kernel replay | 8 (0 copies) | 31 | 0 | 0 | 0 | 31 |
| `companions/torus_filtration_20261011` | QP-C54-T5-001: quarantine pending kernel replay | 7 (0 copies) | 130 | 1 | 0 | 0 | 129 |
| `specifications/lean_statements_20261010` | statements as `Prop`-valued definitions; nothing is proved, so there is no target | 6 (0 copies) | 0 | 0 | 0 | 0 | 0 |
| `reviews/w3_side24_source_proposal_20261008` | review material (`W3Scratch`) | 3 (0 copies) | 77 | 0 | 7 | 0 | 70 |
| `companions/d2_square_schur_20261006` | companion | 2 (0 copies) | 4 | 0 | 2 | 0 | 2 |
| `companions/d2_atom_integral_20261006` | companion | 2 (0 copies) | 6 | 0 | 0 | 0 | 6 |
| `companions/d2_moment_bridge_20261006` | companion | 1 (0 copies) | 16 | 0 | 3 | 0 | 13 |
| `frontiers/cap_first_exit_lean_20261002` | QP-C54-T5-003: quarantine pending kernel replay (topology; only its real-arithmetic lemmas are reachable) | 1 (0 copies) | 138 | 2 | 3 | 0 | 133 |

## Why UNSUPPORTED (classes, all groups)

| Class | Declarations |
|---|---:|
| binder outside the interpreter | 2604 |
| proposition outside the interpreter | 205 |
| unparsed statement | 135 |
| unknown name | 94 |
| Mathlib notation | 92 |
| existential quantifier | 91 |
| definition result type outside the interpreter | 85 |
| unknown function | 82 |
| structural conclusion | 77 |
| transcendental function | 49 |
| definition outside the interpreter | 43 |
| other | 39 |
| hypothesis outside the interpreter | 31 |
| unbounded quantifier inside the statement | 13 |
| ascription to an unsupported type | 4 |
| application outside the interpreter | 3 |

## The binder types the interpreter does not model (top 40)

Structures of the Gaussian-field and Morse-theoretic tracks (`SmoothSample`, `RealizedData`, `ElderTrap`, `E d`, `SourceMode`), finite-dimensional vectors (`Fin n → ℝ`, `Fin d → ℤ`), functions on ℝ, and instance binders. A finite-model backend for `Fin n → ℝ` / `Fin n → ℤ` vectors with `∑` sums is the next extension with measurable payoff (several hundred binders); statements over the field structures are beyond sampling and belong in the index as *G12 not applicable, construct named*.

| Binder type | Count |
|---|---:|
| `SmoothSample 2 L` | 299 |
| `Fin 2 → ℝ` | 139 |
| `E d` | 87 |
| `IsUnit ( sourcePinCovarianceAt L r )` | 84 |
| `Fin 6 → ℝ` | 72 |
| `RealizedData X` | 70 |
| `ℝ → ℝ` | 56 |
| `Fact` | 54 |
| `superlevelBar_E d → ℝ` | 54 |
| `Fin d → ℤ` | 53 |
| `( Fin 2 → ℝ ) → ℝ` | 47 |
| `SourceMode 2 → ℝ` | 39 |
| `Fin d → ℝ` | 37 |
| `ElderTrap g M P m d D` | 36 |
| `Fin 6` | 35 |
| `ℝ × ℝ → ℝ` | 34 |
| `SmoothSample d L` | 34 |
| `SourceMode 2` | 33 |
| `SourceMode d` | 31 |
| `RawPinFreeFrame L r n` | 31 |
| `C` | 31 |
| `Fin 8 → ℝ` | 30 |
| `TopCat . { w }` | 30 |
| `Fin 2` | 29 |
| `Summable a` | 26 |
| `Point2` | 26 |
| `SourceMode d → ℝ` | 25 |
| `NormedAddCommGroup` | 24 |
| `E 2` | 23 |
| `E d → ℝ` | 22 |
| `Ω → Fin n → ℝ` | 19 |
| `E 2 → ℝ` | 17 |
| `ℝ × ℝ` | 16 |
| `Fin 4` | 16 |
| `ℝ ≥ 0` | 16 |
| `ι` | 16 |
| `torus_E d → ℝ` | 16 |
| `TrigPoly` | 15 |
| `SmoothParameter` | 15 |
| `Fin m → E d` | 15 |

## The decided statements

Every PASS/CORROBORATED statement by group (the receipt JSON has detail, seed and counts):

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
- CORROBORATED: `I4SourceActualQuotientSaddleCollar.sc64_weighted_positive`
- CORROBORATED: `I4SourceActualSaddleUppercuts.ug65_square_mono`
- CORROBORATED: `I4SourceActualSaddleUppercuts.ug65_square_of_abs_le`
- CORROBORATED: `SourceCompactFoldWeightedProbability.sourceCompactCutoffA_nonneg`
- CORROBORATED: `SourceCompactFoldWeightedProbability.varying_cutoff_failure_inclusion`
- CORROBORATED: `SourceFoldWeightedProbability.sourceScale_pos`
- CORROBORATED: `SourceFoldWeightedProbability.sourceCutoffEps_pos`
- CORROBORATED: `SourceFoldWeightedProbability.sourceCutoffA_nonneg`
- CORROBORATED: `SourceFoldWeightedProbability.cutoff_failure_inclusion`
- CORROBORATED: `I4SourceFramedConfluentCovariance.average_abs_le`
- CORROBORATED: `I4SourceIntrinsicH0WeightedProbability.planar_det_neg_of_signs`
- CORROBORATED: `I4SourcePinCurvatureBounds.endpoint_det_abs_le`
- CORROBORATED: `I4SourcePinCurvatureBounds.curvature_slab_of_upper`
- CORROBORATED: `I4ActualQuarticTransfer.corrected_axial_constant`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_raw_regularity`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_scaled_error_budget`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_adaptive_tolerance`
- CORROBORATED: `I4SourcePinPreservingQuartic.cutoff_curvature_threshold`
- CORROBORATED: `I4SourcePinPreservingQuartic.certificate_width_geometry`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_three_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_two_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_add_six_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_add_five_le`
- CORROBORATED: `I4QuarticCaptureAlgebra.strip_coordinate_bounds`
- CORROBORATED: `I4QuarticCaptureAlgebra.abs_mul_four_le`
- CORROBORATED: `ResearchFormalCoreR1.dm2_scale`
- CORROBORATED: `ResearchFormalCoreR1.dm2_factors`
- CORROBORATED: `ResearchFormalCoreR1.egm_factor_bound`
- CORROBORATED: `ResearchFormalCoreR1.enb_maximum_factor`
- CORROBORATED: `ResearchFormalCoreR1.gm8_sqrt_pow`

**`specifications/lean_statement_proofs_20261011`**
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.quarter_pow_identity`
- CORROBORATED: `UniversalLaw.Proofs.P15Budget.Aux.quarter_ratio_le`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_lower`
- CORROBORATED: `UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_upper`
- CORROBORATED: `UniversalLaw.Proofs.Side24ConeMoment.Aux.cm_quad_pos`
- CORROBORATED: `UniversalLaw.Proofs.Side24Elementary.Aux.pow_succ_le`
- PASS: `UniversalLaw.Proofs.Side24Elementary.Aux.sqrt_three_eighths`

**`frontiers/determinant_moment_20261025`**
- CORROBORATED: `ResearchFormalCoreR1.ec005_fold_gap`
- CORROBORATED: `ResearchFormalCoreR1.ec008_factor_expansion`
- CORROBORATED: `ResearchFormalCoreR1.ec010_generic`
- CORROBORATED: `ResearchFormalCoreR1.ec010_transverse`
- CORROBORATED: `ResearchFormalCoreR1.ec011_scalar_cancellation`
- CORROBORATED: `ResearchFormalCoreR1.ec014_contact_power`
- CORROBORATED: `ResearchFormalCoreR1.dm2_scale`
- CORROBORATED: `ResearchFormalCoreR1.dm2_factors`
- CORROBORATED: `ResearchFormalCoreR1.egm_factor_bound`
- CORROBORATED: `ResearchFormalCoreR1.enb_maximum_factor`
- CORROBORATED: `ResearchFormalCoreR1.gm8_sqrt_pow`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_r2_cancel`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_cross_multiplied`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm008_quotient_bound`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_bad_event_threshold`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_power40_identity`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_r4_le_r3`
- CORROBORATED: `ResearchFormalCoreR1.p02_lm009_palm_r4_to_r3`

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

```sh
mkdir -p /tmp/g12-census/math && git -C <Math- clone> archive dfa9b30a27ea8f8ee647bb067549fa53c7d06ef6 frontiers companions specifications reviews formal | tar -x -C /tmp/g12-census/math
bash cbz/evidence/g12-statement-fuzz-01n/census_math_dfa9b30a_20261104/run_census.sh      # from a main checkout; about 12 s
```
Each `<group>.json` is the full report (`per_target` with detail, `copies`, `file_sha256` of every Lean file read, `unsupported_reasons`); each `.stdout.txt` is the CLI summary with every non-PASS line. `SHA256SUMS` binds them. Seeds and sample counts are the defaults (1, 64); the run is deterministic.

## Per-record annotation of the literal kernel index (main#695)

`g12_annotations_literal_index_82d9e480.json` binds each of the **367** CLOSED records of `cbz/evidence/kernel/all-closed-declarations.json` (main#695 at `82d9e480`) to a target of this census or of the main `formal/` receipt `main_formal_ec6962ff_r4.json`, by the sha256 of the source file the record names **and** the declaration's qualified name — every record binds (0 unbound): **34 PASS · 69 CORROBORATED · 264 not applicable · 0 FAIL**.

| Index family | PASS | CORROBORATED | not applicable |
|---|---:|---:|---:|
| `Math-:c297` | 0 | 0 | 13 |
| `Math-:formal` | 3 | 56 | 4 |
| `Math-:frontiers/determinant_moment_20261025` | 0 | 13 | 53 |
| `Math-:smooth` | 0 | 0 | 116 |
| `Math-:spec` | 0 | 0 | 78 |
| `main:formal` | 31 | 0 | 0 |

The 264 *not applicable* records name their construct: binder outside the interpreter 156, unknown name 57, proposition outside the interpreter 16, unparsed statement 11, structural conclusion 9, definition outside the interpreter 5, definition result type outside the interpreter 5, unbounded quantifier inside the statement 5. Each record carries the receipt file and its sha256, the target key, instances, and for sampled verdicts the seed and accepted/rejected/infeasible counts; the proposed row text is `G12: PASS | CORROBORATED | not applicable (<construct>)`. This is an annotation for Root's index, not a CLOSED decision.

## What the census says for the index (proposal, for Root)

1. For the groups under CLOSED decisions — `source_smooth_realization_20261011`, the C291 candidate `source_lifetime_realization_20261017`, main `formal/` and the P15 rows, Math- `formal/`, the Math-#514 and Math-#536 packages — the applicable empirical verdict is now on record per declaration: PASS/CORROBORATED where the interpreter reaches the statement, and *not applicable* with the construct named where it does not. The addendum rule is satisfiable only through that explicit path: a CLOSED row carries `G12: PASS | CORROBORATED | not applicable (<construct>)`, never a silent gap, and UNSUPPORTED is never upgraded.
2. The two CLOSED-track Gaussian-field packages (`source_smooth_realization`, `source_lifetime_realization`) have **no** declaration the empirical layer reaches: every statement quantifies over `SmoothSample`, `RealizedData` or similar structures. Their rows will all read *not applicable*; that is the honest state, and the index should say so rather than let the rule be read as satisfied.
3. No statement in any group was refuted. A FAIL in a future run is a transcription or statement defect to look at first, not kernel news.

Limits as in the parent README: CORROBORATED is bounded by 64 exact instances; the census mode has no manifest and is coverage evidence, never a gate (its exit code is 1 unless everything is PASS).
