# Ledger MISSING source-cell recheck — 26 October 2026

> **NOT A RESULT.** Engineering source accounting only. Scientific effect **NONE**. This record does not edit the
> ledger, RETRO-MATH-BIND-01, any GB-8 file, the graph, the v2 register or any count; the ledger owner decides adoption.
> "Source document present" means a file in the unit states mathematics. It is not a claim that the mathematics is correct.

## Pins

- Ledger: `main:data/hypergraph/math_coverage_ledger_20261023.md` at `99adffc8902b1fbfb7ab487020f9c0267f28f88d`, blob `03fe9db3f21d22b4d1b8db409a9c6d887763ec64`.
- Unit cuts: Math- `9f3fb6918b28e6378c034ebc19e76d432f5391cb`, main `a487cfd41bfe095842d9d9f5cec26fc23877b028` (the cuts the ledger rows were built on).
- Post-cut carriers: landed Math- merges up to `b29c4b769cac85437b409d152651d5cce4140015`.
- Pickup: [619/6405600087](https://github.com/d6g8k5htny-coder/main/issues/619#issuecomment-6405600087). Performer: Anthropic Claude, session_01S16677s1SGeAJmN3QMz4ip (Claude S).

## Result

- The ledger has **129** rows whose proof/lemma cell is `MISSING`.
- **83** of them hold at least one document that states mathematics (reader-assessed).
- **45** hold no mathematical source document; 11 of these have no statement document at all.
- **1** row is pinned to an open PR and was not checked.

Reader classes of the 218 statement documents: FORMAL_SOURCE 5, FORMAL_SOURCE_NO_DECLS 2, INDEX_ONLY 36, MATH_REVIEW 8, MATH_STATEMENT 94, OTHER 43, PROTOCOL_ONLY 30.

## Method

1. The script lists every member of each row unit at its pinned cut and takes the `.md`, `.tex`, `.txt` and `.lean` files
   as statement documents, with blob, byte size and SHA-256.
2. It gives each document a structural class from marker counts (`marker_regex` in the JSON). This first pass is
   reproducible but coarse: against the reader pass it was CORRECT 187, TOO_HIGH 15, TOO_LOW 16. ASCII mathematics (`<=`, `integral_`) pushes it too low; README link labels push it too high.
3. Model-assisted content reading (Anthropic Claude subagents of session_01S16677s1SGeAJmN3QMz4ip): five readers covered every statement document of the 129 rows, one verified the carriers, one reconciled; Claude S spot-checked rows 27, 148, 223, 263, 283. These classes are a reader judgment, not reproducible by the script, and make no correctness claim.
4. A row is SOURCE_DOCUMENT_PRESENT when at least one document is MATH_STATEMENT or MATH_REVIEW, or FORMAL_SOURCE with theorem/lemma declarations. Formal statement maps without in-file proofs are classed OTHER and do not count.

Regenerate and check: `python3 docs/retro/RETRO_SOURCE_CELL_RECHECK_20261026.py <main clone> <Math- clone> --check`.

## Rows with a mathematical source document

| Row | Unit | Mathematical source documents | Note |
|---|---|---|---|
| 4 | `frontiers/c2_exact_20261001` | `NOTE.md` `a2798b87` |  |
| 6 | `frontiers/c2_torus_transfer_20261001` | `NOTE.md` `5ce5bad2` |  |
| 7 | `frontiers/c6_cluster_coefficients_numerics_20260930` | `NOTE.md` `845881d4` | uncertified numerics with exact identities |
| 14 | `frontiers/c6_microscopic_tail_numerics_20260930` | `NOTE.md` `f5224e0b` | uncertified numerics with exact identities |
| 17 | `frontiers/c6_remote_coefficient_d3_numerics_20260930` | `NOTE.md` `98a14c82` | uncertified numerics with exact identities |
| 18 | `frontiers/c6_remote_coefficient_numerics_20260930` | `NOTE.md` `47fc0fed` | uncertified numerics with exact identities |
| 23 | `frontiers/c8_cap_failure_coefficient_dims_20261001` | `NOTE.md` `b42be5b8` |  |
| 24 | `frontiers/c8_cap_failure_coefficient_planar_20260930` | `NOTE.md` `2838b76b` |  |
| 25 | `frontiers/c8_elder_failure_coefficient_20261001` | `NOTE.md` `9903dd6f` |  |
| 26 | `frontiers/c8_lifetime_difference_constant_20261001` | `NOTE.md` `845112c9` |  |
| 27 | `frontiers/c8_normalizer_floor_planar_20260930` | `NOTE.md` `d321d52e` |  |
| 28 | `frontiers/c8_theorem_a_constants_d3_20261001` | `NOTE.md` `718dd17e` |  |
| 29 | `frontiers/c8_theorem_a_constants_planar_20260930` | `NOTE.md` `49ebecf4` |  |
| 30 | `frontiers/c8_torus_vanishing_remainder_20261001` | `NOTE.md` `8758ddf2` |  |
| 31 | `frontiers/c8_window_coefficient_d3_torus_20260930` | `NOTE.md` `549dc7cc` |  |
| 32 | `frontiers/c8_window_coefficient_d4_20261001` | `NOTE.md` `d0961238` |  |
| 33 | `frontiers/c8_window_coefficient_d5_20261001` | `NOTE.md` `8399c6ba` |  |
| 34 | `frontiers/c8_window_coefficient_d6_20261001` | `NOTE.md` `654c0aca` |  |
| 35 | `frontiers/c8_window_coefficient_d7_20261001` | `NOTE.md` `66193992` |  |
| 36 | `frontiers/c8_window_coefficient_planar_20260930` | `NOTE.md` `8d73ae38` |  |
| 43 | `frontiers/contact_kernel_tail_20260925` | `NOTE.md` `8da0cc85` |  |
| 59 | `frontiers/equal_height_mass_value_20261001` | `NOTE.md` `95d622c8` |  |
| 78 | `frontiers/local_pairing_20260930` | `LOCAL_PAIRING.md` `1f8c9455` |  |
| 82 | `frontiers/ordinary_short_bar_occurrence_all_d_20261001` | `NOTE.md` `a7031f68` |  |
| 122 | `frontiers/third_order_coefficient_20261001` | `NOTE.md` `fa448934` | formal claim, not proved for d>=2 |
| 124 | `frontiers/three_fronts_20260924` | `LIFETIME_REMAINDER.md` `247b3ecf`; `P15_PRICE_BOUNDARY.md` `1d694665`; `P15_REALIZED_COVERS.md` `17388191`; `RN_COUNT_INTERFACE.md` `371fd6d1` |  |
| 131 | `frontiers/window_multiplicity_laws_20260928` | `ELDER_LOWER_AND_DENSITY_GAP.md` `aaefc8da`; `HEIGHT_MARKS.md` `015d7696`; `LOCAL_MULTIPLICITY.md` `b5c18aac`; `REMOTE_DISTANCE_MOMENTS.md` `d077743b`; `REMOTE_PAIR_LAW.md` `3fb60204` |  |
| 133 | `reviews/c6_fourier_completion_20260929` | `READING_NOTE.md` `682d7045` | reading note |
| 136 | `reviews/c6_tail_constant_certified_20260930` | `README.md` `5f96b3d3` |  |
| 137 | `reviews/c6_witness_collision_reconciliation_20260929` | `RECONCILIATION.md` `efd6aa65` | reconciliation; no new mathematical claim |
| 139 | `reviews/c7_total_k_scaling_recon_20260929` | `RETRACTION.md` `e96888c2` | retraction; the proof is in frontiers/c7_total_bounded_20260929/PROOF.md, outside the unit |
| 141 | `reviews/candidates_pending_20260928` | `CANDIDATES.md` `02455f05` | candidate catalog |
| 143 | `reviews/collision_mechanism_20260925` | `CUMULATIVE_TRANSFER_CORRECTION.md` `044ac5fd`; `NOTE.md` `3ee30829` |  |
| 144 | `reviews/contact_angular_tail_20260925` | `NOTE.md` `df008a07` |  |
| 145 | `reviews/contact_kernel_substitute_20260926` | `SUBSTITUTE.md` `ae3b8e1f` |  |
| 146 | `reviews/contact_kernel_tail_20260925` | `ANNULUS_ASYMPTOTIC_BRIDGE.md` `1ab83027`; `KERNEL_TAILS_AND_SMALL_GAP.md` `4d03f110` |  |
| 153 | `reviews/d1_external_audit_20260930` | `AUDIT.md` `fdfccaf5` |  |
| 158 | `reviews/d2_joint_mass_schur_20261006` | `NOTE.md` `19640662` |  |
| 160 | `reviews/d2_sharp_scalar_floor_20261006` | `NOTE.md` `8741b0c8` |  |
| 161 | `reviews/d2_two_atom_floors_20261006` | `NOTE.md` `cedbb44a` |  |
| 162 | `reviews/d2_two_mass_floors_20261006` | `NOTE.md` `b3e3b337` |  |
| 166 | `reviews/d5_consumption_bridge_20260928` | `BRIDGE.md` `7e1ef1c1`; `DOMAIN_CORRECTION.md` `62cf830a` | consumption bridge |
| 174 | `reviews/d5_microdisk_20260926` | `NOTE.md` `19efd64e`; `QUARTIC.md` `16e29976` |  |
| 176 | `reviews/d5_pin_disk_all_directions_20260928` | `NOTE.md` `c9a8ec2f` |  |
| 177 | `reviews/d5_pin_microdisk_20260927` | `NOTE.md` `ac09361c` |  |
| 181 | `reviews/d5_reconciliation_20260929` | `RECONCILIATION.md` `4062d76b` | reconciliation; no new mathematical claim |
| 183 | `reviews/d5_short_edge_20260928` | `NOTE.md` `e5fde88b` |  |
| 185 | `reviews/downstream_boundary_20260925` | `TRANSVERSE_BOUND_CANDIDATE.md` `024d9277` |  |
| 190 | `reviews/iba1_periodic_jet_claude_20261005` | `NOTE.md` `e36e341f` | uncertified numerics with exact identities |
| 191 | `reviews/iba2_009_matching_20261005` | `REPORT.md` `a0b398f5` |  |
| 195 | `reviews/iba2_012_localized_moment_20261006` | `NOTE.md` `9dd7d583` |  |
| 196 | `reviews/iba2_012_log_cap_concentration_20261006` | `NOTE.md` `e37ea10e` |  |
| 197 | `reviews/iba2_012_log_cap_transfer_20261006` | `NOTE.md` `b158965f` |  |
| 198 | `reviews/iba2_012_marked_sector_20261006` | `NOTE.md` `aaa51260` |  |
| 199 | `reviews/lean_cycle2_20261005` | `MomentTail_finite_MT_N2.lean.txt` `7c9278ba`; `NOTES_MT_N2.md` `a408e765`; `MomentTail_family_interval_MT_N1.lean.txt` `739a7c40`; `NOTES_MT_N1.md` `20ecb346` | uncompiled Lean proposals |
| 204 | `reviews/literature_classical_neighbors_20261001` | `ADDENDUM.md` `b289c7cd` |  |
| 206 | `reviews/lm006_common_regression_rate_20261006` | `NOTE.md` `3242441f` |  |
| 207 | `reviews/lm006_cubic_sign_band_20261006` | `NOTE.md` `88dae06c` |  |
| 208 | `reviews/lm006_endpoint_extension_20261005` | `NOTE.md` `551aa563` |  |
| 209 | `reviews/lm006_normalizer_stability_20261005` | `NOTE.md` `5790aaa0` |  |
| 210 | `reviews/lm006_positive_radius_density_20261006` | `NOTE.md` `a6d5209a` |  |
| 211 | `reviews/lm006_product_transverse_20261006` | `NOTE.md` `1532dad3` |  |
| 212 | `reviews/lm006_quantified_regression_20261005` | `NOTE.md` `a58948d2` |  |
| 213 | `reviews/lm006_tilted_endpoint_law_20261005` | `NOTE.md` `6633cdf0` |  |
| 214 | `reviews/lm006_typed_sign_suppression_20261005` | `NOTE.md` `9015971f` |  |
| 215 | `reviews/lm006_wrong_sign_asymptotic_20261006` | `NOTE.md` `ea914c1b` |  |
| 221 | `reviews/periodic_tau_envelope_20261005` | `REPORT.md` `2166869f` |  |
| 222 | `reviews/periodic_tau_positive_floor_20261005` | `NOTE.md` `50d7ea35` |  |
| 223 | `reviews/pin_micro_covariance_20260926` | `NOTE.md` `36cc6cfc` |  |
| 225 | `reviews/pin_neighborhood_recon_20260926` | `NOTE.md` `c83640bc` | reconnaissance-level result |
| 251 | `reviews/side24_d2_certified_small_L_claude_20261005` | `NOTE.md` `852f7403` |  |
| 256 | `imports/hardening_ebedb780` | `proof.md.export.txt` `7189b4d1`; `proof.md.export.txt` `7bcafa24`; `proof.md.export.txt` `fce94e3b`; `proof.md.export.txt` `de9995e3`; `proof.md.export.txt` `d1fd7ecc`; `proof.md.export.txt` `8fa13431`; `proof.md.export.txt` `f895c56c`; `proof.md.export.txt` `70112651`; `P15-B_PALETTE_SEPARATED_LOCALIZATION.md` `deab15eb`; `ISSUE_BODY_TRANSCRIPTION.md` `8ea2464e` | EC-014, EC-015 and P02-LM-001 keep their derivations in external Drive files |
| 262 | `coefficients/lm006_wrong_sign_20261006_r20` | `NOTE.md` `a79f0808` |  |
| 263 | `coefficients/side24_d4_20260930` | `NOTE.md` `e00de466` |  |
| 264 | `coefficients/side24_d4_closed_form_20260930` | `NOTE.md` `8c97ea36` |  |
| 265 | `coefficients/side24_d5_20260930` | `NOTE.md` `7d12f050` |  |
| 266 | `coefficients/side24_d5_closed_form_20260930` | `NOTE.md` `42cb7672` | conditional on the unmerged Math-#200 |
| 267 | `coefficients/side24_d6_20260930` | `NOTE.md` `e150b851` |  |
| 285 | `formal/MOMENT_TAIL.md` | `MOMENT_TAIL.md` `d45f5259` |  |
| 291 | `formal/WEIGHT_PERTURBATION.md` | `WEIGHT_PERTURBATION.md` `dbd204fb` |  |
| 292 | `formal/blueprint` | `content.tex` `2afd3e89` |  |
| 299 | `formal/originals` | `Algebra.lean.txt` `4554c810`; `ProbabilityCompanions.lean.txt` `82aa4208` |  |
| 301 | `repairs/c026_reciprocal_20260927` | `README.md` `f359e163` |  |

## Rows without a mathematical source document

| Row | Unit | Documents present (reader class) | Note |
|---|---|---|---|
| 1 | `frontiers/README.md` | `README.md` INDEX_ONLY |  |
| 51 | `frontiers/downstream_gate_20260925` | `D0_CI_UNBLOCK.md` PROTOCOL_ONLY; `README.md` PROTOCOL_ONLY; `SCOPE.md` PROTOCOL_ONLY |  |
| 134 | `reviews/c6_numerics_protocol_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 140 | `reviews/c7_total_protocol_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 147 | `reviews/cusp_workflow_protocol_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 148 | `reviews/d0_custody_audit_20260926` | `AUDIT.md` PROTOCOL_ONLY |  |
| 163 | `reviews/d3_l4_round2_audit_20261005` | `REPLAY_SCOPE.md` PROTOCOL_ONLY |  |
| 164 | `reviews/d5_c6_workflow_repair_20261005` | `REPLAY_SCOPE.md` PROTOCOL_ONLY |  |
| 169 | `reviews/d5_graph_fold_20260929` | `README.md` INDEX_ONLY |  |
| 184 | `reviews/direct_import_trigger_coverage_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 186 | `reviews/drive_hole_substitutes_20260926` | `LEDGER.md` PROTOCOL_ONLY |  |
| 188 | `reviews/equal_height_protocol_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 189 | `reviews/fixed_r_protocol_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 192 | `reviews/iba2_010_interface_ledger_20261005` | `LEDGER.md` INDEX_ONLY |  |
| 201 | `reviews/legacy_json_workflow_repair_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 202 | `reviews/legacy_trigger_coverage_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 203 | `reviews/legacy_tty_diagnostic_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 237 | `reviews/register_execution_readiness_20260930` | `README.md` PROTOCOL_ONLY |  |
| 238 | `reviews/remainder_rate_protocol_20261006` | `SCOPE.md` PROTOCOL_ONLY |  |
| 255 | `imports/gp_data_114_embedded_20260930` | `README.md` PROTOCOL_ONLY |  |
| 272 | `certificates/bf_source_integrity.py` | none |  |
| 273 | `certificates/test_bf_source_integrity.py` | none |  |
| 278 | `formal/.gitignore` | none |  |
| 279 | `formal/COMPATIBILITY.md` | `COMPATIBILITY.md` PROTOCOL_ONLY |  |
| 280 | `formal/D2_SCHUR.md` | `D2_SCHUR.md` OTHER | formal statement map (Lean declarations, no in-file proof); proofs live in the formal module |
| 281 | `formal/GLOSSARY.md` | `GLOSSARY.md` INDEX_ONLY |  |
| 282 | `formal/LINEAGE_VALIDATION.md` | `LINEAGE_VALIDATION.md` PROTOCOL_ONLY |  |
| 283 | `formal/MEASURE_BRIDGE.md` | `MEASURE_BRIDGE.md` OTHER | formal statement map; proofs in formal/ResearchFormalCoreR1/MeasureBridge.lean |
| 284 | `formal/MOMENT_GENERALITY.md` | `MOMENT_GENERALITY.md` OTHER | formal statement map; proofs in formal/ResearchFormalCoreR1/MomentGenerality.lean |
| 286 | `formal/README.md` | `README.md` INDEX_ONLY |  |
| 288 | `formal/ResearchFormalCoreR1.lean` | `ResearchFormalCoreR1.lean` FORMAL_SOURCE_NO_DECLS | import-only root module, no declarations |
| 289 | `formal/SCOPE.md` | `SCOPE.md` OTHER | formal statement map (SCOPE.md) |
| 290 | `formal/WEIGHTED_LAW.md` | `WEIGHTED_LAW.md` OTHER | formal statement map; proofs in formal/ResearchFormalCoreR1/WeightedLaw.lean |
| 293 | `formal/evidence` | none |  |
| 294 | `formal/gate.py` | none |  |
| 295 | `formal/lake-manifest.json` | none |  |
| 296 | `formal/lakefile.toml` | none |  |
| 297 | `formal/lean-toolchain` | none |  |
| 298 | `formal/manifest.json` | none |  |
| 300 | `formal/tests` | none |  |
| 305 | `claims/LANDING_CLAIMS.json` | none |  |
| 306 | `experiments/board_capacity_probe_20261022` | `README.md` PROTOCOL_ONLY |  |
| 308 | `experiments/reactor_v0` | `README.md` OTHER |  |
| 309 | `experiments/semantic_memory` | `TASK1-SOURCE-README.md` PROTOCOL_ONLY; `D01.txt` OTHER; `D02.txt` OTHER; `D03.txt` OTHER; `D04.txt` OTHER; `D05.txt` OTHER; … 27 more |  |
| 313 | `formal/UniversalLaw.lean` | `UniversalLaw.lean` FORMAL_SOURCE_NO_DECLS | import-only root module, no declarations |

## Landed post-cut carriers

These Math- merges landed after the cut and touch ledger rows; none of them supplies a proof or lemma source.

| Row | Path | Landing | Merge | Blob | Role | Note |
|---|---|---|---|---|---|---|
| 34 | `frontiers/c8_window_coefficient_d6_20261001/SOURCE_FILES.json` | Math-#500 | `c99ad22f` | `eed81587` | MANIFEST_REPIN | declarative manifest; one blob id expanded to 40 hex |
| 215 | `tools/lm006_sign_coefficient_replay.py` | Math-#366 | `b29c4b76` | `93d35b65` | DEPENDENCY_PIN_ONLY | pins reviews/lm006_wrong_sign_asymptotic_20261006/NOTE.md ea914c1b; runs no code of this unit |
| 257 | `imports/lifetime_parent_20260925/test_verify.py` | Math-#501 | `96633d22` | `7ebb74f3` | NEW_CARRIER | new negative-control test for verify.py |
| 257 | `imports/lifetime_parent_20260925/verify.py` | Math-#501 | `96633d22` | `ae8d4efd` | REVISED_EXISTING_CARRIER | listed in the row; revised after the cut (was 74235319) |
| 261 | `tests/test_d3_l4_replay.py` | Math-#316 | `67adf884` | `caa5cac8` | REVISED_EXISTING_CARRIER | both files existed at the cut (aed704e7, 47c538b1) but are absent from the row implementation cell |
| 261 | `tools/d3_l4_replay.py` | Math-#316 | `67adf884` | `ec9c1583` | REVISED_EXISTING_CARRIER | both files existed at the cut (aed704e7, 47c538b1) but are absent from the row implementation cell |
| 262 | `.github/workflows/lm006-sign-coefficient.yml` | Math-#366 | `b29c4b76` | `21361d33` | NEW_CARRIER | replay tool pins the unit NOTE.md a79f0808 and runs the unit enclose.py |
| 262 | `tests/test_lm006_sign_coefficient_replay.py` | Math-#366 | `b29c4b76` | `6e6d5cea` | NEW_CARRIER | replay tool pins the unit NOTE.md a79f0808 and runs the unit enclose.py |
| 262 | `tools/lm006_sign_coefficient_replay.py` | Math-#366 | `b29c4b76` | `93d35b65` | NEW_CARRIER | replay tool pins the unit NOTE.md a79f0808 and runs the unit enclose.py |

Rows 257 and 261 are not `MISSING` rows; they appear because their implementation carriers changed after the cut.

## Manifest and test re-pins (Math-#508, Math-#509)

Every non-workflow path these two merges changed, with the ledger rows whose unit contains it. A row that is not a
`MISSING` row is listed here only. None of these changes touches mathematical content.

| Rows | Path | Landing | Merge | Blob before → after |
|---|---|---|---|---|
| 274 | `companions/d2_atom_integral_20261006/SOURCE_FILES.json` | Math-#508 | `10f13851` | `c68d2d7a` → `82953007` |
| 275 | `companions/d2_moment_bridge_20261006/SOURCE_FILES.json` | Math-#508 | `10f13851` | `76b1c59d` → `b72d7de7` |
| 276 | `companions/d2_square_schur_20261006/SOURCE_FILES.json` | Math-#508 | `10f13851` | `f424d054` → `af1d447e` |
| 277 | `companions/torus_filtration_20261011/SOURCE_FILES.json` | Math-#508 | `10f13851` | `4234b2ae` → `ae2eff1f` |
| 80 | `frontiers/nongaussian_smoothing_admission_20261017/FOLLOWON_SOURCE_INDEX.json` | Math-#508 | `10f13851` | `fa21d332` → `a92ee5b0` |
| 161 | `reviews/d2_two_atom_floors_20261006/SOURCE_FILES.json` | Math-#508 | `10f13851` | `1c627ddf` → `9df8ca21` |
| 162 | `reviews/d2_two_mass_floors_20261006/SOURCES.json` | Math-#508 | `10f13851` | `9ddeae97` → `f1d079de` |
| 304 | `specifications/lean_statements_20261010/README.md` | Math-#508 | `10f13851` | `30e61717` → `2eb55662` |
| no ledger unit | `tests/test_c6_factorial_job_budget.py` | Math-#508 | `10f13851` | `671ca014` → `5531fc90` |
| no ledger unit | `tests/test_d5_reconciliation_workflow.py` | Math-#508 | `10f13851` | `b383166b` → `1790e408` |
| no ledger unit | `tests/test_elder_lower_all_d_workflow.py` | Math-#508 | `10f13851` | `5a52984b` → `9e88a0c2` |
| no ledger unit | `tests/test_elder_lower_review_workflow.py` | Math-#508 | `10f13851` | `f2524a66` → `d4388f81` |
| no ledger unit | `tests/test_full_suite_routing.py` | Math-#508 | `10f13851` | `07f05865` → `f2f29811` |
| 276 | `companions/d2_square_schur_20261006/SOURCE_FILES.json` | Math-#509 | `dd2d13b4` | `af1d447e` → `265c3517` |

The same merges changed 182 workflow files under `.github/workflows/`, listed with blobs in
`postcut_workflow_repins` in the JSON. #508 changes only `uses:` action pins; #509 changes one
`timeout-minutes` value (25 → 60) in `d2-square-schur.yml`. 110 ledger rows cite one of them in their implementation cell; for those rows:

| Workflow | Landing | Blob before → after | Ledger rows citing it |
|---|---|---|---|
| `bf-six-pin-scalar.yml` | Math-#508 | `ccba07ef` → `148494f2` | 272, 273 |
| `c2-torus-transfer.yml` | Math-#508 | `819f1046` → `e8c11afd` | 38, 57 |
| `c297-regular-superlevel.yml` | Math-#508 | `1201e225` → `c3ef2964` | 295, 296, 297 |
| `c6-cluster-coefficients-numerics.yml` | Math-#508 | `0ab0fe0e` → `05985f18` | 7 |
| `c6-cluster-law.yml` | Math-#508 | `3a09eabf` → `dc7d93aa` | 8 |
| `c6-factorial-moment.yml` | Math-#508 | `4da40eba` → `984b08dc` | 10 |
| `c6-fourier-cutoff.yml` | Math-#508 | `99045d18` → `110bfe86` | 19, 179, 181, 257 |
| `c6-hard-direction-factorization.yml` | Math-#508 | `342b09e7` → `9016a5d4` | 12 |
| `c6-marked-fourier.yml` | Math-#508 | `33ad7f26` → `4f7ccd47` | 15, 50 |
| `c6-microscopic-tail-numerics.yml` | Math-#508 | `8e57caaa` → `d7d8769c` | 14 |
| `c6-remote-coefficient-d3-numerics.yml` | Math-#508 | `6725cc6e` → `36727925` | 17, 18, 96 |
| `c6-residual-closure.yml` | Math-#508 | `c8577515` → `da5d7071` | 135 |
| `c6-tail-constant-certified.yml` | Math-#508 | `53b09857` → `9adcd4f6` | 136 |
| `c6-witness-collision-reconciliation.yml` | Math-#508 | `a176b69a` → `2a27acea` | 133, 137, 141, 175, 182, 187 |
| `c7-nonvanishing-review.yml` | Math-#508 | `9043f1d6` → `bb3167da` | 138 |
| `c7-total-bounded.yml` | Math-#508 | `a19097df` → `de4887a8` | 21 |
| `c8-cap-failure-coefficient-dims.yml` | Math-#508 | `b74f83b2` → `afb65fea` | 23 |
| `c8-cap-failure-coefficient-planar.yml` | Math-#508 | `a9c9daba` → `f4c7e791` | 24, 77 |
| `c8-lifetime-difference-constant.yml` | Math-#508 | `1f99cdab` → `ba3b9c09` | 151, 155 |
| `c8-normalizer-floor-planar.yml` | Math-#508 | `042f07de` → `c3eeb175` | 27 |
| `c8-theorem-a-constants-d3.yml` | Math-#508 | `8a1fdad2` → `c315ed12` | 28 |
| `c8-theorem-a-constants-planar.yml` | Math-#508 | `3a5ee3eb` → `85fffc79` | 29 |
| `c8-window-coefficient-d3-torus.yml` | Math-#508 | `0194bba0` → `225a2727` | 31 |
| `c8-window-coefficient-d4.yml` | Math-#508 | `b6cc2ef4` → `f179abce` | 32 |
| `c8-window-coefficient-d5.yml` | Math-#508 | `af29c8c2` → `35c7279e` | 33 |
| `c8-window-coefficient-d6.yml` | Math-#508 | `71828761` → `5eed3f30` | 34 |
| `c8-window-coefficient-d7.yml` | Math-#508 | `d2087b5b` → `482c29af` | 35 |
| `c8-window-coefficient-planar.yml` | Math-#508 | `b5061549` → `e1f49959` | 36 |
| `candidate-parity-rate.yml` | Math-#508 | `b244693c` → `024ca105` | 37, 49, 94, 95, 122, 123 |
| `cap-first-exit-lean.yml` | Math-#508 | `2ca2252f` → `2e7a0730` | 149 |
| `collision-mechanism-checks.yml` | Math-#508 | `47837be0` → `0a00f042` | 143 |
| `contact-small-gap.yml` | Math-#508 | `9cde8e04` → `86cadd90` | 185, 228 |
| `d1-cap-elder-partner-second-pass.yml` | Math-#508 | `4b688ce1` → `93623fd2` | 156 |
| `d1-cde-fulldepth-review.yml` | Math-#508 | `089df771` → `50b247c6` | 150 |
| `d1-line.yml` | Math-#508 | `68b548a3` → `a6485ec2` | 47, 48, 204 |
| `d1-third-order-law.yml` | Math-#508 | `e188a408` → `e3a8d989` | 205 |
| `d2-remainder-fulldepth-review.yml` | Math-#508 | `dbbba357` → `736a07f3` | 159 |
| `d2-two-mass-floors.yml` | Math-#508 | `5d4a4ef2` → `3e3c6099` | 251 |
| `d3-l4-anisotropic.yml` | Math-#508 | `73efbf12` → `3bf9e353` | 190 |
| `d5-intermediate-window-review.yml` | Math-#508 | `ecf5c711` → `10ede983` | 171 |
| `d5-pin-disk.yml` | Math-#508 | `70697b1d` → `8b1afd40` | 176, 224 |
| `downstream-gate.yml` | Math-#508 | `f05bbec8` → `53046a26` | 234 |
| `elder-cusp-parity.yml` | Math-#508 | `29be167a` → `d74de2a1` | 52 |
| `elder-lower-all-d.yml` | Math-#508 | `89cd0618` → `603a3bed` | 54 |
| `elder-lower-review.yml` | Math-#508 | `4a26f8b6` → `6a5b113e` | 152 |
| `elder-third-order.yml` | Math-#508 | `f2474073` → `f5a06fce` | 62 |
| `far-elder-flat-ridge.yml` | Math-#508 | `957099fb` → `a946de7b` | 129 |
| `formal-lean.yml` | Math-#508 | `1d64378f` → `b9165ec9` | 294 |
| `h5-rim-contract.yml` | Math-#508 | `b2c8f51a` → `2b63690e` | 259 |
| `hardening-proof-custody.yml` | Math-#508 | `d1dffe7e` → `aac15653` | 256 |
| `landing-claims.yml` | Math-#508 | `0516a6df` → `8eb48df6` | 229, 305 |
| `lifetime-moment-boundary.yml` | Math-#508 | `6c9fa1d8` → `7c4b75bb` | 235 |
| `local-cluster-tightness.yml` | Math-#508 | `072a5a4d` → `b1697d53` | 131 |
| `ordinary-short-bar-occurrence-all-d.yml` | Math-#508 | `1687c868` → `16ef2996` | 111 |
| `p15-overlap-tail-review.yml` | Math-#508 | `78812077` → `ca314513` | 220 |
| `pin-micro-covariance.yml` | Math-#508 | `4579bb5d` → `2274350c` | 223 |
| `price-budget.yml` | Math-#508 | `da03eca7` → `4238d6c3` | 1 |
| `register-alignment.yml` | Math-#508 | `74bbb0b2` → `cdc94c9d` | 219, 236, 252 |
| `remote-distance-moments-review.yml` | Math-#508 | `7ccebfad` → `76ff499a` | 239 |
| `remote-height-marks-review.yml` | Math-#508 | `a019deed` → `ab79ba21` | 241 |
| `remote-height-review.yml` | Math-#508 | `c947ba57` → `17ef7fae` | 240 |
| `remote-inverse-separation-review.yml` | Math-#508 | `54a3f691` → `910ebe46` | 242 |
| `remote-mixed-inverse-review.yml` | Math-#508 | `1939dd17` → `c7861b28` | 243 |
| `remote-pair-law-review.yml` | Math-#508 | `0229ba5e` → `1c6a3c9c` | 244 |
| `sard-g-a2-fixed-frame-review.yml` | Math-#508 | `5942d27d` → `429e6878` | 248 |
| `sard-g-a2-regularity.yml` | Math-#508 | `5791830f` → `52119c7e` | 107 |
| `sard-g-pinned-transfer.yml` | Math-#508 | `542bbd13` → `b0eebd5b` | 108 |
| `sard-g-r1r2-review.yml` | Math-#508 | `475888b2` → `522b318f` | 249 |
| `sard-g-r3r4-review.yml` | Math-#508 | `97b87e13` → `149eda16` | 250 |
| `side24-d4-closed-form.yml` | Math-#508 | `1ef15b05` → `55229360` | 264 |
| `side24-d4-coefficient.yml` | Math-#508 | `8ea01321` → `406f628b` | 263 |
| `side24-d5-closed-form.yml` | Math-#508 | `40fe9334` → `7c398929` | 266 |
| `side24-d5-coefficient.yml` | Math-#508 | `93ac2e5e` → `9f46fffa` | 265 |
| `side24-d6-coefficient.yml` | Math-#508 | `c45d9a5a` → `6059c2ec` | 267 |
| `side24-periodization-firstvar.yml` | Math-#508 | `15d650c5` → `47cbcc0c` | 112 |
| `side24-periodization-remainder.yml` | Math-#508 | `59227d63` → `33d499f5` | 113 |
| `soft-closed-form.yml` | Math-#508 | `efb3b211` → `5dd1ebf7` | 115, 116, 117 |
| `soft-fold-limit.yml` | Math-#508 | `839b99a4` → `792b2845` | 42 |
| `strict-unique-replacement-coefficient.yml` | Math-#508 | `452b78c0` → `f8271d8f` | 120 |

## Limits

- Only the units of `MISSING` rows were read; other ledger cells, rows and counts were not rechecked.
- The reader classes are a model-assisted judgment and can be wrong; each document carries its quoted evidence in the JSON.
- Formal statement maps (rows 280, 283, 284, 289, 290) and import-only Lean roots (rows 288, 313) are recorded but not
  counted as source documents; see `rule_sensitive_note`.
- Open-PR content is not promoted. Drive, Dropbox and comment-only sources are outside this record.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
