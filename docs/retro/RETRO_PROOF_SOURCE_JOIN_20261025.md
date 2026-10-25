# Retroactive proof-source and implementation-reference join — 25 October 2026

> **NOT A RESULT.** Organizational source accounting only. Scientific effect **NONE**. No graph node, typed edge, theorem status, or proof-to-code equivalence is created.

## Pinned source contract

- Proof-path order: `d6g8k5htny-coder/Math-:docs/integration/2026-10-23-postcut-proof-sources.md@ab856c1e99bfa12c37c7663d65f8b17c4551d61a` (blob `a1008e928c56cb360509db0f894b5fafae435df3`; observed Math cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`).
- Main snapshot: `6332601a0169890ccebd67f9d34e0aaa2ac67f08` (tree `de8982fab32c61f129f4a1248047c2d5a6f54d28`).
- Unit inventory blob: `09cd239635708ef0226f443ad679f72bc6e40d14`.
- Implementation-presence audit blob: `49082238102aa4b5139a86813249e87d47552eb5`.
- Open-PR overlay blob: `26f16dfc06f9b83632e0336fb87f1aa386401045`; used as an exclusion boundary, not a landed source.
- v2 binding register blob: `6b67c9c91d69056a447357dfe6d438a0eaeff492`.

The JSON companion maps each of the pinned inventory's 73 proof entries by its 1-based list ordinal. The pinned inventory remains the authoritative path/blob carrier; the join supplies packet, role, implementation, proposal, and binding dispositions.

## Result

- **73** proof paths → **70** unique proof blobs → **34** canonical packets.
- Roles: **20** canonical top-level, **48** nested packet proofs, **4** embedded input copies, **1** embedded source copy.
- **28** packets have local code/workflow references; **6** remain `NO IMPLEMENTATION FOUND` at the pinned audit.
- **32** packets have landed node-candidate proposals; **2** GB-8s packets intentionally have inventory only and no node IDs.
- Existing v2 bindings: **0**. Open-PR identities promoted as landed source: **0**.

## Implementation-reference gaps

- `frontiers/far_sector_lifetime_refinement_20261022`
- `frontiers/full_spectrum_enclosure_method_20261018`
- `frontiers/gaussian_a4_elder_sector_20261017`
- `frontiers/gaussian_short_bar_theory_20261017`
- `frontiers/nongaussian_smoothing_admission_20261017`
- `frontiers/original_finite_l_gaussian_binding_20261017`

The label means only that the pinned implementation-presence audit recorded no reference; it is not a universal nonexistence claim.

## Duplicate proof blobs

| Group | Blob | Canonical path | Alias/copy path |
|---|---|---|---|
| `dup-01` | `66d9c4c7f7862b51ef3461197e8949731fa5b5de` | `frontiers/gaussian_short_bar_theory_20261017/asymmetric_elder/PROOF.md` | `frontiers/compact_morse_elder_adapter_20261018/inputs/Math-/frontiers/gaussian_short_bar_theory_20261017/asymmetric_elder/PROOF.md` |
| `dup-02` | `9c79a672f5d81e04972032b2f1faf185f4664fa5` | `frontiers/unique_replacement_bar_intensity_20261001/PROOF.md` | `frontiers/replacement_bar_occurrence_20261001/sources/c50/PROOF.md` |
| `dup-03` | `d61fd11a826856baecb4cf4cd8a76ad5c25350dd` | `frontiers/gaussian_short_bar_theory_20261017/smooth_gaussian_manifold_admission/PROOF.md` | `frontiers/compact_morse_elder_adapter_20261018/inputs/Math-/frontiers/gaussian_short_bar_theory_20261017/smooth_gaussian_manifold_admission/PROOF.md` |

These aliases are provenance/replay copies, not additional mathematical results.

## Packet join

| Packet | Impl class | Implementation references | Lane | Proposal | v2 |
|---|---|---|---|---|---|
| `frontiers/c6_hard_direction_factorization_20260930` | `local_code_language_reference` | `frontiers/c6_hard_direction_factorization_20260930/factorization.py`<br>`.github/workflows/c6-hard-direction-factorization.yml` | `GB-8k` | landed proposal | no |
| `frontiers/c7_elder_two_thirds_20261007` | `local_code_language_reference` | `frontiers/c7_elder_two_thirds_20261007/C7/author_controls.py`<br>`frontiers/c7_elder_two_thirds_20261007/replay.py`<br>`.github/workflows/lifetime-c7-elder.yml`<br>`tests/test_lifetime_c7_elder_replay_evidence.py` | `GB-8e` | landed proposal | no |
| `frontiers/candidate_parity_rate_20261001` | `local_code_language_reference` | `frontiers/candidate_parity_rate_20261001/parity_check.py`<br>`.github/workflows/candidate-parity-rate.yml`<br>`.github/workflows/d1-sharp-remainder.yml` | `GB-8r` | landed proposal | no |
| `frontiers/compact_morse_elder_adapter_20261018` | `local_code_language_reference` | `frontiers/compact_morse_elder_adapter_20261018/inputs/C291/SourceLifetimeField.lean`<br>`frontiers/compact_morse_elder_adapter_20261018/inputs/C291/SourceLifetimeGeometry.lean` | `GB-8l` | landed proposal | no |
| `frontiers/cusp_torus_transfer_20261001` | `local_code_language_reference` | `frontiers/cusp_torus_transfer_20261001/test_sources.py`<br>`frontiers/cusp_torus_transfer_20261001/test_transfer.py`<br>`.github/workflows/c2-torus-transfer.yml`<br>`.github/workflows/cusp-torus-transfer.yml` | `GB-8l` | landed proposal | no |
| `frontiers/d1_line_20261004` | `local_code_language_reference` | `frontiers/d1_line_20261004/line_check.py`<br>`.github/workflows/d1-line.yml` | `GB-8e` | landed proposal | no |
| `frontiers/d1_sharp_remainder_20261001` | `local_code_language_reference` | `frontiers/d1_sharp_remainder_20261001/d1p_check.py`<br>`.github/workflows/d1-line.yml`<br>`.github/workflows/d1-sharp-remainder.yml` | `GB-8e` | landed proposal | no |
| `frontiers/d1_third_order_law_20261001` | `local_code_language_reference` | `frontiers/d1_third_order_law_20261001/d1_check.py`<br>`.github/workflows/candidate-parity-rate.yml`<br>`.github/workflows/d1-line.yml` | `GB-8e` | landed proposal | no |
| `frontiers/em_intermediate_elder_20261006` | `local_code_language_reference` | `frontiers/em_intermediate_elder_20261006/EM/author_controls.py`<br>`frontiers/em_intermediate_elder_20261006/replay.py`<br>`.github/workflows/em-intermediate-elder.yml`<br>`tests/test_em_intermediate_elder_replay_evidence.py` | `GB-8b` | landed proposal | no |
| `frontiers/far_elder_flat_ridge_20260930` | `local_code_language_reference` | `frontiers/far_elder_flat_ridge_20260930/c89_replay.py`<br>`frontiers/far_elder_flat_ridge_20260930/c89_reviewer_controls.py`<br>`.github/workflows/far-elder-flat-ridge.yml` | `GB-8l` | landed proposal | no |
| `frontiers/far_sector_lifetime_refinement_20261022` | `no_implementation_found` | — | `GB-8g` | landed proposal | no |
| `frontiers/finite_l_d2_coefficient_20261018` | `local_code_language_reference` | `frontiers/finite_l_d2_coefficient_20261018/coefficient.py`<br>`frontiers/finite_l_d2_coefficient_20261018/contract.py` | `GB-8e` | landed proposal | no |
| `frontiers/full_spectrum_enclosure_method_20261018` | `no_implementation_found` | — | `GB-8w` | landed proposal | no |
| `frontiers/gaussian_a4_elder_sector_20261017` | `no_implementation_found` | — | `GB-8m` | landed proposal | no |
| `frontiers/gaussian_short_bar_theory_20261017` | `no_implementation_found` | — | `GB-8w` | landed proposal | no |
| `frontiers/lifetime_c3r_rate_20261013` | `local_code_language_reference` | `frontiers/lifetime_c3r_rate_20261013/controls/c3r_exact.py`<br>`frontiers/lifetime_c3r_rate_20261013/controls/c3r_mutants.py`<br>`.github/workflows/lifetime-c3r-rate.yml`<br>`tests/test_lifetime_c3r_replay.py` | `GB-8e` | landed proposal | no |
| `frontiers/lifetime_cj_model_j_20261012` | `local_code_language_reference` | `frontiers/lifetime_cj_model_j_20261012/CJ/author_controls.py`<br>`frontiers/lifetime_cj_model_j_20261012/replay.py`<br>`.github/workflows/lifetime-cj-model-j.yml`<br>`tests/test_lifetime_cj_model_j_replay.py` | `GB-8f` | landed proposal | no |
| `frontiers/lifetime_dd_model_d_20261013` | `local_code_language_reference` | `frontiers/lifetime_dd_model_d_20261013/DD/author_controls.py`<br>`frontiers/lifetime_dd_model_d_20261013/replay.py`<br>`.github/workflows/lifetime-dd-model-d.yml`<br>`tests/test_lifetime_dd_model_d_replay.py` | `GB-8f` | landed proposal | no |
| `frontiers/lifetime_two_thirds_chain_20261007` | `local_code_language_reference` | `frontiers/lifetime_two_thirds_chain_20261007/C3/author_controls.py`<br>`frontiers/lifetime_two_thirds_chain_20261007/EM1/author_controls.py`<br>`.github/workflows/lifetime-two-thirds-chain.yml`<br>`main:tests/test_reader_links.py` | `GB-8f` | landed proposal | no |
| `frontiers/moving_remote_collision_20261003` | `local_code_language_reference` | `frontiers/moving_remote_collision_20261003/author_controls.py`<br>`frontiers/moving_remote_collision_20261003/review/independent_controls.py`<br>`.github/workflows/moving-remote-collision.yml` | `GB-8b` | landed proposal | no |
| `frontiers/nongaussian_smoothing_admission_20261017` | `no_implementation_found` | — | `GB-8w` | landed proposal | no |
| `frontiers/ordinary_short_bar_occurrence_20261001` | `local_code_language_reference` | `frontiers/ordinary_short_bar_occurrence_20261001/check.py`<br>`frontiers/ordinary_short_bar_occurrence_20261001/test_check.py`<br>`.github/workflows/ordinary-short-bar-occurrence-all-d.yml`<br>`.github/workflows/ordinary-short-bar-occurrence.yml` | `GB-8s` | inventory only; no node IDs | no |
| `frontiers/original_finite_l_gaussian_binding_20261017` | `no_implementation_found` | — | `GB-8w` | landed proposal | no |
| `frontiers/planar_compact_k_chain_20261006` | `local_code_language_reference` | `frontiers/planar_compact_k_chain_20261006/A4_1/author_controls.py`<br>`frontiers/planar_compact_k_chain_20261006/A4_2/author_controls.py`<br>`.github/workflows/planar-compact-k-chain.yml`<br>`tests/test_planar_compact_k_replay_evidence.py` | `GB-8n` | landed proposal | no |
| `frontiers/planar_rejected_endpoint_margin_20261003` | `local_code_language_reference` | `frontiers/planar_rejected_endpoint_margin_20261003/A4/author_controls.py`<br>`frontiers/planar_rejected_endpoint_margin_20261003/replay.py`<br>`.github/workflows/planar-rejected-endpoint-margin.yml` | `GB-8n` | landed proposal | no |
| `frontiers/planar_soft_layer_chain_20261003` | `local_code_language_reference` | `frontiers/planar_soft_layer_chain_20261003/C101/author_controls.py`<br>`frontiers/planar_soft_layer_chain_20261003/C101/checker.py`<br>`.github/workflows/planar-soft-layer-chain.yml`<br>`main:tests/test_reader_links.py` | `GB-8n` | landed proposal | no |
| `frontiers/qs_d3_soft_layer_chain_20261005` | `local_code_language_reference` | `frontiers/qs_d3_soft_layer_chain_20261005/A3/author_controls.py`<br>`frontiers/qs_d3_soft_layer_chain_20261005/A3_1/author_controls.py`<br>`.github/workflows/qs-d3-soft-layer-chain.yml`<br>`tests/test_qs_d3_replay_evidence.py` | `GB-8i` | landed proposal | no |
| `frontiers/replacement_bar_occurrence_20261001` | `local_code_language_reference` | `frontiers/replacement_bar_occurrence_20261001/check_exact.py`<br>`frontiers/replacement_bar_occurrence_20261001/test_exact.py`<br>`.github/workflows/replacement-bar-occurrence.yml`<br>`main:tests/test_reader_links.py` | `GB-8s` | inventory only; no node IDs | no |
| `frontiers/small_lifetime_shape_20260930` | `local_code_language_reference` | `frontiers/small_lifetime_shape_20260930/shape.py`<br>`frontiers/small_lifetime_shape_20260930/test_jacobian.py`<br>`.github/workflows/small-lifetime-shape.yml` | `GB-8f` | landed proposal | no |
| `frontiers/strict_unique_replacement_coefficient_20261003` | `local_code_language_reference` | `frontiers/strict_unique_replacement_coefficient_20261003/check.py`<br>`.github/workflows/strict-unique-replacement-coefficient.yml`<br>`main:tests/test_reader_links.py` | `GB-8i` | landed proposal | no |
| `frontiers/tl_soft_layer_20261006` | `local_code_language_reference` | `frontiers/tl_soft_layer_20261006/TL/author_controls.py`<br>`frontiers/tl_soft_layer_20261006/replay.py`<br>`.github/workflows/tl-soft-layer.yml`<br>`tests/test_tl_soft_layer_replay_evidence.py` | `GB-8o` | landed proposal | no |
| `frontiers/ts_small_kappa_20261006` | `local_code_language_reference` | `frontiers/ts_small_kappa_20261006/TS/author_controls.py`<br>`frontiers/ts_small_kappa_20261006/replay.py`<br>`.github/workflows/ts-small-kappa.yml`<br>`tests/test_ts_small_kappa_replay_evidence.py` | `GB-8o` | landed proposal | no |
| `frontiers/unique_replacement_bar_intensity_20261001` | `local_code_language_reference` | `frontiers/unique_replacement_bar_intensity_20261001/check_exact.py`<br>`frontiers/unique_replacement_bar_intensity_20261001/test_sources.py`<br>`.github/workflows/unique-replacement-bars.yml`<br>`main:tests/test_reader_links.py` | `GB-8t` | landed proposal | no |
| `frontiers/v24_dc_two_thirds_sign_20261007` | `local_code_language_reference` | `frontiers/v24_dc_two_thirds_sign_20261007/DC/author_controls.py`<br>`frontiers/v24_dc_two_thirds_sign_20261007/V24/author_controls.py`<br>`.github/workflows/lifetime-v24-dc-sign.yml`<br>`tests/test_lifetime_v24_dc_sign_replay_evidence.py` | `GB-8f` | landed proposal | no |

## Interpretation limits

- This packet records source presence and repository-local implementation references; it does not assert semantic proof-to-implementation equivalence.
- A node-candidate proposal is not a graph node, accepted theorem, scientific status change, or v2 binding.
- All existing_v2_binding_present values are false at the pinned v2 register.
- Embedded input/source copies and byte-identical aliases are not counted as additional mathematical results.
- The open-PR overlay is cited only as an input boundary and is not used to promote an unmerged source identity.
- Any graph edge or dependency assertion requires a later source-bound semantic review under the applicable scientific-review rules.
- NO IMPLEMENTATION FOUND means no implementation reference was recorded by the pinned implementation-presence audit; it is not proof that no implementation exists anywhere.

## Next boundary

A packet-specific semantic review must determine whether each named program implements, verifies, replays, or merely references the mathematical source. Only that later review can justify a typed dependency edge.
