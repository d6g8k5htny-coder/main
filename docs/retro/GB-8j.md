# GB-8j: D3 / coefficients (side24, LM006) node-candidate proposals (NOT A RESULT)

Lane GB-8j, bot agent 10, integrator agent 11. Assignment main#475 comment 6369972075; inventory main#527 (head e755231e). Pins at frozen Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`. Per GB-5 (#522 e1d1ebc2): proposals only, no GRAPH.json edit, no v2 records, no `hist.*`/URL sources, nothing tracks a branch tip. Classification capped at AUTHOR_SIDE_CANDIDATE / ENGINEERING_CONTROL. Full per-file blob + sha256 in `GB-8j.proposals.json`.

| Unit | Proposed node id | Cap | Tree ref | Impl paths |
|---|---|---|---|---|
| `reviews/lm006_normalizer_stability_20261005` | `math.lm006-normalizer-stability` | AUTHOR_SIDE_CANDIDATE | `d6g8k5htny-coder/Math-:reviews/lm006_normalizer_stability_20261005@9f3fb6918b28e6378c034ebc19e76d432f5391cb#364d74d0d1baee398989e465a537e8aa80ae05a9` | `stability_check.py`, `test_stability.py` |
| `reviews/lm006_positive_radius_density_20261006` | `math.lm006-positive-radius-density` | AUTHOR_SIDE_CANDIDATE | `d6g8k5htny-coder/Math-:reviews/lm006_positive_radius_density_20261006@9f3fb6918b28e6378c034ebc19e76d432f5391cb#466c50ce2a03855951a9ab7d6ed69c3ae84e6f55` | `joint_rank.py`, `test_joint_rank.py` |
| `reviews/lm006_product_transverse_20261006` | `math.lm006-product-transverse` | AUTHOR_SIDE_CANDIDATE | `d6g8k5htny-coder/Math-:reviews/lm006_product_transverse_20261006@9f3fb6918b28e6378c034ebc19e76d432f5391cb#2cc53c4676afb2cf3cc466bce72fae5fe5c9a1e5` | `product_transverse_check.py`, `test_product_transverse.py` |
| `reviews/lm006_quantified_regression_20261005` | `math.lm006-quantified-regression` | AUTHOR_SIDE_CANDIDATE | `d6g8k5htny-coder/Math-:reviews/lm006_quantified_regression_20261005@9f3fb6918b28e6378c034ebc19e76d432f5391cb#ff93bf249f18a4d87728bee790d3280efab9b883` | `regression_check.py`, `test_regression.py` |
| `reviews/lm006_tilted_endpoint_law_20261005` | `math.lm006-tilted-endpoint-law` | AUTHOR_SIDE_CANDIDATE | `d6g8k5htny-coder/Math-:reviews/lm006_tilted_endpoint_law_20261005@9f3fb6918b28e6378c034ebc19e76d432f5391cb#7f2fe76e453b7088ed633532abd9ddcf937ad635` | `tilt_check.py`, `test_tilt.py` |
| `reviews/lm006_typed_sign_suppression_20261005` | `math.lm006-typed-sign-suppression` | AUTHOR_SIDE_CANDIDATE | `d6g8k5htny-coder/Math-:reviews/lm006_typed_sign_suppression_20261005@9f3fb6918b28e6378c034ebc19e76d432f5391cb#633589b078ed1cc6cc83fffb0d40131d36db0d8a` | `typed_sign_check.py`, `test_typed_sign.py` |
| `reviews/lm006_wrong_sign_asymptotic_20261006` | `math.lm006-wrong-sign-asymptotic` | AUTHOR_SIDE_CANDIDATE | `d6g8k5htny-coder/Math-:reviews/lm006_wrong_sign_asymptotic_20261006@9f3fb6918b28e6378c034ebc19e76d432f5391cb#02d3232522fa86cd1f09cda099b0748f93936064` | `sign_coefficient.py`, `test_sign_coefficient.py` |
| `reviews/side24_v1_coefficient_claude_20260929` | `eng.side24-v1-coefficient-review-claude` | ENGINEERING_CONTROL | `d6g8k5htny-coder/Math-:reviews/side24_v1_coefficient_claude_20260929@9f3fb6918b28e6378c034ebc19e76d432f5391cb#0d3042b7efda251b53c6b2e38e5959b8cc2aee2d` | `coefficient_review_check.py` |
| `reviews/w3_side24_source_proposal_20261008` | `eng.w3-side24-source-proposal-scratch` | ENGINEERING_CONTROL | `d6g8k5htny-coder/Math-:reviews/w3_side24_source_proposal_20261008@9f3fb6918b28e6378c034ebc19e76d432f5391cb#7c089ba34e7f79c98113a444198fa01323e49441` | `W3Scratch/D2SchurLinks.lean`, `W3Scratch/Side24Law.lean`, `W3Scratch/Side24Poisson.lean` |
| `reviews/d3_l4_round2_audit_20261005` | `eng.d3-l4-round2-replay-scope` | ENGINEERING_CONTROL | `d6g8k5htny-coder/Math-:reviews/d3_l4_round2_audit_20261005@9f3fb6918b28e6378c034ebc19e76d432f5391cb#d7acdb687ddc00314754aa311e0a52dc178d6246` | ABSENT (no implementation file in unit at pin) |

Notes:
- `reviews/side24_v1_coefficient_claude_20260929`: Related existing node math.side24-coefficient in frontiers/downstream_gate_20260925/GRAPH.json@9f3fb691; NOT inherited (GB-5 Q4). Inventory also lists workflows .github/workflows/side24-v1-coefficient-review.yml and register-alignment.yml as impl; left out of proposal pins (workflow files, no workflow changes in this lane).
- `reviews/w3_side24_source_proposal_20261008`: README self-labels SCRATCH / ENGINEERING ONLY, unregistered, unreviewed; inventory listed 2 Lean impl files, tree has 3 (Side24Poisson.lean added here).
- `reviews/d3_l4_round2_audit_20261005`: Single file REPLAY_SCOPE.md; implementation ABSENT in unit (inventory: NO IMPLEMENTATION FOUND). Refers to coefficients/d3_l4_anisotropic_20261005 (GB-8h lane) and .github/workflows/d3-l4-anisotropic.yml; not bound here.
- Proposed node ids are lane proposals (UNVERIFIED naming); creating nodes is a separate reviewed Math- graph change.
- Drive: query `LM006` found 10 July-2026 docs, none mapped to a GB-8j unit; not imported. Dropbox: no access, NOT EXECUTED.
- Codex-stalled PRs: none touch these 10 units as far as checked (NOT EXECUTED as a full scan).

Bound: 0 (by rule). Proposed: 10/10. Implementation ABSENT: 1 (d3_l4_round2_audit).

credit 0 · sci NONE · eng≠discharge · no flags · OBL OPEN · NOT A RESULT
