// Reproduced by tools/atlas_formal_evidence_build.py; source identity is not claim coverage.
export const EVIDENCE = {
  "schema_version": 1,
  "claim": {
    "id": "math.side24-coefficient",
    "record": {
      "layer": "D3",
      "kind": "candidate",
      "classification": "AUTHOR_SIDE_CANDIDATE",
      "controlling": false,
      "review_issue": 65,
      "fingerprint": "c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769",
      "source": "coefficients/side24_v1/PROOF.md",
      "notes": "Evaluates parent Eq15.2; parent reconciled in reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md",
      "review_disposition": "ACCEPT",
      "review_provider": "xAI/Grok via Cursor",
      "review_source": "https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490",
      "scope": "coefficient calculation for SIDE24 d=2,3; persistence reading via reconciled parent Theorem C; no finite-radius band"
    }
  },
  "document": {
    "blob": "44b66f04f89fcd87383b3603fa69f1feb64cdddd",
    "bytes": 10272,
    "commit": "b29c4b769cac85437b409d152651d5cce4140015",
    "path": "coefficients/side24_v1/PROOF.md",
    "repository": "d6g8k5htny-coder/Math-",
    "sha256": "c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769",
    "manifest_source": {
      "id": "side24-proof",
      "repository": "d6g8k5htny-coder/Math-",
      "commit": "9d7b6802424fb4715b31999066aafca8ee2f3cca",
      "path": "coefficients/side24_v1/PROOF.md",
      "bytes": 10272,
      "sha256": "c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769",
      "local_copy": "formal/sources/side24_v1/PROOF.md"
    }
  },
  "formal": {
    "repository": "d6g8k5htny-coder/main",
    "commit": "3b39c42d6edcc98ae165270be346f0e31c57f81d",
    "package": "side24-arithmetic-skeleton",
    "manifest_path": "formal/manifest.json",
    "manifest_sha256": "8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090",
    "manifest_blob": "0d694d7051670e954c35d6e2f55e753ceada64a2",
    "manifest_bytes": 17521,
    "scope_sha256": "fd06700628b1fbc81c5cfe3786aaa455032375bcb879914f068a66b726802a17",
    "formalization_status": "proved",
    "alignment_status": "PENDING_INDEPENDENT_REVIEW"
  },
  "declaration": {
    "name": "UniversalLaw.Side24.pairing_sum_six",
    "module": "formal/UniversalLaw/Side24/Ledger.lean",
    "title": "Product-rule pairing count at q=6 equals 76",
    "source": "side24-proof",
    "informal_anchor": "The coefficient sum for q=6 is\n1+15+45+15=76",
    "does_not_claim": "the derivative bound 76|x|^6 e^(-|x|^2/2) itself",
    "line": 38,
    "source_statement": "theorem pairing_sum_six : pairingSum 6 = 76",
    "elaborated_type": "UniversalLaw.Side24.pairing_sum_six :\n  @Eq Nat (UniversalLaw.Side24.pairingSum (@OfNat.ofNat Nat (nat_lit 6) (instOfNatNat (nat_lit 6))))\n    (@OfNat.ofNat Nat (nat_lit 76) (instOfNatNat (nat_lit 76)))",
    "hypothesis_binders": []
  },
  "execution": {
    "checked_commit": "3b39c42d6edcc98ae165270be346f0e31c57f81d",
    "repository": "d6g8k5htny-coder/main",
    "run_id": "40246431034",
    "run_attempt": "1",
    "receipt_sha256": "1ae8341c86a043ba917ec1fe68bf7e66ec528cd94b15170528487009f090c7e0",
    "artifact_id": 12618189537,
    "artifact_sha256": "80ac88cf770afbadd89f4801a4692d6b6540a9269dfb916c249d491c2b41f737",
    "lean_version": "Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)",
    "formalization_status": "kernel-checked",
    "target_axioms": [],
    "types_sha256": "10a954ced98371c5cf98eecada196d50c1e6e570af9ea8e07caade4a80b0aab8"
  },
  "alignment": {
    "record": {
      "disposition": "REJECTED",
      "manifest_sha256": "8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090",
      "scope_sha256": "fd06700628b1fbc81c5cfe3786aaa455032375bcb879914f068a66b726802a17",
      "targets": [
        "UniversalLaw.Side24.pairing_sum_six",
        "UniversalLaw.Side24.pairing_sum_le_six",
        "UniversalLaw.Side24.pairing_terms_six",
        "UniversalLaw.Side24.sixth_moment_double_factorial",
        "UniversalLaw.Side24.lattice_shell_constant",
        "UniversalLaw.Side24.geometric_ratio_constants",
        "UniversalLaw.Side24.period_exponent",
        "UniversalLaw.Side24.image_constant_value",
        "UniversalLaw.Side24.exp_taylor_partial_sum_gt_ten",
        "UniversalLaw.Side24.covariance_relative_bound",
        "UniversalLaw.Side24.odd_block_eigenvalues",
        "UniversalLaw.Side24.odd_block_shifted_minors",
        "UniversalLaw.Side24.smaller_eigenvalue_exceeds_third",
        "UniversalLaw.Side24.exponent_bounds",
        "UniversalLaw.Side24.exponent_values",
        "UniversalLaw.Side24.ratio_constant_ordering",
        "UniversalLaw.Side24.density_comparison_below_reported",
        "UniversalLaw.Side24.conditional_third_derivative_variance",
        "UniversalLaw.Side24.transverse_covariance_entries_thirds",
        "UniversalLaw.Side24.cone_moment_m1_thirds",
        "UniversalLaw.Side24.trace_and_traceless_variances",
        "UniversalLaw.Side24.fourth_moment_of_trace",
        "UniversalLaw.Side24.exponential_moment_of_trace",
        "UniversalLaw.Side24.cone_moment_m2_algebra",
        "UniversalLaw.Side24.cube_root_simplification",
        "UniversalLaw.Side24.pin_determinant_and_joint_dimension",
        "UniversalLaw.Side24.hessian_block_eigenvalues",
        "UniversalLaw.Side24.d2_endpoints_adjacent",
        "UniversalLaw.Side24.d3_endpoints_adjacent",
        "UniversalLaw.Side24.d3_below_d2",
        "UniversalLaw.Side24.endpoints_in_unit_tenth"
      ],
      "author": {
        "provider": "Anthropic",
        "family": "Claude",
        "agent": "Cursor cloud agent, 2026-09-27"
      },
      "reviewer": {
        "provider": "xAI",
        "family": "Grok Bot",
        "agent": "Grok Bot agent 3"
      },
      "evidence": {
        "repository": "d6g8k5htny-coder/main",
        "commit": "6c86473823d32d8eee0d80a0d5550efce066d1f4",
        "path": "audits/formal_alignment/20261026_grok_bot_agent_3.md",
        "sha256": "78c21c96d851dfd6a4fbcf68edb091d3aac21b757885e615403d3fe6337e1692"
      },
      "source_exposure": "statements and docstrings first; proof bodies (all `by decide`) seen but not used for verdict",
      "per_target_verdicts_in_review_file": true
    },
    "selected_target_verdict": "ALIGNED",
    "record_source": {
      "repository": "d6g8k5htny-coder/main",
      "commit": "a228f4410141549462260ba5ea426b7d08c358c1",
      "path": "audits/formal_alignment/20261026_grok_bot_agent_3.json",
      "sha256": "ce0540dd8ac49970fd7ddb92fdb470725e2fb7ee0a69be27a6e955c8b8632852"
    }
  },
  "scientific_effect": "NONE"
};
