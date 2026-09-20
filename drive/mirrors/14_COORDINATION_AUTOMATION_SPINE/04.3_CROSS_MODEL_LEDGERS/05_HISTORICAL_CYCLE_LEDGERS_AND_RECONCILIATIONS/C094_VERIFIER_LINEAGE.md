# C094 Q0-X Intake and Verifier-Lineage Continuity

| Feature | v2 | v3 | v4 | Continuity route |
|---|---:|---:|---:|---|
| proof DAG validation, cycle/supersession checks | ✓ | ✓ | ✓ | imports Grade/ClaimNode/ProofGraph from v3 |
| incremental descendant invalidation and Merkle certificates | ✓ | ✓ | ✓ | v3 dependency + SemanticProofGraph Merkle roots |
| H0 widest-path grounding and exact forest compression | ✓ | ✓ | ✓ | imports SupportEdge/widest_path_grounding from v3 |
| anchored universal-kernel RKHS mismatch | ✓ | ✓ | ✓ | imports anchored_rkhs_mismatch from v3 |
| anisotropic Q-SARD tube bound and effective rank | ✓ | ✓ | ✓ | imports anisotropic_sard_tube_bound/effective_transverse_rank |
| CoverageCertificate | — | ✓ | ✓ | direct import from v3 |
| SelectionAwareRiskLedger and coverage-limited rank | — | ✓ | ✓ | direct import from v3 |
| semantic theorem-contract gates | — | — | ✓ | SemanticProofGraph.validate |
| MARK and COMPOSITION gates | — | — | ✓ | SemanticProofGraph.validate |

## Result

- v4 subsumes every validated v2/v3 component in the directive.
- `q0_llm_verifier_v3.py` remains a direct import dependency of v4.
- the clean-extraction v4 validation passed.
- Q0-X v2 is registered as the experiment and theorem-design source for Q0-IV and Q0-LLM.
- v5 additions are new C094 scope, not retroactive claims about v4.