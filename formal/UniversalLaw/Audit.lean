import UniversalLaw

/-!
# Axiom audit

Run after `lake build`:

    lake env lean UniversalLaw/Audit.lean

Every declaration registered in `formal/registry.json` with status
`kernel-checked` or `proved` must appear below. `tools/formal_gate_check.py`
parses the output and fails closed if a registered declaration is missing, or
depends on `sorryAx`, `Lean.ofReduceBool`, or any axiom outside the allow-list
(`propext`, `Classical.choice`, `Quot.sound`).
-/

open UniversalLaw.Side24

-- Ledger.lean
#print axioms pairing_sum_six
#print axioms pairing_sum_le_six
#print axioms pairing_terms_six
#print axioms sixth_moment_double_factorial
#print axioms lattice_shell_constant
#print axioms geometric_ratio_constants
#print axioms period_exponent
#print axioms image_constant_value
#print axioms exp_taylor_partial_sum_gt_ten
#print axioms covariance_relative_bound
#print axioms odd_block_eigenvalues
#print axioms odd_block_shifted_minors
#print axioms smaller_eigenvalue_exceeds_third
#print axioms exponent_bounds
#print axioms exponent_values
#print axioms ratio_constant_ordering
#print axioms density_comparison_below_reported

-- ConeMoments.lean
#print axioms conditional_third_derivative_variance
#print axioms transverse_covariance_entries_thirds
#print axioms cone_moment_m1_thirds
#print axioms trace_and_traceless_variances
#print axioms fourth_moment_of_trace
#print axioms exponential_moment_of_trace
#print axioms cone_moment_m2_algebra
#print axioms cube_root_simplification
#print axioms pin_determinant_and_joint_dimension
#print axioms hessian_block_eigenvalues

-- Endpoints.lean
#print axioms d2_endpoints_adjacent
#print axioms d3_endpoints_adjacent
#print axioms d3_below_d2
#print axioms endpoints_in_unit_tenth
