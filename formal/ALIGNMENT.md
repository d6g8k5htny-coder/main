# Informal–formal alignment table

Generated from `formal/registry.json` by `tools/formal_gate_check.py --write-alignment`; do not hand-edit.
Each row pairs one Lean declaration with the verbatim informal text it is attached to. The kernel checks the
declaration; the alignment review lane checks that the declaration says what the anchor says. Status here is
formalization status only and changes no Layer 0 review outcome.

| Claim | Layer 0 object | Status | Alignment | Lean declaration | Informal anchor | Does not claim |
|---|---|---|---|---|---|---|
| side24-pairing-sum-six | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.pairing_sum_six` | `The coefficient sum for q=6 is 1+15+45+15=76` | the derivative bound 76\|x\|^6 e^(-\|x\|^2/2) itself |
| side24-pairing-sum-lower-orders | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.pairing_sum_le_six` | `it bounds the smaller orders as well` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| side24-pairing-terms | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.pairing_terms_six` | `1+15+45+15=76` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| side24-sixth-moment | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.sixth_moment_double_factorial` | `At zero, every such contraction has absolute value at most15.` | that Gaussian derivative contractions at zero are bounded by 15 |
| side24-lattice-shell | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.lattice_shell_constant` | `<=729 sum_{j>=1} j^9 e^(-288j^2) <=1458 e^(-288).` | the lattice-point count or the geometric tail sum |
| side24-geometric-ratio | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.geometric_ratio_constants` | `successive terms have ratio at most512e^(-864)<1/2` | that the ratio is below 1/2 |
| side24-period-exponent | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.period_exponent` | `sum_{n!=0} \|n\|^6 e^(-288\|n\|^2)` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| side24-image-constant | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.image_constant_value` | `E := 1458*(76*24^6+15)*10^(-125) = 21175738586478 *10^(-125).` | the inequality \|D^q K_24(0)-D^q K_infty(0)\| < E |
| side24-exp-taylor | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.exp_taylor_partial_sum_gt_ten` | `The code proves e^(288/125)>10 using the positive rational Taylor sum through20` | that the partial sum is below exp(288/125) (positivity of the omitted terms is analysis) |
| side24-covariance-bound | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.covariance_relative_bound` | `epsilon=10^(-108), since60E<epsilon.` | the spectral-norm bound 20E or C_ref >= I/3 |
| side24-odd-block-eigen | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.odd_block_eigenvalues` | `[[1,-3],[-3,15]], whose eigenvalues are8 +/- sqrt(58)` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| side24-odd-block-shift | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.odd_block_shifted_minors` | `subtract I/3 and use first principal minor2/3 and determinant7/9` | Sylvester's criterion |
| side24-eigen-third | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.smaller_eigenvalue_exceeds_third` | `Its smaller eigenvalue exceeds 1/3` | monotonicity of the square root |
| side24-exponent-bounds | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.exponent_bounds` | `For d=2,3, a+2b<14 and2a+b<13.` | the log/exp estimates that follow |
| side24-exponent-values | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.exponent_values` | `a=m+2/3+n/2, b=d+n/2.` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| side24-ratio-constants | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.ratio_constant_ordering` | `ratio between1-13epsilon and 1+28epsilon, hence safely between1-32epsilon and1+32epsilon` | the elementary log/exp bounds |
| side24-density-comparison | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.density_comparison_below_reported` | `\|c_d,24/c_d,ref -1\| < 10^(-106), d=2,3.` | the multiplicative density comparison |
| side24-tau-squared | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.conditional_third_derivative_variance` | `Var(t)=15, Cov(t,G)=(-3,0,...,0)` | the Gaussian conditioning formula or the covariance derivatives |
| side24-transverse-covariance | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.transverse_covariance_entries_thirds` | `Cov(A_ij,A_kl \| V=0) = (2/3)delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk.` | the delta formula itself |
| side24-cone-m1 | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.cone_moment_m1_thirds` | `D_1=E[A^2 1{A<0}]=4/3.` | centred Gaussian symmetry |
| side24-trace-variance | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.trace_and_traceless_variances` | `Var(s)=5/3 and Var(x)=Var(y)=1.` | independence beyond zero covariance, which is Gaussian |
| side24-fourth-moment | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.fourth_moment_of_trace` | `E s^2=5/3, E s^4=25/3` | the Gaussian fourth-moment identity |
| side24-exp-moment | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.exponential_moment_of_trace` | `E exp(-s^2/2)=sqrt(3/8)` | the Gaussian exponential-moment identity |
| side24-cone-m2 | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.cone_moment_m2_algebra` | `D_2 = (1/2)[25/3-20/3+8-8sqrt(3/8)] = 29/6-sqrt(6).` | the integral identity or the Rayleigh cone integration |
| side24-cube-root | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.cube_root_simplification` | `simplifying 6^(2/3)/24^(1/3)=(3/2)^(1/3)` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| side24-pin-dimension | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.pin_determinant_and_joint_dimension` | `Its dimension is at most10.` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| side24-hessian-eigen | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.hessian_block_eigenvalues` | `The reference Hessian block has eigenvalues d+2 on trace and2 on traceless matrices` | the spectral decomposition |
| side24-d2-endpoints | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.d2_endpoints_adjacent` | `"2": { "lower": "0.07340691930603427103", "upper": "0.07340691930603427104"` | that c_2,24 lies between them |
| side24-d3-endpoints | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.d3_endpoints_adjacent` | `"3": { "lower": "0.04177593184059834334", "upper": "0.04177593184059834335"` | that c_3,24 lies between them |
| side24-d3-below-d2 | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.d3_below_d2` | `"scientific_acceptance": false` | anything about the true coefficients |
| side24-endpoints-range | D3 — SIDE24 coefficient calculation | kernel-checked | open | `UniversalLaw.Side24.endpoints_in_unit_tenth` | `"relative_periodization_bound": "1e-106"` | anything about the true coefficients |
| d3-side24-coefficient-bound | D3 — SIDE24 coefficient calculation | none | not-applicable | — | — | any formal verification; Layer 0 review status is unchanged |
| d2-lifetime-remainder | D2 — unrestricted lifetime remainder | none | not-applicable | — | — | any formal verification; Layer 0 review status is unchanged |
| d4-fixed-remote-rn-count | D4 — fixed-remote RN count theorem | none | not-applicable | — | — | any formal verification; Layer 0 review status is unchanged |
| d6-p15-full-price | D6 — P15 full-price theorem | none | not-applicable | — | — | any formal verification; Layer 0 review status is unchanged |
| d1-parent-selection-chain | D1 — parent quantitative Theorem-A selection chain | none | not-applicable | — | — | any formal verification; Layer 0 review status is unchanged |
| d5-pin-neighborhoods | D5 — pin neighborhoods / microdisk | none | not-applicable | — | — | any formal verification; Layer 0 review status is unchanged |
| sard-g-a1-a6 | SARD-G A1/A6 | none | not-applicable | — | — | any formal verification; Layer 0 review status is unchanged |
