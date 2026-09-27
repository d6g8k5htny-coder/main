# Informal–formal alignment table

Generated from `formal/manifest.json` by `tools/formal_gate_check.py --write-alignment`; do not hand-edit.
Each row pairs one Lean target with the verbatim informal text it is attached to. The kernel checks the
declaration; the independent alignment review (see REVIEW_LANE.md) checks that the declaration says what
the anchor says. Source-level formalization status of every row is `proved`; only a trusted run receipt
reports `kernel-checked`. Nothing here is a Layer 0 review outcome.

| Lean target | Title | Informal anchor | Source | Does not claim |
|---|---|---|---|---|
| `UniversalLaw.Side24.pairing_sum_six` | Product-rule pairing count at q=6 equals 76 | `The coefficient sum for q=6 is 1+15+45+15=76` | side24-proof | the derivative bound 76\|x\|^6 e^(-\|x\|^2/2) itself |
| `UniversalLaw.Side24.pairing_sum_le_six` | Pairing counts for q<=6 are at most 76 | `it bounds the smaller orders as well` | side24-proof | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.pairing_terms_six` | Individual pairing terms 1, 15, 45, 15 | `1+15+45+15=76` | side24-proof | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.sixth_moment_double_factorial` | Double factorial 5!! = 15 | `At zero, every such contraction has absolute value at most15.` | side24-proof | that Gaussian derivative contractions at zero are bounded by 15 |
| `UniversalLaw.Side24.lattice_shell_constant` | Shell constants 27·27 = 729 and 729·2 = 1458 | `<=729 sum_{j>=1} j^9 e^(-288j^2) <=1458 e^(-288).` | side24-proof | the lattice-point count or the geometric tail sum |
| `UniversalLaw.Side24.geometric_ratio_constants` | Ratio constants 2^9 = 512 and 288·3 = 864 | `successive terms have ratio at most512e^(-864)<1/2` | side24-proof | that the ratio is below 1/2 |
| `UniversalLaw.Side24.period_exponent` | Period-24 exponent 24^2/2 = 288 | `sum_{n!=0} \|n\|^6 e^(-288\|n\|^2)` | side24-proof | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.image_constant_value` | Image constant 1458·(76·24^6+15) = 21175738586478 | `E := 1458*(76*24^6+15)*10^(-125) = 21175738586478 *10^(-125).` | side24-proof | the inequality \|D^q K_24(0)-D^q K_infty(0)\| < E |
| `UniversalLaw.Side24.exp_taylor_partial_sum_gt_ten` | Degree-20 Taylor partial sum of exp(288/125) exceeds 10 | `The code proves e^(288/125)>10 using the positive rational Taylor sum through20` | side24-proof | that the partial sum is below exp(288/125) (positivity of the omitted terms is analysis) |
| `UniversalLaw.Side24.covariance_relative_bound` | 60E < 10^(-108) as the integer inequality 60·21175738586478 < 10^17 | `epsilon=10^(-108), since60E<epsilon.` | side24-proof | the spectral-norm bound 20E or C_ref >= I/3 |
| `UniversalLaw.Side24.odd_block_eigenvalues` | Odd block trace 16 and determinant 6 = 8^2 - 58 | `[[1,-3],[-3,15]], whose eigenvalues are8 +/- sqrt(58)` | side24-proof | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.odd_block_shifted_minors` | Shifted odd block minors 2/3 and 7/9 (scaled by 3: 2 and 7) | `subtract I/3 and use first principal minor2/3 and determinant7/9` | side24-proof | Sylvester's criterion |
| `UniversalLaw.Side24.smaller_eigenvalue_exceeds_third` | 8 - sqrt(58) > 1/3 via 23^2 > 58·9 | `Its smaller eigenvalue exceeds 1/3` | side24-proof | monotonicity of the square root |
| `UniversalLaw.Side24.exponent_bounds` | Exponent inequalities a+2b<14 and 2a+b<13 for d=2,3 (in sixths) | `For d=2,3, a+2b<14 and2a+b<13.` | side24-proof | the log/exp estimates that follow |
| `UniversalLaw.Side24.exponent_values` | Exact a, b values for d=2,3 | `a=m+2/3+n/2, b=d+n/2.` | side24-proof | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.ratio_constant_ordering` | 13 <= 32 and 28 <= 32 | `ratio between1-13epsilon and 1+28epsilon, hence safely between1-32epsilon and1+32epsilon` | side24-proof | the elementary log/exp bounds |
| `UniversalLaw.Side24.density_comparison_below_reported` | 32·10^(-108) < 10^(-106) | `\|c_d,24/c_d,ref -1\| < 10^(-106), d=2,3.` | side24-proof | the multiplicative density comparison |
| `UniversalLaw.Side24.conditional_third_derivative_variance` | Conditional variance tau^2 = 15 - 9 = 6 | `Var(t)=15, Cov(t,G)=(-3,0,...,0)` | side24-proof | the Gaussian conditioning formula or the covariance derivatives |
| `UniversalLaw.Side24.transverse_covariance_entries_thirds` | Transverse covariance entries 8/3, 1, 2/3 from the delta formula | `Cov(A_ij,A_kl \| V=0) = (2/3)delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk.` | side24-proof | the delta formula itself |
| `UniversalLaw.Side24.cone_moment_m1_thirds` | D_1 = 4/3 is half of 8/3 | `D_1=E[A^2 1{A<0}]=4/3.` | side24-proof | centred Gaussian symmetry |
| `UniversalLaw.Side24.trace_and_traceless_variances` | Var(s) = 5/3, Var(x) = Var(y) = 1, Cov(s,x) = 0 from the 2x2 covariance entries | `Var(s)=5/3 and Var(x)=Var(y)=1.` | side24-proof | independence beyond zero covariance, which is Gaussian |
| `UniversalLaw.Side24.fourth_moment_of_trace` | E s^4 = 3·(5/3)^2 = 25/3 | `E s^2=5/3, E s^4=25/3` | side24-proof | the Gaussian fourth-moment identity |
| `UniversalLaw.Side24.exponential_moment_of_trace` | 1 + 5/3 = 8/3 behind E exp(-s^2/2) = sqrt(3/8) | `E exp(-s^2/2)=sqrt(3/8)` | side24-proof | the Gaussian exponential-moment identity |
| `UniversalLaw.Side24.cone_moment_m2_algebra` | D_2 algebra: rational part 29/6 and 4·sqrt(3/8) = sqrt(6) | `D_2 = (1/2)[25/3-20/3+8-8sqrt(3/8)] = 29/6-sqrt(6).` | side24-proof | the integral identity or the Rayleigh cone integration |
| `UniversalLaw.Side24.cube_root_simplification` | 6^2·2 = 3·24 behind 6^(2/3)/24^(1/3) = (3/2)^(1/3) | `simplifying 6^(2/3)/24^(1/3)=(3/2)^(1/3)` | side24-proof | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.pin_determinant_and_joint_dimension` | det diag(3,1,..,1) = 3 and joint dimensions 6, 10 | `Its dimension is at most10.` | side24-proof | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.hessian_block_eigenvalues` | Hessian block eigenvalues 4, 5, 2 exceed 1/3 | `The reference Hessian block has eigenvalues d+2 on trace and2 on traceless matrices` | side24-proof | the spectral decomposition |
| `UniversalLaw.Side24.d2_endpoints_adjacent` | Published d=2 endpoints are adjacent 20-digit decimals | `"2": { "lower": "0.07340691930603427103", "upper": "0.07340691930603427104"` | side24-enclosure | that c_2,24 lies between them |
| `UniversalLaw.Side24.d3_endpoints_adjacent` | Published d=3 endpoints are adjacent 20-digit decimals | `"3": { "lower": "0.04177593184059834334", "upper": "0.04177593184059834335"` | side24-enclosure | that c_3,24 lies between them |
| `UniversalLaw.Side24.d3_below_d2` | Published d=3 interval lies below the d=2 interval | `"scientific_acceptance": false` | side24-enclosure | anything about the true coefficients |
| `UniversalLaw.Side24.endpoints_in_unit_tenth` | Published endpoints lie in (0, 1/10) | `"relative_periodization_bound": "1e-106"` | side24-enclosure | anything about the true coefficients |
