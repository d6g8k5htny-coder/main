# Exact formal scope and alignment review — SIDE24 arithmetic skeleton

Scientific effect: NONE. This package offers 31 exact integer statements that are
the arithmetic skeleton of the SIDE24 coefficient note, `Math-` commit
`9d7b6802424fb4715b31999066aafca8ee2f3cca`, `coefficients/side24_v1/PROOF.md`
(10,272 bytes, SHA-256 `c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769`),
and of its published `ENCLOSURE.json` (1,090 bytes, SHA-256
`72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811`). Both sources
are preserved byte-identically under `sources/side24_v1/`. Core Lean has no
rational type, so each rational identity is stated as the equivalent
cross-multiplied integer identity; the docstring gives the rational form and the
independent alignment review must check that the two agree.

This package addresses the **first step only** of the "Numerical formalizer" work
offer in [docs/FORMAL_VERIFICATION.md](../docs/FORMAL_VERIFICATION.md). It does
**not** supply the exact definition of the SIDE24 coefficient, any real-analytic
enclosure or remainder step, or a proof-producing enclosure pipeline. Per the
[reconnaissance memo](../docs/reconnaissance/2026-09-27-formal-verification.md):
rational arithmetic on two endpoints cannot establish that an analytically
defined coefficient lies between them, and nothing below claims that it does.

| Targets | Exact coverage | Not established |
|---|---|---|
| `pairing_sum_six`, `pairing_sum_le_six`, `pairing_terms_six` | Product-rule pairing counts `∑ q!/(2^j j! (q−2j)!)`: value `1+15+45+15 = 76` at `q = 6`, dominance of `q ≤ 6` | The Gaussian derivative bound `76|x|^6 e^{-|x|^2/2}` on `|x| ≥ 1` |
| `sixth_moment_double_factorial` | `1·3·5 = 15` | That contractions of `D^6 e^{-|x|^2/2}` at `0` are bounded by 15 |
| `lattice_shell_constant`, `geometric_ratio_constants`, `period_exponent` | `27·27 = 729`, `729·2 = 1458`, `2^9 = 512`, `288·3 = 864`, `24^2/2 = 288` | Lattice-point counting, the geometric tail sum, the ratio being below `1/2` |
| `image_constant_value` | `1458·(76·24^6+15) = 21175738586478` | The image inequality `|D^q K_24(0) − D^q K_∞(0)| < E` |
| `exp_taylor_partial_sum_gt_ten` | Degree-20 Taylor partial sum of `exp(288/125)`, cleared of denominators, exceeds `10` | That the partial sum is below `exp(288/125)` (positivity of omitted terms is analysis) |
| `covariance_relative_bound` | `2·10·3 = 60` and `60·21175738586478 < 10^17` (i.e. `60E < 10^-108`) | The spectral-norm bound `20E`, `C_ref ≥ I/3`, positive-semidefinite ordering |
| `odd_block_eigenvalues`, `odd_block_shifted_minors`, `smaller_eigenvalue_exceeds_third` | Trace/determinant of `[[1,−3],[−3,15]]`; minors `2`, `7` of `3M − I`; `58·9 < 23^2` | Sylvester's criterion, spectral decomposition, monotonicity of `√` |
| `exponent_bounds`, `exponent_values` | For `d ∈ {2,3}`: `6a+12b < 84`, `12a+6b < 78` with `6a = 6(d−1)+4+3n`, `6b = 6d+3n`, `n = (d−1)d/2` | The `log(1±ε)` / `exp` estimates that use these exponents |
| `ratio_constant_ordering`, `density_comparison_below_reported` | `13 ≤ 32`, `28 ≤ 32`, `32·10^106 < 10^108` | The multiplicative Gaussian density comparison and its integration |
| `conditional_third_derivative_variance` | `15 − (−3)(−3)/1 = 6` | The Gaussian conditioning formula, covariance differentiation |
| `transverse_covariance_entries_thirds`, `cone_moment_m1_thirds` | Entries `8/3`, `1`, `2/3` from the displayed delta formula; `D_1 = 4/3` as half of `8/3` | The delta formula itself; centred Gaussian symmetry |
| `trace_and_traceless_variances`, `fourth_moment_of_trace`, `exponential_moment_of_trace` | `Var s = 5/3`, `Var x = 1`, `Cov(s,x) = 0` from the `2×2` entries; `3·(5/3)^2 = 25/3`; `1 + 5/3 = 8/3` | Gaussian independence from zero covariance, the fourth-moment and exponential-moment identities |
| `cone_moment_m2_algebra` | `4·5 = 20`; `25 − 20 + 24 = 29`; `4^2·3 = 6·8` behind `D_2 = 29/6 − √6` | The integral `∫_0^a (a−z)^2 e^{-z/2} dz/2`, the Rayleigh cone integration |
| `cube_root_simplification` | `6^2·2 = 3·24` behind `6^(2/3)/24^(1/3) = (3/2)^(1/3)` | Real cube roots |
| `pin_determinant_and_joint_dimension`, `hessian_block_eigenvalues` | `det diag(3,1,1) = 3`; joint dimensions `6`, `10`; eigenvalues `4`, `5`, `2` exceed `1/3` | The joint covariance structure, the Hessian block decomposition |
| `d2_endpoints_adjacent`, `d3_endpoints_adjacent`, `d3_below_d2`, `endpoints_in_unit_tenth` | The published 20-digit endpoints, transcribed as integers in `10^-20` units, are adjacent, ordered, and in `(0, 1/10)` | **That either coefficient lies between its endpoints.** This is a consistency check of the frozen artifact, not an enclosure |

None of these targets is the SIDE24 coefficient enclosure, `Γ(7/6)`, `π`, `exp`
or `log` bounds, the Stirling remainder, Gaussian conditioning, the cone
integrals, the parent lifetime theorem, elder selection or Kac–Rice. The Layer 0
status of D3 in [STATUS.md](../STATUS.md) is unchanged by anything here.

## Review contract

Identical to the Math- lane contract. The manifest is an evidence sidecar, not a
scientific register. A trusted successful `--run-lean` execution establishes
kernel evidence only for these 31 target declarations and their displayed
hypotheses, relative to Lean's kernel, standard foundations and the pinned
toolchain. Independent review must compare each Lean statement (and each
cross-multiplication) to this scope note and the informal anchor in
`manifest.json`. Review author, provider, family and agent must be explicit and
differ from the author's (`Anthropic` / `Claude` / Cursor cloud agent). Validate
a record with `python3 tools/formal_gate_check.py --alignment review.json`; the
validator checks structure, digests, coverage and lineage, and does not
authenticate that a review happened.

The record contains disposition `ACCEPTED`, `manifest_sha256`, `scope_sha256`
(the hash of this file as bound in the manifest), exactly the 31 targets,
author/reviewer objects with provider/family/agent, and evidence
repository/path/40-character commit/sha256. At publication no independent
alignment review is claimed. A fresh source, scope, toolchain or manifest digest
stales the old review.
