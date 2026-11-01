# Exact formal scope and alignment review — SIDE24 and P15 arithmetic skeletons

Scientific effect: NONE. This package offers 59 exact integer statements. Thirty-one
(namespace `UniversalLaw.Side24`) are the arithmetic skeleton of the SIDE24
coefficient note; twenty-eight (namespace `UniversalLaw.P15`, added 2026-10-10)
are the exact finite arithmetic displayed in four P15 price/cover notes and are
scoped in the section [P15 exact arithmetic](#p15-exact-arithmetic-added-2026-10-10)
below. The SIDE24 note is `Math-` commit
`9d7b6802424fb4715b31999066aafca8ee2f3cca`, `coefficients/side24_v1/PROOF.md`
(10,272 bytes, SHA-256 `c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769`),
and of its published `ENCLOSURE.json` (1,090 bytes, SHA-256
`72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811`). Both sources
are preserved byte-identically under `sources/side24_v1/`. Core Lean's `Rat` is
not kernel-decidable by `decide` (its `DecidableEq` instance does not reduce), so
each rational identity is stated as the equivalent cross-multiplied identity over
`Nat`; the docstring gives the rational form and the independent alignment
review must check that the two agree.

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

## P15 exact arithmetic (added 2026-10-10)

Scientific effect: NONE. Twenty-eight targets in namespace `UniversalLaw.P15`
(modules `UniversalLaw/P15/PriceBoundary.lean`, `RealizedCovers.lean`,
`FullPrice.lean`) restate, as cross-multiplied integer facts proved by `decide`
(one positivity conjunct by core `Nat.pow_pos`), the exact finite arithmetic that
four P15 notes display or immediately imply. Four conjuncts — `1·1 + 1 = 2`,
`3 + 1 = 4`, `16·10²⁰ < 27·(lower endpoint)`, `3·6 − 11 = 7` — are not displayed
in the registered source and are marked as such in their rows and docstrings.
All four notes are pinned at
`Math-` commit `760340e921ac4ceda296b8118da936f1133e956e` and preserved
byte-identically under `sources/p15_v1/`:

| Source id | Original path | Bytes | SHA-256 |
|---|---|---|---|
| `p15-price-boundary` | `frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md` | 2,266 | `498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b` |
| `p15-realized-covers` | `frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md` | 11,467 | `c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9` |
| `p15-full-price` | `frontiers/full_price_20260924/PROOF.md` | 11,352 | `87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9` |
| `p15-price-budget` | `frontiers/price_budget_20260924/PROOF.md` | 7,935 | `3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535` |

The notes' author is OpenAI / ChatGPT; each carries its own disposition
("author-side mathematical derivation", nonauthor review open), which is
transcribed here and not re-derived. The Lean text was written by Anthropic /
Claude (Claude Code session `session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10) at
zero organizational-independence credit. Core Lean's `Rat` is not
kernel-decidable by `decide` (its `DecidableEq` instance does not reduce), so each
rational identity is the equivalent cross-multiplied identity over `Nat`, with
the rational form in the docstring, and the independent alignment review must
check that the two agree. Binomial coefficients use the exact multiplicative recursion
`C(n,k+1) = C(n,k)(n−k)/(k+1)`, ceilings use `(a+b−1)/b`, and the two-coordinate
ground set of the counterexample is enumerated as the four Boolean pairs.

| Targets | Exact coverage | Not established |
|---|---|---|
| `smallest_member_block_size` | `1·1 + 1 = 2`: block size of the demand-one member `a_1 = d_1 = 1`, `X = {x,y}` (the formula `n_i = a_i d_i + 1` is display (P1) of `p15-realized-covers`, not of this note) | The realized-cover family or theorem |
| `demand_one_good_sets`, `demand_one_obstruction` | Of the four subsets of `{x,y}`, the capacity-one filter keeps exactly `∅, {x}, {y}`; its complement is exactly `{X}` | The downset `D` beyond this enumeration; `I_1(D) = D` in general; cover semantics |
| `good_measure_three_quarters` | At `p_x = p_y = 1/2` the four quarter-weights total `4 = 2²` and the good sets total `3`, i.e. `mu_p(D) = 3/4` cross-multiplied | The product-measure semantics of `mu_p`; `min(1, −log mu_p(D)) = log(4/3)`; `log(4/3) < 1/3` |
| `generator_prices_ninths`, `minimum_generator_price_exceeds_third` | At `c_x = c_y = 2/3` the generator prices over `9` are `9, 6, 6, 4` (`1, 2/3, 2/3, 4/9`); their minimum is `4`; `4/9 > 1/3` as `1·9 < 4·3` | That `phi(1/2) = log 2 > 2/3` (admissibility of `c`); that the minimum over subsets of `X` is the exact optimal cover price (the set-theoretic steps); `1/3 > log(4/3)`; **the source's conclusion that the same-palette extension to `c ≤ phi(p)` is false in general** |
| `benchmark_block_and_ground_sizes`, `four_block_supports_fifteen` | `1·408 + 1 = 409`, `6·409 = 2454`, `C(6,4) = 15` | The construction (P1)–(P2), Proposition P1 |
| `minimal_forbidder_counts` | (P10) `6·C(409,2) = 500616`, `15·409⁴ = 419743994415` | That these count the minimal forbidders (Proposition P1) |
| `palette_816_from_two_label_classes`, `palette_815_fails_by_counting`, `whole_ground_chromatic_818` | `408 + 408 = 816`; `2454 − 6 = 2448`, `815·3 < 2448`, `6·408 = 2448`; `2454/3 = 818` exactly | Theorem P2 (cover iff `K ≥ K_H(d)`); that the palette satisfies (P6) and no smaller one does; that a colour class holds at most three vertices; the chromatic number of the ground set |
| `elementary_palette_formula_instances`, `macro_triangle_three_colors`, `ceil_block_over_capacity_instances` | The right-hand side of (P9) evaluated at `(b,s,d_i) = (6,3,408)`, `(6,3,409)`, `(3,1,1)`; `3 + 1 = 4` (not displayed: `s + 1 = 4` with `s = 3`); `⌈(ad+1)/a⌉ = d+1` at five `(a,d)` instances | The formula (P9) itself; that the benchmark's `K_H` is given by (P9); the general ceiling identity; the triangle's chromatic claim |
| `block_occupancy_one_percent`, `low_failure_union_bound_arithmetic`, `cover_price_numerator_and_denominator_positive` | `409·100 = 40900`; (P11) `500616/40900² + 15/100⁴ = 2449227/8180000000 < 3/10000` cross-multiplied; `10000 − 3 = 9997`; `6/40900⁴⁰⁹ > 0` as `0 < 6` and `0 < 40900⁴⁰⁹` (core `Nat.pow_pos`; the power is not evaluated) | The union bound and the independence of block occupancies; that `Q = 1 − mu_p(D)`; the price identity `6/40900⁴⁰⁹` for the six-generator cover; (P4)–(P5), (P8) |
| `rho_star_endpoints_adjacent`, `rho_star_upper_endpoint_below_six_sevenths` | The published 20-digit endpoints of (F3), as integers in `10⁻²⁰` units, are adjacent; `…68·7 < 6·10²⁰` (upper endpoint `< 6/7`) | **That `rho_star = 1/(3 − log(3e − 2))` lies between the endpoints**; `rho_star < 6/7` as a real-number statement; Theorem F (F2); the constant's optimality |
| `sixteen_twentysevenths_below_six_sevenths` | `16·7 < 6·27`; `16·10²⁰ < 27·(lower endpoint)` (not displayed; implied by the displayed numbers) | Theorem (T1) or its domain `p_v ≤ 1/4`; that `16/27 < rho_star` as a real-number statement |
| `minimal_block_size_instances` | `1·2 + 1 = 3`; `2a + 1 ≤ ad + 1` at five `(a,d)` with `d ≥ 2` | The implication for all `a ≥ 1, d ≥ 2`; the binomial tail comparison (F9)–(F11); the sharpness argument |
| `e_rational_bounds_ordering`, `exp_eleven_sixths_taylor_certificate`, `h_star_rational_certificate_constants` | `31967·32 < 87·11760`; `87/32 − 31967/11760 = 11/23520` cross-multiplied; degree-5 Taylor numerator of `exp(11/6)` equals `197·29160 + 26081` over `933120 = 6⁵·5! = 32·29160`; `3·87 − 2·32 = 197`; `3·6 − 11 = 7` (not displayed; the arithmetic behind `h_star > 7/6`) | `e < 31967/11760`, `e < 87/32`, that the partial sum is below `exp(11/6)`, `197/32 < exp(11/6)`, monotonicity of `log`, `h_star > 7/6`, `rho_star < 6/7` as real-number statements |
| `t3_ratio_at_minimal_block`, `t3_bound_finite_instances` | (T3) at `a = 1, n = 3`: `4³ = 4¹·4²`, `4² = 16`, `3³ = 27`, `(4/3)(4/9)¹ = 16/27`; `4^(a+1)·27 ≤ 16·3ⁿ` at seven `(a,n)` with `n ≥ 2a+1`; `4·4ᵃ·27 ≤ 16·3·9ᵃ` for `a = 1..6` | The ratio chain (T3) in general (it uses (T2), `p_v ≤ 1/4` and independence); (T2); Theorem (T1) |
| `explicit_rational_price_above_p` | `2·40900 + 1 = 81801`, `2·40900² = 3345620000`, `3345620000 < 81801·40900` (i.e. `c = p + p²/2 = 81801/3345620000 > p` at `p = 1/40900`) | Admissibility `c ≤ phi(p)`; the price bound (T7); validity of the 816-label cover at these prices |

None of these targets is Theorem F (F2), Theorem (T1), Theorem P2, Proposition
P1, the enclosure (F3) of `rho_star`, any inequality involving `log`, `exp` or
`e`, the probability semantics of `mu_p`, the union bound (P11), `p_star`, the
P15-B amalgamation, or any prize status; every prize claim in this repository
carries `original_prize_closed: false` and nothing here touches it. No Lean
`Prop` stating any of those research claims exists in this package, so their
formal-progress label stays `none`; the targets above are arithmetic
sub-identities whose own label is `proved` at source and `kernel-checked` only
in a trusted receipt. The Layer 0 dispositions of the four notes are unchanged.

## Review contract

Identical to the Math- lane contract. The manifest is an evidence sidecar, not a
scientific register. A trusted successful `--run-lean` execution establishes
kernel evidence only for these 59 target declarations and their displayed
hypotheses, relative to Lean's kernel, standard foundations and the pinned
toolchain. Independent review must compare each Lean statement (and each
cross-multiplication) to this scope note and the informal anchor in
`manifest.json`. Review author, provider, family and agent must be explicit and
differ from the author's (`Anthropic` / `Claude` / the `author.agent` string of
`manifest.json`, which names the Cursor cloud agent of 2026-09-27 for the Side24
modules and Claude Code session `session_01Cz7WZybv8znP64SpPj6sWY` of 2026-10-10
for the P15 modules). Validate
a record with `python3 tools/formal_gate_check.py --alignment review.json`; the
validator checks structure, digests, coverage and lineage, and does not
authenticate that a review happened.

The record contains disposition `ACCEPTED`, `manifest_sha256`, `scope_sha256`
(the hash of this file as bound in the manifest), exactly the 59 targets,
author/reviewer objects with provider/family/agent, and evidence
repository/path/40-character commit/sha256. At publication no independent
alignment review is claimed. A fresh source, scope, toolchain or manifest digest
stales the old review.
