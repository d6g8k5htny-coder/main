# VERDICT
The core mathematics holds up; the problems are in wording, calibration and citations. Approve once findings 1–9 are fixed. My checks are in /tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/review-math/.

I re-derived these by hand, and where marked also with my own numerics, and they are correct:
- **Proposition 1 coefficients.** κ_r = 4C/A − 2B²/A², κ_ℓ = 6B²/A² − 48C/(5A), κ_t = 3B²/A² − 18C/(5A), and the midpoint drift 2B/A. Two derivations agree: root expansion, and the weighted-integral identity with moments r³/6 and r⁵/120. kappa_indep.py checks them on an exact backward-heat polynomial of degree 6 with A < 0 and nonzero v5 and v6 terms; the measured values match to 1e-5 at τ = 1e-7. The zeros are real for τ > 0 whatever the sign of A, and Im z/√(2|τ|) → 1.
- **Proof of Proposition 1.** Parts (a), (b) and (c) are sound, including the elder argument and the symmetry h(−y,−σ) = h(y,σ).
- **Fold constants.** ℓ = (|F'''|/12)r³(1 + O(r²)), k = |A|/6, and 8√2/3.
- **Density transfers.** r^α dr with ℓ = κr³ gives density (1/3)κ^{−β}ℓ^{β−1} with β = (α+1)/3, so ℓ^{−1/3} for α = 1 and ℓ⁰ for α = 2. The mark phase diagram q > 0 / q = 0 / q < 0 matches Math- §D2.
- **GUE numbers.** The R_2 Taylor coefficients, and the two-term Hill expectations 2.84, 2.91, 2.96, 2.98, are reproduced exactly.
- **Theta identities.** The Mellin identity in L, ½(2π²)^{−s/2}Γ(s/2)Z_d(s/2) (via u = 2π/L²), and the Z_4 factorization are correct.
- **Normalization and zero velocity.** H_0 = Ξ(z/2)/8 holds to 1e-47 in a third implementation (indep_ht.py, tanh-sinh quadrature). The exact zero velocity H_zz/H_z = −0.5796306, −0.7549955, −0.4654455 matches the finite-difference values.
- **Both collisions, from my own Newton solve.**
  - Pair 9/10: t* = −1.877909072972535264766839718, z0 = 99.32846930557575094802277. This agrees with the claims job to about 27 digits.
  - Pair 34/35: t* = −0.384561116627981414864807288345, z0 = 223.2757801856155199135, A = −1.76353e-36.
  - κ_r and κ_ℓ match the tables at both collisions.
- **Lobe identity.** It equals −6.31861166707933e-4 on both sides.
- **Citations.** The venues and years I recall for Rodgers–Tao, Polymath15, Platt–Trudgian, Ki–Kim–Lee, CSV, COSV, Odlyzko, SGD, de Bruijn, Newman, Montgomery and Rudnick–Sarnak match the draft. This is from memory, not fetched.
- **Line references.** I spot-checked the project line references in fold_controls.py, fold_structural_universality/PROOF.md, periodized PROOF.md:80-83 and PROTOCOL.md:59-64; all match. The worktree is now at 8977619e, but none of the cited files changed since 03acb7be.

No statement claims RH progress, with one exception: an unconditional sentence that is effectively equivalent to RH (finding 1).

# FINDINGS
[
 {
  "severity": "high",
  "location": "synthesis.assessment (1), bullet 'All collisions occur at t < 0.'",
  "problem": "Stated without qualification, this is close to equivalent to RH. A collision at some t0 > 0 would mean H_t has non-real zeros just before t0, so \u039b \u2265 t0 > 0 and RH fails. The sentence asserts as fact something only the computed sample shows.",
  "fix": "Write 'Every collision we found (lowest 46 zeros) is at t < 0.' The not_worth list already words it this way. Check every summary or PR-body sentence derived from this bullet."
 },
 {
  "severity": "high",
  "location": "synthesis.assessment (3) 'The numerics are decisive at the heights tested'; README line 37 (bottom-line table) 'Numerically decisive'",
  "problem": "The bootstrap intervals cover sampling error only. At the same cutoff 0.01 they exclude the theoretically expected value in two of the three controls: Gaussian \u22120.350 [\u22120.359, \u22120.341] excludes \u22121/3, and CUE N=128 \u22120.065 [\u22120.080, \u22120.049] excludes 0. The \u03b6 [1e5,1e6] interval only just reaches 0 (upper end +0.001). The data are decisive against \u22121/3 for \u03b6, and only consistent with 0 once an estimator bias of a few hundredths is allowed.",
  "fix": "Write 'decisive against \u22121/3; consistent with \u22480 given finite-\u2113 estimator bias of order 0.02\u20130.07 seen in the controls; intervals do not include systematic error'. Show the Gaussian and CUE128 interval misses next to the \u03b6 row."
 },
 {
  "severity": "medium",
  "location": "synthesis.assessment (2) 'New but elementary: \u222b(N_F(L,1)\u22121)L^{\u2212s\u22121}dL = \u2026'; README \u00a74 lines 314-315 'These are new but elementary \u2026 were not searched for in the literature'",
  "problem": "The identity in L is Riemann's Mellin formula after the substitution u = 2\u03c0/L\u00b2, with nothing added. The other \u00a74 items are also one-line restatements: the lobe identity is integration by parts plus the heat equation, and the weight lemma is a Taylor expansion. Calling them 'new' while saying no literature search was done is unsupported.",
  "fix": "Relabel \u00a74 as 'Elementary restatements in the project's variables (novelty not claimed; no literature search)'. Remove 'new' from the assessment."
 },
 {
  "severity": "medium",
  "location": "synthesis.worth_exploring #1 ('fallback is the cumulative statement (D2)') and README \u00a75.3 line 515 / \u00a76 citing collision_mechanism (D2)",
  "problem": "The cumulative statement (D2) as written in NOTE.md was marked AMEND_REQUIRED in reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md:60,184-197. The reviewed replacement is reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md, which restores the ratio hypothesis h/(\u03bar^m) \u2192 1. Citing the unamended (D2) cites a statement whose constant can be wrong. Separately, the drafts use equation labels (D1)/(D2) and subsection labels \u00a7D1 (counterexample) / \u00a7D2 (phase diagram) interchangeably, which invites confusion.",
  "fix": "Cite CUMULATIVE_TRANSFER_CORRECTION.md for the cumulative fallback, and note which statements the PR25 review accepted: (D1), the counterexample and the phase diagram. Write 'equation (D1)' versus 'subsection \u00a7D2' explicitly."
 },
 {
  "severity": "medium",
  "location": "README \u00a71.3 lines 97-99 ('Theta transformation \u2026 The primary source is Riemann 1859')",
  "problem": "Misattribution. The theta transformation predates Riemann (Poisson, Cauchy, Jacobi; from memory). Riemann 1859 itself cites Jacobi's Fundamenta Nova for it. Riemann's contribution is using it to derive \u03be(s) = \u03be(1\u2212s), which is how the synthesis correction #10 correctly words it.",
  "fix": "Write: 'The transformation is classical (Jacobi; Poisson summation). Its use to prove \u03be(s)=\u03be(1\u2212s) is Riemann 1859 (second proof). Mumford is expository.' Mark it as from memory."
 },
 {
  "severity": "medium",
  "location": "README line 481 'Small-gap CDF slope (= \u03b2 + 1, cut r < 0.2)' vs README \u00a72 '\u03b2 = (\u03b1+1)/m'; synthesis corrections_to_note #14 (\u03b2 = gap exponent, '\u03b2 = 2') vs worth_exploring #1 (\u03b2 = 1)",
  "problem": "\u03b2 has two meanings. Under \u00a72's definition (Math- \u00a7D), GUE has \u03b1 = 2 and \u03b2 = 1, so 'slope = \u03b2 + 1' would predict 2, but 3 was measured. The header only works if \u03b2 means the gap exponent. In the formula in #14, '\u03b2 = 2' collides with the '\u03b2 = 1' cited from \u00a7D for the same GUE case.",
  "fix": "Use \u03b1 for the pair-intensity or gap exponent everywhere: gap CDF slope = \u03b1 + 1, and \u03b2 = (\u03b1+1)/m only in the Math- sense. Rewrite #14 as \u03bd(\u2113) = (c/3)E[k^{\u2212(\u03b1+1)/3}]\u2113^{(\u03b1\u22122)/3}."
 },
 {
  "severity": "medium",
  "location": "synthesis corrections_to_note #17 ('Under RH every fold event is at t < 0, accumulating at 0\u207b'); README line 385 ('If RH holds, close pairs at greater height collide ever closer to t = 0\u207b [Rodgers\u2013Tao]')",
  "problem": "Overstated. RH does not exclude a multiple zero of \u039e, which would be a fold exactly at t0 = 0, so RH gives only t \u2264 0. Accumulation at 0\u207b needs, beyond Rodgers\u2013Tao's \u039b \u2265 0, simple zeros at t = 0 and no zeros escaping to infinity, and is not proved there. 'Ever closer' is the Lehmer-pair heuristic, not a Rodgers\u2013Tao theorem.",
  "fix": "Write: 'Under RH, fold events occur only at t \u2264 0 (t = 0 only if \u039e has a multiple zero). Heuristically (Lehmer-pair mechanism), collision times approach 0 from below as height grows.' Drop the bare [Rodgers\u2013Tao] tag or mark it as heuristic."
 },
 {
  "severity": "medium",
  "location": "README \u00a76 Conjecture Z (lines 536-541) and synthesis worth_exploring #2",
  "problem": "As stated the conjecture is ill-posed. \u03b6 is deterministic, so 'expected short-bar density' has no meaning. On any fixed window the empirical density is 0 below the smallest bar, so \u2113 \u2192 0 is vacuous. With c(T) free to drift, '\u224d c(T)\u2113\u2070' asserts almost nothing. Separately, the decision rule 'coefficient within the CUE(N) band' ignores the arithmetic factors that appear in the analogous moment conjectures (HKO a(k), from memory). A 'fail' on unnormalized bars would then be uninterpretable.",
  "fix": "Define the empirical bar measure on [T, T+H] normalized per zero. State the limit with order: T \u2192 \u221e first, then \u2113 in a range [\u2113_min(T), \u2113_0] with \u2113_min(T) \u2192 0. Make the frozen test about amplitude-normalized bars. Either say the coefficient comparison expects an arithmetic correction, or restrict it to the exponent."
 },
 {
  "severity": "medium",
  "location": "README \u00a75.3 lines 514-516 ('This is exactly the transfer rule in Math- collision_mechanism \u00a7D'); synthesis assessment (3) 'This is exactly the project's own transfer theorem'",
  "problem": "(D1) is a theorem about intensity measures of random models under hypotheses 1\u20134. For the deterministic \u03b6 none of these is defined or verified: no Palm/intensity measure, no domination, no uniform comparison. Calling the \u03b6 result 'exactly' that theorem promotes a heuristic analogy to a theorem application.",
  "fix": "Write 'consistent with the (D1) prediction for \u03b1 = 2, m = 3 (heuristic for \u03b6; for CUE it is the theorem target T-CUE)'."
 },
 {
  "severity": "low",
  "location": "synthesis corrections_to_note #18, last line ('None of these is theta-specific beyond \u03a6 > 0, evenness and decay'); README \u00a71.2",
  "problem": "Misleading about the literature. de Bruijn's lemma and Newman's monotonicity are kernel-generic. Rodgers\u2013Tao uses arithmetic input about \u03b6's zeros (Montgomery pair-correlation estimates and zero counting, per its abstract). Ki\u2013Kim\u2013Lee uses \u03b6-specific zero-distribution facts (from memory). The Rodgers\u2013Tao energy argument is also a lower-bound proof technique, not a global constraint that rules out configurations for t > 0.",
  "fix": "Write: 'de Bruijn and Newman hold for broad kernel classes; Ki\u2013Kim\u2013Lee and Rodgers\u2013Tao use \u03b6-specific zero information (arithmetic, not theta structure).'"
 },
 {
  "severity": "low",
  "location": "synthesis.assessment (1) 'Two real collisions were reproduced \u2026 two independent implementations agreeing to about 27 digits'; not_worth 'reproduces a 1990s bound'",
  "problem": "Only the 9/10 collision was cross-checked by two implementations (claims/collision_point.txt). The 34/35 collision came from the fold job alone. 'Reproduced' also implies these were literature values. The \u22120.385 match is unconfirmed: the README rightly hedges it, and, from memory, the 1991 Csordas\u2013Ruttan\u2013Varga bound \u22120.0991 is stronger than the 1992 Norfolk\u2013Ruttan\u2013Varga \u22120.385, so conventions or pairs need checking.",
  "fix": "Write 'computed', not 'reproduced'. This review's independent tanh-sinh implementation (review-math/indep_ht.py, c910.out, c3435.out) now confirms both collisions to 27\u201330 digits; cite that if desired. Replace 'reproduces a 1990s bound' with 'numerically coincides with the reported \u22120.385 bound (pair used not confirmed)'."
 },
 {
  "severity": "low",
  "location": "README \u00a71.2 line 66 ('\u03a6 is even and positive (termwise, since 2\u03c0 > 3)')",
  "problem": "Termwise positivity holds only for u \u2265 0. The n = 1 term is negative for e^{4u} < 3/(2\u03c0), i.e. u < \u22120.185. Positivity on u < 0 follows from evenness.",
  "fix": "Write: 'positive for u \u2265 0 termwise (2\u03c0 > 3), hence everywhere by evenness'."
 },
 {
  "severity": "low",
  "location": "README \u00a71.2 line 73 (zero dynamics dz_j/dt = 2\u03a3 1/(z_j \u2212 z_k))",
  "problem": "The sum does not converge absolutely; it must be summed symmetrically. The 300-zero ODE values (\u22120.579717 \u2026) are truncation-limited at about 1.5e-4. The exact velocity is H_zz/H_z.",
  "fix": "Add 'summed symmetrically, e.g. \u017c_j = 2[1/(2z_j) + \u03a3_{k\u2260j} 2z_j/(z_j\u00b2 \u2212 z_k\u00b2)] over positive zeros'. Quote the exact H_zz/H_z values (\u22120.5796306, \u22120.7549955, \u22120.4654455, which match the finite differences)."
 },
 {
  "severity": "low",
  "location": "README \u00a71.3 lines 103-110; synthesis correction #12",
  "problem": "Two imprecisions. First, Davenport\u2013Heilbronn concerns positive-definite integral binary quadratic forms with class number > 1, so it does not apply to the project's \u2124^d forms. Second, Z_4 does have an Euler product, so it is not an example of 'without an Euler product' (the wording of claims 4.6). It shows that a theta functional equation does not force zeros onto the symmetry line for an imprimitive, shifted product. The general-d Mellin identity also lacks its domain, Re w > d/2 (Re s > d in \u00a74.1).",
  "fix": "Add 'binary', note that Z_4 is an imprimitive product, and state the domains Re w > d/2 and Re s > d."
 },
 {
  "severity": "low",
  "location": "synthesis worth_exploring #1 ('main work is \u00a7D hypotheses 3\u20134') vs README \u00a76 line 567 ('hypotheses 1\u20134')",
  "problem": "The two drafts disagree. All four (D1) hypotheses are needed, including domination (1\u20132). Hypothesis 4 (fixed-constant comparison) is the one most likely to fail near triple coincidences. The cumulative fallback still needs the envelope and the retained ratio hypothesis.",
  "fix": "Use 'hypotheses 1\u20134' in both, and name the corrected cumulative theorem's hypotheses for the fallback."
 },
 {
  "severity": "low",
  "location": "README \u00a74.3 line 341 and synthesis worth_exploring #4 ('coefficient picks up a w^{-1}-weighted average')",
  "problem": "In general the weight is w^{\u2212\u03b2} with \u03b2 = (\u03b1+1)/m: w^{\u22121} only for \u03b1 = 2, and w^{\u22122/3} for the Gaussian \u03b1 = 1. Also, with x0 at the midpoint the error is O(r\u00b2), not just O(r).",
  "fix": "Write 'w^{\u2212\u03b2}-weighted (w^{\u22121} for GUE-type repulsion)'."
 },
 {
  "severity": "low",
  "location": "README \u00a74 line 342 heading 'The lobe area is a time derivative'; README lines 377-380 'Later events'; synthesis/README 'CUE N=128 \u2026 explained by small-amplitude marks'",
  "problem": "Loose wording. The lobe area is a weighted integral of \u2202_tH, not a time derivative of anything. 'Later events' are at more negative t, i.e. earlier in t. The small-amplitude 'explanation' is a post-hoc reading of data that were inspected first.",
  "fix": "Use 'The lobe area as a weighted integral of \u2202_tH', 'further back in t', and 'consistent with small-amplitude marks'."
 }
]