# VERDICT
The numerics are sound. Nothing I found overturns a headline conclusion. Several readings go further than the data, and a few numbers need correcting. Scratch dir: /tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/review-numerics/ (scripts and .out logs; the same review files are also listed in the findings).

What I re-ran and confirmed:

**Fold numerics (sound).**
- Collision points:
  - An independent composite Gauss–Legendre quadrature (2880 nodes, dps 90) evaluated at the published (x0, t*) gives Newton corrections of at most 3e-44 in x and 3e-46 in t.
  - This holds for both collisions, including (34/35), which had previously been cross-checked only by the same trapezoid code.
  - refine_check.py re-run: the shifts are 2.6e-41 and 1.2e-40 in t*, and 1.4e-39 and 2.5e-39 in x0.
- κ_r, κ_ℓ, κ_t and the midpoint drift 2B/A match to 10 digits on random heat polynomials v2–v7, using Richardson extrapolation.
- The fold relative errors are about τ^1.000000, within 6e-6 to 3.5e-5 of κτ.
- The window counts are valid confirmations. As t decreases, the number of real zeros in a fixed window can only drop (no births), unless zeros cross the window edges.

**Stats numerics (sound).**
- Barcode: analyze.barcode matches a brute-force elder-rule computation on 400 random multi-path cases and on an analytic test function. Max difference is 4e-16.
- Riemann–Siegel: fresh error is 2.6e-9 against siegelz.
- Zero completeness: the zero sets are complete.
  - Exact sub-window counts match (538/538 and 573/573).
  - 52 within-cell close pairs were recovered, against a GUE expectation of 55.1 (and 4 against 4.4 for zeta A).
  - CUE eigenangles are exact.
- Gaussian control: of the 17 flagged cells, 5 are real hidden pairs (6.0 expected). Adding them shifts s by −0.0014 at cut 0.01 and −0.0028 at cut 0.003, so this is immaterial.

**The main conclusion stands.** The zeta and CUE short-bar exponent is clearly not −1/3; the difference of about 0.33 is far larger than any error source.

**Where the write-up overreaches:**
1. **"Decisive ≈0" and the CIs.** The CIs are sampling-only. The cut bias, calibrated on the controls, is −0.017 (Gaussian, whose CI at cut 0.01 excludes its known truth 2/3), −0.020 (CUE N=12) and −0.065 (CUE N=128). "≈0" is consistent with the data but not pinned down.
2. **"3.8% vs 10% chance".** The baseline is wrong. All close gaps (r<0.1) also show 4.0% with this local_sigma, mostly because of neighbour level rigidity. The correct reading is that short bars show no extra amplitude selection.
3. **"Zeta behaves like CUE N≈log(T/2π)".** At moderate r the zeta gap shape does not match CUE N=12.
   - Zeta B's Hill slope at cut 0.4 is 2.896 [2.876, 2.916], against CUE N=12 at 2.86 and GUE at 2.84.
   - It falls between CUE N=2 (2.92) and N=3 (2.87). Zeta A is 3.05, above even N=2.
   - This matches the finite-height effective N of Bogomolny et al. 2006, about 2–3 here (from memory).
   - So any coefficient-level comparison needs a matching rule declared in advance.
4. **T-CUE coefficient comparison.** The CUE numerics cut the circle at the first eigenangle, which gives a non-periodic F, and drop the endpoint lobes. Those lobes hold 11.25% of short lobes, not 2/12, because the wrap-around gap is length-biased. Predicted c_N must use the same convention as the numerics.
5. **Fold convergence is tautological.** Convergence to τ = 1e-12 holds for any analytic solution of the backward heat equation, including the quadrature approximant. It confirms the κ algebra, not anything specific to ζ.
6. **Mark condition.** "The mark condition does no harm numerically" cannot be tested at this resolution. Mark effects of |Δs| ≈ 0.02 appear in CUE N=12 and borderline in zeta B at q = 0.003.

Plus minor reporting nits. Nothing here bears on RH.

# FINDINGS
[
 {
  "severity": "medium",
  "location": "synth/README_draft.md \u00a75.3 table, 'Reading' paragraph, and synthesis assessment item (3) ('numerics are decisive', density exponent \u22120.020 [\u22120.041, +0.001])",
  "problem": "The bootstrap CIs cover sampling error only. The estimator has a systematic finite-cutoff bias, visible in the controls at cut 0.01:\n- Gaussian s = 0.650 [0.641, 0.659], whose CI excludes the theoretical 2/3 (bias \u22120.017; \u22120.018 after adding the missed hidden pairs);\n- CUE N=12: 0.980 (\u22120.020);\n- CUE N=128: 0.935 (\u22120.065).\n\n'Decisive' holds for excluding \u22121/3. It does not pin the exponent at 0. The data also cannot separate \u2113^0 from a mark-driven \u2113^q with |q| \u2272 0.04.",
  "fix": "Say 'decisive against \u22121/3 (difference \u22480.33)'. For the zeta value, say 'consistent with 0 after the cut-0.01 bias of about \u22120.02 seen in the Gaussian and CUE N=12 controls'. Report CIs as sampling-only, with the control-calibrated bias stated next to them. Do not describe \u22120.020 [\u22120.041, +0.001] as a measurement of 0."
 },
 {
  "severity": "medium",
  "location": "README_draft \u00a75.3 'What makes zeta's short bars short' ('only 3.8% are in the lowest local-amplitude decile (10% would be chance)'); synthesis assessment (3) and worth_exploring #2",
  "problem": "10% is the wrong baseline. local_sigma is the median peak of the 12 nearest lobes, and level rigidity makes a close pair's neighbours larger.\n\nConditioning only on the gap, with no bar lengths, the same statistic is:\n- zeta B: 4.0% for gap < 0.1 and for gap < 0.2;\n- CUE N=12: 6.1\u20136.3%;\n- Gaussian: about 9%.\n\nA far-neighbour proxy, skipping the nearest 6 lobes on each side, gives 8.0\u20138.1% for zeta B. So roughly half the deficit is the definition and half is a real correlation.\n\nThe 3.8% therefore means short bars are not selected for low amplitude beyond what their small gaps imply. It does not mean amplitude is anti-selected. (review-numerics/localamp_artifact.out)",
  "fix": "Replace the baseline with the gap-conditioned rate: 'Short bars lie in the lowest local-amplitude decile at 3.8%, the same as all close gaps (4.0%), so bar shortness adds no amplitude selection.' CUE N=128 is the contrast: 33% against 9.5% for close gaps, which supports the mark reading there."
 },
 {
  "severity": "medium",
  "location": "README_draft \u00a75.3 small-gap table (juxtaposed with 'two-term GUE expectations 2.84, 2.91, 2.96, 2.98'); \u00a76 Conjecture Z ('coefficient matching CUE at N \u2248 log(T/2\u03c0)'); worth_exploring #2; \u00a77 item 2",
  "problem": "Zeta's short-gap shape at moderate r is significantly off GUE and off CUE N=12.\n\nGap Hill slope at cut 0.4:\n- zeta A: 3.05 [2.98, 3.13];\n- zeta B: 2.896 [2.876, 2.916];\n- GUE two-term: 2.836;\n- CUE N=12: 2.859 [2.837, 2.880].\n\nMy own CUE simulation at cut 0.4 gives N=2: 2.924, N=3: 2.872, N=4: 2.864, N=6: 2.851, N=12: 2.855, so zeta B lies between N=2 and N=3. This matches the Bogomolny\u2013Bohigas\u2013Leboeuf\u2013Monastra (2006) effective dimension for spacing statistics, N_eff = log(T/2\u03c0)/\u221a(12\u039b) \u2248 2.2\u20132.8 here (formula from memory).\n\nKeating\u2013Snaith N \u2248 log(T/2\u03c0) is the matching for value distribution and moments. The gap part and the amplitude part of a short-bar coefficient therefore match CUE at very different N. 'Coefficient within the CUE(N) band' is ill-posed until a matching rule is fixed. Exponents are unaffected. (review-numerics/cue_gap_shape.out)",
  "fix": "State that the zeta windows agree with CUE in the leading exponent only, not in gap shape or coefficients. In the frozen protocol for worth_exploring #2, declare in advance which N (or which separate N for gaps and for amplitudes) defines the CUE reference, or compare amplitude-normalized bars against a zeta-calibrated reference. Cite Bogomolny et al. 2006 as from memory, pending check."
 },
 {
  "severity": "medium",
  "location": "stats/cue.py plus analyze.barcode convention, against the T-CUE target (README_draft \u00a76, worth_exploring #1 'coefficient-level comparison of predicted c_12 and c_128 with the existing CUE data')",
  "problem": "The CUE numerics differ from the T-CUE target, which uses F_U on a fixed arc.\n- Each circle is cut at \u03b8_1, the first eigenangle after angle 0.\n- F = \u222bZ_U is treated as a path. F is not periodic, because Z_U's constant Fourier mode is nonzero.\n- Bars born at the endpoints are dropped, so the two endpoint-adjacent lobes can never give short bars.\n\nThose two lobe positions are not typical. The wrap-around gap is length-biased: mean 1.178, P(gap<0.2) = 0.0012 against about 0.0087. The adjacent positions have more short gaps (0.011). The excluded lobes therefore hold 11.25% of short lobes, not 2/12 = 16.7%.\n\nThe exponent is unaffected: lobe Hill 0.977 for all lobes against 0.973 for interior lobes, and gap Hill 3.003 against 3.003. At N=12 the coefficient is shifted by roughly 10%.",
  "fix": "Before any coefficient comparison, recompute the CUE bars under exactly the T-CUE convention: a fixed deterministic arc with explicit end handling, or the full-circle barcode. Alternatively, derive c_N for the 'cut at first eigenangle' convention. Record the convention in the packet."
 },
 {
  "severity": "low",
  "location": "README_draft \u00a75.2 'How well the theory fits' and \u00a73 'Checks'; synthesis (1) ('errors fall exactly like \u03c4, down to \u03c4 = 1e\u221212')",
  "problem": "The computed H\u0303 is a finite sum \u03a3 W_k e^{t u_k\u00b2} cos(z u_k), and is itself an exact analytic solution of \u2202_tH = \u2212\u2202_z\u00b2H. The \u03c4^1 convergence and the \u03ba\u03c4 agreement down to \u03c4 = 1e\u221212 would therefore hold even if H\u0303 approximated H_t poorly.\n\nThese runs validate the \u03ba algebra and the code. Their \u03b6-specific content is only (t*, x0, A, B/A, C/A) and the t = 0 rows. The algebra is independently right: on random heat polynomials v2\u2013v7, \u03ba_r, \u03ba_\u2113, \u03ba_t and the midpoint drift all match to 10 digits (review-numerics/kappa_indep.out).",
  "fix": "Add one sentence: 'The small-\u03c4 convergence holds for any real-analytic backward-heat solution, including the quadrature approximant; it checks the \u03ba formulas and code. Agreement with \u03b6 rests on the quadrature accuracy checks (H_0 = \u039e/8, independent quadratures).'"
 },
 {
  "severity": "low",
  "location": "README_draft \u00a75.1 ('agree \u2026 to about 27 significant digits in t\u2080 and 29 in z\u2080'; 'moves t* by \u2264 10\u207b\u2074\u2070 and x\u2080 by \u2264 10\u207b\u00b3\u2079'); fold/ has no refine_check log",
  "problem": "Two issues.\n\nFirst, the 27-digit agreement reflects the claims run (624 Gauss\u2013Legendre nodes, dps 60). It is not the accuracy of the published points. An independent Gauss\u2013Legendre evaluation (2880 nodes, dps 90) at the published (x0, t*) gives Newton corrections \u2264 3.3e\u221244 in x and \u2264 2.6e\u221246 in t for (34/35), and 2.8e\u221244 / 1.5e\u221245 for (9/10). The (34/35) collision previously had only a same-code cross-check.\n\nSecond, refine_check.py had no saved output. Re-run, it gives \u0394t* = 2.6e\u221241 and 1.2e\u221240, and \u0394x0 = 1.4e\u221239 and 2.5e\u221239. These slightly exceed the stated bounds.",
  "fix": "Update to: 'independent quadratures agree to about 1e\u221243 for both collisions'. Correct the bounds to '\u0394t* \u2264 1.2e\u221240, \u0394x0 \u2264 2.5e\u221239'. Commit review-numerics/refine_check.out and indep_collision.out (or regenerate them) and list them in MANIFEST.json."
 },
 {
  "severity": "low",
  "location": "README_draft \u00a75.3 Matched lower tail and 'Reading' ('needs E[\u03ba^{\u22121}] < \u221e \u2026 numerically it does no harm at these heights'); synthesis ('CUE N=128 converges slowly; explained by small-amplitude marks')",
  "problem": "The matched-tail differences s(area) \u2212 s(gap\u00b3) are not confined to CUE N=128:\n- CUE N=12, q=0.01: \u22120.019 [\u22120.032, \u22120.005] (significant);\n- zeta B, q=0.003: \u22120.014 [\u22120.031, +0.0005] (borderline);\n- zeta A, q=0.003: +0.058 [0.007, 0.119].\n\nThe matched-tail test detects mark effects only of size |\u0394s| \u2273 0.02 at q \u2265 0.001. It cannot verify E[\u03ba^{\u22121}] < \u221e, and cannot exclude \u2113^q with small negative q.\n\nFor CUE N=128 the mark reading is supported: short bars fall in the low-amplitude decile at 33%, against 9.5% for close gaps. But 'explained' overstates the evidence; the difference shrinks to \u22120.024 [\u22120.058, +0.006] at q = 0.003.",
  "fix": "Say: 'No mark effect larger than about 0.02 in s is detected for zeta at q = 0.01. Small effects (about 0.02) appear in CUE N=12, and larger ones in CUE N=128, consistent with the \u00a7D2 mark mechanism at finite \u2113. The admission condition itself is untested.'"
 },
 {
  "severity": "low",
  "location": "stats/gauss_proc.py (hidden_cells flagged, not resolved); README_draft \u00a75.3 Samples / Gaussian ('17 cells were flagged as possible hidden pairs')",
  "problem": "The 17 flagged cells were never resolved, so any real hidden pairs are missing from the Gaussian control. Resolving them with the cubic Hermite interpolant gives 5 real pairs (expected 6.0 from the empirical r\u00b2 law at h = 0.0038), with unfolded gaps 0.0012\u20130.0024.\n\nAdding the missing tiny bars moves:\n- the bar Hill s from 0.6500 to 0.6486 (cut 0.01) and from 0.6567 to 0.6538 (cut 0.003);\n- the gap Hill from 1.982 to 1.980 (cut 0.2).\n\nThe effect is immaterial, but it pushes the cut-0.003 Gaussian estimate slightly further below 2/3. (review-numerics/gauss_hidden.out)",
  "fix": "Resolve the hidden pairs inside gauss_proc.py with a Hermite subgrid search. Otherwise, report: '5 hidden pairs resolved after the fact; effect on s \u2264 0.003'."
 },
 {
  "severity": "low",
  "location": "README_draft \u00a75.3 (CIs); stats/analyze.py cluster definition for zeta (blocks of 2000 lobes)",
  "problem": "log|\u03b6| is approximately log-correlated, so 2000-lobe blocks are not independent at the amplitude level. The block-bootstrap CIs may therefore be somewhat optimistic for amplitude-sensitive statistics (coefficients, mark diagnostics). Short-gap counts decorrelate quickly, so the exponent CIs are probably little affected.",
  "fix": "Add a sensitivity check with block sizes 500, 2000 and 20000 (cheap, using the existing npz files). Report the coefficient CIs from the largest block size."
 },
 {
  "severity": "low",
  "location": "claims/check_Z_persistence.py and README_draft \u00a75.4 (zero counts compared only to the Riemann\u2013von Mangoldt main term); claims/pers1d.py endpoint convention",
  "problem": "First, the small Hardy-Z windows compare zero counts only with the main term (228.0 and 213.2). The exact counts from mpmath.nzeros are 228 and 214, and they match the found zeros. The main term is not a completeness check.\n\nSecond, pers1d keeps bars born at the first and last critical point. stats/analyze drops them. This is a 1\u20132 bar inconsistency per window.",
  "fix": "Cite the exact nzeros counts (228/228, 214/214) in \u00a75.4. Use the same endpoint convention in both barcode codes, or state the difference."
 },
 {
  "severity": "info",
  "location": "stats/analyze.py barcode(); stats/rs_zeta.py; zeta zero completeness",
  "problem": "Confirmed correct, no defect found.\n- barcode() matches a brute-force elder-rule computation on 400 random multi-path alternating sequences, with 0 mismatches, and on an analytic test function, with max difference 4.4e\u221216 (review-numerics/test_barcode.out).\n- Riemann\u2013Siegel error at 14 fresh points, including the midpoint of the closest pair (t \u2248 663318.5098), is \u2264 2.6e\u22129.\n- Exact sub-window counts match: [500000, 500300] 538/538 and [999700, 1e6] 573/573.\n- Within-cell close pairs: 52 recovered against a GUE expectation of 55.1 for h = 0.05 (4 against 4.4 for window A). Gaps < 0.05: 205 against a GUE estimate of 220.6.\n- No missed close zeros bias the zeta short-gap tail. One minor point: the Chebyshev 'fit residual \u2264 3e\u221215' is measured on the fit nodes. The end-to-end siegelz check is the real validation.",
  "fix": "Optionally add the near-miss expectation check (observed against N\u03c0\u00b2h\u00b3/36) and the brute-force barcode test to test_numeric.py and test_exact.py."
 },
 {
  "severity": "info",
  "location": "fold/track.py first-collision claims; README_draft \u00a75.2",
  "problem": "The claims are supported, with stated caveats.\n- The tracker uses DT = 0.01 stepping on critical-point values and aborted at t \u2248 \u22121.355 ('ordering violated'). Event lists are therefore complete only down to that t.\n- Direct counts (46 at t = 0 and \u22120.38, 44 at \u22120.39) are a genuine confirmation. In backward time the real-zero count in a window can only decrease, except through boundary flux, and Proposition 1 gives no births as t decreases.\n- Zero 46 sits 0.5 from the z = 270 edge, but moves inward (269.513 at t = 0, 269.496 at \u22120.38), so there is no boundary flux.",
  "fix": "Add the monotonicity remark as the reason the counts confirm the 'first collision' claims, and note the edge distance of zero 46."
 }
]