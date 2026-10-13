# SUMMARY
I ran the note's "interesting test" on three kinds of sample. The prediction holds: on the zeta zeros the short bars behave like the quadratic-repulsion (GUE-type) case, with bar density close to ell^0, and clearly not like the smooth-Gaussian case, which gives ell^-1/3. The three samples were: Hardy Z(t) for t in [1e4,1e5] and [1e5,1e6] (1.74M zeros, every zero accounted for); a 1D periodized squared-exponential Gaussian process (1.0M lobes); and CUE with N=128 and N=12 (1.03M lobes each). The fitted exponents follow directly from the gap exponent through the cubic fold: bar length is proportional to gap^3. Short bars for zeta come from close zero pairs, not from regions of small amplitude. Amplitude only slows the approach to the limit for CUE with N=128. None of this bears on RH.

# DETAILS
I ran the note's "interesting test" on three kinds of sample. The prediction holds: on the zeta zeros the short bars behave like the quadratic-repulsion case (bar density ~ ell^0) and not like the smooth-Gaussian case (ell^-1/3). Total compute was about 7 minutes.

NOTATION
- s is the slope of the cumulative distribution of bar lengths: CDF ~ ell^s, so the density goes like ell^alpha with alpha = s − 1.
- Predictions: Gaussian s = 2/3 (alpha = −1/3); quadratic repulsion s = 1 (alpha = 0).
- Gaps are measured with mean spacing 1, and their CDF goes like r^(beta+1). Predictions: beta = 1 (linear repulsion), beta = 2 (quadratic).
- Bar lengths are normalized by their median, written ell/med.
- Main estimator: the maximum-likelihood power-law fit to all values below a cutoff x_hi. 95% confidence intervals come from a bootstrap that resamples whole clusters (B=400): one Gaussian torus, one CUE matrix, or a block of 2000 zeta lobes.

METHOD
- Zeta (rs_zeta.py):
  - Riemann–Siegel formula in numpy: the main sum plus the first three correction terms C0, C1, C2. The correction functions come from mpmath derivatives fitted by Chebyshev series (fit residual ≤ 3e-15).
  - Accuracy against mpmath.siegelz at 60 random points per window: max |error| 3.4e-10 on [1e4,1e5] and 6.5e-9 on [1e5,1e6]. With only one correction term the error is 6e-5 (the extra terms really do improve it).
  - Four spot-checked zeros agree with mpmath.zetazero to ≤ 1.7e-10.
  - Zeros found from a grid with 20 points per mean spacing, plus an extra search wherever |Z| dips toward zero without changing sign. That search found 4 and 52 pairs closer together than one grid step, so they would otherwise have been missed.
  - Counts found: 127,927 and 1,609,077. Both equal mpmath.nzeros(hi) − nzeros(lo) exactly, so no zero in either window is missing.
  - Lobe integrals use 16-point Gauss–Legendre quadrature. Checked against mpmath.quad(siegelz): relative error 3e-11 to 5e-9 on typical lobes, and 1.6e-5 on the smallest lobe (a pair at t ≈ 663318.508 only 0.003 apart).
  - Before building F, Z was put on a unit-mean-spacing scale and divided by sqrt(log(t/2π)). Running on raw Z gives the same exponents.
- Gaussian (gauss_proc.py):
  - This is the 1D case of the project's torus field: weights q_n ∝ exp(−ω²n²/2), ω = 1/M, M = 1000, 500 independent samples.
  - Built by FFT on 2^19 points, with the antiderivative computed exactly in Fourier space. Zeros and F come from cubic Hermite interpolation.
  - Against direct trigonometric sums: lobe-area relative error 6e-11.
- CUE (cue.py):
  - Haar unitary matrices; the real function Z_U(θ) = Π_j 2 sin((θ − θ_j)/2), evaluated as a direct product with no logarithms.
  - Quadrature check, 16 versus 32 nodes: relative difference ≤ 2e-13.
- Barcode (analyze.py): H0 superlevel persistence of the antiderivative, built on the alternating sequence of critical values with the elder rule (union-find). The infinite bar and bars born at the window endpoints are dropped.

RESULTS: SHORT-BAR CDF SLOPE s

| Sample | Lobes / bars | s, ell/med < 0.1 | s, < 0.01 | s, < 0.003 | Log-log fit on [1e-3, 1e-2] |
|---|---|---|---|---|---|
| Gaussian | 998,836 / 498,674 | 0.631 | 0.650 [0.641, 0.659] | 0.657 [0.643, 0.671] | 0.643 [0.631, 0.657] |
| CUE N=128 | 1,032,192 / 501,103 | 0.791 | 0.935 [0.920, 0.951] | 0.971 [0.942, 1.001] | 0.949 [0.926, 0.972] |
| CUE N=12 | 1,032,192 / 386,701 | 0.901 | 0.980 [0.954, 1.007] | 1.019 [0.973, 1.070] | 0.982 [0.948, 1.018] |
| Zeta [1e4, 1e5] | 127,926 / 63,962 | 1.002 [0.977, 1.029] | 1.049 [0.968, 1.145] | 1.001 [0.873, 1.147] | 1.034 [0.921, 1.173] |
| Zeta [1e5, 1e6] | 1,609,076 / 804,536 | 0.947 [0.941, 0.953] | 0.980 [0.959, 1.001] | 0.972 [0.938, 1.009] | 0.978 [0.950, 1.007] |

- Gaussian: local slopes are 0.66–0.68 for ell/med between 1e-4 and 1e-3, consistent with 2/3.
- Zeta [1e5,1e6], cutoff 0.01: alpha = −0.020 [−0.041, +0.001]. The Gaussian value of −1/3 is far outside this interval.

RESULTS: SMALL-GAP SLOPE (cutoff r < 0.2, CDF slope = beta + 1)
- Gaussian: 1.98 [1.97, 2.00]
- CUE N=128: 2.99
- CUE N=12: 3.00
- Zeta [1e4,1e5]: 3.11 [2.91, 3.35]
- Zeta [1e5,1e6]: 2.93 [2.88, 2.98]

For comparison, the GUE expected values at cutoffs 0.4, 0.3, 0.2 and 0.15 are 2.84, 2.91, 2.96 and 2.98. I derived these from the first two terms of 1 − (sin πr/πr)². CUE N=128 matches them: 2.84, 2.90, 2.99, 3.00.

FOLD CHECK
Among lobes with gap r < 0.35, in every ensemble:
- the slope of log(lobe area) against log(gap) is 3.00–3.03;
- the slope of log(peak/local amplitude) against log(gap) is 2.02–2.05.

SHORT BARS: CLOSE PAIRS OR SMALL AMPLITUDE?
For zeta, short bars are not dominated by amplitude fluctuations. For the 8,352 zeta [1e5,1e6] bars below 0.01·median:
- 100% are a maximum paired with an immediately adjacent minimum, and 100% have their gap in the lowest decile of all gaps (median gap 0.143).
- Only 3.8% sit in the lowest decile of local amplitude, which is below the 10% you would expect by chance.
- The average log10 shortfall is −2.70. It splits into: gap −0.87, peak/local amplitude −1.90 (about twice the gap term, as the fold predicts), local amplitude +0.05, shape +0.02.
- Within this set, rank correlation with gap is 0.82 and with local amplitude −0.02.

Comparing at the same lower-tail fraction q = 0.01:
- Zeta [1e5,1e6]: s(lobe area) = 0.9796 and s(gap³) = 0.9793, difference 0.0004 [−0.009, 0.010]. The small shortfall below 1 is inherited entirely from the gap distribution at r ≈ 0.15–0.22; amplitude plays no part.
- CUE N=128: amplitude is a real secondary effect. Short bars are 3.3× over-represented in the lowest-amplitude decile, the amplitude term is −0.40, and the q = 0.01 difference is −0.034 [−0.052, −0.016]. This explains its slow convergence.
- At moderate lengths (0.1 to 1 × median) all five samples bend downward. The order of the bend follows the spread of log-amplitude: zeta [1e4,1e5], then zeta [1e5,1e6], then CUE N=12, then CUE N=128.
- One stray result: zeta [1e4,1e5] at q = 0.003 shows a +0.058 [0.007, 0.119] difference on 384 lobes. With this many comparisons I read it as noise.

INTERPRETATION AND CAVEATS
- **What the result says:** this is exactly the transfer rule in the project's own theorem: Math- reviews/collision_mechanism_20260925/NOTE.md, §D, results (D1)/(D2). There, the density exponent is β − 1 with β = (gap exponent + 1)/3. That gives −1/3 for linear repulsion and 0 for quadratic repulsion.
  - The rule needs a condition on the curvature: small curvature values must be rare enough (q + 1 > β in that note; at q + 1 = β a log(1/ell) correction appears).
  - For zeta this condition concerns how small |ζ'| can be at a close pair. It is not proved. Numerically it does no harm at these heights.
  - So the test confirms the note's warning against carrying the −1/3 exponent over to zeta. The exponent is set by the probability law of the gaps, not by the fold geometry alone.
- **Moderate lengths mislead:** fitting over the upper decade gives misleading in-between exponents. For example, CUE N=128 below 0.1·median gives alpha ≈ −0.21.
- **The GUE link is conjectural:** GUE statistics for zeta zeros are a conjecture (Montgomery's pair-correlation conjecture, with Odlyzko's numerics). The heights here (1e4–1e6) are modest.
- **No bearing on RH:** the barcode only sees real zeros of Z. Zeros off the critical line would show up only as a shortfall in the zero count. The fact that every zero in these windows lies on the line is classical numerics, verified to far greater heights by others.
- **Citations from memory, not web-checked:** Montgomery (1973), Odlyzko, Keating–Snaith's matching N ≈ log(t/2π), Selberg's central limit theorem, finite-height corrections by Bogomolny et al., Platt–Trudgian's RH verification to about 3e12, and the Riemann–Siegel coefficient formulas from Edwards (these last ones were checked numerically against mpmath).

FILES
All under /tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/:
- Scripts: rs_zeta.py, gauss_proc.py, cue.py, analyze.py, matched_tail.py, make_plots.py, check_gauss.py, check_zeta_quad.py
- Summaries: summary.json, matched_tail.json, rs_validation.json, zeta_quad_check.json, *_info.json, analyze.log
- CSVs: ecdf_bars_lobes.csv, ecdf_gaps.csv, local_slopes.csv
- Plots: matplotlib is not installed, so these are hand-written SVGs with PNG copies drawn by Pillow — effective_exponent_bars.svg/.png and effective_exponent_gaps.svg/.png
- Raw data: zeta_A.npz, zeta_B.npz, gauss.npz, cue12.npz, cue128.npz, ecdf_samples.npz

# ARTIFACTS
[
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/rs_zeta.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/gauss_proc.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/cue.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/analyze.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/matched_tail.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/make_plots.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/check_gauss.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/check_zeta_quad.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/summary.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/matched_tail.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/rs_validation.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/zeta_quad_check.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/zeta_A_info.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/zeta_B_info.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/gauss_info.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/cue12_info.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/cue128_info.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/ecdf_bars_lobes.csv",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/ecdf_gaps.csv",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/local_slopes.csv",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/effective_exponent_bars.svg",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/effective_exponent_bars.png",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/effective_exponent_gaps.svg",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/effective_exponent_gaps.png",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/analyze.log",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/zeta_A.npz",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/zeta_B.npz",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/gauss.npz",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/cue12.npz",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/cue128.npz",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats/ecdf_samples.npz"
]