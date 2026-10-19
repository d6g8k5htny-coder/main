# Evidence for C8's review of the non-Gaussian smoothing-admission proof (slice H0)

This folder holds the checks behind `C8-NG-XPROV-REVIEW-H0-20261019`. It is a provider-distinct nonauthor review by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8): Dylan Roy — delegated AI review. The canonical pickup is [Math-#476/6275564531](https://github.com/d6g8k5htny-coder/Math-/pull/476#issuecomment-6275564531), and the verdict is posted in the same thread. Credit 0. Scientific effect NONE.

## Source reviewed

- `frontiers/nongaussian_smoothing_admission_20261017/PROOF.md` in Math-: blob `9df1a2c4`, 38,656 B, SHA-256 `322195ff6693f17047d9ec60c52f8178f5f3a215c4f4cba42dd342ff59dbc116`.
  - It landed in Math-#476 as `4e722922`.
  - It is unchanged on Math- main `b6fc0e64` and `4eb86ed3`.
- **Scope:** the p = 0 (superlevel-H0) theorem, d ≥ 1, and every step it uses.
- **Excluded:** steps used only for p ≥ 1, the §12 cluster estimate and Poisson law, and the A4 and finite-L packets.
- **Imports** were checked against the landed GAD source (SHA-256 `8d992e19…faf319`) and the finite smooth incidence appendix (`b1b928ce…`).

## Order of work

1. **19:31Z, 19 October.** C8 read the packet README and the first 25 lines of `REVIEW.md`, which give its verdict and reviewer identity.
2. **To 19:40:34Z.** C8 read PROOF.md and the GAD sections it uses, and ran `ng_controls.py`. It then wrote `OWN_FINDINGS_NG_H0_before_REVIEW.md` (SHA-256 `995bb81a2d071f501e8d7d7380f0885b5e8b70059821a5859fad51ab8d8952f0`). That file's simulation line says PENDING. The file is kept unchanged.
3. **19:41–20:04Z.** The first Monte Carlo runs: ε = 0 and ε = 1, and the covariance-matched Gaussian at ε = 1. Also the kernel check of κ.
4. **19:58Z.** C8 read `REVIEW.md` §§2, 7, 9, 13 and 14. **About 20:45Z:** it read §§1, 3–6, 8 and 10. It did not read §§11–12, which are outside this slice.
5. **20:32–20:42Z.** The ε = 2 and ε = 3 runs, and the matched Gaussian at ε = 3. A first launch at 20:05Z was lost when the container restarted.
6. **20:45–21:00Z.** The 16384-point rerun at ε = 3, the cubic-jet diagnostic and the grid-loss computation.

The Monte Carlo is numerical and does not depend on the native review's text.

## Algebra controls

`ng_controls.py` (sympy 1.14.0) writes `ng_controls_log.txt`:

| Control | What it checks |
|---|---|
| C1 | \|det dU/dO\| = 12 r^−(d+3) and the targets v_r, for d = 1, 2, 3 |
| C2 | The midpoint-map determinant is 1 |
| C3 | The block-determinant identity (17) |
| C4 | r^−2 \|dr/dk\| = (1/3) ℓ^−1/3 k^−2/3; the factors 4 and 144; 144·12^(−7/3) = 12^(−1/3) |
| C5 | E[(X⁴−3)³] = 9504 and E[(X²−1)³] = 8 |
| C6 | After the b-integral the large-k exponent is 1/3 − M, integrable iff M > 4/3 |
| C7 | N(1,1) cone masses: E[A²; A>0] = 1.925 and E[A²; A<0] = 0.075 |

## End-to-end Monte Carlo of (6) in d = 1

**Model.**
- The circle X = R/2πZ, with V = 2π, and F = G + ε(H² − 1).
- G and H are independent stationary Gaussian cosine series with 26 modes (n = 0…25). Their weights are ∝ exp(−n²/2s²), with s = 5 for G and s = 4 for H, normalised to unit variance.
- These are finite trigonometric series, so G lacks the dense span the theorem assumes. The test checks the coefficient formula, which uses only finite-jet densities. It is not an instance of the theorem's hypotheses.

**Prediction** (`kappa.py`).
- In d = 1, (6) gives C_0 = 12^(−1/3) κ, with κ = p_(F′,F″)(0,0) E[|F‴|^(4/3) | F′ = F″ = 0]. Then E N((0,t]) ~ V (3/2) C_0 t^(2/3).
- κ is computed through the literal mixture kernel (9). Condition on the jets of H first; given them, (G′, G″, G‴) is Gaussian.
- `kappa_kernel.py` checks κ independently, by kernel smoothing of joint jet samples (`kk_eps0.json`, `kk_eps1.json`).
- The Gaussian value with F's covariance uses the exact spectral moments of K_G + 2ε² K_H² (`gauss_matched.py`).
- **Correction.** The fields `kappa_cov_matched_gauss`, `pred_coeff_gauss` and `ratio` in `kappa.py` use sample variances in L6 − L4²/L2, which amplifies Monte Carlo error. They are superseded by the exact values below. That includes the 4.161 in the findings note, which should be 4.125.

**Bar counts.** `simulate.py`, `simulate_bisect.py` and `simulate_small.py` count bars of F; `gauss_matched.py` counts bars of the matched Gaussian.
- Critical points are sign changes of the exact F′ on a J-point grid, refined on the exact series.
  - `simulate.py` refines with Newton; the others use 30 bisection steps.
  - `newton_vs_bisect_log.txt` shows identical counts on a common seed.
- Critical values are exact.
- Superlevel-H0 persistence on the circle is exact: union-find over maxima with the elder rule.
- The rate is N((0,t]) / (R t^(2/3)).

**Cubic-jet diagnostic** (`inflect.py`), on the same fields.
- At every inflection point with F′F‴ < 0, the cubic jet has a max/min pair of height ℓ = (2/3) 2^(3/2) |F′|^(3/2) |F‴|^(−1/2).
- By Rice at inflection points, the count of these pairs with ℓ ≤ t has the same t^(2/3) limit as (6).
- Its finite-t corrections come only from the jet density, with no higher Taylor terms and no grid loss.

**Grid loss** (`gridloss.py`).
- A pair is missed when both roots of F′ fall in one grid cell. N((0,t]) is cumulative, so it includes such ultra-short pairs.
- From the jet law, the expected loss at J = 4096 is:
  - 1.7–3.6% at t = 1e−5 for F, and 2.9–5.7% for the matched Gaussians;
  - 0.4–1.2% at t = 1e−4;
  - at most 0.3% for t ≥ 1e−3.
- It falls by a factor of 16 at J = 16384.
- `pool_log.txt` lists the raw and corrected rate for every t.

**Results.** Rates for t = 1e−4 … 3e−3, corrected for grid loss:

| ε | (6) | Gaussian with F's covariance | bar counts of F | cubic-jet count of F (t = 1e−5 … 1e−4) | realisations |
|---|---|---|---|---|---|
| 0 | 5.3201 | 5.3201 | 5.27–5.33 | 5.24–5.40 | 2×10⁵ (bars), 3×10⁵ (cubic) |
| 1 | 4.5991 | 4.1250 | 4.52–4.55 (J = 4096), 4.51–4.60 (J = 8192) | 4.57–4.76 | 1.7×10⁶ and 10⁵ (bars), 3×10⁵ (cubic) |
| 2 | 3.8689 | 2.8960 | 3.73–3.82 | — | 3×10⁵ |
| 3 | 3.3923 | 2.2625 | 3.27–3.38 (J = 4096), 3.30–3.45 (J = 16384) | 3.41–3.56 | 3×10⁵ and 1.5×10⁵ (bars), 3×10⁵ (cubic) |

The matched Gaussian's own bar counts give 4.10–4.19 at ε = 1, against its exact 4.1250 (5×10⁵ realisations), and 2.28–2.34 at ε = 3, against 2.2625 (3×10⁵).

`bins_log.txt` gives the independent lifetime bins.

## Reading

- **The simulation follows (6), not the Gaussian with the same covariance.** That Gaussian is 10%, 25% and 33% below (6) at ε = 1, 2 and 3. Simulations of that Gaussian reproduce its own exact value, so the pipeline is not biased toward (6).
- **The cubic-jet count tests the jet law of (6) directly.** At small t it agrees with (6) within its statistical error, for example −1.5%, −0.5% and +0.6% at t = 1e−5 for ε = 0, 1 and 3, each with SE about 4%. As t grows it rises up to about 4% above (6).
- **The residual is a finite-t effect, not a coefficient error.**
  - After the grid correction, the bar counts of F (ε ≥ 1) lie between 3.7% below and 1.8% above (6), mostly below.
  - The Gaussian controls deviate by −0.9% to +3.5% from their exact values.
  - The remaining NG deviation does not grow with the non-Gaussian excess, which grows 2.4-fold from ε = 1 to ε = 3.
  - The theorem gives no rate, so none of this is a finding against it.
- **The low counts at t ≤ 3e−5 on the 4096-point grid are mostly the computed grid loss.** After the correction they lie within about two standard errors of their predictions; the largest is −6.5% ± 3.3% at t = 3e−5 for ε = 3. The 16384-point rerun at ε = 3 is consistent with (6) there.

## Limits

These are finite algebra checks and a Monte Carlo of finite trigonometric fields in d = 1 only. They do not establish:
- a proof;
- anything for d ≥ 2, the Hp degrees or the Poisson law;
- anything for the theorem's dense-component class;
- a numerical onset or a sampler qualification.

The grid-loss model uses the leading cubic law. No author or earlier reviewer program was run.

## Reproduce

Each script runs as `python3 -I <script> <args>` with Python 3.11, numpy 2.4.6 and mpmath 1.3.0. `ng_controls.py` also needs sympy 1.14.0. Outputs are written to stdout.

| Output | Command |
|---|---|
| `ng_controls_log.txt` | `ng_controls.py` |
| `kappa_eps{0,1,2,3}.json` | `kappa.py {0.0,1.0,2.0,3.0}` |
| `kk_eps0.json`, `kk_eps1.json` | `kappa_kernel.py 0.0`, `kappa_kernel.py 1.0` |
| `sim_eps0_J4096.json` | `simulate.py 0.0 200000 11 4096` |
| `sim_eps1_J4096.json` | `simulate.py 1.0 400000 7 4096` |
| `sim_eps1_J8192.json` | `simulate.py 1.0 100000 13 8192` |
| `simb_eps1_R6e5.json` | `simulate_bisect.py 1.0 600000 21 4096` |
| `sims_eps1_a.json`, `sims_eps1_b.json` | `simulate_small.py 1.0 350000 31 4096`, `… 32 4096` |
| `sims_eps2.json` | `simulate_small.py 2.0 300000 52 4096` |
| `sims_eps3.json` | `simulate_small.py 3.0 300000 51 4096` |
| `sims_eps3_J16384.json` | `simulate_small.py 3.0 150000 61 16384` |
| `gm_eps1.json`, `gm_eps3.json` | `gauss_matched.py 1.0 500000 41`, `gauss_matched.py 3.0 300000 53` |
| `infl_eps{0,1,3}.json` | `inflect.py 0.0 300000 73 4096`, `1.0 300000 72 4096`, `3.0 300000 71 4096` |
| `gridloss.json` | `gridloss.py` |
| `newton_vs_bisect_log.txt` | `simulate.py 1.0 4000 99 4096` and `simulate_bisect.py 1.0 4000 99 4096` |
| `pool_log.txt`, `bins_log.txt` | `pool.py .`, `bins.py .` (run in this folder) |

The two Newton-vs-bisection runs print timing lines, so their `seconds` fields vary. `SHA256SUMS` covers every file here.
