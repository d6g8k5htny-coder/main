# Validation report and audit

**Revised 2026-10-19. This replaces the version in commit `468da8d` (2025-12-16), whose
"83% success rate" and "no falsifications" claims are withdrawn.**

This branch holds the retired December 2025 README and manuscript generator, plus Python
code written for them on 2025-12-16. The README and `body` are byte-identical to the copies
the owner archived on `main` under
[`history/2025/`](https://github.com/d6g8k5htny-coder/main/blob/0a7e5145b55f020e02e7782fed318266d13c2447/history/2025/README.md),
which records that their validation claims "are not current repository facts or verified
research results." Nothing on this branch changes that. The code checks arithmetic and
internal consistency. It does not supply empirical evidence for the framework.

## Current results

Output of `python -m src.validation`:

| Claim | Status | σ from claim | Notes |
|-------|--------|--------------|-------|
| Three Generations | consistent | 2.0 | N_ν = 2.984 ± 0.008; re-analysis 2.9963 ± 0.0074 (0.5σ). n = 3 is the centre of the penalty exp(α(n-3)²), an input, and is the minimum only for α > ln(1 + K₁) |
| No 4th Generation | consistent | - | README-stated limit m_t' > 656 GeV; the excluded n is an input |
| Helicity Barrier tau | not_independent | - | τ is the manuscript's own fitted coefficient; no independent measurement or reproducible analysis is in this repository |
| Critical Beta | not_independent | - | Stated in the README and manuscript, citing McIntyre et al.; no data or analysis here |
| f_NL < O(1) | consistent | 0.2 | Planck: f_NL = -0.9 ± 5.1, which cannot distinguish f_NL of order 1 or below from many alternatives |
| Stelle Ratio | unsupported | - | Manuscript Thm 5.1 equates γ₁R² + γ₂R_μν² at ratio -1/4 with Gauss-Bonnet (dropping its Riemann² term), then asserts -1/2 from unstated loop corrections. Mod Gauss-Bonnet the Weyl² coefficient is γ₂/2, so any γ₂ ≠ 0 keeps the spin-2 ghost. No measurement exists |
| Cl(6) Mass Ratios | falsified | 58.2 | Observed/predicted over six adjacent ratios: 1.6 to 55; 3 miss by more than 10×. The manuscript cites RG running without a calculation; schemes are mixed |

"Consistent" means only that the data do not contradict the claim. In each such row the
claimed value was either built into the model or is too loose for current data to test.

## What was run

Environment: Python 3.11.17, numpy 2.4.6, scipy 1.17.1, mpmath 1.4.1, pytest 9.1.1.

- `pytest tests/test_all.py`: **94 passed**, also with `-W error::RuntimeWarning`.
- Original suite at `468da8d`, run for the first time on 2026-10-19: **4 failed, 62 passed**.
  The 2025-12-16 commit had been pushed without running it.
- Each module's `python -m src.<module>` demo runs to completion.
- Zeta zeros (`data/riemann_zeros/first_100_zeros.json`, Git blob `16c7c489`, SHA-256
  `5c54fc3358f696b345b2c2411707f0f628c8fc3aeb8f73dfd0d69f08816a70d1`), rechecked
  2026-10-19 with mpmath 1.4.1:
  - **Reproduction:** the file's recorded command (`mpmath.zetazero`, 30 significant
    digits, printed to 20) reproduces all 100 digit strings exactly.
  - **Ordering:** the values are strictly increasing and unique (smallest gap 0.716).
  - **Bracketed counts:** `mpmath.nzeros(b_k) = k` for every k = 0..100. b₀ is below
    the first entry, b_k are the midpoints between entries, and b₁₀₀ = 237.147 lies
    between entry #100 and zero #101. Each entry lies inside its bracket, and the
    Riemann–Siegel Z function changes sign across every bracket.
  - **Residuals:** max |ζ(½ + iγ)| is 2.45e-17 at the stored 20-digit values and
    5.66e-14 at their float64 roundings. The `468da8d` file fails the residual check
    (max |ζ| = 2.2e-6).

  Together these show, numerically, that the list is exactly the first 100 zeros, in
  order. All of it uses mpmath, the library that generated the list, in floating point.
  Nothing is interval-certified, and no independent implementation or published table
  has been compared (see "Not checked"). `verify_rh_local` and its tests repeat these
  checks, and reject a duplicated, swapped, missing or perturbed entry.
- Citations added in this revision were checked by web search on 2026-10-19: Janot &
  Jadach (PLB 803, 135319: N_ν = 2.9963 ± 0.0074), Rodgers & Tao (Forum Math. Pi 8, e6),
  Platt & Trudgian (BLMS 53, 792–797, including Λ ≤ 0.2), Tristram et al. (PRD 105,
  083524: r < 0.032), and the PDG top mass (172.69 ± 0.30 GeV in 2022, 172.57 ± 0.29 in
  2024). Polymath15 is cited by arXiv number only, because its journal details were not
  confirmed.

## Defects in the 2025-12-16 implementation, fixed here

| # | Defect | Fix |
|---|--------|-----|
| 1 | τ = 0.022 ± 0.008 and β_c = 0.5 were compared with identical hard-coded "measurements" attributed to McIntyre et al. 2025 and Parker Solar Probe, and reported as CONFIRMED. | Reported as `not_independent`; the values are kept as manuscript-reported, not measured. |
| 2 | The report omitted the comparisons that failed: mass ratios off by 1.6× to 55×, and the code's CKM guesses off by 24σ to 163σ. | Mass ratios reported as `falsified`. The CKM guesses were not the manuscript's (see 8) and were removed. |
| 3 | Invented data: PSP encounter β ranges and barrier-detection flags, GUE "fit R²" values (0.999, 0.998, 0.997) attributed to Odlyzko, a "GR deviation 0 ± 0.05 rad", "BBN N_eff = 2.85 ± 0.15", a placeholder author e-mail and repository URL. | Removed. PSP analysis now requires caller-supplied observations. |
| 4 | 36 of the 100 stored zeta zeros (from #53 on) were wrong by up to 5.2e-7, yet the file said "All verified". `verify_rh_local` only read back hard-coded flags. | Zeros regenerated with `mpmath.zetazero`. `verify_rh_local` now evaluates ζ, checks strict ordering and uniqueness, and counts zeros at brackets through the last entry. |
| 5 | The anomaly check added right-handed fields with the wrong sign, giving U(1)³ = -4/9 and SU(3)²U(1) = 2/3. The Standard Model is in fact anomaly-free. | Left-handed Weyl convention, exact rational arithmetic; all four anomalies are exactly 0. |
| 6 | Zero spacings were normalised by log(t)/2π instead of log(t/2π)/2π, giving mean 1.88 instead of 1.00. The test's (0.5, 2.0) bounds hid this. | Formula fixed; test bound tightened to (0.9, 1.1). |
| 7 | C_B was treated as cross-helicity σ_c; the manuscript defines it as magnetic compressibility. Standard errors of a₀ and a₁ were mis-copied (0.01 and 0.02 instead of 0.012 and 0.031). | Separate compressibility input; manuscript errors restored. |
| 8 | The mass model put generations on grades 1, 3, 5, leaving generation 2 unsuppressed. The manuscript uses grades 2, 1, 0. The CKM rule V_ij ~ ε^Δgrade and the ratio m_t/m_b = 1 were inventions, not manuscript claims. | Manuscript's model implemented; inventions removed; the manuscript's fitted Cabibbo exponent is reported instead. |
| 9 | SO(10) 16 Casimir was 15/2 instead of 45/8. `anomaly_coefficient` returned 0 for any group without computing anything. E6 was compared with no matter content. | Fixed; undefined matter content raises `ValueError`. |
| 10 | ν_μ and ν_τ mass limits were stored 1000× too small. The coupling analysis mixed α(0) with M_Z-scale sin²θ_W, and a non-GUT α₁ with the GUT b₁ = 41/10. The CKM first-column unitarity entry (0.9999) contradicted the listed elements (0.9971). Λ ≥ 0 was credited to Polymath (it is Rodgers & Tao), and the upper bound 0.22 has since been improved to 0.2 (Platt & Trudgian 2021). | Fixed. |
| 11 | Data labelled "PDG 2024" were typed from memory. Where checked (m_t), they match the 2022 Review. | Relabelled as such; still not re-verified value by value. |
| 12 | Tests asserted the minimum is at n = 3 for every α in [0.1, 10], which fails. | Tests assert the true threshold rule and pin each finding above. |

## Observations on the retired manuscript's claims

These are analytic and numerical checks by an AI session. They have not been independently
reviewed, and they are not a scientific adjudication of the manuscript. Line numbers refer to
`body` (identical to `history/2025/body.original.txt` on `main`).

1. **Generation "theorem" (README).** "Minimum at n = 3 for all α > 0" is false. With
   C(n) = nK + exp(α(n-3)²), C(n) - C(3) = (n-3)K + exp(α(n-3)²) - 1. This is positive
   for every n > 3. For n = 2 it is positive exactly when α > ln(1 + K), and that condition
   also covers n = 1. So n = 3 is the unique minimum iff α > ln(1 + K) (≈ 0.693 for K = 1);
   at α = 0.1 the minimum is n = 1. Because the penalty is centred on 3 by hand, even the
   corrected statement restates an input.
2. **Stelle ratio (Theorem 5.1, line 294).** The proof first sets γ₁R² + γ₂R_μνR^μν with
   γ₁/γ₂ = -1/4 equal to the Gauss–Bonnet density. That step drops the R_μνρσR^μνρσ term.
   The proof then asserts -1/2 from "loop corrections" with no calculation. In 4D, modulo
   Gauss–Bonnet, γ₁R² + γ₂R_μνR^μν = (γ₂/2)C² + (γ₁ + γ₂/3)R². Stelle's massive spin-2
   ghost is absent only when the C² coefficient vanishes, so in this basis no finite ratio,
   including -1/2, is ghost-free.
3. **τ = 0.022 ± 0.008 (§12.4, Appendix C).** This is a fitted regression coefficient
   (t = 2.75), not a prediction. The ACE data and analysis are not available. The reported
   statistics are mutually inconsistent: for two regressors, R² = 0.82 with df = 156 implies
   F ≈ 355, not 247.3. The fitted polynomial is negative for 1.007 < C_B < 8.32. The
   thresholds β_c ≈ 0.5 and σ_c ≳ 0.4 are cited to McIntyre (2024) and PRX 15, 031008; they
   were not checked against those papers here.
4. **Mass ratios (§7.3, Appendix B).** The manuscript's own tables show misses of 12.6, 55,
   4.2, 1.9, 1.6 and about 20, yet describe them as "order-of-magnitude agreement". Three of
   the six miss by more than 10×. The geometric-mean ε implied by the data is 0.0126, against
   6/64 = 0.094. The quoted masses mix renormalisation schemes.
5. **Cabibbo angle (eq. 7.4).** |V_us| = ε^0.632, whereas the manuscript writes
   ε^0.66 ≈ 0.225; in fact ε^0.66 = 0.210. Either way the exponent is fitted.
6. **Gauge-group selection.** Under the implemented K(G) = rank × h^∨, the subgroups U(1),
   SU(2) and SU(3) all score below SU(3)×SU(2)×U(1). The claim that the SM minimises
   complexity therefore needs constraints the material does not state.
7. **Riemann Hypothesis.** The complexity argument is a hypothesis. The established
   numerical facts cited (RH verified to height 3·10¹², GUE spacing statistics) do not depend
   on the framework. The README's 0 ≤ Λ ≤ 0.22 is now 0 ≤ Λ ≤ 0.2.

## Not checked

- PDG and Planck values against current editions; they were transcribed from memory.
- Whether the cited heliophysics papers state the β_c and σ_c thresholds.
- Manuscript chapters with no code here, such as the Born-rule and cosmology chapters.
- Any review by a non-Claude or human reviewer.
- An independent check of the 100 zeros, against another implementation or a published
  table such as Odlyzko's. A download of Odlyzko's table on 2026-10-19 got no HTTP
  response, so this stays open.

## Provenance and exposure

- Audited source: `468da8da44c697d106beafe70589cfa79d8473ae` (2025-12-16), on base
  `f25b04bb931df2eaee302b666db014913486166b`. Its README and `body` match Git blobs
  `108b169941ba8986394f1227459047846e227267` and `03f0a2a5be8b8e6a65176f6b6b7bd36d6107759c`,
  which `main` archives under `history/2025/`.
- The revision and this report are by Claude Opus 5.5 (`claude-opus-5-5`), in the same Claude
  Code session that produced `468da8d`; that commit does not record which model wrote it.
  This is an author-side correction by the same provider. It earns no
  organizational-independence credit.
- The sibling branch `claude/setup-complexity-physics-8H96d` (`45f7dee`, PR #1) belongs to a
  different session. It was not modified.
