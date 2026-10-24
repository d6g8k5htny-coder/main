# SUMMARY
I checked the note's local fold claim numerically at real collisions of de Bruijn–Newman zeros, and it holds. At two collisions, the gap between the merging zeros satisfies r² = 8τ, and the area between them satisfies ell = |A|r³/6. Both relative errors fall off linearly in τ, which is like r², all the way down to τ = 1e-12, and the slopes match second-order coefficients I derived by hand. Near the merge point, the two zeros match the predicted ±sqrt(2(t−t*)) offsets, either side of the axis for t above t* and as a complex pair x0 ± i·sqrt(2(t*−t)) below it. The first collision is t* = −0.3845611166279814… among the lowest 46 zeros (the pair γ≈111.03 and 111.87), and t* = −1.8779090729725353… among the lowest 10 (the pair γ≈48.01 and 49.77). The fold formulas are exact only in the limit. At t = 0, the real zeta zeros, they are off by 7% and 33% for the first pair, and by 17% and 128% for the second. Part (ii) is true of any generic fold and does not use zeta or the heat equation. Nothing here bears on RH. These are high-precision numerics, not interval-certified.

# DETAILS
Everything is saved under /tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/. The results are numerical, not proved. They use mpmath with 80–110 digits, but no interval arithmetic.

## How H_t is computed and how it was checked
- **Formula:** H_t(z) = ∫_0^∞ e^{tu²} Φ(u) cos(zu) du, with Φ exactly as in the brief.
- **Quadrature:** Φ is even, analytic in |Im u| < π/8 and decays doubly exponentially. So I use the trapezoid rule on the whole line, halved by evenness: step h = 1/160, cut-off u ≤ 1.25 (200 nodes), Φ summed until the terms are negligible (in practice about n ≤ 10).
- **Precision:** H_t(z) is about e^{-πz/8} in size while the integrand is of order 1, so the sum is done in about 80 digits. I used exact Python-integer fixed-point arithmetic, which takes about 1 ms per evaluation.
- **Closed forms:** all z-derivatives and the antiderivative F_t(z) = ∫_0^z H_t come from the same sum (∫cos(zu)dz = sin(zu)/u).
- **Match to ξ:** H_0(z) equals ξ(1/2 + iz/2)/8 to a relative 1e-44 or better for z ≤ 250.
- **Match to zeta zeros:** the zeros of H_0 match 2·Im(zetazero(k)) for k = 1..41 to 30 digits, which is the precision of the comparison (`validate_H0.out`).
- **Heat equation:** ∂_t H = −∂_z² H holds to working precision.
- **Robustness:** recomputing with h = 1/256, u ≤ 1.5 and 110 digits moves t* by at most 1e-40 and x0 by at most 1e-39 (`refine_check.py`).

## Finding the collisions
- **Tracking (`track.py`):** for each adjacent pair I follow the critical point c_k(t) between the two zeros using Newton's method. The test value D_k = sign·H_t(c_k(t)) changes sign exactly when the pair merges. I bisect to about 1e-45, then refine the double zero (x0, t*) by solving H = 0 and H_z = 0 together (2-D Newton).
- **Direct counts confirm it:**
  - window z ≤ 270: 46 real zeros at t = 0 and at t = −0.38, but 44 at t = −0.39;
  - window z ≤ 103: 10 real zeros at t = −1.87, but 8 at t = −1.89.
- **Lowest 46 zeros (γ < 135):** the first collision is the pair 34/35 (γ = 111.0295, 111.8747) at t* = −0.384561116627981414864807288345, x0 = 223.27578018561551991354. Here A = H''/2 = −1.7635e-36, B/A = −0.48692 (B = H'''/6), C/A = 0.070091 (C = H''''/24).
  - Later events in that window: 27/28 at −0.8811, 45/46 at −0.9600, 24/25 at −1.1439, 40/41 at −1.1519, 19/20 at −1.2755, 13/14 at −1.2853.
  - The tracker then stopped at t ≈ −1.355 because its own bookkeeping broke. That is a limitation of my code, not a mathematical event, and it comes after the first event.
- **Lowest 10 zeros:** the first collision is the pair 9/10 (γ = 48.0052, 49.7738) at t* = −1.87790907297253526477, x0 = 99.328469305575750948. Here A = 2.7627e-16, B/A = −0.41906, C/A = 0.062986.
  - Later events: 4/5 at −3.9548, 7/8 at −4.0338, 2/3 at −12.4945. The remaining pair, zeros 1 and 6, does not collide before t = −40.
- **Below t*:** the pair becomes complex, as expected. Newton's method on the complex sum gives zeros x0 + 2(B/A)(t−t*) ± i·y. The ratio y/sqrt(2(t*−t)) is 1.00097, 1.0000097 and 1.0000001 at t*−t = 1e-2, 1e-4, 1e-6.
- **"First collision" depends on the window.** Assuming RH and Λ ≥ 0 (Rodgers–Tao), close pairs at greater heights collide ever closer to t = 0. So "largest t* < 0" only makes sense for a fixed set of zeros.
- **Literature match (inference only):** a web search confirmed the historical table of lower bounds: −50 (Csordas–Norfolk–Varga 1988), −5 (te Riele 1991), −0.385 (Norfolk–Ruttan–Varga 1992), −0.0991 (Csordas–Ruttan–Varga 1991), and so on, down to Rodgers–Tao Λ ≥ 0 (Forum Math. Pi 2020). The full papers were blocked, so I could not see which zero pairs they used. My t* = −0.38456 for the γ≈111.03/111.87 pair matches the −0.385 bound, so that pair was plausibly the one used, but I have not confirmed it.

## The fold test
Write τ = t − t* for t above the collision.

**Second-order predictions.** These are my own hand derivation, using the heat-polynomial expansion H = A(x²−2τ) + B(x³−6xτ) + C(x⁴−12x²τ+12τ²) + …:
- r²/(8τ) − 1 ≈ κ_r·τ, with κ_r = 4C/A − 2B²/A²;
- ell/(|A|r³/6) − 1 ≈ κ_ℓ·τ, with κ_ℓ = 6B²/A² − 48C/(5A);
- the midpoint of the two zeros moves as x0 + 2(B/A)τ;
- ell = (8√2/3)|A|τ^{3/2}·(1 + (3B²/A² − 18C/(5A))τ).

**Relative errors at the pair 34/35** (κ_r = −0.19382, κ_ℓ = +0.74969):

| τ | r²/8τ − 1 | ell/(\|A\|r³/6) − 1 |
|---|---|---|
| 0.3846 (that is, t = 0) | −7.14% | +33.1% |
| 0.1 | −1.92% | +7.77% |
| 0.01 | −1.94e-3 | +7.52e-3 |
| 1e-3 | −1.94e-4 | +7.50e-4 |
| 1e-6 | −1.94e-7 | +7.50e-7 |
| 1e-12 | −1.94e-13 | +7.50e-13 |

**Relative errors at the pair 9/10** (κ_r = −0.09927, κ_ℓ = +0.44899):

| τ | r²/8τ − 1 | ell/(\|A\|r³/6) − 1 |
|---|---|---|
| 1.878 (that is, t = 0) | −16.7% | +128% |
| 1 | −9.4% | +56% |
| 0.1 | −0.99% | +4.6% |
| 1e-3 | −9.93e-5 | +4.49e-4 |
| 1e-12 | −9.93e-14 | +4.49e-13 |

**Decay rate and agreement.** On a log-log fit over τ ≤ 1e-4, both errors fall like τ^1.000000, at both collisions. In that range, the measured errors agree with κ·τ to within 6e-6 to 3.5e-5 relative, which is the size of the next-order term. The midpoint drift and the ell(τ) law agree to the same accuracy.

## What this says about the note
- **The local algebra in Sec. 3 is correct.** One imprecision: x_± ≈ ±sqrt(2τ) is only leading order, measured from x0. The pair's midpoint moves by 2(B/A)τ. That is a relative O(sqrt τ) correction to each zero, whereas the gap's correction is only O(τ).
- **Check (ii) is generic fold geometry.** For any smooth family with a nondegenerate fold, ell = |F'''|r³/12·(1+O(r²)), and here F''' = H'' = 2A. Neither the heat equation nor zeta enters, so "same cubic relationship as the project" just means both are the same fold.
- **What the heat flow adds is check (i).** The heat equation fixes how the fold opens up in time, so r² = 8τ with no free constant at every collision, whatever A is. This matches the classical motion of zeros near a collision, d(r²)/dt = 8 (from dz_k/dt = 2Σ1/(z_k−z_j)), which I cite from memory.
- **The cubic law does not describe the zeta zeros themselves.** At the low collisions, t* is between −0.38 and −12. The fold law is accurate to a few percent only for τ ≲ 0.1, and at t = 0 the area prediction is off by 33–128%.
- **None of this touches RH.** All the collisions are at t < 0. They only re-derive the known bound Λ ≥ t* (for example −0.3846), which is much weaker than Rodgers–Tao's Λ ≥ 0, and they say nothing about the "positive-time transition" the note's route needs.
- **How to classify the pieces:**
  - classical: the normalization, the heat equation, the motion of zeros, and the lower-bound history;
  - new but elementary: the second-order coefficients κ_r and κ_ℓ and the midpoint drift;
  - numerical: everything above.
- **The note's Sec. 5 arithmetic is right:** a gap density of r²dr gives a density of ell^0 d(ell), and r dr gives ell^{-1/3} d(ell).

## Files (all in the fold directory)
- **Scripts:** `hflow.py` (H, derivatives, antiderivative, ξ check), `zeros.py` (scan and refine), `test_basic.py`, `validate_H0.py`, `track.py` (the 10-zero run is in `low10/`), `fold_check.py`, `refine_check.py`, `count_check.py`, `analyze.py`.
- **Results:**
  - `fold_tau_r_ell.csv`: the requested τ, r, ell and predicted values, for both collisions;
  - `fold_pair34_35.csv` / `fold_pair9_10.csv`: full columns, including the midpoint and the second-order predictions;
  - `fold_*.json`: t*, x0, A, B/A, C/A, κ and the complex-zero check;
  - `collision_events.csv`: all collision events found;
  - logs: `fold_*.out`, `track.out`, `low10/track.out`, `count_*.out`, `validate_H0.out`.

# ARTIFACTS
[
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/fold_tau_r_ell.csv",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/fold_pair34_35.csv",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/fold_pair9_10.csv",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/fold_pair34_35.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/fold_pair9_10.json",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/collision_events.csv",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/hflow.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/zeros.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/track.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/fold_check.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/refine_check.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/count_check.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/analyze.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/validate_H0.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/validate_H0.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/fold_pair34_35.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/fold_pair9_10.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/track.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/fold/low10/track.out"
]