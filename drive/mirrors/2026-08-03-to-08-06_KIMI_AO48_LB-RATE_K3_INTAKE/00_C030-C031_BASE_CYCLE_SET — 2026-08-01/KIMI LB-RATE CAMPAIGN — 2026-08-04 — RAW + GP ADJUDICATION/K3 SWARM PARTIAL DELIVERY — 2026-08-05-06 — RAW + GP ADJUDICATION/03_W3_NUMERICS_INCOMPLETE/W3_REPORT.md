# W3 — Rigorous WP Numerics (SIDE24/q0 lower-rate repair) — STATUS: DRIVER RUNNING (pre-freeze draft)

## Model (exact, independent re-derivation)
Normalized periodized Bargmann–Fock field on T^2_24: Var f = 1,
K(x,y) = K1(x1-y1)K1(x2-y2), K1(s) = Z^{-1} sum_{j in (pi/12)Z} e^{-j^2/2} e^{ijs}.
Level b = 6/5; window [b-ell, b], ell = r^3/6, r = 0.025;
pins (f, d1f, d2f): M=(-0.0125,0) f=b; S=(0.0125,0) f=b-ell; Y=(-0.0315,0.006) f=mu_t=1.1999986986564927790, gradients 0.
Estimand rho_WP(y;r) = p_{grad f~}(0) * E[|det H| 1{det<0} 1{b-ell<f~<b} | grad f~=0], I_WP = int_Z rho_WP dy.

## Certified machinery (all interval, fail-closed; no float64 padding, no eigenvalue clipping)
1. **Conditional Gaussian law**: 18x18 Gram at 350-bit iv; whitener C = Lambda^{-1/2} Q^T with Neumann-certified inverse (residual certification, Neumaier-style).
2. **TaylorLaw (box law)**: Taylor models of V~(y) = Cov(J(y), pins) about box centers; coefficients point-exact; global remainder via Cauchy–Schwarz on kernel derivative sups (wrapped-lattice tails certified to derivative order 41).
3. **w3_tm.py**: TM polynomial arithmetic; coefficient-level cancellation for vt = det3/det2 (Schur complement, vt ~ 6.8e-5 at the hot ridge), mt, pgrad, Hessian law V = Vnum/det3.
4. **g(u) = E[|det H| 1{det<0}]** via Gil-Pelaez on phi(t) = D^{-1/2} exp(itN/D): cumulants -> combined U4 (min of moment bounds and |D|^{-1/2}-decaying Bell bounds), per-panel Simpson–Peano remainders, Cauchy-series head, algebraic tail. **Interval explosion fixed rigorously**: |phi(t)| <= 1 (ch.f.) => h=(1-Re phi)/t^2 in [0, 2/t^2]; hval clamped to this (valid for all V in the box).
5. **dPhi**: certified Phi-differences (Taylor-in-h with probabilists'-Hermite terms and a Cramer remainder bound) replacing too-coarse Mills brackets — used in Pwin and all convolution pieces.
6. **Box-integral lower bound**: int_box rho dy >= pgrad_lo * g_lo * ell/sig_hi * area * F_lo with
   - g_lo = lo of the interval g-quadrature over the box (contains every V in the box);
   - F_lo = certified inf over the window of E[phi(max(1,(|c-X|+Rm)/sig_lo))], X = grad(mt).Uniform(box) (trapezoid convolution, closed-form pieces; validated against brute force to 6 digits);
   - Rm = certified quadratic remainder of mt(y) from TM Hessian sups;
   - psi(m;u) >= phi(max(1, |u-m|/sig_lo))/sig_hi (exact min over [sig_lo, sig_hi]).

## Point-certified anchors (tight intervals, point law)
- rho(0, 0.60) = [1.0572974460e-04, 1.0581169866e-04]  (harness CHECK1; mid-tier upper 1.0794e-4, only 2% above exact)
- g(anchor) = [0.647581909, 0.647787533]
- KIMI witness region (-0.04,-0.60): rho <= 1.945e-8 (type-killed).

## Drivers (background, deterministic)
- **LOWER (primary)**: w3_lbox.py over x in [-0.08,0.08], y in [0.50,0.68], 0.0025 boxes (4608), two processes (y-halves), gtol 1e-5, 160-bit box phase. Per box: boxint_lo certified; running accumulator in lbox_A.txt / lbox_B.txt. Also emits per-box upper min(cheap,mid)*area (band upper, ~1e-2 class).
- First certified boxes: boxint_lo = 1.3691e-10 at (-0.0037,0.5988); g_lo = 0.156 within g=[0.156,1.27]; F_lo = 0.1217.
- **Expected outcome (calibration)**: per-box yield ~20-30% of the estimated true box integral (dominant loss: interval g width over the box, factor ~4; psi bound factor ~1.3-1.65). With the band integral at the ~4.8e-6 scale, the certified LOWER will land ~1-1.5e-6, i.e. BELOW the 3.328125e-6 = 0.213 r^3 refutation threshold. **The formal refutation is expected to be INCONCLUSIVE (OPEN), not a refutation and not a confirmation.**
- **UPPER (secondary)**: full-plane adaptive driver queued behind the lower pass (2 cores); band-upper partial sums from the same TM pass.

## Harness (fail-closed)
w3_harness.py + w3_run_harness.sh; receipts/ holds .out/.err/.exit/.sha256 per run.
- CHECK1 anchor truth; CHECK2 hierarchy exact <= min(cheap,mid); CHECK3 quadrature convergence; CHECK4 F_conv vs brute force; CHECK5 dPhi width.
- normal: exit 0; -O: exit 0; byte-identity: receipts/byteident.txt.
- Mutations (each must exit nonzero): sqrt (missing Cauchy–Schwarz sqrt restored -> caught by CHECK2), pin (pin value +1e-4 -> caught by CHECK1), zone (level b +1e-3 -> caught by CHECK1).

## Independence statement
No KIMI WP conclusions (KIMI-DER-025, verify_wp_witness_v1.py, THM-023 drafts), no W4 report, and no other subagent's pipeline were read or used. All kernel/law/estimand quantities re-derived from the model statement.

## Freeze
Pending driver completion (~15h ETA from 05:12Z). Final numbers, hashes (sha256 of all code + transcripts + receipts) to follow at freeze.
