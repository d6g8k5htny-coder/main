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
- **LOWER (primary)**: w3_lbox.py over x in [-0.08,0.08], y in [0.50,0.68], 0.0025 boxes (4608), two processes (y-halves), gtol 1e-5, 160-bit box phase. Per box: boxint_lo certified; running accumulator in lbox_A.txt / lbox_B.txt. Checkpoint/resume: append-only logs; rerun the same command to resume after a restart (W3_RESUME=1 default; 0 for fresh).
- Three accumulators per box:
  * acc (LOWER): certified box-integral lower (boxint_lo);
  * acc_up (UPPERBAND): min(cheap,mid)*area (CS-class ~1e-2);
  * acc_ex (UPPEREX): hi(rho_exact_tier)*area — certified exact-tier band upper
    (per-box rho interval contains rho(y) for all y in the box); expected ~3-50x truth,
    i.e. ~1e-5-1e-4 class — sharper than the corrected-CS 1e-2 class on the band.
- Validation: brute-force true box integral at (-0.0037,0.5988) via 4 certified point
  evals: true = [6.619e-10, 6.624e-10] >= certified boxint_lo = 1.3691e-10 (yield 20.7%).
- First certified boxes: boxint_lo = 1.3691e-10 at (-0.0037,0.5988); g_lo = 0.156 within g=[0.156,1.27]; F_lo = 0.1217.
- **Expected outcome (calibration)**: per-box yield ~20-30% of the estimated true box integral (dominant loss: interval g width over the box, factor ~4; psi bound factor ~1.3-1.65). With the band integral at the ~4.8e-6 scale, the certified LOWER will land ~1-1.5e-6, i.e. BELOW the 3.328125e-6 = 0.213 r^3 refutation threshold. **The formal refutation is expected to be INCONCLUSIVE (OPEN), not a refutation and not a confirmation.**
- **UPPER (secondary)**: full-plane adaptive driver queued behind the lower pass (2 cores); band-upper partial sums from the same TM pass.

## Harness (fail-closed) — COMPLETE (post-restart)
w3_harness.py + w3_run_harness.sh; receipts/ holds .out/.err/.exit/.sha256 per run.
- CHECK0 law fingerprint (9-component target vector, 12 significant digits, hardcoded);
  CHECK1 anchor truth; CHECK2 hierarchy exact <= min(cheap,mid); CHECK3 quadrature
  convergence; CHECK4 F_conv vs brute force; CHECK5 dPhi width.
- harness_normal: exit 0; harness_O: exit 0; byteident.txt = BYTEIDENT PASS.
- Mutations (all fail closed, exit 1):
  * sqrt (missing Cauchy–Schwarz sqrt restored): caught by CHECK2
    (exact_hi=1.058117e-04 > mutated min(cheap,mid)_hi=4.250869e-09);
  * pin (pin M f-value +1e-2): caught by CHECK0 (law fingerprint mismatch);
  * zone (level b +1e-3): caught by CHECK0 (law fingerprint mismatch, since pin
    values include b and b-ell).
- failclosed.txt = FAILCLOSED PASS.

## Restart state (sandbox restart at ~box 15/4608)
Drivers relaunched from box 0 (old logs kept as lbox_A_run1.txt / lbox_B_run1.txt):
  python3 w3_lbox.py -0.08 0.08 0.50 0.565 0.0025 1e-5 lbox_A.txt
  python3 w3_lbox.py -0.08 0.08 0.565 0.68 0.0025 1e-5 lbox_B.txt
Resumption: rerun the same two commands (W3_RESUME=1 default: skips completed boxes
and seeds acc/acc_up/acc_ex from the log). Throughput-rebalanced split:
  python3 w3_lbox.py -0.08 0.08 0.50 0.575 0.0025 1e-5 lbox_A.txt   (1920 boxes)
  python3 w3_lbox.py -0.08 0.08 0.575 0.68 0.0025 1e-5 lbox_B.txt   (2688 boxes)
EXACT PARTITION of the band (A y-centers 0.5012..0.5737; B 0.5762..0.6787);
LOWER = LOWER(A) + LOWER(B), UPPEREX = UPPEREX(A) + UPPEREX(B).
WARNING: lbox_B_run3.txt (superseded) contains boxes at y<0.575 now owned by A --
do NOT sum it into the final result. lbox_A_run1/run2, lbox_B_run1/run2 are
superseded partial runs, likewise excluded.
Liveness: each box (and skip) rewrites lbox_{A,B}.txt.live (box id, driver clock);
watch.log heartbeats every 10 min include live_age seconds (dead vs slow
distinguishable: live_age > ~4x the per-box time means dead).
Automation: w3_deadman.sh (120s cadence) writes DEATH A/B or A/B COMPLETE +
ALL COMPLETE sentinels to watch.log; w3_supervisor.sh (300s cadence) relaunches a
dead driver from checkpoint (max 3 attempts per driver, then exits with a sentinel
line); a completed driver (EXITOK in log) is never relaunched.
Measured per-box cost at edge columns: A ~100s, B ~71s; ETA ~53h (both bands),
improving if interior boxes match the 23-45s smoke-test rates.
Completion marker: each log ends with DONE / LOWER= / UPPERBAND= / TIME= / EXITOK.
Final lower bound = LOWER(A) + LOWER(B) (+ the two run1 partial accumulators are
subsumed, since the rerun covers the same boxes).

## Change log
- v1 (pre-restart): initial drivers, harness draft.
- v2 (post-restart-1): drivers relaunched; harness completed (normal/-O/3 mutations,
  BYTEIDENT PASS, FAILCLOSED PASS); CHECK0 law fingerprint added; pin mutation
  strengthened to +1e-2; zone mutation fixed (w3c.Fr).
- v3: checkpoint/resume added (append-only logs, W3_RESUME); acc_ex (exact-tier band
  upper) accumulator added; hval clamp upgraded with the |D|-decaying bound;
  F_conv sharpened to phi(max(1,(|c-X|+Rm)/sig_lo)) (1.32x).
- v4 (post-restart-2): REBALANCE BUG caught and fixed: an A/B re-split to
  [0.50,0.575]/[0.575,0.68] had left old-B's seeded boxes at y in [0.565,0.575)
  inside A's range (double-count) plus a 252-box hole at y in [0.565,0.575),
  x > -0.0788. Fixed by restarting B FRESH on [0.575,0.68] (old log preserved as
  lbox_B_run3.txt, marked DO-NOT-SUM) and resuming A on [0.50,0.575]; the final
  cover is an exact partition. All superseded logs preserved (lbox_A_run1/run2.txt,
  lbox_B_run1/run2/run3.txt).
- v5: .live liveness receipts (per-box rewrite of lbox_{A,B}.txt.live) + watcher
  live_age heartbeats; approved as standard by the lead. Lead decision: full band
  at the conservative ~53h, no schedule constraint; report only at band completion
  or detected driver death.
- v6 (post-restart-3): SECOND infrastructure death (timeline: first death Sep 13
  08:41 machine restart, A at 15 boxes / B at 22; second death Sep 14 08:27,
  A at 26 boxes acc=1.466068e-10 / B at 19 boxes acc=4.282164e-10). Root cause
  (identified by the lead, canary-verified): pgrep -f SELF-MATCHES its own pattern
  in this environment (pgrep does not exclude itself), so every keeper/supervisor
  guard checking liveness via pgrep -f '<pattern>' saw itself and concluded the
  drivers were alive forever -- no relaunch ever fired. Fix: bracket-trick
  patterns (pgrep -f '[w]3_lbox.py ...' never self-matches) in w3_deadman.sh and
  w3_supervisor.sh (w3_watch.sh uses no pgrep). Verified from a clean script
  context: count 0 with drivers dead, 1 each with drivers running. Both bands
  resumed from checkpoint (integrity re-verified: complete lines, monotone
  accumulators): A from box 26 (acc=1.466068e-10), B from box 19
  (acc=4.282164e-10); heartbeats confirmed resetting live_age before idle.

## Independence statement
No KIMI WP conclusions (KIMI-DER-025, verify_wp_witness_v1.py, THM-023 drafts), no W4 report, and no other subagent's pipeline were read or used. All kernel/law/estimand quantities re-derived from the model statement.

## Freeze
Pending driver completion (~15h ETA from 05:12Z). Final numbers, hashes (sha256 of all code + transcripts + receipts) to follow at freeze.
