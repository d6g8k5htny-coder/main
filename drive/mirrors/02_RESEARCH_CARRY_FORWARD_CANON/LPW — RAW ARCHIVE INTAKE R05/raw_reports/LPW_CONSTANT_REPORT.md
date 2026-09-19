# LPW_CONSTANT: certified explicit constant and radius for LPW-CAND-20260912

**Record:** LPW-CONSTANT-20260912-v1.0
**Role:** constant evaluation layer on top of the endorsed existential proof
(Kimi verdict APPROVE, six interfaces PASS WITHIN SCOPE), executed under the
acceptance contract `03_CONSTANT_CERTIFICATE_CONTRACT.md`
(LPW_Review_Reconciliation_2026-09-12).
**Result may be tiny; validity, not attractiveness, is the goal (contract §5/§7).**

## 1. Certified headline

For the exact two-dimensional side-24 six-pin typed maximum-saddle pair-Palm law
of the pinned candidate (full-file sha256
`cf58f72eb0399c626160c143b50f674ff3bd5c449225211edd6fd378a374e1a5`),

    1 - q(r, 6/5)  >=  c r^3      for all 0 < r <= r_0,

with the CERTIFIED explicit pair

    c   >=  260 / (3790446482793 * 2^40 * 10^21)  =  6.239e-44  >=  10^(-44)  > 0
    r_0  =  1 / 2414592  =  1 / (256 * 9432)      ~  4.14149e-7              > 0

tighter machine estimates (not load-bearing): c ~ 1.04556e-43, m ~ 1.676e-21.
The positivity of c is demonstrated by exact rational arithmetic and outward
power-of-ten enclosures, not by displayed decimals (contract §5): the chain is
m >= 10^(-21)  =>  c = 260 m delta^4 / B_3 >= 260 * 10^(-21) / (B3_rat * 2^40),
B3_rat = 3790446482793 an integer upper bound, delta^4 = 2^(-40) exact.

Final arithmetic (contract §5): K = max(1, 2 B_4) <= K_rat = 9432,
C_Z = 4 B_3 <= 15161785931172, r_0 = min(R, 1, 1/(256 K)) = 1/(256 K_rat)
(rounded downward; 1/(256 K_rat) <= R = 1e-5 so the K-term binds).
Hard checks with exact rationals: 16 delta + 8 K r_0 = 3/64 <= 3/64 < 1/16,
path clearance 103/3840 > 0, exponent ledger 1 + 4 - 2 = 3, weight base
(81/16)(1673/128) >= 65, 65*16 = 1040 = 4*260, thin-box volume 32 delta^4 r.

## 2. Exact objects (contract §1)

- Field: normalized periodized side-24 2D covariance. Spectral form used here:
  frequencies k in (pi/12) Z^2, weight exp(-|k|^2/2), Z-normalized,
  Cov(f(z1), f(z2)) = (1/Z) sum_k w(k) cos(k.(z1 - z2)).  Identity with the
  candidate's real-space k_24(x) k_24(y) is Poisson summation: |Z - 288/pi| =
  8.96e-59 (bound 1.0e-45), and the a_2/a_4/a_6 spectral vs real-space
  derivative forms agree to 1e-130 at 150 dps (falsifier F3).
- Conditioning: six-pin continuous Gaussian regression version, Hermite
  combinations U_r -> U_0 = (f, f_x, f_y, f_xx/2, f_xy, f_xxx/6);
  J = (f_yy, f_xxy/2, f_xyy/2, f_yyy/6); b = 6/5; u_r = (b - r^3/12, -r^2/4,
  0, 0, 0, 1/3); delta = 1/1024; epsilon = 1/16 (unchanged).
- Norms: ||f||_{C^k} = max_{|alpha|<=k} sup_T^2 |d^alpha f| (candidate §4
  convention); M_3 = max(1, ||f||_{C^3}); M_4 = max_{|alpha|=4} sup |d^alpha f|
  (R3 addendum confirms the full C^4 norm as dominating auxiliary).
- CHANGED compact set (contract item 1; stated explicitly per contract §5 of
  the endpoint supplement): for the certified radius R = 1/100000,

      JBOX(R) = [-(10 + 2 delta) R, 0] x [2 - delta, 2 + delta]
                x [-delta, delta]^2 ,

  which contains every thin box E_r = {|q/(2r)+5|<=delta, |A-2|<=delta,
  |B|<=delta, |D3|<=delta} for 0 < r <= R (exact rational containment, checked).
  ALL dependent bounds (density floor m, B_4, conditional means M) are computed
  on this set, not on the candidate's larger K_J.

## 3. Method per contract

**(2) Exact finite-torus moments.** a_{2m} = ME[2m] = (1/Z1) sum_n e^{-a n^2}
(h n)^{2m}, h = pi/12, a = pi^2/288, computed at 60 dps on |n| <= 70 with
certified geometric tails (p = 0..12, e.g. p=0: 1.9e-75, p=12: 3.24e-60) and a
blanket rounding allowance ROUND = 1e-45 per quantity.  Never planar rationals:
1 - a_2 = 9.65254179896e-123 > 0 certified at 150 dps (both the spectral and
the supplement's 576-sum forms), so a_2 < 1 exactly; planar deviation of a_4,
a_6 likewise < 1e-100 certified. Endpoint identities verified in the exact
moments: Cov(J|U_0) = diag(a_4 - a_2^2, a_2(a_4-a_2^2)/4, a_2(a_4-a_2^2)/4,
(a_6 - a_4^2/a_2)/36) and E[J|U_0=u_0] = (-a_2 b, 0, 0, 0), max deviation
1.6e-61.

**(3) Radius by interval modulus (no sampled rungs).** Every covariance entry
of the ten-jet Gamma_r is an exact 1D spectral sum,
Cov_ij(r) = ME[B] (1/Z1) sum_n w1 k1^A g_i(k1 r/2) g_j(k1 r/2), with explicit
Hermite profiles g_i (smooth at r = 0; cross-validated against the candidate's
displayed T_r matrix route with correct sine moments: agreement <= 2.1e-56 at
r = 0.5, 0.25, 0.1, 0.025, falsifier F1).  Analytic profile derivative bounds
|g_i| <= pb_i, |g_i'| <= pb1_i (proved by |sin theta| <= |theta| and
|int_0^theta t sin t dt| <= min(theta^2/2, |theta|^3/3); grid-sanity-checked in
F4) give, for r in [0, R]:

    ||d Gamma_r/dr||_F <= L_1 = 19.071156   =>
    ||Gamma_r - Gamma_0||_F <= L_1 R = 1.90712e-4.

Endpoint floor of record (supplement §3, author-side PROVED via ten exact
Sylvester minors of Gflat - (1/8) I plus the < 1e-102 periodic correction;
minors independently reproduced with exact rational arithmetic in falsifier
F2): lambda_min(Gamma_0) > 31/250 = 0.124.  This engine's independent eigfloor
(0.126553449667289 - 3.19e-60 Frobenius residual - 5.29e-44 tails) is
consistent and stronger; the record value is used in the chain.  Hence, for
all r in [0, R]:

    lambda_min(Gamma_r) >= lam = 31/250 - L_1 R = 0.123809288439 > 0,

inherited by the six-pin block (Cauchy interlacing) and by the Schur
complement S_r (S^{-1} = (Gamma^{-1})_{JJ}), i.e. lambda_min(S_r) >= lam and
lambda_min(GU_r) >= lam.  r_G := R = 1e-5 is thereby certified.

**(4) Density floor (contract boxed formula).** On JBOX(R): R_J = sup |j| =
2.000977042; conditional means sup_{[0,R]} |mu_r| <= M = 1.222575834 (exact
|mu_0| = a_2 b plus interval correction through L_GU, L_GX, lam); Lambda =
3.000... (Gershgorin upper bound on lambda_max(Cov J) >= lambda_max(S_r),
S_r Loewner-below the r-independent Cov J).  Then for every r in [0, R] and
j in JBOX(R),

    p_{J|U}(j | u_r) >= m = (2 pi)^-2 Lambda^-2 exp(-(R_J + M)^2 / (2 lam))
    exponent <= 41.96491748  =>  m ~ 1.676e-21,  certified floor m >= 10^(-21)

((2 pi)^-2 > 1/40 exactly; log10 enclosures outward; N_m = 21 exact).
Cross-check: m <= 1.2832e-3, the review's thin-box center diagnostic, OK.

**(5) B_3, B_4 (contract §4 regression-operator bounds; residual never assumed
stationary).**  A_k = sup_{r,x,|alpha|<=k} ||d^alpha C_r(x)||_2 *
sup_r ||Gamma_r^{-1}||_2 <= sqrt(M_{2k} tr_sup) / lam with exact moments
M_6 = 15, M_8 = 105 and trace sups from the same interval modulus:
A_3 = 71.10596 (six-pin), A_4 = 262.81357 (ten-jet).  Global field moments
through the spectral majorant ||f||_{C^p} <= sum_k sqrt(w_k/Z)(1+|k|)^p
(|xi_k|+|eta_k|) with full certified Fourier tails: S_3 = 554.3913563 (+/-
4.07e-32), S_4 = 2041.832733 (+/- 1.11e-30), E zeta = 2 sqrt(2/pi),
E zeta^4 = 12 + 16/pi:

    E||f||_{C^3}^4 <= mu_4 S_3^4 = 1.6146672e12,   E||f||_{C^4} <= mu_1 S_4 = 3258.2936,

    B_3 <= [1 + (E||f||_{C^3}^4)^{1/4} + A_3((sup E||U_r||^4)^{1/4} + sup|u_r|)]^4
         <= 3.790446483e12  <= B3_rat = 3790446482793,
    B_4 <= E||f||_{C^4} + A_4(sup E||V10_r|| + sup||(u_r, j)||)
         <= 4715.749569,   K <= K_rat = 9432,

using E||U_r||^4 = (tr GU)^2 + 2||GU||_F^2 <= 38.23... (Isserlis) and
sup||(u_r, j)|| <= 2.3701308 on [0,R] x JBOX(R).

**(6) Final arithmetic:** §1 above.  c > 0 and r_0 > 0 are exact-rational
statements.

## 4. Remainder ledger (every certified number carries an interval/rational error)

| item | value used | error treatment |
|---|---|---|
| 1D spectral tails p=0..12 | 1.9e-75 .. 3.24e-60 | exact geometric ratio bound, ratio < 1/2 checked |
| rounding allowance ROUND | 1e-45 | 60-dps mpf eps 2^-199 ~ 1.2e-60; magnitudes <= 1e6, <= 1e5 ops |
| moments a_{2m} | ME[2m] +/- (tail/Z1 share + ROUND) | upward in all upper bounds |
| Poisson/normalization | |Z - 288/pi| = 8.96e-59 | <= 2*tail + ROUND, checked |
| a_2 deviation | 9.65254179896e-123 | 150-dps, both forms, 0 < dev < 1e-100 checked |
| endpoint eigfloor of record | 31/250 | author-side proved; here: eig residual 3.19e-60 + entry tails 5.29e-44 consistency |
| interval modulus L_1 | 19.071156 | analytic pb/pb1 bounds + moment tails + ROUND, upward |
| lam (uniform on [0,R]) | 0.123809288439 | = 31/250 - L_1 R - ROUND |
| diag/trace/Frobenius sups | entry |Gamma_ij(0)| + C1_ij R + ROUND | upward |
| M (mean sup) | 1.222575834 | exact |mu_0| + interval correction terms, upward |
| R_J | 2.000977042 | exact sqrt + ROUND, upward |
| Lambda | 3.0(+allowances) | Gershgorin on exact moments, upward |
| density exponent | <= 41.96491748 | (R_J+M)^2/(2 lam) + ROUND, upward |
| m | >= 10^(-21) | log10 enclosures outward; (2 pi)^-2 > 1/40 exact; estimate 1.676e-21 |
| S_3, S_4 | 554.3913563, 2041.832733 | + certified 2D shell tails 4.07e-32 / 1.11e-30 + ROUND, upward |
| mu_1, mu_4 | 2 sqrt(2/pi), 12 + 16/pi | + ROUND, upward |
| B_3, B_4 | <= 3.790446483e12, <= 4715.749569 | all ingredients upward; integer ceilings B3_rat, K_rat |
| K, C_Z | <= 9432, <= 15161785931172 | integer outward enclosures |
| r_0 | 1/2414592 | exact rational, rounded downward (K_rat >= K) |
| c | >= 6.239e-44 >= 10^(-44) | exact Fraction c_lo = 260/(B3_rat 2^40 10^21); power-of-ten floor by integer comparison |
| hard checks | 16d + 8 K r_0 = 3/64; 103/3840 > 0; 1+4-2 = 3 | exact Fractions |

## 5. Executable falsifier

`falsify.py` (exits nonzero on any failure):
- F1: covariance engine cross-check, profile route vs the candidate's displayed
  T_r Hermite-matrix route with correct sine moments, four rungs (<= 2.1e-56).
- F2: exact rational recomputation of the ten leading principal minors of
  Gflat - (1/8) I (matches supplement §3: 7/8, 49/64, ..., 203/1073741824) and
  the correction-chain arithmetic 1/8 - 1e-102 > 31/250.
- F3: a_2 deviation positive, both forms; a_4 = k_24^{(4)}(0),
  a_6 = -k_24^{(6)}(0) Poisson identities at 150 dps.
- F4: analytic profile bounds |g_i| <= pb_i, |g_i'| <= pb1_i sanity-checked on
  theta in [-10, 10].
- F5: planar endpoint density diagnostics (supplement §5): 1.121261590e-3 at
  (0,2,0,0); 1.282523307e-3 at q = -1/4; planar K_J corner min 5.9833432e-14.
- F6: final assembly re-derived from the transcript's certified integers with
  exact rationals (c_lo, r_0, hard check).
- F7: mutation suite — each defect is rejected with nonzero exit in both modes:
  `eigenfloor_up` (upward-rounded eigenfloor 0.126553449817289 rejected),
  `hessian_power` (r^3 in place of r^4), `box_width` (16 delta^4 in place of
  32 delta^4 r), `planar_moments` (1 - a_2 = 0).
- F8: transcript byte-identity (normal vs -O).

Both `lpw_constant.py` and `falsify.py` run under `python` and `python -O`
with byte-identical program output (sha256 pairs in RECEIPT.json); run
metadata/hashes live in RECEIPT.json, not in program output.  ck() is
fail-closed (SystemExit nonzero); no bare asserts; no randomness; no wall
clock.

## 6. Disambiguation notes (not load-bearing for c, r_0)

- RB's part2 rung script zeroes odd-|alpha+beta| covariance pairs; pairs with
  odd x-order are sine sums and are nonzero for pin separation t != 0.  Their
  endpoint (t = 0) is exact; their rung table carries O(r) entry errors (e.g.
  Var(U_6) off by ~8/r^4), which do not affect the endorsed proof or this
  certificate: the T_r route with correct sine moments agrees with the profile
  route to <= 2.1e-56 (F1), and the profile route reproduces every reference
  value (endpoint Schur diag, conditional mean, lambda_min(Gamma_0) =
  0.126553449667289, center density 1.2832254e-3 vs RB's 1.2831682e-3, the
  4.5e-5 relative gap being precisely the omitted sine terms).
- The 1.28e-3 figures are point densities, not floors; the certified m is an
  inf over JBOX(R) x [0, R] by the contract's boxed formula.
- If a sum-convention C^3 norm were intended instead of the candidate's
  max-convention (§4), the field-moment bounds gain a factor 10 (number of
  multi-indices |alpha| <= 3); all checks would still pass with the same code
  path (constants degrade; positivity unchanged).

## 7. Scope boundary

This is a constant certificate for the endorsed LPW lower bound only: not a
matching 2D upper theorem, not a limiting-coefficient evaluation, not W8's
Lambda floor, not WP closure, not P0.1.  The single-rung W8 failure does not
enter this construction.  Mathematical positivity of (c, r_0) is reported
separately from any usefulness for numerical experiments (contract §6).

## 8. Deliverables (manifest: MANIFEST.sha256)

`lpw_constant.py` (certifier), `falsify.py` (executable falsifier),
`out_normal.txt` / `out_O.txt` (byte-identical transcripts),
`falsify_normal.txt` / `falsify_O.txt`, `RECEIPT.json` (environment,
dependency versions, input and output hashes), `MANIFEST.sha256`,
this report.  Input hashes: candidate cf58f72e...e1a5 (verified); contract
617cee2b...afb5a; endpoint supplement 650b4bca...26d8f.
