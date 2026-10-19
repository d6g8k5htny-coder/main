# Own result, recorded before reading REVIEW_A_V2.md and REVIEW_B_V2.md

Claim `M193-V2S2-XFAM-INPUT-017-20261019`, pickup Math-193/6278524199 (22:13:07Z).
Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. Recorded 2026-10-19T22:24Z.
Input only; organizational-independence credit 0; scientific effect NONE.

## Sources
- V2: Math-#483 head 963d70f7, frontiers/full_spectrum_enclosure_method_20261018/PROOF.md,
  blob 74746107, 32,593 B, SHA-256 8bdea4b42e9efd2d846cd1aa73691e8194fa26cd937ac02b79991334f88204f6.
- ORIGINAL: block in Math-193/6243478311 between the BEGIN marker line and the END marker,
  minus the one newline the relay adds: 24,027 B, 629 lines,
  SHA-256 f38c085ab5b15368e6989a1fd6298490f3b671f5a6c3a01bcdc81b516022bc33.

## Result: CONCUR — V2 §2 (PROOF.md lines 24–68) restates the ORIGINAL faithfully
No blocking, material or minor finding. Every displayed formula and every normalization
statement in §2 matches the ORIGINAL, with renaming only (q_m = h m, q_m^2/2 = alpha m^2).

| V2 §2 element | ORIGINAL (line) | result |
|---|---|---|
| physical torus, volume L^d | L12–14, L308–309 | same |
| h, alpha, z, mu_2k | L66–72 (q_m, z_L, mu_2k) | identical after renaming |
| K(x)=E cos(Q.x), P(Q_i=hm)=e^{-alpha m^2}/z, Z_L=z^d | L83–85 | same |
| centered, constant mode retained, no samplewise mean removal | L86–89 | same; unit variance is K(0)=1 |
| a,b,c, kappa4, kappa6 | (4) L76–80 | identical |
| M1, K_d(a) | (1) L21–28 | identical (V2 merges the two prefactors) |
| d omega = full surface area | L31–32 | same |
| M2 B(e) | (6) L117–118 | identical |
| M3 V(e) | (8) L132–137 | identical |
| M4 A_e, F, any orthonormal U | (10) L161–168, L198 | identical (w(A) is V2's name for the weight) |
| Gamma_e = ORIGINAL (11), s(A) entries once | L176–212, (11a) | same |
| height unconditioned; f=0 conditioning changes Gamma_e, F | L171–174, L375–377 | same |
| even jet independent of gradient and odd jet; V after G=0 | L111–112, L129–137 | same |
| maximum first, saddle second, full sphere | L229–231, L365–367 | same |
| gap ell = u r^3/12 under both zero gradients | (16) L262–273 | same |
| no role-counting half | L367–371 | same |
| no extra L^{-d} | L308–309, L380–381 | same |
| d=2: F = Var(A_e)/2, half from the negative half-axis | (12) L214–225 | same |
| h = 2 pi/L, not a normalized wave number | L378–379 | same |

V2 uses only the right side of ORIGINAL (1) and does not import the barcode identification
(2)–(3); that separation (V2 §1) is a correct narrowing, not a fidelity defect.

## Exact checks (sympy 1.13.3): 38/38 PASS
check_exact.py 68874bdacfcae0cf664e573936430aceb7dc0ac67237f41dcda7fd77679c2151,
output d324b05fa2da5afea93ec0abe51e41dbe0e369a416523facfa970c2ea8f4d338.
- d = 2..5, symbolic e, moments taken directly from the coordinatewise law: (5); M2 = (6);
  (7); (9); M3 = (8).
- d = 2..5, exact rational orthonormal frames (Cayley): R and Cov(U^T H U) blocks of (11)
  equal the tensor contractions; Gamma_e equals (11); at a=1, kappa4=0, Gamma_e is the
  covariance of GOE_(d-1) + sqrt(2/3) Z I, as ORIGINAL (30) says.
- d=2, 12 exact directions: Var(A_e)/2 = ORIGINAL (12).
- symbolic d: 12^(-1/3) p_G(0) p_He(0) times the (23) prefactor equals K_d(a); (23); (21);
  M1 at a=1, kappa=0 equals ORIGINAL (31); d=2 gives (32) = 0.0734069193060342710301359629577740500177.

## Numerical checks (mpmath 1.3.0)
check_numeric.py f77b61a1175f3b17402efb823138a8b9173f9ec7cbfc40d1f58245a087f3517b,
output f8df98c0ea1280ce3b87c2d011851ff89ecde3a37804889a598cef5e4a32d75f;
check_numeric_b2b.py de20d6d3913183252ff3c6b213023887401d3de321b4b3edb6f93bcb87fae6b4,
output 7ac0253d32c01bdc9ad915853545731f48cd6b72d865750db11673d240937e4a.
- Moments from V2's text and from the ORIGINAL's text agree exactly at L = 1, 2, 3
  (2.5e-41 at L=24). Finite-L terms are large: kappa4/a^2 = 1.9e8, 67.5, 2.54 at L = 1, 2, 3.
- Closed forms M2, M3 and (11) against direct d-dimensional sums of the literal lattice law,
  d = 2 and 3, L = 1, 2, 3, 4–5 directions each, 80 digits: worst relative error 1.1e-56 at
  L=1 (on-axis V = 2.27e-28 cancels about 24 digits), 3.9e-76 at L=2, 2.0e-78 at L=3.
- C_alg(2,L) from V2's path (M1, F = Var/2 from (11)) and from the ORIGINAL's path ((1) with
  (12)) agree exactly at L = 1, 2, 3, 24: 2.353105115826696e-8, 0.012617593130692105902,
  0.049879432666614182044, and at L=24 the value of (32) to 30 digits.
- d=3: F(e) by (11a) (2-D reduction, 20 digits) from closed-form (11) and from the literal
  law agree to 1.5e-13 (L=1), 2.6e-20 (L=2), 2e-21 (L=3); Monte Carlo of definition (10)
  agrees within 0.4 sigma at L=3 and for F*_2 = 2.38384359055016 at L=infinity, which with
  ORIGINAL (31) gives c_{3,inf} = 0.0417759318405983.
- These numbers confirm my reading and the closed forms against the literal law; the agreement
  between the two text paths is expected once the texts are identical.

## Post-hoc observation, adjacency, not part of this result's basis
After computing B3, I compared with my own 5 October main#259/6002327092 diagnostic from the
Math-#297 checker (which I wrote): c_{2,L}/c_2 = 0.171885610375 (L=2), 0.679492248662 (L=3),
c_{2,2} = 0.012617593130692106. V2's M1 gives 0.171885610375355, 0.679492248662095 and
0.0126175931306921059. The ORIGINAL states it was derived without reading project coefficients.

## Notes, non-blocking
- Corroboration of Claude S's N2: at L=1 the on-axis V(e) = c - b^2/a = 2.27e-28 while
  c = 3.3e-4, so evaluating M3 from moment enclosures loses about 24 digits there.

## Not established
- ORIGINAL's barcode identification (2)–(3), G1–G4, §§8–9; M31.
- V2 §§3–10 (Claude S's input) and §§1, 11–13.
- Any interval enclosure, project test, Lean or hosted run; no repository write.

## Exposure before this record
- Pickup list (V2 §§1–2 and outline; #483 README, ROOT_V1_ADVERSE_REVIEW, V1_DOMAIN_FINDING_NATIVE,
  ROOT_V2_FINAL_DISPOSITION; Claude S's pickup and input; openings of 01Cz's 6243464440,
  6243882168, 6244357886, 6246944026; 01Cz's 6249533285).
- After the pickup: 014K's offer 6278343295 and C8's pickup 6278469127 (main#419); 01Cz's
  6243882168 §§3–5 (its comparison of the binding, not V2, with the ORIGINAL), read at 22:14Z
  for my note to C8, before my own comparison; ORIGINAL lines 1–439 and 575–629; my own
  main#259/6002327092, after B3.
- Not read: REVIEW_A_V2.md, REVIEW_B_V2.md, the binding PROOF.md.
