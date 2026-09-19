# DECISION_LEDGER — K3 swarm

2026-08-05 intake: K3 blocking defect CONFIRMED (missing CS square root; witness-point 175.3x inflation).
  Invalidated as upper bounds: I_cs 1.30399e-5, WP ub rung table, 5.5e-3*r^1.6 envelope, WP-ub AO branch,
  THM-023 v1.1 WP row. Manuscript errata issued (SIDE24_pre_peer_review/ERRATA_2026-08-05.md).
  Unaffected (verified): LB-1 above-b channel (sqrt-valid), I_true estimate, witness laws, m(P*)/m(Q*),
  DER-027c Kantorovich, DER-027a variance channel, SARD-G, measurability, ell^{-1/3} law.

2026-08-05 dispatch: 11 workstreams live (W2,W3,W4 frozen-before-compare discipline; W12/W13 staged).

2026-08-05 freeze receipts:
  W2 symbolic (FROZEN, body 26e2d1dd3b36f1f7..., 114/114 checks) — bound chain min{UB_global, UB_slice},
    probe UB_slice 5.0e-23 vs rigidity budget 3.6e-15 (11 orders headroom); structural finding: 9-pin
    conditioning breaks isotropic independence; quadratic-form machinery required.
  LEAD symbolic (FROZEN, LEAD_SYMBOLIC_WP.md) — independent, predates W2.
  GATE G1: PASS (agreement on integrand + validity classes; see CANONICAL_STATE.json).
  W5 forensic (FROZEN, 9b033b57...): rigidity figure = single-station lam_sad(d=1) x full disk area
    (3.5578e-15); 0.213r3 = rigidity + transition 1.9513e-6 + far 1.3701e-6 = 3.32143e-6 = 0.2125714r3
    (matches c030 assembly.json to 3.6e-12); kill premise entered at C030 pkg line 5 licensed by line 11;
    G-F2a d1 FAILED-AS-WRITTEN waived; verbatim zone/collar definitions extracted; record gaps named.
  W7 gamma-LOC (FROZEN, DISCHARGED-BY-MINIMALITY): no node consumes 1-O(exp); minimal statement
    gamma-LOC-min discharged with r0^3 = (20*0.089569/7)*P0; r0 = 0.04343 at P0 = 3.2e-4 (< 0.05);
    'sufficiently small r' is P0-unconditional; F4 does not fire (W10 confirms constant c_cond class).
  W10 dependency (FROZEN, 67c7fe63...): assembly does NOT need WP = O(r^3); minimal WP lemma = any valid
    explicit modulus E_WP(r) -> 0 covering (0, r0]; o(1) buys liminf form with 0.9144-class coefficient;
    honest remainder (1 - O(r))*(1 - O(r^{1.45})), not (1 - O(r^3)); eta_r explicit, no LB-side loss.
  W11 literature (FROZEN, 5dcd89be...): methods stack (DE quadrature explicit constants; Arb engine;
    Marcum-Q; HW 144/16sqrt2; Laurent-Massart; Kantorovich-Gragg-Tapia radii; r^{2k+2} as derivation).

Routings: W10 -> W6, W8 (acceptance targets); W11 -> W3 (methods menu); W7 r0 addendum requested.
Held for W3/W4 freeze receipts: W2 bound chain, W5 forensic, W10 WP-min spec (independence discipline).

2026-08-05 lead cross-validation of W2's frozen probe (lead's independent pipeline, mpmath dps-60):
  at (1.0, 0.3), r = 0.025: mu_t = 1.429401 (> b; rim-arc point), Pw = 2.03e-22 (window 8.9 sd below mean),
  lead valid global-CS ub = 5.76e-12 vs W2 UB_global 3.5e-12 (same order; bound-family variation);
  lead slice-scale 8.2e-23 vs W2 UB_slice 5.0e-23 (same order). CONSISTENT. Note mu_t > b here confirms
  the rim-arc mean-above-b structure inside the rigidity zone (premise-falsification geography).
2026-08-05 G2/G6 prespecified and registered before any W3/W4 comparison (see CANONICAL_STATE.json).

2026-08-05 freeze receipts (cont.):
  W4 independent implementation (FROZEN, W4_REPORT 669401f7..., MANIFEST b8399855...):
    I_WP(0.025) = 4.7569e-6 ESTIMATE (+-0.4%), wrapped-theta real-space + direct transformed-coordinate
    quadrature (no spectral/GH); dominant wedge theta~90deg, peak 1.0729e-4 at (-0.0025, 0.580);
    pin collars dead (rho <= 1.1e-171); KIMI witness exact rho 7.2e-21 (not dominant);
    corrected-CS integral 8.5743e-4 (180x estimate; global CS alone cannot close WP);
    f-H correlation sign flip via Y pin independently found; probe laws at 25 digits filed.
    Cross-validation vs lead's spectral/GH estimate 4.7602e-6: |delta|/value = 0.07% (G2 tol 5%).
  W9c DER-027c scope (FROZEN, 5b8ef639...): r-uniform ridge certification SUCCEEDED at house grade,
    every r in [1e-6, 0.05], 22 chained blocks, uniform alpha <= 0.341, margins converge
    (lam_inf 2.5654/1.9428); interval tier CLOSED-NEGATIVE with measured obstruction (iv pivots
    straddle zero at 2e-4 blocks; viability needs |I| <~ r^8); 36 certified rungs to 1.455e-12;
    nonuniform obstruction = pin-Gram conditioning (recedes with dps), NOT topology.
  W8 Lambda-side (RUNNING): nonclosure-or-closure document drafted placeholder-free; exact missing
    piece = certified cell-sup sigma(D^2 lambda) / regional sup|D^3 lambda| (interval eval 7500x junk,
    global sup-bounds compound >=1e10x); tiers 0.946 measured / 0.9091 derived / 0.9001 H-B3 floor.
  W9a DER-027a scope (RUNNING, ~80%): F64PAD replaced by interval box enclosures (d=3 at interval
    grade, 12/12 variance knots PASS); uniform continuation in r SOLVED on (0, 1/20]
    (obstruction: lambda_min(G(r)) ~ r^5.75 rank drop 7->3).

2026-08-05 freeze receipts (cont.):
  W6 uniform-r (FROZEN, W6_REPORT beda91f6..., certs byte-identical):
    WP-min SATISFIED: E^0_r[N_ws(B3\collars)] <= I_CS(r) <= E_WP(r) = 4.2*r^{3/2} on (0, 0.05];
    pointwise corrected-CS validity PROVED (max exact/CS = 0.045 on 84 samples); 21-rung grid certified
    (scaled coeff max 3.9697 <= 4.2); E_WP(0.05) = 0.0470 <= 0.06 budget. Two NAMED premises (C-3 pattern):
    F1 log-slope <= 0.327 (measured <= 0.20); F2 sub-0.001 tail |C(r)-C0| <= 0.25, C0 <= 3.95; F3 quadrature note.
    Limit jet exact: span ann(n), n=(0,1,21/2,209/3); f_xxx -> +2 (trap -4 refuted: gradient pins force
    f_x(0) = -r^2/4); c = -3476069/6953125 confirmed to 1.5e-13 by independent pipeline.
    Variance laws: v_t ~ r^4 REFUTED (fixed-y r^0+O(r); cluster v_t ~ r^8, v ~ r^6; r^4 is the crossover).
    Transient mechanism: apparent 1.45-1.6 slope on [0.005, 0.05] is the z-drift transient; U-turn at r ~ 0.003;
    honest remainder (1 - O(r))*(1 - O(r^{1.45})).
  FLAG for G2/red-team (estimate reconciliation): W6 truth I_WP(0.025) = 7.50e-6 vs lead 4.7602e-6 and
    W4 4.7569e-6 (those two agree to 0.07%). All three exceed the falsified budget 3.328125e-6, but the
    57% gap must be explained (likely domain/shell handling) before the assembly prints a truth figure.
    W3's rigorous enclosure arbitrates.

2026-08-05 estimate reconciliation RESOLVED (lead): W6's truth 7.50e-6 is over the FULL WP channel domain
  B3\collars (radius 3); lead's 4.7602e-6 and W4's 4.7569e-6 are the rigidity zone d <= 1.5 only.
  Domain decomposition: rigidity 4.7569e-6 (W4) + outer shell 1.55-2.5 = 1.5866e-6 (W4) + far shell
  (2.5-3, print 1.37e-6) ~ 7.5e-6 = W6's full-zone figure. All three estimates CONSISTENT.
  Consequence: the WP channel truth ~ 7.5e-6 = 0.48*r^3 at r = 0.025 exceeds the falsified channel
  budget 0.213r^3 = 3.328125e-6 by ~2.25x (rigidity piece alone by 1.43x). Collar neighborhoods
  contribute <= 1.5e-18 (W6) / rho <= 1.1e-171 (W4) - dead.

2026-08-05 freeze receipts (cont.):
  W9a DER-027a scope (FROZEN, report cdf8c973..., certificate 94 checks PASS both modes byte-identical):
    scope table C1-C7: variance channel (d0 = 3, d in [3,12], any pinned values, now INTERVAL-GRADE
    with rigorous mean-value box enclosures: Delta(3) <= 0.0209033/0.0216751 (7-pin) <= 2.24e-2,
    12/12 knots PASS; d=5 <= 4.2e-8; d=6.1 <= 7.04e-13); value-law channel = full one-point-law TV
    bounds ONLY for the 6-pin exact-beta family; v* = -1/2 isolation CONFIRMED at interval grade
    (variance channel bitwise value-free by mutation M1). Uniform continuation in r SUCCEEDED on
    (0, 1/20] via jet frame (tau(1/20) = 6.78e-6/5.03e-7 < 1); nonuniform obstruction named:
    raw-frame lambda_min(G(r)) ~ r^5.75 rank drop 7->3; projection form bypasses it; uniform
    constant ~0.61 at d=3 weaker than exact-rung 0.021 (exact-rung status preserved as the sharp
    statement). Remaining gaps: d >= 4 Lipschitz pad (sharpness not rigor); r > 1/20 on demand
    (tau ~ r^3, threshold r* ~ 0.08-0.1); closed-form non-IB proof of the d=3 constant open.
  W7 r0 addendum (FROZEN, c0dedd63...): r0(P0) = (20*kappa/7)^{1/3}*P0^{1/3} = 0.634887184*P0^{1/3}
    (r0 = 0.04342567254 at P0 = 3.2e-4; 0.02946885264 at 1e-4; 0.006348871840 at 1e-6);
    exists-vs-explicit stated precisely (P0-unconditional existential; explicit needs OBL-P0-FLOOR);
    coefficient reconciliation: ledger 20*kappa/7 = 0.25591142857142857 vs area-ratio 54*kappa/19 =
    0.25456452631578947 (structure r0 ~ (theta*kappa*P0)^{1/3} invariant).
  W6 addendum (F2 strengthened): C0 = 3.860069 computed DIRECTLY from the limit law (limit Gram,
    no rungs), confirming C0 <= 3.95; report re-hashed b61835f0....

2026-08-05 W6 SELF-CORRECTION (preempting the red team): the modulus E_WP = 4.2*r^{3/2} is INVALID as
  stated for r <~ 0.003: the global-CS bound has a genuine near-cluster singularity on the ARCH direction
  (rho_CS ~ ell^{1/2} d^{-4}; a bound artifact - the truth is dead there, rho <= 1e-171 on rings 0.002-0.05
  per W4). Scaled coefficient refines 3.97 -> 6.29 (still climbing) at r = 0.0015. UNAFFECTED: the
  pointwise CS validity (PROVED), E[det^2] MC validation, mandates 1-3 (limit jet, c, variance laws),
  and the assembly-relevant value E_WP(0.05) = 0.047 <= 0.06. Fix in progress: (a) converged CS law for
  correct (alpha, C); (b) zone split - UB_slice near the cluster (kills the artifact), global CS outside.
  Recorded as a campaign-caught defect (pre-red-team), upgrading FAILED_APPROACHES #13.

2026-08-05 W12 RED-TEAM (5e6b00bb...) findings and dispositions:
  B1 HIGH - W6 modulus invalid r <~ 0.003 (self-registered, independently corroborated; the frozen report's
    collar-deadness claim was a category error: exact-rho deadness != rho_CS deadness). G6 OPEN. C1 blocking
    clarification for minting: zone-split repair + converged-quadrature recertification.
  B2 MODERATE - lead Isserlis display drops mean terms (overcount exactly 2(det nu nu^T)^2; independently
    verified by lead to 1e-9). Corrections filed: LEAD addendum, 00_READ_FIRST, CANONICAL_STATE, this ledger.
    Canonical values corrected (1.13229e-4 / 1.98542e-2; ratio 175.35 unchanged). Blast radius checked:
    DER-025 D2-dependent (quarantined anyway), LB-1 above-b bounds (conservative, margins absorb), LB-2
    (unaffected), W4/W6/I_true (direct quadrature, unaffected).
  B3 MODERATE - F1 mis-quantified (worst gap 4.236 > 4.2 under stated premise; 4.129 under measured slope).
    In W6's repair scope.
  B4 LOW-MOD - gamma-LOC retirement scoped to Forms B/C/D; Form A's exit node still consumes P-NMZ-gamma.
  HOLDS: W2 (114/114 re-run), W4 (off-grid probes, no hidden mass), W6 limit jet/c/variance laws, W10
    (F0-F3 exhaustive), W9a/W9c, W5, shell/gate register (properly noncontrolling).
  Red-team disposition: strongest defensible result = Form C liminf at measured/mixed tier (0.9144-class);
    0.213r3 refutation solid at estimate tier; formal refutation needs W3's box integration; Form A not
    closable today on the WP node (B1).

2026-08-05 W6 modulus REPAIRED (exact-integrand form): E^0_r[N_ws(B3\collars)] <= I_WP(r) <=
  3.5e-3*r^{3/2} on (0, 0.05]; validity PROVED (Kac-Rice equality + zone subseteq B3, rho >= 0;
  NO Cauchy-Schwarz loss, ~1000x tighter); 11-rung grid certified (C_I sup 3.202e-3 at r0);
  E_WP(0.05) = 3.91e-5 <= 0.06 (margin x1533); single named premise P-mono (d log I_WP/d log r >= 3/2;
  measured 1.66-2.37 asymptote 3). CS modulus 4.2*r^{3/2} REFUTED and superseded; red-team B1/B3 CLOSED.
  Gate G6: PASS at the W10 spec. Gate G12: minting awaits W3 (formal lower bound) and W8 (Lambda-side).

2026-08-05 W8 Lambda-side (FORMAL NONCLOSURE at the certificate standard, b9ca525d...): the exact
  missing piece is a certified cell-sup sigma(D^2 lambda) / regional sup|D^3 lambda| (interval eval
  ~7500x cancellation junk; global sup-bounds compound >=1e10x - both routes documented dead). Tiers:
  0.946 measured / 0.9091 derived / 0.9001 H-B3-conditional certified floor. r->0 limit object
  certified; r* continuity radius not certified. Transcripts (~2-2.5h) resumable via region caching;
  autonomous finisher armed. Assembly H1 finalized with the conditional floor; liminf constants
  0.9144036 (measured anchor) / 0.8705 (H-B3 floor) displayed as mixed-tier anchors, not fixed constants.
