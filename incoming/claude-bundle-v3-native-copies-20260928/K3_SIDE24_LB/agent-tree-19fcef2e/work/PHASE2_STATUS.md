# PHASE 2 STATUS (resume point)

## RUN STATE (as of this update)
- 4 shard drivers running: `python3 -u phase2_cert.py {C,M,O,F}` (flock
  driver_{C,M,O,F}.lock; per-shard phase2_cells_{s}.jsonl,
  phase2_ledger_{s}.jsonl, phase2_{s}.live, phase2_receipt_{s}.txt).
- 2 keepers running: `bash keeper2.sh A` / `bash keeper2.sh B` (flock
  singletons, peer respawn via `bash`, shard relaunch budget 4, receipt
  stand-down, 30s heartbeats keeper2_{A,B}.hb).
- Cell order: easiest first (c6v inf descending) — hard small-c6v cells last.
- Resume after crash: just relaunch the same commands; banked jsonl/ledger
  entries are skipped (done-set from both files).
- 2 cores only: drivers get ~35% CPU each; first cell lands ~10-20 min.

## BUILD STATE (final, validated)
- Exact J3 jets mirror dag.py J2 (dag3.py import-time patch + build_frame
  spy captures the 9-frame; ONE J3 station run per assembly, ~25s).
- Order-4 SV/CS everywhere (optional s4; Faà inv/sqrt/exp/Phi; Phi M4 =
  0.551 verified numerically vs true max 0.55058784).
- Tight G_k: gk_cs = exact J3 center jets + chain sigma^4 recipe with
  FACTORED-det S4R sups (dets are exactly 2-term: xy^4(x-1), y^5(2x-1);
  7-13x loose, was 1e6x).
- Tight vser: symbolic reduction vser_k = red_k/c6v^e_k (e_k <= 5), pickled
  vser_sym.pkl, validated EXACT 0.0 vs dag.sdiv at two stations; CSs from
  J3-sdiv center jets + quotient-Leibniz sups (exact-Bernstein red sups,
  (1/c6v^e) Faà) — Leibniz floor shrinks 16^-d with subdivision.
- Outputs: J3-center + chain-sigma^4 recipe (cM/cS/cY/trc); lambda recipe
  [S0..S3] with lam.s4 from the chain. lam_assemble_J3 exact.
- Depth cap 6 (parameter change, recorded): C|0,0 measured to close at
  depth 6 (floor 8e19 at depth 0; recipe closes when 16^-d * rho_d^4 beats
  cM ~ 51.5).
- Per-cell sigma^4 sup box = depth-0 cell box (recorded).
- tp.DMAX runtime raises recorded (tp.py default 17 silently truncates):
  vser_tight 100, vser_symbolic 400; all pickled artifacts audited clean.

## KNOWN COSTS
- per-assembly ~2 min single-core (station_dag3 25s, vser Bernstein ~25s,
  S4R 15s, chain ~30s, rest ~25s); depth-0 cell ~2 min, depth-6 cell hours.
- Expect: c6v >= ~0.01 cells (739/843) certify at depth <= 4; ~104 cells
  c6v in [1e-3, 1e-2] need depth 5-6; ~26 cells c6v < 1e-3 may ledger at
  depth 6 with 'type-gate straddle (cM) (depth 6)' — arithmetic-attributed
  (vser sigma^4 conditioning), NOT a genuine obstruction (TRUE sigma4(cM)
  ~ 1.5e6 measured by exact J3 sampling at C|0,0).

## REMAINING WORK (in order)
1. Monitor shards (phase2_{s}.live); wait for 4 receipts.
2. Complete verify_lambda_grid_v4.py: full per-cell re-computation with
   byte-exact record comparison (skeleton written: integrity, completeness
   1707 = certified + phase2-ledger + phase1-ledger, aggregate max
   sigma(D^2 lam)/sigma(D^3 lam) sups, modes A/B byte-identical receipt,
   FREEZE block). Add the mutation suite (tampered jsonl copies must FAIL)
   and run both modes; merge shard receipts -> final report with hashes.
3. Ledger reconciliation: depth-6 ledgers stand as the failed-cell ledger
   (complete, honest, arithmetic-attributed). CANNOT-VERIFY stays separate
   (none so far).

## OPEN ISSUE (update)
- M|0,13 + O|0,9 (high-c6v cells) ledgered 'type-gate straddle (cM) (depth 6)'
  with the tight build — a residual cM-chain stage is still loose. Diag
  running: /tmp/diag.py (certify_cell on O|12,9, driver-identical). If a bug:
  fix -> clean phase2_* banks -> relaunch. If genuine deep arithmetic: ledgers
  stand. NOTE: running shards predate the vser dps-25 Bernstein speedup
  (~2-3x on the vser stage) — applies on next relaunch only; soundness
  unaffected (iv rounds outward at any dps).

## DIAGNOSIS RESOLVED (this session): the clip-1/high-c6v type-gate failures
Three stacked cancellation-blindness layers, ALL arithmetic (none genuine):
1. Constant-G_k sup blindness: 399/567 (9A) [391 9B] G_k entries are exactly
   Q_k = c*det^2 (certified constants, zero sups); 100 [108] more have one
   det factor (G_k = red/det). FIXED: s4r.GkSups.red reduction (const_val /
   e=1 quotient / e=0 paths) + gk_cs const short-circuit with exactness
   assert vs the J3 center value.
2. red sigma-sups now Bernstein-tight (sigma_sups_tp_bern) in e=1/e=2/poly
   and factored paths.
3. ROOT of the depth-6 failures: the depth-0 sigma^4-box rule (recorded
   parameter change earlier) FROZE the sups at coarse depth-0 values; the
   X9 Neumann tower (measured s4 ~ 1e15 vs TRUE sigma4(X9) ~ 1.3e3 by exact
   J3 finite differences at O|0,9) then never shrinks -> type gate
   unclosable at any depth. REVERTED: sigma^4 sup box = current sub-box
   (box-keyed caches were already per-box; sound, strictly tighter).
   Measured at O|0,9: TRUE cM jets tame (cM=0.77, s1c=2.07, s3c=30.4);
   W-path G0 T^1 sup 3.6e6 vs tight-path 54 (tight path 5 orders better).
VERDICT: bug (arithmetic), not a genuine obstruction. The 6 ledgered
Phase-2 cells (M|0,13, M|1,13, O|0,9, O|1,9, F|5,5,*) are INVALID under
the fixed build -> banks cleaned before relaunch.
- depth cap 6 -> 8: the X9 Neumann tower collapses ~1024^-d once
  R(d) = sum_j s0(gk(j)) < 1 (measured depth ~5 at h=0.1); h=1/10 cells
  certify at depth ~7-8, h=1/40 at ~3-5. (Measured: tower ratio 15-20 at
  O|0,9 depth 3; TRUE sigma4(X9) ~ 1.3e3 by exact J3 FD.)

## LEAD DECISION (Sep 15): run the full grid at cap 8 with the 3-layer fix;
map the frontier; report at completion or >=200 cells. STOP conditions:
ledgered fraction > 15% of grid, or any Phase-1-certified cell fails at
cap 8 (K9r decision re-opens then). F clip-mixed cells stay ledgered as
Phase-1 left them.
## OPS LESSON: box has 4GB RAM, no swap. Concurrent heavy diagnostics +
4 drivers -> OOM burst -> keepers exhaust relaunch budget. RULE: no
concurrent diagnostics while the run is live; stagger driver launches.

## RESUME POINT (Sep 15 ~11:45): run live with all fixes:
3-layer fix (const/det-reduced G_k + Bernstein red sups + per-sub-box
sigma4 sups) + depth cap 8 + interior-first ascending-c6v order +
s4r sups at iv.dps=25 (~2x speedup) + per-assembly STAGE timing prints.
s4r._sups25 key bug found+fixed (11:25:22); all drivers post-fix.
Measured stage costs (C cell): dag3=11s vser=85s asm9=208s (pre-dps25).
First cell outcomes pending (cold GkSups init ~5-15 min/driver).
OPS RULES: no concurrent heavy diagnostics (4GB box, OOM kills all);
stagger driver launches; keepers self-heal code fixes via respawn.
NEXT: first outcomes -> frontier map -> report at >=200 cells or stop
conditions (ledgered >15% or any Phase-1 cell failing cap 8).
- sigma-sups method change (on record): vser red + gk red sigma-sups
  switched from Bernstein to the certified termwise iv bound
  (sound: triangle inequality + iv enclosure). Measured only 1.3-1.5x
  looser than Bernstein (e1 red: 155 vs 107; e0 red: 848 vs 650) and
  ~1000x faster; Bernstein cost (400-470s/assembly under contention)
  priced out the run. Depth cost ~0.3 levels.
- stage costs now: dag3 ~11-55s, vser 0.5-2.3s, asm9 15-24s.
- tree visibility: STAGE/TREE(depth<=2)/CERT/LEDGER prints.
- turn-end reaping CONFIRMED as the recurring killer (all deaths
  coincide with turn ends or concurrent-diagnostics OOM). Keepers
  self-heal mid-turn; lead's :17 cron sweep wakes for relaunch.
- BUG (DMAX class, 3rd instance): tp_qk.pkl was built with tp.DMAX=17;
  the Q_k = adj*Spp*adj products (true deg <= 41) were SILENTLY
  TRUNCATED at deg 17. Effects: Q_6 wrong at 23 entries (X9 series
  validation caught it: 36/567 X_k mismatches, all order 6); s4r's
  reductions (red/const_val) for high-degree e=0/e=1 entries were built
  on truncated polynomials -> their sups were WRONG (potentially unsound;
  no cell certified under them, so no bad certs exist). Fix: rebuild
  tp_qk with DMAX=100, then rebuild x9_series + revalidate. The
  DMAX-17-vs-100 audit earlier covered tp_sections (deg <= 17 genuine)
  and vser_sym (DMAX=400) but missed tp_qk -- recorded.
- BUG (DMAX class, 4th instance): s4r.py:315-316 sets _tp.DMAX=100
  inside GkSups construction. Any TP product AFTER a GkSups init with
  degree > 100 silently truncates. This corrupted the x9_series X6
  numerators (deg ~142 -> 16 wrong entries, block pattern {1,2,7,8}^2).
  In the Phase-2 run: s4r init happens before vser/gk products (all
  deg <= 41+reductions) -- no truncation expected, and no cell
  certified, so no unsound certs; recorded for the receipt. Fix
  (x9_series only): reset DMAX=600 after build_R.
- X9 TOWER BROKEN (symbolic layer, all validated exact):
  * detR = det(G_0) = dnum/det^4: dnum deg 24 / 13 terms (pair-Bareiss,
    exact division collapses everything; validated == direct determinant
    and == 1/det(X0) at the station).
  * adjR via 81 pair-Bareiss cofactors; validated R.adjR == detR*I exactly.
  * X_k series (k<=6) = N_k/(dnum^a det^b), a<=4 b<=1, numerators deg <=
    102 / <= 289 terms, built in 18s (x9_series.pkl); validated 0/567
    against the DAG's exact X9 at the O|0,9 station. Files: x9_exact.py,
    x9_series.py, x9_css.py, x0_derivs.pkl (derivative polys, 2s build).
  * TWO DMAX-class bugs found en route (3rd/4th instances): (i) tp_qk.pkl
    was built with tp.DMAX=17 -> Q_6 silently truncated at deg 17 (true
    deg <= 41); rebuilt with DMAX=100, revalidated 0/567. The s4r red
    sups derived from the truncated Q were wrong for high-degree entries
    (no cell certified under them -> no unsound certs; recorded).
    (ii) s4r.py:315-316 resets _tp.DMAX=100 inside GkSups init; any TP
    product after a GkSups init with degree >100 silently truncates
    (corrupted x9_series X6; fixed by resetting DMAX=600 after build_R).
    ratpoly.deriv_polys also hardcodes DMAX=60 (reimplemented at 600 in
    x9_css._deriv_polys600).
  * REMAINING BLOCKER (optimization, not mathematical): per-box tight
    sigma^4 sups of the X_0 derivative polynomials P_a (deg <= 168, ~5k
    terms). Measured routes: (a) quotient rule on adjR/detR: FAILS --
    depth-independent floor ~1e5-1e17 (the 24*b1^4/uinf^5 term; detR's
    s1 doesn't shrink with depth) -- the true sigma4(X0) ~1e2-1e3 cancels
    adj/detR-internally, invisible to the rule. (b) deriv_polys exact
    derivative numerators + shifted-Taylor crude: CORRECT (cancellations
    exact in P) and the P-build is fast (2s/81 entries, banked
    x0_derivs.pkl) -- but the per-box tight sup evaluation of deg-168 P
    is the wall: iv binomial shift ~30-90s/P -> 10-30h/box naive; exact
    Fraction shift worse; origin-crude is cancellation-blind (useless).
    NAMED ROUTE for the next build session: K-truncated exact shift
    (exact Fraction Taylor coefficients for m+n<=K~8 via TP.diff +
    tp_eval_fr ~45 evals/P ~5s, crude-bounded tail suppressed by
    h^(K+1) ~ 1e-17) or iv-Horner substitution; target <= 2 min/box for
    all 81 entries x 15 P's. With that, X_0 CSs seed neumann6_cs's
    g0i_in (wired: phase2_frame.neumann6_cs(g0i_in=...), assemble9's
    x9_in, phase2_cert.certify_cell) and the gate (5 cells, depth <= 2)
    can run. cert wiring currently points at x0_cs_matrix (quotient
    route, floor-bound) -- MUST be switched to x0_cs_matrix_dp once the
    fast P-sup lands.
  * Frontier run remains PAUSED (.paused in place; 4 ledgers banked).
- X_0 SUP ROUTE RESOLVED (measurement): the exact-Fraction center shift
  of the banked derivative polynomials P_a (x0_derivs.pkl) is TIGHT
  (sup 3.7e-29 for the order-4 P of X0[1][1] at O|0,9 -- the dnum^5-
  scaled value scale) at 2.6s/P, while iv/Horner enclosures are ~25
  orders loose (dependency problem; measured). Per-box cost ~53 min
  (81 entries x 15 P's). x9_css.x0_cs_matrix_dp now uses the exact
  shift; phase2_cert wired to it (was the quotient route, floor-bound).
  X_0 s4 estimate ~5e3 at depth 0 (vs 4.7e17 quotient floor). Gate
  runner gate5.py written (5 cells, DEPTH_MAX=2); x0full.log measures
  the per-depth sups at O|0,9 first (keeper_gate.sh, flock-guarded).
