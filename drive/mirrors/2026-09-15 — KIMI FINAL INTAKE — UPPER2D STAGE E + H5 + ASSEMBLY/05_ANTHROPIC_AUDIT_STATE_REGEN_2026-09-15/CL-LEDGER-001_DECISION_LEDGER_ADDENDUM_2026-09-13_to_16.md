# CL-LEDGER-001-v1.0 — DECISION_LEDGER ADDENDUM, 2026-09-13 (post-H4) → 2026-09-16

AUTHOR: Claude (Anthropic) · CREATED: 2026-09-15, UPDATED 2026-09-16 · CLASS: LEDGER — addendum to `K3_SIDE24_LB/DECISION_LEDGER.md`
STATUS: PROPOSED (for the operator to append; the frozen ledger is not edited) · AUTHORITY: none
FORMAT: the ledger's own — date, lane, DELIVERED/DECIDED/REFUTED, carrier + body hash. Every hash from bytes.

2026-09-13 H5 (C1-lane raw integral, H5_closure/ body 465c97c0…) DELIVERED:
  Certified interval upper on the chart window-saddle integral at r = 0.05; box cover with rim/axis at
  named-lemma grade; v1 totals 731.4311·r³ (frozen). Later tightened (refine-3) to v3 clean 647.8048·r³
  (h5_totals_v3.json 8d7028e4…); v2 totals I_lo CONTAMINATED by a rung-unscoped merger glob (root-caused
  2026-09-15, H5_TOTALS_ERRATA; I_hi verified clean two ways). H5-AXIS v3 frozen in full in the tightening
  amendment (body 7fefa17b…).

2026-09-13/14 D1 (assembly, D1_ASSEMBLY.md → v2_2.md) DELIVERED / STAGE-E REVIEWED:
  v1.0 006b8a7d… → v1.1 634338b4… → v1.2 a7e1958c… → v2.0 86882dca… → v2.1 bcc177fe… → v2.2 490ad6b2… (FINAL).
  Stage-E (five reviews: proof/numerics/provenance/scope/topology) broke and repaired: F1 H4-JC level
  conflation (→ H4JC-R1 event-level, 42ee88da…, OLD 132/175 violations pinned, NEW 0/439); F2 Z-rung
  out-of-scope (→ R2 certified interval Z_0.05 ∈ [7.7592917375327855e-3, 1.1468646473404396e-2]); F3 κ_far
  dropped (→ κ_far ≤ 0.68, B_remote = 21.9279·r³ floor-consistent, MC-free); hypothesis gaps → named premises;
  gate-red process break → whole-file cks in gate v3; V2 → B4.loc moved to validity premises.
  THEOREM D1 v2.2(1): E_w(0.05) ≤ 8.1272827e-2 = 650.1827·(0.05)³, C_RN ≤ 3.46, on Reg ∩ TYP, named-hypothesis
  grade. THEOREM D1 v2.2(2): ∃ r₀, C: 1 − q ≤ C·r³ on (0, r₀], CONDITIONAL on five validity premises.
  Gate d1_falsify_v3.py: 91 cks + summary, digest d800849e…; both modes byte-identical.

2026-09-14 BRANCH (BRANCH_dir/ body 8821c8fa…) DELIVERED: B1.dir ⊆ NearSwap(d_max) ∪ FarRoute(d_max),
  d_max = 4r–5r; NearSwap CERTIFIED super-algebraic (barrier margin Θ(1/r)); FarRoute handed to PERC.
  rung05: N = 4000, 1−q = 1.162·r³ (evidence); rung025 LANDED 2026-09-14T21:47:19Z, N = 1500, 1−q = 0.688·r³.
  Receipts errata 7547e76a… (determinism-charter repair; certificate + falsifier byte-identical both modes).

2026-09-15 RETURN_06 ASSEMBLED (00:05:36Z): executive state, theorem table, DAG, obligation ledger,
  adversarial report, running tasks, package verification; SOURCE_CAPSULE 22/22; TREE_MANIFEST 1031 files.

2026-09-15 H3 (H3_closure/ H3_BAND_FLOOR.md body 281477c3…) DELIVERED: CERTIFIED uniform normalizer floor
  E[G_r] = Z_r/r² ≥ 2.30659559567154 > c_Z ∀ r ∈ (0, 0.05], worst margin +42.78%; no whitening, no MC,
  exact Laurent-series Gram algebra + exact series division (J = 60). REFUTED: the whitened/box pipeline for
  interval-r (LDL pivots destroyed; Gram-entry width 3.3e7). Discharges the normalizer side of OBL-D1-PROMOTE.
  ADDENDUM-1 to RETURN_06 (00:42) records it.

2026-09-15 D3 (D3_percolation/PERC_DECAY.md body 5137a811…, engine ee68ac75…) DELIVERED + REFUTED:
  The o(r³) far-lane reading (PERC-DECAY as phrased in v2.0–v2.2) is NOT reachable: window ℓ = r³/6 fixes
  Θ(r³); certified corridor-alive display (μ − s > 0 from S to x = 1.0, bottleneck z = 0.68 = Θ(1)); raw
  window count Θ(r³) at every D < 12. RESTATED: far lanes absorb at Θ(r³) with certified constants —
  B1.dir-far D3-part ≤ 23.2119·r³ (FarRoute ⊆ {N_w ≥ 1}); B2-far ≤ 17.6802·r³; B4.rem local dam per-section
  κ = Θ(1/r). PD-CONN named as input (constants-not-order): (i) explicit-constant planar-BF RSW arm bound,
  (ii) field-level Kolmogorov-entropy patch, (iii) torus→planar (≤ 1e-120). Falsifier F1–F7; FREEZE 6bbe6e27….

2026-09-15 D2 (B4LOC_damline/B4LOC_DAMLINE.md body 0d5c1b32…, driver bf3b0225…) DELIVERED — CLOSES a validity
  premise: THEOREM B4LOC-R1: B4 ⊆ E1 ∪ E2 (exact, deterministic; loc + rem); Borell–TIS sup-tails under the
  6-pin law with all constants explicit; P_r(B4) ≤ 1.22e-9 / 1.52e-45 / 1.03e-197 at r = 0.05/0.025/0.0125;
  grade exp(−c_eff/r²), c_eff = 0.0513/0.0645/0.0709. The cut-net ≡ 9-pin-tube identification PROVEN FALSE
  (law/tube/level/scaling); 9-pin tube = proof template only. H4-SD distinction verified. Discrete mirror
  3 B4 corridors cut / 387 clear none-B4. Mutations M1–M3 fail closed. FREEZE 5235f98a….

2026-09-15 D1 (D1_ASSEMBLY_v2_2_REGISTER_NOTE.md body c1d5e95d…) DECIDED (effective at v2.3):
  CLOSED: B4.loc dam line AND B4.rem (whole-B4 ADJUDICATED YES — a wrap route at level s is by definition not
  B4); PERC-DECAY in restated form (validity content absorbed into E_w accounting); dir remainder at the rung.
  ADVERSARIAL LEDGER: the o(r³)-far-lane reading REFUTED. STRENGTHENED Theorem (1′): A, B1_wit, FarRoute
  pairwise disjoint ⊆ {N_w ≥ 1} → one E_w payment; the C_RN·√Q(B1.dir) term prices nothing beyond NearSwap +
  free-rider; v2.3 form 1 − q(0.05) ≤ 8.1272827e-2 + P(NearSwap) + P(B2) + P(B4). DEMOTED: OBL-B1-BRANCH(loop|B1)
  to refinement. REMAINING VALIDITY PREMISES OF THEOREM (2): OBL-D1-PROMOTE, D3-LEMMA-RN-UNIF. Flagged:
  B2-far free-rider pinning (N_β* ≤ N_w vs C2's exact definition). ADDENDUM-2 to RETURN_06 (03:34) records the
  two carriers with recomputed hashes.

2026-09-15 H1/LPW (LPW_CONSTANT/OPTIMIZATION_ANALYSIS.md, v3/, v4/) DELIVERED:
  Exponent ledger: e = v + w − n = 1 + 4 − 2 = 3 (Lemmas L1–L4; w = 4 forced by the pins). Three levers:
  L1 density floor m ≥ 1e-21 → 9/10000 (exact quadratic-form minimum, 16 vertices; ×9.1e17); L2 δ = 1/1024 →
  14587/2621440 with split ε′ = 57/640, clearance exactly 1/3840 (×959.12; δ = 1/256 REFUTED — fails path
  clearance); L4 normalizer cap 4B₃ → U = 3.66282864761194 from H3_BAND_CEIL (×2.2737e12).
  v3 (06:51): c = 9.8040863135804911886149013570e-23, r₀ = 1/2278031360 = 1/(2¹⁸·8690); F-LEVER4 FLAGGED (band
  FLOOR is the wrong direction for the Palm denominator; refused). Receipts errata (07:26): three wrong
  carrier hashes in the human-readable record only; net factor ×959.96 → ×959.12.
  H3 (H3_BAND_CEIL.md body cfe8a3a4…, 07:37): CERTIFIED uniform ceiling E[G_r] ≤ 3.66282864761194 on
  (0, 0.0025], ≤ 3.74767948915996 on (0, 0.05], cap U ≤ 4 never widened; REFUTED: naive p̄-only envelope
  (9.55 vs 3.23); (0.03,0.04] cell CK_FAIL at 07:26 root-caused to interval ratio-dependency in the α path,
  repaired via Chernoff–CS by 07:37.
  v4 (07:49): c = 394827584594841472652359498125 / 1771235249968883322980290175117427736576 =
  2.2291086664054236617851686509e-10 ≥ 2.22e-10 ≥ 10⁻¹⁰, r₀ unchanged; ONE changed interface, reviewed PASS
  WITHIN SCOPE; new mutation ceiling_consumed_above_U. Cumulative c gain v2→v4 ×1.9626e33; r₀ shrank ×1024.

2026-09-15 D3 (d3_rn_unif.py, 07:43, UNFROZEN) BUILT, UNRUN: the D3-LEMMA-RN-UNIF rung-part closure engine
  (Piece 1 rigidity-decoupling via residual forms + moment-series envelopes; Piece 2 certified Riemann sum);
  stops before invoking its own adaptive certifier. Not recorded in RETURN_06 or any addendum.

2026-09-15 H5 (H5_closure/H5_RUNG2 body 91a34d83…, H5_RUNG3 body f4c3414f…) DELIVERED:
  Rung 2 (r = 0.025): I_hi = 1.038121775e-02 = 664.3979·r³; rung 3 (r = 0.035355): I_hi = 2.923233277e-02 =
  661.4712·r³. Ladder 647.8048 / 661.4712 / 664.3979: +2.11%, +0.44% — decelerating; C_unif 731.4311 margin
  10.1%; falsifier not tripped; coverage census identical to rung 1 (r-scaled cover combinatorially
  invariant). Rung-2 stitch-drift alarm (power ≈ 2.73) RESOLVED at rung 3: first-step offset = rung 1's
  refine-3 advantage; equally-treated rungs scale r³ to 0.54%. Rungs 0.0177 / 0.0125 mid-flight at snapshot.
  NOTE (CL-ERR-001 E4): generating driver code not pinned at current bytes.

2026-09-15 W8 (agent 19fcef2e-c1c2, PHASE2_STATUS.md) DECIDED / REFUTED:
  Diagnosis: three stacked cancellation-blindness layers, all arithmetic (constant-G_k sup blindness; red
  σ-sups; depth-0 σ⁴-box freeze) — VERDICT bug, not obstruction; six ledgered cells invalidated, banks cleaned.
  Lead decision: full grid at cap 8, hardest-first order; stop if ledgered > 15% or any Phase-1 cell fails.
  DMAX-class bugs, instances 3 and 4 (tp_qk DMAX=17 truncation; s4r resets DMAX=100 in GkSups init): no cell
  certified under them → no bad certificates. X_0 sup route RESOLVED by measurement (exact-Fraction shift
  tight at 2.6 s/P; iv/Horner ~25 orders loose); frontier PAUSED; at snapshot 4/4 first cells ledgered
  `type-gate straddle (cM) (depth 8)` (hardest-first; stop rule not evaluable). OPS: turn-end reaping is the
  recurring killer; 4 GB box, no concurrent diagnostics.

2026-09-15 W3 (W3_numerics/) STATE: third infrastructure death. pgrep self-match fixed (bracket trick); then
  drivers + watcher died together after the 08:22 CST resuscitation; last write 00:30 archive-clock; A box
  44/1920 (acc 2.545275e-10), B box 46/2688 (acc 1.017122e-09); measured 487–537 s/box vs planned 71–100;
  pre-registered outcome INCONCLUSIVE.

2026-09-15 OPERATOR (CURRENT_STATE_DELTA in Drive): Kimi usage exhausted until 2026-09-30; next non-Kimi work =
  independent reconstruction, exact-law auditing, discharge of the D1 v2.2 validity premises.

2026-09-15 CL (Anthropic, first pass) DELIVERED: CL-AUD-001 (third-family re-execution + tree custody 985/46/0),
  CL-ERR-001 (E1–E7), CL-REG-001 (extraction-rule register), CL-PIN-001, CL-OBL-001 (ZBAND dischargeable;
  RN-UNIF work order), CL-STATE-001 (RETURN_07 candidate), this addendum.

2026-09-15 CL (Anthropic, second pass) DELIVERED: H5_ZBAND_CONSUMPTION (OBL-H5-ZBAND discharged at consumption
  grade; certificate f5965c43…, digest cbe8603f…, 3/3 mutations fail closed); D1_ASSEMBLY_v2_3_DRAFT (body
  6f0c4a89…; two validity premises; exact bracket 21.927883016; P(B4) consumed at round-UP 1.23e-9 with the
  B4LOC nearest-rounding nit E-B4LOC-1 recorded; retracted LPW token forbidden) with gate d1_falsify_v4.py
  (strict superset of v3; 156 PASS; digest ea8076b4…; 5/5 mutations fail closed; three v3 cks inverted by
  register-note adjudication; bracket tolerance tightened to exact equality). Seventh extraction convention
  (before_hashline, rung certificates) added to CL-REG-001. Drive write approval absent (six refusals);
  bundle re-issued.

2026-09-16 CL (Anthropic, third pass) DELIVERED / REFUTED (CL-RNU-001): RN-UNIF engine located and run to its
  end; blocker root-caused — `chi2_grad_bound` 1e19 too loose (1.57e14 vs true 1.56e-5; absolute-value
  summation through ‖Σ_pair⁻¹‖ ≈ 3.8e9) and crude Wick-moment scales; fix validated — χ² whitened by the y-free
  Σ6 = SPAIR0 reproduces the engine's χ² to 1.6e-84 and the exact whitened gradient matches FD to 1e-39.
  REFUTED: the engine's `mean_grad_exact` is not exact (E-RNU-1: two chain-rule terms missing; up to 185×
  off); corrected `mean_grad_fixed` FD-validated to 3e-42. CONFIRMED: `kappa_far_ds` exact value/∇/∇² to 1e-26.
  Landscape: sup κ_far = 0.67728491 at the zone boundary (5,0), −2.2/unit d, θ→−θ exact, θ→180°−θ not;
  cost 0.51 s/point at dps 100; κ-cap lever costed (0.68/0.69/0.70 → 650.1826/650.2079/650.2332).
  Register: D3-LEMMA-RN-UNIF "no lane executing" → engine exists, blocker root-caused, fix validated, closure
  costed (< 2 h compute once orders 2–4 bounds and a third-order DS land). Drive write live 16:40Z; landing of
  the Anthropic folder set begun (00_LANDING_NOTE, CL-STATE/ERR/AUD/RNU, v2.3 draft + gate v4 + receipts,
  ZBAND carrier + certificate + freeze, corrected mean-gradient and whitened-χ² scripts).
