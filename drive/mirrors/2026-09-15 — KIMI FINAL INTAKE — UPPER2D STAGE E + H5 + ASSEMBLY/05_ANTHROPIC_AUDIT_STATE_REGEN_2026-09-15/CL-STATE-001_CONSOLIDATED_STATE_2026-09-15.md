# CL-STATE-001-v1.0 — CONSOLIDATED CURRENT STATE (RETURN_07 CANDIDATE), 2026-09-15

AUTHOR: Claude (Anthropic), foreign environment · CREATED: 2026-09-15, UPDATED 2026-09-16 · CLASS: STATE — regenerated state document
STATUS: PROPOSED — non-authoritative until the operator promotes it; supersedes nothing by itself
AUTHORITY: none · CANONICAL IMPACT: NONE until promoted; if promoted, replaces the RETURN_06 executive state +
  ADDENDUM-1 + ADDENDUM-2 as the single readable state, and corrects `CURRENT_STATE_DELTA_2026-09-15.md`
SOURCES: every hash below recomputed FROM BYTES of `09152026OKComputer_Project_Gap_Closure.zip`
  (a2136bc0…) in this environment; reproduction receipts in `CL-AUD-001`
FALSIFICATION: any statement here that contradicts a frozen carrier at the hash cited beside it.

**Why this document exists.** RETURN_06 was assembled at archive-clock 00:08 (08:15 CST). The tree's newest
write is 13:11. Three append-deltas (H3 band floor; B4LOC + PERC; the D1 register note) each say "this delta
governs" over documents that were the single source of truth, and none of them fold in LPW v3/v4, the H3
ceiling, or H5 rungs 2–3. `CURRENT_STATE_DELTA_2026-09-15.md` in Drive still lists five validity premises with
B4.loc "asserted-but-not-established" and PERC-DECAY as "o(r³) pricing" — both superseded by a carrier sitting
in the same Drive folder. This is the regenerated, not appended, state. **Kimi is dark until 2026-09-30; the
snapshot is final until then.**

---

## 1. Theorem table (current)

| # | Theorem | Scope | Grade | Chain |
|---|---|---|---|---|
| 1 | **LPW qualitative:** 1 − q(r,6/5) ≥ c·r³ for some c, r₀ > 0 | exact 2D side-24 six-pin typed pair-Palm law | ENDORSED at review grade (Kimi LPW verdict); P0.1 remains HOLD | — |
| 2 | **LPW_CONSTANT v4 (explicit):** 1 − q(r,6/5) ≥ c·r³, **c = 394827584594841472652359498125 / 1771235249968883322980290175117427736576 = 2.2291086664054236617851686509e-10 ≥ 2.22e-10 ≥ 10⁻¹⁰**, **r₀ = 1/2278031360 = 1/(2¹⁸·8690)** | same law; (0, r₀] ⊆ (0, 0.0025] | EXACT (exact Fractions; published ≤ exact; 17 mutation families fail-closed; both modes byte-identical); one changed interface vs v3 reviewed PASS WITHIN SCOPE | v1 (defective, preserved) → v2 (1.1357…e-43, r₀ = 1/2224640) → v3 (9.8040863135804912e-23, levers 1+2, ×8.632e20) → **v4 (lever 4 via H3_BAND_CEIL, ×2.2737e12)**; cumulative ×1.9626e33; **r₀ shrank ×1024** |
| 3 | LPW_CONSTANT v1–v3 | historical | PRESERVED, consumed nowhere as authority (v2's 29-digit display retracted by H1 F-C29; v3's receipts errata on record) | — |
| 4 | **2D UPPER certified rung (D1 v2.2(1))**: 1 − q(0.05,6/5) ≤ Ĩ_hi + C_RN·√Q(B1.dir) + P(B2) + P(B4); **E_w(0.05) ≤ 8.1272827e-2 = 650.1827·r³** (exact 650.1826140…), C_RN ≤ 3.46 | rung r = 0.05, Reg ∩ TYP | Named-hypothesis grade: (a) H5-RIM/H5-AXIS v1–v3 (named lemmas), (b) D3-LEMMA-RN-UNIF(0.05) NOT closed. Body 490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6 (rule marker_strip_LF) | v1.0 → … → **v2.2 FINAL**; **v2.3 pending** (register note c1d5e95d…) |
| 4′ | **Sharpened form, now DRAFTED as D1 v2.3** (register note §4; draft body 6f0c4a89…, gate `d1_falsify_v4.py` = strict superset of v3, 156 PASS, digest ea8076b4…, 5/5 mutations fail closed; PROPOSED): 1 − q(0.05,6/5) ≤ 8.1272827e-2 + P(NearSwap) + P(B2) + P(B4), NearSwap certified super-algebraic (BRANCH_dir), **P(B4) ≤ 1.23e-9** (B4LOC's 1.22e-9 is a 3-sig-fig NEAREST-rounded display — E-B4LOC-1 — consumed at round-UP). The `C_RN·√Q(B1.dir)` term prices nothing beyond NearSwap + the FarRoute free-rider (A, B1_wit, FarRoute pairwise disjoint inside {N_w ≥ 1}, one E_w payment) | rung | recorded strengthening, effective at v2.3 | — |
| 5 | **2D UPPER all-small-r (D1 v2.2(2))**: ∃ r₀ > 0 (existential), ∃ C < ∞: 1 − q(r,6/5) ≤ C·r³ ∀ 0 < r ≤ r₀ | all small r | **CONDITIONAL on TWO validity premises** (was five): OBL-D1-PROMOTE (chart side + uniform-band extension of B4LOC/B2-far/FarRoute); D3-LEMMA-RN-UNIF | see §2 |
| 6 | 3D SIDE24 (AO48-OPR-045) | 3D — separate family | untouched; firewall holds (no citation anywhere in tree) | — |
| 7 | P0.1 / P0.2 | operator Booleans | P0.1 HOLD; P0.2 no occurrence | — |
| 8 | W8 Λ-side certified floor | 837 Phase-1 cells | OPEN; Phase-2 frontier PAUSED; 4/4 first (hardest-first) cells ledgered at depth 8; formal nonclosure receipted | — |
| 9 | W3 WP rigorous lower enclosure | 4608 boxes at r = 0.025 | RUNNING → **DEAD at snapshot** (§4); pre-registered INCONCLUSIVE | — |
| 10 | Coefficient-limit campaign | — | **NOT OPEN**; "limiting coefficient" language prohibited | — |

**Two-sided honesty line (new, not in any carrier):** the upper is certified within ~507× of the evidence value
(1.42·r³) at r = 0.05; the lower is certified within ~6.4×10⁹ of it, on r ≤ 4.4×10⁻¹⁰. These are certificates in
different regimes, not two sides of one law. The manuscript's withdrawal of the two-sided framing (ERRATA
2026-09-13 §1) remains correct.

## 2. Obligation register (current; owner → state)

| obligation | class | state 2026-09-15 | carrier |
|---|---|---|---|
| **OBL-D1-PROMOTE** | VALIDITY (Thm 2) | OPEN. Chart side: **3 certified rungs** 647.8048 / 661.4712 / 664.3979 at r = 0.05 / 0.035355 / 0.025 (+2.11%, +0.44%, decelerating; C_unif 731.4311 margin 10.1%; falsifier not tripped). Rungs 0.0177 / 0.0125 mid-flight, stitch-bound, **frozen until Kimi returns**. Normalizer sub-part DISCHARGED by H3 band floor. Now also carries the uniform-band extension of B4LOC / B2-far / FarRoute | H5_RUNG2 body 91a34d83…, H5_RUNG3 body f4c3414f…; totals f7697bcf… / 808d6901… |
| ↳ OBL-H5-JETMOD | sub | OPEN (display only: κ = 1/8 modulus band, 12 certified points) | H5_PROMOTE_UPDATE |
| ↳ OBL-H5-ZBAND | sub | **DISCHARGED (consumption grade, PROPOSED)** — `h5_zband_consume.py` pins both H3 band carriers from bytes, parses every number from the frozen transcripts, builds the band table, exit 0 both modes byte-identical, digest cbe8603f…, 3/3 mutations fail closed. Carrier H5_ZBAND_CONSUMPTION_2026-09-15.md (9445bdd3…). lo: Z_r/r² ≥ 2.30659559567154, hi: ≤ 3.74767948915996, uniform on (0,0.05] | H3_BAND_FLOOR body 281477c3…; H3_BAND_CEIL body cfe8a3a4… |
| ↳ OBL-H5-REMOTE-THRESHOLD | sub | OPEN; rides with D3-LEMMA-RN-UNIF | — |
| **D3-LEMMA-RN-UNIF** | VALIDITY (Thm 2; rung part is a named hypothesis of Thm 1) | **OPEN — engine EXISTS** (`d3_rn_unif.py`, 2,240 lines, Kimi 09-15 07:43, unfrozen, unmentioned in any state doc; stops before its own certification run). 2026-09-16: blocker root-caused (χ² gradient bound 1e19 too loose through the rigid Σ_pair⁻¹; crude Wick scales), fix validated (whitened-frame χ² identity to 1e-84; exact whitened gradient FD-matched to 1e-39), engine defect E-RNU-1 found and fixed (`mean_grad_exact` missing two terms), exact D-engine confirmed to 1e-26, landscape mapped (sup 0.67728491 at the zone boundary (5,0); −2.2/unit d), cost 0.51 s/point; closure plan and κ-cap lever in CL-RNU-001 | D3_PERCOLATION body 8e7fef6b… §5(ii); CL-RNU-001 |
| PERC-DECAY | was VALIDITY | **REFUTED as phrased (o(r³)) → RESTATED (Θ(r³) with certified constants) → validity content absorbed into the E_w accounting.** Certified: B1.dir-far D3-part ≤ 23.2119·r³; B2-far ≤ 17.6802·r³; B4.rem superseded by B4LOC | PERC_DECAY body 5137a811… (marker_raw) |
| ↳ PD-CONN | NAMED INPUT, upgrade-only | OPEN; constants-not-order. Missing: (i) explicit-constant planar Bargmann–Fock RSW arm bound — the one genuinely external theorem; (ii) field-level conditioned→unconditioned Kolmogorov-entropy patch; (iii) torus→planar (available, ≤ 1e-120) | PERC_DECAY §4 |
| B4.loc dam-line certificate | was VALIDITY | **CLOSED** — Theorem B4LOC-R1, P_r(B4) ≤ exp(−c_eff/r²), c_eff = 0.0513/0.0645/0.0709; whole-B4 (loc + rem); cut-net ≡ 9-pin-tube identification PROVEN FALSE on four axes; B4.rem reconciliation ADJUDICATED YES by D1 (register note §2). Driver re-executed here byte-identical | B4LOC_DAMLINE body 0d5c1b32… |
| OBL-B1-BRANCH(loop\|B1) | REFINEMENT (demoted) | OPEN, constants only | BRANCH_DIR |
| OBL-D2-AO-SHARP (i, iii, v) | REFINEMENT | OPEN | D2 |
| B2-far free-rider pinning (N_β* ≤ N_w vs C2's exact definition) | refinement check | OPEN (register note §4.4) | — |
| H5-RIM / H5-AXIS production paths | NAMED LEMMA | OPEN production paths; named-hypothesis grade suffices for Thm 1 | H5 tightening 7fefa17b… |
| G.7 normalizer at the rung | — | DISCHARGED (R2); continuum band → H3 band floor | — |

## 3. Adversarial / failed-approach ledger — additions since RETURN_06

- **"o(r³) far-lane" reading (PERC-DECAY as phrased v2.0–v2.2)** — REFUTED at evidence grade by the certified
  corridor-alive display (conditioned axial mean never dips below s out to x = 1.0; bottleneck z = 0.68 = Θ(1));
  window ℓ = r³/6 fixes Θ(r³); raw window count Θ(r³) at every D < 12. A successful attack forcing amendment.
- **Cut-net ≡ 9-pin-tube identification** — proven FALSE (law / tube / level / scaling). Replaced by direct
  construction.
- **Whitened/box pipeline for interval-r** — structurally infeasible (LDL pivots ~4.2e-4 / 2.1e-7 destroyed by
  any interval Gram; Gram-entry width 3.3e7 on [0.049, 0.05]). Replaced by no-whitening Wick + exact Laurent
  series. (H3_BAND_FLOOR §2)
- **Naive p̄-only upper envelope** for the normalizer — not viable (9.55 vs 3.23). (H3_BAND_CEIL §3)
- **δ = 1/256** in LPW lever 2 — fails path clearance (3/32 > 343/3840); reverted to δ = 14587/2621440.
- **v3 F-LEVER4** — a band FLOOR is the wrong direction for the Palm denominator; refused, gap named, closed by
  the ceiling eleven minutes after it landed.
- **Rung-2 stitch drift alarm** (effective power ≈ 2.73) — resolved at rung 3: first-step offset is rung 1's
  refine-3 advantage; between equally-treated rungs stitch scales r³ to 0.54%.
- **H5 v2 I_lo contamination** — rung-unscoped merger glob summed rung-0.025 patches into r = 0.05 totals;
  root-caused, v3 clean, merger rung-scoped.
- **W8 DMAX-class silent truncation, instances 3 and 4** — `tp_qk.pkl` built at DMAX=17 (Q_6 wrong at 23
  entries); `s4r.py:315` resets DMAX=100 inside GkSups init (x9_series X6 corrupted). Potentially unsound sups;
  **no cell certified under them → no bad certificates.** Fourth instance of one bug class = architectural
  hazard (global mutable degree cap), not four coincidences.
- **W3 supervision, third failure** — `pgrep -f` self-match fixed by the bracket trick; then turn-end reaping
  killed drivers and watcher together; 22 h and ~13 h undetected outages.
- **RN-UNIF `chi2_grad_bound`** — absolute-value summation of four mutually cancelling terms through
  ‖Σ_pair⁻¹‖ ≈ 3.8e9; bound 1.57e14 against a true 1.56e-5 (CL-RNU-001 §2–3). Whitening by the y-free Σ6
  removes it.
- **RN-UNIF `mean_grad_exact`** — not exact: two chain-rule terms missing (E-RNU-1); corrected and FD-validated
  to 3e-42.

## 4. Lane states at snapshot (13:11 archive-clock; never evidence)

| lane | state | detail |
|---|---|---|
| H5 rungs 4–5 | ALIVE at snapshot, now FROZEN (Kimi dark) | r = 0.0177: 125 cells + 6 stitch; r = 0.0125: 42 cells + 2 stitch; 22 stitch sectors needed per rung |
| W3 LOWER | **DEAD** | A box 44/1920 (acc 2.545275e-10), B box 46/2688 (acc 1.017122e-09); last write 00:30; watcher dead at same instant; resuscitation at 00:22 lasted ~8 min. Measured 487–537 s/box vs planned 71–100 → ~300 h remaining. Pre-registered INCONCLUSIVE |
| W8 Phase-2 | PAUSED | 4/4 first cells ledgered `type-gate straddle (cM) (depth 8)` — **hardest-first order**, so the >15% stop rule is not evaluable yet; X_0 sup route resolved by measurement (exact-Fraction shift, 2.6 s/P, ~53 min/box); gate5 2/2 ledgered at depth 2 |
| H3 band | COMPLETE, consumed | floor + ceiling frozen, in H3 MANIFEST, consumed by LPW v4 |
| BRANCH | COMPLETE | rung025 landed; receipts errata 7547e76a… |
| B4LOC (D2) | COMPLETE, FROZEN | FREEZE.txt 5235f98a… |
| PERC (D3) | COMPLETE, FROZEN | FREEZE_PERC_DECAY 6bbe6e27… |
| LPW | v4 FROZEN | MANIFEST in v4/ |
| D3 RN-UNIF engine | EXISTS, UNRUN beyond probes, UNFROZEN | `d3_rn_unif.py` 09-15 07:43; see CL-RNU-001 |

## 5. CANNOT-VERIFY / record-integrity (open; none blocking)

Carried from RETURN_06 items 1–3, 5, 6, 9. Item 4 (647.8047/647.8048) **RESOLVED**: I_hi/r³ = 647.804714016;
647.8047 is the round-DOWN, only 647.8048 is consumable. Items 7–8 resolved as stated. New (this pass, see
CL-ERR-001): the 21.92788306 "exact bracket" slip (+4.4e-8, conservative); the retracted 29-digit LPW display
still in the executive state and DECISION_LEDGER; seven extraction rules live in one tree (CL-REG-001); H5
rung-2/3 driver code unpinned (CL-PIN-001); DECISION_LEDGER ends 09-13 (CL-LEDGER-001); B4LOC per-rung
displays nearest-rounded (E-B4LOC-1); RN-UNIF engine unmentioned and unfrozen (CL-RNU-001).

## 6. Custody statement

Tree custody vs RETURN_06's TREE_MANIFEST (1031 files): 985 MATCH, 46 DRIFTED, 0 MISSING. All 46 drifts
reconcile to declared volatile lanes or to later freeze records **except** the H5 drivers (§5). SOURCE_CAPSULE:
62/70 rows verify; the 8 misses are superseded rows whose replacements the addenda record. Gates re-executed
here: D1 v2.2 (92 PASS, digest d800849e… = RETURN_06), H4JC (72fdc49b…), B4LOC driver (byte-identical,
43ff9c1e…), PERC engine (verdict reproduced). Full receipts: CL-AUD-001.

## 7. What is blocking, in order

1. **D1 v2.3 promotion** — a gate-passing v2.3 DRAFT now exists (two premises; exact bracket; ZBAND consumed; retired-premise and retracted-token cks inverted/added). Operator promotion makes it the current strongest form.
2. **D3-LEMMA-RN-UNIF** — engine located and root-caused (CL-RNU-001); remaining: whitened χ² bounds at orders 2–4, third-order `DS3`, valid 4th-order envelope, then a < 2 h certification run at the rung. Anthropic proceeding.
3. **OBL-D1-PROMOTE rungs 4–5** — Kimi-gated until 09-30; nothing to do but not lose the banks.
4. **State regeneration rule** — this document or its successor should replace, not join, the addenda.
5. **PD-CONN (i)** — external mathematics; upgrade-only; not closable by compute.
