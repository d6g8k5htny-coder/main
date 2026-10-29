# ADDENDUM-2 2026-09-15 — B4LOC dam line CLOSED (super-algebraic, whole B4) + PERC-DECAY RESTATED (Θ(r³) absorption, certified constants)

Second append-delta to RETURN_06 (assembled 2026-09-15T00:05:36Z; first delta ADDENDUM_2026-09-15_H3_BAND_FLOOR.md).
No frozen file is modified; every hash below was recomputed FROM BYTES by the addendum-2 agent on 2026-09-15
before being cited. Root: /mnt/agents/output/K3_SIDE24_LB/. No file outside RETURN_06/ is touched by this delta;
SOURCE_CAPSULE.sha256 is extended APPEND-ONLY (existing bytes byte-preserved, prefix re-verified).

## 1. Carrier 1 — UPPER2D/B4LOC_damline/ (D2 lane; record B4LOC-R1-20260914-v1.0, FREEZE.txt)

| artifact | rule | recomputed sha256 | bytes | verdict |
|---|---|---|---|---|
| UPPER2D/B4LOC_damline/B4LOC_DAMLINE.md | marker-lines (LF-joined lines strictly between the unique exact lines BEGIN_FROZEN_BODY / END_FROZEN_BODY; exactly one terminal LF; last content line directly preceding END marker) | 0d5c1b3284374f3a2f7630b7f603a71c77f5ec9417fe4579aaac03becd33c05e | 15164 (body 13503) | MATCH vs FREEZE.txt |
| UPPER2D/B4LOC_damline/B4LOC_DAMLINE.md | whole file (custody) | ae4c9c82784aecfbd85056c316f792a331bccb774d952887fc5fe8ea7da615b4 | 15164 | MATCH vs FREEZE.txt |
| UPPER2D/B4LOC_damline/b4loc_driver.py | whole file | bf3b022580a7eea1b4797658fe77a74a92850d59478212f39bca2bb231f20f02 | 26271 | MATCH |
| UPPER2D/B4LOC_damline/b4loc_falsifier.py | whole file | d59ad79088dbf152b6e196bbd48ee0a79ef304d60611148eabaa575e7fd8b10f | 3360 | MATCH |
| UPPER2D/B4LOC_damline/t_normal.txt | whole file | 11baaccc25da202be951731ea4bc91c76795b614a9925a3baf7c59f8b0112b53 | 2735 | MATCH; ≡ t_opt.txt BYTE-IDENTICAL (cmp verified) |
| UPPER2D/B4LOC_damline/t_opt.txt | whole file | 11baaccc25da202be951731ea4bc91c76795b614a9925a3baf7c59f8b0112b53 | 2735 | MATCH (both interpreter modes) |
| UPPER2D/B4LOC_damline/tf_normal.txt | whole file | 1629de83bda724f5a0d971ddd895771ea3db1a2ec7ca462181030c0a952f1a2c | 885 | MATCH; ≡ tf_opt.txt BYTE-IDENTICAL (cmp verified) |
| UPPER2D/B4LOC_damline/tf_opt.txt | whole file | 1629de83bda724f5a0d971ddd895771ea3db1a2ec7ca462181030c0a952f1a2c | 885 | MATCH (falsifier transcripts, both modes) |
| UPPER2D/B4LOC_damline/FREEZE.txt | whole file (custody) | 5235f98a59e5119444b8fda2ea16d2dd3333e0d46f0b52af6362a2b2a0ecd163 | 1213 | records all of the above + B4LOC-DRIVER DIGEST 43ff9c1ebfa9cf8f27c06e27e844030e8378e439e89bdd374949166a1a490a1b, B4LOC-FALSIFIER DIGEST f939eeab4805b9c5b35fb40f0f173b4a4ad8237fdcd15ad53084569c6fbffe55 |

Extraction-rule note (verified from bytes, recorded for the capsule): the capsule's generic `sepbody` rule
(bytes before the first `"\n---\n\n"`) does NOT apply to this file — the document contains no `---` separator
at all (grep-verified); the FREEZE-recorded rule is the marker-lines rule stated in the table, under which the
recomputed body is exactly 13503 bytes and hashes to 0d5c1b32…, an exact MATCH. (Same pattern as the H3
band-floor note in the ADDENDUM 2026-09-15 capsule section.)

## 2. Carrier 1 certified content (verified against the frozen body 0d5c1b32…)

**THEOREM B4LOC-R1 — the B4.loc dam line CLOSED at super-algebraic grade, covering ALL of B4 (loc + rem).**
Verified components:

- **Exact event inclusion B4 ⊆ E1 ∪ E2** (Lemma B4LOC-R1-A, deterministic topological step): E1 = curvature
  zone (0, 1/4] (∃ζ with f_xx(q(ζ)) ≤ 0), E2 = value corridor [1/4, 1) (∃ξ with f(q(ξ)) ≤ s); the proof's
  contradiction covers loc and rem alike (open M-ward channel ⇒ field not in B4).
- **Uniform Borell–TIS sup-tails under the 6-pin law**, all factors explicit: Dudley D = 2√2·J with the
  closed form J = L·T·√π/4 (van Handel APC 550 Thm 5.12); conditioning-shrinks-variance (mpmath dps=100,
  residual ≤ 6.2e-96 per rung); variational Hermite predictor H(ξ) = s + ℓ(3ξ² − 2ξ³); the Λ_p pin-likelihood
  trick with Λ_p² certified = 2.82666 (r = 0.05; 2.82667 finer rungs); covariance factorization (1-D K1 values,
  order ≤ 8); certified 8-level pointwise-exact Leibniz grid-to-continuum transfer (9th-derivative remainder
  ≤ 1e-23); C_RN ≤ 3.47 consumed from H4-RN (H4_CLOSURE body a67d50b9…; driver fail-closes on drift, mutation M3).
- **Certified bounds:** P_r(B4) ≤ 1.22e-9 / 1.52e-45 / 1.03e-197 at r = 0.05 / 0.025 / 0.0125
  (r³ = 1.25e-4 / 1.56e-5 / 1.95e-6). Zone margins at r = 0.05: kap1 = 9.33 (curvature, Q(E1) ≤ e^{−43.536}),
  kap2 = 14.69 (value, Q(E2) ≤ e^{−107.93}) vs the o(r³) threshold √(12 ln(1/r) + 4 ln C_RN + 2 ln 2) = 6.50.
- **Grade:** not merely o(r³) — P_r(B4) ≤ exp(−c_eff/r²), c_eff = 0.0513 / 0.0645 / 0.0709 on the ladder
  (increasing); both zone margins Θ(1/r) (the C2 §4 law κ_* = Θ(1/r)); strongest grade the RN square-root
  transfer can express.
- **Identification resolved NEGATIVELY (disposition (b)):** the pair-level cut net is PROVEN DISTINCT from the
  9-pin M-ward tube on four independent axes — (i) law (6-pin Q_r vs 9-pin Palm law), (ii) tube (full cut net
  vs one of four y-jet channels), (iii) level (s = f(S) vs v = f(y); H4-JC-R1), (iv) scaling (κ_* = Θ(1/r)
  vs κ_dam = Θ(1) — a Θ(1) margin cannot dominate a Θ(1/r) family). No asserted identification survives; the
  9-pin tube is recorded as PROOF TEMPLATE only.
- **H4-SD distinction verified, not assumed:** H4-SD's C⁰-impossibility applies to the B1.dir event (away
  branch on M's lineage, sd_res = Θ(d²)); the B4 dam is the opposite geometry — a C⁰ value/curvature event on
  a FIXED corridor at pin level s, cheapest cut ON the rigid ridge. Independently verified in the driver
  (corridor profile strictly increasing; on-ridge margins).
- **Discrete mirror (B1's frozen PL model, b1_falsifier.py 818df798…):** 3 B4 pairs audited (0 ensemble + 3
  synthetic), EVERY B4 corridor CUT (0 violations); 387 clear-corridor pairs (categories A 154 / B1 1 / B2 66 /
  success 163), NONE of them B4; mirror fail-closes on violation.
- **Mutation suite fails CLOSED:** M1 weaken-margin (×1/2) fires the o(r³) gate (P_bound = 0.104 > r³ = 1.25e-4,
  ratio 835); M2 widen-tube (σ̄, D × 2) fires the o(r³) gate; M3 RN (C_RN × 2) fires the RN-drift gate
  ("consumed RN factor drifted above H4-RN's certified C_RN = 3.47"); F3 verifies mutation tags (no silent
  normal runs). Both interpreter modes byte-identical (t_* ≡, tf_* ≡).

## 3. Carrier 2 — UPPER2D/D3_percolation/PERC_DECAY.md (D3 lane; record D3-PERC-DECAY-20260915-v1.0, FREEZE_PERC_DECAY.txt)

| artifact | rule | recomputed sha256 | bytes | verdict |
|---|---|---|---|---|
| UPPER2D/D3_percolation/PERC_DECAY.md | marker-lines (LF-joined lines strictly between the unique exact lines BEGIN_FROZEN_BODY / END_FROZEN_BODY; normalize line endings to LF; exactly one terminal LF) | 5137a811e6be72ed27e0458c75537bdcfabb8d8e7148b211d8c7b9a1584df400 | 8041 (body 7409) | MATCH vs FREEZE_PERC_DECAY.txt |
| UPPER2D/D3_percolation/PERC_DECAY.md | whole file (custody) | f971b81a54dd2b78924c77e412e0c19b798e5356795057fcdadec3235e960745 | 8041 | MATCH |
| UPPER2D/D3_percolation/d3_perc_decay.py | whole file | ee68ac7510241947c64445cc6aa17f91ccc4f2e5f0dacc2b0de801ef72d8ab64 | 19063 | MATCH |
| UPPER2D/D3_percolation/pd_normal.txt | whole file | 7547476c1ded7f69c63d65a5f3be97ce0568cbfd7746b5dacdd24dc462f17ff7 | 9762 | MATCH; ≡ pd_opt.txt BYTE-IDENTICAL |
| UPPER2D/D3_percolation/pd_opt.txt | whole file | 7547476c1ded7f69c63d65a5f3be97ce0568cbfd7746b5dacdd24dc462f17ff7 | 9762 | MATCH (both interpreter modes; engine digest 58df7e312099d3acd93fe83f728b4b139274063c98386b9963bbc1a6270e2b9b) |
| UPPER2D/D3_percolation/d3_falsifier.py | whole file | abbb40d279606660a0c16af253d9bc15038f7221702fb7aaa6f74b4b893df032 | 32264 | MATCH vs freeze (F1–F7 line) |
| UPPER2D/D3_percolation/tf_normal.txt | whole file | b5e97e6004bbc4d8a2305e94bc5b38b8b43e3fc5cb8f17629563d4beea988049 | 1833 | MATCH; ≡ tf_opt.txt BYTE-IDENTICAL (cmp verified) |
| UPPER2D/D3_percolation/tf_opt.txt | whole file | b5e97e6004bbc4d8a2305e94bc5b38b8b43e3fc5cb8f17629563d4beea988049 | 1833 | MATCH (D3-FALSIFIER PASS, exit 0; falsifier digest fcb37708468875723bbfac838ab7b5ab374edb4ae60566f1ef2b7b5e44958e9f) |
| UPPER2D/D3_percolation/FREEZE_PERC_DECAY.txt | whole file (custody) | 6bbe6e27a91c777b7245294fc5740462af06e7f20b0f92ff4eae15cd2f05edef | 3966 | complete freeze incl. the falsifier F1–F7 line |

F7 status note: at the start of this addendum's assembly FREEZE_PERC_DECAY.txt (3609 B, 10:19) lacked the final
falsifier line and would have been recorded pending-in-addendum per instructions; the D3 lane completed the
two-mode falsifier run and the freeze line LANDED at 11:26 during assembly (file now 3966 B). The landed
hashes were verified FROM BYTES by this agent and MATCH the freeze record (d3_falsifier.py abbb40d2…,
tf_normal ≡ tf_opt b5e97e60…, falsifier digest fcb37708… — the tf transcripts end with "PERC-DECAY re-run:
exit 0, byte-identical; … D3-FALSIFIER PASS digest=fcb37708…"). Nothing is pending.

## 4. Carrier 2 certified content (verified against the frozen body 5137a811…)

**The honest verdict (body §0):** the far lanes are **Θ(r³), NOT o(r³)**. Each far lane is window-priced: its
ORDER is Θ(r³) (the window ℓ = r³/6 fixes the order), and every distance-decaying factor (RN reversion,
crude-spine cap density, kernel-scale connectivity) controls the CONSTANT, not the order. The decisive
corridor-alive certified axial-mean display (body §3): the conditioned axial mean stays ABOVE s from S out to
the ghost-elder territory — μ−s = +2.4e-8 @0.024 (≈S, pinned) rising through +2.3e-3 @0.2 to +9.4e-2 @1.0; the
z-score decays to the bottleneck z = 0.68 = Θ(1) @1.0 (transcript value 0.68056). The connectivity factor at
the sub-kernel handoff (d_max = 0.20–0.25, BRANCH_dir's 4r–5r) does NOT vanish as r → 0, so the o(r³) reading
is false in the saddle-node limit. Consistent with D1 v2.2's Theorem (2), which needs the far lanes inside
C·r³, not literally o(r³).

**Restatement underwritten by the lane:** PERC-DECAY := *the far lanes absorb at Θ(r³) with the certified
constants, plus D-tails certified under the named input PD-CONN.*

**Certified today (theorem-usable):**
- **B1.dir-far = FarRoute ⊆ {N_w ≥ 1}** — the away lobe's merge into C_M is necessarily a WINDOW SADDLE
  (C_M born at b at M, the away lobe at s at S). P(FarRoute) ≤ E_{P_r}[N_w]; the D3-certified part outside
  the C1 chart ≤ B_remote + I_hole ≤ 21.9279 + 1.284 = **23.2119·r³** — inside the v2 remote bracket,
  floor-consistent.
- **B2-far ⊆ E[N_w(d ≥ 0.2)] ≤ 17.6802·r³** (crude-spine cap grade, R2 floor Z_lo = 7.7592917375327855e-3;
  recomputed numerator over radii ≥ d_max; transcript 17.680197·r³).
- **B4.rem:** NOT ⊆ the window count (the non-local separator can close through above-window saddles or the
  torus wrap). Certified content: the LOCAL dam per-section (κ_dam = 71.84 / 129.9 / 391.2 at
  ξ = 0.25 / 0.5 / 0.75 — κ = Θ(1/r), residual sd = Θ(r⁴), clearance the cubic Hermite profile, per-section
  price Φ(−κ_dam) super-algebraic, polarity-safe); the REMOTE part is NAMED (above-window merges at heights
  > b + the torus wrap circuit d ~ 12).

**Named input PD-CONN (deliverable-grade):** P_{Q_r}(A_r(B(pair, 2r), D)) ≤ C₀e^{−c₀D} for all D ≥ D₀ ≥ 1,
r ≤ 0.05, with explicit r-uniform constants C₀, c₀. Consumption: each far lane's D-tail ≤ (certified count
density)·C₀e^{−c₀D}·(1 + κ certified); class totals absorb at Θ(r³) with certified constants; B4.rem's wrap
≤ C₀e^{−12c₀}. **Missing pieces:** (i) an explicit-constant planar-Bargmann–Fock arm bound at positive level
(the literature gives sharpness WITHOUT explicit constants — an explicit RSW constant is the missing theorem);
(ii) the field-level conditioned→unconditioned patch on {d ≥ D₀} (finite-dimensional reversion certified in
the carrier; the field-level version needs continuity-modulus/Kolmogorov-entropy control — not built);
(iii) the torus→planar comparison (available, ≤ 1e-120). **PD-CONN certifies CONSTANTS, not order** — at
sub-kernel D the connectivity factor is Θ(1).

**Displays (body §2, §5):** reversion profile (Var → 1, κ_cross → 6.2e-8 along the ray; the field NOT yet
reverted at the handoff — the far lanes start inside the conditioned zone); annulus cap-density decay
(ρ_cap = 7.8093e-7 @(0.1, 0°) → 1.3715e-9 @5, frozen D4); window-count D-tail
(E_Q[N_w(d ≥ D)]/r³ = (576 − πD²)·J(ℓ)/r³ = 2.9268 / 2.9234 / 2.9115 / 2.8636 / 2.6720 / 1.9056 at
D = 0.2 / 0.5 / 1 / 2 / 4 / 8 — the raw count is Θ(r³) at every D < 12; decay lives entirely in the
connectivity factor). **Machinery:** fail-closed (ck → SystemExit), deterministic, Monte-Carlo-free (every
denominator the R2-certified floor), both modes byte-identical; mutation suite MUT-PD-1..5 (register-taxonomy
tamper, grade-label tamper, R2 floor-pin tamper, count-window tamper, corridor-display tamper) all fire —
verified live in the tf transcripts, including the F7 PERC-DECAY re-run gate (engine re-run exit 0
byte-identical; count bound 17.680197·r³ and bottleneck z = 0.68056 within their windows).

## 5. Register deltas (THEOREM_TABLE / 00_EXECUTIVE_STATE / ADVERSARIAL_REPORT not edited — this delta governs)

- **CLOSED — B4.loc dam-line tube certificate (validity premise #5):** the premise is CLOSED at certificate
  grade by B4LOC_DAMLINE.md (body 0d5c1b32…): uniform Borell–TIS sup-tail over the pair-level cut net under
  the 6-pin law, super-algebraic grade exp(−c_eff/r²), whole-B4 claim (loc + rem). The caveat carried inside
  the premise (cut-net ≡ 9-pin tube item (ii) "ASSERTED, NOT ESTABLISHED") is RESOLVED NEGATIVELY by the
  certificate itself (four-axis proof of distinction; independent direct construction — the item is now
  stated and owned independently of D2's tube lane, exactly the discharge the premise's falsifier/discharge
  clause allows). **Flag:** the whole-B4 claim's wrap/remote coverage vs D3's named B4.rem remote part is the
  reconciliation question of §7 below — under D1 adjudication; the CLOSED status stands for the B4.loc
  premise itself, with the reconciliation tracked as OPEN until adjudicated.
- **RESTATED — PERC-DECAY (validity premise #3):** the premise as phrased ("subcritical level-s⁺ cluster
  decay feeding every far lane's o(r³) pricing") is REFUTED at evidence grade on certified displays and is
  REPLACED by the restatement: **PERC-DECAY := the far lanes absorb at Θ(r³) with the certified constants
  (23.2119·r³ D3-part for B1.dir-far inside the v2 bracket; 17.6802·r³ for B2-far; B4.rem local dam
  super-algebraic per-section) plus D-tails under the named input PD-CONN.** The replacement still serves
  Theorem (2)'s need (far lanes inside C·r³). State: RESTATED, engine COMPLETE and FROZEN; PD-CONN itself
  remains a NAMED INPUT (OPEN) with its three missing pieces (explicit-constant planar-BF RSW arm bound;
  field-level Kolmogorov-entropy patch; torus→planar comparison — available, ≤ 1e-120).
- **Failed-approach register addition (ADVERSARIAL_REPORT.md §6 table — not edited; recorded here):**
  | Failed approach | Type | Exact reason it cannot be resurrected (carrier) |
  |---|---|---|
  | "o(r³) far-lane" reading (the registered PERC-DECAY phrasing: distance-decay factors deliver o(r³) pricing of B1.dir-far / B2-far / B4.rem) | mathematical | Refuted by the corridor-alive certified display (PERC_DECAY.md body 5137a811… §0/§3): the conditioned axial mean never dips below s out to x = 1.0, bottleneck z = 0.68 = Θ(1), so the connectivity factor at the sub-kernel handoff is Θ(1) and does not vanish as r → 0; every decay factor controls the constant, not the order (window ℓ = r³/6 fixes Θ(r³)); the window-count D-tail display (§5) independently shows the raw count is Θ(r³) at every D < 12. A successful attack FORCING AMENDMENT. Replacement: Θ(r³) absorption with certified constants + PD-CONN D-tails (constants, not order). |
- **Dir accounting note (flagged for D1, adjudicating — recorded, not resolved here):** the certified
  inclusion FarRoute ⊆ {N_w ≥ 1} prices B1.dir-far as a FREE-RIDER inside the E_w window-count envelope
  (the same E_{P_r}[N_w] that carries the window count), whereas D1 v2.2's sharper form (1′) carries the dir
  remainder through the C_RN·√Q dir term. The two accountings are different line items for the same event;
  which line carries FarRoute in the final assembly (free-rider inside E_w vs the RN dir term) is a
  bookkeeping/adjudication question for D1. No double-count is certified either way by this addendum.
- **Theorem table row 5 delta:** of the five validity premises of Theorem (2): B4.loc dam-line tube
  certificate → CLOSED (super-algebraic, whole B4; wrap/remote reconciliation flagged OPEN, D1); PERC-DECAY →
  RESTATED (Θ(r³) absorption, certified constants + named PD-CONN input OPEN). OBL-D1-PROMOTE (chart side),
  D3-LEMMA-RN-UNIF, OBL-B1-BRANCH(loop|B1): unchanged by this addendum.

## 6. Obligation-ledger delta (OBLIGATION_LEDGER.md not edited — this delta governs)

- **§5 B4.loc dam-line tube certificate — CLOSED 2026-09-15.** Discharge mechanism: the certified sup-tail
  (net + entropy + RN) over the cut net (Theorem B4LOC-R1, body 0d5c1b32…) PLUS the independent statement and
  ownership of the item (the cut-net ≡ tube identification proven FALSE on four axes; 9-pin tube retained as
  proof template only). Certified numbers: P_r(B4) ≤ 1.22e-9 / 1.52e-45 / 1.03e-197 on the ladder;
  c_eff = 0.0513 / 0.0645 / 0.0709; margins Θ(1/r); C_RN ≤ 3.47 pinned fail-closed. Discrete mirror: 3 B4
  corridors cut, 387 clear none-B4. Mutations M1/M2/M3 fail closed; both modes byte-identical. Carried
  forward as OPEN: the B4.rem wrap/remote reconciliation vs D3's named remote part (see §7; owners D1+D2+D3).
- **§3 PERC-DECAY — RESTATED 2026-09-15.** The falsifier/discharge clause's kill condition partially fired:
  a far-lane-relevant connectivity factor is certified Θ(1) at the handoff, killing the o(r³) SUFFICIENCY of
  the registered phrasing — but per the clause's own scope this is an amendment of the premise, not a kill of
  Theorem (2) (which needs C·r³). New content (D1 to ratify): far lanes absorb at Θ(r³) with certified
  constants (B1.dir-far D3-part ≤ 23.2119·r³ inside the v2 remote bracket; B2-far ≤ 17.6802·r³ cap grade;
  B4.rem local dam certified per-section, κ = Θ(1/r)) + D-tails under named input PD-CONN. Sub-obligations
  created/OPEN: PD-CONN piece (i) explicit-constant planar-BF arm bound (RSW-with-constants — THE missing
  theorem), (ii) field-level conditioned→unconditioned patch (Kolmogorov entropy), (iii) torus→planar
  (available, ≤ 1e-120, bookkeeping). Owner: percolation lane for the engine (COMPLETE, FROZEN); PD-CONN
  ownership to be assigned by D1.
- **§8 / H3 addendum cross-reference:** unchanged; the H3 band-floor discharge stands.

## 7. The B4.rem reconciliation question (OPEN — owners D1 + D2 + D3 jointly)

B4LOC-R1's inclusion lemma (body §2) is proved for ALL of B4 (loc + rem): its contradiction step argues that
with the corridor strictly above s the remote obstruction defining B4.rem is absent — so the certified
sup-tail prices the whole class. D3's PERC-DECAY (body §1, §4) instead states B4.rem is NOT ⊆ the window
count and certifies only the LOCAL dam per-section, NAMING the remote part (above-window merges at heights
> b; the torus wrap circuit d ~ 12; M-ward channel beyond the local cut net d > r) as carried by PD-CONN
(wrap ≤ C₀e^{−12c₀}). The question: does B4LOC's corridor argument genuinely cover the wrap/above-window
remote mechanisms (making D3's named remote part a redundant over-count), or does the pair-level corridor
inclusion miss genuine remote closures that require PD-CONN? Both certificates are individually frozen and
verified; the overlap must be adjudicated before the final assembly prices B4.rem exactly once. **Owner:
D1 + D2 + D3 jointly. State: OPEN.**

## 8. Running-tasks delta (RUNNING_TASKS.md not edited — this delta governs)

- **D2 lane (branch control): B4LOC dam line — COMPLETE, FROZEN 2026-09-15.** B4LOC_DAMLINE.md body
  0d5c1b32… + driver bf3b0225… + falsifier d59ad790… + both-mode byte-identical transcripts (t_* 11baaccc…,
  tf_* 1629de83…) + FREEZE.txt 5235f98a… (driver digest 43ff9c1e…, falsifier digest f939eeab…).
- **D3 lane (percolation): PERC-DECAY — COMPLETE, FROZEN 2026-09-15.** PERC_DECAY.md body 5137a811… + engine
  ee68ac75… + both-mode byte-identical transcripts (pd_* 7547476c…, engine digest 58df7e31…) + falsifier
  F1–F7 (d3_falsifier.py abbb40d2…, tf_* b5e97e60… byte-identical, falsifier digest fcb37708…) +
  FREEZE_PERC_DECAY.txt 6bbe6e27…. NOTE: the falsifier F7 two-mode run was in flight at the start of this
  addendum's assembly and LANDED at 11:26 during assembly (freeze record 3609 B → 3966 B); all landed hashes
  re-verified from bytes — nothing pending. Any post-11:26 revision of FREEZE_PERC_DECAY.txt would supersede
  the custody hash 6bbe6e27… recorded here (the frozen body/engine/transcript hashes it records were
  re-verified stable across the window).
- Lane-summary rows to add: "B4LOC dam (D2) | COMPLETE, FROZEN | B4LOC_DAMLINE.md 0d5c1b32… + FREEZE.txt
  5235f98a… | B4.rem reconciliation (D1+D2+D3)"; "PERC-DECAY (D3) | COMPLETE, FROZEN | PERC_DECAY.md
  5137a811… + FREEZE_PERC_DECAY.txt 6bbe6e27… | PD-CONN pieces (i)–(iii), D1 assignment".
- All other lanes (H5 promotion, W3, W8, BRANCH, H3) unchanged by this addendum.

## 9. Executive-state one-line deltas (00_EXECUTIVE_STATE.md not edited — this delta governs)

- Premise #5 (B4.loc dam-line tube certificate): OPEN → **CLOSED 2026-09-15 — Theorem B4LOC-R1 certified
  super-algebraic (P_r(B4) ≤ exp(−c_eff/r²), c_eff up to 0.071; ladder 1.22e-9 / 1.52e-45 / 1.03e-197),
  whole-B4 claim; cut-net ≡ 9-pin-tube identification proven FALSE on four axes (independent statement);
  wrap/remote reconciliation vs D3 OPEN under D1 adjudication.**
- Premise #3 (PERC-DECAY): OPEN → **RESTATED 2026-09-15 — the o(r³) far-lane reading REFUTED at evidence
  grade (corridor-alive certified display, bottleneck z = 0.68 = Θ(1)); replacement: Θ(r³) absorption with
  certified constants (≤ 23.2119·r³ B1.dir-far D3-part; ≤ 17.6802·r³ B2-far; B4.rem local dam κ = Θ(1/r))
  + named input PD-CONN (constants, not order; three missing pieces, the explicit-constant planar-BF RSW
  arm bound the missing theorem).**

## 10. Custody and verification statement

- Every hash printed in §1 and §3 was recomputed FROM BYTES by this agent before printing and again after
  writing (re-verification table appended to SOURCE_CAPSULE.sha256's ADDENDUM-2 section and re-checked);
  all verdicts MATCH. The two content sections (§2, §4) were verified against the frozen bodies quoted.
- Package manifests: 00_MANIFEST.sha256 (v1) and 00_MANIFEST_v2_2026-09-15.sha256 preserved unmodified;
  00_MANIFEST_v3_2026-09-15.sha256 (new file) covers the whole package including this addendum and the
  appended capsule. SOURCE_CAPSULE.sha256 extended APPEND-ONLY: pre-append state 23075 B sha256
  69d14191652f9b6afcbbe317f7594fa7d83ff3b8211ca12133be788aecd93308 (tail verified = the '# ADDENDUM
  2026-09-15' section); the v1 prefix (first 19117 B) re-verifies to fc4ffdf3a91b9f04d2882016d20a719f5b8ac1831e5582154d4ebff82a366ba2.
