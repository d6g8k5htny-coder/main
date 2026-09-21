# D1_ASSEMBLY v2.2 REGISTER NOTE — B4LOC closure + PERC restatement + adjudications

**Artifact:** D1-ASM-20260915-v2.2-REGNOTE1
**Agent:** D1 (assembly). **Scope:** register actions only — the frozen v2.2 body
(490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6) is NOT
altered; everything below takes effect at the next issuance (v2.3).

BEGIN_FROZEN_BODY

## 0. Carriers (verified from bytes)

| artifact | pin | verify |
|---|---|---|
| B4LOC_damline/B4LOC_DAMLINE.md | body 0d5c1b3284374f3a2f7630b7f603a71c77f5ec9417fe4579aaac03becd33c05e (13,503 B, = its FREEZE.txt) | MATCH (rstrip+LF marker segment) |
| B4LOC driver / falsifier | b4loc_driver.py bf3b022580a7eea1b4797658fe77a74a92850d59478212f39bca2bb231f20f02; b4loc_falsifier.py d59ad79088dbf152b6e196bbd48ee0a79ef304d60611148eabaa575e7fd8b10f | MATCH; driver + falsifier transcripts byte-identical both modes (verified by cmp) |
| D3_percolation/PERC_DECAY.md | body 5137a811e6be72ed27e0458c75537bdcfabb8d8e7148b211d8c7b9a1584df400 | MATCH (rstrip+LF marker segment) |
| PERC engine | d3_perc_decay.py ee68ac7510241947c64445cc6aa17f91ccc4f2e5f0dacc2b0de801ef72d8ab64 | MATCH; pd_normal ≡ pd_opt byte-identical (verified by cmp); MC-free (every denominator the R2 floor) |

## 1. CLOSED — the B4 summand (B4LOC-R1; whole-B4 per the adjudication in §2)

THEOREM B4LOC-R1 (consumed): the exact event inclusion B4 ⊆ E1 ∪ E2 (E1 =
near-S curvature zone, ξ ∈ (0, 1/4]; E2 = value corridor, ξ ∈ [1/4, 1) —
deterministic, verified against C2's B4 definition, covering loc+rem), with
uniform Borell–TIS sup-tails under the 6-pin law, all factors explicit:

    P_r(B4) ≤ 3.47·√(Q1 + Q2) = 1.22e-9 / 1.52e-45 / 1.03e-197
              at r = 0.05 / 0.025 / 0.0125,

margins 9.33/14.69 against the 6.50 o(r³) threshold; grade
P_r(B4) ≤ exp(−c_eff/r²), c_eff = 0.051/0.065/0.071. The old identification
caveat (v2.2's validity-premise item) is ELIMINATED at proof grade: the
pair-level cut net is proven distinct from the 9-pin tube on four independent
axes (law/tube/level/scaling) — no asserted identifications survive; the
9-pin tube is recorded as proof template only. H4-SD's distinction verified
(B1.dir off-ridge ≠ B4 on-ridge geometry). Grade note (rung vs band): the
certificate is ladder-certified at r = 0.05/0.025/0.0125 with the Θ(1/r)
margin law; the uniform-in-(0, r₀] extension rides the promotion band side
(§4), exactly as the rung floors did before the H3 band floor.

## 2. ADJUDICATION of the reconciliation question (lead's item 3): **YES**

Question: does B4 ⊆ E1 ∪ E2 hold for wrap/remote configurations — in
particular, does 'non-adjacency' B4 with the connection closing AROUND the
torus satisfy the inclusion's hypothesis?

Adjudication: YES, the inclusion covers the whole B4 summand, by the
following exact logical structure:

1. The inclusion's premise is the FROZEN GLOBAL definition of B4: M
   separated from S's upper sectors at level s — i.e., NO level-s path from
   S's sectors to M BY ANY ROUTE (B1 taxonomy's B4, the definition C2's
   barrier duality consumes, and the definition B4LOC §1 states and verifies;
   the executable mirror audits the taxonomy classifier directly: 3 B4
   pairs, every corridor cut; 387 clear-corridor pairs, none B4).
2. A configuration whose connection closes AROUND the torus (a wrap route at
   level s from S's sectors to M) is, by that definition, NOT B4 — the
   connection exists, the premise fails, the inclusion is vacuous for it.
   The feared gap ("wrap B4 with the corridor clear") is empty: a clear
   straight corridor IS a level-s connection from S's M-ward sector to M
   (the corridor leaves S along the M-ward axis, endpoint f(M) = b > s), so
   any such field fails the B4 premise.
3. A GENUINE B4 (no connection by any route, wrap included) forces the
   straight M-ward corridor to be blocked somewhere — B4LOC's proof is
   exactly this contrapositive — and the blockage is E1 (within ξ ≤ 1/4) or
   E2 (beyond), certified by the tube events T(ξ) with explicit margins.
4. D3's B4.rem register row (the M-ward channel beyond the local cut net,
   above-window merges at heights > b, the torus wrap circuit at d ~ 12)
   describes PRICING ROUTES for sub-events of B4 (its "not ⊆ the window
   count" observation concerns the N_w-pricing route, not the event
   inclusion). B4.rem ⊆ B4, so the inclusion supersedes that decomposition
   at the event level; no torus-wrap-specific bound is required.

Consequence recorded: **B4.rem is CLOSED by B4LOC-R1 for the O(r³) validity
grade (indeed super-algebraic at the ladder rungs); PD-CONN is upgrade-only
everywhere** (constants-not-order; its only remaining role is constant
sharpening of far-lane D-tails — and PERC's own verdict shows the far lanes
are Θ(r³), so no o(r³) upgrade of any far lane is reachable by inputs of
that kind). The v2.2 validity premise "the B4.loc dam-line tube certificate"
is discharged at a stronger grade than required.

## 3. RESTATED — PERC-DECAY at certificate grade (and an adversarial-ledger entry)

The honest verdict (PERC_DECAY §0, consumed): the far lanes are **Θ(r³),
NOT o(r³)** — the window ℓ = r³/6 fixes the order, and the corridor is alive
in the mean at level s out to the ghost-elder territory (certified axial
display: μ − s = +2.4e-8 @0.024 rising to +9.4e-2 @1.0; bottleneck z = 0.68
= Θ(1)), so the connectivity factor at the sub-kernel handoff does not
vanish as r → 0. **The "far lanes o(r³) under PERC-DECAY" phrasing carried
in v2.0–v2.2 is recorded as REFUTED at evidence grade** — a successful
attack forcing amendment, entered in the adversarial ledger. (No theorem
statement is touched: Theorem (2) claims C·r³, and Θ(r³) ⊆ O(r³); the
refutation kills only the stronger o(r³) reading, which is dropped.)

Restated premise (this lane underwrites it): PERC-DECAY := the far lanes
absorb at Θ(r³) with the certified constants, plus D-tails under the named
input PD-CONN. Certified today (rung, floor-consistent):

- B1.dir-far = FarRoute(d_max) ⊆ {N_w ≥ 1} (the away lobe's merge into C_M
  is necessarily a window saddle — C_M born at b, the away lobe at s, the
  merge height ∈ (s, b)); certified P(FarRoute) ≤ E_{P_r}[N_w]; the
  D3-certified part outside the C1 chart ≤ B_remote + I_hole = 21.9279 +
  1.284 = 23.2119·r³.
- B2-far ⊆ E_{P_r}[N_w(d ≥ 0.2)] ≤ 17.6802·r³ (crude-spine cap grade, R2
  floor).
- B4.rem: superseded by B4LOC-R1 (§1–§2).

PD-CONN (named, constants-not-order): level-s arm bound
P_{Q_r}(A_r(B(pair, 2r), D)) ≤ C₀e^{−c₀D}, r-uniform explicit constants;
missing pieces named exactly: (i) the explicit-constant planar Bargmann–Fock
arm bound (the missing RSW-with-constants theorem); (ii) the field-level
conditioned→unconditioned patch (Kolmogorov entropy machinery, not built;
finite-dimensional reversion certified); (iii) the torus→planar comparison
(available, ≤ 1e-120). Supporting certified displays: reversion profiles,
annulus cap density 7.8093e-7 → 1.3715e-9, B4.loc per-section dam
κ = Θ(1/r) re-verified, window-count D-tail exact (raw count Θ(r³) at every
D < 12 — the decay lives in connectivity, never in the count).

## 4. ADJUDICATION of the dir accounting (lead's question): the free-rider form

Question: with FarRoute ⊆ {N_w ≥ 1} certified, what does the C_RN·√Q(B1.dir)
term price beyond the free-rider inclusion?

Adjudication — the exact accounting:

1. The three events A, B1_wit := B1 ∩ {N_loop ≥ 1}, FarRoute are PAIRWISE
   DISJOINT (taxonomy disjointness of A and B1; the wit/dir split of B1;
   FarRoute ⊆ B1.dir) and each is contained in {N_w ≥ 1}: A ⊆ {N_qual ≥ 1}
   ⊆ {N_w ≥ 1}; B1_wit ⊆ {N_loop ≥ 1} ⊆ {N_w ≥ 1} (N_loop ≤ N_w pointwise);
   FarRoute ⊆ {N_w ≥ 1} (certified, §3).
2. Therefore P(A) + P(B1_wit) + P(FarRoute) = P(A ∪ B1_wit ∪ FarRoute)
   ≤ P(N_w ≥ 1) ≤ E_{P_r}[N_w] = E_w — a SINGLE payment of the carrier
   covers all three. This is a sharpening of, not a change to, H4JC-R1's
   display (its E_w budget already contains the away-merge saddles; the
   disjointness makes the attribution exact).
3. With B1.dir ⊆ NearSwap(d_max) ∪ FarRoute(d_max) (BRANCH_dir, frozen):

       1 − q(r,·) = P(A) + P(B1_wit) + P(B1.dir) + P(B2) + P(B4)
                  ≤ E_w(r) + P_r(NearSwap) + P(B2) + P(B4).

   **The C_RN·√Q(B1.dir) form prices NOTHING beyond NearSwap + the
   free-rider**: at the rung the dir term's content collapses to
   certified-super-algebraic NearSwap plus the FarRoute free-rider inside
   E_w. Recorded as a STRENGTHENING of Theorem (1′) for the next issuance:

       1 − q(0.05, 6/5) ≤ 8.1272827e-2 + P_{0.05}(NearSwap)
                          + P_{0.05}(B2) + P_{0.05}(B4),

   with P(NearSwap) super-algebraic certified (BRANCH_dir) and
   P(B4) ≤ 1.22e-9 certified (B4LOC-R1). The C_RN ≤ 3.47 machinery remains
   consumed where it belongs: the B4 RN transfer (B4LOC's explicit form).
4. Flagged check for the next pass (NOT consumed here): PERC's inclusion
   B2-far ⊆ {N_w(d ≥ 0.2) ≥ 1} would, once the pointwise relation
   N_β* ≤ N_w is pinned against C2's exact N_β* definition, extend the
   free-rider union by B2-far (disjoint from A/B1 by the taxonomy), i.e.
   P(A) + P(B1_wit) + P(FarRoute) + P(B2-far) ≤ P(N_w ≥ 1) ≤ E_w. Until
   pinned, B2 keeps its own certified caps (far ≤ 17.6802·r³; corridor via
   its own line).

## 5. Register state after this round (effective at the next issuance)

**CLOSED:** the B4.loc dam-line tube certificate AND B4.rem (B4LOC-R1,
super-algebraic at the ladder rungs; whole-B4 per §2); PERC-DECAY in its
RESTATED form (the certified far-lane inclusions + caps of §3 — validity
content at the rung fully absorbed into the E_w accounting of §4); the
dir remainder at the rung (NearSwap certified + FarRoute free-rider).

**Adversarial ledger addition:** the o(r³)-far-lanes reading — REFUTED by
PERC's certified corridor-alive display (a successful attack forcing the
restatement); counts with the campaign's prior breaks-and-repairs.

**OPEN (validity premises of Theorem (2), reduced):** OBL-D1-PROMOTE —
chart side (H5's r-scaled rung family + interpolation; RUNNING: r = 0.025
cells done, stitch ~4–8h per the lead's status line) and now also the
uniform-band extension of the B4LOC/B2-far/far-route certificates (the
same interpolation machinery; the H3 band floor has already discharged the
normalizer sub-part); D3-LEMMA-RN-UNIF (rung part precisely stated +
uniform part; foundations).

**REFINEMENT/constants register:** PD-CONN (named; constants-not-order;
missing pieces (i)–(iii)); OBL-B1-BRANCH(loop|B1) (sharp loop factor,
refinement grade per the lead's disposition); OBL-D2-AO-SHARP (i, iii, v);
the B2-far free-rider pinning check (§4.4).

**Theorem-statement consequence (for v2.3):** Theorem (2)'s open validity
premises reduce to OBL-D1-PROMOTE and D3-LEMMA-RN-UNIF; Theorem (1′)
sharpens per §4.3. No frozen v2.2 line is edited by this note.

END_FROZEN_BODY

## Freeze record (outside frozen body)

Extraction rule (corpus convention): lines strictly between the unique exact lines
BEGIN_FROZEN_BODY / END_FROZEN_BODY of this file; normalize line endings to LF;
strip leading/trailing blank lines; retain exactly one terminal LF.

- Frozen body bytes: 10752
- Frozen body SHA-256: c1d5e95d97e019574d6b6b09e6e606fd6e0cbc1fe3aa2e0a54953de6e4f2c470
- Parent issuance (untouched): D1_ASSEMBLY_v2_2.md body
  490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6
- Consumed carriers (re-verified from bytes): B4LOC_DAMLINE.md body
  0d5c1b3284374f3a2f7630b7f603a71c77f5ec9417fe4579aaac03becd33c05e;
  PERC_DECAY.md body 5137a811e6be72ed27e0458c75537bdcfabb8d8e7148b211d8c7b9a1584df400.
