# DEPENDENCY_DAG — RETURN_06 theorem dependency graph

Notation: `A ──consumes B @ grade G──▶` means A's certificate/theorem depends on B at grade G.
Grades: EXACT / CERTIFIED-INTERVAL / NAMED-LEMMA / VALIDITY-PREMISE (open) / REFINEMENT (open) /
EVIDENCE-DIAGNOSTIC (never load-bearing) / HISTORICAL (consumed nowhere).
All nodes hash-pinned in SOURCE_CAPSULE.sha256.

## Foundations layer

```
LEAD_INTENSITY_DERIVATION.md (L1)/(L2) ──@EXACT──▶ B1, C1, C2, D2, D3
H2_foundations/cov_exact.py  f08c1c5f… ──@EXACT library──▶ C1, C2, D2, D3, H3, H4, H5, BRANCH, W8-lane
H2_foundations/pin_transform.py c6988ac7… ──@EXACT library──▶ (same consumers)
H2_foundations/reg_lemmas.py e7005a5a… ──@EXACT (Lemma R countersignature G1/G4/G5)──▶ B1 (taxonomy gaps named by B2b closed), D2
```

## Deterministic topology layer (2D)

```
B1_TAXONOMY.md 7d7ddbec… (coverage/disjointness: on Reg ∩ TYP, 1−q = P(A)+P(B1)+P(B2)+P(B4))
   ──@EXACT──▶ C1, C2, D2, D3, D1(1)
   attacked by B2a (outside-cover), B2b (boundary), B2c (ties), B2d (torus): all NO-BREAK / gaps closed
C1_ALPHA_INTENSITY.md ca2c02b8… (α-class exact intensity; kernel factors EXACT @80 dps; g(v), Z_r,
   plane integral EVIDENCE-GRADE QMC) ──@EXACT kernel + chart envelope──▶ D3 (I_hole = 1.284·r³), D1(1)
C2_B_CLASSES.md 2aba099d… (B1 loop / B2 younger-merge / B4 non-adjacency classes; B4 lane per-cut
   P ≤ (√E[W²]/Z_r)·Q(sup_ζ f ≤ s)^{1/2}) ──@EXACT──▶ D1(1), D2
D2_BRANCH_CONTROL.md 9e95648a… (D2-COMPLEMENT 4-channel partition PRE/NA/LOOP/YNG; G-table) 
   ──@EXACT──▶ D1(1); its tube item (ii) ≡ OBL-D2-AO-SHARP(ii) ──@ASSERTED-NOT-ESTABLISHED──▶ B4.loc premise
```

## Certified quantitative layer (2D)

```
H3_CLOSURE.md 387e4bae… (EXISTENTIAL: Z_r ≥ c_Z r², c_Z = c₀/2 ≥ 1.615489267643502474, r₀ existential)
   ──@EXISTENTIAL──▶ D1(2)'s r₀; consumed NOWHERE at the rung (out of scope there — proof F2)
H3_RUNG_FLOOR.md 6347275d… (R2: Z_{0.05} ∈ [7.7592917375327855e-3, 1.1468646473404396e-2] certified
   interval, +92.12% margin; stretch rungs 0.045/0.055)
   ──@CERTIFIED-INTERVAL──▶ D3_REMOTE_AMENDMENT_v2 (κ pieces + I_ann at Z_lo, drift-ck pinned),
                             D1(1) C_RN(0.05) ≤ 3.46, G.7-at-the-rung DISCHARGE
H4_CLOSURE.md a67d50b9… (H4 branch dams; H4-SD slice-dam vacuity — no single-slice C⁰ dam, certified;
   C¹ tube route displayed certified-dead; §2.3 pointwise H4-JC RETRACTED, preserved historical)
   ──@EXACT-minus-retraction──▶ D1(1)
H4_JC_repair/H4_JC_EVENT_LEVEL.md 42ee88da… (H4JC-R1 event-level joint carrier; PASS-WITHSTOOD round 2;
   OLD 132/175 violations pinned fail-closed; NEW 0/439) ──@EXACT──▶ D1(1) once-payment of I_hi
H5_CLOSURE.md 465c97c0… (v1 rung interval) + H5_CLOSURE_TIGHTENING 7fefa17b… (H5-AXIS v3 in full) +
   h5_totals_v3.json 8d7028e4… (I_hi(v3) = 8.0975589252e-2 = 647.8048·r³ round-UP, version-pinned,
   downward-only drift discipline)
   ──@CERTIFIED-INTERVAL (box cover) + @NAMED-LEMMA (H5-RIM rim band; H5-AXIS v1+v2+v3 wedges/S-disk)──▶ D1(1)
D3_PERCOLATION.md 8e7fef6b… (A.rem spine: 20.9·r³ form; base carrier) 
   + D3_REMOTE_AMENDMENT_v2.md 6796deea… (CURRENT: B_remote = 17.6804 + 2.5282637·(1+κ_far),
   κ_far ≤ 0.68 floor-certified = 21.9279·r³, MC-free; v1 amendment 63d91cdd… HISTORICAL)
   ──@CERTIFIED (modulo D3-LEMMA-RN-UNIF)──▶ D1(1) remote part of Ĩ_hi
BRANCH_dir/BRANCH_DIR.md 8821c8fa… (q(0.05) = 0.99982, 1−q = 1.42·r³ exact continuous-field MC;
   channel price ladder R_ch; OBL-B1-BRANCH(dir) content) ──@EVIDENCE-DIAGNOSTIC──▶ consistency only, never premise
```

## Assembly layer (2D UPPER theorems)

```
D1_ASSEMBLY_v2_2.md 490ad6b2… ── consumes ALL of the above at their grades; gate d1_falsify_v3.py
   (92 cks, digest d800849e…, re-executed live exit 0) re-verifies the whole chain fail-closed.
Theorem (1)/(1′) [named-hypothesis grade]  ◀── H5-RIM, H5-AXIS v1+v2+v3 [NAMED-LEMMA]
                                        ◀── D3-LEMMA-RN-UNIF(r = 0.05) [VALIDITY, open]
Theorem (2) [CONDITIONAL] ◀── OBL-D1-PROMOTE [VALIDITY, open] ◀──┐
                          ◀── D3-LEMMA-RN-UNIF [VALIDITY, open]  │
                          ◀── PERC-DECAY [VALIDITY, open] ◀── convergence point of ALL far lanes:
                          │        B1.dir-far (FarRoute, handed off with the exact certified boundary),
                          │        B2-far, B4.rem
                          ◀── OBL-B1-BRANCH(loop|B1) [VALIDITY, open; re-pointed from the retracted
                          │        H4-JC loop factor; constant-level; E[N_loop] ≥ P(A) side consequence]
                          ◀── B4.loc dam-line tube certificate [VALIDITY, open; incl. the cut-net ≡ D2
                                   9-pin M-ward-tube (ii) identification — ASSERTED, NOT ESTABLISHED]
Refinement register: OBL-D2-AO-SHARP (i, iii, v) [REFINEMENT, open] ── sharpens constants only
OBL-D1-PROMOTE ──▶ OBL-H5-JETMOD / OBL-H5-ZBAND / OBL-H5-REMOTE-THRESHOLD (promotion sub-obligations)
                ──▶ H3 band-floor assessment (interval-r enclosure of R2's pointwise floors; RUNNING)
H5-RIM production path: shell_hi_fi (parked, p_sup-dominated); H5-AXIS production path: factored-det
   expansion det Σ_gg = (d_M d_S)⁴·Q(1+O(d)) (documented, not run)
```

## LPW lower lane (separate family within 2D)

```
LPW qualitative (endorsed) ──@review-grade──▶ LPW_CONSTANT v2 (c = 1.1357762846347807215698804e-43,
   r₀ = 1/2224640) ──@EXACT──▶ H1_v2_hardening lpw_constant_v3.py re-certification (same pair)
LPW_CONSTANT v1 [HISTORICAL DEFECTIVE] — consumed nowhere
```

## THE FIREWALL (structural, verified)

```
3D SIDE24 ratified upper chain (AO48-OPR-045)  ✕──no edges──✕  every 2D node above
   · no AO48-OPR-045 citation exists anywhere in the tree (provenance hunt-list §4.1 PASS);
   · the only "3-dim" string in a consumed carrier is C2's 3-component Gaussian MC-calibration marginal
     (not the 3D theorem); D3's DER-027a reference is labeled consistency-display-only;
   · combining the 3D upper with the endorsed 2D lower into an existential Θ(r³) statement requires
     separate operator adjudication and is NOT made by any carrier.
```

## Auxiliary open quantitative campaigns (no edges into the theorems above)

```
W3_numerics (WP lower-bound enclosure, RUNNING) — feeds only the WP disposition, not D1/LPW.
W8_lambda + Phase-2 workspace (Λ-side certified floor, RUNNING) — feeds only the Λ floor obligation.
Coefficient-limit campaign — NOT OPEN; no node; forbidden until the two-sided 2D theorem closes.
```
