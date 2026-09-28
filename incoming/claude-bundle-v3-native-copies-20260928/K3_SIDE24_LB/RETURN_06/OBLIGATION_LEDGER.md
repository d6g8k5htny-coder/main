# OBLIGATION_LEDGER — RETURN_06 (every named obligation; states read fresh at assembly 2026-09-15)

Content quoted from the obligation's own carrier (hash in SOURCE_CAPSULE.sha256). Grade classes:
VALIDITY PREMISE (of D1 v2.2 Theorem (2)) / REFINEMENT / NAMED LEMMA / UPGRADE-ONLY.

## 1. OBL-D1-PROMOTE — VALIDITY PREMISE of Theorem (2)

- **Content (D1 v2.2):** uniform certified chart envelope on (0, r₀′] — the all-small-r promotion; the
  existential two-sided normalizer H3 stands separately (closed since v1.1); the continuum band is
  unexecuted and routed here. C's leading content in Theorem (2) is the uniform envelope constant.
- **Falsifier/discharge:** a certified rung family + band enclosure delivering the uniform envelope; killed
  by any rung whose certified envelope/r³ drifts UP beyond the ledger (falsify_promote.sh watches
  r-law displays: patches lo ratio vs r³ within 30%, edge-probe ratios vs r¹ within 35%).
- **Owner:** H5 lane + D1. **Execution state (fresh):** EXECUTING — r = 0.05 COMPLETE (v3 647.8048·r³);
  r = 0.025: patches COMPLETE (lo 7.1329e-10), probes COMPLETE, cells EXECUTING (83+19 banked jsonl
  lines), stitch EXECUTING (6 lines); r = 0.035355: cells EXECUTING (42+20), stitch 3; last writes
  04:53–05:08 CST 2026-09-15; κ = 1/8 modulus band displayed (12 certified points, c2(r) = 1/2 − r²/8
  + O(r⁴), max residual 3.4e-6 — DISPLAY, certified band enclosure remains OBL-H5-JETMOD).
- **Sub-obligations:**
  - **OBL-H5-JETMOD:** certified interval bounds for the full 24-jet set over r-bands [r_{k+1}, r_k] with
    lattice-tail constants re-certified uniformly in band. Falsifier: a band enclosure whose width exceeds
    the claimed modulus. State: OPEN (display only, see above).
  - **OBL-H5-ZBAND:** Z_r window bounds over bands (point-rung windows banked; lo rides H3's c_Z·r²
    theorem-grade uniform bound; hi side needs the band LPW bracket). State: OPEN; the H3 band-floor
    assessment (h3_band_floor.py, Z_r ≥ c_Z r² for ALL r ∈ (0, 0.05], no whitening, no MC in the
    certificate) is RUNNING — script active 07:58 CST, band_normal.txt empty, no banked output yet.
  - **OBL-H5-REMOTE-THRESHOLD:** certify D3's remote bracket at the r-scaled threshold d ≥ 2r per rung, or
    extend the chart's machine cover to an absolute d₀. Rides with D3-LEMMA-RN-UNIF. State: OPEN.

## 2. D3-LEMMA-RN-UNIF — VALIDITY PREMISE (rung part named hypothesis of Theorem (1) itself)

- **Content (D1 v2.2(1)(b) / D3 carriers):** zone uniformity of the RN brackets. Rung part (r = 0.05):
  the rigidity-decoupling lemma for τ(y) uniform over {d ≥ 5}; a certified interval-box Riemann sum for the
  annulus crude-spine integral. Closed at the rung WITHIN the item: all station κ pieces (floor-certified),
  the exact far-zone main term (576 − 25π)J(ℓ), monotone-decay exact-kernel evidence. Uniform-in-r part:
  as frozen.
- **Falsifier/discharge:** foundations countersignature of the two missing pieces; a counterexample station
  with τ-uniformity violation kills it. D3's falsifier enforces the bracket edge: rejects the κ = 0 sum
  19.5465 (operative window [20.54646, 22.5]; assembly gate extends: rejects below 21.92788306).
- **Owner:** foundations. **State:** NOT CLOSED (rung part precisely stated; uniform part frozen).

## 3. PERC-DECAY — VALIDITY PREMISE; the far-lane convergence point

- **Content (D1 v2.2):** subcritical level-s⁺ cluster decay feeding every far lane's o(r³) pricing:
  B1.dir-far (FarRoute, handed off with the exact certified boundary), B2-far, B4.rem.
- **Falsifier/discharge:** a certified subcritical decay theorem at the exact boundary; a far-lane summand
  certified Θ(r³)-or-larger would kill the premise's sufficiency (B2-corridor's raw envelope is already
  Θ(r³)-or-better by its power ledger, verified v2.1, unchallenged round 2 — that lane's kill stays
  refinement-grade).
- **Owner:** percolation lane. **State:** OPEN (named validity premise since v2.1; no lane execution
  recorded in the tree at assembly time).

## 4. OBL-B1-BRANCH(loop|B1) — VALIDITY PREMISE (re-pointed)

- **Content (D1 v2.2):** the re-pointed sharp loop factor given B1, constant-level; the E[N_loop] ≥ P(A)
  side consequence is carried. Re-pointed from the retracted H4-JC pointwise form after R1.
- **Falsifier/discharge:** a certified constant-level bound on the loop fraction via BRANCH_dir's
  exact-MC/certified-ladder machinery (channel price ladder R_ch(r) = 0.00142/0.00134/0.00114·r³ at
  r = 0.05/0.025/0.0125, CK3); a counterexample loop channel exceeding the constant kills it.
- **Owner:** branch-control lane. **State:** OPEN; BRANCH_dir carriers complete (rung025 landed
  2026-09-14T21:47:19Z; certificate byte-identical both modes post-errata).

## 5. B4.loc dam-line tube certificate — VALIDITY PREMISE (scope ruling V2)

- **Content (D1 v2.2, exact):** the uniform Gaussian sup-tail over the pair-level cut net with the RN
  prefactor (C2's B4 lane: per-cut P ≤ (√E[W²]/Z_r)·Q(sup_ζ f ≤ s)^{1/2}; the certified barrier-margin
  ladder κ = Θ(1/r) is kernel-grade already). **Caveat carried as part of the item:** the assembly's
  identification of the pair-level cut net with D2's 9-pin M-ward-tube item (ii) (OBL-D2-AO-SHARP(ii) ≡
  C2's tube item) is ASSERTED, NOT ESTABLISHED — the identification must be established, or the item
  stated and owned independently of D2's tube lane.
- **Why validity, not refinement:** no raw counted-class envelope for B4.loc exists anywhere in the
  carriers (grep-verified across C2/D2/D3/B1/H4 by the scope instance); the dam-line certificate is the
  ONLY route to any O(r³) bound on that summand.
- **Falsifier/discharge:** the certified sup-tail (net + entropy + RN) over the cut net, plus the
  identification proof or independent statement. Gate v3 cks the premise's presence in Theorem (2)
  (removal kills the gate).
- **Owner:** branch-control / foundations lanes. **State:** OPEN.

## 6. OBL-D2-AO-SHARP (i, iii, v) — REFINEMENT register only

- **Content (D2 G-table):** (i–iii) the certified net+entropy+RN form of (G2) P_y^w(NA_y) ≤
  RN_y·exp(−κ̄²/4 + entropy) — κ_dam displayed Θ(1) scale-invariant (mid-window 11.98–1.9·10⁵ across the
  α support at r = 0.05; weakest carrying station arch45c 11.98; probes, evidence); (v) the certified
  profile form of (G4) P_y^w(YNG_y) ≤ E[N_band(y)] + far. With AO ≤ 1 exact these sharpen constants, NOT
  validity. Item (ii) has MOVED to the validity premises (item 5 above); item (iv) (G3 confinement) is
  outside the round-2 refinement register per D1 v2.2.
- **Owner:** branch-control lane. **State:** OPEN (refinement; probe-grade displays stand).

## 7. H5-RIM and H5-AXIS v1+v2+v3 — NAMED LEMMAS inside the rung certificate

- **Content (tightening amendment 7fefa17b…):** H5-AXIS v3 in full — ρ_w(y) ≤ C_region per region with
  machine-certified constants C_scone = 0.18998, C_mfwd = 1.3254, C_mbwd = 1.6522e-05, C_disk = 0.34268
  (each = 2 × max certified edge-probe ρ_hi; CoarseBox + coarse_fine_hi, η = 2.5e-4, K = 8); flatness
  hypothesis stated per region with receipts. v1→v2→v3 chain: cones only → S-disk 0.03 (poison cell
  th15_dl0.066) → S-disk 0.062. H5-RIM: the rim-band region of the rung certificate.
- **Production paths:** H5-AXIS — the factored-det expansion det Σ_gg = (d_M d_S)⁴·Q(1 + O(d)) with Q
  certified nonzero (documented, NOT RUN; retires the v3 quantifiers); H5-RIM — shell_hi_fi machine
  certification (parked, p_sup-dominated). Until they land, the lemmas re-certify per rung in r-scaled form.
- **Grade:** NAMED LEMMA (probe constants machine-certified; flatness stated with receipts).
- **Owner:** H5 lane. **State:** OPEN production paths; named-hypothesis grade suffices for Theorem (1).

## 8. H3 band-floor assessment — normalizer promotion (feeds OBL-D1-PROMOTE / OBL-H5-ZBAND)

- **Content (h3_band_floor.py header):** CERTIFIED uniform normalizer floor Z_r ≥ c_Z r² for ALL r ∈
  (0, 0.05] — exact Taylor series at r = 0 of the corrected-pin/cross/soft Grams, exact series division,
  interval evaluation on sub-intervals, certified band lower bound via Wick/conditional-Wick Gaussian
  algebra; no whitening, no MC in the certificate; fail-closed, both modes byte-identical, mutation harness.
- **Owner:** H3 lane. **State:** RUNNING (script modified 07:58:13 CST; band_normal.txt created empty
  07:58:22 CST — no banked output at assembly time). **Never evidence until receipted.**

## Closed-at-the-rung items (for the record; not open)

G.7-scope normalizer AT THE RUNG — DISCHARGED by R2 (every consumed 1/Z_r divides Z_lo =
7.7592917375327855e-3); OBL-B1-REG — CLOSED; H3 existential form — CLOSED (c_Z = c₀/2); H2 at the O(r³)
grade — CLOSED; H4 except the re-pointed loop factor — CLOSED; H4-JC as H4JC-R1 — CLOSED
(PASS-WITHSTOOD); OBL-B1-BRANCH(dir) — CLOSED (NearSwap certified super-algebraic; FarRoute → PERC-DECAY);
the remote bracket floor-consistent — CLOSED modulo D3-LEMMA-RN-UNIF; the H5 rung interval (v3 totals) —
CLOSED (version-pinned, downward-only).
