# STAGE-E REVIEW — EXACT-LAW / PROVENANCE (U2D-UPPER, D1 v2.0 frozen package)

**Reviewer:** independent exact-law/provenance lane (this file's only author).
**Date:** 2026-09-14. **Object:** the frozen upper-theorem package D1-ASM-20260914-v2.0 and
every carrier it consumes, under `/mnt/agents/output/K3_SIDE24_LB/UPPER2D/`.
**Prior exposure:** none — before this audit I had seen only the tasking prompt (the law
capsule, the register, the hunt list). No package content was known to me in advance.
**Coordination:** none; no other reviewer was contacted. Files under `STAGE_E/` owned by
other lanes (rerun transcripts, `h5snap/`, `attack_h4jc.py`) were read where noted and
left untouched.

## 0. DISPOSITION: **FAIL**

Six findings (two material to the consumed certified rung), three CANNOT-VERIFY items,
and a custody PASS on every register hash. The failures are localized and repairable;
none requires re-computation of the law, and the largest (F2) is a consumption-side
understatement of the remote bracket against its own source formula.

## 1. Custody (body-hash verification) — PASS on all nine register carriers

Convention per carrier as recorded in its own FREEZE block (marker segment for
B1/C2/D2/H4/D1; whole-file sha256 for C1/H3; separator rule for H5). All recomputed
from the bytes by this reviewer:

| carrier | body sha256 | status |
|---|---|---|
| D1_assembly/D1_ASSEMBLY_v2_0.md | 86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0 | MATCH (17988 B) |
| B1_taxonomy/B1_TAXONOMY.md | 7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930 | MATCH (= FREEZE.txt) |
| C1_alpha_intensity/C1_ALPHA_INTENSITY.md | ca2c02b81a05e4cbc955b1aeab1f4120e9b8d4fda328aa191da08a49db72853f | MATCH (whole file, = C1_RECEIPTS.txt) |
| C2_B_classes/C2_B_CLASSES.md | 2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2 | MATCH (= FREEZE.txt) |
| D2_branch_control/D2_BRANCH_CONTROL.md | 9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa | MATCH (= FREEZE.txt) |
| D3_percolation/D3_PERCOLATION.md | 8e7fef6b4fb989404624bf50083c9d8624fa486eeda62e4784447f8edb3eac47 | MATCH (= FREEZE.txt) |
| H3_closure/H3_CLOSURE.md | 387e4bae4881a036486f17f317ae5e26b7d94bd5e1e829cc76946d090fb81637 | MATCH (whole file, = MANIFEST/RECEIPT) |
| H4_closure/H4_CLOSURE.md | a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3 | MATCH (16734 B, = FREEZE.txt) |
| H5_closure/H5_CLOSURE.md | 465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301 | MATCH (body = bytes before "\n---\n\n") |

Companion pins independently re-verified: LEAD_INTENSITY_DERIVATION.md body
729644f3521d29c5013abc1182ed2db6a61e5b150ea5be4e5355f0fbadf01ef3 (own convention) MATCH;
H2 cov_exact.py f08c1c5f653f…3553e783, pin_transform.py c6988ac7fcfa…9891fd41 MATCH;
H5 h5_kernel.py b2e6014c…ad83ad7, falsify.py 5bc9dbcc…bfeee0c MATCH CODE_HASHES.txt;
H5 falsify_modeA.txt ≡ falsify_modeB.txt = 3ef903bb…0d53920 MATCH; historical carriers
D1_ASSEMBLY.md 006b8a7d…, v1.1 addendum 634338b4…, v1.2 addendum a7e1958c… all MATCH
(intact, historical). `d1_falsify.py` re-run by this reviewer: PASS, exit 0, and
python3 vs python3 -O byte-identical (digest 2bc0ff9f80882c19fbc8780068220775f8bc71c361cb2f5f001e94f2aceacf2c,
matching D1_V2_RECEIPTS.txt; t_normal.txt ≡ t_opt.txt = 11b711d3…).

Custody note (not a carrier): `H5_closure/H5_STATE.md` line 29 quotes
`BODY_HASH.txt = 465c97c07cf633d7ac3d1f06d3a6ae5d2cbc051843245426c62c01b7964823ff`, which
does NOT equal the actual frozen body hash (465c97c0553e…; the on-disk BODY_HASH.txt
itself is correct). H5_STATE.md is an unfrozen status note, not consumed; the D1 ledger
cites the correct value and was independently recomputed. Reported as a misquote in a
non-carrier; no custody failure.

## 2. Findings (FAIL), exact locations

### F2 (MATERIAL) — the consumed remote bracket drops D3's own (1+κ_far) factor; a QMC estimate rides inside a "certified" bracket. Rules 4 + 6.

D3's frozen theorem line (D3_PERCOLATION.md, frozen body lines 277–279):

> "I_far(r) = E_{P_r}[N_w({d ≥ 5})] ≤ (576 − 25π)·J(ℓ)·(1 + κ_far(5))
>  = 2.5283·r³·(1 + κ_far),  κ_far(5) ≤ 0.63 + ε_QMC
>  (certified cross bound 0.63 axis-worst + QMC RN 1.00±0.01)"

but H5 (H5_CLOSURE.md §1 table, R5 row) and D1 (D1_ASSEMBLY_v2_0.md §3, remote row;
§2(1)) consume it with κ_far set to zero:

> H5: "R5 (remote) | d_pair ≥ 0.1 | D3-certified envelope (17.02 + 2.53)·r³ = 19.55·r³ | D3-certified (consumed)"
> D1: "remote = 2.4438e-3 | 19.55 | the D3 spine bracket (I_ann 17.02 + I_far 2.5283 = 19.5483 exactly, consumed at the conservative round-up 19.55·r³ = 2.44375e-3), consumed once"

The far-zone RN cross term is certified positive (κ_cross(5) ≤ 0.626 axis / 0.047
transverse, D3 §4 table; the crude κ-free spine at d = 5 is ≈ 5× the plateau, so the
2.5283 figure can only be the refined route's κ = 0 base). Against D3's own certified
formula the far term is ≤ 2.5283·(1.63 + ε)·r³ ≈ 4.12·r³; the consumed 19.55·r³ bracket
is an **upper bound taken below its machine-readable authority** by up to
2.5283 × 0.63 × r³ ≈ 1.99e-4 absolute at r = 0.05 — comparable to the entire
frozen-vs-current slack of I_hi (2.93e-4). D3's own FREEZE.txt headline is internally
inconsistent in the same direction: "…<= (1.284 [CONSUMES C1] + 17.02 + 2.53(1+kap_far)) r^3
= 20.9 r^3 spine form" — with the (1+kap_far) factor present the sum is ≈ 22.4, not 20.9.
Additionally the bound as stated carries "ε_QMC" / "QMC RN 1.00±0.01" — a measured
estimate inside what is consumed as a certified bracket (rule 6). Repair: either consume
17.02 + 2.5283·(1+κ_far) with κ_far's certified bound (raising the remote bracket to
≈ 21.1·r³, I_hi impact ≈ +2.0e-3·r³-scaled, well inside the 569× honesty margin), or
produce a certified rung-level statement that the far-zone RN factor is ≤ 1 exactly.

### F1 — H3's prose "certified lower bound" for c₀ is rounded UP against its own certified interval. Rule 4.

H3_CLOSURE.md line 162 (and identically line 32):

> "**Certified lower bound: c_0 ≥ 3.230978535287004948096246569018163150765 > 0.**"

The certified interval printed three lines above (lines 159–160) and the machine-readable
authority (RECEIPT_h3.json `c0_certified_lower`) both give the lower endpoint
3.23097853528700494809624656901816315076**4858**0105587…; the prose display
…50765 exceeds it by +1.42e-40 (39-digit round-to-nearest used where a lower bound must
truncate DOWN). The H3 cert transcripts (cert_normal.txt/cert_O.txt lines 67–68) print
the same collapsed 39-digit interval. Immaterial downstream — every consumed c_Z display
(H3 receipt 1.615489267643502474048123284509081575382; D1 §4(a) 1.615489267643502474;
H4 receipt truncation …4705656) truncates down correctly, and H4's CK2 recomputes c₀
independently to 1e-50 — but the sentence as written is a lower bound rounded upward.

### F3 — H5-AXIS v3 amendment is not carried by the frozen, consumed carrier; the three records disagree on the S-disk radius (0.03 / 0.06 / 0.062). Rule 8 / provenance.

- v1 is preserved as historical ✓ (H5_CLOSURE.md line 70: "v1 above preserved as
  historical record"); v2 carries its own certified edge-probe receipts ✓
  (h5_results_r0.05_sdisk.jsonl; five ring values quoted in the frozen body).
- v3 exists in the frozen body ONLY as the headline parenthetical "M-ring sec 0
  collapsed to 0 via the v3 S-disk skip" (line 18). The frozen H5-AXIS statement (§1)
  quantifies only v1 wedges + the v2 disk **d_S < 0.03**, and the frozen
  coverage-completeness paragraph still routes "d_S < 0.03 → S-disk". The v3 enlargement
  (S-annulus inner band [0.024, 0.06] excluded from the stitch sum; S-disk boundary
  exemption at 0.062; merger area π·0.062²) lives only in (a) the drifted on-disk
  h5_merge.py, (b) the UNFROZEN tightening amendment
  H5_CLOSURE_TIGHTENING_2026-09-14.md whose own freeze block reads "body-sha256:
  PENDING", and (c) the unfrozen H5_STATE.md ("sdisk 3.876e-3 (d_S<0.062 envelope)").
- v3's edge-probe receipts DO exist on disk (h5_results_r0.05_sring.jsonl: five probes
  at d_S = 0.065 plus one at 0.085, ρ_hi ≤ 0.1231, all below the envelope constant
  0.17134) — but no frozen or semifrozen text states the v3 quantifier or cites them.
- The consumed number itself matches neither claimed radius in the current records:
  on-disk h5_totals.json sdisk = 0.0038755839… = 0.3426768601·π·**0.06**² exactly, while
  D1 §3 consumes "sdisk = 3.876e-3 (H5-AXIS v2+v3 full S-disk **d_S < 0.062**)" and the
  current h5_merge.py computes with **0.062**² (which would give 4.138e-3). The
  [0.06, 0.062] annulus is still machine-paid inside the stitch S-annulus in the
  consumed merge (band exclusion reads [0.024, 0.06]), so there is no coverage hole in
  the consumed totals — but the radius claimed in the assembly (0.062) is not the radius
  charged (0.06), and neither is the radius documented in the frozen carrier (0.03).

### F4 — D1 theorem (1) presents the remote bracket as "D3-certified" inside the certified rung while its zone-uniformity lemma (D3-LEMMA-RN-UNIF) is OPEN in D1's own register. Scope/grade rule.

D1_ASSEMBLY_v2_0.md §2(1): "…(H5; machine-certified box cover of the whole chart plus
the **D3-certified remote bracket** 19.55·(0.05)³ = 2.4438e-3 consumed exactly once
inside it; the rim and axis envelope regions stand at named-lemma grade — H5-RIM,
H5-AXIS v1+v2+v3 …)". The parenthetical discloses named-lemma grade for rim/axis but
not for the remote part. D3's own grade summary (D3_PERCOLATION.md §5) states the
spine is theorem-shaped only "modulo (i) and the zone-uniformity step (ii)", with
(ii) = D3-LEMMA-RN-UNIF (sup over each zone attained on the displayed probe net up to
smoothness margins) — registered OPEN in D1 §5 row 3 and needed at the rung itself, not
only for the promotion. The same document's §8 is honest ("the remote spine at D3's
grade"); the theorem statement is not. Combined with F2, the rung's remote ingredient
currently stands at named-lemma grade with a number below the source formula.

### F5 (display-grade) — the C_RN ladder display rounds the r = 0.025 rung DOWN; part-ledger displays are round-to-nearest, not round-up. Rule 4.

H4_CLOSURE.md lines 146–147:

> "C_RN(r) := √(E_{Q_r}[(det H_M det H_S)²]) / (c_Z r²)  = 3.44, 3.46, 3.46, 3.47 at r = 0.1, 0.05, 0.025, 0.0125 → Θ(1) EXPLICIT."

Against the authority record (RECEIPT_h4.json, `C_RN_cert_ladder_certificate_rigorous`
= [3.438178, 3.459094, **3.464340**, 3.465651]), the r = 0.025 entry displays 3.46 <
3.464340 — downward for an upper-bound prefactor (the other three rungs round up).
D1 §7(4) inherits it ("vs 3.44/3.46/3.46/3.47"), and D1 §4(b)'s numerator display
"30.85/31.23/31.32/31.35" likewise rounds the same rung's numerator down (31.32 vs
31.32023 implied by the certified ladder). Mitigation: the operative theorem-grade
claims round correctly — D1 theorem (1) C_RN(0.05) ≤ 3.46 ≥ 3.459094 ✓; H4 line 243
"C_RN(r) ≤ 3.47" uniform ✓. Same display-semantics family: D1's "I_hi = 9.142887704e-02
= 731.43·(0.05)³" (true coefficient 731.43102; display down by 1.0e-3) and the H5 part
ledger displays (round-to-nearest throughout; see CV2 for the stitch line). All are
cosmetic at the assembly's slack scale; flagged because the campaign rule is
round-outward on every certified display, not round-to-nearest.

### F6 (display-grade) — C1 states the planar moment a₆ = 15 flatly under the heading "Exact lattice moments". Rule 2.

C1_ALPHA_INTENSITY.md lines 33–35:

> "Exact lattice moments a₂ = K1''(0)-magnitude = 1 − ε with ε = 9.65…e-123 (Poisson
> dual route; the torus law is never replaced by the planar one), a₄ = 3 (to < 1e-60),
> a₆ = 15."

Recomputed by this reviewer (mpmath, dps = 160): a₆ − 15 = **−3.119518111e-117** ≠ 0
(a₄ − 3 = +5.50194882540715e-120, so the sibling qualifier "(to < 1e-60)" is true; the
a₂ display 9.65254179896e-123 is correct round-to-nearest of 9.65254179895991e-123).
a₆ = 15 is a planar-limit value presented as an exact finite-torus identity, one line
after the document swears off exactly that move. No computational path uses a₆ (grep:
the string occurs only in this sentence; cov_exact.py carries no a₆ constant), so the
impact is nil; the sentence needs the same tolerance qualifier as a₄.

## 3. CANNOT-VERIFY (separate from FAIL)

- **CV1 — the frozen H5 first-merge full-precision totals.** The frozen carrier quotes
  I_hi = 9.142887704e-02 and 4–5-digit part displays; the full-precision h5_totals.json
  of that merge was overwritten by the refine-3 re-merge (on-disk now
  9.1399568624463…e-02). I therefore cannot verify that 9.142887704e-02 rounds UP the
  frozen full-precision sum, nor reconcile the frozen part displays exactly. Strongly
  mitigated: the frozen parts at their displayed values sum to 0.09137368 ≤
  0.09142888 with 5.5e-5 margin, the drift direction is monotonically downward and is
  now thrice corroborated (frozen 9.1428877e-02 > on-disk 9.1399569e-02 > the other
  lane's snapshot STAGE_E/h5snap/totals_at_snapshot.json 8.6318918e-02), and the
  consumed object is the total, which is conservative.
- **CV2 — part-level drift of the stitch sector.** The frozen ledger displays
  "stitch = 6.8550e-2"; the current on-disk json reads stitch = 0.06857627282…, LARGER
  than the frozen display by 2.6e-5 (part-level drift UP while the total drifted DOWN,
  plausibly from the v3 band-boundary move shifting area between stitch and sdisk).
  d1_falsify.py cks the drift direction only on the TOTAL; the frozen per-part
  certified values are unrecoverable (CV1). No rule is violated by the total, but the
  statement in D1 §1 that "refinement only lowers certified uppers" is verified only at
  total level.
- **CV3 — the "45 cks" receipt count.** d1_falsify.py printed 44 PASS lines plus the
  digest line in both modes on my re-run; D1_V2_RECEIPTS.txt and D1 §7 say "45 cks".
  Presumably the digest/two-mode identity is counted as the 45th; the inventory does
  not enumerate them, so the exact count is unverifiable from the receipts. Cosmetic.

## 4. Hunt-list disposition (items not already covered above)

1. **3D firewall — PASS.** `ind_S = 2` occurs only as (a) the H3 §4 NAMED AMENDMENT
   documenting the transcription defect (literal reading gives c_0 = 0, demonstrated
   executably; 2D typing det H_S < 0 / index 1 consumed instead — the certified c₀
   interval uses the corrected event), and (b) the historical v1.1 addendum (labeled).
   H3's receipt line "result: CLOSED … with one named amendment (2D typing; literal
   ind_S=2 transcription gives c_0=0)" is exactly the correct handling. No AO48-OPR-045
   citation exists anywhere in the tree. The only "3-dim" string in a consumed carrier
   is C2's "3-dim marginal" (a 3-component Gaussian vector marginal used to calibrate an
   MC engine) — not the 3D theorem. D1 §4(e)/§6 explicitly register MU-5. D3's
   reference to "DER-027a (variance channel floor 0.9795957, d >= 3, 7-pin lower
   family; consistency display only)" is labeled consistency-only, never proof — PASS.
2. **Planar-limit values — PASS except F6.** Finite-torus moments are used exactly
   wherever load-bearing (H3 c₀ certificate, Poisson identities ck'd in every driver;
   H1 hardening mutation m6 enforces rejection of the 1−a₂ = 0 substitution; H3's
   planar diagnostic is labeled and lies inside the certified interval; D3's
   "s_u² = (a₄−a₂²)/2 = a₂² up to 2.8e-120 (certified Poisson correction; the finite
   torus is never planar)" is an honestly-qualified approximation). No Schur
   diag(2, 1/2, 1/2, 1/6) presentation exists.
3. **Superseded carriers — PASS.** v1.0/1.1/1.2 assemblies are hash-preserved and
   explicitly superseded by v2.0 §0; C*_env = 1.284 is labeled evidence-grade and
   explicitly retired by D1 §2 ("supersedes … the evidence-grade C*_env = 1.284"); the
   C020/C021/C012 lineage is "historical … structure and display only" (C1 line 33-ff.,
   D3 §4); the LPW side is disclaimed as premise (D1 §6; B1 §7 placement only,
   "starting material, not authority"); no W8 rung and no defective WP missing-sqrt
   bound is cited as authority in any consumed carrier. H3's citation of the
   LPW_CONSTANT package for λ_min(Γ_0) > 31/250 is backed by in-falsifier reproduction
   (F2, ten exact Sylvester minors) — self-certifying, and the derived uniform floor
   "0.12380…" truncates DOWN correctly (0.124 − 19.071156e-5 = 0.12380929).
4. **Rounding direction — FAIL: F1, F2, F5 (see §2).** All other consumed constants
   check out: I_hi 9.142887704e-02 ≥ on-disk 0.0913995686… ✓; remote 2.4438e-3 ≥
   2.44375e-3 ✓; c_Z displays truncate down ✓; H4 uniform C_RN ≤ 3.47 ✓; D1's falsifier
   recomputes C_RN(0.05) ≤ 3.46 in exact Decimal ✓; the λ_min floor 31/250 is never
   used as a live bound in a consumed carrier (H5's pin-frame λ_min ≥ 0.1190 is a
   separate per-run certificate printed in logs at %.4f — log display only).
5. **"independent" — PASS (one observation).** Uses are provenance-classified
   ("independent code path", "independent quadrature class", "independent falsifier
   engine (direct lattice sum at N=96, certified tail 4.1e-129)", "independently in
   interval arithmetic (CK2)", D3's Gaussian "η independent of H_pair" is a derived
   regression fact). B1 line 317's "(Four independent adversaries re-attack after
   freeze…)" does not name them — contextually the B2a–B2d attack lanes; a status
   remark, not a probabilistic claim. Observation only.
6. **Measured estimates as theorem constants — FAIL only via F2's ε_QMC.** Otherwise
   disciplined: the projected ~6–7e-2 tightening is "DOCUMENTED, NOT consumed, never
   presented as current" (D1 §3/§6); the 0.025 rung is "LABELED CRUDE; no exponent
   fitted" (D1 §6, H5 §5); C_unif exists only as the named OBL-D1-PROMOTE; AO ~ 0.9997
   is "labeled, never proof" (D2 FREEZE); QMC tables display SEs and are grade-fenced
   in each carrier's grade summary.
7. **Zero observed counts as zero probability — PASS.** H3's zero occurrences of
   {H_S ≺ 0} in 1.5e6×4 + 1.2e6×3 samples is corroboration for an a.s.-limit argument
   backed by a Cantelli bound (≤ 2.7e-5 → 0), not a probability assertion; B2a's
   "never observed" remarks attach to definitional impossibility arguments; P_r(N) = 0
   everywhere rests on Lemma R (analytic), not on sampling. D2-COUNT's
   N_qual ∈ {0,1} is a proved structural statement with an executable mirror
   (1437/1437), not a sample count.
8. **H5-AXIS v1→v2→v3 — FAIL: F3.** v1 preserved ✓; v2 receipted ✓; v3 unfrozen and
   radius-inconsistent (0.03/0.06/0.062) with receipts existing only on disk.
9. **I_hi drift — PASS.** The on-disk json reads SMALLER (9.1399568624463…e-02); D1
   consumes the FROZEN 9.142887704e-02; the direction is conservative; d1_falsify.py
   block (3) cks it fail-closed (re-verified by this reviewer in both modes). The code
   drift (h5_run.py 38813a02…/h5_merge.py 0d65872f… vs receipt pair 10546c2c…/1a0d82b1…
   vs lead-ledger pair 07acc4d0…/91abc378…) is honestly receipted in D1 §1; the two
   load-bearing code pins (h5_kernel.py, falsify.py) are unchanged and verified.

## 5. Load-bearing dependencies (paths + hashes, all verified by this reviewer)

- D1_assembly/D1_ASSEMBLY_v2_0.md — body 86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0
- B1_taxonomy/B1_TAXONOMY.md — body 7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930 (+ b1_falsifier.py 818df798… pinned by C2/D2/D3/H4)
- C1_alpha_intensity/C1_ALPHA_INTENSITY.md — whole ca2c02b81a05e4cbc955b1aeab1f4120e9b8d4fda328aa191da08a49db72853f
- C2_B_classes/C2_B_CLASSES.md — body 2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2
- D2_branch_control/D2_BRANCH_CONTROL.md — body 9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa
- D3_percolation/D3_PERCOLATION.md — body 8e7fef6b4fb989404624bf50083c9d8624fa486eeda62e4784447f8edb3eac47
- H3_closure/H3_CLOSURE.md — whole 387e4bae4881a036486f17f317ae5e26b7d94bd5e1e829cc76946d090fb81637 (+ RECEIPT_h3.json c₀ interval)
- H4_closure/H4_CLOSURE.md — body a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3 (+ RECEIPT_h4.json C_RN ladder)
- H5_closure/H5_CLOSURE.md — body 465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301 (+ h5_kernel.py b2e6014c2fd6eb9e72b2104cc828a764e280f76b8396a02edd37654b6ad83ad7, falsify.py 5bc9dbccb384fc49366a32c9c19d431a68dea1ccae74233d1fbc458e5bfeee0c, falsify transcripts 3ef903bbe2609374b10dae984a2bcb521db6f800b6e481a6cf27bb7260d53920)
- LEAD_INTENSITY_DERIVATION.md — body 729644f3521d29c5013abc1182ed2db6a61e5b150ea5be4e5355f0fbadf01ef3 (coordinator (L1)/(L2))
- H2_foundations/cov_exact.py f08c1c5f653f2ffd2e85d39e1112f80d15db04d15f932e5ee42db2ca3553e783; pin_transform.py c6988ac7fcfa32dcc03c1374e13fb4c26af1c12c25cb824a315e6dce8991fd41
- D1 falsifier: d1_falsify.py, two-mode digest 2bc0ff9f80882c19fbc8780068220775f8bc71c361cb2f5f001e94f2aceacf2c (re-run PASS by this reviewer)

## 6. One-paragraph summary for the lead

Custody is clean end-to-end (all nine register body hashes, the LEAD foundation, the H2
library pins, the H5 code/transcript pins, and the historical carriers verify; the D1
falsifier re-runs PASS and two-mode byte-identical; the I_hi drift is conservative and
fail-closed ck'd). The package nevertheless FAILS exact-law/provenance review on six
localized findings, two of them material to the certified rung: (F2) the remote bracket
is consumed at 19.55·r³ = 17.02 + 2.5283 with D3's own certified (1+κ_far) factor
(κ_far ≤ 0.63 + ε_QMC, itself carrying a QMC estimate) silently zeroed — the source
formula supports ≈ 21.1·r³, an understatement of ≈ 1.99e-4 at r = 0.05, and D3's own
freeze headline arithmetic ("…2.53(1+kap_far)) r^3 = 20.9 r^3") is internally
inconsistent; (F4) theorem D1 v2.0(1) calls that bracket "D3-certified" while
D3-LEMMA-RN-UNIF (zone uniformity, needed at the rung) is OPEN in D1's own register —
the remote ingredient currently stands at named-lemma grade, like rim/axis, and the
theorem statement should say so. The four remaining findings are display/provenance
repairs: H3's prose c₀ lower bound rounds UP by 1.4e-40 against its own interval and
machine record (F1; downstream c_Z displays are all correct); H5-AXIS v3 is not carried
by the frozen consumed carrier and the S-disk radius reads 0.03 (frozen text) / 0.06
(consumed charge) / 0.062 (D1 §3, H5_STATE.md, current code) with v3's edge-probe
receipts unfrozen (F3); the C_RN ladder display rounds the r = 0.025 rung down to 3.46
against the certified 3.464340 (F5; the operative bounds ≤ 3.46 at r = 0.05 and ≤ 3.47
uniform are correct); C1 states "a₆ = 15" flatly under "Exact lattice moments" though
a₆ − 15 = −3.12e-117 (F6; no computational use). CANNOT-VERIFY, separately: the frozen
H5 first-merge full-precision totals (overwritten; total-level conservatism thrice
corroborated), part-level stitch drift direction (only the total is ck'd), and the
"45 cks" receipt count (44 PASS lines observed). Non-carrier note: H5_STATE.md
misquotes BODY_HASH.txt. The 3D firewall, planar-moment discipline (bar F6),
superseded-carrier hygiene, zero-count discipline, "independent" provenance, and the
frozen-value consumption of I_hi all PASS.

---

body-sha256: 43706f1997820d89427fecc8f9ab17e7e2b64b7d43e771200326ded59419377d
