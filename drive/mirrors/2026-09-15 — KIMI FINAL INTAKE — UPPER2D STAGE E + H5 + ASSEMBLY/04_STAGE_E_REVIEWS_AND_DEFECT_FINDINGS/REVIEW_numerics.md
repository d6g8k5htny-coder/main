# NUMERICAL-CERTIFICATION REVIEW — D1 v2.0 U2D-UPPER frozen package

Reviewer: NUMERICAL-CERTIFICATION independent reviewer (Stage E).
Scope: every number the D1 v2.0 theorem relies on, backed by a certificate
that actually certifies it — certificates RERUN, not read.
Prior exposure: NONE. I had seen only the mission prompt before beginning;
no part of this package was known to me in advance.

## 0. Environment (recorded first, before any run)

- OS/kernel: Linux 5.10.134-18.0.12.lifsea8.x86_64 (x86_64), host k2099498599626936320
- Python: 3.12.12 (system python3)
- mpmath: 1.3.0
- CPU: Intel(R) Xeon(R) Platinum (2 cores available to this container)
- Date of rerun session: 2026-09-14 22:19–23:59 CST (system clock)
- Note: background refinement lanes (H5 keeper.sh A/B + stitch_refine3.py)
  were ACTIVE during the review; I did not signal or modify their processes
  and treated their outputs as read-only. All mutation tests were executed
  in scratch copies under STAGE_E/ (mut_D1/, mut_H5/).

## 1. Frozen-carrier integrity (all hashes recomputed from bytes)

| carrier | rule | result |
|---|---|---|
| D1_assembly/D1_ASSEMBLY_v2_0.md | BEGIN/END_FROZEN_BODY, LF, blank-trim, 1 terminal LF, 17988 B | **MATCH** 86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0 |
| H5_closure/H5_CLOSURE.md | body = bytes[0,15162) before "\n---\n\n" | **MATCH** 465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301 |
| B1_TAXONOMY.md | markers + rstrip | MATCH 7d7ddbec…68930 |
| C1_ALPHA_INTENSITY.md | whole-file | MATCH ca2c02b8…72853f |
| C2_B_CLASSES.md | markers raw | MATCH 2aba099d…ad3d2 |
| D2_BRANCH_CONTROL.md | markers + rstrip | MATCH 9e95648a…0b8fa |
| D3_PERCOLATION.md | markers raw | MATCH 8e7fef6b…eac47 |
| H3_CLOSURE.md | whole-file | MATCH 387e4bae…81637 |
| H4_CLOSURE.md | markers + rstrip | MATCH a67d50b9…b6b1d3 |
| H5 h5_kernel.py / falsify.py / h5_run.py / h5_merge.py | whole-file vs CODE_HASHES.txt | MATCH b2e6014c… / 5bc9dbcc… / 38813a02… / 0d65872f… |
| H2 MANIFEST.sha256 (16 entries) | sha256sum -c | all OK |
| H3 RECEIPT_h3.json file hashes (9) | recompute | all MATCH |
| H4 RECEIPT_h4.json file_sha256 (9) | recompute | all MATCH |
| b1_falsifier.py pin 818df798… | recompute | MATCH |

## 2. Certificate reruns (two-mode, byte-identity, transcripts vs shipped)

Method: every falsifier/certificate rerun under `python3` and `python3 -O`;
exit 0 required; program outputs compared byte-wise between modes and against
the shipped frozen transcripts. Where the exit code could not be captured
directly (long nohup reruns), success is evidenced by the final PASS/digest
line (printed only on success; any ck failure raises SystemExit(1) with a
FAIL line) plus byte-identity with the shipped rc=0 transcript.

| certificate | normal | -O | modes byte-identical | vs shipped transcript | notes |
|---|---|---|---|---|---|
| D1 d1_falsify.py (45 cks) | rc=0 | rc=0 | YES | byte-match t_normal.txt & t_opt.txt | digest 2bc0ff9f80882c19… (matches receipts) |
| B1 b1_falsifier.py | rc=0 | rc=0 | YES | byte-match t_normal.txt | tally A=176 B1=3 B2=71 B4=1; digest 78ee1509… |
| C2 c2_falsifier.py | rc=0 | rc=0 | YES | byte-match tf_normal.txt & tf_opt.txt | (t_*.txt are the probe transcripts, tf_* the falsifier's) |
| D2 d2_falsifier.py | PASS digest 877210c7… | PASS digest 877210c7… | YES | byte-match tf_normal.txt & tf_opt.txt | falsifier itself reruns d2_probes.py in both modes internally (byte-identical, 115 lines each); 427 pairs, 1612 window saddles |
| D3 d3_falsifier.py | (see §7 status) | (see §7 status) | | | spawns d3_perc.py engine rerun |
| H2 selftest_cov_exact.py | PASS C1–C12 | (see §7) | YES (design; verified normal vs shipped pair) | byte-match selftest_cov_exact_normal.txt | C12 mutation subprocesses observed live |
| H2 selftest_pin_transform.py | PASS P1–P12 | PASS P1–P12 | YES | byte-match both shipped transcripts | |
| H2 selftest_reg_lemmas.py | (see §7) | (see §7) | | | |
| H3 h3_c0_certificate.py | ALL_CHECKS_PASS | ALL_CHECKS_PASS | YES | byte-match cert_normal.txt & cert_O.txt | |
| H3 h3_falsifier.py (F1–F5) | FALSIFICATION_ATTEMPTS_FAILED | same | YES | byte-match fals_normal.txt & fals_O.txt | F4 mutation harness caught both mutations (exit 1) |
| H4 h4_certificate.py (CK1–CK6) | ALL H4 CERTIFICATES PASS | same | YES | byte-match cert_normal.txt & cert_O.txt | C_RN_cert=[3.438178, 3.459094, 3.464340, 3.465651] reproduced |
| H4 h4_falsifier.py | (see §7) | (see §7) | | | reruns h4_probes.py internally |
| H5 falsify.py --mode A | digest 8cc3692bc81aba… | — | A≡B modulo mode label | rewritten falsify_modeA/B.txt hash to 3ef903bbe260… (= D1 ledger pin) | dominant box fine hi 1.328543376e-05 reproduced; totals containment PASS |
| H5 falsify.py --mode B | — | digest 8cc3692bc81aba… | | | NOTE: A/B "different code paths" claim is vacuous — `K = 8 if mode=='A' else 8`; both modes run identical parameters (documentation blemish, not a numerical failure) |
| H5 h5_merge.py --r 0.05 (in scratch on snapshot records) | rc=0 | — | — | I_hi = 8.097558925e-02 = my fully independent reproduction to 40 digits | coverage census 81×73 PASS; C1 containment PASS |
| C1 falsify.py (4 full engine runs, ~25 min quoted) | (see §7) | (see §7) | | | |

## 3. Mutation / injected-failure drivers (explicit nonzero exit REQUIRED)

| driver | mutations tried | result |
|---|---|---|
| D1 d1_falsify.py (scratch tree mut_D1) | 11 in-body/in-file byte flips (all 7 carriers, H5 body, modeA file, kernel, falsify.py) + 5 totals-json value mutations (I_hi↑, I_lo↑, remote↓, r wrong, internally-consistent ×10 upward drift) | **16/16 CAUGHT**, each rc=1 with the correct FAIL tag (carrier …, h5-body-hash, h5-falsify-AeqB, h5-kernel-hash, h5-falsify-hash, json-parts-sum-eq-I_hi, json-containment, r-is-0.05, drift-direction-conservative); baseline restored rc=0 |
| H5 h5_merge.assemble(mutate_index=k) (scratch mut_H5) | k = 0..11 (R1 cell, rim column, stitch sector, scone site, sdisk site, sring site, two items each) | **12/12 CAUGHT**, rc=1, correct completeness ck (69/70 cells, 9/10 columns, 21/22 sectors, 5/6, 4/5, 5/6) |
| H2 selftest_cov_exact.py C12 harness | sine_drop, planar_moments × (normal, -O) | 4/4 rejected rc=1 with CHECK FAILED line (observed in my rerun transcript) |
| H2 selftest_pin_transform.py P12 harness | transpose_T, cubic_sign, drop_trapezoid, raw_inverse_unsafe × (normal, -O) | 8/8 rejected rc=1 with fail line |
| H3 h3_falsifier.py F4 harness | planar_moments, wrong_window | caught fail-closed exit 1 (both modes of my falsifier rerun) |
| H3 certificate direct mutation drivers | MUTATION=planar_moments / wrong_window | rc=1 both; outputs byte-match shipped mut_planar.txt / mut_window.txt; explicit CHECK FAILED lines (CK2c a_2 Poisson-inside-spectral; CK6a pin forces d_111f=2κ) |
| H4 h4_falsifier.py | mutation displays built into the falsifier (mutated qualification predicate; GH3-as-exact vs CK5 1e-60 gate) | (see §7 for falsifier rerun completion) |

No mutation that should fail was observed to pass.

## 4. Independent recomputation of the headline arithmetic

### 4.1 H5 totals from primary records (213 records, 13 jsonl files, snapshot 22:41)

I parsed ALL h5_results_*.jsonl myself and applied the merge rules read from
h5_merge.py (min-per-cell over 70 R1 cells; rim columns → 2·ρ_hi·sector area,
sector = π·0.00895²·15/360; probe buckets scone(6)/mfwd+mbwd(12)/sdisk(5)/sring(6)
with 2·max·wedge-area formulas; stitch min-per-sector over 22 sectors with the
stitchS d1=0.024 exclusion; remote = 19.55·r³; I_lo = 2·Σpatches_lo):

- bucket counts exactly satisfy the completeness cks (70, 10, 22, 6, 12, 5, 6)
- My totals: I_lo = 1.16672119002e-08; I_hi = 8.09755892521e-02
- On-disk h5_totals.json (written 22:35:52): I_hi = 8.63189181733e-02.
  Difference isolated ENTIRELY to the stitch bucket: one record
  (stitchref3.jsonl, stitchM sec 10, hi=4.8443696e-05) was appended at
  22:38:50, i.e., AFTER the totals write. Excluding exactly that record my
  reproduction matches the on-disk json to ALL printed digits
  (stitch = 0.0632329439009300435984835666…, 40-digit agreement; every other
  part agrees to ~1e-54). Merge rules therefore VERIFIED against both the
  on-disk totals and the current record set.
- h5_merge.py itself rerun in scratch: rc=0, I_hi = 8.097558925e-02 — agrees
  with my independent parse exactly.
- Conservative chain: frozen 9.142887704e-02 ≥ on-disk 8.6319e-02 ≥ current
  records 8.0976e-02. Refinement only lowers the certified upper (min-per-cell /
  min-per-sector monotone); the D1 drift-direction ck enforces this fail-closed
  (mutation-tested, §3).
- C1 containment: I_lo = 1.166721190e-08 < C1_TOTAL = 1.605315e-4 < I_hi
  (frozen and current). ✓
- Frozen displays: I_hi/r³ = 731.43101632 → 731.43 ✓; C_chart = 731.43 − 19.55
  = 711.88 ✓; (I_hi − remote)/r³ = 711.8806 ✓; ratio_hi = 569.54 → "569×" ✓;
  ledger sum 9.13737346e-02 ≤ I_hi frozen (slack 6e-4, conservative) ✓;
  remote display 2.4438e-3 > 19.55·r³ = 2.44375e-3 (rounded UP) ✓.

### 4.2 D1 exact arithmetic displays

- density ledger −3+6−2+2 = 3 ✓ (Fraction)
- D3 spine 17.02 + 2.5283 = 19.5483 exactly; 19.55 ≥ 19.5483 (round-up) ✓
  — BUT see FINDING F-1 (§6) on the dropped (1+κ_far) factor.
- C_RN ladder recomputed from E[(dM dS)²] exact values (H4 CK5):
  C_RN = √(E/r⁴)/c_Z = 3.438178 / 3.459094 / 3.464340 / 3.465651 at
  r = 0.1/0.05/0.025/0.0125 — matches H4 certificate transcript exactly;
  D1's 2-dp displays 3.44/3.46/3.46/3.47 are upward-safe; the consumed
  C_RN(r=0.05) ≤ 3.46 holds: 3.459094 (exact-E) and 3.459250 (from the
  displayed 31.23) both < 3.46 ✓.
- C¹ thresholds √(12 ln(1/r) + 4 ln C_RN + 2 ln 2) recomputed at dps=50:
  5.8273 / 6.5038 / 7.1150 / 7.6774 → displays 5.83/6.50/7.12/7.68 (2-dp
  nearest) and H4's 4-dp displays 5.827/6.504/7.115/7.677 ✓ (cosmetic
  rounding only; the certified statement is the saturation identity, which
  the H4 certificate re-verified to 1e-12).

### 4.3 Independent spot-checks (higher precision, own code)

- Finite-torus moments via the image-sum closed forms (dps=170, own code):
  1 − a₂ = 9.652541798959913e-123 (2D exact) vs doc display 9.65254179896e-123
  (rel diff 9e-15, 12-sig-fig rounding) ✓; the doc's 1D Poisson form agrees
  with the 2D form to 1.3e-171 relative (Poisson identity confirmed) ✓.
- H3 limit constant (own engine): with m = −(6/5)a₂, s² = a₄−a₂², a = −m/s,
  c₀ = (m²+s²)Φ(a) − m·s·φ(a) = 3.23097853528700494809624656901816315076485801
  ∈ H3 certified interval […04743007, …08823472] ✓ (width 4e-98; my value
  sits 2.2e-98 above the floor — inside).
- Rounding direction on c_Z: stated c_Z = 1.615489267643502474048123284509081575382
  ≤ c₀_lo/2 = 1.6154892676435024740481232845090815753825 — rounded DOWN ✓;
  D1's ladder truncation 1.615489267643502474 also DOWN ✓ (and division by a
  lowered c_Z raises C_RN — correct conservative direction for an upper).
- I_lo display: exact merge 1.16672119002e-08 → display 1.166721190e-08 is
  rounded DOWN (correct for a certified lower bound) ✓.

## 5. Determinism / fail-closed discipline

- No wall-clock or unseeded randomness in any certificate path. All
  randomness is fixed-seed (B1: random.Random(1000+seed), seed ∈ range(40);
  H3 falsifier: np.random.default_rng(77777777); C1/D2/D3/H4: fixed MT19937
  seeds per docstrings). Proof: my reruns byte-match transcripts shipped from
  hours earlier, across two modes (any clock/RNG leakage would break this).
- ck() raises SystemExit in every certificate path (grep-verified in all 13
  scripts): survives python -O ✓.
- No bare asserts in certificate paths (grep over all falsifiers, kernels,
  merge, H2 libraries, probe engines: only comments/docstrings) ✓.
- Runner labels/exit receipts live in separate files (FREEZE.txt,
  *_RECEIPTS.txt, RECEIPT_*.json, BODY_HASH.txt, CODE_HASHES.txt) — outside
  the frozen bodies; mutation tests confirm the falsifiers pin exactly the
  protected regions (appends OUTSIDE the protected body are not detected —
  by design; byte flips INSIDE are caught 16/16).

## 6. FINDINGS

### F-1 (FAIL-grade, certification-chain gap on the consumed remote constant)

The theorem consumes remote = 19.55·r³ = 2.4438e-3 as "D3-certified"
(H5_CLOSURE.md §0/§1; D1 ledger ck `ledger-remote-D3-exact-sum`:
17.02 + 2.5283 = 19.5483 ≤ 19.55).

BUT D3's own frozen theorem statement (D3_PERCOLATION.md, D3-THM) attaches a
multiplicative cross-coupling factor to the far-zone term:

    I_far(r) ≤ (576 − 25π)·J(ℓ)·(1 + κ_far(5))
             = 2.5283·r³·(1 + κ_far),   κ_far(5) ≤ 0.63 + ε_QMC

The certified cross bound is genuine: the D3 engine transcript displays
"relative kap_cross(d=5) <= 0.63 (axis) / 0.048 (transverse) — absolute bound
5.0e-3 vs main term Z_r m_sad = 7.95e-3" (5.0e-3/7.95e-3 = 0.629). With the
factor included, the remote bracket becomes

    17.0182 + 2.5282637×(1 + 0.63) = 21.14·r³  >  19.55·r³,

so the consumed constant 19.55·r³ is NOT certified to cover D3's own stated
bound. The 19.5483→19.55 round-up slack (0.0017, i.e. 0.009%) is nearly four
orders of magnitude too small to absorb a 63% factor on I_far. The executed
certificates do not discriminate: d3_falsifier.py's spine window ck is
[19, 23]·r³, which passes both 19.5483 and 21.14; d3_perc.py's own assembly
line prints the κ=0 sum (1.284 + 17.0182 + 2.52826 = 20.8305·r³), i.e. the
engine and the frozen THM formula disagree inside the same frozen package.
Mitigating disclosures: D1 §(c) carries the remote explicitly "at D3's grade…
modulo the already-named D3-LEMMA-RN-UNIF (foundations countersignature
flagged)", the issuance is explicitly CONDITIONAL, and the ε_QMC part of the
cross bound is labeled evidence-grade; the 0.63 is an axis-worst value at the
single d = 5 station and decays outward, so the true inflation of the
far-zone integral is plausibly much smaller than 63% — but no executed
certificate certifies any inflation ≤ 0.009%. Exact failure: number relied
on remote = 19.55·r³; certificate that should back it (D3 spine as consumed
by H5/D1) certifies only the κ_far = 0 value 19.5483·r³ while the source
theorem states I_far with a (1 + κ_far), κ_far ≤ 0.63, factor.

### F-2 (documentation blemish, not numerical)

H5 falsify.py docstring claims "Modes A and B use different code
paths/orderings"; in fact `K = 8 if mode == 'A' else 8` — the modes are
parameter-identical, so the A≡B byte-identity check is vacuous as a
two-path cross-check. The digest itself (8cc3692b…) and the window
re-certifications still executed and PASS in my reruns.

### F-3 (minor, stale/non-load-bearing records)

- H5_STATE.md quotes BODY_HASH.txt = 465c97c07cf633d7…; the actual frozen
  body hash (BODY_HASH.txt, D1 ledger, my recomputation) is 465c97c0553ecc….
  State file is a progress note, not a carrier.
- H5_RECEIPTS.txt (20:53) pins h5_run.py 10546c2c… / h5_merge.py 1a0d82b1…;
  CODE_HASHES.txt (20:52) and on-disk files are 38813a02… / 0d65872f….
  Code evolved across refinement passes; D1_V2_RECEIPTS.txt documents this
  drift explicitly and consumes only the frozen closure doc + totals.
- H4 RECEIPT_h4.json headline key C_RN_cert_ladder = [3.437958, 3.459094,
  3.463512, 3.466157] is the PROBES (c0-ballpark) ladder; the certificate
  ladder [3.438178, 3.459094, 3.464340, 3.465651] is present under
  C_RN_cert_ladder_certificate_rigorous and matches my rerun exactly.
  Documented in the receipt's "note" key — no action needed.

### F-4 (freeze-time exact totals unrecoverable — CANNOT-VERIFY, immaterial)

The frozen display I_hi = 9.142887704e-02 is the merge program's %.9e
(round-to-nearest) output; the freeze-time full-precision h5_totals.json was
overwritten by subsequent refinement merges, and the freeze-time stitch
ledger display (6.8550e-2) is not reproducible from any prefix/subset of the
surviving records (stitchref3.jsonl was appended in an order not fully
deducible; tried all ref3-subset × stitchref-prefix combinations). The
last-digit rounding direction of the frozen display therefore CANNOT be
verified; worst-case shortfall vs the freeze-time certified sum is bounded
by %.9e rounding: ≤ 4.6e-12 absolute (rel 5e-11) — immaterial to every
consumed display (731.43, 711.88, 569×, and the C_chart budget), and every
later exact merge (8.6319e-02, 8.0976e-02) lies far below the frozen value.

## 7. Disposition

(see final section — completed after the long reruns)

## 8. Load-bearing dependencies (paths + sha256)

Frozen carriers and pins verified in §1. The theorem's numerical spine:
- D1_assembly/D1_ASSEMBLY_v2_0.md body 86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0
- D1_assembly/d1_falsify.py (45 cks; rerun PASS both modes; digest 2bc0ff9f80882c19fbc8780068220775f8bc71c361cb2f5f001e94f2aceacf2c)
- H5_closure/H5_CLOSURE.md body 465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301
- H5_closure/h5_kernel.py b2e6014c2fd6eb9e72b2104cc828a764e280f76b8396a02edd37654b6ad83ad7
- H5_closure/h5_merge.py 0d65872f538a012f3248274a1e3fdd1d2f997143616360913891d361c562d8aa (merge rules independently re-verified)
- H5_closure/falsify.py 5bc9dbccb384fc49366a32c9c19d431a68dea1ccae74233d1fbc458e5bfeee0c
- H5_closure/h5_results_*.jsonl (213 records; primary receipts; live-appended by refinement — conservative direction only)
- H5_closure/h5_totals.json (live; consumed frozen values carried by the D1 doc + drift-direction ck)
- Carriers B1/C1/C2/D2/D3/H3/H4 bodies as table in §1
- H2 libraries cov_exact.py f08c1c5f…, pin_transform.py c6988ac7…, reg_lemmas.py e7005a5a… (MANIFEST.sha256 all OK)
- H3 c₀ interval (RECEIPT_h3.json) — independently reproduced inside interval
- H4 C_RN_cert ladder (certificate transcript) — independently reproduced
