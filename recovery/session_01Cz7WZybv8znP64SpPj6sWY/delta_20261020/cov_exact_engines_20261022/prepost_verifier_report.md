PRE-POST CHECK: <SCRATCH>/covtrace/posts/record.md

Verdict: **POST_WITH_FIXES.** Two fixes change what the draft says. Fix 1: one sentence calls a value conservative when for odd q it is not a bound at all. Fix 2: the draft sets aside a file that is actually a certificate built on class-3 values. The rest are line numbers and wording.

Below, "UPPER2D/…" means <SCRATCH>/covtrace/src/K3_SIDE24_LB/UPPER2D/…. I posted nothing, edited nothing and ran no engine. My one script is an independent re-implementation of a single lattice sum: <SCRATCH>/prepost_chk/lamq.py, run with the scratchpad venv python, `-I -B`.

## CONFIRMED

- **The archive.** The zip is 30,148,285 B, 1,794 members, sha256 a2136bc033f3…, with 251 .py members. All 251 extracted .py files are byte-identical to their zip members. objects.json on a01c72f1 (origin/chatgpt/drive-github-hardening-20260919, not merged into main) lists a2136bc0… at 30,148,285 B.
- **The table.**
  - Every byte size and sha256 prefix is correct, including the 3 identical h5_kernel copies.
  - TM lines 435, 356, 433, 354, 491, 672, 701, 703, 311, 339, 457, 459, 458 are correct.
  - H3 MANIFEST lines 9 and 7, RECEIPT_h3.json lines 89 and 87, FREEZE_PERC_DECAY.txt:32, d3_perc_decay.py.txt:65 and CL-PIN-001…:9 are correct.
  - audits/vault_99/2026-09-25/EXTRACTED_FACTS_BATCH1.md:64 has `85d7725fab42…` (12 hex) and 103166 B. No other pin for d3_rn_unif exists on b7f72a71. main has no 99_DO_NOT_OPEN paths.
  - SOURCE_CAPSULE (lines 142–169) carries every digest except d3_rn_unif.
- **Line citations that are correct.**
  - h3_rung_floor: L22-25, L79, L80-82, L143.
  - h3_c0_certificate: every citation (L36-38, 44, 84-86, 108-119, 128-129, 136-141, 203, 216-220).
  - h5_kernel: L104, L463-478, L520-529, L1196, L1272-1273.
  - Also c2_probes L419, d3_amend_v2 L66-67, pin_transform L289-302, cov_exact L281-286 and L297-302, and d3_rn_unif L169-184 as the validation block.
- **Classifications that are correct.**
  - Rung floor: every M–S entry is k1(n1, ±r)·k1(n2, 0), spectral, at r ≠ 0. That is class 3, and it is used two-sided.
  - h3_c0: a₂, a₄, a₆ come from moment_a(m), spectral iv, which is ±e_sum(2m, 0). That is class 1. c₀ depends only on a₂, a₄ and iv_Phi; a₆ enters only det G₀. Of the six iv covs, 2 are even order and 4 odd order at s = 0, and they feed only CK3, CK3b and CK3c. one_minus_a2_poisson(iv) at L93 is the Poisson path and a gate only.
  - c2_probes L419 is at (dx, dy) = (0.03, 0.004) and is a gate.
  - h5_kernel: pair_law appears only as its 4 definitions and is never called. anchors is called only from falsify.py:42, in both copies (H5_closure/ and STAGE_E/mut_D1/H5_closure/). PinFrame evaluates the twin at t ∈ {0, ±r}, and KernelSeries at dxM, dxS, dyv and 0.
  - The twin's formula has the same shape as the e_sum/s_sum iv branch: tot/Z_T + [−τ, τ], with the same tail formula and J = 116.
- **Completeness claims that are correct.**
  - 32 of the 251 files name cov_exact.
  - The dps claims hold: h3 files at L49-51 and L56-58, c2_probes at L79, h5_run imports at L22 and sets dps at L25-26, the r* drivers at L46 and L49-50. configure() refuses dps below 60.
  - The mpf-only files are all mpf, and d3_perc has exactly 12 evaluator calls.
  - pin_covariance and reg_lemmas are called only from their own selftests.
  - The numbers in "What this shows" match the disposition 6329640342, as does "feeds D3's Z_LO and D1's C_RN".

## DISCREPANCIES

**1. `lambda_q` is called class-2 / conservative, but for odd q it is not a bound (draft line 45).**
- **What the code does.** `UPPER2D/H5_closure/h5_kernel.py:79` sums `wj*(kj**q)` with the sign kept, not |k|^q. Over the symmetric lattice, odd q cancels.
- **Evidence from my script** (dps 40, same lattice):
  - q = 7: the signed sum over Z1.a is about ±2e-38; the true Σw|k|^7/Z is 38.30.
  - q = 9: about ±2e-37 against 306.4.
  - q = 11: about ±2e-36 against 3063.9.
  - Compare the H3 `abs_sum_A`, the actual class-2 pattern: it uses `abs(kk)**n` (h3_band_floor.py:331).
- **Odd q is actually used.** The Taylor-model remainder is `LAT.lambda_q(n+TD+1)` (L538).
  - StationBox has TD = 5 (L559), so q is odd whenever n is odd, e.g. f×fxy in the Grr cross block (L607).
  - CoarseBox has TD = 6 (L1328), so q is odd whenever n is even.
  - h5_run uses both classes.
- **Proposed wording:** "`lambda_q` (L75-82) divides the signed sum Σ w k^q by `Z1.a`. For even q that is the class-2 pattern. For odd q the sum cancels to about 1e-38, so it is not the bound on sup|K1^(q)| that its docstring states (Σw|k|^7/Z ≈ 38.3). The Taylor-model remainder calls it with q = n + TD + 1 (L538), which is odd for odd n under TD = 5 (L559) and for even n under TD = 6 (L1328). This is outside 01Fi's rule. I did not trace its effect on any printed number."

**2. reg_lemmas is set aside as "reached only from its own selftest", but that selftest is a certificate (line 63).**
- `UPPER2D/H2_foundations/selftest_reg_lemmas.py:3-4` says it is "certifying obligations OBL-B1-REG-G1, G4, G5".
- It calls `rg.cond_cov(..., 'iv')` and `rg.pin_inverse(R05, 'iv')` (L91, L111, L150, L178, L193, L228). These reach `ce.cov(..., 'spectral', mode)` (reg_lemmas.py:100-101) at nonzero separations: the pins are at ±r/2 and the points at e.g. (1, 0.5).
- The results are eigenvalue lower bounds of interval Schur complements, e.g. "R1 certified active floor … > 6.1e-3" (L94-128). That is class 3, used two-sided, at dps 100 and cutoff 30.
- **Proposed wording:** "`reg_lemmas.py`'s iv `cov` (L100) is reached only from `selftest_reg_lemmas.py`, but that selftest is the certificate for OBL-B1-REG-G1/G4/G5. Its floors (e.g. R1 > 6.1e-3, L94-128) rest on spectral iv `cov` at nonzero separation, used two-sided: class 3. It is not in the disposition's list. I did not trace who consumes those certificates."
- Also scope the headline to "of the engines the disposition listed, two …", and add this case to "What this does and does not show".

**3. Line numbers that are off.**
- Rung floor "three Grams (L146-148)" should be L147-149. L146 is a print.
- "Lattice twin (L47-97)" should be L46-98.
- "symmetric tail pad (L84-97)": the pad is at L98, so cite L84-98.
- "|k| ≤ 30 only (L56-59)" should be L55-59, and the range is |j| ≤ 116, i.e. |k| ≤ 116π/12 ≈ 30.37, the same lattice as the library at KMAX = 30.
- "lambda_q (L76-82)" should be L75-82.

**4. d3_rn_unif's two calls are described wrongly (line 60).** "2 cov calls validating its closed-form kernel" is not right. Only L178 validates the kernel. L189 (`_covidx`) builds the y-free pin Gram G6, X6, H6 and SPAIR0 (L188-207). Both are mpf, so the conclusion stands. Proposed wording: "2 mpf `cov` calls: one validates its closed-form kernel (L169-184), the other builds its y-free pin Gram and pair block (L188-207)."

**5. "Hash pin only, no library call" is too strong (line 62).** These files do reach cov_exact at runtime, through another module, all in mpf:
- d3_amend, d3_amend_v2 and d3_perc_decay call `E.station` and friends in d3_perc.
- h4_falsifier imports h4_probes (L319, `h4p.PinKit`).
- certify_axis_break and isserlis_break use `CG.GridLaw` from c1_grid, which calls `ce.cov(...,'mpf')` at c1_grid.py:73.
- Proposed wording: "Hash pin, no direct evaluator call. They reach cov_exact only through d3_perc, h4_probes or c1_grid, all mpf."

**6. BRANCH_dir does use the library (line 30).** It does not name cov_exact, but it gets the module through h4_probes:
- br_mc.py:47 and br_probes.py:27 do `ce = h4p.ce`, via spec_from_file_location on h4_probes.py.
- They make mpf `ce.cov` calls (br_mc.py:102, 147; br_probes.py:67, 69, 90, 97, 100, 103).
- br_mc.py:49-50 read the private mpf lattice `ce._LATT`. There is no iv call.
- W8_lambda neither names nor imports it, but has its own mpf π/12 lattice.
- Proposed wording: "BRANCH_dir/ (4 Python files) does not name `cov_exact` but uses it through h4_probes (`ce = h4p.ce`): mpf `cov` calls and a read of the mpf `_LATT`. There is no iv call. W8_lambda/ (3 Python files) does not use it."
- Add to Limits: "the scanner only reads files whose text names `cov_exact` or `f08c1c5f`."

**7. The "Not traced" list reads as complete but is not (line 73).**
- About 52 files outside the 32 contain a π/12 lattice.
- Ones that use interval arithmetic and are not listed: K3_SIDE24_LB/W3_numerics/w3_cert.py (an iv spectral derivative engine, L129-143), W9_scope_027a/, lb2_cert.py and its KIMI copy, 19fcef2e-c302…/verify_ridge_two_sided_v1.py, and QC_RETURN03_REVIEW/.
- The path "H1_v2_hardening/" should read "UPPER2D/H1_v2_hardening/".
- Proposed wording: "…does not name `cov_exact`, including LPW_CONSTANT/, UPPER2D/H1_v2_hardening/lpw_constant_v3.py, W3_numerics/w3_cert.py, W9_scope_027a/ and 027c/, REVIEW_LPW/, W2_symbolic/, lb2_cert.py and others (about 50 files)."

**8. "Orders 9–18 come only from the twin" overstates the range used (line 44).**
- The certified path uses the twin for every order, from 0 up to NMAX = 4 + TD + 2, which is at most 12 (L525). lambda_q goes up to q = 11. 18 is only the twin's allowed ceiling.
- Proposed wording: "Every order on the certified path comes from the twin, up to 12 at TD = 6 (the twin accepts 18). The library stops at 8."

**9. Minor wording.**
- Give CL-PIN-001's full path: `incoming/claude-audit-package-recovery-20260928/05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/…`. Every other cell is `<v3>`-relative.
- "(4 files)" and "(3 files)" should say "Python files". BRANCH_dir has 31 members.
- "No file calls configure() with a non-default cutoff": selftest_cov_exact.py:339 calls `configure(kmax=0)` as a test that must be refused. Say "no file sets a non-default cutoff".
- c2_probes: write "(dx, dy) = (0.03, 0.004)".
- anchors: say "only the two `falsify.py` copies call it".
- PinFrame and KernelSeries "include nonzero arguments"; they also evaluate at 0.
- The L36-38 quote is cut off. It continues "…or from the in-program certified series for Phi".
- Attribute "proven contained" in the headline to the rule ("class 1, proven contained per 01Fi's note").
- Line 58 extends the disposition's grade note, which named only B4LOC and D3. Say it "would apply".
- The pickup promised all 32 files, and 11 go unmentioned. Suggested line: "The other 11 are cov_exact.py itself; make_receipts.py (names only); selftest_cov_exact.py and selftest_pin_transform.py (iv only in self-test gates; pin_covariance in iv at r = 1e-5, 0.025, 0.5 for P11); b4loc_driver, c1_grid, c1_intensity, h3_falsifier and d3_rn_unif_dbg (mpf only); and h3_band_ceil/floor (mpf `cov`, the iv Poisson guard, and the class-2 `abs_sum_A` over `_LATT_IV` at L325-335, already assessed by the disposition)." selftest_reg_lemmas is covered by item 2.

## Fixes required before posting
1. Correct the `lambda_q` sentence (item 1).
2. Add the reg_lemmas selftest certificate finding and scope the headline (item 2).
3. Fix the line numbers in item 3.
4. Reword items 4–8.
5. Optionally, the minor wording in item 9.