# Review guide: H3 band certificate successor v2

**Status: a draft for review. Scientific effect NONE. Review status REVIEW_REQUIRED.** This guide is for the package `experiments/h3_band_v2/` of PR #447. It was written by the Anthropic author's session (`session_01Fij9F1cYrAUDW7PhWQze1G`), and same-provider agents checked each section. It therefore carries zero independence credit. It is a map for a reviewer from another model family, not a review. For each claim that rests on a written derivation rather than on a run, it gives the statement, the derivation with line references, what the run checks, and what it does not check. Where the guide and the code disagree, the code governs, and the disagreement is a finding.

**References.**
- `h3_band_v2.py:N`, or a bare line number inside a scope section, means line N of `experiments/h3_band_v2/h3_band_v2.py`. That file has sha256 `7776d78a0df5ff04d6aabd9f0e853c1d70c69ed039007919e77b64c182860431`, the digest bound in `results/ceil.json` and `results/floor.json`.
- `README:N` is the package README.md, `test:N` is `test_h3_band_v2.py`, and `cov_exact.py:N` is `h2/cov_exact.py`.
- `ceil.txt:N` and `floor.txt:N` are the frozen `h3_band_ceil.py.txt` and `h3_band_floor.py.txt`, in the frozen directory named at README:11. The frozen `H3_BAND_CEIL.md` and `H3_BAND_FLOOR.md` sit beside them.

**Provenance of the sections.** The four scope sections were written and adversarially checked against an earlier program, sha256 `99eceb98…`. The final program differs from it by fail-closed guards, comments, and record checks that run only after the certified values are computed. Its certified decimals and both transcripts are byte-identical: transcript sha256 `d0def90d…` (ceil) and `c80b7a0f…` (floor). The differences:
- `ck(up(var_iv) > 0)` in `t0_lower` (414);
- `ck(x.b < 1)` in `gram_tail` (884);
- `ck(var.a > 0)` in `trunc_mom_iv` (890);
- `ck(oneside is not None or a == -b)` in `piece_outside`, before the two-sided tail (1234);
- `structure` mode under `--mutation`, which now prints `H3_BAND_V2 MUTATED STRUCTURE RUN (NAME): …; NOT A CERTIFICATE` and exits 3 (1451-1454);
- the comments at 1149-1151 and 1203;
- the record checks:
  - the runtime gate in `compare_record` (1406-1427);
  - the producer pin `MPMATH_PIN` (1366);
  - the transcript-header mode `transcript_mode` (1369-1375);
  - `verify-record`'s checks for an unknown header and a non-object record (1458-1477);
  - the replay's record check before the run (1484-1488).

  These read only records and transcripts, so they change no certified decimal.

Every line reference below was remapped to the final file and re-read there. Where a section had described one of these guards as missing, it now describes the final code.

## How to use this guide

### What the run checks by itself

- **Run commands:** README:97-113. They cover `structure`, `ceil --verify`, `floor --verify` and the unit tests. A run prints one summary line, and the exit codes are listed at README:109-112. `structure` needs only the standard library (the import guard at 38-42).
- **Fail-closed checks.** Every `ck` (101-104) that fails stops the run with `H3_BAND_V2 FAIL: …` (in `structure` mode, `STRUCTURE FAIL`) and exit 1. No PASS line is printed, and no transcript or record is written: those are written only after `run_certificate` returns (1489-1501).
- **Replay.** `--verify` requires the fresh record to equal the reviewed one: the program digest, the cov_exact digest, the mpmath version, the transcript digest and every certified decimal (`compare_record`, 1406-1427). `verify-record` compares the same fields, without recomputing, so it does not depend on the verifier's runtime and needs only the standard library (1458-1477).
  - **Producer pin.** Every check requires the reviewed record to name the pinned producer mpmath, `MPMATH_PIN` (1366, 1420-1424). A test binds that constant to `requirements.txt`. Only a recomputing replay also compares the verifier's own mpmath with it.
  - **Mode.** `verify-record` reads the mode from the transcript's first line (`transcript_mode`, 1369-1375), which the transcript digest binds, so a record relabelled to the other envelope fails.
  - **Malformed inputs.** A record that is valid JSON but not an object, and a transcript whose first line names no envelope, give `VERIFY_FAIL` instead of passing or a traceback (1462-1471). The replay checks the record's type before it runs (1484-1488). Invalid JSON and a non-UTF-8 transcript still stop with a traceback and exit 1. The transcript digest is taken over the text as read, so a copy with CRLF line endings still matches.

  The tests run `verify-record` with no site-packages and with a faked mpmath version. They check that a changed check line is caught by the transcript digest, that a record naming another mpmath or relabelled to the other mode is rejected, and that a replay still rejects a record made with another version (test:131-222, 321-345).
- **Negative controls** (README:115-145). There are fourteen mutations. Twelve are caught inside the run at a named check, and `drop_fail_piece` and `old_remainder` only by the replay. The record controls are listed there too. The tests are at test:74-98 (structure mode) and test:367-423 (full runs).
- **What the controls do and do not show.** They show that each named check can fire, and that the checks are not vacuous against those regressions. They do not show that the inequalities used between the checks are true. Those inequalities are the subject of this guide.

### What rests on a written derivation

| scope | claim | entry points (final lines) | imports |
|---|---|---|---|
| 1. CB2s pole lemma and the first cell | the Gram Laurent series have only the poles (a)-(b) allow; μ and Σ have no pole, exactly | `structure_lemma` 212-237 (tables 154-170, `laurent_expand` 173-183, `solve_in_span` 186-209), run at 268 and by `structure` (1444-1457); the Gram series 460-568; used by `first_cell_law` 1118-1122 | CB1 moments |
| 2. CB4 R-majorant induction | on (0, 2⁻¹⁴], μ and Σ lie within MT and ST of their order-60 Taylor polynomials, and G(t) is invertible | 705-885: A_n(ρ) 706-726, block sups 728-743, θ 778-783, R 787-800, `Maj` 806-851, `mu_tail`/`Sig_tail` 861-873; used by `first_cell_law` 1118-1122 | scope 1 |
| 3. B2 Neumann remainder | on the seven Neumann cells, G(t) is invertible, and μ, Σ lie within T_mu, T_Sig of the same polynomials | `neumann_law` 1048-1116; `gram_tail` 882-885; the CB3 polynomials 617-676 and 1042-1046 | scope 1; the block sups of scope 2 (706-742) |
| 4. Envelope algebra | given the cell law, E[G_r] ≤ ⌈hi⌉ and E[G_r] ≥ lo.a | `ceil_hi` 1148-1198, `band_lo` 1200-1247; helpers 397-433, 888-998, 1132-1146 | scope 2 on the first cell, scope 3 elsewhere |

**Lanes.** The scopes can be reviewed separately, one lane per scope. Each section states the hypotheses it imports from the others and does not re-prove them. The sections below follow dependency order, 1 to 4.

**Outside all four scopes:**
- the definition of G_r and the reduction to it (README:16, README:91);
- CB0 and CB1, including CB1d;
- `iv_Phi` itself;
- the CB7 cross-check;
- trust in mpmath `iv` (README:88).

## Scope 1. CB2s: the pole-structure lemma and its use on the first cell

### 1.1 Notation
- **The field.** f is the centred Gaussian field with covariance K(x−y) = K1(x1−y1)·K1(x2−y2), where K1(s) = Z⁻¹ Σ_{k∈(π/12)ℤ} e^{−k²/2} e^{iks}. K and K1 are defined in the module docstring, cov_exact.py:9-13.
  - The covariance by multi-index is `cov`, cov_exact.py:417-431: Cov(∂^α f(x), ∂^β f(y)) = (−1)^{|β|} K1^{(α1+β1)}(dx) K1^{(α2+β2)}(dy), with (dx, dy) = x − y.
  - K1 is entire, because Σ_k e^{−k²/2} e^{|k|R} < ∞ for every R.
- **Kernel derivatives at 0.** κ(n) := K1^{(n)}(0). It is 0 for odd n and (−1)^{n/2} a_n for even n, where a_n = Z⁻¹ Σ e^{−k²/2} kⁿ are the torus moments. `A2[m]` encloses a_{2m} (comment 293); that is CB1, outside this scope.
- **z** is the real separation r. It also serves as the Laurent variable.
- **Stations.** M = 0 and S = z·e_x (comment 150-151). `DISP[(a,b)]` (541) is x_a − x_b in units of z·e_x.
- **Table terms.** A term (st, α, w, k) (154-169) is the random variable w·z^k·∂^α f(x_st). A functional is a list of terms.
- **Formal space 𝓛.** 𝓛 is the ℚ-span of the formal symbols ∂^β (β ∈ ℕ²). Each symbol stands for ∂^β f(0).
- **Bilinear form B.** B(∂^β, ∂^γ) = (−1)^{|γ|} κ(β1+γ1) κ(β2+γ2), which is Cov(∂^β f(0), ∂^γ f(0)). B is symmetric: a nonzero value needs both index sums even, so |β| ≡ |γ| mod 2.
- **Formal Laurent coefficients F^{(p)} ∈ 𝓛.** An M-term adds w·∂^α at p = k. An S-term adds (w/q!)·∂^{α+(q,0)} at p = k+q, for each q ≥ 0 (Taylor expansion in x1). This is `laurent_expand` (173-183), which is complete for every p ≤ `top`.
- **The tables.**
  - V = (V0..V5) are the pins, `PINS_EXACT` (154-161).
  - Y = (Y0..Y5) = (q_M, q_S, a_M, a_S, b_M, b_S) are the soft variables, `SOFT_EXACT` (162-169).
  - zV = (6/5, 0, 0, 0, 0, −1/6), `ZV_EXACT` (170), is the conditioning value, a constant. **zV is a name, not z·V.**
- **Gram series.** G = Cov(V,V), A = Cov(Y,V) and GY = Cov(Y,Y) are `GP_ser` (551), `GYV_ser` (553) and `GY_ser` (552). Their Laurent coefficients are G_p, A_p and GY_p.
- **Derived series.**
  - H = G⁻¹ = Σ H_j z^j, w = H·zV = Σ W_j z^j, μ = A·w = Σ X_k z^k, and Σ = GY − A·H·Aᵀ = Σ D_k z^k (624-664).
  - `mu_ser` = X[0..J] (648) and `Sig_ser` = D[0..J] (664), with J = 60 and NK = J+2 (455-456).
- 𝔼 denotes expectation; E denotes the matrix below.

### 1.2 The tables, expanded (each line can be checked by hand)
- **Pins.**
  - V0 = ∂00, V1 = ∂10 and V2 = ∂01.
  - V3 = (S10−M10)/z = ∂20 + (z/2)∂30 + O(z²), and V4 = (S01−M01)/z = ∂11 + (z/2)∂21 + O(z²).
  - V5 = (S00−M00)/z³ − (S10+M10)/(2z²). Its two parts are (S00−M00)/z³ = z⁻²∂10 + z⁻¹∂20/2 + ∂30/6 + z∂40/24 + … and −(S10+M10)/(2z²) = −z⁻²∂10 − z⁻¹∂20/2 − ∂30/4 − z∂40/12 − …. So V5 = −∂30/12 − (z/24)∂40 + …, and the z⁻² and z⁻¹ terms cancel.
- **Soft variables.** Y0 = ∂02 and Y1 = ∂02 + z∂12 + …; Y2 = z⁻¹∂20 and Y3 = z⁻¹∂20 + ∂30 + …; Y4 = z⁻¹∂11 and Y5 = z⁻¹∂11 + ∂21 + ….
- **Hence:**
  - V^{(0)} = (∂00, ∂10, ∂01, ∂20, ∂11, −∂30/12). These are linearly independent in 𝓛, so E is unique. The proof does not need uniqueness: any E satisfying (c) and (d) works.
  - Y^{(−1)} = (0, 0, ∂20, ∂20, ∂11, ∂11).
  - E has rows (0, 0, e3, e3, e4, e4), with columns 0-indexed; rows 0 and 1 are zero.
  - E·zV = (0, 0, zV3, zV3, zV4, zV4) = 0.

`h3_band_v2.py structure` prints exactly this E (1455-1456). A re-run on the final program printed `E = [0 0 0 0 0 0] [0 0 0 0 0 0] [0 0 0 1 0 0] [0 0 0 1 0 0] [0 0 0 0 1 0] [0 0 0 0 1 0]`.

### 1.3 Exact statement
**Hypotheses.** `structure_lemma` (212-237) decides each of them in `Fraction` arithmetic, on the tables that `mutated_tables()` returns (66-82). It calls `laurent_expand(·, 1)`, which is complete for p ≤ 1.
- **(a)** V_k^{(p)} = 0 for every p < 0 (223-225).
- **(b)** Y_a^{(p)} = 0 for every p < −1 (226-228).
- **(c)** Y_a^{(−1)} = Σ_k E_{ak} V_k^{(0)} in 𝓛, for an exact E ∈ ℚ^{6×6} (229-234).
  - `solve_in_span` (186-209) is exact Gauss–Jordan elimination. It returns None if and only if a row of the form 0 = c with c ≠ 0 is left (204-205). Otherwise it returns the solution with the free variables set to 0.
  - The returned E is not substituted back at run time, so its correctness rests on this short routine.
  - Two things cover it. An independent substitution (pre-checks, scope 1) confirms E. Exactness rests on that rational identity. The numeric check at 691-695 (observed |A_{−1} − E·G0| = 8.9e−100, threshold 1e−80) is corroboration only: it catches a grossly wrong E, but a perturbation of E below about 1e−80 that still satisfies E·zV = 0 would pass it (scope-1 review 5607220361).
- **(d)** E·zV = 0 exactly, row by row (235-236).

**Extra hypothesis (e): G_0 is invertible.** This is not part of CB2s; CB3 proves it at 612-614.
- The check bounds every entry of the iv product G0·Ginv0 − I by 1e−70; observed 8.26e−98.
- So ‖GX−I‖_∞ ≤ 6e−70 < 1 for every real G in the box G0 and every real X in the box Ginv0, and every such G is invertible.
- The true G_0 is among them provided it lies in the box G0. That rests on Step 1 below and on CB1's moment enclosures.
- If a pivot interval in `iv_solve6` contains 0, mpmath iv division returns an unbounded interval (checked under mpmath 1.4.1): [−inf, inf] when 0 is interior or the divisor is [0, 0], and a half-line such as [1/b, inf] when 0 is an endpoint. Either way `ub` of an affected entry is inf, and the check fails closed.

**Conclusions.**
- **(i)** G_p = 0 for p < 0; A_p = 0 for p ≤ −2; GY_p = 0 for p ≤ −3. Also A_{−1} = E·G_0 and GY_{−2} = E·G_0·Eᵀ.
- **(ii)** Assuming (e), μ and Σ are analytic at 0: X_{−1} = 0 and D_{−2} = D_{−1} = 0. The result for μ needs (a)-(e); the result for Σ needs only (a)-(c) and (e).

### 1.4 Proof
**Step 1: the formal expansion and the Gram series agree.** For one pair of terms, Cov(∂^α f(x_a), ∂^β f(x_b)) = (−1)^{|β|} K1^{(α1+β1)}(d·z)·κ(α2+β2), where d = DISP ∈ {−1, 0, 1}. In the code this is `cov_ser` (520-534). Since K1 is entire, K1^{(n)}(dz) = Σ_q κ(n+q)(dz)^q/q!. `kernel_ser` (500-511) stores coefficient j as κ(n+j)/j!, skips odd n+j, and flips the sign of odd j when d = −1. The three cases:
- **(S,M):** the term is Σ_q (z^q/q!)·B(∂^{α+(q,0)}, ∂^β).
- **(M,S):** the term is Σ_q (z^q/q!)·B(∂^α, ∂^{β+(q,0)}), because the (−1)^q comes from |β+(q,0)|.
- **(S,S) or (M,M):** d = 0, and the term is the constant B(∂^α, ∂^β). The formal double sum gives the same value, since Σ_{q+q'=N} C(N,q')(−1)^{q'} = 0 for N ≥ 1.

Multiply by w_a z^{k_a} · w_b z^{k_b} and sum over term pairs (`gram_entry_ser`, 543-548). For every pair of functionals this gives Cov(F_a, F_b) = Σ_p z^p Σ_{i+j=p} B(F_a^{(i)}, F_b^{(j)}), which converges absolutely for z ≠ 0. By bilinearity, the same holds for any rational combination of functionals.

**Step 2: conclusion (i).**
- G_p = Σ_{i+j=p} B(V^{(i)}, V^{(j)}) is 0 for p < 0, by (a). In particular G_0 = B(V^{(0)}, V^{(0)}).
- In A_p, (b) forces i ≥ −1 and (a) forces j ≥ 0, so A_p = 0 for p ≤ −2. Then A_{−1} = B(Y^{(−1)}, V^{(0)}) = E·B(V^{(0)}, V^{(0)}) = E·G_0, by (c).
- In the same way, GY_p = 0 for p < −2, and GY_{−2} = B(Y^{(−1)}, Y^{(−1)}) = E·G_0·Eᵀ.

**Step 3: conclusion (ii).** Let Ỹ := Y − E·V/z. Then Ỹ^{(p)} = Y^{(p)} − E·V^{(p+1)}. For p ≤ −2 this is 0 by (b) and (a); for p = −1 it is 0 by (c). Applying Step 1 to Ỹ, both Ã := Cov(Ỹ, V) = A − E·G/z and GỸ := Cov(Ỹ, Ỹ) have no negative powers. G is entire with no negative orders, so by (e) H = G⁻¹ is analytic near 0.
- **For μ:** μ = (Ã + E·G/z)·H·zV = Ã·H·zV + E·zV/z = Ã·H·zV, by (d).
- **For Σ:** using GH = HG = I and G = Gᵀ, GY = GỸ + (Ã·Eᵀ + E·Ãᵀ)/z + E·G·Eᵀ/z² and A·H·Aᵀ = Ã·H·Ãᵀ + (Ã·Eᵀ + E·Ãᵀ)/z + E·G·Eᵀ/z². So Σ = GỸ − Ã·H·Ãᵀ.

Both right-hand sides are analytic at 0. Laurent coefficients on a punctured disc are unique, so X_{−1} = 0 and D_{−2} = D_{−1} = 0. In coefficients, these are the quantities the code evaluates:
- X_{−1} = A_{−1}·W_0 = E·G_0·G_0⁻¹·zV = E·zV = 0. This is X[−1] at 642-647, where k = −1 admits only i = −1.
- D_{−2} = GY_{−2} − A_{−1}·H_0·A_{−1}ᵀ = E·G_0·Eᵀ − E·G_0·Eᵀ = 0. D[−2] and D[−1] are computed at 658-663.
- D_{−1} involves G_1 through H_1 = −G_0⁻¹G_1G_0⁻¹. G_1 ≠ 0 for these tables, so this cancellation is not trivial.

This completes the proof.

**Probabilistic reading.** For a centred Gaussian, 𝔼[Y | V = v] = A·G⁻¹·v and Cov(Y | V) = GY − A·G⁻¹·Aᵀ; this is standard conditioning. The docstring (218-220) states the resulting conclusion: 𝔼[Y | V = zV] = 𝔼[Ỹ | V = zV] + E·zV/z has no pole, and Cov(Y | V) = Cov(Ỹ | V).

### 1.5 The lemma is about the objects the series use
- **Same objects.** At 265, `run_certificate` takes `pins_exact`, `soft_exact` and `zv_exact` from `mutated_tables()` and passes them to `structure_lemma` (268). From those same lists it builds `PinsV` and `SoftY` (539-540, through `to_iv_func` 536-537: the power is kept, and the weight becomes an iv enclosure of the rational) and `zV` (609). `E_sel` is used only at 694.
- **Same geometry.** `laurent_expand` places S at z·e_x (175, 181-182). DISP[('S','M')] = +1 (541) means x_S − x_M = +z·e_x, which matches `cov`'s convention (dx, dy) = x − y.
- **Complete kernel window.**
  - For a cross-station pair of terms, an order-p coefficient needs kernel order j = p − k_a − k_b, and `kernel_ser` keeps j ≤ J+8−n1 (502).
  - Over all cross-station term pairs in the tables, the maximum of n1 − k_a − k_b is 6. It is attained by V5×V5, V5×Y2, V5×Y3 and Y2×Y3. So every coefficient with p ≤ NK = J+2 is complete (comment 497-499).
  - The lowest orders present are −6 (GP), −4 (GYV) and −2 (GY). All are ≥ LMIN = −8 (457), so `ser_trim` (493-495) removes nothing at the bottom. Above NK it removes incomplete orders, which the Cauchy tails replace.
- **Same as the frozen tables.** The tables equal the frozen PinsV, SoftY and zV of both ceil.txt and floor.txt (lines 172-188 and 237), compared programmatically.
- **CB7 is independent.** CB7 re-encodes the functionals by hand (1281-1305), separately from the tables, with M at −r/2 and S at +r/2; the covariance is translation-invariant. It is a cross-check at r = 1/20 against the Neumann law, not load-bearing.

### 1.6 How the conclusions are used
**C1: every cell, using (i), that is (a) and (b).**
- For the law, the code reads only GP orders 0..NK (608, 617), GYV orders −1..NK (619-620) and GY orders −2..NK (618, 621-622).
- The GP orders {−6, −4, −3, −2, −1} and the GYV orders {−4, −3, −2} are computed (|c| ≤ 6.3e−100), checked at 555-567, and discarded, because their true values are 0.
- The recursions for H and W (624-634) and θ(R), which sums from i = 1 (778-783), solve G·H = I order by order. They are valid only because G has no negative orders.
- The Neumann law uses G_P from order 0 (1067) and A_P = A_{−1}/t + Σ_{j≥0} A_j t^j (1094-1095).

**C2: the first cell only, using (ii).**
- `first_cell_law` (1118-1122) returns, entrywise, μ = Σ_{k=0}^{J} X_k t^k ± MT and Σ = Σ_{k=0}^{J} D_k t^k ± ST on t ∈ [0, up(1/16384)] = [0, 2⁻¹⁴]. It omits X_{−1}/t, D_{−2}/t² and D_{−1}/t.
- t can be arbitrarily small on (0, 1/16384], so any nonzero pole coefficient, however small, would make these terms unbounded. The omission is valid only because the pole coefficients are exactly 0.
- The heavy coefficients X_k and D_k for k ≤ J are complete: they need A up to J+1, W up to J+1 and H up to J+2 = NK.
- MT and ST (861-878) bound only the orders k > J. They include the pole-index terms A_{−1}·W_{k+1} and so on (864, 871-872).
- The floor's rotated law (1205-1206) is LR applied to the same series (681-683), so its pole coefficients LR·X_{−1} and so on also vanish.

**C3: the Neumann cells do not use (c) or (d).** They add |X[−1]|/r_lo and |D[−2]|/r_lo² + |D[−1]|/r_lo (1099, 1108-1109), with r_lo > 0 checked at 1063. They do rely on C1.

### 1.7 What the program checks at run time
- **(a)-(d), exactly**, at 268, failing closed through `ck` (101-104).
  - There are four controls (72-79), tested at test:80-89 (structure mode) and test:371-374 (full runs). Each exits 1 at its own branch; `structure_window` fails at row 2.
  - In `structure` mode, a mutation the lemma cannot see now ends `H3_BAND_V2 MUTATED STRUCTURE RUN (…): the lemma holds for these tables; NOT A CERTIFICATE` and exits 3 (1451-1454; test:91-98). Such mutations are `wrong_window` and those that do not touch the tables. For `wrong_window` the lemma really does hold on the mutated tables, because column 5 of E is zero.
- **(e):** the G0 inverse residual is below 1e−70 (612-614); observed 8.26e−98.
- **Gram pole pattern (555-567).** Every coefficient that `ser_trim` kept (orders LMIN..NK) and that the law then discards has |enclosure| ≤ 1e−60; observed ≤ 6.3e−100. `ser_trim` would remove any order below LMIN = −8 before this check, so the check never sees such orders. The tables have none (lowest −6), and the exact lemma runs first.
- **Law pole enclosures:** |X[−1]| < 1e−55 (687), and |D[−2]|, |D[−1]| < 1e−50 (690). Observed: 3.4e−99, 8.3e−99 and 1.0e−98. Each enclosure contains 0.
- **A_{−1} = E·G0 entrywise within 1e−80 (691-695).** This ties the exact E to the interval series; observed 8.9e−100. Conclusion (i)'s GY_{−2} = E·G_0·Eᵀ is not checked at run time, because the law does not need it; a scratch run observed 8.9e−100.
- The program has no run-time cross-check of its own for the first-cell law. CB7 runs at r = 1/20 only.

### 1.8 What rests only on the derivation
1. **Step 1 (500-548).** The interval Gram Laurent coefficients equal B applied to `laurent_expand`'s formal coefficients. This rests on four things: K1 being entire, the DISP sign convention (541), the extra (−1)^q in the (M,S) case, and the cancellation of the S-S double sum. At run time the evidence is only numeric: the pole pattern (555-567), A_{−1} = E·G0 to 1e−80 (691-695), the CB3 limit identities (697-702), and CB7 at r = 1/20.
2. **Steps 2-3.** The algebra by which (a)-(d) plus (e) imply (i) and (ii), through Ỹ = Y − E·V/z, is not checked anywhere. The numeric checks at 555-567, 687 and 690 agree with it, but they cannot tell 0 from a coefficient below 1e−60, 1e−55 or 1e−50 respectively.
3. **That every discarded coefficient is exactly zero.** This covers GP orders < 0 and GYV orders ≤ −2 in every cell, since the law reads only orders 0..NK, −1..NK and −2..NK (608, 617-622). It also covers X_{−1}, D_{−2} and D_{−1} in the first cell (`first_cell_law`, 1118-1122). There, any nonzero pole coefficient would make the enclosure invalid for small enough t.
4. **That (c) holds for the E that `solve_in_span` returns.** E is not substituted back at run time; 691-695 corroborates it numerically.
5. **That the H and W recursions (624-634) and θ(R) (778-783, summing from i = 1) are the true order-by-order inverse.** This needs G to have no negative orders, which is lemma (a).
6. **That the kernel window is complete through order NK** (comment 497-499, range at 502). The maximum of n1 − k_a − k_b over cross-station pairs is 6. No run-time check would detect a missing kernel term at orders near NK.
7. **That the true G_0 lies in the box G0**, so that the residual check at 612-614 proves (e). This depends on Step 1 and on CB1's moment enclosures.
8. **That μ = A·G⁻¹·zV and Σ = GY − A·G⁻¹·Aᵀ are the conditional law of Y given V = zV.** This is centred-Gaussian conditioning with zV a constant vector, as in CB7's zVd (1301).
9. **That the tables (154-170) are the upstream corrected pins, soft variables and conditioning value that define G_r.** README:16 and README:91 say this is not re-derived. CB3's limit identities (697-702) and CB7's hand-coded functionals (1284-1289) are only partial evidence.
10. **That the first-cell Taylor series of μ and Σ converge to the true law on (0, 1/16384], with tails MT and ST (857-878).** That is scope 2; this lemma supplies only the absence of negative orders.

## Scope 2. CB4: the R-majorant induction and the first-cell tails

This scope covers 705-885 and the places where the first cell uses those results: 1118-1122, 1256-1258, 1264 and 1269, plus the 2MT/4ST widening at 1205-1206.

### 2.1 Notation
- **Series variable.** The program identifies the series variable z with the cell variable r. `first_cell_law` builds t_iv = [0, r*] (1119), and the cell loop passes t_iv as r to `ceil_hi` and `band_lo` (1257, 1264, 1269). Here r* = R_STAR = 1/16384 = 2⁻¹⁴ (857), and t ranges over (0, r*].
- **Orders.** J = 60 and NK = J+2 = 62 (455-456).
- **The circle.** ρ = RHO = mpf('0.11') at dps 100 (458).
  - RHO is a dyadic rational, and RHO − 0.11 ≈ 2.9e−103. Every CB4 use reads this one mpf, so the Cauchy circle is exactly |z| = RHO.
  - QRHO = up(1/RHO) ≥ 1/RHO (769-770).
- **Gram blocks**, all 6×6: G(z) is the pin Gram, A(z) the soft–pin cross Gram, and GY(z) the soft Gram.
  - Their Laurent ranges, G = Σ_{j≥0} G_j z^j, A = Σ_{j≥−1} A_j z^j and GY = Σ_{j≥−2} GY_j z^j, are imported from scope 1, (a) and (b) (see H1).
  - The program's enclosures are `Gj[j]`, `Aj[j]` and `GYj[j]` for 0 ≤ j ≤ NK, plus `A_m1`, `GY_m1` and `GY_m2` (617-622).
  - `mat_coeff` (571-572) reads only these orders, so the program drops the numeric coefficients at every other order.
  - `Ac(i)` returns A_{−1} for i = −1 and A_i otherwise (636-637).
- **Inverse and w.** H(z) = G(z)⁻¹ = Σ_{j≥0} H_j z^j and w(z) = H(z) zV = Σ W_j z^j. zV is ZV_EXACT = (6/5, 0, 0, 0, 0, −1/6) (170); the program's `zV` is its interval enclosure (609).
- **The law.** μ(z) = A(z) w(z) and Σ(z) = GY(z) − A(z) H(z) A(z)ᵀ.
  - Laurent coefficients: μ_k = Σ_{i=−1}^{k} A_i W_{k−i}, and Σ_k = GY_k − Σ_{a+b+c=k} A_a H_b A_cᵀ with a, c ≥ −1 and b ≥ 0.
  - Enclosures: X[k] and D[k] (641-664). `mu_ser` = X[0..J] (648) and `Sig_ser` = D[0..J] (664).
- **Norms and helpers.** ‖P‖_max = max_{a,b} |P_ab| and ‖v‖_max = max_a |v_a|. `ub(x)` = max(|lo|, |hi|) (241-245) and `up(x)` is the upper endpoint (248-250). `matmax` and `vecmax` return the maximum of `ub` over the entries (745-753).

### 2.2 Exact claim
For every t ∈ (0, r*] and all indices a, b:

  μ_a(t) ∈ Horner(X[0..J]_a, [0, r*]) + [−MT, MT],  Σ_ab(t) ∈ Horner(D[0..J]_ab, [0, r*]) + [−ST, ST].

- MT = `mu_tail(tS)` and ST = `Sig_tail(tS)` (875-876). Horner is interval Horner (1001-1008), applied at 1118-1122.
- The transcript prints MT = 1.04e−44 and ST = 1.84e−38, as 3-digit round-to-nearest `nstr` values. The certificate uses the full mpf upper endpoints, 1.03890e−44 and 1.83842e−38.
- G(t) is invertible for 0 ≤ t < 1/R, which contains [0, r*].

### 2.3 Hypotheses imported from other scopes
- **H1 (scope 1, exact in `Fraction`; 212-237, run at 268 before any series is built).** It proves four facts:
  - (a) the pins have no negative powers, so G(z) is a power series;
  - (b) the soft functionals have no power below z⁻¹, so A has at most a z⁻¹ pole and GY at most a z⁻² pole;
  - (c) Y^{(−1)} = E V^{(0)} for an exact matrix E;
  - (d) E zV = 0.

  How CB4 uses them. (a) and (b) fix the index ranges: the H and W recursions with G0 as leading coefficient, the θ sum from i = 1, i ≥ −1 in μ_k, and the a, c ≥ −1 pole groups in Σ_k. (c) and (d) give μ_{−1} = Σ_{−2} = Σ_{−1} = 0 exactly, which is why the first-cell law drops X[−1], D[−2] and D[−1].

  A direct algebraic check of the three cancellations, with C_pq = cov(V^{(p)}, V^{(q)}):
  - A_{−1} = cov(Y^{(−1)}, V^{(0)}) = E G0, hence μ_{−1} = A_{−1} G0⁻¹ zV = E zV = 0.
  - Σ_{−2} = GY_{−2} − A_{−1} G0⁻¹ A_{−1}ᵀ = E G0 Eᵀ − E G0 Eᵀ = 0.
  - A_0 = cov(Y^{(−1)}, V^{(1)}) + cov(Y^{(0)}, V^{(0)}) and H_1 = −G0⁻¹(C01 + C10)G0⁻¹. Then A_{−1}H_0A_0ᵀ + A_0H_0A_{−1}ᵀ + A_{−1}H_1A_{−1}ᵀ = GY_{−1}, so Σ_{−1} = 0.

  The docstring's route (218-220) gives the same result at all orders. Ỹ = Y − EV/z is regular by (a)-(c), Cov(Y|V) = Cov(Ỹ|V), and E[Y|V=zV] = E[Ỹ|V=zV] + E zV/z, where the last term vanishes because E zV = 0. The numeric checks at 555-567 (Gram pole pattern), 687-690 (poles of μ and Σ tiny) and 692-695 (A_{−1} = E G0) are consistency checks only.
- **H2 (CB2 window).** The Gram enclosures through order NK enclose the true coefficients. The kernel series uses `range(0, J+9-n)` (502), argued only in the comment at 497-499.
  - Only cross-station pairs (d = ±1) use the truncated kernel series; same-station pairs are constants (525-528).
  - A cross pair reaches order k through kernel index j = k − k_a − k_b, and `kernel_ser(n1)` keeps j ≤ J+8−n1. So order NK = J+2 is complete exactly when (−k_a + α_{a,x}) + (−k_b + α_{b,x}) ≤ 6.
  - The per-term weight −k + α_x is at most 3 in the tables (154-169); pin V5's four terms and soft Y2 and Y3 reach 3. So the condition holds.
  - No run-time check ties the window to the tables. A mutation that would exceed it (`structure_soft_pole2`, weight 4) is stopped by CB2s (b) before CB2 runs.
- **H3 (CB3).** `Ginv0` encloses G0⁻¹ (610-614). H[j] and W[j] for j ≤ NK enclose the true H_j and W_j (624-634). X[k] and D[k] for k ≤ J are complete: they use W only up to k+1 ≤ NK and H only up to k+2 ≤ NK (641-663).
- **H4 (covariance formula).** Each Gram entry is f(z) = Σ_pairs w_a w_b z^{k_a+k_b} (−1)^{|β|} K1^{(n1)}(dz) K1^{(n2)}(0), with d ∈ {−1, 0, 1} (520-548). Here K1(x) = Z⁻¹ Σ_{k∈(π/12)ℤ} e^{−k²/2} cos(kx), Z is the full lattice sum, and K1 is entire. The program evaluates the Gram through the Poisson moment table A2, which CB1d cross-checks against the spectral form.
- **H5.** mpmath `iv` rounds outward correctly.

### 2.4 Derivation
1. **Kernel sup (706-726).** For complex w with |w| ≤ ρ, K1^{(n)}(w) = Z⁻¹ Σ e^{−k²/2} (ik)^n e^{ikw}, and |e^{ikw}| ≤ e^{|k||Im w|} ≤ e^{|k|ρ}. Hence |K1^{(n)}(w)| ≤ A_n(ρ) := Z⁻¹ Σ e^{−k²/2} |k|^n e^{|k|ρ}.
   - **Lattice part:** |j| ≤ Jl = 116, where Jl = ceil(30/(π/12)) + 1 (cov_exact.py:152). |k| is replaced by its upper endpoint, and k = 0 contributes 1 because iv 0**0 = 1 (716-717).
   - **Tail:** |j| ≥ j0 = Jl+1, where k0 = j0·h ≈ 30.63. For |k| ≥ 4ρ, −k²/2 + |k|ρ ≤ −k²/4. Put u_j = (jh)^n e^{−(jh)²/4}; then u_{j+1}/u_j = e^{−(2j+1)h²/4}(1+1/j)^n. This ratio decreases in j, so for j ≥ j0 it is at most q (721). The tail is therefore at most 2u_{j0}/(1−q) (723); the factor 2 covers both signs of k.
   - **Normalization:** the program divides by Z1_iv, an interval enclosing the truncated lattice sum. Its lower endpoint is at most the exact truncated sum, which is at most Z. So the upper endpoint of the interval quotient bounds the quotient by Z (724). Only up(A_rho[n]) is consumed downstream (scope-2 review 5607205948).
   - A_rho[0..8] is computed (726), but only n1, n2 ≤ 4 occur: n1 = 4 from soft (2,0)+(2,0), and n2 = 4 from soft (0,2)+(0,2).
2. **Block sup and Cauchy bound (728-743).**
   - On |z| = ρ the triangle inequality gives |f(z)| ≤ Σ_pairs |w_a||w_b| ρ^{k_a+k_b} A_{n1}(ρ) A_{n2}(ρ). This uses |K1^{(n)}(0)| ≤ A_n(ρ), which also covers the d = 0 pairs.
   - The block constants are the maxima over the 36 entries: M_GP = 2.9579e6, M_GY = 343.659 and M_GYV = 19473.5.
   - Each f is a finite sum of Laurent monomials times entire functions, so it is analytic on ℂ\{0}. Cauchy's estimate on |z| = ρ then gives |f_j| ≤ M ρ^{−j} ≤ M·QRHO^j for every j ∈ ℤ, negative j included.
3. **Majorant sequences (755-761).**
   - g_i = G_m[i] for 1 ≤ i ≤ J, and M_GP ρ^{−i} for i > J. θ uses the Cauchy bound even at J+1 and J+2, where computed values exist; that is valid and only conservative.
   - a_i = A_m[i] for 0 ≤ i ≤ NK, and M_GYV·QRHO^i beyond. y_i = GY_m[i] for 0 ≤ i ≤ NK, and M_GY·QRHO^i beyond.
   - α = A1_m ≥ ‖A_{−1}‖_max (α = 3), and γ = Ginv0_m ≥ ‖G0⁻¹‖_max (γ = 24).
   - By H2, H3 and step 2, each sequence dominates the true ‖·‖_max. The B1 guard (762-767) checks matmax ≥ ub(entry); in unmutated code this holds by definition.
4. **The constant 36 and the induction.** For 6×6 matrices, ‖PQ‖_max ≤ 6‖P‖_max‖Q‖_max and ‖Pv‖_max ≤ 6‖P‖_max‖v‖_max.
   - **Recursions.** From GH = I, H_j = −G0⁻¹ Σ_{i=1}^{j} G_i H_{j−i} for j ≥ 1 (628-632). From Gw = zV, W_j = −G0⁻¹ Σ_{i=1}^{j} G_i W_{j−i} (633-634).
   - **Where 36 comes from.** Put h_j := ‖H_j‖_max. The first 6 is for G0⁻¹ and the second for G_i H_{j−i}: h_j ≤ 6γ Σ_i 6 g_i h_{j−i} = 36γ Σ_{i=1}^{j} g_i h_{j−i}. The same holds for w_j := ‖W_j‖_max.
   - **θ.** θ(R) = 36γ [Σ_{i=1}^{J} G_m[i] R^{−i} + M_GP x^{J+1}/(1−x)], with x = 1/(ρR) < 1. This is exactly `theta` (778-783).
   - **Constants.** C_H = max(1, max_{j≤NK} up(H_m[j]/R^j)), and C_W is defined the same way from W_m (802-804).
   - **Induction hypothesis:** h_j ≤ C_H R^j and w_j ≤ C_W R^j for every j ≥ 0.
   - **Base, j ≤ NK:** h_j ≤ H_m[j] ≤ C_H R^j. The second inequality holds because the iv quotient contains H_m[j]/R^j.
   - **Step, j > NK (strong induction):** h_j ≤ 36γ Σ_{i=1}^{j} g_i C_H R^{j−i} ≤ C_H R^j θ(R). The same bound holds for W.
   - So **θ(R) ≤ 1 suffices.** The program checks the stronger θ(R) ≤ 1/2, together with ρR > 1, in interval arithmetic at 800. The 1/2 is inherited from ceil.txt:364-381.
   - **Search.** The search (787-799) updates R in round-to-nearest mp. R is just an exact mpf, and only the iv check at 800 on the final R is load-bearing.
   - **Consequence.** Σ H_j t^j converges absolutely for |t| < 1/R. Its Cauchy product with G(t) is I, and G is entire. So G(t) is invertible for 0 ≤ t < 1/R, with G(t)⁻¹ = H(t) and w(t) = G(t)⁻¹ zV. The first cell has no other invertibility argument.
5. **The numbers.**
   - γ = 24, G_m[1] = 1.5 and α = 3. The leading term of θ is 36·24·1.5/R = 1296/R.
   - The program finds R = 2622.3492, ρR = 288.46 and θ = 0.49437, of which the leading term is 0.49421.
   - The smallest R with θ ≤ 1 is 1296.83, and with θ ≤ 1/2 it is 2592.83. The 2% grid then returns 2622.35.
   - C_W = 4 = W_m[0] and C_H = 24 = H_m[0]; both maxima are attained at j = 0. The next-largest ratios are 3.4e−4 for W and 1.1e−3 for H.
6. **`Maj`, P2 and P3 (806-851).**
   - A `Maj` sequence takes the computed values c_j for j ≤ N = NK, and C q^j beyond.
   - The suffix sums T[m] = Σ_{j≥m} (term)·t^j are sums of nonnegative terms, with T[N+1] = C(qt)^{N+1}/(1−qt). The condition qt < 1 is checked at 824 and 837.
   - P2(A, B, m, t) = Σ_{i+l≥m} a_i b_l t^{i+l}; P3 is the triple analogue.
   - The instances are majW = (W_m, C_W, R), majH = (H_m, C_H, R), majA = (A_m, M_GYV, QRHO) and majGY = (GY_m, M_GY, QRHO) (853-856).
7. **μ tail (861-865).** For k > J, split off the i = −1 term. Entrywise, |Σ_{k>J} μ_{k,a} t^k| ≤ 6 P2(majA, majW, J+1, t) + 6α Σ_{l≥J+2} w_l t^{l−1}. The second term is `majW.tail_from(J+2, t)/t`.
8. **Σ tail (867-873).** Each entry of A_a H_b A_cᵀ has at most 36 terms. Entrywise, for k > J, there are four groups:
   - the GY part, `majGY.tail_from(J+1)`;
   - a, c ≥ 0: 36·P3(majA, majH, majA, J+1);
   - exactly one of a, c equal to −1. Then b + c = k+1 ≥ J+2 (or a + b = k+1 ≥ J+2 for the mirror term). The two groups have equal sums, giving 2·36·α·P2(majH, majA, J+2)/t;
   - a = c = −1. Then b = k+2 ≥ J+3, giving 36α²·`majH.tail_from(J+3)`/t².
9. **Monotone in t.** After the divisions by t and t², every bound in steps 7 and 8 is a power series in t with nonnegative coefficients and exponents ≥ J+1. Its value at tS = 2⁻¹⁴ (859) therefore bounds it at every t ∈ (0, r*].
10. **First-cell guard (858).** The program checks R·r* < 0.2; the actual value is 0.16005.
    - Convergence itself needs only R·r* < 1 and QRHO·r* < 1. R·r* < 1 follows from 858 and is checked again in `tails` for majW and majH. QRHO·r* < 1 is checked only in `tails`, for majA and majGY (824, 837).
    - The 0.2 is a margin inherited from the frozen program (ceil.txt:383).
    - The frozen first cell (0, 1/400] gives R/400 ≈ 6.56 > 1, so the majorant genuinely diverges there. This is the `wide_first_cell` control (README:48 and README:128; test:378).
11. **Enclosure (1118-1122).**
    - μ_a(t) = Σ_{k=0}^{J} μ_{k,a} t^k + (tail), with no k = −1 term by H1. Each μ_{k,a} lies in X[k]_a (H3), and t ∈ [0, r*]. So Horner encloses the polynomial, and ±MT covers the tail. The same argument applies to Σ with ST.
    - `band_lo` widens the rotated law by 2MT and 4ST (1205-1206). This is valid because each row of LR (680) has ℓ¹ norm at most 2: a rotated-vector tail is ≤ 2MT and a rotated-matrix tail ≤ 2·2·ST.
    - The Neumann law adds numeric pole terms (1099, 1108-1109). The first cell cannot, because 1/t is unbounded there, so it rests on the exact H1.

### 2.5 Checked numerically at run time
- 720: k0 ≥ 4ρ. 722: the kernel tail ratio q < 1, for each n.
- 767: B1 dominance. This holds by definition in unmutated code; it is a regression guard for `signed_maxima`.
- 776: the Cauchy check, for the computed orders j ≤ NK, for j = −1 (A, GY) and for j = −2 (GY). It is not vacuous: GY_{−2} sits at 0.72 of its bound. The other blocks are loose (GY_0 at 0.022, GYV at 1.5e−4, GP at 1e−6), and it says nothing about j > NK.
- 793: θ(R) ≤ 1/2 after the upward search; not load-bearing.
- 800: ρR > 1 and θ(R) ≤ 1/2 for the final R, in iv; load-bearing.
- 824 and 837: qt < 1 for every `Maj` used.
- 858: R·r* < 0.2.
- 877-878: 0 ≤ ST < 1e−5 and 0 ≤ MT < 1e−5.
- Upstream, consistency checks only: 567 (Gram pole pattern), 614 (G0 inverse residual) and 687-695 (pole consistency).
- **Rounding.** Every load-bearing constant is an iv bound. A_rho, M_*, θ, C_W, C_H, QRHO, MT and ST are upper endpoints; G_m, GY_m, A_m, H_m, W_m, Ginv0_m and A1_m are ub (max |endpoint|). R is an exact mpf, and any value of it is acceptable once 800 passes.

### 2.6 What rests only on the written derivation
The two items that carry the most weight are the Cauchy estimate beyond the computed orders (j > NK in majA and majGY, and i > J in θ) and the induction step.
- |K1^{(n)}(w)| ≤ A_n(ρ) for complex |w| ≤ ρ, from the spectral form of K1 and |e^{ikw}| ≤ e^{|k|ρ} (706-717).
- **The kernel lattice tail (718-723).** Only k0 ≥ 4ρ and q < 1 are checked. The rest is derivation:
  - |k| ≥ 4ρ implies e^{−k²/2+|k|ρ} ≤ e^{−k²/4};
  - the term ratio is monotone in j, so it is bounded by q;
  - the tail is ≤ 2u_{j0}/(1−q).
- Dividing by Z1_iv gives an upper bound, because the lower endpoint of Z1_iv is at most the exact truncated sum, which is at most Z (724).
- The triangle-inequality sup bound of each Gram entry on |z| = ρ, including |K1^{(n)}(0)| ≤ A_n(ρ) (728-737).
- **Analyticity and Cauchy's estimate.** Every Gram entry is analytic on ℂ\{0}, so Cauchy's estimate |f_j| ≤ M ρ^{−j} holds for every j. This is load-bearing for j > NK in majA and majGY and for i > J inside θ. The run-time check at 772-776 covers only the computed orders.
- **H1 (a), (b):** the Laurent ranges of G, A and GY. The program drops the numeric coefficients outside them, with only the consistency check at 555-567.
- **H1 (c), (d) ⇒ μ_{−1} = Σ_{−2} = Σ_{−1} = 0.** CB2s itself is exact in `Fraction`. The step from (a)-(d) to the cancellations is the algebra in H1 above, or equivalently the docstring at 218-220. `first_cell_law` relies on it with no numeric pole term (1118-1122).
- **H2:** the kernel window `range(0, J+9-n)` makes every Gram coefficient through NK complete (497-511). There is no run-time tie to the tables.
- **H3:** interval Gauss–Jordan and the series recursions enclose the true G0⁻¹, H_j, W_j, X[k] and D[k] (574-585, 610-664). The G0 residual check at 614 is a consistency check.
- The 6×6 max-norm product inequalities that give 36 = 6·6 in θ (783).
- **The induction.** θ(R) ≤ 1, the base j ≤ NK and the definitions of C_W and C_H imply ‖H_j‖_max ≤ C_H R^j and ‖W_j‖_max ≤ C_W R^j for all j. Only θ ≤ 1/2 is checked (800).
- G(t) is invertible, with G(t)⁻¹ = Σ H_j t^j, for 0 ≤ t < 1/R. This follows from absolute convergence and the Cauchy product; there is no separate first-cell check.
- The Laurent coefficient identities for μ_k and Σ_k, the split into the i = −1, a = −1 and c = −1 pole groups, and the index bounds J+1, J+2 and J+3 in `mu_tail` and `Sig_tail` (861-873).
- The closed form T[N+1] = C(qt)^{N+1}/(1−qt), and that P2 and P3 equal the stated sums over i+l ≥ m (819-851).
- Every tail bound is monotone in t, so evaluating it at t = 2⁻¹⁴ covers all of (0, r*] (859, 875-876).
- The 2MT/4ST widening of the rotated law in `band_lo` (1205-1206), which uses that each row of LR has ℓ¹ norm at most 2.

## Scope 3. B2: the Neumann remainder in `neumann_law` (1048-1116)

**Where it is used.** `neumann_law(c1, c2)` gives the law on cells 2 to 8 (1260). Two other calls are not load-bearing: one at the degenerate cell [1/20, 1/20] for the CB7 cross-check (1308), and, under `--mc` only, calls at 1/20 and 1/100 for the non-certifying Monte Carlo (1333).

**What it consumes:**
- helpers: `horner` 1001-1005, `ser_eval` 1007-1008, `mat_poly_eval` 1010-1016, `vec_poly_eval` 1018-1023, `frob_up` 1025-1026, `l2_up` 1028-1029, `mp_inverse6` 1031-1040, `gram_tail` 882-885, `ck` 101-104, and `ub`/`up`/`iv_frac` 241-255;
- CB3 data: 617-676, and `Xhi`, `Dhi`, `RWl`, `RHl`, `SQRT6` (1042-1046);
- CB4 constants: `M_GP`, `M_GY`, `M_GYV` (740-742) and `RHOi` (769). `NK` (456) fixes the tail order.

**What it does not use.**
- `mu_tail` and `Sig_tail` (861-873). They are evaluated once at 875-876, and their stored values `MT`, `ST` are used only by `first_cell_law` (1118-1122) on the first cell.
- `polys` (1132-1146). `ceil_hi` (1156) and `band_lo` (1212) call it later on the same `t_iv`.
- `H_m` (758). It appears only in the `old_remainder` control branch (1077-1087), which runs only under that mutation and is not load-bearing.

### 3.1 Notation
- J = 60 and NK = J+2 = 62 (455-456). ρ = `RHO` = mpf('0.11') (458). The same binary number is used for the M_* constants (through `iv.mpf(RHO)` in 712-737) and for `gram_tail`.
- z is the complex Laurent variable. On a cell, z = t = r is real, with t ∈ [c1, c2] and 0 < c1.
- V = (V_0..V_5) are the pin functionals (`PINS_EXACT`, 154-161), and Y = (q_M, q_S, a_M, a_S, b_M, b_S) are the soft functionals (`SOFT_EXACT`, 162-169).
- **zV** is the constant conditioning value `ZV_EXACT` = (6/5, 0, 0, 0, 0, −1/6) (170, 609). It means "the value of V", not z·V.
- **Gram blocks.** G(t) = Cov(V, V) is 6×6. A(t) = Cov(Y, V) is soft × pin (code `GYV`, 553, 619). GY(t) = Cov(Y, Y) (552).
- **True Laurent coefficients:** G_j (j ≥ 0), A_j (j ≥ −1), GY_j (j ≥ −2). Their enclosures for j ≤ NK are `Gj[j]`, `Aj[j]`, `A_m1`, `GYj[j]`, `GY_m1` and `GY_m2` (617-622).
- **Truncations:** G_P = Σ_{j=0}^{NK} G_j t^j, A_P = Σ_{j=−1}^{NK} A_j t^j and GY_P = Σ_{j=−2}^{NK} GY_j t^j. Then δG = G − G_P, δA = A − A_P and δGY = GY − GY_P.
- **Taylor points.** `W[j]` and `H[j]` (j ≤ NK) are the interval vectors and matrices from the recursion at 624-634.
  - Fix any points Ŵ_j ∈ W[j] and Ĥ_j ∈ H[j], once for the whole cell, and set W_P = Σ Ŵ_j t^j and H_P = Σ Ĥ_j t^j.
  - Nothing below needs W_P or H_P to be the Taylor polynomials of G⁻¹zV and G⁻¹; they only have to be polynomials. Being close to Taylor affects only how tight the bound is.
- w = G⁻¹zV, e_w = w − W_P and E_H = G⁻¹ − H_P. μ = A w and Σ = GY − A G⁻¹ Aᵀ (Gaussian conditioning, centred field).
- **Product polynomials:**
  - X(t) = A_P W_P = Σ_{k=−1}^{2NK} X_k t^k (`X`, 641-647);
  - D(t) = GY_P − A_P H_P A_Pᵀ = Σ_{k=−2}^{3NK} D_k t^k (`AH`, `D`, 650-663);
  - RW(t) = zV − G_P W_P and RH(t) = I − G_P H_P, each Σ_{k=0}^{2NK} (`RW`, `RH`, 666-676).
- **Index ranges, all complete:**
  - X (644): i ∈ [−1, NK] with l ∈ [0, NK];
  - AH (653): a ∈ [−1, NK] with b ∈ [0, NK];
  - D (660): c ∈ [−1, NK] with m ∈ [−1, 2NK], so every triple a ∈ [−1, NK], b ∈ [0, NK], c ∈ [−1, NK] appears;
  - RW and RH (671): i ∈ [0, NK] with l ∈ [0, NK].
- **Heavy parts:** μ_h = Σ_{k=0}^{J} X_k t^k (`mu_ser`, 648) and Σ_h = Σ_{k=0}^{J} D_k t^k (`Sig_ser`, 664).
- **Preconditioner.** B is the mpf point matrix ≈ (midpoint of the cell enclosure `GP_iv`)⁻¹ (1069-1071); under the `no_preconditioner` control, B = I. E(t) = I − B G(t).
- **Norms:** ‖·‖₂ is the spectral norm for matrices and the Euclidean norm for vectors, ‖·‖_F is the Frobenius norm, and |·|_max is the largest entry.

### 3.2 Exact claim
For every Neumann cell and every t ∈ [c1, c2] (indeed every t ∈ `t_iv`):
- G(t) is invertible;
- |μ_a(t) − μ_h,a(t)| ≤ T_mu for all a;
- |Σ_ab(t) − Σ_h,ab(t)| ≤ T_Sig for all a, b.

Hence μ_a(t) ∈ `mu[a]` and Σ_ab(t) ∈ `Sig[a][b]`, the values returned at 1114-1116.

### 3.3 Hypotheses consumed from other scopes (not re-proved here)
- **H1 (scope 1, CB2).** In the exact Laurent structure, G has no negative powers, A has none below t⁻¹ and GY has none below t⁻². The dropped coefficients are exactly 0, and μ, Σ are given by the conditioning formula for these functionals. CB2s is an exact `Fraction` lemma that is run and checked (268). The passage from the functional expansions to the Gram expansions is derivation-only.
- **H2 (CB2 kernel window, 497-511, with the CB1 moment table).** The enclosures in 617-622 contain the true coefficients for every j ≤ NK.
- **H3 (scope 2, 706-742).** M_GP ≥ sup_{|z|=ρ} max_ab |G_ab(z)|, and likewise M_GYV for A and M_GY for GY. K1 is entire, so each block is meromorphic on ℂ, and its only possible pole is at 0 (removable for G by H1). So the Laurent series converge on 0 < |z| < ∞, the Cauchy estimate |c_j| ≤ M ρ^{−j} holds for every j, and the series values at real t ∈ (0, r_hi] are the Gram matrices at r = t.
- **H4.** mpmath `iv` is inclusion-monotone with outward rounding.

### 3.4 Derivation, step by step
**S0. The cell (1058-1063).** `t_iv` = [lo.a, hi.b] ⊇ [c1, c2], with `r_lo` = lo.a ≤ c1 and `r_hi` = hi.b ≥ c2. The code checks `r_lo > 0` (1063).

**S1. Gram tails (1064-1066; `gram_tail`, 882-885).**
- For 0 < t ≤ r_hi and x = r_hi/ρ, H3 gives |δG_ab(t)| ≤ Σ_{j≥NK+1} M_GP (t/ρ)^j ≤ M_GP x^{NK+1}/(1−x) = `tailG`. The bound holds for all t in the cell, because every term is nonnegative and increasing in t.
- `tailA` (from M_GYV) and `tailY` (from M_GY) are the same bound, applied entrywise to δA and δGY.
- **x < 1 is checked.** The geometric sum needs x < 1. `gram_tail` now checks it on the interval x = r_hi/ρ, as `ck(x.b < 1, 'CB6 Gram tail needs r < rho')` (884). The actual values are x = 0.0227, 0.0455, 0.0909, 0.182, 0.273, 0.364 and 0.4545 on cells 2-8, and 0.4545 at the 1/20 calls, so x ≤ 0.4546 on every call.

**S2. Consistent point choice (lemma P).**
- Take the true G_j, A_j, GY_j, which lie in their enclosures by H2, and the fixed Ŵ_j, Ĥ_j.
- Every interval that `neumann_law` reads is an interval extension of the corresponding point expression. Those intervals are `X[k]`, `D[k]`, `RW[k]`, `RH[k]`, `GP_iv`, `WP`, `HP` and `AP`, and each is built from the same objects `Gj`, `Aj`, `A_m1`, `GY*`, `W`, `H`.
- So by H4, each point quantity at t lies in its interval at `t_iv`. The dependency problem only widens intervals.
- This is what lets one write exact identities between point quantities and then bound each term by an interval quantity.

**S3. Neumann bound on G⁻¹ (1067-1076).**
- `GP_iv` = G_P(t_iv) ± tailG, so it contains G(t) by S1 and S2.
- `Emat` ⊇ {I − B G(t)}, and `rhoE` = `frob_up(Emat)` ≥ ‖E(t)‖_F ≥ ‖E(t)‖₂.
- **Checked:** `rhoE < 0.5` (1074). The lemma itself needs only < 1.
- Then BG = I − E is invertible, with ‖(BG)⁻¹‖₂ ≤ 1/(1 − ‖E‖₂). Hence G(t) is invertible on the whole cell. This is load-bearing, because the conditional law exists only then. Also G⁻¹ = (BG)⁻¹B.
- So ‖G⁻¹‖₂ ≤ ‖B‖₂/(1 − ‖E‖₂) ≤ ‖B‖_F/(1 − rhoE) ≤ `Gnorm`, where `BF` = `frob_up(Biv)`.
- Any B works: the midpoint inverse affects only tightness.

**S4. Exact identities.** These hold for any polynomials W_P, H_P; each line follows by expanding.
- (I1) G e_w = zV − G W_P = RW − δG W_P.
- (I2) G E_H = I − G H_P = RH − δG H_P.
- (I3) μ = (A_P + δA)(W_P + e_w) = A_P W_P + δA W_P + A e_w. The last term uses the full A.
- (I4) A G⁻¹ Aᵀ = (A_P + δA)(H_P + E_H)(A_P + δA)ᵀ. Therefore Σ = D + δGY − [A_P H_P δAᵀ + δA H_P A_Pᵀ + δA H_P δAᵀ] − A E_H Aᵀ.

**S5. Bound on e_w (1089-1093).** ‖e_w‖₂ ≤ ‖G⁻¹‖₂ (‖RW‖₂ + ‖δG‖₂ ‖W_P‖₂), using ‖δG‖₂ ≤ ‖δG‖_F ≤ 6·tailG. In code, `Ew` = Gnorm·(rWn + 6·tailG·WPn), with `rWn` = `l2_up`(RW(t_iv)) and `WPn` = `l2_up`(W_P(t_iv)).

**S6. Bound on μ (1094-1100).** `AP` = A_P(t_iv), including `A_m1/t_iv`. `AmaxP` ≥ |A_P|_max, and `Amax` = AmaxP + tailA ≥ |A|_max. For each a:
- (i) the high orders: |Σ_{k=J+1}^{2NK} X_k t^k| ≤ `ext_mu` (Horner of `Xhi` times t^{J+1});
- (ii) the pole: |X_{−1}/t| ≤ max|X_{−1}|/r_lo = `pole_mu`;
- (iii) |(δA W_P)_a| ≤ Σ_l |δA_al||W_P,l| ≤ 6·tailA·WPmax;
- (iv) |(A e_w)_a| ≤ ‖row_a A‖₂ ‖e_w‖₂ ≤ √6·Amax·Ew.

By (I3), μ − μ_h is exactly the sum of the four quantities bounded in (i)-(iv). The sum of their bounds is `T_mu`.

**S7. Bound on E_H (1101-1104).** From (I2), ‖E_H‖₂ ≤ Gnorm·(‖RH‖₂ + ‖δG‖₂ ‖H_P‖₂) ≤ Gnorm·(RHF + 6·tailG·HPF) = `EH`. This uses ‖XY‖₂ ≤ ‖X‖₂‖Y‖₂ and ‖·‖₂ ≤ ‖·‖_F.

**S8. Bound on Σ (1105-1112).** For each a, b:
- the high orders of D: ≤ `ext_S` (orders J+1..3NK; GY_k is included there for k ≤ NK);
- the poles: |D_{−2}/t² + D_{−1}/t| ≤ `pole_S`;
- |δGY_ab| ≤ `tailY`;
- each cross term: |Σ_{k,l} δA_ak H_kl A_P,bl| ≤ tailA·AmaxP·Σ_{kl}|H_kl| ≤ tailA·AmaxP·6‖H_P‖_F, by Cauchy–Schwarz over the 36 entries. The two cross terms together give `12·tA·HPF·AmaxP`;
- |(δA H_P δAᵀ)_ab| ≤ 6·tA²·HPF;
- |(A E_H Aᵀ)_ab| = |row_aᵀ E_H row_b| ≤ (√6·Amax)²·‖E_H‖₂ = 6·Amax²·EH.

By (I4), Σ − Σ_h is exactly the sum of these terms, and the sum of their bounds is `T_Sig`.

**S9. Assembly (1114-1115).** μ_a ∈ Horner(`mu_ser`) ± T_mu and Σ_ab ∈ Horner(`Sig_ser`) ± T_Sig. Downstream, `band_lo` uses 2·T_mu and 4·T_Sig for the δ = q_S − q_M transform (1205-1206; `LR` at 680, `mu2_ser`/`Sig2_ser` at 681-683). That is valid because every row of `LR` has absolute sum ≤ 2, so |(LR δΣ LRᵀ)_ab| ≤ 2·2·T_Sig.

**Poles.**
- With the exact Taylor coefficients (W_0 = G_0⁻¹zV, H_0 = G_0⁻¹, H_1), CB2s makes X_{−1} = E G_0 G_0⁻¹ zV = E zV = 0, and likewise D_{−2} = D_{−1} = 0.
- For the arbitrary points of S2 they are only tiny, not zero. That is why `pole_mu` and `pole_S` are a required part of the bound. The enclosures `X[-1]`, `D[-2]` and `D[-1]` all contain 0 (checker scratch).
- The Gram-level coefficients below the kept orders (G below t⁰, A below t⁻¹) are dropped, not bounded (H1). GY has no orders below t⁻² to drop. Their 25 stored enclosures all contain 0, with max |c| = 6.3e−100 (checker scratch).
- The run-time guard at 555-567 only requires < 1e−60. That is far too weak to protect T_mu: a 1e−60 coefficient at t⁻⁶ on cell 2 would add about 1.9e−35. So validity here rests on CB2s being exact, not on that guard.

**Changes from the frozen remainder (`old_remainder`, 1077-1087).**
- The frozen code (ceil.txt:604 and 618-627, with the same lines in floor.txt) applies G⁻¹ − G_P⁻¹ = −G⁻¹ δG G_P⁻¹. Its comment drops the sign, which is immaterial for norms.
- But it applies the identity to the truncated Taylor polynomial H_P (`GinvPnorm`, from the `H_m` majorants), not to G_P⁻¹. For H_P the exact relation is G⁻¹ − H_P = G⁻¹(RH − δG H_P).
- The frozen bound therefore omits RH and, for the mean, the residual RW. It also omits the high orders `ext_mu`/`ext_S`: orders J+1..2NK of A_P W_P and J+1..3NK of D.

### 3.5 Checked numerically at run time
- `r_lo > 0` (1063), x < 1 in each `gram_tail` call (884) and `rhoE < 0.5` (1074). These are the only hypotheses of the derivation that are checked at run time. The `no_preconditioner` control fails the contraction check (test:381).
- `T_mu, T_Sig < 1e−2` (1113). This is a sanity bound only; no inequality depends on it.
- **Upstream consistency checks, not proofs:**
  - CB2 shows that the dropped Gram pole enclosures are < 1e−60 (555-567). This is far weaker than the scale that matters here; see Poles.
  - CB3 shows that the μ/Σ pole enclosures are tiny and that A_{−1} = E·G0 (686-695).
  - CB4 checks |c_j| ≤ M ρ^{−j} on the computed coefficients of orders −2..NK only (772-776). The orders above NK, which are the ones the tails need, are not checked.
- **CB7 (1307-1312)** checks that the enclosure at r = 1/20 contains a dps-100 direct law.
  - The printed deviation, 1.37e−91 for μ and 2.06e−91 for Σ, is exactly that reference's own rounding error, |direct dps-100 − direct dps-200|. Against dps 200, the Neumann midpoint is off by 6.7e−101 and 1.1e−100.
  - The corrections there are T_mu = 1.49e−10 and T_Sig = 1.23e−7. So CB7 cannot detect an under-estimated remainder anywhere between about 1e−91 and 1e−2.
  - The heavy part's half-width (≈ 9e−98) is below the reference error, so CB7 would also reject a valid remainder below about 1e−91. It tests the heavy polynomial at about 1e−91 resolution, not the remainder.
- The `old_remainder` control is caught only by the replay digests (test:384), not inside the run.

### 3.6 What rests only on this derivation
Everything from S1 to S9 except the checks just listed:
- identities I1-I4 and lemma P;
- the Neumann lemma and every norm inequality;
- the geometric Cauchy tail (its condition x < 1 is now checked at 884; the tail bound itself is not);
- monotonicity of the tail in t;
- the pole bound with r_lo;
- H1-H4, including the exact vanishing of the dropped Gram-level orders and the identification of the Laurent series with the Gram matrices at real t.

## Scope 4. Envelope algebra (`ceil_hi` 1148-1198, `band_lo` 1200-1247)

### 4.0 Claim and imported hypotheses
**Claim.** Take a cell C from 1249-1250. For every r in C, E[G_r] ≤ ⌈hi⌉ (1194, used at 1264-1266) and E[G_r] ≥ lo.a (1243, used at 1269-1271).

**Hypotheses.** None of these is checked in this scope.
- **H_law.** The law step returns the following.
  - An interval t with C ⊂ t ⊂ [0, ∞). On Neumann cells, t comes from 1060, and the check at 1063 makes t.a > 0. On the first cell, t comes from 1119, with t.a = 0.
  - Boxes `mu` (6 entries) and `Sig` (6×6) with μ(r) ∈ mu and Σ(r) ∈ Sig, entrywise, for every r in C.
  - Numbers Tm, Ts that bound every entry of the error of the order-J Taylor polynomials, whose coefficients are enclosed by `mu_ser` and `Sig_ser`.
- **H_iv.** mpmath `iv` is outward-rounded and inclusion-isotone. Also mp.prec = iv.prec = 336: both are set to dps 100 at 271-272, cov_exact sets them again, and 286 checks the dps. So every conversion mpf(x.a), mpf(x.b) (in `up`, `ub` and `t0_lower`) and every iv.mpf(mpf) of a constant is exact.
- **H_Φ.** `iv_Phi` (367-395) encloses Φ, and `iv_phi` (364-365) encloses φ.
- **H_def.** G_r = p·1_T with T = {q_M < 0, D_M > 0, D_S < 0} (README:12-14). This is the upstream definition and is not re-derived here.

### 4.1 Notation
- **Variables.** X = (q_M, q_S, a_M, a_S, b_M, b_S), indices 0..5 as in `p_affine` (1133), with X ~ N(μ(r), Σ(r)).
- **Polynomials (1134-1143).** D_M = a_M q_M − r b_M², D_S = a_S q_S − r b_S², and p = −D_M D_S. u_M = ½(a_M q_M)² + ½ + r b_M², u_S is the same with the S variables, and p̄ = u_M u_S.
- **Rotated coordinates.** δ = q_S − q_M, L = `LR` (680), and LX = (q_M, δ, a_M, a_S, b_M, b_S). p̄₂ is p̄ written with q_S = q_M + δ (1144-1145), so p̄₂(LX) = p̄(X).
- **Constants (1124-1129).** These are mpf values at dps 100: I_LO = −3.9, I_HI = −0.4, η = 0.3, BB = 2, AB = 0.8, EDGE_AM = −0.6. Each event and each margin check uses the same binary number. The events convert it with iv.mpf, which is exact by H_iv; the margins convert it with `mpf_to_fraction`, which is exact.
- **Functions.** Q(t) = 1 − Φ(t), ⌈v⌉ = `up(v)` (248), and `ub(v)` = max(|v.a|, |v.b|) (241).
- **Shorthand.** TE(P; c; [α, β]) = E[P(X)·1{α ≤ X_c ≤ β}]. E2p̄ = ⌈E[p̄²]⌉ (1162, 1222) and E2p = ⌈E[p²]⌉ (1163) are upper endpoints of the Wick enclosures.

### 4.2 Gaussian primitives
Each primitive is an exact identity, evaluated by interval extension on (t, mu, Sig).
- **G1. Wick (907-929).** E[X_i m(X)] = μ_i E[m] + Σ_j Σ_ij E[∂_j m]. This is Gaussian integration by parts, applied recursively to monomials (916-918).
- **G2. Conditional law (971-998).** X | X_c = x ~ N(μ + Σ_{·c}(x − μ_c)/Σ_cc, Σ − Σ_{·c}Σ_{c·}/Σ_cc) (974-975). The mean is stored as the affine pair [μ_i − Σ_ic μ_c/Σ_cc, Σ_ic/Σ_cc]. The G1 recursion with polynomial-in-x coefficients (985-987) gives E[P | X_c = x] = Σ_k π_k x^k.
- **G3. Truncated moments (888-905).** Write X_c = μ_c + sZ, α' = (α − μ_c)/s, β' = (β − μ_c)/s and I_j = ∫_{α'}^{β'} z^j φ.
  - I_0 = Φ(β') − Φ(α') (895) and I_1 = φ(α') − φ(β') (896).
  - I_j = α'^{j−1}φ(α') − β'^{j−1}φ(β') + (j−1)I_{j−2} (898). This is integration by parts with φ' = −zφ.
  - E[X_c^k 1{α ≤ X_c ≤ β}] = Σ_j C(k, j) s^j μ_c^{k−j} I_j (903).
- **G4. Truncated expectation.** TE = Σ_k π_k E[X_c^k 1{…}], by the tower property (1166-1170, 1214-1217, 1228-1231).
- **Enclosure.** By H_iv, H_Φ and H_law, G1-G4 contain the true value at (r, μ(r), Σ(r)) for every r in C.
  - G2-G4 are used only for c ∈ {0, 1, 4}, and each G2 use is followed by a G3 call on the same Sig[c][c].
  - Σ_cc(r) > 0 for these c follows from H_law once Sig[c][c].a > 0, and that is now checked: `trunc_mom_iv` requires `var.a > 0` (890) before its square root (891), at every truncated moment. In the recorded runs the lower endpoints are 2.0, 2.0 and at least 0.4997.
  - If Sig[c][c].a ≤ 0, the divisions at 974-975 run first and do not raise. When the enclosure contains 0 they give unbounded intervals (in mpmath 1.4.1, 1/[0,1] = [1, +∞] and 1/[−1,1] = [−∞, +∞]). Either way the run then stops at the check at 890 with `H3_BAND_V2 FAIL`.
- **G5. Tail bound (397-433).** Take X ~ N(m, v) with m ∈ mu_x and 0 ≤ v ≤ v̄ = ⌈var_x⌉.
  - `t0_lower` (413-415) first requires v̄ > 0 (414). It then takes t0 as the lower endpoint of an interval quotient over iv.sqrt(v̄) (415).
  - Only v̄ enters, so a negative lower endpoint of var_x is harmless. Such endpoints do occur: on the first cell, Sig[2][2].a = Sig[3][3].a ≈ −1.8e−38 and Sig2[1][1].a ≈ −7.4e−38.
  - The check at 432 enforces t0 > margin. Production margins are 2 (1177, 1236) and 20 (1182, 1185, 1189); margin 1 is used only in the self-test at 450.
  - **'gt' (424).** t0 ≤ (e − ⌈mu_x⌉)/√v̄. Since t0 > 0, e > m and (e − m)/√v ≥ t0. So P(X > e) ≤ Q(t0).
  - **'lt' (427).** The same argument with mu_x.a gives P(X < e) ≤ Q(t0).
  - **'abs' (430-431).** M = ub(mu_x) ≥ |m| and t0 ≤ (e − M)/√v̄, so e > M. Then P(|X| > e) = Q((e − m)/√v) + Q((e + m)/√v) ≤ 2Q((e − |m|)/√v) ≤ 2Q(t0). This uses sides = 2.
  - The function returns ⌈iv_Phi(−t0)⌉·sides (399, 402). Multiplying by 2 is exact.
- **G6. Cauchy–Schwarz.** Since p̄ ≥ 0, 0 ≤ E[p̄ 1_A] ≤ √(E[p̄²]·P(A)) ≤ √(E2p̄·P̄_A), where P̄_A is the G5 bound. The code returns [0, ⌈√·⌉] (1172-1173, 1237).
  - RESp (1182) uses |E[p 1_A]| ≤ √(E[p²]·P(A)).
  - The checks at 1164 and 1223 make sure the square-root argument is nonnegative.
  - The check labels at 1177 and 1236 call these pieces "Chernoff". A label is printed only in the message of a failing margin check, so neither reviewed transcript contains the word. No exponential-moment bound is used, and the comment at 1150-1151 now says so.

### 4.3 The majorant (derivation only)
- Take r ≥ 0. Then |D_M| ≤ |a_M q_M| + r b_M² ≤ u_M, using |x| ≤ (x² + 1)/2. Likewise |D_S| ≤ u_S.
- Hence |p| ≤ p̄ and p̄ ≥ ¼. So for every event A, ±E[p 1_A] ≤ E[p̄ 1_A]; if A ⊂ ∪A_i, then E[p̄ 1_A] ≤ Σ_i E[p̄ 1_{A_i}].
- P and P̄ are built with the interval r = t (1156, 1212). The condition r ≥ 0 comes from t ⊂ [0, ∞) (H_law).

### 4.4 Ceiling (`ceil_hi`)
- **U1.** {q_M < 0} \ T = A_M ∪ A_S, with A_M = {q_M < 0, D_M ≤ 0} and A_S = {q_M < 0, D_S ≥ 0}. So E[G_r] = E[p 1{q_M < 0}] − E[p 1_{A_M ∪ A_S}] ≤ E[p 1{q_M < 0}] + E[p̄ 1_{A_M}] + E[p̄ 1_{A_S}].
- **U2.** A_M ⊂ {a_M > EDGE_AM} ∪ {|b_M| > BB} ∪ {I_HI ≤ q_M ≤ 0}. Suppose a point of A_M lies in none of the three sets: then a_M ≤ EDGE_AM, |b_M| ≤ BB and q_M < I_HI. In that case a_M q_M > (−EDGE_AM)(−I_HI) > ⌈t⌉·BB² ≥ r b_M², so D_M > 0, a contradiction. The middle inequality is the exact Fraction check at 1155, made per cell.
- **U3.** {D_S ≥ 0} ⊂ {q_S ≥ 0} ∪ {a_S ≤ 0}: if q_S < 0 < a_S, then D_S ≤ a_S q_S < 0. No run-time check is needed.
- **U4.** {q_M < 0} splits into the windows [I_LO, I_HI], [I_HI, 0] and [−100, I_LO], plus the residue {q_M < −100}. p is signed, so this has to be a partition, not just a cover. The shared endpoints are null sets, because Σ_00 > 0.

| term | line | upper endpoint bounds | method |
|---|---|---|---|
| Wp1, Wp2, Wp3 | 1179-1181 | the three TE(p; 0; window) | G4, two-sided |
| RESp | 1182 | E[p 1{q_M < −100}] | G6 with E2p; 'lt', e = −100, margin 20 |
| BaM | 1183 | E[p̄ 1{a_M > −0.6}] | G6; 'gt', margin 2 (1175-1177) |
| BbM | 1184-1186 | E[p̄ 1{\|b_M\| > 2}] | TE on [−100, −2] and [2, 100]; G6 'abs' at e = 100, margin 20 (two tails) |
| BqM | 1187 | E[p̄ 1{I_HI ≤ q_M ≤ 0}] | G4 |
| BqS | 1188-1189 | E[p̄ 1{q_S ≥ 0}] | TE on [0, 100]; G6 'gt' at e = 100, margin 20 |
| BaS | 1190 | E[p̄ 1{a_S ≤ 0}] | G6; 'lt', e = 0, margin 2. t0 > 2 forces m > 0, so P(a_S < 0) = P(a_S ≤ 0) |

hi is the sum of these nine terms (1194), and U1-U4 with 4.3 give E[G_r] ≤ ⌈hi⌉. The cell decimal is rounded up exactly at 1266, after the cap check at 1265. The uniform bound is the maximum over cells (1274-1277).

### 4.5 Floor (`band_lo`)
- **F0. Rotated law (1205-1206).** Lμ and LΣLᵀ are built from series-level coefficients (681-683), with tails 2Tm and 4Ts, rounded up. Each row of L has ℓ¹ norm ≤ 2, so H_law bounds the errors by 2Tm for the means and by 2·2·Ts = 4Ts for the covariances. This gives δ ~ N(μ_1 − μ_0, Σ_00 − 2Σ_01 + Σ_11), enclosed by mu2[1] and Sig2[1][1].
- **F1. The box.** Q = {I_LO ≤ q_M ≤ I_HI}, and B_cert = Q ∩ {|δ| ≤ η} ∩ {a_M ≤ −AB} ∩ {a_S ≥ AB} ∩ {|b_M| ≤ BB} (1201).
- **F2. B_cert ⊂ T** (exact checks at 1209-1211). On B_cert:
  - q_M ≤ I_HI < 0;
  - D_M ≥ AB(−I_HI) − ⌈t⌉BB² > 1/20 (1209);
  - q_S = q_M + δ ≤ I_HI + η < 0. Since a_S ≥ AB > 0 and q_S ≤ 0, D_S ≤ a_S q_S ≤ AB·q_S ≤ AB(I_HI + η) < −1/20 (1210).

  So the fact q_S < 0 is load-bearing for D_S < 0. The separate check at 1211 is redundant: 1210 with AB > 0 already forces I_HI + η < 0.
- **F3.** p > 0 on T, so p 1_T ≥ p 1_{B_cert}. Also Q \ B_cert ⊂ {|δ| > η} ∪ {|b_M| > 2} ∪ {a_M > −0.8} ∪ {a_S < 0.8}. With 4.3 this gives

  E[G_r] ≥ E[p 1_Q] − E[p̄ 1{|δ| > η}] − E[p̄ 1{|b_M| > 2}] − E[p̄ 1{a_M > −0.8}] − E[p̄ 1{a_S < 0.8}].

| term | line | encloses / bounds above | method |
|---|---|---|---|
| EqI | 1214-1217 | E[p 1_Q] | G4, two-sided |
| Edelta | 1239 | E[p̄ 1{\|δ\| > η}] | forced G6; 'abs' on (mu2[1], Sig2[1][1]) at e = η, margin 2, two tails (B3). E2p̄ comes from the unrotated law, which is valid because p̄₂(LX) = p̄(X) |
| Ebeta | 1240 | E[p̄ 1{\|b_M\| > 2}] | if Sig[4][4].a > 1e−4 (1227): Ebar − TE(p̄; 4; [−2, 2]) (1231); otherwise G6 'abs' at e = 2. The first branch is taken on every recorded cell (Sig[4][4].a ≥ 0.4997) |
| EaM | 1241 | E[p̄ 1{a_M > −0.8}] | forced G6; 'gt' at e = −AB |
| EaS | 1242 | E[p̄ 1{a_S < 0.8}] | forced G6; 'lt' at e = AB |

- lo = EqI − Edelta − Ebeta − EaM − EaS (1243). Interval subtraction gives lo.a ≤ EqI.a − Σ⌈piece⌉, which is at most the right side of F3.
- The cell decimal is rounded down exactly at 1271, after the c_Z check at 1270.
- The event map at 1235 reads only b for the 'abs' kind, so it is correct only when a = −b. The check at 1234 now enforces a = −b on the tail branch whenever no side is given. Both call sites (1239, 1240) satisfy it.
- PBAR2 and Ebar2 are passed to Edelta but not used there (`force_chern`). They serve only the consistency check at 1221.

### 4.6 Frozen programs versus v2
**Unchanged.** This was checked by reading the code, plus a normalized diff of the helpers.
- The events, the constants, the ±100 windows, and the margins 2 and 20.
- The nine ceiling summands (ceil.txt:693-712 against 1179-1194).
- The five floor terms (floor.txt:699-702 and 704 against 1239-1243) and their branch choices, including the 1e−4 switch (floor.txt:681 against 1227).
- p, p̄ and p̄₂ (ceil.txt:653-662; floor.txt:651-660 and 668-673).
- G1-G4. The frozen ceil.txt:472-562 and floor.txt:472-562 are byte-identical to each other. Apart from formatting and the guard at 890, the only change is that `trunc_mom_iv` now takes a and b as intervals, and its callers wrap them with `iv.mpf`.

**Changed.**
- **B3.** The frozen floor's 'abs' branch used one tail, P = iv_Phi(−t0).b (floor.txt:692-697); v2 uses two (431). The frozen ceiling's far tail at |b_M| > 100 already used two (ceil.txt:705). Every tail now goes through one function, `event_prob_upper` (417-433).
- **Rounding of t0 and of the square-root pieces.** The frozen programs used round-to-nearest mp (ceil.txt:685-691 and 696-710; floor.txt:687-697). v2 uses an interval lower bound for t0 (415) and an interval upper bound for the square root (1173, 1237).
- **Margin checks.** The frozen checks were done in mpf: the ceiling cover used the constant 0.05 (ceil.txt:652), and the floor used the mpf decimal cell end (floor.txt:645-649). v2 checks exactly in Fraction, with ⌈t⌉ taken per cell.
- **Added:** the CB5 tail-helper and event-bound self-tests (404-411, 435-452); the second-moment checks (1164, 1223); the fail-closed guards (414, 890, 1234); and the mutation hooks `tail_helper_one_side` (400-401), `one_tail` (431) and `drop_fail_piece` (1191-1192).
- **Removed:** the ceiling nesting check (ceil.txt:724), listed at README:84; the floor check "piece enclosures sane" (floor.txt:703), now listed at README:45; and the dead non-forced path in the frozen ceiling's upper_half (ceil.txt:679-683), both of whose calls forced the tail path.
- **Indirect (H_Φ scope).** These changes reach every tail and every truncated moment. O2 removes float() in the Gordon–Mills branches (378 and 382, against frozen lines 451 and 456). `iv_phi` uses iv.exp (365, against the frozen iv.e ** at line 437). The `iv_Phi` series tail is outward (394, against frozen line 469).

**Cross-check against the transcripts.** On cells where the law did not change at the printed precision, frozen floor − v2 floor = (1 − 1/√2)·Edelta_v2:

| cell | frozen − v2 floor | ratio to (1 − 1/√2)·Edelta_v2 |
|---|---|---|
| (1/100, 1/50] | 1.26e−12 | 1.0004 |
| (1/50, 3/100] | 9.43e−6 | 0.9995 |
| (3/100, 1/25] | 2.540e−3 | 1.0002 |

On (1/25, 1/20] the ratio is 0.578, because B2 changes the law there. The ceilings on (1/400, 1/200] through (1/50, 3/100] equal the frozen ones up to the last digit, which v2 now rounds outward.

### 4.7 What the program checks at run time
- The cover margin (1155) and the type margins (1209-1211; 1211 is implied by 1210).
- t0 > margin (432) at every tail evaluation:
  - per ceiling cell: RESp 1182, BaM 1183, the BbM far tail 1185, the BqS far tail 1189, and BaS 1190;
  - per floor cell: Edelta 1239, EaM 1241 and EaS 1242;
  - Ebeta 1240 only when Sig[4][4].a ≤ 1e−4, which never happens in the recorded runs.
- v̄ > 0 before every tail's square root (414); Sig[c][c].a > 0 before every truncated moment (890); a = −b on the two-sided tail branch of `piece_outside` (1234).
- The second moments are ≥ 0 (1164, 1223), and the two E[p̄] enclosures intersect (1221).
- The tail and event self-tests against Gordon's lower bound (406-410, 442-451). These test a necessary condition on point laws only.
- Consumption: U < 4 (1265) and lo > c_Z (1270). These two are not validity checks.

### 4.8 What rests only on the derivation
1. H_def: E[G_r] = E[p 1_T] with T = {q_M < 0, D_M > 0, D_S < 0} (README:12-14).
2. 4.3: |p| ≤ p̄ for r ≥ 0, hence ±E[p 1_A] ≤ E[p̄ 1_A] and the union bound. The code only builds the polynomials (1132-1146).
3. p > 0 on T, used in the floor for p 1_T ≥ p 1_{B_cert}.
4. F2: the step from the exact checks at 1209-1210 to B_cert ⊂ T. It includes q_S < 0, which is needed for a_S q_S ≤ AB q_S.
5. U2: the step from 1155 to A_M ⊂ {a_M > −0.6} ∪ {|b_M| > 2} ∪ {I_HI ≤ q_M ≤ 0}.
6. U3: {D_S ≥ 0} ⊂ {q_S ≥ 0} ∪ {a_S ≤ 0}.
7. U1, U4 and F3: the decomposition identities, the completeness of the nine ceiling summands (1194) and of the five floor terms (1243), and the partition of {q_M < 0} into the windows plus the residue. This item also covers the null-set conventions: shared endpoints, a_S < 0 versus a_S ≤ 0, and |b_M| = 2 in Ebar − TE.
8. G6: Cauchy–Schwarz (1172-1173, 1182, 1237).
9. G5: Gaussian tail monotonicity for 'gt' and 'lt', and P(|X| > e) ≤ 2Q((e − |m|)/√v) for 'abs' (417-433).
10. G1-G4: Isserlis, the conditional-Gaussian formulas, the truncated-moment recurrence with the binomial expansion, and the tower property (916-918, 974-975, 895-903, 1170, 1217, 1231).
11. F0: the rotated-law tails 2Tm and 4Ts, from the ℓ¹ row norms of LR (680, 1205-1206). Also that E2p̄ of the unrotated law serves for Edelta, because p̄₂(LX) = p̄(X).
12. The enclosure principle: evaluating G1-G4 and the polynomials on (t, mu, Sig) encloses the value at the true (r, μ(r), Σ(r)). This trusts H_iv, H_Φ and H_law.

## Same-family pre-checks (credit 0)

Same-provider agents ran these checks: the section drafters and one adversarial checker per section. They ran on the earlier program (`99eceb98…`), whose certified decimals and transcripts the final program reproduces byte for byte. The scripts are scratch files, not part of this PR. The checks earn no independence credit. They are listed so a reviewer can see what has been tried numerically, and where each check stops.

### Scope 1 (CB2s)
- **Exact recomputation (drafter).** An independent re-expansion equals `laurent_expand` on all 12 functionals, and E·V^{(0)} = Y^{(−1)} holds by direct substitution. The program's own lines 460-548 were run on an exact Fraction shim, with planar moments and with random rational moments. They match the formal B-Gram coefficient by coefficient on orders −8..62, with 0 mismatches; the only nonzero negative orders are GYV −1 and GY −2, −1. X_{−1}, D_{−2} and D_{−1} are exactly 0, and A_{−1} = E·G_0 and GY_{−2} = E·G_0·Eᵀ hold exactly.
- **Independent exact recomputation (checker; does not import the program).** It uses hand-transcribed tables, its own expansion and its own B. It recomputes (a)-(d), E and the window margin of 6 (attained by V5×V5, V5×Y2, V5×Y3 and Y2×Y3), and gets the three pole coefficients exactly 0 for planar and for random rational moments.
  - Non-vacuity: zV3 = 1 gives X_{−1} = (0, 0, 1, 1, 0, 0); Y0 = M(0,2)/z gives X_{−1,0} = −6/5 and |D_{−2}| = 2; `wrong_window` keeps the poles at 0 and moves μ_aM(0) to −1/2.
- **Interval series against formal B (checker; unmodified source with a hook after CB3).** All 7668 interval Gram coefficients (orders −8..62, 108 entries) match independent formal-B enclosures built from the program's A2 to 2.1e−99, with no disjoint pair. Σ_{p≤NK} c_p z^p matches the direct covariance `ce.cov(mode='iv')` to 5.3e−86 at z = 1/128 and to 8.4e−88 at z = 1/64. That interval path omits a denominator term (review 5607640770) of relative size about 4e−205 at the library defaults, far below these agreements.
- **First cell (drafter).** The first-cell heavy polynomials were compared with the program's direct point law (dps 100) at t = 1/16384, 1/32768, 1e−6 and 1e−9. The deviations are 5.33e−74, 2.03e−72, 5.09e−63 and 1.18e−45 for μ, and 8.0e−74, 3.04e−72, 7.64e−63 and 1.77e−45 for Σ. They grow as t → 0 because the direct law loses precision through the r⁻⁶ cancellation, and all lie below MT and ST. **Limit:** this cannot see a pole coefficient of order 1e−99.
- **Other checks.** The frozen ceiling and floor tables equal v2's. mpmath iv behaviour on zero pivots was probed (see 1.3 (e)). The pole magnitudes quoted in 1.6-1.7 come from the unmodified source run in memory with a reporting hook.
- **Final program (this guide).** `structure` prints PASS with the E of 1.2. `structure --mutation wrong_window` prints `MUTATED STRUCTURE RUN (wrong_window) … NOT A CERTIFICATE` and exits 3. `structure --mutation structure_window` fails at "CB2s (d) E zV = 0 required, row 2" and exits 1.

### Scope 2 (CB4)
- **θ.** θ was recomputed exactly in `Fraction` from the program's mpf inputs: 0.49437044607085834, with up(θ(R)) ≥ it. By exact bisection, the smallest R with θ ≤ 1 is 1296.833 and with θ ≤ 1/2 it is 2592.833. The 2% grid returns R = 2622.3491580327, and R·r* = 0.16005.
- **Recursion past NK.** The majorant recursion stays at most 2.7e−146·C R^j for H and for W; the checker ran j ≤ 900 and the drafter j ≤ 1600.
- **MT and ST.** Recomputed by plain convolutions at dps 120, with K = 700 terms and exact 1/ρ, they agree with the program to relative 3.3e−101 and 3.5e−101.
  - MT = 1.03890e−44, of which the A_{−1} pole term is 1.03884e−44. That term equals 18·[W_m[62] t^61 + C_W (Rt)^63/((1−Rt)t)], with the second part 5.77e−46. The main part is 6.3e−49.
  - ST = 1.83842e−38, of which the a = c = −1 term is 1.83820e−38. The one-pole groups contribute 2.2e−42, the product group 6.8e−47 and the GY part 2.6e−203.
- **A_n(ρ).** It exceeds a direct dps-130 sum over |j| ≤ 400 with the full Z by 1.5e−99 (n = 0), 1.9e−99 (n = 2), 2.6e−97 (n = 4) and 2.2e−91 (n = 8).
- **Sampled sup against M.** GP 3.02 against 2.96e6, GY 255.5 against 343.7, and GYV 27.5 against 1.95e4.
- **Cauchy margins on the computed orders.** max_j ‖c_j‖_max ρ^j / M is 1.0e−6 for GP, 1.5e−4 for GYV and 0.022 for GY (at j = 0); 1.4e−3 for A_{−1}, 0.026 for GY_{−1} and 0.72 for GY_{−2}. So an M_GY understated by more than a factor of about 1.39 would fail the run-time check, while the GP and GYV parts would catch only errors of 3 to 6 orders of magnitude.
- **H2, two ways.**
  - The drafter widened the window to J+14−n. Every coefficient through NK, and R, C, MT and ST, stayed identical.
  - The checker evaluated every GP, GY and GYV entry directly from the spectral kernel (|j| ≤ 300, dps 170) at 256 points on |z| = 3, and took a DFT. This route has no kernel truncation. Every coefficient at orders 0..62, and A_{−1}, GY_{−1} and GY_{−2}, lies inside the program's enclosure: max |DFT − mid| is 4.3e−100 and max deviation/radius is 0.79. A truncated window would show errors near |c_62|, about 1e−43.
  - Three exactly-zero pole entries show about 1e−130 in the DFT. That is aliasing from order 254/255: −3·3^256/(2^127·127!) = −8.13584919065e−130 matches the GY(0,1) z⁻² value digit for digit.
- **True tail.** Σ_{k=61}^{124} max|X[k]| r*^k ≈ 1.2e−287, far below MT. The computed |H_j|^{1/j} is about 0.40 and |W_j|^{1/j} about 0.34, so R = 2622 is a very loose but valid growth bound.
- **Limit.** All of these use computed orders or sampled points. None proves the Cauchy estimate beyond NK or the induction step.

### Scope 3 (B2)
- **Drafter.** An instrumented copy reproduced `results/ceil.transcript.txt` byte for byte (sha256 `d0def90d…`). It compared each bound with a dps-200 direct law at c1, at the midpoint and at c2 of every cell.
- **Checker, at the same 21 points.** The dps-200 interval direct Grams G(t), A(t) and GY(t) intersect the program's point enclosures (Horner at [t] ± tail). They come from the library's interval path, which omits a denominator term of relative size about 4e−205 (review 5607640770), below dps-200 resolution and far below these comparisons. The dps-200 μ and Σ lie inside both the point-cell and the cell enclosures, and ‖G(t)⁻¹‖₂ ≤ 25.56 < Gnorm.

| cell | x = r_hi/ρ | rhoE | Gnorm | tailG | Ew | EH | T_mu | T_Sig |
|---|---|---|---|---|---|---|---|---|
| (1/16384,1/400] | 0.0227 | 0.0126 | 26.0 | 8.8e−98 | 6.3e−95 | 3.5e−94 | 7.5e−90 | 5.1e−84 |
| (1/400,1/200] | 0.0455 | 0.0129 | 26.0 | 8.3e−79 | 5.9e−76 | 3.3e−75 | 1.7e−72 | 2.9e−68 |
| (1/200,1/100] | 0.0909 | 0.0258 | 26.4 | 8.0e−60 | 5.8e−57 | 3.3e−56 | 8.5e−54 | 7.0e−50 |
| (1/100,1/50] | 0.182 | 0.0516 | 27.1 | 8.2e−41 | 6.1e−38 | 3.4e−37 | 4.5e−35 | 1.9e−31 |
| (1/50,3/100] | 0.273 | 0.0516 | 27.1 | 1.2e−29 | 8.5e−27 | 4.8e−26 | 3.1e−24 | 6.5e−21 |
| (3/100,1/25] | 0.364 | 0.0517 | 27.1 | 9.8e−22 | 7.2e−19 | 4.1e−18 | 1.8e−16 | 2.4e−13 |
| (1/25,1/20] | 0.4545 | 0.0518 | 27.1 | 1.45e−15 | 1.1e−12 | 6.1e−12 | 2.0e−10 | 2.0e−7 |

- **Actual errors.** At all 21 points, |μ − μ_h| ≤ 6.9e−101 and |Σ − Σ_h| ≤ 1.2e−100; also ‖e_w‖₂ ≤ 3.5e−100 and ‖E_H‖₂ ≤ 2.1e−99. These are at the rounding level of the dps-100 coefficients.
- **Dominant terms.** T_mu ≈ √6·Amax·Ew and T_Sig ≈ 6·Amax²·EH. Ew and EH are driven by their 6·tailG terms, while rWn ≈ 1.5e−98 to 1.9e−98 and RHF ≈ 9.1e−98 to 9.5e−98 are set by interval width. So the printed remainder is effectively the Cauchy tail of the pin Gram, times Gnorm and |A| or |A|². A carries the 1/t pole: Amax = 4.9e4 on cell 2 and 75 on cell 8. On cell 2, pole_mu = 5.6e−95 and pole_S = 2.2e−90.
- **Why the T_Sig scale matters.** On the last cell T_Sig = 2e−7, which is large enough to move the printed 15-digit decimals there.
- **Dropped coefficients.** All 25 stored dropped Gram-level enclosures contain 0 (max |c| = 6.29e−100), and so do X[−1], D[−2] and D[−1].
- **CB7.** The printed deviation (1.37e−91 for μ, 2.06e−91 for Σ) equals |direct dps-100 − direct dps-200|. The Neumann midpoint is within 6.74e−101 (μ) and 1.14e−100 (Σ) of the dps-200 law, and the μ enclosure's half-width without T_mu is 9.03e−98.
- **x > 1, on the earlier program, before the guard at 884 existed.** On the hypothetical cells [1/20, 3/25], [1/20, 1/5] and [1/10, 1/9], plain Python stopped with mpmath's ordering AssertionError when building `GP_iv` (now line 1068). Under -O, every call failed the contraction check (now 1074), because min_{x>1} x^63/(x−1) = 169.89 makes |tailG| ≥ 5.0e8. So that path was already fail-closed; the guard now names it.
- **README:92** reports a dps-200 comparison "at 32 points". That figure is from the earlier same-family adversarial pass during authoring, on the program as it then stood (certified decimals as now). Its skeptic recomputed the direct law with `cov_exact` at dps 200 at four points per cell, c1 + (c2 − c1)·{0.001, 0.37, 0.999, 1}, over all eight cells including the first. All 32 points lay inside their cell's enclosure, with |direct − heavy| at most 6.9e−101 for μ and 1.14e−100 for Σ. The checker above used 21 points and found the same.
- **Limit.** The actual errors (≈ 1e−100) sit at rounding level, far below the corrections on every cell. No numeric test here can show that the remainder bound is valid, only that it is not violated at the sampled points.

### Scope 4 (envelope)
- **Replays.** `ceil --verify` and `floor --verify` both PASS. `floor --mutation one_tail` fails at "CB5 event bound P(|X| > t) self-test at 2.5", with exit 1.
- **Primitives.** `trunc_mom_iv` agrees with `mp.quad` for k ≤ 8 on three windows, with relative error < 1e−25. On every cell's actual box, `event_prob_upper` dominates the exact normal-CDF supremum over the box corners at every site with t0 ≤ 60. For Edelta on (1/400, 1/200] the ratio is about 2.001, the two-tail factor.
- **Polynomials.** P, PBAR and PBAR2 (1132-1146) enclose the closed-form p and p̄ at 20 random points, for each of r = 0, 0.0371 and 0.05.
- **G1-G4 against independent quadrature.** At each cell's midpoint point law, the program's values agree with numpy Gauss–Hermite (5-D) × Gauss–Legendre quadrature to a worst relative difference of 2.3e−15. The quantities compared were E[p], E[p̄], E[p̄²], E[p²], Wp1-3, BqM, the BqS window, the BbM windows and TE(p̄; 4; [−2, 2]).
- **Point law at r = 1/20** (the CB7 direct law), with the helpers extracted verbatim.
  - The envelope gives hi = 3.66697 and lo = 2.48761. Monte Carlo with 4e6 samples gives E[G] = 3.22672 ± 0.00224, inside that range.
  - BqS is 0.365102 against 0.366134 ± 0.0012, and Wp1 is 2.65847 against 2.66014 ± 0.0017.
  - Edelta is valid but loose: the Cauchy–Schwarz bound is 0.120, while Monte Carlo gives E[p̄ 1{|δ| > η}] ≈ 1.9e−4.
  - No sample violated p̄ ≥ |p|, B_cert ⊂ T, U2 or U3.
- **Per-cell Monte Carlo.** At each cell's midpoint law, 1e6 samples give E[G] between 3.223 and 3.235 (± 0.0045). Every value lies inside that cell's [floor, ceiling].
- **Unbounded intervals (this guide, mpmath 1.4.1).** With a variance enclosure [−1e−38, 2], [0, 2] or [0, 0], the conditional-law divisions and the products after them return unbounded intervals without raising. So the guard at 890 is reached.
- **Limit.** Quadrature and Monte Carlo use midpoint point laws. They test the primitives and the envelope numbers, not the set inclusions for every r in a cell.

### Confirmed lower-severity findings and their disposition
Every confirmed finding has severity minor or doc. The envelope checker reported no inequality in the wrong direction, no missing term, no norm mismatch and no rounding that fails to be outward. The CB4 checker reported no blocking or major defect.

**Fixed in the final program:**
1. `gram_tail` had no x < 1 guard (scopes 2 and 3; minor). It now has `ck(x.b < 1, 'CB6 Gram tail needs r < rho')` (884).
2. A negative variance enclosure reached `iv.sqrt` and raised ComplexResult: a negative lower endpoint in `trunc_mom_iv`, or a negative upper endpoint in `t0_lower`. `main` does not catch that exception (1491), so the run ended without the documented FAIL line (scope 4; minor). The run still failed closed: exit 1, no PASS, no transcript, no record. Both square roots are now preceded by named checks, at 414 and 890.
3. The 'abs' event map in `piece_outside` used only b, assuming a = −b (scope 4; minor). The check at 1234 now enforces it.
4. `structure --mutation` printed `STRUCTURE PASS` and exited 0 for a mutation the lemma cannot see (scope 1; doc). It now prints `H3_BAND_V2 MUTATED STRUCTURE RUN (NAME): …; NOT A CERTIFICATE` and exits 3 (1451-1454; test:91-98).
5. The `ceil_hi` comment pointed to a direction analysis in README.md, which has none (scope 4; doc). The comment at 1149 now points to section 3 of the frozen H3_BAND_CEIL.md (its lines 59-87).
6. The "Chernoff" labels hid that these pieces are Cauchy–Schwarz combined with an exact Gaussian tail (scope 4; doc). The comment at 1150-1151 now says so.
7. The `band_lo` comment used an undefined I′ (scope 4; doc). The comment at 1203 now writes [I_LO, I_HI].

**Documentation corrected (README):**
- README:45 now lists the dropped floor check (floor.txt:703) and the added second-moment checks.
- README:47 calls the B1 guard a regression guard.
- README:48 says that θ(R) ≤ 1/2 and the 0.2 are margins inherited from the frozen program, and states the actual convergence conditions. It also says the first cell drops the pole coefficients on CB2s alone.
- README:63 states the `structure` behaviour under a mutation, and README:64 lists the fail-closed guards.
- README:69 calls E rational, 0/1 for the tables of record. The checker had judged the earlier "0/1 matrix" wording a nit, not a defect.
- README:72 calls the numeric pole checks consistency checks only (orders from −8, threshold 1e−60).
- README:76 restricts the Cauchy sanity check to orders −2..J+2, with the higher orders resting on the derivation.
- README:92 now calls CB7's 2e−91 the rounding error of its dps-100 reference, not the true truncation error.

**Left as is, with the reason:**
- **The 1e−60 threshold of the CB2 pole-pattern check (555-567)**, and 1e−55 and 1e−50 at 687 and 690. These are consistency checks, and validity rests on the exact CB2s. A sharper check (0 ∈ c, or |c| < 1e−95) would also pass, but it would not change what is proved.
- **The check cannot see orders below LMIN = −8.** The tables have none, and the exact lemma runs first.
- **The B1 dominance guard (767) is identically true in unmutated code.** It is a regression guard, and it catches `signed_maxima`.
- **The Cauchy sanity check (776) is loose for GP and GYV and says nothing beyond NK.** It is a sanity check; the higher orders rest on the derivation, as 2.6 says.
- **θ(R) ≤ 1/2 and R·r* < 0.2** are margins inherited from the frozen program. Validity needs only θ(R) ≤ 1, R·r* < 1 and QRHO·r* < 1.
- **No run-time tie between the kernel window and the tables.** The DFT check confirms the window for the tables of record, and `structure_soft_pole2` is stopped by CB2s (b).
- **The first cell drops the pole coefficients with no numeric fallback.** This is not a gap: the cancellations follow from (c) and (d) by the algebra in 2.3, and README:48 now says so.
- **The check at 1211 is implied by 1210.** It is harmless. The property it states, q_S < 0 on B_cert, is load-bearing and is proved by 1210.
- **The word "Chernoff" stays** in the check labels at 1177 and 1236, and in README:78. The labels are printed only in the message of a failing margin check, so neither reviewed transcript contains the word, and the comment at 1150-1151 says what the pieces are.

**Rejected as not a defect:** the suggestion that q_S < 0 is not needed for the README definition of T. It is needed: the step a_S q_S ≤ AB q_S uses q_S ≤ 0, since a_S ≥ AB is unbounded above.

## What a reviewer still has to check

These are the inequalities and identities that no run checks. Each item points to where this guide derives it.

**Scope 1 (CB2s)**
- [ ] The Gram Laurent coefficients equal B on the formal coefficients (1.4 Step 1). This covers K1 being entire, the DISP sign (541), the (−1)^q in the (M,S) case, and the S-S cancellation Σ_{q+q'=N} C(N,q')(−1)^{q'} = 0 (500-548).
- [ ] (a)-(d) plus (e) imply (i) and (ii), through Ỹ = Y − E·V/z, including the expansions of GY and A·H·Aᵀ (1.4 Steps 2-3).
- [ ] `solve_in_span` returns an E that satisfies (c) whenever it does not return None (186-209).
- [ ] The maximum cross-station weight n1 − k_a − k_b is 6, so the window at 502 is complete through NK (1.5).
- [ ] The H and W recursions (624-634) and θ's sum from i = 1 are the true inverse, given (a).
- [ ] The true G_0 lies in the box G0, so 612-614 proves (e); this also needs CB1.
- [ ] μ and Σ are the conditional law given V = zV; and, outside this scope, the tables are the upstream functionals.

**Scope 2 (CB4)**
- [ ] |K1^{(n)}(w)| ≤ A_n(ρ) on |w| ≤ ρ, with the lattice tail ≤ 2u_{j0}/(1−q), and the lower endpoint of Z1_iv at most the exact truncated sum, which is at most Z (706-724).
- [ ] The sup bound on |z| = ρ (728-737), and Cauchy's estimate for every j, including j > NK in majA and majGY and i > J in θ.
- [ ] ‖PQ‖_max ≤ 6‖P‖_max‖Q‖_max; the induction h_j ≤ C_H R^j θ(R) and w_j ≤ C_W R^j θ(R); θ(R) ≤ 1 suffices (778-783, 800).
- [ ] G(t) is invertible on [0, 1/R), with inverse Σ H_j t^j.
- [ ] The coefficient identities for μ_k and Σ_k, the four Σ-tail groups, and the index bounds J+1, J+2 and J+3 (861-873).
- [ ] The closed form of the `Maj` suffix sums and the P2/P3 sums (819-851), and monotonicity in t, so that evaluation at 2⁻¹⁴ suffices (859, 875-876).
- [ ] The 2MT/4ST widening (1205-1206).

**Scope 3 (B2)**
- [ ] The Gram tail M x^{NK+1}/(1−x) for x < 1, monotone in t (S1); and H3, which identifies the Laurent series with the Gram matrices at real t.
- [ ] Lemma P: the point quantities lie in the interval quantities for any fixed Ŵ_j, Ĥ_j (S2).
- [ ] The Neumann lemma and ‖G⁻¹‖₂ ≤ ‖B‖_F/(1 − rhoE) (S3).
- [ ] Identities I1-I4 (S4).
- [ ] The norm inequalities of S5-S8, including the constants 6, √6 and 12 and the Frobenius-to-spectral steps.
- [ ] The pole bounds with r_lo, and that the dropped Gram-level orders are exactly zero (via scope 1).
- [ ] The 2T_mu/4T_Sig widening in `band_lo` (S9).

**Scope 4 (envelope)**
- [ ] H_def, upstream: E[G_r] = E[p 1_T] (README:12-14).
- [ ] |p| ≤ p̄ for r ≥ 0 and the union bound (4.3); p > 0 on T.
- [ ] U1-U4: the ceiling decomposition, the cover U2 from 1155, the inclusion U3, and the partition with its null-set conventions.
- [ ] F2: B_cert ⊂ T from 1209-1210, using q_S < 0. F3: the floor decomposition.
- [ ] G1-G4: Isserlis, conditioning, the truncated-moment recurrence and the tower property.
- [ ] G5: the 'gt', 'lt' and 'abs' tail bounds, including the factor 2 for 'abs'. G6: Cauchy–Schwarz.
- [ ] F0: the rotated tails 2Tm and 4Ts, and p̄₂(LX) = p̄(X).
- [ ] The enclosure principle under H_iv, H_Φ and H_law.
