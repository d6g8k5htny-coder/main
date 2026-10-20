# H3 CLOSURE — the normalizer lower bound Z_r ≥ c_Z r² (U2D-UPPER)

**Date:** 2026-09-13. **Verifier:** H3-closure line (adversarial verification mode).
**Candidate closure under test:** Theorem G.3.1 of
`/mnt/agents/output/SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md` (lines 1534–1567),
with Part G.1–G.2 (lines 1446–1532) and G.7 (lines 1632–1653) as its proof context.

**VERDICT: CLOSED for the frozen upper-campaign law, at theorem grade, with one
named amendment and one named non-explicit constant.**

- The supplement's Theorem G.3.1 *as printed* is stated for the **3D** side-24
  field (conventions line 9: `T^3_24`; weight line 11: `1{H_M≺0, ind(H_S)=2}`;
  8 raw pins). It does **not**, as a cited statement, cover the frozen 2D law.
- The **2D instance of the same theorem** is the manuscript's Theorem
  3.11(i)–(ii) (`manuscript_v2_2026-09-13.md` line 200–207, stated explicitly
  "For the typed pair-Palm law (Preliminary 2.5)"), and its proof is the
  dimension-generic Part-G argument re-run in dimension two. Every step of that
  argument is verified or certified below (§2); nothing in the argument is
  3D-specific.
- The assembly's transcribed limit object
  `c_0 = E[|α_M det Q_M|·|α_S det Q_S|·1{Q_M≺0, α_M<0} ∩ {ind_S=2}]` carries the
  **3D index convention**: on the 2D field `ind_S=2` reads `H_S ≺ 0` (S a
  *maximum*), whose limit probability is **0** — read literally, the transcribed
  formula gives c_0 = 0 and does **not** close H3 (demonstrated, §4). The
  correct 2D typing (S a saddle ⟺ `det H_S < 0` ⟺ index 1) gives the corrected
  limit object, certified here at 60+ digits:

      c_0 = E[ U² · 1{U<0} ],   U ~ N( −(6/5)·a₂ , a₄ − a₂² ),
      c_0 ∈ [3.230978535287004948096246569018163150764858010558710998470,
             3.230978535287004948096246569018163150764858010558710998471]  (±5e-97 tail)

  in particular **c_0 ≥ 3.230978535287004948096246569018163150765 > 0
  (certified, interval arithmetic with certified spectral tails)**.

---

## 1. Scope verification (pin by pin)

Frozen upper-campaign law (assembly H3): `Q_r` = continuous Gaussian regression
version of the six-pin conditioning `f(M)=b=6/5`, `f(S)=b−r³/6`,
`∇f(M)=∇f(S)=0` at `M=(−r/2,0)`, `S=(r/2,0)` on the normalized periodized
side-24 **2D** Bargmann–Fock field; `W_r = |det H_M det H_S|·1{M max, S saddle}`;
`Z_r = E_{Q_r} W_r`; `dP_r = W_r dQ_r / Z_r`.

| Coordinate | Assembly H3 | Manuscript Preliminary 2.5 (2.7) | Supplement G.3.1 | Verdict |
|---|---|---|---|---|
| field | periodized BF, `T²_24`, weights `e^{−|k|²/2}` on `(π/12)Z²` | same (lines 13, 110) | `T³_24`, `(π/12)Z³` (line 9) | **G.3.1 3D; 2D instance = Thm 3.11** |
| stations | `M=(−r/2,0)`, `S=(r/2,0)` | identical (line 81) | `M=x−rt/2`, `S=x+rt/2`, `t∈S²` | match (2D axis-aligned frame) |
| pin count | 6 (value+gradient at M,S) | 6 (V_r, (2.7)) | 8 (3D) | match in 2D |
| pin values | `b=6/5`, `b−r³/6`, gradients 0 | identical (lines 81, 112–115) | `b`, `b−κr³/6` | match with κ=1 |
| pin basis | raw six pins ↔ corrected V_r | corrected basis V_r (2.7) | corrected basis (G.1.3) | **same σ-algebra: det B_r = r⁻⁵ ≠ 0** (§2.S1) |
| weight | `|det H_M det H_S|·1{type}` | `W_r = |det H_M det H_S| 1_type` (line 115) | same (line 1539) | match |
| typing | M max, S saddle (2D: `H_M≺0`, `det H_S<0`) | "maximum at M and a saddle at S" (line 110) | `H_M≺0, ind(H_S)=2` (3D index-2 saddle) | match in 2D convention; **the transcribed c_0 formula's `ind_S=2` is the 3D convention, wrong if read in 2D (§4)** |
| mark | `b=6/5>0`, κ=1: compact positive singleton | `b=6/5` standing (line 81) | "uniformly on compact positive (b,κ) sets" | covered (§2.S9) |
| RN structure | `P_r = W_r dQ_r/Z_r` | `dP⁰/dP = W_r/Z_r` (line 115) | (A.1) | match |

The identification of this exact law with the campaign's estimand of record is
already adjudicated: `REVIEW_LPW/RA_R1_R6.md` §R1(a), items 1–7 (all exact
match; sole residue there — the GP-DER-118 body — is a provenance note, not a
mathematical input to H3).

## 2. The assembled 2D proof of Theorem G.3.1(2D) — step grades

Statement proved: for the frozen law, uniformly on compact positive (b,κ) sets
(in particular at (6/5, 1)): `c_Z r² ≤ Z_r ≤ C_Z r²` and `E_{Q_r}[W_r²] ≤ C r⁴`
for `0 < r ≤ r_max`, with `c_Z = c_0/2` explicit and `r_0 > 0` (the small-r
range) existential (see §5 for the exact honesty boundary).

- **S1 — Basis invariance (machine-verified, symbolic).** The change of basis
  `P6_r → V_r` has `det B_r = r⁻⁵ ≠ 0` for all `r>0`; Gaussian conditioning
  depends only on the pin span (regression equivariance — analytic). Verified:
  H2 `pin_transform.py` selftest P5 (re-run by this closure, both modes
  byte-identical); independently, `REVIEW_LPW/RA_R1_R6.md` item 3 (12 fail-closed
  checks, PASS), including pinned-value agreement
  `(b,0,0,0,0,−1/6)` and `det T_r = −r⁻⁵`. **G.3.1's proof, which conditions on
  V_r, therefore proves the theorem for Q_r (raw pins) unchanged.**
- **S2 — Endpoint kernels (machine-verified, dimension-free).**
  `h(∓r/2) = ∓(1/r)∫(r/2∓s)h'(s)ds` on the zero-mean class:
  `verify_palm_soft_factors.py` (K1), re-run by this closure:
  `ALL_ASSERTIONS_PASS`, normal/`-O` byte-identical.
- **S3 — Soft factorization (machine-verified, dimension-free).**
  `det H = rD`, `D = α det Q − r βᵀ adj(Q) β`; in 2D `Q=q` scalar,
  `D = αq − rβ²`: script item (S1), same PASS.
- **S4 — Value window (machine-verified).** `f(S)−f(M) = ∫(w²/2−r²/8)g'''dw`,
  kernel mass `−r³/12` (script item K2, PASS). Consequence: the corrected pin
  forces `∂₁₁₁f(0) = 2κ` in the blow-up (`−(1/12)·∂₁₁₁f(0) = −κ/6`).
- **S5 — Corrected-pin floor (analytic + certified endpoint + numeric
  corroboration).** 2D limit jet `V_0 = (f, ∂₁f, ∂₂f, ∂₁₁f, ∂₁₂f, ∂₁₁₁f)(0)`:
  six **distinct** derivative functionals; Gram SPD by full spectral support
  (manuscript Def 2.2 / Lemma 3.1 — certified there; analytic argument:
  independence of exponentials/monomials, dimension-free, identical to
  supplement G.1.1). Uniform floor on `(x,t,r)` compacta: continuity +
  compactness (supplement G.1.3, dimension-free). **Certified endpoint of
  record:** `λ_min(Γ_0) > 31/250 = 0.124` (ten exact Sylvester minors,
  LPW_CONSTANT package, reproduced in its falsifier F2), with certified modulus
  `‖Γ_r − Γ_0‖_F ≤ 19.071156·r` on `[0, 1e-5]` → uniform floor `0.12380…` on
  `[0, 1e-5]`, inherited by the six-pin block (Cauchy interlacing).
  This closure adds the explicit limit-Gram determinant certificate
  `det G_0 = (a₄−a₂²)(a₂a₆−a₄²)a₂³ ∈ [12−2.4e-100, 12+2.5e-100] > 0`
  (CK4, iv path) and NUMERIC corroboration: the 6×6 `Cov(V_r)` smallest
  eigenvalue `≥ 0.0391` on `r ∈ {0.1,…,0.005}` (CK8a; labeled NUMERIC).
- **S6 — Upper bounds (analytic; independently derived for this law).**
  `W_r ≤ r²|D_MD_S|` with bounded Gaussian conditional moments (S3 + S5 floors)
  ⇒ `Z_r ≤ C_Z r²`, `E W_r² ≤ Cr⁴` (Cauchy–Schwarz). Independently derived for
  exactly this six-pin 2D law in the LPW candidate §9 with `C_Z = 4B_3`,
  reviewed PASS (RD_R4_R5 §R5(i), polarity confirmed).
- **S7 — Lower bound (analytic, verified line-by-line in 2D).**
  `Z_r/r² = E|D_MD_S|1{type}`. The scaled soft variables converge in law under
  `Q_r` (Gaussian regression continuous in the pin Gram — SPD by S5 — and in the
  bounded pin values): `α_M → −κ`, `α_S → +κ` **deterministically** (the pins
  fix `∂₁₁f(0)=0` and `∂₁₁₁f(0)=2κ`; the conditional variances of the α's are
  `O(r²·poly)` → 0; NUMERIC: `Var(α_M|pins) = 1.2e-5` at `r=0.005`, monotone in
  r, CK8 table), `β's → ∓(1/2)∂₁₁₂f(0)`, `q_M, q_S → U = ∂₂₂f(0)`. The type
  events: M max ⟺ `q_M<0 ∧ D_M>0`; S saddle ⟺ `D_S<0`; both converge to the
  single open event `{U<0}` (since `D_M→−κU`, `D_S→+κU`), whose boundary
  `{U=0}` is null under the limit law (`s² = a₄−a₂² > 0` certified, CK5b). The
  pinned directions `α_M=−κ<0`, `α_S=+κ>0` lie strictly inside the event —
  the partial degeneracy of the limit law is harmless. Weak convergence +
  uniform integrability (Gaussian tails dominate the degree-2 polynomial
  integrand, uniformly via S5's ceilings) give `Z_r/r² → c_0`. Note: the
  supplement's phrase "nondegenerate centered Gaussian" for the scaled-variable
  limit is imprecise (the α-coordinates degenerate to ±κ); the load-bearing
  nondegeneracy is that of the transverse block `U`, which holds and is
  certified. Verified.
- **S8 — c_0 > 0 (machine-certified, this closure).** §3.
- **S9 — Compact-mark uniformity (analytic; coverage of b=6/5 confirmed).**
  `c_0(b,κ) = κ²·E[U²1{U<0}]` with `U ~ N(−b a₂, a₄−a₂²)` is continuous in
  `(b,κ)` and strictly positive on every compact positive mark set
  (`s²>0` certified; `P(U<0) = Φ(ba₂/√(a₄−a₂²)) > 0`); the singleton
  `(b,κ)=(6/5,1)` is compact positive. The remainder of the uniformity
  argument (convergence continuous in marks; positivity of `Z_r/r²` at fixed
  `r` under the nondegenerate pin law; compactness) is dimension-free and
  applies verbatim. **Coverage of b=6/5: confirmed.**
- **S10 — A.4.4–A.4.6 audit.** These lemmas are the capture/escape arithmetic
  feeding G.5/G.7, **not** the normalizer; their machine audit
  (`verify_a44_a46_flow_v1.py`: 17 checks at R∈{1,2,5,50} + symbolic
  Young/eigen-slope chains; re-run by this closure: PASS, both modes
  byte-identical) supports the G.7 corollary chain only. The normalizer
  theorem itself consumes no A.4.x step.

## 3. The limit object c_0 — certified computation on the exact field

By S7, `c_0 = E[|α_M q_M|·|α_S q_S|·1{type-limit}] = κ²·E[U² 1{U<0}]` with
`U = ∂₂₂f(0)` under the limit pins. Gaussian regression (the odd covariances
vanish exactly — sine sums at `s=0`, CK3):

`m = E[U|V_0] = −(6/5)·a₂`,   `s² = Var(U|V_0) = a₄ − a₂²`,

with `a_{2m} = (1/Z₁)Σ_{k∈(π/12)Z} e^{−k²/2} k^{2m}` the **exact finite-torus
moments** (never planar: `1−a₂ = 9.65254179896e-123 > 0` certified, errata
2(b) of record). Closed form:

`c_0 = (m²+s²)·Φ(−m/s) − m·s·φ(−m/s)`.

**Certified value (iv path of `cov_exact.py`, 100 dps, certified spectral
tails carried; Φ by in-program certified power series with geometric tail
certificate):**

    c_0 ∈ [3.23097853528700494809624656901816315076485801055871099847056563983657866727484113533199277175345604743007,
           3.23097853528700494809624656901816315076485801055871099847056563983657866727484113533199277175345608823472]

(interval width < 5e-97). **Certified lower bound: c_0 ≥ 3.230978535287004948096246569018163150765 > 0.**

Cross-checks (all pass, see certificate/falsifier transcripts):
- planar diagnostic (`a₂=1, a₄=3`) gives `3.23097853528700494809624656901816315076485801055871099847057…`,
  inside the certified interval (CK6d) — finite-torus correction `< 1e-95`;
- independent falsifier engine (direct lattice sum at N=96, certified tail
  `4.1e-129`; Poisson/Hermite identity; `mpmath.ncdf`): reproduces
  `c_0 = 3.2309785352870049480962465690181631507648580105587…` (F1);
- full 6×6 regression solve agrees with the 2×2 reduction to `<1e-85` (CK5d/e);
- the same regression structure appears independently in the LPW_CONSTANT
  package: `E[J|U_0] = (−a₂b,0,0,0)`, `Cov(J|U_0) = diag(a₄−a₂²,…)` (matches);
- NUMERIC (labeled): finite-r Monte Carlo of `Z_r/r²` under the exact
  conditional law (corrected-pin frame, two independent covariance
  implementations — spectral and image-kernel): `3.206, 3.224, 3.229, 3.227`
  at `r = 0.1, 0.05, 0.025, 0.0125` (σ≈0.0037), approaching `c_0`; typed-event
  probability → `0.80` (`= Φ(−m/s) = 0.801928…`).

## 4. The named amendment — the transcribed `ind_S=2`

The assembly's H3 statement quotes the limit object with `1{…}∩{ind_S=2}`,
verbatim from the 3D supplement. On the 2D field, `ind_S=2` is `H_S ≺ 0`
(S a **maximum**), not a saddle. Since `α_S → +κ > 0` a.s. under the limit pin
law, the limit event `{H_S ≺ 0}` is empty up to a null boundary: the literal
formula evaluates to `c_0 = 0` and **cannot** close H3. Demonstrated
executably: at `r = 0.1, 0.05, 0.025, 0.0125`, zero occurrences of
`{H_S ≺ 0}` in `1.5e6×4` and `1.2e6×3` samples from the exact conditional law
(two engines), while the true saddle event occurs with frequency ≈ 0.79–0.80;
Cantelli bound `P(α_S<0|pins) ≤ 2.7e-5 → 0` (CK8c, F3). The assembly must
consume the 2D typing (`S` saddle ⟺ `det H_S<0` ⟺ index 1) and the corrected
limit object of §3. With that amendment the closure is exact — the assembly's
`W_r` typing ("M max, S saddle") already means the 2D event; only the quoted
c_0 display carried the 3D index.

## 5. Honest grade

- **Machine-certified (this closure, iv path, certified tails, both modes
  byte-identical, mutations caught fail-closed):** exact-torus moments; exact
  orthogonality structure; limit Gram determinant > 0; conditional law
  `(m, s²)`; `s² > 0`; the certified interval for c_0 (hence c_0 > 0);
  dependency hashes of the consumed H2 libraries.
- **Machine-verified (supplement/H2 scripts, re-run here, PASS, byte-identical
  two-mode):** kernel identities (K1), value window (K2), soft factorization
  (S1/S2), pin-basis determinants `det B_r = r⁻⁵`, `det T_r = −r⁻⁵`
  (symbolic); A.4.4–A.4.6 arithmetic (not a normalizer input).
- **Analytic (dimension-free, verified step-by-step):** spectral independence
  and compactness floors; regression equivariance; weak convergence + UI;
  boundary-nullity; compact-mark uniformity; Cauchy–Schwarz upper bounds
  (independently derived with `C_Z = 4B_3` for this law, reviewed PASS).
- **NUMERIC corroboration only (labeled, never load-bearing):** finite-r
  Monte-Carlo tables; the V_r eigenfloor grid.

**Grade: theorem-grade for the assembled 2D statement as written in §2–§3**
(every step either analytic-and-verified or machine-certified; no named
unproved premise is consumed by H3). Two explicit honesty boundaries:
(a) Theorem G.3.1 *as printed in the supplement* is the 3D instance — anyone
quoting it must quote the 2D instance (manuscript Theorem 3.11(i)–(ii)) or §2
here; (b) `r_0` is existential in the record of record (the convergence S7 has
no certified modulus). A certified explicit radius is within reach of existing
instruments (the pin-block floor is already certified uniformly on
`[0, 1e-5]` via `λ_min ≥ 0.1238…`, LPW_CONSTANT) but certifying a modulus for
the *expectation* is an unexecuted extension — **not claimed**. The NUMERIC
evidence is consistent with `c_Z = c_0/2` already holding at `r ≤ 0.1`.

## 6. Closure statement the assembly can consume

> **H3 (2D normalizer lower bound).** Let `Q_r` be the continuous Gaussian
> regression version of the six-pin conditioning `f(M)=6/5`,
> `f(S)=6/5−r³/6`, `∇f(M)=∇f(S)=0`, `M=(−r/2,0)`, `S=(r/2,0)`, on the
> normalized periodized side-24 2D Bargmann–Fock field; let
> `W_r = |det H_M det H_S|·1{H_M≺0, det H_S<0}` and `Z_r = E_{Q_r}W_r`. Then
> there exists `r_0 > 0` such that
>
>     Z_r ≥ c_Z r²,   0 < r ≤ r_0,
>     c_Z = c_0/2,  c_0 = E[U²1{U<0}],  U ~ N(−(6/5)a₂, a₄−a₂²),
>     c_Z ≥ 1.615489267643502474048123284509081575382  (certified),
>
> and additionally `Z_r ≤ C_Z r²` and `E_{Q_r}W_r² ≤ Cr⁴` (analytic; `C_Z=4B_3`
> route of record). At `(b,κ)=(6/5,1)`; uniformly on compact positive mark
> sets. The weighted law `P_r = W_r dQ_r/Z_r` is well-defined on `(0, r_0]`.

## 7. Reproduction and receipts

```
cd /mnt/agents/output/K3_SIDE24_LB/UPPER2D/H3_closure
python3 h3_c0_certificate.py  > cert_normal.txt 2>&1   # exit 0, ALL_CHECKS_PASS
python3 -O h3_c0_certificate.py > cert_O.txt 2>&1      # byte-identical
python3 h3_falsifier.py       > fals_normal.txt 2>&1   # exit 0, FALSIFICATION_ATTEMPTS_FAILED
python3 -O h3_falsifier.py    > fals_O.txt 2>&1        # byte-identical
MUTATION=planar_moments python3 h3_c0_certificate.py   # exit 1 (caught CK2)
MUTATION=wrong_window   python3 h3_c0_certificate.py   # exit 1 (caught CK6)
```

Dependencies re-verified live by this closure (all PASS, byte-identical
two-mode): `SIDE24_gap_fill/scripts/verify_palm_soft_factors.py`,
`verify_a44_a46_flow_v1.py`; H2 `selftest_cov_exact.py`,
`selftest_pin_transform.py`; H2 MANIFEST sha256 pins checked in-program (CK0).

Receipts are separate files: `cert_normal.txt`, `cert_O.txt`,
`fals_normal.txt`, `fals_O.txt`, `mut_planar.txt`, `mut_window.txt`,
`RECEIPT_h3.json`, `MANIFEST.sha256`.
