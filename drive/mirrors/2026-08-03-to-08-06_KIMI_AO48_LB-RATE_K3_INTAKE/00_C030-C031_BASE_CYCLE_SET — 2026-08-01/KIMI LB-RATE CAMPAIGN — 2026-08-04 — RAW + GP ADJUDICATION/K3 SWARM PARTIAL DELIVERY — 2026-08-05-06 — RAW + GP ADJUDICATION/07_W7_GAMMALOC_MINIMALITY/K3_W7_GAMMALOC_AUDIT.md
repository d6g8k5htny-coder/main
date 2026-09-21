# K3-W7 — γ-LOC / P-NMZ-γ audit and discharge attempt (K3 SWARM Phase 2)

**Mandate:** 19fb9293-5c52-85f2-8000-095a601387af (Phase 2, superseding WO-063 Task 3's frame). **WO-063 echo:** sha256 `e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2`. **Starting record:** KIMI-DER-026 (sha256 `d6bde0636b400d995f889e55446892631f3c2d462e060b34c27532fbd4a05bfd`), not treated as settled.
**Date:** 2026-08-05 · **Labels in this file; machine receipts in the transcripts** (`verify_t_normal.txt` = `verify_t_O.txt`, sha256 `19c3c0099e0c1ca06fd1908827baeb40534e71759dbfd28b4efa244cc7864310`; `mutation_t_normal.txt` = `mutation_t_O.txt`, sha256 `2a04b2872365327d3f2b4f1384dc70ac9968d0b559904d7759db54a9e2783e03`).
**Programs:** `verify_k3_w7_gammaloc_v1.py` sha256 `e16054ef132749029f6dd0075f26d609fa296905db470c74ef300adcbf9c84f8` (fail-closed `ck()`→`SystemExit`, no bare `assert`, byte-identical `python3` / `python3 -O` output, 16 record hash bindings, 13 record-audit substring receipts, mutation self-tests); `mutation_driver.py` sha256 `095ce955bfb358aaf078901fd46f75c0d107aeb66471321d96f7af60b3874d73` (executable falsifier harness: 1 pristine accept + 8 mutated rejects; byte-identical both modes).
**Disposition (exact):** **DISCHARGED-BY-MINIMALITY.** The γ-LOC statement the lower composition genuinely requires is strictly weaker than (ii-c)'s 1−O(exp); that weaker statement stands at the assembly's theorem grade on the current record (its proved clauses from LB-3; its constant floor from C025-R2 + C028/R3′), and the 1−O(exp) clause is consumed at **no node** of the theorem-grade assembly of record. P-NMZ-γ is therefore **retired from the load path** — not proved, not refuted — and retained as an open, non-load-bearing mathematical question with measured support. Zero new conditionals are introduced; the assembly's pre-existing named set (OBL-FAR-COMPOSE route (ii), sup-over-zone, mean-ridge connectivity, KR validity + collar exclusion, residue #12 slab density, OBL-P0-FLOOR for the explicit r-range) stands exactly as C031 registered it and is owned by the assembly (W10), not by γ-LOC. **Coordination caveat (W10):** this disposition is certified against the current record's load path (receipts A1–A8); if the recomposed assembly consumes a 1−O(exp) rate at any node, falsifier F4 fires and the disposition reverts to DER-026's CONDITIONAL. No status or Boolean claims are made.

---

## 1. P-NMZ-γ, verbatim from KIMI_DER_026.md (mandate item 1)

> **Premise P-NMZ-γ (analytic, named, exactly one).** *There exist, for all r ≤ r₀: (i) a deterministic tube family 𝒯_r around the pin-conditioned nominal other-branch ascent of the window saddle y, from y to the horizon section {d = d₀ = 5}; (ii) a margin function μ(r) := the C¹(𝒯_r)-distance from the pin-conditioned nominal field m_r (conditional mean plus pin shifts, on the Palm fiber) to the capture-below-b boundary ∂𝒞_b = {fields whose window saddle's other branch terminates at a maximum of value ≤ b}; and (iii) a sub-Gaussian scale s(r) for the conditioned residual's C¹(𝒯_r) tube seminorm, P(‖E‖_{C¹(𝒯_r)} > t) ≤ 2 exp(−t²/(2 s(r)²)) — such that the clearance ratio grows:*
>
> **Λ(r) := μ(r)/s(r) → ∞ as r ↓ 0, in the form Λ(r)/√(2 ln(1/r)) → ∞** *(sufficient for o(r^p) tails for every fixed p: certificate S3).*

(Presence bound in certificate receipts A12–A13 against the DER-026 file bytes.)

## 2. The minimal γ-LOC genuinely required by the lower composition (mandate item 2)

**Audit of the load path (record receipts A1–A8, fail-closed against file bytes).** The theorem-grade assembly of record (C031, sha256 `e7998ef0…f32e`) states verbatim: "**Theorem grade (conditional):** AO ≥ c_cond − 2.1·(ℓ/2) − [named non-adjacency channels] with c_cond > 0 the R3′ conditional-barrier constant" and "1 − q ≥ c_cond·c_Λ·r³ conditional on the same set". The AO floor's source chain is C025's R-architecture (sha256 `d1afd84b…792a`): "AO-failure ⊆ {non-adjacency} ∪ {terminal ≤ b}"; "**(R2)** P(terminal ≤ b) ≤ E[N_maxband(B_d₀)] + far band-termination"; "**(R4)** … p̄(δ₀) = P(unconditional ascent from height b − δ₀ terminates ≤ b) ≤ 1 − c". Read with R1 (monotone ascent ⇒ band entry within ℓ/2 of b) and C028/R3′ (c_cond = 0.089569·P₀ > 0 by the support theorem, conditioning-robust at 36.0076× margin on the d₀ = 5 slab), this yields a **constant, r-free shortfall**, not an exponential rate:

P(terminal ≤ b) ≤ E[N_maxband(B₅) | 9 pins] + p̄(δ₀) ≤ 2.1·(ℓ/2) + (1 − c_cond), hence **P(terminal > b) ≥ c_cond − 2.1·(ℓ/2) − [named non-adjacency channels] ≥ c_cond/2 > 0 for all r ≤ r₀**, with the near term EXACT: 2.1·(ℓ/2) = 7r³/40 (r = 1/20: 7/320000 = 2.1875e-5; r = 1/40: 7/2560000 = 2.734375e-6 [EXACT]) and the explicit range

**r₀³ = (20·0.089569/7)·P₀ = 0.25591142857142857·P₀** [decimal-dps-17], i.e. r₀ = 0.6349·P₀^{1/3} [decimal-dps-10].

γ-LOC's own register entry in C031 §6 is verbatim: "**γ-LOC tube-local** (architecture) — inherited by the C022 positivity chain." Its consumption there is clause (i) (the qualification verdict's σ(f|N)-locality/measurability — what makes E[N_qual] a local Palm functional and the barrier's C¹-ball success event evaluable) and clauses (ii-a)/(ii-b) (the terminal exists and is a maximum, making "terminal ≤ b" a well-defined event). All three are **PROVED** in LB-3 (§2.1–§2.2). The 1−O(exp) stabilization (ii-c) appears at no node of the theorem-grade assembly: the assembly's AO floor is the constant c_cond, and its named conditional set (receipt A5: "OBL-FAR-COMPOSE route (ii); the sup-over-zone step; mean-ridge connectivity; KR validity + collar exclusion; and the R0/γ-LOC architecture") contains no exp-rate item.

**The minimal sufficient statement (γ-LOC-min), discharged:** (a) clause (i) measurability — PROVED (LB-3 §2.1); (b) clauses (ii-a)/(ii-b) termination at a maximum — PROVED (LB-3 §2.2); (c) the constant floor P(other-branch terminal > b | adjacency, pins) ≥ a₀ > 0 uniformly for r ≤ r₀ — standing at the assembly's theorem grade with a₀ = c_cond/2, via C025-R2 + C028/R3′, modulo the assembly's named set. The lower bound 1 − q ≥ c_cond·c_Λ·r³·(1 − o(1)) needs nothing stronger.

**Two honest consequences of the arithmetic (certificate S2).** (i) The explicit r-range inherits OBL-P0-FLOOR: with P₀ at the C028 honest-null 95% upper bound 3.2e-4 [decimal-dps-2, measured], r₀ = 0.04342567254 [decimal-dps-10] — the r = 0.05 rung is NOT covered; at P₀ = 1e-4, r₀ = 0.02946885264; only "for all sufficiently small r" is unconditional in P₀ (c_cond > 0 by the support theorem). (ii) The measured-grade routes to a stronger AO (C025 ν_T consequence (i): far failure ≈ density·(ℓ/2) = O(r³) with density proxy 0.62–0.69 [decimal-dps-2, measured]; the AO ≈ 1 measurement qual/raw 0.9997/0.99986) remain measured-grade; even those are polynomial, not exponential — the record contains **no mechanism, at any grade, that consumes or produces a 1−O(exp) terminal-value rate**.

**What this does to P-NMZ-γ:** it is the sufficient condition for a statement (ii-c, 1−O(exp)) that the composition does not use. Proving it would upgrade the AO floor from a constant to 1 − o(r^p) — relevant only to converting the measured-grade constant C* ≈ 0.87–0.97 to theorem grade with value ≈ 1, not to the existence of the rate c·r³. It stays on the record as an open question (§5), with its DER-026 discharge chain (CONDITIONAL) intact for whenever a consumer appears.

## 3. The seven-way separation (mandate item 3)

| # | Component | Status on the record | Grade |
|---|---|---|---|
| 1 | **Nominal ascent-tube geometry** | Endpoint saddle S: cone C_E + strip S_E + inward tube T_in, threshold-free (F.4; machine-verified, DER-026 S1 re-verified: 17 exact-rational checks × R = 1,2,5,50). Third saddle y (arch/rim): NO nominal tube in any artifact of record — the arch/rim branch is not the endpoint branch. | endpoint: proved (normal-form class, F.6 carried residue "side-24 normal-form verification"); third saddle: open |
| 2 | **Uniform C¹ clearance from the capture-below-b boundary** | Horizon slab d ∈ [5, 5.5]: certified — total C¹ perturbation 0.003629792782641913 ≤ gate 0.1307, slack 0.127070207217358087, 36.0075651219052× margin [decimal-dps-16]. Near/moderate d ≲ 3: no station of record; clearance measured 4.6–9.7σ at r = 0.05 (diagnostic). | slab: R3′/C028 (on-grid residue #12); d ≲ 3: open (P-NMZ-γ core) |
| 3 | **Full arch/rim parameter family** | Λ-support: 730-station grid, C*_∞ = 0.9091 ± 0.038 (quad) ± 0.009 (shell), collar residual |ỹ| < 0.15 quantified (C027); rim disk δ_w = √(2ℓ/stiff) ≈ 0.005; arch filament |y| ≈ 0.010–0.012 — scaled positions 0.2–0.24 at r = 0.05, inside the blow-up chart. | derived-on-grid (C027 class, residue #11) |
| 4 | **Near/moderate region d ≲ 3** | Lemma FD stabilization does not reach (horizon d ≈ 3–5); C030 variance table v = 0.018/0.55/0.976 at d = 1/2/3 (r = 0.025) [decimal-dps-3, derived-on-grid]; conditional-mean needle response order-one-tenths of σ (dual weights 6.556e4/5.183e5, C022). | certified as stated; clearance: open |
| 5 | **Conditional Gaussian C¹ tube-tail** | Borell–TIS / sub-Gaussian seminorm calculus: standard and available; converts any certified (μ, s) into exp(−μ²/2s²); o(r^p) conversion e^{−c/r²} ≤ p!c^{−p}r^{2p} machine-verified (p = 1…32, DER-026 S3; regression here S3). | machinery proved; inputs (μ, s) on d ≲ 3: open |
| 6 | **Pinned-to-pair-Palm transfer** | Palm ∝ pinned·W₂ null-set equivalence (MS Shift §2); probability-bound transfer by Cauchy–Schwarz against E⁰[W_r²] ≤ Cr⁴ (supplement G.3/G.5 pattern) — P^Palm(F) ≤ C·P⁰(F)^{1/2}, which preserves superpolynomial rates. Not an obstruction to O(exp). | proved pattern (G.3/G.5) |
| 7 | **Explicit r dependence; meaning of O(exp)** | O(exp) = superpolynomial in r: exp(−Λ(r)²/2) is o(r^p) ∀p iff Λ(r)/√(2 ln(1/r)) → ∞ (thresholds 3.461636765204571 at r = 0.05, 3.841291165279683 at r = 0.025 [decimal-dps-16]). Record holds Λ only at r = 0.05, measured: [4.6, 9.7]; two-rung qual/raw failure ratio 2.1428571429 [decimal-dps-10, measured] cannot resolve growth. No proved r-dependence for Λ exists. | mechanism EXACT; Λ(r): measured one-rung only |

## 4. The DER-027a caution (mandate item 5), made concrete

The variance channel (DER-027a class; LB-3 S2–S4 envelopes: value-variance deviation ≤ 2.24% at d ≥ 3, ≤ 4.6e-6 at d = 5, rigorous envelopes beyond d ≈ 6.1–7.5) controls **only** the conditional variance of one-point jets. It does **not** control: the conditional mean (the needle response is the counterexample — LB-3 §2.3 records that the full one-point-law TV *including the mean* at testbed pin values is 6.6%–31% at d ≥ 3, not 2.2%, via the Gram-null amplification ‖G6⁻¹v‖₁ = 65560.02/518320.01); full TV of the conditional law; flow topology (separatrix structure is a C¹-over-tube property, invisible to one-point variances); terminal values (path functionals of the whole trajectory). This audit's far-end clearance accordingly uses the R3′ station table (mean pull m, gradient pull gradm, projection field proj_c1) and never the variance envelope. Nothing in this deliverable promotes a variance certificate to a mean/TV/topology/terminal claim.

## 5. Discharge attempt: what the strong premise would still require (mandate item 4)

Outcome selected from the mandate's list: **weaker-but-sufficient statement established (§2); the strong premise neither proved nor refuted; zero new conditionals.** A full proof of P-NMZ-γ with explicit constants and r-range would require, in order: (i) a certified station net across the near/moderate tube (d ≲ 3) on the exact periodized kernel (LB-3 S2–S4 class: lattice (π/12)ℤ², masses e^{−|k|²/2}, |k| ≤ 30, tail ≤ 4.4e-185) computing the conditional mean's C¹ profile along the nominal other branch of the arch/rim saddle family; (ii) a deterministic confinement lemma for that branch in the third-saddle configuration (the F.4 endpoint lemmas do not apply — the scaled potential's third critical point is outside their scope); (iii) a sub-Gaussian scale s(r) for the conditioned residual's C¹ tube seminorm on the same net (Borell–TIS; covariance floors/ceilings from Lemma G.1.3-class bounds); (iv) the growth conclusion Λ(r)/√(2 ln(1/r)) → ∞ from the r-family of (i)–(iii) — the step with no current measured resolution (one-rung data only); (v) the Palm transfer (row 6 — no obstruction). A refutation would require a certified configuration whose nominal clearance is ≤ 0 or shrinking — none exists in the record; the measured signs point the other way (m* 4.6–9.7σ; esc = 0 in 46,400 pooled flows, diagnostic).

## 6. Falsifier spec (mandatory; executable parts shipped)

- **F0 (executable, shipped):** `mutation_driver.py` — 9 falsification instances against the certificate (pristine accept; total/gate/v/count/sum/rice/closes/truncation rejects). Any acceptance of a mutated artifact proves the verifier vacuous.
- **F1 (premise counterexample):** a certified near/moderate-zone computation showing the pin-conditioned nominal's other-branch clearance μ(r) ≤ 0, or Λ(r)/√(2 ln(1/r)) bounded — refutes P-NMZ-γ (would convert this deliverable's §5 into a kill).
- **F2 (record falsifier):** any receipt string A1–A13 ceasing to hold in a re-frozen C031/C025/LB-3/UB0/DER-026 (the certificate's S1 is fail-closed against exactly this).
- **F3 (arithmetic falsifier):** a counterexample to any EXACT link (7r³/40 near-term; ledger 65/256; p₀ values; escape > b; r₀³ = 20c_cond/7; tail inequality) — machine-pinned in S2–S3.
- **F4 (coordination falsifier, W10):** if the recomposed assembly consumes a 1−O(exp) terminal-value rate at any node, the minimality finding of §2 is void for that assembly and the disposition reverts to DER-026's CONDITIONAL (P-NMZ-γ again load-bearing). Request to W10 (via lead): confirm the recomposed AO floor's source is the constant c_cond class, or name the consuming node.
- **F5 (measured falsifier, stands from UB0 §6/A.4):** "a rim/arch flow whose other branch terminates below b at an off-image maximum (none in 11,200 + 28,800 + 6,400 pooled compatible-ensemble flows)".

## 7. Dependency table (name — sha256 — role)

| artifact | sha256 (16 or full) | role |
|---|---|---|
| AO48-WO-063 | e699bf44b7b84fca… (full in §0 receipts) | tasking frame (superseded in part by mandate 19fb9293) |
| C031_LBRATE_Integration.md | e7998ef0d17d951b | theorem-grade assembly; AO-floor source; named register (A1–A5) |
| C025 TerminalHeight Package.md | d1afd84b3143ed51 | R1–R4 architecture; ν_T measured context (A6–A8) |
| C025 Freeze.md | 0f8cb3dcb337b5c1 | freeze binding for C025 |
| Lemma UB0.md | c6a90469b6e43c59 | γ-LOC scope; measured m* profile (diagnostic); falsifier |
| Lemma UB0 Addendum.md | 224438cb740565ff | two-rung qual/raw diagnostic; pooled counts |
| C028 FarAscent Package.md | 64b94b6e5342b3f8 | c_cond = 0.089569·P₀; η = 0.1307; support theorem; OBL-P0-FLOOR |
| C029 BasinPersistence Package.md | b23c3d42ccc2182e | adjacency channel context |
| C030 CountingLemmas Package.md | aed6683edaed6d48 | KR-MB near term 2.1·(ℓ/2); horizon table |
| C022 Observed Update.json | 9bc0647b885931ba | Lemma FD; needle dual weights |
| c031 verify.json | 83313c96d1610130 | station table of record (far-end clearance) |
| SIDE24_GAP_FILL_SUPPLEMENT.md | 33609b70d8c4d970 | F.4/G-stack; Palm-transfer pattern; carried residue |
| LB3_DISCHARGE.md | 08153fa42a2d4c01 | proved clauses (i)/(ii-a)/(ii-b); verbatim (ii-c); needle-TV record |
| KIMI_DER_026.md | d6bde0636b400d99 | P-NMZ-γ verbatim source; CONDITIONAL chain (starting record) |
| DER-026 certificate + transcripts | ece7244acaa69a2f / 61e4dc8a46de2dd5 | regressed links (S3) |
| verify_k3_w7_gammaloc_v1.py (this) | e16054ef13274902 | fail-closed certificate S0–S5 |
| mutation_driver.py (this) | 095ce955bfb358aa | executable falsifier harness |
| verify transcripts (this) | 19c3c0099e0c1ca0 | receipts, byte-identical both modes |
| mutation transcripts (this) | 2a04b2872365327d | receipts, byte-identical both modes |

## 8. Reproduction

```
cd /mnt/agents/output/K3_SIDE24_LB/W7_gammaloc
python3 verify_k3_w7_gammaloc_v1.py > verify_t_normal.txt   # exit 0
python3 -O verify_k3_w7_gammaloc_v1.py > verify_t_O.txt     # exit 0
cmp verify_t_normal.txt verify_t_O.txt                      # byte-identical
python3 mutation_driver.py > mutation_t_normal.txt          # exit 0
python3 -O mutation_driver.py > mutation_t_O.txt            # exit 0
cmp mutation_t_normal.txt mutation_t_O.txt                  # byte-identical
```
Runtime < 10 s total. Fail-closed throughout: any failed check prints `CHECK FAILED: …` and exits nonzero; `K3W7_CERTIFICATE_PASS` / `MUTATION_DRIVER_PASS` print only after all checks.
