# W12 — INDEPENDENT ADVERSARIAL AUDIT (red team), K3 SIDE24/q0 lower-rate repair

**Workstream:** W12 (independent adversarial reviewer). **Date:** 2026-08-05.
**Method:** no upstream PASS label inherited. Every closure/estimand-transfer/inequality/all-small-r
inference/evidence-tier promotion was attacked directly: algebraic re-derivation by hand, independent
recomputation (own MC, own Isserlis implementations, own quadrature), re-execution of the fail-closed
certificates, and cross-pipeline probes (W6's engine used against W4's frozen claims and vice versa).
**Verdict vocabulary:** HOLDS / HOLDS-WITH-CLARIFICATION / BROKEN / CANNOT-ASSESS.

**Certificates re-executed by this reviewer (all exit 0, frozen-transcript-identical where applicable):**
- `W2_sanity.py` — 114/114 PASS, exit 0.
- `w6_cert.py` (mandates 1–3) — 31 checks PASS, exit 0.
- `w6_wp.py` tail grid `W6_GRID=17:21` — reproduces frozen C values 3.9035 / 3.9520 / 3.9697 / 3.9593 exactly; exit 0.
- `w10_sanity.py` — PASS, normal/`-O` byte-identical, matches frozen `t_normal.txt`.
- `verify_k3_w7_gammaloc_v1.py` + `mutation_driver.py` — PASS (1 pristine accept + 9 mutated rejects), match frozen transcripts.
- Hash bindings: W2 freeze body-sha256 `26e2d1dd…9f` verified against the file; W4 MANIFEST 39/39 OK; W6 MANIFEST 15/15 OK.

---

## HEADLINE BREAKS (details in per-artifact sections)

| # | severity | break |
|---|---|---|
| B1 | **HIGH** | W6's WP-min modulus `E_WP(r) = 4.2·r^{3/2}` on (0, 0.05] is **invalid for r ≲ 0.003** — self-registered by the campaign (DECISION_LEDGER final entry, FAILED_APPROACHES #13) and independently corroborated here: the frozen `I_CS` quadrature truncates at `r_lo = 0.06`, and the report's justification ("collar neighborhoods contribute ≤ 1.5e-18") is a **category error** — that deadness holds for the exact ρ, not for ρ_CS, which carries a near-cluster singularity `ρ_CS ~ ℓ^{1/2}·d^{−4}` (scaled coefficient 3.97 → 6.29 and climbing at r = 0.0015 per the campaign's refined computation). The WP-min lemma of W10 §5.6 is therefore **not delivered** by the frozen artifact; gate G6 cannot pass on it. |
| B2 | **MODERATE** | LEAD_SYMBOLIC_WP §3(A)'s Isserlis display is **false as stated for nonzero means** (missing `−2·ν_iν_jν_kν_l`; overcount exactly `2·(det ννᵀ)²`, counterexample verified 7 vs 5). The campaign's canonical witness absolutes **1.31310e-4 / 2.30247e-2 are inflated by exactly the resulting factor √1.34488 = 1.159689** (reproduced to all six printed digits). Correct values: **1.13229e-4 / 1.98542e-2**. The ratio 175.3 (my independent value 175.35) and the whole qualitative defect-confirmation D1 are intact; the direction is conservative (inflates upper bounds, never deflates). Record correction owed wherever the absolutes are quoted (00_READ_FIRST fact 1, CANONICAL_STATE, DECISION_LEDGER, W10 §5.3, G1 text). |
| B3 | **MODERATE** | W6 premise **F1 is mis-quantified**: `3.9697·2^{0.327/4} = 4.20114 > 4.2` (report prints "= 4.200 ≤ 4.2"); the true 2^{1/4}-grid threshold is |slope| ≤ 0.32544, not 0.327. Worse, the rung grid is **not** 2^{1/4}-uniform in the tail (gaps 0.0026223/0.002 = 1.3139, 0.002/0.0015 = 1.3333, 0.0015/0.001 = 1.5000); under the stated premise the worst inter-rung sup of C is **4.236 > 4.2** (gap (0.001, 0.0015)). Under the *measured* slope 0.20 the worst sup is 4.129 — the conclusion survives only after restatement, with margin 1.7%, not the advertised 5.8%. |
| B4 | **LOW–MODERATE** | W7's headline "P-NMZ-γ **retired from the load path**" conflicts with W10's own recomposition, which still carries the exit channel as "theorem-grade form conditional on P-NMZ-γ (C-1)" (§3 row 5) and "A-exit (γ-LOC terminal clause — premise P-NMZ-γ (C-1), or any other explicit exit modulus)" (Form A). Under W7's own falsifier F4 a consuming node must be named or the disposition reverts; the consuming node is **exit (B4)**. Minimality is verified for the branch₂/terminal-value node only. |

Everything else attacked held; see below.

---

## 1. W2_symbolic (W2_DERIVATION.md + W2_sanity.py) — **HOLDS**

Attacks executed and defeated:

- **Isotropic closed forms (falsifier F1 of W2).** Independent 2e6-draw MC in (A,B,C) coordinates:
  E[|det|1{det<0}|u] vs √2·e^{−u²/4}: u = 0/0.7/1.2/2.0 → deviations −0.2/+0.3/−1.6/+0.6 MC-se;
  P(det<0|u) vs 2^{−1/2}e^{−u²/4} matches; E[|det|1{H≺0}|1.2] MC 1.41410 vs printed quadrature 1.41363. **Holds.**
- **Chernoff/Bernstein tails (W2-14/W2-15).** Re-implemented from scratch (own Cholesky/eigendecomposition,
  own θ-grid inf, own MC, 14 instances including the rare-event regime δ > 0, p ∈ [4.7e-2, 0.96]):
  every bound ≥ true probability; W2-15 ≥ W2-14 always (as proved); Cantelli valid. The W2-15 exponent
  algebra was re-checked line by line: the admissibility θ* ≤ 1/(4c_neg) ⟺ 0 ≤ 2V; the final drop
  −δ²(V+4cδ)/(4(V+2cδ)²) ≤ −δ²/(4(V+2cδ)) ⟺ V+4cδ ≥ V+2cδ ✓; the log(1+x) ≥ x−x²/2 and the
  y ≤ 1/2 series bounds ✓. **Holds.**
- **Conditioning order (C7).** Two-stage ≡ one-shot is the block-Schur identity; verified to 1e-16 by W2's
  own P4 and consistent with W4/W6's independent pipelines (probe cross-matches below). No conditioning-order
  error found. The derivative sign convention C2 and the Kac–Rice Jacobian convention C6 (|det|, multiplicity 1)
  are correct; det = zᵀMz with M signature (1,2) ⇒ Sylvester one-positive/two-negative eigen-structure ✓.
- **Box+Šidák lower bound (W2-17).** Vertex-min step re-verified structurally: μ_{F|H}(h) affine ⇒ Δ convex ⇒
  max at a vertex; Pw decreasing in Δ (symmetric unimodal density on a fixed symmetric interval) ✓; Šidák applies
  (box centered at the slice mean) ✓. Note: this is the instrument for the *formal* refutation of 0.213r³ and it
  has **not yet been integrated over a subregion by anyone** — the refutation remains estimate-tier until W3 does so.
- **Typing/Jacobian slips:** none found. The honesty ledger (V1–V12 / I1–I4) is accurate, including the
  I2 warning (isotropic independence destroyed under pins; |off-diag| sum 0.379) — independently confirmed by
  LEAD's pipeline and by the W6 law6() covariances used in my probes.
- `W2_sanity.py` re-run: 114 PASS, 0 FAIL, exit 0; freeze body hash verified byte-exact.

**Residual gap (not a break):** the "sup-over-zone" quadrature step is explicitly deferred to W3 (named formality).

## 2. LEAD_SYMBOLIC_WP.md — **BROKEN IN PART (one display); otherwise HOLDS**

- **B2 (the break).** §3(A): "E[det²H] = M(00,22) − 2M(01,12) + M(11,11), M(ij,kl) = C_{ij}C_{kl} + C_{ik}C_{jl}
  + C_{il}C_{jk}" with C = Σ_H + ννᵀ. The Gaussian fourth moment in second-moment form requires
  `−2·ν_iν_jν_kν_l`; without it the formula overcounts by `2·(det ννᵀ)²` in the det-combination.
  Executable counterexample (this review): z ~ N((1,0,1), I₃): MC E[det²] = 4.987, printed formula = 7.0,
  overcount 2.01 ≈ 2·(1·1−0)² = 2 exactly.
  **Consequence traced to the canonical record:** at the witness (−0.04, −0.58), r = 0.025, my independent
  evaluation (W6 engine, exact Isserlis) gives dropped-root 1.13229e-4 and CS-valid 1.98542e-2; applying the
  LEAD-formula inflation √1.344878 = 1.1596887 reproduces the canonical 1.31310e-4 to all six printed digits
  (0.0001313104 computed vs 1.31310e-4 printed). The 175.3× ratio (mine: 175.35) and the defect confirmation
  are **unaffected**; both absolute values in 00_READ_FIRST.md fact 1 and CANONICAL_STATE.json are wrong at 16%.
  Direction: conservative everywhere (an upper bound with an inflated E[det²] remains an upper bound).
- **(B) u-slice factorization:** independently re-derived and verified valid — P(f∈W|H,∇=0) depends on H only
  through u = μ̃ + β·(H−ν) (β = Σ_H⁻¹c, s_u² = cᵀΣ_H⁻¹c, s_f² = ṽ − s_u² > 0); the tower argument requires only
  σ(u)-measurability of Pw(u(H)) ✓; LEAD's Pw sign convention equals W2's after the Φ(−x) = 1−Φ(x) flip ✓;
  E[u] = μ̃, Var(u) = s_u² ✓.
- **(C) per-slice bounds:** a(u) = ν + (c/s_u²)(u−μ̃), V = Σ_H − ccᵀ/s_u² ✓ (standard Gaussian conditioning);
  in-slice CS valid; "Vdet(u) u-independent up to the mean shift" is correct (dependence enters only through a(u)).
- **(D) box lower bound:** valid (same structure as W2-17).
- **G1 note:** the gate text "two independent derivations agree on … Isserlis" overstates: W2-9 (full expansion)
  is correct; LEAD's compact display is not. G1's substantive agreement (integrand, slice factorizations,
  validity classes, dropped-root invalidity) stands.

## 3. W4_independent (I_WP(0.025) = 4.7569e-6 estimate) — **HOLDS (as an ESTIMATE, with one naming caveat)**

- **Dominant-region completeness — attacked with W6's independent engine at r = 0.025 (different kernel
  representation, different E_win code) at 20+ points chosen OFF W4's grids.** Results: argmax (−0.0025, 0.580)
  ρ = 1.0729e-4 — **matches W4's ρ_max to all five printed digits**; peak (0, 0.5672): 1.0529e-4 vs 1.053e-4 ✓;
  wedge edges (65°/115°, r = 0.567): 1.4e-12/1.7e-15, consistent with "ρ ≤ 2e-12 outside [65°,115°]" ✓;
  mirror (0, −0.567): 4.7e-20 (W4: E_win 1.8e-20 — same kill class) ✓; 30°/150°/250° off-wedge rays:
  8.5e-37/5.2e-19/3.1e-8 — all ≤ W4's arc caps (≤ 2e-7) ✓; down-tail (0, −1.2): 1.4e-7 ≤ 2e-7 ✓;
  outer shell ray (0, 2.0)/(0, 2.8): 1.6e-7/9.8e-8 — magnitude consistent with the reported shell integral
  1.5866e-6 ✓; inner ray (0, 0.06–0.3): exact-ρ underflow (< 1e-300) with onset between 0.3 and 0.41
  (W4's radial profile: 1.8e-6 @ 0.41; mine 1.57e-6 ✓). **No hidden mass found anywhere.**
- **Assembly arithmetic:** pieces sum to 4.756917658e-6 vs printed 4.7569177e-6 ✓; medium-grid whole-annulus
  cross-check (4.8012e-6 vs assembled 4.7569e-6) internally consistent with the −4.11e-8 sliver ✓.
- **Corrected-CS integral:** pieces sum to 8.572785e-4 vs printed 8.5743e-4 — 0.018% display-rounding
  inconsistency, immaterial (cosmetic; C6 below). Ratio I_CS/I_WP = 180.2 ✓.
- **±10σ truncations:** with ~1e5 points at < 1e-22/point, worst-case leakage ~1e-17 ≪ 4.76e-6 ✓; the window
  is either inside the truncation or the point is genuinely dead — sound.
- **Dead collar (κ ≥ 466):** my ring probes (r = 0.01, 0.05 at r = 0.025) underflow to 0 — consistent with
  ρ ≤ 1.1e-171; not independently certified by me, but the same deadness is confirmed by W6's certified
  inner-annulus bounds (3e-224 at r = 0.01) and by the variance-law structure (v ~ r⁶, v_t ~ r⁸ at cluster scale).
- **Naming caveat (C5):** W4's "I_WP(0.025)" is the **rigidity-zone piece only** (Z = union of 1.5-disks;
  outer shell 1.5866e-6 reported separately). The full B₃∖collars estimand is W6's 7.50e-6. The
  DECISION_LEDGER reconciliation (4.7569 + 1.5866 + ~1.16 far ≈ 7.5) is arithmetically verified here.
  Any assembly printing "I_WP(0.025)" must print its domain; the prespecified G2 comparison must be same-domain.
- **Estimate-tier discipline:** W4 itself labels I_WP an ESTIMATE (±0.4%), not an enclosure; the
  remainder-domain (±50%) and sliver (±20%) terms dominate the budget honestly. No tier promotion found.

## 4. W6_uniform_r (modulus + limit jet + variance laws) — **MODULUS BROKEN (B1, B3); the rest HOLDS**

**B1 — the modulus (highest-value target).** Three independent defects converge:
1. *Self-registered (pre-review):* DECISION_LEDGER's final entry and FAILED_APPROACHES #13 record that
   `4.2·r^{3/2}` fails for r ≲ 0.003 (ρ_CS ~ ℓ^{1/2}d^{−4} near-cluster singularity on the arch direction;
   refined scaled coefficient 3.97 → 6.29, still climbing, at r = 0.0015). This post-dates the frozen report
   (b61835f0…) under review.
2. *Category error in the frozen validity bullet (verified):* "Integration over B₃ ⊇ B₃∖collars is
   conservative; collar neighborhoods contribute ≤ 1.5e-18" — the 1.5e-18 deadness is a statement about the
   **exact** ρ, not about ρ_CS. The frozen `I_CS(r, r_lo = 0.06)` excludes the ball r < 0.06 at **every** rung;
   for rungs r < 0.01828 (where 3.2825·r < 0.06) that ball contains **non-collar** near-pin annulus — precisely
   the region where the CS bound (not the truth) blows up. So the frozen rung values C(r_k) for r_k ≤ 0.0177
   are integrals over a **strict subset** of the required domain, and the omission is unquantified in the
   frozen artifact. I probed the region with the frozen `point_rhoCS` (r = 0.0015; d ∈ [2.1r, 0.06]; 6 rays +
   full angular scans around all three pins): every evaluation underflows to 0 in float64 — i.e. **the frozen
   pipeline is numerically blind to exactly the mass the self-correction reports**. Mechanism check (mine):
   near a pin the conditional gradient covariance vanishes quadratically, so (det Σ_G)^{−1/2} ~ d^{-2}-class
   blow-up of the Kac–Rice density factor; my factor probes confirm √E[det²] and the window factor do **not**
   vanish near the cluster (e.g. at r = 0.00625, d = 10r: √E[det²] = 6.5e-2, s_F = 1.7e-8) — whether ρ_CS
   survives is decided solely by p_grad's direction-dependent exponential, which is exactly the artifact the
   campaign reports. The break is real; its magnitude (6.29) is the campaign's number, not mine — the frozen
   code cannot reproduce it (that is itself the defect).
3. *What survives:* the chain E⁰[N] ≤ I_CS ≤ E_WP at **r₀ = 0.05** stands up to my attacks: at r = 0.05 the
   excluded centroid-ball {d < 0.06} lies inside the pin collars (radius 2r = 0.1; worst-case separation
   arithmetic verified) so nothing required is omitted, and my probes there give ρ_CS ≤ 5.5e-88.
   C(0.05) = 2.377 full-domain ⇒ I_CS(0.05) = 2.66e-2 ≤ E_WP(0.05) = 0.0470 ≤ 0.06 budget ✓ (also ×742
   above the truth 3.58e-5). For 0.003 ≲ r ≤ 0.0177 the omission exists but is margin-absorbed (C ≤ 3.79,
   headroom ≥ 0.4); below ~0.003 the modulus is dead as stated.

**B3 — F1 quantification (independent of B1).** Numbers in §HEADLINE. Additionally: the report's phrase
"each inter-rung interval of the 2^{1/4}-grid in (0.001, 0.05)" is internally inconsistent — the 2^{1/4}
subgrid ends at k = 17 (r = 0.0026223) and never reaches 0.001. A repaired F1 (piecewise two-sided log-slope
envelopes using the measured discrete slopes ≤ 0.20, plus the turnover) certifies ≤ 4.129; the margin to 4.2
is then 1.7%.

**F2 note:** C₀ = 3.860069 is computed from the §2 limit law — the *fixed-y* (d = O(1)) limiting conditional
law, which has the same near-cluster blind spot as the rung quadrature. F2 must be re-derived against the
repaired (zone-split) CS law.

**Limit jet — HOLDS (attacked hard).** The f_xxx → +2 derivation re-done from the pins:
(P2−P1)/r = f_x + (r²/24)f_xxx + … = −r²/6 with (P3+P4)/2 = f_x + (r²/8)f_xxx = 0 ⇒ −(r²/12)f_xxx = −r²/6 ⇒
f_xxx → +2, f_x(0) = −r²/4 ✓ — the gradient-pin drift is correctly accounted; the naive difference-quotient
value −4 is indeed the trap (it assumes f_x = 0). The printed 10-component limit mean vector is internally
consistent: A3 = 0.50000000002 (claimed 1/2), D3 = 10.4994085490 vs (1.52−2c)/0.24 = 10.4994085393 (dev 1e-8),
annihilator n·jet = −2.7e-7 ≈ 0, residual-variance ratios 110.25 : 4853.4 vs 10.5² : (209/3)² ✓.
c = −3476069/6953125 = −0.4999290247191011… ✓ (exact rational, independently reproduced by W6's pipeline to
1.5e-13 and matching W4's 25-digit μ_t: (μ_t−b)/ℓ = −0.49971601 at r = 0.025 vs W2's −0.4997159 probe ✓).

**Variance laws (v_t r⁰ fixed-y / r⁸ cluster):** not independently re-derived by me (certificate PASS, exit 0,
31 checks); the refutation of "v_t ∝ r⁴" is consistent with the z-drift transient I observe in the P2 factor
display (P_W/r³: 0.00095 → 9.51 over the rungs).

**84-sample validity guard:** adequate *as an implementation guard* (the pointwise inequality is W2-7, a
theorem — validity does not rest on samples); but note the samples are wedge-only (60°–120°) at two rungs
(0.025, 0.00625) with `rx > 0` skipping underflowed points. Fine for the guard's purpose; it does not extend
to the quadrature (F3, named open) or to the r_lo truncation (B1).

## 5. W7_gammaloc (DISCHARGED-BY-MINIMALITY + r₀ addendum) — **HOLDS-WITH-CLARIFICATION (B4)**

- **Minimality for the terminal-value node: verified.** C025's R2 decomposition (quoted in the audit) bounds
  P(terminal ≤ b) by a near count + an r-free far constant; W10's §2.4 recomposition consumes exactly that
  (constant AO floor c_cond = 0.089569·P₀, no exp rate). No 1−O(exp) consumption exists at the branch₂ node
  in the assembly of record. The r₀ dominance algebra is exact and I reproduced every displayed number:
  2.1·(ℓ/2) = 7r³/40 = 0.175r³ ✓; r₀³ = 20κ/7·P₀ = 0.25591142857142857·P₀ ✓; r₀ = 0.634887184·P₀^{1/3}
  (my cube root: 0.634887184018) ✓; r₀(P₀ = 3.2e-4) = 0.043425672541 vs printed 0.04342567254 ✓ (the 0.05
  rung correctly flagged uncovered); r₀(1e-4) = 0.0294688526394 ✓; the 19/9-vs-2.1 coefficient sensitivity
  (0.53%) is honestly disclosed ✓. Verifier + mutation driver re-run: PASS, byte-identical to frozen.
- **B4 — the overreach.** "P-NMZ-γ is retired from the load path" is true of the branch₂/terminal node but
  **false of the exit node (B4)**: W10 §3 row 5 keeps exit's theorem-grade form "conditional on P-NMZ-γ (C-1)"
  and Form A carries "A-exit (γ-LOC terminal clause — premise P-NMZ-γ (C-1), or any other explicit exit
  modulus)"; D3's named conditional set includes C-1. W7's own F4 ("if the recomposed assembly consumes a
  1−O(exp) rate at any node … name the consuming node") is therefore live: the consuming node is exit.
  Resolution required before minting: either (i) re-scope the disposition explicitly to the terminal-value
  node and keep C-1 carried for exit (W10's D3 already does — then the joint record must stop saying
  "retired from the load path" without qualification), or (ii) supply a non-γ-LOC explicit exit modulus
  (exit is measured 0/1750 per rung; a rung-cert + inter-rung formality like LB-2's is the natural route).
- **Falsifier harness:** 1 pristine accept + 9 mutated rejects — verified PASS; the verifier is not vacuous.

## 6. W10_dependency (composition theory) — **HOLDS**

- **F0–F3 exhaustiveness — attacked with the full merge-case analysis.** Given a.s.-Morse, a.s. distinct
  critical values, and the Morse–Smale conclusion: at S's level either (i) M's component already merged with
  an older component in (b−ℓ, b) → F1; (ii) S borders M's component, other side ≤ b → F3; (iii) S borders,
  other side > b → success; (iv) S does not border → F2 (any later merge below b−ℓ is D(M) ≠ S via F2);
  degenerate/tie cases are null → F0. Merges of M's component with *younger* components never kill M
  (elder rule) and need no mode. **Exhaustive.**
- **Q(y) ⊆ F1 inclusion:** mountain-pass/elder-rule structure is sound; the hypotheses (R1 monotone ascent,
  B3 conditional mountain pass, B4 separate) are named at their grades.
- **Bonferroni:** P(N ≥ 1) ≥ E[N] − ½E[N(N−1)] ✓ standard; Campbell/Palm disintegration of E[N_qual] ✓.
- **η_r firewall:** the LB chain lower-bounds 1−p_r directly via a sub-event of {D(M) ≠ S}; no conditioning
  on A, no typed-to-adjacent transfer — the "no η_r loss" claim is correct for this route; the identity
  p_r = a_r q_r^adj + η_r and the upper-side load-bearing identity I_r q_r^adj = J_r(p_r − η_r) are correctly
  placed (upper chain out of scope). The ω-limit-vs-adjacency-mark distinction (arch "non-adjacency" ≠ A_f(M,S))
  is correctly maintained.
- **WP-min spec (§5.6):** correctly states the assembly's need (any valid explicit vanishing modulus with
  interval coverage + budget). Note this spec is exactly what B1 fails to deliver today — W10's architecture
  is unaffected; only the WP node's status is.
- `w10_sanity.py` re-run: PASS both modes, byte-identical, matches frozen transcript. Its S4/S5b arithmetic
  (budget 0.06658 at r₀ = 0.05; |Δc| = 1.79e-5; c(r₀) insensitivity 0.91434–0.91440) verified in output.
  (§5.3 quotes the inflated witness absolutes — inherited B2, non-load-bearing.)

## 7. W9 scope audits — **HOLD (transcript-verified; not re-executed)**

- **W9a (DER-027a):** scope table is internally consistent and honestly tiered: variance channel now
  interval-grade at all 12 knots incl. d = 3 (Δ̄(3) ≤ 0.0209/0.0217 ≤ 2.24e-2 with rigorous lower bounds
  0.0175/0.0186 — the pads are derived); the uniform continuation Ê_unif on (0, 1/20] is **valid but honestly
  weaker** (0.61 at d = 3 vs 0.021 exact-rung; the jet-frame projection removes more variance than the 7-pin
  projection — disclosed, not hidden); v* = −1/2 isolation certified at interval grade with the value-freeness
  mutation M1; the mesh cap concerns only between-knot envelope, carried by the DPS80 certified-constant
  envelope. Transcript tail: "ALL CHECKS PASS" (94 checks, both modes byte-identical). The named residual
  (continuation to r > 1/20, closed-form d = 3 constant) is correctly scoped as open/sharpness.
- **W9c (DER-027c):** r-uniform [1e-6, 0.05] certified at **house grade** with the C3 mesh cap
  (1.25 × 9-point mesh max of ‖m‴‖_H) explicitly named as the one non-exact ingredient; the interval tier is
  reported CLOSED-NEGATIVE with a measured, explained obstruction (interval pivots straddle zero; viability
  needs |I| ≲ r⁸) — this is the honest outcome, not a concealed failure; the 36-rung exact ladder to
  1.455e-12 and the converging margins (λmin → 2.5654/1.9428) support the "conditioning, not topology"
  diagnosis. Transcript tail: "CERTIFICATE PROGRAM COMPLETE: all checks passed"; four mutation receipts exist
  and are appropriately sensitive. The one promotion risk — "PROVED [UNIFORM-R]" — is adequately fenced by the
  H3 hypothesis line naming the mesh cap a formality.
- **CANNOT-ASSESS component:** I did not re-execute the 94-check and 73-check audit programs (heavy); my
  verdict rests on transcript integrity (PASS tails, byte-identical modes, manifest hash checks) plus
  internal-consistency review. A full reproduction is W13's job.

## 8. W5_forensic (provenance) — **HOLDS**

- The two load-bearing reconstructions are arithmetical and I verified them: lam_sad = φ₂·φ_f·ℓ·Es =
  5.033320408862532e-16 (15-digit ledger match) ⇒ rigidity = ×π·1.5² = 3.557844544420372e-15 → "3.6e-15"/"~3e-15" ✓;
  WP total = 3.5578e-15 + 1.5528090728834476e-07·4π + 0.030449079155158258·ℓ·2.75π·2 = 3.3214276e-6
  = 0.21257137·r³ vs ledger 0.21257114 ✓; rung ratios 7.7559/8.1897 vs printed 7.76/8.19 ✓; "1.28·ℓ"
  = 0.212571×6 = 1.2754 ✓.
- Every provenance step that matters rests on **quoted bytes** (verbatim lines with sha256) or on the agent's
  own independent re-derivation (station row to 12–14 digits on the variance entries), not on inference.
  The one inference-flavored step (the 0.1–5% deviations on phi2/phif/Es attributed to v* sensitivity) is
  explicitly labeled and non-load-bearing.
- The key structural finding (single-station × full-disk-area; d1 G-F2a gate FAILED-AS-WRITTEN and waived;
  kill weakens toward the zone boundary so d = 1 is not a demonstrated sup) is documented against the
  program's own named register ("sup-over-zone", "station-density/on-grid") and is consistent with W2 §0's
  independent reading of the same quotes. No promotion found: W5 itself grades the rigidity figure
  "an exact product of a station measurement and a zone area whose zone-wide validity is an open named
  formality".

## 9. Gate register (CANONICAL_STATE.json) + SUCCESSOR_SHELL_NONCONTROLLING.md — **HOLDS-WITH-CLARIFICATION**

- **Register:** G0/G1 PASS entries are supported (with the G1 Isserlis caveat of §2/B2). G2/G6 are
  PRESPECIFIED-pending as stated; no gate is over-claimed. Controlling fact 1 quotes the inflated witness
  absolutes (B2) — correction owed; the fact's substance (missing sqrt ⇒ invalid; 175.3×) stands.
- **Shell:** language audit — "NOT A THEOREM; NOT SEALED; NOT FOR CITATION", every WP/Λ slot is [GATE],
  Form trajectories are conditional ("Shell's current trajectory"), the four tiers are kept distinct, and
  §2.4's I_WP ≈ 4.7602e-6 is labeled "derived-on-grid… formal refutation pending [GATE]". **No language
  reads as a completed theorem.** One staleness note: the shell's γ-LOC row prints "DISCHARGED-BY-MINIMALITY"
  without the B4 exit-node qualification, and its Form A hypothesis list ("exit measured carried at its tier")
  is honest about exit's grade — keep it that way.
- **Noncontrolling status: maintained.** No Boolean/status claims found in any reviewed artifact.

---

## CLARIFICATIONS REQUIRED BEFORE THE SUCCESSOR ASSEMBLY MAY BE MINTED

- **C1 (blocking, from B1):** complete the W6 repair (zone split: UB_slice near-cluster + global CS outside,
  or a converged-CS law) and re-certify the modulus at **converged quadrature including the near-cluster
  region**, at the smallest rungs; alternatively, restrict the modulus's claimed interval to where the
  collars provably cover the r_lo truncation (r ≥ 0.0183 by the 3.2825·r < 0.06 computation) plus a
  margin-absorption argument — and say so. Until then the WP-min lemma (W10 §5.6) is **undelivered** and
  G6 stays OPEN. (E_WP(0.05) = 0.047 ≤ 0.06 survives scrutiny; the small-r coverage does not.)
- **C2 (from B2):** fix LEAD §3(A) (add `−2ν_iν_jν_kν_l` or restate via centered moments); re-issue the
  canonical witness numbers as dropped-root **1.13229e-4** / CS-valid **1.98542e-2** (ratio 175.3 unchanged)
  in 00_READ_FIRST fact 1, CANONICAL_STATE, DECISION_LEDGER, W10 §5.3, and the G1 register text.
- **C3 (from B3):** restate F1 against the true grid (piecewise two-sided slope envelopes; threshold
  0.32544 for a uniform 2^{1/4} grid; the tail gaps 1.3139/1.3333/1.5000 handled explicitly); re-derive F2's
  C₀ from the repaired CS law (the current 3.860069 inherits the fixed-y blind spot).
- **C4 (from B4):** reconcile W7's disposition language with W10's C-1 exit clause — scope the minimality to
  the terminal node or produce a non-γ-LOC exit modulus; keep the F4 watch active for the exit node
  (OPEN_OBLIGATIONS #14 already does — make the W7 headline match it).
- **C5 (from §3):** print the domain with every WP-truth figure (rigidity-zone 4.7569e-6 vs full B₃∖collars
  7.50e-6); run the prespecified G2 comparison on the same domain.
- **C6 (cosmetic):** W4's I_CS piece-sum display (8.572785e-4 vs printed 8.5743e-4).

## OVERALL DISPOSITION — the campaign's strongest defensible result

1. **Strongest:** **Form C (liminf) at measured/mixed tier** — liminf (1−q)/r³ ≥ AO⁰·c_Λ^∞ (0.9144-class).
   Its WP requirement is only WP(r) → 0, which is supported by two independent integrators (W4 4.7569e-6,
   lead 4.7602e-6 — 0.07% agreement — on the rigidity zone; W6 7.50e-6 on the full domain, domain-reconciled)
   and by the rung table I_WP(r) → 0 with a decade slope 1.45–1.6 whose transient mechanism is understood
   (z-drift; W6 §4 verified in structure). The remaining Form-C inputs carry their own named conditionals
   (H-MS, C-6, H-B3, exit) — none WP-related.
2. **Evidence-tier refutation of 0.213r³: solid as evidence, not yet formal.** Rigidity-zone estimate alone
   (4.7569e-6) exceeds the falsified budget (3.328125e-6) by 1.43×; full-domain estimate by 2.25×. The
   *formal* refutation needs W2-17 integrated over a certified subregion (W3's pending enclosure) — every
   ingredient for it is verified valid here.
3. **Explicit-r₀ theorem (Form A): NOT currently closable on the WP node** — the frozen modulus fails at
   small r (B1); the repair path is identified and the r₀-side value is intact. This is the campaign's one
   material open load path.
4. **The D1 defect-confirmation itself survives this audit in full** (ratio exact, direction exact, all
   downstream quarantines justified); only its quoted absolute values need the 16% correction (B2).

*End of W12 audit. Hash block below (computed at delivery).*
