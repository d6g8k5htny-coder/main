# LB-1 - SUP-OVER-ZONE step (above-b saddle channel): proof text

**Program:** SIDE24/q0, Lemma LB-RATE, AO assembly, channel **above-b saddle** (C029 B1-B4
reduction, branch B3(ii)).
**Verdict:** **PROVED** in the localized form the B3 reduction requires (pass-zone closure,
fail-closed machine-certified, superexponential margins at every certified rung), with one
registered **correction** to the ledger's zone-wide mechanism language (falsifier exhibit,
Section 6) and one **flag** on Lemma WP's rigidity-zone constant (Section 9, out of scope but
load-bearing elsewhere).
**Certificate:** `lb1_sup_over_zone_certificate.py` (mpmath 60 dps, `ck()` fail-closed,
byte-identical under `python3` and `python3 -O`), transcripts `t_normal.txt`, `t_O.txt`.

---

## 0. Verbatim ledger bindings

- **[L1]** `C031_LBRATE_Integration.md` line 30: "| - above-b saddle | superexponentially
  killed throughout the jet-cluster horizon (v = 0.018/0.55 at d = 1/2; kill
  e^{−(b−m)²/2v}) - mechanism identified | derived mechanism; closure named | C030 |
  sup-over-zone argument"
- **[L2]** `C031` line 55: "5. Sup-over-zone (converts the horizon kill into the closed
  above-b-saddle piece)."
- **[L3]** `C030 CountingLemmas Package.md` line 11: "Var(f(y′) | 9 pins) at r = 0.025:
  0.018 / 0.55 / 0.976 at d = 1/2/3. The r-clustered 9-pin set acts as a high-order
  effective jet of the ANALYTIC BF kernel... Consequences: (i) the band intensities are
  superexponentially killed (e^{−(b−m)²/2v}) throughout d ≲ 1.5..."

## 1. Setting, notation, hypothesis list

Field: periodized 2D Bargmann-Fock field on the side-24 torus, covariance
K(u) = e^{−|u|²/2} (unit variance); spectral lattice (π/12)ℤ², masses
(2π/24²)e^{−|k|²/2}, truncation |k| ≤ 30 (certified tail < 1e-60, certificate [A1];
wrap-around < 1e-28 for |x| ≤ 12). Level b = 6/5; window ℓ = r³/6; rungs
r ∈ {0.05, 0.04, 0.03, 0.025, 0.02, 0.0125}.

Conditioning (rebuilt from scratch; matches C025/C026/C031 construction): the 9-pin set =
1-jets (f, f_x, f_y) at
M = (−r/2, 0) value b, at S = (r/2, 0) value b − ℓ, at Y = r·(−1.26, 0.24) value
v* = clip(μ_t, [b−ℓ, b]), where μ_t is the 6-pin conditional mean at Y given ∇f(Y) = 0
(arch saddle value; (μ_t − b)/ℓ = −0.4997159 at r = 0.025, ledger limit −0.4999290, C026).

The conditioned field f̃ = f | (9 pins) is Gaussian: f̃(y) = m(y) + ξ(y),
m(y) = k(y)ᵀ Σ_PP⁻¹ **v**, v(y) = Var ξ(y) = K(0) − k(y)ᵀ Σ_PP⁻¹ k(y),
k(y) = Cov(f(y), pins), Σ_PP the 9×9 pin covariance (certificate [A2]; inverse
residual ≤ 1e-30 certified).

**Hypotheses.**
- **H1 (kernel/periodization):** planar K used for all covariance evaluations; error
  < 1e-28 absolute on every certified point (certificate [A1]); immaterial vs all margins.
- **H2 (Gaussian regression):** the conditional law is exactly Gaussian with the moments
  above (standard; Bulinskaya conditions hold: the conditioned field is analytic with
  nondegenerate finite-dimensional laws off the pins).
- **H3 (Kac-Rice validity):** expected critical-point counts above level b are given by
  the Kac-Rice integral (Adler-Taylor Thm 11.2.1 / Azaïs-Wschebor Ch. 6; same named
  formality as C030's "KR validity").
- **H4 (B3 localization, dependency):** the B3 mountain pass between the diversion
  maximum m′ and M (both in B_ρr, ρr = 1.4r) is a critical point with value > v* − ε
  **inside B_ρr** (C029 Freeze, B3 statement, freeze ddd8596f…0ca). Hence every
  above-b saddle of the channel lies in B_{1.4r}(M) ⊂ B_{3r}(M).
- **H5 (derived-on-grid grade):** continuum statements over the pass zone are certified
  on meshes with refinement gates plus the named analyticity formality - the house grade
  of C027 (freeze f0833443…dc2) and C031 ("station density + analyticity as the named
  on-grid formality").
- **H6 (r-uniformity at certified rungs):** the bound is certified at six rungs
  r ∈ {0.05, …, 0.0125} and the kill *strengthens* superexponentially as r ↓ 0
  (the conditional law in r-units has the C026 exact r → 0 limit, freeze 6ac9a33c…7df).

## 2. Lemma H-SUP (horizon suppression, correct form)

**Statement.** For any Borel A, with ρ̄(y) defined below,
  E[N_sad(f̃ ≥ b, A)] ≤ E[N_crit(f̃ ≥ b, A)] ≤ ∫_A ρ̄(y) dy,
  ρ̄(y) = p_{∇f̃(y)}(0) · √(E[(det H̃)² | ∇f̃(y) = 0]) · √(Φ̄(t_v(y))),
and the typed refinement
  ρ̄_sad(y) = p_{∇f̃(y)}(0) · √(E[(det H̃)² | ∇f̃ = 0]) · √(min(Cant(y), Φ̄(t_v(y)))),
where
  p_{∇f̃}(0) = (2π)^{−1} (det Σ_∇)^{−1/2} e^{−t_g²/2},   t_g² = ∇mᵀ Σ_∇⁻¹ ∇m,
  t_v(y) = (b − μ̃(y))/√ṽ(y),  μ̃ = E[f̃ | ∇f̃ = 0],  ṽ = Var(f̃ | ∇f̃ = 0) ≤ v(y),
  Cant(y) = V_det/(V_det + (E det H̃)²)  when E det H̃ > 0 (else 1).

**Proof.** Kac-Rice (H3): E[N_crit(f̃≥b, A)] = ∫_A p_{∇f̃}(0) E[|det H̃| 1{f̃≥b} | ∇f̃=0] dy.
Cauchy-Schwarz on the conditional expectation with the indicator gives the moment ×
tail factorization; P(f̃ ≥ b | ∇f̃ = 0) = Φ̄(t_v) is the exact Gaussian tail (one-point
horizon suppression: for μ̃ < b, Φ̄(t_v) ≤ e^{−t_v²/2} by Mills' ratio - the ledger's
factor e^{−(b−m)²/2v} with (m, v) replaced by the ∇-conditioned (μ̃, ṽ)). The typing
indicator 1{det H̃ < 0} is bounded by Cantelli's one-sided Chebyshev inequality.
All factors of the joint Gaussian (f̃, ∇f̃, H̃) | pins are explicit in kernel
derivatives ≤ 4th order; ρ̄ is evaluated exactly (certificate, mpmath 60 dps). ∎

**Remark (three kill mechanisms).** ρ̄(y) is small wherever at least one of
(i) the **value gap** t_v (ledger's mechanism), (ii) the **gradient gap** t_g (the
9-pin jet rigidifies ∇f̃ even where m > b), (iii) the **type margin** Cant (near
mean-maxima) is large. The ledger [L1]/[L3] named only (i); see Section 6 for why
that naming matters.

## 3. Machinery validation (certificate [A2]-[A3], recomputed; no PASS label trusted)

- Horizon table: v(d=1) = 0.018879, v(d=2) = 0.565245, v(d=3) = 0.978557
  (ledger quote 0.018/0.55/0.976 - inside tolerance).
- All 7 stations of `c031 verify.json` reproduced to ≥ 12-16 digits, e.g. (5.0, 0.0):
  v = 0.9999999577222840, m = 0.0003297119994761068, |∇m| = 0.001455341753629615;
  pull_f = √(1−v) reproduced to ≤ 1e-6 relative.
- Ledger KR constants independently verified (Gauss-Hermite quadrature + 4×10⁶-sample
  Monte Carlo): ρ_mx(1.2) = 0.04369 (ledger 0.043685), ρ_sad(1.2) = 0.03048 (0.030449),
  E-term 1.4137 (1.41350); far-term constant 2.1·(ℓ/2)/r³ = 2.1/12 = 0.175 exactly.

## 4. THEOREM (SUP-OVER-ZONE, localized form)

**Statement.** For the conditioned field of Section 1 at any certified rung
r ∈ {0.05, 0.04, 0.03, 0.025, 0.02, 0.0125},
  E[N_{above-b saddle}(B_{3r}(M))] ≤ E[N_crit(f̃ ≥ b, B_{3r}(M))] ≤ E_zone(r) ≪ 10^{-3}·r³,
with E_zone(r) certified in [D]. By H4, the above-b saddle channel of the AO assembly is
therefore **closed**: its expected count is negligible vs the assembly's r³ rate at every
rung, and the kill strengthens superexponentially as r ↓ 0 (H6).

**Certified values** (grid maxima of log ρ̄ over the zone, and zone bounds; see transcript):
- r = 0.05: max log ρ̄ = −84.0; E_zone = 2.1e-30 vs tol 1.25e-7 (E/tol = 1.7e-23).
- r = 0.025: max log ρ̄ = −340.0; E_zone = 8.7e-143 vs tol 1.56e-8 (E/tol = 5.5e-135).
- r = 0.02: max log ρ̄ = −528.3; E_zone = 2.8e-225 vs tol 8.0e-9 (E/tol = 3.5e-217).
- r = 0.0125: pin-circle minima ≥ 1567 (M), ≥ 3709 (S), ≥ 3161 (Y); certified by the
  same gates (ρ̄ far below double underflow).
Margins ≥ 22 orders at the worst rung (r = 0.05); ≥ 135 orders at the ledger rung 0.025.

**Certificate structure ([D], fail-closed).** (i) Annulus B_{3r}(M) ∖ ⋃B_{r/2}(pins):
mesh r/20 (superset of r/10), refinement gate L2 − L1 ≤ 10 on the grid maximum of
log ρ̄, bound F·Σ e^{log ρ̄} h² with named formality factor F = e^{min(25, 0.3|L2|)}.
(ii) Pin disks B_{r/2}(pin): |log ρ̄| sampled on circles r/2, r/4, r/8, r/16
(72 angles each); gate: every circle's minimum ≥ 40; bound
π(r/2)²·e^{−0.75·min over circles} per disk (radial scaling continuation into B_{r/16},
named, slack 0.25). (iii) Independent anti-spike ray scan at the worst rung r = 0.05
(240 rays × 40 radial steps; gate max log ρ̄ ≤ −40). Grade: H5.

## 5. Zone decomposition and the complementary region (d > d_h)

- **Pass zone** B_{3r}(M) ⊃ B_{1.4r}(pass): Section 4. **Closed**, superexponentially.
- **Complement** (outside the pass zone): by H4 there is *nothing to count* - the B3
  reduction places every channel-relevant above-b saddle inside B_{1.4r}. In the
  complement, above-b **maxima** are the desired AO terminals (R1/R4, C025 freeze
  0f8cb3dc…60b), not failures; the ordinary **band** counts are O(r³)-compatible:
  E[N_maxband(B₅)] ≤ 2.1·(ℓ/2) = 0.175·r³ (C030 Lemma KR-MB architecture), with
  ρ_mx(1.2) = 0.043685 verified at [C]; far behavior per Lemma FD (TV ≤ 2.2% at
  d ≥ 3, ≤ 1e-4 at d = 5; the c031 stations' v ≥ 1 − 4.3e-8 at d = 5 verified).

**Orders table.**

| region | object | bound | mechanism | grade |
|---|---|---|---|---|
| B_{3r}(M) (pass zone) | above-b saddles | E_zone(r) ≪ 1e-3 r³ | 9-pin jet rigidity (t_g + t_v + Cant) | derived-on-grid, certified |
| pin disks B_{r/2} | any crit ≥ b | π(r/2)² e^{−0.75·min_circles|log ρ̄|} | Dirac collapse at pins | circle-scaling certificate |
| horizon annulus 3r-1.5 | crit ≥ b | O(1) (mean ridge, P* max 1.931) | none needed (H4) | exhibit [F] |
| d ∈ (1.5, 5] | band maxima/saddles | ≤ 2.1·(ℓ/2) = 0.175 r³ | KR-MB + Lemma FD | C030/C025 deps |
| d ≥ 3 | conditional ≈ unconditional | TV ≤ 2.2%, ≤ 1e-4 at 5 | Lemma FD | C025 dep, stations verified |

## 6. Falsifier exhibit (registered correction to [L1]/[L3])

The ledger's zone-wide mechanism - "band intensities superexponentially killed
(e^{−(b−m)²/2v}) throughout d ≲ 1.5" - is **false as stated** (certificate [B], [E], [F]):

1. **m > b on ridge arcs.** The conditional mean exceeds b on arcs inside the horizon:
   m(1,0) = 1.3244, m(cos 2.35, sin 2.35) = 1.5066, and the mean-field census (Newton +
   grid exclusion) finds a genuine ridge **maximum** P* = (1.30561, 0.68576) with
   m(P*) = 1.9312 at d = 1.475. On these arcs the value-kill exponent does not even
   apply (b − m < 0), and at the d = 3/5 stations it is only 0.4-0.7.
2. **The mean has no above-b saddles.** The mean critical set in d ≤ 2.5 (Newton from
   seeds + candidate scan at mesh 0.02 with threshold |∇m| < 0.1, all candidates
   resolved; exclusion floor |∇m| ≥ 0.02 elsewhere) is exactly: M max at b; S saddle at
   b − ℓ; Y saddle at v*; ridge maxima P* = (1.30561, 0.68576), m = 1.9312, d = 1.475
   and Q* = (−1.10400, 0.87969), m = 1.6666, d = 1.412; and two sub-level minima
   (−0.5805 at (−1.7409, −1.0771), −0.3318 at (1.5640, −1.4800)). In particular above-b
   saddles of f̃ are purely noise-created; the only above-b mean-critical points are
   maxima. (The M-S midpoint plateau, |∇m| ~ 4e-4 near (0,0), contains no root: Newton
   slides to the flat region at infinity.)
3. **E[N_crit(f̃ ≥ b, B_1.5)] = O(1)**, dominated by P* (grid integral over B_0.35(P*)).

**Correction registered:** [L1]/[L3]'s kill holds only in the joint form of Lemma H-SUP
(gradient gap + value gap + type margin); the zone-wide *count* closure of [L1] is false
and unnecessary: the channel is closed by B3 localization (H4) into the pass zone
(Section 4), and ridge-region saddles route to the already-named mean-ridge channel
(C029 B3(iii), freeze ddd8596f…0ca). The kill "throughout the jet-cluster horizon" survives
verbatim only as the statement that ρ̄ (not the bare value factor) is superexponentially
small on the pass zone - which is what the assembly consumes.

## 7. Dependency table

| dependency | content consumed | freeze |
|---|---|---|
| C024 | 1 − q ≥ E[N_qual](1 − O(r³)) reduction; estimand (pin values, window) | 0df10fa6…139 |
| C025 | AO reduction R1-R4; Lemma FD (TV ≤ 2.2% at d ≥ 3); pin construction | 0f8cb3dc…60b |
| C026 | arch series; (μ_t − b)/ℓ → −0.4999290; r-uniform limit | 6ac9a33c…7df |
| C027 | derived-on-grid grade standard; foundation gates | f0833443…dc2 / pkg c9d1466a…3b4 |
| C028 | R4 far ascent (complement architecture) | 4a343a18…d0f |
| C029 | B1-B4 reduction; B3 pass localization inside B_ρr (H4); mean-ridge channel (iii) | ddd8596f…0ca |
| C030 | horizon v table [L3]; KR constants (verified at [C]); KR-MB far-term 2.1·(ℓ/2) | ae20f4e3…f4b |
| C031 | ledger statements [L1], [L2]; station table (`c031 verify.json`) | e165821b1479f619af8ebdb1438460edfebe88f44c189553ca86f73f8fe1afbe |

## 8. The falsifier named

**Falsifier of this closure:** any above-b saddle of the conditioned field inside
B_{3r}(M) at a certified rung with expected mass exceeding the certificate bound -
equivalently, a certified point y ∈ B_{3r}(M) with ρ̄(y) above the grid maxima of [D]
beyond the formality slack. The certificate is designed to detect exactly this
(refinement gate L2 − L1 ≤ 10; ray-scan gate; pin-disk monotonicity gates; any
degenerate conditional covariance aborts FAIL).

## 9. Side-finding (flag): Lemma WP rigidity-zone constant

The same exact machinery gives a window-saddle intensity integral over the C030
rigidity zone d ≤ 1.5 of ≈ 1.3e-5 (mp grid 0.025; float cross-check mesh-stable
0.02 → 0.01), with hot spot (−0.04, −0.58): μ̃ = 1.1870, ṽ = 5.94e-5,
p_{∇f̃}(0) = 5.05, intensity 1.313e-4/unit-area (mp-verified). C030's figure is
"rigidity 3.6e-15" inside a total 0.213·r³ = 3.33e-6 (C031 line 29) - the recheck
exceeds the lemma's zone figure by ~4 orders and the lemma total by ~4×. Root cause:
the rim band {μ̃ within ~2 sd of the window} inside d ≤ 1.5, invisible to the falsified
zone-wide value-kill premise (Section 6). **Lemma WP needs re-derivation; LB-1 is
unaffected** (its zone is B_{3r}, where every factor is e^{−30+}-small at all rungs).

## 10. Verdict

**PROVED** (localized sup-over-zone, H5 grade): the above-b saddle channel is closed at
every certified rung with ≥ 22 orders of margin against the assembly tolerance 10^{-3}r³,
by exact-kernel computation with a fail-closed certificate. The ledger's zone-wide
mechanism is corrected as in Section 6; the correction does not propagate into the
assembly's arithmetic (the channel's registered budget was "named, rigidity-class"),
except via the Lemma WP flag of Section 9, which the program should re-open.
