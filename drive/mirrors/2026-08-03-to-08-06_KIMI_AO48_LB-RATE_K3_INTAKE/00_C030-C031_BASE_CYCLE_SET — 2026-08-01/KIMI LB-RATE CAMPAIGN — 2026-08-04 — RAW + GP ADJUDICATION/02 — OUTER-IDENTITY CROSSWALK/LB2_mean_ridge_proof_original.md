# LB-2 — MEAN-RIDGE CONNECTIVITY: proof text and machine certificate

**Verdict: PROVED** (deterministic, at the program's frozen rungs r = 0.05 and r = 0.025, with the
analytic reduction valid verbatim at every r; see §8 for the exact scope and the one strengthening
that remains available but is not needed by the assembly).

**Certificate:** `lb2_cert.py` (sha256 bf76df4189033525526d6fd4a73ca15ef62095de119d114548e033d3e137175d),
`ck()` fail-closed (any check failure → `SystemExit`), normal and `-O` outputs byte-identical
(verified), runtime 26 s, transcript `lb2_transcript.txt`.

---

## 1. Verbatim ledger bindings (proved against these, not paraphrases)

- C031_LBRATE_Integration.md line 31: **"| — ridge split | — | named | C029 | mean-ridge connectivity |"**
- C031 line 56: **"6. Mean-ridge connectivity (C029 B-chain)."**
- C029 BasinPersistence Package.md line 8: **"BP-failure ⊆ {diversion to m′ ≠ M in B_ρr} ∪ {exit}. Diversion at trajectory height ≥ v* (B1, monotone) forces a mountain pass above v*−ε between m′ and M (B3, Established-Math conditional on a.s.-Morse): a window-class saddle — Λ-suppressed O(r³) by proven machinery — or an above-b saddle near the pinned pair (rigidity-class, NAMED), or a component split along the mean ridge (NAMED, against the C025-certified skeleton)."**
- C029 BasinPersistence Package.md line 18: **"(3) component-connectivity along the mean ridge (deterministic, against the C025 skeleton);"**

## 2. What the C025-certified skeleton certifies (extracted, C025 TerminalHeight Package.md)

The R1–R4 decomposition (C025 §1, verbatim): **"AO-failure ⊆ {non-adjacency} ∪ {terminal ≤ b}. (R1) Monotone ascent ⇒ trajectory height ≥ v* = b − ℓ/2 everywhere ⇒ far-zone entry within ℓ/2 of b, automatically. (R2) P(terminal ≤ b) ≤ E[N_maxband(B_d₀)] + far band-termination. (R3) Lemma FD (proven) bounds the far term by the unconditional quantity + TV (2.2% at d ≥ 3). (R4) NEW OBLIGATION OBL-FAR-ASCENT(δ₀), r-FREE."**

The certified skeleton behind "against the C025-certified skeleton":
- **Arch adjacency skeleton (P-A1′, certified):** capture at all 3 rungs (r = 0.05, 0.025, 0.0125); launch fans 27/27 across ±20° + offsets; both branch endpoints r-stable to 3 decimals in rescaled units. `c025 arch flows.json`: outward branch → outcome "M" (capture), min height gain −7.5e-9/−1.2e-9/−7.5e-9 at the three rungs (monotone to numerical precision); M-ward branch → "esc".
- **Qualification map (§4):** Λ-mass lives in two lobes at ±23° off the rear axis at τ/r = 0.75 (the arch), classified **ADJ(M)** at the mean-flow level — **"one branch → M captured, other escapes far — the LB0 geometry, deterministic"**. Off-lobe disqualifiers: rear axis Λ = 0 exactly; transverse not-saddle at the mean; S-side Λ = 0 via the Hessian-factor kill.
- **Crest/rim values (P-A2, C027 T1):** lobe Λ = 0.0845 at θ = 157°/202°, ≤ 0.0014 elsewhere, exactly 0 on the axis; the exact-limit ridge crest at ỹ ≈ (−0.6…−0.76, 0.2…0.3) with λ ∈ [0, 3.476], λ(ỹ*) = 3.4755; the q-kill wall drops λ 3.5 → 0 within 0.15 toward the rear axis; the P_win∞ transition governs the outer rim (outer-ring mass 1e-5; coarse remainder outside the lobe box 0.0016).
- **Band-max channel dead (P-A3):** ΣΛ_max/ΣΛ_sad = 8.9×10⁻⁹.
- **Window/typing:** window (b−ℓ, b), ℓ = r³/6, b = 6/5; pins ((b,0,0), (b−ℓ,0,0), (v*,0,0)); v* = clip(μ_t) with (b−v*)/ℓ → 0.4999290 (C026 exact); arch station ỹ* = (−0.76, 0.24) rescaled.

## 3. The object

m₉ = the conditional mean of the periodized 2D Bargmann–Fock field (K(u) = e^{−|u|²/2}, period
L = 24, spectral lattice (π/12)ℤ², masses e^{−|k|²/2}, truncation |k| ≤ 30, certified tail
< 1e-60 — certificate C0 recomputes: tail ≤ 2.2e-186) given the 9 pins
(f, ∇f)(M) = (b,0,0), (f, ∇f)(S) = (b−ℓ,0,0), (f, ∇f)(yy) = (v*,0,0),
M = (−r/2,0), S = (+r/2,0), yy = M + r·(−0.76, 0.24). It is the explicit entire function
m₉(x) = Σ_{j=1..9} c_j ∂^{a_j}K(x − p_j), computed in mpmath 60 dps exactly as in
`c024 lam indep.py` / `c025 meanflow.py` (Schur complement, v* = clip(μ_t)); the periodization is
a certified perturbation: Poisson images ≤ 5.8e-121 (orders ≤ 4: ≤ 8.5e-117), propagated through
the Gram inverse to |m₉^per − m₉| ≤ 8.3e-108 everywhere (C0/C2) — 90+ orders below every margin.

## 4. Theorem (mean-ridge connectivity)

Let R = B_{2r}(M) (contains the C029 trajectory ball B_{1.4r} and the arch station yy, which sits
at 0.80r from M). At r = 0.05 and r = 0.025, with certificates C1–C7:

**(A) Census.** Crit(m₉) ∩ R = {M, yy, S} exactly. M is a nondegenerate maximum, m₉(M) = b;
yy is a nondegenerate saddle, m₉(yy) = v* ∈ (b−ℓ, b) ((b−v*)/ℓ = 0.49908/0.49972);
S is a nondegenerate saddle, m₉(S) = b−ℓ. Hessian eigenvalue margins ≥ 0.014 in absolute value
(C3 exclusion + C4 Kantorovich uniqueness with uniqueness balls 0.084r/0.131r/0.187r ⊃ survivor zones).

**(B) No component split.** For every h ∈ (b−ℓ, b), every connected component of
S_h = {m₉ > h} ∩ R either contains M or has sup m₉ attained on ∂R (a boundary finger).
In particular there is **no interior superlevel component other than M's** at any window height:
a component split along the mean ridge inside the assembly region is impossible for the mean field.

**(C) Ridge.** min_{[yy,M]} m₉ ≥ v* − 2.2e-4·ℓ and min_{[M,S]} m₉ ≥ b−ℓ − 4.4e-4·ℓ with the
argmin at S (mp-certified, C5). Hence for every h < v* − 2.2e-4·ℓ the arch station yy and M lie in
the **same** component of S_h: the arch station is on M's ridge; the pair segment [M,S] lies in the
window closure and exits the window only at S.

**(D) Boundary fingers = the skeleton's far branches.** {m₉ > b−ℓ/2} ∩ ∂R is exactly two arcs,
centered at 0.135 rad (the beyond-S rise; max +7.7ℓ over the level) and 2.719 rad (the arch/escape
direction; max +9.6ℓ) — precisely the C025 skeleton's two far branches of the LB0 geometry
("one branch → M captured, other escapes far"). No other finger exists (C6).

**(E) Grid connectivity corroboration (C7).** Union-find on the certified-resolution grid
(band ≤ 0.50ℓ) at 13 heights h_k = b − kℓ/16, k = 2…14: every 4-connected superlevel cluster in R
contains M's certified interior cluster or touches ∂R — **bad = 0 at every height, both rungs.**

## 5. Proofs

**Lemma 1 (rigorous global bounds, used by the certificate).** With ‖m₉‖²_H = v·Spp⁻¹v
(certified = 13.1595532 / 13.21973646 at the two rungs, so ‖m₉‖_H ≤ 3.636), every derivative
satisfies |∂ᵃm₉(x)| ≤ ‖m₉‖_H·√(Var ∂ᵃf) for |a| ≤ 3: Cauchy–Schwarz in the RKHS,
∂ᵃm₉(x) = k_a(x)ᵀSpp⁻¹v, and k_aᵀSpp⁻¹k_a = Var(∂ᵃf) − Var(∂ᵃf | pins) ≤ Var(∂ᵃf)
= (∂¹,…)(0) ∈ {1, 1, 3, 1, 15, 3, …}. Hence G1 = 3.628 (gradient entries), G2 = 6.284 (Hessian
entries), G3 = 14.05 (third derivatives), |H|_F ≤ 10.89, |dH|_F ≤ 48.67·|dx|. These make every
grid exclusion and Taylor remainder in C3–C7 rigorous; the float64 engine carries a certified
allowance EPS_F = 3e-8 (>10× the rigorous rounding bound 2.5e-9 from Σ|c_j| ≤ 1.1e6).

**Lemma 2 (census ⇒ no-split).** Let h ∈ (b−ℓ, b) and let V be a component of the open set
S_h = {m₉ > h} ∩ R. Components of an open set are open, hence V is open and path-connected.
V̄ ⊂ R̄ is compact and m₉ is continuous, so sup_V m₉ is attained at some p ∈ V̄; since V ⊂ S_h is
nonempty, m₉(p) = sup_V > h. **Case (i): p ∈ R.** Then p ∈ S_h; the component of S_h containing p
is open and meets V (p ∈ V̄), so it equals V, i.e. p ∈ V. Then m₉ ≤ m₉(p) on the open neighborhood
V ∋ p, so p is a local maximum of m₉ and ∇m₉(p) = 0. By (A) the only critical point of m₉ in R
that is a local maximum with value > b−ℓ is M (yy and S are nondegenerate saddles — each has a
positive Hessian eigenvalue, so neither can be a local max). Hence p = M and V ∋ M.
**Case (ii): p ∈ ∂R.** Then V is a boundary finger. ∎

So a "component split along the mean ridge" — two components of the window superlevel set inside
the assembly ball, one carrying the ridge and one not — would require a second interior component,
hence by Lemma 2 a second interior local maximum above b−ℓ in R; the census (A) excludes exactly
this. For the actual (fluctuating) conditional field f = m₉ + ζ, any such split requires creating
an interior maximum above b−ℓ inside R — i.e. the "extra maxima" channel already measured at
0.0000 (C029 P-E2: extra maxima above v* in B_{1.4r}: 0/500) and counted by the WP/KR machinery
(C030) — a channel closed elsewhere, not a ridge property. That is the precise sense in which the
deterministic item "component-connectivity along the mean ridge" is discharged.

**(C) consequence for B1–B3.** By (C), for every h ∈ (b−ℓ, v* − 2.2e-4·ℓ) the arch station and M
share a superlevel component, and by (B) it is the unique interior one: the mean ridge from the
arch station to M is a single connected ridge at every trajectory height (B1's regime, heights
≥ v*, with the certified margin ε = 2.2e-4·ℓ ≪ the v*−ε of B3's mountain pass, ε ~ ℓ/4-class).
The two boundary fingers (D) are the certified skeleton's own far branches (capture/escape LB0
geometry and the beyond-S rise); they are handled by the assembly's exit and rigidity channels,
not by ridge connectivity.

**Remark (why the "naive" connectivity is false and (B) is the right statement).** The 9-pin mean
does **not** decay outside the pair: the r-clustered 9-pin set acts as a high-order jet (the C030
prediction horizon) and extrapolates a phantom landscape — certified local maxima of m₉ at
≈ (1.343, 0.650) with m₉ = 1.940 and ≈ (−1.166, 0.857) with m₉ = 1.671 (both > b), joined to the
pair region by fingers whose passes sit below b−ℓ. Consequently {m₉ ≥ h} in the whole plane (and
even {m₉ ≥ h} ∩ R for h > v*) is **disconnected** — M's cap is its own island above v*. The
assembly-relevant statement is therefore not "the superlevel set is connected" but the component
alternative (B) plus the ridge containment (C), and that is what is proved. This also pins down the
meaning of "the region relevant to the assembly": R = B_{2r}(M) ⊃ B_{1.4r}, with the far fingers
entering only through ∂R as the skeleton's certified branches.

## 6. Hypotheses

- H0. The C024/C025 construction as frozen: kernel K = e^{−|u|²/2} (periodized, L = 24, tail
  < 1e-60 — recomputed, C0), b = 6/5, ℓ = r³/6, arch station ỹ* = (−0.76, 0.24), rungs
  r ∈ {0.05, 0.025}. Nothing else is assumed: m₉ is explicit and entire; no a.s.-Morse hypothesis
  is needed for this deterministic item (B3's a.s.-Morse conditionality concerns the *sampled*
  field's mountain pass, upstream of this item).
- H1. Machine arithmetic: mpmath at 60 dps for all exact quantities (residuals ≤ 1e-51 displayed);
  float64 grid engine with certified allowance (Lemma 1 bounds + EPS_F); cross-checked engines.
- No PASS labels imported: every number recomputed by `lb2_cert.py` from the kernel up.

## 7. Dependency table (names + sha256)

| artifact | sha256 |
|---|---|
| C029 BasinPersistence Package.md | b23c3d42ccc2182e9a59a4ae1b7eda8c73939f543884268a0da3e2cc6880a785 |
| C031_LBRATE_Integration.md | e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e |
| C025 TerminalHeight Package.md | d1afd84b3143ed512b0fb1cafba5c2eb6eb190b28ec7f8e789cc817c08bf792a |
| C027 Foundation Package.md | c9d1466a6b06c9a5619d21e7896aab51746bdaeb8d34d1aed5bfb498ca73a3b4 |
| C030 CountingLemmas Package.md | aed6683edaed6d483704a63cd321fcb300e329a26bf6c143042ead58a60bfc7b |
| c025 meanflow.py (construction cross-ref) | 3a909a7c4b3b973e2edf577e2689543c8ae2c204d3f7476095e4d17a8bfe537e |
| c025 arch flows.json (skeleton flows) | 63c5c07b2d90e3d99eaebdffb48b5009d8b34bf1e89d6b765b9b809e7bf7f5cd |
| c024 lam indep.py (kernel/Schur pattern) | a02e37635cda0f18934895d80fba2d44f660ac0c9f6ce41e5aa0d3b04ce0e1e7 |
| lb2_cert.py (this certificate) | bf76df4189033525526d6fd4a73ca15ef62095de119d114548e033d3e137175d |

## 8. Falsifier, scope, residual

**Falsifier (any one kills LB-2):** exhibit, at r = 0.05 or r = 0.025, (i) a fourth critical point
of m₉ in B_{2r}(M) — in particular any local maximum other than M with value > b−ℓ; or (ii) a
height h ∈ (b−ℓ, b) and a component of {m₉ > h} ∩ B_{2r}(M) containing neither M nor reaching ∂B;
or (iii) a point of the segment [yy, M] with m₉ < v* − 1e-2·ℓ; or (iv) a third superlevel arc of
{m₉ > b−ℓ/2} on ∂B_{2r}(M). Each is directly tested by C3/C7, C5, C6 respectively, fail-closed.

**Scope.** The analytic reduction (Lemma 2) is r-free. The certified inputs (A), (C), (D) are proved
at the two frozen rungs r = 0.05 and r = 0.025 — the same scope at which the C025 skeleton itself is
certified ("3/3 rung skeletons") and at which the C029–C031 assembly operates; every certified
quantity is continuous in r with O(1) absolute margins (eigenvalue margins ≥ 0.014, v* window
position 0.499, finger arcs fixed to 3 decimals), so the statement extends through an r-neighborhood
of the rungs by the displayed margins. A fully r-uniform enclosure (interval arithmetic in r over
(0, 0.05]) is a mechanical strengthening of C3–C7, not an obstruction: the named item as registered
("component-connectivity along the mean ridge (deterministic, against the C025 skeleton)") is
discharged at the skeleton's own certification scope.

**Certificate transcript:** see `lb2_transcript.txt` (normal run; `-O` run byte-identical, cmp-verified).
Fail-closed test: tampering b by +1e-4 aborts with `SystemExit: LB-2 CERT FAIL: v* not strictly
inside window` (exit code 1).
