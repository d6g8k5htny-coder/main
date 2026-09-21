# KIMI-DER-026 — γ-LOC condition (ii-c) discharge (AO48-WO-063, Task 3)

**Date:** 2026-08-05 · **Inputs:** AO48-WO-063 sha256 `e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2` (echoed per WO spec); station table of record `c031 verify.json` sha256 `83313c96d1610130451a7e936245a06e80d8cc84df00a20ae9e46e7aeef2a271` (bound in certificate S4); corpus bindings in §7.
**Certificate:** `verify_gamma_loc_iic_v1.py` sha256 `ece7244acaa69a2f8719d35bd73f5ea9f6542902c8ae86efd6a3ed14ffece570`, exit 0 in normal and `python3 -O` modes, transcripts byte-identical (`cmp` clean), sha256 `61e4dc8a46de2dd58e5fe6048af73e7e4394994403f32c98f6029741c214f057` (both). Fail-closed: `ck()` raises `SystemExit(1)`; no bare `assert`; `IIC_CERTIFICATE_PASS` printed only after all S1–S6 checks.
**Verdict:** **CONDITIONAL** — a proof of (ii-c) modulo **exactly one named analytic premise, P-NMZ-γ** (§3), with the premise's quantified sufficient condition checked at its far end against the station table of record (certificate S4) and its near end against the machine-verified F.4 template (certificate S1–S2). Every other link in the chain is proved or machine-verified here (§4, chain D1–D7). Per the KIMI-DER-009/009b discipline the premise is named, never smuggled: §3 states precisely which segment of the sufficient condition is *not* certified by any artifact of record.
**Grade labels:** theorem-grade / carried-grade / measured-diagnostic are marked per statement. Measured values are diagnostic only and never enter the proof. Precision labels EXACT vs decimal-dps-N throughout. No status or Boolean claims are made anywhere in this deliverable.

---

## 1. The condition, verbatim as LB-3 states it

LB-3 (`LB3_DISCHARGE.md` sha256 `08153fa42a2d4c01fe9932bd5430ecfc62cfda3f43523420f253a4c0c8526f0c`), §2.0 decomposition of γ-LOC:

> "γ-LOC (L3) = clause (i) locality/measurability + clause (ii) terminal-value stabilization, which itself splits: (ii-a) the other branch terminates; (ii-b) the terminal is a maximum; **(ii-c) the terminal value is > b with probability 1 − O(exp)**."

with (L3), the exact UB0 scope (Lemma UB0 §4, sha256 `c6a90469b6e43c592b661a63c85a6fa6590505f3d22fd888a328ebf7926d41c0`):

> "**Hypothesis (γ-LOC).** The qualification verdict of a window saddle y — adjacency to x_M and the other-branch terminal's value comparison against b — is measurable with respect to the field on the union of (i) the O(r) pair neighborhood and (ii) the other branch's ascent tube up to its terminal, and the terminal-value law stabilizes: **for rim/arch saddles the other branch exits the conditioned zone with running height > b and terminates at an ambient maximum of value > b with probability 1 − O(exp)**."

and LB-3's own isolation of the open clause (§2.4):

> "**The gap, in one sentence:** γ-LOC(ii-c) lacks exactly a uniform C¹-over-the-ascent-tube margin between the pin-conditioned nominal field and the capture-below-b boundary for the third-saddle arch/rim escape configuration in the near/moderate zone d ≲ 3 (where Lemma FD's stabilization does not reach and the conditional mean's needle response is order-one-tenths of σ), together with the transfer of that margin to a 1 − O(exp) Gaussian tube-tail bound under the conditional measure — F.4's machine-verified margins being proved, as the supplement's own F.6 map records, for the endpoint-saddle normal form with 'side-24 normal-form verification' still outstanding, and the Palm-transfer premise (F.4 residual (i)–(v)) covering the capture side only."

Clauses (i), (ii-a), (ii-b) are **PROVED** by LB-3 (§2.1–§2.2: compact torus + gradient ω-limit + Bulinskaya + no saddle–saddle connections by SARD-G at the KIMI-AUD-022 specialist-reviewed-conditional grade); this deliverable consumes them as cited and discharges (ii-c) at the grade stated above.

## 2. Setting and the reduction used throughout

Window (arch/rim) saddle y, f(y) ∈ (b−ℓ, b), b = 6/5 [EXACT], ℓ = r³/6 [EXACT] (testbed κ = 1; Lemma G.2.3 K2: the pin value window is exactly κr³/6), one ascending branch adjacent to x_M, the other branch β. On the a.s. Morse–Smale event (LB-3 item (A) + Bulinskaya), β terminates at a single nondegenerate maximum z ((ii-a),(ii-b)). Testbed: periodized side-24 Bargmann–Fock field, rungs r = 0.05 = 1/20, 0.025 = 1/40 [EXACT].

## 3. THE SINGLE NAMED PREMISE — P-NMZ-γ (near/moderate-zone growing tube clearance)

**Premise P-NMZ-γ (analytic, named, exactly one).** *There exist, for all r ≤ r₀: (i) a deterministic tube family 𝒯_r around the pin-conditioned nominal other-branch ascent of the window saddle y, from y to the horizon section {d = d₀ = 5}; (ii) a margin function μ(r) := the C¹(𝒯_r)-distance from the pin-conditioned nominal field m_r (conditional mean plus pin shifts, on the Palm fiber) to the capture-below-b boundary ∂𝒞_b = {fields whose window saddle's other branch terminates at a maximum of value ≤ b}; and (iii) a sub-Gaussian scale s(r) for the conditioned residual's C¹(𝒯_r) tube seminorm, P(‖E‖_{C¹(𝒯_r)} > t) ≤ 2 exp(−t²/(2 s(r)²)) — such that the clearance ratio grows:*

> **Λ(r) := μ(r)/s(r) → ∞ as r ↓ 0, in the form Λ(r)/√(2 ln(1/r)) → ∞** *(sufficient for o(r^p) tails for every fixed p: certificate S3).*

**Quantified sufficient condition, and what is checked against the record.**

- **Far end (d ∈ [5, 5.5], the horizon slab) — CHECKED against the station table of record** (certificate S4, on-grid class per C031 named residue #12, "slab station density + analyticity"): the conditional mean's C¹ pull is max station (|m|, |gradm|) = (0.0003701380190700305, 0.0016341647279050065) [decimal-dps-16/19], the projection field contributes proj_c1 = 0.001995628054736906 [decimal-dps-16], total C¹ perturbation = 0.003629792782641913 ≤ gate 0.1307 [decimal-dps-16], slack 0.127070207217358087, margin factor 36.0075651219052 [decimal-dps-16] (the C031 "36× margin" stands), and the variance ratios v ≥ 0.9999999577222840 [decimal-dps-16, station (5.0, 0.0)] — i.e., at the far end of 𝒯_r the residual is at full unconditional strength (r-free s) and the gate slack is r-free positive, so the far-end segment of the sufficient condition holds at the R3′/C028 grade. The C028 barrier's existence at gate η = 0.1307 on D (C028, theorem grade; CM factor e^{−4.82550294/2} = 0.089569 [decimal-dps-6], ‖ũ₀‖²_H = 4.82550294 [decimal-dps-8 print of an exact-mp value]) certifies that the capture-below-b boundary is at positive C¹ distance from the near-unconditional nominal on the slab.
- **Near end (the blow-up chart, d = O(r)) — the machine-verified F.4 template** (certificate S1–S2): for the endpoint saddle S of the side-24 normal form (A.10)/(A.11), the outward branch is confined to the cone C_E = {0 ≤ ξ ≤ δ, E(Y) ≤ 4(R²/τ)ξ²} and strip S_E with F_X ≥ (3/4)ξ and Ġ_E ≤ −(2/3)q², reaches X = 5/4 with P > 65/256 [EXACT ledger], and persists under pin-preserving C¹ perturbations ‖Dℰ‖∞ ≤ η_R = 1/(8192R³) with the corrected tube cost 12(η_R/R)(R²/τ)h² ≤ 12/8192 [EXACT, machine-verified at R = 1, 2, 5, 50]. This is the deterministic template of the required margin (and discharges the analogous statement for the endpoint saddle's own outward branch — the β-channel object — at the supplement's carried grade, F.6 row "Deterministic capture/escape (F.4) | repaired under explicit hypotheses | side-24 normal-form verification", which remains carried by the supplement and is **not** consumed as this deliverable's premise).
- **The unchecked segment — the premise proper, not smuggled:** the tube segment through the **near/moderate zone d ≲ 3 for the third-saddle arch/rim configuration**, where Lemma FD's stabilization does not reach (FD horizon d ≈ 3–5; the C030 horizon table has v = 0.018/0.55/0.976 at d = 1/2/3, r = 0.025 [decimal-dps-3, derived-on-grid]) and the conditional mean's needle response to the ℓ-pin is order-one-tenths of σ (LB-3 §2.3–2.4; dual weights 6.556e4/5.183e5 in C022 Observed Update, sha256 `9bc0647b…d5cb`). No artifact of record certifies μ(r) or s(r) on this segment; the growth Λ(r) → ∞ is supported only by measurement (UB0 §4: m* profile 4.6–9.7σ above b at the probes, r = 0.05 — diagnostic; certificate S3 verifies 4.6 and 9.7 both exceed the r = 0.05 threshold 2√(ln 20) = 3.461636765204571 [decimal-dps-16], giving e^{−Λ²/2} ≤ 2.54193e-5 … 3.70353e-21 [decimal-dps-6]).

**Why exactly one premise suffices:** the chain in §4 shows (ii-c) follows from P-NMZ-γ plus items already proved (LB-3's (i)/(ii-a)/(ii-b); the deterministic monotonicity D1; the Gaussian-tube-tail calculus D4–D5). And exactly this premise is necessary in the sense of LB-3's isolation: every other segment of the escape is covered by an artifact of record (F.4/G-stack: blow-up chart, endpoint template; FD/C030: stabilization from d ≈ 3; R3′/C028: the horizon slab), and no record artifact certifies the near/moderate third-saddle margin.

## 4. The discharge chain, step by step

**D1 (monotone reduction — deterministic, proved here).** Gradient ascent gives f(z) = sup_{t≥0} f(β(t)); hence {f(z) ≤ b} = {β's running height never exceeds b}. Once the running height exceeds b at any point, the terminal value exceeds b for *every* smooth field — no perturbation budget, no probability. **Consequence:** the entire stochastic content of (ii-c) is the event that β stays ≤ b *through the near/moderate zone*; the far zone contributes zero failure on the escape event. ∎ (certificate: no numeric content; the logical form is displayed.)

**D2 (the far-barrier-only route is structurally insufficient — proved here, certificate S5).** The alternative suggested route — discharge (ii-c) via the C028/R3′ far barrier with the conditional-barrier constant as the escape floor — cannot yield 1 − O(exp): the barrier gives P(capture ≤ b | arrival in the band at d₀ = 5) ≤ 1 − c with c = 0.089569·P₀ ≤ 0.089569 (P₀ ≤ 1), hence failure ≥ (1 − c)·P(arrival below b) with 1 − c ≥ 0.910431 [decimal-dps-6] — an O(1) factor. The route therefore reduces (ii-c) to P(arrival below b in the band) = O(exp), which is exactly the near/moderate-zone content of P-NMZ-γ. Recorded so the premise is not hidden behind the barrier. (C028's barrier remains load-bearing at its own scope: the far-end clearance of the sufficient condition, §3.) ∎

**D3 (tube decomposition — deterministic, proved here).** Let 𝒯_r be the nominal tube of §3 and E the conditioned residual. Then
{β terminates ≤ b} ⊆ {nominal margin fails} ∪ {‖E‖_{C¹(𝒯_r)} ≥ μ(r)} ∪ {fifth-derivative good event fails}.
Under P-NMZ-γ the first event is empty (the nominal m_r is at C¹ distance μ(r) from ∂𝒞_b along the whole tube, near/moderate segment included). ∎

**D4 (the residual tube tail — conditional on P-NMZ-γ item (iii)).** P(‖E‖_{C¹(𝒯_r)} ≥ μ(r)) ≤ 2 exp(−Λ(r)²/2), and with Λ(r)/√(2 ln(1/r)) → ∞: for every fixed p, eventually Λ(r)²/2 ≥ (p+1) ln(1/r), so 2e^{−Λ²/2} ≤ 2r^{p+1} = o(r^p) [EXACT statement; threshold form 2√(ln(1/r)) verified at both rungs in certificate S3: 3.461636765204571 at r = 0.05, 3.841291165279683 at r = 0.025, decimal-dps-16]. ∎

**D5 (fifth-derivative good-event tail — carried grade, supplement G.5 "proved"; P02-LM-007).** P(ℋ₅ fails at level u = r⁻¹) ≤ C e^{−c/r²}, and e^{−c/r²} ≤ p! c^{−p} r^{2p} = o(r^p) for every fixed p [EXACT: e^x ≥ x^p/p!; instances p = 1…32 at both rungs verified in certificate S3, decimal-dps-60 against exact-rational right-hand sides]. The capture/escape robustness threshold η_R = 1/(8192R³) is a fixed polynomial in the envelope R (supplement G.5), so its violation is dominated by the same tails. ∎

**D6 (the terminal ledger and the escape-level-vs-b check — EXACT, certificate S2).** In the normal form (A.10), p₀(X) = X³/3 − X/4 − 1/12: p₀(−1/2) = 0 = P(M), p₀(1/2) = −1/6 = P(S), p₀(5/4) = 49/192 [all EXACT]. The height scale is pin-consistent: physical f = b + κr³P (supplement G.6: "the fold potential is divided by κr³"; κ = 1 testbed), so P(S) = −1/6 maps to f(x_S) = b − r³/6 = b − ℓ [EXACT at both rungs]. The terminal ledger P(5/4, Y) > 49/192 − 1/768 = 65/256 > 1/4 > 0 = P(M) [EXACT] therefore reads in physical units:
escape level > b + (65/256)·κr³ > b = 6/5 for all r > 0 [EXACT; margins 3.173828125e-05 at r = 0.05, 3.96728515625e-06 at r = 0.025, decimal-dps-16].
**Answers to the work-order checks:** (a) *"is the relevant terminal level the escape-to-X=5/4 level, and does it exceed b?"* — for the endpoint-saddle template, YES: by D1 the terminal maximum's value is ≥ the running height at X = 5/4, which exceeds b by (65/256)κr³ [EXACT]. The comparison "65/256 vs 6/5" is a **units error**: 65/256 = 0.25390625 < 1.2 [EXACT] is irrelevant — 65/256 is a margin *above the M-pin level*, and the M-pin level is b itself (P(M) = 0 ⟺ f = b). The escape clears b by (65/256)/(1/6) = 195/128 = 1.5234375 [EXACT] window heights ℓ. (b) For the third saddle (the γ-LOC object), the X = 5/4 section of the endpoint template is not the relevant section — its branch is not the endpoint branch; the relevant level is the running height at the near/moderate-zone exit, governed by P-NMZ-γ. Not smuggled: stated in §3.

**D7 (assembly).** Under P-NMZ-γ, D1 + D3 + D4 + D5 give
P(f(z) ≤ b) ≤ 0 + 2e^{−Λ(r)²/2} + Ce^{−c/r²} = o(r^p) for every fixed p,
i.e. superpolynomial decay in r — **1 − O(exp)** as (ii-c) states. The terminal is a maximum ((ii-b), LB-3) and its value exceeds b (D1). ∎ **CONDITIONAL on P-NMZ-γ alone.**

**What is proved without the premise (no premise consumed):** D1, D2, D3, D6; the S1–S2 machine-verified F.4 margins and ledger; the S3 tail calculus; the S4 station-table facts; the S5 barrier arithmetic and projection fact ‖ũ₀′‖_H ≤ ‖ũ₀‖_H [EXACT statement: pin annihilation is an orthogonal projection in H]. The F.4/G-stack endpoint-saddle escape (β-channel analogue) stands at the supplement's carried grade ("side-24 normal-form verification" remains the supplement's own named residue; the Palm-transfer RP-A/RP-L is CLOSED at Part G.7, 2026-08-02, for the O(r³) adjacency channel — a polynomial rate, which is precisely why the 1 − O(exp) clause cannot be inherited from that stack: its confinement-hypothesis failures include the shallow-eigenvalue layer at Palm probability O(r³), G.7's own power count. The exp rate must come from nominal clearance — P-NMZ-γ — not from hypothesis-failure tails.)

## 5. Hypothesis list (everything consumed beyond the cited artifacts)

(H1) LB-3 clauses (i), (ii-a), (ii-b) at their grades (SARD-G at the KIMI-AUD-022 specialist-reviewed-conditional grade, clarifications A–C execution-level and not touching the consumption; Bulinskaya nondegeneracy and distinct critical values; compactness of the side-24 torus).
(H2) Gaussian measure theory: Borell–TIS for continuous centered Gaussian fields on compacta; sub-Gaussianity of the C¹ tube seminorm; Cameron–Martin quasi-invariance, support theorem, and Anderson's inequality (C028's usage); Gaussian regression (conditional mean linear in the pins, conditional covariance PSD-dominated by the unconditional one).
(H3) The F.4 deterministic stack at the supplement's grade ("repaired under explicit hypotheses"; corrected tube cost 12(η_R/R)(R²/τ)h²; machine-verified margins re-verified here in S1) — consumed as the near-end *template* and the β-channel analogue; the side-24 normal-form verification remains the supplement's carried residue and is not consumed as this deliverable's premise.
(H4) Supplement G.5/P02-LM-007 uniform conditioned fifth-derivative supremum tails (carried grade), and G.6's exact transverse/height scaling (κr³ fold-potential division), machine-verified K2 value window κr³/6.
(H5) Testbed constants: b = 6/5, ℓ = r³/6, κ = 1, rungs r ∈ {1/20, 1/40} [EXACT].
(H6) The station table of record at the on-grid class (C031 named residue #12) for the far-end check only.
**The one analytic premise:** P-NMZ-γ (§3). Nothing else.

## 6. Falsifier section (mandatory)

1. **Premise falsifier (sufficient-condition check):** a certified near/moderate-zone computation (d ≲ 3) of the pin-conditioned nominal's other-branch tube showing μ(r) ≤ 0 (nominal itself captured below b), or s(r) growing/μ(r) shrinking so that Λ(r)/√(2 ln(1/r)) stays bounded — kills P-NMZ-γ and re-opens (ii-c).
2. **UB0 §6 / addendum A.4 falsifier (measured-grade, stands):** "a rim/arch flow whose other branch terminates below b at an off-image maximum (none in 11,200 (C020) + 28,800 (C021) + 6,400 (ballfix) pooled compatible-ensemble flows)"; addendum: "none in 11,200 + 28,800 + 6,400 pooled flows".
3. **Chain falsifier:** a counterexample to any displayed deterministic step (D1 monotonicity; the p₀ values; the ledger 49/192 − 1/768 = 65/256; the pin-consistency b + r³P(S) = b − ℓ; the tail inequality e^{−c/r²} ≤ p!c^{−p}r^{2p}) — each is machine-pinned in the certificate; a failing instance kills the corresponding link.
4. **Station-table falsifier:** an independent recomputation of the c031 station table giving total > gate 0.1307, or min v materially below 0.9999999577222840, kills the far-end check (§3, first bullet) and with it the checked segment of the sufficient condition.
5. **Units falsifier (recorded, resolved):** any reading that compares 65/256 directly against b = 6/5 is a units error (65/256 is a scaled margin above the M-pin level b; certificate S2 pins both directions: 65/256 < 6/5 [EXACT] and b + (65/256)κr³ > 6/5 [EXACT]).

## 7. Dependency table (name — sha256 — role)

| artifact | sha256 | role |
|---|---|---|
| AO48-WO-063 work order | e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2 | tasking; echoed in Inputs |
| LB3_DISCHARGE.md | 08153fa42a2d4c01fe9932bd5430ecfc62cfda3f43523420f253a4c0c8526f0c | verbatim (ii-c) statement; proved clauses (i)/(ii-a)/(ii-b); gap isolation |
| Lemma UB0.md | c6a90469b6e43c592b661a63c85a6fa6590505f3d22fd888a328ebf7926d41c0 | γ-LOC exact scope (§4); falsifier; measured m* profile (diagnostic) |
| Lemma UB0 Addendum.md | 224438cb740565ff8b2f0acab83a75c8dc80cc15265a55451f43ff930894c94c | corrected rim picture; pooled-flow falsifier counts |
| C031_LBRATE_Integration.md | e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e | R3′ conditional-barrier transfer (36× margin); G1 repair; named-residue register |
| c031 verify.json | 83313c96d1610130451a7e936245a06e80d8cc84df00a20ae9e46e7aeef2a271 | station table of record (far-end check) |
| C028 FarAscent Package.md | 64b94b6e5342b3f8d2f3a6a8dfd5e1aa38e769ba319eeee87d657074b568e606 | barrier c = 0.089569·P₀; η = 0.1307; support-theorem positivity |
| C029 BasinPersistence Package.md | b23c3d42ccc2182e9a59a4ae1b7eda8c73939f543884268a0da3e2cc6880a785 | BP reduction context; measured capture (diagnostic) |
| C030 CountingLemmas Package.md | aed6683edaed6d483704a63cd321fcb300e329a26bf6c143042ead58a60bfc7b | horizon table v = 0.018/0.55/0.976 at d = 1/2/3 (near/moderate zone scope) |
| C022 Observed Update.json | 9bc0647b885931ba4dc862bedfb879f9bd12c4cd6d39f51c6e98e54c7f75d5cb | Lemma FD / γ-LOC corollary; needle dual weights 6.556e4/5.183e5 |
| SIDE24_GAP_FILL_SUPPLEMENT.md | 33609b70d8c4d970d3594ebdc2cb59200508bfb2e53002ae7c6b10b3bc19d41f | F.4 (A.10)/(A.11) margins; G.5 tails; G.6 κr³ scaling; F.6 carried residue; G.7 power count |
| verify_capture_escape_budgets.py | 56206651a85d128b3b2d3395f12773b74a55a5efff7539eb86392a2ab416fb38 | F.4 margin source script (re-implemented self-contained in S1) |
| lb3_certificate.py | 6daa149cb5a7580a4c4e16e973da2cc08486db09219fbe83963319bbd17a6f75 | LB-3 certificate (cited) |
| verify_gamma_loc_iic_v1.py (this deliverable) | ece7244acaa69a2f8719d35bd73f5ea9f6542902c8ae86efd6a3ed14ffece570 | fail-closed certificate S1–S6 |
| transcript_normal.txt = transcript_O.txt | 61e4dc8a46de2dd58e5fe6048af73e7e4394994403f32c98f6029741c214f057 | byte-identical both-mode transcripts |

## 8. The remaining premise, one paragraph

Exactly one analytic premise remains between this deliverable and a full proof of γ-LOC(ii-c): **P-NMZ-γ**, the growing tube clearance of the pin-conditioned nominal field's third-saddle (arch/rim) other branch through the near/moderate zone d ≲ 3. Every other segment of the escape is covered by an artifact of record — the blow-up chart by the machine-verified F.4 stack (near end; endpoint-saddle template with the corrected tube cost and pin-preserving radius η_R = 1/(8192R³)), the stabilization zone by Lemma FD/C030 (v = 0.55 → 0.976 across d = 2 → 3), and the horizon slab by R3′/C028 (total C¹ perturbation 0.003629792782641913 against gate 0.1307, a 36.0076× margin, variance ratio 0.9999999577222840 at (5.0, 0.0)) — but the conditional mean's needle response in d ≲ 3 (order-one-tenths of σ per LB-3, dual weights 6.556e4/5.183e5) means the nominal's clearance there is not inherited from any of these, and no station of record samples that zone. The measured support (m* profile 4.6–9.7σ above b at r = 0.05, esc = 0 in 46,400 pooled flows) is consistent with Λ(r) well above the 2√(ln(1/r)) threshold and with growth, but measurement cannot prove growth. Closing the premise requires exactly what LB-3 isolated: a certified C¹-over-the-ascent-tube margin μ(r) between the pin-conditioned nominal and the capture-below-b boundary for the third-saddle configuration on d ≲ 3, with a sub-Gaussian scale s(r) such that μ(r)/s(r)/√(2 ln(1/r)) → ∞; the natural instrument is the LB-3 S2–S4 exact-kernel machinery (spectral lattice (π/12)ℤ², masses e^{−|k|²/2}, certified tails) extended to a certified station net across the near/moderate tube. Until that exists, (ii-c) stands at CONDITIONAL modulo P-NMZ-γ, with every other link proved or machine-verified in certificate verify_gamma_loc_iic_v1.py.

## 9. Reproduction

```
cd /mnt/agents/output/gamma_loc_iic
python3 verify_gamma_loc_iic_v1.py > transcript_normal.txt    # exit 0
python3 -O verify_gamma_loc_iic_v1.py > transcript_O.txt      # exit 0
cmp transcript_normal.txt transcript_O.txt                    # byte-identical
```
Runtime < 5 s per mode. Fail-closed: any check failure prints `CHECK FAILED: …` and raises `SystemExit(1)`; `IIC_CERTIFICATE_PASS` prints only after all S1–S6 checks (17 exact-rational F.4 checks × 4 R values; the EXACT ledger chain; 24 tail instances; the Decimal-exact station-table recomputation; the dps-60 barrier arithmetic; 13 corpus hash bindings).
