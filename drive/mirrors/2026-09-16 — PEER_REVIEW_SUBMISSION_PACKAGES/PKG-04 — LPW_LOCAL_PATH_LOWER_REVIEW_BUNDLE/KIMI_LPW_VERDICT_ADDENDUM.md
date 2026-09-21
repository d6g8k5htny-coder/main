# ADDENDUM to KIMI_LPW_REVIEW_VERDICT.md — corrections and clarifications (KIMI line, 2026-09-13)

The frozen verdict (body d399e28378d177f88d08032c81f1573f35c9af2b600b9b775ebfe40f555cbbad, whole-file
04bcbdf2013d83b113c18518bd101bb8c1a96e01e8c66a6aab2d2390e5e4d1e3cf) remains unchanged in the record; its
core endorsement (all six interfaces PASS WITHIN SCOPE; the theorem LPW endorsed at the exact candidate
hash cf58f72e…e1a5) STANDS. The following surrounding claims are corrected, per the author-side
reconciliation (LPW_Review_Reconciliation_2026-09-12, zip sha256 782cac8e…b1f, 21/21 manifest files
verified). Extraction rule for the verdict's declared body hash, per request: the body is everything
before the line "SHA-256 of this report body (text after this line is excluded):"; the hash is the
SHA-256 of those body bytes (UTF-8, LF); the marker line and the hex line after it are excluded.

## 1. WITHDRAWN: Consequence 4 (the two-sided composition) — dimension/law mismatch, owned

The verdict's fourth program consequence said combining LPW with "the ratified upper chain
(AO48-OPR-045)" gives a two-sided Θ(r³) at the existential level. That is a scope error. AO48-OPR-045
ratifies, at its stated scope, the compact-positive-mark estimate sup(1 − p_r) ≤ C r³ and the theorem
ν₃,₂₄(ℓ) = c₃,₂₄ℓ^{−1/3}(1+o(1)) for the normalized periodized Bargmann–Fock field on the side-24
**THREE-torus**, and its own firewall text states: "this ratification concerns the SIDE24 3D track only.
The 2D q0 program is untouched: … no sealing, no release, no cross-track inference." The 2D lower bound
1 − q₂(r, 6/5) ≥ c r³ (LPW) and the 3D upper bound are different random objects; they do not combine into
a two-sided estimate for either. The valid composition requires a matching 2D UPPER theorem with the same
dimension, field normalization, pins, determinant weight, event, conditional-law version, and small-r
domain — a separate open dependency (the C020-era 2D upper assembly is a different, unratified carrier at
its own grades). Consequence 4 is withdrawn and replaced by: LPW (endorsed, 2D lower, existential) PLUS a
matching 2D upper theorem (open, unratified) is what a two-sided 2D law requires; no such composition is
claimed here. The manuscript v2's Corollary 3.28 (two-sided rate) carries a separate correction note
(SIDE24_pre_peer_review_v2/ERRATA_AND_CLARIFICATIONS_2026-09-13.md).

## 2. Numerical certification repairs (owned)

(a) **Eigenfloor rounding direction.** The pasted handoff and verdict printed
"λ_min ≈ 0.126553449667289, certified ≥ 0.1265534497": the printed floor exceeds the printed value by
3.2711e-11 — an upward-rounded lower bound, wrong direction. A certified lower bound must round DOWNWARD
(e.g., ≥ 0.1265534496) and, more importantly, a single eigenpair residual is not a spectrum floor: a
valid certificate needs the matrix enclosure, the tail control, and an argument covering the smallest
eigenvalue. The author-side's independent supplement supplies a proved conservative endpoint floor
**λ_min(Γ_{0,24}) > 31/250 = 0.124** (exact rational Sylvester minors of Γ_{0,∞} − I/8, all ten positive;
infinite-image tail bound ‖Γ_{0,24} − Γ_{0,∞}‖_op < 10⁻¹⁰²). I have INDEPENDENTLY VERIFIED that
supplement: the ten exact minors match exactly (7/8, 49/64, 343/512, 931/4096, 6517/32768, 931/786432,
4263/2097152, 11571/16777216, 11571/134217728, 203/1073741824), and the finite-torus moment corrections
are real (a₂ − 1 = −9.6525e-123, a₄ − 3 = −4.0219e-124, recomputed at 140 dps). The 0.124 floor is
adopted as the proved endpoint bound; RB's reported value 0.126553449667289 stands as evidence, not as
the certified floor.
(b) **The Schur "exact rationals" were planar reference values.** The verdict/lead pack called
Cov(J | U_0) = diag(2, 1/2, 1/2, 1/6) and the conditional mean (−b, 0, 0, 0) "exact rationals at machine
precision." For the exact side-24 field those are the PLANAR-limit identities (a₂ = 1, a₄ = 3, a₆ = 15);
the exact finite-torus formulas are E[J | U_0 = u_0] = (−a₂b, 0, 0, 0) and
Cov(J | U_0) = diag(a₄ − a₂², a₂(a₄ − a₂²)/4, a₂(a₄ − a₂²)/4, (a₆ − a₄²/a₂)/36), with
a₂ = 1 − 576Σn²e^{−288n²}/Σe^{−288n²} < 1 (deviations ~1e-123, verified). Exact-torus claims must use
the moment-dependent forms.
(c) **The ~1.28e-3 density is a diagnostic, not the uniform m.** The LPW constant m is
inf_{r ∈ [0, r_G], j ∈ K_J} p_{J | U_r = u_r}(j); the center/corner densities at one rung are diagnostics
(the lead pack said so, and it is repeated here). The planar reference minimum over the original large
K_J is ~5.98e-14 at the q = −11 corner; a quantitative successor may use a tighter explicit compact set
with all dependent bounds recomputed there (the acceptance contract 03_CONSTANT_CERTIFICATE_CONTRACT.md
governs the follow-up).

## 3. Independence clarified (as requested)

Recorded precisely: **one external Kimi provider-family review containing five task/implementation
branches with reported author-script independence** (lead + RA + RB + RC + RD). The branches share a
provider, coordinator, exposure to the author's proof (ordinary hostile review, not blind
reconstruction), and the problem's interpretation. They are NOT five demonstrated mutually independent
external confirmations of the theorem. Each branch's scripts and receipts ship in the immutable review
bundle accompanying this addendum; the verdict's "five independent review lineages" phrasing is hereby
corrected to the recorded form.

## 4. R3 addendum — CONFIRMED at its hash

01_R3_ADDENDUM_AND_SOURCE_BRIDGE.md (in the reconciliation bundle, manifest-verified) is reviewed and
CONFIRMED as written: it makes the residual's nonstationarity explicit (g_r = f − L_rV_r,
‖g_r‖_{L^p(C⁴)} ≤ ‖f‖_{L^p(C⁴)} + A₄‖V_r‖_{L^p}, with A₄ = C4*·G*, no stationarity asserted, no
independence between f and V_r needed for that step), evaluates conditioning through the continuous
regression kernel at every j, keeps the Markov-then-integrate order (never a fixed unconditional tail
subtraction from the O(r) box mass), and fixes the M₄ convention (exact-order seminorm; the full C⁴ norm
as a dominating auxiliary bound; B_3 and B_4 distinct). This is exactly the substance RC's A1/A2 asked
for. The candidate file cf58f72e…e1a5 remains unchanged; the addendum stands at its own hash
(verified via the bundle MANIFEST entry).

## 5. RA's source-access residue — CLOSED byte-exactly by my own extraction

GP-DER-118-v1.10 (Drive ID 1qc6ep2S4PIoPMEWDsdDQI1dr0LOwJiK9) verified: whole file 30,933 B,
sha256 c1d6e559274d7e87a3fa92f44cb2534a5e277a3b5941fd40db380599ddba0564 (matches); frozen theorem body
extracted by the declared rule (exclude the unique BEGIN_FROZEN_THEOREM_BODY/END_FROZEN_THEOREM_BODY
lines, LF endings, one terminal LF): 29,293 B, sha256
9b7901e112a4857e3ea59942858684f72fc09a2825b1efa859360dd8fa55f014 (matches, recomputed by me on the
delivered bytes). The scoped correspondence is recorded exactly: its Section 0 matches LPW's pins
(Q_{6,r} ≡ Q_r), values (b and b − r³/6, both gradients zero), determinant weight, normalizer, and typed
Palm quotient; its selected-branch event A_r is NOT LPW's elder-pairing defect event D_f(M_r) ≠ S_r —
the two arguments use the same underlying law for different conclusions. No conclusion transfer is
claimed (in particular, nothing here touches P0.1).

## 6. K3-THM-001 acknowledgment (separateness preserved; two owned defects)

GP-LB-STAT-004's disposition is recorded, and LPW's endorsement does not rehabilitate K3-THM-001
(sha256 715124dba1f089430594a3322b7d32c3508fbcb10c14f6591a432ff582f8854d). Two specific defects in my
K3 artifacts are owned here:
(a) an arithmetic error in K3-THM-001's H1 display: 0.9001 × 0.9666 = 0.87003666, not the printed
0.8705 — the corrected value is 0.87004 (the printed 0.8705 was a miscalculation);
(b) the WP-min modulus's grade must be restated precisely: E_WP(r) = 3.5e-3·r^{3/2} is a strong
NUMERICAL bound (midpoint/trapezoidal quadrature with a sup-at-r₀ margin display), not an
interval-enclosed rigorous bound — the margin argument is not a continuum spatial error enclosure; the
interval-enclosed form is W3's open job (its exact-tier band upper acc_ex is the intended instrument).
Additionally recorded for any H1–H12 successor: the hypothesis list must include W10's Bonferroni pair
term o(r³); the uniform Radon–Nikodym/transfer from the unweighted conditional law to the exact
weighted triple-Palm law over all arch locations is an open load-path gap; the far factor's tiers must
stay separated (0.9666 measured vs 0.089569·P₀ theorem tier); measured exit zero counts are not a proof
that the exit loss is o(1); and the Λ-side premise H-B3's rung-stability evidence FAILED at the
certified window [0.4, 2.5] (W8 v2, ratio 3.5598022, transcripts byte-identical) — the v3 repair (1/80
reference rung) is a NEW disclosed artifact, not a rewrite of the failed v2.
GP-LB-STAT-004's §7 conditional implication schema is accepted as the correct strongest conditional
form for any successor assembly.

## 7. What is NOT changed

The LPW endorsement itself (all six interfaces PASS WITHIN SCOPE at cf58f72e…e1a5); the candidate
file; the status firewall: this addendum changes no LS-CTL Boolean, eligibility predicate, theorem
status, RP status, AO48 operator record, or q0 package status. Any status change requires separate
operator adjudication. P0.1 remains HOLD.

SHA-256 of this addendum body (text after this line is excluded):
ca58855f8344c1c108dcb59cbacb5581b79bad503effde985c0a2e8497835b54
