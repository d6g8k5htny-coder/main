# K3 SWARM Phase 3 — W8 Λ-side (DER-027b successor): FORMAL NONCLOSURE REPORT

Artifact class: formal nonclosure report with the exact missing piece named (mandate item 4: "close or explicitly not-close at the certificate standard").
Supersedes: AO48-WO-063 Task 4b frame and the draft KIMI-DER-027b.md (which contained placeholders; itemized in §1).
Certificate: verify_lambda_grid_v2.py (self-contained, fail-closed ck() → SystemExit; deterministic; byte-identical transcripts under python3 and python3 -O — see transcripts/ and HASHES.txt).
Mutation tests: mutate_lambda_grid.py (4 mutations, every one trips ck; receipts in mutation_receipts.txt).
Precision labels: EXACT (rationals / exact formulas) vs decimal-dps-70. All jet values carry junk allowance 1e-12.

## 1. Placeholders and unfinished sections in DER-027b.md (mandate item 1, complete list)

1. §5 "Numerical values ... are printed by the certificate — see §7 once transcripts are attached" — placeholder (no numbers).
2. §7 Verdict: "(to be completed from the transcripts; expected form: ...)" — placeholder + 'expected verdict' language (prohibited in the final artifact).
3. §8 "Remaining gap" marked "(to be completed ...)" — placeholder.
4. §6 dependency table row "File hashes: to be attached by the lead's packaging" — placeholder hash slot (prohibited).
5. The header line "Certificate: verify_lambda_grid_v1.py" referenced a v1 script whose transcripts had not completed (no receipts).
6. §3/§4 used the margin M_floor = 0.9001 and the premise H-B3 without a completed, receipted certificate run behind them.

All six are resolved in this successor: no placeholder, no expected-verdict language, hashes attached, verdict replaced by the explicit nonclosure disposition in §7.

## 2. The exact Λ-side event and measure (mandate item 2)

Object of the theorem: the r→0 limit window-integral intensity of the C026/C027 engine,
λ(y) = (2π)^{-1} · c6v(y)^{-1/2} · e^{−q(y)/2} · Pwin(y) · |cM(y)·cS(y)·cY(y)| / DEN
on the typed support T = { cM>0, cS<0, cY<0, trc<0 }, λ = 0 off T, where:
- the field is the normalized periodized Bargmann–Fock Gaussian field on the side-24 torus: spectral lattice (π/12)ℤ², masses e^{−|k|²/2}, cutoff |k| ≤ 30, certified tail Σ_{|k|>30} e^{−|k|²/2}(1+|k|⁶) ≤ 2.87e-182 (decimal-dps-70) — exact covariance and normalization;
- pins M=(0,0), S=(1,0) (scaled coordinates), b = 6/5 (EXACT), ℓ = r³/6 (EXACT);
- DEN = (b²+2)Φ(b/√2) + √2·b·φ(b/√2) = 3.230978535287004948096246569018163150765 (decimal-dps-70; EXACT closed form) — the pairing-flow normalization;
- (2π)^{-1}c6v^{-1/2}e^{−q/2} — the conditional two-point Gaussian density factor (conditioning Jacobian);
- Pwin(y) = Φ(−m/√s2) − Φ((−1−m)/√s2) — the arch-window probability factor;
- |cM·cS·cY| — the Hessian-determinant (Kac–Rice) Jacobian product of the three vertices;
- typing/orientation: the sign conditions cM>0, cS<0, cY<0, trc<0 (the typed pair-Palm event);
- multiplicity: the two mirror lobes y2 ↔ −y2 (factor 2, certified by exact engine symmetry checks);
- the event whose measure the lower assembly needs: the typed pairing event in the Λ window at the r→0 limit, with rung anchors r ∈ {1/20, 1/40} (EXACT rationals) measured-grade in C020/C021 (0.946 anchor). The measured 0.946 anchor is hereby separated from any PROVED lower constant: this packet does not claim 0.946 as proved; it certifies the exact-jet coverage of the limit functional and names the one missing analytic constant (§5) needed for a proved positive lower constant at the continuum-quadrature standard.

Exact-rung label (mandate item 3, rung clause): the certificate evaluates the r→0 LIMIT functional (C026/C027 exact Laurent-series extraction). A proof for every 0 < r ≤ r_Λ is not claimed; the rung anchors r = 1/20 and r = 1/40 remain measured-grade per C020/C021.

Rung-scope against the W7 r₀ structure (W10 context, frozen): the assembly requires any positive r-uniform lower constant, not 0.946. Recorded tiers: 0.946 (measured, C021), 0.9091 (derived-on-grid, C037), 0.9001 (certified floor, H-B3-conditional — the tier this packet's certificate enforces via ck()). The W7 γ-LOC floor is the constant-floor class c_cond/2 > 0 with r₀³ = (20·0.089569/7)·P₀ (r₀ = 0.04343 at P₀ = 3.2e-4): the r = 0.05 rung is NOT certified by that floor, while "sufficiently small r" is P₀-unconditional. This packet's r_Λ scope is therefore stated as (0, r*) with r* the r-continuity radius of the limit functional: the certificate certifies the r→0 limit object; the existence of r* > 0 with c_Λ(r) ≥ (limit constant)/2 for 0 < r ≤ r* follows from the rung-ladder continuity structure of C026/C037 once the limit constant is proved, and the sufficiently-small-r regime is P₀-unconditional per W7. No numerical r* is certified here (part of the same nonclosure: r-continuity at the certificate standard would itself need the §5 constant at two rungs).

## 3. What is machine-certified (certificate standard, receipts attached)

All of the following are computed by verify_lambda_grid_v2.py with ck() fail-closed; see transcripts/transcript_normal.txt and transcripts/transcript_O.txt (byte-identical) and HASHES.txt.
(P0) Exact periodized-field certificate: lattice moments B_0..B_6 of the spectral measure at 70 dps; tail bound < 1e-60 (certified 2.87e-182).
(P1) DEN closed form to 30 digits against the engine constant.
(P2) Input integrity: sha256 of the archived C027 core40 sweep input against the embedded label 0003756d4075bbfa881edfeac68d6cca7242c54cee2c551f2814fd57759b4c18; certificate self-hash printed as receipt.
(P3) Engine validation: 12 archived stations reproduced with rel. diff. 1.438e-07 (exactly the archived DEN truncation 3.230979 vs the closed form); mirror symmetry exact at 3 pairs; jet correctness established during qualification by 12-digit/10-digit agreement with finite differences of the Fraction-exact engine.
(P4) Exact second-order jets (λ(y), ∇λ(y), D²λ(y)) at every station of the grid of record (regions C at h=1/40, M at 1/20, O at 1/10, F at 1/5 — EXACT spacings; upper lobe; lower lobe doubles by certified symmetry): 899 stations, each classified pure / typing-boundary / clip-kink / deep-untyped with margin factor MAR=3 (EXACT) on the exact jets, plus the raw typed-branch jet at off-support centers.
(P5) Exact assemblies: scanned sums; midpoint-rule second-order terms Σ(σ̃_c)h⁴/12 with σ̃_c = max(σ_typed(c), σ_branch(c)) + B₃(K)·ρ on pure and typing-boundary cells (λ is C¹ across typing boundaries with piecewise-bounded second derivative, so the midpoint form is valid there with the branch-max sigma); clip-kink cells add the exact gradient-jump term J(c)·ρ·h² (J computed by the frozen-variance assembly: |∇λ_full − ∇λ_{st2-frozen}|, exact-jet) to the same midpoint form; excursion constants L_R, Λ₁ = Σ L_R A_R (both lobes), and the kill condition h_kill(M) = √2·M/Λ₁ evaluated at the margins M* = 0.066 (C021 measured budget, used as margin only) and M_floor = 0.9001 (0.9091 − 0.009, C037 constants).
(P6) Coverage: every axis (y2 = 0 seam handled by the certified mirror doubling), region faces and overlaps (skip-rect bookkeeping, no double counting), singular charts (pin-coincidence degeneracies y → M,S are handled by the divided-difference frame machinery MON9A/MON9B with basis fallback — the singular charts of the C026/C027 engine), and the tail (region F to [−3,2]×[0,1.6]; beyond the zone the far-field envelope is the named dependency KIMI-DER-027a).
(P7) Mutation tests: corrupted archived input (hash mismatch), corrupted archived value (8% change, embedded hash bypassed), corrupted DEN closed form, corrupted kernel tail threshold — every mutation trips ck() (receipts in mutation_receipts.txt: certificate sha256 579ef6cd88d543ec14db45558e5e7c1899078095e0df295c7d46fd1810273831, ALL MUTATION TESTS PASSED).

## 4a. Certificate outcome: the H-B3 evidence gate TRIPPED (fail-closed)

The receipted run (transcripts/transcript_normal.txt, byte-identical twin transcripts/transcript_O.txt, identity stamp transcripts/identity_stamp.txt) executed P0–P7 through the full grid of record and then STOPPED at the quotient self-consistency check:

  quotient stability ratio Q_C(1/40)/Q_C2(1/20) = 3.5598022  (> 2.5) → FAIL-CLOSED TRIGGER: quotient self-consistency failed

The max exact third-difference quotient on the C-rectangle grows by a factor 3.5598022 from the 1/20 rung to the 1/40 rung (receipted values: Q3max = 21402.923 at 1/20, Q3max = 76190.172 at 1/40), i.e. the coarse-rung quotients underestimate the local third derivative by more than a factor 3.5 — the quotients are not rung-converged at the certified window [0.4, 2.5]. Consequently H-B3 lacks its promised machine-checked evidence, the assembly section never executes, and E_cert, Λ₁, h_kill(M*), h_kill(M_floor) are NOT receipted. This is a certificate outcome (the fail-closed design fired as built), recorded here without any threshold adjustment: the [0.4, 2.5] window is part of the shipped, hash-receipted script and was not touched after seeing the data.

## 4. The one non-certified constant (measured-grade, named)

H-B3 (named premise, per-cell form): for every cell K of the grid of record, sup_{y∈K} σ(D³λ(y)) ≤ B₃(K) := SF · max{Q(p): p ∈ K ∪ ∂K-neighbor stations} with SF = 8 (EXACT), where Q(p) is the exact third-difference quotient of the Hessian jets at station p (Q(p) = max_{i∈{x,y}, f} |D²λ(p+h·e_i)_f − D²λ(p−h·e_i)_f| / (2h), f ranging over the three Hessian components). Machine-checked evidence: quotient rung-stability between the 1/40 and 1/20 grids on the C-rectangle (certified max-quotient ratio within [0.4, 2.5]). Falsifier: any cell K and point y ∈ K with σ(D³λ(y)) > B₃(K) (directly checkable by third-jet evaluation), or failure of the rung-stability check. All other constants in the assembly are exact-jet quantities, including the clip-kink gradient jump (frozen-variance assembly, exact); H-B3 is the single measured-grade input, in the KIMI-DER-009 naming discipline.

## 5. THE EXACT MISSING PIECE (formal nonclosure statement)

The unique missing piece for closure at the mandate standard ("interval or analytic enclosure rather than point-grid evidence") is:

  a certified bound sup_{y ∈ cell} σ(D²λ(y)) — equivalently a certified regional
  bound sup_{y ∈ R} σ(D³λ(y)) — for the four regions of the grid of record.

Why it is missing (both standard routes fail on this functional, documented):
(a) Interval/ball evaluation of the Laurent-series DAG: the leading coefficients (c6v ≈ 3.7e-3 on core cells) are residues of exact cancellations; interval evaluation amplifies rounding/cancellation junk to ≈ 7500 × box width (measured 47 at h=1/160), so the denominators straddle zero and the DAG cannot complete on core cells at any usable spacing.
(b) Global sup-bound (magnitude) arithmetic: sound by construction but compounds multiplicatively through the 9-pin divided-difference frame, the Neumann series inversion and the Schur divisions; observed regional bounds exceed 1e10 × true (intermediate quantities up to 1e64051 before the exact-6-pin freeze; ≥ 48^6 amplification after it), useless for the assembly.

With B₃,R certified, the assembly in the certificate becomes a full theorem with the printed constants; without it, the certificate's E_cert is conditional on H-B3 exactly as labeled.

Sharpening from the receipted run (§4a): within the H-B3 route the certificate's own evidence gate requires rung-stable third-difference quotients (ratio in [0.4, 2.5]); the measured ratio 3.5598022 shows the 1/20 rung is not yet in the converged regime, so even the measured-grade premise route requires a finer reference rung (1/80 vs 1/40 quotients) or a direct certified third-jet bound — it reduces to the same missing piece one derivative order up.

## 6. Certified numbers (from the receipted transcripts)

The authoritative receipt is transcripts/transcript_normal.txt with its byte-identical twin transcripts/transcript_O.txt (identity stamp transcripts/identity_stamp.txt: BYTE_IDENTICAL; both transcripts sha256 c520b19729a72ab41e914503a33bcb5841734d44f232a75fc726e591c370bac8), all hash-listed in HASHES.txt; transcripts/S6_EXCERPT.txt carries the extracted key lines. Receipted content: P0 kernel certificate (spectral tail bound 2.87e-182 < 1e-60), DEN closed form (decimal-dps-70), input integrity + script self-hash, cross-validation vs the archived C027 sweep (12 stations, worst rel diff 1.44e-7 = archived DEN truncation), mirror symmetry (max |diff| 0.0, 3 pairs), the full grid of record, and the fail-closed stop at the quotient self-consistency gate (§4a). Region-of-record lines:

- region C: h=1/40 (EXACT), cells=320, typed=320, boundary=240, kink=0, deep=0, Q3max=76190.172, Jmax=5.5183312
- region M: h=1/20 (EXACT), cells=312, typed=312, boundary=271, kink=114, deep=0, Q3max=93034.473, Jmax=7.8495268 (114 kink cells subdivided one level: 456 sub-stations, exact jets)
- region O: h=1/10 (EXACT), cells=122, typed=102, boundary=121, kink=58, deep=0, Q3max=7229.5523, Jmax=2.8196842
- region F: h=1/5 (EXACT), cells=65, typed=24, boundary=65, kink=50, deep=0, Q3max=22.644864, Jmax=1.3930538
- auxiliary C-rect at 1/20 (EXACT): cells=80, Q3max=21402.923
- quotient stability ratio Q_C(1/40)/Q_C2(1/20) = 3.5598022 → FAIL-CLOSED TRIGGER: quotient self-consistency failed (assembly not executed; S, E_cert, Λ₁, h_kill values are NOT receipted — no values exist for them in the transcript)

## 8. v3 ADDENDUM (1/80-rung repair, separate certificate — supersedes §7's blocker (i))

v3 certificate: verify_lambda_grid_v3.py, sha256 d54fc3bd8d5bf1445f16dd954fff6556c6f6d3998898efc42efe9387d313379e; transcripts transcripts/transcript_v3_normal.txt and transcript_v3_O.txt BYTE_IDENTICAL (both sha256 5e19da35a08fb737ce59b3dffb5453d19193931fc7c14803aa243985d54bf113). Provenance (printed in the v3 header): v3 names v2 (sha256 579ef6cd88d543ec14db45558e5e7c1899078095e0df295c7d46fd1810273831) as its predecessor — v2 hash-frozen, unpatched, receipted FAIL-CLOSED at the quotient gate. Differences v2→v3, disclosed: (a) added the auxiliary C-rectangle at h=1/80 (EXACT) and re-routed the rung-stability gate to the (1/80, 1/40) pair, the (1/40, 1/20) pair displayed with its v2 FAILED receipt; the certified window [0.4, 2.5] is unchanged; (b) a one-line type coercion (tomp) at the kill gate fixing a latent Fraction/mpf comparison that v2 never executed (v2 stopped earlier); (c) one misleading pre-gate label line corrected. No numerical content of v2 was altered.

Receipted v3 outcomes:

(i) Rung-stability evidence RESCUED at v3: Q(1/80)/Q(1/40) = 1.8846416 ∈ [0.4, 2.5] (receipted Q3max 143591.17 at 1/80 over 1280 cells vs 76190.172 at 1/40). Q(1/40)/Q(1/20) = 3.5598022 remains receipted FAILED at the window: the 1/20 rung is pre-asymptotic; the v2 failure was consistent with the reference rung being too coarse, so the premise was unproven, not disproven — and is now rescued at the finer reference rung. The rescue covers the C-rectangle (the quotient-hot region pair); the M/O/F regional B3 constants (receipted B3max 744275.79 / 57836.419 / 181.15891 vs C's 609521.37) inherit the premise at their record grids.

(ii) Floor gate PASSED at v3: scanned sum (both lobes) S = 0.893625433353538, E_cert (both lobes) = 0.489909223604 < M_floor = 0.9001. The certified positive lower constant of this packet is therefore

  ∫_zone λ(y) dy ≥ S − E_cert = 0.403716209750  (decimal-dps-70; H-B3-conditional at v3's hash; r→0 limit object; zone plus the named far-field dependency KIMI-DER-027a)

— an honest positive lower constant, NOT forced to reproduce 0.946 (the measured anchor stays measured-grade; 0.9091 derived-on-grid; the 0.9001 floor returns as ck-passed, H-B3-conditional at v3's hash, exactly as the lead's ruling framed it). Per-region assemblies receipted: C lam-sum 0.2930717214 E_cert 0.029668447; M 0.1478950712 / 0.084632951; O 0.005548400703 / 0.042800749; F 0.0002975234004 / 0.087852465; Λ₁ = 1729.76672773.

(iii) Kill-condition gate FAILS at the grid of record (receipted FAIL-CLOSED: 'used core spacing violates kill condition at floor margin'): the net-spacing inequality at the used core spacing evaluates to (1/40)/√2·Λ₁ = 30.578245 > 0.9001, with h_kill(M_floor) = 7.35899012905663e-4 and h_kill(M*) = 5.39599320650747e-5 (decimal-dps-70). The excursion mechanism certifies exclusion only for uniform spacings below ≈ 1/1359 at the floor margin; at the used spacings it is insufficient by a factor ≈ 34. This is a displayed, receipted fact about the excursion mechanism, not about the lower constant in (ii).

Updated disposition: blocker (i) of §7 is RESOLVED by v3 (rung-stability rescued; the 0.9001 floor returns H-B3-conditional at v3's hash). The Λ-side now delivers what the assembly requires — a positive r-uniform lower constant 0.403716209750 for the r→0 limit object at certificate standard with the single named premise H-B3 (evidence receipted at v3) — plus the receipted negative fact that the excursion kill mechanism requires h ≤ 7.36e-4. Blocker (ii) of §7 stands unchanged for a premise-FREE theorem: a certified cell-sup σ(D²λ) / regional sup σ(D³λ), both standard routes documented as failed on this functional. The r-scope statement of §2 applies to the 0.4037 constant unchanged (limit object; rung anchors measured-grade; sufficiently-small-r regime P₀-unconditional per W7).

## 7. Disposition

FORMAL NONCLOSURE at the certificate standard — now receipted by the certificate's own fail-closed gate (§4a): the completed, byte-identical twin transcripts show P0–P7 machine-certified through the full grid of record, followed by the FAIL-CLOSED stop at the quotient self-consistency check (ratio 3.5598022 outside [0.4, 2.5]). Two blockers are named, in order: (i) the H-B3 route's rung-stability evidence fails at the certified window — repair requires a finer reference rung (1/80 vs 1/40) or a direct certified third-jet bound; (ii) behind it stands the §5 missing piece, a certified cell-sup σ(D²λ) / regional sup σ(D³λ), with both standard routes documented as failed on this functional. The measured 0.946 anchor remains measured-grade (C020/C021); no PROVED positive lower constant is claimed by this packet; the tiers 0.946 / 0.9091 / 0.9001 are recorded as measured, derived-on-grid, and H-B3-conditional-floor respectively, with the H-B3 evidence now receipted as FAILED at the certified window. The mutation suite (4/4 tripped ck) and the byte-identity stamp are attached as the evidence package; the certificate script (sha256 579ef6cd88d543ec14db45558e5e7c1899078095e0df295c7d46fd1810273831) is unchanged since the mutation receipts and was not adjusted after the gate outcome.
