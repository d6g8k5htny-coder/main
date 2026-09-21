# K3-W7 — r₀(P₀) addendum for the successor assembly (K3 Phase 5 support)

**Parent:** K3_W7_GAMMALOC_AUDIT.md (sha256 `03f2e944ecc054dd36e92f907ed2a2c39b5143147716110da427d068954cc416`), disposition DISCHARGED-BY-MINIMALITY, confirmed by W10 (`W10_REPORT.md` sha256 `67c7fe63e168060b9db1f2279de5d88e7f804508a0e4e98a90afbec468ee21d1`: far tier AO⁰ = 0.089569·P₀, constant class; falsifier F4 does not fire). **Receipts:** `verify_k3_w7_gammaloc_v1.py` (sha256 `e16054ef132749029f6dd0075f26d609fa296905db470c74ef300adcbf9c84f8`) section S2, transcript `verify_t_normal.txt` = `verify_t_O.txt` (sha256 `19c3c0099e0c1ca06fd1908827baeb40534e71759dbfd28b4efa244cc7864310`, byte-identical both modes). No status or Boolean claims.

## 1. The r₀(P₀) function, explicit

**Setting (labels of record).** Floor constant c_cond = κ·P₀ with κ = e^{−4.82550294/2} = 0.089569 [decimal-dps-6; ‖ũ₀‖²_H = 4.82550294 decimal-dps-8, C028 exact-mp; κ recomputed at dps-60 = 0.089568510149350869443, |κ − 0.089569| < 5e-7]. Near (R2) band-max term: 2.1·(ℓ/2) with ℓ = r³/6 [EXACT], i.e.

> near(r) = 2.1·(ℓ/2) = 7r³/40 = 0.175·r³ [EXACT, ledger form "2.1"; area-ratio form 19/9 gives 19r³/108 = 0.175926r³ [EXACT] — see §3].

**Derivation (receipt S2, machine-pinned).** The γ-LOC-min floor is P(terminal > b) ≥ c_cond − near(r) − [named non-adjacency channels]. Driving the near term below the θ-share of the floor:

> near(r) ≤ θ·c_cond  ⟺  7r³/40 ≤ θ·κ·P₀  ⟺  **r³ ≤ (40θκ/7)·P₀**.

At the standard split θ = 1/2 (floor retains ≥ c_cond/2):

> **r₀(P₀) = (20κ/7)^{1/3}·P₀^{1/3} = (0.25591142857142857·P₀)^{1/3} = 0.634887184·P₀^{1/3}** [coefficient decimal-dps-17 / r₀ decimal-dps-10].

Evaluations (decimal-dps-10): P₀ = 1 → r₀ = 0.634887184; P₀ = 3.2e-4 (C028 honest-null 95% UB, measured) → **r₀ = 0.04342567254** (does NOT cover the r = 0.05 rung); P₀ = 1e-4 → r₀ = 0.02946885264; P₀ = 1e-6 → r₀ = 0.006348871840. Consistency side-conditions (non-binding): band entry needs ℓ/2 ≤ δ₀ = 1/5, i.e. r³ ≤ 12/5 [EXACT]; all displayed r₀ values satisfy it.

## 2. The precise ∃-versus-explicit statement

- **P₀-unconditional existence (the ∃-form).** κ > 0 is explicit and P₀ > 0 holds by the Gaussian support theorem (C028: the seminorm ball on D = [0,1/2]×[−3/10,3/10] at η = 0.1307 is open, symmetric, convex, and 0 lies in the support of the conditioned centered field on C¹(D̄); Cameron–Martin shift + Anderson). Therefore **for every P₀ > 0 there EXISTS r₀ > 0** — namely the displayed function r₀(P₀) — such that near(r) ≤ c_cond/2 for all r ∈ (0, r₀], hence the γ-LOC-min floor satisfies P(terminal > b) ≥ c_cond/2 > 0 on that interval. This statement is free of any numeric P₀ floor: positivity of c_cond and the displayed shrinking law suffice jointly (W10 Form B pattern).
- **Explicit r₀ (the explicit-theorem form).** A *number* r₀ requires an explicit positive lower bound P₀ ≥ P₀⁻, because r₀(P₀) ∝ P₀^{1/3} vanishes as P₀ ↓ 0. The record holds only an upper bound (G-D4b honest null: measured 0/12000, 95% UB 3.2e-4, decimal-dps-2) — no floor. This is exactly **OBL-P0-FLOOR** (C028's optional named residual, C031 register item 8): discharge it with any explicit P₀⁻ > 0 and r₀ = 0.634887184·(P₀⁻)^{1/3} is immediate [decimal-dps-10 coefficient]. Warning for the assembly: at the current evidence ceiling P₀ = 3.2e-4 (an upper bound, not a floor), r₀ = 0.04342567254 < 0.05, so the r = 0.05 rung is not certifiable from this channel without a P₀ floor.

## 3. Scope, sensitivity, and assembly notes

- **Scope.** This r₀ governs only the dominance of the R2 near band-max term against the floor. The assembly's other o(1) channels carry their own shrinking laws (e.g. W10 §5.4's WP-channel display r₀ ≤ (0.089569·P₀/(2·5.5e-3))^{1/1.6} = 0.0117 at P₀ = 1e-4 [decimal-dps-2/4, observed-envelope grade]) — at P₀ = 1e-4 the R2 channel (0.02946885264) is NOT the binding constraint; the effective assembly r₀ is the minimum over channels.
- **Sensitivity of the coefficient.** Ledger form "≈ 2.1·(ℓ/2)" (C031 print): coefficient 20κ/7 = 0.25591142857142857. Area-ratio form 0.76·(25/9) = 19/9 [EXACT; the cross-check |19/9 − 2.1| = 1/90 is receipt-pinned]: near = 19r³/108, coefficient 54κ/19 = 0.25456452631578947 [decimal-dps-17]. The structure **r₀ ∝ (θ·κ·P₀)^{1/3}** is invariant; the coefficient moves 0.53% between the two R2 prints. The assembly should evaluate the coefficient with whichever R2 constant it certifies.
- **What this addendum does not do.** It does not floor P₀, does not touch the far-selection item (OBL-FAR-COMPOSE route (ii) / selection-lemma class), and makes no use of any 1−O(exp) terminal-value rate (none is consumed; W10-confirmed). The general split form is r³ ≤ (40θκ/7)·P₀, floor ≥ (1 − θ)·c_cond, any θ ∈ (0, 1) [EXACT algebra].
