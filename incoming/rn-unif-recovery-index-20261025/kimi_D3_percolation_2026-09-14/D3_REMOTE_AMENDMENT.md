# D3 REMOTE AMENDMENT — the certified κ_far and the assembly-facing remote bracket

Stage E repair R3, amendment to the frozen D3_PERCOLATION.md (body sha256
`8e7fef6b…`, UNTOUCHED). Finding repaired: the frozen D3-THM stated
I_far ≤ (576−25π)·J(ℓ)·(1+κ_far) with "κ_far(5) ≤ 0.63 + ε_QMC" — a QMC
estimate riding inside a certified constant — while D1 v2.0/H5 consumed the
κ=0 display 19.5483·r³ flat; and the lane's displays were internally
inconsistent (engine 20.8305 vs THM 20.9 with no discriminating gate).
Engine: `d3_amend.py` (fail-closed, deterministic, both modes byte-identical:
`ta_normal.txt` ≡ `ta_opt.txt`). Falsifier: `d3_falsifier.py` (updated:
discriminating bracket window + mutation suite).

BEGIN_FROZEN_BODY

## 1. The certified κ_far decomposition (ε_QMC resolution)

At the axis-worst station (d=5, θ=0°), r=0.05, v = window mid, every factor
exact-kernel (mpmath dps=100, residual-certified linear algebra; the
endpoint variation v ∈ {s, b} displayed and absorbed):

    ρ^{P_r}(y)/ρ⁰(y) ≤ R_pg · R_wm · [(1+κ_pair)(1+κ_y) + κ_cross]

    κ_cross = 0.62556315      the frozen 1-Lipschitz regression bound
                              √3(EW⁴)^{1/4}(Eq²)^{1/4}√τ / (Z_r m_sad)
                              [τ-form; certified; relative at probe Z_r,
                               CONSUMES G.7-scope two-sided normalizer]
    κ_pair  = 0.0024234       pair-block law shift Z_r^{yv}/Z_r, certified by
                              the EXACT chi-square closed form:
                              |E_9[h]−E_6[h]| ≤ ||h||_{2,6}√χ²(p9||p6),
                              χ² = 1.9443e-6 via determinants (M ≻ 0
                              certified, min-eig 0.125); keeps the FULL typed
                              W_r (trace indicator included — no Lipschitz
                              smoothing, no Σ⁻¹ score blow-up)
    κ_y     = 0.02437         y-block typed moment m_y^{yv,res} vs the closed
                              form m_sad(v): W2-coupling bound, W2² = 1.84e-5
                              (Bures trace Weyl-clamped, residual-certified)
    R_pg    = 0.99999994      exact pgrad ratio at the station
    R_wm    = 1.0002592       exact needle window-mass ratio

    κ_far assembled = 0.65284336
    κ_far ≤ 0.66              CERTIFIED (round-UP absorbing the displayed
                              endpoint variation ±1.2e-5 and all station
                              margins)

**ε_QMC resolution: the QMC is demoted to consistency display only.** The
certified constant 0.66 contains no Monte-Carlo factor: κ_cross, κ_pair,
κ_y are exact-kernel bounds; R_pg, R_wm exact; the Z_r probe value
(8.06118054e-3) appears only inside the relative forms' denominators under
the campaign-wide [G.7-scope] two-sided normalizer consumption (verbatim the
C2 §5.0 polarity caveat). The QMC displays (g/(Z_r m_sad) ∈ [0.098, 1.38],
1.00±0.01 at d=5) sit far below the certified κ and are labeled evidence.

## 2. The assembly-facing certified bracket (task 1: option (a))

    B_remote := I_ann + I_far ≤ 17.0183 + 2.52826·(1+κ_far),  κ_far ≤ 0.66
              = 17.0183 + 4.19692 = **21.2153·r³**   (upper bounds rounded UP:
              I_ann 17.0182368→17.0183; I_far 4.1969178→4.19692;
              sum 21.2152178→21.2153)

    exact consumable form for H5's symbolic bracket:
        B_remote = 17.0183 + 2.52826·(1+κ_far),  κ_far ≤ 0.66 (certified),
    and the full remote-class statement:
        P_r(A.rem) ≤ I_hole + B_remote ≤ 1.284 + 21.2153 = 22.4993·r³
        [I_hole CONSUMES C1's chart envelope, as frozen].

Option (b) (zone-averaged κ) was assessed and NOT taken: the zone-uniform
certification of the cross bound is blocked by the rigid-direction
cancellation problem (§4) — per the mandate, (a) is delivered instead. The
decay display (A1b, exact-kernel evidence) shows the pieces collapsing with
d — κ_cross: 0.6256 (5,0°), 0.0474 (5,90°), 0.0103 (6,0°), 4.9e-5 (7,0°),
7.2e-8 (8,0°); κ_pair: 2.4e-3 → 8.2e-11; κ_y: 2.4e-2 → 8.9e-10 — so a future
zone-averaged form would sit near 19.6, but its certification needs the
missing uniformity input (§4), not more computation of this kind.

## 3. Reconciliation of the three displays (task 2)

    (i)  engine κ=0 remote sum: 17.0182 + 2.52826 = 19.5465·r³ — a POINT
         DISPLAY (RN factor at its QMC-central value 1), NOT an upper bound;
         the falsifier gate now rejects it as the bracket (§5);
    (ii) frozen THM display "≤ 20.9·r³" (total incl. I_hole): the same κ=0
         reading rounded up — superseded by this amendment;
   (iii) certified bracket (this amendment): B_remote ≤ 21.2153·r³, full
         P_r(A.rem) ≤ 22.4993·r³ (rounded UP).
   Difference vs the R3 finding's 21.14: the finding kept κ = 0.63
   (κ_cross alone); the full decomposition adds κ_pair = 0.00242 and
   κ_y = 0.02437, i.e. κ_far ≤ 0.66 — stated so D1 v2.1/H5 consume the
   complete certified value.

## 4. D3-LEMMA-RN-UNIF: exact scope at the rung r = 0.05 (task 3)

**Disposition: NOT closed; the missing pieces are precisely identified.**

Closed at r=0.05 (this amendment + the frozen engine): every κ piece at the
axis-worst station, exact-kernel certified; the far-zone main term
(576−25π)J(ℓ) exact; the monotone-decay evidence on the exact-kernel nets
(axis worst at d=5, all pieces decreasing; A1b table).

OPEN — precisely:
  (i) Far-zone uniformity of κ_cross and κ_pair: needs sup over {d≥5} of
      τ(y) = tr(ΔᵀΣ_pair⁻¹Δ) and of χ²(p9||p6). Entrywise kernel envelopes
      overbound τ by ×5.3e3 at the reference station (displayed:
      ||Δ||²_F/λ_min = 1.881 vs true τ = 3.53e-4) because the pins EXPLAIN
      the rigid directions of the pair block — the cancellation is between
      the bare covariance and the pin-regression reduction, which no
      entrywise envelope can see. Missing: the named
      **RIGIDITY-DECOUPLING lemma** (for foundations): the rigid directions
      of the pair Hessian block (λ_min = 2.6e-10 at r=0.05) lie in the
      pin-determined subspace, hence their conditional covariance with
      remote jets carries an explicit extra decay factor. Evidence
      (displayed): rigid-aligned ||v_RᵀΔ|| = 2.7e-7 / 6.0e-7 vs generic
      ||Δ||_F = 2.2e-5 — 80×/37× below generic.
 (ii) Annulus crude-spine zone-integral uniformity (trapezoid → certified
      Riemann sum over {2r ≤ d ≤ 5}): needs interval evaluation of
      ρ_spine(y) over a box covering; the machinery exists (cov_exact iv
      mode carries certified tails; inversion residual-certified), costed
      ~10³ boxes; not built in this lane.
The uniform-in-r part remains OPEN as frozen (named, foundations
countersignature).

## 5. Falsifier changes (task 4)

`d3_falsifier.py` now carries the discriminating bracket gate (F3):
parse B_remote from the amendment transcript and require
    20.5 ≤ B_remote ≤ 22.5  AND  B_remote ≥ 19.5483 + 1.0
— the κ=0 sum 19.5465 is rejected by construction; the mutation suite
(F4) feeds the gate the κ=0 sum, the old 20.9 display's remote part, a
below-range and an above-range value, and requires rejection of exactly the
first group (self-test of the gate), plus hash-pin tamper mutations of the
amendment engine requiring SystemExit. Both modes byte-identical
(`tf_normal.txt` ≡ `tf_opt.txt`).

## 6. Grade statement

Theorem-grade (fail-closed certified, exact-kernel): the κ_far decomposition
(κ_cross τ-form; κ_pair χ² closed form; κ_y W2/Bures), the assembled
κ_far ≤ 0.66, the bracket B_remote ≤ 21.2153·r³, all rounding directions.
Consumption (verbatim the frozen lanes): the G.7-scope two-sided normalizer
inside the relative forms. Evidence-labeled: the Z_r QMC probe (consistency
anchor for the relative forms), the A1b decay display, the rigidity-aligned
norms. Open: D3-LEMMA-RN-UNIF per §4.

END_FROZEN_BODY
