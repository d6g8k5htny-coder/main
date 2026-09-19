# PERC-DECAY — the far-excursion decay certificate for the far lanes

Theorem (2) v2.2 validity premise, D3 lane (OBL-B1-PERC). Classes:
B1.dir-far = FarRoute(d_max) (BRANCH_dir's handoff at d_max = 4r–5r, frozen
`8821c8fa…`), B2-far, B4.rem. Engine: `d3_perc_decay.py` (fail-closed,
deterministic, Monte-Carlo-free — every denominator is the R2-certified
floor `Z_lo = 7.7592917375327855e-3`, pinned artifact
`H3_RUNG_FLOOR.md` sha256 `6347275d86c5…`; both modes byte-identical:
`pd_normal.txt` ≡ `pd_opt.txt`). Frozen carriers (D3_PERCOLATION.md,
D3_REMOTE_AMENDMENT*.md) untouched.

BEGIN_FROZEN_BODY

## 0. The honest opening verdict (mandated)

**The o(r³)-order reading of the far lanes is NOT reachable with this
lane's machinery, and the registered percolation-decay input would not
deliver it either.** Each far lane is a window-priced event: its ORDER is
Θ(r³) (the window ℓ = r³/6), and every distance-decaying factor (RN
reversion, crude-spine cap density, kernel-scale connectivity) controls the
CONSTANT, not the order. The decisive display (certified means, §3): the
conditioned axial mean stays ABOVE s from S out to the ghost-elder
territory (x = 1.0, bottleneck z = 0.68 = Θ(1)) — the level-s connection
from the pair region toward the far terrain exists with Θ(1) probability at
the handoff distance d_max = 0.20–0.25 (sub-kernel), so the connectivity
factor in every far-lane price is Θ(1) and does not vanish as r → 0. The
far lanes are **Θ(r³), not o(r³)** (evidence-grade structural assessment on
certified displays; consistent with the joint-carrier structure of D1
v2.2's Theorem (2), which needs the far lanes inside C·r³, not literally
o(r³)).

**Recommended restatement of the premise** (this lane underwrites it):
PERC-DECAY := *the far lanes absorb at Θ(r³) with the certified constants
of §1, plus D-tails certified under the named input PD-CONN (§4)*.

## 1. Certified Θ(r³) bounds with exact inclusions (theorem-usable today)

- **B1.dir-far = FarRoute(d_max):** the away lobe's merge into C_M is a
  saddle of height in (s, b) — C_M is born at b (at M), the away lobe at s
  (at S); the merge is therefore a WINDOW SADDLE: FarRoute ⊆ {N_w ≥ 1}.
  Certified: P(FarRoute) ≤ E_{P_r}[N_w]; the D3-certified part outside the
  C1 chart ≤ B_remote + I_hole ≤ 21.9279 + 1.284 = **23.2119·r³** (v2
  bracket, floor-consistent, amendment v2 body `6796deea…`).
- **B2-far:** C2's exact separation (B2 ⊆ {N_β^* ≥ 1}, witnesses are window
  saddles): P(B2-far) ≤ E_{P_r}[N_w(d ≥ 0.2)] ≤ **17.6802·r³** (crude-spine
  cap grade, R2 floor; recomputed numerator over radii ≥ d_max).
- **B4.rem:** NOT ⊆ the window count — the non-local separator can close
  through ABOVE-window saddles (the away lobe joining an elder component at
  height > b) or the torus wrap. Certified content: the LOCAL dam (§2c);
  the remote part is the named gap (§4).

## 2. Certified decay displays (exact-kernel, per station)

(a) **Reversion profile** (away-axial and transverse rays): Var(f|6pins),
the excursion z = (μ−s)/σ, and κ_cross(d)/(m_sad·Z_lo) displayed at
d = 0.05…8: Var → 1 and κ_cross → 6.2e-8 along the ray; at the handoff
d_max = 0.20–0.25 the field is NOT yet reverted (Var ≈ 1e-3–1e-2 axial) —
the far lanes start INSIDE the conditioned zone, and the RN factor there is
carried by the crude-spine cap, not by the κ decomposition.

(b) **The annulus cap density decays** (frozen D4, exact-kernel certified):
ρ_cap = 7.8093e-7 @(0.1, 0°) → 1.3715e-9 @5 — the count envelope's own
decay display.

(c) **B4.loc dam re-verified** (plain six-pin law; answers the lead's grade
question for C2's exp(−c/r²) display): per-section, exact-kernel CERTIFIED:
clearance = the cubic Hermite profile ℓ(3ξ²−2ξ³) (displayed matching to
4 digits: 3.2218e-6 vs 3.2552e-6 at ξ=0.25), residual sd = (0.0072–0.0128)·r⁴,
κ_dam = 71.84 / 129.9 / 391.2 at ξ = 0.25 / 0.5 / 0.75 — κ = Θ(1/r), so the
per-section price Φ(−κ_dam) is super-algebraic; C2's ladder display
(log₁₀Φ(−κ_*) = −126 at this rung) is consistent. Grade: the PER-SECTION
bound P(sup_section f ≤ s) ≤ Φ(−κ_dam) is polarity-safe and certified; the
UNIFORM dam over the cut net is the dam line (uniformity = promotion scope;
the RN CS factor Θ(1) supplied).

## 3. The corridor-alive display (the decisive structural evidence)

Axial conditional mean μ(x) (plain six-pin law, CERTIFIED) from M through S
toward the ghost elder (1.052, 0): μ−s = +2.4e-8 at x=0.024 (≈S, pinned),
rising to +2.0e-5 @0.05, +2.3e-3 @0.2, +7.3e-3 @0.3, +1.6e-2 @0.4,
+2.8e-2 @0.5, +5.9e-2 @0.7, +9.4e-2 @1.0 — the mean NEVER dips below s
along the whole axis; the z-score decays from rigid (49.6 @0.024) through
7.3 @0.2 to the bottleneck 0.68 @1.0 = Θ(1). Between M and S the channel
stays above s rigidly (z = 42–4.2e4). Conclusion (evidence-grade): the
connectivity factor at the handoff is Θ(1); it does not vanish as r → 0.

## 4. The named-gap register and PD-CONN (deliverable-grade)

| class | certified bound | missing input | exact uncontrolled region |
|---|---|---|---|
| B1.dir-far | E_{P_r}[N_w]; D3-part ≤ 23.2119·r³ | PD-CONN | level-s connectivity between the away lobe (mean extent ≥ 1.0, §3) and C_M through the far field, d ∈ [d_max, O(1)] — the bridge annulus; the transverse part over (0, d_max] is certified (BRANCH_dir) |
| B2-far | E_{P_r}[N_w(d≥0.2)] ≤ 17.6802·r³ | PD-CONN | the far younger component touching S's sector lobes at level s, d ∈ [d_max, O(1)] — the same bridge annulus |
| B4.rem | local dam per-section certified (κ = Θ(1/r)) | PD-CONN + the above-window merge count | the M-ward channel beyond the local cut net (d > r); above-window merges (heights > b, not covered by the window count); the torus wrap circuit (d ~ 12) |

**PD-CONN (named input).** Let Q_r be the six-pin law at rung r ≤ 0.05,
s = b − r³/6, and A_r(K, D) the event that {f > s} has a connected
component meeting both K and the complement of B(K, D). REQUIRED: explicit
C₀, c₀ > 0, D₀ ≥ 1 (r-independent) with P_{Q_r}(A_r(B(pair, 2r), D)) ≤
C₀e^{−c₀D} for all D ≥ D₀, r ≤ 0.05. CONSUMPTION: each far lane's D-tail
≤ (certified count density)·C₀e^{−c₀D}·(1 + κ certified); class totals
absorb at Θ(r³) with certified constants; B4.rem's wrap ≤ C₀e^{−12c₀}.
MISSING PIECES to certify PD-CONN: (i) an explicit-constant arm bound for
planar Bargmann–Fock at positive level (the literature gives sharpness
WITHOUT explicit constants — an explicit RSW constant is the missing
theorem); (ii) the conditioned-to-unconditioned FIELD-LEVEL patch on
{d ≥ D₀} (finite-dimensional reversion certified here; the field-level
version needs continuity-modulus/entropy control — Kolmogorov machinery,
not built); (iii) the torus-to-planar comparison (T1/C.6: ≤ 1e-120,
available). NOTE (the verdict): PD-CONN certifies CONSTANTS, not order —
at sub-kernel D the connectivity factor is Θ(1) (§3), so the classes remain
Θ(r³); the o(r³) reading is not reachable by any input of this kind.

## 5. Window-count D-tail (why the decay is not in the count)

E_Q[N_w(d ≥ D)]/r³ = (576 − πD²)·J(ℓ)/r³ = 2.9268 / 2.9234 / 2.9115 /
2.8636 / 2.6720 / 1.9056 at D = 0.2 / 0.5 / 1 / 2 / 4 / 8 (exact, κ=0
kernel part): the raw count is Θ(r³) at every D < 12 — a class price's
D-tail decay lives entirely in the connectivity factor (PD-CONN), never in
the count.

## 6. Certificate machinery

Engine fail-closed (ck → SystemExit), deterministic, MC-free; both modes
byte-identical. Mutation suite (all fire): MUT-PD-1 register-taxonomy
tamper; MUT-PD-2 grade-label tamper (evidence presented as certified → the
grade-ledger ck fires); MUT-PD-3 R2 floor-pin tamper; MUT-PD-4 count-window
tamper; MUT-PD-5 corridor-display tamper. Grades are stated per display;
no evidence display is presented as certified (the grade ledger is ck'd).

END_FROZEN_BODY
