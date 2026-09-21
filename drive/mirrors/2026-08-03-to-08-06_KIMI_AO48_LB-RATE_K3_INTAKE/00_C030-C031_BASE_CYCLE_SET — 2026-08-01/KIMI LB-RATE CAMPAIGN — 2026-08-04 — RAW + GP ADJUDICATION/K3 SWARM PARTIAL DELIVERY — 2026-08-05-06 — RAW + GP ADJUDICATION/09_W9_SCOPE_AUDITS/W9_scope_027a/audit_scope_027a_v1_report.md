# K3 Phase 4 — DER-027a SCOPE AUDIT (W9_scope_027a)
**Certificate program:** `audit_scope_027a_v1.py` (fail-closed `ck()`; deterministic; normal
and `python3 -O` transcripts byte-identical; mutation tests in §A8; labels/receipts §A10).
**Starting record:** KIMI-DER-027a (WO-063 Task 4a), receipts in §A0 of the transcript.
**Model (EXACT):** periodized side-24 2D Bargmann–Fock field, spectral lattice (π/12)ℤ²,
masses e^{−|k|²/2}, truncation |k| ≤ 30, certified tail T₀/Z₃₀ ≤ 2.77e-194 (< 4.4e-185,
≪ 1e-60). Pins: (f, ∇f) at M = (−r/2, 0), S = (+r/2, 0), plus witness station
Y = r·(−63/50, 6/25); values b = 6/5, b − r³/6, v* at Y. Rungs r = 1/20, 1/40.
All numbers below are transcribed from the certificate transcript.

## 1. Recomputation (mandate item 1)
Every load-bearing quantity of DER-027a was recomputed from the spectral definition in a
fresh program and cross-validated against the DER-027a anchors:
- Z₃₀ = 91.67324722093171340287705 matches the independent Poisson identity
  Z∞ = (2π/a²)(1 + 4e^{−288} + …) to 5.4e-79 (dps-80 π-ulp scale; the certified tail is
  2.77e-194, so the normalization is cross-validated 10^115× below the tail bound).
- Origin moments ∂^{2i,2k}C(0) match the EXACT double-factorial values
  (−1)^{i+k}(2i−1)!!(2k−1)!! to < 1e-55.
- Certified Gram data: λ̂_min(G) = 2.5677067e-10 / 4.7566847e-12 (7-pin) and
  3.2552060e-10 / 5.0862628e-12 (6-pin) — reproduces DER-027a anchors; inverse residuals
  ε_M ≤ 4.9e-58; moment-coefficient certification error cWerr ≤ 8.7e-58.
- Envelope (Hermite-exact power–Gaussian sup functions P_{i,k}(d), Schur-complement
  moments, certified Taylor tails) reproduces DER-027a grid values to ~0.1% and is
  certified non-increasing on [3, 12] for all four configurations.
- Knot suprema cross-validated against DER-027a's F64PAD certificates by the
  cross-validity invariant (each framework's rigorous lower bound below the other's
  certified upper bound — all 12 knots PASS).

## 2. Exact scope of DER-027a (mandate item 2)
DER-027a proves MORE than variance reduction but LESS than a full law bound, in the
following exact sense (evidence tiers after this audit's upgrade):

| # | claim | d-range | r-range | conditioning family | tier |
|---|-------|---------|---------|--------------------|------|
| C1 | Δ̄(d) ≤ Ê(d), monotone non-increasing (value-variance deviation channel) | d ∈ [3,12], d₀ = 3 | {1/20, 1/40} exact + uniform (0, 1/20] (§4) | 7-pin, ANY pinned values | IB at knots {3,5,6.1} + DPS80 envelope between |
| C2 | Var(f(y)|pins) ≥ 0.979 (derived floor) | d ≥ 3 | as C1 | 7-pin, any values | IB |
| C3 | recorded constants 2.24% @3, 1e-4 @5, 7.5e-13 @6.1 | {3,5,6.1} | {1/20,1/40} exact | 7-pin (values-independent) | IB (interval box enclosures) |
| C4 | full one-point-law TV bound T̄(d), monotone | [3,12] | {1/20,1/40} exact | 6-pin with EXACT β = (6/5, b−r³/6) | DPS80 envelope; IB |μ| sup at d = 3, 5 |
| C5 | recorded value law 7.7e-5/8.1e-5 @ d=5 | d = 5 | r = 1/20 / 1/40 | 6-pin exact β | this audit: IB |μ| sup → T̄(5) = 8.01/9.09e-5; DER-027a: exact DPS80 TV at argmax 7.67/8.08e-5 |
| C6 | v* = −1/2 isolation: value-law stabilization FAILS | d = 5 | {1/20,1/40} | 7-pin tasked v* = −1/2 | IB certified |μ|̄(5) > 0.1 |
| C7 | uniform continuation Ê_unif(d;r) ≤ [(√Ê_J(d)+τ(r))/(1−τ(r))]² | [3,12] | ALL r ∈ (0, 1/20] | 7-pin & 6-pin families, any values (variance channel) | DPS80 certified τ + IB jet knots (certified spectral Lipschitz) |

So: the VARIANCE channel is a theorem about the conditioned field's law (it is a
statement on the conditional covariance operator, value-free); the VALUE-LAW channel is
a full one-point-law bound (mean + variance → TV) that holds only for the 6-pin value
family and for witness values consistent with the 6-pin mean at Y; it is NOT a full law
bound for the tasked 7-pin family (§5). No statement in DER-027a covers d < 3, r > 1/20,
or the continuum r ∈ (1/20, ∞) — except that C7 now covers r ∈ (0, 1/20].

## 3. Interval control replacing F64PAD (mandate item 3)
The heuristic float padding is REPLACED by rigorous mean-value box enclosures:
every knot supremum is now certified by covering the annulus {dist(y, hull(pins)) ≥ d}
with polar boxes and bounding Δ (resp. |μ|) on each box by interval arithmetic whose
every term is derived: (i) box half-widths from the certified Bker Lipschitz table
(BkM(m+eᵢ, din)·ρ); (ii) coefficient intervals from the certified cWerr; (iii) Taylor
tails R (certified, q-scaled from a common certified base); (iv) inverse-residual term
ε_M·vcrude; (v) image+truncation corrections kcorr; (vi) a stated float64 center-rounding
bound (2⁻⁵³ per rounding, Horner-count factor). Branch-and-bound: any box whose
upper bound exceeds the best rigorous lower bound is replaced by 3×3 sub-boxes (2 rounds).
Tier reached: **full interval grade (IB) at all recorded knots, INCLUDING d = 3**
(transcript values; rigorous lower bounds in parentheses):
- 7-pin r = 1/20: Δ̄(3) ≤ 0.0209033 (0.0174637); Δ̄(5) ≤ 3.80652e-8; Δ̄(6.1) ≤ 6.1046e-13.
- 7-pin r = 1/40: Δ̄(3) ≤ 0.0216751 (0.0186422); Δ̄(5) ≤ 4.20004e-8; Δ̄(6.1) ≤ 7.03871e-13.
- 6-pin r = 1/20: Δ̄(3) ≤ 0.0201457 (0.0177549); Δ̄(5) ≤ 3.56026e-8; Δ̄(6.1) ≤ 5.46773e-13.
- 6-pin r = 1/40: Δ̄(3) ≤ 0.0215501 (0.0190828); Δ̄(5) ≤ 4.07769e-8; Δ̄(6.1) ≤ 6.56556e-13.
All recorded constants certified: 2.24e-2 at d = 3, 1e-4 at d = 5, 7.5e-13 at d = 6.1
(PASS, all four configurations). Certificates carry ≤ 20% pad over the rigorous lower
bounds; the pad is now derived, not heuristic. Between knots the derived DPS80 envelope
Ê (all of whose constants are certified bounds) carries the monotone statement; the F64
label survives only for center evaluations, never load-bearing without IB half-widths.

## 4. Uniform continuation in r (mandate item 4) — SUCCEEDED on (0, 1/20], weaker constant
Attempt: the raw Gram frame is NONUNIFORM — measured λ_min(G(r)) ∝ r^{5.75} (7-pin)
/ r^{6.0} (6-pin) — the pins collapse to 3 independent functionals at r = 0 (rank drop
7 → 3), so any envelope built on ‖G(r)⁻¹‖ norms diverges as r → 0. This is the named
nonuniform obstruction of the raw frame; it is bypassed by the jet frame: the pin
functionals are Taylor-expanded about the origin to jet order N = 8 with certified
residual norms ‖ρ_p‖² (exact moment series), the realization matrix Â (pins × jets
≤ K′ = 4, entries ξ_p^{m−α}/(m−α)!) is certified full rank (σ̃_min = 0.0359 (7-pin) /
0.0524 (6-pin), σ_min(A(r)) ≥ σ̃_min·r⁴), and the jet Gram (EXACT double-factorial
entries) is certified PD (λ_min = 0.038445). With τ(r) = c_res(r)/(σ̃_min r⁴ √λ_J):
- τ(1/20) = 6.78e-6 (7-pin), 5.03e-7 (6-pin) — both < 1 certified;
- THEOREM (uniform continuation): for all r ∈ (0, 1/20], all d ∈ [3, 12],
  Δ̄(d; r) ≤ Ê_unif(d; r) = [(√Ê_J(d) + τ(r))/(1 − τ(r))]², non-increasing in d, where
  Ê_J(d) is the r-free jet-ensemble deviation, certified at interval grade by the same
  box-enclosure layer with certified per-q spectral Lipschitz constants
  Λ_q = Z₃₀⁻¹Σ_k m_k|Q_q(ik)|·(|k_1|+|k_2|) + certified tail (max Λ_q = 1.49) — the
  per-term triangle bound is vacuous for this ensemble (cancellation) and is used only
  beyond t = 8.6 where it is < 1e-4-class;
- values (transcript): Ê_J(3) = 0.612786 (rigorous lower bound 0.5086), Ê_J(4) = 0.140987,
  Ê_J(5) = 0.140497, Ê_J(6.1) = 0.139523, Ê_J(≥ 6.1) = 0.139523 (running-min monotone
  envelope, valid by annulus nesting); Ê_unif(3; 1/20) = 0.612804 (7-pin) / 0.612787
  (6-pin), Ê_unif(6.1; 1/20) = 0.13953, Ê_unif(12; 1/20) = 0.13953; dominance over all
  exact-rung rigorous lower bounds verified (PASS). The d ≥ 4 certificates are
  pad-limited (~0.14 from the global spectral Lipschitz constants; the true jet
  deviation there is ~0.01–0.001) — valid upper bounds, stated honestly.
  The uniform bound is VALID but WEAKER than the exact-rung certificates (0.021):
  conditioning on all 45 jets removes ~51% of the variance at d = 3, far more than the
  7-pin projection (2%), so the jet-frame continuation inherits the larger envelope.
  Exact-rung status (C1–C3) is preserved and remains the sharp statement at the rungs;
  the uniform theorem is the correct statement for the continuum r ∈ (0, 1/20], with
  the r → 0 limit the exact 3-jet ensemble Δ₃jet(y) = C(y)² + |∇C(y)|²;
- N = 8 was chosen for τ margin (τ ~ 7e-6); N = 6 already gives τ ~ 0.1 < 1 (valid);
  N = 4 gives τ ~ 0.11 with a smaller projection — the (N, constant) trade-off does not
  improve on 0.61 materially, so N = 8 is the run configuration of record.

## 5. Isolation: scope boundary variance vs value-law (mandate item 5)
The v* = −1/2 isolation is CONFIRMED at interval grade and marks the exact scope
boundary: the value −1/2 at Y is macroscopically inconsistent with the 6-pin conditional
mean ≈ 1.2 there, forcing dual weights ~ 1e5 and a macroscopic far mean:
- certified |μ|̄(5) = 1.24362 (r = 1/20, argmax y = (4.9959, 2.0129)) and
  5.11551 (r = 1/40, argmax y = (4.9679, 1.6872)) ≫ 0.1 — value-law stabilization
  FAILS for the tasked 7-pin family (one-point TV ≈ 0.44 / ≈ 0.99);
- the variance channel is provably unaffected (mutation test M1: Δ bitwise-invariant
  under value mutations — pinned values never enter the Schur-complement variance);
- with the C022-consistent witness value b − ℓ/2 at Y, TV(5) ≈ 8.5e-5/8.2e-5 (≤ 1e-4).
For the 6-pin family, the full one-point-law bounds are: IB |μ|̄(3) = 0.167962/0.178056,
|μ|̄(5) = 2.00754e-4/2.27920e-4; TV knot bounds T̄(3) = 0.0818/0.0859 (consistent with the
recorded 6.6%–31% band), T̄(5) = 8.010e-5 / 9.094e-5 ≤ 1e-4 (recorded 7.7/8.1e-5;
certified bounds exceed the recorded measured values by 4%/12% — expected direction,
margin note; DER-027a's heuristic-pad layer gave the tighter 7.88/8.30e-5).
Scope boundary, exactly: variance-channel claims (C1–C3, C7) hold for the pin POSITION
family with arbitrary values; value-law claims (C4–C5) hold only for value vectors in a
neighborhood of the 6-pin-consistent values (the witness must carry b − ℓ/2-class values).

## 6. Mutation tests (transcript §A8)
M1 value mutation → variance channel bitwise-invariant (PASS, proves value-freeness);
M2 mesh refinement → certified sup non-increasing (PASS); M3 kernel sign-flip → caught
by the cross-representation check (PASS); M4 envelope scaled ×1e-3 → fails the
dominance check as required (PASS).

## 7. Hypotheses, dependencies, falsifier
Hypotheses: H1 exact model as in the header; H2 pin positions/values as tasked;
H3 recorded constants from C022/LB-3 as reconciliation targets (not axioms);
H4 mpmath dps-80 elementary functions correctly rounded to ≤ 1 ulp (used only inside
certified-residual frameworks). Dependencies with hashes: transcript §A0/A10 receipts
(WO-063 e699bf44…, C022 9bc0647b…, c022 fd 394a9599…, DER-027a md/py/transcripts).
Deliverable hashes (sha256):
audit_scope_027a_v1.py = 12af9e2db57c950b1f2e1a364fba871c9db5e878ee624618b400c16c69e937e0;
transcripts (normal and -O, byte-identical) = d6cfc8fb7b351ee21e24b9e941e8e5747d51f2487ee6e8bb71acab2b9d244f46;
94 checks PASS, 0 FAIL.
Falsifier: recompute any one of {IB knot certificates, τ(1/20), σ̃_min, λ̂_min(G),
Ê grid} from the spectral definition; if any transcript value is exceeded by its bound
(any FAIL under `python3 audit_scope_027a_v1.py`), the corresponding claim is false.

## 8. Remaining gap (one paragraph)
The interval layer certifies the recorded knots but its pads (~8–12% at d = 3, ~10–16%
at 6.1e-13 scale) remain above the old heuristic layer's; closing that last factor is a
sharpness question (direction-resolved Hermite sups), not a rigor gap. The uniform
continuation covers r ∈ (0, 1/20]; continuation to r ∈ (1/20, R] is open (the jet
certificate works for any r with τ(r) < 1 — τ grows like r³·poly and the threshold
r* ≈ 0.08–0.1 could be certified on demand — but pin separations then leave the
near-field regime the envelope was tuned for, and the d = 3 recorded constant is not
expected to survive much larger r). The value-law scope boundary is exact as stated in
§5: nothing in this audit (or DER-027a) rescues the tasked v* = −1/2 family for law
bounds; if the K3 frame needs a 7-pin law theorem it must re-pin Y at b − ℓ/2-class
values (TV(5) ≈ 8.5e-5, this audit's float64 check) or accept isolation C6 as the
definitive answer for v* = −1/2. The d = 3 analytic envelope Ê(3) (0.054–0.121) still
does not by itself match the recorded 2.24% — the recorded knot is carried by the IB
layer; a pure closed-form proof of the d = 3 constant remains open.
