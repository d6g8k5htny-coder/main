# B4LOC_DAMLINE.md — the B4 dam-line tube certificate

Artifact: B4LOC-R1-20260914-v1.0
Lane: D2 branch control (U2D-UPPER campaign, side-24 periodized normalized
Bargmann–Fock field).
Scope-ruling context: Stage-E ruling V2 — with no raw counted-class envelope
for B4.loc anywhere in the carriers, the dam line is the only route to ANY
O(r^3) bound on that summand.  This document closes it at certificate grade.

Frozen carriers consumed (hash-pinned, freeze-before-consume):
  C2_B_CLASSES.md body 2aba099d… (B1 taxonomy: B4 definition, §4 dam family,
    kappa_* = Theta(1/r), the registered open item)
  D2 body 9e95648a… (the 9-pin M-ward tube, kappa_dam scale-invariant)
  B1_taxonomy body 7d7ddbec… (PL ensemble + classify_pair)
  H4_CLOSURE.md body a67d50b9… (H4-RN theorem, H4-SD)
  H2_foundations/cov_exact.py f08c1c5f… (exact covariance, spectral moments)
  b1_falsifier.py 818df798… (PL model consumed by the discrete mirror)

Notation.  Q_r = the 6-pin law (f(M) = b = 6/5, ∇f(M) = 0, f(S) = s = b − ℓ,
∇f(S) = 0; ℓ = r^3/6; M = (−r/2, 0), S = (r/2, 0)); P_r = W_r dQ_r/Z_r.
The B4 dam event (C2 §4): the separating event between M and S's sectors at
level s.  B4.loc = its local (pair-level) slice; B4.rem = the remainder.
This certificate bounds ALL of B4 (loc + rem), which is strictly stronger
than the registered B4.loc item.

====================================================================
BEGIN_FROZEN_BODY
====================================================================

## 1. THE IDENTIFICATION — DISPOSITION (b): the pair-level cut net is NOT
##    the 9-pin M-ward tube.  No asserted identification survives.

The registered caveat reads: "the pair-level cut-net ≡ the 9-pin M-ward-tube
item (ii) identification is ASSERTED, NOT ESTABLISHED."  We resolve it
negatively.  The two events differ on four independent axes, each pinned to
the frozen carriers:

  (i) LAW.  The cut net lives under the pair-level 6-pin law Q_r (C2 §2:
      pins at M and S only).  The M-ward tube lives under the 9-pin law
      (D2 body 9e95648a…: the 6 pins plus the 3-jet at the branch point y,
      a Palm conditioning at y).  Different σ-algebras of conditions,
      different conditional variances at every point of the corridor.
  (ii) TUBE.  The 9-pin tube is S's M-ward channel selected by y's jet
      (D2-COMPLEMENT's 4-channel decomposition: the tube is one of four
      angular sectors of the branch at y).  The pair-level cut net is the
      full set of separating cuts of the M–S pair at level s, with no
      reference to any branch point y or channel.
  (iii) LEVEL.  The tube is pinned at the branch level v = f(y); the cut
      net is pinned at the pair level s = f(S).  H4-JC-R1 (H4_JC_repair,
      frozen) is precisely the finding that level-v and level-s events are
      distinct with positive-probability discrepancy (death-saddle
      double-membership, 175/175 in the mirror).
  (iv) SCALING.  The 9-pin tube margin is scale-invariant: kappa_dam =
      Θ(1) (D2's certified headline, halving ratios 0.886–0.997).  The
      pair-level cut-net margin scales as kappa_* = Θ(1/r) (C2 §4, and
      §6 below: certified kap·r ∈ [0.467, 0.533] and [0.722, 0.769] for the
      two zones).  A Θ(1) margin cannot dominate a Θ(1/r) margin family;
      the events cannot be identified.

Disposition (b) is therefore in force: the pair-level certificate is built
DIRECTLY (§2–§4), and the 9-pin M-ward tube is recorded as the PROOF
TEMPLATE: the corridor-geometry + Borell–TIS + entropy architecture of §3
is the pair-level reincarnation of the tube argument, with the 6-pin law
replacing the 9-pin law and the Θ(1/r) margin law replacing kappa_dam.

## 2. THE EVENT-INCLUSION LEMMA (B4 ⊆ E1 ∪ E2)

Corridor: q(ξ) = S − rξ·x̂, ξ ∈ [0, 1] — the straight axis from S to M.
Zones: Z1 = (0, ξ0], Z2 = [ξ0, 1), ξ0 = 1/4.

LEMMA B4LOC-R1-A (deterministic topological step).  For every C^2 field f
in the B4 dam event (M a local max, S the pair saddle, M separated from
S's upper sectors at level s), at least one of
    E1:  ∃ ζ ∈ (0, ξ0]  with  f_xx(q(ζ)) ≤ 0
    E2:  ∃ ξ ∈ [ξ0, 1) with  f(q(ξ)) ≤ s
holds.  Hence B4 ⊆ E1 ∪ E2 as events (and a fortiori B4.loc ⊆ E1 ∪ E2).

PROOF.  Suppose both fail: f_xx > 0 on all of Z1 and f > s on all of Z2.
Taylor with integral remainder along the corridor, using the pin data
f(S) = s and ∇f(S) = 0:
    f(q(ξ)) − s = (rξ)^2 ∫_0^1 (1−t) f_xx(q(tξ)) dt.
For ξ < ξ0 the whole segment q([0, ξ]) ⊂ Z1, where f_xx > 0, so f(q(ξ)) > s
— extending the ¬E2 hypothesis onto Z1 as well.  Thus the full open
corridor q((0,1)) lies strictly above s, with endpoint values f(S) = s,
f(M) = b > s and ∇f(S) = ∇f(M) = 0.  The corridor is then an upper path
from S to M at level s: following the corridor out of S, the level-s upper
component containing the corridor reaches M.  Hence M belongs to S's
level-s upper cluster along the corridor direction — i.e. S has an
M-ward-open sector and M is NOT separated from S's sectors at s: the field
is not in B4 (for the local slice: the separator does not exist; for the
remainder: the M-ward channel is open, so the remote obstruction defining
B4.rem is absent).  Contrapositively B4 ⇒ E1 ∪ E2.  ∎

Remark (why two zones).  Near S the value margin f − s is O(r^2 ξ^2)-thin
and cannot be certified at the needed grade; the curvature test E1 carries
the near-S region (a separating cut must pass where the ridge is convex-up,
forcing f_xx ≤ 0 somewhere in (0, ξ0]).  Away from S the value test E2 is
the cheap cut.  The split at ξ0 = 1/4 balances the two certified exponents
(43.5 and 107.9 at r = 0.05; §4).

## 3. THE CERTIFICATE (uniform Gaussian sup-tails over the cut net)

THEOREM B4LOC-R1.  For r ∈ {0.05, 0.025, 0.0125},
    P_r(B4)  ≤  P_r(E1) + P_r(E2)
             ≤  C_RN(r)·( √Q_r(E1) + √Q_r(E2) )  ≤  r^3,
with every factor explicit and certified:
    r = 0.05:   P_r(B4) ≤ 1.22e-9    (r^3 = 1.25e-4;  ratio 9.77e-6)
    r = 0.025:  P_r(B4) ≤ 1.52e-45   (r^3 = 1.56e-5;  ratio 9.73e-41)
    r = 0.0125: P_r(B4) ≤ 1.03e-197  (r^3 = 1.95e-6;  ratio 5.28e-192)

The chain of explicit factors:

(3a) RN transfer (consumed theorem-grade, event-independent).  H4-RN
(H4_CLOSURE body a67d50b9…, §3): P_r(F) ≤ C_RN(r)·√Q_r(F) for every event
F, C_RN(r) = √(E_Q[(det H_M det H_S)^2])/(c_Z r^2) ≤ 3.47 on the ladder
(numerator an exact Isserlis polynomial moment, denominator H3's closed
partition floor).  Consumed as C_RN = 3.47; the driver fail-closes if the
consumed value drifts above it (mutation M3).

(3b) Gaussian sup-tails under Q_r.  Both E1 and E2 are sup-events of
centered Gaussian processes on compact parameter intervals:
    E1 = { sup_{ζ∈(0,ξ0]} (−f_xx(q(ζ)) + h2(ζ)) ≥ h_min }  (zone process
         X1(ζ) = −f_xx(q(ζ)) − E_Q[f_xx(q(ζ))], threshold h_min =
         inf_ζ E_Q[f_xx(q(ζ))] > 0),
    E2 = { sup_{ξ∈[ξ0,1)} (s − f(q(ξ)) + u(ξ)) ≥ u_* }     (X2(ξ) =
         −(f(q(ξ)) − E_Q[f(q(ξ))]), threshold u_* =
         inf_ξ (E_Q[f(q(ξ))] − s) > 0).
Borell–TIS (the Borell–Tsirelson–Ibragimov–Sudakov inequality): for a
centered continuous Gaussian process X on T with sup-standard-deviation
σ̄ and Dudley entropy expectation D,
    Q(sup_T X ≥ m) ≤ exp( −(m − D)^2 / (2 σ̄^2) ),   m ≥ D.
We take D = 2√2·J with J the Dudley integral (van Handel, APC 550,
Thm 5.12); with the canonical metric d(t,t') = sd(X_t − X_t') ≤ L·|t−t'|
the covering bound N(ε) ≤ L·T/(2ε) gives the closed form
    J = ∫_0^{LT/2} √(ln(LT/(2ε))) dε = (LT/2)·∫_0^1 √(ln(1/u)) du
      = L·T·√π/4.
All three constants (2√2, √π/4, the covering factor 1/2) are explicit.

(3c) The conditional-law factors, all certified:
  * CONDITIONING SHRINKS VARIANCE.  For any linear functional L and the
    6-pin vector P,  Var_Q(L) = Var(L) − Cov(L,P)^T A^{−1} Cov(L,P)
    ≤ Var(L),  A the pin covariance (inverted at dps = 100, residual
    ≤ 6.2e-96 per rung).
  * VARIATIONAL PRINCIPLE.  For ANY explicit predictor a:  Var_Q(L) ≤
    Var_uncond(L − a^T P).  We use the cubic Hermite predictor H of the
    pins (H(ξ) = s + ℓ(3ξ^2 − 2ξ^3) along the corridor, the smoothstep):
    every variance and metric constant is computed as an EXACT
    unconditional covariance of the explicit functional (f − H) and its
    ξ-derivatives.  Tightness: ŝ ≈ 2.1× the true conditional sd.
  * PIN-LIKELIHOOD MEAN BOUNDS (the Λ_p trick).  For any linear
    functional g,  |E_Q[g]| ≤ √(Var_uncond g)·√(p^T A^{−1} p) = √a_{2n}·Λ_p
    with p the pin values; Λ_p^2 certified = 2.82666 (r = 0.05; 2.82667 on
    the finer rungs) — the pin configuration is typical, so ALL conditional
    means admit global O(1) bounds.  Used for: the curvature-profile
    4th-derivative bound |d^4 E[f_xx]/dζ^4| ≤ r^4·Λ_p·√a_8, the value-
    profile 2nd-derivative bound |u''| ≤ r^2·Λ_p·√a_4, and the corridor
    Lipschitz corrections.
  * COVARIANCE FACTORIZATION.  Cov(∂^α f(x), ∂^β f(y)) =
    (−1)^|β| K1^{(α1+β1)}(dx)·K1^{(α2+β2)}(dy); the corridor lies on the
    x-axis, so every covariance entering the certificate is a 1-D K1 value
    of order n ≤ 8, inside cov_exact's certified spectral-tail regime.
  * SPECTRAL MOMENTS.  a_{2m} = E[N(0,1)^{2m}] = (2m−1)!!: a_2 = 1,
    a_4 = 3, a_6 = 15, a_8 = 105, with certified spectral tails for the
    absolute moments through order 22 (used only in the crude remainder
    bounds of (3d)).
  * CERTIFIED GRID-TO-CONTINUUM TRANSFER.  Margins: pointwise-exact
    (value, first derivative) on the grid + a global Λ_p second-derivative
    bound at O(g^2).  Variance suprema: the FULL 8-level pointwise-exact
    Taylor chain of Var(F(ξ)) — d^n/dξ^n Var(F) = Σ_{i+j=n} c_{ij}
    Cov(F_i, F_j) with Leibniz coefficients (2, 2, 6, 8, 6, 10, 20, 12,
    30, 20, 14, 42, 70, 16, 56, 112, 70 …) — every covariance within the
    order-8 cap, the isolated over-cap pairs bounded by the crude
    spectral-product bound, and the 9th-derivative remainder bounded
    globally by the coefficient-sum 2^9 = 512 times the crude products at
    the O(g^9/9!) level (negligible: ≤ 1e-23).

(3d) Certified numbers per rung (driver transcript, both modes
byte-identical):

  r = 0.05 (Λ_p^2 = 2.82666):
    Z1: h_min = 0.024593 (0.4919·r), σ̄1 = 0.002201 (0.8803·r^2),
        L1 = 0.0129, D1 = 0.00406, kap1 = 9.3312 ⇒ Q(E1) ≤ exp(−43.536)
    Z2: u_* = 2.9342e-6 (0.1408·ℓ), σ̄2 = 1.668e-7 (0.008007·ℓ),
        L2 = 5.14e-7, D2 = 4.83e-7, kap2 = 14.692 ⇒ Q(E2) ≤ exp(−107.93)
    P_r(B4) ≤ 3.47·√(e^{−43.536} + e^{−107.93}) = 1.22e-9
  r = 0.025: kap1 = 20.439 (exp −208.88), kap2 = 30.762 (exp −473.16),
    P ≤ 1.52e-45.  r = 0.0125: kap1 = 42.653 (exp −909.65),
    kap2 = 57.793 (exp −1670.0), P ≤ 1.03e-197.

## 4. GRADE: BETTER THAN O(r^3) — THE BOUND IS O(exp(−c/r^2))

The o(r^3) threshold form (H4 §3.5) requires margin/sd ≥ √(12 ln(1/r) +
4 ln C_RN + 2 ln 2) = 6.50 at r = 0.05.  The certified margins are
kap1 = 9.33 and kap2 = 14.69 at r = 0.05, and BOTH GROW like 1/r (§6):
the certificate is not merely o(r^3) but
    P_r(B4) ≤ exp( −c_eff / r^2 ),  c_eff = −r^2 ln P_bound
            = 0.0513, 0.0645, 0.0709  on the ladder (increasing),
an exponentially small bound — the strongest grade the RN square-root
transfer can express.  The registered B4.loc dam line is closed at
validity grade, covering B4.rem as well (B4.rem's PERC consumption is
discharged for the O(r^3) validity grade: the inclusion of §2 does not
distinguish loc from rem).

## 5. THE H4-SD DISTINCTION (verified, not assumed)

H4-SD's C^0-impossibility applies to the B1.dir slice: there the away
branch rides M's lineage at EVERY window level, so any separating cut must
leave the rigid ridge, where sd_res = Θ(d^2) dwarfs the window ℓ — the
certified margins were 0.55/0.054/−0.022 against threshold 6.5: vacuous.
The B4 dam line is the OPPOSITE geometry: a C^0 VALUE/CURVATURE event on a
FIXED corridor at the PIN level s, and the cheapest cut lives ON the rigid
ridge (kappa_* ≈ 48 = Θ(1/r) at r = 0.05 — C2 §4's registered law).  The
C^1-necessity finding of H4-SD constrains the B1.dir event, a different
event; it is not a constraint on this certificate.  Verified independently
in the driver: the corridor profile E_Q[f(q(ξ))] − s is strictly increasing
from 0 to b − s (the high channel is a narrow axial ridge, C2 P2), the
margins above are ON-ridge margins.

## 6. SCALE-INVARIANCE DISPLAY (extension of the kappa_dam display to the
##    B4.loc tube)

Certified on the ladder {0.05, 0.025, 0.0125}:
    kap1 (curvature zone): 9.3312, 20.439, 42.653;  halving ratios
        0.4565, 0.4792;  kap1·r = 0.4666, 0.511, 0.5332
    kap2 (value corridor): 14.692, 30.762, 57.793;  halving ratios
        0.4776, 0.5323;  kap2·r = 0.7346, 0.7691, 0.7224
i.e. BOTH zone margins are Θ(1/r) (the C2 §4 law kappa_* = Θ(1/r)),
contrasting with the 9-pin tube's scale-invariant kappa_dam = Θ(1)
(identification axis (iv), §1).  The κ·r constants are r-independent up to
the slowly-varying entropy composition (windowed and fail-closed in the
driver: kap1·r ∈ (0.4, 0.65), kap2·r ∈ (0.6, 1.6), halving ratios within
0.075 of 1/2, c_eff ∈ (0.04, 1.2)).

## 7. THE DISCRETE MIRROR (B1's frozen PL model)

The inclusion "clear corridor ⇒ ¬B4" is mirrored exactly in B1's frozen
piecewise-linear lattice ensemble (b1_falsifier.py 818df798…): for each of
the 40 frozen ensemble fields (427 close pairs, cheb ≤ 4), each registered
synthetic pair, and 2 carved close-pair B4 witnesses (the ensemble contains
no close B4 pair), the deterministic hex-greedy M-ward path from S is
audited: 3 B4 pairs audited (0 ensemble + 3 synthetic), EVERY B4 corridor
CUT (0 violations); 387 pairs with a clear corridor (categories
{A: 154, B1: 1, B2: 66, success: 163}), none of them B4; 0 pairs with no
greedy hex path.  The mirror fail-closes on any violation.

## 8. FALSIFIER AND MUTATION SUITE

b4loc_falsifier.py (both modes byte-identical): F1 re-runs the driver
fresh and pins its digest to the frozen expectation; F2 drives the
mutation suite at r = 0.05, each mutation must exit 1 with its registered
firing string:
    M1 weaken-margin (margins × 1/2): fires the o(r^3) gate
       (P_bound = 0.104 > r^3 = 1.25e-4, ratio 835);
    M2 widen-tube (σ̄, D × 2): fires the o(r^3) gate (same assembly);
    M3 rn (C_RN × 2): fires the RN-drift gate ("consumed RN factor
       drifted above H4-RN's certified C_RN = 3.47").
F3 verifies the mutation tags (no silent normal runs).  A weakened margin
or widened tube therefore fails CLOSED, as required.

====================================================================
END_FROZEN_BODY
====================================================================

Freeze record (outside the frozen body): see FREEZE.txt — body hash,
artifact hashes, transcripts, digests.
