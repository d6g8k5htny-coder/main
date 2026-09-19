# FAILED_APPROACHES — K3 swarm record (append-only; every dead end with its cause)

## Inherited (pre-K3, confirmed and quarantined 2026-08-05)

1. **Dropped-square-root Cauchy–Schwarz (the blocking defect).** Factor `p_grad·√(E[det²H])·min(P_type, P_window)`
   used as an upper bound; valid form requires `√(min(...))`. Cause: the bound was assembled by analogy to
   LB-1's rho_CS/rho_sad (which carry the root) without re-deriving the chain for the window factor.
   Detected by the K3 work order; confirmed numerically (175.3× at the witness point). Repair: not the
   mechanical root-insertion (too weak, ~1e-2 zone class) — the per-slice conditioning chains (W2/LEAD).
2. **Finite-rung tables + fitted exponents toward an all-small-r statement.** The 12-rung WP table and the
   5.5e-3·r^1.6 envelope fitted to the (invalid) surrogate were presented as an r-uniform envelope.
   Cause: extrapolation without a scaled asymptotic law. Repair path: W6's limiting-jet derivation
   (v ∝ r^{2k+2} class) with explicit remainder, or interval coverage.
3. **O(r^1.6) correction mislabeled O(r³).** v1.1 draft prose claimed "every correction is O(r³)" while the
   WP truth scales ~r^1.6. W10's honest remainder: (1 − O(r))·(1 − O(r^{1.45})).
4. **T5 certificate defects.** KeyError tbl['0.005'] (absent from RUNGS) crashing before PASS; no complete
   -O companion; runner labels embedded in transcript files (G4 now requires labels/receipts separate).
5. **float64 inversion of the r-clustered 9-pin Gram** (det ~1.3e-42, λ_min ~1e-14): silent garbage in the
   float cross-check grid (exp overflow). Repair: mpmath at 100 dps with certified residuals; W11 adds:
   mpmath iv is NOT safe either — approximate-inverse + interval residual certification (Neumaier 1990).
6. **Complex-arithmetic spectral covariance** ((1j·KX)**order): NaN/complex propagation crashed the first
   WP build. Repair: real parity decomposition (cos/sin sums) — the pattern that passed both modes.
7. **Double-hex-line encoding defect (3 occurrences).** Hash-substitution pattern kept the stale hash line
   after the marker. Caught by in-loop assertions; defect notes filed; K3 standard: exactly one hex line,
   verified at build.

## In-campaign (K3)

8. **Interval-box evaluation of the Λ-grid functional (DER-027b).** Structural cancellation produced junk
   ~7500× box width. Abandoned for exact J2 jets + exact midpoint assemblies + one named premise (H-B3).
9. **Global sup-bound arithmetic for the Λ-grid.** Magnitude compounding through frame/Neumann stages.
   Abandoned for the exact-jet design.
10. **Zone-wide value-kill as a uniform bound (C030 consequence (i)).** A pointwise interior kill
    (3.94e-15-class at (−0.45, −0.125)) was extended zone-wide; the rim band (area 2.0175, 92.8% of the
    upper-bound integral) and the ridge arcs (m > b, certified at P*/Q*) refute uniformity. W5 forensic:
    the printed rigidity figure was a single-station value (d = 1, one angle 2.35 rad) multiplied by the
    full disk area — and the G-F2a d1 scaling gate had FAILED-AS-WRITTEN and was waived.
11. **Direct tensor Gauss–Hermite on the tail-dominated 4D expectation** (lead, pre-K3): unstable
    (1.1e-21 → 4.1e-21 across meshes) because the integrand's mass sits in the far tail. Repair: exact
    1D slice reduction (u-slice) with the rare part integrated analytically — the form both W2 and the
    lead independently converged on.
12. **Cantelli as the in-slice saddle-probability bound near the window:** too loose where Edet(u) is
    several SDs out (in-slice CS with Chernoff/HW tails required instead — W2-14/15).

(Appended as further workstreams report failures.)
13. **Global-CS modulus near the cluster (W6 self-caught 2026-08-05).** The corrected-CS bound has a
    genuine singularity rho_CS ~ ell^{1/2} d^{-4} on the arch direction near the pin cluster (the phi(z_F)
    kill is O(1) while pgrad ~ d^{-4}), so a single-power modulus 4.2*r^{3/2} fails for r <~ 0.003. The
    truth is dead there (rho <= 1e-171); the failure is a bound artifact from the sqrt-loss maximized at
    small P_W. Repair: zone split (UB_slice near-cluster, global CS outside) or the measured converged
    CS law. Lesson: a modulus must be validated at CONVERGED quadrature at the smallest r, not only at
    assembly rungs.
14. **Missing-sqrt-restoration mutation (accepted gap, W13 F6).** No landed certificate implements
    the 'restore the missing WP square root' mutation, by design: the defective factor is quarantined
    upstream (D1) and the WP-min closure uses the exact-integrand chain, which contains no such factor
    to restore. The mutation classes that DO exist (input corruption, sign flip, pin/normalization,
    interval/gate omission) are verified fail-closed by W13. Recorded as an accepted scope gap, not a
    certificate defect.
