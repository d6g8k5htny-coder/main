# C091 CORRECTION LEDGER

**Date:** 2026-07-16  
**Discipline:** append-only. A correction changes the live dependency graph; it does not erase the historical artifact.

## E-C091-1 — Upper interpolation gate over-strengthened at C090

**Prior form:** require a bound on \(\sup|U''|\).  
**Finding:** the linear-interpolation remainder has \((r-a)(r-b)\le0\), so only negative curvature can lift \(U\) above its endpoint chord.  
**Live successor:** `UPPER_ONE_SIDED_INTERPOLATION`, requiring \(U''\ge-M\).  
**Effect on theorem:** reduces numerical burden; no theorem is weakened.

## E-C091-2 — H4 one-step Markov product killed

**Prior form:** use \(q_{step}^{n}\) for an \(n\)-checkpoint corridor.  
**Finding:** exact six-pin multivariate Gaussian diagnostics give corridor probabilities larger than \(q_{step}^{n}\) by factors up to about 3.35 in the tested skeletons.  
**Classification:** structural probability error, not a rounding issue.  
**Live successor:** `H4_PALM_ORTHANT`, using finite-dimensional Gaussian Chernoff bounds, shifted Palm determinant moments, and explicit path entropy.  
**Effect on theorem:** the C089 far coefficient \(0.0199\) returns to OPEN until re-certified.

## E-C091-3 — Bonferroni near-diagonal mark mismatch

**Prior form:** cite spatial critical-point repulsion to assert a two-point intensity per unit height squared bounded by \(C_{nd}d\), then factor \(\ell^2\).  
**Finding:** shrinking-window second-moment theory shows that nearby critical values can cost only one explicit height-window factor; same-type saddle repulsion is unmarked and does not by itself restore \(\ell^2\).  
**Classification:** incomplete source-to-estimand transfer. The old claim is quarantined, not declared impossible.  
**Live successor:** `BONFERRONI_MARKED_TWO_SCALE`, combining

\[
K_{ss}(d)\le C_{rep}d^3\log(e/d)
\]

with a bounded density for

\[
A=(u_1+u_2)/2,
\qquad
Z=(u_2-u_1)/d^3.
\]

**Effect on theorem:** lower decimal tiers are conditional on the new uniform constants. The one-point \(\Lambda\) machinery remains live.

## E-C091-4 — Exact torus transfer discharged in declared matrix scopes

**Prior status:** exact periodized ensemble defined, downstream matrix transfer open.  
**Finding:** explicit Neumann/Schur perturbation bounds preserve the local and far covariance floors with large margins.  
**Live status:** local and far-station transfer CLOSED for derivative order \(\le4\), pin dimension \(\le9\), target dimension \(\le6\), local radius \(3\), and far distance interval \([5,12]\).  
**Residual:** global integrals must partition their domains and declare any larger matrix/order use.

## E-C091-5 — Upper uncertainty polarity fixed

**Prior form:** a symmetric \(\pm0.02\) assembly band was attached to an upper theorem.  
**Finding:** upper claims consume only the positive side.  
**Live rule:** use \(+0.02\) unless the inputs are proved to be already one-sided upper limits.  
**Residual:** confidence level, multiplicity, sample size, and construction of the band remain unlocated.

## Status delta

```text
CLOSED:
    exact torus local transfer
    exact torus far-station transfer
    one-sided interpolation theorem
    upper uncertainty semantics
    Gaussian corridor Chernoff theorem
    Palm-weighted Chernoff theorem
    planar d^3 critical-value-gap law

KILLED:
    H4 q_step^n product rule

QUARANTINED:
    pointwise C_nd d claim as a uniform per-height^2 Bonferroni discharge

OPEN:
    six-pin marked saddle-repulsion constants
    H4 numerical path assembly
    upper shape certificate
    upper uncertainty calibration

BLOCKED FROM PROMOTION:
    finite lower 0.84
    asymptotic lower 0.8501
    continuous upper 0.99/1.01
```
