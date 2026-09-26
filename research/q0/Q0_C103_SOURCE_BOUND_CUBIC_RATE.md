# Q0-C103 — Source-bound typed pair-Palm cubic-rate successor

**Date:** 2026-09-25
**Disposition:** AUTHOR-SIDE SOURCE-BOUND SUCCESSOR / NONCONTROLLING / REVIEW REQUIRED
**Scientific effect:** NONE. This file does not alter Q0-C101, Q0_MASTER, the claim graph, or any frozen status.

## 1. Object identity

Let the exact normalized periodized Bargmann–Fock field live on the two-torus of side 24. For 0 < r <= 0.025 put

    M = x - (r/2)t,    S = x + (r/2)t,

and impose the six pins

    f(M)=b,  f(S)=b-r^3/6,  grad f(M)=grad f(S)=0.

Condition on M being a maximum and S an index-one saddle with the determinant-pair weight. Let P_MS(r,b) denote this typed maximum–saddle pair-Palm law and define

    q_MS(r,b) := P_MS(r,b){ D(M)=S },

where D(M) is the elder-rule merging saddle of the component born at M.

This object is deliberately named q_MS. Historical core-pairing sources also contain an adjacency-conditioned quantity called q. This successor does NOT assert equality between that older adjacency object and q_MS. If a later controlling source aliases q to this exact typed pair-Palm object, that alias must be stated and reviewed separately.

## 2. Conditional theorem

Assume:

1. the field is Morse with distinct critical values;
2. there is no index-one saddle–saddle heteroclinic connection, so the deterministic Morse–Smale reduction in Section 3 applies;
3. the exact source-bound Kac–Rice, collar, near, fixed-annulus and exterior estimates listed in Section 4 hold at their stated domains.

Then there is a finite constant C, uniform for 0 < r <= 0.025, such that

    0 <= 1 - q_MS(r,6/5) <= C r^3.

Consequently q_MS(r,6/5) -> 1 as r -> 0.

No numerical value of C is claimed.

## 3. Deterministic reduction

The exact source C102_DETERMINISTIC_REDUCTION proves, under the stated genericity assumptions,

    {D(M) != S} subseteq Pi union Gamma.

Pi is interception by an additional relevant critical point in the value window/collars, and Gamma is the event that the second ascending branch of S reaches a maximum with gain in (0,r^3/6). The loop/crater alternative is absorbed by the same interceptor event. The reduction is topological and does not use probabilistic independence.

Hence

    1 - q_MS(r,6/5) <= P_MS(Pi) + P_MS(Gamma).

## 4. Source-bound probabilistic estimates

Every load-bearing source below is named by exact repository path and Git blob in Q0_C103_SOURCE_MAP.json.

### C1 — deterministic defect inclusion

Source: C102_DETERMINISTIC_REDUCTION.md, blob 25917befb388211e3848b00c5ba90670e9b716b0.

### C2/C3/C4 — global interceptor count

Source: C101_GLOBAL_INTERCEPTOR_CLOSURE.md, blob 1daec2574e2505ab8162c9a36d8beead34360dd0.

It uses the typed pair-Palm Kac–Rice intensity and partitions the torus into collars, the singular pair-scaled region, a fixed annulus and the exterior.

The singular ledger is read with every factor retained. At witness radius t:

- value/gradient density: O(t^-6);
- three physical Hessian determinants together: O(r^6);
- pair-Palm normalizer denominator: Z_r >= c r^2;
- value-window width: O(r^3);
- planar polar area: t dt.

Therefore the singular contribution is bounded by

    C r^-2 r^6 r^3 integral_{c r}^{t0} t^-6 t dt
      = C r^7 integral_{c r}^{t0} t^-5 dt
      = O(r^3).

This is the corrected exponent ledger. The shorter standalone C101 prose that omitted the area/normalizer bookkeeping is not used as the proof of this line.

### C5 — direct-gain Gamma / maximum window

The inclusion Gamma subseteq {additional maximum in (b-r^3/6,b)} is combined with:

- exterior transfer: C098_GAMMA_EXTERIOR_TRANSFER.md, blob 3e3037b6dbe57d98b7b5932fb1db62b8e8c29ea6;
- pair-scaled near maximum no-go and residual integral: C099_NEAR_CUBIC_CLOSURE.md, blob dddf55a3356823b34f333da28ba1e6551f1db5bc;
- collar maximum count: C100_COLLAR_CUBIC_CLOSURE.md, blob 04d6e1c5a012d1a0e87abe8c9c2e3f7e921478a8.

The older conditional Malliavin route C097_GAMMA_DENSITY_REDUCTION is not consumed by this successor.

### C6 — collar residues

Source: C100_COLLAR_CUBIC_CLOSURE.md, blob 04d6e1c5a012d1a0e87abe8c9c2e3f7e921478a8.

For y=r xi it gives gradient density r^-3, three-determinant factor r^6 and pair-Palm normalizer r^2, hence physical intensity r F_r(xi). Near a conditioned endpoint the rho^-2 density singularity is cancelled by the rho^2 determinant zero; the axis is exponentially suppressed. Physical area r^2 dxi then yields O(r^3).

### C7 — chart coverage and finite-jet nondegeneracy

- C102_CHART_ATLAS.md, blob 38a849f6725d76e310faebf17e21a7faadf87454;
- C102_FINITE_JET_NONDEGENERACY.md, blob 1bea9e62ef1ec621f51953dd3f5d5410fba616af.

These sources cover the pair-corrected, max-near generic/axis/transverse, collar generic/pin/axis/transverse, interceptor, fixed-annulus, exterior and type-boundary charts used by the T1 estimates.

## 5. Genericity / SARD-G gate

The full-Gaussian corollary additionally requires the almost-sure genericity input, including exclusion of saddle–saddle heteroclinic connections.

Current source: the 19,242-byte Gaussian transversality working source, Git blob a0f4aa6ab6d0e21e7a24b4a6e1aa4fdebc40efdc.

Its conservative theorem remains conditional on Sections 8–11 charting/measurability/exceptional-parameter interfaces.

Specialist review KIMI-AUD-022, Git blob e968385a3bf4e8318a9ab6cb4f57818828f01b18, verifies the architecture and many individual identities, assesses several remaining steps as standard, but explicitly leaves execution-level Clarification A (measurable-selection/chart bookkeeping) and Clarification C (endpoint coefficient formulas or an exact citation package).

Therefore this successor separates two levels:

- Conditional generic-field theorem: Sections 2–4 above, assuming the genericity hypotheses.
- Full-Gaussian corollary: HOLD-WITH-DOMAIN until the SARD-G execution-level Clarifications A/C are supplied by exact source objects and reviewed.

No review label alone is treated as a substitute for those missing execution details.

## 6. Uniformity in r

The domain 0 < r <= 0.025 is inherited only from the exact T1 companion sources. Each regional constant must be uniform on its declared compact chart and the finite chart cover. This successor does not manufacture a larger radius or extrapolate beyond the stated range.

## 7. Squeeze

Under the conditional hypotheses and source-bound regional estimates,

    P_MS(Pi) <= C_Pi r^3,    P_MS(Gamma) <= C_Gamma r^3,

so

    1 - q_MS(r,6/5) <= (C_Pi + C_Gamma) r^3.

Since r^3 -> 0, the selection limit follows.

## 8. What this successor repairs

- It fixes object identity by using q_MS instead of an overloaded historical q.
- It binds every load-bearing regional estimate to exact source objects.
- It prints the complete singular power ledger, including normalizer, height and area factors.
- It separates the generic-field theorem from the not-yet-fully-executed SARD-G full-Gaussian corollary.
- It does not claim a numerical upper coefficient, a lower bound, a 3D theorem, Theorem B, or RN/JETMOD closure.

Independent review should return C1–C8 dispositions against this exact successor and separately classify the SARD-G full-Gaussian gate.
