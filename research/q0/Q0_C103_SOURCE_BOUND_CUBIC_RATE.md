# Q0-C103 — Source-bound typed pair-Palm cubic-rate successor

**Date:** 2026-09-25
**Disposition:** AUTHOR-SIDE SOURCE-BOUND SUCCESSOR / NONCONTROLLING / REVIEW REQUIRED
**Scientific effect:** NONE. This file does not alter Q0-C101, Q0_MASTER, the claim graph, or any frozen status.
**C1/C2 repair tip:** 2026-09-26 author-side honesty repair (branch-attached object + named Z-lower premise). Does not touch C3–C8 regional arithmetic beyond restating their dependence on those gates. No status promotion.
**C1 object AMEND tip:** 2026-09-26 — names conditioned object `q_MS^br`, joint-event identity for `1 − q_MS`, and unresolved envelope `B_miss ∩ {D(M)≠S}` (review 5850461900).

## 1. Object identity

Let the exact normalized periodized Bargmann–Fock field live on the two-torus of side 24. For 0 < r <= 0.025 put

    M = x - (r/2)t,    S = x + (r/2)t,

and impose the six pins

    f(M)=b,  f(S)=b-r^3/6,  grad f(M)=grad f(S)=0.

Condition on M being a maximum and S an index-one saddle with the determinant-pair weight. Let P_MS(r,b) denote this typed maximum–saddle pair-Palm law and define the **unconditioned** selection probability

    q_MS(r,b) := P_MS(r,b){ D(M)=S },

where D(M) is the elder-rule merging saddle of the component born at M.

This object is deliberately named q_MS. Historical core-pairing sources also contain an adjacency-conditioned quantity called q. This successor does NOT assert equality between that older adjacency object and q_MS. If a later controlling source aliases q to this exact typed pair-Palm object, that alias must be stated and reviewed separately.

### 1.1 Branch event and conditioned object (C1)

C102_DETERMINISTIC_REDUCTION (blob `25917befb388211e3848b00c5ba90670e9b716b0`) proves

    {D(M) != S} subseteq Pi union Gamma

only after assuming **H_branch**: one ascending branch of S already terminates at M (together with Morse, distinct critical values, and Morse–Smale). Those genericity hypotheses alone do **not** force the attachment.

Define the complementary branch-miss event

    B_miss := { neither ascending branch of S terminates at M }.

Note: B_miss does **not** imply D(M) ≠ S. The component of M just above f(S) can already contain an arm of S via a maximum N ≠ M merged into it, with the other arm in an older component. The unresolved defect class that escapes Π ∪ Γ is therefore the joint event

    B_miss ∩ {D(M) ≠ S},

not B_miss alone.

Partition identity (always):

    1 - q_MS
      = P_MS( {D(M)≠S} ∩ H_branch ) + P_MS( {D(M)≠S} ∩ B_miss ).

C102 licenses only the joint-event inequality

    P_MS( {D(M)≠S} ∩ H_branch ) <= P_MS(Pi) + P_MS(Gamma).

It does **not** license a restriction of the number `1 − q_MS` “to the H_branch event.”

Define the **branch-conditioned selection object**

    q_MS^br(r,b) := P_MS( D(M)=S | H_branch ).

Then, on H_branch,

    1 - q_MS^br = P_MS( {D(M)≠S} | H_branch )
               = P_MS( {D(M)≠S} ∩ H_branch ) / P_MS(H_branch),

so a cubic bound on the joint event becomes a cubic bound on `1 − q_MS^br` only after dividing by P_MS(H_branch). That mass is isolated as **PREMISE-BRANCH-MASS** below. This tip’s cubic theorem is stated for `q_MS^br`, not for the unconditioned `q_MS`.

## 2. Conditional theorem (branch-conditioned object)

Assume:

1. the field is Morse with distinct critical values;
2. there is no index-one saddle–saddle heteroclinic connection (Morse–Smale), so every ascending branch of an index-one saddle terminates at a local maximum;
3. **H_branch** is the C102 attachment event (one ascending branch of S terminates at M);
4. **PREMISE-BRANCH-MASS**: inf_{0 < r <= 0.025} P_MS(H_branch) > 0 at the compact mark scope used below;
5. **PREMISE-Z-LOWER**: liminf_{r ↓ 0} Z_r / r^2 > 0 at that mark scope (see Section 4 / C2); on any compact interval [δ, 0.025] ⊂ (0, 0.025], positivity of Z_r follows from continuity once the jet Gram is positive definite (C102_FINITE_JET_NONDEGENERACY), so the load-bearing gap is the liminf at zero;
6. the exact source-bound Kac–Rice, collar, near, fixed-annulus and exterior estimates listed in Section 4 hold at their stated domains, reading every division by Z_r under PREMISE-Z-LOWER.

Then there is a finite constant C, uniform for 0 < r <= 0.025, such that

    0 <= 1 - q_MS^br(r,6/5) <= C r^3.

Consequently, under those same hypotheses, q_MS^br(r,6/5) -> 1 as r -> 0.

No numerical value of C is claimed. This tip does **not** assert a cubic envelope for the unconditioned `1 − q_MS`, nor for P_MS(B_miss ∩ {D(M)≠S}).

## 3. Deterministic reduction (joint event under H_branch)

The exact source C102_DETERMINISTIC_REDUCTION proves, under Morse + distinct critical values + Morse–Smale + **H_branch**,

    {D(M) != S} subseteq Pi union Gamma

on the H_branch locus. Pi is interception by an additional relevant critical point in the value window/collars, and Gamma is the event that the second ascending branch of S reaches a maximum with gain in (0,r^3/6). The loop/crater alternative is absorbed by the same interceptor event. The reduction is topological and does not use probabilistic independence.

Hence the joint-event inequality

    P_MS( {D(M)≠S} ∩ H_branch ) <= P_MS(Pi) + P_MS(Gamma).

Combined with PREMISE-BRANCH-MASS,

    1 - q_MS^br <= ( P_MS(Pi) + P_MS(Gamma) ) / P_MS(H_branch).

## 4. Source-bound probabilistic estimates

Every load-bearing source below is named by exact repository path and Git blob in Q0_C103_SOURCE_MAP.json.

### C1 — deterministic defect inclusion

Source: C102_DETERMINISTIC_REDUCTION.md, blob 25917befb388211e3848b00c5ba90670e9b716b0.

Consumed only as a joint-event inclusion under H_branch (Sections 1.1 and 3). The false transfer that cited Π ∪ Γ under Morse + Morse–Smale alone is withdrawn. The false reading that “restricts `1 − q_MS` to H_branch” is also withdrawn; the theorem object is `q_MS^br`.

### C2 — typed pair-Palm / Kac–Rice normalization (named unresolved premise)

Sources: C101_GLOBAL_INTERCEPTOR_CLOSURE.md, blob 1daec2574e2505ab8162c9a36d8beead34360dd0; with companion intensity statements in C098/C100 as mapped.

C101 §2 and C100 §2 write the typed intensity: pins J_6, weight W_MS = |det H_M det H_S| 1_{H_M ≺ 0} 1_{det H_S < 0}, normalizer Z_r = E[W_MS | J_6], with no second typing division. C098’s Hessian count gives det H_M, det H_S = O(r), hence W_MS = O(r^2), which bounds the numerator. C098’s three-row table (r = 0.05, 0.025, 0.0125; Z_r/r^2 ≈ 3.2337, 3.2245, 3.2232) is diagnostic; positivity of a limiting z_0 is marked as input. C099/C100 likewise assert inf z_r > 0 without proof.

**PREMISE-Z-LOWER (unresolved at zero):** 

    liminf_{r ↓ 0} Z_r / r^2 > 0

at the exact compact-mark scope of the successor (mark b = 6/5 under the six-pin typed law). On any compact [δ, 0.025] ⊂ (0, 0.025], C102_FINITE_JET_NONDEGENERACY (blob `1bea9e62ef1ec621f51953dd3f5d5410fba616af`) supplies a positive-definite distinct-point jet Gram, hence Z_r is continuous and positive there; that compact piece is not the open gap. Every regional estimate that divides by Z_r near r ↓ 0 is conditional on this liminf premise. A later tip may discharge it by an exact proof object or keep it named.

### C3/C4 — global interceptor count / singular ledger

Source: C101_GLOBAL_INTERCEPTOR_CLOSURE.md, blob 1daec2574e2505ab8162c9a36d8beead34360dd0.

It uses the typed pair-Palm Kac–Rice intensity and partitions the torus into collars, the singular pair-scaled region, a fixed annulus and the exterior.

Collars of radius 2r about (±r/2, 0) exclude the singular witness radius below

    t_* = (√15 / 2) r

(equivalently η^2 ≤ 4/15 at the collar boundary; C101 §4). Reading the singular ledger with every factor retained, and **assuming PREMISE-Z-LOWER**, at witness radius t:

- value/gradient density: O(t^-6);
- three physical Hessian determinants together: O(r^6);
- pair-Palm normalizer denominator: under PREMISE-Z-LOWER, Z_r / r^2 stays bounded away from 0 along the liminf;
- value-window width: O(r^3);
- planar polar area: t dt.

Therefore the singular contribution is bounded by

    C r^-2 r^6 r^3 integral_{t_*}^{t0} t^-6 t dt
      = C r^7 integral_{(√15/2) r}^{t0} t^-5 dt
      = C r^7 [ -1/(4 t^4) ]_{(√15/2) r}^{t0}
      = O(r^3),

with lower-limit constant 1/(4 t_*^4 / r^4) = 4/225. The integration lower limit is written `(√15/2) r`, not a bare `c r`, so it does not collide with the constant `c` in informal Z_r ≥ c r^2 shorthand.

This is the corrected exponent ledger. The shorter standalone C101 prose that omitted the area/normalizer bookkeeping is not used as the proof of this line. Without PREMISE-Z-LOWER the division step near r ↓ 0 is not discharged.

### C5 — direct-gain Gamma / maximum window

The inclusion Gamma subseteq {additional maximum in (b-r^3/6,b)} is combined with:

- exterior transfer: C098_GAMMA_EXTERIOR_TRANSFER.md, blob 3e3037b6dbe57d98b7b5932fb1db62b8e8c29ea6;
- pair-scaled near maximum no-go and residual integral: C099_NEAR_CUBIC_CLOSURE.md, blob dddf55a3356823b34f333da28ba1e6551f1db5bc;
- collar maximum count: C100_COLLAR_CUBIC_CLOSURE.md, blob 04d6e1c5a012d1a0e87abe8c9c2e3f7e921478a8.

The older conditional Malliavin route C097_GAMMA_DENSITY_REDUCTION is not consumed by this successor. (C5 content otherwise unchanged; still review-pending.)

### C6 — collar residues

Source: C100_COLLAR_CUBIC_CLOSURE.md, blob 04d6e1c5a012d1a0e87abe8c9c2e3f7e921478a8.

For y=r xi it gives gradient density r^-3, three-determinant factor r^6 and pair-Palm normalizer r^2, hence physical intensity r F_r(xi). Near a conditioned endpoint the rho^-2 density singularity is cancelled by the rho^2 determinant zero; the axis is exponentially suppressed. Physical area r^2 dxi then yields O(r^3). The normalizer step near r ↓ 0 is read under PREMISE-Z-LOWER. (C6 content otherwise unchanged.)

### C7 — chart coverage and finite-jet nondegeneracy

- C102_CHART_ATLAS.md, blob 38a849f6725d76e310faebf17e21a7faadf87454;
- C102_FINITE_JET_NONDEGENERACY.md, blob 1bea9e62ef1ec621f51953dd3f5d5410fba616af.

These sources cover the pair-corrected, max-near generic/axis/transverse, collar generic/pin/axis/transverse, interceptor, fixed-annulus, exterior and type-boundary charts used by the T1 estimates. Finite-jet also underwrites the compact-[δ, 0.025] positivity half of the Z_r story (Section 4 / C2). (C7 content otherwise unchanged.)

## 5. Genericity / SARD-G gate

The full-Gaussian corollary additionally requires the almost-sure genericity input, including exclusion of saddle–saddle heteroclinic connections.

Current source: the 19,242-byte Gaussian transversality working source, Git blob a0f4aa6ab6d0e21e7a24b4a6e1aa4fdebc40efdc.

Its conservative theorem remains conditional on Sections 8–11 charting/measurability/exceptional-parameter interfaces.

Specialist review KIMI-AUD-022, Git blob e968385a3bf4e8318a9ab6cb4f57818828f01b18, verifies the architecture and many individual identities, assesses several remaining steps as standard, but explicitly leaves execution-level Clarification A (measurable-selection/chart bookkeeping) and Clarification C (endpoint coefficient formulas or an exact citation package).

Therefore this successor separates two levels:

- Conditional generic-field theorem: Sections 2–4 above, for `q_MS^br`, assuming the genericity hypotheses, **H_branch**, **PREMISE-BRANCH-MASS**, and **PREMISE-Z-LOWER**.
- Full-Gaussian corollary: HOLD-WITH-DOMAIN until the SARD-G execution-level Clarifications A/C are supplied by exact source objects and reviewed.

No review label alone is treated as a substitute for those missing execution details.

## 6. Uniformity in r

The domain 0 < r <= 0.025 is inherited only from the exact T1 companion sources. Each regional constant must be uniform on its declared compact chart and the finite chart cover. This successor does not manufacture a larger radius or extrapolate beyond the stated range.

## 7. Squeeze

Under H_branch, PREMISE-BRANCH-MASS, PREMISE-Z-LOWER, the conditional hypotheses, and source-bound regional estimates,

    P_MS(Pi) <= C_Pi r^3,    P_MS(Gamma) <= C_Gamma r^3,

so

    P_MS( {D(M)≠S} ∩ H_branch ) <= (C_Pi + C_Gamma) r^3

and therefore

    1 - q_MS^br(r,6/5) <= (C_Pi + C_Gamma) r^3 / P_MS(H_branch).

Since r^3 -> 0 and PREMISE-BRANCH-MASS keeps the denominator bounded away from 0, the selection limit for `q_MS^br` follows. The squeeze does **not** bound P_MS(B_miss ∩ {D(M)≠S}) and does not bound the unconditioned `1 − q_MS`.

## 8. What this successor repairs

- It fixes object identity by using q_MS instead of an overloaded historical q, and by naming the theorem object `q_MS^br`.
- It binds every load-bearing regional estimate to exact source objects.
- It prints the complete singular power ledger, including normalizer, height and area factors, with lower limit `(√15/2) r`.
- It separates the generic-field theorem from the not-yet-fully-executed SARD-G full-Gaussian corollary.
- **C1:** withdraws the false transfer of C102’s inclusion without H_branch; withdraws the false “`1 − q_MS` restricted to H_branch” reading; states the cubic theorem for `q_MS^br`; names PREMISE-BRANCH-MASS; names unresolved envelope `B_miss ∩ {D(M)≠S}`.
- **C2:** isolates liminf_{r↓0} Z_r/r^2 > 0 as PREMISE-Z-LOWER; records that compact-[δ,0.025] positivity is already underwritten by finite-jet continuity.
- It does not claim a numerical upper coefficient, a cubic bound for unconditioned `1 − q_MS`, a 3D theorem, Theorem B, or RN/JETMOD closure.
- C3–C8 regional contents are not redesigned in this tip; they inherit the named gates above.

Independent review should return C1–C8 dispositions against this exact successor and separately classify the SARD-G full-Gaussian gate.
