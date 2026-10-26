# Sanitized explanatory derivative

This is a publication derivative of the historical 25 October 2026 source document. Mathematical statements are preserved; internal workflow identifiers are removed. It is not the exact pinned source consumed by the tested implementation. References below to authoring-stage NOT_RUN, source-only permission, execution holds, deadlines, private receipts or current Root plans describe that historical scope and have no current operational effect. The accompanying README supplies the later bounded development evidence and its limits.

Original source SHA256: `4393d967cabb3ff67852e4b2b3888478030cabad88bb6a37d59e338b481fcb1e`. The derivative has a different byte identity.

---

# Required proof-carrying interfaces for the complete-field falsifier

Source-only design,25October2026. This document accompanies DESIGN_PROOF.md and Root's current DESIGN.md/PLAN.md. Root historical source-author assignment authorizes source implementation only. All program/test/numerical/formal execution is still NO_GO_NOT_RUN. This file specifies checkable evidence; it is not a passing implementation result.

## Immutable realization descriptor

The realized object is the complete TWO field, not only its core polynomial. Freeze the exact source identities of all17 reconnaissance inputs and b79 weighted source, plus the current Root DESIGN/PLAN/source-GO records.

The descriptor must contain d=2,L=1,w=(1/3,1/9),b=4/9,R=1/8,r1=1/32,A=2^-16,u0=3A^2,v0=2A^3; explicit piecewise S,c,chi; complete physical seed/chart/collar/exterior definition; the hand-proved S'/B/D/g/root/height bounds; lambda=b-16A^4; scale2^80; the global physical-fiber argument and complete cyclic axis-critical inventory. Higher derivative bounds are finite by joint smoothness/compactness, not numerically evaluated constants.

The whole physical evaluator must define the collar and exterior and use centered residuals. A grid evaluation may use the exact core fast path only after checking that all nodes are inside that core. It must cite the separately qualified whole-exterior certificate. It must not rename the core polynomial as the complete field.

## Exact control and law objects

ControlCell:
- dyadic Qa<Qb,Z a<Z b within[0,1], fixed epsilon in{-1,+1};
- center Qc=(Qa+Qb)/2,Zc=(Za+Zb)/2;
- dQ=(Qb-Qa)/2,dZ=(Zb-Za)/2;
- u=3A^2Q^2,v=epsilon*2A^3Q^3(1-Z^8);
- four exact masses m_k=1/2*(Qb^5-Qa^5)*(Zb^k-Za^k),k8,12,15,16.

Mass accounting must cover the entire square and both signs with no overlaps except measure-zero boundaries. Q=0,Z=0,Z=1 are null loci; a positive-area cell touching them is not null and remains full-mass unresolved unless independently certified. Preserve every planned cell, including failed and budget-ended cells. Do not normalize over successes.

## Uniform root-exclusion certificate

RootCertificate(C) supplies three disjoint intervals Ii=[alpha_i,beta_i] in[-3,3], width at most h<=1, with the complete order beta1<alpha2<beta2<alpha3. For the cubic f=-x^3+3Q^2x+epsilon*2Q^3(1-Z^8), give exact rational interval or Bernstein certificates proving for every(Q,Z) in C:
- f(alpha1)>0,f(beta1)<0 and f'<0 on I1;
- f(alpha2)<0,f(beta2)>0 and f'>0 on I2;
- f(alpha3)>0,f(beta3)<0 and f'<0 on I3.

Certify all three roots and their branch types, not just candidate locations. Strict interval derivative tests may use I1 left of -Qb,I2 inside(-Qa,Qa),I3 right of Qb; if that sufficient condition fails, refine rather than assuming a branch. Reflection handles either control sign. The cubic degree and these three distinct roots exclude further core roots. The realization descriptor's annular/seed argument excludes every other axis maximum and fixes the one deep old minimum.

Any failed inequality, overlap, invalid rational, boundary degeneracy, or bisection cap yields UNRESOLVED_ROOTS with the original full cell mass. No target code executes merely because this certificate is specified.

## Actual64-node observed grid

GridCertificate contains:
- x=[-8,alpha1,beta1,alpha2,beta2,alpha3,beta3,8];
- y=[-16,-8,-4,-2,0,2,4,16];
- row-major integer V[i*8+j]=floor(2^80*max{Hc(x_i,y_j;Qc,Zc),-16});
- all underlying exact rational nodal residuals, with a certificate of exact integer division;
- explicit axis row/column convention: exact_h0 uses values[x_index*8+y_index], so the y=0 column index is4;
- all floor boundary rows/columns and periodic seam definitions.

Hc is the exact complete field's certified core residual. No critical height formula, kappa, population_certificate.critical_fixture, or authored count supplies these samples.

UFObservation is the unchanged pinned exact_h0.compute(V,8) result. BFSObservation is an independently executed unchanged exact_h0.verify_by_connectivity(V,8,result) plus the extended witness below. Preserve complete returned finite intervals, birth vertices, essential class and zero-count. Do not replace an unexpected result with the theorem's expected pair.

## Independent observed death/merge witness

For every observed positive finite bar (B,D,birth_vertex):
1. Enumerate every regional maximum plateau by the BFS route and retain its vertex set/canonical representative. Bind the actual nonessential birth plateau and actual older plateau by observed integer birth values and the engine's exact tie order.
2. Run threshold BFS from that observed younger plateau at D. Retain an actual path reaching the observed older plateau/vertex.
3. Run threshold BFS at D+1. Retain the entire reachable set and its frontier/cut showing that no older vertex is reached.
4. Project the D path to axis columns. Verify that it crosses the intervening I2 strip, and that it does not cross the floor seam at an above-floor threshold.
5. Verify from actual sampled values that the observed D equals the minimum of the two I2 axis samples, while the observed positive birth and essential plateaus are in the outer-root columns. Retain a concrete axis bottleneck vertex at D.

The root certificate localizes the actual critical saddle after this observed merger is established. It must not manufacture D or a chosen younger maximum. The existing compute() returns no death vertex, so the separate witness is mandatory for physical saddle claims. It is valid for the engine's event vertex to lie elsewhere on an equal-valued threshold plateau; report the certified critical bracket instead of relabeling that event vertex as the continuum saddle.

## Derived whole-cell H0 coupling

Given the actual observed output, calculate exactly
 E=15h^2+45dQ+48dZ+2^-80.
The proof binds these terms to endpoint bracketing, whole-cell parameter variation and integer flooring. No user-supplied epsilon is accepted.

An optional certified tighter mode records X=max absolute critical-bracket endpoint, M2=3Qb^2+3X^2, MQ=3QbX^2+6Qb^2(1-Za^8)X, MZ=16Qb^3Zb^7X, and uses E=(M2/2)h^2+MQ*dQ+MZ*dZ+2^-80. Its rational constants are part of the certificate and every downstream comparison must use that same E. This is the explicit refinement in DESIGN_PROOF, not a caller tolerance.

SelectorCouplingCertificate is RESOLVED only if:
- every realization/root/grid proof check holds;
- actual UF and independent BFS agree;
- actual output has exactly one positive finite interval and one essential class;
- observed finite lifetime (B-D)/2^80>2E;
- observed essential-minus-finite birth gap (Bessential-B)/2^80>2E;
- the plateau and actual merge-witness conditions above hold.
- the actual observed death satisfies D>-16*2^80; the proof gives this from the center critical floor-3/4, h<=1, Taylor loss<=15 and rounding loss<2^-80.

These tests bind actual node observations; they are not assumed as input. Failure of UF/BFS agreement or malformed output is IMPLEMENTATION_FAILURE. Precision/root/gap uncertainty is UNRESOLVED with full cell mass. A contradiction with a valid full certificate requires independent analysis before any mathematical falsification label.

For a resolved cell and every actual control in it:
- true physical finite birth belongs to b+A^4*[B/2^80-E,B/2^80+E];
- true physical death belongs to b+A^4*[D/2^80-E,D/2^80+E];
- true lifetime belongs to A^4*[(B-D)/2^80-2E,(B-D)/2^80+2E];
- the finite bar and essential class exhaust ordinaryH0; no unaccounted short positive bars are hidden near the diagonal.

The proof of exhaustion is the complete-field natural component reduction plus the uniform full axis-critical list. It is not inferred merely from a graph bottleneck estimate. No Hk certificate follows.

## Uniform physical coordinate and mark certificates

Map each uniform critical bracket through rho(x)=arcsin(Ax sqrt(3/2))/pi. Retain its exact interval plus the explicit per-endpoint outward2^-120 arithmetic allowance. Use128 dyadic sqrt bisections,96 alternating terms for each Machin arctan,17 arcsin terms, and the stated exact tail estimates. Each root bracket contains the actual branch for the entire control cell, so the mapped location remains valid even if a particular maximizing grid vertex changes inside that bracket.

Certificate fields:
- actual UF-selected younger outer branch and its mapped physical x interval;
- mapped middle critical x interval and exact physical y2=0;
- observed oriented displacement sign, compared with epsilon only as a prediction;
- shortest-torus-distance interval obtained by interval subtraction, with the proved diameter<1/2;
- optional observed grid birth coordinate with error<=A h+2^-120;
- centered birth-value interval and lifetime interval above.

The current finite-window observables are:
 t/A^4 in{1/4,1/16,1/64},
 count event0<ell<=t,
 joint versions with r/A<=1/16,r/A<=1/8,
 and centered birth(B-b)/A^4<=0.
Freeze the exact requested conjunction/table before production. Any ambiguous boundary comparison remains[0,1] for that cell/event. Exact equality may be classified only with a proof of the non-strict threshold relation.

These are Root's dated DESIGN V2 bins, replacing the preserved uninformative V1 bins1/2,1. DESIGN_PROOF gives exact positive-mass witnesses for all windows: Z255/256 and Q1/16,1/8,1/4 give marks below,between,above the two bins, with ell/A^4<1/64. The strict rational/physical bounds persist on good open patches and every four-law density is positive there. A prediction freeze must retain this nonvacuity certificate.

J and its limiting distribution are not production acceptance predicates. There is no qualified finite-window asymptotic remainder.

## Probability bounds and quality thresholds

For each law/event, sum m_k(C,epsilon)*lo_C and m_k(C,epsilon)*hi_C, with(lo,hi)=(0,0),(1,1),or(0,1) only. Unresolvedcells, failed precision and unfinished scheduled work all receive(0,1). No statistical confidence or IID assertion is needed for this deterministic control-cell integration.

Prediction uses a separately frozen exact finite-window theorem implementation with no observation input. Shared Fraction/interval primitives are allowed; shared selector/gap integration implementation is not allowed to create both observed and predicted answers.

For a positive prediction interval with PL>0, a consistency-row label requires the qualified observed interval width<=min(1/64,PL/4), all coupling/source gates, and overlap. If PL=0<PU, refine the prediction or label INCONCLUSIVE_PRECISION. A prediction proved exactly[0,0] passes an exact-zero row only if the qualified observed upper bound is0; otherwise it is inconclusive or, if rigorously disjoint, a candidate falsification. This is an explicit zero rule, with no arbitrary hidden tolerance. The same quality rule applies to joint rows.

An interval[0,1], an incomplete mass partition, or merely agreeing centers cannot pass. A disjoint qualified comparison is a finite-window candidate falsification, subject to nonauthor examination of numerical/implementation/coupling causes. It is not a direct asymptotic/PRM rejection.

## Prospective adverse controls to design before implementation

These are source requirements, all NOT_RUN:
- cutoff splice and exterior points, physical versus chart units, exact center residual versus cancellation;
- altered chi derivative guard and omitted exterior strip rejected by complete descriptor;
- control mass sums including both signs and all k; dropped failed cells rejected;
- grid value mutation that changes an actual UF pair; independent BFS must reject;
- hidden extra local maximum plateau, equal-value plateaus, periodic seam shortcuts and zero-bar accounting;
- death+1 false connectivity and death false nonconnectivity; stored path/cut must reject;
- swapped physical birth/saddle or chart-distance-for-physical-distance substitution;
- h/dQ/dZ or rounding allowance erased from a coupling certificate;
- boundary/near-fold or near-tie control cells must remain full-mass unresolved when robust separation fails;
- successful-row renormalization, source drift, missing terminal and cap exhaustion withdraw consistency qualification;
- prediction-only code called in the observation route rejected structurally.

A genuine semantic failing test and later GREEN can only be observed after Root issues an exact bounded development execution contract. None is observed in this source-design task.

