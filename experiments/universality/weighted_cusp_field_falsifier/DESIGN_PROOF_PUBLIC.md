# Sanitized explanatory derivative

This is a publication derivative of the historical 25 October 2026 source document. Mathematical statements are preserved; internal workflow identifiers are removed. It is not the exact pinned source consumed by the tested implementation. References below to authoring-stage NOT_RUN, source-only permission, execution holds, deadlines, private receipts or current Root plans describe that historical scope and have no current operational effect. The accompanying README supplies the later bounded development evidence and its limits.

Original source SHA256: `5a20d81b57a216212f88940b79317f1e3593456855914f206a031e50dfb7767d`. The derivative has a different byte identity.

---

# Same-family complete-field H0 falsifier: mathematical design proof

Status: **HAND-DERIVED SOURCE DESIGN; NOT IMPLEMENTED / NOT EXECUTED / NOT FORMALIZED.** Actual author OpenAI/Codex delegated source author, delegated by the project coordinator on25October2026. Two separately dispatched source-exposed OpenAI/Codex helpers checked the cutoff proof and the H0/coupling argument; their exact roles and findings are retained in RECEIPT.json. This is same-provider conventional mathematical work, ORG0/BLIND0/HUMAN0/SCIENCE_NONE. It neither creates a claim nor changes Root's formal execution HOLD.

The proposed realization and a complete-field H0 coupling are mathematically supportable. The construction below supplies an explicit8×8 periodic integer-grid carrier and a derived error; no unproved epsilon-matching is left to a caller. It does not claim successful implementation, resource feasibility within900seconds, an observed falsifier result, an Hk theorem, a Gaussian law or an asymptotic finite-window remainder.

## 1. Exact field and smooth cutoff realization

Freeze d=2,L=1,w1=1/3,w2=1/9,b=4/9,R=1/8,r1=1/32,A=2^-16,u0=3A^2,v0=2A^3. On the actual torus let
 H(y1,y2)=(1/3)cos(2pi y1)+(1/9)cos(2pi y2).
In the origin chart |yi|<1/4 put
 s=sqrt(2/3)sin(pi y1), t=sqrt(2/9)sin(pi y2), q=s^2+t^2.
Then H=b-q exactly, and the entire seed superlevel above b-R^2 is the chart ball D_R.

Define S=0 on(-infinity,0], S=1 on[1,infinity), and on(0,1)
 S(z)=exp(-1/z)/(exp(-1/z)+exp(-1/(1-z))).
Use the exact functions
 c(q)=q/4+(1-q/4)S((q-r1^2)/(R^2/2-r1^2)),
 chi(q)=1-S((q-r1^2/4)/(3r1^2/4)).
Set
 F0=b-s^2 c(q)-t^2 in D_R, F0=H outside,
 F_(u,v)=F0+chi(q)K(s), K(s)=u s^2/2+v s.
Outside the support of chi the perturbation is zero. This fixes an allowed previously unspecified cutoff realization within the exact TWO/quartic family. It does not assert that an earlier program evaluated these functions.

The flat extension of exp(-1/z) is C-infinity. Its derivatives are polynomial expressions in1/z times exp(-1/z), hence vanish at0+. The denominator of S remains positive near each endpoint. Reflection gives smooth flat gluing at1. Therefore S,c,chi are C-infinity. The c function equals q/4 below r1^2 and1 above R^2/2; chi equals1 below r1^2/4 and0 above r1^2. On the c transition,
 c'=(1-S)/4+(1-q/4)S'/(R^2/2-r1^2)>=0.
Here q<R^2/2<1/2. Thus0<=c<=1,c(q)>0 for q>0. On a full chart collar c=1,chi=0, so F=H with all derivatives. The complete field is jointly C-infinity in physical position,u,v. Every fixed spatial C^j norm is uniformly finite on the compact closed control rectangle; no numerical all-order norm is claimed.

### Independent explicit first-derivative bounds

For0<z<1 write a=exp(-1/z),d=exp(-1/(1-z)). Then
 S'=ad/(a+d)^2 [z^-2+(1-z)^-2]>=0.
On[1/4,3/4], ad/(a+d)^2<=1/4 and each reciprocal square<=16, so S'<=8. For0<z<=1/4 put x=1/z>=4:
 S'<=exp(4/3-x)(x^2+16/9)
    <=(160/9)exp(-8/3)<160/33<8.
The middle inequality uses that x^2 exp(-x) and exp(-x) decrease for x>=4; the final one uses exp(8/3)>1+8/3=11/3. Reflection covers[3/4,1). Thus0<=S'<=8 globally.

Consequently |chi_q|<=32/(3r1^2), |grad_z chi|<=64/(3r1),
 B0<=r1^2/2, D0<=r1,
 B1<=r1+(r1^2/2)64/(3r1)=35r1/3<33r1,
 D1<=1+r1*64/(3r1)=67/3<65.
These B1,D1 are the z-gradient norms explicitly allowed by TWO; they are not untransformed physical-coordinate gradient norms.

On r1/2<=|z|<=r1, c(q)=q/4 and
 |F0,s|=|s|(q+s^2)/2>=q|s|/2,
 |F0,t|=2(1+s^2/4)|t|>=q|t|/2.
Hence |grad F0|>=q|z|/2>=r1^3/16, stronger than the old g*>=r1^3/64.

The old exact guards follow with no existential shrink:
 a*=r1/4=2^-7, delta=r1^4/64=2^-26,
 u0<2^-16=a*^2/4, v0=2^-47<2^-23=a*^3/4,
 B1u0+D1v0<=101441/2^47<2^-22<=g*/2,
 B0u0+D0v0<=1537/2^52<2^-28=delta/4.
Also R^2=1/64<1/9 and eta_c u0^(3/2)=2A^3=v0. These are hand exact inequalities, not observed program PASS statuses.

## 2. Exact four-law coordinates and cell masses

Use a=0,p=1 and the symmetric power weights w_epsilon(eta) proportional to(eta_c-eta)^theta, with equal signs. Let independent Q,Z in[0,1] have densities5Q^4 and k Z^(k-1), respectively, and epsilon be a fair sign. Define
 u=3A^2 Q^2,
 v=epsilon*2A^3 Q^3(1-Z^8),
 k=8(theta+1).
For theta=0,1/2,7/8,1 the exact k values are8,12,15,16.

Indeed Ubar=u/u0=Q^2 has density(5/2)Ubar^(3/2). The wedge coordinate eta=|v|/u^(3/2)=eta_c(1-Z^8), so Dbar=(eta_c-eta)/eta_c=Z^8 has density(theta+1)Dbar^theta. These are precisely the normalized a=0 symmetric p=1 laws. The good domain Q>0,0<Z<1 is inside the open wedge and off the Maxwell line. Its boundary has probability zero; no boundary strip has zero mass merely because it touches that boundary.

For a dyadic cell C=[Qa,Qb]×[Za,Zb] with fixed epsilon, its exact probability is
 m_k(C,epsilon)=1/2*(Qb^5-Qa^5)*(Zb^k-Za^k).
This includes both normalizations. Summing both signs over the full square gives1. Reusing one cell certificate for four weights is intentional coupling, not four independent samples.

## 3. A new direct complete-field H0 argument

This section derives the actual field selection structure and does not assume the author's N in{0,1}, its lifetime formula, or a critical_fixture output.

### Global vertical projection preserves actual path components

Inside the chart,
 F_t=-2t[1+s^2 c'(q)-chi_q(q)K(s)].
On the cutoff support |s|<=r1,
 |chi_q K|<=32/(3r1^2)*(u0 r1^2/2+v0 r1)
           =16A^2+64A^3/(3r1)<2^-27.
Outside that support the product is zero. Therefore the bracket is strictly positive everywhere in the chart.

For a physical coordinate y2 in(0,1/2), the field strictly decreases as y2 increases away from0, except for the seed's zero derivative at the endpoint1/2. In the chart dt/dy2=pi*sqrt(2/9)*cos(pi*y2)>0 because |y2|<1/4; t has the sign of y2, so the displayed chart derivative has the required physical sign by the chain rule. Outside D_R the field is the seed, whose derivative is -(2pi/9)sin(2pi y2)<0. Smooth collar agreement covers every crossing on a fixed-y1 vertical fiber. Reflection proves the analogous statement on(-1/2,0).

The projection p(y1,y2)=(y1,0) is globally continuous and height increasing. Each point has a vertical path to its projection whose height never decreases: choose either arc at y2=1/2. A continuous global choice of those paths is neither needed nor asserted. For each closed superlevel, p and the axis inclusion induce inverse maps on actual path components, because every point is joined to its projection and p sends every path to an axis path. The maps commute with every superlevel inclusion. This proves a **natural all-level pathcomponent equivalence**, hence a natural ordinary-H0 equivalence over every field.

There is no torus-to-circle global deformation retraction claim. In particular this does not remove or identify higher homology.

### Complete cyclic axial profile

On the exact core the axial residual is
 P(s)=-s^4/4+u s^2/2+v s.
The old guards exclude every extra chart critical point on the annulus. On the axis, the transverse derivative is identically zero, so annular gradient exclusion also excludes axial critical points there. Outside the chart the only axis critical point is y1=1/2, a strict minimum, with field value -2/9. All new axial critical points are the real roots of
 -s^3+u s+v=0,
and lie in |s|<a*. This is a complete list, independently justified by the actual derivatives and the seed.

For good Q,Z, put s=Ax. The equation is
 f_C(x)=-x^3+3Q^2 x+epsilon*2Q^3(1-Z^8)=0.
It has three simple roots x1<x2<x3 in[-2,2], with maxima at the outer roots and a minimum at x2. For epsilon=+ they lie in(-2Q,-Q),(-Q,0),(Q,2Q); epsilon=- is reflected. This follows from the signs at the turning points plus the cubic's monotonicity. No root is inserted as a supplied maximum certificate.

The cyclic axis thus has two local maxima, the intervening high minimum, and one deep exterior minimum. Its superlevel components are determined by this cyclic critical profile: two births, the first merger across the high intervening minimum, and no further H0 birth at the deep minimum. The global vertical equivalence transfers this actual component behavior to the full torus. Ordinary H0 has one essential class and exactly one positive finite interval for good controls; the higher observed peak survives. The physical critical point above the intervening minimum has positive axial and negative transverse second derivative, so it is the actual killing saddle in dimension2.

Which outer maximum is younger must be read from the field-grid UF/BFS output in the experiment. The conventional peak-order argument may be compared as a held-out prediction; it is not the observed selector.

## 4. Exact floor and scaled box: all exterior is certified

Put s=Ax,t=A^2y. In the exact core,
 Hc(x,y)=(F-b)/A^4
 =P_c(x)-(1+A^2x^2/4)y^2,
 P_c(x)=-x^4/4+(3/2)Q^2x^2+epsilon*2Q^3(1-Z^8)x.
The closed box B=[-8,8]×[-16,16] is wholly in the exact core:
 s^2+t^2<=64A^2+256A^4<r1^2/4.

For every x and every declared control, P_c(x)<=6. For a=|x|,
 6-[-a^4/4+(3/2)a^2+2a]
   =(a-2)^2(a^2+4a+6)/4>=0.
For a>=8 the same upper polynomial decreases and is<=-912. Therefore, within the exact core, outside B:
 |x|>=8 implies Hc<=-912;
 |y|>=16 implies Hc<=6-256=-250.
Outside the exact core, the old guard proof gives
 F-b<=-3delta/4=-3*2^36 A^4<-16A^4.
Thus **the entire complement of the open box has F<b-16A^4**. No omitted annulus or external bypass remains in this bound.

Set lambda=b-16A^4 and floor the computational field at lambda. Above lambda the complete full-field superlevel is precisely the local-box set. Its fibers contract to y=0 while increasing F, naturally with the level. Every new axial critical height is above lambda: at a critical root,
 P_c(x)=(3/4)(x^2-Q^2)^2-(3/4)Q^4>=-3/4.
The deep old minimum lies below lambda. The cyclic profile and vertical projection therefore show that the original full-field superlevel is already connected at and below lambda. Raising its lower values to lambda changes no ordinary-H0 persistence object, up to the described natural component isomorphisms. It can change higher topology, for which no claim is made.

This uses the same full field and a proved H0-preserving computational floor. It is not a replacement local family.

## 5. A certifiable8×8 periodic graph, with derived error

For each parameter cell C fix its dyadic center(Qc,Zc). A **uniform root certificate** must isolate all three roots over the whole cell in disjoint rational intervals
 I1=[alpha1,beta1], I2=[alpha2,beta2], I3=[alpha3,beta3]
contained in[-3,3], in their displayed order. Enforce endpoint signs and strict derivative branch bounds for every control in C: f decreases through I1, increases through I2, decreases through I3. Direct exact polynomial bounds suffice. A missing sign, overlapping interval, boundary degeneracy or exhausted refinement is UNRESOLVED, with the cell's full mass retained. No root oracle is assumed correct without a checkable polynomial certificate.

Use exactly eight x nodes
 [-8,alpha1,beta1,alpha2,beta2,alpha3,beta3,8]
and eight y nodes
 [-16,-8,-4,-2,0,2,4,16].
The periodic seam represents the long, below-floor exterior route. All boundary rows and columns are exactly at the floor. This is a periodic graph carrier of the proved floor reduction, not a claim that these tiny chart coordinates uniformly tile the physical torus.

At every node compute the exact rational core value at(Qc,Zc), floor it at-16, and set
 V_ij=floor(2^80*max{Hc(x_i,y_j),-16}).
Q,Z,x,y,A are rational, so these nodal values are rational expressions followed by exact integer division. No exponential or transcendental field evaluation is needed inside the certified exact core. The full field and exterior certificate remain part of the frozen object.

Feed these64 integer samples into the actual unchanged exact_h0.compute and separately exact_h0.verify_by_connectivity. Their returned positive intervals and births are the observed graph barcode. Do not call critical_fixture or create those observed endpoints from critical-height formulas.

For every above-floor integer threshold each active vertical column is an interval containing the y=0 node. Adjacent columns join exactly when their y=0 nodes join: any horizontal active edge has both axis endpoints active, and the axis edge then exists. Projection to the axis gives inverse natural component maps for this graph. Quantization is monotone, so it cannot introduce extra axial extrema. The certified alternating root branches imply at most two regional maximum plateaus and their one intervening valley. Robust gaps below ensure that neither the finite bar nor peak order collapses under quantization.

Let h=max_i(beta_i-alpha_i), dQ=(Qb-Qa)/2,dZ=(Zb-Za)/2. Require h<=1 and every critical bracket lies in[-3,3]. There |P_c''|=|3Q^2-3x^2|<=30. A root's height differs from either bracket-end height by at most15h^2, by Taylor's theorem with zero first derivative. At all six critical-bracket nodes,
 |partial_Q P_c|<=3x^2+6|x|<=45,
 |partial_Z P_c|<=16|x|<=48.
The maximum or minimum of a finite pair is1-Lipschitz in the pair's sup norm. Integer rounding loses less than2^-80. Hence set the explicit derived scaled endpoint bound
 **E(C)=15h^2+45dQ+48dZ+2^-80.**
For every control in the whole cell the corresponding true critical birth/death heights are within A^4E(C) of the actual observed grid endpoints, expressed as b+A^4 V/2^80. This is a source-derived bound, not a caller-supplied epsilon.

A tighter, still explicit optional bound can use X=max absolute critical-bracket endpoint, Qb and Zb from the actual cell:
 M2=3Qb^2+3X^2,
 MQ=3Qb X^2+6Qb^2(1-Za^8)X,
 MZ=16Qb^3 Zb^7 X,
 E_local=(M2/2)h^2+MQ*dQ+MZ*dZ+2^-80.
The same derivative/Taylor argument proves this replacement. The implementation must retain the exact chosen constants and inequality certificates; an arbitrary smaller tolerance is not accepted. This tightening may matter near Q=0 or Z=0 but supplies no performance guarantee. All subsequent uses of E must consistently use the proved selected bound.

Accept a resolved selector-coupling certificate only if the observed UF/BFS output has one positive finite bar and one essential class, its two birth plateaus lie in the two outer-root columns, its intervening axial bottleneck agrees with the observed death, and both the observed finite lifetime and the observed essential-minus-finite birth gap exceed2E(C) in scaled units. Otherwise the cell is unresolved or implementation-failed, as appropriate. These robust conditions bind peak identity and prevent disappearance into the diagonal; a generic sup-norm bottleneck bound would not suffice.

The exact matching then has
 endpoint error <=A^4E(C),
 lifetime error <=2A^4E(C),
 zero unaccounted positive finite bars.
The corresponding essential birth has the same endpoint error. The all-level natural reductions and the complete alternating profile establish the interval matching; no spatial interpolation theorem for a coarse uniform mesh is borrowed.

An independent BFS path at the observed death, and its failure at death+1 integer unit, supply the actual merger witness. The path's axial projection must cross I2 unless it uses the floor seam, which the robust death level excludes. The axis bottleneck at that level identifies the killing-critical bracket I2. Retain this witness and its root-location certificate; the existing UF result alone does not emit a physical saddle coordinate.

The death is explicitly above the floor: at the evaluated center the true middle critical value is at least-3/4. Each I2 endpoint differs by at most15h^2 and integer flooring loses less than2^-80. Since h<=1, the observed axis bottleneck is at least-3/4-15-2^-80>-16. The checked observation must also satisfy D>-16*2^80. Thus a floor seam cannot provide the observed earlier merge; this is proved independently of the positive lifetime margin.

## 6. Physical endpoint and metric certificates over entire cells

At all three critical points t=0. In the centered physical torus chart define
 rho(x)=pi^-1 arcsin(Ax/sqrt(2/3)).
For |x|<=3 the argument has absolute value<4A=2^-14. Direct differentiation and elementary rational bounds3<pi<4 give
 A/4<rho'(x)<A.
For example sqrt(2/3)>4/5 and sqrt(1-(3/2)A^2x^2)>4/5 make the upper bound strict. Thus a critical bracket of width h has physical width<A h. The full x-root intervals, not just a center control, localize the true birth and killing saddle for every control in C.

Map I_younger and I2 through the monotone rho and subtract with the observed orientation to enclose the actual separation r. All positions are in a neighborhood of0 of diameter<6A<1/2, so this positive axial separation is the unique shortest torus distance; no Euclidean lift ambiguity remains. Haar translation may be attached afterward with its own exact or finite-word location certificate. It is not needed for translation-invariant lifetime/r experiments and is not an observed spatial process by itself.

A fully finite rational inverse-chart certificate can be fixed now. Isolate sqrt(3/2) to absolute width<=2^-128 by128 dyadic bisections. Enclose pi by Machin's identity16 arctan(1/5)-4 arctan(1/239), using96 alternating terms per arctan; the alternating remainders are explicit. For arcsin use terms n=0,...,16 of
 arcsin z=sum [binom(2n,n)/(4^n(2n+1))] z^(2n+1).
For |z|<2^-14, the remaining tail has absolute value at most
 |z|^35/(1-z^2)<2^-489.
Apply the odd extension for negative z and rational interval arithmetic. The sqrt/pi enclosure sensitivity and this tail together are safely below2^-120 per endpoint. The certificate must retain these exact interval operations and tail inequalities; it cannot merely assert a floating arcsin tolerance.

Consequently each reported physical endpoint enclosure includes the exact mapped root bracket plus an outward2^-120 arithmetic allowance, and separation includes both endpoint allowances. No unknown physical epsilon is left to the caller. A graph birth vertex can additionally be mapped as an observed approximate location, with positional error<=A h+2^-120 to its true critical point.

Root's dated actual DESIGN V2 controls the production schedule: **t=A^4 times{1/4,1/16,1/64}**, with joint marks **r/A<=1/16, r/A<=1/8, and (birth-b)/A^4<=0**. V2 preserves and corrects V1's uninformative1/2,1 separation bins before any program/test/prediction/production observations. Lifetime and physical endpoint enclosures above classify these closed-threshold events; every straddling cell remains unresolved at full mass. J is not an acceptance predicate. If retained for optional descriptive output, fix ell*=u0^2=9A^4,r*=A and use exact outward logarithm intervals. There is no supplied asymptotic numerical remainder for the limiting J distribution.

### Exact positive-mass nonvacuity of the V2 separation bins

Fix Z=255/256 and epsilon=+. Bernoulli's inequality gives a=1-Z^8 in(0,1/32). With x=Q*xi the cubic is -xi^3+3xi+2a=0. Its younger outer root is in(-2,-3/2), and its middle root is in(-1/16,0): the signs at these rational endpoints follow from a<1/32, and the branch monotonicities give uniqueness. Therefore23/16<xi_middle-xi_younger<2. The physical map has the stronger bounds3A/8<rho'<A/2 on[-3,3]. For the lower bound use sqrt(3/2)>6/5 and pi<16/5, the latter from Machin's formula and arctan(1/5)<1/5. For the upper use pi>3,sqrt(2/3)>4/5 and sqrt(1-z^2)>9/10. Hence
 69Q/128<r/A<Q.
At Q=1/16 the mark is strictly below1/16; at Q=1/8 it lies strictly between1/16 and1/8; at Q=1/4 it is strictly above1/8. These are actual same-family physical marks.

All three controls satisfy every chosen lifetime window. For the younger negative root its scaled height is at most(9/4)Q^4, since its positive-tilt linear term is negative and the unperturbed quartic-plus-quadratic maximum is9Q^4/4. The middle critical height is at least-3Q^4/4 by the critical-height identity. Thus0<ell/A^4<=3Q^4<=3/256<1/64. Strict root,mark and lifetime inequalities persist on open Q,Z neighborhoods of these good interior controls. Every law density5Q^4*kZ^(k-1) is strictly positive there, so every selected window has positive mass below, between and above the new separation bins. No observed field evaluation or kappa integral is used in this nonvacuity proof.

## 7. What an implementation must observe and what may remain inconclusive

The observed route is full-field definition/exterior proof -> exact nodal samples -> actual64-node UF -> separate BFS -> checked continuum/physical matching. The prediction route is the separately frozen weighted theorem finite CDF/joint marked formulas. The observed route must not import population_certificate.critical_fixture, its gap integral, or its finite-count result.

Per-law lower and upper event probabilities are sums of exact m_k(C,epsilon) times the cell's certified0/1 event enclosure. For an unresolved cell use[0,1], including every boundary strip and every budget/precision failure. Exact total mass bookkeeping must show resolved plus unresolved mass1 for every law. Shared cells across laws are correlated by design.

A certified, source-bound graph/field mismatch is first an implementation/coupling failure until its arithmetic/topological certificates receive actual review. A structural counterexample requires a fully checked same-family field and contradictory actual selection evidence. A qualified finite-window discrepancy requires disjoint rigorous observed and predicted enclosures after all error and unresolved mass. Overlap is finite-window consistency only. An unqualified root/coupling/mark cell or budget exhaustion is INCONCLUSIVE, not a theorem verdict.

Mathematically the coupling route is established at this hand-proof scope; engineering completion and its promised time/resource budget remain unestablished. Exact arithmetic and64 vertices reduce the geometry cost but do not prove that4096 adaptive cells or512 bisections suffice for a specified probability interval. The plan must freeze a refinement policy, terminal budgets and uncertainty reporting before any implementation execution. Root subsequently reported actual historical source-author assignment and ROOT_SOURCE_IMPLEMENTATION_GO.json16:58:00UTC. That is source/implementation GO only; programs/tests/sampler/numerical/formal remain NO_GO_NOT_RUN, and this design supplies no execution admission. No target program, mathematical evaluation, test, sampler or compiler ran here.

