# P11-B — A no-dilution, hazard-compatible rank-three gate

Grade: complete author-side derivation, subject to external correctness and novelty review. No prize closure. Constants are deliberately loose. Unlike a theorem only at good probability3/4, the conclusion below works at EVERY probability and at inflated child-cover weights; that is the composability required by P10.

Let D be a downset on finite X with minimal forbidden sets of size at most3. Let independent coordinates have probabilities p_v in[0,1], and let costs satisfy 0<=w_v<=phi(p_v), where phi(p)=min(1,-log(1-p)), phi(1)=1. Put Q=1-mu_p(D).

**Theorem.** D^(262144) has a generator cover of w-cost <=phi(Q). If there are no singleton witnesses and 0<H=-log mu_p(D)<1, the constructed nontrivial cover costs strictly less than

    (9437/12288) H <(4/5)H.

The proof works for unequal p_v. There is NO probability dilution. Zero/unit probabilities and empty/full downsets are treated explicitly at the end.

## 1. Work first with pair and triple witnesses, no singletons

It suffices initially to assume 0<H<1; then Q<2/3 and exp(H)-1<2H. Set

    q_v=p_v/2,    t_v=p_v/(2-p_v).

The independent union of a q-sample and a t-sample has the p law since q_v+t_v-q_v t_v=p_v. Also t_v>=p_v/2.

For each pair ab let d_p(ab)=sum_x p_x over DISTINCT x with abx an original minimal triple. Form a simple graph P containing every original pair witness and every pair with d_p(ab)>10. Let H0 consist of the original triples containing NO edge of P.

Every pair of a low triple has weighted extension sum <=10, including when only a subfamily of H0 remains.

## 2. First-sprinkle graph probability

When the q-sample contains an edge of P, select its first edge by an order fixed before either sample. If it is an original pair witness, the union is bad already. Otherwise its independent t-extensions hit with probability

    1-product_x(1-t_x) >=1-exp(-sum_x t_x)>1-exp(-5)>99/100.

Thus the q-random graph non-independence probability Q_P satisfies

    Q_P <= alpha Q,     alpha=100/99.

The second sample is independent of the choice; no conditional independence inside the first sample is asserted.

Since Q<2/3, Q_P<200/297. Define H_P=-log(1-Q_P). Monotonicity and integration give

    H_P-H <= integral_Q^(alpha Q) dt/(1-t)
             <=(alpha-1)Q/(1-alpha Q)
             <=(3/97)Q <=(3/97)H.

If H_P<H the inequality is automatic. Hence H_P<=(100/97)H.

## 3. High-pair graph cover, at the ORIGINAL cost vector w

Peel vertices of P with CURRENT q-weighted degree >2 as in P11-A. Write C_P for the core and R_P for the residual. The same disjoint first-core-selected events show

    product_(v in C_P)(1-q_v) >=1-(7/6)Q_P >=(1-Q_P)^2,

because Q_P<200/297<5/6. Thus S_P=sum_C_P[-log(1-q_v)]<=2H_P.

Here w_v<=2p_v=4q_v. Consequently

    sum_C_P w_v <=4S_P <=(800/97)H <9H.

Core-internal pair w-mass is <(81/2)H^2<=(81/2)H. On the residual, the graph second-moment inequality gives

    mu_q <=5Q_P/(1-Q_P)
           <=(1500/97)Q <16H.

The residual pair w-mass is <16*16H=256H. Use256 colors on C_P and a DISJOINT768 colors on R_P. Conditional expectation gives one1024-coloring for which the monochromatic-edge cover has w-cost

    < [81/512+1/3]H = (755/1536)H.              (A)

All crossing edges have different colors. This is an actual fixed-coloring cover, not just an averaged success probability.

## 4. Low-triple vertex peeling

For a current vertex v in H0, its extension graph L_v has edges xy whenever vxy remains. Every vertex of L_v has p-weighted neighborhood sum <=10, because its value is a low-pair extension sum. If its total edge mass s_v exceeds84, the graph second moment gives

    P_p(L_v contains a selected edge) >=s_v/(s_v+21)>4/5.

Greedily remove such v, deleting incident triples. In the first-core-selected events, earlier extracted coordinates are absent and links use only remaining coordinates, so the events are independent where required and disjoint across steps. Their total is at most Q. For this low-triple core C_0,

    product_C_0(1-p_v) >=1-(5/4)Q >=(1-Q)^2,

since Q<2/3<3/4. Hence sum_C_0 w_v<=sum_C_0[-log(1-p_v)]<=2H.

Core-internal triple w-mass is <=(sum_C_0 w_v)^3/6 <=(4/3)H^3 <=(4/3)H.

## 5. Low-triple residual

Every residual vertex extension mass is <=84; every residual pair extension mass is <=10. For a fixed residual triple e, partition other intersecting triples by exact intersection size. One-vertex intersections contribute at most3*84; two-vertex intersections contribute at most3*10. The directed dependency load is therefore <=282.

For any uniform hypergraph, let mu=sum_e product_(v in e)p_v and let c=max_e sum_(f!=e,f intersects e) product_(v in f\e)p_v. Its selected-edge count obeys

    E Y^2 <=mu^2+(1+c)mu.

Indeed disjoint pairs are covered by mu^2 and ordered intersecting pairs by c mu; the diagonal is <=mu. Cauchy--Schwarz gives mu<=(1+c)Q/(1-Q).

Applied here,

    mu_p <=283Q/(1-Q) <566H.

Because w_v<=2p_v, residual triple w-mass is <8*566H=4528H.

Use128 colors on C_0 and a DISJOINT128 colors on its residual. Crossing triples are automatically nonmonochromatic. A chosen256-coloring gives a monochromatic-triple cover with cost

    <(4/3+4528)H/128^2 = (3397/12288)H.       (B)

## 6. Compose the two covers

A set avoiding cover(A) is1024-colorable in P by the fixed graph coloring. A set avoiding cover(B) is256-colorable in H0 by the fixed triple coloring. Their ordered-pair coloring uses1024*256=262144 colors. Every original pair is an edge of P; every removed triple contains an edge of P; every low triple belongs to H0. No original witness is monochromatic.

Thus the union of the two generator families covers D^(262144), with total cost

    <(755/1536+3397/12288)H=(9437/12288)H<H.

No multiplication of cover costs is substituted for their required ADDITION. Product palettes and summed cover costs are different operations here.

## 7. Singleton witnesses and endpoints

Minimal singleton witnesses J occur in no other minimal witness. Original good probability therefore factors into their absence probability times the independent pair/triple subsystem's good probability. Let H_J=sum_J[-log(1-p_v)] and H_*=the remaining hazard. Include singleton generators J at cost <=H_J and the constructed pair/triple cover at cost <=H_*. Cap by an empty generator if needed. This gives <=min(1,H_J+H_*), the claimed phi(Q).

If H_*=0, all pair/triple products are zero and the entire witness list is a zero-cost cover. If total good probability is zero or total hazard>=1, the empty generator has cost1 and suffices. The full downset has empty obstruction and empty cover. The empty downset has empty witness and the trivial cost-one cover. These cases are kept distinct in the implementation.

## 8. What is and is not improved

At good probability>=3/4, this proves a rank<=3 K=262144 no-dilution cover. More importantly it proves the all-probability TRANSFORMED-WEIGHT inequality required for unbounded-depth composition. It does not dominate the smaller-palette/diluted rank-three statements on all parameter axes. The original unrestricted problem permits arbitrary-rank primitive gates and remains open here.
