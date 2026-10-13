# P11-A — Recoloring cores gives a composable graph-potential bound

Status: complete local author-side derivation; independent review and historical novelty unestablished. Admission/deployment depends on the separate custody receipt. This is NOT a solution of unrestricted discrete convexity.

## Definitions

For a finite simple graph G on V and independent coordinate probabilities p_v in [0,1], put
\[
 Q=P(G[X_{\mathbf p}]\text{ has an edge}),\quad
 \phi(t)=\min\{1,-\log(1-t)\},\quad \phi(1)=1.
\]
A generator family covers Ind(G)^(K) if every induced subgraph requiring more than K colors contains a generator. For nonnegative prices z_v define its cost by sum_g product_(v in g) z_v. Prices need not be probabilities. Empty family costs 0; the family containing the empty generator costs 1 and covers everything. Always t<=phi(t)<=2t.

**Theorem A.** At prices z_v=phi(p_v), Ind(G)^(128) has a generator cover of cost <=phi(Q), for EVERY finite graph and EVERY product vector. More strongly, if Q<=2/3 and 0<=z_v<=2p_v, a cover consisting of graph edges has cost <=(85/96)Q. The same conclusion holds for every K>=128 by obstruction monotonicity.

The new feature relative to a constant-smallness theorem is the full probability-dependent bound at INFLATED child prices. It is exactly the invariant needed for arbitrary-depth disjoint substitution.

## 1. Second moment, with exact ordered-pair factors

For a graph H let s_v=sum_(u~v)p_u, c=max s_v, Y=count selected edges and mu=EY=sum_(uv in E)p_up_v. Write EY^2 as diagonal plus ordered distinct pairs. Disjoint pairs contribute no more than mu^2. Intersecting distinct edges meet at one vertex; their ordered contribution is at most
\[
 \sum_v p_v s_v^2\le c\sum_vp_vs_v=2c\mu.
\]
Therefore EY^2<=mu^2+(1+2c)mu. Cauchy–Schwarz on Y 1_(Y>0) gives
\[
 P(Y>0)\ge\frac{\mu}{\mu+1+2c}.
\]
If P(Y>0)<=epsilon<1, then mu<=epsilon(1+2c)/(1-epsilon). When mu=0 the assertion is immediate, without division by zero.

## 2. Deterministic core and disjoint first-selected events

Starting from G, remove a vertex v_i whenever its CURRENT weighted neighborhood mass exceeds 2; use any fixed deterministic tie rule. Let C be the extracted core and R=V\C. Current neighbors exclude every earlier extracted vertex.

Let E_i mean v_i selected, all earlier core vertices absent, and some current neighbor selected. These E_i are disjoint, each forces an original edge, and
\[
 P(E_i)>\frac67p_{v_i}\prod_{j<i}(1-p_{v_j})
\]
when the prefactor is nonzero. Indeed 1-product_neighbors(1-p_u)>1-e^(-2)>6/7; e^2>7 follows by a positive Taylor remainder after degree4.

Consequently, with P_C=product_(v in C)(1-p_v),
\[
 1-P_C\le\frac76Q.
\]
For Q<=2/3, P_C>=2/9>0, and
\[
 s_C:=\sum_{v\in C}p_v\le-\log P_C
 \le-\log(1-7Q/6)
 \le\frac{3Q}{2}\log(9/2)<\frac52Q.                 (A1)
\]
The penultimate inequality is the chord bound for the convex function on [0,2/3], zero at0. The last uses e^(5/3)>391/81>9/2, a degree3 Taylor lower bound.

## 3. Do not cover the whole core by singleton generators

Instead assign C a palette of32 colors and R a DISJOINT palette of96 colors. Then no crossing edge C–R is monochromatic.

Randomly color within each palette; a deterministic realization no worse than the mean exists. Alternatively weighted greedy graph coloring realizes the same bound: at each vertex choose a color minimizing the already-colored monochromatic incident weight.

For C, sum z_v<=2s_C<=5Q, so its expected monochromatic edge cost is at most
\[
 \frac1{32}\sum_{uv\in E(C)}z_uz_v
 \le\frac{(5Q)^2}{64}
 \le\frac{25}{96}Q.                              (A2)
\]
For R, max weighted degree<=2. Its failure probability is at most Q, so its p-edge mass is <=5Q/(1-Q)<=15Q. Inflation z<=2p multiplies edge weights by at most4. Thus its monochromatic cost is <=60Q/96=5Q/8.

Total <=(25/96+5/8)Q=(85/96)Q<=phi(Q). Every set avoiding these monochromatic edges is properly128-colorable under the fixed coloring, so the generators cover the required obstruction, not merely a random event. This proves the low-Q case.

For Q>2/3, phi(Q)=1 since e<3. The empty generator gives cost1. Q=0 and coordinates at0 or1 are handled by the nonnegative formulas; no conditioning on an event of zero probability occurs. QED.

## 4. Robust inflated-price variant

If Q<=3/4 and 0<=z_v<=a p_v, a>=1, the same peeling gives P_C>=1-7Q/6>=1/8 and
\[
 s_C\le\frac{4Q}{3}\log8<3Q,
\]
since log2<3/4. Use k_C=ceil(32a^2), k_R=ceil(128a^2). Then
\[
 \operatorname{cost}\le
 \frac{9a^2Q^2}{2k_C}+
 \frac{20a^2Q}{k_R}
 \le(27/256+5/32)Q=\frac{67}{256}Q.               (A3)
\]
The bound is on a specific edge-generator cover. For a=4 the palette is512+2048=2560. For a=2 Theorem A's sharper128 construction is preferable; the robust version serves the later sprinkling step.

No external rounding theorem is consumed. Standard second moments and random/greedy coloring are not claimed novel. The fact that earlier graph-smallness results already exist is separate from priority of this precise composable inequality.
