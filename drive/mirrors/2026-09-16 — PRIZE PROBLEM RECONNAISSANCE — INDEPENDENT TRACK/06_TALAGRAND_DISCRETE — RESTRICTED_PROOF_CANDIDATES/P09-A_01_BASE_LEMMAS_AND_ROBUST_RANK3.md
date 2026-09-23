# P09-A — Elementary cover lemmas and a robust rank-three base

Grade: complete author-side proof candidate; external review and novelty open. Definitions are in 00_DEFINITIONS_AND_SOURCE_BOUNDARY.md. Every claim concerns finite hypergraphs; all random coordinates are independent. Our covering algorithm may be exponentially expensive. No polynomial-time assertion is made.

## 1. Second moments, with the ordered-pair factors retained

For a simple graph G put s_v=sum_{u~v} p_u, c=max_v s_v, and let Y count its selected edges. Write mu=EY=sum_{uv in E}p_up_v.

The unordered pairs sharing v contribute

\[
\sum_v\sum_{\{u,w\}\subseteq N(v)}p_vp_up_w
\le\frac12\sum_vp_vs_v^2\le c\mu,
\]

because sum_v p_v s_v=2mu. Expanding EY^2 counts these pairs twice. The disjoint pairs' ordered contribution is at most mu^2, and the diagonal is mu. Hence

\[
EY^2\le\mu^2+(1+2c)\mu.                 \tag{1}
\]

Cauchy--Schwarz, applied to Y 1_{Y>0}, gives P(Y>0)>=mu^2/EY^2 if mu>0. Thus if P(Y>0)<=epsilon<1,

\[
\mu\le\frac{\epsilon(1+2c)}{1-\epsilon}. \tag{2}
\]

For mu=0 the same upper bound is immediate.

For a simple hypergraph F, let w(e)=product_{v in e}p_v and

\[
c_F=\max_{e\in F}\sum_{f\ne e,\ f\cap e\ne\varnothing}
               \prod_{v\in f\setminus e}p_v.
\]

Now the ordered intersecting joint-probability sum is at most c_F mu. Consequently

\[
EY^2\le\mu^2+(1+c_F)\mu,\qquad
\mu\le\frac{\epsilon(1+c_F)}{1-\epsilon}. \tag{3}
\]

For a uniformly random h-coloring, an edge of size j is monochromatic with probability h^(1-j). If every edge has size at least j, some deterministic coloring has monochromatic-edge cost at most mu/h^(j-1). These edges cover every induced set that is not h-colorable: their avoidance makes this particular fixed coloring proper. A finite conditional-expectation algorithm can choose such a coloring.

## 2. Disjoint-event singleton-core lemma

Consider a deterministic peeling procedure. At stage i, delete all previously selected core vertices and all incident edges. Choose a new vertex v_i whose CURRENT link-completion probability, using only other remaining coordinates, exceeds theta. Let C be all chosen vertices. Assume the original witness-hit probability is at most epsilon<theta.

Let E_i be the event that v_i is present, all earlier core vertices are absent, and a current link is completed. The events E_i are pairwise disjoint and imply original failure. Independence of the remaining coordinates gives

\[
P(E_i)>\theta p_{v_i}\prod_{j<i}(1-p_{v_j}).
\]

If C is nonempty, summing and telescoping yields

\[
\prod_{v\in C}(1-p_v)>1-\epsilon/\theta,
\qquad
\sum_{v\in C}p_v
\le-\log\prod_{v\in C}(1-p_v)
< -\log(1-\epsilon/\theta).              \tag{4}
\]

For an empty C the eventual strict positive upper bounds hold directly. No independence BETWEEN the E_i or between overlapping links is used. Selecting vertices using the realized random set, rather than a deterministic property of the current graph and probability vector, would require a different argument.

## 3. Sprinkling lemma

Let q,t satisfy p_v=q_v+t_v-q_v t_v coordinatewise. Independent q- and t-samples have union law p. Let K be a deterministic family of nonempty cores. For every A in K, assume either A is itself an original forbidden edge, or independently sampling the link H(A) at t completes an original forbidden edge with probability greater than theta.

Choose a fixed total order of K. On the event X_q contains a core, select its first contained core using X_q alone. The t-sample remains independent of this selection, and completion yields an original witness in the union. Therefore

\[
P_q(\exists A\in K:A\subseteq X_q)\le\epsilon/\theta. \tag{5}
\]

One may replace K by its inclusion-minimal members: the core-hit event is unchanged and each retained member keeps its own completion justification. Mere large EXPECTED link count is not a substitute for the completion-probability hypothesis.

## 4. Base theorem

**Theorem A.** If the minimal witnesses of D have size at most three and mu_p(D)>=2/3, then D^(1600) has a cover at p/4 with cost strictly less than

\[
C_3=\frac{135054337}{282419200}<\frac{12}{25}. \tag{6}
\]

This also holds for nonuniform p_v in (0,1).

### 4.1 Singleton witnesses

Let C be the singleton witnesses. They are disjoint from every other minimal witness. The good event requires their absence, so product_{v in C}(1-p_v)>=2/3. Thus sum_C p_v<=log(3/2)<41/100. The p/4 singleton cost is <41/400.

After removing C, the witness-hit probability on the remaining coordinates is at most 1/3: restricting to fewer forbidden events only decreases failure probability.

### 4.2 High pairs and the graph cover

For a pair ab let s_ab=sum_{x:abx is a triple witness}p_x. Form a graph P from all genuine pair witnesses and all pairs with s_ab>10. A triple is low if none of its pairs belongs to P.

Put q_v=1-sqrt(1-p_v). Then two independent q-samples have union law p, and q_v>=p_v/2: indeed 1-p_v/2 is nonnegative and its square is at least 1-p_v. Each high pair has completion probability greater than 1-exp(-5)>99/100 in the second sample. Genuine pair witnesses complete with probability one. By (5), P has edge-hit probability at q at most 100/297.

Peel graph vertices with current weighted q-neighborhood mass >4. Each has neighbor-hit probability >1-exp(-4)>49/50. Equation (4) bounds the q-cost of this core by

\[
\log(14553/9553)<3/7.
\]

In the residual graph c<=4, so (2) gives expected q-edge mass at most

\[
\frac{(100/297)9}{197/297}=900/197.
\]

Choose a 40-coloring of this residual graph. Its monochromatic edges have q-cost at most 900/(197*40). Because p_v/4<=q_v/2, the singleton core and these edges cost at most, with strictness in the core bound,

\[
\frac3{14}+\frac{900}{197\cdot160}          \tag{7}
\]

at p/4. They cover the sets whose induced P cannot be 40-colored.

### 4.3 Low triples

For a vertex v, the link graph of low triples has weighted p-neighborhood mass at most 10 at each vertex. Peel v if its current link edge mass exceeds 100. Equation (1) gives link-hit probability >100/(100+21)=100/121. Applying (4), with epsilon=1/3, bounds the p-cost of the extracted singleton core by

\[
\log(300/179)<21/40.
\]

In the residual low-triple hypergraph, vertex link mass is at most 100 and pair link mass at most 10. For a fixed triple, other triples sharing exactly one vertex have directed dependency contribution at most 3*100; those sharing two vertices contribute at most 3*10. Thus c_F<=330 and (3) gives total edge mass <=331/2. A 40-coloring has monochromatic triple p-cost <=331/(2*40^2).

At p/4, the singleton cost is <21/160 and the triple cost <=331/204800. These generators cover residual non-40-colorability, and the extracted core covers all sets meeting it.

### 4.4 Composition and budget

A set avoiding original singleton witnesses and colorable with 40 colors in both P and the low-triple hypergraph is 1600-colorable in H: refine the two colorings into ordered pairs. Every removed high triple contains a P-edge. Every genuine pair is a P-edge. Every remaining triple is low.

The union of the three covers consequently covers D^(1600). Its cost is strictly below

\[
\frac{41}{400}+\frac3{14}+\frac{900}{197\cdot160}
+\frac{21}{160}+\frac{331}{204800}
=\frac{135054337}{282419200}<12/25.
\]

This proves Theorem A.

## 5. Rational certificates, not unbounded numerical tails

All exponential comparisons above follow from positive finite Taylor partial sums, followed where necessary by positivity of the omitted terms:

- sum_{j=0}^3 (41/100)^j/j! > 3/2;
- sum_{j=0}^6 5^j/j! > 100;
- sum_{j=0}^7 4^j/j! > 50;
- sum_{j=0}^3 (3/7)^j/j! > 14553/9553;
- sum_{j=0}^3 (21/40)^j/j! > 300/179.

The exact checker verifies these inequalities. None of the all-dimensional graph, sprinkling, or coloring arguments is replaced by checking finitely many probability parameters.
