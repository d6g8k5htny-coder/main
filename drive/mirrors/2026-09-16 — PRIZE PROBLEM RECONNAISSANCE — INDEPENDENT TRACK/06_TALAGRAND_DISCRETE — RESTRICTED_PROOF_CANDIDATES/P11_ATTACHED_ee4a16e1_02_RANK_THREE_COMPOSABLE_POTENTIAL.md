# P11-B — Rank-three gates satisfy the same hazard potential without dilution

Status: local author-side proof; not externally reviewed; historical novelty unestablished. Uses the elementary P11-A proof, not an unverified numerical constant.

**Theorem B.** Let D be a decreasing family on a finite ground set, containing the empty set, with minimal forbidden size at most3. Put Q=1-mu_p(D) for ANY independent product vector p. Then at prices z_v=phi(p_v), D^(368640) has a generator cover of cost <=phi(Q).

For the part with no singleton forbidden sets, the stronger statement holds: if Q<=2/3 and z_v<=2p_v, there is a cover of cost
\[
 \le\frac{3761}{4608}Q<Q \quad\text{when }Q>0.        (B0)
\]
No raw or weighted degree, codegree, arity, or crossing bound is assumed. The larger palette buys a no-dilution ALL-PROBABILITY invariant suitable for repeated composition; it is not a numerical improvement of Phase08's400 pieces at p/4.

## 1. Dependency lemma

For a uniform r-hypergraph F, w(e)=product_(v in e)p_v and
\[
 c=\max_e\sum_{f\ne e,\ f\cap e\ne\varnothing}
             \prod_{v\in f\setminus e}p_v,
\]
the edge count satisfies EY^2<=mu^2+(1+c)mu. This follows by comparing ordered pairs to mu^2 and charging intersecting pairs by their exact extra-coordinate weight. Therefore if hit probability<=Q<1,
\[
 \mu\le\frac{Q(1+c)}{1-Q}.                         (B1)
\]
A random k-coloring has monochromatic r-edge cost <=sum_e z(e)/k^(r-1). The resulting monochromatic generators cover every induced non-k-colorable set. This use of a fixed coloring is deterministic after averaging.

## 2. High pairs; a monotone two-sprinkle reduction

First assume no singleton witnesses. Include every genuine pair witness in P. Also include each pair ab for which
\[
 \sum_{x:abx\text{ triple witness}}p_x>5.
\]
Remove all triples containing any edge of P; let F0 be the remaining triples. Every pair link of F0 has p-mass<=5.

Take TWO INDEPENDENT samples with coordinate parameters q_v=p_v/2. Their union parameters are p_v-p_v^2/4<=p_v. They are NOT claimed equal to p; increasingness and monotone coupling give union failure probability<=Q.

If the first sample contains P, choose a deterministic first edge. A genuine pair witness already forces failure. A high pair has second-sample completion probability >1-e^(-5/2)>9/10, because the extension vertices are different coordinates and sum q_x>5/2. Choice of pair depends only on the FIRST sample.

Thus P's q-hit probability alpha satisfies
\[
 \alpha\le\frac{10}{9}Q\le\frac{20}{27}<\frac34.
\]
The prices obey z<=2p=4q. P11-A's robust a=4 variant supplies a2560-coloring and edge-generator cover at z with cost
\[
 \le\frac{67}{256}\alpha\le\frac{335}{1152}Q.      (B2)
\]
These generators may be high pairs which are not original violations; overcovering is allowed and is necessary here.

## 3. Low triples: peel vertices and RECOLOR the core

In the current F0, the extension graph of v has edges ab with vab in F0. Its weighted degree at any vertex a is sum_(b:vab in F0)p_b<=5.

Peel v if the extension graph's p-edge mass s_v exceeds100, deleting v and incident triples. By the graph second-moment lower bound, link-hit probability is >100/111>9/10. Let C0 be the peeled vertices and R0 the residual vertices.

Disjoint first-selected-core events as in P11-A give
\[
 1-\prod_{v\in C0}(1-p_v)\le\frac{10}{9}Q.
\]
For Q<=2/3, the product is >=7/27 and
\[
 \sum_{v\in C0}p_v
 \le-\log(1-10Q/9)
 \le\frac{3Q}{2}\log(27/7)<3Q.                   (B3)
\]
The last inequality uses e^2>27/7. Thus sum_(C0) z<=6Q.

Give C0 a16-color palette and R0 a DISJOINT128-color palette. A triple crossing C0 and R0 is never monochromatic. The monochromatic internal-core triple cost has expectation <=
\[
 \frac{(6Q)^3}{6\cdot16^2}
 =\frac{9}{64}Q^3\le\frac{Q}{16}.                (B4)
\]
In the residual, each vertex extension mass is<=100, and each pair extension mass<=5. A fixed triple has directed dependency load at most
\[
 c\le3(100)+3(5)=315.
\]
The one-vertex and two-vertex intersections are charged separately; the former is bounded by vertex-link masses, the latter by pair-link masses. Overcounting only increases this upper bound.

Equation(B1) gives p-edge mass<=948Q. Price inflation costs a factor at most8. Random128-coloring gives residual cost <=
\[
 \frac{8\cdot948}{128^2}Q=\frac{237}{512}Q.
\]
Hence the144-color low-triple cover costs <=(269/512)Q.

## 4. Combine the two whole-ground-set colorings

Refine the2560-color high-pair coloring and144-color low-triple coloring by ordered pairs. This uses2560*144=368640 colors. If a set avoids both generator families, it has no monochromatic high pair or low triple under these colorings, so every original pair/triple is avoided in each product color.

The union of the two generator families is a valid cover, of cost at most
\[
 \frac{335}{1152}Q+\frac{269}{512}Q
 =\frac{3761}{4608}Q< Q\le\phi(Q).
\]
This proves(B0). A family with only pairs or only triples is included; empty edges lists give empty covers. If Q>2/3 use the empty generator, since phi(Q)=1.

## 5. Singleton witnesses and exact hazard addition

Let S be the singleton minimal witnesses. Minimality ensures that no other minimal witness intersects S. Thus the good probability factors:
\[
 1-Q=\prod_{v\in S}(1-p_v)(1-Q'),
\]
where Q' is the hit probability of the pair/triple family on the remaining coordinates. Cover every selected singleton by {v}, at price phi(p_v), and use the preceding cover on the rest.

If Q<1, its cost before capping is at most
\[
 \sum_{v\in S}\phi(p_v)+\phi(Q')
 \le\sum_{v\in S}-\log(1-p_v)-\log(1-Q')
 =-\log(1-Q).
\]
Cap by the empty generator at cost1. Q=1 is immediate. This proves Theorem B for all product vectors and all rank<=3 decreasing families. QED.

The singleton argument requires the ORIGINAL minimal antichain. An arbitrary redundant edge list must be minimalized first; otherwise its remaining witnesses need not have disjoint support from S.
