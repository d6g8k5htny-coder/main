# P11-D — Every fixed primitive rank has a no-dilution, composable hazard bound

Status: local author-side theorem and proof. Not externally reviewed; historical novelty unresolved. Palette depends on the primitive rank. This does not assert one palette for all ranks.

The stronger induction below explains why the graph/rank-three examples are not isolated. It consumes the link-moment recurrence from P09-B, restated and justified here. It does not consume a new unproved structural decomposition.

## 1. Coloring certificates, not just obstruction covers

For a hypergraph H, a coloring certificate is a pair (kappa,G), where kappa uses at most K colors and G is a finite generator family with this property:

> Every H-edge monochromatic under kappa contains some generator g in G.

Every set avoiding G is therefore properly K-colorable by restriction of kappa. Thus G covers Ind(H)^(K). The certificate is stronger than a mere abstract cover.

If every generator has size at least2, independently refine kappa with h extra colors and keep only generators monochromatic in the extra coloring. The certificate remains valid, and the expected new cost is at most oldcost/h. Some deterministic refinement realizes this. In particular THREE extra colors divide cost by at least3, at a factor3 in the palette. This operation is invalid in the presence of singleton generators.

## 2. Induction statement and exact constants

For integer r>=2 set
\[
 e_r=(2r+1)/(3r).
\]
For inflation A>=1, let H_r(A) be the following positive integer palette:
\[
 H_2(A)=192\lceil A^2\rceil.
\]
For r>=3 put
\[
 d=r(2r-1),\quad a=1-1/d=e_r/e_{r-1},\quad \lambda=d-1.
\]
Define integer moments
\[
 u_j=\lambda\left(1+\sum_{i=1}^{j-1}\binom ji u_{j-i}\right),\quad
 T_r=1+\sum_{i=1}^{r-1}\binom ri u_{r-i},
\]
and
\[
 k_C=64\lceil A^2\rceil,\qquad
 k_R=\left\lceil48T_r(Ad)^r\right\rceil,
\]
\[
 H_r(A)=k_C+3H_{r-1}(A/a)k_R.                  (D1)
\]

**Induction claim.** Let H be a finite hypergraph with edge sizes in {2,...,r}. Under independent probabilities p, let Q=P(H is hit)<=e_r. For arbitrary nonnegative prices z_v<=A p_v there is an H-coloring certificate with <=H_r(A) colors, generators of sizes in {2,...,r}, and cost <=Q/2.

No constraint on dimension, incidence or codegrees is imposed. The algorithm may enumerate exponentially many sets. No polynomial complexity claim is made.

## 3. Base r=2

Peel graph vertices with current weighted degree>2. The disjoint first-selected events from P11-A give product_(v in C)(1-p_v)>=1-7Q/6. For Q<=5/6 this is>=1/36. Convexity gives
\[
 \sum_Cp_v\le(6Q/5)\log36<5Q.
\]
For the strict inequality, e^4>36 and 4<25/6, so log36<25/6; finite positive Taylor terms certify e^4>36.

Use64 ceil(A^2) colors on C and128 ceil(A^2) disjoint colors on its residual. Core edge expectation<=25A^2Q^2/(2k_C)<=125Q/768. Residual weighted degree<=2 gives p-edge mass<=5Q/(1-Q)<=30Q and price-edge mass<=30A^2Q; coloring costs<=15Q/64. Total<=305Q/768<Q/2. Crossing edges cannot be monochromatic. The monochromatic internal edges are the generators.

## 4. Proper-link moment lemma

Let F be uniform, and suppose the hit probability of every nonempty proper link is<=a under a product vector t. A link of rank j has weighted edge mass<=u_j as defined in(D1).

Proof by increasing j. For j=1, the hit probability bound gives product(1-t)<=1 but more precisely sum t<=-log(1-a)<=a/(1-a)=lambda. For general j, group ordered pairs of link edges by their exact nonempty intersection. An intersection of size i leaves a further link of rank j-i and mass<=u_(j-i). Thus the dependency load is<=sum_(i=1)^(j-1)binom(j,i)u_(j-i). The second-moment inequality from P11-B and the hit cap a give exactly the recurrence u_j. For a full r-edge count whose hit probability is<=Q, its mass is therefore<=Q T_r/(1-Q). No independence of overlapping link events is asserted.

## 5. Peel high singleton links

Work at the smaller coordinate vector t_v=p_v/d. Greedily delete a vertex v if its CURRENT link (including all current edge ranks) is hit at t with probability>a. Let C be the deleted vertices. Earlier deleted vertices are absent from each current link.

Since p>=t, each current link is also hit at p with probability>a. The same disjoint first-selected events imply
\[
 1-\prod_C(1-p_v)\le Q/a\le e_{r-1}\le5/6.
\]
Using Q<=e_r and the convex chord bound,
\[
 \sum_Cp_v\le-\log(1-Q/a)
 \le (Q/e_r)\log(1/(1-e_{r-1}))
 \le(3Q/2)\log6<3Q.                            (D2)
\]
Here e_r>=2/3 and e^2>6.

Give C its own k_C colors. All original edges wholly in C have size>=2. With M=sum_C z<=3AQ,
\[
 E[\text{core generator cost}]
 \le\sum_{j=2}^r\frac{M^j}{j!k_C^{j-1}}
 \le\frac{M^2}{2k_C}e^{M/k_C}
 \le\frac{M^2}{k_C}
 \le\frac{15}{128}Q.                           (D3)
\]
The exponential bound uses M/k_C<=5/128<1/2 and e^(1/2)<2. Only monochromatic core edges are retained as generators.

## 6. Lower-rank cores in the residual

On V\C, collect all original edges of size<r and all nonempty proper subsets A of r-edges whose residual r-uniform link is hit at t with probability>a. Remove supersets from this collection to form its minimal antichain K.

There are NO singleton cores: peeling stopped with every singleton link hit probability<=a, and the original hypergraph had no singleton edges. Thus K has sizes between2 and r-1.

Set
\[
 q_v=(p_v-t_v)/(1-t_v).
\]
Because d>=15 and p_v<=1, the denominator is positive. Independent q and t samples have union law p exactly, and q_v>=a p_v.

Choose the first core contained in the q sample using a fixed ordering. A core that is an original edge already forces failure. Any other core completes with independent t-hit probability>a. Therefore
\[
 \alpha=P_q(K\text{ is hit})\le Q/a\le e_{r-1}.
\]
The prices obey z<=A p<=(A/a)q. Induction provides a coloring certificate for K of cost<=alpha/2<=Q/(2a). Refinement by THREE extra colors, retaining only matching generators, reduces its cost to<=Q/(6a)<=5Q/28, since a>=14/15. Its palette is3H_(r-1)(A/a).

## 7. Residual uniform r-edges

Delete every r-edge containing a core. Call the remaining uniform family F. Every nonempty proper link of F is hit at t with probability<=a: otherwise its core was collected (or was a singleton that would have been peeled), and those edges could not survive.

Also P_t(F hit)<=Q by monotonicity. The moment lemma gives t-edge mass<=Q T_r/(1-Q)<=6Q T_r. Prices satisfy z_v<=Ad t_v, so price-edge mass<=6Q T_r(Ad)^r.

Random k_R-coloring makes each r-edge monochromatic with probability k_R^(1-r). Since r>=3 and k_R>=1,
\[
 E[\text{residual cost}]
 \le6Q T_r(Ad)^r/k_R^{r-1}\le Q/8.             (D4)
\]
This explicitly retains the potentially large (Ad)^r factor.

## 8. Combine palettes and cost

On V\C use the PRODUCT of the refined core coloring and residual coloring; on C use its separate palette. Any original edge crossing C and V\C is nonmonochromatic. A residual edge removed because it contains a core is covered by the core certificate if monochromatic; an unremoved edge is covered by the residual monochromatic generator. Thus this is an H-coloring certificate. Generators remain of size>=2 and <=r.

Cost is at most
\[
 (15/128+5/28+1/8)Q=(377/896)Q<Q/2.
\]
Palette count is(D1), proving the induction.

## 9. From the induction to arbitrary-depth gate hierarchies

Fix r. For a gate with minimal forbidden size<=r, separate genuine singleton witnesses. For the remaining gate and prices phi(p)<=2p, if Q<=2/3 then the induction applies because e_r>2/3 and yields cost<=Q/2<=phi(Q). If Q>2/3 use the empty generator at cost1=phi(Q). Singleton hazards add exactly on disjoint remaining coordinates as in P11-B.

Therefore every such gate is H_r(2)-compatible with phi. Using P10-A/B, any read-once hierarchy of arbitrary-arity thresholds and arbitrary gates of primitive rank<=r has a SAME-PALETTE H_r(2), no-dilution cover bound
\[
 \operatorname{covercost}_{\mathbf p}(D^{(H_r(2))})
 \le\min\{1,-\log\mu_{\mathbf p}(D)\}.
\]
The sharper128 and368640 palettes from P11-A/B may be used for primitive ranks2 and3 respectively. The general recursion is intentionally very conservative.

## 10. Quantifier boundary

This proves: for each fixed primitive rank r, there exists a palette H_r(2) valid at ALL finite depths and ALL ground-set sizes in the stated class. It does not prove a palette independent of r, nor safe reuse of original variables across child supports. Previously established general bounded-rank results and the possibility of related stronger literature must be checked before claiming novelty.
