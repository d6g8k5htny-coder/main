# P13-W1 — Probability-adaptive prefix certificates

Status: complete author-side derivation; no external review; historical novelty unestablished. This is an extension of the sorted-prefix mechanism attributed to Talagrand in Frankston–Kahn–Park, Proposition 2.1, not a claim to have invented that mechanism.

## 1. Finite setting and exact-cell normalization

Let X be a finite ordered set, p_i in [0,1] independent selection probabilities, and B an increasing family not containing the empty set. Put Q=P_p(B). For a prefix V of X let lambda=sum_(i in V)p_i and Z=product_(i in V)(1+p_i).

Set t_i=p_i/(1+p_i). Direct normalization gives the exact identity

\[
 \sum_{S\subseteq V,\ S\in B}\prod_{i\in S}p_i
 =Z\,P_{\mathbf t}(B\cap2^V)
 \le ZQ\le e^\lambda Q. \tag{1}
\]

The inequality uses t_i<=p_i and monotonicity, followed by 1+p_i<=e^(p_i). This remains valid at p_i=0 and p_i=1. It involves *all bad subsets of V*, not just minimal witnesses. It does not assume that overlapping containment events are disjoint: the probability on the left after normalization is a sum of exact configurations. The normalization is product(1+p_i), not product(1-p_i).

## 2. A seed-extension counting bound

Let F_k be a collection of k-subsets of V such that every g in F_k has a subset S in B of size exactly s. Write m=k-s. At arbitrary nonnegative prices z_i<=A p_i, A>=1,

\[
 \sum_{g\in F_k}\prod_{i\in g}z_i
 \le A^k ZQ\,\frac{\lambda^m}{m!}
 \le A^k e^\lambda Q\,\frac{\lambda^m}{m!}. \tag{2}
\]

Choose one eligible seed S(g) deterministically. The pair (S(g),g\S(g)) determines g. Sum over all eligible s-seeds and all m-element extensions disjoint from each seed. The extension sum is at most lambda^m/m! by the elementary symmetric-polynomial bound. Equation (1) bounds the seed sum even though (1) includes additional seed sizes. Counting extra descriptions is an upper bound. No equality of a set-cover cost with its event probability is assumed.

## 3. The modified prefix family

Let a_1>=...>=a_n>=0 be finite real coefficients, with a_i<1. Define

\[
 P_j=\sum_{i\le j}p_i,\quad M=\sum_i a_i p_i,\quad
 n_k=\max\{j\in\{0,\ldots,n\}:P_j\le k/R\}.
\]

For L>0, retain only the *heavy* prefix generators

\[
 \mathcal G_{R,L}=\bigcup_{k=1}^n
 \{g\subseteq[n_k]:|g|=k,\ a(g)\ge L\}. \tag{3}
\]

The cutoff uses original probability mass p, not generator-price mass z. Price inflation is handled separately by A^k in (2). The weight filter a(g)>=L is essential.

Then

\[
 \{U:a(U)>L+RM\}\subseteq\langle\mathcal G_{R,L}\rangle. \tag{4}
\]

Proof: suppose U={u_1<...<u_l} avoids (3). If a(U)<L, there is nothing to prove. Otherwise let k_0 be the first index with sum_(j<=k_0)a_(u_j)>=L. For every j>=k_0, the j-element initial segment of U is heavy, so u_j>n_j and P_(u_j)>j/R. On price-mass intervals [P_(i-1),P_i), set f(t)=a_i. This is nonincreasing and its integral is M. For each j>=k_0, the interval [(j-1)/R,j/R) lies below P_(u_j); its f-integral is at least a_(u_j)/R. These intervals are disjoint. Thus sum_(j>=k_0)a_(u_j)<=RM, while the earlier sum is <L. Therefore a(U)<L+RM, proving (4). Zero-length intervals and zero-probability coordinates do not invalidate the argument. Zero-cost generators must be preserved.

## 4. Multi-resource bad seeds

Let d>=1 be an integer and let b_(ji)>=0 for j=1,...,d. Set a_i=max_j b_(ji)<1 and

\[
 B=\{S:\exists j,\ \sum_{i\in S}b_{ji}\ge1\}.
\]

Use L=4d in (3). Every retained k-generator has k>=4d+1, since each a_i<1. Its ceil(k/4) largest a-coefficients have sum at least (ceil(k/4)/k) a(g)>=d. Since sum_j sum_(i in S)b_(ji)>=sum_(i in S)a_i>=d, one original resource row has sum at least one. Thus this seed lies in the *original* failure family B.

Consequently (2) applies with

\[
 s=\lceil k/4\rceil,\qquad m=k-s=\lfloor3k/4\rfloor\ge3d. \tag{5}
\]

This is the point where original failure probability Q, rather than a surrogate scalar failure probability, enters the cost.

## 5. Summable probability-proportional cost

On [n_k], lambda<=k/R. For k>=4d+1>=5, k<=2m. Using m!>=(m/e)^m, e<3, and e^x<=1/(1-x) for 0<=x<1,

\[
 A^k e^\lambda\frac{\lambda^m}{m!}
 \le\left(\frac{2eA^2}{R}e^{2/R}\right)^m
 \le\left(\frac{6A^2}{R-2}\right)^m. \tag{6}
\]

Take

\[
 R=\lceil12A^2\rceil+2. \tag{7}
\]

The last ratio is at most 1/2. The map k -> floor(3k/4) takes each integer value at most twice. Equations (2),(5),(6) therefore give the *whole finite family* bound

\[
 \boxed{\operatorname{cost}_{\mathbf z}(\mathcal G_{R,4d})
 \le 2Q\sum_{m=3d}^{\infty}2^{-m}
 =2^{2-3d}Q\le Q/2.} \tag{8}
\]

This infinite geometric tail is an analytic upper bound on a finite generator family; no omitted numerical tail or depth extrapolation is involved.

The elementary estimates used above can be proved without an external concentration theorem: the factorial bound follows by integrating log on [1,m]; e<3 follows from its positive factorial series; and e^x<=1/(1-x) follows term by term from 1/j!<=1. Equation (8) covers Q=0 by nonnegative identities, not division by Q.

## Interpretation and limitations

The earlier unfiltered prefix cover can waste probability mass on singleton generators even when the original failure is quadratic in p. The new weight filter forces each generator to contain an original bad seed occupying at most about one quarter of its vertices. The remaining vertices then make its total price summable. This is not a general hypergraph decomposition: the seed property comes from positive linear resource weights.

No polynomial-time claim: the symbolic prefix family can have exponentially many generators, and evaluating its weighted filter may require exponential computation. Arbitrary real nonnegative coefficients are admitted in the mathematical proof; the companion program tests rational inputs.
