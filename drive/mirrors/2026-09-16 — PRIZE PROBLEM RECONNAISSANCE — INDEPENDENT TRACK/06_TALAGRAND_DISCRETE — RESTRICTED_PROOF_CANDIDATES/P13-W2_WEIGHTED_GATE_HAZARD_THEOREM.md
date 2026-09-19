# P13-W2 — Weighted-budget gates satisfy a composable hazard bound

Status: complete author-side proof candidate; external review and historical novelty unresolved. This closes the specific terminal-versus-composable weighted-gate interface identified in P12-E. It does not assert the unrestricted discrete-convexity conjecture.

## Definitions

Let X be finite, d>=1, and b_(ji) be arbitrary finite nonnegative real coefficients. Capacities are positive and normalized to one. The good family is

\[
 \mathcal D=\{U\subseteq X: \sum_{i\in U}b_{ji}<1\ \text{for every }j\le d\}.
\]

Good capacity is STRICT; equality is a failure. D_(K) is the family of unions of K members of D, allowing empty parts. D^(K)=2^X\D_(K). Because D is decreasing, these unions can be turned into partitions.

Under independent probabilities p_i in [0,1], let Q=P_p(D^c). For nonnegative generator prices z, cover cost is sum_(g in G) product_(i in g)z_i. An empty family costs zero. The family containing the empty generator costs one.

## Strong low-failure theorem, without singleton-forbidden coordinates

Suppose max_j b_(ji)<1 for every i, Q<=2/3, A>=1, and z_i<=A p_i. Set

\[
 R=\lceil12A^2\rceil+2,\qquad K(d,A)=8d(R+1).
\]

Then there is an explicit finite generator cover of D^(K(d,A)) of cost at most

\[
 \boxed{2^{2-3d}Q\le Q/2.} \tag{1}
\]

It is the family in P13-W1 with a_i=max_j b_(ji), descending a-order, cutoff by original cumulative p, and weight filter a(g)>=4d.

### Mean bound

For each resource row, write W_j=sum_i b_(ji)X_i and m_j=EW_j. Since b_(ji)<1, Var(W_j)<=m_j. If m_j>=4, Cantelli gives

\[
 P(W_j<1)\le\frac{m_j}{m_j+(m_j-1)^2}\le4/13<1/3.
\]

This contradicts P(W_j<1)>=P(D)>=1/3. Thus m_j<4 and

\[
 M=\sum_i a_i p_i\le\sum_jm_j<4d. \tag{2}
\]

For completeness, Cantelli follows from Markov applied to (W-m-t)^2 and minimizing (Var+t^2)/(m-1+t)^2 over t>=0; zero variance is immediate. The rational function m/(m+(m-1)^2) decreases for m>1.

### Packing implies a heavy prefix generator

Scalar next-fit packing with item sizes a_i<1 and bin load strictly below one has the following elementary property: if U requires more than K bins, a(U)>K/2. Indeed every adjacent pair of bins has combined load at least one; if K is even, pair the first K bins and add the positive first item in bin K+1; if K is odd, pair the first K+1 bins. Zero-size items fit anywhere.

Every scalar bin is also good for all original rows. Therefore failure of K-piece decomposition in D implies failure of scalar packing, hence

\[
 a(U)>K/2=4d(R+1)>4d+RM.
\]

P13-W1 gives coverage and cost (1). This is a sufficient scalar packing argument, not an assertion that multidimensional packing equals scalar packing.

## Full all-probability, transformed-price theorem

Define phi(t)=min(1,-log(1-t)) for t<1, phi(1)=1. Then t<=phi(t)<=2t. For every product vector and every c_i<=phi(p_i),

\[
 \boxed{\operatorname{covercost}_{\mathbf c}(\mathcal D^{(408d)})
 \le\phi(Q)=\min\{1,-\log\mu_{\mathbf p}(\mathcal D)\}.} \tag{3}
\]

Proof: if Q>2/3, then phi(Q)=1 (because e<3), and the empty generator works. Assume Q<=2/3. Let J={i:max_j b_(ji)>=1}, the true singleton-forbidden coordinates. No p_i=1 occurs in J in this range. On X\J, let Q_R be the failure probability of the restricted constraints; Q_R<=Q. Independence and decreasingness give the exact factorization

\[
 1-Q=\prod_{i\in J}(1-p_i)(1-Q_R). \tag{4}
\]

Use singleton generators for J and (1) on the remainder with A=2. Here R=50 and K=408d. The uncapped cost is at most

\[
 \sum_{i\in J}c_i+2^{2-3d}Q_R
 \le\sum_{i\in J}-\log(1-p_i)-\log(1-Q_R)
 =-\log(1-Q).
\]

Capping the cover at one by the empty generator proves (3). Singleton prices are bounded by their hazards, not merely by 2p; it would be false to claim the stronger Q/2 conclusion for arbitrary singleton obstructions.

For Q=0, the positive-cost part vanishes; zero-cost generators remain present if they are required for setwise coverage. A direct alternative is to use all minimal forbidden sets, each of product price zero.

## Consequences

- A single arbitrary weighted constraint has K=408, no dilution, and the all-probability potential bound. No coefficient-ratio, arity or witness-rank bound is imposed.
- Any fixed number d of possibly crossing resource inequalities has K=408d and the same invariant.
- If mu_p(D)>=3/4, the cost is <=log(4/3)<3/10; if mu_p(D)>=2/3, it is <=log(3/2)<5/12<1/2. Positive Taylor partial sums certify both strict comparisons.
- The conclusion holds for K larger than the stated palette by obstruction monotonicity.
- Expressing arbitrary downsets using an unbounded number d of inequalities does not remove the d-dependence. This is not a universal-palette theorem for all downsets.

## Source and priority boundary

Talagrand's original sorted-prefix rounding, presented in Frankston–Kahn–Park (2021), Proposition2.1, underlies the organization by coefficient order. P12 supplied the heterogeneous-prefix and strict-packing interfaces. The new step proved here is the heavy-generator filter and exact-cell bad-seed count yielding a Q-proportional bound at inflated prices. No historical novelty is inferred from the bounded search. Full specialist review remains outstanding.
