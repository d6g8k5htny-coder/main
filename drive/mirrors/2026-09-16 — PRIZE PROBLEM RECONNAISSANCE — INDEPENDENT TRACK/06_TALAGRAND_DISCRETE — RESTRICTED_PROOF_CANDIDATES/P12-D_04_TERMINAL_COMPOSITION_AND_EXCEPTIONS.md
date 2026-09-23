# P12-D — Disjoint subtree composition, and cheap exceptions

Status: author-side corollary. The exact common-palette identity from P10-A is reproduced below; application to P11's classes inherits their author-side/review-pending status.

Partition the ground set into disjoint blocks X_i. Let D_i be decreasing families on those blocks, each containing the empty set. Write q_i=P(X_i,p notin D_i). Choose one integer K and assume every D_i^(K) has a generator cover G_i of cost c_i<=phi(q_i).

Let the outer family A be an intersection of d positive weighted constraints as in P12-C. Define

\[
 D=\{U:\{i:U\cap X_i\notin D_i\}\in A\}.
\]
If mu_p(D)>=3/4 and K>=192d, then

\[
 \boxed{\operatorname{covercost}_{\mathbf p}(D^{(K)})<31/70.} \tag{D1}
\]
There is NO extra probability dilution, and no product palette at this disjoint substitution.

## Exact substitution proof

Let J_K(U)={i:U cap X_i notin (D_i)_(K)}. Then

\[
 U\in D_{(K)}\iff J_K(U)\in A_{(K)}. \tag{D2}
\]
Forward: any K good parts of U induce K allowed active-port sets; every hard block must be active in at least one. Reverse: assign all of each hard block to one color from a K-cover of J_K(U) in A. Every soft block has its own K-part good decomposition, making it inactive in every color. Combine on disjoint supports. No color labels are multiplied.

Original child-failure events are independent because supports are disjoint, so the root's original probability equals mu_q(A). Apply P12-C at prices c_i. For each outer generator J take all unions of one child generator from each i in J. These unions cover D^(K), and their cost is at most product_{i in J}c_i, exactly because the supports are disjoint. Sum over the outer cover. D1 follows.

## Applications to earlier exact input snapshots

- Arbitrary-depth disjoint threshold and graph subtrees from attached Phase11 can use any K>=128 by obstruction monotonicity. Therefore one terminal root with d weighted resource inequalities gives D1 with K=max(128,192d)=192d.
- Allow arbitrary primitive rank-three subtrees: use K=max(368640,192d).
- For a fixed general primitive rank r: use K=max(H_r(2),192d), preserving the rank dependence of H_r.

These examples allow unbounded original witness rank and arbitrarily many original coordinates. They do NOT permit repeated original coordinates across child supports.

## Cheap arbitrary exceptional constraints

Suppose D' subset D and the discrepancy D minus D' is covered by generators of total original product cost beta. Every U avoiding those generators has the same available K-decompositions in D and D', so

\[
 (D')^{(K)}\subseteq D^{(K)}\cup\langle R\rangle.
\]
If mu_p(D')>=3/4 and beta<=1/20, then mu_p(D)>=3/4 and

\[
 \operatorname{covercost}((D')^{(K)})<31/70+1/20=69/140<1/2. \tag{D3}
\]
This allows arbitrary crossing extra constraints, but their actual generator cover must be proved. A small exceptional probability is not that premise.

## Exact remaining limitation

P12-C returns a constant <31/70. It has not been proved here to return <=phi(Q_root) at ALL root probabilities. Therefore it cannot be substituted recursively as another hazard-compatible gate. This pass closes a terminal/root interface. P12-E demonstrates the failure of that inference for the particular prefix certificate. It does not disprove that a stronger weighted-gate theorem could exist.
