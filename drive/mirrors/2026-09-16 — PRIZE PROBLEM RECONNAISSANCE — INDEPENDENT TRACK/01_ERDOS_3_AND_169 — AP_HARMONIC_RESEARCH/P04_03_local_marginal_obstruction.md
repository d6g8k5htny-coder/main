# PR-AP-015 — Fixed-order local marginals cannot certify a uniform harmonic bound

2026-09-16. Elementary complete author-side proof; no novelty claim. This is an obstruction to a DEFINED relaxation, not to finite proofs or all local methods.

Fix N>=r>=2 and k>=3. For each S subset[N] with |S|<=r, a rank-r local-marginal relaxation assigns a probability distribution mu_S on subsets of S. Require:

1. nonnegativity and total mass1;
2. support only on k-free subsets;
3. for T subset S, the projection of mu_S onto T equals mu_T.

Maximize sum_(i=1)^N mu_{ {i} }({i})/i.

**Theorem.** This relaxation has feasible objective H_N/r, where H_N=sum_(i<=N)1/i.

Proof. Assign

    mu_S(empty)=1-|S|/r,
    mu_S({i})=1/r for each i in S,
    mu_S(A)=0 otherwise.

All masses are nonnegative because |S|<=r. They sum to1 and only empty/singleton sets have mass, so support is k-free. Under projection to T, each singleton in T retains mass1/r; the empty mass becomes 1-|S|/r+(|S|-|T|)/r=1-|T|/r. This proves exact consistency. The objective is H_N/r. QED.

Consequently no fixed r gives an N-uniform bound for this relaxation. To certify an upper C from this relaxation it is necessary that r>=H_N/C. This logarithmic-order condition is necessary, not sufficient.

For N>r these marginals do not extend to ANY global probability distribution, whether or not AP-freeness is imposed. Every pair has joint inclusion probability zero, so a global random set would have at most one element almost surely. Yet the marginal expected cardinality is N/r>1, contradiction.

For r=1 the objective can be H_N directly; the pairwise nonextension argument uses r>=2.

This does not assert the existence of a k-free integer set with divergent reciprocal sum. The fake solution exists only in the relaxation. It does not rule out higher-rank hierarchies, semidefinite constraints, global inequalities, unbounded derivation depth, or a finite proof using suitable mathematical invariants. The 3-AP/4-AP predicate makes no difference to this particular singleton construction.
