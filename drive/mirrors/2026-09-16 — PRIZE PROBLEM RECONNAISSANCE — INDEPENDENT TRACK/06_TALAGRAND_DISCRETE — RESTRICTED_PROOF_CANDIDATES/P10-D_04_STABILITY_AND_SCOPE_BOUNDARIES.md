# P10-D — Stability under cheap exceptional causes, and strict scope boundaries

Author-side elementary proofs; external review and novelty unestablished.

## 1. A cover-stability lemma

Let D subset D0 be decreasing families on the same finite ground set. Suppose the difference D0 minus D is covered by an increasing generator family <R> with product-weight cost at most beta. Then, for every k,

\[
\boxed{D^{(k)}\subseteq D0^{(k)}\cup\langle R\rangle.} \tag{1}
\]

Indeed, if U avoids <R> and is a union of k sets V_i in D0, then each V_i subset U also avoids <R>. Therefore each V_i belongs to D. The implication and cover-cost addition follow.

If D0 is in P10-C's class, mu_p(D)>=3/4, and beta<=1/5, then

\[
\operatorname{covercost}_{\mathbf p}(D^{(64)})
\le -\log\mu_{\mathbf p}(D0)+\beta
\le\log(4/3)+1/5<1/2.
\]

This permits arbitrarily crossing exceptional witnesses, of arbitrary rank, PROVIDED their exceptional effect has the explicit small generator cover. There is no claim that every hypergraph admits such a decomposition.

## 2. Cheap repeated-variable core

Let F be any monotone zero-preserving formula, possibly reusing coordinates. Suppose setting a set C of coordinates to zero makes the remaining formula a P10-C disjoint threshold/small-gate tree, and suppose sum_(v in C)p_v<=1/5. Let D={U:F(U)=0}, and assume mu_p(D)>=3/4.

The zero-restricted surrogate is weaker, so its good-event probability is at least that of D. Every discrepancy between surrogate and original must meet C. Singleton generators C cost at most1/5. Applying (1) gives

\[
\boxed{D^{(64)}\text{ is product-measure small}.}
\]

The size and number of occurrences of C are unrestricted. Its total coordinate probability is the load-bearing condition. Deleting a high-cost repeated-variable core does not satisfy this theorem.

## 3. Unbounded rank and nontrivial color obstruction are allowed

A single s-of-n threshold gate is in the class for arbitrary s,n. Its minimal forbidden sets have size s, and the full ground set needs ceil(n/(s-1)) colors when s>=2. Thus the theorem is not merely a theorem about globally64-colorable hypergraphs: taking n>64(s-1) gives a nonempty64-piece obstruction. Small enough independent coordinate probabilities make the good-event hypothesis hold.

The complete transversal example in Phase09 is also in the class: an AND of OR gates on disjoint blocks. It is globally two-colorable. The new exact threshold recursion detects its empty obstruction without estimating its enormous random-monochromatic-edge expectation.

## 4. An explicit family outside the fixed64 expression class

Let G be the path on65 vertices. Its bad function is OR_(uv in E(G))(x_u AND x_v). This function cannot be represented by a read-once tree whose gates are arbitrary monotone gates of arity at most64 or unrestricted unweighted threshold gates.

Proof. Delete irrelevant ports and unary identity gates. At the top node, a singleton minterm port would isolate all witnesses of that child from every other child: in a minimal outer antichain no other minterm contains that singleton port. The original graph is connected, so there can be no such top-level singleton port (unless the entire function has collapsed to one child, already removed).

All minimal witnesses of G have size2. Every remaining outer minterm uses at least two child ports; every used child has a nonempty minterm. Therefore every outer minterm has exactly two ports, and all minterms of every relevant child are singletons. Otherwise composition would produce a minimal original witness of size at least3 on disjoint supports. Each child is consequently the OR of its own coordinates. The graph is a complete blowup of a template on the top-level ports; vertices in the same non-singleton block would be nonadjacent false twins, with identical outside neighborhoods.

The path on65 vertices has no distinct nonadjacent false twins, so all blocks are singletons and the root needs65 ports. It cannot be a permitted arity64 arbitrary gate. A threshold root with only pair minterms must be a 2-of-65 gate and would give a complete graph, not a path. Contradiction.

This is a limitation of the expression class, NOT a counterexample to the discrete conjecture. A path is itself two-colorable and has an empty64-piece obstruction. Earlier graph results already handle arbitrary graphs. The example simply prevents a false assertion that this representation automatically covers every decreasing family.

## 5. General remaining obligation

P10-A works for any outer gate on disjoint children. What is not known here is a universal gate-level cover inequality at transformed weights phi(q_i), without a bound on arity and without threshold structure. Establishing it for every zero-preserving monotone gate would solve a stronger uniform version of the original target. It cannot be assumed in the composition theorem.
