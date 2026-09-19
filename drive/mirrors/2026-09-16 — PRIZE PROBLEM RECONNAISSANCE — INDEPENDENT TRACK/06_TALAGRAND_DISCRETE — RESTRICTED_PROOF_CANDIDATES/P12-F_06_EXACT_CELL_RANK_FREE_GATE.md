# P12-F — A rank-free compatible gate certificate from exact configurations

Status: complete elementary author-side argument, external review0, historical novelty UNESTABLISHED. It uses exact disjoint configuration probabilities and random coloring, not a new concentration or rounding theorem. The explicit probability condition is load-bearing and is not implied by a large good-event probability.

Let A be a decreasing family on finite I containing the empty set. Separate the singleton minimal forbidden ports J. Let H be the remaining nonempty antichain of minimal forbidden sets, all of size at least r>=2, and let W be the union of its edges. Minimality makes J and W disjoint. Ports outside J union W are irrelevant.

Use independent probabilities q_i. Put

\[
 Q_R=P_q(H\text{ is hit}),\qquad
 \pi_0=\prod_{i\in W}(1-q_i).
\]
If H is empty, the gate only has singleton obstructions and the conclusion below is immediate. Otherwise suppose

\[
 \boxed{K^{r-1}\pi_0\ge1} \tag{F0}
\]
for an integer K>=2. In particular pi_0>0, so every relevant nonsingleton probability is below1.

For prices c_i<=phi(q_i), phi(q)=min(1,-log(1-q)), the K-piece obstruction has a cover satisfying

\[
 \boxed{\operatorname{covercost}_{\mathbf c}(A^{(K)})
 \le\phi\bigl(1-\mu_q(A)\bigr).} \tag{F1}
\]
The number of ports, number of witnesses, maximal witness rank and overlap geometry are unrestricted. F0 depends on actual input probabilities.

## Exact-cell lemma

Write o_i=q_i/(1-q_i) on W. Every exact outcome X_q cap W=e for e in H is a distinct event and implies failure. Therefore

\[
 \pi_0\sum_{e\in H}\prod_{i\in e}o_i
 =\sum_{e\in H}P(X_q\cap W=e)\le Q_R. \tag{F2}
\]
Because -log(1-q)<=q/(1-q), we have c_i<=phi(q_i)<=o_i. Hence

\[
 \sum_{e\in H}\prod_{i\in e}c_i\le Q_R/\pi_0. \tag{F3}
\]
No event-count second-moment estimate or independence of intersecting witness events is used. The events in F2 specify the ENTIRE selected set on W, making them disjoint. Replacing them by 'e is contained in X' would be a different, generally overlapping event.

## Color and cover

Color W with K uniformly independent colors. An edge e is monochromatic with probability K^(1-|e|)<=K^(1-r). Thus some deterministic coloring has monochromatic-edge cost <=Q_R/(K^(r-1)pi_0)<=Q_R. Use its monochromatic minimal edges as generators. Any set avoiding those generators is properly K-colorable by restricting that fixed coloring, so the generators cover the nonsingleton obstruction.

Add singleton generators for J. Their cost is <=sum_{j in J}-log(1-q_j). Independence on the disjoint coordinate sets J and W gives

\[
 \mu_q(A)=\prod_{j\in J}(1-q_j)(1-Q_R).
\]
Therefore the uncapped cover cost is at most

\[
 \sum_{j\in J}-\log(1-q_j)+Q_R
 \le-\log\mu_q(A).
\]
If needed, use the empty generator as a trivial cover of cost1. This proves F1. If a singleton port has q_j=1 then total failure=1 and that trivial cover suffices. Zero prices and zero probabilities cause no problem; zero-cost generators must not be discarded merely because they have zero weight.

## Fixed-palette hierarchy consequence

Fix K>=128. In addition to Phase11-compatible thresholds/graphs (and rank-three gates if K>=368640), allow an arbitrary-arity, arbitrary-rank gate whenever F0 holds for its ORIGINAL independent child-failure probabilities, after removing singleton and irrelevant ports. Common-palette substitution proves the same phi invariant at every node, at all finite depths on disjoint original supports.

A simple rank-independent sufficient condition is pi_0>=1/K, using r>=2. For K=128, a gate may have arbitrarily many inputs and arbitrary witness geometry as long as the probability of no relevant nonsingleton input being active is at least1/128. When the minimum witness size is larger, F0 is correspondingly weaker.

At nodes with original failure Q>2/3 the trivial cap phi(Q)=1 already works, so F0 need only be checked where the nontrivial bound is used. It is not imposed retroactively on graph/threshold gates already justified by their separate theorems.

## Nonvacuity and limitation

Take arbitrary r>=2, n=Kr+1, and forbid every r-subset of n ports except one. Maximum allowed set size is r, so the full set cannot be K-covered. The gate is not an ordinary unweighted threshold, because one r-set is allowed while the other r-sets are forbidden. Let q_i=1/(100n). Then pi_0>=1-sum_i q_i=99/100 and the original hit probability is <=binom(n,r)q^r <=(1/100)^r/r!, so the new criterion applies with ample probability margin. This family is specified mathematically, not enumerated at arbitrary rank.

Conversely, many disjoint edges at small probabilities can have high independence probability but very small pi_0. They are easily2-colorable, yet F0 can fail. Thus F0 is sufficient, not necessary and not a universal decomposition principle.

This result does not solve the unrestricted conjecture. The remaining issue includes arbitrary high-rank gates at large aggregate activation that do not meet F0 and lack another compatible structural argument.

For an exact failure of the automatic-premise inference at K=128, take128 disjoint graph edges and q_i=1/32 on all256 endpoints. Failure probability is <=128/32^2=1/8, whereas pi_0=(31/32)^256<1/128, certified by the integer comparison 128*31^256<32^256. The graph is properly2-colorable. Thus even good-event probability at least7/8 does not imply F0; failing F0 is not failing the desired conclusion.

## Componentwise sharpening: global empty mass is unnecessary

Let W_1,...,W_m be the connected components of the witness-overlap graph on relevant ports (two ports are adjacent when they occur in the same nonsingleton minimal witness). Every witness belongs wholly to one component. Let H_j be its component family, r_j its minimum edge size, pi_j=product_(i in W_j)(1-q_i), and Q_j=P(H_j hit).

It is enough to require

\[
 \boxed{K^{r_j-1}\pi_j\ge1\quad\text{for each component }j.} \tag{F4}
\]

Apply the exact-cell construction independently to each component, using the SAME K color names and taking the union of their generator families. A nonsingleton obstruction can occur only if some component is not K-colorable, since disconnected component colorings combine without extra colors. Component j's generator cost is at most Q_j. Independence of disjoint coordinate supports gives

\[
 \sum_j Q_j\le\sum_j-\log(1-Q_j)
 =-\log\prod_j(1-Q_j).
\]

Adding singleton hazards and capping at one yields (F1) again. The number of components and the total aggregate activation may now be arbitrarily large. In particular the 128 disjoint-edge example fails the global condition F0 but satisfies F4 for K=128 (indeed already for K=2 at the stated probabilities). This is a scope extension by exact decomposition, not an assertion that connected high-activation components can always be split.

The fixed-palette hierarchy consequence remains valid with F4 in place of F0 at any admitted gate. The unresolved case is thus a genuinely connected high-activation witness system that lacks another established compatible primitive or a cheap exceptional cover.
