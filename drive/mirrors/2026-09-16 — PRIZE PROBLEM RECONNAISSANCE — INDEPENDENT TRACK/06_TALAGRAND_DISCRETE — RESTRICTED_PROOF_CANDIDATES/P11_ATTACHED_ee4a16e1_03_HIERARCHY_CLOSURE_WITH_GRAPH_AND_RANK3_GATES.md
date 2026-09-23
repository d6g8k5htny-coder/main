# P11-C — Common-palette closure for graph and rank-three gates

Status: complete local author-side derivation using P10-A/B plus P11-A/B; external correctness review and priority review pending. Not a theorem about arbitrary repeated-variable circuits.

## 1. Exact common-palette identity (restated)

Let X_i be pairwise disjoint supports; D_i downsets containing the empty set; A an outer downset on child indices. Let D consist of U whose active-child set {i:U intersect X_i notin D_i} belongs to A.

For J_k(U)={i:U intersect X_i belongs to D_i^(k)},
\[
 U\in D_{(k)}\iff J_k(U)\in A_{(k)}.               (C1)
\]
Forwards: each hard child must be active in at least one of the k parent parts. Reverse: assign each hard child's whole block to one admissible outer color, while partitioning every soft child into k child-admissible parts. Supports are disjoint, so choices combine without conflict. This reuses the same k colors at every node.

If G_i covers child D_i^(k) with cost c_i and F covers A^(k) at prices c_i, replacing each index generator by all disjoint unions of corresponding child generators gives a parent cover of cost at most sum_(J in F) product_(i in J)c_i. Duplicated descriptions can only increase the displayed upper bound.

## 2. Gate compatibility

Call an outer gate K-compatible when, for EVERY independent input vector q, its K-piece obstruction has a cover at prices phi(q_i) costing at most phi(Q), Q=gate failure probability. Any lower child costs c_i<=phi(q_i) also work by positivity.

- Every threshold gate of arbitrary arity is K-compatible for K>=64, by P10-B's scalar elementary-symmetric inequality.
- Every arbitrary zero-preserving gate of arity<=K is K-compatible: only singleton-forbidden ports can be K-hard. The sum of their hazards is at most the parent hazard.
- Every rank<=2 gate of arbitrary arity is K-compatible for K>=128, by P11-A plus exact singleton hazard addition.
- Every rank<=3 gate of arbitrary arity is K-compatible for K>=368640, by P11-B.

K-compatibility is monotone in K because A^(K') subset A^(K) for K'>=K. No new randomization or palette multiplication is needed to pass to larger K.

## 3. Two complete hierarchy theorems

Let a finite rooted formula have each original coordinate as a leaf EXACTLY ONCE. Children therefore have disjoint supports. All gates are monotone and zero-preserving; constants may first be simplified with exact accounting.

**Graph hierarchy theorem.** Permit arbitrary-arity threshold gates, arbitrary-arity rank<=2 gates, and arbitrary monotone gates of arity<=128. At independent original probabilities p, if D is the root-good family, then
\[
 \operatorname{covercost}_{\mathbf p}(D^{(128)})
 \le\min\{1,-\log\mu_{\mathbf p}(D)\}.             (C2)
\]

**Rank-three hierarchy theorem.** Permit arbitrary-arity threshold gates, arbitrary-arity rank<=3 gates, and arbitrary monotone gates of arity<=368640. Then
\[
 \operatorname{covercost}_{\mathbf p}(D^{(368640)})
 \le\min\{1,-\log\mu_{\mathbf p}(D)\}.             (C3)
\]

Proof. At a leaf x use {x}, cost p_x<=phi(p_x). At each internal node, independent disjoint child supports give its exact product-vector failure probability. By induction the child cover costs are<=phi(q_i). Gate compatibility and(C1) compose those covers with cost<=phi(Q), using the same global K. Induct on the finite tree. Depth, arity of permitted unbounded gates, ground-set size and original witness rank never enter K. QED.

At good probability>=3/4 both conclusions give cost<=log(4/3)<3/10. At good probability>=2/3 they give cost<=log(3/2)<5/12. Uniform p needs NO dilution.

The graph theorem uses a smaller palette on a smaller class. The rank-three theorem strictly includes arbitrary local pair/triple overlaps, but original coordinates still cannot be shared between sibling subtrees. The numerical increase128 to368640 is not advertised as an optimization.

## 4. Cheap exceptional-cover stability

For D subset D0 with D0\D covered by R of product cost<=beta,
\[
 D^{(K)}\subseteq D0^{(K)}\cup\langle R\rangle.
\]
An avoided generator cannot appear in any subset of an avoided set; hence a D0-decomposition of that set is also a D-decomposition. Thus any theorem above extends with additive beta, and beta<=1/5 is sufficient at good probability>=3/4.

This applies to repeated-variable coordinates if setting a core C to zero leaves a permitted hierarchy AND sum_(v in C)p_v<=1/5. No assertion that a cheap such core exists universally is made.

## 5. New coverage and nonvacuity

The65-vertex path-edge gate that P10-D excluded is now one permitted rank-two gate. Complete graph gates on arbitrary n are also permitted. For n=129, the full ground set has chi129 and is a NONEMPTY128-piece obstruction. Taking all p_v=1/n^2 gives failure probability<=binom(n,2)p^2<1/4, so(C2) is not only an empty-obstruction theorem.

Replacing each vertex by a disjoint AND of m fresh leaves produces minimal original witnesses of size2m, with m arbitrary. The same128 palette still applies. Alternating these with arbitrary-fan-in thresholds gives unbounded depth and unbounded original rank.

## 6. Boundary

Any Boolean function has a one-gate description, but an arbitrary large-rank/high-arity gate is not automatically in this class. Copying repeated variables into distinct leaves changes the probability law. No result here turns every hypergraph into a disjoint hierarchy at bounded cost. The unresolved general gate or overlap-compression problem remains the prize-level interface.
