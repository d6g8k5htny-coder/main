# P10-A — Exact common-palette substitution

Date: 2026-09-16 America/Chicago. Author-side derivation; no external review or claim of historical novelty. No numerical premise is used.

## Setting

Let the finite sets X_i be pairwise disjoint. Let D_i be decreasing families on X_i with empty set in D_i. Let A be a decreasing family on the index set I, also containing the empty set. Define

\[
J(U)=\{i:U\cap X_i\notin D_i\},\qquad
D=\{U\subseteq\bigcup_i X_i:J(U)\in A\}.
\]

For a decreasing family B let B_(k) be unions of k of its members and B^(k) its complement. Unions can always be made into partitions by deleting repeated elements; empty parts are allowed.

## The exact identity

For every k>=1, put J_k(U)={i:U intersect X_i belongs to D_i^(k)}. Then

\[
\boxed{U\in D_{(k)}\iff J_k(U)\in A_{(k)}.} \tag{1}
\]

Proof, forward. Partition U into k sets V_c in D. For each c, its active index set J(V_c) belongs to A. If i belongs to J_k(U), the k parts V_c intersect X_i cannot all belong to D_i, so i is active in at least one c. Thus J_k(U) is a subset of the union of k members J(V_c) of A. Decreasingness gives J_k(U) in A_(k).

Proof, reverse. Partition J_k(U) into k sets I_c in A. For each hard block i in I_c place all of U intersect X_i into color c. For each soft block i outside J_k(U), choose a k-part partition of U intersect X_i with every part in D_i. Combine blockwise. In color c only hard blocks assigned c can be active: all soft blocks have been made inactive in EVERY color. Hence the active index set of color c is a subset of I_c and belongs to A. All k combined parts belong to D. QED.

This uses ONE palette of k colors, not a product of palettes. Neither child rank nor number of children appears in (1). The supports' disjointness is essential.

## Threshold gate specialization

If A permits at most s-1 active children, then A_(k) permits at most k(s-1). Consequently

\[
\boxed{D^{(k)}=\{U:|J_k(U)|\ge k(s-1)+1\}.} \tag{2}
\]

If k(s-1)>=|I| the obstruction is empty. This is exact, not a bound obtained from random coloring.

## Exact cover composition

Suppose G_i covers D_i^(k) and has product-weight cost at most c_i. Suppose F covers A^(k). For every J in F, take all unions of one generator from each G_i, i in J. By (1), these unions cover D^(k). Disjoint supports imply that each union's weight is the product of the chosen generators' weights. Thus their cost, counting duplicate descriptions if present, is at most

\[
\boxed{\sum_{J\in F}\prod_{i\in J}c_i.} \tag{3}
\]

For a threshold gate use all m=k(s-1)+1 subsets of children, obtaining the elementary symmetric bound e_m(c_1,...,c_n). For m>n the bound is zero. The empty generator is permitted as an intermediate trivial cover of cost one, so every node can cap its cost at one.

Probability is a separate identity: under independent original coordinates, active child events on disjoint supports are independent with probabilities q_i=P(X_i,p notin D_i). The parent's failure probability is the probability of A^c at this product vector q. Equation (3) uses COVER COSTS c_i, not occupancy probabilities q_i. They are not interchangeable.

## Explicit failed generalization

Reuse the same variable x in two children, and use a 2-of-2 gate. The resulting bad function is x, so {x} is impossible to cover by any number of good parts. Each child is 2-hard at {x}, but a formal disjoint-child count would predict two hard children fit into two colors of capacity one, incorrectly declaring success. The fake independent failure probability is p^2; the actual probability is p. This counterexample is a required regression.
