# P10-C — A fixed-palette, no-dilution theorem for disjoint threshold hierarchies

Author-side complete derivation; external review and historical novelty unresolved. This proves a structural subclass, not Talagrand's unrestricted discrete-convexity conjecture.

## Exact class

A finite rooted monotone formula has pairwise distinct original coordinates as its leaves. Every leaf occurs ONCE in the entire formula. Each internal node is either:

1. an unweighted s-of-n threshold gate, 1<=s<=n, with arbitrary finite n; or
2. an arbitrary zero-preserving monotone Boolean gate of arity at most 64.

The supports of different children are therefore disjoint. Gates of the second type may be specified by their nonempty minimal forbidden subsets of child indices. Constants can be eliminated first; a constant-true child is NOT permitted as an unsimplified leaf because empty-set admissibility is used in the composition proof.

Let F(U) be the root output and D={U:F(U)=0}. Coordinates are independently selected with arbitrary p_v in [0,1]. No gate count, depth, ground-set size, or maximal original witness-rank bound is imposed.

## Main theorem

For K=64 there is a generator cover G of D^(K) with

\[
\boxed{\sum_{g\in G}\prod_{v\in g}p_v
\le\min\{1,-\log\mu_{\mathbf p}(D)\}.} \tag{1}
\]

At mu=0 interpret the right side as one. In particular,

\[
\boxed{\mu_{\mathbf p}(D)\ge3/4\implies
D^{(64)}\text{ is product-measure small, with cost }<3/10.} \tag{2}
\]

A stronger probability range is available:

\[
\boxed{\mu_{\mathbf p}(D)\ge2/3\implies
D^{(64)}\text{ has cost }<5/12<1/2.} \tag{3}
\]

For uniform p these are p-small covers with NO dilution. The elementary strict comparisons use e^(3/10)>4/3 and e^(5/12)>3/2, verified by finite Taylor lower bounds.

## Proof by one invariant through the entire formula

At each node v, let q_v be its original failure probability. Construct a cover of its K-piece obstruction whose cost c_v obeys

\[
c_v\le\phi(q_v),\qquad \phi(q)=\min\{1,-\log(1-q)\}.
\]

At a leaf x, use generator {x}, of cost p_x<=phi(p_x).

At a threshold node, P10-A identifies the obstruction EXACTLY as a threshold m=K(s-1)+1 of child obstructions. Compose their generator covers using all m-child combinations. The summed cost is at most e_m(c_i). If this exceeds one, use the empty generator as a trivial cover of cost one. Monotonicity and P10-B give c_v<=min(1,e_m(phi(q_i)))<=phi(q_v).

For a general monotone gate of arity n<=K, let J be the indices whose singleton input is forbidden. Its K-piece obstruction consists exactly of sets meeting J: any subset avoiding J can be partitioned into at most n allowed singletons. Therefore P10-A says its composed obstruction is the union of child obstructions indexed by J. Its cover cost is at most min(1,sum_(i in J)c_i). Original gate failure contains the event that at least one of those children is active. Consequently

\[
-\log(1-q_v)\ge\sum_{i\in J}-\log(1-q_i)
\ge\sum_{i\in J}\phi(q_i),
\]

when q_v<1, and the capped inequality is immediate when q_v=1. The invariant holds.

Induction on the finite tree proves (1). All nodes use the SAME K colors by P10-A, so neither depth nor original witness rank multiplies the palette. The good-event corollaries follow.

## General bounded-gate version

Replace 64 in the arbitrary-gate arity restriction by b. The same proof works with K=max(64,b), because P10-B holds for every K>=64. The constants are uniform in depth and all threshold fan-ins, but K then depends on b.

## Scope and constructive meaning

The proof produces an explicit finite symbolic generator-family expression using unions, Cartesian unions on disjoint blocks, and an optional empty generator. It may be exponentially large if expanded. No polynomial-time claim is made.

The program computes exact rational upper costs and exact probabilities from finite rational inputs. It does not numerically evaluate logarithms to establish (1); the logarithmic bound is the analytic invariant.

A general monotone Boolean function can be written as one gate of arity |X|, but that does NOT place it in the fixed-64 class. It would replace the constant K by a dimension-dependent bound. Nor can arbitrary formulas be made read-once by copying variables: that changes both the probability law and the coloring identity.

This does not subsume the earlier all-rank theorem, which covers arbitrary rank-bounded hypergraphs with no disjoint-formula structure. The two theorems trade different hypotheses.
