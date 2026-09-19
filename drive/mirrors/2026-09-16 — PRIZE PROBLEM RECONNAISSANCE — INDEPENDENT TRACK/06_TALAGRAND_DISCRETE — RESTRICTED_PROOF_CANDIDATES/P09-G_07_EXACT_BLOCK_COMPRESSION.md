# P09-G — Exact block compression and a conditional cover-transfer lemma

Author-side elementary derivation; no historical novelty claim. This gives a structured-coloring interface suggested by P09-F. It does not assert that every witness hypergraph has a nontrivial block representation.

## 1. Complete occupancy lift

Partition X into nonempty disjoint blocks B_i indexed by finite I. Let A be a decreasing family on I and define

\[
\pi(U)=\{i:U\cap B_i\ne\varnothing\},\qquad
D=\{U\subseteq X:\pi(U)\in A\}.
\]

Equivalently, every minimal forbidden set S of A is lifted to ALL choices of one point from each block indexed by S. This full Cartesian-product condition is essential.

For every integer k>=1,

\[
\boxed{U\in D_{(k)}\iff\pi(U)\in A_{(k)}.}       \tag{1}
\]

If U is covered by k members of D, their occupied-block sets cover pi(U) by members of A. Conversely, if pi(U) is covered by k sets in A, assign each occupied block to one covering set and put all of U's points from that block in the corresponding part. Decreasingness ensures each part belongs to D.

Under independent coordinate probabilities p_v, the occupied-block indicators are independent with parameters

\[
q_i=1-\prod_{v\in B_i}(1-p_v),\qquad
\mu_{\mathbf p}(D)=\mu_{\mathbf q}(A).           \tag{2}
\]

These are exact identities, not an independence approximation to overlapping blocks.

## 2. Exact lifting of a generator cover

For a generator S subset I, lift it to all sets choosing one point from each block B_i, i in S. Under a target vector z, their total cost is

\[
\prod_{i\in S}\left(\sum_{v\in B_i}z_v\right). \tag{3}
\]

A template cover of A^(k) lifts, by (1), to a cover of D^(k). Multiple lifted descriptions may duplicate generators; removing duplicates only decreases cost.

The factor in (3) is a SUM of coordinate probabilities, not q_i. Replacing it by q_i without an inequality is wrong.

## 3. Light-block transfer

If q_i<=1/2, then

\[
\sum_{v\in B_i}p_v
\le-\log(1-q_i)
\le\frac{q_i}{1-q_i}\le2q_i.
\]

Hence any cover at template vector q/L transfers at coordinate vector p/(2L), with no increase in cost. This is uniform in block sizes.

## 4. Heavy blocks cost two additional pieces, not an uncontrolled cover sum

Suppose mu_q(A)>=3/4 and put H={i:q_i>1/2}, J=I\H. The restrictions A_H and A_J are decreasing and each has its own good-event probability at least 3/4; removing other coordinates only makes membership easier.

On H, monotonicity gives mu_(1/2)(A_H)>=3/4. If R is a uniform half-random subset of H, both R and H\R have probability at least 3/4 of belonging to A_H. A union bound gives positive probability that both belong. Thus there is a fixed partition H=H_1 disjoint-union H_2 with both H_i in A.

The full unions of blocks over H_1 and H_2 belong to D. Therefore the heavy part of ANY U can always be covered by these two allowed pieces.

If a template cover of A_J^(k) at q_J/L has cost at most beta, its lift covers D^(k+2) at p/(2L) with cost at most beta. Explicitly,

\[
\boxed{
\mu_{\mathbf p}(D)\ge3/4,
\quad A_J^{(k)}\text{ has a }(\mathbf q_J/L)\text{-cover of cost }\le\beta
\ \Longrightarrow\
D^{(k+2)}\text{ has a }(\mathbf p/(2L))\text{-cover of cost }\le\beta.}
\]

The assumption on the template cover is not automatic. If a class of template families has a proven uniform cover theorem and is closed under coordinate restriction, this lemma transfers it to their full occupancy lifts, at the cost of two pieces and a factor-two dilution.

## 5. Role in the rank-independent problem

For the transversal example in P09-F, the template has a single forbidden set I. It is properly two-colorable, and (1) explains why all vertex-level transversals can be handled by coloring whole blocks together.

This supplies one precise way for structured coloring to beat the independent uniform-color first-moment bound. It does not prove that arbitrary high-rank witnesses admit such a template, nor that a bounded number of occupancy lifts covers every instance. Those would be genuinely new structural obligations.
