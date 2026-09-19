# P09-F — An explicit limitation of the one-parameter moment reduction

Author-side elementary derivation. This is NOT a counterexample to Talagrand's conjecture or to P09-D. It pinpoints information that the current residual-moment estimate discards.

## Statement

Under only the following hypotheses at one parameter t,

1. an r-uniform hypergraph has hit probability <=1/4;
2. every nonempty proper link has hit probability <=a_r=1-1/r^2;

there is no universal constant C such that its expected selected-edge count is at most C^r for every r.

The counterexamples can even be written with t=p/r^2 and p in (0,1), as in P09-D. HOWEVER they fail P09-D's separate high-probability independence assumption at the LARGER parameter p. That missing larger-parameter condition is exactly the information lost when P09-D merely bounds the full hit probability at t by the one at p.

## Explicit family

Fix m>=1 and r=2^(m+1). Partition X into r disjoint blocks, each of size n=m r^2. Let F consist of all transversals: each edge chooses exactly one vertex from each block. Put

\[
t=1-2^{-1/r^2},\qquad p=r^2t.
\]

Bernoulli's inequality gives (1-1/(2r^2))^(r^2)>=1/2, hence t>=1/(2r^2). Also 1-exp(-x)<x gives p<log 2<1. Thus 1/2<=p<1 and t=p/r^2.

At parameter t a block is nonempty with probability

\[
\theta=1-(1-t)^n=1-2^{-m}.
\]

All r blocks must be hit for a transversal to appear. Their independence gives

\[
P_t(F\text{ is hit})=\theta^r
\le\exp(-r2^{-m})=e^{-2}<1/4.
\]

Any nonempty proper core contained in an edge selects at most one point per block. Its link requires hitting the remaining blocks. The link-hit probability is theta^j for some 1<=j<r, and

\[
\theta^j\le\theta=1-2^{-m}<1-1/r^2=a_r.
\]

Nevertheless the expected selected-edge count is

\[
M_t(F)=n^rt^r=(m r^2t)^r\ge(m/2)^r.
\]

For every fixed C, taking m>2C gives M_t(F)>C^r. This proves the claimed limitation.

## Why the full theorem has more information

At the larger parameter p>=1/2 each block is empty with probability at most 2^(-m r^2). A union bound gives

\[
P_p(F\text{ is hit})\ge1-r2^{-m r^2}>3/4
\]

for every m>=1. Therefore these examples do NOT satisfy mu_p(Ind(F))>=3/4. They cannot refute our new theorem or the original conjecture.

The conclusion is narrower and useful: improving the factorial link-moment bound to a universal exponential bound cannot follow from its one-parameter hypotheses alone. An improved proof must exploit additional structure, such as the original larger-parameter rarity, not merely recompute the same recurrence more accurately.

The family is specified mathematically by its r disjoint blocks. The computer checks representative exact parameter inequalities; it does not enumerate the exponentially many transversal edges at large m, and those checks are not the proof for every m.

## Stronger diagnostic: the full hypothesis still does not rescue uniform random-color averaging

There is a simpler transversal family showing a limitation of the RANDOM-COLOR FIRST-MOMENT STEP itself, even under the full original probability hypothesis.

Fix positive constants L and an integer K>=2. Choose an integer m>2LK, let r=2^(m+1), and take r disjoint blocks, now of size m each. Again let F be every transversal. Use the ORIGINAL sampling parameter p=1/2.

Then

\[
P_p(F\text{ is hit})=(1-2^{-m})^r\le e^{-2}<1/4.
\]

At the P09-D second-sprinkle parameter t=1/(2r^2), every nonempty proper link has hit probability at most the single-block hit probability

\[
1-(1-t)^m\le mt=m/(2r^2)<1-1/r^2=a_r.
\]

No proper core is declared high by that step, so this family remains entirely in the residual. At the target z=p/L, the expected edge mass is

\[
M_z(F)=(m/(2L))^r.
\]

Under an UNCONDITIONED UNIFORM K-coloring, the expected total cost of monochromatic witnesses is exactly

\[
\boxed{K^{1-r}M_z(F)
=K\left(\frac{m}{2LK}\right)^r.}
\]

For fixed K,L this can be arbitrarily large. Therefore the estimate 'there exists a coloring with cost at most its uniform-random average' does not establish a uniform small-cover budget through this numerical upper bound.

Yet F is properly TWO-colorable: color one entire block red and every other block blue. Every transversal meets both colors. Thus F^(2) is empty and the best obstruction cover has cost zero.

This is not a lower bound on the optimal coloring cost. It is an exact counterexample to treating a large uniform-random-color expectation as evidence of a difficult obstruction, and an exact limitation on closing the rank-independent problem by this particular average-cost estimate alone.

The next improvement must exploit structured color choices or a sharper cover representation. It cannot simply interpret our factorial palette as evidence that the instance inherently requires that many colors. The example's parameters are prescribed; no finite experimental extrapolation is used in this all-m argument.
