# P15 successor: finite-rank complete-transversal macro composition

Target: the higher-rank macro-template interface explicitly left open in
`P15_NEXT_WORK.md`, item 5. This is a numbered author-side successor, not a
replacement of P15-D, a novelty determination or an unrestricted prize result.
It consumes the exact P14-A scalar theorem and the phase-qualified
`P11_ATTACHED_ee4a16e1` primitive theorems recorded in `SOURCES.json`.
Those are existing author-side proofs; the finite companion below does not
constitute independent review or a formal verification of their analytic proofs.

## 1. Objects and theorem

Partition a finite original coordinate set X into disjoint nonempty blocks X_i.
Let A_i be decreasing families on X_i. Let H be a finite macro hypergraph on
the block indices, with every edge of size at most a fixed integer r>=3.
Define the actual global decreasing family

    D = {U: U intersect X_i belongs to A_i for every i,
            Occ(U) = {i: U intersect X_i is nonempty} contains no H-edge}.

Thus every macro edge is lifted to ALL choices of one original coordinate in
each of its blocks. Incomplete lifting is not included. The local and macro
conditions share the original coordinates; they are not independent events.

For a decreasing family A, write Ob_K(A) for the sets that cannot be partitioned
into at most K members of A. A generator family G covers an event when every
set in that event contains some g in G. At nonnegative coordinate prices z,
its cost is sum over g in G of product over v in g of z_v. The empty generator
has cost one and covers every set; an empty generator FAMILY has cost zero
and covers no set. Duplicate descriptions may be removed, decreasing cost.

Let J be the genuine original singleton forbidders of D. Restrict them absent,
delete any resulting empty block, and delete every macro edge incident to a
deleted block. Do NOT shrink such edges. Suppose each remaining actual local
restriction has a P14-A strict scalar sandwich of width at most kappa>=1.
This means there are 0<=a_v<1 with, for every subset V of that block,

    sum_V a_v < 1  implies V is locally good;
    V locally good implies sum_V a_v < kappa.

Set L=ceil(408*kappa). Let H_r(A) be the exact finite palette recurrence in
section 2 of the attached P11-D source; its parameter r is MACRO rank, not
the possibly unbounded rank of the original local witnesses. Then

    K = L * H_r(2)

satisfies, for every original independent product vector p and every
0<=z_v<=phi(p_v), with phi(t)=min(1,-log(1-t)) and phi(1)=1,

    covercost_z(Ob_K(D)) <= phi(1-mu_p(D)).                 (T)

For macro rank at most three, the sharper attached P11-B proof gives

    K = 737280 * L.                                      (T3)

For a singleton-free system with Q=1-mu_p(D)<=2/3, the constructed rank-three
cover has uncapped cost at most (8369/9216)*[-log(mu_p(D))]. This finer
coefficient is NOT asserted after adding arbitrary singleton hazards, and
does not describe the high-Q cost-one fallback. The preceding graph-only
P15-D theorem retains its smaller 256*L palette; this successor does not
replace that bound by H_2(2)*L or claim a palette independent of macro rank.

## 2. Exact imported primitive interfaces

P14-A proves that, with singleton-admissible local scalar width <=kappa,
local failure probability Q_i<=2/3 and prices z_v<=2p_v, there is a cover
G_i of Ob_L(A_i) with cost at most Q_i/2. All its setwise hypotheses, strict
endpoints and zero-probability configurations are retained.

Attached P11-D proves that, for any finite hypergraph with edge sizes in
{2,...,r}, product vector q, failure probability Q_H<=e_r=(2r+1)/(3r),
and prices c_i<=A*q_i, there is a fixed H_r(A)-coloring certificate whose
generator sizes are in {2,...,r} and whose total cost is at most Q_H/2.
Every monochromatic macro edge contains a generator of the certificate.
We use A=2 and Q_H<=2/3<e_r. We require this stronger low-Q interface;
merely consuming the all-probability phi(Q_H) conclusion would not leave
the budget for the local covers.

For rank three, attached P11-B constructs a 368640-coloring certificate with
generator sizes two or three and cost <=(3761/4608)*Q_H under Q_H<=2/3 and
c_i<=2q_i. Refine that MACRO coloring by a second independent two-color
label. Keep a generator only when all its vertices receive one refining
label. Every macro edge monochromatic in the refined coloring still contains
a retained generator. A generator of size s>=2 survives with probability
2^(1-s)<=1/2. Averaging therefore supplies a deterministic refinement using
737280 colors and cost <=(3761/9216)*Q_H. This refinement occurs BEFORE
occupancy lifting: lifted generators can have size one or zero, for which
the asserted cost-halving operation would not be valid.

## 3. Original occupancy law and transformed prices

First assume D contains the empty set and all original singletons, and
0<Q<=2/3. Let Q_i=P(U_i not in A_i), and put H_global=-log(mu_p(D)).
Every local failure is a global failure, so Q_i<=Q. Disjoint original blocks
make the LOCAL good events independent, hence

    mu_p(D) <= product_i(1-Q_i),
    sum_i Q_i <= sum_i[-log(1-Q_i)] <= H_global.           (1)

The first inequality is generally strict. We do not multiply a local-good
probability by a macro-good probability: those conditions share coordinates.

The occupancy indicators are independent across the original blocks, with

    q_i = 1 - product_{v in X_i}(1-p_v).

Use as an occupancy cover C_i all singleton coordinates of the block when
sum z_v<=1, and otherwise the family containing the empty generator. Its
actual description cost is c_i=min(1,sum z_v). It covers EVERY nonempty
subset of its block, including zero-probability subsets. In particular,
zero-cost singletons cannot be deleted just because their costs vanish.
Since z_v<=phi(p_v)<=-log(1-p_v),

    c_i <= min(1, sum_v[-log(1-p_v)]) = phi(q_i) <= 2q_i. (2)

If a p_v=1, the first logarithmic sum is infinite, but c_i<=1=phi(q_i)
still proves (2). To see phi(t)<=2t, for t<=1/2 use
-log(1-t)<=t/(1-t)<=2t; for t>=1/2 use phi(t)<=1<=2t.

Let Q_H be the probability that the occupied macro indices contain an H-edge.
Complete transversal lifting makes this an actual cause of global failure,
so Q_H<=Q. Therefore Q_H<=H_global as well. This implication is false for
arbitrary incomplete cross-block lifting, which remains outside the theorem.

## 4. Setwise lifting and the product palette

If Q_H=0, use all original macro edges as zero-cost generators: every such
edge has product q_i zero, so (2) makes its product c_i zero. A constant
macro coloring is then a valid certificate, and both advertised palettes
can accommodate it. Retain these generators even if the global Q is positive
because of a local failure. This is the zero-Q repair explicitly supplied
in the pinned P11-D addendum and avoids any use of a strict 0<0 argument.
For Q_H>0, apply the chosen macro primitive at q and the ACTUAL costs c
from (2). For
each macro generator g, form all unions of one occupancy generator from C_i
for every i in g. Disjoint block supports give total indexed description
cost exactly product_{i in g}c_i. Removing duplicate lifted unions only
decreases this cost. Empty occupancy generators are legitimate: their
contribution is one. No occupancy probability is substituted for c_i.

If an original U avoids the lifted cover, Occ(U) avoids every macro
generator. Indeed, if g were contained in Occ(U), the occupancy-cover
property would provide a generator h_i subset U_i for each i in g; their
union would be a lifted generator contained in U, a contradiction.
Consequently Occ(U) is properly M-colorable by the macro certificate.

If U also avoids all G_i, each U_i is partitionable into at most L members
of A_i. Pair its local part label with its block's macro color. Every one
of the resulting at most M*L original parts is locally good, and its
occupied macro indices contain no macro edge. Thus it belongs to D.
The union of local and lifted macro generators covers Ob_(M*L)(D).

This product palette is essential. The fixed local labels and macro labels
cannot in general be replaced by their maximum. No duplicated original
coordinates or independent clones occur in the construction.

## 5. Dependence budget and endpoint completion

For the P11-D finite-rank primitive, the union cover costs at most

    (1/2)*sum_i Q_i + (1/2)*Q_H <= H_global.              (3)

For the refined rank-three primitive, its cost is at most

    (1/2)*sum_i Q_i + (3761/9216)*Q_H
        <= (8369/9216)*H_global.                         (4)

These inequalities use two separate upper bounds by the same true global
hazard. They do not assume local and macro failures are independent, and
they do not claim their probabilities add to Q. Cap the cost at one by
substituting the empty-generator cover if that improves the bound. If
Q>2/3, e<3 gives -log(1-Q)>1, so the cost-one cover proves (T) directly.
The same cap is available whenever the uncapped total exceeds one.

If Q=0, every original minimal forbidden set has product probability zero,
and hence product price zero because p_v=0 forces z_v=0. Keep these
zero-cost generators: they cover D^c and therefore its K-obstruction.
If there are no forbidden sets, use the empty generator family. If an
empty macro edge or an empty local forbidden witness makes D empty, Q=1
and the empty-generator cover proves the theorem without a local primitive.

For original singleton forbidders J, global goodness requires all of them
absent. This includes every coordinate in any macro-singleton block. Set
them absent and restrict each local family. Restriction preserves a scalar
sandwich by simply restricting its coefficients and the same kappa.
If a block becomes empty, delete it; a macro edge incident to it can never
be fully occupied and is deleted in its entirety. Shrinking it would create
a different, generally stronger bad event. Remaining macro edges retain
complete lifting and rank at most r. No genuine singleton remains.

Let Q_R denote the remaining system's failure probability. Independence of
the ORIGINAL removed and remaining coordinates gives exactly

    1-Q = product_{v in J}(1-p_v) * (1-Q_R).

Add the singleton generators {v}, cost z_v<=phi(p_v), to the remainder
cover. Before capping their total cost is at most

    sum_{v in J}[-log(1-p_v)] - log(1-Q_R) = -log(1-Q).

If a removed singleton has p_v=1, Q=1 and the trivial case applies. If the
remainder is empty of coordinates, its good probability is one and its
cover cost zero. This proves (T)/(T3) with the same K, while (4) remains
specifically a singleton-free low-Q statement.

## 6. What is closed here, and what is still outside the result

The new analytic implication supplies the transformed occupancy-cover and
shared-coordinate dependence budget requested by P15_NEXT_WORK item 5,
for every FIXED finite macro rank. Because it holds for all original product
probabilities and all prices z<=phi(p), the resulting family is itself a
hazard-compatible primitive for disjoint-support substitution, preserving
the existing source's substitution requirements. Repeatedly multiplying
block palettes at arbitrary shared-variable nodes is not licensed.

There is no universal discovery/decomposition theorem, no bounded-width
claim for arbitrary downsets, no incomplete-lifting theorem, and no palette
independent of r. Original prize problems solved remain zero. No q0 claim,
canonical scientific status, independence requirement or publication grade
is changed. Historical novelty is unestablished. Source proof review,
finite companion checks and same-provider cross-review are separate facts.

The companion checks exact finite setwise composition, probability directions,
occupancy prices, singleton restriction, palette arithmetic and endpoint
regressions. It does not enumerate every ground-set size or prove the
imported analytic primitives by sampling. No executable source archive is
run; all selected source bodies are read and hashed as data.
