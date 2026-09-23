# P15-C — Connected unbounded-rank, unbounded-scalar-width family with one816-color certificate

Status: author-side exact construction/proof; not an unrestricted theorem, record claim, or historical novelty determination. Large families are specified by formulas and certified symbolically, NOT exhaustively enumerated.

## C1. Construction

Let s>=2, m>=2, r=2s. Arrange b=2m blocks X_0,...,X_(2m-1) on an even cycle. Each block has N>=r vertices. Forbid:

1. every r-subset contained in one block;
2. for each adjacent pair of blocks, every union of an s-subset of each block.

All forbidden sets have size r, hence are minimal. Their vertex-overlap graph is connected. Degrees and number of crossings may be arbitrarily large.

A set U is good EXACTLY when its block counts c_i satisfy

    0<=c_i<=2s-1, and no adjacent pair both has c_i>=s.      (C1)

Its local restriction on each block is the uniform family |U_i|<r. The scalar weights1/r give a valid P14 sandwich of width1. Therefore each local block admits the408-color compatible cover. Give even blocks labels0,...,407 and odd blocks labels408,...,815. Every crossing witness meets opposite palettes. P15-B proves, at all original product probabilities and prices c<=phi(p),

    covercost_c(D^(816)) <= min(1,-log mu_p(D)).            (C2)

Thus mu_p(D)>=3/4 implies a no-dilution816-piece small cover, regardless of m,s,N.

## C2. Exact critical threshold value is unbounded

The maximum cardinality of a good set is

    M = m(3s-2).                                         (C3)

Indeed, the blocks whose counts are at least s form an independent set in a2m-cycle, so at most m are high. Relative to the baseline count s-1 in every block, each high block contributes at most s extra elements. This gives M<=2m(s-1)+ms. Equality is realized by2s-1 selected vertices in every even block and s-1 in every odd block.

Uniform weights1/r are feasible on all forbidden sets, hence alpha<=M/r. For the reverse inequality, put a probability distribution on good sets by choosing which parity is high with probability1/2 and selecting subsets of the prescribed counts uniformly within each block. Every coordinate is then included with probability (3s-2)/(2N).

Put a probability distribution on forbidden sets by choosing a block uniformly and an r-subset in it uniformly. Every coordinate has inclusion probability r/(2mN). Scale that distribution by M/r=m(3s-2)/(2s). Its coordinate loads exactly equal those of the good-set distribution. P15-A's weak-duality certificate gives

    alpha(D)=m(3s-2)/(2s).                               (C4)

Every scalar sandwich therefore needs kappa>alpha(D), which tends to infinity with m. Yet (C2) has the fixed816 palette. This addresses P14's large-width obstruction by an actual structural coloring certificate, not a conjectured universal bound on width.

The earlier normalized resource row cover is also large in the canonical representation: each minimal witness supplies an indicator row normalized by r. The dual weight1/r on every vertex and uniform fractional covering by all local r-edges give tau_*=2mN/r. This is not used in proving(C2).

## C3. Actual816-obstruction is nonempty

Choose

    N=816(r-1)+1.

The full set of one block requires ceil(N/(r-1))=817 good pieces. It therefore belongs to D^(816). The theorem's obstruction is not empty merely because a coarse macro-coloring exists.

On the other hand, the local bad/good flags do NOT determine global goodness. The empty set and a set with exactly s vertices in each of two adjacent blocks both have every local restriction good, but the second is globally bad. Hence the partition used here is not a read-once substitution based solely on each block's local failure bit. No claim is made that no entirely different representation could exist.

## C4. High good probability and large total activation

Set m=2^s and p=1/(100N) on every original coordinate. For a block count Z, factorial Markov gives

    P(Z>=k)<=binom(N,k)p^k<=(Np)^k/k!.

There are b=2^(s+1) internal block tests and b cycle-edge tests. Adjacent block selections are independent (not the overlapping cycle tests themselves). Therefore

    Q <= b * 100^(-2s) * [1/(2s)! + 1/(s!)^2] <1/4        (C5)

for all s>=2: the bracket is at most2, so the displayed bound is at most4*(1/5000)^s<1/4. This is a union bound with an explicit analytic all-s bound; no independence among the b bad events is asserted.

The global expected number of selected coordinates is b/100, which tends to infinity. Let t=100N, so p=1/t. Since (1-1/t)^t<1/2,

    pi0=(1-p)^(bN) <=2^(-floor(b/100)).                   (C6)

For s>=14, floor(2^(s+1)/100)>10(2s-1). This follows at s=14 from327>270 and persists by doubling. Since816<2^10,

    816^(r-1) pi0 <1.                                  (C7)

Thus the older P12 exact-empty-activation sufficient condition fails; its connected-component variant also fails because the witness system is connected. Yet(C2) applies. This is a failure of that sufficient CONDITION, not of its theorem.

For s=14: m=16384, b=32768, r=28, N=22033, |X|=721977344. The exact critical threshold is163840/7. The example is an implicit mathematical object; neither all original vertices nor the forbidden-set family is claimed enumerated.

## C5. Optional finite-size exact probability formula

Let L=P(Bin(N,p)<s), H=P(s<=Bin(N,p)<2s). Then

    mu_p(D)=trace( [[L,H],[L,0]]^(2m) ).                 (C8)

This is the weighted independent-set transfer matrix of the cycle of block states. Each transition contributes the weight of its new state, and a closed walk forbids high-high adjacency. Small instances are checked against full original-coordinate enumeration. The large family uses(C5), not an infeasibly huge exact probability expansion.
