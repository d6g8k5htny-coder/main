# PR-AP-002 — Adaptive base-11 digit-rank compression

Date: 2026-09-16. Grade: complete author-side proof, conditional only on a fully supplied 1024-case finite certificate checked by two code paths. No external independent review. The canonical alphabet itself is Walker's published 2020 construction; priority of the adaptive extension is unestablished.

## Exact class

Read base-11 digits from LEAST significant to most significant. At every reachable finite digit prefix w, choose an alphabet D_w subset {0,...,10} containing 0 and containing no nonconstant modular 4-AP. The alphabet may depend arbitrarily on the entire previous prefix and depth. Let T be the nonnegative integers whose finite base-11 expansion follows the allowed tree. Containment of 0 permits arbitrary leading-zero extension, so membership does not depend on how many leading zeros are written. This is a specified subclass of 4-AP-free sets, not all such sets.

Let D*=(0,1,2,4,5,7), in increasing order, and C*=C(11,D*).

## Finite domination lemma

Every allowed alphabet D, ordered d_0<...<d_(s-1), has s<=6 and d_i>=D*_i for every i<s.

The supplied certificate covers all 2^10=1024 subsets containing 0. Each subset either contains an explicit modular 4-AP witness or lies in one of these twelve admissible maximal alphabets:

(0,1,2,4,5,7)
(0,1,2,6,8,9)
(0,1,3,4,5,9)
(0,1,3,4,6,10)
(0,1,3,7,8,9)
(0,1,5,7,8,10)
(0,2,3,4,8,10)
(0,2,3,5,6,7)
(0,2,3,5,9,10)
(0,2,6,7,8,10)
(0,4,5,6,8,9)
(0,4,6,7,9,10).

Every listed alphabet is coordinatewise at least D*. Any subset of one has its i-th smallest member at least that of the containing alphabet. The exhaustive certificate therefore proves the lemma. There are 228 admissible subsets; the other 796 cases contain their explicit forbidden witness. This certificate is finite and exact: no floating point or random search establishes completeness.

## Theorem: one canonical tree dominates all adaptive trees in this class

There exists an injective map phi:T -> C* with phi(n)<=n for every n. Consequently, for every X>=0,

|T intersect [0,X]| <= |C* intersect [0,X]|,

and for every nonnegative nonincreasing weight w on the nonnegative integers,

sum_(n in T) w(n) <= sum_(c in C*) w(c),

where infinite values are allowed.

Proof. For n in T, traverse its digits from least to most significant. If its digit at a node has rank i in that node's ordered alphabet, replace it by D*_i. Each digit weakly decreases by the finite domination lemma. Hence phi(n)<=n. Different accepted words have different rank words: given a rank word, the original word is reconstructed successively using the fixed tree's alphabet at the reconstructed prefix. Thus phi is injective. A zero digit has rank zero and is mapped to zero, so padding by leading zeros does not create a representation ambiguity. The counting inequality follows by injecting T intersect [0,X] into C* intersect [0,X]. For weights, w(n)<=w(phi(n)); sum over the injection and then over C*. QED.

## AP-freeness for adaptive trees

Suppose an integer 4-AP lies in T, with difference d>0. Let t be the largest integer with 11^t dividing d. All four numbers share the same t least-significant digits. At that common node, their next digits form a nonconstant modular 4-AP, contrary to the local rule. Padding with leading zeros is permitted, so numbers of different digit lengths cause no exception. Translation by 1 preserves AP-freeness.

## Exact harmonic corollary

Taking w(n)=1/(n+1) yields, for every such infinite adaptive tree,

sum_(n in T)1/(n+1) <= S(11,D*)
 <= 4421747532398232527 / 10^18.

Equality is attained by taking D_w=D* at every node. Thus this closes the optimization over the ENTIRE infinite adaptive class at base 11, not just a depth-limited search.

The proof also works for any other base and hereditary alphabet family that possesses a coordinatewise dominant canonical alphabet. The displayed finite certificate verifies that prerequisite only for the stated base-11 family.

## Why this does not solve Erdős #3

The local modular digit restriction is stronger than integer AP-freeness. For example, the integer set {1,2,3,5,6,8,9} is 4-AP-free, but its shifted units alphabet {0,1,2,4,5,7,8} contains the modular progression 0,4,8,1 modulo 11. It lies outside the class above.

Moreover, the exact reformulation of PR-AP-001 ranges over unbounded primes. A uniform bound across that family is not established. This theorem tells us precisely that more adaptive branching WITHIN the present base-11 local rule cannot improve this construction. It does not prohibit a different prime, a less restrictive digit grammar, or a non-digital argument.

## Priority boundary

Walker (2020) already supplied D* and its approximate harmonic sum. The search here rediscovered it, and that rediscovery is recorded, not advertised as new. The all-depth adaptive dominance statement was derived in this session; a narrow web search found no specific matching statement, but that is not a novelty certificate. Obtain specialist review and a broader priority search before treating it as a new publishable theorem.
