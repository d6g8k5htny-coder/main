# PR-AP-003: Adaptive mixed-radix four-progression-free trees

Date: 2026-09-16. Grade: complete author-side proof with finite computational premises supplied and reproduced by two implementations; external review and novelty audit remain open. This is NOT a solution of Erdos #3 or #142.

## 1. Definition and unconditional progression-freeness

At each node w of a rooted tree select an integer radix b_w>=2 and a set D_w subset {0,...,b_w-1} containing 0. Require that for no a and no nonzero t modulo b_w do all four residues a,a+t,a+2t,a+3t belong to D_w. Repeated residues in a short-period cyclic progression are included in this prohibition. In particular, excluding only four-DISTINCT-residue patterns is a different, weaker condition.

Decode nonnegative integers least-significant digit first. At a node divide n by its radix: n=d+b_w n'; reject if d is not allowed, otherwise move to child d and continue with n'. Stop at quotient zero. Zero is accepted at every node, and additional zero digits do not change membership. Let T_w be the accepted set at node w and T=T_empty. Thus

    T_w = disjoint union_(d in D_w) (d+b_w T_(wd)).

This is a definition of a set of INTEGERS, not merely a formal language. Different root residues cannot overlap. Quotients decrease while positive, so membership terminates.

**Lemma 1.** T and 1+T contain no nontrivial integer 4-AP.

Proof. If four accepted integers form an AP with positive difference Delta, then their root residues form a modular AP. If b does not divide Delta it is forbidden by the local condition. Otherwise all residues coincide, and the four quotients are accepted at the SAME child and form an AP of positive difference Delta/b. Repeating strictly decreases the positive integer difference, so eventually a forbidden split occurs. This does not require prime radices and allows arbitrary history-dependent choices. Translation by 1 preserves the property. QED.

## 2. Pole removal and dynamic programming

For a in (0,1] write

    S_T(a)=sum_(n in T) 1/(n+a)=1/a+B_T(a),
    B_T(a)=sum_(n in T,n>0) 1/(n+a).

B_T extends to a=0. For an action (b,D) define

    h_D(a)=sum_(d in D,d>0)1/(d+a),
    L_(b,D) B(a)=h_D(a)+(1/b)sum_(d in D) B((d+a)/b).

These are the EXACT recursive formulas after removing the common pole. The d=0 pole must not be counted twice. A descendant's shift always remains in [0,1].

When radices belong to {2,...,30}, the finite alphabet certificate gives |D|/b<=2/3; also h_D<=H_29<4. Unfolding the recursion therefore bounds every B_T by 12. Finite-depth regular sums increase to B_T and their omitted contribution is uniformly <=12(2/3)^L. This independently justifies convergence, maximization over child choices, and the absence of a hidden infinite-depth limit assumption.

For every such tree, B_T is decreasing, convex, and 2-Lipschitz on [0,1], because its derivative is -sum_(n>0,n in T)(n+a)^(-2), and sum_(n>=1)n^(-2)<=2. The same monotonicity and Lipschitz bounds hold for a supremum over trees. Convexity also holds for a supremum.

## 3. Canonical actions at bases 11 and 22

Set

    D11=(0,1,2,4,5,7),
    D22=(0,1,2,4,5,7,8,9,14,17).

Complete enumeration shows that every allowed base-11 alphabet has at most 6 members and its i-th smallest digit is >=D11_i. Every allowed base-22 alphabet has at most 10 members and its i-th smallest digit is >=D22_i.

There are respectively 228 and 11,775 admissible alphabets containing zero. The assertions were checked by Python ascending-order enumeration and a separate C++ reverse-order implementation enumerating ALL admissible subsets, not just consuming Python's maximal list. Both canonicals themselves satisfy the modular exclusion.

For any nonnegative decreasing continuation S(a)=1/a+B(a), these rank inequalities imply that D11 and D22 dominate all other alphabets at their respective bases. This comparison is about the actual positive S, with its pole canceled afterwards; it is NOT an unsupported coordinatewise comparison of unrelated regularized functions.

## 4. The two-action value function

Let U(a) be the optimal REGULAR reciprocal value over all adaptive trees using the two canonical actions (11,D11),(22,D22). Then

    U(a)=max{L11 U(a), L22 U(a)}.                    (1)

The map on bounded functions on [0,1] is a sup-norm contraction with constant max(6/11,10/22)=6/11. Its unique bounded fixed point equals the tree supremum: finite-horizon optimization gives the iterates from zero, and the uniform discounted tail tends to zero. Both immediate rewards are <4, hence U<=44/5<9.

An exact optimizing tree exists: at every state choose an action attaining the finite maximum in (1), then recurse. The residual contribution after L steps is at most 9(6/11)^L, proving that the resulting tree has value U, not merely a finite-horizon approximation to it. U is decreasing and convex as a supremum of the tree sums.

Let B0(a)=sum_(n>0,n in C11)(n+a)^(-1) be the regular part of Walker's stationary base-11 construction. L11 B0=B0.

## 5. Certified exclusion of every other radix from 2 through 30

The local tables use exact rational polynomial enclosures for B0, inherited from the centered-moment recurrence in Phase 01 with its complete uniform remainder. The pole-removed recurrence is valid at a=0 as well.

For every b in {2,...,30}\{11,22}, for every allowed D, and every a in [0,1], the supplied tables prove

    B0(a)-L_(b,D)B0(a) >= 1/16.                    (2)

Proof of the computational-to-continuous bridge: each side is the difference of two regular reciprocal sums of integer sets. Each individual derivative belongs to [-2,0], so the DIFFERENCE is 2-Lipschitz, not 4-Lipschitz. Seventeen equally spaced grid points give nearest-point distance <=1/32. Subtracting 1/16 from the exact grid gap therefore covers the entire interval. The smallest resulting table bound is

    67244915254749 / 10^15 > 1/16,

at radix 27. Every inclusion-maximal alphabet was included, and positivity of the weights covers every subset. Totals: 2,345,575 admissible alphabets and 123,057 inclusion-maximal alphabets over all 29 radices. The C++ independent enumeration agrees on every count and every grid maximum.

For the exceptional radix 22 the finer 1/1024 grid, the same Lipschitz argument, and the canonical alphabet's dominance give

    L22 B0(a)-B0(a) <= 1/200                       (3)

uniformly. The actual certified upper produced by that calculation is below 0.004029; the published 1/200 cap is deliberately loose.

Let epsilon=11/1000. Then B0 is a subsolution of (1), and B0+epsilon is a supersolution: the largest possible residual is at most 1/200+(6/11)epsilon=epsilon. Monotone iteration/contraction gives

    B0 <= U <= B0+11/1000.                         (4)

For any other action, positivity and |D|/b<1 imply, using (2),

    L_(b,D)U <= L_(b,D)B0+(|D|/b)epsilon
               <= B0-1/16+epsilon < B0 <= U.

Therefore EVERY other radix is strictly dominated in the full adaptive optimization. Base-11/base-22 noncanonical alphabets are dominated by the rank lemma.

**Theorem 2 (bounded-radix complete optimization).** The supremum of sum_(n in T)1/(n+1) over ALL history-dependent trees with allowed radices 2 through 30 and the stated local modular exclusion equals 1+U(1). It is attained by a tree using only the two canonical actions. No limitation on depth, periodicity, or memory is imposed.

## 6. Certified value

The exact-integer continuous-state enclosure in PR-AP-004 proves

    4422891010185/10^12 <= 1+U(1) <= 4422891978614/10^12.

Equivalently,

    4.422891010185 <= optimum <= 4.422891978614.

These are bounds on the entire infinite-depth class, not on a sampled list of candidate sets. The gap is less than 10^(-6). There is no claim that this is the unrestricted four-AP harmonic supremum.

## 7. Further uniform count/tail consequence

The alphabet certificate also gives |D|^4<=b^3 for every allowed action with 2<=b<=30. Stop each digit path at the first prefix product Q>N. Its product obeys N<Q<=30N. Assign weight Q^(-3/4) to this prefix. The sum of child weights never exceeds the parent weight, since |D|b^(-3/4)<=1. Thus the stopping leaves have total weight <=1 and their count is <=(30N)^(3/4).

Every accepted integer at most N lies in one stopping cylinder, and Q>N permits at most one such integer in that cylinder. Hence

    |T intersect [0,N]| <= (30N)^(3/4), N>=1.

For A=1+T, partial summation consequently gives

    sum_(a in A,a>N) 1/a <= 4*30^(3/4)*N^(-1/4) < 52*N^(-1/4).

This is an independent, much less sharp convergence bound for the entire bounded-radix class. It is NOT a bound on r_4(N) for arbitrary AP-free sets.

## 8. Scope and priority

Walker's base-11 alphabet and stationary sum are prior work (2020); the base-22 alphabet is also published in his later paper arXiv:2203.06045. That paper already has a larger base-55 sum of 4.43975, so this bounded-radix optimum is NOT a record for unrestricted sets. Walker's Theorem 2.1 also establishes the substance of Phase01's digital-reformulation direction. The present mixed-radix construction, bounded-radix Bellman reduction, and quantitative certificate were derived here; external novelty is UNESTABLISHED. The prime-uniform harmonic problem remains open in this work. Composite bases and history dependence do not magically discharge the unbounded-base obligation.
