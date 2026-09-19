# PR-AP-012 — Separated extremal blocks, a canonical test set, and attainment

2026-09-16. Complete author-side derivation; no external review or novelty claim. The separated-extremal-block idea has an explicit prior appearance in Will Sawin's MathOverflow comments of 14 March 2025, using base 4. Gerver (1977) already proves the global bounded-harmonic reformulation. The statements below give a self-contained same-k base-3 argument, a factor-six comparison, and compactness consequences. Their novelty is UNESTABLISHED; reconstructing an existing organizing idea is not a prize-problem solution.

## 1. Exact definitions

Fix k>=3. A set is k-free if it contains no nontrivial k-term integer arithmetic progression. Let

    r_k(N) = max{|A| : A subset {1,...,N} is k-free},
    M_k = sup{sum_(a in A)1/a : A subset positive integers is k-free},
    I_k = integral_1^infinity r_k(floor x) dx/x^2
        = sum_(n>=1) r_k(n)/(n(n+1)).

All suprema, sums, and integrals initially take values in the extended nonnegative reals. The supremum over finite A equals that over all A by monotone convergence of finite prefixes. No assumption of finiteness is being smuggled into the definitions.

For every j>=0, choose the lexicographically first maximum-cardinality k-free set R_j in [1,3^j]. This is a mathematical algorithm (finite exhaustive search) and is not claimed computationally efficient. Define one infinite set

    B_k = union_(j>=0) (2*3^j + R_j).

Membership of any fixed integer in B_k is decidable by a finite search in its possible block. No huge block is enumerated in this package.

## 2. The bounded-block separation lemma

If F subset [1,N] and G subset [2N+1,3N] are each k-free, then F union G is k-free.

Proof. An increasing mixed AP has its F terms first, since every G term exceeds every F term. If at least two terms are in F, its difference is <=N-1, so the first term in G is <=2N-1, contradiction. If exactly one term is in F, its second term is >=2N+1, so its third is >=3N+2, contradiction. There are at least three terms because k>=3. QED.

All earlier blocks of B_k lie in [1,3^j] when its j-th block is adjoined. The lemma proves every finite block union k-free. Any forbidden AP in the infinite union would occur in a finite union, so B_k is k-free.

Let a_j=r_k(3^j)/3^j. Each element of the j-th block is at most 3^(j+1), so

    H(B_k) >= (1/3) sum_(j>=0) a_j.                 (1)

## 3. Exact comparison to the extremal profile

For any k-free A, Tonelli's theorem gives

    H(A) = integral_1^infinity A(x) dx/x^2 <= I_k.  (2)

On [3^j,3^(j+1)], monotonicity gives r_k(floor x)<=r_k(3^(j+1)). Thus

    I_k <= 2 sum_(j>=1) a_j <=2 sum_(j>=0) a_j.    (3)

Combining (1)--(3), including the possibility of infinity, proves

    H(B_k) <= M_k <= I_k <= 6 H(B_k),              (4)
    I_k/6 <= M_k <= I_k.                           (5)

The factor6 is a valid constant, not asserted optimal. These inequalities compare suprema and profiles; they do not identify an optimizing set or the value of M_k.

## 4. Equivalent formulations of the fixed-k problem

The following are equivalent:

1. Every infinite k-free set has a convergent reciprocal sum.
2. M_k is finite.
3. I_k is finite.
4. sum_j r_k(3^j)/3^j is finite.
5. The one canonical computable k-free set B_k has a convergent reciprocal sum.

For 1=>5 use k-freeness of B_k. Equations (1)--(4) give the remaining nontrivial implications; 2=>1 is immediate. No change from k to k+1 occurs. If the fixed-k conjecture is false, B_k is itself a computable counterexample. This is conditional and does not assert that its sum diverges.

This formalizes the central obstacle of the prize track: the missing ingredient is summability of the unrestricted finite extremal density profile. Merely noting that different finite maximizers are incompatible does not avoid this obstacle, because they can be separated into the blocks above at bounded harmonic cost.

## 5. Uniform tails and actual maximizers (conditional where necessary)

Assume M_k<infinity. Then I_k<infinity. Uniformly over all k-free sets A,

    sum_(a in A, a>N)1/a
      = integral_N^infinity (A(x)-A(N)) dx/x^2
      <= integral_N^infinity r_k(floor x) dx/x^2 ->0.     (6)

The space X_k of k-free subsets of the positive integers is closed in the compact product {0,1}^N: every excluded AP has finitely many coordinates. Its harmonic functional is continuous by the uniform tail bound (6). Therefore M_k is ATTAINED whenever it is finite.

The same condition is equivalent to the uniform-tail formulation: for every epsilon>0 there exists N such that every k-free set with minimum>N has reciprocal sum<epsilon. The reverse implication gives M_k<=H_[1,N]+epsilon<infinity; the forward implication is (6).

For k=3, known Bloom--Sisask estimates imply M_3<infinity, so the attainment conclusion is unconditional. For k>=4, this argument DOES NOT prove M_k finite; attainment is conditional on the unresolved finiteness assertion. The value of M_3 is not determined here.

## 6. Necessary counting-rate consequence

Partitioning [1,3N] into three length-N intervals gives r_k(3N)<=3r_k(N). Hence a_j is nonincreasing. A nonnegative nonincreasing summable sequence has j*a_j->0: bound floor(j/2)*a_j by the tail from floor(j/2) to j. Therefore

    M_k<infinity => r_k(N)=o(N/log N).                (7)

To pass from powers3 to general N, use 3^j<=N<3^(j+1) and r_k(N)<=r_k(3^(j+1)); the remaining multiplicative factor is bounded. Conversely, an o(N/log N) bound alone is NOT enough for summability; for example 1/(j log(j+1)) tends to zero faster than 1/j but still has divergent sum. A quantitative summable improvement or equivalent structural control remains necessary.

## 7. Local variational inequality for an actual maximizer

If A attains finite M_k, retain its prefix F=A intersect [1,N] and replace its entire tail by a translated extremal set 2N+R with |R|=r_k(N), R subset [1,N]. The separation lemma makes the competitor k-free. Maximality implies

    sum_(a in A,a>N) 1/a >= sum_(x in R)1/(2N+x)
                         >= r_k(N)/(3N).             (8)

Together, (6) and (8) squeeze every maximizing tail between a local cardinality benchmark and the integrated extremal profile. They do not determine the tail or prove the fixed-k conjecture.

For summable weights sigma>1, the same replacement yields

    sum_(a in A,a>N) a^(-sigma) >= r_k(N)/(3N)^sigma

at any maximizing set. PR-AP-009 then excludes all polynomially sparse maximizers; PR-AP-010 excludes every fixed-base automatic maximizer.

## 8. What the finite companion establishes

`extremal_profile.cpp` exhaustively computes r_k(N) and a canonical maximizing witness for k=3,4 and 1<=N<=27. It uses exact integer branch-and-bound with optimistic bound current_cardinality+unprocessed_positions; inclusion tests every AP ending at the new position. Every admissible subset occurs on one branch. A separately written whole-subset Python check verifies N<=12. Finite witnesses, separated block unions, harmonic lower bounds, and finite integral comparisons are verified exactly in `verify_profile_bridge.py`.

Those computations test the implementation and elementary finite instances. They are NOT the proof of (4)--(8), which is the written infinite argument above, and do not evaluate any unrestricted infinite supremum.

## 9. Research and priority boundary

The core separated-block strategy is credited to the pre-existing Sawin discussion. The factor-six packaging, explicit canonical set, conditional attainment and tail inequality were derived in this session; a focused search does not establish novelty. No original prize is closed. In particular, the required summability is still unknown here for k>=4, and proving only one fixed progression length would not prove the full all-length Erdős statement.
