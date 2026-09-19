# PR-AP-009 — Sparse-tail replacement and strict nonextremality

Date: 2026-09-16. Complete author-side derivation below; no external reviewer or theorem-prover validation. Novelty is unestablished. The sphere construction is classical Behrend machinery, not a new invention. Related head-and-block improvement methods are already used in the literature. This note states and proves the general implication needed by this campaign.

## 1. Exact definitions and result

A subset of the positive integers is k-free if it contains no k-term arithmetic progression with positive common difference. Put A(x)=|A intersect [1,x]| and H_s(A)=sum_(a in A) a^(-s), with H=H_1. All series are nonnegative.

**Theorem 1 (harmonic version).** Let k>=3, 0<=alpha<1, C<infinity, and A be k-free with A(x)<=C*x^alpha for x>=1. Then there is a FINITE k-free F with H(F)>H(A). Consequently A does not maximize the unrestricted harmonic sum over k-free sets. This does not assume the unrestricted supremum is finite or attained.

**Theorem 2 (weighted version).** More generally, if s>alpha, the same conclusion holds for H_s. In particular it holds for every s>=1 under the hypotheses of Theorem 1.

The witnesses specified below may be enormously large. This package gives a finite mathematical prescription, not a literal enumeration of their members.

## 2. A bounded separated-block lemma — valid even for k=3

If F subset [1,M] and B subset [2M+1,3M] are each k-free, with k>=3, then F union B is k-free.

Proof. In an increasing AP crossing from F to B, the F terms come first. If at least two terms lie in F, the difference is at most M-1. Thus the first B term is at most 2M-1, impossible. If just one term lies in F, the second term is at least 2M+1 and the third is at least 2(2M+1)-M=3M+2, also impossible. APs wholly in one component are excluded by hypothesis. QED.

The upper endpoint 3M matters. For k=3, the weaker condition B subset [2M+1,infinity) is insufficient: F={1}, B={3,5} gives the AP 1,3,5. The bounded-block version is what the argument consumes. The lower endpoint also needs slack: {M,2M,3M} defeats the version with B starting at 2M.

## 3. Classical sphere block, written out

Choose integers m>=2 and d>=1. Partition the m^d vectors v in {0,...,m-1}^d by their integer squared norm sum_i v_i^2. There are at most d(m-1)^2+1 norm values, so the largest layer V has

    |V| >= m^d / [d(m-1)^2+1] >= m^d/(d m^2).

Choose the smallest radius maximizing cardinality to make V definite. Encode

    e(v)=sum_(i=0)^(d-1) v_i(2m)^i,
    N=(2m)^d.

The set Q=e(V) lies in [0,N-1] and is 3-free. Indeed if e(u)+e(w)=2e(v), all coordinate sums and doubled digits are at most 2m-2<2m, so no carries occur and u+w=2v coordinatewise. Equal squared norms imply

    ||u||^2+||w||^2-2||(u+w)/2||^2 = (1/2)||u-w||^2=0.

Thus u=v=w. Injectivity of base expansion is exact.

Translate Q to

    B=2N+1+Q subset [2N+1,3N].

B is 3-free, hence k-free for every k>=3. Its weighted mass obeys

    H_s(B) >= |Q|/(3N)^s
           >= N^(1-s)/(3^s d 2^d m^2).

In particular, take d=2t and m=2^t for an integer t>=1. Then

    N_t=2^(2t(t+1)),
    H_s(B_t) >= N_t^(1-s)/(2t*3^s*2^(4t)),
    H(B_t) >= 1/(6t*2^(4t)).                 (1)

No computational estimate of a giant sphere is required for (1); the elementary pigeonhole and no-carry arguments prove it for every t.

## 4. Compare new mass with the OLD omitted tail

Partial summation and A(x)<=Cx^alpha give, for s>alpha,

    T_s(M):=sum_(a in A,a>M) a^(-s)
       = -A(M)M^(-s)+s integral_M^infinity A(x)x^(-s-1)dx
       <= Cs/(s-alpha) M^(alpha-s).          (2)

Dropping the nonpositive boundary term is safe. The series converges by this bound.

The empty-set case is immediate by adjoining {1}. Otherwise enlarge the count constant to C>=1, so the ratio below never divides by zero. Let F_t=(A intersect [1,N_t]) union B_t. The bounded separated-block lemma proves k-freeness without assumptions about cross-digit modular behavior. Equations (1),(2) give

    H_s(F_t)-H_s(A)
      >= N_t^(1-s)/(2t*3^s*2^(4t))
         - Cs/(s-alpha) N_t^(alpha-s).      (3)

The ratio of the first term to the second is

    [(s-alpha)/(2Cs3^s)] * 2^(2(1-alpha)t(t+1)-4t)/t,

which tends to infinity because alpha<1. Thus (3) is strictly positive for all sufficiently large t. This proves both theorems. QED.

The s=1 proof needs only an old tail estimate at the selected scales, not a full power law: if T_1(N_t)<1/(6t2^(4t)) for some t, the same replacement strictly improves A.

## 5. Exact applications to Phase02's entire menus

These are relative inequalities against the exact menu suprema, NOT new claims of an unrestricted record. They are far weaker numerically than a separate reported head-and-block improvement identified in the source register.

### Every bases-2..30 adaptive tree

Phase02 proved, uniformly over that class, T_1(N)<52N^(-1/4). Choose t=10, N=2^220. Then every such tree A admits a finite 4-free improvement of at least

    gamma_small = 1/(60*2^40) - 52/2^55 > 0.

Hence the unrestricted M_4 exceeds the exact Phase02 menu optimum by at least gamma_small, whether or not M_4 is finite. The first term is the whole Behrend block's guaranteed mass; the second includes ALL of A's discarded tail.

### The expanded Walker menu

Every allowed alphabet in the expanded menu obeys |D|^5<=b^4 and b<=193. This includes all affine images of D55 because their cardinality is unchanged. The stopping-cylinder proof therefore gives A(x)<193^(4/5)*x^(4/5) and

    T_1(N) <=5*193^(4/5)N^(-1/5) <965 N^(-1/5).

At t=13, N=2^364,

    gamma_expanded = 1/(78*2^52) - 965/2^72 > 0.

The second term rounds the true tail upper bound UP, because N^(-1/5)=2^(-72.8)<2^(-72). Thus every tree in that whole menu can be improved by a finite 4-free set by this common positive amount. In particular, the exact stationary Walker55 sum is not an unrestricted maximum. Its construction remains prior work.

Using the baseline degree40 rational enclosure of the exact stationary Walker55 sum, the explicit finite head-plus-sphere prescription at N=2^364 has H(F)>4439753369254540650/10^18. The input enclosure and exact guard are in FORMAL_WALKER_IMPROVEMENT.json. This is not an enumerated witness and does not beat the newer reported head/block improvement.

The finite hypotheses |D|^5<=b^4 for the explicit larger alphabets and the maximal small-base cardinalities are checked in `verify_phase03.py`; the adaptive stopping-line proof is written in PR-AP-003.

## 6. What this does NOT prove

- It gives no uniform upper bound for M_4 and no solution of Erdos #3 or #169.
- It does not create a divergent harmonic set: an infinite chain of strict improvements may have summable gains.
- It does not prove that finite-state verification tools cannot prove a conjecture. It restricts particular WITNESS sets, not all proof methods.
- It does not say a sequence with growing menu/state complexity cannot approach the supremum.
- The large finite head and sphere are mathematical prescriptions, not claimed computer-enumerated data.
- It is not claimed to beat the newer first-party numerical record report or to be a novel historical result.


## 7. Every bounded-base adaptive digit menu is globally suboptimal

This conclusion is not restricted to the two menus enumerated in Phase02. Fix B>=2. At each node allow an arbitrary base 2<=b<=B and any PROPER alphabet D containing0. Assume the resulting integer set is k-free; strong local modular k-AP exclusion is one sufficient condition. No depth or history bound is imposed.

Take L=B^2 and beta=1-1/L. Since |D|<=b-1, Bernoulli's inequality gives

    (b/(b-1))^L >=1+L/(b-1)>=b,
    |D|^L <=(b-1)^L<=b^(L-1).

Thus sum of child cylinder weights Q^(-beta) is no larger than the parent weight. Stopping at the first prefix product Q>N gives N<Q<=BN. There are at most (BN)^beta stopping cylinders, and each holds at most one integer <=N. Therefore the accepted positive translate A satisfies

    A(N)<=(BN)^beta<=B*N^beta,
    sum_(a in A,a>N)1/a <=B^3*N^(-1/B^2).

Set e=1/B^2,C=B and t=16*C/e=16B^3. The calculation in PR-AP-010 Section7 gives a uniform strictly positive replacement gain

    Delta_B=1/(12*t*2^(4*t)).

Hence the unrestricted k-free harmonic supremum is strictly larger than the supremum over this entire bounded-base class by at least Delta_B, including arbitrary adaptive choices. When the menu is finite and guarantees k-freeness locally, its own optimum is attained by the usual finite-action Bellman compactness/contraction argument, but none of those optimizers is a global optimizer.

This does NOT say bounded-base automata or finite-state proof tools cannot VERIFY an arbitrary mathematical proof. It is a theorem about the sparsity and harmonic extremality of specified WITNESS families. It leaves unbounded radices and increasing automaton complexity outside its scope.
