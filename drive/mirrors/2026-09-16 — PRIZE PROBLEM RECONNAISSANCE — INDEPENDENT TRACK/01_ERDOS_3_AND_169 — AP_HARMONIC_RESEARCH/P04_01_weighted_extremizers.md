# PR-AP-013 — Every summably weighted maximizer has full all-scale counting dimension

2026-09-16. Grade: complete author-side proof, not externally reviewed and not claimed historically novel. The sphere construction is classical Behrend machinery, supplied in Phase03 PR-AP-009. This does not prove finiteness of the unrestricted harmonic four-progression supremum.

## 1. Setting and a quantitative variational inequality

Fix a progression length k>=3 and a weight exponent sigma>1. A set is k-free if it contains no nontrivial k-term arithmetic progression. Put

    H_sigma(A) = sum_(a in A) a^(-sigma),
    M_(k,sigma) = max_(A k-free) H_sigma(A).

The maximum exists: the k-free family is closed in the compact product space {0,1}^N and the functional is continuous, since its tail is uniformly bounded by the tail of zeta(sigma). This is the Phase03 compactness argument, not a new proof of a prize problem.

For any k-free A, define its objective gap Delta=M_(k,sigma)-H_sigma(A)>=0. Let r_k(N) be the maximum cardinality of a k-free subset of [1,N]. For integers L>N>=1,

**Variational theorem.**

    A(L) >= r_k(N)/3^sigma - Delta*N^sigma
                         - N^sigma*L^(1-sigma)/(sigma-1).       (1)

Proof. Choose a cardinality extremizer R in [1,N]. The set

    F=(A intersect [1,N]) union (2N+R)

is k-free by the bounded separated-block lemma: any mixed AP with two terms in the first interval has first upper term <=2N-1; an AP with only one lower term has third term >=3N+2. Both contradict the block [2N+1,3N].

Since H_sigma(F)<=M_(k,sigma),

    sum_(a in A,a>N) a^(-sigma) >= r_k(N)/(3N)^sigma - Delta.

On the other hand,

    sum_(a in A,a>N) a^(-sigma)
       <= A(L)*N^(-sigma) + sum_(n>L) n^(-sigma)
       <= A(L)*N^(-sigma) + L^(1-sigma)/(sigma-1).

Multiply by N^sigma to obtain (1). Counting some prefix elements in A(L) only enlarges the upper estimate, so there is no lost boundary term. QED.

Equation (1) also applies to near-maximizers. A finite objective gap cannot be ignored: its contribution is Delta*N^sigma and grows with N.

## 2. Behrend cardinality at all large scales

At N_t=2^(2t(t+1)), the explicit sphere construction from PR-AP-009 gives

    r_k(N_t) >= N_t/(2t*2^(4t)),    t>=1.             (2)

It is 3-free and therefore k-free for every k>=3. Monotonicity fills the gaps between these prescribed scales. Indeed log N_(t+1)/log N_t ->1 and log(2t2^(4t))/log N_t ->0. Consequently, for each delta>0 there is N_delta such that

    r_k(N)>=N^(1-delta),   N>=N_delta.               (3)

This is a classical Behrend-scale consequence, not a new extremal lower exponent. No giant sphere is enumerated by this package.

## 3. Full limit, not just a limsup

**Theorem.** For every maximizing set A and every fixed sigma>1,

    lim_(X->infinity) log A(X)/log X = 1.             (4)

Proof. Fix eta>0 and choose 0<delta<eta(sigma-1). Take Delta=0 in (1), L=ceil(N^(1+eta)), and consume (3). The first positive term is at least N^(1-delta)/3^sigma. The last term is O(N^(1-eta(sigma-1))), hence is eventually at most half the positive term. Thus

    A(ceil(N^(1+eta))) >= c_(sigma,eta,delta)*N^(1-delta)

for every sufficiently large N. For an arbitrary sufficiently large X, choose N=floor((X-1)^(1/(1+eta))). Then ceil(N^(1+eta))<=X and N is comparable to X^(1/(1+eta)), so

    liminf log A(X)/log X >= (1-delta)/(1+eta).

Let eta decrease to zero, choosing for example delta=min(eta(sigma-1)/2,eta/2). The right side tends to1. The upper bound A(X)<=X completes (4). QED.

This strengthens Phase03's limsup-only consequence for these summable weights. It does not imply positive asymptotic density or that A(X)/X converges. The argument loses its universal zeta-tail bound at sigma=1; setting sigma=1 in (1) is illegal.

## 4. A completely explicit all-scale bound at sigma=2

**Theorem.** Every maximizing k-free set for H_2 satisfies

    A(X) >= X * 2^(-16*sqrt(log_2 X)),   X>=9216.      (5)

Proof. Define h_t=2t2^(4t), N_t as in (2), and

    L_t=18h_t N_t=36t2^(4t)N_t.

Equation (1), Delta=0, sigma=2, and (2) give

    A(L_t) >= N_t/(9h_t) - N_t^2/L_t
            = N_t/(18h_t)=N_t/(36t2^(4t)).           (6)

If L_t<=X<L_(t+1), monotonicity gives the same lower bound. Let x=log_2X. Comparing x with the logarithm of (6),

    x-log_2[N_t/(36t2^(4t))]
      <=12t+8+log_2[1296t(t+1)]
      <=14t+18.

Here 1296<2^11 and t(t+1)<=2^(2t-1). Meanwhile x>=log_2 L_t>=2t^2+6t>=2(t+1)^2. Finally

    [16 sqrt(2)(t+1)]^2-(14t+18)^2
      =316t^2+520t+188>0.

Both sides being positive, 16sqrt(x)>14t+18. Equation (5) follows. The starting value is L_1=9216. QED.

The exact integer tests verify the algebra and a finite collection of instances, but the inequalities above prove (5) for ALL t and X. This is not extrapolated from a finite table.

## 5. Scope

The result concerns the structure of maximizers whose existence is already guaranteed for sigma>1. It supplies neither their exact objective value nor an explicit infinite optimizing set. It is not a theorem about positive density, harmonic sigma=1 for k>=4, or the convergence of arbitrary k-free reciprocal sums. Novelty requires a dedicated review of weighted extremal-set literature.
