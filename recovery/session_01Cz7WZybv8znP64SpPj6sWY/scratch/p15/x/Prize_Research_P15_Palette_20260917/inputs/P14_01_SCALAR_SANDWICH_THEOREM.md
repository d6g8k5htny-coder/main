# P14-A — A scalar-sandwich criterion for composable covers

Date: 2026-09-17. Grade: complete author-side argument; exposed same-author work, no independent review; historical novelty unestablished. The statement is a new scope extension within this project, not a claim to have invented fractional covering or sorted-prefix rounding.

## 1. Hypotheses and exact conclusion

Let D be a decreasing family on a finite ground set X, containing the empty set and every singleton. Suppose numbers 0<=a_i<1 and a real kappa>=1 satisfy, for EVERY U subset X,

(S1) a(U)<1 implies U in D;

(S2) U in D implies a(U)<kappa,

where a(U)=sum_(i in U)a_i. Both conditions are setwise, including zero-probability configurations. (S2) cannot be replaced by a bound only on a sampled set or on a(U)'s expectation.

Under independent coordinate probabilities p_i in[0,1], let Q=P_p(D^c). At arbitrary nonnegative prices z_i<=A p_i, A>=1, set

    R=ceil(12 A^2)+2,
    K=ceil(8 kappa (R+1)).

If Q<=2/3, there is an explicit finite integral generator cover of D^(K) of price cost <=Q/2. In particular K=ceil(408 kappa) works for z<=2p.

Define phi(t)=min(1,-log(1-t)), with phi(1)=1. For every probability vector and prices c_i<=phi(p_i),

    covercost_c(D^(ceil(408 kappa))) <= phi(Q).

No ground-set size, row count, maximal witness rank, or overlap bound appears. Existence of the scalar sandwich is the load-bearing structural condition.

## 2. Exact normalization and seed counting

For a prefix V, let lambda=sum_(i in V)p_i, Z=product_(i in V)(1+p_i), and t_i=p_i/(1+p_i). If B=D^c is increasing, then

    sum_(S subset V, S in B) product_(i in S)p_i
      = Z P_t(B on V, all outside coordinates absent)
      <= Z Q <= exp(lambda) Q.                         (1)

This sums disjoint EXACT configurations after normalization, not overlapping containment events. It is valid at p_i=0 or1.

If each k-element generator g in V has a bad subset S(g) of size s, choose one deterministically. The pair (S(g),g\S(g)) determines g. For m=k-s,

    sum_g product_(i in g)z_i
      <= A^k exp(lambda) Q lambda^m/m!.               (2)

The extension sum is at most lambda^m/m!, because ordered expansion counts each distinct m-product m! times. Dropping disjointness in this upper bound only adds nonnegative terms. No division by Q occurs.

## 3. Filtered prefixes and coverage

Sort coordinates by decreasing a_i, breaking ties deterministically. Put

    P_j=sum_(i<=j)p_i, M=sum_i a_i p_i,
    n_k=max{j in{0,...,n}:P_j<=k/R}.

Use the generator family

    G=union_(k=1)^n {g subset[n_k]: |g|=k, a(g)>=4 kappa}.   (3)

If U={u_1<...<u_l} avoids G, either a(U)<4 kappa, or let k0 be its first prefix reaching 4 kappa. For j>=k0, u_j>n_j, hence P_(u_j)>j/R. On each original-probability interval [P_(i-1),P_i) set f(t)=a_i. Zero-length intervals disappear. It is nonincreasing, nonnegative, and integrates to M. Each interval [(j-1)/R,j/R), j>=k0, contributes at least a_(u_j)/R. Summing gives

    a(U) < 4 kappa + R M.

Consequently a(U)>4 kappa+RM forces a generator. The strict endpoint is more than sufficient for the application; equality in a retained generator's weight is INCLUDED.

## 4. Every heavy generator contains an original bad seed

If |g|=k and a(g)>=4 kappa, then k>4 kappa>=4 because each a_i<1. Its s=ceil(k/4) largest coefficients have sum at least (s/k)a(g)>=kappa. By the CONTRAPOSITIVE of (S2), this s-set is in the original bad family. Thus (2) applies with

    m=k-ceil(k/4)=floor(3k/4)>=3.

Since k<=2m, A>=1, lambda<=k/R, and m!>=(m/e)^m,

    A^k exp(lambda) lambda^m/m!
      <= [ (2e A^2/R) exp(2/R) ]^m
      <= [6 A^2/(R-2)]^m <=2^(-m).                    (4)

Here e<3 and exp(x)<=1/(1-x) for 0<=x<1. These follow directly from their positive power series; R>2. Each m is attained by floor(3k/4) at most twice. Therefore the ENTIRE family in(3) costs

    <=2Q sum_(m=3)^infinity 2^(-m) = Q/2.            (5)

The infinite series bounds a finite collection. Noninteger kappa is allowed; the proof only needs k>4 kappa and m>=3, so no unjustified rounding of kappa is used.

## 5. Mean and strict packing

Let W=sum_i a_i X_i and M=EW. Since a_i<1, Var(W)<=M. By(S2), P(W<kappa)>=P(D)>=1/3. If M>=4 kappa, the one-sided second-moment inequality gives

    P(W<kappa) <= M/[M+(M-kappa)^2]
               <= 4/(4+9 kappa) <=4/13<1/3,

a contradiction. Thus M<4 kappa. The rational function decreases for M>kappa. Zero variance is immediate; the one-sided bound follows by applying Markov to (W-M-t)^2 and minimizing over t>=0.

A scalar bin with load strictly below1 is good by(S1). Next-fit with each a_i<1 uses <=K bins whenever a(U)<=K/2: if K+1 bins open, adjacent bin loads sum to at least1; pairing bins and retaining the last positive item proves total>K/2. Hence U in D^(K) implies

    a(U)>K/2 >=4 kappa(R+1)>4 kappa+RM.

Section3 supplies coverage, and(5) supplies its cost.

## 6. Endpoints, singletons and general downsets

If Q=0, each original minimal bad set has zero probability product and hence zero price product. Keeping these zero-cost generators gives a valid cover; they must not be discarded because of their numerical cost. If Q>2/3, phi(Q)=1 (as e<3), so the empty generator costs1 and supplies the capped conclusion.

For a general decreasing D, first separate J={i:{i} notin D}. On X\J the restricted family contains every singleton. Suppose that family has a scalar sandwich of width kappa. The identity

    1-Q=product_(i in J)(1-p_i)*(1-Q_R)

holds exactly. Add singleton generators at prices c_i<=phi(p_i)<=-log(1-p_i), use the nonsingleton result, and cap the combined cover at1. Its cost is at most -log(1-Q). A p_i=1 on J implies Q=1 and is covered by the trivial case. Thus the same all-probability conclusion extends with the same palette.

This argument strengthens a SUFFICIENT structural criterion. It does not prove a bounded sandwich for arbitrary decreasing families. See P14-D for a connected counterexample to that automatic inference.
