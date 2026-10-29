# Rare-cluster consequences of the sharp C6 moment bounds

**Object:** OA-C6-RARE-CLUSTER-20260929-v1. **Author:** OpenAI / ChatGPT, current owner-authorized session, 29 September 2026.
**Disposition:** author-side conditional proof candidate; nonauthor review OPEN. **Scientific effect: NONE.** No source proof, status register, Boolean, prize or acceptance disposition is changed. Same GitHub account as the other model lanes: no organizational-independence claim.

## 1. Exact interface and dependency boundary

Use the count N=N_r under the original determinant-tilted pinned law Q_r^W of [C6, Section 1.1]: all-index critical points, excluding the two conditioned endpoints, with heights strictly between b-k r^3 and b. Fix d>=2, the torus side L, compact birth marks and compact positive gap marks k>=k_->0. The source covariance, conditioning, full normalizer and frames are unchanged. All constants below inherit this scope; none is numerical or uniform as d,L or the marks escape it.

Put e=r^3. The ONLY analytic inputs to this note are

    (H1) E N <= A e,
    (H2) P(N>=2) >= a e, where a>0,
    (Hq) E (N)_q <= A_q e for every fixed integer q>=2.

Here (N)_q=N(N-1)...(N-q+1), and A_1=A. [D5, Theorem G_d], summed over the fixed d+1 indices, gives H1. [C6, Corollary Theta] gives H2, and [C6, Theorem Q] gives Hq. Thresholds may be decreased for each finite collection of moment orders; no uniform-in-q assertion is needed. These premises imply 2a<=A.

Both sources are pinned to commit 78ac1100217279e7b692f0959f86eee34cd1910e, with full byte identities in SOURCE_PINS.json. Their recorded nonauthor reviews do not constitute a review of this new note. The parent Gaussian proofs are not revalidated here. A defect in an input blocks the corresponding application, not the elementary abstract implication.

[C6] frontiers/c6_palm_route_20260929/PROOF.md, Theorem Q and Corollary Theta.
[D5] frontiers/d5_dimension_lift_20260929/PROOF.md, Theorem G_d.
[LOCAL] frontiers/window_multiplicity_laws_20260928/LOCAL_MULTIPLICITY.md, Section 7, already proves the exact-mean Bernoulli-configuration obstruction. That result is prior in-project work, not a new claim of this note.

For probability laws, dTV is the supremum over measurable events, equivalently half the l1 distance on this countable state space. Norms of finite or signed measures written ||.||_1 mean full l1, without the factor 1/2.

## 2. Conditional counts and size-biased counts are different

Write p=P(N>0), mu=E N, let J have law F=Law(N | N>0), and let Y have size-biased law P(Y=n)=n P(N=n)/mu. Then

    a e <= p <= A e,             2a e <= mu <= A e.                 (2.1)

For each integer k>=1, define B_k=sum_{j=1}^k S(k,j) A_j, where S(k,j) are the nonnegative Stirling numbers of the second kind. The pointwise identity n^k=sum_j S(k,j)(n)_j gives

    E N^k <= B_k e,
    E J^k <= B_k/a,              E Y^k <= B_{k+1}/(2a).            (2.2)

Consequently both families are tight and every fixed power is uniformly integrable (use the next power in Markov's tail bound). Moreover

    P(J>=2) >= a/A,
    E J-1 >= a/A,
    2a/A <= E(Y-1)=E(N)_2/mu <= A_2/(2a).                          (2.3)

The exact difference between the two excess means is

    E(Y-1) - E(J-1) = Var(J)/E J >= 0.                            (2.4)

Indeed P(Y=n)=n F(n)/E J, so E Y=E J^2/E J. This proves (2.4). It does not assert that the two sampling laws are equal: J taking 1 and 3 with probabilities 1/2 each has mean excess 1, while Y has mean excess 3/2. The new use of all-order C6 bounds is (2.2) and its compactness consequences, not merely the distinction already stated in C6's Corollary P.

## 3. Sharp ordinary-Poisson obstruction, including optimized intensity

**Theorem O.** Under H1 and H2 alone, for all sufficiently small r,

    (a/2) r^3 <= inf_{lambda>=0} dTV(Law N, Pois(lambda)) <= A r^3. (3.1)

The same order holds for lambda=mu. Thus an ordinary Poisson law cannot achieve o(r^3) error, even by fitting an arbitrary intensity. Absolute distance still tends to zero; this is an obstruction at the rare-event scale, not a claim that the distance is bounded away from zero.

**Proof.** Take e<=min{a/(4A^2),1/(4A)}. If lambda<=2Ae, then

    P(Pois(lambda)>=2) <= lambda^2/2 <= 2A^2 e^2,

by the second factorial moment of a Poisson variable. Testing {n>=2} gives distance at least ae/2. If lambda>2Ae, test {n>0}; monotonicity and 1-exp(-x)>=x-x^2/2 give

    1-exp(-lambda)-p >= 1-exp(-2Ae)-Ae
                         >= Ae-2A^2 e^2 >= Ae/2 >= ae/2.

The upper bound in (3.1) uses Pois(0)=delta_0 and dTV(Law N,delta_0)=p. For mean matching, the triangle inequality through delta_0 gives at most p+1-exp(-mu)<=2Ae, and the same lower bound applies. This proves the theorem. Notice that Hq is not needed.

Conditioning a mean-matched Poisson variable on positivity does not cure the discrepancy: its probability of being >=2 tends to zero as mu->0, whereas F([2,infinity))>=a/A. Therefore the conditional TV discrepancy has liminf at least a/A.

## 4. Canonical r-dependent compound Poissonization: sharp r^6 error

Let C_r have the compound Poisson law CP(p,F): K~Pois(p), independent positive integer marks J_1,J_2,... with law F, and C_r=sum_{i=1}^K J_i. The empty sum is zero.

**Theorem C.** Under H1 and H2, for p<=1,

    p^2/3 <= dTV(Law N, Law C_r) <= p(1-exp(-p)) <= p^2.           (4.1)

In particular the error for THIS approximant is Theta(r^6), with bounds a^2 r^6/3 and A^2 r^6. Its mean is exactly mu.

**Proof.** Law N=(1-p)delta_0+pF. Apply the same mark-summing probability kernel to a Bernoulli(p) count and a Pois(p) count. Total variation contracts under that kernel. The Bernoulli law has its sole positive difference at count 1, of mass p-p exp(-p), so their TV distance is exactly p(1-exp(-p))<=p^2. Positivity of all marks means P(C_r=0)=exp(-p). Testing the zero event gives

    dTV >= exp(-p)-1+p >= p^2/2-p^3/6 >= p^2/3.

The Taylor inequalities follow from their signed integral remainders on p>=0. Finally E C_r=p E J=E N.

**Essential limitation.** F is the actual, generally unknown, r-dependent conditional count law. This is a universal re-expression/Poissonization of a rare occurrence, not an explicitly identified cluster law, an asymptotic independence theorem, or a spatial Poisson-cluster process. No novelty is asserted for this elementary generic probability construction. The cluster rate is p, NOT mu: replacing it by mu changes the mean to mu E J in general. The r^6 lower bound is for CP(p,F), not a minimax lower bound over every conceivable compound Poisson parameterization.

## 5. Factorial-moment matching to second rare-event order

Let a_j=E(N)_j. Under Hq, for each fixed integer q>=1,

    E(C_r)_q = sum_{pi in Partitions({1,...,q})} product_{B in pi} a_{|B|}. (5.1)

For q=1 this equals mu. For q>=2,

    0 <= E(C_r)_q-E(N)_q = O_q(e^2)=O_q(r^6).                    (5.2)

**Proof.** Conditional on K, expand the ordered selections of q distinct items in the sum of its independent integer-valued marks. A set partition of the q labels specifies which labels belong to the same mark; if it has j blocks its expectation is E(K)_j times the product of the mark factorial moments. Since E(K)_j=p^j and p E(J)_b=a_b, summing gives (5.1). All terms are nonnegative and moments finite, so Tonelli applies. No moment-generating-function analyticity is assumed. The single-block partition contributes a_q. Every other partition has at least two blocks; each a_b=O(e), and there are finitely many partitions for fixed q, proving (5.2). In particular the second-moment correction is exactly mu^2.

## 6. Subsequential finite cluster-intensity measures, not a unique limit

Define a finite measure on the positive integers by

    nu_r(n)=r^(-3) P(N_r=n),       lambda_r=nu_r(N_+)=p/r^3.

Then a<=lambda_r<=A, nu_r({2,3,...})>=a, and sum n^k nu_r(n)<=B_k for k>=1 (use B_0=A for mass).

**Theorem S.** Every sequence r_i->0 has a subsequence and a finite measure nu on N_+ such that for every fixed nonnegative integer k,

    sum_{n>=1} n^k |nu_{r_i}(n)-nu(n)| -> 0.                    (6.1)

Its mass lambda is in [a,A], and nu({2,3,...})>=a. Along that subsequence the positive conditional laws converge to nu/lambda in TV and all fixed moments. The size-biased laws converge to n nu(n)/sum_m m nu(m), also in TV and all fixed moments.

**Proof.** Each coordinate is bounded by A. Diagonal selection gives coordinatewise convergence. Fatou preserves the moment bounds. For every k>=0 and M>=1,

    sum_{n>M} n^k nu_r(n) <= B_{k+1}/M,

and the same holds for nu. On the finite set n<=M coordinatewise convergence gives weighted l1 convergence. First take the subsequence limit and then M->infinity. The SAME selected subsequence works for every k because it was selected only by coordinates and has all these tail bounds. Mass and the multiple-count lower bound pass to the limit by k=0 convergence. Denominators for the two normalized laws stay bounded below by a and 2a, respectively, so their claims follow from (6.1), using one extra power for size bias.

There is also the exact signed-measure identity

    Law N_r = delta_0 + r^3 (nu_r-lambda_r delta_0).               (6.2)

Thus (6.1) gives a first-order signed-measure expansion along each selected subsequence. It gives neither uniqueness nor a rate for nu_r->nu.

**Sharp non-implications.** Set P(N_r>0)=r^3 and let the positive count alternate between 2 and 3 on successive dyadic intervals of r. H1,H2 and every Hq hold, but nu_r alternates between delta_2 and delta_3. Taking N_r=2 Bernoulli(r^3) also shows that no positive lower bound for the third or higher factorial moments follows. Finally, all polynomial moments need not imply any exponential moment: let P(J=2^j) be proportional to 2^(-j^2), j>=1. Every polynomial moment is finite, whereas E exp(tJ)=infinity for each t>0. The rare mixture with probability r^3 satisfies H1,H2,Hq. These examples concern what the premises alone imply, not counterexamples to the specific Gaussian model having additional structure.

## 7. Independent-replica limit, with an explicit extra assumption

Take m_r=floor(t/r^3) INDEPENDENT copies of the entire count N_r, for fixed t>=0, and let T_r be their sum. Along a subsequence from Theorem S,

    dTV(Law T_r, CP(t nu))
       <= t A^2 r^3 + A r^3 + t ||nu_r-nu||_1 -> 0.              (7.1)

Here CP(t nu) denotes the compound Poisson law with finite Levy measure t nu, with no jump of size zero.

**Proof.** Couple each copy to its Section 4 approximant: the total error is at most m_r p^2<=t A^2 r^3. The sum of the independent approximants has Levy measure m_r r^3 nu_r. Two finite Levy measures eta,zeta can be coupled by sharing their pointwise minimum and independent residual Poisson jumps; their compound count laws have TV distance at most ||eta-zeta||_1. Apply

    ||m_r r^3 nu_r-t nu||_1 <= r^3 A + t ||nu_r-nu||_1.

No independence of windows or spatial cells in ONE field has been proved. Such a use needs a new dependence estimate. Equation (7.1) is a theorem about deliberately independent experiments, not a new spatial claim about the original torus.

## 8. Review and verification boundary

Requested nonauthor slices: (A) optimized-intensity Poisson bound and exact binary-kernel constants; (B) factorial partitions, weighted-l1 compactness and the independent-replica coupling; (C) source interface and non-claims. The checker covers exact finite identities, rational probes and explicit false variants only. Neither it nor a green workflow proves H1/H2/Hq, validates all infinite-dimensional arguments, or promotes any register. There is no Lean formalization of this packet. External primary-source reconnaissance is recorded separately, without claiming a comprehensive novelty audit.
