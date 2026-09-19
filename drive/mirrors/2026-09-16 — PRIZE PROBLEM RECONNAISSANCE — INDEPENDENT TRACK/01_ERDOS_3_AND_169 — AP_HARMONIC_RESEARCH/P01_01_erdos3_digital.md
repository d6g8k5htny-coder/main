# PR-AP-001 — Harmonic reduction, digital reformulation, and exact summation

Date: 2026-09-16. Grade: complete author-side derivations with exact computational companions; not independently reviewed, not a solution of a prize problem. Novelty is not established. Uniform harmonic bounds already occur in Gerver (1977); prime-base AP-free constructions and the base-11 example occur in Walker (2020); fast missing-digit summation predates this work. References are in `sources/SOURCE_REGISTER.md`.

## 1. Exact target

A nontrivial k-term arithmetic progression is `{a,a+d,...,a+(k-1)d}` with d>0. Fix k>=3. For finite A contained in the positive integers define H(A)=sum_(a in A) 1/a. Let

C_k = sup { H(A) : A is finite and contains no nontrivial k-AP }.

The fixed-k divergent-reciprocal assertion says that every infinite positive-integer set of divergent reciprocal sum contains a k-AP. The full Erdős #3 target requires this for every k. A result for k=4 alone is not the full prize statement.

## 2. Same-k gluing lemma

Let F be finite k-AP-free, M=max F (M=0 for the empty set), and choose a prime q>k(M+1). For any finite k-AP-free B of positive integers, put C=qB+(q-1). Then F union C is k-AP-free and H(C)>=H(B)/(2q).

Proof. An AP entirely in F or C is impossible. In a mixed k-AP with at least two C terms, their positions differ by an integer between 1 and k-1, invertible modulo q. Since the two values are both -1 modulo q, the AP difference is 0 modulo q. Every term would then be -1 modulo q, impossible for a member of F contained in [1,M]. A mixed AP with at most one C term contains at least two F terms. Its difference is at most M-1, so its largest term is less than kM; but min C>=2q-1>kM. Finally qb+q-1<=2qb for b>=1, which proves the reciprocal estimate. QED.

Consequently C_k<infinity is equivalent to the fixed-k divergent-reciprocal assertion. One direction follows by bounding all finite partial sums. For the other, if C_k=infinity, start with F empty and repeatedly choose q as above and B with H(B)>2q. Adjoin qB+q-1. Every finite stage is k-AP-free and contributes more than 1 new reciprocal mass. The union is infinite, k-AP-free (any finite AP would occur at a finite stage), and has divergent reciprocal sum. QED.

This supplies a self-contained same-k argument; no claim of historical priority is made.

## 3. Exact digital reformulation

Let p>=k be prime. A digit alphabet D subset {0,...,p-1}, containing 0, is admissible if it contains no modular k-AP `{a+jd mod p:0<=j<k}` with d not 0 modulo p. Write C(p,D) for the nonnegative integers all of whose base-p digits belong to D and

S(p,D) = sum_(n in C(p,D)) 1/(n+1).

The set 1+C(p,D) is k-AP-free. Indeed, if an AP difference is not divisible by p, the units digits violate admissibility. If it is divisible by p, subtract the common units digit and divide by p, repeatedly, until this contradiction is obtained. Because D is proper, s=|D|<p, and the highest-nonzero-digit decomposition bounds the tail by a geometric series with ratio rho=s/p; hence S(p,D)<infinity.

**Theorem.** In the extended nonnegative reals,

C_k = sup_(p,D admissible) S(p,D).

Proof. Each finite prefix of the digital set is k-AP-free, so S(p,D)<=C_k. Conversely take a finite k-AP-free F and let D=F-min F. Choose prime p>max(k,2 diam F). If a modular k-AP had all its representatives x_j in D, each second difference x_j-2x_(j+1)+x_(j+2) would be a multiple of p of absolute value less than p, hence zero. The representatives would form a nonconstant integer AP, contrary to F. Thus D is admissible. Moreover F-min F+1 is contained in 1+C(p,D), and shifting left does not decrease reciprocal mass. Therefore S(p,D)>=H(F). Taking suprema proves the theorem. QED.

**The remaining obligation has not disappeared:** one needs a bound UNIFORM in the prime and alphabet. Finiteness for each fixed digital construction does not imply the supremum is finite. The equivalence is an organizing reduction, not a solution or a claimed easier theorem.

## 4. An exact rational certificate for the entire infinite sum

This derivation concerns arbitrary digits D containing 0, with s<p; modular AP-freeness is a separate predicate. For rational 0<a<=1 set S_a=sum_(n in C(p,D))1/(n+a), rho=s/p. Let U_l be independent uniform digits in D, X_0=a, X_(l+1)=(U_l+X_l)/p, Y_l=X_l-1/2. Then X_l in [0,1]. Its distribution equals (n+a)/p^l for a uniformly chosen lower l-digit word n.

Partitioning by the highest nonzero digit yields the exact identity

S_a = 1/a + sum_(l>=0) rho^l sum_(d in D,d>0) E[1/(d+X_l)].

Define b_j=(1/s)sum_(d in D)(d-(p-1)/2)^j and
M_j=sum_(l>=0)rho^l E[Y_l^j].
All converge absolutely. The affine recursion gives

M_0=1/(1-rho),

(1-rho*p^(-j)) M_j = (a-1/2)^j
 + rho*p^(-j) sum_(t=0)^(j-1) binom(j,t)b_(j-t)M_t,  j>=1.

Thus the M_j are exact rationals, recursively computable.

Expanding each reciprocal around d+1/2 through degree t gives

P_t = 1/a + sum_(d>0) sum_(j=0)^t (-1)^j M_j/(d+1/2)^(j+1).

Since |Y_l|<=1/2, the geometric-series remainder is bounded by

E_t = (1/(1-rho)) sum_(d>0) [1/d]*(1/(2d+1))^(t+1).

Indeed |Y/(d+1/2)|<=1/(2d+1), and the denominator left after the geometric remainder is at least d. Summing the uniform bound with rho^l proves

|S_a-P_t|<=E_t.

This is an infinite-series certificate, not a finite-digit truncation treated as exact. Every quantity in the implementation and every outward interval endpoint is rational. Polynomial degree is the only truncation, and its complete remainder is displayed. Infinite-sum interchange is justified by the absolute bounds above.

## 5. Restricted complete search

`run_digital_complete.py` enumerates every admissible alphabet containing 0 for each p in {5,7,11,13,17,19}. Inclusion-maximal alphabets suffice because the digital set and its harmonic sum are monotone under adding digits. The search has 16,063 admissible alphabets and 2,054 maximal alphabets. Every maximal sum is enclosed by the formula above; interval separation certifies the winning fixed alphabet in each base and across these six bases.

The winner is the ALREADY PUBLISHED Walker alphabet D*={0,1,2,4,5,7} in base 11. A degree-40 certificate gives

4421747532398232526 / 10^18 <= S(11,D*) <= 4421747532398232527 / 10^18.

This improves the precision of this package's certificate, NOT a claimed literature record. It does not optimize all primes, all AP-free sets, or all digit languages.

A separate seeded exploratory search tried 300 alphabets at each of 23 prime bases up to 97 (6,900 greedy trials). Those heuristic scores are not certified extrema; they found no better candidate than D*. They are stored separately in `digital_probe.jsonl`.

## 6. A restricted linear-relaxation obstruction

Consider only 0<=x_n<=1 and sum_(n in P)x_n<=k-1 for each k-AP P in [N]. The constant assignment x_n=(k-1)/k is feasible. Its harmonic objective is ((k-1)/k)sum_(n<=N)1/n, which diverges as N grows. Therefore this LP relaxation, and inequalities obtained solely as valid linear combinations of its constraints, cannot produce an N-uniform harmonic upper bound. This does NOT rule out stronger relaxations, integrality reasoning, higher-order constraints, or new mathematics.

## 7. Finite optimal sets and the greedy trap

`ap_exact.cpp` solves the finite harmonic-weighted problem, not the cardinality problem r_k(N). The exact denominator is lcm(1,...,N). The DFS takes each integer in increasing order; inclusion is forbidden precisely when an AP ending there has all its other members already selected. Every admissible subset is represented by a path. A node is pruned only when current value plus the sum of ALL remaining positive weights cannot exceed the current best. This is a valid optimistic upper bound. Hence termination certifies a finite maximum.

The implementation is explicitly limited to N<=48, where the integer arithmetic fits signed 128-bit storage. The independent Python implementation checks small cases using the whole-AP relation rather than last-element masks. Larger cases remain one main implementation with witness checks and deterministic replay, not independent whole-search certification.

At N=24 the greedy prefix is not optimal: for k=3, a better set is {1,2,4,5,11,12,15,16,24}, improving the greedy reciprocal sum by 127/21840. This is a finite-horizon counterexample only; it neither refutes infinite greedy optimality nor supplies an asymptotic r_k bound. Similar finite counterexamples for k=4 are in the result table.

## 8. Research disposition

Full Erdős #3: OPEN here. Erdős #142: no new asymptotic bound. What is completed is a self-contained digital reformulation, exact infinite-sum certifier, restricted finite optimization, and an explicit obstruction to a naive LP route. The adaptive base-11 extremality theorem is in the companion file. No monetary award or success probability is claimed.
