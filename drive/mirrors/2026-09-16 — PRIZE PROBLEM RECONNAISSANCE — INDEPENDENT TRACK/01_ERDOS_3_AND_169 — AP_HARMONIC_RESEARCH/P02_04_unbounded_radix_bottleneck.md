# PR-AP-006 — A quantitative location of the unbounded-base obstruction

2026-09-16. Author-side proof; no novelty claim. All harmonic quantities refer to the positive translate by one. This is not a uniform bound solving Erdos #3.

Let b>=2, 0 in D subset {0,...,b-1}, s=|D|<b, rho=s/b. Let C be its stationary digit set, S=sum_(n in C)1/(n+1), and H=sum_(d in D)1/(d+1). No AP-freeness is required for the following elementary bound:

    H-rho <= (1-rho)S <= H.                         (1)

Indeed, separating one-digit terms from longer words gives

    S=H+sum_(l>=1)rho^l sum_(d in D,d>0) E[1/(d+X_l)],

where 0<=X_l<=1. Let h=sum_(d>0)1/d. Then

    H + rho/(1-rho)*(H-1) <= S <= H + rho/(1-rho)*h.

Moreover

    h-(H-1)=sum_(d>0)1/(d(d+1)) <= 1,

so h<=H. These two inequalities prove (1). Also S>=H directly because the digital set contains its digits. Hence

    1 <= S/H <= 1/(1-rho).                         (2)

Thus along ANY family with rho tending to zero, S/H tends uniformly to one: the relative contribution of recursive tail self-similarity vanishes. Exact positivity of each individual S does not yield an upper bound uniform over alphabets and bases.

For prime modular k-AP-free alphabets with fixed k, the density tends to zero as the prime grows, by the ordinary Szemeredi density theorem applied to the digit representatives. Consequently (2) shows that the unbounded-prime version of the harmonic problem asymptotically reduces to controlling the first-level digit harmonic mass H. It cannot be resolved merely by sharpening the convergent tail estimate of each fixed digital construction.

This is a bottleneck clarification, not a new density theorem and not a proof that the harmonic mass is bounded. Positive density exclusion alone permits nonsummable normalized harmonic masses. The applicable general density theorem is prior work; see sources/SOURCE_REGISTER.md.

## Fixed bases versus unbounded bases

A finite menu of proper alphabets has a uniform contraction ratio below one. A complete finite-menu Bellman proof is therefore possible and supplied in this phase. Passing from every finite menu to a common upper bound over all bases is the unresolved step. Neither taking increasingly many tested bases nor observing stable maxima proves that step.

## Exact-check companion

The verifier tests (1) by independently bounded positive digit sums for small bases. Those finite tests only check the algebra/implementation. The argument above proves the stated all-base inequality.

## A useful further distinction: contraction is uniform even across all bases

For every allowed modular4-AP-free alphabet D in base b>=2, each cyclic run (a,a+1,a+2,a+3) contains at most three allowed positions, counting multiplicities when b<4. Summing over all a modulo b gives

    4|D| <= 3b,
    rho <= 3/4.

Thus the Bellman continuation part has a uniform contraction factor at most3/4 even when the radix is UNBOUNDED. What is not established is a uniform bound on its immediate harmonic reward h_D(a), not the continuation factor.

A concrete sufficient route to the fixed-k=4 conjecture would be a bounded nonnegative function V on[0,1] satisfying, for every allowed radix and alphabet,

    h_D(a)+(1/b)sum_(d in D)V((d+a)/b) <= V(a).       (3)

Unfold (3) on a finite tree. Nonnegativity bounds every truncated regular sum by V(a); monotone convergence bounds the full sum. Stationary constructions are a subfamily, and Walker's digital approximation theorem then gives the unrestricted finite harmonic bound. The uniform3/4 continuation factor also gives the usual contractive control where the global operator is bounded.

Equation(3) is the exact kind of inequality certified on restricted menus in this phase. No bounded V satisfying it over all bases has been found. In particular, the finite-menu proof for B55 must NOT be extrapolated to(3).
