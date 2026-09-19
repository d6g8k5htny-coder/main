# PR-AP-014 — Greedy prefixes, nongreedy maximizers, and noncommuting limits

2026-09-16. Complete author-side derivation; external review and novelty unresolved. The ternary greedy description and elementary tail comparison are classical ingredients. No original prize closure.

## 1. Every sufficiently strongly weighted optimizer has a greedy head

Let G_k be the usual greedy k-free set: scan positive integers in increasing order and include an integer exactly when it creates no k-AP with already included elements.

**Theorem.** If sigma>N+1 and A maximizes H_sigma over k-free sets, then

    A intersect [1,N]=G_k intersect [1,N].             (1)

The same conclusion holds for finite-horizon maximizers whose horizon is at least N.

Proof. Suppose n<=N is the first difference. A cannot include a greedy-excluded n, because the previously selected prefix is the same and already certifies a forbidden AP. Thus G_k includes n and A excludes it. Replace A by its prefix through n-1, together with n. That finite competitor is k-free. The gain is at least

    n^(-sigma)-sum_(j>n)j^(-sigma)
      >= n^(-sigma)-n^(1-sigma)/(sigma-1)>0,

since sigma-1>N>=n. This contradicts maximality. A finite-horizon tail is even smaller. QED.

This is prefix stabilization, not global greedy optimality for any fixed sigma.

## 2. The three-term greedy set

Let C be the nonnegative integers with only0 and1 as base-3 digits. Then G_3=1+C.

AP-freeness follows by the usual lowest nonconstant residue test. To prove greediness, let z=n-1 be excluded by the digit rule. At each ternary digit choose digits a_i,b_i according to

    z_i=0 ->(0,0); z_i=1 ->(1,1); z_i=2 ->(0,1).

There are no carries in z=2b-a. At least one2 occurs, so a<b<z. Both a,b belong to C, hence earlier accepted numbers a+1,b+1 make n=z+1 a forbidden AP completion. This proves the greedy description inductively.

Its counting dimension is log_3 2, as is immediate from counts at powers of3 and monotonicity between those scales. Phase03's sparse-tail replacement implies that G_3 is NOT an H_sigma maximizer for any fixed finite sigma>1.

## 3. Two limits disagree

Choose any maximizing set A_sigma for each sigma>1. Equation (1) implies coordinatewise convergence A_sigma->G_3 as sigma->infinity. On the other hand PR-AP-013 gives full counting dimension1 for every fixed sigma.

Therefore

    lim_(sigma->infinity) lim_(X->infinity)
          log A_sigma(X)/log X =1,

whereas

    lim_(X->infinity) lim_(sigma->infinity)
          log A_sigma(X)/log X =log_3 2.             (2)

This explains why optimal finite or strongly weighted heads can look greedy even though no full optimizer is greedy or automatic. It is not a thermodynamic limit claim and does not involve a probabilistic distribution on optimizers.

## 4. Approaching the harmonic endpoint

For every k, M_(k,sigma) increases to M_k in the extended nonnegative reals as sigma decreases to1. Indeed it is bounded above by M_k, while any finite k-free witness F has H_sigma(F)->H_1(F); taking suprema proves the reverse limiting inequality.

Thus a uniform bound on M_(k,sigma) for sigma close to1 is equivalent to finiteness of the harmonic supremum. Merely knowing each sigma>1 problem has an optimizer is not such a bound.

If M_k<infinity, Phase03's uniform harmonic-tail theorem and product compactness show that every product-limit point of A_sigma as sigma decreases to1 is a harmonic maximizer: the uniform harmonic tail controls the truncation error, and on finite prefixes the objectives and indicators converge. For k>=4 this conclusion is conditional on the unresolved premise M_k<infinity.
