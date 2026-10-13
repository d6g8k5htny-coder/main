# P14-B — Replace resource-row count by an exact fractional cover

Author-side complete derivation; no independent review or historical novelty determination. The fractional covering LP and its dual are standard. The claim here is their exact use as a certificate for P14-A/P13's probability-proportional integral-cover construction.

## 1. Normalize the actual coefficient profiles

Let D={U:sum_(i in U)b_(ji)<1 for every j<=d}, with b_(ji)>=0 finite. Remove true singleton forbidders J={i:max_j b_(ji)>=1}, then irrelevant columns with maximum zero. For each remaining coordinate put

    a_i=max_j b_(ji),       T_(ji)=b_(ji)/a_i.

Then 0<a_i<1, 0<=T_(ji)<=1, and each column of T has maximum1.

A fractional row cover is lambda_j>=0 satisfying

    sum_j lambda_j T_(ji)>=1      for every remaining i.     (FC)

Set tau=sum_j lambda_j. If any relevant coordinate remains, tau>=1. The all-ones lambda is feasible, so the minimum tau_* exists and tau_*<=d. Minimizing is optional: ANY exactly checked feasible lambda gives the stated theorem. In particular a floating solver's report of success is not a certificate.

## 2. The scalar sandwich

For any U, a(U)<1 implies every b_j(U)<=a(U)<1, so U is good. Conversely, if U is good,

    a(U)<=sum_j lambda_j b_j(U)<sum_j lambda_j=tau.

Strictness is valid because at least one lambda_j>0 and every b_j(U)<1. Thus (S1)/(S2) hold with kappa=tau.

P14-A proves, at ALL product vectors and all c_i<=phi(p_i),

    covercost_c(D^(ceil(408 tau))) <=phi(1-mu_p(D)).         (1)

Singletons are dealt with as in P14-A. If no relevant nonsingleton coordinates remain, only singleton generators are needed; assign any palette >=1 rather than a nonexistent fractional optimization problem.

The low-Q inflated-price version is K=ceil(8 tau(ceil(12A^2)+3)), cost<=Q_R/2 on the restricted family.

## 3. A distribution formulation

Equivalently, let pi be a probability distribution over rows and assume

    sum_j pi_j T_(ji)>=beta>0       for every i.             (2)

Then lambda=pi/beta is feasible with tau=1/beta, so K=ceil(408/beta). The number of rows can be arbitrarily large. The average must be a proved coordinatewise bound, not a sample average.

Conversely a feasible lambda with mass tau gives (2) for pi=lambda/tau and beta=1/tau. Thus the optimal beta is exactly 1/tau_*. A maximum exists on the finite probability simplex; this equivalence does not require importing a duality theorem.

## 4. Exact lower/upper certificates

If y_i>=0 and sum_i T_(ji)y_i<=1 for each row, then for any feasible lambda,

    sum_i y_i <=sum_i y_i sum_j lambda_j T_(ji)
              =sum_j lambda_j sum_i T_(ji)y_i <=sum_j lambda_j.

Thus any matching pair with sum y=sum lambda proves optimality EXACTLY by weak duality. We do not need to trust a solver tolerance or use the assertion 'strong duality says so' without checking feasibility. Rational inputs/certificates can be checked using fractions; the mathematical theorem also admits arbitrary real certificates.

## 5. Elementary behavior

Exact duplicate rows leave tau_* unchanged (combine their weights). Deleting a coefficientwise dominated row does not increase tau_* (shift its weight to a dominating row). The total count d may change without changing tau_*.

Multiplying one column by a positive scalar does not change T, so the structural parameter is a property of normalized profiles, not of coefficient magnitudes. The singleton-removal boundary must nevertheless be re-evaluated after such a scaling.

Deleting coordinates cannot increase tau_*. A certificate computed on a sampled or smaller coordinate set, however, cannot be used on missing coordinates without checking(FC) there too.

## 6. Disjoint-component and hierarchy versions

If the resource system splits into disjoint coordinate components, apply(1) componentwise with one common palette K>=max_j ceil(408 tau_j). Each component obstruction is covered; their union covers the global obstruction. Cover costs sum, while independent good probabilities multiply:

    sum_j phi(Q_j)<=sum_j[-log(1-Q_j)]=-log(mu_p(D)).

Capping at1 gives the same invariant. Thus the MAXIMUM componentwise cover width, not their sum, is sufficient. Any row crossing two proposed components invalidates that factorization.

At nodes of a finite read-once hierarchy, suppose each normalized coefficient system has an exact feasible fractional cover of mass<=T_*. For K=ceil(408 T_*), P14-A supplies the cover at CHILD COVER PRICES c_i<=phi(q_i). Exact disjoint substitution, restated in P14-E, gives

    covercost_p(D_root^(K))<=min(1,-log(mu_p(D_root))).

Row count, depth, coefficient ratio, and original witness rank need not be bounded. T_* is still a genuine hypothesis; a universal bound on it would be false in general.

## 7. Directed repair of an approximate rational proposal

For a proposed nonnegative rational lambda with

    gamma=min_i sum_j lambda_j T_(ji)>0,

lambda/gamma is an EXACT feasible primal cover. Its mass(sum lambda)/gamma is therefore a valid upper bound, whether or not the proposal was originally feasible. No solver tolerance enters this statement.

Similarly, for nonnegative y and eta=max_j sum_i T_(ji)y_i>0, y/eta is exactly dual-feasible. It gives a lower bound(sum y)/eta. The zero dual is always valid and needs no division. Equality of the repaired objectives proves optimality; a nonzero gap must be reported, not rounded away.

The CLI accepts explicit rational strings and rejects binary floating-point input. Its optional --repair flag performs this normalization openly and emits both factors and both bounds. A proposal with gamma=0 is rejected: some coordinate has no positive coverage. This is a practical certificate conversion, not a claim that a numerical optimum has been independently proved.
