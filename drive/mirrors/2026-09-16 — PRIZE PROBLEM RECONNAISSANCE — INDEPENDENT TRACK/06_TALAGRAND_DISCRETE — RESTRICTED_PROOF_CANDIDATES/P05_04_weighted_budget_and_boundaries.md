# PR-TAL-006 — A weighted overlap certificate and exact boundaries

2026-09-16. Author-side complete derivation using standard fractional Holder. No external review or novelty claim.

## 1. Weighted overlap charging

For positive scopes B let q_B be their failure probabilities and suppose the local witness cost w_B satisfies w_B<=a_B q_B, with nonnegative certified coefficients a_B. Define

    Lambda=max_(x in X) sum_(B contains x)a_B.

If Lambda>0, theta_B=a_B/Lambda is a fractional matching of the scope hypergraph. The product inequality of PR-TAL-004 gives

    sum_B a_B q_B<=Lambda*(-log mu_p(D_R)).           (1)

Thus a cover with these certified per-scope bounds has total cost at most

    l0/L + Lambda*l1 <=max(1/L,Lambda)*[-log mu_p(D)],

after the zero-scope extraction, with L the probability dilution factor. Lambda is a weighted incidence load, not the number of scopes. For five-copy witnesses the rank-dependent a_r in PR-TAL-003 decay geometrically. This can make(1) much sharper than replacing all of them by3/5 and multiplying by the maximum incidence.

This is an a posteriori CERTIFICATE: a decomposition/coverage theorem still has to show that the witness family covers the desired D^(K). A small Lambda alone does not prove coverage. Conversely good colorability alone does not prove smallness.

## 2. Variable expansion thresholds

Choose integers t_B>=2 separately for each positive scope. If |S intersect B|<=t_B*r_B for every B, the same greedy argument partitions S into

    K=1+max_x sum_(B contains x)(t_B-1)

admissible parts. Thus witnesses(t_B*r_B+1)-subsets provide a valid cover of D^(K). Combine any certified block estimates with(1). The exact input certificate must bind BOTH loads: the coloring load K and the probability load Lambda. They are different quantities and must not be substituted for one another.

This does not create universal constants for general downsets; either load may grow without bound.

## 3. Why dropping read-d is false

For two scopes {a,b} and {a,c}, capacity1, independent Bernoulli(1/2) coordinates, each successful constraint has probability3/4, but their joint probability is5/8>9/16. Treating overlapping constraints as independent would assert the false reverse inequality. The correct read2 comparison is (5/8)^2<=(3/4)^2.

One can make the naive probability charge arbitrarily bad even with positive capacities and no duplicate or contained scopes. Take scopes {a,b,y_i}, each with capacity1, for i=1,...,m. Use p_a=p_b=1/4 and p_(y_i)=1/(100m^2). Every q_i>=1/16, so sum_i q_i>=m/16. But the event a,b not both selected has probability15/16, and by a union bound the extra y-failures cost at most1/(100m). Hence

    mu_p(D)>=15/16-1/(100m)>9/10.

The aggregate individual failures can grow while the actual good event retains a fixed high probability. This refutes a multiplicity-free sum(q_i) bound without a structural hypothesis. It does NOT refute Talagrand: better witnesses can charge the shared cause efficiently.

## 4. Four separate scope boundaries

- Coordinate independence is required for the probability budget; scope-event independence is not assumed.
- Bounded coordinate incidence is NOT bounded crossing-graph chromatic number. Complete-graph star scopes have incidence2 but arbitrarily high crossing chromatic number.
- The exact k-capacity-to-k-pieces identity needs laminar or bipartite structure. General bounded incidence supplies a weaker inclusion and more colors.
- All-dimensional at fixed d is NOT universal in d. The original prize formulation requires constants independent of all structural complexity.

No q0 Gaussian/Palm theorem is consumed. Transfer is the decomposition-first and exact-certificate architecture, not a mathematical implication from persistence topology.
