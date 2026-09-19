# P15-A — Exact crosswalk to the established critical threshold value

Status: author-side proof of the translation, not a claim of a new parameter. The critical threshold value and its winning/losing optimization are established in Gvozdeva–Hemaspaandra–Slinko and Freixas–Kurz, and in Hof–Kern–Kurz–Pashkovich–Paulusma, arXiv:1810.08841v1, §1. We use the latter inspected source for the exact normalization. Our contribution here is the strict-endpoint translation to the earlier P14 certificate and an exact checker.

Let D be a proper decreasing family on finite X, containing every singleton and the empty set. Its nonempty minimal forbidden family H has edge sizes at least two. Write

    alpha(D) = min { max_{U in D} w(U) : w_i>=0, w(e)>=1 for every e in H }.

This is the source's critical threshold value, with D as the losing family and D^c as the winning family. H constraints suffice by monotonicity. An optimum exists: clipping each coordinate of a feasible w at one preserves all forbidden-set inequalities and does not increase good-set weights. Thus optimization may be restricted to the compact cube [0,1]^X. Irrelevant coordinates are permitted.

## A1. Which strict scalar widths are admissible?

For kappa>=1, a P14 scalar sandwich exists if and only if

    kappa > alpha(D).                                      (A1)

Here a sandwich means 0<=a_i<1, a(U)<1 implies U in D, and U in D implies a(U)<kappa, for every U.

Necessity: the first implication says a(e)>=1 for each forbidden e. The finite maximum of a(U) over good U is strictly below kappa, and alpha is no larger than that maximum.

Sufficiency: take an optimal feasible w and clip at one. If alpha<1 and kappa=1, good singleton constraints already give w_i<=alpha<1 and the desired sandwich. In all other cases take

    a_i=(1-theta) w_i + theta/2,

with 0<theta<1 sufficiently small. Each a_i<1. Both w and the constant vector 1/2 assign weight at least one to every forbidden set, because forbidden sets have at least two elements. Therefore a is feasible on every forbidden set. For good U,

    a(U) <= (1-theta) alpha + theta |X|/2 < kappa

when theta is small enough. This proves (A1), including alpha=1. In particular the infimum scalar width is max(1,alpha), but when alpha>=1 the strict endpoint kappa=alpha is NOT admissible.

If an explicit rational feasible w has good-set maximum beta, use it instead of an unknown optimum. For beta>=1 set K=floor(408 beta)+1, kappa=K/408 and theta=(kappa-beta)/(|X|+1). This is in(0,1), and beta+theta |X|/2<kappa. For beta<1 use kappa=1 and w itself. Thus an exactly proved upper bound beta gives

    K=max(408, floor(408 beta)+1).                         (A2)

P14-A then gives the composable potential bound at K colors. This is a consequence of the preceding author-side theorem, not an external acceptance of it.

The case D=2^X is separate: there are no obstructions, and one color with the empty generator FAMILY has zero cost.

## A2. Exact lower certificates

Let mu_e>=0 be weights on forbidden sets. Let nu_U>=0 be weights on good sets with sum nu=1. If, for every coordinate i,

    sum_{e contains i} mu_e <= sum_{U contains i} nu_U,     (A3)

then alpha(D)>=sum mu. For any feasible w,

    sum mu <= sum_e mu_e w(e)
            <= sum_U nu_U w(U) <= max_{U in D}w(U).

A feasible primal w and such a dual pair with matching objectives therefore prove alpha exactly, using weak duality only. This does not require an optimizer's floating-point success status. All support sets must really be good/forbidden in the original family.

## A3. Known examples and provenance consequences

- For the independent sets of K_(m,m), alpha=m/2. Primal: w_i=1/2. Dual: a perfect matching with mu=1/2 on each matching edge; nu=1/2 on each complete bipartition side. This is the same obstruction already observed in P14, now identified as a critical-threshold example. It is not a new complexity parameter.
- For a complete r-uniform forbidden family on n>=r vertices, alpha=(r-1)/r. Uniform weights 1/r and uniform good(r-1)-set / forbidden r-set distributions give matching certificates.
- The earlier P14 heavy/tail example has alpha=1, not merely a feasible scalar width just above one. Its two permitted complementary tail s-sets and two forbidden complementary tail s-sets give a balanced dual of value one. The supplied scalar envelope has maximal good weight one. Thus 409 is the smallest palette delivered by the strict P14 formula from this exact alpha, not a proven optimal number of colors for the actual probabilistic problem.

The published general bound alpha<=n/4 is established prior work. Feeding that dimension-dependent bound into (A2) gives no universal-palette result and, for large n, is weaker than trivially assigning different colors to all allowed singletons. We do not promote that known bound into a prize advance.
