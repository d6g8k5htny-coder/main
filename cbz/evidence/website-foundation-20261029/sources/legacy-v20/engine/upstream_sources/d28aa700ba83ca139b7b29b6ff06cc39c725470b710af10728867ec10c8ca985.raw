# Even capacities: read-two overlap without bipartiteness

Object: OA-P15-EVEN-READ-TWO-20260928-v1.
Author: OpenAI / ChatGPT, 28 September 2026.
Disposition: AUTHOR-SIDE CANDIDATE; independent analytic review required.
Scientific effect NONE. This is an explicit alternative to the bipartite
hypothesis of PROOF.md, not permission to omit that hypothesis for odd capacities.

## E1. Exact coloring under the alternative structural hypothesis

Keep the finite original ground set X, covering blocks X_i and capacity downset
D of (O1), but replace bipartiteness by:

    every original coordinate belongs to at most two blocks;
    every capacity a_i is a POSITIVE EVEN integer.                    (E1)

**Theorem E1.** For all U subset X,

    chi_D(U)=max_i ceil(|U intersect X_i|/a_i).                         (E2)

*Proof.* Necessity is unchanged. Write k for the right side and assume U is
nonempty. Represent coordinates by edges of the possibly NONBIPARTITE block
incidence multigraph: a shared coordinate joins its two distinct block vertices;
a private coordinate joins its block to a private auxiliary vertex.

At each real vertex i, split incident edges as evenly as possible among a_i/2
auxiliary vertices. Each new vertex has degree at most
ceil(2|U intersect X_i|/a_i)<=2k. Private auxiliary vertices have degree one.
Each original edge keeps its one original-coordinate label.

The number of odd-degree vertices is even. Pair them arbitrarily and add one
auxiliary edge per pair. Every odd degree was at most 2k-1, so every augmented
degree is even and at most 2k. Each component has an Euler orientation, with
indegree=outdegree=degree/2<=k at every vertex. Form the bipartite directed
incidence graph with one outgoing and one incoming copy of each vertex; an
oriented edge becomes one edge from its tail's outgoing copy to its head's
incoming copy. Maximum degree is at most k. The matching decomposition in the
proof of (O2) colors this bipartite graph with at most k colors.

In the undirected split graph, a color has at most one outgoing and one incoming
edge at each split vertex, thus at most two incident edges. Remove the auxiliary
edges and combine the a_i/2 split vertices back into block i: at most a_i
original coordinates of each color lie in X_i. This is a coloring of ORIGINAL
coordinates, not independent copies of them. It proves (E2). QED.

For |X_i|=a_i d_i+1, K=max_i d_i, the exact active-full-block obstruction (O4)
and chi_D(X)=K+1 follow from (E2) by the same elementary argument. This holds
even on an odd cycle of blocks. It does not extend to the odd-capacity triangle
counterexample of PROOF.md, where capacities are one.

## E2. A self-contained read-q hazard inequality

The two-sided Cauchy-Schwarz proof of (O9) used a bipartition. It cannot be
reused unchanged for an odd cycle. We prove the necessary replacement rather
than assuming independence of the overlapping events.

**Lemma E2.** Let mu be a finite product probability measure. Let A_i be events
depending on coordinate sets S_i, each coordinate lying in at most q of those
sets. If D is contained in every A_i, then

    sum_i [-log mu(A_i)] <= q[-log mu(D)].                            (E3)

The inequality uses extended hazards; mu(D)=0 is immediate.

*Proof for mu(D)>0.* Set nu=mu(.|D). Denote relative entropy by
D_KL(nu||mu), with the usual zero-mass conventions. Then
D_KL(nu||mu)=-log mu(D). Each marginal nu_{S_i} is supported on A_i, whence

    D_KL(nu_{S_i}||mu_{S_i})
       = D_KL(nu_{S_i}||mu_{S_i}(.|A_i)) - log mu(A_i)
       >= -log mu(A_i).                                               (E4)

Order all coordinates. The chain rule and product form of mu give

    D_KL(nu_S||mu_S)
      = sum_{v in S} E_nu D_KL(nu(v | earlier coordinates in S)||mu_v).

Conditioning on only the earlier coordinates in S gives a mixture of the laws
conditioned on all earlier coordinates. Convexity of relative entropy in its
first argument (the finite log-sum inequality) therefore bounds each summand by

    c_v := E_nu D_KL(nu(v | all earlier coordinates)||mu_v) >= 0.

Thus D_KL(nu_S||mu_S)<=sum_{v in S} c_v. Summing over S_i and using read-q
incidence gives

    sum_i D_KL(nu_{S_i}||mu_{S_i}) <= q sum_v c_v
                                  = q D_KL(nu||mu).

Combine with (E4). Deterministic coordinates (probability zero or one) have zero
conditional divergence and can also be removed from the finite product support.
No absolute-continuity problem arises because nu is a conditioning of mu. QED.

This is the entropy form of a classical read-q/Finner-type inequality. Its proof
is included; no result about arbitrary dependent measures is asserted. It
applies to just the active blocks in (O4), as required for charging generators.

## E3. Full transformed prices and a sub-2/3 factor

Assume additionally |X_i|=a_i d_i+1, with d_i positive integers and K=max_i d_i>=4.
For an active block a_i is even, hence a_i>=2. The Chernoff calculation in Lemma
O3 gives its reference probability at p_*=1-exp(-1) at most 25/512, uniformly
in ALL such capacities. Hence its reference hazard is at least

    h_even=log(512/25)>3.                                              (E5)

The last inequality has a short rational certificate:

    e < 87/32,
    512/25 - (87/32)^3 = 314641/819200 > 0.

The separate-concavity transfer (O5) yields price(X_i)<=h_i/h_even. Lemma E2 with
q=2 yields sum_active h_i<=2[-log mu_p(D)]. Since the active full-block cover is
proved by Theorem E1, summing and applying the price-one empty-generator cap gives

    covercost_c(O_K(D))
       <= min(1, [2/log(512/25)] [-log mu_p(D)])
       <= min(1, (2/3)[-log mu_p(D)]).                                (E6)

Here 2/log(512/25)<2/3; neither factor is claimed optimal. Probabilities are the
original independent Bernoulli probabilities in [0,1], and all prices
0<=c_v<=min(1,-log(1-p_v)) are admitted. The zero/infinite-hazard cases are handled
as in PROOF.md. The global palette remains K, not 2K.

This supplies a second justified structural class: arbitrary read-two block
incidence with EVEN capacities. The first class permits arbitrary positive
capacities but requires bipartiteness. Their hypotheses should not be mixed or
silently dropped. Inactive smaller demands, including demand one, are harmless
to the generator budget in both classes, but (E1) still requires their capacities
to be even for the displayed coloring proof.

## E4. Verification and literature boundary

The standard-library implementation follows the displayed construction:
capacity-two splitting, pairing odd vertices, Euler orientation, a bipartite
matching coloring, and one final color per original coordinate. It is checked
against an independent partition oracle on small nonbipartite examples with
heterogeneous even capacities. A triangle with four shared edges per side and
one private coordinate per block has K=4: the twelve shared coordinates are
4-colorable, while the full fifteen-coordinate set needs five colors.

The read-two hazard inequality is tested on exact heterogeneous Bernoulli grids
on triangle incidence. Finite grids do not prove (E3); its entropy argument does.

Aboulker-Aubian-Huang, arXiv:2201.11548, studies defective edge coloring and states
the uniform even-capacity phenomenon. This note does not claim invention of that
classical structural mechanism. The variable-capacity construction and the
original-coordinate price/cover assembly are written out here; no unpublished
original checker from another packet is used. Independent mathematical review
and any formalization are still separate obligations.
