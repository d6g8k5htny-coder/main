# CORE-PALETTE-001 — Conditional core deletion for exact-cell covers

Status: author-side elementary corollary candidate. No novelty claim, organizationally independent review, prize closure, or promotion of another result. This proof rederives its probabilistic input and does not assume P12-F has been accepted.

Scope: an additional sufficient criterion for finite decreasing families. This does not assert that a suitable core always exists. It does not establish the active P13 all-probability weighted-gate target.

## 1. Definitions and statement

Let I be finite and A⊆2^I a decreasing family containing ∅. Let K≥2 be an integer. Write A^(K) for sets not expressible as unions of K members of A. A generator family G covers B when each U∈B contains some g∈G. Its cost at nonnegative prices c is Σ_(g∈G) ∏_(i∈g)c_i; the empty generator costs 1.

Let q_i∈[0,1] be independent Bernoulli selection probabilities, μ_q(A)=P(X_q∈A), and φ(q)=min{1,−log(1−q)}, with φ(1)=1. Assume 0≤c_i≤φ(q_i).

Separate singleton minimal forbidden vertices J. Let H be the remaining minimal forbidden sets; each has size at least two. Put W=∪_(e∈H)e. Minimality gives J∩W=∅. Other vertices are irrelevant.

Choose S⊆W and an integer t≥0. Require a proper t-coloring κ of H[S]: every edge of H wholly contained in S must use at least two colors. For t=0 require S=∅; for nonempty S require t≥1. Set L=K−t and require L≥2.

The residual hypergraph is INDUCED:
H_R={e∈H:e⊆W\S}.
No crossing edge is shortened. On the union of residual edges let W_1,…,W_m be its connected components, where vertices are adjacent if they belong to a common residual edge. Let H_j be the edges of component j, r_j=min_(e∈H_j)|e|≥2, π_j=∏_(i∈W_j)(1−q_i), and Q_j=P(∃e∈H_j:e⊆X_q).

Assume, for every nonempty residual component,

    L^(r_j−1) π_j ≥ 1.                                      (C)

If H_R is empty, (C) is vacuous.

**Conclusion.** There is a cover G of A^(K) with

    cost_c(G) ≤ min{1, Σ_(i∈J)c_i + Σ_j Q_j}
              ≤ φ(1−μ_q(A)).                               (1)

Thus highly active core vertices need not enter (C). The core's t colors are genuinely additional to the L residual colors.

## 2. Exact-cell cost estimate, rederived

Fix residual component j. Condition (C) implies π_j>0, hence q_i<1 for all its vertices. Set o_i=q_i/(1−q_i). The events {X_q∩W_j=e}, e∈H_j, are distinct exact configurations, hence disjoint, and each implies the component failure event. Therefore

    π_j Σ_(e∈H_j)∏_(i∈e)o_i ≤ Q_j.                          (2)

For 0≤q<1, −log(1−q)=∫_0^q(1−u)^−1du ≤ q/(1−q).
Consequently c_i≤φ(q_i)≤o_i, so

    Σ_(e∈H_j)∏_(i∈e)c_i ≤ Q_j/π_j.                          (3)

This remains true when some q_i=0. Zero-cost generators cannot be dropped on probabilistic grounds: covering is a set-theoretic requirement.

Independently color the vertices of W_j uniformly with L residual colors. Edge e is monochromatic with probability L^(1−|e|)≤L^(1−r_j). The expected total cost of monochromatic edges is at most

    L^(1−r_j) Q_j/π_j ≤ Q_j.                                (4)

A finite average has some coloring attaining at most its average. Fix one such coloring for each residual component. Reuse the SAME L residual color names across components; no residual witness crosses them. Assign any residual color to vertices of W\S outside residual edges.

## 3. Cover inclusion

Use a palette disjoint from the residual colors for κ on S. Every H-edge entirely in S is nonmonochromatic by hypothesis. Every H-edge meeting both S and W\S is nonmonochromatic because the palettes are disjoint. Each remaining edge lies in H_R.

Let G_0 contain all singleton generators {i}, i∈J, together with all monochromatic H_R edges from the fixed residual colorings. Its cost is at most U=Σ_(i∈J)c_i+Σ_jQ_j.

If a set B contains no generator of G_0, it contains no singleton-forbidden vertex. Restrict the fixed coloring to B∩W and color its irrelevant vertices arbitrarily with one of the K available colors. No color class can contain a minimal forbidden set: singleton ones are absent; core and crossing ones are nonmonochromatic; a residual monochromatic edge would be a generator contained in B. Thus all K color classes belong to A, including empty classes. Their union is B. Contrapositively, G_0 covers A^(K).

This also explains the coloring equivalence: if B is a union of K A-members, assign each vertex to one containing member. The resulting disjoint classes are subsets of A-members and hence in A because A is decreasing. No equivalence with an overlapping independent-coordinate model is invoked.

Taking G_0 or {∅}, whichever has smaller cost, gives the first inequality of (1).

## 4. Full probability bound, including endpoints

If μ_q(A)=0, φ(1−μ_q(A))=1 and the trivial cap suffices. Otherwise every q_i<1 for i∈J. Success of A entails absence of singleton vertices and absence of each residual witness. These necessary events have disjoint coordinate supports, so

    μ_q(A) ≤ ∏_(i∈J)(1−q_i) ∏_j(1−Q_j).                    (5)

Notice the direction: core and crossing restrictions can only further reduce success probability. Core probabilities may equal one.

Taking logarithms, using c_i≤−log(1−q_i) and Q_j≤−log(1−Q_j), gives

    U ≤ Σ_(i∈J)−log(1−q_i)+Σ_j−log(1−Q_j)
      ≤ −log μ_q(A).                                        (6)

Hence min(1,U)≤min(1,−log μ_q(A)), proving (1). All residual Q_j<1 since π_j>0. If μ_q(A)=1, (5) and (6) force U=0; the argument still retains any necessary zero-cost generators. If a singleton q_i=1, μ_q(A)=0 and the earlier cap case applies.

## 5. A strict extension of the original exact-cell criterion

For any K≥3 and r≥3, take N=K(r−1)+2 vertices. Let H consist of ALL r-subsets except one distinguished r-set E_0. Choose a vertex s∉E_0 as core S={s}, t=1, and put n=N−1. Set

    q_s=1−1/(100 K^(r−1)),  q_i=1/(100n) for i≠s.

H is connected: every pair belongs to some r-edge other than E_0, since N≥8. Its minimum rank is r. Its original whole-component empty-mass criterion fails:

    K^(r−1)∏_(i∈I)(1−q_i)
    = (1/100)(1−1/(100n))^n < 1.

The core is properly 1-colored because it contains no edge. The residual hypergraph remains connected and has minimum rank r. By the union bound,

    π_R=(1−1/(100n))^n ≥ 99/100,
    (K−1)^(r−1)π_R ≥ 4·99/100 > 1.

Thus (C) holds. This is a strict extension of the ORIGINAL CRITERION, not a claim of coverage beyond every existing project primitive or all prior literature.

The K-obstruction is nonempty. Every good set has size at most r, and its sole possible r-member is E_0; every larger set contains a forbidden r-subset. In a disjoint partition into K good classes at most one class can equal E_0. Hence any union of K good sets has size at most K(r−1)+1<N, and the full ground set is obstructed.

The probability bound is nontrivial. Failure is contained in the union of “core selected and at least r−1 residual vertices selected” and “at least r residual vertices selected.” With ε=1/(100n),

    Q ≤ C(n,r−1) ε^(r−1)+C(n,r) ε^r
      ≤ 100^(−(r−1))/(r−1)! + 100^(−r)/r!
      < 1/1000.

Therefore φ(Q)<1. This is not a cap-only example, and its witness system is not the complete unweighted threshold because E_0 is allowed while all other r-sets are forbidden.

A small exact instance is K=3,r=3,N=8,E_0={1,2,3},s=0,
q_s=899/900 and q_i=1/700 otherwise. The accompanying independent finite checker checks this instance and the cover inclusion explicitly.

## 6. Credit, limits, and custody

The deterministic operation is ordinary disjoint-palette concatenation. For conventional induced-hypergraph and weak-coloring definitions see Schweser–Stiebitz, arXiv:1804.04894v2 §§1.1,1.3; induced deletion must be distinguished from shrinking. Generator-cost/smallness language is standard; see Frankston–Kahn–Park, arXiv:2105.10905v1 §§1–2. No theorem from either source supplies a hidden probability hypothesis here.

The core criterion extends P12-F's sufficient condition by an explicit reduction and reproof. It does not solve connected high-activation systems for which no useful core and residual certificate exist, the unrestricted constant-palette conjecture, or P13's weighted-gate objective. Finite checks supplement rather than prove unbounded quantifiers.

Exact P12-F source: Drive1q1xNZ6yZINV53FNb5CHfokH2iF8NOChC; 7112B SHA2560510460cf04086bd994b328f0917f226542d545965a6df8f039d956fbe816b1c. Prior scoped reconnaissance: RECON_SEPARATOR_20260917_v1_0.md, SHA2569161e7b83ce1d8b1023871be2d19dc8c8278aa23891f0dcf683e6ae777584206; custody verified by root before this derivation.

