# PR-TAL-009 — Talagrand covers for bounded crossing complexity

Date: 2026-09-16. Grade: complete author-side derivation conditional on PR-TAL-001 (Phase04 laminar theorem). External independent review and historical novelty remain open. This is NOT the unrestricted discrete Talagrand conjecture.

## 1. Setup
Let X be finite. A capacity constraint is (B,r_B), with B subset X. Put
D={S subset X: |S intersect B|<=r_B for every constraint B}.

The crossing graph has one vertex per normalized scope and an edge when two scopes intersect but neither contains the other. Any independent set of scopes is laminar. Let d be any certified proper coloring number and let D_i be the family defined by color i. Then D=intersection_i D_i.

For decreasing E, let E_(k) be unions of k members of E and E^(k)=2^X\E_(k).

PR-TAL-001 proves for laminar D_i under uniform Bernoulli-p, eps_i=1-mu_p(D_i):
(A) if eps_i<=1/4, D_i^(2) has a (p/4)-cover of cost <=(29/20) eps_i/(1-eps_i);
(B) if eps_i<=1/8, D_i^(8) has a p-cover of cost <=(15/7) eps_i/(1-eps_i).

## 2. Intersection-refinement lemma
If S belongs to D_i_(k) for every i=1,...,d, then S belongs to D_(k^d).

Proof. For each i choose S=A_{i,1} union ... union A_{i,k}, A_{i,j} in D_i. For each vector j=(j_1,...,j_d) in [k]^d define C_j=intersection_i A_{i,j_i}. The C_j cover S. Since C_j is a subset of A_{i,j_i} for every i and every D_i is decreasing, C_j lies in every D_i, hence in D. QED.

Thus
D^(k^d) subset union_i D_i^(k).                    (1)

## 3. Heterogeneous probability budgets
If eps_i<=1/4 and
sum_i eps_i/(1-eps_i) <= 10/29,
then unioning the PR-TAL-001(A) covers and using (1) gives total (p/4)-cost <=1/2. Therefore D^(2^d) is (p/4)-small.

If eps_i<=1/8 and
sum_i eps_i/(1-eps_i) <= 7/30,
then D^(8^d) is p-small.

## 4. Corollaries from mu_p(D)
If mu_p(D)>=1-eps, then D subset D_i implies eps_i<=eps. Hence Theorem A applies when
eps <= min(1/4, 10/(29d+10)),
and Theorem B applies when
eps <= min(1/8, 7/(30d+7)).

These improve the Phase04 convenience hypotheses from exponential decay in d to order 1/d. The number of pieces remains exponential in d.

## 5. Bounded crossing degree
If the crossing graph has maximum degree Delta, greedy coloring gives d<=Delta+1. Substitute d=Delta+1 above. Any other certified coloring upper bound may be used. In particular, treewidth t gives d<=t+1.

## 6. Scope
Arbitrary constraint families can have crossing chromatic number growing with |X|, so this does not yield a universal piece count. It also does not show that arbitrary decreasing families have useful low-crossing capacity presentations.

## 7. Novelty boundary
The refinement lemma and coloring observation are elementary. The quantitative combination with PR-TAL-001 is new to this project, but historical originality is unestablished.
