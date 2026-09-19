# PR-TAL-007 — Remove the nonuniform probability loss by reserving two pieces

2026-09-16. Complete author-side proof, historical novelty and external review unestablished. This is a general elementary extension lemma, not the unrestricted Convexity Conjecture.

## 1. Disjoint-coordinate decomposition

Let D be any downset on X with mu_p(D)>1/2 for an arbitrary vector of independent coordinate probabilities. Put H={i:p_i>1/2} and L=X minus H. Let D_H,D_L be the restrictions of D to those coordinate sets (the other coordinates absent).

Then H is the union of two D_H-members. Indeed deleting L preserves membership, so mu_(p_H)(D_H)>=mu_p(D)>1/2; monotonicity gives mu_(1/2)(D_H)>1/2. In a uniform random two-color partition of H, each color has that latter law. The probability that both colors belong to D_H is at least1-2[1-mu_(1/2)(D_H)]>0. A suitable partition exists. Its parts, and all their subsets, also belong to D.

Therefore for every k>=1,

    D^(k+2) subset {S subset X: S intersect L belongs to (D_L)^(k)}.  (1)

Every generator cover for the right-hand light obstruction, viewed as subsets of X, is a cover for D^(k+2) at EXACTLY the same product cost. No probability change is necessary on heavy coordinates because the generators contain none of them. This is an inclusion, not an equality.

Unlike the common-refinement construction for intersecting downsets on the same ground set, the pieces here add: the heavy and light ground sets are disjoint. No compatible-coloring assumption is hidden.

## 2. Consequences of the new restricted theorems

All probabilities below are arbitrary independent vectors in[0,1]^X.

(A) Laminar capacities: if mu_p(D)>=6/7, then D^(7) is p-small. The cost bound from the light five-copy argument is at most (16/15)*(1/6)=8/45.

(B) Residual coordinate incidence<=d>=1: with K=4d+3, mu_p(D)>=1-1/K implies D^(K) is p-small, cost at most max(1,3d/5)/(4d+2)<=1/6.

(C) The same incidence class, K=d+3: mu_p(D)>=1-1/(2K) implies D^(K) is(p/4)-small, cost at most3d/[5(2d+5)]<3/10.

(D) Bipartite vertex-capacity families: if mu_p(D)>=6/7, then D^(7) is p-small, cost at most(6/5)*(1/6)=1/5.

Proof. Restriction to L preserves each respective constraint class and does not increase coordinate incidence. mu_(p_L)(D_L)>=mu_p(D); all light probabilities are<=1/2. Apply the light theorem with its epsilon-dependent bound and then(1). Heavy elements consume two additional D-members. The integers in A--D refer to the new total number of pieces, so the displayed good-event assumptions match them explicitly.

For uniform p the sharper no-overhead theorems remain preferable: if p<=1/2 there is no heavy part, and if p>=1/2 the obstruction is already empty. This lemma is a separate strengthening of the nonuniform extensions, not a replacement for the uniform statements.

## 3. Boundary

One cannot declare the whole heavy part admissible merely from high probability. For D={S subset[5]:|S|<=4} and p_i=3/5, mu_p(D)=1-(3/5)^5>4/5 but H=[5] is not in D. Two pieces suffice. Also, coordinate independence is still needed to identify the restricted product probabilities and the random-partition comparison.

The existence proof need not furnish an efficient heavy-part coloring for arbitrary downsets. Small computational companions find it by exhaustive search. No polynomial-time algorithm for an unspecified oracle downset is claimed.
