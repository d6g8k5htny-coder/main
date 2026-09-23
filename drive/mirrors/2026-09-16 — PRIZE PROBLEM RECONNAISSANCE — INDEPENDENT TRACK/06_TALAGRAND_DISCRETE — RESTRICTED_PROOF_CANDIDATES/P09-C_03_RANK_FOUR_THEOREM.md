# P09-C — Rank at most four, 6,400 pieces, dilution 8

**Author-side theorem candidate.** External review and novelty unresolved. This is not the unrestricted Talagrand conjecture. The proof is self-contained using P09-A and P09-B in this package.

## Statement

Let D be a decreasing family on finite X whose inclusion-minimal forbidden sets have size at most four. Let p_v in (0,1) be independent inclusion probabilities. If

\[
\mu_{\mathbf p}(D)\ge3/4,
\]

then D^(6400) has a cover at coordinate probabilities p_v/8 of cost strictly less than

\[
\boxed{
\frac{845318147}{1694515200}
=\frac12-\frac{1939453}{1694515200}<\frac12.}
\]

In particular, for uniform p, D^(6400) is (p/8)-small. The number of coordinates, number of constraints, degrees, and codegrees are unrestricted.

## 1. Extract cores by ACTUAL completion probability

Let H be the minimal-forbidden antichain, H_4 its four-element edges, and set q_v=1-sqrt(1-p_v). Then two independent q-samples have union law p and q_v>=p_v/2.

Form a family K consisting of:

1. every original edge of size at most three;
2. every nonempty proper subset A of an edge of H_4 whose link H_4(A) has q-hit probability greater than 3/4.

Replace K by its inclusion-minimal members. This does not change its hit event. Each retained core has its own completion certificate. Let F contain precisely those edges of H_4 containing NO member of K.

This is a finite construction. For rational q the link probabilities can be computed exactly by finite enumeration. Real q is also well-defined mathematically, and a choice of K is not claimed efficiently computable from approximate inputs.

## 2. High cores reduce to the robust rank-three theorem

Choose a first contained core from the first q-sample according to a deterministic total order. If it is an original edge, failure already occurred. Otherwise the second q-sample completes it with probability greater than 3/4. The sprinkling lemma yields

\[
P_q(X_q\text{ contains a K-core})\le (1/4)/(3/4)=1/3.
\]

The core hypergraph has rank at most three. Theorem A therefore covers its non-1600-colorable induced sets at q/4 for cost <C_3. Since p_v/8<=q_v/4, the very same cover has cost <C_3 at p/8.

## 3. The residual has uniformly controlled proper links

For every nonempty proper A contained in a residual edge, the link F(A) has q-hit probability at most 3/4. Otherwise the larger original link H_4(A) would have hit probability greater than 3/4, putting A in the pre-minimalized core family. A contains a minimal retained core, so that residual edge would have been removed, a contradiction.

F itself is a subfamily of H, and q_v<=p_v. Thus its q-hit probability is at most 1/4.

P09-B consequently gives

\[
M_q(F)\le1015/3.
\]

Take a 4-coloring with monochromatic residual-edge q-cost at most (1015/3)/4^3. Those monochromatic edges cover F^(4). Every such edge has four elements. Since p_v/8<=q_v/4, its cost at p/8 is at most 4^(-4) times its q-cost. The residual obstruction cover therefore costs at most

\[
\frac{1015}{3\cdot4^3\cdot4^4}=\frac{1015}{49152}.
\]

All four powers in the parameter change are necessary. The monochromatic-probability power is 4^(-3), not 4^(-4).

## 4. Full failure-event coverage by product coloring

If S is 1600-colorable with respect to K and 4-colorable with respect to F, refine both colorings into their ordered pairs. The resulting 6400 color classes contain no edge of H:

- each original edge of size at most three contains a retained K-core;
- each removed four-edge contains a retained K-core;
- each remaining four-edge belongs to F.

An edge containing a nonmonochromatic core is itself nonmonochromatic. Hence

\[
D^{(6400)}\subseteq K^{(1600)}\cup F^{(4)}.
\]

The union of the two generator covers is a valid cover even if their generator families overlap; summing costs remains an upper bound. The total is

\[
C_3+1015/49152
=\frac{845318147}{1694515200}<1/2.
\]

This proves the statement.

## 5. Scope

This removes codegree and geometry restrictions for rank at most four. It does not prove that every decreasing family has rank at most four, that four-element witnesses can model arbitrary-rank witnesses without loss, or that the number 6400 remains adequate for unbounded rank. No probability estimate is promoted from sampled finite instances; the proof covers every finite ground set through the written inequalities.
