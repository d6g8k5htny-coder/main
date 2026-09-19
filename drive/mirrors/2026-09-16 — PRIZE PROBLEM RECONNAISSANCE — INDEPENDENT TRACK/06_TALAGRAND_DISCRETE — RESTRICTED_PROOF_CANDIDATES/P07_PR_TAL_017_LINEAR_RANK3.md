# PR-TAL-017 --- Universal theorem for linear 3-uniform witness hypergraphs

**Grade:** author-side complete derivation; no external review; novelty
unestablished.

Let (`\mathcal`{=tex}H) be a 3-uniform **linear** hypergraph: distinct
edges meet in at most one vertex. Let (`\mathcal`{=tex}D) be the family
of sets containing no edge of (`\mathcal`{=tex}H). Choose vertices
independently with arbitrary probabilities (p_v), and assume \[
`\mu`{=tex}\_{`\mathbf`{=tex}p}(`\mathcal`{=tex}D)`\ge`{=tex}`\frac34`{=tex}.
\]

Then \[
`\boxed{\mathcal D^{(4)} \text{ has a product-measure generator cover of cost }<\frac12.}`{=tex}
\]

Thus arbitrary vertex degrees and arbitrarily many crossing rank-three
constraints are allowed, provided the minimal witnesses form a linear
3-uniform hypergraph.

## 1. Greedy common-cause extraction

For a current hypergraph and a vertex (v), define its extension mass \[
s_v= `\sum`{=tex}*{`\substack{e\in\mathcal H\\v\in e}`{=tex}}
`\prod`{=tex}*{u`\in`{=tex}e`\setminus`{=tex}{v}}p_u. \] While some
current vertex has (s_v\>2), put it into a core (C) and delete it and
its incident edges.

Because the hypergraph is linear, for fixed (v) the two-element sets
(e`\setminus`{=tex}{v}) are pairwise disjoint. Hence their completion
events are independent. Conditional on selecting (v), the probability
that at least one current incident witness is completed is \[
1-`\prod`{=tex}*{e`\ni`{=tex}v}
`\left`{=tex}(1-`\prod`{=tex}*{u`\in`{=tex}e`\setminus`{=tex}{v}}p_u`\right`{=tex})
`\ge1`{=tex}-e^{-s_v}\>1-e^{-2}. \]

Define (E_i) as in PR-TAL-015: the (i)-th core vertex is selected, all
earlier core vertices are unselected, and one of its current incident
witnesses is completed. These events are disjoint and each is a failure
of (`\mathcal`{=tex}D). The identical rational calculation from
PR-TAL-015 yields \[ `\boxed{\sum_{v\in C}p_v<\frac{171}{500}.}`{=tex}
\]

## 2. Residual dependency load

In the residual hypergraph every vertex satisfies (s_v`\le2`{=tex}).

Fix an edge (e={a,b,c}). By linearity, every other edge meeting (e)
meets it in exactly one vertex. Therefore the directed dependency load
from PR-TAL-016 satisfies \[
`\sum`{=tex}*{`\substack{f\ne e\\f\cap e\ne\varnothing}`{=tex}}
`\prod`{=tex}*{u`\in`{=tex}f`\setminus`{=tex}e}p_u
`\le`{=tex}s_a+s_b+s_c`\le6`{=tex}. \] The residual good-event
probability remains at least (3/4).

PR-TAL-016 with witness rank (r=3), dependency load (6), and (k=4)
therefore gives a monochromatic-witness cover of residual cost at most
\[ `\frac{(1/4)(1+6)}{(3/4)4^2}`{=tex} =`\frac7{48}`{=tex}. \]

Together with singleton generators on (C), the total cost is strictly
less than \[ `\frac{171}{500}`{=tex}+`\frac7{48}`{=tex} =
`\frac{2927}{6000}`{=tex} = `\frac12`{=tex}-`\frac{73}{6000}`{=tex}. \]

This proves the theorem.

## 3. Boundary

Linearity is load-bearing in two places: 1. it makes the extension
events around a core vertex independent; 2. it converts residual vertex
extension bounds into a bounded edge-dependency load.

The next rank-three frontier is therefore **codegree**: many triples may
share a pair or may have highly overlapping extension graphs around a
vertex. Pair-shared cores are promising because their extensions are
singletons; vertex-shared cores lead to a graph of pair extensions and
may be attackable using the already established pair-supported
machinery.
