# PR-TAL-018 --- Bounded-codegree rank-three theorem

**Grade:** author-side complete derivation; no external review; novelty
unestablished.

Let (`\mathcal`{=tex}H) be a 3-uniform minimal-witness hypergraph with
maximum pair-codegree at most (D`\ge1`{=tex}): every pair of vertices
belongs to at most (D) witnesses. Let (`\mathcal`{=tex}D) be the family
of sets containing no witness. Choose coordinates independently with
arbitrary probabilities (p_v), and assume \[
`\mu`{=tex}\_{`\mathbf`{=tex}p}(`\mathcal`{=tex}D)`\ge`{=tex}`\frac34`{=tex}.
\]

Set \[
k=`\left`{=tex}`\lceil11`{=tex}`\sqrt`{=tex}D`\right`{=tex}`\rceil`{=tex}.
\] Then \[
`\boxed{\mathcal D^{(k)} \text{ has a product-measure generator cover of cost }<\frac12.}`{=tex}
\]

## 1. Extension graphs and greedy core extraction

For a current hypergraph and vertex (v), form the extension graph (L_v)
on the other vertices: \[
ab`\in`{=tex}E(L_v)`\iff`{=tex}{v,a,b}`\in`{=tex}`\mathcal`{=tex}H. \]
Its weighted edge mass is \[
s_v=`\sum`{=tex}\_{ab`\in`{=tex}E(L_v)}p_ap_b. \] Because pair-codegree
is at most (D), every vertex of (L_v) has ordinary degree at most (D),
hence weighted neighborhood mass at most (D).

PR-TAL-012's second-moment calculation, applied to (L_v), gives \[
P(L_v\[X\_{`\mathbf`{=tex}p}\]`\text{ contains an edge}`{=tex})
`\ge`{=tex} `\frac{s_v}{s_v+1+2D}`{=tex}. \]

Greedily extract a vertex whenever (s_v\>12D), deleting it and its
incident witnesses. At extraction time the displayed probability is at
least \[ `\frac{12D}{14D+1}`{=tex}`\ge`{=tex}`\frac45`{=tex}. \]

Define disjoint first-core-selected failure events exactly as in
PR-TAL-015. Since the total failure probability is at most (1/4), \[
`\frac45`{=tex}`\left`{=tex}(1-`\prod`{=tex}*{v`\in`{=tex}C}(1-p_v)`\right`{=tex})`\le`{=tex}`\frac14`{=tex}.
\] Hence \[
`\prod`{=tex}*{v`\in`{=tex}C}(1-p_v)`\ge`{=tex}`\frac{11}{16}`{=tex} \]
and \[
`\sum`{=tex}\_{v`\in`{=tex}C}p_v`\le`{=tex}`\log`{=tex}`\frac{16}{11}`{=tex}\<`\frac38`{=tex}.
\] The final strict inequality is certified by the degree-4 Taylor lower
bound for (e\^{3/8}).

## 2. Residual dependency load

After extraction, every residual vertex has (s_v`\le`{=tex}12D).

Fix a residual witness (e={a,b,c}). Other witnesses meeting (e) in
exactly one vertex contribute at most \[ s_a+s_b+s_c`\le`{=tex}36D \] to
PR-TAL-016's directed dependency load.

Witnesses sharing a pair with (e) contribute at most (D) per pair
because each extra coordinate probability is at most one. There are
three pairs. Therefore \[ c\_{`\rm dep`{=tex}}`\le`{=tex}39D. \]

PR-TAL-016 with (`\varepsilon=1`{=tex}/4), rank (3), and
(k=`\lceil11`{=tex}`\sqrt`{=tex}D`\rceil`{=tex}) gives residual cover
cost at most \[ `\frac{1+39D}{3k^2}`{=tex} `\le`{=tex}
`\frac{40D}{3\cdot121D}`{=tex} = `\frac{40}{363}`{=tex}. \]

Adding the singleton core, \[
`\operatorname{cost}`{=tex}(`\mathcal`{=tex}D\^{(k)}) \<
`\frac38`{=tex}+`\frac{40}{363}`{=tex} = `\frac{1409}{2904}`{=tex} =
`\frac12`{=tex}-`\frac{43}{2904}`{=tex} \<`\frac12`{=tex}. \]

## 3. Meaning

This permits arbitrary rank-three crossing geometry and unbounded vertex
degree. The number of pieces grows only as (O(`\sqrt`{=tex}D)) with
pair-codegree.

The theorem does not give a universal constant when pair-codegree is
unbounded. The next structural question is whether high pair-codegree
can itself be compressed by paying once for pair cores, which are
especially favorable because a fixed pair's witness extensions are
singleton events.
