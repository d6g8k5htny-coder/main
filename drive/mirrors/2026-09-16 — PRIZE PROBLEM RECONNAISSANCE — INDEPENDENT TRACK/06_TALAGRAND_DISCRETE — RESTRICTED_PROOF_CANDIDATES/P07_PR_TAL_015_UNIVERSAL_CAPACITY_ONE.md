# PR-TAL-015 --- Universal capacity-one theorem by greedy common-cause extraction

**Grade:** author-side complete derivation; no external review;
historical novelty unestablished.

Let (G=(V,E)), let (`\mathcal`{=tex}D) be its independent sets, and
choose vertices independently with arbitrary probabilities (p_v). Assume
\[
`\mu`{=tex}\_{`\mathbf`{=tex}p}(`\mathcal`{=tex}D)`\ge`{=tex}`\frac34`{=tex}.
\]

Then \[
`\boxed{\mathcal D^{(11)}\text{ admits an explicit product-measure small cover of cost }<\frac12.}`{=tex}
\] In particular, for uniform (p), (`\mathcal`{=tex}D\^{(11)}) is
(p)-small.

## 1. Greedy extraction

Starting with (G_0=G), while the current graph contains a vertex (v_i)
whose weighted neighborhood mass \[
s_i=`\sum`{=tex}*{u`\in`{=tex}N*{G\_{i-1}}(v_i)}p_u \] exceeds (2), put
(v_i) into a core (C) and delete (v_i). Stop when every remaining vertex
has weighted neighborhood mass at most (2).

For each extracted (v_i), define (E_i) to be the event that: - (v_i) is
selected; - every earlier core vertex is unselected; - at least one
current neighbor of (v_i) is selected.

The events (E_i) are pairwise disjoint and each implies that the random
set is not independent.

Moreover \[ P(E_i) = p\_{v_i}`\prod`{=tex}*{j\<i}(1-p*{v_j})
`\left`{=tex}(1-`\prod`{=tex}*{u`\in`{=tex}N_i}(1-p_u)`\right`{=tex}).
\] Since (s_i\>2) and (1-x`\le`{=tex}e\^{-x}), \[
1-`\prod`{=tex}*{u`\in`{=tex}N_i}(1-p_u)\>1-e\^{-2}. \] Therefore \[
(1-e\^{-2})
`\left`{=tex}(1-`\prod`{=tex}\_{v`\in`{=tex}C}(1-p_v)`\right`{=tex})
`\le`{=tex}`\frac14`{=tex}. \]

A completely rational enclosure is enough. The first seven terms of the
exponential series give \[
e\^2\>`\frac{331}{45}`{=tex}\>`\frac{125}{17}`{=tex}, \] so
(e\^{-2}\<17/125) and (1-e\^{-2}\>108/125). Hence \[
1-`\prod`{=tex}*{v`\in`{=tex}C}(1-p_v)\<`\frac{125}{432}`{=tex},
`\qquad`{=tex}
`\prod`{=tex}*{v`\in`{=tex}C}(1-p_v)\>`\frac{307}{432}`{=tex}. \] Using
(-`\log`{=tex}(1-x)`\ge`{=tex}x), \[ `\sum`{=tex}*{v`\in`{=tex}C}p_v
`\le`{=tex}-`\log`{=tex}`\prod`{=tex}*{v`\in`{=tex}C}(1-p_v)
\<`\log`{=tex}`\frac{432}{307}`{=tex}. \] Finally the degree-4 Taylor
lower bound for (e\^{171/500}) exceeds (432/307), so \[
`\boxed{\sum_{v\in C}p_v<\frac{171}{500}.}`{=tex} \]

## 2. Light remainder

The residual graph (G-C) has weighted neighborhood mass at most (2). Its
independent-set probability is at least that of (G), hence at least
(3/4).

PR-TAL-012 with (`\varepsilon=1`{=tex}/4) and (c=2) gives a fixed
11-coloring whose monochromatic residual edges have total generator cost
at most \[ `\frac1{11}`{=tex}`\frac{(1/4)(1+4)}{3/4}`{=tex} =
`\frac5{33}`{=tex}. \]

Use singleton generators ({v}) for (v`\in`{=tex}C), together with those
monochromatic residual edges. Any set intersecting (C) is covered by a
singleton. Any set disjoint from (C) whose induced graph has chromatic
number (\>11) contains a monochromatic residual edge.

Thus \[ `\operatorname{cost}`{=tex}(`\mathcal`{=tex}D\^{(11)}) \<
`\frac{171}{500}`{=tex}+`\frac5{33}`{=tex} = `\frac{8143}{16500}`{=tex}
= `\frac12`{=tex}-`\frac{107}{16500}`{=tex} \<`\frac12`{=tex}. \]

This proves the theorem.

## 3. Meaning and boundary

This removes **all structural assumptions on overlap** for capacity-one
systems: no laminarity, incidence bound, crossing-graph bound,
bipartiteness, or high-rank hypothesis remains.

It does not solve the full discrete convexity conjecture because minimal
forbidden witnesses may have rank (3) or larger. The next frontier is
therefore genuinely higher-rank witness hypergraphs.

The proof should be compared carefully with existing pair-supported
Talagrand results before any novelty claim. In particular,
Frankston--Kahn--Park (2021) prove a broad pair-supported
weak-to-integral smallness theorem by different methods.
