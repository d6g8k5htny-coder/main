# PR-TAL-019 --- Rank-three minimal-obstruction theorem

**Status:** author-side theorem candidate; no external independent
review; novelty unestablished.

Let (X) be finite and (`\mathcal `{=tex}D`\subseteq2`{=tex}\^X)
decreasing. Assume every inclusion-minimal forbidden set has size at
most three. For uniform Bernoulli parameter (p), if \[
`\mu`{=tex}\_p(`\mathcal `{=tex}D)`\ge`{=tex}`\frac34`{=tex}, \] then \[
`\boxed{\mathcal D^{(400)}\text{ is }(p/4)\text{-small}.}`{=tex} \] The
constructed cover has cost strictly less than \[
`\boxed{\frac{106087}{249600}
=\frac12-\frac{18713}{249600}.}`{=tex} \] This is a restricted theorem,
not a solution of the unrestricted discrete convexity conjecture.

## Singleton witnesses

Let (C) be the singleton minimal witnesses. Then \[
(1-p)\^{\|C\|}`\ge3`{=tex}/4,`\qquad `{=tex}\|C\|p`\le`{=tex}-`\log`{=tex}(3/4)\<3/10.
\] At parameter (p/4), singleton generators cost (\<3/40).

## High pairs and sprinkling

For a pair (ab), let (d(ab)) be the number of triple witnesses
containing it. Form a graph (P) containing all pair witnesses and every
pair with (pd(ab)\>5). The remaining triples are low.

Set (q=1-`\sqrt{1-p}`{=tex}). Two independent (q)-random sets have union
distributed as the (p)-random set, and (q`\ge `{=tex}p/2).

If the first sprinkle contains an edge of (P), choose a deterministic
first such edge. A pair witness already causes failure. For a high pair,
the second sprinkle hits an extension with probability \[
1-(1-q)^{d(ab)}\>1-e^{-5/2}\>9/10. \] Since the union violates
(`\mathcal `{=tex}D) with probability at most (1/4), \[
P_q(X_q`\text{ contains an edge of }`{=tex}P)`\le5`{=tex}/18. \]

## Cover the high-pair graph

Apply the graph common-cause argument to (P) under (q). Greedily extract
vertices of weighted neighborhood mass (\>2). Disjoint
first-core-selected events give core (q)-cost \[
\<`\log`{=tex}(108/73)\<2/5. \] The residual graph has weighted
neighborhood mass at most 2. Its expected edge mass is at most \[
`\frac{(5/18)(1+4)}{1-5/18}`{=tex}=`\frac{25}{13}`{=tex}. \] A
20-coloring therefore exists whose monochromatic residual edges have
(q)-cost at most (5/52).

Since (p/4`\le `{=tex}q/2), at parameter (p/4) the graph obstruction
cover costs \[ `\boxed{<1/5+5/208=233/1040.}`{=tex} \]

## Cover the low triples

For a vertex (v), let (L_v) be the extension graph whose edges (ab)
correspond to low triples (vab). Every vertex of (L_v) has weighted
neighborhood mass at most 5.

Greedily extract (v) while the weighted edge mass (s_v) of (L_v) exceeds
20. The graph second-moment estimate gives completion probability
(\>20/31). Disjoint first-core-selected events imply \[
`\sum`{=tex}\_{v`\in `{=tex}C_0}p\<`\log`{=tex}(80/49)\<1/2. \]

After extraction, every vertex extension mass is at most 20 and every
pair extension mass at most 5. For a residual triple (e={a,b,c}), the
directed dependency load is at most \[
3`\cdot20`{=tex}+3`\cdot5`{=tex}=75. \] The rank-three
second-moment/random-coloring lemma with (k=20) gives residual triple
generator cost at parameter (p) \[ 19/300. \] At parameter (p/4), the
low-triple cover costs \[ `\boxed{<1/8+19/19200=2419/19200.}`{=tex} \]

## Product-color composition

If a set avoids singleton witnesses, its induced high-pair graph is
20-colorable, and its induced low-triple hypergraph is 20-colorable,
refine the two colorings to ordered pairs, giving at most 400 colors.

Pair witnesses are blocked by the graph coloring. Every high triple
contains a high pair and is blocked by the graph coloring. Every low
triple is blocked by the hypergraph coloring. Hence every refined color
class belongs to (`\mathcal `{=tex}D).

Thus \[ `\mathcal `{=tex}D\^{(400)}
`\subseteq`{=tex}`\langle `{=tex}C`\rangle`{=tex}`\cup `{=tex}P^{(20)}`\cup`{=tex}`\mathcal `{=tex}H_0^{(20)}.
\] The total ((p/4))-cost is \[
\<`\frac3{40}`{=tex}+`\frac{233}{1040}`{=tex}+`\frac{2419}{19200}`{=tex}
=`\boxed{\frac{106087}{249600}<\frac12}`{=tex}. \]

## Boundary

The proof uses the fact that a rank-three witness can be reduced through
a pair core whose extension is a singleton. The unrestricted conjecture
allows arbitrary minimal-witness rank. The next question is whether this
sprinkling/common-core/product-color architecture can be iterated
without constants growing with rank.
