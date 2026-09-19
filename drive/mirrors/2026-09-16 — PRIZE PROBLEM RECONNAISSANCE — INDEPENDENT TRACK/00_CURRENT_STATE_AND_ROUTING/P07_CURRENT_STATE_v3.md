# Phase07 current state

New author-side results:

1.  **PR-TAL-011:** exact minimal-witness hypergraph reformulation:
    (`\mathcal`{=tex}D\^{(k)}={S:`\chi`{=tex}(`\mathcal`{=tex}H\[S\])\>k}).
2.  **PR-TAL-012:** arbitrary capacity-one overlap under bounded
    weighted neighborhood mass. In particular, \[
    `\mu`{=tex}\_p(`\mathcal`{=tex}D)`\ge3`{=tex}/4,`\quad`{=tex}p`\Delta`{=tex}(G)`\le1`{=tex}
    `\Rightarrow`{=tex}`\mathcal`{=tex}D\^{(2)}`\text{ is }`{=tex}p`\text{-small}`{=tex}.
    \]
3.  **PR-TAL-013:** a small singleton common-cause core plus a light
    arbitrary-crossing remainder yields a p-small certificate; the
    concrete theorem gives four pieces.
4.  **PR-TAL-014:** the weighted deletion parameter (`\Psi`{=tex}\_k) is
    an exact sufficient certificate target for the capacity-one
    frontier.

No original prize conjecture is closed. No external review has occurred.

The next load-bearing problem is the high-weighted-degree regime after
common-cause extraction.

## Major successor: PR-TAL-015

The capacity-one arbitrary-overlap problem is now closed at author-side
proof level:

\[ `\boxed{
\mu_{\mathbf p}(\mathrm{Ind}(G))\ge3/4
\Longrightarrow
\mathrm{Ind}(G)^{(11)}
\text{ has product-measure generator cost }<1/2.
}`{=tex} \]

For uniform (p), this is exactly an 11-piece (p)-small theorem.

The proof greedily extracts high weighted-degree vertices as cheap
singleton common causes and applies PR-TAL-012 to the residual graph.
The final certified cost is \[ 8143/16500=1/2-107/16500. \]

This eliminates arbitrary graph overlap as the primary barrier. The next
genuine frontier is rank-(`\ge3`{=tex}) minimal-witness hypergraphs.

## PR-TAL-016 --- higher-rank light-dependency theorem

For minimal witnesses of size at least (r), define the exact directed
dependency load \[
c=`\max`{=tex}*e`\sum`{=tex}*{f`\ne `{=tex}e, f`\cap `{=tex}e`\ne`{=tex}`\varnothing`{=tex}}`\prod`{=tex}*{v`\in `{=tex}f`\setminus `{=tex}e}p_v.
\] If
(`\mu`{=tex}*{`\mathbf `{=tex}p}(`\mathcal `{=tex}D)`\ge1`{=tex}-`\varepsilon`{=tex}),
then \[ `\mathcal `{=tex}D\^{(k)} \] has generator cost at most \[
`\varepsilon`{=tex}(1+c)/\[(1-`\varepsilon`{=tex})k\^{r-1}\]. \] In
particular, for witness rank at least three, \[
`\mu`{=tex}\_{`\mathbf `{=tex}p}(`\mathcal `{=tex}D)`\ge3`{=tex}/4,`\quad `{=tex}c`\le5`{=tex}
`\Rightarrow `{=tex}`\mathcal `{=tex}D\^{(2)}
`\text{ is product-measure small}`{=tex}. \]

The remaining frontier is therefore high-dependency rank-(`\ge3`{=tex})
witness structure.

## PR-TAL-017 --- linear rank-three theorem

If the minimal witnesses form a linear 3-uniform hypergraph and \[
`\mu`{=tex}\_{`\mathbf `{=tex}p}(`\mathcal `{=tex}D)`\ge3`{=tex}/4, \]
then \[ `\boxed{\mathcal D^{(4)}
\text{ is product-measure small}.}`{=tex} \] The proof allows unbounded
vertex degree. A greedy vertex-core extraction costs (\<171/500);
linearity leaves dependency load at most (6), and PR-TAL-016 gives
residual cost (7/48). Total: \[ 2927/6000=1/2-73/6000. \]

The next frontier is non-linear rank-three overlap (high codegree /
overlapping pair extensions).
