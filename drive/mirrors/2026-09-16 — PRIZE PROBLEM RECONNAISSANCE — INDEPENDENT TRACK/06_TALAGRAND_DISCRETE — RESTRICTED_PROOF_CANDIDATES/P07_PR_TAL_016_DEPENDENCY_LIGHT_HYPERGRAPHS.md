# PR-TAL-016 --- Dependency-light higher-rank witness hypergraphs

**Grade:** author-side complete derivation; no external review; novelty
unestablished.

Let (`\mathcal`{=tex}H) be the minimal-witness hypergraph of a
decreasing family (`\mathcal`{=tex}D). Choose coordinates independently
with probabilities (p_v). For (e`\in`{=tex}`\mathcal`{=tex}H), write \[
w(e)=`\prod`{=tex}*{v`\in`{=tex}e}p_v. \] Assume every witness has size
at least (r`\ge2`{=tex}). Define the directed dependency load \[ c=
`\max`{=tex}*{e`\in`{=tex}`\mathcal`{=tex}H}
`\sum`{=tex}*{`\substack{f\in\mathcal H,\ f\ne e\\f\cap e\ne\varnothing}`{=tex}}
`\prod`{=tex}*{v`\in`{=tex}f`\setminus`{=tex}e}p_v. \] Let (Y) count
witnesses contained in the random set and put \[
`\mu`{=tex}=EY=`\sum`{=tex}*{e`\in`{=tex}`\mathcal`{=tex}H}w(e). \]
Assume \[
P(Y=0)=`\mu`{=tex}*{`\mathbf`{=tex}p}(`\mathcal`{=tex}D)`\ge1`{=tex}-`\varepsilon`{=tex}.
\]

## Second moment

For intersecting (e,f), \[
P(e`\cup`{=tex}f`\subseteq`{=tex}X\_{`\mathbf`{=tex}p})
=w(e)`\prod`{=tex}\_{v`\in`{=tex}f`\setminus`{=tex}e}p_v. \] Therefore
the ordered sum of intersecting joint probabilities is at most
(c`\mu`{=tex}).

Comparing (EY\^2) with (`\mu`{=tex}\^2), the disjoint pairs cancel
exactly and the diagonal correction is at most (`\mu`{=tex}). Hence \[
EY^2`\le`{=tex}`\mu`{=tex}^2+(1+c)`\mu`{=tex}. \] Paley--Zygmund gives
\[ P(Y\>0)`\ge`{=tex}`\frac{\mu}{\mu+1+c}`{=tex}, \] so \[
`\boxed{ \mu\le \frac{\varepsilon(1+c)}{1-\varepsilon}.}`{=tex} \]

## Random coloring gives the p-small cover

Color the ground set uniformly with (k) colors. A witness (e) is
monochromatic with probability \[ k^{1-\|e\|}`\le`{=tex}k^{1-r}. \] Thus
some deterministic coloring has monochromatic-witness generator cost at
most \[
`\boxed{ \frac{\varepsilon(1+c)} {(1-\varepsilon)k^{r-1}}.}`{=tex} \]
Every set whose induced witness hypergraph has chromatic number (\>k)
contains one of those monochromatic witnesses. By PR-TAL-011 this is
exactly every member of (`\mathcal`{=tex}D\^{(k)}).

Consequently, \[
`\boxed{ \frac{\varepsilon(1+c)} {(1-\varepsilon)k^{r-1}}\le\frac12 \Longrightarrow \mathcal D^{(k)} \text{ is product-measure small}.}`{=tex}
\]

Concrete rank-at-least-three corollary: \[
`\boxed{ \mu_{\mathbf p}(\mathcal D)\ge\frac34,\quad c\le5 \Longrightarrow \mathcal D^{(2)} \text{ is product-measure small}.}`{=tex}
\]

This theorem handles arbitrary overlap geometry when the dependency load
is bounded. It does not supply the missing universal extraction theorem
for high dependency load. After PR-TAL-015 removes rank-two witnesses as
a universal obstruction, that extraction problem is the principal
remaining structural interface.
