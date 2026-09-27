# Side-24 gap-fill supplement

**Companion to SIDE24-PRE-REVIEW-2026-07-31**
Prepared 2026-08-01. Addresses all twelve items of the proof-gap and correction
register in `02_PRE_REVIEW_DOSSIER.pdf` §12, in register order.

**Conventions.** All notation is that of the manuscript (`03_SIDE24_MANUSCRIPT.pdf`)
and dossier. $f=f_{24}$ is the normalized side-24 periodized Bargmann–Fock field on
$\mathbb{T}^3_{24}$, $M=x-\frac r2 t$, $S=x+\frac r2 t$, the pins are
$f(M)=b$, $f(S)=b-\kappa r^3/6$, $\nabla f(M)=\nabla f(S)=0$, and $W_r =
|\det H_M|\,|\det H_S|\,\mathbf 1_{\{H_M\prec 0,\ \mathrm{ind}(H_S)=2\}}$.
Throughout, "Gap item $k$" refers to row $k$ of the register.

**Status labels.** Each section ends with a status line. Where a full proof is
given, the item is marked CLOSED (subject to the same independent review as the
rest of the package). Where something remains assumed, the exact assumption is
boxed as a RESIDUAL PREMISE — nothing is silently closed. Sections superseded
by the V3 corrective integration (Part F) carry a **HISTORICAL** banner: their
original "CLOSED" labels are quoted as the pre-V3 record and no longer
describe the current state; any claim of theirs that does not survive
(unconditional rates, withdrawn proofs) is named explicitly in the banner.

**Claim taxonomy (precision convention).** Every quantitative claim in this
supplement is one of the following, and the text says which:
(i) **exact/symbolic** — proved or verified by exact symbolic computation
(SymPy/Fraction). This covers every quantitative constant in the supplement:
the covariances, the elimination (B.1), the cone constant $D_2$ (closed form,
B.2), the first-variation variations and $P_3(24)=-620813376/35$, the energy
margins;
(ii) **numerical cross-check** — independent numerical evaluations retained
only as corroboration of category-(i) results, never load-bearing;
(iii) **flagged task** — the paywalled-database literature pass (item 12).
The word "derived" is used only for category (i); numerical agreement is never
called a derivation.

---

## Part A — Formal statements and proofs (Gap items 1–5)

> **Supersession note (2026-08-01, V3 integration).** Sections A.1 and A.2
> are superseded by Part F.1–F.2 (the level-transition/sector elder mark and
> the disjoint elder-failure exhaustion from the V3 corrective addendum), and
> the proof of Theorem A.4.6 is superseded by Part F.4 (threshold-free
> inward tube, pin-preserving robustness). They are retained verbatim below
> as the correction history; the current statements live in Part F.

### A.1 Gap item 3 (done first: it is the interface the others use).
**The determinant-weighted Palm law and Borel measurability of the elder mark.**

**Definition A.1.1 (pins and weight).** For $0<r<12$, $t\in S^2$, $b\in\mathbb R$,
$\kappa>0$, let $\Pi_{r,x,t}=(f(M),f(S),\nabla f(M),\nabla f(S))$ and
$z_r(b,\kappa)=(b,\,b-\kappa r^3/6,\,0,\,0)$. The conditional law
$\mathbb E^0_{r,t,b,\kappa}[\cdot]=\mathbb E[\cdot\mid \Pi_{r,x,t}=z_r(b,\kappa)]$
is the regular conditional Gaussian distribution (it exists: the pin vector is
Gaussian; the corrected-pin covariance is nondegenerate by Module F, so the
conditional distributions have a continuous version in $(r,t,b,\kappa)$).
Put $Z_r(t,b,\kappa)=\mathbb E^0_{r,t,b,\kappa}[W_r]$. By Module H (G2),
$c_Z r^2\le Z_r\le C_Z r^2$ on compact positive mark sets, in particular
$Z_r>0$. The **all-typed determinant-weighted Palm law** is
$$\mathbb E^{MS}_{r,t,b,\kappa}[F]=\frac{\mathbb E^0_{r,t,b,\kappa}[W_r F]}{Z_r(t,b,\kappa)}. \tag{A.1}$$

**Definition A.1.2 (elder mark, constructed measurably).** Let
$\Omega_{\mathrm{Mo}}$ be the event that $f$ is Morse with pairwise distinct
critical values ($\mathbb P(\Omega_{\mathrm{Mo}})=1$, Proposition 1.1). For a
continuous $g:\mathbb T^3_{24}\to\mathbb R$ and $u,v\in\mathbb T^3_{24}$ define
the **connection level**
$$\mathfrak m_g(u,v)=\sup_{\gamma:u\to v}\min_{s\in[0,1]} g(\gamma(s)),$$
the supremum over continuous paths. Then
$|\mathfrak m_g(u,v)-\mathfrak m_{g'}(u,v)|\le \|g-g'\|_\infty$, so
$(g,u,v)\mapsto\mathfrak m_g(u,v)$ is continuous, hence Borel, on
$C(\mathbb T^3_{24})\times(\mathbb T^3_{24})^2$. Fix a countable dense set
$D\subset\mathbb T^3_{24}$ and define the **birth level of the open superlevel
component of $u$ above height $a$**,
$$\mathfrak b_g(u,a)=\sup\{\,g(z): z\in D,\ \mathfrak m_g(u,z)>a\,\}
\qquad(\sup\varnothing:=-\infty),$$
a countable supremum of Borel functions (each
$g(z)\mathbf 1_{\{\mathfrak m_g(u,z)>a\}}$ plus the constant $-\infty$), hence
Borel in $(g,u,a)$. The strict inequality is essential (v2 correction):
$\{\mathfrak m_g(u,\cdot)>a\}$ is the *open* superlevel component of $u$ in
$\{g>a\}$, so the supremum over $D$ equals the maximum of $g$ on that
component — on $\Omega_{\mathrm{Mo}}$ the unique birth maximum — with no
contamination from other components and no artificial floor.

At an index-two saddle $S$ with $h=f(S)$, the unstable line is the positive
eigenline of the Hessian. Choose a measurable unit vector $v(S)$ spanning it
(the eigenline is unique and varies measurably on the nondegenerate locus; what
follows is independent of the sign choice). The sector probes use a
**field-dependent** radius (v2 correction — no universal fixed $\delta$
exists): for each sign, let
$$\ell_\pm(f;S)=\sup\{t>0: f(S+sv(S))>h\ \text{for all }0<s<t\},\qquad
\delta(f;S)=\tfrac12\min\big(\ell_+,\ell_-,\ \operatorname{dist}(S,\operatorname{Crit}(f)\setminus\{S\}),\ 1\big),$$
measurable in $(f,S)$ (each $\ell_\pm$ is a supremum over rational $s$ of
closed conditions on continuous evaluations, and $\operatorname{Crit}(f)$ is
finite on $\Omega_{\mathrm{Mo}}$ with distances continuous in $f$). On
$\Omega_{\mathrm{Mo}}$, $\delta(f;S)>0$ and
$u_\pm(S)=S\pm\delta(f;S)\,v(S)$ lie one in each upper local sector of $S$,
with $f(u_\pm)>h$. Define
$$e(f;M,S)=\mathbf 1\big\{\mathfrak m_f(u_+,u_-)=h\ \text{and}\ f(M)=\min\big(\mathfrak b_f(u_+,h),\,\mathfrak b_f(u_-,h)\big)\big\}. \tag{A.2}$$

**Lemma A.1.3 (the mark is correct and Borel).** On $\Omega_{\mathrm{Mo}}$,
$e(f;M,S)=1$ if and only if (i) the two upper sectors of $S$ lie in distinct
components of $\{f>h\}$; (ii) $M$ is the birth maximum of one of them; (iii)
the birth value at $M$ is the smaller of the two — exactly the elder-pair
condition of dossier §6. Moreover $(f,M,S)\mapsto e(f;M,S)$ is Borel
measurable on the Morse locus.

**Proof.** Since $u_\pm$ join through $S$ with path minimum exactly $h$, one
always has $\mathfrak m_f(u_+,u_-)\ge h$; and the sectors lie in one component
of $\{f>h\}$ iff a connecting path with minimum $>h$ exists, iff
$\mathfrak m_f(u_+,u_-)>h$. Hence (i) is exactly
$\{\mathfrak m_f(u_+,u_-)=h\}$ (equivalently $\{\mathfrak m_f\le h\}$; the
equality in (A.2) is a genuine generic condition, not a degeneracy — an
earlier draft used $\mathfrak m<h$, which is empty and was wrong). By
construction of $\mathfrak b$ with the strict inequality, on
$\Omega_{\mathrm{Mo}}$ the numbers $\mathfrak b_f(u_\pm,h)$ are the birth
values of the two sector components of $\{f>h\}$ separately: the route through
$S$ has minimum $h$, not $>h$, so it does not merge the two suprema. Since
critical values are distinct, $f(M)=\mathfrak b_f(u_\pm,h)$ identifies the
birth maximum uniquely. So (A.2) is equivalent to (i)–(iii). Measurability:
$\mathfrak m$ and $\mathfrak b$ are Borel as shown; $v(S)$ and $\delta(f;S)$
are measurable; $h=f(S)$ and $f(M)$ are evaluations; $\{\mathfrak m=h\}$ is
Borel as $\{\mathfrak m\le h\}\cap\{\mathfrak m\ge h\}$. $\blacksquare$

**Definition A.1.4 (selection probability).**
$p_r(t,b,\kappa)=\mathbb E^{MS}_{r,t,b,\kappa}[e(f;M,S)]$, so that with
$K^{\mathrm{all}}_r=p_{\Pi}(z_r(b,\kappa))\,\mathbb E^0[W_r]$,
$$K^{\mathrm{sel}}_r(t,b,\kappa)=p_r(t,b,\kappa)\,K^{\mathrm{all}}_r(t,b,\kappa). \tag{A.3}$$

**Status: HISTORICAL — superseded by Part F.1 (V3).** The "CLOSED" label
below is the pre-V3 record and no longer describes the current state. The
elder mark of this section (the V2 sector definition) is replaced by the
level-transition/sector mark of Part F.1 (Theorems F.1.1–F.1.2); the V2
requirement $\mathfrak m<h$ never fires (Lemma F.1.11), so the mark below
must not be cited. The Palm-law definitions (A.1)–(A.3) themselves are
retained.

> **Status: CLOSED** *(pre-V3 historical record).* The measurability repair
> requested by the register and by
hostile-review Q7 is supplied by Lemma A.1.3; the Palm law is defined in one
place ((A.1)–(A.3)), with neither adjacency nor selection in the base law.

---

### A.2 Gap item 4. **The elder-selection trichotomy as a formal lemma.**

**Setup.** On $\Omega_{\mathrm{Mo}}$, fix an ordered typed pair $(M,S)$ at
distance $r$, and let $h=f(S)$, $b=f(M)$. Define events:

* $A_r$ (adjacency): at least one unstable half-branch of $S$ under gradient
  ascent converges to $M$. (Boolean mark; if both do, the pair still counts
  once — Module L §5.)
* $E_r=\{e(f;M,S)=1\}$ (actual elder selection, (A.2)).
* $L_{\mathrm{same},r}$ (same-component self-attachment): the two upper
  sectors of $S$ coincide, $\mathfrak m_f(u_+,u_-)>h$.
* $B_r$ (third-critical witness): there exists a critical point
  $y\notin\{M,S\}$ of $f$ with $f(y)\in[h,b]$ (value in the pair window).
  Every such $y$ lies in exactly one of the following three regions — a true
  partition, with $C_1=2$ and $\delta>0$ fixed (v2 correction; an earlier
  draft mixed $\{M,S\}$-based and midpoint-based radii and was neither
  disjoint nor exhaustive):
  (i) collar: $\operatorname{dist}(y,\{M,S\})\le C_1 r$;
  (ii) singular-near: $\operatorname{dist}(y,\{M,S\})>C_1 r$ and
  $\operatorname{dist}(y,x)\le\delta$, where $x$ is the pair midpoint;
  (iii) fixed-distance: $\operatorname{dist}(y,\{M,S\})>C_1 r$ and
  $\operatorname{dist}(y,x)>\delta$.
  In region (ii), $s:=\operatorname{dist}(y,x)\ge
  \operatorname{dist}(y,\{M,S\})-r/2>C_1r-r/2=3r/2>r$, so $\eta=r/s\le2/3<1$:
  a compact $\eta$-chart, as required.

**Lemma A.2.1 (exhaustive trichotomy).** On $\Omega_{\mathrm{Mo}}$,
$$A_r\cap E_r^{c}\subset B_r\cup L_{\mathrm{same},r}. \tag{A.4}$$

**Proof.** Work on $A_r$: a half-branch of $S$ ascends to $M$, so $M$ is a
maximum of the upper-sector component it lies in; call the sectors
$C_1\ni M$-ward and $C_2$. If $C_1=C_2$ we are in $L_{\mathrm{same},r}$.
Assume $C_1\ne C_2$ with birth values $b_1,b_2$ and birth maxima $M_1,M_2$
(distinct values make them unique).

*Case 1: $M\ne M_1$.* Then $M_1$ and $M$ are distinct maxima in one component
of $\{f>h\}$. Any path in that component from $M$ to $M_1$ has minimum $>h$;
following the component's merge tree downward from $\min(b_1,f(M))$, the two
sub-components containing $M_1$ and $M$ first merge at an index-two saddle $S'$
with $h<f(S')\le f(M)=b$. (Existence: a Morse component of a superlevel set
containing two distinct local maxima contains an index-$(d-1)$ merge saddle on
every connecting route; take the highest threshold at which they disconnect.)
Then $y=S'$ is a critical point with $f(y)\in(h,b]$, a window witness.

*Case 2: $M=M_1$ but $e=0$.* Since $C_1\ne C_2$ and (ii) of (A.2) holds,
failure of (A.2) forces $f(M)>\min(b_1,b_2)$, i.e. $b_2<f(M)$: $C_2$ is
younger and is the component killed at $S$. The required witness is $C_2$'s
own birth maximum: $y=M_2$ is a critical point (a local maximum), its value
satisfies $h<b_2=f(M_2)<f(M)=b$ — the lower bound because $C_2$ is a
component of the *open* superlevel set $\{f>h\}$ containing an upper sector
of $S$, so its birth value exceeds $h$ — and $M_2\notin\{M,S\}$ ($M_2\ne M$
as they lie in distinct components; $M_2\ne S$ as $f(S)=h<b_2$). So $y=M_2$
is a window witness. (Note the merge saddle of $M$'s surviving class would
not serve here: its value can lie below $h$, outside the window.)

In both cases $y$ is a critical point with value in the pair window; the
spatial partition (i)–(iii) places it in $B_r$. $\blacksquare$

**Corollary A.2.2 (selection bound).** With the Palm law (A.1),
$$1-p_r\le \mathbb P^{MS}_r(A_r^c)+\mathbb P^{MS}_r(B_r)+\mathbb P^{MS}_r(L_{\mathrm{same},r}),$$
and the three terms are $O(r^3)$, $O(r^3)$, $O(r^3)$ respectively by:
$\mathbb P^{MS}(B_r)\le \mathbb E^{MS}N_{\mathrm{col}}+\mathbb E^{MS}N_{\mathrm{sing}}+\mathbb E^{MS}N_{\mathrm{far}}$
(Markov; the three regional expectations are $O(r^3)$ by Module B §§3–6 /
Module E / Module I §3), and the adjacency and self-attachment failures are
$O(r^3)$ by Module I §4 together with the consolidated capture/escape theorem
of Section A.4 below. Hence, uniformly on compact positive $(b,\kappa)$ sets,
$$1-p_r(t,b,\kappa)\le C_{\mathrm{sel}}\,r^3. \tag{A.5}$$

**Status: HISTORICAL — superseded by Part F.2 (V3).** The "CLOSED" label
below is the pre-V3 record. Two claims of this section do not survive: the
concluding bound $1-p_r\le C_{\mathrm{sel}}r^3$ in Corollary A.2.2 is
stated unconditionally above, but its three inputs are exactly the open
residual premises RP-A/RP-L/RP-C/RP-S/RP-F of Part F.2, so the $O(r^3)$
rate is **conditional**, not proved; and the region partition is replaced
by the genuinely disjoint single-distance partition of Part F.2 with the
proved trichotomy (Lemma F.2.1) and selection inequality.

> **Status: CLOSED** *(pre-V3 historical record)* (the compressed paragraph
> of the manuscript §3.4 is replaced
by Lemma A.2.1; the region partition (i)–(iii) is a spatial trichotomy, so no
failure mode is unclassified, answering hostile-review Q6).

---

### A.3 Gap items 1 and 2. **The near/far decomposition, the exact near-density
pushforward, and the off-diagonal lemma.**

Fix $0<\rho<12$ and split the expected lifetime measure of Proposition 1.2 as
$$\nu_{3,24}(\ell)=\nu^{\mathrm{near}}_\rho(\ell)+\nu^{\mathrm{far}}_\rho(\ell), \tag{A.6}$$
according as the elder pair's own separation is $<\rho$ or $\ge\rho$.

**Proposition A.3.1 (exact near-density pushforward; Gap item 2).** Let
$K^{\mathrm{sel}}_r$ be as in (A.3) and $r_\ell(\kappa)=(6\ell/\kappa)^{1/3}$.
For every nonnegative test function $\varphi$,
$$\int_0^\infty\varphi(\ell)\,\nu^{\mathrm{near}}_\rho(\ell)\,d\ell
=\int_{S^2}\!\!\int_{\mathbb R}\!\!\int_0^\infty\!\!\int_0^\rho
\varphi\Big(\frac{\kappa r^3}{6}\Big)K^{\mathrm{sel}}_r(t,b,\kappa)\,\frac{r^3}{6}\,r^2\,dr\,d\kappa\,db\,d\sigma_2(t), \tag{A.7}$$
(the factor $r^3/6$ is the affine gap change $|dh|=(r^3/6)\,d\kappa$, included
in the display explicitly)
and the exact density representative is
$$\nu^{\mathrm{near}}_\rho(\ell)=\frac{6^{2/3}}{18}\,\ell^{-1/3}\int_{S^2}\int_{\mathbb R}\int_{6\ell/\rho^3}^{\infty}
\kappa^{-2/3}\,G^{\mathrm{sel}}_{r_\ell(\kappa)}(t,b,\kappa)\,d\kappa\,db\,d\sigma_2(t), \tag{A.8}$$
where $G^{\mathrm{sel}}_r=r^4 K^{\mathrm{sel}}_r$ (dimension three).

**Proof.** Marked two-point Kac–Rice for the ordered typed pair process with
the bounded Borel mark $e$ (Lemma A.1.3) on $\{r<\rho\}$, the midpoint/polar
change of variables $dM\,dS=dx\,r^2dr\,d\sigma_2$ (Module L, $|\det|=1$), the
per-volume normalization $24^{-3}\int dx=1$, and the affine gap change
$|dh|=(r^3/6)\,d\kappa$ give (A.7). Writing
$K\,r^5/6=(r^4K)\cdot r/6=G^{\mathrm{sel}}_r\,r/6$ and substituting
$r=r_\ell(\kappa)$, $r\,dr=\frac{6^{2/3}}{3}\kappa^{-2/3}\ell^{-1/3}d\ell$
(Module K, exact), the scalar $(1/6)\cdot(6^{2/3}/3)=6^{2/3}/18$ gives (A.8);
the lower limit $r<\rho$ becomes $\kappa>6\ell/\rho^3$. $\blacksquare$

**Proposition A.3.2 (off-diagonal lemma; Gap item 1).** For each finite
$\ell_0$ the far density is uniformly bounded:
$$\sup_{0<\ell\le\ell_0}\nu^{\mathrm{far}}_\rho(\ell)\le C_{\rho,\ell_0}<\infty,
\qquad\text{hence }\ \nu^{\mathrm{far}}_\rho(\ell)=O(1)=o(\ell^{-1/3})
\ \ \text{as }\ell\downarrow0. \tag{A.9}$$
(A v2 downgrade in the statement, not the conclusion: the density at the
slice $h=b-\ell$ is $O(1)$; an earlier draft claimed $O(\ell)$ via a window
argument, which double-counts the width. The $O(1)$ bound is what the proof
supports, and $O(1)=o(\ell^{-1/3})$ is all the theorem uses.)

**Proof.** Since the elder mark satisfies $e\le 1$, $\nu^{\mathrm{far}}_\rho$ is
bounded above by the *all-typed* ordered pair density at lifetime $\ell$:
writing $\Xi_{x,y}=(f(x),f(y),\nabla f(x),\nabla f(y))$,
$$\nu^{\mathrm{far}}_\rho(\ell)\le 24^{-3}\!\!\int_{D_\rho}\int_{\mathbb R}
p_{\Xi_{x,y}}(b,b-\ell,0,0)\;\mathbb E\big[|\det H_x\det H_y|\,\mathbf 1_{\mathrm{types}}\,\big|\,\Xi_{x,y}=(b,b-\ell,0,0)\big]\,db\,dx\,dy,$$
where $D_\rho=\{(x,y):\operatorname{dist}(x,y)\ge\rho\}$. This is the marked
two-point Kac–Rice density on a compact separated configuration set: the full
positive Fourier support (Module F, Theorem 3.1 and Corollary 3.2) makes
$\Xi_{x,y}$ uniformly nondegenerate on $D_\rho$, so
$p_{\Xi}(b,b-\ell,0,0)\le C_\rho e^{-c_\rho b^2}$ uniformly in
$0<\ell\le\ell_0$ (the smallest eigenvalue of $\operatorname{Cov}\Xi$ is
bounded below and the mean-zero target has
$\|(b,b-\ell,0,0)\|^2\ge b^2/2$ for $\ell$ bounded). Gaussian regression
against this nondegenerate block gives conditional Hessian means affine in
$(b,b-\ell)$ and uniformly bounded conditional covariances, so by the
degree-six moment bound
$$\mathbb E[|\det H_x\det H_y|\mid\Xi]\le C_\rho\big(1+(|b|+\ell)^6\big).$$
Integrating over the compact $D_\rho$ and $b\in\mathbb R$,
$$\nu^{\mathrm{far}}_\rho(\ell)\le C_\rho\!\int_{\mathbb R}\big(1+(|b|+\ell_0)^6\big)e^{-c_\rho b^2}\,db=:C_{\rho,\ell_0}<\infty,$$
uniformly in $0<\ell\le\ell_0$. $\blacksquare$

**Theorem A.3.3 (asymptotic; Gap item 2 completed).** Suppose
(H1) $G^{\mathrm{sel}}_r\to G^{\mathrm{sel}}_0$ pointwise on the mark space;
(H2) one integrable majorant for $\kappa^{-2/3}G^{\mathrm{sel}}_r$ on
$S^2\times\mathbb R\times(0,\infty)$, uniform as $\kappa\downarrow0$;
(H3) $\lim G^{\mathrm{sel}}_0=\lim G^{\mathrm{all}}_0$ (selected and all-typed
limiting integrands coincide).
Then $\nu_{3,24}(\ell)=c_{3,24}\ell^{-1/3}(1+o(1))$ with
$c_{3,24}=\frac{6^{2/3}}{18}\int\kappa^{-2/3}G^{\mathrm{all}}_0$.

**Proof.** (A.8) + dominated convergence using (H1)–(H2), with the lower
$\kappa$-limit $6\ell/\rho^3\downarrow0$; (H3) identifies the constant;
Proposition A.3.2 removes $\nu^{\mathrm{far}}_\rho$. The hypotheses hold for
the side-24 field: (H1) by the corrected-pin convergence (Module A) and
$p_r\to1$ ((A.5)); (H2) is the all-mark majorant
$C[1+(|b|+\kappa)^N]e^{-c(b^2+\kappa^2)}\kappa^{-2/3}$ of Module H §6 /
Module I (25), integrable including $\kappa\downarrow0$ (answering hostile
Q3: the $\kappa^{-2/3}$ singularity is integrable at $0$ and the Gaussian in
$\kappa$ excludes higher-contact double scaling on compact $\kappa$ sets;
quartic contacts have $\kappa=0$, a measure-zero mark set with no
$|\det|$-weighted mass); (H3) follows from $0\le p_r\le1$, $p_r\to1$, and the
same majorant. $\blacksquare$

**Status: CLOSED.** The global formula is now a named proved statement
(Proposition A.3.1), the DCT hypotheses are isolated and each is discharged,
and the off-diagonal lemma is proved (Proposition A.3.2, answering hostile
Q10).

---

### A.4 Gap item 5. **Consolidated capture/escape proof (no predecessor references).**

This section replaces Module J's reliance on "previously reviewed v1.2
estimates" with one self-contained statement and proof. Constants are chosen
for convenience; no attempt is made to match historical values.

**Normalized potential.** On the fixed compact convex chart $D^+$ consider, for
$X\in\mathbb R$, $Y\in\mathbb R^2$,
$$P(X,Y)=p_0(X)+\tfrac12(X^2-\tfrac14)a\!\cdot\! Y-\tfrac12 Y^{\!\top}TY+\tfrac12 X\,Y^{\!\top}WY+\tfrac16 C[Y,Y,Y]+r\,Q(X,Y), \tag{A.10}$$
$p_0(X)=X^3/3-X/4-1/12$, $T\succ0$ symmetric, $W$ symmetric, $C$ a symmetric
cubic tensor, $Q(M)=Q(S)=0$, $\nabla Q(M)=\nabla Q(S)=0$ at $M=(-\tfrac12,0)$,
$S=(\tfrac12,0)$. Put
$$R=1+|a|+\|W\|+\|C\|+\|Q\|_{C^2(D^+)}\ge1,\qquad \tau=\lambda_{\min}(T).$$
For the conditioned side-24 field, $T=-Q_\perp/(\kappa r)$ (Module I (12); the
factor $1/r$ is essential). Fix
$$K=4096,\qquad \varepsilon_0=\tfrac{1}{1024K},\qquad
\text{assume}\quad rR^5\le\varepsilon_0,\quad \tau\ge 2KR^5+2R. \tag{A.11}$$
Write $E(Y)=Y^{\!\top}TY$, $q(Y)=|TY|$; then $q^2\ge\tau E$ and
$|Y|^2\le E/\tau$. The gradient is
$$F_X=X^2-\tfrac14+Xa\!\cdot\!Y+\tfrac12Y^{\!\top}WY+rQ_X,\qquad
F_Y=\tfrac12(X^2-\tfrac14)a-TY+XWY+\tfrac12C[Y,Y,\cdot]+r\nabla_YQ. \tag{A.12}$$

**Lemma A.4.1 (endpoint energy slope).** The Jacobian at $S$ has block form
$J_S=\begin{pmatrix}A&B^{\!\top}\\ B&-T+E_D\end{pmatrix}$ with $A\ge3/4$,
$|B|\le R$, $\|E_D\|\le 3R/4$ (after one fixed small-$r$ reduction), and
exactly one positive eigenvalue. Its positive eigenvector $v=(v_X,v_Y)$
satisfies
$$E(v_Y)\le 4(R^2/\tau)|v_X|^2. \tag{A.13}$$

**Proof.** From $J_Sv=\lambda_+v$, the transverse block gives
$(\lambda_+I+T-E_D)v_Y=Bv_X$. Take the inner product with $v_Y$:
$$E(v_Y)=v_Y^{\!\top}Tv_Y\le |B|\,|v_X|\,|v_Y|+\|E_D\|\,|v_Y|^2
\le R\,|v_X|\,|v_Y|+\tfrac{3R}{4}|v_Y|^2,$$
using $\lambda_+>0$ and $v_Y^{\!\top}E_Dv_Y\le\|E_D\||v_Y|^2$. Apply Young's
inequality $R|v_X||v_Y|\le \tfrac{R^2}{\tau}|v_X|^2+\tfrac{\tau}{4}|v_Y|^2$
and the energy bound $|v_Y|^2\le E(v_Y)/\tau$:
$$E(v_Y)\le \tfrac{R^2}{\tau}|v_X|^2+\Big(\tfrac{\tau}{4}+\tfrac{3R}{4}\Big)|v_Y|^2
\le \tfrac{R^2}{\tau}|v_X|^2+\Big(\tfrac14+\tfrac{3}{8}\Big)E(v_Y),$$
where $\tau\ge 2R$ (from (A.11)) gives $\tfrac{3R}{4}\cdot\tfrac1\tau\le
\tfrac38$. Hence $\tfrac38 E(v_Y)\le\tfrac{R^2}{\tau}|v_X|^2$, i.e.
$E(v_Y)\le\tfrac83(R^2/\tau)|v_X|^2\le4(R^2/\tau)|v_X|^2$. The strict final
inequality leaves a fixed normalized margin, which persists under the exact
$C^1$ perturbation of Lemma A.4.5. $\blacksquare$

**Lemma A.4.2 (outward energy cone).** Let $\xi=X-\tfrac12$,
$\delta=1/(512R^3)$, and
$\mathcal C_E=\{0\le\xi\le\delta,\ E(Y)\le 4(R^2/\tau)\xi^2\}$. On $\mathcal C_E$:
$q\ge 2R\xi$ on the nonzero boundary, $|Y|\le \xi/K$, $F_X\ge 3\xi/4$, and
$G_E=E(Y)-4(R^2/\tau)\xi^2$ satisfies
$$\tfrac12\dot G_E\le -\tfrac23 q^2<0\quad\text{on }\{G_E=0\}. \tag{A.14}$$
Hence the outward half-branch stays in $\mathcal C_E$ until $\xi=\delta$.

**Proof.** On $G_E=0$, $E=4(R^2/\tau)\xi^2$ so
$q\ge\sqrt{\tau E}=2R\xi$ and $|Y|\le\sqrt{E/\tau}=2R\xi/\tau
\le \xi/(KR^4)$ (the full strength $\tau\ge2KR^5$ is used here, not merely
$\tau\ge2KR$ — this matters for the next bound when $R$ is large).
Then $|Xa\!\cdot\!Y|\le(\tfrac12+\delta)R|Y|\le(\tfrac12+\delta)\,\xi/(KR^3)
\le\xi/16$ (as $(\tfrac12+\delta)/(KR^3)\le 1/K\le 1/16$) and
$\tfrac12|Y^{\!\top}WY|\le \tfrac{R}{2}|Y|^2\le
\tfrac{R}{2}\cdot\tfrac{\xi^2}{K^2R^8}\le\tfrac{\xi}{16}\cdot
\tfrac{8\xi}{K^2R^7}\le\xi/16$ (as $8\xi\le 8\delta\le 1\le K^2R^7$) and —
using that $Q$ and $\nabla Q$ vanish at $S$ by (A.10), so
$|Q_X(X,Y)|\le\|Q\|_{C^2}\,\operatorname{dist}((X,Y),S)\le R(\xi+|Y|)$ on
$\mathcal C_E$ —
$$|rQ_X|\le rR(\xi+|Y|)\le rR\,\xi\big(1+\tfrac{1}{KR^4}\big)\le 2rR\,\xi
\le\frac{2\varepsilon_0}{R^4}\,\xi\le\frac{\xi}{16},$$
which is proportional to $\xi$ and hence uniform down to $\xi=0$ (a v2
correction: bounding $rQ_X$ by the constant $rR$ cannot beat $\xi/16$ as
$\xi\downarrow0$). Since $X^2-\tfrac14=\xi(1+\xi)\ge\xi$,
$F_X\ge\xi-3\xi/16\ge 3\xi/4$. For the flux,
$\tfrac12\dot G_E=(TY)\!\cdot\! F_Y-4(R^2/\tau)\xi F_X$; the moving-boundary
term is $\le0$ by $F_X\ge0$. In $(TY)\!\cdot\!F_Y$ the leading term is
$-q^2$. Dividing the four adverse terms by $q^2$, using
$q\ge2R\xi$, $|Y|\le2R\xi/\tau$, $|X|\le\tfrac12+\delta$,
$X^2-\tfrac14=\xi(1+\xi)$, and — for the last — that $\nabla Q$ vanishes at
$S$ by (A.10), so $|\nabla_Y Q|\le R(\xi+|Y|)$ on $\mathcal C_E$ by the
$C^2$ bound:
$$\begin{aligned}
\frac{|(TY)\cdot\tfrac12(X^2-\tfrac14)a|}{q^2}
&\le\frac{\tfrac12\xi(1+\xi)\,R\,q}{q^2}
=\frac{\xi(1+\xi)R}{2q}\le\frac{1+\xi}{4}\le\frac{1+\delta}{4}\le\frac{257}{1024},\\[2pt]
\frac{|(TY)\cdot XWY|}{q^2}
&\le\frac{|X|\,R\,|Y|\,q}{q^2}=\frac{|X|\,R\,|Y|}{q}
\le\frac{(\tfrac12+\delta)\,R\,(2R\xi/\tau)}{2R\xi}
=\frac{(\tfrac12+\delta)R}{\tau}
\le\frac{257}{1024K},\\[2pt]
\frac{|(TY)\cdot\tfrac12C[Y,Y,\cdot]|}{q^2}
&\le\frac{\tfrac12 R|Y|^2 q}{q^2}\le\frac{R(2R\xi/\tau)^2}{2\cdot2R\xi}
=\frac{R^2\xi}{\tau^2}\le\frac{\delta}{4K^2R^8}\le\frac{1}{2048K^2},\\[2pt]
\frac{|(TY)\cdot r\nabla_YQ|}{q^2}
&\le\frac{rR(\xi+|Y|)}{q}\le\frac{rR\,\xi(1+2R/\tau)}{2R\xi}
=\frac{r\,(1+2R/\tau)}{2}\le 2\varepsilon_0,
\end{aligned}$$
where the second line uses $q\ge2R\xi$ and $|Y|\le2R\xi/\tau$, the third uses
$\tau\ge2KR^5$ and $\xi\le\delta=1/(512R^3)$, and the fourth uses
$r\le\varepsilon_0/R^5\le\varepsilon_0$ and $2R/\tau\le 1/(KR^4)\le1$. The
four brackets $\frac{257}{1024}$, $\frac{257}{1024K}$,
$\frac{1}{2048K^2}$, $2\varepsilon_0$ sum to $<\frac13$ (exact Fraction
arithmetic, re-verified in `scripts/arithmetic_ledgers.py`). Hence
$\tfrac12\dot G_E\le-(1-\tfrac13)q^2$. $\blacksquare$

**Lemma A.4.3 (outward energy strip and level bound).** Let
$\mathcal S_E=\{\tfrac12+\delta\le X\le\tfrac54,\ E(Y)\le E_*\}$,
$E_*=\tfrac{9}{16}(R^2/\tau)$. On its lateral boundary, $q\ge 3R/4$,
$|Y|\le 3R/(4\tau)$, $F_X\ge 3\delta/4>0$, and
$$\tfrac12\dot E\le -\tfrac1{10}q^2<0\quad\text{on }\{E=E_*\}. \tag{A.15}$$
The branch therefore reaches $X=\tfrac54$ with
$P(\tfrac54,Y)>\tfrac{49}{192}-\tfrac{1}{768}=\tfrac{65}{256}>\tfrac14>0=P(M)$:
it reaches a point strictly above the maximum level and cannot terminate at
$M$.

**Proof.** At $\xi=\delta$, $E\le4(R^2/\tau)\delta^2=\frac{4\delta^2}{9/16}E_*
=\frac{4\delta^2}{9/16}E_*<E_*$ since $\delta=1/(512R^3)<3/8$. On
$\{E=E_*\}$, $q\ge\sqrt{\tau E_*}=3R/4$ and the adverse-to-$q^2$ ratios are
$\frac78$, $\frac{5}{8K}$, $\frac{3}{32K^2}$, $\frac43\varepsilon_0$, summing
to $<\frac9{10}$ (re-verified in `scripts/arithmetic_ledgers.py`), giving
(A.15); invariance plus $F_X>0$ gives arrival at $X=\tfrac54$. The level:
$\frac12 Y^{\!\top}TY\le\tfrac12 E_*=9R^2/(32\tau)\le9/(64KR^3)<1/(4K)$ —
this is the point of the energy adaptation: no bound on $\lambda_{\max}(T)$
is needed, because $E_*$ itself is $O(R^2/\tau)$. The remaining adverse level
terms total $21/(32K)+1/K^2+3\varepsilon_0+3/8192$; the complete bracket is
$9869/16777216<1/768$ (exact Fraction arithmetic, re-verified). $\blacksquare$

**Lemma A.4.4 (inward capture — replacing the v1.2 tube equations).** Let
$m=2R/\tau$ and consider the tapered tube
$\mathcal T_{\mathrm{in}}=\{-\tfrac12<X<\tfrac12,\ |Y|\le m(\tfrac14-X^2)\}$.
Then:

(i) The inward half-branch of $S$ enters $\mathcal T_{\mathrm{in}}$: the
stable axial eigenvector at $S$ satisfies the slope bound
$|v_Y|/|v_X|<2R/\tau=m$ (the same Young-inequality computation as Lemma
A.4.1, applied to the stable eigenvalue), and inside the tube
$F_X=X^2-\tfrac14+O(R|Y|,R|Y|^2,rR)<0$ whenever
$\tfrac14-X^2\ge\delta_0$ (with $\delta_0=1/(512R^4)$ as in (ii); the
arithmetic is displayed at the end of (iii) below), so the branch moves
leftward into the tube.

(ii) *Lateral flux.* On $\{|Y|=m(\tfrac14-X^2)\}$ with
$\tfrac14-X^2\ge\delta_0:=\tfrac{1}{512R^4}$, the function
$G=Y^2-m^2(\tfrac14-X^2)^2$ satisfies $\dot G<0$. (The threshold $\delta_0$
is chosen so that $2Kr\le\delta_0$: indeed
$2Kr\le 2K\varepsilon_0/R^5=\tfrac{1}{512R^5}\le\tfrac{1}{512R^4}=\delta_0$,
using $rR^5\le\varepsilon_0$, $K\varepsilon_0=1/1024$, and $R\ge1$. This is
what makes the $r$-adverse terms $\le 1/(4K)$ below.) Indeed, on $G=0$,
$$\dot G=2Y\!\cdot\!F_Y+4m^2X(\tfrac14-X^2)F_X.$$
The leading term $-2Y^{\!\top}TY\le-2\tau m^2(\tfrac14-X^2)^2$. The adverse
terms, using $|Y|=m(\tfrac14-X^2)$, $|X|\le\tfrac12$,
$\tfrac14-X^2\le\tfrac14$, and
$|F_X|\le(\tfrac14-X^2)+\tfrac{R}{2}|Y|+\tfrac{R}{2}|Y|^2+rR$:
$$\begin{aligned}
|2Y\!\cdot\!\tfrac12(X^2-\tfrac14)a|&\le R|Y|(\tfrac14-X^2)=mR(\tfrac14-X^2)^2,\\
|2Y\!\cdot\!XWY|=2|X|\,|Y^{\!\top}WY|&\le 2|X|\,m^2R(\tfrac14-X^2)^2\le m^2R(\tfrac14-X^2)^2,\\
|Y\!\cdot\!C[Y,Y,\cdot]|&\le m^3R(\tfrac14-X^2)^3\le \tfrac14 m^3R(\tfrac14-X^2)^2,\\
|2Y\!\cdot\! r\nabla Q|&\le 2mrR(\tfrac14-X^2),
\end{aligned}$$
and dividing the last by the leading $2\tau m^2(\tfrac14-X^2)^2$ gives the
ratio $rR/(\tau m(\tfrac14-X^2))\le rR/(2R\,\delta_0)=r/(2\delta_0)\le
1/(4K)$, using $\tau m=2R$ and $2Kr\le\delta_0$. The moving-boundary
term satisfies (factoring $(\tfrac14-X^2)^2$ out of the $|F_X|$ bound, with
$(\tfrac14-X^2)\le\tfrac14$ in the quadratic term and
$(\tfrac14-X^2)^{-1}\le\delta_0^{-1}$ in the last)
$$|4m^2X(\tfrac14-X^2)F_X|\le 2m^2(\tfrac14-X^2)^2\Big(1+\frac{Rm}{2}+\frac{Rm^2}{8}+\frac{rR}{\delta_0}\Big),$$
where $rR/\delta_0\le rR\cdot 512R^4=512rR^5\le 512\varepsilon_0
=\tfrac{1}{2K}=\tfrac1{8192}$.
Collecting and dividing by $2\tau m^2(\tfrac14-X^2)^2$, the adverse ratio is
at most
$$\frac{R}{2\tau m}+\frac{R}{2\tau}+\frac{mR}{8\tau}+\frac{1}{4K}
+\frac{1}{\tau}\Big(1+\frac{Rm}{2}+\frac{Rm^2}{8}+\frac{1}{8192}\Big)
=\frac14+\frac{R}{2\tau}+\frac{R^2}{4\tau^2}+\frac{1}{4K}
+\frac{1}{\tau}\Big(\frac{8193}{8192}+\frac{R^2}{\tau}+\frac{R^3}{2\tau^2}\Big),$$
where $R/(2\tau m)=R/(4R)=\tfrac14$ and $m=2R/\tau$. Since
$\tau\ge 2KR^5+2R$ and $K=4096$: $R/(2\tau)\le 1/(4K)$,
$R^2/(4\tau^2)\le 1/(16K^2R^6)$, $1/\tau\le 1/(2K)$, and the final bracket is
$\le\tfrac{8193}{8192}+\tfrac{1}{2K}+\tfrac{1}{8K^2}\le\tfrac32$, so the total ratio is
$\le\tfrac14+\tfrac{1}{4K}+\tfrac{1}{16K^2}+\tfrac{1}{4K}
+\tfrac{1}{2K}\big(\tfrac32+\tfrac{1}{2K}+\tfrac{1}{8K^2}\big)
<\tfrac14+\tfrac{3}{K}<\tfrac13$. Hence $\dot G\le-\tfrac23\tau
m^2(\tfrac14-X^2)^2<0$: trajectories cannot exit through the lateral walls.

(iii) *Capture.* At $M$ the Jacobian
$J_M=\begin{pmatrix}A_M&B_M^{\!\top}\\B_M&-T+E_D\end{pmatrix}$ is negative
definite: for $w=(w_X,w_Y)$,
$w^{\!\top}J_Mw\le-\tfrac34 w_X^2+2R|w_X||w_Y|-\tau|w_Y|^2+\tfrac{3R}{4}|w_Y|^2
\le-\tfrac38 w_X^2-\tfrac{\tau}{2}|w_Y|^2$
by Young's inequality and $\tau\ge2KR^5+2R$. Hence, on the ball
$B(M,\rho_B)$, $\rho_B:=1/(16R)$, the Lyapunov function
$V=(X+\tfrac12)^2+|Y|^2$ satisfies
$\dot V\le-2\min(\tfrac38,\tfrac{\tau}{2})\,V+O(RV^{3/2})<0$ on
$\partial B(M,\rho_B)$, so the ball is forward-invariant and every trajectory
entering it converges to the sink $M$. Let $X_1=-\sqrt{\tfrac14-\delta_0}$,
the left endpoint of the tube domain (a v2 correction: at
$X=-\tfrac12+\delta_0$ one has $\tfrac14-X^2=\delta_0-\delta_0^2<\delta_0$,
which is *outside* the lateral-flux domain — the flux was proved for
$\tfrac14-X^2\ge\delta_0$, i.e. for $X\ge X_1$). Note
$X_1-(-\tfrac12)=\tfrac12-\sqrt{\tfrac14-\delta_0}
=\delta_0/(\tfrac12+\sqrt{\tfrac14-\delta_0})\le2\delta_0$. The tube
cross-section at $X=X_1$ has half-width
$$m(\tfrac14-X_1^2)=m\delta_0=\frac{2R/\tau}{512R^4}
=\frac{1}{256\,\tau R^3}<\frac{1}{16R}=\rho_B,$$
and $|X_1+\tfrac12|\le2\delta_0=\tfrac{1}{256R^4}<\tfrac{1}{16R}=\rho_B$;
since $X$ decreases monotonically along the inward branch inside the tube
($F_X<0$ there: the adverse-to-$(\tfrac14-X^2)$ ratio is at most
$Rm+\tfrac{Rm^2}{8}+\tfrac{rR}{\delta_0}
\le\tfrac{1}{KR^3}+\tfrac{1}{8K^2R^7}+\tfrac1{8192}<1$), the branch reaches
the section $X=X_1$ with
$(X_1+\tfrac12)^2+|Y|^2\le 4\delta_0^2+\big(\tfrac{1}{256\tau R^3}\big)^2
<\rho_B^2$, i.e. inside $B(M,\rho_B)$, and is captured. $\blacksquare$

**Lemma A.4.5 (exact-field robustness, self-contained).** Every strict
inequality above has a margin of at least $\tfrac23 q^2$ (Lemma A.4.2),
$\tfrac1{10}q^2$ (A.4.3), $\tfrac13$ of the leading term (A.4.4(ii)), or a
fixed fraction of $\rho_B^2$ (A.4.4(iii)); the adverse brackets were
verified at $\le\tfrac13$, $\le\tfrac9{10}$, $\le\tfrac{1}{768}$ of their
budgets. Let $\tilde F=F+\mathrm{err}$ be a pin-preserving perturbation with
$\|\mathrm{err}\|_{C^1(D^+)}\le\eta_R$, $\eta_R=1/(8192R^3)$ — so $M,S$
remain critical points of the perturbed potential with unchanged values, in
particular $\mathrm{err}(S)=0$, and
$|\mathrm{err}_X|,|\mathrm{err}_Y|\le\eta_R\,\operatorname{dist}((X,Y),S)$.
Its effect on each flux is bounded directly (no external module is used):
(a) in $\mathcal C_E$, $\operatorname{dist}\le\xi+|Y|\le\xi(1+1/(KR^4))$, so
the $F_X$-error is $\le\eta_R\xi(1+1/(KR^4))\le\xi/16$ (uniformly down to
$\xi=0$, exactly like the $rQ_X$ term) and the flux error is
$|(TY)\cdot\mathrm{err}_Y|+4(R^2/\tau)\xi|\mathrm{err}_X|
\le q\eta_R\xi(1+\tfrac{1}{KR^4})+\tfrac{4R^2}{\tau}\xi\cdot\eta_R\xi
(1+\tfrac{1}{KR^4})\le 3q\eta_R\xi$; divided by $q^2\ge 2R\xi q$ this is
$\le\frac{3\eta_R}{2R}<\frac13\cdot\frac13$;
(b) on $\{E=E_*\}$, $\operatorname{dist}\le\tfrac34+\tfrac{3R}{4\tau}\le1$,
so $|\dot{\tilde E}-\dot E|\le 2q\eta_R$, and
$2\eta_R/q\le 2\eta_R/(3R/4)=\frac{8\eta_R}{3R}
=\frac{8}{3\cdot8192R^4}<\frac1{20}$;
(c) in $\mathcal T_{\mathrm{in}}$, $\operatorname{dist}\le1$, so
$|\dot{\tilde G}-\dot G|\le(2|Y|+4m^2(\tfrac14-X^2))\eta_R
\le 6m(\tfrac14-X^2)\eta_R$, against the floor
$\tfrac23\tau m^2(\tfrac14-X^2)^2$: the ratio is
$9\eta_R/(\tau m(\tfrac14-X^2))\le 9\eta_R/(2R\delta_0)
=9\cdot512R^4/(2R\cdot8192R^3)=9/32<\tfrac12$;
(d) the Lyapunov margin in $B(M,\rho_B)$ absorbs
$2\rho_B\eta_R\ll\rho_B^2/4$.
For the conditioned side-24 field,
$\|F_{\mathrm{exact}}-F\|_{C^1(D^+)}\le C_+r$ on the fifth-derivative good
event, and $C_+r\le\eta_R$ for
$r\le r_{\mathrm{mat}}=\min\{1,[1/(8192C_+\varepsilon_0^{3/5})]^{5/2}\}$.
$\blacksquare$

**Theorem A.4.6 (energy-adapted capture and escape, consolidated).** Under
(A.10)–(A.11) and $0<r\le r_{\mathrm{mat}}$: (i) the $M$-ward unstable
half-branch of $S$ converges to $M$ through $\mathcal T_{\mathrm{in}}$;
(ii) the outward half-branch remains in $\mathcal C_E\cup\mathcal S_E$ and
reaches $X=\tfrac54$ at a level strictly above $f(M)$, hence cannot terminate
at $M$; (iii) both hold for the exact conditioned field on the
fifth-derivative good event. No bound on $\lambda_{\max}(T)$ is assumed or
used.

**Status: HISTORICAL — proof superseded by Part F.4 (V3).** The "CLOSED"
label below is the pre-V3 record. The proof of Theorem A.4.6 above is
replaced by the threshold-free inward-tube proof of Part F.4: the V2
auxiliary threshold $\delta_0$ used here rested on the false inequality
$h(-\tfrac12+\delta_0)\ge\delta_0$ (the left side is
$\delta_0-\delta_0^2$), and the pin-preserving robustness budget of Part
F.4 (with the corrected tube cost $12(\eta_R/R)(R^2/\tau)h^2$) governs.
The deterministic statement of Theorem A.4.6 itself survives under the
explicit hypotheses of Part F.4 item 3; the Palm-transfer consequence
does not (open premise).

> **Status: CLOSED** *(pre-V3 historical record).* The proof above is
> self-contained: Lemma A.4.4 supplies
the inward-tube flux and convergence derivation that Module J imported from
the superseded v1.2 text, and every rational margin is re-verified from exact
Fraction arithmetic in `scripts/arithmetic_ledgers.py` (which prints
`ALL_ASSERTIONS_PASS`). Hostile Q8 is answered by construction: only
$\tau=\lambda_{\min}(T)$ enters.

---

## Part B — Computations displayed and re-executed (Gap items 6 and 7)

All scripts below are in `scripts/`, were executed in this build environment
(Python 3.12, SymPy 1.14, mpmath, SciPy), and print `ALL_ASSERTIONS_PASS`.
Transcripts: `scripts/*.out.txt`.

### B.1 Gap item 6. **The triple-contact elimination, displayed.**

Two independent pieces are supplied.

**(a) The covariance side — verified from first principles.** The limiting
collar and singular-near conditional covariances invoked by Module B are
*derived* (not transcribed) from $K(z)=e^{-|z|^2/2}$ via
$\operatorname{Cov}(\partial^\alpha f,\partial^\beta f)=(-1)^{|\beta|}
\partial^{\alpha+\beta}K(0)$ in `scripts/contact_covariance_verification.py`:
$$\det\operatorname{Cov}(P)=12,\qquad
\operatorname{Var}(L_1\mid P)=\tfrac{\rho^2(4X^2+\rho^2)}{2},\qquad
C_\perp=\begin{pmatrix}2Y^2+Z^2&YZ\\ YZ&Y^2+2Z^2\end{pmatrix},\ \det C_\perp=2\rho^4,$$
$$\det\operatorname{Cov}(G_{\mathrm{col}}\mid P)=\rho^6(4X^2+\rho^2),$$
$$C_{00}=\tfrac{\rho^2(c^2+\rho^2)(c^2+3\rho^2)}{72},\quad
C_{01}=-\tfrac{c\rho^2(c^2+\rho^2)}{6},\quad
C_{11}=\tfrac{\rho^2(4c^2+\rho^2)}{2},$$
$$\det C_{(L_0,L_1)}=\tfrac{\rho^6(c^2+\rho^2)(3c^2+\rho^2)}{48},\qquad
\det\operatorname{Cov}(L_0,L_1,L_2,L_3\mid P)=\tfrac{\rho^{10}(c^2+\rho^2)(3c^2+\rho^2)}{24}.$$
Every displayed formula of Module B §§3 and 5 matches exactly. (Answers the
covariance half of hostile Q4 and Q5.)

**(b) The Hessian-determinant elimination — displayed in a declared chart.**
`scripts/triple_contact_elimination.py` solves the universal two-dimensional
pin system symbolically: cubic $p(X,Y)$, critical points $M=(-\tfrac12,0)$,
$S=(\tfrac12,0)$, $p(M)=0$, $p(S)=-\kappa/6$, third critical point
$P=(c/\eta,s/\eta)$, $\eta=r/t$, with the transverse endpoint curvatures
$q_M,q_S$ prescribed (in the application they are the $O(1)$ contact-law
blocks). The elimination is unique and exact:
$$-\det H_M=\frac{\kappa^2(4c^2-\eta^2)^2+4\kappa s^2\big[(12c^2+\eta^2)q_M+(4c^2-\eta^2)q_S\big]+4s^4(q_M-q_S)^2}{64\,c^2s^2}, \tag{B.1}$$
$$-\det H_S=\frac{\kappa^2(4c^2-\eta^2)^2-4\kappa s^2\big[(4c^2-\eta^2)q_M+(12c^2+\eta^2)q_S\big]+4s^4(q_M-q_S)^2}{64\,c^2s^2},$$
i.e. **the $\kappa$-linear term changes sign** relative to $-\det H_M$ (a v2
correction: an earlier draft printed it positive as a purported $M
\leftrightarrow S$ mirror — that was wrong; the true formulas are not
mirrors of each other, since the mirror exchanges the *values*
$p(M)=0$, $p(S)=-\kappa/6$ as well). Both are printed verbatim from the
symbolic elimination (script transcript lines 1–2), and the v5 verifier now
checks the printed coefficients against the symbolic output term by term.
$$\det H_P=\frac{\eta\big[\kappa^2(\eta^4-8c^2\eta^2+16c^4)+\cdots\big]+16c\,(q_M^2-q_S^2)s^4+\cdots}{64\,c^2\eta s^2}
\quad\text{(exact form in script output)}.$$
The script asserts: common denominators $64c^2s^2$ (endpoints) and
$64c^2\eta s^2$ (third point); the $\kappa^2$ coefficient
$(4c^2\mp\eta^2)^2/(64c^2s^2)$; the symmetric $(q_M-q_S)^2s^4$ structure;
product of exact degree $6$ in $\kappa$; and that the only angular
singularities are the axis strata $c=0$, $s=0$ (removable after the
$\rho$-blow-up, consistent with $\rho^6$-type factors above) and the collision
strata $2c=\pm\eta$.

> **DISPOSITION — RESOLVED BY REPLACEMENT (Gap item 6).** The historical
> LS-DER-024 normalization
> $\det H=B\,\kappa^2\eta^2/(64c^2s^2)$ per Hessian is **not** recovered in
> this or the other natural charts tested: in declared charts the endpoint
> determinants are generically $O(\kappa^2)$ as $\eta\to0$, not $O(\kappa^2\eta^2)$.
> The load-bearing consequence used by the actual side-24 proof is the
> anisotropic envelope $[r(r+t^2)]^2(r^2+t^3)$ (Module B eq. (41), interface
> G6′), whose derivation is displayed in Module B §6 and whose six radial
> terms integrate to $O(r^3)$ — that argument does not require the
> per-Hessian $\kappa^2\eta^2$ factor. What remains open is cosmetic-but-
> necessary: either reproduce LS-DER-024's exact chart conventions so that
> its $B_M,B_S,B_X$ formulas hold as printed, or replace the citation of
> that elimination by (B.1) and the G6′ envelope throughout. Since the older
> source also contains the false loop reduction, the supplement recommends
> the latter. **Resolution adopted (v2 manuscript):** the LS-DER-024 citation
> is replaced throughout by display (B.1) together with the G6′ envelope of
> Module B §6. The historical normalization is retained only as a provenance
> note. **Status: CLOSED AS CITATION REPLACEMENT — with an explicit
> dependency statement (v2 precision).** What this supplement proves: the
> chart algebra (B.1) and the covariance side from first principles. What it
> does **not** re-prove: (i) the derivation of the G6′ anisotropic envelope
> $[r(r+t^2)]^2(r^2+t^3)$ from the corrected formulas across every singular
> chart; (ii) the uniform finite-$r$ transfer to the conditioned field;
> (iii) removability of the angular-axis singularities and collision-stratum
> control; (iv) the $O(r^3)$ angular/radial integration of the six terms.
> Items (i)–(iv) are the content of frozen Modules B §6, E, and I §3,
> unchanged by this supplement; the repair is that they no longer rest on a
> false citation but on (B.1) plus those modules, stated as an explicit
> dependency. Re-deriving (i)–(iv) from the corrected formulas within this
> supplement is a tracked strengthening, not claimed here.

### B.2 Gap item 7. **The cone constant $D_2=29/6-\sqrt6$ and the Euclidean coefficient, derived.**

**Derivation displayed.** Conditional on $h_t=0$, Module C (36) gives
$Q=G_2+\sqrt{2/3}\,ZI_2$ with $G_2$ standard GOE$_2$ (diagonal $N(0,2)$,
off-diagonal $N(0,1)$, independent) and $Z\sim N(0,1)$ independent. This law
is itself *derived* in `scripts/goe_cone_coefficient.py`: the conditional
covariance $\Gamma_t=\Sigma_q-C_{qh}\Sigma_h^{-1}C_{hq}$ is computed from the
kernel and equals
$\begin{pmatrix}8/3&0&2/3\\0&1&0\\2/3&0&8/3\end{pmatrix}$ (eigenvalues
$\{1,2,10/3\}$, matching Module D), which is exactly the covariance of
$(g_1+\sqrt{2/3}Z,\ g_2,\ g_3+\sqrt{2/3}Z)$. Also derived from the kernel:
$\Sigma_g=I_3$, $\Sigma_h=\operatorname{diag}(3,1,1)$, $\sigma_t^2=1/24$.

**GOE-plus-scalar cone integral.** By orthogonal invariance the scalar shift
$Z$ translates both eigenvalues: with eigenvalues $\lambda_1,\lambda_2$ of
$G_2$ (joint density $C|\lambda_1-\lambda_2|e^{-(\lambda_1^2+\lambda_2^2)/4}$)
and $s=\sqrt{2/3}$,
$$D_2=\mathbb E\Big[\prod_{i=1}^2(\lambda_i+sZ)^2\,\mathbf 1_{\{Z<-\max_i\lambda_i/s\}}\Big], \tag{B.2}$$
**Closed-form evaluation (exact — proved, not numerics).**
`scripts/d2_closed_form.py` evaluates (B.2) in closed form. The key
observation is that in trace/difference coordinates
$T=\lambda_1+\lambda_2$, $D=\lambda_1-\lambda_2\ge0$ (jacobian $1/2$), the
GOE$_2$ eigenvalue density becomes proportional to
$D\,e^{-(T^2+D^2)/8}$ — so $T\sim N(0,4)$ is *independent* of $D$ — and both
the determinant and the cone condition depend on $(T,Z)$ only through the
single linear combination
$$W=T+2sZ\sim N(0,\,\tfrac{20}{3}):\qquad
\det Q=\frac{W^2-D^2}{4},\qquad Q\prec0\iff W<-D.$$
Hence, with $M_0=\Phi(m)$, $M_2=M_0-m\phi(m)$, $M_4=3M_0-(m^3+3m)\phi(m)$ the
standard truncated normal moments at $m=-D/\sqrt{20/3}$,
$$D_2=\frac1{16}\int_0^\infty \frac{D\,e^{-D^2/8}}{4}\,
\Big[\tfrac{400}{9}M_4-\tfrac{40}{3}D^2M_2+D^4M_0\Big]\,dD. \tag{B.2′}$$
The $\phi$-terms are pure Gaussian moments (combined exponent
$a+c^2/2=\tfrac18+\tfrac{3}{40}=\tfrac15$); the $\Phi$-terms are the
half-line integrals $\int_0^\infty D^{k}\Phi(-cD)e^{-D^2/8}\,dD$ with $k$
odd, which are elementary by integration by parts:
$$\textstyle I_1=2-\frac{\sqrt6}{2},\qquad I_3=16-\frac{21\sqrt6}{4},\qquad
I_5=256-\frac{747\sqrt6}{8}.$$
Term-by-term recombination (executed symbolically in the script) gives
$$\boxed{\,D_2=\frac{29}{6}-\sqrt6\,}\qquad\text{exactly,}$$
with the script asserting equality to the target and to the manuscript
decimal. Two earlier independent numerical evaluations (adaptive quadrature
of (B.2), $3\times10^6$-draw Monte Carlo) agree with this value to
$1.2\times10^{-8}$ and $4\times10^{-4}$ respectively; they are retained in
`scripts/goe_cone_coefficient.py` as cross-checks but are no longer
load-bearing. **Every constant in the recombination of $c_{3,\infty}$ is now
exact/symbolic.**

**Coefficient recombination.** `scripts/goe_cone_coefficient.py` also verifies
symbolically/numerically:
$$c_{3,\infty}=\frac{2^{1/6}3^{1/3}(29\sqrt6-36)\Gamma(1/6)}{432\pi^{5/2}}
=2^{-23/6}3^{-8/3}(29\sqrt6-36)\Gamma(1/6)\pi^{-5/2}
=0.041775931840598343342936665428575556466681519661\ldots$$
(matching the manuscript decimal to all 46 printed digits), and
$$P_3(L)=-\frac{L^2(10L^4-147L^2+315)}{105}=-\frac{2}{21}L^6+\frac75L^4-3L^2,\qquad
P_3(24)=-\frac{620813376}{35}\ \text{(exact)}. \tag{B.3}$$
The first-variation identity (B.3) is the output of the rotational-projection
algebra of Module H §4; the polynomial identity and evaluation are verified
symbolically, and the tail bounds $q=e^{-288}<10^{-125}$,
$(3/2)^6q^5<\tfrac12$, $\|R_{\mathrm{jet}}\|<10^{-230}$ arithmetic are
re-executed in `scripts/arithmetic_ledgers.py`.

**First variation, derived from first principles (residual closed).**
`scripts/first_variation_derivation.py` now performs the full chain-rule
computation through the nonlinear functional, closing the former residual.
With the perturbed kernel
$K_p(z)=e^{-|z|^2/2}\big[1+2q\sum_i(\cosh(Lz_i)-1)\big]$, $q=e^{-L^2/2}$,
the script differentiates the directional moments of $K_p$ at $q=0$,
$$\delta m_{2k}=2(-1)^k\big(\operatorname{He}_{2k}(L)-\operatorname{He}_{2k}(0)\big),$$
and Haar-projects via the spherical average
$\mathbb E[\prod_i t_i^{2a_i}]=\prod_i(2a_i-1)!!\big/(3\cdot5\cdots(2n+1))$,
recovering Module H (14) **exactly and independently**:
$$\delta a=-2L^2,\qquad \delta m_4=\tfrac65L^4-12L^2,\qquad
\delta\chi=-\tfrac67L^6+18L^4-90L^2.$$
It then computes the *pure* partials of $\log c$ through the full nonlinear
functional (including the homogeneity-of-degree-2 term of the cone moment
$D(\Gamma)$, which contributes $+2/m_4$ to the $m_4$ partial):
$$\partial_a\log c=-\tfrac12,\qquad \partial_{m_4}\log c=-\tfrac12,\qquad
\partial_\chi\log c=\tfrac19,$$
so that $\delta\log c=-\tfrac12\,\delta a-\tfrac12\,\delta m_4+\tfrac19\,\delta\chi$.
The Module H display (15), $\delta\log c=\delta\Omega/9+\delta m_4/6-(13/6)\delta a$,
is exactly this expression regrouped via $\delta\Omega=\chi\,\delta a+a\,\delta\chi-2m_4\,\delta m_4$;
the script asserts both forms equal $P_3(L)$ symbolically. A scale-family
consistency check gives $d\log c/d\log\lambda=3/2$, matching critical-density
scaling. **Status: CLOSED.**

---

## Part C — Editorial repairs (Gap items 8–11)

### C.1 Gap item 8. **Corrected Fourier prefactor in Module F.**

Poisson summation for $g(z)=e^{-|z|^2/2}$ with $\hat g(k)=(2\pi)^{3/2}e^{-|k|^2/2}$ gives
$$K_{24}(z)=Z_{24}^{-1}\sum_{n\in\mathbb Z^3}g(z+24n)
=\frac{(2\pi)^{3/2}}{24^3 Z_{24}}\sum_{m\in\mathbb Z^3}e^{-2\pi^2|m|^2/24^2}\,e^{2\pi i m\cdot z/24}, \tag{C.1}$$
so the correct spectral weight in Module F eqs. (1)–(2) is
$a_m=(2\pi)^{3/2}24^{-3}e^{-2\pi^2|m|^2/24^2}$ (the displayed extract omitted
the prefactor $(2\pi)^{3/2}24^{-3}$). The Gram identity (4) becomes
$\operatorname{Var}[T(f)]=Z_{24}^{-1}\frac{(2\pi)^{3/2}}{24^3}\sum_m
e^{-2\pi^2|m|^2/24^2}|\hat T(m)|^2$. Since the omitted factor is a positive
constant, Theorem 3.1, Corollary 3.2, and every positivity conclusion of
Module F are unaffected; the correction is display-level but must enter the
journal version. **Status: CLOSED (replacement display supplied).**

### C.2 Gap item 9. **Restored Module E hypotheses.**

The extract of Module E begins at its §3; §§1–2 are restored as follows.
**(1) Jet ball.** $B=\{J:\|J-J_0\|_{\max}\le10^{-110}\}$, with $J_0$ the planar
Bargmann–Fock even jet through order six — this is the ball referenced as
"(1)" in Module E §§3–5 (the perturbation bound $8\cdot729\cdot10^{-110}$
appearing in its eq. (4) fixes the radius uniquely). **(2) Standing
hypotheses.** $K$ is a stationary, even, real-analytic covariance kernel with
$J_6(K)\in B$; the associated centered Gaussian field has the finite-family
derivative nondegeneracy of Module F (full positive Fourier support, which
the normalized side-24 kernel has by (C.1)); all compact charts, mark sets,
and constants are those of Module B. Under (1)–(2) the derivative envelope
used in Module E §§4–5 is the coarse coefficient-map bound
$\|dA_{\mathrm{col}}\|,\|dA_{\mathrm{sing}}\|\le10^{40}$ of Module D applied
to the analytic dependence of conditional covariances on $J_6$, giving
$A_{\mathrm{col}}(K)\ge1-10^{-70}$ and $A_{\mathrm{sing}}(K)\ge1/24-10^{-70}$.
**Status: CLOSED (hypotheses and the envelope's provenance restored).**

### C.3 Gap item 10. **Module K generic symbols defined.**

In Module K §5: $n=d-1$; for the isotropic kernel with one-dimensional
sectional covariance $K^{(1)}$,
$$a=-(K^{(1)})''(0)>0,\qquad m_4=(K^{(1)})^{(4)}(0)>0,\qquad \chi=-(K^{(1)})^{(6)}(0)>0,\qquad
\Omega=a\chi-m_4^2, \tag{C.2}$$
so $a=\operatorname{Var}(\partial_t f)$, $m_4=\operatorname{Var}(\partial_{tt}f)$,
$\chi=\operatorname{Var}(\partial_{ttt}f)$, and
$\Omega/a=\operatorname{Var}(\partial_{ttt}f\mid\partial_t f)$ is the
conditional third-derivative variance (for Bargmann–Fock:
$a=1$, $m_4=3$, $\chi=15$, $\Omega=6$ — consistent with $\sigma_t^2=1/24$
via $\Omega/(2a)=3=72\sigma_t^2$). $\beta=m_4/3$ is the transverse
second-derivative variance parameter, and
$D_n=\mathbb E[(\det M_n)^2\mathbf 1_{\{M_n\prec0\}}]$ is the centered
negative-definite determinant moment of the conditional transverse Hessian in
dimension $n$, exactly as in Module C (18). **Status: CLOSED.**

### C.4 Gap item 11. **Density convention emphasized.**

Insert at the manuscript's definition of $\nu_{3,24}$:
> *Convention.* $\nu_{3,24}$ is the **first-moment intensity density per unit
> ordinary torus volume**: $\nu_{3,24}(\ell)\,d\ell$ is the expected number of
> finite nonessential superlevel $H_0$ bars per unit volume with lifetime in
> $[\ell,\ell+d\ell]$. It is not the probability density of the lifetime of a
> randomly sampled bar; such a law requires normalizing by the expected bar
> count per unit volume and specifying the sampling rule, neither of which is
> used in Theorem 2.1. The asymptotic $\nu_{3,24}(\ell)=c_{3,24}\ell^{-1/3}(1+o(1))$
> holds for the pointwise Kac–Rice representative of the Radon–Nikodym
> derivative of Proposition 1.2, i.e. for Lebesgue-a.e. $\ell$ in the
> sequential sense selected by that representative.

**Status: CLOSED (insert text supplied).**

---

## Part D — Gap item 12: systematic literature search

Searches executed 2026-08-01 (general web indexes; targeted queries: expected
persistence diagram density asymptotics, near-diagonal/short-lifetime laws,
Kac–Rice maximum–saddle pairing and elder rule, Bargmann–Fock persistence,
$H_0$ lifetime exponents for smooth Gaussian fields).

1. **Persistence densities for random fields/complexes.**
   Adler–Bobrowski–Borman–Subag–Weinberger (IMS Collections 6, 2010) —
   structure and simulation of persistence for random fields; explicitly notes
   the death side of $H_0$ bars is global and not determined locally. No
   near-diagonal asymptotic.
2. **Expected-diagram density existence.**
   Chazal–Divol (JoCG 10(2), 2019) — existence and kernel estimation of the
   density of expected persistence diagrams for broad random filtrations;
   no smooth-Gaussian-field near-diagonal power law.
3. **Kac–Rice critical-point asymptotics.**
   Klein–Agam (J. Phys. A 45, 2012) and Ancona–Gass–Letendre–Stecconi
   (arXiv:2501.10226) — critical-point correlations, Kac–Rice singularities,
   cumulant asymptotics; no elder-rule/persistence selection imposed.
   Hirsch–Lachièze-Rey (arXiv:2411.11429) — functional CLT for topological
   functionals of Gaussian critical points; different question.
4. **Gaussian-field topology/percolation.**
   Pranav (arXiv:2109.08721), Feldbrugge–van Engelen–van de Weygaert–Pranav–
   Vegter (2019), and the Bargmann–Fock percolation line
   (Duminil-Copin et al.) — Betti numbers, Euler characteristic, connectivity;
   no selected maximum–saddle lifetime law.
5. **Persistence-diagram probabilistic models.**
   Maroulas–Mike–Oballe (JMLR 20, 2019), Adler et al. point-process models,
   and the "universal null distribution" work (Nature Sci. Rep. 2023) treat
   diagram-level noise models or filtrations of point-cloud complexes
   (Čech/Rips), not smooth-field Morse persistence.

**Conclusion.** As of 2026-08-01, no located work proves a selected
near-diagonal short-lifetime $H_0$ intensity law
$\nu(\ell)=c\ell^{-1/3}(1+o(1))$ for a smooth Gaussian field, nor the
Bargmann–Fock coefficient $c_{3,\infty}$, nor a fixed-side periodization
correction of the form (2.10). This remains *evidence of novelty, not a
determination*: MathSciNet and zbMATH subscription searches and a
backward/forward citation pass from references 2, 3, and 6–7 of the
manuscript should still be executed before submission (this environment has
no MathSciNet/zbMATH access).

**Status: CLOSED at the level possible here** (RESIDUAL PREMISE: paywalled
database pass outstanding).

---

## Part E — Reproducibility repairs

1. **Coefficient-bound transcript (reproduction guide §9).** The disclosed
   source/result mismatch is resolved as recommended: the replacement ledger
   `scripts/arithmetic_ledgers.py` prints *both* scales, so source and
   transcript agree by construction:
   `TAYLOR_REMAINDER_AT_1e-113_UNNORMALIZED_SCALE < 5e-187`,
   `DECLARED_RELATIVE_HESSIAN_BOUND = 1e42`,
   `TAYLOR_REMAINDER_AT_1e-113 < 5e-185`. Module H §2.2 already documents
   that the single historical line referred to the unnormalized $10^{40}$
   scale; the theorem-level $|\varepsilon_{24}|<10^{-180}$ is unaffected.
2. **Environment record.** These ledgers were executed under Python 3.12 with
   SymPy 1.14, mpmath, NumPy/SciPy (exact versions in the run transcripts),
   repairing the "not executed / version not recorded" deficiencies for the
   regenerated checks.
3. **New checks added.** `contact_covariance_verification.py` (Module B
   covariances from first principles), `goe_cone_coefficient.py` ($D_2$, jet
   objects, $c_{3,\infty}$, $P_3(24)$), `triple_contact_elimination.py`
   (displayed elimination), `first_variation_derivation.py` (full first-variation
   chain-rule: $\delta a,\delta m_4,\delta\chi$ from kernel differentiation and
   Haar projection; pure partials through the nonlinear functional; exact
   recovery of $P_3(L)$; $\lambda$-scale check), `d2_closed_form.py` (exact
   closed-form evaluation of $D_2=29/6-\sqrt6$ via the trace/difference
   reduction (B.2′)). Each prints `ALL_ASSERTIONS_PASS`; transcripts in
   `scripts/*.out.txt`.

---

## Part F — V3 corrective integration (2026-08-01)

This part integrates the V3 corrective addendum `05_V3_CORRECTIVE_ADDENDUM.pdf`,
which is **embedded in the V4 package** (same directory as the rendered form
of this supplement) so the controlling source travels with the archive.
Where it conflicts with Parts A–B, **Part F governs**; the superseded text is
retained above as correction history, with HISTORICAL banners on every claim
that no longer holds. All exact algebra of this part is machine-verified by
the three scripts `verify_constrained_triple_contact.py`,
`audit_density_vs_cumulative.py`, `verify_capture_escape_budgets.py`
(transcripts in `scripts/*.out.txt`; each prints `ALL_ASSERTIONS_PASS`
together with an explicit scope limit). Since V4, every script check uses an
explicit raising helper rather than bare `assert`, and the verifier
`verifier/v7` re-executes all nine scripts in both normal and `python -O`
modes (23/23 checks, transcripts `verifier/v7/*.out.txt`).

### F.1 Elder mark, V3 form (supersedes A.1) — full statement and proof

This subsection is self-contained: every event, function, and constant used
below is defined here. The source is the embedded
`05_V3_CORRECTIVE_ADDENDUM.pdf` (§3), expanded so that no external draft is
needed.

**Setting and notation.** $X=\mathbb T^3_{24}$ is the flat torus with
injectivity radius $\iota=12$; $g\in C(X)$ (sup-norm topology) plays the
role of a sample field; $U_a(g)=\{x\in X:g(x)>a\}$ is the open superlevel
set. Fix a countable basis $\mathcal B=\{B_k\}$ of $X$ by rational
coordinate balls, and a countable dense set $D=\{q_j\}$. On the locus of
Morse functions, $M$ denotes a local maximum with $b=g(M)$ and $S$ an
index-two saddle (Morse index two: the Hessian has exactly one positive
eigenvalue) with $h=g(S)<b$.

**Definition F.1.1 (Borel component relation).**
$\mathcal R(g,a;x,y)$ holds iff there is a finite chain
$B_{k_1},\dots,B_{k_n}\in\mathcal B$ with
$\overline{B_{k_i}}\subset U_a(g)$, $x\in B_{k_1}$, $y\in B_{k_n}$, and
$B_{k_i}\cap B_{k_i+1}\neq\varnothing$.

*Lemma F.1.2.* $\mathcal R(g,a;x,y)$ iff $x,y$ lie in the same connected
component of $U_a(g)$; and $\mathcal R$ is Borel in $(g,a,x,y)$.

*Proof.* A chain lies in one component because each $\overline B$ is
connected and contained in $U_a(g)$. Conversely, $U_a(g)$ is open and $X$
is locally path-connected, so each component $C$ is open and
path-connected; for $x,y\in C$ cover a joining path (compact) by basis
balls whose closures lie in $U_a(g)$ — possible because $\mathcal B$ is a
basis and $U_a(g)$ open — and take a finite subcover ordered along the
path. Borelness: $\mathcal R(g,a;x,y)$ is the countable union, over all
finite index sequences, of the conditions
$\inf_{B_{k_i}}g>a$, $x\in B_{k_1}$, $y\in B_{k_n}$,
$B_{k_i}\cap B_{k_{i+1}}\neq\varnothing$; each map
$g\mapsto\inf_{B_k}g$ is 1-Lipschitz in sup norm, and the membership and
intersection conditions are clopen in $(x,y)$ and independent of $g$.
$\square$

**Definition F.1.3 (overshoot event).**
$\mathcal O(g,M,a)$ holds iff $g(M)>a$ and there exists $q_j\in D$ with
$\mathcal R(g,a;M,q_j)$ and $g(q_j)>g(M)$.

*Lemma F.1.4.* $\mathcal O(g,M,a)$ iff the component of $M$ in $U_a(g)$
contains a point $z$ with $g(z)>g(M)$ — i.e. iff $M$ is **not** the
representative (pointwise-maximal) point of its own superlevel component.
$\mathcal O$ is Borel.

*Proof.* If such $z$ exists, the component $C$ is open and
$C\cap\{g>g(M)\}$ is open and nonempty, so it contains some $q_j$;
Lemma F.1.2 gives $\mathcal R(g,a;M,q_j)$. The converse is immediate.
Borelness is by the countable definition over $q_j$ and the Borelness of
$\mathcal R$. $\square$

**Definition F.1.5 (generic locus).** $\mathcal G\subset C^2(X)$ is the set
of Morse functions all of whose critical values are distinct. Morse
functions are open and dense in $C^2(X)$, and the distinct-critical-value
condition removes a further closed nowhere-dense set, so $\mathcal G$ is a
Borel (in fact $G_\delta$-dense) locus; every sample field considered below
lies on $\mathcal G$ almost surely, and all marks are restricted to
$\mathcal G$.

**Definition F.1.6 (elder pair).** Sweeping the level $a$ downward, the
class of $M$ in the superlevel $H_0$ persistence module is born at $a=b$.
An index-two saddle $S$ at level $h$ either (merge) joins two distinct
components of $U_{h-q}$ or (attachment) operates within one component. On
a merge, the *elder rule* pairs the saddle with the younger of the two
classes — the one whose representative maximum has the smaller value —
and that class dies; the elder class survives with its representative
unchanged. $(M,S)$ is the *elder pair* iff the class born at $M$ dies at
$S$ under this rule.

**Definition F.1.7 (level-transition mark).**
$$e_{\mathrm{lev}}(g;M,S)=1\{b>h\}\cdot 1\Big\{\exists n\ge1,\ n^{-1}<b-h:\ \forall q\in\mathbb Q\cap(0,n^{-1}),\ \neg\mathcal O(g,M,h+q)\ \text{and}\ \mathcal O(g,M,h-q)\Big\}.$$
$e_{\mathrm{lev}}$ is Borel: the inner condition quantifies over countably
many rationals and the Borel events $\mathcal O$.

**Theorem F.1.1 (V3 Thm A.1).** On $\mathcal G$,
$e_{\mathrm{lev}}(g;M,S)=1$ iff $(M,S)$ is the elder pair.

*Proof.* Between consecutive critical values the superlevel sets deform
onto one another by the Morse deformation lemma (gradient flow with no
critical points in the slab), so the partition into components — and
hence the representative status of $M$, i.e. the truth value of
$\mathcal O(g,M,a)$ — is constant on each open interval between critical
values and can change only at a critical value. Since critical values are
distinct, $S$ is the unique critical point at level $h$; hence both sides
of the transition in $e_{\mathrm{lev}}$ are determined by the topological
change at $h$ alone. There are two cases.

(i) *Attachment.* The handle at $S$ operates within a single component of
$U_{h-q}$. Then the component structure below and above $h$ differs by an
internal operation that creates no new component and merges none, so the
representative of $M$'s component is unchanged: $\mathcal O(g,M,h-q)$
has the same truth value as $\mathcal O(g,M,h+q)$, and
$e_{\mathrm{lev}}=0$. Consistently, no class dies at $S$, so $(M,S)$ is
not an elder pair.

(ii) *Merge.* Two distinct components $C_1,C_2$ of $U_{h-q}$ (with
representatives of values $b_1>b_2$) merge at $S$. For $a=h+q$ the two
germs are separate components; for $a=h-q$ they form one component whose
representative is the elder (value $b_1$). The class born at the younger
representative (value $b_2$) dies at $S$. Now $M$ lies in at most one of
the merging components. If $M$ is the younger representative ($g(M)=b_2$):
above $h$, $M$ is the maximal point of its own germ component, so
$\neg\mathcal O(g,M,h+q)$; below $h$, $M$'s component contains the elder
representative of value $b_1>b_2=g(M)$, so $\mathcal O(g,M,h-q)$; the
transition occurs and $e_{\mathrm{lev}}=1$ — and $(M,S)$ is exactly the
elder pair. If $M$ is the elder representative ($g(M)=b_1$): below $h$
the merged component's representative is still $M$, so
$\neg\mathcal O(g,M,h-q)$ as well and $e_{\mathrm{lev}}=0$; the dying
class is the other one, so $(M,S)$ is not the pair. If $M$ is in neither
merging component, its component is untouched and $e_{\mathrm{lev}}=0$.
$\square$

**Sector probes and the certified radius.** On $\mathcal G$, write
$H=D^2g(S)$; $H$ has exactly one positive eigenvalue $\mu(H)>0$
(necessarily simple), with spectral projection $P_+(H)$ continuous in $H$
on the index-two locus. Choose $j(H)$ to be the smallest coordinate index
maximizing $\|P_+(H)e_j\|$ (nonzero, since $P_+(H)\neq0$) and set
$v(H)=P_+(H)e_{j(H)}/\|P_+(H)e_{j(H)}\|$: a Borel choice of unit
orientation of the unstable axis, uniquely defined up to sign. Let
$$\omega_n(g,S)=\sup_{x,y\in B(S,\,2^{-n}\iota)}\|D^2g(x)-D^2g(y)\|,$$
and let $N(g,S)$ be the least $n$ with $\omega_n(g,S)<\mu(H)/16$ (finite
because $D^2g$ is continuous; Borel because $\omega_n$ is lower
semicontinuous in $g$ and $\mu(H)$ is continuous). Set the dyadic radius
$\rho(g,S)=2^{-N(g,S)-4}\iota$ and the probes
$u_\pm=\exp_S(\pm\rho(g,S)\,v(H))$.

*Lemma F.1.8 (certificate).* On $B(S,\rho)$ the unstable eigenvalue of
$D^2g$ stays in $(\tfrac{15}{16}\mu,\tfrac{17}{16}\mu)$ and the negative
eigenvalues stay below $-\tfrac{15}{16}\lambda_*$, where
$-\lambda_*$ is the largest (least negative) negative eigenvalue of $H$.
Consequently $t\mapsto g(\exp_S(tv(H)))$ is strictly convex on
$|t|\le\rho$ with minimum value $h$ at $t=0$, and in Morse coordinates
adapted to $H$ the set $\{g>h\}\cap B(S,\rho)$ has exactly two connected
components, distinguished by the sign of the axial coordinate; $u_+$ and
$u_-$ lie one in each.

*Proof.* The eigenvalue windows follow from
$\|D^2g(x)-H\|\le\omega_N<\mu(H)/16$ on $B(S,2^{-N}\iota)\supset B(S,\rho)$
by Weyl's inequality (and $\mu(H)/16\le\lambda_*/16$ is not needed: the
negative floor uses the same perturbation bound applied to the negative
part). Strict convexity of the axial profile follows since its second
derivative is the $(v,v)$ entry of $D^2g$, which exceeds
$\mu-\omega_N>\tfrac{15}{16}\mu>0$; the minimum is at $t=0$ because
$\nabla g(S)=0$. The Morse lemma on $B(S,\rho)$ gives coordinates
$(\xi,w)$, $\xi$ the axial coordinate, with
$g=h+\tfrac12A(\xi,w)$, $A$ quadratic with positive $\xi^2$ coefficient
and negative-definite $w$ part; then $\{A>0\}$ is a double cone
$\{|\xi|>\|w\|_{-A_{ww}^{1/2}}\cdot c\}$, exactly two components by the
sign of $\xi$, and $u_\pm$ sit on the two rays. $\square$

**Definition F.1.9 (connection level).** For $u,v\in X$,
$\mathfrak m_g(u,v)=\sup_{\gamma:u\to v}\min_{x\in\gamma}g(x)$, the
supremum over continuous paths.

*Lemma F.1.10.* $\mathfrak m_g$ is continuous in $(g,u,v)$ (sup norm in
$g$), and $\mathcal R(g,a;u,v)$ holds iff $\mathfrak m_g(u,v)>a$.

*Proof.* $|\mathfrak m_g-\mathfrak m_{g'}|\le\|g-g'\|_\infty$ by
comparison along the same paths; path continuity gives the rest. If
$\mathfrak m_g(u,v)>a$ there is a path with $\min g>a$; covering it as in
Lemma F.1.2 gives a chain, hence $\mathcal R(g,a;u,v)$. Conversely a
chain yields a path with $\min g>a$, so $\mathfrak m_g(u,v)>a$.
$\square$

*Lemma F.1.11 (probes at the saddle level).* For the certified probes,
the broken path $u_+\to S\to u_-$ along the two axial rays has minimum
value exactly $h$ (by Lemma F.1.8 the profile exceeds $h$ off $S$), so
$\mathfrak m_g(u_+,u_-)\ge h$ **always** — the level can never drop below
$h$; this is the fatal defect of the V2 formulation, which required
$\mathfrak m<h$ to signal distinct sectors and therefore never fired.
Moreover: $u_+,u_-$ lie in the same component of $U_{h-q}$ for all small
rational $q$ iff $\mathfrak m_g(u_+,u_-)>h$; they lie in distinct local
upper-sector germs (equivalently, their components in $U_{h-q}$ are
distinct for all small $q$) iff $\mathfrak m_g(u_+,u_-)=h$.

*Proof.* The bound $\mathfrak m\ge h$ is the displayed broken path. The
level set that separates the sectors is $\{g>h\}$ itself: by Lemma
F.1.8, $\{g>h\}\cap B(S,\rho)$ is a double cone with exactly two
components, one per germ. If $u_+,u_-$ lie in the same component of
$\{g>h\}$ (globally), that component is path-connected (open in $X$), so
a joining path has $\min g>h$ and $\mathfrak m>h$; conversely any path
with $\min g>h$ exhibits such a shared component. If they lie in
different components of $\{g>h\}$, every joining path enters
$\{g\le h\}$, so $\mathfrak m\le h$, and the broken axial path gives
$\mathfrak m=h$. (Note: for every $q>0$ the neck
$\{g>h-q\}\cap B(S,\rho)$ is connected, so components of $U_{h-q}$ with
$q>0$ never separate the probes; the sector structure lives at the exact
level $h$.) $\square$

**Definition F.1.12 (negative-safe component birth).**
$$\beta_g(u,a)=\sup\big\{g(q_j):\ q_j\in D,\ \mathcal R(g,a;u,q_j)\big\},
\qquad \sup\varnothing=-\infty.$$
This is a supremum of *values of eligible points*, with $-\infty$ for the
empty set — **never** value-times-indicator: multiplying by an indicator
would insert $0$ for ineligible components and corrupt every negative
birth height (fields with all relevant maxima negative are legitimate
configurations).

*Lemma F.1.13.* If the component $C$ of $u$ in $U_a(g)$ contains a local
maximum, then $\beta_g(u,a)$ equals the value of the highest such maximum
(the representative value of $C$); otherwise $\beta_g(u,a)\le a$.

*Proof.* $C$ is open; the set of values $\{g(q_j):q_j\in C\}$ is dense in
$g(C)$ by continuity of $g$ and density of $D$, so the supremum equals
$\sup_C g$, which on $\mathcal G$ is attained at the highest maximum of
$C$ if one exists. $\square$

**Definition F.1.14 (sector mark).** With $\beta_\pm=\beta_g(u_\pm,h)$,
the negative-safe birth heights (Definition F.1.12) of the two sector
components at the exact level $h$:
$$e_{\mathrm{sec}}(g;M,S)=1\big\{\mathfrak m_g(u_+,u_-)=h\big\}\cdot 1\big\{g(M)=\min(\beta_+,\beta_-)\big\}.$$

**Theorem F.1.2 (V3 Thm A.3).** On $\mathcal G$,
$e_{\mathrm{sec}}=e_{\mathrm{lev}}=1\{(M,S)\ \text{elder pair}\}$.

*Proof.* The first factor of $e_{\mathrm{sec}}$ is $1$ iff the two local
upper sectors at $S$ lie in distinct components of $U_{h-q}$
(Lemma F.1.11), i.e. iff $S$ is a merge saddle (case (ii) of Theorem
F.1.1). Under a merge, the two components' representative values are
$\beta_+,\beta_-$ (Lemma F.1.13), and the elder rule kills the class of
the younger (smaller) representative: $(M,S)$ is the elder pair iff
$g(M)=\min(\beta_+,\beta_-)$, which is the second factor. Under an
attachment the first factor is $0$ and, by Theorem F.1.1,
$e_{\mathrm{lev}}=0$ as well. Hence $e_{\mathrm{sec}}=e_{\mathrm{lev}}$,
and Theorem F.1.1 identifies both with the elder-pair indicator.
$\square$

Henceforth $e:=e_{\mathrm{sec}}=e_{\mathrm{lev}}$.

### F.2 Elder-failure exhaustion, V3 form (supersedes A.2) — full statement and proof

Source: embedded `05_V3_CORRECTIVE_ADDENDUM.pdf` (§3 and §7), expanded to
be self-contained. Notation from F.1 is in force; $C>2$ and $\delta_0>0$
are fixed constants, $r<\delta_0/C$.

**Definition F.2.1 (adaptive sector components).** On $\mathcal G$, let
$C_+$ and $C_-$ be the connected components of $U_h(g)=\{g>h\}$
containing the probes $u_+$ and $u_-$ (with $C_+=C_-$ in the attachment
case, by Lemma F.1.11). The *representative* $Q_\pm$ of $C_\pm$ is its
highest local maximum (Lemma F.1.13), well-defined on $\mathcal G$.

**Definition F.2.2 (adjacency and self-attachment events).**
$A=\{M\in C_+\cup C_-\}$ ("$M$ is adjacent to $S$ through an upper
sector") and $L=\{C_+=C_-\}$ ("self-attachment").

*Lemma F.2.2.* $A$ and $L$ are Borel.

*Proof.* $M\in C_\pm$ iff $\mathcal R(g,h;M,u_\pm)$, a Borel condition by
Lemma F.1.2, so $A$ is Borel. $L=\{\mathfrak m_g(u_+,u_-)>h\}$ by Lemma
F.1.11, Borel by the continuity of $\mathfrak m$. $\square$

**Lemma F.2.1 (V3 Lemma A.4, elder-failure trichotomy).** On
$A\cap\{e=0\}$, exactly one of the following holds:
(1) *earlier death*: $M$ is not the representative of its incident sector
component; then the elder death saddle $R$ of $M$'s class is unique,
satisfies $R\notin\{M,S\}$, and lies in the open value window
$h<g(R)<b$;
(2) *same-component attachment*: $M$ is the representative of its
incident component and $C_+=C_-$;
(3) *younger opposite component*: $M$ is the representative of its
incident component, $C_+\neq C_-$, and the representative maximum $N$ of
the other component satisfies $N\notin\{M,S\}$ and $h<g(N)<b$.
The three cases are disjoint and exhaustive, and cases 1 and 3 supply a
canonical third critical point ($R$ resp. $N$) in the open window
$(h,b)$.

*Proof.* Exhaustiveness is by successive alternatives on $A\cap\{e=0\}$:
either $M$ is not the representative of its incident component (case 1),
or it is and $C_+=C_-$ (case 2), or it is and $C_+\neq C_-$ (case 3).
Disjointness is immediate from the alternatives. It remains to verify the
two witness claims.

Case 1: $M$ is not the representative of its incident component $C_\pm$,
a component of $U_h(g)$. Track $M$'s class in the downward sweep from
$b=g(M)$: since $C_\pm$ contains a maximum of value $>g(M)$
(Lemma F.1.13), $M$'s class must have died at some level in $(h,b)$ —
the class would survive down to level $h$ only if $M$ stayed the
representative of its component at level $h$, contradicting
$\mathcal O(g,M,h)$. Class deaths on $\mathcal G$ occur only at saddle
values, and distinct critical values make the death saddle $R$ unique;
$R\neq S$ since $g(R)>h=g(S)$, and $R\neq M$ since $R$ is a saddle. So
$h<g(R)<b$.

Case 3: $M$ is the representative of its incident component and
$C_+\neq C_-$, so $S$ is a merge saddle and the two sector components
have representatives $M$ and some other maximum $N$ (Lemma F.1.13), with
$N\notin\{M,S\}$ (it is a maximum, $S$ a saddle). Since
$u_-\in C_-\subset U_h(g)$ with $g(u_-)>h$, the representative satisfies
$g(N)\ge g(u_-)>h$. If $g(N)>g(M)=b$, the elder rule would kill $M$'s
class at $S$, giving $e=1$ by Theorem F.1.1 — contrary to $e=0$. Hence
$h<g(N)<b$. $\square$

**Genuinely disjoint spatial partition (proved).** Use the **single**
distance function $d_P(y)=d(y,\{M,S\})$ for all three regions:
$$\text{collar } \mathcal R_{\mathrm{col}}=\{d_P\le Cr\},\qquad
\text{singular-near } \mathcal R_{\mathrm{sing}}=\{Cr<d_P<\delta_0\},\qquad
\text{far } \mathcal R_{\mathrm{far}}=\{d_P\ge\delta_0\}.$$
Because one and the same function defines all three, they are trivially
disjoint and exhaustive (the boundary sphere $\{d_P=Cr\}$ lies inside the
collar; the annulus is open). If $s(y)$ is the radial coordinate of a
midpoint chart centered within $r/2$ of $\{M,S\}$, the triangle
inequality gives $|s(y)-d_P(y)|\le r/2$, so chart translations move the
region boundaries by $O(r)$ and change only constants in every estimate
that follows — no region depends on a second distance scale.

**Canonical Borel witnesses.** The critical-point graph
$\Gamma=\{(g,y)\in\mathcal G\times X:\nabla g(y)=0\}$ has finite fibers
on $\mathcal G$ (Morse functions have finitely many critical points), and
the value-window and region conditions are Borel, so the Lusin–Novikov
uniformization theorem supplies Borel selections of the case-1 witness
$R$ (unique, hence canonical) and of the case-3 witness $N$ (the
representative, unique by distinct critical values). Define the
all-witness counts, for
$j\in\{\mathrm{col},\mathrm{sing},\mathrm{far}\}$:
$$N_j=\#\big\{y\in X\setminus\{M,S\}:\ \nabla g(y)=0,\ h<g(y)<b,\ y\in\mathcal R_j\big\}.$$

*Lemma F.2.3 (exact disjoint failure partition).* With $F_{\mathrm{self}}$
the case-2 event, $F_{\mathrm{adj}}=A^c\cap\{e=0\}$, and $F_j$ the event
that case 1 or case 3 holds with canonical witness in $\mathcal R_j$:
$$\{e=0\}=F_{\mathrm{adj}}\ \dot\cup\ F_{\mathrm{self}}\ \dot\cup\ F_{\mathrm{col}}\ \dot\cup\ F_{\mathrm{sing}}\ \dot\cup\ F_{\mathrm{far}},
\qquad 1_{F_j}\le 1_{\{N_j\ge1\}}\le N_j.$$

*Proof.* On $A\cap\{e=0\}$ the trichotomy applies; case 2 is
$F_{\mathrm{self}}$; in cases 1 and 3 the canonical witness is a critical
point in the window $(h,b)$ lying in exactly one region
$\mathcal R_j$ (disjointness of the partition), so the events $F_j$ are
disjoint, cover the remainder of $\{e=0\}$, and imply $N_j\ge1$.
$\square$

**Selection inequality (proved).** Let $\mathbb P^{MS}_{r,t,b,\kappa}$ be
the determinant-weighted Palm law of the marked pair and
$p_r=\mathbb E^{MS}[e]$ the elder-selection probability. Then
$$1-p_r=\mathbb P^{MS}(e=0)\le\mathbb P^{MS}(A^c)+\mathbb P^{MS}(F_{\mathrm{self}})+\mathbb E^{MS}N_{\mathrm{col}}+\mathbb E^{MS}N_{\mathrm{sing}}+\mathbb E^{MS}N_{\mathrm{far}}.$$

*Proof.* Split $\{e=0\}$ as $\{e=0\}\subset A^c\cup(A\cap\{e=0\})$ and
apply Lemma F.2.3: on $A\cap\{e=0\}$, either $F_{\mathrm{self}}$ occurs
or $N_{\mathrm{col}}+N_{\mathrm{sing}}+N_{\mathrm{far}}\ge1$. Take
expectations: union bound for the first two events and Markov's
inequality $1_{\{N_j\ge1\}}\le N_j$ for the three counts. $\square$

Residual premise set (V3 (A.39), sufficient for the $O(r^3)$ selection
rate, uniformly on compact positive mark sets): RP-A
$\mathbb P^{MS}(A^c)\le C_A r^3$; RP-L
$\mathbb P^{MS}(F_{\mathrm{self}})\le C_L r^3$; RP-C
$\mathbb E^{MS}N_{\mathrm{col}}\le C_{\mathrm{col}}r^3$; RP-S
$\mathbb E^{MS}N_{\mathrm{sing}}\le C_{\mathrm{sing}}r^3$; RP-F
$\mathbb E^{MS}N_{\mathrm{far}}\le C_{\mathrm{far}}r^3$ (retained from the
frozen baseline lemma, not re-proved here). RP-A/RP-L depend on the
Palm-transfer premise of F.4; RP-C/RP-S on the joint blow-up lemma of
F.3. **The uniform $1-p_r=O(r^3)$ estimate is CONDITIONAL on all five
bounds.**

### F.3 Triple contact with the third-value pin (amends B.1)

**Notation (contact plane only — read before the displays).** Throughout
F.3, $H_M,H_S,H_P$ denote the $2\times2$ Hessians
$\begin{pmatrix}p_{XX}&p_{XY}\\p_{XY}&p_{YY}\end{pmatrix}$ of the
two-dimensional contact-plane normal form $p(X,Y)$ at the three contact
points. They are **not** the full $3\times3$ Hessians of the side-24
field, and no $3\times3$ determinant identity is stated or implied here.
Transferring these contact-plane determinant estimates to the full
$3\times3$ Hessian determinants is precisely one of the tasks of the
still-open finite-$r$ joint-blow-up lemma stated at the end of this
subsection. The checker `verify_constrained_triple_contact.py` likewise
verifies contact-plane ($2\times2$) identities only.

The B.1 elimination did not impose the witness-value condition
$p(P)=-\alpha\kappa/6$ ($0\le\alpha\le1$); without it the system is a
diagnostic only and cannot replace the historical elimination. With the pin
($M=(-1/2,0)$, $S=(1/2,0)$, $P=(c/\eta,s/\eta)$, $d=q_M-q_S$, $m=q_M+q_S$):
$$p(X,Y)=\kappa\Big(\tfrac{X^3}{3}-\tfrac X4-\tfrac1{12}\Big)+\lambda\Big(X^2-\tfrac14\Big)Y+\tfrac12\Big(\tfrac m2-dX\Big)Y^2+\tfrac\gamma6Y^3,$$
$\lambda=[-\kappa(4c^2-\eta^2)+2ds^2]/(8cs)$ from $p_X(P)=0$; eliminating
$\gamma$ via $p_Y(P)=0$ and imposing the value pin gives the missing linear
relation
$$2s^2(2cm-\eta d)=\kappa\eta R_\alpha,\qquad R_\alpha=(2c+\eta)^2-8\alpha c\eta.$$
With $u=[2s^2d+\kappa(2c+\eta)^2]/(8\kappa c\eta)$, exact elimination
(machine-verified, `verify_constrained_triple_contact.py`):
$$-\det H_M=\frac{\kappa^2\eta^2}{s^2}\big(u^2-\alpha\big),\quad
-\det H_S=\frac{\kappa^2\eta^2}{s^2}\big((u-1)^2-(1-\alpha)\big),\quad
\det H_P=-\frac{\kappa^2\eta^2}{s^2}\big((u-\alpha)^2+\alpha(1-\alpha)\big).$$
The maximum type at $M$ forces $u^2<\alpha$, hence $|u|\le1$ and
$|\det H_M\det H_S\det H_P|\le C\kappa^6\eta^6 s^{-6}$ on every angular
subchart $|s|\ge s_0>0$ — the contact-face factor $O(\kappa^6\eta^6)$
unavailable from the value-unpinned system. The $c=0$ chart is regular:
$d=-\kappa\eta^2/2$, $\lambda=(\kappa\eta s/2)(\mu-1+2\alpha)$ with
$\mu=m/(\kappa\eta^2)$, determinants $\kappa^2\eta^2[1/4-\mu/2-(\mu-1+2\alpha)^2/4]$,
$\kappa^2\eta^2[\mu/2+1/4-(\mu-1+2\alpha)^2/4]$,
$-\kappa^2\eta^2[\mu^2+4\alpha(1-\alpha)]/4$, and the type window
$-2\alpha-2\sqrt\alpha<\mu<-2\alpha+2\sqrt\alpha$ — all three determinants
uniformly $O(\kappa^2\eta^2)$ on the typed support. The collinear axis
$s=0$ carries no nondegenerate triple contact (a quadratic derivative cannot
have three roots) and is handled by the corrected conditional density
$p_{\mathrm{corr}}\le C s^{-5}e^{-c_0/s^4}$, whose product with the
$s^{-10}$ angular loss is integrable. The six-term radial expansion of
$r\,t^{-5}[r(r+t^2)]^2(r^2+t^3)$ integrates to $O(r^3)$
($O(r^3),O(r^4),O(r^5\log(1/r)),O(r^4),O(r^4),O(r^3)$) — conditional on the
uniform finite-$r$ envelope estimate (V3 (6.1)).

**Exact residual premise (open).** The finite-$r$ joint-blow-up lemma:
normalized divided-difference functionals for the exact side-24 field
extending continuously over the collar compactification of $(r,\rho)=(0,0)$
and the singular-near compactification of $(t,\rho)=(0,0)$, with (i) pair-pin
independence, (ii) conditional-covariance eigenfloors or quantified
mean-mismatch penalties on every boundary face, (iii) the angularly weighted
ninth-degree Hessian-moment estimate for (6.1) and its collar analogue,
(iv) uniform constants on the full compactification. Until this lemma is
proved: corrected limiting cubic elimination **proved**; $c=0$ chart and
contact angular integrability **proved**; six-term radial power count
**proved conditional on (6.1)**; uniform finite-$r$ collar and
singular-near bounds **open**.

### F.4 Capture/escape, V3 threshold-free form (supersedes the proof of A.4.4–A.4.6)

Normal form (A.10) with endpoint pins $Q(M)=Q(S)=0$,
$\nabla Q(M)=\nabla Q(S)=0$; $R=1+|a|+\|W\|+\|C\|+\|Q\|_{C^2}\ge1$,
$K=4096$, $\varepsilon_0=1/(1024K)$, $rR^5\le\varepsilon_0$,
$\tau\ge2KR^5+2R$, $m=2R/\tau$; endpoint Jacobian block bounds
$A\ge3/4$, $|B|\le R$, $\|E_D\|\le3R/4$, exactly one positive eigenvalue.
Machine-verified margins (`verify_capture_escape_budgets.py`, exact
Fraction arithmetic at $R=1,2,5,50$ with $\tau$ at its floor):

1. **Outward branch.** Cone $\mathcal C_E=\{0\le\xi=X-\tfrac12\le\delta,\
E(Y)\le4(R^2/\tau)\xi^2\}$, then strip
$\mathcal S_E=\{\tfrac12+\delta\le X\le\tfrac54,\ E\le9R^2/(16\tau)\}$. On the
cone $|Y|\le m\xi\le\xi/(KR^4)$ and the pins give
$|\nabla Q|\le R(\xi+|Y|)\le R(1+m)\xi$, so
$F_X\ge[1-(\tfrac12+\delta)Rm-\tfrac12Rm^2\delta-rR(1+m)]\xi\ge\tfrac34\xi$;
on $G_E=0$ ($q=|TY|\ge2R\xi$) the four adverse ratios sum to
$257/1024+257/(1024K)+1/(2048K^2)+\varepsilon_0<1/3$, hence
$\tfrac12\dot G_E\le-\tfrac23q^2<0$. On the strip $F_X\ge\tfrac34\delta$ and
$\tfrac12\dot E\le-\tfrac1{10}q^2$ at $E=9R^2/(16\tau)$; terminal ledger
$P(5/4,Y)>49/192-1/768=65/256>1/4>0=P(M)$. No $\lambda_{\max}(T)$ bound used.
2. **Inward branch, no threshold.** In
$\mathcal T_{\mathrm{in}}=\{-\tfrac12<X<\tfrac12,\ |Y|\le m h(X)\}$,
$h(X)=\tfrac14-X^2$, the pins give $|\nabla Q|\le R(2h(X)+|Y|)\le3Rh(X)$, so
$F_X\le-[1-Rm/2-Rm^2/8-3rR]h\le-\tfrac34h<0$ ($X$ strictly decreasing); on
$|Y|=mh$ the leading term $-Y^\top TY\le-\tau m^2h^2=-4(R^2/\tau)h^2$ dominates
the adverse budget
$(1+1/K+1/(4K^2)+6\varepsilon_0+4R^2/(K\tau))(R^2/\tau)h^2<2(R^2/\tau)h^2$,
so $\dot G_{\mathrm{in}}/2\le-2(R^2/\tau)h^2<0$. Since $X$ decreases
strictly, a limit above $-\tfrac12$ would leave $h$ bounded away from $0$ and
contradict the longitudinal bound; hence $X(t)\to-\tfrac12$ and
$|Y(t)|\le m h(X(t))\to0$: convergence to $M$. The V2 auxiliary threshold
$\delta_0$ and terminal section are removed entirely (the V2 claim
$h(-\tfrac12+\delta_0)\ge\delta_0$ was false: the left side is
$\delta_0-\delta_0^2$; regression-pinned in the budget script).
3. **Pin-preserving exact-field robustness (explicit hypotheses).** For
$\tilde F=F+\mathcal E$ with $\mathcal E(M)=\mathcal E(S)=0$,
$\|D\mathcal E\|_\infty\le\eta_R:=1/(8192R^3)$, perturbed block bounds
$\tilde A\ge3/4-\eta_R$, $|\tilde B|\le R+\eta_R$,
$\|\tilde E_D\|\le3R/4+\eta_R$, exactly one positive eigenvalue, and
$\tilde P(M)=P(M)$: the endpoint equalities give
$|\mathcal E|\le2\eta_R\xi$ on the cone and $|\mathcal E|\le3\eta_R h$ in the
tube, with flux costs $3\eta_R/R$ (cone), $4\eta_R/(3R)$ (strip), and the
corrected tube cost $12(\eta_R/R)(R^2/\tau)h^2$ — the V3 integration had
printed the under-counted $(\eta_R/12)(R^2/\tau)h^2$; the correct, larger
bound still fits comfortably, since
$12\eta_R/R\le12/8192\ll2$, so all three costs remain strictly below the
margins above (machine-checked at $R=1,2,5,50$); since
$\eta_R=\delta_0/16$ the strip longitudinal margin persists. A bare absolute $C^1$ error without endpoint pin preservation is
insufficient (margins vanish at the endpoints).

**Residual Palm-transfer premise (open).** To infer
$\mathbb P^{MS}_{r,t,b,\kappa}(\mathcal D_r^c)=O(r^3)$ uniformly on compact
positive mark sets one still needs, in one finite-$r$ side-24 coordinate
system: (i) $cr^2\le Z_r\le Cr^2$ and $\mathbb E^0W_r^2\le Cr^4$; (ii) exact
endpoint soft-mode factorizations $\det H_M=rD_M$, $\det H_S=rD_S$ with
uniform polynomial moments; (iii) the weighted shallow-eigenvalue expansion
$W_r\le Cr^2[\lambda_*+rP_0(U_r)]^2(1+\|U_r\|)^N$ and eigenvalue/coarea
density $\mathbb E^0[W_r1_{\{0<\lambda\le rP_0(U_r)\}}]\le Cr^5$;
(iv) uniform sub-Gaussian finite-jet and fifth-derivative tails with
polynomial dependence of all approximation constants; (v) the exact scaling
identification $\tau_r=\lambda_*/(\kappa r)$.

### F.5 Density versus cumulative mass (regression audit)

The withdrawn $O(\ell)$ far-density claim assigned a window width to a
density; a window mass is $O(\Delta\ell)$ and division by $\Delta\ell$ leaves
an $O(1)$ density. `audit_density_vs_cumulative.py` pins this with a
Gaussian countermodel: the differentiated window mass is the density
difference $|d\,\mathrm{mass}/d\ell|=|p(b-\ell)-p(b-\ell-D)|$, which is
trivially $\le2/\sqrt{2\pi}=O(1)$ and, by the mean value theorem, also
$\le D\,e^{-1/2}/\sqrt{2\pi}=O(D)$ — neither bound is the error; the
error of the withdrawn argument was assigning the window *width* $D$ to
the density at the window edge, which over-claims by exactly $1/D$. The
retained statement is the uniform $O(1)$ bound
$\sup_{0<\ell\le\ell_0}\nu^{\mathrm{far}}_\rho(\ell)<\infty$, sufficient
because bounded functions are $o(\ell^{-1/3})$.

### F.6 V3 status map (controlling)

| Layer | V3 disposition | Still required |
|---|---|---|
| Borel elder-pair mark (F.1) | PROVED | independent proofread only |
| Deterministic elder-failure exhaustion (F.2) | PROVED | independent proofread only |
| Near-density pushforward (A.3.1, A.3.3) | retained | independent Kac–Rice review |
| Separated-pair contribution (A.3.2) | retained as uniform $O(1)$ | independent Kac–Rice review |
| Contact-face cubic elimination (F.3) | repaired, exactly checked | uniform finite-$r$ transfer (joint blow-up lemma) |
| Deterministic capture/escape (F.4) | repaired under explicit hypotheses | side-24 normal-form verification |
| Palm transfer (RP-A, RP-L) | **CLOSED (Part G.7, 2026-08-02)** | independent proofread only |
| Collar/singular-near witness expectations (RP-C, RP-S) | OPEN | uniform two-sided control of the normalized witness covariance on the compactified strata (G.9) |
| Fixed-distance witness (RP-F) | **CLOSED (audit, Part G.8, 2026-08-02)** | independent proofread only |
| Uniform elder selection $1-p_r=O(r^3)$ | CONDITIONAL on RP-C, RP-S only | G.9 residual step |
| Candidate theorem | **HOLD — not yet proved** | RP-C, RP-S plus independent review |

---

## Part G — Palm transfer and the regional witness bounds (2026-08-02)

This part attacks the open residual premises of Part F. Notation is that of
F.2–F.4 and of the frozen baseline Module I (04_TECHNICAL_APPENDIX.pdf
pp. 42–46). Three new scripts support this part:
`verify_palm_soft_factors.py` (exact kernel and soft-mode algebra),
`verify_corrected_pin_floor.py` (NUMERIC corroboration of the corrected-pin
eigenfloor), `verify_collar_factorization.py` (NUMERIC corroboration of the
collar determinant law). Numeric checks are corroboration only; every
load-bearing step below is proved analytically.

### G.1 Spectral nondegeneracy and covariance eigenfloors (proved)

**Lemma G.1.1 (independence).** The periodized side-24 Bargmann–Fock field
has spectral masses $w_k=e^{-|k|^2/2}>0$ at every lattice point
$k\in(\pi/12)\mathbb Z^3$ — full Fourier support. Consequently every finite
family of pairwise distinct derivative-evaluation functionals
$\varphi_j(f)=\partial^{\alpha_j}f(x_j)$ (distinct pairs
$(x_j,\alpha_j)$) is linearly independent in $L^2(\text{spectral})$: a
dependence relation is an exponential polynomial
$\sum_j c_j(ik)^{\alpha_j}e^{ik\cdot x_j}$ vanishing on the whole lattice,
hence (as an entire function) identically zero, hence all $c_j=0$ by
independence of the exponentials at distinct base points and of the
monomials $(ik)^\alpha$ at a common base point. $\blacksquare$

**Lemma G.1.2 (floors on compacta).** For any such family varying
continuously over a compact parameter set with the members staying
distinct, the Gram (covariance) determinant is a continuous positive
function on a compact set, hence has positive minimum and finite maximum:
uniform eigenfloors and ceilings. $\blacksquare$

**Lemma G.1.3 (corrected-pin floor, the blow-up).** For the raw pin vector
$\Phi_r=(f(M),\nabla f(M),f(S),\nabla f(S))$, $M=x-\frac r2t$,
$S=x+\frac r2t$, define the corrected vector
$$V_r=\Big(f(M),\ \nabla f(M),\ \tfrac{\nabla f(S)-\nabla f(M)}r,\ \tfrac{f(S)-f(M)-\frac r2(t\cdot\nabla f(S)+t\cdot\nabla f(M))}{r^3}\Big).$$
The map $\Phi_r\mapsto V_r$ is triangular with diagonal
$(1,1,1,1,r^{-1},r^{-1},r^{-1},r^{-3})$ and off-diagonal blocks bounded
uniformly in $r$, so conditioning on $\Phi_r$ equals conditioning on $V_r$,
and the weighted raw covariance $\Sigma_r=D_r\,\mathrm{Cov}(\Phi_r)\,D_r$
(with the matching diagonal weights) is equivalent to $\mathrm{Cov}(V_r)$.
As $r\downarrow0$, $V_r$ converges pointwise in $L^2$ to the local jet
$$\Big(f(x),\ \nabla f(x),\ Hf(x)\,t,\ -\tfrac1{12}\partial_t^3f(x)\Big),$$
eight distinct derivative functionals (for $t\neq0$), whose Gram matrix is
strictly positive definite by Lemma G.1.1. Hence:

(i) for $r\in(0,r_0]$: $\mathrm{Cov}(V_r)\to\mathrm{Cov}(V_0)\succ0$, so
the eigenfloor extends to the closed interval $[0,r_0]$ by continuity;

(ii) for $r\in[r_0,r_{\max}]$: $\mathrm{Cov}(V_r)\succ0$ pointwise
(distinct base points $M\neq S$; Lemma G.1.1), and continuity plus
compactness in $(x,t,r)$ over the compact parameter set
($x\in\mathbb T^3_{24}$, $t\in S^2$, $r\in[r_0,r_{\max}]$) gives the
uniform floor. $\blacksquare$

Numerically corroborated (`verify_corrected_pin_floor.py`): the smallest
eigenvalue of $\mathrm{Cov}(V_r)$ computed from the truncated lattice sum
converges to $\approx34.4$ as $r\downarrow0$ for three probe directions and
stays positive on the sampled grid.

### G.2 Exact soft-mode identities (proved; machine-verified)

**Lemma G.2.1 (endpoint kernels).** Let $h\in C^1([-r/2,r/2])$ with
$\int_{-r/2}^{r/2}h=0$. Then
$$h(-\tfrac r2)=-\frac1r\int_{-r/2}^{r/2}\Big(\frac r2-s\Big)h'(s)\,ds,\qquad
h(\tfrac r2)=\frac1r\int_{-r/2}^{r/2}\Big(s+\frac r2\Big)h'(s)\,ds.$$
*Proof.* Integration by parts:
$\int(\frac r2-s)h'\,ds=[(\frac r2-s)h]_{-r/2}^{r/2}+\int h\,ds=-r\,h(-\frac r2)+0$;
the mirror computation gives the second identity. Both kernels lie in
$[0,1]$. $\blacksquare$

Applied with $h(s)=\partial_i f(x+st)$ — for which the gradient pins give
$\int\partial_{ti}f(x+st)\,ds=\partial_i f(S)-\partial_i f(M)=0$ — this
yields, with $H_p$ the endpoint Hessians in the frame $(t,v,w)$:
$$H_M=\begin{pmatrix}r\alpha_M & r\beta_M^{\mathsf T}\\ r\beta_M & Q_M\end{pmatrix},\qquad
H_S=\begin{pmatrix}r\alpha_S & r\beta_S^{\mathsf T}\\ r\beta_S & Q_S\end{pmatrix},$$
where $(\alpha,\beta)$ are kernel averages of third derivatives of $f$ on
the segment, hence centered Gaussian variables whose joint law (given the
pins) has all polynomial moments bounded uniformly on compact mark sets
(the third-derivative spectral moments are finite and the corrected-pin
covariance has a uniform ceiling, Lemma G.1.3, so Gaussian regression has
bounded regression coefficients and bounded conditional covariance).

**Lemma G.2.2 (soft factorization).**
$\det H_M=rD_M$, $\det H_S=rD_S$ with
$D_M=\alpha_M\det Q_M-r\beta_M^{\mathsf T}\mathrm{adj}(Q_M)\beta_M$ —
a polynomial of degree $\le3$ in the scaled variables. Machine-verified
(`verify_palm_soft_factors.py`, (S1)). $\blacksquare$

**Lemma G.2.3 (value window).** With $g(s)=f(x+st)$ and the gradient pins,
$$f(S)-f(M)=\int_{-r/2}^{r/2}\Big(\frac{w^2}2-\frac{r^2}8\Big)g'''(w)\,dw,\qquad
\frac{w^2}2-\frac{r^2}8\in\Big[-\frac{r^2}8,0\Big],\quad
\int\Big(\frac{w^2}2-\frac{r^2}8\Big)dw=-\frac{r^3}{12}.$$
Machine-verified (K2). In particular the pin value window
$\kappa r^3/6$ is exactly of order $r^3$. $\blacksquare$

### G.3 The normalizer (F.4 item (i); proved)

**Theorem G.3.1.** Uniformly on compact positive $(b,\kappa)$ sets:
$c_Zr^2\le Z_r\le C_Zr^2$ and $\mathbb E^0[W_r^2]\le Cr^4$, where
$Z_r=\mathbb E^0[W_r]$,
$W_r=|\det H_M|\,|\det H_S|\,\mathbf 1_{\{H_M\prec0,\ \mathrm{ind}(H_S)=2\}}$.

*Proof.* *Upper bounds.* By Lemma G.2.2,
$W_r\le r^2|D_M||D_S|$, and $D_M,D_S$ are degree-3 polynomials in scaled
variables with uniformly bounded Gaussian moments (Lemma G.2.1), so
$\mathbb E^0[W_r]\le r^2(\mathbb E|D_M|^4\mathbb E|D_S|^4)^{1/2}\cdot\ldots
\le C_Zr^2$ and $\mathbb E^0[W_r^2]\le r^4(\mathbb E|D_MD_S|^4)^{1/2}\le Cr^4$
by Cauchy–Schwarz.

*Lower bound.* $Z_r/r^2=\mathbb E^0[|D_M||D_S|\mathbf 1_{\{\mathrm{type}\}}]$.
As $r\downarrow0$ the joint conditional law of the scaled variables
converges to a nondegenerate centered Gaussian law (the limiting
functionals are distinct local derivative functionals, Lemma G.1.1;
nondegenerate Gram matrix), continuously in the marks on compact sets.
Now $H_M\prec0$ iff $Q_M\prec0$ and $\alpha_M<r\beta_M^{\mathsf T}Q_M^{-1}\beta_M$,
an event converging to $\{Q_M\prec0,\ \alpha_M<0\}$, whose boundary
($\det Q_M=0$ or $\alpha_M=0$) has measure zero under the limit law;
similarly $\mathrm{ind}(H_S)=2$ converges to an open event of the same
form. On the limit event, $D_M\to\alpha_M\det Q_M<0$ a.s., so
$$Z_r/r^2\;\longrightarrow\;\mathbb E\big[|\alpha_M\det Q_M|\,|\alpha_S\det Q_S|\,\mathbf 1_{\{Q_M\prec0,\alpha_M<0\}\cap\{\mathrm{ind}_S=2\}}\big]\;=:\;c_0>0,$$
the expectation being of a positive a.s. integrand over a
positive-probability open event, with convergence by weak convergence plus
uniform integrability (Gaussian tails dominate the polynomial integrands).
The limit $c_0$ is continuous in the marks and positive, so uniformly on
compact mark sets $Z_r/r^2\ge c_0/2=:c_Z$ for $r\le r_0$; for
$r\in[r_0,r_{\max}]$, $Z_r/r^2$ is a positive continuous function (the
type event has positive probability under the nondegenerate pin law at
each fixed $r$, and $W_r>0$ on it), so its minimum on the compact set is
positive. $\blacksquare$

### G.4 Weighted shallow-eigenvalue bounds (F.4 items (iii)–(iv); proved)

**Lemma G.4.1 (weighted expansion).** On the typed support, with
$\lambda_*=\lambda_{\min}(-Q_S)$ and $U$ the scaled jet,
$$W_r\le Cr^2\,[\lambda_*+rP_0(U)]^2(1+\|U\|)^N,\qquad P_0(U)=1+|\beta|^2,$$
for fixed $C,N$. *Proof.* For $Q\prec0$ symmetric $2\times2$ with
eigenvalues $-\lambda_{\max}\le-\lambda_*<0$:
$|\det Q|=\lambda_*\lambda_{\max}\le\lambda_*\|Q\|$ and
$\|\mathrm{adj}\,Q\|=\lambda_{\max}=\|Q\|$ (machine-verified (S2)). Hence
$$|D_M|\le|\alpha_M|\lambda_*\|Q_M\|+r|\beta_M|^2\|Q_M\|
\le\|Q_M\|(1+|\alpha_M|)\,[\lambda_*+r|\beta_M|^2],$$
and multiplying the two endpoint bounds and absorbing polynomial factors
into $(1+\|U\|)^N$ gives the claim. $\blacksquare$

**Lemma G.4.2 (shallow-layer mass).**
$\mathbb E^0\big[W_r\mathbf 1_{\{0<\lambda_*\le rP_0(U)\}}\big]\le Cr^5$.
*Proof.* On the event, $\lambda_*+rP_0\le2rP_0$, so by Lemma G.4.1
$$\mathbb E^0[W_r\mathbf 1_{\{\lambda_*\le rP_0\}}]
\le4Cr^4\,\mathbb E\big[P_0^2(1+\|U\|)^N\mathbf 1_{\{\lambda_*\le rP_0\}}\big].$$
The variable $\lambda_*=\lambda_{\min}(-Q_S)$ is the smallest eigenvalue of
a $2\times2$ symmetric matrix whose entries have a bounded joint density
(they are Gaussian with a covariance floored by Lemma G.1.3 applied to the
local jet; densities of conditioned Gaussians with floored covariance are
bounded). The boundary of the PSD cone in $\mathrm{Sym}_2$ is the
rectifiable quadratic hypersurface $\{\det=0\}$, and $\lambda_{\min}$ is
1-Lipschitz (Weyl), so
$\{0<\lambda_{\min}\le\varepsilon\}\cap\{\|A\|\le R\}$ is contained in an
$\varepsilon$-tube of that hypersurface and has volume $\le C_R\varepsilon$;
hence $\lambda_*$ has a bounded density near $0$, and by Cauchy–Schwarz
against the polynomial moments,
$\mathbb E[P_0^2(1+\|U\|)^N\mathbf 1_{\{\lambda_*\le rP_0\}}]
\le C'r$ (integrating $\lambda_*$ first: the inner mass is
$\le C''rP_0$). Total: $Cr^4\cdot r=Cr^5$. $\blacksquare$

Dividing by $Z_r\ge c_Zr^2$ (Theorem G.3.1):
$\mathbb P^{MS}_r(\text{shallow layer})\le Cr^3$.

### G.5 Jet and derivative tails (F.4 item (iv); proved)

The fifth-derivative field of the fixed side-24 field is a continuous
centered Gaussian field on the compact torus with a bounded affine mean
shift and covariance dominated by the unconditional one after pin
conditioning (Gaussian regression; floors/ceilings from Lemma G.1.3), so
Borell–TIS gives $\mathbb P(\mathcal H_5>u)\le Ce^{-cu^2}$; taking
$u=r^{-1}$ gives failure probability $\le Ce^{-c/r^2}=o(r^3)$. The
capture/escape robustness threshold $\eta_R=1/(8192R^3)$ is a fixed
polynomial in the envelope $R=1+|a|+\|W\|+\|C\|+\|Q\|_{C^2}$, so its
failure is dominated by the same high-jet tails; every approximation
constant is a fixed polynomial of $R$, giving the required polynomial
dependence. Regular-domain failures are handled by Cauchy–Schwarz against
$\mathbb E^0[W_r^2]\le Cr^4$ (Theorem G.3.1):
$\mathbb P^{MS}(F_r)\le C\,\mathbb P^0(F_r)^{1/2}=o(r^3)$. $\blacksquare$

### G.6 Transverse scaling (F.4 item (v); proved)

In the scaled normal form the physical transverse displacement is $rY$ and
the fold potential is divided by $\kappa r^3$; a physical quadratic term
$\frac{r^2}2Y^{\mathsf T}Q_\perp Y$ therefore becomes
$\frac1{2\kappa r}Y^{\mathsf T}Q_\perp Y$, i.e.
$T_r=-Q_\perp/(\kappa r)$ and
$\tau_r=\lambda_{\min}(T_r)=\lambda_*(-Q_\perp)/(\kappa r)$ exactly.
$\blacksquare$

### G.7 Palm transfer — RP-A and RP-L CLOSED

**Theorem G.7.1.** Uniformly on compact positive mark sets,
$$\mathbb P^{MS}_r(A^c)\le C_Ar^3,\qquad \mathbb P^{MS}_r(L_{\mathrm{same}})\le C_Lr^3.$$

*Proof.* By the deterministic capture/escape theorem (Part F.4, proved
under its explicit hypotheses) applied with the exact transverse
identification $\tau_r=\lambda_*/(\kappa r)$ (G.6): on the typed support,
adjacency failure $A^c$ requires failure of one of the hypotheses
$rR^5\le\varepsilon_0$ or $\tau_r\ge2KR^5+2R$. The second failure is
exactly the shallow-eigenvalue event
$0<\lambda_*\le\kappa r[2KR^5+2R]=rP(U)$ on compact positive $\kappa$
sets; its Palm probability is $\le Cr^3$ by Lemma G.4.2 divided by
$Z_r\ge c_Zr^2$. The first failure and every regular-domain/fifth-derivative
failure is $o(r^3)$ by G.5 with Cauchy–Schwarz against
$\mathbb E^0[W_r^2]\le Cr^4$. The self-attachment bound uses the same two
branches: on $L_{\mathrm{same}}$ with $e=0$, both upper-sector branches
must reach $M$'s component without the outward escape, which again
requires one of the same hypothesis failures (the outward branch reaching
the forward section above $b$ is what separates the components). Hence
$\mathbb P^{MS}(L_{\mathrm{same}})\le C_Lr^3$ by the identical power
count. $\blacksquare$

### G.8 RP-F audit — CLOSED

**Audit verdict.** The retained fixed-distance lemma (frozen baseline
Module G7/Module I §3, equations (25)–(32) there) is **verified**: its
G7-specific inputs are proved as follows, and its two formerly imported
inputs are now Theorem G.3.1 and Lemma G.2.2.

1. *Density-block floor (their (26)).* The joint vector
$\mathcal A_r=(P_r^{\mathrm{corr}},f(y),\nabla f(y))$ with
$d(y,\{M,S\})\ge\delta$ consists of distinct derivative functionals at
base points separated by $\ge\delta$; Lemma G.1.1 gives strict positivity
pointwise, and compactness of the separated configuration region gives
$c_\delta I\le\Sigma_{\mathcal A}\le C_\delta I$. ✔ (This closes the step
the V3 review flagged as imported.)
2. *Gaussian penalty (their (29)–(30)).* With
$\|L_r(b,\kappa)\|^2\ge\tfrac12(b^2+\kappa^2)$ (their normalization,
checked) and $\lambda_{\max}(\Sigma_{\mathcal A})\le C_\delta$, the joint
density satisfies $p_{\mathcal A}(a_r)\le Ce^{-c(b^2+\kappa^2)}$;
integrating the value window $u\in[b-\kappa r^3/6,b]$ contributes the
exact factor $\kappa r^3/6$ (Lemma G.2.3). ✔
3. *Degree-nine moment with the exact $r^2$ (their (6), (31)).* By Lemma
G.2.2, $|\det H_M\det H_S\det H_y|=r^2|D_MD_S\det H_y|$; the conditional
mean is linear in $a_r$ with bounded operator norm and the conditional
covariance is bounded PSD (Gaussian regression with the floor/ceiling of
step 1), so the conditional expectation of the degree-nine product is
$\le Cr^2[1+(|b|+\kappa)^9]$. ✔
4. *Assembly (their (9)–(11)).*
$\mathbb E^0[W_rN_{\mathrm{far}}]\le Cr^2(\kappa r^3/6)[1+(|b|+\kappa)^9]e^{-c(b^2+\kappa^2)}\le Cr^5[1+(|b|+\kappa)^{10}]e^{-c(b^2+\kappa^2)}$,
and dividing by $Z_r\ge c_Zr^2$ (Theorem G.3.1 — the step previously
marked open) gives
$$\mathbb E^{MS}_rN_{\mathrm{far}}\le Cr^3\,[1+(|b|+\kappa)^{10}]\,e^{-c(b^2+\kappa^2)}.$$
After the lifetime pushforward, multiplication by $\kappa^{-2/3}$ remains
integrable at $0$ and the Gaussian factor controls infinity. ✔

No gap was found in the retained argument; the lemma stands with all
inputs now proved in this supplement. RP-F is CLOSED.

### G.9 RP-C/RP-S — status after this round

*What is now proved or verified.*

(a) *Reduction.* With Lemma G.1.3 (pin floor), Theorem G.3.1 ($Z_r$), and
Lemma G.2.2 (soft factors), the collar and singular-near expectations
reduce to two-sided control of the witness-block conditional covariance
determinant on the respective compactifications: the Kac–Rice density of
$\nabla f(y)=0$ is $\le(2\pi)^{-3/2}(\det C_{\mathrm{cond}})^{-1/2}$
times bounded conditional determinant moments (the anisotropic envelope
$[r(r+t^2)]^2(r^2+t^3)$, whose six radial terms integrate to $O(r^3)$ —
already machine-verified in `verify_constrained_triple_contact.py`).

(b) *Collar law (numeric corroboration).* For the conditional covariance
of $\nabla f(y)$ given the corrected pins, $y=M+\rho u$:
$\det C_{\mathrm{cond}}(\rho,u)=\rho^6A(\rho,u)$ with
$A(\rho,u)\to A(u)$, $0<A(u)<\infty$, at every probe direction including
the pair axis; successive-refinement slopes tend to $6.00$
(`verify_collar_factorization.py`). The positivity mechanism is analytic:
$\nabla f(M+\rho u)/\rho\to H_M u$ in $L^2$, and the limiting functionals
together with the corrected-pin limit jet are distinct derivative
functionals (Lemma G.1.1), so the face Gram determinant is positive
pointwise on the compactified direction sphere.

*The residual step (OPEN, now sharply localized).* The pointwise
positivity $A(u)>0$ must be upgraded to **uniform two-sided control of
$A(\rho,u)$ on the whole compactification** — i.e. continuity of the
normalized determinant at every boundary stratum, including the angular
strata where the limiting functionals collide (the pair axis and its
analogues in the singular-near chart, where the baseline records the
extra factors $(c^2+\rho^2)$ and $(3c^2+\rho^2)$). The numerics show the
coefficient varying over about four decades across directions while
remaining positive and finite, consistent with a uniform bound; proving
that bound requires the exact-planar covariance computation of
LS-DER-053/065 to be re-derived (not merely cited) for the side-24
periodized field, stratum by stratum. Until then
$\mathbb E^{MS}N_{\mathrm{col}}\le C_{\mathrm{col}}r^3$ and
$\mathbb E^{MS}N_{\mathrm{sing}}\le C_{\mathrm{sing}}r^3$ remain OPEN,
and with them the uniform elder-selection estimate and the candidate
theorem (**HOLD**).

### G.10 Updated premise accounting

| Premise | Status |
|---|---|
| RP-A (adjacency failure $O(r^3)$) | **CLOSED** (Theorem G.7.1) |
| RP-L (self-attachment $O(r^3)$) | **CLOSED** (Theorem G.7.1) |
| RP-F (fixed-distance witness $O(r^3)$) | **CLOSED** (audit, G.8) |
| RP-C (collar witness $O(r^3)$) | OPEN — residual step of G.9 |
| RP-S (singular-near witness $O(r^3)$) | OPEN — residual step of G.9 |
| Uniform $1-p_r=O(r^3)$ | CONDITIONAL on RP-C, RP-S only |
| Candidate theorem | **HOLD** — two premises remain |

---

## Register reconciliation summary

| # | Item | State after this supplement |
|---|------|-----------------------------|
| 1 | Off-diagonal elder pairs | Proved: Proposition A.3.2 ($O(1)$ density bound, hence $o(\ell^{-1/3})$; the stronger $O(\ell)$ form was withdrawn as unsupported) |
| 2 | Exact near pushforward | Proved: Proposition A.3.1 + Theorem A.3.3 with explicit DCT hypotheses |
| 3 | Palm law and elder mark | PROVED (V3, Part F.1): level-transition mark $e_{\mathrm{lev}}$ = sector mark $e_{\mathrm{sec}}$ = elder pair; Borel via countable component relation; supersedes A.1 |
| 4 | Elder-selection trichotomy | PROVED deterministically (V3, Part F.2): disjoint three-case exhaustion with canonical witnesses; disjoint spatial partition (single distance, $C>2$); selection inequality proved. The uniform $O(r^3)$ selection rate is CONDITIONAL on RP-C and RP-S only (RP-A/RP-L closed in Part G.7, RP-F audited closed in Part G.8) |
| 5 | Capture/escape consolidation | PROVED deterministically (V3, Part F.4): threshold-free inward tube, pin-preserving robustness with $\eta_R=\delta_0/16$ and the corrected tube cost $12(\eta_R/R)(R^2/\tau)h^2$ (V4 repair); margins machine-verified. Palm transfer (RP-A/RP-L) CLOSED in Part G.7 |
| 6 | Triple-contact elimination | CLOSED as citation replacement + V3 contact-face repair (Part F.3): the missing third-value pin restored; exact one-variable identities with type cone $u^2<\alpha$ and $O(\kappa^6\eta^6)$ factor; $c=0$ chart regular; radial count proved conditional on (6.1). Uniform finite-$r$ transfer (joint blow-up lemma) OPEN |
| 7 | Cone and first-variation constants | CLOSED, all constants exact/symbolic: cone law and jet objects derived; $D_2=29/6-\sqrt6$ proved in closed form (B.2′); $\delta a,\delta m_4,\delta\chi$, $\delta\log c$ and $P_3(L)$ exact (scale check $3/2$); numerics retained only as cross-checks |
| 8 | Fourier prefactor | Corrected display (C.1); positivity conclusions unaffected |
| 9 | Module E hypotheses | Restored (C.2) |
| 10 | Module K notation | Symbols defined (C.3), consistency with $\sigma_t^2=1/24$ checked |
| 11 | Density convention | Insert text supplied (C.4) |
| 12 | Literature novelty | Systematic web-level search executed; no conflicting theorem located; paywalled-database pass remains |

Status after the V3 integration and the V4 repair (Part F, 2026-08-01): the V4
repair expanded F.1–F.2 into self-contained proofs with all notation defined,
restored contact-plane ($2\times2$) Hessian notation in F.3, corrected the
tube cost in F.4, repaired the density-audit wording in F.5, marked all
superseded claims as HISTORICAL, and replaced the weakened scripts
(explicit raising checks, re-executed by verifier v7 in normal and
`python -O` modes). Substantively: the definition,
correctness, measurability, and deterministic-exhaustion defects are closed
(items 3, 4), the deterministic capture/escape is closed under explicit
hypotheses (item 5), and the contact-face elimination is repaired and
exactly checked (item 6). The two former RESIDUAL PREMISES (items 6 and 7)
are resolved at the computational level: item 6 by replacing the LS-DER-024
citation with the displayed value-pinned elimination (F.3), item 7 by the
full first-principles first-variation derivation. Part G (2026-08-02) then
closed three of the five probabilistic premises: the Palm transfer
(RP-A/RP-L, Theorem G.7.1 — soft-mode identities, normalizer bounds
$c_Zr^2\le Z_r\le C_Zr^2$, weighted shallow-eigenvalue expansion, the
$Cr^5$ shallow-layer mass, Borell tails, and the exact transverse scaling
$\tau_r=\lambda_*/(\kappa r)$ are all proved) and the fixed-distance
witness (RP-F, audited closed in G.8 with every input now proved in this
supplement). What remains open is narrower and named: the collar and
singular-near witness expectations (RP-C/RP-S), whose sole residual step
is uniform two-sided control of the normalized witness-covariance
determinant on the compactified boundary strata (Part G.9). Accordingly
the candidate theorem remains on HOLD: the uniform elder-selection
estimate $1-p_r=O(r^3)$ is now conditional on exactly two premises
(RP-C, RP-S), and no submission package should be assembled until they
are proved. One literature task remains
open in the ordinary editorial sense (item 12: a paywalled MathSciNet/zbMATH
pass), flagged here rather than hidden; no conflicting theorem was located at
web level.
