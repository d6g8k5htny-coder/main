# Fresh independent adversarial audit — second pass

Owner-facing attribution: **Dylan Roy — delegated AI review**. Actual performer: **Codex (OpenAI), agent /root**; fresh mathematical source review and independent calculation, with the coverage and exposure limits below. No human review or organizational-independence credit is claimed. Dylan’s personal reading: **PENDING** (retrospective only).

Scientific effect: **NONE**. This is a new append-only report. No proof bodies, source files, scientific status fields, earlier reports, or acceptance records were changed.

## Executive conclusion

**The central leading law and bounded remainder are apparently supported by the inspected mathematical chain.** I reconstructed the pin normalization, weighted selection estimate, all-mark integration, and global cap argument. I did not find a counterexample to

$$
\nu_{\rm eld}(\ell)\sim c_{d,L}\ell^{-1/3}
\quad\text{or}\quad
|\nu_{\rm eld}(\ell)-c_{d,L}\ell^{-1/3}|\le C
$$

for fixed $d\ge2,L>0$.

**I did find a false auxiliary uniformity claim:** the normalized-form remark in [K], lines 134–136, cannot hold over all positive $k$ with only $b$ restricted to a compact set. In the exact periodized field, $d=2,u=e_1,b=0,k=r^{-2}$ gives

$$
Q^W_{r,0,r^{-2}}(G_r^c)\longrightarrow1,
\qquad (r/k)^3=r^9\longrightarrow0.
$$

This is a counterexample to that remark's literal unrestricted quantifiers, including along lifetimes $\ell=kr^3=r\downarrow0$. It is **not** a counterexample to Theorem K's polynomial-target-growth estimate, to elder pairing itself, or to the lifetime theorem. The downstream remainder arguments inspected here use the unnormalized estimate, not this remark.

The quartic crossover and third-order assembly survived the attacks described below. The cluster-law deductions also survived as deductions from their stated regional and factorial-moment interfaces. **This is not an entire-program completion claim.** I did not independently rederive every imported D5/remote/Palm regional estimate, every higher-degeneracy sector, every later coefficient-remainder packet, or the formal verification lane.

## Source pin, independence, and coverage

Canonical source snapshot: Math- commit **b55b8f34d09084300088105b90fe60759fd7e144**. All source links below are immutable at that commit. The repository tree was fetched without truncation; repository instructions were read. The older stop instruction permits resumption on a new explicit owner instruction; the present audit/publication instruction supplies that authorization.

No prior audit issue, including main #252, PR review, discussion comment, or other agent's report was fetched. No agent was contacted or delegated work. Some prior conclusions were already present in conversation context; some canonical proof files also contain embedded review history and verdicts. Those passages were encountered but **excluded from evidence**. Accordingly this is an independently reconstructed fresh analysis with disclosed exposure, not a claim of pristine informational blindness. Source status labels have no evidentiary role.

The table at the end identifies actual mathematical coverage. Reading a downstream proof is not represented as revalidating every ancestor. The Borel Kac–Rice replacement was read from the canonical source copy [B], restricted to its mathematical replacement section. Its surrounding review narrative was not used.

Levels: **A** mathematical error; **B** missing proof in an inspected supplied argument; **C** excessive scope; **D** robust under the stated scope. “Not independently audited” is an exposure limitation, not automatically Level B.

## Attack ledger

Stable IDs in this report are IBA2-001 through IBA2-013.

| ID | Claim attacked | Result / classification | Severity and required action |
|---|---|---|---|
| IBA2-001 | [K] normalized bound uniform over unrestricted $k$ | **Fails, A**; quantifier/scope defect | Correct auxiliary remark; retain target growth or restrict $k$. Core theorem unaffected by this finding. |
| IBA2-002 | Exact finite-period covariance may be replaced by isotropy | Replacement fails; actual directional formulation **D** | Keep $u$-dependent covariance. Numerical isotropic agreement at $L=24$ is not exact equality. |
| IBA2-003 | Pin, height, polar, lifetime and coefficient factors | **D** in inspected formulas | No repair found necessary. |
| IBA2-004 | Local cap forces global elder pairing | **D** under explicit cap hypotheses | All exit faces and an older endpoint are essential; local adjacency alone is insufficient. |
| IBA2-005 | Weighted soft-eigenvalue suppression, including $m=1$ | **D** on compact marks; unnormalized growth bounds survive | Do not replace weighted numerator by raw event probability. |
| IBA2-006 | Small-$k$ integration gives an absolute $O(1)$ remainder | **D** for [R] bounds and exact lower cutoff | Retain $k\ge\ell/r_0^3$; no uniform normalized probability is needed. |
| IBA2-007 | Quartic scales cannot affect the analysis | Such a blanket assertion would fail; [CU]'s crossover is **D** | Quartic scale contributes the $\ell^{1/4}$ correction; not a competing leading singularity. |
| IBA2-008 | Third-order assembly and far contribution | Assembly **D** with stated kernel inputs; far proof independently checked | No arithmetic failure found; no stronger remainder inferred. |
| IBA2-009 | Fixed-$\kappa$ parity implies a global absence of an $\ell^{1/2}$ term | Only compact-cusp conclusion tested, **D** in that scope | A global conclusion requires additional uniform matching/tail work; [N] is not treated as supplying it. |
| IBA2-010 | Sharp factorial moments and the support-$\{1,2\}$ cluster law | Conditional deductions survive; complete upstream verification remains open in this audit | Reconstruct named regional covariance/collision interfaces before declaring full closure. |
| IBA2-011 | Two-scale localization and sub-pin collisions | **D**, conditional on stated configuration convergence and moments | No algebraic collision rate or $k\downarrow0$ uniformity follows. |
| IBA2-012 | General higher-degeneracy and multi-soft sectors are negligible | Compact-sector estimates survive; exhaustive sector classification not completed | Do not extend the power counts below beyond their hypotheses. |
| IBA2-013 | Global Borel mark is admissible in Kac–Rice | **D** for finite-measure/cylinder extension | Use the actual Borel extension, not continuity of the elder indicator. |

## IBA2-001 — concrete counterexample to the all-$k$ normalized remark

**Exact challenged claim.** [K], §3, “Remark (normalized form),” lines 134–136 asserts, on compact $b$-sets,

$$
1-p_r\le Q^W(G_r^c)\le C_B(r/k)^3,\qquad r\le ck.
$$

No upper bound on $k$ is stated; Theorem K's surrounding domain is $k>0$. The source explicitly says Corollary T does not use this remark.

**Construction within the specified Gaussian model.** Take $d=2$, any fixed $L>0$, axis direction $u=e_1$, $b=0$, and

$$
k=r^{-2},\quad M=(-r/2,0),\quad S=(r/2,0),
\quad f(M)=0,\quad f(S)=-r.
$$

These satisfy $r\le ck$ for every fixed $c>0$ once $r$ is small.

The covariance factorizes exactly:

$$
K_L(x,y)=q_L(x)q_L(y),\qquad
a=-q_L''(0)>0,\quad
\sigma^2=q_L^{(4)}(0)-a^2>0.
$$

Positivity of $\sigma^2$ is the nonzero variance of the squared one-dimensional spectral frequency; the positive lattice spectrum has more than one squared frequency.

Define along the axis

$$
Z(x)=f_{yy}(x,0)+a f(x,0).
$$

For every axis point $t$,

$$
\operatorname{Cov}(Z(x),f(t,0))
=q_L(x-t)\{q_L''(0)+a\}=0.
$$

Differentiation in $t$ also gives zero covariance with $f_x(t,0)$. Reflection symmetry gives zero covariance with $f_y(t,0)$. Thus the entire Gaussian process $Z$ is independent of all six pin observations. Under the pinned law its distribution is unchanged and

$$
A_M=f_{yy}(M)=Z(-r/2),\qquad
A_S=f_{yy}(S)=Z(r/2)+ar.
$$

Consequently $A_M,A_S\to Z(0)$ in every finite $L^p$, where $Z(0)\sim N(0,\sigma^2)$. In particular the transverse curvature remains tight even though $k\to\infty$.

The axial regression gives

$$
r f_{xx}(M)\to-6,\qquad r f_{xx}(S)\to6
$$

in every finite $L^p$. Here is the uniformity justification for this particular sequence: midpoint inversion splits the pin rows into even and odd blocks. The even target is $(-r/2,0)$; the odd axial target is $(-1,12r^{-2})$. The centered pin covariance and its inverse remain bounded. Thus the regression mean of $f_{xxxxx}$ is $O(r^{-2})$, its centered moments are bounded, and the pin Taylor identities give
$f_{xxx}(0)=12r^{-2}+O_{L^p}(1)$,
$f_{xx}(0)=O_{L^p}(r^2)$, and the displayed endpoint limits. Mixed Hessian entries have zero regression mean (transverse reflection) and bounded conditional moments.

It follows that, for the actual typed weight,

$$
r^2W_r\longrightarrow
36Z(0)^2\,1_{\{Z(0)<0\}}
\quad\text{in }L^1,
\qquad
r^2Z_r\longrightarrow18\sigma^2>0.
$$

The same polynomial Gaussian bounds give a uniformly bounded second moment of $r^2W_r$. Therefore the weighted law of $\lambda=-A_M>0$ is tight; its limiting distribution is the negative half of the scalar Gaussian tilted by $Z^2$.

But the exact Hermite pin identity forces

$$
M_3\ge12k=12r^{-2}.
$$

The cap event used in [K] requires

$$
\lambda>\frac4{3k}rM_3^2
\ge192rk=\frac{192}{r}.
$$

Hence $Q^W(G_r)\to0$. For example, bound its numerator with Cauchy–Schwarz using the bounded second moment of $r^2W_r$ and the unchanged Gaussian tail of $A_M$; divide by $r^2Z_r\to18\sigma^2$. Thus $Q^W(G_r^c)\to1$, contradicting $C_Br^9$.

**What this does and does not invalidate.** It invalidates the displayed uniform normalized cap-failure bound. It does not show $1-p_r\to1$: failure of a sufficient cap criterion is not failure of elder pairing. It does not invalidate (K2), whose right side includes $(1+|b|+k)^N$, or the lifetime integrals, whose pin density suppresses this sequence by $\exp(-c/r^4)$.

**Minimal repair.** Delete the unrestricted normalized remark, or explicitly restrict $k$ to a fixed compact positive interval and use the parent normalizer floor. A version uniform as $k\downarrow0$ with a fixed upper bound needs its own relative-normalizer estimate and constants. It cannot be asserted uniformly to $k=\infty$ by dropping polynomial target growth. The inspected core theorem does not change.

A fresh 85-digit Decimal regression calculation for $L=4$ confirms the intermediate scales, without serving as the proof:

$$
a=0.9892723933277496\ldots,\quad
\sigma^2=2.1286161985998463\ldots.
$$

At $r=0.01$, the two means of $rf_{xx}$ are approximately $-5.99987238,5.99987238$; $E A_M=0$, $E A_S=0.00989272$, and $\operatorname{Var}A_M=\sigma^2$. The cap demands $\lambda>19200$, while the proposed factor is $r^9=10^{-18}$. The reproducing script is included below.

## IBA2-002/003 — exact periodic covariance and normalization

The positive Fourier weights are

$$
a_n=\frac{\exp(-2\pi^2|n|^2/L^2)}
{\sum_{j\in\mathbb Z^d}\exp(-2\pi^2|j|^2/L^2)}.
$$

Every mode has positive variance for every fixed $L>0$. Rapid summability gives smooth sample paths and all finite derivative-supremum moments. A zero-variance finite jet combination would be a distribution made of derivatives of point masses with every Fourier coefficient zero; it must vanish. Distinct-site jets, and nonduplicated symmetric entries at one site, are consequently nondegenerate. Compact frame/separated-site sets give the needed uniform covariance floors. These arguments give no uniform constants as $L\downarrow0$, $L\to\infty$, or $d\to\infty$; none are supplied by this audit.

The field is stationary on the torus and has signed-coordinate-permutation symmetries. It is not generally $O(d)$-isotropic. If $\mu_j$ are the moments of one spectral coordinate, fourth-order isotropy would require $\mu_4=3\mu_2^2$. Direct product-spectrum evaluation violates this at finite periods. At $L=24$, high-precision image derivatives give

$$
\mu_4-3\mu_2^2\simeq5.5598640762\times10^{-120}.
$$

Binary64 cannot resolve this quantity reliably. Changing coordinates to a frame containing $u$ is legitimate; replacing the covariance in that frame by its $e_1$ value is not.

I recomputed the complete leading normalization:

$$
|\det T_r|=12r^{-(d+3)},\quad
dM\,dS=dz\,dh,\quad dh=r^{d-1}dr\,d\sigma,
\quad d f(M)\,d f(S)=r^3\,db\,dk.
$$

The axial four-row block has determinant $12/r^4$; each transverse two-row block contributes $1/r$. The height Jacobian is taken at fixed $r$; allowing $r$ to vary leaves the complete transformation block triangular. On compact positive $k$-sets,

$$
W_r/r^2\to36k^2(\det A)^2\,1_{\{A<0\}}.
$$

The resulting radial intensity is $rA_r\,dr\,db\,dk\,d\sigma$, with $A_r=12\pi_rE_Q(W_r/r^2)$. All dimension powers cancel:
$(d-1)+3-(d+3)+2=1$.
There is no factor $1/2$, since the pin roles are ordered maximum/saddle roles. Surface measure is ordinary sphere area. Translation integration is divided by the torus volume for the per-volume density.

The lifetime Jacobian is

$$
r\,\frac{dr}{d\ell}=\frac13k^{-2/3}\ell^{-1/3}.
$$

Parity, rather than isotropy, separates odd and even jets. Integrating birth and gap yields

$$
c_{d,L}
=\frac{\Gamma(7/6)}{24^{1/3}\sqrt\pi}
\int_{\mathbb S^{d-1}}
p_G(0)p_{Hu}(0)
\{\operatorname{Var}(f_{uuu}\mid G=0)\}^{2/3}
E[(\det A)^2 1_{\{A<0\}}\mid Hu=0]\,d\sigma(u).
$$

I checked the scalar integral
$144\int_0^\infty k^{4/3}\phi_\tau(12k)\,dk
=\Gamma(7/6)\tau^{4/3}/(24^{1/3}\sqrt\pi)$.
The 24 here is unrelated to the side length. Gaussian moments and compact directions give finiteness; full conditional support gives strict positivity. Field-amplitude scaling gives $c_{af}=a^{-2/3}c_f$, an independent dimensional check.

In $d=2$, the transverse block is scalar and centered after conditioning $Hu=0$, so its cone moment is exactly half its conditional variance. I used that identity to compute the coefficient by independent spectral moments and angular quadrature:

| $L$ | Independent approximate $c_{2,L}$ |
|---|---:|
| 2 | 0.01261759313 |
| 3 | 0.04987943267 |
| 4 | 0.06756865760 |
| 8 | 0.07340691928 |
| 24 | 0.07340691931 |

These are convergence probes, **not certified enclosures**. For $L=2$, 512 versus 1024 angular points was insufficient (difference $3.23\times10^{-8}$); refinement through 2048, 4096, 8192, and 16384 points stabilized the value reported above. For $L=24$, the displayed digits cannot distinguish periodic from infinite-volume values. No new certified value of $c_{3,24}$ is claimed.

For [PER]'s limited first-variation statement, I independently recover

$$
P_3(L)=-\frac{2}{21}L^6+\frac75L^4-3L^2,
\quad P_3(24)=-620813376/35,
$$

and $P_3(24)e^{-288}\simeq-1.4862170291\times10^{-118}$.
This is a first-variation term, not a bound on the full nonlinear remainder or its sign. That distinction is respected in [PER]; later remainder packets were not audited.

## IBA2-004/013 — topology and the marked Kac–Rice bridge

A local maximum-saddle pair need not be an elder pair. I tried to break the actual cap theorem by an exterior older maximum, a bypass saddle, reentry into the cap, and by separating the transverse ridge from the gradient trajectory.

Those attacks fail under [CAP]'s hypotheses. Strict transverse concavity supplies a unique ridge. Its third derivative is obtained by differentiation along the ridge; the $h''$ terms cancel through the transverse critical equation, not by identifying the ridge as a flow line. The proof controls the left end face, the entire transverse side, and the face through $S$. Every boundary value is at most $f(S)$, with equality on that face only at $S$. The cap contains no point above $b$. A ridge path reaches a point above $b$ with minimum $f(S)$. Therefore every global older-point path has minimum at most $f(S)$, and one achieves it. No outside continuation can bypass its first exit.

For Morse functions with distinct critical values this maximin equality identifies the unique elder death saddle. The separate gradient-branch argument uses monotonicity and confinement inside the cap; it does not assume the ridge is an ascending trajectory. The general-$k$ rescaling of the deterministic theorem is valid. It does not rescale the Gaussian covariance or justify IBA2-001's probability remark.

For [CU]/[E3]'s cusp decision, I checked the same two-sided logic: accepted pairs have a path above the saddle bottleneck plus a bounded trap; positive and negative rejected cusp sectors are treated differently. The negative sector traps the candidate maximum below the proposed death level; the positive sector supplies a strictly better path. Exterior topology cannot alter those conclusions on the stated margins.

There is also no need to assume a deterministic theorem that every short bar has nearby endpoints. All distinct pairs are counted with the global mark. [FAR] controls distant pairs by a different argument:

$$
\ell(M)\ge c_L\lambda_{\min}(-H_M)^3/K^2.
$$

This follows from the height drop on a full embedded sphere around $M$. Conditioning on the two Hessians, retaining $|\det H_M|$, and integrating the soft eigenvalue gives

$$
\int_0^\infty \lambda\min\{1,C\ell\lambda^{-3}\}
(1+\lambda)^N e^{-c\lambda^2}\,d\lambda=O(\ell^{2/3}).
$$

Thus fixed-separation elder density is $O(\ell^{2/3})$; the determinant factor is essential. Constants may diverge when the separation cutoff shrinks.

I checked the primary Kac–Rice source, Armentano–Azaïs–León, arXiv:2304.07424v3, Theorems 2.2 and 7.1:
https://arxiv.org/html/2304.07424v3 .
The Gaussian theorem applies off the diagonal. Its weighted theorem directly permits the continuous cylinder stage; it does not directly declare every Borel elder indicator continuous. [B] then identifies two finite measures on location × $C^2$ using a generating algebra and the functional monotone-class theorem. This is a valid route to the global Borel mark. Full-field regression supplies the required conditional kernel. Height disintegration and diagonal exhaustion preserve the single Hessian-product factor. The density statements concern the specified canonical version; arbitrary versions differ on lifetime-null sets.

## IBA2-005/006 — weighted law, dimensions, and the dangerous gap boundary

The relevant numerator is $E_Q[W_r1_{G_r^c}]$. On typed support, the canceled-pivot estimate gives

$$
W_r\le \frac{r^2h^2}{4}
\lambda_1(\lambda_1+\tfrac32rh)
\prod_{j=2}^{m}\lambda_j(\lambda_j+rh),
\quad m=d-1.
$$

The first soft factor comes from the maximum determinant; the second includes the saddle mixed-square term. This is not an estimate of independent Hessian entries.

The eigenvalue Jacobian is a Lebesgue-coordinate identity. A full Gaussian density bound, integrated over the compact angular space, suffices; GOE structure is unnecessary. For $m\ge2$, a width $O(rU^2)$ gives
$\int\lambda(\lambda+CrU)d\lambda=O(r^3\operatorname{poly}U)$.
The remaining eigenvalues are integrated down to zero, so corank intersections are not discarded.

For $d=2$, one cannot hold the largest eigenvalue fixed while integrating the smallest. The separate scalar split

$$
\lambda\le Dr(J+\lambda)^2
\ \Longrightarrow\
\lambda\le4DrJ^2\quad\text{or}\quad\lambda>(4Dr)^{-1}
$$

is necessary and is supplied in [P]/[R]. The second branch is controlled with the actual weighted moment. For $d=3$ there is one hard integration variable; for $d=4$, two; for $d\ge5$, the same polynomial Gaussian argument applies in higher finite dimension. None introduces inverse moments at the boundary.

For the unrestricted leading law, the unnormalized majorant is
$A_r\le H(b,k)$, with Gaussian decay in $b,k$, and
$H(b,k)k^{-2/3}$ is integrable at zero. This alone establishes dominated convergence for the leading coefficient; it would not alone prove an absolute bounded remainder.

[R] supplies the stronger estimates

$$
|A_r-A_0|\le r(k+r)H,\qquad
B_r\le(r/k)^3H\quad(r\le k),
$$

where $B_r$ is the unnormalized cap-loss intensity and $H$ contains polynomial target growth and Gaussian decay. At lifetime $\ell$, the actual lower endpoint is $a=\ell/r_0^3$. For the candidate error,

$$
k^{-2/3}rk=\ell^{1/3},\qquad
k^{-2/3}r^2=\ell^{2/3}k^{-4/3}.
$$

Integrating the latter from $a$, not from zero, gives $O(\ell^{1/3})$ in the scaled density.

The split $\eta=\ell^{1/6}$ is sufficient, not uniquely necessary. Its two selection-error powers are
$\eta^{7/3}=\ell^{7/18}$ and
$\ell\eta^{-11/3}=\ell^{7/18}$, both smaller than the required $O(\ell^{1/3})$ scaled error. The small-$k$, $r/k\gg1$ region uses the direct intensity bound. Gaussian tails absorb large $k,b$. Adding the bounded far term yields the claimed absolute $O(1)$ remainder. This derivation never uses the false normalized remark.

## IBA2-007/008/009/012 — alternative degeneracies and higher-order remainders

For a mechanism with radial intensity $r^\alpha dr$ and gap comparable to $r^m$, the transformed density is proportional to
$\ell^{(\alpha+1)/m-1}$. The height exponent $m$ alone therefore cannot decide dominance.

With bounded derivatives and two exact stationary endpoints, a nondegenerate quadratic longitudinal profile cannot produce two nearby distinct critical points. The axial Hessian must soften. The ordinary fold has $\alpha=1,m=3$, yielding $-1/3$. A distant near-tie is another mechanism; the global elder mark and the far barrier handle it, rather than a local cubic approximation.

The first competing local scale is real: $k=\kappa r$, $\ell=\kappa r^4$. Its $k$-window has width $O(r)$, and its determinant product has an additional $r^2$ suppression. Thus its integrated radial mass is $r^4dr$, giving a density of order $\ell^{1/4}$, not $\ell^{-1/4}$.

Eliminating the transverse fiber in [CU] gives

$$
g(X)=\kappa\{2(X+\tfrac12)^2(X-1)+3\phi(X^2-\tfrac14)^2\},
\quad
\phi=\frac{f_4-3\gamma^TA^{-1}\gamma}{72\kappa}.
$$

Then $g'=6\kappa(X^2-\tfrac14)(1+2\phi X)$.
The typed window is $|\phi|<1$, whereas the actual elder window is $|\phi|<1/3$. The difference is substantial and defeats the hypothesis that every typed cusp pair is a persistence pair.

Writing $\Delta=\det A$ and $Y=f_4\Delta/12-\gamma^T\operatorname{adj}(A)\gamma/4$, I recover

$$
w^{\rm cand}=(36\kappa^2\Delta^2-Y^2)1_{A<0,\ |Y|<6\kappa|\Delta|},
\quad
w^{\rm eld}=(36\kappa^2\Delta^2-Y^2)1_{A<0,\ |Y|<2\kappa|\Delta|}.
$$

The two scalar integrations are

$$
\int_0^\infty\min(Y^2,36\Delta^2s^{-8})\,ds
=\frac87\,6^{1/4}|Y|^{7/4}|\Delta|^{1/4},
$$

and the elder loss integral is

$$
\frac{16}{7}\,2^{1/4}|Y|^{7/4}|\Delta|^{1/4}.
$$

Hence the second coefficient is negative and
$I^{\rm cand}/c_1=3^{1/4}/2$. I found no lost factor of two in these formulas.

The shared first determinant correction cancels the $O(r)$ term in their product. In [T], the fold coefficient satisfies $A_2(0)=-12\pi_0E[Y^2 1_{A<0}]$, and

$$
c_2=\frac13\int(A_2(b,k,u)-A_2(b,0,u))k^{-4/3}\,db\,dk\,d\sigma.
$$

Subtracting the constant makes the zero-end integral convergent; the constant itself remains integrable at infinity against $k^{-4/3}$. The linear-in-cutoff overlap terms cancel between fold and cusp regions.

For $\rho_f=\ell^{3/11}$, $\rho_c=\ell^{2/9}$, I independently evaluated the error exponents:

$$
6/11,\ 4/11,\ 5/11,\ 4/11,\ 4/9,\ 2/3,\ 4/9,\ 5/9.
$$

Their minimum is $4/11>1/3$. [E3]'s intermediate split at $\ell^{2/15}$ contributes $O(\ell^{2/5})$; [FAR] contributes $O(\ell^{2/3})$. Thus the assembly really gives the stated $O(\ell^{4/11})$ once the stated kernel estimates are granted, not merely a little-o of the leading singularity. I also checked [E3]'s rescaled-Hessian covariance argument and the conditional norm layers used to charge the soft maximum twice.

For [N], the first odd cusp correction has zero expectation under the centered odd-jet law. Moving to target $k=\kappa r$ costs another factor $k$, so fixed compact $\kappa$-windows have an $O(r^2)$ error. This removes an $\ell^{1/2}$ contribution **from that compact cusp window**, not from every possible matching/tail region. I did not derive a complete next-order global expansion from this parity statement.

For several soft transverse eigenvalues, on compact marks and bounded derivative/hard-eigenvalue sectors, $q$ eigenvalues of order $r$ cost

$$
r^{q}\times r^{q(q-1)/2}\times r^{2q}
=r^{3q+q(q-1)/2}
$$

under the normalized endpoint weight: widths, internal Vandermonde, and two determinants, respectively. This gives $r^3,r^7,r^{12},r^{18}$ for $q=1,2,3,4$. The estimate includes the determinant suppression and is not just a codimension slogan. It is not, by itself, a global estimate for all clusters or all $k$. Arbitrarily small effective quartic coefficients and higher-order normal forms require uniform sector/tail control if they are to be assigned their own sharp expansion terms. The general leading/remainder majorants cover exceptional configurations without needing to classify each separately; this audit did not complete a sharp separate quintic-and-higher taxonomy.

## IBA2-010/011 — clusters, factorial moments, and collisions

I treated the C6 and spectral cluster claims separately from the lifetime-density theorem.

In the log-loss route [CL], an analytic polynomial-approximation degree yields a critical-count majorant with stretched-exponential tail. A split at a degree of order $\log^2(1/r)$, together with the first-moment estimate, explains the logarithmic factorial bound. This does not by itself prove the sharper $E_W(N)_q=O(r^3)$.

[CP] uses a witnessed/conditional analytic-count argument for that upgrade. I inspected its conditional degree-tail and marked-count steps. The residual Gaussian supremum bounds survive conditioning, but the quantitative mean and complex-covariance floors depend on separate local, collar, shell, and remote estimates. I did not fully rederive all those source interfaces in this pass. Thus I am not labeling the entire sharp C6 theorem independently revalidated.

In [SC], the one-soft-eigenvalue rescaling, original determinant weight, finite nonempty spectral measure, and cubic root classifier are mutually consistent in the inspected derivation. I attempted to falsify the classifier by independently eliminating the two quadratic critical equations and counting roots whose heights lie strictly between the pins. With seed 20261003, 19,909 nonboundary sampled typed jets were tested; **zero mismatches** occurred. Samples with a small elimination denominator or values near a decision boundary were excluded. This is a finite falsification attempt, not a proof of the classifier or of Gaussian convergence.

The spectral proof needs more than generic simplicity: it controls the large-soft-coordinate branch and uses an integrable occurrence-weight majorant. Its global count law also uses regional first moments, remote singleton estimates, and mixed near/far suppression. Their exact imports remain part of the conclusion.

Given those inputs, I checked [TS]'s two-scale configuration argument. A remote singleton has positive limiting mass, so scaling every point by $1/r$ cannot give the proposed tight global Euclidean law. For $d\ge3$, the limiting microscopic doublet has collinear transverse positions while finite-$r$ factorial spatial laws are absolutely continuous; spatial total-variation convergence would be false. Weak configuration convergence and discrete count total variation are compatible with this. The source keeps these distinctions.

[SUB]'s conclusion for separations $r\epsilon(r)$, $\epsilon(r)\to0$, follows from finite ordered-pair measure convergence and no mass on the limiting coincidence diagonal. Its multiplicity upgrade correctly uses higher moments:

$$
r^{-3}E[N^pC_r]\le
(r^{-3}EN^{2p+4})^{1/2}
(r^{-3}P(C_r>0))^{1/2}\to0.
$$

Then $Z_r\sim z_0r^2$ restores the original numerator $o(r^5)$. On compact positive $k$-windows, the collision-marked lifetime density is $o(\ell^{2/3})$, using the exact $k^{-5/3}$ Jacobian factor. No algebraic rate, all-$k$ uniformity, or identification of this collision mark with all selection failures follows.

Finally, the rare-count compound-Poisson construction in [RARE] uses the conditional nonempty mark law. Its approximation is a legitimate abstract rare-variable statement; it does not prove independent spatial clusters in one field. Likewise factorial bounds alone neither determine a unique limiting cluster law nor exclude subscale collisions. The separate spectral/configuration convergence inputs are indispensable.

## Required repairs and remaining work

1. **Actual repair required:** correct [K]'s normalized-form remark as in IBA2-001. Preserve its unnormalized theorem and downstream dependency distinction. No core theorem change follows from this finding.
2. **For an independently complete cluster closure:** rederive the precise D5/remote/collar/witness covariance and factorial inputs consumed by [CP]/[SC], including their compact-mark uniformity. This is unfinished audit coverage, not a demonstrated omission in those uninspected proofs.
3. **For a next global remainder term:** prove estimates uniform across matching endpoints; compact-$\kappa$ parity cannot be integrated over an expanding window without that work.
4. **For certified finite-period coefficients:** supply interval quadrature and explicit Fourier/image tails with full angular covariance. My numbers are not a certificate; [PER]'s first variation alone does not certify the full correction.
5. **For sharp higher-degeneracy or collision rates:** supply weighted sector estimates and spatial-tail uniform integrability, not only zero-probability exact degeneracy statements or count-moment bounds.

No numerical confidence score is assigned. No other agent is being asked to validate this report.

## Suggested falsification experiments

- Run exact-period regression on $k=r^{-2}$ and measure the cap event separately from the elder event. The transverse-curvature distribution stays bounded while the cap threshold diverges; confusing those two events would conceal IBA2-001.
- Simulate the cusp ridge on both sides of $|\phi|=1/3$ and $|\phi|=1$, retaining the determinant weight. Compare candidate and actual elder decisions after altering the exterior field.
- Compute $c_{2,L}$ at $L=2,3,4$ using both full directional covariance and an isotropic substitution. The discrepancy is numerically resolvable there, unlike at $L=24$.
- Stress cubic classifier discriminants and height ties with arbitrary precision. The random probe intentionally avoided precisely those boundary layers.
- For cluster laws, measure ordered pairs and factorial moments under the original single weight $W/Z$; distinguish microscopic doublets, remote singletons, and mixed events. Unweighted histograms do not test these theorems.
- Test small effective quartic coefficient jointly with a soft transverse eigenvalue; record weighted mass, not merely the frequency of exact degeneracies.

## Dependency map

| Conclusion | Necessary mathematical inputs | Outcome of this pass |
|---|---|---|
| Leading $c\ell^{-1/3}$ law | Positive spectrum/jet rank → pins and Kac–Rice → determinant weight → cap → all-mark domination | Reconstructed; no contradiction found |
| Absolute $O(1)$ remainder | Index-filtered determinant stability + target-growth bounds + lower-cutoff/small-$k$ integration + far bound | Reconstructed; survives |
| $c_1\ell^{1/4}$ correction | Shared determinant correction + cusp model + full boundary traps + weighted tails | Inspected; survives attacks described |
| Third term and $O(\ell^{4/11})$ | Fold expansion + overlap subtraction + elder cusp estimate + rescaled soft maximum + far estimate | Assembly checked; no stronger rate claimed |
| Fixed-window no-linear cusp correction | Odd-jet parity + quantitative moving-edge control | Scope retained; not a global next-term theorem |
| Support-$\{1,2\}$ rare count law | Regional first moments + witnessed factorial bounds + one-soft spectral convergence + remote/mixed estimates | Conditional chain inspected; full regional rederivation incomplete |
| Sub-pin collisions $o(r^3)$ | Two-scale ordered-pair convergence + simple limiting roots + count-moment UI | Downstream deduction checked |
| Uniform $Q^W(G_r^c)\le C_B(r/k)^3$, all $k>0$ | Removal of polynomial target growth | **Contradicted by IBA2-001; not needed by the rows above** |

## Immutable source inventory

Coverage refers to mathematical statements and proof bodies only; embedded opinion/status text is excluded. Git blob IDs are content identifiers, not proof certificates.

| Key | Source at pinned commit | Git blob ID | Actual coverage |
|---|---|---|---|
| [P] | [imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md) | dfed3b8d318a3ab1950957f393307733a4bef3f2 | §§2–7, 9–11, 13–15; global genericity/measurability read with B |
| [CAP] | [imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md) | 0633aca3c2a2882b0de4399da0a75d64c2e6b2e1 | deterministic §§1–5; excluded historical numerical probability corollary |
| [B] | [frontiers/lifetime_moment_boundary_20260930/sources/BOREL.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/lifetime_moment_boundary_20260930/sources/BOREL.md) | fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a | replacement §9 only |
| [R] | [frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md) | 247b3ecf80bfbe896948d5d489b2d5842a81c481 | §§2–7 |
| [K] | [frontiers/c7_total_bounded_20260929/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/c7_total_bounded_20260929/PROOF.md) | 28748b086ec6761ef467dc67cba475fbdf8b7455 | Theorem K, proofs §§2–4, normalized remark; lower bound import not audited |
| [E] | [frontiers/remainder_vanishing_20260930/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/remainder_vanishing_20260930/PROOF.md) | 441152dfc1d8d00cb268cebdb0394212a5bfacdb | Lemma E, determinant expansion and inertia cases |
| [RP] | [frontiers/remainder_rate_20260930/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/remainder_rate_20260930/PROOF.md) | abfb98ae41d66a86d3cb094162db6547d0f29e7e | §§0–4 mathematical chain |
| [FAR] | [frontiers/far_elder_rate_20260930/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/far_elder_rate_20260930/PROOF.md) | 072601145601a1021037a2e91bf8826c1c971ecc | statement and §§1–3 |
| [CU] | [frontiers/cusp_second_order_20261001/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/cusp_second_order_20261001/PROOF.md) | f6df5a7356a54771b0052e42a864a140709903ad | §§0–7 mathematical chain |
| [T] | [frontiers/candidate_third_order_20261001/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/candidate_third_order_20261001/PROOF.md) | 70ca57ef05199e75417d7d57928c1e1c93bf944e | §§0–4 |
| [E3] | [frontiers/elder_third_order_20261001/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/elder_third_order_20261001/PROOF.md) | c8767ddecf1041d234f685e95af582073a7be393 | §§0–4 |
| [N] | [frontiers/elder_cusp_parity_20261002/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/elder_cusp_parity_20261002/PROOF.md) | 16a1db0658918057ff741df888c13702763d9919 | §§0–5; improvement consumes an additional packet not fully audited |
| [PER] | [frontiers/side24_periodization_20260929/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/side24_periodization_20260929/PROOF.md) | 1a84f215c53d62918f37517300c350693da62586 | first-variation argument; not later nonlinear remainder packets |
| [CL] | [frontiers/c6_factorial_moment_20260929/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/c6_factorial_moment_20260929/PROOF.md) | f5bd013b7134be9efd246fc9bbb8d18ad5a728bc | analytic-count/log-loss route |
| [CP] | [frontiers/c6_palm_route_20260929/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/c6_palm_route_20260929/PROOF.md) | 89eb8adf08fe7afc2cab9662d3a9875c05ae5cc5 | §§4–6 conditional analytic majorant/Palm upgrade; imported regional estimates not fully audited |
| [RARE] | [frontiers/c6_rare_cluster_laws_20260929/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/c6_rare_cluster_laws_20260929/PROOF.md) | 2ab625cedfc2e47575c4a7fa4853413418c532da | abstract rare-count deductions |
| [SC] | [frontiers/spectral_cluster_closure_20260929/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/spectral_cluster_closure_20260929/PROOF.md) | 16c56821b52fd76b0be791622b9c3809eafde75a | §§1–8 mathematical construction; retained named regional imports |
| [TS] | [frontiers/two_scale_cluster_geometry_20260929/TWO_SCALE_LAW.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/two_scale_cluster_geometry_20260929/TWO_SCALE_LAW.md) | a32fd5f7d941bbe1fe943df045b1e0fbec8d691c | §§2–6 downstream deductions |
| [SUB] | [frontiers/subpin_collision_20261002/PROOF.md](https://github.com/d6g8k5htny-coder/Math-/blob/b55b8f34d09084300088105b90fe60759fd7e144/frontiers/subpin_collision_20261002/PROOF.md) | 4d903d0f834fb2077257104cd91822fe253dde8d | §§1–4 |

## Reproduction appendix

The following independent scripts use only the Python standard library. They were executed in this pass. Floating arithmetic, finite image/Fourier sums, and random tests do not prove asymptotic bounds. The large-$k$ contradiction above is analytic and does not rely on either script.

The first script uses 85-digit Decimal arithmetic with image indices $-12,\ldots,12$ at $L=4$. Its SHA-256 is 22bbfeb73ee4b7badf0a8bfcc3f0666199ec77b4562351520fb8655ddc6a5270.

```python
"""Decimal regression falsification check; analytic argument is in the report."""
from decimal import Decimal as D, getcontext
getcontext().prec = 85
L = D(4)
def hermite(n,x):
    a,b=D(1),x
    if n==0:return a
    for j in range(1,n):a,b=b,x*b-j*a
    return b
norm=sum((-(L*j)**2/2).exp() for j in range(-12,13))
def kd(n,z):
    return sum((-1)**n*hermite(n,z+L*j)*(-(z+L*j)**2/2).exp() for j in range(-12,13))/norm
def cov(p,q):
    x,alpha=p;y,beta=q
    return (-1)**sum(beta)*kd(alpha[0]+beta[0],x-y)*kd(alpha[1]+beta[1],D(0))
def solve(a,b):
    a=[row[:]+[v] for row,v in zip(a,b)];n=len(b)
    for i in range(n):
        j=max(range(i,n),key=lambda j:abs(a[j][i]));a[i],a[j]=a[j],a[i]
        v=a[i][i];a[i]=[x/v for x in a[i]]
        for j in range(n):
            if j!=i:
                v=a[j][i];a[j]=[x-v*y for x,y in zip(a[j],a[i])]
    return [row[-1] for row in a]
a=-kd(2,D(0));sig2=kd(4,D(0))-a*a
print('a=',a,'sigma2=',sig2,'limit r^2 E_Q W=',18*sig2)
for r in map(D,['.1','.03','.01','.003']):
    pins=[(x,alpha) for x in [-r/2,r/2] for alpha in [(0,0),(1,0),(0,1)]]
    mat=[[cov(p,q) for q in pins] for p in pins]
    v=[D(0),D(0),D(0),-r,D(0),D(0)]
    coeff=solve(mat,v)
    hs=[(x,alpha) for x in [-r/2,r/2] for alpha in [(2,0),(1,1),(0,2)]]
    means=[sum(cov(h,p)*z for p,z in zip(pins,coeff)) for h in hs]
    am=[cov(hs[2],p) for p in pins]
    va=cov(hs[2],hs[2])-sum(x*y for x,y in zip(am,solve(mat,am)))
    print('r=',r,'r Hxx means=',float(r*means[0]),float(r*means[3]),'A means=',float(means[2]),float(means[5]),'Var A_M=',float(va),'cap necessary lambda>',float(192/r),'(r/k)^3=',r**9)
```

The coefficient/classifier probe has SHA-256 f2475dfc6073bdb6ab972613e6ecd0e4aef27b882963efebb9b3dfc379dc9e11. Its initial L=2 angular result requires the refinement described above; that initial output is deliberately not presented as the final approximation.

```python
"""Independent falsification probes. Floating-point checks are not proofs."""
import json, math, random
from decimal import Decimal, localcontext
from fractions import Fraction

def moments(L,N):
    z=sum(math.exp(-.5*(2*math.pi*j/L)**2) for j in range(-N,N+1))
    return {p:sum((2*math.pi*j/L)**p*math.exp(-.5*(2*math.pi*j/L)**2) for j in range(-N,N+1))/z for p in (0,2,4,6)}

def mul(a,b):
    out={}
    for (i,j),x in a.items():
        for (k,l),y in b.items():out[i+k,j+l]=out.get((i+k,j+l),0)+x*y
    return out

def power(a,n):
    out={(0,0):1}
    for _ in range(n):out=mul(out,a)
    return out

def expectation(a,m):return sum(v*m.get(i,0)*m.get(j,0) for (i,j),v in a.items())

def inner(a,b,m):return expectation(mul(a,b),m)

def inverse2(c):
    det=c[0][0]*c[1][1]-c[0][1]**2
    return [[c[1][1]/det,-c[0][1]/det],[-c[0][1]/det,c[0][0]/det]],det

def quadratic(v,a):return sum(v[i]*a[i][j]*v[j] for i in range(2) for j in range(2))

def integrand(theta,m):
    u={(1,0):math.cos(theta),(0,1):math.sin(theta)}
    v={(1,0):-math.sin(theta),(0,1):math.cos(theta)}
    x={(1,0):1};y={(0,1):1}
    V=[power(u,2),mul(u,v)];A=power(v,2);T=power(u,3)
    C=[[inner(a,b,m) for b in V] for a in V]
    ci,det=inverse2(C)
    av=[inner(A,a,m) for a in V]
    variance=inner(A,A,m)-quadratic(av,ci)
    tau2=inner(T,T,m)-(inner(T,x,m)**2+inner(T,y,m)**2)/m[2]
    assert min(det,variance,tau2)>0
    return (1/(2*math.pi*m[2]))*(1/(2*math.pi*math.sqrt(det)))*tau2**(2/3)*variance/2

def coefficient(L,N,n):
    m=moments(L,N)
    val=sum(integrand(2*math.pi*(i+.5)/n,m) for i in range(n))*2*math.pi/n
    return math.gamma(7/6)/(24**(1/3)*math.sqrt(math.pi))*val

def roots2(a,b,c):
    if abs(a)<1e-12:return [-c/b] if abs(b)>1e-12 else []
    disc=b*b-4*a*c
    if disc<0:return []
    return [(-b-math.sqrt(disc))/(2*a),(-b+math.sqrt(disc))/(2*a)]

def classify_numeric(k,s,a,b,c):
    # Extra critical roots obey Ax+Bz+C=0 after eliminating x^2-1/4.
    A=a-12*k*b/a;B=b/2-6*k*c/a;C=-12*k*s/a
    if abs(A)<1e-8:return None
    q=-B/A;t=-C/A
    zs=roots2(6*k*q*q+a*q+b/2,12*k*q*t+a*t,6*k*(t*t-.25))
    count=0
    for z in zs:
        x=q*z+t
        h=2*k*x**3-1.5*k*x-.5*k+s*z*z/2+a*(x*x-.25)*z/2+b*x*z*z/2+c*z**3/6
        if abs(h)<1e-7 or abs(h+k)<1e-7:return None
        if -k<h<0:count+=1
    return count

out={}
for L in (2,3,4,8,24):
    N=max(20,math.ceil(3*L))
    m=moments(L,N)
    c1=coefficient(L,N,512);c2=coefficient(L,N+10,1024)
    out[str(L)]={"c_d2":c2,"refinement_difference":abs(c2-c1),"mu4_minus_3mu2sq":m[4]-3*m[2]**2,"direction_integrand_ratio_axis_diagonal":integrand(0,m)/integrand(math.pi/4,m)}

# Separate arbitrary-precision image-sum derivatives; do not subtract near-equal binary64 numbers.
with localcontext() as ctx:
    ctx.prec=170
    L=Decimal(24);q=(-L*L/2).exp();z=Decimal(1);nums=[Decimal(-1),Decimal(3),Decimal(-15)]
    for n in range(1,4):
        x=L*n;ex=(-x*x/2).exp();z+=2*ex
        he=[x*x-1,x**4-6*x*x+3,x**6-15*x**4+45*x*x-15]
        nums=[a+2*b*ex for a,b in zip(nums,he)]
    a=-nums[0]/z;m4=nums[1]/z
    anis=m4-3*a*a
    out['side24_decimal']={"anisotropy_m4_minus_3a2":str(anis),"leading_anisotropy_ratio":str(anis/(2*L**4*q)),"P3_exact":str(-Fraction(620813376,35)),"P3q":str((-Decimal(620813376)/35)*q)}

rng=random.Random(20261003);tested=0;fail=[]
for _ in range(20000):
    k=rng.uniform(.2,2);a=rng.uniform(-4,4);b=rng.uniform(-4,4);c=rng.uniform(-4,4)
    if abs(a)<.02:continue
    B=b-a*a/(12*k);s=-abs(B)/2-rng.uniform(.02,4)
    D=(c-a*b/(4*k)+a**3/(72*k*k))/2
    threshold=-(s-B)**2*(B+2*s)/(12*k)
    pred=(2 if D*D<threshold else 1) if s>B else (1 if D*D>threshold else 0)
    got=classify_numeric(k,s,a,b,c)
    if got is None:continue
    tested+=1
    if pred!=got:fail.append([k,s,a,b,c,pred,got])
out['cubic_classifier']={"tested":tested,"mismatches":len(fail),"examples":fail[:3]}

# Integrate the cusp deficit directly, splitting at the exact change of branch.
y=1.7;d=.8;s0=(6*d/abs(y))**.25
direct=-(y*y*s0+36*d*d/(7*s0**7))
formula=-(8/7)*6**.25*abs(y)**1.75*abs(d)**.25
out['cusp_deficit']={"direct":direct,"formula":formula,"difference":abs(direct-formula)}
print(json.dumps(out,indent=2))
```

