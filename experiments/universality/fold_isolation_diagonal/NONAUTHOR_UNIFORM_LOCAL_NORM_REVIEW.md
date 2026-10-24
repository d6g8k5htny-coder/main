# Nonauthor review of the uniform local-norm diagonal proposal

**Verdict: ACCEPT_SCOPED.** I found no blocking mathematical error in the frozen proposal's physical spectral-moment implication, explicit cutoff/trace calculation, finite PSD Gaussian exponential moment, quantitative H6-to-C4 operator estimate, deterministic covering argument, or growing-K/L diagonal cancellation. No changes are required for equations (1)–(24) under the stated hypotheses. This verdict concerns the new local mathematical argument. It is not source-law admission, a formal proof, or point-process/scientific promotion.

Actual reviewer: OpenAI/Codex `/root/c261a_local_runtime_20261016`, 16 October 2026 UTC. Root and `/root/core_theorem_audit_20261015` are the authors. I did not author the frozen proposal or its control files. I read all new files fully and checked the bounded primary Fourier import directly. I consumed the two earlier lemmas at their exact accepted identities; I did not duplicate their first reviews or rerun their controls. This is a separate actual nonauthor, source-exposed review in the same OpenAI lineage. **Organizational-independence credit: 0.** No other reviewer is implied by this report.

## 1. Exact reviewed object and evidence

| New frozen input | Bytes | SHA-256 |
|---|---:|---|
| `UNIFORM_LOCAL_NORM_DIAGONAL_PROPOSAL.md` | 26,687 | `8e1eefca9888d43824a9fd9caca34c44d2de4bf8e30c668b474fa2097491c76b` |
| `uniform_local_norm_controls.py` | 6,486 | `4b84755b771598ed34193c075718a41da447719f90a2ff5c6a17d48d0abc3dc4` |
| `UNIFORM_LOCAL_NORM_CONTROLS.json` | 1,790 | `5d6cdb2d4925c6ba0b1b545d7a5aca1977655c1d18aa1ea29a82273f30845ece` |
| `uniform-local-norm-primary-sources/SOURCE_IDENTITY.json` | 443 | `92bf5297b309e947f42ee1dc837487775e7341bc7c1051e493899963ff85891a` |

The saved Simon PDF is 940,975 bytes, SHA-256 `e1f1b2f51d2558aa467a8c064fe82e7b9e157ad28c9c75f01e38596cde656870`; I verified those bytes. I directly read Lecture 5's Fourier properties (a)–(h), including the unitary derivative, inversion and Parseval/Plancherel formulas, and its displayed Cauchy–Schwarz Sobolev proof. I do not claim to have read the entire 118-page PDF. The proposal's explicit derivative constants are calculated below using its own stated norm, rather than borrowed from an unspecified embedding constant. [Leon Simon, Lectures on PDE, Lecture 5](https://math.stanford.edu/~lms/lecs-on-pde.pdf)

The inherited isolation proposal/review remain SHA-256 `827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30` / `f06c61e3f3e60c795a1a14e824640f1689d4143dba4f6698cada7a41929eec90`. The finite-count proposal/review remain `621cee96854aa440498cecf5452a8d1e23f90a4458779c3286c18eba106fe138` / `16152427f7013822ac1bba81142429800334e6bf0f354a6439b23a1997bc4524`. The exact source law is still Main `85b928b326ff0586715a56ae91f51d3fbd89557a`, `finite_gaussian_contact/PROOF.md`, 25,367 bytes, SHA-256 `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`. The six preceding author files and both preceding reviews were verified unchanged.

At 2026-10-16T02:16:05Z I ran the new author controls and my separate `nonauthor-uniform-local-norm-checks/review_checks.py` using `/Library/Developer/CommandLineTools/usr/bin/python3 -B`, Python 3.9.6. Both exited 0 with empty stderr. Author stdout reproduces all 1,790 frozen bytes exactly. My 2,132 exact finite cases passed, including 2,001 unequal/singular PSD spectra and a separately reindexed trace-constant calculation. Commands, timestamps, wall times, stdout/stderr and hashes are retained in `nonauthor-uniform-local-norm-checks/CHECK_RUN_SUMMARY.json` and its process-record sets. Fourteen consumed input identities were verified before and after, unchanged. No previous control program was rerun.

These finite checks supplement the mathematical review; they do not mechanically establish continuous derivative bounds, Fourier theorems, Gaussian integrals, isolation, genericity or asymptotic limits.

## 2. The uniform family hypothesis is sufficient and is not inferred from smoothness

The actual law is centered finite Gaussian with deterministic spectral parameters, independent standard real sine/cosine coefficients, positive symmetric full-square weights, and period `L>=Lmin>0`. The weights may vary between family members; they are not selected from a realized field. For a coordinate derivative of order j<=6 its exact variance is the sum of the corresponding squared physical-frequency monomials times q. The inequality

`r^(2j)<=1+r^12`, `0<=j<=6`,

therefore gives the required bound `1+M12` under normalized total mass. It covers the zeroth derivative as well as all orders through six. With total mass merely bounded by a fixed M0, the same calculation gives `M0+M12` throughout. This replacement propagates to the trace T and the eventual exponential constant c. A fixed M0 and a uniform physical M12 are sufficient for this variant.

M12 alone does not bound the full localized H6 norm. A growing independent DC Gaussian changes the zeroth variance and, after multiplication by the cutoff, all its cutoff-derivative contributions, while leaving every positive physical spectral moment unchanged. The proposed `F_A=A Z0+G` example makes the trace diverge as `A^2||chi_b||_H6^2`. This is a correct falsifier of the stated full-norm trace inference without mass control. Its purpose concerns this full-norm method; it does not claim that adding a constant changes critical locations or gap marks.

Likewise, finite smoothness at each K does not bound physical derivative variances across a family. At fixed L, a fixed positive normalized mass on the axial K-mode pair makes the sixth derivative variance grow as `(omega K)^12`; the inner-cutoff integral gives the stated trace lower bound. Fourth derivative variance already grows as K^8. These examples are outside (U), rather than counterexamples to the theorem with (U).

For normalized weights and `K/L<=Lambda`, every physical frequency is at most `2pi sqrt(2)Lambda`, so the displayed uniform M12 is valid. No positive lower bound on individual weights is needed for the moment/trace argument. Strict positivity remains a separate requirement for the accepted almost-sure real-parameter critical-count theorem. The proposal correctly refrains from certifying (U) for arbitrary project spectra or moving families without a source-bound check.

All asserted constants can depend on fixed `Lmin`, M12 (or M0,M12), a, `k_-`, rho and the fixed lifetime-band cap. They are uniform in K,L and coefficient dimension within those bounds. Allowing Lmin to vanish or the lower marks to vanish is not covered.

## 3. The explicit smooth cutoff and trace calculation check out

With `b=min(1,Lmin/16)`, one has `b>0`, `b<=1` and `6b<Lmin<=L`. The denominator in J is positive on the whole real line. On its transition interval [0,1], at least one argument of psi is at least 1/2, giving the denominator lower bound `exp(-2)`. Outside that interval J is exactly 0 or 1. The flatness of `exp(-1/t)` at zero follows because an exponential dominates every fixed inverse power; all derivatives are `exp(-u)` times finite polynomials in u. Consequently all endpoint derivatives match the constant pieces. Theta is identically one near the only possible absolute-value singularity s=0. Thus the product cutoff is compactly supported and smooth, with precisely the claimed inner square and support square.

The recurrence `L_(j+1)=u^2(L_j-L_j')` follows by differentiating with `du/dt=-u^2`. The Aj bounds apply on u>=1, exactly the range required for transition derivatives. The reciprocal-denominator bound is at most `exp(2)<8`; differentiating `D times D^-1=1` gives the stated En recurrence. The Leibniz formula then gives Gn. For positive derivative orders theta's support has total length two, giving `d_j=2G_j^2`; order zero instead uses support length six and the actual bound `|theta|<=1`. The different d0 choice is correct.

Scaling the two one-dimensional derivative integrals gives

`||partial^gamma chi_b||_2^2 <= b^(2-2|gamma|) d_gamma1 d_gamma2`.

Since `|gamma|<=6` and b<=1, the exponent is at least -10 and its factor is at most `b^-10`. There is no missing power of b or extra dimension factor.

The H6 norm is explicitly the unitary Fourier norm with weight `(1+|xi|^2)^6`. Expanding `(1+xi1^2+xi2^2)^6` and applying Plancherel gives the exact multinomial weights w_alpha in (9), not an untracked norm equivalence.

For a localized derivative the pointwise weighted Cauchy inequality is

`|sum_gamma c_gamma X_gamma|^2 <= (sum_gamma c_gamma) sum_gamma c_gamma |X_gamma|^2`,

with `c_gamma=binom(alpha,gamma)` and sum `2^|alpha|`. Using the variance bound for each field derivative and integrating proves (10). Correlation between the terms is allowed. Omitting that weight sum by pretending the derivatives were independent would be an error; the proposal does not do so. My separate maximally correlated summand checks retain this negative control.

I independently recomputed A_theta by a different aggregation. Summing first over `alpha-gamma`, the remaining multinomial contribution is `(1+2+2)^(6-|gamma|)=5^(6-|gamma|)`. Thus the author's nested sum is exactly

`sum_(|gamma|<=6) [6!/(gamma1! gamma2! (6-|gamma|)!)] 2^|gamma| 5^(6-|gamma|) d_gamma1 d_gamma2`.

This gives the same integer `430299201784277045424464881586844996`. My cutoff coefficients were obtained from the closed alternating factorial/binomial formula, and the reciprocal constants from a truncated exponential generating-series inverse; they match all Aj,Ej,Gj,dj values. These are algebraic cross-checks of the displayed constants, not sampled derivative maxima.

It follows that `T=(1+M12)b^-10 A_theta` uniformly bounds `E||f_g||_H6^2` for every deterministic center g. Its very large magnitude is honestly retained. Nothing in the proof establishes a useful finite numerical window.

## 4. The finite PSD Gaussian MGF is genuinely dimension-free

For a fixed deterministic g, the localized field is a finite real linear combination of deterministic compact smooth functions. Its H6 Gram matrix Qg is real symmetric PSD. Under the unconditioned standard coefficient vector,

`S_g^2=Z^T Qg Z`, `tr Qg=E S_g^2<=T`.

A finite orthogonal diagonalization preserves the independent standard Gaussian law. Each eigenvalue is nonnegative and at most T, so the one-dimensional Gaussian integrals at exponent `1/(4T)` are finite and give exactly the displayed product. Setting `a_j=lambda_j/(2T)`, all are nonnegative and their sum is at most 1/2. Induction yields

`product_j(1-a_j)>=1-sum_j a_j>=1/2`.

Therefore `E exp(S_g^2/(4T))<=sqrt(2)`, with no rank or dimension factor. Zero eigenvalues are harmless. A rank-one matrix with eigenvalue T attains the product bound, so the constant is valid at the extreme trace allocation, not merely for many equal small eigenvalues. My 2,001 rational spectra include unequal allocations, zeros, changing dimension and the rank-one extreme.

PSD is essential to this deduction: a hypothetical matrix with eigenvalues `3T,-2T` has trace T but its positive direction makes the proposed exponential moment diverge. The actual Qg cannot have this defect because it is a Gram matrix. The tail (14) now follows directly by Markov for every nonnegative v. No infinite-dimensional Gaussian result, covariance-rank floor or unrelated jet regression is imported.

The centered, unconditioned law matters. The document does not assert that the whole count theorem transfers to arbitrary regressions, and it correctly retains conditional mean/algebraic-exception boundaries.

## 5. The quantitative H6-to-C4 operator constants are correct

With the displayed unitary transform in dimension two, inversion contributes `(2pi)^-1`. For any collection of unit derivative slots, its Fourier multiplier has magnitude at most `|xi|^j`. Cauchy–Schwarz against the exact H6 weight therefore controls the full multilinear operator norm, simultaneously in all locations and directions. This does not require the original field's covariance to be rotationally invariant.

Changing variables `u=r^2`, then `v=u/(1+u)`, gives

`I_j/pi=integral_0^1 v^j(1-v)^(4-j) dv=j!(4-j)!/5!`, for j<=4.

In particular I3=pi/20 and I4=pi/5. Combining with the unitary inversion factor gives squared constants `1/(80pi)` and `1/(20pi)`. The rational majorants `p3=1/15`, `p4=2/15` already follow from pi>3. My separate polynomial integrations of the beta integral produce all five displayed values. Absolute convergence also supplies the asserted continuity; for these compact smooth functions classical differentiation is already available.

Where the cutoff equals one, the derivatives of f_g are exactly the physical derivatives of F. The operator estimate controls arbitrary rotated eigenframes directly; no hidden coordinate-partial inflation or extra K/L factor appears. Orders through six in the trace are sufficient for this C4 estimate, so no unmentioned uniform higher spectral moment is needed.

## 6. The deterministic physical cover supplies the required domains

For each L the grid uses `n=ceil(L/b)`, giving spacing at most b and nearest-center coordinate distance at most b/2 in short torus lifts. Since b<=1,

`n<=(L+b)/b<=(L+1)/b`.

This proves the stated `n^2<=(1+L)^2/b^2` bound. The grid depends on deterministic family parameters, not on the field draw.

A rotated radius-b square about its assigned point M lies within coordinate distance `b/2+sqrt(2)b<2b` of g. Thus the entire closed chart, not just its anchor or pair endpoints, lies in the plateau of chi_b. Its support side 6b is strictly less than L, so the translated support embeds even across a canonical torus seam. The chart square also lies below torus injectivity. The first pair segment, of length at most `rho eta<=rho b/4`, stays well inside the plateau. The center and endpoint short lifts can be chosen consistently at these scales.

My separate controls test rational rotations and a seam where subtracting canonical `[0,L)` coordinates would fail. The underlying all-orientation margin is the analytic bound `|cos|+|sin|<=sqrt(2)<3/2`, not inference from a sampled set of angles.

For an eta-cluster, the displacement between its maxima is at most `(1+rho)eta<=b/2`. If maxima coincide, both distinct saddle endpoints remain within rho eta of that anchor. These are precisely the displacements needed by the two incidence cases. No bound relative to the actual center separation is introduced.

## 7. Localization preserves the inherited isolation hypotheses

For the covering center of the first maximum, set `S=||chi_b F(g+.)||_H6`, `W=1+S`. The local operator bounds `B=p3 W`, `C=p4 W` hold on the whole radius-b chart and on the pair segment by the preceding two sections. This domain check is the substantive new bridge to the accepted deterministic chart.

Supposing `W^2<gamma/eta`, the first term of gamma in (18) ensures `r_1B<=a` using `eta<=eta0`. The second controls the entire endpoint-to-eigenframe cubic error, with `W<=W^2` handling the C term. The maximum's pair-transverse negative curvature supplies a hard eigenvalue at most -a, and its cubic-gap mark gives an oriented soft cubic at least `tau=6k_-`. These are the exact accepted mark-conversion and chart premises, without changing any old statement.

The unchanged radius algebra now gives `R>=kappa/W^2>(1+rho)eta`. Different maxima contradict maximum uniqueness; different candidates sharing the same maximum give three distinct critical points, contradicting the chart count. A shared saddle is included in the different-maxima case. Cross-role equality is incompatible with the physical Hessian types. The identical candidate is excluded. This proves (19) and preserves the older height-barrier/mixed-cubic falsifiers: no enclosing superlevel region or elder selection is inferred.

The first maximum and a suitable center can depend on the realized field. That does not justify taking an unconditioned norm law at a random maximum; the proposal does not do so. It only establishes that the event implies a large norm at at least one member of the predetermined finite grid. Ties in choosing a nearest center need no measurable tie-break for this existence-to-finite-union inclusion.

## 8. The unconditional probability and integrated-count estimates follow

For `eta<=min(eta0,gamma/4)`, the large-W condition gives `S_g^2>=gamma/(4eta)` at some deterministic grid center. Applying (14) separately to every center and taking a union bound gives

`P(cluster)<=sqrt(2)b^-2(1+L)^2 exp[-gamma/(16T eta)]`.

No independence of cubes is assumed. Every probability is under the same unconditioned field law, so no field-selected covariance, conditional mean or Palm/Rice weight is hidden in the argument.

The accepted fixed-degree critical-count result applies almost surely to each individual positive full-square law. No uniform lower spectral weight is required to say a proper polynomial zero set has measure zero under that law. The exceptional set may depend on the family member; an expectation inequality for each member with common deterministic constants does not require a common probability-one event over an uncountable family. This is sufficient for the asserted limit of expectations and does not provide a joint increasing-domain coupling.

Endpoint identity and physical types are preserved: candidates have no reflected-frame duplicates, and the ordered factorial sum excludes `p=q` while including shared endpoints. On the source-specific full-probability set, `D<=(16K^2)^4 1{D>0}`. The completed-space finite-configuration measurability argument is consumed at its accepted scope. Borel selectors bounded by one only reduce the count.

Consequently the explicit bound is

`E D(t,eta)<=Ccov 16^4 K^8 (1+L)^2 exp(-c/eta)`,

with `Ccov=sqrt(2)b^-2` and `c=gamma/(16T)>0`. These constants are independent of K,L,N under the fixed family/mark assumptions. The remaining K^8 is openly retained and handled by the new cutoff. There is no second Hessian determinant weight or spatial-area normalization inserted into this integrated endpoint count.

## 9. The K/L/t cancellation and radius condition are correctly separated

For `H(t)=8log K+2log(1+L)+4log(1/t)` and `eta=c/H`, the equality

`exp(-c/eta)=K^-8 (1+L)^-2 t^4`

is exact. Multiplying by the coarse count and cover powers leaves `Ccov 16^4 t^4`. The requested normalization therefore gives `O(t^(8/3))`.

H is positive since K>=2, L>0 and 0<t<1; moreover `H>=4log(1/t)` forces eta below the fixed eta_cap eventually. With fixed `Tgap,k_-`, every lifetime-band radius is at most `A t^(1/3)`, `A=(Tgap/k_-)^(1/3)`. The additional condition `t^(1/3)H(t)->0` then ensures `A t^(1/3)<=rho c/H(t)` eventually. Only at this point may the radius-restricted D be said to include every near-center candidate pair in the stated lifetime band. Without that condition, the bound for D with its explicit radius restriction remains true, but full-band inclusion is not established.

Polynomial growth of K,L implies `H=O(log(1/t))`, so it satisfies the condition. In the example `L=t^-1/3`, `K=ceil L`, one has K>=2 and L>=Lmin eventually, K/L<=2, and `H=(22/3)log(1/t)+O(1)`. Thus (U) holds for arbitrary positive symmetric normalized weights and the lifetime-radius ratio tends to zero. This is a valid example of the new diagonal estimate, without establishing a constant-total-intensity regime.

The generalized coefficients p>=8, q>=2 and u>4/3 also work with their corresponding H: the residual K and `(1+L)` powers are nonpositive, and the normalized t exponent is positive. The corresponding radius condition must still be imposed.

The fast-growth falsifier is correctly distinguished from moment failure. At fixed L, normalized positive weights proportional to `exp(-|n|^2)` on each finite square give a uniform physical M12, because their normalizer is bounded below by the constant-mode contribution and the full lattice moment sum converges. Hence (U) can coexist with `log K` of order `t^-1/2`. The selected eta is then at most of order `t^1/2`, and the generic lifetime radius upper bound `t^1/3` is not o(eta). This does not disprove the radius-restricted estimate or assert that every actual pair attains the upper bound; it prevents claiming automatic inclusion from that bound.

If Tgap grows, the condition must contain `Tgap(t)^(1/3)`. The example `Tgap=1/t` makes the radius upper bound constant, so it cannot be automatically included in a cutoff tending to zero. Moving upper birth/cubic marks do not harm this proof because only fixed positive lower marks and the lifetime cap are used. Vanishing lower curvature/cubic marks remove the fixed positive gamma and are outside the claim.

Optional division by L^2 or L^4 is bounded above by an Lmin-dependent constant and would change the prefactor, not the t exponent. The displayed count is total and contains no such division. Its cumulative normalization proves neither the one-point `t^(2/3)` law nor density `ell^(-1/3)`.

## 10. Disposition and retained limits

There are **no blocking findings** within the frozen scope. The following boundaries remain necessary:

- Uniform total spectral mass and physical M12, with fixed positive Lmin, are real family assumptions. Finite smoothness, a cutoff degree alone, or the word Gaussian does not verify them for a source sequence.
- Gaussianity, centeredness and deterministic spectral parameters underlie the Gram MGF. The proof does not cover non-Gaussian coefficients or adaptive conditioning of the whole count theorem.
- Full positive square support is used by the inherited almost-sure count theorem; it is not needed by the local trace estimate alone. No infinite-carrier polynomial count cap is supplied.
- The cover and operator estimates control the actual chart and segment. They do not license a pointwise derivative bound without a surrounding norm domain, or a Gaussian law conditioned at a random maximum.
- Ordered endpoint identity, short displacements, positive a and `k_-`, the distinct-candidate exclusion and the radius cutoff remain part of D. A full square count including the identical-pair diagonal has a different scale.
- Growing K,L require the explicit radius condition to claim the whole lifetime sector. Growing Tgap changes that condition. The very loose constants do not certify a useful experimental window.
- A uniform integrated diagonal estimate does not supply moving far-center bounds, one-point intensity/remainders, global H0 bar correspondence, essential classes, higher homology, mixing, increasing-domain tightness, an infinite-field statement, or a limiting point process. Existing K2/K3 enlarged height-covariance singularities remain unchanged.

My 2,132 exact finite cases comprise eight independent cutoff/reindexed-trace checks, 28 H6 multinomial coefficients, seven beta-integral/operator checks, 2,001 unequal/rank-deficient PSD product cases, 28 maximally correlated Leibniz cases, 42 lower-moment inequalities, 12 cover/rotation parameter cases, one periodic seam and five growth/normalization identities. Negative controls retain the failures from dropping PSD, dropping the correlation-safe weight sum, dropping mass or uniform M12, excessive log-degree growth, and a growing lifetime cap. No fields, numerical Sobolev integrals, approximate critical roots, or process experiments were sampled.

All new author files, previous author/review files, pinned source and saved primary source retain their identities. The new manifest and integrity receipt bind this standalone review and its own artifacts. No production/native/remote claim, source adoption, Lean execution, kernel audit, human/provider-distinct approval or scientific promotion occurred. Actual separate OpenAI/Codex nonauthor, source-exposed, organizational-independence credit 0.
