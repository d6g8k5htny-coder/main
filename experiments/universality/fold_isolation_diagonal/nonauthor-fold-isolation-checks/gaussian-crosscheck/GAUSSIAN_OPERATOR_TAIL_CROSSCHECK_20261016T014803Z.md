# Independent Gaussian operator and occurrence-tail cross-check

Reviewer: OpenAI/Codex `/root/c261a_local_runtime_20261016/gaussian_tail_crosscheck`, dispatched by OpenAI/Codex `/root/c261a_local_runtime_20261016`. Organizational-independence credit: **0**. Source fully exposed. Local conventional-mathematics review only. No production or remote writes, Lean result, global-bar result, process result, or scientific acceptance is asserted.

## Verdict and exact scope

**ACCEPT_SCOPED** for the finite coefficient/operator estimates, arbitrary-angle mark conversion, norm dichotomy, and unconditional occurrence tails in proposal Sections 4–5, conditional on the deterministic isolation chart in Section 2. I found no error requiring an amendment to those statements as presently scoped.

**ACCEPT_SCOPED** for extending those occurrence tails to the canonical Gaussian law under any fixed deterministic **zero-target linear conditioning**. This includes singular observation covariance, and needs no uniform positive covariance eigenvalue. The precise law and its limits are stated below; the proposal does not presently state this extension.

**AMEND before any expanded claim** that the same constants automatically cover nonzero targets without conditional-mean control, adaptive selection of the conditioning locations, ordinary densities for rank-deficient observations, or determinant/count-weighted expectations. Those assertions do not follow from the occurrence proof. These are boundaries of an extension, not defects in the present unconditional result.

I have not re-reviewed the old collinear calculation. I independently checked the Gaussian and arbitrary-angle parts against the actual finite-source formula.

## Sources read and frozen identities

The complete proposal was read: `QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md`, 19,138 bytes, SHA-256 `827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30`.

The complete bound source was read: `experiments/universality/finite_gaussian_contact/PROOF.md`, 25,367 bytes, SHA-256 `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`. This agrees with the identity quoted in the proposal. Its source law is lines 41–57; literal endpoint convention is lines 68–76; divided differences and target are lines 145–186; coefficient regression is lines 205–224; physical Hessian weights are lines 253–307; finite-support/rank boundaries are lines 425–477.

The three frozen proposal/control files were only read. Their identities observed before the note was written were:

- proposal: 19,138 bytes, SHA-256 `827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30`;
- `fold_isolation_controls.py`: 6,308 bytes, SHA-256 `13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5`;
- `FOLD_ISOLATION_CONTROLS.json`: 795 bytes, SHA-256 `ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf`.

## 1. Actual finite law and multilinear operator bound

The bound source, lines 41–57, has independent standard real coordinates

`Z=(C_0,(C_n,S_n)_(n in P_K)) in R^N`, `N=(2K+1)^2`,

and `sqrt(2 q_n)[C_n cos(omega n.x)+S_n sin(omega n.x)]` for each nonzero pair representative. Here `K>=2`, `omega=2pi/L`, `q_n=q_-n>0`, and `sum q_n=1`.

For arbitrary unit derivative slots `v_1,...,v_j`, set `p_n=product_l (omega n.v_l)`. The two row coefficients belonging to mode pair `n` have squared sum exactly `2 q_n p_n^2`, since differentiation only shifts the sine/cosine phase. Therefore the squared norm of the entire row is

`sum_(n in S_K) q_n product_l (omega n.v_l)^2 <= sum_(n in S_K) q_n (omega |n|)^(2j) = beta_j^2`.

Cauchy–Schwarz applies for **every** location and every collection of unit slots with the same bound. Taking the supremum subsequently proves the proposal's operator estimate (15), with no coordinate-to-operator conversion factor and no rotation-invariance assumption. Including the constant coefficient in `||Z||` is conservative. The positivity of `beta_3,beta_4` follows from the source's positive nonzero modes.

The simple frequency bound is `beta_j <= (2pi sqrt(2) K/L)^j`. Thus there is no hidden independence from `K,L`. The proposal correctly fixes the law; the prefactor also involves `N`. Its later claim does not use any lower covariance eigenvalue, support-rank theorem, or independent-copy replacement.

## 2. Endpoint normalization and arbitrary direction

The source's distance is `r=|S-M|`, with `M=z-r u/2`, `S=z+r u/2`, not a half-distance (source lines 68–76). At the pinned critical endpoints the fourth observation has target `12k` (lines 149–171). Changing from the midpoint variable to distance `s` measured from `M` gives exactly

`12k = integral_0^r [6s(r-s)/r^3] D^3F(M+s u)[u,u,u] ds`.

This positive kernel has mass one and first moment `r/2`, so an operator fourth-derivative bound gives `|F_uuu(M)-12k| <= Cr/2`. The endpoint estimate in (9) and the factor 12 are correct.

The vector Taylor formula in (10) gives `||H u|| <= Br/2`. Since `H` is negative definite and a pair-transverse Rayleigh quotient is at most `-a`, its hard eigenvalue has absolute value at least `a`. Its smallest singular value, hence its soft eigenvalue in absolute value, is at most `Br/2`. Under `rB<=a` the two eigenspaces are distinct. For the hard unit vector `v_h`, `t=|u.v_h|<=Br/(2a)<=1/2`. Choose the soft unit vector `e` with `e.u>=0`. Then

`||e-u||^2 = 2(1-sqrt(1-t^2)) = 2t^2/(1+sqrt(1-t^2)) <= 2t^2`.

A three-term multilinear telescoping expansion yields

`|D^3F[e,e,e]-D^3F[u,u,u]| <= 3B||e-u|| <= (3sqrt(2)/2) B^2r/a <= 3B^2r/a`.

Thus (11) is conservative, and (12)–(13) indeed imply `F_eee(M)>=6k_-`. No upper mark bound or assumption that the pair frame diagonalizes the Hessian entered this argument. The negative-definite anchor, the pair-segment derivative bound, and the two displayed derivative error conditions are essential.

## 3. Radius, norm dichotomy, and constants

With `W=1+||Z||>=1`, `B=beta_3 W`, `C=beta_4 W`, the radius terms satisfy

`R0 >= R0/W^2`,

`a/(32B) >= [a/(32beta_3)]/W^2`,

`tau/[32(C+B^2/a)] >= {tau/[32(beta_4+beta_3^2/a)]}/W^2`.

Hence (16), `R>=kappa/W^2`, is valid. If another maximum is at Euclidean distance less than `R`, each of its coordinates in the first maximum's eigenframe has absolute value less than `R`; it is in the chart's square. This step has no relative-angle restriction.

For the pair formulation, let the **actual** center distance be `s`. The triangle inequality gives `delta<=s+(r_1+r_2)/2<=(1+rho)s`. Assume `W^2<c_*/s`. Then

`r_1 B <=rho beta_3 s W <rho beta_3 sqrt(c_* s) <=rho beta_3 sqrt(c_* eta0) <=a`,

and, using `W<=W^2`,

`r_1(C/2+3B^2/a) <=rho s (beta_4 W/2+3beta_3^2W^2/a) <=rho s D0 W^2 <rho D0 c_* <=6k_-`.

The chart therefore applies at `M_1`, with

`R>=kappa/W^2>kappa s/c_*>=(1+rho)s>=delta`,

a contradiction. Each term of (18) has the stated role, and no missing factor of two occurs. The second pair's marks are stronger hypotheses than the contradiction needs; it suffices that its maximum is distinct. This harmless redundancy does not invalidate the statement.

For a cutoff event with `0<s<=eta`, the hypotheses must retain `r_i<=rho s` for each realized configuration. Replacing that with only `r_i<=rho eta` would change the event and would not establish the same dichotomy. The proposal's paragraph after (20) explicitly uses the correct pointwise interpretation.

All required lower thresholds are strictly positive for the fixed law and `a,k_->0`, `0<rho<1`, `eta0>0`. There is no claimed uniform positive lower bound as `a` or `k_-` vanishes. In particular `c_*` contains an `a^2` term and a `k_-` term; neither dependence may be discarded. The embedded-lift assumptions in the proposal keep the Euclidean comparisons legitimate on the torus.

## 4. Unconditional occurrence probability

Put `c=kappa` or `c=c_*` and let `t` denote the appropriate distance cutoff. The deterministic result gives `W^2>=c/t`. If `t<=c/4`, then `T=sqrt(c/t)>=2`, so

`||Z|| >=T-1>=T/2`, and therefore `||Z||^2>=c/(4t)`.

The exact finite Gaussian moment is

`E exp(||Z||^2/4)=2^(N/2)`.

Markov's inequality gives `P(E_t)<=2^(N/2) exp[-c/(16t)]`, precisely (17) or (20). The deterministic inclusion also establishes the outer-probability version if event measurability has not yet been proved. There is no counting factor here, and existence over arbitrary locations or angles requires no union bound because the norm obstruction is simultaneous in all of them.

## 5. Zero-target linear conditioning, including singular observations

Let `A:R^N -> R^m` be any deterministic real linear map, possibly dependent on fixed geometry or a deterministic scale, with rank `q`. Use the **canonical Gaussian conditional law on the support of `AZ`**. The zero target always lies in that support. If `P` is orthogonal projection onto `ker A` and `G~N(0,I_N)`, this conditional law is

`Z | {AZ=0} = P G` in distribution.

This statement uses the Gaussian disintegration, not a positive-probability interpretation of the equality event and not an arbitrary choice of a regular conditional version at a null target. It remains valid when `AA^T` is singular. The exact exponential moment is

`E[exp(||Z||^2/4) | AZ=0] =2^((N-q)/2) <=2^(N/2)`.

Every deterministic derivative and configuration implication above remains true on `ker A`. Consequently

`P(E_t | AZ=0) <=2^((N-q)/2) exp[-c/(16t)]`, `t<=c/4`.

This is uniform over all such deterministic zero-target maps if the original `2^(N/2)` prefactor is retained. Rank changes and small nonzero observation eigenvalues cause no failure of this tail statement. The conclusion also holds for conditional outer probability.

Zero gradient pins at fixed sites are an example. Adding a redundant zero observation does not require a new covariance lower bound for this tail alone. It still does **not** provide a nonsingular Lebesgue density of the observations, a valid Rice Jacobian, any normalizer, or a determinant/count-weighted estimate. The finite source explicitly reserves enlarged conditioning lists and finite-support rank control for those other uses (lines 425–477).

## 6. Nonzero targets and adaptive conditioning: limits of the extension

For a nonzero target `y`, first require `y in im A`; otherwise the canonical supported conditional law at that target is not defined. For a supported target let `m=A^dagger y`, the minimum-norm solution, so

`Z | {AZ=y} =m+P G`, with `m` orthogonal to `ker A`.

Thus

`E[exp(||Z||^2/4) | AZ=y] = exp(||m||^2/4) 2^((N-q)/2)`,

and the same deterministic obstruction yields the valid estimate

`P(E_t | AZ=y) <=2^((N-q)/2) exp[||m||^2/4-c/(16t)]`, `t<=c/4`.

A uniform bound on `||m||` therefore suffices for a uniform constant. Merely bounding the target or allowing singular conditioning does not suffice. For example, `A_epsilon z=epsilon z_1`, `y=1`, fixes `z_1=1/epsilon` and makes the norm arbitrarily large as `epsilon->0`. This is a counterexample to transferring the centered radial tail uniformly over bounded nonzero targets; it is not a constructed two-pair event in this finite Fourier law.

The bound source does supply mean control for its **specific** `U_r=v_r` conditioning when `r<=r_*` and `b,k` lie in a compact rectangle: source (7) has a uniform covariance lower bound and lines 211–224 use it. Explicitly,

`||m||^2=v_r^T (L_U L_U^T)^(-1) v_r <= ||v_r||^2/m_source`.

For `r<=1`, `|b|<=b_max`, `k<=k_+`, one may use `||v_r||^2<=2b_max^2+(291/2)k_+^2`. That source-bound case admits the shifted-mean estimate. It does not establish uniform mean control for arbitrary four-site height targets, unbounded birth/cubic marks, or enlarged rank-degenerate lists. Source lines 305–307 expressly limit its inverse-normalizer result too.

Finally, the observation map must be fixed before the field draw, or its conditional Gaussian law must be independently justified. Selecting critical sites or configurations from the realized field generally changes the conditioning law; it is not licensed just by writing an apparently linear observation at the selected sites. External random geometry independent of `Z` can be handled by first conditioning on that geometry. Field-dependent selection, Palm/Rice reweighting, and factorial counts require separate arguments.

## 7. Measurability

The proposal correctly offers outer probability and so has no measurability gap in its stated tail. For the usual literal configuration event here, a Borel formulation is also available. The field, its derivatives, endpoint height gaps, transverse curvatures, and Hessian eigenvalues are jointly continuous in `Z` and the finite-dimensional torus/frame/radius variables. Negative definiteness of an anchored critical point is precisely nondegenerate local maximality.

Use compact torus/frame parameter spaces and the stated short-lift domain. Exhaust strict conditions with `||Z||<=n`, radii and positive/distinct distances at least `1/n`, and maximum Hessian eigenvalues at most `-1/n`; an open patch condition can likewise be exhausted by compact subsets. All remaining equalities and weak inequalities are closed. Projection of each resulting compact configuration set to the finite coefficient space is compact. A countable union over the exhaustion and a finite/countable chart atlas gives a Borel event. Alternatively, its Borel configuration set has analytic projection and hence is universally measurable under the completed Gaussian law. Either route is enough; no count measurability or selected-bar measurability is inferred.

## Scope of this review

These are analytic, finite-dimensional source checks. No Gaussian draws, persistence computations, source edits, remote updates, Lean compilation, kernel audit, or new global mathematical admission occurred. The occurrence tail is valid within its fixed positive curvature/cubic windows and short-pair geometry. The organizational independence remains zero.
