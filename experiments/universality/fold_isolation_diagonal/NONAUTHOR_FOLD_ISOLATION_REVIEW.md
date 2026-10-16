# Nonauthor review of quantitative fold isolation

**Verdict: ACCEPT_SCOPED.** I found no blocking error in the frozen deterministic isolation statement, its literal mark conversion, or its fixed-law unconditional occurrence bounds. No source amendment is required for those claims as written. The zero-target conditioned tail stated in this review is a separately checked extension; the frozen proposal itself states the unconditional result. Nonzero targets, adaptive conditioning, weighted counts, global persistence pairing and process conclusions do not inherit this verdict.

Actual reviewer: OpenAI/Codex `/root/c261a_local_runtime_20261016`, 16 October 2026 UTC. I did not author the frozen proposal, its control program, or Root's sketch. I read the complete frozen proposal, program and output before reviewing, and the full pinned finite-Gaussian source. This is a source-exposed conventional mathematical review in the same OpenAI parent-thread lineage. **Organizational-independence credit: 0.** Root is the sketch author; this report is the actual separate nonauthor read, not a relabeling of Root's or the proposal author's work.

I also delegated an isolated Gaussian/conditioning cross-check to OpenAI/Codex `/root/c261a_local_runtime_20261016/gaussian_tail_crosscheck`, then read its complete note and checked its conclusions against my own derivation. It supplies another technical check and no provider or organizational independence.

## Exact reviewed object and retained execution

| Frozen file | Bytes | SHA-256 |
|---|---:|---|
| `QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md` | 19,138 | `827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30` |
| `fold_isolation_controls.py` | 6,308 | `13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5` |
| `FOLD_ISOLATION_CONTROLS.json` | 795 | `ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf` |

The consumed `experiments/universality/finite_gaussian_contact/PROOF.md` is 25,367 bytes, SHA-256 `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`, Git blob `5f345201309cfa1781e92d6292f542de4bcb2999`, joined to the retained Main tree at `85b928b326ff0586715a56ae91f51d3fbd89557a`. I use its literal finite Fourier law and endpoint convention, rather than importing an isotropic covariance, an independent field copy, or arbitrary finite-jet independence. I did not rerun or re-review the earlier collinear theorem.

At 2026-10-16T01:48:33Z I ran both the author's exact control program and my separate `nonauthor-fold-isolation-checks/review_checks.py` with `/Library/Developer/CommandLineTools/usr/bin/python3 -B` (Python 3.9.6), from this task directory. Both exited 0 with empty stderr. The author control stdout matches all 795 frozen output bytes exactly. My 255 exact-rational finite cases passed. Commands, start/end times, exit codes, complete stdout/stderr and their digests are retained in `nonauthor-fold-isolation-checks/CHECK_RUN_SUMMARY.json` and the two `*.meta.json`, `*.stdout`, `*.stderr` sets. The three frozen inputs were verified before and after execution and were not changed.

These finite controls supplement the following analytic review. Their passage is not a mechanical proof of the analytic statements.

## 1. The deterministic chart survives the boundary and type attacks

Proposal lines 15–48 state an adequate domain: C4 on an open neighborhood of the closed square, a critical anchor, diagonal negative Hessian at the anchor, a negative hard pivot at most `-a`, and a nonzero anchored x-cubic. The argument actually needs no additional ordering of the two negative eigenvalues in this abstract chart. In the pair application the x-axis is separately identified with the soft eigenvector.

The multilinear operator norm is the correct norm for a rotated chart. Bounds on coordinate partials in one frame alone would not be invariant. The proposed conversions `2 sqrt(2) M3` and `4 M4` follow by expanding the multilinear tensor and applying the two-dimensional l1/l2 bound in every slot. The global Gaussian estimate below already controls the operator norm and needs no further conversion factor.

The radius is positive because `B>=tau>0`, `a,tau,R0>0`, and `C>=0`. With `BR<=a/32`, the actual hard-curvature estimate is even stronger than the displayed relaxation: `F_yy<=-a+2BR<=-15a/16<-a/2`. The bound `|F_y(x,0)|<=Bx^2/2` uses both `F_y(0)=0` and `F_xy(0)=0`. It supplies the strict opposing face margins `31aR/64`. Thus for every closed-interval x, including `x=+/-R`, the unique vertical zero lies strictly between the y-faces. It cannot escape through a corner or a horizontal face.

The implicit-function theorem applies locally at each point of the ridge, including its two endpoints, because the domain is an open neighborhood and `F_yy` is nonzero. The roots agree on overlaps by uniqueness. C3 regularity and the local endpoint continuations are sufficient for all displayed derivatives. Integrating the negative y-derivative gives `|h(x)|<=Bx^2/a<=R/32`, and `h'=-F_xy/F_yy` gives `|h'|<=4BR/a<=1/8` without assuming the ridge remains on `y=0`.

I independently differentiated the reduced function. Differentiating `f''=F_xx+F_xy h'` first gives

`f'''=F_xxx+2F_xxy h'+F_xyy(h')^2+F_xy h''`.

The second differentiated ridge equation is

`F_yy h''=-(F_xxy+2F_xyy h'+F_yyy(h')^2)`.

Using `F_xy=-F_yy h'` produces exactly the proposal's coefficients `1,3,3,1`. Thus no unbounded `h''` term was silently discarded. The operator fourth-derivative bound controls the variation of `F_xxx` by `C(|x|+|h|)<=2CR`. For `p<=1/8`, `3p+3p^2+p^3<=4p`; the remaining contribution is at most `16B^2R/a`. Finally `2C+16B^2/a<=16(C+B^2/a)` and the third radius term give the fixed-sign margin `tau/2`. This verifies the chain-rule and radius constants in (3)–(5).

Put `g=f'`. Strict convexity or concavity gives at most two zeros even on the closed x-interval: an interior third zero between two others contradicts the strict chord inequality. Nondegeneracy of every root does not follow from strict convexity alone. Here the anchor supplies the missing information: `g(0)=0` and `g'(0)<0`. If `g''>0`, a second zero must be to the right; Rolle gives a point with derivative zero strictly between the roots and strict increase makes the second derivative value `g'(x_2)>0`. If `g''<0`, the second zero must be to the left and the same argument with decreasing `g'` again gives `g'(x_2)>0`. This works when the second zero is on an x-face. A double second root cannot coexist with this anchored maximum.

At every critical point, `det Hess F=F_yy f''`. With `F_yy<0`, the second simple root has negative determinant and is an ambient index-one saddle. The negative-hard Schur sign is correct; there is no sign reversal turning it into a second maximum. The anchor has two negative eigenvalues. The language expressly excludes constrained extrema of the rectangle, so a boundary-constrained maximum is not misclassified as an ambient one.

My independent boundary controls take `a=tau=B=1`, `C=0`, `R0=1`, giving the proposed `R=1/32`, and `mu=1/64`. The cubic fold has its second critical point exactly at `x=R`, with positive reduced second derivative; changing the cubic sign places it exactly at `x=-R`. Both are saddles. This attacks the closed-square case directly, beyond the author's existing finite checks.

## 2. The height and mixed-cubic obstructions are preserved correctly

Integrating `F_yy<=-a/2` twice from the ridge gives the vertical estimate `F(x,y)<=f(x)-a(y-h(x))^2/4`. Since `|h|<=R/32`, the y-face drop is at least `aR^2/8`. It is relative to `f(x)`, not to `F(0)`. No enclosing superlevel barrier or path-trapping conclusion follows.

The false full-height-barrier example remains valid at the actual radius, not merely at an arbitrary test point. With the radius `R=1/32` above and `mu=1/1000`, the positive cubic example gives

`F(R,0)-F(0)=113/24576000>0`.

The proposal retains this obstruction and does not infer an elder relation from the vertical barrier.

The `D4F=0` mixed-cubic example also works exactly: substituting `h=bx^2/(2a)` produces `b^2x^4/(8a)`, hence `f'''=tau+3b^2x/a`. The reduced cubic can change sign despite a zero ambient fourth derivative. This explains the `B^2/a` term; it is not presented as two maxima inside the valid isolation square. My separate check confirms the sign change at `-2/49` for the author's rational parameters.

The two-maxima polynomial has cubic magnitudes `tau` at its maxima and fourth derivative magnitude `2tau/delta`. It is a valid obstruction to an isolation radius independent of derivative bounds. These deterministic polynomial controls are not claimed to occur with positive probability as exact samples of the finite Fourier law. All three negative boundaries must remain attached to reuse of this review.

## 3. Literal pair marks and arbitrary orientation are handled without losing factors

The source has `M=z-ru/2`, `S=z+ru/2`: r is the full endpoint distance. For a univariate restriction `p(s)=F(M+su)`, direct integration by parts gives

`integral_0^r [6s(r-s)/r^3] p'''(s) ds = (6/r^3){r[p'(0)+p'(r)]-2[p(r)-p(0)]}`.

At critical endpoints this is `12k`. The positive kernel has unit mass and mean `r/2`, proving the endpoint error `Cr/2`. My separate monomial-integration controls verify this identity before and after imposing endpoint criticality for degrees 3 through 9 and three rational radii. This avoids relying only on the normal-form example. The example's `k=1/12` is correct. The quartic with positive gap but zero endpoint cubic correctly saturates the `Cr/2` error and prevents dropping that error hypothesis.

The vector Taylor formula is also correctly normalized: dividing the gradient remainder by r gives `||Hu||<=Br/2`. At a nondegenerate maximum, the hard eigenvalue has magnitude at least a by the transverse Rayleigh quotient. The soft eigenvalue magnitude is at most `||Hu||<=a/2` under `rB<=a`; the eigenspaces are distinct. Projecting onto the hard eigenvector bounds its u-component by `Br/(2a)`.

With the soft unit vector oriented toward u, `||u-e||^2=2(1-sqrt(1-t^2))<=2t^2`, where `t=|u.v_h|`. The multilinear telescoping difference of the two cubic tensors is at most `3B||u-e||`, giving the sharper constant `(3sqrt(2)/2)B^2r/a`; the proposal's `3B^2r/a` is conservative. Therefore (12)–(13) really imply the anchored eigenframe cubic `F_eee(M)>=6k_-`. No pair-frame diagonalization, covariance isotropy, upper cubic mark, or saddle-type inference is smuggled into this implication.

## 4. Fixed-law norm radius and two-pair dichotomy check out

For the literal finite law, the two sine/cosine coefficient squares for each frequency pair add to `2q_n` times the squared product of directional frequency projections. Thus for arbitrary unit derivative slots the entire coefficient-row squared norm is

`sum_(n in S_K) q_n product_j (omega n.v_j)^2 <= beta_j^2`.

Cauchy–Schwarz is simultaneous in location and all slots, so taking their supremum proves the full multilinear operator bound. No extra coordinate factor or rotation-invariant spectrum is needed. The constant Gaussian coefficient does not affect derivatives but its inclusion in `||Z||` is harmlessly conservative. Positive nonzero modes in the pinned K>=2 law ensure `beta_3,beta_4>0`.

For `W>=1`, the first two radius terms dominate their respective constants divided by `W^2`, and `beta_4 W+beta_3^2W^2/a <=(beta_4+beta_3^2/a)W^2` handles the third. Hence `R>=kappa/W^2`. A second maximum at Euclidean distance less than R lies inside the first chart's square regardless of its own eigenframe. The embedded-scale assumptions avoid torus-lift ambiguity.

For two pairs let s be their actual center distance. The full-distance convention gives `|M_2-M_1|<=s+(r_1+r_2)/2<=(1+rho)s`. Under `W^2<c_*/s`, the first term of (18) gives `r_1B<a`; the second controls the entire cubic-error term using `W<=W^2`; and the third gives `R>(1+rho)s`. The other maximum is strictly inside the chart, a contradiction. This verifies all factors and the strict inequality used to prove `W^2>=c_*/s`.

For the cutoff event `s<=eta`, the radius hypothesis must remain `r_i<=rho s` for each realized configuration. It cannot be weakened silently to `r_i<=rho eta`. The proposal's paragraph after (20) uses the correct pointwise interpretation. Its positive constants depend on a, `k_-`, rho, `eta0` and the fixed spectrum through `beta_j`, kappa and N. No uniformity as a or `k_-` vanishes, or as K,L,spectra vary, is established. The bound can exceed one at the outer cutoff; it remains a valid bound and may simply be capped at one.

A nonblocking redundancy: with `rho<1`, `s>0` and `r_i<=rho s`, one already has `|M_2-M_1|>=(1-rho)s>0`; a shared maximum is geometrically impossible within this regime. Explicitly requiring distinct maxima is safe. Likewise the second pair's derivative/mark conditions are stronger than the contradiction needs; it only uses its other maximum. Neither redundancy invalidates the statement or warrants changing the frozen source.

## 5. Unconditional and zero-target conditioned Gaussian tails

The deterministic obstruction places the entire occurrence event inside one global coefficient-norm tail. For `t<=c/4`, where c is kappa or `c_*`, `W^2>=c/t` implies `||Z||>=sqrt(c/t)-1>=sqrt(c/t)/2`. Since `E exp(||Z||^2/4)=2^(N/2)`, Markov gives exactly

`P(E_t)<=2^(N/2) exp[-c/(16t)]`.

There is no union bound over locations or angles: the derivative and norm implications already hold simultaneously in all of them. The outer-probability formulation avoids an unstated measurability premise and is sufficient for the stated estimate. The finite-dimensional configuration event can also be exhausted by compact sets after bounding coefficients and separating all strict degeneracies, but no count or selected-bar measurability is being inferred here.

The requested zero-target extension is valid with the following precise law. Let A be any deterministic linear observation map, possibly depending on fixed sites, frames and scale, of rank q. Under the canonical Gaussian disintegration on the support of AZ, set `P=I-A^dagger A`, the orthogonal projection onto `ker A`. Then

`Z | AZ=0` has the law of `P G`, with `G` standard Gaussian in `R^N`,

`E[exp(||Z||^2/4) | AZ=0]=2^((N-q)/2)`.

This statement includes redundant or rank-deficient observations; the target zero always belongs to the support. It requires no observation covariance lower bound. The same pointwise deterministic obstruction therefore gives

`P_A^0(E_t)<=2^((N-q)/2) exp[-c/(16t)]`, for `t<=c/4`.

Keeping the original `2^(N/2)` prefactor is uniform over all deterministic A, even if its rank changes with geometry. Zero-gradient pins at fixed sites are an example. This is a canonical Gaussian regression statement, not conditioning on an equality event as if it had positive probability, and not an arbitrary regular-conditional version assigned at a null target. This extension is recorded under this reviewer's derivation and the Gaussian cross-check; it is not retroactively attributed to the frozen proposal.

For a supported nonzero target y the mean is `m=A^dagger y`, orthogonal to `ker A`, and the exact MGF gains `exp(||m||^2/4)`. Thus a valid estimate is

`P_A^y(E_t)<=2^((N-q)/2) exp[||m||^2/4-c/(16t)]`.

Uniform use needs a uniform bound on m. A bounded target alone is insufficient: `epsilon Z_1=1` forces `Z_1=1/epsilon`. The pinned source supplies mean control for its particular stable `U_r=v_r` observations on compact birth/cubic windows; it supplies no automatic arbitrary four-site-height extension. Adaptive field-selected locations, Palm/Rice weighting and nonsingular observation densities need separate arguments. None of these missing extensions is asserted by the present proposal, so they are boundaries rather than blockers to this verdict.

## 6. Precise disposition and limits

There are **no blocking findings** against equations (1)–(20) in their stated domain. The following are retained conditions or separate obligations, not waived defects:

- C4 bounds are needed on the actual chart and pair segment; operator norms and embedded lifts cannot be replaced by unverified coordinate or torus-distance conventions.
- The anchor is a nondegenerate maximum, the hard pivot is bounded away from zero, and the cubic lower bound is fixed. The second root classification uses the anchor slope, not strict convexity alone.
- Pair marks require the displayed `rB` and cubic-error controls or the full norm dichotomy. Positive k alone does not give a nonzero endpoint cubic.
- Occurrence probability is distinct from a factorial kernel or expectation. Rank-deficient conditional norm tails do not create ordinary pin densities, weighted determinant bounds, normalizer bounds or a finite-support counting theorem.
- The vertical barrier supplies no enclosing superlevel region, actual elder selection or global persistence pairing. The D4-zero mixed-cubic and full-height-barrier obstructions remain attached.
- Small a or cubic marks, pairs not short relative to their actual center distance, other collision geometries, global H0 decomposition, essential classes, higher homology, complement estimates, same-field expanding-domain control, process convergence and literal lifetime-density asymptotics remain outside this acceptance.

My independent finite cases comprise 96 mixed jet cancellations, 21 general kernel/endpoint identities, 2 closed-face saddle examples, 18 negative-hard Schur classifications, 2 retained obstruction checks, 16 exact anisotropic spectral row bounds, 96 inverse-W-squared radius inequalities and 4 singular centered-regression projector/MGF checks. They use Fraction arithmetic and no field sampling. No production or frozen-source edit, remote action, Lean build, kernel audit, formal theorem, human/provider-distinct approval, global bar result, point-process result or scientific acceptance occurred.

The Gaussian cross-check note is `nonauthor-fold-isolation-checks/gaussian-crosscheck/GAUSSIAN_OPERATOR_TAIL_CROSSCHECK_20261016T014803Z.md`, 13,955 bytes, SHA-256 `c06161e9a8b697b2cb6fd0c6ff597c2dcdff91f653d074c3e47564cada6f2f67`. The final integrity receipt and manifest bind this report, the check source, actual process records and that note, while keeping the three frozen author files unchanged.
