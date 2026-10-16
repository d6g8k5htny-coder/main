# Nonauthor review of the fixed finite Fourier count and diagonal route

**Verdict: ACCEPT_SCOPED.** I found no blocking mathematical error in the frozen proposal's almost-sure bound `Ncrit<=16K^2`, fresh outer-cutoff isolation lemma, integrated ordered factorial-count bound, or logarithmic-cutoff normalization. No amendment is required for equations (1)–(21) with the stated fixed-law, endpoint-identity, type, mark, cutoff and embedding hypotheses. This is a local conventional mathematical review, not source adoption or scientific promotion.

Actual reviewer: OpenAI/Codex `/root/c261a_local_runtime_20261016`, 16 October 2026 UTC. Root and `/root/core_theorem_audit_20261015` are the authors; I did not author their sketch, the frozen extension or its controls. I read all three new files in full, checked the cited primary imports, and consumed the exact previously accepted isolation result with its hypotheses unchanged. I did not rerun the old isolation controls or claim another first review of the unchanged chart. Source exposure is explicit; **organizational-independence credit is 0**. An attempted separate subagent dispatch returned `agent thread limit reached`; no additional agent participated in this finite-count review.

## 1. Frozen inputs, primary imports and actual checks

| New frozen file | Bytes | SHA-256 |
|---|---:|---|
| `FINITE_FOURIER_CRITICAL_COUNT_DIAGONAL_ROUTE.md` | 25,699 | `621cee96854aa440498cecf5452a8d1e23f90a4458779c3286c18eba106fe138` |
| `finite_fourier_count_controls.py` | 5,153 | `fd6f29739280a55017945526cb75fb46ebc80d86c012ff4a1ff65baa404c31de` |
| `FINITE_FOURIER_COUNT_CONTROLS.json` | 849 | `bbe147fdf409a3e37b450d4cbcd5d2a927140f95bd3ce0f36cafe16861123bfd` |

The consumed isolation proposal/control/output retain SHA-256 `827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30`, `13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5`, and `ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf`. My earlier `NONAUTHOR_FOLD_ISOLATION_REVIEW.md` remains 18,409 bytes, SHA-256 `f06c61e3f3e60c795a1a14e824640f1689d4143dba4f6698cada7a41929eec90`.

The pinned finite source remains Main `85b928b326ff0586715a56ae91f51d3fbd89557a`, `experiments/universality/finite_gaussian_contact/PROOF.md`, 25,367 bytes, SHA-256 `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`, Git blob `5f345201309cfa1781e92d6292f542de4bcb2999`. Its actual real finite coefficient law, rather than a complex-generic unrelated law, is used below.

I verified the saved primary `SOURCE_IDENTITIES.json` at 796 bytes / SHA-256 `2c42d92a4732f09a58aebd9b9ef8672b51b672ab99f3bbdad61681a0dcf4f504` and all three listed source byte identities. I directly read the complete six-page Guth lecture and the full two named Stacks lemmas/proofs. Guth's Theorem 0.1 is the coprime affine-polynomial inequality over a field, sufficient here over C; it does not require irreducibility. [Guth, MIT lecture 13](https://math.mit.edu/~lguth/PolyMethod/lect13.pdf)

The projective maps below use that a projective source is proper, and a map from a proper source to a separated target has closed image. The cited tags supply precisely these implications. [Stacks 01WC](https://stacks.math.columbia.edu/tag/01WC), [Stacks 01W6](https://stacks.math.columbia.edu/tag/01W6)

I also directly read the MIT text's Section 7.8.1 and following reducible-form note/proof on PDF pages 178–179, and Larson–Neeman–Shelah Section 1 through the analytic-set universal-measurability conclusion on printed pages 174–175. I do not claim to have read the entire 216-page text or the entire measure-theory paper, or to have saved byte identities for those two live-browsed PDFs. [MIT algebraic geometry text](https://math.mit.edu/classes/18.721/ag-may31-2021.pdf), [Larson–Neeman–Shelah](https://shelah.logic.at/files/94751/947.pdf)

At 2026-10-16T01:59:41Z I ran the new author controls and my separate `nonauthor-finite-count-checks/review_checks.py`, using `/Library/Developer/CommandLineTools/usr/bin/python3 -B` (Python 3.9.6). Both exited 0 with empty stderr. The author stdout matches the frozen 849-byte output exactly. My 226 exact-rational finite cases passed. Full commands, timestamps, wall times, stdout/stderr and digests are in `nonauthor-finite-count-checks/CHECK_RUN_SUMMARY.json` and its two process-record sets. Twelve consumed input identities were verified before and after, unchanged. The exact finite tests do not mechanically prove genericity, Bézout, all-K coprimeness or the analytic cutoff lemma; the following arguments are my conventional review.

## 2. The full-degree witness belongs to the actual real family

The passage to Laurent variables is correctly normalized. With `z=e^(i omega x)` and `w=e^(i omega y)`, the two real gradient coordinates vanish exactly when the two Laurent sums with multipliers m,n vanish: their omitted factor `i omega` is nonzero. Clearing by `z^K w^K` changes no zero on `(C*)^2`. Each resulting polynomial has exponents between 0 and 2K in each variable, so total degree is at most 4K. The torus map to the unit-circle product is injective modulo the original period L; there is no K-fold covering factor in the critical-point count.

The witness consists of the axial K-modes and diagonal `(K,K)` mode, together with their negative-frequency conjugates, all within the full square. For any fixed strictly positive q, the real sine/cosine Gaussian coordinates map invertibly onto all real trigonometric coefficients. Set the three relevant sine coordinates to zero and choose the cosine amplitudes to give `1,1,c`; the other coordinates can be zero. This gives a permitted real parameter point with `a_-n=conjugate(a_n)`. It need not have positive probability as one exact draw: its role is to certify a nonzero polynomial on that same real parameter space.

Computing the actual Laurent derivatives gives the displayed normalized pair

`P_*=W0(Z0^2-1)+c(Z0^2W0^2-1)`,

`Q_*=Z0(W0^2-1)+c(Z0^2W0^2-1)`,

and `P_*-Q_*=(Z0-W0)(Z0W0+1)`. The normalization is `2/K`, so both top terms really have coefficient c and total degree 4K when `c!=0`.

Checking coprimeness only in `C[Z0,W0]` would not be the stated proof after substitution. The proposal instead checks the actual factors in `C[z,w]`, which is adequate. Every common irreducible divisor divides the difference and hence is one of `z-zeta w` with `zeta^K=1`, or `zw-rho0` with `rho0^K=-1`, up to a unit. The latter are irreducible because their quotient identifies z with `rho0/w` and is a Laurent polynomial domain. There is no hidden coordinate-axis factor.

On every line factor the restriction is the same nonzero polynomial `W0^3-W0+c(W0^4-1)`, whose top coefficient c is nonzero. On every hyperbola factor the restriction is `W0^(-1)-W0`, nonzero in its Laurent quotient. Thus no member of either family divides `P_*`, for every K and every nonzero real c. This is a valid all-K factor argument, not an inference from the finite K tests.

Both affine forms have exact degree D=4K. Their degree-D homogenizations therefore have no factor T0. Any other common homogeneous irreducible would dehomogenize to a nonconstant common affine factor; a homogeneous polynomial whose dehomogenization is a nonzero constant can only be a power of T0. The witness homogenizations are consequently coprime. Their equal restrictions at infinity give intersection points there, not a shared curve or line at infinity.

The failed `c=0` witness is retained correctly. It has affine coprime derivatives but degree 3K, so homogenizing both to the fixed degree 4K inserts a common `T0^K`. It cannot certify properness of the selected fixed-degree bad cone. The diagonal mode supplies exactly the missing degree. No change to this part is required.

## 3. Closedness, real restriction and the null exceptional set are justified

For each `1<=d<=D`, the map from `P(V_d) x P(V_(D-d) direct_sum V_(D-d))` sends `[H],[A,B]` to `[HA,HB]`. Joint projectivization of A,B is the right parameter space: it includes pairs with exactly one zero form and allows their relative scale. The coordinates transform by the same product of the two source scalars. They cannot all vanish, since H is nonzero, at least one of A,B is nonzero, and the polynomial ring has no zero divisors. The map is everywhere a morphism defined by sections of `O(1,1)`; there is no unresolved rational-map base locus.

Its source is a product of projective spaces, hence projective/proper over C. Its target is projective, hence separated. The named closed-image import therefore applies to each actual map. The finite union over d captures exactly the nonzero homogeneous pairs having a positive-degree common factor: the common factor can be chosen homogeneous and the quotients have degree D-d. The case of one zero form is included, for example by choosing the other whole form when d=D. The affine cone, with the zero pair included, is an algebraic closed set.

The witness shows this cone is proper **on the image of the actual real parameter family**, not merely proper in the ambient complex form space. In particular some defining complex polynomial is nonzero on the witness image. Pulling it back through the real-linear homogenization map gives a complex-coefficient polynomial in the real Gaussian coordinates. At least one real/imaginary part is a nonzero real polynomial, since its value at the real witness is nonzero. The exceptional preimage is contained in that real polynomial zero set.

The induction/Fubini proof that a nonzero real polynomial has a Lebesgue-null zero set is sound. The fixed positive weights make the real coefficient distribution nondegenerate; multiplying independent standard Gaussian coordinates by fixed nonzero scales preserves a full real density. The constant coefficient lies in the gradient map's kernel, but does not obstruct the argument: the pullback remains a nonzero polynomial in the full parameter space, possibly independent of that coordinate. Hence the exceptional set has probability zero under the actual Gaussian law.

It is harmless that the chosen exceptional set also removes some lower-degree affine-coprime pairs. Only a sufficient null exceptional set is needed. No unsupported conclusion about the smallest discriminant or global Morse regularity is being used.

## 4. Bézout counts distinct real torus critical points without a Morse premise

Off that one null set, the affine P,Q are coprime over C. The affine Bézout inequality bounds all common complex affine zeros by `(deg P)(deg Q)<=16K^2`. Every real torus critical point maps to a distinct such zero with z,w nonzero. Extra complex zeros and coordinate-axis zeros only enlarge this upper bound. Degenerate isolated roots are still distinct zeros and each consumes at least one intersection in the inequality; transversality and a Morse assumption are unnecessary.

This establishes the almost-sure deterministic bound on all distinct critical points for the fixed K law. It is not claimed for every coefficient vector. A one-variable field may have critical circles and lie in the exceptional family. Setting the later count to zero on that null exceptional set is an allowed measurable modification under the completed Gaussian measure.

The proof's full-real-density step is unconditional. This almost-sure argument cannot automatically be transported to an arbitrary zero-target conditioned coefficient subspace, which might lie in the algebraic exceptional set. The previous isolation review's centered conditional norm tail alone does not supply a conditional version of this global critical-point theorem. No such conditioned count claim occurs here, so this is a retained boundary rather than a blocker.

## 5. Candidate identity, finite count and measurability are adequate

Each candidate is an ordered pair of actual endpoints `(M,S)` with negative-definite Hessian at M and negative determinant at S. Its displacement is the unique short geodesic displacement; r,u, midpoint, gap, k and the fixed `v=Ju` follow from the endpoints and field. A reflected v is not a new candidate. There is no continuously free frame label being counted. Reverse endpoint order fails the maximum/saddle roles and positive gap. This preserves the endpoint-pair identity required for `B_K^4`.

On the full-probability finite set, the number of endpoint candidates is no more than `Ncrit^2`, and the number of ordered pairs of different candidates is no more than `Ncrit^4`. The sharper bounds `Ncrit(Ncrit-1)` and `C(C-1)` are unnecessary. Counting once by endpoints, rather than by frame presentations, makes the displayed coarse bound valid. An unordered convention would divide by two; no such division is hidden in D.

The identical candidate `p=q` is explicitly excluded. Distinct candidates can share a maximum or saddle. Cross-role equality is impossible because the physical Hessian at the same point cannot be negative definite and have negative determinant. This is a pointwise type fact in the same field, not an independence statement.

The count is measurable on the completed Gaussian space. For each integer m, `D>=m` can be expressed by m distinct ordered endpoint-pair configurations satisfying Borel equalities/inequalities in the finite coefficient vector and position variables. Short displacement/midpoint conventions are Borel on their specified domain. Projection gives an analytic event, which is measurable for the completed probability law. The exceptional set is Borel and null, so modifying there is legitimate. A nonnegative Borel weighted count with weights at most one is pointwise dominated by D. Its measurability can also be obtained by the same finite-configuration projection argument, using existence of finitely many distinct configurations whose weights sum above a rational threshold. No joint Rice density is needed for either step.

## 6. The fresh outer-cutoff lemma closes the actual-separation gap

The proposal correctly identifies that `r_i<=rho eta` is weaker than `r_i<=rho s` when the realized center distance s can be far smaller than eta. Its explicit old-ratio counterexample is valid. The new Section 6 is a new argument; it does not claim that the old hypothesis follows from short lifetimes.

The common-lift geometry is sufficient. With `eta<=eta0<=R0/4` and `R0<=L/16`, choose the shortest center displacement and the two short endpoint displacements. The endpoints lie within distance at most `(1+rho/2)eta` of the first center, and their displacements from the first maximum are bounded by `(1+rho)eta`. These scales are far below torus injectivity. The rotated chart itself embeds since its square lies in a ball of radius `sqrt(2)R0<L/2`. Thus a periodic seam does not change the Euclidean comparisons, even when the centers coincide. My separate controls explicitly test a seam where subtraction of canonical `[0,L)` coordinates would be wrong.

Assume `W^2<c_*/eta`. The first term defining `c_*` gives `r_1B<a` using `eta<=eta0`; the second gives the full endpoint-to-eigenframe cubic error below `6k_-`, using `W<=W^2`. The first candidate is a nondegenerate maximum with a hard eigenvalue at most `-a` and an oriented soft cubic at least `6k_-`. These are exactly the consumed hypotheses of the accepted chart. Its unchanged lower radius estimate gives

`R>=kappa/W^2>kappa eta/c_*>=(1+rho)eta`.

If the maxima differ, the triangle inequality places the second maximum inside that chart, contradicting unique-maximum isolation. This includes the shared-saddle case. If they share their maximum, distinct endpoint identity forces two different saddles, each within `rho eta<R` of the anchor. These are three distinct critical points in one chart, contradicting its full at-most-two conclusion. Merely knowing maximum uniqueness would not establish this second incidence case; the proposal correctly uses the stronger chart statement.

The identical-pair diagonal cannot be excluded this way and remains omitted from D. No division by s is used, so `s=0` and `s<<r_i` cause no gap. This proves `W^2>=c_*/eta` under the new bounding-cutoff hypothesis with the same displayed constants.

## 7. The expected count bound is a valid unconditioned integrated estimate

For `eta<=min(eta0,c_*/4)`, any nonzero D implies `||Z||^2>=c_*/(4eta)`. The finite standard Gaussian exponential moment is `2^(N/2)` at parameter 1/4, so the displayed norm-tail estimate follows. This is a global deterministic inclusion for the random set's existence event, not Gaussian conditioning at the field-selected sites. No adaptive-conditioning assumption, covariance inverse, density normalization or Hessian/Rice weight is hidden in it.

Almost surely,

`D<=B_K^4 1{D>0}`.

Taking expectations therefore gives exactly

`E D(t,eta)<= (16K^2)^4 2^(N/2) exp[-c_*/(16eta)]`.

This is the new finite combinatorial input absent from an occurrence estimate by itself. It works because K and the carrier are fixed and the endpoint candidate population has the almost-sure uniform finite cap. It bounds an integrated ordered factorial count, not a pointwise four-site Rice kernel. The fixed finite-support height-covariance obstructions remain untouched. Borel selectors in `[0,1]` can reduce this count, but no actual selected-bar identity is supplied by that domination.

## 8. Lifetime cutoff, explicit threshold and normalization are correct

For a candidate with `ell<=Tt` and `k>=k_-`, the literal identity `ell=kr^3` gives `r<=A t^(1/3)`, `A=(T/k_-)^(1/3)`. The comparison needed is to the outer cutoff `rho eta(t)`, not to s. For `n=log(1/t)` and `eta(t)=c0/n`, that comparison is `A n exp(-n/3)<=rho c0`.

The explicit threshold (21) is sufficient. The elementary bound `n<=6 exp(n/6)` gives `n exp(-n/3)<=6 exp(-n/6)`. If `6A/(rho c0)>1`, the last term of n0 makes its right side times A at most `rho c0`; if that ratio is at most one, the same conclusion follows already from `6A<=rho c0`. The other threshold terms ensure `n>=1`, `eta<=eta0`, and `eta<=c_*/4`. Equality in the non-strict geometric bounds is allowed; the contradiction radius remains strict. Thus for all `0<t<=exp(-n0)`, all candidates in the lifetime band satisfy the outer radius cutoff.

For `c0=c_*/64`, the tail exponent is exactly `4 log(1/t)`, giving `t^4`. Multiplication by the requested twofold cumulative normalization gives `t^(4-4/3)=t^(8/3)->0`. More generally `0<c0<3c_*/64` is exactly sufficient to make the exponent exceed 4/3. The claimed big-O constants and the threshold are deterministic for the fixed law and mark parameters. An optional division by fixed torus area once or twice changes only the prefactor under the stated spatial convention; D as defined contains neither division.

This proves neither a one-point cumulative asymptotic nor a pointwise lifetime density. The document preserves the distinction between cumulative `t^(2/3)` and density `ell^(-1/3)` and uses `t^(-4/3)` only as the twofold cumulative scale being tested.

## 9. Disposition, checks and retained boundaries

There are **no blocking findings** within the frozen scope. The following conditions must remain attached to reuse:

- Fixed finite full-square support with strictly positive symmetric weights is essential to this real-parameter genericity argument. The real witness and fixed-degree correction cannot be replaced by an unrelated complex-generic claim or the failed degree-3K witness.
- The count is an almost-sure fixed-K result. No all-coefficient-vector, arbitrary conditioned-law, unbounded-K, changing-support or infinite-Fourier critical-count theorem is established here.
- The count is by endpoint identity. Continuous frame labels, reflected duplicates, full square counts including `p=q`, and other presentation multiplicities require a different bound or explicit correction.
- The new cutoff lemma uses positive a and `k_-`, both physical endpoint types, short embedded displacements, and the stated c_* conditions. It covers shared maxima through three-critical-point isolation. It is not the old actual-center-ratio lemma under a renamed hypothesis.
- No pointwise Rice density, enlarged covariance-rank assertion, determinant-weighted conditioning estimate, or global selected-bar correspondence follows from the integrated count.
- Moving far-center cutoffs, small cubic/transverse marks, other lifetime/collision sectors, expanding-domain uniformity, essential classes, global H0 decomposition, higher homology, the infinite-carrier separated-center theorem and a full point-process limit remain open here. A fixed-separation theorem whose constant may diverge cannot silently supply the moving far-center complement.

My separate 226 finite cases include 36 lifted witness/factor/full-degree/real-parameter controls, 24 failed lower-degree homogenization controls, 36 ordered-factorial/shared-role counting cases, 12 cutoff geometry cases with coincident or very close centers, 48 rational arbitrary-orientation cases, one shared-maximum incidence, one periodic-seam control, 64 c_*/radius contradiction cases and four literal logarithmic exponent checks. They use exact Fraction arithmetic, no sampled Gaussian fields, no approximate root counts, no gcd oracle and no Rice integrals. The conventional proof review above, rather than those finite cases alone, supports the verdict.

All new author files, all three accepted isolation files, the previous review, the pinned finite source and the named saved primary sources retain their original identities. The new review manifest and integrity receipt bind only this review and its own check artifacts. No source adoption, production edit, remote write, Lean/formal execution, kernel audit, human/provider-distinct approval or scientific-status change occurred. Actual OpenAI/Codex nonauthor review; source-exposed; organizational-independence credit 0.
