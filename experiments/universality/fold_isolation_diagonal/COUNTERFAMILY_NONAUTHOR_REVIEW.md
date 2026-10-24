# Nonauthor review of the repeated-spectrum candidate counterfamily

**Verdict: ACCEPT_SCOPED for §§3–5 of the exact frozen obstruction proposal.** I found no blocking mathematical error in the loss of one-contact rank under upper moments alone, the stronger uniformly one-contact-nondegenerate repeated-spectrum family, its macroscopic two-contact degeneracy, the explicit empty-in-probability candidate process, or the separate deterministic diagonal construction preserving a finite positive first moment. The mean-preserving version is an existence construction of smaller positive fills; it is not established here for the explicit choice epsilon_N=N^-12. No source amendment is required within this review's scope.

This is a new conventional mathematical review by OpenAI/Codex `/root/c261a_local_runtime_20261016`, an actual separate nonauthor agent. The frozen proposal's authors are the core-theorem agent and Root steering. I read their source before reviewing it. This is source-exposed, same-provider OpenAI work with organizational-independence credit0, blind0 and human0. It is not a Lean/formal verification, scientific acceptance, point-process theorem for the project source, actual-global-bar counterexample, source admission, custody review, or authority transfer. The integrating reviewer separately owns §§6–9; I make no verdict on those sections here.

## 1. Exact object, consumed inputs and execution

The reviewed object is `JET_NONDEGENERACY_FACTORIZATION_OBSTRUCTION.md`, 34627 UTF-8 bytes, SHA-256 `0dfdddfbcac47b2fd07449f4152b7f5d935f92aed15272fb8333ed9cc60b5766`. I read the whole note to understand the boundary, and substantively checked only §§3–5 (lines39–177 of the frozen local file). The author control script is5818B / `5b1e8fc18a6e32e3a312a253e3e4d1110bb674385f3d49447a45e78ad4df541f`; its frozen JSON is1618B / `1d79c4f420d8a007638eff98a0b93ffed74e5b1acbac60c2f7777c2cf012e764`.

The specific consumed source imports were checked in their retained immutable bytes:

| Source | Bytes | SHA-256 | Imported scope |
|---|---:|---|---|
| finite_gaussian_contact/PROOF.md | 25367 | 7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474 | actual real-coefficient law, all-frame seven-row rank, small-r augmented rank/regression, determinant-weighted contact limit |
| finite_candidate_lifetime/PROOF.md | 29190 | ed31f562f0ffaaf6d4bd2ceb1ae3d234798e983a55d55c6983b128fce27133f5 | Sard Morse genericity, distinct critical values, all-Borel ordered-pair Rice identity, lifetime Jacobian and coefficient |
| FINITE_FOURIER_CRITICAL_COUNT_DIAGONAL_ROUTE.md | 25699 | 621cee96854aa440498cecf5452a8d1e23f90a4458779c3286c18eba106fe138 | only the accepted unconditional a.s. critical-point cap16K² |
| NONAUTHOR_FINITE_COUNT_DIAGONAL_REVIEW.md | 21880 | 16152427f7013822ac1bba81142429800334e6bf0f354a6439b23a1997bc4524 | existing scoped acceptance of that finite-law cap; no duplicate review |

The first two are pinned to Main85b928b326ff0586715a56ae91f51d3fbd89557a. I inspected the relevant local proofs directly, including FG§§1–5 and FC§§4–6,8–10, and opened the [pinned contact source](https://raw.githubusercontent.com/d6g8k5htny-coder/main/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_gaussian_contact/PROOF.md) and [pinned candidate source](https://raw.githubusercontent.com/d6g8k5htny-coder/main/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_candidate_lifetime/PROOF.md) as primary-source cross-checks. No new external theorem is required beyond those named imports and elementary finite-dimensional Gaussian, contraction and measure arguments explained below. I did not rerun the earlier accepted norm/count/isolation reviews.

The actual command records are in `counterfamily-nonauthor-checks-20261016/EXECUTION_RECORD.json`. I ran the frozen author controls and my own independent `counterfamily_checks.py` with `python3 -B`. Both exited0, with empty stderr. The author stdout matches its1618B frozen JSON byte for byte. My187 exact Fraction/integer checks passed. The tests include18 rational oriented/reflected frames, independent moment-based covariance construction, full degree-three grid rank, determinant/trace/cubic-Schur identities, five mixture cases, nonprimitive-resonance residue arithmetic, short lifts across periodic seams, and meaningful negative controls. There are no field draws, approximate roots, numerical eigenvalues, Rice integrals or process simulations. The three author files were hashed before and after both runs and remained unchanged.

The finite controls support arithmetic. The all-frame proof, a.s. genericity/stability, boundary null sets, diagonal existence and process conclusion are conventional arguments, not consequences inferred from sampled frames or control success.

## 2. §3: upper moments do not supply a lower variance

With physical frequencies xi_n=n/N and n in[-2N,2N]², every frequency satisfies |xi_n|²<=8. Thus sum q_n|xi_n|^12<=8^6=262144 for every displayed normalized mixture. Full positive fill gives q_n>0, symmetry and exact total mass1; the dominant DC contribution vanishes under every directional derivative. Therefore Var(F_u)<=8epsilon_N for every unit u. Selecting that coordinate in the seven-row covariance gives the claimed Rayleigh upper bound. Positivity for each fixed positive full square is consistent with collapse of its common lower bound.

Putting the dominant mass instead on the four physical axial modes of length1/N gives directional gradient variance(1-epsilon_N)/(2N²) plus fill variance<=8epsilon_N. This verifies the second collapse without a dominant DC term. The small remaining DC weight from the positive fill is allowed; no claim that q_0 was removed entirely is necessary. Normalized variance, upper twelfth moments and individual smoothness do not impose any lower derivative variance.

## 3. §4: actual law, full positivity and every-frame contact floor

G has25 real independent Gaussian coordinates under the actual sine/cosine convention. Repeating it on the torus of side2piN places its physical frequencies at m, through Fourier indices n=Nm. Its derivatives are consequently the original physical derivatives, not derivatives scaled by1/N. Independent H_N supplies one independent Gaussian sine/cosine coordinate per mode pair plus the DC coordinate on the full square. At overlaps, summing independent Gaussian coefficients produces exactly the variance sum in equation(1); different mode pairs remain independent. The result is the stated centered real ideal Gaussian law with all(4N+1)² positive symmetric mode weights and unit variance. No independence between repeated cells is asserted or used.

For a seven-row linear functional in any real orthonormal frame, its complex multiplier is a polynomial of total degree<=3 in the two physical frequency coordinates. Vanishing variance with positive weights on all25 grid points forces the polynomial to vanish at each point. Successively interpolating in each coordinate (five points, degree<=3) forces a polynomial identity. The invertible frame change then separates1,a,a²,a³,c,ac,c², forcing all seven real coefficients to vanish. Thus every frame, including reflections, has positive covariance. Entries are continuous in O(2); its compactness gives a common positive floor. This proves the all-frame assertion; an axial LDL calculation alone would not suffice.

I also derived an independent explicit uniform lower bound for this particular uniform5×5 base. Let u=(c,s), v=(-s,c), c²+s²=1, and q=c²s² in[0,1/4]. For independent uniform X,Y in{-2,-1,0,1,2}, their moments are E X²=2, E X⁴=34/5, E X⁶=26. Odd and even derivative rows decouple by inversion parity. The conditional variance of F_uuu after both gradient rows is

```text
E(cX+sY)^6 - (|E[X(cX+sY)^3]|²+|E[Y(cX+sY)^3]|²)/2
 = (26+126q) - (578+2106q)/25
 = (72+1044q)/25 >=72/25.
```

Changing from(H_xx,H_xy,H_yy) to(H_uu,H_uv,H_vv) has determinant±1. Hence the even block(F,H_uu,H_uv,H_vv) has determinant784/25, using the conditional diagonal-square variance14/5 and mixed variance4. With the two-gradient determinant4, this gives

```text
det Gamma_G(u)=3136(72+1044q)/625 >=225792/625,
tr Gamma_G(u)=(243+682q)/5 <=827/10.
```

Reflections only change signs of rows involving v and do not affect these identities. Since each other eigenvalue is at most the trace,

```text
m_G >= (225792/625)/(827/10)^6
    =361267200/319913861015774089 >0.
```

The polynomial/compactness proof already establishes the required result; this additional conservative floor checks the orientation issue quantitatively. The independent controls verify these identities at18 rational oriented/reflected frames and rank10 for the complete total-degree-three multiplier space. A K=1 negative control has the exact axial annihilator F_u+F_uuu=0, demonstrating why one cannot omit the required support argument.

The covariance decomposition is a sum of positive semidefinite matrices. Thus Gamma_N>=(1-epsilon_N)Gamma_G and epsilon_N<=1/2 yields the claimed floor m_G/2, uniformly in N and frame. Every positive mixture has M12<=262144. The one-contact cubic variance stays positive; this obstruction does not discard the fold jet.

## 4. §4: macroscopic coherence and two-contact degeneracy

For j=1,...,N-1, the integer interval[-2N,2N] has four copies of every residue moduloN and one extra zero residue. Evaluating its character at any nontrivial Nth root gives sum1, whether or not the root is primitive. The transverse sum contributes4N+1. Therefore the normalized fill covariance at h=(2pij,0) is exactly1/(4N+1). The repeated G and all its physical derivatives agree at those translations, so R_N(h)=1-epsilon_N+epsilon_N/(4N+1).

The unit two-value difference vector has variance1-R_N(h)=epsilon_N(1-1/(4N+1))<epsilon_N. It is a legitimate vector in the fourteen-row two-contact covariance, independently of the chosen derivative frames. Hence its least eigenvalue collapses. With j=floor(N/2), torus distance lies betweenL_N/4 andL_N/2 forN>=4, so this is a genuinely far resonance. Value correlation tends to1; far covariance decay/factorization cannot be deduced from the one-contact floor and upper moments.

This spectral obstruction alone would not disprove every Poisson conclusion: an empty limit is Poisson with zero intensity, and failure of one sufficient covariance estimate is not failure of a limit theorem. The actual process argument in§5 is therefore necessary and was reviewed separately below.

## 5. §5: the global C4 estimate and uniform repeated-cell stability

There are15 coordinate multiindices alpha with |alpha|<=4. At each x, the physical coefficient row for partial^alpha H_N has squared norm equal to its derivative variance, at most64²: normalized mass1 and |xi|<=2sqrt2 give row norm<=(2sqrt2)^4=64, with the same bound for lower orders. Cauchy–Schwarz against the real coefficient vector Z_N gives the global supremum bound for each row, including mixed derivatives. Summing gives the stated coordinate sum-of-suprema bound||H_N||C4<=960||Z_N||. There are exactly(4N+1)² real coordinates, not twice that number. Thus the noise squared-norm expectation is at most960²epsilon_N(4N+1)²=O(N^-10) when epsilon_N=N^-12.

The amplitude adjustment of G is also harmless: 1-sqrt(1-epsilon)<=epsilon and G has a finite C4 second moment in25 coordinates. Its repeated physical C4 norm equals its base norm. Combining the two terms, with a factor2 when squaring their sum, proves the required global perturbation convergence. The constant960 is for the stated coordinate C4 sum; ordinary directional/operator conventions differ by fixed dimension constants and do not change convergence or the deterministic radius choice. No local-norm theorem from the previous review is imported.

The fixed-base source establishes that G is Morse almost surely before Rice, and uses the all-Borel pair formula to prove distinct critical values. Its critical set is finite by compactness and isolation. Take disjoint lifted balls around its critical points, smaller than the base injectivity scale. At each p, write H_p=Hess G(p), and arrange||H_p^-1(Hess G(x)-H_p)||op<1/4 throughout the ball. Choose the perturbation radius to make the analogous additional Hessian term<1/4. Then

```text
T_p(x)=x-H_p^-1 grad f(x)
```

has Lipschitz constant<1/2. Existence, as well as uniqueness, follows by also choosing||H_p^-1 grad f(p)||<=R_p/2: T_p maps the closed ball of radiusR_p into itself and is a contraction. This supplies a unique critical point there. The sufficiently small Hessian error preserves its Morse index. A positive base gradient floor on the complement excludes every other root. Repeating those finite balls and the same gradient floor proves all assertions with a radius independent of the number of cells. Large perturbation frequency cannot create additional roots when this global C2 bound holds.

A C1-only statement would be false: for a local base derivative x, a perturbation sin(m²x)/m is uniformly small but gives signs+,-,+ at pi/m²,3pi/(2m²),2pi/m² for m>=8, hence two additional roots besides0. Its Hessian error is of orderm, so it violates precisely the C2 hypothesis used above. This is retained as a negative control against silently weakening stability.

Let Delta(G) be the minimum absolute height gap between a base maximum and a base saddle. It is positive almost surely. Shrink the stability radius so each perturbed critical height is withinDelta/4 of its corresponding base height. Distinct repetitions of the same base critical point have the same Morse index and cannot provide a maximum–saddle pair with base gap0. Thus every positive typed candidate gap in the large torus is at leastDelta/2. Height ties between repetitions of one point are allowed; asserting distinct global critical values for G_repeated would be false. The endpoint role restriction is essential here.

For completeness the random-radius argument needs both the radius and the gap. Given a small error tolerance, choose deterministic d,D>0 with P(delta(G)<=d or Delta(G)<=D) small. For largeN, Tt_N<D/2 and the probability of a global perturbation of norm>=d tends to0. On the remaining event there is no candidate of gap<=Tt_N. This uses an unconditional union bound, not Gaussian conditioning at a data-selected critical point. A measurable positive radius can be chosen from countable coefficient neighborhoods of the finite base Morse branches; no probabilistic independence of that radius and the perturbation is required.

It follows that every bounded-lifetime compact candidate mark window has zero mass with probability tending to1. Rescaling locations byL_N does not change that statement. For a nonnegative continuous compactly supported test, the difference of its Laplace functional from1 is bounded by the probability that the supporting window contains a candidate. Therefore the process tends to the empty process on this sector. This excludes a universal **nonzero** Poisson limit under U plus the one-contact floor; it remains compatible with zero intensity. It does not prove a statement about an elder/global H0 selector.

## 6. §5: fixed-base restricted coefficient is finite and strictly positive

The additional compact window b in[-1,1], k in[1,2], A_M in[-2,-1] is an allowed coefficient/position Borel mark in the literal ordered-pair Rice identity. For sufficiently small r, the seven-row augmented covariance is uniformly positive over all frames. Consequently A_M conditioned on the six exact pins has positive scalar variance; at contact its Gaussian density is positive on every point of[-2,-1]. It need not be centered. The six-pin density at v_0=(b,0,0,12k,0,0) is likewise strictly positive. Thus z_0^W>0 pointwise, and compact continuity/domination gives a finite strictly positive integral. No coefficient positivity is postulated or inferred from an axial numerical test.

The indicator on A has zero limiting Gaussian boundary mass. Conditional Gaussian coupling plus uniform moments therefore transfers the determinant-weighted contact limit with that window. On this compact negative-curvature interval, type limits have no additional singular A=0 boundary. The contact determinant product36k²A² and the six-pin density occur exactly once each.

I checked the constants independently from the literal ledger:

```text
(r dr d sigma)(r^3 db dk)(12r^-5 pin_density)(r^2 V_r)
 =12r dr db dk d sigma (pin_density V_r),
r dr/dell = (1/3)ell^-1/3 k^-2/3.
```

Thus the lifetime density has coefficient4 times the displayed b,k,u integral. Integration over[t/2,t] multiplies by(3/2)(1-2^-2/3)t^(2/3), giving coefficient6V_0, exactly as in equation(8). V_0=(2pi)² is the base torus volume. The arclength measure has total2pi; it is not normalized. Endpoint roles are ordered with u from maximum to saddle and v=Ju, so there is no extra1/2 or reflected-frame multiplicity. This verifies c_W and its positive finite limiting mean without computing its numerical value.

## 7. §5: exact short lifts and null boundaries for diagonal continuity

Choose N sufficiently large that N^-1<min(r0,r_*), where r_* is the fixed-base small-r augmented-rank band. In the stated window, ell<=N^-3 and k>=1 imply r<=N^-1. Hence r0 is an inactive cutoff: its boundary cannot occur in the window. The endpoints lie below the base injectivity scale. Projecting a short pair to the base torus is unique; for each lifted maximum there is one short lift of its specified base saddle. This remains true across a base-cell seam. There are exactlyN² translations, notN^4 arbitrary combinations of endpoint lifts. My independent finite cyclic-lift enumeration checks both ordinary and seam-crossing examples and rejects unrestricted multiplicity as a substitute.

For each fixed sufficiently largeN, the base field has finitely many critical branches and is away from type degeneracy almost surely. Relevant births/lifetimes lie off fixed boundaries by the source's absolutely continuous two-height Rice disintegration. Boundaries k=1 or2 are null by the positive height-to-k Jacobian r³ at each nonzero separation. Boundaries A_M=-2 or-1 are null using the small-r positive conditional scalar variance and the same Borel Rice identity. The determinant weight is integrable for this fixed finite field; multiplying a zero-probability scalar boundary event by it does not produce mass. Torus cut loci are absent in the relevant short-radius window. Orientation has no additional restricted boundary.

The copied field's same-type height ties do not invalidate this argument: every relevant typed short pair projects to two different base critical points, where the null-boundary proof is applied. Base max–saddle gaps are not0; copied points cannot change index under the sufficiently small perturbation. There are only finitely many branches/pairs at each fixedN. Away from the finitely many mark-boundary events the count is therefore locally constant in C4 (indeed C2 suffices for these branch/curvature marks). Since F_epsilon,N tends almost surely to G_repeated in C4 as epsilon decreases at fixedN, bounded convergence of the disagreement indicator proves P(C_epsilon,N!=C_dom,N)->0.

The proof only needs sufficiently largeN; arbitrary initial finitely many fills can be chosen positive. It does not need null curvature-boundary claims for a non-small-r band where the augmented-rank source supplied no such assertion.

## 8. §5: deterministic diagonal, count cap and first-moment transfer

For each sufficiently large fixedN, choose a deterministic dyadic epsilon_N>0 with epsilon_N<=N^-12 and disagreement probability<=N^-10. Existence follows from the fixed-N limit just proved. For example take the first dyadic satisfying both inequalities; the eligible set is nonempty. The probabilities concern the unconditioned fixed law of(G,H_N), so this is a choice of a spectral coefficient sequence before the field draw, not an adaptive sample-dependent choice or conditioning. No computable rate, evaluated probability or numerical epsilon sequence is claimed.

For every positive epsilon_N, the full coefficient law on K=2N is absolutely continuous. The accepted fixed-K critical bound can be applied separately at eachN, retaining its explicit K dependence: Ncrit<=16(2N)²=64N² a.s. Thus the total endpoint-candidate cap is4096N^4. No new covariance/Rice weight is inserted in this direct count bound. The sparse repeated dominant field is not assigned full-carrier absolute continuity; its short count is bounded instead by the exactN² lifts of the fixed-base count<=4096, giving4096N². This distinction avoids an invalid application of the generic bound to the degenerate sparse coefficient law.

On the disagreement event, the absolute difference is at most the sum of those caps. Therefore

```text
E|C_epsilon_N,N-C_dom,N|
 <= (4096N^4+4096N²)N^-10
 <=8192N^-6 ->0.
```

A merely N^-4 disagreement estimate would not establish this transfer: its first cap contribution remains4096. That failure is checked independently. All constants needed from the earlier count lemma are tracked here; no uniform tail/cutoff constants from that earlier fixed-law review are being imported acrossN.

The exact lift identity and fixed-base asymptotic give

```text
E C_dom,N(N^-3)
 =N² E C_G(N^-3)
 -> c_W(1-2^-2/3),  with 0<c_W<infinity,
```

because N²(N^-3)^(2/3)=1. The diagonal error tends to0, while epsilon_N<=N^-12 preserves the preceding global perturbation/empty-process argument. Hence this strictly positive full-square family has a finite positive limiting mean in the chosen compact sector and an empty-in-probability process. It is a failure of uniform integrability, so the first moment cannot identify a nonzero Poisson limit. This is not contradictory: rare repeated base fold events can contribute many translated copies.

## 9. Acceptance boundaries and nonblockers

Accepted: the exact ideal Gaussian counterfamilies; all-frame one-contact floor in the second family; macroscopic two-contact degeneracy; global C4 convergence for explicit epsilon_N=N^-12; uniform repeated-cell C2 Morse/height stability; empty compact candidate process at t_N=N^-3; positive fixed-base compact coefficient; short-pairN² multiplicity; and the separate deterministic existence-only diagonal preserving its mean.

Retained limits, not blockers: no explicit numerical diagonal epsilon values or continuity rate; no claim that explicit N^-12 transfers the mean; no evaluated c_W; no iid-cell approximation; no global distinct-critical-value assertion for the repeated field; no conclusion for infinite spectra or the project's actual changing source; no elder/global selector, actual persistence-bar process, higher homology or universality conclusion. The failure of uniform covariance alone is not presented as a process theorem. §§6–9 need their separate review, and this report does not discharge their source-specific hypotheses or estimates.

No frozen author or previous review file was edited. All new outputs are this standalone report, its manifest and the separate local check directory. No hosted/archive/register/native or production action was taken.
