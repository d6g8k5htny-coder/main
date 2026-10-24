# SUMMARY
I checked every claim in the note. Its mathematics is mostly right. None of it moves toward RH, and its proposed decisive step is not just "tied to" RH: it is RH, restated.

The core identities all check out numerically with mpmath:
- **Normalization:** H_0(z) = Xi(z/2)/8, relative error at most 7e-41.
- **Kernel:** Phi(u) is an explicit derivative of the theta function, and Phi is even.
- **Heat equation:** the sign of the PDE dH/dt = -d²H/dz² is confirmed.
- **Zero motion:** zeros move by dz_j/dt = 2 sum_{k≠j} 1/(z_j - z_k), so they repel as t increases (agreement to about 1e-4).
- **A real collision:** zeros of H_t meet as a double zero at z0 ≈ 99.3285, t0 ≈ -1.8779. Near it H ≈ A(x² - 2τ): r²/(8τ) and area/(|A|r³/6) both tend to 1. The two zeros are real for τ > 0 and a complex pair for τ < 0. The antiderivative gains exactly one new persistence bar there.
- **Theta and zeta:** the Mellin formula, the theta functional equation, the Epstein extension, and the project's normalizer as a Jacobi theta value with its Poisson image-sum all verify.
- **GUE and the test:** the GUE small-r expansion and the density-transfer test are right (gaps ~ r² give ℓ^0; gaps ~ r give ℓ^(-1/3)).

Errors and imprecisions in the note:
1. It says "two bridges" but lists three.
2. The time direction of collisions is muddled. Real zeros approach as t decreases; going forward, a complex pair lands on the axis and splits into two real zeros.
3. It omits Phi and the 1/8 normalization.
4. Xi's own zeros (t = 0) are not known to collide. Collisions happen for H_t at times t ≤ Λ, for example the one at t0 ≈ -1.878.
5. "No positive-time transition" is equivalent to Λ ≤ 0, which is equivalent to RH.
6. Off-line zeros can be detected from real-axis data: compare the real-zero count with the argument-principle count, which is how RH is verified numerically.
7. Describing collisions in persistence language is standard birth–death (Cerf theory, vineyards), not a new contribution.
8. The ℓ^0 test needs a condition on the bar-height coefficient k: E[k^(-1)] must be finite. The project's own phase diagram (reviews/collision_mechanism_20260925/NOTE.md §D2) already covers this case: ℓ^0, log(1/ℓ) or ℓ^q, depending on how k is distributed near 0.
9. The "global constraint" the note asks for in §7(b) already exists: de Bruijn's strip lemma Λ ≤ t0 + y0²/2, Ki–Kim–Lee 2009, and Rodgers–Tao's zero dynamics.
10. "(Mumford)" is an expository source; the primary source is Riemann 1859.

New-but-elementary observations:
- ζ is, up to a Gamma factor, the Mellin transform of the project's normalizer taken over the torus side length L. Zeta therefore lives along the scale direction that the project holds fixed.
- At L = 24 the theta periodization changes the covariance and its derivatives at a point by about e^(-288) ≈ 1e-125 (roughly e^(-L²/2)). It is invisible to the local fold law; the project uses Poisson summation only to get strictly positive Fourier weights.
- The persistence diagram of an antiderivative depends on how H is normalized, not just on its zeros. Data from Hardy's Z on two windows (T in [600, 900] and [5000, 5200]): about 80% of bars sit between adjacent zeros. Lobe area tracks gap³ (slope 3.0). Rescaling by the Xi weight changes about 30% of the pairings, and Xi's overall sign changes all of them.

One conjecture, conditional on GUE behaviour of small gaps and a finite E[k^(-1)]: for Hardy-normalized zeta the short-bar density is ~ℓ^0, against C0·ℓ^(-1/3) for stationary Gaussian processes (the project's one-dimensional theorem). A random-matrix version of this is a well-posed theorem target. It is not a route to RH.

Citation caveat: arXiv and journal pages could not be fetched (proxy 403 / DNS). Bibliographic details were checked against search-engine snippets only, and items recalled without checking are marked "from memory" in the table.

# DETAILS
## Claim-by-claim verification of the RH ↔ persistence note

Workdir: /tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/ (scripts + .out files listed in artifacts). Project files read (read-only): Math- 79cb314 `reviews/radial_moment_corollaries_20260930/P.md` (= UNIFORM-MATRIX-CAP-LIFETIME; torus covariance §1–2, radial ledger r A_r dr §10–11, Theorems B/C ν~cℓ^{-1/3}), `reviews/collision_mechanism_20260925/NOTE.md` (§D1 counterexample, §D2 phase diagram), `frontiers/d1_line_20261004/README.md` (1D theorem ν(ℓ)=C0ℓ^{-1/3}+C1ℓ^{1/4}+…). I did not touch wt-guide.

Verdict key: **C** correct · **CC** correct-with-caveat · **I** imprecise · **W** wrong.

### Table

| # | Note claim | Verdict | Precise / corrected statement; evidence |
|---|---|---|---|
| 0a | "Two bridges: (1)(2)(3)" | I | Three bridges are listed, not two. |
| 0b | "Genuine connections, not evidence the project proves RH" | C | Agreed. Nothing below changes this. |
| 1.1 | ζ(s)=Σn^{-s} for Re s>1, continued | CC | ζ continues meromorphically to ℂ. Its only singularity is a simple pole at s=1 with residue 1 (Riemann 1859; Titchmarsh ch. II, from memory). |
| 1.2 | Nontrivial zeros lie in 0<Re s<1 | C | This is a theorem: ζ(1+it)≠0 (Hadamard / de la Vallée Poussin 1896) plus the functional equation. The trivial zeros are at −2, −4, …. |
| 1.3 | RH: all nontrivial zeros on Re s=1/2 | C | Agreed. |
| 2.1 | f=x³/3−ax, f'=x²−a, critical points ±√a collide at a=0 | C | This is the A2 fold, i.e. Cerf birth–death of a critical pair. |
| 2.2 | Height gap ∝ r³ | C | Exactly: f(−√a)−f(√a)=(4/3)a^{3/2}=r³/6 with r=2√a. In the project's convention ℓ=k r³, k=\|f'''\|/12 at contact (P.md §3: the U3 target is 12k), so this normal form has k=1/6. |
| 2.3 | Combined with configuration probability it gives the exponent | CC | −1/3 comes from the pair intensity ∝ r (ledger r A_r dr) pushed through ℓ=k r³ with admissible marks and a positive pairing limit (Theorems B/C; 1D: frontiers/d1_line_20261004). Geometry alone does not fix it: the project's §D1 shows cubic splitting need not give a density coefficient, and §D2 shows the exponent changes when the mark density blows up near 0. |
| 3.1 | H_t(z)=∫_0^∞ e^{tu²}Φ(u)cos(zu)du | CC (incomplete) | Φ(u)=Σ_{n≥1}(2π²n⁴e^{9u}−3πn²e^{5u})exp(−πn²e^{4u}) (Rodgers–Tao / Polymath15 conventions). **Verified** H_0(z)=Ξ(z/2)/8, with Ξ(x)=ξ(1/2+ix) and ξ(s)=½s(s−1)π^{−s/2}Γ(s/2)ζ(s); relative difference ≤7e−41 at z∈{0,10,28.27,40,75,150}. Also verified to 20 digits: Φ(−u)=Φ(u), and Φ(u)=e^u[2x²ψ''(x)+3xψ'(x)] at x=e^{4u}, ψ=(Θ−1)/2. So the de Bruijn–Newman kernel *is* a theta derivative, and its evenness is the theta transformation (classical; Titchmarsh §2.16, from memory). |
| 3.2 | At t=0, completed zeta in coordinates where RH ⇔ all zeros real | C | H_0(z)=ξ(1/2+iz/2)/8, so a zero ρ=1/2+iγ corresponds to z=2γ. |
| 3.3 | ∂_tH=−∂_z²H (backward heat) | C | Derivation: ∂_tH=∫u²e^{tu²}Φcos(zu)du and ∂_z²H=−∫u²e^{tu²}Φcos(zu)du. Differentiating under the integral is justified by Φ(u)=O(e^{9u−πe^{4u}}). Finite-difference check at (t,z)=(0,30),(0.3,55),(−1,20): residual ≤4e−27 against terms of size 1e−4 to 1e−10. |
| 3.4 | Zeros move; real zeros approach, collide and "with time reversed" leave the axis | I (direction) | For simple zeros, ż_j=2Σ_{k≠j}1/(z_j−z_k). This ODE is from Csordas–Smith–Varga 1994 and is used by Rodgers–Tao. Absolutely convergent form with mirror zeros: ż_j=2[1/(2z_j)+Σ_{k≠j}2z_j/(z_j²−z_k²)]. Derivation: ż=−H_t/H_z=H_zz/H_z, then the Hadamard product H=CΠ(1−z²/z_k²). Consequence: as t increases zeros repel; as t decreases neighbours attract, collide at a real double zero, and become a conjugate pair for t<t0. **Verified at t=0:** finite-difference ż for zeros #1, #2, #5 = −0.579631, −0.754995, −0.465445. The ODE with 300 zeros plus a tail estimate gives −0.579717, −0.755108, −0.465554, which rules out the opposite sign. |
| 3.5 | (Intro) "zeros of the Riemann xi function collide under a heat flow" | I | It is the zeros of H_t that collide, and only at t≤Λ; for t>Λ zeros are simple (CSV). No multiple zero of Ξ itself (t=0) is known. Example found: a double zero at z0=99.328469305575750948, t0=−1.8779090729725352648 (the merger of H_0's zeros at 2γ_9 and 2γ_10). Negative-time collisions say nothing about RH, since Λ≥0 is already proved. |
| 3.6 | All zeros of H_t are real iff t≥Λ | C | de Bruijn, Duke Math. J. 17 (1950) 197–226: zeros are real for t≥1/2, and real at t implies real at every t'>t. Newman, PAMS 61 (1976) 245–251: Λ exists and Λ≤1/2. Ki–Kim–Lee, Adv. Math. 222 (2009) 281–306: Λ<1/2, and for every t>0 all but finitely many zeros are real and simple. Pitfall: the time variable rescales by 4 between normalizations (in the Ξ(x) variable de Bruijn's 1/2 becomes 1/8). |
| 3.7 | RH ⇔ Λ≤0 | C | Follows from 3.2 and 3.6. |
| 3.8 | Rodgers–Tao Λ≥0, so RH ⇔ Λ=0 | C | B. Rodgers, T. Tao, "The de Bruijn–Newman constant is non-negative", Forum Math. Pi 8 (2020) e6, arXiv:1801.05914. Method: assume Λ<0, so H_t has only real zeros for t∈(Λ,0]. Run the CSV zero dynamics with a mollified energy to show H_0's zeros are in local equilibrium (gaps on average like an arithmetic progression). This contradicts "known results on the local distribution of zeros … such as the pair correlation estimates of Montgomery" (abstract wording via search snippet). Since Λ<0 implies RH, RH-conditional results are available inside the argument. |
| 3.9 | Upper bounds (task item) | C | D.H.J. Polymath, Res. Math. Sci. 6 (2019) art. 31, arXiv:1904.12438: Λ≤0.22, from de Bruijn's lemma (zeros of H_{t0} in \|Im z\|≤y0 ⇒ Λ≤t0+y0²/2) with t0=y0=0.2. The earlier test problem was t0=y0=0.4 ⇒ 0.48. Platt–Trudgian (arXiv:2004.09765; Bull. LMS 53 (2021) 792–797, venue per secondary source): Λ≤0.2. |
| 3.10 | Lower bounds via Lehmer pairs (task item) | C | Csordas–Smith–Varga, Constr. Approx. 10 (1994) 107–129: defines Lehmer pairs; each yields a lower bound. Csordas–Odlyzko–Smith–Varga, ETNA 1 (1993) 104–111: Λ>−5.895·10⁻⁹. Odlyzko, Numer. Algorithms 25 (2000) 293–303: Λ>−2.7·10⁻⁹, from a pair near zero #10^20. Saouter–Gourdon–Demichel, Math. Comp. 80 (2011) 2281–2287: Λ>−1.14541·10⁻¹¹. From memory: Csordas–Norfolk–Varga 1988 (−50), te Riele 1991 (−5). Mechanism: a very close pair at t=0 must have collided at some t_c<0 near 0, so Λ≥t_c. Two-body heuristic: ṙ≈4/r, so r²≈8(t−t_c), consistent with 3.11. CSV make this rigorous with a condition on the other zeros; I did not reproduce their exact inequality. |
| 3.11 | Local form H≈A(x²−2τ), zeros ±√(2τ) | C (side unspecified) | At a double zero, H=H_z=0 and A=H_zz/2≠0. The PDE gives H_t=−2A and H_zt=−H_zzz, so H=A(x²−2τ)+(H_zzz/6)(x³−6xτ)+O(x⁴+x²\|τ\|+τ²). These are heat polynomials, and the fold is automatically transversal in t. Zeros are real for τ>0 (t>t0) and form a conjugate pair z0+O(\|τ\|)±i√(2\|τ\|) for τ<0, whatever the sign of A. **At the actual collision:** r²/(8τ)=0.999008, 0.9999007, 0.99999007, 0.9999999 for τ=1e−2, 1e−3, 1e−4, 1e−6. Im z/√(2\|τ\|)=1.0005 at τ=−1e−2 and 1.000005 at τ=−1e−4. While approaching, d(gap²)/dt=7.92 at gap 0.55 (two-body value 8). |
| 3.12 | F_t'=H_t gives F≈A(x³/3−2τx)+c, whose critical points are the two zeros | C | Holds for τ>0. |
| 3.13 | Height gap ℓ≈\|A\|r³/6 | C | F(x−)−F(x+)=(4/3)A(2τ)^{3/2}=A r³/6 with r=2√(2τ). So the bar is (8√2/3)\|A\|τ^{3/2}, and in the project's convention k=\|F'''\|/12=\|A\|/6. **Numerics:** area/(\|A\|r³/6)=1.0045, 1.00045, 1.000045, 1.00000045 (≈1+0.45τ). Superlevel H0 persistence of the sampled F_t at τ=+0.01 has exactly one finite bar in the window, from the new max to the new min, length 1.0445e−18 (exact area 1.0450e−18). At τ=−0.01 there is no finite bar. By the elder rule a newborn pair is always paired with itself. |
| 3.14 | Follows analytically near a nondegenerate double zero | C | Nondegenerate means H_zz≠0. |
| 3.15 | Literature studies such collisions; the new part is persistence language | CC | The literature is 3.10 plus Rodgers–Tao (dynamics, local equilibrium) and Polymath15 (zero dynamics, barrier). The persistence phrasing is itself standard birth–death in a generic one-parameter family (Cerf 1970; vineyards, Cohen-Steiner–Edelsbrunner–Morozov 2006; from memory). There is no new mathematics here. |
| 4.1 | q_n=e^{−ω²\|n\|²/2}/Σ_j e^{−ω²\|j\|²/2}, ω=2π/L | C | Matches P.md §2: a_n=e^{−2π²\|n\|²/L²}/Σ_j(…). |
| 4.2 | Same Gaussian-lattice construction as Jacobi theta | CC | Exactly Σ_j e^{−ω²\|j\|²/2}=Θ(ω²/2π)^d=Θ(2π/L²)^d. Verified at L=24: d=2 gives 91.6732472209…, d=3 gives 877.7360231549…. The image normalizer is Σ_n e^{−L²\|n\|²/2}=Θ(L²/2π)^d, and Poisson/Jacobi gives N_C=(2π)^{d/2}L^{−d}N_F (verified). The covariance is K_L(z)=Π_i ϑ_3(πz_i/L,q)/ϑ_3(0,q) with q=e^{−2π²/L²}, the normalized torus heat kernel at time 1/2; image sum = Fourier series = ϑ_3 ratio verified to 18–20 digits. **Caveat:** at L=24 the periodization changes the jets at a point by O(e^{−L²/2}·poly(L)), about 1e−125; K'', K⁗, K⁽⁶⁾ at 0 equal −1, 3, −15 to working precision. The local fold law therefore sees the planar squared-exponential kernel. The project uses Poisson summation only to get a_n>0 (nondegeneracy); the modular symmetry u↦1/u plays no role. |
| 4.3 | π^{−s/2}Γ(s/2)ζ(s)=½∫_0^∞(Θ(u)−1)u^{s/2−1}du for Re s>1 | C | Verified at s=2, 3, 4.5, 2+3i (18 digits) and s=1.3 (5e−6; limited by quadrature). For Re s≤1 the integral diverges at u→0, where the integrand behaves like u^{(s−3)/2}. |
| 4.4 | Θ(u)=u^{−1/2}Θ(1/u) gives ξ(s)=ξ(1−s) ("Mumford") | CC | Correct. The primary source is Riemann 1859 (second proof); Mumford's Tata Lectures I is expository. Verified the transformation at u=0.37, 2, 0.05 (direct series). Verified Riemann's continued form Λ(s)=∫_1^∞ψ(u)(u^{s/2}+u^{(1−s)/2})du/u−1/s−1/(1−s) at s=0.3+5i and s=ρ_1, with Λ(s)=Λ(1−s). |
| 4.5 | Higher-dimensional sums lead to Epstein zeta | C | ∫_0^∞(Θ^d−1)u^{w−1}du=π^{−w}Γ(w)Z_d(w), with Z_d(w)=Σ'_{n∈ℤ^d}\|n\|^{−2w} for Re w>d/2, and Λ_d(w)=Λ_d(d/2−w). Special cases: Z_2=4ζ(w)L(w,χ_{−4}); Z_4=8(1−4^{1−w})ζ(w)ζ(w−1). Factorization and functional equation verified for d=2 and 4. |
| 4.6 | The functional equation does not locate zeros | C (sharpened) | Z_4 has zeros at 1/2+iγ and 3/2+iγ, off its symmetry line Re w=1 (verified at γ_1). Davenport–Heilbronn, JLMS 11 (1936) 181–185 and 307–312: Epstein zetas of forms with class number >1 have infinitely many zeros in Re s>1. Theta structure plus a functional equation, without an Euler product, gives no RH-type statement. Also, the heat flow destroys the Euler product for t≠0: Polymath's approximation uses the non-multiplicative coefficients b_n^t=exp((t/4)log²n). |
| 4.7 | (New, elementary) | — | Mellin in the side length: ∫_0^∞(N_F(L,d)−1)L^{−s−1}dL=½(2π²)^{−s/2}Γ(s/2)Z_d(s/2); for d=1 this is (2π²)^{−s/2}Γ(s/2)ζ(s). Verified (d=1 at s=3 and 2+i; d=2 at s=3; d=4 at s=6). ζ appears only after integrating over all torus sizes, a direction the project holds fixed. No consequence for zeros. |
| 4.8 | Analytic bridge, not a proof bridge | C | Agreed. |
| 5.1 | R_2(r)=1−(sinπr/πr)² ~ (π²/3)r² | C | This is the conjectured pair correlation (Montgomery's conjecture; Dyson/GUE). Taylor series π²/3·r²−(2π⁴/45)r⁴+…, verified. The GUE nearest-gap density has the same π²/3·s² leading term; the Wigner surmise gives 32/π². |
| 5.2 | Project's contact intensity ∝ r | C | P.md ledger r A_r dr. In 1D: zeros of f' for a stationary Gaussian process repel linearly, and the project's 1D theorem gives ν=C0ℓ^{−1/3}+…. The 1D theorem is the right comparison object for the antiderivative picture (F↔f, H↔f'). |
| 5.3 | GUE statistics of zeta zeros remain conjectural | C | Montgomery, Proc. Sympos. Pure Math. 24 (1973) 181–193: assuming RH, F(α)=(1+o(1))T^{−2α}log T+α+o(1) for 0≤α≤1 (some statements use 1−ε). This is pair correlation for test functions with Fourier support in (−1,1). Rudnick–Sarnak, Duke 81 (1996): n-level correlations with restricted support. Odlyzko, Math. Comp. 48 (1987) 273–308: first 10^5 zeros and zeros 10^12+1 to 10^12+10^5, with later computations at 10^20 and beyond. The r→0 law is not proved; band-limited averages cannot resolve it. |
| 5.4 | Test: ℓ=kr³ with gap ~r²dr gives ℓ^0, not ℓ^{−1/3} | CC | The computation is right: ν(ℓ)=(c/3)k^{−(β+1)/3}ℓ^{(β−2)/3}. **Missing admission condition:** you need E[k^{−(β+1)/3}]<∞, i.e. E[k^{−1}] for β=2 against E[k^{−2/3}] for β=1. With mark density ∝k^q near 0, the project's §D2 phase diagram (α=2, m=3) gives ℓ^0 if q>0, log(1/ℓ) if q=0, and ℓ^q if −1<q<0. Verified by exact quadrature: β=2 with k~U(0,1) gives ν=log(1/ℓ) exactly; with k fixed or k~U[0.5,2] the local slope is 0 (β=2) or −1/3 (β=1). |
| 5.5 | Which parts are fold geometry and which are measure | C | The project's §D1/§D2 already answer this in the product-intensity model: m=3 is geometry, while α and q come from the measure; exponent (α+1)/m−1 when q+1>(α+1)/m. |
| 5.6 | (Task) What the antiderivative's persistence measures | — | A bar is F(max)−max(left valley, right valley): a signed sum of lobe areas ∫H, which is a single lobe only when the death point is adjacent. It depends on the positive normalization and on the sign. **Data:** the relation Ξ(t)=−½(t²+¼)π^{−1/4}\|Γ(¼+it/2)\|Z(t) was verified at t=600.123 and 5000.123. **T∈[600,900], Hardy Z:** 228 zeros (Riemann–von Mangoldt main term 228.0); 113 bars, 80.5% single-lobe. Regressing log(lobe area) on log(gap) gives slope 2.995, residual sd 0.257 (total sd 1.264). The 10 shortest bars are all adjacent close pairs with ℓ/gap³ between 0.50 and 1.01. **T∈[5000,5200]:** 214 zeros, 106 bars, 82.1% single-lobe; slope 3.054, residual sd 0.351 (total 1.394); shortest bar 0.0035 at normalized gap 0.129. **Normalization dependence:** with the Ξ weight (same sign), 80/113 and 82/106 pairs are shared with the Hardy version. With Ξ's actual sign, 0/113 and 0/106 are shared (superlevel and sublevel roles swap). A mild e^{0.01t} weight shares 112/113 and 104/106. Under Ξ normalization, lobe area/gap³ spans factors of 1e100 and 5e67. The amplitude marks broaden slowly with height (Selberg CLT: log\|ζ(½+it)\|/√(½log log T)→N(0,1); Selberg 1946, Radziwiłł–Soundararajan 2017, from memory), so no compact mark window holds uniformly in T. The growth of the residual sd from 0.26 to 0.35 is consistent in direction; it is not a test. |
| 6.1 | Route ends at "prove no positive-time transition, then Λ=0, then RH" | CC→I | The logic is valid but circular: "no positive-time transition" ⇔ Λ≤0 ⇔ RH. The step is equivalent to RH, not merely "closely tied" to it. |
| 6.2 | (Implicit) the transition at Λ is a real collision | plausible, unverified | If Λ>0, the Ki–Kim–Lee finiteness for t∈(0,Λ), if it holds locally uniformly in t, would force a real double zero at t=Λ. I have not verified the uniformity. |
| 6.3 | Real-axis persistence may miss off-line zeros | I | A purely local analysis misses them. Comparing the number of real zeros (critical points of F_t) with the total count N(T) from the argument principle detects them. This is how RH is verified numerically (Riemann–Siegel plus Turing's method; Platt–Trudgian to 3·10^12). Our window count 228 matches the main term. |
| 6.4 | Fundamental obstacle (random vs deterministic; local fold says nothing global) | C | Agreed. |
| 7(a) | Persistence of F_t on finite intervals tied to zeros | CC | Bar births and deaths correspond to real double zeros, plus zeros crossing the interval ends. Pairing switches happen at equal critical values with no zero event. The diagram depends on normalization (5.6). For t>Λ there are no folds at all, since zeros are simple and repel. So under RH no fold events occur for t>0; fold events live at t<0. |
| 7(b) | Find invariants or global theta constraints | CC | These already exist: de Bruijn's strip lemma, Newman's monotonicity, Ki–Kim–Lee, and the Rodgers–Tao energy (survey: Newman–Wu, Bull. AMS 57 (2020), arXiv:1901.06596). They hold for broad kernel classes; I know of no theta-specific global constraint beyond Φ>0, evenness and decay. |
| 7(c) | Compare collision statistics with fold theory | CC | The project's theorems are about Gaussian random fields and do not apply to the deterministic H_t. A random model is needed (see the conjecture below). |
| 7-table | Lattice↔theta↔zeta exact; folds↔dBN exact locally; persistence↔RH not established; exponent does not explain critical line | C | Add: the theta link runs through a Mellin transform over scale (4.7), and the local fold correspondence needs H_zz≠0. |

### Errors flagged in the note
1. "Two bridges" but three are listed.
2. The time direction of real-zero collisions is muddled (3.4).
3. Φ and the 1/8 normalization are missing (3.1).
4. "Zeros of Ξ collide" should be "zeros of H_t collide, at t≤Λ" (3.5).
5. The decisive step is equivalent to RH (6.1).
6. Off-line zeros are detectable by counting (6.3).
7. "New part = persistence language" is standard Cerf/vineyard material (3.15).
8. The ℓ^0 test is missing its admission condition, which the project's §D2 already covers (5.4).
9. The global constraints asked for in 7(b) already exist.
10. "(Mumford)" is a non-primary attribution.

### Labelled outputs worth keeping
- **Classical, now verified:** everything marked C above.
- **New but elementary:**
  - 4.7 (ζ as the Mellin transform of the project normalizer over L).
  - The L=24 jet-invisibility estimate (4.2).
  - A heat-equation fold lemma for any real solution of ∂_tH=−∂_z²H with a double zero where H_zz≠0. Then: real zeros exactly for t>t0 locally; r²=8(t−t0)(1+O(t−t0)); and the antiderivative gains one bar of length (8√2/3)\|A\|(t−t0)^{3/2}(1+O(t−t0)). The O(t−t0) correction is observed numerically, not proved.
  - Persistence pairings depend on the normalization (5.6).
- **Conjecture:** for Hardy-normalized zeta zeros, the short-bar density of ∫Z is ~ℓ^0, against C0·ℓ^{−1/3} for stationary Gaussian processes. It is conditional on GUE behaviour of small gaps and E[k^{−1}]<∞, and can be tested on Odlyzko/Platt zero data at large height. A well-posed random-matrix model to prove it in: real-normalized CUE/GUE characteristic polynomials, i.e. the project's structural theorem with the Kac–Rice contact kernel ∝r replaced by the determinantal two-point function ∝r². This is a research target, not an RH route.
- **Numerics:** our two windows (~110 bars each) are far too small to estimate an exponent; none is claimed.

### Citation provenance
Direct fetches of arXiv, Cambridge and Wikipedia were blocked (proxy 403 / DNS). Bibliographic data were confirmed only from search-engine snippets: Rodgers–Tao, Polymath15, Platt–Trudgian, Ki–Kim–Lee, CSV 1994, COSV 1993, Odlyzko 2000, SGD 2011, Newman 1976, de Bruijn 1950, Montgomery 1973, Odlyzko 1987, Rudnick–Sarnak 1996, Davenport–Heilbronn 1936, Newman–Wu. Items marked "from memory" were not checked.

### Reproduce
Run in the claims directory: `python3 check_dbn_basic.py` (~75 s), `python3 check_collision.py` (~95 s), `python3 check_collision_persistence.py`, `python3 check_theta.py`, `python3 check_gue_transfer.py`. Then `python3 sample_Z.py 600 900 0.02` and `python3 sample_Z.py 5000 5200 0.015` (each takes several minutes), followed by `python3 check_Z_persistence.py Z_grid_600_900.npz` (and the same for the 5000 window).

# ARTIFACTS
[
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/dbn_common.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_dbn_basic.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_dbn_basic.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_collision.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_collision.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/collision_point.txt",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_collision_persistence.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_collision_persistence.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/pers1d.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_theta.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_theta.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_gue_transfer.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_gue_transfer.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/sample_Z.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/check_Z_persistence.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/Z_grid_600_900_persistence.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/Z_grid_5000_5200_persistence.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/Z_grid_600_900.npz",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims/Z_grid_5000_5200.npz"
]