1. Count. The note says "two bridges" but lists three.

2. Time direction (§3). Under ∂_tH = −∂_z²H, simple zeros obey dz_j/dt = 2Σ_{k≠j} 1/(z_j−z_k), so real zeros repel as t increases.
- Collisions happen as t decreases: two real zeros approach, meet at a real double zero at t0, and for t < t0 become a conjugate pair z0 + O(|τ|) ± i√(2|τ|).
- Read forward in time, a conjugate pair lands on the axis at t0 and splits into two real zeros.
- Verified: the finite-difference dz/dt at t = 0 for zeros #1, #2, #5 is −0.579631, −0.754995, −0.465445. The 300-zero ODE gives −0.579717, −0.755108, −0.465554.

3. Missing kernel and normalization (§3).
- The kernel is Φ(u) = Σ_{n≥1}(2π²n⁴e^{9u} − 3πn²e^{5u})exp(−πn²e^{4u}). Φ is even and positive, and is a derivative of the Jacobi theta function.
- H_0(z) = Ξ(z/2)/8 with Ξ(x) = ξ(½+ix), so ρ = ½+iγ corresponds to z = 2γ. Verified to relative error ≤ 7e−41.
- The time variable rescales by 4 between the z and Ξ(x) conventions: de Bruijn's 1/2 becomes 1/8.

4. "Zeros of the Riemann xi function collide." It is the zeros of H_t that collide, and only at t ≤ Λ: for t > Λ all zeros are simple (Csordas–Smith–Varga 1994).
- No multiple zero of Ξ itself (t = 0) is known.
- Concrete collisions:
  - t* = −1.877909072972535264767, z0 = 99.328469305575750948 (zeros 9/10);
  - t* = −0.384561116627981414865, z0 = 223.275780185615519914 (zeros 34/35).
- Negative-time collisions say nothing about RH, since Λ ≥ 0 is proved.

5. The local form is leading order only, and needs H_zz ≠ 0.
- The zeros are x0 + 2(B/A)τ ± √(2τ)(1+O(τ)): the midpoint drifts.
- r² = 8τ(1 + κ_r τ + O(τ²)) with κ_r = 4C/A − 2B²/A².
- ℓ = (|A| r³/6)(1 + κ_ℓ τ + O(τ²)) with κ_ℓ = 6B²/A² − 48C/(5A).
- Here A = H_zz/2, B = H_zzz/6, C = H_zzzz/24.
- The zeros are real for τ > 0 whatever the sign of A.
- At t = 0 the fold predictions are off by 7%/17% (r²) and 33%/128% (ℓ) at the two collisions above. "Same cubic relationship" therefore holds only asymptotically near the collision time.

6. "Same cubic relationship as the project" is generic fold geometry: ℓ = (|F'''|/12) r³ (1+O(r²)) for any nondegenerate fold.
- This matches the project's k = |F'''|/12 and fold_controls.py's separation³/6.
- The heat equation adds only automatic time-transversality (∂_tH = −2A ≠ 0) and the opening law r² = 8τ with no free constant.

7. "The new part is expressing collisions in persistence language." This is standard birth–death of a critical pair in a generic one-parameter family: Cerf 1970, and the vineyards of Cohen-Steiner–Edelsbrunner–Morozov 2006. It is not new mathematics.

8. Description of the project (§2). The note's "central theorem" merges distinct results:
- the conditional structural theorem (d ≥ 1, interfaces COUNT/CONTACT/SELECT/ENVELOPE/FAR, positive coefficient required);
- the proved periodized-Gaussian adapter (fixed d ≥ 2 and L, existential constants);
- finite Gaussian laws;
- the author-side 1D theorems on the circle and the line.

"Combined with configuration probability gives the exponent" needs three things: pair intensity ∝ r (the four-factor ledger (d−1)+3−(d+3)+2 = 1), admissible marks, and a positive pairing limit. Cubic splitting alone need not even give a density coefficient (Math- collision_mechanism §D1).

9. Theta link (§4), stated precisely.
- Σ_j e^{−ω²|j|²/2} = Θ(2π/L²)^d, and q_n = e^{−πu|n|²}/Θ(u)^d with u = 2π/L².
- K_L(z) = Π_i ϑ₃(πz_i/L, q)/ϑ₃(0, q) with q = e^{−2π²/L²}.
- The project's image/Fourier identity is Jacobi's transformation at a single u.
- ζ needs the Mellin integral over all u, i.e. all L: ∫_0^∞(N_F(L,1)−1)L^{−s−1}dL = (2π²)^{−s/2}Γ(s/2)ζ(s) for Re s > 1. The project fixes L and disclaims variable-L results (periodized_gaussian_h0/PROOF.md:82-83).
- At L = 24 the periodization changes the jets by about 1e−125. The project uses Poisson summation only to make every Fourier weight positive.

10. Attribution. Θ(u) = u^{−1/2}Θ(1/u) ⇒ ξ(s) = ξ(1−s) is due to Riemann 1859 (his second proof). Mumford's Tata Lectures on Theta I is an expository source.

11. Mellin formula domain. It holds only for Re s > 1; at u → 0 the integrand behaves like u^{(s−3)/2}. Continuation needs Riemann's split at u = 1.

12. "The functional equation does not locate zeros" is right, and should be sharpened.
- Z_4(w) = 8(1−4^{1−w})ζ(w)ζ(w−1) has zeros at ½+iγ and 3/2+iγ, off its symmetry line Re w = 1 (verified at γ_1).
- Davenport–Heilbronn (1936): Epstein zetas of forms with class number > 1 have infinitely many zeros in Re s > 1.
- So theta structure plus a functional equation gives no RH-type statement.
- The heat flow also destroys the Euler product for t ≠ 0.

13. GUE (§5).
- R_2(r) = 1 − (sin πr/πr)² is Montgomery's pair-correlation conjecture.
- Montgomery's theorem (assuming RH) covers only test functions with Fourier support in (−1, 1). The r → 0 law is unproved.
- The bar test should use the nearest-neighbour spacing, whose leading term is the same (π²/3)s². The Wigner surmise gives 32/π² instead.

14. The ℓ⁰ test lacks an admission condition. In ν(ℓ) = (c/3) E[k^{−(β+1)/3}] ℓ^{(β−2)/3}, the case β = 2 needs E[k^{−1}] < ∞.
- Math- collision_mechanism §D2 (α = 2, m = 3) gives the full picture for mark density ∝ k^q near 0: ℓ⁰ if q > 0, log(1/ℓ) if q = 0, and ℓ^q if −1 < q < 0. Exact quadrature check: k ~ U(0,1) gives exactly log(1/ℓ).
- For ζ, k depends on how H is normalized: the persistence of ∫H is not determined by the zeros alone.

15. The "decisive step" (§6) is circular. "No positive-time transition" ⇔ Λ ≤ 0 ⇔ RH. It is equivalent to RH, not merely "closely tied" to it.

16. Off-line zeros (§6) can be detected from real-axis data. Compare the number of real zeros (the critical points of F_t) with N(T) from the argument principle, using Riemann–Siegel and Turing's method. This is how RH is verified numerically (to 3·10^12, Platt–Trudgian). Only a purely local analysis misses off-line zeros.

17. §7(a): what changes the barcode of F_t.
- For t > Λ, zeros are simple and repel, so F_t has no fold events. Under RH every fold event is at t < 0, accumulating at 0⁻.
- On a window, the barcode changes only through births and deaths (real double zeros), zeros crossing the window ends, and pairing switches at equal critical values (which involve no zero event).
- The diagram depends on normalization. Hardy Z versus the Ξ weight shares 80/113 and 82/106 pairs; Ξ's actual sign shares 0; a mild e^{0.01t} weight shares 112/113 and 104/106.

18. §7(b): the global constraints the note asks for already exist.
- de Bruijn's strip lemma: Λ ≤ t0 + y0²/2.
- Newman's monotonicity.
- Ki–Kim–Lee 2009: for every t > 0, all but finitely many zeros are real and simple; Λ < 1/2.
- Rodgers–Tao's energy and local-equilibrium argument (Λ ≥ 0).
- Upper bounds: Polymath15 Λ ≤ 0.22; Platt–Trudgian Λ ≤ 0.2.

None of these is theta-specific beyond Φ > 0, evenness and decay.

19. §7(c). The project's theorems concern random laws on closed manifolds admitted through stated interfaces. They do not apply to the deterministic H_t. Any theorem would need a random model, for example CUE.

20. Implicit in §6: "if Λ > 0, the transition at Λ is a real collision". This is plausible but unverified; it needs Ki–Kim–Lee finiteness to hold locally uniformly in t.

21. Minor (§1). ζ continues meromorphically to ℂ, with only a simple pole at s = 1 of residue 1. The statement that nontrivial zeros lie in 0 < Re s < 1 is a theorem: ζ(1+it) ≠ 0 plus the functional equation.