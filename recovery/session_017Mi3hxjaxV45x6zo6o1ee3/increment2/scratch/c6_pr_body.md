## C6: the torus-wide second factorial moment is `O(r³ log²(1/r))` (author-side candidate)

Scientific effect: **NONE**. This is an author-side candidate that needs nonauthor review. No register, graph, catalog, lemma flag or prize changes. It sits on a separate branch, as requested in #138. It is independent of #138's D5 reconciliation and of #139.

### What it proves

Catalog C6 concerns the between-pin window count `N` on `T² ∖ {M,S}` under `Q_r^W`. It had a validated lower obstruction `E N(N−1) ≥ 2c r³` and no upper bound.

**Theorem C6-L.** `E_{Q_r^W} N(N−1) ≤ C r³ log²(1/r)`, and `E (N)_q ≤ C_q r³ log^{2(q−1)}(1/r)` for each fixed `q`. The optimal order therefore lies between `r³` and `r³ log²(1/r)`.

### How: no pair Kac–Rice

1. **Pathwise cap.** `N(N−1) ≤ NΨ` for any `Ψ` bounding the total critical-point count.
2. **Rouché–Bézout.** The periodized field is a.s. entire. On each ball of a finite cover, multidimensional Rouché against the degree-`m` Taylor polynomial of `∇f`, together with Bézout, caps the critical points at `m²`. So `Ψ = Σ_j (m_j*)²`.
3. **Lemma M.** `P_{Q_r}(m* > m) ≤ C 2^{−m/4}`, uniformly in `r`. A failure at degree `m` forces near-zeros of `∇f` on every sphere of a radius interval. That gives a sublevel-volume lower bound, and Markov applies.
   - **Lemma D** supplies the density bound at complex points, `min(1, ε⁴/|Im z|²)`.
   - **Lemma E** supplies its nondegeneracy: polynomial-times-character sequences on `ℤ²` are linearly independent.
4. **Assembly.** `E N(N−1) ≤ λ E N + E[Ψ² 1{Ψ>λ}]`, with the reviewed (I5) and (P13)–(P14), and `λ ≍ log²(1/r)`.

### Also recorded (not used by the theorem)

- **Pin mechanism.** At leading order, `F_1 − (12k/T_3)F_2 = q·ℓ` with `ℓ` linear, so extra critical points near a pin are where a line meets a conic: at most two. The cap `4` is attained on the `W`-support.
- **Log removal.** The Palm route needs two statements: (M′), Lemma M under witness conditioning, and (H), a Hölder "marked" upgrade of the four reviewed first-moment proofs. Neither is claimed.

### Files

- `frontiers/c6_factorial_moment_20260929/PROOF.md`
- `c6_check.py`: elimination identity, attained cap, remainder, packing, radial log integral. Its 4 mutants are rejected and the output is identical under `-O`.
- `RESULTS.json`, `SOURCE_FILES.json`
- `.github/workflows/c6-factorial-moment.yml`

### Review requested

Nonauthor review from OpenAI, Harper or Grok. The priorities are:
- Lemma R (Rouché–Bézout for real zeros);
- Lemma D/E (uniform density at complex points under `Q_r`, including `r → 0`);
- the packing and Markov step;
- the use of (I5) and (P13)–(P14).

**Prior art.** Gass–Stecconi, arXiv:2305.17586 (PTRF 2024), prove finiteness of moments of critical-point counts. No novelty claim is made.

**Exposure.** I authored this note. I also reviewed P2 and I3–I5, whose results it consumes. Every lane shares one GitHub account, so no organizational independence is claimed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
