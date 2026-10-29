# Nonauthor review: D1 Theorem A (parent §§2–7, A1–A7), the cap import, and the §9 Borel repair

Scientific effect: **NONE**. This file does not change `lemma_closed`, prizes, premises, `STATUS`,
`PROOF_INDEX`, `GRAPH` or any author source. It records verdicts. Any integration is for a separate
integrator, and a second non-Claude review is requested.

## Objects reviewed

All paths are on Math- `main` at `aeed37683702e987f8a3cf081acb407bb942f107`.

| Object | Path | Blob | Bytes | SHA256 |
|---|---|---|---|---|
| D1 parent, UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1 (OpenAI / ChatGPT) | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | `dfed3b8d318a3ab1950957f393307733a4bef3f2` | 40261 | `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7` |
| Congruence erratum, …-ERRATUM-1 | `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md` | `213594d6ca6a86fb938110f4d166d9ce275a02d0` | 1782 | `bad7ef609c4ad8c41ad6af562c1b6807921e19a9d556ed793ad1a0db6e202028` |
| Cap import, MARKED-CYLINDER-CAP-20260924-v1 | `imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md` | `0633aca3c2a2882b0de4399da0a75d64c2e6b2e1` | 15160 | `0bf922b9203c29088b12388807aa0e2ecd020485eb0f6e919679841b5b2636fc` |
| §9 repair, D1-SECTION9-BOREL-REPAIR-20260925-v1.1 | `reviews/d1_section9_borel_repair_20260925/REPAIR.md` | `fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a` | 9062 | `845abf9f9c99d672c2a10a887b5a2e7206a3d2de3d876f35f75ff6e2dc13e62f` |

Interface IDs A1–A7 follow the Grok round-3 table (`main/incoming/grok-session-20260926-replay/D1_INTERFACE_TABLE.md`):

- A1 = §2
- A2 = §3
- A3 = §5 with the erratum
- A4 = §4
- A5 = (6.1)
- A6 = (6.2)
- A7 = §7

The earlier D1-A–E review on [main#63](https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841570965)
already accepted §§8–15. Parent line numbers below refer to the blob above.

## Reviewer provenance

| Field | Value |
|---|---|
| Provider / tool | Anthropic Claude, Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3` |
| Relation to author | Different provider from the author (OpenAI). Same GitHub account as every lane, so no organizational independence is claimed. |
| Exposure | Read: the parent, the erratum, the cap proof, `CAP_PAIRING_IDENTITIES.md`, the §9 repair and its reconnaissance memo, and Grok's D1 files (interface table, A2 residual gap, A3 type convergence, A3 UI majorant, `A_M_TO_A0`, Section 7 notes, embedded chart and Morse). Every derivation below was redone. The primary source arXiv:2304.07424v3 was read directly from arxiv.org for C1. |

## Verdicts

| ID | Parent locus | Verdict |
|---|---|---|
| A1 | §2 positive spectrum, jet rank, smooth version | **ACCEPT** |
| A2 | §3 contact frame, uniform covariance gap, density (3.5) | **ACCEPT**. Grok's "single missing estimate" is supplied below. The theorem is existential, so no numerical minorant is required. |
| A3 | §5 (5.1)–(5.5) with the erratum | **ACCEPT**, with one wording amendment (W1) that leaves the proof unchanged. Grok's five open items 1–5 are each discharged. |
| A4 | §4 (4.1)–(4.3) | **ACCEPT** |
| A5 | (6.1), every m | **ACCEPT**. Sharp as a matrix relaxation: the bound is attained under its matrix constraints. |
| A6 | (6.2), every m | **ACCEPT**. Sharp as a matrix relaxation. The 3/2 mixed-square term is necessary for the relaxation. |
| A7 | §7 (7.1)–(7.8) | **ACCEPT**. The r^5 numerators are rederived. For m=1, the majorant event that (7.5) splits contains a non-empty far branch, so the decomposition must keep it (corrected in v2; see Revision history). |
| CAP | Cap theorem §§2–5 (pathwise implication and its constants) | **ACCEPT**, agreeing with Grok's `EMBEDDED_CHART_AND_MORSE.md`. Every rational constant is checked exactly, and identity (11) is checked symbolically for m = 1 and m = 2. |
| Theorem A | (1.1): `1-p_r <= C r^3`, uniform over B, K and frames | **ACCEPT at its stated existential scope.** This composes A1–A7, CAP and the previously accepted §8. |
| §9 C1–C6 | Borel repair v1.1 | **ACCEPT** all six, with two clarifications (N1, N2) that strengthen the text without changing it. The PR112 theorem-number objection is **withdrawn** on primary-source grounds. |

Theorem A is accepted with **no numerical** `C`, `r_*` or `z_*`. It says nothing about SIDE24 numerics, Theorems B/C
beyond §9, D2–D6, or RN/24-jet certificates.

---

## Part I — Theorem A

### A1 (§2) — ACCEPT

- Poisson summation gives `a_n > 0` for every `n`, and `sum sqrt(a_n)(1+|n|)^q < ∞`.
- The random Fourier series therefore converges almost surely and in every `L^p`, in every `C^q`.
- A zero-variance finite combination of derivative functionals annihilates every Fourier mode. As a distribution on
  the torus with all coefficients zero, it is zero. Jets prescribed by bump functions then make every coefficient zero.
- Frames enter only through `P(R^T n)` vanishing on `Z^d`. That reduces to a polynomial vanishing on the lattice,
  which is the zero polynomial.

### A2 (§3) — ACCEPT

**Exact algebra** (`theorem_a_exact_check.py`, group `A2_contact_frame`):

- The axial rows (3.1) have determinant exactly `12 r^-4`.
- Each transverse pair has determinant `r^-1`, so `|det T_r| = 12 r^-(d+3)` (3.2).
- The target (3.3) is exact.

**A stronger form of the "dangerous cancellation" remark (l. 103).** `U3` is exactly a positive average:

```
U3 = ∫_a^c w_r(t) f_xxx(t) dt,   w_r(t) = 6 (t-a)(c-t)/r^3 >= 0,   ∫ w_r = 1.
```

This is an exact polynomial identity for a generic degree-9 profile. Likewise:

- `U1` is the average of `f_x`;
- `U2` is the average of `f_xx`;
- `V_j1` is the average of `f_xy_j`;
- `U0` and `V_j0` are endpoint means.

`U_r - U0*` contains only even powers `r^2, r^4, …`. For example, `U3 = f_xxx + r^2 f^(5)/40 + O(r^4)`, and the
`f^(4)` term cancels.

**Uniform covariance gap: the estimate Grok's A2_RESIDUAL_GAP asks for.**

- Every coordinate of `U_r ⊕ vec A_r` has the form `∫ ∂^α f(tu) dμ_r(t)`, where `μ_r` is a probability measure (a
  point mass or one of the averages above) and `μ_r -> δ_0` as `r -> 0`.
- The covariance entries `∫∫ ∂^α∂^β K_L(R(t-s)) dμ_r dμ_r` are therefore jointly continuous on `[0,r_0] × O(d)`. The
  `r=0` value is the contact covariance of (3.4) plus `f_{y_i y_j}`.
- These functionals are distinct, so by A1 that contact covariance is positive definite at every frame.
- `λ_min` is continuous and `[0,r_0] × O(d)` is compact, so `λ_min >= c_* > 0` on `[0,r_1] × O(d)`.

This is a complete proof of the existential gap. An explicit minorant is needed only for a numerical `r_*`, which
Theorem A does not claim.

**Density (3.5).** The `Q`-law of `A_r` is Gaussian:

- mean `Cov(A_r,U_r) Σ_r^{-1} v_r`, which is bounded;
- covariance equal to a Schur complement, lying between `c_*` and `Var(A_r)`.

Hence `density <= C_0 exp(-c_0 ||A||_F^2)`.

### A4 (§4) — ACCEPT

- **(4.1).** Covariance Cauchy–Schwarz gives `|E_Q ξ| <= (v_r^T Σ_r^{-1} v_r)^{1/2}` and `Var_Q ξ <= 1`. Minkowski over
  the summable Fourier coefficients then gives (4.1).
- **(4.2).** Under `Q`, the pair `(f, A_r)` is jointly Gaussian.
  - Put `g_r := f - E_Q[f | A_r]`, with `E_Q[f|A_r] = μ_r + B_r·(A_r - E_Q A_r)`.
  - `B_r(z) = Cov_Q(f(z),A_r) Cov_Q(A_r)^{-1}` is smooth in `z`. Its `C^4` norm is at most `c_*^{-1}` times a
    bounded cross-covariance.
  - `g_r` is uncorrelated with `A_r`, so it is independent of `A_r` on a countable dense set of evaluations. By
    continuity it is independent as a smooth field.
- **(4.3).** `M3 <= ||μ_r||_{C^3} + ||B_r||_{C^3} ||A_r - E A_r|| + ||g_r||_{C^3}`. Partial-block operator norms
  differ from coordinate norms by factors that depend only on `d`. `J_r = 1 + ||g_r||_{C^4}` has all moments because
  `g_r` is a bounded linear image of `(f, A_r)`.

### A3 (§5 plus erratum) — ACCEPT, amendment W1

**Exact identities** (group `A3_endpoint_and_erratum`), each an exact identity for a generic degree-9 profile:

- Trapezoid/Hermite: `f(c)-f(a) = (r/2)(f'(a)+f'(c)) - ½∫(t-a)(c-t) f''' dt`, with kernel mass `r^3/6`.
- `f'(c)-f'(a) = r f''(a) + ∫(c-s)f''' = r f''(c) - ∫(s-a)f'''`, with kernel masses `r^2/2`.

Under the pins, `α_M = -(1/r^2)∫(c-s)f'''` and `α_S = (1/r^2)∫(s-a)f'''`. The height pin makes the
`6(t-a)(c-t)/r^3` average of `f'''` exactly `12k`. All three averages lie in the range of `f'''` on `[a,c]`, and
that range has width `<= r M4`. This gives (5.2): `|α_M + 6k|, |α_S - 6k| <= r M4 / 2`.

(5.1) uses `(1/r)∫|t-a| dt = r/2`, applied to `f_xx` and to the vector `grad_y f_x`. Both have zero average by the
gradient pins, so no simultaneous vector Rolle point is used.

**Block determinant.** (5.3) `det H = r(α det A - r β^T adj(A) β)` holds as a symbolic identity for m = 1, 2, 3.

**Congruence.**

- With `D_r = diag(r^{-1/2}, I)`, the scaled matrix has entries `[[α, √r β^T], [√r β, A]]` and
  `det(D_r H D_r) = det H / r`. These are the erratum's claims.
- The displayed `diag(√r, I)` would put `α r^2` in the corner, which has a singular limit. This confirms Grok's C1.

**Grok's five open items (`A3_TYPE_CONVERGENCE.md`):**

1. *Erratum in tree.* Landed by Math- PR64 (`d8f5505…`). Read with the parent here.
2. *Residual moments bounding β and M3.* Accepted under A4.
3. *`A_M -> A_0` in law, uniformly.* `A_M` is Gaussian under `Q`, with mean `Cov(A_r,U_r)Σ_r^{-1}v_r` and a
   Schur-complement covariance. Both are jointly continuous in `(r,b,k,R)` up to `r=0`, where they equal those of
   `A_0` (`U_0 = U0*`, `v_0 = (b,0,0,12k,0,…)`). Convergence in law is therefore locally uniform on the compact
   parameter set.
4. *Uniform integrability.*
   - By (5.3), `|det H_i / r| <= (M3/2)||A_i||^m + C r M3^2 ||A_i||^{m-1}`.
   - `||A_S|| <= ||A_M|| + r M3`.
   - With A4, `sup E (W_r/r^2)^2 < ∞`, which gives UI.
5. *`P(det A_0 = 0) = 0`.* `A_0` is a nondegenerate Gaussian on `Sym_m` (the covariance is a Schur complement of a
   positive definite matrix). Its law is absolutely continuous, and `{det = 0}` is the zero set of a nonzero
   polynomial.

`z_0 > 0` holds because the Gaussian density is positive on the open set `{A<0}`, where `det^2 > 0`. `z_0` is
continuous in the parameters by dominated convergence, so compactness gives (5.5).

**W1 (wording; proof unchanged).** Parent l. 165 says the congruence-scaled Hessians "converge, in probability" to
`diag(∓6k, A_0)`. `A_0` is a limit in law of a different random matrix, so the correct statement is:

- `D_rH_MD_r - diag(-6k, A_M)` and `D_rH_SD_r - diag(6k, A_M)` tend to 0 in probability, by (5.1), (5.2) and
  `||A_S - A_M|| <= rM3`;
- and `A_M -> A_0` in law.

Slutsky then gives joint convergence in law, and the continuous-mapping theorem applies. The discontinuity set lies
in `{det A_0 = 0}`, which is null. The parent's subsequence paragraph (l. 176) already carries out exactly this
in-law argument.

### A5 (6.1) — ACCEPT, sharp as a matrix relaxation

**Derivation.**

- Schur on the `(1,1)` entry, together with the determinant lemma, gives `det H_M = rα det(A - (r/α)ββ^T)`. This is
  checked symbolically for m = 1, 2, 3.
- On `H_M < 0`: `α < 0`, and `S := B - (r/|α|)ββ^T` is positive definite. Since `S <= B`, `det S <= det B`.
- Hence `|det H_M| = r|α| det S <= r|α| det B <= r(h/2) det B`.

This holds for every m; Grok had closed only m=1.

**Probe.** The exact-rational probe checks both the sharper `r|α| det B` and the stated `r(h/2) det B`:

- 700 instances per m ∈ {1,2,3} (1,962 typed maxima in total);
- 0 violations;
- equality at `β_M = 0`, `α_M = -h/2`.

### A6 (6.2) — ACCEPT, sharp as a matrix relaxation; the 3/2 term is necessary

**Derivation.**

- Weyl's inequality for singular values, with `||A_S - A_M|| <= rh`, gives `σ_j(A_S) <= λ_j + rh`.
- `||adj A_S||` is the product of the largest `m-1` singular values.
- (5.3), together with `|α_S|, ||β_S|| <= h/2`, then gives
  `(h/2)∏(λ_j+rh) + r(h^2/4)∏_{j>=2}(λ_j+rh) = (h/2)[λ_1 + (3/2)rh]∏_{j>=2}(λ_j+rh)`.

The last identity and the product (6.2) are both checked symbolically.

**Sharpness.** Take:

- `A_S = -(B + rhI)`;
- `α_S = h/2`;
- `β_S = (h/2)v_1`, where `v_1` is the λ_1-eigenvector.

Then `|det H_S| = r(h/2)[λ_1 + (3/2)rh]∏_{j>=2}(λ_j+rh)` with **equality**. This happens in all 300 extremal probe
instances (100 per m). So (6.2) cannot be improved from the hypotheses (5.1) and `||A_S - A_M|| <= rh` alone.

- Replacing 3/2 by 1 (mutant `mixed-square`) fails 601 exact checks.
- Dropping the `+rh` shift for `j >= 2` (mutant `no-shift`) fails 401 exact checks, all with m ≥ 2.

**Scope of this sharpness (v2).** It is sharpness of the *matrix relaxation*: under the displayed matrix constraints
alone, (6.1) and (6.2) cannot be improved, so a tighter deterministic weight bound cannot come from those constraints.
It is **not** a lower bound on the Gaussian failure probability. It does not show that every extremizer is realized
by a field with the same derivative-supremum budget, and it does not show that the `r^3` rate of Theorem A is
attained.

### A7 (§7) — ACCEPT

- **(7.1).** Depth failure is `λ_1 <= (4/(3k)) r M3^2`. With `k >= k_-` and `M3 <= K_0(J + √m Λ) <= K U`, this gives
  `λ_1 <= D r U^2` with `D = 4K^2/(3k_-)`.
- **(7.2).** `||A||_F^2 = Σ λ_i^2` is orthogonally invariant, so the majorant (3.5) passes to eigenvalue coordinates
  without any invariance of the law. The Vandermonde factor satisfies `∏|λ_j-λ_i| <= Λ^{m(m-1)/2}`.
- **(7.3).** An exact identity (checked).
- **Ledger for m ≥ 2.**
  - Prefactor: `r^2 h^2/4 · ∏_{j>=2} λ_j(λ_j+rh) <= C r^2 U^{2m}`.
  - Integral: `C r^3 U^6`, using `U >= 1`.
  - Numerator: `r^5 U^{2m+6}`, which matches the exponent in (7.4).
  - `U = J + λ_m` is not a function of `λ_1` when m ≥ 2, so extending `λ_1` to `[0, DrU^2]` is legitimate.
  - `λ_2, …, λ_m` range over their whole ordered domain, so corank ≥ 2 is included.
- **m = 1, (7.5).**
  - `(J+λ)^2 <= 2J^2 + 2λ^2` is exact (the difference is `(J-λ)^2`).
  - If `4Drλ <= 1`, then `λ(1-2Drλ) >= λ/2`, which forces `λ <= 4DrJ^2`.
  - 8,000 exact random candidates confirm the split of the majorant event `{λ <= Dr(J+λ)^2}` (4,000 random λ and
    4,000 with λ ≥ 1/(Dr)).
  - Every `λ >= 1/(Dr)` lies in that **majorant event**, since `Dr(J+λ)^2 >= Drλ^2 >= λ`. So the majorant event
    contains a non-empty far branch for every J, and **this upper-bound decomposition** must keep it; mutant
    `drop-far-branch` is refuted.
  - **Correction (v2).** v1 said such λ "fail the depth condition". That reverses the implication. The source uses
    only depth failure ⟹ majorant membership, and `M3 <= K(J+λ)` is only an upper bound. Nothing here shows that
    *actual* depth failure `λ <= [4/(3k)] r M3^2` occurs on the far branch. The OpenAI review's example
    `k=K=1, D=4/3, r=1/1000, J=M3=12, λ=750` lies in the majorant, but the actual threshold is `24/125 = 0.192`.
    It is checked exactly as `majorant_membership_does_not_imply_depth_failure`.
  - Near branch: with `h <= K(1+4D)J^2`, the exact integral gives a numerator of order `r^5 J^10`, as claimed.
- **(7.6) and (7.7).** `1_far <= (4Drλ)^4` and `1{rM4 > 3k_-/10} <= (10rM4/(3k_-))^4` are pointwise. The joint moments
  are finite by A3 and A4, so both numerators are `O(r^6)`.
  - `G_r` uses `3k/10`, and `{rM4 > 3k/10} ⊂ {rM4 > 3k_-/10}`.
- **(7.8).** Dividing by `Z_r >= z_* r^2` gives `C_3 r^3 + C_4 r^4`.

### CAP (cap theorem §§2–5) — ACCEPT

All of the following are exact (group `CAP_marked_cylinder`):

- kernel mass `r^3/6`, and average 2 at κ = 1/6, so `m >= 2`;
- product distance `<= 9r/2 < 5r`;
- `f_xxx >= 7/4`;
- `δ > rm(8m-5) >= 22r`;
- `max_{|x|<=2r}|x^2 - r^2/4| = 15r^2/4`, and the averaged `|x-t|` kernel has maximum `2r`;
- `u(11) = 24/121` and `m·u = (1+6/k+5/k^2)/4 = 48/121`, both decreasing for `k >= 11`;
- `(48/121)(3 + 3u + u^2) = 2554128/1771561`, and `F'' > 2184415/7086244 > 1/4`;
- the exterior integrals of `(x-a)(x-c)/8` equal `9r^3/32` on both sides, and `b - 9r^3/32 = s - 11r^3/96`;
- `(δ/2)(20r/11)^2 > 4400r^3/121 > r^3/6`;
- the κ-rescaling gives `4/(3κ)`, `3κ/10`, `3κ/2`, `27κ/16`, `11κ/16` and `26400κ/121`, which match `G_r` in the parent.

**Identity (11).** `F'' = f_xxx + 3f_xxy[h'] + 3f_xyy[h',h'] + f_yyy[h',h',h']`. It is checked symbolically on
polynomial families with explicit ridge `y = q(x)`, both for scalar transverse `y` and for 2-vector transverse `y`.
The check confirms that `F'' ≠ f_xxx` on the family (the correction terms are live), and a coefficient-2 mutant is
refuted.

**Pathwise logic.**

- `g <= b` on `C`, because `F > 0` left of `a` and `F < 0` on `(a,c)`.
- Every boundary exit of `C` is at level `<= s`.
- The ridge from `M` through `S` to `(2r, h(2r))` has minimum exactly `s` and an endpoint above `b`.

Hence the maximin level is `s`, and the elder partner is `S`.

### Composition

§8 (accepted in D1-A–E, and again in Grok's `EMBEDDED_CHART_AND_MORSE.md`) gives the pinned Morse/distinct-value
locus and Borel measurability. CAP gives global elder pairing on `G_r` there. Therefore
`1 - p_r <= Q^W(G_r^c) <= C r^3` (1.1), uniformly over `B × K × O(d)`, with existential constants.

---

## Part II — §9 Borel repair v1.1 (C1–C6)

**C1 (theorem attribution) — ACCEPT. The PR112 theorem-number objection is withdrawn.** arXiv:2304.07424v3
(5 Dec 2023), read directly:

- **Theorem 2.1** (Rice formula for the expectation) assumes: (i) `C^1` paths; (ii) a continuous, locally uniformly
  bounded density of `X(t)`; (iii) the conditional law given `X(t)=v` is continuous in `v` for the `C^1` topology.
- **Theorem 2.2** is the Gaussian version. **Remark 7** says it follows from 2.1 "once the conditional distributions
  are defined by means of regression formulas".
- **Theorem 6.1** is Crofton's formula.
- **Theorem 7.1** (Expected integral on the level set, formula (7.2)) is stated under the hypotheses of Theorem 2.1,
  with a weight `g(t, Z(·))` that is (a) lower semicontinuous in `t` and (b) lower semicontinuous in `Z(·)`, and with
  (c) continuity of the conditional law of `Z` given `X(t)=v`.
- **Remark 8** says (c) is automatic when `X` and `Z` are jointly Gaussian.

The repair's numbering is correct.

**C2 (C²-valued regular kernel) — ACCEPT.**

- `R_t = f - Cov(f,G(t))Cov(G(t))^{-1}G(t)` is a Gaussian random element of the separable Banach space `C^2(X)`,
  independent of `G(t)`.
- `t ↦ R_t` is pathwise continuous in `C^2`, and `m_{t,z}` is jointly continuous.
- By dominated convergence, `(t,z) ↦ ∫F dK(t,z,·)` is therefore continuous for **every** bounded continuous `F` on
  `C^2(X)`, not only for cylinder functions. The repair asks only for the cylinder case, which is weaker.

**C3 (finiteness and equality on continuous cylinders) — ACCEPT.**

- `D` is compact inside the open off-diagonal set, and `Cov G` is uniformly nondegenerate there. The Theorem 2.2
  intensity is continuous and bounded on `D`, so `μ(D×E) < ∞`.
- `ν(D×E)` equals the same integral.
- For a bounded nonnegative continuous `h(t, ∂^{α_1}φ(q_1), …)` with fixed sites `q_i`, take `Z` to be the constant
  vector of those evaluations. Hypotheses (a) and (b) hold by continuity, and (c) holds by Remark 8.

**C4 (generation and monotone class) — ACCEPT.**

- The `C^2` distance `||φ - φ_0||` is a supremum over the countable dense set `Q` of `|∂^αφ(q) - ∂^αφ_0(q)|`, so balls
  are measurable for the coordinate σ-algebra. In a separable metric space every open set is a countable union of
  balls, and `Borel(D×E) = Borel(D) ⊗ Borel(E)`.
- The class of bounded continuous cylinder functions is closed under products, which is the multiplicative-class
  hypothesis of the functional monotone-class theorem.
- `μ` and `ν` are finite, so dominated convergence gives closure under bounded pointwise limits.
- The constants belong to the class by the unweighted formula.

**C5 (elder/type mark, determinant accounting) — ACCEPT, with clarification N1.** There is one `Δ`. Disintegrating the
two heights conditional on `G(t) = 0` gives the full-pin density `× Z_r × p_r` at `(b, b - kr^3)`. There is no second
determinant.

**N1 (strengthening).** The mark is Borel on **all** of `D × E`, not only on the Morse locus:

- Work in the finitely many coordinate charts of parent §8. For each chart and each rational polygon template `γ`,
  let `γ_x` run straight from `x` to the template's first vertex and then through its remaining vertices. The functional
  `T_γ(x,φ) = min φ∘γ_x` if `φ(end) > φ(x)`, and `-∞` otherwise, has open superlevel sets.
- Rational polygonal approximation of any admissible path gives `d_φ(x) = sup_γ T_γ(x,φ)`. So `d` is lower
  semicontinuous, and `{d_φ(x) = φ(y)}` is Borel.
- The Morse distinct-value locus is needed only to *interpret* this mark as the elder partner. It has full `μ`
  measure (unconditional genericity) and full `ν` measure (the §8 fixed-pin argument, applied to `K(t,0,·)` for each
  `t`).

**C6 (scope boundary) — ACCEPT.** The scope matches §3 of the repair. Exhausting `D` toward the diagonal imports parent
§10 and is not re-reviewed here.

**N2.** Theorem 7.1 is stated under Theorem 2.1, not under 2.2. For this Gaussian `G`, Remark 7 is the bridge. A
one-line citation of Remark 7 would make the attribution airtight.

---

## Checks run

```
python -B -S theorem_a_exact_check.py                         # rc 0
python -B -O -S theorem_a_exact_check.py                      # rc 0, byte-identical to RESULTS.json
python -B -S theorem_a_exact_check.py --mutant u3-row             # rc 1
python -B -S theorem_a_exact_check.py --mutant bordered-det       # rc 1
python -B -S theorem_a_exact_check.py --mutant mixed-square       # rc 1
python -B -S theorem_a_exact_check.py --mutant drop-far-branch    # rc 1
python -B -S theorem_a_exact_check.py --mutant ridge-coefficient  # rc 1
python -B -S theorem_a_exact_check.py --mutant pivot-quarter      # rc 1
python -B -S theorem_a_exact_check.py --mutant no-shift           # rc 1
```

Python standard library only. The checks are exact rational and polynomial identities plus one exact-rational
falsification probe. They are not a Gaussian or continuum proof; the probabilistic steps are the written arguments
above.

## Not established here

- Numerical `C`, `r_*`, `z_*`, or any SIDE24 finite-band constant.
- Theorem B's quantitative difference and Theorem C beyond the §9 interface. D1-A–E already covers §§10–15.
- D2 remainder, D3 coefficient, D4 RN count, D5 global, D6 P15.
- Organizational independence: all lanes share one GitHub account.

A second, non-Claude review of Part I is still requested before any status integration. The OpenAI review below is
exposed-source (OpenAI authored the parent), so it is not an independent acceptance. An xAI/Grok check of W1 and A2
remains welcome.

## Revision history

- **v1**, head `ce975e6`: initial packet.
- **v2**, responding to the OpenAI technical review
  [5341593068](https://github.com/d6g8k5htny-coder/Math-/pull/106#pullrequestreview-5341593068) at `ce975e6`:
  - That review confirms W1. It independently reproduces the A5/A6 extremal fixtures for m = 1..5, the failure of the
    missing 3/2 term, the Hermite-kernel identities and the congruence cases.
  - A7 correction: "every λ ≥ 1/(Dr) fails the depth condition" is replaced by "lies in the majorant event". The
    result keys are renamed to `majorant_event_contains_far_branch_l_ge_1_over_Dr` and
    `7.5_near_far_split_of_majorant_exact_random_4000`. The reviewer's example is added as the exact check
    `majorant_membership_does_not_imply_depth_failure`.
  - The A5/A6 sharpness claim is restricted to the matrix relaxation. It is no longer a statement about the Gaussian
    failure probability or the `r^3` rate.
  - Every verdict is unchanged. The parent's upper bound (7.5)–(7.8) is unaffected, because it uses only depth
    failure ⟹ majorant membership.
