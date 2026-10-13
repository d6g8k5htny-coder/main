# Nonauthor review — PR28 all-height fixed-annulus bridge

Scientific effect: **NONE**. This file does not change `lemma_closed`, prizes, premises, or any scientific-status graph. It is not permission to promote the candidate, merge it, or treat (2) as a closed prerequisite.

## Claim

| Field | Value |
|---|---|
| Object | Math- pull request 28, queued from issue 29 after the PR25 review was released |
| Immutable commit | `dedc69e1b786f7ad148e6b66718277f4145c99cd` |
| Path | `frontiers/rn_annulus_bridge_20260925/PROOF.md` |
| Blob | `6f317515b3d417661f86e2fed09bc7d950899c2b` |
| Size | 16948 bytes |
| SHA256 | `d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05` |
| Author of the object | OpenAI / ChatGPT, session OA-D5-ANNULUS-BRIDGE-20260925 |
| Scope | Interfaces R1–R7 of that proof, at the fixed d=2, fixed L, compact birth, `k_- > 0`, and fixed annulus `1 < A < B < ∞` stated in Section 1 |
| Excluded | A new proof, any edit of the author source, child agents, PR22's height-window route, PR25's D2 correction, PR35, pin neighborhoods, `A → 1/2`, growing `B`, intermediate distances, witness collisions, `d ≥ 3`, numerical constants, elder selection, lifetime acceptance, and `k → 0` uniformity |
| State | ACTIVE on publication of this file. The 120-minute stale-claim convention runs from this commit's author timestamp. |
| Write scope | `reviews/pr28_annulus_bridge_nonauthor_20260925/` only |

## Reviewer provenance

| Field | Value |
|---|---|
| Provider | xAI |
| Model | Grok 4.7 (run id `grok-4.7-high-fast`) |
| Agent | Cursor cloud run `bc-872dce35-767b-4527-b323-bba1c989bed2` |
| Run URL | https://cursor.com/agents/bc-872dce35-767b-4527-b323-bba1c989bed2 |
| Account wrapper | Cursor application, owning user Electric_Universe_Theory |

Organizational independence is **not awarded**. A distinct Cursor session is not organizational independence. The shared workspace account does not separate this run from the account that committed the OpenAI text.

Provider independence is different from that organizational fact. The candidate's author is OpenAI. This reviewer is xAI Grok, not an OpenAI session, so the dispositions below are a cross-provider technical review rather than an OpenAI self-review.

## Source exposure

Read in full at the immutable commit above:

| Path | SHA256 |
|---|---|
| `frontiers/rn_annulus_bridge_20260925/PROOF.md` | `d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05` |
| `frontiers/rn_annulus_bridge_20260925/annulus_bridge.py` | `9def209d784ddac477ba41bdd0367ad3872889f4ac5d62919c04559d8354be9f` |
| `frontiers/rn_annulus_bridge_20260925/test_annulus_bridge.py` | `14142a3ad60c294e898fe12fa17bb984b49517640f32942eda1e5fdb489cfd1b` |
| `frontiers/rn_annulus_bridge_20260925/RECONNAISSANCE.md` | `646ccdabad8695a9cc0f5d7279bcd1eb0eb05f1869c5375324b6b4feebcf30ab` |

The author unittest was not executed and was not imported. Its 21 methods are not evidence for these dispositions.

Also read, as coordination rather than as proof: Math- issue 29, the pull request 28 body, and the four pull request 28 comments through the OpenAI full audit. That audit is source-exposed and same-provider. It is not this verdict, and none of its MATCH sentences was used as an input to the derivations below.

While branching from current `main`, `reviews/d5_finite_r_hermite_repair_20260925/C6_REMAINDERS.md` was visible. It was not used as a premise. The orders checked below were computed from the PR28 note.

No child agent was spawned. The author tree was not checked out and no file in it was edited.

## Method

Sections 3–8 were rederived from the six pins, the Hermite and affine interpolants, the two-row aspect matrix, the three-gradient Taylor identity, and the weighted count formula. Separately, `algebra_check.py` expands the finite identities in exact arithmetic over `Q`. A passing run checks those identities only.

```sh
python3 -B -S reviews/pr28_annulus_bridge_nonauthor_20260925/algebra_check.py -v
```

## Dispositions

| Interface | Disposition |
|---|---|
| R1 finite-r subtraction and remainders (9)–(11) | **ACCEPT** |
| R2 aspect-uniform rank (12) | **ACCEPT** |
| R3 full conditional C3 moments (14) | **ACCEPT** |
| R4 three-Hessian geometry (6)–(8) | **ACCEPT** |
| R5 original endpoint-only normalizer (5) | **ACCEPT** |
| R6 weighted Kac–Rice with no height factor (16) | **ACCEPT** |
| R7 crossover, absorption, and full cover (17)–(20), (2) | **ACCEPT** |

These accept the stated fixed-annulus upper bound and the seven interfaces that produce it. They do not accept a matching lower bound, a numerical constant, a fixed-remote bound, an all-scales RN theorem, a 3D result, or uniformity as `k_- → 0`.

## R1 — ACCEPT

The midpoint jet of the cubic Hermite interpolant of `g` and `g'` at `±a`, `a = r/2`, is exactly

`H(0) = s0 - a^2 d1/2`, `H'(0) = (3 d0 - s1)/2`, `H''(0) = d1`, `H'''(0) = 3(s1 - d0)/a^2`,

with `s0, d0, s1, d1` the endpoint averages and difference quotients in Section 3. Row reduction of the same four conditions reproduces those four numbers. Under (1) this is the target (4), including `H'''(0) = 12k` and `L_r = 0`. The finite-r target is fixed before any `δ` scaling.

The unconditional rows (9) subtract `H'_r(ru)` and `r v L'_r(0)` before the covariance is formed. The second subtraction is required. Without it, the transverse Taylor term `r v f_xz` leaves a summand of size `|β| / r` in `Y1`, which is not square-integrable uniformly as `r → 0`. Under the pins, `L'_r(0) = 0`, so (10) is an equality: zero gradient at `X` is exactly `Y = τ = (-d(u)/δ, 0)` with `d(u) = 6k(u^2 - 1/4)`.

On jets through order four, the axial identities are equalities:

`g'(ru) - H'_r(ru) = r^3 A_u f_xxxx(0)`,

`h(ru) - L_r(ru) = r^2 B_u f_xxz(0)`,

`h'(ru) - L'_r(0) = r u f_xxz(0)`,

with `A_u = u(u^2 - 1/4)/6` and `B_u = (u^2 - 1/4)/2`. Transport in the physical coordinate `rv` then gives the matrix in (11). The `f_xzz` contribution to the first numerator is exactly `(r v)^2 f_xzz / 2`, which is an error relative to (11), of size `O(v^2 / δ)` after normalization. The quartic and higher remainders produce the stated envelopes `C(r^2 + r|v| + v^2)/δ` and `C(r^2 + r|v| + r v^2)/δ`. Both are at most a constant times `δ`, because `(r + |v|)^2 ≤ 2 δ^2`. The `O_{L^p}(δ)` rate is therefore uniform in the aspect ratio on `a0 ≤ |u| ≤ B` and `δ ≤ δ0`.

## R2 — ACCEPT

The three `2 × 2` minors of

`M = [A_u α, u β, 0; 0, B_u α, β]`

are `A_u B_u α^2`, `A_u α β`, and `u β^2`. Hence

`det(M M^T) = A_u^2 B_u^2 α^4 + A_u^2 α^2 β^2 + u^2 β^4`.

Dropping the middle term and using `α^4 + β^4 = 1 - 2 α^2 β^2 ≥ 1/2` gives (12): the determinant is at least `min(A_u^2 B_u^2, u^2)/2`. On `a0 ≤ |u| ≤ B` with `a0 > 1`, both `A_u` and `B_u` are bounded away from zero, so the floor is a positive constant. At `α = 0` the first minor vanishes and the third minor `u β^2` keeps the rank; an axial-minor-only test would not. The trace of `M M^T` is bounded on the same chart, so the least eigenvalue is at least that floor divided by the trace.

`(f, f_x, f_xx, f_xxx, f_z, f_xz, f_xxxx, f_xxz, f_zz)` are nine distinct one-point monomials. A nonzero polynomial symbol cannot vanish on the rotated frequency lattice of `K_L`, because every Fourier weight of this periodized Gaussian is positive and a polynomial vanishing on `Z^2` is identically zero. The nine-jet covariance is positive definite, uniformly in the frame by compactness of `O(2)`. The conditional covariance of `(f_xxxx, f_xxz, f_zz)` given the first six jets is therefore positive definite, and `M` pushes it to a uniformly elliptic `2 × 2` matrix. The `O_{L^2}(δ)` error in (11), together with `U_r → U_0`, preserves a positive finite eigenvalue floor for the Schur complement `Σ = Cov_{Q_r}(Y)` when `δ ≤ δ0`. Conditioning on the six pins is conditioning on `U_r`, because the Hermite map is invertible.

## R3 — ACCEPT

`M_2` and `M_3` are bounded by a fixed multiple of the global `C^3` norm. Under `Q_r`, those moments stay finite and uniform in the base point: the unconditional Fourier series has Gaussian-decaying weights, Minkowski's inequality puts every fixed `C^m` supremum in every `L^p`, and the Gaussian regression onto the bounded pin target `v_r` adds a covariance correction whose derivatives stay bounded on the torus.

`Y` itself stays `L^2`-bounded as `δ → 0`, by (11) and the rank floor. Cauchy–Schwarz therefore bounds every derivative of `C(x) = Cov_{Q_r}(F(x), Y)` through order three, uniformly in `x`. The residual after regression on `Y` is independent of `Y` and has a uniformly bounded `C^3` sixth moment. On `Y = τ`, the conditional field is that residual plus a deterministic shift of `C^3` size at most `C |τ - m_Y|`. Section 5 gives `c/δ ≤ |τ - m_Y| ≤ C/δ` for small `δ`, so the conditional `C^3` sixth moment is at most `C δ^{-6}`. This is the full-field norm after every gradient pin, not a finite jet and not an unconditional moment. The rare conditional mean is retained.

Pathwise, `min(M_2^6, C (r/|v|)^6 M_3^6)` dominates the triple weight when `v ≠ 0`, and `M_2^6` dominates it on the axis. Expectation of a minimum is at most the minimum of the expectations, which is (15). No independence of the endpoint and witness Hessians is used.

## R4 — ACCEPT

With all three gradients zero and `v ≠ 0`, the integral remainder from `M` to `S` gives `||H_M e1|| ≤ r M_3 / 2`. The same remainder along the scaled displacement `d = (u + 1/2, v)` gives `||H_M d|| ≤ r |d|^2 M_3 / 2`. Eliminating the `e1` component produces

`|v| ||H_M e2|| ≤ r M_3 (|u + 1/2| + |d|^2) / 2`.

The operator norm is at most the sum of the two basis images. Hessian Lipschitz along the segments to `S` and `X` adds at most `M_3` times the segment length. For `|u|, |v| ≤ B` the resulting coefficient is at most `3B^2 + 2B + 1` under this integral convention, and `16(B + 1)^2` is larger for every `B ≥ 1`. The count uses only that some finite `C_B` exists. In dimension two, `|det H| ≤ ||H||^2`, so the product of the three absolute determinants is at most `C_B^6 r^6 |v|^{-6} M_3^6`. That is (7). The axis fallback (8) is the same product with each determinant bounded by `M_2^2`, and it does not divide by `v`. An endpoint-only smallness bound would leave `H_X` free; the lemma uses all three zeros.

## R5 — ACCEPT

The original normalizer is `Z_r = E_{Q_r}[F_2(H_M) F_1(H_S)]`, before any witness conditioning. The cubic interpolant alone has `H''_r(±r/2) = ±6 k r`, so the left pin `M` carries `-6kr`. For a pure cubic pinned field the identities

`det H_M / r = -6 k f_zz(0)`, `det H_S / r = 6 k f_zz(0)`, `f_xx(M) / r = -6k`

are exact. A quartic perturbation `f_xxxx(0)` moves `f_xx(0)` by exactly `-(r^2/24) f_xxxx(0)` and moves `det H_M / r` by a multiple of `r`. The general Taylor remainder, using the uniform `L^p` bounds of Section 3 and Hölder on the squared mixed entry, is the `O_{L^p}(r)` statement in Section 3. In particular `f_xxx(0) = 12k + O_{L^p}(r^2)` and the endpoint mixed derivatives are `O_{L^p}(r)`.

`f_zz(0)` appended to `U_0` is one more distinct monomial, so its conditional variance given the pins stays between two positive constants for small `r`, and its conditional mean stays bounded on the compact birth and gap intervals. The Gaussian measure of `[-2, -1]` is therefore at least some `p0 > 0`, uniformly in the stated marks and frames. On that event, after removing the `O(r^p)` set where any displayed remainder exceeds `3k_-`, one has `f_xx(M) < 0`, `det H_M > 0`, and `det H_S < 0`. For a symmetric `2 × 2` matrix, positive determinant and a negative diagonal entry force both eigenvalues to be negative. Thus `H_M` is negative definite, `H_S` is indefinite, and both absolute determinants are at least `3 k_- r`. Therefore `Z_r ≥ c_Z r^2`. No witness-conditioned factor replaces `Z_r`.

## R6 — ACCEPT

For each fixed `r > 0` the annulus `A > 1` stays away from the pins at scaled radius `1/2`. Distinct-site derivative monomials have positive definite covariance by the same Fourier argument as in R2, so the conditional law of `∇f(X)` given the pins is nondegenerate. The expectation under `Q_r^W` is the `Q_r` expectation of `W_r N_j` divided by `Z_r`. Kac–Rice applied to the gradient, with the nonnegative mark `W_r` times the index indicator, contributes one factor `|det H_X|`. That factor is already inside `F_j`, so (16) does not count it twice. `Q_r^W` is not used as a Gaussian law.

In dimension two each `F_j` extends continuously by zero through singular matrices, because the determinant factor vanishes on the index boundary. The product weight is continuous and nonnegative. The sixth-moment bound (14) makes the polynomial growth integrable, which is what truncation needs. Height is not a coordinate of the mark, and no coarea factor in the witness height is inserted. The parenthetical rewriting `≤ C' k r^3` in Section 1 uses `k ≥ k_-` only; the proof of (2) never multiplies by a height-window length, and the note correctly refuses uniformity as `k → 0`.

The external Kac–Rice references are framework for this representation. They are not the source of (13)–(20).

## R7 — ACCEPT

Divide (13) and (15) by (5). The Jacobian `det diag(r^2 δ, r δ) = r^3 δ^2` produces the `r^{-3} δ^{-2}` in the gradient density, and `Z_r^{-1}` produces `r^{-2}`. Together with the conditional sixth moment this is (17): `r^{-5} δ^{-8} exp(-c/δ^2) min(1, (r/|v|)^6)`.

On `|v| ≤ r` one has `r ≤ δ ≤ r √2`, so `δ^{-8} ≤ r^{-8}` and `exp(-c/δ^2) ≤ exp(-c/(2r^2))`. The minimum is 1, and the intensity is at most `C r^{-13} exp(-c/(2r^2))`. On `r ≤ |v|` the minimum contributes `(r/|v|)^6` and `|v| ≤ δ ≤ |v| √2`, so the intensity is at most `C r |v|^{-14} exp(-c/(2v^2))`.

The series `exp(c0/s^2) ≥ (c0/s^2)^7 / 7!` is exactly (18). The sixth positive term would leave `r^{-13} · r^{12} = r^{-1}` on the inner chart, which is not `O(r)`. The seventh term supplies `s^{14}`, cancels `r^{-13}` up to one power of `r`, and cancels `|v|^{-14}` exactly. Both pieces are therefore at most `C r` throughout the fixed strip `a0 ≤ |u| ≤ B`, `δ ≤ δ0`.

For the cover, `a0 = (A + 1)/2` satisfies `1 < a0 < B` whenever `1 < A < B`. The gap `A^2 - a0^2 = (3A + 1)(A - 1)/4` is positive, so a fixed `ε > 0` exists with `√(A^2 - ε^2) > a0`. Every annulus point with `|v| ≤ ε` then falls in the strip of Section 7 once `r` is small. On `|v| ≥ ε` the exact scaling `J = (f_x/r^2, f_z/r)` has Jacobian `r^3` back to the gradient. Its leading coefficient matrix on `(f_zz, f_xxz, f_xzz)` has `a,c1` minor `-v^3/2`, of size at least `ε^3/2`. Those three monomials are distinct from the six midpoint pins, so `J` has a uniform covariance floor and a bounded mean for small `r`. Conditioning on `J = 0` therefore shifts the field by a bounded `C^3` amount, and the conditional sixth moment stays `O(1)`. Estimate (7) contributes `O(r^6)`. Division by `Z_r` contributes `r^{-2}`. The product is `O(r)` on the off-axis chart as well.

The two charts cover `K_AB`. Physical area contributes `r^2`, so the expected all-height count on `rE` is at most `C r^3 area(E)`. Null chart boundaries do not affect the absolutely continuous intensity. There is one endpoint normalizer and one witness determinant, and no height-window Jacobian.

## Boundary of this acceptance

(2) is an upper bound for one fixed scaled annulus, one fixed period, dimension two, and gap marks bounded away from zero. The Gaussian constant `c` in (13) depends on `k_-` through `d(u) ≥ 6 k_-(a0^2 - 1/4)`. Markov's inequality upgrades (2) only to an upper bound on witness existence. The between-pin-height count is smaller and inherits the same upper bound; that inclusion does not put a height factor into the all-height ledger.
