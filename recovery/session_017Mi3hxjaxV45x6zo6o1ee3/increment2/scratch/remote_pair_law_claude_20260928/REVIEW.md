# Nonauthor analytic review: the leading remote factorial-pair measure and its height-gap law (Math-#116, C3)

Scientific effect: **NONE**. This file changes no register, status, graph node, lemma flag, prize or author source.
It records a verdict on candidate C3 of `reviews/candidates_pending_20260928/CANDIDATES.md`. Integration is a
separate act.

## Object

| Field | Value |
|---|---|
| Object | OA-WINDOW-MULTIPLICITY-REMOTE-20260928-v1 (OpenAI / ChatGPT) |
| Branch / head | `chatgpt/window-multiplicity-laws-20260928` / `0507e3a1dbd3dede84cbeb805947c70aa16ebc36` ([Math-#116](https://github.com/d6g8k5htny-coder/Math-/pull/116)) |
| `REMOTE_PAIR_LAW.md` | Git blob `3fb602040b37158b16ba61bdfc62cc8df1ccc616`, 15878 B, SHA256 `b818a11f1abcf37616d592ecda6341b0d99365cf3ac5047bc2dbaede95da4c5e`, 325 lines |
| Consumed, on main | [RC] `frontiers/remote_collision_20260928/PROOF.md` (blob `7b48a88e…`): (3.1), (4.1)–(4.3), Lemmas 1, 2, 3, 5. [RM] `frontiers/remote_window_20260924/PROOF.md` (blob `b383bfcc…`): contact frame, (8), §4. The D1 normalizer `Z_r/r² → z_0` (reconciled in Math-#126, open). |
| Request | [Owner comment 5880152796](https://github.com/d6g8k5htny-coder/Math-/pull/116#issuecomment-5880152796); claimed in [5880826796](https://github.com/d6g8k5htny-coder/Math-/pull/116#issuecomment-5880826796) |

## Provenance and exposure

| Field | Value |
|---|---|
| Reviewer | Anthropic Claude, Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3` |
| Relation | Different provider from the OpenAI author. Same GitHub account as every lane, so no organizational independence is claimed. |
| Exposure | **I authored [RC]**, whose Lemmas 1, 2 and 5 and formula (3.1) C3 imports verbatim. I also reviewed LP Theorem A (#106), IW (#109) and C1/C2 (#123), and I reconciled D1 (#126). This review therefore checks that C3 uses [RC] correctly. It is not independent evidence for [RC] itself. [RC]'s nonauthor review is the xAI/Harper review 5342481786 on #110, which accepted its finite identities, the (3.1) mark and the Lemma 1 mechanism. |
| Author code | `algebra.py` is absent from the tree, as `PUBLICATION.json` discloses. I did not run, request or reconstruct it. |
| Independent checks | `remote_pair_review_check.py` is my own exact-rational suite, written from the markdown. It covers finite identities only. |

## Verdicts

| Interface (lines) | Verdict |
|---|---|
| §1 (8–26): ordered-pair count `T_ij`, `Σ T_ij = N(N−1)`; no event asymptotic inferred | **ACCEPT** |
| §3 (81–121): coupled regression on `(U_r, V_δ)`; convergence to the `(U_0, V_0)` regression; full conditional support of `(B_0, A, T)` | **ACCEPT**; see §1 below. |
| §4 (123–159): (R7) and the congruence limits (R8); `W_r/r² → w_0`; adjacency; same-index pairs are higher order | **ACCEPT**; see §2 below. |
| §5 (161–233): blow-up ledger, windows (R9), domination, fixed-Borel translation, (R10), coefficient (R11) = (R3), separated pairs, positivity and symmetry | **ACCEPT**; see §3 below. |
| **Theorem R (R4)–(R5)**: `E T_ij(E) = r⁵ ∫_E K_ij + o(r⁵)`; `K_ij > 0` iff `|i−j| = 1`; `E N_j(N_j−1) = o(r⁵)` | **ACCEPT** at the stated fixed-`E`, fixed-`ρ`, existential scope |
| **Theorem H2 (R12)–(R15)**: TV convergence of the factorial-pair height measure to `(5/9)|a−b|^{−1/3}`; Beta(2/3, 2) gap; moments | **ACCEPT**; see §4 below. |
| §7 (306–325) exclusions | **ACCEPT** as stated. They are needed: see N2. |

No defect was found. Notes N1–N4 delimit the result without changing it.

---

## §1 — the coupled conditional frame (§3)

`V_δ` is exactly [RC] (4.1). Its `δ = 0` extension (R6) follows from the Peano form [RC] (4.2), where
`D3_δ → −(1/12)∂_e³f(x)`.

[RC] Lemma 1 gives the covariance sandwich for `(U_r, V_δ)` on the compact parameter set, including
`r = δ = 0`. [RC] Lemma 2 gives conditional moments polynomial in `t` and a target density decaying faster than any
power of `t`.

C3's extra step: regress one smooth field on `O_{r,δ} = (U_r, V_δ)` and let `(r, δ) → 0`. This is sound. The
regression coefficients `Cov(F(z), O)Cov(O)⁻¹` converge because every entry is an integral derivative average
against kernels that tend to point masses, and the inverse is uniformly bounded by the floor. The targets
`a_{r,δ} → (v_0, 0, 0, b, t)`. Hence convergence holds in `L^p(C^q)` on compact target sets. The mean need not
vanish, and C3 does not claim it does (line 114).

**Full conditional support.** The functionals `(U_0, B_0)` at `0` and `(Y, A, T)` at `x`, with `|x| ≥ ρ`, are
distinct jets of order ≤ 3 at two distinct torus points. They are jointly nondegenerate. A null combination gives
`P(ω) + Q(ω)e^{iω·x} = 0` on the dual lattice. Choose a lattice direction `v` with `v·x ∉ 2πℤ`. Along each lattice
line in direction `v`, a polynomial in `n` would equal a polynomial times `e^{in v·x}`. That forces both polynomials
to vanish, so `P = Q = 0`.

This argument is mine; the manuscript cites [RM]'s distributional argument. It gives full support of `(B_0, A, T)`
given `(U_0, Y)`. `H ↦ He` is onto `ℝ^d`, and `He` together with `A` exhausts the Hessian entries at `x`.

## §2 — determinant limits without inverses (§4)

(R7): on `∇f(x) = ∇f(x+δe) = 0`,
`0 = ∫_0^δ H(x+τe)e dτ = δH_x e + (δ²/2)D³f(x)[e,e,·] + O(δ³‖f‖_{C⁴})`, which is the first line. The second follows
from `H_{x'}e = H_x e + δD³f[e,e,·] + O(δ²)`.

In a frame beginning with `e`, conjugate by `diag(δ^{−1/2}, I)`:
- the `ee` entry becomes `∓T/2 + O(δ)`;
- the cross entries are `O(√δ)`;
- the lower block is exactly `A` at `x` and `A + O(δ)` at `x'`.

Congruence preserves inertia and multiplies determinants by `δ^{−1}`. `F_j` is continuous everywhere, including on
singular matrices: it is `|det|` times an index indicator, and `|det| → 0` at the index boundaries. Polynomial moments
then give (R8) in `L^p`. The limit product `(T²/4)(det A)²χ_ij` has index `a+1` at `x` and `a` at `x'` when `T > 0`,
which is (R2). The normalizer `W_r/r² → w_0` and `Z_r/r² → z_0` are the D1 normalizer (I), read with the congruence
erratum `D_r = diag(r^{−1/2}, I)`.

Exact checks (`PAIR`) use the explicit field
`f(u,v) = (T/6)u³ − (T/4)δu² + ½vᵀAv + (u² − δu)c·v` in d = 2, 3, 4:
- `∇f` vanishes at `0` and `δe`;
- `det H/δ ∓ (T/2)det A = −δ·cᵀadj(A)c` exactly;
- the orientation rule (R2) matches the exact inertia for both signs of `T` and every index of `A`.

The same-index remark (lines 156–159) is confirmed: `f = c·v(u² − δu) + (λ/2)v²` has `T = 0`, two saddles, and
`det = −c²δ²`, which is order `δ²`.

## §3 — blow-up, domination, translation, coefficient (§5)

**Ledger.** The factors are:
- polar coordinates: `δ^{d−1}`;
- the value-gradient change of variables [RC] (4.3): `δ^{−d−3}`;
- the second height `dy' = δ³dt`;
- the determinants: `δ²`.

The δ-exponent is `(d−1) − (d+3) + 3 + 2 = 1` for every d. With `δ = rs`, `y = b − r³z` and `W/Z → w_0/z_0`, the
r-power is `d + 3 + 3 − (d+3) + 2 = 5` (`LEDGER`, d = 2…10).

**Windows (R9).** Both heights in `(b − kr³, b)` means `0 < z < k` and `s³t < z < k + s³t`. The `z`-length is
`(k − s³|t|)_+`. This is checked exactly on 400 rational instances (`WINDOW`).

**Domination.** [RC] Lemmas 3–4 give `W_r ≤ r²K^{2d}` and `|det H_x det H_{x'}| ≤ δ²K^{2d}/4`. [RC] Lemma 2 gives
the envelope `Cs(1+|t|)^{−p}1_{(R9)}`. Integrating over `z` and `t` gives `≤ Cs·min(1, s^{−3})`, whose integral is
finite (`∫_0^∞ s·min(1,s^{−3})ds = 3/2`). The envelope is uniform in `r` and in the compact parameters, and
zero-extension beyond `s < η_0/r` is harmless. Dominated convergence applies.

**Fixed Borel E.** The factor `1_E(x+rse)` is removed by continuity of translation in `L¹(torus)`. On bounded `s` the
shifts are at most `rS_0`, and the envelope controls `s > S_0`. No boundary regularity is used, and none is claimed
for `r`-dependent `E_r`. The argument is correct as written (lines 193–203).

**Coefficient.** Integrating `z` gives `(k − s³|t|)_+`. Then
`∫_0^{s_max} s(k − s³|t|) ds = (3/10)k·s_max² = (3/10)k^{5/3}|t|^{−2/3}` (R10), checked exactly (`OVERLAP`).
Multiplying by `36t²` gives `(54/5)k^{5/3}|t|^{4/3}`. With `t = −T/12`, this becomes
`(54/5)·12^{−4/3} = (3/40)·12^{2/3}`, since `54/5 = (3/40)·144` (`COEFF`). The conditional density of `t` given `Y`
converts to `E[·|Y]` with no extra Jacobian. The surface measure and the ordered-pair convention match (R3). Tonelli
away from `t = 0` is the right way to handle the `|t|^{−2/3}` factor.

**Separated pairs.** [RC] Lemma 5 gives `O(r⁶)|E|²`, which is `o(r⁵)` for fixed `E`.

**Positivity and symmetry.** The integrand is positive on an open set by §1. Under `e ↦ −e`, `T ↦ −T` while `A` and
`w_0` are unchanged, and the linear map on `Y` has `|det| = 1` at the symmetric target `(0,0,b)`. So
`χ_ij(A,−T) = χ_ji(A,T)` (checked), and `∫_S K_ij = ∫_S K_ji`. Hence (R5).

## §4 — the factorial-pair height law (§6)

(R12) is a normalized factorial-moment measure. §6 says explicitly that it is not the event-conditioned pair law, and
this review accepts only the former.

The dominated convergence of §3 is `L¹` convergence of the blown-up nonnegative densities, not only of their masses.
Pushforward to `(θ, θ') = (z/k, (z − s³t)/k)` contracts TV, and dividing by the converging positive total mass
preserves TV convergence.

**Jacobian.** With `g = s³|t|/k`, `s ds = (1/3)(k/|t|)^{2/3}g^{−1/3}dg`. This is checked as the exact identity
`s/(dg/ds) = k/(3s|t|)` (`HEIGHT`). With `dz = k dθ` and `θ ∈ (0, 1−g)`, each sign of `T` gives a density
proportional to `g^{−1/3}` on its triangle, of mass `∫_0^1 g^{−1/3}(1−g)dg = 9/10`. The two triangles carry equal
mass by `e ↦ −e`.

This gives `h = (5/9)|a−b|^{−1/3}`. The index-resolved density is `(10/9)(b−a)^{−1/3}` on `a < b` for `i = j+1`: the
index-`(a+1)` point lies above its adjacent partner. The gap is Beta(2/3, 2), and the following are all exact
(`HEIGHT`):
- the CDF (R14);
- `E g^n = 10/((3n+2)(3n+5))`;
- `E g = 1/4` and `Var g = 9/176`;
- the marginal `(5/6)(a^{2/3} + (1−a)^{2/3})`;
- `E a² = 29/88`, `E ab = 3/11` and `Corr = 2/7`.

## Notes

- **N1.** (R3) has hidden `k`-dependence beyond `k^{5/3}`, because `Q_0` conditions on `f_xxx = 12k`. Nothing in
  C3 claims otherwise. The coefficient should be read as "`k^{5/3}` times a `k`-dependent Gaussian conditional
  integral".
- **N2.** The fixed exclusion `ρ` is load-bearing. [RC] Lemma 1's floor is not uniform as `ρ → 0` ([RC] §8).
  LOCAL_MULTIPLICITY shows that pairs near the pins have order `r³`. §7's exclusions are therefore required, not
  cosmetic.
- **N3.** A factorial-moment asymptotic does not give `Q(N ≥ 2) ~ (r⁵/2)∫ΣK` without higher factorial moments.
  §7 says so, and this review adds nothing there.
- **N4.** Novelty boundary: generic adjacent-index attraction of critical points in the unconditioned isotropic
  setting is prior work (Azaïs–Delmas, arXiv:1911.02300). What this review accepts is the pinned, tilted, fixed-`ρ`
  coefficient (R3) and the factorial-pair height law (R13), at existential scope.

## What this review does not do

It does not review C4 (`REMOTE_DISTANCE_MOMENTS.md`), C5 or C6. It does not accept an event-conditioned pair law, a
global factorial upper bound, uniformity in `ρ → 0` or in oscillating `E_r`, or any numerical coefficient. It does
not run or reconstruct `algebra.py`, and it changes no register.

## Reproduce

    python -B -S remote_pair_review_check.py            # prints RESULTS.json byte for byte
    python -B -O -S remote_pair_review_check.py         # identical
    python -B -S remote_pair_review_check.py --mutant M # exit 1 for each of the 5 mutants
