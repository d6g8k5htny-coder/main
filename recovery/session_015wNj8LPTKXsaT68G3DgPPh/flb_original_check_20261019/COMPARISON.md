# ORIGINAL against the finite-L binding: comparison record

`FLB-ORIGINAL-CORRESPONDENCE-CHECK`, by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Dylan Roy — delegated AI review. Pickup: [main#419/6278469127](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6278469127). Written at about 22:15Z on 19 October 2026. Before writing it, C8 had read none of the following: the binding packet's `REVIEW.md` beyond line 25, the focused tensor review, and Claude 01Cz's input. Credit 0. Scientific effect NONE.

## Objects

- **ORIGINAL:** `independent-derivation.md`, extracted from Math-#193 comment 6242350213 by `extract_original.py` (see `extract_log.txt`). It is 24,027 bytes and 629 lines, with SHA-256 `f38c085ab5b15368e6989a1fd6298490f3b671f5a6c3a01bcdc81b516022bc33` and git blob `d537bade335adb32050bd5b9470fec5a66c947e4`. All four values equal the identity recorded in `FOLLOWON_SOURCE_INDEX.json` (blob `fa21d332`).
- **Binding:** Math- `frontiers/original_finite_l_gaussian_binding_20261017/PROOF.md` on Math- `main` `4eb86ed3`. It is blob `e8bb07fe8fab94616f02bdf792c34e3b337e1cf6`, 445 lines and 22,163 bytes, with SHA-256 `ad9ab5afea28062dfc661d0d19b43f20e67cfd67cb6910d3cc6c3fb8b81398d8`.

C8 read both files in full.

## 1. Law and assumptions: ORIGINAL §2 against binding (3)–(5) and (7). MATCH.

| Point | ORIGINAL | Binding |
|---|---|---|
| Lattice | every m ∈ ℤ in each coordinate; q_m = 2πm/L; P(Q_i = q_m) = e^{−q_m²/2}/z_L, coordinates independent | w_m on all of ℤ^d ("ENTIRE lattice"); w_m is that product law; Z_L = z_L^d (4) |
| Zero mode | moments include "the normalization by the zero mode" | (5) keeps √w₀ Z₀ |
| Mean | "ensemble mean zero …; an additional samplewise spatial-mean removal has not been imposed" | "no … samplewise mean removal"; the spatial mean is √w₀ Z₀ |
| Covariance | K_L(x) = E cos(Q·x) | (7): Σ_m w_m cos(p_m·(x−y)) = E cos(Q·(x−y)) = K_L(x−y) |
| Variance | K_L(0) = 1 | "pointwise variance Σ_m w_m = 1" |
| Coordinates | physical; wave numbers 2πm/L (§6) | physical, on ℝ^d/(Lℤ^d); p_m = 2πm/L |
| Observable | finite ordinary superlevel H0 bars per volume L^d; elder rule; essential bars omitted; heights integrated out (§1) | (2) and §7: same, through GAD's selector; (16) integrates the height out |

## 2. Equations. MATCH, symbol for symbol.

| ORIGINAL | Binding | Object |
|---|---|---|
| (1), with dω the "ordinary surface area measure" | (1), "full sphere AREA measure" | C_alg(d,L) |
| (4) | §5 | a, b, c, κ₄ = b − 3a², κ₆ = c − 15ab + 30a³ |
| (5) | (8) | Cov(G) = aI; Cov(H_ij, H_kl) = a²(three pairings) + κ₄δ_ijkl |
| (6) | (11) | B(e) = a²(I + 2eeᵀ) + κ₄ diag(e_i²) |
| (7) | (9) | Cov(u, G_i) = −v_i(e), v_i = 3a²e_i + κ₄e_i³ |
| (9), (8) | E(Q·e)⁶ display, (10) | V(e) = 6a³ + 9aκ₄Σe_i⁴ + (κ₆ − κ₄²/a)Σe_i⁶ |
| (10) | (14) | F(e) = E[(det A_e)² 1{A_e ≺ 0}], A_e = UᵀHU given He = 0, no height pin |
| (11) | (12), C, (13) | R_ab,i = a²δ_ab e_i + κ₄U_ia U_ib e_i; C; Γ_e = C − R B⁻¹ Rᵀ |
| (11a) | §6 ("free entries … listed once; no orthonormalized off-diagonal factor") | F as a Gaussian integral over free entries a ≤ b |
| (17), (18) | §7 | P_r, D_r, W_r with M = x − re/2 (maximum), S = x + re/2 (saddle), Δ = F(M) − F(S); Jacobian 12r^−(d+3) |
| (19) | (18), first line | p_(D,He)(0,0) = (2π)^−d a^−d/2 (det B)^−1/2 |
| (22) | §8, last paragraph | 12^−1/3 ∫ p_D(0) p_He(0) F(e) E[(T_e)₊^{4/3} \| D = 0] dω |
| (23) | (19), halved | E[(u₊)^{4/3}] = 2^−1/3 Γ(7/6) V^{2/3}/√π |

**Checked by hand:**
- (5)–(9) from the product law. For the sixth moment, the cumulants of Q·e are aΣe_i² = a, κ₄Σe_i⁴ and κ₆Σe_i⁶, so m₆ = k₆ + 15k₄k₂ + 15k₂³.
- R and C, using Uᵀe = 0.
- (13): G(0) = −(r²/8)T(e,e,·) and He = O(r²).
- (16): Δ = r³u/8 − r³u/24 = r³u/12, with W_r → u.
- The Jacobian, by block-triangular structure: r^−d times 12/r³.
- 12^−1/3 · 2^−1/3 = 24^−1/3.

`tensor_check.py` (`tensor_check_log.txt`) confirms (5)–(9), R, C and the (30) limit exactly, in d = 3 and d = 4.

## 3. d ≥ 3. MATCH.

- **Transverse block.**
  - Both condition the full (d−1)×(d−1) block UᵀHU on the whole column He = 0 (the longitudinal entry and the mixed entries), with no height pin, using covariance (11) on the free entries a ≤ b.
  - Both define F(e) as an expectation, so no matrix-measure convention can enter.
  - Both note independence from U: replacing U by UO conjugates A_e by O.
  - ORIGINAL (15) has the determinant limit (r²u²/4)(det A_e)². In the frame (e, U), det H_± = ±(r/2)u det A_e + O(r²), because h_eU h_Ue = O(r²). This is consistent with the (det A_e)² weight that the binding takes from GAD (28).
- **Area measure.**
  - ORIGINAL (1) says dω is "ordinary surface area measure, not normalized probability measure".
  - ORIGINAL §6 integrates the directed max→saddle displacement over the full sphere, with "no extra factor one half".
  - ORIGINAL §6 also states the equivalence E(u₊)^{4/3} = E|u|^{4/3}/2, which is what the binding's §8 says about GAD's half and the absolute moment.
- **ORIGINAL's own continuum checks are consistent with both readings in every d.**
  - At L = ∞, (11) reduces exactly to (30), GOE_{d−1} + √(2/3) Z I (exact in d = 3 and 4).
  - (31) follows from (1) with the area factor |S^{d−1}|, because 6^{2/3}/24^{1/3} = (3/2)^{1/3}.
  - (32) gives c_{2,∞} = 0.073406919306034271…, which agrees with 014K's C(2,24) = 0.0734069193060351 to a relative 1.1e-14.

## Notes, none blocking

- **O1.** ORIGINAL calls Z_L = z_L^d "the supplied normalization", referring to a specification it does not contain. Its own text still fixes the law: the masses, K_L, centering and no mean removal. The binding's (3) satisfies the identity.
- **O2.** ORIGINAL §2 does not say "Gaussian" in the law's definition, but the note treats the field as centered Gaussian throughout: §2 ("u | G=0 is a centered Gaussian"), §3, §8 and §9 ("the Gaussian vector"). The binding's §4 reading is supported.
- **O3.** ORIGINAL leaves d unrestricted. The binding's d ≥ 2 is a scope restriction, not a mismatch.

## Not checked

- ORIGINAL §§7–9. The binding does not rely on them.
- GAD, METHOD V2 and the binding's `INPUT_MANIFEST.json`.
- Any numerical evaluation of C_alg beyond the arithmetic above.
