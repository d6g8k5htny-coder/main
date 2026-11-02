> Same-provider (Anthropic) fresh-context consistency read of draft 14 at `d8aaa0e4`, run by the author session Claude01N as a subagent; read-only; no organizational-independence credit. Kept as the source of draft 15 (CONCORDANCE, "Checkpoint draft 15"). It is a record, not a review verdict.

# Whole-document consistency read: SIDE24 V3 supplement, draft 14

Source: `docs/manuscript/side24_v3_supplement_draft_20261013/` on branch `claude/side24-v3-supplement-draft14-20261021` at commit d8aaa0e4. Statement numbers come from `supp_main.aux`. Line numbers refer to the `.tex` files. This was a read-only review. I edited nothing.

## Summary

- **No ERROR found.** Every cross-section citation I checked supports the step that uses it, with five GAPs (G1–G5).
- **The GAPs are about radii or parametrisations that are never matched explicitly.** In each case a radius or a parametrisation proved in one section is used in another without the sentence that matches the two.
- **The rest are notation collisions (NIT).** The most consequential is that `c_2[K]` means two different things in §S4 and §S11.

## 1. Cross-section citations

### ERROR

None.

### GAP

**G1. Range of r for the Palm normalizer in §S5.**
- Definition S5.1 (s5:19) states `c_Z r^2 ≤ Z_r ≤ C_Z r^2` "on compact positive mark sets", citing §S2.
- (selection-normalizer) at s5:135–137 adds `E^0[W_r^2] ≤ C_W r^4` on the same basis. It is used for division in Lemmas S5.19, S5.23, S5.25, S5.32 and S5.34.
- Lemma S2.9(iii) proves these bounds only for `0 < r ≤ r_0(𝒦)`. Definition S5.1, by contrast, is set for every `0 < r < 12`.
- Theorem S5.36 has an existential `r_0`, but its proof never says what that `r_0` must be below. The lemmas it combines use these thresholds:
  - `r_0` of Lemma S2.9(iii);
  - the `r_0` of Lemmas S5.19 and S5.23;
  - `r ≤ δ/C_1` from Definition S5.6;
  - `r_0 < δ` from Lemma S2.3(iv), which underlies Lemma S5.24;
  - `r_mat` from Proposition S5.35.
- Fix: add one sentence to the proof of Theorem S5.36 taking the minimum of these, and say "for r ≤ r_0 of Lemma S2.9(iii)" at s5:19 and s5:135.

**G2. Theorem A is applied in κ-variables but stated in k-variables.**
- Theorem S12.3 bounds `p_r(b,k,R)` for `(b,k)` in a rectangle `𝓑 × 𝒦`, with `f(S) = b − k r^3` and an arbitrary frame R.
- The proof of Proposition S6.4 (s6:174) and Remark S5.37 (s5:929) use it for `p_r(t,b,κ)` with `f(S) = b − κr^3/6`.
- Remark S5.37 matches the two *events* through Lemma S5.3. Neither place states the parameter conversion: `k = κ/6`, `𝒦 = I_κ/6`, and any frame R with `Re_1 = t`.
- The conversion is written out only for the kernels (s6:85–89).
- Fix: one clause at s6:174.

**G3. §S7's r_0 against §S12's r_0.**
- §S7 fixes its own `r_0` in (rate-r0) (s7:89–93).
- Three statements are attributed to results proved in §S12 for §S12's `r_0`, which is fixed at s12:535:
  - Hypothesis S7.6(b) (`0 < Z_r < ∞` for all b, k) is attributed to Lemma S12.27(c).
  - Hypothesis S7.6(c) is attributed to Proposition S12.28 (eq parent-135).
  - Hypothesis S7.7 is attributed to Lemma S12.29.
- It is true that any `r_0` meeting §S7's constraints is admissible at s12:535. Both need the eigenvalue bounds of Lemma S12.11, `r_0 ≤ 1` and `r_0 < L/(4√2)`. The paragraph after Lemma S12.29 (s12:602) also extends that lemma to any ρ > 0.
- But §S7 says neither. Fix: one sentence after (rate-r0).

**G4. r_R is not tied to the range of §S7.**
- Proposition S8.2 (s8:62) reads "There is `r_R ∈ (0, 1/2]`". All its items are taken from §S7, which proves them only for `0 < r ≤ r_0` of (rate-r0).
- Its proof (s8:82) never says `r_R ≤ r_0`. The proof of Proposition S8.3 then sets `r_K := min{r_R, r_0}` (s8:94), which suggests `r_R` may exceed `r_0`.
- (remainder-sep) at s8:40 is used for `ρ' ≤ r_R` and relies on Hypothesis S7.6(b), which holds only for `r ≤ r_0`.
- Fix: set `r_R := min{1/2, r_0}`.
- Related NIT: s10:60 says "From §S7, there is `r_R ∈ (0,1]`", while Proposition S8.2 has `(0, 1/2]`. §S10 then imposes `r ≤ 1/2` separately (s10:69, 242), so nothing breaks.

**G5. The d = 2 enclosure of c_{2,24} has no source.**
- Corollary S9.16 (s9:411) gives `c_{2,24} ∈ [0.07340691930603427103, …104]` with no citation.
- The ingredients are all in the supplement:
  - Theorem S4.4 at d = 2;
  - the closed form of `c[K_ref]` for d = 2 in Theorem S11.5;
  - the identification `c_{2,ref}` (Definition S4.2) = `c[K_ref]` (s11:101) = `c_{2,∞}` (s9:304).
- That identification is stated only for d = 3: Lemma S4.3, and s11:121 together with s9:346 for the `c_{d,∞}`–`c[K_ref]` link.
- For d = 2 it follows by comparing (correction-coef), (rate-c) and Proposition S12.31, but nobody says so.
- The d = 3 bound (s9:403) and Corollary S11.12(iii) (s11:423) quote the main text, where Corollary S4.5 and Corollary S3.18 would do (NIT).

### NIT

- **N-c1. Imprecise section pointers.**
  - s5:15 cites §S1 for corrected-pin nondegeneracy; it is Lemma S2.3(i).
  - s6:8 and s6:47 cite main-text Convention 2.2 rather than Convention S1.6.
- **N-c2. Prop S8.2(iv) is used at k = 0.** It is stated for k > 0 but used at k = 0: (remainder-W4) at s8:377, Lemma S9.9 (k' ∈ [0, k]) and (third-W1). The Taylor bounds involved do not depend on k, and item (i) is already stated for k ≥ 0, so adding "k ≥ 0" to (iv) suffices.
- **N-c3. Wrong equation cited for the unrestricted densities.** Convention S8.1(b) (s8:12) and s10:9 cite (parent-112) for the canonical versions of the *unrestricted* densities. (parent-112) is the compact-window density of Theorem S12.4. The correct pointer is (parent-135) plus Lemma S12.29, as in Theorem S12.5.
- **N-c4. Mutual pointer.** s2:221 refers to "the proof of Proposition S6.4" for uniform moment bounds when r < ϱ, while that proof (s6:187) refers back to "§S2" for the same bounded variances. §S6 does contain the argument, so this is circular in wording only.

### Checked citations (46)

Each was checked against the cited statement and, where cited, the specific item.

- **S1 → S2**
  - s1:108 → Lemma S2.3(ii) (contact list up to −1/12).
  - s1:210 → Theorem S2.12.
- **S2 → S1 and S12**
  - s2:77, 81, 83 → Theorem S1.3 and Proposition S1.5(i),(ii).
  - s2:287 → Proposition S1.8 (Jacobian).
  - s2:296 → Proposition S12.22.
- **S3 → S2, S4, S6**
  - s3:209 → Lemma S2.3(iii), Proposition S2.8, Lemma S2.9(ii).
  - s3:219 → Proposition S6.4, Corollary S6.5.
  - s3:505 → Lemma S4.20(b) and Definition S4.2. I verified `Cov h_t`, `σ_t² = Ω/(144a)` and `D_t = β^n D_n`, and checked numerically that `(1/6)(6^{2/3}/3)` × the gap integral equals the prefactor `Γ(7/6)/(24^{1/3}√π)`.
- **S4 ↔ S3:** Lemma S4.3 against B₃ in Theorem S3.10.
- **S5 → S1, S2, S4, S12**
  - s5:19 and s5:137 → Lemma S2.9 (gives G1).
  - s5:201 → Lemma S4.29 (the bound < 10⁻¹¹³).
  - s5:350, 364, 1025 → (setting-list) and Theorem S1.3.
  - s5:929 → Theorem S12.3 (gives G2).
- **S6 → S5, S7, S12**
  - s6:68 and s6:102 → Proposition S12.22 on compact Ω.
  - s6:89 → Lemma S7.2(a) and (rate-A). I verified that `G^sel_r · r dr dκ` matches `r 𝒜_r dr dk` with `dκ = 6 dk`, and that `(6^{2/3}/18)κ^{-2/3} dκ = (1/3)k^{-2/3} dk`.
  - s6:125 → Lemma S12.29 plus s12:602.
  - s6:172 → Proposition S2.8 and Lemma S2.9.
  - s6:174 → Theorems S12.3 and S5.36.
- **S7 → S12, S13:** Lemmas S7.1–S7.3 and Hypotheses S7.4–S7.7 against Lemmas S12.7, S12.8, S12.10, S12.11, S12.14, S12.16, S12.19–S12.21, S12.26–S12.29, Propositions S12.22, S12.23, S12.28 and Theorem S13.2. The statements match, apart from the radius (G3). Also:
  - s7:415 → Lemma S12.13 (the residual law does not depend on the target).
  - s7:423 → Lemma S12.17.
- **S8 → S7, S12, S13**
  - s8:64–82 → Lemmas S7.11–S7.13 and S7.16, Corollary S7.17 (gives G4).
  - s8:94 → Lemma S8.16, (rate-r0), Lemma S12.21, Theorem S13.2, Lemma S12.19(a).
  - s8:107 → Lemma S12.29: the pointwise bound `(1+|b|)^{2d} e^{-cb²}` follows from its density and moment bounds.
  - s8:216, 462 → Lemma S7.14 (`F_j = Φ_j` is stated).
  - s8:292, 311 → Lemma S12.8.
  - s8:417–427 → (rate-51)–(rate-53), Lemma S12.13, Lemma S7.2(b),(c).
  - s8:44 → (rate-c) and Proposition S12.28.
- **S9 → S8, S11, S12**
  - s9:30, 236 → Lemma S12.30 and Proposition S12.31. At k = 0, `φ_τ(0) = (2π)^{-1/2} τ^{-1}` matches.
  - s9:158–172 → (remainder-detK), (remainder-window), Proposition S8.2.
  - s9:223 → Proposition S8.3 (`r ≤ min(k, r_K)` holds since s ≤ 1) and Lemma S8.14.
  - s9:346 → Theorem S11.5. I checked the d = 2 and d = 3 closed forms against Corollary S3.18 and against the A_d^jet formula of Proposition S3.19, with D₁ = 4/3.
- **S10 → S8, S9, S12:** inputs s10:60–122 against Propositions S8.2, S8.3, S8.13, Lemmas S8.12, S8.14, S9.9, S12.8, S12.30.
- **S11 → S10, S12**
  - s11:51 and s11:398 → Lemma S10.12, together with (third-Omega), (third-F1)–(third-F3). Here `r_* = min(r_0^*, 1/2)`.
  - s11:398 → Lemma S12.8, for the hypothesis of Theorem S11.7.
- **S12 → S13, S5, S6**
  - s12:219 → (cap-hermite) in Lemma S13.3. The weighted average of `f_xxx` is 12k, as stated.
  - s12:289 and s12:386 → Theorem S13.2: the cap conditions match and the embedding needs `r < L/(4√2)`.
  - s12:378 → Lemma S5.3 (the measurability argument does not depend on dimension).
  - s12:602 → Proposition S6.2.

## 2. Notation

- **N1, the most consequential.** `c_2[K]` means two things:
  - In §S4, `c_d[K]` at d = 2 (Definition S4.2) is the 2-D leading coefficient. That is the source of `c_{2,ref}` and `c_{2,24}`, which Corollary S9.16 uses.
  - In §S11, `c_2[K]` (Definition S11.2) is the third-order fixed-cone expression, and `c_2` is the ℓ^{1/3} coefficient (§S10).
  - No statement uses both in a way that confuses them, but they sit two sections apart. Rename one.
- **N2. J₀ and J_r.**
  - J₀(t,b,κ) is the limiting contact density in Theorem S2.12.
  - J₀ is the Bargmann–Fock jet in Convention S3.1, Definition S4.16 and Hypothesis S5.11.
  - J_r also has three meanings: in §S2, in §S12 (`1 + ‖g_r‖`) and J_L in §S3.
- **N3. 𝒦.**
  - A compact set of (b, κ) in Hypothesis S2.7 and Lemma S2.9.
  - A covariance kernel in §S5 (Hypothesis S5.11, Lemmas S5.17ff).
  - The gap-mark interval in **k** in Definition S12.1.
  - 𝒦_δ in Proposition S9.6.
  - The κ ↔ k change in the windows is the issue under G2.
- **N4. Radii.**
  - `r_0` has at least five meanings: Lemma S2.9(iii); Theorem S5.36 and locally in Lemma S5.19; (rate-r0); s12:535; and in §S8, a free variable in Theorem S8.6 and Lemmas S8.11–S8.12, while Lemma S8.16 in the same section uses "r_0 as in (rate-r0)".
  - `r_1` has four: Lemma S7.2(b), s12:159, Lemma S9.10, and Remark S8.7.
  - `r_*` has two: Theorem S12.3 versus `min(r_0^*, 1/2)` in §S11.
  - The double use inside §S8 is the most likely to mislead.
- **N5. Locally defined but overloaded symbols.**
  - Ψ: S4 Ψ(C); a polynomial in S5; Ψ(b,k) in S7 and S12 (the same object there); Ψ^cand_ℓ in S8.
  - B: B₃, B₀, the jet ball B, the event B_r, B_{d,L}, 𝓑, and `∂_x D_y² f`.
  - A_t: the Cholesky factor in Proposition S3.11 versus the transverse block in Definition S4.2.
  - Z_24 / Z_L (lattice normaliser) versus Z_r (Palm normaliser).
  - B_{d,L}, 𝒜_r, Z_r, p_r and P = 1+|b|+k are consistent from §S6 to §S11.
- **N6. ν_cand.** In §S12 it means the windowed density of Theorem S12.4 (the unrestricted one is ν^all). In §§S7–S10 it means the unrestricted density. This compounds N-c3.
- **Conversions that are stated:**
  - κ = 6k: s7:14, s8:17, s10:15, s11:13.
  - κ as the cusp variable: s9:11, s10:15.
  - `τ_t² = 144 σ_t²`: Lemma S4.3, Remark S7.22.
  - `G^all_r = 𝒜_r(b, κ/6, t)`: s6:87.
  - `τ_r = −12 V^-_corr`: s11:17.
  - The |det| = 12r^{-(d+3)} transformation: Lemmas S7.2 and S12.10.
- **Normalisations.** K_L has variance one in every section. Z_r = E^0[W_r] is the same normaliser in §S2, §S5, §S7 and §S12.

## 3. Hypothesis scope (general d, L ↔ d = 3, L = 24)

Checked and in order:

- Theorem S12.3 at (3, 24): compact windows hold. The variable change is G2.
- Proposition S12.22 on compact Ω (§S6).
- Lemma S12.29 at any separation (s12:602, s6:125).
- Proposition S8.13 at ρ < 12 (s8:334).
- Theorem S7.9 → Remark S7.22.
- §§S8–S10 are stated for all (d, L) and specialised in Remarks S10.7 and S11.12.
- Lemma S9.15 for L ≥ 24, d ∈ {2,3}; Corollary S11.10 for L ≥ 10.
- Theorem S3.10 for all L: it states its conditional clause (s3:219).
- Morse property under the Palm law: Remark S5.38. Without pins: Lemma S12.19(b), which gives Proposition 1.1 of the main text.
- Embedding `r < L/(4√2)`: stated at every use of Theorem S13.2 (s7:127, s8:94, s12:382).

One implicit condition (NIT): Proposition S9.6 needs Φ to be injective on 𝒲 = [−3, 2] × {|Ξ| ≤ 30δ⁻²}. Theorem S9.7(iii) and Lemma S9.10 use it at small r without saying "for r small enough that Φ is injective".

## 4. Constants and radii

G1, G3 and G4 cover this category. Other radius transfers I checked and found consistent:

- `r_0^* ≤ r_K ≤ r_0`(S7) `< L/(4√2)` (s8:32).
- `ρ_1 ≤ min(r_1, r_0^*)` and `s_*ℓ^{1/4} ≤ r_K` (s9:211).
- `r ≤ min(k, r_0^*)` for (third-K2) (s10:71).
- `r_* = min(r_0^*, 1/2)` (s11:398).
- The collar and singular-near `ε_1`, `r_0` (s5:387, s5:567).
- Lemma S2.3(iv), which needs `r_0 < δ`, against region (iii) of Definition S5.6, which needs `r ≤ δ`. They are compatible.

ℬ (the S12 birth window) is used only inside §S12. The compact set 𝒦 is covered by N3 and G2.

## 5. Global structure

- **No Part has an opening paragraph.** Each `\part*` is followed directly by `\input` (supp_main.tex:59, 67, 72). Only the abstract (supp_main.tex:54) describes the Parts.
- **The abstract is accurate except for one phrase (NIT).** It says Part II gives the results "with their proofs". Theorems S9.13 and S11.5 have "(method)" proofs that describe computer-assisted certificates (s9:361, s11:145). s8:473 says so, but the abstract does not.
- **Every "proved in §X" announcement is fulfilled.** The ones I traced:
  - s2:375 → §S6;
  - s3:560 → §S4;
  - s4:76 → s4:564;
  - s8:107 → §S8.17 (`ssec:remainder-Z`);
  - s8:405 → Lemma S8.16;
  - s10:203 → s10:634;
  - s12:5 → §S13;
  - S0's claim that Proposition S12.24 has a proof → s12:491.
  - Theorems S10.1–S10.5 and Proposition S10.6 are proved at s10:568–634.
- **Every `\ref` resolves.** By script: 236 cross-file statement or equation references, none missing.
- **Section openings match their content.**
  - §S5 says the main text cites Theorem A and that §S5 gives a second proof.
  - §S7 lists its imports from §S12.
  - §S10 announces the 4/11 and 3/7 remainders.
- **Presentation (NIT).** Hypotheses S7.4–S7.7 are labelled "Hypothesis" although §§S12–S13 prove them.

## Scope actually checked

- **Read in full:** supp_main.tex; §S0 (orientation only); §S1; §S2; §S3; §S4; §S5; §S6; §S7; §S8; §S9; §S11; §S12; §S13.
- **§S10:** lines 1–330 (setup, inputs, results, Lemmas S10.8–S10.14) and 336–410 in part, read in full. Lines 395–410 and 530–636 (Proposition S10.6 and the proofs of the theorems) read in full. Lines 410–530 (Lemmas S10.17–S10.24, elder-kernel internals) only scanned for cross-section `\ref`s, of which there are 2, both checked.
- **Scripted:** the cross-file `\ref` census and label resolution against `supp_main.aux`.

## Not checked

- The internal correctness of single-section arguments; the slice-level reads covered these. In particular:
  - the S5 capture–escape rational ledgers;
  - the internals of the S10 elder decision (Lemmas S10.15–S10.24);
  - the S11 stability constants.
- The computer-assisted certificates and the `checks/` scripts (not run).
- Agreement with the main text's equation numbers, CONCORDANCE.md, and the bibliography.
- The compiled PDF (I did not rebuild it). Numbers come from the existing `supp_main.aux`.
