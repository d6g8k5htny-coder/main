# D1 Theorem A selection chain — source-bound reconciliation

**Object:** D1-CHAIN-RECONCILIATION-20260928-v1.
**Reconciler:** Anthropic Claude, Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3`.
**Effect:** register reconciliation. This record introduces **no new mathematical claim** about the parent. It binds
existing nonauthor reviews to exact source bytes, discharges every residual item they left open, reviews the two
parent sections that no earlier review covered (§§11–12), and states the resulting scoped status. The register edits
it justifies are listed in §9. `lemma_closed`, prizes and premises are untouched.

## 1. The reconciled object and its reading rule

The D1 parent is OpenAI/ChatGPT's UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1, with the imported deterministic
MARKED-CYLINDER-CAP-20260924-v1. The parent bytes are immutable. The object reconciled here is the parent **read
together with** the three amendments in the table below. Every consumer must cite this identity set, not the parent
alone.

| Role | Path (Math-) | Git blob | Bytes | SHA256 |
|---|---|---|---|---|
| Parent (P) | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | `dfed3b8d318a3ab1950957f393307733a4bef3f2` | 40261 | `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7` |
| Cap import (CAP) | `imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md` | `0633aca3c2a2882b0de4399da0a75d64c2e6b2e1` | 15160 | `0bf922b9203c29088b12388807aa0e2ecd020485eb0f6e919679841b5b2636fc` |
| Amendment E1: congruence erratum | `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md` | `213594d6ca6a86fb938110f4d166d9ce275a02d0` | 1782 | `bad7ef609c4ad8c41ad6af562c1b6807921e19a9d556ed793ad1a0db6e202028` |
| Amendment E2: §9 replacement v1.1 | `reviews/d1_section9_borel_repair_20260925/REPAIR.md` | `fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a` | 9062 | `845abf9f9c99d672c2a10a887b5a2e7206a3d2de3d876f35f75ff6e2dc13e62f` |
| Amendment E3: wording W1 and embedding radius | this record, §5 | — | — | — |

**Reading rule.**

- Read P §5 with E1. The congruence factor is `D_r = diag(r^{-1/2}, I)`, not the displayed `diag(√r, I)`.
- Read P §9 as replaced by E2.
- Read P l. 165 with W1 (§5 below).
- Read P §1's "sufficiently small r" as including the explicit embedding bound `r < L/(4√2)` (§5 below).

No formula of P changes.

**Machine binding.** Each component above is also its own source node in the hard-gate graph
(`frontiers/downstream_gate_20260925/GRAPH.json`), required by the D1 node: E1, E2, CAP (whole file; only §§1–5
are consumed) and this record's directory, which carries E3 and `LEDGER.json`. A byte change or deletion of any
component therefore proposes D1, D2 and D3 for revalidation even when P's bytes are unchanged (§9, §10).

**Name collision.** "Theorem A" here always means P (1.1). The fixed-remote mean-measure theorem in
`frontiers/remote_window_20260924/PROOF.md` is also called "Theorem A". That is a different theorem (D4), and
nothing in this record concerns it.

## 2. Reconciled statements

The model is P §1:

- the exact L-periodized Gaussian field on `R^d/(LZ^d)`, for fixed `d >= 2` and `L > 0`;
- the six pins, generally `2(d+1)` observations;
- the typed weight `W_r` and the FULL normalizer `Z_r = E_Q W_r`;
- `p_r(b,k,R)`, the `Q^W`-probability that the global ordinary superlevel elder partner of `M` is `S`.

The three statements below are accepted at existential scope. There is no numerical `C`, `r_*`, `z_*`, `c_{B,K}`
or `c_{d,L}`.

- **Theorem A (P 1.1).** For compact `B` and `K ⊂ (0,∞)`, there are `C` and `r_* > 0` such that
  `0 <= 1 - p_r <= C r^3`. This holds uniformly over `b ∈ B`, `k ∈ K`, all orthonormal frames and `0 < r <= r_*`.
- **Theorem B (P 1.2).** Let `B` and `K ⊂ (0,∞)` be compact intervals **of positive length** (P l. 37; a singleton
  interval gives zero mark-window intensity). The compact-window densities satisfy
  `ν_cand ~ ν_eld ~ c_{B,K} ℓ^{-1/3}` with `c_{B,K} > 0`. Their difference satisfies `0 <= ν_cand - ν_eld <= C ℓ^{2/3}`.
- **Theorem C (P 1.3).** All births, all positive gaps and all separations are included, and the essential class is
  excluded. Then `ν_cand^all ~ ν_eld^all ~ c_{d,L} ℓ^{-1/3}` with `0 < c_{d,L} < ∞` given by (15.2). This is the
  leading term only: no unrestricted remainder or difference rate is included.
- **§12 corollaries.** These are the cumulative and moment consequences (12.1)–(12.3) and their unrestricted leading
  forms.

**Sharpness in d = 2 (composition, separately sourced).** In the plane, the direct lower bound
`1 - p_r >= c r^3` combines with Theorem A to give `1 - p_r = Θ(r^3)` and `ν_cand - ν_eld = Θ(ℓ^{2/3})` on compact
marks. The lower bound is (E2) of `frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md`,
blob `aaefc8da…`, reviewed in `reviews/d1_elder_lower_claude_20260928/`. **No lower bound is asserted for `d >= 3`.**

## 3. The four components of the chain

| Component | Sections | Output | Interfaces |
|---|---|---|---|
| (I) Matrix-cap normalizer | P §§2–5 with E1 and W1 | `Z_r/r^2 -> z_0 > 0` uniformly; floor `z_* r^2 <= Z_r <= z^* r^2` (5.4)–(5.5) | A1, A2, A4, A3 |
| (II) Cubic pairing-defect bound | P §§6–7 | `E_Q[W_r 1_{G_r^c}] <= C r^5`, hence `Q^W(G_r^c) <= C r^3` (7.8) | A5, A6, A7 |
| (III) Global elder selection | CAP §§2–5 and P §8 | on `G_r` ∩ {Morse, distinct values}, the global elder partner of `M` is `S`; hence `1-p_r <= Q^W(G_r^c)` | CAP, D1-A |
| (IV) Marked Kac–Rice and lifetime | E2 (§9), P §§10–15 | intensity identities, radial ledger, pushforward, majorant, off-diagonal bound, coefficient (15.2) | D1-B/E2, D1-C, §11, §12, D1-D, D1-E |

Theorem A = (I) + (II) + (III). Theorem B = Theorem A + (IV) through §11. Theorem C = (IV) + Theorem A, which gives
`p_r -> 1` pointwise in §13.

## 4. Interface ledger

**Providers.** OpenAI is the author of P, CAP, E1 and E2. The nonauthor providers are xAI/Grok and Anthropic/Claude.
All lanes share one GitHub account, so no organizational independence is claimed anywhere. The full machine-readable
ledger, with every review locator, is `LEDGER.json`.

**Review records.**

| ID | Provider and session | Record | Verdicts |
|---|---|---|---|
| G1 | xAI Grok 4.7 via Cursor, `bc-ce4bf0bc` | [main#63 comment 5841570965](https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841570965) | D1-A–E ACCEPT |
| C1 | Anthropic Claude, this session | `reviews/d1_theorem_a_nonauthor_20260928/REVIEW.md` (blob `bc11369c…`, Math-#106 v2) and [main#63 comment 5873873903](https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5873873903) | A1–A7, CAP, Theorem A and E2 C1–C6 ACCEPT; W1 |
| G2 | xAI Grok, Lucas lane | `reviews/d1_section9_borel_grok_lucas_20260928/REVIEW.md` (blob `48ea9ad5…`) | E2 C1–C6 ACCEPT. Narrative record; its runner is absent. |
| G3 | xAI Grok via Cursor, replay rounds 3–9 | main `incoming/grok-session-20260926-replay/` | Partial slices; see the table below |
| G4 | xAI Grok 4.7 via Cursor, `bc-81faa729` and `bc-dfbce246` | [main#67 5841270276](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841270276), [5841782206](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206) | While reviewing D2, re-derived P (6.2), the depth-failure integral, the m=1 split, and the far and M4 Markov pieces. Also checked the (11.3) = (13.6) coefficient identity. |
| C2 | Anthropic Claude, this record | §6 below | §11 and §12 ACCEPT; spot re-derivations of (13.1), (13.4) and (15.2) |
| O1 | OpenAI technical review of Math-#106, [5341593068](https://github.com/d6g8k5htny-coder/Math-/pull/106#pullrequestreview-5341593068) | — | Confirms W1 and the A5/A6 fixtures. **Author provider, so non-discharge.** |

**Interfaces.** "Full" means a nonauthor provider re-derived the whole interface. "Partial" means a nonauthor provider
checked part of it; partial checks never discharge on their own. `reconciliation_check.py` computes the full-depth
counts from `LEDGER.json`.

| Interface | P lines | Claim | Full-depth nonauthor ACCEPT | Partial second-provider check |
|---|---|---|---|---|
| S1 §1 | 9–53 | model, conventions, statements of A–C | G1, C1 (xAI, Anthropic) | — |
| A1 §2 | 54–68 | positive spectrum, jet rank, smooth version | C1 | G3 "qualitative positivity OK" |
| A2 §3 | 69–114 | contact frame, `12 r^{-(d+3)}`, uniform gap, (3.5) | C1 (including the gap proof) | G3 certified `det T_r` |
| A4 §4 | 115–137 | (4.1)–(4.3), independent residual | C1 | — |
| A3 §5 | 138–183 | (5.1)–(5.5), type convergence, UI, floor | C1 (O1 agrees but is author-provider) | G3 UI-majorant structure and `A_M -> A_0` |
| A3-E1 | erratum | `D_r = diag(r^{-1/2}, I)` | C1, G3 | — |
| A5 (6.1) | 184–197 | canceled-pivot bound, all m | C1 | G3 (m=1) |
| A6 (6.2) | 198–211 | two soft factors, including the 3/2 saddle term | C1, G4 | — |
| A7 §7 | 212–275 | `r^5` numerators, corank strata, m=1 far branch, M4, (7.8) | C1 | G4 numerator structure |
| CAP §§1–5 | cap 1–168 | deterministic separating cap, global maximin | C1, G3 | — |
| D1-A §8 | 276–297 | pinned genericity, Borel elder event, `G_r =>` partner S | G1, C1 | — |
| D1-B §9 | 298–309 | Borel marked Kac–Rice as written | G1 | — |
| D1-B-E2 | E2 | §9 replacement v1.1, C1–C6 | C1, G2 | — |
| D1-C §10 | 310–336 | radial ledger `r A_r dr db dk dσ` | G1 | C1 (#123 E9 check) |
| §11 | 337–364 | pushforward, (11.3), difference (1.2) | C2 | G4 ((11.1) and the (11.3) identity) |
| §12 | 365–385 | cumulative and moment corollaries | C2 | — |
| D1-D §§13–14 | 386–446 | target-growth majorant, `k -> 0`, off-diagonal (14.1) | G1 | C2 spot checks |
| D1-E §15 | 447–481 | parity, disintegration, (15.2) | G1 | C2 gamma factor |

Every interface has a full-depth ACCEPT from at least one nonauthor provider. Six have full-depth ACCEPTs from both
nonauthor providers: S1, A3-E1, A6, CAP, D1-A and D1-B-E2. Nine more have a partial check by the other provider. Three
have a single provider at any depth: A4, §12, and D1-B. D1-B is the original §9 text, which the reading rule replaces by
D1-B-E2 (two providers), so it needs no second check. The project's established reconciliations (D2, D3, D4, D6 in STATUS)
each rest on one provider-distinct nonauthor review, so this meets the existing bar. §8 invites second-provider checks,
prioritized by that gap, but they are not a precondition.

## 5. Residual items and their discharge

Every item that an earlier record left open, AMEND or "not closed" is listed below with where it is discharged.

| Open item | Raised in | Discharge |
|---|---|---|
| A3: wrong congruence factor `diag(√r,I)` | main#63 5841947606; G3 `A3_TYPE_CONVERGENCE` C1 | E1, landed by Math-#64 at `d8f5505`. Algebra confirmed by G3 and C1. |
| A3: UI of `W_r/r^2` | G3 C4 | C1 A3 item 4: `sup E(W_r/r^2)^2 < ∞` from (5.3), (5.1), `‖A_S‖ <= ‖A_M‖ + rM3` and A4. G3 `A3_UI_MAJORANT` gives the same majorant. |
| A3: type-indicator convergence under `D_r` | G3 | C1 W1: Slutsky joint convergence in law. The discontinuity set lies in `{det A_0 = 0}`, which is null (C1 item 5). |
| A3: `z_* > 0` | G3 C3 | C1: `A_0` is a nondegenerate Gaussian on `Sym_m`, so it charges the open cone `{A < 0}`. Continuity plus compactness gives (5.5). |
| A3: `A_M -> A_0` uniformly | G3 `A_M_TO_A0` (conditional on A1–A2) | C1 A3 item 3 and A1–A2 below. |
| A2: "single missing estimate", the uniform Gramian gap | G3 `A2_RESIDUAL_GAP` | C1 A2. Each coordinate of `U_r ⊕ vec A_r` is `∫ ∂^α f(tu) dμ_r(t)` with `μ_r -> δ_0`, so the covariance is jointly continuous on `[0,r_0] × O(d)`. It is positive definite at `r=0` by A1. Compactness gives `λ_min >= c_* > 0`. This is existential, which is all Theorem A claims. |
| A5 for `m >= 2` | G3 table | C1 A5: `det H_M = rα det(B' )` with `B' = A - (r/α)ββ^T`, and `0 < -B' <= B` on typed support. Symbolic for m=1,2,3; probed on 1,962 exact instances. |
| A7 `r^5` integrals; far branch | G3 table; C1 v1 | C1 A7, and G4. The v2 correction is that `λ >= 1/(Dr)` lies in the **majorant** event, not necessarily in depth failure. The far branch is kept in the upper bound. |
| Cap pairing: embedded chart and Morse/§8 | G3 `EMBEDDED_CHART_AND_MORSE`, `CAP_PAIRING_IDENTITIES` | §8 by G1 and C1. The embedding radius is made explicit below. |
| Embedding radius not displayed | G3 `EMBEDDED_CHART_AND_MORSE` ("AMEND the slogan") | Below: `r < L/(4√2)`. |
| §9 write-up gap (monotone-class passage) | Math-PR112 review, recorded in E2 | E2 v1.1, reviewed by C1 and G2. C1 withdrew the theorem-number objection on primary-source grounds (arXiv:2304.07424v3, Theorems 2.1, 2.2, 7.1, Remarks 7–8). |
| §§11–12 never reviewed | This reconciliation (line audit) | §6 below. |
| "Second non-Claude review of W1 and A2 before integration" | C1 request | W1 is wording only and is confirmed by O1 (non-discharge) and G3's in-law route. For A2 see the provider note in §4. The request stands as §8's open invitation. It is not a blocker under the existing bar. |

**W1 (wording; proof unchanged).** P l. 165 says the scaled endpoint Hessians converge "in probability" to
`diag(∓6k, A_0)`. Correctly:

- `D_r H_M D_r - diag(-6k, A_M)` and `D_r H_S D_r - diag(6k, A_M)` tend to 0 in probability, by (5.1), (5.2) and
  `‖A_S - A_M‖ <= rM3`;
- `A_M -> A_0` in law.

Slutsky then gives joint convergence in law, and continuous mapping applies off the null set `{det A_0 = 0}`. P's
subsequence paragraph (l. 176) already argues in law.

**Embedding radius.** Let `D = [-2r, 2r] × B̄(0, 2r)`, the cap cylinder, which contains all of `G_r`'s geometry. Then
`D ⊂ B̄(0, 2√2 r)`. On `R^d/(LZ^d)` the exponential chart at any point is injective on the open ball of radius
`L/2`. So `D` is embedded whenever `2√2 r < L/2`, that is `r < L/(4√2)`. Theorem A's existential `r_*` may be, and is,
taken below this. This makes explicit what P §1 says with "for sufficiently small r".

## 6. Review of the previously unreviewed §§11–12 (C2)

The line audit found that no earlier record reviewed P lines 337–384. G1 lists 215–217, 276–296, 298–308,
88–94 and 310–335, 386–445, and 447–480. C1 assumed that D1-A–E covered "§§10–15", which it did not. Both sections
were reviewed here from the bytes.

**§11 — ACCEPT.**

- (11.1): `r = (ℓ/k)^{1/3}` gives `r dr/dℓ = (1/3) k^{-2/3} ℓ^{-1/3}`.
- Restriction: for `ℓ < k_- r_*^3`, every `k ∈ K` has `r < r_*`, so the whole pushforward stays inside Theorem A's
  range.
- (11.2): at fixed `(b,k,u)`, pushing `r A_r dr` forward to `ℓ` gives `ℓ^{-1/3} A_{(ℓ/k)^{1/3}}/(3k^{2/3}) dℓ`.
  Integrating over `B × K × S^{d-1}` gives the density version; `ν_eld` carries the extra factor `p_r`.
- (11.3): as `ℓ -> 0`, `A_r -> A_0` uniformly with a uniform bound (10.3), and `p_r -> 1` uniformly by Theorem A.
  Dominated convergence on the compact domain gives `ℓ^{1/3} ν -> ∫ A_0/(3k^{2/3}) = 4∫ k^{-2/3} π_0 z_0 = c_{B,K}`,
  using `A_0 = 12 π_0 z_0`. Positivity follows from `π_0, z_0 > 0` on a set of positive measure. G4 verified the same
  identity in (13.6)'s form.
- Difference: `ν_cand - ν_eld = ℓ^{-1/3}∫ A_r(1-p_r)/(3k^{2/3})`. Theorem A gives `<= ℓ^{-1/3} ∫ A^* C r^3/(3k^{2/3})`
  with `r^3 = ℓ/k`, so the bound is `C ℓ^{2/3} ∫ A^*/(3k^{5/3})`. This is finite because `k >= k_- > 0`, which proves
  (1.2). The exponent `ℓ^{2/3} k^{-5/3}` is checked exactly.

**§12 — ACCEPT.**

- (12.1): `∫_0^t c ℓ^{-1/3}(1+o(1)) dℓ = (3/2)c t^{2/3}(1+o(1))`, and `∫_0^t C ℓ^{2/3} = (3/5)C t^{5/3}`.
- (12.2): for `q > -2/3` the integrand `ℓ^{q-1/3}` is integrable at 0, giving `c t^{q+2/3}/(q+2/3)`.
- (12.3): for `q > -5/3`, `∫_0^t ℓ^q C ℓ^{2/3} = C t^{q+5/3}/(q+5/3)`.
- For `q <= -2/3`, each individual expectation diverges because `c > 0`. (12.3) is stated for the nonselected measure
  directly, not as a difference of infinities.
- The unrestricted leading forms follow from Theorem C in the same way. The compact difference rates are correctly
  **not** extended.

**Spot re-derivations for §§13–15.** These do not replace G1; they are a second provider's check.

- (13.1): `(b - kr^3/2)^2 + 144k^2 >= (b^2+k^2)/2` for `r <= 1`. Checked exactly at sample rationals and by the
  inequality `b^2 <= 2(b - kr^3/2)^2 + k^2/2`.
- (13.4): `W_r/r^2 <= C(1+‖f‖_{C^3})^{2d}` follows from (5.1) and (5.3) with no division by `k`.
- (15.2): with `t = 12k` and `w = s^2/2`, `144∫_0^∞ k^{4/3}φ_τ(12k)dk = Γ(7/6)τ^{4/3}/(24^{1/3}√π)`. Checked
  numerically to relative `1e-9` in `reconciliation_check.py`, and exactly for the power bookkeeping.

## 7. Consumption contract

| Consumer | Consumes from D1 | After this reconciliation |
|---|---|---|
| D2 Theorem R, `frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md` (blob `247b3ecf…`) | The §8 comparison `1-p_r <= Q^W(G_r^c)`, §9 (read as E2), §10, §14 (14.1), and (15.2) as `c`. **Not** (1.1). | Every consumed interface is reconciled. `claims` moves FAIL_CLOSED -> REVIEWED_SCOPED at Theorem R's stated `O(1)` scope. |
| D3 SIDE24 coefficient, `coefficients/side24_v1/PROOF.md` (blob `44b66f04…`) | (15.2) as the leading coefficient of Theorem C | The persistence interpretation is reconciled. `claims` moves HOLD_WITH_DOMAIN -> REVIEWED_SCOPED. The arithmetic acceptance (#65) is unchanged. |
| Direct lower bound, `ELDER_LOWER_AND_DENSITY_GAP.md` (E4)/(E11) | Theorem A (1.1) upper half, §§9–11 | The two-sided planar composition is now fully sourced. (E2) itself stays as reviewed in #123. |
| D4 fixed-remote (RM), RC (#110), IW (#107), #120 | P's model and observation **conventions** only, plus RM's own "Theorem A" | Unaffected. No D1 theorem is consumed. |
| Any new consumer | — | Cite the §1 identity set, this record, and a scope no larger than §2. |

**Still not consumable.**

- Numerical `C`, `r_*`, `z_*`, `c_{B,K}` or `c_{d,L}`.
- (1.2)'s difference rate outside compact marks.
- Any lower bound for `d >= 3`.
- A globally uniform `O(r^3)` constant over unbounded marks.
- RN, 24-jet, JETMOD or H3/LPW numerical certificates.
- `lemma_closed`, any prize, or "unconditional acceptance of the lifetime parent", which is a forbidden global
  inference in `claims/LANDING_CLAIMS.json`. This acceptance is scoped.

## 8. Independence, lineage and requests

**Dual role.** The reconciler is also reviewer C1 of A1–A7, CAP and E2, and reviewer C2 of §§11–12. This is disclosed
rather than hidden.

**What the reconciliation relies on, and what it does not.** Nothing here is accepted on the strength of a green test,
a hash, a same-provider review or O1. Every interface rests on a written nonauthor derivation.

**Merge.** The register PR carrying this record must be merged by a non-Claude lane after reading it. Claude does not
self-merge.

**Invitation.** A second-provider check is invited in this priority order:

1. A4 and §12, which have one provider at any depth;
2. the full-depth gaps behind partial checks: A2's uniform gap, A5 for `m >= 2`, A7's division by the floor, and
   §11's difference bound;
3. the reverse direction for D1-C, D1-D and D1-E, where xAI is full-depth and Claude is partial.

Any lane may return COUNTEREXAMPLE with an exact equation. That would reopen only the affected interface and its
dependents, and the graph's reverse-impact rule would mark them REVALIDATION_REQUIRED.

## 9. Register transitions justified by this record

Math-, in the same PR:

- `PROOF_INDEX.md`: D1 moves to "Reviewed scoped results" with the §2 scope. The §9 repair is recorded as reviewed by
  C1 and G2. The D2 entry is updated to name its reconciled imports.
- `claims/LANDING_CLAIMS.json`:
  - `lifetime-remainder` FAIL_CLOSED -> REVIEWED_SCOPED;
  - `side24-coefficient` HOLD_WITH_DOMAIN -> REVIEWED_SCOPED;
  - dependency reviews are bound to local review files.
- `frontiers/downstream_gate_20260925/GRAPH.json`:
  - `math.uniform-matrix-cap-lifetime` AUTHOR_SIDE_CANDIDATE -> PROVED_REVIEWED, with this record as evidence and
    the §2 scope. It is the only classification that satisfies a still-required premise, and D2 and D3 require it.
  - D2 and D3 gain `review_disposition: ACCEPT` metadata under the existing convention (commit `cc3a991`).
  - Four reading-rule component nodes (E1, E2, CAP and this record's directory) are added as `PROVED_REVIEWED`
    source nodes with required edges from D1. Each carries the ledger reviews its role needs: C1 and G3 for E1 and
    CAP, C1 and G2 for E2. This record is an Anthropic, nonauthor record; its W1 and radius items were raised in C1
    and G3 and its §6 is C2. OpenAI's technical check of this record (review 5345687857) is listed as author-provider
    non-discharge. This closes the gap that review found: before it, E1, E2 and W1 were named only in metadata, so
    editing them left D1's source snapshot unchanged.
  - No node becomes `controlling`. Satisfying D2's and D3's required edge to D1 is not promotion: their own
    classifications stay author-side, so `promotion_allowed` still refuses both.
  - The fail-closed tests are kept on a pre-reconciliation fixture, so they still prove that an unreviewed parent
    blocks both dependents. New tests make real two-commit edits and deletions of each component with P unchanged,
    and require D1, D2 and D3 to be proposed `REVALIDATION_REQUIRED`. An unrelated edit must propose nothing, and
    the pre-binding graph is shown to miss the component edit.

main, in a companion PR pinned to this record's merged commit:

- `STATUS.md`: D1 moves from AMEND/open to ACCEPT — scoped.
- The glossary's D1 row is updated.
- The museum and shop projections are regenerated.

## 10. Checks

`reconciliation_check.py` (Python standard library):

1. Binds every §1 source and every local review record by Git blob, bytes and SHA256.
2. Cross-checks the hard-gate graph against `LEDGER.json`:
   - D1 is bound to P's exact bytes;
   - E1, E2, CAP and this directory are D1 source nodes behind required edges;
   - each component node carries exactly the ledger reviews of its interfaces;
   - the record's author provider differs from P's;
   - two negative controls (a dropped edge; an author-only review basis) are rejected.
3. Validates `LEDGER.json`:
   - every interface has at least one nonauthor ACCEPT;
   - author-provider reviews are non-discharge;
   - P's §§1–16 line ranges are fully covered;
   - every residual item has a discharge pointer.
4. Re-verifies the reading-rule mathematics exactly:
   - E1 congruence versus the displayed factor;
   - (5.3), (6.1) and (6.2) on exact rational instances;
   - (7.3) and the (7.5) split;
   - the cap constants and κ-scaling;
   - the embedding radius, the radial and lifetime ledgers, (11.1)–(11.3), the difference exponent and the §12
     integrals;
   - the (13.1) coercivity bound;
   - the (15.2) gamma factor, numerically.
5. Rejects nine semantic mutants.

A green run is evidence that the bookkeeping and finite identities hold. It is not the analytic review, which lives in
C1, G1, G2, G4 and §6.
